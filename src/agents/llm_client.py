"""
src/agents/llm_client.py
========================
Wrapper LLM portabile con supporto Gemini API, OpenAI-compatible (Ollama/vLLM)
e un MockLLMClient per testing senza costi API.

Features
--------
- Backend selezionato da config.yaml (llm.backend: "api" | "local")
- Retry automatico con exponential backoff su errori transitori
- Estrazione JSON robusta dall'output testuale (strip fences, fallback)
- TokenBudget globale con warn e hard-limit
- MockLLMClient deterministico per test e dry-run locali
- DiskCache per caching persistente delle risposte (utile per Kaggle 12h limit)

Utilizzo
--------
    from src.agents.llm_client import LLMClient, MockLLMClient, LLMResponse
    client = LLMClient.from_config(cfg)
    response = client.chat([{"role": "user", "content": "..."}])
    print(response.content, response.total_tokens)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Risposta normalizzata da qualsiasi backend LLM."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_s: float = 0.0
    raw: Any = field(default=None, repr=False)
    is_fallback: bool = False      # True se si e' usato il fallback JSON

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# DiskCache (Persistenza per run lunghe su Kaggle)
# ---------------------------------------------------------------------------

class LLMDiskCache:
    """
    Cache su disco thread-safe per le risposte LLM.
    Evita di rifare chiamate API costose se la sessione (es. Kaggle) viene
    interrotta a meta' di uno step lungo.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, cache_dir: str = "results/checkpoints"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLMDiskCache, cls).__new__(cls)
                cls._instance._init(cache_dir)
            return cls._instance

    def _init(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "llm_cache.jsonl"
        self.cache = {}
        
        # Load existing cache
        if self.cache_file.exists():
            logger.info("[LLMDiskCache] Caricamento cache da %s...", self.cache_file)
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        self.cache[data["key"]] = data["response"]
                logger.info("[LLMDiskCache] Caricate %d risposte.", len(self.cache))
            except Exception as e:
                logger.error("[LLMDiskCache] Errore caricamento cache: %s", e)
        
        self._file_handle = open(self.cache_file, "a", encoding="utf-8")
        self._write_lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        return self.cache.get(key)

    def set(self, key: str, response: dict) -> None:
        if key in self.cache:
            return
        self.cache[key] = response
        with self._write_lock:
            try:
                line = json.dumps({"key": key, "response": response}, ensure_ascii=False)
                self._file_handle.write(line + "\n")
                self._file_handle.flush()
            except Exception as e:
                logger.error("[LLMDiskCache] Errore scrittura cache: %s", e)


# ---------------------------------------------------------------------------
# TokenBudget (singleton globale)
# ---------------------------------------------------------------------------

class TokenBudget:
    """
    Tracker cumulativo dei token consumati durante la simulazione.
    """
    _total_input: int = 0
    _total_output: int = 0
    _warn_at: int = 50_000
    _hard_limit: int = 200_000
    _lock = threading.Lock()

    @classmethod
    def configure(cls, warn_at: int, hard_limit: int) -> None:
        with cls._lock:
            cls._warn_at = warn_at
            cls._hard_limit = hard_limit

    @classmethod
    def record(cls, input_tokens: int, output_tokens: int) -> None:
        with cls._lock:
            cls._total_input += input_tokens
            cls._total_output += output_tokens
            total = cls._total_input + cls._total_output
            hit_hard = total >= cls._hard_limit
            hit_warn = total >= cls._warn_at
        if hit_hard:
            raise RuntimeError(
                f"[TokenBudget] Hard limit raggiunto: {total} token "
                f"(limite={cls._hard_limit}). Simulazione interrotta."
            )
        if hit_warn:
            logger.warning(
                "[TokenBudget] ⚠  %d token totali consumati (warn_at=%d).",
                total, cls._warn_at,
            )

    @classmethod
    def summary(cls) -> dict[str, int]:
        return {
            "total_input": cls._total_input,
            "total_output": cls._total_output,
            "grand_total": cls._total_input + cls._total_output,
        }

    @classmethod
    def reset(cls) -> None:
        cls._total_input = 0
        cls._total_output = 0


# ---------------------------------------------------------------------------
# JSON extraction utility
# ---------------------------------------------------------------------------

FALLBACK_AGENT_OUTPUT = {
    "reasoning": "Fallback: LLM output non parsabile.",
    "opinion": "",
    "susceptibility": 0.5,
    "proposed_state": "S",
    "spread_intent": False,
}

def extract_json(text: str, fallback: dict | None = None) -> tuple[dict, bool]:
    fb = fallback if fallback is not None else FALLBACK_AGENT_OUTPUT.copy()

    try:
        return json.loads(text.strip()), False
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned), False
    except json.JSONDecodeError:
        pass

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        try:
            return json.loads(cleaned[start_idx:end_idx+1]), False
        except json.JSONDecodeError:
            pass

    logger.warning(
        "[LLMClient] JSON non parsabile — uso fallback. Estratto risposta: %r",
        text[:200],
    )
    return fb, True


# ---------------------------------------------------------------------------
# Backend: Gemini
# ---------------------------------------------------------------------------

class _GeminiBackend:
    def __init__(self, cfg: dict) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise ImportError("pip install google-generativeai") from e

        api_key = os.environ.get(cfg.get("api_key_env", "GEMINI_API_KEY"))
        if not api_key:
            raise EnvironmentError(
                f"Variabile d'ambiente '{cfg.get('api_key_env')}' non impostata."
            )
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = cfg.get("model", "gemini-2.0-flash")
        self._temperature = cfg.get("temperature", 0.7)
        self._max_tokens = cfg.get("max_tokens", 512)

    def chat(self, messages: list[dict]) -> LLMResponse:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        history_msgs = [m for m in messages if m["role"] != "system"]

        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction="\n\n".join(system_parts) or None,
            generation_config=self._genai.GenerationConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            ),
        )

        gemini_history = []
        for m in history_msgs[:-1]:
            role = "model" if m["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [m["content"]]})

        session = model.start_chat(history=gemini_history)
        last = history_msgs[-1]["content"] if history_msgs else ""

        t0 = time.perf_counter()
        resp = session.send_message(last)
        latency = time.perf_counter() - t0

        usage = resp.usage_metadata
        return LLMResponse(
            content=resp.text,
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
            model=self._model_name,
            latency_s=latency,
            raw=resp,
        )


# ---------------------------------------------------------------------------
# Backend: OpenAI-compatible (OpenAI / Ollama / vLLM)
# ---------------------------------------------------------------------------

class _OpenAICompatibleBackend:
    def __init__(self, cfg: dict, backend_type: str = "openai") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("pip install openai") from e

        if backend_type == "openai":
            api_key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
            base_url = None
        else:
            api_key = "ollama"
            base_url = cfg.get("base_url", "http://localhost:11434/v1")

        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = cfg.get("model", "llama3")
        self._temperature = cfg.get("temperature", 0.7)
        self._max_tokens = cfg.get("max_tokens", 512)

    def chat(self, messages: list[dict]) -> LLMResponse:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=None, 
        )
        latency = time.perf_counter() - t0
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            content=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=resp.model,
            latency_s=latency,
            raw=resp,
        )


# ---------------------------------------------------------------------------
# LLMClient (pubblico)
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Client LLM portabile con retry, token budget e disk cache.
    """

    def __init__(self, llm_config: dict, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        self._cache = LLMDiskCache()
        backend_key = llm_config.get("backend", "api")

        if backend_key == "api":
            api_cfg = llm_config.get("api", {})
            provider = api_cfg.get("provider", "gemini")
            if provider == "gemini":
                self._backend: _GeminiBackend | _OpenAICompatibleBackend = _GeminiBackend(api_cfg)
            else:
                self._backend = _OpenAICompatibleBackend(api_cfg, "openai")
            logger.info("[LLMClient] API backend: %s", provider)
        elif backend_key == "local":
            local_cfg = llm_config.get("local", {})
            self._backend = _OpenAICompatibleBackend(local_cfg, "local")
            logger.info("[LLMClient] Local backend: %s", local_cfg.get("model"))
        else:
            raise ValueError(f"Backend LLM non valido: '{backend_key}'")

        budget = llm_config.get("token_budget", {})
        if budget:
            TokenBudget.configure(
                warn_at=budget.get("warn_at", 50_000),
                hard_limit=budget.get("hard_limit", 200_000),
            )

    @classmethod
    def from_config(cls, cfg: "Config") -> "LLMClient":
        import dataclasses
        llm_cfg = dataclasses.asdict(cfg.llm)
        return cls(llm_cfg)

    def chat(self, messages: list[dict]) -> LLMResponse:
        msg_key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        msg_hash = hashlib.md5(msg_key.encode("utf-8")).hexdigest()

        cached_data = self._cache.get(msg_hash)
        if cached_data:
            return LLMResponse(
                content=cached_data["content"],
                input_tokens=cached_data.get("input_tokens", 0),
                output_tokens=cached_data.get("output_tokens", 0),
                model=cached_data.get("model", "cached"),
                latency_s=0.0,
                is_fallback=cached_data.get("is_fallback", False)
            )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._backend.chat(messages)
                TokenBudget.record(response.input_tokens, response.output_tokens)
                
                self._cache.set(msg_hash, {
                    "content": response.content,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "model": response.model,
                    "is_fallback": response.is_fallback
                })

                logger.debug(
                    "[LLMClient] tokens in=%d out=%d latency=%.2fs",
                    response.input_tokens, response.output_tokens, response.latency_s,
                )
                return response
            except RuntimeError:
                raise  # Hard limit — propaga subito
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "[LLMClient] Tentativo %d/%d fallito: %s. Retry in %ds.",
                    attempt + 1, self._max_retries, exc, wait,
                )
                time.sleep(wait)

        logger.error("[LLMClient] Tutti i retry esauriti. Uso fallback.")
        fallback_resp = LLMResponse(
            content=json.dumps(FALLBACK_AGENT_OUTPUT),
            is_fallback=True,
        )
        self._cache.set(msg_hash, {
            "content": fallback_resp.content,
            "input_tokens": 0,
            "output_tokens": 0,
            "model": "fallback",
            "is_fallback": True
        })
        return fallback_resp

    @staticmethod
    def token_summary() -> dict[str, int]:
        return TokenBudget.summary()


# ---------------------------------------------------------------------------
# MockLLMClient — per test e dry-run senza API
# ---------------------------------------------------------------------------

class MockLLMClient:
    """
    Client LLM deterministico per test unitari e dry-run locali.
    """
    def __init__(self, seed: int = 42, infection_rate: float = 0.3) -> None:
        self._seed = seed
        self._infection_rate = infection_rate
        self._call_count = 0
        self._lock = threading.Lock()
        self._cache = LLMDiskCache()

    def chat(self, messages: list[dict]) -> LLMResponse:
        import hashlib
        import random

        msg_key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        msg_hash = hashlib.md5(msg_key.encode("utf-8")).hexdigest()

        cached_data = self._cache.get(msg_hash)
        if cached_data:
            return LLMResponse(
                content=cached_data["content"],
                input_tokens=cached_data.get("input_tokens", 0),
                output_tokens=cached_data.get("output_tokens", 0),
                model=cached_data.get("model", "mock"),
                latency_s=0.0,
                is_fallback=cached_data.get("is_fallback", False)
            )

        with self._lock:
            self._call_count += 1
            n = self._call_count

        msg_hash_int = int(msg_hash, 16)
        rng = random.Random(self._seed + msg_hash_int)

        proposed = "I" if rng.random() < self._infection_rate else "S"
        susceptibility = round(rng.uniform(0.1, 0.9), 2)
        spread = proposed == "I"

        convincing = "convincente" if spread else "non convincente"
        opinion_text = "La narrativa e' reale e va diffusa." if spread else "Resto scettico."
        payload = {
            "reasoning": f"Mock reasoning #{n}: Ho analizzato il feed e concludo che il tema e' {convincing}.",
            "opinion": f"Opinion #{n}: {opinion_text}",
            "susceptibility": susceptibility,
            "proposed_state": proposed,
            "spread_intent": spread,
        }

        resp = LLMResponse(
            content=json.dumps(payload),
            input_tokens=50,
            output_tokens=80,
            model="mock",
            latency_s=0.001,
        )

        self._cache.set(msg_hash, {
            "content": resp.content,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "model": resp.model,
            "is_fallback": resp.is_fallback
        })

        return resp

    @property
    def call_count(self) -> int:
        return self._call_count
lf._call_count