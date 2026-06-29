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
import time
from dataclasses import dataclass, field
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
# TokenBudget (singleton globale)
# ---------------------------------------------------------------------------

class TokenBudget:
    """Tracker cumulativo dei token consumati durante la simulazione."""

    _total_input: int = 0
    _total_output: int = 0
    _warn_at: int = 50_000
    _hard_limit: int = 200_000

    @classmethod
    def configure(cls, warn_at: int, hard_limit: int) -> None:
        cls._warn_at = warn_at
        cls._hard_limit = hard_limit

    @classmethod
    def record(cls, input_tokens: int, output_tokens: int) -> None:
        cls._total_input += input_tokens
        cls._total_output += output_tokens
        total = cls._total_input + cls._total_output
        if total >= cls._hard_limit:
            raise RuntimeError(
                f"[TokenBudget] Hard limit raggiunto: {total} token "
                f"(limite={cls._hard_limit}). Simulazione interrotta."
            )
        if total >= cls._warn_at:
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
        """Reset per test o nuove run."""
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
    """
    Estrae un oggetto JSON dall'output testuale dell'LLM.
    Strategie (in ordine):
      1. Parse diretto.
      2. Strip di markdown code fences.
      3. Regex: primo blocco { ... } valido.
      4. Fallback dict.

    Returns
    -------
    (parsed_dict, is_fallback)
    """
    fb = fallback if fallback is not None else FALLBACK_AGENT_OUTPUT.copy()

    # 1. Parse diretto
    try:
        return json.loads(text.strip()), False
    except json.JSONDecodeError:
        pass

    # 2. Strip fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned), False
    except json.JSONDecodeError:
        pass

    # 3. Primo blocco JSON completo
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), False
        except json.JSONDecodeError:
            pass

    logger.warning("[LLMClient] JSON non parsabile — uso fallback.")
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
        import google.generativeai as genai  # type: ignore

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
    Client LLM portabile con retry e token budget.

    Parameters
    ----------
    llm_config : dict
        Sezione 'llm' del config.yaml (gia' parsata).
    max_retries : int
        Numero di tentativi su errori transitori (default 3).
    """

    def __init__(self, llm_config: dict, max_retries: int = 3) -> None:
        self._max_retries = max_retries
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
        """Factory method: costruisce il client dalla Config dataclass."""
        import dataclasses
        llm_cfg = dataclasses.asdict(cfg.llm)
        return cls(llm_cfg)

    def chat(self, messages: list[dict]) -> LLMResponse:
        """Invia messaggi con retry e registra token nel budget globale."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._backend.chat(messages)
                TokenBudget.record(response.input_tokens, response.output_tokens)
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
        return LLMResponse(
            content=json.dumps(FALLBACK_AGENT_OUTPUT),
            is_fallback=True,
        )

    @staticmethod
    def token_summary() -> dict[str, int]:
        return TokenBudget.summary()


# ---------------------------------------------------------------------------
# MockLLMClient — per test e dry-run senza API
# ---------------------------------------------------------------------------

class MockLLMClient:
    """
    Client LLM deterministico per test unitari e dry-run locali.
    Genera risposte JSON valide basate su hash del messaggio + seed.

    Parameters
    ----------
    seed : int
        Seed per la generazione deterministica delle risposte.
    infection_rate : float
        Probabilita' che il mock risponda con stato "I" (default 0.3).
    """

    def __init__(self, seed: int = 42, infection_rate: float = 0.3) -> None:
        self._seed = seed
        self._infection_rate = infection_rate
        self._call_count = 0

    def chat(self, messages: list[dict]) -> LLMResponse:
        import random
        self._call_count += 1
        rng = random.Random(self._seed + self._call_count)

        # Determina stato proposto in modo deterministico
        proposed = "I" if rng.random() < self._infection_rate else "S"
        susceptibility = round(rng.uniform(0.1, 0.9), 2)
        spread = proposed == "I"

        convincing = "convincente" if spread else "non convincente"
        opinion_text = "La narrativa e' reale e va diffusa." if spread else "Resto scettico."
        payload = {
            "reasoning": f"Mock reasoning #{self._call_count}: "
                         f"Ho analizzato il feed e concludo che il tema e' {convincing}.",
            "opinion": f"Opinion #{self._call_count}: {opinion_text}",
            "susceptibility": susceptibility,
            "proposed_state": proposed,
            "spread_intent": spread,
        }

        return LLMResponse(
            content=json.dumps(payload),
            input_tokens=50,
            output_tokens=80,
            model="mock",
            latency_s=0.001,
        )

    @property
    def call_count(self) -> int:
        return self._call_count
