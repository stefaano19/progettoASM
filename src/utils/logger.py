"""
src/utils/logger.py
===================
Logger strutturato JSONL per la simulazione.

Ogni riga del file di log e' un oggetto JSON autonomo con:
  - timestamp ISO8601
  - step temporale
  - tipo di evento (metric, agent_decision, rewire, checkpoint, ...)
  - payload dati

Fornisce anche un configuratore del logging di sistema (rich handler).

Utilizzo
--------
    from src.utils.logger import setup_logging, SimLogger
    setup_logging("DEBUG")
    sim_log = SimLogger(log_path="results/logs/run_001.jsonl", run_id="run_001")
    sim_log.log_metrics(step=0, metrics={"Q": 0.45, "ECI": 0.61})
    sim_log.log_agent_decision(step=1, node_id=42, decision={...})
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Sistema logging (console) con rich
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """
    Configura il logging di sistema con output formattato su console.
    Usa rich se disponibile, altrimenti logging standard.

    Parameters
    ----------
    level : str
        "DEBUG" | "INFO" | "WARNING" | "ERROR"
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=numeric_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
            force=True,
        )
    except ImportError:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
            force=True,
        )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# SimLogger — JSONL structured logger
# ---------------------------------------------------------------------------

class SimLogger:
    """
    Logger strutturato per la simulazione.
    Scrive ogni evento come riga JSONL nel file di log.

    Parameters
    ----------
    log_path : str | Path
        Percorso del file .jsonl di output.
    run_id : str | None
        Identificatore univoco per questa run. Auto-generato se None.
    """

    def __init__(self, log_path: str | Path, run_id: str | None = None) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self._logger = logging.getLogger("SimLogger")
        self._file = open(self.log_path, "a", encoding="utf-8")
        self._logger.info("[SimLogger] Logging su: %s (run_id=%s)", self.log_path, self.run_id)

    # ------------------------------------------------------------------
    # Core write
    # ------------------------------------------------------------------

    def _write(self, event_type: str, step: int | None, payload: dict[str, Any]) -> None:
        record = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "event": event_type,
            **payload,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    # ------------------------------------------------------------------
    # Event methods
    # ------------------------------------------------------------------

    def log_run_start(self, config_hash: str, seed: int, extra: dict | None = None) -> None:
        """Primo record del file: metadati della run."""
        self._write("run_start", step=None, payload={
            "config_hash": config_hash,
            "random_seed": seed,
            **(extra or {}),
        })
        self._logger.info("[SimLogger] Run avviata — config_hash=%s", config_hash)

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Metriche di rete aggregate (Q, ECI, BP, IR, ...)."""
        self._write("metrics", step=step, payload={"metrics": metrics})
        self._logger.debug("[SimLogger] step=%d metrics=%s", step, metrics)

    def log_agent_decision(self, step: int, node_id: int, decision: dict) -> None:
        """Decisione di un singolo agente (belief update, azioni, token)."""
        self._write("agent_decision", step=step, payload={
            "node_id": node_id,
            "decision": decision,
        })

    def log_rewire(self, step: int, added: list[tuple], removed: list[tuple]) -> None:
        """Variazioni topologiche prodotte dal modulo GNN."""
        self._write("rewire", step=step, payload={
            "edges_added": added,
            "edges_removed": removed,
            "delta_edges": len(added) - len(removed),
        })
        self._logger.info(
            "[SimLogger] step=%d rewire: +%d -%d archi",
            step, len(added), len(removed),
        )

    def log_state_transition(
        self,
        step: int,
        transitions: dict[int, tuple[str, str]],
    ) -> None:
        """Transizioni di stato S->I, I->R, ecc."""
        self._write("state_transitions", step=step, payload={
            "transitions": {str(k): list(v) for k, v in transitions.items()},
            "count": len(transitions),
        })

    def log_celf_seeds(self, step: int, seeds: list[int], expected_spread: float) -> None:
        """Seed selezionati da CELF per Influence Maximization."""
        self._write("celf_seeds", step=step, payload={
            "seeds": seeds,
            "expected_spread": expected_spread,
        })
        self._logger.info(
            "[SimLogger] step=%d CELF seeds=%s spread=%.2f",
            step, seeds, expected_spread,
        )

    def log_checkpoint(self, step: int, checkpoint_path: str) -> None:
        """Salvataggio checkpoint del grafo."""
        self._write("checkpoint", step=step, payload={"path": checkpoint_path})
        self._logger.info("[SimLogger] Checkpoint salvato: %s", checkpoint_path)

    def log_token_usage(self, step: int, total_tokens: int, delta_tokens: int) -> None:
        """Consumo cumulativo di token LLM."""
        self._write("token_usage", step=step, payload={
            "total_tokens": total_tokens,
            "delta_tokens": delta_tokens,
        })

    def log_error(self, step: int | None, error: str, context: dict | None = None) -> None:
        """Errore non fatale durante la simulazione."""
        self._write("error", step=step, payload={
            "error": error,
            "context": context or {},
        })
        self._logger.error("[SimLogger] step=%s ERROR: %s", step, error)

    def close(self) -> None:
        """Chiude il file di log in modo pulito."""
        self._write("run_end", step=None, payload={})
        self._file.close()
        self._logger.info("[SimLogger] Log chiuso: %s", self.log_path)

    def __enter__(self) -> "SimLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
