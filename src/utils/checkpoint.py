"""
src/utils/checkpoint.py
=======================
Salvataggio e caricamento dello stato completo della simulazione.

Un checkpoint contiene TUTTO il necessario per riprendere da un
qualsiasi step t:
  - NetworkManager (grafo + stati agenti + post + embeddings)
  - Pesi GNN
  - Step corrente
  - Metriche cumulative
  - Token budget consumato

Il checkpoint e' un singolo file .pkl (pickle) con struttura
documentata in `CheckpointData`.

Utilizzo
--------
    from src.utils.checkpoint import CheckpointManager
    cm = CheckpointManager(cfg)
    cm.save(step=5, network_manager=nm, gnn_weights=model.get_weights())
    state = cm.load_latest()
    state = cm.load(step=5)
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.graph.network_manager import NetworkManager
    from src.utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CheckpointData (struttura serializzata)
# ---------------------------------------------------------------------------

@dataclass
class CheckpointData:
    """Payload serializzato in ogni checkpoint."""
    step: int
    timestamp: float
    config_hash: str
    # Grafo + stati agenti (dal NetworkManager)
    graph_payload: dict[str, Any]
    # Pesi GNN (numpy arrays)
    gnn_weights: dict[str, Any]
    # Metriche cumulative per il resume
    cumulative_metrics: list[dict]
    # Token budget consumed so far
    token_budget: dict[str, int] = field(default_factory=dict)
    # Stato del seeder (nodi paziente zero)
    patient_zero_ids: list[int] = field(default_factory=list)
    # Metadata varie
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Gestisce il salvataggio e il caricamento dei checkpoint di simulazione.

    I checkpoint sono salvati in:
        {cfg.paths.checkpoints}/ckpt_step_{step:04d}.pkl

    Parameters
    ----------
    cfg : Config
    keep_last : int
        Numero di checkpoint da mantenere (gli altri vengono eliminati).
    """

    FILENAME_PATTERN = "ckpt_step_{step:04d}.pkl"

    def __init__(self, cfg: "Config", keep_last: int = 3) -> None:
        self._cfg = cfg
        self._keep_last = keep_last
        self._checkpoint_dir = cfg.project_root / cfg.paths.checkpoints
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._cumulative_metrics: list[dict] = []

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        network_manager: "NetworkManager",
        gnn_weights: dict | None = None,
        patient_zero_ids: list[int] | None = None,
        meta: dict | None = None,
    ) -> Path:
        """
        Salva lo stato completo al passo `step`.

        Parameters
        ----------
        step : int
        network_manager : NetworkManager
        gnn_weights : dict | None
            Pesi del modello GNN (numpy arrays).
        patient_zero_ids : list[int] | None
        meta : dict | None  Extra info (run_id, ecc.)

        Returns
        -------
        Path del file checkpoint creato.
        """
        from src.agents.llm_client import TokenBudget

        # Serializza il NetworkManager
        import networkx as nx
        graph_payload = {
            "graph": network_manager.G,
            "agent_states": network_manager.get_all_states(),
            "community_map": network_manager._community_map,
            "embeddings": network_manager._embeddings,
            "post_store": network_manager._post_store,
        }

        data = CheckpointData(
            step=step,
            timestamp=time.time(),
            config_hash=self._cfg.config_hash,
            graph_payload=graph_payload,
            gnn_weights=gnn_weights or {},
            cumulative_metrics=list(self._cumulative_metrics),
            token_budget=TokenBudget.summary(),
            patient_zero_ids=patient_zero_ids or [],
            meta=meta or {},
        )

        path = self._checkpoint_dir / self.FILENAME_PATTERN.format(step=step)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("[Checkpoint] Salvato: %s (step=%d)", path.name, step)

        # Pulisci vecchi checkpoint
        self._prune_old_checkpoints()
        return path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, step: int) -> CheckpointData:
        """Carica il checkpoint a step `step`."""
        path = self._checkpoint_dir / self.FILENAME_PATTERN.format(step=step)
        return self._load_from_path(path)

    def load_latest(self) -> CheckpointData:
        """Carica il checkpoint piu' recente (step piu' alto disponibile)."""
        checkpoints = sorted(self._checkpoint_dir.glob("ckpt_step_*.pkl"))
        if not checkpoints:
            raise FileNotFoundError(
                f"Nessun checkpoint trovato in: {self._checkpoint_dir}"
            )
        return self._load_from_path(checkpoints[-1])

    def restore_network_manager(
        self,
        data: CheckpointData,
    ) -> "NetworkManager":
        """
        Ricostruisce un NetworkManager dal payload del checkpoint.
        """
        from src.graph.network_manager import NetworkManager
        nm = NetworkManager(
            G=data.graph_payload["graph"],
            cfg=self._cfg,
            community_map=data.graph_payload.get("community_map"),
            node_features=data.graph_payload.get("embeddings"),
        )
        nm._agent_states = data.graph_payload.get("agent_states", {})
        nm._post_store = data.graph_payload.get("post_store", {})
        logger.info("[Checkpoint] NetworkManager ripristinato (step=%d).", data.step)
        return nm

    def list_checkpoints(self) -> list[tuple[int, Path]]:
        """Elenca tutti i checkpoint disponibili (step, path)."""
        result = []
        for p in sorted(self._checkpoint_dir.glob("ckpt_step_*.pkl")):
            try:
                step = int(p.stem.split("_")[-1])
                result.append((step, p))
            except ValueError:
                pass
        return result

    def has_checkpoint(self, step: int | None = None) -> bool:
        """True se esiste almeno un checkpoint (o specificamente per `step`)."""
        if step is not None:
            return (self._checkpoint_dir / self.FILENAME_PATTERN.format(step=step)).exists()
        return bool(list(self._checkpoint_dir.glob("ckpt_step_*.pkl")))

    # ------------------------------------------------------------------
    # Metrics accumulation
    # ------------------------------------------------------------------

    def record_metrics(self, step: int, metrics: dict) -> None:
        """Accumula metriche per il resume (incluse nel prossimo checkpoint)."""
        self._cumulative_metrics.append({"step": step, **metrics})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_path(self, path: Path) -> CheckpointData:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info("[Checkpoint] Caricato: %s (step=%d)", path.name, data.step)
        return data

    def _prune_old_checkpoints(self) -> None:
        """Mantieni solo gli ultimi `keep_last` checkpoint."""
        checkpoints = sorted(self._checkpoint_dir.glob("ckpt_step_*.pkl"))
        to_delete = checkpoints[: max(0, len(checkpoints) - self._keep_last)]
        for p in to_delete:
            try:
                p.unlink()
                logger.debug("[Checkpoint] Eliminato vecchio checkpoint: %s", p.name)
            except OSError:
                pass
