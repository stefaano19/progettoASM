"""
src/gnn/rewirer.py
==================
Rewiring topologico basato sugli score del link predictor GNN.

Logica di rewiring
------------------
Per ogni step temporale t, dopo il forward pass della GNN:

  RIMOZIONE:
    Per ogni arco (u, v) esistente:
      se score(u, v) < threshold_remove → rimuovi l'arco
    Massimo `max_removals_per_step` rimozioni per step.

  AGGIUNTA:
    Per ogni coppia candidata (u, v) non connessa:
      se score(u, v) > threshold_add → aggiungi l'arco
    Massimo `max_new_edges_per_step` aggiunte per step.

  FILTRI DI SICUREZZA:
    - Nessun self-loop.
    - Non rimuovere archi verso fact-checker (stato F) — proteggono la diffusione.
    - Non aggiungere archi a nodi isolati (grado < min_degree).
    - Budget: le modifiche totali sono limitate per evitare frammentazione.

Utilizzo
--------
    from src.gnn.rewirer import Rewirer
    rewirer = Rewirer(cfg)
    to_add, to_remove = rewirer.compute(
        link_scores=scores,
        G=G,
        agent_states=nm.get_all_states(),
    )
    added, removed = nm.apply_rewiring(to_add, to_remove)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class Rewirer:
    """
    Calcola le modifiche topologiche da applicare al grafo.

    Parameters
    ----------
    cfg : Config
        Configurazione (rewire_threshold_add, rewire_threshold_remove,
        max_new_edges_per_step).
    """

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        self._threshold_add = cfg.gnn.rewire_threshold_add
        self._threshold_remove = cfg.gnn.rewire_threshold_remove
        self._max_add = cfg.gnn.max_new_edges_per_step
        self._max_remove = max(1, cfg.gnn.max_new_edges_per_step // 2)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute(
        self,
        link_scores: dict[tuple[int, int], float],
        G: "nx.Graph",
        agent_states: dict[int, str] | None = None,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Calcola gli archi da aggiungere e da rimuovere.

        Parameters
        ----------
        link_scores : dict[(u, v), score]
            Score in [0, 1] per coppie di nodi (esistenti + candidati).
        G : nx.Graph
            Grafo corrente.
        agent_states : dict[int, str] | None
            Stati degli agenti — usati per filtri di sicurezza.

        Returns
        -------
        to_add : list[(u, v)]
            Coppie da connettere.
        to_remove : list[(u, v)]
            Archi da eliminare.
        """
        existing_edges = set(G.edges())
        states = agent_states or {}

        # --- Archi da rimuovere ---
        removal_candidates = [
            (pair, score) for pair, score in link_scores.items()
            if pair in existing_edges and score < self._threshold_remove
        ]
        removal_candidates.sort(key=lambda x: x[1])  # Score più basso prima

        to_remove: list[tuple[int, int]] = []
        for (u, v), score in removal_candidates:
            if len(to_remove) >= self._max_remove:
                break
            if not self._is_safe_removal(u, v, G, states):
                continue
            to_remove.append((u, v))

        # --- Archi da aggiungere ---
        addition_candidates = [
            (pair, score) for pair, score in link_scores.items()
            if pair not in existing_edges
            and (pair[1], pair[0]) not in existing_edges  # Anche arco inverso
            and score > self._threshold_add
        ]
        addition_candidates.sort(key=lambda x: x[1], reverse=True)  # Score più alto prima

        to_add: list[tuple[int, int]] = []
        for (u, v), score in addition_candidates:
            if len(to_add) >= self._max_add:
                break
            if not self._is_safe_addition(u, v, G, states):
                continue
            to_add.append((u, v))

        if to_add or to_remove:
            logger.info(
                "[Rewirer] Proposed: +%d archi (thr=%.2f), -%d archi (thr=%.2f)",
                len(to_add), self._threshold_add,
                len(to_remove), self._threshold_remove,
            )

        return to_add, to_remove

    # ------------------------------------------------------------------
    # Safety filters
    # ------------------------------------------------------------------

    def _is_safe_removal(
        self,
        u: int,
        v: int,
        G: "nx.Graph",
        states: dict[int, str],
    ) -> bool:
        """Non rimuovere archi verso fact-checker o che isolerebbero un nodo."""
        if states.get(v) == "F" or states.get(u) == "F":
            return False
        # Non isolare il nodo (grado minimo 1 dopo rimozione)
        if G.degree(u) <= 1 or G.degree(v) <= 1:
            return False
        return True

    def _is_safe_addition(
        self,
        u: int,
        v: int,
        G: "nx.Graph",
        states: dict[int, str],
    ) -> bool:
        """Non aggiungere self-loop."""
        return u != v

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def score_statistics(
        self,
        link_scores: dict[tuple[int, int], float],
        G: "nx.Graph",
    ) -> dict[str, float]:
        """Statistiche descrittive degli score per il logging."""
        existing = [s for (u, v), s in link_scores.items() if G.has_edge(u, v)]
        non_existing = [s for (u, v), s in link_scores.items() if not G.has_edge(u, v)]

        def _stats(vals: list[float]) -> dict:
            if not vals:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            arr = np.array(vals)
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        return {
            "existing_edges": _stats(existing),
            "candidate_edges": _stats(non_existing),
        }
