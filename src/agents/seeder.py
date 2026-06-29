"""
src/agents/seeder.py
====================
Identificazione e selezione dei "Pazienti Zero" (patient zero nodes).

I pazienti zero sono i nodi che ricevono lo stato iniziale "I" (Infected)
all'inizio della simulazione e fungono da sorgente della diffusione.

Strategie di selezione
-----------------------
  "pagerank"    : top-k nodi per PageRank (influencer globali)
  "betweenness" : top-k per Betweenness centrality (bridge nodes)
  "katz"        : top-k per Katz centrality (influenza locale + globale)
  "degree"      : top-k per grado (hub locali, rapido calcolo)
  "combined"    : media pesata di pagerank + katz + degree (default)
  "cross_community": seleziona k nodi distribuiti su community diverse
                     (massimizza copertura iniziale)
  "random"      : campionamento casuale riproducibile con seed

Utilizzo
--------
    from src.agents.seeder import Seeder
    seeder = Seeder(cfg)
    patient_zero_ids = seeder.select(
        G=subG,
        centralities=centrality_dict,
        community_map=community_map,
        k=5,
    )
    seeder.inject(network_manager, patient_zero_ids)
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.graph.network_manager import NetworkManager
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class Seeder:
    """
    Seleziona e inietta i nodi paziente zero nel grafo.

    Parameters
    ----------
    cfg : Config
        Configurazione globale.
    strategy : str
        Strategia di selezione (override di config se fornita).
    """

    VALID_STRATEGIES = {
        "pagerank", "betweenness", "katz",
        "degree", "combined", "cross_community", "random",
    }

    def __init__(
        self,
        cfg: "Config",
        strategy: str = "combined",
    ) -> None:
        self._cfg = cfg
        self._seed = cfg.execution.random_seed
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Strategia non valida: '{strategy}'. "
                f"Scegli tra {self.VALID_STRATEGIES}."
            )
        self._strategy = strategy

    # ------------------------------------------------------------------
    # Selezione
    # ------------------------------------------------------------------

    def select(
        self,
        G: nx.Graph,
        centralities: dict[int, dict[str, float]],
        community_map: dict[int, int],
        k: int | None = None,
    ) -> list[int]:
        """
        Seleziona i k nodi paziente zero.

        Parameters
        ----------
        G : nx.Graph
            Sottografo su cui operare.
        centralities : dict[int, dict[str, float]]
            Dizionario {node_id: {metric: value}} da compute_centralities().
        community_map : dict[int, int]
            {node_id: community_id} da Louvain.
        k : int | None
            Numero di seed. Default: initial_infection_rate * n_nodes.

        Returns
        -------
        list[int]
            Lista dei node_id selezionati come pazienti zero.
        """
        n = G.number_of_nodes()
        if k is None:
            rate = self._cfg.simulation.initial_infection_rate
            k = max(1, int(round(rate * n)))

        k = min(k, n)  # Non piu' nodi di quanti ce ne sono

        logger.info(
            "[Seeder] Selezione %d pazienti zero | strategia='%s' | n=%d",
            k, self._strategy, n,
        )

        strategy_fn = {
            "pagerank": self._by_pagerank,
            "betweenness": self._by_betweenness,
            "katz": self._by_katz,
            "degree": self._by_degree,
            "combined": self._by_combined,
            "cross_community": self._by_cross_community,
            "random": self._by_random,
        }[self._strategy]

        selected = strategy_fn(G, centralities, community_map, k)

        logger.info(
            "[Seeder] Pazienti zero selezionati: %s",
            selected,
        )
        return selected

    # ------------------------------------------------------------------
    # Iniezione
    # ------------------------------------------------------------------

    def inject(
        self,
        network_manager: "NetworkManager",
        patient_zero_ids: list[int],
        initial_state: str = "I",
    ) -> None:
        """
        Inietta lo stato iniziale nei nodi paziente zero.

        Parameters
        ----------
        network_manager : NetworkManager
            Grafo dinamico.
        patient_zero_ids : list[int]
            Nodi da portare allo stato `initial_state`.
        initial_state : str
            "I" per infezione, "F" per fact-checker CELF.
        """
        for node_id in patient_zero_ids:
            network_manager.set_state(node_id, initial_state)
            # Aggiungi un post iniziale per rendere la diffusione visibile
            if initial_state == "I":
                content = (
                    f"[SEED] I am convinced about '{self._cfg.simulation.topic}'. "
                    "Join me — the evidence is overwhelming."
                )
            else:
                content = (
                    f"[FACT-CHECK] Let us examine the claims about "
                    f"'{self._cfg.simulation.topic}' carefully."
                )
            network_manager.add_post(node_id, {
                "node_id": node_id,
                "step": 0,
                "content": content,
                "author_state": initial_state,
            })

        logger.info(
            "[Seeder] Iniettati %d nodi in stato '%s'.",
            len(patient_zero_ids), initial_state,
        )

    # ------------------------------------------------------------------
    # Score helper
    # ------------------------------------------------------------------

    def _scores_to_ranked(
        self,
        scores: dict[int, float],
        k: int,
    ) -> list[int]:
        """Ordina per score decrescente e prende i top-k."""
        return sorted(scores, key=lambda n: scores[n], reverse=True)[:k]

    # ------------------------------------------------------------------
    # Strategie
    # ------------------------------------------------------------------

    def _by_pagerank(
        self,
        G: nx.Graph,
        centralities: dict[int, dict[str, float]],
        community_map: dict[int, int],
        k: int,
    ) -> list[int]:
        scores = {n: d.get("pagerank", 0.0) for n, d in centralities.items()}
        return self._scores_to_ranked(scores, k)

    def _by_betweenness(
        self,
        G: nx.Graph,
        centralities: dict[int, dict[str, float]],
        community_map: dict[int, int],
        k: int,
    ) -> list[int]:
        scores = {n: d.get("betweenness", 0.0) for n, d in centralities.items()}
        return self._scores_to_ranked(scores, k)

    def _by_katz(
        self,
        G: nx.Graph,
        centralities: dict[int, dict[str, float]],
        community_map: dict[int, int],
        k: int,
    ) -> list[int]:
        scores = {n: d.get("katz", 0.0) for n, d in centralities.items()}
        return self._scores_to_ranked(scores, k)

    def _by_degree(
        self,
        G: nx.Graph,
        centralities: dict[int, dict[str, float]],
        community_map: dict[int, int],
        k: int,
    ) -> list[int]:
        scores = {n: float(G.degree(n)) for n in G.nodes()}
        return self._scores_to_ranked(scores, k)

    def _by_combined(
        self,
        G: nx.Graph,
        centralities: dict[int, dict[str, float]],
        community_map: dict[int, int],
        k: int,
    ) -> list[int]:
        """
        Score combinato: media pesata di PageRank (40%) + Katz (40%) + Degree (20%).
        Ogni metrica e' normalizzata in [0, 1] prima di combinarla.
        """
        def normalize(vals: dict[int, float]) -> dict[int, float]:
            mx = max(vals.values()) if vals else 1.0
            mn = min(vals.values()) if vals else 0.0
            rng = mx - mn
            if rng < 1e-12:
                return {n: 0.0 for n in vals}
            return {n: (v - mn) / rng for n, v in vals.items()}

        pr = normalize({n: d.get("pagerank", 0.0) for n, d in centralities.items()})
        kz = normalize({n: d.get("katz", 0.0) for n, d in centralities.items()})
        dg = normalize({n: float(G.degree(n)) for n in G.nodes()})

        combined = {
            n: 0.4 * pr.get(n, 0.0) + 0.4 * kz.get(n, 0.0) + 0.2 * dg.get(n, 0.0)
            for n in G.nodes()
        }
        return self._scores_to_ranked(combined, k)

    def _by_cross_community(
        self,
        G: nx.Graph,
        centralities: dict[int, dict[str, float]],
        community_map: dict[int, int],
        k: int,
    ) -> list[int]:
        """
        Seleziona il nodo con score combinato piu' alto per ogni community,
        distribuendo i k seed su piu' community possibili.
        """
        # Raggruppa per community
        comm_nodes: dict[int, list[int]] = {}
        for node, comm in community_map.items():
            if node in G:
                comm_nodes.setdefault(comm, []).append(node)

        # Score combinato dentro ogni community
        def _combined_score(node_id: int) -> float:
            d = centralities.get(node_id, {})
            return (
                0.4 * d.get("pagerank", 0.0)
                + 0.4 * d.get("katz", 0.0)
                + 0.2 * float(G.degree(node_id))
            )

        # Prendi il top-1 per community, ciclicamente fino a k
        per_comm_tops: dict[int, list[int]] = {}
        for comm, nodes in comm_nodes.items():
            sorted_nodes = sorted(nodes, key=_combined_score, reverse=True)
            per_comm_tops[comm] = sorted_nodes

        selected: list[int] = []
        comm_ids = sorted(per_comm_tops.keys())
        idx = 0
        ranks: dict[int, int] = {c: 0 for c in comm_ids}

        while len(selected) < k:
            comm = comm_ids[idx % len(comm_ids)]
            rank = ranks[comm]
            nodes_in_comm = per_comm_tops[comm]
            if rank < len(nodes_in_comm):
                candidate = nodes_in_comm[rank]
                if candidate not in selected:
                    selected.append(candidate)
                ranks[comm] += 1
            idx += 1
            # Sicurezza: se abbiamo esaurito tutti i nodi disponibili
            if idx > k * len(comm_ids) * 2:
                break

        return selected[:k]

    def _by_random(
        self,
        G: nx.Graph,
        centralities: dict[int, dict[str, float]],
        community_map: dict[int, int],
        k: int,
    ) -> list[int]:
        rng = random.Random(self._seed)
        return rng.sample(list(G.nodes()), k)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def describe_seeds(
        self,
        patient_zero_ids: list[int],
        centralities: dict[int, dict[str, float]],
        community_map: dict[int, int],
    ) -> list[dict]:
        """
        Restituisce un report descrittivo dei nodi paziente zero.
        Utile per il logging e il report finale.
        """
        report = []
        for node_id in patient_zero_ids:
            d = centralities.get(node_id, {})
            report.append({
                "node_id": node_id,
                "community": community_map.get(node_id, -1),
                "pagerank": round(d.get("pagerank", 0.0), 6),
                "katz": round(d.get("katz", 0.0), 6),
                "degree": int(d.get("degree", 0)),
            })
        return report
