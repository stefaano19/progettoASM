"""
src/graph/metrics.py
====================
Metriche topologiche del grafo per la Fase 0 baseline e il monitoring
continuo durante la simulazione.

Metriche calcolate:
  - Densita', grado medio, clustering coefficient
  - PageRank, Betweenness (campionato), Katz centrality
  - Diameter (approssimato via BFS da campione di nodi)
  - Modularity Q-score
  - Echo Chamber Index (ECI)
  - Belief Polarisation Index (BP)

Design note: le metriche costose (betweenness, diameter) sono disabilitate
per default in locale e possono essere abilitate via config.yaml.

Utilizzo
--------
    from src.graph.metrics import compute_all_metrics, compute_centralities
    metrics = compute_all_metrics(G, cfg, community_map, belief_states)
    centralities = compute_centralities(G, cfg)
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def compute_centralities(
    G: nx.Graph,
    cfg: "Config",
) -> dict[int, dict[str, float]]:
    """
    Calcola le centralita' per ogni nodo del grafo.

    Returns
    -------
    dict[node_id, dict[metric_name, value]]
    """
    seed = cfg.execution.random_seed
    alpha_pr = cfg.metrics.pagerank_alpha
    alpha_katz = cfg.metrics.katz_alpha
    bet_sample = cfg.metrics.betweenness_sample

    result: dict[int, dict[str, float]] = {n: {} for n in G.nodes()}

    # --- Degree centrality (O(n)) ---
    logger.debug("[Metrics] Degree centrality...")
    deg = nx.degree_centrality(G)
    for n, v in deg.items():
        result[n]["degree_centrality"] = v
        result[n]["degree"] = G.degree(n)

    # --- PageRank (O(n * iterations)) ---
    logger.debug("[Metrics] PageRank...")
    try:
        pr = nx.pagerank(G, alpha=alpha_pr)
        for n, v in pr.items():
            result[n]["pagerank"] = v
    except Exception as e:
        logger.warning("[Metrics] PageRank fallito: %s", e)

    # --- Katz centrality (O(n)) ---
    logger.debug("[Metrics] Katz centrality...")
    try:
        katz = nx.katz_centrality_numpy(G, alpha=alpha_katz)
        for n, v in katz.items():
            result[n]["katz"] = v
    except Exception as e:
        logger.warning("[Metrics] Katz fallita: %s", e)

    # --- Betweenness (campionato, costoso) ---
    if cfg.metrics.compute_betweenness:
        logger.info("[Metrics] Betweenness (sample=%d)...", bet_sample)
        try:
            bc = nx.betweenness_centrality(G, k=bet_sample, seed=seed, normalized=True)
            for n, v in bc.items():
                result[n]["betweenness"] = v
        except Exception as e:
            logger.warning("[Metrics] Betweenness fallita: %s", e)
    else:
        # Placeholder per compatibilita' con il resto del sistema
        for n in G.nodes():
            result[n]["betweenness"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Topological metrics (snapshot)
# ---------------------------------------------------------------------------

def compute_topological_metrics(G: nx.Graph, cfg: "Config") -> dict[str, Any]:
    """
    Metriche topologiche globali del grafo.
    Sicure e veloci (nessuna complessita' O(n^2)).
    """
    seed = cfg.execution.random_seed
    n = G.number_of_nodes()
    m = G.number_of_edges()

    metrics: dict[str, Any] = {
        "num_nodes": n,
        "num_edges": m,
        "density": nx.density(G),
        "avg_degree": (2 * m / n) if n > 0 else 0.0,
        "num_connected_components": nx.number_connected_components(G),
    }

    # Clustering coefficient (mediato sui nodi)
    logger.debug("[Metrics] Clustering coefficient...")
    try:
        metrics["avg_clustering"] = nx.average_clustering(G)
    except Exception as e:
        logger.warning("[Metrics] Clustering fallito: %s", e)
        metrics["avg_clustering"] = float("nan")

    # Degree distribution summary
    degrees = [d for _, d in G.degree()]
    metrics["max_degree"] = max(degrees) if degrees else 0
    metrics["min_degree"] = min(degrees) if degrees else 0
    metrics["std_degree"] = float(np.std(degrees)) if degrees else 0.0

    # Diameter approssimato (se abilitato)
    if cfg.metrics.compute_diameter:
        logger.info("[Metrics] Diameter approssimato...")
        metrics["approx_diameter"] = _approx_diameter(G, seed=seed, samples=30)
    else:
        metrics["approx_diameter"] = None

    return metrics


def _approx_diameter(G: nx.Graph, seed: int, samples: int = 30) -> int:
    """
    Stima del diametro campionando BFS da `samples` nodi random.
    Complessita' O(samples * (n + m)).
    """
    rng = random.Random(seed)
    lcc = G.subgraph(max(nx.connected_components(G), key=len))
    sampled = rng.sample(list(lcc.nodes()), min(samples, lcc.number_of_nodes()))
    max_ecc = 0
    for s in sampled:
        lengths = nx.single_source_shortest_path_length(lcc, s)
        max_ecc = max(max_ecc, max(lengths.values()))
    return max_ecc


# ---------------------------------------------------------------------------
# Echo Chamber metrics
# ---------------------------------------------------------------------------

def compute_modularity(
    G: nx.Graph,
    community_map: dict[int, int],
) -> float:
    """
    Q-score (Modularity) tramite assegnazione community da Louvain.
    Usa nx.community.modularity su grafo non diretto.
    """
    if not community_map:
        return float("nan")

    communities_dict: dict[int, set] = {}
    for node, comm_id in community_map.items():
        if node in G:
            communities_dict.setdefault(comm_id, set()).add(node)
    communities = list(communities_dict.values())

    if not communities:
        return float("nan")

    try:
        q = nx.community.modularity(G, communities)
    except Exception as e:
        logger.warning("[Metrics] Modularity fallita: %s", e)
        q = float("nan")
    return q


def compute_echo_chamber_index(
    G: nx.Graph,
    community_map: dict[int, int],
) -> float:
    """
    Echo Chamber Index (ECI):
    Per ogni nodo, calcola la frazione di archi che vanno verso la stessa
    community. ECI = media di questa frazione su tutti i nodi con vicini.
    Range: [0, 1]. Valori alti indicano forte chiusura informativa.
    """
    if not community_map:
        return float("nan")

    ratios: list[float] = []
    for node in G.nodes():
        neighbours = list(G.neighbors(node))
        if not neighbours:
            continue
        my_comm = community_map.get(node, -1)
        intra = sum(
            1 for nb in neighbours
            if community_map.get(nb, -2) == my_comm
        )
        ratios.append(intra / len(neighbours))

    return float(np.mean(ratios)) if ratios else 0.0


def compute_belief_polarisation(belief_states: dict[int, float]) -> float:
    """
    Belief Polarisation Index:
    Varianza degli stati di belief/infezione (numerici) normalizzata a [0, 1].
    Massima varianza per valori binari 0/1 = 0.25.
    """
    if not belief_states:
        return 0.0
    values = list(belief_states.values())
    variance = float(np.var(values))
    # Normalizza: max variance = 0.25 (distribuzione 50/50 tra 0 e 1)
    return min(variance / 0.25, 1.0)


# ---------------------------------------------------------------------------
# All-in-one snapshot
# ---------------------------------------------------------------------------

def compute_all_metrics(
    G: nx.Graph,
    cfg: "Config",
    community_map: dict[int, int] | None = None,
    belief_states: dict[int, float] | None = None,
) -> dict[str, Any]:
    """
    Calcola tutte le metriche disponibili in un unico dict.
    Adatto per logging a ogni step temporale.
    """
    metrics = compute_topological_metrics(G, cfg)

    if community_map:
        metrics["modularity_q"] = compute_modularity(G, community_map)
        metrics["echo_chamber_index"] = compute_echo_chamber_index(G, community_map)
    else:
        metrics["modularity_q"] = None
        metrics["echo_chamber_index"] = None

    if belief_states:
        values = list(belief_states.values())
        metrics["belief_polarisation"] = compute_belief_polarisation(belief_states)
        metrics["mean_belief"] = float(np.mean(values))
        metrics["infection_rate"] = sum(1 for v in values if v > 0.5) / len(values)
    else:
        metrics["belief_polarisation"] = None
        metrics["mean_belief"] = None
        metrics["infection_rate"] = None

    return metrics
