"""
src/graph/metrics.py
====================
Metriche topologiche del grafo per la Fase 0 baseline e il monitoring
continuo durante la simulazione.

Metriche calcolate:
  - Densita', grado medio, clustering coefficient (campionato per grafi grandi)
  - PageRank, Betweenness (campionato), Katz centrality (scipy sparse)
  - Diameter (approssimato via BFS da campione di nodi)
  - Modularity Q-score
  - Echo Chamber Index (ECI)
  - Belief Polarisation Index (BP)

Design note: le metriche costose sono automaticamente adattate alla scala
del grafo. Per grafi > 10k nodi si usano approssimazioni campionate.

Utilizzo
--------
    from src.graph.metrics import compute_all_metrics, compute_centralities
    metrics = compute_all_metrics(G, cfg, community_map, belief_states)
    centralities = compute_centralities(G, cfg)
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)

# Soglia oltre la quale usiamo versioni campionate/approssimate
_LARGE_GRAPH_THRESHOLD = 10_000


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
    n = G.number_of_nodes()

    result: dict[int, dict[str, float]] = {n_id: {} for n_id in G.nodes()}

    # --- Degree centrality (O(n)) ---
    logger.info("[Metrics] Degree centrality (%d nodi)...", n)
    deg = nx.degree_centrality(G)
    for n_id, v in deg.items():
        result[n_id]["degree_centrality"] = v
        result[n_id]["degree"] = G.degree(n_id)

    # --- PageRank (scipy sparse, O(n * iterations)) ---
    logger.info("[Metrics] PageRank...")
    try:
        pr = nx.pagerank(G, alpha=alpha_pr, max_iter=100, tol=1e-6)
        for n_id, v in pr.items():
            result[n_id]["pagerank"] = v
    except Exception as e:
        logger.warning("[Metrics] PageRank fallito: %s", e)

    # --- Katz centrality (scipy sparse diretto) ---
    logger.info("[Metrics] Katz centrality (scipy sparse)...")
    try:
        katz = _katz_centrality_sparse(G, alpha=alpha_katz)
        if katz is not None:
            for n_id, v in katz.items():
                result[n_id]["katz"] = v
    except Exception as e:
        logger.warning("[Metrics] Katz fallita: %s", e)

    # --- Betweenness (campionato, costoso) ---
    if cfg.metrics.compute_betweenness:
        logger.info("[Metrics] Betweenness (sample=%d)...", bet_sample)
        try:
            bc = nx.betweenness_centrality(G, k=bet_sample, seed=seed, normalized=True)
            for n_id, v in bc.items():
                result[n_id]["betweenness"] = v
        except Exception as e:
            logger.warning("[Metrics] Betweenness fallita: %s", e)
    else:
        # Placeholder per compatibilita' con il resto del sistema
        for n_id in G.nodes():
            result[n_id]["betweenness"] = 0.0

    return result


def _katz_centrality_sparse(G: nx.Graph, alpha: float) -> dict[int, float] | None:
    """
    Katz centrality via scipy sparse: risolve (I - alpha*A)^{-1} * 1
    usando un risolutore iterativo (spsolve o bicgstab).
    Memoria: O(E) — nessuna matrice densa.
    """
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError:
        logger.warning("[Metrics] scipy non disponibile, Katz saltata.")
        return None

    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Matrice di adiacenza sparsa
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csc", dtype=np.float64)

    # Risolvi (I - alpha*A) * x = 1
    I = sp.eye(n, format="csc", dtype=np.float64)
    M = I - alpha * A
    b = np.ones(n, dtype=np.float64)

    try:
        # Prova risolutore diretto (velocissimo per grafi fino a ~500k)
        x = spla.spsolve(M, b)
    except Exception:
        # Fallback a iterativo
        logger.info("[Metrics] Katz: fallback a risolutore iterativo (bicgstab)...")
        x, info = spla.bicgstab(M, b, tol=1e-5, maxiter=500)
        if info != 0:
            logger.warning("[Metrics] Katz bicgstab non convergente (info=%d)", info)
            return None

    # Normalizza
    norm = np.sign(x.sum()) * np.linalg.norm(x)
    if abs(norm) > 1e-12:
        x = x / norm

    return {nodes[i]: float(x[i]) for i in range(n)}


# ---------------------------------------------------------------------------
# Topological metrics (snapshot)
# ---------------------------------------------------------------------------

def compute_topological_metrics(G: nx.Graph, cfg: "Config") -> dict[str, Any]:
    """
    Metriche topologiche globali del grafo.
    Per grafi grandi usa approssimazioni campionate.
    """
    seed = cfg.execution.random_seed
    n = G.number_of_nodes()
    m = G.number_of_edges()
    is_large = n > _LARGE_GRAPH_THRESHOLD

    metrics: dict[str, Any] = {
        "num_nodes": n,
        "num_edges": m,
        "density": nx.density(G),
        "avg_degree": (2 * m / n) if n > 0 else 0.0,
    }

    # Connected components — veloce O(n+m), ma lo facciamo solo una volta
    logger.info("[Metrics] Connected components...")
    metrics["num_connected_components"] = nx.number_connected_components(G)

    # Clustering coefficient — campionato per grafi grandi
    if is_large:
        sample_size = min(5000, n)
        logger.info("[Metrics] Clustering coefficient (campionato, %d nodi)...", sample_size)
        rng = random.Random(seed)
        sampled_nodes = rng.sample(list(G.nodes()), sample_size)
        try:
            metrics["avg_clustering"] = nx.average_clustering(G, nodes=sampled_nodes)
        except Exception as e:
            logger.warning("[Metrics] Clustering fallito: %s", e)
            metrics["avg_clustering"] = float("nan")
    else:
        logger.info("[Metrics] Clustering coefficient (esatto)...")
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
