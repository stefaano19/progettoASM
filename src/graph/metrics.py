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

try:
    import cugraph
    import cudf
    _CUGRAPH_AVAILABLE = True
    logger.debug("[Metrics] cuGraph disponibile — accelerazione GPU abilitata.")
except ImportError:
    _CUGRAPH_AVAILABLE = False


def _parallel_diameter_chunk(args: tuple[nx.Graph, list[int]]) -> int:
    """Helper per calcolare i path più lunghi in parallelo (deve essere top-level)."""
    import networkx as nx
    G, sources = args
    max_ecc = 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(G, s)
        if lengths:
            max_ecc = max(max_ecc, max(lengths.values()))
    return max_ecc


def _parallel_betweenness_chunk(args: tuple[nx.Graph, list[int]]) -> dict[int, float]:
    """Helper per calcolare betweenness in parallelo su un chunk di sorgenti."""
    import networkx as nx
    G, sources = args
    return nx.betweenness_centrality_subset(
        G, sources, list(G.nodes()), normalized=False, weight=None
    )


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

    # --- PageRank ---
    logger.info("[Metrics] PageRank...")
    try:
        if _CUGRAPH_AVAILABLE:
            import cugraph
            import cudf
            G_cu = cugraph.from_networkx(G)
            df = cugraph.pagerank(G_cu, alpha=alpha_pr)
            pr = df.to_pandas().set_index("vertex")["pagerank"].to_dict()
            for n_id, v in pr.items():
                result[n_id]["pagerank"] = v
        else:
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

    # --- Betweenness (campionato, efficiente in parallelo o su GPU) ---
    if cfg.metrics.compute_betweenness:
        logger.info("[Metrics] Betweenness (sample=%d)...", bet_sample)
        try:
            sample_size = min(bet_sample, n)
            
            if _CUGRAPH_AVAILABLE:
                import cugraph
                import cudf
                logger.info("[Metrics] Betweenness su GPU (cugraph)...")
                G_cu = cugraph.from_networkx(G)
                # cuGraph scale is sometimes different, ma normalized=True gestisce il default
                df = cugraph.betweenness_centrality(G_cu, k=sample_size, normalized=True)
                bc = df.to_pandas().set_index("vertex")["betweenness_centrality"].to_dict()
                for n_id, v in bc.items():
                    result[n_id]["betweenness"] = v
            else:
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor
                
                rng = random.Random(seed)
                sampled = rng.sample(list(G.nodes()), sample_size)
                
                n_cores = max(1, mp.cpu_count() - 1)
                chunk_size = max(1, len(sampled) // n_cores)
                chunks = [sampled[i:i + chunk_size] for i in range(0, len(sampled), chunk_size)]
                
                bc_raw = {n_id: 0.0 for n_id in G.nodes()}
                ctx = mp.get_context("spawn")  # Evita CUDA initialization crash in Kaggle
                with ProcessPoolExecutor(max_workers=n_cores, mp_context=ctx) as executor:
                    futures = []
                    for chunk in chunks:
                        futures.append(executor.submit(_parallel_betweenness_chunk, (G, chunk)))
                    for f in futures:
                        res = f.result()
                        for node, v in res.items():
                            bc_raw[node] += v
                
                # Normalizzazione
                scale = 1.0
                if n > 2:
                    scale = 1.0 / ((n - 1) * (n - 2))
                scale *= float(n) / sample_size
                
                is_directed = G.is_directed() if hasattr(G, 'is_directed') else False
                if not is_directed:
                    scale *= 2.0  # In nx.betweenness_centrality undirected scale is 2 / ((n-1)(n-2))
                
                for n_id, v in bc_raw.items():
                    result[n_id]["betweenness"] = v * scale
                
        except Exception as e:
            logger.warning("[Metrics] Betweenness fallita: %s", e)
    else:
        # Placeholder per compatibilita' con il resto del sistema
        for n_id in G.nodes():
            result[n_id]["betweenness"] = 0.0

    return result


def _katz_centrality_sparse(
    G: nx.Graph,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict[int, float] | None:
    """
    Katz centrality via power iteration su matrice sparsa.

    NOTA STORICA: la versione precedente risolveva (I - alpha*A)^{-1} * 1
    con spsolve() (fattorizzazione LU diretta). Su grafi con distribuzione
    di grado "scale-free"/hub-based (tipico di reti sociali ed
    epidemiologiche) la fattorizzazione LU sparsa soffre di fill-in
    catastrofico: il fattore L diventa denso anche se A e' sparsissima, e
    il tempo/memoria richiesti esplodono SENZA che scipy lanci un'eccezione
    o vada in timeout — il processo resta semplicemente bloccato.

    Questa versione usa solo prodotti matrice-vettore sparsi (A @ x),
    O(E) per iterazione: nessuna fattorizzazione, nessun fill-in. Scala
    bene anche su grafi da centinaia di migliaia di nodi.
    """
    try:
        import scipy.sparse.linalg as spla
    except ImportError:
        logger.warning("[Metrics] scipy non disponibile, Katz saltata.")
        return None

    n = G.number_of_nodes()
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=np.float64)

    # La serie di Neumann (I - alpha*A)^-1 = sum_k (alpha*A)^k converge solo
    # se alpha < 1 / lambda_max(A). Stimiamo lambda_max con eigsh (anch'esso
    # basato solo su prodotti matrice-vettore, quindi economico) e, se
    # necessario, riduciamo alpha per garantire la convergenza.
    try:
        lambda_max = float(
            spla.eigsh(A, k=1, which="LA", return_eigenvectors=False)[0]
        )
        if lambda_max > 0 and alpha >= 1.0 / lambda_max:
            safe_alpha = 0.9 / lambda_max
            logger.warning(
                "[Metrics] Katz: alpha=%.4f non garantisce convergenza "
                "(serve < %.4f). Uso alpha=%.4f per questa run.",
                alpha, 1.0 / lambda_max, safe_alpha,
            )
            alpha = safe_alpha
    except Exception as e:
        logger.warning(
            "[Metrics] Katz: stima lambda_max fallita (%s), procedo con alpha originale.",
            e,
        )

    beta = 1.0
    x = np.zeros(n, dtype=np.float64)
    converged = False
    for _ in range(max_iter):
        x_new = alpha * (A @ x) + beta
        if np.linalg.norm(x_new - x, ord=1) < tol:
            x = x_new
            converged = True
            break
        x = x_new

    if not converged:
        logger.warning(
            "[Metrics] Katz: power iteration non convergente dopo %d iterazioni.",
            max_iter,
        )

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
    Stima del diametro campionando BFS da `samples` nodi random in parallelo.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import random

    rng = random.Random(seed)
    lcc = G.subgraph(max(nx.connected_components(G), key=len))
    sampled = rng.sample(list(lcc.nodes()), min(samples, lcc.number_of_nodes()))
    
    n_cores = max(1, mp.cpu_count() - 1)
    chunk_size = max(1, len(sampled) // n_cores) if sampled else 1
    chunks = [sampled[i:i + chunk_size] for i in range(0, len(sampled), chunk_size)]
    
    max_ecc = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_cores, mp_context=ctx) as executor:
        futures = []
        for chunk in chunks:
            futures.append(executor.submit(_parallel_diameter_chunk, (lcc, chunk)))
        for f in futures:
            max_ecc = max(max_ecc, f.result())
            
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
    Usa python-louvain (community_louvain.modularity) per massima efficienza.
    """
    if not community_map:
        return float("nan")

    try:
        import community.community_louvain as community_louvain
        q = community_louvain.modularity(community_map, G)
    except ImportError:
        # Fallback a networkx se python-louvain non è disponibile
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
    Ottimizzato scorrendo gli archi invece dei vicini di ogni nodo (O(E) vs O(V * avg_deg)).
    """
    if not community_map:
        return float("nan")

    intra_degree = {n: 0 for n in G.nodes()}
    
    # Scorriamo tutti gli archi una sola volta (più veloce di G.neighbors per ogni nodo)
    for u, v in G.edges():
        if u not in community_map or v not in community_map:
            continue
        if community_map[u] == community_map[v]:
            intra_degree[u] += 1
            intra_degree[v] += 1

    ratios: list[float] = []
    for node in G.nodes():
        deg = G.degree(node)
        if deg > 0:
            ratios.append(intra_degree[node] / deg)

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

# Cache globale per metriche topologiche statiche
_CACHE = {
    "num_edges": None,
    "topological_metrics": None,
    "modularity_q": None,
    "echo_chamber_index": None
}

def compute_all_metrics(
    G: nx.Graph,
    cfg: "Config",
    community_map: dict[int, int] | None = None,
    belief_states: dict[int, float] | None = None,
    force_topology_update: bool = False,
) -> dict[str, Any]:
    """
    Calcola tutte le metriche disponibili in un unico dict.
    Adatto per logging a ogni step temporale.
    Ottimizzato con caching: ricalcola le metriche topologiche solo se il numero di archi cambia 
    oppure se force_topology_update è True.
    """
    global _CACHE
    m = G.number_of_edges()

    # Ricalcola se esplicitamente richiesto, se la cache è vuota, o se num_edges è cambiato
    if force_topology_update or _CACHE["topological_metrics"] is None or _CACHE["num_edges"] != m:
        _CACHE["topological_metrics"] = compute_topological_metrics(G, cfg)
        if community_map:
            _CACHE["modularity_q"] = compute_modularity(G, community_map)
            _CACHE["echo_chamber_index"] = compute_echo_chamber_index(G, community_map)
        _CACHE["num_edges"] = m

    metrics = _CACHE["topological_metrics"].copy()

    if community_map:
        metrics["modularity_q"] = _CACHE["modularity_q"]
        metrics["echo_chamber_index"] = _CACHE["echo_chamber_index"]
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
