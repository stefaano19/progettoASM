"""
src/graph/community.py
======================
Community detection sul sottografo estratto.

Algoritmi supportati (config.community.algorithm):
  - "louvain"           : Algoritmo di Louvain (python-louvain)
  - "label_propagation" : Label Propagation (networkx, nessuna dep extra)

Output:
  - community_map: dict {node_id -> community_id}
  - n_communities: int
  - modularity Q associato all'assegnazione

Utilizzo
--------
    from src.graph.community import detect_communities
    community_map, n_comm, q = detect_communities(G, cfg)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


def detect_communities(
    G: nx.Graph,
    cfg: "Config",
) -> tuple[dict[int, int], int, float]:
    """
    Esegui community detection e restituisce la mappa nodo -> community.

    Parameters
    ----------
    G : nx.Graph
        Grafo (preferibilmente il sottografo estratto).
    cfg : Config
        Configurazione con sezione 'community'.

    Returns
    -------
    community_map : dict[int, int]
        {node_id: community_id} per ogni nodo.
    n_communities : int
        Numero di community trovate.
    modularity_q : float
        Q-score dell'assegnazione.
    """
    output_path = cfg.project_root / cfg.community.output_file
    algorithm = cfg.community.algorithm
    seed = cfg.execution.random_seed

    # --- Cache hit ---
    if output_path.exists():
        logger.info("[Community] Caricamento community map da cache: %s", output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        community_map = {int(k): int(v) for k, v in data["community_map"].items()}
        
        # Validazione cache: se la dimensione non combacia con il grafo, scarta la cache
        if len(community_map) == G.number_of_nodes():
            return community_map, data["n_communities"], data["modularity_q"]
        else:
            logger.warning("[Community] La cache non corrisponde alla dimensione del grafo! Ricalcolo...")

    # --- Algoritmo ---
    logger.info("[Community] Community detection con algoritmo='%s'...", algorithm)

    if algorithm == "louvain":
        community_map, q = _louvain(G, cfg)
    elif algorithm == "label_propagation":
        community_map, q = _label_propagation(G, seed)
    else:
        raise ValueError(f"Algoritmo non supportato: '{algorithm}'")

    n_communities = len(set(community_map.values()))
    logger.info(
        "[Community] Trovate %d community | Q=%.4f", n_communities, q
    )

    # Aggiungi attributo nodo al grafo (utile per NetworkManager)
    nx.set_node_attributes(G, community_map, name="community")

    # --- Salvataggio ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "community_map": {str(k): v for k, v in community_map.items()},
                "n_communities": n_communities,
                "modularity_q": q,
                "algorithm": algorithm,
            },
            f,
            indent=2,
        )
    logger.info("[Community] Community map salvata: %s", output_path)

    return community_map, n_communities, q


def _louvain(G: nx.Graph, cfg: "Config") -> tuple[dict[int, int], float]:
    """Louvain tramite python-louvain."""
    try:
        import community as community_louvain  # python-louvain
    except ImportError as e:
        raise ImportError(
            "Installa python-louvain: pip install python-louvain\n"
            "Oppure: pip install -r requirements.txt"
        ) from e

    resolution = cfg.community.resolution
    seed = cfg.execution.random_seed

    partition: dict[int, int] = community_louvain.best_partition(
        G,
        resolution=resolution,
        random_state=seed,
    )
    q = community_louvain.modularity(partition, G)
    return partition, q


def _label_propagation(
    G: nx.Graph,
    seed: int,
) -> tuple[dict[int, int], float]:
    """
    Label Propagation via networkx (nessuna dipendenza extra).
    Meno stabile di Louvain ma non richiede python-louvain.
    """
    communities = nx.community.label_propagation_communities(G)
    partition: dict[int, int] = {}
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            partition[node] = comm_id

    # Calcola Q con nx
    community_sets = list(
        {cid: set() for cid in set(partition.values())}.values()
    )
    for node, cid in partition.items():
        # Rigenera sets per modularity
        pass

    comm_dict: dict[int, set] = {}
    for node, cid in partition.items():
        comm_dict.setdefault(cid, set()).add(node)
    communities_list = list(comm_dict.values())

    try:
        q = nx.community.modularity(G, communities_list)
    except Exception:
        q = float("nan")

    return partition, q


def community_stats(
    G: nx.Graph,
    community_map: dict[int, int],
) -> dict:
    """
    Statistiche descrittive sulle community trovate.
    Utile per la fase di reporting e validazione.
    """
    comm_sizes: dict[int, int] = {}
    for node, cid in community_map.items():
        comm_sizes[cid] = comm_sizes.get(cid, 0) + 1

    sizes = list(comm_sizes.values())
    import numpy as np

    return {
        "n_communities": len(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "mean_size": float(np.mean(sizes)),
        "std_size": float(np.std(sizes)),
        "size_distribution": dict(sorted(comm_sizes.items())),
    }
