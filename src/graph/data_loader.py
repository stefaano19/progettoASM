"""
src/graph/data_loader.py
========================
Download e caricamento di ogbl-collab tramite Open Graph Benchmark (OGB).

Converte il dataset dal formato OGB (edge_index tensore) a un grafo
NetworkX non diretto, con attributi sui nodi (anno, feature).

ogbl-collab facts:
  - ~235,868 nodi (autori)
  - ~1,285,465 archi (co-autorships)
  - Node features: 128-dim media word2vec dei titoli dei paper
  - Edge features: anno di collaborazione (1963-2020)

Utilizzo
--------
    from src.graph.data_loader import load_collab_graph
    G = load_collab_graph(cfg)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


def load_collab_graph(cfg: "Config") -> tuple[nx.Graph, np.ndarray]:
    """
    Scarica (se necessario) e carica ogbl-collab come grafo NetworkX.

    Parameters
    ----------
    cfg : Config
        Config oggetto dal loader YAML.

    Returns
    -------
    G : nx.Graph
        Grafo non diretto con attributi nodo (feature, degree).
    node_features : np.ndarray
        Matrice (n_nodes, 128) delle feature OGB.
    """
    ogb_root = cfg.project_root / cfg.paths.data_raw
    cache_path = cfg.project_root / cfg.paths.data_processed / "full_graph.gpickle"

    # --- Cache hit ---
    if cache_path.exists():
        logger.info("[DataLoader] Caricamento grafo da cache: %s", cache_path)
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["graph"], data["features"]

    # --- Caricamento OGB ---
    logger.info("[DataLoader] Download/caricamento ogbl-collab da OGB...")
    try:
        from ogb.linkproppred import LinkPropPredDataset
    except ImportError as e:
        raise ImportError(
            "Installa ogb: pip install ogb\n"
            "Oppure: pip install -r requirements.txt"
        ) from e

    dataset = LinkPropPredDataset(name="ogbl-collab", root=str(ogb_root))
    graph_data = dataset[0]  # dict con 'edge_index', 'edge_year', 'node_feat'

    edge_index: np.ndarray = graph_data["edge_index"]      # (2, E)
    node_feat: np.ndarray = graph_data["node_feat"]         # (N, 128)
    edge_year: np.ndarray | None = graph_data.get("edge_year")  # (E, 1) o None

    n_nodes = node_feat.shape[0]
    n_edges = edge_index.shape[1]
    logger.info(
        "[DataLoader] ogbl-collab: %d nodi, %d archi",
        n_nodes, n_edges,
    )

    # --- Costruzione grafo NetworkX ---
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Applica filtro anno se configurato
    year_filter = cfg.dataset.year_filter
    src_arr = edge_index[0]
    dst_arr = edge_index[1]

    if year_filter is not None and edge_year is not None:
        years = edge_year.flatten()
        mask = years <= year_filter
        src_arr = src_arr[mask]
        dst_arr = dst_arr[mask]
        logger.info(
            "[DataLoader] Filtro anno <= %d: %d archi rimasti (su %d)",
            year_filter, mask.sum(), n_edges,
        )

    # Aggiungi archi in batch (molto piu' veloce di add_edge per ogni arco)
    edges = list(zip(src_arr.tolist(), dst_arr.tolist()))
    G.add_edges_from(edges)

    logger.info(
        "[DataLoader] Grafo costruito: %d nodi, %d archi",
        G.number_of_nodes(), G.number_of_edges(),
    )

    # --- Salva cache ---
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"graph": G, "features": node_feat}, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("[DataLoader] Cache salvata: %s", cache_path)

    return G, node_feat


def graph_summary(G: nx.Graph) -> dict:
    """Statistiche rapide del grafo completo (senza metriche costose)."""
    largest_cc = max(nx.connected_components(G), key=len)
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_directed": G.is_directed(),
        "largest_cc_nodes": len(largest_cc),
        "largest_cc_frac": len(largest_cc) / G.number_of_nodes(),
        "num_connected_components": nx.number_connected_components(G),
        "density": nx.density(G),
    }
