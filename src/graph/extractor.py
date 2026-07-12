"""
src/graph/extractor.py
======================
Campionamento di un sottografo connesso e rappresentativo da ogbl-collab.

Strategie disponibili (config.subgraph.strategy):
  - "bfs_seed"      : BFS a partire da un nodo ad alto grado (default)
  - "random_walk"   : random walk con restart
  - "random_nodes"  : campionamento puramente casuale di nodi + archi indotti
  - "forest_fire"   : Forest Fire Sampling (Leskovec & Faloutsos, KDD 2006)

Il sottografo viene salvato come .gpickle e le feature vengono allineate.

Utilizzo
--------
    from src.graph.extractor import extract_subgraph
    subG, sub_features = extract_subgraph(G, node_features, cfg)
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_subgraph(
    G: nx.Graph,
    node_features: np.ndarray,
    cfg: "Config",
) -> tuple[nx.Graph, np.ndarray, dict[int, int]]:
    """
    Estrai un sottografo connesso di dimensione target_nodes.

    Parameters
    ----------
    G : nx.Graph
        Grafo completo (ogbl-collab).
    node_features : np.ndarray
        Feature matrix originale (N, D).
    cfg : Config
        Configurazione con sezione 'subgraph'.

    Returns
    -------
    subG : nx.Graph
        Sottografo estratto (nodi rinumerati 0..n-1).
    sub_features : np.ndarray
        Feature matrix allineata al sottografo (n, D).
    node_map : dict[int, int]
        Mappa old_node_id -> new_node_id nel sottografo.
    """
    output_path = cfg.project_root / cfg.subgraph.output_file
    feature_path = cfg.project_root / cfg.gnn.embedding_file

    # --- Cache hit ---
    if output_path.exists() and feature_path.exists():
        logger.info("[Extractor] Caricamento sottografo da cache: %s", output_path)
        with open(output_path, "rb") as f:
            data = pickle.load(f)
        sub_features = np.load(str(feature_path))
        return data["graph"], sub_features, data["node_map"]

    # --- Estrazione ---
    strategy = cfg.subgraph.strategy
    target_n = cfg.subgraph.target_nodes
    seed = cfg.execution.random_seed
    seed_strategy = cfg.subgraph.seed_strategy

    logger.info(
        "[Extractor] Strategia='%s', target=%d nodi, seed=%d",
        strategy, target_n, seed,
    )

    # Lavora sulla largest connected component per garantire connettivita'
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc).copy()
    logger.info(
        "[Extractor] LCC: %d nodi, %d archi",
        G_lcc.number_of_nodes(), G_lcc.number_of_edges(),
    )

    if target_n is None or target_n >= G_lcc.number_of_nodes():
        logger.info("[Extractor] target_n >= LCC size. Prendo tutto il grafo connesso.")
        selected_nodes = list(G_lcc.nodes())
    elif strategy == "bfs_seed":
        selected_nodes = _bfs_sample(G_lcc, target_n, seed, seed_strategy)
    elif strategy == "random_walk":
        selected_nodes = _random_walk_sample(G_lcc, target_n, seed, seed_strategy)
    elif strategy == "random_nodes":
        selected_nodes = _random_nodes_sample(G_lcc, target_n, seed)
    elif strategy == "forest_fire":
        forward_prob = getattr(cfg.subgraph, 'forward_prob', 0.5)
        selected_nodes = _forest_fire_sample(
            G_lcc, target_n, seed, seed_strategy, forward_prob=forward_prob,
        )
        logger.info(
            "[Extractor] Forest Fire: forward_prob=%.2f", forward_prob,
        )
    else:
        raise ValueError(f"Strategia di campionamento non valida: '{strategy}'")

    # Sottografo indotto
    subG_raw = G_lcc.subgraph(selected_nodes).copy()

    # Verifica connettivita' e prendi LCC del sottografo
    if not nx.is_connected(subG_raw):
        lcc_sub = max(nx.connected_components(subG_raw), key=len)
        subG_raw = subG_raw.subgraph(lcc_sub).copy()
        logger.warning(
            "[Extractor] Sottografo non connesso -> LCC: %d nodi",
            subG_raw.number_of_nodes(),
        )

    # --- Rinumerazione nodi 0..n-1 ---
    old_nodes = sorted(subG_raw.nodes())
    node_map = {old: new for new, old in enumerate(old_nodes)}
    subG = nx.relabel_nodes(subG_raw, node_map)

    # --- Feature allineate ---
    sub_features = node_features[old_nodes]  # (n, D)

    logger.info(
        "[Extractor] Sottografo finale: %d nodi, %d archi | density=%.4f",
        subG.number_of_nodes(),
        subG.number_of_edges(),
        nx.density(subG),
    )

    # --- Salvataggio ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(
            {"graph": subG, "node_map": node_map, "old_nodes": old_nodes},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    np.save(str(feature_path), sub_features)
    logger.info("[Extractor] Salvato: %s | %s", output_path, feature_path)

    return subG, sub_features, node_map


# ---------------------------------------------------------------------------
# Strategie di campionamento
# ---------------------------------------------------------------------------

def _get_seed_node(G: nx.Graph, strategy: str, rng_seed: int) -> int:
    """Seleziona il nodo di partenza per BFS/random walk."""
    import random
    rng = random.Random(rng_seed)

    if strategy == "high_degree":
        # Nodo con grado piu' alto (hub della rete)
        return max(G.nodes(), key=lambda n: G.degree(n))
    elif strategy == "random":
        return rng.choice(list(G.nodes()))
    else:
        raise ValueError(f"seed_strategy non valida: '{strategy}'")


def _bfs_sample(
    G: nx.Graph,
    target_n: int,
    seed: int,
    seed_strategy: str,
) -> list[int]:
    """
    BFS dal nodo seme fino a raggiungere target_n nodi.
    Visita i vicini in ordine randomizzato per diversita'.
    """
    import random
    rng = random.Random(seed)

    start = _get_seed_node(G, seed_strategy, seed)
    visited: set[int] = {start}
    queue: deque[int] = deque([start])

    while queue and len(visited) < target_n:
        node = queue.popleft()
        neighbours = list(G.neighbors(node))
        rng.shuffle(neighbours)
        for nb in neighbours:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= target_n:
                    break

    logger.debug("[Extractor] BFS campionati %d nodi (target=%d)", len(visited), target_n)
    return list(visited)


def _random_walk_sample(
    G: nx.Graph,
    target_n: int,
    seed: int,
    seed_strategy: str,
    restart_prob: float = 0.15,
) -> list[int]:
    """
    Random walk con restart (RWR) per campionamento piu' omogeneo.
    Evita di rimanere intrappolato in zone dense.
    """
    import random
    rng = random.Random(seed)
    nodes_list = list(G.nodes())

    start = _get_seed_node(G, seed_strategy, seed)
    current = start
    visited: set[int] = {current}

    max_iters = target_n * 20
    for _ in range(max_iters):
        if len(visited) >= target_n:
            break
        if rng.random() < restart_prob:
            current = start
        else:
            neighbours = list(G.neighbors(current))
            if not neighbours:
                current = start
                continue
            current = rng.choice(neighbours)
        visited.add(current)

    logger.debug("[Extractor] RWR campionati %d nodi (target=%d)", len(visited), target_n)
    return list(visited)


def _random_nodes_sample(G: nx.Graph, target_n: int, seed: int) -> list[int]:
    """
    Campionamento puramente casuale di nodi.
    Il sottografo indotto potrebbe essere disconnesso.
    """
    import random
    rng = random.Random(seed)
    nodes = list(G.nodes())
    sampled = rng.sample(nodes, min(target_n, len(nodes)))
    logger.debug("[Extractor] Random sample: %d nodi", len(sampled))
    return sampled


def _forest_fire_sample(
    G: nx.Graph,
    target_n: int,
    seed: int,
    seed_strategy: str,
    forward_prob: float = 0.5,
) -> list[int]:
    """
    Forest Fire Sampling (Leskovec & Faloutsos, KDD 2006).

    Preserva meglio le proprieta' strutturali del grafo rispetto a BFS
    e random walk: distribuzione dei gradi, coefficiente di clustering,
    struttura delle comunita' e densita' locale.

    Algoritmo:
      1. Scegli un nodo seme (ambassador).
      2. Genera x ~ Geometric(1 - forward_prob) — numero di vicini da "bruciare".
      3. Seleziona x vicini non visitati a caso e aggiungili alla frontiera.
      4. Per ogni vicino bruciato, ripeti ricorsivamente (con stack, non ricorsione).
      5. Se la frontiera si esaurisce, ricomincia da un nuovo nodo random.

    Parameters
    ----------
    G : nx.Graph
        Grafo sorgente (deve essere connesso).
    target_n : int
        Numero target di nodi.
    seed : int
        Seed per riproducibilita'.
    seed_strategy : str
        Strategia per il nodo seme iniziale.
    forward_prob : float
        Probabilita' di propagazione del fuoco. Valori tipici: 0.4-0.7.
        Valori piu' alti producono sottografi piu' densi (simili a BFS).
        Valori piu' bassi producono campioni piu' sparsi e diversificati.
    """
    import random
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    start = _get_seed_node(G, seed_strategy, seed)
    visited: set[int] = {start}
    stack: list[int] = [start]

    while len(visited) < target_n:
        if not stack:
            # Fuoco spento: ricomincia da un nodo random non visitato
            remaining = list(set(G.nodes()) - visited)
            if not remaining:
                break
            new_start = rng.choice(remaining)
            visited.add(new_start)
            stack.append(new_start)
            continue

        node = stack.pop()
        neighbours = [nb for nb in G.neighbors(node) if nb not in visited]
        if not neighbours:
            continue

        # Geometrica: quanti vicini "bruciare"
        n_burn = min(
            int(np_rng.geometric(1.0 - forward_prob)),
            len(neighbours),
        )
        n_burn = max(1, n_burn)  # Brucia almeno 1 vicino

        burned = rng.sample(neighbours, n_burn)
        for nb in burned:
            if len(visited) >= target_n:
                break
            visited.add(nb)
            stack.append(nb)

    logger.debug(
        "[Extractor] Forest Fire campionati %d nodi (target=%d, p=%.2f)",
        len(visited), target_n, forward_prob,
    )
    return list(visited)
