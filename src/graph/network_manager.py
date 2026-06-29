"""
src/graph/network_manager.py
============================
CRUD layer sul grafo dinamico durante la simulazione.

Il NetworkManager e' il SOLO punto di accesso alla topologia del grafo:
  - nessun modulo esterno tocca G direttamente durante la simulazione
  - ogni operazione di lettura/scrittura passa da qui
  - centralizza il logging delle modifiche strutturali

Responsabilita':
  - load / save del grafo (gpickle)
  - accesso al feed del vicinato (sliding window di post)
  - add / remove archi (rewiring)
  - aggiornamento degli embedding dei nodi
  - snapshot delle metriche a ogni step

Utilizzo
--------
    from src.graph.network_manager import NetworkManager
    nm = NetworkManager(subG, cfg, community_map=community_map)
    feed = nm.get_feed(node_id=5, window=5)
    nm.add_post(node_id=5, post={...})
    nm.add_edge(3, 7)
    nm.remove_edge(3, 2)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class NetworkManager:
    """
    Layer di accesso e mutazione del grafo sociale dinamico.

    Parameters
    ----------
    G : nx.Graph
        Sottografo estratto (nodi rinumerati 0..n-1).
    cfg : Config
        Configurazione globale.
    community_map : dict[int, int] | None
        Mappa node_id -> community_id (da Louvain).
    node_features : np.ndarray | None
        Feature matrix iniziale (n, D). Modificata in-place dagli agenti.
    """

    def __init__(
        self,
        G: nx.Graph,
        cfg: "Config",
        community_map: dict[int, int] | None = None,
        node_features: np.ndarray | None = None,
    ) -> None:
        self.G: nx.Graph = G.copy()
        self._cfg = cfg
        self._community_map: dict[int, int] = community_map or {}

        # Embedding dei nodi (mutabili dagli agenti LLM)
        if node_features is not None:
            self._embeddings: np.ndarray = node_features.copy().astype(np.float32)
            actual_dim = self._embeddings.shape[1]
            if actual_dim != cfg.gnn.embedding_dim:
                raise ValueError(
                    f"node_features ha dim={actual_dim} ma cfg.gnn.embedding_dim={cfg.gnn.embedding_dim}. "
                    f"Aggiorna embedding_dim in config.yaml."
                )
        else:
            n = self.G.number_of_nodes()
            d = cfg.gnn.embedding_dim
            self._embeddings = np.zeros((n, d), dtype=np.float32)

        # Post store: {node_id: [post_dict, ...]}
        self._post_store: dict[int, list[dict]] = {n: [] for n in self.G.nodes()}

        # Stato degli agenti: {node_id: "S"|"I"|"R"|"F"}
        self._agent_states: dict[int, str] = {n: "S" for n in self.G.nodes()}

    # ------------------------------------------------------------------
    # Proprieta' di accesso
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> list[int]:
        return list(self.G.nodes())

    @property
    def num_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.G.number_of_edges()

    def neighbours(self, node_id: int) -> list[int]:
        return list(self.G.neighbors(node_id))

    def node_degree(self, node_id: int) -> int:
        return self.G.degree(node_id)

    def get_community(self, node_id: int) -> int:
        return self._community_map.get(node_id, -1)

    def get_embedding(self, node_id: int) -> np.ndarray:
        return self._embeddings[node_id].copy()

    def get_state(self, node_id: int) -> str:
        return self._agent_states.get(node_id, "S")

    def get_all_states(self) -> dict[int, str]:
        return dict(self._agent_states)

    def get_belief_map(self) -> dict[int, float]:
        """Converte stati discreti in valori float [0, 1] per le metriche."""
        state_to_float = {"S": 0.0, "I": 1.0, "R": 0.5, "F": -0.5}
        return {n: state_to_float.get(s, 0.0) for n, s in self._agent_states.items()}

    # ------------------------------------------------------------------
    # Feed vicinato
    # ------------------------------------------------------------------

    def get_feed(self, node_id: int, window: int = 5) -> list[dict]:
        """
        Ritorna gli ultimi `window` post per ogni vicino del nodo,
        ordinati dal piu' recente al piu' vecchio.
        """
        feed: list[dict] = []
        for nb in self.neighbours(node_id):
            posts = self._post_store.get(nb, [])
            feed.extend(posts[-window:])
        feed.sort(key=lambda p: p.get("step", 0), reverse=True)
        # Limita al totale di window * num_neighbours
        max_feed = window * max(len(self.neighbours(node_id)), 1)
        return feed[:max_feed]

    # ------------------------------------------------------------------
    # Operazioni di scrittura
    # ------------------------------------------------------------------

    def add_post(self, node_id: int, post: dict) -> None:
        """Pubblica un post da parte del nodo."""
        self._post_store.setdefault(node_id, []).append(post)
        logger.debug("[NM] Post da nodo %d: '%s'", node_id, post.get("content", "")[:50])

    def set_state(self, node_id: int, state: str) -> None:
        """Aggiorna lo stato dell'agente (S/I/R/F)."""
        valid_states = {"S", "I", "R", "F"}
        if state not in valid_states:
            raise ValueError(f"Stato non valido: '{state}'. Usa uno di {valid_states}.")
        self._agent_states[node_id] = state

    def perturb_embedding(self, node_id: int, delta: np.ndarray, clip: float = 1.0) -> None:
        """
        Aggiunge un delta all'embedding del nodo (perturbazione da agente LLM).
        Il delta viene clippato per stabilita' numerica.
        """
        delta = np.clip(delta, -clip, clip)
        self._embeddings[node_id] += delta

    def add_edge(self, source: int, target: int) -> bool:
        """Follow: aggiungi arco source -- target. Ritorna False se gia' esiste."""
        if self.G.has_edge(source, target):
            return False
        self.G.add_edge(source, target)
        logger.info("[NM] +arco %d -- %d", source, target)
        return True

    def remove_edge(self, source: int, target: int) -> bool:
        """Unfollow: rimuovi arco source -- target. Ritorna False se assente."""
        if not self.G.has_edge(source, target):
            return False
        self.G.remove_edge(source, target)
        logger.info("[NM] -arco %d -- %d", source, target)
        return True

    def apply_rewiring(
        self,
        to_add: list[tuple[int, int]],
        to_remove: list[tuple[int, int]],
    ) -> tuple[list[tuple], list[tuple]]:
        """
        Applica in batch le modifiche topologiche prodotte dalla GNN.

        Returns
        -------
        actually_added, actually_removed : liste degli archi effettivamente modificati.
        """
        added = [(u, v) for u, v in to_add if self.add_edge(u, v)]
        removed = [(u, v) for u, v in to_remove if self.remove_edge(u, v)]
        if added or removed:
            logger.info(
                "[NM] Rewiring: +%d archi, -%d archi | totale archi: %d",
                len(added), len(removed), self.G.number_of_edges(),
            )
        return added, removed

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path, step: int | None = None) -> None:
        """Salva lo stato completo del NetworkManager (grafo + stati + embeddings)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": self.G,
            "agent_states": self._agent_states,
            "community_map": self._community_map,
            "embeddings": self._embeddings,
            "post_store": self._post_store,
            "step": step,
        }
        with open(p, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[NM] Checkpoint salvato: %s (step=%s)", p, step)

    @classmethod
    def load(cls, path: str | Path, cfg: "Config") -> "NetworkManager":
        """Ricarica un NetworkManager da checkpoint."""
        p = Path(path)
        with open(p, "rb") as f:
            payload = pickle.load(f)
        nm = cls(
            G=payload["graph"],
            cfg=cfg,
            community_map=payload.get("community_map"),
            node_features=payload.get("embeddings"),
        )
        nm._agent_states = payload.get("agent_states", {})
        nm._post_store = payload.get("post_store", {})
        logger.info("[NM] Checkpoint caricato: %s (step=%s)", p, payload.get("step"))
        return nm

    # ------------------------------------------------------------------
    # Iteratori
    # ------------------------------------------------------------------

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        yield from self.G.edges()

    def iter_nodes_shuffled(self, seed: int | None = None) -> Iterator[int]:
        """Itera i nodi in ordine randomizzato (per aggiornamento asincrono)."""
        import random
        nodes = self.nodes
        random.Random(seed).shuffle(nodes)
        yield from nodes
