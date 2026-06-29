"""
src/gnn/embeddings.py
=====================
Inizializzazione e gestione degli embedding dei nodi.

Flusso
------
1. Se disponibili le feature OGB (128-dim) -> PCA a `embedding_dim` dim.
2. Altrimenti -> embedding strutturali: mean di one-hot + degree features.
3. Normalizza su sfera unitaria (L2).
4. Salva/carica da file .npy.

I embedding sono la "memoria strutturale" del nodo: vengono
perturbati dagli agenti LLM (agent.py) e usati dalla GNN per
calcolare le probabilita' di link.

Utilizzo
--------
    from src.gnn.embeddings import EmbeddingManager
    em = EmbeddingManager(cfg)
    embeddings = em.initialize(subG, node_features)   # (n, embedding_dim)
    em.save(embeddings)
    embeddings = em.load()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Gestisce l'inizializzazione, salvataggio e caricamento degli embedding.

    Parameters
    ----------
    cfg : Config
        Configurazione globale.
    """

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        self._dim = cfg.gnn.embedding_dim
        self._seed = cfg.execution.random_seed
        self._path = cfg.project_root / cfg.gnn.embedding_file

    # ------------------------------------------------------------------
    # Inizializzazione
    # ------------------------------------------------------------------

    def initialize(
        self,
        G: "nx.Graph",
        node_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Inizializza gli embedding per tutti i nodi del grafo.

        Se `node_features` e' fornito (n x feat_dim), usa PCA per ridurre
        alla dimensione target. Altrimenti usa embedding strutturali.

        Parameters
        ----------
        G : nx.Graph
            Grafo (nodi 0..n-1).
        node_features : np.ndarray | None
            Feature matrix OGB (n x 128) o None.

        Returns
        -------
        embeddings : np.ndarray
            Matrice (n, embedding_dim) normalizzata L2.
        """
        n = G.number_of_nodes()
        target_dim = self._dim

        if node_features is not None and node_features.shape[0] == n:
            if node_features.shape[1] == target_dim:
                logger.info("[Embeddings] Feature gia' nella dimensione target (%d).", target_dim)
                embeddings = node_features.astype(np.float32)
            elif node_features.shape[1] > target_dim:
                logger.info(
                    "[Embeddings] PCA: %d -> %d dim.",
                    node_features.shape[1], target_dim,
                )
                embeddings = self._pca_reduce(node_features, target_dim)
            else:
                # Feature dim < target: pad con zeri
                pad = np.zeros((n, target_dim - node_features.shape[1]), dtype=np.float32)
                embeddings = np.concatenate([node_features.astype(np.float32), pad], axis=1)
        else:
            logger.info("[Embeddings] Nessuna feature OGB — uso embedding strutturali.")
            embeddings = self._structural_embeddings(G, target_dim)

        embeddings = self._l2_normalize(embeddings)
        logger.info(
            "[Embeddings] Shape finale: %s | norm media: %.4f",
            embeddings.shape,
            float(np.mean(np.linalg.norm(embeddings, axis=1))),
        )
        return embeddings

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, embeddings: np.ndarray) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self._path), embeddings)
        logger.info("[Embeddings] Salvati: %s", self._path)

    def load(self) -> np.ndarray:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Embedding file non trovato: {self._path}. "
                "Esegui prima EmbeddingManager.initialize()."
            )
        emb = np.load(str(self._path))
        logger.info("[Embeddings] Caricati da: %s | shape: %s", self._path, emb.shape)
        return emb

    def exists(self) -> bool:
        return self._path.exists()

    # ------------------------------------------------------------------
    # Metodi interni
    # ------------------------------------------------------------------

    def _pca_reduce(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """
        PCA manuale via SVD (scipy/numpy, senza sklearn per portabilita').
        """
        X = features.astype(np.float64)
        # Centra
        mean = X.mean(axis=0)
        X_c = X - mean
        # SVD
        try:
            U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            reduced = U[:, :target_dim] * S[:target_dim]
        except np.linalg.LinAlgError:
            logger.warning("[Embeddings] SVD fallita — uso random init.")
            rng = np.random.default_rng(self._seed)
            reduced = rng.standard_normal((features.shape[0], target_dim))
        return reduced.astype(np.float32)

    def _structural_embeddings(self, G: "nx.Graph", dim: int) -> np.ndarray:
        """
        Embedding strutturali quando le feature OGB non sono disponibili.
        Combina: degree normalizzato, clustering coefficient, community one-hot.
        """
        import networkx as nx

        n = G.number_of_nodes()
        rng = np.random.default_rng(self._seed)

        # Base: random gaussiano
        base = rng.standard_normal((n, dim)).astype(np.float32)

        # Arricchisci con segnali strutturali (prime 4 dim)
        nodes = sorted(G.nodes())
        node_to_idx = {v: i for i, v in enumerate(nodes)}

        degrees = np.array([G.degree(v) for v in nodes], dtype=np.float32)
        max_deg = max(degrees.max(), 1.0)
        degrees_norm = degrees / max_deg

        try:
            clustering = np.array(
                [nx.clustering(G, v) for v in nodes], dtype=np.float32
            )
        except Exception:
            clustering = np.zeros(n, dtype=np.float32)

        # community info dal node attr (se disponibile)
        community_ids = np.array(
            [G.nodes[v].get("community", v % 4) for v in nodes], dtype=np.float32
        )
        max_comm = max(community_ids.max(), 1.0)
        community_norm = community_ids / max_comm

        # Embed segnali strutturali nelle prime dimensioni
        if dim >= 3:
            base[:, 0] = degrees_norm
            base[:, 1] = clustering
            base[:, 2] = community_norm

        return base

    @staticmethod
    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return X / norms
