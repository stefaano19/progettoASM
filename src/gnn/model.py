"""
src/gnn/model.py
================
GraphSAGE con doppia modalita' di esecuzione:

  MODE 1 — NumPy (always available, no GPU)
    Forward pass deterministico con pesi fissi (seed-based).
    Aggregazione: mean pooling dei vicini.
    Usato in locale per test e dry-run.

  MODE 2 — PyTorch (optional, GPU on Kaggle)
    GraphSAGE con 2 layer, gradiente completo, training via SGD.
    Attivato automaticamente se `import torch` ha successo.
    Usa lo stesso schema di aggregazione del numpy mode.

Link Prediction
---------------
Entrambe le modalita' usano il dot product sui vettori di output:
    score(u, v) = sigmoid(h_u . h_v)

Utilizzo
--------
    from src.gnn.model import GraphSAGEModel
    model = GraphSAGEModel(in_dim=64, hidden_dim=128, out_dim=64, seed=42)
    embeddings = model.forward(G, node_embeddings)       # (n, out_dim)
    score = model.link_score(embeddings[u], embeddings[v])  # float in [0,1]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detect PyTorch
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
    logger.debug("[GraphSAGE] PyTorch disponibile — modalita' torch.")
except ImportError:
    _TORCH_AVAILABLE = False
    logger.debug("[GraphSAGE] PyTorch non disponibile — modalita' numpy.")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


# ---------------------------------------------------------------------------
# NumPy GraphSAGE (fallback always available)
# ---------------------------------------------------------------------------

class _NumpySAGELayer:
    """
    Single GraphSAGE layer (mean aggregation) in numpy.
    Weights are fixed at initialization (no gradients).
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int, layer_idx: int) -> None:
        rng = np.random.default_rng(seed + layer_idx * 1000)
        # Xavier init
        limit = np.sqrt(6.0 / (2 * in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, (out_dim, 2 * in_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, h: np.ndarray, adj: list[list[int]]) -> np.ndarray:
        n = h.shape[0]
        h_new = np.empty((n, self.W.shape[0]), dtype=np.float32)
        for i in range(n):
            h_self = h[i]
            nbrs = adj[i]
            if nbrs:
                h_agg = np.mean(h[nbrs], axis=0)
            else:
                h_agg = h_self  # Self-loop fallback
            h_cat = np.concatenate([h_self, h_agg])
            h_new[i] = _relu(self.W @ h_cat + self.b)
        return h_new


class _NumpyGraphSAGE:
    """
    2-layer GraphSAGE in pure numpy.
    Weights are fixed (no training) — used for local testing.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int) -> None:
        self.layer1 = _NumpySAGELayer(in_dim, hidden_dim, seed, layer_idx=0)
        self.layer2 = _NumpySAGELayer(hidden_dim, out_dim, seed, layer_idx=1)
        self._in_dim = in_dim
        self._out_dim = out_dim

    def forward(self, adj: list[list[int]], h: np.ndarray) -> np.ndarray:
        h1 = self.layer1.forward(h, adj)
        h2 = self.layer2.forward(h1, adj)
        # L2 normalize output
        norms = np.linalg.norm(h2, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return h2 / norms

    def get_weights(self) -> dict[str, np.ndarray]:
        return {
            "layer1_W": self.layer1.W.copy(),
            "layer1_b": self.layer1.b.copy(),
            "layer2_W": self.layer2.W.copy(),
            "layer2_b": self.layer2.b.copy(),
        }

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.layer1.W = weights["layer1_W"].copy()
        self.layer1.b = weights["layer1_b"].copy()
        self.layer2.W = weights["layer2_W"].copy()
        self.layer2.b = weights["layer2_b"].copy()


# ---------------------------------------------------------------------------
# PyTorch GraphSAGE (optional)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class _TorchSAGELayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(2 * in_dim, out_dim)

        def forward(self, h: "torch.Tensor", adj: list[list[int]]) -> "torch.Tensor":
            n = h.shape[0]
            agg = torch.zeros_like(h)
            for i in range(n):
                nbrs = adj[i]
                if nbrs:
                    agg[i] = h[nbrs].mean(dim=0)
                else:
                    agg[i] = h[i]
            h_cat = torch.cat([h, agg], dim=1)
            return F.relu(self.linear(h_cat))

    class _TorchGraphSAGE(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
            super().__init__()
            self.layer1 = _TorchSAGELayer(in_dim, hidden_dim)
            self.layer2 = _TorchSAGELayer(hidden_dim, out_dim)

        def forward(self, adj: list[list[int]], h: "torch.Tensor") -> "torch.Tensor":
            h1 = self.layer1(h, adj)
            h2 = self.layer2(h1, adj)
            norms = h2.norm(dim=1, keepdim=True).clamp(min=1e-8)
            return h2 / norms

        def get_weights(self) -> dict[str, np.ndarray]:
            return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

        def set_weights(self, weights: dict[str, np.ndarray]) -> None:
            state = {k: torch.tensor(v) for k, v in weights.items()}
            self.load_state_dict(state)


# ---------------------------------------------------------------------------
# Public facade: GraphSAGEModel
# ---------------------------------------------------------------------------

class GraphSAGEModel:
    """
    GraphSAGE facade che seleziona automaticamente numpy o torch backend.

    Parameters
    ----------
    in_dim : int       Dimensione input embedding.
    hidden_dim : int   Dimensione layer nascosto.
    out_dim : int      Dimensione embedding output.
    seed : int         Per init deterministico (numpy mode).
    force_numpy : bool Forza la modalita' numpy anche se torch e' disponibile.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        seed: int = 42,
        force_numpy: bool = False,
    ) -> None:
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._use_torch = _TORCH_AVAILABLE and not force_numpy

        if self._use_torch:
            import torch
            self._model = _TorchGraphSAGE(in_dim, hidden_dim, out_dim)
            torch.manual_seed(seed)
            logger.info("[GraphSAGE] Backend: PyTorch | in=%d hid=%d out=%d", in_dim, hidden_dim, out_dim)
        else:
            self._model = _NumpyGraphSAGE(in_dim, hidden_dim, out_dim, seed)
            logger.info("[GraphSAGE] Backend: NumPy | in=%d hid=%d out=%d", in_dim, hidden_dim, out_dim)

    @property
    def uses_torch(self) -> bool:
        return self._use_torch

    def forward(self, G: "nx.Graph", embeddings: np.ndarray) -> np.ndarray:
        """
        Forward pass: calcola nuovi embedding per tutti i nodi.

        Parameters
        ----------
        G : nx.Graph            Grafo corrente (topologia).
        embeddings : np.ndarray Input embedding (n, in_dim).

        Returns
        -------
        np.ndarray (n, out_dim) — nuovi embedding normalizzati L2.
        """
        nodes = sorted(G.nodes())
        node_to_idx = {v: i for i, v in enumerate(nodes)}
        adj = [
            [node_to_idx[nb] for nb in G.neighbors(v) if nb in node_to_idx]
            for v in nodes
        ]

        if self._use_torch:
            import torch
            h = torch.tensor(embeddings, dtype=torch.float32)
            with torch.no_grad():
                out = self._model.forward(adj, h)
            return out.numpy()
        else:
            return self._model.forward(adj, embeddings)

    def link_score(self, emb_u: np.ndarray, emb_v: np.ndarray) -> float:
        """Dot product link score in [0, 1]."""
        return float(_sigmoid(np.dot(emb_u, emb_v)))

    def score_edges(
        self,
        embeddings: np.ndarray,
        candidates: list[tuple[int, int]],
    ) -> dict[tuple[int, int], float]:
        """
        Calcola lo score per una lista di coppie (u, v).

        Returns
        -------
        dict[(u, v), score]
        """
        return {
            (u, v): self.link_score(embeddings[u], embeddings[v])
            for u, v in candidates
        }

    def get_weights(self) -> dict[str, np.ndarray]:
        return self._model.get_weights()

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._model.set_weights(weights)
