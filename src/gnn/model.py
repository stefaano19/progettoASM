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
import warnings
from typing import TYPE_CHECKING

import numpy as np

warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")

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


# Scala iniziale dei logit: gli embedding di output hanno norma sqrt(scala),
# quindi score(u, v) = sigmoid(scala * cos(h_u, h_v)). Con norma 1 (scala 1)
# il logit restava in [-1, 1] e lo score in [0.27, 0.73]. In PyTorch la scala
# e' un parametro appreso (logit_scale = log(scala)).
INITIAL_LOGIT_SCALE = 5.0


# ---------------------------------------------------------------------------
# NumPy GraphSAGE (fallback always available)
# ---------------------------------------------------------------------------

class _NumpySAGELayer:
    """
    Single GraphSAGE layer (mean aggregation) in numpy.
    Weights are fixed at initialization (no gradients).
    """

    def __init__(
        self, in_dim: int, out_dim: int, seed: int, layer_idx: int, activation: bool = True,
    ) -> None:
        self.activation = activation
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
            z = self.W @ h_cat + self.b
            h_new[i] = _relu(z) if self.activation else z
        return h_new


class _NumpyGraphSAGE:
    """
    2-layer GraphSAGE in pure numpy.
    Weights are fixed (no training) — used for local testing.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int) -> None:
        self.layer1 = _NumpySAGELayer(in_dim, hidden_dim, seed, layer_idx=0)
        # Ultimo layer lineare: vedi nota in _TorchGraphSAGE
        self.layer2 = _NumpySAGELayer(hidden_dim, out_dim, seed, layer_idx=1, activation=False)
        self._in_dim = in_dim
        self._out_dim = out_dim

    def forward(self, adj: list[list[int]], h: np.ndarray) -> np.ndarray:
        h1 = self.layer1.forward(h, adj)
        h2 = self.layer2.forward(h1, adj)
        # L2 normalize output, poi scala (vedi INITIAL_LOGIT_SCALE)
        norms = np.linalg.norm(h2, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return (h2 / norms * np.sqrt(np.exp(self.logit_scale))).astype(np.float32)

    logit_scale: float = float(np.log(INITIAL_LOGIT_SCALE))

    def get_weights(self) -> dict[str, np.ndarray]:
        return {
            "layer1_W": self.layer1.W.copy(),
            "layer1_b": self.layer1.b.copy(),
            "layer2_W": self.layer2.W.copy(),
            "layer2_b": self.layer2.b.copy(),
            "logit_scale": np.array(self.logit_scale, dtype=np.float32),
        }

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.layer1.W = weights["layer1_W"].copy()
        self.layer1.b = weights["layer1_b"].copy()
        self.layer2.W = weights["layer2_W"].copy()
        self.layer2.b = weights["layer2_b"].copy()
        if "logit_scale" in weights:
            self.logit_scale = float(weights["logit_scale"])


# ---------------------------------------------------------------------------
# PyTorch GraphSAGE (optional)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    def build_sparse_adj(G: "nx.Graph", device="cpu", dtype=None) -> "torch.Tensor":
        import networkx as nx
        import numpy as np
        
        nodes = sorted(G.nodes())
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="coo", dtype=np.float32)
        
        row = A.row
        col = A.col
        data = np.ones_like(row, dtype=np.float32)
        
        degrees = np.array(A.sum(axis=1)).flatten()
        isolated = np.where(degrees == 0)[0]
        
        if len(isolated) > 0:
            row = np.concatenate([row, isolated])
            col = np.concatenate([col, isolated])
            data = np.concatenate([data, np.ones(len(isolated), dtype=np.float32)])
            degrees[isolated] = 1.0
            
        data = data / degrees[row]
        
        indices = torch.tensor(np.vstack((row, col)), dtype=torch.int64)
        values = torch.tensor(data, dtype=dtype if dtype is not None else torch.float32)
        
        adj_sparse = torch.sparse_coo_tensor(indices, values, torch.Size(A.shape), device=device)
        return adj_sparse.coalesce()

    class _TorchSAGELayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, activation: bool = True) -> None:
            super().__init__()
            self.linear = nn.Linear(2 * in_dim, out_dim)
            self.activation = activation

        def forward(self, h: "torch.Tensor", adj_sparse: "torch.Tensor") -> "torch.Tensor":
            agg = torch.sparse.mm(adj_sparse, h)
            h_cat = torch.cat([h, agg], dim=1)
            out = self.linear(h_cat)
            return F.relu(out) if self.activation else out

    class _TorchGraphSAGE(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
            super().__init__()
            self.layer1 = _TorchSAGELayer(in_dim, hidden_dim)
            # FIX: ultimo layer SENZA ReLU. Con ReLU + normalizzazione L2 gli
            # embedding di output erano tutti non negativi e di norma 1, quindi
            # il prodotto scalare stava in [0, 1] e lo score sigmoid in
            # [0.50, 0.73]: nessun arco poteva scendere sotto la soglia di
            # rimozione e il modello non poteva separare davvero i negativi.
            self.layer2 = _TorchSAGELayer(hidden_dim, out_dim, activation=False)
            # Scala appresa del logit: con output di norma 1 lo score restava
            # in [0.27, 0.73] anche senza ReLU (vedi INITIAL_LOGIT_SCALE).
            self.logit_scale = nn.Parameter(
                torch.tensor(float(np.log(INITIAL_LOGIT_SCALE)))
            )

        def forward(self, adj_sparse: "torch.Tensor", h: "torch.Tensor") -> "torch.Tensor":
            h1 = self.layer1(h, adj_sparse)
            h2 = self.layer2(h1, adj_sparse)
            norms = h2.norm(dim=1, keepdim=True).clamp(min=1e-8)
            scale = self.logit_scale.clamp(max=float(np.log(100.0))).exp().sqrt()
            return h2 / norms * scale

        def get_weights(self) -> dict[str, np.ndarray]:
            return {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}

        def set_weights(self, weights: dict[str, np.ndarray]) -> None:
            device = next(self.parameters()).device
            state = {k: torch.tensor(v, device=device) for k, v in weights.items()}
            # Checkpoint precedenti a logit_scale: mantieni il valore iniziale
            missing, unexpected = self.load_state_dict(state, strict=False)
            if unexpected or [k for k in missing if k != "logit_scale"]:
                raise KeyError(f"Pesi GNN incompatibili: mancanti={missing} inattesi={unexpected}")


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
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # FIX: seed PRIMA di creare il modello (prima veniva impostato dopo,
            # quindi l'inizializzazione dei pesi non dipendeva dal seed).
            torch.manual_seed(seed)
            self._model = _TorchGraphSAGE(in_dim, hidden_dim, out_dim).to(self._device)
            logger.info("[GraphSAGE] Backend: PyTorch (%s) | in=%d hid=%d out=%d", self._device, in_dim, hidden_dim, out_dim)
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
        np.ndarray (n, out_dim) — embedding di norma sqrt(scala dei logit),
        cosi' che il prodotto scalare sia scala * coseno.
        """
        if self._use_torch:
            import torch
            h = torch.tensor(embeddings, dtype=torch.float32, device=self._device)
            adj_sparse = build_sparse_adj(G, device=self._device, dtype=h.dtype)
            with torch.no_grad():
                out = self._model.forward(adj_sparse, h)
            return out.cpu().numpy()
        else:
            nodes = sorted(G.nodes())
            node_to_idx = {v: i for i, v in enumerate(nodes)}
            adj = [
                [node_to_idx[nb] for nb in G.neighbors(v) if nb in node_to_idx]
                for v in nodes
            ]
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
