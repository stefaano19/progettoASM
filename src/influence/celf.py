"""
src/influence/celf.py
=====================
CELF (Cost-Effective Lazy Forward) per Influence Maximization.

Algoritmo
---------
CELF e' un'ottimizzazione dell'algoritmo greedy per l'Influence Maximization
che sfrutta la proprieta' di submodularita' della funzione di spread per
ridurre il numero di costose simulazioni Monte Carlo.

Invece di rivalutare il marginal gain di ogni candidato ad ogni iterazione,
CELF mantiene una coda di priorita' ordinata per upper-bound del marginal
gain e rivaluta solo quando necessario (lazy evaluation).

Complessita': O(k * R * (n + m)) nel caso medio vs O(k * n * R * (n + m))
del greedy puro.

Modello di diffusione
---------------------
Usa Independent Cascade (IC):
  - Ogni arco (u, v) ha probabilita' di attivazione proporzionale
    all'inverso del grado di v (normalizzazione).
  - Si avvia da un insieme di seed S e si propaga finche' non ci sono
    piu' attivazioni nuove.

Utilizzo
--------
    from src.influence.celf import CELF
    celf = CELF(cfg)
    seeds = celf.select(G, budget_k=10, agent_states=nm.get_all_states())
"""

from __future__ import annotations

import heapq
import logging
import random
from typing import TYPE_CHECKING

import numpy as np

def _parallel_celf_chunk(args) -> list[tuple[int, float]]:
    """Helper top-level per valutare i candidati in parallelo bypassando il GIL."""
    import random
    G, candidates, states, n_rounds, act_prob = args
    rng = random.Random()
    results = []
    for node in candidates:
        # Usa CELF._simulate_spread senza istanziare la classe
        from src.influence.celf import CELF
        gain = CELF._simulate_spread(G, [node], states, n_rounds, act_prob, rng)
        results.append((node, gain))
    return results

if TYPE_CHECKING:
    import networkx as nx
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class CELF:
    """
    Cost-Effective Lazy Forward per Influence Maximization.

    Parameters
    ----------
    cfg : Config
        Configurazione globale (usa cfg.influence.* e cfg.execution.random_seed).
    """

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        self._seed = cfg.execution.random_seed
        self._n_rounds = cfg.influence.simulation_rounds

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def select(
        self,
        G: "nx.Graph",
        budget_k: int | None = None,
        agent_states: dict[int, str] | None = None,
        seed_state: str = "F",
    ) -> list[int]:
        """
        Seleziona i `budget_k` nodi seed ottimali per il fact-checking.

        Esclude automaticamente nodi gia' in stato 'I' (infetti) come
        candidati (non ha senso iniettarli come fact-checker).

        Parameters
        ----------
        G : nx.Graph
            Grafo corrente.
        budget_k : int | None
            Numero di seed da selezionare. Default: cfg.influence.budget_k.
        agent_states : dict[int, str] | None
            Stato corrente degli agenti {node_id: "S"|"I"|"R"|"F"}.
        seed_state : str
            Stato da assegnare ai seed (default "F" = Fact-Checker).

        Returns
        -------
        list[int]
            Lista di `budget_k` node_id selezionati come seed.
        """
        k = budget_k if budget_k is not None else self._cfg.influence.budget_k
        states = agent_states or {}

        # Candidati: esclude nodi gia' infetti o gia' fact-checker
        excluded_states = {"I", seed_state}
        candidates = [
            n for n in G.nodes()
            if states.get(n, "S") not in excluded_states
        ]

        if not candidates:
            logger.warning("[CELF] Nessun candidato disponibile (tutti I o F).")
            return []

        k = min(k, len(candidates))
        logger.info(
            "[CELF] Avvio selezione: k=%d | candidati=%d | rounds=%d",
            k, len(candidates), self._n_rounds,
        )

        selected: list[int] = []
        # Heap: (-marginal_gain, last_updated_iteration, node_id)
        # Usiamo heap min-heap con gain negato per simulare max-heap
        heap: list[tuple[float, int, int]] = []

        # Pre-calcola probabilita' di attivazione una sola volta.
        # Ottimizzato: usa numpy per calcolare tutti i gradi in vettore.
        logger.info("[CELF] Pre-calcolo probabilita' di attivazione archi...")
        edges_list = list(G.edges())
        activation_prob: dict[tuple[int, int], float] = {}
        if edges_list:
            # Vettorizza il calcolo dei gradi
            deg_dict = dict(G.degree())
            for u, v in edges_list:
                activation_prob[(u, v)] = 1.0 / max(deg_dict[v], 1)
                activation_prob[(v, u)] = 1.0 / max(deg_dict[u], 1)

        # Calcola guadagno marginale iniziale per tutti i candidati in parallelo
        logger.info("[CELF] Calcolo spread iniziale per tutti i candidati [ProcessPool]...")
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        
        n_cores = max(1, mp.cpu_count() - 1)
        chunk_size = max(1, len(candidates) // n_cores) if candidates else 1
        chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
        
        # Sostituito ThreadPoolExecutor con ProcessPoolExecutor per bypassare il GIL di Python, 
        # dato che il modello IC è cpu-bound. Usa context 'spawn' per evitare CUDA crash in Kaggle.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_cores, mp_context=ctx) as executor:
            futures = [
                executor.submit(_parallel_celf_chunk, (G, chunk, states, self._n_rounds, activation_prob))
                for chunk in chunks
            ]
            for f in futures:
                for node, gain in f.result():
                    heapq.heappush(heap, (-gain, 0, node))  # iteration 0

        # Loop greedy lazy
        for iteration in range(k):
            if not heap:
                break

            while True:
                neg_gain, last_iter, node = heapq.heappop(heap)

                if last_iter == iteration:
                    # Gain e' fresco per questa iterazione: prendi
                    selected.append(node)
                    logger.info(
                        "[CELF] Iter %d/%d: selezionato nodo %d (spread=%.2f)",
                        iteration + 1, k, node, -neg_gain,
                    )
                    break
                else:
                    # Rivaluta il marginal gain con i seed gia' selezionati
                    current_spread = self._simulate_spread(
                        G, selected, states, self._n_rounds, activation_prob
                    )
                    new_spread = self._simulate_spread(
                        G, selected + [node], states, self._n_rounds, activation_prob
                    )
                    marginal = new_spread - current_spread
                    heapq.heappush(heap, (-marginal, iteration, node))

        logger.info("[CELF] Seed selezionati: %s", selected)
        return selected

    # ------------------------------------------------------------------
    # Simulazione di diffusione IC (Independent Cascade)
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_spread(
        G: "nx.Graph",
        seeds: list[int],
        agent_states: dict[int, str],
        n_rounds: int,
        activation_prob: dict[tuple[int, int], float],
        rng: random.Random | None = None,
    ) -> float:
        """
        Stima il numero atteso di nodi raggiunti dalla propagazione
        a partire dai `seeds`, usando Monte Carlo con modello IC.
        Ottimizzato: usa set per il lookup 'in activated' O(1) invece di O(n),
        e precomputa i vicini per ogni nodo (evita chiamate ripetute G.neighbors).
        """
        if not seeds:
            return 0.0

        _rng = rng or random.Random(42)
        total_reached = 0

        # Precomputa adiacency list e stati una sola volta per tutti i round
        nbrs_cache: dict[int, list[int]] = {n: list(G.neighbors(n)) for n in seeds}
        for seed in seeds:
            for nb in nbrs_cache[seed]:
                if nb not in nbrs_cache:
                    nbrs_cache[nb] = list(G.neighbors(nb))

        for _ in range(n_rounds):
            activated: set[int] = set(seeds)
            frontier: list[int] = list(seeds)

            while frontier:
                next_frontier: list[int] = []
                for node in frontier:
                    for nb in G.neighbors(node):  # usa G.neighbors direttamente
                        if nb not in activated:
                            p = activation_prob.get((node, nb), 0.1)
                            state = agent_states.get(nb, "S")
                            if state == "R":
                                p *= 0.3
                            elif state == "F":
                                p = 0.0
                            if p > 0.0 and _rng.random() < p:
                                activated.add(nb)
                                next_frontier.append(nb)
                frontier = next_frontier

            total_reached += len(activated)

        return total_reached / n_rounds

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def estimate_spread(
        self,
        G: "nx.Graph",
        seeds: list[int],
        agent_states: dict[int, str] | None = None,
    ) -> float:
        """Stima pubblica dello spread da un insieme di seed."""
        rng = random.Random(self._seed)
        activation_prob: dict[tuple[int, int], float] = {}
        for u, v in G.edges():
            activation_prob[(u, v)] = 1.0 / max(G.degree(v), 1)
            activation_prob[(v, u)] = 1.0 / max(G.degree(u), 1)
            
        return self._simulate_spread(
            G, seeds, agent_states or {}, self._n_rounds, activation_prob, rng
        )
