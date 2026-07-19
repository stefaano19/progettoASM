"""
src/influence/celf.py
=====================
CELF (Cost-Effective Lazy Forward) for Influence Maximization.

Algorithm
---------
CELF is an optimization of the greedy algorithm for Influence Maximization
that exploits the submodularity property of the spread function to reduce the
number of costly Monte Carlo simulations.

Instead of re-evaluating the marginal gain of every candidate at each iteration,
CELF maintains a priority queue sorted by the upper bound of marginal gains
and re-evaluates only when necessary (lazy evaluation).

Complexity:
    Average case: O(k * R * (n + m)) vs. O(k * n * R * (n + m)) of the pure greedy algorithm,
    where k is the budget size, R is the number of Monte Carlo rounds, n is the number of nodes,
    and m is the number of edges.

Diffusion Model
---------------
Uses the Independent Cascade (IC) model:
  - Each edge (u, v) has an activation probability inversely proportional to the degree of v.
  - Starts from a seed set S and propagates until no new activations occur.
"""

from __future__ import annotations

import heapq
import logging
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class CELF:
    """
    Cost-Effective Lazy Forward solver for Influence Maximization.

    Parameters
    ----------
    cfg : Config
        Global configuration (uses cfg.influence.* and cfg.execution.random_seed).
    """

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        self._random_seed = cfg.execution.random_seed
        self._simulation_rounds = cfg.influence.simulation_rounds

    def select(
        self,
        graph: "nx.Graph",
        budget_k: int | None = None,
        agent_states: dict[int, str] | None = None,
        seed_state: str = "F",
    ) -> list[int]:
        """
        Selects the optimal budget_k seed nodes for fact-checking.

        Automatically excludes nodes that are already infected ('I') or are
        already fact-checkers as seed candidates.

        Parameters
        ----------
        graph : nx.Graph
            The current network graph.
        budget_k : int | None
            Number of seed nodes to select. Defaults to cfg.influence.budget_k.
        agent_states : dict[int, str] | None
            Current states of the agents {node_id: "S"|"I"|"R"|"F"}.
        seed_state : str
            State to assign to the selected seeds (default "F" = Fact-Checker).

        Returns
        -------
        list[int]
            List of budget_k node IDs selected as seeds.
        """
        target_seed_count = budget_k if budget_k is not None else self._cfg.influence.budget_k
        current_states = agent_states or {}

        # Candidates: exclude already infected nodes or existing fact-checkers
        excluded_states = {"I", seed_state}
        candidates = [
            node_id for node_id in graph.nodes()
            if current_states.get(node_id, "S") not in excluded_states
        ]

        if not candidates:
            logger.warning("[CELF] No candidate nodes available (all are infected or fact-checkers).")
            return []

        target_seed_count = min(target_seed_count, len(candidates))
        logger.info(
            "[CELF] Starting seed selection: target_seeds=%d | candidates=%d | simulation_rounds=%d",
            target_seed_count, len(candidates), self._simulation_rounds,
        )

        selected_seeds: list[int] = []
        
        # Max-heap simulated via min-heap with negated values: (-marginal_gain, last_updated_iteration, node_id)
        marginal_gain_heap: list[tuple[float, int, int]] = []

        # Precompute edge activation probabilities once
        logger.info("[CELF] Pre-computing edge activation probabilities...")
        edges = list(graph.edges())
        activation_probabilities: dict[tuple[int, int], float] = {}
        if edges:
            # Vectorized degree dictionary construction for O(1) degree lookups
            node_degrees = dict(graph.degree())
            for source_node, target_node in edges:
                activation_probabilities[(source_node, target_node)] = 1.0 / max(node_degrees[target_node], 1)
                activation_probabilities[(target_node, source_node)] = 1.0 / max(node_degrees[source_node], 1)

        # Compute initial marginal gain for all candidate nodes in parallel
        logger.info("[CELF] Computing initial spread for all candidates via ProcessPoolExecutor...")
        
        num_cores = max(1, mp.cpu_count() - 1)
        chunk_size = max(1, len(candidates) // num_cores) if candidates else 1
        candidate_chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
        
        # Use ProcessPoolExecutor to bypass Python's GIL since IC propagation simulation is CPU-bound.
        # Use 'spawn' start method to prevent potential CUDA context initialization issues on Kaggle GPUs.
        multiprocessing_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_cores, mp_context=multiprocessing_context) as executor:
            futures = [
                executor.submit(
                    _parallel_evaluate_spread_chunk,
                    (graph, chunk, current_states, self._simulation_rounds, activation_probabilities)
                )
                for chunk in candidate_chunks
            ]
            for future in futures:
                for candidate_node, initial_gain in future.result():
                    heapq.heappush(marginal_gain_heap, (-initial_gain, 0, candidate_node))

        # Lazy greedy selection loop
        for iteration_idx in range(target_seed_count):
            if not marginal_gain_heap:
                break

            while True:
                negative_marginal_gain, last_updated_iteration, current_node = heapq.heappop(marginal_gain_heap)

                if last_updated_iteration == iteration_idx:
                    # The gain is fresh for this iteration: select the node
                    selected_seeds.append(current_node)
                    logger.info(
                        "[CELF] Iteration %d/%d: Selected node %d (marginal spread = %.2f)",
                        iteration_idx + 1, target_seed_count, current_node, -negative_marginal_gain,
                    )
                    break
                else:
                    # Re-evaluate the marginal gain relative to the currently selected seeds
                    base_spread = self._simulate_spread(
                        graph=graph,
                        seeds=selected_seeds,
                        agent_states=current_states,
                        simulation_rounds=self._simulation_rounds,
                        activation_probabilities=activation_probabilities,
                    )
                    candidate_spread = self._simulate_spread(
                        graph=graph,
                        seeds=selected_seeds + [current_node],
                        agent_states=current_states,
                        simulation_rounds=self._simulation_rounds,
                        activation_probabilities=activation_probabilities,
                    )
                    marginal_gain = candidate_spread - base_spread
                    heapq.heappush(marginal_gain_heap, (-marginal_gain, iteration_idx, current_node))

        logger.info("[CELF] Selected seeds: %s", selected_seeds)
        return selected_seeds

    @staticmethod
    def _simulate_spread(
        graph: "nx.Graph",
        seeds: list[int],
        agent_states: dict[int, str],
        simulation_rounds: int,
        activation_probabilities: dict[tuple[int, int], float],
        random_generator: random.Random | None = None,
    ) -> float:
        """
        Estimates the expected number of nodes reached by cascading activation
        starting from the seeds using Monte Carlo simulation under the Independent Cascade (IC) model.

        Optimizations:
            1. Uses sets for O(1) membership checks of activated nodes.
            2. Pre-caches neighbor lists to avoid networkx G.neighbors overhead.
        """
        if not seeds:
            return 0.0

        rng = random_generator or random.Random(42)
        total_activated_nodes = 0

        # Precompute neighbor lists for seed nodes and their neighbors once
        neighbors_cache: dict[int, list[int]] = {node: list(graph.neighbors(node)) for node in seeds}
        for seed_node in seeds:
            for neighbor_node in neighbors_cache[seed_node]:
                if neighbor_node not in neighbors_cache:
                    neighbors_cache[neighbor_node] = list(graph.neighbors(neighbor_node))

        for _ in range(simulation_rounds):
            activated_nodes: set[int] = set(seeds)
            active_frontier: list[int] = list(seeds)

            while active_frontier:
                next_active_frontier: list[int] = []
                for active_node in active_frontier:
                    # Fall back to live neighbors query if active_node not cached
                    neighbors = neighbors_cache.get(active_node) or list(graph.neighbors(active_node))
                    for neighbor_node in neighbors:
                        if neighbor_node not in activated_nodes:
                            activation_probability = activation_probabilities.get((active_node, neighbor_node), 0.1)
                            neighbor_state = agent_states.get(neighbor_node, "S")
                            
                            # Scaling rule: Resistant nodes ('R') are harder to activate (30% probability)
                            if neighbor_state == "R":
                                activation_probability *= 0.3
                            # Isolation rule: Fact-checkers ('F') block influence propagation completely
                            elif neighbor_state == "F":
                                activation_probability = 0.0
                                
                            if activation_probability > 0.0 and rng.random() < activation_probability:
                                activated_nodes.add(neighbor_node)
                                next_active_frontier.append(neighbor_node)
                active_frontier = next_active_frontier

            total_activated_nodes += len(activated_nodes)

        return total_activated_nodes / simulation_rounds

    def estimate_spread(
        self,
        graph: "nx.Graph",
        seeds: list[int],
        agent_states: dict[int, str] | None = None,
    ) -> float:
        """Estimates the expected influence spread starting from a set of seed nodes."""
        random_generator = random.Random(self._random_seed)
        activation_probabilities: dict[tuple[int, int], float] = {}
        for source_node, target_node in graph.edges():
            activation_probabilities[(source_node, target_node)] = 1.0 / max(graph.degree(target_node), 1)
            activation_probabilities[(target_node, source_node)] = 1.0 / max(graph.degree(source_node), 1)
            
        return self._simulate_spread(
            graph=graph,
            seeds=seeds,
            agent_states=agent_states or {},
            simulation_rounds=self._simulation_rounds,
            activation_probabilities=activation_probabilities,
            random_generator=random_generator,
        )


def _parallel_evaluate_spread_chunk(
    args: tuple[nx.Graph, list[int], dict[int, str], int, dict[tuple[int, int], float]]
) -> list[tuple[int, float]]:
    """
    Top-level helper function to evaluate candidate nodes in parallel.
    Bypasses the Global Interpreter Lock (GIL) via ProcessPoolExecutor.
    """
    import random
    
    graph, candidates, agent_states, simulation_rounds, activation_probabilities = args
    random_generator = random.Random()
    results = []
    for candidate_node in candidates:
        gain = CELF._simulate_spread(
            graph=graph,
            seeds=[candidate_node],
            agent_states=agent_states,
            simulation_rounds=simulation_rounds,
            activation_probabilities=activation_probabilities,
            random_generator=random_generator
        )
        results.append((candidate_node, gain))
    return results
