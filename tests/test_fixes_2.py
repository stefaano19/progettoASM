"""
tests/test_fixes_2.py
=====================
Test di regressione per la seconda revisione:

  2. Rewiring: candidati diversi a ogni step; modalita' swap a densita' costante.
  3. Soglie dei nodi indipendenti dall'ordine di interrogazione (resume).
  4. Reach dei fact-checker limitato in salti (non piu' sempre 1.0).
  -  Codifica belief coerente con gli embedding (R ed F opposti a I).
  -  Metriche di echo chamber sulle opinioni.
  -  CELF: obiettivo allineato alla StateMachine, nessun guadagno negativo.
  -  LLM: seed per richiesta passato al backend.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    from src.utils.config import load_config
    shutil.copy(PROJECT_ROOT / "config.yaml", tmp_path / "config.yaml")
    c = load_config(tmp_path / "config.yaml")
    c.project_root = tmp_path
    for attr in ("results", "figures", "logs", "checkpoints"):
        (tmp_path / getattr(c.paths, attr)).mkdir(parents=True, exist_ok=True)
    c.gnn.use_torch = False
    c.influence.simulation_rounds = 10
    return c


# ---------------------------------------------------------------------------
# 2. Rewiring
# ---------------------------------------------------------------------------

def _trainer(cfg):
    from src.gnn.model import GraphSAGEModel
    from src.gnn.trainer import GNNTrainer
    model = GraphSAGEModel(in_dim=16, hidden_dim=16, out_dim=16, seed=0, force_numpy=True)
    return GNNTrainer(model, cfg)


def test_candidates_change_between_steps(cfg):
    G = nx.barabasi_albert_graph(400, 3, seed=0)
    tr = _trainer(cfg)
    c0 = tr._generate_candidates(G, step=0)
    c1 = tr._generate_candidates(G, step=1)
    assert c0 and c1
    assert set(c0) != set(c1)
    # Riproducibile a parita' di step
    assert tr._generate_candidates(G, step=0) == c0
    # Nessun candidato e' gia' un arco
    assert not any(G.has_edge(u, v) for u, v in c0)


def test_rewirer_swap_keeps_edge_count(cfg):
    from src.gnn.rewirer import Rewirer
    cfg.gnn.rewire_mode = "swap"
    cfg.gnn.rewire_threshold_add = 0.5
    cfg.gnn.max_new_edges_per_step = 5
    G = nx.barabasi_albert_graph(100, 3, seed=1)
    scores = {e: 0.9 for e in G.edges()}
    non_edges = [(u, v) for u, v in nx.non_edges(G)][:20]
    scores.update({e: 0.95 for e in non_edges})
    to_add, to_remove = Rewirer(cfg).compute(scores, G, {})
    assert len(to_add) == 5
    assert len(to_remove) == len(to_add)


def test_rewirer_threshold_mode_unchanged(cfg):
    from src.gnn.rewirer import Rewirer
    cfg.gnn.rewire_mode = "threshold"
    cfg.gnn.rewire_threshold_add = 0.5
    G = nx.barabasi_albert_graph(100, 3, seed=1)
    scores = {e: 0.9 for e in G.edges()}
    scores[(0, 99)] = 0.95 if not G.has_edge(0, 99) else 0.9
    _, to_remove = Rewirer(cfg).compute(scores, G, {})
    assert to_remove == []          # nessun arco sotto soglia -> nessuna rimozione


# ---------------------------------------------------------------------------
# 3. Soglie stabili tra sessioni
# ---------------------------------------------------------------------------

def test_thresholds_independent_of_query_order():
    from src.agents.state_machine import StateMachine
    a, b = StateMachine(seed=7), StateMachine(seed=7)
    nodes = list(range(50))
    ta = {n: a.get_threshold(n) for n in nodes}
    tb = {n: b.get_threshold(n) for n in reversed(nodes)}
    assert ta == tb
    assert len(set(ta.values())) > 1     # soglie eterogenee


# ---------------------------------------------------------------------------
# 4. Reach dei fact-checker
# ---------------------------------------------------------------------------

def test_fc_reach_is_local():
    from src.influence.metrics import compute_fact_checker_spread
    G = nx.path_graph(10)                       # connesso
    states = {n: "S" for n in G.nodes()}
    states[0] = "F"
    r = compute_fact_checker_spread(G, states, max_hops=1)
    assert r["total_reachable"] == 1           # solo il vicino 1
    assert r["fcs"] < 1.0
    r2 = compute_fact_checker_spread(G, states, max_hops=2)
    assert r2["total_reachable"] == 2


def test_fc_effective_coverage_matches_rules():
    from src.influence.metrics import compute_fact_checker_spread
    G = nx.star_graph(4)                        # centro 0, foglie 1..4
    states = {n: "S" for n in G.nodes()}
    states[1] = "F"
    # Il centro ha 1 F su 4 vicini (0.25 >= 0.10) e nessun I -> coperto
    r = compute_fact_checker_spread(G, states, protection_threshold=0.10)
    assert r["fc_effective_nodes"] == 1


# ---------------------------------------------------------------------------
# Codifica belief e metriche di opinione
# ---------------------------------------------------------------------------

def test_belief_encoding_r_opposite_to_i():
    from src.graph.network_manager import STATE_TO_BELIEF
    assert STATE_TO_BELIEF["I"] > STATE_TO_BELIEF["S"] > STATE_TO_BELIEF["R"]
    assert STATE_TO_BELIEF["R"] == STATE_TO_BELIEF["F"] == -STATE_TO_BELIEF["I"]


def test_opinion_echo_metrics():
    from src.graph.metrics import compute_opinion_echo_metrics
    # Due cricche omogenee unite da un ponte -> assortativita' alta
    G = nx.barbell_graph(5, 0)
    belief = {n: (1.0 if n < 5 else -1.0) for n in G.nodes()}
    m = compute_opinion_echo_metrics(G, belief)
    assert m["belief_assortativity"] > 0.8
    assert m["opinion_homophily"] > 0.9
    # Opinioni alternate su un ciclo pari -> assortativita' negativa
    C = nx.cycle_graph(10)
    alt = {n: (1.0 if n % 2 else -1.0) for n in C.nodes()}
    assert compute_opinion_echo_metrics(C, alt)["belief_assortativity"] < 0


# ---------------------------------------------------------------------------
# CELF
# ---------------------------------------------------------------------------

def test_celf_threshold_objective(cfg):
    from src.influence.celf import CELF
    cfg.influence.celf_objective = "threshold"
    G = nx.barabasi_albert_graph(200, 2, seed=4)
    rng = np.random.default_rng(0)
    states = {n: ("I" if rng.random() < 0.3 else "S") for n in G.nodes()}
    celf = CELF(cfg)
    seeds = celf.select(G, budget_k=5, agent_states=states)
    assert len(seeds) == 5 and len(set(seeds)) == 5
    assert all(states[s] == "S" for s in seeds)
    assert seeds == CELF(cfg).select(G, budget_k=5, agent_states=states)
    # Obiettivo monotono lungo la selezione greedy
    vals = [celf.estimate_spread(G, seeds[:i], states) for i in range(6)]
    assert vals[-1] > 0
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_celf_ic_no_negative_gains(cfg, caplog):
    import logging
    from src.influence.celf import CELF
    cfg.influence.celf_objective = "ic"
    G = nx.barabasi_albert_graph(80, 2, seed=5)
    states = {n: "S" for n in G.nodes()}
    with caplog.at_level(logging.INFO, logger="src.influence.celf"):
        CELF(cfg).select(G, budget_k=6, agent_states=states)
    gains = [float(r.getMessage().split("marginal spread = ")[1].rstrip(")"))
             for r in caplog.records if "marginal spread" in r.getMessage()]
    assert gains and min(gains) >= 0.0


# ---------------------------------------------------------------------------
# LLM seed
# ---------------------------------------------------------------------------

def test_llm_seed_passed_to_backend(tmp_path, monkeypatch):
    from src.agents import llm_client as lc

    seen = []

    class FakeBackend:
        def chat(self, messages, seed=None):
            seen.append(seed)
            return lc.LLMResponse(content='{"proposed_state": "S"}', input_tokens=1,
                                  output_tokens=1, model="fake", latency_s=0.0)

    client = lc.LLMClient.__new__(lc.LLMClient)
    client._max_retries = 1
    client._backend = FakeBackend()

    class NoCache:
        def get(self, k): return None
        def set(self, k, v): pass
    client._cache = NoCache()

    msgs = [{"role": "user", "content": "ciao"}]
    client.chat(msgs)
    client.chat(msgs)
    client.chat([{"role": "user", "content": "altro prompt"}])
    assert seen[0] is not None and seen[0] == seen[1] and seen[0] != seen[2]
