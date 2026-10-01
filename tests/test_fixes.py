"""
tests/test_fixes.py
===================
Test di regressione per le correzioni della revisione:

  1. Resume: stato interno degli Agent allineato al NetworkManager ripristinato.
  2. Belief Polarisation: stessa codifica prima/dopo l'intervento.
  3. Checkpoint: ordinamento/pruning per step numerico.
  4. Direzione di perturbazione embedding condivisa fra agenti.
  5. Fact-checker: inoculazione S -> R.
  6. Rewirer: piu' rimozioni nello stesso step non isolano un nodo.
  7. CELF: selezione riproducibile.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    """Config isolata: tutti gli output in una directory temporanea."""
    from src.utils.config import load_config
    shutil.copy(PROJECT_ROOT / "config.yaml", tmp_path / "config.yaml")
    c = load_config(tmp_path / "config.yaml")
    c.project_root = tmp_path
    for attr in ("results", "figures", "logs", "checkpoints"):
        (tmp_path / getattr(c.paths, attr)).mkdir(parents=True, exist_ok=True)
    c.gnn.use_torch = False
    c.simulation.activation_probability = 1.0
    c.simulation.checkpoint_every = 1
    c.influence.simulation_rounds = 5
    return c


# ---------------------------------------------------------------------------
# 1. Resume
# ---------------------------------------------------------------------------

def test_resume_agents_match_network_manager(cfg):
    from src.orchestrator import SimulationOrchestrator
    logging.disable(logging.CRITICAL)
    try:
        o1 = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True)
        o1.run(n_steps=4)
        o1.close()
        states_saved = o1.network_manager.get_all_states()

        o2 = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True, resume=True)
        assert o2.next_step == 4
        nm_states = o2.network_manager.get_all_states()
        assert nm_states == states_saved
        mismatched = [n for n, a in o2._agents.items() if a.state != nm_states[n]]
        assert mismatched == []

        o2.run(n_steps=2)          # deve partire da 4, non da 0
        assert o2.next_step == 6
        o2.close()
    finally:
        logging.disable(logging.NOTSET)


def test_sync_agent_states_after_injection(cfg):
    from src.orchestrator import SimulationOrchestrator
    from src.influence.injector import FactCheckerInjector
    logging.disable(logging.CRITICAL)
    try:
        o = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True)
        s_nodes = [n for n, s in o.network_manager.get_all_states().items() if s == "S"][:3]
        injected = FactCheckerInjector(cfg).inject(o.network_manager, s_nodes)
        assert o.sync_agent_states(injected) == len(injected)
        assert all(o._agents[n].state == "F" for n in injected)
        o.close()
    finally:
        logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# 2. Belief Polarisation coerente
# ---------------------------------------------------------------------------

def test_belief_polarisation_same_encoding_before_after(cfg):
    from src.graph.metrics import compute_all_metrics
    from src.graph.network_manager import NetworkManager
    from src.influence.metrics import compute_full_influence_report

    G = nx.barabasi_albert_graph(60, 2, seed=1)
    cm = {n: n % 3 for n in G.nodes()}
    nm = NetworkManager(G, cfg, community_map=cm)
    for n in G.nodes():
        nm.set_state(n, "SIRF"[n % 4])

    before = compute_all_metrics(G, cfg, cm, nm.get_belief_map())
    after = compute_full_influence_report(
        G=G, agent_states=nm.get_all_states(), community_map=cm,
        baseline_metrics=before, cfg=cfg,
    )
    # Stessi stati -> nessuna variazione artificiale
    assert after["belief_polarisation"] == pytest.approx(before["belief_polarisation"])
    assert after.get("delta_belief_polarisation", 0.0) == pytest.approx(0.0)
    assert 0.0 <= before["belief_polarisation"] <= 1.0


# ---------------------------------------------------------------------------
# 3. Checkpoint numerici
# ---------------------------------------------------------------------------

def test_checkpoint_numeric_ordering(cfg):
    from src.utils.checkpoint import CheckpointManager
    ck = CheckpointManager(cfg, keep_last=2)
    d = ck._checkpoint_dir
    for step in (9998, 9999, 10000):
        (d / ck.FILENAME_PATTERN.format(step=step)).write_bytes(b"x")
    assert [s for s, _ in ck.list_checkpoints()] == [9998, 9999, 10000]
    ck._prune_old_checkpoints()
    assert [s for s, _ in ck.list_checkpoints()] == [9999, 10000]


# ---------------------------------------------------------------------------
# 4. Direzione embedding condivisa
# ---------------------------------------------------------------------------

def test_infection_direction_shared():
    from src.agents.agent import Agent
    dirs = np.stack([Agent._build_direction(42, n, 128) for n in range(20)])
    sims = dirs @ dirs.T
    assert sims[np.triu_indices(20, 1)].min() > 0.9


# ---------------------------------------------------------------------------
# 5. Inoculazione fact-check
# ---------------------------------------------------------------------------

def test_fact_check_inoculation():
    from src.agents.state_machine import StateMachine
    sm = StateMachine(seed=0, fc_protection_threshold=0.1)
    r = sm.transition("S", 1, {"S": 8, "I": 0, "R": 0, "F": 2},
                      {"susceptibility": 0.5, "proposed_state": "S"})
    assert r.new_state.value == "R"
    # Senza F il comportamento originale non cambia
    r = sm.transition("S", 1, {"S": 10, "I": 0, "R": 0, "F": 0},
                      {"susceptibility": 0.5, "proposed_state": "S"})
    assert r.new_state.value == "S"


# ---------------------------------------------------------------------------
# 6. Rewirer non isola nodi
# ---------------------------------------------------------------------------

def test_rewirer_does_not_isolate(cfg):
    from src.gnn.rewirer import Rewirer
    G = nx.Graph([(0, 1), (0, 2), (3, 4)])
    _, to_remove = Rewirer(cfg).compute({(0, 1): 0.0, (0, 2): 0.0}, G, {})
    H = G.copy()
    H.remove_edges_from(to_remove)
    assert all(H.degree(n) >= 1 for n in (0, 1, 2) if G.degree(n) > 1)
    assert H.degree(0) >= 1


# ---------------------------------------------------------------------------
# 7. CELF riproducibile
# ---------------------------------------------------------------------------

def test_celf_reproducible(cfg):
    from src.influence.celf import CELF
    G = nx.barabasi_albert_graph(80, 2, seed=3)
    states = {n: "S" for n in G.nodes()}
    a = CELF(cfg).select(G, budget_k=4, agent_states=states)
    b = CELF(cfg).select(G, budget_k=4, agent_states=states)
    assert a == b and len(a) == 4
