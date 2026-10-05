"""
tests/test_fixes_3.py
=====================
Test di regressione per la terza revisione:

  1. System prompt ricostruito a ogni cambio di stato (agenti e fact-checker).
  2. Etichetta di influenza dai percentili del grado.
  3. Link predictor: score non piu' bloccati in [0.50, 0.73]; seed torch.
  4. Smart Cache disattivata di default (non congela gli agenti).
  5. Post etichettati con lo stato dopo la transizione.
  6. Risposte di fallback contate nel controllo di disponibilita' dell'LLM.
  -  Cache LLM su disco: niente mock, chiave con modello, file per progetto.
  -  Feed: a parita' di step l'ordine non e' quello di adiacenza.
  -  Timeout delle richieste LLM.
  -  Ricaduta R -> I solo se l'LLM propone I; soglie da config.
  -  Parsing robusto dell'output LLM.
  -  Fase 3: resume da uno step preciso; stato di Adam nei checkpoint.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import types
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
    c.simulation.activation_probability = 1.0
    c.simulation.checkpoint_every = 1
    c.influence.simulation_rounds = 5
    return c


@pytest.fixture
def quiet():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class ScriptedLLM:
    """Client finto: restituisce sempre lo stesso JSON e conta le chiamate."""

    def __init__(self, payload: dict, is_fallback: bool = False):
        self.payload = payload
        self.is_fallback = is_fallback
        self.calls: list[list[dict]] = []

    def chat(self, messages):
        from src.agents.llm_client import LLMResponse
        self.calls.append(messages)
        return LLMResponse(content=json.dumps(self.payload), input_tokens=1,
                           output_tokens=1, model="scripted",
                           is_fallback=self.is_fallback)


def _small_world(cfg, payload, initial_states=None, is_fallback=False):
    """Grafo a stella: nodo 0 al centro, vicini 1..4. Ritorna (nm, agents, llm)."""
    from src.agents.agent import Agent
    from src.agents.state_machine import StateMachine
    from src.graph.network_manager import NetworkManager

    G = nx.star_graph(4)
    feat = np.zeros((5, cfg.gnn.embedding_dim), dtype=np.float32)
    nm = NetworkManager(G, cfg, community_map={n: 0 for n in G}, node_features=feat)
    for n, s in (initial_states or {}).items():
        nm.set_state(n, s)
    llm = ScriptedLLM(payload, is_fallback=is_fallback)
    sm = StateMachine(seed=0, base_threshold_mean=0.1, base_threshold_std=0.0)
    agents = {}
    for n in G.nodes():
        a = Agent(n, cfg, llm, sm, initial_state=nm.get_state(n))
        a.initialize(community=0, centrality=0.0, network_manager=nm)
        agents[n] = a
    return nm, agents, llm


INFECT = {"reasoning": "r", "opinion": "Ci credo.", "susceptibility": 0.9,
          "proposed_state": "I", "spread_intent": True}


# ---------------------------------------------------------------------------
# 1. System prompt aggiornato
# ---------------------------------------------------------------------------

def test_system_prompt_follows_state_change(cfg, quiet):
    nm, agents, llm = _small_world(cfg, INFECT, initial_states={1: "I", 2: "I", 3: "I", 4: "I"})
    a = agents[0]
    assert "NEUTRAL" in a.system_prompt
    d = a.step(0, nm)
    assert d.state_changed and d.new_state == "I"
    assert "CONVINCED" in a.system_prompt and "NEUTRAL" not in a.system_prompt

    # La chiamata successiva usa il prompt aggiornato, senza note "una tantum"
    a.step(1, nm)
    msgs = llm.calls[-1]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "CONVINCED" in msgs[0]["content"]
    assert "Your current state: I" in msgs[1]["content"]


def test_fact_checker_prompt_persists(cfg, quiet):
    nm, agents, llm = _small_world(cfg, INFECT)
    a = agents[1]
    nm.set_state(1, "F")
    a.set_state("F")
    for t in range(3):
        a.step(t, nm)
        assert "FACT-CHECKER" in llm.calls[-1][0]["content"]


def test_prompt_independent_of_history(cfg, quiet):
    """Stesso stato -> stesso system prompt, comunque ci si sia arrivati (resume)."""
    nm, agents, _ = _small_world(cfg, INFECT)
    a, b = agents[1], agents[2]
    a.set_state("I")
    nm2, agents2, _ = _small_world(cfg, INFECT, initial_states={1: "I"})
    assert a.system_prompt == agents2[1].system_prompt
    assert b.system_prompt != a.system_prompt


# ---------------------------------------------------------------------------
# 2. Influenza dai percentili del grado
# ---------------------------------------------------------------------------

def test_influence_levels_percentiles():
    from src.agents.prompts import influence_levels
    G = nx.barabasi_albert_graph(1000, 2, seed=0)
    levels = influence_levels(dict(G.degree()))
    hub = max(G.degree(), key=lambda x: x[1])[0]
    assert levels[hub] == "high"
    counts = {k: sum(1 for v in levels.values() if v == k) for k in ("high", "medium", "low")}
    assert 0 < counts["high"] <= 0.2 * 1000
    assert counts["medium"] > 0 and counts["low"] > 0
    # Grafo regolare: nessuno e' un hub
    reg = influence_levels({n: 3 for n in range(10)})
    assert set(reg.values()) == {"low"}


def test_influence_passed_to_prompt(cfg):
    from src.agents.prompts import build_system_prompt
    p = build_system_prompt(1, 0, "S", centrality=0.02, cfg=cfg, influence="high")
    assert "High influence" in p
    # Senza etichetta: vecchie soglie assolute (0.02 -> low)
    assert "Low influence" in build_system_prompt(1, 0, "S", centrality=0.02, cfg=cfg)


def test_orchestrator_hubs_get_high_influence(cfg, quiet):
    from src.orchestrator import SimulationOrchestrator
    o = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True)
    deg = dict(o.network_manager.G.degree())
    hub = max(deg, key=deg.get)
    assert "High influence" in o._agents[hub].system_prompt
    o.close()


# ---------------------------------------------------------------------------
# 3. Link predictor
# ---------------------------------------------------------------------------

def test_link_scores_not_stuck_numpy():
    from src.gnn.model import GraphSAGEModel
    G = nx.barabasi_albert_graph(200, 3, seed=0)
    emb = np.random.default_rng(0).standard_normal((200, 16)).astype(np.float32)
    model = GraphSAGEModel(16, 16, 16, seed=0, force_numpy=True)
    out = model.forward(G, emb)
    pairs = np.random.default_rng(1).integers(0, 200, size=(2000, 2))
    scores = np.array([model.link_score(out[u], out[v]) for u, v in pairs if u != v])
    # Prima: tutti in [0.50, 0.73]
    assert scores.min() < 0.35 and scores.max() > 0.73


def test_link_scores_separate_after_training(cfg):
    """Con ReLU + norma 1 nessun training poteva portare uno score fuori da [0.5, 0.73]."""
    from src.gnn.model import GraphSAGEModel, _TORCH_AVAILABLE
    from src.gnn.trainer import GNNTrainer
    if not _TORCH_AVAILABLE:
        pytest.skip("torch non disponibile")
    cfg.gnn.use_torch = True
    cfg.gnn.lr = 0.01
    cfg.gnn.epochs_per_step = 30
    G = nx.barabasi_albert_graph(200, 3, seed=0)
    emb = np.random.default_rng(0).standard_normal((200, 16)).astype(np.float32)
    model = GraphSAGEModel(16, 32, 16, seed=0)
    trainer = GNNTrainer(model, cfg)
    for step in range(5):
        trainer.train_step(G, emb, step=step)
    out = model.forward(G, emb)
    pos = np.array([model.link_score(out[u], out[v]) for u, v in G.edges()])
    rng = np.random.default_rng(1)
    neg = np.array([model.link_score(out[u], out[v])
                    for u, v in rng.integers(0, 200, size=(2000, 2))
                    if u != v and not G.has_edge(u, v)])
    assert neg.min() < 0.35 and pos.max() > 0.73
    assert pos.mean() > neg.mean()


def test_torch_seed_controls_init():
    from src.gnn.model import GraphSAGEModel, _TORCH_AVAILABLE
    if not _TORCH_AVAILABLE:
        pytest.skip("torch non disponibile")
    import torch
    torch.manual_seed(123)
    w1 = GraphSAGEModel(8, 8, 8, seed=7).get_weights()
    torch.manual_seed(999)
    w2 = GraphSAGEModel(8, 8, 8, seed=7).get_weights()
    w3 = GraphSAGEModel(8, 8, 8, seed=8).get_weights()
    assert all(np.array_equal(w1[k], w2[k]) for k in w1)
    assert not np.array_equal(w1["layer1.linear.weight"], w3["layer1.linear.weight"])


def test_torch_old_weights_still_load():
    """Checkpoint senza logit_scale: si caricano mantenendo la scala iniziale."""
    from src.gnn.model import GraphSAGEModel, _TORCH_AVAILABLE
    if not _TORCH_AVAILABLE:
        pytest.skip("torch non disponibile")
    m = GraphSAGEModel(8, 8, 8, seed=0)
    w = m.get_weights()
    w.pop("logit_scale")
    m.set_weights(w)


def test_trainer_optimizer_state_roundtrip(cfg):
    from src.gnn.model import GraphSAGEModel, _TORCH_AVAILABLE
    from src.gnn.trainer import GNNTrainer
    if not _TORCH_AVAILABLE:
        pytest.skip("torch non disponibile")
    cfg.gnn.use_torch = True
    cfg.gnn.epochs_per_step = 2
    G = nx.barabasi_albert_graph(60, 2, seed=0)
    emb = np.random.default_rng(0).standard_normal((60, 16)).astype(np.float32)
    m = GraphSAGEModel(16, 16, 16, seed=0)
    t = GNNTrainer(m, cfg)
    t.train_step(G, emb, step=0)
    state = t.get_optimizer_state()
    assert state and state["state"]
    t2 = GNNTrainer(GraphSAGEModel(16, 16, 16, seed=0), cfg)
    t2.set_optimizer_state(state)
    assert t2._optimizer.state_dict()["state"].keys() == state["state"].keys()


# ---------------------------------------------------------------------------
# 4. Smart Cache
# ---------------------------------------------------------------------------

def test_smart_cache_off_by_default(cfg, quiet):
    stay = {"reasoning": "r", "opinion": "", "susceptibility": 0.6,
            "proposed_state": "S", "spread_intent": True}
    nm, agents, llm = _small_world(cfg, stay, initial_states={1: "I"})
    a = agents[0]
    a.step(0, nm)
    a.step(1, nm)
    assert len(llm.calls) == 2   # nessun riuso: l'LLM viene interrogato a ogni step


def test_smart_cache_opt_in(cfg, quiet):
    cfg.simulation.smart_cache = True
    stay = {"reasoning": "r", "opinion": "", "susceptibility": 0.6,
            "proposed_state": "S", "spread_intent": True}
    nm, agents, llm = _small_world(cfg, stay, initial_states={1: "I"})
    a = agents[0]
    ctx0 = a.prepare_step(0, nm)
    a.finalize_step(ctx0, llm.chat(ctx0.messages), nm)
    ctx1 = a.prepare_step(1, nm)
    assert ctx1.cached_response is not None


# ---------------------------------------------------------------------------
# 5. Etichetta del post
# ---------------------------------------------------------------------------

def test_post_tagged_with_new_state(cfg, quiet):
    nm, agents, _ = _small_world(cfg, INFECT, initial_states={1: "I", 2: "I", 3: "I", 4: "I"})
    d = agents[0].step(0, nm)
    assert d.new_state == "I"
    assert nm._post_store[0][-1]["author_state"] == "I"


# ---------------------------------------------------------------------------
# 6. Fallback contati come fallimenti
# ---------------------------------------------------------------------------

def test_step_aborts_on_fallback_responses(cfg, quiet):
    from src.agents.llm_client import FALLBACK_AGENT_OUTPUT, LLMResponse
    from src.orchestrator import SimulationOrchestrator
    o = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True)

    def fallback(*args, **kwargs):
        return LLMResponse(content=json.dumps(FALLBACK_AGENT_OUTPUT), is_fallback=True)

    for agent in o._agents.values():
        agent.llm_client.chat = fallback
    before = o.network_manager.get_all_states()
    with pytest.raises(RuntimeError, match="fallback"):
        o._run_step(0)
    assert o.network_manager.get_all_states() == before
    assert o.next_step == 0
    o.close()


def test_llm_client_down_returns_fallback_and_orchestrator_stops(cfg, quiet, monkeypatch):
    """Server spento: LLMClient non solleva, ma lo step deve fermarsi lo stesso."""
    from src.agents import llm_client as lc
    from src.orchestrator import SimulationOrchestrator
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)

    class DownBackend:
        def chat(self, messages, seed=None):
            raise ConnectionError("Connection refused")

    client = lc.LLMClient.__new__(lc.LLMClient)
    client._max_retries = 2
    client._backend = DownBackend()
    client._cache = lc.LLMDiskCache(cfg.project_root / "cache_down")
    client._cache_namespace = "down"
    assert client.chat([{"role": "user", "content": "x"}]).is_fallback

    o = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True)
    for agent in o._agents.values():
        agent.llm_client.chat = client.chat
    with pytest.raises(RuntimeError, match="LLM"):
        o._run_step(0)
    o.close()


def test_llm_stats_in_metrics(cfg, quiet):
    from src.orchestrator import SimulationOrchestrator
    o = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True)
    m = o._run_step(0)
    assert m["llm_fallbacks"] == 0 and m["llm_calls"] > 0
    assert 0.0 <= m["gnn_score_min"] <= m["gnn_score_max"] <= 1.0
    o.close()


# ---------------------------------------------------------------------------
# Cache LLM su disco
# ---------------------------------------------------------------------------

def test_mock_does_not_touch_disk_cache(tmp_path, monkeypatch):
    from src.agents.llm_client import MockLLMClient
    monkeypatch.chdir(tmp_path)
    MockLLMClient(seed=1).chat([{"role": "user", "content": "x"}])
    assert not (tmp_path / "results").exists()


def test_disk_cache_key_includes_model(tmp_path):
    from src.agents import llm_client as lc

    class Backend:
        def __init__(self, text):
            self.text = text
            self.n = 0

        def chat(self, messages, seed=None):
            self.n += 1
            return lc.LLMResponse(content=json.dumps({"opinion": self.text}),
                                  input_tokens=1, output_tokens=1, model=self.text)

    def make(model, backend):
        c = lc.LLMClient.__new__(lc.LLMClient)
        c._max_retries = 1
        c._backend = backend
        c._cache = lc.LLMDiskCache(tmp_path)
        c._cache_namespace = lc.LLMClient.make_cache_namespace("local", {"model": model})
        return c

    msgs = [{"role": "user", "content": "stesso prompt"}]
    b1, b2 = Backend("A"), Backend("B")
    assert "A" in make("model-a", b1).chat(msgs).content
    assert "B" in make("model-b", b2).chat(msgs).content   # nessun riuso fra modelli
    assert "A" in make("model-a", b1).chat(msgs).content
    assert b1.n == 1                                         # stesso modello: cache


def test_disk_cache_one_instance_per_dir(tmp_path):
    from src.agents.llm_client import LLMDiskCache
    a, b = LLMDiskCache(tmp_path / "a"), LLMDiskCache(tmp_path / "b")
    assert a is not b and a is LLMDiskCache(tmp_path / "a")
    assert a.cache_file.parent == (tmp_path / "a").resolve()


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------

def test_feed_order_not_adjacency(cfg):
    from src.graph.network_manager import NetworkManager
    G = nx.star_graph(30)                  # nodo 0 con 30 vicini
    nm = NetworkManager(G, cfg, node_features=np.zeros((31, cfg.gnn.embedding_dim), np.float32))
    for nb in range(1, 31):
        nm.add_post(nb, {"node_id": nb, "step": 5, "content": f"p{nb}"})
    unseeded = [p["node_id"] for p in nm.get_feed(0, window=3)[:9]]
    assert unseeded == list(range(1, 10))  # vecchio comportamento: sempre i primi vicini
    seen = set()
    for step in range(20):
        seen |= {p["node_id"] for p in nm.get_feed(0, window=3, seed=step)[:9]}
    assert len(seen) > 20                  # col seed ogni vicino puo' comparire
    assert nm.get_feed(0, window=3, seed=4) == nm.get_feed(0, window=3, seed=4)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

def test_openai_backend_passes_timeout(monkeypatch):
    seen = {}

    class FakeCompletions:
        def create(self, **kw):
            seen.update(kw)
            msg = types.SimpleNamespace(content="{}")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                         usage=None, model="m")

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    from src.agents.llm_client import _OpenAICompatibleBackend
    _OpenAICompatibleBackend({"model": "m", "timeout": 42}, "local").chat([])
    assert seen["timeout"] == 42
    _OpenAICompatibleBackend({"model": "m"}, "local").chat([])
    assert seen["timeout"] is not None


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

def test_relapse_requires_proposed_i():
    from src.agents.state_machine import StateMachine
    sm = StateMachine(seed=0)
    nb = {"I": 9, "S": 1, "R": 0, "F": 0}
    stay = sm.transition("R", 1, nb, {"susceptibility": 0.95, "proposed_state": "R"})
    assert stay.new_state.value == "R"
    go = sm.transition("R", 1, nb, {"susceptibility": 0.95, "proposed_state": "I"})
    assert go.new_state.value == "I"


def test_state_machine_thresholds_from_config(cfg):
    from src.agents.state_machine import StateMachine
    cfg.simulation.relapse_threshold = 0.95
    cfg.simulation.base_threshold_mean = 0.7
    cfg.simulation.base_threshold_std = 0.0
    sm = StateMachine.from_config(cfg)
    assert sm.get_threshold(3) == pytest.approx(0.7)
    r = sm.transition("R", 1, {"I": 9, "S": 1}, {"susceptibility": 0.95, "proposed_state": "I"})
    assert r.new_state.value == "R"          # 0.9 < 0.95


def test_spread_intent_string_false():
    from src.agents.state_machine import StateMachine
    sm = StateMachine(seed=0, base_threshold_mean=0.9, base_threshold_std=0.0)
    nb = {"I": 2, "S": 8, "R": 0, "F": 0}
    out = {"susceptibility": 0.1, "proposed_state": "S", "spread_intent": "false"}
    # Prima bool("false") == True bloccava la transizione S -> R
    assert sm.transition("S", 1, nb, out).new_state.value == "R"


# ---------------------------------------------------------------------------
# Parsing output LLM
# ---------------------------------------------------------------------------

def test_parse_helpers():
    from src.agents.llm_client import extract_json, parse_bool, parse_float
    assert parse_bool("false") is False and parse_bool("True") is True
    assert parse_bool(None) is False and parse_bool(1) is True
    assert parse_float("alta") == 0.5 and parse_float("0.8") == 0.8
    assert parse_float(7) == 1.0 and parse_float(float("nan")) == 0.5
    obj, fb = extract_json("[1, 2, 3]")
    assert fb and isinstance(obj, dict)
    obj, fb = extract_json('Ecco: {"opinion": "x"}')
    assert not fb and obj["opinion"] == "x"


def test_finalize_with_bad_fields(cfg, quiet):
    bad = {"reasoning": "r", "opinion": None, "susceptibility": "molto alta",
           "proposed_state": "I", "spread_intent": "yes"}
    nm, agents, _ = _small_world(cfg, bad, initial_states={1: "I"})
    d = agents[0].step(0, nm)
    assert d.susceptibility == 0.5 and d.spread_intent is True and d.opinion == ""


# ---------------------------------------------------------------------------
# Config, resume, checkpoint
# ---------------------------------------------------------------------------

def test_effective_hash_sees_overrides(cfg):
    h1 = cfg.effective_hash()
    cfg.simulation.activation_probability = 0.33
    assert cfg.effective_hash() != h1


def test_resume_from_specific_step(cfg, quiet):
    from src.orchestrator import SimulationOrchestrator
    o1 = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True)
    o1.run(n_steps=3)
    o1.close()
    o2 = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True, resume=True,
                                                  resume_from_step=1)
    assert o2.next_step == 2
    o2.close()


def test_checkpoint_meta_has_optimizer_and_hash(cfg, quiet):
    from src.orchestrator import SimulationOrchestrator
    from src.utils.checkpoint import CheckpointManager
    o = SimulationOrchestrator.build_from_config(cfg, use_mock_llm=True)
    o.run(n_steps=1)
    o.close()
    meta = CheckpointManager(cfg).load_latest().meta
    assert "optimizer_state" in meta
    assert meta["effective_config_hash"] == cfg.effective_hash()
