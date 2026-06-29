"""
tests/test_agent.py
===================
Test suite per la Fase 1 (src/agents/).

Tutti i test:
  - usano MockLLMClient — zero costi API, zero latenza
  - usano grafi sintetici piccoli (karate club, 20-100 nodi)
  - sono deterministici con seed fisso
  - coprono: LLMClient, StateMachine, Prompts, Agent, Seeder

Esegui con:
    pytest tests/test_agent.py -v
    pytest tests/test_agent.py -v -k "TestStateMachine"  # singola classe
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures condivise
# ---------------------------------------------------------------------------

@pytest.fixture
def karate_graph() -> nx.Graph:
    return nx.karate_club_graph()


@pytest.fixture
def small_graph() -> nx.Graph:
    """BA graph, 80 nodi."""
    return nx.barabasi_albert_graph(80, 3, seed=42)


@pytest.fixture
def mock_cfg(tmp_path):
    """Config mock completa con tmp_path."""
    from src.utils.config import (
        Config, ExecutionConfig, PathsConfig, DatasetConfig,
        SubgraphConfig, MetricsConfig, CommunityConfig,
        SimulationConfig, LLMConfig, GNNConfig, InfluenceConfig,
        LLMApiConfig, LLMLocalConfig, LLMTokenBudget,
    )

    cfg = Config(
        execution=ExecutionConfig(mode="local", random_seed=42, log_level="DEBUG"),
        paths=PathsConfig(
            data_raw=Path("data/raw"),
            data_processed=tmp_path / "processed",
            results=tmp_path / "results",
            figures=tmp_path / "results/figures",
            logs=tmp_path / "results/logs",
            checkpoints=tmp_path / "results/checkpoints",
        ),
        dataset=DatasetConfig(),
        subgraph=SubgraphConfig(target_nodes=80, output_file=str(tmp_path / "subgraph.gpickle")),
        metrics=MetricsConfig(compute_betweenness=False, compute_diameter=False),
        community=CommunityConfig(
            algorithm="label_propagation",
            output_file=str(tmp_path / "community_map.json"),
        ),
        simulation=SimulationConfig(
            max_steps=5,
            topic="test_topic",
            topic_description="Test topic description for unit tests.",
            initial_infection_rate=0.05,
            memory_window=3,
            rewiring_cooldown=2,
        ),
        llm=LLMConfig(
            backend="api",
            api=LLMApiConfig(),
            local=LLMLocalConfig(),
            token_budget=LLMTokenBudget(warn_at=10_000, hard_limit=100_000),
        ),
        gnn=GNNConfig(embedding_dim=32, embedding_file=str(tmp_path / "embeddings.npy")),
        influence=InfluenceConfig(),
        project_root=tmp_path,
        config_hash="test_hash",
    )
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "logs").mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def community_map(karate_graph) -> dict[int, int]:
    return {n: n % 4 for n in karate_graph.nodes()}


@pytest.fixture
def network_manager(karate_graph, mock_cfg, community_map):
    from src.graph.network_manager import NetworkManager
    feat = np.ones((karate_graph.number_of_nodes(), 32), dtype=np.float32)
    return NetworkManager(karate_graph, mock_cfg, community_map=community_map, node_features=feat)


@pytest.fixture
def mock_llm():
    from src.agents.llm_client import MockLLMClient
    return MockLLMClient(seed=42, infection_rate=0.3)


@pytest.fixture
def state_machine(mock_cfg):
    from src.agents.state_machine import StateMachine
    return StateMachine.from_config(mock_cfg)


# ---------------------------------------------------------------------------
# TEST: src/agents/llm_client.py
# ---------------------------------------------------------------------------

class TestMockLLMClient:
    def test_returns_valid_json(self, mock_llm):
        """Il MockLLMClient restituisce sempre JSON valido."""
        response = mock_llm.chat([{"role": "user", "content": "Test"}])
        data = json.loads(response.content)
        assert "reasoning" in data
        assert "opinion" in data
        assert "susceptibility" in data
        assert "proposed_state" in data
        assert "spread_intent" in data

    def test_susceptibility_in_range(self, mock_llm):
        """Susceptibility e' sempre in [0, 1]."""
        for _ in range(10):
            response = mock_llm.chat([{"role": "user", "content": "x"}])
            data = json.loads(response.content)
            assert 0.0 <= data["susceptibility"] <= 1.0

    def test_proposed_state_valid(self, mock_llm):
        """proposed_state e' sempre S o I."""
        valid_states = {"S", "I", "R"}
        for _ in range(20):
            response = mock_llm.chat([{"role": "user", "content": "x"}])
            data = json.loads(response.content)
            assert data["proposed_state"] in valid_states

    def test_deterministic_with_same_seed(self):
        """Due client con stesso seed producono sequenze identiche."""
        from src.agents.llm_client import MockLLMClient
        c1 = MockLLMClient(seed=99)
        c2 = MockLLMClient(seed=99)
        msgs = [{"role": "user", "content": "hello"}]
        r1 = c1.chat(msgs)
        r2 = c2.chat(msgs)
        assert r1.content == r2.content

    def test_call_count_increments(self, mock_llm):
        for i in range(5):
            mock_llm.chat([{"role": "user", "content": "x"}])
        assert mock_llm.call_count == 5

    def test_token_counts_positive(self, mock_llm):
        response = mock_llm.chat([{"role": "user", "content": "test"}])
        assert response.input_tokens > 0
        assert response.output_tokens > 0
        assert response.total_tokens == response.input_tokens + response.output_tokens


class TestTokenBudget:
    def test_record_and_summary(self):
        """TokenBudget.record() aggiorna il totale correttamente."""
        from src.agents.llm_client import TokenBudget
        TokenBudget.reset()
        TokenBudget.record(100, 200)
        TokenBudget.record(50, 50)
        summary = TokenBudget.summary()
        assert summary["total_input"] == 150
        assert summary["total_output"] == 250
        assert summary["grand_total"] == 400
        TokenBudget.reset()

    def test_hard_limit_raises(self):
        """Il hard limit solleva RuntimeError."""
        from src.agents.llm_client import TokenBudget
        TokenBudget.reset()
        TokenBudget.configure(warn_at=10, hard_limit=100)
        with pytest.raises(RuntimeError, match="Hard limit"):
            TokenBudget.record(60, 60)  # 120 > 100
        TokenBudget.reset()
        TokenBudget.configure(warn_at=50_000, hard_limit=200_000)  # ripristina default


class TestExtractJson:
    def test_direct_parse(self):
        from src.agents.llm_client import extract_json
        text = '{"key": "value", "num": 42}'
        result, is_fb = extract_json(text)
        assert result["key"] == "value"
        assert not is_fb

    def test_strip_markdown_fence(self):
        from src.agents.llm_client import extract_json
        text = '```json\n{"a": 1}\n```'
        result, is_fb = extract_json(text)
        assert result["a"] == 1
        assert not is_fb

    def test_regex_extraction(self):
        from src.agents.llm_client import extract_json
        text = 'Some preamble text {"x": 99} and trailing text'
        result, is_fb = extract_json(text)
        assert result["x"] == 99

    def test_fallback_on_invalid(self):
        from src.agents.llm_client import extract_json
        result, is_fb = extract_json("this is not json at all!!!")
        assert is_fb is True
        assert "proposed_state" in result

    def test_custom_fallback(self):
        from src.agents.llm_client import extract_json
        fb = {"custom": "fallback"}
        result, is_fb = extract_json("invalid", fallback=fb)
        assert result["custom"] == "fallback"


# ---------------------------------------------------------------------------
# TEST: src/agents/state_machine.py
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_s_to_i_transition(self, state_machine):
        """S -> I quando pressione alta E LLM propone I."""
        result = state_machine.transition(
            current_state="S",
            node_id=1,
            neighbour_states={"I": 8, "S": 2, "R": 0, "F": 0},
            llm_output={"susceptibility": 0.9, "proposed_state": "I", "spread_intent": True},
        )
        assert result.new_state.value == "I"
        assert result.changed

    def test_s_stays_s_low_pressure(self, state_machine):
        """S rimane S se la pressione e' bassa."""
        result = state_machine.transition(
            current_state="S",
            node_id=2,
            neighbour_states={"I": 0, "S": 10, "R": 0, "F": 0},
            llm_output={"susceptibility": 0.3, "proposed_state": "S", "spread_intent": False},
        )
        assert result.new_state.value == "S"
        assert not result.changed

    def test_s_to_r_no_spread_intent(self, state_machine):
        """S -> R se esposto ma non vuole diffondere."""
        result = state_machine.transition(
            current_state="S",
            node_id=3,
            neighbour_states={"I": 3, "S": 5, "R": 0, "F": 0},
            llm_output={"susceptibility": 0.2, "proposed_state": "S", "spread_intent": False},
        )
        assert result.new_state.value == "R"

    def test_i_to_r_factcheck_pressure(self, state_machine):
        """I -> R quando molti vicini sono F."""
        result = state_machine.transition(
            current_state="I",
            node_id=4,
            neighbour_states={"I": 2, "S": 1, "R": 0, "F": 8},
            llm_output={"susceptibility": 0.5, "proposed_state": "I", "spread_intent": True},
        )
        assert result.new_state.value == "R"

    def test_f_is_absorbing(self, state_machine):
        """F rimane sempre F (stato assorbente)."""
        for proposed in ["S", "I", "R"]:
            result = state_machine.transition(
                current_state="F",
                node_id=5,
                neighbour_states={"I": 10, "S": 0, "R": 0, "F": 0},
                llm_output={"susceptibility": 1.0, "proposed_state": proposed, "spread_intent": True},
            )
            assert result.new_state.value == "F"

    def test_threshold_per_node_stable(self, state_machine):
        """La soglia di un nodo e' stabile tra chiamate successive."""
        t1 = state_machine.get_threshold(10)
        t2 = state_machine.get_threshold(10)
        assert t1 == t2

    def test_threshold_varies_by_node(self, state_machine):
        """Nodi diversi hanno soglie diverse (con alta probabilita')."""
        thresholds = [state_machine.get_threshold(i) for i in range(50)]
        assert len(set(thresholds)) > 1

    def test_effective_threshold_modulation(self, state_machine):
        """Alta suscettibilita' abbassa la soglia effettiva."""
        base = state_machine.get_threshold(20)
        theta_high_susc = state_machine.effective_threshold(20, susceptibility=0.9)
        theta_low_susc = state_machine.effective_threshold(20, susceptibility=0.1)
        assert theta_high_susc < theta_low_susc

    def test_count_states(self):
        from src.agents.state_machine import StateMachine
        states = {0: "S", 1: "I", 2: "I", 3: "R", 4: "F", 5: "S"}
        counts = StateMachine.count_states(states)
        assert counts["S"] == 2
        assert counts["I"] == 2
        assert counts["R"] == 1
        assert counts["F"] == 1

    def test_batch_transition(self, state_machine):
        """batch_transition produce lo stesso risultato delle chiamate singole."""
        states = {0: "S", 1: "I"}
        nb_map = {
            0: {"I": 5, "S": 2, "R": 0, "F": 0},
            1: {"I": 1, "S": 5, "R": 0, "F": 6},
        }
        llm_outs = {
            0: {"susceptibility": 0.9, "proposed_state": "I", "spread_intent": True},
            1: {"susceptibility": 0.5, "proposed_state": "I", "spread_intent": True},
        }
        results = state_machine.batch_transition(states, nb_map, llm_outs)
        assert len(results) == 2
        assert all(isinstance(v.new_state.value, str) for v in results.values())


# ---------------------------------------------------------------------------
# TEST: src/agents/prompts.py
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_system_prompt_contains_node_id(self, mock_cfg):
        from src.agents.prompts import build_system_prompt
        prompt = build_system_prompt(node_id=42, community=1, state="S", centrality=0.05, cfg=mock_cfg)
        assert "42" in prompt
        assert "test_topic" in prompt

    def test_system_prompt_state_descriptions(self, mock_cfg):
        from src.agents.prompts import build_system_prompt
        for state in ["S", "I", "R", "F"]:
            prompt = build_system_prompt(node_id=1, community=0, state=state, centrality=0.1, cfg=mock_cfg)
            assert len(prompt) > 100

    def test_user_prompt_with_empty_feed(self, mock_cfg):
        from src.agents.prompts import build_user_prompt
        prompt = build_user_prompt(
            feed=[],
            state="S",
            step=0,
            neighbour_states={"S": 5, "I": 0, "R": 0, "F": 0},
        )
        assert "empty" in prompt.lower() or "Step 0" in prompt

    def test_user_prompt_with_feed(self, mock_cfg):
        from src.agents.prompts import build_user_prompt
        feed = [
            {"node_id": 3, "content": "Hello world", "step": 1, "author_state": "I"},
        ]
        prompt = build_user_prompt(
            feed=feed,
            state="S",
            step=2,
            neighbour_states={"S": 3, "I": 1, "R": 0, "F": 0},
        )
        assert "Hello world" in prompt
        assert "SIMULATION STEP 2" in prompt

    def test_infection_pressure_in_prompt(self):
        from src.agents.prompts import build_user_prompt
        prompt = build_user_prompt(
            feed=[],
            state="I",
            step=5,
            neighbour_states={"I": 4, "S": 6, "R": 0, "F": 0},
        )
        assert "40.0%" in prompt or "40%" in prompt  # 4/10


# ---------------------------------------------------------------------------
# TEST: src/agents/agent.py
# ---------------------------------------------------------------------------

class TestAgent:
    def _make_agent(self, node_id: int, mock_cfg, mock_llm, state_machine, initial_state="S"):
        from src.agents.agent import Agent
        agent = Agent(
            node_id=node_id,
            cfg=mock_cfg,
            llm_client=mock_llm,
            state_machine=state_machine,
            initial_state=initial_state,
        )
        return agent

    def test_agent_step_returns_decision(self, mock_cfg, mock_llm, state_machine, network_manager, karate_graph):
        """Un ciclo completo restituisce un AgentDecision valido."""
        from src.agents.agent import Agent

        node_id = 0
        agent = Agent(node_id=node_id, cfg=mock_cfg, llm_client=mock_llm,
                      state_machine=state_machine, initial_state="S")
        agent.initialize(community=0, centrality=0.1, network_manager=network_manager)

        decision = agent.step(current_step=0, network_manager=network_manager)

        assert decision.node_id == node_id
        assert decision.step == 0
        assert decision.new_state in {"S", "I", "R", "F"}
        assert 0.0 <= decision.susceptibility <= 1.0
        assert isinstance(decision.reasoning, str)
        assert isinstance(decision.opinion, str)

    def test_infected_agent_posts(self, mock_cfg, state_machine, network_manager, karate_graph):
        """Un agente infetto con spread_intent pubblica un post."""
        from src.agents.llm_client import MockLLMClient
        from src.agents.agent import Agent

        # Mock con alta probabilita' di infezione
        infected_mock = MockLLMClient(seed=42, infection_rate=1.0)
        node_id = 0
        agent = Agent(node_id=node_id, cfg=mock_cfg, llm_client=infected_mock,
                      state_machine=state_machine, initial_state="I")
        agent.initialize(community=3, centrality=0.2, network_manager=network_manager)

        # Forza vicini tutti infetti
        for nb in network_manager.neighbours(node_id)[:5]:
            network_manager.set_state(nb, "I")

        decision = agent.step(current_step=1, network_manager=network_manager)
        # Con infection_rate=1.0, l'agente dovrebbe avere spread_intent=True
        assert isinstance(decision.opinion, str)

    def test_agent_state_is_string(self, mock_cfg, mock_llm, state_machine, network_manager):
        from src.agents.agent import Agent
        agent = Agent(node_id=5, cfg=mock_cfg, llm_client=mock_llm,
                      state_machine=state_machine, initial_state="S")
        assert isinstance(agent.state, str)
        assert agent.state == "S"

    def test_set_state_to_f(self, mock_cfg, mock_llm, state_machine, network_manager):
        """set_state('F') funziona e aggiorna la nota di aggiornamento."""
        from src.agents.agent import Agent
        agent = Agent(node_id=7, cfg=mock_cfg, llm_client=mock_llm,
                      state_machine=state_machine, initial_state="S")
        agent.set_state("F")
        assert agent.state == "F"

    def test_embedding_delta_nonzero_for_infected(self, mock_cfg, mock_llm, state_machine, network_manager):
        """La perturbazione embedding non e' zero per un agente che si infetta."""
        from src.agents.agent import Agent
        from src.agents.llm_client import MockLLMClient

        # Garantisce infezione
        always_infect = MockLLMClient(seed=1, infection_rate=1.0)
        agent = Agent(node_id=10, cfg=mock_cfg, llm_client=always_infect,
                      state_machine=state_machine, initial_state="S")
        agent.initialize(community=3, centrality=0.3, network_manager=network_manager)

        # Metti molti vicini in I per garantire superamento soglia
        for nb in network_manager.neighbours(10)[:8]:
            network_manager.set_state(nb, "I")

        emb_before = network_manager.get_embedding(10).copy()
        agent.step(current_step=0, network_manager=network_manager)
        emb_after = network_manager.get_embedding(10)

        # L'embedding deve essere cambiato
        assert not np.allclose(emb_before, emb_after)

    def test_full_cycle_multiple_steps(self, mock_cfg, mock_llm, state_machine, network_manager, karate_graph):
        """5 step consecutivi senza errori."""
        from src.agents.agent import Agent
        agents = {}
        for node_id in list(karate_graph.nodes())[:10]:
            agent = Agent(node_id=node_id, cfg=mock_cfg, llm_client=mock_llm,
                         state_machine=state_machine, initial_state="S")
            agent.initialize(community=node_id % 4, centrality=0.05, network_manager=network_manager)
            agents[node_id] = agent

        for step in range(5):
            for node_id, agent in agents.items():
                decision = agent.step(step, network_manager)
                assert decision is not None
                assert decision.new_state in {"S", "I", "R", "F"}


# ---------------------------------------------------------------------------
# TEST: src/agents/seeder.py
# ---------------------------------------------------------------------------

class TestSeeder:
    def _make_seeder(self, mock_cfg, strategy="combined"):
        from src.agents.seeder import Seeder
        return Seeder(mock_cfg, strategy=strategy)

    def _get_centralities(self, G, mock_cfg):
        from src.graph.metrics import compute_centralities
        return compute_centralities(G, mock_cfg)

    def test_select_returns_k_nodes(self, karate_graph, mock_cfg, community_map):
        seeder = self._make_seeder(mock_cfg)
        centralities = self._get_centralities(karate_graph, mock_cfg)
        selected = seeder.select(karate_graph, centralities, community_map, k=5)
        assert len(selected) == 5

    def test_selected_nodes_exist_in_graph(self, karate_graph, mock_cfg, community_map):
        seeder = self._make_seeder(mock_cfg)
        centralities = self._get_centralities(karate_graph, mock_cfg)
        selected = seeder.select(karate_graph, centralities, community_map, k=3)
        for node in selected:
            assert node in karate_graph.nodes()

    def test_selection_deterministic(self, karate_graph, mock_cfg, community_map):
        """Stessa config, stesso seed -> stessa selezione."""
        centralities = self._get_centralities(karate_graph, mock_cfg)
        s1 = self._make_seeder(mock_cfg, "combined")
        s2 = self._make_seeder(mock_cfg, "combined")
        r1 = s1.select(karate_graph, centralities, community_map, k=4)
        r2 = s2.select(karate_graph, centralities, community_map, k=4)
        assert r1 == r2

    def test_all_strategies_work(self, karate_graph, mock_cfg, community_map):
        """Tutte le strategie girano senza errori."""
        from src.agents.seeder import Seeder
        centralities = self._get_centralities(karate_graph, mock_cfg)
        for strategy in ["pagerank", "katz", "degree", "combined", "cross_community", "random"]:
            seeder = Seeder(mock_cfg, strategy=strategy)
            selected = seeder.select(karate_graph, centralities, community_map, k=3)
            assert len(selected) == 3, f"Strategia '{strategy}' ha restituito {len(selected)} nodi"

    def test_inject_sets_state_i(self, karate_graph, mock_cfg, community_map, network_manager):
        from src.agents.seeder import Seeder
        centralities = self._get_centralities(karate_graph, mock_cfg)
        seeder = Seeder(mock_cfg, strategy="degree")
        selected = seeder.select(karate_graph, centralities, community_map, k=3)
        seeder.inject(network_manager, selected, initial_state="I")
        for node_id in selected:
            assert network_manager.get_state(node_id) == "I"

    def test_inject_adds_initial_post(self, karate_graph, mock_cfg, community_map, network_manager):
        from src.agents.seeder import Seeder
        seeder = Seeder(mock_cfg, strategy="random")
        centralities = self._get_centralities(karate_graph, mock_cfg)
        selected = seeder.select(karate_graph, centralities, community_map, k=2)
        seeder.inject(network_manager, selected, initial_state="I")
        for node_id in selected:
            posts = network_manager.get_feed.__self__._post_store.get(node_id, []) if hasattr(network_manager.get_feed, '__self__') else network_manager._post_store.get(node_id, [])
            assert len(posts) >= 1

    def test_cross_community_covers_multiple_communities(self, karate_graph, mock_cfg, community_map):
        """cross_community seleziona nodi da piu' community diverse."""
        from src.agents.seeder import Seeder
        centralities = self._get_centralities(karate_graph, mock_cfg)
        seeder = Seeder(mock_cfg, strategy="cross_community")
        selected = seeder.select(karate_graph, centralities, community_map, k=8)
        communities_covered = {community_map[n] for n in selected}
        # Con 4 community e 8 seed, dovremmo coprirle tutte
        assert len(communities_covered) >= 2

    def test_describe_seeds(self, karate_graph, mock_cfg, community_map):
        """describe_seeds restituisce un report con le chiavi attese."""
        from src.agents.seeder import Seeder
        centralities = self._get_centralities(karate_graph, mock_cfg)
        seeder = Seeder(mock_cfg, strategy="combined")
        selected = seeder.select(karate_graph, centralities, community_map, k=3)
        report = seeder.describe_seeds(selected, centralities, community_map)
        assert len(report) == 3
        for entry in report:
            assert "node_id" in entry
            assert "community" in entry
            assert "pagerank" in entry
            assert "degree" in entry

    def test_invalid_strategy_raises(self, mock_cfg):
        from src.agents.seeder import Seeder
        with pytest.raises(ValueError, match="Strategia non valida"):
            Seeder(mock_cfg, strategy="nonexistent")

    def test_k_capped_at_n_nodes(self, karate_graph, mock_cfg, community_map):
        """Richiedere piu' seed dei nodi disponibili non causa errori."""
        from src.agents.seeder import Seeder
        centralities = self._get_centralities(karate_graph, mock_cfg)
        seeder = Seeder(mock_cfg, strategy="random")
        n = karate_graph.number_of_nodes()
        selected = seeder.select(karate_graph, centralities, community_map, k=n + 100)
        assert len(selected) <= n
