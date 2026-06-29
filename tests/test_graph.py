"""
tests/test_graph.py
===================
Test suite per la Fase 0 (src/graph/ e src/utils/).

Tutti i test:
  - usano un grafo sintetico piccolo (< 200 nodi) — nessun accesso a OGB
  - usano un mock della Config — nessun accesso a file system reali
  - sono deterministici con seed fisso
  - non fanno chiamate LLM o download di rete

Esegui con:
    pytest tests/test_graph.py -v
    pytest tests/test_graph.py -v --tb=short   # output compatto
"""

from __future__ import annotations

import json
import pickle
import random
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_graph() -> nx.Graph:
    """Grafo BA sintetico connesso con 150 nodi."""
    G = nx.barabasi_albert_graph(150, 3, seed=42)
    assert nx.is_connected(G)
    return G


@pytest.fixture
def small_graph() -> nx.Graph:
    """Grafo piccolo (20 nodi) per test veloci."""
    G = nx.karate_club_graph()
    return G


@pytest.fixture
def node_features(synthetic_graph) -> np.ndarray:
    """Feature matrix casuale (n, 128) per synthetic_graph."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((synthetic_graph.number_of_nodes(), 128)).astype(np.float32)


@pytest.fixture
def mock_cfg(tmp_path):
    """
    Config mock che punta a tmp_path — nessun accesso al filesystem reale.
    Costruito con dataclasses per compatibilita' con il codice sorgente.
    """
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
        dataset=DatasetConfig(name="ogbl-collab"),
        subgraph=SubgraphConfig(
            strategy="bfs_seed",
            target_nodes=80,
            min_component=10,
            seed_strategy="high_degree",
            output_file=str(tmp_path / "processed/subgraph.gpickle"),
        ),
        metrics=MetricsConfig(
            compute_betweenness=False,
            compute_diameter=False,
            pagerank_alpha=0.85,
            katz_alpha=0.01,
        ),
        community=CommunityConfig(
            algorithm="label_propagation",
            resolution=1.0,
            output_file=str(tmp_path / "processed/community_map.json"),
        ),
        simulation=SimulationConfig(),
        llm=LLMConfig(
            backend="api",
            api=LLMApiConfig(),
            local=LLMLocalConfig(),
            token_budget=LLMTokenBudget(),
        ),
        gnn=GNNConfig(
            embedding_dim=128,
            embedding_file=str(tmp_path / "processed/embeddings.npy"),
        ),
        influence=InfluenceConfig(),
        project_root=tmp_path,
        config_hash="test_hash",
    )

    # Crea directory necessarie
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "checkpoints").mkdir(parents=True, exist_ok=True)

    return cfg


# ---------------------------------------------------------------------------
# TEST: src/utils/config.py
# ---------------------------------------------------------------------------

class TestConfigLoader:
    def test_load_config_creates_dirs(self, tmp_path):
        """load_config() deve creare le directory di output."""
        # Crea un config.yaml minimale nella tmp_path
        config_content = """
execution:
  mode: local
  random_seed: 99
  log_level: INFO
paths:
  data_raw: data/raw
  data_processed: data/processed
  results: results
  figures: results/figures
  logs: results/logs
  checkpoints: results/checkpoints
dataset:
  name: ogbl-collab
subgraph:
  strategy: bfs_seed
  target_nodes: 100
  min_component: 10
  seed_strategy: high_degree
  output_file: data/processed/subgraph.gpickle
metrics:
  compute_betweenness: false
  compute_diameter: false
  pagerank_alpha: 0.85
  katz_alpha: 0.01
community:
  algorithm: label_propagation
  resolution: 1.0
  output_file: data/processed/community_map.json
simulation:
  max_steps: 5
gnn:
  embedding_dim: 64
  embedding_file: data/processed/embeddings.npy
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        import sys
        # Override PROJECT_ROOT per il test
        import src.utils.config as config_module
        original_root = config_module.PROJECT_ROOT
        config_module.PROJECT_ROOT = tmp_path

        try:
            cfg = config_module.load_config(config_path)
            assert cfg.execution.random_seed == 99
            assert cfg.execution.mode == "local"
            assert cfg.config_hash != ""
            # Le directory devono essere state create
            assert (tmp_path / "results").exists()
        finally:
            config_module.PROJECT_ROOT = original_root

    def test_config_hash_deterministic(self, tmp_path):
        """Lo stesso file produce sempre lo stesso hash."""
        import hashlib
        content = "execution:\n  random_seed: 42\n"
        p = tmp_path / "test.yaml"
        p.write_text(content)
        h1 = hashlib.sha256(content.encode()).hexdigest()[:12]
        h2 = hashlib.sha256(content.encode()).hexdigest()[:12]
        assert h1 == h2


# ---------------------------------------------------------------------------
# TEST: src/utils/seed.py
# ---------------------------------------------------------------------------

class TestSeedManager:
    def test_set_all_seeds_deterministic(self):
        """Dopo set_all_seeds, random e numpy producono sequenze identiche."""
        from src.utils.seed import set_all_seeds

        set_all_seeds(42)
        r1 = random.random()
        n1 = float(np.random.rand())

        set_all_seeds(42)
        r2 = random.random()
        n2 = float(np.random.rand())

        assert r1 == r2
        assert n1 == n2

    def test_get_rng_isolated(self):
        """get_rng() non altera lo stato globale."""
        from src.utils.seed import set_all_seeds, get_rng

        set_all_seeds(1)
        global_before = random.random()

        set_all_seeds(1)
        rng = get_rng(99)
        _ = rng.random()  # usa rng isolato
        global_after = random.random()

        assert global_before == global_after


# ---------------------------------------------------------------------------
# TEST: src/utils/logger.py
# ---------------------------------------------------------------------------

class TestSimLogger:
    def test_writes_jsonl(self, tmp_path):
        """SimLogger scrive JSONL valido."""
        from src.utils.logger import SimLogger

        log_path = tmp_path / "test.jsonl"
        with SimLogger(log_path, run_id="test01") as sl:
            sl.log_run_start(config_hash="abc123", seed=42)
            sl.log_metrics(step=0, metrics={"Q": 0.45, "ECI": 0.7})
            sl.log_rewire(step=1, added=[(1, 2)], removed=[(3, 4)])

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 3  # run_start, metrics, rewire + run_end
        for line in lines:
            record = json.loads(line)
            assert "run_id" in record
            assert "event" in record
            assert record["run_id"] == "test01"

    def test_metrics_event_structure(self, tmp_path):
        """Il record metrics ha la struttura attesa."""
        from src.utils.logger import SimLogger

        log_path = tmp_path / "metrics.jsonl"
        with SimLogger(log_path) as sl:
            sl.log_metrics(step=5, metrics={"Q": 0.3})

        lines = [json.loads(l) for l in log_path.read_text().strip().split("\n")]
        metric_record = next(r for r in lines if r["event"] == "metrics")
        assert metric_record["step"] == 5
        assert "metrics" in metric_record
        assert metric_record["metrics"]["Q"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# TEST: src/graph/extractor.py
# ---------------------------------------------------------------------------

class TestSubgraphExtractor:
    def test_bfs_sample_size(self, synthetic_graph, node_features, mock_cfg):
        """Il sottografo BFS ha al massimo target_nodes nodi."""
        from src.graph.extractor import extract_subgraph

        mock_cfg.subgraph.strategy = "bfs_seed"
        mock_cfg.subgraph.target_nodes = 60

        subG, sub_feat, node_map = extract_subgraph(synthetic_graph, node_features, mock_cfg)

        assert subG.number_of_nodes() <= 60
        assert subG.number_of_nodes() >= mock_cfg.subgraph.min_component
        assert nx.is_connected(subG)

    def test_features_aligned(self, synthetic_graph, node_features, mock_cfg):
        """Le feature del sottografo hanno la stessa dimensione delle colonne originali."""
        from src.graph.extractor import extract_subgraph

        subG, sub_feat, node_map = extract_subgraph(synthetic_graph, node_features, mock_cfg)

        assert sub_feat.shape[0] == subG.number_of_nodes()
        assert sub_feat.shape[1] == node_features.shape[1]

    def test_node_map_valid(self, synthetic_graph, node_features, mock_cfg):
        """node_map mappa nodi originali a nodi del sottografo."""
        from src.graph.extractor import extract_subgraph

        subG, sub_feat, node_map = extract_subgraph(synthetic_graph, node_features, mock_cfg)

        assert len(node_map) == subG.number_of_nodes()
        # Tutti i nuovi ID sono in [0, n-1]
        new_ids = set(node_map.values())
        assert new_ids == set(range(subG.number_of_nodes()))

    def test_cache_hit(self, synthetic_graph, node_features, mock_cfg):
        """La seconda chiamata usa la cache senza ricalcolare."""
        from src.graph.extractor import extract_subgraph

        subG1, sf1, nm1 = extract_subgraph(synthetic_graph, node_features, mock_cfg)
        subG2, sf2, nm2 = extract_subgraph(synthetic_graph, node_features, mock_cfg)

        assert subG1.number_of_nodes() == subG2.number_of_nodes()
        assert np.allclose(sf1, sf2)

    def test_random_walk_strategy(self, synthetic_graph, node_features, mock_cfg):
        """La strategia random_walk produce un sottografo valido."""
        from src.graph.extractor import extract_subgraph

        # Usa path diverso per evitare cache del test BFS
        mock_cfg.subgraph.strategy = "random_walk"
        mock_cfg.subgraph.output_file = str(
            Path(mock_cfg.subgraph.output_file).parent / "subgraph_rw.gpickle"
        )
        mock_cfg.gnn.embedding_file = str(
            Path(mock_cfg.gnn.embedding_file).parent / "embeddings_rw.npy"
        )

        subG, sub_feat, node_map = extract_subgraph(synthetic_graph, node_features, mock_cfg)
        assert subG.number_of_nodes() > 0
        assert nx.is_connected(subG)


# ---------------------------------------------------------------------------
# TEST: src/graph/metrics.py
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_topological_metrics_complete(self, small_graph, mock_cfg):
        """compute_topological_metrics ritorna tutte le chiavi attese."""
        from src.graph.metrics import compute_topological_metrics

        m = compute_topological_metrics(small_graph, mock_cfg)

        assert "num_nodes" in m
        assert "num_edges" in m
        assert "density" in m
        assert "avg_clustering" in m
        assert "avg_degree" in m
        assert m["num_nodes"] == small_graph.number_of_nodes()

    def test_modularity_range(self, small_graph, mock_cfg):
        """La modularity Q e' in [-0.5, 1.0]."""
        from src.graph.metrics import compute_modularity

        # Crea community map semplice: 2 community
        comm_map = {n: n % 2 for n in small_graph.nodes()}
        q = compute_modularity(small_graph, comm_map)
        assert -0.5 <= q <= 1.0

    def test_eci_range(self, small_graph):
        """ECI e' in [0, 1]."""
        from src.graph.metrics import compute_echo_chamber_index

        comm_map = {n: n % 3 for n in small_graph.nodes()}
        eci = compute_echo_chamber_index(small_graph, comm_map)
        assert 0.0 <= eci <= 1.0

    def test_eci_all_same_community(self, small_graph):
        """Se tutti i nodi sono nella stessa community, ECI deve essere 1.0."""
        from src.graph.metrics import compute_echo_chamber_index

        comm_map = {n: 0 for n in small_graph.nodes()}
        eci = compute_echo_chamber_index(small_graph, comm_map)
        assert eci == pytest.approx(1.0)

    def test_belief_polarisation_max(self):
        """Distribuzione 50/50 {0, 1} -> BP vicino a 1.0."""
        from src.graph.metrics import compute_belief_polarisation

        belief = {i: float(i % 2) for i in range(100)}
        bp = compute_belief_polarisation(belief)
        assert bp == pytest.approx(1.0, abs=0.05)

    def test_belief_polarisation_zero(self):
        """Tutti stessi valori -> BP = 0."""
        from src.graph.metrics import compute_belief_polarisation

        belief = {i: 0.5 for i in range(50)}
        bp = compute_belief_polarisation(belief)
        assert bp == pytest.approx(0.0, abs=1e-6)

    def test_centralities_keys(self, small_graph, mock_cfg):
        """compute_centralities include degree, pagerank, katz per ogni nodo."""
        from src.graph.metrics import compute_centralities

        result = compute_centralities(small_graph, mock_cfg)
        assert len(result) == small_graph.number_of_nodes()
        for node, vals in result.items():
            assert "degree_centrality" in vals
            assert "pagerank" in vals
            assert "katz" in vals


# ---------------------------------------------------------------------------
# TEST: src/graph/community.py
# ---------------------------------------------------------------------------

class TestCommunityDetection:
    def test_label_propagation_covers_all_nodes(self, small_graph, mock_cfg):
        """Ogni nodo deve avere una community assegnata."""
        from src.graph.community import detect_communities

        comm_map, n_comm, q = detect_communities(small_graph, mock_cfg)
        assert len(comm_map) == small_graph.number_of_nodes()
        assert n_comm >= 1

    def test_modularity_positive(self, small_graph, mock_cfg):
        """Q deve essere positivo per il karate club graph."""
        from src.graph.community import detect_communities

        _, _, q = detect_communities(small_graph, mock_cfg)
        assert q > 0.0

    def test_cache_hit(self, small_graph, mock_cfg):
        """La seconda chiamata usa la cache JSON."""
        from src.graph.community import detect_communities

        cm1, n1, q1 = detect_communities(small_graph, mock_cfg)
        cm2, n2, q2 = detect_communities(small_graph, mock_cfg)

        assert cm1 == cm2
        assert n1 == n2
        assert q1 == pytest.approx(q2)

    def test_community_stats(self, small_graph, mock_cfg):
        """community_stats ritorna le chiavi attese."""
        from src.graph.community import detect_communities, community_stats

        comm_map, _, _ = detect_communities(small_graph, mock_cfg)
        stats = community_stats(small_graph, comm_map)

        assert "n_communities" in stats
        assert "min_size" in stats
        assert "max_size" in stats
        assert stats["min_size"] >= 1
        assert stats["max_size"] <= small_graph.number_of_nodes()


# ---------------------------------------------------------------------------
# TEST: src/graph/network_manager.py
# ---------------------------------------------------------------------------

class TestNetworkManager:
    def test_init_states(self, small_graph, mock_cfg):
        """Tutti i nodi partono in stato S."""
        from src.graph.network_manager import NetworkManager

        nm = NetworkManager(small_graph, mock_cfg)
        for node in nm.nodes:
            assert nm.get_state(node) == "S"

    def test_add_remove_edge(self, small_graph, mock_cfg):
        """add_edge e remove_edge aggiornano correttamente la topologia."""
        from src.graph.network_manager import NetworkManager

        nm = NetworkManager(small_graph, mock_cfg)
        n_before = nm.num_edges

        # Trova un arco non esistente
        all_nodes = nm.nodes
        u, v = all_nodes[0], all_nodes[-1]
        if nm.G.has_edge(u, v):
            nm.G.remove_edge(u, v)

        added = nm.add_edge(u, v)
        assert added is True
        assert nm.G.has_edge(u, v)

        removed = nm.remove_edge(u, v)
        assert removed is True
        assert not nm.G.has_edge(u, v)

    def test_add_existing_edge_returns_false(self, small_graph, mock_cfg):
        """add_edge su arco gia' esistente ritorna False."""
        from src.graph.network_manager import NetworkManager

        nm = NetworkManager(small_graph, mock_cfg)
        u, v = list(nm.iter_edges())[0]
        result = nm.add_edge(u, v)
        assert result is False

    def test_post_store_and_feed(self, small_graph, mock_cfg):
        """add_post + get_feed ritornano i post del vicinato."""
        from src.graph.network_manager import NetworkManager

        nm = NetworkManager(small_graph, mock_cfg)
        # Prendi un nodo con almeno un vicino
        node = next(n for n in nm.nodes if len(nm.neighbours(n)) > 0)
        neighbour = nm.neighbours(node)[0]

        nm.add_post(neighbour, {"content": "Test post", "step": 1})
        feed = nm.get_feed(node, window=5)

        assert len(feed) >= 1
        assert feed[0]["content"] == "Test post"

    def test_set_invalid_state_raises(self, small_graph, mock_cfg):
        """set_state con stato non valido solleva ValueError."""
        from src.graph.network_manager import NetworkManager

        nm = NetworkManager(small_graph, mock_cfg)
        with pytest.raises(ValueError, match="Stato non valido"):
            nm.set_state(0, "X")

    def test_perturb_embedding(self, small_graph, mock_cfg):
        """perturb_embedding modifica l'embedding del nodo."""
        from src.graph.network_manager import NetworkManager

        feat = np.ones((small_graph.number_of_nodes(), 128), dtype=np.float32)
        nm = NetworkManager(small_graph, mock_cfg, node_features=feat)

        emb_before = nm.get_embedding(0).copy()
        delta = np.full(128, 0.5, dtype=np.float32)
        nm.perturb_embedding(0, delta)
        emb_after = nm.get_embedding(0)

        assert not np.allclose(emb_before, emb_after)

    def test_save_load_checkpoint(self, small_graph, mock_cfg, tmp_path):
        """save() + load() ripristina lo stato completo del NetworkManager."""
        from src.graph.network_manager import NetworkManager

        nm = NetworkManager(small_graph, mock_cfg)
        nm.set_state(0, "I")
        nm.add_post(0, {"content": "hello", "step": 1})

        ckpt_path = tmp_path / "test_ckpt.gpickle"
        nm.save(ckpt_path, step=1)

        nm2 = NetworkManager.load(ckpt_path, mock_cfg)
        assert nm2.get_state(0) == "I"
        assert len(nm2.all_posts(0)) == 1
        assert nm2.num_nodes == nm.num_nodes

    def test_apply_rewiring(self, small_graph, mock_cfg):
        """apply_rewiring aggiunge e rimuove archi in batch."""
        from src.graph.network_manager import NetworkManager

        nm = NetworkManager(small_graph, mock_cfg)
        existing = list(nm.iter_edges())[0]
        u, v = existing

        # Trova coppia non connessa
        nodes = nm.nodes
        non_edge = None
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if not nm.G.has_edge(nodes[i], nodes[j]):
                    non_edge = (nodes[i], nodes[j])
                    break
            if non_edge:
                break

        if non_edge:
            added, removed = nm.apply_rewiring(
                to_add=[non_edge],
                to_remove=[existing],
            )
            assert len(added) == 1
            assert len(removed) == 1

    def test_iter_nodes_shuffled_reproducible(self, small_graph, mock_cfg):
        """iter_nodes_shuffled con stesso seed produce stesso ordine."""
        from src.graph.network_manager import NetworkManager

        nm = NetworkManager(small_graph, mock_cfg)
        order1 = list(nm.iter_nodes_shuffled(seed=42))
        order2 = list(nm.iter_nodes_shuffled(seed=42))
        assert order1 == order2


# ---------------------------------------------------------------------------
# Metodo di supporto aggiunto a NetworkManager per i test
# ---------------------------------------------------------------------------

def _all_posts_patch(self, node_id: int) -> list[dict]:
    """Helper per recuperare tutti i post di un nodo (usato nei test)."""
    return list(self._post_store.get(node_id, []))


# Monkey-patch solo per i test (il metodo non e' nel sorgente per non appesantire l'API)
from src.graph.network_manager import NetworkManager
NetworkManager.all_posts = _all_posts_patch
