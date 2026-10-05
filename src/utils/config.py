"""
src/utils/config.py
===================
YAML loader che produce un Config dataclass tipizzato.

Utilizzo
--------
    from src.utils.config import load_config
    cfg = load_config()          # cerca config.yaml nella project root
    cfg = load_config("my_config.yaml")

Il loader risolve tutti i path relativi rispetto alla project root
(la directory che contiene config.yaml), quindi il codice funziona
indipendentemente dalla directory di lavoro corrente.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def _find_project_root(start: Path | None = None) -> Path:
    """
    Risale l'albero delle directory finche' non trova config.yaml.
    Fallback: directory di lavoro corrente.
    """
    candidate = start or Path(__file__).resolve().parent
    for parent in [candidate, *candidate.parents]:
        if (parent / "config.yaml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT: Path = _find_project_root()


# ---------------------------------------------------------------------------
# Sub-configs (dataclasses annidati)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    mode: str = "local"
    random_seed: int = 42
    log_level: str = "INFO"


@dataclass
class PathsConfig:
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    results: Path = Path("results")
    figures: Path = Path("results/figures")
    logs: Path = Path("results/logs")
    checkpoints: Path = Path("results/checkpoints")


@dataclass
class DatasetConfig:
    name: str = "ogbl-collab"
    ogb_root: str = "data/raw"
    year_filter: int | None = None


@dataclass
class SubgraphConfig:
    strategy: str = "bfs_seed"
    target_nodes: int = 1000
    min_component: int = 200
    seed_strategy: str = "high_degree"
    forward_prob: float = 0.5
    output_file: str = "data/processed/subgraph.gpickle"


@dataclass
class MetricsConfig:
    compute_betweenness: bool = False
    betweenness_sample: int = 500
    compute_diameter: bool = False
    pagerank_alpha: float = 0.85
    katz_alpha: float = 0.01


@dataclass
class CommunityConfig:
    algorithm: str = "louvain"
    resolution: float = 1.0
    output_file: str = "data/processed/community_map.json"


@dataclass
class SimulationConfig:
    max_steps: int = 20
    topic: str = "echo_chamber_polarization"
    topic_description: str = ""
    initial_infection_rate: float = 0.05
    activation_probability: float = 1.0
    memory_window: int = 5
    rewiring_cooldown: int = 3
    checkpoint_every: int = 5
    seeder_strategy: str = "combined"
    # Parametri StateMachine (prima letti con getattr ma mai caricabili da YAML)
    min_resistance_exposure: float = 0.12
    resistance_susceptibility_cutoff: float = 0.5
    max_llm_failure_rate: float = 0.5   # oltre questa quota di chiamate fallite (o fallback) lo step si ferma
    base_threshold_mean: float = 0.3    # soglia LT base per nodo ~ N(mean, std)
    base_threshold_std: float = 0.1
    relapse_threshold: float = 0.6      # R -> I se frazione vicini I >= soglia ...
    relapse_min_susceptibility: float = 0.8  # ... e susceptibility LLM > questo valore (e proposed_state = I)
    smart_cache: bool = False           # riusa la risposta LLM se feed e vicinato non cambiano (congela gli agenti)


@dataclass
class LLMApiConfig:
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_tokens: int = 512
    api_key_env: str = "GEMINI_API_KEY"
    timeout: float = 300.0              # secondi per richiesta (poi retry e fallback)


@dataclass
class LLMLocalConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 300.0              # secondi per richiesta (poi retry e fallback)


@dataclass
class LLMTokenBudget:
    warn_at: int = 50_000
    hard_limit: int = 200_000


@dataclass
class LLMConfig:
    backend: str = "api"
    max_concurrent_requests: int = 30
    api: LLMApiConfig = field(default_factory=LLMApiConfig)
    local: LLMLocalConfig = field(default_factory=LLMLocalConfig)
    token_budget: LLMTokenBudget = field(default_factory=LLMTokenBudget)


@dataclass
class GNNConfig:
    model: str = "graphsage"
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 0.001
    epochs_per_step: int = 5
    embedding_dim: int = 64
    rewire_threshold_add: float = 0.8
    rewire_threshold_remove: float = 0.2
    max_new_edges_per_step: int = 10
    embedding_file: str = "data/processed/embeddings.npy"
    use_torch: bool = False
    rewire_mode: str = "swap"                 # "swap" (densita' costante) | "threshold"
    rewire_candidate_sources: int = 500       # nodi sorgente esplorati per step
    rewire_candidates_per_source: int = 5     # coppie 2-hop candidate per sorgente


@dataclass
class InfluenceConfig:
    budget_k: int = 10
    simulation_rounds: int = 100
    activation_threshold: float = 0.4
    celf_interval: int = 5
    # Effetto dei fact-checker (F) nella StateMachine
    fc_resistance_threshold: float = 0.25   # I -> R se frazione vicini F >= soglia
    fc_protection_threshold: float = 0.10   # S -> R se frazione vicini F >= soglia
    celf_objective: str = "threshold"       # "threshold" (allineato) | "ic"
    reach_hops: int = 1                     # raggio della metrica di reach F


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    subgraph: SubgraphConfig = field(default_factory=SubgraphConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    community: CommunityConfig = field(default_factory=CommunityConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    influence: InfluenceConfig = field(default_factory=InfluenceConfig)

    # Runtime attributes (non in YAML)
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    config_hash: str = ""

    def resolve(self, relative_path: str) -> Path:
        """Risolvi un path relativo rispetto alla project root."""
        return self.project_root / relative_path

    def effective_hash(self) -> str:
        """
        Hash dei parametri EFFETTIVI (dopo eventuali override da notebook).
        config_hash e' calcolato sul file YAML al caricamento e non vede le
        modifiche fatte a runtime (es. cfg.simulation.activation_probability).
        """
        import dataclasses
        import json
        data = dataclasses.asdict(self)
        data.pop("project_root", None)
        data.pop("config_hash", None)
        blob = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def ensure_dirs(self) -> None:
        """Crea tutte le directory di output se non esistono."""
        for attr in [
            self.paths.data_raw,
            self.paths.data_processed,
            self.paths.results,
            self.paths.figures,
            self.paths.logs,
            self.paths.checkpoints,
        ]:
            (self.project_root / attr).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _nested_to_dataclass(raw: dict, dc_class: type) -> Any:
    """
    Converte ricorsivamente un dict annidato nel dataclass corrispondente.
    Ignora chiavi non riconosciute (forward-compatibility).
    """
    import dataclasses
    if not dataclasses.is_dataclass(dc_class):
        return raw

    field_map = {f.name: f for f in dataclasses.fields(dc_class)}
    kwargs: dict[str, Any] = {}

    for f in dataclasses.fields(dc_class):
        if f.name not in raw:
            continue
        val = raw[f.name]
        # Ricorsione su dataclass annidati
        origin = getattr(f.type, "__origin__", None)
        if dataclasses.is_dataclass(f.type):
            kwargs[f.name] = _nested_to_dataclass(val, f.type)
        elif isinstance(val, dict) and isinstance(f.default_factory, type) and dataclasses.is_dataclass(f.default_factory):  # type: ignore[arg-type]
            kwargs[f.name] = _nested_to_dataclass(val, f.default_factory)
        else:
            # Path conversion per PathsConfig
            if dc_class is PathsConfig and isinstance(val, str):
                kwargs[f.name] = Path(val)
            else:
                kwargs[f.name] = val

    return dc_class(**kwargs)


def load_config(config_path: str | Path | None = None) -> Config:
    """
    Carica config.yaml e restituisce un oggetto Config tipizzato.

    Parameters
    ----------
    config_path : str | Path | None
        Percorso al file YAML. Se None, cerca config.yaml nella project root.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml non trovato in '{config_path}'. "
            f"Project root rilevata: '{PROJECT_ROOT}'."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    # Calcola hash del file per riproducibilita'
    with open(config_path, "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    cfg = Config(
        execution=_nested_to_dataclass(raw.get("execution", {}), ExecutionConfig),
        paths=_nested_to_dataclass(raw.get("paths", {}), PathsConfig),
        dataset=_nested_to_dataclass(raw.get("dataset", {}), DatasetConfig),
        subgraph=_nested_to_dataclass(raw.get("subgraph", {}), SubgraphConfig),
        metrics=_nested_to_dataclass(raw.get("metrics", {}), MetricsConfig),
        community=_nested_to_dataclass(raw.get("community", {}), CommunityConfig),
        simulation=_nested_to_dataclass(raw.get("simulation", {}), SimulationConfig),
        llm=_load_llm_config(raw.get("llm", {})),
        gnn=_nested_to_dataclass(raw.get("gnn", {}), GNNConfig),
        influence=_nested_to_dataclass(raw.get("influence", {}), InfluenceConfig),
        project_root=PROJECT_ROOT,
        config_hash=config_hash,
    )

    cfg.ensure_dirs()
    return cfg


def _load_llm_config(raw: dict) -> LLMConfig:
    """Parser specifico per la sezione llm con sotto-dict annidate."""
    return LLMConfig(
        backend=raw.get("backend", "api"),
        max_concurrent_requests=raw.get("max_concurrent_requests", 30),
        api=_nested_to_dataclass(raw.get("api", {}), LLMApiConfig),
        local=_nested_to_dataclass(raw.get("local", {}), LLMLocalConfig),
        token_budget=_nested_to_dataclass(raw.get("token_budget", {}), LLMTokenBudget),
    )
