"""
src/orchestrator.py
===================
SimulationOrchestrator: il cuore della co-evoluzione Fase 2.

Loop co-evolutivo per step t
-------------------------------
  1. AGENT CYCLE (vedi _agent_cycle per i dettagli)
     a. prepare: per ogni nodo (ordine randomizzato), legge da
        NetworkManager e costruisce i messaggi LLM (sequenziale)
     b. dispatch: le chiamate llm.chat() vengono eseguite in PARALLELO
        su un thread pool (qui si nasconde la latenza di rete)
     c. finalize: parsing + transizione di stato + scrittura su
        NetworkManager, stesso ordine dello shuffle (sequenziale)
     → NetworkManager aggiornato (stati, post, embedding perturbati)
     NOTA: gli agenti dello step t leggono tutti lo stesso snapshot del
     grafo (fine dello step t-1) — aggiornamento "sincrono/batch", non
     piu' "asincrono" come nella versione originale. Vedi docstring di
     _agent_cycle per i dettagli del trade-off.

  2. GNN CYCLE
     trainer.train_step(G_t, embeddings_t)   (fine-tuning)
     out_embeddings = model.forward(G_t, embeddings_t)
     link_scores = trainer.predict_links(G_t, out_embeddings)

  3. REWIRING CYCLE
     to_add, to_remove = rewirer.compute(link_scores, G_t, states)
     nm.apply_rewiring(to_add, to_remove)
     → G_{t+1}

  4. METRICS & LOGGING
     compute_all_metrics(G_{t+1}, ...)
     sim_log.log_metrics(t, metrics)
     sim_log.log_rewire(t, to_add, to_remove)

  5. CHECKPOINT (ogni checkpoint_every step)
     checkpoint_manager.save(t, nm, gnn_weights)

Utilizzo
--------
    from src.orchestrator import SimulationOrchestrator
    orch = SimulationOrchestrator.build_from_config(cfg)
    orch.run(n_steps=10)
    orch.run(n_steps=5, resume_from_step=10)  # Resume
"""

from __future__ import annotations

import concurrent.futures
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.agents.agent import AgentStepContext
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class SimulationOrchestrator:
    """
    Orchestratore della simulazione co-evolutiva.

    Parameters
    ----------
    cfg : Config
    network_manager : NetworkManager
    agents : dict[int, Agent]
    state_machine : StateMachine
    gnn_model : GraphSAGEModel
    gnn_trainer : GNNTrainer
    rewirer : Rewirer
    checkpoint_manager : CheckpointManager
    sim_logger : SimLogger
    community_map : dict[int, int]
    patient_zero_ids : list[int]
    """

    def __init__(
        self,
        cfg: "Config",
        network_manager,
        agents: dict,
        state_machine,
        gnn_model,
        gnn_trainer,
        rewirer,
        checkpoint_manager,
        sim_logger,
        community_map: dict[int, int],
        patient_zero_ids: list[int],
    ) -> None:
        self._cfg = cfg
        self._nm = network_manager
        self._agents = agents
        self._sm = state_machine
        self._model = gnn_model
        self._trainer = gnn_trainer
        self._rewirer = rewirer
        self._ckpt = checkpoint_manager
        self._log = sim_logger
        self._community_map = community_map
        self._patient_zero_ids = patient_zero_ids
        self._current_step = 0
        self._checkpoint_every = cfg.simulation.checkpoint_every

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_from_config(
        cls,
        cfg: "Config",
        use_mock_llm: bool = True,
        resume: bool = False,
        run_id: str | None = None,
    ) -> "SimulationOrchestrator":
        """
        Costruisce l'orchestratore completo da config.
        Carica o inizializza tutti i componenti.
        """
        import json
        import pickle

        from src.utils.seed import set_all_seeds
        from src.utils.logger import SimLogger
        from src.agents.llm_client import MockLLMClient, LLMClient
        from src.agents.state_machine import StateMachine
        from src.agents.seeder import Seeder
        from src.agents.agent import Agent
        from src.graph.network_manager import NetworkManager
        from src.graph.metrics import compute_centralities
        from src.gnn.embeddings import EmbeddingManager
        from src.gnn.model import GraphSAGEModel
        from src.gnn.trainer import GNNTrainer
        from src.gnn.rewirer import Rewirer
        from src.utils.checkpoint import CheckpointManager

        set_all_seeds(cfg.execution.random_seed)
        run_id = run_id or str(uuid.uuid4())[:8]

        # --- Log ---
        log_path = cfg.project_root / cfg.paths.logs / f"phase2_{run_id}.jsonl"
        sim_logger = SimLogger(log_path, run_id=run_id)
        sim_logger.__enter__()

        # --- Grafo ---
        import networkx as nx
        subgraph_path = cfg.project_root / cfg.subgraph.output_file
        community_path = cfg.project_root / cfg.community.output_file
        embedding_path = cfg.project_root / cfg.gnn.embedding_file

        if subgraph_path.exists():
            with open(subgraph_path, "rb") as f:
                sg_data = pickle.load(f)
            subG = sg_data["graph"]
            raw_features = sg_data.get("node_features")
        else:
            logger.warning("Sottografo non trovato — uso grafo sintetico (n=100).")
            subG = nx.barabasi_albert_graph(100, 3, seed=cfg.execution.random_seed)
            raw_features = None

        if community_path.exists():
            with open(community_path) as f:
                comm_data = json.load(f)
            community_map = {int(k): int(v) for k, v in comm_data["community_map"].items()}
        else:
            community_map = {n: n % 4 for n in subG.nodes()}

        # --- Embeddings ---
        em = EmbeddingManager(cfg)
        if embedding_path.exists():
            embeddings = em.load()
        else:
            embeddings = em.initialize(subG, raw_features)
            em.save(embeddings)

        # --- NetworkManager ---
        nm = NetworkManager(subG, cfg, community_map=community_map, node_features=embeddings)

        # --- Centralita' e Seeder ---
        centralities = compute_centralities(subG, cfg)
        seeder = Seeder(cfg, strategy=cfg.simulation.seeder_strategy)
        patient_zero_ids = seeder.select(subG, centralities, community_map)
        seeder.inject(nm, patient_zero_ids, initial_state="I")

        # --- Agenti ---
        llm_client = MockLLMClient(seed=cfg.execution.random_seed) if use_mock_llm \
            else LLMClient.from_config(cfg)
        state_machine = StateMachine.from_config(cfg)

        agents: dict[int, Agent] = {}
        for node_id in nm.nodes:
            initial_state = nm.get_state(node_id)
            agent = Agent(node_id=node_id, cfg=cfg, llm_client=llm_client,
                          state_machine=state_machine, initial_state=initial_state)
            comm = community_map.get(node_id, 0)
            centrality_val = centralities.get(node_id, {}).get("degree_centrality", 0.0)
            agent.initialize(community=comm, centrality=centrality_val, network_manager=nm)
            agents[node_id] = agent

        # --- GNN ---
        dim = cfg.gnn.embedding_dim
        gnn_model = GraphSAGEModel(
            in_dim=dim,
            hidden_dim=cfg.gnn.hidden_dim,
            out_dim=dim,
            seed=cfg.execution.random_seed,
            force_numpy=not cfg.gnn.use_torch,
        )
        gnn_trainer = GNNTrainer(gnn_model, cfg)
        rewirer = Rewirer(cfg)
        ckpt_manager = CheckpointManager(cfg)

        # --- Resume ---
        if resume and ckpt_manager.has_checkpoint():
            ckpt_data = ckpt_manager.load_latest()
            nm = ckpt_manager.restore_network_manager(ckpt_data)
            if ckpt_data.gnn_weights:
                gnn_model.set_weights(ckpt_data.gnn_weights)
            patient_zero_ids = ckpt_data.patient_zero_ids
            logger.info("[Orchestrator] Resume dal step %d.", ckpt_data.step)

        sim_logger.log_run_start(
            config_hash=cfg.config_hash,
            seed=cfg.execution.random_seed,
            extra={
                "phase": 2,
                "mode": "mock" if use_mock_llm else "api",
                "patient_zeros": patient_zero_ids,
                "n_nodes": nm.num_nodes,
                "n_edges": nm.num_edges,
            },
        )

        orch = cls(
            cfg=cfg,
            network_manager=nm,
            agents=agents,
            state_machine=state_machine,
            gnn_model=gnn_model,
            gnn_trainer=gnn_trainer,
            rewirer=rewirer,
            checkpoint_manager=ckpt_manager,
            sim_logger=sim_logger,
            community_map=community_map,
            patient_zero_ids=patient_zero_ids,
        )
        return orch

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, n_steps: int, start_step: int = 0) -> dict:
        """
        Esegui `n_steps` step co-evolutivi.

        Parameters
        ----------
        n_steps : int      Numero di step da eseguire.
        start_step : int   Step iniziale (per resume).

        Returns
        -------
        dict con le metriche finali.
        """
        from src.graph.metrics import compute_all_metrics
        from src.agents.state_machine import StateMachine

        logger.info("=" * 60)
        logger.info("[Orchestrator] Avvio loop | step %d -> %d", start_step, start_step + n_steps - 1)
        logger.info("=" * 60)

        final_metrics: dict = {}

        for t in range(start_step, start_step + n_steps):
            self._current_step = t
            logger.info("\n--- Step %d/%d ---", t, start_step + n_steps - 1)

            try:
                metrics = self._run_step(t)
                final_metrics = metrics
            except RuntimeError as e:
                logger.error("[Orchestrator] Interruzione step %d: %s", t, e)
                break

        logger.info("=" * 60)
        logger.info("[Orchestrator] Loop completato.")
        return final_metrics

    def _run_step(self, step: int) -> dict:
        """Singolo step co-evolutivo."""
        from src.graph.metrics import compute_all_metrics
        from src.agents.state_machine import StateMachine

        # 1. AGENT CYCLE
        n_changed = self._agent_cycle(step)

        # 2. GNN CYCLE
        gnn_loss, link_scores = self._gnn_cycle(step)

        # 3. REWIRING
        n_added, n_removed = self._rewiring_cycle(step, link_scores)

        # 4. METRICS
        belief_map = self._nm.get_belief_map()
        metrics = compute_all_metrics(
            self._nm.G, self._cfg, self._community_map, belief_map
        )
        state_counts = StateMachine.count_states(self._nm.get_all_states())
        metrics.update({
            "n_S": state_counts["S"], "n_I": state_counts["I"],
            "n_R": state_counts["R"], "n_F": state_counts["F"],
            "transitions": n_changed,
            "edges_added": n_added, "edges_removed": n_removed,
            "gnn_loss": gnn_loss,
            "num_edges": self._nm.num_edges,
        })

        self._log.log_metrics(step, {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and v is not None
        })
        self._ckpt.record_metrics(step, metrics)

        logger.info(
            "  States: S=%d I=%d R=%d F=%d | "
            "Rewire: +%d -%d | ECI=%.3f | Loss=%.4f",
            state_counts["S"], state_counts["I"],
            state_counts["R"], state_counts["F"],
            n_added, n_removed,
            metrics.get("echo_chamber_index") or 0.0,
            gnn_loss,
        )

        # 5. CHECKPOINT
        if step % self._checkpoint_every == 0:
            self._ckpt.save(
                step=step,
                network_manager=self._nm,
                gnn_weights=self._model.get_weights(),
                patient_zero_ids=self._patient_zero_ids,
                meta={"run_id": self._log._run_id if hasattr(self._log, "_run_id") else ""},
            )

        return metrics

    # ------------------------------------------------------------------
    # Sub-cycles
    # ------------------------------------------------------------------

    def _agent_cycle(self, step: int) -> int:
        """
        Esegui il ciclo agenti, con le chiamate LLM dispacciate su un thread pool.

        Collo di bottiglia originario: loop Python sequenziale che chiamava
        agent.step() — quindi anche la chiamata LLM, I/O di rete — un nodo
        alla volta. Su ~230k nodi, con latenza reale di un backend API
        (anche solo 1-2s/chiamata), un singolo step costava ore.

        Soluzione in tre fasi:
          1. PREPARE (sequenziale): legge da NetworkManager e costruisce i
             messaggi per l'LLM. Nessuna chiamata di rete — solo operazioni
             in memoria, quindi veloce anche per 230k nodi.
          2. DISPATCH (parallelo): le chiamate llm.chat() di tutti i nodi
             preparati vengono dispacciate su un pool di worker thread. Qui
             si nasconde la latenza di rete — con N worker concorrenti il
             tempo totale scala come (n_nodi / N) invece di n_nodi.
          3. FINALIZE (sequenziale, stesso ordine del PREPARE): applica
             parsing, transizione di stato e scritture su NetworkManager
             (add_post, perturb_embedding, set_state).

        NetworkManager non viene MAI toccato da un worker thread — solo
        letture (fase 1) e scritture (fase 3) dal thread principale, quindi
        nessuna race condition sullo stato del grafo.

        CAMBIO DI SEMANTICA: nella versione sequenziale originale, un nodo
        processato piu' tardi nello shuffle di uno step poteva vedere gli
        effetti (post, stato) di nodi GIA' processati nello STESSO step
        ("aggiornamento asincrono"). Qui invece tutti i nodi dello step t
        leggono lo stesso snapshot, congelato alla fine dello step t-1
        ("aggiornamento sincrono/batch" — schema standard nei modelli ad
        agenti, ma comunque diverso dall'originale). Se la dinamica dipende
        molto dalla propagazione *intra-step*, vale la pena confrontare le
        metriche di un paio di step fra le due versioni prima di fidarsi
        delle run lunghe.

        get_all_states() viene chiamato UNA sola volta per l'intero step,
        non una volta per nodo: se quel metodo e' O(n_nodi), chiamarlo 230k
        volte sarebbe stato O(n^2) sull'intero step — un secondo collo di
        bottiglia indipendente da quello LLM, mascherato finora dal primo.
        """
        transitions: dict[int, tuple[str, str]] = {}
        node_order = list(
            self._nm.iter_nodes_shuffled(seed=self._cfg.execution.random_seed + step)
        )

        # Snapshot unico per l'intero step (vedi nota "cambio di semantica" sopra).
        all_states = self._nm.get_all_states()

        # --- 1. PREPARE (sequenziale, solo letture, no I/O di rete) ---
        contexts: dict[int, "AgentStepContext | None"] = {}
        for node_id in node_order:
            try:
                contexts[node_id] = self._agents[node_id].prepare_step(
                    step, self._nm, all_states=all_states
                )
            except Exception as exc:
                logger.warning("[Orchestrator] Agent %d prepare error: %s", node_id, exc)
                contexts[node_id] = None

        # Chiamate LLM concorrenti. Tara questo numero sui rate limit reali
        # del tuo backend (provider API) o sulla capacita' della tua GPU
        # (backend locale Ollama/vLLM) — un valore troppo alto produce solo
        # piu' retry per throttling/429, non piu' velocita' reale. Aggiungi
        # `max_concurrent_requests` a cfg.llm per renderlo configurabile;
        # finche' non lo fai, resta a 30.
        max_workers = getattr(
            getattr(self._cfg, "llm", None), "max_concurrent_requests", 30
        )

        # --- 2. DISPATCH (parallelo) ---
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        futures: dict[int, concurrent.futures.Future] = {}
        for node_id in node_order:
            ctx = contexts[node_id]
            if ctx is None:
                continue
            futures[node_id] = executor.submit(
                self._agents[node_id].llm_client.chat, ctx.messages
            )

        # --- 3. FINALIZE (sequenziale, stesso ordine — scritture su NetworkManager) ---
        for node_id in node_order:
            ctx = contexts[node_id]
            if ctx is None:
                continue

            try:
                response = futures[node_id].result()
            except RuntimeError:
                # Token budget hard limit raggiunto in un worker thread.
                # Stesso contratto della versione sequenziale: propaga
                # subito all'Orchestratore. wait=False perche' a budget
                # esaurito non ha senso bloccare in attesa di risposte che
                # verrebbero comunque scartate.
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as exc:
                logger.warning("[Orchestrator] Agent %d LLM error: %s", node_id, exc)
                response = None

            try:
                decision = self._agents[node_id].finalize_step(ctx, response, self._nm)
            except Exception as exc:
                logger.warning("[Orchestrator] Agent %d finalize error: %s", node_id, exc)
                continue

            if decision.state_changed:
                transitions[node_id] = (decision.old_state, decision.new_state)

        # A questo punto ogni future e' gia' stato attesto via .result(),
        # quindi questo shutdown e' immediato (non c'e' nulla da aspettare).
        executor.shutdown(wait=True)

        if transitions:
            self._log.log_state_transition(step, transitions)

        return len(transitions)

    def _gnn_cycle(self, step: int) -> tuple[float, dict]:
        """Fine-tuning GNN + calcolo score. Ritorna (loss, link_scores)."""
        embeddings = self._nm._embeddings.copy()

        gnn_loss = self._trainer.train_step(self._nm.G, embeddings, step=step)
        link_scores = self._trainer.predict_links(self._nm.G, embeddings)

        return gnn_loss, link_scores

    def _rewiring_cycle(
        self,
        step: int,
        link_scores: dict,
    ) -> tuple[int, int]:
        """Calcola e applica il rewiring. Ritorna (n_added, n_removed)."""
        # Rewiring solo ogni `rewiring_cooldown` step
        cooldown = self._cfg.simulation.rewiring_cooldown
        if cooldown > 0 and step % cooldown != 0:
            return 0, 0

        to_add, to_remove = self._rewirer.compute(
            link_scores=link_scores,
            G=self._nm.G,
            agent_states=self._nm.get_all_states(),
        )

        added, removed = self._nm.apply_rewiring(to_add, to_remove)
        n_added, n_removed = len(added), len(removed)

        if n_added or n_removed:
            self._log.log_rewire(step, added, removed)

        return n_added, n_removed

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def network_manager(self):
        return self._nm

    @property
    def state_summary(self) -> dict[str, int]:
        from src.agents.state_machine import StateMachine
        return StateMachine.count_states(self._nm.get_all_states())