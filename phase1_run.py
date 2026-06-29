"""
phase1_run.py
=============
Entry point per la Fase 1: Logica Agente (Infezione).

Esegue un ciclo di N step con agenti LLM su un sottografo estratto.
In modalita' locale usa il MockLLMClient (zero costi API).
In modalita' 'api' usa il backend reale configurato in config.yaml.

Utilizzo
--------
    python phase1_run.py                      # mock LLM, 5 step
    python phase1_run.py --steps 10           # mock LLM, 10 step
    python phase1_run.py --real-llm           # LLM reale (richiede API key)
    python phase1_run.py --strategy combined  # strategia seeding
    python phase1_run.py --k 3               # 3 pazienti zero

Pre-requisiti
-------------
    Fase 0 completata:
      data/processed/subgraph.gpickle
      data/processed/embeddings.npy
      data/processed/community_map.json
    Oppure: lo script usa grafi sintetici se i file non esistono.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(args: argparse.Namespace) -> None:
    import uuid
    import logging

    from src.utils.config import load_config
    from src.utils.seed import set_all_seeds
    from src.utils.logger import setup_logging, SimLogger
    from src.agents.llm_client import MockLLMClient, LLMClient, TokenBudget
    from src.agents.state_machine import StateMachine
    from src.agents.seeder import Seeder
    from src.agents.agent import Agent

    cfg = load_config(args.config)
    setup_logging(cfg.execution.log_level)
    set_all_seeds(cfg.execution.random_seed)

    logger = logging.getLogger("phase1")
    logger.info("=" * 60)
    logger.info("FASE 1 — Logica Agente (Infezione)")
    logger.info("LLM mode    : %s", "REAL API" if args.real_llm else "MOCK")
    logger.info("Steps       : %d", args.steps)
    logger.info("Strategy    : %s", args.strategy)
    logger.info("=" * 60)

    # -------------------------------------------------------
    # 1. Carica artefatti Fase 0 (o costruisce grafo sintetico)
    # -------------------------------------------------------
    subgraph_path = cfg.project_root / cfg.subgraph.output_file
    community_path = cfg.project_root / cfg.community.output_file
    embedding_path = cfg.project_root / cfg.gnn.embedding_file

    import networkx as nx
    import numpy as np

    if subgraph_path.exists():
        import pickle
        logger.info("[1/5] Caricamento sottografo da Phase 0...")
        with open(subgraph_path, "rb") as f:
            data = pickle.load(f)
        subG = data["graph"]
        sub_features = np.load(str(embedding_path)) if embedding_path.exists() else None
    else:
        logger.warning("[1/5] Sottografo non trovato — costruisco grafo sintetico (n=100).")
        subG = nx.barabasi_albert_graph(100, 3, seed=cfg.execution.random_seed)
        sub_features = None

    if community_path.exists():
        with open(community_path, "r") as f:
            comm_data = json.load(f)
        community_map = {int(k): int(v) for k, v in comm_data["community_map"].items()}
    else:
        community_map = {n: n % 4 for n in subG.nodes()}

    logger.info("Grafo: %d nodi, %d archi", subG.number_of_nodes(), subG.number_of_edges())

    # -------------------------------------------------------
    # 2. Inizializza NetworkManager
    # -------------------------------------------------------
    logger.info("[2/5] Inizializzazione NetworkManager...")
    from src.graph.network_manager import NetworkManager
    nm = NetworkManager(subG, cfg, community_map=community_map, node_features=sub_features)

    # -------------------------------------------------------
    # 3. Calcola centralita' e seleziona pazienti zero
    # -------------------------------------------------------
    logger.info("[3/5] Calcolo centralita' e selezione pazienti zero...")
    from src.graph.metrics import compute_centralities
    centralities = compute_centralities(subG, cfg)

    seeder = Seeder(cfg, strategy=args.strategy)
    k = args.k if args.k else None
    patient_zero_ids = seeder.select(subG, centralities, community_map, k=k)
    seeder.inject(nm, patient_zero_ids, initial_state="I")

    seed_report = seeder.describe_seeds(patient_zero_ids, centralities, community_map)
    logger.info("Pazienti zero:")
    for s in seed_report:
        logger.info(
            "  Node %d | community=%d | degree=%d | pagerank=%.5f",
            s["node_id"], s["community"], s["degree"], s["pagerank"],
        )

    # -------------------------------------------------------
    # 4. Crea agenti
    # -------------------------------------------------------
    logger.info("[4/5] Creazione agenti (%d nodi)...", nm.num_nodes)

    if args.real_llm:
        llm_client = LLMClient.from_config(cfg)
    else:
        llm_client = MockLLMClient(
            seed=cfg.execution.random_seed,
            infection_rate=0.35,
        )

    state_machine = StateMachine.from_config(cfg)

    agents: dict[int, Agent] = {}
    for node_id in nm.nodes:
        initial_state = nm.get_state(node_id)  # I se paziente zero, S altrimenti
        agent = Agent(
            node_id=node_id,
            cfg=cfg,
            llm_client=llm_client,
            state_machine=state_machine,
            initial_state=initial_state,
        )
        comm = community_map.get(node_id, 0)
        centrality_val = centralities.get(node_id, {}).get("degree_centrality", 0.0)
        agent.initialize(community=comm, centrality=centrality_val, network_manager=nm)
        agents[node_id] = agent

    # -------------------------------------------------------
    # 5. Loop di simulazione
    # -------------------------------------------------------
    run_id = str(uuid.uuid4())[:8]
    log_path = cfg.project_root / cfg.paths.logs / f"phase1_{run_id}.jsonl"

    from src.graph.metrics import compute_all_metrics
    import random

    with SimLogger(log_path, run_id=run_id) as sim_log:
        sim_log.log_run_start(
            config_hash=cfg.config_hash,
            seed=cfg.execution.random_seed,
            extra={
                "phase": 1,
                "mode": "mock" if not args.real_llm else "api",
                "patient_zeros": patient_zero_ids,
            },
        )

        logger.info("[5/5] Avvio loop di simulazione (%d step)...", args.steps)
        logger.info("=" * 60)

        for step in range(args.steps):
            # Ordine randomizzato degli agenti (async update)
            node_order = list(nm.iter_nodes_shuffled(seed=cfg.execution.random_seed + step))
            transitions: dict[int, tuple[str, str]] = {}

            for node_id in node_order:
                try:
                    decision = agents[node_id].step(step, nm)
                except RuntimeError as e:
                    logger.error("Token budget esaurito: %s", e)
                    break

                if decision.state_changed:
                    transitions[node_id] = (decision.old_state, decision.new_state)

                sim_log.log_agent_decision(step=step, node_id=node_id, decision={
                    "old_state": decision.old_state,
                    "new_state": decision.new_state,
                    "state_changed": decision.state_changed,
                    "susceptibility": decision.susceptibility,
                    "spread_intent": decision.spread_intent,
                    "opinion": decision.opinion[:80] if decision.opinion else "",
                    "tokens_in": decision.tokens_in,
                    "tokens_out": decision.tokens_out,
                    "is_fallback": decision.is_fallback,
                })

            # Log transizioni
            if transitions:
                sim_log.log_state_transition(step=step, transitions=transitions)

            # Metriche di step
            belief_map = nm.get_belief_map()
            metrics = compute_all_metrics(subG, cfg, community_map, belief_map)
            state_counts = state_machine.count_states(nm.get_all_states())
            metrics.update({
                "n_S": state_counts["S"],
                "n_I": state_counts["I"],
                "n_R": state_counts["R"],
                "n_F": state_counts["F"],
            })
            sim_log.log_metrics(step=step, metrics={
                k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None
            })

            # Token usage
            token_summary = LLMClient.token_summary() if not args.real_llm else MockLLMClient.token_summary() if hasattr(MockLLMClient, "token_summary") else {"grand_total": 0}
            if not args.real_llm:
                mock_total = llm_client.call_count * 130
                sim_log.log_token_usage(step=step, total_tokens=mock_total, delta_tokens=len(node_order) * 130)
            else:
                token_summary = LLMClient.token_summary()
                sim_log.log_token_usage(
                    step=step,
                    total_tokens=token_summary["grand_total"],
                    delta_tokens=token_summary["grand_total"],
                )

            logger.info(
                "Step %2d | S=%3d I=%3d R=%3d F=%3d | ECI=%.3f | Transitions=%d",
                step,
                state_counts["S"], state_counts["I"],
                state_counts["R"], state_counts["F"],
                metrics.get("echo_chamber_index") or 0.0,
                len(transitions),
            )

        # -------------------------------------------------------
        # Report finale
        # -------------------------------------------------------
        final_states = nm.get_all_states()
        final_counts = state_machine.count_states(final_states)
        n_total = nm.num_nodes

        logger.info("=" * 60)
        logger.info("RIEPILOGO FASE 1")
        logger.info("=" * 60)
        logger.info("Nodi totali    : %d", n_total)
        logger.info("Pazienti zero  : %d", len(patient_zero_ids))
        logger.info("Step simulati  : %d", args.steps)
        logger.info("")
        logger.info("Stato finale   :")
        logger.info("  S (Susceptible) : %d (%.1f%%)", final_counts["S"], 100 * final_counts["S"] / n_total)
        logger.info("  I (Infected)    : %d (%.1f%%)", final_counts["I"], 100 * final_counts["I"] / n_total)
        logger.info("  R (Resistant)   : %d (%.1f%%)", final_counts["R"], 100 * final_counts["R"] / n_total)
        logger.info("  F (FactChecker) : %d (%.1f%%)", final_counts["F"], 100 * final_counts["F"] / n_total)
        logger.info("")
        logger.info("Log salvato    : %s", log_path)
        logger.info("Fase 1 completata con successo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo Chamber Framework — Fase 1 Agenti")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", type=int, default=5, help="Numero di step da simulare")
    parser.add_argument("--real-llm", action="store_true", help="Usa LLM reale (richiede API key)")
    parser.add_argument("--strategy", default="combined",
                        choices=["pagerank", "betweenness", "katz", "degree",
                                 "combined", "cross_community", "random"],
                        help="Strategia di selezione pazienti zero")
    parser.add_argument("--k", type=int, default=None, help="Numero pazienti zero (default: da config)")
    args = parser.parse_args()
    main(args)
