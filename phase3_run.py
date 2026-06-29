"""
phase3_run.py
=============
Script di entry point per la Fase 3: Deployment & Fact-Checking (CELF).

Esegui con:
    python phase3_run.py
    python phase3_run.py --config config.yaml
    python phase3_run.py --budget-k 15   (override del budget CELF)
    python phase3_run.py --no-celf       (solo metriche, senza iniezione)
    python phase3_run.py --steps 5       (step post-intervento)

Flusso:
  1. Carica checkpoint Fase 2 (o inizializza ex-novo se non esiste).
  2. Calcola metriche baseline (pre-intervento).
  3. Esegui CELF per selezionare i seed fact-checker ottimali.
  4. Inietta fact-checker nel grafo.
  5. Esegui N step post-intervento per osservare la propagazione.
  6. Calcola e stampa il confronto before/after.
  7. Salva results/phase3_report.json.

Output:
  - results/logs/phase3_<run_id>.jsonl
  - results/phase3_report.json
  - Stampa a console tabella before/after
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(args: argparse.Namespace) -> None:
    from src.utils.config import load_config
    from src.utils.seed import set_all_seeds
    from src.utils.logger import setup_logging

    cfg = load_config(args.config)
    setup_logging(cfg.execution.log_level)
    set_all_seeds(cfg.execution.random_seed)

    import logging
    logger = logging.getLogger("phase3")

    budget_k = args.budget_k if args.budget_k is not None else cfg.influence.budget_k
    n_post_steps = args.steps if args.steps is not None else max(cfg.simulation.max_steps // 4, 3)

    logger.info("=" * 60)
    logger.info("FASE 3 — Deployment & Fact-Checking (CELF)")
    logger.info("Project root : %s", cfg.project_root)
    logger.info("Config hash  : %s", cfg.config_hash)
    logger.info("Seed         : %d", cfg.execution.random_seed)
    logger.info("Budget k     : %d", budget_k)
    logger.info("CELF         : %s", "disabilitato" if args.no_celf else "abilitato")
    logger.info("Post-steps   : %d", n_post_steps)
    logger.info("=" * 60)

    # -------------------------------------------------------
    # 1. Carica/costruisce orchestratore e stato pre-intervento
    # -------------------------------------------------------
    logger.info("\n[1/5] Caricamento stato simulazione...")
    from src.orchestrator import SimulationOrchestrator
    from src.utils.checkpoint import CheckpointManager

    ckpt_manager = CheckpointManager(cfg)
    orch = SimulationOrchestrator.build_from_config(
        cfg,
        use_mock_llm=True,
        resume=ckpt_manager.has_checkpoint(),
    )

    nm = orch.network_manager
    logger.info(
        "Stato caricato: %d nodi | %d archi | step corrente: %d",
        nm.num_nodes, nm.num_edges, orch.current_step,
    )

    # -------------------------------------------------------
    # 2. Metriche baseline (pre-intervento)
    # -------------------------------------------------------
    logger.info("\n[2/5] Calcolo metriche baseline (pre-intervento)...")
    import json as _json

    from src.graph.metrics import compute_all_metrics
    from src.agents.state_machine import StateMachine

    community_map = nm._community_map
    belief_map = nm.get_belief_map()
    baseline_metrics = compute_all_metrics(nm.G, cfg, community_map, belief_map)
    baseline_states = nm.get_all_states()
    baseline_counts = StateMachine.count_states(baseline_states)
    n_total = max(nm.num_nodes, 1)
    baseline_infection_rate = baseline_counts["I"] / n_total
    baseline_metrics["infection_rate"] = baseline_infection_rate
    baseline_metrics["n_S"] = baseline_counts["S"]
    baseline_metrics["n_I"] = baseline_counts["I"]
    baseline_metrics["n_R"] = baseline_counts["R"]
    baseline_metrics["n_F"] = baseline_counts["F"]

    logger.info(
        "Baseline: S=%d I=%d R=%d F=%d | infection_rate=%.3f | ECI=%.4f | Q=%.4f",
        baseline_counts["S"], baseline_counts["I"],
        baseline_counts["R"], baseline_counts["F"],
        baseline_infection_rate,
        baseline_metrics.get("echo_chamber_index") or 0.0,
        baseline_metrics.get("modularity_q") or 0.0,
    )

    celf_seeds: list[int] = []
    injected_nodes: list[int] = []

    if not args.no_celf:
        # -------------------------------------------------------
        # 3. CELF — selezione seed ottimali
        # -------------------------------------------------------
        logger.info("\n[3/5] CELF — selezione %d seed fact-checker...", budget_k)
        from src.influence.celf import CELF

        celf = CELF(cfg)
        celf_seeds = celf.select(
            G=nm.G,
            budget_k=budget_k,
            agent_states=nm.get_all_states(),
        )
        logger.info("CELF seeds selezionati: %s", celf_seeds)

        # -------------------------------------------------------
        # 4. Iniezione fact-checker
        # -------------------------------------------------------
        logger.info("\n[4/5] Iniezione fact-checker nei nodi selezionati...")
        from src.influence.injector import FactCheckerInjector

        injector = FactCheckerInjector(cfg)
        injected_nodes = injector.inject(nm, celf_seeds, step=orch.current_step)
        logger.info("Fact-checker iniettati: %d nodi", len(injected_nodes))

        # Aggiorna stati degli agenti per i nodi iniettati
        for node_id in injected_nodes:
            if node_id in orch._agents:
                from src.agents.agent import AgentState
                orch._agents[node_id]._state = AgentState.from_str("F")
    else:
        logger.info("\n[3/5] CELF disabilitato (--no-celf). Skip.")
        logger.info("[4/5] Nessuna iniezione.")

    # -------------------------------------------------------
    # 5. Step post-intervento
    # -------------------------------------------------------
    logger.info("\n[5/5] Esecuzione %d step post-intervento...", n_post_steps)
    start_step = orch.current_step + 1
    post_metrics = orch.run(n_steps=n_post_steps, start_step=start_step)

    # -------------------------------------------------------
    # Report finale
    # -------------------------------------------------------
    from src.influence.metrics import compute_full_influence_report

    final_states = nm.get_all_states()
    report = compute_full_influence_report(
        G=nm.G,
        agent_states=final_states,
        community_map=community_map,
        baseline_metrics=baseline_metrics,
        cfg=cfg,
    )
    report["celf_seeds"] = celf_seeds
    report["injected_nodes"] = injected_nodes
    report["n_post_steps"] = n_post_steps
    report["config_hash"] = cfg.config_hash
    report["seed"] = cfg.execution.random_seed
    report["budget_k"] = budget_k

    # -------------------------------------------------------
    # Tabella before/after
    # -------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("CONFRONTO BEFORE / AFTER INTERVENTO")
    logger.info("=" * 60)

    metrics_to_compare = [
        ("Infection Rate",     "infection_rate",      "%.4f"),
        ("Echo Chamber Idx",   "echo_chamber_index",  "%.4f"),
        ("Modularity Q",       "modularity_q",        "%.4f"),
        ("Belief Polarisation","belief_polarisation",  "%.4f"),
        ("Nodi S",             "n_S",                 "%d"),
        ("Nodi I",             "n_I",                 "%d"),
        ("Nodi R",             "n_R",                 "%d"),
        ("Nodi F",             "n_F",                 "%d"),
        ("Archi totali",       "num_edges",           "%d"),
    ]

    col_w = 22
    logger.info(
        "%-*s  %12s  %12s  %12s",
        col_w, "Metrica", "PRIMA", "DOPO", "DELTA"
    )
    logger.info("-" * 65)
    for label, key, fmt in metrics_to_compare:
        before = baseline_metrics.get(key)
        after = report.get(key)
        delta = report.get(f"delta_{key}")
        before_s = (fmt % before) if before is not None else "n/a"
        after_s  = (fmt % after)  if after  is not None else "n/a"
        delta_s  = (("%.4f" % delta) if isinstance(delta, float) else ("n/a"))
        logger.info("%-*s  %12s  %12s  %12s", col_w, label, before_s, after_s, delta_s)

    logger.info("-" * 65)
    logger.info(
        "Fact-Checker Spread (FCS) : %.4f  (%.0f%% dei nodi raggiungibili)",
        report.get("fcs", 0.0),
        report.get("fcs", 0.0) * 100,
    )
    logger.info("=" * 60)

    # Salva JSON
    report_path = cfg.project_root / cfg.paths.results / "phase3_report.json"
    serializable_report = {
        k: v for k, v in report.items()
        if isinstance(v, (int, float, str, list, dict, bool, type(None)))
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable_report, f, indent=2)

    logger.info("Report salvato in   : %s", report_path)
    logger.info("Fase 3 completata con successo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Echo Chamber Framework — Fase 3 CELF Fact-Checking"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path al file config.yaml",
    )
    parser.add_argument(
        "--budget-k", type=int, default=None,
        help="Numero di seed fact-checker da iniettare (default: config.influence.budget_k)",
    )
    parser.add_argument(
        "--no-celf", action="store_true",
        help="Salta CELF e iniezione (solo metriche post-step)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Step post-intervento (default: max_steps // 4)",
    )
    args = parser.parse_args()
    main(args)
