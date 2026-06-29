"""
phase2_run.py
=============
Script di entry point per la Fase 2: Dinamiche di Rete (Co-evoluzione).

Esegui con:
    python phase2_run.py
    python phase2_run.py --config config.yaml
    python phase2_run.py --steps 10
    python phase2_run.py --no-mock-llm     (usa LLM reale, costa token!)
    python phase2_run.py --resume          (riprende dall'ultimo checkpoint)

Output:
  - results/logs/phase2_<run_id>.jsonl
  - results/checkpoints/ckpt_step_XXXX.pkl
  - results/phase2_report.json
  - Stampa a console delle metriche per step
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Compatibilita' Kaggle: assicura che src/ sia nel path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(args: argparse.Namespace) -> None:
    from src.utils.config import load_config
    from src.utils.seed import set_all_seeds
    from src.utils.logger import setup_logging

    cfg = load_config(args.config)
    setup_logging(cfg.execution.log_level)
    set_all_seeds(cfg.execution.random_seed)

    import logging
    logger = logging.getLogger("phase2")

    n_steps = args.steps if args.steps is not None else cfg.simulation.max_steps
    use_mock = not args.no_mock_llm

    logger.info("=" * 60)
    logger.info("FASE 2 — Dinamiche di Rete (Co-evoluzione)")
    logger.info("Project root : %s", cfg.project_root)
    logger.info("Config hash  : %s", cfg.config_hash)
    logger.info("Seed         : %d", cfg.execution.random_seed)
    logger.info("Steps        : %d", n_steps)
    logger.info("LLM          : %s", "mock" if use_mock else "api")
    logger.info("Resume       : %s", args.resume)
    logger.info("=" * 60)

    # -------------------------------------------------------
    # Costruzione orchestratore
    # -------------------------------------------------------
    logger.info("\n[1/3] Costruzione SimulationOrchestrator...")
    from src.orchestrator import SimulationOrchestrator

    orch = SimulationOrchestrator.build_from_config(
        cfg,
        use_mock_llm=use_mock,
        resume=args.resume,
    )
    logger.info(
        "Grafo: %d nodi | %d archi | GNN: %s",
        orch.network_manager.num_nodes,
        orch.network_manager.num_edges,
        "numpy" if not cfg.gnn.use_torch else "torch",
    )

    # -------------------------------------------------------
    # Loop co-evolutivo
    # -------------------------------------------------------
    logger.info("\n[2/3] Avvio loop co-evolutivo (%d step)...", n_steps)

    final_metrics: dict = {}
    try:
        final_metrics = orch.run(n_steps=n_steps)
    except KeyboardInterrupt:
        logger.warning("\n[Fase 2] Interrotto dall'utente — salvo checkpoint...")
        from src.utils.checkpoint import CheckpointManager
        ckpt = CheckpointManager(cfg)
        ckpt.save(
            step=orch.current_step,
            network_manager=orch.network_manager,
            gnn_weights=orch._model.get_weights(),
            patient_zero_ids=orch._patient_zero_ids,
        )
        logger.info("Checkpoint salvato. Puoi riprendere con --resume.")

    # -------------------------------------------------------
    # Report finale
    # -------------------------------------------------------
    logger.info("\n[3/3] Report finale...")
    state_summary = orch.state_summary

    logger.info("\n" + "=" * 60)
    logger.info("RIEPILOGO FASE 2")
    logger.info("=" * 60)
    logger.info("Step eseguiti       : %d", orch.current_step + 1)
    logger.info("Nodi S (Susceptible): %d", state_summary.get("S", 0))
    logger.info("Nodi I (Infected)   : %d", state_summary.get("I", 0))
    logger.info("Nodi R (Resistant)  : %d", state_summary.get("R", 0))
    logger.info("Nodi F (Fact-Check) : %d", state_summary.get("F", 0))

    n_total = max(orch.network_manager.num_nodes, 1)
    infection_rate = state_summary.get("I", 0) / n_total
    logger.info("Infection Rate      : %.3f", infection_rate)

    if final_metrics:
        logger.info(
            "Echo Chamber Index  : %.4f",
            final_metrics.get("echo_chamber_index") or float("nan"),
        )
        logger.info(
            "Modularity Q        : %.4f",
            final_metrics.get("modularity_q") or float("nan"),
        )
        logger.info("GNN Loss (ultimo)   : %.4f", final_metrics.get("gnn_loss", 0.0))
        logger.info("Archi finali        : %d", final_metrics.get("num_edges", 0))

    logger.info("=" * 60)

    # Salva report JSON
    report_path = cfg.project_root / cfg.paths.results / "phase2_report.json"
    report = {
        "phase": 2,
        "config_hash": cfg.config_hash,
        "seed": cfg.execution.random_seed,
        "n_steps": n_steps,
        "llm_mode": "mock" if use_mock else "api",
        "final_state_counts": state_summary,
        "infection_rate": infection_rate,
        **{k: v for k, v in (final_metrics or {}).items()
           if isinstance(v, (int, float)) and v is not None},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Report salvato in   : %s", report_path)
    logger.info("Fase 2 completata.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Echo Chamber Framework — Fase 2 Co-evoluzione"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path al file config.yaml",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Numero di step (default: config.simulation.max_steps)",
    )
    parser.add_argument(
        "--no-mock-llm", action="store_true",
        help="Usa LLM reale invece del mock (costa token!)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Riprendi dall'ultimo checkpoint salvato",
    )
    args = parser.parse_args()
    main(args)
