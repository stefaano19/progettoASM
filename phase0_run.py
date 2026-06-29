"""
phase0_run.py
=============
Script di entry point per la Fase 0: Setup & Baseline.

Esegui con:
    python phase0_run.py
    python phase0_run.py --config config.yaml
    python phase0_run.py --skip-download  (se ogbl-collab e' gia' in cache)

Output:
  - data/processed/subgraph.gpickle
  - data/processed/embeddings.npy
  - data/processed/community_map.json
  - results/logs/phase0_<run_id>.jsonl
  - Stampa a console delle metriche baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Assicura che src/ sia nel path (compatibile con Kaggle)
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(args: argparse.Namespace) -> None:
    # --- Setup ---
    from src.utils.config import load_config
    from src.utils.seed import set_all_seeds
    from src.utils.logger import setup_logging, SimLogger

    cfg = load_config(args.config)
    setup_logging(cfg.execution.log_level)
    set_all_seeds(cfg.execution.random_seed)

    import logging
    logger = logging.getLogger("phase0")
    logger.info("=" * 60)
    logger.info("FASE 0 — Setup & Baseline")
    logger.info("Project root : %s", cfg.project_root)
    logger.info("Config hash  : %s", cfg.config_hash)
    logger.info("Seed         : %d", cfg.execution.random_seed)
    logger.info("=" * 60)

    import uuid
    run_id = str(uuid.uuid4())[:8]
    log_path = cfg.project_root / cfg.paths.logs / f"phase0_{run_id}.jsonl"

    with SimLogger(log_path, run_id=run_id) as sim_log:
        sim_log.log_run_start(
            config_hash=cfg.config_hash,
            seed=cfg.execution.random_seed,
            extra={"phase": 0, "mode": cfg.execution.mode},
        )

        # -------------------------------------------------------
        # 1. Caricamento ogbl-collab
        # -------------------------------------------------------
        if not args.skip_download:
            logger.info("\n[1/4] Caricamento ogbl-collab...")
            from src.graph.data_loader import load_collab_graph, graph_summary
            G_full, node_features = load_collab_graph(cfg)
            summary = graph_summary(G_full)
            logger.info("Grafo completo: %s", summary)
            sim_log.log_metrics(step=-1, metrics={k: v for k, v in summary.items() if isinstance(v, (int, float))})
        else:
            logger.info("[1/4] Skip download — carico da cache...")
            from src.graph.data_loader import load_collab_graph
            G_full, node_features = load_collab_graph(cfg)

        # -------------------------------------------------------
        # 2. Estrazione sottografo
        # -------------------------------------------------------
        logger.info("\n[2/4] Estrazione sottografo (target=%d nodi)...", cfg.subgraph.target_nodes)
        from src.graph.extractor import extract_subgraph
        subG, sub_features, node_map = extract_subgraph(G_full, node_features, cfg)

        logger.info(
            "Sottografo: %d nodi, %d archi | density=%.4f",
            subG.number_of_nodes(),
            subG.number_of_edges(),
            __import__("networkx").density(subG),
        )

        # -------------------------------------------------------
        # 3. Community detection
        # -------------------------------------------------------
        logger.info("\n[3/4] Community detection (algoritmo=%s)...", cfg.community.algorithm)
        from src.graph.community import detect_communities, community_stats
        community_map, n_comm, q = detect_communities(subG, cfg)
        stats = community_stats(subG, community_map)

        logger.info(
            "Community: %d trovate | Q=%.4f | size: min=%d max=%d mean=%.1f",
            n_comm, q, stats["min_size"], stats["max_size"], stats["mean_size"],
        )

        # -------------------------------------------------------
        # 4. Metriche baseline
        # -------------------------------------------------------
        logger.info("\n[4/4] Calcolo metriche baseline...")
        from src.graph.metrics import compute_all_metrics, compute_centralities
        metrics = compute_all_metrics(subG, cfg, community_map=community_map)
        sim_log.log_metrics(step=0, metrics={
            k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None
        })

        centralities = compute_centralities(subG, cfg)
        # Mostra top-5 nodi per PageRank
        top_pr = sorted(
            [(n, d.get("pagerank", 0)) for n, d in centralities.items()],
            key=lambda x: x[1], reverse=True,
        )[:5]

        # -------------------------------------------------------
        # Report finale
        # -------------------------------------------------------
        logger.info("\n" + "=" * 60)
        logger.info("RIEPILOGO FASE 0")
        logger.info("=" * 60)
        logger.info("Nodi sottografo  : %d", subG.number_of_nodes())
        logger.info("Archi sottografo : %d", subG.number_of_edges())
        logger.info("Densita'         : %.4f", metrics["density"])
        logger.info("Avg Clustering   : %.4f", metrics.get("avg_clustering", float("nan")))
        logger.info("N. Community     : %d", n_comm)
        logger.info("Modularity Q     : %.4f", q)
        logger.info("Echo Chamber Idx : %.4f", metrics.get("echo_chamber_index", float("nan")))
        logger.info("Top-5 PageRank   : %s", [(n, f"{pr:.4f}") for n, pr in top_pr])
        logger.info("=" * 60)
        logger.info("Log salvato in   : %s", log_path)
        logger.info("Fase 0 completata con successo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo Chamber Framework — Fase 0 Setup")
    parser.add_argument("--config", default="config.yaml", help="Path al file config.yaml")
    parser.add_argument("--skip-download", action="store_true", help="Salta il download OGB e usa la cache")
    args = parser.parse_args()
    main(args)
