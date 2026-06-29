"""
src/influence/metrics.py
========================
Metriche di valutazione dell'intervento Fact-Checker (Fase 3).

Metriche calcolate
------------------
  FCS (Fact-Checker Spread):
    Frazione di nodi del grafo raggiungibili dai nodi F in BFS.
    Misura la "copertura potenziale" dei fact-checker.

  n_fact_checkers:
    Numero di nodi correntemente in stato 'F'.

  avg_reach_per_fc:
    Media dei nodi raggiungibili per singolo fact-checker.

  delta_infection_rate, delta_echo_chamber_index, ...:
    Differenza (corrente - baseline) per ogni metrica topologica.
    Valori negativi indicano riduzione del fenomeno (successo dell'intervento).

Utilizzo
--------
    from src.influence.metrics import compute_full_influence_report
    report = compute_full_influence_report(G, agent_states, community_map,
                                           baseline_metrics, cfg)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fact-Checker Spread
# ---------------------------------------------------------------------------

def compute_fact_checker_spread(
    G: nx.Graph,
    agent_states: dict[int, str],
) -> dict[str, float]:
    """
    Calcola la copertura potenziale dei nodi Fact-Checker (stato 'F').

    La "copertura" e' definita come il numero di nodi raggiungibili
    dai nodi F tramite BFS (componente connessa contenente almeno un F).

    Parameters
    ----------
    G : nx.Graph
        Grafo corrente (post-rewiring).
    agent_states : dict[int, str]
        Stati agenti {node_id: "S"|"I"|"R"|"F"}.

    Returns
    -------
    dict con:
      - fcs (float): frazione di nodi raggiungibili da almeno un F [0, 1]
      - n_fact_checkers (int): numero di nodi F
      - avg_reach_per_fc (float): nodi raggiungibili in media per F
      - total_reachable (int): nodi unici raggiungibili da tutti gli F
    """
    n = G.number_of_nodes()
    if n == 0:
        return {"fcs": 0.0, "n_fact_checkers": 0, "avg_reach_per_fc": 0.0, "total_reachable": 0}

    fc_nodes = [node for node, state in agent_states.items() if state == "F"]
    n_fc = len(fc_nodes)

    if n_fc == 0:
        return {"fcs": 0.0, "n_fact_checkers": 0, "avg_reach_per_fc": 0.0, "total_reachable": 0}

    # BFS da ogni fact-checker
    reach_per_fc: list[int] = []
    all_reachable: set[int] = set()

    for fc in fc_nodes:
        if fc not in G:
            continue
        reachable = set(nx.single_source_shortest_path_length(G, fc).keys())
        reach_per_fc.append(len(reachable))
        all_reachable.update(reachable)

    total_reachable = len(all_reachable)
    fcs = total_reachable / n
    avg_reach = float(np.mean(reach_per_fc)) if reach_per_fc else 0.0

    logger.info(
        "[InfluenceMetrics] FCS=%.3f | n_FC=%d | avg_reach=%.1f | total_reach=%d/%d",
        fcs, n_fc, avg_reach, total_reachable, n,
    )

    return {
        "fcs": fcs,
        "n_fact_checkers": n_fc,
        "avg_reach_per_fc": avg_reach,
        "total_reachable": total_reachable,
    }


# ---------------------------------------------------------------------------
# Delta metrics (confronto baseline vs post-intervento)
# ---------------------------------------------------------------------------

def compute_intervention_delta(
    baseline_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
) -> dict[str, float]:
    """
    Calcola la variazione di ogni metrica rispetto alla baseline.

    delta = current - baseline

    Valori negativi per metriche come infection_rate, echo_chamber_index
    indicano miglioramento (riduzione del fenomeno).

    Parameters
    ----------
    baseline_metrics : dict
        Metriche misurate prima dell'intervento (post-Fase 1/2).
    current_metrics : dict
        Metriche misurate dopo l'intervento CELF (post-Fase 3).

    Returns
    -------
    dict con chiavi "delta_<metric_name>" per ogni metrica numerica comune.
    """
    delta: dict[str, float] = {}

    for key in baseline_metrics:
        base_val = baseline_metrics.get(key)
        curr_val = current_metrics.get(key)

        if base_val is None or curr_val is None:
            continue
        if not isinstance(base_val, (int, float)) or not isinstance(curr_val, (int, float)):
            continue

        delta[f"delta_{key}"] = float(curr_val) - float(base_val)

    return delta


# ---------------------------------------------------------------------------
# Report completo
# ---------------------------------------------------------------------------

def compute_full_influence_report(
    G: nx.Graph,
    agent_states: dict[int, str],
    community_map: dict[int, int],
    baseline_metrics: dict[str, Any],
    cfg: "Config",
) -> dict[str, Any]:
    """
    Report completo dell'intervento Fact-Checker per la Fase 3.

    Combina:
      - Metriche di spread dei fact-checker (FCS, avg_reach)
      - Metriche topologiche post-intervento (ECI, Q, infection_rate)
      - Delta rispetto alla baseline pre-intervento

    Parameters
    ----------
    G : nx.Graph
        Grafo post-intervento.
    agent_states : dict[int, str]
        Stati finali degli agenti.
    community_map : dict[int, int]
        Mappa community Louvain.
    baseline_metrics : dict
        Snapshot metriche pre-intervento.
    cfg : Config
        Configurazione globale.

    Returns
    -------
    dict con tutte le metriche per il report finale.
    """
    from src.graph.metrics import compute_all_metrics
    from src.agents.state_machine import StateMachine

    # Metriche topologiche post-intervento
    belief_map = {
        node: 1.0 if state == "I" else (0.5 if state == "F" else 0.0)
        for node, state in agent_states.items()
    }
    current_metrics = compute_all_metrics(G, cfg, community_map, belief_map)

    # Conteggi stati
    state_counts = StateMachine.count_states(agent_states)
    n_total = max(len(agent_states), 1)
    current_metrics["infection_rate"] = state_counts["I"] / n_total
    current_metrics["n_S"] = state_counts["S"]
    current_metrics["n_I"] = state_counts["I"]
    current_metrics["n_R"] = state_counts["R"]
    current_metrics["n_F"] = state_counts["F"]

    # Spread dei fact-checker
    fcs_metrics = compute_fact_checker_spread(G, agent_states)

    # Delta
    delta_metrics = compute_intervention_delta(baseline_metrics, current_metrics)

    report: dict[str, Any] = {
        **current_metrics,
        **fcs_metrics,
        **delta_metrics,
        "budget_k": cfg.influence.budget_k,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }

    logger.info(
        "[InfluenceMetrics] Report: infection_rate=%.3f | ECI=%.3f | FCS=%.3f | "
        "delta_infection=%.3f | n_F=%d",
        current_metrics.get("infection_rate", 0.0),
        current_metrics.get("echo_chamber_index", 0.0),
        fcs_metrics.get("fcs", 0.0),
        delta_metrics.get("delta_infection_rate", 0.0),
        state_counts["F"],
    )

    return report
