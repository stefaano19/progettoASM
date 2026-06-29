"""
src/agents/state_machine.py
============================
Macchina a stati per gli agenti della simulazione.

Stati possibili
---------------
  S — Susceptible  : esposto ma non ancora convinto dalla narrazione
  I — Infected     : crede e diffonde attivamente la narrazione polarizzante
  R — Resistant    : esposto, critico, non si lascia convincere; vettore fact-check
  F — Fact-Checker : seed iniettato da CELF; non transita verso I

Modello di transizione: Linear Threshold (LT) esteso
------------------------------------------------------
Il classico LT usa una soglia fissa theta_u. Qui la soglia e' modulata
dall'output dell'agente LLM:

    effective_threshold(u) = base_threshold(u) * (1 - susceptibility_modifier)

dove `susceptibility_modifier` e' il campo `susceptibility` dell'output LLM,
normalizzato in [-0.5, +0.5] attorno alla base.

Regole di transizione:
  S -> I  : se frazione_vicini_I >= effective_threshold  AND proposed_state == "I"
  S -> R  : se spread_intent == False
            AND frazione_vicini_I >= min_resistance_exposure (esposizione non banale)
            AND susceptibility < resistance_susceptibility_cutoff (genuinamente poco suscettibile)
            Altrimenti resta S — "in bilico": esposto ma non ancora schierato,
            potra' ancora convertirsi in I piu' avanti se la pressione cresce.
  I -> R  : se frazione_vicini_F > resistance_threshold  OR proposed_state == "R"
  I -> I  : altrimenti (rimane infetto)
  R -> I  : se fraction_I >= relapse_threshold AND susceptibility > 0.8  (ricaduta)
  R -> R  : altrimenti (rimane resistente)
  F -> F  : lo stato F e' assorbente (iniettato da CELF, non torna indietro)

Utilizzo
--------
    from src.agents.state_machine import StateMachine, AgentState
    sm = StateMachine(cfg)
    new_state = sm.transition(
        current_state="S",
        node_id=5,
        neighbours_states={"I": 3, "S": 2, "R": 1, "F": 0},
        llm_output={"susceptibility": 0.8, "proposed_state": "I", "spread_intent": True},
    )
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentState enum
# ---------------------------------------------------------------------------

class AgentState(str, Enum):
    S = "S"   # Susceptible
    I = "I"   # Infected
    R = "R"   # Resistant
    F = "F"   # Fact-Checker (seed CELF)

    @classmethod
    def from_str(cls, s: str) -> "AgentState":
        try:
            return cls(s.upper())
        except ValueError:
            logger.warning("[StateMachine] Stato non valido '%s', fallback S.", s)
            return cls.S


# ---------------------------------------------------------------------------
# Transition result
# ---------------------------------------------------------------------------

@dataclass
class TransitionResult:
    old_state: AgentState
    new_state: AgentState
    reason: str
    infection_pressure: float   # fraction_I nel vicinato
    effective_threshold: float

    @property
    def changed(self) -> bool:
        return self.old_state != self.new_state


# ---------------------------------------------------------------------------
# StateMachine
# ---------------------------------------------------------------------------

class StateMachine:
    """
    Linear Threshold esteso con modulazione LLM.

    Parameters
    ----------
    seed : int
        Seed per assegnazione stocastica delle soglie base per nodo.
    base_threshold_mean : float
        Media della soglia base (default 0.3).
    base_threshold_std : float
        Deviazione standard della soglia base (default 0.1).
    resistance_threshold : float
        Frazione minima di vicini F per far transitare I -> R.
    relapse_threshold : float
        Soglia di ricaduta R -> I (alta per renderla rara).
    """

    def __init__(
        self,
        seed: int = 42,
        base_threshold_mean: float = 0.3,
        base_threshold_std: float = 0.1,
        resistance_threshold: float = 0.25,
        relapse_threshold: float = 0.6,
        min_resistance_exposure: float = 0.12,
        resistance_susceptibility_cutoff: float = 0.5,
    ) -> None:
        self._rng = random.Random(seed)
        self._base_threshold_mean = base_threshold_mean
        self._base_threshold_std = base_threshold_std
        self._resistance_threshold = resistance_threshold
        self._relapse_threshold = relapse_threshold
        self._min_resistance_exposure = min_resistance_exposure
        self._resistance_susceptibility_cutoff = resistance_susceptibility_cutoff
        self._node_thresholds: dict[int, float] = {}

    @classmethod
    def from_config(cls, cfg: "Config") -> "StateMachine":
        # getattr con default: non rompe se questi campi non esistono ancora
        # nella tua dataclass Config — puoi aggiungerli quando vuoi.
        sim_cfg = getattr(cfg, "simulation", None)
        return cls(
            seed=cfg.execution.random_seed,
            min_resistance_exposure=getattr(sim_cfg, "min_resistance_exposure", 0.12),
            resistance_susceptibility_cutoff=getattr(
                sim_cfg, "resistance_susceptibility_cutoff", 0.5
            ),
        )

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def get_threshold(self, node_id: int) -> float:
        """
        Soglia base per il nodo, assegnata una volta e poi fissa.
        Campionata da N(mean, std) e clippata in [0.05, 0.95].
        """
        if node_id not in self._node_thresholds:
            raw = self._rng.gauss(
                self._base_threshold_mean,
                self._base_threshold_std,
            )
            self._node_thresholds[node_id] = max(0.05, min(0.95, raw))
        return self._node_thresholds[node_id]

    def effective_threshold(self, node_id: int, susceptibility: float) -> float:
        """
        Soglia effettiva modulata dall'output LLM.

        susceptibility in [0, 1]:
          - 0.5 = neutro (nessuna modulazione)
          - >0.5 = piu' suscettibile (soglia scende)
          - <0.5 = piu' resistente (soglia sale)

        Formula: theta_eff = theta_base * (1.5 - susceptibility)
        Range: theta_base * 0.5 ... theta_base * 1.5
        """
        base = self.get_threshold(node_id)
        modifier = 1.5 - susceptibility  # 1.5 quando susc=0, 0.5 quando susc=1
        return max(0.05, min(0.99, base * modifier))

    # ------------------------------------------------------------------
    # Core transition
    # ------------------------------------------------------------------

    def transition(
        self,
        current_state: str | AgentState,
        node_id: int,
        neighbour_states: dict[str, int],
        llm_output: dict,
    ) -> TransitionResult:
        """
        Calcola la transizione di stato per un nodo.

        Parameters
        ----------
        current_state : str | AgentState
            Stato corrente del nodo ("S", "I", "R", "F").
        node_id : int
            ID del nodo (per la soglia personalizzata).
        neighbour_states : dict[str, int]
            Conteggio degli stati dei vicini: {"S": n, "I": m, "R": k, "F": j}.
        llm_output : dict
            Output JSON dell'agente LLM:
            {susceptibility, proposed_state, spread_intent, ...}

        Returns
        -------
        TransitionResult
        """
        # Estrai il valore stringa puro: se è già un'istanza AgentState usa .value,
        # altrimenti converti a stringa (evita 'AgentState.S' da str(enum)).
        raw_state = current_state.value if isinstance(current_state, AgentState) else str(current_state)
        state = AgentState.from_str(raw_state)

        # Estrai parametri LLM
        susceptibility = float(llm_output.get("susceptibility", 0.5))
        susceptibility = max(0.0, min(1.0, susceptibility))
        proposed_raw = str(llm_output.get("proposed_state", state.value))
        proposed = AgentState.from_str(proposed_raw)
        spread_intent = bool(llm_output.get("spread_intent", False))

        # Pressione di infezione: fraction of infected neighbours
        total_neighbours = max(sum(neighbour_states.values()), 1)
        n_infected = neighbour_states.get("I", 0)
        n_factcheck = neighbour_states.get("F", 0)
        fraction_I = n_infected / total_neighbours
        fraction_F = n_factcheck / total_neighbours

        theta = self.effective_threshold(node_id, susceptibility)

        # --- Regole di transizione ---
        new_state = state
        reason = "no_change"

        if state == AgentState.F:
            # F e' assorbente
            new_state = AgentState.F
            reason = "fact_checker_permanent"

        elif state == AgentState.S:
            if fraction_I >= theta and proposed == AgentState.I:
                new_state = AgentState.I
                reason = f"lt_infection (f_I={fraction_I:.2f} >= theta={theta:.2f})"
            elif (
                not spread_intent
                and fraction_I >= self._min_resistance_exposure
                and susceptibility < self._resistance_susceptibility_cutoff
            ):
                new_state = AgentState.R
                reason = (
                    f"active_resistance (f_I={fraction_I:.2f} >= "
                    f"{self._min_resistance_exposure:.2f}, susc={susceptibility:.2f} < "
                    f"{self._resistance_susceptibility_cutoff:.2f})"
                )
            # else: rimane S (esposto ma in bilico — non abbastanza esposizione,
            # o susceptibility troppo alta per "chiudersi" gia' ora)

        elif state == AgentState.I:
            if fraction_F >= self._resistance_threshold or proposed == AgentState.R:
                new_state = AgentState.R
                reason = (
                    f"fact_check_pressure (f_F={fraction_F:.2f})"
                    if fraction_F >= self._resistance_threshold
                    else "llm_self_correction"
                )
            # else: rimane I

        elif state == AgentState.R:
            # Ricaduta: solo se vicinato molto infetto E alta suscettibilita'
            if fraction_I >= self._relapse_threshold and susceptibility > 0.8:
                new_state = AgentState.I
                reason = f"relapse (f_I={fraction_I:.2f}, susc={susceptibility:.2f})"
            # else: rimane R

        if new_state != state:
            logger.info(
                "[StateMachine] Nodo %d: %s -> %s (%s)",
                node_id, state.value, new_state.value, reason,
            )

        return TransitionResult(
            old_state=state,
            new_state=new_state,
            reason=reason,
            infection_pressure=fraction_I,
            effective_threshold=theta,
        )

    # ------------------------------------------------------------------
    # Batch transition (per tutto il grafo)
    # ------------------------------------------------------------------

    def batch_transition(
        self,
        states: dict[int, str],
        neighbour_states_map: dict[int, dict[str, int]],
        llm_outputs: dict[int, dict],
    ) -> dict[int, TransitionResult]:
        """
        Applica la transizione a tutti i nodi del grafo in un passo.

        Parameters
        ----------
        states : dict[int, str]
            Stato corrente per ogni nodo.
        neighbour_states_map : dict[int, dict[str, int]]
            Per ogni nodo, il conteggio degli stati dei suoi vicini.
        llm_outputs : dict[int, dict]
            Output LLM per ogni nodo.

        Returns
        -------
        dict[int, TransitionResult]
        """
        results: dict[int, TransitionResult] = {}
        for node_id, current_state in states.items():
            nb_states = neighbour_states_map.get(node_id, {})
            llm_out = llm_outputs.get(node_id, {})
            results[node_id] = self.transition(
                current_state=current_state,
                node_id=node_id,
                neighbour_states=nb_states,
                llm_output=llm_out,
            )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def count_states(states: dict[int, str]) -> dict[str, int]:
        """Conta quanti nodi sono in ogni stato."""
        counts: dict[str, int] = {"S": 0, "I": 0, "R": 0, "F": 0}
        for s in states.values():
            counts[s] = counts.get(s, 0) + 1
        return counts

    @staticmethod
    def get_neighbour_state_counts(
        node_id: int,
        all_states: dict[int, str],
        neighbours: list[int],
    ) -> dict[str, int]:
        """Helper: restituisce il conteggio stati dei vicini di un nodo."""
        counts: dict[str, int] = {"S": 0, "I": 0, "R": 0, "F": 0}
        for nb in neighbours:
            s = all_states.get(nb, "S")
            counts[s] = counts.get(s, 0) + 1
        return counts