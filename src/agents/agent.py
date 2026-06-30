"""
src/agents/agent.py
===================
Classe Agent: un nodo cognitivo nella rete sociale simulata.

Ciclo di vita per step temporale t
-------------------------------------
  1. perceive()  — legge il feed e i metadati del vicinato da NetworkManager
  2. cognize()   — chiama l'LLM con system+user prompt, ottiene JSON
  3. act()       — pubblica il post (se non silenzioso), calcola delta embedding
  4. transition()— delega alla StateMachine per il nuovo stato

Il metodo pubblico `step()` orchestra l'intero ciclo e restituisce
un `AgentDecision` che l'Orchestratore usa per:
  - aggiornare il NetworkManager (stato, embedding, post)
  - loggare la decisione nel SimLogger
  - decidere il rewiring (Fase 2)

Perturbazione dell'embedding
------------------------------
L'agente non genera direttamente un vettore 128-dim (troppo costoso via LLM).
Invece, la perturbazione e' calcolata internamente:

  delta = infection_direction * magnitude * sign

dove:
  - infection_direction: vettore fisso assegnato alla creazione (seed-based)
  - magnitude: |susceptibility - 0.5| * scale  (max effetto a susc=0 o 1)
  - sign: +1 se si infetta, -1 se resiste

Questo fornisce segnale utile alla GNN in Fase 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.agents.llm_client import LLMClient, MockLLMClient, extract_json
from src.agents.prompts import (
    build_system_prompt,
    build_user_prompt,
    build_state_update_note,
)
from src.agents.state_machine import AgentState, StateMachine

if TYPE_CHECKING:
    from src.agents.llm_client import LLMResponse
    from src.graph.network_manager import NetworkManager
    from src.utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentDecision
# ---------------------------------------------------------------------------

@dataclass
class AgentDecision:
    """Output strutturato di un ciclo completo dell'agente."""
    node_id: int
    step: int
    old_state: str
    new_state: str
    state_changed: bool
    opinion: str              # Testo postato (vuoto se silenzioso)
    reasoning: str
    susceptibility: float
    spread_intent: bool
    infection_pressure: float
    effective_threshold: float
    embedding_delta_norm: float   # Norma L2 del delta embedding
    tokens_in: int = 0
    tokens_out: int = 0
    is_fallback: bool = False     # True se LLM ha usato fallback JSON


@dataclass
class AgentStepContext:
    """
    Stato intermedio fra prepare_step() e finalize_step().

    Esiste per permettere all'Orchestratore di disaccoppiare la fase di
    percezione/costruzione prompt (CPU, veloce) dalla chiamata LLM vera e
    propria (I/O di rete, lenta) e dalla fase di scrittura su
    NetworkManager (deve restare sequenziale). Vedi
    SimulationOrchestrator._agent_cycle per l'uso in batch concorrente.
    """
    old_state: str
    current_step: int
    messages: list[dict]
    nb_state_counts: dict[str, int]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    Agente cognitivo su un nodo della rete sociale.

    Parameters
    ----------
    node_id : int
        ID del nodo (rinumerato 0..n-1 da extractor).
    cfg : Config
        Configurazione globale.
    llm_client : LLMClient | MockLLMClient
        Client LLM (reale o mock per testing).
    state_machine : StateMachine
        Istanza condivisa della macchina a stati (soglie per nodo).
    initial_state : str
        Stato iniziale ("S" per default, "I" per pazienti zero, "F" per seed CELF).
    """

    _EMBEDDING_SCALE = 0.05   # Ampiezza massima della perturbazione

    def __init__(
        self,
        node_id: int,
        cfg: "Config",
        llm_client: LLMClient | MockLLMClient,
        state_machine: StateMachine,
        initial_state: str = "S",
    ) -> None:
        self.node_id = node_id
        self._cfg = cfg
        self._llm = llm_client
        self._sm = state_machine
        self._state = AgentState.from_str(initial_state)
        self._memory_window = cfg.simulation.memory_window

        # System prompt costruito alla creazione (rimane stabile salvo cambio stato)
        self._system_prompt: str = ""
        self._state_update_note: str = ""

        # Direzione fissa per perturbazione embedding (seed per riproducibilita')
        rng = np.random.default_rng(cfg.execution.random_seed + node_id)
        dim = cfg.gnn.embedding_dim
        direction = rng.standard_normal(dim).astype(np.float32)
        self._infection_direction: np.ndarray = direction / (np.linalg.norm(direction) + 1e-8)

    def initialize(
        self,
        community: int,
        centrality: float,
        network_manager: "NetworkManager",
    ) -> None:
        """
        Completa l'inizializzazione dell'agente con i dati del grafo.
        Chiamato dall'Orchestratore dopo la costruzione.
        """
        self._community = community
        self._centrality = centrality
        self._system_prompt = build_system_prompt(
            node_id=self.node_id,
            community=community,
            state=self._state.value,
            centrality=centrality,
            cfg=self._cfg,
        )
        logger.debug("[Agent %d] Inizializzato | state=%s | community=%d", self.node_id, self._state.value, community)

    # ------------------------------------------------------------------
    # Stato corrente
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def llm_client(self) -> LLMClient | MockLLMClient:
        """
        Espone il client LLM condiviso. Serve all'Orchestratore per dispacciare
        le chiamate .chat() su un thread pool (vedi prepare_step/finalize_step)
        senza dover accedere all'attributo privato _llm.
        """
        return self._llm

    def set_state(self, new_state: str) -> None:
        """Forza lo stato (usato da CELF per iniezione fact-checker)."""
        old = self._state.value
        self._state = AgentState.from_str(new_state)
        if old != new_state:
            self._state_update_note = build_state_update_note(old, new_state)

    # ------------------------------------------------------------------
    # Ciclo completo
    # ------------------------------------------------------------------

    def step(
        self,
        current_step: int,
        network_manager: "NetworkManager",
    ) -> AgentDecision:
        """
        Ciclo completo percezione -> cognizione -> azione -> transizione,
        eseguito in un solo colpo (LLM chiamato qui direttamente).

        Equivalente a prepare_step() + chiamata LLM + finalize_step().
        Usalo per test, dry-run, MockLLMClient, o ovunque non serva
        parallelizzare. Per grafi grandi con backend reale, l'Orchestratore
        chiama prepare_step()/finalize_step() separatamente per dispacciare
        le chiamate LLM di piu' agenti su un thread pool — vedi
        SimulationOrchestrator._agent_cycle.

        Parameters
        ----------
        current_step : int
            Step temporale corrente (t).
        network_manager : NetworkManager
            Grafo dinamico — fonte di verità per feed, stati, embedding.

        Returns
        -------
        AgentDecision
            Decisione completa da usare per logging e aggiornamento del grafo.
        """
        ctx = self.prepare_step(current_step, network_manager)

        try:
            response = self._llm.chat(ctx.messages)
        except RuntimeError:
            raise  # Token budget esaurito — propaga all'Orchestratore
        except Exception as exc:
            logger.error("[Agent %d] LLM error: %s", self.node_id, exc)
            response = None

        return self.finalize_step(ctx, response, network_manager)

    def prepare_step(
        self,
        current_step: int,
        network_manager: "NetworkManager",
        all_states: dict[int, str] | None = None,
    ) -> AgentStepContext:
        """
        Percezione + costruzione messaggi — NESSUNA chiamata LLM qui dentro.

        Tutte le letture da NetworkManager avvengono qui. Pensato per essere
        chiamato in un loop sequenziale dal thread principale dell'Orchestratore,
        PRIMA di dispacciare le chiamate LLM (lente, I/O di rete) su un thread
        pool: in questo modo NetworkManager non viene mai toccato da thread
        diversi da quello principale.

        Parameters
        ----------
        all_states : dict[int, str] | None
            Snapshot pre-calcolato di network_manager.get_all_states().
            Se l'Orchestratore chiama prepare_step() per centinaia di
            migliaia di nodi nello stesso step, ricalcolare get_all_states()
            ad ogni chiamata rischia di essere O(n) per chiamata e quindi
            O(n^2) sull'intero step. Passandolo gia' calcolato una volta
            sola si evita il problema. Se None (uso standalone via step()),
            viene recuperato qui come prima.
        """
        old_state = self._state.value

        # 1. Percezione
        feed = network_manager.get_feed(self.node_id, window=self._memory_window)
        neighbours = network_manager.neighbours(self.node_id)
        if all_states is None:
            all_states = network_manager.get_all_states()
        nb_state_counts = StateMachine.get_neighbour_state_counts(
            self.node_id, all_states, neighbours
        )

        # 2. Costruzione prompt (nessuna chiamata LLM)
        user_prompt = build_user_prompt(
            feed=feed,
            state=self._state.value,
            step=current_step,
            neighbour_states=nb_state_counts,
            memory_window=self._memory_window,
        )

        messages = [{"role": "system", "content": self._system_prompt}]
        if self._state_update_note:
            messages.append({"role": "system", "content": self._state_update_note})
            self._state_update_note = ""
        messages.append({"role": "user", "content": user_prompt})

        return AgentStepContext(
            old_state=old_state,
            current_step=current_step,
            messages=messages,
            nb_state_counts=nb_state_counts,
        )

    def finalize_step(
        self,
        ctx: AgentStepContext,
        response: "LLMResponse | None",
        network_manager: "NetworkManager",
    ) -> AgentDecision:
        """
        Parsing, transizione di stato, scrittura su NetworkManager — a
        partire da una risposta LLM gia' ottenuta (es. da un thread pool).

        IMPORTANTE: va sempre chiamato dal thread principale, nello stesso
        ordine usato per prepare_step(), per evitare scritture concorrenti
        su NetworkManager (add_post, perturb_embedding, set_state non sono
        garantite thread-safe e qui assumiamo non lo siano).
        """
        old_state = ctx.old_state
        current_step = ctx.current_step
        nb_state_counts = ctx.nb_state_counts

        # 3. Parsing output
        if response is not None:
            llm_output, parse_fallback = extract_json(response.content)
            # response.is_fallback e' True quando LLMClient ha esaurito i retry
            # e ha restituito il fallback come JSON valido (quindi extract_json
            # non lo rileva come tale, perche' il parsing in se' riesce).
            # Senza l'OR, questi casi venivano contati come risposte "vere".
            is_fallback = response.is_fallback or parse_fallback
            tokens_in = response.input_tokens
            tokens_out = response.output_tokens
        else:
            from src.agents.llm_client import FALLBACK_AGENT_OUTPUT
            llm_output = FALLBACK_AGENT_OUTPUT.copy()
            is_fallback = True
            tokens_in = tokens_out = 0

        # Clamp susceptibility
        susc = max(0.0, min(1.0, float(llm_output.get("susceptibility", 0.5))))
        opinion = str(llm_output.get("opinion", "")).strip()[:200]
        spread_intent = bool(llm_output.get("spread_intent", False))
        reasoning = str(llm_output.get("reasoning", ""))

        # 4. Transizione di stato
        transition = self._sm.transition(
            current_state=self._state,
            node_id=self.node_id,
            neighbour_states=nb_state_counts,
            llm_output=llm_output,
        )
        new_state_enum = transition.new_state
        state_changed = transition.changed

        # 5. Azione — pubblica post e aggiorna NetworkManager
        if opinion:
            network_manager.add_post(self.node_id, {
                "node_id": self.node_id,
                "step": current_step,
                "content": opinion,
                "author_state": old_state,
            })

        # 6. Perturbazione embedding
        delta = self._compute_embedding_delta(susc, new_state_enum.value)
        network_manager.perturb_embedding(self.node_id, delta)
        delta_norm = float(np.linalg.norm(delta))

        # 7. Aggiorna stato nel NetworkManager e nell'agente
        if state_changed:
            self._state = new_state_enum
            network_manager.set_state(self.node_id, new_state_enum.value)
            self._state_update_note = build_state_update_note(old_state, new_state_enum.value)

        logger.debug(
            "[Agent %d] step=%d | %s->%s | susc=%.2f | opinion=%s",
            self.node_id, current_step, old_state, new_state_enum.value,
            susc, repr(opinion[:40]) if opinion else "silent",
        )

        return AgentDecision(
            node_id=self.node_id,
            step=current_step,
            old_state=old_state,
            new_state=new_state_enum.value,
            state_changed=state_changed,
            opinion=opinion,
            reasoning=reasoning,
            susceptibility=susc,
            spread_intent=spread_intent,
            infection_pressure=transition.infection_pressure,
            effective_threshold=transition.effective_threshold,
            embedding_delta_norm=delta_norm,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            is_fallback=is_fallback,
        )

    # ------------------------------------------------------------------
    # Embedding perturbation
    # ------------------------------------------------------------------

    def _compute_embedding_delta(self, susceptibility: float, new_state: str) -> np.ndarray:
        """
        Calcola il vettore di perturbazione dell'embedding.

        Logica:
          - Agenti che si infettano spingono l'embedding nella direzione
            della "narrazione polarizzante" (direction positiva).
          - Agenti resistenti spingono nella direzione opposta.
          - Magnitudine proporzionale all'intensita' della suscettibilita'.
        """
        magnitude = abs(susceptibility - 0.5) * self._EMBEDDING_SCALE

        if new_state in ("I", "F"):
            # Infetto o Fact-Checker: perturbazione nella direzione di infezione
            sign = 1.0 if new_state == "I" else -1.0
        elif new_state == "R":
            sign = -1.0   # Resistente: direzione opposta
        else:
            sign = 0.0    # Suscettibile neutro: nessuna perturbazione netta

        return (self._infection_direction * magnitude * sign).astype(np.float32)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Agent(id={self.node_id}, state={self._state.value})"