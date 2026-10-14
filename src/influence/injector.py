"""
Iniezione di agenti Fact-Checker nel grafo tramite CELF.

Il FactCheckerInjector si occupa di:
  1. Decidere quando attivare CELF (basandosi su infection_rate e celf_interval).
  2. Iniettare lo stato 'F' nei nodi seed selezionati da CELF.
  3. Aggiungere un post iniziale di fact-checking per avviare la diffusione.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.graph.network_manager import NetworkManager
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class FactCheckerInjector:
    """Gestisce l'iniezione degli agenti Fact-Checker nel grafo."""

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        self._activation_threshold = cfg.influence.activation_threshold
        self._celf_interval = cfg.influence.celf_interval
        self._topic = cfg.simulation.topic
        self._injection_log: list[dict] = []

    def should_activate(self, step: int, infection_rate: float) -> bool:
        """
        Determina se CELF deve girare a questo step.

        Condizioni di attivazione (entrambe devono essere vere):
          1. step % celf_interval == 0
          2. infection_rate >= activation_threshold
        """
        interval_ok = (self._celf_interval > 0) and (step % self._celf_interval == 0)
        threshold_ok = infection_rate >= self._activation_threshold

        if interval_ok and threshold_ok:
            logger.info(
                "[Injector] CELF attivato: step=%d | infection_rate=%.3f >= threshold=%.3f",
                step, infection_rate, self._activation_threshold,
            )
        elif interval_ok:
            logger.debug(
                "[Injector] Step %d: intervallo OK ma infection_rate=%.3f < threshold=%.3f",
                step, infection_rate, self._activation_threshold,
            )

        return interval_ok and threshold_ok

    def inject(
        self,
        network_manager: "NetworkManager",
        celf_seeds: list[int],
        step: int = 0,
    ) -> list[int]:
        """
        Inietta lo stato 'F' nei nodi seed selezionati da CELF.

        Salta automaticamente nodi gia' in stato 'F' o 'R'.
        Per ogni nodo iniettato aggiunge un post di fact-checking iniziale
        visibile ai vicini.
        """
        injected: list[int] = []
        skipped: list[int] = []

        for node_id in celf_seeds:
            current_state = network_manager.get_state(node_id)

            if current_state in ("F", "R"):
                logger.debug(
                    "[Injector] Nodo %d gia' in stato '%s' - skip.",
                    node_id, current_state,
                )
                skipped.append(node_id)
                continue

            # Imposta stato Fact-Checker
            network_manager.set_state(node_id, "F")

            # Post iniziale di fact-checking
            content = (
                f"[FACT-CHECK seed] I have been designated to critically examine "
                f"claims about '{self._topic}'. Let us evaluate the evidence together "
                f"and distinguish facts from misinformation."
            )
            network_manager.add_post(node_id, {
                "node_id": node_id,
                "step": step,
                "content": content,
                "author_state": "F",
            })

            injected.append(node_id)
            self._injection_log.append({
                "step": step,
                "node_id": node_id,
                "previous_state": current_state,
            })

        logger.info(
            "[Injector] Fact-checker iniettati: %d | saltati: %d | nodi: %s",
            len(injected), len(skipped), injected,
        )

        return injected

    @property
    def injection_history(self) -> list[dict]:
        """Storico di tutte le iniezioni effettuate."""
        return list(self._injection_log)

    def total_injected(self) -> int:
        """Numero totale di fact-checker iniettati dall'inizio."""
        return len(self._injection_log)
