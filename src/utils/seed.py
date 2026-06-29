"""
src/utils/seed.py
=================
Seed manager centralizzato per la riproducibilita' completa.

Setta il seed su: random, numpy, torch (CPU + CUDA se disponibile).
Il seed viene letto da config.yaml (execution.random_seed).

Utilizzo
--------
    from src.utils.seed import set_all_seeds
    set_all_seeds(42)
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """
    Imposta il seed su tutti i motori di randomness usati nel progetto.

    Parameters
    ----------
    seed : int
        Il valore del seed. Usa sempre lo stesso valore per riproducibilita'.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # PyTorch (opzionale — non installato in Fase 0)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Determinismo completo (piu' lento ma riproducibile)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug("[Seed] PyTorch seed impostato: %d", seed)
    except ImportError:
        logger.debug("[Seed] PyTorch non disponibile — skip torch seed.")

    logger.info("[Seed] Tutti i seed impostati a: %d", seed)


def get_rng(seed: int | None = None) -> random.Random:
    """
    Restituisce un'istanza isolata di random.Random con seed fisso.
    Utile per sampling riproducibile senza alterare lo stato globale.
    """
    rng = random.Random(seed)
    return rng


def get_np_rng(seed: int | None = None) -> np.random.Generator:
    """
    Restituisce un numpy Generator isolato (API moderna).
    Preferire questo a np.random.seed() nelle funzioni.
    """
    return np.random.default_rng(seed)
