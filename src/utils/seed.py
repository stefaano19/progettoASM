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

Note
----
Su alcuni ambienti (es. Kaggle) numpy.random puo' essere rotto per
binary incompatibility. Questo modulo aggira il problema importando
il Generator direttamente dal modulo C, senza passare per la catena
numpy.random -> _pickle -> mtrand che causa il crash.
Usare sempre ``safe_default_rng()`` al posto di ``np.random.default_rng()``.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risoluzione numpy.random binary incompatibility
# ---------------------------------------------------------------------------
# np.random.default_rng() passa per numpy/random/__init__.py che importa
# _pickle.py -> mtrand.pyx, e quest'ultimo crasha se i dtype C sono
# incompatibili.  Importando direttamente _generator e _pcg64 evitiamo
# quella catena.
# ---------------------------------------------------------------------------
_NP_RANDOM_OK = True
try:
    # Test: se funziona, usiamo la via normale
    np.random.default_rng(0)
except Exception:
    _NP_RANDOM_OK = False
    try:
        from numpy.random._generator import Generator as _Generator
        from numpy.random._pcg64 import PCG64 as _PCG64
        logger.warning(
            "[Seed] numpy.random rotto (binary incompatibility). "
            "Uso import diretto di Generator/PCG64 come workaround."
        )
    except Exception:
        # Anche l'import diretto fallisce — numpy e' completamente rotto
        _Generator = None  # type: ignore[assignment,misc]
        _PCG64 = None  # type: ignore[assignment,misc]
        logger.error(
            "[Seed] numpy.random completamente inutilizzabile. "
            "Il seeding numpy e' disabilitato."
        )


def safe_default_rng(seed: int | None = None) -> np.random.Generator:
    """
    Drop-in replacement per ``np.random.default_rng()`` che funziona
    anche quando numpy.random e' rotto per binary incompatibility.

    Usare questa funzione ovunque al posto di ``np.random.default_rng()``.
    """
    if _NP_RANDOM_OK:
        return np.random.default_rng(seed)
    if _Generator is not None and _PCG64 is not None:
        return _Generator(_PCG64(seed))
    raise RuntimeError(
        "numpy.random e' completamente rotto in questo ambiente. "
        "Esegui: !pip install --force-reinstall numpy && riavvia il kernel."
    )


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

    # Numpy seeding
    if _NP_RANDOM_OK:
        np.random.seed(seed)
    else:
        # Il legacy seeding (np.random.seed) non funziona, ma possiamo
        # comunque creare Generator isolati con safe_default_rng().
        logger.warning(
            "[Seed] np.random.seed() saltato (binary incompatibility). "
            "I Generator creati con safe_default_rng() funzionano comunque."
        )

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
    return safe_default_rng(seed)
