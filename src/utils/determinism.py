"""Determinism control.

Seeds the framework-agnostic sources once at startup and supplies the root JAX
PRNG key every stochastic operation draws from.

JAX carries no global random state. Reproducibility comes from threading an
explicit key and splitting it.
"""

from __future__ import annotations

import os
import random

import jax
import numpy as np

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def set_determinism(seed: int) -> None:
    """Seed the framework-agnostic sources of randomness for a reproducible run.

    Seeds Python's ``random``, NumPy and ``PYTHONHASHSEED``. JAX is seeded
    separately through :func:`jax_prng_key`.

    Args:
        seed: The seed to apply. One of config.SEEDS.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info("determinism set for seed %s (python, numpy)", seed)


def describe_active_backend() -> str:
    """Return a human-readable description of the active JAX backend.

    Reports what `jax.default_backend()` and `jax.devices()` resolved to at
    runtime, for a one-line startup log so a run's device is in its log.

    Returns:
        A string like "gpu (1 device: cuda:0)".
    """
    backend = jax.default_backend()
    devices = jax.devices(backend)
    device_list = ", ".join(str(d) for d in devices)
    return f"{backend} ({len(devices)} device(s): {device_list})"


def jax_prng_key(seed: int) -> jax.Array:
    """Return the root JAX PRNG key for a run.

    Every stochastic operation derives from this key by splitting. Two runs
    with the same seed produce the same draws on a given backend.

    Args:
        seed: The seed to apply. One of config.SEEDS.

    Returns:
        The root PRNG key, to be split by callers. Never reuse one key for two
        draws.
    """
    return jax.random.PRNGKey(seed)
