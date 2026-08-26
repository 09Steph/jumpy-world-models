"""Apple-Silicon-only JAX backend override.

`jax-metal` cannot run this project's ops, so the local backend is forced to
CPU with ``JAX_PLATFORMS=cpu``. The override is conditional. The cluster's
Linux hosts need CUDA.

Dependency-free, importing only ``os`` and ``platform``, so it is safe to call
before any other module's ``import jax``.
"""

from __future__ import annotations

import os
import platform


def is_apple_silicon() -> bool:
    """Return whether this process is running on Apple Silicon (Darwin).

    The single ground truth for which platform gets the CPU override.
    """
    return platform.system() == "Darwin"


def force_cpu_backend_on_apple_silicon() -> None:
    """Set ``JAX_PLATFORMS=cpu``, but only when running on Apple Silicon.

    Idempotent. **Must be called before ``import jax`` anywhere in the
    process**. JAX reads the variable when its backend first initialises.

    A no-op off Apple Silicon, so CUDA hosts auto-select their own backend.
    """
    if is_apple_silicon():
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
