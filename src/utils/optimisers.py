"""Optimiser construction for the training loop.

One chain, clip then Adam. Clipping the raw gradient bounds the quantity the
published settings bound, and reversing the two would clip an
already-adaptively-scaled update.
"""

from __future__ import annotations

import optax

from config import OPTIMISER_ADAM, TrainConfig
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def cosine_schedule(config: TrainConfig) -> optax.Schedule:
    """Return the cosine annealing schedule.

    optax takes `alpha`, a fraction of the peak, so this converts the absolute
    final rate config states. The period is `total_steps`.

    Args:
        config: Training configuration carrying the two rates and the period.

    Returns:
        An optax schedule from `learning_rate` down to `lr_schedule_final` over
        `total_steps`.
    """
    return optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.total_steps,
        alpha=config.lr_schedule_final / config.learning_rate,
    )


def build_optimiser(config: TrainConfig) -> optax.GradientTransformation:
    """Return the optimiser chain: gradient clipping, then Adam on a schedule.

    Adam, not AdamW, matching the published comparator, **whose settings were
    measured on continuous control and pixel benchmarks, not a discrete symbolic
    grid.** `optax.chain` applies left to right, so clipping
    runs first on the raw gradient.

    Args:
        config: Training configuration carrying the optimisation fields. Its
            `__post_init__` has already rejected an unbuildable combination.

    Returns:
        The composed optax chain, clip -> Adam(schedule).

    Raises:
        ValueError: If the optimiser name is not one this function builds.
            Unreachable through a constructed TrainConfig, and kept so a
            hand-built one fails loudly instead of silently getting Adam.
    """
    if config.optimiser_name != OPTIMISER_ADAM:
        raise ValueError(
            f"build_optimiser has no branch for {config.optimiser_name!r}. "
            "OPTIMISERS has grown and this function has not."
        )
    schedule = cosine_schedule(config)
    logger.info(
        "optimiser %s lr %.3g -> %.3g over %d steps, grad clip %.3g",
        config.optimiser_name,
        config.learning_rate,
        config.lr_schedule_final,
        config.total_steps,
        config.grad_clip_norm,
    )
    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(learning_rate=schedule),
    )
