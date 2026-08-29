"""Slip injection. The environment sometimes executes an action it was not given.

A pure function over an action array, taking the action-space size from the
caller so it stays environment-agnostic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Folded into a step key to derive the slip key. Any fixed value serves.
SLIP_KEY_FOLD_INDEX: int = 1


def apply_slip(
    action: jax.Array,
    rng: jax.Array,
    slip_probability: float,
    num_actions: int,
) -> jax.Array:
    """Return the action the environment executes, which may not be the one commanded.

    With probability slip_probability each element is resampled uniformly over
    the whole action space, the commanded action included. The executed action
    therefore differs from the commanded one at a rate of
    slip_probability * (1 - 1 / num_actions), not slip_probability. The
    marginal action distribution stays uniform under a uniform policy.

    Args:
        action: Commanded actions, shape (n,).
        rng: PRNG key for this step's draw.
        slip_probability: Probability each element is resampled. Zero returns
            the input unchanged and draws nothing.
        num_actions: Size of the discrete action space.

    Returns:
        The executed actions, same shape and dtype as the input.

    Raises:
        ValueError: If slip_probability falls outside [0, 1], or num_actions is
            not positive.
    """
    if not 0.0 <= slip_probability <= 1.0:
        raise ValueError(
            f"slip_probability must lie in [0, 1], got {slip_probability}"
        )
    if num_actions <= 0:
        raise ValueError(f"num_actions must be positive, got {num_actions}")
    if slip_probability == 0.0:
        return action
    resample_key, substitute_key = jax.random.split(rng)
    resample = jax.random.uniform(resample_key, action.shape) < slip_probability
    substitute = jax.random.randint(substitute_key, action.shape, 0, num_actions)
    return jnp.where(resample, substitute, action).astype(action.dtype)
