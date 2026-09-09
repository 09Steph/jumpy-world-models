"""Reconstruction losses and their observation-target helpers.

A discrete observation is scored by per-cell categorical cross-entropy. A
continuous one is scored by squared error on values rescaled from their
declared bounds to [0, 1]. Both sum over cells and channels rather than
averaging, so both are totals per example, and the two are in different units
and never comparable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

def observation_class_targets(
    obs: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Reshape flattened observations into integer class indices on the grid.

    The cast truncates, so the stored values must be integral.

    Args:
        obs: Flattened observations, shape (..., obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Integer class indices, shape (..., height, width, channels).
    """
    height, width = grid_shape
    return obs.reshape(*obs.shape[:-1], height, width, -1).astype(jnp.int32)


def reconstruction_loss(
    logits: list[jax.Array], obs: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Per-cell categorical cross-entropy, summed over cells and channels.

    Summed, not averaged, so the value scales with the number of grid cells.

    Args:
        logits: Per-channel logits from a grid decoder, each
            (..., height, width, classes).
        obs: Flattened observations, shape (..., obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Summed cross-entropy per element, shape obs.shape[:-1].
    """
    targets = observation_class_targets(obs, grid_shape)
    total = jnp.zeros(obs.shape[:-1])
    for channel, channel_logits in enumerate(logits):
        log_probs = jax.nn.log_softmax(channel_logits, axis=-1)
        picked = jnp.take_along_axis(
            log_probs, targets[..., channel][..., None], axis=-1
        )[..., 0]
        total = total - jnp.sum(picked, axis=(-2, -1))
    return total


def observation_pixel_targets(
    obs: jax.Array, grid_shape: tuple[int, int], value_range: tuple[int, int]
) -> jax.Array:
    """Reshape flattened observations onto the grid and normalise to [0, 1].

    Args:
        obs: Flattened observations, shape (..., obs_dim).
        grid_shape: Spatial grid shape, (height, width).
        value_range: Inclusive (low, high) bounds of the stored values.

    Returns:
        Normalised values, shape (..., height, width, channels).
    """
    low, high = value_range
    height, width = grid_shape
    grid = obs.reshape(*obs.shape[:-1], height, width, -1)
    return (grid.astype(jnp.float32) - low) / (high - low)


def pixel_reconstruction_loss(
    prediction: jax.Array,
    obs: jax.Array,
    grid_shape: tuple[int, int],
    value_range: tuple[int, int],
) -> jax.Array:
    """Squared error on normalised values, summed over cells and channels.

    Summed, not averaged, matching reconstruction_loss.

    Args:
        prediction: Decoder output in [0, 1], shape
            (..., height, width, channels).
        obs: Flattened observations, shape (..., obs_dim).
        grid_shape: Spatial grid shape, (height, width).
        value_range: Inclusive (low, high) bounds of the stored values.

    Returns:
        Summed squared error per element, shape obs.shape[:-1].
    """
    targets = observation_pixel_targets(obs, grid_shape, value_range)
    return jnp.sum((prediction - targets) ** 2, axis=(-3, -2, -1))


def check_observation_values_in_bounds(
    obs: jax.Array, grid_shape: tuple[int, int], value_range: tuple[int, int]
) -> None:
    """Raise if any observation value falls outside its declared bounds.

    The continuous counterpart to check_observation_values_in_range. A host-side
    check on a sampled batch, not inside a jitted step.

    Args:
        obs: Flattened observations, shape (..., obs_dim).
        grid_shape: Spatial grid shape, (height, width).
        value_range: Inclusive (low, high) bounds of the stored values.

    Raises:
        ValueError: If any value lies outside the declared bounds.
    """
    low, high = value_range
    height, width = grid_shape
    grid = obs.reshape(*obs.shape[:-1], height, width, -1)
    lowest = float(jnp.min(grid))
    highest = float(jnp.max(grid))
    if lowest < low or highest > high:
        raise ValueError(
            f"observation holds values in [{lowest}, {highest}], outside the "
            f"declared [{low}, {high}] range. See "
            f"config.CONTINUOUS_VALUE_RANGES"
        )


def check_observation_values_in_range(
    obs: jax.Array, grid_shape: tuple[int, int], channel_classes: tuple[int, ...]
) -> None:
    """Raise if any observation code falls outside its channel's class range.

    A host-side check on a sampled batch, not inside a jitted step.

    Args:
        obs: Flattened observations, shape (..., obs_dim).
        grid_shape: Spatial grid shape, (height, width).
        channel_classes: Number of classes per channel.

    Raises:
        ValueError: If any code is negative or at least its channel's class
            count.
    """
    targets = observation_class_targets(obs, grid_shape)
    for channel, num_classes in enumerate(channel_classes):
        values = targets[..., channel]
        lowest = int(jnp.min(values))
        highest = int(jnp.max(values))
        if lowest < 0 or highest >= num_classes:
            raise ValueError(
                f"observation channel {channel} holds values in "
                f"[{lowest}, {highest}], outside the configured "
                f"[0, {num_classes}) class range. See "
                f"config.OBS_CHANNEL_CLASSES_NAVIX"
            )
