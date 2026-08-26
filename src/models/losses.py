"""Reconstruction loss and its observation-target helpers.

Per-cell categorical cross-entropy over the three categorical observation
channels.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

def observation_class_targets(
    obs: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Reshape flattened observations into integer class indices on the grid.

    Observations are stored as small integer codes and flattened to float for
    the encoder. The values are exactly representable in float32, so the cast
    back is lossless.

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

    Summed, not averaged, so the value scales with the number of grid cells and
    is much larger than a per-cell mean.

    Args:
        logits: Per-channel logits from GridDecoder, each
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


def check_observation_values_in_range(
    obs: jax.Array, grid_shape: tuple[int, int], channel_classes: tuple[int, ...]
) -> None:
    """Raise if any observation code falls outside its channel's class range.

    A host-side check on a sampled batch, not inside a jitted step. It costs
    one host sync per call.

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
