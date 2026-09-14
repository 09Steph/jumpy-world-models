"""The reported metric suite for a trained model.

Observations arrive flattened. Every routine reshapes them onto `grid_shape`
through `observation_class_targets` or `observation_pixel_targets`, so the
channel count comes from the data, never a literal.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from config import AGENT_CHANNEL_INDEX
from src.models.losses import (
    observation_class_targets,
    observation_pixel_targets,
)
from src.pipeline.metrics_schema import (
    COPY_MOVER_MSE_KEY,
    MOVER_CHANGED_PIXELS_KEY,
    MOVER_MSE_KEY,
)

# Keys of mover_restricted_accuracy's result. MOVER_EXCLUDED_KEY is shared with
# mover_restricted_mse, and the changed-cells mean counts cell-channels.
MOVER_ACCURACY_KEY: str = "mover_restricted_accuracy"
MOVER_CHANGED_CELLS_KEY: str = "mean_changed_cells"
MOVER_EXCLUDED_KEY: str = "excluded_fraction"


def _predictions(logits: list[jax.Array]) -> list[jax.Array]:
    """Return the argmax class index per channel.

    Args:
        logits: Per-channel logits, each (..., height, width, classes).

    Returns:
        One integer array per channel, each (..., height, width).
    """
    return [jnp.argmax(channel, axis=-1) for channel in logits]


def _assert_one_array_per_channel(
    logits: list[jax.Array], truth: jax.Array
) -> None:
    """Raise unless the prediction holds one array per observation channel.

    Every categorical metric calls this. A continuous prediction is a
    one-element list whose array's last axis is channels, and argmaxing that
    axis as though it were classes returns a plausible number. The check
    compares counts, so on a one-channel observation it cannot tell a
    continuous prediction from a categorical one.

    Args:
        logits: The prediction being scored.
        truth: Class indices, shape (..., height, width, channels).

    Raises:
        ValueError: If the prediction does not hold one array per channel.
    """
    channels = int(truth.shape[-1])
    if len(logits) == channels:
        return
    raise ValueError(
        f"a categorical metric was given {len(logits)} prediction array(s) for "
        f"an observation with {channels} channel(s). A continuous prediction "
        f"is one array whose last axis is channels, not classes, and it has no "
        f"categorical reading. Score it with model_mse instead."
    )


def _correct_mask(
    logits: list[jax.Array], targets: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Return a boolean array marking every correctly predicted cell-channel.

    The accuracy metrics below all read this one comparison.

    Args:
        logits: Per-channel logits, each (..., height, width, classes).
        targets: Flattened end observations, shape (..., obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Boolean array of shape (..., height, width, channels).
    """
    truth = observation_class_targets(targets, grid_shape)
    _assert_one_array_per_channel(logits, truth)
    stacked = jnp.stack(_predictions(logits), axis=-1)
    return stacked == truth


def per_cell_accuracy(
    logits: list[jax.Array], targets: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Return the fraction of cell-channels whose argmax matches the target.

    Scored over every cell, so a predictor that copies its input scores highly
    wherever little of the grid changes.

    Args:
        logits: Per-channel logits, each (batch, height, width, classes).
        targets: Flattened end observations, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Scalar accuracy over every cell, channel and example.

    Raises:
        ValueError: If the prediction does not hold one array per observation
            channel.
    """
    return jnp.mean(_correct_mask(logits, targets, grid_shape).astype(jnp.float32))


def exact_grid_match_rate(
    logits: list[jax.Array], targets: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Return the fraction of examples predicted perfectly in every cell.

    One wrong cell-channel scores the whole example zero.

    Args:
        logits: Per-channel logits, each (batch, height, width, classes).
        targets: Flattened end observations, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Scalar rate over the batch.

    Raises:
        ValueError: If the prediction does not hold one array per observation
            channel.
    """
    correct = _correct_mask(logits, targets, grid_shape)
    per_example = jnp.all(correct, axis=(-3, -2, -1))
    return jnp.mean(per_example.astype(jnp.float32))


def agent_position_accuracy(
    logits: list[jax.Array], targets: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Return per-cell accuracy on channel AGENT_CHANNEL_INDEX.

    That is the object-type channel, in which the agent is one class of
    several. Every cell of the channel is scored, not the agent's cell alone.
    The evaluate stage records it on top-down runs only, since the egocentric
    view shows floor in the agent's cell and the agent class never appears.

    Args:
        logits: Per-channel logits, each (batch, height, width, classes).
        targets: Flattened end observations, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Scalar accuracy over the agent channel.

    Raises:
        ValueError: If the prediction does not hold one array per observation
            channel.
    """
    correct = _correct_mask(logits, targets, grid_shape)
    return jnp.mean(correct[..., AGENT_CHANNEL_INDEX].astype(jnp.float32))


def model_cross_entropy(
    logits: list[jax.Array], targets: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Return mean per-cell cross-entropy, summed over channels.

    Mean per cell, where `reconstruction_loss` sums. Must stay the same
    quantity as `baselines.copy_cross_entropy`.

    Args:
        logits: Per-channel logits, each (batch, height, width, classes).
        targets: Flattened end observations, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Scalar mean per-cell cross-entropy in nats, summed over channels.

    Raises:
        ValueError: If the prediction does not hold one array per observation
            channel.
    """
    truth = observation_class_targets(targets, grid_shape)
    _assert_one_array_per_channel(logits, truth)
    total = jnp.zeros(())
    for channel, channel_logits in enumerate(logits):
        log_probs = jax.nn.log_softmax(channel_logits, axis=-1)
        picked = jnp.take_along_axis(
            log_probs, truth[..., channel][..., None], axis=-1
        )[..., 0]
        total = total - jnp.mean(picked)
    return total


def mover_mask(
    states: jax.Array, targets: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Return which cell-channels change between the start and end observation.

    Takes stored values, not normalised ones. Both are cast to integers.

    Args:
        states: Flattened start observations s_t, shape (batch, obs_dim).
        targets: Flattened end observations s_{t+h}, same shape as states.
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Boolean mask of shape (batch, height, width, channels).
    """
    start = observation_class_targets(states, grid_shape)
    end = observation_class_targets(targets, grid_shape)
    return start != end


def model_mse(
    prediction: jax.Array,
    targets: jax.Array,
    grid_shape: tuple[int, int],
    value_range: tuple[int, int],
) -> jax.Array:
    """Return mean squared error per cell-channel on normalised values.

    Mean, where `pixel_reconstruction_loss` sums. Must stay the same quantity
    as `baselines.copy_mse`.

    Args:
        prediction: Decoder output in [0, 1], shape
            (batch, height, width, channels).
        targets: Flattened end observations, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).
        value_range: Inclusive (low, high) bounds of the stored values.

    Returns:
        Scalar mean squared error on the [0, 1] scale.
    """
    truth = observation_pixel_targets(targets, grid_shape, value_range)
    return jnp.mean((prediction - truth) ** 2)


def mover_restricted_accuracy(
    logits: list[jax.Array],
    targets: jax.Array,
    states: jax.Array,
    grid_shape: tuple[int, int],
) -> dict:
    """Return accuracy over only the cell-channels that change.

    The mask is per cell-channel and taken from ground truth, so a pure copy
    scores exactly zero and the prediction cannot move the mask. Examples with
    no change are excluded and their fraction reported.

    Args:
        logits: Per-channel logits, each (batch, height, width, classes).
        targets: Flattened end observations s_{t+h}, shape (batch, obs_dim).
        states: Flattened start observations s_t, same shape as targets.
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Mapping with MOVER_ACCURACY_KEY, MOVER_CHANGED_CELLS_KEY and
        MOVER_EXCLUDED_KEY. The accuracy is None when every example was
        excluded.
    """
    moved = mover_mask(states, targets, grid_shape)
    correct = _correct_mask(logits, targets, grid_shape)

    per_example_moved = jnp.sum(moved, axis=(-3, -2, -1))
    kept = per_example_moved > 0
    num_kept = int(jnp.sum(kept))
    num_examples = int(kept.shape[0])

    scored = jnp.sum(jnp.where(moved & correct, 1.0, 0.0), axis=(-3, -2, -1))
    # Pooled over every changed cell-channel, not averaged per example.
    accuracy = (
        float(jnp.sum(scored) / jnp.sum(per_example_moved)) if num_kept else None
    )
    return {
        MOVER_ACCURACY_KEY: accuracy,
        # Averaged over every example, excluded ones included.
        MOVER_CHANGED_CELLS_KEY: float(jnp.mean(per_example_moved)),
        MOVER_EXCLUDED_KEY: (
            float((num_examples - num_kept) / num_examples) if num_examples else 0.0
        ),
    }


def mover_restricted_mse(
    prediction: jax.Array,
    targets: jax.Array,
    states: jax.Array,
    grid_shape: tuple[int, int],
    value_range: tuple[int, int],
) -> dict:
    """Return MSE over only the pixel-channels that change, model and copy.

    The continuous counterpart of `mover_restricted_accuracy`. The mask is
    `mover_mask` on the stored ground-truth values before normalisation, so
    any change of stored value counts and the prediction cannot move the mask.
    The copy's error on the same set is returned beside the model's. Both are
    pooled over every changed pixel-channel, and examples with no change are
    excluded with their fraction reported.

    Args:
        prediction: Decoder output in [0, 1], shape
            (batch, height, width, channels).
        targets: Flattened end observations s_{t+h}, shape (batch, obs_dim).
        states: Flattened start observations s_t, same shape as targets.
        grid_shape: Spatial grid shape, (height, width).
        value_range: Inclusive (low, high) bounds of the stored values.

    Returns:
        Mapping with MOVER_MSE_KEY, COPY_MOVER_MSE_KEY,
        MOVER_CHANGED_PIXELS_KEY and MOVER_EXCLUDED_KEY. Both errors are None
        when every example was excluded.
    """
    moved = mover_mask(states, targets, grid_shape)
    truth = observation_pixel_targets(targets, grid_shape, value_range)
    copy = observation_pixel_targets(states, grid_shape, value_range)

    per_example_moved = jnp.sum(moved, axis=(-3, -2, -1))
    num_kept = int(jnp.sum(per_example_moved > 0))
    num_examples = int(per_example_moved.shape[0])
    total_moved = jnp.sum(per_example_moved)

    def pooled(squared: jax.Array) -> float | None:
        """Mean of the squared errors over the changed pixel-channels."""
        if not num_kept:
            return None
        return float(jnp.sum(jnp.where(moved, squared, 0.0)) / total_moved)

    return {
        MOVER_MSE_KEY: pooled((prediction - truth) ** 2),
        COPY_MOVER_MSE_KEY: pooled((copy - truth) ** 2),
        MOVER_CHANGED_PIXELS_KEY: float(jnp.mean(per_example_moved)),
        MOVER_EXCLUDED_KEY: (
            float((num_examples - num_kept) / num_examples) if num_examples else 0.0
        ),
    }
