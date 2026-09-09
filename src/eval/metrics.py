"""The reported metric suite for a trained model.

Every routine reads its shape from its arguments, never a literal. The channel
count comes from `len(logits)` and the grid from `grid_shape`.

Observations arrive flattened, as `reconstruction_loss` takes them, and the
decomposition comes from `observation_class_targets`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from config import AGENT_CHANNEL_INDEX
from src.models.losses import (
    observation_class_targets,
    observation_pixel_targets,
)

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

    A categorical prediction is one logits array per channel, so its last axis
    is classes. A continuous prediction is a single array over all channels, so
    its last axis is channels. Both are lists, and every metric here iterates
    the list and argmaxes the last axis, so a continuous prediction reaching
    one of them scores its channels as though they were classes of channel
    zero and returns a plausible number.

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

    One place the comparison happens. The categorical metrics below are
    readings of this one array.

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

    Over all cells, so it is not a headline. Almost none of a grid changes
    between s_t and s_{t+h}, so a predictor that copies its input scores highly.

    Args:
        logits: Per-channel logits, each (batch, height, width, classes).
        targets: Flattened end observations, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Scalar accuracy over every cell, channel and example.

    Raises:
        ValueError: If given a continuous prediction, which has no
            categorical reading.
    """
    return jnp.mean(_correct_mask(logits, targets, grid_shape).astype(jnp.float32))


def exact_grid_match_rate(
    logits: list[jax.Array], targets: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Return the fraction of examples predicted perfectly in every cell.

    The strictest reading, and the one that falls off fastest with horizon. A
    single misplaced agent scores zero for the whole example.

    Args:
        logits: Per-channel logits, each (batch, height, width, classes).
        targets: Flattened end observations, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Scalar rate over the batch.

    Raises:
        ValueError: If given a continuous prediction, which has no
            categorical reading.
    """
    correct = _correct_mask(logits, targets, grid_shape)
    per_example = jnp.all(correct, axis=(-3, -2, -1))
    return jnp.mean(per_example.astype(jnp.float32))


def agent_position_accuracy(
    logits: list[jax.Array], targets: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Return accuracy restricted to the agent channel, AGENT_CHANNEL_INDEX.

    Top-down only, and a diagnostic, not a verdict. The agent sits at the centre
    of the egocentric view by construction, so the stage marks this undefined
    there. Scores every cell of the agent channel, not whether the agent's own
    cell was located; `src/pipeline/displacement.py` takes the stricter
    localisation reading.

    Args:
        logits: Per-channel logits, each (batch, height, width, classes).
        targets: Flattened end observations, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Scalar accuracy over the agent channel.

    Raises:
        ValueError: If given a continuous prediction, which has no
            categorical reading.
    """
    correct = _correct_mask(logits, targets, grid_shape)
    return jnp.mean(correct[..., AGENT_CHANNEL_INDEX].astype(jnp.float32))


def model_cross_entropy(
    logits: list[jax.Array], targets: jax.Array, grid_shape: tuple[int, int]
) -> jax.Array:
    """Return mean per-cell cross-entropy, summed over channels.

    Mean per cell, where `reconstruction_loss` sums over them, so the value is
    comparable across observation modes' different grid sizes. The same
    quantity as `baselines.copy_cross_entropy`, asserted by a test on one
    predictor.

    Args:
        logits: Per-channel logits, each (batch, height, width, classes).
        targets: Flattened end observations, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Scalar mean per-cell cross-entropy in nats, summed over channels.

    Raises:
        ValueError: If given a continuous prediction, which has no
            categorical reading.
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

    Ground truth only. No prediction is involved, so the mask is defined
    wherever a start and an end observation exist.

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

    Mean, where `pixel_reconstruction_loss` sums, so the value is comparable
    across grid sizes. The same quantity as `baselines.copy_mse`, asserted by a
    test on one predictor.

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

    Detects the copy trap. A pure copy scores exactly zero on the changed set,
    and the per cell-channel mask is what makes that zero exact. The mover set
    comes from ground truth, never the prediction, and examples with an empty
    mover set are excluded with the fraction reported.

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
    # Pooled over kept examples, never averaged over their individual
    # accuracies, which would weight a two-cell change like a forty-cell one.
    accuracy = (
        float(jnp.sum(scored) / jnp.sum(per_example_moved)) if num_kept else None
    )
    return {
        MOVER_ACCURACY_KEY: accuracy,
        # Averaged over every example, excluded ones included. A mean over
        # movers alone would overstate how much of the grid moves.
        MOVER_CHANGED_CELLS_KEY: float(jnp.mean(per_example_moved)),
        MOVER_EXCLUDED_KEY: (
            float((num_examples - num_kept) / num_examples) if num_examples else 0.0
        ),
    }
