"""The copy baselines and the climatology floors.

A categorical observation gets a calibrated copy: a smoothed transition table
fitted on the training split. A continuous one gets the stationary copy,
uncalibrated. Each representation also gets an input-blind climatology floor,
an entropy over classes or the mean-image MSE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.models.losses import (
    observation_class_targets,
    observation_pixel_targets,
)

# Pseudo-count added to every transition pair and class before normalising.
SMOOTHING_COUNT: float = 1.0

# Keys of smoothing_report's per-channel entries.
ZERO_PAIRS_KEY: str = "zero_count_pairs"
TOTAL_PAIRS_KEY: str = "total_pairs"
SMOOTHED_MASS_KEY: str = "smoothed_fraction_of_pairs"


def copy_transition_counts(
    states: jax.Array,
    targets: jax.Array,
    grid_shape: tuple[int, int],
    obs_channel_classes: tuple[int, ...],
) -> list[jax.Array]:
    """Count observed (start code, end code) pairs per channel.

    Returns raw counts, which smoothing_report reads to find the pairs never
    observed.

    Args:
        states: Flattened start observations s_t from the training split only,
            shape (batch, obs_dim).
        targets: Flattened end observations s_{t+h} for the same pairs, one
            horizon, same shape as states.
        grid_shape: Spatial grid shape, (height, width).
        obs_channel_classes: Classes per channel.

    Returns:
        One (classes, classes) integer count array per channel, in channel
        order. Entry [c_prev, c_next] counts cells that held c_prev at t and
        c_next at t+h.
    """
    start = observation_class_targets(states, grid_shape)
    end = observation_class_targets(targets, grid_shape)
    counts: list[jax.Array] = []
    for channel, num_classes in enumerate(obs_channel_classes):
        previous = start[..., channel].reshape(-1)
        following = end[..., channel].reshape(-1)
        flat = jnp.bincount(
            previous * num_classes + following,
            length=num_classes * num_classes,
        )
        counts.append(flat.reshape(num_classes, num_classes))
    return counts


def calibrated_copy_tables(
    states: jax.Array,
    targets: jax.Array,
    grid_shape: tuple[int, int],
    obs_channel_classes: tuple[int, ...],
) -> list[jax.Array]:
    """Estimate one row-normalised conditional table per observation channel.

    Fit on the training split only. Every pair count is raised by
    SMOOTHING_COUNT first, so an unseen (c_prev, c_next) pair keeps a non-zero
    probability. Without it the copy cross-entropy is infinite and nothing
    raises.

    Args:
        states: Flattened start observations s_t.
        targets: Flattened end observations s_{t+h} for the same pairs, one
            horizon.
        grid_shape: Spatial grid shape, (height, width).
        obs_channel_classes: Classes per channel.

    Returns:
        One (classes, classes) row-normalised table per channel, in channel
        order. Entry [c_prev, c_next] is
        P(s_{t+h}[i] = c_next | s_t[i] = c_prev).
    """
    tables: list[jax.Array] = []
    for counts in copy_transition_counts(
        states, targets, grid_shape, obs_channel_classes
    ):
        smoothed = counts.astype(jnp.float32) + SMOOTHING_COUNT
        tables.append(smoothed / jnp.sum(smoothed, axis=-1, keepdims=True))
    return tables


def copy_cross_entropy(
    tables: list[jax.Array],
    states: jax.Array,
    targets: jax.Array,
    grid_shape: tuple[int, int],
) -> jax.Array:
    """Return the calibrated copy baseline's mean per-cell cross-entropy.

    Must stay the same quantity as `metrics.model_cross_entropy`, mean per cell
    and summed over channels. If the two definitions diverge, the skill score
    is wrong and nothing fails.

    Args:
        tables: Row-normalised conditionals from calibrated_copy_tables,
            estimated on the training split at this horizon.
        states: Flattened start observations of the windows being scored.
        targets: Flattened end observations of the same windows.
        grid_shape: Spatial grid shape, (height, width).

    Returns:
        Scalar mean per-cell cross-entropy in nats, summed over channels.
    """
    start = observation_class_targets(states, grid_shape)
    end = observation_class_targets(targets, grid_shape)
    total = jnp.zeros(())
    for channel, table in enumerate(tables):
        probabilities = table[start[..., channel], end[..., channel]]
        total = total - jnp.mean(jnp.log(probabilities))
    return total


def copy_mse(
    states: jax.Array,
    targets: jax.Array,
    grid_shape: tuple[int, int],
    value_range: tuple[int, int],
) -> jax.Array:
    """Return the stationary copy's mean squared error on normalised values.

    The copy predicts the start observation unchanged. Must stay the same
    quantity as `metrics.model_mse`, mean per cell-channel on the [0, 1] scale.

    Args:
        states: Flattened start observations of the windows being scored.
        targets: Flattened end observations of the same windows.
        grid_shape: Spatial grid shape, (height, width).
        value_range: Inclusive (low, high) bounds of the stored values.

    Returns:
        Scalar mean squared error on the [0, 1] scale.
    """
    start = observation_pixel_targets(states, grid_shape, value_range)
    end = observation_pixel_targets(targets, grid_shape, value_range)
    return jnp.mean((start - end) ** 2)


def climatology_mse(
    fit_targets: jax.Array,
    scored_targets: jax.Array,
    grid_shape: tuple[int, int],
    value_range: tuple[int, int],
) -> jax.Array:
    """Return the mean squared error of predicting the per-horizon mean image.

    The input-blind floor for a continuous representation: the error of
    ignoring the start observation and emitting the fit set's mean end image.
    The evaluate stage fits it on the training split. A squared error, so it is
    not comparable to `climatology_entropy`, and it is never the skill score's
    denominator.

    Args:
        fit_targets: Flattened end observations the mean image is fitted on,
            one horizon, shape (n, obs_dim).
        scored_targets: Flattened end observations of the windows being
            scored, same horizon, shape (batch, obs_dim).
        grid_shape: Spatial grid shape, (height, width).
        value_range: Inclusive (low, high) bounds of the stored values.

    Returns:
        Scalar mean squared error on the [0, 1] scale.

    Raises:
        ValueError: If the fit set is empty.
    """
    if int(fit_targets.shape[0]) == 0:
        raise ValueError(
            "climatology_mse was given no fit windows, so there is no mean "
            "image to predict"
        )
    mean_image = jnp.mean(
        observation_pixel_targets(fit_targets, grid_shape, value_range), axis=0
    )
    truth = observation_pixel_targets(scored_targets, grid_shape, value_range)
    return jnp.mean((truth - mean_image) ** 2)


def smoothing_report(tables_counts: list[jax.Array]) -> dict:
    """Return, per channel, how many (c_prev, c_next) pairs had zero count.

    Args:
        tables_counts: Raw counts from copy_transition_counts. Smoothed tables
            hold no zeros and report none.

    Returns:
        Mapping of channel index, as a string, to its zero-count pairs, total
        pairs, and the zero-count fraction.
    """
    report: dict = {}
    for channel, counts in enumerate(tables_counts):
        total = int(counts.size)
        zeros = int(jnp.sum(counts == 0))
        report[str(channel)] = {
            ZERO_PAIRS_KEY: zeros,
            TOTAL_PAIRS_KEY: total,
            SMOOTHED_MASS_KEY: float(zeros / total) if total else 0.0,
        }
    return report


def climatology_entropy(
    targets: jax.Array,
    grid_shape: tuple[int, int],
    obs_channel_classes: tuple[int, ...],
) -> jax.Array:
    """Return H(p) for the marginal class frequencies, the input-blind floor.

    Reported beside the copy cross-entropy, never as the skill score's
    denominator. The evaluate stage fits it on the training split.

    Args:
        targets: Flattened observations whose marginal class frequencies
            define the floor.
        grid_shape: Spatial grid shape, (height, width).
        obs_channel_classes: Classes per channel.

    Returns:
        Scalar entropy in nats, summed over channels, on the same scale as the
        copy cross-entropy.
    """
    codes = observation_class_targets(targets, grid_shape)
    total = jnp.zeros(())
    for channel, num_classes in enumerate(obs_channel_classes):
        counts = jnp.bincount(
            codes[..., channel].reshape(-1), length=num_classes
        ).astype(jnp.float32)
        # Without smoothing, a class absent from the fit set gives 0 * log(0),
        # which is NaN.
        smoothed = counts + SMOOTHING_COUNT
        probabilities = smoothed / jnp.sum(smoothed)
        total = total - jnp.sum(probabilities * jnp.log(probabilities))
    return total
