"""The calibrated copy baseline and climatology."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.models.losses import observation_class_targets

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

    Counting is a separate step from smoothing. Once add-one has been applied
    the number of pairs it touched cannot be recovered, and that number is
    reported.

    Args:
        states: Flattened start observations s_t from the training split only,
            shape (batch, obs_dim).
        targets: Flattened end observations s_{t+h} for the same pairs, one
            horizon, same shape as states.
        grid_shape: Spatial grid shape, (height, width). Read from config,
            never a literal.
        obs_channel_classes: Classes per channel, read from config. A
            single-channel environment with thousands of classes is covered by
            this signature unchanged.

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

    Training split only. Estimating on held-out data is leakage and flatters
    the baseline. Add-one smoothing is load-bearing: an unseen
    (c_prev, c_next) pair has count zero, and an unsmoothed zero sends
    cross-entropy to infinity.

    Args:
        states: Flattened start observations s_t from the training split only.
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

    The same quantity as the model's. Both are mean per-cell categorical
    cross-entropy summed over channels, so the ratio compares like with like.
    If the two diverge in definition the skill score becomes meaningless
    without anything failing, so a test feeds this and
    `metrics.model_cross_entropy` the same predictor and asserts they agree.

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


def smoothing_report(tables_counts: list[jax.Array]) -> dict:
    """Return, per channel, how many (c_prev, c_next) pairs had zero count.

    Measures smoothing's reach. Per horizon and per channel. Channels with
    different class counts have different numbers of pairs to fill.

    Args:
        tables_counts: Raw, unsmoothed counts from copy_transition_counts.

    Returns:
        Mapping of channel index, as a string so the report is JSON-keyed, to
        its zero-count pairs, total pairs, and the fraction smoothing invented.
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

    Reported alongside, never as the denominator. Climatology predicts the same
    marginal for every cell, so it is strictly weaker than a copy baseline and
    normalising the skill score by it would inflate every reported value.
    Estimated from the training split, like the copy tables.

    Args:
        targets: Flattened observations from the training split, whose marginal
            class frequencies define the floor.
        grid_shape: Spatial grid shape, (height, width).
        obs_channel_classes: Classes per channel.

    Returns:
        Scalar entropy in nats, summed over channels, directly comparable with
        the two cross-entropies above.
    """
    codes = observation_class_targets(targets, grid_shape)
    total = jnp.zeros(())
    for channel, num_classes in enumerate(obs_channel_classes):
        counts = jnp.bincount(
            codes[..., channel].reshape(-1), length=num_classes
        ).astype(jnp.float32)
        # Smoothed on the same rule as the copy tables. A class absent from
        # the training split would otherwise contribute 0 * log(0).
        smoothed = counts + SMOOTHING_COUNT
        probabilities = smoothed / jnp.sum(smoothed)
        total = total - jnp.sum(probabilities * jnp.log(probabilities))
    return total
