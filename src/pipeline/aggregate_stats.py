"""Pure statistics for cross-seed aggregation, over numbers only."""

from __future__ import annotations

import statistics
from typing import Any, Sequence

import numpy as np
from rliable import library as rliable_library
from rliable import metrics as rliable_metrics

# Bootstrap parameters, recorded in the artefact.
BOOTSTRAP_REPS: int = 50_000
BOOTSTRAP_SEED: int = 42
CONFIDENCE_INTERVAL_SIZE: float = 0.95

# Minimum seeds an IQM is meaningful over. Below this it is reported as
# absent, never as a number.
MIN_SEEDS_FOR_IQM: int = 3

STD_DDOF: int = 1
"""Sample rather than population standard deviation. Records what
`sample_std` computes; `statistics.stdev` is not parameterised by it."""


def iqm_of(scores: np.ndarray) -> np.ndarray:
    """Return the interquartile mean of a (runs, tasks) score array.

    Args:
        scores: Scores shaped (num_runs, num_tasks).

    Returns:
        A one-element array, the shape rliable's bootstrap resamples through.
    """
    return np.array([rliable_metrics.aggregate_iqm(scores)])


def sample_std(values: Sequence[float]) -> float:
    """Return the sample standard deviation of a metric across seeds.

    Args:
        values: One metric's value on each seed.

    Returns:
        The sample standard deviation, or 0.0 for a single value where it is
        undefined.
    """
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def horizon_mean(series: Sequence[float] | None) -> float | None:
    """Return the mean of a per-step series, or None if it is absent or empty.

    Args:
        series: A per-step metric series from an artefact.

    Returns:
        The horizon mean, or None where no window survived episode-boundary
        masking, which is not the same as a mean of zero.
    """
    if not series:
        return None
    return float(sum(series)) / len(series)


def delta(model: float | None, baseline: float | None) -> float | None:
    """Return the model-minus-baseline gap, or None if either side is absent.

    Args:
        model: The model's horizon-mean accuracy.
        baseline: The stationary-agent baseline's horizon-mean accuracy.
    """
    if model is None or baseline is None:
        return None
    return model - baseline


def steps_ahead(
    model: Sequence[float] | None, baseline: Sequence[float] | None
) -> float | None:
    """Return how many horizon steps the model beats the baseline at.

    Strictly greater. Matching a baseline that assumes the agent never moved
    is not beating it. Ragged inputs are truncated to the shorter series.

    Args:
        model: Per-step model accuracy.
        baseline: Per-step stationary-agent accuracy.

    Returns:
        The count of steps ahead, or None if either series is absent.
    """
    if not model or not baseline:
        return None
    return float(sum(one > two for one, two in zip(model, baseline)))


def aggregate_scalar(
    values: Sequence[float],
    reps: int = BOOTSTRAP_REPS,
    statistic_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Return mean, sample std, IQM and a bootstrap CI for one metric.

    The IQM is deterministic. The interval comes from a stratified bootstrap,
    so it reproduces only with the same reps and seed.

    Args:
        values: One metric's value on each usable seed.
        reps: Bootstrap resamples.
        statistic_seed: Seed for the bootstrap's random state.

    Returns:
        A uniform block. iqm, ci_low and ci_high are None below
        MIN_SEEDS_FOR_IQM.
    """
    numbers = [float(value) for value in values]
    block: dict[str, Any] = {
        "mean": float(sum(numbers) / len(numbers)),
        "std": sample_std(numbers),
        "iqm": None,
        "ci_low": None,
        "ci_high": None,
        "values": numbers,
    }
    if len(numbers) < MIN_SEEDS_FOR_IQM:
        return block

    scores = {"metric": np.array(numbers, dtype=np.float64).reshape(-1, 1)}
    # pylint: disable=no-member
    random_state = np.random.RandomState(statistic_seed)
    point, interval = rliable_library.get_interval_estimates(
        scores,
        iqm_of,
        reps=reps,
        confidence_interval_size=CONFIDENCE_INTERVAL_SIZE,
        random_state=random_state,
    )
    block["iqm"] = float(point["metric"][0])
    block["ci_low"] = float(interval["metric"][0][0])
    block["ci_high"] = float(interval["metric"][1][0])
    return block


def aggregate_vector(per_seed: Sequence[Sequence[float] | None]) -> list[dict]:
    """Aggregate a per-step vector across seeds, position by position.

    Truncates to the shortest series present, so a short seed silently drops
    the tail of every longer one.

    Args:
        per_seed: Each seed's per-step series.

    Returns:
        One {step, mean, std, n} entry per horizon position, 1-indexed. Empty
        when no seed carries the series.
    """
    usable = [series for series in per_seed if series]
    if not usable:
        return []
    length = min(len(series) for series in usable)
    return [
        {
            "step": index + 1,
            "mean": float(sum(series[index] for series in usable) / len(usable)),
            "std": sample_std([series[index] for series in usable]),
            "n": len(usable),
        }
        for index in range(length)
    ]
