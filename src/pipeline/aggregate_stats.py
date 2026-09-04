"""Pure statistics for cross-seed aggregation, over numbers only."""

from __future__ import annotations

import statistics
from typing import Any, Sequence

import numpy as np
from rliable import library as rliable_library

# Bootstrap parameters, recorded in the artefact.
BOOTSTRAP_REPS: int = 50_000
BOOTSTRAP_SEED: int = 42
CONFIDENCE_INTERVAL_SIZE: float = 0.95

# The fraction trimmed from each end of the sorted scores before averaging.
IQM_PROPORTION_TO_CUT: float = 0.25

# Minimum seeds an IQM is meaningful over. Below this it is reported as
# absent, never as a number.
MIN_SEEDS_FOR_IQM: int = 3

STD_DDOF: int = 1
"""Sample rather than population standard deviation. Records what
`sample_std` computes; `statistics.stdev` is not parameterised by it."""


def iqm_of(scores: np.ndarray) -> np.ndarray:
    """Return the interquartile mean of a (runs, tasks) score array.

    Trims `IQM_PROPORTION_TO_CUT` from each end of the flattened scores and
    averages the rest, matching `scipy.stats.trim_mean` operation for operation
    including its use of `np.partition`.

    Args:
        scores: Scores shaped (num_runs, num_tasks).

    Returns:
        A one-element array, the shape rliable's bootstrap resamples through.
    """
    flat = np.asarray(scores, dtype=np.float64).ravel()
    lowercut = int(IQM_PROPORTION_TO_CUT * flat.shape[0])
    uppercut = flat.shape[0] - lowercut
    trimmed = np.partition(flat, (lowercut, uppercut - 1))
    return np.array([np.mean(trimmed[lowercut:uppercut])])


def sample_std(values: Sequence[float]) -> float:
    """Return the sample standard deviation of a metric across seeds.

    Args:
        values: One metric's value on each seed.

    Returns:
        The sample standard deviation, or 0.0 for fewer than two values,
        where it is undefined.
    """
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def horizon_mean(series: Sequence[float] | None) -> float | None:
    """Return the mean of a per-step series, or None if it is absent or empty.

    Args:
        series: A per-step metric series from an artefact.

    Returns:
        The horizon mean, or None where the series is absent or empty, which
        is not the same as a mean of zero.
    """
    if not series:
        return None
    return float(sum(series)) / len(series)


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
