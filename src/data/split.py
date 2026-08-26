"""Trajectory-level splitting with a leakage check.

Partitions trajectory indices three ways, by trajectory and not by sample.
Checks each split's horizon support against the pooled figure and redraws when
they deviate. Reports any index appearing in more than one split.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from config import (
    SPLIT_FRACTION_SUM_TOLERANCE,
    SPLIT_MAX_REDRAW_ATTEMPTS,
    SPLIT_SUPPORT_TOLERANCE,
    ExperimentConfig,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Smallest trajectory count a split may hold.
MIN_SPLIT_TRAJECTORIES: int = 1


class SplitName(Enum):
    """Which partition a trajectory or a batch belongs to."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitComposition:
    """Descriptive statistics for one split, reported pass or fail.

    Attributes:
        name: Which split these statistics describe.
        num_trajectories: Trajectories allocated to it.
        mean_length: Mean episode length.
        median_length: Median episode length.
        horizon_support_fraction: Share of this split's trajectories with
            L >= horizon_max.
    """

    name: SplitName
    num_trajectories: int
    mean_length: float
    median_length: float
    horizon_support_fraction: float


@dataclass(frozen=True)
class TrajectorySplit:
    """A three-way partition of trajectory indices.

    Attributes:
        train: Indices allocated to training.
        validation: Indices allocated to validation.
        test: Indices allocated to test.
        split_seed: The seed that produced this partition.
        redraw_attempts: How many draws the composition check needed.
        composition: Per-split statistics, ordered train, validation, test.
        pooled_support_fraction: Share of all trajectories with
            L >= horizon_max.
    """

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    split_seed: int
    redraw_attempts: int
    composition: tuple[SplitComposition, ...]
    pooled_support_fraction: float

    def indices_for(self, name: SplitName) -> tuple[int, ...]:
        """Return the trajectory indices allocated to one split."""
        return {
            SplitName.TRAIN: self.train,
            SplitName.VALIDATION: self.validation,
            SplitName.TEST: self.test,
        }[name]

    def composition_for(self, name: SplitName) -> SplitComposition:
        """Return the descriptive statistics for one split.

        Args:
            name: Which partition to read.

        Returns:
            That split's composition record.

        Raises:
            KeyError: If the partition carries no record for that split.
        """
        for entry in self.composition:
            if entry.name is name:
                return entry
        raise KeyError(f"no composition record for {name}")

    def as_provenance(self) -> dict:
        """Return a JSON-serialisable record of this partition.

        Returns:
            Mapping of split sizes, indices, seed, attempt count and the
            per-split composition.
        """
        return {
            "split_seed": self.split_seed,
            "redraw_attempts": self.redraw_attempts,
            "pooled_support_fraction": self.pooled_support_fraction,
            "splits": {
                name.value: {
                    "num_trajectories": len(self.indices_for(name)),
                    "indices": list(self.indices_for(name)),
                    "mean_length": self.composition_for(name).mean_length,
                    "median_length": self.composition_for(name).median_length,
                    "horizon_support_fraction": (
                        self.composition_for(name).horizon_support_fraction
                    ),
                }
                for name in SplitName
            },
        }


def _validate_split_inputs(
    num_trajectories: int,
    horizon_max: int,
    fractions: dict[str, float],
    support_tolerance: float,
    max_redraw_attempts: int,
) -> None:
    """Reject arguments that cannot produce a usable partition.

    Args:
        num_trajectories: Trajectories being partitioned.
        horizon_max: Ceiling used for the support check.
        fractions: The three split shares, keyed by argument name.
        support_tolerance: Allowed deviation in horizon-support fraction.
        max_redraw_attempts: Redraw bound.

    Raises:
        ValueError: If any argument cannot produce a usable partition.
    """
    if num_trajectories <= 0:
        raise ValueError("cannot split an empty set of trajectories")
    if horizon_max < 1:
        raise ValueError(f"horizon_max must be at least 1, got {horizon_max}")
    for name, value in fractions.items():
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"{name} must lie strictly between 0 and 1, got {value}"
            )
    total = sum(fractions.values())
    if abs(total - 1.0) > SPLIT_FRACTION_SUM_TOLERANCE:
        raise ValueError(
            f"split fractions must sum to 1.0, got {total}. Values: {fractions}"
        )
    if not 0.0 <= support_tolerance <= 1.0:
        raise ValueError(
            f"support_tolerance must lie in [0, 1], got {support_tolerance}"
        )
    if max_redraw_attempts < 1:
        raise ValueError(
            f"max_redraw_attempts must be at least 1, got {max_redraw_attempts}"
        )


def _allocate_counts(
    num_trajectories: int, train_fraction: float, validation_fraction: float
) -> tuple[int, int, int]:
    """Split a trajectory count three ways.

    Floors train and validation and gives the remainder to test.

    Args:
        num_trajectories: Total trajectories.
        train_fraction: Share for training.
        validation_fraction: Share for validation.

    Returns:
        Counts for train, validation and test, summing to num_trajectories.

    Raises:
        ValueError: If the rule leaves any split below MIN_SPLIT_TRAJECTORIES.
    """
    num_train = int(num_trajectories * train_fraction)
    num_validation = int(num_trajectories * validation_fraction)
    num_test = num_trajectories - num_train - num_validation
    counts = {
        SplitName.TRAIN: num_train,
        SplitName.VALIDATION: num_validation,
        SplitName.TEST: num_test,
    }
    too_small = {
        name.value: count
        for name, count in counts.items()
        if count < MIN_SPLIT_TRAJECTORIES
    }
    if too_small:
        raise ValueError(
            f"a three-way split of {num_trajectories} trajectories leaves "
            f"{too_small} below {MIN_SPLIT_TRAJECTORIES}. Generate more "
            "trajectories rather than dropping a split"
        )
    return num_train, num_validation, num_test


def _compose(
    name: SplitName, indices: np.ndarray, lengths: np.ndarray, horizon_max: int
) -> SplitComposition:
    """Measure one split's length distribution and horizon support.

    Args:
        name: Which partition this describes.
        indices: Trajectory indices in this partition.
        lengths: Episode length per trajectory, indexed globally.
        horizon_max: Ceiling the support fraction is measured against.

    Returns:
        The split's composition record.
    """
    selected = lengths[indices]
    return SplitComposition(
        name=name,
        num_trajectories=int(selected.size),
        mean_length=float(statistics.fmean(selected.tolist())),
        median_length=float(statistics.median(selected.tolist())),
        horizon_support_fraction=float(np.mean(selected >= horizon_max)),
    )


def split_trajectories(  # pylint: disable=too-many-arguments,too-many-locals
    lengths: Sequence[int],
    *,
    horizon_max: int,
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    support_tolerance: float,
    max_redraw_attempts: int,
) -> TrajectorySplit:
    """Partition trajectory indices three ways, by trajectory, with a check.

    Train and validation are floored and test takes the remainder. If any
    split's horizon-support fraction deviates from the pooled fraction by more
    than support_tolerance, the seed is advanced and the partition redrawn, up
    to max_redraw_attempts.

    Args:
        lengths: Episode length per trajectory, in store order.
        horizon_max: Ceiling used for the support check.
        split_seed: Attempt k uses split_seed + k - 1.
        train_fraction: Share for training.
        validation_fraction: Share for validation.
        test_fraction: Share for reported results.
        support_tolerance: Maximum allowed deviation in horizon-support
            fraction, as a proportion.
        max_redraw_attempts: Redraw bound before raising.

    Returns:
        The partition, with its composition statistics and attempt count.

    Raises:
        ValueError: If the fractions do not partition, if any split would be
            empty, or if the composition check fails max_redraw_attempts times.
    """
    fractions = {
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    }
    lengths_array = np.asarray(lengths, dtype=np.int64)
    _validate_split_inputs(
        int(lengths_array.size),
        horizon_max,
        fractions,
        support_tolerance,
        max_redraw_attempts,
    )
    num_train, num_validation, _ = _allocate_counts(
        int(lengths_array.size), train_fraction, validation_fraction
    )
    pooled_support = float(np.mean(lengths_array >= horizon_max))

    worst_deviation = float("inf")
    for attempt in range(1, max_redraw_attempts + 1):
        rng = np.random.default_rng(split_seed + attempt - 1)
        order = rng.permutation(int(lengths_array.size))
        parts = {
            SplitName.TRAIN: order[:num_train],
            SplitName.VALIDATION: order[num_train : num_train + num_validation],
            SplitName.TEST: order[num_train + num_validation :],
        }
        composition = tuple(
            _compose(name, indices, lengths_array, horizon_max)
            for name, indices in parts.items()
        )
        worst_deviation = max(
            abs(entry.horizon_support_fraction - pooled_support)
            for entry in composition
        )
        if worst_deviation <= support_tolerance:
            split = TrajectorySplit(
                train=tuple(sorted(int(i) for i in parts[SplitName.TRAIN])),
                validation=tuple(
                    sorted(int(i) for i in parts[SplitName.VALIDATION])
                ),
                test=tuple(sorted(int(i) for i in parts[SplitName.TEST])),
                split_seed=split_seed,
                redraw_attempts=attempt,
                composition=composition,
                pooled_support_fraction=pooled_support,
            )
            logger.info(
                "split %d trajectories %d/%d/%d on seed %d after %d attempt(s); "
                "pooled h>=%d support %.4f, worst split deviation %.4f",
                int(lengths_array.size),
                len(split.train),
                len(split.validation),
                len(split.test),
                split_seed,
                attempt,
                horizon_max,
                pooled_support,
                worst_deviation,
            )
            return split

    raise ValueError(
        f"the composition check failed {max_redraw_attempts} times on seed "
        f"{split_seed}: best deviation {worst_deviation:.4f} from the pooled "
        f"h >= {horizon_max} support of {pooled_support:.4f}, against a "
        f"tolerance of {support_tolerance}. Regenerate with more trajectories, "
        "or lower horizon_max"
    )


def split_from_config(
    config: ExperimentConfig, lengths: Sequence[int]
) -> TrajectorySplit:
    """Draw the partition an experiment configuration implies.

    Seeded from the data seed.

    Args:
        config: The composed experiment configuration.
        lengths: Episode length per trajectory, in store order.

    Returns:
        The three-way partition, with its composition statistics.

    Raises:
        ValueError: Anything `split_trajectories` raises, unchanged.
    """
    return split_trajectories(
        lengths,
        horizon_max=config.train.horizon_max,
        split_seed=config.data_seed,
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
        test_fraction=config.data.test_fraction,
        support_tolerance=SPLIT_SUPPORT_TOLERANCE,
        max_redraw_attempts=SPLIT_MAX_REDRAW_ATTEMPTS,
    )


def detect_trajectory_leakage(split: TrajectorySplit) -> frozenset[int]:
    """Return trajectory indices appearing in more than one split.

    An empty set means no leak.

    Args:
        split: The partition to check.

    Returns:
        Indices present in at least two of the three splits.
    """
    train = set(split.train)
    validation = set(split.validation)
    test = set(split.test)
    return frozenset(
        (train & validation) | (train & test) | (validation & test)
    )
