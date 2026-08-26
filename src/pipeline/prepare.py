"""Dataset preparation stage: partition a generated dataset and freeze its
evaluation windows.

Decides which trajectories are training data, which are held out during
development and which are opened once for the reported result. Splitting is by
trajectory, never by sample. Two windows from one episode overlap.

Under ONLINE sampling the windows are reproducible only from a seed plus the
sampler code, so a refactor changes them silently. A file on disk cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

from config import (
    EVALUATED_SPLITS,
    SPLIT_PROVENANCE_FILENAME,
    ExperimentConfig,
)
from src.data.split import SplitName, split_from_config
from src.data.trajectory import Trajectory
from src.data.trajectory_store import TrajectoryStore
from src.data.window_sampler import WindowSampler
from src.pipeline.base import Stage, sampler_identity_fields
from src.utils.logging_setup import get_logger
from src.utils.paths import safe_rel

logger = get_logger(__name__)


def frozen_evaluation_key(config: ExperimentConfig, name: SplitName) -> jax.Array:
    """Return the PRNG key one split's frozen evaluation windows are drawn with.

    One key per (data seed, split), so the frozen set is reproducible from the
    dataset alone and two splits cannot draw the same offsets. Module-level:
    the evaluator asks the same question, and a second derivation would score a
    different set while every shape agreed.

    Args:
        config: The composed experiment configuration.
        name: The split whose windows are being drawn.

    Returns:
        The PRNG key for that split's draw.

    Raises:
        ValueError: If the split has no frozen set.
    """
    if name.value not in EVALUATED_SPLITS:
        raise ValueError(
            f"split '{name.value}' has no frozen evaluation set: "
            f"EVALUATED_SPLITS is {EVALUATED_SPLITS}. TRAIN is excluded "
            "deliberately -- it is not evaluated."
        )
    return jax.random.fold_in(
        jax.random.PRNGKey(config.data_seed),
        EVALUATED_SPLITS.index(name.value),
    )


class PrepareDatasetStage(Stage):
    """Split one generated dataset and freeze its evaluation windows.

    Produces the three-way partition with its composition report, and one
    frozen evaluation file per evaluated split for the configured mode. One
    invocation writes one mode's sets, and generation records both modes in
    every shard, so the second costs no regeneration.

    Attributes:
        store: The trajectory store this stage reads through.
    """

    name: str = "prepare"
    # The partition and the frozen windows are properties of the dataset, so
    # they follow the data seed.
    dataset_derived: bool = True

    @property
    def sentinel_key(self) -> str:
        """Return a sentinel subdirectory scoped to the observation mode.

        One sentinel per mode. The frozen sets carry the mode in their
        filename and not as a directory level, so both coexist under one
        dataset directory and a single sentinel could describe only one.

        Returns:
            `"prepare_<observation_mode>"`, e.g. `"prepare_top_down"`.
        """
        return f"{self.name}_{self.config.sampler.observation_mode}"

    def __init__(self, config: ExperimentConfig) -> None:
        """Build the stage.

        Args:
            config: The composed experiment configuration.
        """
        super().__init__(config)
        self.store = TrajectoryStore()

    def run(self) -> None:
        """Partition the dataset, record the partition, freeze the windows."""
        target = self.dataset_dir
        trajectories = self._load(target)
        split = self._split(trajectories)
        self._write_provenance(split, target)
        self._freeze_evaluation_sets(trajectories, split)

    def _load(self, target: Path) -> list[Trajectory]:
        """Read every shard the generation stage wrote for this data seed.

        Args:
            target: The dataset directory to read.

        Returns:
            All complete episodes, in shard order.

        Raises:
            FileNotFoundError: If the dataset directory holds no shards.
        """
        return self.store.read_dataset(target, "dataset preparation")

    def _split(self, trajectories: list[Trajectory]):
        """Partition trajectory indices three ways, by trajectory.

        Args:
            trajectories: Every episode in this dataset, in store order.

        Returns:
            The partition, with its composition statistics.
        """
        # split_from_config, not a hand-assembled call, or two stages could
        # drift onto different partitions.
        split = split_from_config(
            self.config, [len(trajectory) for trajectory in trajectories]
        )
        for composition in split.composition:
            logger.info(
                "split %-10s n=%-5d mean=%-7.1f median=%-6.1f support=%.4f",
                composition.name.value,
                composition.num_trajectories,
                composition.mean_length,
                composition.median_length,
                composition.horizon_support_fraction,
            )
        if split.redraw_attempts > 1:
            logger.info(
                "the partition needed %d draws to pass the composition check; "
                "the count is recorded in %s",
                split.redraw_attempts,
                SPLIT_PROVENANCE_FILENAME,
            )
        return split

    @staticmethod
    def _write_provenance(split, target: Path) -> None:
        """Write the partition and its composition report to disk.

        Written whether or not the check passed first time. The verdict is not
        a substitute for the numbers behind it.

        Args:
            split: The partition to record.
            target: Directory to write into.
        """
        path = target / SPLIT_PROVENANCE_FILENAME
        path.write_text(json.dumps(split.as_provenance(), indent=2), encoding="utf-8")
        logger.info("wrote the trajectory partition -> %s", safe_rel(path))

    def _freeze_evaluation_sets(self, trajectories: list[Trajectory], split) -> None:
        """Write one frozen evaluation file per evaluated split.

        The sampler owns the naming, scoping every frozen file by split and
        mode, so no path is constructed here.

        Args:
            trajectories: Every episode in this dataset, in store order.
            split: The partition drawn for it.
        """
        for name in (SplitName(value) for value in EVALUATED_SPLITS):
            sampler = WindowSampler.from_config(
                self.config, trajectories, split, name
            )
            batch = sampler.evaluation_set(
                frozen_evaluation_key(self.config, name)
            )
            logger.info(
                "froze %d %s windows in %s mode",
                len(batch),
                name.value,
                self.config.sampler.observation_mode,
            )

    def sentinel_identity(self) -> dict:
        """Return the identity guarding this dataset's partition and windows.

        Carries the split fractions, the horizon ceiling, the evaluation grid
        and the mode, each of which changes what the frozen files contain while
        leaving their paths identical.

        Returns:
            JSON-serialisable identity fields.
        """
        identity = super().sentinel_identity()
        identity.update(
            {
                "num_trajectories": self.config.data.num_trajectories,
                "horizon_max": self.config.train.horizon_max,
                "split_fractions": [
                    self.config.data.train_fraction,
                    self.config.data.validation_fraction,
                    self.config.data.test_fraction,
                ],
                "evaluated_splits": list(EVALUATED_SPLITS),
                **sampler_identity_fields(self.config),
            }
        )
        return identity
