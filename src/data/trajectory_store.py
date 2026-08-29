"""Persist complete trajectories and measure their length and copy statistics.

Storage is HDF5 on Minari's per-episode schema. `terminated` and `truncated`
map to `terminations` and `truncations` on the way out and back on the way in.
Both observation modes for one trajectory share an episode group.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from config import TRAJECTORY_SHARD_GLOB
from src.data.trajectory import ObservationMode, Trajectory
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

# HDF5 layout. The reader and the writer must agree on these names.
EPISODE_GROUP_TEMPLATE: str = "episode_{index:06d}"
OBSERVATIONS_GROUP: str = "observations"
ACTIONS_DATASET: str = "actions"
REWARDS_DATASET: str = "rewards"
TERMINATIONS_DATASET: str = "terminations"
TRUNCATIONS_DATASET: str = "truncations"
GOAL_POSITION_DATASET: str = "goal_position"
EXECUTED_ACTIONS_DATASET: str = "executed_actions"
PROVENANCE_ATTR: str = "provenance_json"
SCHEMA_ATTR: str = "schema"
SCHEMA_NAME: str = "minari-like/v2"

# Per-episode flag fields, in the order they are written.
FLAG_FIELDS: tuple[tuple[str, str], ...] = (
    (ACTIONS_DATASET, "actions"),
    (REWARDS_DATASET, "rewards"),
    (TERMINATIONS_DATASET, "terminated"),
    (TRUNCATIONS_DATASET, "truncated"),
    (GOAL_POSITION_DATASET, "goal_position"),
    (EXECUTED_ACTIONS_DATASET, "executed_actions"),
)

# The one dataset a shard may lack, and the dataset it is backfilled from.
# Every other entry in FLAG_FIELDS is required on read.
BACKFILLED_FIELD: tuple[str, str] = (EXECUTED_ACTIONS_DATASET, ACTIONS_DATASET)


class TrajectoryStore:
    """Write and read complete trajectories in both observation modes.

    One file per shard. The round trip preserves dtypes. Observations are
    uint8, and a read path promoting them to int64 quadruples the dataset on
    disk.
    """

    def write_shard(self, trajectories: Sequence[Trajectory], path: Path) -> None:
        """Write one shard of trajectories to an HDF5 file.

        Args:
            trajectories: Episodes to write. May be empty, which writes a valid
                shard carrying zero episode groups.
            path: Destination file. Parent directories are created.
        """
        ensure_dir(path.parent)
        with h5py.File(path, "w") as handle:
            handle.attrs[SCHEMA_ATTR] = SCHEMA_NAME
            for index, trajectory in enumerate(trajectories):
                self._write_episode(handle, index, trajectory)
        logger.info(
            "wrote %d trajectories -> %s", len(trajectories), safe_rel(path)
        )

    @staticmethod
    def _write_episode(
        handle: h5py.File, index: int, trajectory: Trajectory
    ) -> None:
        """Write one episode group.

        Args:
            handle: Open HDF5 file.
            index: Position of this episode within the shard.
            trajectory: The episode to write.
        """
        group = handle.create_group(EPISODE_GROUP_TEMPLATE.format(index=index))
        group.attrs[PROVENANCE_ATTR] = json.dumps(trajectory.provenance)
        observations = group.create_group(OBSERVATIONS_GROUP)
        for mode, frames in trajectory.observations.items():
            observations.create_dataset(mode.value, data=np.asarray(frames))
        for dataset_name, field_name in FLAG_FIELDS:
            group.create_dataset(
                dataset_name, data=np.asarray(getattr(trajectory, field_name))
            )

    def read_dataset(self, directory: Path, consumer: str) -> list[Trajectory]:
        """Read every shard one dataset directory holds, in shard order.

        Preparation, training and evaluation each re-derive the partition by
        index, so all three must agree on the order.

        Args:
            directory: The dataset directory holding the shard files.
            consumer: What is reading, named in the error message.

        Returns:
            All complete episodes, in shard order.

        Raises:
            FileNotFoundError: If the directory holds no shards.
        """
        shards = sorted(directory.glob(TRAJECTORY_SHARD_GLOB))
        if not shards:
            raise FileNotFoundError(
                f"no trajectory shards under {safe_rel(directory)}. "
                f"{consumer} consumes the generation stage's output and "
                "creates none of its own, so run generation first."
            )
        trajectories: list[Trajectory] = []
        for shard in shards:
            trajectories.extend(self.read_shard(shard))
        logger.info(
            "read %d trajectories from %d shards in %s",
            len(trajectories),
            len(shards),
            safe_rel(directory),
        )
        return trajectories

    def read_shard(self, path: Path) -> list[Trajectory]:
        """Read one shard back.

        Args:
            path: Shard file written by write_shard.

        Returns:
            The episodes it holds, in write order, as numpy arrays.
        """
        with h5py.File(path, "r") as handle:
            return [
                self._read_episode(handle[name])
                for name in sorted(handle.keys())
            ]

    @staticmethod
    def _read_episode(group: h5py.Group) -> Trajectory:
        """Rebuild one Trajectory from its episode group.

        Args:
            group: The episode group.
        """
        observations = {
            ObservationMode(name): group[OBSERVATIONS_GROUP][name][()]
            for name in group[OBSERVATIONS_GROUP].keys()
        }
        fields = {
            field_name: TrajectoryStore._read_field(group, dataset_name)
            for dataset_name, field_name in FLAG_FIELDS
        }
        return Trajectory(
            observations=observations,
            provenance=json.loads(group.attrs[PROVENANCE_ATTR]),
            **fields,
        )

    @staticmethod
    def _read_field(group: h5py.Group, dataset_name: str) -> np.ndarray:
        """Read one per-episode dataset, backfilling the one v1 shards lack.

        A v1 shard carries no executed-action dataset, and its executed action
        equals its commanded one, every such shard predating any slip code in
        this repository.

        Args:
            group: The episode group.
            dataset_name: HDF5 dataset to read.

        Returns:
            The dataset's contents, or the backfill source's when the dataset
            named by BACKFILLED_FIELD is absent.

        Raises:
            KeyError: If any other dataset is missing.
        """
        backfilled_name, source_name = BACKFILLED_FIELD
        if dataset_name == backfilled_name and dataset_name not in group:
            return group[source_name][()]
        return group[dataset_name][()]

    @staticmethod
    def length_statistics(trajectories: Sequence[Trajectory]) -> dict:
        """Return episode-length count, extremes, central tendency and histogram.

        Args:
            trajectories: Episodes to measure.

        Returns:
            Mapping with count, min, max, mean, median and a length histogram
            keyed by length as a string. An empty input returns count 0 and
            None for every statistic, never raising.
        """
        lengths = [len(trajectory) for trajectory in trajectories]
        if not lengths:
            return {
                "count": 0, "min": None, "max": None,
                "mean": None, "median": None, "histogram": {},
            }
        return {
            "count": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "histogram": {
                str(length): count
                for length, count in sorted(Counter(lengths).items())
            },
        }

    @staticmethod
    def copy_and_mover_statistics(
        trajectories: Sequence[Trajectory], horizons: Sequence[int]
    ) -> dict:
        """Measure the stationary-copy and mover-mask statistics, per mode per h.

        Measured over all valid (t, t+h) pairs within each trajectory rather
        than over the evaluation windows, so these describe the generated data
        and not the baseline the model was scored against. No copy-baseline
        cross-entropy is computed here. It lives with the baseline in
        `src/eval/baselines.py`, which owns the smoothing convention.

        Args:
            trajectories: Episodes to measure.
            horizons: Horizons to measure at. A horizon longer than an episode
                contributes no pairs from it.

        Returns:
            Mapping of observation mode value to horizon (as a string) to the
            per-horizon statistics. A horizon no episode was long enough to
            supply is absent, not null.
        """
        modes = sorted(
            {mode for trajectory in trajectories for mode in trajectory.observations},
            key=lambda mode: mode.value,
        )
        return {
            mode.value: TrajectoryStore._statistics_for_mode(
                trajectories, mode, horizons
            )
            for mode in modes
        }

    @staticmethod
    def _statistics_for_mode(
        trajectories: Sequence[Trajectory],
        mode: ObservationMode,
        horizons: Sequence[int],
    ) -> dict:
        """Accumulate copy and mover statistics for one observation mode.

        Args:
            trajectories: Episodes to measure.
            mode: Which stored view to measure.
            horizons: Horizons to measure at.

        Returns:
            Mapping of horizon as a string to that horizon's statistics.
        """
        frames = [
            np.asarray(trajectory.observations[mode])
            for trajectory in trajectories
            if mode in trajectory.observations
        ]
        results = {}
        for horizon in horizons:
            usable = [frame for frame in frames if frame.shape[0] > horizon]
            if not usable:
                continue
            results[str(horizon)] = TrajectoryStore._statistics_at_horizon(
                usable, horizon
            )
        return results

    @staticmethod
    def _statistics_at_horizon(frames: Sequence[np.ndarray], horizon: int) -> dict:
        """Measure one (mode, horizon) cell over every episode supplying pairs.

        A cell counts as moving when any of its channels differs. Copy accuracy
        is reported per channel entry instead, matching the decoder's per-cell
        categorical over each channel separately.

        Args:
            frames: Per-episode observation arrays, each (num_steps + 1, ...),
                already filtered to those longer than the horizon.
            horizon: The horizon h, comparing s_t against s_{t+h}.

        Returns:
            Mapping with the pair count, copy accuracy, mover fraction, mean
            moving-cell count and empty-mover fraction.
        """
        matching_entries = 0
        total_entries = 0
        moving_cells = 0
        total_cells = 0
        empty_mover_pairs = 0
        total_pairs = 0
        for frame in frames:
            identical = frame[:-horizon] == frame[horizon:]
            moves = ~identical.all(axis=-1)
            matching_entries += int(identical.sum())
            total_entries += identical.size
            moving_cells += int(moves.sum())
            total_cells += moves.size
            per_pair_moves = moves.reshape(moves.shape[0], -1).sum(axis=1)
            empty_mover_pairs += int((per_pair_moves == 0).sum())
            total_pairs += int(per_pair_moves.shape[0])
        return {
            "num_pairs": total_pairs,
            "copy_accuracy": matching_entries / total_entries,
            "mover_fraction": moving_cells / total_cells,
            "mean_moving_cells": moving_cells / total_pairs,
            "empty_mover_fraction": empty_mover_pairs / total_pairs,
        }
