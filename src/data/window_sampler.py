"""Turn whole trajectories into (s_t, a_{t:t+h}) -> s_{t+h} examples.

Sampling draws the horizon first: `h ~ U(horizon_min, horizon_max)`, then a
trajectory long enough to support h, then `t ~ U(0, L - h)`. Every horizon is
drawn equally often, but the number of distinct windows falls as h approaches
L, so every batch carries its per-horizon window count.

One observation mode per instance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

import jax
import jax.numpy as jnp

from config import (
    WINDOWS_EVAL_TEMPLATE,
    WINDOWS_TRAIN_TEMPLATE,
    ExperimentConfig,
    OFFLINE_BYTE_BUDGET,
    SamplerMode,
    data_dir,
)
from src.data.split import SplitName, TrajectorySplit
from src.data.tokeniser import check_horizons_valid
from src.data.trajectory import ObservationMode, Trajectory
from src.utils.logging_setup import get_logger
from src.utils.paths import safe_rel

logger = get_logger(__name__)

# Value written into masked action slots. The attention mask excludes them, so
# any valid index would do. Zero is inert if the mask is ever dropped.
PAD_ACTION_INDEX: int = 0

# Bytes per int32 action slot.
ACTION_INDEX_BYTES: int = 4

# HDF5 dataset and attribute names for a stored window batch.
STATES_DATASET: str = "states"
ACTIONS_DATASET: str = "actions"
HORIZONS_DATASET: str = "horizons"
TARGETS_DATASET: str = "targets"
HORIZON_COUNTS_DATASET: str = "horizon_counts"
SPLIT_ATTR: str = "split"
OBSERVATION_MODE_ATTR: str = "observation_mode"


@dataclass(frozen=True)
class WindowBatch:
    """One batch of jumpy-prediction examples.

    A frozen dataclass, not a tuple. `states` and `targets` share a shape and
    dtype, so a swap would pass every shape assertion.

    Attributes:
        states: Start observations s_t, shape
            (batch, *field_shape, channels).
        actions: Action indices a_{t:t+h-1}, padded to num_action_tokens with
            PAD_ACTION_INDEX. The count of unmasked slots is the horizon, via
            `tokeniser.action_padding_mask` over `horizons`.
        horizons: True horizon per example, shape (batch,).
        targets: End observations s_{t+h}, same shape as states.
        split: Which partition these came from.
        horizon_counts: Distinct windows available in this split at each
            example's horizon.
        observation_mode: The single mode these windows record.
    """

    states: jax.Array
    actions: jax.Array
    horizons: jax.Array
    targets: jax.Array
    split: SplitName
    horizon_counts: jax.Array
    observation_mode: ObservationMode

    def __len__(self) -> int:
        """Return the number of examples."""
        return int(self.horizons.shape[0])


def bytes_per_window(
    field_shape: tuple[int, ...],
    channels: int,
    num_action_tokens: int,
    observation_itemsize: int = 1,
) -> int:
    """Return the on-disk cost of one materialised window.

    Two frames plus one padded action sequence. The OFFLINE byte budget is
    calibrated against this.

    Args:
        field_shape: Spatial shape of one observation, excluding channels.
        channels: Channel count of one observation.
        num_action_tokens: Padded action-sequence length.
        observation_itemsize: Bytes per observation cell. One for the uint8
            observations this project stores.

    Returns:
        Bytes one window occupies.
    """
    frame = math.prod(field_shape) * channels * observation_itemsize
    return 2 * frame + num_action_tokens * ACTION_INDEX_BYTES


def select_windows(batch: WindowBatch, picks: np.ndarray) -> WindowBatch:
    """Return the sub-batch at the given positions.

    The gather runs through numpy, so the pool stays on the host until a batch
    is wanted.

    Args:
        batch: The batch to select from.
        picks: Positions to take, which may repeat.

    Returns:
        A batch holding those positions, with the source's split and
        observation mode unchanged.
    """
    return WindowBatch(
        states=jnp.asarray(np.asarray(batch.states)[picks]),
        actions=jnp.asarray(np.asarray(batch.actions)[picks]),
        horizons=jnp.asarray(np.asarray(batch.horizons)[picks]),
        targets=jnp.asarray(np.asarray(batch.targets)[picks]),
        split=batch.split,
        horizon_counts=jnp.asarray(np.asarray(batch.horizon_counts)[picks]),
        observation_mode=batch.observation_mode,
    )


def write_window_batch(path: Path, batch: WindowBatch) -> None:
    """Write one window batch to HDF5, matching trajectory_store's format.

    Args:
        path: Destination file. Its parent must exist.
        batch: The batch to store.
    """
    with h5py.File(path, "w") as handle:
        handle.create_dataset(STATES_DATASET, data=np.asarray(batch.states))
        handle.create_dataset(ACTIONS_DATASET, data=np.asarray(batch.actions))
        handle.create_dataset(HORIZONS_DATASET, data=np.asarray(batch.horizons))
        handle.create_dataset(TARGETS_DATASET, data=np.asarray(batch.targets))
        handle.create_dataset(
            HORIZON_COUNTS_DATASET, data=np.asarray(batch.horizon_counts)
        )
        handle.attrs[SPLIT_ATTR] = batch.split.value
        handle.attrs[OBSERVATION_MODE_ATTR] = batch.observation_mode.value


def read_window_batch(path: Path) -> WindowBatch:
    """Read a window batch written by write_window_batch.

    Args:
        path: File to read.

    Returns:
        The stored batch, with its split and observation mode read back from
        the file attributes.
    """
    with h5py.File(path, "r") as handle:
        return WindowBatch(
            states=jnp.asarray(handle[STATES_DATASET][()]),
            actions=jnp.asarray(handle[ACTIONS_DATASET][()]),
            horizons=jnp.asarray(handle[HORIZONS_DATASET][()]),
            targets=jnp.asarray(handle[TARGETS_DATASET][()]),
            split=SplitName(handle.attrs[SPLIT_ATTR]),
            horizon_counts=jnp.asarray(handle[HORIZON_COUNTS_DATASET][()]),
            observation_mode=ObservationMode(
                handle.attrs[OBSERVATION_MODE_ATTR]
            ),
        )


class WindowSampler:  # pylint: disable=too-many-instance-attributes
    """Draws (s_t, a_{t:t+h}, s_{t+h}) examples from one split.

    The pool is numpy in host memory, the PRNG is JAX.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        trajectories: Sequence[Trajectory],
        indices: Sequence[int],
        *,
        split: SplitName,
        observation_mode: ObservationMode,
        horizon_min: int,
        horizon_max: int,
        batch_size: int,
        mode: SamplerMode = SamplerMode.HYBRID,
        artefact_dir: Path | None = None,
        offline_byte_budget: int = OFFLINE_BYTE_BUDGET,
        evaluation_horizons: Sequence[int] | None = None,
        evaluation_windows_per_trajectory: int = 1,
        offline_pool_size: int = 0,
    ) -> None:
        """Materialise one split's trajectories into a rectangular pool.

        Args:
            trajectories: Every trajectory in the dataset, in store order.
            indices: Which of them this split owns, read from a
                TrajectorySplit.
            split: The partition being sampled, carried into every batch.
            observation_mode: The single mode this sampler emits.
            horizon_min: Smallest horizon to draw.
            horizon_max: Largest horizon to draw, and the padded action length
                for training batches.
            batch_size: Examples per training batch.
            mode: Where training and evaluation windows come from.
            artefact_dir: Directory for the frozen evaluation file and the
                OFFLINE pool. Required by every mode except ONLINE.
            offline_byte_budget: Ceiling on what OFFLINE may materialise.
            evaluation_horizons: The evaluation grid. Defaults to horizon_max
                alone and is not bounded by it.
            evaluation_windows_per_trajectory: Windows drawn from each
                supporting trajectory at each evaluation horizon.
            offline_pool_size: Windows OFFLINE materialises before reading
                training batches back.

        Raises:
            ValueError: If the split is empty, the horizon range is malformed,
                or any horizon in the range is supported by no trajectory in
                this split. The range is never quietly narrowed.
        """
        if not indices:
            raise ValueError(
                f"the {split.value} split holds no trajectories: a sampler "
                "over an empty split can draw nothing"
            )
        if horizon_min < 1:
            raise ValueError(
                f"horizon_min must be at least 1, got {horizon_min}. "
                "h = 0 is not a prediction"
            )
        if horizon_max < horizon_min:
            raise ValueError(
                f"horizon_max ({horizon_max}) is below horizon_min "
                f"({horizon_min})"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        self.split = split
        self.observation_mode = observation_mode
        self.horizon_min = horizon_min
        self.horizon_max = horizon_max
        self.batch_size = batch_size
        self.mode = mode
        self.artefact_dir = artefact_dir
        self.offline_byte_budget = offline_byte_budget
        self.evaluation_horizons = (
            (horizon_max,)
            if evaluation_horizons is None
            else tuple(int(horizon) for horizon in evaluation_horizons)
        )
        self.evaluation_windows_per_trajectory = evaluation_windows_per_trajectory
        self.offline_pool_size = offline_pool_size
        self._offline_pool: WindowBatch | None = None

        selected = [trajectories[index] for index in indices]
        self._lengths = np.array([len(t) for t in selected], dtype=np.int64)
        self._observations, self._actions = self._materialise_pool(selected)
        # Longest first, so the set supporting horizon h is always a prefix.
        # The trajectory draw is then one uniform integer per example.
        self._by_length = np.argsort(-self._lengths, kind="stable")
        self._sorted_lengths = self._lengths[self._by_length]
        self._support_counts = self._build_support_table()
        self._window_counts = self._build_window_count_table()
        self._check_every_training_horizon_is_supported()

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        trajectories: Sequence[Trajectory],
        split: TrajectorySplit,
        name: SplitName,
    ) -> WindowSampler:
        """Build a sampler for one split from the experiment configuration.

        The single place a config becomes a sampler. The artefact directory
        keys on data_seed, not seed. The frozen evaluation set is a property of
        the dataset and the partition.

        Args:
            config: The composed experiment configuration.
            trajectories: Every trajectory in the dataset, in store order.
            split: The partition these trajectories were split into.
            name: Which partition this sampler draws from.

        Returns:
            A sampler over that split.
        """
        return cls(
            trajectories,
            split.indices_for(name),
            split=name,
            observation_mode=ObservationMode(config.sampler.observation_mode),
            horizon_min=config.train.horizon_min,
            horizon_max=config.train.horizon_max,
            batch_size=config.train.batch_size,
            mode=config.sampler.mode,
            artefact_dir=data_dir(
                config.run_name, config.data_seed, config.fast, config.env.name
            ),
            offline_byte_budget=config.sampler.offline_byte_budget,
            evaluation_horizons=config.sampler.evaluation_horizons,
            evaluation_windows_per_trajectory=(
                config.sampler.evaluation_windows_per_trajectory
            ),
            offline_pool_size=config.train.target_examples,
        )

    def _materialise_pool(
        self, selected: Sequence[Trajectory]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad this split's episodes into two rectangular arrays.

        The padding is never read; a window satisfies t + h <= L by
        construction.

        Args:
            selected: This split's episodes.

        Returns:
            Observations of shape (n, max_length + 1, *field_shape, channels)
            and actions of shape (n, max_length).
        """
        max_length = int(self._lengths.max())
        first = selected[0].observations[self.observation_mode]
        frame_shape = np.asarray(first).shape[1:]
        observations = np.zeros(
            (len(selected), max_length + 1, *frame_shape),
            dtype=np.asarray(first).dtype,
        )
        actions = np.zeros((len(selected), max_length), dtype=np.int32)
        for row, trajectory in enumerate(selected):
            frames = np.asarray(trajectory.observations[self.observation_mode])
            steps = len(trajectory)
            observations[row, : steps + 1] = frames[: steps + 1]
            actions[row, :steps] = np.asarray(trajectory.actions)[:steps]
        return observations, actions

    def _build_support_table(self) -> np.ndarray:
        """Return, for each horizon, how many trajectories can supply it.

        Indexed by horizon, so entry h is the count of episodes with L >= h.
        Entry 0 is the whole split and is never drawn against.

        Returns:
            Integer array of length max_length + 2.
        """
        max_length = int(self._lengths.max())
        horizons = np.arange(max_length + 2)
        return np.array(
            [int((self._lengths >= h).sum()) for h in horizons], dtype=np.int64
        )

    def _build_window_count_table(self) -> np.ndarray:
        """Return, for each horizon, how many distinct windows this split holds.

        sum_i max(0, L_i - h + 1). Goes into every batch and every emitted
        table. Equal sampling frequency across horizons does not mean equal
        information across horizons.

        Returns:
            Integer array of length max_length + 2.
        """
        max_length = int(self._lengths.max())
        horizons = np.arange(max_length + 2)
        return np.array(
            [
                int(np.clip(self._lengths - h + 1, 0, None).sum())
                for h in horizons
            ],
            dtype=np.int64,
        )

    def _check_every_training_horizon_is_supported(self) -> None:
        """Raise if any horizon in the training range has no window at all.

        Raises:
            ValueError: If some h in the range is supported by no trajectory,
                naming the largest horizon that is. The range is never
                narrowed silently.
        """
        supported = self._support_counts[
            self.horizon_min : self.horizon_max + 1
        ]
        if int(supported.min()) > 0:
            return
        longest = int(self._lengths.max())
        raise ValueError(
            f"the {self.split.value} split cannot supply the horizon range "
            f"[{self.horizon_min}, {self.horizon_max}]: its longest episode is "
            f"{longest} steps, so every h above that has no window. Horizons "
            "are drawn uniformly and none can be skipped. Lower horizon_max, "
            "or generate longer episodes"
        )

    def windows_at_horizon(self, horizon: int) -> int:
        """Return how many distinct windows this split holds at one horizon.

        Args:
            horizon: The horizon to count at.

        Returns:
            Distinct (trajectory, start) pairs, zero past the longest episode.
        """
        if horizon >= self._window_counts.size:
            return 0
        return int(self._window_counts[horizon])

    def window_byte_cost(self, num_action_tokens: int | None = None) -> int:
        """Return the measured on-disk cost of one window from this sampler.

        Args:
            num_action_tokens: Padded action length. Defaults to horizon_max,
                which is the training batch's padding.

        Returns:
            Bytes one window occupies.
        """
        tokens = self.horizon_max if num_action_tokens is None else num_action_tokens
        frame_shape = self._observations.shape[2:-1]
        channels = int(self._observations.shape[-1])
        return bytes_per_window(
            tuple(int(axis) for axis in frame_shape),
            channels,
            tokens,
            int(self._observations.dtype.itemsize),
        )

    def _assemble(
        self,
        rows: np.ndarray,
        starts: np.ndarray,
        horizons: np.ndarray,
        num_action_tokens: int,
    ) -> WindowBatch:
        """Gather one batch from pool rows, start indices and horizons.

        Validates the assembled horizons through
        `tokeniser.check_horizons_valid`.

        Args:
            rows: Pool row per example.
            starts: Start index t per example.
            horizons: Horizon h per example.
            num_action_tokens: Length to pad the action sequences to.

        Returns:
            The assembled batch.

        Raises:
            ValueError: If any horizon lies outside
                [MIN_HORIZON, num_action_tokens].
        """
        horizons_device = jnp.asarray(horizons, dtype=jnp.int32)
        check_horizons_valid(horizons_device, num_action_tokens)
        slots = np.arange(num_action_tokens)
        real = slots[None, :] < horizons[:, None]
        # Masked slots index position 0, not past the end. The value is
        # overwritten below, and an out-of-range index would fail the gather.
        action_index = np.where(real, starts[:, None] + slots[None, :], 0)
        actions = np.where(
            real, self._actions[rows[:, None], action_index], PAD_ACTION_INDEX
        )
        return WindowBatch(
            states=jnp.asarray(self._observations[rows, starts]),
            actions=jnp.asarray(actions, dtype=jnp.int32),
            horizons=horizons_device,
            targets=jnp.asarray(self._observations[rows, starts + horizons]),
            split=self.split,
            # int32, not int64. JAX is 32-bit by default here and the largest
            # window count sits well inside the range.
            horizon_counts=jnp.asarray(
                self._window_counts[horizons], dtype=jnp.int32
            ),
            observation_mode=self.observation_mode,
        )

    def next_batch(self, key: jax.Array) -> WindowBatch:
        """Draw one training batch.

        Three explicit key splits, one per draw. The horizon, the trajectory
        among those that support it, and the start position within it. No key
        is reused across two draws.

        Args:
            key: JAX PRNG key.

        Returns:
            One WindowBatch of batch_size examples, carrying its split name and
            per-horizon window counts.

        Raises:
            ValueError: If any drawn horizon is outside
                [MIN_HORIZON, horizon_max].
        """
        horizon_key, trajectory_key, start_key = jax.random.split(key, 3)
        horizons = np.asarray(
            jax.random.randint(
                horizon_key,
                (self.batch_size,),
                self.horizon_min,
                self.horizon_max + 1,
            )
        )
        rows, starts = self._draw_positions(
            trajectory_key, start_key, horizons
        )
        return self._assemble(rows, starts, horizons, self.horizon_max)

    def _draw_positions(
        self,
        trajectory_key: jax.Array,
        start_key: jax.Array,
        horizons: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw a supporting trajectory and a start position for each horizon.

        The trajectory is drawn uniformly from those with L >= h. The pool is
        ordered longest first, so that set is a prefix and the draw is one
        uniform integer below the support count.

        Args:
            trajectory_key: PRNG key for the trajectory draw.
            start_key: PRNG key for the start-position draw.
            horizons: Horizon per example, already drawn.

        Returns:
            Pool row and start index per example, satisfying t + h <= L.
        """
        support = self._support_counts[horizons]
        rank = np.asarray(
            jax.random.randint(
                trajectory_key,
                horizons.shape,
                0,
                jnp.asarray(support),
            )
        )
        rows = self._by_length[rank]
        starts = np.asarray(
            jax.random.randint(
                start_key,
                horizons.shape,
                0,
                jnp.asarray(self._lengths[rows] - horizons + 1),
            )
        )
        return rows, starts

    def frozen_evaluation_set(
        self, key: jax.Array, horizons: Sequence[int]
    ) -> WindowBatch:
        """Draw the evaluation windows once, for HYBRID mode to store.

        The horizons argument is not bounded by horizon_max, so a model trained
        to h_max can be swept past it without retraining. `check_horizons_valid`
        takes num_action_tokens as an argument rather than reading config, so
        the sweep ceiling can be passed. Every supporting trajectory
        contributes the same number of windows at each horizon, giving each
        equal weight.

        Args:
            key: JAX PRNG key. One subkey per horizon, so adding a horizon to
                the grid does not redraw the windows at the others.
            horizons: Evaluation grid. May exceed horizon_max.

        Returns:
            One WindowBatch covering the whole grid, padded to the longest
            requested horizon.

        Raises:
            ValueError: If the grid is empty, or if some horizon is supported
                by no trajectory in this split.
        """
        if not horizons:
            raise ValueError("an empty evaluation grid scores nothing")
        unsupported = [
            horizon
            for horizon in horizons
            if horizon >= self._support_counts.size
            or self._support_counts[horizon] == 0
        ]
        if unsupported:
            raise ValueError(
                f"the {self.split.value} split supports none of the horizons "
                f"{unsupported}: its longest episode is "
                f"{int(self._lengths.max())} steps. An evaluation grid may "
                "exceed horizon_max but it cannot exceed the data"
            )
        num_action_tokens = int(max(horizons))
        rows_parts: list[np.ndarray] = []
        starts_parts: list[np.ndarray] = []
        horizons_parts: list[np.ndarray] = []
        for subkey, horizon in zip(
            jax.random.split(key, len(horizons)), horizons
        ):
            eligible = self._by_length[: int(self._support_counts[horizon])]
            rows = np.repeat(eligible, self.evaluation_windows_per_trajectory)
            drawn = np.asarray(
                jax.random.randint(
                    subkey,
                    rows.shape,
                    0,
                    jnp.asarray(self._lengths[rows] - horizon + 1),
                )
            )
            rows_parts.append(rows)
            starts_parts.append(drawn)
            horizons_parts.append(np.full(rows.shape, horizon, dtype=np.int64))
        return self._assemble(
            np.concatenate(rows_parts),
            np.concatenate(starts_parts),
            np.concatenate(horizons_parts),
            num_action_tokens,
        )

    def _artefact_path(self, template: str) -> Path:
        """Return the path for one of this sampler's stored artefacts.

        Args:
            template: WINDOWS_EVAL_TEMPLATE or WINDOWS_TRAIN_TEMPLATE.

        Returns:
            The scoped path inside artefact_dir.

        Raises:
            ValueError: If no artefact directory was given. ONLINE stores
                nothing and never reaches here; the other two cannot work
                without one.
        """
        if self.artefact_dir is None:
            raise ValueError(
                f"sampler mode {self.mode.value} needs an artefact_dir: only "
                "ONLINE stores nothing"
            )
        return self.artefact_dir / template.format(
            split=self.split.value, mode=self.observation_mode.value
        )

    def evaluation_set(self, key: jax.Array) -> WindowBatch:
        """Return this split's evaluation windows, honouring the sampler mode.

        ONLINE draws them and keeps nothing. HYBRID and OFFLINE read a frozen
        file, writing it on first use. That file is why HYBRID is the reporting
        path. Under ONLINE the windows are only reproducible from seed plus
        sampler code, so a refactor changes them silently. A frozen file cannot
        drift.

        Args:
            key: PRNG key for the draw, fixed for the evaluation set.

        Returns:
            The evaluation batch.

        Raises:
            ValueError: If a frozen file exists whose horizons differ from the
                requested grid.
        """
        horizons = self.evaluation_horizons
        if self.mode is SamplerMode.ONLINE:
            return self.frozen_evaluation_set(key, horizons)
        path = self._artefact_path(WINDOWS_EVAL_TEMPLATE)
        if path.exists():
            batch = read_window_batch(path)
            stored = tuple(sorted({int(h) for h in np.asarray(batch.horizons)}))
            wanted = tuple(sorted(set(horizons)))
            # The filename carries the split and the mode but not the grid, so a
            # changed grid reaches a file drawn for the old one.
            if stored != wanted:
                raise ValueError(
                    f"frozen evaluation set at {safe_rel(path)} holds horizons "
                    f"{stored} but this run asks for {wanted}. Delete the file "
                    "to redraw it; reusing it would score the new grid on the "
                    "old draw."
                )
            return batch
        batch = self.frozen_evaluation_set(key, horizons)
        write_window_batch(path, batch)
        logger.info(
            "froze %d evaluation windows over horizons %s to %s",
            len(batch),
            horizons,
            path.name,
        )
        return batch

    def training_batch(self, key: jax.Array) -> WindowBatch:
        """Return one training batch, honouring the sampler mode.

        HYBRID and ONLINE share this path exactly, asserted bit-identical under
        one key, so any divergence is a defect, not a mode difference.

        Args:
            key: JAX PRNG key.

        Returns:
            One batch of batch_size examples.
        """
        if self.mode is SamplerMode.OFFLINE:
            return self._batch_from_offline_pool(key)
        return self.next_batch(key)

    def _batch_from_offline_pool(self, key: jax.Array) -> WindowBatch:
        """Draw a training batch by index from the materialised pool.

        The pool is written on first use and cached in memory afterwards, so a
        training loop pays the read once, not per step.

        Args:
            key: JAX PRNG key for the index draw.

        Returns:
            One batch of batch_size examples taken from the pool.

        Raises:
            ValueError: If no pool size was configured. OFFLINE cannot decide
                on its own how many windows the dataset should hold.
        """
        # The pool draw and the index draw take separate keys. One key serving
        # both couples the batch to the pool it is drawn from.
        pool_key, picks_key = jax.random.split(key)
        if self._offline_pool is None:
            path = self._artefact_path(WINDOWS_TRAIN_TEMPLATE)
            if not path.exists():
                if self.offline_pool_size < 1:
                    raise ValueError(
                        "OFFLINE mode needs offline_pool_size. from_config "
                        "takes it from train.target_examples"
                    )
                self.materialise_training_windows(pool_key, self.offline_pool_size)
            self._offline_pool = read_window_batch(path)
        pool = self._offline_pool
        picks = np.asarray(
            jax.random.randint(picks_key, (self.batch_size,), 0, len(pool))
        )
        return select_windows(pool, picks)

    def materialise_training_windows(
        self, key: jax.Array, num_windows: int
    ) -> Path:
        """Write a pool of training windows to disk for OFFLINE mode.

        The byte budget is a hard guard and its error names the measured
        per-window cost.

        Args:
            key: PRNG key for the draw.
            num_windows: How many windows to materialise.

        Returns:
            The path written.

        Raises:
            ValueError: If the requested pool exceeds offline_byte_budget.
        """
        per_window = self.window_byte_cost()
        required = per_window * num_windows
        if required > self.offline_byte_budget:
            raise ValueError(
                f"OFFLINE mode refuses to materialise {num_windows} windows. "
                f"At a MEASURED {per_window} bytes per window that is "
                f"{required} bytes, above the {self.offline_byte_budget}-byte "
                "budget. Use HYBRID, or raise the budget"
            )
        horizon_key, trajectory_key, start_key = jax.random.split(key, 3)
        horizons = np.asarray(
            jax.random.randint(
                horizon_key,
                (num_windows,),
                self.horizon_min,
                self.horizon_max + 1,
            )
        )
        rows, starts = self._draw_positions(
            trajectory_key, start_key, horizons
        )
        batch = self._assemble(rows, starts, horizons, self.horizon_max)
        path = self._artefact_path(WINDOWS_TRAIN_TEMPLATE)
        write_window_batch(path, batch)
        logger.info(
            "materialised %d training windows (%d bytes) to %s",
            num_windows,
            required,
            path.name,
        )
        return path
