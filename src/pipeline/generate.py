"""Trajectory generation stage. Rolls a uniform-random policy and stores episodes.

The only stage that touches an environment. It writes a dataset plus the
measurements the sampler and the observability comparison need, using
``NavixEnv``'s compiled batched step. A batched rollout interleaves episodes, so
``cut_episodes`` finds the boundaries per row, and both observation modes are
re-rendered from one rollout's state.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import navix
import numpy as np

from config import (
    DECODED_SAMPLES_FILENAME,
    OBS_CHANNEL_CLASSES_NAVIX,
    TRAJECTORY_SHARD_GLOB,
    TRAJECTORY_SHARD_TEMPLATE,
    TRAJECTORY_STATS_FILENAME,
    ExperimentConfig,
    config_snapshot,
)
from src.data.trajectory import (
    ObservationMode,
    Trajectory,
)
from src.data.trajectory_source import (
    NAVIX_GOAL_ENTITY,
    NAVIX_OBSERVATION_FNS,
    UNIFORM_RANDOM_POLICY,
)
from src.data.trajectory_store import TrajectoryStore
from src.envs.navix_env import NavixEnv
from src.envs.slip import SLIP_KEY_FOLD_INDEX, apply_slip
from src.models.losses import check_observation_values_in_range
from src.pipeline.base import Stage
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

# Hard ceiling on rollout length. The loop's real exit is the episode count.
ROLLOUT_STEP_SAFETY_FACTOR: int = 4

# How many timesteps of a decoded trajectory to render, and which channel.
# Channel 0 is the entity tag.
DECODE_FRAMES: int = 4
DECODE_CHANNEL: int = 0


@dataclass(frozen=True)
class _Rollout:  # pylint: disable=too-many-instance-attributes
    """One batched rollout, as numpy arrays, before it is cut into episodes.

    Observations carry num_steps + 1 entries, index 0 being the reset. Every
    other stream carries num_steps, where entry i - 1 is the result of the step
    reaching index i.

    Attributes:
        observations: Per mode, shape (num_steps + 1, num_envs, *grid, 3).
        actions: Commanded actions, shape (num_steps, num_envs).
        rewards: Shape (num_steps, num_envs).
        terminated: Shape (num_steps, num_envs).
        truncated: Shape (num_steps, num_envs).
        goal_position: Shape (num_steps + 1, num_envs, num_goals, 2).
        timestep_index: NAVIX's own `t`, shape (num_steps + 1, num_envs). A
            return to 0 marks the first observation of a new episode.
        executed_actions: What the environment stepped on, shape
            (num_steps, num_envs). Differs from `actions` only under slip.
    """

    observations: dict[ObservationMode, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    goal_position: np.ndarray
    timestep_index: np.ndarray
    executed_actions: np.ndarray


class GenerateTrajectoriesStage(Stage):
    """Roll a uniform-random policy in NAVIX and write complete trajectories.

    Attributes:
        store: The trajectory store this stage writes through.
    """

    name: str = "generate"
    dataset_derived: bool = True

    def __init__(self, config: ExperimentConfig) -> None:
        """Build the stage."""
        super().__init__(config)
        self.store = TrajectoryStore()

    def run(self) -> None:
        """Generate, cut, store and measure the trajectories.

        Writes under the data seed, not the run seed, so runs varying only the
        model seed share one dataset.
        """
        trajectories = self._generate()
        # Before the first shard is written.
        self._assert_cardinality(trajectories)
        target = self.dataset_dir
        ensure_dir(target)
        self._write_shards(trajectories, target)
        self._write_statistics(trajectories, target)
        self._write_decoded_samples(trajectories, target)

    def _generate(self) -> list[Trajectory]:
        """Roll the environment until enough complete episodes exist.

        Returns:
            Exactly `config.data.num_trajectories` complete episodes, or every
            complete episode found if the safety ceiling was hit first.
        """
        env = NavixEnv(self.config.env)
        rollout = self._roll(env)
        trajectories = self.cut_episodes(rollout)
        wanted = self.config.data.num_trajectories
        if len(trajectories) < wanted:
            logger.warning(
                "collected %d complete trajectories, wanted %d: the rollout hit "
                "its safety ceiling. Episodes may be longer than "
                "max_episode_steps implies",
                len(trajectories),
                wanted,
            )
        return trajectories[:wanted]

    def _roll(self, env: NavixEnv) -> _Rollout:
        """Step the batched environment until enough episodes have ended.

        Args:
            env: The batched NAVIX adapter.

        Returns:
            The accumulated rollout.
        """
        wanted = self.config.data.num_trajectories
        num_envs = self.config.env.num_envs
        num_actions = int(env.env.action_space.maximum) + 1
        # Enough for `wanted` episodes at the full cap, with a safety factor.
        max_steps = ROLLOUT_STEP_SAFETY_FACTOR * (
            -(-wanted // num_envs) * (self.config.env.max_episode_steps + 1)
        )

        # The data seed, matching the directory the shards go to.
        key = jax.random.PRNGKey(self.config.data_seed)
        key, reset_key = jax.random.split(key)
        timestep = env.reset(reset_key)
        streams = _StreamAccumulator(self._observe(timestep))
        streams.add_reset(timestep)

        completed = 0
        for _ in range(max_steps):
            key, step_key = jax.random.split(key)
            action = jax.random.randint(step_key, (num_envs,), 0, num_actions)
            # Folded, not split: step_key must stay usable above or the
            # commanded action itself changes and slip_probability 0.0 stops
            # reproducing a pre-slip run.
            executed = apply_slip(
                action,
                jax.random.fold_in(step_key, SLIP_KEY_FOLD_INDEX),
                self.config.env.slip_probability,
                num_actions,
            )
            timestep = env.step(timestep, executed)
            streams.add_step(timestep, action, self._observe(timestep), executed)
            completed += int(np.asarray(NavixEnv.done(timestep)).sum())
            if completed >= wanted:
                break

        logger.info(
            "rollout finished: %d steps x %d envs, %d episode ends observed",
            streams.num_steps,
            num_envs,
            completed,
        )
        return streams.finish()

    @staticmethod
    def _observe(
        timestep: navix.environments.Timestep,
    ) -> dict[ObservationMode, np.ndarray]:
        """Render every observation mode from one batched timestep's state.

        Args:
            timestep: Batched Timestep.

        Returns:
            One numpy array per mode, batch axis retained, converted off-device
            immediately.
        """
        return {
            mode: np.asarray(jax.vmap(fn)(timestep.state))
            for mode, fn in NAVIX_OBSERVATION_FNS.items()
        }

    def cut_episodes(self, rollout: _Rollout) -> list[Trajectory]:
        """Split a batched rollout into complete per-environment episodes.

        The cut rule:

        - An episode's first observation is one where `t == 0`, meaning index 0
          for a row's first episode and the auto-reset thereafter.
        - Its last is the observation at the step where `done()` is True, which
          is the true endpoint and not the reset.
        - The reset lands on the following step, for a termination exactly as
          for a truncation.
        - The action at a reset step is discarded.
        - `terminated` and `truncated` are carried separately.
        - A partial trailing episode is discarded, never padded.

        Returned in completion order, ties broken by row.

        Args:
            rollout: The accumulated batched rollout.

        Returns:
            Complete episodes, deterministically ordered.
        """
        spans = []
        for row in range(rollout.timestep_index.shape[1]):
            spans.extend((end, row, start) for start, end in _row_spans(rollout, row))
        return [
            self._build_trajectory(rollout, row, start, end)
            for end, row, start in sorted(spans)
        ]

    def _build_trajectory(
        self, rollout: _Rollout, row: int, start: int, end: int
    ) -> Trajectory:
        """Assemble one Trajectory from a row's episode span.

        Args:
            rollout: The accumulated rollout.
            row: Environment index.
            start: Index of the episode's first observation, where `t == 0`.
            end: Index of the observation at which the episode ended.

        Returns:
            The episode, carrying end - start actions and end - start + 1
            observations per mode.
        """
        # Observations are inclusive of both ends; every other stream is offset
        # by one.
        return Trajectory(
            observations={
                mode: frames[start : end + 1, row]
                for mode, frames in rollout.observations.items()
            },
            actions=rollout.actions[start:end, row],
            rewards=rollout.rewards[start:end, row],
            terminated=rollout.terminated[start:end, row],
            truncated=rollout.truncated[start:end, row],
            goal_position=rollout.goal_position[start, row],
            provenance={
                "env_name": self.config.env.name,
                "max_episode_steps": self.config.env.max_episode_steps,
                "policy": UNIFORM_RANDOM_POLICY,
                "seed": self.config.data_seed,
                "env_row": row,
                "rollout_start_index": start,
            },
            executed_actions=rollout.executed_actions[start:end, row],
        )

    def _assert_cardinality(self, trajectories: Sequence[Trajectory]) -> None:
        """Raise if any observation code exceeds its declared class count.

        Reuses the loss's checker. Never widen the declaration to fit what was
        measured, which sizes an output head for an unexplained value.

        Args:
            trajectories: Episodes to check, in both observation modes.

        Raises:
            ValueError: If any code is out of range, naming the mode, the
                channel, the observed maximum and the declaration.
        """
        for mode in ObservationMode:
            for trajectory in trajectories:
                frames = np.asarray(trajectory.observations[mode])
                grid_shape = frames.shape[1:3]
                try:
                    check_observation_values_in_range(
                        frames.reshape(frames.shape[0], -1),
                        (int(grid_shape[0]), int(grid_shape[1])),
                        OBS_CHANNEL_CLASSES_NAVIX,
                    )
                except ValueError as error:
                    observed = [
                        int(frames[..., channel].max())
                        for channel in range(frames.shape[-1])
                    ]
                    raise ValueError(
                        f"{self.config.env.name} emits observation codes outside "
                        f"OBS_CHANNEL_CLASSES_NAVIX {OBS_CHANNEL_CLASSES_NAVIX} in "
                        f"{mode.value}: observed per-channel maxima {observed}. "
                        f"Nothing has been written. {error}"
                    ) from error
        logger.info(
            "cardinality verified for %d trajectories in both modes against %s",
            len(trajectories),
            OBS_CHANNEL_CLASSES_NAVIX,
        )

    def _channel_measurements(self, trajectories: Sequence[Trajectory]) -> dict:
        """Measure per-channel observed maxima and informativeness, per mode.

        Recorded, never acted on. Every channel is stored raw so the scored set
        can be chosen at interpretation time.

        A channel is uninformative when it holds one value across the dataset,
        which a model then scores for free.

        Args:
            trajectories: Episodes to measure.

        Returns:
            Mapping of observation mode to per-channel maxima, distinct-value
            counts and an informative flag.
        """
        measurements: dict[str, dict] = {}
        for mode in ObservationMode:
            frames = np.concatenate(
                [np.asarray(t.observations[mode]) for t in trajectories]
            )
            channels = frames.shape[-1]
            per_channel = []
            for channel in range(channels):
                values = frames[..., channel]
                distinct = int(np.unique(values).size)
                per_channel.append(
                    {
                        "channel": channel,
                        "observed_max": int(values.max()),
                        "observed_min": int(values.min()),
                        "distinct_values": distinct,
                        "declared_classes": OBS_CHANNEL_CLASSES_NAVIX[channel],
                        "informative": distinct > 1,
                    }
                )
            measurements[mode.value] = {
                "channels": per_channel,
                "informative_channels": [
                    entry["channel"] for entry in per_channel if entry["informative"]
                ],
            }
        return measurements

    def sentinel_identity(self) -> dict:
        """Return the identity and integrity fields guarding this dataset.

        Returns:
            JSON-serialisable identity and integrity fields.
        """
        identity = super().sentinel_identity()
        shards = sorted(self.dataset_dir.glob(TRAJECTORY_SHARD_GLOB))
        identity.update(
            {
                "num_trajectories": self.config.data.num_trajectories,
                "shard_size": self.config.data.shard_size,
                "storage_format": self.config.data.storage_format,
                "max_episode_steps": self.config.env.max_episode_steps,
                # Both act during generation, so a dataset carries the settings
                # it was generated under. Left out, a rerun of one run name at a
                # different setting matches this sentinel and trains on data
                # generated under another.
                "slip_probability": self.config.env.slip_probability,
                "disable_early_termination": (
                    self.config.env.disable_early_termination
                ),
                "observation_modes": [mode.value for mode in ObservationMode],
                "split_fractions": [
                    self.config.data.train_fraction,
                    self.config.data.validation_fraction,
                    self.config.data.test_fraction,
                ],
                "shard_count": len(shards),
                "shard_sizes": [shard.stat().st_size for shard in shards],
            }
        )
        return identity

    def _write_shards(self, trajectories: list[Trajectory], target: Path) -> None:
        """Write the trajectories out in fixed-size shards.

        Args:
            trajectories: Episodes to write.
            target: Directory to write into.
        """
        shard_size = self.config.data.shard_size
        for index, offset in enumerate(range(0, len(trajectories), shard_size)):
            self.store.write_shard(
                trajectories[offset : offset + shard_size],
                target / TRAJECTORY_SHARD_TEMPLATE.format(index=index),
            )

    def _write_statistics(
        self, trajectories: list[Trajectory], target: Path
    ) -> None:
        """Write the length histogram and the copy and mover statistics.

        Args:
            trajectories: Episodes to measure.
            target: Directory to write into.
        """
        horizons = range(
            self.config.train.horizon_min, self.config.train.horizon_max + 1
        )
        stats = {
            "provenance": config_snapshot(self.config),
            "lengths": self.store.length_statistics(trajectories),
            "copy_and_mover": self.store.copy_and_mover_statistics(
                trajectories, list(horizons)
            ),
            # Reads the same arrays as the cardinality assert.
            "channels": self._channel_measurements(trajectories),
        }
        path = target / TRAJECTORY_STATS_FILENAME
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        lengths = stats["lengths"]
        logger.info(
            "episode lengths: count=%s min=%s max=%s mean=%s median=%s -> %s",
            lengths["count"],
            lengths["min"],
            lengths["max"],
            lengths["mean"],
            lengths["median"],
            safe_rel(path),
        )

    def _write_decoded_samples(
        self, trajectories: list[Trajectory], target: Path
    ) -> None:
        """Render a few trajectories to a text file for hand inspection.

        Args:
            trajectories: Episodes to sample from.
            target: Directory to write into.
        """
        path = target / DECODED_SAMPLES_FILENAME
        sample = trajectories[: self.config.data.decode_samples]
        blocks = [
            _decode_trajectory(index, trajectory)
            for index, trajectory in enumerate(sample)
        ]
        path.write_text(_decode_header(sample, trajectories) + "\n".join(blocks),
                        encoding="utf-8")
        logger.info("decoded %d trajectories -> %s", len(blocks), safe_rel(path))


class _StreamAccumulator:
    """Collects per-step rollout arrays and stacks them once at the end.

    Growing a numpy array per step would reallocate the whole rollout each step.

    Attributes:
        num_steps: Steps accumulated so far, excluding the reset.
    """

    def __init__(self, reset_observations: dict[ObservationMode, np.ndarray]) -> None:
        """Start an accumulator from the reset observations.

        Args:
            reset_observations: Per-mode observations at the reset.
        """
        self._observations = {
            mode: [frame] for mode, frame in reset_observations.items()
        }
        self._streams: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "actions", "rewards", "terminated", "truncated",
                "executed_actions",
            )
        }
        self._goal_position: list[np.ndarray] = []
        self._timestep_index: list[np.ndarray] = []
        self.num_steps = 0

    def add_reset(self, timestep: navix.environments.Timestep) -> None:
        """Record the per-step fields that exist at the reset.

        Args:
            timestep: The batched Timestep returned by reset().
        """
        self._add_state_fields(timestep)

    def add_step(
        self,
        timestep: navix.environments.Timestep,
        action: jax.Array,
        observations: dict[ObservationMode, np.ndarray],
        executed_action: jax.Array,
    ) -> None:
        """Record one step's observations, actions and outcome flags.

        Args:
            timestep: The batched Timestep returned by step().
            action: Actions commanded at this step, shape (num_envs,).
            observations: Per-mode observations rendered from this timestep.
            executed_action: Actions the environment stepped on, shape
                (num_envs,). Equal to `action` unless slip was injected.
        """
        for mode, frame in observations.items():
            self._observations[mode].append(frame)
        self._streams["actions"].append(np.asarray(action))
        self._streams["executed_actions"].append(np.asarray(executed_action))
        self._streams["rewards"].append(np.asarray(timestep.reward))
        self._streams["terminated"].append(np.asarray(NavixEnv.terminated(timestep)))
        self._streams["truncated"].append(np.asarray(NavixEnv.truncated(timestep)))
        self._add_state_fields(timestep)
        self.num_steps += 1

    def _add_state_fields(self, timestep: navix.environments.Timestep) -> None:
        """Record the fields sampled at every observation index, reset included.

        Args:
            timestep: A batched Timestep.
        """
        self._goal_position.append(
            np.asarray(timestep.state.entities[NAVIX_GOAL_ENTITY].position)
        )
        self._timestep_index.append(np.asarray(timestep.t))

    def finish(self) -> _Rollout:
        """Stack everything accumulated into a _Rollout.

        Returns:
            The rollout, as numpy arrays.
        """
        return _Rollout(
            observations={
                mode: np.stack(frames)
                for mode, frames in self._observations.items()
            },
            actions=np.stack(self._streams["actions"]),
            rewards=np.stack(self._streams["rewards"]),
            terminated=np.stack(self._streams["terminated"]),
            truncated=np.stack(self._streams["truncated"]),
            goal_position=np.stack(self._goal_position),
            timestep_index=np.stack(self._timestep_index),
            executed_actions=np.stack(self._streams["executed_actions"]),
        )


def _row_spans(rollout: _Rollout, row: int) -> list[tuple[int, int]]:
    """Return one row's complete episodes as (start, end) observation indices.

    Args:
        rollout: The accumulated rollout.
        row: Environment index.

    Returns:
        Inclusive (start, end) index pairs, one per COMPLETE episode. A trailing
        run of observations with no episode end is omitted.
    """
    starts = np.flatnonzero(rollout.timestep_index[:, row] == 0)
    # done is indexed from 1: entry i - 1 describes the step reaching index i.
    ends = np.flatnonzero(rollout.terminated[:, row] | rollout.truncated[:, row]) + 1
    spans = []
    for start in starts:
        following = ends[ends > start]
        if following.size:
            spans.append((int(start), int(following[0])))
    return spans


def _decode_header(
    sample: Sequence[Trajectory], trajectories: Sequence[Trajectory]
) -> str:
    """Return a header stating how the sample was chosen and how it differs.

    Args:
        sample: The trajectories actually rendered below.
        trajectories: The whole dataset, for the comparison figures.

    Returns:
        A header block, ending in a blank line.
    """
    def terminated_fraction(items: Sequence[Trajectory]) -> float:
        if not items:
            return 0.0
        ended = sum(
            bool(np.asarray(item.terminated).any()) for item in items
        )
        return ended / len(items)

    return (
        f"# Hand-decode sample: the first {len(sample)} episodes in "
        "completion order, so the\n"
        "# shortest, and short episodes disproportionately reached the goal.\n"
        f"# Terminated here: {terminated_fraction(sample):.1%}. "
        f"In the dataset: {terminated_fraction(trajectories):.1%} "
        f"of {len(trajectories)}.\n"
        "# Read this to check grids and fields. trajectory_stats.json carries "
        "the distribution.\n\n"
    )


def _decode_trajectory(index: int, trajectory: Trajectory) -> str:
    """Render one trajectory's opening frames as readable text.

    Args:
        index: Position of this trajectory in the dataset, for the heading.
        trajectory: The episode to render.

    Returns:
        A text block showing both observation modes side by side per frame.
    """
    lines = [
        f"=== trajectory {index} "
        f"(length {len(trajectory)}, "
        f"terminated={bool(np.asarray(trajectory.terminated).any())}, "
        f"truncated={bool(np.asarray(trajectory.truncated).any())}) ===",
        f"provenance: {trajectory.provenance}",
        f"goal_position: {np.asarray(trajectory.goal_position).tolist()}",
        f"actions: {np.asarray(trajectory.actions).tolist()}",
    ]
    for frame in range(min(DECODE_FRAMES, len(trajectory) + 1)):
        lines.append(f"-- frame {frame} (channel {DECODE_CHANNEL}) --")
        for mode in sorted(trajectory.observations, key=lambda m: m.value):
            grid = np.asarray(trajectory.observations[mode])[frame, ..., DECODE_CHANNEL]
            lines.append(f"  {mode.value}:")
            lines.extend(
                "    " + " ".join(f"{int(cell):2d}" for cell in grid_row)
                for grid_row in grid
            )
    return "\n".join(lines) + "\n"
