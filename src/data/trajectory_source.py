"""One interface for reading offline training data.

A source yields finished episodes and describes its own observations.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import navix

from config import (
    DEFAULT_STORED_OBSERVATION_MODES,
    OBS_CHANNEL_CLASSES_NAVIX,
    OBS_VALUE_RANGE_RGB,
    POLICY_UNIFORM_RANDOM,
    REPRESENTATION_RGB,
    REPRESENTATION_SYMBOLIC,
    EnvConfig,
)
from src.data.trajectory import (
    FieldSpec,
    ObservationMode,
    ObservationSpec,
    Trajectory,
)
from src.envs.navix_env import NavixEnv
from src.envs.slip import SLIP_KEY_FOLD_INDEX, apply_slip
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# NAVIX observation functions per representation and mode, all derived from the
# same state.
NAVIX_OBSERVATION_FNS = {
    REPRESENTATION_SYMBOLIC: {
        ObservationMode.TOP_DOWN: navix.observations.symbolic,
        ObservationMode.EGOCENTRIC: navix.observations.symbolic_first_person,
    },
    REPRESENTATION_RGB: {
        ObservationMode.TOP_DOWN: navix.observations.rgb,
        ObservationMode.EGOCENTRIC: navix.observations.rgb_first_person,
    },
}

# Name of the goal entity in a NAVIX state's entity dictionary.
NAVIX_GOAL_ENTITY: str = "goal"

# Field name reported by the observation spec. NAVIX has exactly one field.
NAVIX_GRID_FIELD: str = "grid"

# The generating policy recorded in every trajectory's provenance.
UNIFORM_RANDOM_POLICY: str = POLICY_UNIFORM_RANDOM


@runtime_checkable
class TrajectorySource(Protocol):
    """A source of complete trajectories for offline training."""

    def spec(self, mode: ObservationMode) -> ObservationSpec:
        """Describe this source's observations and action space for one mode.

        Args:
            mode: The observation mode being described. Shape differs per mode.

        Returns:
            The specification a tokeniser is constructed from.
        """

    def trajectories(self, rng_key: jax.Array) -> Iterator[Trajectory]:
        """Yield complete trajectories, deterministic given rng_key.

        Args:
            rng_key: PRNG key. The same key yields the same trajectories.

        Yields:
            One Trajectory per finished episode.
        """


class NavixTrajectorySource:
    """Rolls a uniform-random policy in NAVIX and emits complete episodes.

    The stored observation modes are recorded per episode from the same state,
    and termination and truncation are kept as separate streams rather than one
    `done` flag. NAVIX auto-resets internally, so the observation at the
    boundary step is the endpoint of the episode that just ended. The fresh
    reset appears on the next step, where `timestep.t` returns to 0.

    Attributes:
        config: Environment configuration.
        representation: Selects the observation functions and the field domain.
        stored_modes: The observation modes recorded per episode.
        env: The underlying NAVIX adapter, built with num_envs forced to 1.
    """

    def __init__(
        self,
        config: EnvConfig,
        *,
        representation: str = REPRESENTATION_SYMBOLIC,
        stored_modes: tuple[str, ...] | None = None,
    ) -> None:
        """Build a single-environment NAVIX rollout source.

        Args:
            config: Environment configuration. num_envs is overridden to 1, so
                the episode boundary is unambiguous.
            representation: Symbolic or RGB, selecting the observation
                functions and the field's declared domain.
            stored_modes: Modes recorded per episode. None takes the
                representation's default, which differs between
                representations.
        """
        self.config = config
        self.representation = representation
        resolved = (
            DEFAULT_STORED_OBSERVATION_MODES[representation]
            if stored_modes is None
            else stored_modes
        )
        self.stored_modes = tuple(ObservationMode(mode) for mode in resolved)
        self.observation_fns = NAVIX_OBSERVATION_FNS[representation]
        # A field-by-field rebuild would silently drop any field the dataclass
        # gains.
        self.env = NavixEnv(replace(config, num_envs=1))
        logger.info(
            "NavixTrajectorySource ready: name=%s max_steps=%d policy=%s",
            config.name,
            config.max_episode_steps,
            UNIFORM_RANDOM_POLICY,
        )

    def spec(self, mode: ObservationMode) -> ObservationSpec:
        """Describe NAVIX's observations and action space for one mode.

        Args:
            mode: The observation mode being described.

        Returns:
            A single-field spec carrying the grid shape for this mode and the
            per-channel class counts.
        """
        probe = self._observe(self.env.reset(jax.random.PRNGKey(0)), mode)
        grid_shape = tuple(int(dim) for dim in probe.shape[1:-1])
        domain = (
            {"value_range": OBS_VALUE_RANGE_RGB}
            if self.representation == REPRESENTATION_RGB
            else {"cardinality": tuple(OBS_CHANNEL_CLASSES_NAVIX)}
        )
        return ObservationSpec(
            fields=(
                FieldSpec(name=NAVIX_GRID_FIELD, shape=grid_shape, **domain),
            ),
            num_actions=int(self.env.env.action_space.maximum) + 1,
        )

    def trajectories(self, rng_key: jax.Array) -> Iterator[Trajectory]:
        """Yield complete episodes indefinitely, deterministic given rng_key.

        One key is split per episode, so no key is reused across two draws.

        Args:
            rng_key: PRNG key seeding the whole sequence.

        Yields:
            One Trajectory per finished episode, in both observation modes.
        """
        key = rng_key
        episode_index = 0
        while True:
            key, episode_key = jax.random.split(key)
            yield self._rollout_one(episode_key, episode_index)
            episode_index += 1

    def _observe(self, timestep: navix.environments.Timestep,
                 mode: ObservationMode) -> jax.Array:
        """Derive one observation mode from a batched timestep's state.

        Args:
            timestep: Batched Timestep, leading axis of size 1.
            mode: Which view to render.

        Returns:
            Observation array with the batch axis retained.
        """
        return jax.vmap(self.observation_fns[mode])(timestep.state)

    def _rollout_one(self, episode_key: jax.Array,
                     episode_index: int) -> Trajectory:
        """Roll a single episode to its boundary and assemble a Trajectory.

        Args:
            episode_key: PRNG key for this episode's reset and action draws.
            episode_index: Position in the yielded sequence, recorded in
                provenance.

        Returns:
            The finished episode, with num_steps + 1 observations per mode.
        """
        reset_key, action_key = jax.random.split(episode_key)
        timestep = self.env.reset(reset_key)
        num_actions = int(self.env.env.action_space.maximum) + 1

        observations: dict[ObservationMode, list[jax.Array]] = {
            mode: [self._observe(timestep, mode)[0]] for mode in self.stored_modes
        }
        streams: dict[str, list[jax.Array]] = {
            name: []
            for name in (
                "actions", "rewards", "terminated", "truncated",
                "executed_actions",
            )
        }

        # NAVIX truncates at max_episode_steps, so this bound is a guard
        # against an infinite loop.
        for _ in range(self.config.max_episode_steps):
            action_key, step_key = jax.random.split(action_key)
            action = jax.random.randint(step_key, (1,), 0, num_actions)
            # Folded, not split: step_key must stay usable above or the
            # commanded action itself changes and slip_probability 0.0 stops
            # reproducing a pre-slip run.
            executed = apply_slip(
                action,
                jax.random.fold_in(step_key, SLIP_KEY_FOLD_INDEX),
                self.config.slip_probability,
                num_actions,
            )
            timestep = self.env.step(timestep, executed)

            streams["actions"].append(action[0])
            streams["executed_actions"].append(executed[0])
            streams["rewards"].append(timestep.reward[0])
            streams["terminated"].append(NavixEnv.terminated(timestep)[0])
            streams["truncated"].append(NavixEnv.truncated(timestep)[0])
            for mode in self.stored_modes:
                observations[mode].append(self._observe(timestep, mode)[0])

            if bool(NavixEnv.done(timestep)[0]):
                break

        return Trajectory(
            observations={
                mode: jnp.stack(frames) for mode, frames in observations.items()
            },
            actions=jnp.stack(streams["actions"]),
            rewards=jnp.stack(streams["rewards"]),
            terminated=jnp.stack(streams["terminated"]),
            truncated=jnp.stack(streams["truncated"]),
            goal_position=timestep.state.entities[NAVIX_GOAL_ENTITY].position[0],
            provenance={
                "env_name": self.config.name,
                "max_episode_steps": self.config.max_episode_steps,
                "policy": UNIFORM_RANDOM_POLICY,
                "episode_index": episode_index,
            },
            executed_actions=jnp.stack(streams["executed_actions"]),
        )
