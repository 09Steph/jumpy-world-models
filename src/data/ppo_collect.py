"""Collect trajectories from a PPO policy's own training experience.

navix's PPO exposes each update's buffer only by passing it to ``Agent.log``
under ``PPOHparams.debug``, so collection overrides that method.

``Buffer`` pairs ``info`` at t + 1 with ``state`` and ``t`` at t, so the
termination flags and executed actions carried in ``info`` describe the step
that reached t + 1.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
import navix
import numpy as np
from flax import struct
from navix.agents import PPO, ActorCritic, ConvEncoder, PPOHparams
from navix.environments.environment import Environment, Timestep

from config import EnvConfig
from src.data.trajectory import ObservationMode
from src.data.trajectory_source import NAVIX_GOAL_ENTITY
from src.envs.navix_env import NavixEnv
from src.envs.slip import SLIP_KEY_FOLD_INDEX, apply_slip
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# Keys the environment wrapper adds to `Timestep.info`, which navix's PPO
# copies into `Buffer.info` unchanged.
STEP_TYPE_INFO_KEY: str = "step_type"
EXECUTED_ACTION_INFO_KEY: str = "executed_action"

# Which navix update log carries the update index, used to restore call order.
UPDATE_INDEX_LOG_KEY: str = "iter/updates"


def _with_recorded_info(timestep: Timestep, executed: jax.Array) -> Timestep:
    """Return the timestep with step_type and the executed action in info.

    Args:
        timestep: The timestep to annotate.
        executed: The action the environment executed.
    """
    return timestep.replace(
        info={
            **timestep.info,
            STEP_TYPE_INFO_KEY: timestep.step_type,
            EXECUTED_ACTION_INFO_KEY: jnp.asarray(executed, dtype=jnp.int32),
        }
    )


def _slip_key(timestep: Timestep) -> jax.Array:
    """Return one step's slip key.

    `State.key` is advanced on reset and never inside a step, so a key derived
    from it alone repeats for every step of one episode. The timestep index is
    folded in to separate them.

    Args:
        timestep: The timestep at time t.
    """
    return jax.random.fold_in(
        jax.random.fold_in(timestep.state.key, timestep.t), SLIP_KEY_FOLD_INDEX
    )


@functools.lru_cache(maxsize=None)
def slip_and_info_class(base_class: type) -> type:
    """Return a subclass of one navix environment that records what it executed.

    The executed action differs from the commanded one under slip, and
    `step_type` separates a termination from a truncation where `done` merges
    them. Neither is recoverable from the buffer without this.

    Args:
        base_class: The concrete navix environment class to extend.

    Returns:
        A subclass carrying `slip_probability` and `num_actions`.
    """

    class _SlipAndInfoEnvironment(base_class):  # pylint: disable=too-few-public-methods
        """One navix environment, recording step_type and the executed action."""

        slip_probability: float = struct.field(pytree_node=False, default=0.0)
        num_actions: int = struct.field(pytree_node=False, default=0)

        def reset(self, key: jax.Array, cache=None) -> Timestep:
            """Reset, seeding the info keys every later timestep carries.

            navix's `step` selects between `reset` and `_step` with `lax.cond`,
            so both branches must add the keys.

            Args:
                key: PRNG key for the reset.
                cache: Rendering cache to reuse, or None to build one.
            """
            return _with_recorded_info(
                super().reset(key, cache), jnp.asarray(0, dtype=jnp.int32)
            )

        def _step(self, timestep: Timestep, action: jax.Array) -> Timestep:
            """Advance one step, applying slip and recording what it executed.

            Overrides the inner step, so slip never touches an auto-reset's
            discarded action.

            Args:
                timestep: The timestep at time t.
                action: The commanded action.
            """
            executed = apply_slip(
                action,
                _slip_key(timestep),
                self.slip_probability,
                self.num_actions,
            )
            return _with_recorded_info(
                super()._step(timestep, executed), executed
            )

    return _SlipAndInfoEnvironment


def build_collection_environment(
    config: EnvConfig, observation_fn: Callable
) -> Environment:
    """Return the wrapped environment PPO trains and collects on.

    Args:
        config: Environment configuration, supplying the slip probability.
        observation_fn: The navix observation function the policy reads.

    Returns:
        The registered environment, extended to record step_type and the
        executed action.
    """
    base = NavixEnv(config, observation_fn=observation_fn).env
    fields = {
        field.name: getattr(base, field.name)
        for field in dataclasses.fields(base)
    }
    return slip_and_info_class(type(base))(
        **fields,
        slip_probability=config.slip_probability,
        num_actions=int(base.action_space.maximum) + 1,
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def collect_rollout_arrays(
    config: EnvConfig,
    budget_frames: int,
    num_steps: int,
    entropy_coefficient: float,
    observation_fn: Callable,
    observation_fns: dict[ObservationMode, Callable],
    rng: jax.Array,
) -> dict[str, Any]:
    """Train PPO and return the experience it trained on, as rollout arrays.

    Args:
        config: Environment configuration. `num_envs` sizes PPO's batch.
        budget_frames: Environment frames to train for.
        num_steps: Steps per parallel environment per update.
        entropy_coefficient: PPO's entropy bonus.
        observation_fn: The view the policy reads.
        observation_fns: Views to render and store, one per mode.
        rng: PRNG key seeding initialisation, collection and training.

    Returns:
        Arrays keyed as `generate._Rollout`'s fields, concatenated over every
        captured update. `observations`, `goal_position` and `timestep_index`
        carry one entry more than the per-step streams.

    Raises:
        RuntimeError: If PPO's debug callback yielded no experience, which
            means the buffer was never emitted and the dataset would be empty.
    """
    env = build_collection_environment(config, observation_fn)
    captured: list[tuple[int, Any]] = []

    class _CapturingPPO(PPO):  # pylint: disable=abstract-method
        """PPO whose update log appends the buffer to this call's sink."""

        def log(self, logs, inspectable=None):
            """Record one update's experience against its update index."""
            captured.append((int(logs[UPDATE_INDEX_LOG_KEY]), inspectable))

    agent = _CapturingPPO(
        hparams=PPOHparams(
            budget=budget_frames,
            num_envs=config.num_envs,
            num_steps=num_steps,
            ent_coef=entropy_coefficient,
            debug=True,
        ),
        network=ActorCritic(
            action_dim=int(env.action_space.maximum) + 1,
            actor_encoder=ConvEncoder(),
            critic_encoder=ConvEncoder(),
        ),
        env=env,
    )
    logger.info(
        "collecting PPO experience: %d frames, %d envs, %d steps per update",
        budget_frames,
        config.num_envs,
        num_steps,
    )
    train_state, _ = agent.train(rng)
    if not captured:
        raise RuntimeError(
            "PPO emitted no experience: the debug callback never fired, so "
            "the collected dataset would be empty. Check that PPOHparams.debug "
            "is set and that navix still calls Agent.log with the buffer."
        )
    # jax.debug.callback does not guarantee call order; the update index
    # restores it.
    buffers = [buffer for _, buffer in sorted(captured, key=lambda item: item[0])]
    logger.info("captured %d PPO updates", len(buffers))
    return _rollout_arrays(buffers, train_state.env_state, observation_fns)


def _rollout_arrays(
    buffers: list[Any],
    final: Timestep,
    observation_fns: dict[ObservationMode, Callable],
) -> dict[str, Any]:
    """Assemble one batched rollout from every captured update.

    Args:
        buffers: Captured buffers, in update order.
        final: The timestep training ended on, supplying the endpoint state.
        observation_fns: Views to render, one per mode.

    Returns:
        Arrays keyed as `generate._Rollout`'s fields.
    """
    states = [buffer.state for buffer in buffers]
    observations = {
        mode: np.concatenate(
            [_render(fn, state) for state in states]
            + [np.asarray(jax.vmap(fn)(final.state))[None]],
        )
        for mode, fn in observation_fns.items()
    }
    step_types = np.concatenate(
        [np.asarray(buffer.info[STEP_TYPE_INFO_KEY]) for buffer in buffers]
    )
    return {
        "observations": observations,
        "actions": np.concatenate(
            [np.asarray(buffer.action) for buffer in buffers]
        ),
        "rewards": np.concatenate(
            [np.asarray(buffer.reward) for buffer in buffers]
        ),
        "terminated": step_types == int(navix.StepType.TERMINATION),
        "truncated": step_types == int(navix.StepType.TRUNCATION),
        "goal_position": np.concatenate(
            [_goal_positions(state) for state in states]
            + [np.asarray(_goal_position_of(final.state))[None]],
        ),
        "timestep_index": np.concatenate(
            [np.asarray(buffer.t) for buffer in buffers]
            + [np.asarray(final.t)[None]]
        ),
        "executed_actions": np.concatenate(
            [
                np.asarray(buffer.info[EXECUTED_ACTION_INFO_KEY])
                for buffer in buffers
            ]
        ),
    }


def _render(fn: Callable, state: Any) -> np.ndarray:
    """Render one update's states, which carry a step and an environment axis."""
    return np.asarray(jax.vmap(jax.vmap(fn))(state))


def _goal_position_of(state: Any) -> jax.Array:
    """Return the goal entity's position for a batch of states."""
    return state.entities[NAVIX_GOAL_ENTITY].position


def _goal_positions(state: Any) -> np.ndarray:
    """Return one update's goal positions, over its step and environment axes."""
    return np.asarray(_goal_position_of(state))
