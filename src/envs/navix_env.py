"""NAVIX environment adapter.

NAVIX is a JAX-native reimplementation of MiniGrid. The environment runs inside
the same JIT-compiled graph as the model, with no Gym interface and no CPU-GPU
boundary.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import navix

from config import EnvConfig
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def never_terminate(
    prev_state: navix.states.State,
    action: jax.Array,
    state: navix.states.State,
) -> jax.Array:
    """Return False everywhere, so only truncation ends an episode.

    Written here rather than taken from navix.terminations, whose
    check_truncation takes (terminated, truncated) and cannot be used as a
    termination function. The required signature is the one on_goal_reached
    carries.

    Removing a termination does not change the transition function, so the
    agent still cannot occupy an obstacle's cell. Pinned by
    tests/test_navix_env.py.

    Args:
        prev_state: Environment state before the action.
        action: The action taken.
        state: Environment state after the action.

    Returns:
        Scalar boolean array, always False, matching what navix's own
        on_goal_reached returns.
    """
    del prev_state, action, state
    return jnp.asarray(False, dtype=jnp.bool_)


class NavixEnv:
    """Adapter around a batched NAVIX environment.

    NAVIX runs entirely as JAX arrays inside one JIT graph. `reset()` and
    `step()` are pure functions over `navix.environments.Timestep`, which is
    the environment state, so `step()` takes the previous one directly.
    Batching over config.num_envs uses `jax.vmap`.

    NAVIX auto-resets once max_steps is reached, and its own default would
    apply otherwise, so max_steps is always passed explicitly.

    Attributes:
        config: Environment configuration (name, num_envs, max_episode_steps).
        env: The underlying navix.Environment, built from config.name with
            max_steps=config.max_episode_steps.
    """

    def __init__(
        self, config: EnvConfig, observation_fn: Callable | None = None
    ) -> None:
        """Build the underlying NAVIX environment from config.

        Args:
            config: Environment configuration.
            observation_fn: NAVIX observation function selecting the view.
                None keeps the library default, which is not `symbolic`.

                Trajectory generation does not use this. Both observation modes
                are derived from one rollout's state.

                Not `navix.observations.categorical`, which returns a 2D int32
                grid matching no part of the per-cell contract.
        """
        self.config = config
        # penality_coeff passed explicitly, not inherited from navix's
        # default. observation_fn is passed only when given; navix.make's own
        # default is a real function that None would overwrite.
        observation_kwargs = (
            {} if observation_fn is None else {"observation_fn": observation_fn}
        )
        # Same reason as observation_fn: navix.make's own termination_fn is a
        # real function, so the key is passed only when it is being replaced.
        termination_kwargs = (
            {"termination_fn": never_terminate}
            if config.disable_early_termination
            else {}
        )
        self.env = navix.make(
            config.name,
            max_steps=config.max_episode_steps,
            penality_coeff=config.penality_coeff,
            **observation_kwargs,
            **termination_kwargs,
        )
        self._reset_fn = jax.jit(jax.vmap(self.env.reset))
        self._step_fn = jax.jit(jax.vmap(self.env.step))
        logger.info(
            "NavixEnv initialised: name=%s num_envs=%d max_steps=%d "
            "early_termination=%s",
            config.name,
            config.num_envs,
            config.max_episode_steps,
            not config.disable_early_termination,
        )

    def reset(self, rng: jax.Array) -> navix.environments.Timestep:
        """Reset config.num_envs parallel environments.

        Args:
            rng: PRNG key, split internally into one key per environment so no
                key is reused across the batch.

        Returns:
            Batched Timestep, each leaf of shape (num_envs, ...).
        """
        keys = jax.random.split(rng, self.config.num_envs)
        return self._reset_fn(keys)

    def step(
        self, timestep: navix.environments.Timestep, action: jax.Array
    ) -> navix.environments.Timestep:
        """Advance config.num_envs parallel environments by one step each.

        Args:
            timestep: Batched Timestep from reset() or a prior step. This is
                the environment state.
            action: Discrete actions, shape (num_envs,), each a valid index
                for the environment's action space.

        Returns:
            Next batched Timestep. NAVIX auto-resets once
            config.max_episode_steps is reached, so no explicit reset is needed
            at episode boundaries.
        """
        return self._step_fn(timestep, action)

    @staticmethod
    def done(timestep: navix.environments.Timestep) -> jax.Array:
        """Derive an episode-end boolean from step_type.

        True whenever an episode has ended, for any reason. NAVIX's step_type
        is 3-way. TRANSITION (ongoing), TRUNCATION (hit max_steps),
        TERMINATION (an absorbing state). Use terminated() or truncated() when
        the reason matters.

        Args:
            timestep: A Timestep from reset() or step().

        Returns:
            Boolean array, shape matching timestep.step_type.
        """
        return timestep.step_type != navix.StepType.TRANSITION

    @staticmethod
    def success(timestep: navix.environments.Timestep) -> jax.Array:
        """True where an episode ended in task success.

        Lives on the adapter. What counts as success is environment semantics.
        For NAVIX FourRooms it is exactly `terminated()`, the goal being the
        only absorbing state. A separate method, as the two need not coincide
        elsewhere.

        Not `done()`, which is also true for truncation.

        Args:
            timestep: A Timestep from reset() or step().

        Returns:
            Boolean array, shape matching timestep.step_type.
        """
        return NavixEnv.terminated(timestep)

    @staticmethod
    def terminated(timestep: navix.environments.Timestep) -> jax.Array:
        """True at a real absorbing state (e.g. the goal was reached).

        Args:
            timestep: A Timestep from reset() or step().

        Returns:
            Boolean array, shape matching timestep.step_type.
        """
        return timestep.step_type == navix.StepType.TERMINATION

    @staticmethod
    def truncated(timestep: navix.environments.Timestep) -> jax.Array:
        """True when the episode ended on max_steps alone.

        The state was not absorbing. Distinct from terminated(), and a consumer
        computing a bootstrapped return must treat the two differently.

        Args:
            timestep: A Timestep from reset() or step().

        Returns:
            Boolean array, shape matching timestep.step_type.
        """
        return timestep.step_type == navix.StepType.TRUNCATION
