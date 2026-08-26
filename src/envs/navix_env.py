"""NAVIX environment adapter.

NAVIX is a JAX-native reimplementation of MiniGrid. The environment runs inside
the same JIT-compiled graph as the model, with no Gym interface and no CPU-GPU
boundary.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import navix

from config import EnvConfig
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class NavixEnv:
    """Adapter around a batched NAVIX environment.

    NAVIX runs entirely as JAX arrays inside one JIT graph. `reset()` and
    `step()` are pure functions over `navix.environments.Timestep`, which is
    the environment state, so `step()` takes the previous one directly.
    Batching over config.num_envs uses `jax.vmap`.

    NAVIX auto-resets once max_steps is reached. Its own default is 100, so
    max_steps is always passed explicitly.

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
                None keeps the library default, which is not `symbolic`, so
                existing callers observe what they did before.

                Trajectory generation does not use this. Both observation modes
                are derived from one rollout's state.

                Not `navix.observations.categorical`, which returns a 2D int32
                grid matching no part of the per-cell contract.
        """
        self.config = config
        # penality_coeff passed explicitly, not inherited from navix's 0.0
        # default. observation_fn is passed only when given; navix.make's own
        # default is a real function that None would overwrite.
        observation_kwargs = (
            {} if observation_fn is None else {"observation_fn": observation_fn}
        )
        self.env = navix.make(
            config.name,
            max_steps=config.max_episode_steps,
            penality_coeff=config.penality_coeff,
            **observation_kwargs,
        )
        self._reset_fn = jax.jit(jax.vmap(self.env.reset))
        self._step_fn = jax.jit(jax.vmap(self.env.step))
        logger.info(
            "NavixEnv initialised: name=%s num_envs=%d max_steps=%d",
            config.name,
            config.num_envs,
            config.max_episode_steps,
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
            action: Discrete actions, shape (num_envs,), each in [0, 7).

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
        is 3-way. TRANSITION (0, ongoing), TRUNCATION (1, hit max_steps),
        TERMINATION (2, an absorbing state). Use terminated() or truncated()
        when the reason matters.

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
