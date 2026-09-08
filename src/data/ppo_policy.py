"""Train a navix PPO policy and expose it as a sampling interface.

Training is navix's. What lives here is the mapping onto ``PPOHparams`` and the
network sized to one environment's action space.

The policy is not stored, so a run is reproduced by re-running it with the same
key, on the same device and the same compilation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from navix.agents import PPO, ActorCritic, ConvEncoder, PPOHparams
from navix.environments import Environment

from config import (
    PPO_BUDGET_FRAMES,
    PPO_ENTROPY_COEFFICIENT,
    PPO_NUM_ENVS,
    PPO_NUM_STEPS,
)
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PpoHparams:
    """Budget and rollout shape for one PPO training run."""

    budget: int = PPO_BUDGET_FRAMES
    num_envs: int = PPO_NUM_ENVS
    num_steps: int = PPO_NUM_STEPS
    ent_coef: float = PPO_ENTROPY_COEFFICIENT


@dataclass(frozen=True)
class TrainedPolicy:
    """A trained policy's parameters and the distribution they parametrise."""

    params: Any
    policy: Callable[[Any, jax.Array], Any]

    def sample(self, observation: jax.Array, key: jax.Array) -> jax.Array:
        """Return one action per parallel environment.

        Args:
            observation: Batched observations, leading with the environment
                axis.
        """
        distribution = self.policy(self.params, observation)
        return jnp.asarray(distribution.sample(seed=key))


def train_policy(
    env: Environment, hparams: PpoHparams, rng: jax.Array
) -> TrainedPolicy:
    """Train a PPO policy on one environment and return a sampling interface.

    Args:
        rng: PRNG key seeding initialisation and collection.
    """
    num_actions = int(env.action_space.maximum) + 1
    agent = PPO(
        hparams=PPOHparams(
            budget=hparams.budget,
            num_envs=hparams.num_envs,
            num_steps=hparams.num_steps,
            ent_coef=hparams.ent_coef,
        ),
        network=ActorCritic(
            action_dim=num_actions,
            actor_encoder=ConvEncoder(),
            critic_encoder=ConvEncoder(),
        ),
        env=env,
    )
    logger.info(
        "training PPO over %d actions for %d frames", num_actions, hparams.budget
    )
    train_state, _ = agent.train(rng)
    return TrainedPolicy(params=train_state.params, policy=train_state.policy)
