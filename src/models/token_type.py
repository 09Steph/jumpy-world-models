"""Token-type embedding, built but not wired in.

The transformer reads one sequence holding two kinds of token, the state
tokens then the action tokens. This adds a learned vector to each saying which
kind it is. The state side is one token under one-token tokenisation and
height * width under per-cell.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


STATE_TOKEN_TYPE: int = 0
ACTION_TOKEN_TYPE: int = 1
NUM_TOKEN_TYPES: int = 2


class TokenTypeEmbedding(nn.Module):
    """Add a learned per-kind vector to each token.

    ``config.ModelConfig.use_token_type_embedding`` declares the switch and no
    module under ``src/`` reads it, so setting the flag alone changes nothing.
    Enabling this needs the model builder wired to it.

    Attributes:
        d_model: Token width. Must match the sequence the embedding is added to.
        num_types: How many kinds of token the sequence contains.
    """

    d_model: int
    num_types: int = NUM_TOKEN_TYPES

    @nn.compact
    def __call__(self, tokens: jax.Array, type_ids: jax.Array) -> jax.Array:
        """Add the type vector for each token.

        Args:
            tokens: Sequence of shape (batch, seq, d_model).
            type_ids: Integer kind per position, shape (seq,) or (batch, seq).
                Use STATE_TOKEN_TYPE and ACTION_TOKEN_TYPE, not literals.

        Returns:
            The sequence with type information added, shape unchanged.
        """
        table = nn.Embed(
            num_embeddings=self.num_types,
            features=self.d_model,
            name="token_type",
        )
        return tokens + table(type_ids.astype(jnp.int32))


def default_type_ids(num_state_tokens: int, num_action_tokens: int) -> jax.Array:
    """Return the type id per position for the default sequence layout.

    Args:
        num_state_tokens: How many state tokens the tokenisation produces, one
            under one-token and height * width under per-cell.
        num_action_tokens: Padded action length of the batch in hand.
            HORIZON_MAX for a training batch, and the evaluation grid's
            ceiling for an evaluation batch.

    Returns:
        Integer type ids, shape (num_state_tokens + num_action_tokens,).
    """
    return jnp.concatenate(
        [
            jnp.full((num_state_tokens,), STATE_TOKEN_TYPE, dtype=jnp.int32),
            jnp.full((num_action_tokens,), ACTION_TOKEN_TYPE, dtype=jnp.int32),
        ]
    )
