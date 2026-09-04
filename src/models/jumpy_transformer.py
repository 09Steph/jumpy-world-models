"""Arm 1: direct variable-horizon state prediction with a transformer encoder.

Given a state and a sequence of ``h`` actions, predict the state after
executing all of them in one forward pass, without generating any intermediate
state.

The sequence is joint, the state tokens followed by the action tokens, with one
transformer over both. Attention is full and there is no causal mask.

The horizon is the mask. Action sequences are padded to a fixed length and the
count of unmasked action tokens is ``h``.

The readout is the state tokens' output positions, sliced by the count the
tokeniser returned.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from config import (
    ATTENTION_WINDOW,
    ModelConfig,
)
from src.data.tokeniser import (
    action_padding_mask,
    tokeniser_for_spec,
)
from src.data.trajectory import ObservationSpec
from src.models.layers import get_activation
from src.models.positional import RotaryPositionalEmbedding
from src.data.window_sampler import WindowBatch

# Flax's RNG stream name for dropout.
DROPOUT_RNG_NAME: str = "dropout"


def attention_mask(
    valid_tokens: jax.Array, window: int | None = None
) -> jax.Array:
    """Build the pairwise attention mask from which tokens are real.

    Symmetric by construction, not lower triangular.

    Padded action slots are masked as keys, so no real token attends to them.
    They are left unmasked as queries. Masking a query row entirely would leave
    a softmax over nothing.

    Args:
        valid_tokens: True where a sequence position holds a real token, shape
            (batch, seq).
        window: Local attention half-width. None means full bidirectional
            attention, which is the default and the built behaviour. An integer
            restricts attention to pairs at most that far apart.

    Returns:
        Boolean mask broadcastable to (batch, 1, seq, seq), True where a
        query may attend to a key. Full attention returns (batch, 1, 1, seq)
        and relies on broadcasting over queries; a window materialises the full
        (batch, 1, seq, seq). The head axis is singleton in both.
    """
    mask = valid_tokens[:, None, None, :]
    if window is None:
        return mask
    positions = jnp.arange(valid_tokens.shape[1])
    within = jnp.abs(positions[:, None] - positions[None, :]) <= window
    return mask & within[None, None, :, :]


class TransformerBlock(nn.Module):
    """One pre-norm transformer encoder block with rotary attention.

    Attributes:
        d_model: Token width.
        num_heads: Attention heads. d_model must divide by it.
        ffn_multiplier: Feed-forward hidden width as a multiple of d_model.
        dropout_rate: Dropout applied to each sublayer's output.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
    """

    d_model: int
    num_heads: int
    ffn_multiplier: int = 4
    dropout_rate: float = 0.1
    activation: str = "silu"
    norm_eps: float = 1e-4

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        mask: jax.Array,
        *,
        deterministic: bool,
    ) -> jax.Array:
        """Run attention and the feed-forward sublayer over one sequence.

        Args:
            tokens: Sequence of shape (batch, seq, d_model).
            mask: Pairwise attention mask, shape (batch, 1, seq, seq).
            deterministic: True to disable dropout.

        Returns:
            The sequence, shape unchanged.

        Raises:
            ValueError: If d_model does not divide by num_heads.
        """
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model must divide by num_heads: {self.d_model} % "
                f"{self.num_heads} != 0"
            )
        head_dim = self.d_model // self.num_heads

        attended = nn.RMSNorm(epsilon=self.norm_eps, name="attention_norm")(
            tokens
        )
        queries, keys, values = (
            nn.DenseGeneral(
                features=(self.num_heads, head_dim), axis=-1, name=name
            )(attended)
            for name in ("query", "key", "value")
        )
        # Rotary encoding applies to queries and keys only, never to values.
        queries, keys = RotaryPositionalEmbedding(
            head_dim=head_dim, name="rotary"
        )(queries, keys)

        scores = jnp.einsum("bqhd,bkhd->bhqk", queries, keys) / jnp.sqrt(
            jnp.asarray(head_dim, dtype=queries.dtype)
        )
        # The dtype minimum, not -inf, so a fully masked row would give
        # a uniform row instead of NaN.
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1)
        pooled = jnp.einsum("bhqk,bkhd->bqhd", weights, values)
        projected = nn.DenseGeneral(
            features=self.d_model, axis=(-2, -1), name="attention_out"
        )(pooled)
        tokens = tokens + nn.Dropout(
            rate=self.dropout_rate, deterministic=deterministic
        )(projected)

        hidden = nn.RMSNorm(epsilon=self.norm_eps, name="ffn_norm")(tokens)
        hidden = nn.Dense(self.d_model * self.ffn_multiplier, name="ffn_in")(
            hidden
        )
        hidden = get_activation(self.activation)(hidden)
        hidden = nn.Dense(self.d_model, name="ffn_out")(hidden)
        return tokens + nn.Dropout(
            rate=self.dropout_rate, deterministic=deterministic
        )(hidden)


class JumpyTransformer(nn.Module):
    """Predict the state after `h` actions, directly and in one pass.

    Attributes:
        tokeniser: The seam. Supplies state tokens and action tokens.
        decoder: The head built by that tokeniser, matching its readout shape.
        num_layers: Transformer encoder depth.
        d_model: Token width.
        num_heads: Attention heads per layer.
        ffn_multiplier: Feed-forward width as a multiple of d_model.
        dropout_rate: Dropout inside each block.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
        attention_window: Local attention half-width, or None for full
            bidirectional attention. None is the built default.
    """

    tokeniser: nn.Module
    decoder: nn.Module
    num_layers: int
    d_model: int
    num_heads: int
    ffn_multiplier: int = 4
    dropout_rate: float = 0.1
    activation: str = "silu"
    norm_eps: float = 1e-4
    attention_window: int | None = None

    @nn.compact
    def __call__(
        self,
        observation: jax.Array,
        actions: jax.Array,
        horizons: jax.Array,
        *,
        deterministic: bool,
    ) -> list[jax.Array]:
        """Predict the observation reached after each example's action sequence.

        Args:
            observation: Discrete observation codes at time t, shape
                (batch, height, width, num_channels).
            actions: Discrete action indices, padded to the model's maximum
                horizon, shape (batch, num_action_tokens).
            horizons: True horizon per example, shape (batch,). Values must
                lie in [1, num_action_tokens], enforced eagerly by
                ``tokeniser.check_horizons_valid`` where a batch is built.
            deterministic: True to disable dropout.

        Returns:
            One logits array per observation channel, each of shape
            (batch, height, width, classes_for_that_channel).
        """
        observation_tokens = self.tokeniser.encode_state(observation)
        action_tokens = self.tokeniser.encode_actions(actions)
        num_observation_tokens = observation_tokens.shape[1]

        tokens = jnp.concatenate([observation_tokens, action_tokens], axis=1)
        # Only the padded tail of the action sequence is not real.
        valid = jnp.concatenate(
            [
                jnp.ones(
                    (tokens.shape[0], num_observation_tokens), dtype=bool
                ),
                action_padding_mask(horizons, action_tokens.shape[1]),
            ],
            axis=1,
        )
        mask = attention_mask(valid, self.attention_window)

        for layer in range(self.num_layers):
            tokens = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ffn_multiplier=self.ffn_multiplier,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                norm_eps=self.norm_eps,
                name=f"block_{layer}",
            )(tokens, mask, deterministic=deterministic)
        # The last block's output comes straight off a residual path and has
        # never been normalised.
        tokens = nn.RMSNorm(epsilon=self.norm_eps, name="output_norm")(tokens)

        # The readout, sliced by count and not by kind.
        return self.decoder(tokens[:, :num_observation_tokens, :])


def predict_endpoint(model, params, batch: WindowBatch) -> list:
    """Apply an arm to one window batch and return per-channel logits.

    One call site for the model's argument order. ``states`` and ``targets``
    have identical shapes and dtypes, so a transposed pair is invisible to
    every shape assertion and simply inverts the prediction task.

    Deterministic, with no flag to pass. Every caller here is reporting rather
    than training. The training loop calls ``model.apply`` directly.

    Args:
        model: The unbound transformer.
        params: Its parameter tree.
        batch: The windows to predict.

    Returns:
        One logits array per observation channel, each
        (batch, height, width, classes).
    """
    return model.apply(
        {"params": params},
        batch.states,
        batch.actions,
        batch.horizons,
        deterministic=True,
    )


def jumpy_transformer_for_spec(
    spec: ObservationSpec, config: ModelConfig, *, depth_extent: int
) -> JumpyTransformer:
    """Build arm 1 for an environment and a model configuration.

    The single construction site for the direct model.

    ``depth_extent`` reaches the encoder and nothing else. The spec decides what
    is predicted, so the egocentric mode predicts a smaller grid while both
    modes share one encoder depth.

    Args:
        spec: The observation being predicted. Sizes the decoder's output grid
            and supplies the vocabularies.
        config: Model architecture settings.
        depth_extent: The environment's largest grid extent, driving encoder
            depth only. Callers pass EnvConfig.max_grid_extent, which
            resolve_observation_contract sets from the environment's row.

    Returns:
        The direct transformer, unbound.
    """
    tokeniser = tokeniser_for_spec(spec, config, depth_extent=depth_extent)
    return JumpyTransformer(
        tokeniser=tokeniser,
        decoder=tokeniser.build_decoder(),
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        ffn_multiplier=config.ffn_multiplier,
        dropout_rate=config.dropout_rate,
        activation=config.activation,
        norm_eps=config.norm_eps,
        attention_window=ATTENTION_WINDOW,
    )
