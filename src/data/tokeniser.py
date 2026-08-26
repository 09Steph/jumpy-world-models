"""Tokenisation between an environment and the transformer.

An environment describes itself with an `ObservationSpec`. A `Tokeniser` turns
its observations and actions into the `d_model` vectors the transformer
consumes, and builds the matching decoder.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

import flax.linen as nn
import jax
import jax.numpy as jnp

from config import (
    STATE_TOKENS_ONE_TOKEN,
    STATE_TOKENS_PATCH,
    STATE_TOKENS_PER_CELL,
    ModelConfig,
)
from src.data.trajectory import ObservationSpec
from src.models.decoder import (
    OneTokenGridDecoder,
    PerCellGridDecoder,
)
from src.models.encoder import encoder_for_spec

# Smallest horizon a prediction may be made at. Enforced in this module.
MIN_HORIZON: int = 1


class StateTokens(Enum):
    """How an observation becomes transformer input.

    Values match config's STATE_TOKENS_* strings. ONE_TOKEN is the default.
    """

    ONE_TOKEN = STATE_TOKENS_ONE_TOKEN
    """Whole observation to one d_model vector. Sequence is 1 + h. The decoder
    expands that vector back to every cell."""

    PER_CELL = STATE_TOKENS_PER_CELL
    """One token per grid cell. Sequence is (H*W) + h, with H and W read off
    the ObservationSpec. The decoder is a shared head read off each cell's own
    output position."""

    PATCH = STATE_TOKENS_PATCH
    """Cells grouped into blocks. Not implemented, see PATCH_TRIGGERS."""


# Every condition must hold before PATCH is worth building.
PATCH_TRIGGERS: tuple[str, ...] = (
    "ONE_TOKEN does not beat the copy baseline",
    "PER_CELL clearly beats ONE_TOKEN",
    "PER_CELL makes the autoregressive arm unaffordable once timed",
)


class Tokeniser(Protocol):
    """Turns an environment's observations and actions into model tokens.

    The decoder is part of this Protocol.
    """

    def encode_state(self, observation: jax.Array) -> jax.Array:
        """Encode a batch of observations to (batch, num_tokens, d_model).

        num_tokens is 1 under ONE_TOKEN and H*W under PER_CELL. The token axis
        is always present, so ONE_TOKEN returns (batch, 1, d_model) and not
        (batch, d_model).

        Args:
            observation: Discrete observation codes, shape
                (batch, height, width, num_channels).

        Returns:
            Tokens of shape (batch, num_tokens, d_model).
        """

    def encode_actions(self, actions: jax.Array) -> jax.Array:
        """Encode an action sequence to (batch, h, d_model) tokens.

        Args:
            actions: Discrete action indices, shape (batch, h), padded to the
                configured maximum horizon.

        Returns:
            Tokens of shape (batch, h, d_model).
        """

    def build_decoder(self) -> nn.Module:
        """Return the head mapping model output back to per-field predictions.

        Returns:
            An unbound Flax module taking the transformer's state-token outputs
            and returning one logits array per observation channel.
        """


class GridTokeniser(nn.Module):
    """Map discrete grid observations and action sequences to model tokens.

    Satisfies the `Tokeniser` Protocol. Single-field grid observations only.

    The state encodes to one token by default and the actions to a learned
    embedding per step, in one joint sequence padded to HORIZON_MAX. The
    unmasked count from `action_padding_mask` is the horizon.

    Attributes:
        spec: Grid shape, per-channel cardinalities and action count. Must be
            single-field.
        d_model: Token width, shared by state and action tokens.
        code_embed_dim: Width of each observation channel's embedding table.
        encoder_channels: Convolution widths of the encoder trunk.
        depth_extent: The environment's largest grid extent, driving encoder
            depth.
        decoder_channels: Convolution widths of the decoder trunk.
        state_tokens: How an observation becomes tokens. ONE_TOKEN by default.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.

    Raises:
        NotImplementedError: If state_tokens is PATCH.
    """

    spec: ObservationSpec
    d_model: int
    code_embed_dim: int
    encoder_channels: tuple[int, ...]
    decoder_channels: tuple[int, ...]
    depth_extent: int
    state_tokens: StateTokens = StateTokens.ONE_TOKEN
    activation: str = "silu"
    norm_eps: float = 1e-4

    def __post_init__(self) -> None:
        """Reject the unimplemented tokenisation at construction.

        Raises:
            NotImplementedError: If state_tokens is PATCH.
        """
        if self.state_tokens is StateTokens.PATCH:
            raise NotImplementedError(
                "PATCH state tokenisation is defined but not implemented. "
                "Build it only when every one of these holds: "
                + "; ".join(PATCH_TRIGGERS)
                + ". Otherwise the answer is ONE_TOKEN or PER_CELL, "
                "both of which are built."
            )
        super().__post_init__()

    def setup(self) -> None:
        """Build the observation encoder and the action embedding table.

        The vocabularies come from the spec and the encoder's depth from
        `depth_extent`. Uses `setup`; a compact method may only be one, and
        this class exposes two encoding methods.
        """
        # pylint: disable=attribute-defined-outside-init
        self.encoder = encoder_for_spec(
            self.spec,
            code_embed_dim=self.code_embed_dim,
            channels=self.encoder_channels,
            d_model=self.d_model,
            pool_to_one_token=self.state_tokens is StateTokens.ONE_TOKEN,
            depth_extent=self.depth_extent,
        )
        # A lookup, not a one-hot projection. The table is
        # (num_actions, d_model) regardless of action-set size.
        self.action_embedding = nn.Embed(
            num_embeddings=self.spec.num_actions,
            features=self.d_model,
            name="action_embedding",
        )

    def encode_state(self, observation: jax.Array) -> jax.Array:
        """Encode a batch of observations to tokens.

        Args:
            observation: Discrete observation codes, shape
                (batch, height, width, num_channels).

        Returns:
            Tokens of shape (batch, 1, d_model) under ONE_TOKEN and
            (batch, height * width, d_model) under PER_CELL. The token axis is
            present in both cases.
        """
        tokens = self.encoder(observation)
        if self.state_tokens is StateTokens.ONE_TOKEN:
            # The encoder pools to (batch, d_model). The token axis is
            # restored here so the encoder keeps one shape per pooling mode.
            return tokens[:, None, :]
        return tokens

    def encode_actions(self, actions: jax.Array) -> jax.Array:
        """Encode a padded action sequence to tokens.

        Padding indices are embedded like any other and excluded from attention
        by the mask.

        Args:
            actions: Discrete action indices, shape (batch, num_action_tokens).

        Returns:
            Tokens of shape (batch, num_action_tokens, d_model).
        """
        return self.action_embedding(actions.astype(jnp.int32))

    @nn.nowrap
    def build_decoder(self) -> nn.Module:
        """Return the decoder matching this tokenisation.

        Both decoders take different readout shapes and return the same
        per-channel logits shape. `nn.nowrap`: this constructs a module and
        does not use one.

        Returns:
            An unbound decoder module taking the transformer's state-token
            outputs, shape (batch, num_tokens, d_model).
        """
        field = self.spec.single_field()
        height, width = field.shape
        if self.state_tokens is StateTokens.ONE_TOKEN:
            return OneTokenGridDecoder(
                obs_grid_shape=(height, width),
                obs_channel_classes=field.cardinality,
                channels=self.decoder_channels,
                activation=self.activation,
                norm_eps=self.norm_eps,
            )
        return PerCellGridDecoder(
            obs_grid_shape=(height, width),
            obs_channel_classes=field.cardinality,
            channels=self.decoder_channels,
            activation=self.activation,
            norm_eps=self.norm_eps,
        )


def tokeniser_for_spec(
    spec: ObservationSpec, config: ModelConfig, *, depth_extent: int
) -> GridTokeniser:
    """Build the grid tokeniser an environment and a model config imply.

    Resolves the configured tokenisation name to the enum, so an unknown string
    fails here with the valid options listed.

    Args:
        spec: The observation being predicted. Supplies the vocabularies and
            the decoder's output grid.
        config: Model architecture settings supplying the widths.
        depth_extent: The environment's largest grid extent, driving encoder
            depth only. Required and keyword-only.

    Returns:
        A GridTokeniser sized to the spec at the requested depth.

    Raises:
        ValueError: If config.state_tokens is not a known tokenisation name.
        NotImplementedError: If it names PATCH.
    """
    try:
        state_tokens = StateTokens(config.state_tokens)
    except ValueError as error:
        valid = ", ".join(option.value for option in StateTokens)
        raise ValueError(
            f"unknown state tokenisation {config.state_tokens!r}. "
            f"Valid options: {valid}."
        ) from error
    return GridTokeniser(
        spec=spec,
        d_model=config.d_model,
        code_embed_dim=config.code_embed_dim,
        encoder_channels=config.encoder_channels,
        decoder_channels=config.decoder_channels,
        depth_extent=depth_extent,
        state_tokens=state_tokens,
        activation=config.activation,
        norm_eps=config.norm_eps,
    )


def action_padding_mask(
    horizons: jax.Array, num_action_tokens: int
) -> jax.Array:
    """Return which action-token slots hold a real action.

    The count of unmasked action tokens is the horizon. Pure and jit-safe, so
    the h >= 1 rule is enforced by `check_horizons_valid` instead.

    Args:
        horizons: True horizon per example, shape (batch,).
        num_action_tokens: Padded sequence length, config's HORIZON_MAX.

    Returns:
        Boolean mask of shape (batch, num_action_tokens), True where the slot
        holds a real action.
    """
    slots = jnp.arange(num_action_tokens)
    return slots[None, :] < horizons[:, None]


def check_horizons_valid(horizons: jax.Array, num_action_tokens: int) -> None:
    """Raise if any horizon is outside [MIN_HORIZON, num_action_tokens].

    A host-side check on a sampled batch, not part of the forward pass. It
    costs one host sync per call. Callers run it when a batch is built.

    Args:
        horizons: True horizon per example, shape (batch,).
        num_action_tokens: Padded sequence length, config's HORIZON_MAX.

    Raises:
        ValueError: If any horizon is below MIN_HORIZON or above the padded
            length.
    """
    lowest = int(jnp.min(horizons))
    highest = int(jnp.max(horizons))
    if lowest < MIN_HORIZON or highest > num_action_tokens:
        raise ValueError(
            f"horizons hold values in [{lowest}, {highest}], outside the "
            f"valid [{MIN_HORIZON}, {num_action_tokens}] range. h = 0 is not a "
            "prediction, and h past the padded length would unmask slots "
            "holding no action."
        )
