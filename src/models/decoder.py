"""Decoders from a model embedding back to an observation grid.

``GridDecoder`` and ``ConvGridDecoder`` emit per-channel categorical logits for
every cell. ``ConvPixelDecoder`` emits a continuous grid in [0, 1].
``OneTokenGridDecoder``, ``PerCellGridDecoder`` and ``OneTokenPixelDecoder``
adapt the transformer's readout to either tokenisation.
"""

from __future__ import annotations

import flax.linen as nn
import jax

import jax.numpy as jnp

from src.models.layers import (
    DECODER_OUTSCALE,
    conv_trunk,
    mlp_trunk,
    output_kernel_init,
)

class GridDecoder(nn.Module):
    """Reconstruction head emitting per-cell categorical logits.

    One categorical distribution per grid cell per channel, over discrete
    codes.

    Must not revert to a regression loss. Mean squared error treats the codes
    as ordinal.

    Attributes:
        obs_grid_shape: Spatial grid shape, (height, width).
        obs_channel_classes: Number of classes per observation channel.
        hidden_size: Hidden width of the MLP trunk.
        dec_layers: Depth of the MLP trunk.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
    """

    obs_grid_shape: tuple[int, int]
    obs_channel_classes: tuple[int, ...]
    hidden_size: int
    dec_layers: int
    activation: str = "silu"
    norm_eps: float = 1e-4

    @nn.compact
    def __call__(self, embedding: jax.Array) -> list[jax.Array]:
        """Return per-channel categorical logits over the observation grid.

        Args:
            embedding: Model output for one prediction, shape
                (..., embedding_dim).

        Returns:
            One logits array per channel, each of shape
            (..., height, width, classes_for_that_channel). A list, not a
            stacked array; the channels have different class counts.
        """
        hidden = mlp_trunk(
            embedding,
            self.hidden_size,
            self.dec_layers,
            self.activation,
            self.norm_eps,
        )
        height, width = self.obs_grid_shape
        logits = []
        for channel, num_classes in enumerate(self.obs_channel_classes):
            # Named explicitly: these names become checkpoint keys.
            flat = nn.Dense(
                height * width * num_classes,
                kernel_init=output_kernel_init(DECODER_OUTSCALE),
                name=f"logits_channel_{channel}",
            )(hidden)
            logits.append(
                flat.reshape(*flat.shape[:-1], height, width, num_classes)
            )
        return logits


class ConvGridDecoder(nn.Module):
    """Expand one embedding back to per-cell categorical logits, convolutionally.

    The matched partner to GridEncoder. It broadcasts the embedding across the
    grid, adds a learned position, and convolves.

    Every convolution is grid-size independent and the positional parameters
    are factorised into a row vector and a column vector, so one decoder serves
    any grid shape. Without the learned positions every cell would decode
    identically.

    Attributes:
        obs_grid_shape: Spatial grid shape, (height, width).
        obs_channel_classes: Number of classes per observation channel.
        channels: Convolution widths, one per trunk layer.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
    """

    obs_grid_shape: tuple[int, int]
    obs_channel_classes: tuple[int, ...]
    channels: tuple[int, ...]
    activation: str = "silu"
    norm_eps: float = 1e-4

    @nn.compact
    def __call__(self, embedding: jax.Array) -> list[jax.Array]:
        """Return per-channel categorical logits over the observation grid.

        Args:
            embedding: Model output for one prediction, shape
                (batch, embedding_dim).

        Returns:
            One logits array per channel, each of shape
            (batch, height, width, classes_for_that_channel).
        """
        height, width = self.obs_grid_shape

        features = jnp.broadcast_to(
            embedding[:, None, None, :],
            (embedding.shape[0], height, width, embedding.shape[-1]),
        )
        rows = self.param(
            "row_position",
            nn.initializers.normal(stddev=0.02),
            (height, embedding.shape[-1]),
        )
        columns = self.param(
            "column_position",
            nn.initializers.normal(stddev=0.02),
            (width, embedding.shape[-1]),
        )
        features = features + rows[None, :, None, :] + columns[None, None, :, :]

        features = conv_trunk(
            features, self.channels, self.activation, self.norm_eps
        )

        return [
            nn.Conv(
                features=num_classes,
                kernel_size=(1, 1),
                kernel_init=output_kernel_init(DECODER_OUTSCALE),
                name=f"logits_channel_{channel}",
            )(features)
            for channel, num_classes in enumerate(self.obs_channel_classes)
        ]


class ConvPixelDecoder(nn.Module):
    """Expand one embedding back to a continuous grid, convolutionally.

    The matched partner to PixelEncoder, sharing ConvGridDecoder's broadcast
    and factorised positions. The output head is one 1x1 convolution over all
    channels at once, ending in a sigmoid so the prediction lands in [0, 1].

    Attributes:
        obs_grid_shape: Spatial grid shape, (height, width).
        num_channels: Channels in the predicted observation.
        channels: Convolution widths, one per trunk layer.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
    """

    obs_grid_shape: tuple[int, int]
    num_channels: int
    channels: tuple[int, ...]
    activation: str = "silu"
    norm_eps: float = 1e-4

    @nn.compact
    def __call__(self, embedding: jax.Array) -> list[jax.Array]:
        """Return the predicted grid, normalised to [0, 1].

        One array in a list, matching the categorical decoders' return. The
        last axis is channels here and classes there, and a consumer reads
        which from the field spec rather than from the shape.

        Args:
            embedding: Model output for one prediction, shape
                (batch, embedding_dim).

        Returns:
            One array of shape (batch, height, width, num_channels).
        """
        height, width = self.obs_grid_shape

        features = jnp.broadcast_to(
            embedding[:, None, None, :],
            (embedding.shape[0], height, width, embedding.shape[-1]),
        )
        rows = self.param(
            "row_position",
            nn.initializers.normal(stddev=0.02),
            (height, embedding.shape[-1]),
        )
        columns = self.param(
            "column_position",
            nn.initializers.normal(stddev=0.02),
            (width, embedding.shape[-1]),
        )
        features = features + rows[None, :, None, :] + columns[None, None, :, :]

        features = conv_trunk(
            features, self.channels, self.activation, self.norm_eps
        )
        prediction = nn.Conv(
            features=self.num_channels,
            kernel_size=(1, 1),
            kernel_init=output_kernel_init(DECODER_OUTSCALE),
            name="pixels",
        )(features)
        return [nn.sigmoid(prediction)]


class OneTokenPixelDecoder(nn.Module):
    """Read a continuous prediction off a single state token.

    The continuous counterpart to OneTokenGridDecoder.

    Attributes:
        obs_grid_shape: Spatial grid shape, (height, width).
        num_channels: Channels in the predicted observation.
        channels: Convolution widths of the expansion trunk.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
    """

    obs_grid_shape: tuple[int, int]
    num_channels: int
    channels: tuple[int, ...]
    activation: str = "silu"
    norm_eps: float = 1e-4

    @nn.compact
    def __call__(self, state_tokens: jax.Array) -> list[jax.Array]:
        """Return the predicted grid, normalised to [0, 1].

        Delegates to ConvPixelDecoder. Do not wrap the result again; that would
        nest the list.

        Args:
            state_tokens: Transformer output at the state token's position,
                shape (batch, 1, d_model).

        Returns:
            One array of shape (batch, height, width, num_channels).

        Raises:
            ValueError: If more than one state token is supplied, meaning the
                model was built with per-cell tokenisation and this decoder.
        """
        if state_tokens.shape[1] != 1:
            raise ValueError(
                f"OneTokenPixelDecoder expects exactly one state token, got "
                f"{state_tokens.shape[1]}. Per-cell tokenisation pairs with "
                "PerCellGridDecoder."
            )
        return ConvPixelDecoder(
            obs_grid_shape=self.obs_grid_shape,
            num_channels=self.num_channels,
            channels=self.channels,
            activation=self.activation,
            norm_eps=self.norm_eps,
            name="grid",
        )(state_tokens[:, 0, :])


class OneTokenGridDecoder(nn.Module):
    """Read the prediction off a single state token and expand it to the grid.

    The readout under one-token tokenisation. Readout decoders take
    (batch, num_tokens, d_model).

    Attributes:
        obs_grid_shape: Spatial grid shape, (height, width).
        obs_channel_classes: Number of classes per observation channel.
        channels: Convolution widths of the expansion trunk.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
    """

    obs_grid_shape: tuple[int, int]
    obs_channel_classes: tuple[int, ...]
    channels: tuple[int, ...]
    activation: str = "silu"
    norm_eps: float = 1e-4

    @nn.compact
    def __call__(self, state_tokens: jax.Array) -> list[jax.Array]:
        """Return per-channel categorical logits over the observation grid.

        Args:
            state_tokens: Transformer output at the state token's position,
                shape (batch, 1, d_model).

        Returns:
            One logits array per channel, each of shape
            (batch, height, width, classes_for_that_channel).

        Raises:
            ValueError: If more than one state token is supplied, meaning the
                model was built with per-cell tokenisation and this decoder.
        """
        if state_tokens.shape[1] != 1:
            raise ValueError(
                f"OneTokenGridDecoder expects exactly one state token, got "
                f"{state_tokens.shape[1]}. Per-cell tokenisation pairs with "
                "PerCellGridDecoder."
            )
        return ConvGridDecoder(
            obs_grid_shape=self.obs_grid_shape,
            obs_channel_classes=self.obs_channel_classes,
            channels=self.channels,
            activation=self.activation,
            norm_eps=self.norm_eps,
            name="grid",
        )(state_tokens[:, 0, :])


class PerCellGridDecoder(nn.Module):
    """Read the prediction off one token per cell, with a shared head.

    The readout under per-cell tokenisation. The token sequence is folded back
    into a grid and a shared 1x1 head scores every cell. No learned positional
    parameters. Position is carried by which token a cell's vector came from.

    Attributes:
        obs_grid_shape: Spatial grid shape, (height, width).
        obs_channel_classes: Number of classes per observation channel.
        channels: Convolution widths of the trunk.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
    """

    obs_grid_shape: tuple[int, int]
    obs_channel_classes: tuple[int, ...]
    channels: tuple[int, ...]
    activation: str = "silu"
    norm_eps: float = 1e-4

    @nn.compact
    def __call__(self, state_tokens: jax.Array) -> list[jax.Array]:
        """Return per-channel categorical logits over the observation grid.

        Args:
            state_tokens: Transformer output at the state tokens' positions,
                shape (batch, height * width, d_model).

        Returns:
            One logits array per channel, each of shape
            (batch, height, width, classes_for_that_channel).

        Raises:
            ValueError: If the token count is not height * width.
        """
        height, width = self.obs_grid_shape
        if state_tokens.shape[1] != height * width:
            raise ValueError(
                f"PerCellGridDecoder expects {height * width} state tokens for "
                f"a {height}x{width} grid, got {state_tokens.shape[1]}. One "
                "token pairs with OneTokenGridDecoder."
            )
        features = state_tokens.reshape(
            state_tokens.shape[0], height, width, state_tokens.shape[-1]
        )
        features = conv_trunk(
            features, self.channels, self.activation, self.norm_eps
        )
        return [
            nn.Conv(
                features=num_classes,
                kernel_size=(1, 1),
                kernel_init=output_kernel_init(DECODER_OUTSCALE),
                name=f"logits_channel_{channel}",
            )(features)
            for channel, num_classes in enumerate(self.obs_channel_classes)
        ]
