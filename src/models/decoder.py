"""Per-cell categorical decoders over a discrete observation grid.

``GridDecoder`` and ``ConvGridDecoder`` emit per-channel logits for every cell.
``OneTokenGridDecoder`` and ``PerCellGridDecoder`` adapt the transformer's
readout to either tokenisation and return the same logits.
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

    One categorical distribution per grid cell per channel. The observations
    hold discrete codes.

    Must not revert to a regression loss. Mean squared error treats the codes
    as ordinal, and under it the decoder collapsed to a near-constant, worse
    than a repeat-the-last-observation baseline at every step.

    Per-channel heads. The three channels have cardinalities 11, 6 and 4.

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
    are factorised into a row vector and a column vector, so the same decoder
    serves the 19x19 top-down mode, the 7x7 egocentric mode and a 21x79
    NetHack map. Without the learned positions every cell would decode
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

        # Broadcast the single embedding to every cell, then add a learned
        # per-cell vector so the positions are distinguishable.
        features = jnp.broadcast_to(
            embedding[:, None, None, :],
            (embedding.shape[0], height, width, embedding.shape[-1]),
        )
        # Factorised into a row vector and a column vector, costing (H + W)*d
        # against H*W*d. A distinct (row, column) pair still identifies every
        # cell uniquely.
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

        # One 1x1 convolution per channel, shared across every cell.
        return [
            nn.Conv(
                features=num_classes,
                kernel_size=(1, 1),
                kernel_init=output_kernel_init(DECODER_OUTSCALE),
                name=f"logits_channel_{channel}",
            )(features)
            for channel, num_classes in enumerate(self.obs_channel_classes)
        ]


class OneTokenGridDecoder(nn.Module):
    """Read the prediction off a single state token and expand it to the grid.

    The readout under one-token tokenisation. Both readout decoders take
    (batch, num_tokens, d_model) and return the same logits shape.

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

    Returns the same logits shape as OneTokenGridDecoder.

    Attributes:
        obs_grid_shape: Spatial grid shape, (height, width).
        obs_channel_classes: Number of classes per observation channel.
        channels: Convolution widths of the trunk. Undilated.
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
