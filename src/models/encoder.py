"""Convolutional encoders over an observation grid.

A discrete grid has its codes looked up in a learned embedding table per
channel. A continuous grid is rescaled from its declared bounds to [0, 1].
Both then run the same convolution trunk.

The trunk serves both state-tokenisation modes. Only the final step differs,
pooling to one token or keeping one per cell.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.data.trajectory import ObservationSpec
from src.models.layers import conv_trunk, dilations_for_grid

class GridEncoder(nn.Module):
    """Encode a discrete observation grid to tokens.

    Attributes:
        obs_channel_classes: Number of codes per observation channel.
        code_embed_dim: Width of each channel's embedding.
        channels: Convolution widths, one per trunk layer.
        dilations: Dilation rate per trunk layer. The trunk must span the whole
            grid before pooling discards spatial position. See
            config.dilations_for_grid.
        d_model: Output token width.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
        pool_to_one_token: Pool to one token, or keep one per cell.
    """

    obs_channel_classes: tuple[int, ...]
    code_embed_dim: int
    channels: tuple[int, ...]
    d_model: int
    dilations: tuple[int, ...] | None = None
    activation: str = "silu"
    norm_eps: float = 1e-4
    pool_to_one_token: bool = True

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        """Encode one batch of observation grids.

        Args:
            observation: Integer codes, shape
                (batch, height, width, num_channels). Floats are cast, so the
                values must be integral.

        Returns:
            Tokens of shape (batch, d_model) when pool_to_one_token is True,
            otherwise (batch, height * width, d_model).
        """
        codes = observation.astype(jnp.int32)

        embedded = [
            nn.Embed(
                num_embeddings=num_classes,
                features=self.code_embed_dim,
                name=f"code_embed_channel_{channel}",
            )(codes[..., channel])
            for channel, num_classes in enumerate(self.obs_channel_classes)
        ]
        features = jnp.concatenate(embedded, axis=-1)

        features = conv_trunk(
            features,
            self.channels,
            self.activation,
            self.norm_eps,
            dilations=self.dilations,
        )
        features = nn.Dense(self.d_model, name="to_token")(features)
        if not self.pool_to_one_token:
            return features.reshape(features.shape[0], -1, self.d_model)
        return jnp.mean(features, axis=(1, 2))


class PixelEncoder(nn.Module):
    """Encode a continuous observation grid to tokens.

    Attributes:
        value_range: Inclusive (low, high) bounds of the stored values, which
            supply the normalisation divisor.
        channels: Convolution widths, one per trunk layer.
        d_model: Output token width.
        dilations: Dilation rate per trunk layer. The trunk must span the whole
            grid before pooling discards spatial position. See
            config.dilations_for_grid.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
        pool_to_one_token: Pool to one token, or keep one per cell.
    """

    value_range: tuple[int, int]
    channels: tuple[int, ...]
    d_model: int
    dilations: tuple[int, ...] | None = None
    activation: str = "silu"
    norm_eps: float = 1e-4
    pool_to_one_token: bool = True

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        """Encode one batch of observation grids.

        Args:
            observation: Stored values, shape
                (batch, height, width, num_channels).

        Returns:
            Tokens of shape (batch, d_model) when pool_to_one_token is True,
            otherwise (batch, height * width, d_model).
        """
        low, high = self.value_range
        features = (observation.astype(jnp.float32) - low) / (high - low)

        features = conv_trunk(
            features,
            self.channels,
            self.activation,
            self.norm_eps,
            dilations=self.dilations,
        )
        features = nn.Dense(self.d_model, name="to_token")(features)
        if not self.pool_to_one_token:
            return features.reshape(features.shape[0], -1, self.d_model)
        return jnp.mean(features, axis=(1, 2))


def encoder_for_spec(  # pylint: disable=too-many-arguments
    spec: ObservationSpec,
    code_embed_dim: int,
    channels: tuple[int, ...],
    d_model: int,
    pool_to_one_token: bool = True,
    *,
    depth_extent: int,
) -> GridEncoder | PixelEncoder:
    """Build an encoder for one spec at a depth the caller fixes.

    The spec drives the decoder's output grid. The extent drives the encoder's
    depth. They are separate arguments so the two can differ.

    Args:
        spec: The observation being predicted. Must be single-field. Supplies
            the per-channel vocabularies and, through the tokeniser, the
            decoder's output grid.
        code_embed_dim: Width of each channel's embedding table.
        channels: Convolution widths, one per layer derived from
            ``depth_extent``.
        d_model: Output token width.
        pool_to_one_token: Pool to one token, or keep one per cell.
        depth_extent: The environment's largest grid extent, in cells. Held
            constant across observation modes, so encoders built for different
            modes match on capacity.

    Returns:
        A GridEncoder for a discrete field or a PixelEncoder for a continuous
        one.

    Raises:
        ValueError: If the spec is multi-field, or if ``channels`` does not
            have one entry per derived layer.
    """
    field = spec.single_field()
    dilations = dilations_for_grid(depth_extent)
    if len(channels) != len(dilations):
        raise ValueError(
            f"depth_extent {depth_extent} derives {len(dilations)} dilated "
            f"layers {dilations}, but {len(channels)} channel widths were "
            "given. Depth follows from depth_extent, not from the predicted "
            f"grid {field.shape}. Supply one width per layer."
        )
    if field.value_range is not None:
        return PixelEncoder(
            value_range=field.value_range,
            channels=channels,
            dilations=dilations,
            d_model=d_model,
            pool_to_one_token=pool_to_one_token,
        )
    return GridEncoder(
        obs_channel_classes=field.cardinality,
        code_embed_dim=code_embed_dim,
        channels=channels,
        dilations=dilations,
        d_model=d_model,
        pool_to_one_token=pool_to_one_token,
    )
