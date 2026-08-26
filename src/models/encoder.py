"""Convolutional encoder over a discrete symbolic observation grid.

Codes are looked up in a learned embedding table per channel, then a small
convolution trunk runs over the feature grid. The embedding keeps cost
independent of vocabulary size and the convolution keeps parameter count
independent of grid size.

The same trunk serves both state-tokenisation modes, and only the final step
differs, pooling to one token or keeping one per cell.
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
        dilations: Dilation rate per trunk layer. A correctness setting rather
            than a tuning knob. The encoder must span the whole grid before
            pooling discards spatial position. See config.ENCODER_DILATIONS.
        d_model: Output token width.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
        pool_to_one_token: When True the feature grid is mean-pooled to a
            single (d_model,) token. When False one token per cell is returned,
            shape (height * width, d_model).
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
                (batch, height, width, num_channels). Floats are accepted and
                cast; the stored codes are small integers exactly
                representable in float32.

        Returns:
            Tokens of shape (batch, d_model) when pool_to_one_token is True,
            otherwise (batch, height * width, d_model).
        """
        codes = observation.astype(jnp.int32)

        # One embedding table per channel; cardinalities differ.
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
        # Mean over the spatial axes, so the pooled width is grid-independent.
        return jnp.mean(features, axis=(1, 2))


def encoder_for_spec(  # pylint: disable=too-many-arguments
    spec: ObservationSpec,
    code_embed_dim: int,
    channels: tuple[int, ...],
    d_model: int,
    pool_to_one_token: bool = True,
    *,
    depth_extent: int,
) -> GridEncoder:
    """Build an encoder for one spec at a depth the caller fixes.

    Depth and vocabulary are both read off the environment's description. NAVIX
    yields four dilated layers over cardinalities (11, 6, 4); NetHack's glyph
    map yields six over (5991,).

    The spec drives the decoder's output grid. The extent drives the encoder's
    depth. They are separate arguments so the two can differ.

    Args:
        spec: The observation being predicted. Must be single-field; a
            multi-field environment needs a composite encoder, see
            ``encoder_nle.py``. Supplies the per-channel vocabularies and,
            through the tokeniser, the decoder's output grid. It does not
            supply the depth.
        code_embed_dim: Width of each channel's embedding table.
        channels: Convolution widths. Its length must match the depth derived
            from ``depth_extent``, not from the spec.
        d_model: Output token width.
        pool_to_one_token: Pool to one token, or keep one per cell.
        depth_extent: The environment's largest grid extent, in cells. Drives
            encoder depth only. Held constant across observation modes, so
            encoders built for different modes match on capacity.

    Returns:
        A GridEncoder whose vocabulary follows the spec and whose depth follows
        ``depth_extent``.

    Raises:
        ValueError: If the spec is multi-field, if the field is not discrete,
            or if ``channels`` does not have one entry per derived layer.
    """
    field = spec.single_field()
    if field.cardinality is None:
        raise ValueError(
            f"field {field.name!r} is continuous. The loss is a per-cell "
            "categorical cross-entropy, so a continuous field needs a "
            "different decoder and a different loss."
        )
    dilations = dilations_for_grid(depth_extent)
    if len(channels) != len(dilations):
        raise ValueError(
            f"depth_extent {depth_extent} derives {len(dilations)} dilated "
            f"layers {dilations}, but {len(channels)} channel widths were "
            "given. Depth follows from depth_extent, not from the predicted "
            f"grid {field.shape}. Supply one width per layer."
        )
    return GridEncoder(
        obs_channel_classes=field.cardinality,
        code_embed_dim=code_embed_dim,
        channels=channels,
        dilations=dilations,
        d_model=d_model,
        pool_to_one_token=pool_to_one_token,
    )
