"""Rotary positional embedding (RoPE) for the action-sequence transformer.

The transformer is an encoder with no causal mask, so without this it is
permutation-invariant over its inputs. Rotary carries no per-position
parameters. Position enters as a rotation angle, and the attention score
between two tokens depends only on their separation.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

# Wavelength base for the rotation frequencies. Sets how quickly the rotation
# angle varies with dimension index.
ROPE_BASE: float = 10_000.0


def rotary_angles(
    positions: jax.Array, head_dim: int, base: float = ROPE_BASE
) -> tuple[jax.Array, jax.Array]:
    """Return the cosine and sine tables for a set of sequence positions.

    Args:
        positions: Integer positions, shape (seq,).
        head_dim: Width of one attention head. Must be even; the rotation acts
            on pairs of dimensions.
        base: Wavelength base for the frequency schedule.

    Returns:
        Two arrays of shape (seq, head_dim // 2), the cosines and sines of the
        rotation angle for each position and dimension pair.

    Raises:
        ValueError: If head_dim is odd.
    """
    if head_dim % 2 != 0:
        raise ValueError(
            f"head_dim must be even for a rotary embedding, got {head_dim}. "
            "The rotation acts on pairs of dimensions."
        )
    half = head_dim // 2
    inverse_frequencies = 1.0 / (
        base ** (jnp.arange(half, dtype=jnp.float32) * 2.0 / head_dim)
    )
    angles = positions.astype(jnp.float32)[:, None] * inverse_frequencies[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rotary(
    x: jax.Array, cos: jax.Array, sin: jax.Array
) -> jax.Array:
    """Rotate a query or key tensor by its position-dependent angle.

    Split-half convention: dimension ``i`` pairs with ``i + head_dim // 2``.
    Queries and keys must use the same convention.

    Args:
        x: Queries or keys, shape (batch, seq, heads, head_dim).
        cos: Cosine table, shape (seq, head_dim // 2).
        sin: Sine table, shape (seq, head_dim // 2).

    Returns:
        Rotated tensor of the same shape as `x`.
    """
    first, second = jnp.split(x, 2, axis=-1)
    # Broadcast the tables over batch and head axes.
    cos_b = cos[None, :, None, :]
    sin_b = sin[None, :, None, :]
    rotated_first = first * cos_b - second * sin_b
    rotated_second = first * sin_b + second * cos_b
    return jnp.concatenate([rotated_first, rotated_second], axis=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Applies rotary position information to queries and keys.

    Carries no learned parameters.

    Attributes:
        head_dim: Width of one attention head. Must be even.
        base: Wavelength base for the frequency schedule.
    """

    head_dim: int
    base: float = ROPE_BASE

    @nn.compact
    def __call__(
        self,
        queries: jax.Array,
        keys: jax.Array,
        positions: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Rotate queries and keys in place along the sequence axis.

        Args:
            queries: Shape (batch, seq, heads, head_dim).
            keys: Shape (batch, seq, heads, head_dim).
            positions: Integer positions, shape (seq,). Defaults to
                0, 1, ..., seq - 1. Supplied explicitly when padding means a
                token's slot index is not its true position.

        Returns:
            The rotated queries and keys, shapes unchanged.
        """
        sequence_length = queries.shape[1]
        if positions is None:
            positions = jnp.arange(sequence_length)
        cos, sin = rotary_angles(positions, self.head_dim, self.base)
        return apply_rotary(queries, cos, sin), apply_rotary(keys, cos, sin)
