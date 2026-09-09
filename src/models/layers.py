"""Generic network building blocks shared by the models."""
from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

# Re-exported, not used here. `src.models.encoder` and the tests import
# `dilations_for_grid` and `receptive_field` from this module.
from config import dilations_for_grid, receptive_field  # noqa: F401

__all__ = ["dilations_for_grid", "receptive_field"]

ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
}

# Output-layer kernel scaling.
DECODER_OUTSCALE: float = 1.0

CONV_KERNEL: tuple[int, int] = (3, 3)


def get_activation(name: str) -> Callable[[jax.Array], jax.Array]:
    """Look up a named activation function.

    Raises:
        KeyError: If the name is not a known activation.
    """
    return ACTIVATIONS[name]


def output_kernel_init(outscale: float) -> Callable[..., jax.Array]:
    """Build a kernel initialiser scaled by a multiplier.

    Applies to a component's final layer only, never its trunk.

    Args:
        outscale: Multiplier on an already-initialised kernel.

    Returns:
        A Flax kernel initialiser taking (key, shape, dtype).
    """
    base = nn.initializers.lecun_normal()

    def scaled(
        key: jax.Array, shape: tuple[int, ...], dtype=jnp.float32
    ) -> jax.Array:
        return base(key, shape, dtype) * outscale

    return scaled


def conv_trunk(
    features: jax.Array,
    channels: tuple[int, ...],
    activation: str,
    norm_eps: float,
    dilations: tuple[int, ...] | None = None,
) -> jax.Array:
    """Shared convolution trunk: Conv, then RMSNorm, then activation, repeated.

    Must be called inside an @nn.compact __call__.

    Args:
        features: Input feature grid, shape (batch, height, width, channels).
        channels: Output width per layer, one entry per layer.
        activation: Activation name, resolved by get_activation.
        norm_eps: RMSNorm epsilon.
        dilations: Dilation rate per layer, same length as ``channels``. None
            leaves every layer undilated.

    Returns:
        Feature grid of shape (batch, height, width, channels[-1]). SAME
        padding preserves the spatial dimensions at every dilation rate.

    Raises:
        ValueError: If ``dilations`` is given and its length differs from
            ``channels``.
    """
    rates = (1,) * len(channels) if dilations is None else dilations
    if len(rates) != len(channels):
        raise ValueError(
            f"dilations has {len(rates)} entries against {len(channels)} "
            f"channel widths. One dilation per layer, or None for undilated."
        )
    activation_fn = get_activation(activation)
    for layer, (width, rate) in enumerate(zip(channels, rates)):
        features = nn.Conv(
            features=width,
            kernel_size=CONV_KERNEL,
            kernel_dilation=(rate, rate),
            padding="SAME",
            name=f"conv_{layer}",
        )(features)
        features = nn.RMSNorm(epsilon=norm_eps)(features)
        features = activation_fn(features)
    return features


def mlp_trunk(
    features: jax.Array,
    hidden_size: int,
    layers: int,
    activation: str,
    norm_eps: float,
) -> jax.Array:
    """Shared MLP trunk: Dense, then RMSNorm, then activation, repeated.

    Must be called inside an @nn.compact __call__.

    Args:
        features: Input features, shape (..., feature_dim).
        activation: Activation name, resolved by get_activation.
        norm_eps: RMSNorm epsilon.

    Returns:
        Trunk output, shape (..., hidden_size).
    """
    activation_fn = get_activation(activation)
    hidden = features
    for _ in range(layers):
        hidden = nn.Dense(hidden_size)(hidden)
        hidden = nn.RMSNorm(epsilon=norm_eps)(hidden)
        hidden = activation_fn(hidden)
    return hidden
