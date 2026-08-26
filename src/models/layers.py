"""Generic network building blocks shared by the models.

Harvested from the pre-rescope tree on 2026-08-17, unchanged in behaviour.
"""
from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

# Named activations. A table, so an unknown name fails with a KeyError.
ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
}

# Output-layer kernel scaling. 1.0 is a no-op, 0.0 gives a zero kernel.
DECODER_OUTSCALE: float = 1.0

# Convolution kernel size throughout every trunk.
CONV_KERNEL: tuple[int, int] = (3, 3)


def get_activation(name: str) -> Callable[[jax.Array], jax.Array]:
    """Look up a named activation function.

    Args:
        name: Activation name; one of "gelu", "silu", "relu", "tanh".

    Returns:
        The activation function.

    Raises:
        KeyError: If the name is not a known activation.
    """
    return ACTIVATIONS[name]


def output_kernel_init(outscale: float) -> Callable[..., jax.Array]:
    """Build a kernel initialiser scaled by a multiplier.

    Applies to a component's final layer only, never its trunk.

    Args:
        outscale: Multiplier on an already-initialised kernel. 1.0 is a no-op,
            0.0 gives an exactly zero kernel.

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

    Used by both the encoder and the convolutional decoder. Must be called
    inside an @nn.compact __call__ so Flax scopes the layers to the calling
    module.

    Args:
        features: Input feature grid, shape (batch, height, width, channels).
        channels: Output width per layer, one entry per layer.
        activation: Activation name, resolved by get_activation.
        norm_eps: RMSNorm epsilon.
        dilations: Dilation rate per layer, same length as ``channels``. None
            leaves every layer undilated, the right default for a trunk that
            expands from an already-global vector. A trunk that summarises a
            grid needs dilation. See GridEncoder.

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


def dilations_for_grid(grid_extent: int) -> tuple[int, ...]:
    """Return the dilation schedule a grid of this size needs, and no more.

    Doubling dilations until the receptive field spans the grid gives the
    shallowest stack that can see the whole layout. A dilation wider than the
    grid samples only zero padding.

    Pass the largest grid the environment produces, not the current mode's, so
    every observation mode shares one encoder shape.

    Args:
        grid_extent: The larger of the grid's height and width, for the largest
            observation mode the environment emits.

    Returns:
        Doubling dilations, shortest schedule whose receptive field reaches
        `grid_extent`.

    Raises:
        ValueError: If grid_extent is not positive.
    """
    if grid_extent < 1:
        raise ValueError(f"grid_extent must be positive, got {grid_extent}")
    dilations: list[int] = []
    rate = 1
    while receptive_field(tuple(dilations)) < grid_extent:
        dilations.append(rate)
        rate *= 2
    return tuple(dilations)


def receptive_field(dilations: tuple[int, ...]) -> int:
    """Return the receptive field of a stack of 3x3 stride-1 convolutions.

    Each layer adds ``2 * dilation`` to the span, giving
    ``1 + 2 * sum(dilations)``.

    Args:
        dilations: Dilation rate per layer.

    Returns:
        Receptive field in cells, along one spatial axis.
    """
    return 1 + 2 * sum(dilations)


def mlp_trunk(
    features: jax.Array,
    hidden_size: int,
    layers: int,
    activation: str,
    norm_eps: float,
) -> jax.Array:
    """Shared MLP trunk: Dense, then RMSNorm, then activation, repeated.

    Must be called inside an @nn.compact __call__ so Flax scopes the layers to
    the calling module.

    Args:
        features: Input features, shape (..., feature_dim).
        hidden_size: Hidden width of each trunk layer.
        layers: Number of trunk layers.
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
