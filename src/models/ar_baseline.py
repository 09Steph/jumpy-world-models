"""Arms 2 and 3: the autoregressive comparators, and the rollouts they share.

Arm 2 is autoregressive under the endpoint objective, trained by
backpropagating through an unrolled rollout to the endpoint at sampled ``h``.
Arm 3 is autoregressive under the one-step objective, trained at ``h = 1`` and
rolled out on its own argmax at evaluation.

Both reuse ``TransformerBlock`` and ``attention_mask`` from
``jumpy_transformer`` and take arm 1's tokeniser and decoder, so the arms match
on architecture by construction.

``rollout_tokens`` runs in token space, never discretises, and is the
differentiable training path. ``rollout_observations`` decodes, argmaxes and
re-encodes at every step, and is the evaluation path for both arms. Arm 2's
reported gap therefore carries a train/test discretisation difference alongside
the objective difference.
"""
# pylint: disable=duplicate-code
# The arms match arm 1 on encoder, backbone, decoder and parameter budget.
# test_parameter_counts_match_between_arms asserts it.

import flax.linen as nn
import jax
import jax.numpy as jnp

from config import (
    ARM_AR_ENDPOINT,
    ARM_AR_ONE_STEP,
    ATTENTION_WINDOW,
    ModelConfig,
)
from src.data.tokeniser import tokeniser_for_spec
from src.data.trajectory import ObservationSpec
from src.models.jumpy_transformer import (
    DROPOUT_RNG_NAME,
    TransformerBlock,
    attention_mask,
)

# Objective names, used in code, artefacts and the report alike.
OBJECTIVE_ENDPOINT: str = "endpoint"
OBJECTIVE_ONE_STEP: str = "one_step"

# Which objective each arm is trained under. Arm 1 is absent from this mapping.
# It trains on the endpoint through a different architecture.
OBJECTIVE_BY_ARM: dict[int, str] = {
    ARM_AR_ENDPOINT: OBJECTIVE_ENDPOINT,
    ARM_AR_ONE_STEP: OBJECTIVE_ONE_STEP,
}


def objective_for_arm(arm: int) -> str:
    """Return the training objective an autoregressive arm is trained under.

    Args:
        arm: One of config.ARM_AR_ENDPOINT or config.ARM_AR_ONE_STEP.

    Returns:
        OBJECTIVE_ENDPOINT or OBJECTIVE_ONE_STEP.

    Raises:
        KeyError: If the arm is not one this module builds. Arm 1 is
            ``jumpy_transformer``'s.
    """
    if arm not in OBJECTIVE_BY_ARM:
        raise KeyError(
            f"arm {arm} is not an autoregressive arm. This module builds "
            f"{ARM_AR_ENDPOINT} and {ARM_AR_ONE_STEP}. Arm 1 is built by "
            "jumpy_transformer_for_spec."
        )
    return OBJECTIVE_BY_ARM[arm]


class AutoregressiveBaseline(nn.Module):
    """One single-step core, rolled out. Arms 2 and 3 differ only in training.

    Both arms share this class. What differs is the horizon distribution their
    training batches are drawn at, which belongs to the sampler.

    The sequence is the state tokens followed by one action token. Arm 1
    attends over the state tokens plus ``h`` action tokens in one pass; this
    attends over the state tokens plus one, ``h`` times.

    Attributes:
        tokeniser: The seam, supplying state tokens and action tokens.
        decoder: The head built by that tokeniser. Applied once at the endpoint
            in the token rollout, and once per step in the observation rollout.
        num_layers: Transformer encoder depth, matched to arm 1.
        d_model: Token width, matched to arm 1.
        num_heads: Attention heads per layer.
        ffn_multiplier: Feed-forward width as a multiple of d_model.
        dropout_rate: Dropout inside each block.
        activation: Activation name, resolved by layers.get_activation.
        norm_eps: RMSNorm epsilon.
        attention_window: Local attention half-width, or None for full
            attention. Under one-token state tokenisation the sequence is short
            enough that any window covers it.
        remat_rollout: Recompute the scanned body on the backward pass instead
            of storing one set of activations per step. Numerically identical
            either way.
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
    remat_rollout: bool = False

    def setup(self) -> None:
        """Build the shared single-step backbone.

        Submodules are assigned here. A compact method may only be one, and
        this class exposes several.
        """
        # pylint: disable=attribute-defined-outside-init
        self.blocks = [
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ffn_multiplier=self.ffn_multiplier,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                norm_eps=self.norm_eps,
                name=f"block_{layer}",
            )
            for layer in range(self.num_layers)
        ]
        # The last block's output comes straight off a residual path and has
        # never been normalised.
        self.output_norm = nn.RMSNorm(epsilon=self.norm_eps, name="output_norm")

    def step(
        self,
        state_tokens: jax.Array,
        action_token: jax.Array,
        *,
        deterministic: bool,
    ) -> jax.Array:
        """Advance one state by one action, in token space.

        The only place the backbone is applied. Every token is real, so the
        mask is all-true. The horizon is the number of times this is called.

        Args:
            state_tokens: The current state, (batch, num_state_tokens, d_model).
            action_token: One action, (batch, d_model).
            deterministic: True to disable dropout.

        Returns:
            The next state's tokens, the same shape as `state_tokens`.
        """
        num_state_tokens = state_tokens.shape[1]
        tokens = jnp.concatenate(
            [state_tokens, action_token[:, None, :]], axis=1
        )
        valid = jnp.ones(tokens.shape[:2], dtype=bool)
        mask = attention_mask(valid, self.attention_window)
        for block in self.blocks:
            tokens = block(tokens, mask, deterministic=deterministic)
        tokens = self.output_norm(tokens)
        # Sliced by count, so per-cell tokenisation changes the sequence
        # length and no line here.
        return tokens[:, :num_state_tokens, :]

    def _scan(self, body, carry, xs, length: int):
        """Run a rollout body over the action axis with parameters shared.

        One core is applied at every step, so its parameters are broadcast
        and not stacked. ``variable_broadcast="params"`` with
        ``split_rngs={"params": False}`` is required; without it initialisation
        raises ``InvalidRngError`` from inside the first ``RMSNorm``. Dropout is
        split per step.

        A scan, not a Python loop. An unrolled loop would place
        ``horizon_max`` times ``num_layers`` transformer blocks in one graph.

        Args:
            body: Callable `(module, carry, x) -> (carry, None)`.
            carry: Initial carry.
            xs: Per-step inputs, stacked on a leading time axis.
            length: Number of steps, which must be static.

        Returns:
            The final carry.
        """
        if self.remat_rollout:
            body = nn.remat(body)
        scanned = nn.scan(
            body,
            variable_broadcast="params",
            split_rngs={"params": False, DROPOUT_RNG_NAME: True},
            in_axes=0,
            length=length,
        )
        final, _ = scanned(self, carry, xs)
        return final

    def rollout_tokens(
        self,
        observation: jax.Array,
        actions: jax.Array,
        horizons: jax.Array,
        *,
        deterministic: bool,
    ) -> list[jax.Array]:
        """Predict the endpoint through a latent rollout. The training path.

        Used by arm 2 at sampled ``h`` and by arm 3 at ``h = 1``. Nothing is
        discretised, so the state stays a token throughout and the decoder is
        applied once, at the end.

        The scan runs to the padded length and masks, so every batch pays
        ``actions.shape[1]`` steps whatever horizons it drew.

        Args:
            observation: Discrete observation codes at time t,
                (batch, height, width, num_channels).
            actions: Discrete action indices padded to the model's maximum
                horizon, (batch, num_action_tokens).
            horizons: True horizon per example, (batch,). Every value is at
                least 1, which lets the initial carry be overwritten
                unconditionally at step 0.
            deterministic: True to disable dropout.

        Returns:
            One logits array per observation channel, each
            (batch, height, width, classes_for_that_channel).
        """
        state = self.tokeniser.encode_state(observation)
        action_tokens = self.tokeniser.encode_actions(actions)

        def body(module, carry, action_token):
            tokens, index = carry
            stepped = module.step(
                tokens, action_token, deterministic=deterministic
            )
            active = (index < horizons)[:, None, None]
            return (jnp.where(active, stepped, tokens), index + 1), None

        tokens, _ = self._scan(
            body,
            (state, jnp.zeros(horizons.shape, jnp.int32)),
            jnp.swapaxes(action_tokens, 0, 1),
            actions.shape[1],
        )
        return self.decoder(tokens)

    def rollout_observations(
        self,
        observation: jax.Array,
        actions: jax.Array,
        horizons: jax.Array,
        *,
        deterministic: bool,
    ) -> list[jax.Array]:
        """Predict the endpoint by rolling out on the model's own argmax.

        The evaluation path for both arms. The endpoint's logits come from the
        step that produced them, not from re-decoding the final observation,
        and the rollout runs in int32 throughout.

        Args:
            observation: Discrete observation codes at time t,
                (batch, height, width, num_channels).
            actions: Discrete action indices padded to the model's maximum
                horizon, (batch, num_action_tokens).
            horizons: True horizon per example, (batch,).
            deterministic: True to disable dropout.

        Returns:
            One logits array per observation channel, each
            (batch, height, width, classes_for_that_channel).
        """
        current = observation.astype(jnp.int32)
        action_tokens = self.tokeniser.encode_actions(actions)

        def body(module, carry, action_token):
            observed, tokens, index = carry
            stepped = module.step(
                module.tokeniser.encode_state(observed),
                action_token,
                deterministic=deterministic,
            )
            logits = module.decoder(stepped)
            predicted = jnp.stack(
                [jnp.argmax(channel, axis=-1) for channel in logits], axis=-1
            ).astype(jnp.int32)
            active = index < horizons
            return (
                jnp.where(active[:, None, None, None], predicted, observed),
                jnp.where(active[:, None, None], stepped, tokens),
                index + 1,
            ), None

        _, tokens, _ = self._scan(
            body,
            (
                current,
                self.tokeniser.encode_state(current),
                jnp.zeros(horizons.shape, jnp.int32),
            ),
            jnp.swapaxes(action_tokens, 0, 1),
            actions.shape[1],
        )
        return self.decoder(tokens)

    def __call__(
        self,
        observation: jax.Array,
        actions: jax.Array,
        horizons: jax.Array,
        *,
        deterministic: bool,
    ) -> list[jax.Array]:
        """Predict the endpoint the way this arm is scored.

        The discretising rollout. Arm 1's ``__call__`` is one forward pass,
        arms 2 and 3's is ``h`` steps on their own argmax, so
        ``predict_endpoint`` needs no branch on the arm.

        The training path is not reachable from here. Arm 2's loss calls
        ``rollout_tokens`` by name.

        Args:
            observation: Discrete observation codes at time t.
            actions: Discrete action indices, padded.
            horizons: True horizon per example.
            deterministic: True to disable dropout.

        Returns:
            One logits array per observation channel.
        """
        return self.rollout_observations(
            observation, actions, horizons, deterministic=deterministic
        )


def ar_baseline_for_spec(
    spec: ObservationSpec, config: ModelConfig, *, depth_extent: int
) -> AutoregressiveBaseline:
    """Build arms 2 and 3 for an environment and a model configuration.

    The single construction site, mirroring ``jumpy_transformer_for_spec``.
    There is no ``arm`` argument. Arms 2 and 3 are architecturally identical
    and the arm selects the sampler, in the training stage.

    Every argument is the one arm 1 was built with.

    Args:
        spec: The observation being predicted. Sizes the decoder's output grid
            and supplies the vocabularies.
        config: Model architecture settings.
        depth_extent: The environment's largest grid extent, driving encoder
            depth only. Callers pass EnvConfig.max_grid_extent, which
            resolve_observation_contract sets from the environment's row.

    Returns:
        The autoregressive comparator, unbound.
    """
    tokeniser = tokeniser_for_spec(spec, config, depth_extent=depth_extent)
    return AutoregressiveBaseline(
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
        remat_rollout=config.remat_rollout,
    )
