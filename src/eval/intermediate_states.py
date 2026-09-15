"""Decode the states an autoregressive arm produces between start and endpoint.

The shipped scan emits no per-step output, so each rollout is rebuilt from the
model's public methods through ``apply(method=...)``. A test checks the rebuilt
endpoint against the shipped one.

``rollout_tokens`` is the training path and keeps the state as a token.
``rollout_observations`` is the discrete scoring path, re-encoding an argmax at
every step, and is rebuilt for discrete representations only. Continuous arms
are scored through ``rollout_values``, which is not rebuilt, so pixel
intermediates come from the training path. A number read off one path does not
reproduce on another.

Each per-step record names its distance-from-truth unit: a differing-cell
fraction on a discrete representation, a squared error on a continuous one.

Called from --intermediate-states only. It reads checkpoints and datasets,
trains no model and writes only under its own directory.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib

# Agg before pyplot. This runs headless on the cluster and in pytest.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

from config import (  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
    EVALUATION_HORIZONS,
    INTERMEDIATE_STATE_ARMS,
    INTERMEDIATE_STATES_FIGURE_TEMPLATE,
    INTERMEDIATE_STATES_MAX_EPISODE_DRAWS,
    INTERMEDIATE_STATES_RECORDS_FILENAME,
    INTERMEDIATE_STATES_SUMMARY_FILENAME,
    METRICS_TEMPLATE,
    OUTPUTS_DIR,
    REPRESENTATION_SPECS,
    REPRESENTATION_SYMBOLIC,
    ROLLOUT_PATH_OBSERVATIONS,
    ROLLOUT_PATH_TOKENS,
    ROLLOUT_PATHS,
    SPLIT_PROVENANCE_FILENAME,
    TRUTH_SOURCE_FRESH_ROLLOUT,
    TRUTH_SOURCE_TEST_TRAJECTORY,
    ExperimentConfig,
    apply_fast_mode,
    checkpoints_dir,
    data_dir,
    eval_dir,
    intermediate_states_dir,
    resolve_family_defaults,
    resolve_observation_contract,
)
from src.data.trajectory import ObservationMode  # noqa: E402  pylint: disable=wrong-import-position
from src.data.trajectory_store import TrajectoryStore  # noqa: E402  pylint: disable=wrong-import-position
from src.eval.metrics import model_mse  # noqa: E402  pylint: disable=wrong-import-position
from src.eval.figure_style import FIGURE_DPI  # noqa: E402  pylint: disable=wrong-import-position
from src.eval.plots import CHANNEL_LABELS  # noqa: E402  pylint: disable=wrong-import-position
from src.pipeline.displacement import hamming_displacement  # noqa: E402  pylint: disable=wrong-import-position
from src.pipeline.train import (  # noqa: E402  pylint: disable=wrong-import-position
    CHECKPOINT_BEST_PARAMS_KEY,
    build_arm,
    build_training_source,
    restore_checkpoint,
    restore_target,
)
from src.utils.logging_setup import get_logger  # noqa: E402  pylint: disable=wrong-import-position
from src.utils.optimisers import build_optimiser  # noqa: E402  pylint: disable=wrong-import-position
from src.utils.paths import ensure_dir, safe_rel  # noqa: E402  pylint: disable=wrong-import-position

logger = get_logger(__name__)

# One decoded grid per column, one channel per row.
STRIP_COLUMN_WIDTH: float = 0.35
STRIP_ROW_HEIGHT: float = 0.42
STRIP_LABEL_FONTSIZE: int = 4
STRIP_ROW_LABEL_FONTSIZE: int = 6

# Metadata cleared so reruns write byte-identical files.
FIGURE_METADATA: dict[str, None] = {"Software": None}

# The bounds a continuous decoder's sigmoid output lies between.
DECODED_VALUE_BOUNDS: tuple[float, float] = (0.0, 1.0)

# Record fields.
RECORD_STEP_KEY: str = "step"
RECORD_VALID_KEY: str = "grid_is_valid"
RECORD_TRUTH_DISTANCE_KEY: str = "cells_differing_from_truth"
RECORD_TOKEN_DISTANCE_KEY: str = "token_distance_from_previous"
RECORD_DECODED_DISTANCE_KEY: str = "decoded_distance_from_previous"

# What RECORD_TRUTH_DISTANCE_KEY is measured in. A discrete distance is a
# fraction of cells and a continuous one is a squared error, and the two must
# not be averaged together in a summary that cannot tell them apart.
RECORD_TRUTH_DISTANCE_UNIT_KEY: str = "truth_distance_unit"
TRUTH_DISTANCE_UNIT_CELLS: str = "fraction_of_cells_differing"
TRUTH_DISTANCE_UNIT_MSE: str = "mean_squared_error"

# The summary's own fields. endpoint_cells_differing is a squared error on a
# continuous domain, and a summary row does not name its unit.
SUMMARY_DISTINCT_KEY: str = "distinct_cells_differing"
SUMMARY_TOKEN_MEDIAN_KEY: str = "token_distance_median"
SUMMARY_TOKEN_MEAN_KEY: str = "token_distance_mean"
SUMMARY_DECODED_MEDIAN_KEY: str = "decoded_distance_median"
SUMMARY_DECODED_MEAN_KEY: str = "decoded_distance_mean"
SUMMARY_TOKEN_STEP_1_KEY: str = "token_distance_step_1"
SUMMARY_ENDPOINT_DIFFERING_KEY: str = "endpoint_cells_differing"
SUMMARY_ENDPOINT_ACCURACY_KEY: str = "endpoint_per_cell_accuracy"
SUMMARY_REPORTED_ACCURACY_KEY: str = "reported_per_cell_accuracy"
SUMMARY_VALIDATION_DELTA_KEY: str = "validation_delta"
SUMMARY_STEPS_KEY: str = "steps"
SUMMARY_SOURCE_KEY: str = "source"
SUMMARY_TRAJECTORY_INDEX_KEY: str = "trajectory_index"
SUMMARY_DRAW_SEED_KEY: str = "draw_seed"
SUMMARY_ELIGIBLE_COUNT_KEY: str = "eligible_count"
SUMMARY_VALID_ALL_KEY: str = "grid_is_valid_all"

# What identifies one summary row. cell_fields omits the arm, so it cannot key
# the summary. Rollout path and truth source are not part of the key.
SUMMARY_KEY_FIELDS: tuple[str, ...] = ("seed", "mode", "arm", "horizon")

# What identifies one per-step record. Rollout path and truth source are not
# part of the key.
RECORD_KEY_FIELDS: tuple[str, ...] = (
    "seed",
    "mode",
    "arm",
    "horizon",
    RECORD_STEP_KEY,
)

# The longest horizon a drawn episode has to serve. Eligibility is monotone in
# the horizon, so a draw from this set is valid at every shorter one.
LONGEST_EVALUATION_HORIZON: int = max(EVALUATION_HORIZONS)


@dataclass(frozen=True)
class ObservationDomain:
    """How one representation's decoded values are read, checked and compared.

    Attributes:
        grid_shape: Spatial grid shape, (height, width).
        cardinality: Class count per channel for a discrete representation,
            None for a continuous one.
        value_range: Inclusive (low, high) stored bounds for a continuous
            representation, None for a discrete one.
    """

    grid_shape: tuple[int, int]
    cardinality: tuple[int, ...] | None
    value_range: tuple[int, int] | None

    def is_continuous(self) -> bool:
        """Return whether decoded values are continuous."""
        return self.value_range is not None


def domain_for(config: ExperimentConfig) -> ObservationDomain:
    """Resolve the decode domain from the configuration's representation.

    Args:
        config: The configuration a cell was trained under.

    Returns:
        The domain, continuous when the representation declares stored bounds.

    Raises:
        KeyError: If the representation has no entry.
    """
    return ObservationDomain(
        grid_shape=config.model.obs_grid_shape,
        cardinality=config.model.obs_channel_classes,
        value_range=REPRESENTATION_SPECS[
            config.data.representation
        ].value_range,
    )


def argmax_grid(logits: Sequence[jax.Array]) -> jax.Array:
    """Return per-channel argmax classes stacked into a grid.

    Args:
        logits: One logits array per observation channel.

    Returns:
        Integer classes, (batch, height, width, num_channels).
    """
    return jnp.stack(
        [jnp.argmax(channel, axis=-1) for channel in logits], axis=-1
    ).astype(jnp.int32)


def decode_state(
    logits: Sequence[jax.Array], value_range: tuple[int, int] | None
) -> jax.Array:
    """Return the state a decoder output stands for, per representation.

    A discrete decoder emits one logits array per channel and the state is
    their argmax. A continuous decoder emits the state itself, one array over
    all channels, and there is no class axis to reduce.

    Args:
        logits: The decoder's output.
        value_range: Stored bounds for a continuous representation, None for a
            discrete one.

    Returns:
        The decoded state, (batch, height, width, num_channels).
    """
    if value_range is None:
        return argmax_grid(logits)
    return logits[0]


def token_rollout(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    module, observation, actions, horizons, value_range, *, deterministic
):
    """Rebuild ``rollout_tokens`` in a Python loop, keeping every state.

    Mirrors the shipped scan body: one ``step`` per action token, masked by the
    per-example horizon, the carry overwritten only while active.

    Args:
        module: The bound module, supplied by ``apply``.
        observation: Stored observation values at time t.
        actions: Discrete action indices, padded.
        horizons: True horizon per example.
        value_range: Stored bounds for a continuous representation, None for a
            discrete one.
        deterministic: True to disable dropout.

    Returns:
        The per-step grids and token carries, each stacked on a leading step
        axis, and the endpoint logits.
    """
    tokens = module.tokeniser.encode_state(observation)
    action_tokens = module.tokeniser.encode_actions(actions)
    index = jnp.zeros(horizons.shape, jnp.int32)
    grids = [decode_state(module.decoder(tokens), value_range)]
    carries = [tokens]
    for position in range(actions.shape[1]):
        stepped = module.step(
            tokens, action_tokens[:, position], deterministic=deterministic
        )
        active = (index < horizons)[:, None, None]
        tokens = jnp.where(active, stepped, tokens)
        index = index + 1
        grids.append(decode_state(module.decoder(tokens), value_range))
        carries.append(tokens)
    return jnp.stack(grids), jnp.stack(carries), module.decoder(tokens)


def observation_rollout(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    module, observation, actions, horizons, value_range, *, deterministic
):
    """Rebuild ``rollout_observations`` in a Python loop, keeping every grid.

    Discrete only. The shipped method it mirrors argmaxes over a class axis and
    carries the result in int32, and `trace_rollout` refuses this path under a
    continuous representation rather than reproducing that here.

    Args:
        module: The bound module, supplied by ``apply``.
        observation: Discrete observation codes at time t.
        actions: Discrete action indices, padded.
        horizons: True horizon per example.
        value_range: None. Present so both rollouts share one signature.
        deterministic: True to disable dropout.

    Returns:
        The per-step grids and token carries, each stacked on a leading step
        axis, and the endpoint logits.
    """
    del value_range
    current = observation.astype(jnp.int32)
    action_tokens = module.tokeniser.encode_actions(actions)
    tokens = module.tokeniser.encode_state(current)
    index = jnp.zeros(horizons.shape, jnp.int32)
    grids = [current]
    carries = [tokens]
    for position in range(actions.shape[1]):
        stepped = module.step(
            module.tokeniser.encode_state(current),
            action_tokens[:, position],
            deterministic=deterministic,
        )
        predicted = argmax_grid(module.decoder(stepped))
        active = index < horizons
        current = jnp.where(active[:, None, None, None], predicted, current)
        tokens = jnp.where(active[:, None, None], stepped, tokens)
        index = index + 1
        grids.append(current)
        carries.append(tokens)
    return jnp.stack(grids), jnp.stack(carries), module.decoder(tokens)


# The rollout each path name reconstructs.
ROLLOUT_FUNCTIONS: dict[str, Callable] = {
    ROLLOUT_PATH_TOKENS: token_rollout,
    ROLLOUT_PATH_OBSERVATIONS: observation_rollout,
}


@dataclass(frozen=True)
class RolloutTrace:
    """One arm's rollout, kept step by step.

    Attributes:
        grids: Decoded states, (steps, batch, height, width, channels). Step 0
            is the decoded start on the tokens path and the input observation
            on the observations path.
        carries: Token carries before the decoder,
            (steps, batch, num_state_tokens, d_model).
        endpoint: One logits array per channel, or one array of values on a
            continuous domain.
    """

    grids: jax.Array
    carries: jax.Array
    endpoint: list[jax.Array]


@dataclass(frozen=True)
class DecodeRequest:
    """What a decode reads and where it writes.

    Attributes:
        run_name: Run whose checkpoints are read and whose name the artefacts
            are filed under.
        env_name: Registered environment name.
        rollout_path: One of config.ROLLOUT_PATHS, naming which rollout is
            drawn.
        checkpoint_root: Replaces `outputs/` when the checkpoints live off the
            repository, keeping every level below it. None reads in place. Also
            rearranges the dataset path, which sits under the same root.
        fast: When True the tree hangs off `outputs/fast/`.
        truth_source: One of config.TRUTH_SOURCES. The held-out episode is the
            default and falls back to a fresh roll when no shards are present.
        representation: One of config.REPRESENTATIONS, naming what the
            checkpoints were trained on. It selects the decode domain, so a
            request that names the wrong one restores against the wrong
            template.
    """

    run_name: str
    env_name: str
    rollout_path: str = ROLLOUT_PATH_TOKENS
    checkpoint_root: Path | None = None
    fast: bool = False
    truth_source: str = TRUTH_SOURCE_TEST_TRAJECTORY
    representation: str = REPRESENTATION_SYMBOLIC


def config_for(request: DecodeRequest, seed: int) -> ExperimentConfig:
    """Compose the configuration a cell's checkpoint restores against.

    Only the environment, representation and fast mode come from the request.
    Collection policy, slip and early termination stay at their defaults, so a
    fresh-rollout truth series is drawn under default dynamics and a uniform
    policy whatever the run used.

    Args:
        request: What is being decoded.
        seed: The reporting seed.

    Returns:
        The configuration, with the environment's observation contract and its
        family defaults resolved, and fast mode applied when the checkpoints
        were written under it.

    Note:
        Fast mode scales the model, so a fast checkpoint restores only against
        a fast template. The three are applied in main.py's own order: the
        contract, then the family defaults it feeds, then the scaling over
        both.
    """
    config = ExperimentConfig(seed=seed, run_name=request.run_name)
    config = replace(
        config,
        env=replace(config.env, name=request.env_name),
        data=replace(config.data, representation=request.representation),
    )
    config = resolve_observation_contract(config)
    config = resolve_family_defaults(config)
    return apply_fast_mode(config) if request.fast else config


def checkpoint_directory(
    request: DecodeRequest, seed: int, mode: ObservationMode, arm: int
) -> Path:
    """Locate one cell's checkpoint, honouring an off-repository root.

    Args:
        request: What is being decoded.
        seed: The reporting seed.
        mode: The observation mode the arm was trained on.
        arm: One of config.INTERMEDIATE_STATE_ARMS.

    Returns:
        The directory holding that cell's checkpoint.
    """
    directory = checkpoints_dir(
        request.run_name,
        seed,
        request.fast,
        request.env_name,
        observation_mode=mode.value,
        arm=arm,
    )
    if request.checkpoint_root is None:
        return directory
    return request.checkpoint_root / directory.relative_to(OUTPUTS_DIR)


def load_arm(
    request: DecodeRequest,
    config: ExperimentConfig,
    mode: ObservationMode,
    arm: int,
):
    """Restore one arm's best-loss parameters.

    Args:
        request: What is being decoded.
        config: The configuration that cell was trained under.
        mode: The observation mode the arm was trained on.
        arm: One of config.INTERMEDIATE_STATE_ARMS.

    Returns:
        The unbound model and its best-loss parameter tree.

    Raises:
        FileNotFoundError: If the checkpoint directory is absent.
    """
    directory = checkpoint_directory(request, config.seed, mode, arm)
    if not directory.is_dir():
        raise FileNotFoundError(f"no checkpoint under {safe_rel(directory)}")
    model, template = build_arm(config, mode, arm=arm)
    restored = restore_checkpoint(
        directory,
        restore_target(
            template,
            build_optimiser(config.train).init(template),
            config.train.total_steps,
        ),
    )
    return model, restored[CHECKPOINT_BEST_PARAMS_KEY]


def trace_rollout(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model, params, observation, actions, *, rollout_path: str,
    domain: ObservationDomain,
) -> RolloutTrace:
    """Reconstruct one rollout and keep every step of it.

    Args:
        model: The unbound arm.
        params: The parameter tree to apply.
        observation: The start observation, (1, height, width, channels).
        actions: The action sequence, (1, horizon).
        rollout_path: One of config.ROLLOUT_PATHS.
        domain: The decode domain of the representation being traced.

    Returns:
        The per-step grids, the token carries and the endpoint logits.

    Raises:
        KeyError: If the path is not one this module reconstructs.
        ValueError: If the observations path is asked of a continuous
            representation, which the shipped rollout does not support.
    """
    if rollout_path not in ROLLOUT_FUNCTIONS:
        raise KeyError(
            f"unknown rollout path {rollout_path!r}, expected one of "
            f"{ROLLOUT_PATHS}"
        )
    if domain.is_continuous() and rollout_path == ROLLOUT_PATH_OBSERVATIONS:
        raise ValueError(
            f"the {ROLLOUT_PATH_OBSERVATIONS!r} rollout is defined for a "
            "discrete representation only: AutoregressiveBaseline."
            "rollout_observations argmaxes over a class axis, which a "
            f"continuous representation does not have. Use "
            f"{ROLLOUT_PATH_TOKENS!r}."
        )
    horizons = jnp.asarray([actions.shape[1]], jnp.int32)
    grids, carries, endpoint = model.apply(
        {"params": params},
        observation,
        actions,
        horizons,
        domain.value_range,
        deterministic=True,
        method=ROLLOUT_FUNCTIONS[rollout_path],
    )
    return RolloutTrace(grids=grids, carries=carries, endpoint=endpoint)


def grid_is_valid(grid: jax.Array, cardinality: Sequence[int]) -> bool:
    """Return whether every cell holds a legal class for its channel.

    Args:
        grid: Integer classes, (..., height, width, num_channels).
        cardinality: Class count per channel.

    Returns:
        True when every entry lies inside its own channel's range.
    """
    limits = jnp.asarray(cardinality)
    return bool(jnp.all(grid >= 0) and jnp.all(grid < limits))


def values_are_valid(values: jax.Array) -> bool:
    """Return whether every decoded value lies inside the decoder's bounds.

    Always True for the shipped decoder, which ends in a sigmoid.

    Args:
        values: One decoded state, (..., height, width, channels).

    Returns:
        True when every entry lies inside DECODED_VALUE_BOUNDS.
    """
    low, high = DECODED_VALUE_BOUNDS
    return bool(jnp.all(values >= low) and jnp.all(values <= high))


def state_is_valid(state: jax.Array, domain: ObservationDomain) -> bool:
    """Return whether one decoded state holds legal values for its domain.

    Args:
        state: One decoded state, (height, width, channels).
        domain: The decode domain of the representation.

    Returns:
        Class membership on a discrete domain, bounds membership on a
        continuous one.
    """
    if domain.is_continuous():
        return values_are_valid(state)
    return grid_is_valid(state, domain.cardinality)


def cells_differing(predicted: jax.Array, truth: jax.Array) -> float:
    """Return the fraction of cells differing in any channel.

    Args:
        predicted: One decoded grid, (height, width, channels).
        truth: The true state at that step, the same shape.

    Returns:
        The any-channel differing fraction.
    """
    columns = hamming_displacement(
        np.asarray(predicted)[None], np.asarray(truth)[None]
    )
    return float(columns[0, -1])


def pixel_error_from_truth(
    predicted: jax.Array,
    truth: jax.Array,
    grid_shape: tuple[int, int],
    value_range: tuple[int, int],
) -> float:
    """Return a decoded state's squared error against the true state.

    The evaluation's own model error, so a decoded state is scored the way
    every reported continuous prediction is. The decoded side is the decoder's
    output and the truth side is stored, which is the pairing `model_mse`
    takes; the baseline's `copy_mse` normalises both sides and is not it.

    Args:
        predicted: One decoded state, (height, width, channels), as the
            decoder emitted it.
        truth: The true state at that step, stored, the same shape.
        grid_shape: Spatial grid shape, (height, width).
        value_range: Inclusive (low, high) bounds of the stored values.

    Returns:
        The mean squared error on the [0, 1] scale.
    """
    return float(
        model_mse(
            jnp.asarray(predicted)[None],
            jnp.asarray(truth).reshape(1, -1),
            grid_shape,
            value_range,
        )
    )


def distance_from_truth(
    predicted: jax.Array, truth: jax.Array, domain: ObservationDomain
) -> float:
    """Return one decoded state's distance from the true state.

    Args:
        predicted: One decoded state, (height, width, channels).
        truth: The true state at that step, the same shape.
        domain: The decode domain of the representation.

    Returns:
        The any-channel differing fraction on a discrete domain, the mean
        squared error on a continuous one. The record names which.
    """
    if domain.is_continuous():
        return pixel_error_from_truth(
            predicted, truth, domain.grid_shape, domain.value_range
        )
    return cells_differing(predicted, truth)


def token_distances(carries: jax.Array) -> list[float]:
    """Return the token-space distance from each step's carry to the previous.

    Taken before the decoder, so it separates a latent that moves without
    crossing a class boundary from one that does not move.

    Args:
        carries: Token carries, (steps, batch, num_state_tokens, d_model).

    Returns:
        One Euclidean distance per step, the first being zero by definition.
        Unnormalised, so it scales with the number of state tokens.
    """
    values = [0.0]
    for index in range(1, carries.shape[0]):
        difference = carries[index] - carries[index - 1]
        values.append(float(jnp.linalg.norm(difference)))
    return values


def decoded_distances(frames: jax.Array) -> list[float]:
    """Return the mean squared change from each decoded state to the previous one.

    Applied on every domain. On a discrete one it squares differences of class
    codes, so its size depends on the class numbering.

    Args:
        frames: Decoded states, (steps, height, width, channels), as the
            decoder emitted them.

    Returns:
        One value per step, the first being zero by definition.
    """
    values = [0.0]
    for index in range(1, frames.shape[0]):
        difference = frames[index] - frames[index - 1]
        values.append(float(jnp.mean(difference.astype(jnp.float32) ** 2)))
    return values


def trace_records(
    trace: RolloutTrace,
    truth: jax.Array,
    domain: ObservationDomain,
    fields: dict[str, Any],
) -> list[dict]:
    """Measure one trace step by step.

    Args:
        trace: The reconstructed rollout.
        truth: True states at every step, (steps, height, width, channels).
        domain: The decode domain of the representation.
        fields: Identifying fields copied onto every record.

    Returns:
        One record per step, carrying validity, distance from the true state
        with the unit it is measured in, the token-space step distance and the
        decoded-space step distance.

    Raises:
        ValueError: If the truth series does not cover every decoded step.
    """
    if truth.shape[0] != trace.grids.shape[0]:
        raise ValueError(
            f"truth covers {truth.shape[0]} steps against the rollout's "
            f"{trace.grids.shape[0]}"
        )
    distances = token_distances(trace.carries)
    changes = decoded_distances(trace.grids[:, 0])
    unit = (
        TRUTH_DISTANCE_UNIT_MSE
        if domain.is_continuous()
        else TRUTH_DISTANCE_UNIT_CELLS
    )
    records = []
    for step, grid in enumerate(trace.grids):
        records.append(
            {
                **fields,
                RECORD_STEP_KEY: step,
                RECORD_VALID_KEY: state_is_valid(grid, domain),
                RECORD_TRUTH_DISTANCE_KEY: distance_from_truth(
                    grid[0], truth[step], domain
                ),
                RECORD_TRUTH_DISTANCE_UNIT_KEY: unit,
                RECORD_TOKEN_DISTANCE_KEY: distances[step],
                RECORD_DECODED_DISTANCE_KEY: changes[step],
            }
        )
    return records


def episode_for_horizon(
    config: ExperimentConfig, mode: ObservationMode, horizon: int
):
    """Draw one real episode long enough to be scored at a horizon.

    The action sequence and the true states come from the project's own
    generator, so the model is conditioned on actions that were really executed
    and each intermediate has a true state to be measured against.

    Args:
        config: The configuration that cell was trained under.
        mode: Which view to take.
        horizon: Steps the episode must cover.

    Returns:
        The start observation with a batch axis, the commanded actions with a
        batch axis, and the true states for every step including the start.

    Raises:
        RuntimeError: If no episode of the required length is drawn within
            config.INTERMEDIATE_STATES_MAX_EPISODE_DRAWS attempts.
    """
    source = build_training_source(config)
    episodes = source.trajectories(jax.random.PRNGKey(config.data_seed))
    for _ in range(INTERMEDIATE_STATES_MAX_EPISODE_DRAWS):
        episode = next(episodes)
        if len(episode) < horizon:
            continue
        frames = episode.observations[mode]
        return (
            frames[:1],
            episode.actions[None, :horizon],
            frames[: horizon + 1],
        )
    raise RuntimeError(
        f"no episode of at least {horizon} steps in "
        f"{INTERMEDIATE_STATES_MAX_EPISODE_DRAWS} draws on "
        f"{config.env.name}. Raise max_episode_steps or lower the horizon."
    )


@dataclass(frozen=True)
class TruthSource:
    """One episode's start, its commanded actions and its true states.

    Attributes:
        observation: Start observation with a batch axis.
        actions: Commanded actions with a batch axis.
        truth: True states at every step including the start.
        source: One of config.TRUTH_SOURCES.
        trajectory_index: Position in the store, on the held-out path.
        draw_seed: The seed the draw used, on the held-out path.
        eligible_count: How many held-out episodes could have been drawn.
    """

    observation: jax.Array
    actions: jax.Array
    truth: jax.Array
    source: str
    trajectory_index: int | None = None
    draw_seed: int | None = None
    eligible_count: int | None = None

    def provenance(self) -> dict[str, Any]:
        """Return the fields naming where this episode came from."""
        return {
            SUMMARY_SOURCE_KEY: self.source,
            SUMMARY_TRAJECTORY_INDEX_KEY: self.trajectory_index,
            SUMMARY_DRAW_SEED_KEY: self.draw_seed,
            SUMMARY_ELIGIBLE_COUNT_KEY: self.eligible_count,
        }


def dataset_directory(request: DecodeRequest, seed: int) -> Path:
    """Locate one seed's dataset, honouring an off-repository root.

    Applies the same root rearrangement as `checkpoint_directory`.

    Args:
        request: What is being decoded.
        seed: The seed whose dataset is read.

    Returns:
        The directory holding that seed's shards and split provenance.
    """
    directory = data_dir(request.run_name, seed, request.fast, request.env_name)
    if request.checkpoint_root is None:
        return directory
    return request.checkpoint_root / directory.relative_to(OUTPUTS_DIR)


def test_split_indices(directory: Path) -> list[int]:
    """Read the persisted held-out indices.

    Args:
        directory: The dataset directory.

    Returns:
        The test split's trajectory indices, in written order.

    Raises:
        FileNotFoundError: If the split provenance is absent.
    """
    path = directory / SPLIT_PROVENANCE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no split provenance at {safe_rel(path)}")
    written = json.loads(path.read_text(encoding="utf-8"))
    return list(written["splits"]["test"]["indices"])


def eligible_indices(
    trajectories: Sequence[Any], indices: Sequence[int], horizon: int
) -> list[int]:
    """Return the indices whose episode can be scored at a horizon.

    Args:
        trajectories: Every episode, in store order.
        indices: The candidate indices.
        horizon: Steps the episode must cover.

    Returns:
        The subset covering the horizon.
    """
    return [index for index in indices if len(trajectories[index]) >= horizon]


def draw_test_episode(
    request: DecodeRequest,
    config: ExperimentConfig,
    mode: ObservationMode,
    horizon: int,
) -> TruthSource:
    """Draw one held-out episode and take its truth series.

    One draw per seed, from the set eligible at the longest evaluation horizon,
    shared by both observation modes and every horizon.

    Args:
        request: What is being decoded.
        config: The configuration that cell was trained under.
        mode: Which view to take.
        horizon: Steps the episode must cover.

    Returns:
        The episode with its provenance.

    Raises:
        FileNotFoundError: If the dataset holds no shards or no split.
        RuntimeError: If no held-out episode covers the longest horizon.
        ValueError: If the drawn episode is shorter than the horizon.
    """
    directory = dataset_directory(request, config.data_seed)
    indices = test_split_indices(directory)
    trajectories = TrajectoryStore().read_dataset(
        directory, "the intermediate-state decoder"
    )
    eligible = eligible_indices(
        trajectories, indices, LONGEST_EVALUATION_HORIZON
    )
    if not eligible:
        raise RuntimeError(
            f"no held-out episode covers {LONGEST_EVALUATION_HORIZON} steps "
            f"in {safe_rel(directory)}"
        )
    draw_seed = int(config.seed)
    position = int(
        jax.random.randint(
            jax.random.PRNGKey(draw_seed), (), 0, len(eligible)
        )
    )
    chosen = eligible[position]
    episode = trajectories[chosen]
    if len(episode) < horizon:
        raise ValueError(
            f"held-out episode {chosen} covers {len(episode)} steps against "
            f"the requested {horizon}"
        )
    frames = jnp.asarray(episode.observations[mode])
    return TruthSource(
        observation=frames[:1],
        actions=jnp.asarray(episode.actions)[None, :horizon],
        truth=frames[: horizon + 1],
        source=TRUTH_SOURCE_TEST_TRAJECTORY,
        trajectory_index=chosen,
        draw_seed=draw_seed,
        eligible_count=len(eligible),
    )


def truth_for_horizon(
    request: DecodeRequest,
    config: ExperimentConfig,
    mode: ObservationMode,
    horizon: int,
) -> TruthSource:
    """Return one cell's truth series from the requested source.

    Falls back to a fresh roll when the held-out shards or split are absent,
    and records the source used.

    Args:
        request: What is being decoded.
        config: The configuration that cell was trained under.
        mode: Which view to take.
        horizon: Steps the episode must cover.

    Returns:
        The episode with its provenance.
    """
    if request.truth_source == TRUTH_SOURCE_TEST_TRAJECTORY:
        try:
            return draw_test_episode(request, config, mode, horizon)
        except FileNotFoundError as absent:
            logger.warning(
                "held-out truth unavailable (%s), falling back to a fresh "
                "roll", absent
            )
    observation, actions, truth = episode_for_horizon(config, mode, horizon)
    return TruthSource(
        observation=observation,
        actions=actions,
        truth=truth,
        source=TRUTH_SOURCE_FRESH_ROLLOUT,
    )


def summary_statistics(
    records: Sequence[dict], domain: ObservationDomain
) -> dict[str, Any]:
    """Reduce one arm's per-step records to its reported statistics.

    Both step-distance statistics run over steps 1 to h. Step 0 is excluded
    because each series returns [0.0] and appends, so it is a definitional
    zero. Step 1 is kept and named separately: it is the first autoregressive
    application and is an outlier the mean alone would bury. Median and mean
    are reported together for the same reason.

    The distinct count is the number of distinct distance-from-truth values, a
    lower bound on distinct states, and is None on a continuous domain.

    Args:
        records: One cell's records, in any order.
        domain: The decode domain of the representation.

    Returns:
        The record-derived summary fields.
    """
    ordered = sorted(records, key=lambda record: record[RECORD_STEP_KEY])
    distances = [record[RECORD_TOKEN_DISTANCE_KEY] for record in ordered]
    changes = [record[RECORD_DECODED_DISTANCE_KEY] for record in ordered]
    differing = [record[RECORD_TRUTH_DISTANCE_KEY] for record in ordered]
    after_start = distances[1:]
    changes_after_start = changes[1:]
    return {
        SUMMARY_DISTINCT_KEY: (
            None if domain.is_continuous() else len(set(differing))
        ),
        SUMMARY_TOKEN_MEDIAN_KEY: float(statistics.median(after_start)),
        SUMMARY_TOKEN_MEAN_KEY: float(statistics.fmean(after_start)),
        SUMMARY_TOKEN_STEP_1_KEY: float(after_start[0]),
        SUMMARY_DECODED_MEDIAN_KEY: float(
            statistics.median(changes_after_start)
        ),
        SUMMARY_DECODED_MEAN_KEY: float(statistics.fmean(changes_after_start)),
        SUMMARY_ENDPOINT_DIFFERING_KEY: differing[-1],
        SUMMARY_STEPS_KEY: len(ordered),
        SUMMARY_VALID_ALL_KEY: all(
            record[RECORD_VALID_KEY] for record in ordered
        ),
    }


def endpoint_per_cell_accuracy(
    decoded: jax.Array, truth: jax.Array
) -> float:
    """Score a decoded grid the way `per_cell_accuracy` scores logits.

    Over cells and channels, as `per_cell_accuracy` reduces, so it is not one
    minus the any-channel differing fraction.

    Args:
        decoded: The decoded classes, (height, width, channels).
        truth: The true state, the same shape.

    Returns:
        The fraction of cell-channels that match.
    """
    return float(np.mean(np.asarray(decoded) == np.asarray(truth)))


def reported_per_cell_accuracy(
    request: DecodeRequest,
    seed: int,
    mode: ObservationMode,
    arm: int,
    horizon: int,
) -> float | None:
    """Read the evaluation's own figure for one cell, where one exists.

    Read only at horizons in `EVALUATION_HORIZONS`. The reported block also
    carries the extrapolation sweep's longer horizons, and pairing one of those
    with an episode drawn from the shorter eligible set compares two different
    things.

    Read from the named run, which is the validation split unless the run name
    carries the test suffix.

    Args:
        request: What is being decoded.
        seed: The reporting seed.
        mode: The observation mode the arm was trained on.
        arm: The arm scored.
        horizon: The horizon rolled to.

    Returns:
        The reported accuracy, or None where no comparable figure exists.
    """
    if horizon not in EVALUATION_HORIZONS:
        return None
    directory = eval_dir(
        request.run_name, seed, request.fast, request.env_name, arm=arm
    )
    if request.checkpoint_root is not None:
        directory = request.checkpoint_root / directory.relative_to(OUTPUTS_DIR)
    path = directory / METRICS_TEMPLATE.format(mode=mode.value)
    if not path.is_file():
        logger.warning("no reported metrics at %s", safe_rel(path))
        return None
    written = json.loads(path.read_text(encoding="utf-8"))
    block = written.get("per_horizon", {}).get(str(horizon))
    if block is None:
        return None
    return block.get("per_cell_accuracy")


def _draw_arm_row(
    axes, grids: np.ndarray, row: tuple[int, int, int], *, label_columns: bool
) -> None:
    """Draw one arm's whole rollout for one channel, one column per step.

    Args:
        axes: One axes per step, in order.
        grids: That arm's decoded grids, leading axis step.
        row: The arm, the channel drawn, and that channel's class count. The
            class count fixes the colour scale, so both arms are drawn on one.
        label_columns: True to write the step index above each column. Set on
            the top row alone, where the numbers are readable.
    """
    arm, channel, limit = row
    for step, axis in enumerate(axes):
        axis.imshow(
            grids[step][0][..., channel], cmap="gray", vmin=0,
            vmax=max(limit - 1, 1), interpolation="nearest",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        if label_columns:
            axis.set_title(str(step), fontsize=STRIP_LABEL_FONTSIZE, pad=2)
    label = CHANNEL_LABELS.get(str(channel), f"channel {channel}")
    axes[0].set_ylabel(
        f"arm {arm}\n{label}", fontsize=STRIP_ROW_LABEL_FONTSIZE,
        rotation=0, ha="right", va="center", labelpad=4,
    )


def _draw_pixel_row(
    axes, frames: np.ndarray, arm: int, *, label_columns: bool
) -> None:
    """Draw one arm's whole rollout as images, one column per step.

    Args:
        axes: One axes per step, in order.
        frames: That arm's decoded states, leading axis step, values on the
            decoder's normalised scale.
        arm: The arm drawn, for the row label.
        label_columns: True to write the step index above each column.
    """
    low, high = DECODED_VALUE_BOUNDS
    for step, axis in enumerate(axes):
        axis.imshow(
            np.clip(frames[step][0], low, high), vmin=low, vmax=high,
            interpolation="nearest",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        if label_columns:
            axis.set_title(str(step), fontsize=STRIP_LABEL_FONTSIZE, pad=2)
    axes[0].set_ylabel(
        f"arm {arm}", fontsize=STRIP_ROW_LABEL_FONTSIZE,
        rotation=0, ha="right", va="center", labelpad=4,
    )


def draw_intermediate_strip(
    traces: dict[int, RolloutTrace],
    domain: ObservationDomain,
    target: Path,
    fields: dict[str, Any],
) -> Path:
    """Draw both arms' rollouts as one strip, arm above arm.

    One column per step, from the start state through every intermediate to the
    endpoint. A discrete domain gives one row per arm and channel, each on its
    channel's own colour scale; a continuous one gives one image row per arm.
    Both arms share a scale either way, so a difference in the picture is a
    difference in the states.

    Args:
        traces: Arm number to its reconstructed rollout.
        domain: The decode domain of the representation.
        target: Directory to write into.
        fields: Identifying fields, naming the file and titling the figure.

    Returns:
        The path written.
    """
    continuous = domain.is_continuous()
    channels = (1,) if continuous else tuple(range(len(domain.cardinality)))
    rows = [(arm, channel) for arm in sorted(traces) for channel in channels]
    steps = traces[rows[0][0]].grids.shape[0]
    figure, axes = plt.subplots(
        len(rows),
        steps,
        figsize=(steps * STRIP_COLUMN_WIDTH, len(rows) * STRIP_ROW_HEIGHT),
        squeeze=False,
    )
    for row, (arm, channel) in enumerate(rows):
        if continuous:
            _draw_pixel_row(
                axes[row],
                np.asarray(traces[arm].grids),
                arm,
                label_columns=row == 0,
            )
            continue
        _draw_arm_row(
            axes[row],
            np.asarray(traces[arm].grids),
            (arm, channel, domain.cardinality[channel]),
            label_columns=row == 0,
        )
    figure.suptitle(
        f"Decoded intermediate states, {fields['path']} rollout -- "
        f"{fields['env']} seed {fields['seed']} {fields['mode']}, "
        f"h = {fields['horizon']}",
        fontsize=8,
    )
    figure.tight_layout()
    path = target / INTERMEDIATE_STATES_FIGURE_TEMPLATE.format(**fields)
    figure.savefig(path, dpi=FIGURE_DPI, metadata=FIGURE_METADATA)
    plt.close(figure)
    logger.info("wrote intermediate-state strip -> %s", safe_rel(path))
    return path


def trace_arms(
    request: DecodeRequest,
    config: ExperimentConfig,
    mode: ObservationMode,
    inputs: tuple[jax.Array, jax.Array],
    domain: ObservationDomain,
) -> dict[int, RolloutTrace]:
    """Reconstruct the requested rollout for every arm that has one.

    Args:
        request: What is being decoded.
        config: The configuration those checkpoints were trained under.
        mode: The observation mode the arms were trained on.
        inputs: The start observation and the action sequence, both carrying a
            batch axis.
        domain: The decode domain of the representation.

    Returns:
        Arm number to its reconstructed rollout.
    """
    observation, actions = inputs
    traces: dict[int, RolloutTrace] = {}
    for arm in INTERMEDIATE_STATE_ARMS:
        model, params = load_arm(request, config, mode, arm)
        traces[arm] = trace_rollout(
            model, params, observation, actions,
            rollout_path=request.rollout_path,
            domain=domain,
        )
    return traces


def cell_fields(
    request: DecodeRequest,
    seed: int,
    mode: ObservationMode,
    horizon: int,
) -> dict[str, Any]:
    """Return the fields identifying one cell.

    One definition, used to name the figure, title it and stamp every record,
    so a file name and the rows inside it cannot disagree about what was
    decoded.

    Args:
        request: What is being decoded.
        seed: The reporting seed.
        mode: The observation mode the arms were trained on.
        horizon: The horizon rolled to.

    Returns:
        The environment, seed, mode, rollout path and horizon.
    """
    return {
        "env": request.env_name,
        "seed": seed,
        "mode": mode.value,
        "path": request.rollout_path,
        "horizon": horizon,
    }


def decode_cell(  # pylint: disable=too-many-locals
    request: DecodeRequest,
    seed: int,
    mode: ObservationMode,
    horizon: int,
    target: Path,
) -> tuple[list[dict], list[dict]]:
    """Decode both arms for one seed, mode and horizon, and draw the pair.

    Args:
        request: What is being decoded.
        seed: The reporting seed.
        mode: The observation mode the arms were trained on.
        horizon: The horizon to roll to.
        target: Directory the figure is written into.

    Returns:
        One record per arm and step, and one summary row per arm.
    """
    config = config_for(request, seed)
    episode = truth_for_horizon(request, config, mode, horizon)
    domain = domain_for(config)
    fields = cell_fields(request, seed, mode, horizon)
    traces = trace_arms(
        request, config, mode, (episode.observation, episode.actions), domain
    )
    records: list[dict] = []
    summaries: list[dict] = []
    for arm, trace in traces.items():
        arm_fields = {**fields, "arm": arm}
        arm_records = trace_records(
            trace, episode.truth, domain, arm_fields
        )
        records.extend(arm_records)
        summaries.append(
            {
                **arm_fields,
                **summary_statistics(arm_records, domain),
                **_decoder_validation(
                    request, seed, mode, arm, horizon, trace, episode, domain
                ),
                **episode.provenance(),
            }
        )
    draw_intermediate_strip(traces, domain, target, fields)
    return records, summaries


def _decoder_validation(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    request: DecodeRequest,
    seed: int,
    mode: ObservationMode,
    arm: int,
    horizon: int,
    trace: RolloutTrace,
    episode: TruthSource,
    domain: ObservationDomain,
) -> dict[str, Any]:
    """Score the decoded endpoint against the evaluation's own figure.

    A smoke test for gross disagreement, not an equality claim. One episode
    against a many-window aggregate is not an equality claim whatever number is
    attached, so no tolerance is stated: none has been measured. It
    catches a decoder wired to the wrong parameter tree or horizon, and
    does not catch small systematic error.

    Undefined on a continuous domain, where the evaluation reports no per-cell
    accuracy to compare against.

    Args:
        request: What is being decoded.
        seed: The reporting seed.
        mode: The observation mode the arm was trained on.
        arm: The arm scored.
        horizon: The horizon rolled to.
        trace: That arm's rollout.
        episode: The truth series it was measured against.
        domain: The decode domain of the representation.

    Returns:
        The decoded accuracy, the reported one and their difference.
    """
    if domain.is_continuous():
        return {
            SUMMARY_ENDPOINT_ACCURACY_KEY: None,
            SUMMARY_REPORTED_ACCURACY_KEY: None,
            SUMMARY_VALIDATION_DELTA_KEY: None,
        }
    decoded = endpoint_per_cell_accuracy(
        trace.grids[-1][0], episode.truth[-1]
    )
    reported = reported_per_cell_accuracy(request, seed, mode, arm, horizon)
    return {
        SUMMARY_ENDPOINT_ACCURACY_KEY: decoded,
        SUMMARY_REPORTED_ACCURACY_KEY: reported,
        SUMMARY_VALIDATION_DELTA_KEY: (
            None if reported is None else decoded - reported
        ),
    }


def _record_key(record: dict, fields: Sequence[str]) -> tuple:
    """Return the tuple identifying one row."""
    return tuple(record.get(field) for field in fields)


def append_records(path: Path, records: Iterable[dict]) -> None:
    """Append records as JSON lines, refusing a repeated key.

    The key omits the rollout path and truth source, so a second path or source
    into the same file is refused too. A refusal ends the pass before the
    summary is written.

    Args:
        path: The records file.
        records: The records to append.

    Raises:
        ValueError: If a record repeats a key already in the file or in this
            batch.
    """
    seen: set[tuple] = set()
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    seen.add(
                        _record_key(json.loads(line), RECORD_KEY_FIELDS)
                    )
    pending: list[dict] = []
    for record in records:
        key = _record_key(record, RECORD_KEY_FIELDS)
        if key in seen:
            raise ValueError(
                f"record {dict(zip(RECORD_KEY_FIELDS, key))} is already in "
                f"{safe_rel(path)}. Delete the file to re-decode a cell."
            )
        seen.add(key)
        pending.append(record)
    with path.open("a", encoding="utf-8") as handle:
        for record in pending:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _write_summary(path: Path, summaries: Sequence[dict]) -> None:
    """Write the summary rows, replacing any row with the same key.

    Rewritten whole. A row with the same seed, mode, arm and horizon is
    replaced, whatever its rollout path or truth source.

    Args:
        path: The summary file.
        summaries: Every summary row produced this pass.
    """
    existing: dict[tuple, dict] = {}
    if path.is_file():
        for row in json.loads(path.read_text(encoding="utf-8")):
            existing[_record_key(row, SUMMARY_KEY_FIELDS)] = row
    for row in summaries:
        existing[_record_key(row, SUMMARY_KEY_FIELDS)] = row
    ordered = [existing[key] for key in sorted(existing, key=str)]
    path.write_text(
        json.dumps(ordered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_intermediate_states_cli(
    request: DecodeRequest,
    seeds: Sequence[int],
    modes: Sequence[ObservationMode],
    horizons: Sequence[int],
) -> Path:
    """Decode every requested cell and write the strips and the records.

    Any failure inside a cell is logged and skipped, but a repeated record ends
    the pass.

    Args:
        request: What is being decoded.
        seeds: Reporting seeds to decode.
        modes: Observation modes to decode.
        horizons: Horizons to roll to.

    Returns:
        The records path written.
    """
    target = ensure_dir(
        intermediate_states_dir(request.run_name, request.fast)
    )
    records_path = target / INTERMEDIATE_STATES_RECORDS_FILENAME
    summary_path = target / INTERMEDIATE_STATES_SUMMARY_FILENAME
    logger.info(
        "intermediate states: run %s, %s rollout, %s truth, seeds %s, "
        "modes %s, horizons %s, backend %s",
        request.run_name,
        request.rollout_path,
        request.truth_source,
        tuple(seeds),
        tuple(mode.value for mode in modes),
        tuple(horizons),
        jax.default_backend(),
    )
    produced: list[dict] = []
    for seed in seeds:
        for mode in modes:
            for horizon in horizons:
                try:
                    records, summaries = decode_cell(
                        request, seed, mode, horizon, target
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "FAILED seed %d %s h=%d: %s: %s",
                        seed, mode.value, horizon, type(exc).__name__,
                        str(exc).splitlines()[0][:120],
                    )
                    continue
                append_records(records_path, records)
                produced.extend(summaries)
    if produced:
        _write_summary(summary_path, produced)
        logger.info("wrote %s", safe_rel(summary_path))
    logger.info("wrote %s", safe_rel(records_path))
    return records_path
