"""Training stage. Trains whichever arm it is built for.

Arms 2 and 3 share an architecture and differ only in the horizons their
batches are drawn at, arm 2 over the sampled range and arm 3 at h = 1.
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import contextlib
import json
import math
import signal
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from config import (
    ARM_AR_ONE_STEP,
    ARM_DIRECT,
    ARMS,
    CONTINUOUS_VALUE_RANGES,
    ENV_FAMILY_NAVIX,
    ExperimentConfig,
    SamplerMode,
    assert_config_resolved,
    checkpoints_dir,
    env_family,
    offline_source_for_env,
)
from src.data.offline_sources import build_offline_source
from src.data.split import (
    SplitName,
    TrajectorySplit,
    split_from_config,
)
from src.data.trajectory import ObservationMode, Trajectory
from src.data.trajectory_source import NavixTrajectorySource
from src.data.trajectory_store import TrajectoryStore
from src.data.window_sampler import WindowBatch, WindowSampler
from src.utils.logging_setup import format_duration
from src.models.ar_baseline import (
    OBJECTIVE_ENDPOINT,
    AutoregressiveBaseline,
    ar_baseline_for_spec,
    objective_for_arm,
)
from src.models.jumpy_transformer import (
    jumpy_transformer_for_spec,
    predict_endpoint,
)
from src.models.losses import (
    check_observation_values_in_bounds,
    check_observation_values_in_range,
    pixel_reconstruction_loss,
    reconstruction_loss,
)
from src.pipeline.base import ArmScopedStage
from src.utils.logging_setup import get_logger
from src.utils.optimisers import build_optimiser
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

# Payload version. Orbax matches on structure, so without it wrong weights load
# and nothing raises.
CHECKPOINT_SCHEMA_VERSION: int = 2

# Keys of the saved pytree.
CHECKPOINT_PARAMS_KEY: str = "params"
CHECKPOINT_OPT_STATE_KEY: str = "opt_state"
CHECKPOINT_STEP_KEY: str = "step"
CHECKPOINT_SCHEMA_KEY: str = "schema_version"
# Folded at an index no training step can reach, so the validation batch cannot
# collide with a training draw.
VALIDATION_BATCH_FOLD: int = 10**9

# Best-model tracking, separate from `params`, which a resume continues from.
CHECKPOINT_BEST_PARAMS_KEY: str = "best_params"
CHECKPOINT_BEST_LOSS_KEY: str = "best_loss"
CHECKPOINT_BEST_STEP_KEY: str = "best_step"
# The budget this checkpoint was trained under. The schedule anneals over it.
CHECKPOINT_TOTAL_STEPS_KEY: str = "total_steps"

# Parameter-count summary. The mode is in the name; the content differs by mode.
PARAMETER_COUNT_TEMPLATE: str = "parameter_count_{mode}.json"

# The horizon arm 3's one-step objective trains at.
ONE_STEP_HORIZON: int = 1


@dataclass(frozen=True)
class TrainedState:
    """Everything one training run carries between steps and into the payload.

    Attributes:
        params: The last step's parameters. The resume point.
        opt_state: The last step's optimiser state, describing `params` alone.
        step: Gradient steps completed.
        best_params: Parameters from the lowest validation loss observed,
            sampled at the logging interval.
        best_loss: The validation loss at those parameters.
        best_step: The step it came from.
        total_steps: The budget this run was scheduled for.
    """

    params: dict
    opt_state: object
    step: int
    best_params: dict
    best_loss: float
    best_step: int
    total_steps: int


@dataclass
class IntervalTimer:
    """Wall-clock bookkeeping for the training loop's progress lines.

    Only `split` resets the interval.

    Attributes:
        run_started_at: `perf_counter` at the first step of this invocation.
        interval_started_at: `perf_counter` at the last progress line.
        steps: Steps taken since that line.
    """

    run_started_at: float
    interval_started_at: float
    steps: int = 0

    @classmethod
    def started(cls) -> IntervalTimer:
        """Return a timer running from now.

        Returns:
            A timer whose run and interval both start at this instant.
        """
        now = time.perf_counter()
        return cls(run_started_at=now, interval_started_at=now)

    def split(self) -> tuple[float, float, int]:
        """Close the current interval and open the next one.

        Returns:
            Seconds since the invocation started, seconds in the interval just
            closed, and the number of steps it covered.
        """
        now = time.perf_counter()
        interval = now - self.interval_started_at
        steps = self.steps
        self.interval_started_at = now
        self.steps = 0
        return now - self.run_started_at, interval, steps

    def elapsed(self) -> float:
        """Return seconds since this invocation started.

        Returns:
            Wall-clock seconds.
        """
        return time.perf_counter() - self.run_started_at


def checkpoint_template(state: TrainedState) -> dict:
    """Return the checkpoint payload for a state, used for both save and restore.

    Orbax restores into whatever template it is given, so a second payload
    that drifts by a key loads plausible wrong weights.

    Args:
        state: The state to persist, or one of the right structure to restore
            into.

    Returns:
        The JAX pytree written to and read from the checkpoint directory.
    """
    return {
        CHECKPOINT_PARAMS_KEY: state.params,
        CHECKPOINT_OPT_STATE_KEY: state.opt_state,
        CHECKPOINT_STEP_KEY: jnp.asarray(state.step, dtype=jnp.int32),
        CHECKPOINT_SCHEMA_KEY: jnp.asarray(
            CHECKPOINT_SCHEMA_VERSION, dtype=jnp.int32
        ),
        CHECKPOINT_BEST_PARAMS_KEY: state.best_params,
        CHECKPOINT_BEST_LOSS_KEY: jnp.asarray(
            state.best_loss, dtype=jnp.float32
        ),
        CHECKPOINT_BEST_STEP_KEY: jnp.asarray(state.best_step, dtype=jnp.int32),
        CHECKPOINT_TOTAL_STEPS_KEY: jnp.asarray(
            state.total_steps, dtype=jnp.int32
        ),
    }


def restore_target(params: dict, opt_state, total_steps: int) -> dict:
    """Return the template a restore reads a checkpoint into.

    Args:
        params: A freshly initialised parameter tree of the right shape.
        opt_state: An optimiser state built from that tree.
        total_steps: The budget placed in the template. Nothing here compares
            it with the checkpoint's.

    Returns:
        The restore template.
    """
    return checkpoint_template(
        TrainedState(
            params=params,
            opt_state=opt_state,
            step=0,
            best_params=params,
            best_loss=0.0,
            best_step=0,
            total_steps=total_steps,
        )
    )


def count_parameters(params: dict) -> int:
    """Return the total number of scalar parameters in a pytree.

    Args:
        params: A Flax parameter tree.

    Returns:
        The summed size of every leaf.
    """
    return int(sum(leaf.size for leaf in jax.tree.leaves(params)))


def save_checkpoint(directory: Path, state: dict) -> None:
    """Write the payload and wait for the asynchronous save to finish.

    `force=True` overwrites the only copy, so an interrupted write leaves no
    checkpoint to fall back to.

    Args:
        directory: Where the checkpoint tree is written. Its parent is created
            if absent.
        state: The payload from `checkpoint_template`.
    """
    ensure_dir(directory.parent)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(directory, state, force=True)
    checkpointer.wait_until_finished()
    logger.info("wrote checkpoint -> %s", safe_rel(directory))


def restore_checkpoint(directory: Path, target: dict) -> dict:
    """Read a checkpoint back, given a template of the expected shape.

    A restored scalar comes back as a JAX array. Callers convert explicitly.

    Args:
        directory: The checkpoint tree written by `save_checkpoint`.
        target: A pytree of the expected structure, used as the restore
            template.

    Returns:
        The restored payload.
    """
    checkpointer = ocp.StandardCheckpointer()
    return checkpointer.restore(directory, target=target)


def build_training_source(config: ExperimentConfig):
    """Return the trajectory source the configured environment is described by.

    The model builder reads only its spec. An offline environment resolves its
    corpus through `offline_source_for_env`, as the offline generate stage
    does. Building an Atari source needs the archive on disk, so training or
    evaluating an Atari model does too.

    Args:
        config: The composed experiment configuration.

    Returns:
        A source answering spec(observation_mode).

    Raises:
        ValueError: If the environment matches no family prefix, or its family
            registers no offline corpus.
        FileNotFoundError: If an Atari archive shard is absent.
    """
    family = env_family(config.env.name)
    if family == ENV_FAMILY_NAVIX:
        return NavixTrajectorySource(
            config.env,
            representation=config.data.representation,
            stored_modes=config.data.modes_stored(),
        )
    source, _ = build_offline_source(offline_source_for_env(config.env.name))
    return source


def build_arm(
    config: ExperimentConfig, observation_mode: ObservationMode, *, arm: int
):
    """Construct one arm for one observation mode and initialise its parameters.

    Every arm is built from the same ModelConfig. Encoder depth follows the
    configured extent, not the observed grid, so every mode shares it. The
    evaluator rebuilds the model through this function to restore a checkpoint.

    Args:
        config: The composed experiment configuration.
        observation_mode: The single mode this run predicts.
        arm: One of config.ARMS.

    Returns:
        The unbound model and its initialised parameter tree.

    Raises:
        ValueError: If the arm is not one of config.ARMS, or if the
            configuration's observation contract is unresolved.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    assert_config_resolved(config)
    source = build_training_source(config)
    spec = source.spec(observation_mode)
    builder = (
        jumpy_transformer_for_spec if arm == ARM_DIRECT else ar_baseline_for_spec
    )
    model = builder(
        spec, config.model, depth_extent=config.env.max_grid_extent
    )
    field = spec.single_field()
    height, width = field.shape
    num_channels = config.model.obs_channels
    init_key = jax.random.PRNGKey(config.model_seed)
    params_key, dropout_key = jax.random.split(init_key)
    params = model.init(
        {"params": params_key, "dropout": dropout_key},
        jnp.zeros((1, height, width, num_channels), jnp.int32),
        jnp.zeros((1, config.train.horizon_max), jnp.int32),
        jnp.ones((1,), jnp.int32),
        deterministic=True,
    )["params"]
    logger.info(
        "arm %d built: %s mode, predicted grid %dx%d, encoder depth from "
        "extent %d, %d parameters",
        arm,
        observation_mode.value,
        height,
        width,
        config.env.max_grid_extent,
        count_parameters(params),
    )
    return model, params


class TrainStage(ArmScopedStage):
    """Train one arm on real data.

    Reads this dataset's shards, re-derives its partition, draws training
    windows, steps the optimiser for the configured budget, decodes one
    demonstration prediction and writes the final checkpoint. Its checkpoint,
    metrics and sentinel directories are scoped by model seed and arm.

    Attributes:
        store: The trajectory store this stage reads through.
        arm: Which arm this instance trains, one of config.ARMS.
    """

    name: str = "train"
    dataset_derived: bool = False

    @property
    def checkpoint_dir(self) -> Path:
        """Return this run's checkpoint directory.

        Keyed on the model seed, the observation mode and the arm.
        """
        return checkpoints_dir(
            self.config.run_name,
            self.artefact_seed,
            self.config.fast,
            self.config.env.name,
            observation_mode=self.config.sampler.observation_mode,
            arm=self.arm,
        )

    def __init__(
        self, config: ExperimentConfig, *, arm: int = ARM_DIRECT
    ) -> None:
        """Build the stage.

        Args:
            config: The composed experiment configuration.
            arm: Which arm to train, one of config.ARMS.
        """
        super().__init__(config, arm=arm)
        self.store = TrajectoryStore()
        # Set by _train, read by sentinel_identity. None means training has not
        # run here, and the identity then reports the full budget.
        self._completed_steps: int | None = None
        self._stop_requested: bool = False
        # Built in _train once the model exists.
        self._validation_step: Callable | None = None

    @property
    def objective(self) -> str:
        """Return the training objective this arm is trained under.

        Returns:
            "endpoint" for arms 1 and 2, "one_step" for arm 3. Arm 1 is named
            here; `objective_for_arm` refuses it.
        """
        if self.arm == ARM_DIRECT:
            return OBJECTIVE_ENDPOINT
        return objective_for_arm(self.arm)

    @property
    def training_horizon_max(self) -> int:
        """Return the largest horizon this arm's training batches are drawn at.

        Arm 3's one-step objective is a horizon distribution, not a different
        loss, so h = 1 here is the objective.

        Returns:
            1 for arm 3, otherwise `config.train.horizon_max`.
        """
        if self.arm == ARM_AR_ONE_STEP:
            return ONE_STEP_HORIZON
        return self.config.train.horizon_max

    @property
    def training_config(self) -> ExperimentConfig:
        """Return the configuration this arm's training sampler is built from.

        A local copy, never written back to `self.config`, which every artefact
        records. Arm 3's copy is narrowed to h = 1.

        Returns:
            `self.config` for arms 1 and 2, and a copy narrowed to h = 1 for
            arm 3.

        Raises:
            ValueError: If arm 3 is run under SamplerMode.OFFLINE, whose pool
                file cannot tell arm 3's pool from the mixed-horizon one.
        """
        if self.arm != ARM_AR_ONE_STEP:
            return self.config
        if self.config.sampler.mode is SamplerMode.OFFLINE:
            raise ValueError(
                "arm 3 cannot train under SamplerMode.OFFLINE. The pool path "
                "carries only the split and the observation mode, so its "
                "one-step pool and the mixed-horizon pool resolve to one file "
                "and overwrite one another. Run arm 3 under HYBRID."
            )
        return replace(
            self.config,
            train=replace(
                self.config.train,
                horizon_min=ONE_STEP_HORIZON,
                horizon_max=ONE_STEP_HORIZON,
            ),
        )

    def run(self) -> None:
        """Load, partition, train, decode, and checkpoint."""

        trajectories = self._load_trajectories()
        split = self._partition(trajectories)
        sampler = WindowSampler.from_config(
            self.training_config, trajectories, split, SplitName.TRAIN
        )
        # From `training_config`, so arm 3's validation batch is at h = 1 too.
        validation_batch = WindowSampler.from_config(
            self.training_config, trajectories, split, SplitName.VALIDATION
        ).next_batch(
            jax.random.fold_in(
                jax.random.PRNGKey(self.config.model_seed),
                VALIDATION_BATCH_FOLD,
            )
        )
        model, params = self._build_model()
        with self._graceful_stop():
            state = self._train(model, params, sampler, validation_batch)
        self._decode_one_prediction(model, state.params, sampler)
        self._write_checkpoint(state)

    @contextlib.contextmanager
    def _graceful_stop(self) -> Iterator[None]:
        """Install a SIGTERM/SIGINT handler for the duration of training.

        Yields:
            None, with the handlers installed.
        """
        previous_term = signal.signal(
            signal.SIGTERM, self._request_graceful_stop
        )
        previous_int = signal.signal(signal.SIGINT, self._request_graceful_stop)
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)

    def _request_graceful_stop(self, signum: int, frame: object) -> None:
        """Ask the training loop to stop after the current step.

        Only sets a flag, so the handler never raises inside a compiled step.

        Args:
            signum: The delivered signal number.
            frame: The interrupted stack frame, unused.
        """
        del frame
        logger.warning(
            "signal %d received -- checkpointing after this step, then "
            "stopping. Re-run to resume",
            signum,
        )
        self._stop_requested = True

    def _load_trajectories(self) -> list[Trajectory]:
        """Read every shard this data seed's dataset holds.

        Returns:
            All complete episodes, in shard order.

        Raises:
            FileNotFoundError: If the dataset directory holds no shards.
        """
        return self.store.read_dataset(self.dataset_dir, "training")

    def _partition(self, trajectories: list[Trajectory]) -> TrajectorySplit:
        """Re-derive this dataset's partition by trajectory index.

        Uses the preparation stage's call. The recorded split file is not read
        or compared, so shards regenerated in a different order are split
        differently from the recorded partition, and nothing raises.

        Args:
            trajectories: Every episode in this dataset, in store order.

        Returns:
            The partition, identical to the recorded one while the shards are
            unchanged.
        """
        return split_from_config(
            self.config, [len(trajectory) for trajectory in trajectories]
        )

    def _build_model(self):
        """Construct this stage's arm and initialise its parameters.

        Returns:
            The unbound model and its initialised parameter tree.
        """
        return build_arm(self.config, self.observation_mode, arm=self.arm)

    @property
    def value_range(self) -> tuple[int, int] | None:
        """Return the stored-value bounds this arm trains against.

        Read from CONTINUOUS_VALUE_RANGES, as the generate and evaluate stages
        read it.

        Returns:
            The bounds for a continuous representation, None for a discrete
            one, which selects the categorical loss instead.
        """
        return CONTINUOUS_VALUE_RANGES.get(self.config.data.representation)

    @property
    def _training_method(self):
        """Return the model method this arm's loss is computed through.

        Arm 1 uses `__call__`, its one-shot forward pass. Arms 2 and 3 use
        `rollout_tokens`, the differentiable token-space rollout, their own
        `__call__` carrying no gradient. Every arm is scored through
        `__call__`, so only training differs.

        Returns:
            None for arm 1, so `model.apply` uses `__call__`, otherwise
            `AutoregressiveBaseline.rollout_tokens`.
        """
        if self.arm == ARM_DIRECT:
            return None
        return AutoregressiveBaseline.rollout_tokens

    def _loss(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        model,
        params,
        states,
        actions,
        horizons,
        targets,
        dropout_key,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        """Return the mean endpoint reconstruction loss over one batch.

        The endpoint is the only supervised state, and one loss serves every
        arm: categorical cross-entropy for a discrete representation, squared
        error for a continuous one. Arm 3's one-step objective is this loss on
        batches drawn at h = 1. Mean over the batch and summed over cells and
        channels, so the value scales with grid size.

        Args:
            model: The unbound transformer.
            params: Its parameter tree.
            states: Start observations s_t.
            actions: Padded action indices. Its width is the rollout length
                for arms 2 and 3. A compiled scan's length is static and
                follows this array's shape, not `horizons`.
            horizons: True horizon per example, used for the attention mask.
            targets: End observations s_{t+h}.
            dropout_key: PRNG key for dropout, fresh per step. Passed even
                when `deterministic` is True, keeping one call shape.
            deterministic: True turns dropout off, as the validation loss does.

        Returns:
            Scalar loss.
        """
        logits = model.apply(
            {"params": params},
            states,
            actions,
            horizons,
            deterministic=deterministic,
            rngs={"dropout": dropout_key},
            method=self._training_method,
        )
        flat = targets.reshape(targets.shape[0], -1)
        if self.value_range is not None:
            # The decoder returns a list for every representation. A continuous
            # prediction is its single element, whose last axis is channels
            # rather than classes.
            return jnp.mean(
                pixel_reconstruction_loss(
                    logits[0], flat, targets.shape[1:3], self.value_range
                )
            )
        return jnp.mean(
            reconstruction_loss(logits, flat, targets.shape[1:3])
        )

    def _train(  # pylint: disable=too-many-locals
        self,
        model,
        params,
        sampler: WindowSampler,
        validation_batch: WindowBatch,
    ) -> TrainedState:
        """Run the gradient loop and return the trained state.

        The observation-domain check runs on this invocation's first batch.
        Each step's key is folded from the step number, so a resumed run draws
        the same batches as an uninterrupted one. Both losses are taken at the
        pre-update parameters, the best model is sampled at the logging
        interval, and the first timing interval includes compilation.

        Args:
            model: The unbound model for this arm.
            params: Its initialised parameter tree.
            sampler: The train-split sampler. Built from `training_config`, so
                arm 3's batches are h = 1.
            validation_batch: One validation batch, drawn once and reused.

        Returns:
            The trained state, carrying the final parameters, the optimiser
            state, the steps completed, and the best parameters observed.
        """
        optimiser = build_optimiser(self.config.train)
        total = self.config.train.total_steps
        state = self._resume_or_start(
            TrainedState(
                params=params,
                opt_state=optimiser.init(params),
                step=0,
                best_params=params,
                best_loss=math.inf,
                best_step=0,
                total_steps=total,
            )
        )
        compute_step = self._compiled_step(model, optimiser)
        self._validation_step = self._compiled_validation(model)
        base_key = jax.random.PRNGKey(self.config.model_seed)
        started_at = state.step
        timer = IntervalTimer.started()
        for step in range(started_at, total):
            batch, dropout_key = self._draw(sampler, base_key, step)
            if step == started_at:
                self._check_observation_domain(batch)
            new_params, new_opt_state, loss_value = compute_step(
                state.params,
                state.opt_state,
                batch.states,
                batch.actions,
                batch.horizons,
                batch.targets,
                dropout_key,
            )
            timer.steps += 1
            # Before the state advances, so both losses read the pre-update
            # parameters.
            selection_loss = self._maybe_log(
                state, step, loss_value, validation_batch, timer
            )
            state = self._advanced(
                state, new_params, new_opt_state, step, selection_loss
            )
            if self._is_periodic_checkpoint(state.step, total):
                self._write_checkpoint(state)
            if self._stop_requested:
                break

        # sentinel_identity reads this, so a short run writes a sentinel a
        # later invocation will not match.
        self._completed_steps = state.step
        self._log_finished(state, total, timer, started_at)
        return state

    def _log_finished(
        self,
        state: TrainedState,
        total: int,
        timer: IntervalTimer,
        started_at: int,
    ) -> None:
        """Emit the closing line for one training invocation.

        Args:
            state: The final state.
            total: The full step budget.
            timer: The loop's wall-clock bookkeeping.
            started_at: The step this invocation began from.
        """
        elapsed = timer.elapsed()
        steps_taken = state.step - started_at
        logger.info(
            "arm %d seed %d training finished after %d/%d steps, best loss "
            "%.4f at step %d, elapsed %s, mean %.4f s/step over %d steps "
            "this invocation",
            self.arm,
            self.config.model_seed,
            state.step,
            total,
            state.best_loss,
            state.best_step,
            format_duration(elapsed),
            elapsed / steps_taken if steps_taken else 0.0,
            steps_taken,
        )

    def _compiled_step(self, model, optimiser):
        """Return a `jax.jit`-compiled full training step for this arm.

        The batch crosses the boundary as arrays and the optimiser update runs
        inside it. Compiled results differ bitwise from the uncompiled path, so
        a reporting set must come from one code path.

        Args:
            model: The unbound model for this arm.
            optimiser: The optax transform, compiled in with the gradient.

        Returns:
            A compiled callable taking `(params, opt_state, states, actions,
            horizons, targets, dropout_key)` and returning
            `(new_params, new_opt_state, loss)`.
        """

        @jax.jit
        def step(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            params, opt_state, states, actions, horizons, targets, dropout_key
        ):
            loss, grads = jax.value_and_grad(
                lambda tree: self._loss(
                    model,
                    tree,
                    states,
                    actions,
                    horizons,
                    targets,
                    dropout_key,
                )
            )(params)
            updates, opt_state = optimiser.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        return step

    def _compiled_validation(self, model):
        """Return a compiled, dropout-free loss for the validation batch.

        Must share `_loss` with the training step, or the gap stops comparing
        like with like.

        Args:
            model: The unbound model for this arm.

        Returns:
            A compiled callable over the same arrays as `_compiled_step`.
        """

        @jax.jit
        def loss(  # pylint: disable=too-many-arguments,too-many-positional-arguments
            params, states, actions, horizons, targets, dropout_key
        ):
            return self._loss(
                model,
                params,
                states,
                actions,
                horizons,
                targets,
                dropout_key,
                deterministic=True,
            )

        return loss

    def _maybe_log(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        state: TrainedState,
        step: int,
        loss_value,
        validation_batch: WindowBatch,
        timer: IntervalTimer,
    ) -> float | None:
        """Emit this step's progress line if one is due, and return its loss.

        Returns the validation loss, which best-model tracking also needs.
        Both losses are taken at `state.params`, before the update.

        Args:
            state: The state before this step's update.
            step: The zero-based index of the step just taken.
            loss_value: The training loss at `state.params`, still on device.
            validation_batch: The fixed validation batch for this run.
            timer: The loop's wall-clock bookkeeping.

        Returns:
            The validation loss as a host float when this step logged,
            otherwise None.
        """
        total = self.config.train.total_steps
        completed = step + 1
        if not self._logs(completed, total):
            return None
        observed = float(loss_value)
        validation_loss = self._validation_loss(
            state.params, validation_batch
        )
        elapsed_seconds, interval_seconds, interval_steps = timer.split()
        self._log_progress(
            completed=completed,
            total=total,
            training_loss=observed,
            validation_loss=validation_loss,
            interval_seconds=interval_seconds,
            interval_steps=interval_steps,
            elapsed_seconds=elapsed_seconds,
        )
        return validation_loss

    def _logs(self, completed: int, total: int) -> bool:
        """Return whether this step logs.

        Args:
            completed: Gradient steps completed so far.
            total: The full budget.

        Returns:
            True when this step should emit a progress line.
        """
        return (
            completed % self.config.train.log_every_steps == 0
            or completed == total
        )

    def _validation_loss(self, params, batch: WindowBatch) -> float:
        """Return the loss on the fixed validation batch, dropout off.

        Not a reported result, but it selects `best_params`, which evaluation
        scores by default.

        Args:
            params: The parameter tree to score, pre-update.
            batch: The fixed validation batch, drawn once per run.

        Returns:
            The mean endpoint loss on that batch, as a host float.
        """
        return float(
            self._validation_step(
                params,
                batch.states,
                batch.actions,
                batch.horizons,
                batch.targets,
                jax.random.PRNGKey(self.config.model_seed),
            )
        )

    def _log_progress(  # pylint: disable=too-many-arguments
        self,
        *,
        completed: int,
        total: int,
        training_loss: float,
        validation_loss: float,
        interval_seconds: float,
        interval_steps: int,
        elapsed_seconds: float,
    ) -> None:
        """Emit one progress line and the projected finish time.

        The gap is validation minus training. The ETA extrapolates the latest
        interval, and elapsed and ETA cover this invocation only.

        Args:
            completed: Gradient steps completed.
            total: The full budget.
            training_loss: Loss on this step's training batch, dropout on.
            validation_loss: Loss on the fixed validation batch, dropout off.
            interval_seconds: Wall-clock seconds since the last progress line.
            interval_steps: Steps taken inside that interval.
            elapsed_seconds: Wall-clock seconds since this invocation started.
        """
        per_step = interval_seconds / interval_steps if interval_steps else 0.0
        remaining = (total - completed) * per_step
        logger.info(
            "arm %d seed %d step %d/%d loss %.4f eval %.4f gap %+.4f "
            "s/step %.4f elapsed %s eta %s",
            self.arm,
            self.config.model_seed,
            completed,
            total,
            training_loss,
            validation_loss,
            validation_loss - training_loss,
            per_step,
            format_duration(elapsed_seconds),
            format_duration(remaining),
        )

    def _draw(self, sampler: WindowSampler, base_key, step: int):
        """Return this step's training batch and its dropout key.

        Both are folded from the model seed and the step number, so a resumed
        run draws what an uninterrupted one would. GPU training is still not
        bit-identical across runs.

        Args:
            sampler: The train-split sampler.
            base_key: This run's root key, derived from the model seed.
            step: The zero-based index of the step about to be taken.

        Returns:
            The batch, and the dropout key for the same step.
        """
        batch_key, dropout_key = jax.random.split(
            jax.random.fold_in(base_key, step)
        )
        return sampler.training_batch(batch_key), dropout_key

    def _advanced(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        state: TrainedState,
        new_params,
        new_opt_state,
        step: int,
        selection_loss: float | None,
    ) -> TrainedState:
        """Record one completed update, then fold in best-model tracking.

        Selection reads the validation loss at `state.params`, before the
        update.

        Args:
            state: The state before this step's update.
            new_params: The updated parameter tree.
            new_opt_state: The optimiser state after the same update.
            step: The zero-based index of the step just taken.
            selection_loss: The held-out loss at `state.params`, or None when
                this step did not log.

        Returns:
            The state after the update, with the best model updated when this
            step logged and improved on it.
        """
        advanced = replace(
            state,
            params=new_params,
            opt_state=new_opt_state,
            step=step + 1,
        )
        if selection_loss is None or selection_loss >= state.best_loss:
            return advanced
        return replace(
            advanced,
            best_params=state.params,
            best_loss=selection_loss,
            best_step=step,
        )

    def _is_periodic_checkpoint(self, completed: int, total: int) -> bool:
        """Return whether a periodic checkpoint is due at this step.

        The final step is excluded. `run()` writes the closing checkpoint
        itself.

        Args:
            completed: Gradient steps completed so far.
            total: The full budget.

        Returns:
            True when periodic saving is on and this step lands on the
            interval, and the run has not reached its final step.
        """
        every = self.config.train.checkpoint_every_steps
        return every > 0 and completed % every == 0 and completed < total

    def _resume_or_start(self, fresh: TrainedState) -> TrainedState:
        """Continue an interrupted run, or start clean.

        A checkpoint recording a complete run is ignored and training starts
        over. A different schema version or budget raises. Nothing else is
        compared, so an incomplete checkpoint resumes under a changed learning
        rate, batch size or other setting.

        Args:
            fresh: The state a clean run would start from.

        Returns:
            `fresh`, or the restored state when an incomplete checkpoint exists.

        Raises:
            ValueError: If a checkpoint exists at a different schema version or
                under a different training budget.
        """
        directory = self.checkpoint_dir
        if not directory.is_dir():
            return fresh
        restored = restore_checkpoint(directory, checkpoint_template(fresh))
        version = int(restored[CHECKPOINT_SCHEMA_KEY])
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"the checkpoint under {safe_rel(directory)} is schema version "
                f"{version} and this code writes {CHECKPOINT_SCHEMA_VERSION}. "
                "Restoring across versions matches on structure and loads "
                "wrong weights. Delete it to retrain."
            )
        budget = int(restored[CHECKPOINT_TOTAL_STEPS_KEY])
        if budget != self.config.train.total_steps:
            raise ValueError(
                f"the checkpoint under {safe_rel(directory)} was trained "
                f"under a budget of {budget} steps and this run asks for "
                f"{self.config.train.total_steps}. The schedule anneals over "
                "the budget, so resuming replays the completed steps on the "
                "wrong period. Delete it to retrain."
            )
        # int(), explicitly. A restored scalar is a JAX array and this is a
        # loop bound.
        step = int(restored[CHECKPOINT_STEP_KEY])
        if step >= self.config.train.total_steps:
            logger.info(
                "a complete checkpoint (%d steps) is on disk but its "
                "sentinel is gone or stale -- retraining from scratch",
                step,
            )
            return fresh
        logger.info(
            "resuming from step %d/%d <- %s",
            step,
            self.config.train.total_steps,
            safe_rel(directory),
        )
        return TrainedState(
            params=restored[CHECKPOINT_PARAMS_KEY],
            opt_state=restored[CHECKPOINT_OPT_STATE_KEY],
            step=step,
            best_params=restored[CHECKPOINT_BEST_PARAMS_KEY],
            best_loss=float(restored[CHECKPOINT_BEST_LOSS_KEY]),
            best_step=int(restored[CHECKPOINT_BEST_STEP_KEY]),
            total_steps=budget,
        )

    def _check_observation_domain(self, batch: WindowBatch) -> None:
        """Raise if any value in the first batch falls outside its declared domain.

        Class counts bound a discrete observation and stored bounds a
        continuous one. Only this invocation's first batch is checked.

        Args:
            batch: The first training batch drawn this run.

        Raises:
            ValueError: If any value is outside the declared domain.
        """
        flat = batch.targets.reshape(batch.targets.shape[0], -1)
        if self.value_range is not None:
            check_observation_values_in_bounds(
                flat, batch.targets.shape[1:3], self.value_range
            )
            return
        check_observation_values_in_range(
            flat, batch.targets.shape[1:3], self.config.model.obs_channel_classes
        )

    def _decode_one_prediction(self, model, params, sampler: WindowSampler) -> None:
        """Decode one training batch and log the per-channel agreement.

        A log line, never a result. It has no baseline, a copying model scores
        close to one, and on a continuous representation it is meaningless.

        Args:
            model: The unbound transformer.
            params: Its trained parameters.
            sampler: The sampler to draw the demonstration batch from.
        """
        batch = sampler.training_batch(jax.random.PRNGKey(self.config.model_seed))
        logits = predict_endpoint(model, params, batch)
        predictions = [jnp.argmax(channel, axis=-1) for channel in logits]
        matches = [
            float(jnp.mean(prediction == batch.targets[..., channel]))
            for channel, prediction in enumerate(predictions)
        ]
        logger.info(
            "decoded a prediction: %d channels, shape %s per channel, "
            "train-set cell agreement %s",
            len(predictions),
            tuple(int(dim) for dim in predictions[0].shape),
            ", ".join(f"{value:.4f}" for value in matches),
        )

    def _write_checkpoint(self, state: TrainedState) -> None:
        """Save the trained state and record the parameter budget beside it.

        The payload holds both the final and the best parameters. Evaluation
        scores `EvalConfig.params_selection`, the best tree by default, and the
        test pass scores both.

        Args:
            state: The state to persist.
        """
        directory = self.checkpoint_dir
        save_checkpoint(directory, checkpoint_template(state))
        record = directory.parent / PARAMETER_COUNT_TEMPLATE.format(
            mode=self.config.sampler.observation_mode
        )
        record.write_text(
            json.dumps(
                {
                    "arm": self.arm,
                    "objective": self.objective,
                    "observation_mode": self.config.sampler.observation_mode,
                    "parameters": count_parameters(state.params),
                    "depth_extent": self.config.env.max_grid_extent,
                    "training_horizon_max": self.training_horizon_max,
                    "completed_steps": state.step,
                    "best_loss": state.best_loss,
                    "best_step": state.best_step,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("wrote the parameter budget -> %s", safe_rel(record))

    def sentinel_identity(self) -> dict:
        """Return the identity guarding this run's trained model.

        `completed_steps` reports the full budget before training and the steps
        actually taken after it, so an interrupted run's sentinel does not
        match the next invocation. The identity is not exhaustive. Among what
        it omits are `horizon_min`, `dropout_rate`, `remat_rollout`, the
        checkpoint interval and every dataset setting, so a change to one of
        those under the same run name skips training and reuses the old model.

        Returns:
            JSON-serialisable identity fields.
        """
        identity = super().sentinel_identity()
        identity.update(
            {
                "arm": self.arm,
                "objective": self.objective,
                "total_steps": self.config.train.total_steps,
                "completed_steps": (
                    self.config.train.total_steps
                    if self._completed_steps is None
                    else self._completed_steps
                ),
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "batch_size": self.config.train.batch_size,
                "learning_rate": self.config.train.learning_rate,
                "optimiser_name": self.config.train.optimiser_name,
                "lr_schedule_final": self.config.train.lr_schedule_final,
                "grad_clip_norm": self.config.train.grad_clip_norm,
                "horizon_max": self.training_horizon_max,
                "observation_mode": self.config.sampler.observation_mode,
                "sampler_mode": self.config.sampler.mode.value,
                "d_model": self.config.model.d_model,
                "num_layers": self.config.model.num_layers,
                "num_heads": self.config.model.num_heads,
                "code_embed_dim": self.config.model.code_embed_dim,
                "encoder_channels": list(self.config.model.encoder_channels),
                "decoder_channels": list(self.config.model.decoder_channels),
                "state_tokens": self.config.model.state_tokens,
                "depth_extent": self.config.env.max_grid_extent,
            }
        )
        return identity
