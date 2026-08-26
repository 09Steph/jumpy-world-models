"""Training stage. Trains whichever arm it is built for.

The arm is a constructor argument. Arms 2 and 3 share an architecture and
differ only in the horizon their batches are drawn at, arm 2 at the sampled
range and arm 3 at h = 1.

No metric and no baseline. Those belong to evaluation.
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
    NAVIX_MAX_GRID_EXTENT,
    ExperimentConfig,
    SamplerMode,
    checkpoints_dir,
)
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
    check_observation_values_in_range,
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

# The horizon arm 3's one-step objective trains at. A bare 1 would read as an
# off-by-one guard.
ONE_STEP_HORIZON: int = 1


@dataclass(frozen=True)
class TrainedState:
    """Everything one training run carries between steps and into the payload.

    A dataclass. `params` and `best_params` share a structure and dtype, so a
    transposed pair would pass every shape assertion.

    Attributes:
        params: The last step's parameters. The resume point.
        opt_state: The last step's optimiser state, describing `params` alone.
        step: Gradient steps completed.
        best_params: Parameters from the lowest loss observed, sampled at the
            logging interval.
        best_loss: The held-out loss at those parameters, not the training one.
        best_step: The step it came from.
        total_steps: The budget this run was scheduled for. A checkpoint
            resumed under a different one is not the checkpoint it claims.
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

    The interval is reset by `split`, not by the caller. Resetting in two
    places double-counts an interval and understates the rate.

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
    """Return the checkpoint payload for a state, or a restore template for one.

    One definition of the payload shape, serving both save and restore. Orbax
    matches a restore against the template it is given, so a hand-written copy
    fails by loading plausible wrong weights.

    Args:
        state: The state to persist, or a zero-filled one of the right
            structure to restore into.

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

    A zero-filled state of the right structure. Orbax matches against the
    template it is given, not against what is on disk.

    Args:
        params: A freshly initialised parameter tree of the right shape.
        opt_state: An optimiser state built from that tree.
        total_steps: The budget this run asks for, compared against the one
            recorded in the checkpoint.

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
    """Write params, optimiser state and step, and WAIT for the write.

    The save is asynchronous, so `wait_until_finished()` is not optional, and
    `force=True` is needed to overwrite. Retention keeps exactly one checkpoint,
    so a corrupted mid-write leaves nothing to fall back to.

    Args:
        directory: Where the checkpoint tree is written. Created if absent.
        state: The payload, carrying params, optimiser state, step and
            CHECKPOINT_SCHEMA_KEY.
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


def build_arm(
    config: ExperimentConfig, observation_mode: ObservationMode, *, arm: int
):
    """Construct one arm for one observation mode and initialise its parameters.

    One builder for every arm, so the matched parameter budget holds by
    construction. Module-level: the evaluator restores this stage's checkpoint
    against a template and has to build a structurally identical model. The
    configured extent fixes encoder depth at the value both modes share.

    Args:
        config: The composed experiment configuration.
        observation_mode: The single mode this run predicts.
        arm: One of config.ARMS.

    Returns:
        The unbound model and its initialised parameter tree.

    Raises:
        ValueError: If the arm is not one of config.ARMS.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    source = NavixTrajectorySource(config.env)
    spec = source.spec(observation_mode)
    builder = (
        jumpy_transformer_for_spec if arm == ARM_DIRECT else ar_baseline_for_spec
    )
    model = builder(spec, config.model, depth_extent=NAVIX_MAX_GRID_EXTENT)
    field = spec.single_field()
    height, width = field.shape
    init_key = jax.random.PRNGKey(config.model_seed)
    params_key, dropout_key = jax.random.split(init_key)
    params = model.init(
        {"params": params_key, "dropout": dropout_key},
        jnp.zeros((1, height, width, len(field.cardinality)), jnp.int32),
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
        NAVIX_MAX_GRID_EXTENT,
        count_parameters(params),
    )
    return model, params


class TrainStage(ArmScopedStage):
    """Train one arm on real data and decode a prediction.

    Reads this dataset's shards, re-derives its partition, draws training
    windows, and steps the optimiser for the configured budget. Ends by
    decoding one prediction and writing one checkpoint.

    Model-seed derived. The arm is an instance attribute shadowing `Stage.arm`,
    reaching the checkpoint directory, the metrics directory and the sentinel
    through `arm_root`.

    Attributes:
        store: The trajectory store this stage reads through.
        arm: Which arm this instance trains, one of config.ARMS.
    """

    name: str = "train"
    dataset_derived: bool = False

    @property
    def checkpoint_dir(self) -> Path:
        """Return this run's checkpoint directory, scoped by mode.

        Keyed on the model seed through `artefact_seed`, and on the observation
        mode. Two modes of one run would otherwise write to one directory with
        no warning.

        Returns:
            The directory `save_checkpoint` writes into.
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
        """Return the largest horizon THIS ARM's training batches are drawn at.

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
        """Return the configuration this arm's TRAINING sampler is built from.

        A local derivation, not a change to `self.config`, which every artefact
        records. Arm 3 trains at h = 1 and is evaluated across the full grid by
        rolling out. OFFLINE is refused: the train-window filename carries
        neither the horizon range nor the arm, so arm 3's one-step pool and the
        mixed-horizon pool resolve to one file.

        Returns:
            `self.config` for arms 1 and 2, and a copy narrowed to h = 1 for
            arm 3.

        Raises:
            ValueError: If arm 3 is run under SamplerMode.OFFLINE.
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
        """Load, partition, train, decode, and checkpoint.
        """

        trajectories = self._load_trajectories()
        split = self._partition(trajectories)
        sampler = WindowSampler.from_config(
            self.training_config, trajectories, split, SplitName.TRAIN
        )
        # From `training_config`, as the train sampler is. Arm 3 trains at
        # h = 1, so the full range would measure a horizon mismatch.
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
        """Ask the training loop to stop at the end of the current step.

        Sets a flag and returns. Raising from a handler lands wherever the
        interpreter happened to be, which may be inside a compiled step.

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

        Args:
            None.

        Returns:
            All complete episodes, in shard order.

        Raises:
            FileNotFoundError: If the dataset directory holds no shards.
        """
        return self.store.read_dataset(self.dataset_dir, "training")

    def _partition(self, trajectories: list[Trajectory]) -> TrajectorySplit:
        """Re-derive this dataset's three-way partition.

        Re-derived through the same call the preparation stage makes, so the
        two cannot drift. The provenance file is for a reader, not an input.

        Args:
            trajectories: Every episode in this dataset, in store order.

        Returns:
            The partition, identical to the recorded one.
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
    def _training_method(self):
        """Return the model method this arm's LOSS is computed through.

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
        """Return the mean endpoint cross-entropy over one batch.

        The endpoint is the only supervised state, and one loss serves every
        arm. Arm 3's one-step objective is this loss on batches drawn at h = 1,
        where the endpoint is the next state. Mean over the batch, sum over
        cells, so the value scales with grid size and is not comparable across
        observation modes without normalisation.

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
        return jnp.mean(
            reconstruction_loss(logits, flat, targets.shape[1:3])
        )

    def _train(  # pylint: disable=too-many-locals
        # Over pylint's limit by the validation batch, the timer and the
        # synced loss.
        self,
        model,
        params,
        sampler: WindowSampler,
        validation_batch: WindowBatch,
    ) -> TrainedState:
        """Run the gradient loop and return the trained state.

        The cardinality check runs once, on the first batch. Each step's key is
        folded from the step number, so a resumed run draws the same sequence as
        an uninterrupted one. Both losses are taken at the pre-update
        parameters, the best model is sampled at the logging interval, and the
        validation batch is drawn once and held through `next_batch`. The first
        timing interval includes compilation.

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
                self._check_cardinality(batch)
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

        The batch is unpacked to arrays at the boundary. `WindowBatch` is a
        frozen dataclass, not a pytree, and cannot cross `jit`. `model` is
        closed over, so it compiles once, and the optimiser update stays inside
        the boundary. **Compilation changes operation fusion, so results shift
        bitwise against the uncompiled path and a reporting set must come from
        one code path.**

        Args:
            model: The unbound model for this arm.
            optimiser: The optax transform, compiled in with the gradient.

        Returns:
            A compiled callable taking `(params, opt_state, states, actions,
            horizons, targets, dropout_key)` and returning
            `(new_params, new_opt_state, loss)`.
        """

        @jax.jit
        # The state plus the batch. An object cannot cross a jit boundary.
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

        Shares `_loss` with the training path. Two definitions is how a gap
        stops measuring what is being optimised.

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

        Returns the synced training loss, which best-model tracking also
        needs. Both losses are taken at `state.params`, before the update.

        Args:
            state: The state before this step's update.
            step: The zero-based index of the step just taken.
            loss_value: The training loss at `state.params`, still on device.
            validation_batch: The fixed validation batch for this run.
            timer: The loop's wall-clock bookkeeping.

        Returns:
            The validation loss as a host float when this step logged,
            otherwise None. The validation loss, not the training loss. The
            best model is selected on it.
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
        """Return the held-out loss at these parameters, dropout off.

        A training diagnostic, not a reported result. One fixed batch of the
        validation split, so it says whether the run is overfitting and nothing
        more. Reported numbers come from the evaluation stage.

        Args:
            params: The parameter tree to score, pre-update.
            batch: The fixed validation batch, drawn once per run.

        Returns:
            The mean endpoint cross-entropy on that batch, as a host float.
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

        The gap is signed: positive is validation above training, negative
        usually means dropout is costing the training number. The ETA comes from
        the most recent interval. `elapsed` and `eta` are per seed and reset
        between them, which is why the line names its seed, and the cumulative
        figure is the runner's closing summary.

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

        The key is folded from the step number, making it a function of the
        seed and N alone, so a resumed run and a clean one match bitwise.

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

        Selection is on the validation loss, never the training loss, which
        falls monotonically under overfitting and would track the final
        parameters. The scored parameters are `state.params`, before the update.

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

        Resumption is never a skip. The sentinel decides what runs, so a
        checkpoint recording a complete run is ignored and the run starts over.
        A schema mismatch raises, orbax being willing to match an older payload
        wherever the structures coincide, and so does a changed budget.

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

    def _check_cardinality(self, batch: WindowBatch) -> None:
        """Raise if any observation code falls outside its channel's range.

        Once on the first batch. An out-of-range code gathers a meaningless
        log-probability instead of raising, and it is a dataset property.

        Args:
            batch: The first training batch drawn this run.

        Raises:
            ValueError: If any code is outside its channel's class range.
        """
        check_observation_values_in_range(
            batch.targets.reshape(batch.targets.shape[0], -1),
            batch.targets.shape[1:3],
            self.config.model.obs_channel_classes,
        )

    def _decode_one_prediction(self, model, params, sampler: WindowSampler) -> None:
        """Decode one batch and log what came back.

        The accuracy logged here is not a result and must not be quoted. It is
        measured on training windows with no baseline beside it, where a model
        that only copies its input scores close to one.

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

        The best model is recorded, not substituted. The params key holds the
        final parameters and the evaluation stage scores those, so a run ending
        on a loss spike stays recoverable without retraining.

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
                    "depth_extent": NAVIX_MAX_GRID_EXTENT,
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

        Every field changes the weights and leaves the path alone, which is
        the case the identity check exists for.

        `completed_steps` is what makes an interrupted run refuse to be
        skipped. Before training it reports the full budget, after training
        what happened, and the stage re-reads this once `run()` returns.

        Absent: the checkpoint interval and the rematerialisation flag, neither
        of which changes the weights.

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
                "depth_extent": NAVIX_MAX_GRID_EXTENT,
            }
        )
        return identity
