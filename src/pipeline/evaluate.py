"""Evaluation stage: compute every reported metric for one trained model.

The scored split is a parameter, validation by default. The test pass also
scores the extrapolation horizons, scores both stored parameter trees and
records their gap, and can read its dataset and checkpoints from a source run
while writing only to its own.
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from config import (
    ARM_DIRECT,
    CONTINUOUS_VALUE_RANGES,
    EVALUATED_SPLITS,
    EVALUATION_BATCH_CHUNK,
    METRICS_TEMPLATE,
    OBS_MODE_TOP_DOWN,
    PARAMS_SELECTION_BEST,
    PARAMS_SELECTION_FINAL,
    ExperimentConfig,
    config_snapshot,
    data_dir,
    eval_dir,
    evaluation_grid_for_split,
    probability_log_cap,
)
from src.data.split import (
    SplitName,
    TrajectorySplit,
    split_from_config,
)
from src.data.trajectory import Trajectory
from src.data.trajectory_store import TrajectoryStore
from src.data.window_sampler import WindowBatch, WindowSampler
from src.eval.baselines import (
    calibrated_copy_tables,
    climatology_entropy,
    climatology_mse,
    copy_cross_entropy,
    copy_mse,
    copy_transition_counts,
    smoothing_report,
)
from src.eval.metrics import (
    MOVER_ACCURACY_KEY,
    MOVER_CHANGED_CELLS_KEY,
    MOVER_EXCLUDED_KEY,
    agent_position_accuracy,
    exact_grid_match_rate,
    model_cross_entropy,
    model_mse,
    mover_restricted_accuracy,
    mover_restricted_mse,
    per_cell_accuracy,
)
from src.eval.probability_log import (
    LogSettings,
    probabilities_for,
    select_log_examples,
    write_probability_log,
)
from src.models.jumpy_transformer import predict_endpoint
from src.pipeline.base import ArmScopedStage, sampler_identity_fields
from src.pipeline.metrics_schema import (
    AGENT_ACCURACY_KEY,
    ARM_KEY,
    CLIMATOLOGY_KEY,
    CLIMATOLOGY_MSE_KEY,
    COPY_MOVER_MSE_KEY,
    ERROR_METRIC_CROSS_ENTROPY,
    ERROR_METRIC_KEY,
    ERROR_METRIC_MSE,
    EXACT_MATCH_KEY,
    GAP_SIGN_CONVENTION,
    MOVER_CHANGED_PIXELS_KEY,
    MOVER_MSE_KEY,
    MOVER_MSE_SKILL_KEY,
    OBSERVATION_MODE_KEY,
    PARAMS_PROVENANCE_KEY,
    PARAMS_TREES_BOTH,
    PER_CELL_ACCURACY_KEY,
    PER_HORIZON_FINAL_STEP_KEY,
    PER_HORIZON_GAP_KEY,
    PER_HORIZON_KEY,
    PROBABILITY_LOG_KEY,
    RUN_NAME_KEY,
    SKILL_SCORE_KEY,
    SMOOTHING_KEY,
    SOURCE_RUN_NAME_KEY,
    SPLIT_KEY,
    WINDOWS_KEY,
    copy_error_key,
    model_error_key,
)
from src.pipeline.prepare import frozen_evaluation_key
from src.pipeline.train import (
    CHECKPOINT_BEST_LOSS_KEY,
    CHECKPOINT_BEST_PARAMS_KEY,
    CHECKPOINT_BEST_STEP_KEY,
    CHECKPOINT_PARAMS_KEY,
    CHECKPOINT_STEP_KEY,
    TrainStage,
    build_arm,
    restore_checkpoint,
    restore_target,
)
from src.utils.logging_setup import get_logger
from src.utils.optimisers import build_optimiser
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

# Carries the observation mode. The `.npz` is explicit. numpy would otherwise
# append it, and the recorded size would name a path that does not exist.
PROBABILITY_LOG_TEMPLATE: str = "probability_log_{mode}.npz"

# The subdirectory holding the secondary tree's probability log, under the
# primary's filename. `sweep.probability_log_paths` globs non-recursively and
# keeps names ending in the mode, so a renamed file beside the primary's would
# be dropped or collected as the primary's without raising.
SECONDARY_PARAMS_DIR_NAME: str = "final_step"

# Key fold for the training windows the baselines are fitted on.
COPY_TABLE_FOLD: int = 97


def _gap_block(primary: dict, secondary: dict) -> dict:
    """Return primary minus secondary for every numeric metric they share.

    Every primary key except the window count is kept, None where either side
    is not a number.

    Args:
        primary: One horizon's block from the primary parameter tree.
        secondary: The same horizon's block from the secondary tree.

    Returns:
        {metric: primary - secondary}, None where either side is not a number.
    """
    gap: dict = {}
    for metric, value in primary.items():
        if metric == WINDOWS_KEY:
            continue
        other = secondary.get(metric)
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        comparable = isinstance(other, (int, float)) and not isinstance(
            other, bool
        )
        gap[metric] = value - other if numeric and comparable else None
    return gap


def _batch_rows(batch: WindowBatch, rows: np.ndarray) -> WindowBatch:
    """Return the same batch restricted to the given rows.

    Rows only. `actions` keeps its full padded width, so every horizon presents
    one input shape to the compiler.

    Args:
        batch: The batch to restrict.
        rows: Indices into its batch axis.

    Returns:
        The same batch carrying only those rows.
    """
    index = jnp.asarray(rows)
    return replace(
        batch,
        states=batch.states[index],
        actions=batch.actions[index],
        horizons=batch.horizons[index],
        targets=batch.targets[index],
        horizon_counts=batch.horizon_counts[index],
    )


def _padded_chunk(rows: np.ndarray, start: int, chunk: int) -> tuple[np.ndarray, int]:
    """Return one block of rows and how many of them are real.

    A subdivided slice pads its last block to the full width by repeating the
    final row, so one input shape reaches the compiler. A slice that fits in a
    single block is passed at its own width.

    Args:
        rows: The slice's row indices.
        start: Offset of this block within the slice.
        chunk: Block width.

    Returns:
        The block's row indices, and the count that is not padding.
    """
    taken = rows[start : start + chunk]
    real = int(taken.size)
    if real < chunk < rows.size:
        taken = np.concatenate([taken, np.repeat(taken[-1:], chunk - real)])
    return taken, real


class _WeightedMean:
    """Combine one metric's per-block values into the whole slice's value.

    A single contribution is returned untouched, so a slice that is never
    subdivided carries the value it would carry without blocking. A subdivided
    slice recombines values already rounded to float32, which agrees with a
    single pass to about float32 epsilon rather than exactly.
    """

    def __init__(self) -> None:
        """Start with no contributions."""
        self._parts: list[tuple[float, float]] = []

    def add(self, value: float | None, weight: float) -> None:
        """Record one block's value and the weight it carries.

        Args:
            value: The block's value, or None where the metric is undefined.
            weight: Its weight in the combination.
        """
        if value is not None and weight > 0:
            self._parts.append((float(value), float(weight)))

    def value(self) -> float | None:
        """Return the combined value, or None if nothing was recorded."""
        if not self._parts:
            return None
        if len(self._parts) == 1:
            return self._parts[0][0]
        total = sum(weight for _, weight in self._parts)
        return sum(value * weight for value, weight in self._parts) / total


class EvaluateStage(ArmScopedStage):
    """Compute every reported metric for one trained model.

    Reads one split's frozen evaluation set, restores a checkpoint, and writes
    one metrics file plus a probability log per parameter tree scored.

    Attributes:
        store: The trajectory store this stage reads through.
        arm: Which arm's checkpoint this instance scores, one of config.ARMS.
        split: Which partition is scored. VALIDATION unless a caller names TEST.
        source_run: Run whose shards and checkpoints are read, when that is not
            the run being written to. None on an ordinary run.
    """

    name: str = "evaluate"
    dataset_derived: bool = False

    @property
    def value_range(self) -> tuple[int, int] | None:
        """Return the stored-value bounds of this run's representation.

        Returns:
            The bounds for a continuous representation, None for a discrete
            one.
        """
        return CONTINUOUS_VALUE_RANGES.get(self.config.data.representation)

    @property
    def error_metric(self) -> str:
        """Return the error metric this run's representation is scored under.

        Returns:
            The value written under ERROR_METRIC_KEY and read by every
            downstream key selection.
        """
        if self.value_range is None:
            return ERROR_METRIC_CROSS_ENTROPY
        return ERROR_METRIC_MSE

    @property
    def scores_both_trees(self) -> bool:
        """Return whether this pass scores both stored parameter trees."""
        return self.split is SplitName.TEST

    @property
    def horizons(self) -> tuple[int, ...]:
        """Return the horizon grid this pass actually scores."""
        return evaluation_grid_for_split(
            self.config.sampler.evaluation_horizons, self.split.value
        )

    @property
    def source_config(self) -> ExperimentConfig:
        """Return the configuration naming the run this stage reads from.

        Only the dataset and the checkpoint are read through it. Every write
        keys on `config.run_name`.
        """
        if self.source_run is None:
            return self.config
        return replace(self.config, run_name=self.source_run)

    @property
    def scoring_config(self) -> ExperimentConfig:
        """Return the configuration carrying the grid this pass scores at.

        `run_name` stays on the target, so the frozen window file lands in this
        run's own tree.
        """
        return replace(
            self.config,
            sampler=replace(
                self.config.sampler, evaluation_horizons=self.horizons
            ),
        )

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory, resolved against the source run."""
        source = self.source_config
        return data_dir(
            source.run_name, source.data_seed, source.fast, source.env.name
        )

    @property
    def secondary_probability_log_path(self) -> Path:
        """Return the secondary parameter tree's probability log."""
        return (
            self.eval_directory
            / SECONDARY_PARAMS_DIR_NAME
            / PROBABILITY_LOG_TEMPLATE.format(
                mode=self.config.sampler.observation_mode
            )
        )

    @property
    def eval_directory(self) -> Path:
        """Return the directory this stage's artefacts are written to.

        Keyed on the model seed through `artefact_seed`.

        Returns:
            The run's evaluation artefact directory.
        """
        return eval_dir(
            self.config.run_name,
            self.artefact_seed,
            self.config.fast,
            self.config.env.name,
            arm=self.arm,
        )

    @property
    def metrics_path(self) -> Path:
        """Return this run's metrics file, scoped by observation mode.

        Returns:
            `<eval dir>/metrics_<mode>.json`.
        """
        return self.eval_directory / METRICS_TEMPLATE.format(
            mode=self.config.sampler.observation_mode
        )

    @property
    def probability_log_path(self) -> Path:
        """Return this run's probability log, scoped by observation mode.

        Returns:
            `<eval dir>/probability_log_<mode>.npz`.
        """
        return self.eval_directory / PROBABILITY_LOG_TEMPLATE.format(
            mode=self.config.sampler.observation_mode
        )

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        arm: int = ARM_DIRECT,
        split: SplitName = SplitName.VALIDATION,
        source_run: str | None = None,
    ) -> None:
        """Build the stage.

        Args:
            config: The composed experiment configuration.
            arm: Which arm's checkpoint to score, one of config.ARMS.
            split: Which partition to score.
            source_run: Run to read shards and checkpoints from. None reads the
                run it writes to.

        Raises:
            ValueError: If the split has no frozen evaluation set, if the test
                pass is configured with a params_selection other than best, or
                if a source run is given for a split other than test.
        """
        super().__init__(config, arm=arm)
        if split.value not in EVALUATED_SPLITS:
            raise ValueError(
                f"split '{split.value}' has no frozen evaluation set: "
                f"EVALUATED_SPLITS is {EVALUATED_SPLITS}"
            )
        if (
            split is SplitName.TEST
            and config.eval.params_selection != PARAMS_SELECTION_BEST
        ):
            # The CLI refuses --params on this pass, so the only way here is a
            # changed default in config.py.
            raise ValueError(
                "the test pass scores both parameter trees with the "
                f"{PARAMS_SELECTION_BEST!r} tree primary, so "
                f"params_selection={config.eval.params_selection!r} would be "
                "recorded in the artefact while describing no choice that was "
                "made."
            )
        if source_run is not None and split is not SplitName.TEST:
            raise ValueError(
                f"a source run is only read for the test pass, got split "
                f"'{split.value}'. Reading {source_run!r} while scoring "
                "validation would file a validation number under the test "
                "pass's own run name."
            )
        self.split = split
        self.source_run = source_run
        self.store = TrajectoryStore()
        # Read off the checkpoint by _restore_model. Initialised here so the
        # truncation guard's field is never silently absent.
        self.completed_steps: int | None = None

    def run(self) -> None:
        """Score this split against the copy baseline and record it."""
        trajectories = self._load_trajectories()
        split = self._partition(trajectories)
        batch = self._evaluation_batch(trajectories, split)
        model, trees = self._restore_model()
        payload = self._build_payload(trajectories, split, batch, model, trees)
        self._write(payload)

    def _load_trajectories(self) -> list[Trajectory]:
        """Read every shard this data seed's dataset holds.

        Returns:
            All complete episodes, in shard order.

        Raises:
            FileNotFoundError: If the dataset directory holds no shards.
        """
        return self.store.read_dataset(self.dataset_dir, "evaluation")

    def _partition(self, trajectories: list[Trajectory]) -> TrajectorySplit:
        """Re-derive this dataset's partition, through the call `prepare` makes.

        Args:
            trajectories: Every episode in this dataset, in store order.

        Returns:
            The partition, identical to the one `prepare` recorded while the
            shards are unchanged. The recorded file is not compared.
        """
        return split_from_config(
            self.config, [len(trajectory) for trajectory in trajectories]
        )

    def _evaluation_batch(
        self, trajectories: list[Trajectory], split: TrajectorySplit
    ) -> WindowBatch:
        """Return this split's frozen evaluation windows.

        Built from `scoring_config`, so the windows cover the grid this pass
        scores and the frozen file lives under the target run, never the
        source. Under HYBRID or OFFLINE an existing file is read, and a missing
        one is drawn from the current shards and written, so a deleted file on
        a corpus that does not reproduce comes back as different windows
        without raising.

        Args:
            trajectories: Every episode in this dataset, in store order.
            split: The partition they were split into.

        Returns:
            The frozen evaluation batch, covering every horizon this pass
            scores.

        Raises:
            ValueError: If the batch does not name the split that was asked
                for.
        """
        config = self.scoring_config
        # On a source-run pass `prepare` never ran here, so the directory is
        # created.
        ensure_dir(
            data_dir(
                config.run_name, config.data_seed, config.fast, config.env.name
            )
        )
        sampler = WindowSampler.from_config(
            config, trajectories, split, self.split
        )
        batch = sampler.evaluation_set(
            frozen_evaluation_key(config, self.split)
        )
        if batch.split is not self.split:
            raise ValueError(
                f"the evaluation batch names the {batch.split.value} split "
                f"where {self.split.value} was asked for."
            )
        return batch

    def _training_pairs(
        self,
        trajectories: list[Trajectory],
        split: TrajectorySplit,
    ) -> WindowBatch:
        """Draw the training windows the baselines are fitted on.

        Drawn at the scored horizons and never frozen, so they are reproducible
        from the data seed only while the shards are unchanged.

        Args:
            trajectories: Every episode in this dataset, in store order.
            split: The partition they were split into.

        Returns:
            One batch of training windows covering every evaluated horizon.
        """
        sampler = WindowSampler.from_config(
            self.scoring_config, trajectories, split, SplitName.TRAIN
        )
        key = jax.random.fold_in(
            jax.random.PRNGKey(self.config.data_seed), COPY_TABLE_FOLD
        )
        return sampler.frozen_evaluation_set(key, self.horizons)

    def _restore_model(self):
        """Rebuild this arm and restore both of its stored parameter trees.

        The checkpoint directory comes from a trainer built from
        `source_config` with this stage's arm. An arm reaching the model
        builder but not that trainer would restore one arm's architecture from
        another's directory.

        Returns:
            The unbound model and {selection: parameter tree} for both trees.

        Raises:
            FileNotFoundError: If no checkpoint exists for this run, seed, arm
                and observation mode, which means training has not run for it.
        """
        trainer = TrainStage(self.source_config, arm=self.arm)
        directory = trainer.checkpoint_dir
        if not directory.is_dir():
            raise FileNotFoundError(
                f"no checkpoint under {safe_rel(directory)}. Evaluation reads "
                "a trained model and trains nothing, so run training for this "
                f"seed in {self.config.sampler.observation_mode} mode first."
            )
        model, template = build_arm(
            self.config, self.observation_mode, arm=self.arm
        )
        restored = restore_checkpoint(
            directory,
            restore_target(
                template,
                self._optimiser_template(template),
                self.config.train.total_steps,
            ),
        )
        self.completed_steps = int(restored[CHECKPOINT_STEP_KEY])
        trees = {
            PARAMS_SELECTION_BEST: restored[CHECKPOINT_BEST_PARAMS_KEY],
            PARAMS_SELECTION_FINAL: restored[CHECKPOINT_PARAMS_KEY],
        }
        logger.info(
            "restored arm %d, trained for %d steps, best loss %.4f at step %d, "
            "scoring %s parameters <- %s",
            self.arm,
            self.completed_steps,
            float(restored[CHECKPOINT_BEST_LOSS_KEY]),
            int(restored[CHECKPOINT_BEST_STEP_KEY]),
            ", ".join(self._scored_selections()),
            safe_rel(directory),
        )
        return model, trees

    def _scored_selections(self) -> tuple[str, ...]:
        """Return the parameter trees this pass scores, primary first.

        The test pass scores both, best tree first. Every other pass scores the
        one EvalConfig.params_selection names.
        """
        if self.scores_both_trees:
            return (PARAMS_SELECTION_BEST, PARAMS_SELECTION_FINAL)
        return (self.config.eval.params_selection,)

    def _optimiser_template(self, params):
        """Return an optimiser-state template matching the saved checkpoint.

        Nothing reads the restored optimiser state, but the template must
        describe the whole payload, so it is built from the trainer's call.

        Args:
            params: The freshly initialised parameter tree.

        Returns:
            The optimiser state a fresh optimiser would start from.
        """
        return build_optimiser(self.config.train).init(params)

    def _build_payload(
        self,
        trajectories: list[Trajectory],
        split: TrajectorySplit,
        batch: WindowBatch,
        model,
        trees: dict,
    ) -> dict:
        """Assemble the metrics payload for this run and mode.

        Args:
            trajectories: Every episode in this dataset.
            split: Its three-way partition.
            batch: The frozen evaluation batch.
            model: The unbound model to score with.
            trees: {selection: parameter tree}, both trees from the checkpoint.

        Returns:
            The JSON-serialisable payload.
        """
        training = self._training_pairs(trajectories, split)
        horizons = [int(value) for value in self.horizons]
        primary = self._scored_selections()[0]

        blocks, log_record = self._score_tree(
            model,
            trees[primary],
            batch,
            training,
            horizons,
            self.probability_log_path,
        )
        payload = {
            RUN_NAME_KEY: self.config.run_name,
            SOURCE_RUN_NAME_KEY: self.source_config.run_name,
            "seed": self.artefact_seed,
            ARM_KEY: self.arm,
            OBSERVATION_MODE_KEY: self.config.sampler.observation_mode,
            SPLIT_KEY: self.split.value,
            ERROR_METRIC_KEY: self.error_metric,
            "config": config_snapshot(self.config),
            # From the checkpoint, not the config. The aggregator's truncation
            # guard compares it with the budget.
            "completed_steps": self.completed_steps,
            "stopped_early": False,
            PER_HORIZON_KEY: blocks,
            PROBABILITY_LOG_KEY: log_record,
        }
        if not self.scores_both_trees:
            return payload
        return {**payload, **self._secondary_blocks(
            batch, model, trees, training, horizons, blocks
        )}

    def _secondary_blocks(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        batch: WindowBatch,
        model,
        trees: dict,
        training: WindowBatch,
        horizons: list[int],
        primary_blocks: dict,
    ) -> dict:
        """Score the secondary parameter tree and return it with the gap.

        Returned as sibling blocks, leaving `per_horizon` holding the primary
        tree. The primary blocks arrive already reduced, so no first-pass logits
        are still held when the second pass runs.

        Args:
            batch: The frozen evaluation batch.
            model: The unbound model to score with.
            trees: {selection: parameter tree}.
            training: Training windows for the baseline.
            horizons: The horizons scored, as integers.
            primary_blocks: The primary tree's per-horizon blocks.

        Returns:
            The secondary series, the gap, and the provenance naming both.
        """
        primary, secondary = self._scored_selections()
        secondary_blocks, log_record = self._score_tree(
            model,
            trees[secondary],
            batch,
            training,
            horizons,
            self.secondary_probability_log_path,
        )
        return {
            PER_HORIZON_FINAL_STEP_KEY: secondary_blocks,
            PER_HORIZON_GAP_KEY: {
                horizon: _gap_block(primary_blocks[horizon], block)
                for horizon, block in secondary_blocks.items()
            },
            PARAMS_PROVENANCE_KEY: {
                "primary": primary,
                "secondary": secondary,
                "gap_sign_convention": GAP_SIGN_CONVENTION,
                "secondary_probability_log": log_record,
            },
        }

    def _score_tree(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        model,
        params,
        batch: WindowBatch,
        training: WindowBatch,
        horizons: list[int],
        log_path: Path,
    ) -> tuple[dict, dict]:
        """Score one parameter tree over every horizon and write its log.

        Takes the batch one horizon at a time and each horizon in blocks.
        `select_log_examples` is called once, on the whole batch. It shares one
        generator across horizon bins, so a per-horizon call would write a
        different file.

        Args:
            model: The unbound model to score with.
            params: The parameter tree to score.
            batch: The frozen evaluation batch.
            training: Training windows for the baseline.
            horizons: The horizons scored, as integers.
            log_path: Where this tree's probability log is written.

        Returns:
            The per-horizon blocks, and the probability log's provenance.
        """
        settings = self._log_settings()
        labels = np.asarray(batch.horizons)
        picks = select_log_examples(batch.horizons, settings.cap, settings.seed)
        picked_labels = labels[picks]
        grid_shape = tuple(int(dim) for dim in batch.targets.shape[1:3])

        blocks: dict = {}
        positions: list[np.ndarray] = []
        logged: list[list] = []
        for horizon in horizons:
            wanted = np.flatnonzero(picked_labels == horizon)
            block, rows_logged = self._score_horizon(
                horizon,
                batch,
                (model, params),
                training,
                np.flatnonzero(labels == horizon),
                picks[wanted],
                grid_shape,
            )
            blocks[str(horizon)] = block
            if rows_logged is not None:
                positions.append(wanted)
                logged.append(rows_logged)
        record = self._write_probability_log(
            log_path, picks, batch, positions, logged, settings
        )
        return blocks, record

    def _score_horizon(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        horizon: int,
        batch: WindowBatch,
        scorer: tuple,
        training: WindowBatch,
        rows: np.ndarray,
        logged_rows: np.ndarray,
        grid_shape: tuple[int, int],
    ) -> tuple[dict, list | None]:
        """Score one horizon's windows in blocks and combine them.

        The copy tables are fitted per horizon. A table pooled across horizons
        would misstate the baseline at every horizon without raising.

        Args:
            horizon: The horizon to score.
            batch: The frozen evaluation batch, covering every horizon.
            scorer: The unbound model and the parameter tree to apply.
            training: Training windows for the baseline, covering every horizon.
            rows: This horizon's rows in the evaluation batch.
            logged_rows: The subset of those rows the probability log keeps.
            grid_shape: Spatial grid shape, (height, width).

        Returns:
            The horizon's metric block, and the logged rows' logits, or None
            where this horizon contributes none.
        """
        model, params = scorer
        whole = jnp.asarray(rows)
        scored_targets = self._flatten(batch.targets[whole]) if rows.size else None
        baseline = self._baseline_at(training, horizon, grid_shape, scored_targets)
        # The copy's error on the whole scored slice, never per block and never
        # on the training draw, so the skill score divides two errors on the
        # same windows.
        copy_value = (
            self._copy_error_on(
                baseline,
                self._flatten(batch.states[whole]),
                scored_targets,
                grid_shape,
            )
            if rows.size
            else None
        )
        accumulators: dict[str, _WeightedMean] = {}
        collected: list[list] = []
        for start in range(0, int(rows.size), EVALUATION_BATCH_CHUNK):
            taken, real = _padded_chunk(rows, start, EVALUATION_BATCH_CHUNK)
            logits = predict_endpoint(model, params, _batch_rows(batch, taken))
            if real < int(taken.size):
                logits = [channel[:real] for channel in logits]
            scored_rows = taken[:real]
            index = jnp.asarray(scored_rows)
            measured = self._score_block(
                logits,
                self._flatten(batch.states[index]),
                self._flatten(batch.targets[index]),
                grid_shape,
                real,
            )
            for key, (value, weight) in measured.items():
                accumulators.setdefault(key, _WeightedMean()).add(value, weight)
            if logged_rows.size:
                local = np.flatnonzero(np.isin(scored_rows, logged_rows))
                if local.size:
                    collected.append(
                        [channel[jnp.asarray(local)] for channel in logits]
                    )
            del logits
        block = self._finalise_block(
            rows, accumulators, baseline, copy_value, self.error_metric
        )
        if not collected:
            return block, None
        return block, [
            jnp.concatenate([part[channel] for part in collected], axis=0)
            for channel in range(len(collected[0]))
        ]

    def _copy_error_on(
        self,
        baseline: dict,
        states: jax.Array,
        targets: jax.Array,
        grid_shape: tuple[int, int],
    ) -> float:
        """Return the stationary copy's error on the windows being scored.

        The discrete copy applies tables fitted on the training draw. The
        continuous copy fits nothing.

        Args:
            baseline: The baseline report at this horizon.
            states: Flattened start observations of the scored windows.
            targets: Flattened end observations of the same windows.
            grid_shape: Spatial grid shape, (height, width).

        Returns:
            The copy baseline's error under this run's declared metric.
        """
        if self.value_range is None:
            return float(
                copy_cross_entropy(
                    baseline["tables"], states, targets, grid_shape
                )
            )
        return float(copy_mse(states, targets, grid_shape, self.value_range))

    def _score_block(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        logits: list,
        states: jax.Array,
        targets: jax.Array,
        grid_shape: tuple[int, int],
        windows: int,
    ) -> dict:
        """Score one block of windows and return each value with its weight.

        The logits exist only per block, so these metrics are measured a block
        at a time and recombined by weight. A mean over windows carries the
        block's window count, and a mover-restricted ratio carries the block's
        moved cell-channel or pixel-channel count.

        Args:
            logits: The model's logits over this block.
            states: Its flattened start observations.
            targets: Its flattened end observations.
            grid_shape: Spatial grid shape, (height, width).
            windows: How many windows the block holds.

        Returns:
            {metric: (value, weight)}.
        """
        # The categorical metrics are None on a continuous representation. The
        # mover metrics have a leg per representation sharing the excluded
        # fraction.
        continuous = self.value_range is not None
        mover = (
            None
            if continuous
            else mover_restricted_accuracy(logits, targets, states, grid_shape)
        )
        mover_mse = (
            mover_restricted_mse(
                logits[0], targets, states, grid_shape, self.value_range
            )
            if continuous
            else None
        )
        changed = None if mover is None else mover[MOVER_CHANGED_CELLS_KEY]
        changed_pixels = (
            None if mover_mse is None else mover_mse[MOVER_CHANGED_PIXELS_KEY]
        )
        mover_reading = mover if mover is not None else mover_mse
        return {
            model_error_key(self.error_metric): (
                float(
                    model_mse(logits[0], targets, grid_shape, self.value_range)
                    if continuous
                    else model_cross_entropy(logits, targets, grid_shape)
                ),
                windows,
            ),
            PER_CELL_ACCURACY_KEY: (
                None
                if continuous
                else float(per_cell_accuracy(logits, targets, grid_shape)),
                windows,
            ),
            EXACT_MATCH_KEY: (
                None
                if continuous
                else float(exact_grid_match_rate(logits, targets, grid_shape)),
                windows,
            ),
            MOVER_ACCURACY_KEY: (
                None if mover is None else mover[MOVER_ACCURACY_KEY],
                0 if changed is None else changed * windows,
            ),
            MOVER_CHANGED_CELLS_KEY: (changed, windows),
            MOVER_EXCLUDED_KEY: (mover_reading[MOVER_EXCLUDED_KEY], windows),
            # Top-down only. The egocentric view shows floor in the agent's
            # cell, so the agent class never appears there.
            AGENT_ACCURACY_KEY: (
                float(agent_position_accuracy(logits, targets, grid_shape))
                if not continuous
                and self.config.sampler.observation_mode == OBS_MODE_TOP_DOWN
                else None,
                windows,
            ),
            MOVER_MSE_KEY: (
                None if mover_mse is None else mover_mse[MOVER_MSE_KEY],
                0 if changed_pixels is None else changed_pixels * windows,
            ),
            COPY_MOVER_MSE_KEY: (
                None if mover_mse is None else mover_mse[COPY_MOVER_MSE_KEY],
                0 if changed_pixels is None else changed_pixels * windows,
            ),
            MOVER_CHANGED_PIXELS_KEY: (changed_pixels, windows),
        }

    @staticmethod
    def _finalise_block(
        rows: np.ndarray,
        accumulators: dict,
        baseline: dict,
        copy_value: float | None,
        error_metric: str,
    ) -> dict:
        """Combine one horizon's blocks into its reported metric block.

        The key order is fixed, with the continuous keys last and None on a
        discrete run. The model and copy keys resolve from one declared metric,
        so the skill score divides like with like. A mismatched pair would not
        raise.

        Args:
            rows: This horizon's rows in the evaluation batch.
            accumulators: The combined per-metric accumulators.
            baseline: The copy baseline's report at this horizon.
            copy_value: The copy baseline's error, measured on the whole slice.
            error_metric: The metric this run is scored under.

        Returns:
            One horizon's metric block.
        """
        empty = _WeightedMean()
        model_value = accumulators.get(
            model_error_key(error_metric), empty
        ).value()
        mover_model = accumulators.get(MOVER_MSE_KEY, empty).value()
        mover_copy = accumulators.get(COPY_MOVER_MSE_KEY, empty).value()
        return {
            WINDOWS_KEY: int(rows.size),
            model_error_key(error_metric): model_value,
            copy_error_key(error_metric): copy_value,
            SKILL_SCORE_KEY: (
                1.0 - model_value / copy_value
                if model_value is not None and copy_value
                else None
            ),
            CLIMATOLOGY_KEY: baseline[CLIMATOLOGY_KEY],
            PER_CELL_ACCURACY_KEY: accumulators.get(
                PER_CELL_ACCURACY_KEY, empty
            ).value(),
            EXACT_MATCH_KEY: accumulators.get(EXACT_MATCH_KEY, empty).value(),
            SMOOTHING_KEY: baseline[SMOOTHING_KEY],
            MOVER_ACCURACY_KEY: accumulators.get(
                MOVER_ACCURACY_KEY, empty
            ).value(),
            MOVER_CHANGED_CELLS_KEY: accumulators.get(
                MOVER_CHANGED_CELLS_KEY, empty
            ).value(),
            MOVER_EXCLUDED_KEY: accumulators.get(
                MOVER_EXCLUDED_KEY, empty
            ).value(),
            AGENT_ACCURACY_KEY: accumulators.get(
                AGENT_ACCURACY_KEY, empty
            ).value(),
            CLIMATOLOGY_MSE_KEY: baseline[CLIMATOLOGY_MSE_KEY],
            MOVER_MSE_KEY: mover_model,
            COPY_MOVER_MSE_KEY: mover_copy,
            MOVER_MSE_SKILL_KEY: (
                1.0 - mover_model / mover_copy
                if mover_model is not None and mover_copy
                else None
            ),
            MOVER_CHANGED_PIXELS_KEY: accumulators.get(
                MOVER_CHANGED_PIXELS_KEY, empty
            ).value(),
        }

    def _baseline_at(
        self,
        training: WindowBatch,
        horizon: int,
        grid_shape: tuple[int, int],
        scored_targets: jax.Array | None,
    ) -> dict:
        """Estimate what the baselines need fitting at one horizon.

        Both branches return the same keys, so every consumer subscripts the
        same dict whatever the representation.

        Args:
            training: Training windows covering every evaluated horizon.
            horizon: The horizon to estimate at.
            grid_shape: Spatial grid shape, (height, width).
            scored_targets: Flattened end observations of the windows being
                scored at this horizon, or None where it has none.

        Returns:
            The baseline report for this run's representation.
        """
        if self.value_range is None:
            return self._categorical_baseline_at(training, horizon, grid_shape)
        return self._continuous_baseline_at(
            training, horizon, grid_shape, scored_targets
        )

    def _categorical_baseline_at(
        self, training: WindowBatch, horizon: int, grid_shape: tuple[int, int]
    ) -> dict:
        """Fit the copy tables and the climatology floor at one horizon.

        Both come from the same training draw.

        Args:
            training: Training windows covering every evaluated horizon.
            horizon: The horizon to estimate at.
            grid_shape: Spatial grid shape, (height, width).

        Returns:
            The row-normalised tables, the raw counts' smoothing report, the
            climatology entropy, and the climatology MSE as None.
        """
        classes = self.config.model.obs_channel_classes
        picks = jnp.flatnonzero(training.horizons == horizon)
        states = self._flatten(training.states[picks])
        targets = self._flatten(training.targets[picks])
        return {
            "tables": calibrated_copy_tables(states, targets, grid_shape, classes),
            SMOOTHING_KEY: smoothing_report(
                copy_transition_counts(states, targets, grid_shape, classes)
            ),
            CLIMATOLOGY_KEY: float(
                climatology_entropy(targets, grid_shape, classes)
            ),
            CLIMATOLOGY_MSE_KEY: None,
        }

    def _continuous_baseline_at(
        self,
        training: WindowBatch,
        horizon: int,
        grid_shape: tuple[int, int],
        scored_targets: jax.Array | None,
    ) -> dict:
        """Return the baseline report for a continuous representation.

        The categorical keys are None, since each estimates a distribution over
        classes. The climatology MSE is fitted on this horizon's training
        windows and read on the windows being scored.

        Args:
            training: Training windows covering every evaluated horizon.
            horizon: The horizon to estimate at.
            grid_shape: Spatial grid shape, (height, width).
            scored_targets: Flattened end observations of the windows being
                scored at this horizon, or None where it has none.

        Returns:
            The same keys the categorical branch returns, the categorical ones
            None.
        """
        climatology = None
        if scored_targets is not None:
            picks = jnp.flatnonzero(training.horizons == horizon)
            climatology = float(
                climatology_mse(
                    self._flatten(training.targets[picks]),
                    scored_targets,
                    grid_shape,
                    self.value_range,
                )
            )
        return {
            "tables": None,
            SMOOTHING_KEY: None,
            CLIMATOLOGY_KEY: None,
            CLIMATOLOGY_MSE_KEY: climatology,
        }

    @staticmethod
    def _flatten(observations: jax.Array) -> jax.Array:
        """Flatten (batch, height, width, channels) observations to (batch, obs_dim).

        The form every metric and baseline in `src/eval/` takes.
        """
        return observations.reshape(observations.shape[0], -1)

    def _log_settings(self) -> LogSettings:
        """Return the settings this run's probability logs are written under."""
        return LogSettings(
            seed=self.config.eval.probability_log_seed,
            dtype=self.config.eval.probability_log_dtype,
            cap=probability_log_cap(self.config.env.name),
        )

    @staticmethod
    def _write_probability_log(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        path: Path,
        picks: np.ndarray,
        batch: WindowBatch,
        positions: list,
        logged: list,
        settings: LogSettings,
    ) -> dict:
        """Write the seeded probability subsample and return its provenance.

        The rows arrive grouped by horizon and are restored to batch order
        before writing.

        Args:
            path: Where to write it. Each parameter tree gets its own, so the
                second cannot overwrite the first.
            picks: The selected rows, in batch order.
            batch: The frozen evaluation batch.
            positions: Per horizon, where its rows sit within `picks`.
            logged: Per horizon, the logits of those rows.
            settings: The settings in force.

        Returns:
            The provenance record, written into the metrics payload.
        """
        ensure_dir(path.parent)
        selected: list = []
        if logged:
            joined = [
                jnp.concatenate([part[channel] for part in logged], axis=0)
                for channel in range(len(logged[0]))
            ]
            selected = probabilities_for(
                joined, np.argsort(np.concatenate(positions), kind="stable")
            )
        return write_probability_log(
            path,
            selected,
            batch.horizons[jnp.asarray(picks)],
            settings,
        )

    def _write(self, payload: dict) -> None:
        """Write this run's metrics file for its observation mode.

        Args:
            payload: The assembled metrics.
        """
        ensure_dir(self.metrics_path.parent)
        self.metrics_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.info("wrote the metric suite -> %s", safe_rel(self.metrics_path))

    def sentinel_identity(self) -> dict:
        """Return the identity guarding this run's metrics.

        The evaluation grid is recorded at the grid actually scored. The test
        pass records its split and PARAMS_TREES_BOTH in place of
        params_selection. The identity carries nothing describing how a metric
        is computed, so a changed definition under an unchanged key name
        matches and skips. Adding a field marks every existing evaluate sentinel
        stale, so a backfill must reach a tree before code that emits the field.

        Returns:
            JSON-serialisable identity fields.
        """
        identity = super().sentinel_identity()
        fields = sampler_identity_fields(self.config)
        fields["evaluation_horizons"] = list(self.horizons)
        identity.update(
            {
                **fields,
                # Already a directory level. Included so a sentinel missing the
                # field mismatches.
                "arm": self.arm,
                # Without it, a run rescored under another representation
                # matches, skips and keeps the previous numbers.
                "representation": self.config.data.representation,
                "total_steps": self.config.train.total_steps,
                "probability_log_seed": self.config.eval.probability_log_seed,
                "probability_log_dtype": self.config.eval.probability_log_dtype,
            }
        )
        if self.scores_both_trees:
            identity["split"] = self.split.value
            identity["parameter_trees"] = PARAMS_TREES_BOTH
        else:
            identity["params_selection"] = self.config.eval.params_selection
        return identity
