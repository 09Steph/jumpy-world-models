"""Evaluation stage: compute every reported metric for one trained model.

One stage with a linear dependency chain. The copy tables need the training
split, the skill score needs both cross-entropies, and the probability log needs
the metrics' own forward pass. It branches on the arm only to pick which
architecture to restore into and which directory to use, and it asserts on the
batch's own split name.

Which split is scored is a parameter and defaults to validation. The test split
opens once, so it is reached only when a caller asks for it by name.

The test pass differs from the validation pass in ways that travel together. It
scores the extrapolation horizons as well as the reporting grid, it scores both
stored parameter trees and records their gap, and it reads its dataset and
checkpoints from a source run while writing everything to its own. Reading and
writing one tree is what would overwrite a completed result, so the two
directions are resolved separately and asserted apart.
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
    copy_cross_entropy,
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
    mover_restricted_accuracy,
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
    COPY_CE_KEY,
    EXACT_MATCH_KEY,
    GAP_SIGN_CONVENTION,
    MODEL_CE_KEY,
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

# Carries the observation mode. The `.npz` is explicit; numpy appends it
# otherwise, leaving the recorded size describing a path that does not exist.
PROBABILITY_LOG_TEMPLATE: str = "probability_log_{mode}.npz"

# Where the SECONDARY parameter tree's probability log goes: a subdirectory,
# with the filename unchanged.
#
# A directory and not a filename variant, because sweep.probability_log_paths
# globs the eval directory and filters on `stem.endswith(mode)`. A suffix falls
# out of that filter and is silently omitted; a prefix passes it and is silently
# collected as the primary's. Both leave a sweep artefact that looks finished.
# A subdirectory is invisible to a non-recursive glob.
SECONDARY_PARAMS_DIR_NAME: str = "final_step"

# Fold for the copy tables' training windows. Distinct from the evaluated
# splits' folds, so the baseline is never estimated on what it is scored on.
COPY_TABLE_FOLD: int = 97


def _gap_block(primary: dict, secondary: dict) -> dict:
    """Return primary minus secondary for every numeric metric they share.

    Non-numeric entries are carried as None rather than dropped, so the gap
    block has the same keys as the series it describes and a reader cannot
    mistake an absent key for an absent metric. The window count is skipped:
    both trees are scored on one batch, so its difference is always zero.

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
    Model-seed derived. Metrics describe the fitted model, not the dataset.

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
        """Return the configuration naming the run this stage READS from.

        Every write keys on `config.run_name`; only the dataset and the
        checkpoint follow this. A pass that read and wrote one tree would drop
        a test artefact into the run it was scored from.
        """
        if self.source_run is None:
            return self.config
        return replace(self.config, run_name=self.source_run)

    @property
    def scoring_config(self) -> ExperimentConfig:
        """Return the configuration carrying the grid this pass scores at.

        Keeps `run_name` on the target, so the sampler's artefact directory and
        the frozen window file it writes land in this run's own tree.
        """
        return replace(
            self.config,
            sampler=replace(
                self.config.sampler, evaluation_horizons=self.horizons
            ),
        )

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory, resolved against the SOURCE run.

        Overrides Stage.dataset_dir, leaving the inheriting stages untouched.
        """
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
            split: Which partition to score. Defaults to validation, so the
                test split is reached only by asking for it.
            source_run: Run to read shards and checkpoints from. None reads the
                run it writes to.

        Raises:
            ValueError: If the split has no frozen evaluation set, or if a
                source run is given without asking for the test split. The
                second combination would write a validation score into a tree
                named for the test pass.
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
        """Re-derive this dataset's three-way partition.

        The same call the other two stages make, not a matching one.

        Args:
            trajectories: Every episode in this dataset, in store order.

        Returns:
            The partition, identical to the one `prepare` recorded.
        """
        return split_from_config(
            self.config, [len(trajectory) for trajectory in trajectories]
        )

    def _evaluation_batch(
        self, trajectories: list[Trajectory], split: TrajectorySplit
    ) -> WindowBatch:
        """Return this split's frozen evaluation windows.

        Built from `scoring_config`, so the sampler draws at the grid this pass
        scores and writes its frozen file under the target run. On the test pass
        the target tree is new, so the windows are drawn fresh there and the
        source run's frozen files are never opened.

        Args:
            trajectories: Every episode in this dataset, in store order.
            split: The partition they were split into.

        Returns:
            The frozen evaluation batch, covering every horizon this pass
            scores.

        Raises:
            ValueError: If the batch does not name the split that was asked
                for, which can only fire if the sampler stops honouring the
                split it was handed.
        """
        config = self.scoring_config
        # write_window_batch requires the parent to exist. On an ordinary run
        # `prepare` has made it; on a source-run pass `prepare` never runs
        # there, so this stage owns creating it.
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
        """Draw the training windows the copy baseline is estimated from.

        Drawn, not read from a frozen file. Reproducible from the data seed
        alone, and folded so these windows cannot coincide with the validation
        or test draws.

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
        # Drawn at the evaluated horizons, not at the training distribution, so
        # widening the grid widens the baseline with it. Without that the copy
        # baseline would be undefined at exactly the extrapolation horizons,
        # and it is a required per-horizon key.
        return sampler.frozen_evaluation_set(key, self.horizons)

    def _restore_model(self):
        """Rebuild this arm and restore both of its stored parameter trees.

        The arm must reach the trainer constructed below, which is the call
        site that gets missed. This stage builds one purely to ask where the
        checkpoint lives, so an arm reaching the model builder but not this
        constructor would restore one arm's architecture from another's
        directory.

        The trainer is built from `source_config`, so the checkpoint path
        follows the run being read rather than the run being written.

        The checkpoint carries both trees whatever is scored, so both are
        returned from one restore.

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
        # One definition of the payload shape. A hand-written copy would drift
        # by a key, and orbax would restore plausible wrong weights.
        restored = restore_checkpoint(
            directory,
            restore_target(
                template,
                self._optimiser_template(template),
                self.config.train.total_steps,
            ),
        )
        # int(), explicitly. A restored scalar is a JAX array, which JSON
        # cannot serialise.
        self.completed_steps = int(restored[CHECKPOINT_STEP_KEY])
        # Both keys are in the restore target, so both trees are present or the
        # restore above has already failed.
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

        The test pass scores both, best-loss tree primary, so `per_horizon`
        keeps holding what existing readers expect. Every other pass scores the
        one EvalConfig.params_selection names.
        """
        if self.scores_both_trees:
            return (PARAMS_SELECTION_BEST, PARAMS_SELECTION_FINAL)
        return (self.config.eval.params_selection,)

    def _optimiser_template(self, params):
        """Return an optimiser-state template matching the saved checkpoint.

        Restored though nothing reads it. Orbax matches a template of the whole
        payload, and one missing a key does not describe the file on disk. Built
        from the same call the trainer used.

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
        """Assemble metrics.json for this run.

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
            # Which run's shards and checkpoints produced these numbers. Equal
            # to run_name unless a source override was given, so nothing on
            # disk is left implying the target trained itself.
            SOURCE_RUN_NAME_KEY: self.source_config.run_name,
            "seed": self.artefact_seed,
            # In the path and in the payload. The path stops the arms
            # overwriting one file, the field tells a reader holding one which
            # arm produced it.
            ARM_KEY: self.arm,
            OBSERVATION_MODE_KEY: self.config.sampler.observation_mode,
            SPLIT_KEY: self.split.value,
            "config": config_snapshot(self.config),
            # Read by the aggregator's truncation guard. From the checkpoint
            # not the config, so a truncated run is visible here.
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

        Sibling blocks rather than a change to `per_horizon`, so the primary
        series stays byte-identical to a single-tree score. The primary tree's
        blocks arrive already reduced, so no logits from the first pass are
        still bound when the second runs.

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

        One routine per parameter tree, so the gap the test pass reports is a
        difference in weights rather than in code. The batch is taken one
        horizon at a time and each horizon in blocks, so no array spanning the
        whole batch is ever built.

        `select_log_examples` is called once, on the whole batch. It shares one
        generator across horizon bins, so a call per horizon would restart that
        stream and write a different file.

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

        Per horizon throughout, including the copy tables. The conditional
        P(s_{t+h}[i] = c | s_t[i] = c') is a different distribution at each h,
        and that difference is the task getting harder, so a pooled table would
        understate the baseline at short horizons and overstate it at long ones.

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
        baseline = self._baseline_at(training, horizon, grid_shape)
        # Computed on the whole slice, not per block. It reads observations and
        # never the logits, and observations are small enough that the slice
        # fits where the logits do not. Combining it per block would reassociate
        # a mean over log-probabilities whose dynamic range is wide, measured
        # drifting 1.65e-5 where the model's own metrics drift 1e-7.
        whole = jnp.asarray(rows)
        copy_ce = float(
            copy_cross_entropy(
                baseline["tables"],
                self._flatten(batch.states[whole]),
                self._flatten(batch.targets[whole]),
                grid_shape,
            )
        ) if rows.size else None
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
        block = self._finalise_block(rows, accumulators, baseline, copy_ce)
        if not collected:
            return block, None
        return block, [
            jnp.concatenate([part[channel] for part in collected], axis=0)
            for channel in range(len(collected[0]))
        ]

    def _score_block(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        logits: list,
        states: jax.Array,
        targets: jax.Array,
        grid_shape: tuple[int, int],
        windows: int,
    ) -> dict:
        """Score one block of windows and return each value with its weight.

        Every metric here reads the logits, so every one of them has to be
        measured a block at a time. The weights are what let them recombine:
        each is a mean over windows and carries the block's window count,
        except the mover-restricted accuracy, which is a ratio of two sums and
        carries the block's moved-cell count instead.

        Args:
            logits: The model's logits over this block.
            states: Its flattened start observations.
            targets: Its flattened end observations.
            grid_shape: Spatial grid shape, (height, width).
            windows: How many windows the block holds.

        Returns:
            {metric: (value, weight)}.
        """
        mover = mover_restricted_accuracy(logits, targets, states, grid_shape)
        changed = mover[MOVER_CHANGED_CELLS_KEY]
        return {
            MODEL_CE_KEY: (
                float(model_cross_entropy(logits, targets, grid_shape)),
                windows,
            ),
            PER_CELL_ACCURACY_KEY: (
                float(per_cell_accuracy(logits, targets, grid_shape)),
                windows,
            ),
            EXACT_MATCH_KEY: (
                float(exact_grid_match_rate(logits, targets, grid_shape)),
                windows,
            ),
            MOVER_ACCURACY_KEY: (mover[MOVER_ACCURACY_KEY], changed * windows),
            MOVER_CHANGED_CELLS_KEY: (changed, windows),
            MOVER_EXCLUDED_KEY: (mover[MOVER_EXCLUDED_KEY], windows),
            # Undefined in the egocentric mode, where the agent sits at the
            # centre by construction, so the number would be a constant that
            # reads as a finding.
            AGENT_ACCURACY_KEY: (
                float(agent_position_accuracy(logits, targets, grid_shape))
                if self.config.sampler.observation_mode == OBS_MODE_TOP_DOWN
                else None,
                windows,
            ),
        }

    @staticmethod
    def _finalise_block(
        rows: np.ndarray,
        accumulators: dict,
        baseline: dict,
        copy_ce: float | None,
    ) -> dict:
        """Combine one horizon's blocks into its reported metric block.

        Key order follows the block a single unblocked pass wrote, so the
        artefact is unchanged byte for byte and not merely by value.

        Args:
            rows: This horizon's rows in the evaluation batch.
            accumulators: The combined per-metric accumulators.
            baseline: The copy baseline's report at this horizon.
            copy_ce: The baseline's cross-entropy, measured on the whole slice.

        Returns:
            One horizon's metric block.
        """
        empty = _WeightedMean()
        model_ce = accumulators.get(MODEL_CE_KEY, empty).value()
        return {
            WINDOWS_KEY: int(rows.size),
            MODEL_CE_KEY: model_ce,
            COPY_CE_KEY: copy_ce,
            # Stored, not derived at plotting time, so the reported number and
            # this one cannot diverge.
            SKILL_SCORE_KEY: (
                1.0 - model_ce / copy_ce
                if model_ce is not None and copy_ce
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
        }

    def _baseline_at(
        self, training: WindowBatch, horizon: int, grid_shape: tuple[int, int]
    ) -> dict:
        """Estimate the copy baseline and the climatology floor at one horizon.

        Per horizon, from training windows only. Both estimates come from one
        draw, so the floor and the denominator describe the same dataset.

        Args:
            training: Training windows covering every evaluated horizon.
            horizon: The horizon to estimate at.
            grid_shape: Spatial grid shape, (height, width).

        Returns:
            The row-normalised tables, the raw counts' smoothing report, and
            the climatology entropy.
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
        }

    @staticmethod
    def _flatten(observations: jax.Array) -> jax.Array:
        """Flatten grid-shaped observations for the metric and loss helpers.

        One convention, the loss's. Every routine in `src/eval/` takes
        flattened observations plus a grid shape, so the model's cross-entropy
        and the baseline's are the same computation.

        Args:
            observations: Grid-shaped observations, (batch, height, width,
                channels).

        Returns:
            The same values as (batch, obs_dim).
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
        before writing, so the artefact records the order `select_log_examples`
        returned rather than the order the horizons were scored in.

        Args:
            path: Where to write it. Each parameter tree gets its own, so the
                second cannot overwrite the first.
            picks: The selected rows, in batch order.
            batch: The frozen evaluation batch.
            positions: Per horizon, where its rows sit within `picks`.
            logged: Per horizon, the logits of those rows.
            settings: The settings in force.

        Returns:
            The provenance record, written into metrics.json.
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
        """Write metrics.json for this run and mode.

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

        Every field here changes the numbers and leaves the path alone, which
        is the case the identity check exists for. The evaluation grid is
        included, at the grid actually scored: a different set of horizons
        produces a different file under the same name.

        The validation pass's identity must stay as it is field for field, or
        every sentinel already on disk goes stale. The test pass adds its split
        and replaces `params_selection` with a marker, because it scores both
        trees and refuses `--params`: an identity must not carry a flag its own
        run rejects.

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
