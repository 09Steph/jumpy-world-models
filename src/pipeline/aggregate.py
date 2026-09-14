"""Cross-seed aggregation of one run's per-seed artefacts.

Combines the reporting seeds of one run and arm into ``aggregate_arm<n>.json``,
carrying the mean, the sample standard deviation and the interquartile mean
with a bootstrap confidence interval, per horizon and across seeds, for the
metrics the per-seed artefacts carry. The runner's auto-trigger skips an
incomplete series and the CLI raises.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from config import (
    ARM_DIR_PREFIX,
    ARM_DIRECT,
    ARMS,
    DEFAULT_ENV_NAME,
    METRICS_GLOB,
    METRICS_TEMPLATE,
    POLICY_METRICS_FILENAME,
    SEED_DIR_PREFIX,
    SEEDS,
    ExperimentConfig,
    run_root,
)
from src.eval.metrics import MOVER_ACCURACY_KEY
from src.pipeline.aggregate_stats import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CONFIDENCE_INTERVAL_SIZE,
    STD_DDOF,
    aggregate_scalar,
    horizon_mean,
    sample_std,
)
from src.pipeline.metrics_schema import (
    COPY_MOVER_MSE_KEY,
    ERROR_METRIC_CROSS_ENTROPY,
    ERROR_METRIC_KEY,
    MOVER_MSE_KEY,
    MOVER_MSE_SKILL_KEY,
    PER_HORIZON_FINAL_STEP_KEY,
    PER_HORIZON_GAP_KEY,
    PER_HORIZON_KEY,
    SKILL_SCORE_KEY,
    WINDOWS_KEY,
    copy_error_key,
    model_error_key,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import safe_rel

logger = get_logger(__name__)

# Filename for one arm's cross-seed artefact. The observation mode is a key
# inside it.
AGGREGATE_TEMPLATE: str = "aggregate_arm{arm}.json"

AGGREGATE_FILENAME: str = AGGREGATE_TEMPLATE.format(arm=ARM_DIRECT)

def required_per_horizon_keys(error_metric: str) -> tuple[str, ...]:
    """Return the per-horizon keys a run under one metric must carry.

    Every other metric is discovered from the artefact rather than required.

    Args:
        error_metric: The value the artefact declares under ERROR_METRIC_KEY.

    Returns:
        The required keys for that metric.
    """
    return (
        model_error_key(error_metric),
        copy_error_key(error_metric),
        SKILL_SCORE_KEY,
        WINDOWS_KEY,
    )

# Written empty into every mode's block. An undefined per-horizon metric raises
# or is recorded as None under its own name instead.
OMITTED_METRICS_KEY: str = "omitted_metrics"

# Suffix for the per-horizon window count, which is reported as a range.
WINDOWS_RANGE_KEY: str = f"{WINDOWS_KEY}_range"

# Metrics whose None on some seeds gives a thin cell rather than refusing the
# series: the mover-restricted ones, None where no scored window changed. A
# seed whose artefact predates one of these also reads None, and is treated
# as an empty sample in the same way.
SAMPLING_NULLABLE_METRICS: tuple[str, ...] = (
    MOVER_ACCURACY_KEY,
    MOVER_MSE_KEY,
    COPY_MOVER_MSE_KEY,
    MOVER_MSE_SKILL_KEY,
)

# Why a cell was built from fewer than every seed. Nothing else may be written
# into a cell's reason field.
SAMPLING_NULL_ON_SOME_SEEDS: str = "sampling_null_on_some_seeds"
BELOW_SEED_FLOOR: str = "below_seed_floor"

# Fewest defined seeds a cell may be aggregated from. Below it the cell reports
# itself unavailable rather than aggregating.
MIN_SEEDS_FOR_CELL: int = 3

# The statistics an unavailable cell carries as null: every aggregate_scalar
# field except `values`.
UNAVAILABLE_STATISTICS: tuple[str, ...] = (
    "mean",
    "std",
    "iqm",
    "ci_low",
    "ci_high",
)

# Written into every mode's block, empty where every cell used every seed.
THIN_CELLS_KEY: str = "thin_cells"

# Fields a thin or unavailable cell carries beyond its statistics.
CELL_SEEDS_KEY: str = "n_seeds"
CELL_ABSENT_SEEDS_KEY: str = "absent_seeds"
CELL_REASON_KEY: str = "reason"
CELL_VALUES_KEY: str = "values"

# Inert. Nothing writes a policy artefact.
POLICY_METRICS: tuple[str, ...] = (
    "success_rate",
    "truncation_rate",
    "mean_discounted_return",
    "mean_undiscounted_return",
    "mean_episode_length",
    "mean_successful_episode_length",
    "critic_value_rank_correlation",
    "critic_value_mean_absolute_error",
)

# Nothing under src/ writes `held_out_curve` or `training_loss_curve`, so
# `_aggregate_curves` returns empty for all of these.
HELD_OUT_CURVE_VALUE_FIELD: str = "agent_position_accuracy_per_step"
TRAINING_LOSS_SERIES: tuple[str, ...] = (
    "loss",
    "loss_pred",
    "loss_dyn",
    "loss_rep",
    "kl_mean",
    "dyn_ent",
    "rep_ent",
)

# Policy fields copied onto the aggregate unaveraged. Inert, like the rest of
# the policy side.
POLICY_PROVENANCE_FIELDS: tuple[str, ...] = (
    "headline_action_rule",
    "random_policy_success_rate",
    "random_policy_success_ci",
)

class UnusableSeedError(ValueError):
    """Raised when a present seed must not enter a reporting aggregate.

    A missing seed is SeriesIncompleteError's case.
    """


class SeriesIncompleteError(ValueError):
    """Raised when the seeds do not form one comparable series.

    Covers missing seeds without --allow-partial, nothing to aggregate,
    disagreeing horizon grids or error metrics, a missing required key, and a
    metric or parameter-tree block present on some seeds only.
    """


def infer_run_name(series_dir: Path) -> str:
    """Return a series directory's own name as the run name.

    Under the current layout a series directory is `<run>/<env>`, so this
    returns the environment name.

    Args:
        series_dir: A directory whose children are `seed<n>` directories.

    Returns:
        The directory's own name.

    Raises:
        SeriesIncompleteError: If no `seed<n>` child directory is present, which
            means this is not a run directory under the current layout.
    """
    seed_dirs = [
        child
        for child in (sorted(series_dir.iterdir()) if series_dir.exists() else [])
        if child.is_dir() and child.name.startswith(SEED_DIR_PREFIX)
    ]
    if not seed_dirs:
        raise SeriesIncompleteError(
            f"no seed directory under {safe_rel(series_dir)} -- expected "
            f"children named {SEED_DIR_PREFIX}<seed>. Pass --run-name "
            f"explicitly if the layout differs."
        )
    return series_dir.name


class CrossSeedAggregator:
    """Combine one run's per-seed artefacts into a reportable aggregate.

    Reads, never writes, the per-seed artefacts, one run and arm at a time.
    Refuses a truncated seed: `completed_steps` below the recorded budget, or
    `stopped_early`, which the evaluate stage always writes as False.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        run_name: str,
        series_dir: Path | None = None,
        seeds: Sequence[int] | None = None,
        reps: int | None = None,
        statistic_seed: int | None = None,
        arm: int = ARM_DIRECT,
        env: str | None = None,
        fast: bool = False,
    ) -> None:
        """Store what the aggregation reads and how it resamples.

        Args:
            run_name: The series-level run name, without any seed suffix.
            series_dir: Directory holding this series' per-seed directories.
                Defaults to the parent of `config.run_root`'s seed directory.
            seeds: The reporting seeds this series must carry.
            reps: Bootstrap resamples for the confidence interval.
            statistic_seed: Seed for the bootstrap's random state.
            arm: Which arm's series to aggregate, one of config.ARMS. Arms are
                never averaged together.
            env: Registered environment name, defaulting to DEFAULT_ENV_NAME.
                Ignored when series_dir is given.
            fast: Whether the series was produced by a fast run. Ignored when
                series_dir is given.

        Raises:
            ValueError: If the arm is not one of config.ARMS.
        """
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
        self.arm = arm
        self.run_name = run_name
        self.env = env or DEFAULT_ENV_NAME
        # From `run_root`, so the environment and fast levels match where the
        # seeds were written.
        self.series_dir = (
            series_dir
            if series_dir is not None
            else run_root(run_name, SEEDS[0], fast, self.env).parent
        )
        self.seeds = tuple(SEEDS if seeds is None else seeds)
        self.reps = BOOTSTRAP_REPS if reps is None else reps
        self.statistic_seed = (
            BOOTSTRAP_SEED if statistic_seed is None else statistic_seed
        )

    # -- paths ---------------------------------------------------------------

    def seed_dir(self, seed: int) -> Path:
        """Return one seed's evaluation directory for this arm.

        Mirrors `config.eval_dir`: `<series>/seed<n>/arm<n>/eval`.

        Args:
            seed: The reporting seed.

        Returns:
            The directory holding that seed's metrics files for this arm.
        """
        return (
            self.series_dir
            / f"{SEED_DIR_PREFIX}{seed}"
            / f"{ARM_DIR_PREFIX}{self.arm}"
            / "eval"
        )

    def aggregate_path(self) -> Path:
        """Return this arm's aggregate artefact path, beside its seed directories."""
        return self.series_dir / AGGREGATE_TEMPLATE.format(arm=self.arm)

    # -- completeness --------------------------------------------------------

    def discovered_modes(self) -> tuple[str, ...]:
        """Return the observation modes this series has metrics files for.

        Returns:
            The modes found, sorted.
        """
        prefix, suffix = METRICS_TEMPLATE.split("{mode}")
        found = {
            path.name[len(prefix) : -len(suffix)]
            for seed in self.seeds
            for path in sorted(self.seed_dir(seed).glob(METRICS_GLOB))
        }
        return tuple(sorted(found))

    def present_seeds(self, mode: str) -> tuple[int, ...]:
        """Return the seeds whose metrics file for one mode is on disk.

        Args:
            mode: The observation mode to look for.

        Returns:
            The present seeds, in the configured order.
        """
        return tuple(
            seed
            for seed in self.seeds
            if (self.seed_dir(seed) / METRICS_TEMPLATE.format(mode=mode)).exists()
        )

    def series_is_complete(self) -> bool:
        """Return whether every discovered mode carries every configured seed.

        Counted per mode. A truncated seed counts as present here and is
        refused when loaded.

        Returns:
            True when at least one mode was found and every mode found has
            every configured seed.
        """
        modes = self.discovered_modes()
        return bool(modes) and all(
            len(self.present_seeds(mode)) == len(self.seeds) for mode in modes
        )

    # -- loading and validation ----------------------------------------------

    def _declared_error_metric(self, records: Sequence[dict]) -> str:
        """Return the error metric this mode's seeds were scored under.

        An artefact without the field is read as cross-entropy.

        Args:
            records: The loaded per-seed records for one mode.

        Returns:
            The declared metric.

        Raises:
            SeriesIncompleteError: If the seeds disagree, which means they were
                scored under different metrics and cannot be pooled.
        """
        declared = {
            record["metrics"].get(ERROR_METRIC_KEY, ERROR_METRIC_CROSS_ENTROPY)
            for record in records
        }
        if len(declared) > 1:
            raise SeriesIncompleteError(
                f"seeds of run '{self.run_name}' disagree about "
                f"{ERROR_METRIC_KEY}: {sorted(declared)}. They were scored "
                f"under different error metrics and cannot be pooled."
            )
        return declared.pop()

    def _load_seed_artefacts(
        self, mode: str, allow_partial: bool = False
    ) -> list[dict]:
        """Load and validate every present seed's artefacts for one mode.

        Args:
            mode: The observation mode this series is being loaded for.
            allow_partial: Waive the all-seeds-present requirement, and
                nothing else. A truncated or mismatched seed is still refused.

        Returns:
            One record per seed, holding the seed and its parsed artefacts.

        Raises:
            SeriesIncompleteError: If seeds are missing and allow_partial is
                not set, or if no seed is present at all.
            UnusableSeedError: If a seed is truncated, disagrees with its
                siblings on config, or records a different mode from its name.
        """
        present = self.present_seeds(mode)
        missing = [seed for seed in self.seeds if seed not in present]
        if missing and not allow_partial:
            raise SeriesIncompleteError(
                f"run '{self.run_name}' has {len(present)} of "
                f"{len(self.seeds)} seeds in {mode} mode under "
                f"{safe_rel(self.series_dir)} -- missing {missing}. Run them, "
                f"or pass --allow-partial to aggregate a below-protocol "
                f"series deliberately."
            )
        if not present:
            raise SeriesIncompleteError(
                f"no seed of run '{self.run_name}' is present in {mode} mode "
                f"under {safe_rel(self.series_dir)} -- nothing to aggregate."
            )

        records = [self._load_one_seed(seed, mode) for seed in present]
        self._assert_configs_agree(records)
        return records

    def _load_one_seed(self, seed: int, mode: str) -> dict:
        """Load one seed's artefacts and apply the truncation guard.

        Args:
            seed: The reporting seed to load.
            mode: The observation mode whose metrics file to read.

        Returns:
            A record holding the seed and both parsed artefacts.

        Raises:
            UnusableSeedError: If the run was cut short, or if the file's
                recorded observation mode disagrees with its filename.
        """
        directory = self.seed_dir(seed)
        metrics = json.loads(
            (directory / METRICS_TEMPLATE.format(mode=mode)).read_text(
                encoding="utf-8"
            )
        )
        # Checked against the content, since discovery reads the mode off the
        # filename.
        recorded = metrics.get("observation_mode")
        if recorded is not None and recorded != mode:
            raise UnusableSeedError(
                f"seed {seed} of run '{self.run_name}' has a metrics file "
                f"named for {mode} whose contents record {recorded} -- the "
                f"name and the body disagree, so neither can be trusted."
            )
        completed = metrics.get("completed_steps")
        budget = metrics.get("config", {}).get("total_steps")
        if metrics.get("stopped_early"):
            raise UnusableSeedError(
                f"seed {seed} of run '{self.run_name}' stopped early at step "
                f"{completed} of {budget} -- a truncated seed must not be "
                f"averaged into a reporting series."
            )
        if completed is not None and budget is not None and completed < budget:
            raise UnusableSeedError(
                f"seed {seed} of run '{self.run_name}' completed {completed} "
                f"of {budget} steps -- a truncated seed must not be averaged "
                f"into a reporting series."
            )

        policy_path = directory / POLICY_METRICS_FILENAME
        policy = (
            json.loads(policy_path.read_text(encoding="utf-8"))
            if policy_path.exists()
            else None
        )
        return {"seed": seed, "metrics": metrics, "policy": policy}

    def _assert_configs_agree(self, records: Sequence[dict]) -> None:
        """Refuse a series whose seeds' config snapshots differ.

        Only the fields `config_snapshot` records are compared, so seeds
        differing in anything outside it, such as representation or collection
        policy, pass as one series.

        Args:
            records: The loaded per-seed records.

        Raises:
            UnusableSeedError: If any seed's config snapshot differs.
        """
        reference = records[0]["metrics"].get("config")
        for record in records[1:]:
            snapshot = record["metrics"].get("config")
            if snapshot != reference:
                differing = sorted(
                    key
                    for key in set(reference or {}) | set(snapshot or {})
                    if (reference or {}).get(key) != (snapshot or {}).get(key)
                )
                raise UnusableSeedError(
                    f"seed {record['seed']} of run '{self.run_name}' ran under "
                    f"a different configuration from seed "
                    f"{records[0]['seed']} (differing fields: {differing}) -- "
                    f"a multi-variable series is not a series."
                )

    # -- statistics ----------------------------------------------------------

    def _aggregate_named_scalars(
        self, named_values: dict[str, list[float | None]]
    ) -> tuple[dict, dict]:
        """Aggregate a group of policy metrics, omitting any with a missing value.

        A metric None on any seed is omitted, never imputed.

        Args:
            named_values: Metric name to its value on each seed, in seed order.

        Returns:
            The aggregated blocks, and a mapping of omitted metric to reason.
        """
        aggregated: dict[str, dict] = {}
        omitted: dict[str, str] = {}
        for name, values in named_values.items():
            if any(value is None for value in values):
                omitted[name] = (
                    "undefined on at least one seed, so the metric is reported "
                    "as absent rather than imputed"
                )
                logger.warning(
                    "metric %s is undefined on at least one seed of run %s -- "
                    "omitted from the aggregate rather than imputed",
                    name,
                    self.run_name,
                )
                continue
            aggregated[name] = aggregate_scalar(
                [value for value in values if value is not None],
                reps=self.reps,
                statistic_seed=self.statistic_seed,
            )
        return aggregated, omitted

    # -- per-horizon metrics -------------------------------------------------

    def _agreed_horizons(
        self, records: Sequence[dict], block_key: str = PER_HORIZON_KEY
    ) -> list[str]:
        """Return the horizon grid every seed carries, in numeric order.

        Args:
            records: The loaded per-seed records.
            block_key: Which per-horizon block to read the grid from.

        Returns:
            The horizon keys as EvaluateStage wrote them, sorted numerically so
            the artefact's key order does not depend on JSON parse order.

        Raises:
            SeriesIncompleteError: If a seed carries no such block, or if the
                seeds do not carry identical horizon grids.
        """
        grids = {
            record["seed"]: set(record["metrics"].get(block_key, {}))
            for record in records
        }
        empty = sorted(seed for seed, grid in grids.items() if not grid)
        if empty:
            raise SeriesIncompleteError(
                f"seeds {empty} of run '{self.run_name}' carry no "
                f"'{block_key}' block -- there is nothing to aggregate, "
                f"and an artefact reporting that silently is what this check "
                f"exists to prevent."
            )
        reference = grids[records[0]["seed"]]
        for seed, grid in grids.items():
            if grid != reference:
                raise SeriesIncompleteError(
                    f"seed {seed} of run '{self.run_name}' was evaluated at "
                    f"horizons {sorted(grid, key=int)} against seed "
                    f"{records[0]['seed']}'s {sorted(reference, key=int)} -- "
                    f"seeds evaluated on different grids are not a series."
                )
        return sorted(reference, key=int)

    def _aggregate_per_horizon(
        self,
        records: Sequence[dict],
        block_key: str = PER_HORIZON_KEY,
        counts_windows: bool = True,
    ) -> dict:
        """Aggregate every per-horizon metric across seeds, horizon by horizon.

        Across seeds at a fixed horizon, never across horizons. Metric names
        come from the first seed's block and the required keys are asserted, so
        a metric the first seed lacks is dropped without raising.

        Args:
            records: The loaded per-seed records.
            block_key: Which per-horizon block to aggregate.
            counts_windows: Whether the block carries a window count. False for
                the gap block.

        Returns:
            {horizon: {metric: aggregate block}}, plus a per-horizon window
            range where the block carries one.

        Raises:
            SeriesIncompleteError: If the seeds disagree on the horizon grid,
                if a required_per_horizon_keys entry is absent, or if a metric
                outside SAMPLING_NULLABLE_METRICS is undefined on some seeds
                but not all.
        """
        return {
            horizon: self._aggregate_one_horizon(
                horizon, records, block_key, counts_windows
            )
            for horizon in self._agreed_horizons(records, block_key)
        }

    def _aggregate_one_horizon(
        self,
        horizon: str,
        records: Sequence[dict],
        block_key: str = PER_HORIZON_KEY,
        counts_windows: bool = True,
    ) -> dict:
        """Aggregate one horizon's metrics across every seed.

        Args:
            horizon: The horizon key to aggregate.
            records: The loaded per-seed records.
            block_key: Which per-horizon block to read.
            counts_windows: Whether to require and summarise the window count.

        Returns:
            That horizon's block of aggregated metrics.

        Raises:
            SeriesIncompleteError: If a required key is absent, or a metric
                outside SAMPLING_NULLABLE_METRICS is undefined on some seeds
                but not all.
        """
        blocks = {
            record["seed"]: record["metrics"][block_key][horizon]
            for record in records
        }
        first = blocks[records[0]["seed"]]
        if counts_windows:
            required = required_per_horizon_keys(
                self._declared_error_metric(records)
            )
            missing = [key for key in required if key not in first]
            if missing:
                raise SeriesIncompleteError(
                    f"run '{self.run_name}' is missing {missing} at horizon "
                    f"{horizon} in '{block_key}' -- the reader and the writer "
                    f"disagree about the artefact's shape, so no number here "
                    f"is trustworthy."
                )

        aggregated: dict[str, Any] = {}
        for metric in first:
            if metric == WINDOWS_KEY:
                continue
            values = {seed: block.get(metric) for seed, block in blocks.items()}
            aggregated[metric] = self._aggregate_one_metric(
                metric, horizon, values
            )
        if counts_windows:
            # A range across seeds, not a mean.
            counts = [block[WINDOWS_KEY] for block in blocks.values()]
            aggregated[WINDOWS_RANGE_KEY] = {
                "min": min(counts),
                "max": max(counts),
            }
        return aggregated

    def _aggregate_one_metric(
        self, metric: str, horizon: str, values: dict[int, Any]
    ) -> dict | None:
        """Aggregate one metric at one horizon, or record it as inapplicable.

        None on every seed returns None. None on some seeds raises, except for
        SAMPLING_NULLABLE_METRICS, which give a thin cell. A non-numeric value,
        such as the smoothing report, returns None.

        Args:
            metric: The metric name.
            horizon: The horizon being aggregated, for the error message.
            values: Each seed's value for this metric.

        Returns:
            The aggregate block, a thin block, or None.

        Raises:
            SeriesIncompleteError: If a metric outside SAMPLING_NULLABLE_METRICS
                is undefined on some seeds but not all.
        """
        undefined = sorted(seed for seed, value in values.items() if value is None)
        if len(undefined) == len(values):
            return None
        if undefined and metric not in SAMPLING_NULLABLE_METRICS:
            raise SeriesIncompleteError(
                f"metric '{metric}' of run '{self.run_name}' is undefined on "
                f"seeds {undefined} at horizon {horizon} and defined on the "
                f"rest -- seeds disagreeing about which metrics exist are not "
                f"comparable, so averaging the defined ones would report a "
                f"partial series as a complete one."
            )
        if any(
            not isinstance(value, (int, float))
            for value in values.values()
            if value is not None
        ):
            return None
        if undefined:
            return self._aggregate_thin_cell(metric, horizon, values, undefined)
        return aggregate_scalar(
            [float(value) for value in values.values()],
            reps=self.reps,
            statistic_seed=self.statistic_seed,
        )

    def _aggregate_thin_cell(
        self,
        metric: str,
        horizon: str,
        values: dict[int, Any],
        undefined: list[int],
    ) -> dict:
        """Aggregate a sampling-nullable metric from the seeds that measured it.

        Below MIN_SEEDS_FOR_CELL the statistics are null and the cell reports
        itself unavailable. The values are keyed by seed either way.

        Args:
            metric: The metric name, for the log line.
            horizon: The horizon being aggregated, for the log line.
            values: Each seed's value for this metric, some of them None.
            undefined: The seeds whose value is None, sorted.

        Returns:
            The thin block, carrying n_seeds, the absent seeds and a reason.
        """
        defined = {
            seed: float(value)
            for seed, value in sorted(values.items())
            if value is not None
        }
        if len(defined) < MIN_SEEDS_FOR_CELL:
            block: dict[str, Any] = dict.fromkeys(UNAVAILABLE_STATISTICS)
            reason = BELOW_SEED_FLOOR
            logger.warning(
                "metric %s of run %s at horizon %s was measured on %d seed(s), "
                "below the floor of %d -- reported unavailable",
                metric,
                self.run_name,
                horizon,
                len(defined),
                MIN_SEEDS_FOR_CELL,
            )
        else:
            block = aggregate_scalar(
                list(defined.values()),
                reps=self.reps,
                statistic_seed=self.statistic_seed,
            )
            reason = SAMPLING_NULL_ON_SOME_SEEDS
        block[CELL_VALUES_KEY] = {
            str(seed): value for seed, value in defined.items()
        }
        block[CELL_SEEDS_KEY] = len(defined)
        block[CELL_ABSENT_SEEDS_KEY] = undefined
        block[CELL_REASON_KEY] = reason
        return block

    # -- policy --------------------------------------------------------------

    def _aggregate_policy(self, records: Sequence[dict]) -> dict | None:
        """Aggregate task-success metrics for each action rule.

        Returns None on every current run: nothing writes a policy artefact.

        Args:
            records: The loaded per-seed records.

        Returns:
            The policy block, or None when no seed trained a policy.

        Raises:
            UnusableSeedError: If some seeds carry a policy artefact and others
                do not, or if the seeds disagree on which arm is the headline.
        """
        with_policy = [record for record in records if record["policy"] is not None]
        if not with_policy:
            logger.info(
                "no policy artefact for any seed of run %s -- this series "
                "trained no policy, so it reports no task-success number",
                self.run_name,
            )
            return None
        if len(with_policy) != len(records):
            without = [
                record["seed"] for record in records if record["policy"] is None
            ]
            raise UnusableSeedError(
                f"seeds {without} of run '{self.run_name}' have no policy "
                f"artefact while their siblings do -- a series mixing "
                f"world-model-only and full runs cannot be aggregated on the "
                f"policy side."
            )

        headlines = {record["policy"]["headline_action_rule"] for record in records}
        if len(headlines) > 1:
            raise UnusableSeedError(
                f"seeds of run '{self.run_name}' disagree on the headline "
                f"action rule ({sorted(headlines)}) -- the headline must be "
                f"the same arm on every seed for the aggregate to mean "
                f"anything."
            )

        block: dict[str, Any] = {}
        omitted: dict[str, str] = {}
        for rule in sorted(records[0]["policy"]["final"]):
            named: dict[str, list[float | None]] = {
                metric: [
                    _policy_metric_value(record["policy"]["final"][rule], metric)
                    for record in records
                ]
                for metric in POLICY_METRICS
            }
            aggregated, arm_omitted = self._aggregate_named_scalars(named)
            block[rule] = aggregated
            omitted.update(
                {f"{rule}.{name}": reason for name, reason in arm_omitted.items()}
            )
        block["omitted_metrics"] = omitted
        for field in POLICY_PROVENANCE_FIELDS:
            block[field] = records[0]["policy"][field]
        return block

    # -- curves --------------------------------------------------------------

    def _aggregate_curves(self, records: Sequence[dict]) -> dict:
        """Aggregate the held-out fidelity curve and the loss series.

        Returns empty curves on every current run: nothing writes either.

        Args:
            records: The loaded per-seed records.

        Returns:
            The curves block of the aggregate artefact.

        Raises:
            UnusableSeedError: If the curves are sampled at different steps
                across seeds.
        """
        held_out = [record["metrics"].get("held_out_curve") or [] for record in records]
        loss = [
            record["metrics"].get("training_loss_curve") or [] for record in records
        ]
        seeds = [record["seed"] for record in records]
        return {
            "held_out": self._aggregate_curve(
                held_out,
                seeds,
                "held_out_curve",
                lambda point: horizon_mean(
                    point.get(HELD_OUT_CURVE_VALUE_FIELD)
                ),
            ),
            "training_loss": {
                series: self._aggregate_curve(
                    loss,
                    seeds,
                    f"training_loss_curve.{series}",
                    lambda point, key=series: point.get(key),
                )
                for series in TRAINING_LOSS_SERIES
            },
        }

    def _aggregate_curve(
        self,
        per_seed: Sequence[Sequence[dict]],
        seeds: Sequence[int],
        label: str,
        value_of: Callable[[dict], float | None],
    ) -> list[dict]:
        """Aggregate one curve across seeds, asserting step alignment first.

        Args:
            per_seed: Each seed's list of curve points.
            seeds: The seeds, in the same order, for error messages.
            label: Curve name, for error messages.
            value_of: Extracts the scalar of interest from one curve point.

        Returns:
            One {step, mean, std, n} entry per step, or an empty list when no
            seed carries the curve.

        Raises:
            UnusableSeedError: If the seeds' step grids differ.
        """
        if not any(per_seed):
            logger.info(
                "no %s in any seed of run %s -- the runs predate periodic "
                "evaluation or trained with it off",
                label,
                self.run_name,
            )
            return []
        self._assert_steps_align(per_seed, seeds, label)

        aggregated = []
        for index, step in enumerate([point["step"] for point in per_seed[0]]):
            values = [
                value
                for curve in per_seed
                if (value := value_of(curve[index])) is not None
            ]
            if not values:
                continue
            aggregated.append(
                {
                    "step": step,
                    "mean": float(sum(values) / len(values)),
                    "std": sample_std(values),
                    "n": len(values),
                }
            )
        return aggregated

    def _assert_steps_align(
        self,
        per_seed: Sequence[Sequence[dict]],
        seeds: Sequence[int],
        label: str,
    ) -> None:
        """Refuse curves sampled at different steps across seeds.

        Args:
            per_seed: Each seed's list of curve points.
            seeds: The seeds, in the same order.
            label: Curve name, for the error message.

        Raises:
            UnusableSeedError: If any seed's step grid differs from the first.
        """
        reference = [point["step"] for point in per_seed[0]]
        for seed, curve in zip(seeds[1:], per_seed[1:]):
            steps = [point["step"] for point in curve]
            if steps != reference:
                raise UnusableSeedError(
                    f"seed {seed} of run '{self.run_name}' sampled {label} at "
                    f"{len(steps)} steps against {len(reference)} on seed "
                    f"{seeds[0]}, or at different step values -- averaging "
                    f"misaligned curves produces a meaningless line."
                )

    # -- entry points --------------------------------------------------------

    def aggregate(self, allow_partial: bool = False) -> dict:
        """Build the aggregate payload, one series per discovered mode.

        Args:
            allow_partial: Waive the all-seeds-present requirement. CLI only.

        Returns:
            The aggregate artefact as a dict, keyed by observation mode under
            `by_mode`.

        Raises:
            SeriesIncompleteError: If no mode has artefacts at all, or if seeds
                are missing and allow_partial is not set.
            UnusableSeedError: If any present seed must not be reported.
        """
        modes = self.discovered_modes()
        if not modes:
            raise SeriesIncompleteError(
                f"no metrics file matching {METRICS_GLOB} under "
                f"{safe_rel(self.series_dir)} for run '{self.run_name}' -- "
                "nothing to aggregate."
            )
        return {
            "run_name": self.run_name,
            "observation_modes": list(modes),
            "by_mode": {
                mode: self._aggregate_one_mode(mode, allow_partial=allow_partial)
                for mode in modes
            },
        }

    def _aggregate_one_mode(self, mode: str, allow_partial: bool = False) -> dict:
        """Build one observation mode's cross-seed series.

        Args:
            mode: The observation mode to aggregate.
            allow_partial: Waive the all-seeds-present requirement. CLI only.

        Returns:
            That mode's series block.

        Raises:
            SeriesIncompleteError: If seeds are missing and allow_partial is
                not set.
            UnusableSeedError: If any present seed must not be reported.
        """
        records = self._load_seed_artefacts(mode, allow_partial=allow_partial)
        seeds = [record["seed"] for record in records]
        per_horizon = self._aggregate_per_horizon(records)
        logger.info(
            "aggregating run %s in %s mode over %d seed(s) %s from %s",
            self.run_name,
            mode,
            len(seeds),
            seeds,
            safe_rel(self.series_dir),
        )
        return {
            "run_name": self.run_name,
            "observation_mode": mode,
            "seeds": seeds,
            # The real seed count, below the protocol under --allow-partial.
            "n_seeds": len(seeds),
            "per_seed_completed_steps": {
                str(record["seed"]): record["metrics"].get("completed_steps")
                for record in records
            },
            "config": records[0]["metrics"].get("config"),
            ERROR_METRIC_KEY: self._declared_error_metric(records),
            "statistic": {
                "iqm_reps": self.reps,
                "iqm_seed": self.statistic_seed,
                "confidence_interval_size": CONFIDENCE_INTERVAL_SIZE,
                "std_ddof": STD_DDOF,
            },
            PER_HORIZON_KEY: per_horizon,
            OMITTED_METRICS_KEY: {},
            THIN_CELLS_KEY: self._thin_cells(mode, per_horizon),
            "policy": self._aggregate_policy(records),
            "curves": self._aggregate_curves(records),
            **self._aggregate_parameter_trees(records),
        }

    def _aggregate_parameter_trees(self, records: Sequence[dict]) -> dict:
        """Aggregate the secondary parameter tree and the gap, when present.

        Only the test pass writes them. A block present on some seeds and not
        others raises.

        Args:
            records: The loaded per-seed records.

        Returns:
            The two aggregated blocks, or an empty mapping when no seed carries
            them.

        Raises:
            SeriesIncompleteError: If some seeds carry a block and others do
                not.
        """
        aggregated: dict[str, Any] = {}
        for block_key, counts_windows in (
            (PER_HORIZON_FINAL_STEP_KEY, True),
            (PER_HORIZON_GAP_KEY, False),
        ):
            present = sorted(
                record["seed"]
                for record in records
                if record["metrics"].get(block_key)
            )
            if not present:
                continue
            if len(present) != len(records):
                absent = sorted(
                    record["seed"]
                    for record in records
                    if not record["metrics"].get(block_key)
                )
                raise SeriesIncompleteError(
                    f"run '{self.run_name}' carries '{block_key}' on seeds "
                    f"{present} and not on {absent} -- a block reported for "
                    f"some seeds cannot be aggregated as a series."
                )
            aggregated[block_key] = self._aggregate_per_horizon(
                records, block_key, counts_windows
            )
        return aggregated

    @staticmethod
    def _thin_cells(mode: str, per_horizon: dict) -> list[dict]:
        """List every cell built from fewer than every seed.

        Args:
            mode: The observation mode these cells belong to.
            per_horizon: The aggregated per-horizon blocks.

        Returns:
            One record per thin cell, empty where every cell used every seed.
        """
        return [
            {
                "observation_mode": mode,
                "horizon": horizon,
                "metric": metric,
                CELL_SEEDS_KEY: block[CELL_SEEDS_KEY],
                CELL_ABSENT_SEEDS_KEY: block[CELL_ABSENT_SEEDS_KEY],
                CELL_REASON_KEY: block[CELL_REASON_KEY],
            }
            for horizon, metrics in per_horizon.items()
            for metric, block in metrics.items()
            if isinstance(block, dict) and CELL_REASON_KEY in block
        ]

    def write(self, payload: dict) -> Path:
        """Write the aggregate artefact to disk and return its path.

        Args:
            payload: The aggregate produced by aggregate().

        Returns:
            The path written.
        """
        path = self.aggregate_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("cross-seed aggregate written to %s", safe_rel(path))
        return path

    def run(self, allow_partial: bool = False) -> Path:
        """Aggregate this series and write the artefact.

        Args:
            allow_partial: Waive the all-seeds-present requirement. CLI only.

        Returns:
            The path written.
        """
        return self.write(self.aggregate(allow_partial=allow_partial))





def _policy_metric_value(arm: dict, metric: str) -> float | None:
    """Return one policy metric from one arm, mapping sentinels to None.

    mean_successful_episode_length is written as 0.0 when no episode
    succeeded, where 0.0 means undefined, and averaging that would drag the
    aggregate toward zero. Detected from the episode_success array, not from
    the value, so a genuine zero cannot be confused with it.

    Args:
        arm: One action rule's result block from policy_metrics.json.
        metric: The metric name.

    Returns:
        The value, or None where it is undefined.
    """
    value = arm.get(metric)
    if metric == "mean_successful_episode_length" and not any(
        arm.get("episode_success") or []
    ):
        return None
    return None if value is None else float(value)


def aggregate_if_series_complete(
    config: ExperimentConfig, arm: int = ARM_DIRECT
) -> Path | None:
    """Aggregate this run's seeds if every one of them has finished.

    Called by the runner after its stages. An incomplete series returns None
    and leaves any existing aggregate as it was. Failures are logged and
    swallowed, so re-run with `--aggregate` to see them raised.

    Args:
        config: The run's experiment configuration.
        arm: Which arm was trained and evaluated, one of config.ARMS.

    Returns:
        The path written, or None when the series is incomplete or the
        aggregation failed.
    """
    aggregator = CrossSeedAggregator(
        config.run_name, arm=arm, env=config.env.name, fast=config.fast
    )
    if not aggregator.series_is_complete():
        modes = aggregator.discovered_modes()
        logger.info(
            "run %s is not yet a complete series -- %s against %d configured "
            "seeds per mode; skipping cross-seed aggregation",
            aggregator.run_name,
            (
                ", ".join(
                    f"{mode}: {len(aggregator.present_seeds(mode))}"
                    for mode in modes
                )
                or "no metrics file found"
            ),
            len(aggregator.seeds),
        )
        return None
    try:
        return aggregator.run()
    except (OSError, ValueError, KeyError) as error:
        logger.error(
            "cross-seed aggregation failed for run %s and was swallowed so it "
            "cannot fail this seed's pipeline -- re-run it with "
            "`python main.py --aggregate --run-name %s` once fixed: %s",
            aggregator.run_name,
            aggregator.run_name,
            error,
        )
        return None


def run_aggregation_cli(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_name: str | None,
    series_dir: Path | None,
    allow_partial: bool,
    arm: int = ARM_DIRECT,
    env: str | None = None,
    fast: bool = False,
) -> Path:
    """Aggregate a series from the command line and return the artefact path.

    Raises on an incomplete series, where the auto-trigger skips.

    Args:
        run_name: The series-level run name. When None, it is taken from
            series_dir's own name.
        series_dir: Directory holding the seed-scoped run directories.
        allow_partial: Aggregate a below-protocol series.
        arm: Which arm's series to aggregate, one of config.ARMS.
        env: Registered environment name, a directory level under the run
            name. Ignored when series_dir is given.
        fast: Whether the series sits under `outputs/fast/`. Ignored when
            series_dir is given.

    Returns:
        The path written.

    Raises:
        ValueError: If neither run_name nor series_dir is supplied.
    """
    if run_name is None and series_dir is None:
        raise ValueError(
            "--aggregate needs --run-name, or --series-dir to infer it from. "
            "outputs/eval/ holds many unrelated runs, so a run name cannot be "
            "inferred there."
        )
    resolved = run_name if run_name is not None else infer_run_name(series_dir)
    if run_name is None:
        logger.info(
            "inferred run name '%s' from the seed directories under %s",
            resolved,
            safe_rel(series_dir),
        )
    if allow_partial:
        logger.warning(
            "--allow-partial is set: a series below the five-seed reporting "
            "protocol will be aggregated, and its real seed count is recorded "
            "in the artefact"
        )
    return CrossSeedAggregator(
        resolved, series_dir=series_dir, arm=arm, env=env, fast=fast
    ).run(allow_partial=allow_partial)
