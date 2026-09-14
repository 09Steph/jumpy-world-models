"""Cross-arm error against horizon, read from the per-arm aggregates.

Puts every arm's per-horizon error on one axis, with the endpoint error ratio
and the compounding error per arm, in one artefact the fit reads. Arms may be
read from different runs. Fits nothing and writes no aggregate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import (
    ARMS,
    COMPOUNDING_DISCOUNT,
    DEFAULT_ENV_NAME,
    EXTRAPOLATION_HORIZONS,
    PROBABILITY_LOG_GLOB,
    REPORTED_RATIO_HORIZONS,
    SWEEP_FILENAME,
)
from src.pipeline.aggregate import (
    AGGREGATE_TEMPLATE,
    WINDOWS_RANGE_KEY,
    CrossSeedAggregator,
    infer_run_name,
)
from src.pipeline.aggregate_stats import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CONFIDENCE_INTERVAL_SIZE,
    aggregate_scalar,
)
from src.pipeline.metrics_schema import (
    COPY_SIGMA_DISCOUNTED_INTEGRAL_KEY,
    COPY_SIGMA_INTEGRAL_KEY,
    COPY_SIGMA_SUM_KEY,
    ERROR_METRIC_CROSS_ENTROPY,
    ERROR_METRIC_KEY,
    PER_HORIZON_KEY,
    SIGMA_DISCOUNTED_INTEGRAL_KEY,
    SIGMA_INTEGRAL_KEY,
    SIGMA_KEY,
    SIGMA_SKILL_KEY,
    SIGMA_SUM_KEY,
    SKILL_SCORE_KEY,
    copy_error_key,
    model_error_key,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import safe_rel

logger = get_logger(__name__)

def sweep_metrics(error_metric: str) -> tuple[str, ...]:
    """Return the per-horizon metrics the sweep carries for one error metric.

    Args:
        error_metric: The value the aggregates declare under ERROR_METRIC_KEY.

    Returns:
        The model error key and the skill score.
    """
    return (model_error_key(error_metric), SKILL_SCORE_KEY)


RATIO_KEY: str = "endpoint_error_ratio"

# The same ratio read out to the widest extrapolation horizon, kept apart from
# RATIO_KEY.
EXTRAPOLATION_RATIO_KEY: str = "extrapolation_error_ratio"

# Where the EXTRAPOLATION_HORIZONS members are carried. The fit and the
# compounding error read only "horizons", so a member is never fitted.
# Membership decides, not the trained range: on the long-horizon environment
# these horizons lie inside it.
EXTRAPOLATION_HORIZONS_KEY: str = "extrapolation_horizons"

# Recorded where an arm's aggregate carries no params_selection.
UNKNOWN_PARAMS_SELECTION: str = "unknown"

PARAMS_SELECTION_KEY: str = "params_selection"

# Fewest horizons the compounding error is read over.
MIN_SIGMA_HORIZONS: int = 2

# The compounding error's readings, in the order compounding_readings returns
# them, for the model and for the copy baseline.
SIGMA_READING_KEYS: tuple[str, str, str] = (
    SIGMA_SUM_KEY,
    SIGMA_INTEGRAL_KEY,
    SIGMA_DISCOUNTED_INTEGRAL_KEY,
)
COPY_SIGMA_READING_KEYS: tuple[str, str, str] = (
    COPY_SIGMA_SUM_KEY,
    COPY_SIGMA_INTEGRAL_KEY,
    COPY_SIGMA_DISCOUNTED_INTEGRAL_KEY,
)


class CrossArmSweep:  # pylint: disable=too-many-instance-attributes
    """Combine per-arm aggregates into one cross-arm horizon artefact.

    An arm without an aggregate is skipped and recorded in the flags.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        run_name: str,
        arm_runs: dict[int, str] | None = None,
        series_dir: Path | None = None,
        env: str | None = None,
        fast: bool = False,
        reps: int | None = None,
        statistic_seed: int | None = None,
    ) -> None:
        """Store which run each arm is read from and where the artefact lands.

        Args:
            run_name: The run every arm is read from unless overridden.
            arm_runs: Per-arm run name overrides, keyed by arm. An arm absent
                from this mapping falls back to run_name.
            series_dir: Directory holding run_name's per-seed directories.
                Used only for arms not named in arm_runs.
            env: Registered environment name, a directory level under the run.
            fast: Whether the series sits under the fast tree.
            reps: Bootstrap resamples for the ratio's interval. See
                aggregate_stats.BOOTSTRAP_REPS.
            statistic_seed: Seed for the bootstrap's random state.

        Raises:
            ValueError: If arm_runs names an arm outside config.ARMS.
        """
        unknown = sorted(set(arm_runs or {}) - set(ARMS))
        if unknown:
            raise ValueError(
                f"unknown arm(s) {unknown} in the run mapping, expected one "
                f"of {ARMS}"
            )
        self.run_name = run_name
        self.arm_runs = dict(arm_runs or {})
        self.env = env or DEFAULT_ENV_NAME
        self.fast = fast
        self._series_dir = series_dir
        self.reps = BOOTSTRAP_REPS if reps is None else reps
        self.statistic_seed = (
            BOOTSTRAP_SEED if statistic_seed is None else statistic_seed
        )
        self.flags: list[str] = []
        # sweep() sets this from what the aggregates declare. An absent
        # declaration raises there, so this value is never read.
        self.error_metric: str = ERROR_METRIC_CROSS_ENTROPY

    # -- paths ---------------------------------------------------------------

    def arm_run(self, arm: int) -> str:
        """Return the run name one arm is read from."""
        return self.arm_runs.get(arm, self.run_name)

    def series_dir(self, arm: int) -> Path:
        """Return the directory holding one arm's per-seed directories.

        An arm named in arm_runs resolves through its own run name. Any other
        arm uses series_dir when one was given.

        Args:
            arm: The arm to locate.

        Returns:
            That arm's series directory.
        """
        if arm not in self.arm_runs and self._series_dir is not None:
            return self._series_dir
        return CrossSeedAggregator(
            self.arm_run(arm), env=self.env, fast=self.fast, arm=arm
        ).series_dir

    def aggregate_path(self, arm: int) -> Path:
        """Return where one arm's aggregate is expected."""
        return self.series_dir(arm) / AGGREGATE_TEMPLATE.format(arm=arm)

    def sweep_path(self) -> Path:
        """Return where the cross-arm artefact is written.

        Beside the requested run's aggregates, even when an arm is read from
        another run.
        """
        base = self._series_dir
        if base is None:
            base = CrossSeedAggregator(
                self.run_name, env=self.env, fast=self.fast
            ).series_dir
        return base / SWEEP_FILENAME

    def probability_log_paths(
        self, arm: int, mode: str, seeds: Sequence[int]
    ) -> list[str]:
        """Return the per-seed probability-log paths for one arm and mode.

        Found by glob and filtered on the mode suffix.

        Args:
            arm: The arm whose logs are wanted.
            mode: The observation mode to filter to.
            seeds: The seeds the aggregate reported.

        Returns:
            Repository-relative paths, omitting seeds whose log is absent.
        """
        aggregator = CrossSeedAggregator(
            self.arm_run(arm),
            series_dir=self.series_dir(arm),
            env=self.env,
            fast=self.fast,
            arm=arm,
        )
        found = []
        for seed in seeds:
            matches = sorted(
                path
                for path in aggregator.seed_dir(seed).glob(PROBABILITY_LOG_GLOB)
                if path.stem.endswith(mode)
            )
            found.extend(safe_rel(path) for path in matches)
        return found

    # -- loading -------------------------------------------------------------

    def load_arms(self) -> dict[int, dict]:
        """Load every arm whose aggregate exists, naming the ones that do not.

        Returns:
            {arm: aggregate payload} for the arms present.
        """
        loaded = {}
        for arm in ARMS:
            path = self.aggregate_path(arm)
            if not path.is_file():
                message = (
                    f"arm {arm} has no aggregate at {safe_rel(path)} and is "
                    f"absent from the sweep"
                )
                logger.warning("%s", message)
                self.flags.append(message)
                continue
            loaded[arm] = json.loads(path.read_text(encoding="utf-8"))
        return loaded

    def _declared_error_metric(self, loaded: dict[int, dict]) -> str:
        """Return the error metric the loaded aggregates declare.

        Read from the per-mode blocks under `by_mode`, falling back to an
        aggregate's top level where none of its modes declares one.

        Args:
            loaded: {arm: aggregate payload} for the arms present.

        Returns:
            The declared metric.

        Raises:
            ValueError: If the arms or modes disagree, or if no aggregate
                declares one.
        """
        declared: set[str] = set()
        for payload in loaded.values():
            per_mode = {
                block.get(ERROR_METRIC_KEY)
                for block in (payload.get("by_mode") or {}).values()
            } - {None}
            declared |= per_mode or {payload.get(ERROR_METRIC_KEY)} - {None}
        if not declared:
            raise ValueError(
                f"no aggregate of run '{self.run_name}' declares "
                f"{ERROR_METRIC_KEY}, under 'by_mode' or at the top level. "
                f"Re-run the aggregate before the sweep, since every key the "
                f"sweep selects is resolved from that one value."
            )
        if len(declared) > 1:
            raise ValueError(
                f"arms or modes of run '{self.run_name}' disagree about "
                f"{ERROR_METRIC_KEY}: {sorted(declared)}. They were scored "
                f"under different error metrics, so no cross-arm curve or "
                f"ratio over them is readable."
            )
        return declared.pop()

    # -- assembly ------------------------------------------------------------

    def _provenance(self, arm: int, mode_block: dict) -> dict:
        """Return one arm's provenance for one mode.

        The parameter selection is read per mode, since an arm's modes can be
        scored under different selections.

        Args:
            arm: The arm being described.
            mode_block: That arm's block for this observation mode.

        Returns:
            The provenance fields recorded against this arm.
        """
        path = self.aggregate_path(arm)
        selection = (mode_block.get("config") or {}).get(PARAMS_SELECTION_KEY)
        if selection is None:
            selection = UNKNOWN_PARAMS_SELECTION
            message = (
                f"arm {arm} in {mode_block.get('observation_mode')} reports no "
                f"{PARAMS_SELECTION_KEY}: its evaluation predates the field, "
                f"so which parameter tree produced these numbers is not "
                f"recorded in the artefact"
            )
            logger.warning("%s", message)
            self.flags.append(message)
        return {
            "run_name": mode_block.get("run_name", self.arm_run(arm)),
            "aggregate_mtime": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            PARAMS_SELECTION_KEY: selection,
        }

    def _ratio(self, mode_block: dict) -> dict | None:
        """Return the endpoint error ratio for one arm and mode.

        Anchored at REPORTED_RATIO_HORIZONS, never at the block's first and
        last horizons.

        Args:
            mode_block: One arm's block for one observation mode.

        Returns:
            The ratio block, or None when it cannot be computed.
        """
        low, high = REPORTED_RATIO_HORIZONS
        return self._ratio_between(mode_block, str(low), str(high))

    def _extrapolation_ratio(self, mode_block: dict) -> dict | None:
        """Return the error ratio out to the widest extrapolation horizon.

        From REPORTED_RATIO_HORIZONS[0] to the widest EXTRAPOLATION_HORIZONS
        member present.

        Args:
            mode_block: One arm's block for one observation mode.

        Returns:
            The ratio block, or None when it cannot be computed.
        """
        present = [
            horizon
            for horizon in sorted_horizons(mode_block.get(PER_HORIZON_KEY, {}))
            if int(horizon) in EXTRAPOLATION_HORIZONS
        ]
        if not present:
            return None
        return self._ratio_between(
            mode_block, str(REPORTED_RATIO_HORIZONS[0]), present[-1]
        )

    def _ratio_between(
        self, mode_block: dict, low: str, high: str
    ) -> dict | None:
        """Return the error ratio between two named horizons.

        Computed per seed, pairing the two horizons' values by position, then
        aggregated over seeds. Both endpoints are named in the result. A zero
        denominator on any seed returns None without a flag.

        Args:
            mode_block: One arm's block for one observation mode.
            low: The denominator's horizon key.
            high: The numerator's horizon key.

        Returns:
            The ratio block, or None when it cannot be computed.
        """
        per_horizon = mode_block.get(PER_HORIZON_KEY, {})
        if low not in per_horizon or high not in per_horizon:
            return None
        model_key = model_error_key(self.error_metric)
        numerator = (per_horizon[high].get(model_key) or {}).get("values")
        denominator = (per_horizon[low].get(model_key) or {}).get("values")
        if not numerator or not denominator:
            return None
        if len(numerator) != len(denominator):
            message = (
                f"{mode_block.get('observation_mode')} carries "
                f"{len(numerator)} seeds at horizon {high} and "
                f"{len(denominator)} at horizon {low}, so no per-seed ratio "
                f"can be formed"
            )
            logger.warning("%s", message)
            self.flags.append(message)
            return None
        if any(value == 0 for value in denominator):
            return None
        block = aggregate_scalar(
            [
                high_value / low_value
                for high_value, low_value in zip(numerator, denominator)
            ],
            reps=self.reps,
            statistic_seed=self.statistic_seed,
        )
        block["numerator_horizon"] = high
        block["denominator_horizon"] = low
        block["metric"] = model_error_key(self.error_metric)
        return block

    def _sigma(self, mode_block: dict) -> dict | None:
        """Return the compounding error for one arm and mode.

        Over every horizon outside EXTRAPOLATION_HORIZONS. Per seed, for the
        model and the copy baseline: the plain sum, the trapezoidal integral
        over h, and the integral weighted by COMPOUNDING_DISCOUNT ** h, each
        aggregated across seeds. The skill score is 1 - model integral / copy
        integral, per seed.

        Args:
            mode_block: One arm's block for one observation mode.

        Returns:
            The compounding-error block, or None when the curves cannot be
            read.
        """
        per_horizon = mode_block.get(PER_HORIZON_KEY, {})
        horizons = [
            horizon
            for horizon in sorted_horizons(per_horizon)
            if int(horizon) not in EXTRAPOLATION_HORIZONS
        ]
        if len(horizons) < MIN_SIGMA_HORIZONS:
            return None
        model_curves = self._seed_curves(
            mode_block, horizons, model_error_key(self.error_metric)
        )
        copy_curves = self._seed_curves(
            mode_block, horizons, copy_error_key(self.error_metric)
        )
        if model_curves is None or copy_curves is None:
            return None
        if len(model_curves) != len(copy_curves):
            message = (
                f"{mode_block.get('observation_mode')} carries "
                f"{len(model_curves)} model seeds and {len(copy_curves)} copy "
                f"seeds, so no per-seed compounding error can be formed"
            )
            logger.warning("%s", message)
            self.flags.append(message)
            return None
        axis = [int(horizon) for horizon in horizons]
        per_seed = {
            **dict(zip(SIGMA_READING_KEYS, compounding_readings(axis, model_curves))),
            **dict(
                zip(COPY_SIGMA_READING_KEYS, compounding_readings(axis, copy_curves))
            ),
        }
        block: dict[str, Any] = {
            key: aggregate_scalar(
                values, reps=self.reps, statistic_seed=self.statistic_seed
            )
            for key, values in per_seed.items()
        }
        model_integral = per_seed[SIGMA_INTEGRAL_KEY]
        copy_integral = per_seed[COPY_SIGMA_INTEGRAL_KEY]
        block[SIGMA_SKILL_KEY] = (
            aggregate_scalar(
                [
                    1.0 - model / copy
                    for model, copy in zip(model_integral, copy_integral)
                ],
                reps=self.reps,
                statistic_seed=self.statistic_seed,
            )
            if all(copy_integral)
            else None
        )
        block["horizons"] = horizons
        block["discount"] = COMPOUNDING_DISCOUNT
        block["metric"] = model_error_key(self.error_metric)
        return block

    def _seed_curves(
        self, mode_block: dict, horizons: list[str], key: str
    ) -> list[list[float]] | None:
        """Return one error curve per seed over the named horizons.

        Args:
            mode_block: One arm's block for one observation mode.
            horizons: The horizon keys to read, in order.
            key: The per-horizon metric to read.

        Returns:
            One list per seed, one value per horizon, or None when a horizon
            lacks the metric or the seed counts disagree.
        """
        per_horizon = mode_block.get(PER_HORIZON_KEY, {})
        columns = [
            (per_horizon[horizon].get(key) or {}).get("values")
            for horizon in horizons
        ]
        if not all(columns):
            return None
        if len({len(column) for column in columns}) > 1:
            message = (
                f"{mode_block.get('observation_mode')} carries different seed "
                f"counts for {key} across horizons {horizons}, so no per-seed "
                f"compounding error can be formed"
            )
            logger.warning("%s", message)
            self.flags.append(message)
            return None
        return [list(curve) for curve in zip(*columns)]

    def _horizon_entry(self, mode: str, horizon: str, arms: dict[int, dict]) -> dict:
        """Build one horizon's entry for one observation mode.

        The copy error and window range are read from the first arm present,
        which is named, and every other arm is checked against them.

        Args:
            mode: The observation mode.
            horizon: The horizon key.
            arms: {arm: that arm's block for this mode}.

        Returns:
            The horizon entry.
        """
        entry: dict[str, Any] = {"arms": {}}
        source_arm = None
        for arm, mode_block in sorted(arms.items()):
            block = mode_block.get(PER_HORIZON_KEY, {}).get(horizon)
            if block is None:
                continue
            entry["arms"][str(arm)] = {
                metric: block.get(metric)
                for metric in sweep_metrics(self.error_metric)
            }
            if source_arm is None:
                source_arm = arm
                copy_key = copy_error_key(self.error_metric)
                entry[copy_key] = block.get(copy_key)
                entry[WINDOWS_RANGE_KEY] = block.get(WINDOWS_RANGE_KEY)
                continue
            self._check_shared_field(mode, horizon, arm, source_arm, block, entry)
        entry["baseline_source_arm"] = None if source_arm is None else str(source_arm)
        return entry

    def _check_shared_field(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        mode: str,
        horizon: str,
        arm: int,
        source_arm: int,
        block: dict,
        entry: dict,
    ) -> None:
        """Flag an arm whose copy error or window range differs from the source arm's.

        The difference is flagged and the sweep still completes.

        Args:
            mode: The observation mode, for the message.
            horizon: The horizon key, for the message.
            arm: The arm being checked.
            source_arm: The arm the carried values came from.
            block: That arm's per-horizon block.
            entry: The horizon entry holding the carried values.
        """
        for field in (copy_error_key(self.error_metric), WINDOWS_RANGE_KEY):
            if block.get(field) != entry.get(field):
                message = (
                    f"arm {arm} disagrees with arm {source_arm} about {field} "
                    f"at horizon {horizon} in {mode}: the arms were not "
                    f"evaluated on the same windows"
                )
                logger.warning("%s", message)
                self.flags.append(message)

    def _mode_entry(self, mode: str, arms: dict[int, dict]) -> dict:
        """Build one observation mode's cross-arm block.

        `horizons` holds every horizon outside EXTRAPOLATION_HORIZONS and is
        what the fit reads. `extrapolation_horizons` holds the members.

        Args:
            mode: The observation mode.
            arms: {arm: that arm's block for this mode}.

        Returns:
            That mode's block.
        """
        horizons = sorted(
            {
                horizon
                for mode_block in arms.values()
                for horizon in mode_block.get(PER_HORIZON_KEY, {})
            },
            key=int,
        )
        reported = [h for h in horizons if int(h) not in EXTRAPOLATION_HORIZONS]
        extrapolated = [h for h in horizons if int(h) in EXTRAPOLATION_HORIZONS]
        return {
            "observation_mode": mode,
            "arms_present": [str(arm) for arm in sorted(arms)],
            "arms": {
                str(arm): {
                    "provenance": self._provenance(arm, mode_block),
                    "seeds": mode_block.get("seeds"),
                    "n_seeds": mode_block.get("n_seeds"),
                    RATIO_KEY: self._ratio(mode_block),
                    EXTRAPOLATION_RATIO_KEY: self._extrapolation_ratio(
                        mode_block
                    ),
                    SIGMA_KEY: self._sigma(mode_block),
                    "probability_logs": self.probability_log_paths(
                        arm, mode, mode_block.get("seeds") or []
                    ),
                }
                for arm, mode_block in sorted(arms.items())
            },
            "horizons": {
                horizon: self._horizon_entry(mode, horizon, arms)
                for horizon in reported
            },
            EXTRAPOLATION_HORIZONS_KEY: {
                horizon: self._horizon_entry(mode, horizon, arms)
                for horizon in extrapolated
            },
        }

    def sweep(self) -> dict:
        """Build the cross-arm payload.

        Returns:
            The sweep artefact as a dict.

        Raises:
            FileNotFoundError: If no arm has an aggregate at all.
            ValueError: If the aggregates declare no error metric, or disagree
                about it.
        """
        self.flags = []
        loaded = self.load_arms()
        if not loaded:
            raise FileNotFoundError(
                f"no arm of run '{self.run_name}' has an aggregate under "
                f"{safe_rel(self.series_dir(ARMS[0]))} -- aggregate at least "
                f"one arm before sweeping."
            )
        self.error_metric = self._declared_error_metric(loaded)
        by_arm_mode = {
            arm: payload.get("by_mode", {}) for arm, payload in loaded.items()
        }
        modes = sorted({mode for blocks in by_arm_mode.values() for mode in blocks})
        return {
            "run_name": self.run_name,
            ERROR_METRIC_KEY: self.error_metric,
            "arm_runs": {str(arm): self.arm_run(arm) for arm in sorted(loaded)},
            "arms_present": [str(arm) for arm in sorted(loaded)],
            "arms_absent": [str(arm) for arm in ARMS if arm not in loaded],
            "observation_modes": modes,
            # Settings for the ratios and the compounding error, the only
            # quantities resampled here.
            "ratio_statistic": {
                "reps": self.reps,
                "seed": self.statistic_seed,
                "confidence_interval_size": CONFIDENCE_INTERVAL_SIZE,
            },
            "by_mode": {
                mode: self._mode_entry(
                    mode,
                    {
                        arm: blocks[mode]
                        for arm, blocks in sorted(by_arm_mode.items())
                        if mode in blocks
                    },
                )
                for mode in modes
            },
            "flags": list(self.flags),
        }

    def write(self, payload: dict) -> Path:
        """Write the sweep artefact and return its path.

        Args:
            payload: The payload produced by sweep().

        Returns:
            The path written.
        """
        path = self.sweep_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("cross-arm sweep written to %s", safe_rel(path))
        return path

    def run(self) -> Path:
        """Build the sweep and write it.

        Returns:
            The path written.
        """
        payload = self.sweep()
        path = self.write(payload)
        logger.info(
            "swept arms %s over %d observation mode(s); %d flag(s) recorded",
            ", ".join(payload["arms_present"]) or "none",
            len(payload["observation_modes"]),
            len(payload["flags"]),
        )
        return path


def sorted_horizons(per_horizon: dict) -> list[str]:
    """Return a per-horizon block's keys in numeric order.

    Args:
        per_horizon: A per-horizon mapping keyed by horizon string.

    Returns:
        The horizon keys sorted by value rather than lexically.
    """
    return sorted(per_horizon, key=int)


def trapezoid(axis: Sequence[float], values: Sequence[float]) -> float:
    """Return the trapezoidal integral of values over an unevenly spaced axis.

    Args:
        axis: The sample points, ascending.
        values: One value per sample point.

    Returns:
        The integral from the first sample point to the last.
    """
    return sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for left_x, right_x, left_y, right_y in zip(
            axis, axis[1:], values, values[1:]
        )
    )


def compounding_readings(
    axis: Sequence[int], curves: Sequence[Sequence[float]]
) -> tuple[list[float], list[float], list[float]]:
    """Return each curve's plain sum, integral and discounted integral over h.

    Args:
        axis: The horizons, ascending.
        curves: One error curve per seed, one value per horizon.

    Returns:
        Three lists with one value per curve, in SIGMA_READING_KEYS order.
    """
    weights = [COMPOUNDING_DISCOUNT**horizon for horizon in axis]
    return (
        [sum(curve) for curve in curves],
        [trapezoid(axis, curve) for curve in curves],
        [
            trapezoid(axis, [weight * error for weight, error in zip(weights, curve)])
            for curve in curves
        ],
    )


def parse_arm_runs(pairs: Sequence[str] | None) -> dict[int, str]:
    """Parse repeated `N=RUN_NAME` arguments into a per-arm run mapping.

    Args:
        pairs: The raw `--arm-run` values.

    Returns:
        {arm: run name}.

    Raises:
        ValueError: If a value is not `N=RUN_NAME` with an integer arm.
    """
    mapping = {}
    for pair in pairs or []:
        arm, separator, name = pair.partition("=")
        if not separator or not name or not arm.strip().isdigit():
            raise ValueError(
                f"--arm-run expects N=RUN_NAME, got {pair!r}"
            )
        mapping[int(arm)] = name
    return mapping


def run_sweep_cli(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_name: str | None,
    series_dir: Path | None,
    arm_runs: Sequence[str] | None = None,
    env: str | None = None,
    fast: bool = False,
) -> Path:
    """Sweep a run's arms from the command line and return the artefact path.

    Args:
        run_name: The run every arm is read from unless overridden. When None,
            it is taken from series_dir's own name.
        series_dir: Directory holding the fallback run's seed directories.
        arm_runs: Repeated `N=RUN_NAME` per-arm overrides.
        env: Registered environment name.
        fast: Whether the series sits under the fast tree.

    Returns:
        The path written.

    Raises:
        ValueError: If neither run_name nor series_dir is supplied.
    """
    if run_name is None and series_dir is None:
        raise ValueError(
            "--sweep needs --run-name, or --series-dir to infer it from. "
            "outputs/ holds many unrelated runs, so a run name cannot be "
            "inferred without one of them."
        )
    resolved = run_name if run_name is not None else infer_run_name(series_dir)
    mapping = parse_arm_runs(arm_runs)
    if mapping:
        logger.info(
            "reading %s from another run: %s",
            "arm" if len(mapping) == 1 else "arms",
            ", ".join(f"arm {arm} from {name}" for arm, name in sorted(mapping.items())),
        )
    return CrossArmSweep(
        resolved,
        arm_runs=mapping,
        series_dir=series_dir,
        env=env,
        fast=fast,
    ).run()
