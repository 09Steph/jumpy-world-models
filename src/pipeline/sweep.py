"""Cross-arm error against horizon, read from the per-arm aggregates.

Puts every arm on one horizon axis so error against horizon can be compared
across them. Aggregation is per arm and writes one file each; this reads those
files and writes one artefact carrying all of them.

Reads, never writes, the per-arm aggregates. Fits nothing: the power-law fit
and the usable-horizon diagnostic read this artefact rather than living here.

Arms may come from different runs. Arm 1's checkpoints are reused across
experiments and the checkpoint directory follows the run name, so the arms of
one comparison need not share a directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from config import (
    ARMS,
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
    ERROR_METRIC_CROSS_ENTROPY,
    ERROR_METRIC_KEY,
    PER_HORIZON_KEY,
    SKILL_SCORE_KEY,
    copy_error_key,
    model_error_key,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import safe_rel

logger = get_logger(__name__)

def sweep_metrics(error_metric: str) -> tuple[str, ...]:
    """Return the per-horizon metrics the sweep carries for one error metric.

    The error the ratio and the horizon axis are read on, resolved from the
    metric the aggregates declare. The fit reads the model series under
    whichever key that metric writes to, so naming one here would carry
    cross-entropy onto a run that never scored it.

    Args:
        error_metric: The value the aggregates declare under ERROR_METRIC_KEY.

    Returns:
        The model error key and the skill score that normalises it.
    """
    return (model_error_key(error_metric), SKILL_SCORE_KEY)


RATIO_KEY: str = "endpoint_error_ratio"

# The same ratio read across the extrapolation horizons, under its own name.
# A separate result, never folded into RATIO_KEY: horizons past the trained
# ceiling test extrapolation of the jump function, which is a different question
# from horizon robustness inside the trained range, and one number cannot answer
# both.
EXTRAPOLATION_RATIO_KEY: str = "extrapolation_error_ratio"

# Where the horizons past the reporting grid are carried. The fit reads
# "horizons" and nothing else, so keeping these apart is what stops the reported
# exponent being fitted over out-of-distribution points.
EXTRAPOLATION_HORIZONS_KEY: str = "extrapolation_horizons"

# Recorded where an arm's aggregate predates the field. Never raised: an arm
# evaluated before the selection existed is a fact about the artefact, and
# refusing here would make the whole sweep unreadable for one absent string.
UNKNOWN_PARAMS_SELECTION: str = "unknown"

PARAMS_SELECTION_KEY: str = "params_selection"


class CrossArmSweep:  # pylint: disable=too-many-instance-attributes
    """Combine per-arm aggregates into one cross-arm horizon artefact.

    An absent arm is skipped and named rather than raised, so the sweep is
    readable while the remaining arms are still training.
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
                Overrides the derived location for the fallback run only;
                overridden arms always resolve through their own run name.
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
        # Overwritten in sweep() from what the aggregates declare. Defaulted so
        # every key selection has a value before an aggregate is read.
        self.error_metric: str = ERROR_METRIC_CROSS_ENTROPY

    # -- paths ---------------------------------------------------------------

    def arm_run(self, arm: int) -> str:
        """Return the run name one arm is read from."""
        return self.arm_runs.get(arm, self.run_name)

    def series_dir(self, arm: int) -> Path:
        """Return the directory holding one arm's per-seed directories.

        Derived the same way CrossSeedAggregator derives it, so an arm read
        from another run resolves through that run's own name rather than
        through this one's directory.

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

        Beside the aggregates of the run that was asked for, which is the run
        the comparison belongs to even when one arm was read from elsewhere.
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

        Referenced rather than copied. Discovered by glob, so the filename
        template is not restated here.

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

        An aggregate written before the field existed carries cross-entropy by
        construction: it is the only metric this codebase scored then.

        Args:
            loaded: {arm: aggregate payload} for the arms present.

        Returns:
            The declared metric.

        Raises:
            ValueError: If the arms disagree. Arms scored under different
                metrics cannot share a horizon axis or a ratio.
        """
        declared = {
            payload.get(ERROR_METRIC_KEY, ERROR_METRIC_CROSS_ENTROPY)
            for payload in loaded.values()
        }
        if len(declared) > 1:
            raise ValueError(
                f"arms of run '{self.run_name}' disagree about "
                f"{ERROR_METRIC_KEY}: {sorted(declared)}. They were scored "
                f"under different error metrics, so no cross-arm curve or "
                f"ratio over them is readable."
            )
        return declared.pop()

    # -- assembly ------------------------------------------------------------

    def _provenance(self, arm: int, mode_block: dict) -> dict:
        """Return one arm's provenance for one mode.

        The parameter selection is read per mode, not per arm: an arm's modes
        can be scored under different selections when only one of them has
        been re-scored.

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
        """Return the pre-registered endpoint error ratio for one arm and mode.

        Anchored at REPORTED_RATIO_HORIZONS, never at the widest horizons the
        block carries. Reading `horizons[0], horizons[-1]` instead would
        silently redefine the ratio under this same key whenever a horizon past
        the reporting grid is added.

        Args:
            mode_block: One arm's block for one observation mode.

        Returns:
            The ratio block, or None when it cannot be computed.
        """
        low, high = REPORTED_RATIO_HORIZONS
        return self._ratio_between(mode_block, str(low), str(high))

    def _extrapolation_ratio(self, mode_block: dict) -> dict | None:
        """Return the error ratio across the extrapolation horizons.

        Read from the pre-registered denominator, REPORTED_RATIO_HORIZONS[0],
        to the widest extrapolation horizon present, so it answers how far the
        curve moves outside the trained range.

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

        Computed per seed and then aggregated, so the interval is over seeds
        rather than over a ratio of two aggregates. Both endpoints are named in
        the result, so a reader need not trust the key.

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

    def _horizon_entry(self, mode: str, horizon: str, arms: dict[int, dict]) -> dict:
        """Build one horizon's entry for one observation mode.

        The copy baseline and the window count are properties of the data
        rather than of an arm, so they are carried once and the arm they were
        read from is named.

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
        """Flag an arm whose baseline or window count differs from the source.

        Both are properties of the evaluation data. Arms disagreeing about
        them were not evaluated on the same windows, which makes the horizon
        curve a comparison of two things at once.

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

        The two horizon blocks are separate. `horizons` carries the reporting
        grid and is what the fit reads, so the reported exponent is fitted
        inside the trained range whatever else was scored.
        `extrapolation_horizons` carries the rest under its own name.

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
            # Carried from the aggregates so the fit resolves its own key from
            # one declaration rather than inferring the metric from which keys
            # the artefact happens to hold.
            ERROR_METRIC_KEY: self.error_metric,
            "arm_runs": {str(arm): self.arm_run(arm) for arm in sorted(loaded)},
            "arms_present": [str(arm) for arm in sorted(loaded)],
            "arms_absent": [str(arm) for arm in ARMS if arm not in loaded],
            "observation_modes": modes,
            # The endpoint ratio is the only quantity resampled here. Every
            # per-horizon block was resampled by the aggregator and each
            # aggregate records the settings that produced it.
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
        run_name: The run every arm is read from unless overridden. Required
            unless series_dir is given, which infers it.
        series_dir: Directory holding the seed run directories.
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
