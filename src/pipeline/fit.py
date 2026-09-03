"""Power-law fits and the usable-horizon diagnostic, read from the sweep.

The headline metric is the exponent c in error(h) = a + b*h^c. That estimator
is not always identified, so every estimator is computed on every run and the
order in which they are reported is fixed here rather than chosen after the
numbers are seen. The endpoint ratio is carried through from the sweep as the
readout that depends on no fit at all.

Reads the cross-arm sweep artefact and writes one file beside it. Fits nothing
that the sweep already measured and recomputes none of its aggregates.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from config import (
    DEFAULT_ENV_NAME,
    FIT_ARTEFACT_NAME,
    FIT_IDENTIFIED_MAX_RSE,
    FIT_INSUFFICIENT_SEEDS,
    FIT_MAX_EVALUATIONS,
    FIT_NO_WINNING_HORIZON,
    FIT_START_AGREE_TOL,
    FIT_START_TWO,
    FIT_STARTS_THREE,
    SEEDS,
    SWEEP_FILENAME,
)
from src.pipeline.aggregate import CrossSeedAggregator, infer_run_name
from src.pipeline.aggregate_stats import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CONFIDENCE_INTERVAL_SIZE,
    aggregate_scalar,
)
from src.pipeline.metrics_schema import MODEL_CE_KEY, SKILL_SCORE_KEY
from src.pipeline.sweep import RATIO_KEY, sorted_horizons
from src.utils.logging_setup import get_logger
from src.utils.paths import safe_rel

logger = get_logger(__name__)

# Estimator names, used as artefact keys and as the value of the reported
# exponent's source field.
THREE_PARAMETER: str = "three_parameter"
TWO_PARAMETER: str = "two_parameter"
LOG_LOG: str = "log_log"

# Why an exponent was not identified. A closed vocabulary; a free-text reason
# cannot be compared across runs.
REASON_RELATIVE_ERROR: str = "relative_standard_error_above_ceiling"
REASON_COVARIANCE: str = "covariance_not_finite"
REASON_STARTS_DISAGREE: str = "start_points_disagree"
REASON_NO_CONVERGENCE: str = "optimiser_did_not_converge"
REASON_NON_POSITIVE: str = "non_positive_errors"
REASON_TOO_FEW_HORIZONS: str = "too_few_horizons"

# A three-parameter fit needs more points than parameters before a covariance
# exists at all. `fit_two_parameter` needs one fewer; `log_log_slope` uses the
# three-parameter minimum.
MIN_POINTS_THREE: int = 4
MIN_POINTS_TWO: int = 3


@dataclass(frozen=True)
class FitResult:  # pylint: disable=too-many-instance-attributes
    """One estimator's answer on one seed's curve.

    identified is the field callers act on. An exponent read without it is a
    number the data does not support.
    """

    estimator: str
    parameters: dict[str, float]
    exponent: float | None
    standard_error: float | None
    residuals: list[float]
    identified: bool
    reason: str | None = None
    starts: tuple[dict, ...] = ()
    start_spread: float | None = None
    starts_agree: bool | None = None

    def as_dict(self) -> dict:
        """Return the JSON-serialisable form."""
        block: dict[str, Any] = {
            "estimator": self.estimator,
            "parameters": dict(self.parameters),
            "exponent": self.exponent,
            "standard_error": self.standard_error,
            "identified": self.identified,
            "reason": self.reason,
        }
        if self.starts:
            block["starts"] = [dict(start) for start in self.starts]
            block["start_spread"] = self.start_spread
            block["starts_agree"] = self.starts_agree
        return block


@dataclass(frozen=True)
class UsableHorizon:
    """Where one arm beats the stationary-copy baseline.

    The set and the range are both carried because the winning horizons are not
    always a prefix of the axis.
    """

    horizons: tuple[int, ...] = ()
    range_low: int | None = None
    range_high: int | None = None
    gate_scalar: int | None = None
    starts_above_min: bool = False
    reason: str | None = None

    def as_dict(self) -> dict:
        """Return the JSON-serialisable form."""
        return {
            "set": list(self.horizons),
            "range": (
                None
                if self.range_low is None
                else {"low": self.range_low, "high": self.range_high}
            ),
            "gate_scalar": self.gate_scalar,
            "starts_above_min": self.starts_above_min,
            "reason": self.reason,
        }


def _three_parameter_model(
    horizons: np.ndarray, offset: float, scale: float, exponent: float
) -> np.ndarray:
    """Return a + b*h^c, the pre-registered form."""
    return offset + scale * horizons**exponent


def _two_parameter_model(
    horizons: np.ndarray, scale: float, exponent: float
) -> np.ndarray:
    """Return b*h^c, the offset-free form."""
    return scale * horizons**exponent


def is_identified(estimate: float | None, standard_error: float | None) -> bool:
    """Return whether an exponent met the pre-registered identifiability rule.

    The rule is a relative standard error at or below FIT_IDENTIFIED_MAX_RSE,
    with a non-finite standard error counting as unidentified.

    Args:
        estimate: The fitted exponent.
        standard_error: Its standard error, or None when not estimable.

    Returns:
        Whether the exponent is identified.
    """
    if estimate is None or standard_error is None:
        return False
    if not math.isfinite(estimate) or not math.isfinite(standard_error):
        return False
    return standard_error <= FIT_IDENTIFIED_MAX_RSE * abs(estimate)


def _curve_fit(
    model: Callable[..., np.ndarray],
    horizons: np.ndarray,
    errors: np.ndarray,
    start: Sequence[float],
    exponent_index: int,
) -> tuple[list[float] | None, float | None]:
    """Fit one model from one start point.

    curve_fit returns a covariance that may hold inf or nan without raising, so
    the standard error is only taken from a covariance that is finite.

    Args:
        model: The callable being fitted.
        horizons: The evaluation horizons.
        errors: One seed's error at each horizon.
        start: The start point.
        exponent_index: Which parameter is the exponent.

    Returns:
        The parameters and the exponent's standard error, either of which is
        None when the optimiser did not converge or the covariance is not
        finite.
    """
    with warnings.catch_warnings():
        # The covariance being inestimable is the finding and is recorded as
        # one.
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            parameters, covariance = curve_fit(
                model,
                horizons,
                errors,
                p0=tuple(start),
                maxfev=FIT_MAX_EVALUATIONS,
            )
        except (RuntimeError, TypeError, ValueError):
            return None, None
    variance = covariance[exponent_index][exponent_index]
    if not np.all(np.isfinite(covariance)) or variance < 0:
        return [float(value) for value in parameters], None
    return [float(value) for value in parameters], float(math.sqrt(variance))


def _residuals(
    model: Callable[..., np.ndarray],
    horizons: np.ndarray,
    errors: np.ndarray,
    parameters: Sequence[float],
) -> list[float]:
    """Return observed minus fitted at every horizon."""
    return [float(value) for value in errors - model(horizons, *parameters)]


def _start_point(start: Sequence[float | None], errors: np.ndarray) -> list[float]:
    """Resolve a configured start point against one seed's curve.

    None in the leading slot means that seed's error at the shortest horizon,
    so one start is derived from the data and the others are fixed.
    """
    return [float(errors[0]) if value is None else float(value) for value in start]


def fit_three_parameter(
    horizons: np.ndarray,
    errors: np.ndarray,
) -> FitResult:
    """Fit error(h) = a + b*h^c, the pre-registered form.

    This is the primary estimator. It is frequently unidentified on flat
    curves: as c approaches zero, h^c approaches one at every horizon and the
    model collapses to the constant a + b, which infinitely many (a, b) pairs
    reproduce. The standard error is one piece of evidence for that and the
    spread across start points is the other, so callers must consult
    FitResult.identified rather than reading the exponent alone.

    Args:
        horizons: The evaluation horizons, ascending.
        errors: One seed's error at each horizon, same length and order.

    Returns:
        The fit from the first start point, the standard error on c, the
        residuals, every start point's exponent, and whether c met the rule.
    """
    if len(horizons) < MIN_POINTS_THREE:
        return FitResult(
            THREE_PARAMETER, {}, None, None, [], False, REASON_TOO_FEW_HORIZONS
        )
    attempts = []
    for start in FIT_STARTS_THREE:
        resolved = _start_point(start, errors)
        parameters, standard_error = _curve_fit(
            _three_parameter_model, horizons, errors, resolved, exponent_index=2
        )
        attempts.append(
            {
                "start": resolved,
                "parameters": parameters,
                "exponent": None if parameters is None else parameters[2],
                "standard_error": standard_error,
            }
        )
    first = attempts[0]
    exponents = [
        attempt["exponent"] for attempt in attempts if attempt["exponent"] is not None
    ]
    spread = (
        float(max(exponents) - min(exponents)) if len(exponents) == len(attempts) else None
    )
    agree = spread is not None and spread <= FIT_START_AGREE_TOL
    if first["parameters"] is None:
        return FitResult(
            THREE_PARAMETER,
            {},
            None,
            None,
            [],
            False,
            REASON_NO_CONVERGENCE,
            tuple(attempts),
            spread,
            agree,
        )
    offset, scale, exponent = first["parameters"]
    identified = is_identified(exponent, first["standard_error"]) and agree
    return FitResult(
        THREE_PARAMETER,
        {"a": offset, "b": scale, "c": exponent},
        exponent,
        first["standard_error"],
        _residuals(
            _three_parameter_model, horizons, errors, first["parameters"]
        ),
        identified,
        _reason(exponent, first["standard_error"], agree),
        tuple(attempts),
        spread,
        agree,
    )


def _reason(
    exponent: float, standard_error: float | None, agree: bool
) -> str | None:
    """Return why an exponent was not identified, or None when it was."""
    if standard_error is None:
        return REASON_COVARIANCE
    if not is_identified(exponent, standard_error):
        return REASON_RELATIVE_ERROR
    if not agree:
        return REASON_STARTS_DISAGREE
    return None


def fit_two_parameter(
    horizons: np.ndarray,
    errors: np.ndarray,
) -> FitResult:
    """Fit error(h) = b*h^c, the offset-free form.

    Reported when the three-parameter fit is unidentified. Removing a removes
    the cancellation that makes c unidentifiable, at the cost of asserting
    there is no irreducible error floor, which is false. The exponent is
    therefore not the same quantity as the pre-registered c and is named
    separately in the artefact so the two cannot be confused.

    Args:
        horizons: The evaluation horizons, ascending.
        errors: One seed's error at each horizon, same length and order.

    Returns:
        The fitted scale and exponent, the standard error on the exponent, the
        residuals, and whether the exponent met the rule.
    """
    if len(horizons) < MIN_POINTS_TWO:
        return FitResult(
            TWO_PARAMETER, {}, None, None, [], False, REASON_TOO_FEW_HORIZONS
        )
    resolved = _start_point(FIT_START_TWO, errors)
    parameters, standard_error = _curve_fit(
        _two_parameter_model, horizons, errors, resolved, exponent_index=1
    )
    if parameters is None:
        return FitResult(
            TWO_PARAMETER, {}, None, None, [], False, REASON_NO_CONVERGENCE
        )
    scale, exponent = parameters
    identified = is_identified(exponent, standard_error)
    return FitResult(
        TWO_PARAMETER,
        {"b": scale, "c": exponent},
        exponent,
        standard_error,
        _residuals(_two_parameter_model, horizons, errors, parameters),
        identified,
        _reason(exponent, standard_error, agree=True),
    )


def log_log_slope(
    horizons: np.ndarray,
    errors: np.ndarray,
) -> FitResult:
    """Fit the slope of log(error) against log(h) by ordinary least squares.

    A robustness check on fit_two_parameter rather than a third opinion. The
    two differ only in weighting: this minimises relative error, so every
    horizon counts equally, while fit_two_parameter minimises absolute error
    and is dominated by the largest values. Agreement between them is evidence
    that the exponent does not depend on that choice.

    Errors must be strictly positive; a non-positive value returns an
    unidentified result rather than raising, because a cross-entropy of zero
    is a legitimate artefact value and not a caller error.

    Args:
        horizons: The evaluation horizons, ascending.
        errors: One seed's error at each horizon, same length and order.

    Returns:
        The fitted slope and intercept, the standard error on the slope, the
        residuals in log space, and whether the slope met the rule.
    """
    if len(horizons) < MIN_POINTS_THREE:
        return FitResult(LOG_LOG, {}, None, None, [], False, REASON_TOO_FEW_HORIZONS)
    if np.any(errors <= 0):
        return FitResult(LOG_LOG, {}, None, None, [], False, REASON_NON_POSITIVE)
    log_horizons = np.log(horizons)
    log_errors = np.log(errors)
    coefficients, covariance = np.polyfit(log_horizons, log_errors, 1, cov=True)
    slope, intercept = float(coefficients[0]), float(coefficients[1])
    variance = covariance[0][0]
    standard_error = (
        float(math.sqrt(variance))
        if np.all(np.isfinite(covariance)) and variance >= 0
        else None
    )
    fitted = intercept + slope * log_horizons
    return FitResult(
        LOG_LOG,
        {"slope": slope, "intercept": intercept},
        slope,
        standard_error,
        [float(value) for value in log_errors - fitted],
        is_identified(slope, standard_error),
        _reason(slope, standard_error, agree=True),
    )


def usable_horizons(
    horizons: Sequence[int],
    skill_blocks: Sequence[dict | None],
    use_interval: bool,
) -> UsableHorizon:
    """Return where the model beats the copy baseline, as a set and a range.

    The usable horizon is the largest h at which the model still beats the
    copy baseline, which assumes the model wins at short horizons and stops.
    The measured shape is the reverse on egocentric observations, where the
    model loses at h = 1 and wins at every longer horizon, because a copy is
    nearly perfect when the view has barely changed. The full set is therefore
    returned alongside the range, and a range whose lower bound exceeds the
    shortest horizon is flagged so no reader takes the upper bound as a span
    starting at one.

    An empty set returns None with a reason rather than zero. Zero is a number
    and would be averaged and plotted; the reason distinguishes losing at every
    horizon from being inapplicable.

    Args:
        horizons: The evaluation horizons, ascending.
        skill_blocks: One aggregated skill-score block per horizon, in the
            same order.
        use_interval: Whether a horizon counts as won only when the confidence
            interval clears zero, rather than the point estimate alone.

    Returns:
        The winning set, the contiguous range holding the largest winning
        horizon, the scalar the gate reads, and whether that range starts above
        the shortest horizon.
    """
    field_name = "ci_low" if use_interval else "iqm"
    readable = [
        block is not None and block.get(field_name) is not None
        for block in skill_blocks
    ]
    if not any(readable):
        return UsableHorizon(reason=FIT_INSUFFICIENT_SEEDS)
    winners = tuple(
        horizon
        for horizon, block, ok in zip(horizons, skill_blocks, readable)
        if ok and block[field_name] > 0
    )
    if not winners:
        return UsableHorizon(reason=FIT_NO_WINNING_HORIZON)
    ordered = list(horizons)
    high = winners[-1]
    low = high
    for position in range(ordered.index(high), 0, -1):
        if ordered[position - 1] not in winners:
            break
        low = ordered[position - 1]
    return UsableHorizon(
        horizons=winners,
        range_low=low,
        range_high=high,
        gate_scalar=high,
        starts_above_min=low > ordered[0],
    )


def aggregate_exponents(
    results: Sequence[FitResult],
    reps: int,
    statistic_seed: int,
) -> dict | None:
    """Aggregate one estimator's per-seed exponents across seeds.

    Args:
        results: One estimator's result on each seed.
        reps: Bootstrap resamples.
        statistic_seed: Seed for the bootstrap's random state.

    Returns:
        aggregate_scalar's block, or None when no seed produced an exponent.
    """
    values = [
        result.exponent
        for result in results
        if result.exponent is not None and math.isfinite(result.exponent)
    ]
    if not values:
        return None
    return aggregate_scalar(values, reps=reps, statistic_seed=statistic_seed)


class HorizonFit:
    """Fit the exponent and the usable horizon from a written sweep artefact."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        run_name: str,
        series_dir: Path | None = None,
        env: str | None = None,
        fast: bool = False,
        reps: int | None = None,
        statistic_seed: int | None = None,
    ) -> None:
        """Store where the sweep is read from and where the fit is written.

        The series directory is resolved once here; both artefacts sit in it.

        Args:
            run_name: The run whose sweep artefact is read.
            series_dir: Directory holding the sweep artefact. Overrides the
                derived location.
            env: Registered environment name, a directory level under the run.
            fast: Whether the series sits under the fast tree.
            reps: Bootstrap resamples for the exponent's interval.
            statistic_seed: Seed for the bootstrap's random state.
        """
        self.run_name = run_name
        self.series_dir = series_dir or CrossSeedAggregator(
            run_name, env=env or DEFAULT_ENV_NAME, fast=fast
        ).series_dir
        self.reps = BOOTSTRAP_REPS if reps is None else reps
        self.statistic_seed = BOOTSTRAP_SEED if statistic_seed is None else statistic_seed
        self.flags: list[str] = []

    def source_path(self) -> Path:
        """Return where the sweep artefact is expected."""
        return self.series_dir / SWEEP_FILENAME

    def fit_path(self) -> Path:
        """Return where the fit artefact is written."""
        return self.series_dir / FIT_ARTEFACT_NAME

    def load(self) -> dict:
        """Read the sweep artefact.

        Returns:
            The sweep payload.

        Raises:
            FileNotFoundError: If the sweep has not been written.
        """
        path = self.source_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"no sweep artefact at {safe_rel(path)} -- run --sweep before "
                f"--fit, since the fit reads what the sweep measured."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def _curves(self, mode_block: dict, arm: str) -> tuple[list[int], list[list[float]]]:
        """Return the horizon axis and one error curve per seed for one arm.

        Args:
            mode_block: One observation mode's block from the sweep.
            arm: The arm being read.

        Returns:
            The horizons and one curve per seed. Both are empty when the arm
            carries no readable series.
        """
        horizons = sorted_horizons(mode_block.get("horizons", {}))
        columns = []
        for horizon in horizons:
            block = (
                mode_block["horizons"][horizon]
                .get("arms", {})
                .get(arm, {})
                .get(MODEL_CE_KEY)
            )
            values = None if block is None else block.get("values")
            if not values:
                message = (
                    f"{mode_block.get('observation_mode')} arm {arm} carries no "
                    f"per-seed {MODEL_CE_KEY} at horizon {horizon}, so no curve "
                    f"can be fitted"
                )
                logger.warning("%s", message)
                self.flags.append(message)
                return [], []
            columns.append(values)
        widths = {len(column) for column in columns}
        if not columns or len(widths) != 1:
            if widths:
                message = (
                    f"{mode_block.get('observation_mode')} arm {arm} carries "
                    f"{sorted(widths)} seeds across its horizons, so the curves "
                    f"cannot be read seed by seed"
                )
                logger.warning("%s", message)
                self.flags.append(message)
            return [], []
        curves = [
            [float(column[index]) for column in columns]
            for index in range(next(iter(widths)))
        ]
        return [int(horizon) for horizon in horizons], curves

    def _arm_block(self, mode_block: dict, arm: str) -> dict:
        """Fit every estimator for one arm of one observation mode.

        Args:
            mode_block: One observation mode's block from the sweep.
            arm: The arm being fitted.

        Returns:
            That arm's fit block.
        """
        horizons, curves = self._curves(mode_block, arm)
        arm_entry = mode_block.get("arms", {}).get(arm, {})
        seeds = arm_entry.get("seeds") or list(SEEDS[: len(curves)])
        axis = np.array(horizons, dtype=float)
        results: dict[str, list[FitResult]] = {
            THREE_PARAMETER: [],
            TWO_PARAMETER: [],
            LOG_LOG: [],
        }
        for curve in curves:
            errors = np.array(curve, dtype=float)
            results[THREE_PARAMETER].append(fit_three_parameter(axis, errors))
            results[TWO_PARAMETER].append(fit_two_parameter(axis, errors))
            results[LOG_LOG].append(log_log_slope(axis, errors))
        block = {
            name: self._estimator_block(name, results[name], seeds)
            for name in (THREE_PARAMETER, TWO_PARAMETER, LOG_LOG)
        }
        block["residuals"] = {
            name: {
                str(seed): dict(zip((str(h) for h in horizons), result.residuals))
                for seed, result in zip(seeds, results[name])
                if result.residuals
            }
            for name in (THREE_PARAMETER, TWO_PARAMETER, LOG_LOG)
        }
        block["reported_exponent"] = self._reported(block)
        # Read through rather than recomputed. The ratio is the readout that
        # depends on no fit, and it travels in this file.
        block[RATIO_KEY] = arm_entry.get(RATIO_KEY)
        block["seeds"] = list(seeds)
        block["horizons"] = list(horizons)
        return block

    def _estimator_block(
        self, name: str, results: Sequence[FitResult], seeds: Sequence[int]
    ) -> dict:
        """Assemble one estimator's per-seed results and their aggregate.

        Args:
            name: The estimator name.
            results: That estimator's result on each seed.
            seeds: The seeds, in the order the curves were read.

        Returns:
            The estimator block.
        """
        identified = [result.identified for result in results]
        return {
            "estimator": name,
            "per_seed": [
                dict(result.as_dict(), seed=seed)
                for seed, result in zip(seeds, results)
            ],
            "se": [result.standard_error for result in results],
            "identified": bool(identified) and all(identified),
            "n_identified": sum(identified),
            "n_seeds": len(results),
            "aggregate": aggregate_exponents(
                results, self.reps, self.statistic_seed
            ),
        }

    def _reported(self, block: dict) -> dict:
        """Return which exponent is reported, following the fixed ladder.

        The order is fixed before the data is seen: the pre-registered
        three-parameter exponent, and the offset-free exponent only when the
        first is unidentified. The offset-free exponent is a different quantity
        and is named as one.

        Args:
            block: The arm's estimator blocks.

        Returns:
            The source, the value and the reason.
        """
        primary = block[THREE_PARAMETER]
        if primary["identified"]:
            return {
                "source": THREE_PARAMETER,
                "value": primary["aggregate"],
                "quantity": "a + b*h^c exponent, pre-registered",
                "reason": "the pre-registered fit is identified on every seed",
            }
        fallback = block[TWO_PARAMETER]
        return {
            "source": TWO_PARAMETER,
            "value": fallback["aggregate"],
            "quantity": "b*h^c exponent, NOT the pre-registered c",
            "reason": (
                f"the pre-registered fit is identified on "
                f"{primary['n_identified']} of {primary['n_seeds']} seed(s)"
            ),
        }

    def _mode_block(self, mode: str, mode_block: dict) -> dict:
        """Fit every arm of one observation mode and read its usable horizons.

        Args:
            mode: The observation mode.
            mode_block: That mode's block from the sweep.

        Returns:
            That mode's fit block.
        """
        arms = sorted(mode_block.get("arms", {}), key=int)
        horizons = [int(horizon) for horizon in sorted_horizons(mode_block.get("horizons", {}))]
        return {
            "observation_mode": mode,
            "arms_present": list(arms),
            "arms": {arm: self._arm_block(mode_block, arm) for arm in arms},
            "usable_horizon": {
                "arms": {
                    arm: {
                        "point": usable_horizons(
                            horizons, self._skill(mode_block, arm), use_interval=False
                        ).as_dict(),
                        "interval": usable_horizons(
                            horizons, self._skill(mode_block, arm), use_interval=True
                        ).as_dict(),
                    }
                    for arm in arms
                }
            },
        }

    @staticmethod
    def _skill(mode_block: dict, arm: str) -> list[dict | None]:
        """Return one arm's aggregated skill-score block at every horizon."""
        return [
            mode_block["horizons"][horizon]
            .get("arms", {})
            .get(arm, {})
            .get(SKILL_SCORE_KEY)
            for horizon in sorted_horizons(mode_block.get("horizons", {}))
        ]

    def fit(self) -> dict:
        """Build the fit payload.

        Returns:
            The fit artefact as a dict.
        """
        self.flags = []
        sweep = self.load()
        modes = sweep.get("observation_modes") or sorted(sweep.get("by_mode", {}))
        payload = {
            "run_name": sweep.get("run_name", self.run_name),
            "source_artefact": safe_rel(self.source_path()),
            "source_mtime": datetime.fromtimestamp(
                self.source_path().stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "fit_config": {
                "starts_three": [list(start) for start in FIT_STARTS_THREE],
                "start_two": list(FIT_START_TWO),
                "max_evaluations": FIT_MAX_EVALUATIONS,
                "identified_max_rse": FIT_IDENTIFIED_MAX_RSE,
                "start_agree_tol": FIT_START_AGREE_TOL,
                "reps": self.reps,
                "seed": self.statistic_seed,
                "confidence_interval_size": CONFIDENCE_INTERVAL_SIZE,
            },
            "observation_modes": list(modes),
            "by_mode": {
                mode: self._mode_block(mode, sweep["by_mode"][mode])
                for mode in modes
                if mode in sweep.get("by_mode", {})
            },
        }
        payload["flags"] = list(self.flags)
        return payload

    def write(self, payload: dict) -> Path:
        """Write the fit artefact and return its path.

        Args:
            payload: The payload produced by fit().

        Returns:
            The path written.
        """
        path = self.fit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("horizon fit written to %s", safe_rel(path))
        return path

    def run(self) -> Path:
        """Build the fit and write it.

        Returns:
            The path written.
        """
        payload = self.fit()
        path = self.write(payload)
        for mode, block in sorted(payload["by_mode"].items()):
            for arm, arm_block in sorted(block["arms"].items()):
                reported = arm_block["reported_exponent"]
                logger.info(
                    "%s arm %s: reporting the %s exponent (%s)",
                    mode,
                    arm,
                    reported["source"],
                    reported["reason"],
                )
        logger.info("%d flag(s) recorded", len(payload["flags"]))
        return path


def run_fit_cli(
    run_name: str | None,
    series_dir: Path | None,
    env: str | None = None,
    fast: bool = False,
) -> Path:
    """Fit a run's swept horizon curves and return the artefact path.

    Args:
        run_name: The run whose sweep is read. Required unless series_dir is
            given, which infers it.
        series_dir: Directory holding the sweep artefact.
        env: Registered environment name.
        fast: Whether the series sits under the fast tree.

    Returns:
        The path written.

    Raises:
        ValueError: If neither run_name nor series_dir is supplied.
    """
    if run_name is None and series_dir is None:
        raise ValueError(
            "--fit needs --run-name, or --series-dir to infer it from. "
            "outputs/ holds many unrelated runs, so a run name cannot be "
            "inferred without one of them."
        )
    resolved = run_name if run_name is not None else infer_run_name(series_dir)
    return HorizonFit(
        resolved,
        series_dir=series_dir,
        env=env,
        fast=fast,
    ).run()
