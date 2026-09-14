"""Key constants for the per-horizon metrics artefact and what is derived from it.

The evaluate stage writes the per-horizon artefact. The sweep writes the
compounding-error block and the fit carries it. Changing a key's string
orphans every artefact already written, and a reader that looks keys up with
``.get`` then finds nothing instead of raising. Imports nothing from ``src/``.
The keys ``mover_restricted_accuracy`` returns are defined in
``src/eval/metrics.py``.
"""

from __future__ import annotations

# Top-level keys of each metrics_{mode}.json.
#
# PER_HORIZON_KEY holds the primary parameter tree whether or not a secondary
# is scored. The secondary and the gap are sibling blocks.
PER_HORIZON_KEY: str = "per_horizon"
PER_HORIZON_FINAL_STEP_KEY: str = "per_horizon_final_step"
PER_HORIZON_GAP_KEY: str = "per_horizon_gap"
PARAMS_PROVENANCE_KEY: str = "params_provenance"
OBSERVATION_MODE_KEY: str = "observation_mode"
ARM_KEY: str = "arm"
PROBABILITY_LOG_KEY: str = "probability_log"
SPLIT_KEY: str = "split"
RUN_NAME_KEY: str = "run_name"

# Which run's checkpoints and shards produced the numbers, when that is not the
# run they were written under. Equal to RUN_NAME_KEY on an ordinary run.
SOURCE_RUN_NAME_KEY: str = "source_run_name"

# The gap's sign, recorded in the artefact. Positive means the primary tree's
# value is higher, which on a loss-like metric means it scored worse.
GAP_SIGN_CONVENTION: str = "primary_minus_secondary"

# Sentinel-identity value under parameter_trees, in place of params_selection,
# for a pass that scores both parameter trees.
PARAMS_TREES_BOTH: str = "both"

# Which error the model and copy series hold, as declared by the run.
ERROR_METRIC_KEY: str = "error_metric"
ERROR_METRIC_CROSS_ENTROPY: str = "cross_entropy"
ERROR_METRIC_MSE: str = "mse"

# Keys inside one per-horizon block.
WINDOWS_KEY: str = "num_windows"
MODEL_CE_KEY: str = "model_cross_entropy"
MODEL_MSE_KEY: str = "model_mse"
COPY_CE_KEY: str = "copy_cross_entropy"
COPY_MSE_KEY: str = "copy_mse"
SKILL_SCORE_KEY: str = "copy_normalised_skill_score"
CLIMATOLOGY_KEY: str = "climatology_entropy"
PER_CELL_ACCURACY_KEY: str = "per_cell_accuracy"
EXACT_MATCH_KEY: str = "exact_grid_match_rate"
AGENT_ACCURACY_KEY: str = "agent_position_accuracy"
SMOOTHING_KEY: str = "smoothing_report"

# Continuous analogues of the categorical block, None on a categorical run. The
# mover pair is scored on the pixel-channels whose stored value changes, and the
# climatology floor predicts the training-split mean image at each horizon.
MOVER_MSE_KEY: str = "mover_restricted_mse"
COPY_MOVER_MSE_KEY: str = "copy_mover_restricted_mse"
MOVER_MSE_SKILL_KEY: str = "mover_restricted_skill_score"
MOVER_CHANGED_PIXELS_KEY: str = "mean_changed_pixels"
CLIMATOLOGY_MSE_KEY: str = "climatology_mse"

# The compounding error: one arm's error accumulated over the scored horizons,
# extrapolation horizons excluded, per seed, for the model and the copy
# baseline. The sweep writes the block under SIGMA_KEY and the fit carries it.
# The sum weights every horizon equally, the integral is the trapezoid over h,
# and the discounted integral is the trapezoid of the error weighted by
# COMPOUNDING_DISCOUNT ** h. The skill score is 1 - model integral / copy
# integral, per seed.
SIGMA_KEY: str = "compounding_error"
SIGMA_SUM_KEY: str = "compounding_error_sum"
SIGMA_INTEGRAL_KEY: str = "compounding_error_integral"
SIGMA_DISCOUNTED_INTEGRAL_KEY: str = "compounding_error_discounted_integral"
COPY_SIGMA_SUM_KEY: str = "copy_compounding_error_sum"
COPY_SIGMA_INTEGRAL_KEY: str = "copy_compounding_error_integral"
COPY_SIGMA_DISCOUNTED_INTEGRAL_KEY: str = (
    "copy_compounding_error_discounted_integral"
)
SIGMA_SKILL_KEY: str = "compounding_skill_score"


# The per-horizon keys holding each error metric's model and copy series.
MODEL_ERROR_KEYS: dict[str, str] = {
    ERROR_METRIC_CROSS_ENTROPY: MODEL_CE_KEY,
    ERROR_METRIC_MSE: MODEL_MSE_KEY,
}
COPY_ERROR_KEYS: dict[str, str] = {
    ERROR_METRIC_CROSS_ENTROPY: COPY_CE_KEY,
    ERROR_METRIC_MSE: COPY_MSE_KEY,
}


def model_error_key(error_metric: str) -> str:
    """Return the per-horizon key holding the model series for one metric.

    Args:
        error_metric: The value the artefact declares under ERROR_METRIC_KEY.

    Raises:
        ValueError: If the declared metric is not one this codebase writes.
    """
    if error_metric not in MODEL_ERROR_KEYS:
        raise ValueError(
            f"unknown {ERROR_METRIC_KEY} {error_metric!r}, expected one of "
            f"{sorted(MODEL_ERROR_KEYS)}"
        )
    return MODEL_ERROR_KEYS[error_metric]


def copy_error_key(error_metric: str) -> str:
    """Return the per-horizon key holding the copy series for one metric.

    Args:
        error_metric: The value the artefact declares under ERROR_METRIC_KEY.

    Raises:
        ValueError: If the declared metric is not one this codebase writes.
    """
    if error_metric not in COPY_ERROR_KEYS:
        raise ValueError(
            f"unknown {ERROR_METRIC_KEY} {error_metric!r}, expected one of "
            f"{sorted(COPY_ERROR_KEYS)}"
        )
    return COPY_ERROR_KEYS[error_metric]
