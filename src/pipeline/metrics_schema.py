"""Key constants for the per-horizon metrics artefact.

Written by ``src/pipeline/evaluate.py`` and read across the aggregation path.
Every consumer imports these rather than restating them, so a rename breaks an
import instead of producing an empty series. Imports nothing from ``src/``,
keeping aggregation clear of the training stack. The ``MOVER_*`` keys live in
``src/eval/metrics.py``, beside the function that produces them.
"""

from __future__ import annotations

# Top-level keys of metrics.json.
#
# PER_HORIZON_KEY holds the primary parameter tree and does not move when a
# second tree is scored beside it. Every downstream reader indexes it, so the
# secondary and the gap arrive as sibling blocks that an older reader ignores.
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

# The gap's sign, recorded in the artefact so a reader need not infer it.
# Positive means the primary tree scored higher, which on a loss-like metric
# means it scored worse.
GAP_SIGN_CONVENTION: str = "primary_minus_secondary"

# What the test pass records in place of params_selection in its sentinel
# identity. It scores both trees and refuses --params, so the identity must not
# carry a flag that run rejects.
PARAMS_TREES_BOTH: str = "both"

# Which error the artefact's model series holds. Declared by the run rather
# than inferred from which keys are present, so a run scored under one metric
# can never be read under the other.
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


# The per-horizon keys each declared error metric writes its series to. The
# model and the copy baseline are scored under one metric, so both are selected
# from the same declaration rather than named at each call site.
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
