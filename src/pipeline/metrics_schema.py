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

# Keys inside one per-horizon block.
WINDOWS_KEY: str = "num_windows"
MODEL_CE_KEY: str = "model_cross_entropy"
COPY_CE_KEY: str = "copy_cross_entropy"
SKILL_SCORE_KEY: str = "copy_normalised_skill_score"
CLIMATOLOGY_KEY: str = "climatology_entropy"
PER_CELL_ACCURACY_KEY: str = "per_cell_accuracy"
EXACT_MATCH_KEY: str = "exact_grid_match_rate"
AGENT_ACCURACY_KEY: str = "agent_position_accuracy"
SMOOTHING_KEY: str = "smoothing_report"
