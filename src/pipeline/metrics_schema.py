"""Key constants for the per-horizon metrics artefact.

Written by ``src/pipeline/evaluate.py`` and read across the aggregation path.
Every consumer imports these rather than restating them, so a rename breaks an
import instead of producing an empty series. Imports nothing from ``src/``,
keeping aggregation clear of the training stack. The ``MOVER_*`` keys live in
``src/eval/metrics.py``, beside the function that produces them.
"""

from __future__ import annotations

# Top-level keys of metrics.json.
PER_HORIZON_KEY: str = "per_horizon"
OBSERVATION_MODE_KEY: str = "observation_mode"
ARM_KEY: str = "arm"
PROBABILITY_LOG_KEY: str = "probability_log"

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
