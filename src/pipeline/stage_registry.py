"""Which stage class each dataset stage name selects.

Read by the runner and the manifest. config holds the routing rule and imports
no stage.
"""

from __future__ import annotations

from config import CANONICAL_STAGES
from src.pipeline.base import Stage
from src.pipeline.displacement import DisplacementDiagnostic
from src.pipeline.generate import GenerateTrajectoriesStage
from src.pipeline.offline_generate import OfflineGenerateStage
from src.pipeline.prepare import PrepareDatasetStage

# Which of these one environment gets comes from config.dataset_stage_names.
DATASET_STAGE_CLASSES: dict[str, type[Stage]] = {
    cls.name: cls
    for cls in (
        GenerateTrajectoriesStage,
        OfflineGenerateStage,
        PrepareDatasetStage,
        DisplacementDiagnostic,
    )
}
# Order is per environment, so this compares membership.
if set(DATASET_STAGE_CLASSES) != set(CANONICAL_STAGES):
    raise RuntimeError(
        "CANONICAL_STAGES and DATASET_STAGE_CLASSES have drifted: "
        f"{sorted(CANONICAL_STAGES)} against {sorted(DATASET_STAGE_CLASSES)}"
    )
