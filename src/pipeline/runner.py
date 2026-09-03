"""Pipeline runner: derive one configuration per reporting seed and execute.

Each stage is single-dataset and is instantiated once per seed, so one stage
stays one unit of work behind one sentinel.

Dataset stages file under `data_seed`, training and evaluation under
`model_seed`. A reporting run couples the two; the attribution arm does not.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

from config import (
    ARM_DIRECT,
    ARMS,
    SEEDS,
    SELECTABLE_STAGES,
    STAGE_NAME_DISPLACEMENT,
    STAGE_NAME_EVALUATE,
    STAGE_NAME_TRAIN,
    ExperimentConfig,
    dataset_stage_names,
    displacement_skip_note,
)
from src.data.split import SplitName
from src.pipeline.aggregate import aggregate_if_series_complete
from src.pipeline.base import (
    STAGE_FAILED,
    STAGE_SUCCESS,
    Stage,
    log_run_summary,
)
from src.pipeline.evaluate import EvaluateStage
from src.pipeline.manifest import write_manifest
from src.pipeline.stage_registry import DATASET_STAGE_CLASSES
from src.pipeline.train import TrainStage
from src.utils.logging_setup import RUN_START_BANNER, get_logger

logger = get_logger(__name__)

# config's stage names must be the classes' own, or --stages accepts a name that
# selects nothing.
if (TrainStage.name, EvaluateStage.name) != (
    STAGE_NAME_TRAIN,
    STAGE_NAME_EVALUATE,
):
    raise RuntimeError(
        "the model stage names have drifted from config: "
        f"{(TrainStage.name, EvaluateStage.name)} against "
        f"{(STAGE_NAME_TRAIN, STAGE_NAME_EVALUATE)}"
    )


def _assert_seeds_are_coupled(configs: Sequence[ExperimentConfig]) -> None:
    """Raise unless every reporting configuration has `data_seed == model_seed`."""
    for config in configs:
        if config.data_seed != config.model_seed:
            raise ValueError(
                "the reporting matrix requires coupled seeds, got "
                f"data_seed={config.data_seed} against "
                f"model_seed={config.model_seed}. Uncoupled seeds send a "
                "dataset and its model to different directories. Only the "
                "attribution arm may diverge, and it has its own entry point."
            )


def configs_for_datasets(config: ExperimentConfig) -> list[ExperimentConfig]:
    """Derive one configuration per reporting seed, all three seeds equal.

    One independently generated dataset per seed. Seeds drawing from a shared
    dataset are not independent replications, and intervals over them would be
    conditional on that single draw. Holding `model_seed` fixed resolves every
    dataset to one checkpoint directory.

    The attribution arm pins `data_seed` and varies `model_seed`, and does not
    come through here.

    Returns:
        One configuration per entry in SEEDS, in order.

    Raises:
        ValueError: If SEEDS holds duplicates, or if any derived configuration
            leaves the two seeds uncoupled.
    """
    if len(set(SEEDS)) != len(SEEDS):
        raise ValueError(
            f"SEEDS holds duplicates {SEEDS}: two datasets would share one "
            "directory and one sentinel, and the second would report complete "
            "without having been generated"
        )
    # Set model_seed explicitly. __post_init__ resolves at construction and
    # replace() would carry the old value forward. Do not build a fresh
    # ExperimentConfig here, which discards every CLI override already applied.
    seed_configs = [
        replace(
            config, seed=data_seed, data_seed=data_seed, model_seed=data_seed
        )
        for data_seed in SEEDS
    ]
    _assert_seeds_are_coupled(seed_configs)
    return seed_configs


def build_stages(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: ExperimentConfig,
    arm: int = ARM_DIRECT,
    stages_wanted: Sequence[str] | None = None,
    split: SplitName = SplitName.VALIDATION,
    source_run: str | None = None,
) -> list[Stage]:
    """Build the stage list for one dataset, in execution order.

    The dataset stages come from config.dataset_stage_names, which the manifest
    reads too. The first is a live rollout for NAVIX and a conversion of a
    recorded corpus otherwise. An omitted diagnostic logs at WARNING with its
    declared reason.

    The arm reaches only the model stages. The dataset stages keep their shared
    paths, and the sentinel is what stops generation re-running per arm. A stage
    `stages_wanted` omits is never constructed, so it cannot create a directory
    or read a path.

    Args:
        config: The configuration for one data seed.
        arm: Which arm to train and evaluate, one of config.ARMS.
        stages_wanted: Restrict to these stage names, keeping the pipeline's
            own order. None runs every stage the environment declares.
        split: Which split evaluation scores.
        source_run: Run evaluation reads shards and checkpoints from.

    Returns:
        The stages to run for that dataset.

    Raises:
        ValueError: If the environment's family declares no source stage, or if
            it has no observation contract row.
    """
    names = dataset_stage_names(config.env.name)
    wanted = None if stages_wanted is None else set(stages_wanted)
    if wanted is None and STAGE_NAME_DISPLACEMENT not in names:
        logger.warning(
            "displacement skipped for %s: %s",
            config.env.name,
            displacement_skip_note(config.env.name),
        )
    stages: list[Stage] = [
        DATASET_STAGE_CLASSES[name](config)
        for name in names
        if wanted is None or name in wanted
    ]
    if wanted is None or STAGE_NAME_TRAIN in wanted:
        stages.append(TrainStage(config, arm=arm))
    if wanted is None or STAGE_NAME_EVALUATE in wanted:
        stages.append(
            EvaluateStage(
                config, arm=arm, split=split, source_run=source_run
            )
        )
    return stages


def run_pipeline(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config: ExperimentConfig,
    arm: int = ARM_DIRECT,
    stages: Sequence[str] | None = None,
    split: SplitName = SplitName.VALIDATION,
    source_run: str | None = None,
) -> None:
    """Execute every stage of every dataset, aggregate, then write the manifest.

    The manifest is written in a `finally`, so an interrupted run still records
    how far it got. Cross-seed aggregation runs only after every stage has
    succeeded, and skips rather than raising when the series is short.

    One invocation runs one arm across every seed, so the arm is a parameter
    and not a loop.

    Args:
        config: The base configuration built from the command line.
        arm: Which arm to train and evaluate, one of config.ARMS.
        stages: Restrict the run to these stage names. None runs every stage
            the environment declares.
        split: Which split evaluation scores.
        source_run: Run evaluation reads shards and checkpoints from.

    Raises:
        ValueError: If the arm is unknown, if a requested stage name is not
            selectable, or if the selection matches no stage for this
            environment. The first two are raised before any stage is
            constructed.
        Exception: Anything a stage raises, re-raised after the summary and
            the manifest have been written.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    unknown = sorted(set(stages or ()) - set(SELECTABLE_STAGES))
    if unknown:
        raise ValueError(
            f"unknown stage(s) {unknown}, expected some of {SELECTABLE_STAGES}"
        )
    seed_configs = configs_for_datasets(config)
    built: list[Stage] = []
    for seed_config in seed_configs:
        built.extend(
            build_stages(seed_config, arm, stages, split, source_run)
        )
    if not built:
        raise ValueError(
            f"stage selection {sorted(set(stages or ()))} matched nothing for "
            f"environment {config.env.name!r}, which declares "
            f"{dataset_stage_names(config.env.name)} plus "
            f"{(STAGE_NAME_TRAIN, STAGE_NAME_EVALUATE)}. A run with no stages "
            "would report success having done nothing."
        )
    logger.info(RUN_START_BANNER, config.run_name, config.seed, len(built))
    logger.info(
        "%d datasets x %d stages, data seeds %s, observation mode %s, arm %d, "
        "split %s",
        len(seed_configs),
        len(built) // len(seed_configs),
        list(SEEDS),
        config.sampler.observation_mode,
        arm,
        split.value,
    )
    started = time.perf_counter()
    outcome = STAGE_FAILED
    try:
        for position, stage in enumerate(built, start=1):
            stage.execute(position, len(built))
        outcome = STAGE_SUCCESS
        aggregate_if_series_complete(config, arm=arm)
    finally:
        log_run_summary(
            built, config.run_name, time.perf_counter() - started, outcome
        )
        write_manifest(config)
