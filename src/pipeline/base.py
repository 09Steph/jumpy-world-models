"""Pipeline stage base class.

Defines the abstract contract for a pipeline stage, skips a stage whose sentinel
file exists, and wraps each stage in START and END log banners carrying its
wall-clock duration. Subclasses implement ``run``, and ``log_run_summary`` closes
a run with a per-stage roll-up.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from config import ARM_DIRECT, ARMS, ExperimentConfig, data_dir
from src.data.trajectory import ObservationMode
from src.utils.logging_setup import (
    RUN_END_BANNER,
    STAGE_END_BANNER,
    STAGE_SKIP_BANNER,
    STAGE_START_BANNER,
    format_duration,
    get_logger,
)
from src.utils.sentinels import is_stage_done, mark_stage_done

logger = get_logger(__name__)

STAGE_SUCCESS: str = "SUCCESS"
STAGE_FAILED: str = "FAILED"
NOT_REACHED: str = "not reached"


def banner_outcome(outcome: str) -> str:
    """Return the END banner's outcome suffix, empty when the outcome is fine.

    Args:
        outcome: STAGE_SUCCESS or STAGE_FAILED.

    Returns:
        "" for STAGE_SUCCESS, otherwise the outcome with a leading space.
    """
    return "" if outcome == STAGE_SUCCESS else f" {outcome}"


def sampler_identity_fields(config: ExperimentConfig) -> dict:
    """Return the sampler settings any stage reading frozen windows is held to.

    A field that changes what the frozen windows contain must appear in every
    reader's identity, or one silently skips on a stale artefact.

    Returns:
        JSON-serialisable identity fields describing the window draw.
    """
    return {
        "observation_mode": config.sampler.observation_mode,
        "sampler_mode": config.sampler.mode.value,
        "evaluation_horizons": list(config.sampler.evaluation_horizons),
        "evaluation_windows_per_trajectory": (
            config.sampler.evaluation_windows_per_trajectory
        ),
    }


class Stage(ABC):
    """A single coarse stage of the experiment pipeline.

    Subclasses implement ``run`` and set ``name`` to a unique,
    filesystem-safe identifier. The sentinel subdirectory comes from
    ``sentinel_key``, which defaults to ``name``.
    """

    name: str = "unnamed_stage"

    # Dataset stages follow data_seed, a trainer follows model_seed.
    dataset_derived: bool = False

    # Which arm produced this stage's artefacts, None when every arm shares
    # them. None on the dataset stages stops every arm rebuilding one dataset.
    arm: int | None = None

    def __init__(self, config: ExperimentConfig) -> None:
        """Store the experiment configuration and initialise lifecycle state."""
        self.config = config
        # Lifecycle state, written by execute() and read by log_run_summary().
        # None means execute() was never called; 0.0 means it skipped.
        self.duration_seconds: float | None = None
        self.was_skipped: bool | None = None
        self.failed: bool = False

    @property
    def artefact_seed(self) -> int:
        """Return the seed that determines this stage's artefacts.

        Every path this stage writes, its sentinel included, keys on this and
        never on `config.seed`.

        Returns:
            `config.data_seed` for a dataset-derived stage, otherwise
            `config.model_seed`.
        """
        return (
            self.config.data_seed if self.dataset_derived else self.config.model_seed
        )

    @property
    def sentinel_key(self) -> str:
        """Return the sentinel subdirectory name for this stage.

        A stage keyed on something its path does not carry needs one sentinel
        per value, or the second overwrites the first's completion record.
        Distinct from `name`, which the banners and the manifest use.

        Returns:
            The sentinel subdirectory name, `name` unless a stage widens it.
        """
        return self.name

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory this stage reads from or writes to.

        Always keyed on `data_seed`, including on stages whose own artefacts
        follow the model seed.

        Returns:
            The directory holding this dataset's shards and its derived
            artefacts.
        """
        return data_dir(
            self.config.run_name,
            self.config.data_seed,
            self.config.fast,
            self.config.env.name,
        )

    def sentinel_identity(self) -> dict:
        """Return the fields this stage's sentinel is held to on a rerun.

        A field left out can change without forcing a rerun, so whatever a
        subclass returns here is the whole of what it is held to. The base
        returns the fields every stage shares. An empty dict would reduce the
        check to bare existence.

        Returns:
            JSON-serialisable identity fields.
        """
        return {
            "data_seed": self.config.data_seed,
            "model_seed": self.config.model_seed,
            "env_name": self.config.env.name,
            "fast": self.config.fast,
        }

    @abstractmethod
    def run(self) -> None:
        """Execute the stage's work. Implemented by subclasses."""

    @staticmethod
    def _position_label(position: int | None, total: int | None) -> str:
        """Return the banner's position counter, or empty when unknown.

        Args:
            position: 1-based position of this stage in the run.
            total: How many stages the run will execute in all.

        Returns:
            e.g. ``"3/6"``, or ``""`` when either argument is None.
        """
        if position is None or total is None:
            return ""
        return f"{position}/{total}"

    def execute(self, position: int | None = None, total: int | None = None) -> None:
        """Run the stage unless its sentinel exists for this run_name and seed.

        Skips only on a matching run name, artefact seed, environment, fast
        flag, arm and recorded identity. Every one of those scopings is
        load-bearing. Records duration_seconds, was_skipped and failed on self.

        Args:
            position: This stage's 1-based position in the run. The banner
                omits the counter without it.
            total: How many stages this invocation will execute in all.

        Raises:
            Anything ``run()`` raises, re-raised unchanged, after a FAILED END
            banner carrying the elapsed time.
        """
        counter = self._position_label(position, total)
        identity = self.sentinel_identity()
        if is_stage_done(
            self.sentinel_key,
            self.config.run_name,
            self.artefact_seed,
            self.config.fast,
            identity,
            self.config.env.name,
            self.arm,
        ):
            self.was_skipped = True
            self.duration_seconds = 0.0
            logger.info(STAGE_SKIP_BANNER, counter, self.name)
            logger.info(
                "stage %s (run_name=%s seed=%s env=%s) already done -- skipping",
                self.name,
                self.config.run_name,
                self.artefact_seed,
                self.config.env.name,
            )
            return

        self.was_skipped = False
        logger.info(STAGE_START_BANNER, counter, self.name)
        started = time.perf_counter()
        completed = False
        try:
            self.run()
            completed = True
        finally:
            self.duration_seconds = time.perf_counter() - started
            if not completed:
                self.failed = True
                logger.error(
                    STAGE_END_BANNER,
                    counter,
                    self.name,
                    banner_outcome(STAGE_FAILED),
                    format_duration(self.duration_seconds),
                )

        # Re-read, never reusing the pre-run dict. Integrity fields such as
        # shard sizes do not exist until run() has written them.
        mark_stage_done(
            self.sentinel_key,
            self.config.run_name,
            self.artefact_seed,
            self.config.fast,
            self.sentinel_identity(),
            self.config.env.name,
            self.arm,
        )
        logger.info(
            STAGE_END_BANNER,
            counter,
            self.name,
            banner_outcome(STAGE_SUCCESS),
            format_duration(self.duration_seconds),
        )


class ArmScopedStage(Stage):
    """A stage whose artefacts are keyed on the arm and the observation mode.

    Attributes:
        arm: Which arm this instance is for, one of config.ARMS.
    """

    def __init__(
        self, config: ExperimentConfig, *, arm: int = ARM_DIRECT
    ) -> None:
        """Store the configuration and the arm.

        Args:
            config: The composed experiment configuration.
            arm: Which arm this stage is for, one of config.ARMS.

        Raises:
            ValueError: If the arm is not one of config.ARMS. Rejected at
                construction so an unknown arm cannot reach a path function
                and create a directory a later run reads as real.
        """
        super().__init__(config)
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
        self.arm = arm

    @property
    def sentinel_key(self) -> str:
        """Return a sentinel subdirectory scoped to the observation mode.

        One sentinel per mode. Both modes share a seed directory and carry the
        mode in the artefact filename rather than as a directory level, so a
        single `<name>/done.json` could describe only one of them. The arm is
        already a directory level under `sentinels_dir`.

        Returns:
            `"<name>_<observation_mode>"`, e.g. `"train_top_down"`.
        """
        return f"{self.name}_{self.config.sampler.observation_mode}"

    @property
    def observation_mode(self) -> ObservationMode:
        """Return the observation mode this run uses, resolved from its string."""
        return ObservationMode(self.config.sampler.observation_mode)


def _stage_outcome(stage: Stage) -> str:
    """Return a stage's one-word outcome for the run summary.

    Returns:
        "not reached", "FAILED", "skipped" or "ran".
    """
    if stage.duration_seconds is None:
        return NOT_REACHED
    if stage.failed:
        return STAGE_FAILED
    return "skipped" if stage.was_skipped else "ran"


def log_run_summary(
    stages: Sequence[Stage], run_name: str, total_seconds: float, outcome: str
) -> None:
    """Log the end-of-run banner and a per-stage roll-up.

    Emits one line per stage giving whether it ran, was skipped or failed, its
    wall-clock duration and its artefact seed. A stage whose duration_seconds
    is still None never reached execute() and is reported as "not reached",
    never as 0s.

    Args:
        stages: The stages, in execution order.
        run_name: The run name, without a seed suffix. The seed is already a
            directory level.
        total_seconds: Total wall-clock for the whole run.
        outcome: STAGE_SUCCESS or STAGE_FAILED.
    """
    logger.info(
        RUN_END_BANNER, run_name, banner_outcome(outcome),
        format_duration(total_seconds)
    )
    for stage in stages:
        state = _stage_outcome(stage)
        duration = (
            NOT_REACHED
            if stage.duration_seconds is None
            else format_duration(stage.duration_seconds)
        )
        logger.info(
            "  %-20s seed=%-5s %-11s %s",
            stage.name,
            stage.artefact_seed,
            state,
            duration,
        )
