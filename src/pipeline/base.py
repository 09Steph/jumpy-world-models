"""Pipeline stage base class.

Defines the abstract contract for a coarse pipeline stage and wires in
sentinel-file skipping so a completed stage is not re-run unless its sentinel is
deleted. Concrete stages subclass this and implement ``run``.

Also owns the run log's stage boundaries: every stage logs a START and an END
banner with its wall-clock duration, and ``log_run_summary`` closes a run with
a per-stage roll-up.
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
    """Return the END banner's outcome suffix, EMPTY when the outcome is fine.

    Silence means success. A banner naming the outcome on every stage makes a
    failure compete for attention. The summary rows are unaffected.

    Args:
        outcome: STAGE_SUCCESS or STAGE_FAILED.

    Returns:
        "" for STAGE_SUCCESS, otherwise a leading-space-prefixed outcome so the
        banner reads "END FAILED" and not "ENDFAILED".
    """
    return "" if outcome == STAGE_SUCCESS else f" {outcome}"


def sampler_identity_fields(config: ExperimentConfig) -> dict:
    """Return the sampler settings any stage reading frozen windows is held to.

    Two stages are held to the same fields. A field that changes what the
    frozen windows contain has to appear in both identities, or one silently
    skips on a stale artefact.

    Args:
        config: The composed experiment configuration.

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

    Subclasses implement ``run`` with the stage's actual work and set
    ``name`` to a unique, filesystem-safe identifier used for the sentinel
    subdirectory.
    """

    name: str = "unnamed_stage"

    # Whether this stage's artefacts follow the data seed or the model seed.
    # Dataset stages follow data_seed, a trainer follows model_seed.
    dataset_derived: bool = False

    # Which arm produced this stage's artefacts, or None when every arm shares
    # them. None on the dataset stages stops every later arm rebuilding it.
    arm: int | None = None

    def __init__(self, config: ExperimentConfig) -> None:
        """Store the experiment configuration and initialise lifecycle state.

        Args:
            config: The composed experiment configuration for this run.
        """
        self.config = config
        # Lifecycle state, written by execute() and read by log_run_summary().
        # None means execute() was never called, which the summary must tell
        # apart from a stage that ran in 0.0s or was skipped.
        self.duration_seconds: float | None = None
        self.was_skipped: bool | None = None
        self.failed: bool = False

    @property
    def artefact_seed(self) -> int:
        """Return the seed that determines this stage's artefacts.

        Every path this stage writes, its sentinel included, keys on this
        and not on `config.seed`, so an artefact sits under the seed that
        decides its content.

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

        Defaults to `name` and exists so a stage can widen it. A stage keyed on
        something the path does not carry needs one sentinel per value of it,
        or the second overwrites the first's completion record.

        Separate from `name`, which is what the banners and the manifest use.
        This is the on-disk identity only.

        Returns:
            The sentinel subdirectory name, `name` unless a stage widens it.
        """
        return self.name

    @property
    def dataset_dir(self) -> Path:
        """Return the dataset directory this stage reads from or writes to.

        One function serves both directions, so a separately computed path
        cannot drift and leave a stage silently reading a stale dataset.

        Keyed on `data_seed` and the environment, never on `config.seed`.

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

        Whatever a stage returns here is what it is held to. A parameter left
        out can change without forcing a rerun, which is the stale-artefact
        failure this check exists to prevent.

        The base implementation returns the fields every stage shares. It is
        not empty; an empty dict restores pure existence checking.

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
        flag and recorded identity. Every one of those scopings is
        load-bearing.

        Records duration_seconds, was_skipped and failed on self.

        Args:
            position: This stage's 1-based position in the run, for the
                banner. Optional, and the banner omits the counter without it.
            total: How many stages this invocation will execute in all.

        Raises:
            Anything ``run()`` raises, re-raised unchanged, after a FAILED END
            banner carrying the elapsed time.

        Note:
            try/finally, not try/except, so it needs no broad-exception
            disable and still banners a KeyboardInterrupt or SystemExit. It
            sits outside ``run()``, so it runs after any signal handlers a
            stage installs and restores in its own finally.
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
            # Kept alongside the banner as the only line carrying run_name and
            # seed. A skip naming the wrong run_name hides in plain sight.
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
    """A stage whose artefacts are keyed on the ARM and the observation mode.

    Two stages are held to the same rules, and a field that scopes an artefact
    has to appear in both or one silently writes over the other. The dataset
    stages do not subclass this; they describe a dataset every arm shares and
    keep `Stage.arm = None`.

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
                construction, not at use, so an unknown arm cannot reach a path
                function and create a directory a later run reads as real.
        """
        super().__init__(config)
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
        self.arm = arm

    @property
    def sentinel_key(self) -> str:
        """Return a sentinel subdirectory scoped to the observation mode.

        One sentinel per mode. Mode-specific artefacts carry the mode in their
        filename and not as a directory level, so a single `<name>/done.json`
        could describe only one of them. The arm is absent from this key, being
        a directory level that `sentinels_dir` already separates.

        Returns:
            `"<name>_<observation_mode>"`, e.g. `"train_top_down"`.
        """
        return f"{self.name}_{self.config.sampler.observation_mode}"

    @property
    def observation_mode(self) -> ObservationMode:
        """Return the single observation mode this run uses.

        Returns:
            The configured observation mode, resolved from its string.
        """
        return ObservationMode(self.config.sampler.observation_mode)


def _stage_outcome(stage: Stage) -> str:
    """Return a stage's one-word outcome for the run summary.

    Args:
        stage: The stage to describe.

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
        stages: The stage objects in the order they were executed.
        run_name: The run name, without a seed suffix. The seed is a directory
            level, so a name carrying one would put it in the path twice.
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
