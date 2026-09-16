"""Central configuration for the jumpy world model codebase.

All hyperparameters, paths, seeds and named constants, as constants and nested
dataclasses.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

from collections.abc import Sequence
import math
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.utils.paths import REPO_ROOT

# --- Reproducibility ---------------------------------------------------------
# Reporting seeds: every result quoted in the dissertation aggregates across
# exactly these five.

SEEDS: tuple[int, ...] = (42, 67, 999, 1, 500)
DEFAULT_SEED: int = SEEDS[0]

# Tuning seeds, disjoint from SEEDS. Any tuning pass runs on these only.
TUNING_SEEDS: tuple[int, ...] = (7, 13, 21)

# --- Repository-anchored paths ----------------------------------------------
# REPO_ROOT is defined in src/utils/paths.py, not recomputed here. Never log
# absolute paths, use src.utils.paths.rel().
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"

# --- Run-first output layout -------------------------------------------------
#
#     outputs/<run_name>/<env>/seed<n>/[arm<n>/]{checkpoints_<mode>,eval,logs,sentinels}/
#     outputs/fast/<run_name>/...
#
FAST_DIR_NAME: str = "fast"
SEED_DIR_PREFIX: str = "seed"
CHECKPOINTS_DIR_NAME: str = "checkpoints"
DATA_DIR_NAME: str = "data"
EVAL_DIR_NAME: str = "eval"
LOGS_DIR_NAME: str = "logs"
SENTINELS_DIR_NAME: str = "sentinels"

# --- The arm, a run axis and a directory level -------------------------------
# The arm sits below the shared dataset. `data_dir` is not arm-scoped, so every
# arm reads one set of shards.
ARM_DIR_PREFIX: str = "arm"
ARM_DIRECT: int = 1
ARM_AR_ENDPOINT: int = 2
ARM_AR_ONE_STEP: int = 3
ARMS: tuple[int, ...] = (ARM_DIRECT, ARM_AR_ENDPOINT, ARM_AR_ONE_STEP)

# The environment every path helper falls back to when no name is passed.
# Defined above EnvConfig because the path helpers must not depend on it;
# EnvConfig.name takes its default from here so the two cannot drift.
DEFAULT_ENV_NAME: str = "Navix-FourRooms-v0"

# --- Derived run manifest ----------------------------------------------------
# A VIEW over the sentinels, regenerated on every invocation and never an input
# to a skip decision. See src/pipeline/manifest.py.
MANIFEST_FILENAME: str = "manifest.json"
MANIFEST_SCHEMA_VERSION: int = 1
MANIFEST_STATE_MISSING: str = "missing"
MANIFEST_STATE_DONE: str = "done"
MANIFEST_STATE_STALE: str = "stale"
MANIFEST_STATES: tuple[str, ...] = (
    MANIFEST_STATE_MISSING,
    MANIFEST_STATE_DONE,
    MANIFEST_STATE_STALE,
)
# Dataset stage names. dataset_stage_names selects the subset one environment
# runs, and the runner and the manifest both read that answer.
STAGE_NAME_GENERATE: str = "generate"
STAGE_NAME_OFFLINE_GENERATE: str = "offline_generate"
STAGE_NAME_PREPARE: str = "prepare"
STAGE_NAME_DISPLACEMENT: str = "displacement"
# Keyed by a CANONICAL list rather than by what happens to exist on disk, so a
# stage that never ran still appears, as MANIFEST_STATE_MISSING. The union
# across environments, so it stays the list of every dataset stage this project
# has while the per-environment order comes from dataset_stage_names.
CANONICAL_STAGES: tuple[str, ...] = (
    STAGE_NAME_GENERATE,
    STAGE_NAME_OFFLINE_GENERATE,
    STAGE_NAME_PREPARE,
    STAGE_NAME_DISPLACEMENT,
)
# The two stages describing a fitted model rather than a dataset. Named here as
# strings so --stages can be validated at parse time; runner.build_stages
# asserts them against the classes' own names, so a rename cannot drift.
STAGE_NAME_TRAIN: str = "train"
STAGE_NAME_EVALUATE: str = "evaluate"
# Everything --stages may name. The union across environments, like
# CANONICAL_STAGES: an environment that declares no source stage still rejects
# the name at the parser rather than at the runner.
SELECTABLE_STAGES: tuple[str, ...] = CANONICAL_STAGES + (
    STAGE_NAME_TRAIN,
    STAGE_NAME_EVALUATE,
)
# Characters of the config hash kept. Long enough not to collide within one
# project, short enough to read in a log line.
CONFIG_DIGEST_LENGTH: int = 12


def run_root(
    run_name: str, seed: int, fast: bool = False, env: str | None = None
) -> Path:
    """Return the single directory holding every artefact of one run and seed.

    The path names what decides an artefact's content: the run, the environment
    and the seed. Observation mode is not a level, because one shard holds both
    modes; mode-specific artefacts carry the mode in their filename.

    Args:
        run_name: Experiment run name, WITHOUT any seed suffix.
        seed: The seed this artefact is scoped to. Callers pass the seed that
            determines the content: data_seed for dataset-derived artefacts,
            model_seed for run-derived ones.
        fast: When True the whole tree hangs off `outputs/fast/`.
        env: Registered environment name, defaulting to DEFAULT_ENV_NAME. A
            caller running a non-default environment must pass it, or its
            artefacts land in the default environment's tree.

    Returns:
        `outputs/[fast/]<run_name>/<env>/seed<seed>`.
    """
    base = OUTPUTS_DIR / FAST_DIR_NAME if fast else OUTPUTS_DIR
    return base / run_name / (env or DEFAULT_ENV_NAME) / f"{SEED_DIR_PREFIX}{seed}"


def arm_root(
    run_name: str,
    seed: int,
    fast: bool = False,
    env: str | None = None,
    *,
    arm: int | None = None,
) -> Path:
    """Return the root for artefacts that DEPEND on which arm produced them.

    `arm=None` means no arm level, which the dataset stages take: generation,
    preparation and the displacement diagnostic describe a dataset every arm
    shares. An arm level on those would make each arm rebuild the dataset,
    because `Stage.execute` skips on the sentinel alone.

    Args:
        run_name: Experiment run name, WITHOUT any seed suffix.
        seed: The seed this artefact is scoped to.
        fast: When True the whole tree hangs off `outputs/fast/`.
        env: Registered environment name, defaulting to DEFAULT_ENV_NAME.
        arm: One of ARMS, or None for an artefact every arm shares. Callers pass
            `Stage.arm` rather than deciding per call site.

    Returns:
        `run_root(...)` when arm is None, otherwise `run_root(...)/arm<arm>`.
    """
    root = run_root(run_name, seed, fast, env)
    if arm is None:
        return root
    return root / f"{ARM_DIR_PREFIX}{arm}"


def checkpoints_dir(  # pylint: disable=too-many-arguments
    run_name: str,
    seed: int,
    fast: bool = False,
    env: str | None = None,
    *,
    observation_mode: str,
    arm: int,
) -> Path:
    """Return the checkpoint directory for one run, seed and observation mode.

    The mode is in the leaf's name. Without it, two runs differing only in
    observation mode resolve to one directory, with no error and a plausible
    checkpoint on disk. Both scopings are keyword-only and required, so no
    caller can omit one.

    Args:
        run_name: Experiment run name, without any seed suffix.
        seed: The seed determining this artefact's content. A trained model
            follows the MODEL seed.
        fast: When True the tree hangs off `outputs/fast/`.
        env: Registered environment name, defaulting to DEFAULT_ENV_NAME.
        observation_mode: One of OBS_MODES, named in the directory.
        arm: One of ARMS. A directory level rather than part of this leaf name.

    Returns:
        `outputs/[fast/]<run_name>/<env>/seed<seed>/arm<arm>/checkpoints_<mode>`.
    """
    leaf = f"{CHECKPOINTS_DIR_NAME}_{observation_mode}"
    return arm_root(run_name, seed, fast, env, arm=arm) / leaf


def data_dir(
    run_name: str, seed: int, fast: bool = False, env: str | None = None
) -> Path:
    """Return the generated-trajectory directory for one run and seed.

    Holds the shard files and TRAJECTORY_STATS_FILENAME. Callers pass DATA_SEED
    here, because a dataset's content follows the data seed and not the model
    seed.
    """
    return run_root(run_name, seed, fast, env) / DATA_DIR_NAME


def eval_dir(
    run_name: str,
    seed: int,
    fast: bool = False,
    env: str | None = None,
    *,
    arm: int | None = None,
) -> Path:
    """Return the evaluation-artefact directory for one run, seed and arm.

    Arm-scoped when an arm is given. The metrics artefact differs by arm; the
    displacement diagnostic describes the dataset and passes None.

    Args:
        run_name: Experiment run name, without any seed suffix.
        seed: The seed determining this artefact's content.
        fast: When True the tree hangs off `outputs/fast/`.
        env: Registered environment name, defaulting to DEFAULT_ENV_NAME.
        arm: One of ARMS, or None for an artefact every arm shares.

    Returns:
        The evaluation directory, under `arm<arm>` when an arm is given.
    """
    return arm_root(run_name, seed, fast, env, arm=arm) / EVAL_DIR_NAME


def logs_dir(
    run_name: str,
    seed: int,
    fast: bool = False,
    env: str | None = None,
    *,
    arm: int | None = None,
) -> Path:
    """Return the log directory for one run, seed and arm.

    Args:
        run_name: Experiment run name, without any seed suffix.
        seed: The seed this run's log is scoped to.
        fast: When True the tree hangs off `outputs/fast/`.
        env: Registered environment name, defaulting to DEFAULT_ENV_NAME.
        arm: One of ARMS, or None. None is correct on the aggregation path,
            which spans seeds for an arm it is told separately.

    Returns:
        The log directory, under `arm<arm>` when an arm is given.
    """
    return arm_root(run_name, seed, fast, env, arm=arm) / LOGS_DIR_NAME


def displacement_sweep_dir(run_name: str, fast: bool = False) -> Path:
    """Return the directory holding one displacement sweep's artefacts.

    RUN-LEVEL, NOT run_root's. The sweep spans every registered environment and
    every reporting seed, so neither is a directory level: both are columns in
    the artefacts instead.

    Args:
        run_name: Name this sweep's artefacts are filed under.
        fast: When True the tree hangs off `outputs/fast/`.

    Returns:
        `outputs/[fast/]<run_name>`.
    """
    base = OUTPUTS_DIR / FAST_DIR_NAME if fast else OUTPUTS_DIR
    return base / run_name


def intermediate_states_dir(run_name: str, fast: bool = False) -> Path:
    """Return the directory holding one intermediate-state decode's artefacts.

    RUN-LEVEL, like the displacement sweep's. One invocation spans seeds, modes
    and horizons, so none of them is a directory level: each is a field in the
    figure name and a column in the records.

    Args:
        run_name: Name these artefacts are filed under.
        fast: When True the tree hangs off `outputs/fast/`.

    Returns:
        `outputs/[fast/]<run_name>`.
    """
    base = OUTPUTS_DIR / FAST_DIR_NAME if fast else OUTPUTS_DIR
    return base / run_name


def sentinels_dir(
    run_name: str,
    seed: int,
    fast: bool = False,
    env: str | None = None,
    *,
    arm: int | None = None,
) -> Path:
    """Return the sentinel directory for one run, seed and arm.

    The stage subdirectory and filename are added by src.utils.sentinels.

    The arm is passed per stage and is None for the dataset stages, whose
    sentinels every arm shares.

    Args:
        run_name: Experiment run name, without any seed suffix.
        seed: The seed this sentinel is scoped to.
        fast: When True the tree hangs off `outputs/fast/`.
        env: Registered environment name, defaulting to DEFAULT_ENV_NAME.
        arm: One of ARMS, or None for a stage every arm shares. Callers pass
            `Stage.arm`.

    Returns:
        The sentinel directory, under `arm<arm>` when an arm is given.
    """
    return arm_root(run_name, seed, fast, env, arm=arm) / SENTINELS_DIR_NAME


def run_manifest_path(
    run_name: str, fast: bool = False, env: str | None = None
) -> Path:
    """Return the derived run manifest's path, which spans every seed.

    One level above run_root, so no seed owns a record about the others. No date
    appears in it: a dated path would find no existing sentinel.

    Args:
        run_name: Experiment run name.
        fast: Whether this is a prototype run.
        env: Registered environment name, defaulting to DEFAULT_ENV_NAME.

    Returns:
        `outputs/[fast/]<run_name>/<env>/manifest.json`.
    """
    return run_root(run_name, DEFAULT_SEED, fast, env).parent / MANIFEST_FILENAME

# =============================================================================
# ENVIRONMENT OBSERVATION CONTRACT -- FACTS, NOT CHOICES
# =============================================================================
# Everything in this block describes what an environment EMITS.
#
# THE RULE: these constants are read by `NavixTrajectorySource` to build an
# `ObservationSpec`, AND BY NOTHING ELSE. Models take shape and cardinality as
# constructor arguments supplied from that spec. A model that imports one of
# these names has hardcoded the environment and the seam is fiction.

# Classes per NAVIX observation channel, for the per-cell categorical decoder.
# The observation is a (19, 19, 3) uint8 grid of discrete codes: entity tag,
# colour, symbolic state.
#
# NEVER read these from observation_space.maximum, which declares 8 and is
# wrong: measured channel-0 values reach 10. Checked by
# tests/test_navix_obs_cardinality.py.
OBS_CHANNEL_CLASSES_NAVIX: tuple[int, ...] = (11, 6, 4)

# Spatial shape of the NAVIX observation grid, (height, width).
# INVARIANT, asserted in tests: prod(shape) * len(channel_classes) == obs_dim.
OBS_GRID_SHAPE_NAVIX: tuple[int, int] = (19, 19)

# Egocentric NAVIX observation, (height, width).
# BOTH MODES SHARE OBS_CHANNEL_CLASSES_NAVIX: measured per-channel maxima are
# (10, 5, 3) top-down against (8, 5, 0) egocentric, so every egocentric code is
# a subset of its top-down channel. Cross-entropy is therefore comparable
# ACROSS modes, which is what experiment E4 depends on.
OBS_GRID_SHAPE_NAVIX_EGOCENTRIC: tuple[int, int] = (7, 7)

# Pixels per grid cell along each axis in a NAVIX RGB render, matching navix's
# own TILE_SIZE. It is what scales a symbolic grid to its rendered one.
NAVIX_TILE_SIZE: int = 8

# First-person RGB NAVIX observation, (height, width). navix renders the 7x7
# egocentric view at an 8-pixel tile, so the two carry the same scene.
OBS_GRID_SHAPE_NAVIX_RGB: tuple[int, int] = (56, 56)

# Colour channels in an RGB observation, and the inclusive bounds of a stored
# value. The encoder derives its normalisation divisor from the bounds.
OBS_CHANNELS_RGB: int = 3
OBS_VALUE_RANGE_RGB: tuple[int, int] = (0, 255)

# The LARGEST grid extent any NAVIX observation mode emits, in cells. Drives the
# encoder's DEPTH and nothing else.
#
# DERIVING DEPTH PER MODE WOULD GIVE EXPERIMENT E4 TWO ARMS OF DIFFERENT
# CAPACITY: dilations_for_grid(7) returns (1, 2) against (1, 2, 4, 8) for 19,
# and E4's only question is observability, so a capacity difference is the one
# confound it cannot carry.
#
# READ BY THE TRAINING STAGE AND PASSED IN AS depth_extent. It is not imported
# by the tokeniser or the encoder.
NAVIX_MAX_GRID_EXTENT: int = 19

# NAVIX observation function names for the two modes. Names rather than the
# functions themselves so config.py imports no environment library.
#
# `categorical` is deliberately NOT used for either mode: it returns a flat
# (19, 19) int32 grid whose values reach -1, and the per-cell categorical
# contract has no negative class.
OBS_FN_TOP_DOWN_NAVIX: str = "symbolic"
OBS_FN_EGOCENTRIC_NAVIX: str = "symbolic_first_person"

# The NetHack terminal region the offline observation is cut from, as
# (start, stop) row and column bounds into a (24, 80) tty frame. Row 0 is the
# message line and rows 22-23 are the status lines.
KATAKOMBA_MAP_ROWS: tuple[int, int] = (1, 22)
KATAKOMBA_MAP_COLS: tuple[int, int] = (0, 79)

# Spatial shape of the NLE observation grid, (height, width), and the classes
# per channel for the two-channel (tty_chars, tty_colors) state.
#
# Measured across all 38 Katakomba character builds: chars take 93 distinct
# values in 32-124 and colours 18 distinct values in 0-23. The colour channel
# has NO headroom, so an out-of-range code must raise rather than be clipped.
OBS_GRID_SHAPE_NLE: tuple[int, int] = (21, 79)
OBS_CHANNEL_CLASSES_NLE: tuple[int, ...] = (128, 24)

# Atari's stored frame. One greyscale channel over 84x84, which is what the
# archive holds and what the model consumes, so nothing is rescaled on the way
# in. The values are an intensity rather than a label, so the field declares a
# range and no channel classes.
OBS_GRID_SHAPE_ATARI: tuple[int, int] = (84, 84)
OBS_CHANNELS_GREYSCALE: int = 1
OBS_VALUE_RANGE_GREYSCALE: tuple[int, int] = (0, 255)

# Size of the offline action embedding table. Katakomba records raw terminal
# keypress bytes rather than NLE action indices, and they are carried through
# unmapped.
NLE_ACTION_VOCABULARY: int = 128

# The two observation modes, named as STRINGS because config.py is imported BY
# the data and model packages and must not import from them.
# src.data.trajectory.ObservationMode is the enum these resolve to, and
# tests/test_config.py pins the two spellings together.
#
# ONE MODE PER SAMPLER INSTANCE. Mode is a RUN axis and not a batch axis:
# mixing modes within a batch destroys the stationary-copy baseline, which is
# defined per mode and is the control every horizon result is read against.
OBS_MODE_TOP_DOWN: str = "top_down"
OBS_MODE_EGOCENTRIC: str = "egocentric"
OBS_MODES: tuple[str, ...] = (OBS_MODE_TOP_DOWN, OBS_MODE_EGOCENTRIC)

# The E3 slip grid, a RUN AXIS wired as main.py's --slip choices, so a value
# outside it cannot be run. Pre-registered before any E3 result was seen.
#
# EACH ENTRY IS THE PROBABILITY THAT AN ACTION IS RESAMPLED, not the probability
# that the executed action differs from the commanded one. The substitute is
# drawn over the whole action space, so the differing rate is
# p * (1 - 1 / num_actions). src.envs.slip.apply_slip is where that holds.
#
# 0.0 is the deterministic control and is not run as its own dataset: it is what
# the no-slip runs already produce.
SLIP_PROBABILITIES: tuple[float, ...] = (0.0, 0.1, 0.25)

# Channel and class index identifying the agent in a NAVIX observation.
#
# CODE 10 MARKS THE AGENT IN THE TOP-DOWN VIEW AND NEVER APPEARS IN THE
# EGOCENTRIC VIEW, because a first-person view is rendered from the agent's own
# cell. Agent-position accuracy is therefore a TOP-DOWN-ONLY DIAGNOSTIC: it
# carries no verdict and is marked undefined on the egocentric arm.
AGENT_CHANNEL_INDEX: int = 0
AGENT_CLASS_INDEX: int = 10

# Artefact filenames. POLICY_METRICS_FILENAME names an artefact no current
# stage writes; it is read by src/pipeline/aggregate.py.
#
# The metrics file carries the observation mode in its leaf name, so two runs
# differing only in mode cannot resolve to one file. METRICS_GLOB is the
# discovery half: the aggregation path builds no config, so it finds the mode by
# looking. It does not match POLICY_METRICS_FILENAME, which begins `policy_`.
METRICS_TEMPLATE: str = "metrics_{mode}.json"
METRICS_GLOB: str = "metrics_*.json"
POLICY_METRICS_FILENAME: str = "policy_metrics.json"

# The cross-arm artefact, written beside the per-arm aggregates in the
# environment directory. It carries no arm in its name: one file holds every
# arm, which is the axis it exists to put on one page.
SWEEP_FILENAME: str = "sweep_arms.json"

# Discovery half for the per-cell probability logs, whose filename template
# lives beside the code that writes them. The sweep references these by path
# and never opens them, so it finds them by looking rather than by rebuilding
# a name it does not own.
PROBABILITY_LOG_GLOB: str = "probability_log_*.npz"

# The fit artefact, written beside the sweep it reads. Separate from the sweep
# because the sweep carries measurements and this carries estimates over them.
FIT_ARTEFACT_NAME: str = "fit_arms.json"

# Filename for the per-mode error figure. Its only writer,
# plots.draw_error_against_horizon_figure, has no callers, so nothing writes it.
ERROR_HORIZON_FIGURE_FILENAME: str = "error_against_horizon.pdf"

# Print widths of the thesis template's text block. Every figure is saved at
# one of the two, so it is placed without rescaling.
THESIS_TEXT_WIDTH_IN: float = 6.5
THESIS_HALF_WIDTH_IN: float = THESIS_TEXT_WIDTH_IN / 2

# Directory under outputs/ the thesis figures are written to.
FIGURES_DIR_NAME: str = "figures"

# Horizons that figure marks. Unused: no module reads this.
ERROR_HORIZON_MARKER_HORIZONS: tuple[int, ...] = (1, 10, 100)

# Start points for the three-parameter fit error(h) = a + b*h^c, tried in this
# order. None in the first slot means the seed's error at the shortest horizon,
# so the first start is derived from the data and the other two are fixed.
#
# THREE RATHER THAN ONE BECAUSE THE START POINT SELECTS THE ANSWER ON A FLAT
# CURVE. Measured on the real series, the three disagree by over 100 in the
# exponent, and that disagreement is read as an identifiability signal
# alongside the standard error.
FIT_STARTS_THREE: tuple[tuple[float | None, float, float], ...] = (
    (None, 1e-3, 0.5),
    (0.08, 1e-3, 0.5),
    (0.0, 0.1, 1.0),
)

# Start point for the two-parameter fit error(h) = b*h^c, same None convention.
FIT_START_TWO: tuple[float | None, float] = (None, 0.01)

# Optimiser evaluation ceiling, passed to curve_fit as maxfev. The value every
# recorded fit measurement was taken at.
FIT_MAX_EVALUATIONS: int = 400_000

# Relative standard error at or below which an exponent counts as identified.
# One is a full hundred per cent, a generic convention rather than a threshold
# chosen against this project's curves.
FIT_IDENTIFIED_MAX_RSE: float = 1.0

# Absolute spread across FIT_STARTS_THREE above which the exponent counts as
# unidentified regardless of its standard error.
FIT_START_AGREE_TOL: float = 0.05

# The closed vocabulary an empty usable-horizon set reports. Lost at every
# horizon and could not be read are different findings with different
# consequences at the gate.
FIT_NO_WINNING_HORIZON: str = "no_horizon_beats_baseline"
FIT_INSUFFICIENT_SEEDS: str = "insufficient_seeds"
FIT_EMPTY_REASONS: tuple[str, ...] = (
    FIT_NO_WINNING_HORIZON,
    FIT_INSUFFICIENT_SEEDS,
)

# Trajectory generation artefacts, written by GenerateTrajectoriesStage.
# TRAJECTORY_STATS_FILENAME carries the episode-length histogram, the per-mode
# stationary-copy baselines and the mover-mask statistics.
TRAJECTORY_STATS_FILENAME: str = "trajectory_stats.json"
TRAJECTORY_SHARD_TEMPLATE: str = "trajectories_{index:04d}.h5"
# Matches every shard TRAJECTORY_SHARD_TEMPLATE can produce. A named constant
# rather than a pattern rebuilt at each call site: the integrity check counts
# and sizes these files, so a glob that silently stopped matching would report a
# complete dataset as empty.
TRAJECTORY_SHARD_GLOB: str = "trajectories_*.h5"
SPLIT_PROVENANCE_FILENAME: str = "split.json"
# The splits a frozen evaluation set is written for. TRAIN is excluded because
# it is not evaluated. Strings rather than the SplitName enum because config.py
# is imported BY src.data and must not import from it. They must match
# SplitName's values: prepare.frozen_evaluation_sets resolves SplitName(value)
# over this tuple, so a drift raises there rather than going unnoticed.
SPLIT_NAME_VALIDATION: str = "validation"
SPLIT_NAME_TEST: str = "test"
EVALUATED_SPLITS: tuple[str, ...] = (SPLIT_NAME_VALIDATION, SPLIT_NAME_TEST)
# Window-sampler artefacts, scoped by split name AND observation mode because
# both change what the file holds. An unscoped name would let a top-down
# evaluation set be read back into an egocentric run with nothing raising.
WINDOWS_EVAL_TEMPLATE: str = "windows_eval_{split}_{mode}.h5"
WINDOWS_TRAIN_TEMPLATE: str = "windows_train_{split}_{mode}.h5"
DECODED_SAMPLES_FILENAME: str = "decoded_samples.txt"

# Serialisation backend for the trajectory store. HDF5 following Minari's
# per-episode schema, so the generated dataset is releasable alongside the
# dissertation rather than stored in a format invented here.
STORAGE_FORMAT_HDF5: str = "hdf5"
STORAGE_FORMATS: tuple[str, ...] = (STORAGE_FORMAT_HDF5,)

# The offline corpora OfflineGenerateStage can read. `offline_source_for_env`
# resolves one per environment or family; OFFLINE_SOURCE itself is unused.
OFFLINE_SOURCE_KATAKOMBA: str = "katakomba"
OFFLINE_SOURCE_ATARI: str = "atari"
OFFLINE_SOURCE_ATARI_LONG: str = "atari-long"
OFFLINE_SOURCE_ATARI_P24: str = "atari-p24"
OFFLINE_SOURCE_ATARI_P49: str = "atari-p49"
OFFLINE_SOURCES: tuple[str, ...] = (
    OFFLINE_SOURCE_KATAKOMBA,
    OFFLINE_SOURCE_ATARI,
    OFFLINE_SOURCE_ATARI_LONG,
    OFFLINE_SOURCE_ATARI_P24,
    OFFLINE_SOURCE_ATARI_P49,
)
OFFLINE_SOURCE: str = OFFLINE_SOURCE_KATAKOMBA

# Where the Katakomba HDF5 corpus is read from. Resolved by `katakomba_root()`
# at call time, never at import: the corpus lives on an external volume and a
# module-level Path would make `import config` depend on it being mounted.
KATAKOMBA_ROOT_ENV_VAR: str = "JWM_KATAKOMBA_ROOT"
KATAKOMBA_ROOT_DEFAULT: Path = REPO_ROOT / "datasets" / "katakomba"

# Episodes the corpus admits, over 38 character builds. Episodes below the
# frame floor are refused rather than truncated, so this is the admitted count
# and not the corpus's record count.
KATAKOMBA_NUM_EPISODES: int = 26017

# Frames taken from each episode. Episode lengths run from 53 to 583,971 with a
# median of 28,208, so the cap binds on all but a handful and equalises what
# each episode contributes to a per-cell mean.
KATAKOMBA_MAX_EPISODE_STEPS: int = 256

# Where each episode's window starts.
#
# "random" draws a per-episode offset from the run's seeded PRNG key, "prefix"
# starts at frame 0, and "fixed_fraction" starts at
# KATAKOMBA_WINDOW_OFFSET_FRACTION of the usable range. A prefix of a
# median-28,208-step game measures the opening menus.
KATAKOMBA_WINDOW_OFFSET_RANDOM: str = "random"
KATAKOMBA_WINDOW_OFFSET_PREFIX: str = "prefix"
KATAKOMBA_WINDOW_OFFSET_FIXED_FRACTION: str = "fixed_fraction"
KATAKOMBA_WINDOW_OFFSET_MODES: tuple[str, ...] = (
    KATAKOMBA_WINDOW_OFFSET_RANDOM,
    KATAKOMBA_WINDOW_OFFSET_PREFIX,
    KATAKOMBA_WINDOW_OFFSET_FIXED_FRACTION,
)
KATAKOMBA_WINDOW_OFFSET_MODE: str = KATAKOMBA_WINDOW_OFFSET_RANDOM
# Read only when the mode is "fixed_fraction".
KATAKOMBA_WINDOW_OFFSET_FRACTION: float = 0.5

# Episodes per shard on the offline path, and the gzip level they are written
# at. Measured on real 256-frame windows: 0.850 MB raw against 0.029 MB at
# level 4, so the full corpus is 0.76 GB rather than 21 GB.
KATAKOMBA_SHARD_SIZE: int = 256
KATAKOMBA_SHARD_COMPRESSION: int = 4

# Environment name the offline dataset is written under. Distinct from any live
# NLE name, because `Stage.dataset_dir` is keyed by it and the two routes carry
# different data.
KATAKOMBA_ENV_NAME: str = "nle-katakomba"

# Provenance field the dataset split stratifies on, written per episode by
# src/data/katakomba_source.py. The corpus holds one file per character build
# and 678 to 696 episodes in each, so a stratified draw is a per-build shuffle
# with no proportional weighting.
#
# Stratifying makes each split representative of the corpus. It does NOT make
# the experiment a test of transfer between builds, and Methods states that.
STRATUM_PROVENANCE_KEY: str = "build"

# --- Atari, the ordered DQN Replay archive -----------------------------------
# Where the RLDS shards are read from, resolved by `atari_root()` at call time
# for the reason KATAKOMBA_ROOT_ENV_VAR gives.
ATARI_ROOT_ENV_VAR: str = "JWM_ATARI_ROOT"
ATARI_ROOT_DEFAULT: Path = REPO_ROOT / "datasets" / "atari"

# Environment name the offline dataset is written under, keying
# `Stage.dataset_dir`.
ATARI_ENV_NAME: str = "atari-dqn-replay"

# The long-horizon environment. It converts ATARI_LONG_HORIZON_GAME at
# ATARI_LONG_HORIZON steps and reports on LONG_HORIZON_EVALUATION_HORIZONS, so
# the window length, the game and the evaluation grid are selected together by
# the environment rather than set one at a time per invocation.
ATARI_LONG_HORIZON_ENV_NAME: str = "atari-dqn-replay-long"

# The staged ladder above stage 0. Same screen and same window as
# ATARI_ENV_NAME, differing only in which archive position is converted.
ATARI_LADDER_D_ENV_NAME: str = "atari-dqn-replay-p24"
ATARI_LADDER_E_ENV_NAME: str = "atari-dqn-replay-p49"

# Shard filenames, and which of them one conversion reads.
#
# Selection is scoped to a single training run before positions are chosen. The
# archive's five runs are independent DQN training seeds rather than quality
# levels, so drawing positions across runs varies the seed and the position
# together and neither can be read from the result.
ATARI_RUN_INDEX: int = 1
ATARI_SHARD_TEMPLATE: str = "run_{run}-{index:05d}-of-00050"
# Matches a shard at one position whatever its trailing shard count. Two games
# in the archive hold 49 shards rather than 50, so a name built from the
# template alone cannot open them.
ATARI_SHARD_GLOB: str = "run_{run}-{index:05d}-of-*"
# Archive positions one conversion reads. Stage 0 of the staged ladder is
# position 0 alone. The settings dict is keyed on positions, so adding one here
# would regenerate the stage-0 dataset rather than inherit the previous
# conversion; levels 24 and 49 therefore take their own environments below
# rather than extending this tuple.
ATARI_CHECKPOINT_POSITIONS: tuple[int, ...] = (0,)

# Levels 24 and 49 of the staged ladder, each converted under its own
# environment so stage 0's dataset sentinel is untouched. Reported as a
# robustness axis on checkpoint quality: the archive's positions are training
# checkpoints, so a later position is a stronger behaviour policy.
ATARI_LADDER_POSITIONS_D: tuple[int, ...] = (24,)
ATARI_LADDER_POSITIONS_E: tuple[int, ...] = (49,)

# The games that clear both gates on the archive matrix: a displacement at
# HORIZON_MAX of at least ATARI_DISPLACEMENT_BAR_PCT, and an episode at least
# ATARI_EPISODE_LENGTH_GATE decision steps long at ATARI_LENGTH_PERCENTILE.
# Ordered by measured displacement.
ATARI_DISPLACEMENT_BAR_PCT: float = 10.0
ATARI_EPISODE_LENGTH_GATE: int = 150
ATARI_LENGTH_PERCENTILE: int = 10
ATARI_GAME_POOL: tuple[str, ...] = (
    "Zaxxon",
    "UpNDown",
    "BankHeist",
    "Robotank",
    "BattleZone",
    "BeamRider",
    "YarsRevenge",
    "Krull",
    "TimePilot",
    "Riverraid",
    "Enduro",
    "SpaceInvaders",
    "CrazyClimber",
    "Assault",
    "Frostbite",
    "FishingDerby",
)

# Archive positions the checkpoint sweep measures for every game, and the
# episodes decoded at each.
ATARI_SWEEP_POSITIONS: tuple[int, ...] = (0, 12, 24, 37, 49)
ATARI_SWEEP_EPISODES: int = 40

# The games one conversion reads, as a subset of the pool. Adding, removing or
# reordering a name here is the whole change: no count is derived from the
# length of this tuple anywhere.
# The top five of the shipped ranking at archive position 0, in its order.
# Reproduce with --rank-atari-games --position 0.
ATARI_GAMES: tuple[str, ...] = (
    "Zaxxon",
    "UpNDown",
    "BankHeist",
    "Robotank",
    "BattleZone",
)

# The one game the long-horizon experiment runs on, selected by measured
# displacement across the reported archive positions subject to the episode
# length gate.
ATARI_LONG_HORIZON_GAME: str = "Robotank"
ATARI_LONG_HORIZON: int = 1024

# Frames an episode needs to supply one pair at the long horizon. Shorter
# episodes are refused rather than truncated, so every converted window reaches
# the horizon the environment exists to measure.
ATARI_LONG_HORIZON_MIN_EPISODE_FRAMES: int = ATARI_LONG_HORIZON + 1

# Episodes the long-horizon corpus admits from its game's shard. Episodes below
# ATARI_LONG_HORIZON_MIN_EPISODE_FRAMES are refused rather than truncated, so
# this is the admitted count and not the shard's record count.
ATARI_LONG_HORIZON_EPISODES_PER_CELL: int = 361

# Episodes taken from each game at each checkpoint position. Uniform across
# positions so sample size cannot confound with behaviour-policy quality: the
# count a shard supplies falls as the policy improves, and the smallest binds
# everywhere.
ATARI_EPISODES_PER_CELL: int = 150

# Frames taken from each episode, and where the window starts. Measured episode
# lengths across the selected games run from 299 to 5,082, so the cap binds on
# every one.
ATARI_MAX_EPISODE_STEPS: int = 256
ATARI_WINDOW_OFFSET_MODE: str = KATAKOMBA_WINDOW_OFFSET_RANDOM

# Episodes per shard on the Atari path, and the gzip level they are written at.
ATARI_SHARD_SIZE: int = 256
ATARI_SHARD_COMPRESSION: int = 4

# Actions are indices into each game's own action set, so a game using fewer
# than the full eighteen leaves the upper indices unused. One shared vocabulary
# keeps the action space identical across games.
ATARI_ACTION_VOCABULARY: int = 18

# Provenance fields recorded per episode. The game is what the split stratifies
# on, so each split carries every game in proportion.
ATARI_GAME_PROVENANCE_KEY: str = "game"
ATARI_CHECKPOINT_PROVENANCE_KEY: str = "checkpoint_idx"

# --- Dataset split -----------------------------------------------------------
# THREE-WAY, and the third split is not optional. GATE C reads VALIDATION
# repeatedly while decisions are made; the TEST split is opened once, at the
# end, with the configuration frozen. A split read repeatedly stops being held
# out.
#
# REMAINDER RULE: FLOOR TRAIN AND VALIDATION, GIVE THE REMAINDER TO TEST. The
# fractions define a proportion and not a partition, so without a stated rule
# the split is whatever the implementation happens to round to and the same seed
# yields different test sets on different machines. The surplus lands on the
# split the at-least-100-held-out bar binds on.

# SPLIT COMPOSITION CHECK. A random split by trajectory can hand one split an
# unusual share of SHORT episodes, and a short episode supports no window at
# high h at all -- an 11-step episode contributes nothing at h = 100. That lands
# exactly where the thesis is measured.
#
# THE CHECKED QUANTITY IS THE FRACTION OF TRAJECTORIES SUPPORTING h_max, not
# mean episode length. Mean length is a proxy; support decides whether a
# trajectory contributes at the reported horizon.
#
# IF IT FIRES: advance the split seed deterministically and redraw, bounded by
# SPLIT_MAX_REDRAW_ATTEMPTS. Hitting the cap raises rather than proceeding,
# because that means the dataset is skewed and no re-split can fix it.
SPLIT_SUPPORT_TOLERANCE: float = 0.05
SPLIT_MAX_REDRAW_ATTEMPTS: int = 10

TRAIN_FRACTION: float = 0.8
VALIDATION_FRACTION: float = 0.1
TEST_FRACTION: float = 0.1
SPLIT_FRACTION_SUM_TOLERANCE: float = 1e-9

# --- Displacement diagnostic -------------------------------------------------
# Artefacts written by DisplacementDiagnostic, under eval_dir(run_name, seed).
DISPLACEMENT_CSV_FILENAME: str = "displacement.csv"
DISPLACEMENT_FIGURE_FILENAME: str = "displacement.pdf"
DISPLACEMENT_CHANNELS_FIGURE_FILENAME: str = "displacement_channels.pdf"
DISPLACEMENT_VERDICT_FILENAME: str = "displacement_verdict.json"

# Horizons the diagnostic MARKS on the figure. The curve itself is computed at
# every horizon in [HORIZON_MIN, HORIZON_MAX]; these carry markers, error bars
# and sample counts. h = 10 and h = 100 are here because the saturation
# criterion and the ratio are read at them and must be visibly marked.
DISPLACEMENT_MARKER_HORIZONS: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100)

# THE BINDING SATURATION CRITERION. Saturation is declared when fewer than this
# percentage of grid cells differ between s_t and s_{t+100}.
#
# THE LEVEL, NOT A RATIO. Displacement plateaus in a BOUNDED environment
# whatever the policy does, so a ratio test is non-discriminating: measured
# top-down 1.25 and egocentric 1.17, so both fire, while one of those datasets
# is dead at 99.7% copy accuracy and the other is learnable at 72%.
SATURATION_ABSOLUTE_PCT_THRESHOLD: float = 10.0
SATURATION_BINDING_CRITERION: str = "absolute_pct_at_100 < 10"

# REPORTED, NOT BINDING. A free 2D random walk grows as sqrt(steps), so a
# tenfold horizon increase gives sqrt(100)/sqrt(10) = 3.16. A bounded gridworld
# can only fall below that.
FREE_DIFFUSION_RATIO_REFERENCE: float = 3.1622776601683795
SATURATION_RATIO_ANCHORS: tuple[int, int] = (10, 100)

AGENT_RISING_RATIO_MIN: float = 1.5
"""Separates branch 3, where generation proceeds and the saturated mode is
reported as a limitation, from branch 2, where generation is blocked pending a
collection-policy change. The test is strict, so this value is not rising."""


# --- Displacement sweep across the registry ----------------------------------
# Artefacts written by run_displacement_sweep_cli, under
# displacement_sweep_dir(run_name). The records file is JSON lines: one object
# per environment, seed, arm, view and horizon, appended as each environment and
# seed completes. --skip-existing skips any (environment, seed) with at least one
# record. The CSV is the tabular view and is rewritten from the records at the
# end.
DISPLACEMENT_SWEEP_RECORDS_FILENAME: str = "displacement_sweep.jsonl"
DISPLACEMENT_SWEEP_CSV_FILENAME: str = "displacement_sweep.csv"

# The horizons the sweep measures. DELIBERATELY NOT EVALUATION_HORIZONS. It
# drops h = 2 and h = 5, whose displacement separates no collection policy, and
# takes h = 200 from EXTRAPOLATION_HORIZONS so one grid spans the range the test
# split is scored over. The two grids meet at h = 100, which is where the
# saturation criterion is defined and where a swept figure is comparable with a
# stored verdict. Methods states the divergence.
DISPLACEMENT_SWEEP_HORIZONS: tuple[int, ...] = (1, 10, 25, 50, 100, 200)

# Rollout length, longer than EnvConfig.max_episode_steps so a pair at the
# widest horizon can exist inside one episode that started after a reset.
DISPLACEMENT_SWEEP_ROLLOUT_STEPS: int = 400


# --- Intermediate state decoding ---------------------------------------------
# Artefacts written by run_intermediate_states_cli, under
# intermediate_states_dir(run_name). The records file is JSON lines: one object
# per arm, rollout path and step, appended as each cell completes. The figure
# name carries every field that varies, so two cells never resolve to one file.
INTERMEDIATE_STATES_RECORDS_FILENAME: str = "intermediate_states.jsonl"
INTERMEDIATE_STATES_FIGURE_TEMPLATE: str = (
    "intermediate_states_{env}_seed{seed}_{mode}_{path}_h{horizon}.png"
)

# One row per cell, reduced from the records above. A separate artefact rather
# than rows inside the records file, which is what gets double counted.
INTERMEDIATE_STATES_SUMMARY_FILENAME: str = "intermediate_states_summary.json"

# Where a cell's true states come from. A held-out test-split episode carries
# every frame, so it supplies per-step truth and held-out provenance at once.
# The fresh roll is the fallback for a tree whose shards are not present.
TRUTH_SOURCE_TEST_TRAJECTORY: str = "test_trajectory"
TRUTH_SOURCE_FRESH_ROLLOUT: str = "fresh_rollout"
TRUTH_SOURCES: tuple[str, ...] = (
    TRUTH_SOURCE_TEST_TRAJECTORY,
    TRUTH_SOURCE_FRESH_ROLLOUT,
)

# The two rollouts an autoregressive arm has. TOKENS is the training path: the
# state stays a token and the decoder runs once, at the endpoint. OBSERVATIONS
# is the path a discrete arm is scored through, decoding and re-encoding an
# argmax every step; a continuous arm is scored through rollout_values. A figure
# drawn from one path and labelled with another is not reproducible, so the path
# is named in the artefact.
ROLLOUT_PATH_TOKENS: str = "tokens"
ROLLOUT_PATH_OBSERVATIONS: str = "observations"
ROLLOUT_PATHS: tuple[str, ...] = (
    ROLLOUT_PATH_TOKENS,
    ROLLOUT_PATH_OBSERVATIONS,
)

# The arms carrying intermediate states. Arm 1 predicts the endpoint in one
# forward pass and has nothing between its start and its endpoint to decode.
INTERMEDIATE_STATE_ARMS: tuple[int, ...] = (ARM_AR_ENDPOINT, ARM_AR_ONE_STEP)

# Episodes drawn while looking for one long enough to score a horizon. A
# uniform-random policy terminates early often enough that the first draw does
# not always reach h = 100.
INTERMEDIATE_STATES_MAX_EPISODE_DRAWS: int = 64

# PPO settings for the sweep's trained arm, mapped onto navix's PPOHparams by
# src/data/ppo_policy.py. The entropy coefficient is raised above navix's own
# default: under a policy that reaches no goal it is the only force acting, and
# the default collapses the policy to a near-motionless one.
# `DR248`: matched to the uniform datasets, which hold 501,455 transitions.
# A PPO arm sized differently from its control compares dataset SIZE rather
# than collection POLICY, which is the confound this experiment exists to avoid.
PPO_BUDGET_FRAMES: int = 500_000
PPO_NUM_ENVS: int = 64
PPO_NUM_STEPS: int = 128
PPO_ENTROPY_COEFFICIENT: float = 0.05

# The policy a dataset was collected under. The value reaches the generate
# sentinel identity and every trajectory's provenance, so two datasets
# differing only in how they were collected do not share a hash.
POLICY_UNIFORM_RANDOM: str = "uniform_random"
POLICY_PPO: str = "ppo"
COLLECTION_POLICIES: tuple[str, ...] = (POLICY_UNIFORM_RANDOM, POLICY_PPO)

# What policy_observation records for a policy that reads no observation, and
# the navix observation function a PPO policy is trained on otherwise.
POLICY_OBSERVATION_NOT_APPLICABLE: str = "not_applicable"
PPO_POLICY_OBSERVATION: str = "symbolic"


# How an observation is rendered. Separate from ObservationMode, which selects a
# viewpoint: RGB is a rendering of the egocentric viewpoint, not a third one.
# The value reaches the generate sentinel identity, so a symbolic and an RGB
# dataset of the same environment do not share a hash.
REPRESENTATION_SYMBOLIC: str = "symbolic"
REPRESENTATION_RGB: str = "rgb"
# One intensity per cell, which is what an Atari frame stores. Continuous like
# RGB and scored the same way, and separate from it because the channel count
# differs and a representation declares that rather than an environment.
REPRESENTATION_GREYSCALE: str = "greyscale"
REPRESENTATIONS: tuple[str, ...] = (
    REPRESENTATION_SYMBOLIC,
    REPRESENTATION_RGB,
    REPRESENTATION_GREYSCALE,
)

# The stored-value bounds each continuous representation declares. Keyed by
# representation, so a second continuous representation joins by adding a row.
CONTINUOUS_VALUE_RANGES: dict[str, tuple[int, int]] = {
    REPRESENTATION_RGB: OBS_VALUE_RANGE_RGB,
    REPRESENTATION_GREYSCALE: OBS_VALUE_RANGE_GREYSCALE,
}

# Which observation modes a dataset stores, by representation.
# DataConfig.stored_observation_modes overrides it. RGB stores the first-person
# view alone, because storing both fills the disk quota partway through.
DEFAULT_STORED_OBSERVATION_MODES: dict[str, tuple[str, ...]] = {
    REPRESENTATION_SYMBOLIC: OBS_MODES,
    REPRESENTATION_RGB: (OBS_MODE_EGOCENTRIC,),
    # A game screen is one view. It is filed under top_down because that is the
    # name every single-view environment here already uses, and calling it
    # first-person would put a false mode label on every artefact.
    REPRESENTATION_GREYSCALE: (OBS_MODE_TOP_DOWN,),
}


# --- Evaluation horizon grid --------------------------------------------------
# Its own declaration, not an alias of DISPLACEMENT_MARKER_HORIZONS: the two
# hold equal values independently, so editing either leaves the other alone.
# Equal tuple literals in one module are interned, so compare the values and
# never the identity. The reported endpoint ratio and fitted exponent are read
# on this grid and no other.
EVALUATION_HORIZONS: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100)

# The grid the long-horizon environment reports on. It EXTENDS the grid above
# rather than re-spacing it, so every horizon the other environments report on
# is also read here, and the four added points double to ATARI_LONG_HORIZON.
LONG_HORIZON_EVALUATION_HORIZONS: tuple[int, ...] = (
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    128,
    256,
    512,
    ATARI_LONG_HORIZON,
)

# Horizons scored past the trained ceiling, on the TEST pass alone. Measured
# support at h = 150 and h = 200: FourRooms 94.0 and 92.7 per cent,
# Dynamic-Obstacles 100 and 100, Katakomba 99.9 and 99.8.
#
# GLOBAL RATHER THAN PER ENVIRONMENT. Every tabled environment supports both, so
# nothing here varies by environment and a second per-environment table would be
# one more place to forget one.
#
# Training samples h ~ U(HORIZON_MIN, HORIZON_MAX), so these lie outside the
# training distribution: scoring here tests extrapolation of the jump function
# rather than interpolation within it.
EXTRAPOLATION_HORIZONS: tuple[int, ...] = (150, 200)

# The endpoints the reported error ratio is read on, as (denominator,
# numerator).
#
# LITERAL, NOT DERIVED FROM THE GRID ABOVE. It is a pre-registration, so it must
# not follow a grid that moves. The ratio was pre-registered as err(100)/err(1)
# and a gate has already passed on it.
REPORTED_RATIO_HORIZONS: tuple[int, int] = (1, 100)

# Per-step discount for the discounted compounding-error reading. The weight
# COMPOUNDING_DISCOUNT ** h decays geometrically, so on a grid reaching far past
# 1 / (1 - COMPOUNDING_DISCOUNT) that reading is set by the short horizons alone.
COMPOUNDING_DISCOUNT: float = 0.99

# Largest relative change in a cell's primary error a test-split re-score may
# make. Re-scoring one checkpoint on the same frozen windows moves it by
# reduction-order noise alone.
RESCORE_DRIFT_TOLERANCE: float = 1e-3


def validate_horizon_grid(horizons: tuple[int, ...], field_name: str) -> None:
    """Raise unless a horizon grid is non-empty, ascending and unique.

    Shared so one grid is not held to two definitions. SamplerConfig adds the
    HORIZON_MIN floor and is authoritative for the composed value; this runs at
    the declaration too, so a bad contract row fails where it is written.

    Args:
        horizons: The grid to check.
        field_name: The declaring field, named in the message.

    Raises:
        ValueError: If the grid is empty, or not strictly ascending and unique.
    """
    if not horizons:
        raise ValueError(
            f"{field_name} is empty: a frozen evaluation set with no horizons "
            "scores nothing"
        )
    ascending = list(horizons)
    if ascending != sorted(set(ascending)):
        raise ValueError(
            f"{field_name} must be strictly ascending and unique, got "
            f"{horizons}. A repeated horizon would be scored twice and "
            "weighted twice in any mean over the grid"
        )


def evaluation_grid_for_split(
    horizons: tuple[int, ...], split: str
) -> tuple[int, ...]:
    """Return the horizons one split is scored at.

    The test pass carries EXTRAPOLATION_HORIZONS; every other split is scored on
    the reporting grid alone. Confining the widening to one split leaves
    sampler_identity_fields unchanged for every run already on disk, so no
    existing prepare sentinel goes stale and no frozen window file is redrawn
    under a completed result.

    Args:
        horizons: The environment's reporting grid.
        split: The split being scored, one of EVALUATED_SPLITS.

    Returns:
        The grid, widened only on the test split.
    """
    if split != SPLIT_NAME_TEST:
        return horizons
    return tuple(sorted(set(horizons) | set(EXTRAPOLATION_HORIZONS)))


# --- Per-environment observation contract -------------------------------------
# One row per environment, holding the three ModelConfig observation fields and
# the encoder's depth extent. resolve_observation_contract applies the row after
# the --env override, so an environment name is the only thing a run sets.
#
# AN UNTABLED ENVIRONMENT RAISES rather than inheriting a row. Cardinality is a
# measured property of a layout: Navix-KeyCorridorS6R3-v0 puts values up to 254
# in channel 2 against the (11, 6, 4) declared here, so a silent default would
# be one --env away from a spec describing a different environment.

# Second stochasticity mechanism for E3. Obstacles move independently of the
# agent, so the transition is a distribution even with the action known.
DYNAMIC_OBSTACLES_ENV_NAME: str = "Navix-Dynamic-Obstacles-16x16-v0"

# Spatial shape of the Dynamic-Obstacles top-down grid, (height, width).
# Measured per-channel maxima are (10, 5, 3) top-down and (6, 5, 0) egocentric,
# both inside OBS_CHANNEL_CLASSES_NAVIX, so this layout shares NAVIX's
# cardinality and cross-entropy stays comparable across the two layouts.
OBS_GRID_SHAPE_DYNAMIC_OBSTACLES: tuple[int, int] = (16, 16)

# The room PPO collection targets. Chosen on displacement: uniform-random play
# moves 10.926 per cent here at h = 100 against FourRooms' 0.5429.
DOORKEY_ENV_NAME: str = "Navix-DoorKey-Random-5x5-v0"

# Spatial shape of the DoorKey top-down grid, (height, width).
OBS_GRID_SHAPE_DOORKEY: tuple[int, int] = (5, 5)


@dataclass(frozen=True)
class EnvironmentGeometry:
    """The grids one environment's map emits, per observation mode.

    Geometry only. How a grid of cells becomes stored numbers is a
    RepresentationSpec, and the two compose in resolve_observation_contract.
    A new map costs one row here; a new representation costs none.

    Attributes:
        grid_shapes: Grid shape per observation mode, (height, width), in
            cells. One entry per mode the environment emits.
        symbolic_channel_classes: Classes per channel of this environment's own
            discrete encoding. None where the environment emits no symbolic
            observation, which makes every discrete representation invalid for
            it.
        runs_displacement: Whether the displacement diagnostic is in this
            environment's stage list.
        displacement_note: Why the diagnostic is skipped, logged at WARNING
            where it is. Empty on an environment that runs it.
        evaluation_horizons: The grid this environment's results are reported
            on, derived from the episode lengths it produces.
        discrete_extent_pin: The extent a discrete run records, overriding the
            value max_grid_extent computes. Set only where completed runs
            recorded a different value that derives the same encoder.
    """

    grid_shapes: dict[str, tuple[int, int]]
    symbolic_channel_classes: tuple[int, ...] | None
    runs_displacement: bool
    displacement_note: str
    evaluation_horizons: tuple[int, ...]
    discrete_extent_pin: int | None = None

    def __post_init__(self) -> None:
        """Enforce the relations the declared values must satisfy.

        Raises:
            ValueError: If no mode is declared, if a mode is not one of
                OBS_MODES, if any side is non-positive, if a row that skips the
                displacement diagnostic gives no reason, or if the evaluation
                grid is empty or not strictly ascending.
        """
        if not self.grid_shapes:
            raise ValueError("grid_shapes must declare at least one mode")
        unknown = sorted(set(self.grid_shapes) - set(OBS_MODES))
        if unknown:
            raise ValueError(
                f"grid_shapes names modes {unknown} that are not in {OBS_MODES}"
            )
        for mode, shape in self.grid_shapes.items():
            if len(shape) != 2 or min(shape) <= 0:
                raise ValueError(
                    f"grid_shapes[{mode!r}] must be a positive (height, width), "
                    f"got {shape}"
                )
        if not self.runs_displacement and not self.displacement_note:
            raise ValueError(
                "a row that skips the displacement diagnostic must state why: "
                "displacement_note is empty"
            )
        validate_horizon_grid(self.evaluation_horizons, "evaluation_horizons")

    def emitted(self, modes: Sequence[str]) -> tuple[str, ...]:
        """Return which of the named modes this environment actually emits.

        A run asks for the modes its representation stores, and an environment
        that emits fewer contributes only those. NetHack emits one.

        Args:
            modes: Observation modes the run would store.

        Returns:
            The subset this environment emits, in the order given.

        Raises:
            ValueError: If it emits none of them.
        """
        available = tuple(mode for mode in modes if mode in self.grid_shapes)
        if not available:
            raise ValueError(
                f"this environment emits none of {list(modes)}; it declares "
                f"{sorted(self.grid_shapes)}"
            )
        return available

    def shapes_for(self, modes: Sequence[str]) -> tuple[tuple[int, int], ...]:
        """Return the grid shape of each mode this environment emits."""
        return tuple(self.grid_shapes[mode] for mode in self.emitted(modes))

    def largest_grid_shape(self, modes: Sequence[str]) -> tuple[int, int]:
        """Return the largest grid the named modes emit, by cell count."""
        return max(self.shapes_for(modes), key=math.prod)

    def max_grid_extent(self, modes: Sequence[str]) -> int:
        """Return the largest side of any of the named modes.

        Encoder depth derives from this, and the encoder must span the whole
        grid of whichever mode is trained, so the maximum runs over every mode
        the run stores rather than over the top-down one alone.
        """
        return max(max(shape) for shape in self.shapes_for(modes))


@dataclass(frozen=True)
class RepresentationSpec:
    """How a grid of cells becomes stored numbers.

    One entry per representation, shared by every environment. A discrete
    representation takes the environment's own channel classes; a continuous
    one declares its channel count and stored bounds here, because a rendering
    is a property of the renderer rather than of the map.

    Attributes:
        value_range: Inclusive (low, high) stored bounds for a continuous
            representation. None marks the representation discrete.
        channels: Values stored per cell for a continuous representation. None
            where the environment's channel classes supply the count.
        tile_size: Stored units per cell along each axis. One where a cell is
            itself the stored unit.
    """

    value_range: tuple[int, int] | None = None
    channels: int | None = None
    tile_size: int = 1

    def __post_init__(self) -> None:
        """Enforce that a continuous representation is fully declared.

        Raises:
            ValueError: If exactly one of value_range and channels is set, or
                if tile_size is not positive.
        """
        if (self.value_range is None) != (self.channels is None):
            raise ValueError(
                "a continuous representation declares BOTH value_range and "
                f"channels: got value_range={self.value_range!r}, "
                f"channels={self.channels!r}"
            )
        if self.tile_size < 1:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")

    def is_continuous(self) -> bool:
        """Return whether this representation stores continuous values."""
        return self.value_range is not None


# One row per map, carrying geometry alone. Every row reports on
# EVALUATION_HORIZONS. The field is required rather than defaulted because the
# grid follows from the episode lengths an environment produces, and those were
# measured per environment rather than assumed.
ENVIRONMENT_GEOMETRIES: dict[str, EnvironmentGeometry] = {
    DEFAULT_ENV_NAME: EnvironmentGeometry(
        grid_shapes={
            OBS_MODE_TOP_DOWN: OBS_GRID_SHAPE_NAVIX,
            OBS_MODE_EGOCENTRIC: OBS_GRID_SHAPE_NAVIX_EGOCENTRIC,
        },
        symbolic_channel_classes=OBS_CHANNEL_CLASSES_NAVIX,
        runs_displacement=True,
        displacement_note="",
        evaluation_horizons=EVALUATION_HORIZONS,
    ),
    DYNAMIC_OBSTACLES_ENV_NAME: EnvironmentGeometry(
        grid_shapes={
            OBS_MODE_TOP_DOWN: OBS_GRID_SHAPE_DYNAMIC_OBSTACLES,
            OBS_MODE_EGOCENTRIC: OBS_GRID_SHAPE_NAVIX_EGOCENTRIC,
        },
        symbolic_channel_classes=OBS_CHANNEL_CLASSES_NAVIX,
        runs_displacement=True,
        displacement_note="",
        evaluation_horizons=EVALUATION_HORIZONS,
    ),
    # THE ROOM WHERE EGOCENTRIC EXCEEDS TOP-DOWN, 7x7 against 5x5. Declaring
    # both shapes is what settles which reading max_grid_extent takes.
    #
    # discrete_extent_pin holds the value this room's completed runs recorded.
    # The two-mode maximum is 7; dilations_for_grid returns (1, 2) at 5 and at
    # 7, and the widths derive from that depth, so both build the same encoder
    # and only the recorded value would move.
    DOORKEY_ENV_NAME: EnvironmentGeometry(
        grid_shapes={
            OBS_MODE_TOP_DOWN: OBS_GRID_SHAPE_DOORKEY,
            OBS_MODE_EGOCENTRIC: OBS_GRID_SHAPE_NAVIX_EGOCENTRIC,
        },
        symbolic_channel_classes=OBS_CHANNEL_CLASSES_NAVIX,
        runs_displacement=True,
        displacement_note="",
        evaluation_horizons=EVALUATION_HORIZONS,
        discrete_extent_pin=5,
    ),
    # symbolic_channel_classes is None: the archive stores rendered frames and
    # no symbolic grid, so a discrete representation cannot be resolved for it.
    ATARI_ENV_NAME: EnvironmentGeometry(
        grid_shapes={OBS_MODE_TOP_DOWN: OBS_GRID_SHAPE_ATARI},
        symbolic_channel_classes=None,
        runs_displacement=False,
        displacement_note=(
            "displacement is measured on the archive before conversion, "
            "across the whole game pool rather than per run, and the "
            "selection recorded in ATARI_GAME_POOL is its output"
        ),
        evaluation_horizons=EVALUATION_HORIZONS,
    ),
    # The same screen and window as ATARI_ENV_NAME. They differ only in which
    # archive position OFFLINE_SOURCE_BY_ENV converts, so the geometry and the
    # evaluation grid are identical to stage 0's and the results are comparable.
    ATARI_LADDER_D_ENV_NAME: EnvironmentGeometry(
        grid_shapes={OBS_MODE_TOP_DOWN: OBS_GRID_SHAPE_ATARI},
        symbolic_channel_classes=None,
        runs_displacement=False,
        displacement_note=(
            "displacement is measured on the archive before conversion, "
            "across the whole game pool rather than per run, and the "
            "selection recorded in ATARI_GAME_POOL is its output"
        ),
        evaluation_horizons=EVALUATION_HORIZONS,
    ),
    ATARI_LADDER_E_ENV_NAME: EnvironmentGeometry(
        grid_shapes={OBS_MODE_TOP_DOWN: OBS_GRID_SHAPE_ATARI},
        symbolic_channel_classes=None,
        runs_displacement=False,
        displacement_note=(
            "displacement is measured on the archive before conversion, "
            "across the whole game pool rather than per run, and the "
            "selection recorded in ATARI_GAME_POOL is its output"
        ),
        evaluation_horizons=EVALUATION_HORIZONS,
    ),
    # The same screen as the row above. It differs in the corpus it converts,
    # which OFFLINE_SOURCE_BY_ENV maps, and in the grid it reports on.
    ATARI_LONG_HORIZON_ENV_NAME: EnvironmentGeometry(
        grid_shapes={OBS_MODE_TOP_DOWN: OBS_GRID_SHAPE_ATARI},
        symbolic_channel_classes=None,
        runs_displacement=False,
        displacement_note=(
            "displacement is measured on the archive before conversion, "
            "across the whole game pool rather than per run, and the "
            "selection recorded in ATARI_GAME_POOL is its output"
        ),
        evaluation_horizons=LONG_HORIZON_EVALUATION_HORIZONS,
    ),
    KATAKOMBA_ENV_NAME: EnvironmentGeometry(
        grid_shapes={OBS_MODE_TOP_DOWN: OBS_GRID_SHAPE_NLE},
        symbolic_channel_classes=OBS_CHANNEL_CLASSES_NLE,
        runs_displacement=False,
        displacement_note=(
            "its saturation verdict is already recorded at "
            "absolute_pct_at_100 8.13, and the stage costs hours per seed"
        ),
        evaluation_horizons=EVALUATION_HORIZONS,
    ),
}


# One entry per representation, shared by every environment.
REPRESENTATION_SPECS: dict[str, RepresentationSpec] = {
    REPRESENTATION_SYMBOLIC: RepresentationSpec(),
    REPRESENTATION_RGB: RepresentationSpec(
        value_range=OBS_VALUE_RANGE_RGB,
        channels=OBS_CHANNELS_RGB,
        tile_size=NAVIX_TILE_SIZE,
    ),
    # tile_size stays one: the archive stores the frame already rendered, so a
    # cell is the stored unit and the grid needs no scaling.
    REPRESENTATION_GREYSCALE: RepresentationSpec(
        value_range=OBS_VALUE_RANGE_GREYSCALE,
        channels=OBS_CHANNELS_GREYSCALE,
    ),
}


@dataclass(frozen=True)
class EnvConfig:
    """Environment settings.

    Attributes:
        name: Registered environment identifier, checked by
            tests/test_navix_api_verification.py.
        num_envs: Parallel environment count.
        max_episode_steps: Hard cap on episode length, passed as
            navix.make()'s max_steps. Must be passed explicitly; NAVIX's own
            default is 100.
        penality_coeff: Coefficient on NAVIX's built-in time penalty. Zero,
            which makes undiscounted episode return the success indicator.
        slip_probability: One of SLIP_PROBABILITIES. The probability that a
            commanded action is resampled uniformly over the action space
            before the environment executes it. Zero is the deterministic
            default, so every run definition predating E3 is unchanged.
        disable_early_termination: Replace the environment's termination
            function with never_terminate, so only max_episode_steps ends an
            episode. Off by default, and recorded in the generate stage's
            sentinel identity, so changing it reruns generation.
        max_grid_extent: The larger of the observation grid's height and width,
            for the largest mode the environment emits. Drives the encoder's
            depth and nothing else. Set from ENVIRONMENT_GEOMETRIES by
            resolve_observation_contract.
    """

    name: str = DEFAULT_ENV_NAME
    num_envs: int = 64
    max_episode_steps: int = 256
    penality_coeff: float = 0.0
    slip_probability: float = 0.0
    disable_early_termination: bool = False
    max_grid_extent: int = NAVIX_MAX_GRID_EXTENT


# =============================================================================
# ARCHITECTURE -- CHOICES, NOT FACTS
# =============================================================================
# PROVENANCE TAGS. Every architecture value below carries one:
#
#   [MEASURED]  A fact about the environment, obtained by running it.
#   [DERIVED]   Computed from a measured fact plus a stated rule.
#   [SOURCED]   Taken from a published paper, cited at the point of use.
#   [CHOSEN]    Picked by the implementer with no citation and no derivation.
#               *** PROVISIONAL. Every [CHOSEN] value names what would revise
#               it. *** Do not tune these against a validation curve
#               mid-project and leave the record saying they were chosen.

# --- State tokenisation -------------------------------------------------------
# A string rather than an Enum because config.py is imported BY the model
# packages and must not import from them; the tokeniser resolves the name.
STATE_TOKENS_ONE_TOKEN: str = "one_token"
STATE_TOKENS_PER_CELL: str = "per_cell"
STATE_TOKENS_PATCH: str = "patch"

# --- Encoder receptive field --------------------------------------------------
# Stacked 3x3 stride-1 convolutions grow the receptive field as 1 + 2*sum(d).
# THIS IS A CORRECTNESS CONSTRAINT, not a tuning knob: the model predicts the
# state after up to HORIZON_MAX actions, by which point the agent may be
# anywhere, so the encoder must see the whole grid BEFORE pooling discards
# spatial position. Mean pooling does not substitute -- it averages features
# that each saw only their own receptive field.
#
#   plain, 2 layers   ->  5    does NOT cover 19x19
#   dilated 1,2,4,8   -> 31    covers 19x19 in FOUR layers
#   dilated to 32     -> 127   covers 21x79 in SIX layers
#
# Dilation buys receptive field at NO parameter cost.
ENCODER_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)

# NLE schedule, reaching 127 against a 79-wide map. Not the default because it
# is two layers deeper than NAVIX needs.
ENCODER_DILATIONS_NLE: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

# --- Attention span -----------------------------------------------------------
# None means FULL bidirectional attention, which is the default and the built
# behaviour. An integer means non-causal LOCAL attention at that window.
#
# THE COST IS QUADRATIC IN HORIZON_MAX, so the full-attention default is not
# scale-free: at 1 + 100 tokens it is roughly 10,200 pairs, which is nothing,
# but raising the ceiling to the 256-step episode cap would cost about 6.5x.
# Re-check this default before raising HORIZON_MAX, do not assume it carries.
ATTENTION_WINDOW: int | None = None

# --- Horizon ------------------------------------------------------------------
# h ~ U(HORIZON_MIN, HORIZON_MAX), sampled per training example. h = 0 is
# rejected at the sampler and the tokeniser rather than clamped here: a
# zero-step prediction is not a prediction, and silently correcting it would
# hide a sampler defect.
HORIZON_MIN: int = 1
HORIZON_MAX: int = 100

# --- Encoder trunk widths, one set per environment family ---------------------
# One width per dilated layer. dilations_for_grid derives the LENGTH from the
# environment's extent, so a set whose length does not match raises in
# encoder_for_spec. Extent 19 derives four layers, extent 79 derives six.
#
# The NLE set keeps the NAVIX widths as an exact prefix and adds two layers at
# the trailing width, so an NLE encoder is the NAVIX encoder plus two
# wider-dilation layers. Depth already scales the model by derivation and
# code_embed_dim already scales per-cell richness over 152 codes.
#
# WIDTHS ARE [CHOSEN], provisional. No metric and no citation separates the
# three NLE sets: DPWM Table 3 sources the transformer only and publishes no
# encoder width schedule.
# The width schedule, stated as a rule rather than a per-family tuple (`DR245`).
# Both tuples this project committed to are "two narrow layers, then wide":
# NAVIX is (64, 64, 128, 128) at four layers and NLE is
# (64, 64, 128, 128, 128, 128) at six. `encoder_widths_for_depth` reproduces
# both exactly, so no completed result changes, and it answers any depth the
# grid extent derives instead of raising when the lengths disagree.
ENCODER_NARROW_WIDTH: int = 64
ENCODER_WIDE_WIDTH: int = 128
ENCODER_NARROW_LAYERS: int = 2

ENCODER_CHANNELS_NAVIX: tuple[int, ...] = (64, 64, 128, 128)
NLE_ENCODER_CHANNELS: tuple[int, ...] = (64, 64, 128, 128, 128, 128)
NLE_ENCODER_CHANNELS_SCALED: tuple[int, ...] = (96, 96, 192, 192, 192, 192)
NLE_ENCODER_CHANNELS_TAPERED: tuple[int, ...] = (64, 64, 128, 128, 256, 256)

# Names --encoder-widths accepts, and the set each selects. Every set is six
# layers, and main.py applies the override after the widths are derived, so on a
# four-layer environment the build raises in encoder_for_spec.
ENCODER_WIDTHS_DEFAULT: str = "default"
ENCODER_WIDTHS_SCALED: str = "scaled"
ENCODER_WIDTHS_TAPERED: str = "tapered"
ENCODER_WIDTH_CANDIDATES: dict[str, tuple[int, ...]] = {
    ENCODER_WIDTHS_DEFAULT: NLE_ENCODER_CHANNELS,
    ENCODER_WIDTHS_SCALED: NLE_ENCODER_CHANNELS_SCALED,
    ENCODER_WIDTHS_TAPERED: NLE_ENCODER_CHANNELS_TAPERED,
}


@dataclass(frozen=True)
class ModelConfig:  # pylint: disable=too-many-instance-attributes
    """Model architecture settings.

    Holds the observation contract and the Stage 1 architecture. The
    attribute-count limit is disabled locally rather than raised project-wide:
    this is a frozen data container, and the limit exists for classes with
    behaviour.

    Attributes:
        obs_dim: Flattened observation dimensionality. 1083 = 19*19*3 for NAVIX.
        obs_grid_shape: **[MEASURED.]** Spatial shape of the observation grid,
            (height, width).
        obs_channel_classes: **[MEASURED.]** Classes per observation channel.
            The codes are unordered labels, which is why the loss is
            cross-entropy rather than MSE.
        activation: Activation used in the model's MLP blocks.
        code_embed_dim: Width of the learned embedding each discrete cell code
            is looked up in, per channel. A lookup rather than a one-hot: an
            embedding table is (vocabulary, width) whatever the vocabulary.
            **[CHOSEN], provisional.**
        encoder_channels: Convolution widths of the encoder trunk, one entry per
            layer. Parameter count is independent of grid size, so the same
            encoder serves both observation modes. **LENGTH is [DERIVED],
            WIDTHS are [CHOSEN], provisional.** THE LENGTH IS NOT CHECKED HERE
            AND CANNOT BE, because the depth follows from the environment's OWN
            spec at runtime; encoder_for_spec enforces it at the point of use.
        decoder_channels: Convolution widths of the decoder trunk.
            **[CHOSEN], provisional.** UNDILATED, deliberately: the decoder
            EXPANDS from a vector that is already global, so its convolutions
            only need to make neighbouring cells locally coherent.
        d_model: **[SOURCED, DPWM Table 3.]** Transformer token width.
        num_layers: **[SOURCED, DPWM Table 3.]** Transformer encoder depth.
        num_heads: **[SOURCED, DPWM Table 3.]** Attention heads per layer.
            d_model must divide by this.
        ffn_multiplier: **[SOURCED, DPWM Table 3.]** Feed-forward width as a
            multiple of d_model.
        dropout_rate: **[SOURCED, DPWM Table 3.]** Dropout inside the
            transformer.
        use_token_type_embedding: **[CHOSEN] OFF.** Whether to add a learned
            per-kind vector marking state tokens apart from action tokens.
            Built and testable but off. See src/models/token_type.py for the
            three triggers that would justify turning it on.
        state_tokens: How an observation becomes tokens; one of the
            STATE_TOKENS_* constants.
        norm_eps: RMSNorm epsilon.
        remat_rollout: **[CHOSEN] OFF, and it is a COMPUTE switch that cannot
            change a number.** Recompute the autoregressive rollout's
            activations on the backward pass instead of storing one set per
            step.

            **AN OUT-OF-MEMORY RESULT WITH THIS OFF IS NOT AN ANSWER ABOUT
            WHETHER ARM 2 FITS.** The remat retry must run before arm 2 is
            declared unaffordable, or the fallbacks fire on a false premise.

            Absent from sentinel_identity: rematerialisation reaches the same
            gradient, identical to float precision rather than bit-identical.
    """

    obs_dim: int = 1083
    obs_grid_shape: tuple[int, int] = OBS_GRID_SHAPE_NAVIX
    obs_channel_classes: tuple[int, ...] | None = OBS_CHANNEL_CLASSES_NAVIX
    obs_channels: int = len(OBS_CHANNEL_CLASSES_NAVIX)
    activation: str = "silu"
    code_embed_dim: int = 32
    encoder_channels: tuple[int, ...] = ENCODER_CHANNELS_NAVIX
    decoder_channels: tuple[int, ...] = (128, 64)
    d_model: int = 256
    num_layers: int = 5
    num_heads: int = 4
    ffn_multiplier: int = 4
    dropout_rate: float = 0.1
    state_tokens: str = STATE_TOKENS_ONE_TOKEN
    use_token_type_embedding: bool = False
    norm_eps: float = 1e-4
    remat_rollout: bool = False

    def __post_init__(self) -> None:
        """Enforce the architecture constraints this class already declares.

        Checking them at construction is what makes fast mode safe to scale the
        model: every config is validated where it is written, rather than a
        violation being stumbled over whenever a full-size run happens to be
        made.

        Raises:
            ValueError: If any declared constraint is violated.
        """
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must divide by num_heads: "
                f"{self.d_model} % {self.num_heads} != 0"
            )
        # A continuous observation has no embedding table, so the bound is
        # on the discrete path alone.
        if self.obs_channel_classes is not None:
            embedded = self.code_embed_dim * len(self.obs_channel_classes)
            if embedded > self.d_model:
                raise ValueError(
                    "code_embed_dim * channels must not exceed d_model, so the "
                    f"trunk expands rather than compresses: {embedded} > "
                    f"{self.d_model}"
                )
            if len(self.obs_channel_classes) != self.obs_channels:
                raise ValueError(
                    "obs_channels must match the declared classes: "
                    f"{self.obs_channels} against "
                    f"{len(self.obs_channel_classes)}"
                )
        # THE THIRD CONSTRAINT IS DELIBERATELY NOT CHECKED HERE. The true depth
        # depends on the SPEC, and this file cannot know which environment will
        # be used without importing the model packages that import it. The
        # check lives where the derivation does, in encoder_for_spec, covered
        # by tests/test_layers.py::test_channel_widths_must_match_the_derived_depth.


# --- Optimisation -------------------------------------------------------------
# DPWM's settings, adopted where they transfer. ADAM RATHER THAN ADAMW: AdamW is
# the transformer convention and DPWM specifies plain Adam, and matching the
# comparator beats following convention where the two disagree.
#
# THE CAVEAT TRAVELS WITH THE NUMBERS AND REACHES METHODS: DPWM MEASURED THEM ON
# CONTINUOUS CONTROL AND PIXEL BENCHMARKS, NOT ON A DISCRETE SYMBOLIC GRID.
OPTIMISER_ADAM: str = "adam"
OPTIMISERS: tuple[str, ...] = (OPTIMISER_ADAM,)

@dataclass(frozen=True)
class TrainConfig:  # pylint: disable=too-many-instance-attributes
    """Training loop settings.

    The attribute-count limit is disabled locally for the reason ModelConfig
    carries: a frozen data container with no behaviour beyond validation.

    Attributes:
        total_steps: Gradient steps for a full run. Read back from the config
            snapshot by aggregate.py to reject a truncated seed. ALSO THE COSINE
            SCHEDULE'S PERIOD. Changing it re-anneals the learning rate rather
            than merely lengthening or truncating the run, so two budgets are
            two optimisations and their curves are not directly comparable.
        batch_size: Examples per gradient step. **[CHOSEN], and a STATED
            DEVIATION from DPWM's 256 and 128.**
        learning_rate: Peak optimiser learning rate. **[SOURCED, DPWM.]**
        optimiser_name: Which optimiser build_optimiser assembles. One of
            OPTIMISERS. Plain Adam, not AdamW.
        lr_schedule_final: Floor the cosine schedule anneals TO.
            **[SOURCED, DPWM.]** Expressed as an absolute rate here and
            converted to optax's `alpha` fraction inside build_optimiser, so the
            config states the quantity DPWM published.
        grad_clip_norm: Maximum global gradient norm. **[SOURCED, DPWM.]**
            Applied BEFORE the optimiser, so the bound is on the raw gradient.
        log_every_steps: Logging interval.
        horizon_min: Smallest sampled horizon. One, never zero.
        horizon_max: Largest sampled horizon, and the length every action
            sequence is padded to. The count of unmasked action tokens IS the
            horizon, so no separate horizon token is supplied.

            A horizon cannot exceed the episode it is cut from, and distinct
            start positions for horizon h number L - h + 1, so a horizon near
            the episode cap leaves one window per trajectory.
        checkpoint_every_steps: Gradient steps between periodic checkpoints.
            ZERO TURNS PERIODIC SAVING OFF and leaves only the final write.
            THE VALUE IS ALSO THE RESUME GRANULARITY: a run resumes from the
            last periodic write, so the expected work lost to an interruption is
            half of this. DERIVE IT FROM `total_steps`: left behind by a budget
            change it silently becomes a much larger fraction of a run.
        target_examples: Number of (state, action sequence, endpoint) examples
            the dataset build targets.
    """

    total_steps: int = 50_000
    batch_size: int = 16
    learning_rate: float = 3e-4
    optimiser_name: str = OPTIMISER_ADAM
    lr_schedule_final: float = 1e-10
    grad_clip_norm: float = 1.0
    log_every_steps: int = 1_000
    horizon_min: int = HORIZON_MIN
    horizon_max: int = HORIZON_MAX
    checkpoint_every_steps: int = 5_000
    target_examples: int = 500_000

    def __post_init__(self) -> None:
        """Reject an optimisation setting that cannot produce a valid schedule.

        Checked at construction for ModelConfig.__post_init__'s stated reason: a
        config that cannot build must fail where it is written, not several
        stages later inside a jitted step where the message names a shape.

        Raises:
            ValueError: If the optimiser is unknown, if either learning rate is
                non-positive, if the schedule does not decay, or if the
                gradient-clipping norm is non-positive.
        """
        if self.optimiser_name not in OPTIMISERS:
            valid = ", ".join(OPTIMISERS)
            raise ValueError(
                f"unknown optimiser {self.optimiser_name!r}. Valid: {valid}."
            )
        if self.learning_rate <= 0.0 or self.lr_schedule_final <= 0.0:
            raise ValueError(
                "both learning rates must be positive: "
                f"initial {self.learning_rate}, final {self.lr_schedule_final}. "
                "A cosine schedule's floor is expressed as a fraction of its "
                "peak, so a zero or negative peak has no schedule at all."
            )
        if self.lr_schedule_final > self.learning_rate:
            raise ValueError(
                f"lr_schedule_final {self.lr_schedule_final} exceeds "
                f"learning_rate {self.learning_rate}: cosine ANNEALING decays "
                "from the peak, so a higher floor is a sign the two were "
                "transposed."
            )
        if self.grad_clip_norm <= 0.0:
            raise ValueError(
                f"grad_clip_norm must be positive, got {self.grad_clip_norm}. "
                "Clipping off is expressed by removing the transform, not by "
                "passing a bound no gradient can satisfy."
            )
        if self.checkpoint_every_steps < 0:
            raise ValueError(
                "checkpoint_every_steps must not be negative, got "
                f"{self.checkpoint_every_steps}. Periodic saving is turned OFF "
                "with zero, which still writes the final checkpoint; a "
                "negative interval has no meaning and would silently never "
                "fire."
            )


# Environment families, for settings whose right value depends on the size of an
# observation rather than on the individual environment. NAVIX is 361 cells over
# vocabularies of (11, 6, 4); NLE is 1,659 cells over (128, 24). A single
# constant covering both cannot be right for either.
ENV_FAMILY_NAVIX: str = "navix"
ENV_FAMILY_NLE: str = "nle"
ENV_FAMILY_ATARI: str = "atari"
# Prefixes rather than an entry per environment: the five NAVIX environments
# share every size-driven setting, so listing them individually would be five
# chances to forget one.
ENV_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("Navix-", ENV_FAMILY_NAVIX),
    ("NetHack", ENV_FAMILY_NLE),
    ("MiniHack", ENV_FAMILY_NLE),
    ("nle-", ENV_FAMILY_NLE),
    ("atari-", ENV_FAMILY_ATARI),
)

# Which offline corpus each family converts, read by OfflineGenerateStage. A
# family absent here has no offline corpus; offline_source_for_env raises
# rather than falling back. Declared with the family constants rather than
# beside OFFLINE_SOURCES, because those are defined further up this module and
# a table there could not name them.
OFFLINE_SOURCE_BY_FAMILY: dict[str, str] = {
    ENV_FAMILY_NLE: OFFLINE_SOURCE_KATAKOMBA,
    ENV_FAMILY_ATARI: OFFLINE_SOURCE_ATARI,
}

# Environments whose corpus is not the one their family converts, read by
# offline_source_for_env before the family table. Two environments of one
# family read the same archive under different window settings, and the family
# name cannot distinguish them.
OFFLINE_SOURCE_BY_ENV: dict[str, str] = {
    ATARI_LONG_HORIZON_ENV_NAME: OFFLINE_SOURCE_ATARI_LONG,
    ATARI_LADDER_D_ENV_NAME: OFFLINE_SOURCE_ATARI_P24,
    ATARI_LADDER_E_ENV_NAME: OFFLINE_SOURCE_ATARI_P49,
}

# The reported conditions and the run each arm is read from, read by the figure
# module. A run name is a free command-line argument, so a tree generated
# elsewhere holds differently named directories; a reader repoints the figures
# by editing this table rather than the figure code.
@dataclass(frozen=True)
class ReportedRun:
    """One reported condition and the runs its arms are read from."""

    room: str
    policy: str
    slip: str
    env: str
    representation: str
    arm_runs: tuple[tuple[int, str], ...]

    @property
    def arms(self) -> tuple[int, ...]:
        """The arms the condition carries."""
        return tuple(arm for arm, _ in self.arm_runs)

    def run_for(self, arm: int) -> str:
        """The run an arm is read from, without the split suffix."""
        return dict(self.arm_runs)[arm]


def _every_arm(run: str, arms: tuple[int, ...] = ARMS) -> tuple[tuple[int, str], ...]:
    """Map every arm to one run."""
    return tuple((arm, run) for arm in arms)


# The arms a pixel condition carries, and the slip label of a deterministic one.
PIXEL_ARMS: tuple[int, ...] = (ARM_DIRECT, ARM_AR_ONE_STEP)
NO_SLIP: str = "no slip"

REPORTED_RUNS: dict[str, ReportedRun] = {
    "FOURROOMS_NO_SLIP": ReportedRun(
        "FourRooms", "uniform", NO_SLIP, DEFAULT_ENV_NAME, REPRESENTATION_SYMBOLIC,
        ((ARM_DIRECT, "e1_50k"), (ARM_AR_ENDPOINT, "e2_base"), (ARM_AR_ONE_STEP, "e2_base")),
    ),
    "FOURROOMS_SLIP_010": ReportedRun(
        "FourRooms", "uniform", "slip 0.10", DEFAULT_ENV_NAME, REPRESENTATION_SYMBOLIC,
        _every_arm("e3_p010"),
    ),
    "FOURROOMS_SLIP_025": ReportedRun(
        "FourRooms", "uniform", "slip 0.25", DEFAULT_ENV_NAME, REPRESENTATION_SYMBOLIC,
        _every_arm("e3_p025"),
    ),
    "DYNAMIC_OBSTACLES_UNIFORM": ReportedRun(
        "Dynamic-Obstacles", "uniform", NO_SLIP, DYNAMIC_OBSTACLES_ENV_NAME,
        REPRESENTATION_SYMBOLIC, _every_arm("e3_dynobs"),
    ),
    "DOORKEY_UNIFORM": ReportedRun(
        "DoorKey", "uniform", NO_SLIP, DOORKEY_ENV_NAME, REPRESENTATION_SYMBOLIC,
        _every_arm("e6_doorkey_uniform"),
    ),
    "FOURROOMS_PPO": ReportedRun(
        "FourRooms", "PPO", NO_SLIP, DEFAULT_ENV_NAME, REPRESENTATION_SYMBOLIC,
        _every_arm("e5_fr_ppo"),
    ),
    "DOORKEY_PPO": ReportedRun(
        "DoorKey", "PPO", NO_SLIP, DOORKEY_ENV_NAME, REPRESENTATION_SYMBOLIC,
        _every_arm("e10b_dk_ppo_3arm"),
    ),
    "DYNAMIC_OBSTACLES_PPO": ReportedRun(
        "Dynamic-Obstacles", "PPO", NO_SLIP, DYNAMIC_OBSTACLES_ENV_NAME,
        REPRESENTATION_SYMBOLIC, _every_arm("e11b_dynobs_ppo_3arm"),
    ),
    "PIXEL_FOURROOMS": ReportedRun(
        "FourRooms", "PPO", NO_SLIP, DEFAULT_ENV_NAME, REPRESENTATION_RGB,
        _every_arm("e7_fr_rgb_ppo", PIXEL_ARMS),
    ),
    "PIXEL_DOORKEY": ReportedRun(
        "DoorKey", "PPO", NO_SLIP, DOORKEY_ENV_NAME, REPRESENTATION_RGB,
        _every_arm("e8_dk_rgb_ppo", PIXEL_ARMS),
    ),
    "PIXEL_DYNAMIC_OBSTACLES": ReportedRun(
        "Dynamic-Obstacles", "PPO", NO_SLIP, DYNAMIC_OBSTACLES_ENV_NAME, REPRESENTATION_RGB,
        _every_arm("e9_dynobs_rgb_ppo", PIXEL_ARMS),
    ),
    "ATARI_POSITION_0": ReportedRun(
        "Atari", "DQN Replay", "position 0", ATARI_ENV_NAME, REPRESENTATION_GREYSCALE,
        _every_arm("a_base_p0", PIXEL_ARMS),
    ),
    "ATARI_POSITION_24": ReportedRun(
        "Atari", "DQN Replay", "position 24", ATARI_LADDER_D_ENV_NAME, REPRESENTATION_GREYSCALE,
        _every_arm("a_ladder_p24", PIXEL_ARMS),
    ),
    "ATARI_POSITION_49": ReportedRun(
        "Atari", "DQN Replay", "position 49", ATARI_LADDER_E_ENV_NAME, REPRESENTATION_GREYSCALE,
        _every_arm("a_ladder_p49", PIXEL_ARMS),
    ),
    "LONG_HORIZON": ReportedRun(
        ATARI_LONG_HORIZON_GAME, "DQN Replay", f"trained to h = {ATARI_LONG_HORIZON}",
        ATARI_LONG_HORIZON_ENV_NAME, REPRESENTATION_GREYSCALE,
        _every_arm("b44_robotank_h1024", PIXEL_ARMS),
    ),
}

# The run each trained horizon of the ablation is read from. A rung carries a
# trained width where a condition carries arms, so it is its own table. Its last
# rung is the run LONG_HORIZON also reads.
ABLATION_RUNS: tuple[tuple[int, str], ...] = (
    (128, "b44_ablation_h128"),
    (256, "b44_ablation_h256"),
    (512, "b44_ablation_h512"),
    (ATARI_LONG_HORIZON, "b44_robotank_h1024"),
)

# Environments that fix the trained horizon, read by
# training_horizon_max_for_env. An environment absent here takes HORIZON_MAX.
TRAINING_HORIZON_MAX_BY_ENV: dict[str, int] = {
    ATARI_LONG_HORIZON_ENV_NAME: ATARI_LONG_HORIZON,
}

# Encoder trunk widths per family. Unused: resolve_family_defaults derives the
# widths from the depth through encoder_widths_for_depth.
ENCODER_CHANNELS_BY_FAMILY: dict[str, tuple[int, ...]] = {
    ENV_FAMILY_NAVIX: ENCODER_CHANNELS_NAVIX,
    ENV_FAMILY_NLE: NLE_ENCODER_CHANNELS,
}

# Examples per horizon bin written to the per-cell probability log.
#
# NAVIX IS None -- LOG THE WHOLE FROZEN SET. The frozen validation evaluation
# set holds AT MOST 200 windows per horizon, because
# EVALUATION_WINDOWS_PER_TRAJECTORY is 1, DataConfig.num_trajectories is 2,000
# and VALIDATION_FRACTION is 0.1. Any cap above that would read as a deliberate
# limit while being inert.
#
# NLE IS CAPPED AT 20 rather than left uncapped. One NLE example is 1,659 cells
# over 152 classes, so 7 bins x 20 examples is roughly 135 MB per run at
# float32. Logging the whole frozen set would write tens of GB.
#
# A family with no entry still raises in `probability_log_cap` rather than
# defaulting, because both available defaults are wrong: None logs the whole
# population and a borrowed number silently truncates.
#
# ATARI IS CAPPED AT 20 for NLE's reason and more so: one frame is 7,056 cells
# against NLE's 1,659. The entry is required whatever the representation, and
# it is inert while that representation is continuous, because a continuous
# field carries no per-cell class distribution to log.
EVAL_PROB_LOG_N: dict[str, int | None] = {
    ENV_FAMILY_NAVIX: None,
    ENV_FAMILY_NLE: 20,
    ENV_FAMILY_ATARI: 20,
}

# Seed the probability log's subsample is drawn with.
#
# FIXED, AND NEVER THE RUN SEED. Drawing the subsample with the run seed makes
# each of the five reporting seeds log a DIFFERENT set of examples, so their
# distributions cannot be compared across seeds -- which is the entire purpose
# of logging them.
#
# 1000 IS DELIBERATE AND IS NOT A TYPO FOR 999. SEEDS holds 999 and
# TUNING_SEEDS holds (7, 13, 21); 1000 is disjoint from both.
EVAL_PROB_LOG_SEED: int = 1000

# Storage dtype for the logged probabilities, one entry per environment family.
#
# NAVIX IS float16. It carries about three decimal digits, ample for calibration
# on 11, 6 and 4 classes: a uniform distribution over the widest NAVIX channel
# is 0.09 and the subnormal floor is 6.1e-5.
#
# NLE IS float32. Uniform over its 128-class channel is 7.8e-3, but the tail of
# a 128-class distribution sits far below float16's subnormal floor, and a
# probability that stores as zero is indistinguishable from one the model never
# assigned. The cap keeps the cost bounded, so the wider dtype is affordable.
#
# ATARI IS float32. Inert while the representation is continuous, and the value
# is the one a 256-intensity vocabulary would need if a discrete Atari
# representation is ever added: wider than NLE's 128-class channel, whose tail
# already falls below float16's subnormal floor.
EVAL_PROB_LOG_DTYPE: dict[str, str] = {
    ENV_FAMILY_NAVIX: "float16",
    ENV_FAMILY_NLE: "float32",
    ENV_FAMILY_ATARI: "float32",
}

# The evaluation forward pass runs in blocks of this many windows within each
# horizon. Sized from measured cost per window rather than from the logits
# arithmetic, which understates the peak on some environments and overstates it
# on others.
EVALUATION_BATCH_CHUNK: int = 1024

# Which stored parameter tree the evaluation stage scores.
#
# The checkpoint carries both. `best_params` is the lowest held-out loss
# observed, `params` is the last step's. DR103 reports the best tree and keeps
# the final one reachable for the appendix comparison.
PARAMS_SELECTION_BEST: str = "best"
PARAMS_SELECTION_FINAL: str = "final"
PARAMS_SELECTIONS: tuple[str, ...] = (
    PARAMS_SELECTION_BEST,
    PARAMS_SELECTION_FINAL,
)


def env_family(env_name: str) -> str:
    """Return the environment family a registered environment name belongs to.

    Args:
        env_name: Registered environment name, e.g. "Navix-FourRooms-v0".

    Returns:
        One of the ENV_FAMILY_* constants.

    Raises:
        ValueError: If no prefix matches, naming the prefixes that are known.
            Loud rather than defaulting, because every caller uses the family to
            size something and a wrong size is a silently wrong artefact.
    """
    for prefix, family in ENV_FAMILY_PREFIXES:
        if env_name.startswith(prefix):
            return family
    known = ", ".join(prefix for prefix, _ in ENV_FAMILY_PREFIXES)
    raise ValueError(
        f"environment '{env_name}' matches no known family prefix ({known}). "
        "Add it to ENV_FAMILY_PREFIXES rather than letting a size-driven "
        "setting take another family's value."
    )


def dataset_stage_names(env_name: str) -> tuple[str, ...]:
    """Return the dataset stage names for one environment, in execution order.

    The one routing rule the runner and the manifest both read, so the two
    cannot describe different pipelines for the same environment. Returns names
    rather than classes, so this module imports no stage and each caller maps
    the names onto its own classes.

    The source stage follows the family. Whether the displacement diagnostic
    follows is declared per environment rather than per family, because one
    family holds environments whose saturation is settled and environments
    whose saturation is not.

    Args:
        env_name: Registered environment name.

    Returns:
        The dataset stage names, source stage first.

    Raises:
        ValueError: If the environment's family declares no source stage, or if
            the environment has no row in ENVIRONMENT_GEOMETRIES and so has not
            declared whether it runs the displacement diagnostic.
    """
    family = env_family(env_name)
    if family == ENV_FAMILY_NAVIX:
        names = [STAGE_NAME_GENERATE]
    elif family in (ENV_FAMILY_NLE, ENV_FAMILY_ATARI):
        names = [STAGE_NAME_OFFLINE_GENERATE]
    else:
        raise ValueError(
            f"environment family {family!r} declares no source stage. "
            f"Families with one: {ENV_FAMILY_NAVIX}, {ENV_FAMILY_NLE}, "
            f"{ENV_FAMILY_ATARI}."
        )
    names.append(STAGE_NAME_PREPARE)
    geometry = ENVIRONMENT_GEOMETRIES.get(env_name)
    if geometry is None:
        raise ValueError(
            f"no geometry for environment {env_name!r}, so whether it runs the "
            "displacement diagnostic is undeclared. Add a row to "
            f"ENVIRONMENT_GEOMETRIES. Tabled: {sorted(ENVIRONMENT_GEOMETRIES)}"
        )
    if geometry.runs_displacement:
        names.append(STAGE_NAME_DISPLACEMENT)
    return tuple(names)


def offline_source_for_env(env_name: str) -> str:
    """Return the offline corpus one environment converts.

    An environment named in OFFLINE_SOURCE_BY_ENV takes that corpus; every
    other takes its family's.

    Args:
        env_name: Registered environment name.

    Returns:
        One of OFFLINE_SOURCES.

    Raises:
        ValueError: If the environment's family has no row in
            OFFLINE_SOURCE_BY_FAMILY. A family that generates its own data
            reaches STAGE_NAME_GENERATE instead, so the message names that
            case rather than implying the corpus is missing from disk.
    """
    named = OFFLINE_SOURCE_BY_ENV.get(env_name)
    if named is not None:
        return named
    family = env_family(env_name)
    source = OFFLINE_SOURCE_BY_FAMILY.get(family)
    if source is None:
        raise ValueError(
            f"environment family {family!r} registers no offline corpus, so "
            f"{env_name!r} cannot be converted. A family that generates its "
            f"own data runs {STAGE_NAME_GENERATE!r} and never reaches "
            f"{STAGE_NAME_OFFLINE_GENERATE!r}. Families with a corpus: "
            f"{sorted(OFFLINE_SOURCE_BY_FAMILY)}"
        )
    return source


def training_horizon_max_for_env(env_name: str) -> int:
    """Return the largest horizon an environment's runs train on.

    Args:
        env_name: Registered environment name.

    Returns:
        The environment's entry in TRAINING_HORIZON_MAX_BY_ENV, or HORIZON_MAX.
    """
    return TRAINING_HORIZON_MAX_BY_ENV.get(env_name, HORIZON_MAX)


def displacement_skip_note(env_name: str) -> str:
    """Return the reason an environment's row gives for skipping the diagnostic.

    Args:
        env_name: Registered environment name, tabled in
            ENVIRONMENT_GEOMETRIES.

    Returns:
        The row's displacement_note.
    """
    return ENVIRONMENT_GEOMETRIES[env_name].displacement_note


def katakomba_root() -> Path:
    """Return the directory holding the Katakomba HDF5 corpus.

    Reads KATAKOMBA_ROOT_ENV_VAR, falling back to KATAKOMBA_ROOT_DEFAULT.
    Resolved at call time so importing this module never touches an external
    volume.

    Returns:
        The corpus directory, which is not checked for existence here.
    """
    override = os.environ.get(KATAKOMBA_ROOT_ENV_VAR)
    if override:
        return Path(override)
    return KATAKOMBA_ROOT_DEFAULT


def atari_root() -> Path:
    """Return the directory holding the ordered Atari RLDS shards.

    Reads ATARI_ROOT_ENV_VAR, falling back to ATARI_ROOT_DEFAULT. Resolved at
    call time so importing this module never touches an external volume.

    Returns:
        The archive directory, which is not checked for existence here.
    """
    override = os.environ.get(ATARI_ROOT_ENV_VAR)
    if override:
        return Path(override)
    return ATARI_ROOT_DEFAULT


def atari_games(names: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return the games a conversion reads, checked against the measured pool.

    Args:
        names: Games to convert. None takes ATARI_GAMES.

    Returns:
        The games, in the order given.

    Raises:
        ValueError: If the selection is empty, holds a duplicate, or names a
            game outside ATARI_GAME_POOL. A game outside the pool failed the
            displacement bar or the episode-length gate, and converting one
            silently would put an unreadable cell in the run matrix.
    """
    selected = tuple(ATARI_GAMES if names is None else names)
    if not selected:
        raise ValueError(
            "no Atari games selected. Name at least one of "
            f"{ATARI_GAME_POOL} in ATARI_GAMES."
        )
    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"Atari games {duplicates} are named more than once, which would "
            "convert the same episodes twice and weight them double in the "
            "split"
        )
    unknown = sorted(set(selected) - set(ATARI_GAME_POOL))
    if unknown:
        raise ValueError(
            f"Atari games {unknown} are not in ATARI_GAME_POOL. The pool holds "
            "the games measured to clear both "
            f"{ATARI_DISPLACEMENT_BAR_PCT} per cent displacement at h = "
            f"{HORIZON_MAX} and {ATARI_EPISODE_LENGTH_GATE} decision steps at "
            f"percentile {ATARI_LENGTH_PERCENTILE}: {ATARI_GAME_POOL}"
        )
    return selected


def atari_shard_path(
    game_dir: Path, position: int, run: int = ATARI_RUN_INDEX
) -> Path | None:
    """Return a game's shard at one archive position, or None if it is absent.

    Args:
        game_dir: The game's directory in the archive.
        position: Archive position within the run.
        run: Which training run to read.

    Returns:
        The shard, or None.

    Raises:
        ValueError: If more than one shard matches the position.
    """
    pattern = ATARI_SHARD_GLOB.format(run=run, index=position)
    matches = sorted(path for path in game_dir.glob(pattern) if path.is_file())
    if len(matches) > 1:
        raise ValueError(
            f"{game_dir.name} holds {len(matches)} shards at position "
            f"{position}: {[path.name for path in matches]}. One is expected."
        )
    return matches[0] if matches else None


def atari_shard_paths(
    root: Path,
    game: str,
    positions: Sequence[int] = ATARI_CHECKPOINT_POSITIONS,
    run: int = ATARI_RUN_INDEX,
) -> tuple[Path, ...]:
    """Return one game's selected shard paths, in position order.

    Args:
        root: The archive directory.
        game: Game name, which is also its subdirectory.
        positions: Shard positions within the run.
        run: Which training run to read.

    Returns:
        One path per position.

    Raises:
        FileNotFoundError: If any of them is absent, naming every missing
            position's pattern. A game named in config but not on disk fails
            here rather than part way through a conversion.
        ValueError: If a position matches more than one shard.
    """
    resolved = {
        position: atari_shard_path(root / game, position, run)
        for position in positions
    }
    missing = [position for position, path in resolved.items() if path is None]
    if missing:
        listed = ", ".join(
            ATARI_SHARD_GLOB.format(run=run, index=position) for position in missing
        )
        raise FileNotFoundError(
            f"game '{game}' is selected but {len(missing)} of its shards are "
            f"absent under {root / game}: {listed}. Fetch them, or drop the "
            "game from ATARI_GAMES."
        )
    return tuple(resolved[position] for position in positions)


def window_offset_mode(mode: str) -> str:
    """Return a validated window-offset mode.

    Args:
        mode: One of KATAKOMBA_WINDOW_OFFSET_MODES.

    Returns:
        The mode, unchanged.

    Raises:
        ValueError: If the mode is not declared, naming the ones that are.
    """
    if mode not in KATAKOMBA_WINDOW_OFFSET_MODES:
        known = ", ".join(KATAKOMBA_WINDOW_OFFSET_MODES)
        raise ValueError(
            f"unknown window offset mode '{mode}'. Declared modes are "
            f"({known}); add it to KATAKOMBA_WINDOW_OFFSET_MODES rather than "
            "falling back to a default, which would silently change where "
            "every window is cut from."
        )
    return mode


def probability_log_cap(env_name: str) -> int | None:
    """Return the probability log's per-horizon example cap for one environment.

    Args:
        env_name: Registered environment name.

    Returns:
        The cap, or None to log every example the evaluation set holds.

    Raises:
        ValueError: If this environment's family has no entry. See
            EVAL_PROB_LOG_N for why an absent family raises rather than
            defaulting.
    """
    family = env_family(env_name)
    if family not in EVAL_PROB_LOG_N:
        raise ValueError(
            f"no probability-log cap is declared for environment family "
            f"'{family}'. Derive it against that family's real observation "
            f"size and add it to EVAL_PROB_LOG_N; do not borrow another "
            f"family's value."
        )
    return EVAL_PROB_LOG_N[family]


def probability_log_dtype(env_name: str) -> str:
    """Return the probability log's storage dtype for one environment.

    Args:
        env_name: Registered environment name.

    Returns:
        The dtype name.

    Raises:
        ValueError: If this environment's family has no entry. Absent raises
            for probability_log_cap's reason: a borrowed dtype either wastes
            space or stores small probabilities as zero.
    """
    family = env_family(env_name)
    if family not in EVAL_PROB_LOG_DTYPE:
        raise ValueError(
            f"no probability-log dtype is declared for environment family "
            f"'{family}'. Check the smallest probability that family's "
            f"vocabulary produces against the dtype's subnormal floor and add "
            f"it to EVAL_PROB_LOG_DTYPE; do not borrow another family's value."
        )
    return EVAL_PROB_LOG_DTYPE[family]


def receptive_field(dilations: tuple[int, ...]) -> int:
    """Return the receptive field of a stack of 3x3 stride-1 convolutions.

    Each layer adds ``2 * dilation`` to the span. Pure arithmetic, kept here
    rather than beside the layers it sizes so that `config` stays importable
    without the model stack (`DR249`) -- `src/models/layers.py` re-exports it.

    Args:
        dilations: Dilation rate per layer.

    Returns:
        The span in cells.
    """
    return 1 + sum(2 * dilation for dilation in dilations)


def dilations_for_grid(grid_extent: int) -> tuple[int, ...]:
    """Return the dilation schedule a grid of this size needs, and no more.

    Doubling dilations until the receptive field spans the grid gives the
    shallowest stack that can see the whole layout. A dilation wider than the
    grid samples only zero padding.

    Pass the largest grid the environment produces, not the current mode's, so
    every observation mode shares one encoder shape.

    Args:
        grid_extent: The larger of the grid's height and width, for the largest
            observation mode the environment emits.

    Returns:
        Doubling dilations, shortest schedule whose receptive field reaches
        `grid_extent`.

    Raises:
        ValueError: If grid_extent is not positive.
    """
    if grid_extent < 1:
        raise ValueError(f"grid_extent must be positive, got {grid_extent}")
    dilations: list[int] = []
    rate = 1
    while receptive_field(tuple(dilations)) < grid_extent:
        dilations.append(rate)
        rate *= 2
    return tuple(dilations)


def encoder_widths_for_depth(depth: int) -> tuple[int, ...]:
    """Return one convolution width per encoder layer, for any depth.

    The schedule is ENCODER_NARROW_LAYERS layers at ENCODER_NARROW_WIDTH,
    then ENCODER_WIDE_WIDTH for the rest. It reproduces both width tuples this
    project already committed to, so nothing that has run changes.

    Args:
        depth: Number of dilated layers the grid extent derives.

    Returns:
        One width per layer, narrow first.

    Raises:
        ValueError: If depth is not positive.
    """
    if depth <= 0:
        raise ValueError(f"encoder depth must be positive, got {depth}")
    narrow = min(ENCODER_NARROW_LAYERS, depth)
    return (ENCODER_NARROW_WIDTH,) * narrow + (ENCODER_WIDE_WIDTH,) * (depth - narrow)


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation settings.

    Attributes:
        num_eval_envs: Environment rows reserved for held-out evaluation.
        probability_log_seed: Seed the probability log's subsample is drawn
            with. A FIXED constant rather than the run seed, so every reporting
            seed logs the same examples and their distributions are comparable.
        probability_log_dtype: Storage dtype for the logged probabilities. The
            NAVIX entry is the default; resolve_family_defaults sets it from
            the configured environment's family.
        params_selection: Which stored parameter tree is scored, one of
            PARAMS_SELECTIONS. The checkpoint carries both.
    """

    num_eval_envs: int = 8
    probability_log_seed: int = EVAL_PROB_LOG_SEED
    probability_log_dtype: str = EVAL_PROB_LOG_DTYPE[ENV_FAMILY_NAVIX]
    params_selection: str = PARAMS_SELECTION_BEST

    def __post_init__(self) -> None:
        """Validate the parameter selection.

        Raises:
            ValueError: If params_selection is not one of PARAMS_SELECTIONS.
        """
        if self.params_selection not in PARAMS_SELECTIONS:
            raise ValueError(
                f"unknown params_selection {self.params_selection!r}, expected "
                f"one of {PARAMS_SELECTIONS}"
            )

@dataclass(frozen=True)
class DataConfig:  # pylint: disable=too-many-instance-attributes
    """Trajectory generation settings, consumed by GenerateTrajectoriesStage.

    Attributes:
        num_trajectories: Trajectories to generate, PER DATASET. The run
            generates one dataset per reporting seed, so this is 10,000
            trajectories in total.

            SIZED ON MEASUREMENT RATHER THAN ON THE MINIMUM. The
            at-least-100-held-out bar needs 1,000 per dataset under the 80/10/10
            split. Two thousand doubles that floor, bought cheaply: 95.7 per
            cent of build trajectories support h = 100.

            INCREASE TRIGGER, PRE-REGISTERED: regenerate at 4,000 if the
            per-split support fraction falls below 80 per cent in any dataset.

            NOT DERIVED FROM target_examples. The bootstrap resamples
            trajectories, so sizing is on held-out trajectory count: overlapping
            windows from one episode are not independent observations.
        representation: How observations are rendered, symbolic or rgb.
        stored_observation_modes: Which modes generation writes to disk. None
            takes the representation's default from
            DEFAULT_STORED_OBSERVATION_MODES.
        shard_size: Trajectories per file on disk.
        decode_samples: Trajectories rendered to text for hand inspection.
        storage_format: Serialisation backend.
        train_fraction: Share of trajectories used for training.
        validation_fraction: Share used by the E1 diagnosis sweep and by GATE C.
            Deliberately separate from test so a repeated gate check never reads
            the split the headline is reported on.
        test_fraction: Share held out for reported results, opened ONCE with the
            configuration already frozen. The at-least-100-held-out bar binds
            HERE rather than on validation, because only this split produces a
            reported interval.

            SPLITTING IS BY TRAJECTORY, NEVER BY SAMPLE. Two windows from one
            episode overlap, so a sample-level split puts near-copies on both
            sides and leaks worst at the long horizons the whole claim rests on.
    """

    num_trajectories: int = 2000
    shard_size: int = 64
    decode_samples: int = 3
    storage_format: str = STORAGE_FORMAT_HDF5
    collection_policy: str = POLICY_UNIFORM_RANDOM
    representation: str = REPRESENTATION_SYMBOLIC
    stored_observation_modes: tuple[str, ...] | None = None
    ppo_budget_frames: int = PPO_BUDGET_FRAMES
    train_fraction: float = TRAIN_FRACTION
    validation_fraction: float = VALIDATION_FRACTION
    test_fraction: float = TEST_FRACTION

    def modes_stored(self) -> tuple[str, ...]:
        """Return the observation modes this run writes to disk.

        The one source of truth for the stored set. Generation, the cardinality
        guard, the channel census, the manifest and the sentinel identity all
        read it, so the identity records what was stored rather than what the
        code could store.

        Returns:
            The configured modes, or the representation's default.
        """
        if self.stored_observation_modes is not None:
            return self.stored_observation_modes
        return DEFAULT_STORED_OBSERVATION_MODES[self.representation]

    def __post_init__(self) -> None:
        """Validate the fields that silently produce a useless dataset.

        Raises:
            ValueError: If any count is non-positive, if decode_samples asks
                for more trajectories than are generated, or if the storage
                format is not one this codebase can write.
        """
        for name in ("num_trajectories", "shard_size", "decode_samples"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.decode_samples > self.num_trajectories:
            raise ValueError(
                f"decode_samples ({self.decode_samples}) exceeds "
                f"num_trajectories ({self.num_trajectories}): the hand-decode "
                "step cannot inspect more trajectories than exist"
            )
        if self.collection_policy not in COLLECTION_POLICIES:
            raise ValueError(
                f"unknown collection_policy {self.collection_policy!r}, "
                f"expected one of {COLLECTION_POLICIES}"
            )
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"unknown representation {self.representation!r}, "
                f"expected one of {REPRESENTATIONS}"
            )
        if self.stored_observation_modes is not None:
            modes = self.stored_observation_modes
            unknown = [mode for mode in modes if mode not in OBS_MODES]
            if unknown or not modes or len(set(modes)) != len(modes):
                raise ValueError(
                    f"stored_observation_modes {modes!r} must be a non-empty "
                    f"tuple of distinct values from {OBS_MODES}"
                )
        if self.ppo_budget_frames <= 0:
            raise ValueError(
                f"ppo_budget_frames must be positive, got {self.ppo_budget_frames}"
            )
        if self.storage_format not in STORAGE_FORMATS:
            raise ValueError(
                f"unknown storage_format {self.storage_format!r}, "
                f"expected one of {STORAGE_FORMATS}"
            )
        fractions = {
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
        }
        for name, value in fractions.items():
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"{name} must lie strictly between 0 and 1, got {value}. "
                    "A zero share is not a two-way split, it is a missing "
                    "split, and DR61 requires all three."
                )
        total = sum(fractions.values())
        if abs(total - 1.0) > SPLIT_FRACTION_SUM_TOLERANCE:
            raise ValueError(
                f"split fractions must sum to 1.0, got {total}. "
                f"Values: {fractions}"
            )


class SamplerMode(Enum):
    """Where a window sampler gets its training and evaluation examples.

    An Enum defined HERE rather than in src.data: this is a configuration choice
    with no meaning outside config, so it introduces no import from the packages
    config is imported by.

    HYBRID IS THE DEFAULT AND THE ONLY REPORTING PATH. Every number quoted in
    the dissertation comes from it. The other two are verification instruments,
    and Methods says so in one sentence.

    IF TWO MODES EVER DISAGREE ON A REPORTED NUMBER, THE DISSERTATION NAMES
    HYBRID. The other two exist to be compared against it, and a disagreement is
    a bug report rather than a result.
    """

    HYBRID = "hybrid"
    ONLINE = "online"
    OFFLINE = "offline"


# EVALUATION_HORIZONS is declared beside EXTRAPOLATION_HORIZONS, above the
# observation contracts that carry it per environment.
#
# DELIBERATELY NOT BOUNDED BY TrainConfig.horizon_max. The test pass runs a
# model trained at h_max = 100 out to h = 200 on the existing dataset, with no
# regeneration and no retraining. An evaluation sampler capped at the training
# ceiling cannot run that experiment at all, so SamplerConfig validates the
# grid against HORIZON_MIN only.

# Windows the frozen evaluation set draws from each SUPPORTING trajectory at
# each horizon.
#
# *** [CHOSEN], AND IT IS FLAGGED RATHER THAN SETTLED. It owes a decision
# register row. ***
#
# One is the bootstrap-honest value. The bootstrap resamples TRAJECTORIES, so
# the trajectory is the independent unit behind every interval; drawing one
# window per trajectory per horizon makes each unit contribute exactly once and
# equally. Drawing k > 1 adds within-trajectory windows the bootstrap does not
# treat as independent, which narrows an interval without adding evidence.
#
# WHAT WOULD REVISE IT: a measured per-horizon cross-entropy whose variance is
# dominated by window noise rather than by trajectory noise. The fix would then
# be more TRAJECTORIES rather than more windows per trajectory.
EVALUATION_WINDOWS_PER_TRAJECTORY: int = 1

# Ceiling on what OFFLINE mode may materialise, in bytes. It admits the target
# window count and refuses the full pool, so the limit arrives as a refusal
# rather than as a full disk.
OFFLINE_BYTE_BUDGET: int = 2 * 1024**3


@dataclass(frozen=True)
class SamplerConfig:
    """Window-sampling settings.

    Attributes:
        mode: Which sampling strategy. HYBRID is the default and every reported
            number comes from it; ONLINE and OFFLINE are verification
            instruments, never a reporting path.
        observation_mode: The single mode this sampler emits, as one of
            OBS_MODES. Mode is a RUN axis, not a batch axis. The default is
            top-down, the trivial control: a run that forgets to set the mode
            then produces the visibly degenerate number -- 99.7% copy accuracy
            on FourRooms -- rather than a plausible headline. It is a default
            and not a privileging; both modes are run and reported.
        evaluation_horizons: Horizons the frozen evaluation set covers. NOT
            bounded by TrainConfig.horizon_max.
        evaluation_windows_per_trajectory: Windows drawn from each supporting
            trajectory at each evaluation horizon. [CHOSEN] and owed a register
            row; see EVALUATION_WINDOWS_PER_TRAJECTORY.
        offline_byte_budget: Refuse to materialise beyond this many bytes in
            OFFLINE mode. A top-down window is 2,566 bytes MEASURED.
    """

    mode: SamplerMode = SamplerMode.HYBRID
    observation_mode: str = OBS_MODE_TOP_DOWN
    evaluation_horizons: tuple[int, ...] = EVALUATION_HORIZONS
    evaluation_windows_per_trajectory: int = EVALUATION_WINDOWS_PER_TRAJECTORY
    offline_byte_budget: int = OFFLINE_BYTE_BUDGET

    def __post_init__(self) -> None:
        """Validate the fields that silently produce an unusable sampler.

        Raises:
            ValueError: If the mode is not a SamplerMode, the observation mode
                is not one this codebase generates, the evaluation horizons are
                empty, non-ascending or below HORIZON_MIN, or the byte budget
                is not positive.
        """
        if not isinstance(self.mode, SamplerMode):
            raise ValueError(
                f"mode must be a SamplerMode, got {self.mode!r}. The CLI "
                "resolves the string; the config field holds the enum"
            )
        if self.observation_mode not in OBS_MODES:
            raise ValueError(
                f"unknown observation_mode {self.observation_mode!r}, expected "
                f"one of {OBS_MODES}"
            )
        validate_horizon_grid(self.evaluation_horizons, "evaluation_horizons")
        if min(self.evaluation_horizons) < HORIZON_MIN:
            raise ValueError(
                f"evaluation_horizons hold {min(self.evaluation_horizons)}, "
                f"below HORIZON_MIN {HORIZON_MIN}. h = 0 is not a prediction "
                "(D19)"
            )
        if self.evaluation_windows_per_trajectory < 1:
            raise ValueError(
                "evaluation_windows_per_trajectory must be at least 1, got "
                f"{self.evaluation_windows_per_trajectory}"
            )
        if self.offline_byte_budget <= 0:
            raise ValueError(
                f"offline_byte_budget must be positive, got "
                f"{self.offline_byte_budget}"
            )


@dataclass(frozen=True)
class ExperimentConfig:  # pylint: disable=too-many-instance-attributes
    """Top-level configuration composing all sub-configs.

    THE SEED IS THREE FIELDS RATHER THAN ONE, and two of them are normally left
    alone. `seed` names the run and the output directory. `data_seed` drives the
    dataset and the split; `model_seed` drives initialisation and batch order.
    Both default to `seed`, so a reporting run written as
    ExperimentConfig(seed=67) has all three equal and nothing to remember. The
    five-run attribution arm is the case that needs them apart: it pins
    data_seed = 42 and varies model_seed, isolating model variance from data
    variance.

    Attributes:
        seed: Active seed for this run (one of SEEDS). Names the output
            directory, so every artefact of a run sits under one seed level.
        fast: Whether this is a prototype run. Fast mode is a DIRECTORY level
            rather than a name prefix. A fast run is never reportable.
        run_name: Run identifier used for the output directory. Generic by
            default; pass --run-name explicitly for a real experiment.
        env: Environment configuration.
        model: Model architecture configuration.
        train: Training loop configuration.
        eval: Evaluation configuration.
        data: Trajectory generation configuration.
        sampler: Window-sampling configuration.
        data_seed: Seed for the dataset and the trajectory-level split. Passed
            as None to follow `seed`, and resolved to an int during
            construction, so it is never None on a constructed config.
        model_seed: Seed for parameter initialisation and batch order. Drives
            set_determinism. Passed as None to follow `seed`, resolved the same
            way.
    """

    seed: int = DEFAULT_SEED
    run_name: str = "run"
    fast: bool = False
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    data_seed: int | None = None
    model_seed: int | None = None

    def __post_init__(self) -> None:
        """Resolve the two derived seeds to `seed` when they were not given.

        Defaulting to `seed` rather than to DEFAULT_SEED is the safe direction:
        a run constructed as ExperimentConfig(seed=999) gets three 999s, where
        a literal default would have left the dataset silently on 42 while the
        run called itself 999.

        CAVEAT, PINNED BY tests/test_config.py: this resolves at CONSTRUCTION.
        `dataclasses.replace(config, seed=999)` on an already-constructed config
        carries the old resolved seeds forward, because they are ints by then
        and no longer None. Build a new ExperimentConfig to change the seed;
        apply_fast_mode does not touch it, which is why fast mode is unaffected.
        """
        if self.data_seed is None:
            object.__setattr__(self, "data_seed", self.seed)
        if self.model_seed is None:
            object.__setattr__(self, "model_seed", self.seed)


# --- Fast/prototyping mode ---------------------------------------------------
# Smoke-test sizes, consumed only by apply_fast_mode. Never for reporting.
FAST_NUM_ENVS: int = 8
FAST_MAX_EPISODE_STEPS: int = 32
FAST_TOTAL_STEPS: int = 20
FAST_BATCH_SIZE: int = 4
FAST_LOG_EVERY_STEPS: int = 5
# Fires TWICE inside FAST_TOTAL_STEPS = 20, so the smoke run exercises the
# periodic-checkpoint path AND the overwrite a second save performs. A value
# equal to total_steps would leave both untested while still looking configured.
FAST_CHECKPOINT_EVERY_STEPS: int = 10
FAST_NUM_EVAL_ENVS: int = 2
# Horizon and dataset size are QUANTITIES OF WORK, not architecture. 8 keeps the
# sampled range U(1, 8) non-degenerate while cutting arm 2's backpropagated
# rollout from 100 sequential passes to 8.
FAST_HORIZON_MAX: int = 8
# Architecture is scaled too, because --fast is a pipeline wiring check.
# ModelConfig.__post_init__ enforces the architecture constraints on every
# config. The dilation schedule is not scaled: it is derived from the
# environment's grid.
FAST_D_MODEL: int = 64
FAST_NUM_LAYERS: int = 2
FAST_NUM_HEADS: int = 2
FAST_CODE_EMBED_DIM: int = 8
# Channel widths are scaled by ratio rather than replaced by a literal tuple, so
# the tuple LENGTH follows whatever the real config carries. That keeps fast
# mode correct when NetHack derives six encoder layers instead of four.
FAST_CHANNEL_DIVISOR: int = 4
FAST_MIN_CHANNEL_WIDTH: int = 4
# 256 leaves headroom above FAST_TOTAL_STEPS * FAST_BATCH_SIZE = 80 and keeps a
# trajectory-level split non-degenerate.
FAST_TARGET_EXAMPLES: int = 256
# The pair is chosen so 50 / 10 = 5 shards are written: a single-shard smoke run
# would not exercise the multi-file read path.
#
# TWO SEPARATE FLOORS, BOTH MEASURED RATHER THAN REASONED:
#
#   1. All three splits must be non-empty. Under 0.8/0.1/0.1, 8 trajectories
#      allocate (6, 0, 2) and RAISE. Ten is the smallest count that allocates.
#   2. Every split's h_max-support fraction must sit within
#      SPLIT_SUPPORT_TOLERANCE of the pooled fraction, which a very small split
#      cannot satisfy: its own fraction quantises to 1.00 and no redraw passes.
#
# The check is sized for reporting-scale datasets. At smoke scale it measures
# quantisation rather than skew, and raising the count is what buys it
# resolution -- lowering the tolerance would weaken the real check to suit the
# fake one.
FAST_NUM_TRAJECTORIES: int = 50
FAST_SHARD_SIZE: int = 10
# Two PPO updates. One update is PPO_NUM_ENVS * PPO_NUM_STEPS frames, and fast
# mode does not scale either, so a smaller budget completes no update at all and
# leaves the collection path untested.
FAST_PPO_BUDGET_FRAMES: int = 16_384
# The frozen evaluation grid is scaled as a CORRECTNESS requirement rather than
# a speed one. Fast mode caps episodes at FAST_MAX_EPISODE_STEPS, so the
# reporting grid's h = 25, 50 and 100 have no supporting trajectory at all and
# the sampler would raise rather than run. The top value matches
# FAST_HORIZON_MAX so a smoke run also exercises the h == horizon_max edge.
FAST_EVALUATION_HORIZONS: tuple[int, ...] = (1, 2, 4, FAST_HORIZON_MAX)


SEED_RUN_SUFFIX: str = "_seed"

# Suffix marking the tree a test-split pass writes into. The target is DERIVED
# from the source run rather than typed, so a mismatched pair cannot be
# expressed: `--source-run e2_base` can only write `e2_base_test`.
#
# The tree also carries the extrapolation horizons and the parameter-tree gap,
# because one pass produces all three.
TEST_RUN_SUFFIX: str = "_test"

# Format for a run_name generated when the operator supplies none. Sortable,
# filesystem-safe, and unique to the second.
RUN_NAME_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
RUN_NAME_DEFAULT_PREFIX: str = "run_"


def seed_scoped_run_name(run_name: str, seed: int) -> str:
    """Return a run name carrying its seed, so seeds cannot share a directory.

    IDEMPOTENT by design. `--run-name e1_base_seed42 --seed 42` is a natural
    thing to type and must not become `e1_base_seed42_seed42`, so a name already
    carrying this exact seed is returned unchanged. A name carrying a DIFFERENT
    seed is still suffixed, because trusting it would reintroduce the collision
    this exists to prevent.

    Args:
        run_name: Operator-supplied or generated run identifier.
        seed: Active seed for this run.

    Returns:
        The run name with its seed suffix, applied at most once.
    """
    suffix = f"{SEED_RUN_SUFFIX}{seed}"
    if run_name.endswith(suffix):
        return run_name
    return run_name + suffix


def test_scoped_run_name(source_run: str) -> str:
    """Return the run name a test-split pass over one source run writes to.

    IDEMPOTENT, following seed_scoped_run_name: a name already carrying the
    suffix is returned unchanged, so re-deriving it cannot produce
    `e2_base_test_test`.

    Args:
        source_run: The run whose checkpoints and shards are read.

    Returns:
        The source run name with the test suffix, applied at most once.
    """
    if source_run.endswith(TEST_RUN_SUFFIX):
        return source_run
    return source_run + TEST_RUN_SUFFIX


def default_run_name(now: datetime | None = None) -> str:
    """Return a timestamped run name for when none is supplied.

    A literal default would let two unnamed runs share every artefact path.

    Args:
        now: Timestamp to format. Defaults to the current local time,
            injectable so the behaviour is testable.

    Returns:
        A run name of the form run_YYYYMMDD_HHMMSS.
    """
    stamp = (now or datetime.now()).strftime(RUN_NAME_TIMESTAMP_FORMAT)
    return f"{RUN_NAME_DEFAULT_PREFIX}{stamp}"


def scale_channels(channels: tuple[int, ...]) -> tuple[int, ...]:
    """Narrow a channel-width tuple for fast mode, preserving its length.

    Length is preserved because it must keep matching the DERIVED dilation
    depth, which follows from the environment's grid size and not from anything
    fast mode may change.

    Args:
        channels: Real-scale convolution widths, one per layer.

    Returns:
        The same number of widths, each divided down and floored at
        FAST_MIN_CHANNEL_WIDTH so no layer is scaled out of existence.
    """
    return tuple(
        max(FAST_MIN_CHANNEL_WIDTH, width // FAST_CHANNEL_DIVISOR)
        for width in channels
    )


def apply_fast_mode(config: ExperimentConfig) -> ExperimentConfig:
    """Return config with scale fields overridden to fast-mode sizes.

    Scales how much work is done, not what is computed. `fast=True` puts every
    artefact under `outputs/fast/`, so a prototype run cannot collide with a
    reporting one.

    Args:
        config: The experiment configuration to scale down.

    Returns:
        A new ExperimentConfig with env, train, eval, data, model and sampler
        scaled to the FAST_* constants and fast set True. Seeds and run_name
        unchanged.

    Note:
        `model` IS scaled, and the encoder's DEPTH is not. Depth stays real
        because encoder_for_spec derives it from the grid rather than reading it
        here, which is also why scale_channels preserves tuple length.
    """
    return replace(
        config,
        fast=True,
        env=replace(
            config.env,
            num_envs=FAST_NUM_ENVS,
            max_episode_steps=FAST_MAX_EPISODE_STEPS,
        ),
        train=replace(
            config.train,
            total_steps=FAST_TOTAL_STEPS,
            batch_size=FAST_BATCH_SIZE,
            log_every_steps=FAST_LOG_EVERY_STEPS,
            checkpoint_every_steps=FAST_CHECKPOINT_EVERY_STEPS,
            horizon_max=FAST_HORIZON_MAX,
            target_examples=FAST_TARGET_EXAMPLES,
        ),
        eval=replace(config.eval, num_eval_envs=FAST_NUM_EVAL_ENVS),
        data=replace(
            config.data,
            num_trajectories=FAST_NUM_TRAJECTORIES,
            shard_size=FAST_SHARD_SIZE,
            ppo_budget_frames=FAST_PPO_BUDGET_FRAMES,
        ),
        sampler=replace(
            config.sampler, evaluation_horizons=FAST_EVALUATION_HORIZONS
        ),
        model=replace(
            config.model,
            d_model=FAST_D_MODEL,
            num_layers=FAST_NUM_LAYERS,
            num_heads=FAST_NUM_HEADS,
            code_embed_dim=FAST_CODE_EMBED_DIM,
            encoder_channels=scale_channels(config.model.encoder_channels),
            decoder_channels=scale_channels(config.model.decoder_channels),
        ),
    )


def resolve_observation_contract(config: ExperimentConfig) -> ExperimentConfig:
    """Return config with its environment's geometry and representation applied.

    Composes two tables rather than reading one product row: the environment
    supplies the grid, the representation supplies how a cell becomes numbers,
    and obs_dim is computed from the pair rather than stored beside them.

    Call after any --env or --representation override, or it resolves the
    previous pair. POSITION RELATIVE TO apply_fast_mode IS FIXED: this must run
    FIRST, because it sets evaluation_horizons and fast mode then overrides that
    field with FAST_EVALUATION_HORIZONS. Reversed, every smoke run would score
    the full reporting grid, whose longer horizons no fast-mode episode
    supports.

    Args:
        config: The composed experiment configuration.

    Returns:
        A new ExperimentConfig whose model observation fields, env
        max_grid_extent and sampler evaluation horizons follow from the pair.

    Raises:
        ValueError: If the environment has no geometry, if the representation
            has no entry, or if a discrete representation is asked of an
            environment that emits no symbolic observation.
    """
    geometry = ENVIRONMENT_GEOMETRIES.get(config.env.name)
    if geometry is None:
        raise ValueError(
            f"no geometry for environment {config.env.name!r}. Measure the grid "
            "shape it emits per observation mode, then add a row to "
            f"ENVIRONMENT_GEOMETRIES. Tabled: {sorted(ENVIRONMENT_GEOMETRIES)}"
        )
    representation = REPRESENTATION_SPECS.get(config.data.representation)
    if representation is None:
        raise ValueError(
            f"no entry for representation {config.data.representation!r}. "
            f"Tabled: {sorted(REPRESENTATION_SPECS)}"
        )
    if not representation.is_continuous() and geometry.symbolic_channel_classes is None:
        raise ValueError(
            f"environment {config.env.name!r} emits no symbolic observation, so "
            f"the discrete representation {config.data.representation!r} cannot "
            "be resolved for it"
        )

    tile = representation.tile_size
    modes = geometry.emitted(config.data.modes_stored())
    grid_shape = tuple(
        side * tile for side in geometry.largest_grid_shape(modes)
    )
    if representation.is_continuous():
        channel_classes = None
        channels = representation.channels
    else:
        channel_classes = geometry.symbolic_channel_classes
        channels = len(channel_classes)

    extent = geometry.max_grid_extent(modes) * tile
    if not representation.is_continuous() and geometry.discrete_extent_pin:
        extent = geometry.discrete_extent_pin

    return replace(
        config,
        env=replace(config.env, max_grid_extent=extent),
        model=replace(
            config.model,
            obs_grid_shape=grid_shape,
            obs_channel_classes=channel_classes,
            obs_channels=channels,
            obs_dim=math.prod(grid_shape) * channels,
        ),
        # Onto the sampler rather than read from the contract at each use, so
        # one value has one home and the sampler's own validation covers it.
        sampler=replace(
            config.sampler, evaluation_horizons=geometry.evaluation_horizons
        ),
    )


# The observation fields resolve_observation_contract writes. evaluation_horizons
# is not among them: fast mode overrides that field after the contract has run,
# so re-resolving it would restore the full grid.
RESOLVED_OBSERVATION_FIELDS: tuple[str, ...] = (
    "obs_grid_shape",
    "obs_channels",
    "obs_channel_classes",
    "obs_dim",
)


def assert_widths_match_extent(config: ExperimentConfig) -> None:
    """Raise unless one encoder width is configured per derived layer.

    Args:
        config: A configuration whose observation contract and family defaults
            have both been applied.

    Raises:
        ValueError: If the width count differs from the depth the resolved
            extent derives.
    """
    dilations = dilations_for_grid(config.env.max_grid_extent)
    widths = config.model.encoder_channels
    if len(widths) != len(dilations):
        raise ValueError(
            f"extent {config.env.max_grid_extent} derives {len(dilations)} "
            f"dilated layers {dilations}, against {len(widths)} encoder "
            f"widths {widths}. Call resolve_family_defaults after "
            "resolve_observation_contract."
        )


def assert_config_resolved(config: ExperimentConfig) -> None:
    """Raise unless the observation fields match a fresh contract resolution.

    Re-runs resolve_observation_contract and compares what it sets, so the
    check and the contract cannot drift apart.

    Args:
        config: A configuration whose observation contract has been applied.

    Raises:
        ValueError: If the extent or any observation field differs from what
            the environment's geometry and the representation derive.
    """
    expected = resolve_observation_contract(config)
    differences = []
    if config.env.max_grid_extent != expected.env.max_grid_extent:
        differences.append(
            f"max_grid_extent {config.env.max_grid_extent} against "
            f"{expected.env.max_grid_extent}"
        )
    for name in RESOLVED_OBSERVATION_FIELDS:
        held = getattr(config.model, name)
        derived = getattr(expected.model, name)
        if held != derived:
            differences.append(f"{name} {held!r} against {derived!r}")
    if differences:
        raise ValueError(
            f"configuration for {config.env.name!r} at representation "
            f"{config.data.representation!r} is not resolved: "
            + "; ".join(differences)
            + ". Call resolve_observation_contract on it first."
        )


def resolve_family_defaults(config: ExperimentConfig) -> ExperimentConfig:
    """Return config with the settings its environment's family fixes applied.

    The probability-log dtype varies by family. The encoder widths do not: they
    follow the depth, which follows the extent the observation contract has
    already resolved. Reading that extent off the config rather than
    recomputing it from the environment name is what keeps the widths matched
    to the encoder the same extent builds -- a representation that scales the
    grid, such as pixels over a tile, moves the extent and the depth with it.

    Call after resolve_observation_contract, which sets the extent this reads,
    and before apply_fast_mode, so fast mode scales whichever widths this
    selected.

    Args:
        config: The composed experiment configuration.

    Returns:
        A new ExperimentConfig carrying the encoder widths its extent derives
        and its family's probability-log dtype.

    Raises:
        ValueError: If the family has no probability-log entry, if the resolved
            extent is not positive, or if the config reaching this point has an
            unresolved observation contract.
    """
    resolved = replace(
        config,
        model=replace(
            config.model,
            encoder_channels=encoder_widths_for_depth(
                len(dilations_for_grid(config.env.max_grid_extent))
            ),
        ),
        eval=replace(
            config.eval,
            probability_log_dtype=probability_log_dtype(config.env.name),
        ),
    )
    assert_config_resolved(resolved)
    assert_widths_match_extent(resolved)
    return resolved


def config_snapshot(config: ExperimentConfig) -> dict:
    """Return the provenance fields an artefact needs to explain itself.

    A curated list, not the whole config tree: every field here changes what the
    numbers mean.

    Args:
        config: Experiment configuration for this run.

    Returns:
        Flat mapping of provenance field to value, JSON-serialisable.
    """
    return {
        "env_name": config.env.name,
        "num_envs": config.env.num_envs,
        "max_episode_steps": config.env.max_episode_steps,
        "obs_dim": config.model.obs_dim,
        "total_steps": config.train.total_steps,
        "batch_size": config.train.batch_size,
        "learning_rate": config.train.learning_rate,
        "num_eval_envs": config.eval.num_eval_envs,
        "num_trajectories": config.data.num_trajectories,
        "storage_format": config.data.storage_format,
        # The slip level the trajectories were generated at. Artefacts from E2
        # predate this field and state no level; E3's state their own.
        "slip_probability": config.env.slip_probability,
        # Which stored parameter tree the numbers were scored from. Two arms
        # scored from different trees are not comparable, and without this
        # field the artefacts do not say which tree they used.
        "params_selection": config.eval.params_selection,
    }
