"""Central configuration for the jumpy world model codebase.

All hyperparameters, paths, seeds and named constants live here as nested
dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
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
# One seed of one run lives under a single directory:
#
#     outputs/<run_name>/seed<n>/{checkpoints,eval,logs,sentinels}/
#     outputs/fast/<run_name>/seed<n>/...
#
# The four per-run paths below are functions of (run_name, seed).
FAST_DIR_NAME: str = "fast"
SEED_DIR_PREFIX: str = "seed"
CHECKPOINTS_DIR_NAME: str = "checkpoints"
EVAL_DIR_NAME: str = "eval"
LOGS_DIR_NAME: str = "logs"
SENTINELS_DIR_NAME: str = "sentinels"


def run_root(run_name: str, seed: int, fast: bool = False) -> Path:
    """Return the single directory holding every artefact of one run and seed.

    Args:
        run_name: Experiment run name, WITHOUT any seed suffix. The seed is a
            directory level, so appending it to the name as well would put it
            in the path twice.
        seed: The reporting seed this invocation runs.
        fast: When True the whole tree hangs off `outputs/fast/`, so prototype
            artefacts can never be mistaken for reportable ones.

    Returns:
        `outputs/[fast/]<run_name>/seed<seed>`.
    """
    base = OUTPUTS_DIR / FAST_DIR_NAME if fast else OUTPUTS_DIR
    return base / run_name / f"{SEED_DIR_PREFIX}{seed}"


def checkpoints_dir(run_name: str, seed: int, fast: bool = False) -> Path:
    """Return the checkpoint directory for one run and seed."""
    return run_root(run_name, seed, fast) / CHECKPOINTS_DIR_NAME


def eval_dir(run_name: str, seed: int, fast: bool = False) -> Path:
    """Return the evaluation-artefact directory for one run and seed."""
    return run_root(run_name, seed, fast) / EVAL_DIR_NAME


def logs_dir(run_name: str, seed: int, fast: bool = False) -> Path:
    """Return the log directory for one run and seed."""
    return run_root(run_name, seed, fast) / LOGS_DIR_NAME


def sentinels_dir(run_name: str, seed: int, fast: bool = False) -> Path:
    """Return the sentinel directory for one run and seed.

    The stage subdirectory and filename are added by src.utils.sentinels.
    """
    return run_root(run_name, seed, fast) / SENTINELS_DIR_NAME

# Classes per NAVIX observation channel, for the per-cell categorical decoder.
# The observation is a (19, 19, 3) uint8 grid of discrete codes: entity tag,
# colour, symbolic state.
#
# NEVER read these from observation_space.maximum, which declares 8 and is
# wrong: measured channel-0 values reach 10. Checked by
# tests/test_navix_obs_cardinality.py.
OBS_CHANNEL_CLASSES_NAVIX: tuple[int, ...] = (11, 6, 4)
# MiniHack and NLE are not configured here.

# Spatial shape of the NAVIX observation grid, (height, width).
# INVARIANT, asserted in tests: prod(shape) * len(channel_classes) == obs_dim.
OBS_GRID_SHAPE_NAVIX: tuple[int, int] = (19, 19)

# Channel and class index identifying the agent in a NAVIX observation.
AGENT_CHANNEL_INDEX: int = 0
AGENT_CLASS_INDEX: int = 10

# Artefact filenames. POLICY_METRICS_FILENAME names an artefact no current
# stage writes; it is read by src/pipeline/aggregate.py.
METRICS_FILENAME: str = "metrics.json"
POLICY_METRICS_FILENAME: str = "policy_metrics.json"


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
    """

    name: str = "Navix-FourRooms-v0"
    num_envs: int = 64
    max_episode_steps: int = 256
    penality_coeff: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture settings.

    Holds the observation contract. Encoder and transformer fields are added
    here when those models are built.

    Attributes:
        obs_dim: Flattened observation dimensionality. 1083 = 19*19*3 for
            NAVIX.
        obs_grid_shape: Spatial shape of the observation grid, (height, width).
        obs_channel_classes: Number of classes per observation channel. The
            per-cell categorical target is built from this, which is why the
            loss is cross-entropy rather than MSE: the codes are unordered
            labels, so a regression loss would impose an ordering on them.
        activation: Activation used in the model's MLP blocks.
    """

    obs_dim: int = 1083
    obs_grid_shape: tuple[int, int] = OBS_GRID_SHAPE_NAVIX
    obs_channel_classes: tuple[int, ...] = OBS_CHANNEL_CLASSES_NAVIX
    activation: str = "silu"


@dataclass(frozen=True)
class TrainConfig:
    """Training loop settings.

    A stub. The optimiser and schedule fields are added when the training loop
    is built.

    Attributes:
        total_steps: Gradient steps for a full run. Read back from the config
            snapshot by aggregate.py to reject a truncated seed.
        batch_size: Examples per gradient step.
        learning_rate: Optimiser learning rate.
        log_every_steps: Logging interval.
    """

    total_steps: int = 160_000
    batch_size: int = 16
    learning_rate: float = 4e-5
    log_every_steps: int = 1_000


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation settings.

    Attributes:
        num_eval_envs: Environment rows reserved for held-out evaluation.
    """

    num_eval_envs: int = 8

@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration composing all sub-configs.

    Attributes:
        seed: Active seed for this run (one of SEEDS).
        fast: Whether this is a prototype run. Fast mode is a DIRECTORY level
            rather than a name prefix, so the path functions need this flag and
            the run name stays clean. A fast run is never reportable.
        run_name: Run identifier used for the output directory. Generic by
            default; pass --run-name explicitly for a real experiment.
        env: Environment configuration.
        model: Model architecture configuration.
        train: Training loop configuration.
        eval: Evaluation configuration.
    """

    seed: int = DEFAULT_SEED
    run_name: str = "run"
    fast: bool = False
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# --- Fast/prototyping mode ---------------------------------------------------
# Smoke-test sizes, consumed only by apply_fast_mode. Never for reporting.
FAST_NUM_ENVS: int = 8
FAST_MAX_EPISODE_STEPS: int = 32
FAST_TOTAL_STEPS: int = 20
FAST_BATCH_SIZE: int = 4
FAST_LOG_EVERY_STEPS: int = 5
FAST_NUM_EVAL_ENVS: int = 2


SEED_RUN_SUFFIX: str = "_seed"

# Format for a run_name generated when the operator supplies none. Sortable,
# filesystem-safe, and unique to the second.
RUN_NAME_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"
RUN_NAME_DEFAULT_PREFIX: str = "run_"


def seed_scoped_run_name(run_name: str, seed: int) -> str:
    """Return a run name carrying its seed, so seeds cannot share a directory.

    Every per-run artefact path keys on run_name alone, so the seed is applied
    here rather than threaded through each call site. A guard the operator has
    to remember at five call sites is not a guard.

    IDEMPOTENT by design. `--run-name e1_base_seed42 --seed 42` is a natural
    thing to type and must not become `e1_base_seed42_seed42`, so a name
    already carrying this exact seed is returned unchanged. A name carrying a
    DIFFERENT seed is still suffixed, because trusting it would reintroduce the
    collision this exists to prevent.

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


def apply_fast_mode(config: ExperimentConfig) -> ExperimentConfig:
    """Return config with scale fields overridden to fast-mode sizes.

    Scales how much work is done, not what is computed. `fast=True` puts every
    artefact under `outputs/fast/`, so a prototype run cannot collide with a
    reporting one.

    Args:
        config: The experiment configuration to scale down.

    Returns:
        A new ExperimentConfig with env, train and eval scaled to the FAST_*
        constants and fast set True. Seed, model and run_name unchanged.
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
        ),
        eval=replace(config.eval, num_eval_envs=FAST_NUM_EVAL_ENVS),
    )


def config_snapshot(config: ExperimentConfig) -> dict:
    """Return the provenance fields an artefact needs to explain itself.

    A curated list, not the whole config tree: every field here changes what
    the numbers mean.

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
    }
