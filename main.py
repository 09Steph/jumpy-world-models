"""Command-line entry point for the jumpy world model pipeline.

Parses arguments, builds the configuration, configures logging and seeds
determinism, then hands off to the pipeline runner. Each read-only mode is
dispatched to its own module and exits before the pipeline. A run whose recorded
representation differs from the one requested is refused before any stage.

`--stages` decides which stages are built, and sentinels decide which of those
skip. A stage the flag omits is never constructed, and a built stage still skips
on a matching sentinel.

Usage:
    python main.py --seed 42 --run-name e1_base
    python main.py --fast
    python main.py --env Navix-DoorKey-16x16-v0
    python main.py --arm 2 --run-name e2_base
    python main.py --aggregate --run-name e1_base --arm 2
    python main.py --sweep --run-name e2_base --arm-run 1=e1_50k
    python main.py --fit --run-name e2_base
    python main.py --displacement-sweep --run-name displacement_sweep
    python main.py --intermediate-states --run-name e2_base
    python main.py --rank-atari-games --position 0 --out <directory>
    python main.py --sweep-atari-checkpoints --out <directory>
    python main.py --source-run e2_base --stages evaluate --split test --arm 1
    python main.py --rescore-test-split --write-manifest --runs e2_base \
        --manifest-source <backup outputs tree> --manifest <file>
    python main.py --rescore-test-split --runs e2_base --manifest <file> \
        --regenerate --dry-run
    python main.py --figures
    python main.py --publish <directory>
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.utils.platform_guard import force_cpu_backend_on_apple_silicon

force_cpu_backend_on_apple_silicon()

import navix  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order

from config import (  # noqa: E402  pylint: disable=wrong-import-position
    ARM_DIRECT,
    ARMS,
    COLLECTION_POLICIES,
    REPRESENTATIONS,
    DEFAULT_ENV_NAME,
    DEFAULT_SEED,
    ENCODER_WIDTH_CANDIDATES,
    ENV_FAMILY_NAVIX,
    EVALUATED_SPLITS,
    EVALUATION_HORIZONS,
    HORIZON_MAX,
    HORIZON_MIN,
    OBS_MODE_TOP_DOWN,
    OBS_MODES,
    OUTPUTS_DIR,
    ENVIRONMENT_GEOMETRIES,
    PARAMS_SELECTIONS,
    ROLLOUT_PATH_TOKENS,
    ROLLOUT_PATHS,
    SEEDS,
    METRICS_TEMPLATE,
    SELECTABLE_STAGES,
    STAGE_NAME_GENERATE,
    STAGE_NAME_OFFLINE_GENERATE,
    SLIP_PROBABILITIES,
    SPLIT_NAME_TEST,
    SPLIT_NAME_VALIDATION,
    TEST_RUN_SUFFIX,
    TRUTH_SOURCE_TEST_TRAJECTORY,
    TRUTH_SOURCES,
    DataConfig,
    ExperimentConfig,
    SamplerMode,
    apply_fast_mode,
    default_run_name,
    env_family,
    logs_dir,
    resolve_family_defaults,
    resolve_observation_contract,
    run_root,
    test_scoped_run_name,
    training_horizon_max_for_env,
)
from src.data.split import SplitName  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.data.trajectory import ObservationMode  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.eval.atari_archive import (  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
    run_checkpoint_sweep_cli,
    run_game_ranking_cli,
)
from src.eval.figures import FiguresRequest, run_figures_cli, run_publish_cli  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.eval.intermediate_states import (  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
    DecodeRequest,
    run_intermediate_states_cli,
)
from src.pipeline.aggregate import run_aggregation_cli  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.pipeline.displacement_sweep import run_displacement_sweep_cli  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.pipeline.fit import run_fit_cli  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.pipeline.metrics_schema import SOURCE_RUN_NAME_KEY  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.pipeline.rescore import (  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
    RescoreRequest,
    run_rescore_cli,
    write_manifest,
)
from src.pipeline.runner import run_pipeline  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.pipeline.sweep import run_sweep_cli  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.utils.determinism import set_determinism  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.utils.logging_setup import (  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
    configure_logging,
    get_logger,
)
from src.utils.paths import safe_rel  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.utils.sentinels import read_stage_metadata  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Jumpy world model pipeline")
    # Training runs every seed --data-seeds selects, whatever this is set to.
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        choices=SEEDS,
        help="The reporting seed that places the log, seeds global determinism "
        "and picks the --intermediate-states cell. Training runs every seed "
        "--data-seeds selects regardless.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output directory name. Default: timestamped run_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--verbose-deps",
        action="store_true",
        help="Emit third-party logging too. For debugging a library directly.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Log to the terminal only. File logging is on by default.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smoke test at minimal sizes, into outputs/fast/. Not for results.",
    )
    parser.add_argument(
        "--sampler-mode",
        type=str,
        default=None,
        choices=tuple(mode.value for mode in SamplerMode),
        help="Override the window sampler mode. Default: the config value.",
    )
    # Validated at parse time, before logging, determinism or any directory.
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Override the environment for this run. Default: the config value.",
    )
    # One invocation trains and evaluates one observation mode.
    parser.add_argument(
        "--obs-mode",
        type=str,
        default=None,
        choices=OBS_MODES,
        help="Override the observation mode for this run. Default: the config value.",
    )
    # The checkpoint carries both trees, so this re-scores without retraining.
    # It is in the evaluate sentinel identity, so a changed selection re-runs
    # evaluation.
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        choices=PARAMS_SELECTIONS,
        help="Which stored parameter tree evaluation scores. Default: the "
        "config value.",
    )
    # One invocation is one arm. The arm is a field on the model stages, not the
    # config.
    parser.add_argument(
        "--arm",
        type=int,
        default=ARM_DIRECT,
        choices=ARMS,
        help="Which arm to run: 1 direct, 2 autoregressive/endpoint, "
        "3 autoregressive/one-step. Default: 1.",
    )
    # Not on the config and in no sentinel identity, so a subset run and a full
    # run agree seed for seed.
    parser.add_argument(
        "--data-seeds",
        type=_data_seed_subset,
        default=SEEDS,
        metavar="SEED[,SEED...]",
        help="Run only these reporting seeds, in SEEDS order. Default: every "
        "seed in SEEDS.",
    )
    # A value outside SLIP_PROBABILITIES is refused at the command line.
    parser.add_argument(
        "--slip",
        type=float,
        default=None,
        choices=SLIP_PROBABILITIES,
        help="Probability an action is resampled before the environment "
        "executes it. Default: the config value.",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Gradient steps for this run. Default: the config value.",
    )
    # In TrainStage.sentinel_identity, so a changed value retrains.
    parser.add_argument(
        "--training-horizon-max",
        type=int,
        default=None,
        help="Largest horizon this run trains on. Default: the environment's "
        "value.",
    )
    # In GenerateStage.sentinel_identity only, so under the same run name the
    # data regenerates but the trained model is reused. Use a fresh run name.
    parser.add_argument(
        "--collection-policy",
        type=str,
        default=None,
        choices=COLLECTION_POLICIES,
        help="Which behaviour policy collects the dataset. Default: the config "
        "value.",
    )
    # A run whose data was stored under another representation is refused before
    # any stage.
    parser.add_argument(
        "--representation",
        type=str,
        default=None,
        choices=REPRESENTATIONS,
        help="How observations are rendered and stored. Default: the config "
        "value.",
    )
    # encoder_channels is in TrainStage.sentinel_identity, so a changed set
    # retrains.
    parser.add_argument(
        "--encoder-widths",
        type=str,
        default=None,
        choices=sorted(ENCODER_WIDTH_CANDIDATES),
        help="Which declared encoder width set to build. Default: the set the "
        "environment's family selects.",
    )
    parser.add_argument(
        "--disable-early-termination",
        action="store_true",
        help="End episodes on the step cap alone, removing the environment's "
        "termination. Changes what is generated, so use a fresh run name.",
    )
    # A flag rather than a config field, so it is typed on every invocation.
    parser.add_argument(
        "--split",
        type=str,
        default=SPLIT_NAME_VALIDATION,
        choices=EVALUATED_SPLITS,
        help="Which split evaluation scores. The test split is the reported "
        "result and also scores both parameter trees and the extrapolation "
        "horizons. Default: validation.",
    )
    # Restricts the run to the named stages. See the module docstring for why
    # this does not compete with the sentinels.
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=None,
        metavar="STAGE",
        help="Run only these stages, in the pipeline's own order. Default: "
        "every stage the environment declares.",
    )
    # Which run the evaluation READS from, when that is not the run it writes
    # to. The target is derived from this and never typed, so a mismatched pair
    # cannot be expressed.
    parser.add_argument(
        "--source-run",
        type=str,
        default=None,
        help="Read shards and checkpoints from this run and write to "
        f"<run>{TEST_RUN_SUFFIX}. Requires --split test.",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate an existing seed series and exit. Usually automatic.",
    )
    # The aggregate is written inside the given directory.
    parser.add_argument(
        "--series-dir",
        type=Path,
        default=None,
        help="Directory of seed run directories for --aggregate, --sweep or "
        "--fit; infers --run-name.",
    )
    # Waives the all-seeds-present requirement only. A truncated or
    # config-mismatched seed is still refused, and the auto-trigger cannot
    # reach this flag.
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Aggregate fewer than all of config.SEEDS, recording the real count.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Combine the per-arm aggregates into one cross-arm artefact and "
        "exit. Reads only; aggregate the arms first.",
    )
    # Arm 1 is reused across experiments and the checkpoint directory follows
    # the run name, so the arms of one comparison can sit under different runs.
    # Unset arms fall back to --run-name.
    parser.add_argument(
        "--arm-run",
        action="append",
        default=None,
        metavar="N=RUN_NAME",
        help="Read arm N from another run. Repeatable. --sweep only.",
    )
    # Reads the sweep artefact and writes the fits beside it. Read-only in the
    # same sense --sweep is: it opens what has been written and adds a file.
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit the exponent and the usable horizon from a written sweep "
        "and exit. Reads only; sweep the arms first.",
    )
    # Exits before the pipeline. It trains a PPO policy per environment and seed
    # but no world model.
    parser.add_argument(
        "--displacement-sweep",
        action="store_true",
        help="Measure displacement across every registered NAVIX environment "
        "under both collection policies and exit.",
    )
    # Resumption for --displacement-sweep, which has no sentinel: the records
    # file it appends to is what a resumed run reads.
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip any environment and seed already in the records file.",
    )
    # Reads checkpoints, writes figures and exits before the pipeline.
    parser.add_argument(
        "--intermediate-states",
        action="store_true",
        help="Decode arms 2 and 3 step by step from a trained checkpoint, "
        "write the strips and the per-step measurements, and exit.",
    )
    parser.add_argument(
        "--rollout-path",
        choices=ROLLOUT_PATHS,
        default=ROLLOUT_PATH_TOKENS,
        help="Which rollout --intermediate-states draws. The training path is "
        "the default; it holds the states the scored path never exposes.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Read checkpoints from this root instead of outputs/, keeping "
        "every directory level below it. For trees held off the repository.",
    )
    parser.add_argument(
        "--truth-source",
        choices=TRUTH_SOURCES,
        default=TRUTH_SOURCE_TEST_TRAJECTORY,
        help="Where --intermediate-states takes its true states. The held-out "
        "episode is the default and falls back to a fresh roll when its shards "
        "or split are absent. A fresh roll uses the default policy and dynamics.",
    )
    # A cell is one seed and one mode, drawing one strip per evaluation horizon.
    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="Decode every reporting seed and both observation modes, instead "
        "of the one cell named by --seed and --obs-mode.",
    )
    # Read the Atari archive rather than a run, and write to --out. Like the
    # modes above they exit before the pipeline and train nothing.
    parser.add_argument(
        "--rank-atari-games",
        action="store_true",
        help="Measure every game in the Atari archive at one position against "
        "the displacement bar and the length gate, write the ranking to "
        "--out, and exit.",
    )
    parser.add_argument(
        "--sweep-atari-checkpoints",
        action="store_true",
        help="Measure every game in the Atari archive at each swept position, "
        "write the sweep to --out, and exit.",
    )
    parser.add_argument(
        "--position",
        type=_archive_position,
        default=None,
        help="Archive position --rank-atari-games measures.",
    )
    parser.add_argument(
        "--positions",
        type=_archive_position,
        nargs="+",
        default=None,
        metavar="POSITION",
        help="Archive positions --sweep-atari-checkpoints measures. Default: "
        "the config value.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory the Atari archive modes write their JSON and text "
        "matrix to. Must lie outside outputs/.",
    )
    _add_rescore_arguments(parser)
    _add_figures_arguments(parser)
    args = parser.parse_args()
    _validate_aggregate_args(parser, args)
    _validate_archive_args(parser, args)
    _validate_rescore_args(parser, args)
    _validate_env_arg(parser, args)
    _validate_evaluation_args(parser, args)
    return args


def _add_rescore_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --rescore-test-split and the flags that belong to it.

    It scores existing checkpoints through child invocations of this file and
    trains nothing, so like the other read-only modes it exits before the
    pipeline.

    Args:
        parser: The parser to extend.
    """
    parser.add_argument(
        "--rescore-test-split",
        action="store_true",
        help="Re-score every existing test-split cell of --runs under the "
        "current code, then aggregate, sweep and fit them, and exit.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        default=None,
        metavar="RUN",
        help="Source runs --rescore-test-split covers, without the test suffix.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="The shard manifest --rescore-test-split gates against, or "
        "writes with --write-manifest.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Digest the original shards under --manifest-source into "
        "--manifest instead of re-scoring.",
    )
    parser.add_argument(
        "--manifest-source",
        type=Path,
        default=None,
        help="The outputs tree holding the original shards. --write-manifest "
        "only.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate a seed's shards when they are absent, instead of "
        "refusing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --rescore-test-split, list every cell and sentinel it would "
        "touch; with --figures, report what would be drawn. Touches nothing.",
    )
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help="Score a seed the manifest has no entry for, recorded as ungated. "
        "Without it such a seed is skipped.",
    )


def _add_figures_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --figures, --publish and the flag that belongs to them.

    Both read artefacts already on disk and train nothing, so like the other
    read-only modes they exit before the pipeline.

    Args:
        parser: The parser to extend.
    """
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Draw every registered thesis figure the artefacts under outputs/ "
        "support into outputs/figures/, report each one drawn, partial, skipped "
        "or failed, and exit. --run-name keeps only the figures reading that "
        "run, and the manifest then lists only those plus any skipped or failed.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="With --figures, also draw the development figures: the validation "
        "twins and every scalar metric of every discovered run.",
    )
    parser.add_argument(
        "--publish",
        type=Path,
        default=None,
        metavar="DIR",
        help="Copy the drawn and partial reported-split figures in the manifest "
        "into DIR, report each one new, updated or unchanged, and exit.",
    )


def _data_seed_subset(raw: str) -> tuple[int, ...]:
    """Resolve --data-seeds to the named reporting seeds, in SEEDS order.

    Args:
        raw: The comma-separated value passed to --data-seeds.

    Returns:
        The named seeds, ordered by SEEDS rather than by how they were typed.

    Raises:
        argparse.ArgumentTypeError: If the value names no seed, names one that
            is not an integer, names one SEEDS does not hold, or repeats one.
    """
    fields = [field.strip() for field in raw.split(",") if field.strip()]
    if not fields:
        raise argparse.ArgumentTypeError(
            f"--data-seeds {raw!r} names no seed. Omit the flag to run every "
            f"seed in {list(SEEDS)}."
        )
    seeds: list[int] = []
    for field in fields:
        try:
            seed = int(field)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"--data-seeds value {field!r} is not an integer"
            ) from error
        if seed not in SEEDS:
            raise argparse.ArgumentTypeError(
                f"--data-seeds value {seed} is not a reporting seed. A subset "
                f"selects from {list(SEEDS)} and cannot introduce a seed, "
                "which would file a reported cell outside the protocol."
            )
        if seed in seeds:
            raise argparse.ArgumentTypeError(
                f"--data-seeds names {seed} twice: the repeat would resolve to "
                "one directory and one sentinel"
            )
        seeds.append(seed)
    return tuple(seed for seed in SEEDS if seed in seeds)


def _archive_position(raw: str) -> int:
    """Return one Atari archive position.

    Args:
        raw: The value passed to --position or --positions.

    Returns:
        The position.

    Raises:
        argparse.ArgumentTypeError: If the value is not a non-negative integer.
    """
    try:
        position = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{raw} is not an archive position: positions are shard indices"
        ) from error
    if position < 0:
        raise argparse.ArgumentTypeError(
            f"{raw} is not an archive position: positions are shard indices "
            "and start at 0"
        )
    return position


def _validate_evaluation_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject evaluation flag combinations that cannot mean anything.

    Args:
        parser: The parser, used to emit a usage error and exit.
        args: The parsed arguments.
    """
    if args.stages is not None:
        unknown = sorted(set(args.stages) - set(SELECTABLE_STAGES))
        if unknown:
            parser.error(
                f"unknown --stages {unknown}. Selectable stages: "
                f"{', '.join(SELECTABLE_STAGES)}"
            )
    if args.source_run is not None:
        if args.run_name is not None:
            parser.error(
                "--source-run and --run-name are mutually exclusive: the "
                f"target is derived as <source>{TEST_RUN_SUFFIX}, so naming it "
                "as well is the only way the two could disagree."
            )
        if args.split != SPLIT_NAME_TEST:
            parser.error(
                f"--source-run requires --split {SPLIT_NAME_TEST}: reading one "
                "run while scoring validation would file a validation number "
                "under the test pass's own run name."
            )
        directory = run_root(args.source_run, args.seed, args.fast, args.env)
        if not directory.parent.is_dir():
            parser.error(
                f"--source-run {args.source_run!r} has no run directory at "
                f"{safe_rel(directory.parent)}. A typo here would create a "
                "third tree and the pass would report success."
            )
    if args.split == SPLIT_NAME_TEST and args.params is not None:
        parser.error(
            "--params does not apply to the test pass: it scores BOTH stored "
            "parameter trees and reports their gap, so there is no selection "
            "to make."
        )


def _validate_env_arg(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject an --env value the family that owns it does not recognise.

    NAVIX names are checked against `navix.registry()`. Names in every other
    family are checked against ENVIRONMENT_GEOMETRIES.

    Args:
        parser: The parser, used so a bad value exits with argparse's usage
            message rather than a traceback.
        args: Parsed arguments.
    """
    if args.env is None:
        return
    tabled = sorted(ENVIRONMENT_GEOMETRIES)
    try:
        family = env_family(args.env)
    except ValueError:
        parser.error(
            f"unknown --env {args.env!r}: it matches no environment family. "
            f"Tabled environments: {', '.join(tabled)}"
        )
        return
    if family == ENV_FAMILY_NAVIX:
        registered = sorted(navix.registry())
        if args.env not in registered:
            parser.error(
                f"unknown --env {args.env!r}. NAVIX registers "
                f"{len(registered)} environments: {', '.join(registered)}"
            )
        return
    if args.env not in tabled:
        parser.error(
            f"unknown --env {args.env!r} in family {family!r}. Tabled "
            f"environments: {', '.join(tabled)}"
        )


# Read-only modes that write a named artefact under outputs/, so a run name has
# to place it. A mode writing to --out needs none.
OUTPUTS_ARTEFACT_MODES: tuple[str, ...] = (
    "--aggregate",
    "--sweep",
    "--fit",
    "--displacement-sweep",
    "--intermediate-states",
)


def _validate_aggregate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject read-only flag combinations that cannot mean anything.

    Exits via parser.error when a flag is passed without the read-only mode it
    belongs to, when more than one read-only mode is asked for at once, or when
    a mode in OUTPUTS_ARTEFACT_MODES has neither --run-name nor --series-dir to
    resolve a run name.

    Args:
        parser: The parser, used to emit a usage error and exit.
        args: The parsed arguments.
    """
    _validate_figures_args(parser, args)
    asked = [
        flag
        for flag, given in (
            ("--aggregate", args.aggregate),
            ("--sweep", args.sweep),
            ("--fit", args.fit),
            ("--displacement-sweep", args.displacement_sweep),
            ("--intermediate-states", args.intermediate_states),
            ("--rank-atari-games", args.rank_atari_games),
            ("--sweep-atari-checkpoints", args.sweep_atari_checkpoints),
            ("--rescore-test-split", args.rescore_test_split),
            ("--figures", args.figures),
            ("--publish", args.publish is not None),
        )
        if given
    ]
    if len(asked) > 1:
        parser.error(
            f"{' and '.join(asked)} write different artefacts; run them as "
            "separate invocations so each names the one it produced"
        )
    if args.allow_partial and not args.aggregate:
        parser.error("--allow-partial only applies to --aggregate")
    if args.arm_run and not args.sweep:
        parser.error("--arm-run only applies to --sweep")
    if args.skip_existing and not args.displacement_sweep:
        parser.error("--skip-existing only applies to --displacement-sweep")
    for flag, given in (
        ("--rollout-path", args.rollout_path != ROLLOUT_PATH_TOKENS),
        ("--checkpoint-root", args.checkpoint_root is not None),
        ("--all-cells", args.all_cells),
        ("--truth-source", args.truth_source != TRUTH_SOURCE_TEST_TRAJECTORY),
    ):
        if given and not args.intermediate_states:
            parser.error(f"{flag} only applies to --intermediate-states")
    # Only --aggregate, --sweep and --fit read a seed series.
    if args.series_dir is not None and args.intermediate_states:
        parser.error("--series-dir does not apply to --intermediate-states")
    if args.series_dir is not None and args.displacement_sweep:
        parser.error("--series-dir does not apply to --displacement-sweep")
    seriesless = (
        args.rank_atari_games,
        args.sweep_atari_checkpoints,
        args.rescore_test_split,
        args.figures,
        args.publish is not None,
    )
    if args.series_dir is not None and any(seriesless):
        parser.error(f"--series-dir does not apply to {asked[0]}")
    if not asked:
        if args.series_dir is not None:
            parser.error(
                "--series-dir only applies to --aggregate, --sweep or --fit"
            )
        return
    if (
        asked[0] in OUTPUTS_ARTEFACT_MODES
        and args.run_name is None
        and args.series_dir is None
    ):
        parser.error(
            f"{asked[0]} needs --run-name, or --series-dir to infer it from. "
            "outputs/ holds many unrelated runs, so a run name cannot be "
            "inferred without one of them."
        )


def _validate_archive_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject Atari archive flag combinations that cannot mean anything.

    Exits via parser.error when --position, --positions or --out is passed
    without its mode, when a mode is missing a flag it needs, when --positions
    repeats a position, or when --out lies under outputs/.

    Args:
        parser: The parser, used to emit a usage error and exit.
        args: The parsed arguments.
    """
    if args.position is not None and not args.rank_atari_games:
        parser.error("--position only applies to --rank-atari-games")
    if args.positions is not None and not args.sweep_atari_checkpoints:
        parser.error("--positions only applies to --sweep-atari-checkpoints")
    archive_mode = next(
        (
            flag
            for flag, given in (
                ("--rank-atari-games", args.rank_atari_games),
                ("--sweep-atari-checkpoints", args.sweep_atari_checkpoints),
            )
            if given
        ),
        None,
    )
    if archive_mode is None:
        if args.out is not None:
            parser.error(
                "--out only applies to --rank-atari-games or "
                "--sweep-atari-checkpoints"
            )
        return
    if args.rank_atari_games and args.position is None:
        parser.error(
            "--rank-atari-games needs --position: one invocation measures one "
            "archive position"
        )
    if args.out is None:
        parser.error(
            f"{archive_mode} needs --out: its results are written outside "
            f"{safe_rel(OUTPUTS_DIR)}/"
        )
    if args.out.resolve().is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error(
            f"--out must lie outside {safe_rel(OUTPUTS_DIR)}/: the archive "
            "modes write nothing under an experiment run"
        )
    if args.positions is not None:
        repeated = sorted({p for p in args.positions if args.positions.count(p) > 1})
        if repeated:
            parser.error(
                f"--positions names {', '.join(map(str, repeated))} twice: "
                "each position is one row per game"
            )


def _validate_figures_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject figure flag combinations that cannot mean anything.

    Args:
        parser: The parser, used to emit a usage error and exit.
        args: The parsed arguments.
    """
    if args.all and not args.figures:
        parser.error("--all only applies to --figures")
    if args.fast and (args.figures or args.publish is not None):
        parser.error("--figures and --publish read the reported tree, not the --fast one")


def _validate_rescore_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject re-score flag combinations that cannot mean anything.

    Exits via parser.error when a re-score flag is passed without the mode,
    when the mode is missing --runs or --manifest, when a flag belongs to the
    other of its two steps, or when a run is named with its test suffix.

    Args:
        parser: The parser, used to emit a usage error and exit.
        args: The parsed arguments.
    """
    given = [
        flag
        for flag, value in (
            ("--runs", args.runs is not None),
            ("--manifest", args.manifest is not None),
            ("--write-manifest", args.write_manifest),
            ("--manifest-source", args.manifest_source is not None),
            ("--regenerate", args.regenerate),
            ("--dry-run", args.dry_run and not args.figures),
            ("--allow-ungated", args.allow_ungated),
        )
        if value
    ]
    if not args.rescore_test_split:
        if given:
            parser.error(f"{given[0]} only applies to --rescore-test-split")
        return
    if args.fast:
        parser.error("--rescore-test-split re-scores reported runs, not --fast ones")
    if args.runs is None:
        parser.error("--rescore-test-split needs --runs")
    suffixed = [run for run in args.runs if run.endswith(TEST_RUN_SUFFIX)]
    if suffixed:
        parser.error(
            f"--runs names source runs, without {TEST_RUN_SUFFIX}: {suffixed}"
        )
    if args.write_manifest:
        for flag in ("--regenerate", "--dry-run", "--allow-ungated"):
            if flag in given:
                parser.error(f"{flag} does not apply to --write-manifest")
        if args.manifest is None or args.manifest_source is None:
            parser.error("--write-manifest needs --manifest and --manifest-source")
        if not args.manifest_source.is_dir():
            parser.error("--manifest-source is not a directory")
        return
    if args.manifest_source is not None:
        parser.error("--manifest-source only applies to --write-manifest")
    if args.manifest is None and not args.dry_run:
        parser.error(
            "--rescore-test-split needs --manifest: every seed is gated "
            "against it before it is scored"
        )


def resolve_run_name(args: argparse.Namespace) -> str:
    """Return the base run name every artefact for this invocation keys on.

    Derived from --source-run when one is given. The two flags are mutually
    exclusive, so a target cannot be typed that disagrees with its source.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The derived target for a source-run pass, args.run_name when given,
        otherwise a timestamped default. Carries neither the fast marker nor
        the seed: both are directory levels.
    """
    # getattr, because this is called with hand-built namespaces that carry
    # only the fields under test. The parser always sets it.
    source_run = getattr(args, "source_run", None)
    if source_run is not None:
        return test_scoped_run_name(source_run)
    return args.run_name if args.run_name is not None else default_run_name()


REPRESENTATION_SENTINEL_KEY: str = "representation"

# The dataset stages that record which representation a run stored. A run
# carries one of them: NAVIX generates and the offline families convert.
DATASET_STAGE_NAMES: tuple[str, ...] = (
    STAGE_NAME_GENERATE,
    STAGE_NAME_OFFLINE_GENERATE,
)


def recorded_representation(config: ExperimentConfig) -> str | None:
    """Return the representation a run's data was stored under, if recorded.

    A generated run records it in its `generate` sentinel and an offline run in
    `offline_generate`. A test-split run writes no dataset stage, so its
    representation is read from its source run.

    Args:
        config: The composed experiment configuration.

    Returns:
        The recorded representation, or None where no run has stored data yet.
    """
    own = _representation_recorded_by(config.run_name, config)
    if own is not None:
        return own
    return _representation_via_source_run(config)


def _representation_recorded_by(
    run_name: str, config: ExperimentConfig
) -> str | None:
    """Return the representation one named run recorded beside its dataset.

    Args:
        run_name: The run whose dataset sentinel is read.
        config: The composed experiment configuration, supplying the seed,
            fast flag and environment the sentinel is scoped by.

    Returns:
        The recorded representation, or None where that run wrote no dataset
        stage or recorded no representation.
    """
    for stage in DATASET_STAGE_NAMES:
        metadata = read_stage_metadata(
            stage,
            run_name,
            config.data_seed,
            config.fast,
            config.env.name,
        )
        if metadata and REPRESENTATION_SENTINEL_KEY in metadata:
            return metadata[REPRESENTATION_SENTINEL_KEY]
    return None


def _source_run_of(run_name: str) -> str | None:
    """Return the run a test-scoped name was derived from.

    The inverse of `test_scoped_run_name`.

    Args:
        run_name: This run's name.

    Returns:
        The source run name, or None where this name carries no test suffix
        and so was not derived from another run.
    """
    if not run_name.endswith(TEST_RUN_SUFFIX):
        return None
    return run_name[: -len(TEST_RUN_SUFFIX)]


def _representation_via_source_run(config: ExperimentConfig) -> str | None:
    """Return the representation of the run this one scores against.

    The source run is first derived from this run's name. Where that run has
    no directory or recorded nothing, the source named in an existing metrics
    artefact is read instead.

    Args:
        config: The composed experiment configuration.

    Returns:
        The source run's recorded representation, or None where this run names
        no source or the source has recorded nothing.
    """
    derived = _source_run_of(config.run_name)
    if derived and run_root(
        derived, config.data_seed, config.fast, config.env.name
    ).exists():
        recorded = _representation_recorded_by(derived, config)
        if recorded is not None:
            return recorded

    # Searched from the seed root: metrics artefacts are arm-scoped.
    root = run_root(
        config.run_name, config.data_seed, config.fast, config.env.name
    )
    for path in sorted(root.rglob(METRICS_TEMPLATE.format(mode="*"))):
        try:
            artefact = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        source = artefact.get(SOURCE_RUN_NAME_KEY)
        if source is None or source == config.run_name:
            continue
        recorded = _representation_recorded_by(source, config)
        if recorded is not None:
            return recorded
    return None


def assert_representation_matches(config: ExperimentConfig) -> None:
    """Raise if this run's data was stored under another representation.

    Checked before any stage runs.

    Args:
        config: The composed experiment configuration.

    Raises:
        ValueError: If a recorded representation differs from the requested
            one.
    """
    recorded = recorded_representation(config)
    if recorded is None or recorded == config.data.representation:
        return
    raise ValueError(
        f"run {config.run_name!r} stored its data under representation "
        f"{recorded!r}, and this invocation asks for "
        f"{config.data.representation!r}. A representation is different data, "
        f"not a different reading of the same data. Pass --representation "
        f"{recorded} to read that data, or a different --run-name to store "
        f"another."
    )


def _with_environment_horizon(config: ExperimentConfig) -> ExperimentConfig:
    """Return config carrying the trained horizon its environment fixes.

    Args:
        config: The configuration being composed.

    Returns:
        The configuration, with train.horizon_max set from the environment.
    """
    horizon = training_horizon_max_for_env(config.env.name)
    if horizon == config.train.horizon_max:
        return config
    logger.warning(
        "training horizon set to %d by environment %s, against the configured "
        "%d",
        horizon,
        config.env.name,
        HORIZON_MAX,
    )
    return replace(config, train=replace(config.train, horizon_max=horizon))


def _with_horizon_override(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentConfig:
    """Return config carrying the trained horizon this invocation asks for.

    Args:
        config: The configuration being composed.
        args: Parsed command-line arguments.

    Returns:
        The configuration, unchanged where the flag was not passed.

    Raises:
        ValueError: If the requested horizon is below HORIZON_MIN.
    """
    if args.training_horizon_max is None:
        return config
    if args.training_horizon_max < HORIZON_MIN:
        raise ValueError(
            f"--training-horizon-max must be at least {HORIZON_MIN}, got "
            f"{args.training_horizon_max}"
        )
    logger.warning(
        "training horizon overridden to %d for this invocation only, against "
        "the configured %d",
        args.training_horizon_max,
        config.train.horizon_max,
    )
    return replace(
        config,
        train=replace(config.train, horizon_max=args.training_horizon_max),
    )


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Construct the experiment configuration from parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The composed configuration with every passed override applied.

    Raises:
        ValueError: If the environment has no row in ENVIRONMENT_GEOMETRIES,
            --total-steps is not positive, or --training-horizon-max is below
            HORIZON_MIN.
    """
    config = ExperimentConfig(seed=args.seed, run_name=resolve_run_name(args))
    if args.env is not None:
        config = replace(config, env=replace(config.env, name=args.env))
        logger.warning(
            "environment overridden to %s for this invocation only", args.env
        )
    if args.representation is not None:
        config = replace(
            config,
            data=replace(config.data, representation=args.representation),
        )
        logger.info(
            "representation overridden to %s for this invocation only",
            args.representation,
        )
    # After --env and --representation, which it composes, on every run.
    config = resolve_observation_contract(config)
    # After the contract, which fixes the extent the widths must match, and
    # before fast mode, so fast mode scales whichever set is selected.
    config = resolve_family_defaults(config)
    # Before fast mode, so a smoke run scales the environment's horizon.
    config = _with_environment_horizon(config)
    if args.encoder_widths is not None:
        config = replace(
            config,
            model=replace(
                config.model,
                encoder_channels=ENCODER_WIDTH_CANDIDATES[args.encoder_widths],
            ),
        )
        logger.warning(
            "encoder widths overridden to %s (%s) for this invocation only",
            args.encoder_widths,
            ENCODER_WIDTH_CANDIDATES[args.encoder_widths],
        )
    if args.disable_early_termination:
        config = replace(
            config, env=replace(config.env, disable_early_termination=True)
        )
        logger.warning(
            "early termination disabled: episodes end on the step cap alone"
        )
    if args.fast:
        config = apply_fast_mode(config)
        logger.warning(
            "FAST MODE active (run_name=%s), results not for reporting",
            config.run_name,
        )
    if args.sampler_mode is not None:
        mode = SamplerMode(args.sampler_mode)
        config = replace(config, sampler=replace(config.sampler, mode=mode))
        if mode is not SamplerMode.HYBRID:
            logger.warning(
                "sampler mode overridden to %s: this run is a verification "
                "instrument, not a reporting run",
                mode.value,
            )
    if args.obs_mode is not None:
        config = replace(
            config, sampler=replace(config.sampler, observation_mode=args.obs_mode)
        )
        logger.info(
            "observation mode overridden to %s for this invocation only",
            args.obs_mode,
        )
    if args.slip is not None:
        config = replace(
            config, env=replace(config.env, slip_probability=args.slip)
        )
        logger.info(
            "slip probability overridden to %s for this invocation only",
            args.slip,
        )
    if args.total_steps is not None:
        if args.total_steps < 1:
            raise ValueError(
                f"--total-steps must be positive, got {args.total_steps}"
            )
        configured_steps = config.train.total_steps
        config = replace(
            config, train=replace(config.train, total_steps=args.total_steps)
        )
        logger.warning(
            "total steps overridden to %d for this invocation only, against "
            "the configured %d",
            args.total_steps,
            configured_steps,
        )
    # After fast mode, so a named horizon is trained at even in a smoke run.
    config = _with_horizon_override(config, args)
    if args.collection_policy is not None:
        config = replace(
            config,
            data=replace(config.data, collection_policy=args.collection_policy),
        )
        logger.info(
            "collection policy overridden to %s for this invocation only",
            args.collection_policy,
        )
    if args.params is not None:
        config = replace(
            config, eval=replace(config.eval, params_selection=args.params)
        )
        logger.info(
            "parameter selection overridden to %s for this invocation only",
            args.params,
        )
    return config


# Parent directory under outputs/ for the logs of read-only modes run without a
# run name.
CLI_LOG_PARENT: str = "cli"


def log_run_name(args: argparse.Namespace) -> str:
    """Return the run name this invocation's log is filed under.

    --figures and --publish always log under CLI_LOG_PARENT, one directory per
    mode. The other read-only modes do so only when given no --run-name. Every
    other invocation logs under the run name its artefacts use.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The run name passed to `logs_dir`.
    """
    for mode, given in (("figures", args.figures), ("publish", args.publish is not None)):
        if given:
            return f"{CLI_LOG_PARENT}/{mode}"
    for mode, given in (
        ("aggregate", args.aggregate),
        ("sweep", args.sweep),
        ("fit", args.fit),
        ("displacement_sweep", args.displacement_sweep),
        ("rank_atari_games", args.rank_atari_games),
        ("sweep_atari_checkpoints", args.sweep_atari_checkpoints),
        ("rescore_test_split", args.rescore_test_split),
    ):
        if given:
            if args.run_name is not None:
                return args.run_name
            return f"{CLI_LOG_PARENT}/{mode}"
    return resolve_run_name(args)


def main() -> None:  # pylint: disable=too-many-return-statements,too-many-branches
    """Program entry point. Each read-only mode returns once it has run."""
    args = parse_args()
    run_name = log_run_name(args)
    log_path = None
    if not args.no_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        # One log per arm under --seed's directory, whichever data seeds the
        # pipeline runs. Read-only modes log above the arm level.
        log_path = (
            logs_dir(
                run_name,
                args.seed,
                args.fast,
                args.env,
                arm=(
                    None
                    if args.aggregate
                    or args.sweep
                    or args.fit
                    or args.displacement_sweep
                    or args.intermediate_states
                    or args.rank_atari_games
                    or args.sweep_atari_checkpoints
                    or args.rescore_test_split
                    or args.figures
                    or args.publish is not None
                    else args.arm
                ),
            )
            / f"run_{timestamp}.log"
        )
    configure_logging(
        level=getattr(logging, args.log_level),
        log_file=log_path,
        verbose_deps=args.verbose_deps,
    )
    if args.aggregate:
        # No set_determinism and no config: aggregation trains nothing and
        # touches no PRNG args.seed controls. Its bootstrap reps and seed are
        # passed explicitly by aggregate.py.
        run_aggregation_cli(
            args.run_name,
            args.series_dir,
            args.allow_partial,
            args.arm,
            args.env,
            args.fast,
        )
        return
    if args.sweep:
        run_sweep_cli(
            args.run_name,
            args.series_dir,
            args.arm_run,
            args.env,
            args.fast,
        )
        return
    if args.fit:
        run_fit_cli(
            args.run_name,
            args.series_dir,
            args.env,
            args.fast,
        )
        return
    if args.displacement_sweep:
        # Every PRNG key it uses is derived from an explicit seed.
        run_displacement_sweep_cli(
            args.run_name,
            args.skip_existing,
            args.fast,
        )
        return
    if args.intermediate_states:
        # Every key it uses is derived from the seed of the cell being decoded.
        run_intermediate_states_cli(
            DecodeRequest(
                run_name=args.run_name,
                env_name=args.env if args.env is not None else DEFAULT_ENV_NAME,
                rollout_path=args.rollout_path,
                checkpoint_root=args.checkpoint_root,
                fast=args.fast,
                truth_source=args.truth_source,
                representation=args.representation or DataConfig.representation,
            ),
            SEEDS if args.all_cells else (args.seed,),
            (
                tuple(ObservationMode)
                if args.all_cells
                else (
                    ObservationMode(
                        args.obs_mode
                        if args.obs_mode is not None
                        else OBS_MODE_TOP_DOWN
                    ),
                )
            ),
            # Never narrowed: every evaluation horizon is decoded.
            EVALUATION_HORIZONS,
        )
        return
    if args.rank_atari_games or args.sweep_atari_checkpoints:
        # Both read the archive and write to --out. They train nothing and draw
        # no random number, so they need no config. A sweep given no
        # --positions takes them from config.
        if args.rank_atari_games:
            run_game_ranking_cli(args.position, args.out)
        else:
            run_checkpoint_sweep_cli(args.positions, args.out)
        return
    if args.rescore_test_split:
        # Every stage it runs is a child invocation of this file, which builds
        # its own config, so this branch needs none.
        if args.write_manifest:
            write_manifest(
                tuple(args.runs), args.manifest_source, args.data_seeds, args.manifest
            )
        else:
            run_rescore_cli(
                RescoreRequest(
                    runs=tuple(args.runs),
                    seeds=tuple(args.data_seeds),
                    manifest=args.manifest,
                    regenerate=args.regenerate,
                    dry_run=args.dry_run,
                    allow_ungated=args.allow_ungated,
                )
            )
        return
    if args.figures:
        # Reads artefacts and draws them. It trains nothing, and the only random
        # numbers it draws are the bootstrap's own seeded resamples.
        run_figures_cli(
            FiguresRequest(
                run_filter=args.run_name, dry_run=args.dry_run, development=args.all
            )
        )
        return
    if args.publish is not None:
        run_publish_cli(args.publish)
        return
    config = build_config(args)
    # Before determinism and any stage.
    assert_representation_matches(config)
    # Seeds the global RNG with the base config's model seed, while the runner
    # derives one config per data seed. This is safe only while no stage reads
    # the global RNG: TrainStage derives its keys from its own model seed, and
    # the numpy generators in split.py, aggregate_stats.py and probability_log.py
    # are each explicitly seeded. Re-check this when a stage adds a random call.
    set_determinism(config.model_seed)
    run_pipeline(
        config,
        args.arm,
        stages=args.stages,
        split=SplitName(args.split),
        source_run=args.source_run,
        data_seeds=args.data_seeds,
    )


if __name__ == "__main__":
    main()
