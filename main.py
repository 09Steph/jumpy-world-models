"""Thin orchestrator for the jumpy world model pipeline.

Responsibilities are limited to argument parsing, configuration construction,
logger setup, determinism, and calling pipeline stages in order. No experiment
logic lives here. That belongs in the stage classes under src/pipeline/.

Usage:
    python main.py --seed 42 --run-name e1_base
    python main.py --fast
    python main.py --aggregate --run-name e1_base
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

from config import (
    DEFAULT_SEED,
    SEEDS,
    ExperimentConfig,
    apply_fast_mode,
    default_run_name,
    logs_dir,
)
from src.pipeline.aggregate import run_aggregation_cli
from src.pipeline.base import (
    STAGE_FAILED,
    STAGE_OK,
    Stage,
    log_run_summary,
)
from src.utils.determinism import set_determinism
from src.utils.logging_setup import (
    RUN_START_BANNER,
    configure_logging,
    get_logger,
)

logger = get_logger(__name__)

# Number of pipeline stages run_pipeline executes. Reported in the run banner.
STAGE_COUNT_STAGE1: int = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Jumpy world model pipeline")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        choices=SEEDS,
        help="Random seed; one of the five fixed reporting seeds.",
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
        "--aggregate",
        action="store_true",
        help="Aggregate an existing seed series and exit. Usually automatic.",
    )
    # The aggregate is written inside the given directory.
    parser.add_argument(
        "--series-dir",
        type=Path,
        default=None,
        help="Directory of seed run directories to aggregate; infers --run-name.",
    )
    # Waives the all-seeds-present requirement only. A truncated or
    # config-mismatched seed is still refused, and the auto-trigger cannot
    # reach this flag.
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Aggregate fewer than all of config.SEEDS, recording the real count.",
    )
    args = parser.parse_args()
    _validate_aggregate_args(parser, args)
    return args


def _validate_aggregate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject aggregation flag combinations that cannot mean anything.

    Exits via parser.error when --series-dir or --allow-partial is passed
    without --aggregate, or when --aggregate has neither --run-name nor
    --series-dir to resolve a run name from.

    Args:
        parser: The parser, used to emit a usage error and exit.
        args: The parsed arguments.
    """
    if not args.aggregate:
        if args.series_dir is not None or args.allow_partial:
            parser.error(
                "--series-dir and --allow-partial only apply to --aggregate"
            )
        return
    if args.run_name is None and args.series_dir is None:
        parser.error(
            "--aggregate needs --run-name, or --series-dir to infer it from. "
            "outputs/ holds many unrelated runs, so a run name cannot be "
            "inferred without one of them."
        )


def resolve_run_name(args: argparse.Namespace) -> str:
    """Return the base run name every artefact for this invocation keys on.

    Args:
        args: Parsed command-line arguments.

    Returns:
        args.run_name when given, otherwise a timestamped default. Carries
        neither the fast marker nor the seed: both are directory levels.
    """
    return args.run_name if args.run_name is not None else default_run_name()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Construct the experiment configuration from parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The composed experiment configuration, scaled down via
        config.apply_fast_mode when args.fast is set.
    """
    config = ExperimentConfig(seed=args.seed, run_name=resolve_run_name(args))
    if args.fast:
        config = apply_fast_mode(config)
        logger.warning(
            "FAST MODE active (run_name=%s), results not for reporting",
            config.run_name,
        )
    return config


def run_pipeline(config: ExperimentConfig) -> None:
    """Execute the pipeline stages in order, then log a run summary.

    The stage list is currently empty, so this logs the run banner, states that
    no stages ran, and logs the summary.

    Args:
        config: The composed experiment configuration.
    """
    logger.info(RUN_START_BANNER, config.run_name, config.seed, STAGE_COUNT_STAGE1)
    started = time.perf_counter()
    stages: list[Stage] = []
    outcome = STAGE_FAILED
    try:
        # Logged at INFO so a run that exits zero is not read as a run that
        # did something.
        logger.info(
            "no pipeline stages are wired: this run produced no artefacts "
            "and supports no claim"
        )
        outcome = STAGE_OK
    finally:
        log_run_summary(
            stages, config.run_name, time.perf_counter() - started, outcome
        )


def main() -> None:
    """Program entry point."""
    args = parse_args()
    # Same helper build_config() uses, so the log file lands under the run_name
    # the other artefacts use. On the aggregation path the name is taken
    # directly, since an aggregate spans every seed.
    run_name = resolve_run_name(args)
    if args.aggregate:
        run_name = args.run_name if args.run_name is not None else "aggregate"
    log_path = None
    if not args.no_log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_path = logs_dir(run_name, args.seed, args.fast) / f"run_{timestamp}.log"
    configure_logging(
        level=getattr(logging, args.log_level),
        log_file=log_path,
        verbose_deps=args.verbose_deps,
    )
    if args.aggregate:
        # No set_determinism and no config: aggregation trains nothing and
        # touches no PRNG args.seed controls. Its bootstrap reps and seed are
        # passed explicitly by aggregate.py.
        run_aggregation_cli(args.run_name, args.series_dir, args.allow_partial)
        return
    set_determinism(args.seed)
    config = build_config(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
