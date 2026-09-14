"""Re-score the existing test-split cells of named runs under the current code.

Per seed, the pass regenerates absent shards, gates them against a manifest of
shard digests, re-evaluates every scored cell, checks the model's primary error
for drift and deletes the shards it regenerated. Each run is then aggregated,
swept and fitted. The drift check does not read the baselines, so nothing
protects an ungated seed's baselines, and a pass over a subset of seeds
aggregates mixed metrics files.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from config import (
    ARM_DIR_PREFIX,
    DATA_DIR_NAME,
    DEFAULT_SEED,
    EVAL_DIR_NAME,
    METRICS_TEMPLATE,
    RESCORE_DRIFT_TOLERANCE,
    SEED_DIR_PREFIX,
    SPLIT_NAME_TEST,
    STAGE_NAME_EVALUATE,
    STAGE_NAME_GENERATE,
    SWEEP_FILENAME,
    TEST_RUN_SUFFIX,
    TRAJECTORY_SHARD_GLOB,
    data_dir,
    eval_dir,
    run_root,
    sentinels_dir,
)
from src.pipeline.metrics_schema import (
    ERROR_METRIC_KEY,
    PER_HORIZON_KEY,
    model_error_key,
)
from src.pipeline.prune import PROTECTED_GLOBS, prune_shards
from src.utils.logging_setup import get_logger
from src.utils.paths import REPO_ROOT, safe_rel
from src.utils.sentinels import SENTINEL_FILENAME, read_stage_metadata

logger = get_logger(__name__)

MANIFEST_VERSION: int = 1
MANIFEST_HASH: str = "sha256"
DIGEST_CHUNK_BYTES: int = 8 * 1024 * 1024

# The directory each pass copies replaced metrics files into, one timestamped
# directory per pass under outputs/. It mirrors run paths, so a glob over
# outputs/*/.../eval/metrics_*.json also counts the backups.
BACKUP_DIR_NAME: str = "rescore_backups"

# Runs one main.py invocation from its arguments and returns the exit code.
CommandRunner = Callable[[Sequence[str]], int]


class GateState(str, Enum):
    """How one seed's shards compare with the manifest."""

    PASSED = "passed"
    FAILED = "failed"
    UNGATED = "ungated"


class RescoreRefused(ValueError):
    """The pass refused before touching anything."""


class RescoreStopped(RuntimeError):
    """The pass stopped partway. A seed stopped before its prune keeps its shards."""


@dataclass(frozen=True)
class GateResult:
    """One seed's gate state, and which shards disagree with the manifest."""

    state: GateState
    mismatched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredCell:
    """One scored test cell: a run, environment, seed, arm and mode."""

    run: str
    env: str
    seed: int
    arm: int
    mode: str

    @property
    def test_run(self) -> str:
        """The run the test pass wrote this cell under."""
        return f"{self.run}{TEST_RUN_SUFFIX}"

    @property
    def metrics_path(self) -> Path:
        """The cell's test metrics file."""
        return eval_dir(
            self.test_run, self.seed, env=self.env, arm=self.arm
        ) / METRICS_TEMPLATE.format(mode=self.mode)

    @property
    def sentinel_path(self) -> Path:
        """The cell's test evaluate sentinel."""
        return (
            sentinels_dir(self.test_run, self.seed, env=self.env, arm=self.arm)
            / f"{STAGE_NAME_EVALUATE}_{self.mode}"
            / SENTINEL_FILENAME
        )


@dataclass(frozen=True)
class SeedPlan:  # pylint: disable=too-many-instance-attributes
    """What the pass does for one run, environment and seed."""

    run: str
    env: str
    seed: int
    cells: tuple[ScoredCell, ...]
    identity: tuple[str, ...]
    stored_modes: tuple[str, ...]
    shards_present: bool
    manifest_entry: dict | None

    @property
    def label(self) -> str:
        """The seed as it is named in the log."""
        return f"{self.run} {self.env} seed {self.seed}"


@dataclass(frozen=True)
class RescoreRequest:
    """The runs and seeds to re-score, and how."""

    runs: tuple[str, ...]
    seeds: tuple[int, ...]
    manifest: Path | None
    regenerate: bool = False
    dry_run: bool = False
    allow_ungated: bool = False


def _outputs_root() -> Path:
    """The directory every run tree hangs off, resolved at call time."""
    return run_root(BACKUP_DIR_NAME, DEFAULT_SEED).parents[2]


def _test_run_dir(run: str) -> Path:
    """The directory the test pass writes one source run under."""
    return _outputs_root() / f"{run}{TEST_RUN_SUFFIX}"


# -- the manifest --------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Return a file's SHA-256, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(DIGEST_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def file_digests(directory: Path, patterns: Sequence[str]) -> dict[str, str]:
    """Return {file name: SHA-256} for every file a pattern matches."""
    return {
        path.name: _sha256(path)
        for pattern in patterns
        for path in sorted(directory.glob(pattern))
    }


def build_manifest(
    runs: Sequence[str], source_root: Path, seeds: Sequence[int]
) -> dict:
    """Digest every shard of the named runs and seeds under an outputs tree.

    A seed whose data directory holds no shards gets no entry, so it gates as
    ungated.

    Args:
        runs: Source run names.
        source_root: The outputs tree holding the original shards.
        seeds: The seeds to include.

    Returns:
        The manifest, keyed run, environment, seed.

    Raises:
        RescoreRefused: If a run has no directory under the source root.
    """
    entries: dict[str, dict] = {}
    for run in runs:
        if not (source_root / run).is_dir():
            raise RescoreRefused(f"no run {run!r} under the manifest source")
        pattern = f"*/{SEED_DIR_PREFIX}*/{DATA_DIR_NAME}"
        for directory in sorted((source_root / run).glob(pattern)):
            seed = int(directory.parent.name[len(SEED_DIR_PREFIX):])
            digests = file_digests(directory, (TRAJECTORY_SHARD_GLOB,))
            if seed in seeds and digests:
                env = directory.parent.parent.name
                entries.setdefault(run, {}).setdefault(env, {})[str(seed)] = {
                    "shards": digests
                }
    return {
        "manifest_version": MANIFEST_VERSION,
        "hash": MANIFEST_HASH,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": entries,
    }


def write_manifest(
    runs: Sequence[str], source_root: Path, seeds: Sequence[int], path: Path
) -> Path:
    """Build the manifest and write it as JSON.

    Args:
        runs: Source run names.
        source_root: The outputs tree holding the original shards.
        seeds: The seeds to include.
        path: Where to write it.

    Returns:
        The path written.
    """
    manifest = build_manifest(runs, source_root, seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    counted = sum(
        len(entry["shards"])
        for run in manifest["runs"].values()
        for env in run.values()
        for entry in env.values()
    )
    logger.info("manifest of %d shards written to %s", counted, safe_rel(path))
    return path


def load_manifest(path: Path) -> dict:
    """Read a manifest, refusing one of another version.

    Raises:
        RescoreRefused: If the file is missing or of another version.
    """
    if not path.is_file():
        raise RescoreRefused(f"no manifest at {safe_rel(path)}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise RescoreRefused(
            f"manifest {safe_rel(path)} is version "
            f"{manifest.get('manifest_version')!r}, expected {MANIFEST_VERSION}"
        )
    return manifest


def manifest_entry(
    manifest: dict | None, run: str, env: str, seed: int
) -> dict | None:
    """Return one seed's manifest entry, or None when it has none."""
    if manifest is None:
        return None
    return manifest["runs"].get(run, {}).get(env, {}).get(str(seed))


def is_ungated(entry: dict | None) -> bool:
    """Whether a manifest entry lists no shards, so nothing can be checked."""
    return not (entry or {}).get("shards")


def gate_seed(entry: dict | None, directory: Path) -> GateResult:
    """Compare one seed's shards with its manifest entry, file by file.

    Args:
        entry: The seed's manifest entry, or None.
        directory: The seed's data directory.

    Returns:
        Ungated when the entry lists no shards, passed when every shard
        matches by name and digest, failed otherwise, naming the files.
    """
    if is_ungated(entry):
        return GateResult(GateState.UNGATED)
    expected = entry["shards"]
    actual = file_digests(directory, (TRAJECTORY_SHARD_GLOB,))
    shared = expected.keys() & actual.keys()
    result = GateResult(
        GateState.FAILED,
        mismatched=tuple(sorted(n for n in shared if expected[n] != actual[n])),
        missing=tuple(sorted(expected.keys() - actual.keys())),
        unexpected=tuple(sorted(actual.keys() - expected.keys())),
    )
    if result.mismatched or result.missing or result.unexpected:
        return result
    return GateResult(GateState.PASSED)


# -- the cells and the commands ------------------------------------------------


def discover_cells(run: str, seeds: Sequence[int]) -> list[ScoredCell]:
    """Return every scored test cell of one run, from its metrics files."""
    prefix, suffix = METRICS_TEMPLATE.split("{mode}")
    pattern = (
        f"*/{SEED_DIR_PREFIX}*/{ARM_DIR_PREFIX}*/{EVAL_DIR_NAME}/{prefix}*{suffix}"
    )
    cells = []
    for path in sorted(_test_run_dir(run).glob(pattern)):
        seed = int(path.parents[2].name[len(SEED_DIR_PREFIX):])
        if seed in seeds:
            cells.append(
                ScoredCell(
                    run=run,
                    env=path.parents[3].name,
                    seed=seed,
                    arm=int(path.parents[1].name[len(ARM_DIR_PREFIX):]),
                    mode=path.name[len(prefix):-len(suffix)],
                )
            )
    return cells


def identity_flags(metadata: dict) -> tuple[str, ...]:
    """Return the generate flags recorded in a dataset's generate sentinel.

    Regeneration from them reproduces a corpus only where generation is
    deterministic, and the manifest gate is what checks it.
    """
    flags = [
        "--env",
        metadata["env_name"],
        "--representation",
        metadata["representation"],
        "--collection-policy",
        metadata["policy"],
        "--slip",
        str(metadata["slip_probability"]),
    ]
    if metadata.get("disable_early_termination"):
        flags.append("--disable-early-termination")
    return tuple(flags)


def regenerate_args(plan: SeedPlan) -> list[str]:
    """Return the main.py arguments that regenerate one seed's shards."""
    args = [
        "--run-name",
        plan.run,
        "--stages",
        STAGE_NAME_GENERATE,
        "--data-seeds",
        str(plan.seed),
        *plan.identity,
    ]
    if len(plan.stored_modes) == 1:
        args += ["--obs-mode", plan.stored_modes[0]]
    return args


def evaluate_args(cell: ScoredCell, identity: Sequence[str]) -> list[str]:
    """Return the main.py arguments that score one test cell."""
    return [
        "--source-run",
        cell.run,
        "--stages",
        STAGE_NAME_EVALUATE,
        "--split",
        SPLIT_NAME_TEST,
        "--arm",
        str(cell.arm),
        "--obs-mode",
        cell.mode,
        "--data-seeds",
        str(cell.seed),
        *identity,
    ]


def run_main(args: Sequence[str]) -> int:
    """Run one main.py invocation in a child process and return its exit code."""
    logger.info("running main.py %s", " ".join(args))
    command = [sys.executable, str(REPO_ROOT / "main.py"), *args]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def primary_drift(before: dict, after: dict) -> tuple[float, float]:
    """Return the largest absolute and relative change in the model's primary error.

    Over every horizon the earlier file scored, on the primary parameter tree
    only. A value that appears or disappears at one of those horizons counts as
    unbounded change.

    Raises:
        KeyError: If the earlier file declares no error metric.
    """
    key = model_error_key(before[ERROR_METRIC_KEY])
    worst_absolute = worst_relative = 0.0
    for horizon, block in before[PER_HORIZON_KEY].items():
        old = block.get(key)
        new = (after[PER_HORIZON_KEY].get(horizon) or {}).get(key)
        if old is None and new is None:
            continue
        if old is None or new is None:
            return math.inf, math.inf
        change = abs(new - old)
        worst_absolute = max(worst_absolute, change)
        worst_relative = max(worst_relative, change / abs(old) if old else math.inf)
    return worst_absolute, worst_relative


# -- the pass ------------------------------------------------------------------


class RescorePass:
    """Plan and run one re-score over the named runs and seeds."""

    def __init__(
        self, request: RescoreRequest, runner: CommandRunner = run_main
    ) -> None:
        """Hold the request and the command runner."""
        self.request = request
        self.runner = runner
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.backup_root = _outputs_root() / BACKUP_DIR_NAME / stamp
        self.drift: dict[str, tuple[float, float]] = {}
        self.pruned: dict[str, tuple[int, int]] = {}

    def plan(self) -> list[SeedPlan]:
        """Return one plan per run, environment and seed, refusing first.

        Raises:
            RescoreRefused: If the pass has no manifest outside a dry run, a
                run has no scored test cells, a seed has no readable generate
                sentinel or one naming no observation mode, or a seed's shards
                are absent without --regenerate.
        """
        request = self.request
        if request.manifest is None and not request.dry_run:
            raise RescoreRefused("a re-score needs --manifest to gate the shards")
        manifest = None if request.manifest is None else load_manifest(request.manifest)
        plans: list[SeedPlan] = []
        for run in request.runs:
            cells = discover_cells(run, request.seeds)
            if not cells:
                raise RescoreRefused(
                    f"run {run!r} has no scored test cells under "
                    f"{safe_rel(_test_run_dir(run))}"
                )
            for env, seed in sorted({(cell.env, cell.seed) for cell in cells}):
                plans.append(self._seed_plan(run, env, seed, cells, manifest))
        absent = [plan.label for plan in plans if not plan.shards_present]
        if absent and not request.regenerate:
            raise RescoreRefused(
                "shards are absent and --regenerate was not given for: "
                + "; ".join(absent)
            )
        return plans

    @staticmethod
    def _seed_plan(
        run: str, env: str, seed: int, cells: list[ScoredCell], manifest: dict | None
    ) -> SeedPlan:
        """Build one seed's plan from its cells and its generate sentinel.

        A run converted by the offline stage has no generate sentinel and is
        refused.
        """
        metadata = read_stage_metadata(STAGE_NAME_GENERATE, run, seed, env=env) or {}
        try:
            identity = identity_flags(metadata)
        except KeyError as error:
            raise RescoreRefused(
                f"{run} {env} seed {seed} has no generate sentinel field "
                f"{error}, so its dataset cannot be reproduced"
            ) from error
        stored_modes = tuple(metadata.get("observation_modes") or ())
        if not stored_modes:
            raise RescoreRefused(
                f"{run} {env} seed {seed} has no generate sentinel field "
                "'observation_modes', so the mode its dataset was written in "
                "is unknown. --allow-ungated covers a manifest mismatch, not "
                "an unknown observation mode"
            )
        return SeedPlan(
            run=run,
            env=env,
            seed=seed,
            cells=tuple(c for c in cells if (c.env, c.seed) == (env, seed)),
            identity=identity,
            stored_modes=stored_modes,
            shards_present=any(data_dir(run, seed, env=env).glob(TRAJECTORY_SHARD_GLOB)),
            manifest_entry=manifest_entry(manifest, run, env, seed),
        )

    def run(self) -> dict:
        """Plan, then re-score every seed and summarise every complete run.

        Returns:
            The cells re-scored, the seeds skipped as ungated, the test runs
            summarised, the largest drift per test run, and the shards pruned
            per seed, zero for a seed whose shards were found on disk.

        Raises:
            RescoreRefused: From plan().
            RescoreStopped: When a seed fails its gate, a command exits
                non-zero, a frozen artefact changes, a metrics file declares
                no error metric, a cell drifts, or a sentinel path falls
                outside the test run.
        """
        plans = self.plan()
        if self.request.dry_run:
            return self._report_plan(plans)
        skipped: list[str] = []
        rescored = 0
        for plan in plans:
            if is_ungated(plan.manifest_entry) and not self.request.allow_ungated:
                logger.warning(
                    "%s is UNGATED: the manifest has no entry for it. Skipped; "
                    "--allow-ungated scores it unverified",
                    plan.label,
                )
                skipped.append(plan.label)
                continue
            self._rescore_seed(plan)
            rescored += len(plan.cells)
        incomplete = {
            (plan.run, plan.env) for plan in plans if plan.label in skipped
        }
        summarised = []
        for run, env in dict.fromkeys((plan.run, plan.env) for plan in plans):
            if (run, env) in incomplete:
                logger.warning(
                    "%s %s is not aggregated: a seed under this environment "
                    "was not re-scored, and one run must not carry two metric "
                    "sets",
                    run, env,
                )
                continue
            cells = [c for p in plans if (p.run, p.env) == (run, env) for c in p.cells]
            self._summarise(run, env, cells)
            summarised.append(f"{run}{TEST_RUN_SUFFIX}")
        return {
            "cells": rescored,
            "ungated_skipped": skipped,
            "summarised": summarised,
            "drift": dict(self.drift),
            "pruned": dict(self.pruned),
        }

    @staticmethod
    def _report_plan(plans: list[SeedPlan]) -> dict:
        """Log what a pass would do, touching nothing."""
        sentinels = [cell.sentinel_path for plan in plans for cell in plan.cells]
        for plan in plans:
            logger.info(
                "DRY RUN %s: %d cells, shards %s, %s",
                plan.label,
                len(plan.cells),
                "present" if plan.shards_present else "to regenerate",
                GateState.UNGATED.value if is_ungated(plan.manifest_entry) else "gated",
            )
            for cell in plan.cells:
                logger.info("DRY RUN   would delete %s", safe_rel(cell.sentinel_path))
        runs = sorted({plan.run for plan in plans})
        logger.info(
            "DRY RUN: %d cells across %d runs, nothing touched",
            len(sentinels),
            len(runs),
        )
        return {
            "cells": len(sentinels),
            "runs": runs,
            "sentinels": [safe_rel(path) for path in sentinels],
        }

    def _rescore_seed(self, plan: SeedPlan) -> None:
        """Regenerate if absent, gate, score every cell, check drift, prune.

        The prune removes only shards this pass regenerated. A seed whose
        shards were found on disk keeps them and records zero pruned.
        """
        directory = data_dir(plan.run, plan.seed, env=plan.env)
        if not plan.shards_present:
            protected = file_digests(directory, PROTECTED_GLOBS)
            self._require(
                self.runner(regenerate_args(plan)) == 0,
                f"{plan.label}: regeneration failed",
            )
            self._require(
                file_digests(directory, PROTECTED_GLOBS) == protected,
                f"{plan.label}: a frozen window, split or statistics file "
                "changed during regeneration",
            )
        gate = gate_seed(plan.manifest_entry, directory)
        logger.info("%s gate %s", plan.label, gate.state.value)
        self._require(
            gate.state is not GateState.FAILED,
            f"{plan.label} FAILED its gate: mismatched {list(gate.mismatched)}, "
            f"missing {list(gate.missing)}, unexpected {list(gate.unexpected)}. "
            "Restore those shards from the backup rather than scoring them",
        )
        for cell in plan.cells:
            self._backup(cell)
            if cell.sentinel_path.exists():
                self._check_scope(cell)
                cell.sentinel_path.unlink()
            self._require(
                self.runner(evaluate_args(cell, plan.identity)) == 0,
                f"{plan.label} arm {cell.arm} {cell.mode}: evaluate failed. "
                "Shards kept for a retry",
            )
        for cell in plan.cells:
            self._check_drift(cell)
        if plan.shards_present:
            count = size = 0
            logger.info(
                "%s: re-scored %d cells; shards were found rather than "
                "regenerated and are left in place",
                plan.label, len(plan.cells),
            )
        else:
            count, size = prune_shards(directory, execute=True)
            logger.info(
                "%s: re-scored %d cells, pruned %d shards (%d bytes)",
                plan.label, len(plan.cells), count, size,
            )
        self.pruned[plan.label] = (count, size)

    def _backup_path(self, cell: ScoredCell) -> Path:
        """Where a cell's replaced metrics file is kept."""
        return self.backup_root / cell.metrics_path.relative_to(_outputs_root())

    def _backup(self, cell: ScoredCell) -> None:
        """Copy a cell's metrics file aside before it is replaced."""
        target = self._backup_path(cell)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cell.metrics_path, target)

    def _check_drift(self, cell: ScoredCell) -> None:
        """Stop the pass if a cell's primary error moved beyond RESCORE_DRIFT_TOLERANCE.

        The replaced metrics file is left in place, with the earlier one in the
        backup.
        """
        backup = self._backup_path(cell)
        before = json.loads(backup.read_text(encoding="utf-8"))
        after = json.loads(cell.metrics_path.read_text(encoding="utf-8"))
        for path, payload in ((backup, before), (cell.metrics_path, after)):
            self._require(
                ERROR_METRIC_KEY in payload,
                f"{safe_rel(path)} declares no {ERROR_METRIC_KEY!r}, so the "
                "drift guard cannot tell which series is the primary error",
            )
        absolute, relative = primary_drift(before, after)
        worst = self.drift.get(cell.test_run, (0.0, 0.0))
        self.drift[cell.test_run] = (max(worst[0], absolute), max(worst[1], relative))
        self._require(
            relative <= RESCORE_DRIFT_TOLERANCE,
            f"{cell.test_run} seed {cell.seed} arm {cell.arm} {cell.mode}: the "
            f"primary error moved by {relative:.3e} relative ({absolute:.3e} "
            f"absolute). The earlier file is at {safe_rel(backup)}",
        )

    @staticmethod
    def _check_scope(cell: ScoredCell) -> None:
        """Refuse to delete anything but a test run's evaluate sentinel."""
        path = cell.sentinel_path
        if not (
            path.is_relative_to(_test_run_dir(cell.run))
            and path.parent.name.startswith(f"{STAGE_NAME_EVALUATE}_")
            and path.name == SENTINEL_FILENAME
        ):
            raise RescoreStopped(f"refusing to delete {safe_rel(path)}")

    def _summarise(self, run: str, env: str, cells: list[ScoredCell]) -> None:
        """Aggregate every scored arm, then sweep and fit the test run."""
        test_run = f"{run}{TEST_RUN_SUFFIX}"
        commands = [
            ["--aggregate", "--run-name", test_run, "--arm", str(arm), "--env", env]
            for arm in sorted({cell.arm for cell in cells})
        ]
        commands.append(
            ["--sweep", "--run-name", test_run, "--env", env,
             *self._arm_runs(test_run, env)]
        )
        commands.append(["--fit", "--run-name", test_run, "--env", env])
        for args in commands:
            self._require(self.runner(args) == 0, f"{test_run}: {args[0]} failed")

    def _arm_runs(self, test_run: str, env: str) -> list[str]:
        """Return the --arm-run flags the test run's previous sweep used."""
        path = run_root(test_run, DEFAULT_SEED, env=env).parent / SWEEP_FILENAME
        if not path.is_file():
            return []
        rescored = {f"{run}{TEST_RUN_SUFFIX}" for run in self.request.runs}
        arm_runs = json.loads(path.read_text(encoding="utf-8")).get("arm_runs", {})
        flags = []
        for arm, arm_run in sorted(arm_runs.items()):
            if arm_run == test_run:
                continue
            if arm_run not in rescored:
                logger.warning(
                    "%s reads arm %s from %s, which this pass did not re-score",
                    test_run, arm, arm_run,
                )
            flags += ["--arm-run", f"{arm}={arm_run}"]
        return flags

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        """Stop the pass with a message unless the condition holds."""
        if not condition:
            logger.error("%s", message)
            raise RescoreStopped(message)


def run_rescore_cli(
    request: RescoreRequest, runner: CommandRunner = run_main
) -> dict:
    """Run one re-score and log its summary.

    Returns:
        The pass's summary.
    """
    summary = RescorePass(request, runner).run()
    compact = {key: value for key, value in summary.items() if key != "sentinels"}
    logger.info("re-score summary: %s", json.dumps(compact, default=str))
    return summary
