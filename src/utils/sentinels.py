"""Sentinel-file stage skipping.

Each coarse pipeline stage writes a sentinel at
``outputs/[fast/]<run_name>/<env>/seed<seed>/[arm<n>/]sentinels/<stage>/done.json``
on completion. On a rerun matching that scoping, a stage whose sentinel exists is
skipped, and deleting a sentinel forces that stage to run again.

Every scoping is structural, never a naming convention. Without the seed, a
second invocation under one run name skips every stage and returns the first
seed's results under the second's label.

Resuming a half-finished training run is the trainer's job.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import (
    MANIFEST_STATE_DONE,
    MANIFEST_STATE_MISSING,
    MANIFEST_STATE_STALE,
    sentinels_dir,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

SENTINEL_FILENAME: str = "done.json"

# One over pylint's default. Every argument is a scoping, and collapsing them
# into an object would hide them at the call sites.
# pylint: disable=too-many-arguments,too-many-positional-arguments


def _sentinel_path(
    stage: str,
    run_name: str,
    seed: int,
    fast: bool = False,
    env: str | None = None,
    arm: int | None = None,
) -> Path:
    """Return the sentinel path for a stage, scoped to run_name, seed and fast.

    Scoping happens here and nowhere else, so a layout change stays confined.

    Args:
        stage: Sentinel subdirectory. Callers pass `Stage.sentinel_key`, which
            a stage may widen beyond its own `name`.
        run_name: Run name without a seed suffix.
        seed: The seed this sentinel is scoped to.
        fast: Whether this is a prototype run, so the sentinel sits beside the
            artefacts it describes.
        env: Registered environment name, so a sentinel lives beside its own
            dataset.
        arm: One of config.ARMS, or None for a stage every arm shares. The only
            scoping that is legitimately None. A blanket arm level would have
            each arm rebuild the dataset they share.

    Returns:
        The sentinel file path for this stage, run, environment, seed and arm.
    """
    return (
        sentinels_dir(run_name, seed, fast, env, arm=arm) / stage / SENTINEL_FILENAME
    )


def _first_mismatch(expected: dict, recorded: dict) -> tuple[str, object, object] | None:
    """Return the first expected field that the sentinel does not match.

    Args:
        expected: Fields this invocation requires the sentinel to carry.
        recorded: The `metadata` block read back off the sentinel.

    Returns:
        (field, recorded_value, current_value) for the first disagreement, or
        None when every expected field is present and equal. An absent field
        counts as a mismatch; older sentinels carry empty metadata.
    """
    for field, wanted in expected.items():
        if field not in recorded:
            return (field, None, wanted)
        if recorded[field] != wanted:
            return (field, recorded[field], wanted)
    return None


def is_stage_done(
    stage: str,
    run_name: str,
    seed: int,
    fast: bool = False,
    expected: dict | None = None,
    env: str | None = None,
    arm: int | None = None,
) -> bool:
    """Return whether a stage completed AND matches the identity it must have.

    Identity and integrity, not file existence. Existence alone answers whether
    something finished here, which is not the question.

    Integrity travels in `expected` alongside identity, so there is one
    comparison and one place it can be wrong.

    Args:
        stage: Sentinel subdirectory.
        run_name: Required, never defaulted, so two runs cannot be confused.
        seed: Required for the same reason. One run name is reused across
            invocations.
        fast: Whether this is a prototype run, so a smoke test cannot make the
            real run skip every stage.
        expected: Fields the sentinel must carry, compared for equality. None
            restores pure existence checking.
        env: Registered environment name, scoping the sentinel path.
        arm: One of config.ARMS, or None for a stage every arm shares.

    Returns:
        True only if the sentinel exists and every field in `expected` matches
        its recorded metadata. Exact match, no tolerance. Nothing legitimately
        rewrites a generated shard.
    """
    return (
        stage_state(stage, run_name, seed, fast, expected, env, arm)
        == MANIFEST_STATE_DONE
    )


def stage_state(
    stage: str,
    run_name: str,
    seed: int,
    fast: bool = False,
    expected: dict | None = None,
    env: str | None = None,
    arm: int | None = None,
) -> str:
    """Return whether a stage is done, stale or missing.

    The one place the sentinel comparison happens. `is_stage_done` narrows it
    to a bool and the manifest reports it verbatim, so the two cannot disagree.

    Args:
        stage: Stage name.
        run_name: The run this sentinel is scoped to.
        seed: The artefact seed this sentinel is scoped to.
        fast: Whether this is a prototype run.
        expected: Identity and integrity fields the sentinel must carry. None
            reduces this to an existence check.
        env: Registered environment name.
        arm: One of config.ARMS, or None for a stage every arm shares.

    Returns:
        MANIFEST_STATE_MISSING when no sentinel exists, MANIFEST_STATE_STALE
        when one exists but does not match `expected`, otherwise
        MANIFEST_STATE_DONE.
    """
    path = _sentinel_path(stage, run_name, seed, fast, env, arm)
    if not path.exists():
        return MANIFEST_STATE_MISSING
    if not expected:
        return MANIFEST_STATE_DONE
    try:
        recorded = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
    except (json.JSONDecodeError, OSError) as error:
        logger.warning(
            "stage %s sentinel at %s is unreadable (%s) -- treating as stale",
            stage,
            safe_rel(path),
            error,
        )
        return MANIFEST_STATE_STALE
    mismatch = _first_mismatch(expected, recorded)
    if mismatch is None:
        return MANIFEST_STATE_DONE
    field, recorded_value, current_value = mismatch
    logger.info(
        "stage %s is stale: %s recorded %r, current %r",
        stage,
        field,
        recorded_value,
        current_value,
    )
    return MANIFEST_STATE_STALE


def mark_stage_done(
    stage: str,
    run_name: str,
    seed: int,
    fast: bool = False,
    metadata: dict | None = None,
    env: str | None = None,
    arm: int | None = None,
) -> Path:
    """Write a stage's sentinel file, scoped as `_sentinel_path` describes.

    Args:
        stage: Stage name.
        run_name: The experiment run_name this sentinel is scoped to.
        seed: The seed this sentinel is scoped to.
        fast: Whether this is a prototype run, so the sentinel lands beside
            the artefacts it describes.
        metadata: JSON-serialisable identity and integrity fields. Whatever is
            passed here is what the stage is later held to, and anything
            omitted can change without forcing a rerun.
        env: Registered environment name, scoping the sentinel path.
        arm: One of config.ARMS, or None for a stage every arm shares. It must
            match what `is_stage_done` is later called with, or the sentinel is
            invisible and the stage silently re-runs.

    Returns:
        The path of the written sentinel file.
    """
    path = _sentinel_path(stage, run_name, seed, fast, env, arm)
    ensure_dir(path.parent)
    payload = {
        "stage": stage,
        "run_name": run_name,
        "seed": seed,
        "fast": fast,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # safe_rel(), not rel(). The sentinel is already written, so a
    # logging-only path computation must not raise.
    logger.info(
        "stage %s (run_name=%s seed=%s) marked done -> %s",
        stage,
        run_name,
        seed,
        safe_rel(path),
    )
    return path
