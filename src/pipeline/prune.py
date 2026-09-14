"""Delete one seed's trajectory shards, and nothing else.

The directory also holds the frozen evaluation windows, the frozen split and
the trajectory statistics, which every reported cell was scored on and which
must survive. Nothing here checks that the shards are backed up or that
regeneration would reproduce them, and for some corpora it does not.
"""

from __future__ import annotations

from pathlib import Path

from config import TRAJECTORY_SHARD_GLOB
from src.utils.paths import safe_rel

# What a prune must never remove. The names repeat SPLIT_PROVENANCE_FILENAME,
# TRAJECTORY_STATS_FILENAME and WINDOWS_EVAL_TEMPLATE, so a rename there leaves
# every check built on this tuple counting nothing.
PROTECTED_GLOBS: tuple[str, ...] = (
    "windows_eval_*.h5",
    "split.json",
    "trajectory_stats.json",
)

# The only file names a prune may delete.
SHARD_PREFIX: str = "trajectories_"
SHARD_SUFFIX: str = ".h5"


def protected_inventory(directory: Path) -> dict[str, int]:
    """Return how many of each protected artefact kind a directory holds."""
    return {
        pattern: len(list(directory.glob(pattern))) for pattern in PROTECTED_GLOBS
    }


def prune_shards(directory: Path, execute: bool) -> tuple[int, int]:
    """Remove the shards in one data directory, refusing anything else.

    Every path the shard glob matches must carry the shard prefix and suffix,
    checked before any deletion and on a dry run too. Afterwards the count of
    each protected artefact kind must be unchanged. Counts are compared, not
    contents.

    Args:
        directory: One seed's data directory.
        execute: Delete when true, report only when false.

    Returns:
        The shard count and their total size in bytes, measured before any
        deletion.

    Raises:
        ValueError: If a matched path is not a trajectory shard, or if a
            protected artefact count changes across the deletion.
    """
    shards = sorted(directory.glob(TRAJECTORY_SHARD_GLOB))
    total = sum(path.stat().st_size for path in shards)
    for path in shards:
        if not path.name.startswith(SHARD_PREFIX) or path.suffix != SHARD_SUFFIX:
            raise ValueError(
                f"refusing to delete {path.name!r}: the glob matched a file "
                "that is not a trajectory shard, which means the glob is wrong"
            )
    if not execute:
        return len(shards), total
    before = protected_inventory(directory)
    for path in shards:
        path.unlink()
    after = protected_inventory(directory)
    if before != after:
        raise ValueError(
            f"protected artefacts changed across the deletion in "
            f"{safe_rel(directory)}: "
            f"{before} became {after}. The frozen windows and split must "
            "survive a prune."
        )
    return len(shards), total
