"""Repository-anchored path helpers.

Every output path is anchored to REPO_ROOT and logged relative to it, so logs
and manifests stay portable across machines.
"""

from __future__ import annotations

import logging
from pathlib import Path

# This file is src/utils/paths.py, so the root is two parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

_logger = logging.getLogger(__name__)


def rel(path: Path | str) -> str:
    """Return a path relative to REPO_ROOT for reproducible logging.

    Strict. It raises, never falling back to an absolute path. Use it for
    anything citable, and `safe_rel` for informational logging.

    Args:
        path: An absolute or relative filesystem path.

    Returns:
        The path expressed relative to REPO_ROOT, as a string.

    Raises:
        ValueError: If the resolved path does not sit within REPO_ROOT.
    """
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def safe_rel(path: Path | str) -> str:
    """Best-effort variant of rel() for logging call sites that must not raise.

    A path outside REPO_ROOT falls back to the absolute form with a warning,
    so a log line cannot turn a successful operation into a failure.

    Args:
        path: An absolute or relative filesystem path.

    Returns:
        The path relative to REPO_ROOT if possible, otherwise the resolved
        absolute path as a string.
    """
    try:
        return rel(path)
    except ValueError:
        absolute = str(Path(path).resolve())
        _logger.warning("path %s is outside REPO_ROOT -- logging absolute path", absolute)
        return absolute


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory to create.

    Returns:
        The same path, now guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
