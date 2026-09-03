"""Resolve an offline corpus by name.

Separate from `trajectory_source`, which imports the NAVIX environment library
to roll episodes live. The offline path reads recorded files and must not need
an environment installed to convert them.
"""

from __future__ import annotations

from config import (
    KATAKOMBA_MAX_EPISODE_STEPS,
    KATAKOMBA_NUM_EPISODES,
    KATAKOMBA_SHARD_COMPRESSION,
    KATAKOMBA_SHARD_SIZE,
    KATAKOMBA_WINDOW_OFFSET_MODE,
    OFFLINE_SOURCE_KATAKOMBA,
    OFFLINE_SOURCES,
    katakomba_root,
)
from src.data.katakomba_source import KatakombaTrajectorySource


def _katakomba_source() -> tuple[KatakombaTrajectorySource, dict]:
    """Build the Katakomba source and the settings its dataset is keyed on.

    HORIZON_MAX is not among them and also decides which episodes the
    conversion admits, through `katakomba_source.MIN_EPISODE_FRAMES`. A rerun
    at a different value matches the existing sentinel and skips generation.
    """
    source = KatakombaTrajectorySource(
        katakomba_root(),
        num_steps=KATAKOMBA_MAX_EPISODE_STEPS,
        offset_mode=KATAKOMBA_WINDOW_OFFSET_MODE,
    )
    return source, {
        "num_episodes": KATAKOMBA_NUM_EPISODES,
        "max_episode_steps": KATAKOMBA_MAX_EPISODE_STEPS,
        "window_offset_mode": KATAKOMBA_WINDOW_OFFSET_MODE,
        "shard_size": KATAKOMBA_SHARD_SIZE,
        "shard_compression": KATAKOMBA_SHARD_COMPRESSION,
    }


# One entry per offline corpus. The only place a name is mapped to a corpus.
OFFLINE_SOURCE_FACTORIES: dict = {
    OFFLINE_SOURCE_KATAKOMBA: _katakomba_source,
}


def build_offline_source(name: str) -> tuple[KatakombaTrajectorySource, dict]:
    """Build one offline source by name, with the settings it is keyed on.

    Args:
        name: One of OFFLINE_SOURCES.

    Returns:
        The source, and the settings a dataset built from it depends on.

    Raises:
        ValueError: If the name has no factory, naming the ones that do, or if
            it has a factory but is absent from OFFLINE_SOURCES. A fallback
            would convert one corpus into another corpus's directory.
    """
    if name not in OFFLINE_SOURCE_FACTORIES:
        known = ", ".join(sorted(OFFLINE_SOURCE_FACTORIES))
        raise ValueError(
            f"unknown offline source '{name}'. Declared sources are ({known}); "
            "add it to OFFLINE_SOURCE_FACTORIES and OFFLINE_SOURCES rather "
            "than defaulting to one of them."
        )
    if name not in OFFLINE_SOURCES:
        raise ValueError(
            f"offline source '{name}' has a factory but is absent from "
            "OFFLINE_SOURCES, so the two records disagree about what exists."
        )
    return OFFLINE_SOURCE_FACTORIES[name]()
