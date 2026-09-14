"""Resolve an offline corpus by name.

Imports no environment library, unlike `trajectory_source`.
"""

from __future__ import annotations

from config import (
    ATARI_CHECKPOINT_POSITIONS,
    ATARI_EPISODES_PER_CELL,
    ATARI_LONG_HORIZON,
    ATARI_LONG_HORIZON_EPISODES_PER_CELL,
    ATARI_LONG_HORIZON_GAME,
    ATARI_LONG_HORIZON_MIN_EPISODE_FRAMES,
    ATARI_MAX_EPISODE_STEPS,
    ATARI_RUN_INDEX,
    ATARI_SHARD_COMPRESSION,
    ATARI_SHARD_SIZE,
    ATARI_WINDOW_OFFSET_MODE,
    KATAKOMBA_MAX_EPISODE_STEPS,
    KATAKOMBA_NUM_EPISODES,
    KATAKOMBA_SHARD_COMPRESSION,
    KATAKOMBA_SHARD_SIZE,
    KATAKOMBA_WINDOW_OFFSET_MODE,
    ATARI_LADDER_POSITIONS_D,
    ATARI_LADDER_POSITIONS_E,
    OFFLINE_SOURCE_ATARI,
    OFFLINE_SOURCE_ATARI_LONG,
    OFFLINE_SOURCE_ATARI_P24,
    OFFLINE_SOURCE_ATARI_P49,
    OFFLINE_SOURCE_KATAKOMBA,
    OFFLINE_SOURCES,
    atari_games,
    atari_root,
    katakomba_root,
)
from src.data.atari_source import (
    MIN_EPISODE_FRAMES as ATARI_MIN_EPISODE_FRAMES,
    AtariTrajectorySource,
)
from src.data.katakomba_source import (
    MIN_EPISODE_FRAMES as KATAKOMBA_MIN_EPISODE_FRAMES,
    KatakombaTrajectorySource,
)

# Either offline source. They share no base class, only the TrajectorySource
# Protocol.
OfflineSource = AtariTrajectorySource | KatakombaTrajectorySource


def _katakomba_source() -> tuple[KatakombaTrajectorySource, dict]:
    """Build the Katakomba source and the settings its dataset is keyed on.

    The settings carry the episode floor, so a changed floor does not match an
    existing sentinel.
    """
    source = KatakombaTrajectorySource(
        katakomba_root(),
        num_steps=KATAKOMBA_MAX_EPISODE_STEPS,
        offset_mode=KATAKOMBA_WINDOW_OFFSET_MODE,
    )
    return source, {
        "num_episodes": KATAKOMBA_NUM_EPISODES,
        "max_episode_steps": KATAKOMBA_MAX_EPISODE_STEPS,
        "min_episode_frames": KATAKOMBA_MIN_EPISODE_FRAMES,
        "window_offset_mode": KATAKOMBA_WINDOW_OFFSET_MODE,
        "shard_size": KATAKOMBA_SHARD_SIZE,
        "shard_compression": KATAKOMBA_SHARD_COMPRESSION,
    }


def _atari_source_at(
    positions: tuple[int, ...],
) -> tuple[AtariTrajectorySource, dict]:
    """Build the Atari source at given archive positions, and its settings.

    The settings carry the games, positions and episode floor, so a different
    selection does not match an existing sentinel and reuse its dataset.

    Args:
        positions: Archive positions this conversion reads.

    Returns:
        The source and the settings dict its dataset is keyed on.
    """
    games = atari_games()
    source = AtariTrajectorySource(
        atari_root(),
        games=games,
        positions=positions,
        num_steps=ATARI_MAX_EPISODE_STEPS,
        offset_mode=ATARI_WINDOW_OFFSET_MODE,
        episodes_per_cell=ATARI_EPISODES_PER_CELL,
        min_episode_frames=ATARI_MIN_EPISODE_FRAMES,
        run=ATARI_RUN_INDEX,
    )
    cells = len(games) * len(positions)
    return source, {
        "num_episodes": cells * ATARI_EPISODES_PER_CELL,
        "max_episode_steps": ATARI_MAX_EPISODE_STEPS,
        "min_episode_frames": ATARI_MIN_EPISODE_FRAMES,
        "window_offset_mode": ATARI_WINDOW_OFFSET_MODE,
        "shard_size": ATARI_SHARD_SIZE,
        "shard_compression": ATARI_SHARD_COMPRESSION,
        "games": list(games),
        "checkpoint_positions": list(positions),
        "run_index": ATARI_RUN_INDEX,
        "episodes_per_cell": ATARI_EPISODES_PER_CELL,
    }


def _atari_source() -> tuple[AtariTrajectorySource, dict]:
    """Build the base Atari source at ATARI_CHECKPOINT_POSITIONS."""
    return _atari_source_at(ATARI_CHECKPOINT_POSITIONS)


def _atari_ladder_d_source() -> tuple[AtariTrajectorySource, dict]:
    """Build the Atari source at ATARI_LADDER_POSITIONS_D."""
    return _atari_source_at(ATARI_LADDER_POSITIONS_D)


def _atari_ladder_e_source() -> tuple[AtariTrajectorySource, dict]:
    """Build the Atari source at ATARI_LADDER_POSITIONS_E."""
    return _atari_source_at(ATARI_LADDER_POSITIONS_E)


def _atari_long_source() -> tuple[AtariTrajectorySource, dict]:
    """Build the long-horizon Atari source and its dataset settings.

    Reads one game with a window reaching the longest long-horizon evaluation
    horizon. As in `_atari_source_at`, the games, positions and episode floor
    are in the settings, so a changed selection does not reuse a dataset.
    """
    games = (ATARI_LONG_HORIZON_GAME,)
    source = AtariTrajectorySource(
        atari_root(),
        games=games,
        positions=ATARI_CHECKPOINT_POSITIONS,
        num_steps=ATARI_LONG_HORIZON,
        offset_mode=ATARI_WINDOW_OFFSET_MODE,
        episodes_per_cell=ATARI_LONG_HORIZON_EPISODES_PER_CELL,
        min_episode_frames=ATARI_LONG_HORIZON_MIN_EPISODE_FRAMES,
        run=ATARI_RUN_INDEX,
    )
    cells = len(games) * len(ATARI_CHECKPOINT_POSITIONS)
    return source, {
        "num_episodes": cells * ATARI_LONG_HORIZON_EPISODES_PER_CELL,
        "max_episode_steps": ATARI_LONG_HORIZON,
        "min_episode_frames": ATARI_LONG_HORIZON_MIN_EPISODE_FRAMES,
        "window_offset_mode": ATARI_WINDOW_OFFSET_MODE,
        "shard_size": ATARI_SHARD_SIZE,
        "shard_compression": ATARI_SHARD_COMPRESSION,
        "games": list(games),
        "checkpoint_positions": list(ATARI_CHECKPOINT_POSITIONS),
        "run_index": ATARI_RUN_INDEX,
        "episodes_per_cell": ATARI_LONG_HORIZON_EPISODES_PER_CELL,
    }


# One entry per offline corpus. The only place a name is mapped to a corpus.
OFFLINE_SOURCE_FACTORIES: dict = {
    OFFLINE_SOURCE_KATAKOMBA: _katakomba_source,
    OFFLINE_SOURCE_ATARI: _atari_source,
    OFFLINE_SOURCE_ATARI_LONG: _atari_long_source,
    OFFLINE_SOURCE_ATARI_P24: _atari_ladder_d_source,
    OFFLINE_SOURCE_ATARI_P49: _atari_ladder_e_source,
}


def build_offline_source(name: str) -> tuple[OfflineSource, dict]:
    """Build one offline source by name, with the settings it is keyed on.

    Args:
        name: One of OFFLINE_SOURCES.

    Returns:
        The source, and the settings a dataset built from it depends on.

    Raises:
        ValueError: If the name has no factory, or has one but is absent from
            OFFLINE_SOURCES.
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
