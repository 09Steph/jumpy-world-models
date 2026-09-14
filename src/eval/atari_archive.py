"""Measure how much of each Atari game's screen changes, by game and by archive position.

The ranking measures every game at one archive position against the
displacement bar and the episode-length gate. The checkpoint sweep measures
every game holding enough of the swept positions at each of them. Both read
the ordered DQN Replay archive from `atari_root()`, one subdirectory per game,
which is not part of the repository. Neither draws a random number, so the
same shards yield the same JSON.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import closing
from itertools import islice
from pathlib import Path

import numpy as np

from config import (
    ATARI_DISPLACEMENT_BAR_PCT,
    ATARI_EPISODE_LENGTH_GATE,
    ATARI_LENGTH_PERCENTILE,
    ATARI_LONG_HORIZON_MIN_EPISODE_FRAMES,
    ATARI_ROOT_ENV_VAR,
    ATARI_RUN_INDEX,
    ATARI_SHARD_GLOB,
    ATARI_SWEEP_EPISODES,
    ATARI_SWEEP_POSITIONS,
    HORIZON_MAX,
    atari_root,
    atari_shard_path,
)
from src.data.atari_source import (
    CHECKPOINT_KEY,
    OBSERVATIONS_KEY,
    decode_frames,
    iterate_records,
    parse_example,
)
from src.pipeline.displacement import hamming_displacement
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

# Episodes the ranking decodes per game.
EPISODES_PER_GAME: int = 30

# Frames an episode needs to supply one pair at each reported horizon. The
# shard census counts episodes reaching each.
BASE_FLOOR_FRAMES: int = HORIZON_MAX + 1
LONG_FLOOR_FRAMES: int = ATARI_LONG_HORIZON_MIN_EPISODE_FRAMES

# Shard filenames at any position within a game directory, with the trailing
# shard count left open. One position is ATARI_SHARD_GLOB.
ANY_SHARD_GLOB: str = "run_{run}-*-of-*"

# Scale of a percentage, and the places a reported displacement is rounded to.
PERCENT: int = 100
ROUND_DIGITS: int = 4

# Fewest swept positions a game must hold for the sweep to measure it.
MIN_SWEEP_POSITIONS: int = 2

# Artefact names written under the caller's output directory.
RANKING_STEM: str = "atari_game_ranking"
SWEEP_STEM: str = "atari_checkpoint_sweep"

# Text matrix layout.
TEXT_COLUMN_GAP: str = "  "
MISSING_VALUE: str = "-"
VERDICT_LABELS: dict[bool, str] = {True: "PASS", False: "FAIL"}


def shard_path(game_dir: Path, position: int) -> Path | None:
    """Return a game's shard at one position of ATARI_RUN_INDEX, or None.

    Raises:
        ValueError: If more than one shard matches the position.
    """
    return atari_shard_path(game_dir, position, ATARI_RUN_INDEX)


def archive_games(root: Path) -> tuple[str, ...]:
    """Return every non-hidden game directory in the archive, in name order.

    Args:
        root: The archive directory, from `atari_root()`.

    Raises:
        FileNotFoundError: If the directory is absent or no game in it holds a
            shard.
    """
    games: tuple[str, ...] = ()
    if root.is_dir():
        games = tuple(
            sorted(
                path.name
                for path in root.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            )
        )
    pattern = ANY_SHARD_GLOB.format(run=ATARI_RUN_INDEX)
    if not any(any((root / game).glob(pattern)) for game in games):
        reason = (
            "does not exist" if not root.is_dir()
            else "holds no game directory carrying a shard"
        )
        raise FileNotFoundError(
            f"no Atari archive: {safe_rel(root)} {reason}. Set "
            f"{ATARI_ROOT_ENV_VAR} to the directory holding one subdirectory "
            "per game, laid out as "
            f"<Game>/{ATARI_SHARD_GLOB.format(run=ATARI_RUN_INDEX, index=0)}. The "
            "archive is not part of the repository."
        )
    return games


def decode_episode(features: dict[str, list]) -> tuple[np.ndarray, int]:
    """Return one record's frames and the checkpoint index it was recorded at."""
    return decode_frames(features[OBSERVATIONS_KEY]), int(features[CHECKPOINT_KEY][0])


def read_shard(path: Path, limit: int) -> Iterator[tuple[np.ndarray, int]]:
    """Yield the first episodes of a shard in stored order.

    Args:
        path: The shard.
        limit: Episodes to decode before stopping.

    Yields:
        (frames, checkpoint index) per episode, frames shaped
        (steps, height, width, channels).
    """
    with closing(iterate_records(path)) as records:
        for payload in islice(records, limit):
            yield decode_episode(parse_example(payload))


def displacement_pct(frames: np.ndarray, horizon: int) -> float | None:
    """Return the mean percentage of cells differing between frames a horizon apart.

    Averaged over every start step in the episode. A cell differs when any
    channel does.

    Args:
        frames: One episode, shape (steps, height, width, channels).
        horizon: Step gap between the two frames of each pair.

    Returns:
        The percentage, or None if the episode is no longer than the horizon.
    """
    if frames.shape[0] <= horizon:
        return None
    differing = hamming_displacement(frames[:-horizon], frames[horizon:])
    return float(differing[:, -1].mean() * PERCENT)


def length_at_percentile(lengths: Sequence[int]) -> int:
    """Return the episode length at ATARI_LENGTH_PERCENTILE, as an order statistic.

    The rank into the sorted lengths is
    max(0, ATARI_LENGTH_PERCENTILE * n // PERCENT - 1), not an interpolated
    percentile.

    Args:
        lengths: Episode lengths in decision steps.

    Returns:
        The selected length.
    """
    ordered = sorted(lengths)
    rank = max(0, (ATARI_LENGTH_PERCENTILE * len(ordered)) // PERCENT - 1)
    return ordered[rank]


def _scan(shard: Path, episodes: int) -> list[tuple[int, int, float | None]]:
    """Return (checkpoint index, length, displacement) per episode, in stored order.

    Raises:
        ValueError: If the shard holds no episode.
    """
    scanned = [
        (checkpoint, int(frames.shape[0]), displacement_pct(frames, HORIZON_MAX))
        for frames, checkpoint in read_shard(shard, episodes)
    ]
    if not scanned:
        raise ValueError(f"{shard.parent.name}/{shard.name} holds no episode")
    return scanned


def shard_census(shard: Path) -> dict:
    """Return every episode length in one shard, without decoding a frame.

    Covers the whole shard, where `measure_game`'s own length statistics cover
    only the episodes it decodes.

    Args:
        shard: The game's shard at the measured position.

    Returns:
        The episode count, the count reaching each frame floor, the length at
        ATARI_LENGTH_PERCENTILE, the median, the extremes and the frame total.

    Raises:
        ValueError: If the shard holds no episode.
    """
    lengths: list[int] = []
    with closing(iterate_records(shard)) as records:
        for payload in records:
            lengths.append(len(parse_example(payload)[OBSERVATIONS_KEY]))
    if not lengths:
        raise ValueError(f"{shard.parent.name}/{shard.name} holds no episode")
    ordered = sorted(lengths)
    return {
        "episodes": len(ordered),
        "episodes_at_base_floor": sum(1 for n in ordered if n >= BASE_FLOOR_FRAMES),
        "episodes_at_long_floor": sum(1 for n in ordered if n >= LONG_FLOOR_FRAMES),
        "length_p10": length_at_percentile(ordered),
        "length_median": int(statistics.median(ordered)),
        "length_min": ordered[0],
        "length_max": ordered[-1],
        "total_frames": sum(ordered),
    }


def _displacement_fields(values: Sequence[float]) -> dict:
    """Return the scored-episode count, mean displacement and bar outcome.

    `pairs_scored` counts episodes, not pairs, and the mean weights every
    episode equally whatever its length.
    """
    mean = statistics.mean(values) if values else None
    return {
        "pairs_scored": len(values),
        "displacement_pct": None if mean is None else round(mean, ROUND_DIGITS),
        "clears_displacement_bar": mean is not None
        and mean >= ATARI_DISPLACEMENT_BAR_PCT,
    }


def measure_game(shard: Path, episodes: int) -> dict:
    """Measure one game's displacement and episode-length profile in one shard.

    Args:
        shard: The game's shard at the measured position.
        episodes: Episodes to decode.

    Returns:
        The shard name, the decoded episode count, `_displacement_fields` at
        HORIZON_MAX, `length_p10` (the length at ATARI_LENGTH_PERCENTILE) and
        the median length over the decoded episodes, both length-gate
        outcomes, and `shard_census`. `clears_length_gate` reads the decoded
        sample and `clears_length_gate_on_shard` reads the census.
    """
    scanned = _scan(shard, episodes)
    lengths = [length for _, length, _ in scanned]
    values = [value for _, _, value in scanned if value is not None]
    length_p10 = length_at_percentile(lengths)
    census = shard_census(shard)
    return {
        "shard": shard.name,
        "episodes": len(lengths),
        **_displacement_fields(values),
        "length_p10": length_p10,
        "length_median": int(statistics.median(lengths)),
        "clears_length_gate": length_p10 >= ATARI_EPISODE_LENGTH_GATE,
        "clears_length_gate_on_shard": (
            census["length_p10"] >= ATARI_EPISODE_LENGTH_GATE
        ),
        "shard_census": census,
    }


def _ranking_order(row: dict) -> tuple[bool, float, str]:
    """Sort key: highest displacement first, unscored games last, then by name."""
    value = row["displacement_pct"]
    return value is None, -(value or 0.0), row["game"]


def absent_games(root: Path, games: Sequence[str], position: int) -> list[dict]:
    """Return one entry per game holding no shard at one position.

    Args:
        root: Archive directory, from `atari_root()`.
        games: Games considered.
        position: Archive position.

    Returns:
        Entries carrying the game and the positions it does hold.
    """
    entries: list[dict] = []
    for game in games:
        if shard_path(root / game, position) is not None:
            continue
        pattern = ANY_SHARD_GLOB.format(run=ATARI_RUN_INDEX)
        held = sorted(
            int(path.name.split("-")[1])
            for path in (root / game).glob(pattern)
            if path.is_file()
        )
        entries.append({"game": game, "position": position, "positions_present": held})
    return entries


def rank_games(
    root: Path, games: Sequence[str], episodes: int, position: int
) -> list[dict]:
    """Return one ranking row per game holding a shard at one position.

    Displacement is `hamming_displacement` at HORIZON_MAX. Every game with a
    shard is reported, including those failing the bar or the gate. Games
    without one are logged and left out.

    Args:
        root: Archive directory, from `atari_root()`.
        games: Games to measure.
        episodes: Episodes decoded per game.
        position: Archive position measured.

    Returns:
        Rows sorted by displacement, highest first, with unscored games last
        and ties in name order.

    Raises:
        FileNotFoundError: If no game has a shard at the position, naming the
            pattern that matched nothing, before any episode is read.
    """
    shards = {game: shard_path(root / game, position) for game in games}
    present = [game for game, shard in shards.items() if shard is not None]
    for game in games:
        if shards[game] is None:
            logger.info("%s has no shard at position %d, skipped", game, position)
    if not present:
        expected = ATARI_SHARD_GLOB.format(run=ATARI_RUN_INDEX, index=position)
        raise FileNotFoundError(
            f"position {position} is absent under {safe_rel(root)} for every "
            f"one of {len(games)} games: {expected}"
        )
    rows: list[dict] = []
    for index, game in enumerate(present, start=1):
        started = time.perf_counter()
        rows.append({"game": game, **measure_game(shards[game], episodes)})
        logger.info(
            "%d/%d %s: displacement %s per cent, length_p10 %d (%.1f s)",
            index,
            len(present),
            game,
            rows[-1]["displacement_pct"],
            rows[-1]["length_p10"],
            time.perf_counter() - started,
        )
    return sorted(rows, key=_ranking_order)


def _checkpoint_rows(
    game: str, position: int, shard: Path, episodes: int
) -> list[dict]:
    """Return one sweep row per checkpoint index recorded in one shard."""
    lengths: dict[int, list[int]] = defaultdict(list)
    values: dict[int, list[float]] = defaultdict(list)
    for checkpoint, length, value in _scan(shard, episodes):
        lengths[checkpoint].append(length)
        if value is not None:
            values[checkpoint].append(value)
    return [
        {
            "game": game,
            "position": position,
            "checkpoint_idx": checkpoint,
            "shard": shard.name,
            "episodes": len(lengths[checkpoint]),
            **_displacement_fields(values[checkpoint]),
            "length_median": int(statistics.median(lengths[checkpoint])),
        }
        for checkpoint in sorted(lengths)
    ]


def sweep_checkpoints(
    root: Path, games: Sequence[str], positions: Sequence[int], episodes: int
) -> dict:
    """Measure displacement at each archive position, for every game.

    A position absent from a swept game's directory is recorded and skipped. A
    game holding fewer than MIN_SWEEP_POSITIONS of the positions is recorded
    and skipped whole.

    Args:
        root: Archive directory, from `atari_root()`.
        games: Games to sweep.
        positions: Archive positions measured.
        episodes: Episodes decoded per game and position.

    Returns:
        `rows`, one per game, position and checkpoint index; `absent`, the
        positions missing from swept games; and `skipped`, the games not swept
        with the positions they do hold.
    """
    rows: list[dict] = []
    absent: list[dict] = []
    skipped: list[dict] = []
    for game in games:
        present = {
            position: shard
            for position in positions
            if (shard := shard_path(root / game, position)) is not None
        }
        if len(present) < MIN_SWEEP_POSITIONS:
            logger.warning(
                "%s holds %d of the swept positions, skipped", game, len(present)
            )
            skipped.append({"game": game, "positions_present": list(present)})
            continue
        for position in positions:
            if position not in present:
                logger.info("%s has no shard at position %d, skipped", game, position)
                absent.append({"game": game, "position": position})
                continue
            rows.extend(
                _checkpoint_rows(game, position, present[position], episodes)
            )
        logger.info("%s swept at %d positions", game, len(present))
    return {"rows": rows, "absent": absent, "skipped": skipped}


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    """Return fixed-width lines, each column padded to its widest entry."""
    lines = (header, *body)
    widths = [max(len(line[column]) for line in lines) for column in range(len(header))]
    return [
        TEXT_COLUMN_GAP.join(
            cell.ljust(width) for cell, width in zip(line, widths)
        ).rstrip()
        for line in lines
    ]


def _pct(value: float | None) -> str:
    """Return a displacement percentage as printed in the text matrices."""
    return MISSING_VALUE if value is None else f"{value:.{ROUND_DIGITS}f}"


def ranking_text(payload: dict) -> str:
    """Return the ranking as a fixed-width matrix, one line per game.

    The first columns are the decoded sample. The `shard_` columns are over
    every episode in the shard, so `shard_p10` and `shard_gate` are the figures
    a reported selection is read from.
    """
    body = [
        [
            str(rank),
            row["game"],
            _pct(row["displacement_pct"]),
            str(row["length_p10"]),
            str(row["length_median"]),
            VERDICT_LABELS[row["clears_displacement_bar"]],
            VERDICT_LABELS[row["clears_length_gate"]],
            str(row["shard_census"]["episodes"]),
            str(row["shard_census"]["episodes_at_base_floor"]),
            str(row["shard_census"]["episodes_at_long_floor"]),
            str(row["shard_census"]["length_p10"]),
            str(row["shard_census"]["length_median"]),
            VERDICT_LABELS[row["clears_length_gate_on_shard"]],
        ]
        for rank, row in enumerate(payload["games"], start=1)
    ]
    header = [
        "rank", "game", "displacement_pct", "length_p10", "length_median", "bar",
        "gate", "shard_episodes", "shard_at_base", "shard_at_long", "shard_p10",
        "shard_median", "shard_gate",
    ]
    lines = [
        f"# {RANKING_STEM}, archive position {payload['position']}, run "
        f"{payload['run']}, horizon {payload['horizon']}, "
        f"{payload['episodes_per_game']} episodes per game",
        f"# bar {payload['displacement_bar_pct']} per cent; gate "
        f"{payload['episode_length_gate']} steps at percentile "
        f"{payload['length_percentile']}",
        f"# shard_ columns read every episode; floors "
        f"{payload['base_floor_frames']} and {payload['long_floor_frames']} frames",
        *_table(header, body),
        *(
            f"# absent: {entry['game']} holds positions "
            f"{entry['positions_present']}"
            for entry in payload["absent"]
        ),
    ]
    return "\n".join(lines) + "\n"


def sweep_text(payload: dict) -> str:
    """Return the sweep as a fixed-width matrix, one line per game and position."""
    body = [
        [
            row["game"],
            str(row["position"]),
            str(row["checkpoint_idx"]),
            _pct(row["displacement_pct"]),
            VERDICT_LABELS[row["clears_displacement_bar"]],
            str(row["length_median"]),
            str(row["episodes"]),
        ]
        for row in payload["rows"]
    ]
    header = [
        "game", "position", "checkpoint_idx", "displacement_pct", "bar",
        "length_median", "episodes",
    ]
    lines = [
        f"# {SWEEP_STEM}, positions {payload['positions']}, run {payload['run']}, "
        f"horizon {payload['horizon']}, {payload['episodes_per_position']} "
        "episodes per position",
        *_table(header, body),
        *(
            f"# absent: {entry['game']} position {entry['position']}"
            for entry in payload["absent"]
        ),
        *(
            f"# skipped: {entry['game']} holds positions {entry['positions_present']}"
            for entry in payload["skipped"]
        ),
    ]
    return "\n".join(lines) + "\n"


def _write(out_dir: Path, stem: str, payload: dict, text: str) -> Path:
    """Write the JSON artefact and its text matrix, returning the JSON path."""
    target = ensure_dir(out_dir)
    json_path = target / f"{stem}.json"
    text_path = target / f"{stem}.txt"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    logger.info("wrote %s and %s", safe_rel(json_path), safe_rel(text_path))
    return json_path


def run_game_ranking_cli(position: int, out_dir: Path) -> Path:
    """Rank every game in the archive at one position and write the results.

    Args:
        position: Archive position to measure.
        out_dir: Directory the JSON and the text matrix are written to.

    Returns:
        The JSON path written.
    """
    root = atari_root()
    games = archive_games(root)
    logger.info(
        "ranking %d games at archive position %d, %d episodes each",
        len(games),
        position,
        EPISODES_PER_GAME,
    )
    rows = rank_games(root, games, EPISODES_PER_GAME, position)
    payload = {
        "measurement": RANKING_STEM,
        "position": position,
        "run": ATARI_RUN_INDEX,
        "horizon": HORIZON_MAX,
        "episodes_per_game": EPISODES_PER_GAME,
        "base_floor_frames": BASE_FLOOR_FRAMES,
        "long_floor_frames": LONG_FLOOR_FRAMES,
        "displacement_bar_pct": ATARI_DISPLACEMENT_BAR_PCT,
        "episode_length_gate": ATARI_EPISODE_LENGTH_GATE,
        "length_percentile": ATARI_LENGTH_PERCENTILE,
        "absent": absent_games(root, games, position),
        "qualifying": [
            row["game"]
            for row in rows
            if row["clears_displacement_bar"] and row["clears_length_gate"]
        ],
        "games": rows,
    }
    stem = f"{RANKING_STEM}_position{position}"
    return _write(out_dir, stem, payload, ranking_text(payload))


def run_checkpoint_sweep_cli(positions: Sequence[int] | None, out_dir: Path) -> Path:
    """Sweep every game in the archive across archive positions and write the results.

    Args:
        positions: Positions to measure. None takes ATARI_SWEEP_POSITIONS.
        out_dir: Directory the JSON and the text matrix are written to.

    Returns:
        The JSON path written.
    """
    swept = tuple(ATARI_SWEEP_POSITIONS if positions is None else positions)
    root = atari_root()
    games = archive_games(root)
    logger.info(
        "sweeping %d games at positions %s, %d episodes each",
        len(games),
        swept,
        ATARI_SWEEP_EPISODES,
    )
    payload = {
        "measurement": SWEEP_STEM,
        "positions": list(swept),
        "run": ATARI_RUN_INDEX,
        "horizon": HORIZON_MAX,
        "episodes_per_position": ATARI_SWEEP_EPISODES,
        "displacement_bar_pct": ATARI_DISPLACEMENT_BAR_PCT,
        **sweep_checkpoints(root, games, swept, ATARI_SWEEP_EPISODES),
    }
    return _write(out_dir, SWEEP_STEM, payload, sweep_text(payload))
