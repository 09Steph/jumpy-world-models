"""Read the ordered DQN Replay Atari archive as trajectories the pipeline consumes.

Each shard is TFRecord, optionally gzipped, holding one `tf.train.Example` per
episode. It is parsed with the standard library and Pillow, without
TensorFlow. Frames are decoded from PNG, and one window of up to `num_steps`
actions is cut from each admitted episode.
"""

from __future__ import annotations

import gzip
import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import jax
import numpy as np
from PIL import Image

from config import (
    ATARI_ACTION_VOCABULARY,
    ATARI_CHECKPOINT_POSITIONS,
    ATARI_CHECKPOINT_PROVENANCE_KEY,
    ATARI_ENV_NAME,
    ATARI_EPISODES_PER_CELL,
    ATARI_GAME_PROVENANCE_KEY,
    ATARI_MAX_EPISODE_STEPS,
    ATARI_RUN_INDEX,
    ATARI_WINDOW_OFFSET_MODE,
    HORIZON_MAX,
    KATAKOMBA_WINDOW_OFFSET_FIXED_FRACTION,
    KATAKOMBA_WINDOW_OFFSET_FRACTION,
    KATAKOMBA_WINDOW_OFFSET_PREFIX,
    KATAKOMBA_WINDOW_OFFSET_RANDOM,
    OBS_CHANNELS_GREYSCALE,
    OBS_GRID_SHAPE_ATARI,
    OBS_VALUE_RANGE_GREYSCALE,
    STRATUM_PROVENANCE_KEY,
    atari_games,
    atari_shard_paths,
    window_offset_mode,
)
from src.data.trajectory import (
    FieldSpec,
    ObservationMode,
    ObservationSpec,
    Trajectory,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import safe_rel

logger = get_logger(__name__)

# The observation spec's single field name.
ATARI_GRID_FIELD: str = "grid"

# The generating policy recorded in every trajectory's provenance: the replay
# of a DQN agent's own training, so a shard's position within a run is how far
# that agent had trained.
ATARI_POLICY: str = "dqn_replay"

# The single view a rendered game screen supports.
ATARI_OBSERVATION_MODE: ObservationMode = ObservationMode.TOP_DOWN

# Shortest episode that supplies one (t, t + HORIZON_MAX) pair, and the default
# admission floor. A conversion cutting longer windows passes its own. An
# episode below the floor is skipped.
MIN_EPISODE_FRAMES: int = HORIZON_MAX + 1

# TFRecord framing: uint64 length, uint32 CRC of the length, the payload, uint32
# CRC of the payload. The CRCs are skipped, so a corrupted payload is not
# detected here. The protobuf 64-bit and 32-bit wire widths reuse these sizes.
RECORD_LENGTH_BYTES: int = 8
RECORD_CRC_BYTES: int = 4
GZIP_MAGIC: bytes = b"\x1f\x8b"

# Protobuf wire types, and the field numbers of the Example message tree.
WIRE_VARINT: int = 0
WIRE_64BIT: int = 1
WIRE_LENGTH_DELIMITED: int = 2
WIRE_32BIT: int = 5
FEATURES_FIELD: int = 1
MAP_ENTRY_FIELD: int = 1
MAP_KEY_FIELD: int = 1
MAP_VALUE_FIELD: int = 2
BYTES_LIST_FIELD: int = 1
FLOAT_LIST_FIELD: int = 2
INT64_LIST_FIELD: int = 3
LIST_VALUE_FIELD: int = 1

# Feature keys read from each episode. One message is one whole episode, so the
# episode boundary is the message boundary and the terminal step is where the
# discount drops.
OBSERVATIONS_KEY: str = "observations"
ACTIONS_KEY: str = "actions"
REWARDS_KEY: str = "clipped_rewards"
DISCOUNTS_KEY: str = "discounts"
CHECKPOINT_KEY: str = "checkpoint_idx"
EPISODE_INDEX_KEY: str = "episode_idx"

# The discount recorded at an absorbing state.
TERMINAL_DISCOUNT: float = 0.0


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Return the varint at this offset and the offset past it."""
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return value, offset


def _walk_fields(payload: bytes) -> Iterator[tuple[int, int, int]]:
    """Yield one entry per top-level protobuf field in this payload.

    Args:
        payload: A serialised protobuf message.

    Yields:
        (field_number, value_offset, value_length) per field.

    Raises:
        ValueError: On a wire type other than 0, 1, 2 or 5.
    """
    offset = 0
    while offset < len(payload):
        header, offset = _read_varint(payload, offset)
        field_number, wire_type = header >> 3, header & 0x07
        if wire_type == WIRE_LENGTH_DELIMITED:
            length, offset = _read_varint(payload, offset)
            yield field_number, offset, length
            offset += length
        elif wire_type == WIRE_VARINT:
            start = offset
            _, offset = _read_varint(payload, offset)
            yield field_number, start, offset - start
        elif wire_type == WIRE_64BIT:
            yield field_number, offset, RECORD_LENGTH_BYTES
            offset += RECORD_LENGTH_BYTES
        elif wire_type == WIRE_32BIT:
            yield field_number, offset, RECORD_CRC_BYTES
            offset += RECORD_CRC_BYTES
        else:
            raise ValueError(
                f"protobuf wire type {wire_type} at offset {offset} is not one "
                "of 0, 1, 2 or 5, so the record framing has desynchronised"
            )


def _open_shard(path: Path) -> BinaryIO:
    """Return a binary stream over a shard, decompressing it if gzipped."""
    with path.open("rb") as probe:
        compressed = probe.read(len(GZIP_MAGIC)) == GZIP_MAGIC
    return gzip.open(path, "rb") if compressed else path.open("rb")


def iterate_records(path: Path) -> Iterator[bytes]:
    """Yield each TFRecord payload in one shard, in stored order.

    A stream that ends part-way through a length header ends iteration without
    raising.

    Args:
        path: The shard.

    Yields:
        One serialised `tf.train.Example` per record.

    Raises:
        ValueError: If a payload is shorter than its declared length.
    """
    with _open_shard(path) as handle:
        while True:
            header = handle.read(RECORD_LENGTH_BYTES)
            if len(header) < RECORD_LENGTH_BYTES:
                return
            length = struct.unpack("<Q", header)[0]
            handle.read(RECORD_CRC_BYTES)
            payload = handle.read(length)
            if len(payload) < length:
                raise ValueError(
                    f"{safe_rel(path)} declares a {length}-byte record and "
                    f"holds {len(payload)}, so the shard is truncated"
                )
            handle.read(RECORD_CRC_BYTES)
            yield payload


def _parse_feature(feature: bytes) -> tuple[str, list]:
    """Return one Feature's kind and its values.

    Args:
        feature: A serialised Feature message.

    Returns:
        (kind, values), where kind is "bytes", "int64", "float" or "unknown".
    """
    for number, offset, length in _walk_fields(feature):
        block = feature[offset:offset + length]
        if number == BYTES_LIST_FIELD:
            return "bytes", [
                block[value_offset:value_offset + value_length]
                for value_number, value_offset, value_length in _walk_fields(block)
                if value_number == LIST_VALUE_FIELD
            ]
        if number == INT64_LIST_FIELD:
            return "int64", _packed_varints(block)
        if number == FLOAT_LIST_FIELD:
            return "float", _packed_floats(block)
    return "unknown", []


def _packed_varints(block: bytes) -> list[int]:
    """Return the packed varints in an Int64List payload."""
    values: list[int] = []
    for number, offset, length in _walk_fields(block):
        if number != LIST_VALUE_FIELD:
            continue
        chunk = block[offset:offset + length]
        position = 0
        while position < len(chunk):
            value, position = _read_varint(chunk, position)
            values.append(value)
    return values


def _packed_floats(block: bytes) -> list[float]:
    """Return the packed float32 values in a FloatList payload."""
    values: list[float] = []
    for number, offset, length in _walk_fields(block):
        if number != LIST_VALUE_FIELD:
            continue
        chunk = block[offset:offset + length]
        values.extend(struct.unpack(f"<{len(chunk) // 4}f", chunk))
    return values


def parse_example(payload: bytes) -> dict[str, list]:
    """Return one record's feature map, name to its list of values.

    Args:
        payload: A serialised `tf.train.Example`.
    """
    features: dict[str, list] = {}
    for number, offset, length in _walk_fields(payload):
        if number != FEATURES_FIELD:
            continue
        block = payload[offset:offset + length]
        for entry_number, entry_offset, entry_length in _walk_fields(block):
            if entry_number != MAP_ENTRY_FIELD:
                continue
            entry = block[entry_offset:entry_offset + entry_length]
            key, values = _parse_map_entry(entry)
            if key is not None:
                features[key] = values
    return features


def _parse_map_entry(entry: bytes) -> tuple[str | None, list]:
    """Return one map entry's key and values.

    Args:
        entry: A serialised map<string, Feature> entry.

    Returns:
        (key or None, values).
    """
    key: str | None = None
    values: list = []
    for number, offset, length in _walk_fields(entry):
        chunk = entry[offset:offset + length]
        if number == MAP_KEY_FIELD:
            key = chunk.decode("utf-8", errors="replace")
        elif number == MAP_VALUE_FIELD:
            _, values = _parse_feature(chunk)
    return key, values


def decode_frames(encoded: Sequence[bytes]) -> np.ndarray:
    """Decode one episode's PNG frames into a stacked array.

    Args:
        encoded: One PNG payload per decision step.

    Returns:
        Shape (steps, height, width, channels) as uint8.

    Raises:
        ValueError: If a frame does not decode to OBS_GRID_SHAPE_ATARI.
    """
    height, width = OBS_GRID_SHAPE_ATARI
    frames = np.empty(
        (len(encoded), height, width, OBS_CHANNELS_GREYSCALE), dtype=np.uint8
    )
    for step, payload in enumerate(encoded):
        decoded = np.asarray(Image.open(io.BytesIO(payload)))
        if decoded.shape != (height, width):
            raise ValueError(
                f"frame {step} decodes to {decoded.shape}, and the declared "
                f"frame is {(height, width)}. OBS_GRID_SHAPE_ATARI and the "
                "archive disagree."
            )
        frames[step, :, :, 0] = decoded
    return frames


@dataclass(frozen=True)
class ArchiveEpisode:
    """One episode as the archive stores it, before any window is cut.

    Attributes:
        frames: Shape (steps, height, width, channels) as uint8.
        actions: Shape (steps,).
        rewards: Shape (steps,).
        terminal: Whether the final step reaches an absorbing state.
        checkpoint: Shard position within the training run.
        episode_index: The archive's own index for this episode.
    """

    frames: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminal: bool
    checkpoint: int
    episode_index: int

    def __len__(self) -> int:
        """Return the number of recorded frames."""
        return int(self.frames.shape[0])


def decode_archive_episode(features: dict[str, list]) -> ArchiveEpisode:
    """Turn one parsed record into an episode.

    Args:
        features: The record's feature map.

    Returns:
        The episode, with its frames decoded.

    Raises:
        KeyError: If a feature is absent. A missing per-step stream names what
            the record carries.
        ValueError: If the per-step streams disagree in length, or if an action
            falls outside the declared vocabulary.
    """
    required = (OBSERVATIONS_KEY, ACTIONS_KEY, REWARDS_KEY, DISCOUNTS_KEY)
    missing = [key for key in required if key not in features]
    if missing:
        raise KeyError(
            f"record is missing {missing}; it carries "
            f"{sorted(features)}. The archive's feature names have changed."
        )
    frames = decode_frames(features[OBSERVATIONS_KEY])
    actions = np.asarray(features[ACTIONS_KEY], dtype=np.int32)
    rewards = np.asarray(features[REWARDS_KEY], dtype=np.float32)
    discounts = np.asarray(features[DISCOUNTS_KEY], dtype=np.float32)

    lengths = {
        OBSERVATIONS_KEY: frames.shape[0],
        ACTIONS_KEY: actions.size,
        REWARDS_KEY: rewards.size,
        DISCOUNTS_KEY: discounts.size,
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"per-step streams disagree in length: {lengths}. One frame per "
            "decision step is what makes the action stream index the frames."
        )
    outside = actions[(actions < 0) | (actions >= ATARI_ACTION_VOCABULARY)]
    if outside.size:
        raise ValueError(
            f"actions {sorted(set(outside.tolist()))} fall outside "
            f"[0, {ATARI_ACTION_VOCABULARY}). ATARI_ACTION_VOCABULARY and the "
            "archive disagree."
        )
    return ArchiveEpisode(
        frames=frames,
        actions=actions,
        rewards=rewards,
        terminal=bool(discounts[-1] == TERMINAL_DISCOUNT),
        checkpoint=int(features[CHECKPOINT_KEY][0]),
        episode_index=int(features[EPISODE_INDEX_KEY][0]),
    )


def trajectory_from_frames(
    frames: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminal: bool,
    provenance: dict,
) -> Trajectory:
    """Assemble a Trajectory from one episode's arrays.

    The last step is marked terminated when `terminal` is true and truncated
    otherwise, and `executed_actions` is filled with `actions`.

    Args:
        frames: Shape (num_steps + 1, height, width, channels).
        actions: Shape (num_steps,).
        rewards: Shape (num_steps,).
        terminal: Whether the final action reaches an absorbing state.
        provenance: Fields recorded with the episode.

    Returns:
        The episode in the store's layout.

    Raises:
        ValueError: If there is not exactly one more frame than there are
            actions.
    """
    if frames.shape[0] != actions.size + 1:
        raise ValueError(
            f"{frames.shape[0]} frames against {actions.size} actions. A "
            "trajectory carries num_steps + 1 frames, the last being the "
            "endpoint reached after the final action."
        )
    terminated = np.zeros(actions.size, dtype=bool)
    terminated[-1] = terminal
    truncated = np.zeros(actions.size, dtype=bool)
    truncated[-1] = not terminal
    return Trajectory(
        observations={ATARI_OBSERVATION_MODE: frames},
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        goal_position=np.zeros((0, 2), dtype=np.int32),
        provenance=provenance,
        executed_actions=actions,
    )


class AtariTrajectorySource:  # pylint: disable=too-many-instance-attributes
    """Yields one window per admitted episode of the recorded Atari archive.

    Reads the selected games at the selected shard positions, in that order. A
    window carries up to `num_steps` actions, fewer when the episode is
    shorter, and starts where `offset_mode` puts it.

    Attributes:
        root: Directory holding the archive, one subdirectory per game.
        games: The games this conversion reads.
        positions: Shard positions read within each game's run.
        num_steps: Most actions a yielded trajectory carries.
        offset_mode: Where each window starts.
        episodes_per_cell: Episodes taken per game per position.
        min_episode_frames: Frames an episode needs to be admitted.
        shards: Each game's resolved shard paths.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        root: Path,
        *,
        games: Sequence[str] | None = None,
        positions: Sequence[int] = ATARI_CHECKPOINT_POSITIONS,
        num_steps: int = ATARI_MAX_EPISODE_STEPS,
        offset_mode: str = ATARI_WINDOW_OFFSET_MODE,
        episodes_per_cell: int = ATARI_EPISODES_PER_CELL,
        min_episode_frames: int = MIN_EPISODE_FRAMES,
        run: int = ATARI_RUN_INDEX,
    ) -> None:
        """Build a source over one archive directory.

        Resolves every selected game's shards, so a game absent from disk
        raises here.

        Args:
            root: Directory holding the archive.
            games: Games to convert. None takes the configured selection.
            positions: Shard positions within the run.
            num_steps: Actions per yielded trajectory.
            offset_mode: One of KATAKOMBA_WINDOW_OFFSET_MODES.
            episodes_per_cell: Episodes taken per game per position.
            min_episode_frames: Frames an episode needs to be admitted. An
                episode below it is skipped and does not count against
                episodes_per_cell.
            run: Which training run to read.
        """
        self.root = root
        self.games = atari_games(games)
        self.positions = tuple(positions)
        self.num_steps = num_steps
        self.offset_mode = window_offset_mode(offset_mode)
        self.episodes_per_cell = episodes_per_cell
        self.min_episode_frames = min_episode_frames
        self.shards = {
            game: atari_shard_paths(root, game, self.positions, run)
            for game in self.games
        }
        logger.info(
            "AtariTrajectorySource ready: root=%s games=%d positions=%s "
            "num_steps=%d offset=%s min_frames=%d",
            safe_rel(root),
            len(self.games),
            self.positions,
            num_steps,
            self.offset_mode,
            self.min_episode_frames,
        )

    def spec(self, mode: ObservationMode) -> ObservationSpec:
        """Describe the frame and the action space.

        Args:
            mode: Must be the single mode a game screen supports.

        Returns:
            A single-field spec over the frame.

        Raises:
            ValueError: If any mode other than ATARI_OBSERVATION_MODE is
                requested.
        """
        if mode is not ATARI_OBSERVATION_MODE:
            raise ValueError(
                f"the Atari archive records one view only, "
                f"{ATARI_OBSERVATION_MODE.value}, and {mode.value} was "
                "requested. The observation-mode axis does not apply to this "
                "environment."
            )
        return ObservationSpec(
            fields=(
                FieldSpec(
                    name=ATARI_GRID_FIELD,
                    shape=tuple(OBS_GRID_SHAPE_ATARI),
                    value_range=OBS_VALUE_RANGE_GREYSCALE,
                ),
            ),
            num_actions=ATARI_ACTION_VOCABULARY,
        )

    def trajectories(self, rng_key: jax.Array) -> Iterator[Trajectory]:
        """Yield one window per admitted episode, deterministic given rng_key.

        Args:
            rng_key: PRNG key, split once per record read, admitted or not. An
                episode's key depends on how many records precede it, so
                changing the games, positions, per-cell count or admission
                floor can move every later window.

        Yields:
            One Trajectory per admitted episode, up to episodes_per_cell per
            shard.
        """
        key = rng_key
        episode_index = 0
        for game in self.games:
            for path in self.shards[game]:
                taken = 0
                for payload in iterate_records(path):
                    if taken >= self.episodes_per_cell:
                        break
                    key, episode_key = jax.random.split(key)
                    window = self._window_from(
                        decode_archive_episode(parse_example(payload)),
                        game,
                        episode_key,
                        episode_index,
                    )
                    if window is None:
                        continue
                    yield window
                    taken += 1
                    episode_index += 1
                logger.info(
                    "converted %d episodes from %s", taken, safe_rel(path)
                )

    def _window_from(
        self,
        episode: ArchiveEpisode,
        game: str,
        episode_key: jax.Array,
        episode_index: int,
    ) -> Trajectory | None:
        """Cut one window from one recorded episode.

        Args:
            episode: The decoded episode.
            game: Which game it came from.
            episode_key: PRNG key for this episode's offset draw.
            episode_index: Position in the yielded sequence.

        Returns:
            The window, or None when the episode holds fewer frames than
            min_episode_frames.
        """
        length = len(episode)
        if length < self.min_episode_frames:
            return None
        num_steps = min(self.num_steps, length - 1)
        start = self._offset(length, num_steps, episode_key)
        stop = start + num_steps
        # Never true on the archive, which stores one frame per action: the
        # window needs the final action's successor frame, so stop stays below
        # length and every archive window is marked truncated.
        terminal = episode.terminal and stop == length
        return trajectory_from_frames(
            frames=episode.frames[start:stop + 1],
            actions=episode.actions[start:stop],
            rewards=episode.rewards[start:stop],
            terminal=terminal,
            provenance={
                "env_name": ATARI_ENV_NAME,
                "policy": ATARI_POLICY,
                ATARI_GAME_PROVENANCE_KEY: game,
                # The split stratifies on this key.
                STRATUM_PROVENANCE_KEY: game,
                ATARI_CHECKPOINT_PROVENANCE_KEY: episode.checkpoint,
                "episode_index": episode_index,
                "archive_episode_index": episode.episode_index,
                "episode_length": length,
                "window_offset": start,
                "window_offset_mode": self.offset_mode,
                "num_steps": num_steps,
            },
        )

    def _offset(self, length: int, num_steps: int, episode_key: jax.Array) -> int:
        """Return the frame the window starts at.

        Args:
            length: Frames the recorded episode holds.
            num_steps: Actions the window carries.
            episode_key: PRNG key, read only by the random mode.

        Returns:
            A start index leaving num_steps + 1 frames available.

        Raises:
            ValueError: If the mode is declared but has no branch here.
        """
        span = length - (num_steps + 1)
        if span <= 0:
            return 0
        if self.offset_mode == KATAKOMBA_WINDOW_OFFSET_PREFIX:
            return 0
        if self.offset_mode == KATAKOMBA_WINDOW_OFFSET_FIXED_FRACTION:
            return int(round(span * KATAKOMBA_WINDOW_OFFSET_FRACTION))
        if self.offset_mode == KATAKOMBA_WINDOW_OFFSET_RANDOM:
            return int(jax.random.randint(episode_key, (), 0, span + 1))
        raise ValueError(
            f"window offset mode '{self.offset_mode}' is declared but has no "
            "branch here. A new mode must be given one rather than falling "
            "through to another mode's behaviour."
        )
