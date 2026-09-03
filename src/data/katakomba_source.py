"""Read the Katakomba HDF5 corpus as trajectories the pipeline consumes.

Each recorded NetHack game is one episode group. The observation is the map
region of the terminal, as a two-channel (tty_chars, tty_colors) grid, and a
fixed-length window is cut from each game because the recorded games are orders
of magnitude longer than the horizons trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import jax
import numpy as np

from config import (
    HORIZON_MAX,
    KATAKOMBA_ENV_NAME,
    KATAKOMBA_MAP_COLS,
    KATAKOMBA_MAP_ROWS,
    KATAKOMBA_MAX_EPISODE_STEPS,
    KATAKOMBA_WINDOW_OFFSET_FIXED_FRACTION,
    KATAKOMBA_WINDOW_OFFSET_FRACTION,
    KATAKOMBA_WINDOW_OFFSET_MODE,
    KATAKOMBA_WINDOW_OFFSET_PREFIX,
    KATAKOMBA_WINDOW_OFFSET_RANDOM,
    NLE_ACTION_VOCABULARY,
    OBS_CHANNEL_CLASSES_NLE,
    OBS_GRID_SHAPE_NLE,
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

# Corpus layout. One file per character build, one group per recorded game.
KATAKOMBA_FILE_GLOB: str = "*.hdf5"
ACTIONS_DATASET: str = "actions"
DONES_DATASET: str = "dones"
REWARDS_DATASET: str = "rewards"
TTY_CHARS_DATASET: str = "tty_chars"
TTY_COLORS_DATASET: str = "tty_colors"

# Per-game attributes the character build is composed from.
BUILD_ATTRS: tuple[str, ...] = ("role", "race", "align")
BUILD_SEPARATOR: str = "-"

# Field name reported by the observation spec. NetHack has exactly one field.
NLE_GRID_FIELD: str = "grid"

# The generating policy recorded in every trajectory's provenance.
KATAKOMBA_POLICY: str = "autoascend_bot"

# The single view a terminal recording supports.
KATAKOMBA_OBSERVATION_MODE: ObservationMode = ObservationMode.TOP_DOWN

# Channel order of the stacked observation, matching OBS_CHANNEL_CLASSES_NLE.
OBSERVATION_CHANNELS: tuple[str, ...] = (TTY_CHARS_DATASET, TTY_COLORS_DATASET)

# Shortest episode that supplies at least one (t, t + HORIZON_MAX) pair. Shorter
# games are skipped rather than padded.
MIN_EPISODE_FRAMES: int = HORIZON_MAX + 1


def map_keypress_to_nle_action(keypress: int) -> int:
    """Map a recorded terminal keypress byte to an NLE action index.

    Raises:
        NotImplementedError: Always. Nothing reads this.
    """
    raise NotImplementedError(
        "keypress bytes are the action vocabulary on the offline route; "
        f"mapping byte {keypress} to a Discrete(23) index is lossy and is "
        "implemented only if a live-offline comparison enters scope."
    )


def read_episode_arrays(
    path: Path, episode_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one recorded game's actions and terminal frames.

    Args:
        path: A Katakomba HDF5 file.
        episode_name: Name of the episode group within it.

    Returns:
        The actions, tty_chars and tty_colors arrays, unsliced.
    """
    with h5py.File(path, "r") as handle:
        group = handle[episode_name]
        return (
            group[ACTIONS_DATASET][()],
            group[TTY_CHARS_DATASET][()],
            group[TTY_COLORS_DATASET][()],
        )


def episode_names(path: Path) -> list[str]:
    """Return one file's episode group names in a deterministic order."""
    with h5py.File(path, "r") as handle:
        return sorted(handle.keys())


def corpus_files(root: Path) -> list[Path]:
    """Return the corpus files in a deterministic order.

    Raises:
        FileNotFoundError: If the root holds no corpus files.
    """
    files = sorted(root.glob(KATAKOMBA_FILE_GLOB))
    if not files:
        raise FileNotFoundError(
            f"no Katakomba files matching {KATAKOMBA_FILE_GLOB} under "
            f"{safe_rel(root)}. Set the corpus location with the "
            "KATAKOMBA_ROOT_ENV_VAR environment variable."
        )
    return files


class KatakombaTrajectorySource:
    """Yields fixed-length windows of recorded NetHack games.

    The observation is the terminal's map region as a (height, width, 2) grid
    of character and colour codes. Each game contributes one window, whose
    start is drawn per episode from the key passed to `trajectories`.

    Attributes:
        root: Directory holding the corpus files.
        num_steps: Actions per yielded trajectory.
        offset_mode: Where each window starts.
    """

    def __init__(
        self,
        root: Path,
        num_steps: int = KATAKOMBA_MAX_EPISODE_STEPS,
        offset_mode: str = KATAKOMBA_WINDOW_OFFSET_MODE,
    ) -> None:
        """Build a source over one corpus directory.

        Args:
            root: Directory holding the corpus files.
            num_steps: Actions per yielded trajectory.
            offset_mode: One of KATAKOMBA_WINDOW_OFFSET_MODES.
        """
        self.root = root
        self.num_steps = num_steps
        self.offset_mode = window_offset_mode(offset_mode)
        logger.info(
            "KatakombaTrajectorySource ready: root=%s num_steps=%d offset=%s",
            safe_rel(root),
            num_steps,
            self.offset_mode,
        )

    def spec(self, mode: ObservationMode) -> ObservationSpec:
        """Describe the terminal observation and the keypress action space.

        Args:
            mode: Must be the single mode a terminal recording supports.

        Returns:
            A single-field spec over the map region.

        Raises:
            ValueError: If any other mode is requested. A terminal recording
                carries one view, and returning a duplicate under a second
                name would put a false mode label on every artefact.
        """
        if mode is not KATAKOMBA_OBSERVATION_MODE:
            raise ValueError(
                f"the Katakomba corpus records one view only, "
                f"{KATAKOMBA_OBSERVATION_MODE.value}, and {mode.value} was "
                "requested. The observation-mode axis does not apply to this "
                "environment."
            )
        return ObservationSpec(
            fields=(
                FieldSpec(
                    name=NLE_GRID_FIELD,
                    shape=tuple(OBS_GRID_SHAPE_NLE),
                    cardinality=tuple(OBS_CHANNEL_CLASSES_NLE),
                ),
            ),
            num_actions=NLE_ACTION_VOCABULARY,
        )

    def trajectories(self, rng_key: jax.Array) -> Iterator[Trajectory]:
        """Yield one window per recorded game, deterministic given rng_key.

        Args:
            rng_key: PRNG key, split once per episode in corpus order. An
                episode's offset therefore depends on how many episodes
                precede it, so adding or removing a corpus file changes every
                window after it.

        Yields:
            One Trajectory per game long enough to supply a horizon pair.
        """
        key = rng_key
        episode_index = 0
        for path in corpus_files(self.root):
            for name in episode_names(path):
                key, episode_key = jax.random.split(key)
                trajectory = self._window_from(path, name, episode_key,
                                               episode_index)
                if trajectory is None:
                    continue
                yield trajectory
                episode_index += 1

    def _window_from(
        self, path: Path, name: str, episode_key: jax.Array, episode_index: int
    ) -> Trajectory | None:
        """Cut one window from one recorded game.

        Args:
            path: The corpus file.
            name: Episode group name within it.
            episode_key: PRNG key for this episode's offset draw.
            episode_index: Position in the yielded sequence.

        Returns:
            The window, or None when the game is too short to supply a pair at
            HORIZON_MAX.
        """
        with h5py.File(path, "r") as handle:
            group = handle[name]
            length = int(group[ACTIONS_DATASET].shape[0])
            if length < MIN_EPISODE_FRAMES:
                return None
            num_steps = min(self.num_steps, length - 1)
            start = self._offset(length, num_steps, episode_key)
            streams = self._streams(group, start, num_steps)
            build = self._build(group)

        # A window ends in truncation unless its final action is the one that
        # ended the recorded game. On a corpus storing one observation per
        # action that never happens, because the terminal transition's
        # successor state is not recorded and the window needs it.
        truncated = np.zeros(num_steps, dtype=bool)
        truncated[-1] = not bool(streams["terminated"][-1])
        return Trajectory(
            observations={KATAKOMBA_OBSERVATION_MODE: streams["observations"]},
            actions=streams["actions"],
            rewards=streams["rewards"],
            terminated=streams["terminated"],
            truncated=truncated,
            goal_position=np.zeros((0, 2), dtype=np.int32),
            provenance={
                "env_name": KATAKOMBA_ENV_NAME,
                "policy": KATAKOMBA_POLICY,
                "build": build,
                "game_id": name,
                "episode_index": episode_index,
                "episode_length": length,
                "window_offset": start,
                "window_offset_mode": self.offset_mode,
                "num_steps": num_steps,
            },
            executed_actions=streams["actions"],
        )

    def _streams(
        self, group: h5py.Group, start: int, num_steps: int
    ) -> dict[str, np.ndarray]:
        """Read one window's observations and per-step streams.

        Args:
            group: The episode group.
            start: First frame of the window.
            num_steps: Actions the window carries.

        Returns:
            The observations and the action, reward and termination streams.
        """
        stop = start + num_steps
        return {
            "observations": self._observation(group, start, stop + 1),
            "actions": self._actions(group[ACTIONS_DATASET][start:stop]),
            "rewards": np.asarray(group[REWARDS_DATASET][start:stop]),
            "terminated": np.asarray(
                group[DONES_DATASET][start:stop], dtype=bool
            ),
        }

    def _offset(self, length: int, num_steps: int, episode_key: jax.Array) -> int:
        """Return the frame the window starts at.

        Args:
            length: Frames the recorded game holds.
            num_steps: Actions the window carries.
            episode_key: PRNG key, read only by the random mode.

        Returns:
            A start index leaving num_steps + 1 frames available.
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

    @staticmethod
    def _observation(group: h5py.Group, start: int, stop: int) -> np.ndarray:
        """Stack the map region of both terminal channels into one grid.

        Args:
            group: The episode group.
            start: First frame, inclusive.
            stop: Last frame, exclusive.

        Returns:
            A (stop - start, height, width, 2) uint8 array, independent of any
            array the file was read into.

        Raises:
            ValueError: If a code falls outside its channel's declared
                cardinality. The colour channel has no headroom, so an
                out-of-range code would silently exceed the decoder's classes.
        """
        row_start, row_stop = KATAKOMBA_MAP_ROWS
        col_start, col_stop = KATAKOMBA_MAP_COLS
        channels = [
            np.asarray(group[name][start:stop, row_start:row_stop,
                                   col_start:col_stop])
            for name in OBSERVATION_CHANNELS
        ]
        for name, channel, classes in zip(
            OBSERVATION_CHANNELS, channels, OBS_CHANNEL_CLASSES_NLE
        ):
            if channel.min() < 0 or channel.max() >= classes:
                raise ValueError(
                    f"{name} holds codes outside [0, {classes}): observed "
                    f"[{int(channel.min())}, {int(channel.max())}]. The "
                    "declared cardinality sizes the decoder's per-cell "
                    "categorical, so a wider code has no class to land in."
                )
        return np.stack(channels, axis=-1).astype(np.uint8)

    @staticmethod
    def _actions(recorded: np.ndarray) -> np.ndarray:
        """Return the recorded keypress bytes as action indices.

        Args:
            recorded: One window's actions, as stored.

        Returns:
            An int32 array of keypress bytes.

        Raises:
            ValueError: If a byte falls outside the declared vocabulary.
        """
        actions = np.asarray(recorded).astype(np.int32)
        if actions.size and (
            actions.min() < 0 or actions.max() >= NLE_ACTION_VOCABULARY
        ):
            raise ValueError(
                f"recorded keypress outside [0, {NLE_ACTION_VOCABULARY}): "
                f"observed [{int(actions.min())}, {int(actions.max())}]. The "
                "vocabulary sizes the action embedding table."
            )
        return actions

    @staticmethod
    def _build(group: h5py.Group) -> str:
        """Return the character build this game was played on.

        Args:
            group: The episode group, carrying the per-game attributes.

        Returns:
            The role, race and alignment joined in lower case.
        """
        return BUILD_SEPARATOR.join(
            str(group.attrs[attr]).lower() for attr in BUILD_ATTRS
        )
