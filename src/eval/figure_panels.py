"""Thesis figure panels, assembled from the artefacts under outputs/.

Each builder returns its layout, the keys it read, what it left out, caption
notes and any table of plotted values. An absent run, arm, fit or view is
recorded as an omission, and MissingArtefactError is raised only when nothing
the figure draws is present. FigureContractError is raised when a present
artefact lacks a field the figure cannot draw without, declares an error metric
no figure reads, or lists seeds that cannot be paired.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from config import (
    ARM_AR_ENDPOINT,
    ARM_AR_ONE_STEP,
    ARM_DIRECT,
    ARMS,
    ATARI_GAMES,
    ATARI_LONG_HORIZON,
    ATARI_LONG_HORIZON_ENV_NAME,
    ATARI_LONG_HORIZON_GAME,
    DISPLACEMENT_CSV_FILENAME,
    EVALUATION_HORIZONS,
    EXTRAPOLATION_HORIZONS,
    FAST_DIR_NAME,
    FIGURES_DIR_NAME,
    FIT_ARTEFACT_NAME,
    OBS_MODE_EGOCENTRIC,
    OBS_MODE_TOP_DOWN,
    OBS_MODES,
    OUTPUTS_DIR,
    REPORTED_RATIO_HORIZONS,
    REPRESENTATION_GREYSCALE,
    REPRESENTATION_RGB,
    REPRESENTATION_SYMBOLIC,
    SEEDS,
    TEST_RUN_SUFFIX,
)
from src.eval.figure_style import (
    ARM_COLOURS,
    ARM_DISPLAY,
    ARM_LINE_STYLES,
    ARM_MARKERS,
    ARM_SHORT_DISPLAY,
    BASELINE_COLOUR,
    BASELINE_DISPLAY,
    BINDING_BASELINE_COLOUR,
    BINDING_BASELINE_STYLE,
    CLIMATOLOGY_BASELINE_STYLE,
    COMPACT_ROW_HEIGHT_IN,
    COPY_BASELINE_STYLE,
    ENV_DISPLAY,
    ESTIMATOR_COLOURS,
    ESTIMATOR_DISPLAY,
    ESTIMATOR_JITTER,
    EXTRAPOLATION_STYLE,
    FOREST_AXIS_HEIGHT_IN,
    FOREST_ROW_HEIGHT_IN,
    FOREST_SERIES_PER_ROW,
    GROUP_LABEL_ROTATION_DEG,
    HATCH_PATTERN,
    HEATMAP_AXIS_HEIGHT_IN,
    HEATMAP_ROW_HEIGHT_IN,
    LADDER_COLOURS,
    LADDER_SECONDARY_OFFSET,
    LINE_WIDTH_PT,
    METRIC_DISPLAY,
    MODE_DISPLAY,
    MODE_LINE_STYLES,
    NEUTRAL_COLOUR,
    POLICY_DISPLAY,
    POLICY_LINE_STYLES,
    READING_DISPLAY,
    REFERENCE_ALPHA,
    REPRESENTATION_DISPLAY,
    REPRESENTATION_LINE_STYLES,
    ROOM_COLOURS,
    SECONDARY_MARKER,
    SELECTION_ROW_HEIGHT_IN,
    SERIES_MARKER,
    SPLIT_DISPLAY,
    TICK_ROTATION_DEG,
    display,
)
from src.eval.plots import (
    Band,
    BarPanel,
    BarSeries,
    FigureLayout,
    ForestPanel,
    ForestPoint,
    HeatmapPanel,
    LayoutRow,
    LinePanel,
    Proxy,
    Reference,
    ScatterPanel,
    ScatterSeries,
)
from src.pipeline.aggregate_stats import (
    BOOTSTRAP_REPS,
    CONFIDENCE_INTERVAL_SIZE,
    MIN_SEEDS_FOR_IQM,
    aggregate_scalar,
)
from src.pipeline.metrics_schema import (
    CLIMATOLOGY_KEY,
    CLIMATOLOGY_MSE_KEY,
    COPY_SIGMA_DISCOUNTED_INTEGRAL_KEY,
    COPY_SIGMA_INTEGRAL_KEY,
    COPY_SIGMA_SUM_KEY,
    ERROR_METRIC_CROSS_ENTROPY,
    ERROR_METRIC_KEY,
    ERROR_METRIC_MSE,
    MOVER_MSE_SKILL_KEY,
    PER_HORIZON_GAP_KEY,
    PER_HORIZON_KEY,
    SIGMA_DISCOUNTED_INTEGRAL_KEY,
    SIGMA_INTEGRAL_KEY,
    SIGMA_KEY,
    SIGMA_SUM_KEY,
    SKILL_SCORE_KEY,
    copy_error_key,
    model_error_key,
)
from src.utils.paths import REPO_ROOT

# Where the Atari selection artefacts are kept, and how each is cited.
ATARI_RESULTS_DIR: Path = REPO_ROOT / "results" / "atari"
ATARI_RESULTS_CITE: str = f"{ATARI_RESULTS_DIR.parent.name}/{ATARI_RESULTS_DIR.name}"
ATARI_RANKING_GLOB: str = "atari_game_ranking_position*.json"

# The reported split, and the split a development twin reads.
REPORTED_SPLIT: str = "test"
VALIDATION_SPLIT: str = "validation"

# Artefact keys read here that no shared schema module names.
AGGREGATE_FILENAME_TEMPLATE: str = "aggregate_arm{arm}.json"
AGGREGATE_ARM_PATTERN: re.Pattern[str] = re.compile(r"aggregate_arm(\d+)\.json")
BY_MODE_KEY: str = "by_mode"
ARMS_KEY: str = "arms"
SEEDS_KEY: str = "seeds"
N_SEEDS_KEY: str = "n_seeds"
IQM_KEY: str = "iqm"
CI_LOW_KEY: str = "ci_low"
CI_HIGH_KEY: str = "ci_high"
VALUES_KEY: str = "values"
REPORTED_EXPONENT_KEY: str = "reported_exponent"
REPORTED_VALUE_KEY: str = "value"
REPORTED_SOURCE_KEY: str = "source"
N_IDENTIFIED_KEY: str = "n_identified"
RESIDUALS_KEY: str = "residuals"
ENDPOINT_RATIO_KEY: str = "endpoint_error_ratio"
ESTIMATOR_KEYS: tuple[str, ...] = ("three_parameter", "two_parameter", "log_log")
MOVER_ACCURACY_KEY: str = "mover_restricted_accuracy"
DISPLACEMENT_ENVIRONMENT_COLUMN: str = "environment"
DISPLACEMENT_MODE_COLUMN: str = "mode"
DISPLACEMENT_MEASURE_COLUMN: str = "measure"
DISPLACEMENT_CHANNEL_COLUMN: str = "channel"
DISPLACEMENT_HORIZON_COLUMN: str = "horizon"
DISPLACEMENT_MEAN_COLUMN: str = "mean"
DISPLACEMENT_MEASURE: str = "hamming_cell_fraction"
DISPLACEMENT_CHANNEL: str = "all"
RANKING_POSITION_KEY: str = "position"
RANKING_GAMES_KEY: str = "games"
RANKING_GAME_KEY: str = "game"
RANKING_DISPLACEMENT_KEY: str = "displacement_pct"
RANKING_BAR_KEY: str = "displacement_bar_pct"
RANKING_LENGTH_GATE_KEY: str = "clears_length_gate"

# The climatology baseline and the copy reading that pair with each error metric
# and each compounding-error reading.
CLIMATOLOGY_KEYS: dict[str, str] = {
    ERROR_METRIC_CROSS_ENTROPY: CLIMATOLOGY_KEY,
    ERROR_METRIC_MSE: CLIMATOLOGY_MSE_KEY,
}
COPY_READINGS: dict[str, str] = {
    SIGMA_SUM_KEY: COPY_SIGMA_SUM_KEY,
    SIGMA_INTEGRAL_KEY: COPY_SIGMA_INTEGRAL_KEY,
    SIGMA_DISCOUNTED_INTEGRAL_KEY: COPY_SIGMA_DISCOUNTED_INTEGRAL_KEY,
}

# Depths below outputs/ a displacement table sits at, and the trees never read.
DISPLACEMENT_GLOBS: tuple[str, ...] = (
    f"*/*/*/{DISPLACEMENT_CSV_FILENAME}",
    f"*/*/*/*/{DISPLACEMENT_CSV_FILENAME}",
)
EXCLUDED_TREES: frozenset[str] = frozenset({FAST_DIR_NAME, FIGURES_DIR_NAME})

# Margin on mover-restricted accuracy over the copy baseline, whose own
# mover-restricted accuracy is zero, and the horizons the learnability and
# per-horizon bars read.
P1_MOVER_MARGIN: float = 0.20
P1_HORIZONS: tuple[int, ...] = (1, 10)
SUPPLEMENTARY_HORIZONS: tuple[int, ...] = (1, 10, 100)

# The view each representation is read in, and the view the slip figure reads.
PIXEL_MODE: str = OBS_MODE_EGOCENTRIC
ATARI_MODE: str = OBS_MODE_TOP_DOWN
SLIP_FIGURE_MODE: str = OBS_MODE_EGOCENTRIC

PER_GAME_ABSENT: str = "the artefacts carry no per-game breakdown"
INTERVAL_DESCRIPTION: str = (
    f"IQM over seeds with a {round(100 * CONFIDENCE_INTERVAL_SIZE)} per cent "
    "bootstrap interval"
)
IDENTIFIED_NOTE: str = (
    "each exponent is annotated with the seeds on which its reported estimator "
    "is identified"
)


class MissingArtefactError(LookupError):
    """An artefact a figure reads is absent."""


class FigureContractError(ValueError):
    """An artefact is present but does not carry what a figure reads."""


@dataclass(frozen=True)
class Condition:
    """One reported condition and the run each arm is read from."""

    room: str
    policy: str
    slip: str
    env: str
    representation: str
    arm_runs: tuple[tuple[int, str], ...]

    @property
    def room_policy(self) -> str:
        """The room and the collection policy."""
        return f"{self.room}, {self.policy}"

    @property
    def name(self) -> str:
        """The condition on one line."""
        return f"{self.room_policy}, {self.slip}"

    @property
    def wrapped_name(self) -> str:
        """The condition on two lines, for a narrow panel title."""
        return f"{self.room}\n{self.policy}, {self.slip}"

    @property
    def arms(self) -> tuple[int, ...]:
        """The arms the condition carries."""
        return tuple(arm for arm, _ in self.arm_runs)

    @property
    def modes(self) -> tuple[str, ...]:
        """The observation modes the representation is scored in."""
        if self.representation == REPRESENTATION_RGB:
            return (PIXEL_MODE,)
        if self.representation == REPRESENTATION_GREYSCALE:
            return (ATARI_MODE,)
        return OBS_MODES

    def run_for(self, arm: int) -> str:
        """The run an arm is read from, without the split suffix."""
        return dict(self.arm_runs)[arm]


def _every_arm(run: str, arms: Sequence[int] = ARMS) -> tuple[tuple[int, str], ...]:
    """Map every arm to one run."""
    return tuple((arm, run) for arm in arms)


# The reported conditions, in the order the figures draw them.
FOURROOMS_ENV: str = "Navix-FourRooms-v0"
DOORKEY_ENV: str = "Navix-DoorKey-Random-5x5-v0"
DYNAMIC_OBSTACLES_ENV: str = "Navix-Dynamic-Obstacles-16x16-v0"
NO_SLIP: str = "slip 0.00"
PIXEL_ARMS: tuple[int, ...] = (ARM_DIRECT, ARM_AR_ONE_STEP)

FOURROOMS_NO_SLIP = Condition(
    "FourRooms", "uniform", NO_SLIP, FOURROOMS_ENV, REPRESENTATION_SYMBOLIC,
    ((ARM_DIRECT, "e1_50k"), (ARM_AR_ENDPOINT, "e2_base"), (ARM_AR_ONE_STEP, "e2_base")),
)
FOURROOMS_SLIP_010 = Condition(
    "FourRooms", "uniform", "slip 0.10", FOURROOMS_ENV, REPRESENTATION_SYMBOLIC,
    _every_arm("e3_p010"),
)
FOURROOMS_SLIP_025 = Condition(
    "FourRooms", "uniform", "slip 0.25", FOURROOMS_ENV, REPRESENTATION_SYMBOLIC,
    _every_arm("e3_p025"),
)
DYNAMIC_OBSTACLES_UNIFORM = Condition(
    "Dynamic-Obstacles", "uniform", NO_SLIP, DYNAMIC_OBSTACLES_ENV, REPRESENTATION_SYMBOLIC,
    _every_arm("e3_dynobs"),
)
DOORKEY_UNIFORM = Condition(
    "DoorKey", "uniform", NO_SLIP, DOORKEY_ENV, REPRESENTATION_SYMBOLIC,
    _every_arm("e6_doorkey_uniform"),
)
FOURROOMS_PPO = Condition(
    "FourRooms", "PPO", NO_SLIP, FOURROOMS_ENV, REPRESENTATION_SYMBOLIC, _every_arm("e5_fr_ppo"),
)
DOORKEY_PPO = Condition(
    "DoorKey", "PPO", NO_SLIP, DOORKEY_ENV, REPRESENTATION_SYMBOLIC, _every_arm("e10b_dk_ppo_3arm"),
)
DYNAMIC_OBSTACLES_PPO = Condition(
    "Dynamic-Obstacles", "PPO", NO_SLIP, DYNAMIC_OBSTACLES_ENV, REPRESENTATION_SYMBOLIC,
    _every_arm("e11b_dynobs_ppo_3arm"),
)
PIXEL_FOURROOMS = Condition(
    "FourRooms", "PPO", NO_SLIP, FOURROOMS_ENV, REPRESENTATION_RGB,
    _every_arm("e7_fr_rgb_ppo", PIXEL_ARMS),
)
PIXEL_DOORKEY = Condition(
    "DoorKey", "PPO", NO_SLIP, DOORKEY_ENV, REPRESENTATION_RGB,
    _every_arm("e8_dk_rgb_ppo", PIXEL_ARMS),
)
PIXEL_DYNAMIC_OBSTACLES = Condition(
    "Dynamic-Obstacles", "PPO", NO_SLIP, DYNAMIC_OBSTACLES_ENV, REPRESENTATION_RGB,
    _every_arm("e9_dynobs_rgb_ppo", PIXEL_ARMS),
)
ATARI_POSITION_0 = Condition(
    "Atari", "DQN Replay", "position 0", "atari-dqn-replay", REPRESENTATION_GREYSCALE,
    _every_arm("a_base_p0", PIXEL_ARMS),
)
ATARI_POSITION_24 = Condition(
    "Atari", "DQN Replay", "position 24", "atari-dqn-replay-p24", REPRESENTATION_GREYSCALE,
    _every_arm("a_ladder_p24", PIXEL_ARMS),
)
ATARI_POSITION_49 = Condition(
    "Atari", "DQN Replay", "position 49", "atari-dqn-replay-p49", REPRESENTATION_GREYSCALE,
    _every_arm("a_ladder_p49", PIXEL_ARMS),
)
LONG_HORIZON = Condition(
    ATARI_LONG_HORIZON_GAME, "DQN Replay", f"trained to h = {ATARI_LONG_HORIZON}",
    ATARI_LONG_HORIZON_ENV_NAME, REPRESENTATION_GREYSCALE,
    _every_arm("b44_robotank_h1024", PIXEL_ARMS),
)

SYMBOLIC_CONDITIONS: tuple[Condition, ...] = (
    FOURROOMS_NO_SLIP, FOURROOMS_SLIP_010, FOURROOMS_SLIP_025, DYNAMIC_OBSTACLES_UNIFORM,
    DOORKEY_UNIFORM, FOURROOMS_PPO, DOORKEY_PPO, DYNAMIC_OBSTACLES_PPO,
)
PIXEL_CONDITIONS: tuple[Condition, ...] = (
    PIXEL_FOURROOMS, PIXEL_DOORKEY, PIXEL_DYNAMIC_OBSTACLES,
)
SLIP_LADDER: tuple[tuple[str, Condition], ...] = (
    ("No slip", FOURROOMS_NO_SLIP),
    ("Slip 0.10", FOURROOMS_SLIP_010),
    ("Slip 0.25", FOURROOMS_SLIP_025),
)
ROOM_POLICY_CONDITIONS: tuple[Condition, ...] = (
    FOURROOMS_NO_SLIP, FOURROOMS_PPO, DOORKEY_UNIFORM, DOORKEY_PPO,
    DYNAMIC_OBSTACLES_UNIFORM, DYNAMIC_OBSTACLES_PPO,
)
REPRESENTATION_PAIRS: tuple[tuple[Condition, Condition], ...] = (
    (FOURROOMS_PPO, PIXEL_FOURROOMS),
    (DOORKEY_PPO, PIXEL_DOORKEY),
    (DYNAMIC_OBSTACLES_PPO, PIXEL_DYNAMIC_OBSTACLES),
)
ATARI_LADDER: tuple[tuple[int, Condition], ...] = (
    (0, ATARI_POSITION_0), (24, ATARI_POSITION_24), (49, ATARI_POSITION_49),
)
HORIZON_ABLATION: tuple[tuple[int, str], ...] = (
    (128, "b44_ablation_h128"),
    (256, "b44_ablation_h256"),
    (512, "b44_ablation_h512"),
    (ATARI_LONG_HORIZON, "b44_robotank_h1024"),
)

# The saturation figure: the conditions it draws, the arm its data row reads,
# the arms its model row draws, each row's title and metric, and the columns of
# the table of plotted values it writes.
SATURATION_CONDITIONS: tuple[Condition, ...] = ROOM_POLICY_CONDITIONS
SATURATION_DATA_ARM: int = ARM_DIRECT
SATURATION_MODEL_ARMS: tuple[int, ...] = (ARM_DIRECT,)
MEAN_CHANGED_CELLS_KEY: str = "mean_changed_cells"
SATURATION_ROWS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("Data", MEAN_CHANGED_CELLS_KEY, (SATURATION_DATA_ARM,)),
    ("Model", MOVER_ACCURACY_KEY, SATURATION_MODEL_ARMS),
)
SATURATION_TABLE_COLUMNS: tuple[str, ...] = (
    "row", "view", "room", "policy", "arm", "source", "metric", "horizon",
    "iqm", "ci_low", "ci_high",
)
STATE_CHANGE_NOTE: str = (
    "state change counts the cells changed on each room's own grid, so its level "
    "is not comparable between rooms"
)
PIXEL_SATURATION_ABSENT: str = (
    "pixel rows are not drawn: displacement.csv carries neither model_mse nor copy_mse, "
    "and a Hamming cell fraction cannot share the changed-cell axis"
)

# Free text of the slip figure's gap panels.
GAP_READOUTS: tuple[tuple[str, str], ...] = (
    (REPORTED_EXPONENT_KEY, "Gap in fitted exponent"),
    (ENDPOINT_RATIO_KEY, "Gap in endpoint error ratio"),
    (SIGMA_INTEGRAL_KEY, "Gap in compounding error"),
)
GAP_AXIS_LABEL: str = "Arm 3 minus arm 1"
SLIP_LADDER_LEGEND: str = "FourRooms, uniform, slip ladder"
DYNAMIC_OBSTACLES_TICK: str = "Dynamic-\nObstacles"


@dataclass(frozen=True)
class Stat:
    """An IQM, its interval and the per-seed values behind it."""

    iqm: float
    low: float
    high: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class Curve:
    """One metric's IQM and interval at each horizon it is present at."""

    horizons: tuple[int, ...]
    centre: tuple[float, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]


@dataclass(frozen=True)
class FigureBuild:
    """What a builder returns: the layout, the keys read, omissions, notes and plotted values."""

    layout: FigureLayout
    keys: tuple[str, ...]
    omissions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    table: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Artefact:
    """A loaded JSON artefact and the path it is cited by."""

    payload: Any
    cited: str

    def mode(self, mode: str) -> dict:
        """Return the artefact's block for one observation mode."""
        block = (self.payload.get(BY_MODE_KEY) or {}).get(mode)
        if not isinstance(block, dict):
            raise MissingArtefactError(f"{self.cited} carries no {mode} block")
        return block

    def arm(self, mode: str, arm: int) -> dict:
        """Return one arm's block under one observation mode."""
        block = (self.mode(mode).get(ARMS_KEY) or {}).get(str(arm))
        if not isinstance(block, dict):
            raise MissingArtefactError(f"{self.cited} carries no arm {arm} under {mode}")
        return block


@dataclass
class OutputsTree:
    """The outputs tree the figures read, recording every artefact loaded."""

    root: Path
    atari_results: Path = ATARI_RESULTS_DIR
    reps: int = BOOTSTRAP_REPS
    reads: list[str] = field(default_factory=list)
    cache: dict[Path, Any] = field(default_factory=dict)

    def cite(self, path: Path) -> str:
        """Return a path as it is cited, relative to the tree it sits in."""
        for base, name in ((self.root, OUTPUTS_DIR.name), (self.atari_results, ATARI_RESULTS_CITE)):
            if path.is_relative_to(base):
                return f"{name}/{path.relative_to(base).as_posix()}"
        return path.name

    def _record(self, cited: str) -> None:
        """Note that an artefact was read."""
        if cited not in self.reads:
            self.reads.append(cited)

    def load_json(self, path: Path) -> Artefact:
        """Load a JSON artefact once and record that it was read."""
        cited = self.cite(path)
        if path not in self.cache:
            if not path.is_file():
                raise MissingArtefactError(f"{cited} is absent")
            self.cache[path] = json.loads(path.read_text(encoding="utf-8"))
        self._record(cited)
        return Artefact(self.cache[path], cited)

    def load_rows(self, path: Path) -> list[dict[str, str]]:
        """Load a CSV artefact and record that it was read."""
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self._record(self.cite(path))
        return rows

    def run_dir(self, run: str, split: str) -> Path:
        """Return a run's directory for one split."""
        return self.root / (run + TEST_RUN_SUFFIX if split == REPORTED_SPLIT else run)

    def aggregate(self, run: str, env: str, arm: int, split: str) -> Artefact:
        """Load one arm's aggregate."""
        return self.load_json(
            self.run_dir(run, split) / env / AGGREGATE_FILENAME_TEMPLATE.format(arm=arm)
        )

    def fit(self, run: str, env: str, split: str) -> Artefact:
        """Load a run's fit artefact."""
        return self.load_json(self.run_dir(run, split) / env / FIT_ARTEFACT_NAME)


def stat_of(block: Any) -> Stat | None:
    """Return a statistic block as a Stat, or None where it carries no IQM.

    A missing interval bound falls back to the IQM, which draws a zero-width
    interval.
    """
    if not isinstance(block, dict) or block.get(IQM_KEY) is None:
        return None
    centre = float(block[IQM_KEY])
    low, high = block.get(CI_LOW_KEY), block.get(CI_HIGH_KEY)
    return Stat(
        centre,
        centre if low is None else float(low),
        centre if high is None else float(high),
        tuple(float(value) for value in block.get(VALUES_KEY) or ()),
    )


def horizon_stat(mode_block: dict, horizon: int, key: str) -> Stat | None:
    """Return one metric's statistic at one horizon of a mode block."""
    return stat_of(((mode_block.get(PER_HORIZON_KEY) or {}).get(str(horizon)) or {}).get(key))


def sigma_stat(arm_block: dict, reading: str) -> Stat | None:
    """Return one compounding-error reading from a fit arm block."""
    return stat_of((arm_block.get(SIGMA_KEY) or {}).get(reading))


def interval(values: Sequence[float], reps: int) -> Stat | None:
    """Return the IQM and bootstrap interval of per-seed values, or None when too few."""
    if len(values) < MIN_SEEDS_FOR_IQM:
        return None
    return stat_of(aggregate_scalar(list(values), reps=reps))


def curve_of(
    mode_block: dict,
    key: str,
    *,
    section: str = PER_HORIZON_KEY,
    horizons: Iterable[int] | None = None,
) -> Curve:
    """Return one metric's curve from a mode block, skipping horizons without it."""
    per_horizon = mode_block.get(section) or {}
    wanted = None if horizons is None else set(horizons)
    points: list[tuple[int, Stat]] = []
    for name in sorted(per_horizon, key=int):
        if wanted is not None and int(name) not in wanted:
            continue
        stat = stat_of((per_horizon[name] or {}).get(key))
        if stat is not None:
            points.append((int(name), stat))
    return Curve(
        tuple(horizon for horizon, _ in points),
        tuple(stat.iqm for _, stat in points),
        tuple(stat.low for _, stat in points),
        tuple(stat.high for _, stat in points),
    )


def error_keys(artefact: Artefact, mode_block: dict) -> tuple[str, str, str]:
    """Return the model, copy and climatology keys of the declared error metric.

    Read from the mode block, falling back to the aggregate's top level where
    the mode block declares none.

    Raises:
        FigureContractError: If neither declares a metric a figure reads.
    """
    metric = mode_block.get(ERROR_METRIC_KEY) or artefact.payload.get(ERROR_METRIC_KEY)
    if metric not in CLIMATOLOGY_KEYS:
        raise FigureContractError(
            f"{artefact.cited} declares {ERROR_METRIC_KEY} {metric!r}, which no figure reads"
        )
    return model_error_key(metric), copy_error_key(metric), CLIMATOLOGY_KEYS[metric]


def forest_height(rows: int, series: int) -> float:
    """Return the height a forest panel of rows and series needs."""
    return FOREST_AXIS_HEIGHT_IN + FOREST_ROW_HEIGHT_IN * rows * max(
        1.0, series / FOREST_SERIES_PER_ROW
    )


def humanise(key: str) -> str:
    """Return an artefact key or run name with its underscores as spaces."""
    return key.replace("_", " ")


def _band(  # pylint: disable=too-many-arguments
    label: str | None,
    points: Sequence[tuple[int, float, float, float]],
    colour: str,
    line_style: str,
    alpha: float,
    *,
    hollow: bool = False,
) -> Band:
    """Build a band from (horizon, centre, low, high) points."""
    x, centre, low, high = zip(*points)
    return Band(
        label, tuple(float(value) for value in x), tuple(centre), colour,
        tuple(low), tuple(high), line_style=line_style, alpha=alpha, hollow=hollow,
    )


def arm_bands(  # pylint: disable=too-many-arguments
    curve: Curve,
    arm: int,
    *,
    colour: str | None = None,
    line_style: str = "-",
    legend: bool = True,
    split_extrapolation: bool = True,
    alpha: float = 1.0,
) -> list[Band]:
    """Return one arm's curve as bands, extrapolated horizons in their own style.

    With split_extrapolation, an arm whose every point is extrapolated draws no
    band.
    """
    colour = ARM_COLOURS[arm] if colour is None else colour
    points = list(zip(curve.horizons, curve.centre, curve.low, curve.high))
    beyond = [p for p in points if split_extrapolation and p[0] in EXTRAPOLATION_HORIZONS]
    within = [p for p in points if p not in beyond]
    bands: list[Band] = []
    if within:
        label = ARM_DISPLAY[arm] if legend else None
        bands.append(_band(label, within, colour, line_style, alpha))
    if within and beyond:
        bands.append(
            _band(None, [within[-1], *beyond], colour, EXTRAPOLATION_STYLE, alpha, hollow=True)
        )
    return bands


def stat_band(label: str, points: Sequence[tuple[float, Stat]], colour: str) -> Band:
    """Return (x, Stat) points as a line with error bars."""
    return Band(
        label,
        tuple(float(x) for x, _ in points),
        tuple(stat.iqm for _, stat in points),
        colour,
        tuple(stat.low for _, stat in points),
        tuple(stat.high for _, stat in points),
        error_bars=True,
    )


def bar_series(label: str, stats: Sequence[Stat | None], colour: str) -> BarSeries:
    """Return one bar series from a statistic per group."""
    return BarSeries(
        label,
        tuple(None if stat is None else stat.iqm for stat in stats),
        colour,
        tuple(None if stat is None else stat.low for stat in stats),
        tuple(None if stat is None else stat.high for stat in stats),
    )


def extrapolation_proxies(panels: Iterable[LinePanel | None]) -> tuple[Proxy, ...]:
    """Return the extrapolation legend entry when any panel draws an extrapolated segment."""
    hollow = any(
        band.hollow for panel in panels if panel is not None for band in panel.bands
    )
    if not hollow:
        return ()
    return (
        Proxy("Beyond the trained horizon", NEUTRAL_COLOUR, EXTRAPOLATION_STYLE, SERIES_MARKER),
    )


def arm_proxies(arms: Iterable[int]) -> tuple[Proxy, ...]:
    """Return one legend entry per arm colour."""
    return tuple(Proxy(ARM_DISPLAY[arm], ARM_COLOURS[arm]) for arm in arms)


@dataclass
class Reader:
    """One builder's view of the tree: the split it reads and what it recorded."""

    tree: OutputsTree
    split: str
    keys: set[str] = field(default_factory=set)
    omissions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def omit(self, message: str) -> None:
        """Record something the figure leaves out."""
        self.omissions.append(message)

    def with_split(self, split: str) -> Reader:
        """Return a reader of another split sharing this one's records."""
        return replace(self, split=split)

    def aggregate_for(self, condition: Condition, arm: int) -> Artefact:
        """Load one arm's aggregate for a condition."""
        return self.tree.aggregate(condition.run_for(arm), condition.env, arm, self.split)

    def aggregates(self, condition: Condition) -> dict[int, Artefact]:
        """Load every arm's aggregate a condition has, recording the absent ones."""
        found: dict[int, Artefact] = {}
        for arm in condition.arms:
            try:
                found[arm] = self.aggregate_for(condition, arm)
            except MissingArtefactError as error:
                self.omit(f"{condition.name}, arm {arm}: {error}")
        if not found:
            raise MissingArtefactError(f"{condition.name}: no arm has an aggregate")
        return found

    def arm_fit(self, condition: Condition, arm: int, mode: str) -> tuple[Artefact, dict] | None:
        """Load one arm's fit block, recording an absence or a seed shortfall."""
        try:
            artefact = self.tree.fit(condition.run_for(arm), condition.env, self.split)
            block = artefact.arm(mode, arm)
        except MissingArtefactError as error:
            self.omit(f"{condition.name}, arm {arm}, {mode}: {error}")
            return None
        count = len(block.get(SEEDS_KEY) or ())
        if count < len(SEEDS):
            self.omit(f"{artefact.cited} {mode} arm {arm}: {count} of {len(SEEDS)} seeds")
        return artefact, block

    def mode_block(self, artefact: Artefact, mode: str) -> dict | None:
        """Return an aggregate's mode block, recording an absence or a seed shortfall."""
        try:
            block = artefact.mode(mode)
        except MissingArtefactError as error:
            self.omit(str(error))
            return None
        count = block.get(N_SEEDS_KEY)
        if count is not None and count < len(SEEDS):
            self.omit(f"{artefact.cited} {mode}: {count} of {len(SEEDS)} seeds")
        return block

    def finish(self, layout: FigureLayout, table: Sequence[dict] = ()) -> FigureBuild:
        """Return the build, each omission once."""
        return FigureBuild(
            layout,
            tuple(sorted(self.keys)),
            tuple(dict.fromkeys(self.omissions)),
            tuple(dict.fromkeys(self.notes)),
            tuple(table),
        )


Builder = Callable[[OutputsTree, str], FigureBuild]


def baseline_references(
    reader: Reader, mode_block: dict, keys: tuple[str, str, str], where: str, *, binding: bool
) -> tuple[Reference, ...]:
    """Return the copy and climatology baselines, and the lower of the two when asked."""
    _, copy_key, climatology_key = keys
    copy = curve_of(mode_block, copy_key)
    climatology = curve_of(mode_block, climatology_key)
    references: list[Reference] = []
    if copy.horizons:
        references.append(Reference(
            BASELINE_DISPLAY["copy"], tuple(map(float, copy.horizons)), copy.centre,
            line_style=COPY_BASELINE_STYLE,
        ))
        reader.keys.add(copy_key)
    else:
        reader.omit(f"{where}: no {copy_key}")
    if climatology.horizons:
        references.append(Reference(
            BASELINE_DISPLAY[climatology_key], tuple(map(float, climatology.horizons)),
            climatology.centre, line_style=CLIMATOLOGY_BASELINE_STYLE,
        ))
        reader.keys.add(climatology_key)
    else:
        reader.omit(f"{where}: no {climatology_key}, so its baseline is not drawn")
    if binding and copy.horizons and climatology.horizons:
        lookup = dict(zip(climatology.horizons, climatology.centre))
        shared = [(h, min(c, lookup[h])) for h, c in zip(copy.horizons, copy.centre) if h in lookup]
        references.append(Reference(
            BASELINE_DISPLAY["binding"], tuple(float(h) for h, _ in shared),
            tuple(value for _, value in shared), colour=BINDING_BASELINE_COLOUR,
            line_style=BINDING_BASELINE_STYLE, line_width=LINE_WIDTH_PT,
        ))
    return tuple(references)


def error_panel(  # pylint: disable=too-many-arguments
    reader: Reader,
    aggregates: dict[int, Artefact],
    mode: str,
    title: str,
    *,
    binding: bool = False,
    split_extrapolation: bool = True,
    horizons: Iterable[int] | None = None,
) -> LinePanel | None:
    """Return error against horizon for each arm, over its baselines.

    Baselines come from the first arm that supplies them and the axis label from
    the last arm drawn, so every arm is assumed to declare the same error metric.
    """
    bands: list[Band] = []
    references: tuple[Reference, ...] = ()
    y_label = ""
    for arm, artefact in sorted(aggregates.items()):
        block = reader.mode_block(artefact, mode)
        if block is None:
            continue
        keys = error_keys(artefact, block)
        reader.keys.add(keys[0])
        bands.extend(arm_bands(
            curve_of(block, keys[0], horizons=horizons), arm,
            split_extrapolation=split_extrapolation,
        ))
        y_label = METRIC_DISPLAY[keys[0]]
        if not references:
            references = baseline_references(
                reader, block, keys, f"{artefact.cited} {mode}", binding=binding
            )
    if not bands:
        reader.omit(f"{humanise(title)}: no arm has an error curve")
        return None
    return LinePanel(title, tuple(bands), y_label, references=references)


def build_error_against_horizon(tree: OutputsTree, split: str) -> FigureBuild:
    """Error against horizon for the three arms on FourRooms, one panel per view."""
    reader = Reader(tree, split)
    aggregates = reader.aggregates(FOURROOMS_NO_SLIP)
    panels = tuple(
        panel for mode in OBS_MODES
        if (panel := error_panel(reader, aggregates, mode, MODE_DISPLAY[mode])) is not None
    )
    if not panels:
        raise MissingArtefactError(f"{FOURROOMS_NO_SLIP.name}: no view has an error curve")
    return reader.finish(FigureLayout(
        "", (LayoutRow(panels),), proxies=extrapolation_proxies(panels),
    ))


def _heading(prefix: str | None, representation: str, mode: str) -> str:
    """Return a panel title naming the representation and view."""
    text = f"{REPRESENTATION_DISPLAY[representation]}, {MODE_DISPLAY[mode].lower()}"
    return text if prefix is None else f"{prefix}\n{text}"


def compounding_forest(  # pylint: disable=too-many-locals,too-many-arguments
    reader: Reader,
    conditions: Sequence[Condition],
    mode: str,
    reading: str,
    title: str,
    *,
    show_row_labels: bool,
) -> ForestPanel | None:
    """Return one compounding-error reading per condition and arm, with the copy marker."""
    arms = sorted({arm for condition in conditions for arm in condition.arms})
    points: list[ForestPoint] = []
    markers: dict[int, float] = {}
    for row, condition in enumerate(conditions):
        for arm in condition.arms:
            fit = reader.arm_fit(condition, arm, mode)
            if fit is None:
                continue
            artefact, block = fit
            sigma = block.get(SIGMA_KEY)
            if not isinstance(sigma, dict) or reading not in sigma:
                raise FigureContractError(
                    f"{artefact.cited} {mode} arm {arm} carries no {SIGMA_KEY} {reading}"
                )
            stat = stat_of(sigma[reading])
            if stat is None:
                reader.omit(f"{artefact.cited} {mode} arm {arm}: {reading} has no IQM")
                continue
            points.append(ForestPoint(
                row, arms.index(arm), stat.iqm, stat.low, stat.high,
                ARM_COLOURS[arm], ARM_DISPLAY[arm],
            ))
            copy = stat_of(sigma.get(COPY_READINGS[reading]))
            if copy is not None:
                markers.setdefault(row, copy.iqm)
    reader.keys.update({reading, COPY_READINGS[reading]})
    if not points:
        return None
    return ForestPanel(
        title, tuple(condition.name for condition in conditions), len(arms), tuple(points),
        METRIC_DISPLAY[reading], log_x=True, show_row_labels=show_row_labels,
        markers=tuple(markers.items()), marker_label=BASELINE_DISPLAY["copy"],
    )


def _compounding_rows(reader: Reader, reading: str, prefix: str | None) -> list[LayoutRow]:
    """Return the symbolic row, both views, and the pixel row of one reading."""
    rows: list[LayoutRow] = []
    symbolic = tuple(
        compounding_forest(
            reader, SYMBOLIC_CONDITIONS, mode, reading,
            _heading(prefix, REPRESENTATION_SYMBOLIC, mode), show_row_labels=index == 0,
        )
        for index, mode in enumerate(OBS_MODES)
    )
    if any(panel is not None for panel in symbolic):
        rows.append(LayoutRow(symbolic, forest_height(len(SYMBOLIC_CONDITIONS), len(ARMS))))
    pixel = compounding_forest(
        reader, PIXEL_CONDITIONS, PIXEL_MODE, reading,
        _heading(prefix, REPRESENTATION_RGB, PIXEL_MODE), show_row_labels=True,
    )
    if pixel is not None:
        rows.append(LayoutRow((pixel, None), forest_height(len(PIXEL_CONDITIONS), len(PIXEL_ARMS))))
    return rows


def build_compounding_error(tree: OutputsTree, split: str) -> FigureBuild:
    """The undiscounted compounding-error integral for every condition and arm."""
    reader = Reader(tree, split)
    rows = _compounding_rows(reader, SIGMA_INTEGRAL_KEY, None)
    if not rows:
        raise MissingArtefactError("no condition has a fit artefact")
    reader.notes.append(f"reading: {READING_DISPLAY[SIGMA_INTEGRAL_KEY].lower()}, undiscounted")
    return reader.finish(FigureLayout("", tuple(rows), legend_columns=len(ARMS) + 1))


def build_compounding_alt(tree: OutputsTree, split: str) -> FigureBuild:
    """The plain sum and the discounted integral, laid out as the reported reading is."""
    reader = Reader(tree, split)
    rows = [
        row
        for reading in (SIGMA_SUM_KEY, SIGMA_DISCOUNTED_INTEGRAL_KEY)
        for row in _compounding_rows(reader, reading, READING_DISPLAY[reading])
    ]
    if not rows:
        raise MissingArtefactError("no condition has a fit artefact")
    return reader.finish(FigureLayout("", tuple(rows), legend_columns=len(ARMS) + 1))


@dataclass(frozen=True)
class Estimate:
    """One arm's reported exponent, how often it was identified, and its endpoint ratio."""

    exponent: Stat
    identified: int
    attempted: int
    ratio: Stat | None


def estimate_of(reader: Reader, condition: Condition, arm: int, mode: str) -> Estimate | None:
    """Return one arm's reported exponent and endpoint ratio from its fit."""
    fit = reader.arm_fit(condition, arm, mode)
    if fit is None:
        return None
    artefact, block = fit
    reported = block.get(REPORTED_EXPONENT_KEY)
    if not isinstance(reported, dict):
        raise FigureContractError(
            f"{artefact.cited} {mode} arm {arm} carries no {REPORTED_EXPONENT_KEY}"
        )
    reader.keys.update({REPORTED_EXPONENT_KEY, ENDPOINT_RATIO_KEY})
    exponent = stat_of(reported.get(REPORTED_VALUE_KEY))
    if exponent is None:
        reader.omit(f"{artefact.cited} {mode} arm {arm}: the reported exponent has no IQM")
        return None
    estimator = block.get(reported.get(REPORTED_SOURCE_KEY)) or {}
    return Estimate(
        exponent,
        int(estimator.get(N_IDENTIFIED_KEY, 0)),
        int(estimator.get(N_SEEDS_KEY, 0)),
        stat_of(block.get(ENDPOINT_RATIO_KEY)),
    )


def _estimate_points(
    estimate: Estimate, row: int, series: int, arm: int
) -> tuple[ForestPoint, ForestPoint | None]:
    """Return an estimate as an exponent point and, where present, a ratio point."""
    colour, label = ARM_COLOURS[arm], ARM_DISPLAY[arm]
    exponent = ForestPoint(
        row, series, estimate.exponent.iqm, estimate.exponent.low, estimate.exponent.high,
        colour, label, note=f"{estimate.identified} of {estimate.attempted}",
    )
    ratio = estimate.ratio
    if ratio is None:
        return exponent, None
    return exponent, ForestPoint(row, series, ratio.iqm, ratio.low, ratio.high, colour, label)


def ratio_axis_label() -> str:
    """Return the endpoint ratio's axis label, naming the horizons it divides."""
    denominator, numerator = REPORTED_RATIO_HORIZONS
    return f"{METRIC_DISPLAY[ENDPOINT_RATIO_KEY]}, h = {numerator} over h = {denominator}"


def _forest_pair(  # pylint: disable=too-many-arguments
    titles: tuple[str, str],
    rows: tuple[str, ...],
    series: int,
    exponents: Sequence[ForestPoint],
    ratios: Sequence[ForestPoint],
) -> LayoutRow:
    """Return an exponent forest and a ratio forest side by side."""
    exponent_panel = ForestPanel(
        titles[0], rows, series, tuple(exponents), METRIC_DISPLAY[REPORTED_EXPONENT_KEY],
        zero_line=True,
    ) if exponents else None
    ratio_panel = ForestPanel(
        titles[1], rows, series, tuple(ratios), ratio_axis_label(), log_x=True,
        show_row_labels=exponent_panel is None,
    ) if ratios else None
    return LayoutRow((exponent_panel, ratio_panel), forest_height(len(rows), series))


def build_exponent_forest(tree: OutputsTree, split: str) -> FigureBuild:
    """Reported exponent and endpoint ratio per arm on FourRooms, both views."""
    reader = Reader(tree, split)
    rows = tuple(MODE_DISPLAY[mode] for mode in OBS_MODES)
    exponents: list[ForestPoint] = []
    ratios: list[ForestPoint] = []
    for row, mode in enumerate(OBS_MODES):
        for series, arm in enumerate(FOURROOMS_NO_SLIP.arms):
            estimate = estimate_of(reader, FOURROOMS_NO_SLIP, arm, mode)
            if estimate is None:
                continue
            exponent, ratio = _estimate_points(estimate, row, series, arm)
            exponents.append(exponent)
            if ratio is not None:
                ratios.append(ratio)
    if not exponents:
        raise MissingArtefactError(f"{FOURROOMS_NO_SLIP.name}: no arm has a fit artefact")
    reader.notes.append(IDENTIFIED_NOTE)
    layout_row = _forest_pair(
        ("Fitted exponent", "Endpoint error ratio"), rows, len(FOURROOMS_NO_SLIP.arms),
        exponents, ratios,
    )
    return reader.finish(FigureLayout("", (layout_row,), legend_columns=len(ARMS)))


def _ladder_panel(reader: Reader) -> tuple[LinePanel | None, tuple[Proxy, ...]]:
    """Return the slip ladder's error curves, rungs by colour and arms by line style."""
    bands: list[Band] = []
    proxies: list[Proxy] = []
    y_label = ""
    for index, (label, condition) in enumerate(SLIP_LADDER):
        colour = LADDER_COLOURS[index]
        try:
            aggregates = reader.aggregates(condition)
        except MissingArtefactError as error:
            reader.omit(str(error))
            continue
        proxies.append(Proxy(label, colour))
        for arm, artefact in sorted(aggregates.items()):
            block = reader.mode_block(artefact, SLIP_FIGURE_MODE)
            if block is None:
                continue
            model_key = error_keys(artefact, block)[0]
            reader.keys.add(model_key)
            y_label = METRIC_DISPLAY[model_key]
            bands.extend(arm_bands(
                curve_of(block, model_key, horizons=EVALUATION_HORIZONS), arm,
                colour=colour, line_style=ARM_LINE_STYLES[arm], legend=False,
                split_extrapolation=False,
            ))
    proxies.extend(
        Proxy(ARM_DISPLAY[arm], NEUTRAL_COLOUR, ARM_LINE_STYLES[arm]) for arm in ARMS
    )
    if not bands:
        return None, ()
    return LinePanel(MODE_DISPLAY[SLIP_FIGURE_MODE], tuple(bands), y_label), tuple(proxies)


def _readout_values(block: dict, readout: str) -> tuple[float, ...]:
    """Return one arm's per-seed values of a gap readout."""
    if readout == REPORTED_EXPONENT_KEY:
        source = (block.get(REPORTED_EXPONENT_KEY) or {}).get(REPORTED_VALUE_KEY)
    elif readout == SIGMA_INTEGRAL_KEY:
        source = (block.get(SIGMA_KEY) or {}).get(SIGMA_INTEGRAL_KEY)
    else:
        source = block.get(readout)
    stat = stat_of(source)
    return () if stat is None else stat.values


def arm_gap(reader: Reader, condition: Condition, mode: str, readout: str) -> Stat | None:
    """Return arm 3 minus arm 1 on one readout, paired by seed."""
    later = reader.arm_fit(condition, ARM_AR_ONE_STEP, mode)
    earlier = reader.arm_fit(condition, ARM_DIRECT, mode)
    if later is None or earlier is None:
        return None
    (later_artefact, later_block), (earlier_artefact, earlier_block) = later, earlier
    if later_block.get(SEEDS_KEY) != earlier_block.get(SEEDS_KEY):
        raise FigureContractError(
            f"{later_artefact.cited} and {earlier_artefact.cited} list their seeds in "
            "different orders, so the arms cannot be paired"
        )
    minuend = _readout_values(later_block, readout)
    subtrahend = _readout_values(earlier_block, readout)
    if not minuend or len(minuend) != len(subtrahend):
        reader.omit(f"{condition.name}, {mode}: no paired {readout} for arms 3 and 1")
        return None
    if readout == REPORTED_EXPONENT_KEY and (
        (later_block.get(REPORTED_EXPONENT_KEY) or {}).get(REPORTED_SOURCE_KEY)
        != (earlier_block.get(REPORTED_EXPONENT_KEY) or {}).get(REPORTED_SOURCE_KEY)
    ):
        reader.omit(
            f"{condition.name}, {mode}: arms 3 and 1 report exponents from different estimators"
        )
    reader.keys.add(readout)
    return interval([a - b for a, b in zip(minuend, subtrahend)], reader.tree.reps)


def _gap_panel(reader: Reader, readout: str, title: str) -> LinePanel | None:
    """Return one readout's paired gap across the slip ladder and Dynamic-Obstacles."""
    ladder = [
        (float(position), gap)
        for position, (_, condition) in enumerate(SLIP_LADDER)
        if (gap := arm_gap(reader, condition, SLIP_FIGURE_MODE, readout)) is not None
    ]
    bands: list[Band] = []
    if ladder:
        bands.append(stat_band(SLIP_LADDER_LEGEND, ladder, BASELINE_COLOUR))
    secondary_position = len(SLIP_LADDER) - 1 + LADDER_SECONDARY_OFFSET * 2
    secondary = arm_gap(reader, DYNAMIC_OBSTACLES_UNIFORM, SLIP_FIGURE_MODE, readout)
    if secondary is not None:
        bands.append(replace(
            stat_band(DYNAMIC_OBSTACLES_UNIFORM.name, [(secondary_position, secondary)],
                      LADDER_COLOURS[len(SLIP_LADDER)]),
            line_style="none", marker=SECONDARY_MARKER,
        ))
    if not bands:
        return None
    ticks = tuple(
        (float(position), label.replace(" ", "\n", 1))
        for position, (label, _) in enumerate(SLIP_LADDER)
    )
    return LinePanel(
        title, tuple(bands), GAP_AXIS_LABEL, x_label="", log_x=False, zero_line=True,
        x_ticks=(*ticks, (secondary_position, DYNAMIC_OBSTACLES_TICK)),
    )


def build_slip_ladder(tree: OutputsTree, split: str) -> FigureBuild:
    """Error under increasing slip, and the paired arm gap on each readout."""
    reader = Reader(tree, split)
    ladder, proxies = _ladder_panel(reader)
    gaps = tuple(_gap_panel(reader, readout, title) for readout, title in GAP_READOUTS)
    rows = []
    if ladder is not None:
        rows.append(LayoutRow((ladder,)))
    if any(panel is not None for panel in gaps):
        rows.append(LayoutRow(gaps))
    if not rows:
        raise MissingArtefactError("no rung of the slip ladder has an artefact")
    reader.notes.append(f"view: {MODE_DISPLAY[SLIP_FIGURE_MODE].lower()}")
    return reader.finish(FigureLayout("", tuple(rows), proxies=proxies, legend_columns=3))


def build_pixel_error(tree: OutputsTree, split: str) -> FigureBuild:
    """Pixel error against horizon for arms 1 and 3, one panel per room."""
    reader = Reader(tree, split)
    panels: list[LinePanel] = []
    for condition in PIXEL_CONDITIONS:
        try:
            aggregates = reader.aggregates(condition)
        except MissingArtefactError as error:
            reader.omit(str(error))
            continue
        panel = error_panel(
            reader, aggregates, PIXEL_MODE, display(ENV_DISPLAY, condition.env), binding=True
        )
        if panel is not None:
            panels.append(panel)
    if not panels:
        raise MissingArtefactError("no pixel run has an aggregate")
    return reader.finish(FigureLayout(
        "", (LayoutRow(tuple(panels)),), proxies=extrapolation_proxies(panels), legend_columns=3,
    ))


def build_pixel_mover_skill(tree: OutputsTree, split: str) -> FigureBuild:
    """Mover-restricted skill on pixels, with whole-frame skill drawn faintly."""
    reader = Reader(tree, split)
    panels: list[LinePanel] = []
    for condition in PIXEL_CONDITIONS:
        try:
            aggregates = reader.aggregates(condition)
        except MissingArtefactError as error:
            reader.omit(str(error))
            continue
        bands: list[Band] = []
        for arm, artefact in sorted(aggregates.items()):
            block = reader.mode_block(artefact, PIXEL_MODE)
            if block is None:
                continue
            mover = curve_of(block, MOVER_MSE_SKILL_KEY)
            if not mover.horizons:
                reader.omit(f"{artefact.cited} {PIXEL_MODE}: no {MOVER_MSE_SKILL_KEY}")
                continue
            bands.extend(arm_bands(mover, arm))
            bands.extend(arm_bands(
                curve_of(block, SKILL_SCORE_KEY), arm, legend=False,
                split_extrapolation=False, alpha=REFERENCE_ALPHA,
            ))
        if bands:
            panels.append(LinePanel(
                display(ENV_DISPLAY, condition.env), tuple(bands),
                METRIC_DISPLAY[MOVER_MSE_SKILL_KEY], zero_line=True,
            ))
    reader.keys.update({MOVER_MSE_SKILL_KEY, SKILL_SCORE_KEY})
    if not panels:
        raise MissingArtefactError(f"no pixel run carries {MOVER_MSE_SKILL_KEY}")
    proxies = (
        *extrapolation_proxies(panels),
        Proxy("Whole-frame copy-normalised skill, faint", NEUTRAL_COLOUR),
    )
    return reader.finish(FigureLayout(
        "", (LayoutRow(tuple(panels)),), proxies=proxies, legend_columns=3,
    ))


def build_long_horizon(tree: OutputsTree, split: str) -> FigureBuild:
    """Error to the longest horizon on the long-horizon run, arms 1 and 3."""
    reader = Reader(tree, split)
    aggregates = reader.aggregates(LONG_HORIZON)
    panel = error_panel(
        reader, aggregates, ATARI_MODE, ATARI_LONG_HORIZON_GAME, binding=True,
        split_extrapolation=False,
    )
    if panel is None:
        raise MissingArtefactError(f"{LONG_HORIZON.name}: no arm has an error curve")
    for arm in LONG_HORIZON.arms:
        fit = reader.arm_fit(LONG_HORIZON, arm, ATARI_MODE)
        sigma = None if fit is None else sigma_stat(fit[1], SIGMA_INTEGRAL_KEY)
        if sigma is not None:
            reader.notes.append(
                f"{ARM_DISPLAY[arm]}: undiscounted compounding error {sigma.iqm:.4g} "
                f"[{sigma.low:.4g}, {sigma.high:.4g}]"
            )
    reader.keys.add(SIGMA_INTEGRAL_KEY)
    return reader.finish(FigureLayout("", (LayoutRow((panel,)),), legend_columns=3))


def build_room_policy_exponents(tree: OutputsTree, split: str) -> FigureBuild:  # pylint: disable=too-many-locals
    """Reported exponent and endpoint ratio by room and collection policy, arms 1 and 3."""
    reader = Reader(tree, split)
    rows = tuple(condition.room_policy for condition in ROOM_POLICY_CONDITIONS)
    layout_rows: list[LayoutRow] = []
    for mode in OBS_MODES:
        exponents: list[ForestPoint] = []
        ratios: list[ForestPoint] = []
        for row, condition in enumerate(ROOM_POLICY_CONDITIONS):
            for series, arm in enumerate(PIXEL_ARMS):
                estimate = estimate_of(reader, condition, arm, mode)
                if estimate is None:
                    continue
                exponent, ratio = _estimate_points(estimate, row, series, arm)
                exponents.append(exponent)
                if ratio is not None:
                    ratios.append(ratio)
        if exponents:
            view = MODE_DISPLAY[mode].lower()
            layout_rows.append(_forest_pair(
                (f"Fitted exponent, {view}", f"Endpoint error ratio, {view}"), rows,
                len(PIXEL_ARMS), exponents, ratios,
            ))
    if not layout_rows:
        raise MissingArtefactError("no room or policy condition has a fit artefact")
    reader.notes.append(IDENTIFIED_NOTE)
    return reader.finish(FigureLayout("", tuple(layout_rows)))


def build_learnability_bars(tree: OutputsTree, split: str) -> FigureBuild:  # pylint: disable=too-many-locals
    """Arm 1 against the copy baseline at short horizons, and its mover-restricted accuracy."""
    reader = Reader(tree, split)
    arm = ARM_DIRECT
    artefact = reader.aggregate_for(FOURROOMS_NO_SLIP, arm)
    groups: list[str] = []
    model: list[Stat | None] = []
    copy: list[Stat | None] = []
    mover: list[Stat | None] = []
    y_label = ""
    for mode in OBS_MODES:
        block = reader.mode_block(artefact, mode)
        if block is None:
            continue
        model_key, copy_key, _ = error_keys(artefact, block)
        reader.keys.update({model_key, copy_key, MOVER_ACCURACY_KEY})
        y_label = METRIC_DISPLAY[model_key]
        for horizon in P1_HORIZONS:
            per = (block.get(PER_HORIZON_KEY) or {}).get(str(horizon)) or {}
            groups.append(f"{MODE_DISPLAY[mode]}, h = {horizon}")
            model.append(stat_of(per.get(model_key)))
            copy.append(stat_of(per.get(copy_key)))
            mover.append(stat_of(per.get(MOVER_ACCURACY_KEY)))
    if not groups:
        raise MissingArtefactError(f"{artefact.cited} carries no view")
    margin = f"Pre-registered margin, {round(100 * P1_MOVER_MARGIN)} points"
    panels = (
        BarPanel(
            "Model against the copy baseline", tuple(groups),
            (bar_series(ARM_DISPLAY[arm], model, ARM_COLOURS[arm]),
             bar_series(BASELINE_DISPLAY["copy"], copy, NEUTRAL_COLOUR)),
            y_label, tick_rotation=GROUP_LABEL_ROTATION_DEG,
        ),
        BarPanel(
            METRIC_DISPLAY[MOVER_ACCURACY_KEY], tuple(groups),
            (bar_series(ARM_DISPLAY[arm], mover, ARM_COLOURS[arm]),),
            METRIC_DISPLAY[MOVER_ACCURACY_KEY], rules=((P1_MOVER_MARGIN, margin),),
            tick_rotation=GROUP_LABEL_ROTATION_DEG,
        ),
    )
    return reader.finish(FigureLayout("", (LayoutRow(panels),), legend_columns=3))


def _mode_gap_band(reader: Reader, artefact: Artefact) -> Band | None:
    """Return arm 1's top-down minus egocentric skill score, paired by seed."""
    blocks = [reader.mode_block(artefact, mode) for mode in OBS_MODES]
    if any(block is None for block in blocks):
        return None
    top_down, egocentric = blocks
    if top_down.get(SEEDS_KEY) != egocentric.get(SEEDS_KEY):
        raise FigureContractError(
            f"{artefact.cited} lists different seeds in its two views, so they cannot be paired"
        )
    points: list[tuple[float, Stat]] = []
    for horizon in EVALUATION_HORIZONS:
        values = [
            horizon_stat(block, horizon, SKILL_SCORE_KEY)
            for block in (top_down, egocentric)
        ]
        if any(value is None for value in values):
            continue
        gap = interval(
            [a - b for a, b in zip(values[0].values, values[1].values)], reader.tree.reps
        )
        if gap is not None:
            points.append((float(horizon), gap))
    if not points:
        return None
    band = stat_band(ARM_DISPLAY[ARM_DIRECT], points, ARM_COLOURS[ARM_DIRECT])
    return replace(band, error_bars=False)


def build_mode_gap(tree: OutputsTree, split: str) -> FigureBuild:
    """Skill score in both views for every arm, and arm 1's gap between the views."""
    reader = Reader(tree, split)
    aggregates = reader.aggregates(FOURROOMS_NO_SLIP)
    bands: list[Band] = []
    for arm, artefact in sorted(aggregates.items()):
        for mode in OBS_MODES:
            block = reader.mode_block(artefact, mode)
            if block is None:
                continue
            bands.extend(arm_bands(
                curve_of(block, SKILL_SCORE_KEY, horizons=EVALUATION_HORIZONS), arm,
                line_style=MODE_LINE_STYLES[mode], legend=False, split_extrapolation=False,
            ))
    reader.keys.add(SKILL_SCORE_KEY)
    if not bands:
        raise MissingArtefactError(f"{FOURROOMS_NO_SLIP.name}: no arm carries {SKILL_SCORE_KEY}")
    gap = None if ARM_DIRECT not in aggregates else _mode_gap_band(reader, aggregates[ARM_DIRECT])
    gap_panel = None if gap is None else LinePanel(
        "Arm 1, top-down minus egocentric", (gap,), "Difference in skill score", zero_line=True,
    )
    panels = (
        LinePanel("Both views", tuple(bands), METRIC_DISPLAY[SKILL_SCORE_KEY], zero_line=True),
        gap_panel,
    )
    proxies = (
        *arm_proxies(sorted(aggregates)),
        *(Proxy(MODE_DISPLAY[mode], NEUTRAL_COLOUR, MODE_LINE_STYLES[mode]) for mode in OBS_MODES),
    )
    return reader.finish(FigureLayout("", (LayoutRow(panels),), proxies=proxies, legend_columns=3))


def _displacement_tables(tree: OutputsTree) -> dict[str, list[dict[str, str]]]:
    """Return one displacement table per environment, the first in sorted path order."""
    paths = sorted(
        path
        for pattern in DISPLACEMENT_GLOBS
        for path in tree.root.glob(pattern)
        if path.relative_to(tree.root).parts[0] not in EXCLUDED_TREES
    )
    tables: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle), None)
        if first is None or first.get(DISPLACEMENT_ENVIRONMENT_COLUMN) in tables:
            continue
        tables[first[DISPLACEMENT_ENVIRONMENT_COLUMN]] = tree.load_rows(path)
    return tables


def build_layouts(tree: OutputsTree, split: str) -> FigureBuild:
    """Displacement against horizon for every layout measured, one panel per view."""
    reader = Reader(tree, split)
    tables = _displacement_tables(tree)
    if not tables:
        raise MissingArtefactError(f"no {DISPLACEMENT_CSV_FILENAME} under {OUTPUTS_DIR.name}")
    panels: list[LinePanel] = []
    for mode in OBS_MODES:
        bands: list[Band] = []
        for index, environment in enumerate(sorted(tables)):
            selected = sorted(
                (
                    row for row in tables[environment]
                    if row[DISPLACEMENT_MODE_COLUMN] == mode
                    and row[DISPLACEMENT_MEASURE_COLUMN] == DISPLACEMENT_MEASURE
                    and row[DISPLACEMENT_CHANNEL_COLUMN] == DISPLACEMENT_CHANNEL
                ),
                key=lambda row: int(row[DISPLACEMENT_HORIZON_COLUMN]),
            )
            if selected:
                bands.append(Band(
                    display(ENV_DISPLAY, environment),
                    tuple(float(row[DISPLACEMENT_HORIZON_COLUMN]) for row in selected),
                    tuple(100.0 * float(row[DISPLACEMENT_MEAN_COLUMN]) for row in selected),
                    LADDER_COLOURS[index % len(LADDER_COLOURS)], marker=None,
                ))
        if bands:
            panels.append(LinePanel(
                MODE_DISPLAY[mode], tuple(bands), METRIC_DISPLAY[DISPLACEMENT_MEASURE]
            ))
    reader.keys.add(DISPLACEMENT_MEASURE)
    if not panels:
        raise MissingArtefactError(f"no {DISPLACEMENT_MEASURE} rows in any displacement table")
    return reader.finish(FigureLayout("", (LayoutRow(tuple(panels)),), legend_columns=3))


def _residual_panel(reader: Reader, arm: int, mode: str) -> ScatterPanel | None:
    """Return one arm's fit residuals per estimator, every seed and horizon."""
    fit = reader.arm_fit(FOURROOMS_NO_SLIP, arm, mode)
    if fit is None:
        return None
    artefact, block = fit
    residuals = block.get(RESIDUALS_KEY)
    if not isinstance(residuals, dict):
        raise FigureContractError(f"{artefact.cited} {mode} arm {arm} carries no {RESIDUALS_KEY}")
    series: list[ScatterSeries] = []
    for estimator in ESTIMATOR_KEYS:
        points = [
            (int(horizon) * ESTIMATOR_JITTER[estimator], float(value))
            for per_seed in (residuals.get(estimator) or {}).values()
            for horizon, value in (per_seed or {}).items()
            if value is not None
        ]
        if points:
            series.append(ScatterSeries(
                ESTIMATOR_DISPLAY[estimator], tuple(x for x, _ in points),
                tuple(y for _, y in points), ESTIMATOR_COLOURS[estimator],
            ))
    if not series:
        return None
    return ScatterPanel(
        f"{ARM_SHORT_DISPLAY[arm]}, {MODE_DISPLAY[mode].lower()}", tuple(series),
        METRIC_DISPLAY["residual"],
    )


def build_residuals(tree: OutputsTree, split: str) -> FigureBuild:
    """Fit residuals per estimator on FourRooms, one panel per arm and view."""
    reader = Reader(tree, split)
    rows = tuple(
        LayoutRow(tuple(_residual_panel(reader, arm, mode) for arm in FOURROOMS_NO_SLIP.arms))
        for mode in OBS_MODES
    )
    reader.keys.add(RESIDUALS_KEY)
    if all(panel is None for row in rows for panel in row.panels):
        raise MissingArtefactError(f"{FOURROOMS_NO_SLIP.name}: no arm has fit residuals")
    return reader.finish(FigureLayout("", rows, legend_columns=3))


def build_estimator_panel(tree: OutputsTree, split: str) -> FigureBuild:  # pylint: disable=too-many-locals
    """How many seeds each estimator is identified on, per condition, arm and view."""
    reader = Reader(tree, split)
    columns = [(mode, estimator) for mode in OBS_MODES for estimator in ESTIMATOR_KEYS]
    labels: list[str] = []
    values: list[tuple[float | None, ...]] = []
    annotations: list[tuple[str, ...]] = []
    for condition in (*SYMBOLIC_CONDITIONS, *PIXEL_CONDITIONS):
        for arm in condition.arms:
            fits = {
                mode: reader.arm_fit(condition, arm, mode) for mode in condition.modes
            }
            cells: list[tuple[float | None, str]] = []
            for mode, estimator in columns:
                fit = fits.get(mode)
                block = None if fit is None else fit[1].get(estimator)
                attempted = int(block.get(N_SEEDS_KEY, 0)) if isinstance(block, dict) else 0
                if attempted:
                    identified = int(block.get(N_IDENTIFIED_KEY, 0))
                    cells.append((identified / attempted, f"{identified}/{attempted}"))
                else:
                    cells.append((None, ""))
            if any(value is not None for value, _ in cells):
                labels.append(f"{condition.name}, {ARM_SHORT_DISPLAY[arm]}")
                values.append(tuple(value for value, _ in cells))
                annotations.append(tuple(text for _, text in cells))
    reader.keys.update({N_IDENTIFIED_KEY, *ESTIMATOR_KEYS})
    if not labels:
        raise MissingArtefactError("no condition has a fit artefact")
    panel = HeatmapPanel(
        "", tuple(labels),
        tuple(
            f"{MODE_DISPLAY[mode]}, {ESTIMATOR_DISPLAY[estimator].lower()}"
            for mode, estimator in columns
        ),
        tuple(values), tuple(annotations),
    )
    height = HEATMAP_AXIS_HEIGHT_IN + HEATMAP_ROW_HEIGHT_IN * len(labels)
    return reader.finish(FigureLayout("", (LayoutRow((panel,), height),), legend=False))


def build_test_vs_validation(tree: OutputsTree, split: str) -> FigureBuild:
    """FourRooms error against horizon on both splits, one row per view."""
    reader = Reader(tree, split)
    rows: list[LayoutRow] = []
    for mode in OBS_MODES:
        panels: list[LinePanel | None] = []
        for split_name in (VALIDATION_SPLIT, REPORTED_SPLIT):
            split_reader = reader.with_split(split_name)
            try:
                aggregates = split_reader.aggregates(FOURROOMS_NO_SLIP)
            except MissingArtefactError as error:
                reader.omit(str(error))
                panels.append(None)
                continue
            panels.append(error_panel(
                split_reader, aggregates, mode,
                f"{display(SPLIT_DISPLAY, split_name)}, {MODE_DISPLAY[mode].lower()}",
                split_extrapolation=False, horizons=EVALUATION_HORIZONS,
            ))
        rows.append(LayoutRow(tuple(panels)))
    if all(panel is None for row in rows for panel in row.panels):
        raise MissingArtefactError(f"{FOURROOMS_NO_SLIP.name}: neither split has an aggregate")
    return reader.finish(FigureLayout("", tuple(rows), legend_columns=3))


def build_per_horizon_bars(tree: OutputsTree, split: str) -> FigureBuild:  # pylint: disable=too-many-locals
    """Model error and mover-restricted accuracy at chosen horizons, every arm."""
    reader = Reader(tree, split)
    aggregates = reader.aggregates(FOURROOMS_NO_SLIP)
    groups = tuple(f"h = {horizon}" for horizon in SUPPLEMENTARY_HORIZONS)
    rows: list[LayoutRow] = []
    for mode in OBS_MODES:
        errors: list[BarSeries] = []
        movers: list[BarSeries] = []
        y_label = ""
        for arm, artefact in sorted(aggregates.items()):
            block = reader.mode_block(artefact, mode)
            if block is None:
                continue
            model_key = error_keys(artefact, block)[0]
            reader.keys.update({model_key, MOVER_ACCURACY_KEY})
            y_label = METRIC_DISPLAY[model_key]
            per = block.get(PER_HORIZON_KEY) or {}
            at = [per.get(str(horizon)) or {} for horizon in SUPPLEMENTARY_HORIZONS]
            errors.append(bar_series(
                ARM_DISPLAY[arm], [stat_of(block_h.get(model_key)) for block_h in at],
                ARM_COLOURS[arm],
            ))
            movers.append(bar_series(
                ARM_DISPLAY[arm], [stat_of(block_h.get(MOVER_ACCURACY_KEY)) for block_h in at],
                ARM_COLOURS[arm],
            ))
        if errors:
            view = MODE_DISPLAY[mode].lower()
            rows.append(LayoutRow((
                BarPanel(f"Model error, {view}", groups, tuple(errors), y_label),
                BarPanel(
                    f"Mover-restricted accuracy, {view}", groups, tuple(movers),
                    METRIC_DISPLAY[MOVER_ACCURACY_KEY],
                ),
            )))
    if not rows:
        raise MissingArtefactError(f"{FOURROOMS_NO_SLIP.name}: no view has an aggregate")
    return reader.finish(FigureLayout("", tuple(rows), legend_columns=3))


def _saturation_band(  # pylint: disable=too-many-arguments,too-many-locals
    reader: Reader,
    table: list[dict],
    condition: Condition,
    arm: int,
    *,
    mode: str,
    row: tuple[str, str, tuple[int, ...]],
) -> Band | None:
    """Return one condition's curve of one saturation metric, recording each plotted value."""
    name, key, _ = row
    try:
        artefact = reader.aggregate_for(condition, arm)
    except MissingArtefactError as error:
        reader.omit(f"{condition.name}, arm {arm}: {error}")
        return None
    block = reader.mode_block(artefact, mode)
    if block is None:
        return None
    curve = curve_of(block, key)
    if not curve.horizons:
        reader.omit(f"{artefact.cited} {mode}: no {key}")
        return None
    reader.keys.add(key)
    model_row = name == SATURATION_ROWS[-1][0]
    for horizon, centre, low, high in zip(curve.horizons, curve.centre, curve.low, curve.high):
        values = (
            name, mode, condition.room, condition.policy, arm if model_row else "",
            artefact.cited, key, horizon, centre, low, high,
        )
        table.append(dict(zip(SATURATION_TABLE_COLUMNS, values)))
    return Band(
        None, tuple(map(float, curve.horizons)), curve.centre, ROOM_COLOURS[condition.room],
        curve.low, curve.high, line_style=POLICY_LINE_STYLES[condition.policy],
        marker=ARM_MARKERS[arm] if model_row else None,
    )


def build_data_saturation(tree: OutputsTree, split: str) -> FigureBuild:  # pylint: disable=too-many-locals
    """State change in the data and mover-restricted accuracy of each model, a row each."""
    reader = Reader(tree, split)
    table: list[dict] = []
    rows: list[LayoutRow] = []
    for row in SATURATION_ROWS:
        name, key, arms = row
        panels: list[LinePanel | None] = []
        for mode in OBS_MODES:
            bands = [
                band
                for condition in SATURATION_CONDITIONS for arm in arms
                if (band := _saturation_band(reader, table, condition, arm, mode=mode, row=row))
                is not None
            ]
            panels.append(LinePanel(
                f"{name}, {MODE_DISPLAY[mode].lower()}", tuple(bands), METRIC_DISPLAY[key]
            ) if bands else None)
        if any(panel is not None for panel in panels):
            rows.append(LayoutRow(tuple(panels), COMPACT_ROW_HEIGHT_IN))
    if not table:
        raise MissingArtefactError("no room or policy condition carries the saturation metrics")
    reader.notes.append(STATE_CHANGE_NOTE)
    reader.omit(PIXEL_SATURATION_ABSENT)
    rooms = dict.fromkeys(condition.room for condition in SATURATION_CONDITIONS)
    policies = dict.fromkeys(condition.policy for condition in SATURATION_CONDITIONS)
    proxies = (
        *(Proxy(room, ROOM_COLOURS[room]) for room in rooms),
        *(Proxy(POLICY_DISPLAY[policy], NEUTRAL_COLOUR, POLICY_LINE_STYLES[policy])
          for policy in policies),
        *(Proxy(ARM_DISPLAY[arm], NEUTRAL_COLOUR, "none", ARM_MARKERS[arm])
          for arm in SATURATION_MODEL_ARMS),
    )
    layout = FigureLayout("", tuple(rows), proxies=proxies, legend_columns=4)
    return reader.finish(layout, table)


def _ranking_position(artefact: Artefact) -> int:
    """Return the archive position a ranking was measured at."""
    position = artefact.payload.get(RANKING_POSITION_KEY)
    if not isinstance(position, int):
        raise FigureContractError(f"{artefact.cited} carries no {RANKING_POSITION_KEY}")
    return position


def build_atari_displacement(tree: OutputsTree, split: str) -> FigureBuild:
    """Displacement of every Atari game at each archive position, with both gates."""
    reader = Reader(tree, split)
    rankings = sorted(
        (tree.load_json(path) for path in sorted(tree.atari_results.glob(ATARI_RANKING_GLOB))),
        key=_ranking_position,
    )
    if not rankings:
        raise MissingArtefactError(
            f"no {ATARI_RANKING_GLOB} under {tree.cite(tree.atari_results)}"
        )
    entries = [
        {entry[RANKING_GAME_KEY]: entry for entry in ranking.payload.get(RANKING_GAMES_KEY) or ()}
        for ranking in rankings
    ]
    order = sorted(
        entries[0],
        key=lambda game: -float(entries[0][game].get(RANKING_DISPLACEMENT_KEY) or 0.0),
    )
    order += sorted({game for by_game in entries[1:] for game in by_game} - set(order))
    series = tuple(
        BarSeries(
            f"Archive position {_ranking_position(ranking)}",
            tuple(
                by_game[game].get(RANKING_DISPLACEMENT_KEY) if game in by_game else None
                for game in order
            ),
            LADDER_COLOURS[index % len(LADDER_COLOURS)],
            hatched=tuple(
                game in by_game and not by_game[game].get(RANKING_LENGTH_GATE_KEY, True)
                for game in order
            ),
        )
        for index, (ranking, by_game) in enumerate(zip(rankings, entries))
    )
    threshold = rankings[0].payload.get(RANKING_BAR_KEY)
    rules = () if threshold is None else (
        (float(threshold), f"Displacement bar, {float(threshold):g} per cent"),
    )
    reader.keys.update({RANKING_DISPLACEMENT_KEY, RANKING_LENGTH_GATE_KEY})
    reader.notes.append(
        f"selected at position {_ranking_position(rankings[0])} by rank order: "
        f"{', '.join(ATARI_GAMES)}"
    )
    panel = BarPanel(
        "", tuple(order), series, METRIC_DISPLAY[RANKING_DISPLACEMENT_KEY], rules=rules,
        tick_rotation=TICK_ROTATION_DEG,
    )
    return reader.finish(FigureLayout(
        "", (LayoutRow((panel,), SELECTION_ROW_HEIGHT_IN),),
        proxies=(Proxy("Fails the episode length gate", BASELINE_COLOUR, hatch=HATCH_PATTERN),),
        legend_columns=len(rankings) + 2,
    ))


def build_best_vs_final(tree: OutputsTree, split: str) -> FigureBuild:
    """The error gap between the selected and the final parameter tree, every arm."""
    reader = Reader(tree, split)
    aggregates = reader.aggregates(FOURROOMS_NO_SLIP)
    panels: list[LinePanel] = []
    for mode in OBS_MODES:
        bands: list[Band] = []
        y_label = ""
        for arm, artefact in sorted(aggregates.items()):
            block = reader.mode_block(artefact, mode)
            if block is None:
                continue
            model_key = error_keys(artefact, block)[0]
            reader.keys.update({model_key, PER_HORIZON_GAP_KEY})
            curve = curve_of(block, model_key, section=PER_HORIZON_GAP_KEY)
            if not curve.horizons:
                reader.omit(f"{artefact.cited} {mode}: no {PER_HORIZON_GAP_KEY}")
                continue
            bands.extend(arm_bands(curve, arm))
            y_label = f"{METRIC_DISPLAY[model_key]},\n{METRIC_DISPLAY[PER_HORIZON_GAP_KEY].lower()}"
        if bands:
            panels.append(LinePanel(MODE_DISPLAY[mode], tuple(bands), y_label, zero_line=True))
    if not panels:
        raise MissingArtefactError(
            f"{FOURROOMS_NO_SLIP.name}: no aggregate carries {PER_HORIZON_GAP_KEY}"
        )
    return reader.finish(FigureLayout(
        "", (LayoutRow(tuple(panels)),), proxies=extrapolation_proxies(panels), legend_columns=3,
    ))


def build_atari_error(tree: OutputsTree, split: str) -> FigureBuild:
    """Atari error against horizon at archive position 0, the games pooled."""
    reader = Reader(tree, split)
    position, condition = ATARI_LADDER[0]
    aggregates = reader.aggregates(condition)
    panel = error_panel(
        reader, aggregates, ATARI_MODE,
        f"{len(ATARI_GAMES)} games pooled, archive position {position}", binding=True,
    )
    if panel is None:
        raise MissingArtefactError(f"{condition.name}: no arm has an error curve")
    reader.omit(f"per-game panels are not drawn: {PER_GAME_ABSENT}")
    return reader.finish(FigureLayout(
        "", (LayoutRow((panel,)),), proxies=extrapolation_proxies((panel,)), legend_columns=1,
    ))


def build_horizon_ablation(tree: OutputsTree, split: str) -> FigureBuild:  # pylint: disable=too-many-locals
    """Arm 1's error against horizon for every trained horizon."""
    reader = Reader(tree, split)
    bands: list[Band] = []
    lines: list[tuple[float, str]] = []
    y_label = ""
    for index, (trained, run) in enumerate(HORIZON_ABLATION):
        colour = LADDER_COLOURS[index % len(LADDER_COLOURS)]
        try:
            artefact = tree.aggregate(run, ATARI_LONG_HORIZON_ENV_NAME, ARM_DIRECT, split)
        except MissingArtefactError as error:
            reader.omit(str(error))
            continue
        block = reader.mode_block(artefact, ATARI_MODE)
        if block is None:
            continue
        model_key = error_keys(artefact, block)[0]
        reader.keys.add(model_key)
        y_label = METRIC_DISPLAY[model_key]
        curve = curve_of(block, model_key)
        if curve.horizons:
            points = list(zip(curve.horizons, curve.centre, curve.low, curve.high))
            bands.append(_band(f"Trained to h = {trained}", points, colour, "-", 1.0))
            lines.append((float(trained), colour))
    if not bands:
        raise MissingArtefactError("no horizon-ablation run has an aggregate")
    panel = LinePanel(
        f"{ATARI_LONG_HORIZON_GAME}, {ARM_DISPLAY[ARM_DIRECT]}", tuple(bands), y_label,
        log_y=True, vertical_lines=tuple(lines),
    )
    return reader.finish(FigureLayout("", (LayoutRow((panel,)),), legend_columns=len(bands)))


def build_checkpoint_ladder(tree: OutputsTree, split: str) -> FigureBuild:  # pylint: disable=too-many-locals
    """Error at the reported horizon and compounding error against archive position."""
    reader = Reader(tree, split)
    horizon = REPORTED_RATIO_HORIZONS[1]
    error_bands: list[Band] = []
    sigma_bands: list[Band] = []
    y_label = ""
    for arm in PIXEL_ARMS:
        errors: list[tuple[float, Stat]] = []
        sigmas: list[tuple[float, Stat]] = []
        for position, condition in ATARI_LADDER:
            try:
                artefact = reader.aggregate_for(condition, arm)
            except MissingArtefactError as error:
                reader.omit(str(error))
                continue
            block = reader.mode_block(artefact, ATARI_MODE)
            if block is not None:
                model_key = error_keys(artefact, block)[0]
                reader.keys.add(model_key)
                y_label = METRIC_DISPLAY[model_key]
                stat = horizon_stat(block, horizon, model_key)
                if stat is not None:
                    errors.append((float(position), stat))
            fit = reader.arm_fit(condition, arm, ATARI_MODE)
            sigma = None if fit is None else sigma_stat(fit[1], SIGMA_INTEGRAL_KEY)
            if sigma is not None:
                sigmas.append((float(position), sigma))
        if errors:
            error_bands.append(stat_band(ARM_DISPLAY[arm], errors, ARM_COLOURS[arm]))
        if sigmas:
            sigma_bands.append(stat_band(ARM_DISPLAY[arm], sigmas, ARM_COLOURS[arm]))
    reader.keys.add(SIGMA_INTEGRAL_KEY)
    if not error_bands and not sigma_bands:
        raise MissingArtefactError("no archive position has an aggregate")
    ticks = tuple((float(position), str(position)) for position, _ in ATARI_LADDER)
    panels = (
        LinePanel(
            f"Error at h = {horizon}", tuple(error_bands), y_label,
            x_label=METRIC_DISPLAY["archive_position"], log_x=False, x_ticks=ticks,
        ) if error_bands else None,
        LinePanel(
            READING_DISPLAY[SIGMA_INTEGRAL_KEY], tuple(sigma_bands),
            METRIC_DISPLAY[SIGMA_INTEGRAL_KEY],
            x_label=METRIC_DISPLAY["archive_position"], log_x=False, x_ticks=ticks,
        ) if sigma_bands else None,
    )
    reader.omit(f"per-game panels are not drawn: {PER_GAME_ABSENT}")
    return reader.finish(FigureLayout("", (LayoutRow(panels),)))


def build_skill_across_representations(tree: OutputsTree, split: str) -> FigureBuild:
    """Copy-normalised skill on symbolic and pixel observations, one panel per room."""
    reader = Reader(tree, split)
    panels: list[LinePanel] = []
    for symbolic, pixel in REPRESENTATION_PAIRS:
        bands: list[Band] = []
        for condition in (symbolic, pixel):
            try:
                aggregates = reader.aggregates(condition)
            except MissingArtefactError as error:
                reader.omit(str(error))
                continue
            for arm, artefact in sorted(aggregates.items()):
                block = reader.mode_block(artefact, PIXEL_MODE)
                if block is None:
                    continue
                bands.extend(arm_bands(
                    curve_of(block, SKILL_SCORE_KEY, horizons=EVALUATION_HORIZONS), arm,
                    line_style=REPRESENTATION_LINE_STYLES[condition.representation],
                    legend=False, split_extrapolation=False,
                ))
        if bands:
            panels.append(LinePanel(
                display(ENV_DISPLAY, symbolic.env), tuple(bands),
                METRIC_DISPLAY[SKILL_SCORE_KEY], zero_line=True,
            ))
    reader.keys.add(SKILL_SCORE_KEY)
    if not panels:
        raise MissingArtefactError("no matched symbolic and pixel pair has an aggregate")
    proxies = (
        *arm_proxies(ARMS),
        *(
            Proxy(REPRESENTATION_DISPLAY[name], NEUTRAL_COLOUR, REPRESENTATION_LINE_STYLES[name])
            for name in (REPRESENTATION_SYMBOLIC, REPRESENTATION_RGB)
        ),
    )
    reader.notes.append(f"view: {MODE_DISPLAY[PIXEL_MODE].lower()}")
    return reader.finish(FigureLayout(
        "", (LayoutRow(tuple(panels)),), proxies=proxies, legend_columns=3,
    ))


@dataclass(frozen=True)
class DiscoveredRun:
    """A run directory under outputs/ holding at least one per-arm aggregate."""

    run: str
    env: str
    arms: tuple[int, ...]


def discover_runs(root: Path) -> tuple[DiscoveredRun, ...]:
    """Return every run and environment holding an aggregate, excluding non-run trees."""
    found: list[DiscoveredRun] = []
    if not root.is_dir():
        return ()
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if run_dir.name in EXCLUDED_TREES:
            continue
        for env_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
            arms = sorted(
                int(match.group(1))
                for path in env_dir.glob(AGGREGATE_FILENAME_TEMPLATE.format(arm="*"))
                if (match := AGGREGATE_ARM_PATTERN.fullmatch(path.name))
            )
            if arms:
                found.append(DiscoveredRun(run_dir.name, env_dir.name, tuple(arms)))
    return tuple(found)


def development_layouts(tree: OutputsTree, run: DiscoveredRun) -> list[tuple[str, FigureLayout]]:
    """Return one layout per view and per scalar metric a discovered run carries."""
    loaded = {
        arm: tree.load_json(
            tree.root / run.run / run.env / AGGREGATE_FILENAME_TEMPLATE.format(arm=arm)
        )
        for arm in run.arms
    }
    modes = sorted({
        mode for artefact in loaded.values() for mode in artefact.payload.get(BY_MODE_KEY) or {}
    })
    layouts: list[tuple[str, FigureLayout]] = []
    for mode in modes:
        blocks = {
            arm: (artefact.payload.get(BY_MODE_KEY) or {}).get(mode) or {}
            for arm, artefact in loaded.items()
        }
        metrics = sorted({
            key
            for block in blocks.values()
            for per in (block.get(PER_HORIZON_KEY) or {}).values()
            for key, value in (per or {}).items()
            if stat_of(value) is not None
        })
        for metric in metrics:
            bands = [
                band for arm, block in sorted(blocks.items())
                for band in arm_bands(curve_of(block, metric), arm)
            ]
            if not bands:
                continue
            view = MODE_DISPLAY.get(mode, humanise(mode))
            panel = LinePanel(
                ENV_DISPLAY.get(run.env, humanise(run.env)), tuple(bands), humanise(metric)
            )
            layouts.append((f"{mode}.{metric}", FigureLayout(
                f"{humanise(run.run)}, {view.lower()}, {humanise(metric)}", (LayoutRow((panel,)),),
            )))
    return layouts
