"""The intermediate-state figure: how far an autoregressive rollout moves per step.

Each condition's held-out summary sits in its run directory, and a second truth
source, when present, sits in one subdirectory of it. At most one second-source
directory is read per run, and its row count, seeds and horizons are checked
before any row is used.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from config import (
    ARM_AR_ENDPOINT,
    EVALUATION_HORIZONS,
    INTERMEDIATE_STATE_ARMS,
    INTERMEDIATE_STATES_SUMMARY_FILENAME,
    OBS_MODE_EGOCENTRIC,
    OBS_MODES,
    SEEDS,
    TRUTH_SOURCE_FRESH_ROLLOUT,
    TRUTH_SOURCE_TEST_TRAJECTORY,
    TRUTH_SOURCES,
)
from src.eval.figure_panels import (
    REPORTED_SPLIT,
    SYMBOLIC_CONDITIONS,
    Artefact,
    Condition,
    FigureBuild,
    FigureContractError,
    MissingArtefactError,
    OutputsTree,
    Reader,
    interval,
)
from src.eval.figure_style import (
    ARM_COLOURS,
    ARM_DISPLAY,
    COMPACT_ROW_HEIGHT_IN,
    HORIZON_SHORT_AXIS_LABEL,
    METRIC_DISPLAY,
    MODE_DISPLAY,
    NEUTRAL_COLOUR,
    SOURCE_DISPLAY,
    SOURCE_LINE_STYLES,
)
from src.eval.plots import Band, FigureLayout, LayoutRow, LinePanel, Proxy

# The view drawn, the summary field each figure draws, and the conditions per
# row of panels. The decoded distance squares differences of class codes, so its
# size follows the class numbering and it is drawn as its own figure rather than
# beside a metric that does not.
INTERMEDIATE_FIGURE_MODE: str = OBS_MODE_EGOCENTRIC
INTERMEDIATE_FIELDS: tuple[str, ...] = ("token_distance_mean", "decoded_distance_mean")
INTERMEDIATE_PRIMARY_FIELDS: tuple[str, ...] = INTERMEDIATE_FIELDS[:1]
INTERMEDIATE_SECONDARY_FIELDS: tuple[str, ...] = INTERMEDIATE_FIELDS[1:]
CONDITIONS_PER_ROW: int = 4

# Keys of one summary row.
ROW_SEED_KEY: str = "seed"
ROW_MODE_KEY: str = "mode"
ROW_ARM_KEY: str = "arm"
ROW_HORIZON_KEY: str = "horizon"
ROW_SOURCE_KEY: str = "source"

# Rows a complete second-source summary holds: every seed, view, arm and horizon.
EXPECTED_SUMMARY_ROWS: int = (
    len(SEEDS) * len(OBS_MODES) * len(INTERMEDIATE_STATE_ARMS) * len(EVALUATION_HORIZONS)
)

DISTANCE_NOTE: str = (
    "distances are between consecutive steps of one rollout, not from the true state"
)


@dataclass(frozen=True)
class TruthSources:
    """One run's held-out rows and, where a second source exists, its fresh-rollout rows."""

    held_out: tuple[dict, ...]
    fresh: tuple[dict, ...]


def _rows(artefact: Artefact) -> list[dict]:
    """Return a summary artefact's rows."""
    rows = artefact.payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise FigureContractError(f"{artefact.cited} is not a list of summary rows")
    return rows


def second_source_directories(run_dir: Path) -> list[Path]:
    """Return every subdirectory of a run holding an intermediate-state summary."""
    return sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and (path / INTERMEDIATE_STATES_SUMMARY_FILENAME).is_file()
    )


def check_second_source(rows: Sequence[dict], cited: str) -> None:
    """Refuse a second-source summary with the wrong row count or missing any seed or horizon.

    Views, arms and truth sources are not checked.
    """
    horizons = {row.get(ROW_HORIZON_KEY) for row in rows}
    seeds = {row.get(ROW_SEED_KEY) for row in rows}
    complete = horizons == set(EVALUATION_HORIZONS) and seeds == set(SEEDS)
    if len(rows) != EXPECTED_SUMMARY_ROWS or not complete:
        raise FigureContractError(
            f"{cited} holds {len(rows)} rows over {len(horizons)} horizons and {len(seeds)} "
            f"seeds; a second truth source holds {EXPECTED_SUMMARY_ROWS} rows over "
            f"{len(EVALUATION_HORIZONS)} horizons and {len(SEEDS)} seeds"
        )


def read_truth_sources(reader: Reader, run: str) -> TruthSources:
    """Read a run's held-out summary and its one second truth source.

    Raises:
        MissingArtefactError: If the run has no held-out summary.
        FigureContractError: If the held-out summary holds another source, more
            than one second-source directory exists, or the one present is
            incomplete.
    """
    tree = reader.tree
    run_dir = tree.root / run
    live = tree.load_json(run_dir / INTERMEDIATE_STATES_SUMMARY_FILENAME)
    held_out = _rows(live)
    foreign = [row for row in held_out if row.get(ROW_SOURCE_KEY) != TRUTH_SOURCE_TEST_TRAJECTORY]
    if foreign:
        raise FigureContractError(
            f"{live.cited} holds {len(foreign)} rows whose truth source is not the "
            "held-out trajectory"
        )
    directories = second_source_directories(run_dir)
    if not directories:
        reader.omit(f"{run}: no second truth source, so only the held-out trajectory is drawn")
        return TruthSources(tuple(held_out), ())
    if len(directories) > 1:
        raise FigureContractError(
            f"{run} holds {len(directories)} second-source directories "
            f"({', '.join(path.name for path in directories)}); exactly one is read, so "
            "leave one in place before drawing"
        )
    second = tree.load_json(directories[0] / INTERMEDIATE_STATES_SUMMARY_FILENAME)
    second_rows = _rows(second)
    check_second_source(second_rows, second.cited)
    fresh = tuple(
        row for row in second_rows if row.get(ROW_SOURCE_KEY) == TRUTH_SOURCE_FRESH_ROLLOUT
    )
    if len(fresh) < len(second_rows):
        reader.omit(
            f"{second.cited}: {len(second_rows) - len(fresh)} held-out rows ignored, "
            "since the held-out source is the summary beside the run"
        )
    return TruthSources(tuple(held_out), fresh)


def source_bands(
    reader: Reader, rows: Sequence[dict], source: str, field: str, where: str
) -> list[Band]:
    """Return one band per arm of one field from one truth source, over seeds, in the drawn view.

    A horizon short of seeds is still drawn when interval accepts its count, and
    the shortfall is recorded as an omission.
    """
    bands: list[Band] = []
    for arm in INTERMEDIATE_STATE_ARMS:
        values: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            if (
                row.get(ROW_MODE_KEY) == INTERMEDIATE_FIGURE_MODE
                and row.get(ROW_ARM_KEY) == arm
                and row.get(field) is not None
            ):
                values[row[ROW_HORIZON_KEY]].append(float(row[field]))
        points = []
        for horizon in sorted(values):
            if len(values[horizon]) < len(SEEDS):
                reader.omit(
                    f"{where}, {SOURCE_DISPLAY[source]}, arm {arm}, h = {horizon}: "
                    f"{len(values[horizon])} of {len(SEEDS)} seeds"
                )
            stat = interval(values[horizon], reader.tree.reps)
            if stat is not None:
                points.append((horizon, stat))
        if not points:
            if rows:
                reader.omit(f"{where}, {SOURCE_DISPLAY[source]}, arm {arm}: no {field}")
            continue
        bands.append(Band(
            None,
            tuple(float(horizon) for horizon, _ in points),
            tuple(stat.iqm for _, stat in points),
            ARM_COLOURS[arm],
            tuple(stat.low for _, stat in points),
            tuple(stat.high for _, stat in points),
            line_style=SOURCE_LINE_STYLES[source],
        ))
    return bands


def _panel(  # pylint: disable=too-many-arguments
    reader: Reader,
    condition: Condition,
    sources: TruthSources,
    field: str,
    *,
    first_column: bool,
    bottom_row: bool,
) -> LinePanel:
    """Return one condition's panel of one field, both truth sources."""
    bands = [
        *source_bands(
            reader, sources.held_out, TRUTH_SOURCE_TEST_TRAJECTORY, field, condition.name
        ),
        *source_bands(reader, sources.fresh, TRUTH_SOURCE_FRESH_ROLLOUT, field, condition.name),
    ]
    return LinePanel(
        condition.wrapped_name, tuple(bands), METRIC_DISPLAY[field] if first_column else "",
        x_label=HORIZON_SHORT_AXIS_LABEL if bottom_row else "",
    )


def build_intermediate_compounding(  # pylint: disable=too-many-locals
    tree: OutputsTree,
    split: str,
    fields: tuple[str, ...] = INTERMEDIATE_PRIMARY_FIELDS,
) -> FigureBuild:
    """Per-step movement of the autoregressive arms per symbolic condition, both truth sources.

    Args:
        tree: The outputs tree read.
        split: Ignored, since each held-out summary is already scored on test.
        fields: Summary fields drawn, one block of rows each.

    Returns:
        The built figure.
    """
    del split
    reader = Reader(tree, REPORTED_SPLIT)
    present: list[tuple[Condition, TruthSources]] = []
    for condition in SYMBOLIC_CONDITIONS:
        try:
            sources = read_truth_sources(reader, condition.run_for(ARM_AR_ENDPOINT))
            present.append((condition, sources))
        except MissingArtefactError as error:
            reader.omit(f"{condition.name}: {error}")
    if not present:
        raise MissingArtefactError("no condition holds an intermediate-state summary")
    reader.keys.update(fields)
    rows: list[LayoutRow] = []
    for field_index, field in enumerate(fields):
        for start in range(0, len(present), CONDITIONS_PER_ROW):
            chunk = present[start:start + CONDITIONS_PER_ROW]
            bottom_row = (
                field_index == len(fields) - 1
                and start + CONDITIONS_PER_ROW >= len(present)
            )
            panels: list[LinePanel | None] = [
                _panel(
                    reader, condition, sources, field,
                    first_column=index == 0, bottom_row=bottom_row,
                )
                for index, (condition, sources) in enumerate(chunk)
            ]
            panels.extend([None] * (CONDITIONS_PER_ROW - len(panels)))
            rows.append(LayoutRow(tuple(panels), COMPACT_ROW_HEIGHT_IN))
    reader.notes.extend((f"view: {MODE_DISPLAY[INTERMEDIATE_FIGURE_MODE].lower()}", DISTANCE_NOTE))
    proxies = (
        *(Proxy(ARM_DISPLAY[arm], ARM_COLOURS[arm]) for arm in INTERMEDIATE_STATE_ARMS),
        *(
            Proxy(SOURCE_DISPLAY[source], NEUTRAL_COLOUR, SOURCE_LINE_STYLES[source])
            for source in TRUTH_SOURCES
        ),
    )
    return reader.finish(FigureLayout("", tuple(rows), proxies=proxies))


def build_intermediate_compounding_decoded(tree: OutputsTree, split: str) -> FigureBuild:
    """The decoded-grid counterpart of the main-text intermediate-state figure.

    Args:
        tree: The outputs tree read.
        split: Ignored, as above.

    Returns:
        The built figure.
    """
    return build_intermediate_compounding(tree, split, INTERMEDIATE_SECONDARY_FIELDS)
