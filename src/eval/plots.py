"""Figure drawing for the evaluation artefacts and the thesis figures.

Every figure is drawn under the thesis style at a print width. The thesis
figures are drawn from panel descriptions, and their drawing functions read no
artefact key.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

from config import (
    DISPLACEMENT_CHANNELS_FIGURE_FILENAME,
    DISPLACEMENT_FIGURE_FILENAME,
    DISPLACEMENT_MARKER_HORIZONS,
    ERROR_HORIZON_FIGURE_FILENAME,
    SATURATION_ABSOLUTE_PCT_THRESHOLD,
    THESIS_TEXT_WIDTH_IN,
)
from src.eval.figure_style import (
    ANNOTATION_OFFSET_PT,
    ARM_COLOURS,
    ARM_DISPLAY,
    BAR_EDGE_WIDTH_PT,
    BAR_GROUP_WIDTH,
    BASELINE_COLOUR,
    BASELINE_DISPLAY,
    CAP_SIZE_PT,
    COPY_BASELINE_STYLE,
    COPY_MARKER,
    COPY_MARKER_SCALE,
    ENV_DISPLAY,
    FLOOR_LINE_WIDTH_PT,
    FONT_ANNOTATION_PT,
    FOREST_NOTE_MARGIN,
    HATCH_PATTERN,
    HEATMAP_COLOUR_MAP,
    HEATMAP_DARK_TEXT,
    HEATMAP_LIGHT_TEXT,
    HEATMAP_TEXT_SWITCH,
    HORIZON_AXIS_LABEL,
    INTERVAL_ALPHA,
    LADDER_COLOURS,
    LEGEND_COLUMNS,
    MARKER_SIZE_PT,
    METRIC_DISPLAY,
    MODE_DISPLAY,
    PANEL_ROW_HEIGHT_IN,
    REFERENCE_LINE_WIDTH_PT,
    RULE_COLOUR,
    RULE_LINE_STYLE,
    SERIES_MARKER,
    TICK_ROTATION_DEG,
    VERTICAL_LINE_STYLE,
    new_figure,
    save_figure,
    thesis_style,
)
from src.utils.logging_setup import get_logger
from src.utils.paths import safe_rel

logger = get_logger(__name__)

# Display labels for the NAVIX symbolic observation's channels: entity tag,
# colour and symbolic state.
CHANNEL_LABELS: dict[str, str] = {
    "0": "channel 0: object", "1": "channel 1: colour", "2": "channel 2: direction",
}

# Legend label matplotlib skips.
NO_LEGEND: str = "_nolegend_"


@dataclass(frozen=True)
class Band:  # pylint: disable=too-many-instance-attributes
    """One series drawn as a line, its interval as a band or as error bars."""

    label: str | None
    x: tuple[float, ...]
    centre: tuple[float, ...]
    colour: str
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None
    line_style: str = "-"
    marker: str | None = SERIES_MARKER
    alpha: float = 1.0
    hollow: bool = False
    error_bars: bool = False


@dataclass(frozen=True)
class Reference:
    """A line with no interval, such as a baseline."""

    label: str | None
    x: tuple[float, ...]
    y: tuple[float, ...]
    colour: str = BASELINE_COLOUR
    line_style: str = COPY_BASELINE_STYLE
    line_width: float = FLOOR_LINE_WIDTH_PT


@dataclass(frozen=True)
class LinePanel:  # pylint: disable=too-many-instance-attributes
    """Series against one x axis, with optional baselines and rules."""

    title: str
    bands: tuple[Band, ...]
    y_label: str
    references: tuple[Reference, ...] = ()
    x_label: str = HORIZON_AXIS_LABEL
    log_x: bool = True
    log_y: bool = False
    zero_line: bool = False
    rules: tuple[tuple[float, str], ...] = ()
    vertical_lines: tuple[tuple[float, str], ...] = ()
    x_ticks: tuple[tuple[float, str], ...] = ()


@dataclass(frozen=True)
class ForestPoint:  # pylint: disable=too-many-instance-attributes
    """One estimate and its interval in a forest panel."""

    row: int
    series: int
    centre: float
    low: float
    high: float
    colour: str
    label: str
    note: str = ""
    marker: str = SERIES_MARKER


@dataclass(frozen=True)
class ForestPanel:  # pylint: disable=too-many-instance-attributes
    """Estimates with intervals, one row per condition and one offset per series."""

    title: str
    rows: tuple[str, ...]
    series_count: int
    points: tuple[ForestPoint, ...]
    x_label: str
    log_x: bool = False
    show_row_labels: bool = True
    markers: tuple[tuple[int, float], ...] = ()
    marker_label: str | None = None
    zero_line: bool = False


@dataclass(frozen=True)
class BarSeries:
    """One series of bars across the groups of a bar panel."""

    label: str
    values: tuple[float | None, ...]
    colour: str
    low: tuple[float | None, ...] | None = None
    high: tuple[float | None, ...] | None = None
    hatched: tuple[bool, ...] = ()


@dataclass(frozen=True)
class BarPanel:
    """Grouped bars, one group per category and one bar per series."""

    title: str
    groups: tuple[str, ...]
    series: tuple[BarSeries, ...]
    y_label: str
    rules: tuple[tuple[float, str], ...] = ()
    tick_rotation: float = 0.0


@dataclass(frozen=True)
class ScatterSeries:
    """One set of points in a scatter panel."""

    label: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    colour: str


@dataclass(frozen=True)
class ScatterPanel:
    """Points against one x axis about a zero line."""

    title: str
    series: tuple[ScatterSeries, ...]
    y_label: str
    x_label: str = HORIZON_AXIS_LABEL
    log_x: bool = True


@dataclass(frozen=True)
class HeatmapPanel:
    """A grid of values on a fixed [0, 1] colour scale, with optional per-cell annotations."""

    title: str
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    values: tuple[tuple[float | None, ...], ...]
    annotations: tuple[tuple[str, ...], ...]
    column_rotation: float = TICK_ROTATION_DEG


Panel = LinePanel | ForestPanel | BarPanel | ScatterPanel | HeatmapPanel


@dataclass(frozen=True)
class Proxy:
    """A legend entry with no data behind it."""

    label: str
    colour: str
    line_style: str = "-"
    marker: str | None = None
    hatch: str | None = None


@dataclass(frozen=True)
class LayoutRow:
    """One row of panels drawn side by side."""

    panels: tuple[Panel | None, ...]
    height_in: float = PANEL_ROW_HEIGHT_IN
    width_ratios: tuple[float, ...] | None = None


@dataclass(frozen=True)
class FigureLayout:  # pylint: disable=too-many-instance-attributes
    """A whole figure: its title, its rows of panels and its legend."""

    title: str
    rows: tuple[LayoutRow, ...]
    width_in: float = THESIS_TEXT_WIDTH_IN
    proxies: tuple[Proxy, ...] = ()
    legend: bool = True
    legend_columns: int = LEGEND_COLUMNS
    subject: str = ""


def _series(
    rows: Sequence[dict], mode: str, measure: str, channel: str
) -> tuple[list[int], list[float], list[float], list[int]]:
    """Extract one measure's curve for a mode and channel, sorted by horizon.

    Returns:
        Horizons, means, standard deviations and pair counts.
    """
    selected = sorted(
        (
            row for row in rows
            if row["mode"] == mode
            and row["measure"] == measure
            and row["channel"] == channel
        ),
        key=lambda row: row["horizon"],
    )
    return (
        [row["horizon"] for row in selected],
        [row["mean"] for row in selected],
        [row["std"] for row in selected],
        [row["num_pairs"] for row in selected],
    )


def _marker_points(
    horizons: Sequence[int], values: Sequence[float], errors: Sequence[float]
) -> tuple[list[int], list[float], list[float]]:
    """Return the points at DISPLACEMENT_MARKER_HORIZONS, which carry a marker and an error bar."""
    keep = [
        index for index, horizon in enumerate(horizons)
        if horizon in DISPLACEMENT_MARKER_HORIZONS
    ]
    return (
        [horizons[i] for i in keep],
        [values[i] for i in keep],
        [errors[i] for i in keep],
    )


def _draw_hamming_panel(axis: Axes, rows: Sequence[dict]) -> None:
    """Draw the Hamming panel, one line per observation mode."""
    for index, mode in enumerate(sorted({row["mode"] for row in rows})):
        horizons, means, stds, counts = _series(
            rows, mode, "hamming_cell_fraction", "all"
        )
        if not horizons:
            continue
        colour = LADDER_COLOURS[index % len(LADDER_COLOURS)]
        percentages = [100.0 * value for value in means]
        axis.plot(horizons, percentages, color=colour, label=MODE_DISPLAY.get(mode, mode))
        mark_h, mark_v, mark_e = _marker_points(
            horizons, percentages, [100.0 * s for s in stds]
        )
        axis.errorbar(
            mark_h, mark_v, yerr=mark_e, fmt=SERIES_MARKER, capsize=CAP_SIZE_PT,
            color=colour, linewidth=REFERENCE_LINE_WIDTH_PT,
        )
        for position in (0, -1):
            axis.annotate(
                f"n={counts[position]}",
                (horizons[position], percentages[position]),
                textcoords="offset points", xytext=(ANNOTATION_OFFSET_PT, ANNOTATION_OFFSET_PT),
                fontsize=FONT_ANNOTATION_PT, color=colour,
            )
    axis.axhline(
        SATURATION_ABSOLUTE_PCT_THRESHOLD, linestyle=RULE_LINE_STYLE,
        linewidth=FLOOR_LINE_WIDTH_PT, color=RULE_COLOUR,
        label=f"saturation threshold ({SATURATION_ABSOLUTE_PCT_THRESHOLD:g}%)",
    )
    axis.set_xscale("log")
    axis.set_xlabel(HORIZON_AXIS_LABEL)
    axis.set_ylabel("cells differing between $s_t$ and $s_{t+h}$ (%)")
    axis.set_title("Per-cell Hamming displacement")
    axis.legend()
    axis.grid(which="both")


def _draw_agent_panel(axis: Axes, rows: Sequence[dict]) -> None:
    """Draw the agent-displacement panel, which is top-down only."""
    horizons, means, stds, _ = _series(rows, "top_down", "agent_manhattan", "all")
    if horizons:
        colour = LADDER_COLOURS[0]
        axis.plot(horizons, means, color=colour)
        mark_h, mark_v, mark_e = _marker_points(horizons, means, stds)
        axis.errorbar(
            mark_h, mark_v, yerr=mark_e, fmt=SERIES_MARKER, capsize=CAP_SIZE_PT,
            color=colour, linewidth=REFERENCE_LINE_WIDTH_PT,
        )
    axis.set_xscale("log")
    axis.set_xlabel(HORIZON_AXIS_LABEL)
    axis.set_ylabel("agent Manhattan distance (cells)")
    axis.set_title("Agent displacement, top-down view")
    axis.grid(which="both")


def draw_displacement_figure(
    rows: Sequence[dict], target: Path, environment: str
) -> Path:
    """Draw cell-level Hamming and agent Manhattan against h, side by side.

    Args:
        rows: The measured rows.
        environment: Environment id, for the title.
    """
    title = f"Displacement against horizon, {ENV_DISPLAY.get(environment, environment)}"
    with thesis_style(THESIS_TEXT_WIDTH_IN) as size:
        figure = new_figure(size)
        axes = figure.subplots(1, 2)
        _draw_hamming_panel(axes[0], rows)
        _draw_agent_panel(axes[1], rows)
        figure.suptitle(title)
        path = save_figure(figure, target / DISPLACEMENT_FIGURE_FILENAME, title=title)
    logger.info("wrote displacement figure -> %s", safe_rel(path))
    return path


def draw_channel_figure(  # pylint: disable=too-many-locals
    rows: Sequence[dict], target: Path, environment: str
) -> Path:
    """Draw per-channel Hamming displacement, one panel per observation mode.

    Args:
        rows: The measured rows.
        environment: Environment id, for the title.
    """
    modes = sorted({row["mode"] for row in rows})
    title = (
        f"Per-channel Hamming displacement, {ENV_DISPLAY.get(environment, environment)}"
    )
    with thesis_style(THESIS_TEXT_WIDTH_IN) as size:
        figure = new_figure(size)
        axes = figure.subplots(1, len(modes), squeeze=False)[0]
        for axis, mode in zip(axes, modes):
            for index, (channel, label) in enumerate(CHANNEL_LABELS.items()):
                horizons, means, _, _ = _series(
                    rows, mode, "hamming_cell_fraction", channel
                )
                if horizons:
                    axis.plot(
                        horizons, [100.0 * v for v in means], label=label,
                        color=LADDER_COLOURS[index % len(LADDER_COLOURS)],
                    )
            axis.set_xscale("log")
            axis.set_xlabel(HORIZON_AXIS_LABEL)
            axis.set_ylabel("cells differing (%)")
            axis.set_title(MODE_DISPLAY.get(mode, mode))
            axis.legend()
            axis.grid(which="both")
        figure.suptitle(title)
        path = save_figure(
            figure, target / DISPLACEMENT_CHANNELS_FIGURE_FILENAME, title=title
        )
    logger.info("wrote per-channel figure -> %s", safe_rel(path))
    return path


def _arm_horizon_series(
    aggregate: dict, mode: str, measure: str
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Pull one measure's IQM curve and interval out of a per-arm aggregate.

    Args:
        aggregate: A loaded `aggregate_arm*.json`.
        mode: Observation mode key.
        measure: Metric name inside `per_horizon`, for example
            `model_cross_entropy`.

    Returns:
        Horizons, IQM values and interval bounds, ordered by horizon. A horizon
        without the measure is skipped, and a missing bound falls back to the
        IQM, which draws a zero-width interval.
    """
    per_horizon = aggregate["by_mode"][mode]["per_horizon"]
    horizons: list[int] = []
    centre: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for key in sorted(per_horizon, key=int):
        block = per_horizon[key].get(measure)
        if not isinstance(block, dict) or block.get("iqm") is None:
            continue
        horizons.append(int(key))
        centre.append(float(block["iqm"]))
        lower.append(float(block.get("ci_low", block["iqm"])))
        upper.append(float(block.get("ci_high", block["iqm"])))
    return horizons, centre, lower, upper


def _aggregate_panel(mode: str, arms: dict[str, dict]) -> LinePanel:
    """Build one observation mode's cross-entropy panel over the copy floor.

    The copy floor comes from the first arm that supplies it, assuming every
    arm reports the same floor.
    """
    bands: list[Band] = []
    references: list[Reference] = []
    for arm in sorted(arms):
        aggregate = arms[arm]
        if mode not in aggregate.get("by_mode", {}):
            continue
        horizons, centre, lower, upper = _arm_horizon_series(
            aggregate, mode, "model_cross_entropy"
        )
        if not horizons:
            continue
        bands.append(Band(
            ARM_DISPLAY.get(int(arm), str(arm)), tuple(horizons), tuple(centre),
            ARM_COLOURS.get(int(arm), BASELINE_COLOUR), tuple(lower), tuple(upper),
        ))
        if not references:
            floor_h, floor_c, _, _ = _arm_horizon_series(aggregate, mode, "copy_cross_entropy")
            if floor_h:
                references.append(
                    Reference(BASELINE_DISPLAY["copy"], tuple(floor_h), tuple(floor_c))
                )
    return LinePanel(
        MODE_DISPLAY.get(mode, mode), tuple(bands), METRIC_DISPLAY["model_cross_entropy"],
        references=tuple(references),
    )


def draw_error_against_horizon_figure(
    arms_by_mode: dict[str, dict[str, dict]], target: Path, environment: str
) -> Path:
    """Draw error against horizon for every arm, one panel per observation mode.

    Args:
        arms_by_mode: Mode name to a mapping of arm label to loaded aggregate.
        target: Directory to write into.
        environment: Environment id, for the title.

    Returns:
        The path written.
    """
    layout = FigureLayout(
        title=f"Prediction error against horizon, {ENV_DISPLAY.get(environment, environment)}",
        rows=(LayoutRow(tuple(
            _aggregate_panel(mode, arms_by_mode[mode]) for mode in sorted(arms_by_mode)
        )),),
    )
    path = draw_figure(layout, target / ERROR_HORIZON_FIGURE_FILENAME)
    logger.info("wrote error-against-horizon figure -> %s", safe_rel(path))
    return path


def _apply_axes(  # pylint: disable=too-many-arguments
    axis: Axes, title: str, x_label: str, y_label: str, *, log_x: bool, log_y: bool = False
) -> None:
    """Set one panel's title, axis labels, scales and grid."""
    if log_x:
        axis.set_xscale("log")
        axis.xaxis.set_minor_formatter(NullFormatter())
    if log_y:
        axis.set_yscale("log")
        axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True)


def _interval(centre: float, low: float | None, high: float | None) -> tuple[float, float]:
    """Return an interval as non-negative distances below and above its centre."""
    below = 0.0 if low is None else max(centre - low, 0.0)
    above = 0.0 if high is None else max(high - centre, 0.0)
    return below, above


def _draw_band(axis: Axes, band: Band) -> None:
    """Draw one series, as a band or with error bars."""
    style = {
        "color": band.colour,
        "linestyle": band.line_style,
        "marker": band.marker,
        "markersize": MARKER_SIZE_PT,
        "alpha": band.alpha,
        "label": band.label if band.label else NO_LEGEND,
    }
    if band.hollow:
        style["markerfacecolor"] = "none"
    if band.error_bars and band.low is not None and band.high is not None:
        spans = [_interval(c, l, h) for c, l, h in zip(band.centre, band.low, band.high)]
        axis.errorbar(
            band.x, band.centre, yerr=np.array(spans).T, capsize=CAP_SIZE_PT, **style
        )
        return
    axis.plot(band.x, band.centre, **style)
    if band.low is not None and band.high is not None:
        axis.fill_between(
            band.x, band.low, band.high, color=band.colour,
            alpha=INTERVAL_ALPHA * band.alpha, linewidth=0,
        )


def _draw_line_panel(axis: Axes, panel: LinePanel) -> None:
    """Draw a line panel."""
    for band in panel.bands:
        _draw_band(axis, band)
    for reference in panel.references:
        axis.plot(
            reference.x, reference.y, color=reference.colour,
            linestyle=reference.line_style, linewidth=reference.line_width,
            label=reference.label if reference.label else NO_LEGEND,
        )
    for value, label in panel.rules:
        axis.axhline(
            value, color=RULE_COLOUR, linestyle=RULE_LINE_STYLE,
            linewidth=REFERENCE_LINE_WIDTH_PT, label=label,
        )
    for value, colour in panel.vertical_lines:
        axis.axvline(
            value, color=colour, linestyle=VERTICAL_LINE_STYLE,
            linewidth=REFERENCE_LINE_WIDTH_PT,
        )
    if panel.zero_line:
        axis.axhline(0.0, color=BASELINE_COLOUR, linewidth=REFERENCE_LINE_WIDTH_PT)
    _apply_axes(
        axis, panel.title, panel.x_label, panel.y_label,
        log_x=panel.log_x, log_y=panel.log_y,
    )
    if panel.x_ticks:
        axis.set_xticks(
            [position for position, _ in panel.x_ticks],
            labels=[label for _, label in panel.x_ticks],
        )


def _draw_forest_panel(axis: Axes, panel: ForestPanel) -> None:
    """Draw a forest panel, the first row at the top."""
    spacing = BAR_GROUP_WIDTH / max(panel.series_count, 1)
    for point in panel.points:
        offset = (point.series - (panel.series_count - 1) / 2) * spacing
        below, above = _interval(point.centre, point.low, point.high)
        axis.errorbar(
            point.centre, point.row + offset, xerr=[[below], [above]],
            fmt=point.marker, color=point.colour, capsize=CAP_SIZE_PT,
            markersize=MARKER_SIZE_PT, label=point.label,
        )
        if point.note:
            axis.annotate(
                point.note, (point.high, point.row + offset),
                textcoords="offset points", xytext=(ANNOTATION_OFFSET_PT, 0),
                fontsize=FONT_ANNOTATION_PT, va="center",
            )
    for row, value in panel.markers:
        axis.plot(
            value, row, marker=COPY_MARKER, color=BASELINE_COLOUR, linestyle="none",
            markersize=MARKER_SIZE_PT * COPY_MARKER_SCALE,
            label=panel.marker_label if panel.marker_label else NO_LEGEND,
        )
    if panel.zero_line:
        axis.axvline(0.0, color=BASELINE_COLOUR, linewidth=REFERENCE_LINE_WIDTH_PT)
    if any(point.note for point in panel.points):
        axis.margins(x=FOREST_NOTE_MARGIN)
    _apply_axes(axis, panel.title, panel.x_label, "", log_x=panel.log_x)
    axis.set_yticks(
        range(len(panel.rows)),
        labels=list(panel.rows) if panel.show_row_labels else [""] * len(panel.rows),
    )
    axis.set_ylim(len(panel.rows) - 0.5, -0.5)


def _draw_bar_panel(axis: Axes, panel: BarPanel) -> None:
    """Draw a grouped bar panel."""
    width = BAR_GROUP_WIDTH / max(len(panel.series), 1)
    base = np.arange(len(panel.groups), dtype=float)
    for index, series in enumerate(panel.series):
        positions = base + (index - (len(panel.series) - 1) / 2) * width
        values = np.array(
            [np.nan if value is None else value for value in series.values], dtype=float
        )
        bars = axis.bar(
            positions, values, width, color=series.colour, label=series.label,
            linewidth=BAR_EDGE_WIDTH_PT,
        )
        for patch, hatched in zip(bars, series.hatched):
            if hatched:
                patch.set_hatch(HATCH_PATTERN)
                patch.set_edgecolor(BASELINE_COLOUR)
        if series.low is not None and series.high is not None:
            spans = [
                (np.nan, np.nan) if value is None else _interval(value, low, high)
                for value, low, high in zip(series.values, series.low, series.high)
            ]
            axis.errorbar(
                positions, values, yerr=np.array(spans).T, fmt="none",
                ecolor=BASELINE_COLOUR, capsize=CAP_SIZE_PT,
                elinewidth=REFERENCE_LINE_WIDTH_PT,
            )
    for value, label in panel.rules:
        axis.axhline(
            value, color=RULE_COLOUR, linestyle=RULE_LINE_STYLE,
            linewidth=REFERENCE_LINE_WIDTH_PT, label=label,
        )
    _apply_axes(axis, panel.title, "", panel.y_label, log_x=False)
    axis.set_xticks(
        base, labels=list(panel.groups), rotation=panel.tick_rotation,
        ha="right" if panel.tick_rotation else "center",
    )


def _draw_scatter_panel(axis: Axes, panel: ScatterPanel) -> None:
    """Draw a scatter panel about a zero line."""
    for series in panel.series:
        axis.scatter(
            series.x, series.y, s=MARKER_SIZE_PT ** 2, color=series.colour,
            label=series.label,
        )
    axis.axhline(0.0, color=BASELINE_COLOUR, linewidth=REFERENCE_LINE_WIDTH_PT)
    _apply_axes(axis, panel.title, panel.x_label, panel.y_label, log_x=panel.log_x)


def _draw_heatmap_panel(axis: Axes, panel: HeatmapPanel) -> None:
    """Draw an annotated heatmap on a fixed [0, 1] colour scale."""
    values = np.array(
        [[np.nan if value is None else value for value in row] for row in panel.values],
        dtype=float,
    )
    axis.imshow(values, cmap=HEATMAP_COLOUR_MAP, vmin=0.0, vmax=1.0, aspect="auto")
    for row, annotations in enumerate(panel.annotations):
        for column, text in enumerate(annotations):
            if not text:
                continue
            shade = values[row, column]
            dark = np.isnan(shade) or shade < HEATMAP_TEXT_SWITCH
            axis.text(
                column, row, text, ha="center", va="center", fontsize=FONT_ANNOTATION_PT,
                color=HEATMAP_DARK_TEXT if dark else HEATMAP_LIGHT_TEXT,
            )
    axis.set_title(panel.title)
    axis.set_xticks(
        range(len(panel.column_labels)), labels=list(panel.column_labels),
        rotation=panel.column_rotation, ha="left",
    )
    axis.set_yticks(range(len(panel.row_labels)), labels=list(panel.row_labels))
    axis.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)


def _draw_panel(axis: Axes, panel: Panel) -> None:
    """Dispatch one panel to its drawing function."""
    if isinstance(panel, LinePanel):
        _draw_line_panel(axis, panel)
    elif isinstance(panel, ForestPanel):
        _draw_forest_panel(axis, panel)
    elif isinstance(panel, BarPanel):
        _draw_bar_panel(axis, panel)
    elif isinstance(panel, ScatterPanel):
        _draw_scatter_panel(axis, panel)
    else:
        _draw_heatmap_panel(axis, panel)


def _proxy_handle(proxy: Proxy) -> Artist:
    """Return the legend handle for a proxy entry."""
    if proxy.hatch is not None:
        return Patch(
            facecolor="none", edgecolor=proxy.colour, hatch=proxy.hatch,
            linewidth=BAR_EDGE_WIDTH_PT,
        )
    return Line2D(
        [], [], color=proxy.colour, linestyle=proxy.line_style, marker=proxy.marker,
        markersize=MARKER_SIZE_PT,
    )


def _legend_entries(axes: Sequence[Axes], proxies: Sequence[Proxy]) -> dict[str, Artist]:
    """Collect one legend handle per label across every panel, then the proxies."""
    entries: dict[str, Artist] = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                entries.setdefault(label, handle)
    for proxy in proxies:
        entries.setdefault(proxy.label, _proxy_handle(proxy))
    return entries


def _fit_legend(figure: Figure, entries: dict[str, Artist], columns: int) -> None:
    """Add the legend below the axes in as many columns as fit the figure width."""
    width = figure.get_figwidth() * figure.dpi
    for count in range(min(columns, len(entries)), 1, -1):
        legend = figure.legend(
            list(entries.values()), list(entries), loc="outside lower center", ncols=count
        )
        figure.draw_without_rendering()
        if legend.get_window_extent().width <= width:
            return
        legend.remove()
    figure.legend(list(entries.values()), list(entries), loc="outside lower center", ncols=1)


def draw_figure(layout: FigureLayout, path: Path) -> Path:
    """Draw a figure layout and write it at its print width.

    Args:
        layout: The figure's title, rows of panels and legend.
        path: Output path.

    Returns:
        The path written.
    """
    heights = [row.height_in for row in layout.rows]
    with thesis_style(layout.width_in, rows=sum(heights), row_height_in=1.0) as size:
        figure = new_figure(size)
        rows = figure.subfigures(len(layout.rows), 1, height_ratios=heights, squeeze=False)
        drawn: list[Axes] = []
        for subfigure, row in zip(rows[:, 0], layout.rows):
            axes = subfigure.subplots(
                1, len(row.panels), squeeze=False, width_ratios=row.width_ratios
            )[0]
            for axis, panel in zip(axes, row.panels):
                if panel is None:
                    axis.set_axis_off()
                    continue
                _draw_panel(axis, panel)
                drawn.append(axis)
        figure.suptitle(layout.title)
        entries = _legend_entries(drawn, layout.proxies)
        if layout.legend and entries:
            _fit_legend(figure, entries, layout.legend_columns)
        save_figure(figure, path, title=layout.title, subject=layout.subject)
    return path
