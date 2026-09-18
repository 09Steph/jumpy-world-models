"""Thesis figure style: sizes, colours, display text and the save checks.

Figures are drawn at their print width, so a drawn point size is the printed
point size.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
from matplotlib.text import Text

from config import THESIS_HALF_WIDTH_IN, THESIS_TEXT_WIDTH_IN

# Widths a figure may be drawn at. thesis_style refuses any other.
PRINT_WIDTHS_IN: tuple[float, ...] = (THESIS_TEXT_WIDTH_IN, THESIS_HALF_WIDTH_IN)

# Heights in inches, and the forest series count per row. A figure's height is
# its row heights plus FURNITURE_HEIGHT_IN.
PANEL_ROW_HEIGHT_IN: float = 1.5
COMPACT_ROW_HEIGHT_IN: float = 1.45
FOREST_ROW_HEIGHT_IN: float = 0.2
FOREST_AXIS_HEIGHT_IN: float = 0.6
FOREST_SERIES_PER_ROW: float = 2.0
SELECTION_ROW_HEIGHT_IN: float = 2.6
HEATMAP_ROW_HEIGHT_IN: float = 0.15
HEATMAP_AXIS_HEIGHT_IN: float = 1.2
FURNITURE_HEIGHT_IN: float = 0.9

# Type sizes. save_figure refuses visible text at any other size.
FONT_ANNOTATION_PT: float = 7.0
FONT_LABEL_PT: float = 8.0
FONT_TITLE_PT: float = 10.0
FONT_FLOOR_PT: float = FONT_ANNOTATION_PT
FONT_SIZES_PT: tuple[float, ...] = (FONT_ANNOTATION_PT, FONT_LABEL_PT, FONT_TITLE_PT)
FONT_SIZE_TOLERANCE_PT: float = 0.01

# Lines, markers, bands and grids.
LINE_WIDTH_PT: float = 1.2
FLOOR_LINE_WIDTH_PT: float = 1.0
REFERENCE_LINE_WIDTH_PT: float = 0.8
MARKER_SIZE_PT: float = 3.0
CAP_SIZE_PT: float = 2.0
INTERVAL_ALPHA: float = 0.18
REFERENCE_ALPHA: float = 0.4
GRID_ALPHA: float = 0.25
GRID_LINE_WIDTH_PT: float = 0.4
BAR_GROUP_WIDTH: float = 0.8
BAR_EDGE_WIDTH_PT: float = 0.3
HATCH_PATTERN: str = "////"
SERIES_MARKER: str = "o"
REPRESENTATION_MARKERS: dict[str, str] = {"symbolic": "o", "rgb": "s"}
SECONDARY_MARKER: str = "D"
COPY_MARKER: str = "|"
COPY_MARKER_SCALE: float = 2.5
ANNOTATION_OFFSET_PT: float = 3.0
TICK_ROTATION_DEG: float = 90.0
GROUP_LABEL_ROTATION_DEG: float = 30.0
LADDER_SECONDARY_OFFSET: float = 0.5
LEGEND_COLUMNS: int = 3
FOREST_NOTE_MARGIN: float = 0.12

# Colours. Arms keep one colour in every figure. LADDER_COLOURS colours any
# series without its own table.
ARM_COLOURS: dict[int, str] = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}
ROOM_COLOURS: dict[str, str] = {
    "FourRooms": "#4c72b0", "DoorKey": "#dd8452", "Dynamic-Obstacles": "#55a868",
}
LADDER_COLOURS: tuple[str, ...] = (
    "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c",
)
ESTIMATOR_COLOURS: dict[str, str] = {
    "three_parameter": "#4c72b0",
    "two_parameter": "#dd8452",
    "log_log": "#55a868",
}
# Multipliers on the horizon that separate the estimators horizontally.
ESTIMATOR_JITTER: dict[str, float] = {
    "three_parameter": 0.9,
    "two_parameter": 1.0,
    "log_log": 1.1,
}
BASELINE_COLOUR: str = "#000000"
BINDING_BASELINE_COLOUR: str = "#7f7f7f"
RULE_COLOUR: str = "#b22222"
NEUTRAL_COLOUR: str = "#7f7f7f"
HEATMAP_COLOUR_MAP: str = "Blues"
HEATMAP_TEXT_SWITCH: float = 0.6
HEATMAP_DARK_TEXT: str = "#000000"
HEATMAP_LIGHT_TEXT: str = "#ffffff"

# Line styles.
COPY_BASELINE_STYLE: str = "--"
CLIMATOLOGY_BASELINE_STYLE: str = ":"
BINDING_BASELINE_STYLE: str = "-"
EXTRAPOLATION_STYLE: str = ":"
ARM_LINE_STYLES: dict[int, str] = {1: "-", 2: "--", 3: ":"}
SOURCE_LINE_STYLES: dict[str, str] = {"test_trajectory": "-", "fresh_rollout": "--"}
REPRESENTATION_LINE_STYLES: dict[str, str] = {"symbolic": "-", "rgb": "--"}
MODE_LINE_STYLES: dict[str, str] = {"top_down": "-", "egocentric": "--"}
POLICY_LINE_STYLES: dict[str, str] = {"uniform": "-", "PPO": "--"}
RULE_LINE_STYLE: str = "--"
VERTICAL_LINE_STYLE: str = ":"

# Output formats. save_figure writes a PDF for VECTOR_EXTENSION and a PNG for
# RASTER_EXTENSION.
VECTOR_EXTENSION: str = ".pdf"
RASTER_EXTENSION: str = ".png"
FIGURE_DPI: int = 150
PDF_FONT_TYPE: int = 42

# Display text for artefact keys. check_figure_text catches only
# underscore-joined keys, so a lookup that falls back to its key can render a
# raw one.
HORIZON_AXIS_LABEL: str = "Prediction horizon h (steps)"
HORIZON_SHORT_AXIS_LABEL: str = "Horizon h (steps)"
ARM_DISPLAY: dict[int, str] = {
    1: "Arm 1 -- Jumpy",
    2: "Arm 2 -- AR-Endpoint",
    3: "Arm 3 -- AR-Step",
}
ARM_SHORT_DISPLAY: dict[int, str] = {1: "Arm 1", 2: "Arm 2", 3: "Arm 3"}
POLICY_DISPLAY: dict[str, str] = {"uniform": "Uniform collection", "PPO": "PPO collection"}
READING_DISPLAY: dict[str, str] = {
    "compounding_error_sum": "Summed over horizons",
    "compounding_error_integral": "Integral over horizons",
    "compounding_error_discounted_integral": "Discounted integral",
}
MODE_DISPLAY: dict[str, str] = {"top_down": "Top-down view", "egocentric": "Egocentric view"}
SPLIT_DISPLAY: dict[str, str] = {"validation": "Validation split", "test": "Test split"}
SOURCE_DISPLAY: dict[str, str] = {
    "test_trajectory": "Held-out trajectory",
    "fresh_rollout": "Fresh rollout",
}
REPRESENTATION_DISPLAY: dict[str, str] = {
    "symbolic": "Symbolic",
    "rgb": "Pixel",
    "greyscale": "Greyscale pixel",
}
ESTIMATOR_DISPLAY: dict[str, str] = {
    "three_parameter": "Three-parameter",
    "two_parameter": "Two-parameter",
    "log_log": "Log-log slope",
}
ENV_DISPLAY: dict[str, str] = {
    "Navix-FourRooms-v0": "FourRooms",
    "Navix-DoorKey-Random-5x5-v0": "DoorKey",
    "Navix-Dynamic-Obstacles-16x16-v0": "Dynamic-Obstacles",
    "Navix-Dynamic-Obstacles-8x8-v0": "Dynamic-Obstacles 8x8",
    "Navix-DoorKey-16x16-v0": "DoorKey 16x16",
    "Navix-KeyCorridorS6R3-v0": "KeyCorridor",
    "atari-dqn-replay": "Atari (DQN Replay)",
    "atari-dqn-replay-p24": "Atari (DQN Replay), position 24",
    "atari-dqn-replay-p49": "Atari (DQN Replay), position 49",
    "atari-dqn-replay-long": "Atari (DQN Replay), long horizon",
}
METRIC_DISPLAY: dict[str, str] = {
    "model_cross_entropy": "Cross-entropy (nats)",
    "model_mse": "Mean squared error",
    "copy_normalised_skill_score": "Skill score",
    "mover_restricted_skill_score": "Mover-restricted skill score",
    "mover_restricted_accuracy": "Mover-restricted accuracy",
    "mean_changed_cells": "Cells changed from\nthe start (count)",
    "mean_changed_pixels": "Pixels changed from\nthe start (count)",
    "reported_exponent": "Fitted exponent",
    "endpoint_error_ratio": "Endpoint error ratio",
    "compounding_error_integral": "Compounding error\n(integral over horizons)",
    "compounding_error_sum": "Compounding error\n(summed over horizons)",
    "compounding_error_discounted_integral": "Compounding error\n(discounted integral)",
    "residual": "Fit residual",
    "token_distance_mean": "Token distance\nbetween steps",
    "decoded_distance_mean": "Decoded-grid distance\nbetween steps",
    "hamming_cell_fraction": "Displacement (per cent)",
    "displacement_pct": "Displacement at h = 100 (per cent)",
    "per_horizon_gap": "Selected minus final parameters",
    "archive_position": "Archive position",
    "identified_seeds": "Seeds identified",
}
BASELINE_DISPLAY: dict[str, str] = {
    "copy": "Copy baseline",
    "climatology_entropy": "Climatology floor (class frequencies)",
    "climatology_mse": "Climatology floor (mean image)",
    "binding": "Lower of copy and climatology",
}

# Raw artefact keys are lower-case words joined by underscores. Mathtext spans
# are removed before matching.
RAW_KEY_PATTERN: re.Pattern[str] = re.compile(r"[a-z]+_[a-z_]+")
MATH_SPAN_PATTERN: re.Pattern[str] = re.compile(r"\$[^$]*\$")


class FigureStyleError(ValueError):
    """A figure breaks the width, type-size or display-name rules."""


def _style_parameters() -> dict[str, object]:
    """Return the rcParams every thesis figure is drawn under."""
    return {
        "font.size": FONT_LABEL_PT,
        "axes.labelsize": FONT_LABEL_PT,
        "axes.titlesize": FONT_LABEL_PT,
        "figure.titlesize": FONT_TITLE_PT,
        "legend.fontsize": FONT_LABEL_PT,
        "legend.title_fontsize": FONT_LABEL_PT,
        "xtick.labelsize": FONT_ANNOTATION_PT,
        "ytick.labelsize": FONT_ANNOTATION_PT,
        "lines.linewidth": LINE_WIDTH_PT,
        "lines.markersize": MARKER_SIZE_PT,
        "grid.alpha": GRID_ALPHA,
        "grid.linewidth": GRID_LINE_WIDTH_PT,
        "legend.frameon": False,
        "pdf.fonttype": PDF_FONT_TYPE,
        "savefig.bbox": None,
    }


@contextmanager
def thesis_style(
    width_in: float, rows: float = 1, row_height_in: float = PANEL_ROW_HEIGHT_IN
) -> Iterator[tuple[float, float]]:
    """Apply the thesis style and yield the figure size to draw at.

    Args:
        width_in: The width the figure is printed at, in inches.
        rows: Multiplier on row_height_in. May be fractional.
        row_height_in: The height of one row, in inches.

    Yields:
        The (width, height) to construct the figure with.

    Raises:
        FigureStyleError: If the width is not a print width.
    """
    if width_in not in PRINT_WIDTHS_IN:
        raise FigureStyleError(
            f"figure width {width_in} in is not a print width {PRINT_WIDTHS_IN}"
        )
    with matplotlib.rc_context(_style_parameters()):
        yield (width_in, rows * row_height_in + FURNITURE_HEIGHT_IN)


def new_figure(size: tuple[float, float]) -> Figure:
    """Return an empty figure at the given size with a constrained layout."""
    return Figure(figsize=size, layout="constrained")


def display(table: dict, key: object) -> str:
    """Return the display text for a key, refusing a key with no entry.

    Raises:
        FigureStyleError: If the table has no entry for the key.
    """
    if key not in table:
        raise FigureStyleError(f"no display text for {key!r}")
    return table[key]


def _visible_texts(figure: Figure) -> list[Text]:
    """Return every visible, non-empty text element once the figure is laid out."""
    figure.draw_without_rendering()
    return [
        text
        for text in figure.findobj(Text)
        if text.get_visible() and text.get_text().strip()
    ]


def check_figure_text(figure: Figure) -> None:
    """Refuse a figure whose text breaks the size hierarchy or shows a raw key.

    Raises:
        FigureStyleError: On a size outside the hierarchy or a raw artefact key.
    """
    for text in _visible_texts(figure):
        size = text.get_fontsize()
        if not any(abs(size - allowed) <= FONT_SIZE_TOLERANCE_PT for allowed in FONT_SIZES_PT):
            raise FigureStyleError(
                f"text {text.get_text()!r} is {size} pt, outside {FONT_SIZES_PT}"
            )
        plain = MATH_SPAN_PATTERN.sub("", text.get_text())
        if RAW_KEY_PATTERN.search(plain):
            raise FigureStyleError(f"text {text.get_text()!r} shows a raw artefact key")


def save_figure(
    figure: Figure, path: Path, *, title: str = "", subject: str = ""
) -> Path:
    """Check the figure and write it at its declared size.

    Nothing crops the canvas, so the saved width is the drawn width.

    Args:
        figure: The drawn figure.
        path: Output path. A vector extension writes a PDF carrying the title
            and subject as metadata; a raster extension writes a PNG.
        title: Rendered title recorded in the PDF metadata.
        subject: Provenance line recorded in the PDF metadata.

    Returns:
        The path written.

    Raises:
        FigureStyleError: If the text check fails or the extension is unknown.
    """
    check_figure_text(figure)
    if path.suffix == VECTOR_EXTENSION:
        figure.savefig(
            path,
            format="pdf",
            metadata={"Title": title, "Subject": subject, "CreationDate": None},
        )
        return path
    if path.suffix == RASTER_EXTENSION:
        figure.savefig(path, format="png", dpi=FIGURE_DPI, metadata={"Software": None})
        return path
    raise FigureStyleError(f"unknown figure extension {path.suffix!r}")
