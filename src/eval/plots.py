"""Figure drawing for the evaluation artefacts.

Every x-axis is logarithmic in the horizon. The displacement figures mark and
error-bar a subset of horizons and annotate sample counts at the extremes. The
error-against-horizon figure marks every point and shades its interval instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

# Agg before pyplot. This runs headless on the cluster and in pytest.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  pylint: disable=wrong-import-position

from config import (  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
    DISPLACEMENT_CHANNELS_FIGURE_FILENAME,
    DISPLACEMENT_FIGURE_FILENAME,
    DISPLACEMENT_MARKER_HORIZONS,
    ERROR_HORIZON_FIGURE_FILENAME,
    SATURATION_ABSOLUTE_PCT_THRESHOLD,
)
from src.utils.logging_setup import get_logger  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports
from src.utils.paths import safe_rel  # noqa: E402  pylint: disable=wrong-import-position,ungrouped-imports

logger = get_logger(__name__)

FIGURE_DPI: int = 150
FIGURE_SIZE: tuple[float, float] = (11.0, 4.5)
CHANNEL_FIGURE_SIZE: tuple[float, float] = (7.0, 4.5)

# Channel names in the NAVIX symbolic observation.
CHANNEL_LABELS: dict[str, str] = {
    "0": "channel 0: object", "1": "channel 1: colour", "2": "channel 2: direction",
}


def _series(
    rows: Sequence[dict], mode: str, measure: str, channel: str
) -> tuple[list[int], list[float], list[float], list[int]]:
    """Extract one curve from the measured rows, sorted by horizon.

    Args:
        rows: The measured rows.
        mode: Observation mode value.
        measure: Measure name.
        channel: Channel label, or the aggregate label.

    Returns:
        Horizons, means, standard deviations and sample counts.
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
    """Return only the points that carry a marker and an error bar.

    Args:
        horizons: All measured horizons.
        values: The curve's values.
        errors: The curve's dispersions.

    Returns:
        The subset at DISPLACEMENT_MARKER_HORIZONS.
    """
    keep = [
        index for index, horizon in enumerate(horizons)
        if horizon in DISPLACEMENT_MARKER_HORIZONS
    ]
    return (
        [horizons[i] for i in keep],
        [values[i] for i in keep],
        [errors[i] for i in keep],
    )


def _draw_hamming_panel(axis, rows: Sequence[dict]) -> None:
    """Draw the Hamming panel, one line per observation mode.

    Args:
        axis: The axis to draw on.
        rows: The measured rows.
    """
    for mode in sorted({row["mode"] for row in rows}):
        horizons, means, stds, counts = _series(
            rows, mode, "hamming_cell_fraction", "all"
        )
        if not horizons:
            continue
        percentages = [100.0 * value for value in means]
        line = axis.plot(horizons, percentages, label=mode, linewidth=1.4)[0]
        mark_h, mark_v, mark_e = _marker_points(
            horizons, percentages, [100.0 * s for s in stds]
        )
        axis.errorbar(
            mark_h, mark_v, yerr=mark_e, fmt="o", markersize=4, capsize=3,
            color=line.get_color(), linewidth=1.0,
        )
        # Sample counts at the first and last horizon only.
        for index in (0, -1):
            axis.annotate(
                f"n={counts[index]}",
                (horizons[index], percentages[index]),
                textcoords="offset points", xytext=(4, 6), fontsize=7,
                color=line.get_color(),
            )
    axis.axhline(
        SATURATION_ABSOLUTE_PCT_THRESHOLD, linestyle="--", linewidth=1.0,
        color="crimson",
        label=f"saturation threshold ({SATURATION_ABSOLUTE_PCT_THRESHOLD:g}%)",
    )
    axis.set_xscale("log")
    axis.set_xlabel("horizon h (log scale)")
    axis.set_ylabel("cells differing between $s_t$ and $s_{t+h}$ (%)")
    axis.set_title("Per-cell Hamming displacement")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, which="both")


def _draw_agent_panel(axis, rows: Sequence[dict]) -> None:
    """Draw the agent-displacement panel, which is top-down only.

    Args:
        axis: The axis to draw on.
        rows: The measured rows.
    """
    horizons, means, stds, _ = _series(rows, "top_down", "agent_manhattan", "all")
    if horizons:
        line = axis.plot(horizons, means, color="tab:green", linewidth=1.4)[0]
        mark_h, mark_v, mark_e = _marker_points(horizons, means, stds)
        axis.errorbar(
            mark_h, mark_v, yerr=mark_e, fmt="o", markersize=4, capsize=3,
            color=line.get_color(), linewidth=1.0,
        )
    axis.set_xscale("log")
    axis.set_xlabel("horizon h (log scale)")
    axis.set_ylabel("agent Manhattan distance (cells)")
    axis.set_title("Agent displacement (top-down only)")
    axis.grid(alpha=0.3, which="both")


def draw_displacement_figure(
    rows: Sequence[dict], target: Path, environment: str
) -> Path:
    """Draw cell-level Hamming and agent Manhattan against h, side by side.

    Args:
        rows: The measured rows.
        target: Directory to write into.
        environment: Environment id, for the title.

    Returns:
        The path written.
    """
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    _draw_hamming_panel(axes[0], rows)
    _draw_agent_panel(axes[1], rows)
    figure.suptitle(f"Displacement against horizon -- {environment}", fontsize=10)
    figure.tight_layout()
    path = target / DISPLACEMENT_FIGURE_FILENAME
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    logger.info("wrote displacement figure -> %s", safe_rel(path))
    return path


def draw_channel_figure(
    rows: Sequence[dict], target: Path, environment: str
) -> Path:
    """Draw per-channel Hamming displacement, one panel per observation mode.

    Args:
        rows: The measured rows.
        target: Directory to write into.
        environment: Environment id, for the title.

    Returns:
        The path written.
    """
    modes = sorted({row["mode"] for row in rows})
    figure, axes = plt.subplots(
        1, len(modes), figsize=(CHANNEL_FIGURE_SIZE[0] * len(modes),
                                CHANNEL_FIGURE_SIZE[1]),
        squeeze=False,
    )
    for column, mode in enumerate(modes):
        axis = axes[0][column]
        for channel, label in CHANNEL_LABELS.items():
            horizons, means, _, _ = _series(
                rows, mode, "hamming_cell_fraction", channel
            )
            if horizons:
                axis.plot(
                    horizons, [100.0 * v for v in means], label=label,
                    linewidth=1.4,
                )
        axis.set_xscale("log")
        axis.set_xlabel("horizon h (log scale)")
        axis.set_ylabel("cells differing (%)")
        axis.set_title(f"{mode}")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, which="both")
    figure.suptitle(
        f"Per-channel Hamming displacement -- {environment}", fontsize=10
    )
    figure.tight_layout()
    path = target / DISPLACEMENT_CHANNELS_FIGURE_FILENAME
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
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
        Horizons, IQM values and the interval's lower and upper bounds, ordered
        by horizon. A horizon whose measure is absent is skipped, so a partially
        aggregated arm returns a short curve rather than raising.
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


def _draw_horizon_panel(axis, mode: str, arms: dict[str, dict]) -> None:
    """Draw one observation mode's arms against horizon, over the copy floor.

    The floor is read from the first arm that supplies it and drawn once, which
    assumes every arm reports the same floor.
    """
    floor_drawn = False
    for arm in sorted(arms):
        aggregate = arms[arm]
        if mode not in aggregate.get("by_mode", {}):
            continue
        horizons, centre, lower, upper = _arm_horizon_series(
            aggregate, mode, "model_cross_entropy"
        )
        if not horizons:
            continue
        axis.plot(horizons, centre, marker="o", markersize=4, label=f"arm {arm}")
        axis.fill_between(horizons, lower, upper, alpha=0.15)
        if not floor_drawn:
            floor_h, floor_c, _, _ = _arm_horizon_series(
                aggregate, mode, "copy_cross_entropy"
            )
            if floor_h:
                axis.plot(
                    floor_h, floor_c, linestyle="--", color="grey",
                    label="stationary-copy floor",
                )
                floor_drawn = True
    axis.set_xscale("log")
    axis.set_xlabel("horizon h")
    axis.set_ylabel("held-out cross-entropy (IQM)")
    axis.set_title(mode, fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)


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
    modes = sorted(arms_by_mode)
    figure, axes = plt.subplots(1, len(modes), figsize=FIGURE_SIZE, squeeze=False)
    for index, mode in enumerate(modes):
        _draw_horizon_panel(axes[0][index], mode, arms_by_mode[mode])
    figure.suptitle(
        f"Error against horizon, three arms over the copy floor -- {environment}",
        fontsize=10,
    )
    figure.tight_layout()
    path = target / ERROR_HORIZON_FIGURE_FILENAME
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    logger.info("wrote error-against-horizon figure -> %s", safe_rel(path))
    return path
