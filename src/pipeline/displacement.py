"""Displacement diagnostic: does the state travel as the horizon grows?

Error growth with `h` is only meaningful if `s_{t+h}` differs from `s_t` at
large `h`. If the state barely changes, the copy baseline is strong at every
horizon and the curve flattens for reasons unrelated to the model.

`hamming_displacement` is the fraction of cells whose code differs.
`agent_displacement` is how far the agent moved in Manhattan cells, which a
rotation in place leaves at zero. Reads whole trajectories directly, not
through the window sampler.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from config import (
    AGENT_CHANNEL_INDEX,
    AGENT_CLASS_INDEX,
    AGENT_RISING_RATIO_MIN,
    DISPLACEMENT_CSV_FILENAME,
    DISPLACEMENT_VERDICT_FILENAME,
    FREE_DIFFUSION_RATIO_REFERENCE,
    SATURATION_ABSOLUTE_PCT_THRESHOLD,
    SATURATION_BINDING_CRITERION,
    SATURATION_RATIO_ANCHORS,
    TRAJECTORY_SHARD_GLOB,
    ExperimentConfig,
    config_snapshot,
    eval_dir,
)
from src.data.trajectory import ObservationMode, Trajectory
from src.data.trajectory_store import TrajectoryStore
from src.eval.plots import draw_channel_figure, draw_displacement_figure
from src.pipeline.base import Stage
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

# Measure names, used as CSV column values and verdict keys.
MEASURE_HAMMING: str = "hamming_cell_fraction"
MEASURE_AGENT: str = "agent_manhattan"

# CSV column order. One row per (mode, measure, channel, horizon).
#
# The sample count is not optional. A long horizon has far fewer valid start
# positions than a short one, so without it a thin, noisy tail reads as
# saturation.
CSV_COLUMNS: tuple[str, ...] = (
    "environment", "mode", "measure", "channel", "horizon",
    "median", "mean", "std", "num_pairs",
)

# Channel label for a measure aggregated across channels, not per channel.
CHANNEL_ALL: str = "all"


def hamming_displacement(
    earlier: np.ndarray, later: np.ndarray
) -> np.ndarray:
    """Return the fraction of cells differing, per channel and overall.

    The stationary-copy baseline predicts `s_{t+h} = s_t`, so its per-cell
    error is exactly this quantity.

    Args:
        earlier: Observations at time t, shape (pairs, height, width, channels).
        later: Observations at time t + h, the same shape.

    Returns:
        Array of shape (pairs, channels + 1). Columns 0..channels-1 are the
        per-channel differing fractions. The final column is the fraction of
        cells differing in any channel, which is the cell-level mover
        definition the generator's statistics also use.

    Raises:
        ValueError: If the two arrays do not have the same shape.
    """
    if earlier.shape != later.shape:
        raise ValueError(
            f"shape mismatch: {earlier.shape} against {later.shape}"
        )
    differs = earlier != later
    pairs = differs.reshape(differs.shape[0], -1, differs.shape[-1])
    per_channel = pairs.mean(axis=1)
    any_channel = pairs.any(axis=-1).mean(axis=1, keepdims=True)
    return np.concatenate([per_channel, any_channel], axis=1)


def agent_displacement(
    earlier: np.ndarray, later: np.ndarray
) -> np.ndarray:
    """Return the Manhattan distance the agent moved, in grid cells.

    Top-down only. The agent code never appears in the egocentric view, so this
    is undefined there.

    Movement is 4-directional, so Manhattan distance is the net displacement in
    moves.

    Args:
        earlier: TOP_DOWN observations at time t, shape
            (pairs, height, width, channels).
        later: TOP_DOWN observations at time t + h, the same shape.

    Returns:
        Manhattan distance per pair, shape (pairs,), as float.

    Raises:
        ValueError: If the agent code is absent from any observation, meaning
            the wrong mode was supplied. Returning zero would read as "the
            agent did not move", which is the finding this exists to detect.
    """
    positions = []
    for frames in (earlier, later):
        channel = frames[..., AGENT_CHANNEL_INDEX]
        found = channel == AGENT_CLASS_INDEX
        if not found.reshape(found.shape[0], -1).any(axis=1).all():
            raise ValueError(
                f"agent code {AGENT_CLASS_INDEX} absent from at least one "
                "observation. agent_displacement is defined for the TOP_DOWN "
                "mode only: a first-person view is rendered from the agent's "
                "own cell, so the agent never appears in it."
            )
        rows, cols = _first_true_index(found)
        positions.append(np.stack([rows, cols], axis=1))
    return np.abs(positions[0] - positions[1]).sum(axis=1).astype(float)


def _first_true_index(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the (row, column) of the first True per leading-axis entry.

    Args:
        mask: Boolean array of shape (pairs, height, width).

    Returns:
        Row indices and column indices, each of shape (pairs,).
    """
    flat = mask.reshape(mask.shape[0], -1).argmax(axis=1)
    return flat // mask.shape[2], flat % mask.shape[2]


def _summarise(values: np.ndarray) -> dict[str, float]:
    """Return the summary statistics every CSV row carries.

    Args:
        values: One measure's values across pairs.

    Returns:
        Median, mean, population standard deviation and the sample count. Not
        the sample standard deviation the cross-seed aggregate reports.
    """
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "num_pairs": int(values.shape[0]),
    }


def measure_displacement(
    trajectories: Sequence[Trajectory], horizons: Sequence[int]
) -> list[dict]:
    """Measure both displacement measures across modes, channels and horizons.

    Pairs are drawn within a trajectory only. One spanning an episode boundary
    would compare unrelated episodes and inflate displacement where episodes
    are shortest.

    Args:
        trajectories: Complete episodes, as written by the generation stage.
        horizons: Horizons to measure at. A horizon no episode is long enough
            to supply contributes no rows, not a row of nulls.

    Returns:
        One row per (mode, measure, channel, horizon), ready for the CSV.
    """
    rows: list[dict] = []
    modes = sorted(
        {mode for item in trajectories for mode in item.observations},
        key=lambda mode: mode.value,
    )
    for mode in modes:
        frames = [
            np.asarray(item.observations[mode])
            for item in trajectories
            if mode in item.observations
        ]
        for horizon in horizons:
            usable = [frame for frame in frames if frame.shape[0] > horizon]
            if not usable:
                continue
            rows.extend(_rows_for_cell(mode, usable, horizon))
    return rows


def _rows_for_cell(
    mode: ObservationMode, frames: Sequence[np.ndarray], horizon: int
) -> list[dict]:
    """Measure one (mode, horizon) cell across every episode supplying pairs.

    Args:
        mode: The observation mode being measured.
        frames: Per-episode observation arrays, filtered to those long enough.
        horizon: The horizon h, comparing s_t against s_{t+h}.

    Returns:
        The CSV rows for this cell.
    """
    earlier = np.concatenate([frame[:-horizon] for frame in frames])
    later = np.concatenate([frame[horizon:] for frame in frames])
    hamming = hamming_displacement(earlier, later)
    num_channels = hamming.shape[1] - 1
    rows = [
        {
            "mode": mode.value, "measure": MEASURE_HAMMING,
            "channel": str(channel), "horizon": horizon,
            **_summarise(hamming[:, channel]),
        }
        for channel in range(num_channels)
    ]
    rows.append({
        "mode": mode.value, "measure": MEASURE_HAMMING,
        "channel": CHANNEL_ALL, "horizon": horizon,
        **_summarise(hamming[:, -1]),
    })
    if mode is ObservationMode.TOP_DOWN and _declares_agent_class(earlier):
        rows.append({
            "mode": mode.value, "measure": MEASURE_AGENT,
            "channel": CHANNEL_ALL, "horizon": horizon,
            **_summarise(agent_displacement(earlier, later)),
        })
    return rows


def _declares_agent_class(frames: np.ndarray) -> bool:
    """Whether these observations carry an agent code in the agent channel.

    Read at the cell level, so an environment with no agent channel skips the
    measure while one that has it keeps `agent_displacement`'s per-frame raise.

    Args:
        frames: Stacked observations, (num_pairs, *grid_shape, num_channels).

    Returns:
        True if the agent code appears anywhere in the agent channel.
    """
    if frames.shape[-1] <= AGENT_CHANNEL_INDEX:
        return False
    return bool((frames[..., AGENT_CHANNEL_INDEX] == AGENT_CLASS_INDEX).any())


def _curve(rows: Sequence[dict], mode: str, measure: str) -> dict[int, float]:
    """Return one measure's mean against horizon, for the cell-level channel.

    Args:
        rows: The measured rows.
        mode: Observation mode value.
        measure: Measure name.

    Returns:
        Mapping of horizon to mean value.
    """
    return {
        row["horizon"]: row["mean"]
        for row in rows
        if row["mode"] == mode
        and row["measure"] == measure
        and row["channel"] == CHANNEL_ALL
    }


def saturation_verdict(rows: Sequence[dict]) -> dict:
    """Evaluate the pre-registered saturation criterion, per mode.

    A computed verdict, not a plot to interpret. Only the absolute percentage
    binds; the ratio against a free-diffusion reference is reported alongside
    it. Proceeds if any mode shows a real task.

    Branch 0 is the criterion being unevaluable, where no episode reaches the
    anchor. Every branch but 1 is recommended, not decided.

    Args:
        rows: The measured rows from measure_displacement.

    Returns:
        Per-mode verdicts, the overall proceed boolean, the recommended branch,
        and a one-line statement of the outcome.
    """
    anchor_low, anchor_high = SATURATION_RATIO_ANCHORS
    modes = sorted({row["mode"] for row in rows})
    per_mode = {}
    for mode in modes:
        hamming = _curve(rows, mode, MEASURE_HAMMING)
        if anchor_high not in hamming:
            per_mode[mode] = {
                "evaluated": False,
                "reason": (
                    f"no episode was longer than h={anchor_high}, so the "
                    "criterion cannot be evaluated for this mode"
                ),
            }
            continue
        absolute_pct = 100.0 * hamming[anchor_high]
        low = hamming.get(anchor_low)
        per_mode[mode] = {
            "evaluated": True,
            "absolute_pct_at_100": absolute_pct,
            "ratio_100_over_10": (
                hamming[anchor_high] / low if low else None
            ),
            "free_diffusion_ratio_reference": FREE_DIFFUSION_RATIO_REFERENCE,
            "binding_criterion": SATURATION_BINDING_CRITERION,
            "threshold_pct": SATURATION_ABSOLUTE_PCT_THRESHOLD,
            "saturated": absolute_pct < SATURATION_ABSOLUTE_PCT_THRESHOLD,
        }
    return _overall(rows, per_mode)


def _overall(rows: Sequence[dict], per_mode: dict) -> dict:
    """Combine the per-mode verdicts into the single gating decision.

    Args:
        rows: The measured rows, for the agent-displacement discriminator.
        per_mode: The per-mode verdicts.

    Returns:
        The full verdict payload.
    """
    evaluated = {k: v for k, v in per_mode.items() if v.get("evaluated")}
    fired = sorted(k for k, v in evaluated.items() if v["saturated"])
    proceed = bool(evaluated) and len(fired) < len(evaluated)
    agent = _curve(rows, ObservationMode.TOP_DOWN.value, MEASURE_AGENT)
    anchor_low, anchor_high = SATURATION_RATIO_ANCHORS
    agent_ratio = (
        agent[anchor_high] / agent[anchor_low]
        if agent.get(anchor_low) and anchor_high in agent
        else None
    )
    # The discriminator. An agent travelling while the grid does not means the
    # limit is one agent in a static grid.
    agent_still_rising = (
        agent_ratio is not None and agent_ratio > AGENT_RISING_RATIO_MIN
    )
    # Absent on an environment with no agent channel, where branch 3 becomes
    # unreachable and a saturated mode falls to branch 2. Recorded so the
    # branch number is not read as evidence the agent has stopped moving.
    agent_measure_available = bool(agent)
    if not evaluated:
        # Not a pass.
        branch, statement = 0, (
            f"no episode reached h={anchor_high}, so this environment "
            "cannot support the horizon range being measured"
        )
    elif not fired:
        branch, statement = 1, (
            "displacement keeps rising in every evaluated mode, "
            "full-scale generation proceeds unchanged"
        )
    elif agent_still_rising:
        branch, statement = 3, (
            f"displacement in {', '.join(fired)} is structurally bounded "
            f"(agent still travels, ratio {agent_ratio:.2f}); generation "
            f"proceeds and {', '.join(fired)} is reported as a limitation"
        )
    elif agent_measure_available:
        branch, statement = 2, (
            f"displacement saturates in {', '.join(fired)} and the agent "
            "itself has stopped moving; full-scale generation is blocked "
            "pending a collection-policy change"
        )
    else:
        branch, statement = 2, (
            f"displacement saturates in {', '.join(fired)} and this "
            "environment declares no agent channel, so branch 3 could not be "
            "tested; full-scale generation is blocked pending a "
            "collection-policy change, and the agent was not measured rather "
            "than measured stationary"
        )
    return {
        "modes": per_mode,
        "saturated_modes": fired,
        "proceed": proceed,
        "recommended_branch": branch,
        "branch_requires_user_confirmation": branch != 1,
        "agent_manhattan_ratio_100_over_10": agent_ratio,
        "agent_still_rising": agent_still_rising,
        "agent_measure_available": agent_measure_available,
        "statement": statement,
    }


class DisplacementDiagnostic(Stage):
    """Measure how far the state travels between s_t and s_{t+h}.

    Reads the generated dataset and writes the per-horizon table, two figures
    and the computed verdict. Changes no data and applies no fix.

    Attributes:
        store: The trajectory store this stage reads through.
    """

    name: str = "displacement"
    dataset_derived: bool = True

    def __init__(self, config: ExperimentConfig) -> None:
        """Build the stage."""
        super().__init__(config)
        self.store = TrajectoryStore()

    def run(self) -> None:
        """Measure displacement, write the artefacts and log the verdict.

        Raises:
            FileNotFoundError: If no dataset exists for this run and seed.
        """
        trajectories = self._load()
        horizons = range(
            self.config.train.horizon_min, self.config.train.horizon_max + 1
        )
        rows = measure_displacement(trajectories, list(horizons))
        verdict = saturation_verdict(rows)
        # Data seed, not run seed, or runs varying only the model seed would
        # write identical plots under several names.
        target = ensure_dir(
            eval_dir(
                self.config.run_name,
                self.config.data_seed,
                self.config.fast,
                self.config.env.name,
            )
        )
        self._write_csv(rows, target)
        self._write_verdict(verdict, target)
        draw_displacement_figure(rows, target, self.config.env.name)
        draw_channel_figure(rows, target, self.config.env.name)
        logger.info("VERDICT: %s", verdict["statement"])

    def _load(self) -> list[Trajectory]:
        """Read every shard the generation stage wrote for this run and seed.

        Returns:
            All complete episodes, in shard order.

        Raises:
            FileNotFoundError: If the dataset directory holds no shards.
        """
        source = self.dataset_dir
        # Never "*.h5". The directory also holds the frozen evaluation
        # windows, and reading one as a shard fails deep inside h5py.
        shards = sorted(source.glob(TRAJECTORY_SHARD_GLOB))
        if not shards:
            raise FileNotFoundError(
                f"no trajectory shards under {safe_rel(source)}. This "
                "diagnostic reads the generation stage's output and creates "
                "none of its own, so run generation first."
            )
        trajectories = [
            item for shard in shards for item in self.store.read_shard(shard)
        ]
        logger.info(
            "read %d trajectories from %d shards", len(trajectories), len(shards)
        )
        return trajectories

    def _write_csv(self, rows: Sequence[dict], target: Path) -> None:
        """Write the raw per-horizon table.

        Args:
            rows: The measured rows.
            target: Directory to write into.
        """
        path = target / DISPLACEMENT_CSV_FILENAME
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "environment": self.config.env.name})
        logger.info("wrote %d displacement rows -> %s", len(rows), safe_rel(path))

    def _write_verdict(self, verdict: dict, target: Path) -> None:
        """Write the verdict, carrying its criterion and not only its answer.

        Args:
            verdict: The computed verdict.
            target: Directory to write into.
        """
        path = target / DISPLACEMENT_VERDICT_FILENAME
        payload = {"provenance": config_snapshot(self.config), **verdict}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("wrote verdict -> %s", safe_rel(path))
