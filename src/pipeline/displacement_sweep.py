"""Displacement across every registered NAVIX environment, under both policies.

Displacement is the percentage of grid cells that differ in any channel between
the frames at t and t + h. It equals the per-cell error of copying s_t forward
unchanged, which is not the calibrated copy baseline the evaluation reports.

Both arms roll from the same key, so they start from the same resets. A pair is
kept only when the step index advances by exactly h, which excludes pairs that
span an episode reset.

Every view is rendered from the same states inside the scan, and the states are
not kept. Records are appended as JSON lines after each environment and seed.

Called from --displacement-sweep only. It trains a PPO policy per environment
and seed, but writes no dataset and trains no world model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import jax
import navix
import numpy as np

from config import (
    DISPLACEMENT_SWEEP_CSV_FILENAME,
    DISPLACEMENT_SWEEP_HORIZONS,
    DISPLACEMENT_SWEEP_RECORDS_FILENAME,
    DISPLACEMENT_SWEEP_ROLLOUT_STEPS,
    OBS_FN_EGOCENTRIC_NAVIX,
    OBS_FN_TOP_DOWN_NAVIX,
    OBS_MODE_EGOCENTRIC,
    OBS_MODE_TOP_DOWN,
    SEEDS,
    EnvConfig,
    displacement_sweep_dir,
)
from src.data.ppo_policy import PpoHparams, train_policy
from src.pipeline.displacement import hamming_displacement
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

# Parallel environments and the episode cap are EnvConfig's defaults. --fast
# does not change them, and early termination stays on.
_ENV_DEFAULTS = EnvConfig()
SWEEP_NUM_ENVS: int = _ENV_DEFAULTS.num_envs
SWEEP_MAX_EPISODE_STEPS: int = _ENV_DEFAULTS.max_episode_steps

UNIFORM_ARM: str = "uniform"
PPO_ARM: str = "ppo"

# Views rendered from every state, as (mode, NAVIX observation function name).
SWEEP_VIEWS: tuple[tuple[str, str], ...] = (
    (OBS_MODE_TOP_DOWN, OBS_FN_TOP_DOWN_NAVIX),
    (OBS_MODE_EGOCENTRIC, OBS_FN_EGOCENTRIC_NAVIX),
)

# One row per environment, seed, arm, view and horizon. Per-channel values are
# in the records file only.
CSV_COLUMNS: tuple[str, ...] = (
    "environment",
    "seed",
    "arm",
    "mode",
    "horizon",
    "displacement_pct",
    "num_pairs",
)


def roll(
    env: Any, key: jax.Array, action_fn: Callable
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Roll every parallel environment forward, rendering each view in the scan.

    Both views are rendered from the same state at every step, so they describe
    the same trajectories and share one step index.

    Args:
        env: The navix environment.
        key: PRNG key for the initial reset and every action draw.
        action_fn: Takes (timestep, key) and returns one action per environment.

    Returns:
        Stacked frames per mode, each leading with time, and the step indices.
    """
    observation_fns = {
        mode: getattr(navix.observations, fn_name) for mode, fn_name in SWEEP_VIEWS
    }
    timestep = jax.vmap(env.reset)(jax.random.split(key, SWEEP_NUM_ENVS))

    def step(carry, _):
        current, rng = carry
        rng, action_key = jax.random.split(rng)
        nxt = jax.vmap(env.step)(current, action_fn(current, action_key))
        frames = {
            mode: jax.vmap(observation_fn)(current.state)
            for mode, observation_fn in observation_fns.items()
        }
        return (nxt, rng), (frames, current.t)

    _, out = jax.lax.scan(
        step, (timestep, key), None, DISPLACEMENT_SWEEP_ROLLOUT_STEPS
    )
    return out


def displacement_columns(
    frames: np.ndarray, times: np.ndarray, horizon: int
) -> tuple[np.ndarray | None, int]:
    """Mean differing fractions at one horizon, per channel and any-channel.

    Args:
        frames: Stacked frames, leading axis time.
        times: Step indices, shape (time, envs).
        horizon: The gap to measure across.

    Returns:
        The column means from hamming_displacement, whose final entry is the
        any-channel fraction, and the admissible pair count. (None, 0) when no
        pair survives the horizon.
    """
    frames = np.asarray(frames)
    times = np.asarray(times)
    if frames.shape[0] <= horizon:
        return None, 0
    valid = times[horizon:] == times[:-horizon] + horizon
    earlier, later = frames[:-horizon][valid], frames[horizon:][valid]
    if earlier.shape[0] == 0:
        return None, 0
    columns = hamming_displacement(earlier, later).mean(axis=0)
    return columns, int(earlier.shape[0])


def displacement_at(
    frames: np.ndarray, times: np.ndarray, horizon: int
) -> tuple[float, int]:
    """Mean percentage of cells differing between t and t + horizon.

    A percentage, the scale of the saturation threshold, not the fraction
    hamming_displacement returns.

    Args:
        frames: Stacked frames, leading axis time.
        times: Step indices, shape (time, envs).
        horizon: The gap to measure across.

    Returns:
        The percentage and the number of admissible pairs. NaN and a zero count
        when no pair survives the horizon.
    """
    columns, pairs = displacement_columns(frames, times, horizon)
    if columns is None:
        return float("nan"), 0
    return 100.0 * float(columns[-1]), pairs


def _records_for_arm(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    env_name: str,
    seed: int,
    arm: str,
    mode: str,
    frames: np.ndarray,
    times: np.ndarray,
) -> list[dict]:
    """Measure one arm in one view across the horizon grid.

    Percentages are not comparable across views: the egocentric crop turns
    with the agent.

    Args:
        env_name: Registered environment identifier.
        seed: The seed this rollout was drawn under.
        arm: Collection policy name.
        mode: Observation mode the frames were rendered in.
        frames: Stacked frames, leading axis time.
        times: Step indices, shape (time, envs).

    Returns:
        One record per horizon. A horizon no pair survives records null rather
        than being omitted, so the grid is the same width on every row.
    """
    records = []
    for horizon in DISPLACEMENT_SWEEP_HORIZONS:
        columns, pairs = displacement_columns(frames, times, horizon)
        records.append({
            "environment": env_name,
            "seed": seed,
            "arm": arm,
            "mode": mode,
            "horizon": horizon,
            "displacement_pct": (
                None if columns is None else 100.0 * float(columns[-1])
            ),
            "num_pairs": pairs,
            "channel_pct": (
                None
                if columns is None
                else [100.0 * float(value) for value in columns[:-1]]
            ),
        })
    return records


def measure_environment(env_name: str, seed: int) -> list[dict]:
    """Measure both arms in both views on one environment at one seed.

    PPO is trained once and serves both views.

    Args:
        env_name: Registered environment identifier.
        seed: The seed for training, the reset and every action draw.

    Returns:
        The records for both arms in both views across the horizon grid.
    """
    env = navix.make(env_name, max_steps=SWEEP_MAX_EPISODE_STEPS)
    num_actions = int(env.action_space.maximum) + 1
    policy = train_policy(env, PpoHparams(), jax.random.PRNGKey(seed))

    def uniform_actions(timestep, key: jax.Array) -> jax.Array:
        del timestep
        return jax.random.randint(key, (SWEEP_NUM_ENVS,), 0, num_actions)

    def trained_actions(timestep, key: jax.Array) -> jax.Array:
        return policy.sample(timestep.observation, key)

    records = []
    for arm, action_fn in (
        (UNIFORM_ARM, uniform_actions),
        (PPO_ARM, trained_actions),
    ):
        frames, times = roll(env, jax.random.PRNGKey(seed + 1), action_fn)
        times = np.asarray(times)
        for mode, _ in SWEEP_VIEWS:
            records.extend(
                _records_for_arm(
                    env_name, seed, arm, mode, np.asarray(frames[mode]), times
                )
            )
    return records


def read_records(path: Path) -> list[dict]:
    """Return every parsable record in a JSON-lines file, discarding the rest.

    Args:
        path: The records file, which need not exist.
    """
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(
                "discarding an unparsable record line in %s", safe_rel(path)
            )
    return records


def completed_keys(path: Path) -> set[tuple[str, int]]:
    """Return the (environment, seed) pairs holding at least one record.

    A pair whose append was interrupted counts as done even if some of its
    records are missing.

    Args:
        path: The records file, which need not exist.
    """
    return {
        (record["environment"], record["seed"]) for record in read_records(path)
    }


def _append_records(path: Path, records: Sequence[dict]) -> None:
    """Append records to the JSON-lines file, one object per line."""
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_csv(path: Path, records: Iterable[dict]) -> None:
    """Write the CSV view of the records."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in CSV_COLUMNS})


def run_displacement_sweep_cli(
    run_name: str,
    skip_existing: bool = False,
    fast: bool = False,
    seeds: Sequence[int] = SEEDS,
) -> Path:
    """Sweep every registered environment under both policies and write both files.

    Loops environments inside seeds. An environment whose measurement raises is
    logged and skipped. Without skip_existing, records already in the file are
    appended again.

    Args:
        run_name: Name this sweep's artefacts are filed under.
        skip_existing: Skip any (environment, seed) already in the records file.
        fast: When True the tree hangs off `outputs/fast/`.
        seeds: The seeds to sweep, in order.

    Returns:
        The CSV path written.
    """
    target = ensure_dir(displacement_sweep_dir(run_name, fast))
    records_path = target / DISPLACEMENT_SWEEP_RECORDS_FILENAME
    csv_path = target / DISPLACEMENT_SWEEP_CSV_FILENAME
    done = completed_keys(records_path) if skip_existing else set()
    env_names = sorted(navix.registry())
    logger.info(
        "displacement sweep: %d environments, seeds %s, horizons %s, backend %s",
        len(env_names),
        tuple(seeds),
        DISPLACEMENT_SWEEP_HORIZONS,
        jax.default_backend(),
    )
    for seed in seeds:
        for index, env_name in enumerate(env_names, start=1):
            if (env_name, seed) in done:
                logger.info("SKIP %s seed %d, already recorded", env_name, seed)
                continue
            logger.info(
                "%d/%d %s seed %d", index, len(env_names), env_name, seed
            )
            try:
                records = measure_environment(env_name, seed)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error(
                    "FAILED %s seed %d: %s: %s",
                    env_name,
                    seed,
                    type(exc).__name__,
                    str(exc).splitlines()[0][:120],
                )
                continue
            _append_records(records_path, records)
    _write_csv(csv_path, read_records(records_path))
    logger.info("wrote %s and %s", safe_rel(records_path), safe_rel(csv_path))
    return csv_path
