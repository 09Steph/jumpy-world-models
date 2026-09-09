"""Displacement across every registered NAVIX environment, under both policies.

Displacement is the fraction of grid cells differing between the frame at t and
the frame at t + h. It is the stationary-copy baseline's error exactly: that
baseline predicts s_{t+h} = s_t, so the cells that differ are its mistakes. A
dataset whose displacement is near zero leaves a learned model almost nothing to
add, whatever the model is.

Both arms roll through one jitted scan on the same environment with the same
episode handling, so a difference between them is the collection policy and
nothing else.

Pairs spanning a reset are excluded by requiring the step index to advance by
exactly h, which only holds inside one episode. A pair crossing a reset would
compare two unrelated episodes and read as enormous displacement.

Frames are rendered inside the scan and states discarded. A stacked NAVIX state
repeats the sprite atlas on every transition, where a frame rendered from it
does not.

Records are appended one JSON object per line as each environment and seed
completes, so an interrupted sweep resumes from what it already wrote and a torn
final line is discarded on read rather than corrupting the file.

Reached only from --displacement-sweep, never from run_pipeline. The stage
writes no dataset, trains no world model and touches no arm.
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
    OBS_FN_TOP_DOWN_NAVIX,
    SEEDS,
    EnvConfig,
    displacement_sweep_dir,
)
from src.data.ppo_policy import PpoHparams, train_policy
from src.pipeline.displacement import hamming_displacement
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

# Parallel environments and the episode cap come from the generator's own
# defaults, so a swept figure and a generated dataset are measured on the same
# episode shape.
_ENV_DEFAULTS = EnvConfig()
SWEEP_NUM_ENVS: int = _ENV_DEFAULTS.num_envs
SWEEP_MAX_EPISODE_STEPS: int = _ENV_DEFAULTS.max_episode_steps

UNIFORM_ARM: str = "uniform"
PPO_ARM: str = "ppo"

# One row per environment, seed, arm and horizon. The per-channel columns stay
# in the records file, where a channel count that varies by environment costs
# nothing.
CSV_COLUMNS: tuple[str, ...] = (
    "environment",
    "seed",
    "arm",
    "horizon",
    "displacement_pct",
    "num_pairs",
)


def roll(
    env: Any, key: jax.Array, action_fn: Callable
) -> tuple[jax.Array, jax.Array]:
    """Roll every parallel environment forward, rendering frames in the scan.

    Args:
        env: The navix environment.
        key: PRNG key seeding the reset and every action draw.
        action_fn: Takes (timestep, key) and returns one action per environment.

    Returns:
        Stacked frames and step indices, both leading with time.
    """
    # Top-down, because the stored saturation verdicts a swept figure is
    # checked against are read on that mode.
    observation_fn = getattr(navix.observations, OBS_FN_TOP_DOWN_NAVIX)
    timestep = jax.vmap(env.reset)(jax.random.split(key, SWEEP_NUM_ENVS))

    def step(carry, _):
        current, rng = carry
        rng, action_key = jax.random.split(rng)
        nxt = jax.vmap(env.step)(current, action_fn(current, action_key))
        frames = jax.vmap(observation_fn)(current.state)
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

    A percentage, not the fraction hamming_displacement returns. The stored
    saturation verdicts are on this scale, so a figure compared against one
    must be too.

    Args:
        frames: Stacked frames, leading axis time.
        times: Step indices, shape (time, envs).
        horizon: The gap to measure across.

    Returns:
        The percentage and the number of admissible pairs. A zero count is not
        a failure: no episode survived the horizon, which is itself the finding.
    """
    columns, pairs = displacement_columns(frames, times, horizon)
    if columns is None:
        return float("nan"), 0
    return 100.0 * float(columns[-1]), pairs


def _records_for_arm(
    env_name: str, seed: int, arm: str, frames: np.ndarray, times: np.ndarray
) -> list[dict]:
    """Measure one arm across the horizon grid.

    Args:
        env_name: Registered environment identifier.
        seed: The seed this rollout was drawn under.
        arm: Collection policy name.
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
    """Measure both arms on one environment at one seed.

    Args:
        env_name: Registered environment identifier.
        seed: The seed for training, the reset and every action draw.

    Returns:
        The records for both arms across the horizon grid.
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
        records.extend(
            _records_for_arm(
                env_name, seed, arm, np.asarray(frames), np.asarray(times)
            )
        )
    return records


def read_records(path: Path) -> list[dict]:
    """Return every parsable record in a JSON-lines file.

    A line that does not parse is discarded. An interrupted append leaves one,
    and it names work that did not finish.

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
    """Return the (environment, seed) pairs the records file already holds.

    Args:
        path: The records file, which need not exist.
    """
    return {
        (record["environment"], record["seed"]) for record in read_records(path)
    }


def _append_records(path: Path, records: Sequence[dict]) -> None:
    """Append records to the JSON-lines file, one object per line.

    Args:
        path: The records file.
        records: The records to append.
    """
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_csv(path: Path, records: Iterable[dict]) -> None:
    """Write the tabular view of the records.

    Args:
        path: The CSV to write.
        records: Every record the sweep holds.
    """
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

    Environments loop inside seeds, so the breadth claim across the registry is
    complete after one seed and every later seed only narrows it. No environment
    runs twice in succession, so there is no compiled agent to amortise.

    An environment that will not build or train is logged and skipped rather
    than ending the sweep.

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
            # Broad by intent: one environment that will not build or train
            # must not take the rest of the registry with it.
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
