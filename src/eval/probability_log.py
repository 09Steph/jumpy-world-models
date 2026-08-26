"""Per-cell probability logging on a fixed seeded subsample.

On from the first run, never switched on later. An argmax discards the
distribution, so cross-entropy, Brier score, calibration error and predictive
entropy become unrecoverable without retraining.

The subsample is drawn with a fixed seed that is never the run seed, so every
reporting seed logs the same examples.

Settings come from config. See `EVAL_PROB_LOG_N`, `EVAL_PROB_LOG_SEED` and
`EVAL_PROB_LOG_DTYPE`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

HORIZONS_ARRAY: str = "horizons"
CHANNEL_ARRAY_TEMPLATE: str = "channel_{index}"

# Provenance keys returned by write_probability_log and written into metrics.json
LOG_EXAMPLES_KEY: str = "examples"
LOG_BINS_KEY: str = "horizon_bins"
LOG_CELLS_KEY: str = "cells"
LOG_CLASSES_KEY: str = "classes"
LOG_SEED_KEY: str = "subsample_seed"
LOG_DTYPE_KEY: str = "dtype"
LOG_CAP_KEY: str = "per_bin_cap"
LOG_ARRAY_BYTES_KEY: str = "array_bytes"
LOG_FILE_BYTES_KEY: str = "file_bytes"
LOG_PATH_KEY: str = "path"


@dataclass(frozen=True)
class LogSettings:
    """The settings that decide what the probability log contains.

    Bundled. They go into the artefact's provenance block together, and a log
    missing one of them cannot have its subsample reconstructed.

    Attributes:
        seed: The fixed subsample seed, config.EVAL_PROB_LOG_SEED. Never the
            run seed, or each reporting seed logs different examples.
        dtype: Storage dtype, config.EVAL_PROB_LOG_DTYPE.
        cap: Examples per horizon bin, or None to log every one.
    """

    seed: int
    dtype: str
    cap: int | None = None


def select_log_examples(
    horizons: jax.Array, cap: int | None, seed: int
) -> np.ndarray:
    """Return the indices of the examples to log, drawn per horizon bin.

    Per bin, not over the whole set. A uniform draw would log almost nothing at
    the long horizons where the model is weakest. Drawn with `seed` alone, so
    two runs of the same evaluation set select bit-identical examples.

    Args:
        horizons: True horizon per example, shape (batch,).
        cap: Examples to keep per horizon bin, or None to keep every one.
        seed: The fixed subsample seed, config.EVAL_PROB_LOG_SEED.

    Returns:
        Sorted indices into the evaluation batch, in batch order.
    """
    values = np.asarray(horizons)
    if cap is None:
        return np.arange(values.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    picks: list[np.ndarray] = []
    for horizon in np.unique(values):
        members = np.flatnonzero(values == horizon)
        if members.size <= cap:
            picks.append(members)
            continue
        picks.append(rng.choice(members, size=cap, replace=False))
    return np.sort(np.concatenate(picks)) if picks else np.empty(0, dtype=np.int64)


def write_probability_log(
    path: Path,
    logits: list[jax.Array],
    horizons: jax.Array,
    settings: LogSettings,
) -> dict:
    """Write the per-cell probability distributions for the seeded subsample.

    The dtype is per environment; float16 goes subnormal below 6.1e-5.

    Args:
        path: Destination .npz path. Its parent is created.
        logits: Per-channel logits for the selected examples only, each
            (examples, height, width, classes).
        horizons: True horizon per selected example, shape (examples,).
        settings: The settings in force, recorded in the returned provenance so
            the subsample is reconstructable from the artefact alone.

    Returns:
        A provenance dict carrying the example count, the horizon-bin count,
        the cells and classes behind the arithmetic, the settings in force, the
        uncompressed array byte count and the resulting file size.
    """
    ensure_dir(path.parent)
    probabilities = [
        np.asarray(jax.nn.softmax(channel, axis=-1)).astype(settings.dtype)
        for channel in logits
    ]
    arrays = {
        CHANNEL_ARRAY_TEMPLATE.format(index=index): channel
        for index, channel in enumerate(probabilities)
    }
    arrays[HORIZONS_ARRAY] = np.asarray(horizons)
    np.savez_compressed(path, **arrays)

    examples = int(horizons.shape[0])
    height, width = (
        (int(probabilities[0].shape[1]), int(probabilities[0].shape[2]))
        if probabilities
        else (0, 0)
    )
    classes = sum(int(channel.shape[-1]) for channel in probabilities)
    itemsize = np.dtype(settings.dtype).itemsize
    # Uncompressed on purpose, so the figure matches EVAL_PROB_LOG_N's arithmetic.
    array_bytes = examples * height * width * classes * itemsize
    record = {
        LOG_EXAMPLES_KEY: examples,
        LOG_BINS_KEY: int(np.unique(np.asarray(horizons)).size),
        LOG_CELLS_KEY: height * width,
        LOG_CLASSES_KEY: classes,
        LOG_SEED_KEY: settings.seed,
        LOG_DTYPE_KEY: settings.dtype,
        LOG_CAP_KEY: settings.cap,
        LOG_ARRAY_BYTES_KEY: array_bytes,
        LOG_FILE_BYTES_KEY: int(path.stat().st_size),
        LOG_PATH_KEY: safe_rel(path),
    }
    logger.info(
        "logged %d probability examples over %d horizon bins, %s uncompressed "
        "-> %s",
        examples,
        record[LOG_BINS_KEY],
        f"{array_bytes / 1e6:.1f} MB",
        safe_rel(path),
    )
    return record


def probabilities_for(logits: list[jax.Array], picks: np.ndarray) -> list[jax.Array]:
    """Select the logged examples' logits from a full evaluation batch.

    Args:
        logits: Per-channel logits over the whole evaluation batch.
        picks: Indices returned by select_log_examples.

    Returns:
        The same list, restricted to the selected examples.
    """
    index = jnp.asarray(picks)
    return [channel[index] for channel in logits]
