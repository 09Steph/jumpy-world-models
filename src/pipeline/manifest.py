"""Derived run manifest: a view over the sentinels, never an authority.

Records which seeds completed, in which environment, under which mode. It reads
the same `sentinels.stage_state` call the pipeline skips on, and is never an
input to a skip decision.

No dates in any path. The generation time is a field, and a dated directory
would leave every sentinel unfindable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

from config import (
    CONFIG_DIGEST_LENGTH,
    MANIFEST_SCHEMA_VERSION,
    OBS_MODES,
    SEEDS,
    SPLIT_PROVENANCE_FILENAME,
    ExperimentConfig,
    config_snapshot,
    data_dir,
    dataset_stage_names,
    run_manifest_path,
)
from src.pipeline.stage_registry import DATASET_STAGE_CLASSES
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel
from src.utils.sentinels import stage_state

logger = get_logger(__name__)


def config_digest(config: ExperimentConfig) -> str:
    """Return a short stable hash of the provenance fields of a config.

    Hashes `config_snapshot`, the curated fields that change what the numbers
    mean, not the whole tree.

    Returns:
        The first CONFIG_DIGEST_LENGTH hex characters of the SHA-256 digest.
    """
    payload = json.dumps(config_snapshot(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CONFIG_DIGEST_LENGTH]


def _entry(config: ExperimentConfig, observation_mode: str, data_seed: int) -> dict:
    """Build one manifest entry for a single (environment, mode, data seed).

    Returns:
        One JSON-serialisable manifest entry.
    """
    # model_seed follows data_seed, matching the runner.
    scoped = replace(
        config,
        seed=data_seed,
        data_seed=data_seed,
        model_seed=data_seed,
        sampler=replace(config.sampler, observation_mode=observation_mode),
    )
    target = data_dir(
        scoped.run_name, data_seed, scoped.fast, scoped.env.name
    )
    stages = {}
    # The runner's own rule, so the manifest cannot ask for a stage the run
    # never has. An entry keyed on one would never reach `done`.
    for stage_name in dataset_stage_names(scoped.env.name):
        stage = DATASET_STAGE_CLASSES[stage_name](scoped)
        # Keyed by `name`, looked up by `sentinel_key`. They differ for the
        # mode-scoped stage.
        stages[stage_name] = stage_state(
            stage.sentinel_key,
            scoped.run_name,
            stage.artefact_seed,
            scoped.fast,
            stage.sentinel_identity(),
            scoped.env.name,
        )
    return {
        "observation_mode": observation_mode,
        "data_seed": data_seed,
        "model_seed": scoped.model_seed,
        "path": safe_rel(target),
        "stages": stages,
        "composition_ref": safe_rel(target / SPLIT_PROVENANCE_FILENAME),
    }


def build_manifest(config: ExperimentConfig) -> dict:
    """Build the derived manifest for one run in one environment.

    Reports every observation mode and every reporting seed regardless of what
    this invocation ran. OBS_MODES is not per environment, so an environment
    with one meaningful mode carries entries that can never reach `done`.

    Returns:
        The manifest as a JSON-serialisable mapping.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": config.run_name,
        "env_name": config.env.name,
        "config_digest": config_digest(config),
        "entries": [
            _entry(config, observation_mode, data_seed)
            for observation_mode in OBS_MODES
            for data_seed in SEEDS
        ],
    }


def write_manifest(config: ExperimentConfig) -> dict:
    """Write the derived manifest to the run's environment directory.

    Returns:
        The manifest that was written.
    """
    manifest = build_manifest(config)
    path = run_manifest_path(config.run_name, config.fast, config.env.name)
    ensure_dir(path.parent)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    done = sum(
        1
        for entry in manifest["entries"]
        if all(state == "done" for state in entry["stages"].values())
    )
    logger.info(
        "manifest: %d of %d (mode, seed) combinations complete -> %s",
        done,
        len(manifest["entries"]),
        safe_rel(path),
    )
    return manifest
