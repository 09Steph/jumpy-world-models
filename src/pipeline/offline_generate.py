"""Write a dataset from a recorded corpus rather than from a live rollout.

The stage resolves its corpus from the environment and streams the source's
episodes into the shards the rest of the pipeline reads.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator

import jax

from config import (
    TRAJECTORY_SHARD_GLOB,
    TRAJECTORY_SHARD_TEMPLATE,
    ExperimentConfig,
    offline_source_for_env,
)
from src.data.trajectory import Trajectory
from src.data.offline_sources import build_offline_source
from src.data.trajectory_store import TrajectoryStore
from src.pipeline.base import Stage
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

# How often the conversion reports progress, in episodes.
PROGRESS_EVERY: int = 1000

# Provenance key naming which corpus an episode was converted from.
SOURCE_PROVENANCE_KEY: str = "offline_source"


class OfflineGenerateStage(Stage):
    """Convert a recorded corpus into trajectory shards.

    Holds one shard's episodes in memory at a time.

    Attributes:
        store: The shard writer.
        source_name: The registry name of the corpus, resolved from the
            environment.
    """

    name: str = "offline_generate"
    dataset_derived: bool = True

    def __init__(self, config: ExperimentConfig) -> None:
        """Build the stage over the corpus its environment declares.

        Args:
            config: The composed experiment configuration.

        Raises:
            ValueError: If the environment's family registers no corpus.
        """
        super().__init__(config)
        self.store = TrajectoryStore()
        self.source_name = offline_source_for_env(config.env.name)
        self._settings: dict | None = None

    @property
    def settings(self) -> dict:
        """Return the source settings this dataset is keyed on, cached.

        Building an Atari source raises when a shard is absent from disk, so
        the skip check and the manifest both need the archive present.
        """
        if self._settings is None:
            _, self._settings = build_offline_source(self.source_name)
        return self._settings

    def run(self) -> None:
        """Stream the corpus into shards under the dataset directory.

        Existing shards are overwritten by index but never removed, so a rerun
        that writes fewer shards than the last leaves the old higher-index
        shards in place, and every reader globs them.

        Raises:
            ValueError: If the directory holds another corpus's shards, the
                source yields nothing, or it yields fewer episodes than the
                settings declare.
        """
        source, settings = build_offline_source(self.source_name)
        self._settings = settings
        target = self.dataset_dir
        self._reject_a_foreign_dataset(target)
        ensure_dir(target)

        key = jax.random.PRNGKey(self.config.data_seed)
        episodes = _limited(source.trajectories(key), settings["num_episodes"])
        written = 0
        index = 0
        for index, shard in enumerate(_chunked(episodes, settings["shard_size"])):
            self.store.write_shard(
                [self._stamped(episode) for episode in shard],
                target / TRAJECTORY_SHARD_TEMPLATE.format(index=index),
                compression=settings["shard_compression"],
            )
            written += len(shard)
            if written % PROGRESS_EVERY < settings["shard_size"]:
                logger.info(
                    "converted %d episodes into %d shards", written, index + 1
                )

        if not written:
            raise ValueError(
                f"the '{self.source_name}' source yielded no episodes, so no "
                f"dataset was written to {safe_rel(target)}. A source that "
                "silently yields nothing would leave a sentinel over an empty "
                "directory"
            )
        expected = settings["num_episodes"]
        if written < expected:
            raise ValueError(
                f"the '{self.source_name}' source yielded {written} episodes "
                f"against the {expected} this dataset is keyed on, so "
                f"{safe_rel(target)} holds a short dataset. The per-cell counts "
                "are in the log lines above. Left to complete, a shortfall "
                "lands a sentinel and trains as though the corpus were whole"
            )
        logger.info(
            "offline generation complete: %d episodes, %d shards -> %s",
            written,
            index + 1,
            safe_rel(target),
        )

    def _stamped(self, episode: Trajectory) -> Trajectory:
        """Return a copy of the episode with SOURCE_PROVENANCE_KEY set.

        The value is the registry name the stage resolved, which
        `_reject_a_foreign_dataset` compares. A source does not know it.
        """
        return replace(
            episode,
            provenance={
                **episode.provenance,
                SOURCE_PROVENANCE_KEY: self.source_name,
            },
        )

    def _reject_a_foreign_dataset(self, target: Path) -> None:
        """Raise rather than overwrite shards a different corpus wrote.

        `dataset_dir` does not include the corpus, so a corpus routed to an
        environment whose directory already holds another's shards would
        overwrite them. It reads only the first shard's first episode and
        compares the source name, not the settings.

        Args:
            target: The dataset directory about to be written.

        Raises:
            ValueError: If the directory holds shards from another source, or
                shards carrying no readable source.
        """
        shards = sorted(target.glob(TRAJECTORY_SHARD_GLOB))
        if not shards:
            return
        episodes = self.store.read_shard(shards[0])
        if not episodes:
            raise ValueError(
                f"{safe_rel(target)} holds shards with no episodes, so what "
                "wrote them cannot be identified. Delete the directory "
                "deliberately rather than writing over it"
            )
        recorded = episodes[0].provenance.get(SOURCE_PROVENANCE_KEY)
        if recorded is None:
            raise ValueError(
                f"{safe_rel(target)} holds shards recording no offline "
                "source, so whether they came from this corpus cannot be "
                "established. Delete the directory deliberately rather than "
                "writing over it"
            )
        if recorded != self.source_name:
            raise ValueError(
                f"{safe_rel(target)} already holds a dataset written from "
                f"source '{recorded}', and this run writes "
                f"'{self.source_name}'. Two corpora must not share one "
                "dataset directory"
            )

    def sentinel_identity(self) -> dict:
        """Return the identity and integrity fields guarding this dataset.

        Carries the source name, every setting the factory reports, and the
        count and byte size of the shard files present. A setting the factory
        leaves out lets a rerun at a different value match this sentinel and
        train on the previous dataset.

        Returns:
            JSON-serialisable identity and integrity fields.
        """
        identity = super().sentinel_identity()
        shards = sorted(self.dataset_dir.glob(TRAJECTORY_SHARD_GLOB))
        identity.update(
            {
                "offline_source": self.source_name,
                "shard_count": len(shards),
                "shard_sizes": [shard.stat().st_size for shard in shards],
                **self.settings,
            }
        )
        return identity


def _limited(episodes: Iterator[Trajectory], limit: int) -> Iterator[Trajectory]:
    """Yield at most `limit` episodes from an iterator.

    Args:
        episodes: The source's iterator.
        limit: Maximum episodes to take.

    Yields:
        Episodes, stopping at the limit or when the source runs out.
    """
    for count, episode in enumerate(episodes):
        if count >= limit:
            return
        yield episode


def _chunked(
    episodes: Iterable[Trajectory], size: int
) -> Iterator[list[Trajectory]]:
    """Group an iterator into lists of at most `size`.

    Args:
        episodes: Episodes to group.
        size: Maximum episodes per group.

    Yields:
        One list per shard, the last possibly shorter.

    Raises:
        ValueError: If the size is less than 1.
    """
    if size < 1:
        raise ValueError(f"shard size must be at least 1, got {size}")
    shard: list[Trajectory] = []
    for episode in episodes:
        shard.append(episode)
        if len(shard) == size:
            yield shard
            shard = []
    if shard:
        yield shard
