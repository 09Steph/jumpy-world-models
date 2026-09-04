"""Write a dataset from a recorded corpus rather than from a live rollout.

Takes any `TrajectorySource` and streams its episodes into shards the rest of
the pipeline reads. The live path cuts episodes at a termination flag; a
recorded corpus supplies them already finished, so the two construct an episode
by different rules and are separate stages.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator

import jax

from config import (
    OFFLINE_SOURCE,
    TRAJECTORY_SHARD_GLOB,
    TRAJECTORY_SHARD_TEMPLATE,
    ExperimentConfig,
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

    Streams shard by shard, so the corpus is never held in memory at once.

    Attributes:
        store: The shard writer.
        source_name: Which corpus this stage converts.
    """

    name: str = "offline_generate"
    dataset_derived: bool = True

    def __init__(
        self, config: ExperimentConfig, source_name: str = OFFLINE_SOURCE
    ) -> None:
        """Build the stage over one declared corpus.

        Args:
            config: The composed experiment configuration.
            source_name: One of OFFLINE_SOURCES.
        """
        super().__init__(config)
        self.store = TrajectoryStore()
        self.source_name = source_name
        self._settings: dict | None = None

    @property
    def settings(self) -> dict:
        """Return the source settings this dataset is keyed on.

        Resolved once and cached; building a source opens the corpus.

        Returns:
            The settings the factory reports for this source.
        """
        if self._settings is None:
            _, self._settings = build_offline_source(self.source_name)
        return self._settings

    def run(self) -> None:
        """Stream the corpus into shards under the dataset directory."""
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
        logger.info(
            "offline generation complete: %d episodes, %d shards -> %s",
            written,
            index + 1,
            safe_rel(target),
        )

    def _stamped(self, episode: Trajectory) -> Trajectory:
        """Return the episode with the source that produced it recorded.

        The source name is the stage's, not the source's: a source does not
        know the registry name it was resolved under, and that name is what
        the collision guard compares.

        Args:
            episode: The episode about to be written.

        Returns:
            A copy carrying SOURCE_PROVENANCE_KEY in its provenance.
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

        `dataset_dir` is keyed on the run name, seed, fast flag and
        environment name, so two corpora declared under one environment name
        land in one directory and the second silently overwrites the first.

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

        The source name, episode count, window settings and shard size all
        change what the shards contain, so each has to be here. Left out, a
        rerun at a different setting matches this sentinel and trains on the
        previous dataset. HORIZON_MAX is one such setting and is absent; see
        `offline_sources`.

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
        ValueError: If the size is not positive, which would accumulate the
            whole corpus into one shard.
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
