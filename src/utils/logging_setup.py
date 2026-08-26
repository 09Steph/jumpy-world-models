"""Structured logging configuration.

The terminal gets `LEVEL | module | message`, the file the same with a leading
timestamp, and the file is written alongside the terminal and not instead of
it. **Anything parsing a log must check which sink it came from.**

Everything outside `PROJECT_LOGGERS` is held at WARNING.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.paths import safe_rel

# The file format is built from LOG_FORMAT, so the two cannot drift in their
# shared fields. PROJECT_LOGGERS is an allow-list of dotted roots, so "src"
# covers everything beneath it. The banner verbs are fixed width, so changing
# one means changing the width with it.
LOG_FORMAT: str = "%(levelname)s | %(name)s | %(message)s"
LOG_FILE_FORMAT: str = "%(asctime)s | " + LOG_FORMAT
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

PROJECT_LOGGERS: tuple[str, ...] = ("src", "__main__", "config")

BANNER_MARK: str = "===="

_VERB_START: str = "START  "
_VERB_END: str = "END    "
_VERB_SKIP: str = "SKIP   "

STAGE_START_BANNER: str = BANNER_MARK + " " + _VERB_START + " STAGE %s %s " + BANNER_MARK
STAGE_END_BANNER: str = (
    BANNER_MARK + " " + _VERB_END + " STAGE %s %s%s %s " + BANNER_MARK
)
STAGE_SKIP_BANNER: str = (
    BANNER_MARK + " " + _VERB_SKIP + " STAGE %s %s (sentinel) " + BANNER_MARK
)
RUN_START_BANNER: str = (
    BANNER_MARK + " " + _VERB_START + " RUN %s seed=%s %s stages " + BANNER_MARK
)
RUN_END_BANNER: str = (
    BANNER_MARK + " " + _VERB_END + " RUN %s%s %s " + BANNER_MARK
)

SECONDS_PER_MINUTE: int = 60
SECONDS_PER_HOUR: int = 3600


def format_duration(seconds: float) -> str:
    """Render a wall-clock duration in both human and machine-readable form.

    Returns e.g. ``6h12m04s (22324.1s)``. Negative values are rendered as-is,
    not clamped.

    Args:
        seconds: Elapsed wall-clock seconds.

    Returns:
        The formatted duration string.
    """
    if seconds < 0:
        return f"{seconds:.1f}s"
    hours = int(seconds // SECONDS_PER_HOUR)
    minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
    whole_seconds = int(seconds % SECONDS_PER_MINUTE)
    if hours:
        human = f"{hours}h{minutes:02d}m{whole_seconds:02d}s"
    elif minutes:
        human = f"{minutes}m{whole_seconds:02d}s"
    else:
        human = f"{whole_seconds}s"
    return f"{human} ({seconds:.1f}s)"


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    verbose_deps: bool = False,
) -> None:
    """Configure the root logger with the project's canonical format.

    Args:
        level: Minimum level for this project's loggers.
        log_file: Optional path to also write logs to, alongside the terminal.
            Its parent is created if missing. None means terminal only.
        verbose_deps: Emit third-party logging at `level` too. Default False
            holds everything outside PROJECT_LOGGERS at WARNING.

    Note:
        Suppression uses ``Logger.setLevel``, not a handler filter, so a
        suppressed call reaches neither handler.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        # basicConfig applies `format` only to handlers without one, so this
        # keeps the timestamp on disk and off the terminal.
        file_handler.setFormatter(
            logging.Formatter(LOG_FILE_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        handlers.append(file_handler)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    # Root to WARNING, this project's roots back to `level`.
    if not verbose_deps:
        logging.getLogger().setLevel(logging.WARNING)
        for name in PROJECT_LOGGERS:
            logging.getLogger(name).setLevel(level)
    if log_file is not None:
        # safe_rel, not rel: a path outside REPO_ROOT must not raise here.
        logging.getLogger(__name__).info("logging to %s", safe_rel(log_file))


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, conventionally named ``__name__``."""
    return logging.getLogger(name)
