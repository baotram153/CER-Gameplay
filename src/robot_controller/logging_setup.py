"""Central logging configuration for robot_controller.

Every module logs the usual way (`logging.getLogger(__name__)`); this is
the one place that decides where those records actually go -- console,
rotating file, or both -- driven entirely by `LoggingConfig` so that
changing log destinations/verbosity is a config edit, not a code change.

Call `configure_logging()` exactly once, as early as possible in the
entrypoint (before constructing anything that logs at import/build time).
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import LoggingConfig

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def configure_logging(config: LoggingConfig) -> None:
    root = logging.getLogger()
    root.setLevel(config.level)
    root.handlers.clear()

    formatter = logging.Formatter(_FORMAT)

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    config.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        config.log_dir / config.file_name,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Route "X is deprecated"-style warnings.warn() calls (e.g. from
    # dependencies) through the same handlers instead of straight to
    # stderr, so a run's log file is a complete record of what happened.
    logging.captureWarnings(True)
