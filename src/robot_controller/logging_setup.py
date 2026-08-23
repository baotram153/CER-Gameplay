"""Central logging configuration for robot_controller.

Every module logs the usual way (`logging.getLogger(__name__)`); this is
the one place that decides where those records actually go -- console,
combined file, per-module files, or all three -- driven entirely by
`LoggingConfig` so that changing log destinations/verbosity is a config
edit, not a code change.

Call `configure_logging()` exactly once, as early as possible in the
entrypoint (before constructing anything that logs at import/build time).
"""
from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import LoggingConfig
from .logging_context import TurnFilter

_FORMAT = "%(asctime)s %(levelname)-8s [turn=%(turn)s] %(name)s: %(message)s"


class PerModuleFileHandler(logging.Handler):
    """Fans every record out to its own file, named after the record's
    logger (module) name -- e.g. a record logged from
    "perception.detection.npu_detector" goes to
    "<directory>/perception.detection.npu_detector.log", in addition to
    wherever this run's other handlers (console, combined file) send it.

    Per-module handlers are created lazily, the first time each distinct
    logger name actually logs something, so nothing needs to know the
    full module list up front -- add a new module's logger anywhere in
    the codebase and it gets its own file the first time it's used.
    """

    def __init__(
        self, directory: Path, formatter: logging.Formatter, max_bytes: int, backup_count: int
    ) -> None:
        super().__init__()
        self._directory = directory
        self._formatter = formatter
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._handlers: dict[str, RotatingFileHandler] = {}

    def emit(self, record: logging.LogRecord) -> None:
        handler = self._handlers.get(record.name)
        if handler is None:
            self._directory.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                self._directory / f"{record.name}.log",
                maxBytes=self._max_bytes,
                backupCount=self._backup_count,
            )
            handler.setFormatter(self._formatter)
            self._handlers[record.name] = handler
        handler.emit(record)

    def close(self) -> None:
        for handler in self._handlers.values():
            handler.close()
        self._handlers.clear()
        super().close()


def configure_logging(config: LoggingConfig) -> None:
    root = logging.getLogger()
    # The root logger's own level is the floor below which nothing reaches
    # ANY handler, so it has to be the more verbose (numerically lower) of
    # `level` and `console_level` -- each handler then re-applies its own
    # threshold on top of that (see below), independently.
    root.setLevel(min(logging.getLevelName(config.level), logging.getLevelName(config.console_level)))
    root.handlers.clear()

    formatter = logging.Formatter(_FORMAT)
    turn_filter = TurnFilter()

    # One directory per process run (timestamped by start time) holds
    # this run's combined log plus its per-module breakdown, so neither
    # ever mixes with a previous run's -- and a run's whole log output is
    # just that one directory.
    run_dir = config.log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if config.console:
        # Deliberately independent of `level`: `level` governs the full
        # (typically DEBUG, for the per-step tracing scattered through the
        # perception/gameplay modules) detail captured to the files below;
        # `console_level` keeps the terminal itself readable during a live
        # run by only echoing the more important lines there too.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(config.console_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(turn_filter)
        root.addHandler(console_handler)

    # maxBytes/backupCount rotate *within* a run's own file(s) -- a safety
    # net against a single run logging so much it fills the disk, not a
    # way of managing space across runs (each run already gets a fresh
    # directory for that).
    combined_handler = RotatingFileHandler(
        run_dir / config.file_name, maxBytes=config.max_bytes, backupCount=config.backup_count
    )
    combined_handler.setLevel(config.level)
    combined_handler.setFormatter(formatter)
    combined_handler.addFilter(turn_filter)
    root.addHandler(combined_handler)

    per_module_handler = PerModuleFileHandler(
        run_dir / "modules", formatter, config.max_bytes, config.backup_count
    )
    per_module_handler.setLevel(config.level)
    per_module_handler.addFilter(turn_filter)
    root.addHandler(per_module_handler)

    # Route "X is deprecated"-style warnings.warn() calls (e.g. from
    # dependencies) through the same handlers instead of straight to
    # stderr, so a run's log is a complete record of what happened.
    logging.captureWarnings(True)

    logging.getLogger(__name__).info("Logging to %s (combined + per-module files)", run_dir)
