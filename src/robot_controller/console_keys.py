"""Non-blocking console key commands, shared by every dev-tool feature that
reacts to a key press while the game loop runs (snapshot_saver,
detection_recorder, ...).

Deliberately line-buffered (a key + Enter via the normal terminal, not a
raw single-keystroke read): the console's stdin is already used by
ConsoleManipulationAdapter's confirmation prompts (`input()`), on the same
main thread. A raw/cbreak-mode reader on a background thread would race
those blocking `input()` calls for the same fd; polling non-blockingly
from the main loop instead (LudoPerceptionAdapter.capture() calls
ConsoleKeyDispatcher.poll() once per tick) avoids that conflict entirely.
The trade-off: a line typed while a confirmation prompt is actively
blocking on `input()` is consumed by that prompt, not by this dispatcher --
this works best with `manipulation.require_confirmation: false`, where no
prompt ever competes for stdin.

Only one line is ever read per pending input, and every registered handler
shares the same underlying reader, so two features (e.g. the snapshot key
and the recording start/stop keys) never race each other for the same
typed line -- ConsoleKeyDispatcher.poll() drains stdin exactly once per
call and dispatches each line to at most one handler.
"""
from __future__ import annotations

import logging
import select
import sys
from collections.abc import Callable
from typing import Protocol

logger = logging.getLogger(__name__)


class LineReader(Protocol):
    def poll(self) -> str | None:
        """One pending line of input (stripped), or None if nothing is
        waiting right now. Must never block."""
        ...


class StdinLineReader:
    """Non-blocking stdin polling, real terminal only. Silently inert
    (`poll()` always returns None) if stdin isn't a tty -- piped input, a
    service manager, or a test harness -- rather than raising or blocking
    startup."""

    def __init__(self) -> None:
        self._interactive = sys.stdin.isatty()

    def poll(self) -> str | None:
        if not self._interactive:
            return None
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        return sys.stdin.readline().strip()


class ConsoleKeyDispatcher:
    """Maps typed keys to zero-argument callbacks; `poll()` drains every
    pending line and invokes whichever handler (if any) matches it. An
    unrecognized line is logged at debug level and otherwise ignored."""

    def __init__(self, reader: LineReader | None = None) -> None:
        self._reader = reader or StdinLineReader()
        self._handlers: dict[str, Callable[[], None]] = {}

    def on(self, key: str, handler: Callable[[], None]) -> None:
        self._handlers[key] = handler

    def poll(self) -> None:
        while (line := self._reader.poll()) is not None:
            handler = self._handlers.get(line)
            if handler is not None:
                handler()
            elif line:
                logger.debug("Ignoring unrecognized console input: %r", line)
