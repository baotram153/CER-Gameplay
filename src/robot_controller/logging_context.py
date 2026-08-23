"""Tags every log record with the game's current turn, without every call
site across every module needing to pass it explicitly.

`set_current_turn()` is called once per engine.step() (see app.py's
_run_loop); `TurnFilter` reads that value back onto each LogRecord as it
passes through a handler, so %(turn)s can appear in the log format and
show which player's turn a given line happened during -- handy for
untangling interleaved robot/human turns in a shared log.

A ContextVar (not a plain module-level variable) so this stays correct if
the app ever logs from more than one thread/task at once.
"""
from __future__ import annotations

import contextvars
import logging

from common.constants import Color

_current_turn: contextvars.ContextVar[Color | None] = contextvars.ContextVar(
    "current_turn", default=None
)


def set_current_turn(turn: Color | None) -> None:
    _current_turn.set(turn)


class TurnFilter(logging.Filter):
    """Stamps `record.turn` with the current turn's value (or "-" before
    the first turn is known), so a log Formatter can reference %(turn)s.
    Attach to handlers, not loggers -- see configure_logging: a Filter on
    a logger only applies to records handled directly by that logger, not
    ones its handlers receive via propagation from child loggers."""

    def filter(self, record: logging.LogRecord) -> bool:
        turn = _current_turn.get()
        record.turn = turn.value if turn is not None else "-"
        return True
