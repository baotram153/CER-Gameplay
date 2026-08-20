"""Recovery: reconcile the board after Robot's movement couldn't confirm its
own move.

There is no automated corrective actuation yet (see
ports.manipulation_port.ManipulationPort.request_human_help) — a human is
expected to physically fix the board, pieces, or dice. This handler is
therefore intentionally minimal: it signals for help and proceeds straight
to Update Game State, matching the diagram (which draws only a "Movement
valid" edge out of Recovery, no failure edge and no self-loop). Once a real
human-notification/confirmation mechanism exists, this will likely poll
perception (like the Wait-for-... states) until the board matches
ctx.last_validation.corrected before advancing.
"""
from __future__ import annotations

from ..context import GameplayContext
from ..phase import GamePhase
from ..ports.manipulation_port import ManipulationPort


def run(ctx: GameplayContext, manipulation: ManipulationPort) -> GamePhase:
    manipulation.request_human_help(ctx.last_validation)
    return GamePhase.UPDATE_GAME_STATE
