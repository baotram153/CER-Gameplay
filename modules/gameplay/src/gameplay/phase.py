"""The gameplay state machine's phases — one per box in the turn-flow diagram."""
from __future__ import annotations

from enum import StrEnum, unique


@unique
class GamePhase(StrEnum):
    DETERMINE_NEXT_PLAYER = "determine_next_player"
    ROLL_DICE = "roll_dice"
    WAIT_FOR_DICE = "wait_for_dice"
    CHECK_LEGAL_MOVES = "check_legal_moves"
    WAIT_FOR_CHILDREN_MOVEMENT = "wait_for_children_movement"
    ROBOT_MOVEMENT = "robot_movement"
    RECOVERY = "recovery"
    UPDATE_GAME_STATE = "update_game_state"
    END_GAME = "end_game"
