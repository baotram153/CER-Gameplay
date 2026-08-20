"""A read-only record of every (from, to) phase edge the turn-flow diagram
allows. Not consulted by GameplayEngine at runtime — each handler decides
its own next phase directly. This exists purely so tests can assert no
handler ever returns an edge the diagram doesn't draw.
"""
from __future__ import annotations

from .phase import GamePhase

EDGES: frozenset[tuple[GamePhase, GamePhase]] = frozenset({
    (GamePhase.DETERMINE_NEXT_PLAYER, GamePhase.WAIT_FOR_DICE),                    # Children's turn
    (GamePhase.DETERMINE_NEXT_PLAYER, GamePhase.ROLL_DICE),                        # Robot's turn
    (GamePhase.ROLL_DICE, GamePhase.WAIT_FOR_DICE),                                # Rolling complete
    (GamePhase.WAIT_FOR_DICE, GamePhase.WAIT_FOR_DICE),                            # Dice invalid
    (GamePhase.WAIT_FOR_DICE, GamePhase.CHECK_LEGAL_MOVES),                        # Dice valid
    (GamePhase.CHECK_LEGAL_MOVES, GamePhase.WAIT_FOR_CHILDREN_MOVEMENT),           # Children's turn
    (GamePhase.CHECK_LEGAL_MOVES, GamePhase.ROBOT_MOVEMENT),                       # Robot's turn
    (GamePhase.CHECK_LEGAL_MOVES, GamePhase.UPDATE_GAME_STATE),                    # EXTENSION: no legal moves -> auto-skip
    (GamePhase.WAIT_FOR_CHILDREN_MOVEMENT, GamePhase.WAIT_FOR_CHILDREN_MOVEMENT),  # Movement invalid
    (GamePhase.WAIT_FOR_CHILDREN_MOVEMENT, GamePhase.UPDATE_GAME_STATE),           # Movement valid
    (GamePhase.ROBOT_MOVEMENT, GamePhase.UPDATE_GAME_STATE),                       # Movement valid
    (GamePhase.ROBOT_MOVEMENT, GamePhase.RECOVERY),                                # Movement invalid
    (GamePhase.RECOVERY, GamePhase.UPDATE_GAME_STATE),                             # Movement valid (recovered)
    (GamePhase.UPDATE_GAME_STATE, GamePhase.END_GAME),                             # Win
    (GamePhase.UPDATE_GAME_STATE, GamePhase.DETERMINE_NEXT_PLAYER),                # Not win
})
