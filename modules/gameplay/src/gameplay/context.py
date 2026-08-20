"""Per-session mutable state the FSM handlers read and write."""
from __future__ import annotations

from dataclasses import dataclass, field

from common.constants import Color
from common.type import Move, ValidationResult
from reasoning.game_engine import GameState, TurnResult

from .phase import GamePhase
from .player import PlayerType, missing_role_assignments


@dataclass
class GameplayContext:
    """Everything a handler needs, beyond its own arguments, to decide the
    next GamePhase. `game` is the single source of truth for board/turn
    state; every other field is per-turn scratch space, reset by
    determine_next_player at the start of each turn.
    """

    game: GameState
    player_roles: dict[Color, PlayerType]
    phase: GamePhase = GamePhase.DETERMINE_NEXT_PLAYER
    die: int | None = None
    legal_moves: list[Move] = field(default_factory=list)
    chosen_move: Move | None = None
    dice_attempts: int = 0
    movement_attempts: int = 0
    last_validation: ValidationResult | None = None
    last_turn_result: TurnResult | None = None
    winner: Color | None = None
    turns_played: int = 0

    def __post_init__(self) -> None:
        missing = missing_role_assignments(self.game.active_players, self.player_roles)
        if missing:
            raise ValueError(f"player_roles missing entries for active players: {missing}")

    def reset_for_new_turn(self) -> None:
        """Clears per-turn scratch fields so a stale die/move/attempt-count
        from the previous player's turn can never leak into the new one."""
        self.die = None
        self.legal_moves = []
        self.chosen_move = None
        self.dice_attempts = 0
        self.movement_attempts = 0
        self.last_validation = None
