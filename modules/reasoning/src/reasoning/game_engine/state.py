"""Turn-manager: whose turn it is, winners, and the extra-turn-on-6 mechanic."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from common.constants import Color
from common.type import BoardState, Move, Piece

from .apply import apply_move
from .models import TurnResult
from .moves import legal_moves as _legal_moves
from .win import has_player_won

# An arbitrary in-range placeholder, needed before any roll has occurred.
_PLACEHOLDER_DICE = 2


class GameState:
    def __init__(
        self,
        players: Sequence[Color],
        board: BoardState,
        entry_offsets: dict[Color, int],
        num_shared_steps: int,
    ) -> None:
        if not (2 <= len(players) <= 4):
            raise ValueError(f"players must have 2-4 entries, got {len(players)}")
        if board.turn not in players:
            raise ValueError(f"board.turn {board.turn} is not among players {players}")
        self._order = list(players)
        self._entry_offsets = entry_offsets
        self._num_shared_steps = num_shared_steps
        self._winners: list[Color] = []
        self.board = board

    @classmethod
    def new_game(
        cls, players: Sequence[Color], entry_offsets: dict[Color, int], num_shared_steps: int
    ) -> "GameState":
        # BoardState always models all 4 colors (16 pieces) regardless of
        # how many are actually playing — a non-participating color's 4
        # pieces just sit in their yard for the whole game.
        pieces = [Piece(color=c, pos=0) for c in Color for _ in range(4)]
        board = BoardState(pieces=pieces, dice=_PLACEHOLDER_DICE, turn=players[0], timestamp=0.0)
        return cls(players, board, entry_offsets, num_shared_steps)

    @property
    def current_turn(self) -> Color:
        return self.board.turn

    @property
    def entry_offsets(self) -> dict[Color, int]:
        return dict(self._entry_offsets)

    @property
    def num_shared_steps(self) -> int:
        return self._num_shared_steps

    @property
    def winners(self) -> list[Color]:
        return list(self._winners)

    @property
    def active_players(self) -> list[Color]:
        return [c for c in self._order if c not in self._winners]

    def legal_moves(self, die: int) -> list[Move]:
        return _legal_moves(self.board, die, self._entry_offsets, self._num_shared_steps)

    def play_turn(self, die: int, move: Move | None = None) -> TurnResult:
        options = self.legal_moves(die)

        if not options:
            if move is not None:
                raise ValueError("no legal moves this turn; call with move=None to skip")
            extra_turn = die == 6
            self._advance_turn(extra_turn)
            return TurnResult(move=None, skipped=True, extra_turn=extra_turn, winner=None)

        if move is None or move not in options:
            raise ValueError(f"move must be one of the current legal moves: {options}")

        self.board = apply_move(self.board, move)
        winner = None
        if has_player_won(self.board.pieces, move.piece.color):
            winner = move.piece.color
            self._winners.append(winner)

        extra_turn = die == 6
        self._advance_turn(extra_turn)
        return TurnResult(move=move, skipped=False, extra_turn=extra_turn, winner=winner)

    def _advance_turn(self, extra_turn: bool) -> None:
        if extra_turn and self.current_turn not in self._winners:
            return  # same player rolls again
        i = self._order.index(self.current_turn)
        for step in range(1, len(self._order) + 1):
            candidate = self._order[(i + step) % len(self._order)]
            if candidate not in self._winners:
                self.board = replace(self.board, turn=candidate)
                return
        raise RuntimeError("no active players remain")
