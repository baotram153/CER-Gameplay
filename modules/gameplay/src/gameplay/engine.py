"""GameplayEngine: drives the turn-flow diagram one step at a time.

Each call to step() dispatches exactly one handler for the current phase.
The diagram's two self-loops (Wait for dice, Wait for children's movement)
are simply what happens when step() is called again while the phase hasn't
changed — there is no polling/threading/sleep inside this module. Whoever
owns the camera (a future robot_controller composition root) drives the
cadence by deciding how often to call step()/run().
"""
from __future__ import annotations

from collections.abc import Callable

from common.constants import Color
from reasoning.game_engine import GameState

from . import handlers
from .context import GameplayContext
from .errors import GameplayError
from .move_selection import MoveSelector, first_legal_move
from .phase import GamePhase
from .player import PlayerType
from .ports.manipulation_port import ManipulationPort
from .ports.perception_port import PerceptionPort
from .result import GameResult


class GameplayEngine:
    def __init__(
        self,
        game: GameState,
        player_roles: dict[Color, PlayerType],
        perception: PerceptionPort,
        manipulation: ManipulationPort,
        move_selector: MoveSelector = first_legal_move,
    ) -> None:
        self.context = GameplayContext(game=game, player_roles=player_roles)
        self._perception = perception
        self._manipulation = manipulation
        self._move_selector = move_selector
        self._handlers: dict[GamePhase, Callable[[], GamePhase]] = {
            GamePhase.DETERMINE_NEXT_PLAYER: lambda: handlers.determine_next_player.run(self.context),
            GamePhase.ROLL_DICE: lambda: handlers.roll_dice.run(self.context, self._manipulation),
            GamePhase.WAIT_FOR_DICE: lambda: handlers.wait_for_dice.run(self.context, self._perception),
            GamePhase.CHECK_LEGAL_MOVES: lambda: handlers.check_legal_moves.run(self.context),
            GamePhase.WAIT_FOR_CHILDREN_MOVEMENT: lambda: handlers.wait_for_children_movement.run(
                self.context, self._perception
            ),
            GamePhase.ROBOT_MOVEMENT: lambda: handlers.robot_movement.run(
                self.context, self._manipulation, self._perception, self._move_selector
            ),
            GamePhase.RECOVERY: lambda: handlers.recovery.run(self.context, self._manipulation),
            GamePhase.UPDATE_GAME_STATE: lambda: handlers.update_game_state.run(self.context),
        }

    def step(self) -> GamePhase:
        if self.context.phase is GamePhase.END_GAME:
            raise GameplayError(
                "game has already ended; call run() to get its GameResult instead of stepping further"
            )
        next_phase = self._handlers[self.context.phase]()
        self.context.phase = next_phase
        return next_phase

    def run(self, max_steps: int = 10_000) -> GameResult:
        for _ in range(max_steps):
            if self.step() is GamePhase.END_GAME:
                return handlers.end_game.build_result(self.context)
        raise GameplayError(f"game did not reach END_GAME within {max_steps} steps")
