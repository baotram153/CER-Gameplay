# Gameplay

Turn-taking state machine for a robot playing Ludo with children: whose
turn it is, when to roll, when a move counts as valid, and when the game
ends. It's the orchestration layer that sits between the rules engine
(`reasoning`) and the physical world (`perception`/`manipulation`).

```
Determine next player -> Roll dice (robot only) -> Wait for dice
  -> Check legal moves -> Wait for children's movement / Robot's movement
  -> (Recovery on a failed robot move) -> Update Game State -> End game
```
See [`src/gameplay/transitions.py`](src/gameplay/transitions.py) for every
edge this implements, and [`src/gameplay/phase.py`](src/gameplay/phase.py)
for the phase names.

## Layout

- [`phase.py`](src/gameplay/phase.py) / [`player.py`](src/gameplay/player.py) — the `GamePhase` enum and the `PlayerType` (human/robot) mapping.
- [`context.py`](src/gameplay/context.py) — `GameplayContext`: the `reasoning.GameState` plus per-turn scratch state (die, legal moves, retry counters).
- [`ports/`](src/gameplay/ports/) — `PerceptionPort`/`ManipulationPort`, the two `Protocol`s this module needs from the outside world. Gameplay depends only on `common`/`reasoning`, not on `perception`/`manipulation` directly — anything satisfying these two methods works, real or fake.
- [`handlers/`](src/gameplay/handlers/) — one function per phase, each returning the next `GamePhase`.
- [`engine.py`](src/gameplay/engine.py) — `GameplayEngine`: dispatches handlers. `step()` runs one phase; `run()` loops `step()` to `END_GAME` and returns a `GameResult`. There is no polling/sleep inside the engine — calling `step()` again while still on the same phase *is* the diagram's self-loop, so whatever owns the camera controls the cadence.

## Setup

This module is part of the repo-root `uv` workspace — run `uv sync` from
the repo root, not from here.

## Usage

`gameplay` never talks to hardware directly, so a session needs a
`PerceptionPort` and a `ManipulationPort` implementation. Real adapters
(wrapping `perception.ludo.pipeline.LudoStatePipeline` and a manipulation
API) don't exist yet — see [`tests/support.py`](tests/support.py) for
minimal scripted fakes to build against in the meantime.

```python
from common.constants import Color
from gameplay import GameplayEngine, PlayerType
from reasoning.game_engine import GameState

game = GameState.new_game(
    players=[Color.RED, Color.GREEN],
    entry_offsets={Color.RED: 0, Color.GREEN: 15, Color.YELLOW: 30, Color.BLUE: 45},
    num_shared_steps=60,
)
roles = {Color.RED: PlayerType.HUMAN, Color.GREEN: PlayerType.ROBOT}

engine = GameplayEngine(game, roles, perception, manipulation)  # your Port implementations
result = engine.run()
print(result.winner, result.winner_role, result.turns_played)
```

Run the tests with `uv run pytest` from this directory.

## Known limitations

- **Recovery is a stub.** There's no automated corrective actuation yet —
  `handlers/recovery.py` just signals `ManipulationPort.request_human_help`
  and proceeds, trusting a human to physically fix the board.
- **Robot move selection is a placeholder.** `reasoning.action_planner`
  (the intended scoring engine) is unfinished, so `robot_movement` defaults
  to `first_legal_move` via the injectable `MoveSelector` in
  [`move_selection.py`](src/gameplay/move_selection.py).
- **No retry cap** on the two self-loop phases (`WAIT_FOR_DICE`,
  `WAIT_FOR_CHILDREN_MOVEMENT`) — `GameplayContext.dice_attempts`/
  `movement_attempts` are tracked for a future caller-side timeout policy,
  but gameplay itself never gives up on its own.
