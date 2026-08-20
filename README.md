# Cer-Gameplay

A robot that plays board games with children (currently Cờ cá ngựa
(Ludo), with Ô Ăn Quan support in `perception`), reading the physical
board through a camera and taking its own turns through a manipulator.

The repo is a [uv](https://docs.astral.sh/uv/) workspace of independent
modules, wired together by one composition root:

- [`modules/perception`](modules/perception/) — camera image -> board state (CV/YOLO).
- [`modules/reasoning`](modules/reasoning/) — game rules + the robot's move-choosing heuristic.
- [`modules/gameplay`](modules/gameplay/) — the turn-taking state machine tying perception/reasoning/manipulation together.
- [`modules/manipulation`](modules/manipulation/) — physical actuation (no real implementation yet — see below).
- [`modules/common`](modules/common/) — shared types (`BoardState`, `Color`, ...) every other module depends on.
- [`src/robot_controller`](src/robot_controller/) — the app: owns the camera, config, and logging, and drives `gameplay.GameplayEngine`'s loop.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- To run against a real camera: an Intel RealSense device with
  `pyrealsense2` importable in this environment. It's intentionally not a
  `uv`-managed dependency (see the note in the root `pyproject.toml`) since
  on some platforms it's installed via the vendor SDK rather than pip —
  just make sure `python -c "import pyrealsense2"` works before running
  with `camera.backend: realsense`. No camera is needed to develop/test
  against recorded images — see "Run without a camera" below.

## Setup

```bash
uv sync
```

This installs every module in the workspace (editable) plus `robot_controller`
itself.

## Configure

The app is driven by one YAML file, gitignored because it's
device-specific — copy the template and adjust it:

```bash
cp configs/robot_controller/app.example.yaml configs/robot_controller/app.yaml
```

Edit `configs/robot_controller/app.yaml` for this setup: which colors are
playing and who controls each (`game.players`/`player_roles`), camera
settings, and where perception's own config lives. Every key is explained
in the example file's comments; the schema it's parsed into is
[`src/robot_controller/config.py`](src/robot_controller/config.py).

You'll also need every module config `main.py`'s import/startup path
actually reads — all gitignored per-machine data (calibration, weights),
same convention as `app.yaml` above, so a fresh clone only has the
`.example.yaml` templates:

```bash
# common.rules loads this eagerly at import time -- nothing in the app
# (not even `--help`) works without it.
cp modules/common/configs/ludo/rules.example.yaml modules/common/configs/ludo/rules.yaml

# Board rectification + track layout (aruco marker IDs, cells: geometry) --
# read by perception's pipeline, referenced from inference.yaml below.
cp modules/common/configs/ludo/board.example.yaml modules/common/configs/ludo/board.yaml

# Detection model/thresholds; points back at board.yaml above via its own
# board_config: key.
cp modules/perception/configs/ludo/inference.example.yaml modules/perception/configs/ludo/inference.yaml
```

See [perception's README](modules/perception/README.md#setup) for the full
board/camera calibration process — the example files above are untuned
placeholders good enough to start the app, not a substitute for
calibrating `board.yaml`/`inference.yaml` against your actual physical
board and camera. `reasoning/config/scoring.yaml` (the robot's move-choosing heuristic
weights) and `modules/perception/models/best.pt` (the detection checkpoint)
are the other two files the app needs — the former is committed directly
(nothing to copy), the latter is a large trained model that isn't
templated at all and isn't produced by anything in this repo; see
perception's README's "Retraining Ludo's pose checkpoint" section for
where it comes from and how to retrain it.

## Run

```bash
uv run python main.py
# or, for a config file somewhere other than the default:
uv run python main.py --config path/to/app.yaml
```

This starts the camera, builds a new game from `configs/robot_controller/app.yaml`,
and runs `GameplayEngine` until someone wins, an unrecoverable error occurs
(e.g. the camera can't be recovered), or `runtime.max_steps` is reached.
Logs go to the console and to a rotating `logs/robot_controller.log` by
default (see the `logging:` section of the config).

Since `modules/manipulation` has no real actuator yet, the robot's turns
are carried out by a human operator: the app logs what to physically do
("roll the die", "move the green piece from X to Y") and waits for
confirmation on the console. Set `manipulation.require_confirmation: false`
to skip the prompts and assume every action succeeds instead (e.g. for an
unattended dry run).

## Run without a camera

For local development or testing, set in `app.yaml`:

```yaml
camera:
  backend: directory
  directory: modules/perception/data/ludo/raw  # or any folder of board photos
```

This replays a folder of still images instead of a live RealSense stream,
looping once exhausted — the rest of the pipeline (perception, gameplay,
logging) runs exactly as it would against a real camera.

## Tests

```bash
uv run pytest tests/                    # robot_controller (camera, config, adapters)
uv run pytest modules/<name>/tests/      # each module's own tests, e.g. modules/gameplay/tests/
```

## Learn more

Each module's own README has the details for that layer: the
[gameplay state machine](modules/gameplay/README.md), the
[perception pipeline and board calibration](modules/perception/README.md),
and the reasoning/manipulation/common modules.
