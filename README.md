# Cer-Gameplay

A robot that plays board games with children — currently **cờ cá ngựa (Ludo)**,
with an Ô Ăn Quan perception path alongside it. It reads the physical board
through a camera, picks its own moves, and takes its turn (today by instructing
a human operator — no actuator exists yet).

```
camera frame
   │  rectify (ArUco corners) → detect (YOLO26-pose) → assign cells + read die
   ▼
perception ──► BoardState ──► gameplay (turn FSM) ──► reasoning (rules + move scoring)
                                     │
                                     ▼
                              manipulation (console prompts to a human operator)
```

The repo is a [uv](https://docs.astral.sh/uv/) workspace of independent modules
wired together by one composition root:

| Module | Role |
| --- | --- |
| [`modules/perception`](modules/perception/) | Camera image → board state (rectification, YOLO detection, dice reading, motion gating). |
| [`modules/reasoning`](modules/reasoning/) | Ludo rules engine + the robot's move-choosing heuristic. |
| [`modules/gameplay`](modules/gameplay/) | Turn-taking state machine tying perception/reasoning/manipulation together. |
| [`modules/manipulation`](modules/manipulation/) | Physical actuation — **stub**, no real implementation yet. |
| [`modules/common`](modules/common/) | Shared types (`BoardState`, `Color`, …) and rule constants. |
| [`src/robot_controller`](src/robot_controller/) | The app: owns camera, config, logging; drives `GameplayEngine`'s loop. |

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- For a live camera: an Intel RealSense device with `pyrealsense2` importable.
  It is deliberately **not** a uv-managed dependency (see the note in
  [pyproject.toml](pyproject.toml)) because on some targets it ships with the
  vendor SDK rather than pip. Verify with `python -c "import pyrealsense2"`.
  No camera is needed for development — see [Run without a camera](#run-without-a-camera).

## Setup

```bash
uv sync   # installs every workspace module (editable) + robot_controller
```

Then create the per-machine configs. All real `*.yaml` under `configs/` are
gitignored; only `*.example.yaml` templates are committed:

```bash
cp configs/robot_controller/app.example.yaml            configs/robot_controller/app.yaml
cp modules/common/configs/ludo/rules.example.yaml       modules/common/configs/ludo/rules.yaml
cp modules/common/configs/ludo/board.example.yaml       modules/common/configs/ludo/board.yaml
cp modules/perception/configs/ludo/inference.example.yaml          modules/perception/configs/ludo/inference.yaml
cp modules/perception/configs/ludo/roll_detection.example.yaml     modules/perception/configs/ludo/roll_detection.yaml
cp modules/perception/configs/ludo/movement_detection.example.yaml modules/perception/configs/ludo/movement_detection.yaml
```

`rules.yaml` is loaded eagerly at import time — nothing runs (not even
`--help`) without it. You also need the detection checkpoint at
`modules/perception/models/best.pt`; it is **not** produced by anything in this
repo (see [Training](#training)).

> The `.example.yaml` files carry the authoritative per-key comments and are
> kept ahead of the copies on this machine. When in doubt, re-read the example,
> not your local copy.

## Run

```bash
uv run python main.py
uv run python main.py --config path/to/app.yaml   # non-default config
uv run python main.py --debug                     # force the live debug window on
```

Runs `GameplayEngine` until someone wins, an unrecoverable error occurs, or
`runtime.max_steps` is reached. Since `modules/manipulation` has no actuator,
the robot's turns are performed by a human operator: the app logs the action
("roll the die", "move green from X to Y") and waits for Enter.

**CLI flags and environment variables**

| Flag / variable | Effect |
| --- | --- |
| `--config PATH` | App config path. Overrides `$ROBOT_CONTROLLER_CONFIG`. |
| `--debug` | Forces `debug: true` for this run (live camera + detections window). |
| `ROBOT_CONTROLLER_CONFIG` | Default app config path (default `configs/robot_controller/app.yaml`). |
| `ROBOT_CAMERA_SERIAL` | Overrides `camera.serial_number` — handy for swapping rigs. |
| `ROBOT_LOG_LEVEL` | Overrides `logging.level`. |

### Run without a camera

```yaml
# configs/robot_controller/app.yaml
camera:
  backend: directory
  directory: modules/perception/data/ludo/raw   # any folder of board photos
```

Replays a folder of stills in sorted-filename order, looping when exhausted.
Everything downstream behaves exactly as with a live camera.

## Which file do I edit?

| File | Committed? | Controls |
| --- | --- | --- |
| [configs/robot_controller/app.yaml](configs/robot_controller/app.example.yaml) | template only | Players/roles, camera, logging, runtime cadence, debug + dev tools. Schema: [config.py](src/robot_controller/config.py). |
| [modules/common/configs/ludo/rules.yaml](modules/common/configs/ludo/rules.example.yaml) | template only | `Piece.pos` boundaries, winning cells, yard-entry rolls. |
| [modules/common/configs/ludo/board.yaml](modules/common/configs/ludo/board.example.yaml) | template only | ArUco marker IDs, rectified output size, track topology, **cell centers**. |
| [modules/perception/configs/ludo/inference.yaml](modules/perception/configs/ludo/inference.example.yaml) | template only | Model weights, confidence/IoU thresholds, NPU switch, class map, dice-reading strategy. |
| [modules/perception/configs/ludo/roll_detection.yaml](modules/perception/configs/ludo/roll_detection.example.yaml) | template only | `RollDetector` motion/stability/validity hyperparameters. |
| [modules/perception/configs/ludo/movement_detection.yaml](modules/perception/configs/ludo/movement_detection.example.yaml) | template only | `MovementDetector`, same schema, tuned separately. |
| [modules/reasoning/config/scoring.yaml](modules/reasoning/config/scoring.yaml) | **yes, edit directly** | The robot's move-choosing heuristic weights. |
| `modules/perception/configs/ludo/{board,dice}_train.yaml` + `*_dataset.yaml` | templates only | Training hyperparameters for box-only detectors (see [Training](#training)). |
| `modules/perception/configs/oaq/*.yaml`, `modules/common/configs/oaq/board.yaml` | templates only | The Ô Ăn Quan pipeline — placeholder geometry, not wired into the app. |

## Tuning guide

### Game setup

| Goal | File → key |
| --- | --- |
| Who plays, in what turn order | `app.yaml` → `game.players` |
| Which colors the robot controls | `app.yaml` → `game.player_roles` (`human` \| `robot`) |
| Skip operator Enter-prompts (unattended dry run) | `app.yaml` → `manipulation.require_confirmation: false` |
| Rule constants (yard/home-stretch bounds, winning cells, entry rolls) | `rules.yaml` → `pos.*`, `winning_cells`, `yard_entry_rolls` |

### Camera

All in `app.yaml`, under `camera:` unless noted.

| Key | Effect |
| --- | --- |
| `backend` | `realsense` (live) or `directory` (replay stills). |
| `width` / `height` / `fps` | Stream format. |
| `serial_number` | Pick a specific device; `null` = whichever is attached. |
| `frame_timeout_ms`, `max_consecutive_errors` | When to give up on a frame / restart the pipeline. |
| `max_reconnect_attempts`, `reconnect_backoff_s` | Reconnect policy before the run is declared dead. |
| `perception.rotate_frame_180` | Set `true` if the camera is mounted upside down. |

### Board geometry and rectification

All in `modules/common/configs/ludo/board.yaml`.

| Key | Effect |
| --- | --- |
| `aruco.corner_marker_ids` | The 4 marker IDs at the board corners. Order does not matter — each marker's corner role is inferred from its position in the raw frame. |
| `aruco.dictionary` | ArUco dictionary (`DICT_4X4_50`). |
| `aruco.max_consecutive_misses` | How many corner-detection misses before the cached corner hint is discarded. Higher = tolerates longer occlusions; lower = re-searches sooner after the board actually moves. |
| `aruco.full_sweep_backoff` | How many calls to wait before retrying the expensive (~1s) multi-scale sweep after one failed. `0` disables backoff. |
| `rectification.output_size` | `[w, h]` of the board's own region in the rectified image. Rectification keeps the full frame width so the dice bowl beside the board stays visible. |
| `track.num_shared_steps`, `entry_offsets` | Track topology: loop length and each color's home-entry step. |
| `cells` | Normalized `[x, y]` cell centers. **Do not hand-edit** — regenerate with `scripts/generate_ludo_board_config.py` (see [Recalibrating](#recalibrating-the-board)). |

### Detection model and thresholds

All in `modules/perception/configs/ludo/inference.yaml` → `model:`.

| Key | Effect |
| --- | --- |
| `weights` | Checkpoint path, relative to `modules/perception/`. |
| `conf_threshold` | **Raise** if you get phantom pieces/dice; **lower** if pieces go undetected (a missed pawn is assumed to be in its yard). |
| `iou_threshold` | NMS dedup. Lower it if one piece yields two overlapping boxes. |
| `device` | `null` = auto (CUDA if available, else CPU); or `cpu`/`0`/`cuda:0`. |
| `use_npu` | `true` runs `npu_weights` (quantized ONNX) through onnxruntime's QNN provider on Hexagon HTP, ignoring `weights`/`device`. Flip to `false` for plain PyTorch inference with no other change. |
| `npu_weights`, `qnn_backend_path` | The quantized export and the QNN backend library. `null` backend path = use the one bundled with the installed `onnxruntime-qnn` wheel. |
| `num_keypoints` | Must match the checkpoint's `kpt_shape` (2 for the Ludo pose model). |
| `class_names` | `class_id → piece_<color>` / `dice_<1-6>`, exactly as trained. Wrong order = silently wrong colors. |

Verify NPU placement (not just that the provider registered) — from `modules/perception/`:

```bash
uv run python scripts/check_npu_placement.py --weights models/best.int8.onnx --image path/to/frame.png
```

### Dice reading strategy

`inference.yaml` → `dice_reading.method`:

- **`model`** (default) — the die is one more class on the combined pawn+dice
  checkpoint; free, since piece detection already ran that pass.
- **`bowl_classifier`** — Hough Circle Transform locates the bowl rim each
  frame, and the crop goes to a dedicated dice-classification checkpoint.
  Switching also switches piece detection to the separate `pieces_model:`
  section. Keep a valid `model:` section in the file either way — the app
  resolves its paths unconditionally at startup.

| Key (under `dice_reading.bowl_classifier`) | Effect |
| --- | --- |
| `bowl_min_radius` / `bowl_max_radius` | Expected bowl radius in pixels. **Camera-distance dependent — recalibrate per rig.** |
| `search_region` | `(x, y, w, h)` to restrict the search, or `null` for the whole frame. |
| `bowl_crop_scale` | Crop side length, as a multiple of the bowl **radius** (the example file's comment says diameter — the code uses radius, so `1.7` crops to 0.85× the bowl's width). Needs enough margin that an off-center die is never clipped. |
| `min_confidence` | Reject classifications below this; `0.0` trusts top-1 unconditionally. |
| `classifier.weights` / `class_names` | The dice-classification checkpoint and its class map. |

Tune the radii against real captures before trusting them live:

```bash
cd modules/perception
uv run python scripts/test_bowl_detection.py --image-dir data/ludo/rectified
```

### Roll and movement detection (the Wait-for-… phases)

`roll_detection.yaml` and `movement_detection.yaml` share one schema. Both run
*motion gate → stability window → validity confirm*. Tune them **separately**:
a piece sliding across the board is a slower, larger motion than a die settling.

| Section → key | Effect |
| --- | --- |
| `frame_processing.downscale_size` | Frames are downscaled to this before any comparison; keeps every check sub-millisecond. |
| `frame_processing.blur_kernel` | Odd Gaussian kernel smoothing sensor noise so it doesn't read as motion. |
| `motion.pixel_threshold` | Per-pixel grayscale delta (0-255) that counts as changed. **Raise** if ambient noise keeps triggering. |
| `motion.area_ratio` | Fraction of changed pixels needed to call it motion. **Lower** if a real roll/move never triggers the gate. |
| `motion.background_alpha` | EMA rate the background adapts at on quiet frames. `0` = never adapts to lighting drift; `1` = snaps instantly. |
| `stability.window` | How many readings within the lookback must agree before settling. **Raise** for flakier readings, **lower** for faster turnaround. |
| `stability.lookback` | Size of the window those matches are counted over. Deliberately `> window`, so an occasional missed/low-confidence frame is skipped instead of resetting progress. |
| `stability.min_confidence` | Detections below this don't count toward stability. |
| `validity.enabled` | `false` skips both confirm checks — any stable reading confirms immediately. Escape hatch when the checks themselves cost turnaround; loses protection against stale re-reads and against mistaking a move for a roll. |
| `validity.pixel_diff_threshold` / `pixel_diff_area_ratio` | How different the settled frame must be from the *previously confirmed* one to count as genuinely new. |
| `validity.max_confirm_attempts` | Consecutive Invalid confirms tolerated before resetting to the motion gate. `1` = give up immediately. |

The roll confirm additionally requires that no piece moved; the movement confirm
requires that the die reading is unchanged.

### Robot move selection

[modules/reasoning/config/scoring.yaml](modules/reasoning/config/scoring.yaml) —
committed directly, edit in place. Every term reads from it; nothing hardcodes
a default.

`Score(a) = w_p·P + w_h·H + w_c·C + w_e·E − w_r·R`

| Key | Term |
| --- | --- |
| `weights.w_p` / `progress.alpha`, `.beta` | **P**rogress: how far the move advances, scaled up nearer home. |
| `weights.w_h` / `home_stretch.alpha` | **H**ome stretch: bonus for first entering it. |
| `weights.w_c` / `capture.alpha`, `.beta` | **C**apture: bonus, scaled by how far along the captured piece was. |
| `weights.w_e` / `entry.alpha`, `.beta` | **E**ntry: bringing a piece out of the yard, scaled by how many remain. |
| `weights.w_r` / `risk.alpha`, `.beta` | **R**isk: *subtracted*; chance an opponent captures the piece on their next roll. |

To bypass the heuristic entirely, pass a different `MoveSelector` to
`GameplayEngine` (e.g. `first_legal_move` from
[move_selection.py](modules/gameplay/src/gameplay/move_selection.py)).

### Runtime, logging, and debugging

All in `app.yaml`.

| Key | Effect |
| --- | --- |
| `runtime.tick_interval_s` | Delay between `engine.step()` calls — sets camera-polling cadence in the Wait-for-… phases. |
| `runtime.max_steps` | Hard cap on steps before the run is abandoned. |
| `runtime.stuck_warning_attempts` | Warn after this many consecutive steps in the same phase. |
| `logging.level` | What reaches the log files. `DEBUG` gets full per-step perception tracing. |
| `logging.console_level` | Independent, usually higher threshold for the terminal — keeps a live run readable while files still capture everything. |
| `logging.log_dir`, `max_bytes`, `backup_count` | Rotation applies *within* one run's files, as a disk-fill safety net. |
| `debug` / `--debug` | Live window: raw feed beside perception's annotated view. Needs a GUI OpenCV build and a display; silently disables itself otherwise. |
| `debug_window.max_width`, `min_interval_s` | Cap rendered width and throttle redraws so the window can't become the bottleneck. |
| `perception.visualize_dir` | Write rectified + boxes-drawn frames and a state JSON on every capture. Costs disk I/O; `null` to skip. |

Logs go to `logs/run_<start-time>/` — one directory per run, with
`robot_controller.log` (combined) plus `modules/<logger>.log` per module, every
line tagged `[turn=<color>]`.

**Dev tools** (off by default; all write under the repo root):

| Section | Trigger | Output |
| --- | --- | --- |
| `snapshot` | type `key` + Enter | Saves the latest camera frame to `output_dir`. |
| `detection_recording` | `start_key` / `stop_key` + Enter | Saves detection snapshots as JSON every `interval_s`. |
| `stall_capture` | automatic | Saves raw + annotated frames every `interval_s` while a capture is running — for diagnosing a headless run stuck in a Wait-for-… phase. |

The console keys share one stdin dispatcher, so they must all differ (startup
validates this) and work best with `manipulation.require_confirmation: false`.

## Recalibrating the board

Run from `modules/perception/`:

```bash
# 1. Collect photos of the physical board (ENTER captures, Q/ESC quits)
uv run python scripts/collect_data.py --output-dir data/ludo/raw

# 2. Re-derive the geometry constants from those photos
uv run python scripts/calibrate_ludo_board_config.py

# 3. Regenerate the cells: section (writes ../common/configs/ludo/cells.yaml)
uv run python scripts/generate_ludo_board_config.py

# 4. Check the result on a real photo — cell markers should land where expected
uv run python scripts/run_inference.py --game ludo --image data/ludo/raw/c_1_Color.png \
    --config configs/ludo/inference.yaml --turn red --visualize-dir outputs/viz
```

Step 2 **prints** `LANE_OFFSET`, `DEPTH_MARGIN`, `DEPTH_STEP`, and
`YARD_OFFSETS`; paste them into the constants at the top of
`scripts/generate_ludo_board_config.py`. Those are the geometry knobs — never
individual cell centers. Step 3 writes a standalone `cells.yaml`; copy its
contents over `board.yaml`'s own `cells:` section (or pass `--output` /
`--print-only`).

**Orientation caveat:** ArUco corner roles are assigned by each marker's
position in the raw frame, not by ID. A board rotated relative to the
calibrated orientation silently breaks the per-color yard/cell mapping.

## Training

`scripts/train.py` reads a training config and validates the dataset first:

```bash
cd modules/perception
uv run python scripts/train.py --config configs/ludo/board_train.yaml
```

| `*_train.yaml` key | Effect |
| --- | --- |
| `model` | Base checkpoint to fine-tune from (`yolo26n.pt`). |
| `data` | Path to the `*_dataset.yaml` (resolved relative to the train config). |
| `epochs`, `batch`, `imgsz`, `patience`, `seed`, `workers` | Standard Ultralytics knobs. |
| `project` / `name` | Output goes to `<project>/<name>/`; the checkpoint is `weights/best.pt`. |
| `val_viz_every_steps` | Render predictions on the val set every N batches (`0` disables). |
| `*_dataset.yaml` → `path`, `train`, `val`, `names` | Dataset root (relative to the dataset file), splits, class map. |

**Caveat:** this path only trains **box-only** detectors — its label validation
accepts `class cx cy w h` and nothing else. The active Ludo checkpoint
(`models/best.pt`) is a YOLO26-**pose** model (box + center/head keypoints,
10 classes) fine-tuned in the sibling *Auto-Labeling* repo, as is the dice
classifier. `board_train`/`dice_train`/`dice.yaml` describe an earlier
two-box-model design and are kept for reference. Copy a trained
`weights/best.pt` into `modules/perception/models/` and make sure
`inference.yaml`'s `class_names` matches its training class order.

Export for on-device use:

```bash
uv run python -m perception.training.export --weights runs/detect/<name>/weights/best.pt --format onnx
uv run python -m perception.training.export_npu --help   # FP32 ONNX -> static INT8 for QNN
```

## Knobs that are *not* in YAML

Occasionally the thing you need to change is a module constant:

| Constant | File | Effect |
| --- | --- | --- |
| `PIECE_REFERENCE_Y_INSET_FRAC` | [ludo/detector.py](modules/perception/src/perception/ludo/detector.py) | How far up from the bbox bottom edge a pawn's cell-assignment point sits. Raise if pieces snap one cell too far "down". |
| `_ROI_MARGIN`, `DETECTION_SCALES` | [rectification/aruco.py](modules/perception/src/perception/rectification/aruco.py) | Corner-hint search radius and the multi-scale sweep's retry scales. |
| Hough params (`dp`, `minDist`, `param1`, `param2`) | `locate_bowl` in [ludo/dice_reader.py](modules/perception/src/perception/ludo/dice_reader.py) | Bowl-rim detection sensitivity, beyond the configurable radii. |
| `_STABLE_READ_ATTEMPTS`, `_STABLE_READ_REQUIRED_MATCHES` | [handlers/robot_movement.py](modules/gameplay/src/gameplay/handlers/robot_movement.py) | Re-reads (and agreement) required to confirm the robot's own move. |
| `input_size` | [detection/npu_detector.py](modules/perception/src/perception/detection/npu_detector.py) | Letterbox size for the NPU path (640), not exposed in config. |

## Troubleshooting

| Symptom | First knob to try |
| --- | --- |
| "Could not detect all 4 board corner markers" | Framing/lighting; then `aruco.corner_marker_ids`, `aruco.full_sweep_backoff`. |
| Stuck in Wait-for-dice forever | `roll_detection.yaml` → lower `motion.area_ratio`, lower `stability.window`, or `validity.enabled: false`. |
| Stuck in Wait-for-children's-movement | Same knobs in `movement_detection.yaml`. |
| Roll confirmed but rejected as "pieces moved" | Raise `validity.max_confirm_attempts`, or raise `model.conf_threshold` to kill phantom pieces. |
| Pieces assigned to the wrong cell | Recalibrate `cells:`; then `PIECE_REFERENCE_Y_INSET_FRAC`. |
| Wrong piece colors detected | `inference.yaml` → `model.class_names` order vs. the checkpoint. |
| Die read as the wrong face | Try `dice_reading.method: bowl_classifier`; tune `bowl_min_radius`/`bowl_max_radius`. |
| Everything is slow on the edge board | `model.use_npu: true`, raise `runtime.tick_interval_s`, `debug: false`, `perception.visualize_dir: null`. |
| Console input gets "stolen" | Set `manipulation.require_confirmation: false` while using the dev-tool keys. |

## Tests

```bash
uv run pytest tests/                # robot_controller (camera, config, adapters, dev tools)
uv run pytest modules/<name>/tests/ # e.g. modules/gameplay/tests/, modules/perception/tests/
```

## Learn more

Each module's README covers its own layer: the
[gameplay state machine](modules/gameplay/README.md) and the
[perception pipeline and board calibration](modules/perception/README.md).
