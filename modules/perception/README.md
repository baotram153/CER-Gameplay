# Perception

Computer-vision module that reads the physical state of a board game from a
camera image, for the Child Engagement Robot. Two games are supported so far,
each as its own sub-package alongside shared detection/rectification/training
infrastructure:

- **Ô Ăn Quan (OAQ)** — [`src/perception/oaq/`](src/perception/oaq/): piece
  counts per cell. See [`oaq/pipeline.py`](src/perception/oaq/pipeline.py).
- **Cờ cá ngựa (Ludo)** — [`src/perception/ludo/`](src/perception/ludo/):
  16 pawn positions + dice roll. See [`ludo/pipeline.py`](src/perception/ludo/pipeline.py).

## Overall inference pipeline

```
Board Rectification -> Object Detection (YOLO26n, fine-tuned) -> Per-game state extraction
```

OAQ's "state extraction" step is per-cell piece counting; Ludo's is mapping
each detected pawn to a track position (via
[`ludo/track.py`](src/perception/ludo/track.py)) plus a die reading (via
[`ludo/dice.py`](src/perception/ludo/dice.py)). Unlike OAQ, Ludo's object
detector ([`ludo/detector.py`](src/perception/ludo/detector.py)) is a single
YOLO26-**pose** checkpoint that finds pawns and the die together in one
pass — each detection carries a box plus 2 keypoints (center, head); pieces
are assigned to a cell using the box's lower edge (where a standing pawn
actually touches the board), not the keypoints or the box centroid. See
[`ludo/pipeline.py`](src/perception/ludo/pipeline.py) for the full
rectify → detect → assign → snapshot flow, and
[`ludo/models.py`](src/perception/ludo/models.py) for the structured
`LudoBoardSnapshot` it produces (the strict `BoardState` game state, plus
each detection's confidence/bbox/keypoints for visualization and other
consumers like `manipulation`).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management,
as part of the repo-root workspace — run `uv sync` from the repo root, not
from this directory.

Make sure all the boards have ArUco markers attached in the 4 corners.
Ludo's dice bowl sits next to the board in the same raw frame with no
corner markers of its own — rectification keeps the *entire* frame in view
(`perception.rectification.rectify_keep_frame`) instead of cropping to the
board quad, so the bowl stays visible for detection without needing a
separate rectification pass.

Board-layout configs (`board.yaml`) are shared with `game_engine` and live
under `../common/configs/<game>/` instead of this module's own `configs/`.
Copy configuration files and modify them according to your board design and dataset:

```bash
# OAQ
cp ../common/configs/oaq/board.example.yaml ../common/configs/oaq/board.yaml
for f in dataset inference train; do cp configs/oaq/$f.example.yaml configs/oaq/$f.yaml; done

# Ludo
cp configs/ludo/inference.example.yaml configs/ludo/inference.yaml
cp configs/ludo/roll_detection.example.yaml configs/ludo/roll_detection.yaml
cp configs/ludo/movement_detection.example.yaml configs/ludo/movement_detection.yaml
```
`../common/configs/ludo/board.yaml` (aruco marker IDs, `cells:` geometry) is
committed directly rather than via a `.example.yaml` template — regenerate
its `cells:` section with `scripts/generate_ludo_board_config.py` rather
than hand-editing it, and recalibrate both it and `configs/ludo/inference.yaml`
against your physical board (see "Calibration note" below).

Ludo's active inference path (`configs/ludo/inference.yaml`) runs one
combined YOLO26-pose checkpoint (`models/best.pt`) that detects both pawns
and the die in a single pass — box + center/head keypoints per detection.
`configs/ludo/board_train.example.yaml`/`dice_train.example.yaml`/
`board_dataset.example.yaml`/`dice_dataset.example.yaml`/`dice.example.yaml`
describe an earlier two-separate-model design (a box-only pawn detector + a
box-only dice detector with its own dice-bowl rectification) and are kept
for reference only — they aren't wired into the current pipeline.

`configs/ludo/roll_detection.yaml` holds every hyperparameter for
[`ludo/roll_detector.py`](src/perception/ludo/roll_detector.py)'s
motion/stability/validity thresholds (loaded via
`RollDetector.from_config_file`) — this is the one place to tune those
knobs against a real camera/lighting setup; see that file's own comments
for what each value controls.

`configs/ludo/movement_detection.yaml` is the same schema for
[`ludo/movement_detector.py`](src/perception/ludo/movement_detector.py)'s
`MovementDetector` (`.from_config_file`) — the analogous sub-machine for
confirming a settled piece move instead of a settled dice roll; tune it
separately since a piece slide is a slower, larger motion than a die
settling.

All commands below are run from this directory (`modules/perception/`).

## Usage

### Run inference on an image

```bash
# OAQ
uv run python scripts/run_inference.py --game oaq --image path/to/frame.jpg --config configs/oaq/inference.yaml

# Ludo (needs --turn: whose turn it currently is)
uv run python scripts/run_inference.py --game ludo --image path/to/frame.jpg --config configs/ludo/inference.yaml --turn red
```

Pass `--visualize-dir outputs/viz` to also save the rectified image and a
boxes-drawn copy (`rectified/`/`boxes/`, named after `--image`); for
`--game ludo` this also writes a structured board-state JSON snapshot
(`states/<image>.json` — pieces' assigned cells + confidence + center/head
keypoints, the dice reading, and the `common.type.BoardState` it resolves to).

### Prepare a fine-tuning dataset (OAQ)

1. Drop raw camera captures in `data/raw/`.
2. Rectify them in bulk:

   ```bash
   uv run python scripts/rectify_dataset.py --config ../common/configs/oaq/board.yaml --input data/raw --output data/rectified
   ```
3. Label the rectified images with YOLO-format boxes, using the classes in
   `configs/oaq/dataset.yaml`: `0: mandarin_piece`, `1: ordinary_piece`.
4. Move/split the labeled images + `.txt` labels into
   `data/<dataset_name>/images/{train,val}` and `data/<dataset_name>/labels/{train,val}`,
   matching `dataset.yaml`'s `path:`.

### Fine-tune an object detector (OAQ)

Check that every image has a same-named `.txt` file in the corresponding
`labels/train` or `labels/val` directory. The training command validates the
dataset and prints the image/object counts before loading the model:

```bash
uv run python scripts/train.py --config configs/oaq/train.yaml
```

Training outputs are written to `runs/detect/<name>/` (`name:` in the
train config); the checkpoint to use for inference is
`runs/detect/<name>/weights/best.pt`, copied to `models/best.pt` (the
`weights:` path `configs/oaq/inference.yaml` points at).

### Retraining Ludo's pose checkpoint

`configs/ludo/board_train.example.yaml`/`dice_train.example.yaml` +
`board_dataset.example.yaml`/`dice_dataset.example.yaml`/`dice.example.yaml`
describe the *earlier* two-separate-box-only-model design (see "Setup"
above) — `scripts/train.py`'s dataset validation only accepts plain
`class cx cy w h` box labels, so it can't retrain the active pose checkpoint
directly; those configs are kept only as a reference for that older design,
not a working recipe for `models/best.pt`.

`models/best.pt` was fine-tuned in the sibling Auto-Labeling repo (not part
of this one) as a YOLO26-pose model — box + center/head keypoints,
`kpt_shape: [2, 3]` — on 10 classes (`piece_red`/`green`/`yellow`/`blue`,
`dice_1`..`dice_6`), following `configs/yolo26l-pose-2kpt.yaml` there. To
retrain it: label a pose dataset the same way (Auto-Labeling's
`scripts/rectify_dataset.py` + `scripts/auto_label.py`/manual labeling +
`scripts/train.py` with a `model_yaml`/`kpt_shape` config), then copy the
resulting `weights/best.pt` here to `modules/perception/models/best.pt` —
`configs/ludo/inference.yaml`'s `model.class_names` must match the class
order it was trained with.

### Export a trained checkpoint for on-device deployment

```bash
uv run python -m perception.training.export --weights runs/detect/<name>/weights/best.pt --format onnx
```

## Calibration note

Recalibrate `cells:`/`entry_offsets:`/`aruco:` in each game's `board.yaml`
(under `../common/configs/<game>/`, shared with `game_engine`) against your
actual rectified images — OAQ's is entirely placeholder geometry. Ludo's
`aruco.corner_marker_ids` is calibrated against the real board in
`data/ludo/raw/*.png`, but `cells:` is still approximate: it's generated
from a documented 15x15-grid formula (see
`scripts/generate_ludo_board_config.py`) that gets the topology right
(which arm/yard belongs to which color) but not exact pixel placement —
recalibrate by adjusting that script's parameters and re-running it, not by
hand-editing individual cell centers, then re-run
`scripts/run_inference.py --game ludo --visualize-dir ...` on a real photo
and check the cell markers in `boxes/<image>.png` land where you expect.

ArUco corner roles (`top_left`/etc.) are assigned by each marker's position
in the raw frame, not by ID (`perception.rectification.detect_corner_markers`)
— this assumes the board is placed in a consistent orientation relative to
the camera across captures. A board rotated relative to that (observed in
one of the Ludo sample photos) silently breaks the per-color yard/cell
mapping above, since "top_left" no longer lands on the same physical board
corner it was calibrated against.
