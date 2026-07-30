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
[`ludo/track.py`](src/perception/ludo/track.py)) plus a separately
detected dice roll (via [`ludo/dice.py`](src/perception/ludo/dice.py)).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management,
as part of the repo-root workspace — run `uv sync` from the repo root, not
from this directory.

Make sure all the boards have ArUco markers attached in the 4 corners. Ludo's
dice bowl also has its own 4 corner ArUco markers (separate from the board's),
so the dice area is rectified to a top-down view independently — see
`configs/ludo/dice.yaml`.

Copy configuration files and modify them according to your board design and dataset:

```bash
# OAQ
for f in board dataset inference train; do cp configs/oaq/$f.yaml.example configs/oaq/$f.yaml; done
```
Ludo's configs (`configs/ludo/`) don't have `.example` counterparts yet — copy
and edit `board.yaml`/`inference.yaml`/`board_train.yaml`/`dice_train.yaml`/
`board_dataset.yaml`/`dice_dataset.yaml` directly; all the geometry, weights
paths, and dice ROIs in them are placeholders.

All commands below are run from this directory (`modules/perception/`).

## Usage

### Run inference on an image

```bash
# OAQ
uv run python scripts/run_inference.py --game oaq --image path/to/frame.jpg --config configs/oaq/inference.yaml

# Ludo (needs --turn: whose turn it currently is)
uv run python scripts/run_inference.py --game ludo --image path/to/frame.jpg --config configs/ludo/inference.yaml --turn red
```

### Prepare a fine-tuning dataset

1. Drop raw camera captures in `data/raw/`.
2. Rectify them in bulk (works for any board/bowl — pass the matching config,
   e.g. `configs/oaq/board.yaml`, `configs/ludo/board.yaml`, or
   `configs/ludo/dice.yaml` for dice-bowl captures):

   ```bash
   uv run python scripts/rectify_dataset.py --config configs/oaq/board.yaml --input data/raw --output data/rectified
   ```
3. Label the rectified images with YOLO-format boxes, using the classes
   defined in the matching `*_dataset.yaml`:
   - OAQ (`configs/oaq/dataset.yaml`): `0: mandarin_piece`, `1: ordinary_piece`
   - Ludo pawns (`configs/ludo/board_dataset.yaml`): `0: red`, `1: green`, `2: yellow`, `3: blue`
   - Ludo dice (`configs/ludo/dice_dataset.yaml`): `0: pip_1` .. `5: pip_6`
     (rectify dice-bowl captures with `configs/ludo/dice.yaml`, then
     crop/label individual dice faces)
4. Move/split the labeled images + `.txt` labels into
   `data/<dataset_name>/images/{train,val}` and `data/<dataset_name>/labels/{train,val}`,
   matching each `*_dataset.yaml`'s `path:`.

### Fine-tune an object detector

Check that every image has a same-named `.txt` file in the corresponding
`labels/train` or `labels/val` directory. The training command validates the
dataset and prints the image/object counts before loading the model. It's the
same command for any game/model — just point `--config` at the right training
config:

```bash
# OAQ pieces
uv run python scripts/train.py --config configs/oaq/train.yaml

# Ludo pawns
uv run python scripts/train.py --config configs/ludo/board_train.yaml

# Ludo dice
uv run python scripts/train.py --config configs/ludo/dice_train.yaml
```

Training outputs are written to `runs/detect/<name>/` (`name:` in each
train config); the checkpoint to use for inference is
`runs/detect/<name>/weights/best.pt`, copied to the `weights:` path each
inference config points at (e.g. `models/best.pt`, `models/board_best.pt`,
`models/dice_best.pt`).

### Export a trained checkpoint for on-device deployment

```bash
uv run python -m perception.training.export --weights runs/detect/<name>/weights/best.pt --format onnx
```

## Calibration note

Recalibrate `cells:`/`entry_offsets:`/`aruco:` in each `board.yaml`, and
`aruco:`/`rectification:` in Ludo's `dice.yaml`, against your actual rectified
images — everything checked in is placeholder geometry.
