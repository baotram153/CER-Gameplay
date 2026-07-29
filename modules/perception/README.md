# OAQ State Detection

Computer-vision module that reads the physical state of an "Ô Ăn Quan" (OAQ) board
(piece counts per cell) from a camera image, for the Child Engagement Robot.

## Overall Inference Pipeline

```
Board Rectification -> Object Detection (YOLO26n, fine-tuned)-> Per-cell Counting
```

See [`src/oaq_state_detection/pipeline.py`](src/oaq_state_detection/pipeline.py).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
uv sync
```
Copy configuration files and modify them according to your board design and dataset:
```bash
for f in board dataset inference train; do cp configs/$f.yaml.example configs/$f.yaml; done
```

## Usage

### Run inference on an image

```bash
uv run python scripts/run_inference.py --image path/to/frame.jpg --config configs/inference.yaml
```

### Prepare a fine-tuning dataset

1. Drop raw camera captures in `data/raw/`.
2. Rectify them in bulk:

   ```bash
   uv run python scripts/rectify_dataset.py --config configs/board.yaml --input data/raw --output data/rectified
   ```

3. Label the rectified images in `data/rectified/` with YOLO-format, using the two classes
   defined in `configs/dataset.yaml`.
4. Move/split the labeled images + `.txt` labels into
   `data/dataset/images/{train,val}` and `data/dataset/labels/{train,val}`.

### Fine-tune object detection model

The class IDs in each YOLO label must be:

- `0`: `mandarin_piece`
- `1`: `ordinary_piece`

Check that every image has a same-named `.txt` file in the corresponding
`labels/train` or `labels/val` directory. The training command validates the
dataset and prints the image/object counts before loading the model.

Finetuning command:

```bash
uv run python scripts/train.py --config configs/train.yaml
```
Training outputs are written to
`runs/detect/oaq_yolo26n/`; the checkpoint to use for inference is
`runs/detect/oaq_yolo26n/weights/best.pt`.

### Export a trained checkpoint for on-device deployment

```bash
uv run python -m oaq_state_detection.training.export --weights runs/detect/train/weights/best.pt --format onnx
```

## Calibration note

Recalibrate
`cells:` against your actual rectified board image. Each cell layout is normalized to the rectified image.
