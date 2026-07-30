"""Fine-tune YOLO26n on the labeled OAQ piece dataset."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_VAL_VIZ_EVERY_STEPS = 100


def _resolve_path(value: str, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (relative_to / path).resolve()


def _load_dataset(data_config: str, relative_to: Path) -> tuple[dict[str, Any], Path]:
    data_path = _resolve_path(data_config, relative_to)
    dataset = yaml.safe_load(data_path.read_text())
    dataset_root = _resolve_path(dataset["path"], data_path.parent)
    dataset["path"] = str(dataset_root)
    return dataset, dataset_root


def _validate_dataset(dataset: dict[str, Any], dataset_root: Path) -> None:
    names = dataset.get("names", {})
    class_ids = {int(class_id) for class_id in names}
    counts: Counter[int] = Counter()
    errors: list[str] = []

    for split in ("train", "val"):
        image_dir = _resolve_path(dataset[split], dataset_root)
        label_dir = dataset_root / "labels" / image_dir.relative_to(dataset_root / "images")
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: expected directories {image_dir} and {label_dir}")
            continue
        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        labels = {path.stem: path for path in label_dir.glob("*.txt")}

        missing_labels = sorted(images.keys() - labels.keys())
        missing_images = sorted(labels.keys() - images.keys())
        if missing_labels:
            errors.append(f"{split}: images without labels: {', '.join(missing_labels[:5])}")
        if missing_images:
            errors.append(f"{split}: labels without images: {', '.join(missing_images[:5])}")
        if not images:
            errors.append(f"{split}: no images found in {image_dir}")

        for label_path in labels.values():
            for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                except (ValueError, IndexError):
                    errors.append(f"{label_path}:{line_number}: invalid YOLO label")
                    continue
                if len(coordinates) != 4 or not all(0 <= value <= 1 for value in coordinates):
                    errors.append(
                        f"{label_path}:{line_number}: expected four normalized coordinates"
                    )
                if class_id not in class_ids:
                    errors.append(f"{label_path}:{line_number}: unknown class ID {class_id}")
                counts[class_id] += 1

        print(f"{split}: {len(images)} images, {len(labels)} label files")

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))

    class_summary = ", ".join(
        f"{names[class_id]}={counts[class_id]}" for class_id in sorted(class_ids)
    )
    print(f"objects: {class_summary}")


@contextmanager
def _resolved_dataset_config(dataset: dict[str, Any]):
    """Write a temporary YAML because Ultralytics requires a path, not a dictionary."""
    with TemporaryDirectory(prefix="oaq-training-") as temporary_dir:
        data_path = Path(temporary_dir) / "dataset.yaml"
        data_path.write_text(yaml.safe_dump(dataset, sort_keys=False))
        yield data_path


def _collect_val_images(dataset: dict[str, Any], dataset_root: Path) -> list[Path]:
    val_dir = _resolve_path(dataset["val"], dataset_root)
    return sorted(
        path
        for path in val_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _make_val_visualization_callback(model_source: str, val_images: list[Path], every: int):
    """Build an `on_train_batch_end` callback that renders predictions on the
    validation images every `every` training batches, both as boxes-only and
    as boxes-with-class-labels images.
    """
    visual_model: YOLO | None = None
    step_count = 0

    def _on_train_batch_end(trainer) -> None:
        nonlocal step_count, visual_model
        step_count += 1
        if step_count % every != 0:
            return

        if visual_model is None:
            visual_model = YOLO(model_source)
        visual_model.model = deepcopy(trainer.model)
        visual_model.model.to("cpu").eval()

        live_weights = trainer.ema.ema if trainer.ema is not None else trainer.model
        state_dict = {key: value.detach().cpu() for key, value in live_weights.state_dict().items()}
        visual_model.model.load_state_dict(state_dict)

        step_dir = Path(trainer.save_dir) / "val_viz" / f"step_{step_count}"
        boxes_dir = step_dir / "boxes_only"
        classes_dir = step_dir / "with_classes"
        boxes_dir.mkdir(parents=True, exist_ok=True)
        classes_dir.mkdir(parents=True, exist_ok=True)

        results = visual_model.predict(
            source=[str(path) for path in val_images],
            imgsz=trainer.args.imgsz,
            device="cpu",
            verbose=False,
        )
        for image_path, result in zip(val_images, results):
            result.save(filename=str(boxes_dir / image_path.name), labels=False, conf=False)
            result.save(filename=str(classes_dir / image_path.name), labels=True, conf=True)

    return _on_train_batch_end


def train(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    dataset, dataset_root = _load_dataset(config["data"], config_path.parent)
    _validate_dataset(dataset, dataset_root)

    train_args = dict(
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        patience=config["patience"],
        project=config["project"],
        name=config["name"],
        seed=config["seed"],
        workers=config.get("workers", 4),
    )
    if "device" in config:
        train_args["device"] = config["device"]

    model = YOLO(config["model"])

    val_images = _collect_val_images(dataset, dataset_root)
    if val_images:
        every = config.get("val_viz_every_steps", DEFAULT_VAL_VIZ_EVERY_STEPS)
        if every > 0:
            model.add_callback(
                "on_train_batch_end",
                _make_val_visualization_callback(config["model"], val_images, every),
            )

    with _resolved_dataset_config(dataset) as data_path:
        model.train(data=str(data_path), **train_args)
