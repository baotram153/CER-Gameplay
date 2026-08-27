"""Two interchangeable strategies for reading the current die's face value
each perception tick:

- ModelDiceReader: the existing behavior -- the die is just one more class
  ("dice_<1-6>") on the same combined pawn+dice checkpoint LudoDetector
  already runs for pieces, so this is free once that detection pass has
  happened.
- BowlClassifierDiceReader: classical CV to find the ROI, a model to read
  it. Locates the dice bowl itself each frame (Hough Circle Transform
  against its rim, via locate_bowl) rather than assuming a fixed pixel
  crop -- the bowl is a separate physical object from the board and can be
  repositioned independently of it, so a fixed crop drifts out of
  alignment the moment either one moves. The resulting crop is handed to a
  dedicated dice classification model (DiceClassifier) -- a different,
  smaller checkpoint from the combined pawn+dice pose model
  ModelDiceReader uses, trained only to read a die's face value from a
  bowl crop, not to localize anything itself.

Both implement the same DiceReader.read(rectified, detections) -> (value,
Detection) contract pick_dice_value already has -- raising ValueError for
"no confident single-die reading this tick" -- so either can be swapped in
wherever dice.pick_dice_value used to be called directly, via
build_dice_reader's config-driven choice.
"""
from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

from ..detection import Detection, ImageClassifier
from .detector import LudoDetector
from .dice import pick_dice_value

logger = logging.getLogger(__name__)


def locate_bowl(
    image: np.ndarray,
    min_radius: int,
    max_radius: int,
    search_region: tuple[int, int, int, int] | None = None,
) -> tuple[float, float, float] | None:
    """Finds the dice bowl's own circular rim via Hough Circle Transform,
    rather than assuming a fixed pixel location -- see this module's
    docstring for why. Returns (cx, cy, r) in `image`'s own coordinate
    space (already offset back if search_region was given), or None if no
    circle was found.

    Validated against a 93-image real test set spanning two different
    physical camera setups: the single strongest Hough candidate (ranked
    by accumulator score, i.e. circles[0][0]) was the correct bowl every
    time, with zero misses. Still, min_radius/max_radius are the bowl's
    expected pixel size, which is camera-distance-dependent -- recalibrate
    for a different setup.
    """
    if search_region is not None:
        sx, sy, sw, sh = search_region
        search_image = image[sy : sy + sh, sx : sx + sw]
    else:
        sx, sy = 0, 0
        search_image = image
    if search_image.size == 0:
        return None

    gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=max(search_image.shape[1] // 4, 1),
        param1=100,
        param2=60,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None
    cx, cy, r = circles[0][0]
    return float(cx + sx), float(cy + sy), float(r)


class DiceReader(Protocol):
    def read(self, rectified: np.ndarray, detections: list[Detection]) -> tuple[int, Detection]:
        """Returns (face_value, detection) for the one die on the board
        this tick. Raises ValueError if no confident single-die reading
        exists (either strategy's routine "nothing to read yet" case --
        callers already treat that the same way pick_dice_value's own
        ValueError is treated)."""
        ...


class ModelDiceReader:
    """Wraps the shared LudoDetector's own dice_<1-6> classification --
    see this module's docstring."""

    def __init__(self, detector: LudoDetector) -> None:
        self._detector = detector

    def read(self, rectified: np.ndarray, detections: list[Detection]) -> tuple[int, Detection]:
        return pick_dice_value(self._detector.dice_candidates(detections))


class DiceClassifier:
    """Reads a die's face value (1-6) from a crop containing just the die
    (or the bowl around it), via a dedicated classification checkpoint --
    separate from LudoDetector's own pose model entirely. Used by
    BowlClassifierDiceReader against the ROI locate_bowl finds
    classically."""

    def __init__(self, classifier: ImageClassifier, class_names: dict[int, str]) -> None:
        self._classifier = classifier
        self._class_names = class_names

    def classify(self, crop: np.ndarray) -> tuple[int, float]:
        class_id, confidence = self._classifier.classify(crop)
        name = self._class_names.get(class_id, "")
        prefix, _, value = name.partition("_")
        if prefix != "dice" or not value.isdigit():
            raise ValueError(
                f"classifier's top class_id={class_id} (name={name!r}) isn't a dice_<1-6> class"
            )
        return int(value), confidence


class BowlClassifierDiceReader:
    """Classical CV to locate the ROI, a model to read it -- see this
    module's docstring.

    Ignores `detections` entirely; works straight off the rectified frame,
    in two stages:

    1. Locate the dice bowl itself via Hough Circle Transform (locate_bowl)
       -- not a fixed pixel crop. Validated against a 93-image real test
       set spanning two different physical camera setups: 0 misses.

    2. Crop around the located bowl and hand that crop to a dedicated dice
       classification model (DiceClassifier), which reports the face
       value directly. No die-silhouette segmentation, facet-splitting, or
       pip-counting happens here -- the classifier handles all of that
       implicitly, including a die propped at an angle against the bowl's
       curved wall.
    """

    def __init__(
        self,
        classifier: DiceClassifier,
        bowl_min_radius: int = 100,
        bowl_max_radius: int = 200,
        search_region: tuple[int, int, int, int] | None = None,
        bowl_crop_scale: float = 1.7,
        min_confidence: float = 0.0,
    ) -> None:
        self._classifier = classifier
        self._bowl_min_radius = bowl_min_radius
        self._bowl_max_radius = bowl_max_radius
        self._search_region = search_region
        self._bowl_crop_scale = bowl_crop_scale
        self._min_confidence = min_confidence

    def read(self, rectified: np.ndarray, detections: list[Detection]) -> tuple[int, Detection]:
        bowl = locate_bowl(rectified, self._bowl_min_radius, self._bowl_max_radius, self._search_region)
        if bowl is None:
            logger.debug("BowlClassifierDiceReader.read: no bowl located")
            raise ValueError("could not locate the dice bowl in this frame")
        bowl_x, bowl_y, bowl_r = bowl

        side = int(self._bowl_crop_scale * bowl_r)
        crop_x = max(int(bowl_x - side / 2), 0)
        crop_y = max(int(bowl_y - side / 2), 0)
        crop = rectified[crop_y : crop_y + side, crop_x : crop_x + side]
        if crop.size == 0:
            raise ValueError(f"located bowl crop is empty (bowl={bowl}, frame shape={rectified.shape})")

        value, confidence = self._classifier.classify(crop)
        if confidence < self._min_confidence:
            raise ValueError(
                f"classifier confidence {confidence:.3f} below min_confidence={self._min_confidence:.3f}"
            )

        bbox = (crop_x, crop_y, crop_x + crop.shape[1], crop_y + crop.shape[0])
        detection = Detection(
            bbox=bbox,
            center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            class_id=-1,
            confidence=confidence,
        )
        logger.debug(
            "BowlClassifierDiceReader.read: bowl=%s, value=%d (confidence=%.3f)", bowl, value, confidence,
        )
        return value, detection


def build_dice_reader(config: dict, detector: LudoDetector | None) -> DiceReader:
    """Builds the DiceReader named by config["method"] ("model", the
    default, or "bowl_classifier") -- see inference.example.yaml's
    dice_reading section for the schema. `detector` is only used by
    "model"; pass None when building a "bowl_classifier" reader."""
    method = config.get("method", "model")
    if method == "model":
        return ModelDiceReader(detector)
    if method == "bowl_classifier":
        bc_cfg = config["bowl_classifier"]
        search_region = bc_cfg.get("search_region")
        classifier_cfg = bc_cfg["classifier"]
        image_classifier = ImageClassifier(
            weights=classifier_cfg["weights"],
            fallback_weights=classifier_cfg.get("fallback_weights"),
            device=classifier_cfg.get("device"),
        )
        classifier = DiceClassifier(image_classifier, classifier_cfg["class_names"])
        return BowlClassifierDiceReader(
            classifier=classifier,
            bowl_min_radius=bc_cfg.get("bowl_min_radius", 100),
            bowl_max_radius=bc_cfg.get("bowl_max_radius", 200),
            search_region=tuple(search_region) if search_region is not None else None,
            bowl_crop_scale=bc_cfg.get("bowl_crop_scale", 1.7),
            min_confidence=bc_cfg.get("min_confidence", 0.0),
        )
    raise ValueError(f"dice_reading.method must be 'model' or 'bowl_classifier', got {method!r}")
