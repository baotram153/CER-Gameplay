"""Object detection (YOLO26n) and image classification."""
from .classifier import ImageClassifier
from .detector import Detection, ObjectDetector
from .npu_detector import NpuObjectDetector

__all__ = ["Detection", "ObjectDetector", "NpuObjectDetector", "ImageClassifier"]
