"""Object detection (YOLO26n)."""
from .detector import Detection, ObjectDetector
from .npu_detector import NpuObjectDetector

__all__ = ["Detection", "ObjectDetector", "NpuObjectDetector"]
