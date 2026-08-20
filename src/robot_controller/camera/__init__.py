"""Data ingestion: turning a physical camera (or a stand-in, for local
development) into a stream of raw BGR frames -- the FrameSource protocol."""
from .base import FrameSource
from .directory_camera import DirectoryFrameSource
from .realsense_camera import RealSenseCamera

__all__ = ["FrameSource", "RealSenseCamera", "DirectoryFrameSource"]
