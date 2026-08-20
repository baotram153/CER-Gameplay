"""Exceptions raised by the robot_controller composition root."""
from __future__ import annotations


class RobotControllerError(Exception):
    """Base class for errors raised by robot_controller."""


class ConfigError(RobotControllerError):
    """The app configuration file is missing, unreadable, or invalid."""


class CameraError(RobotControllerError):
    """The camera could not be started, or failed unrecoverably while
    running (e.g. reconnection attempts were exhausted).

    Transient per-frame hiccups (a single dropped frame, a momentary
    timeout) are NOT reported this way -- FrameSource.read() just returns
    None for those, the same "no reading this tick" signal a bad camera
    angle would produce. This is reserved for failures a caller should
    treat as fatal to the current run.
    """
