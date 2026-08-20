"""Adapters from robot_controller's infrastructure (camera, console) to the
Protocols gameplay.GameplayEngine actually depends on."""
from .manipulation_adapter import ConsoleManipulationAdapter
from .perception_adapter import LudoPerceptionAdapter

__all__ = ["LudoPerceptionAdapter", "ConsoleManipulationAdapter"]
