"""Structured per-frame Ludo perception output.

Bundles the strict `common.type.BoardState` (color + track position only —
the contract `reasoning`/`game_engine` consume) together with richer
per-detection data (assigned cell, bbox, confidence, and the pose
checkpoint's center/head keypoints) that BoardState has no room for, but
that visualization and other consumers (e.g. manipulation, for a grasp
point + orientation) need.
"""
from __future__ import annotations

from dataclasses import dataclass

from common.constants import Color
from common.type import BoardState


@dataclass
class Keypoints:
    center: tuple[float, float]
    head: tuple[float, float]


@dataclass
class PieceObservation:
    color: Color
    pos: int
    cell_id: str
    confidence: float
    bbox: tuple[float, float, float, float]
    keypoints: Keypoints | None


@dataclass
class DiceObservation:
    value: int
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class LudoBoardSnapshot:
    board_state: BoardState
    pieces: list[PieceObservation]
    dice: DiceObservation
