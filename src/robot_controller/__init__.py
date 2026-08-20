"""Composition root: wires a live camera, perception, gameplay, and
manipulation into one running game.

Every other module in this repo (common, gameplay, perception, reasoning,
manipulation) is a library with no opinion on where its inputs come from or
how often it's called. This package is the one place that actually owns a
camera, a config file, and a run loop -- see gameplay.engine.GameplayEngine's
docstring: "Whoever owns the camera... drives the cadence."
"""
from __future__ import annotations
