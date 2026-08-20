"""Composition root: wires camera -> perception -> gameplay -> manipulation
into one running Ludo game, from a single AppConfig.

`run()` is deliberately not `GameplayEngine.run()` (which just loops
step() to completion) -- it needs to control the polling cadence between
steps, log phase transitions, detect a phase the FSM is stuck in, and
decide which errors are routine-log-and-continue vs. fatal-stop-cleanly.
That policy belongs here, in the composition root, not in gameplay itself.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import yaml
from gameplay import handlers
from gameplay.engine import GameplayEngine
from gameplay.errors import GameplayError
from gameplay.move_selection import action_planner_move_selector
from gameplay.phase import GamePhase
from gameplay.result import GameResult
from perception.ludo import LudoStatePipeline
from reasoning.game_engine import GameState

from .adapters.manipulation_adapter import ConsoleManipulationAdapter
from .adapters.perception_adapter import LudoPerceptionAdapter
from .camera.base import FrameSource
from .camera.directory_camera import DirectoryFrameSource
from .camera.realsense_camera import RealSenseCamera
from .config import REPO_ROOT, AppConfig, CameraConfig
from .console_keys import ConsoleKeyDispatcher
from .debug_window import DebugWindow
from .detection_recorder import DetectionResultRecorder
from .errors import CameraError
from .snapshot_saver import SnapshotSaver

logger = logging.getLogger(__name__)

# modules/perception/configs/ludo/inference.yaml's own `board_config` and
# `model.weights`/`model.fallback_weights` keys are relative paths that
# modules/perception/scripts/run_inference.py only resolves correctly by
# convention (it must be run with modules/perception/ as the CWD -- see
# LudoStatePipeline.__init__/ObjectDetector.__init__, which both just do
# Path(value) against whatever the process's CWD happens to be). robot_controller
# has no reason to depend on its own CWD that way, so _load_ludo_pipeline
# below resolves those keys itself, relative to modules/perception/,
# before constructing LudoStatePipeline directly -- this works regardless
# of where main.py is actually invoked from.
_PERCEPTION_ROOT = REPO_ROOT / "modules" / "perception"


def _resolve_relative_to_perception(value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    return str(path if path.is_absolute() else (_PERCEPTION_ROOT / path).resolve())


def _load_ludo_pipeline(inference_config_path: Path) -> LudoStatePipeline:
    raw = yaml.safe_load(Path(inference_config_path).read_text())
    raw["board_config"] = _resolve_relative_to_perception(raw["board_config"])
    raw["model"]["weights"] = _resolve_relative_to_perception(raw["model"]["weights"])
    raw["model"]["fallback_weights"] = _resolve_relative_to_perception(raw["model"].get("fallback_weights"))
    return LudoStatePipeline(raw)


def build_camera(config: CameraConfig) -> FrameSource:
    if config.backend == "realsense":
        return RealSenseCamera(
            width=config.width,
            height=config.height,
            fps=config.fps,
            serial_number=config.serial_number,
            frame_timeout_ms=config.frame_timeout_ms,
            max_consecutive_errors=config.max_consecutive_errors,
            max_reconnect_attempts=config.max_reconnect_attempts,
            reconnect_backoff_s=config.reconnect_backoff_s,
        )
    return DirectoryFrameSource(config.directory)


def _build_key_dispatcher(config: AppConfig) -> tuple[ConsoleKeyDispatcher | None, SnapshotSaver | None, DetectionResultRecorder | None]:
    """Wires the dev-tool console keys (snapshot save, detection-recording
    start/stop) onto one shared ConsoleKeyDispatcher -- see console_keys.py
    for why they must share a single dispatcher rather than each polling
    stdin independently. Returns (None, None, None) for whichever pieces
    aren't enabled; the dispatcher itself is None only when nothing is.
    """
    snapshot_saver = SnapshotSaver(config.snapshot.output_dir) if config.snapshot.enabled else None
    detection_recorder = (
        DetectionResultRecorder(config.detection_recording.output_dir, config.detection_recording.interval_s)
        if config.detection_recording.enabled
        else None
    )
    if snapshot_saver is None and detection_recorder is None:
        return None, None, None

    dispatcher = ConsoleKeyDispatcher()
    if snapshot_saver is not None:
        dispatcher.on(config.snapshot.key, snapshot_saver.trigger)
    if detection_recorder is not None:
        dispatcher.on(config.detection_recording.start_key, detection_recorder.start)
        dispatcher.on(config.detection_recording.stop_key, detection_recorder.stop)
    return dispatcher, snapshot_saver, detection_recorder


def build_engine(config: AppConfig, camera: FrameSource, debug_window: DebugWindow | None = None) -> GameplayEngine:
    pipeline = _load_ludo_pipeline(config.perception.inference_config)
    key_dispatcher, snapshot_saver, detection_recorder = _build_key_dispatcher(config)

    perception = LudoPerceptionAdapter(
        camera=camera,
        pipeline=pipeline,
        visualize_dir=config.perception.visualize_dir,
        key_dispatcher=key_dispatcher,
        snapshot_saver=snapshot_saver,
        detection_recorder=detection_recorder,
        debug_window=debug_window,
    )
    manipulation = ConsoleManipulationAdapter(require_confirmation=config.manipulation.require_confirmation)

    # entry_offsets/num_shared_steps come from the pipeline (which already
    # loaded board.yaml) rather than re-parsing it here, so the board
    # layout is only ever read from one place.
    game = GameState.new_game(
        players=config.game.players,
        entry_offsets=pipeline.entry_offsets,
        num_shared_steps=pipeline.num_shared_steps,
    )
    move_selector = action_planner_move_selector(pipeline.entry_offsets, pipeline.num_shared_steps)

    return GameplayEngine(
        game=game,
        player_roles=config.game.player_roles,
        perception=perception,
        manipulation=manipulation,
        move_selector=move_selector,
    )


def run(config: AppConfig) -> GameResult | None:
    """Runs one game to completion (or until an unrecoverable error, or
    `runtime.max_steps` is exhausted). Returns the GameResult on a normal
    win, or None if the run ended any other way -- callers that care why
    should read the log, not branch on this.
    """
    camera = build_camera(config.camera)
    debug_window = DebugWindow() if config.debug else None

    try:
        camera.start()
    except CameraError:
        logger.critical("Could not start the camera; aborting.", exc_info=True)
        return None

    try:
        engine = build_engine(config, camera, debug_window)
    except Exception:
        logger.critical("Could not build the gameplay engine; aborting.", exc_info=True)
        camera.stop()
        return None

    logger.info(
        "Starting game loop (players=%s, roles=%s)",
        config.game.players, config.game.player_roles,
    )

    try:
        result = _run_loop(engine, config)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; shutting down.")
        result = None
    finally:
        camera.stop()
        if debug_window is not None:
            debug_window.close()

    if result is not None:
        logger.info(
            "Game finished: winner=%s (%s), turns_played=%d",
            result.winner, result.winner_role, result.turns_played,
        )
    return result


def _run_loop(engine: GameplayEngine, config: AppConfig) -> GameResult | None:
    last_phase: GamePhase | None = None
    stuck_attempts = 0

    for step_count in range(1, config.runtime.max_steps + 1):
        try:
            phase = engine.step()
        except CameraError:
            logger.critical("Camera failed unrecoverably; stopping the game loop.", exc_info=True)
            return None
        except GameplayError:
            logger.critical("Gameplay engine reported a fatal error; stopping the game loop.", exc_info=True)
            return None
        except Exception:
            # Anything else (an adapter bug, an unexpected detector crash
            # that slipped past LudoPerceptionAdapter's own handling, ...)
            # is logged loudly but doesn't end the game -- next tick tries
            # again from wherever the FSM's phase already was.
            logger.error("Unexpected error during engine.step(); skipping this tick.", exc_info=True)
            time.sleep(config.runtime.tick_interval_s)
            continue

        if phase is last_phase:
            stuck_attempts += 1
            if stuck_attempts % config.runtime.stuck_warning_attempts == 0:
                logger.warning(
                    "Still in phase %s after %d attempts (dice_attempts=%d, movement_attempts=%d) -- "
                    "check camera framing/lighting, or whether a human still needs to act.",
                    phase, stuck_attempts, engine.context.dice_attempts, engine.context.movement_attempts,
                )
        else:
            logger.info("Phase -> %s", phase)
            stuck_attempts = 0
        last_phase = phase

        if phase is GamePhase.END_GAME:
            return handlers.end_game.build_result(engine.context)

        time.sleep(config.runtime.tick_interval_s)

    logger.error("Game did not reach END_GAME within %d steps; stopping.", config.runtime.max_steps)
    return None
