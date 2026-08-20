"""Centralized application configuration for robot_controller.

Single entry point (`load_config`) for every setting the composition root
needs -- camera, perception, manipulation, logging, runtime -- so nothing
about how the app is wired lives as a scattered literal in app.py. Backed by
one YAML file (see configs/robot_controller/app.example.yaml for the
schema and every key's meaning); a couple of ops-time knobs can also be
overridden by environment variable without editing the file, e.g. swapping
camera serial numbers between two robot rigs.

Following this repo's convention (see modules/*/configs), the real
app.yaml is gitignored -- it's device-specific -- and only app.example.yaml
is committed. Copy the example to get started.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from gameplay.player import PlayerType

from common.constants import Color

from .errors import ConfigError

# src/robot_controller/config.py -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "robot_controller" / "app.yaml"
CONFIG_PATH_ENV_VAR = "ROBOT_CONTROLLER_CONFIG"
CAMERA_SERIAL_ENV_VAR = "ROBOT_CAMERA_SERIAL"
LOG_LEVEL_ENV_VAR = "ROBOT_LOG_LEVEL"

_VALID_CAMERA_BACKENDS = {"realsense", "directory"}


@dataclass(frozen=True)
class CameraConfig:
    backend: str  # "realsense" | "directory"
    width: int
    height: int
    fps: int
    serial_number: str | None
    frame_timeout_ms: int
    max_consecutive_errors: int
    max_reconnect_attempts: int
    reconnect_backoff_s: float
    directory: Path | None  # only used when backend == "directory"


@dataclass(frozen=True)
class GameConfig:
    game: str  # currently only "ludo" is wired up
    players: list[Color]
    player_roles: dict[Color, PlayerType]


@dataclass(frozen=True)
class PerceptionConfig:
    inference_config: Path
    visualize_dir: Path | None


@dataclass(frozen=True)
class ManipulationConfig:
    require_confirmation: bool


@dataclass(frozen=True)
class SnapshotConfig:
    # Dev tool: type `key` + Enter on the console to save the most
    # recently read camera frame -- see src/robot_controller/snapshot_saver.py.
    enabled: bool
    key: str
    output_dir: Path


@dataclass(frozen=True)
class DetectionRecordingConfig:
    # Dev tool: type `start_key`/`stop_key` + Enter to toggle periodically
    # saving perception's detection result -- see
    # src/robot_controller/detection_recorder.py.
    enabled: bool
    start_key: str
    stop_key: str
    interval_s: float
    output_dir: Path


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_dir: Path
    file_name: str
    max_bytes: int
    backup_count: int
    console: bool


@dataclass(frozen=True)
class RuntimeConfig:
    # Delay between successive GameplayEngine.step() calls -- sets the
    # camera-polling cadence during the Wait-for-... phases.
    tick_interval_s: float
    max_steps: int
    # Log a warning if the FSM stays in the same phase for this many
    # consecutive step() calls (camera framing/lighting, or a human hasn't
    # acted yet).
    stuck_warning_attempts: int


@dataclass(frozen=True)
class AppConfig:
    game: GameConfig
    camera: CameraConfig
    perception: PerceptionConfig
    manipulation: ManipulationConfig
    logging: LoggingConfig
    runtime: RuntimeConfig
    snapshot: SnapshotConfig
    detection_recording: DetectionRecordingConfig
    # Opens a live window (see debug_window.py) showing the raw camera feed
    # next to perception's latest annotated view. Also settable with
    # `--debug` on the command line (main.py), which forces this on
    # regardless of the config file.
    debug: bool


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load and validate the app config from `path` (default: `$ROBOT_CONTROLLER_CONFIG`,
    falling back to configs/robot_controller/app.yaml).

    Raises ConfigError -- never a raw KeyError/yaml.YAMLError/ValueError --
    for anything wrong with the file itself, so callers can log one clear
    message instead of a schema-shaped stack trace. This is the one place
    in the app where a bad setup is expected to stop startup outright: a
    game can't run on a config nobody validated, so failing fast here is
    what lets every later stage assume a well-formed AppConfig.
    """
    resolved_path = Path(path) if path is not None else Path(os.environ.get(CONFIG_PATH_ENV_VAR, DEFAULT_CONFIG_PATH))

    if not resolved_path.is_file():
        example_path = resolved_path.parent / "app.example.yaml"
        raise ConfigError(
            f"Config file not found: {resolved_path}. Copy {example_path} to "
            f"{resolved_path.name} and adjust it for this machine."
        )

    try:
        raw = yaml.safe_load(resolved_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse config file {resolved_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file {resolved_path} must contain a YAML mapping at the top level")

    try:
        return _build_config(raw, base_dir=resolved_path.parent)
    except KeyError as exc:
        raise ConfigError(f"Missing required config key {exc} in {resolved_path}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid config value in {resolved_path}: {exc}") from exc


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _build_config(raw: dict, base_dir: Path) -> AppConfig:
    game_raw = raw["game"]
    camera_raw = raw["camera"]
    perception_raw = raw["perception"]
    manipulation_raw = raw.get("manipulation", {})
    logging_raw = raw.get("logging", {})
    runtime_raw = raw.get("runtime", {})
    snapshot_raw = raw.get("snapshot", {})
    detection_recording_raw = raw.get("detection_recording", {})

    game = GameConfig(
        game=game_raw["type"],
        players=[Color(p) for p in game_raw["players"]],
        player_roles={Color(color): PlayerType(role) for color, role in game_raw["player_roles"].items()},
    )

    directory_raw = camera_raw.get("directory")
    camera = CameraConfig(
        backend=camera_raw.get("backend", "realsense"),
        width=camera_raw.get("width", 1280),
        height=camera_raw.get("height", 720),
        fps=camera_raw.get("fps", 30),
        serial_number=os.environ.get(CAMERA_SERIAL_ENV_VAR, camera_raw.get("serial_number")),
        frame_timeout_ms=camera_raw.get("frame_timeout_ms", 5000),
        max_consecutive_errors=camera_raw.get("max_consecutive_errors", 10),
        max_reconnect_attempts=camera_raw.get("max_reconnect_attempts", 5),
        reconnect_backoff_s=camera_raw.get("reconnect_backoff_s", 2.0),
        directory=_resolve_path(directory_raw) if directory_raw else None,
    )
    if camera.backend not in _VALID_CAMERA_BACKENDS:
        raise ValueError(f"camera.backend must be one of {sorted(_VALID_CAMERA_BACKENDS)}, got {camera.backend!r}")
    if camera.backend == "directory" and camera.directory is None:
        raise ValueError("camera.directory is required when camera.backend == 'directory'")

    visualize_dir_raw = perception_raw.get("visualize_dir")
    perception = PerceptionConfig(
        inference_config=_resolve_path(perception_raw["inference_config"]),
        visualize_dir=_resolve_path(visualize_dir_raw) if visualize_dir_raw else None,
    )

    manipulation = ManipulationConfig(
        require_confirmation=manipulation_raw.get("require_confirmation", True),
    )

    logging_cfg = LoggingConfig(
        level=os.environ.get(LOG_LEVEL_ENV_VAR, logging_raw.get("level", "INFO")),
        log_dir=_resolve_path(logging_raw.get("log_dir", "logs")),
        file_name=logging_raw.get("file_name", "robot_controller.log"),
        max_bytes=logging_raw.get("max_bytes", 5_000_000),
        backup_count=logging_raw.get("backup_count", 5),
        console=logging_raw.get("console", True),
    )

    runtime = RuntimeConfig(
        tick_interval_s=runtime_raw.get("tick_interval_s", 0.2),
        max_steps=runtime_raw.get("max_steps", 10_000),
        stuck_warning_attempts=runtime_raw.get("stuck_warning_attempts", 50),
    )

    snapshot = SnapshotConfig(
        enabled=snapshot_raw.get("enabled", False),
        key=snapshot_raw.get("key", "s"),
        output_dir=_resolve_path(snapshot_raw.get("output_dir", "captures")),
    )

    detection_recording = DetectionRecordingConfig(
        enabled=detection_recording_raw.get("enabled", False),
        start_key=detection_recording_raw.get("start_key", "r"),
        stop_key=detection_recording_raw.get("stop_key", "t"),
        interval_s=detection_recording_raw.get("interval_s", 1.0),
        output_dir=_resolve_path(detection_recording_raw.get("output_dir", "detections")),
    )
    if detection_recording.enabled and detection_recording.start_key == detection_recording.stop_key:
        raise ValueError(
            "detection_recording.start_key and stop_key must differ, "
            f"both are {detection_recording.start_key!r}"
        )
    if snapshot.enabled and detection_recording.enabled:
        # All three are registered on the same ConsoleKeyDispatcher (see
        # app.build_engine); a repeated key would silently overwrite an
        # earlier handler rather than doing both.
        keys = [snapshot.key, detection_recording.start_key, detection_recording.stop_key]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"snapshot.key/detection_recording.start_key/stop_key must all differ, got {keys}"
            )

    return AppConfig(
        game=game,
        camera=camera,
        perception=perception,
        manipulation=manipulation,
        logging=logging_cfg,
        runtime=runtime,
        snapshot=snapshot,
        detection_recording=detection_recording,
        debug=bool(raw.get("debug", False)),
    )
