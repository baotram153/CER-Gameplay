"""app.py's own composition logic -- not a full engine run (that needs the
real YOLO checkpoint + camera hardware), just the pieces that are pure
plumbing: camera backend selection, and resolving inference.yaml's
relative paths independent of the process's current working directory
(see app.py's _load_ludo_pipeline docstring for why this matters)."""
import os

from robot_controller import app
from robot_controller.camera.directory_camera import DirectoryFrameSource
from robot_controller.camera.realsense_camera import RealSenseCamera
from robot_controller.config import load_config


def test_build_camera_selects_directory_backend(tmp_path):
    config = _config_with_camera(tmp_path, {"backend": "directory", "directory": str(tmp_path)})
    assert isinstance(app.build_camera(config.camera), DirectoryFrameSource)


def test_build_camera_selects_realsense_backend(tmp_path):
    config = _config_with_camera(tmp_path, {"backend": "realsense"})
    assert isinstance(app.build_camera(config.camera), RealSenseCamera)


def test_resolve_relative_to_perception_leaves_none_and_absolute_paths_alone(tmp_path):
    assert app._resolve_relative_to_perception(None) is None

    absolute = str(tmp_path / "weights.pt")
    assert app._resolve_relative_to_perception(absolute) == absolute


def test_resolve_relative_to_perception_resolves_against_perception_module_root():
    resolved = app._resolve_relative_to_perception("models/best.pt")
    assert resolved == str(app._PERCEPTION_ROOT / "models" / "best.pt")


def test_resolve_relative_to_perception_is_independent_of_cwd(tmp_path):
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        resolved = app._resolve_relative_to_perception("models/best.pt")
    finally:
        os.chdir(original_cwd)
    assert resolved == str(app._PERCEPTION_ROOT / "models" / "best.pt")


def test_build_key_dispatcher_returns_nothing_when_both_features_are_disabled(tmp_path):
    config = _config_with_camera(tmp_path, {"backend": "directory", "directory": str(tmp_path)})
    assert app._build_key_dispatcher(config) == (None, None, None)


def test_build_key_dispatcher_wires_the_snapshot_key(tmp_path):
    config = _config_with_camera(
        tmp_path,
        {"backend": "directory", "directory": str(tmp_path)},
        extra={"snapshot": {"enabled": True, "key": "s", "output_dir": str(tmp_path / "captures")}},
    )

    dispatcher, snapshot_saver, detection_recorder = app._build_key_dispatcher(config)

    assert snapshot_saver is not None
    assert detection_recorder is None
    dispatcher._handlers["s"]()  # simulate the key being pressed
    assert snapshot_saver._armed is True


def test_build_key_dispatcher_wires_both_recording_keys(tmp_path):
    config = _config_with_camera(
        tmp_path,
        {"backend": "directory", "directory": str(tmp_path)},
        extra={
            "detection_recording": {
                "enabled": True, "start_key": "r", "stop_key": "t", "output_dir": str(tmp_path / "detections"),
            }
        },
    )

    dispatcher, snapshot_saver, detection_recorder = app._build_key_dispatcher(config)

    assert snapshot_saver is None
    assert detection_recorder is not None
    dispatcher._handlers["r"]()
    assert detection_recorder.active is True
    dispatcher._handlers["t"]()
    assert detection_recorder.active is False


def _config_with_camera(tmp_path, camera_overrides: dict, extra: dict | None = None):
    import yaml

    data = {
        "game": {
            "type": "ludo",
            "players": ["green", "yellow"],
            "player_roles": {"green": "robot", "yellow": "human"},
        },
        "camera": camera_overrides,
        "perception": {"inference_config": "modules/perception/configs/ludo/inference.yaml"},
        **(extra or {}),
    }
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path)
