from pathlib import Path

import pytest
import yaml
from common.constants import Color
from gameplay.player import PlayerType
from robot_controller.config import CAMERA_SERIAL_ENV_VAR, LOG_LEVEL_ENV_VAR, load_config
from robot_controller.errors import ConfigError

VALID_CONFIG = {
    "game": {
        "type": "ludo",
        "players": ["green", "yellow", "red", "blue"],
        "player_roles": {"green": "robot", "yellow": "human", "red": "human", "blue": "human"},
    },
    "camera": {"backend": "directory", "directory": "some/frames"},
    "perception": {"inference_config": "modules/perception/configs/ludo/inference.yaml"},
}


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    data = {**VALID_CONFIG, **(overrides or {})}
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_config_applies_defaults(tmp_path):
    config = load_config(_write_config(tmp_path))

    assert config.game.players == [Color.GREEN, Color.YELLOW, Color.RED, Color.BLUE]
    assert config.game.player_roles[Color.GREEN] is PlayerType.ROBOT
    assert config.camera.backend == "directory"
    assert config.camera.width == 1280  # default
    assert config.manipulation.require_confirmation is True  # default
    assert config.runtime.max_steps == 10_000  # default


def test_load_config_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_malformed_yaml_raises_config_error(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text("game: [this is not: a valid mapping")

    with pytest.raises(ConfigError, match="Could not parse"):
        load_config(path)


def test_load_config_missing_required_key_raises_config_error(tmp_path):
    data = {k: v for k, v in VALID_CONFIG.items() if k != "perception"}
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ConfigError, match="perception"):
        load_config(path)


def test_load_config_invalid_role_raises_config_error(tmp_path):
    overrides = {
        "game": {
            **VALID_CONFIG["game"],
            "player_roles": {"green": "not-a-role", "yellow": "human", "red": "human", "blue": "human"},
        }
    }
    with pytest.raises(ConfigError, match="not-a-role"):
        load_config(_write_config(tmp_path, overrides))


def test_directory_backend_requires_directory(tmp_path):
    overrides = {"camera": {"backend": "directory"}}
    with pytest.raises(ConfigError, match="camera.directory"):
        load_config(_write_config(tmp_path, overrides))


def test_unknown_backend_rejected(tmp_path):
    overrides = {"camera": {"backend": "webcam"}}
    with pytest.raises(ConfigError, match="camera.backend"):
        load_config(_write_config(tmp_path, overrides))


def test_camera_serial_env_var_overrides_config(tmp_path, monkeypatch):
    overrides = {"camera": {"backend": "realsense", "serial_number": "from-file"}}
    monkeypatch.setenv(CAMERA_SERIAL_ENV_VAR, "from-env")

    config = load_config(_write_config(tmp_path, overrides))

    assert config.camera.serial_number == "from-env"


def test_log_level_env_var_overrides_config(tmp_path, monkeypatch):
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "DEBUG")

    config = load_config(_write_config(tmp_path))

    assert config.logging.level == "DEBUG"
