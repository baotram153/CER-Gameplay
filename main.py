"""CLI entrypoint: load config, set up logging, run one game."""
from __future__ import annotations

import argparse
import dataclasses
import sys

from robot_controller.config import load_config
from robot_controller.errors import ConfigError
from robot_controller.logging_setup import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the app config YAML (default: $ROBOT_CONTROLLER_CONFIG, "
        "or configs/robot_controller/app.yaml).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Force debug mode on for this run (see debug: in the app config) -- "
        "opens a live window showing the camera feed and perception's detections.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        # Logging isn't configured yet without a loaded config, so this one
        # startup failure goes straight to stderr instead.
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    if args.debug:
        config = dataclasses.replace(config, debug=True)

    configure_logging(config.logging)

    from robot_controller.app import run  # deferred: imports perception's heavy CV/ML deps

    result = run(config)
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
