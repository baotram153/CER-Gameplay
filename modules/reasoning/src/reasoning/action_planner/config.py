"""Loading the scoring heuristic's weights/parameters from `scoring.yaml`.

`common.rules` uses module-constants-loaded-at-import because those rule
invariants are shared globally and never reloaded. Scoring weights are
tunable hyperparameters instead — callers may reasonably want to load a
non-default file (e.g. during tuning experiments) — so this loads into an
explicit, reusable `ScoringConfig` value via a plain function instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_SCORING_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "scoring.yaml"


@dataclass(frozen=True)
class ScoringConfig:
    """One field per symbol in the Score(a) = w_p*P + w_h*H + w_c*C + w_e*E - w_r*R formula."""

    w_p: float
    w_h: float
    w_c: float
    w_e: float
    w_r: float

    alpha_P: float
    beta_P: float

    alpha_H: float

    alpha_C: float
    beta_C: float

    alpha_E: float
    beta_E: float

    alpha_R: float
    beta_R: float


def load_scoring_config(path: str | Path = DEFAULT_SCORING_CONFIG_PATH) -> ScoringConfig:
    """Load a `ScoringConfig` from a YAML file shaped like `scoring.yaml`.

    Fields are constructed by explicit keyword, not `**dict` splatting, so a
    missing or misspelled key raises `KeyError` immediately here rather than
    surfacing later as a wrong score from a silently-defaulted term.
    """
    data = yaml.safe_load(Path(path).read_text())
    weights = data["weights"]
    progress = data["progress"]
    home_stretch = data["home_stretch"]
    capture = data["capture"]
    entry = data["entry"]
    risk = data["risk"]
    return ScoringConfig(
        w_p=weights["w_p"],
        w_h=weights["w_h"],
        w_c=weights["w_c"],
        w_e=weights["w_e"],
        w_r=weights["w_r"],
        alpha_P=progress["alpha"],
        beta_P=progress["beta"],
        alpha_H=home_stretch["alpha"],
        alpha_C=capture["alpha"],
        beta_C=capture["beta"],
        alpha_E=entry["alpha"],
        beta_E=entry["beta"],
        alpha_R=risk["alpha"],
        beta_R=risk["beta"],
    )
