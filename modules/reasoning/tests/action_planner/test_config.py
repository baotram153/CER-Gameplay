from reasoning.action_planner.config import DEFAULT_SCORING_CONFIG_PATH, ScoringConfig, load_scoring_config


def test_default_path_points_at_the_real_file():
    assert DEFAULT_SCORING_CONFIG_PATH.exists()
    assert DEFAULT_SCORING_CONFIG_PATH.name == "scoring.yaml"


def test_loads_default_scoring_config():
    config = load_scoring_config()
    assert isinstance(config, ScoringConfig)
    assert config.w_p == 1.0
    assert config.w_h == 1.0
    assert config.w_c == 1.0
    assert config.w_e == 1.0
    assert config.w_r == 1.0
    assert config.alpha_R == 1.0
    assert config.beta_R == 1.0
