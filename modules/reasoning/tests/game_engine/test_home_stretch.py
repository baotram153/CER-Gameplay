from reasoning.game_engine.home_stretch import home_stretch_target


def test_from_home_entry_target_is_home_entry_plus_die():
    for die in range(1, 7):
        assert home_stretch_target(60, die) == 60 + die


def test_from_home_stretch_cell_requires_strictly_higher_die():
    assert home_stretch_target(63, 3) is None  # cell 3, die 3: not strictly higher
    assert home_stretch_target(63, 2) is None  # lower die: no progress
    assert home_stretch_target(63, 4) == 64  # cell 3, die 4: legal, target cell 4
    assert home_stretch_target(63, 6) == 66
