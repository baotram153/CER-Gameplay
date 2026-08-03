from game_engine.home_stretch import home_stretch_target


def test_from_home_entry_target_is_home_entry_plus_die():
    for die in range(1, 7):
        assert home_stretch_target(56, die) == 56 + die


def test_from_home_stretch_cell_requires_strictly_higher_die():
    assert home_stretch_target(59, 3) is None  # cell 3, die 3: not strictly higher
    assert home_stretch_target(59, 2) is None  # lower die: no progress
    assert home_stretch_target(59, 4) == 60  # cell 3, die 4: legal, target cell 4
    assert home_stretch_target(59, 6) == 62
