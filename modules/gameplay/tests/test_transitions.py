from gameplay.phase import GamePhase
from gameplay.transitions import EDGES


def test_every_phase_except_end_game_has_an_outgoing_edge():
    sources = {edge[0] for edge in EDGES}
    for phase in GamePhase:
        if phase is GamePhase.END_GAME:
            continue
        assert phase in sources, f"{phase} has no documented outgoing edge"


def test_end_game_has_no_outgoing_edge():
    assert all(edge[0] is not GamePhase.END_GAME for edge in EDGES)


def test_recovery_has_exactly_one_outgoing_edge_and_no_self_loop():
    recovery_edges = [edge for edge in EDGES if edge[0] is GamePhase.RECOVERY]
    assert recovery_edges == [(GamePhase.RECOVERY, GamePhase.UPDATE_GAME_STATE)]
