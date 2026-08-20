from common.constants import Color
from common.type import Move, Piece, ValidationResult
from robot_controller.adapters.manipulation_adapter import ConsoleManipulationAdapter

_MOVE = Move(piece=Piece(color=Color.GREEN, pos=1), from_pos=1, to_pos=5)


def test_roll_dice_skips_confirmation_when_not_required():
    adapter = ConsoleManipulationAdapter(require_confirmation=False)
    adapter.roll_dice()  # would block on input() if it tried to confirm


def test_execute_move_defaults_to_success_without_confirmation():
    adapter = ConsoleManipulationAdapter(require_confirmation=False)
    assert adapter.execute_move(_MOVE) is True


def test_execute_move_reports_failure_when_operator_says_so(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "fail")
    adapter = ConsoleManipulationAdapter(require_confirmation=True)

    assert adapter.execute_move(_MOVE) is False


def test_execute_move_succeeds_on_plain_enter(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    adapter = ConsoleManipulationAdapter(require_confirmation=True)

    assert adapter.execute_move(_MOVE) is True


def test_confirmation_handles_no_interactive_input(monkeypatch):
    def _raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    adapter = ConsoleManipulationAdapter(require_confirmation=True)

    # Must not raise/crash even with no TTY available.
    assert adapter.execute_move(_MOVE) is True


def test_request_human_help_does_not_raise():
    adapter = ConsoleManipulationAdapter(require_confirmation=False)
    adapter.request_human_help(ValidationResult(is_valid=False, issues=["piece missing"]))
