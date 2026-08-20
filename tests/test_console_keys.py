from robot_controller.console_keys import ConsoleKeyDispatcher


class _ScriptedReader:
    def __init__(self, lines: list[str]):
        self._lines = list(lines)

    def poll(self) -> str | None:
        return self._lines.pop(0) if self._lines else None


def test_poll_invokes_the_handler_for_a_matching_key():
    calls = []
    dispatcher = ConsoleKeyDispatcher(reader=_ScriptedReader(["s"]))
    dispatcher.on("s", lambda: calls.append("s"))

    dispatcher.poll()

    assert calls == ["s"]


def test_poll_ignores_unrecognized_input():
    calls = []
    dispatcher = ConsoleKeyDispatcher(reader=_ScriptedReader(["xyz"]))
    dispatcher.on("s", lambda: calls.append("s"))

    dispatcher.poll()  # must not raise

    assert calls == []


def test_poll_drains_every_pending_line_in_one_call():
    calls = []
    dispatcher = ConsoleKeyDispatcher(reader=_ScriptedReader(["r", "junk", "t"]))
    dispatcher.on("r", lambda: calls.append("r"))
    dispatcher.on("t", lambda: calls.append("t"))

    dispatcher.poll()

    assert calls == ["r", "t"]


def test_two_features_registered_on_one_dispatcher_dont_steal_each_others_keys():
    snapshot_calls = []
    recording_calls = []
    dispatcher = ConsoleKeyDispatcher(reader=_ScriptedReader(["s", "r", "s"]))
    dispatcher.on("s", lambda: snapshot_calls.append(1))
    dispatcher.on("r", lambda: recording_calls.append(1))

    dispatcher.poll()

    assert len(snapshot_calls) == 2
    assert len(recording_calls) == 1
