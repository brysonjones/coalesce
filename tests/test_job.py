from __future__ import annotations

from typing import ClassVar

import pytest

from coalesce.job import Job
from coalesce.providers.base import JobBackend, LogEntry


class FakeBackend(JobBackend):
    """A backend whose state advances one step per status() call."""

    TERMINAL_STATES: ClassVar[frozenset[str]] = frozenset({"DONE"})

    def __init__(self, states, logs_per_poll):
        self.states = list(states)
        self.logs_per_poll = list(logs_per_poll)
        self.stopped = False
        self.windows = []

    def status(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def fetch_logs(self, start_epoch_ms, end_epoch_ms):
        self.windows.append((start_epoch_ms, end_epoch_ms))
        if not self.logs_per_poll:
            return []
        return self.logs_per_poll.pop(0)

    def stop(self):
        self.stopped = True


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("coalesce.job.time.sleep", lambda _: None)


def make_job(backend):
    return Job(id="j1", name="demo", provider="fake", backend=backend)


def test_stream_logs_prints_each_line_once_and_returns_final_state(capsys) -> None:
    backend = FakeBackend(
        states=["RUNNING", "DONE"],
        logs_per_poll=[
            [LogEntry(key="a", message="line one")],
            [LogEntry(key="a", message="line one"), LogEntry(key="b", message="line two")],
        ],
    )

    state = make_job(backend).stream_logs(poll_interval=0)

    out = capsys.readouterr().out
    assert state == "DONE"
    assert out.count("line one") == 1
    assert out.count("line two") == 1


def test_stream_logs_drains_once_more_after_the_job_finishes() -> None:
    backend = FakeBackend(states=["DONE"], logs_per_poll=[[], []])

    make_job(backend).stream_logs(poll_interval=0)

    # One poll before the terminal check, one after, so a traceback emitted at
    # the very end still reaches the user.
    assert len(backend.windows) == 2


def test_the_first_poll_has_no_lower_bound() -> None:
    backend = FakeBackend(states=["DONE"], logs_per_poll=[[]])

    make_job(backend).stream_logs(poll_interval=0)

    assert backend.windows[0][0] is None


def test_replica_names_prefix_multi_node_output(capsys) -> None:
    backend = FakeBackend(
        states=["DONE"],
        logs_per_poll=[[LogEntry(key="a", message="hello", replica="worker-1")]],
    )

    make_job(backend).stream_logs(poll_interval=0)

    assert "(worker-1) hello" in capsys.readouterr().out


def test_wait_returns_the_terminal_state() -> None:
    backend = FakeBackend(states=["RUNNING", "RUNNING", "DONE"], logs_per_poll=[])

    assert make_job(backend).wait(poll_interval=0) == "DONE"


def test_wait_times_out_rather_than_polling_forever() -> None:
    backend = FakeBackend(states=["RUNNING"], logs_per_poll=[])

    with pytest.raises(TimeoutError, match="RUNNING"):
        make_job(backend).wait(poll_interval=0, timeout=0)


def test_stop_delegates_to_the_backend() -> None:
    backend = FakeBackend(states=["RUNNING"], logs_per_poll=[])

    make_job(backend).stop()

    assert backend.stopped is True


def test_backends_without_stop_say_so() -> None:
    class NoStop(JobBackend):
        TERMINAL_STATES: ClassVar[frozenset[str]] = frozenset()

        def status(self):
            return "RUNNING"

        def fetch_logs(self, start_epoch_ms, end_epoch_ms):
            return []

    with pytest.raises(NotImplementedError, match="does not support stopping"):
        make_job(NoStop()).stop()
