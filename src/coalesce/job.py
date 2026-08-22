"""Provider-agnostic handle to a launched job."""

from __future__ import annotations

import time
from typing import Any

from .providers.base import JobBackend

# Log queries are bounded by wall-clock time, which the remote side does not
# necessarily agree with. Widen the window and rely on de-duplication.
_CLOCK_SKEW_BUFFER_MS = 60_000


class Job:
    """What ``launch_job`` returns, whichever provider ran the work.

    The provider's own object is still available as :attr:`raw` -- an
    ``aiplatform.CustomJob`` for Vertex, the Baseten API response dict for
    Baseten -- for anything this wrapper does not cover.
    """

    def __init__(
        self,
        *,
        id: str,
        name: str,
        provider: str,
        backend: JobBackend,
        raw: Any = None,
    ) -> None:
        self.id = id
        self.name = name
        self.provider = provider
        self.raw = raw
        self._backend = backend

    def __repr__(self) -> str:
        return f"Job(provider={self.provider!r}, id={self.id!r}, name={self.name!r})"

    @property
    def console_url(self) -> str | None:
        return self._backend.console_url

    def status(self) -> str:
        return self._backend.status()

    def stop(self) -> None:
        self._backend.stop()

    def wait(self, poll_interval: int = 10, timeout: int | None = None) -> str:
        """Block until the job reaches a terminal state and return that state."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            state = self._backend.status()
            if self._backend.is_terminal(state):
                return state
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job {self.id} still in state {state} after {timeout}s."
                )
            time.sleep(poll_interval)

    def stream_logs(self, poll_interval: int = 10, timeout: int | None = None) -> str:
        """Print remote output until the job finishes; returns the final state.

        Each provider is polled the same way: pull the logs published since the
        last poll, print whatever has not been seen, then check whether the job
        has finished. A terminal state triggers one final pull so trailing
        output -- usually the traceback that explains the failure -- is not lost.
        """
        seen: set[str] = set()
        last_poll_ms: int | None = None
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            now_ms = int(time.time() * 1000)
            start_ms = None if last_poll_ms is None else last_poll_ms - _CLOCK_SKEW_BUFFER_MS
            self._print_new(seen, start_ms, now_ms + _CLOCK_SKEW_BUFFER_MS)
            last_poll_ms = now_ms

            state = self._backend.status()
            if self._backend.is_terminal(state):
                self._print_new(
                    seen,
                    last_poll_ms - _CLOCK_SKEW_BUFFER_MS,
                    int(time.time() * 1000) + _CLOCK_SKEW_BUFFER_MS,
                )
                print(f"Job {self.id} finished with state: {state}")
                return state

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job {self.id} still in state {state} after {timeout}s."
                )
            time.sleep(poll_interval)

    def _print_new(
        self, seen: set[str], start_epoch_ms: int | None, end_epoch_ms: int | None
    ) -> None:
        for entry in self._backend.fetch_logs(start_epoch_ms, end_epoch_ms):
            if entry.key in seen:
                continue
            seen.add(entry.key)
            message = entry.message
            if not message:
                continue
            prefix = f"({entry.replica}) " if entry.replica else ""
            for line in message.rstrip().splitlines():
                print(f"{prefix}{line}")
