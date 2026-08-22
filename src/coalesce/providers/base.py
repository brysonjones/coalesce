"""The contract every coalesce provider implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class LogEntry:
    """One line of remote output, normalized across providers."""

    key: str
    """Stable identity used to avoid printing the same line twice."""
    message: str
    timestamp_ns: int | None = None
    replica: str | None = None


class JobBackend(ABC):
    """Per-provider handle to a submitted job.

    A backend only has to answer three questions -- what state is the job in,
    what has it logged, and can you stop it -- and :class:`~coalesce.job.Job`
    builds the shared polling and log-streaming behaviour on top.
    """

    TERMINAL_STATES: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def status(self) -> str:
        """Current provider-reported job state."""

    @abstractmethod
    def fetch_logs(
        self, start_epoch_ms: int | None, end_epoch_ms: int | None
    ) -> list[LogEntry]:
        """Logs in the given window, oldest first. Bounds may be ignored."""

    def stop(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support stopping a job."
        )

    @property
    def console_url(self) -> str | None:
        """Where a human can watch this job, if the provider has such a page."""
        return None

    def is_terminal(self, state: str) -> bool:
        return state in self.TERMINAL_STATES
