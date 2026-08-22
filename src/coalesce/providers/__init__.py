"""Compute backends coalesce can launch onto."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..job import Job
    from ..spec import JobSpec

PROVIDERS = ("vertex", "baseten")

# Accepted spellings for each provider, so "gcp" and "vertex" mean the same thing.
_ALIASES = {
    "vertex": "vertex",
    "vertex_ai": "vertex",
    "gcp": "vertex",
    "google": "vertex",
    "baseten": "baseten",
}


def normalize(provider: str) -> str:
    key = provider.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of: {', '.join(PROVIDERS)}."
        )
    return _ALIASES[key]


def get_launcher(provider: str) -> Callable[["JobSpec"], "Job"]:
    """Import and return the launch function for ``provider``.

    Imports are deferred so that using one provider never requires the other's
    dependencies to be installed.
    """
    name = normalize(provider)
    if name == "vertex":
        from . import vertex

        return vertex.launch
    from . import baseten

    return baseten.launch
