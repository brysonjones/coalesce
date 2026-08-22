"""Provider-neutral description of a job.

``launch_job`` normalizes its keyword arguments into a :class:`JobSpec` and
hands that to a provider. Anything a provider cannot honour is either
translated here or reported as a clear error, so the two backends never
silently disagree about what was requested.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical GPU names. Providers translate these into their own vocabulary so
# that `gpu="H100"` means the same thing everywhere.
_VERTEX_GPUS: dict[str, str | None] = {
    "K80": "NVIDIA_TESLA_K80",
    "P4": "NVIDIA_TESLA_P4",
    "P100": "NVIDIA_TESLA_P100",
    "V100": "NVIDIA_TESLA_V100",
    "T4": "NVIDIA_TESLA_T4",
    "L4": "NVIDIA_L4",
    "L40S": None,
    "A10G": None,
    "A100_40GB": "NVIDIA_TESLA_A100",
    "A100": "NVIDIA_A100_80GB",
    "H100_40GB": None,
    "H100": "NVIDIA_H100_80GB",
    "H100_MEGA": "NVIDIA_H100_MEGA_80GB",
    "H200": "NVIDIA_H200_141GB",
    "B200": "NVIDIA_B200",
    "B300": None,
    "GB200": "NVIDIA_GB200",
    "GB300": None,
    "RTX_PRO_6000": "NVIDIA_RTX_PRO_6000",
}

_BASETEN_GPUS: dict[str, str | None] = {
    "K80": None,
    "P4": None,
    "P100": None,
    "V100": "V100",
    "T4": "T4",
    "L4": "L4",
    "L40S": "L40S",
    "A10G": "A10G",
    "A100_40GB": "A100_40GB",
    "A100": "A100",
    "H100_40GB": "H100_40GB",
    "H100": "H100",
    "H100_MEGA": None,
    "H200": "H200",
    "B200": "B200",
    "B300": "B300",
    "GB200": None,
    "GB300": "GB300",
    "RTX_PRO_6000": "RTX_PRO_6000",
}

# Spellings we accept and fold into the canonical name above. Raw Vertex
# accelerator names are included so existing call sites keep working when they
# switch provider.
_GPU_ALIASES: dict[str, str] = {
    "A100_80GB": "A100",
    "H100_80GB": "H100",
    "H100_MEGA_80GB": "H100_MEGA",
    "H200_141GB": "H200",
    "TESLA_K80": "K80",
    "TESLA_P4": "P4",
    "TESLA_P100": "P100",
    "TESLA_V100": "V100",
    "TESLA_T4": "T4",
    "TESLA_A100": "A100_40GB",
}

CANONICAL_GPUS = tuple(_VERTEX_GPUS)


def canonical_gpu(name: str) -> str:
    """Fold a user-supplied GPU name into coalesce's canonical vocabulary."""
    key = name.strip().upper().replace("-", "_").replace(" ", "_")
    if key.startswith("NVIDIA_"):
        key = key[len("NVIDIA_") :]
    key = _GPU_ALIASES.get(key, key)
    if key not in _VERTEX_GPUS:
        raise ValueError(
            f"Unknown GPU {name!r}. Known GPUs: {', '.join(CANONICAL_GPUS)}."
        )
    return key


def _translate(gpu: str, table: dict[str, str | None], provider: str) -> str:
    translated = table[canonical_gpu(gpu)]
    if translated is None:
        available = ", ".join(name for name, value in table.items() if value)
        raise ValueError(
            f"GPU {gpu!r} is not available on {provider}. Available: {available}."
        )
    return translated


def vertex_gpu(gpu: str) -> str:
    """Canonical GPU name -> Vertex AI accelerator type."""
    return _translate(gpu, _VERTEX_GPUS, "Vertex AI")


def baseten_gpu(gpu: str) -> str:
    """Canonical GPU name -> Baseten accelerator name."""
    return _translate(gpu, _BASETEN_GPUS, "Baseten")


@dataclass
class ComputeSpec:
    """Hardware request, expressed in terms both providers can satisfy.

    ``machine_type``, ``boot_disk_type`` and ``boot_disk_size_gb`` are Vertex
    concepts; ``cpu_count`` and ``memory`` are Baseten concepts. Each provider
    uses the fields that apply to it and warns about the ones that do not.
    """

    gpu: str | None = None
    gpu_count: int = 1
    vertex_accelerator_type: str | None = None
    """Raw Vertex accelerator name, forwarded verbatim when `gpu` was not used."""
    machine_type: str = "n1-standard-4"
    cpu_count: int = 4
    memory: str = "16Gi"
    node_count: int = 1
    boot_disk_type: str = "pd-ssd"
    boot_disk_size_gb: int = 100
    # True when the caller left boot disk settings at their defaults, which lets
    # non-Vertex providers stay quiet instead of warning about an unused knob.
    boot_disk_is_default: bool = True


@dataclass
class SchedulingSpec:
    """How the job should be queued.

    ``strategy`` is one of STANDARD, SPOT or FLEX_START. Vertex maps these onto
    its own scheduling strategies; Baseten maps STANDARD/FLEX_START onto
    dedicated capacity (it queues for capacity natively) and SPOT onto
    interruptible capacity.
    """

    strategy: str = "STANDARD"
    max_wait_duration: int = 86400
    priority: int | None = None


@dataclass
class JobSpec:
    func: Callable
    gcp_project_id: str
    bucket: str
    job_name: str
    container_uri: str | None = None
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    scheduling: SchedulingSpec = field(default_factory=SchedulingSpec)
    region: str = "us-central1"
    sync_packages: list[str] = field(default_factory=list)
    config: str | Path | dict[str, Any] | None = None
    extra_packages: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    staging_prefix: str = ".coalesce/tmp"
    sync: bool = True
    dry_run: bool = False
    stream_logs: bool = False
    log_polling_interval: int = 10
    allow_multiline_logs: bool = True
    # Baseten-only knobs.
    baseten_project: str | None = None
    gcp_credentials_secret: str = "gcp_service_account_json"
    container_registry_secret: str | None = None
    mount_datasets: list[str] = field(default_factory=list)
    checkpoint_volume_gb: int | None = None

    def module_name(self) -> str:
        """The importable module the remote runner will load ``func`` from."""
        module_name = self.func.__module__
        if module_name == "__main__":
            raise ValueError(
                f"Function '{self.func.__name__}' has module '__main__' which cannot be "
                f"imported remotely. The function must be imported from an actual module, "
                f"not defined in the script being run. Move the function to a separate "
                f"module and import it."
            )
        return module_name
