"""Launch Python functions on cloud compute.

``launch_job`` is provider-agnostic: it normalizes its arguments into a
:class:`~coalesce.spec.JobSpec` and hands that to the requested backend. GCP
Vertex AI and Baseten both stage through the same GCS bucket and run the same
remote entry point, so switching between them is a one-word change.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from . import providers
from .job import Job
from .spec import ComputeSpec, JobSpec, SchedulingSpec, canonical_gpu

DEFAULT_PROVIDER = "vertex"


def _resolve_gpu(
    gpu: str | None, accelerator_type: str | None
) -> tuple[str | None, str | None]:
    """Work out the canonical GPU name and any raw Vertex passthrough.

    ``gpu`` is the portable spelling. ``accelerator_type`` is the legacy Vertex
    one and is forwarded verbatim so that names coalesce does not know about
    (TPUs, for instance) keep working exactly as they did.
    """
    if gpu is not None:
        return canonical_gpu(gpu), None
    if accelerator_type is None:
        return None, None
    try:
        return canonical_gpu(accelerator_type), accelerator_type
    except ValueError:
        return None, accelerator_type


def _resolve_strategy(scheduling_strategy: str) -> str:
    strategy = scheduling_strategy.upper()
    if strategy == "DWS":
        return "FLEX_START"
    if strategy not in ("STANDARD", "SPOT", "FLEX_START"):
        raise ValueError(
            f"Unknown scheduling_strategy {scheduling_strategy!r}. "
            "Choose STANDARD, SPOT or FLEX_START."
        )
    return strategy


def launch_job(
    func: Callable,
    project_id: str,
    bucket: str,
    provider: str | None = None,
    region: str = "us-central1",
    container_uri: str | None = None,
    machine_type: str = "n1-standard-4",
    accelerator_type: str | None = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    gpu: str | None = None,
    gpu_count: int | None = None,
    cpu_count: int = 4,
    memory: str = "16Gi",
    node_count: int = 1,
    boot_disk_type: str = "pd-ssd",
    boot_disk_size_gb: int = 100,
    sync_packages: list[str] | None = None,
    job_name: str | None = None,
    sync: bool = True,
    config: str | Path | dict[str, Any] | None = None,
    extra_packages: list[str] | None = None,
    env: dict[str, str] | None = None,
    scheduling_strategy: str = "STANDARD",
    max_wait_duration: int = 86400,
    priority: int | None = None,
    stream_logs: bool = False,
    log_polling_interval: int = 10,
    allow_multiline_logs: bool = True,
    staging_prefix: str = ".coalesce/tmp",
    baseten_project: str | None = None,
    gcp_credentials_secret: str = "gcp_service_account_json",
    mount_datasets: list[str] | None = None,
    checkpoint_volume_gb: int | None = None,
) -> Job:
    """
    Launch a Python function on cloud compute.

    Args:
        func: The Python function to run remotely. Must be importable.
        project_id: GCP project ID (e.g., "my-project"). Used for GCS staging on
                    every provider, and as the compute project on Vertex AI.
        bucket: GCS bucket for staging (e.g., "gs://my-bucket" or "my-bucket")
        provider: Where to run: "vertex" (aliases: "gcp") or "baseten".
                  Defaults to $COALESCE_PROVIDER, else "vertex".
        region: GCP region (default: "us-central1"). Vertex AI only; Baseten
                schedules across its own capacity and ignores this.
        container_uri: Docker image URI with required dependencies. Defaults to
                       a Vertex AI prebuilt PyTorch image on Vertex, and to
                       pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime on Baseten.
        machine_type: Compute Engine machine type (default: "n1-standard-4").
                      Vertex AI only; use cpu_count/memory on Baseten.
        accelerator_type: Vertex GPU name (e.g., "NVIDIA_TESLA_T4"). Forwarded
                          verbatim to Vertex AI. Prefer `gpu` for portability.
                          Set to None for CPU-only jobs.
        accelerator_count: Number of GPUs (default: 1)
        gpu: Portable GPU name ("T4", "L4", "A100", "H100", "H200", ...) that
             each provider translates into its own vocabulary. Overrides
             accelerator_type when set.
        gpu_count: Number of GPUs; defaults to accelerator_count.
        cpu_count: vCPUs per node. Baseten only (Vertex derives this from
                   machine_type).
        memory: RAM per node, e.g. "64Gi". Baseten only.
        node_count: Nodes to run on. Baseten only.
        boot_disk_type: Vertex AI boot disk type (default: "pd-ssd")
        boot_disk_size_gb: Vertex AI boot disk size in GiB (default: 100).
                           Vertex AI only; Baseten reports that it ignored these
                           if you set them.
        sync_packages: List of local Python package names to sync to the job.
                      These packages will be zipped and uploaded to GCS, then
                      extracted on the remote machine before running the function.
        job_name: Custom job name (auto-generated if None)
        sync: If True, wait for job completion. If False, return immediately.
        config: Configuration to pass to the function. Can be:
                - Path to a YAML/JSON file (will be uploaded to GCS)
                - Dict (will be serialized to JSON and passed via env var)
                The function receives this as its first argument.
        extra_packages: List of pip packages to install before running the job.
                       Example: ["transformers", "accelerate>=0.20"]
        env: Additional environment variables to set on the remote job.
             Example: {"WANDB_API_KEY": "xxx", "HF_TOKEN": "yyy"}
        scheduling_strategy: Scheduling strategy for the job. Options:
                            - "STANDARD": On-demand resources (default)
                            - "SPOT": Preemptible instances (cheaper, may be interrupted)
                            - "FLEX_START" or "DWS": Queues until resources are available.
                              Required for a3-highgpu-1g/2g/4g machine types on Vertex;
                              on Baseten, queueing for capacity is the default behaviour.
        max_wait_duration: Max wait time in seconds for FLEX_START scheduling
                           (default: 86400 = 24h). Vertex AI only.
        priority: Queue priority for the job. Baseten only.
        stream_logs: If True, submit the job asynchronously and stream remote logs locally.
        log_polling_interval: Polling interval in seconds for log streaming.
        allow_multiline_logs: Retained for compatibility; log lines are always
                              printed intact.
        staging_prefix: Prefix under the staging bucket for temp artifacts.
        baseten_project: Baseten training project to group this job under.
                         Defaults to the function name.
        gcp_credentials_secret: Name of the Baseten secret holding a GCP service
                                account JSON key. This is what lets a Baseten
                                job read and write your GCS bucket.
        mount_datasets: Read-only GCS paths to mount into the container, given as
                        "gs://bucket/path:/container/path". Baseten only.
        checkpoint_volume_gb: Size of Baseten's persistent checkpoint volume, in
                              GiB. Baseten only; the mount is exposed to the job
                              as $BT_CHECKPOINT_DIR.

    Returns:
        A :class:`~coalesce.job.Job` handle. The provider's native object is
        available as ``job.raw``.

    Example:
        def my_training_function(config: dict):
            import torch
            print(f"Learning rate: {config['learning_rate']}")
            # ... training code ...

        launch_job(
            func=my_training_function,
            project_id="my-project",
            bucket="gs://my-bucket",
            provider="baseten",
            gpu="H100",
            config="config.yaml",  # or {"learning_rate": 0.001}
            sync_packages=["my_local_package"],
            extra_packages=["transformers", "accelerate"],
        )
    """
    provider_name = providers.normalize(
        provider or os.environ.get("COALESCE_PROVIDER") or DEFAULT_PROVIDER
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if job_name is None:
        job_name = f"{func.__name__}_{timestamp}"

    resolved_gpu, raw_accelerator = _resolve_gpu(gpu, accelerator_type)

    compute = ComputeSpec(
        gpu=resolved_gpu,
        gpu_count=gpu_count if gpu_count is not None else accelerator_count,
        vertex_accelerator_type=raw_accelerator,
        machine_type=machine_type,
        cpu_count=cpu_count,
        memory=memory,
        node_count=node_count,
        boot_disk_type=boot_disk_type,
        boot_disk_size_gb=boot_disk_size_gb,
        boot_disk_is_default=(boot_disk_type == "pd-ssd" and boot_disk_size_gb == 100),
    )

    job_spec = JobSpec(
        func=func,
        gcp_project_id=project_id,
        bucket=bucket,
        job_name=job_name,
        container_uri=container_uri,
        compute=compute,
        scheduling=SchedulingSpec(
            strategy=_resolve_strategy(scheduling_strategy),
            max_wait_duration=max_wait_duration,
            priority=priority,
        ),
        region=region,
        sync_packages=list(sync_packages or []),
        config=config,
        extra_packages=list(extra_packages or []),
        env=dict(env or {}),
        staging_prefix=staging_prefix,
        sync=sync,
        stream_logs=stream_logs,
        log_polling_interval=log_polling_interval,
        allow_multiline_logs=allow_multiline_logs,
        baseten_project=baseten_project,
        gcp_credentials_secret=gcp_credentials_secret,
        mount_datasets=list(mount_datasets or []),
        checkpoint_volume_gb=checkpoint_volume_gb,
    )

    return providers.get_launcher(provider_name)(job_spec)
