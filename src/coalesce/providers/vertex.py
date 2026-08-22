"""Run coalesce jobs on GCP Vertex AI custom jobs."""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, ClassVar

from google.cloud import aiplatform

from .. import spec as spec_module
from .. import staging
from ..job import Job
from ..spec import JobSpec
from .base import JobBackend, LogEntry

DEFAULT_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest"


def _custom_job_id(resource_name: str) -> str:
    """Extract the final custom job id from a Vertex AI resource name."""
    return resource_name.rsplit("/", 1)[-1]


def _wait_for_resource_name(job: aiplatform.CustomJob, timeout_seconds: int = 300) -> str:
    """Wait for async CustomJob creation to populate resource_name."""
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            return job.resource_name
        except RuntimeError as exc:
            last_error = exc
            time.sleep(1)

    raise RuntimeError(
        "Timed out waiting for Vertex AI to create the CustomJob resource. "
        "The job may still have been submitted; check the Vertex AI console."
    ) from last_error


def _log_entry_message(entry: dict[str, Any]) -> str:
    if "textPayload" in entry:
        return str(entry["textPayload"])
    if "jsonPayload" in entry:
        payload = entry["jsonPayload"]
        if isinstance(payload, dict):
            message = payload.get("message")
            if message is not None:
                return str(message)
        return str(payload)
    if "protoPayload" in entry:
        return str(entry["protoPayload"])
    return ""


class VertexJobBackend(JobBackend):
    """Reads Vertex AI job state and Cloud Logging output through ``gcloud``."""

    TERMINAL_STATES: ClassVar[frozenset[str]] = frozenset(
        {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }
    )

    def __init__(self, *, custom_job_id: str, project_id: str, region: str) -> None:
        self.custom_job_id = custom_job_id
        self.project_id = project_id
        self.region = region

    @property
    def console_url(self) -> str | None:
        return (
            "https://console.cloud.google.com/vertex-ai/locations/"
            f"{self.region}/training/{self.custom_job_id}?project={self.project_id}"
        )

    def _gcloud(self, command: list[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not reach Vertex AI because `gcloud` was not found. "
                "Install the Google Cloud CLI or launch without stream_logs=True."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            detail = f": {stderr}" if stderr else ""
            raise RuntimeError(
                f"gcloud failed with exit code {exc.returncode}{detail}."
            ) from exc

    def status(self) -> str:
        result = self._gcloud(
            [
                "gcloud",
                "ai",
                "custom-jobs",
                "describe",
                self.custom_job_id,
                f"--project={self.project_id}",
                f"--region={self.region}",
                "--format=value(state)",
            ]
        )
        return result.stdout.strip()

    def fetch_logs(
        self, start_epoch_ms: int | None, end_epoch_ms: int | None
    ) -> list[LogEntry]:
        # Cloud Logging is queried by job id rather than by time window; the
        # window is handled by de-duplicating on insertId instead.
        filter_expr = (
            f'resource.type="ml_job" AND resource.labels.job_id="{self.custom_job_id}"'
        )
        result = self._gcloud(
            [
                "gcloud",
                "logging",
                "read",
                filter_expr,
                f"--project={self.project_id}",
                "--format=json",
                "--limit=200",
            ]
        )
        entries = json.loads(result.stdout or "[]")
        logs = []
        for entry in reversed(entries):
            message = _log_entry_message(entry)
            insert_id = entry.get("insertId") or message
            logs.append(LogEntry(key=insert_id, message=message))
        return logs

    def stop(self) -> None:
        self._gcloud(
            [
                "gcloud",
                "ai",
                "custom-jobs",
                "cancel",
                self.custom_job_id,
                f"--project={self.project_id}",
                f"--region={self.region}",
            ]
        )


def _accelerator_type(compute) -> str | None:
    """The Vertex accelerator name to request, if any.

    A raw `accelerator_type=` wins so that names coalesce does not model (TPUs,
    for instance) reach Vertex untouched; otherwise the portable `gpu=` name is
    translated.
    """
    if compute.vertex_accelerator_type:
        return compute.vertex_accelerator_type
    if compute.gpu:
        return spec_module.vertex_gpu(compute.gpu)
    return None


def _print_plan(
    job_spec: JobSpec,
    job_kwargs: dict[str, Any],
    staging_bucket_uri: str,
    environment_variables: dict[str, str],
) -> None:
    print("\n--- dry run: Vertex AI CustomJob that would be submitted ---")
    print(f"  project: {job_spec.gcp_project_id}")
    print(f"  staging bucket: {staging_bucket_uri}")
    for key in sorted(job_kwargs):
        if key == "environment_variables":
            continue
        print(f"  {key}: {job_kwargs[key]}")
    print("  environment_variables:")
    for key in sorted(environment_variables):
        print(f"    {key}={environment_variables[key]}")
    print("--- nothing was submitted and nothing was uploaded ---\n")


def launch(job_spec: JobSpec) -> Job | None:
    """Submit ``job_spec`` as a Vertex AI custom job.

    Returns ``None`` on a dry run, which prints the plan instead of creating
    anything.
    """
    compute = job_spec.compute
    _, _, staging_bucket_uri = staging.staging_paths(job_spec)
    container_uri = job_spec.container_uri or DEFAULT_CONTAINER_URI

    print(f"Launching job: {job_spec.job_name}")
    print("  Provider: GCP Vertex AI")
    print(f"  Region: {job_spec.region}")
    print(f"  Machine: {compute.machine_type}")
    accelerator_type = _accelerator_type(compute)
    if accelerator_type:
        print(f"  GPU: {accelerator_type} x{compute.gpu_count}")
    else:
        print("  GPU: None (CPU-only)")

    if not job_spec.dry_run:
        aiplatform.init(
            project=job_spec.gcp_project_id,
            location=job_spec.region,
            staging_bucket=staging_bucket_uri,
        )

    environment_variables = staging.build_environment(job_spec)
    task_py_dest = staging.stage_task_runner()

    job_kwargs: dict[str, Any] = {
        "display_name": job_spec.job_name,
        "script_path": str(task_py_dest),
        "container_uri": container_uri,
        "machine_type": compute.machine_type,
        "boot_disk_type": compute.boot_disk_type,
        "boot_disk_size_gb": compute.boot_disk_size_gb,
        "environment_variables": environment_variables,
    }

    if job_spec.extra_packages:
        job_kwargs["requirements"] = job_spec.extra_packages
        print(f"  Extra packages: {', '.join(job_spec.extra_packages)}")

    if accelerator_type:
        job_kwargs["accelerator_type"] = accelerator_type
        job_kwargs["accelerator_count"] = compute.gpu_count

    if job_spec.dry_run:
        _print_plan(job_spec, job_kwargs, staging_bucket_uri, environment_variables)
        return None

    job = aiplatform.CustomJob.from_local_script(**job_kwargs)

    from google.cloud.aiplatform_v1.types import custom_job as gca_custom_job_compat

    run_kwargs: dict[str, Any] = {
        "sync": False if job_spec.stream_logs else job_spec.sync
    }
    strategy = job_spec.scheduling.strategy
    if strategy == "SPOT":
        print("  Scheduling: SPOT (preemptible, may be interrupted)")
        run_kwargs["scheduling_strategy"] = gca_custom_job_compat.Scheduling.Strategy.SPOT
        run_kwargs["restart_job_on_worker_restart"] = True
    elif strategy == "FLEX_START":
        print(
            f"  Scheduling: FLEX_START (will queue up to {job_spec.scheduling.max_wait_duration}s)"
        )
        run_kwargs["scheduling_strategy"] = (
            gca_custom_job_compat.Scheduling.Strategy.FLEX_START
        )
        run_kwargs["max_wait_duration"] = job_spec.scheduling.max_wait_duration
    else:
        print("  Scheduling: STANDARD (on-demand)")

    print("Submitting job...")
    job.run(**run_kwargs)

    custom_job_id = _custom_job_id(_wait_for_resource_name(job))
    backend = VertexJobBackend(
        custom_job_id=custom_job_id,
        project_id=job_spec.gcp_project_id,
        region=job_spec.region,
    )
    handle = Job(
        id=custom_job_id,
        name=job_spec.job_name,
        provider="vertex",
        backend=backend,
        raw=job,
    )

    if job_spec.stream_logs:
        print(f"Streaming logs for custom job: {custom_job_id}")
        handle.stream_logs(poll_interval=job_spec.log_polling_interval)
        print(f"Job log stream finished: {job_spec.job_name}")
    elif job_spec.sync:
        print(f"Job completed: {job_spec.job_name}")
    else:
        print(f"Job submitted: {job_spec.job_name}")

    return handle
