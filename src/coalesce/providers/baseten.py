"""Run coalesce jobs on Baseten Training.

Baseten is treated as an interchangeable compute backend: the job is staged to
the same GCS bucket a Vertex job would use, and the same ``task.py`` runs on
the far side. The only Baseten-specific piece is credentials -- a container on
Baseten has no ambient GCP identity, so a service account key stored as a
Baseten secret is injected as an environment variable and turned into
Application Default Credentials by the task runner.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import requests

from .. import spec as spec_module
from .. import staging
from ..job import Job
from ..spec import JobSpec
from .base import JobBackend, LogEntry

DEFAULT_CONTAINER_URI = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"
DEFAULT_APP_URL = "https://app.baseten.co"
DEFAULT_API_URL = "https://api.baseten.co"

# Installed on the remote before task.py runs, because task.py reads its inputs
# straight out of GCS.
TASK_RUNNER_REQUIREMENTS = ("google-cloud-storage", "pyyaml")

# The env var task.py reads its GCP service account key from.
GCP_CREDENTIALS_ENV = "GCP_SERVICE_ACCOUNT_JSON"


def resolve_api_key() -> str:
    """Find the Baseten API key, preferring the truss-specific variable."""
    api_key = os.environ.get("BASETEN_TRUSS_AUTH_API_KEY") or os.environ.get(
        "BASETEN_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "No Baseten API key found. Set BASETEN_API_KEY (or "
            "BASETEN_TRUSS_AUTH_API_KEY) in your environment."
        )
    return api_key


def _app_url() -> str:
    return os.environ.get("BASETEN_TRUSS_AUTH_REMOTE_URL", DEFAULT_APP_URL).rstrip("/")


def _api_url() -> str:
    """REST base URL, derived from the app URL the same way truss derives it."""
    app_url = _app_url()
    if app_url == DEFAULT_APP_URL:
        return DEFAULT_API_URL
    return app_url.replace("://app.", "://api.", 1)


def _ensure_truss_env() -> None:
    """Let ``truss`` authenticate from BASETEN_API_KEY alone.

    truss reads either ``~/.trussrc`` or the pair of BASETEN_TRUSS_AUTH_*
    variables. Deriving the pair from BASETEN_API_KEY means a shell that only
    exports the plain key still works, with no interactive ``truss login``.
    """
    if os.environ.get("BASETEN_TRUSS_AUTH_API_KEY") and os.environ.get(
        "BASETEN_TRUSS_AUTH_REMOTE_URL"
    ):
        return
    api_key = resolve_api_key()
    os.environ["BASETEN_TRUSS_AUTH_API_KEY"] = api_key
    os.environ.setdefault("BASETEN_TRUSS_AUTH_REMOTE_URL", DEFAULT_APP_URL)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {resolve_api_key()}"}


def _parse_mount(mount: str) -> tuple[str, str]:
    """Split a ``gs://bucket/path:/container/path`` mount spec."""
    scheme, _, rest = mount.partition("://")
    if not rest:
        raise ValueError(
            f"Invalid mount {mount!r}; expected 'gs://bucket/path:/container/path'."
        )
    source_rest, sep, mount_location = rest.rpartition(":")
    if not sep or not mount_location.startswith("/"):
        raise ValueError(
            f"Invalid mount {mount!r}; expected 'gs://bucket/path:/container/path'."
        )
    return f"{scheme}://{source_rest}", mount_location


def upsert_secret(name: str, value: str) -> dict[str, Any]:
    """Create or replace a Baseten workspace secret."""
    response = requests.post(
        f"{_api_url()}/v1/secrets",
        headers=_auth_headers(),
        json={"name": name, "value": value},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def upload_gcp_credentials(
    key_file: str | Path, secret_name: str = "gcp_service_account_json"
) -> dict[str, Any]:
    """Store a GCP service account key as a Baseten secret.

    This is the one-time setup that lets Baseten jobs read and write your GCS
    bucket. The key never enters a config file or the job definition -- it is
    referenced by name and injected at runtime.
    """
    payload = Path(key_file).read_text()
    result = upsert_secret(secret_name, payload)
    print(f"Stored GCP credentials as Baseten secret: {secret_name}")
    return result


class BasetenJobBackend(JobBackend):
    """Polls the Baseten Training REST API for job state and logs."""

    TERMINAL_STATES: ClassVar[frozenset[str]] = frozenset(
        {
            "TRAINING_JOB_COMPLETED",
            "TRAINING_JOB_FAILED",
            "TRAINING_JOB_DEPLOY_FAILED",
            "TRAINING_JOB_STOPPED",
        }
    )

    def __init__(self, *, project_id: str, job_id: str) -> None:
        self.project_id = project_id
        self.job_id = job_id
        self._base = f"{_api_url()}/v1/training_projects/{project_id}/jobs/{job_id}"

    @property
    def console_url(self) -> str | None:
        return f"{_app_url()}/training/{self.project_id}/logs/{self.job_id}"

    def _request(self, method: str, path: str = "", **kwargs: Any) -> dict[str, Any]:
        response = requests.request(
            method,
            f"{self._base}{path}",
            headers=_auth_headers(),
            timeout=60,
            **kwargs,
        )
        if not response.ok:
            raise RuntimeError(
                f"Baseten API {method} {path or '/'} failed with "
                f"{response.status_code}: {response.text.strip()}"
            )
        return response.json()

    def describe(self) -> dict[str, Any]:
        return self._request("GET")["training_job"]

    def status(self) -> str:
        return self.describe()["current_status"]

    def error_message(self) -> str | None:
        return self.describe().get("error_message")

    def is_terminal(self, state: str) -> bool:
        # Guard against states added server-side after this was written; a job
        # that has failed should never be polled forever.
        return state in self.TERMINAL_STATES or state.endswith(
            ("_COMPLETED", "_FAILED", "_STOPPED", "_CANCELLED")
        )

    def fetch_logs(
        self, start_epoch_ms: int | None, end_epoch_ms: int | None
    ) -> list[LogEntry]:
        body: dict[str, int] = {}
        if start_epoch_ms is not None:
            body["start_epoch_millis"] = start_epoch_ms
        if end_epoch_ms is not None:
            body["end_epoch_millis"] = end_epoch_ms

        payload = self._request("POST", "/logs", json=body)
        # The API returns newest first; print oldest first.
        entries = []
        for raw in reversed(payload.get("logs") or []):
            timestamp = raw.get("timestamp")
            message = raw.get("message", "")
            replica = raw.get("replica")
            entries.append(
                LogEntry(
                    key=f"{timestamp}-{replica}-{message}",
                    message=message,
                    timestamp_ns=int(timestamp) if timestamp else None,
                    replica=replica,
                )
            )
        return entries

    def stop(self) -> None:
        self._request("POST", "/stop")
        print(f"Requested stop for Baseten training job {self.job_id}")


def build_training_project(job_spec: JobSpec, environment_variables: dict[str, str]):
    """Translate a :class:`JobSpec` into Baseten's training definitions."""
    from truss.base import truss_config
    from truss_train import definitions as td

    compute = job_spec.compute
    container_uri = job_spec.container_uri or DEFAULT_CONTAINER_URI

    runtime_env: dict[str, Any] = dict(environment_variables)
    if job_spec.gcp_credentials_secret:
        runtime_env[GCP_CREDENTIALS_ENV] = td.SecretReference(
            name=job_spec.gcp_credentials_secret
        )

    requirements = list(TASK_RUNNER_REQUIREMENTS) + list(job_spec.extra_packages)
    start_commands = [
        "pip install --no-cache-dir " + " ".join(f"'{req}'" for req in requirements),
        "python -u task.py",
    ]

    checkpointing = td.CheckpointingConfig()
    if job_spec.checkpoint_volume_gb is not None:
        checkpointing = td.CheckpointingConfig(
            enabled=True, volume_size_gib=job_spec.checkpoint_volume_gb
        )

    accelerator = None
    if compute.gpu:
        accelerator = truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator(spec_module.baseten_gpu(compute.gpu)),
            count=compute.gpu_count,
        )
    elif compute.vertex_accelerator_type:
        raise ValueError(
            f"accelerator_type={compute.vertex_accelerator_type!r} is a Vertex AI name "
            "with no Baseten equivalent. Use the portable gpu= argument instead, e.g. "
            'gpu="H100".'
        )

    availability = (
        td.AvailabilityModel.SPOT
        if job_spec.scheduling.strategy == "SPOT"
        else td.AvailabilityModel.DEDICATED
    )

    weights = []
    for mount in job_spec.mount_datasets:
        source, mount_location = _parse_mount(mount)
        weights.append(
            truss_config.WeightsSource(
                source=source,
                mount_location=mount_location,
                auth_secret_name=job_spec.gcp_credentials_secret or None,
            )
        )

    training_job = td.TrainingJob(
        name=job_spec.job_name,
        image=td.Image(base_image=container_uri),
        compute=td.Compute(
            node_count=compute.node_count,
            cpu_count=compute.cpu_count,
            memory=compute.memory,
            accelerator=accelerator,
            availability_model=availability,
        ),
        runtime=td.Runtime(
            start_commands=start_commands,
            environment_variables=runtime_env,
            checkpointing_config=checkpointing,
        ),
        priority=job_spec.scheduling.priority,
        weights=weights,
    )

    return td.TrainingProject(
        name=job_spec.baseten_project or job_spec.func.__name__,
        job=training_job,
    )


def launch(job_spec: JobSpec) -> Job:
    """Submit ``job_spec`` as a Baseten training job."""
    from truss_train import push

    compute = job_spec.compute

    print(f"Launching job: {job_spec.job_name}")
    print("  Provider: Baseten")
    print(f"  CPU/memory: {compute.cpu_count} vCPU / {compute.memory}")
    if compute.gpu:
        print(
            f"  GPU: {compute.gpu} x{compute.gpu_count} "
            f"(Baseten: {spec_module.baseten_gpu(compute.gpu)})"
        )
    else:
        print("  GPU: None (CPU-only)")

    if not compute.boot_disk_is_default:
        print(
            "  Note: boot_disk_type/boot_disk_size_gb are Vertex AI settings and are "
            "ignored on Baseten, which provides ephemeral NVMe scratch at "
            "$BT_SCRATCH_DIR. Use checkpoint_volume_gb for persistent storage."
        )
    if job_spec.scheduling.strategy == "FLEX_START":
        print(
            "  Scheduling: FLEX_START requested; Baseten queues for capacity natively, "
            "so the job runs on dedicated capacity as soon as GPUs free up."
        )
    elif job_spec.scheduling.strategy == "SPOT":
        print("  Scheduling: SPOT (preemptible, may be interrupted)")
    else:
        print("  Scheduling: STANDARD (dedicated)")

    _ensure_truss_env()

    environment_variables = staging.build_environment(job_spec)
    source_dir = Path(tempfile.mkdtemp(prefix="coalesce_baseten_"))
    staging.stage_task_runner(source_dir)

    training_project = build_training_project(job_spec, environment_variables)

    print("Submitting job...")
    response = push(training_project, source_dir=source_dir)

    job_id = response["id"]
    project_id = response["training_project"]["id"]
    backend = BasetenJobBackend(project_id=project_id, job_id=job_id)
    handle = Job(
        id=job_id,
        name=job_spec.job_name,
        provider="baseten",
        backend=backend,
        raw=response,
    )
    print(f"Job submitted: {job_spec.job_name} (job id: {job_id})")
    print(f"  Console: {handle.console_url}")

    if job_spec.stream_logs:
        print(f"Streaming logs for training job: {job_id}")
        state = handle.stream_logs(poll_interval=job_spec.log_polling_interval)
        _report_failure(backend, state)
        print(f"Job log stream finished: {job_spec.job_name}")
    elif job_spec.sync:
        state = handle.wait(poll_interval=job_spec.log_polling_interval)
        _report_failure(backend, state)
        print(f"Job completed: {job_spec.job_name} ({state})")

    return handle


def _report_failure(backend: BasetenJobBackend, state: str) -> None:
    """Surface the server-side reason a job did not complete."""
    if state == "TRAINING_JOB_COMPLETED":
        return
    message = backend.error_message()
    if message:
        print(f"Job did not complete ({state}): {message}")
    else:
        print(f"Job did not complete ({state}).")
