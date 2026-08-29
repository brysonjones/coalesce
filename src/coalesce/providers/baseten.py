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
import time
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

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

# Registries whose images the GCP service account key can already pull.
GOOGLE_REGISTRY_SUFFIXES = (".pkg.dev", "gcr.io")


class _Remote(NamedTuple):
    """How to reach Baseten: the truss remote name, the key, and the app URL."""

    name: str
    api_key: str
    app_url: str


def _env_remote_name() -> str:
    """The literal remote name truss uses for env-supplied credentials."""
    try:
        from truss.remote.remote_factory import ENV_REMOTE_NAME

        return ENV_REMOTE_NAME
    except ImportError:
        return "<environment>"


def _remote_from_trussrc() -> _Remote | None:
    """Credentials from a previous ``truss login``, if there are any."""
    try:
        from truss.remote.remote_factory import RemoteFactory

        configs = RemoteFactory.load_remote_config("baseten").configs
    except Exception:
        return None

    api_key = configs.get("api_key")
    if not api_key:
        return None
    app_url = str(configs.get("remote_url") or DEFAULT_APP_URL).rstrip("/")
    return _Remote(name="baseten", api_key=str(api_key), app_url=app_url)


def resolve_remote() -> _Remote:
    """Work out how to authenticate, and under which truss remote name.

    truss takes credentials either from the BASETEN_TRUSS_AUTH_* pair or from
    ``~/.trussrc``, and the two are mutually exclusive: once the pair is set,
    the *only* remote name truss will accept is its env remote, and passing
    "baseten" is an error. coalesce also calls the REST API directly, so it
    needs the key itself either way.

    A plain BASETEN_API_KEY is the common case; the pair is derived from it so
    that no interactive ``truss login`` is required.
    """
    env_key = os.environ.get("BASETEN_TRUSS_AUTH_API_KEY")
    env_url = os.environ.get("BASETEN_TRUSS_AUTH_REMOTE_URL")
    if env_key and env_url:
        return _Remote(_env_remote_name(), env_key, env_url.rstrip("/"))

    api_key = os.environ.get("BASETEN_API_KEY")
    if api_key:
        app_url = (env_url or DEFAULT_APP_URL).rstrip("/")
        os.environ["BASETEN_TRUSS_AUTH_API_KEY"] = api_key
        os.environ["BASETEN_TRUSS_AUTH_REMOTE_URL"] = app_url
        return _Remote(_env_remote_name(), api_key, app_url)

    from_trussrc = _remote_from_trussrc()
    if from_trussrc:
        return from_trussrc

    raise RuntimeError(
        "No Baseten API key found. Set BASETEN_API_KEY in your environment, or "
        "run `truss login`."
    )


def resolve_api_key() -> str:
    """The Baseten API key coalesce should authenticate REST calls with."""
    return resolve_remote().api_key


def _app_url() -> str:
    return resolve_remote().app_url


def _api_url() -> str:
    """REST base URL, derived from the app URL the same way truss derives it."""
    app_url = _app_url()
    if app_url == DEFAULT_APP_URL:
        return DEFAULT_API_URL
    return app_url.replace("://app.", "://api.", 1)


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


def _registry_host(container_uri: str) -> str:
    return container_uri.split("/", 1)[0]


def _docker_auth(job_spec: JobSpec, container_uri: str, td, truss_config):
    """Credentials for pulling a private image, when the image needs them.

    Baseten has to pull the image itself, so an image in a private registry
    needs a secret. Images in a Google registry reuse the service account key
    that already grants the job access to GCS, which means a private
    Artifact Registry image works with no extra setup beyond granting that
    account artifactregistry.reader.
    """
    host = _registry_host(container_uri)
    secret = job_spec.container_registry_secret
    if secret is None:
        if not host.endswith(GOOGLE_REGISTRY_SUFFIXES):
            return None
        secret = job_spec.gcp_credentials_secret
    if not secret:
        return None

    print(f"  Image pull: authenticating to {host} with Baseten secret {secret!r}")
    if host.endswith(GOOGLE_REGISTRY_SUFFIXES):
        return td.DockerAuth(
            auth_method=truss_config.DockerAuthType.GCP_SERVICE_ACCOUNT_JSON,
            registry=host,
            gcp_service_account_json_docker_auth=td.GCPServiceAccountJSONDockerAuth(
                service_account_json_secret_ref=td.SecretReference(name=secret)
            ),
        )
    return td.DockerAuth(
        auth_method=truss_config.DockerAuthType.REGISTRY_SECRET,
        registry=host,
        registry_secret_docker_auth=td.RegistrySecretDockerAuth(
            secret_ref=td.SecretReference(name=secret)
        ),
    )


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
    payload = Path(key_file).expanduser().read_text()
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


def _import_truss():
    """Import truss, or explain how to install it.

    truss is an optional dependency so that using Vertex AI never requires it;
    the raw ModuleNotFoundError does not say that.
    """
    try:
        from truss.base import truss_config
        from truss_train import definitions as td
        from truss_train import push
    except ImportError as exc:
        raise ImportError(
            "The Baseten provider needs truss, which is not installed. "
            "Install it with `pip install 'coalesce[baseten]'`."
        ) from exc
    return truss_config, td, push


def build_training_project(job_spec: JobSpec, environment_variables: dict[str, str]):
    """Translate a :class:`JobSpec` into Baseten's training definitions."""
    truss_config, td, _ = _import_truss()

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
        image=td.Image(
            base_image=container_uri,
            docker_auth=_docker_auth(job_spec, container_uri, td, truss_config),
        ),
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


# Baseten creates the training project as a side effect of the first push into
# it. Concurrent launches therefore all see the project missing and all try to
# create it, and every loser gets this error even though the project it wanted
# now exists. Retrying resolves it, because the retry finds the winner's project.
_PROJECT_RACE_MARKER = "training project name uniqueness"


def _is_project_race(exc: Exception) -> bool:
    """True when a push failed only because a concurrent push created the project."""
    response = getattr(exc, "response", None)
    body = ""
    if response is not None:
        try:
            body = response.text or ""
        except Exception:  # noqa: BLE001 - a body we cannot read is not a race
            body = ""
    return _PROJECT_RACE_MARKER in f"{exc} {body}".lower()


def _push_with_retry(push, training_project, *, source_dir, remote, attempts=4):
    for attempt in range(1, attempts + 1):
        try:
            return push(training_project, source_dir=source_dir, remote=remote)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is the race
            if attempt == attempts or not _is_project_race(exc):
                raise
            print(
                f"  Another job created project '{training_project.name}' first; "
                f"retrying ({attempt}/{attempts - 1})."
            )
            time.sleep(2.0 * attempt)


def _print_plan(training_project) -> None:
    print("\n--- dry run: Baseten training job that would be submitted ---")
    print(f"  project: {training_project.name}")
    job = training_project.job
    print(f"  name: {job.name}")
    print(f"  image: {job.image.base_image}")
    if job.image.docker_auth:
        print(
            f"  image pull auth: {job.image.docker_auth.auth_method.value} "
            f"for {job.image.docker_auth.registry}"
        )
    print(f"  compute: {job.compute.model_dump()}")
    print(f"  priority: {job.priority}")
    print("  start_commands:")
    for command in job.runtime.start_commands:
        print(f"    $ {command}")
    print("  environment_variables:")
    for key in sorted(job.runtime.environment_variables):
        value = job.runtime.environment_variables[key]
        rendered = (
            f"<baseten secret {value.name!r}>" if hasattr(value, "name") else value
        )
        print(f"    {key}={rendered}")
    print(f"  checkpointing: {job.runtime.checkpointing_config.model_dump()}")
    if job.weights:
        print("  mounts:")
        for weight in job.weights:
            print(f"    {weight.source} -> {weight.mount_location}")
    print("--- nothing was submitted and nothing was uploaded ---\n")


def launch(job_spec: JobSpec) -> Job | None:
    """Submit ``job_spec`` as a Baseten training job.

    Returns ``None`` on a dry run, which prints the plan instead of creating
    anything.
    """
    _, _, push = _import_truss()

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

    remote = None if job_spec.dry_run else resolve_remote()

    environment_variables = staging.build_environment(job_spec)
    source_dir = Path(tempfile.mkdtemp(prefix="coalesce_baseten_"))
    staging.stage_task_runner(source_dir)

    training_project = build_training_project(job_spec, environment_variables)

    if job_spec.dry_run:
        _print_plan(training_project)
        return None

    print("Submitting job...")
    response = _push_with_retry(
        push, training_project, source_dir=source_dir, remote=remote.name
    )

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


def _main() -> None:
    """One-time setup: `python -m coalesce.providers.baseten key.json`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Store a GCP service account key as a Baseten secret so "
        "Baseten jobs can read and write your GCS bucket."
    )
    parser.add_argument("key_file", help="Path to the service account JSON key")
    parser.add_argument(
        "--secret-name",
        default="gcp_service_account_json",
        help="Baseten secret name (default: gcp_service_account_json)",
    )
    args = parser.parse_args()
    upload_gcp_credentials(args.key_file, secret_name=args.secret_name)


if __name__ == "__main__":
    _main()
