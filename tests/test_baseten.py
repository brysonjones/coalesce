from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
import requests

from coalesce import launcher, staging
from coalesce.providers import baseten


def sample_task() -> None:
    pass


@pytest.fixture(autouse=True)
def baseten_credentials(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    monkeypatch.delenv("BASETEN_TRUSS_AUTH_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_TRUSS_AUTH_REMOTE_URL", raising=False)


@pytest.fixture
def mock_push(monkeypatch):
    """Intercept the truss push so tests assert on the submitted definition."""
    response = {
        "id": "job-abc",
        "training_project": {"id": "proj-xyz", "name": "sample_task"},
        "current_status": "TRAINING_JOB_CREATED",
    }
    push = Mock(return_value=response)
    monkeypatch.setattr("truss_train.push", push)
    return push


def submitted_job(push):
    """The TrainingJob that was handed to truss."""
    return push.call_args.args[0].job


def test_launch_returns_a_baseten_job_handle(mock_push) -> None:
    job = launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync=False,
    )

    assert job.provider == "baseten"
    assert job.id == "job-abc"
    assert job.raw is mock_push.return_value
    assert job.console_url == "https://app.baseten.co/training/proj-xyz/logs/job-abc"


def test_gpu_name_is_translated_to_baseten_vocabulary(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        gpu="H100",
        gpu_count=2,
        cpu_count=16,
        memory="128Gi",
        sync=False,
    )

    compute = submitted_job(mock_push).compute
    assert compute.accelerator.accelerator.value == "H100"
    assert compute.accelerator.count == 2
    assert compute.cpu_count == 16
    assert compute.memory == "128Gi"


def test_legacy_vertex_accelerator_names_still_map(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        accelerator_type="NVIDIA_TESLA_T4",
        sync=False,
    )

    assert submitted_job(mock_push).compute.accelerator.accelerator.value == "T4"


def test_vertex_only_accelerator_is_rejected_not_silently_dropped(mock_push) -> None:
    with pytest.raises(ValueError, match="no Baseten equivalent"):
        launcher.launch_job(
            func=sample_task,
            project_id="demo-project",
            bucket="gs://demo-bucket",
            provider="baseten",
            accelerator_type="TPU_V2",
            sync=False,
        )


def test_gpu_unavailable_on_baseten_is_rejected(mock_push) -> None:
    with pytest.raises(ValueError, match="not available on Baseten"):
        launcher.launch_job(
            func=sample_task,
            project_id="demo-project",
            bucket="gs://demo-bucket",
            provider="baseten",
            gpu="GB200",
            sync=False,
        )


def test_cpu_only_job_requests_no_accelerator(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        accelerator_type=None,
        sync=False,
    )

    assert submitted_job(mock_push).compute.accelerator is None


def test_task_runner_and_extra_packages_are_installed_before_the_entrypoint(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        extra_packages=["transformers", "accelerate>=0.20"],
        sync=False,
    )

    commands = submitted_job(mock_push).runtime.start_commands
    assert commands[-1] == "python -u task.py"
    install = commands[0]
    assert install.startswith("pip install --no-cache-dir ")
    for requirement in ("google-cloud-storage", "pyyaml", "transformers", "accelerate>=0.20"):
        assert f"'{requirement}'" in install


def test_gcs_credentials_are_injected_as_a_secret_reference(mock_push) -> None:
    from truss_train import definitions as td

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync=False,
    )

    env = submitted_job(mock_push).runtime.environment_variables
    secret = env[baseten.GCP_CREDENTIALS_ENV]
    assert isinstance(secret, td.SecretReference)
    assert secret.name == "gcp_service_account_json"


def test_task_environment_matches_what_the_runner_expects(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        config={"learning_rate": 0.001},
        env={"WANDB_API_KEY": "xxx"},
        sync=False,
    )

    env = submitted_job(mock_push).runtime.environment_variables
    assert env["TASK_MODULE"] == sample_task.__module__
    assert env["TASK_FUNCTION"] == "sample_task"
    assert json.loads(env["TASK_CONFIG_JSON"]) == {"learning_rate": 0.001}
    assert env["WANDB_API_KEY"] == "xxx"


def test_synced_packages_go_to_the_same_gcs_prefix_as_vertex(monkeypatch, mock_push) -> None:
    package_and_upload = Mock(return_value="gs://demo-bucket/.coalesce/tmp/source/workspace.zip")
    monkeypatch.setattr(staging, "package_and_upload", package_and_upload)

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync_packages=["demo_package"],
        sync=False,
    )

    package_and_upload.assert_called_once_with(
        package_names=["demo_package"],
        bucket_name="demo-bucket",
        project_id="demo-project",
        prefix=".coalesce/tmp/source",
    )
    env = submitted_job(mock_push).runtime.environment_variables
    assert env["SYNC_PACKAGES_GCS_URI"] == package_and_upload.return_value


def test_spot_maps_to_interruptible_capacity(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        scheduling_strategy="SPOT",
        sync=False,
    )

    assert submitted_job(mock_push).compute.availability_model.value == "spot"


def test_flex_start_falls_back_to_dedicated_and_says_so(mock_push, capsys) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        scheduling_strategy="FLEX_START",
        sync=False,
    )

    assert submitted_job(mock_push).compute.availability_model.value == "dedicated"
    assert "queues for capacity natively" in capsys.readouterr().out


def test_boot_disk_settings_are_reported_as_ignored(mock_push, capsys) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        boot_disk_size_gb=500,
        sync=False,
    )

    out = capsys.readouterr().out
    assert "ignored on Baseten" in out


def test_default_boot_disk_settings_stay_quiet(mock_push, capsys) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync=False,
    )

    assert "ignored on Baseten" not in capsys.readouterr().out


def test_checkpoint_volume_enables_persistent_storage(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        checkpoint_volume_gb=250,
        sync=False,
    )

    checkpointing = submitted_job(mock_push).runtime.checkpointing_config
    assert checkpointing.enabled is True
    assert checkpointing.volume_size_gib == 250


def test_mounted_datasets_become_read_only_weights_sources(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        mount_datasets=["gs://demo-bucket/datasets/imagenet:/mnt/data"],
        sync=False,
    )

    weights = submitted_job(mock_push).weights
    assert len(weights) == 1
    assert weights[0].source == "gs://demo-bucket/datasets/imagenet"
    assert weights[0].mount_location == "/mnt/data"
    assert weights[0].auth_secret_name == "gcp_service_account_json"


@pytest.mark.parametrize("mount", ["gs://bucket/path", "not-a-uri:/mnt/data"])
def test_malformed_mount_specs_are_rejected(mount) -> None:
    with pytest.raises(ValueError, match="Invalid mount"):
        baseten._parse_mount(mount)


def test_baseten_project_defaults_to_the_function_name(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync=False,
    )

    assert mock_push.call_args.args[0].name == "sample_task"


def test_baseten_project_can_be_named(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        baseten_project="robot-pretraining",
        sync=False,
    )

    assert mock_push.call_args.args[0].name == "robot-pretraining"


def test_truss_auth_is_derived_from_baseten_api_key(mock_push) -> None:
    import os

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync=False,
    )

    assert os.environ["BASETEN_TRUSS_AUTH_API_KEY"] == "test-key"
    assert os.environ["BASETEN_TRUSS_AUTH_REMOTE_URL"] == "https://app.baseten.co"


def test_push_targets_the_env_remote_when_credentials_come_from_the_environment(
    mock_push,
) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync=False,
    )

    # Once BASETEN_TRUSS_AUTH_* are set, truss rejects the name "baseten" and
    # only accepts its env remote, so the two have to stay in step.
    from truss.remote.remote_factory import ENV_REMOTE_NAME

    assert mock_push.call_args.kwargs["remote"] == ENV_REMOTE_NAME


def test_a_previous_truss_login_is_used_when_no_env_key_is_set(
    monkeypatch, mock_push
) -> None:
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setattr(
        baseten,
        "_remote_from_trussrc",
        lambda: baseten._Remote("baseten", "trussrc-key", "https://app.baseten.co"),
    )

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync=False,
    )

    assert mock_push.call_args.kwargs["remote"] == "baseten"
    assert baseten.resolve_api_key() == "trussrc-key"


def test_missing_api_key_is_a_clear_error(monkeypatch, mock_push) -> None:
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.setattr(baseten, "_remote_from_trussrc", lambda: None)

    with pytest.raises(RuntimeError, match="No Baseten API key found"):
        launcher.launch_job(
            func=sample_task,
            project_id="demo-project",
            bucket="gs://demo-bucket",
            provider="baseten",
            sync=False,
        )


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_backend_reads_status_and_logs_over_rest(monkeypatch) -> None:
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs.get("json")))
        if url.endswith("/logs"):
            return FakeResponse(
                {
                    "logs": [
                        {"timestamp": "200", "message": "second", "replica": "0"},
                        {"timestamp": "100", "message": "first", "replica": "0"},
                    ]
                }
            )
        return FakeResponse({"training_job": {"current_status": "TRAINING_JOB_RUNNING"}})

    monkeypatch.setattr(baseten.requests, "request", request)
    backend = baseten.BasetenJobBackend(project_id="proj-xyz", job_id="job-abc")

    assert backend.status() == "TRAINING_JOB_RUNNING"
    logs = backend.fetch_logs(1000, 2000)
    # The API hands back newest-first; the runner prints oldest-first.
    assert [entry.message for entry in logs] == ["first", "second"]
    assert calls[-1][2] == {"start_epoch_millis": 1000, "end_epoch_millis": 2000}


def test_backend_treats_unknown_failure_states_as_terminal() -> None:
    backend = baseten.BasetenJobBackend(project_id="p", job_id="j")

    assert backend.is_terminal("TRAINING_JOB_COMPLETED")
    assert backend.is_terminal("TRAINING_JOB_SOMETHING_FAILED")
    assert not backend.is_terminal("TRAINING_JOB_RUNNING")


def test_backend_surfaces_api_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        baseten.requests,
        "request",
        lambda *a, **k: FakeResponse({"error": "nope"}, status_code=404),
    )
    backend = baseten.BasetenJobBackend(project_id="p", job_id="j")

    with pytest.raises(RuntimeError, match="404"):
        backend.status()


def test_sync_launch_waits_for_the_job_to_finish(monkeypatch, mock_push, capsys) -> None:
    statuses = iter(["TRAINING_JOB_RUNNING", "TRAINING_JOB_COMPLETED"])
    monkeypatch.setattr(baseten.BasetenJobBackend, "status", lambda self: next(statuses))
    monkeypatch.setattr("coalesce.job.time.sleep", lambda _: None)

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync=True,
    )

    assert "Job completed: " in capsys.readouterr().out


def test_sync_launch_reports_why_a_job_failed(monkeypatch, mock_push, capsys) -> None:
    monkeypatch.setattr(
        baseten.BasetenJobBackend, "status", lambda self: "TRAINING_JOB_FAILED"
    )
    monkeypatch.setattr(
        baseten.BasetenJobBackend, "error_message", lambda self: "OOMKilled"
    )

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        sync=True,
    )

    out = capsys.readouterr().out
    assert "TRAINING_JOB_FAILED" in out
    assert "OOMKilled" in out


def test_upload_gcp_credentials_stores_the_key_as_a_secret(monkeypatch, tmp_path) -> None:
    key_file = tmp_path / "key.json"
    key_file.write_text(json.dumps({"type": "service_account", "project_id": "demo"}))
    posted = {}

    def post(url, headers, json, timeout):
        posted.update(url=url, headers=headers, body=json)
        return Mock(ok=True, status_code=200, json=lambda: {"name": json["name"]}, raise_for_status=lambda: None)

    monkeypatch.setattr(baseten.requests, "post", post)

    baseten.upload_gcp_credentials(key_file, secret_name="gcp_sa")

    assert posted["url"] == "https://api.baseten.co/v1/secrets"
    assert posted["headers"]["Authorization"] == "Bearer test-key"
    assert posted["body"]["name"] == "gcp_sa"
    assert json.loads(posted["body"]["value"])["project_id"] == "demo"


def test_private_google_images_reuse_the_gcp_credentials_secret(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        container_uri="us-docker.pkg.dev/demo-project/images/trainer:latest",
        sync=False,
    )

    auth = submitted_job(mock_push).image.docker_auth
    assert auth.auth_method.value == "GCP_SERVICE_ACCOUNT_JSON"
    assert auth.registry == "us-docker.pkg.dev"
    assert (
        auth.gcp_service_account_json_docker_auth.service_account_json_secret_ref.name
        == "gcp_service_account_json"
    )


def test_public_images_are_pulled_without_credentials(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        container_uri="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime",
        sync=False,
    )

    assert submitted_job(mock_push).image.docker_auth is None


def test_a_named_secret_pulls_from_any_registry(mock_push) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        container_uri="ghcr.io/acme/trainer:latest",
        container_registry_secret="ghcr_token",
        sync=False,
    )

    auth = submitted_job(mock_push).image.docker_auth
    assert auth.auth_method.value == "REGISTRY_SECRET"
    assert auth.registry == "ghcr.io"
    assert auth.registry_secret_docker_auth.secret_ref.name == "ghcr_token"


def _project_race_error() -> requests.HTTPError:
    """The error Baseten returns to whichever concurrent push lost the race."""
    response = FakeResponse(
        {
            "message": "Constraint “Training project name uniqueness across "
            "organization” is violated."
        },
        status_code=400,
    )
    error = requests.HTTPError("400 Client Error: Bad Request")
    error.response = response
    return error


def test_losing_a_project_creation_race_is_retried_not_surfaced() -> None:
    """Parallel launches into one project must not fail on the project itself.

    Baseten creates the training project as a side effect of the first push, so
    concurrent launches all try to create it and all but one are rejected -- even
    though the project they wanted now exists.
    """
    attempts = []

    def push(training_project, source_dir, remote):
        attempts.append(remote)
        if len(attempts) < 3:
            raise _project_race_error()
        return {"id": "job-1"}

    project = Mock()
    project.name = "shared-project"
    with patch.object(baseten.time, "sleep"):
        response = baseten._push_with_retry(
            push, project, source_dir="/tmp/src", remote="baseten"
        )

    assert response == {"id": "job-1"}
    assert len(attempts) == 3


def test_a_push_failure_that_is_not_the_race_is_raised_immediately() -> None:
    """A real rejection must not be retried into a slow, confusing failure."""
    response = FakeResponse({"message": "GPU type T4 is not supported"}, status_code=400)
    error = requests.HTTPError("400 Client Error: Bad Request")
    error.response = response
    attempts = []

    def push(training_project, source_dir, remote):
        attempts.append(remote)
        raise error

    project = Mock()
    project.name = "shared-project"
    with patch.object(baseten.time, "sleep"), pytest.raises(requests.HTTPError):
        baseten._push_with_retry(push, project, source_dir="/tmp/src", remote="baseten")

    assert len(attempts) == 1, "a non-race error should not be retried"


def test_the_race_is_given_up_on_rather_than_retried_forever() -> None:
    def push(training_project, source_dir, remote):
        raise _project_race_error()

    project = Mock()
    project.name = "shared-project"
    with patch.object(baseten.time, "sleep"), pytest.raises(requests.HTTPError):
        baseten._push_with_retry(
            push, project, source_dir="/tmp/src", remote="baseten", attempts=2
        )
