from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock

import pytest

from coalesce import launcher


def sample_task() -> None:
    pass


@pytest.fixture
def mock_vertex(monkeypatch):
    job = Mock()
    job.resource_name = "projects/demo/locations/us-central1/customJobs/123456789"
    from_local_script = Mock(return_value=job)

    monkeypatch.setattr(launcher.aiplatform, "init", Mock())
    monkeypatch.setattr(launcher.aiplatform.CustomJob, "from_local_script", from_local_script)

    return job, from_local_script


def test_launch_job_defaults_to_sync_wait(mock_vertex) -> None:
    job, from_local_script = mock_vertex

    result = launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        container_uri="image",
    )

    assert result is job
    launcher.aiplatform.init.assert_called_once_with(
        project="demo-project",
        location="us-central1",
        staging_bucket="gs://demo-bucket/.coalesce/tmp",
    )
    assert from_local_script.call_args.kwargs["display_name"].startswith("sample_task_")
    job.run.assert_called_once()
    assert job.run.call_args.kwargs["sync"] is True


def test_launch_job_preserves_bucket_prefix_for_staging(mock_vertex) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket/experiments/run-1",
        container_uri="image",
        staging_prefix="tmp/coalesce",
    )

    launcher.aiplatform.init.assert_called_once_with(
        project="demo-project",
        location="us-central1",
        staging_bucket="gs://demo-bucket/experiments/run-1/tmp/coalesce",
    )


def test_launch_job_uploads_synced_packages_under_staging_prefix(monkeypatch, mock_vertex) -> None:
    package_and_upload = Mock(return_value="gs://demo-bucket/.coalesce/tmp/source/workspace.zip")
    monkeypatch.setattr(launcher, "package_and_upload", package_and_upload)

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        container_uri="image",
        sync_packages=["demo_package"],
    )

    package_and_upload.assert_called_once_with(
        package_names=["demo_package"],
        bucket_name="demo-bucket",
        project_id="demo-project",
        prefix=".coalesce/tmp/source",
    )


def test_launch_job_async_without_streaming(mock_vertex) -> None:
    job, _ = mock_vertex

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        container_uri="image",
        sync=False,
    )

    job.run.assert_called_once()
    assert job.run.call_args.kwargs["sync"] is False


def test_wait_for_resource_name_retries_until_available(monkeypatch) -> None:
    class Job:
        attempts = [RuntimeError("CustomJob resource has not been created."), "projects/demo/jobs/123"]

        @property
        def resource_name(self):
            result = self.attempts.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    job = Job()
    monkeypatch.setattr(launcher.time, "sleep", Mock())

    assert launcher._wait_for_resource_name(job, timeout_seconds=5) == "projects/demo/jobs/123"
    launcher.time.sleep.assert_called_once_with(1)


def test_launch_job_streams_logs_with_gcloud(monkeypatch, mock_vertex) -> None:
    job, _ = mock_vertex

    def run(command, **kwargs):
        if command[:3] == ["gcloud", "logging", "read"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    [
                        {
                            "insertId": "log-1",
                            "textPayload": "remote print output",
                        }
                    ]
                ),
            )
        if command[:4] == ["gcloud", "ai", "custom-jobs", "describe"]:
            return subprocess.CompletedProcess(command, 0, stdout="JOB_STATE_SUCCEEDED\n")
        raise AssertionError(f"unexpected command: {command}")

    run = Mock(side_effect=run)
    monkeypatch.setattr(launcher.subprocess, "run", run)

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        region="us-central1",
        container_uri="image",
        stream_logs=True,
        log_polling_interval=7,
    )

    assert job.run.call_args.kwargs["sync"] is False
    run.assert_any_call(
        [
            "gcloud",
            "logging",
            "read",
            'resource.type="ml_job" AND resource.labels.job_id="123456789"',
            "--project=demo-project",
            "--format=json",
            "--limit=200",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run.assert_any_call(
        [
            "gcloud",
            "ai",
            "custom-jobs",
            "describe",
            "123456789",
            "--project=demo-project",
            "--region=us-central1",
            "--format=value(state)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_stream_logs_missing_gcloud_has_clear_error(monkeypatch) -> None:
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("gcloud")

    monkeypatch.setattr(launcher.subprocess, "run", raise_missing)

    with pytest.raises(RuntimeError, match="gcloud"):
        launcher._stream_custom_job_logs(
            custom_job_id="123",
            project_id="demo-project",
            region="us-central1",
            polling_interval=10,
            allow_multiline_logs=True,
        )


def test_stream_logs_failed_command_has_clear_error(monkeypatch) -> None:
    def raise_failed(command, **kwargs):
        raise subprocess.CalledProcessError(returncode=2, cmd=command)

    monkeypatch.setattr(launcher.subprocess, "run", raise_failed)

    with pytest.raises(RuntimeError, match="exit code 2"):
        launcher._stream_custom_job_logs(
            custom_job_id="123",
            project_id="demo-project",
            region="us-central1",
            polling_interval=10,
            allow_multiline_logs=True,
        )
