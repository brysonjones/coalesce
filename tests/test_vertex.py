from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock

import pytest

from coalesce import launcher, staging
from coalesce.providers import vertex


def sample_task() -> None:
    pass


@pytest.fixture
def mock_vertex(monkeypatch):
    job = Mock()
    job.resource_name = "projects/demo/locations/us-central1/customJobs/123456789"
    from_local_script = Mock(return_value=job)

    monkeypatch.setattr(vertex.aiplatform, "init", Mock())
    monkeypatch.setattr(vertex.aiplatform.CustomJob, "from_local_script", from_local_script)

    return job, from_local_script


def test_launch_job_defaults_to_sync_wait(mock_vertex) -> None:
    job, from_local_script = mock_vertex

    result = launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        container_uri="image",
    )

    assert result.raw is job
    assert result.provider == "vertex"
    assert result.id == "123456789"
    vertex.aiplatform.init.assert_called_once_with(
        project="demo-project",
        location="us-central1",
        staging_bucket="gs://demo-bucket/.coalesce/tmp",
    )
    assert from_local_script.call_args.kwargs["display_name"].startswith("sample_task_")
    assert from_local_script.call_args.kwargs["boot_disk_type"] == "pd-ssd"
    assert from_local_script.call_args.kwargs["boot_disk_size_gb"] == 100
    job.run.assert_called_once()
    assert job.run.call_args.kwargs["sync"] is True


def test_launch_job_forwards_boot_disk_settings(mock_vertex) -> None:
    _, from_local_script = mock_vertex

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        container_uri="image",
        boot_disk_type="pd-balanced",
        boot_disk_size_gb=500,
    )

    assert from_local_script.call_args.kwargs["boot_disk_type"] == "pd-balanced"
    assert from_local_script.call_args.kwargs["boot_disk_size_gb"] == 500


def test_launch_job_preserves_bucket_prefix_for_staging(mock_vertex) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket/experiments/run-1",
        container_uri="image",
        staging_prefix="tmp/coalesce",
    )

    vertex.aiplatform.init.assert_called_once_with(
        project="demo-project",
        location="us-central1",
        staging_bucket="gs://demo-bucket/experiments/run-1/tmp/coalesce",
    )


def test_launch_job_uploads_synced_packages_under_staging_prefix(monkeypatch, mock_vertex) -> None:
    package_and_upload = Mock(return_value="gs://demo-bucket/.coalesce/tmp/source/workspace.zip")
    monkeypatch.setattr(staging, "package_and_upload", package_and_upload)

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


def test_default_container_uri_is_the_vertex_prebuilt_image(mock_vertex) -> None:
    _, from_local_script = mock_vertex

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
    )

    assert from_local_script.call_args.kwargs["container_uri"] == vertex.DEFAULT_CONTAINER_URI


def test_legacy_accelerator_type_is_forwarded_verbatim(mock_vertex) -> None:
    _, from_local_script = mock_vertex

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        container_uri="image",
        accelerator_type="TPU_V2",
        accelerator_count=8,
    )

    assert from_local_script.call_args.kwargs["accelerator_type"] == "TPU_V2"
    assert from_local_script.call_args.kwargs["accelerator_count"] == 8


def test_portable_gpu_name_is_translated(mock_vertex) -> None:
    _, from_local_script = mock_vertex

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        container_uri="image",
        gpu="H100",
        gpu_count=4,
    )

    assert from_local_script.call_args.kwargs["accelerator_type"] == "NVIDIA_H100_80GB"
    assert from_local_script.call_args.kwargs["accelerator_count"] == 4


def test_cpu_only_job_requests_no_accelerator(mock_vertex) -> None:
    _, from_local_script = mock_vertex

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        container_uri="image",
        accelerator_type=None,
    )

    assert "accelerator_type" not in from_local_script.call_args.kwargs


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
    monkeypatch.setattr(vertex.time, "sleep", Mock())

    assert vertex._wait_for_resource_name(job, timeout_seconds=5) == "projects/demo/jobs/123"
    vertex.time.sleep.assert_called_once_with(1)


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
    monkeypatch.setattr(vertex.subprocess, "run", run)

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


def test_streamed_logs_are_printed_once(monkeypatch, mock_vertex, capsys) -> None:
    def run(command, **kwargs):
        if command[:3] == ["gcloud", "logging", "read"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps([{"insertId": "log-1", "textPayload": "hello\nworld"}]),
            )
        return subprocess.CompletedProcess(command, 0, stdout="JOB_STATE_SUCCEEDED\n")

    monkeypatch.setattr(vertex.subprocess, "run", Mock(side_effect=run))

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        container_uri="image",
        stream_logs=True,
    )

    out = capsys.readouterr().out
    # The final drain re-reads the same window, so de-duplication is what keeps
    # each line at exactly one occurrence.
    assert out.count("hello") == 1
    assert out.count("world") == 1


def test_stream_logs_missing_gcloud_has_clear_error(monkeypatch) -> None:
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("gcloud")

    monkeypatch.setattr(vertex.subprocess, "run", raise_missing)

    backend = vertex.VertexJobBackend(
        custom_job_id="123", project_id="demo-project", region="us-central1"
    )
    with pytest.raises(RuntimeError, match="gcloud"):
        backend.status()


def test_stream_logs_failed_command_has_clear_error(monkeypatch) -> None:
    def raise_failed(command, **kwargs):
        raise subprocess.CalledProcessError(returncode=2, cmd=command)

    monkeypatch.setattr(vertex.subprocess, "run", raise_failed)

    backend = vertex.VertexJobBackend(
        custom_job_id="123", project_id="demo-project", region="us-central1"
    )
    with pytest.raises(RuntimeError, match="exit code 2"):
        backend.status()
