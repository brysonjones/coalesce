"""A dry run must describe a launch without performing any part of it."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from coalesce import launcher, staging, storage
from coalesce.providers import baseten, vertex


def sample_task() -> None:
    pass


@pytest.fixture(autouse=True)
def no_side_effects(monkeypatch):
    """Make every outbound call explode, so a dry run that leaks is caught."""
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    for module, name in [
        (storage, "upload_config"),
        (staging, "package_and_upload"),
    ]:
        monkeypatch.setattr(module, name, Mock(side_effect=AssertionError(f"{name} called")))
    monkeypatch.setattr(vertex.aiplatform, "init", Mock(side_effect=AssertionError("init called")))
    monkeypatch.setattr(
        vertex.aiplatform.CustomJob,
        "from_local_script",
        Mock(side_effect=AssertionError("from_local_script called")),
    )
    monkeypatch.setattr("truss_train.push", Mock(side_effect=AssertionError("push called")))
    monkeypatch.setattr(
        baseten.requests, "request", Mock(side_effect=AssertionError("requests called"))
    )


@pytest.mark.parametrize("provider", ["vertex", "baseten"])
def test_dry_run_uploads_nothing_and_submits_nothing(provider, tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("learning_rate: 0.01\n")

    result = launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider=provider,
        config=config,
        sync_packages=["demo_package"],
        dry_run=True,
    )

    assert result is None


@pytest.mark.parametrize("provider", ["vertex", "baseten"])
def test_dry_run_shows_where_staged_files_would_land(provider, tmp_path, capsys) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("learning_rate: 0.01\n")

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider=provider,
        config=config,
        sync_packages=["demo_package"],
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "would upload config.yaml to gs://demo-bucket/.coalesce/tmp/configs/" in out
    assert "would upload demo_package to gs://demo-bucket/.coalesce/tmp/source/" in out
    assert "nothing was submitted and nothing was uploaded" in out


def test_dry_run_needs_no_baseten_credentials(monkeypatch, capsys) -> None:
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)

    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        gpu="H100",
        dry_run=True,
    )

    assert "'accelerator': 'H100'" in capsys.readouterr().out


def test_dry_run_masks_the_credentials_secret(capsys) -> None:
    launcher.launch_job(
        func=sample_task,
        project_id="demo-project",
        bucket="gs://demo-bucket",
        provider="baseten",
        dry_run=True,
    )

    out = capsys.readouterr().out
    assert "GCP_SERVICE_ACCOUNT_JSON=<baseten secret 'gcp_service_account_json'>" in out
