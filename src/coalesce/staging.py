"""Turning a :class:`~coalesce.spec.JobSpec` into remote-runner inputs.

Both providers stage the same way: local packages and config files go to GCS,
and the remote ``task.py`` is pointed at them through environment variables.
Keeping this in one place is what makes a job behave identically on Vertex and
on Baseten.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from . import storage
from .packager import package_and_upload
from .spec import JobSpec


def staging_paths(spec: JobSpec) -> tuple[str, str, str]:
    """Return ``(bucket_name, staging_root_prefix, staging_bucket_uri)``."""
    bucket_name, bucket_prefix = storage.normalize_bucket_and_prefix(spec.bucket)
    staging_root_prefix = storage.join_parts(bucket_prefix, spec.staging_prefix)
    staging_bucket_uri = (
        f"gs://{bucket_name}/{staging_root_prefix}"
        if staging_root_prefix
        else f"gs://{bucket_name}"
    )
    return bucket_name, staging_root_prefix, staging_bucket_uri


def build_environment(spec: JobSpec) -> dict[str, str]:
    """Upload the job's inputs to GCS and describe them as environment variables.

    This is the side-effecting half of a launch: by the time it returns, the
    config file and any synced packages are in the bucket and the returned
    mapping tells ``task.py`` where to find them.
    """
    bucket_name, staging_root_prefix, _ = staging_paths(spec)

    environment_variables = {
        "TASK_MODULE": spec.module_name(),
        "TASK_FUNCTION": spec.func.__name__,
    }

    if spec.env:
        environment_variables.update(spec.env)
        print(f"  Env vars: {', '.join(spec.env)}")

    if spec.config is not None:
        if isinstance(spec.config, dict):
            environment_variables["TASK_CONFIG_JSON"] = json.dumps(spec.config)
            print(f"  Config: dict with {len(spec.config)} keys")
        elif isinstance(spec.config, (str, Path)):
            config_path = Path(spec.config)
            print(f"  Config: {config_path.name}")
            environment_variables["TASK_CONFIG_GCS_URI"] = storage.upload_config(
                config_path=config_path,
                bucket_name=bucket_name,
                project_id=spec.gcp_project_id,
                prefix=storage.join_parts(staging_root_prefix, "configs"),
            )
        else:
            raise TypeError(
                f"config must be dict, str, or Path, got {type(spec.config)}"
            )

    if spec.sync_packages:
        print(f"Packaging {len(spec.sync_packages)} package(s) for sync...")
        environment_variables["SYNC_PACKAGES_GCS_URI"] = package_and_upload(
            package_names=spec.sync_packages,
            bucket_name=bucket_name,
            project_id=spec.gcp_project_id,
            prefix=storage.join_parts(staging_root_prefix, "source"),
        )

    return environment_variables


def stage_task_runner(dest_dir: str | Path | None = None) -> Path:
    """Copy ``task.py`` somewhere a provider can upload it from."""
    dest_dir = Path(dest_dir) if dest_dir is not None else Path(tempfile.mkdtemp())
    dest_dir.mkdir(parents=True, exist_ok=True)
    task_py_dest = dest_dir / "task.py"
    shutil.copy2(Path(__file__).parent / "task.py", task_py_dest)
    return task_py_dest

