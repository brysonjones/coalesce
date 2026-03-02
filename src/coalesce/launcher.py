"""Launch Python functions on GCP Vertex AI."""

import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from google.cloud import aiplatform
from google.cloud import storage

from .packager import package_and_upload


def _upload_config_to_gcs(
    config_path: str | Path,
    bucket_name: str,
    project_id: str,
) -> str:
    """Upload a config file to GCS and return the URI."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_blob_name = f"configs/{config_path.stem}_{timestamp}{config_path.suffix}"

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name)
    blob.upload_from_filename(str(config_path))

    gcs_uri = f"gs://{bucket_name}/{gcs_blob_name}"
    print(f"  Uploaded config to: {gcs_uri}")
    return gcs_uri


def launch_job(
    func: Callable,
    project_id: str,
    bucket: str,
    region: str = "us-central1",
    container_uri: str = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest",
    machine_type: str = "n1-standard-4",
    accelerator_type: str | None = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    sync_packages: list[str] | None = None,
    job_name: str | None = None,
    sync: bool = True,
    config: str | Path | dict[str, Any] | None = None,
    extra_packages: list[str] | None = None,
    env: dict[str, str] | None = None,
    scheduling_strategy: str = "STANDARD",
    max_wait_duration: int = 86400,
) -> aiplatform.CustomJob:
    """
    Launch a Python function on GCP Vertex AI.

    Args:
        func: The Python function to run remotely. Must be importable.
        project_id: GCP project ID (e.g., "my-project")
        bucket: GCS bucket for staging (e.g., "gs://my-bucket" or "my-bucket")
        region: GCP region (default: "us-central1")
        container_uri: Docker image URI with required dependencies
        machine_type: Compute Engine machine type (default: "n1-standard-4")
        accelerator_type: GPU type (e.g., "NVIDIA_TESLA_T4", "NVIDIA_TESLA_A100")
                         Set to None for CPU-only jobs.
        accelerator_count: Number of GPUs (default: 1)
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
                              Required for a3-highgpu-1g/2g/4g machine types.
        max_wait_duration: Max wait time in seconds for FLEX_START scheduling (default: 86400 = 24h).

    Returns:
        The CustomJob object

    Example:
        def my_training_function(config: dict):
            import torch
            print(f"Learning rate: {config['learning_rate']}")
            # ... training code ...

        launch_job(
            func=my_training_function,
            project_id="my-project",
            bucket="gs://my-bucket",
            config="config.yaml",  # or {"learning_rate": 0.001}
            sync_packages=["my_local_package"],
            extra_packages=["transformers", "accelerate"],
        )
    """
    if sync_packages is None:
        sync_packages = []

    # Normalize bucket name
    bucket_normalized = bucket[5:] if bucket.startswith("gs://") else bucket

    # Generate job name if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if job_name is None:
        job_name = f"{func.__name__}_{timestamp}"

    print(f"Launching job: {job_name}")
    print(f"  Region: {region}")
    print(f"  Machine: {machine_type}")
    if accelerator_type:
        print(f"  GPU: {accelerator_type} x{accelerator_count}")
    else:
        print("  GPU: None (CPU-only)")

    # Initialize Vertex AI SDK
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f"gs://{bucket_normalized}",
    )

    # Determine module name - function must be imported from an actual module
    module_name = func.__module__
    if module_name == "__main__":
        raise ValueError(
            f"Function '{func.__name__}' has module '__main__' which cannot be imported remotely. "
            f"The function must be imported from an actual module, not defined in the script being run. "
            f"Move the function to a separate module and import it."
        )

    # Set up environment variables
    environment_variables = {
        "TASK_MODULE": module_name,
        "TASK_FUNCTION": func.__name__,
    }

    # Merge user-provided environment variables
    if env:
        environment_variables.update(env)
        print(f"  Env vars: {', '.join(env.keys())}")

    # Handle config parameter
    if config is not None:
        import json

        if isinstance(config, dict):
            # Serialize dict to JSON and pass via env var
            environment_variables["TASK_CONFIG_JSON"] = json.dumps(config)
            print(f"  Config: dict with {len(config)} keys")
        elif isinstance(config, (str, Path)):
            # Upload file to GCS
            config_path = Path(config)
            print(f"  Config: {config_path.name}")
            config_gcs_uri = _upload_config_to_gcs(
                config_path=config_path,
                bucket_name=bucket_normalized,
                project_id=project_id,
            )
            environment_variables["TASK_CONFIG_GCS_URI"] = config_gcs_uri
        else:
            raise TypeError(f"config must be dict, str, or Path, got {type(config)}")

    # Package and upload sync_packages if specified
    if sync_packages:
        print(f"Packaging {len(sync_packages)} package(s) for sync...")
        gcs_uri = package_and_upload(
            package_names=sync_packages,
            bucket_name=bucket_normalized,
            project_id=project_id,
        )
        environment_variables["SYNC_PACKAGES_GCS_URI"] = gcs_uri

    # Copy task.py to a temp directory for upload
    task_py_source = Path(__file__).parent / "task.py"
    temp_dir = tempfile.mkdtemp()
    task_py_dest = Path(temp_dir) / "task.py"
    shutil.copy2(task_py_source, task_py_dest)

    # Build job kwargs
    job_kwargs = {
        "display_name": job_name,
        "script_path": str(task_py_dest),
        "container_uri": container_uri,
        "machine_type": machine_type,
        "environment_variables": environment_variables,
    }

    # Add extra packages to pip install
    if extra_packages:
        job_kwargs["requirements"] = extra_packages
        print(f"  Extra packages: {', '.join(extra_packages)}")

    # Add GPU configuration if specified
    if accelerator_type:
        job_kwargs["accelerator_type"] = accelerator_type
        job_kwargs["accelerator_count"] = accelerator_count

    # Create and run the job
    job = aiplatform.CustomJob.from_local_script(**job_kwargs)

    # Configure scheduling strategy
    from google.cloud.aiplatform_v1.types import custom_job as gca_custom_job_compat

    run_kwargs: dict[str, Any] = {"sync": sync}
    strategy = scheduling_strategy.upper()
    if strategy == "SPOT":
        print("  Scheduling: SPOT (preemptible, may be interrupted)")
        run_kwargs["scheduling_strategy"] = gca_custom_job_compat.Scheduling.Strategy.SPOT
        run_kwargs["restart_job_on_worker_restart"] = True
    elif strategy in ("FLEX_START", "DWS"):
        print(f"  Scheduling: FLEX_START (will queue up to {max_wait_duration}s)")
        run_kwargs["scheduling_strategy"] = gca_custom_job_compat.Scheduling.Strategy.FLEX_START
        run_kwargs["max_wait_duration"] = max_wait_duration
    else:
        print("  Scheduling: STANDARD (on-demand)")

    print(f"Submitting job...")
    job.run(**run_kwargs)

    if sync:
        print(f"Job completed: {job_name}")
    else:
        print(f"Job submitted: {job_name}")

    return job
