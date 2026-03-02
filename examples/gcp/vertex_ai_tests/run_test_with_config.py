#!/usr/bin/env python
"""Launch a test job on Vertex AI with YAML config."""

import sys
from pathlib import Path

# Add the test directory to the path so test modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

from coalesce import launch_job
from test_task_with_config import run_pytorch_test_with_config


def main():
    """Launch a PyTorch test job with YAML config on Vertex AI."""
    print("Launching PyTorch test job with YAML config...")

    # Path to the config file
    config_path = Path(__file__).parent / "config.yaml"

    job = launch_job(
        func=run_pytorch_test_with_config,
        project_id="adjoint-app",
        bucket="gs://adjoint_exp_usc1",
        region="us-central1",
        container_uri="us-docker.pkg.dev/adjoint-app/experiment-images/representation-learning-research:v1",
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        # Sync the test module so it's available on the remote machine
        sync_packages=["test_task_with_config"],
        # Pass the YAML config file
        config=config_path,
        sync=True,
    )

    print(f"\nJob completed!")
    print(f"View logs at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")


if __name__ == "__main__":
    main()
