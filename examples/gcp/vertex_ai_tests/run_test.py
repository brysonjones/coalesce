#!/usr/bin/env python
"""Launch a test job on Vertex AI to validate coalesce setup."""

import sys
from pathlib import Path

# Add the test directory to the path so test_task can be imported
sys.path.insert(0, str(Path(__file__).parent))

from coalesce import launch_job
from test_task import run_pytorch_test


def main():
    """Launch a simple PyTorch test job on Vertex AI."""
    print("Launching PyTorch test job on Vertex AI...")

    job = launch_job(
        func=run_pytorch_test,
        project_id="adjoint-app",
        bucket="gs://adjoint_exp_usc1",
        region="us-central1",
        # Use the existing container that has PyTorch installed
        container_uri="us-docker.pkg.dev/adjoint-app/experiment-images/representation-learning-research:v1",
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        # Sync the test_task module so it's available on the remote machine
        sync_packages=["test_task"],
        sync=True,  # Wait for completion
    )

    print(f"\nJob completed!")
    print(f"View logs at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")


if __name__ == "__main__":
    main()
