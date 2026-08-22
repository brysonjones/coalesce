#!/usr/bin/env python
"""Launch a test job on Vertex AI to validate coalesce setup."""

import os
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
        project_id=os.environ["COALESCE_PROJECT_ID"],
        bucket=os.environ["COALESCE_BUCKET"],
        region="us-central1",
        # Any image with PyTorch; defaults to a Vertex AI prebuilt one
        container_uri=os.environ.get(
            "COALESCE_CONTAINER_URI",
            "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest",
        ),
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
