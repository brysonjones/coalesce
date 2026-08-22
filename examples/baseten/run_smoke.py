#!/usr/bin/env python
"""Launch the coalesce smoke tests and report what passed.

Every test can run on either provider, so the same command validates a Baseten
deployment and proves it behaves like the Vertex AI one:

    ./run_smoke.py gpu --provider baseten
    ./run_smoke.py gpu --provider vertex

Start with --dry-run to see exactly what would be submitted without creating
anything or writing to the bucket.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# smoke_tasks has to be importable by name, both here and on the remote side.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import smoke_tasks  # noqa: E402

from coalesce import launch_job  # noqa: E402

DEFAULT_PROJECT_ID = os.environ.get("COALESCE_PROJECT_ID", "my-project")
DEFAULT_BUCKET = os.environ.get("COALESCE_BUCKET", "gs://my-bucket")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "test",
        choices=["gpu", "config", "gcs", "mount", "all"],
        help=(
            "gpu: CUDA is present and usable. "
            "config: a YAML config reaches the remote function. "
            "gcs: the job can read and write the bucket. "
            "mount: a read-only gs:// path is mounted into the container. "
            "all: every test above except mount."
        ),
    )
    parser.add_argument("--provider", default="baseten", help="baseten or vertex (default: baseten)")
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID, help="GCP project owning the bucket")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket used for staging")

    hardware = parser.add_argument_group("hardware")
    hardware.add_argument("--gpu", default="T4", help="Portable GPU name, or 'none' for CPU-only (default: T4)")
    hardware.add_argument("--gpu-count", type=int, default=1)
    hardware.add_argument("--cpu-count", type=int, default=4, help="Baseten only")
    hardware.add_argument("--memory", default="16Gi", help="Baseten only")
    hardware.add_argument("--machine-type", default="n1-standard-4", help="Vertex AI only")
    hardware.add_argument("--region", default="us-central1", help="Vertex AI only")
    hardware.add_argument("--container-uri", default=None, help="Override the provider default image")
    hardware.add_argument("--spot", action="store_true", help="Request interruptible capacity")

    mount = parser.add_argument_group("mount test")
    mount.add_argument("--dataset", default=None, help="gs:// prefix to mount, e.g. gs://bucket/datasets/demo")
    mount.add_argument("--mount-path", default="/mnt/dataset", help="Where to mount it in the container")

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--dry-run", action="store_true", help="Print the plan, submit nothing")
    behaviour.add_argument("--no-wait", action="store_true", help="Return as soon as the job is submitted")
    behaviour.add_argument("--stream-logs", action="store_true", help="Tail the remote logs locally")
    behaviour.add_argument("--baseten-project", default="coalesce-smoke", help="Baseten project to group jobs under")
    behaviour.add_argument(
        "--credentials-secret",
        default="gcp_service_account_json",
        help="Baseten secret holding the GCP service account key",
    )
    return parser


def common_kwargs(args: argparse.Namespace) -> dict:
    return {
        "project_id": args.project_id,
        "bucket": args.bucket,
        "provider": args.provider,
        "gpu": None if args.gpu.lower() == "none" else args.gpu,
        "accelerator_type": None,
        "gpu_count": args.gpu_count,
        "cpu_count": args.cpu_count,
        "memory": args.memory,
        "machine_type": args.machine_type,
        "region": args.region,
        "container_uri": args.container_uri,
        "scheduling_strategy": "SPOT" if args.spot else "STANDARD",
        "sync_packages": ["smoke_tasks"],
        "baseten_project": args.baseten_project,
        "gcp_credentials_secret": args.credentials_secret,
        "sync": not args.no_wait,
        "stream_logs": args.stream_logs,
        "dry_run": args.dry_run,
    }


def run_gpu(args: argparse.Namespace):
    """Prove the job landed on a working GPU."""
    return launch_job(func=smoke_tasks.gpu_check, **common_kwargs(args))


def run_config(args: argparse.Namespace):
    """Prove a YAML config staged through GCS reaches the remote function."""
    return launch_job(
        func=smoke_tasks.training_step,
        config=HERE / "config.yaml",
        **common_kwargs(args),
    )


def run_gcs(args: argparse.Namespace):
    """Prove the job can read and write the GCS bucket."""
    return launch_job(
        func=smoke_tasks.gcs_roundtrip,
        config={"bucket": args.bucket, "prefix": ".coalesce/smoke"},
        **common_kwargs(args),
    )


def run_mount(args: argparse.Namespace):
    """Prove a read-only gs:// path is mounted into the container."""
    if not args.dataset:
        raise SystemExit(
            "The mount test needs --dataset gs://bucket/prefix pointing at a "
            "prefix with at least one object in it."
        )
    kwargs = common_kwargs(args)
    kwargs["mount_datasets"] = [f"{args.dataset}:{args.mount_path}"]
    return launch_job(
        func=smoke_tasks.mounted_dataset_check,
        config={"mount_path": args.mount_path},
        **kwargs,
    )


TESTS = {"gpu": run_gpu, "config": run_config, "gcs": run_gcs, "mount": run_mount}


def main() -> int:
    args = build_parser().parse_args()
    # 'all' deliberately omits mount, which needs a dataset that exists.
    selected = ["gpu", "config", "gcs"] if args.test == "all" else [args.test]

    results = {}
    for name in selected:
        print("=" * 70)
        print(f"smoke test: {name}  (provider={args.provider})")
        print("=" * 70)
        try:
            TESTS[name](args)
        except Exception as exc:  # noqa: BLE001 - the summary is the point
            results[name] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"\n{name}: FAILED -- {type(exc).__name__}: {exc}\n")
        else:
            results[name] = "passed"
        print()

    print("=" * 70)
    print("summary")
    print("=" * 70)
    for name, outcome in results.items():
        print(f"  {name}: {outcome}")

    return 0 if all(outcome == "passed" for outcome in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
