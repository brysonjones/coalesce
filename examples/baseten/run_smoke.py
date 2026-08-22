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

# Vertex AI only accepts a GPU on a machine type built for it, so picking one
# is not optional the way it is on Baseten. Mirrors the profile table in
# common Vertex GPU and machine-type pairings.
VERTEX_MACHINE_TYPES = {
    ("T4", 1): "n1-standard-4",
    ("T4", 2): "n1-standard-8",
    ("T4", 4): "n1-standard-16",
    ("L4", 1): "g2-standard-4",
    ("L4", 2): "g2-standard-24",
    ("L4", 4): "g2-standard-48",
    ("L4", 8): "g2-standard-96",
    ("A100_40GB", 1): "a2-highgpu-1g",
    ("A100_40GB", 2): "a2-highgpu-2g",
    ("A100_40GB", 4): "a2-highgpu-4g",
    ("A100_40GB", 8): "a2-highgpu-8g",
    ("A100", 1): "a2-ultragpu-1g",
    ("A100", 2): "a2-ultragpu-2g",
    ("A100", 4): "a2-ultragpu-4g",
    ("A100", 8): "a2-ultragpu-8g",
    ("H100", 1): "a3-highgpu-1g",
    ("H100", 2): "a3-highgpu-2g",
    ("H100", 4): "a3-highgpu-4g",
    ("H100", 8): "a3-highgpu-8g",
}


def vertex_machine_type(gpu: str | None, count: int) -> str:
    """The machine type Vertex needs for this GPU, or a CPU-only default."""
    from coalesce.spec import canonical_gpu

    if gpu is None:
        return "n1-standard-4"
    key = (canonical_gpu(gpu), count)
    if key not in VERTEX_MACHINE_TYPES:
        raise SystemExit(
            f"No Vertex machine type known for {gpu} x{count}. Pass "
            f"--machine-type explicitly, or pick one of: "
            + ", ".join(f"{g} x{c}" for g, c in sorted(VERTEX_MACHINE_TYPES))
        )
    return VERTEX_MACHINE_TYPES[key]


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
    hardware.add_argument(
        "--machine-type",
        default=None,
        help="Vertex AI only; chosen from the GPU by default",
    )
    hardware.add_argument("--region", default="us-central1", help="Vertex AI only")
    hardware.add_argument("--container-uri", default=None, help="Override the provider default image")
    hardware.add_argument("--spot", action="store_true", help="Request interruptible capacity")
    hardware.add_argument(
        "--max-wait",
        type=int,
        default=86400,
        help=(
            "Seconds a FLEX_START job may queue for capacity before expiring "
            "(default: 24h). Lower it to a few minutes to probe availability "
            "rather than wait for it."
        ),
    )

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
    gpu = None if args.gpu.lower() == "none" else args.gpu
    machine_type = args.machine_type or vertex_machine_type(gpu, args.gpu_count)

    scheduling = "SPOT" if args.spot else "STANDARD"
    if not args.spot and machine_type.startswith("a3-highgpu"):
        # Vertex requires FLEX_START for a3-highgpu, so a plain STANDARD
        # request for an H100 is rejected outright.
        scheduling = "FLEX_START"
        if args.provider != "baseten":
            print(f"Using FLEX_START scheduling, which {machine_type} requires.\n")

    # Auto-generated names are second-resolution, so a parallel sweep produces
    # several jobs with the same name. Tag each with what it is actually testing.
    tag = f"{gpu.lower()}x{args.gpu_count}" if gpu else "cpu"
    if args.spot:
        tag += "-spot"

    return {
        "job_name_suffix": tag,
        "project_id": args.project_id,
        "bucket": args.bucket,
        "provider": args.provider,
        "gpu": gpu,
        "accelerator_type": None,
        "gpu_count": args.gpu_count,
        "cpu_count": args.cpu_count,
        "memory": args.memory,
        "machine_type": machine_type,
        "region": args.region,
        "container_uri": args.container_uri,
        "scheduling_strategy": scheduling,
        "max_wait_duration": args.max_wait,
        "sync_packages": ["smoke_tasks"],
        "baseten_project": args.baseten_project,
        "gcp_credentials_secret": args.credentials_secret,
        "sync": not args.no_wait,
        "stream_logs": args.stream_logs,
        "dry_run": args.dry_run,
    }


def _with_job_name(func, kwargs: dict) -> dict:
    """Give the job a name that says what it is testing."""
    from datetime import datetime

    kwargs = dict(kwargs)
    suffix = kwargs.pop("job_name_suffix", None)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kwargs["job_name"] = f"{func.__name__}_{suffix}_{stamp}" if suffix else None
    return kwargs


def run_gpu(args: argparse.Namespace):
    """Prove the job landed on a working GPU."""
    func = smoke_tasks.gpu_check
    return launch_job(func=func, **_with_job_name(func, common_kwargs(args)))


def run_config(args: argparse.Namespace):
    """Prove a YAML config staged through GCS reaches the remote function."""
    func = smoke_tasks.training_step
    return launch_job(
        func=func,
        config=HERE / "config.yaml",
        **_with_job_name(func, common_kwargs(args)),
    )


def run_gcs(args: argparse.Namespace):
    """Prove the job can read and write the GCS bucket."""
    func = smoke_tasks.gcs_roundtrip
    return launch_job(
        func=func,
        config={"bucket": args.bucket, "prefix": ".coalesce/smoke"},
        **_with_job_name(func, common_kwargs(args)),
    )


def run_mount(args: argparse.Namespace):
    """Prove a read-only gs:// path is mounted into the container."""
    if not args.dataset:
        raise SystemExit(
            "The mount test needs --dataset gs://bucket/prefix pointing at a "
            "prefix with at least one object in it."
        )
    func = smoke_tasks.mounted_dataset_check
    kwargs = _with_job_name(func, common_kwargs(args))
    kwargs["mount_datasets"] = [f"{args.dataset}:{args.mount_path}"]
    return launch_job(
        func=func,
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
