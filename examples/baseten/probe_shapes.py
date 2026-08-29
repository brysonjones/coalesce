#!/usr/bin/env python
"""Find which CPU/memory shape a Baseten GPU type will actually accept.

Baseten training rejects a job whose vCPU and memory do not match a real
instance for the requested accelerator, and the error names the spec it could
not satisfy rather than the ones it could. This walks a list of candidate shapes
until one is accepted, which is the only way to discover them from outside:

    ./probe_shapes.py --gpus H200,B200

Rejections cost nothing and return immediately, so the sweep is cheap. The first
shape that IS accepted launches a real job, so shapes are tried one at a time
and the sweep stops at the first success for each GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import smoke_tasks  # noqa: E402

from coalesce import launch_job  # noqa: E402

# Ordered smallest first so a success costs as little as possible.
CANDIDATE_SHAPES = [
    (4, "16Gi"),
    (8, "64Gi"),
    (12, "144Gi"),
    (16, "128Gi"),
    (24, "200Gi"),
    (26, "234Gi"),
    (32, "256Gi"),
    (48, "384Gi"),
    (64, "512Gi"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpus", default="H200,B200")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--project-id", default=os.environ.get("COALESCE_PROJECT_ID"))
    parser.add_argument("--bucket", default=os.environ.get("COALESCE_BUCKET"))
    parser.add_argument("--baseten-project", default="coalesce-probe")
    parser.add_argument(
        "--stop-at-first",
        action="store_true",
        default=True,
        help="Stop probing a GPU once a shape is accepted (default)",
    )
    return parser.parse_args()


def probe(args: argparse.Namespace, gpu: str) -> None:
    print(f"\n=== {gpu} x{args.gpu_count} ===")
    for cpu, memory in CANDIDATE_SHAPES:
        stamp = datetime.now().strftime("%H%M%S")
        label = f"{cpu} vCPU / {memory}"
        try:
            job = launch_job(
                func=smoke_tasks.gpu_check,
                job_name=f"probe_{gpu.lower()}_{cpu}c_{stamp}",
                project_id=args.project_id,
                bucket=args.bucket,
                provider="baseten",
                gpu=gpu,
                gpu_count=args.gpu_count,
                cpu_count=cpu,
                memory=memory,
                sync_packages=["smoke_tasks"],
                baseten_project=args.baseten_project,
                sync=False,
            )
        except Exception as exc:  # noqa: BLE001 - the refusal is the measurement
            reason = str(exc).strip().splitlines()[0]
            if "requires an instance with the spec" in reason or "Bad Request" in reason:
                print(f"  {label:<22} rejected")
            else:
                print(f"  {label:<22} rejected: {reason[:110]}")
            continue

        print(f"  {label:<22} ACCEPTED -> job {job.id}")
        print(f"    console: {job.console_url}")
        if args.stop_at_first:
            return
    print(f"  no candidate shape was accepted for {gpu}")


def main() -> int:
    args = parse_args()
    if not args.project_id or not args.bucket:
        raise SystemExit("Set COALESCE_PROJECT_ID and COALESCE_BUCKET.")
    for gpu in [g.strip() for g in args.gpus.split(",") if g.strip()]:
        probe(args, gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
