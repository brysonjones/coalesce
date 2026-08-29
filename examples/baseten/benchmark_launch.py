#!/usr/bin/env python
"""Measure how long each GPU type takes to go from submit to running code.

Launch latency is the number that decides whether a provider is usable for
interactive work, and it is not the same as the time the job takes to run. This
submits the same trivial GPU check to every requested accelerator at once and
reports where the wall-clock actually goes:

    ./benchmark_launch.py --provider baseten
    ./benchmark_launch.py --provider baseten --gpus A10G,H100 --gpu-count 2
    ./benchmark_launch.py --provider vertex --gpus T4,L4 --spot

Every job is submitted in parallel, so the total runtime is the slowest single
GPU rather than the sum. Accelerators the provider refuses are reported as
rejected with its own message instead of failing the run, which is the cheapest
way to discover what a provider will actually schedule.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import smoke_tasks  # noqa: E402

from coalesce import launch_job  # noqa: E402

DEFAULT_PROJECT_ID = os.environ.get("COALESCE_PROJECT_ID")
DEFAULT_BUCKET = os.environ.get("COALESCE_BUCKET")

# Printed by smoke_tasks.gpu_check as its first line of real work, so the first
# log line containing it marks the moment the container stopped installing
# dependencies and started running our code.
CODE_START_MARKER = "torch:"

# Reported separately from the raw status string so both providers line up in
# one table.
RUNNING_STATES = ("RUNNING",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", default="baseten", help="baseten or vertex")
    parser.add_argument(
        "--gpus",
        default="A10G,L4,A100,H100,H200,B200",
        help="Comma-separated portable GPU names to benchmark",
    )
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--cpu-count", type=int, default=4, help="Baseten only")
    parser.add_argument("--memory", default="16Gi", help="Baseten only")
    parser.add_argument("--region", default="us-central1", help="Vertex only")
    parser.add_argument("--spot", action="store_true", help="Request interruptible capacity")
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Give up on a job that has not finished within this many seconds",
    )
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--baseten-project", default="coalesce-bench")
    return parser.parse_args()


class Result:
    """One GPU's timeline, filled in as the job progresses."""

    def __init__(self, gpu: str) -> None:
        self.gpu = gpu
        self.job = None
        self.rejected: str | None = None
        self.error: str | None = None
        self.submitted_at: float | None = None
        self.submitted_wall: float | None = None
        self.transitions: list[tuple[str, float]] = []
        self.seconds_to_code: float | None = None
        self.final_state: str | None = None

    def first_time_in(self, *needles: str) -> float | None:
        """When the job first reached a state whose name contains any needle."""
        for state, at in self.transitions:
            if any(n in state.upper() for n in needles):
                return at
        return None

    def elapsed_to(self, at: float | None) -> float | None:
        if at is None or self.submitted_at is None:
            return None
        return at - self.submitted_at


def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def run_one(args: argparse.Namespace, gpu: str, log_lock: threading.Lock) -> Result:
    """Submit one job and follow it to a terminal state, timing each step."""
    result = Result(gpu)
    started = time.monotonic()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    machine_type = "n1-standard-4"
    if args.provider == "vertex":
        from run_smoke import vertex_machine_type

        machine_type = vertex_machine_type(gpu, args.gpu_count)

    try:
        result.job = launch_job(
            func=smoke_tasks.gpu_check,
            job_name=f"bench_{gpu.lower()}x{args.gpu_count}_{stamp}",
            project_id=args.project_id,
            bucket=args.bucket,
            provider=args.provider,
            gpu=gpu,
            gpu_count=args.gpu_count,
            cpu_count=args.cpu_count,
            memory=args.memory,
            machine_type=machine_type,
            region=args.region,
            scheduling_strategy="SPOT" if args.spot else "STANDARD",
            sync_packages=["smoke_tasks"],
            baseten_project=args.baseten_project,
            sync=False,
        )
    except Exception as exc:  # noqa: BLE001 - the provider's refusal is the result
        result.rejected = str(exc).strip().splitlines()[0][:160]
        with log_lock:
            print(f"  {gpu:<8} rejected at submit: {result.rejected}")
        return result

    result.submitted_at = time.monotonic()
    result.submitted_wall = time.time()
    with log_lock:
        print(f"  {gpu:<8} submitted in {_fmt(result.submitted_at - started)} -> {result.job.id}")

    deadline = result.submitted_at + args.timeout
    seen: str | None = None
    while True:
        try:
            state = result.job.status()
        except Exception as exc:  # noqa: BLE001 - a transient poll failure is not fatal
            with log_lock:
                print(f"  {gpu:<8} poll error: {exc}")
            time.sleep(args.poll_interval)
            continue

        if state != seen:
            now = time.monotonic()
            result.transitions.append((state, now))
            seen = state
            with log_lock:
                print(f"  {gpu:<8} {_fmt(now - result.submitted_at):>7} {state}")

        if result.job._backend.is_terminal(state):
            result.final_state = state
            break
        if time.monotonic() >= deadline:
            result.final_state = f"{state} (timed out after {_fmt(args.timeout)})"
            break
        time.sleep(args.poll_interval)

    result.seconds_to_code = _seconds_to_code_start(result)
    return result


def _seconds_to_code_start(result: Result) -> float | None:
    """How long after submit our own code printed its first line.

    The gap between the job reporting RUNNING and this is container setup we pay
    for on every launch -- dependency installs, mostly -- so it is worth showing
    separately from the provider's own scheduling latency. Log timestamps are
    absolute, so they are compared against the absolute submit time rather than
    the monotonic clock the rest of the timeline uses.
    """
    if result.job is None or result.submitted_wall is None:
        return None
    try:
        entries = result.job._backend.fetch_logs(None, None)
    except Exception:  # noqa: BLE001 - logs are a nicety, not the measurement
        return None

    stamped = [
        e for e in entries if e.timestamp_ns and CODE_START_MARKER in (e.message or "")
    ]
    if not stamped:
        return None
    first = min(e.timestamp_ns for e in stamped)
    return first / 1e9 - result.submitted_wall


def report(args: argparse.Namespace, results: list[Result]) -> None:
    print("\n" + "=" * 78)
    print(f"launch benchmark: {args.provider}, {args.gpu_count} GPU(s) per job"
          + (", spot" if args.spot else ""))
    print("=" * 78)
    header = f"{'GPU':<10} {'submit':>8} {'running':>9} {'code':>8} {'total':>8}  outcome"
    print(header)
    print("-" * 78)

    for r in sorted(results, key=lambda r: r.gpu):
        if r.rejected:
            print(f"{r.gpu:<10} {'--':>8} {'--':>9} {'--':>8} {'--':>8}  rejected: {r.rejected}")
            continue
        running = r.elapsed_to(r.first_time_in(*RUNNING_STATES))
        code = r.seconds_to_code
        total = r.elapsed_to(r.transitions[-1][1]) if r.transitions else None
        print(
            f"{r.gpu:<10} {_fmt(0):>8} {_fmt(running):>9} {_fmt(code):>8} "
            f"{_fmt(total):>8}  {r.final_state}"
        )

    print("-" * 78)
    print("submit  = API accepted the job (measured separately, see above)")
    print("running = provider reported the job running, i.e. capacity was found")
    print("code    = our own code printed its first line, after dependency install")
    print("total   = submit to terminal state")


def main() -> int:
    args = parse_args()
    if not args.project_id or not args.bucket:
        raise SystemExit(
            "Set COALESCE_PROJECT_ID and COALESCE_BUCKET, or pass --project-id "
            "and --bucket."
        )

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    print(f"Benchmarking {len(gpus)} GPU type(s) on {args.provider}, all in parallel:")
    print(f"  {', '.join(gpus)}\n")

    log_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results = list(pool.map(lambda g: run_one(args, g, log_lock), gpus))

    report(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
