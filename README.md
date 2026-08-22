# coalesce
A set of tools, scripts, and patterns to use various cloud provider compute resources, primarily for large ML workloads

Definition: ***coalescence**: in meteorology is the process where colliding water droplets in a cloud merge to form a single, larger droplet*

## Preface
The contents of this repo have not been extensively tested, and should be considered highly experimental

There are likely many edge cases, suboptimal patterns, etc.

The goal of these tools is to make it easier to launch and swap between different cloud provider offerings, specifically so I can train various ML workloads.

## Job launcher

`launch_job` takes any importable Python function and runs it on cloud compute. It handles packaging local code, uploading it to GCS, configuring the remote environment, and submitting the job.

Two providers are supported and they are interchangeable: `provider="vertex"` (aliases: `"gcp"`) runs on GCP Vertex AI, `provider="baseten"` runs on Baseten Training. Both stage through the same GCS bucket and run the same remote entry point, so switching between them is a one-word change. Set `COALESCE_PROVIDER` to change the default.

### What it does

1. **Package syncing** — Resolves local Python packages by name, zips them, uploads to GCS, and extracts them on the remote machine so they're importable. This is very useful when you're rapidly iterating on local code and want to be able to test on a cloud deployment without going through a release process.
2. **Config forwarding** — Accepts a config as a dict (serialized via env var) or a YAML/JSON file path (uploaded to GCS). The remote function receives the parsed config as its first argument.
3. **Job submission** — Wraps the Vertex AI `CustomJob` API or the Baseten Training API. Supports GPU/CPU selection, spot/on-demand scheduling, extra pip dependencies, and custom environment variables.
4. **Portable GPU names** — `gpu="H100"` becomes `NVIDIA_H100_80GB` on Vertex and `H100` on Baseten. A GPU a provider does not offer is an error, never a silent substitution.

### Usage

```python
from coalesce import launch_job

def train(config: dict):
    import torch
    print(config["learning_rate"])
    # ... training code ...

launch_job(
    func=train,
    project_id="my-gcp-project",
    bucket="gs://my-staging-bucket",
    config={"learning_rate": 0.001},          # or path to a YAML file
    machine_type="a2-highgpu-1g",
    gpu="A100_40GB",                          # portable; or accelerator_type= for Vertex
    boot_disk_size_gb=500,                    # larger local disk for dataset copies
    sync_packages=["my_local_lib"],           # local packages to ship to the job
    extra_packages=["transformers"],          # pip install on remote before run
    scheduling_strategy="STANDARD",           # STANDARD | SPOT | FLEX_START
)
```

The function must be importable (not defined in `__main__`). `sync_packages` names are resolved via `importlib`, so they must be installed or on `sys.path` locally.

`launch_job` returns a `Job` handle with `.status()`, `.wait()`, `.stream_logs()` and `.stop()`. The provider's native object is on `.raw`.

### Dry run

`dry_run=True` resolves the whole job, prints exactly what the provider would receive, and returns `None`. Nothing is submitted and nothing is uploaded, so it is safe to point at any bucket.

## Baseten

Baseten is a drop-in alternative to Vertex AI. Storage stays on GCS either way.

```python
launch_job(
    func=train,
    project_id="my-gcp-project",              # still the GCS project
    bucket="gs://my-staging-bucket",          # still your bucket
    provider="baseten",
    gpu="H100",
    gpu_count=2,
    cpu_count=16,
    memory="128Gi",
    sync_packages=["my_local_lib"],
    mount_datasets=["gs://my-bucket/datasets/imagenet:/mnt/data"],
    stream_logs=True,
)
```

### Setup

```bash
pip install 'coalesce[baseten] @ git+https://github.com/brysonjones/coalesce.git'
```

Export `BASETEN_API_KEY`. coalesce derives the `BASETEN_TRUSS_AUTH_*` variables truss needs from it, so there is no `truss login` step.

A Baseten container has no ambient GCP identity, so it needs a service account key to reach your bucket. Create one with `roles/storage.objectAdmin` on the bucket (plus `roles/artifactregistry.reader` if you want Baseten to pull a private image), then store it as a Baseten secret:

```bash
python -m coalesce.providers.baseten ~/coalesce-baseten-key.json
```

coalesce injects it as `GCP_SERVICE_ACCOUNT_JSON` and the task runner turns it into Application Default Credentials before anything touches GCS. On Vertex AI the variable is absent and the job's own service account is used, so the same code runs unchanged on both.

### Differences from Vertex AI

| Argument | Behaviour on Baseten |
|---|---|
| `machine_type` | Ignored; use `cpu_count` and `memory` instead. |
| `region` | Ignored; Baseten schedules across its own capacity. |
| `boot_disk_type`, `boot_disk_size_gb` | No equivalent. Setting them prints a notice. Ephemeral NVMe is at `$BT_SCRATCH_DIR`; use `checkpoint_volume_gb` for persistent storage at `$BT_CHECKPOINT_DIR`. |
| `scheduling_strategy="FLEX_START"` | Runs on dedicated capacity; Baseten queues for GPUs natively. |
| `accelerator_type` | Vertex-only spelling. Names with a Baseten equivalent are mapped; the rest are an error. Use `gpu` instead. |
| `container_uri` | Baseten pulls the image itself, so a private one needs credentials. `*.pkg.dev` and `gcr.io` images reuse the GCP credentials secret automatically; anything else needs `container_registry_secret`. Vertex AI pulls as the job's own service account. |
| `mount_datasets`, `checkpoint_volume_gb`, `priority`, `container_registry_secret`, `baseten_project` | Baseten-only. |

### Validating a deployment

`examples/baseten/` has launchable smoke tests that prove the GPU is real, configs arrive, and the container can read and write your bucket. Each one runs on either provider, so you can check the two agree:

```bash
python examples/baseten/run_smoke.py all --dry-run
```

See [examples/baseten/README.md](examples/baseten/README.md).

## Installation

```bash
pip install git+https://github.com/brysonjones/coalesce.git
```

For the Baseten provider:

```bash
pip install 'coalesce[baseten] @ git+https://github.com/brysonjones/coalesce.git'
```
