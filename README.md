# coalesce
A set of tools, scripts, and patterns to use various cloud provider compute resources, primarily for large ML workloads

Definition: ***coalescence**: in meteorology is the process where colliding water droplets in a cloud merge to form a single, larger droplet*

## Preface
The contents of this repo have not been extensively tested, and should be considered highly experimental

There are likely many edge cases, suboptimal patterns, etc.

The goal of these tools is to make it easier to launch and swap between different cloud provider offerings, specifically so I can train various ML workloads.

## GCP 

### Vertex AI Job Launcher

`launch_job` is a tool that takes any importable Python function and runs it on a GCP Vertex AI instance. It handles packaging local code, uploading it to GCS, configuring the remote environment, and submitting the job.

### What it does

1. **Package syncing** — Resolves local Python packages by name, zips them, uploads to GCS, and extracts them on the remote machine so they're importable. This is very useful when you're rapidly iterating on local code and want to be able to test on a cloud deployment without going through a release process.
2. **Config forwarding** — Accepts a config as a dict (serialized via env var) or a YAML/JSON file path (uploaded to GCS). The remote function receives the parsed config as its first argument.
3. **Job submission** — Wraps the Vertex AI `CustomJob` API. Supports GPU/CPU selection, spot/on-demand/flex scheduling, extra pip dependencies, and custom environment variables.

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
    accelerator_type="NVIDIA_TESLA_A100",
    sync_packages=["my_local_lib"],           # local packages to ship to the job
    extra_packages=["transformers"],          # pip install on remote before run
    scheduling_strategy="STANDARD",           # STANDARD | SPOT | FLEX_START
)
```

The function must be importable (not defined in `__main__`). `sync_packages` names are resolved via `importlib`, so they must be installed or on `sys.path` locally.

### Installation

```bash
pip install git+https://github.com/brysonjones/coalesce.git
```