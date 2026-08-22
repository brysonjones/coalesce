# Baseten smoke tests

Launchable jobs that prove a coalesce Baseten deployment works: the GPU is
real, configs arrive, and the container can reach your GCS bucket. Every test
also runs on Vertex AI with `--provider vertex`, which is how you check the two
backends agree.

## One-time setup

**1. Baseten API key.** `export BASETEN_API_KEY=...` in your shell. coalesce
derives the `BASETEN_TRUSS_AUTH_*` variables truss needs from it, so there is
no `truss login` step.

**2. A GCP service account that can use the bucket.** A Baseten container has
no ambient GCP identity, so it needs a key of its own.

```bash
gcloud iam service-accounts create coalesce-baseten \
  --project my-project \
  --display-name "coalesce Baseten jobs"
```

```bash
gcloud storage buckets add-iam-policy-binding gs://my-bucket \
  --member "serviceAccount:coalesce-baseten@my-project.iam.gserviceaccount.com" \
  --role roles/storage.objectAdmin
```

If you want Baseten to run one of the private `my-project` images rather
than a public PyTorch one, the same account also needs to pull from Artifact
Registry:

```bash
gcloud artifacts repositories add-iam-policy-binding my-repo \
  --project my-project \
  --location us \
  --member "serviceAccount:coalesce-baseten@my-project.iam.gserviceaccount.com" \
  --role roles/artifactregistry.reader
```

```bash
gcloud iam service-accounts keys create ~/coalesce-baseten-key.json \
  --iam-account coalesce-baseten@my-project.iam.gserviceaccount.com
```

**3. Store the key as a Baseten secret.** It is referenced by name from then
on and never lands in a config file or a job definition.

```bash
python -m coalesce.providers.baseten ~/coalesce-baseten-key.json
```

At runtime coalesce injects it as `GCP_SERVICE_ACCOUNT_JSON`, and the task
runner turns it into Application Default Credentials before anything touches
GCS. On Vertex AI the variable is absent and the job's own service account is
used instead, so the same code path works on both.

## Checking the credential setup on its own

Before launching anything, confirm the key you uploaded can actually reach the
bucket. This walks the same path a Baseten container takes -- key in via
`GCP_SERVICE_ACCOUNT_JSON`, turned into ADC by the task runner, then a full
write/read/list/delete cycle -- and touches no Baseten API, so it tells you
which half of the setup is at fault when a job fails:

```bash
python examples/baseten/verify_gcs_credentials.py --keep
```

It clears any ambient ADC first and refuses to run unless the credentials are
a service account, so a pass cannot be your own gcloud login in disguise.

## Requirements on the Baseten side

Baseten Training has to be enabled for your workspace. If it is not, job
submission fails with:

```
403 PERMISSION_DENIED: You are not authorized for Baseten training.
```

`GET /v1/training/capacity` returning empty `gpu_capacities` is the same signal.
Everything else -- the API key, secrets, the GCS credential -- can be set up and
verified before that access lands.

## Running

Start with a dry run. It resolves the whole job and prints what would be
submitted without creating anything or writing to the bucket:

```bash
python examples/baseten/run_smoke.py all --dry-run
```

Then run for real:

```bash
python examples/baseten/run_smoke.py gpu --gpu H100 --stream-logs
```

| Test | What a pass proves |
|---|---|
| `gpu` | The job got a working CUDA device and ran a matmul on it. Raises rather than falling back to CPU. |
| `config` | A YAML config staged through GCS reached the remote function, and autograd runs on the accelerator. |
| `gcs` | The container can write, read, list and delete in your bucket — i.e. the credential plumbing is correct. |
| `mount` | A read-only `gs://` prefix was mirrored and mounted into the container. Needs `--dataset`. |
| `all` | `gpu`, `config` and `gcs` in sequence. |

The mount test needs a prefix with at least one object in it:

```bash
python examples/baseten/run_smoke.py mount --dataset gs://my-bucket/data
```

## Using a private image

Vertex AI pulls as the job's own service account, but Baseten pulls the image
itself and needs credentials. coalesce wires those up automatically for any
`*.pkg.dev` or `gcr.io` image, reusing the same secret:

```bash
python examples/baseten/run_smoke.py gpu --container-uri us-docker.pkg.dev/my-project/my-repo/my-image:latest
```

For a non-Google registry, pass `--container-uri` along with a
`container_registry_secret` in your own `launch_job` call.

## Comparing providers

The point of the abstraction is that this produces the same remote behaviour:

```bash
python examples/baseten/run_smoke.py all --provider baseten --gpu T4
```

```bash
python examples/baseten/run_smoke.py all --provider vertex --gpu T4
```

## Useful flags

- `--gpu H100 --gpu-count 2` — portable GPU names, translated per provider.
- `--gpu none` — CPU-only.
- `--cpu-count 16 --memory 128Gi` — Baseten sizing.
- `--machine-type` — Vertex sizing. Chosen from `--gpu` by default (`H100` -> `a3-highgpu-1g`, which also switches scheduling to FLEX_START because Vertex requires it there).
- `--spot` — interruptible capacity on either provider.
- `--no-wait` — submit and return; `--stream-logs` — tail the remote logs.
- `--container-uri` — override the provider's default image.
