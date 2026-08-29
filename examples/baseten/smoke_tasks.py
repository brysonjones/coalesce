"""Remote functions that prove a coalesce deployment actually works.

Each one exercises a single capability and prints what it observed, so a
failure in the job log tells you which piece broke rather than just that the
job died. They are deliberately provider-agnostic: the same functions run on
Baseten and on Vertex AI, which is how you check the two backends agree.
"""


def _describe_host():
    """Print enough about the machine to tell the providers apart in a log."""
    import os
    import platform

    print(f"Host: {platform.node()} ({platform.platform()})")
    print(f"Python: {platform.python_version()}")

    # Baseten exports BT_*; Vertex AI exports CLOUD_ML_*.
    markers = {k: v for k, v in os.environ.items() if k.startswith(("BT_", "CLOUD_ML_"))}
    if markers:
        print("Runtime markers:")
        for key in sorted(markers):
            print(f"  {key}={markers[key]}")


def gpu_check():
    """Prove the job landed on a working GPU.

    Takes no config, which also exercises the no-argument path through the
    task runner. Raises if CUDA is missing, because a silent fall back to CPU
    would make a broken GPU request look like a passing test.
    """
    import torch

    _describe_host()
    print(f"torch: {torch.__version__} (CUDA build: {torch.version.cuda})")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The job was scheduled without a usable GPU, "
            "or the container image has no CUDA runtime."
        )

    count = torch.cuda.device_count()
    print(f"CUDA devices: {count}")
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"  [{index}] {properties.name} "
            f"sm_{properties.major}{properties.minor} "
            f"{properties.total_memory / 1024**3:.1f} GiB"
        )

    device = torch.device("cuda:0")
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    product = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"2048x2048 matmul on {torch.cuda.get_device_name(0)}: sum={product:.4f}")

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"Peak GPU memory: {peak_gib:.3f} GiB")

    return {"devices": count, "device_name": torch.cuda.get_device_name(0)}


def training_step(config: dict):
    """Prove config forwarding and real autograd on the accelerator.

    The config arrives either as an inline dict or as a YAML file staged
    through GCS; from inside the job the two are indistinguishable, which is
    the point.
    """
    import torch

    _describe_host()

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    num_iterations = config["num_iterations"]
    print("Config received:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")

    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    for iteration in range(num_iterations):
        inputs = torch.randn(batch_size, 64, device=device)
        targets = torch.randn(batch_size, 1, device=device)

        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"  iteration {iteration + 1}/{num_iterations}: loss={loss.item():.6f}")

    print(f"\nFirst loss {losses[0]:.6f} -> last loss {losses[-1]:.6f}")
    return {"final_loss": losses[-1], "device": str(device)}


def gcs_roundtrip(config: dict):
    """Prove the job can read and write the GCS bucket it was staged from.

    On Vertex AI this works through the job's own service account. On Baseten
    it works through the service account key that coalesce injected as
    GCP_SERVICE_ACCOUNT_JSON, so a pass here means the credential plumbing is
    correct end to end.
    """
    import os
    import uuid

    from google.cloud import storage

    _describe_host()

    bucket_name = config["bucket"].removeprefix("gs://").split("/", 1)[0]
    prefix = config.get("prefix", ".coalesce/smoke").strip("/")

    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"\nGOOGLE_APPLICATION_CREDENTIALS: {creds or '(ambient / none)'}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{prefix}/roundtrip_{uuid.uuid4().hex}.txt"
    payload = f"written by coalesce smoke test from {os.uname().nodename}"

    print(f"\nWriting gs://{bucket_name}/{blob_name}")
    bucket.blob(blob_name).upload_from_string(payload)

    print("Reading it back")
    read_back = bucket.blob(blob_name).download_as_text()
    if read_back != payload:
        raise RuntimeError(f"Round trip mismatch: wrote {payload!r}, read {read_back!r}")
    print(f"  contents match: {read_back!r}")

    print(f"Listing gs://{bucket_name}/{prefix}/")
    listed = [blob.name for blob in client.list_blobs(bucket_name, prefix=prefix, max_results=5)]
    for name in listed:
        print(f"  {name}")

    # Deleting proves the last of the four permissions, but it also destroys the
    # evidence. Keeping the object is what you want when a human is going to go
    # and look at the bucket to confirm the job really wrote to it.
    if config.get("keep"):
        print(f"Keeping gs://{bucket_name}/{blob_name}")
        print("\nGCS read, write and list all succeeded.")
        return {"bucket": bucket_name, "blob": blob_name, "listed": len(listed), "kept": True}

    print("Cleaning up")
    bucket.blob(blob_name).delete()

    print("\nGCS read, write, list and delete all succeeded.")
    return {"bucket": bucket_name, "blob": blob_name, "listed": len(listed), "kept": False}


def mounted_dataset_check(config: dict):
    """Prove a read-only gs:// path was mounted into the container.

    Baseten mirrors the GCS prefix and mounts it as a local directory, so the
    job sees plain files rather than having to fetch them.
    """
    import os
    from pathlib import Path

    _describe_host()

    mount_path = Path(config["mount_path"])
    print(f"\nInspecting mount: {mount_path}")

    if not mount_path.exists():
        raise RuntimeError(
            f"{mount_path} does not exist. The dataset mount was not attached; "
            "check that mount_datasets was passed and that the credentials "
            "secret can read the source prefix."
        )

    entries = sorted(mount_path.rglob("*"))
    files = [entry for entry in entries if entry.is_file()]
    print(f"  {len(files)} file(s) visible")
    for entry in files[:10]:
        print(f"    {entry.relative_to(mount_path)} ({entry.stat().st_size} bytes)")

    if not files:
        raise RuntimeError(f"{mount_path} is empty; nothing was mirrored from GCS.")

    sample = files[0]
    head = sample.read_bytes()[:200]
    print(f"\nFirst 200 bytes of {sample.name}: {head!r}")

    writable = os.access(mount_path, os.W_OK)
    print(f"Mount writable: {writable} (expected False -- mounts are read-only)")

    return {"mount_path": str(mount_path), "files": len(files)}
