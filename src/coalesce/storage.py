"""Google Cloud Storage helpers shared by every coalesce provider.

GCS is coalesce's storage layer no matter where the job runs. Staging uploads
always land in your bucket and the remote task runner always reads them back
from there, so a Vertex job and a Baseten job stage their code and configs
through exactly the same path.
"""

from datetime import datetime
from pathlib import Path

from google.cloud import storage


def normalize_bucket_and_prefix(bucket: str) -> tuple[str, str]:
    """Split "gs://bucket/some/prefix" into ("bucket", "some/prefix")."""
    normalized = bucket[5:] if bucket.startswith("gs://") else bucket
    bucket_name, _, prefix = normalized.partition("/")
    if not bucket_name:
        raise ValueError("bucket must include a bucket name")
    return bucket_name, prefix.strip("/")


def join_parts(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def parse_uri(uri: str) -> tuple[str, str]:
    """Split "gs://bucket/path/to/blob" into ("bucket", "path/to/blob")."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    bucket_name, _, blob_name = uri[5:].partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Not a GCS URI: {uri}")
    return bucket_name, blob_name


def upload_config(
    config_path: str | Path,
    bucket_name: str,
    project_id: str,
    prefix: str = "configs",
) -> str:
    """Upload a config file to GCS and return the URI."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix_normalized = prefix.strip("/")
    filename = f"{config_path.stem}_{timestamp}{config_path.suffix}"
    gcs_blob_name = f"{prefix_normalized}/{filename}" if prefix_normalized else filename

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name)
    blob.upload_from_filename(str(config_path))

    gcs_uri = f"gs://{bucket_name}/{gcs_blob_name}"
    print(f"  Uploaded config to: {gcs_uri}")
    return gcs_uri
