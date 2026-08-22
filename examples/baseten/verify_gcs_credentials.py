#!/usr/bin/env python
"""Check that a GCP service account key can drive coalesce's GCS access.

This walks the exact path a Baseten container takes: the key is handed to the
task runner through GCP_SERVICE_ACCOUNT_JSON, turned into Application Default
Credentials, and then used for a full write/read/list/delete cycle against the
bucket. It touches no Baseten API, so the credential half of the setup can be
confirmed on its own -- useful before a first launch, or when a job fails and
you need to know which half broke.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from coalesce import task


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--key-file",
        default="~/coalesce-baseten-key.json",
        help="Service account JSON key, the same one uploaded to Baseten",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("COALESCE_BUCKET"),
        help="GCS bucket to test against (or set COALESCE_BUCKET)",
    )
    parser.add_argument("--prefix", default=".coalesce/smoke")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Leave the test object in the bucket so you can view it in the console",
    )
    args = parser.parse_args()

    if not args.bucket:
        print("Set --bucket (or COALESCE_BUCKET) to the bucket you want to test.")
        return 1

    key_file = Path(args.key_file).expanduser()
    if not key_file.exists():
        print(f"No key file at {key_file}")
        return 1

    # Hand the key over the way the launcher hands it to a Baseten job, and let
    # the task runner do the rest. Clear any ambient ADC first, so a pass here
    # cannot be your own gcloud login in disguise.
    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    os.environ["GCP_SERVICE_ACCOUNT_JSON"] = key_file.read_text()
    task.setup_gcp_credentials()

    from google.cloud import storage

    bucket_name = args.bucket.removeprefix("gs://").split("/", 1)[0]
    prefix = args.prefix.strip("/")
    client = storage.Client()

    acting_as = getattr(client._credentials, "service_account_email", None)
    print(f"Acting as: {acting_as or '(not a service account)'}")
    if not acting_as:
        print("Refusing to continue: these are not service account credentials.")
        return 1

    bucket = client.bucket(bucket_name)
    blob_name = f"{prefix}/credential_check_{uuid.uuid4().hex[:8]}.txt"
    payload = (
        "coalesce credential check\n"
        f"service account: {acting_as}\n"
        "If you can read this in the console, a Baseten job can write here too.\n"
    )

    print(f"\nWriting  gs://{bucket_name}/{blob_name}")
    bucket.blob(blob_name).upload_from_string(payload)

    print("Reading  it back")
    read_back = bucket.blob(blob_name).download_as_text()
    if read_back != payload:
        print(f"Round trip mismatch: wrote {payload!r}, read {read_back!r}")
        return 1
    print("  contents match")

    print(f"Listing  gs://{bucket_name}/{prefix}/")
    for blob in client.list_blobs(bucket_name, prefix=prefix, max_results=10):
        print(f"    {blob.name}  ({blob.size} bytes)")

    if args.keep:
        print("\nLeaving the object in place (--keep).")
    else:
        bucket.blob(blob_name).delete()
        print("\nDeleted the test object.")

    console = (
        f"https://console.cloud.google.com/storage/browser/{bucket_name}/{prefix}"
        f"?project={os.environ.get('GOOGLE_CLOUD_PROJECT', '')}"
    )
    print(f"\nWrite, read, list{'' if args.keep else ' and delete'} all succeeded.")
    print(f"View it at: {console}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
