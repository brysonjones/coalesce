#!/usr/bin/env python
"""Generic entry point for Vertex AI jobs.

This script is uploaded to Vertex AI and runs as the entry point.
It downloads synced packages from GCS, then imports and calls the
specified function via environment variables.

Environment variables:
    SYNC_PACKAGES_GCS_URI: GCS URI of the workspace.zip containing synced packages
    TASK_MODULE: Python module name containing the function to run
    TASK_FUNCTION: Name of the function to call
    TASK_CONFIG_JSON: JSON-serialized config dict (optional)
    TASK_CONFIG_GCS_URI: GCS URI of config file to download (optional)
"""

import importlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml
from google.cloud import storage


def setup_pythonpath():
    """
    Add current working directory to PYTHONPATH so synced packages are importable.
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
        os.environ["PYTHONPATH"] = f"{cwd}:{os.environ.get('PYTHONPATH', '')}"
        print(f"Added {cwd} to PYTHONPATH")


def setup_synced_packages():
    """
    Download and extract synced packages from GCS if SYNC_PACKAGES_GCS_URI is set.
    """
    gcs_uri = os.environ.get("SYNC_PACKAGES_GCS_URI")
    if not gcs_uri:
        return

    print(f"Setting up synced packages from: {gcs_uri}")

    # Parse GCS URI (format: gs://bucket/path/to/file.zip)
    if not gcs_uri.startswith("gs://"):
        print(f"Invalid GCS URI format: {gcs_uri}")
        return

    parts = gcs_uri[5:].split("/", 1)
    if len(parts) != 2:
        print(f"Invalid GCS URI format: {gcs_uri}")
        return

    bucket_name, blob_name = parts
    extract_path = Path(os.getcwd())

    # Download zip file
    print("  Downloading from GCS...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    zip_path = os.path.join(os.getcwd(), "workspace.zip")
    blob.download_to_filename(zip_path)
    print(f"  Downloaded to: {zip_path}")

    # Extract zip file to current working directory
    print(f"  Extracting to: {extract_path}")
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_path)
    print("  Extracted packages")

    # Clean up zip file
    os.unlink(zip_path)

    # Set up PYTHONPATH
    setup_pythonpath()

    # List extracted packages (both directories and .py files)
    extracted_items = [
        item.name
        for item in extract_path.iterdir()
        if (item.is_dir() or (item.is_file() and item.suffix == ".py"))
        and not item.name.startswith(".")
        and item.name != "workspace.zip"
    ]
    if extracted_items:
        print(f"  Available packages: {', '.join(extracted_items)}")


def load_config() -> dict | None:
    """
    Load config from environment variables.

    Checks TASK_CONFIG_JSON first (inline JSON), then TASK_CONFIG_GCS_URI (file).
    Returns None if no config is specified.
    """
    # Check for inline JSON config
    config_json = os.environ.get("TASK_CONFIG_JSON")
    if config_json:
        print("Loading config from TASK_CONFIG_JSON...")
        config = json.loads(config_json)
        print(f"  Loaded {len(config)} config keys")
        return config

    # Check for GCS config file
    config_gcs_uri = os.environ.get("TASK_CONFIG_GCS_URI")
    if config_gcs_uri:
        print(f"Loading config from: {config_gcs_uri}")

        # Parse GCS URI
        if not config_gcs_uri.startswith("gs://"):
            print(f"Invalid GCS URI format: {config_gcs_uri}")
            sys.exit(1)

        parts = config_gcs_uri[5:].split("/", 1)
        if len(parts) != 2:
            print(f"Invalid GCS URI format: {config_gcs_uri}")
            sys.exit(1)

        bucket_name, blob_name = parts

        # Download config file
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Determine file type from extension
        suffix = Path(blob_name).suffix.lower()
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as f:
            blob.download_to_file(f)
            temp_path = f.name

        # Load config based on file type
        with open(temp_path) as f:
            if suffix in (".yaml", ".yml"):
                config = yaml.safe_load(f)
            elif suffix == ".json":
                config = json.load(f)
            else:
                # Try YAML first, then JSON
                content = f.read()
                try:
                    config = yaml.safe_load(content)
                except yaml.YAMLError:
                    config = json.loads(content)

        os.unlink(temp_path)
        print(f"  Loaded {len(config)} config keys")
        return config

    return None


def main():
    """Main entry point - downloads packages and runs the specified function."""
    print("=" * 60)
    print("coalesce task runner")
    print("=" * 60)

    # Setup synced packages
    setup_synced_packages()

    # Load config if provided
    config = load_config()

    # Get task function info from environment
    task_module = os.environ.get("TASK_MODULE")
    task_function = os.environ.get("TASK_FUNCTION")

    if not task_module or not task_function:
        print("Error: TASK_MODULE and TASK_FUNCTION environment variables must be set")
        print(f"  TASK_MODULE: {task_module}")
        print(f"  TASK_FUNCTION: {task_function}")
        sys.exit(1)

    print(f"Running {task_module}.{task_function}()")
    print("=" * 60)

    # Import the module and get the function
    try:
        module = importlib.import_module(task_module)
    except ImportError as e:
        print(f"Error: Failed to import module '{task_module}': {e}")
        sys.exit(1)

    if not hasattr(module, task_function):
        print(f"Error: Module '{task_module}' has no function '{task_function}'")
        sys.exit(1)

    func = getattr(module, task_function)

    # Call the function (with or without config)
    try:
        if config is not None:
            result = func(config)
        else:
            result = func()
        print("=" * 60)
        print(f"Task completed successfully")
        if result is not None:
            print(f"Result: {result}")
    except Exception as e:
        print(f"Error: Task failed with exception: {e}")
        raise


if __name__ == "__main__":
    main()
