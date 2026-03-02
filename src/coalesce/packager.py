"""Package syncing utilities for Vertex AI jobs."""

import importlib
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from google.cloud import storage


def resolve_package_path(package_name: str) -> Path:
    """
    Dynamically resolve the filesystem path of a Python package.

    Args:
        package_name: Name of the package to resolve (e.g., "my_robot_lib")

    Returns:
        Path to the package directory or file

    Raises:
        ImportError: If the package cannot be imported
        ValueError: If the resolved path doesn't exist
    """
    try:
        module = importlib.import_module(package_name)
    except ImportError as e:
        raise ImportError(f"Failed to import package '{package_name}': {e}")

    # Get the __file__ attribute
    if not hasattr(module, "__file__") or module.__file__ is None:
        raise ValueError(f"Package '{package_name}' has no __file__ attribute")

    module_file = Path(module.__file__)

    # If the file is __init__.py, use the parent directory (package root)
    if module_file.name == "__init__.py":
        package_path = module_file.parent
    # If it's a single .py file, use that file
    elif module_file.suffix == ".py":
        package_path = module_file
    else:
        # Fallback: use the directory containing the file
        package_path = module_file.parent

    # Validate that the path exists
    if not package_path.exists():
        raise ValueError(f"Resolved path for '{package_name}' does not exist: {package_path}")

    return package_path


def package_and_upload(
    package_names: list[str],
    bucket_name: str,
    project_id: str,
) -> str:
    """
    Package Python packages and upload to GCS.

    Args:
        package_names: List of package names to package (e.g., ["my_robot_lib"])
        bucket_name: GCS bucket name (e.g., "gs://my-bucket" or "my-bucket")
        project_id: GCP project ID

    Returns:
        GCS URI of the uploaded zip file (e.g., "gs://my-bucket/source/workspace_20240101_120000.zip")
    """
    if not package_names:
        raise ValueError("package_names list cannot be empty")

    # Parse bucket name (handle both "gs://bucket" and "bucket" formats)
    if bucket_name.startswith("gs://"):
        bucket_name = bucket_name[5:]

    # Resolve all package paths
    print(f"Resolving {len(package_names)} package(s)...")
    package_paths = {}
    for package_name in package_names:
        try:
            resolved_path = resolve_package_path(package_name)
            package_paths[package_name] = resolved_path
            print(f"  {package_name} -> {resolved_path}")
        except (ImportError, ValueError) as e:
            print(f"  Failed to resolve {package_name}: {e}")
            raise

    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / "workspace"
        workspace_path.mkdir()

        # Copy resolved package directories into workspace
        print(f"Copying packages to workspace...")
        for package_name, source_path in package_paths.items():
            dest_path = workspace_path / package_name

            if source_path.is_file():
                # Single file module - copy the file with .py extension
                dest_file = workspace_path / f"{package_name}.py"
                shutil.copy2(source_path, dest_file)
                print(f"  Copied file: {package_name}.py")
            else:
                # Package directory - copy the entire directory
                shutil.copytree(
                    source_path,
                    dest_path,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".gitignore"),
                    dirs_exist_ok=False,
                )
                print(f"  Copied directory: {package_name}")

        # Create zip archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"workspace_{timestamp}.zip"
        zip_path = Path(temp_dir) / zip_filename

        print(f"Creating archive: {zip_filename}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(workspace_path):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for file in files:
                    if file.endswith(".pyc"):
                        continue
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(workspace_path)
                    zipf.write(file_path, arcname)

        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  Archive size: {zip_size_mb:.2f} MB")

        # Upload to GCS
        print(f"Uploading to GCS...")
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)

        gcs_blob_name = f"source/{zip_filename}"
        blob = bucket.blob(gcs_blob_name)

        blob.upload_from_filename(str(zip_path))
        gcs_uri = f"gs://{bucket_name}/{gcs_blob_name}"

        print(f"  Uploaded to: {gcs_uri}")

        return gcs_uri
