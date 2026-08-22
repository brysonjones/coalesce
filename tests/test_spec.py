from __future__ import annotations

import pytest

from coalesce import providers
from coalesce.spec import JobSpec, baseten_gpu, canonical_gpu, vertex_gpu


def sample_task() -> None:
    pass


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("h100", "H100"),
        ("H100", "H100"),
        ("NVIDIA_H100_80GB", "H100"),
        ("nvidia-h100-80gb", "H100"),
        ("A100_80GB", "A100"),
        ("NVIDIA_TESLA_A100", "A100_40GB"),
        ("NVIDIA_H200_141GB", "H200"),
        ("t4", "T4"),
    ],
)
def test_gpu_names_fold_to_one_canonical_spelling(given, expected) -> None:
    assert canonical_gpu(given) == expected


def test_unknown_gpu_lists_the_known_ones() -> None:
    with pytest.raises(ValueError, match="Known GPUs"):
        canonical_gpu("RTX_4090")


def test_the_same_name_reaches_each_provider_in_its_own_vocabulary() -> None:
    assert vertex_gpu("H100") == "NVIDIA_H100_80GB"
    assert baseten_gpu("H100") == "H100"
    assert vertex_gpu("A100") == "NVIDIA_A100_80GB"
    assert baseten_gpu("A100") == "A100"


def test_provider_specific_gaps_are_explicit() -> None:
    with pytest.raises(ValueError, match="not available on Vertex AI"):
        vertex_gpu("L40S")
    with pytest.raises(ValueError, match="not available on Baseten"):
        baseten_gpu("K80")


@pytest.mark.parametrize("alias", ["gcp", "GCP", "vertex", "vertex-ai", "Google"])
def test_gcp_aliases_resolve_to_vertex(alias) -> None:
    assert providers.normalize(alias) == "vertex"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        providers.normalize("azure")


def test_functions_defined_in_main_cannot_be_launched() -> None:
    spec = JobSpec(
        func=sample_task,
        gcp_project_id="demo",
        bucket="gs://demo",
        job_name="demo",
    )
    spec.func.__module__ = "__main__"
    with pytest.raises(ValueError, match="cannot be imported remotely"):
        spec.module_name()
