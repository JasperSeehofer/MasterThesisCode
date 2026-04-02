"""Tests for _write_run_metadata SLURM support and indexed filenames."""

import json
from pathlib import Path

import pytest

from master_thesis_code.arguments import Arguments
from master_thesis_code.main import _write_run_metadata

# All 6 SLURM-related env vars that should be captured
_SLURM_ENV_VARS = [
    "SLURM_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "CUDA_VISIBLE_DEVICES",
    "HOSTNAME",
]


def _make_arguments(
    tmp_path: Path,
    *,
    simulation_index: int = 0,
) -> "Arguments":
    """Create a minimal Arguments-like namespace for testing."""
    import argparse

    ns = argparse.Namespace(
        working_directory=str(tmp_path),
        simulation_steps=10,
        simulation_index=simulation_index,
        evaluate=False,
        h_value=0.73,
        snr_analysis=False,
        use_gpu=False,
        num_workers=1,
        seed=42,
        log_level="INFO",
        generate_figures=None,
        combine=False,
        strategy="physics-floor",
    )
    return Arguments(ns)


def _clean_slurm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all SLURM env vars to ensure test isolation."""
    for var in _SLURM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_no_slurm_env_no_slurm_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When SLURM env vars are NOT set, metadata dict has no 'slurm' key,
    and filename is 'run_metadata.json'."""
    _clean_slurm_env(monkeypatch)
    args = _make_arguments(tmp_path)
    _write_run_metadata(str(tmp_path), 42, args)

    metadata_path = tmp_path / "run_metadata.json"
    assert metadata_path.exists(), "Expected run_metadata.json to be written"
    metadata = json.loads(metadata_path.read_text())
    assert "slurm" not in metadata, "No SLURM key expected when no SLURM env vars are set"


def test_slurm_job_id_and_array_task_id_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When SLURM_JOB_ID and SLURM_ARRAY_TASK_ID are set, they appear in metadata['slurm']."""
    _clean_slurm_env(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    args = _make_arguments(tmp_path)
    _write_run_metadata(str(tmp_path), 42, args)

    # Find the written file (may be indexed due to SLURM_ARRAY_TASK_ID)
    json_files = list(tmp_path.glob("run_metadata*.json"))
    assert len(json_files) == 1
    metadata = json.loads(json_files[0].read_text())
    assert "slurm" in metadata
    assert metadata["slurm"]["SLURM_JOB_ID"] == "12345"
    assert metadata["slurm"]["SLURM_ARRAY_TASK_ID"] == "7"


def test_all_six_slurm_vars_captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When all 6 SLURM vars are set, all appear in metadata['slurm']."""
    _clean_slurm_env(monkeypatch)
    env_values = {
        "SLURM_JOB_ID": "99999",
        "SLURM_ARRAY_TASK_ID": "3",
        "SLURM_NODELIST": "gpu-node-01",
        "SLURM_CPUS_PER_TASK": "8",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "HOSTNAME": "gpu-node-01.cluster",
    }
    for var, val in env_values.items():
        monkeypatch.setenv(var, val)
    args = _make_arguments(tmp_path)
    _write_run_metadata(str(tmp_path), 42, args)

    json_files = list(tmp_path.glob("run_metadata*.json"))
    assert len(json_files) == 1
    metadata = json.loads(json_files[0].read_text())
    assert "slurm" in metadata
    for var, val in env_values.items():
        assert metadata["slurm"][var] == val, f"Expected {var}={val}"


def test_simulation_index_3_uses_indexed_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When simulation_index=3, filename is 'run_metadata_3.json'."""
    _clean_slurm_env(monkeypatch)
    args = _make_arguments(tmp_path, simulation_index=3)
    _write_run_metadata(str(tmp_path), 42, args)

    assert (tmp_path / "run_metadata_3.json").exists(), "Expected run_metadata_3.json"
    assert not (tmp_path / "run_metadata.json").exists(), "Should NOT write run_metadata.json"


def test_slurm_array_task_uses_indexed_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When simulation_index=0 and SLURM_ARRAY_TASK_ID is set, filename is 'run_metadata_0.json'."""
    _clean_slurm_env(monkeypatch)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    args = _make_arguments(tmp_path, simulation_index=0)
    _write_run_metadata(str(tmp_path), 42, args)

    assert (tmp_path / "run_metadata_0.json").exists(), "Expected run_metadata_0.json"
    assert not (tmp_path / "run_metadata.json").exists(), "Should NOT write run_metadata.json"


def test_no_slurm_index_0_uses_default_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When simulation_index=0 and no SLURM env, filename is 'run_metadata.json' (backward compat)."""
    _clean_slurm_env(monkeypatch)
    args = _make_arguments(tmp_path, simulation_index=0)
    _write_run_metadata(str(tmp_path), 42, args)

    assert (tmp_path / "run_metadata.json").exists(), "Expected run_metadata.json"
