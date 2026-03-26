"""Tests for CPU-safe MemoryManagement class."""

import logging
from unittest.mock import patch

import pytest

from master_thesis_code.memory_management import MemoryManagement


def test_cpu_import() -> None:
    """Importing MemoryManagement does not raise ImportError."""
    from master_thesis_code.memory_management import MemoryManagement as MemMgmt

    assert MemMgmt is not None


@patch("master_thesis_code.memory_management._GPUTIL_AVAILABLE", False)
@patch("master_thesis_code.memory_management.GPUtil", None)
@patch("master_thesis_code.memory_management._CUPY_AVAILABLE", False)
@patch("master_thesis_code.memory_management.cp", None)
def test_cpu_instantiation() -> None:
    """MemoryManagement() succeeds on CPU; memory_pool is None, _gpu_monitor is empty list."""
    m = MemoryManagement()
    assert m.memory_pool is None
    assert m._gpu_monitor == []
    assert m._fft_cache is None


@patch("master_thesis_code.memory_management._GPUTIL_AVAILABLE", False)
@patch("master_thesis_code.memory_management.GPUtil", None)
@patch("master_thesis_code.memory_management._CUPY_AVAILABLE", False)
@patch("master_thesis_code.memory_management.cp", None)
def test_cpu_instantiation_with_use_gpu_false() -> None:
    """MemoryManagement(use_gpu=False) succeeds, _use_gpu is False."""
    m = MemoryManagement(use_gpu=False)
    assert m._use_gpu is False


@patch("master_thesis_code.memory_management._GPUTIL_AVAILABLE", False)
@patch("master_thesis_code.memory_management.GPUtil", None)
@patch("master_thesis_code.memory_management._CUPY_AVAILABLE", False)
@patch("master_thesis_code.memory_management.cp", None)
def test_gpu_usage_stamp_cpu() -> None:
    """gpu_usage_stamp() appends to time_series, empty list to gpu_usage, 0.0 to memory_pool."""
    m = MemoryManagement()
    m.gpu_usage_stamp()
    assert len(m.time_series) == 1
    assert m.time_series[0] >= 0.0
    assert m.gpu_usage == [[]]
    assert m.memory_pool_gpu_usage == [0.0]


@patch("master_thesis_code.memory_management._GPUTIL_AVAILABLE", False)
@patch("master_thesis_code.memory_management.GPUtil", None)
@patch("master_thesis_code.memory_management._CUPY_AVAILABLE", False)
@patch("master_thesis_code.memory_management.cp", None)
def test_display_gpu_information_cpu(caplog: pytest.LogCaptureFixture) -> None:
    """display_GPU_information() logs 'No GPU available.' and returns without error."""
    m = MemoryManagement()
    with caplog.at_level(logging.INFO):
        m.display_GPU_information()
    assert "No GPU available." in caplog.text


@patch("master_thesis_code.memory_management._GPUTIL_AVAILABLE", False)
@patch("master_thesis_code.memory_management.GPUtil", None)
@patch("master_thesis_code.memory_management._CUPY_AVAILABLE", False)
@patch("master_thesis_code.memory_management.cp", None)
def test_free_gpu_memory_cpu() -> None:
    """free_gpu_memory() returns None without error when memory_pool is None."""
    m = MemoryManagement()
    m.free_gpu_memory()  # should not raise


@patch("master_thesis_code.memory_management._GPUTIL_AVAILABLE", False)
@patch("master_thesis_code.memory_management.GPUtil", None)
@patch("master_thesis_code.memory_management._CUPY_AVAILABLE", False)
@patch("master_thesis_code.memory_management.cp", None)
def test_display_fft_cache_cpu() -> None:
    """display_fft_cache() returns without error when _fft_cache is None."""
    m = MemoryManagement()
    m.display_fft_cache()  # should not raise
