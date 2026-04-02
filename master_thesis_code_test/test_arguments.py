"""Tests for CLI argument parsing (--use_gpu and --num_workers flags)."""

import subprocess
import sys
from unittest.mock import patch

from master_thesis_code.arguments import Arguments


def test_use_gpu_flag_default() -> None:
    """When --use_gpu is not passed, use_gpu should be False."""
    args = Arguments.create(["."])
    assert args.use_gpu is False


def test_use_gpu_flag_set() -> None:
    """When --use_gpu is passed, use_gpu should be True."""
    args = Arguments.create([".", "--use_gpu"])
    assert args.use_gpu is True


def test_num_workers_explicit() -> None:
    """When --num_workers is explicitly set, it should return that value."""
    args = Arguments.create([".", "--num_workers", "4"])
    assert args.num_workers == 4


def test_num_workers_minimum_one() -> None:
    """When --num_workers is set to 0, it should be clamped to 1."""
    args = Arguments.create([".", "--num_workers", "0"])
    assert args.num_workers == 1


def test_num_workers_negative_clamped_to_one() -> None:
    """When --num_workers is negative, it should be clamped to 1."""
    args = Arguments.create([".", "--num_workers", "-3"])
    assert args.num_workers == 1


def test_num_workers_default_uses_affinity() -> None:
    """When --num_workers is omitted, default is sched_getaffinity(0) - 2."""
    args = Arguments.create(["."])
    with patch("os.sched_getaffinity", return_value=set(range(8))):
        result = args.num_workers
    assert result == 6


def test_num_workers_default_fallback_cpu_count() -> None:
    """When sched_getaffinity raises AttributeError, fall back to cpu_count() - 2."""
    args = Arguments.create(["."])
    with (
        patch("os.sched_getaffinity", side_effect=AttributeError),
        patch("os.cpu_count", return_value=4),
    ):
        result = args.num_workers
    assert result == 2


def test_num_workers_default_minimum_one() -> None:
    """When sched_getaffinity returns 2 CPUs (2 - 2 = 0), clamp to 1."""
    args = Arguments.create(["."])
    with patch("os.sched_getaffinity", return_value=set(range(2))):
        result = args.num_workers
    assert result == 1


def test_combine_flag_default() -> None:
    """When --combine is not passed, combine should be False."""
    args = Arguments.create(["."])
    assert args.combine is False


def test_combine_flag_set() -> None:
    """When --combine is passed, combine should be True."""
    args = Arguments.create([".", "--combine"])
    assert args.combine is True


def test_strategy_default() -> None:
    """Default strategy should be physics-floor."""
    args = Arguments.create([".", "--combine"])
    assert args.strategy == "physics-floor"


def test_strategy_exclude() -> None:
    """Strategy should accept 'exclude' value."""
    args = Arguments.create([".", "--combine", "--strategy", "exclude"])
    assert args.strategy == "exclude"


def test_strategy_invalid() -> None:
    """Invalid strategy should cause SystemExit."""
    import pytest

    with pytest.raises(SystemExit):
        Arguments.create([".", "--combine", "--strategy", "invalid"])


def test_help_shows_flags() -> None:
    """--help output should include both --use_gpu and --num_workers."""
    result = subprocess.run(
        [sys.executable, "-m", "master_thesis_code", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "--use_gpu" in result.stdout
    assert "--num_workers" in result.stdout
