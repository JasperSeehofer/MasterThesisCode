"""Tests for CLI argument parsing (--use_gpu and --num_workers flags)."""

import subprocess
import sys
from unittest.mock import patch

import pytest

from darksiren_emri.arguments import Arguments


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
        [sys.executable, "-m", "darksiren_emri", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "--use_gpu" in result.stdout
    assert "--num_workers" in result.stdout


def test_fisher_cond_threshold_default() -> None:
    """Default fisher_cond_threshold should be 1e16."""
    args = Arguments.create(["."])
    assert args.fisher_cond_threshold == pytest.approx(1e16)


def test_fisher_cond_threshold_custom() -> None:
    """--fisher_cond_threshold should accept a custom float value."""
    args = Arguments.create([".", "--fisher_cond_threshold", "1e8"])
    assert args.fisher_cond_threshold == pytest.approx(1e8)


# --- [HIER] theta-hook CLI plumbing (charter node P6/B1.2,
# WAVE2_REGISTRATION_CHECK_20260829.md F-C; ledger rows #216, #221-#223) ---


def test_theta_b_default_is_identity() -> None:
    """Default --theta_b is 0.0 (the literal-skip identity, GATE T-ID)."""
    args = Arguments.create(["."])
    assert args.theta_b == pytest.approx(0.0)


def test_theta_s_default_is_identity() -> None:
    """Default --theta_s is 1.0 (the literal-skip identity, GATE T-ID)."""
    args = Arguments.create(["."])
    assert args.theta_s == pytest.approx(1.0)


def test_theta_sites_default_is_all() -> None:
    """Default --theta_sites is 'all', matching evaluate()'s own default."""
    args = Arguments.create(["."])
    assert args.theta_sites == "all"


def test_theta_b_custom() -> None:
    """--theta_b should accept a custom float value."""
    args = Arguments.create([".", "--theta_b", "0.01"])
    assert args.theta_b == pytest.approx(0.01)


def test_theta_s_custom() -> None:
    """--theta_s should accept a custom float value."""
    args = Arguments.create([".", "--theta_s", "1.2"])
    assert args.theta_s == pytest.approx(1.2)


@pytest.mark.parametrize("site", ["all", "2.1", "2.2", "2.3"])
def test_theta_sites_valid_choices(site: str) -> None:
    """Each site value evaluate() accepts must parse at the CLI layer."""
    args = Arguments.create([".", "--theta_sites", site])
    assert args.theta_sites == site


def test_theta_sites_invalid_rejected() -> None:
    """An invalid --theta_sites value should cause SystemExit (argparse
    choices=), mirroring evaluate()'s own ValueError guard
    (bayesian_statistics.py ~3542-3600) at the CLI layer."""
    with pytest.raises(SystemExit):
        Arguments.create([".", "--theta_sites", "bogus"])


def test_help_shows_theta_flags() -> None:
    """--help output should include all three --theta_* flags."""
    result = subprocess.run(
        [sys.executable, "-m", "darksiren_emri", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "--theta_b" in result.stdout
    assert "--theta_s" in result.stdout
    assert "--theta_sites" in result.stdout
