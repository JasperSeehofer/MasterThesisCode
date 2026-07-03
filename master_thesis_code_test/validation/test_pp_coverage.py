"""Tests for the independent P-P/coverage harness (commission d2 provenance).

Fast tests use a tiny config (8 realizations x 40 events) and assert
determinism, results-structure validity, and the qualitative D2 finding
(volume kernel better calibrated than bare kernel at sigma_z = 0.035).
The slow test uses a medium config and asserts the calibrated kernel's 68%
coverage sits in a generous nominal band with the bare kernel below it.
"""

import dataclasses

import pytest

from master_thesis_code.validation.pp_coverage import PPCoverageConfig, run_coverage

TINY = PPCoverageConfig(
    n_realizations=8,
    n_events=40,
    sigma_z=0.035,
    injected_truths=[0.72],
    seed=42,
    kernel="bare",
)


@pytest.fixture(scope="module")
def tiny_bare() -> dict:
    """Tiny-config run with the bare (production-style) kernel."""
    return run_coverage(TINY)


@pytest.fixture(scope="module")
def tiny_volume() -> dict:
    """Tiny-config run with the volume-weighted (calibrated) kernel."""
    return run_coverage(dataclasses.replace(TINY, kernel="volume"))


def test_determinism_same_seed_identical_results(tiny_bare: dict) -> None:
    """Same seed must give bit-identical results (no unseeded randomness)."""
    assert run_coverage(TINY) == tiny_bare


def test_results_structure_and_validity(tiny_bare: dict) -> None:
    """Results carry config + per-truth stats with valid ranges."""
    assert set(tiny_bare) == {"config", "results"}
    assert tiny_bare["config"]["kernel"] == "bare"
    assert set(tiny_bare["results"]) == {"0.7200"}
    entry = tiny_bare["results"]["0.7200"]
    assert entry["h_true"] == pytest.approx(0.72)
    for level in ("50", "68", "90"):
        assert 0.0 <= entry["coverage"][level] <= 1.0
    assert 0.0 <= entry["rail_fraction"] <= 1.0
    assert TINY.h_min <= entry["map_mean"] <= TINY.h_max
    assert entry["map_std"] >= 0.0
    assert entry["map_bias"] == pytest.approx(entry["map_mean"] - 0.72)


def test_volume_kernel_better_calibrated_than_bare(tiny_bare: dict, tiny_volume: dict) -> None:
    """D2 finding at sigma_z=0.035: volume kernel beats bare kernel.

    The volume-weighted kernel's 68% coverage exceeds the bare kernel's, and
    the bare kernel's MAP bias is more negative (Eddington-in-z low bias).
    """
    bare = tiny_bare["results"]["0.7200"]
    volume = tiny_volume["results"]["0.7200"]
    assert volume["coverage"]["68"] > bare["coverage"]["68"]
    assert bare["map_bias"] < volume["map_bias"]


@pytest.mark.slow
def test_medium_config_calibration_band() -> None:
    """Calibrated kernel's 68% coverage in a generous nominal band; bare below it."""
    config = PPCoverageConfig(
        n_realizations=60,
        n_events=150,
        sigma_z=0.035,
        injected_truths=[0.72],
        seed=7,
        kernel="volume",
    )
    volume = run_coverage(config)["results"]["0.7200"]
    bare = run_coverage(dataclasses.replace(config, kernel="bare"))["results"]["0.7200"]
    assert 0.5 <= volume["coverage"]["68"] <= 0.85
    assert bare["coverage"]["68"] < volume["coverage"]["68"]
    assert abs(volume["map_bias"]) < abs(bare["map_bias"])


def test_tiny_config_exact_value_pins(tiny_bare: dict, tiny_volume: dict) -> None:
    """Exact-float regression pins of the harness output (both kernels).

    Unlike the determinism test (which compares two same-code runs), these
    pins freeze the CURRENT numerical behaviour so any host-z kernel change
    (e.g. a peculiar-velocity sigma_z term) shows up as a deliberate pin
    update in the same diff.
    """
    bare = tiny_bare["results"]["0.7200"]
    assert bare["map_mean"] == pytest.approx(0.6965000000000001, rel=1e-12)
    assert bare["map_bias"] == pytest.approx(-0.023499999999999854, rel=1e-9)
    assert bare["coverage"]["68"] == pytest.approx(0.25, rel=1e-12)
    assert bare["rail_fraction"] == 0.0

    volume = tiny_volume["results"]["0.7200"]
    assert volume["map_mean"] == pytest.approx(0.7185000000000001, rel=1e-12)
    assert volume["map_bias"] == pytest.approx(-0.0014999999999998348, rel=1e-6)
    assert volume["coverage"]["68"] == pytest.approx(0.625, rel=1e-12)
    assert volume["rail_fraction"] == 0.0
