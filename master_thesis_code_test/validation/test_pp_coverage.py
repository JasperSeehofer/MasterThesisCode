"""Tests for the independent P-P/coverage harness (commission d2 provenance).

Fast tests use a tiny config (8 realizations x 40 events) and assert
determinism, results-structure validity, and the qualitative D2 finding
(volume kernel better calibrated than bare kernel at sigma_z = 0.035).
The slow test uses a medium config and asserts the calibrated kernel's 68%
coverage sits in a generous nominal band with the bare kernel below it.
"""

import dataclasses
import json
import math
from pathlib import Path

import pytest

from master_thesis_code.validation.pp_coverage import PPCoverageConfig, main, run_coverage

TINY = PPCoverageConfig(
    n_realizations=8,
    n_events=40,
    sigma_z=0.035,
    injected_truths=[0.72],
    seed=42,
    kernel="bare",
)

TINY_DEEPVENUE = PPCoverageConfig(
    n_realizations=6,
    n_events=30,
    injected_truths=[0.72],
    seed=20260711,
    kernel="volume",
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


def test_z_support_none_golden_pin() -> None:
    """Golden pin measured at HEAD; the z_support=None path MUST stay bit-identical

    after the truncated-mode change (issue #29 harness validation, pin-first per
    ed46390).
    """
    config = PPCoverageConfig(
        n_realizations=2,
        n_events=25,
        injected_truths=[0.72],
        seed=20260710,
        kernel="volume",
    )
    entry = run_coverage(config)["results"]["0.7200"]
    assert entry["map_mean"] == pytest.approx(0.7260000000000001, rel=1e-12)
    assert entry["map_std"] == pytest.approx(0.0020000000000000018, rel=1e-12)
    assert entry["map_bias"] == pytest.approx(0.006000000000000116, rel=1e-12)
    assert entry["coverage"]["50"] == 1.0
    assert entry["coverage"]["68"] == 1.0
    assert entry["coverage"]["90"] == 1.0
    assert entry["rail_fraction"] == 0.0


def test_z_support_at_zmax_pop_matches_untruncated_limiting_case() -> None:
    """z_support >= Z_MAX_POP (0.95) is the untruncated limiting case.

    z_host is sampled in [Z_MIN, Z_MAX_POP], so setting z_support at the
    population ceiling routes zero events into the completion branch: the
    ``results`` block matches the z_support=None run exactly and
    completion_fraction is 0.0 (issue #29 harness validation).
    """
    untruncated = run_coverage(TINY_DEEPVENUE)
    truncated = run_coverage(dataclasses.replace(TINY_DEEPVENUE, z_support=0.95))
    assert truncated["results"] == untruncated["results"]
    assert truncated["results"]["0.7200"]["completion_fraction"] == 0.0


def test_small_z_support_completion_fraction_near_one_and_posterior_finite() -> None:
    """At deep truncation (z_support=0.05) almost all hosts are zero-host events.

    The pure-completion B_num/D posterior must stay finite/normalizable (no
    NaN/inf) and its MAP must remain on the H0 grid.
    """
    config = dataclasses.replace(TINY_DEEPVENUE, z_support=0.05)
    entry = run_coverage(config)["results"]["0.7200"]
    assert entry["completion_fraction"] > 0.9
    assert math.isfinite(entry["map_mean"])
    assert math.isfinite(entry["map_std"])
    assert all(math.isfinite(v) for v in entry["coverage"].values())
    assert TINY_DEEPVENUE.h_min <= entry["map_mean"] <= TINY_DEEPVENUE.h_max


def test_z_support_monotonic_completion_fraction() -> None:
    """completion_fraction is strictly in (0,1) and increases as z_support decreases."""
    cf_moderate = run_coverage(dataclasses.replace(TINY_DEEPVENUE, z_support=0.35))["results"][
        "0.7200"
    ]["completion_fraction"]
    cf_deeper = run_coverage(dataclasses.replace(TINY_DEEPVENUE, z_support=0.2))["results"][
        "0.7200"
    ]["completion_fraction"]
    assert 0.0 < cf_moderate < cf_deeper < 1.0


def test_gray_mode_requires_z_support() -> None:
    """mixture_mode='gray' without a catalogue support edge is undefined."""
    config = dataclasses.replace(TINY_DEEPVENUE, mixture_mode="gray")
    with pytest.raises(ValueError, match="z_support"):
        run_coverage(config)


def test_gray_zmax_limiting_case() -> None:
    """gray + z_support at the population ceiling: all events take the mixture branch.

    completion_fraction is 0 (no zero-host events), the completion-tilt
    diagnostic is the None sentinel, and the posterior stays finite with the
    MAP on the H0 grid.
    """
    config = dataclasses.replace(TINY_DEEPVENUE, z_support=0.95, mixture_mode="gray")
    entry = run_coverage(config)["results"]["0.7200"]
    assert entry["completion_fraction"] == 0.0
    assert entry["dlogL_dh_completion_mean"] is None
    assert entry["dlogL_dh_host_mean"] is not None
    assert math.isfinite(entry["map_mean"])
    assert math.isfinite(entry["map_std"])
    assert all(math.isfinite(v) for v in entry["coverage"].values())
    assert TINY_DEEPVENUE.h_min <= entry["map_mean"] <= TINY_DEEPVENUE.h_max


def test_gray_shallow_venue_close_to_two_branch() -> None:
    """Shallow venue (p_det ~= 1 over the in-catalogue support): gray ~ two_branch.

    SOFT bound, NOT an exact identity: the gray host term p_i = (beta_G *
    N_i/D_g_i + B_num)/D differs from the two_branch N_i/D by construction;
    this is a sanity check that the D_g_i per-host denominator + admixture do
    not blow the estimator up (map_mean within ~3 grid steps).
    z_support=0.1 (p_det at the edge = 1.00000 for h=0.72) rather than 0.05 so
    that several in-catalogue events actually exercise the mixture branch on
    the tiny config (7 events vs 1 at 0.05, measured).
    """
    two_branch = run_coverage(dataclasses.replace(TINY_DEEPVENUE, z_support=0.1))
    gray = run_coverage(dataclasses.replace(TINY_DEEPVENUE, z_support=0.1, mixture_mode="gray"))
    tb_map = two_branch["results"]["0.7200"]["map_mean"]
    gray_map = gray["results"]["0.7200"]["map_mean"]
    assert abs(gray_map - tb_map) < 0.012


def test_conditioned_zmax_matches_two_branch_untruncated() -> None:
    """conditioned + z_support=0.95 reproduces the untruncated two_branch run.

    Exact identity in exact arithmetic: beta_G is computed on D(h)'s own node
    grid, so beta_G == Dh at z_support >= Z_MAX_POP, and N_i reuses the
    two_branch host quadrature, hence N_i/beta_G == num/Dh.
    """
    untruncated = run_coverage(TINY_DEEPVENUE)
    conditioned = run_coverage(
        dataclasses.replace(TINY_DEEPVENUE, z_support=0.95, mixture_mode="conditioned")
    )
    u = untruncated["results"]["0.7200"]
    c = conditioned["results"]["0.7200"]
    assert c["map_mean"] == pytest.approx(u["map_mean"], rel=1e-6)
    assert c["map_std"] == pytest.approx(u["map_std"], rel=1e-6, abs=1e-12)
    assert c["coverage"] == u["coverage"]
    assert c["completion_fraction"] == 0.0


def test_membership_on_observed_changes_completion_fraction() -> None:
    """Observed-z membership (N-2d probe) reroutes boundary events.

    With sigma_z scatter at a moderate z_support, deciding membership on the
    observed z_gal instead of the true z_host flips some events across the
    support edge, so the mean completion_fraction differs (statistical
    assertion, not an exact value).
    """
    base = dataclasses.replace(TINY_DEEPVENUE, z_support=0.3)
    cf_true = run_coverage(base)["results"]["0.7200"]["completion_fraction"]
    cf_obs = run_coverage(dataclasses.replace(base, membership_on_observed=True))["results"][
        "0.7200"
    ]["completion_fraction"]
    assert 0.0 < cf_true < 1.0
    assert 0.0 < cf_obs < 1.0
    assert cf_obs != cf_true


def test_gray_determinism_same_seed() -> None:
    """Two gray-mode runs with the same seed are bit-identical."""
    config = dataclasses.replace(TINY_DEEPVENUE, z_support=0.3, mixture_mode="gray")
    assert run_coverage(config) == run_coverage(config)


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


def test_exact_mode_requires_z_support() -> None:
    """mixture_mode='exact' without a catalogue support edge is undefined."""
    config = dataclasses.replace(TINY_DEEPVENUE, mixture_mode="exact")
    with pytest.raises(ValueError, match="z_support"):
        run_coverage(config)


def test_exact_zmax_matches_two_branch_map() -> None:
    """exact + z_support at the population ceiling matches the untruncated MAP.

    completion_fraction is exactly 0 (z_host is sampled in [Z_MIN, Z_MAX_POP]
    = [1e-4, 0.95], so no event routes to the completion branch at
    z_support=0.95). The MAP is NOT bit-identical: exact clamps the host
    quadrature at z_hi -> min(z_hi, 0.95) while two_branch clamps to
    _Z_GRID[-1] = 1.5; the [0.95, 1.5] kernel mass is negligible because
    Z_MAX_POP = 0.95 caps the population, so the MAPs agree to well within
    the measured tolerance below.
    """
    untruncated = run_coverage(TINY_DEEPVENUE)["results"]["0.7200"]
    exact = run_coverage(dataclasses.replace(TINY_DEEPVENUE, z_support=0.95, mixture_mode="exact"))[
        "results"
    ]["0.7200"]
    assert exact["completion_fraction"] == 0.0
    # Measured at implementation time: map_mean identical to float precision
    # on the tiny config (difference 0.0); assert with a tight rel tolerance.
    assert exact["map_mean"] == pytest.approx(untruncated["map_mean"], rel=1e-12)


def test_exact_deep_truncation_finite_and_completion_matches_two_branch() -> None:
    """Deep truncation (z_support=0.2): finite posterior, identical event routing.

    The membership draws are consumed from the RNG stream BEFORE the branch
    dispatch, so exact and two_branch route bit-identically at the same
    config/seed: completion_fraction must be EXACTLY equal (not approx).
    """
    tb = run_coverage(dataclasses.replace(TINY_DEEPVENUE, z_support=0.2))["results"]["0.7200"]
    ex = run_coverage(dataclasses.replace(TINY_DEEPVENUE, z_support=0.2, mixture_mode="exact"))[
        "results"
    ]["0.7200"]
    assert ex["completion_fraction"] == tb["completion_fraction"]
    assert 0.0 < ex["completion_fraction"] < 1.0
    assert math.isfinite(ex["map_mean"])
    assert math.isfinite(ex["map_std"])
    assert all(math.isfinite(v) for v in ex["coverage"].values())
    assert TINY_DEEPVENUE.h_min <= ex["map_mean"] <= TINY_DEEPVENUE.h_max


def test_exact_determinism_same_seed() -> None:
    """Two exact-mode runs with the same seed are bit-identical."""
    config = dataclasses.replace(TINY_DEEPVENUE, z_support=0.2, mixture_mode="exact")
    assert run_coverage(config) == run_coverage(config)


def test_n_z_quad_cli_flag_threads_into_config(tmp_path: Path) -> None:
    """--n-z-quad threads into config.n_z_quad in the written JSON."""
    out = tmp_path / "r.json"
    main(
        [
            "--n-realizations",
            "2",
            "--n-events",
            "10",
            "--truths",
            "0.72",
            "--seed",
            "20260711",
            "--n-z-quad",
            "480",
            "--output",
            str(out),
        ]
    )
    data = json.loads(out.read_text())
    assert data["config"]["n_z_quad"] == 480
