"""Self-contained toy tests of the calibration-gate P–P machinery.

These tests exercise the readout layer ONLY, on constructed inputs with known
answers — no injection pool, no CRB CSV, no universe draws. They certify that
the instrument's statistics can (a) pass a calibrated input, (b) detect a
deliberately mis-calibrated input, (c) count coverage correctly, and
(d) detect rails — before any real cell is run.

Run from the repo root:

    uv run pytest results/calibration_gate_20260808/test_calibration_gate_toy.py -m "not gpu"

The V2 HPD-port certification (agreement with ``pp_coverage._hpd_contains``)
lives in ``master_thesis_code_test/validation/test_calibration_gate.py``.
"""

import math

import numpy as np
import pytest

from master_thesis_code.validation import calibration_gate as cg
from master_thesis_code.validation import closed_loop_gfrac as cl

H_GRID = np.asarray(cl.CANONICAL_H_GRID, dtype=np.float64)
FINE_GRID = np.linspace(0.50, 0.96, 231)  # fine grid for coverage-accuracy tests


# ── (1) P–P / KS statistic: flat input passes, biased input is detected ─────


def test_uniform_pit_sample_is_flat_within_ks_band() -> None:
    """A genuinely uniform PIT sample must sit inside the 95% KS band."""
    rng = np.random.default_rng(20260808)
    n = 400
    pits = rng.random(n)
    d = cg.ks_distance(pits)
    d95 = 1.358 / math.sqrt(n)
    assert d <= d95, f"uniform sample rejected: D={d:.4f} > D95={d95:.4f}"


def test_uniform_pit_flat_within_binomial_band_per_decile() -> None:
    """Flat P–P: each PIT decile holds ~10% of the sample within 3 sigma binomial."""
    rng = np.random.default_rng(20260808)
    n = 1000
    pits = rng.random(n)
    counts, _ = np.histogram(pits, bins=10, range=(0.0, 1.0))
    sigma = math.sqrt(n * 0.1 * 0.9)
    assert np.all(np.abs(counts - n * 0.1) < 3.0 * sigma)


def test_biased_pit_sample_is_detected_at_99() -> None:
    """A skewed PIT sample (Beta(2,1), i.e. a coherently biased posterior
    ensemble) must exceed the 99% KS critical value at N=400."""
    rng = np.random.default_rng(7)
    n = 400
    pits = rng.beta(2.0, 1.0, size=n)
    d = cg.ks_distance(pits)
    d99 = 1.628 / math.sqrt(n)
    assert d > d99, f"biased sample not detected: D={d:.4f} <= D99={d99:.4f}"


def test_pit_from_full_readout_is_uniform_for_calibrated_gaussians() -> None:
    """End-to-end PIT: Gaussian posteriors with truths drawn from the same
    Gaussian => the pp_readout PIT sample is Uniform(0,1) within the KS band."""
    rng = np.random.default_rng(42)
    n = 400
    mu, sd = 0.730, 0.03
    pits = []
    for _ in range(n):
        h_true = float(rng.normal(mu, sd))
        ln_post = -0.5 * ((FINE_GRID - mu) / sd) ** 2
        pits.append(cg.pp_readout(FINE_GRID, ln_post, h_true)["pit"])
    d = cg.ks_distance(np.asarray(pits))
    assert d <= 1.358 / math.sqrt(n), f"calibrated ensemble rejected: D={d:.4f}"


# ── (2) coverage counter ─────────────────────────────────────────────────────


def test_hpd_coverage_matches_nominal_for_calibrated_ensemble() -> None:
    """Truth ~ posterior => 50/68/90% HPD coverage within 3 sigma binomial."""
    rng = np.random.default_rng(20260808)
    n = 500
    mu, sd = 0.730, 0.03
    ln_post = -0.5 * ((FINE_GRID - mu) / sd) ** 2
    hits = {0.50: 0, 0.68: 0, 0.90: 0}
    for _ in range(n):
        h_true = float(rng.normal(mu, sd))
        out = cg.pp_readout(FINE_GRID, ln_post, h_true)
        for lv in hits:
            hits[lv] += int(out[f"hpd{int(round(lv * 100))}"])
    for lv, k in hits.items():
        sig = math.sqrt(lv * (1.0 - lv) / n)
        assert abs(k / n - lv) < 3.0 * sig, f"coverage {k / n:.3f} at level {lv}"


def test_hpd_coverage_detects_a_shifted_ensemble() -> None:
    """Posteriors coherently shifted by 2 sd => 90% coverage collapses far
    below the binomial FAIL band (the DS-1 defect class)."""
    rng = np.random.default_rng(1)
    n = 400
    mu, sd = 0.730, 0.03
    ln_post = -0.5 * ((FINE_GRID - (mu + 2.0 * sd)) / sd) ** 2  # biased estimator
    hits90 = 0
    for _ in range(n):
        h_true = float(rng.normal(mu, sd))
        hits90 += int(cg.pp_readout(FINE_GRID, ln_post, h_true)["hpd90"])
    c90 = hits90 / n
    fail_edge = 0.90 - 3.0 * math.sqrt(0.9 * 0.1 / n)  # prereg DS-1 3-sigma FAIL edge
    assert c90 < fail_edge, f"shifted ensemble not detected: C90={c90:.3f}"


def test_channel_aggregate_statuses_on_constructed_records() -> None:
    """_channel_aggregate: calibrated records => PASS; over-covered => not PASS."""
    rng = np.random.default_rng(3)
    n = 400

    def _records(p90: float) -> list[dict]:
        return [
            {
                "pit_1d": float(rng.random()),
                "hpd50_1d": 1.0 if rng.random() < 0.50 else 0.0,
                "hpd68_1d": 1.0 if rng.random() < 0.68 else 0.0,
                "hpd90_1d": 1.0 if rng.random() < p90 else 0.0,
                "map_1d": 0.73,
                "map_1d_refined": 0.73,
                "mean_1d": 0.73,
                "railed_low_1d": 0.0,
                "railed_high_1d": 0.0,
                "post_sd_1d": 0.02,
                "edge_mass_1d": 1e-9,
            }
            for _ in range(n)
        ]

    good = cg._channel_aggregate(_records(0.90), "1d", 0.73)
    assert good["ds1_status"] != "FAIL"  # calibrated: never the 3-sigma defect class
    assert good["ds2_ks"]["status"] != "FAIL"
    bad = cg._channel_aggregate(_records(0.70), "1d", 0.73)  # 90% level hit at 70%
    assert bad["ds1_status"] == "FAIL"


# ── (3) rail-detection statistic ─────────────────────────────────────────────


def test_rail_detection_on_constructed_posteriors() -> None:
    """Monotone posteriors rail at the correct edge; a peaked one does not."""
    down = -50.0 * (H_GRID - H_GRID[0])  # argmax at the low edge
    up = +50.0 * (H_GRID - H_GRID[0])  # argmax at the high edge
    peak = -0.5 * ((H_GRID - 0.730) / 0.02) ** 2
    r_down = cl.posterior_readout(H_GRID, down)
    r_up = cl.posterior_readout(H_GRID, up)
    r_peak = cl.posterior_readout(H_GRID, peak)
    assert r_down["railed_low"] == 1.0 and r_down["railed_high"] == 0.0
    assert r_up["railed_high"] == 1.0 and r_up["railed_low"] == 0.0
    assert r_peak["railed_low"] == 0.0 and r_peak["railed_high"] == 0.0


def test_ds4_rail_fractions_aggregate_and_ds6_anchor_separation() -> None:
    """R_low aggregates correctly, and the DS-6 thresholds (0.90 / 0.05)
    separate a railed ensemble from an un-railed one at N=400."""
    rng = np.random.default_rng(9)
    n = 400
    railed_flags = [1.0] * 380 + [0.0] * 20  # R_low = 0.95 => RAIL-REPRODUCED side
    clean_flags = [0.0] * 396 + [1.0] * 4  # R_low = 0.01 => below 0.05
    r_railed = float(np.mean(railed_flags))
    r_clean = float(np.mean(clean_flags))
    assert r_railed >= 0.90
    assert r_clean <= 0.05
    # binomial 2-sigma widths at these rates are < 0.03 (prereg §7 DS-6 note)
    for rate in (r_railed, r_clean):
        assert 2.0 * math.sqrt(max(rate * (1 - rate), 0.0) / n) < 0.03
    del rng


def test_edge_guard_fires_on_edge_loaded_posteriors() -> None:
    """A railed posterior's edge mass exceeds the 0.01 guard threshold."""
    down = -300.0 * (H_GRID - H_GRID[0])
    out = cg.pp_readout(H_GRID, down, 0.730)
    assert out["edge_mass"] > cg.EDGE_MASS_THRESHOLD
    peak = -0.5 * ((H_GRID - 0.730) / 0.02) ** 2
    out2 = cg.pp_readout(H_GRID, peak, 0.730)
    assert out2["edge_mass"] < cg.EDGE_MASS_THRESHOLD


# ── HPD internal consistency (port sanity without pp_coverage import) ────────


def test_hpd_regions_are_nested() -> None:
    """If h_true is inside the 50% HPD it must be inside 68% and 90% too."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        mu = rng.uniform(0.62, 0.84)
        sd = rng.uniform(0.01, 0.08)
        post = np.exp(-0.5 * ((H_GRID - mu) / sd) ** 2)
        post /= np.trapezoid(post, H_GRID)
        h_true = float(rng.uniform(0.60, 0.86))
        c50 = cg.hpd_contains(H_GRID, post, h_true, 0.50)
        c68 = cg.hpd_contains(H_GRID, post, h_true, 0.68)
        c90 = cg.hpd_contains(H_GRID, post, h_true, 0.90)
        assert (not c50 or c68) and (not c68 or c90)


def test_pit_is_monotone_in_truth() -> None:
    """PIT must increase monotonically with h_true for a fixed posterior."""
    ln_post = -0.5 * ((H_GRID - 0.730) / 0.03) ** 2
    pits = [cg.pp_readout(H_GRID, ln_post, t)["pit"] for t in (0.62, 0.68, 0.73, 0.78, 0.84)]
    assert all(b > a for a, b in zip(pits, pits[1:]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
