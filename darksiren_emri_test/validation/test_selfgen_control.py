"""Fast, pool-free tests for C-SG v3 (:mod:`darksiren_emri.validation.selfgen_control`).

Mirrors ``test_correspondence_1d.py``'s discipline: a lightweight FAKE
completeness/detection-probability pair stands in for the real (pool-backed)
production objects everywhere the generator/gates need one, so the whole
suite runs in well under a few seconds and needs no real injection pool for
its FAKE-injected paths. The one exception, :func:`test_matched_channel_reproduces_decompose_matched_channel`,
reads a REAL banked diagnostics CSV (``results/prod2d_closure_20260818/
arm_event_likelihoods/bsel_seed900101/...``) already committed to the repo --
no live pool construction, just a CSV read.

HARD CONSTRAINT (carried from the launch task): no test in this file runs the
registered C-SG measurement -- no scoring of C-SG seeds against a band, no
``mean_h`` fleet statement. Every test exercises MACHINERY at tiny n.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from darksiren_emri.validation import correspondence_1d as c1d
from darksiren_emri.validation import selfgen_control as sg

_N_DONOR = 200
_N_PIX = 12


class _FakeCompleteness:
    """Isotropic, constant-``f_k`` stand-in satisfying :class:`sg.CsgCompletenessModel`."""

    npix = _N_PIX

    def __init__(self, f_value: float = 0.3) -> None:
        self.f_value = f_value

    def f_k(
        self, z: float | npt.NDArray[np.floating[Any]], k: int, h: float = 0.73
    ) -> float | npt.NDArray[np.float64]:
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        out = np.full_like(z_arr, self.f_value)
        return out if np.ndim(z) else float(out[0])

    def f_bar(
        self, z: float | npt.NDArray[np.floating[Any]], h: float = 0.73
    ) -> float | npt.NDArray[np.floating[Any]]:
        return self.f_k(z, 0, h)

    def ang2pix(self, phi: float, theta: float) -> int:
        return 0

    def get_completeness_at_redshift(
        self, z: float | npt.NDArray[np.floating[Any]], h: float = 0.73
    ) -> float | npt.NDArray[np.floating[Any]]:
        return self.f_bar(z, h)

    def pixel_dark_weights(
        self, z_grid: npt.NDArray[np.float64], p_pop: npt.NDArray[np.float64], h: float
    ) -> npt.NDArray[np.float64]:
        w = float(np.trapezoid((1.0 - self.f_value) * p_pop, z_grid))
        return np.full(self.npix, w, dtype=np.float64)

    def sample_sky_in_pixels(
        self, pix: npt.NDArray[np.int_], rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        n = int(pix.size)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        theta = rng.uniform(0.1, np.pi - 0.1, size=n)
        return phi, theta


class _FakeDetectionProbability:
    """Constant-``S_4D`` stand-in; records every call for the single-selection test."""

    def __init__(self, s4d: float = 1.0, dl_max: float = 50.0) -> None:
        self.s4d = s4d
        self.dl_max = dl_max
        self.calls: list[int] = []  # batch sizes passed, one entry per call

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        M_z: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
        z: float | npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        d_l_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        self.calls.append(int(d_l_arr.size))
        return np.full_like(d_l_arr, self.s4d)

    def get_dl_max(self, h: float) -> float:
        return self.dl_max


def _make_donor_csv(path: Path, n_rows: int = _N_DONOR, seed: int = 1234) -> Path:
    rng = np.random.default_rng(seed)
    cols: dict[str, Any] = {
        "SNR": rng.uniform(20.0, 80.0, n_rows),
        "luminosity_distance": rng.uniform(1.0, 3.0, n_rows),
        "phiS": rng.uniform(0.0, 2 * np.pi, n_rows),
        "qS": rng.uniform(0.1, np.pi - 0.1, n_rows),
        "M": rng.uniform(1.0e4, 1.0e6, n_rows),
        "delta_M_delta_M": rng.uniform(1.0e6, 1.0e8, n_rows),
        "delta_luminosity_distance_delta_luminosity_distance": rng.uniform(0.001, 0.05, n_rows),
        "delta_phiS_delta_phiS": rng.uniform(1.0e-4, 1.0e-2, n_rows),
        "delta_qS_delta_qS": rng.uniform(1.0e-4, 1.0e-2, n_rows),
        "delta_phiS_delta_qS": np.zeros(n_rows),
        "delta_phiS_delta_M": rng.uniform(-1.0, 1.0, n_rows),
        "delta_qS_delta_M": rng.uniform(-1.0, 1.0, n_rows),
        "delta_luminosity_distance_delta_M": rng.uniform(-1.0, 1.0, n_rows),
        "host_galaxy_index": -1,
        "in_catalog": False,
        "_coord_frame": "ecliptic_BarycentricTrue_J2000",
        "_cov_frame": "ecliptic_BarycentricTrue_J2000",
    }
    for c in sg.DL_CROSS_COV_COLUMNS:
        cols.setdefault(c, rng.uniform(-1.0, 1.0, n_rows))
    df = pd.DataFrame(cols)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def donor_rows(tmp_path: Path) -> pd.DataFrame:
    path = _make_donor_csv(tmp_path / "donor.csv")
    return pd.read_csv(path)


@pytest.fixture
def fake_completeness() -> _FakeCompleteness:
    return _FakeCompleteness()


@pytest.fixture
def fake_det_full_accept() -> _FakeDetectionProbability:
    return _FakeDetectionProbability(s4d=1.0)


# ── Registry plumbing (A13: each labelled arm carries its intended value) ───


def test_csg_registries_are_consistent() -> None:
    assert set(sg.CSG_H_GEN) == set(sg.CSG_SIGMA_MODE) == set(sg.CSG_SEEDS) == set(sg.CSG_ARMS)
    assert sg.CSG_H_GEN == {"csgf": 0.73, "csge": 0.73, "csgdm": 0.68, "csgdp": 0.78}
    assert sg.CSG_SIGMA_MODE == {
        "csgf": "fixed",
        "csge": "empirical",
        "csgdm": "fixed",
        "csgdp": "fixed",
    }
    assert len(sg.CSG_SEEDS["csgf"]) == 15
    assert len(sg.CSG_SEEDS["csge"]) == 15
    assert len(sg.CSG_SEEDS["csgdm"]) == 8
    assert len(sg.CSG_SEEDS["csgdp"]) == 8
    total = sum(len(v) for v in sg.CSG_SEEDS.values())
    assert total == 46  # prereg section 4: 15+15+8+8
    for arm, seeds in sg.CSG_SEEDS.items():
        assert seeds[0] == 910101, arm
        assert list(seeds) == sorted(seeds), arm
        assert len(set(seeds)) == len(seeds), arm
    # 0.73 (csgf/csge) and the two delta arms (0.68/0.78) are all INTERIOR
    # nodes of H_GRID_41 -- required for score_at_h_gen's central difference.
    grid = set(c1d.H_GRID_41)
    for h_gen in set(sg.CSG_H_GEN.values()):
        assert h_gen in grid


def test_h_gen_for_arm_rejects_unregistered() -> None:
    with pytest.raises(KeyError):
        sg._h_gen_for_arm("not-a-real-arm")


def test_dl_cross_cov_columns_match_pinned_csv_header() -> None:
    """Every rescale column must exist in the real pinned CRB CSV's header."""
    header = pd.read_csv(c1d.CRB_CSV_PATH, nrows=0).columns
    for col in sg.DL_CROSS_COV_COLUMNS:
        assert col in header, col
    assert "delta_luminosity_distance_delta_luminosity_distance" not in sg.DL_CROSS_COV_COLUMNS
    assert "luminosity_distance" not in sg.DL_CROSS_COV_COLUMNS
    assert len(sg.DL_CROSS_COV_COLUMNS) == len(set(sg.DL_CROSS_COV_COLUMNS))


# ── Kernel-direction: the draw is LINEAR, not the ratio form (prereg section 0 item 1) ──


def test_kernel_direction_is_linear_gaussian(
    donor_rows: pd.DataFrame,
    fake_completeness: _FakeCompleteness,
    fake_det_full_accept: _FakeDetectionProbability,
) -> None:
    """``d_hat - d_L_true`` normalized by ``sigma_dL`` is standard-normal-shaped.

    A ratio-kernel draw (the v1 defect the prereg's section 0 item 1
    overturned) would instead produce residuals whose scale grows with
    ``d_L`` in a 1/h-like way and would NOT be centered at zero in the
    LINEAR distance residual -- this test is exactly the falsifier for that
    defect, at machine-checkable scale (n=4000, well beyond the per-event
    noise floor).
    """
    n = 4000
    rows, diag = sg.draw_csg_realization(
        7, "csgf", n, fake_completeness, fake_det_full_accept, donor_rows
    )
    d_hat = rows["luminosity_distance"].to_numpy(dtype=np.float64)
    d_l_true = diag["d_L_true"]
    sigma_dl = diag["sigma_dL"]
    resid = (d_hat - d_l_true) / sigma_dl
    # Standard normal: mean ~0, std ~1, well within tolerance at n=4000.
    assert abs(float(np.mean(resid))) < 0.05
    assert abs(float(np.std(resid)) - 1.0) < 0.05
    # sigma_dL is exactly the STATED fixed fraction of TRUE d_L (never d_hat).
    np.testing.assert_allclose(sigma_dl, sg.SIGMA_FRAC_FIXED * d_l_true, rtol=0, atol=0)


def test_f_arm_sigma_is_exactly_fixed_fraction_of_true_dl(
    donor_rows: pd.DataFrame,
    fake_completeness: _FakeCompleteness,
    fake_det_full_accept: _FakeDetectionProbability,
) -> None:
    """C-SG-F: ``sigma_dL := 0.0373 * d_L(z;h_gen)`` -- stated, not ``0.0373*d_hat``."""
    rows, diag = sg.draw_csg_realization(
        11, "csgf", 30, fake_completeness, fake_det_full_accept, donor_rows
    )
    np.testing.assert_array_equal(diag["sigma_dL"], sg.SIGMA_FRAC_FIXED * diag["d_L_true"])
    d_hat = rows["luminosity_distance"].to_numpy(dtype=np.float64)
    # sigma_dL must NOT equal 0.0373 * d_hat (the refuted ratio-adjacent form).
    assert not np.allclose(diag["sigma_dL"], sg.SIGMA_FRAC_FIXED * d_hat)
    np.testing.assert_array_equal(
        rows["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64),
        diag["sigma_dL"] ** 2,
    )


def test_e_arm_sigma_is_iid_empirical_independent_of_z(
    donor_rows: pd.DataFrame,
    fake_completeness: _FakeCompleteness,
    fake_det_full_accept: _FakeDetectionProbability,
) -> None:
    """C-SG-E: sigma_frac drawn i.i.d. from the pinned pool, independent of z."""
    rows, diag = sg.draw_csg_realization(
        11, "csge", 200, fake_completeness, fake_det_full_accept, donor_rows
    )
    sigma_frac = diag["sigma_dL"] / diag["d_L_true"]
    pool = sg._empirical_sigma_frac_pool()
    assert sigma_frac.min() >= pool.min() - 1e-12
    assert sigma_frac.max() <= pool.max() + 1e-12
    # Independent of z: correlation should be small (unlike production's 0.656).
    corr = float(np.corrcoef(sigma_frac, diag["z_true"])[0, 1])
    assert abs(corr) < 0.3


# ── Single-selection: S_4D applied exactly once, no S_bar_phi factor ────────


def test_single_selection_full_accept_matches_proposal_marginal(
    donor_rows: pd.DataFrame,
    fake_completeness: _FakeCompleteness,
    fake_det_full_accept: _FakeDetectionProbability,
) -> None:
    """With S_4D == 1 everywhere, EVERY candidate is accepted -- one batch, no rejection."""
    n = 500
    rows, diag = sg.draw_csg_realization(
        3, "csgf", n, fake_completeness, fake_det_full_accept, donor_rows
    )
    assert diag["n_batches"] == 1
    assert diag["accept_rate"] == pytest.approx(1.0)
    assert diag["n_candidates_drawn"] >= n
    # S_4D was queried exactly once per drawn candidate (accept ONCE, prereg
    # section 0 item 2/section 2 stage 3) -- not twice, not zero times.
    assert sum(fake_det_full_accept.calls) == diag["n_candidates_drawn"]
    assert len(fake_det_full_accept.calls) == diag["n_batches"]


def test_single_selection_zero_accept_raises_with_diagnostic(
    donor_rows: pd.DataFrame, fake_completeness: _FakeCompleteness
) -> None:
    """S_4D == 0 everywhere: no hidden S_bar_phi factor can rescue acceptance -- must raise."""
    det = _FakeDetectionProbability(s4d=0.0)
    with pytest.raises(RuntimeError, match="accept/reject failed"):
        sg.draw_csg_realization(
            5, "csgf", 20, fake_completeness, det, donor_rows, max_batches=3, batch_floor=32
        )
    # The cap was actually exercised (not raised on the first candidate).
    assert len(det.calls) == 3


def test_single_selection_partial_accept_rate_matches_s4d(
    donor_rows: pd.DataFrame, fake_completeness: _FakeCompleteness
) -> None:
    """A constant S_4D < 1 must be recovered as the empirical accept rate (large n)."""
    det = _FakeDetectionProbability(s4d=0.4)
    n = 300
    _rows, diag = sg.draw_csg_realization(
        9, "csgf", n, fake_completeness, det, donor_rows, oversample_factor=6, batch_floor=512
    )
    assert diag["accept_rate"] == pytest.approx(0.4, abs=0.06)


# ── h_gen threading (0.68/0.78 reach dist_vectorized and the selection-object build) ──


def test_h_gen_threads_to_dist_vectorized(
    donor_rows: pd.DataFrame,
    fake_completeness: _FakeCompleteness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_h: list[float] = []
    real_dist_vectorized = sg.dist_vectorized

    def _recording_dist_vectorized(z: Any, h: float, **kwargs: Any) -> Any:
        seen_h.append(h)
        return real_dist_vectorized(z, h=h, **kwargs)

    monkeypatch.setattr(sg, "dist_vectorized", _recording_dist_vectorized)
    det = _FakeDetectionProbability(s4d=1.0)

    for arm, h_gen in (("csgdm", 0.68), ("csgdp", 0.78)):
        seen_h.clear()
        sg.draw_csg_realization(21, arm, 10, fake_completeness, det, donor_rows)
        assert seen_h, arm
        assert all(h == h_gen for h in seen_h), (arm, seen_h)


def test_h_gen_threads_to_build_csg_selection_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[float] = []
    real_sdp = sg.SimulationDetectionProbability

    class _RecordingSDP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._real = object.__new__(real_sdp)

        def _get_or_build_grid(self, h: float) -> None:
            captured.append(h)

    monkeypatch.setattr(sg, "SimulationDetectionProbability", _RecordingSDP)
    monkeypatch.setattr(sg, "from_cache_or_build", lambda: object())
    sg.build_csg_selection_objects.cache_clear()
    for h_gen in (0.68, 0.78, 0.73):
        sg.build_csg_selection_objects(h_gen=h_gen)
    assert set(captured) == {0.68, 0.78, 0.73}
    sg.build_csg_selection_objects.cache_clear()


def test_gate_h_reports_h_gen_threading(
    donor_rows: pd.DataFrame,
    fake_completeness: _FakeCompleteness,
    fake_det_full_accept: _FakeDetectionProbability,
) -> None:
    for arm, h_gen in sg.CSG_H_GEN.items():
        report = sg.gate_h(
            arm,
            910101,
            n_events=10,
            completeness=fake_completeness,
            detection_probability=fake_det_full_accept,
            skip_s_bar_phi_diagnostic=True,
        )
        assert report["h_gen"] == h_gen
        assert report["threading"]["joint_z_omega_draw_h"] == h_gen
        assert report["threading"]["dist_vectorized_h"] == h_gen
        assert report["threading"]["build_csg_selection_objects_h_gen"] == h_gen
        assert report["s_bar_phi_z_max"] is None
        assert report["allow_low_pdet_coverage_silences_stop"] is True
        assert "design_deviation_gate_h_anchors" in report


# ── GATE Q: cross-covariance rescale keeps a contrived Fisher PD ────────────


def test_gate_q_before_after_rescale_reduces_nonpd_attrition(
    fake_completeness: _FakeCompleteness, tmp_path: Path
) -> None:
    """A DELIBERATELY near-degenerate donor pool: rescale must not make attrition worse.

    Uses large ``M``-cross-covariances relative to the (tiny, forced) new
    ``sigma_dL`` so the UN-rescaled 4D block is likely to fail the
    condition-number/PD gate for at least some rows; the rescaled block uses
    the SAME correlation structure at the correct (small) scale.
    """
    n = 200
    rng = np.random.default_rng(99)
    cols: dict[str, Any] = {
        "SNR": rng.uniform(20.0, 80.0, n),
        "luminosity_distance": rng.uniform(1.0, 3.0, n),
        "phiS": rng.uniform(0.0, 2 * np.pi, n),
        "qS": rng.uniform(0.1, np.pi - 0.1, n),
        "M": rng.uniform(1.0e4, 1.0e6, n),
        "delta_M_delta_M": np.full(n, 1.0e10),  # huge M variance
        # Huge donor sigma_dL (will be replaced by a much smaller forced sigma_dL).
        "delta_luminosity_distance_delta_luminosity_distance": np.full(n, 4.0),
        "delta_phiS_delta_phiS": rng.uniform(1.0e-4, 1.0e-2, n),
        "delta_qS_delta_qS": rng.uniform(1.0e-4, 1.0e-2, n),
        "delta_phiS_delta_qS": np.zeros(n),
        "delta_phiS_delta_M": rng.uniform(-1.0, 1.0, n),
        "delta_qS_delta_M": rng.uniform(-1.0, 1.0, n),
        # Near-maximal correlation with the (huge) donor sigma_dL and (huge) M sigma:
        # cov(d_L, M) ~ 0.999 * sigma_dL_donor * sigma_M -- close to the
        # Cauchy-Schwarz edge, so an UNSCALED sigma_dL swap (new sigma_dL far
        # smaller than the donor's) breaks the correlation-implied bound and
        # is likely non-PD; the PROPORTIONAL rescale preserves the implied
        # correlation and stays valid.
        "delta_luminosity_distance_delta_M": 0.999 * np.sqrt(4.0) * np.sqrt(1.0e10) * np.ones(n),
        "host_galaxy_index": -1,
        "in_catalog": False,
        "_coord_frame": "ecliptic_BarycentricTrue_J2000",
        "_cov_frame": "ecliptic_BarycentricTrue_J2000",
    }
    for c in sg.DL_CROSS_COV_COLUMNS:
        cols.setdefault(c, np.zeros(n))
    donor = pd.DataFrame(cols)

    det = _FakeDetectionProbability(s4d=1.0)
    report = sg.gate_q(
        "csgf", 12345, n_events=n, completeness=fake_completeness, detection_probability=det
    )
    # csgf forces sigma_dL = 0.0373 * d_L (order 0.03-0.1 Gpc), FAR below the
    # donor's sigma_dL=2.0 -- the un-rescaled cross-covariance is far too
    # large relative to the new (tiny) variance, so this is expected to
    # produce MORE non-PD attrition before the rescale than after it.
    assert (
        report["after_rescale"]["fisher_nonpd"]["fraction"]
        <= report["before_rescale"]["fisher_nonpd"]["fraction"]
    )


def test_fisher_pd_exclusion_count_zero_for_diagonal_covariance() -> None:
    """A purely diagonal (zero cross-covariance) row set is always PD; count must be 0."""
    n = 50
    rng = np.random.default_rng(3)
    rows = pd.DataFrame(
        {
            "luminosity_distance": rng.uniform(1.0, 3.0, n),
            "M": rng.uniform(1.0e4, 1.0e6, n),
            "delta_phiS_delta_phiS": rng.uniform(1.0e-4, 1.0e-2, n),
            "delta_qS_delta_qS": rng.uniform(1.0e-4, 1.0e-2, n),
            "delta_phiS_delta_qS": np.zeros(n),
            "delta_phiS_delta_luminosity_distance": np.zeros(n),
            "delta_qS_delta_luminosity_distance": np.zeros(n),
            "delta_luminosity_distance_delta_luminosity_distance": rng.uniform(0.001, 0.05, n),
            "delta_phiS_delta_M": np.zeros(n),
            "delta_qS_delta_M": np.zeros(n),
            "delta_luminosity_distance_delta_M": np.zeros(n),
            "delta_M_delta_M": rng.uniform(1.0e6, 1.0e8, n),
        }
    )
    assert sg._fisher_pd_exclusion_count(rows) == 0


# ── GATE D / GATE V return-field checks ──────────────────────────────────────


def test_gate_d_returns_registered_fields(
    donor_rows: pd.DataFrame,
    fake_completeness: _FakeCompleteness,
    fake_det_full_accept: _FakeDetectionProbability,
) -> None:
    report = sg.gate_d(
        "csgf",
        910101,
        n_events=30,
        n_model_grid=101,
        completeness=fake_completeness,
        detection_probability=fake_det_full_accept,
        skip_model_density=True,
    )
    for key in (
        "arm",
        "seed",
        "h_gen",
        "n_drawn",
        "n_surviving",
        "survival_fraction",
        "max_cdf_gap_surviving_vs_model",
        "max_cdf_gap_drawn_vs_model",
        "d_crit_alpha",
        "d_crit_at_n_surviving",
        "verdict",
    ):
        assert key in report, key
    assert report["verdict"] is None  # skipped
    assert report["n_drawn"] == 30
    assert 0 <= report["n_surviving"] <= 30


def test_ks_d_crit_matches_registered_value_at_n200() -> None:
    """GATE D band: D_crit(5%) at n=200 == 0.0960 (prereg section 6)."""
    assert sg.ks_d_crit(0.05, 200) == pytest.approx(0.0960, abs=5e-4)


def test_ks_d_crit_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        sg.ks_d_crit(0.0, 200)
    with pytest.raises(ValueError):
        sg.ks_d_crit(0.05, 0)


def test_gate_v_span_and_sigma_pass_fields() -> None:
    grid = np.array(sorted(c1d.H_GRID_41), dtype=np.float64)
    n_events, n_nodes = 40, grid.size
    rng = np.random.default_rng(0)
    # A strongly peaked likelihood (informative) -> should pass GATE V.
    peak_idx = int(np.argmin(np.abs(grid - 0.73)))
    vals = np.empty((n_events, n_nodes))
    for i in range(n_events):
        vals[i] = np.exp(-0.5 * ((np.arange(n_nodes) - peak_idx) / 2.0) ** 2) + 1e-6
        vals[i] *= rng.uniform(0.9, 1.1)
    stats = c1d.seed_statistics_from_matrix(vals, 1, grid, h_true=0.73)
    report = sg.gate_v(vals, stats)
    assert report["span_nats"] > 0
    assert "sigma_prior_convention" in report
    assert isinstance(report["pass"], bool)

    # A flat (uninformative) likelihood -> should FAIL the span check.
    flat_vals = np.ones((n_events, n_nodes))
    flat_stats = c1d.seed_statistics_from_matrix(flat_vals, 1, grid, h_true=0.73)
    flat_report = sg.gate_v(flat_vals, flat_stats)
    assert flat_report["span_nats"] == pytest.approx(0.0, abs=1e-9)
    assert flat_report["span_pass"] is False
    assert flat_report["pass"] is False


# ── Matched-channel helper reproduces decompose_matched_channel.py exactly ──

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BSEL_SEED900101_CSV = (
    _REPO_ROOT
    / "results/prod2d_closure_20260818/arm_event_likelihoods/bsel_seed900101/seed900101"
    / "simulations/diagnostics/event_likelihoods.csv"
)


@pytest.mark.skipif(
    not _BSEL_SEED900101_CSV.is_file(), reason="banked bsel_seed900101 fixture not present"
)
def test_matched_channel_reproduces_decompose_matched_channel() -> None:
    """Reproduces ``decompose_matched_channel_output.json``'s bsel_seed900101 per-seed row.

    Reference (banked ``decompose_matched_channel_output.json``, per_seed[0]):
    ``full_mean_h=0.6224394129480753``, matched
    ``mean_h=0.6455447092781811``, ``sigma_h=0.038642223089036316``,
    ``map_h=0.6``, ``c68=False``, ``n_events=174``.
    """
    scores = sg.csg_channel_scores(
        _BSEL_SEED900101_CSV, seed=900101, h_grid=c1d.H_GRID_41, h_true=c1d.H_TRUE
    )
    assert scores["full"]["n_events"] == 174
    assert scores["full"]["mean_h"] == pytest.approx(0.6224394129480753, rel=1e-9)
    matched = scores["matched"]
    assert matched["n_events"] == 174
    assert matched["mean_h"] == pytest.approx(0.6455447092781811, rel=1e-9)
    assert matched["sigma_h"] == pytest.approx(0.038642223089036316, rel=1e-9)
    assert matched["map_h"] == pytest.approx(0.6, abs=1e-12)
    assert matched["c68"] is False
    assert scores["gate_t"]["pass"] is True
    assert scores["gate_t"]["max_rel_spread"] == pytest.approx(0.0, abs=1e-9)


# ── FIX ROUND (adversarial finding #2): regression guard for the delta arms'
# h_gen threading. csg_channel_scores defaults h_true=H_TRUE(0.73); every
# OTHER test in this file only exercises h_gen=0.73 arms (csgf) or an
# explicit h_true=c1d.H_TRUE, so a future refactor that dropped the
# `h_true=h_gen` kwarg at run_csg_arm_seed's csg_channel_scores call site
# would be invisible to the whole suite for exactly the csgdm/csgdp delta
# arms it would corrupt (the launch task's named "A13 false-verdict mode").
# This test scores a SHARPLY h=0.73-peaked synthetic likelihood -- deliberately
# far from BOTH delta arms' h_gen (0.68/0.78) -- so h_true=0.73 (the correct
# value for csgf/csge, and the WRONG default-leak value for csgdm/csgdp)
# falls inside every HPD set while h_true=0.68/0.78 (the correct csgdm/csgdp
# values) fall outside it: c50/c68/c90 flip if-and-only-if h_true is threaded
# correctly.
def _make_peaked_diagnostics_csv(path: Path, peak_h: float = 0.73, n_events: int = 30) -> Path:
    grid = np.array(sorted(c1d.H_GRID_41), dtype=np.float64)
    rows = []
    for h in grid:
        # Gaussian bump around peak_h, narrow enough that h in {0.68, 0.78}
        # (both >= 0.05 from 0.73) sit many sigma outside the HPD sets while
        # h=0.73 sits at the mode -- and both 0.68/0.78 are equidistant, so
        # the SAME synthetic CSV regression-guards both delta arms at once.
        bump = np.exp(-0.5 * ((h - peak_h) / 0.01) ** 2) + 1.0e-8
        for event_idx in range(n_events):
            rows.append(
                {
                    "event_idx": event_idx,
                    "h": h,
                    "alpha_G_phi": 0.0,
                    "D_tilde_phi": 1.0,
                    "B_num": bump,
                    "combined_no_bh": bump,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("h_gen", [0.68, 0.78])
def test_csg_channel_scores_h_true_not_silently_defaulted(tmp_path: Path, h_gen: float) -> None:
    """The A13 false-verdict-mode guard: h_true must thread to h_gen, not H_TRUE."""
    csv_path = _make_peaked_diagnostics_csv(tmp_path / "diag.csv", peak_h=0.73)

    scores_correct = sg.csg_channel_scores(csv_path, seed=1, h_grid=c1d.H_GRID_41, h_true=h_gen)
    scores_leaked = sg.csg_channel_scores(csv_path, seed=1, h_grid=c1d.H_GRID_41)  # default H_TRUE

    for channel in ("full", "matched", "pure"):
        # h_true=h_gen (0.68/0.78) is many sigma from the peak -> excluded
        # from every HPD set for a correctly-threaded call.
        assert scores_correct[channel]["c50"] is False, channel
        assert scores_correct[channel]["c68"] is False, channel
        assert scores_correct[channel]["c90"] is False, channel
        # A silently-defaulted H_TRUE=0.73 call sits AT the peak -> inside
        # every HPD set. If a future refactor drops `h_true=h_gen`, this
        # test's "scores_correct" branch would silently become identical to
        # "scores_leaked" and the assertions above would fail.
        assert scores_leaked[channel]["c50"] is True, channel
        assert scores_leaked[channel]["c68"] is True, channel
        assert scores_leaked[channel]["c90"] is True, channel
        assert scores_correct[channel]["c50"] != scores_leaked[channel]["c50"], channel


def test_run_csg_arm_seed_delta_arm_scores_against_own_h_gen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end guard: ``run_csg_arm_seed("csgdm", ...)`` must score at h_gen=0.68.

    Regression-guards the ACTUAL call site (``run_csg_arm_seed``'s
    ``csg_channel_scores(..., h_true=h_gen)``), not just the function default
    checked above -- catches the kwarg being dropped, not just renamed.
    """
    n_events = 8

    def _fake_build_csg_selection_objects(
        h_gen: float = c1d.H_TRUE, **_kwargs: Any
    ) -> tuple[_FakeCompleteness, _FakeDetectionProbability]:
        return _FakeCompleteness(), _FakeDetectionProbability(s4d=1.0)

    def _fake_load_handler(_path: str) -> object:
        return object()

    def _fake_run_mirror_seed_inprocess(
        work_root: Path,
        events: pd.DataFrame,
        seed: int,
        galaxy_catalog: object,
        h_values: tuple[float, ...] = c1d.H_GRID_41,
        **_kwargs: Any,
    ) -> tuple[Path, float]:
        rows = []
        for h in h_values:
            bump = np.exp(-0.5 * ((h - 0.73) / 0.01) ** 2) + 1.0e-8
            for event_idx in range(len(events)):
                rows.append(
                    {
                        "event_idx": event_idx,
                        "h": h,
                        "alpha_G_phi": 0.0,
                        "D_tilde_phi": 1.0,
                        "B_num": bump,
                        "combined_no_bh": bump,
                    }
                )
        work_root.mkdir(parents=True, exist_ok=True)
        diag_csv = work_root / "diag.csv"
        pd.DataFrame(rows).to_csv(diag_csv, index=False)
        return diag_csv, 0.02

    monkeypatch.setattr(sg, "build_csg_selection_objects", _fake_build_csg_selection_objects)
    monkeypatch.setattr(c1d, "_load_galaxy_catalog_handler", _fake_load_handler)
    monkeypatch.setattr(c1d, "run_mirror_seed_inprocess", _fake_run_mirror_seed_inprocess)
    monkeypatch.setattr(c1d, "check_reduced_catalogue_pin", lambda: True)
    monkeypatch.setattr(c1d, "check_crb_pin", lambda: True)

    out_dir = tmp_path / "out"
    work_root = tmp_path / "work"
    out_path = sg.run_csg_arm_seed(work_root, "csgdm", 910101, out_dir, n_events=n_events)
    record = json.loads(out_path.read_text())
    assert record["h_gen"] == 0.68
    for channel in ("full", "matched", "pure"):
        # h_gen=0.68 is far from the diagnostics' h=0.73 peak -> correctly
        # threaded h_true means h_gen is EXCLUDED from every HPD set. A
        # silently-defaulted H_TRUE=0.73 would instead sit AT the peak and
        # flip these to True -- the exact A13 false-verdict mode.
        assert record["channel_scores"][channel]["c50"] is False, channel
        assert record["channel_scores"][channel]["c68"] is False, channel
        assert record["channel_scores"][channel]["c90"] is False, channel


# ── run_csg_arm_seed: idempotency + banked-JSON shape (plumbing) ────────────


def test_run_csg_arm_seed_idempotent_and_banks_expected_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_events = 8

    def _fake_build_csg_selection_objects(
        h_gen: float = c1d.H_TRUE, **_kwargs: Any
    ) -> tuple[_FakeCompleteness, _FakeDetectionProbability]:
        return _FakeCompleteness(), _FakeDetectionProbability(s4d=1.0)

    def _fake_load_handler(_path: str) -> object:
        return object()

    def _fake_run_mirror_seed_inprocess(
        work_root: Path,
        events: pd.DataFrame,
        seed: int,
        galaxy_catalog: object,
        h_values: tuple[float, ...] = c1d.H_GRID_41,
        **_kwargs: Any,
    ) -> tuple[Path, float]:
        rows = []
        for h in h_values:
            for event_idx in range(len(events)):
                alpha = 0.3 + 0.01 * h
                dtil = 1.0 + 0.01 * h
                b_num = 0.5 + 0.1 * event_idx + 0.05 * h
                rows.append(
                    {
                        "event_idx": event_idx,
                        "h": h,
                        "alpha_G_phi": alpha,
                        "D_tilde_phi": dtil,
                        "B_num": b_num,
                        "combined_no_bh": b_num / dtil,
                    }
                )
        work_root.mkdir(parents=True, exist_ok=True)
        diag_csv = work_root / "diag.csv"
        pd.DataFrame(rows).to_csv(diag_csv, index=False)
        return diag_csv, 0.02

    monkeypatch.setattr(sg, "build_csg_selection_objects", _fake_build_csg_selection_objects)
    monkeypatch.setattr(c1d, "_load_galaxy_catalog_handler", _fake_load_handler)
    monkeypatch.setattr(c1d, "run_mirror_seed_inprocess", _fake_run_mirror_seed_inprocess)
    monkeypatch.setattr(c1d, "check_reduced_catalogue_pin", lambda: True)
    monkeypatch.setattr(c1d, "check_crb_pin", lambda: True)

    out_dir = tmp_path / "out"
    work_root = tmp_path / "work"
    out_path = sg.run_csg_arm_seed(work_root, "csgf", 910101, out_dir, n_events=n_events)
    assert out_path.is_file()
    record = json.loads(out_path.read_text())
    assert record["arm"] == "csgf"
    assert record["seed"] == 910101
    assert record["h_gen"] == 0.73
    assert record["sigma_mode"] == "fixed"
    assert record["n_events_drawn"] == n_events
    for key in (
        "channel_scores",
        "score_at_h_gen",
        "gate_v",
        "h_grid",
        "log_posterior_full_channel",
    ):
        assert key in record, key
    for channel in ("full", "matched", "pure"):
        assert channel in record["channel_scores"]
        assert channel in record["score_at_h_gen"]
    assert record["catalogue_pin_ok"] is True
    assert record["crb_pin_ok"] is True

    # Idempotent: a second call must NOT re-invoke run_mirror_seed_inprocess
    # (would raise/duplicate if it tried, since the fake writes fresh files
    # under work_root/seed<seed>; the real assertion is that the JSON content
    # is byte-identical and no exception is raised).
    mtime_before = out_path.stat().st_mtime_ns
    out_path_2 = sg.run_csg_arm_seed(work_root, "csgf", 910101, out_dir, n_events=n_events)
    assert out_path_2 == out_path
    assert out_path.stat().st_mtime_ns == mtime_before


def test_run_csg_arm_seed_rejects_unregistered_arm(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        sg.run_csg_arm_seed(tmp_path / "work", "not-an-arm", 1, tmp_path / "out")
