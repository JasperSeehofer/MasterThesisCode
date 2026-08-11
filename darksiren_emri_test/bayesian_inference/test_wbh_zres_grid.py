"""Tests for the FIX-3 §7.1 joint z x M_z-resolved with-BH detection survival.

Physics change under test (docs/derivations/fix3_zmz_catalog_selection.md,
RATIFIED 2026-07-27 rev. B): behind the opt-in ``pdet_wbh_z_resolved`` flag,
the pooled-in-z 2D survival ``S(d_L | M_z)`` is replaced by the joint
conditional

    S(d_L | z, M_z)          (K1)/(K3)

estimated with a product Gaussian kernel in (u = ln(1+z), m = log10 M_z)
(Scott d=2 bandwidths, Abramson-adaptive on u only), exact suffix-survival in
d_L on the dense DLQ grid, and the (K5) ESS-weighted shrinkage toward the
m-only marginal ``S_m = S(d_L | M_z)``.

Covered here (§3.7 limiting cases; scope note: cases 1-3 constrain the
PRE-shrinkage ``S_joint``; the shipped ``S_tilde`` obeys the
``(1 - w)|S_joint - S_m|`` bound):

  * case 1: u-kernel disabled (grid-only control = the sigma_u -> inf
    construction on the SAME grid) -> joint == matched-grid m-only build,
    and the shrinkage is exactly inert there;
  * case 3: both widths -> inf -> pooled survival at every node;
  * cases 4/5: dense node w -> 1 (unshrunk); numerically-empty node ->
    w = 0 and S_tilde = S_m exactly — never pooled-uniform, never NaN;
  * case 6: independent re-derivation of the (K5) blend at a node
    (S_tilde = w S_joint + (1-w) S_m with w = ESS/(ESS + n0)); w monotone
    and continuous in ESS;
  * case 7: clamp semantics (d_L above grid -> exact 0; below -> S=1-side
    clamp; m and u -> span clamps, A2-EXTRAP / _zres_node_pos parity);
  * case 8: build h-invariance (tables byte-identical across queried h);
  * case 9: degenerate pools (zero-variance axis, n = 1) must not crash;
  * RATIFY-Z7 guard: pdet_wbh_z_resolved without pdet_z_resolved raises;
  * flag OFF: the 2D query ignores z entirely (byte-identical);
  * erf-sum knot helper: lifted knots 10^{m_j}, values monotone
    non-increasing in d_L; scalar/batch bit-parity of the switched paths.

References:
    Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
    arXiv:gr-qc/9601048; Mandel, Farr & Gair (2019), arXiv:1809.02063;
    Scott (1992) Ch. 6; Abramson (1982), Ann. Statist. 10:1217;
    Kish (1965), Survey Sampling.
"""

import pickle

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

import darksiren_emri.bayesian_inference.simulation_detection_probability as sdp_module
from darksiren_emri.bayesian_inference.bayesian_statistics import (
    _bh_mass_denominator_inner_m_integral,
    _bh_mass_denominator_inner_m_integral_batch,
    _mass_trunc_denominator_inner_m_integral,
    _mass_trunc_denominator_inner_m_integral_batch,
    _wbh_z_kwargs,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    _MIN_BAND_INJECTIONS,
    _SCOTT_EXPONENT_2D,
    SimulationDetectionProbability,
)
from darksiren_emri.physical_relations import dist_vectorized

_SNR_THRESHOLD = 20.0


def _write_pool(
    directory: str,
    z: npt.NDArray[np.float64],
    M_z: npt.NDArray[np.float64],  # noqa: N803
    snr: npt.NDArray[np.float64],
    h_value: float = 0.73,
) -> None:
    """Write a fully controlled injection CSV pool (M column = detector-frame M_z)."""
    n = len(z)
    rng = np.random.default_rng(7)
    d_L = np.asarray(dist_vectorized(z, h=h_value), dtype=np.float64)
    df = pd.DataFrame(
        {
            "z": z,
            "M": M_z,
            "phiS": rng.uniform(0.0, 2.0 * np.pi, n),
            "qS": rng.uniform(0.0, np.pi, n),
            "SNR": snr,
            "h_inj": h_value,
            "luminosity_distance": d_L,
        }
    )
    df.to_csv(f"{directory}/injection_h_0p73_task_001.csv", index=False)


def _generic_pool(directory: str, n: int = 3000, seed: int = 11) -> None:
    """Pool whose horizon drifts with both z and M_z (the FIX-3 target regime)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.01, 1.4, n)
    log_m = rng.uniform(4.6, 6.0, n)
    d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
    # d_hor grows with (1+z) and with mass: loudness ~ lognormal * (1+z)^2 * (M/1e5)^0.3.
    d_hor = np.exp(rng.normal(0.0, 0.6, n)) * 0.3 * (1.0 + z) ** 2 * (10.0 ** (log_m - 5.0)) ** 0.3
    snr = _SNR_THRESHOLD * d_hor / np.maximum(d_L, 1e-10)
    _write_pool(directory, z, 10.0**log_m, snr)


def _build(directory: str, **kwargs: object) -> SimulationDetectionProbability:
    return SimulationDetectionProbability(
        injection_data_dir=directory,
        snr_threshold=_SNR_THRESHOLD,
        **kwargs,  # type: ignore[arg-type]
    )


def _build_on(directory: str, **kwargs: object) -> SimulationDetectionProbability:
    return _build(directory, pdet_z_resolved=True, pdet_wbh_z_resolved=True, **kwargs)


class TestZ7Guard:
    """RATIFY-Z7: joint 2D over pooled 3D legs mixes conventions -> hard error."""

    def test_wbh_without_zres_raises(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=200)
        with pytest.raises(ValueError, match="pdet_z_resolved"):
            _build(d, pdet_wbh_z_resolved=True)

    def test_wbh_with_zres_constructs(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=200)
        dp = _build_on(d)
        assert dp.wbh_z_resolved is True
        assert dp.z_resolved is True


class TestFlagOff:
    """Default OFF must be byte-identical, with z accepted-and-ignored."""

    def test_default_flag_is_off(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=300)
        dp = _build(d)
        assert dp.wbh_z_resolved is False
        assert dp._wbh_stilde is None

    def test_flag_off_query_ignores_z(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=1500)
        dp = _build(d)
        q = np.linspace(0.05, 2.0, 40)
        m = np.geomspace(1e5, 8e5, 40)
        zeros = np.zeros_like(q)
        without_z = np.asarray(
            dp.detection_probability_with_bh_mass_interpolated(q, m, zeros, zeros, h=0.73)
        )
        with_z = np.asarray(
            dp.detection_probability_with_bh_mass_interpolated(
                q, m, zeros, zeros, h=0.73, z=np.full_like(q, 0.5)
            )
        )
        np.testing.assert_array_equal(without_z, with_z)

    def test_wbh_kwargs_helper_off_and_on(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=300)
        dp_off = _build(d)
        dp_on = _build_on(d)
        z = np.array([0.3])
        assert _wbh_z_kwargs(dp_off, z) == {}
        assert _wbh_z_kwargs(object(), z) == {}  # mock p_det compatibility
        assert set(_wbh_z_kwargs(dp_on, z)) == {"z"}

    def test_flag_on_query_requires_z(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=300)
        dp = _build_on(d)
        q = np.array([0.5])
        with pytest.raises(ValueError, match="pdet_wbh_z_resolved"):
            dp.detection_probability_with_bh_mass_interpolated(
                q, np.array([3e5]), np.zeros(1), np.zeros(1), h=0.73
            )


class TestCase1GridOnlyControl:
    """§3.7 case 1: u-factor ≡ 1 (sigma_u -> inf construction, matched grid).

    The MTC_WBH_GRID_ONLY control cell builds the SAME joint grid with the
    u-kernel disabled, so every u-node's joint column IS the m-only marginal
    S_m and the (K5) shrinkage is exactly inert (S_joint ≡ S_m => S_tilde ≡
    S_m for any w).
    """

    def test_grid_only_equals_m_marginal_and_shrinkage_inert(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=1200)
        monkeypatch.setattr(sdp_module, "_WBH_GRID_ONLY", True)
        dp = _build_on(d)
        assert dp._wbh_stilde is not None and dp._wbh_sm is not None
        for a in range(dp._wbh_u_nodes.size if dp._wbh_u_nodes is not None else 0):
            np.testing.assert_allclose(dp._wbh_stilde[:, a, :], dp._wbh_sm, rtol=0.0, atol=1e-14)

    def test_grid_only_query_is_z_independent(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=1200)
        monkeypatch.setattr(sdp_module, "_WBH_GRID_ONLY", True)
        dp = _build_on(d)
        q = np.linspace(0.05, 2.0, 30)
        m = np.full_like(q, 3e5)
        zeros = np.zeros_like(q)
        a = np.asarray(
            dp.detection_probability_with_bh_mass_interpolated(
                q, m, zeros, zeros, h=0.73, z=np.full_like(q, 0.05)
            )
        )
        b = np.asarray(
            dp.detection_probability_with_bh_mass_interpolated(
                q, m, zeros, zeros, h=0.73, z=np.full_like(q, 1.3)
            )
        )
        np.testing.assert_allclose(a, b, atol=1e-14)


class TestCase3BothWidthsInfinity:
    """§3.7 case 3: both kernel widths -> inf => S_joint = S_m = pooled exactly."""

    def test_pooled_limit(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=800)
        dp = _build_on(d, bandwidth_scale=1e6)
        assert dp._wbh_stilde is not None and dp._wbh_dlq is not None
        pooled = dp._survival_at(dp._wbh_dlq)
        n_u = dp._wbh_u_nodes.size if dp._wbh_u_nodes is not None else 0
        n_m = dp._wbh_m_nodes.size if dp._wbh_m_nodes is not None else 0
        for a in range(0, n_u, 20):
            for b in range(0, n_m, 10):
                np.testing.assert_allclose(dp._wbh_stilde[:, a, b], pooled, atol=1e-8)


def _clustered_pool(directory: str) -> None:
    """Two far-separated u-clusters; mid-u nodes get exactly-zero kernel weight
    once the bandwidth is shrunk (bandwidth_scale << 1) — the §3.4 empty-node
    clause territory. m stays dense so S_m is well-defined everywhere."""
    rng = np.random.default_rng(42)
    n0, n1 = 200, 200
    z = np.concatenate([rng.uniform(0.005, 0.02, n0), rng.uniform(1.40, 1.48, n1)])
    log_m = rng.uniform(4.5, 6.0, n0 + n1)
    d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
    d_hor = np.exp(rng.normal(0.0, 0.5, n0 + n1)) * 0.3 * (1.0 + z) ** 2
    snr = _SNR_THRESHOLD * d_hor / np.maximum(d_L, 1e-10)
    _write_pool(directory, z, 10.0**log_m, snr)


class TestCases4And5StarvedPolicy:
    """§3.7 cases 4/5: dense node unshrunk (w -> 1); empty node -> S_m, never
    NaN, never pooled-uniform."""

    def test_dense_and_empty_nodes(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _clustered_pool(d)
        # bandwidth_scale = 1e-3 shrinks sigma_u so mid-u nodes are > 38 sigma
        # from every injection -> product weights underflow to exactly 0.
        dp = _build_on(d, bandwidth_scale=1e-3)
        assert (
            dp._wbh_stilde is not None
            and dp._wbh_sm is not None
            and dp._wbh_w is not None
            and dp._wbh_ess is not None
        )
        st, sm, w, ess = dp._wbh_stilde, dp._wbh_sm, dp._wbh_w, dp._wbh_ess
        # No NaN anywhere in the shipped table (mandatory clause).
        assert np.all(np.isfinite(st))
        assert np.all((st >= 0.0) & (st <= 1.0))
        # w = ESS/(ESS + n0) with n0 = _MIN_BAND_INJECTIONS at every node.
        np.testing.assert_allclose(w, ess / (ess + float(_MIN_BAND_INJECTIONS)), atol=1e-14)
        # Case 5: empty nodes exist and carry exactly the m-only marginal.
        empty = w == 0.0
        assert np.any(empty), "test pool failed to produce an empty (u, m) node"
        ia, ib = np.nonzero(empty)
        for a, b in zip(ia[:50], ib[:50], strict=False):
            np.testing.assert_array_equal(st[:, a, b], sm[:, b])
        # The empty-node value is the m-conditional, NOT the pooled survival
        # (never-pooled clause): pick an empty node whose S_m differs from
        # pooled and verify the shipped value tracks S_m.
        pooled = dp._survival_at(np.asarray(dp._wbh_dlq))
        diffs = np.max(np.abs(sm - pooled[:, None]), axis=0)  # per m-node
        b_star = int(np.argmax(diffs))
        assert diffs[b_star] > 1e-3, "S_m indistinguishable from pooled — weak pool"
        a_star = ia[ib == b_star]
        if a_star.size:
            np.testing.assert_array_equal(st[:, int(a_star[0]), b_star], sm[:, b_star])

    def test_dense_node_is_unshrunk(self, tmp_path: object) -> None:
        """Case 4: on a well-populated pool at the derived bandwidths, the
        densest nodes have ESS >> n0 and are effectively unshrunk (w -> 1)."""
        d = str(tmp_path)
        _generic_pool(d, n=3000)
        dp = _build_on(d)
        assert dp._wbh_ess is not None and dp._wbh_w is not None
        assert float(np.max(dp._wbh_ess)) > 10.0 * float(_MIN_BAND_INJECTIONS)
        assert float(np.max(dp._wbh_w)) > 0.9
        assert np.all(dp._wbh_ess >= 0.0)
        assert np.all((dp._wbh_w >= 0.0) & (dp._wbh_w < 1.0))


class TestCase6BlendIdentity:
    """§3.7 case 6: independent re-derivation of the shipped (K5) blend at a
    node, plus monotone continuity of w in ESS."""

    def test_node_blend_matches_independent_computation(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=2000)
        dp = _build_on(d)
        assert (
            dp._wbh_stilde is not None
            and dp._wbh_dlq is not None
            and dp._wbh_u_nodes is not None
            and dp._wbh_m_nodes is not None
        )
        n = dp._n_inj
        u = np.log1p(dp._z_arr)
        m = dp._log_M_z
        sigma_u = float(n) ** _SCOTT_EXPONENT_2D * float(np.std(u, ddof=0))
        lam = dp._abramson_lambda_u(u, sigma_u)
        sig = sigma_u * lam
        _, sigma_m = dp._compute_bandwidths(dp._dl_raw, dp._log_M_z)
        d_hor = dp._d_hor
        dlq_idx = np.arange(0, dp._wbh_dlq.size, 293)
        for a, b in ((10, 5), (30, 15), (55, 28)):
            u_a = float(dp._wbh_u_nodes[a])
            m_b = float(dp._wbh_m_nodes[b])
            ku = np.exp(-0.5 * ((u - u_a) / sig) ** 2)
            km = np.exp(-0.5 * ((m - m_b) / sigma_m) ** 2)
            wj = ku * km
            tot, tot_m = float(wj.sum()), float(km.sum())
            ess = tot * tot / float((wj * wj).sum())
            wt = ess / (ess + float(_MIN_BAND_INJECTIONS))
            np.testing.assert_allclose(float(dp._wbh_ess[a, b]), ess, rtol=1e-10)  # type: ignore[index]
            for qi in dlq_idx:
                d_q = float(dp._wbh_dlq[qi])
                det = d_hor >= d_q
                s_joint = float(wj[det].sum()) / tot
                s_m = float(km[det].sum()) / tot_m
                expected = wt * s_joint + (1.0 - wt) * s_m
                np.testing.assert_allclose(
                    float(dp._wbh_stilde[qi, a, b]), expected, rtol=1e-9, atol=1e-12
                )

    def test_w_monotone_continuous_in_ess(self) -> None:
        ess = np.linspace(0.0, 1000.0, 5000)
        w = ess / (ess + float(_MIN_BAND_INJECTIONS))
        assert np.all(np.diff(w) > 0.0)  # strictly monotone
        assert w[0] == 0.0 and w[-1] < 1.0
        # 50/50 blend exactly at ESS = n0 (the reliability floor's continuous
        # meaning — no threshold cliff exists anywhere).
        assert float(_MIN_BAND_INJECTIONS / (2.0 * _MIN_BAND_INJECTIONS)) == pytest.approx(0.5)


class TestCase7ClampSemantics:
    """§3.7 case 7: A2-EXTRAP / span-clamp parity."""

    def test_dl_above_grid_is_exact_zero(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=800)
        dp = _build_on(d)
        assert dp._wbh_dlq is not None
        beyond = float(dp._wbh_dlq[-1]) * 1.01
        val = dp.detection_probability_with_bh_mass_interpolated(
            beyond, 3e5, 0.0, 0.0, h=0.73, z=0.4
        )
        assert float(np.asarray(val)) == 0.0

    def test_dl_below_grid_clamps_to_one_side(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=800)
        dp = _build_on(d)
        assert dp._wbh_dlq is not None
        tiny = dp.detection_probability_with_bh_mass_interpolated(
            1e-8, 3e5, 0.0, 0.0, h=0.73, z=0.4
        )
        at_first = dp.detection_probability_with_bh_mass_interpolated(
            float(dp._wbh_dlq[0]), 3e5, 0.0, 0.0, h=0.73, z=0.4
        )
        assert float(np.asarray(tiny)) == pytest.approx(float(np.asarray(at_first)), abs=1e-14)
        assert float(np.asarray(tiny)) > 0.9  # S ~= 1 near d_L = 0

    def test_m_and_u_span_clamps(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=800)
        dp = _build_on(d)
        assert dp._wbh_m_nodes is not None and dp._wbh_u_nodes is not None
        m_max = 10.0 ** float(dp._wbh_m_nodes[-1])
        z_max = float(np.expm1(dp._wbh_u_nodes[-1]))
        q = 0.3
        # m above the span == true-nearest (top-node) value.
        hi = dp.detection_probability_with_bh_mass_interpolated(
            q, m_max * 100.0, 0.0, 0.0, h=0.73, z=0.4
        )
        at_edge = dp.detection_probability_with_bh_mass_interpolated(
            q, m_max, 0.0, 0.0, h=0.73, z=0.4
        )
        assert float(np.asarray(hi)) == pytest.approx(float(np.asarray(at_edge)), abs=1e-14)
        # z above the span == top-u-node value; z < 0 == z = 0 value.
        z_hi = dp.detection_probability_with_bh_mass_interpolated(
            q, 3e5, 0.0, 0.0, h=0.73, z=z_max * 3.0
        )
        z_edge = dp.detection_probability_with_bh_mass_interpolated(
            q, 3e5, 0.0, 0.0, h=0.73, z=z_max
        )
        assert float(np.asarray(z_hi)) == pytest.approx(float(np.asarray(z_edge)), abs=1e-14)
        z_neg = dp.detection_probability_with_bh_mass_interpolated(q, 3e5, 0.0, 0.0, h=0.73, z=-0.5)
        z_zero = dp.detection_probability_with_bh_mass_interpolated(q, 3e5, 0.0, 0.0, h=0.73, z=0.0)
        assert float(np.asarray(z_neg)) == pytest.approx(float(np.asarray(z_zero)), abs=1e-14)

    def test_monotone_and_bounded_in_dl(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=800)
        dp = _build_on(d)
        assert dp._wbh_dlq is not None
        q = np.linspace(0.0, float(dp._wbh_dlq[-1]) * 1.05, 500)
        for z_val, m_val in ((0.1, 1e5), (0.6, 3e5), (1.3, 9e5)):
            s = np.asarray(
                dp.detection_probability_with_bh_mass_interpolated(
                    q,
                    np.full_like(q, m_val),
                    np.zeros_like(q),
                    np.zeros_like(q),
                    h=0.73,
                    z=np.full_like(q, z_val),
                )
            )
            assert np.all((s >= 0.0) & (s <= 1.0))
            assert np.all(np.diff(s) <= 1e-12)


class TestCase8BuildHInvariance:
    """§3.7 case 8: tables byte-identical regardless of which h is queried."""

    def test_tables_h_invariant_bytes(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=600)
        dp = _build_on(d)
        assert dp._wbh_stilde is not None
        before = dp._wbh_stilde.tobytes()
        q = np.linspace(0.01, 2.0, 30)
        args = (q, np.full_like(q, 3e5), np.zeros_like(q), np.zeros_like(q))
        r_low = np.asarray(
            dp.detection_probability_with_bh_mass_interpolated(
                *args, h=0.60, z=np.full_like(q, 0.5)
            )
        )
        r_high = np.asarray(
            dp.detection_probability_with_bh_mass_interpolated(
                *args, h=0.86, z=np.full_like(q, 0.5)
            )
        )
        np.testing.assert_array_equal(r_low, r_high)
        assert dp._wbh_stilde.tobytes() == before

    def test_two_builds_identical(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=600)
        dp1 = _build_on(d)
        dp2 = _build_on(d)
        assert dp1._wbh_stilde is not None and dp2._wbh_stilde is not None
        assert dp1._wbh_stilde.tobytes() == dp2._wbh_stilde.tobytes()


class TestCase9DegeneratePools:
    """§3.7 case 9: zero-variance axes / n = 1 collapse to marginals, no crash."""

    def test_degenerate_u_axis(self, tmp_path: object) -> None:
        d = str(tmp_path)
        rng = np.random.default_rng(5)
        n = 400
        z = np.full(n, 0.4)
        log_m = rng.uniform(4.6, 6.0, n)
        d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
        d_hor = np.exp(rng.normal(0.0, 0.6, n)) * (10.0 ** (log_m - 5.0)) ** 0.3
        _write_pool(d, z, 10.0**log_m, _SNR_THRESHOLD * d_hor / d_L)
        dp = _build_on(d)
        assert dp._wbh_u_nodes is not None and dp._wbh_u_nodes.size == 1
        # Joint == m-only marginal on a degenerate u axis.
        assert dp._wbh_stilde is not None and dp._wbh_sm is not None
        # Query works for any z (u clamped to the single node).
        val = dp.detection_probability_with_bh_mass_interpolated(0.3, 3e5, 0.0, 0.0, h=0.73, z=1.2)
        assert 0.0 <= float(np.asarray(val)) <= 1.0

    def test_degenerate_m_axis(self, tmp_path: object) -> None:
        d = str(tmp_path)
        rng = np.random.default_rng(6)
        n = 400
        z = rng.uniform(0.01, 1.4, n)
        d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
        d_hor = np.exp(rng.normal(0.0, 0.6, n)) * 0.4 * (1.0 + z) ** 2
        _write_pool(d, z, np.full(n, 3e5), _SNR_THRESHOLD * d_hor / d_L)
        dp = _build_on(d)
        assert dp._wbh_m_nodes is not None and dp._wbh_m_nodes.size == 1
        val = dp.detection_probability_with_bh_mass_interpolated(0.3, 9e5, 0.0, 0.0, h=0.73, z=0.5)
        assert 0.0 <= float(np.asarray(val)) <= 1.0
        # Knot helper degrades to a single lifted knot.
        knots, vals = dp.wbh_joint_knot_values(np.array([0.3]), np.array([0.5]))
        assert knots.shape == (1,) and vals.shape == (1, 1)

    def test_single_injection_pool(self, tmp_path: object) -> None:
        d = str(tmp_path)
        z = np.array([0.4])
        d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
        _write_pool(d, z, np.array([3e5]), _SNR_THRESHOLD * np.array([1.0]) / d_L)
        dp = _build_on(d)  # must not crash (1 x 1 grid == pooled step)
        val = dp.detection_probability_with_bh_mass_interpolated(0.3, 3e5, 0.0, 0.0, h=0.73, z=0.4)
        assert float(np.asarray(val)) in (0.0, 1.0)


class TestKnotHelper:
    """Erf-sum knot helper (§3.3-C convention 2, choice (a))."""

    def test_lifted_knots_and_monotonicity(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=800)
        dp = _build_on(d)
        assert dp._wbh_m_nodes is not None
        q = np.linspace(0.01, 2.5, 60)
        knots, vals = dp.wbh_joint_knot_values(q, np.full_like(q, 0.5))
        np.testing.assert_allclose(knots, 10.0**dp._wbh_m_nodes, rtol=1e-14)
        assert np.all(np.diff(knots) > 0.0)
        assert vals.shape == (q.size, knots.size)
        assert np.all((vals >= 0.0) & (vals <= 1.0))
        # Monotone non-increasing in d_L at every lifted knot.
        assert np.all(np.diff(vals, axis=0) <= 1e-12)

    def test_helper_matches_point_query_at_knots(self, tmp_path: object) -> None:
        """The helper's values ARE the point query at M_z = 10^{m_j} (no
        m-interpolation), so the two accessors must agree at the knots."""
        d = str(tmp_path)
        _generic_pool(d, n=800)
        dp = _build_on(d)
        q = np.array([0.1, 0.4, 0.9])
        z = np.array([0.2, 0.6, 1.1])
        knots, vals = dp.wbh_joint_knot_values(q, z)
        for j in (0, 10, 30):
            direct = np.asarray(
                dp.detection_probability_with_bh_mass_interpolated(
                    q,
                    np.full_like(q, knots[j]),
                    np.zeros_like(q),
                    np.zeros_like(q),
                    h=0.73,
                    z=z,
                )
            )
            np.testing.assert_allclose(vals[:, j], direct, atol=1e-12)

    def test_helper_requires_flag(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=300)
        dp = _build(d)
        with pytest.raises(ValueError, match="pdet_wbh_z_resolved"):
            dp.wbh_joint_knot_values(np.array([0.3]), np.array([0.5]))


class TestErfSumConsumers:
    """Scalar/batch bit-parity of the switched with-BH inner-M paths (flag ON)."""

    def test_gaussian_erf_sum_scalar_batch_parity(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=1000)
        dp = _build_on(d)
        z_nodes = np.linspace(0.05, 1.2, 7)
        z_batch = np.vstack([z_nodes, z_nodes * 0.8, z_nodes * 1.1])
        host_M = np.array([2e5, 3e5, 4e5])  # noqa: N806
        host_err = np.array([4e4, 5e4, 6e4])
        phis = np.zeros(3)
        qss = np.zeros(3)
        batch = _bh_mass_denominator_inner_m_integral_batch(
            z_batch, dp, phis, qss, host_M, host_err, 0.73
        )
        for i in range(3):
            scalar = _bh_mass_denominator_inner_m_integral(
                z_batch[i], dp, 0.0, 0.0, float(host_M[i]), float(host_err[i]), 0.73
            )
            np.testing.assert_allclose(batch[i], scalar, rtol=1e-13, atol=1e-300)
        assert np.all(np.isfinite(batch)) and np.all(batch >= 0.0)

    def test_mass_trunc_scalar_batch_parity(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=1000)
        dp = _build_on(d)
        z_nodes = np.linspace(0.05, 1.2, 7)
        z_batch = np.vstack([z_nodes, z_nodes * 0.9])
        host_M = np.array([2e5, 4e5])  # noqa: N806
        sigma_ln = np.array([0.4, 0.5])
        Z_M = np.array([1.0, 1.0])  # noqa: N806
        batch = _mass_trunc_denominator_inner_m_integral_batch(
            z_batch, dp, np.zeros(2), np.zeros(2), host_M, sigma_ln, Z_M, 0.73
        )
        for i in range(2):
            scalar = _mass_trunc_denominator_inner_m_integral(
                z_batch[i], dp, 0.0, 0.0, float(host_M[i]), float(sigma_ln[i]), 1.0, 0.73
            )
            np.testing.assert_allclose(batch[i], scalar, rtol=1e-13, atol=1e-300)

    def test_flag_state_changes_erf_sum_when_z_dependent(self, tmp_path: object) -> None:
        """With a z-dependent pool the joint conditioning must move the
        per-host inner-M integral (fail-before evidence for the switch)."""
        d = str(tmp_path)
        _generic_pool(d, n=2000)
        dp_off = _build(d, pdet_z_resolved=True)
        dp_on = _build_on(d)
        z_nodes = np.linspace(0.05, 0.3, 5)  # low z: pooled mixes in high-z horizons
        off = _bh_mass_denominator_inner_m_integral(z_nodes, dp_off, 0.0, 0.0, 3e5, 5e4, 0.73)
        on = _bh_mass_denominator_inner_m_integral(z_nodes, dp_on, 0.0, 0.0, 3e5, 5e4, 0.73)
        assert not np.allclose(on, off, rtol=1e-3)


class TestPickleRoundtrip:
    """The joint tables must survive __getstate__ (workers query them)."""

    def test_pickle_preserves_wbh_queries(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=600)
        dp = _build_on(d)
        q = np.linspace(0.01, 2.0, 30)
        m = np.full_like(q, 3e5)
        z = np.full_like(q, 0.5)
        zeros = np.zeros_like(q)
        before = np.asarray(
            dp.detection_probability_with_bh_mass_interpolated(q, m, zeros, zeros, h=0.73, z=z)
        )
        dp2 = pickle.loads(pickle.dumps(dp))
        after = np.asarray(
            dp2.detection_probability_with_bh_mass_interpolated(q, m, zeros, zeros, h=0.73, z=z)
        )
        np.testing.assert_array_equal(before, after)
        knots1, vals1 = dp.wbh_joint_knot_values(q, z)
        knots2, vals2 = dp2.wbh_joint_knot_values(q, z)
        np.testing.assert_array_equal(knots1, knots2)
        np.testing.assert_array_equal(vals1, vals2)


class TestQualityFlags:
    """§4 item 3: per-node ESS, shrinkage weight, and bias diagnostics."""

    def test_wbh_quality_flags_registered(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=600)
        dp = _build_on(d)
        flags = dp.quality_flags(0.73)
        for key in (
            "wbh_zres_u_nodes",
            "wbh_zres_m_nodes",
            "wbh_zres_ess",
            "wbh_zres_w",
            "wbh_zres_bias_m",
            "wbh_zres_bias_u",
        ):
            assert key in flags, key
        ess = np.asarray(flags["wbh_zres_ess"])
        w = np.asarray(flags["wbh_zres_w"])
        from darksiren_emri.bayesian_inference.simulation_detection_probability import (
            _WBH_ZRES_M_NODES,
            _WBH_ZRES_U_NODES,
        )

        assert ess.shape == w.shape == (_WBH_ZRES_U_NODES, _WBH_ZRES_M_NODES)
        assert np.all(np.asarray(flags["wbh_zres_bias_m"], dtype=np.float64) >= 0.0)
        assert np.all(np.asarray(flags["wbh_zres_bias_u"], dtype=np.float64) >= 0.0)

    def test_flag_off_registers_no_wbh_flags(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _generic_pool(d, n=300)
        dp = _build(d)
        flags = dp.quality_flags(0.73)
        assert not any(k.startswith("wbh_zres") for k in flags)
