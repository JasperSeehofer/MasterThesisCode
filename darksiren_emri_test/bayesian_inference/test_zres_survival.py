"""Tests for the FIX-2 z-resolved detection survival S(d_L | z).

Physics change under test (results/lcat_h_dependence_20260725/
DERIVATION_ZRESOLVED_SURVIVAL.md, Eq. (4)-(5)): the pooled detection-horizon
survival S(d_L) = P(d_hor >= d_L) is replaced — behind the opt-in
``pdet_z_resolved`` flag — by the z-CONDITIONAL survival

    S(d_L | z) = P(d_hor >= d_L | z)

estimated with a Gaussian kernel in u = ln(1+z) (Scott d=1 bandwidth,
Abramson sqrt-law adaptive) and EXACT suffix-count survival in d_L per kernel
node.

Covered here:
  * flag-off byte-identity (default OFF must be the pooled estimator exactly);
  * limiting case (i): z-independent horizon population -> z-resolved == pooled;
  * limiting case (ii): bandwidth -> infinity -> pooled;
  * monotonicity in d_L and [0, 1] bounds at fixed z;
  * the coherent-consumer guard (flag-on 3D queries REQUIRE z);
  * sky-band x z ESS-floor fallback to the z-only conditional;
  * consumer switch through precompute_completion_denominator /
    precompute_missing_completion_denominator;
  * pickle (worker) round-trip and h-invariance.

References:
    Finn & Chernoff (1993), arXiv:gr-qc/9301003; Finn (1996),
    arXiv:gr-qc/9601048; Mandel, Farr & Gair (2019), arXiv:1809.02063;
    Scott (1992) Ch. 6; Abramson (1982), Ann. Statist. 10:1217.
"""

import pickle

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.physical_relations import dist_vectorized

_SNR_THRESHOLD = 20.0


def _write_pool(
    directory: str,
    z: npt.NDArray[np.float64],
    snr: npt.NDArray[np.float64],
    qS: npt.NDArray[np.float64] | None = None,  # noqa: N803
    h_value: float = 0.73,
) -> None:
    """Write a fully controlled injection CSV pool."""
    n = len(z)
    rng = np.random.default_rng(7)
    d_L = np.asarray(dist_vectorized(z, h=h_value), dtype=np.float64)
    df = pd.DataFrame(
        {
            "z": z,
            "M": rng.uniform(1e5, 5e5, n) * (1.0 + z),
            "phiS": rng.uniform(0.0, 2.0 * np.pi, n),
            "qS": rng.uniform(0.0, np.pi, n) if qS is None else qS,
            "SNR": snr,
            "h_inj": h_value,
            "luminosity_distance": d_L,
        }
    )
    df.to_csv(f"{directory}/injection_h_0p73_task_001.csv", index=False)


def _z_dependent_pool(directory: str, n: int = 4000, seed: int = 11) -> None:
    """Pool whose horizon distribution drifts with z (the FIX-2 target regime)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.01, 1.4, n)
    d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
    # d_hor median grows with (1+z): loudness ~ lognormal * (1+z)^2.
    d_hor = np.exp(rng.normal(0.0, 0.8, n)) * 0.4 * (1.0 + z) ** 2
    snr = _SNR_THRESHOLD * d_hor / np.maximum(d_L, 1e-10)
    _write_pool(directory, z, snr)


def _z_independent_pool_constant_horizon(directory: str, n: int = 800) -> None:
    """z varies, but EVERY injection has the identical horizon d_hor = 1.3 Gpc.

    A degenerate horizon distribution is trivially z-independent, so the
    z-conditional survival equals the pooled one EXACTLY (algebraically, for
    any kernel weights): S = 1 for d_L <= 1.3, 0 above.
    """
    rng = np.random.default_rng(3)
    z = rng.uniform(0.01, 1.4, n)
    d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
    snr = _SNR_THRESHOLD * 1.3 / np.maximum(d_L, 1e-10)
    _write_pool(directory, z, snr)


def _build(directory: str, **kwargs: object) -> SimulationDetectionProbability:
    return SimulationDetectionProbability(
        injection_data_dir=directory,
        snr_threshold=_SNR_THRESHOLD,
        **kwargs,  # type: ignore[arg-type]
    )


class TestFlagOffByteIdentity:
    """Default OFF must be the pooled estimator, byte-identical."""

    def test_default_flag_is_off(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _z_dependent_pool(d)
        dp = _build(d)
        assert dp.z_resolved is False

    def test_flag_off_matches_manual_pooled_survival(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _z_dependent_pool(d)
        dp = _build(d)
        q = np.linspace(0.0, float(np.max(dp._d_hor_sorted)) * 1.2, 500)
        expected = (dp._n_inj - np.searchsorted(dp._d_hor_sorted, q, side="left")) / float(
            dp._n_inj
        )
        got = np.asarray(
            dp.detection_probability_without_bh_mass_interpolated_zero_fill(
                q, np.zeros_like(q), np.zeros_like(q), h=0.73
            )
        )
        np.testing.assert_array_equal(got, np.clip(expected, 0.0, 1.0))

    def test_flag_off_builds_no_zres_state(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _z_dependent_pool(d)
        dp = _build(d)
        assert dp._zres_suffix_w is None
        assert dp._zres_u_nodes is None

    def test_zres_kwargs_helper_off_and_on(self, tmp_path: object) -> None:
        from darksiren_emri.bayesian_inference.bayesian_statistics import (
            _zres_z_kwargs,
        )

        d = str(tmp_path)
        _z_dependent_pool(d)
        dp_off = _build(d)
        dp_on = _build(d, pdet_z_resolved=True)
        z = np.array([0.3])
        assert _zres_z_kwargs(dp_off, z) == {}
        assert _zres_z_kwargs(object(), z) == {}  # mock p_det compatibility
        kw = _zres_z_kwargs(dp_on, z)
        assert set(kw) == {"z"}


class TestLimitingCases:
    """Packet §4.5 limiting cases (i) and (ii)."""

    def test_z_independent_horizon_population_equals_pooled(self, tmp_path: object) -> None:
        """(i) degenerate (z-independent) horizon distribution -> pooled, exactly."""
        d = str(tmp_path)
        _z_independent_pool_constant_horizon(d)
        dp_on = _build(d, pdet_z_resolved=True)
        q = np.linspace(0.0, 2.0, 400)
        pooled = dp_on._survival_at(q)
        for z_val in (0.02, 0.1, 0.5, 1.0, 1.39):
            zres = np.asarray(
                dp_on.detection_probability_without_bh_mass_interpolated_zero_fill(
                    q, np.zeros_like(q), np.zeros_like(q), h=0.73, z=np.full_like(q, z_val)
                )
            )
            np.testing.assert_allclose(zres, pooled, atol=1e-10)

    def test_degenerate_z_pool_equals_pooled(self, tmp_path: object) -> None:
        """All injections at one z: stratification unidentifiable -> pooled."""
        d = str(tmp_path)
        rng = np.random.default_rng(5)
        n = 500
        z = np.full(n, 0.4)
        d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
        d_hor = np.exp(rng.normal(0.0, 0.7, n))
        _write_pool(d, z, _SNR_THRESHOLD * d_hor / d_L)
        dp = _build(d, pdet_z_resolved=True)
        assert dp._zres_degenerate is True
        q = np.linspace(0.0, 4.0, 300)
        np.testing.assert_allclose(
            np.asarray(
                dp.detection_probability_without_bh_mass_interpolated_zero_fill(
                    q, np.zeros_like(q), np.zeros_like(q), h=0.73, z=np.full_like(q, 0.7)
                )
            ),
            dp._survival_at(q),
            atol=1e-15,
        )

    def test_statistical_z_independent_pool_close_to_pooled(self, tmp_path: object) -> None:
        """(i) statistical form: d_hor independent of z -> agreement to sampling noise."""
        d = str(tmp_path)
        rng = np.random.default_rng(21)
        n = 20000
        z = rng.uniform(0.01, 1.4, n)
        d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
        d_hor = np.exp(rng.normal(0.0, 0.6, n))  # independent of z
        _write_pool(d, z, _SNR_THRESHOLD * d_hor / d_L)
        dp = _build(d, pdet_z_resolved=True)
        q = np.linspace(0.05, 4.0, 60)
        pooled = dp._survival_at(q)
        for z_val in (0.2, 0.6, 1.1):
            zres = dp._zres_survival_at(q, np.full_like(q, z_val))
            assert float(np.max(np.abs(zres - pooled))) < 0.05

    def test_bandwidth_to_infinity_reduces_to_pooled(self, tmp_path: object) -> None:
        """(ii) sigma_u -> inf: all kernel weights -> 1 -> pooled, algebraically."""
        d = str(tmp_path)
        _z_dependent_pool(d, n=2000)
        dp = _build(d, pdet_z_resolved=True, bandwidth_scale=1e6)
        q = np.linspace(0.0, float(np.max(dp._d_hor_sorted)) * 1.1, 300)
        pooled = dp._survival_at(q)
        for z_val in (0.05, 0.5, 1.3):
            zres = dp._zres_survival_at(q, np.full_like(q, z_val))
            np.testing.assert_allclose(zres, pooled, atol=1e-8)


class TestEstimatorProperties:
    """Monotonicity, bounds, boundary values, and the measured z-drift direction."""

    def test_monotone_non_increasing_and_bounded(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _z_dependent_pool(d)
        dp = _build(d, pdet_z_resolved=True)
        dl_max = float(np.max(dp._d_hor_sorted))
        q = np.linspace(0.0, dl_max * 1.1, 800)
        for z_val in (0.02, 0.15, 0.4, 0.8, 1.2, 1.39):
            s = dp._zres_survival_at(q, np.full_like(q, z_val))
            assert np.all(s >= 0.0) and np.all(s <= 1.0)
            assert np.all(np.diff(s) <= 1e-12), f"non-monotone at z={z_val}"
            assert s[0] == pytest.approx(1.0)
            beyond = dp._zres_survival_at(np.array([dl_max * 1.01]), np.array([z_val]))
            assert float(beyond[0]) == 0.0

    def test_conditional_repairs_pooled_overestimate_at_low_z(self, tmp_path: object) -> None:
        """Fail-before evidence: pooled mixes high-z (large) horizons into low-z
        queries; the z-conditional at low z must sit BELOW the pooled survival
        at mid-range d_L (packet §1.2 mechanism, direction only)."""
        d = str(tmp_path)
        _z_dependent_pool(d, n=8000)
        dp = _build(d, pdet_z_resolved=True)
        q = np.array([np.median(dp._d_hor_sorted)])
        s_pool = float(dp._survival_at(q)[0])
        s_low = float(dp._zres_survival_at(q, np.array([0.1]))[0])
        s_high = float(dp._zres_survival_at(q, np.array([1.3]))[0])
        assert s_low < s_pool < s_high

    def test_h_invariance_of_queries(self, tmp_path: object) -> None:
        """h enters only through the query d_L(z;h); the tables are h-free."""
        d = str(tmp_path)
        _z_dependent_pool(d)
        dp = _build(d, pdet_z_resolved=True)
        q = np.linspace(0.01, 3.0, 50)
        z = np.linspace(0.05, 1.3, 50)
        a = np.asarray(
            dp.detection_probability_without_bh_mass_interpolated_zero_fill(
                q, np.zeros_like(q), np.zeros_like(q), h=0.60, z=z
            )
        )
        b = np.asarray(
            dp.detection_probability_without_bh_mass_interpolated_zero_fill(
                q, np.zeros_like(q), np.zeros_like(q), h=0.86, z=z
            )
        )
        np.testing.assert_array_equal(a, b)

    def test_pickle_roundtrip_preserves_zres_queries(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _z_dependent_pool(d)
        dp = _build(d, pdet_z_resolved=True)
        q = np.linspace(0.01, 3.0, 40)
        z = np.linspace(0.05, 1.3, 40)
        before = dp._zres_survival_at(q, z)
        dp2 = pickle.loads(pickle.dumps(dp))
        after = dp2._zres_survival_at(q, z)
        np.testing.assert_array_equal(before, after)
        # Sky-band z-conditional also works in the worker (post-pickle).
        sb = dp2.survival_per_band(q, z)
        assert sb.shape == (dp2._n_sky_bands, q.size)


class TestCoherenceGuard:
    """Flag-on 3D queries must carry z (missed consumers fail loudly)."""

    def test_all_3d_accessors_require_z(self, tmp_path: object) -> None:
        d = str(tmp_path)
        _z_dependent_pool(d)
        dp = _build(d, pdet_z_resolved=True)
        q = np.array([0.5])
        zeros = np.zeros(1)
        with pytest.raises(ValueError, match="pdet_z_resolved"):
            dp.detection_probability_without_bh_mass_interpolated_zero_fill(q, zeros, zeros, h=0.73)
        with pytest.raises(ValueError, match="pdet_z_resolved"):
            dp.detection_probability_without_bh_mass_interpolated(q, zeros, zeros, h=0.73)
        with pytest.raises(ValueError, match="pdet_z_resolved"):
            dp.survival_per_band(q)
        with pytest.raises(ValueError, match="pdet_z_resolved"):
            dp.detection_probability_without_bh_mass_sky(q, zeros, zeros, h=0.73)

    def test_2d_accessor_unchanged_no_z_needed(self, tmp_path: object) -> None:
        """The with-BH-mass (M_z-conditioned) grid keeps its current form."""
        d = str(tmp_path)
        _z_dependent_pool(d)
        dp_on = _build(d, pdet_z_resolved=True)
        dp_off = _build(d)
        q = np.linspace(0.05, 2.0, 30)
        m = np.full_like(q, 3e5)
        zeros = np.zeros_like(q)
        on = np.asarray(
            dp_on.detection_probability_with_bh_mass_interpolated(q, m, zeros, zeros, h=0.73)
        )
        off = np.asarray(
            dp_off.detection_probability_with_bh_mass_interpolated(q, m, zeros, zeros, h=0.73)
        )
        np.testing.assert_array_equal(on, off)


class TestSkyBandEssFloor:
    """(band, u-node) cells below the ESS floor fall back to the z-only conditional."""

    def test_starved_band_cell_falls_back_to_z_only(self, tmp_path: object) -> None:
        d = str(tmp_path)
        rng = np.random.default_rng(9)
        # Band 1 (|cos qS| >= 0.5): only 14 injections, all at HIGH z -> its
        # low-z kernel cells are ESS-starved. Band 0: 3000 injections, all z.
        n0, n1 = 3000, 14
        z0 = rng.uniform(0.01, 1.4, n0)
        z1 = rng.uniform(1.2, 1.4, n1)
        z = np.concatenate([z0, z1])
        qS = np.concatenate([np.full(n0, np.pi / 2), np.zeros(n1)])  # noqa: N806
        d_L = np.asarray(dist_vectorized(z, h=0.73), dtype=np.float64)
        d_hor = np.exp(rng.normal(0.0, 0.7, n0 + n1)) * 0.4 * (1.0 + z) ** 2
        _write_pool(d, z, _SNR_THRESHOLD * d_hor / d_L, qS=qS)
        dp = _build(d, pdet_z_resolved=True, n_sky_bands=2)
        assert dp._zres_band_fallback is not None
        fb = dp._zres_band_fallback
        # Low-z nodes of the starved band must be flagged; the populated band
        # must have NO fallback cells.
        assert bool(fb[1, 0]) is True
        assert not np.any(fb[0, :])
        # At a starved cell the band survival equals the z-only conditional.
        q = np.linspace(0.05, 2.0, 20)
        z_q = np.full_like(q, 0.02)  # u-node 0 territory
        s_band = dp._zres_survival_at_band(1, q, z_q)
        s_zonly = dp._zres_survival_at(q, z_q)
        np.testing.assert_allclose(s_band, s_zonly, atol=1e-12)
        # Where the starved band IS populated (high z), it must differ from
        # the fallback only through its own (band-restricted) statistics —
        # i.e. the band estimator is genuinely active there.
        assert not np.all(fb[1, :])


class _ConstCompleteness:
    """Minimal isotropic completeness stub: f_bar(z) = const."""

    def __init__(self, f: float) -> None:
        self._f = f

    def f_bar(self, z: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
        return np.full_like(np.asarray(z, dtype=np.float64), self._f)


class TestConsumerSwitch:
    """The selection integrals switch coherently with the flag."""

    def test_D_h_differs_when_z_dependent_and_matches_when_not(  # noqa: N802
        self, tmp_path: object
    ) -> None:
        from darksiren_emri.bayesian_inference.bayesian_statistics import (
            precompute_completion_denominator,
        )

        d1 = str(tmp_path) + "/zdep"
        d2 = str(tmp_path) + "/zind"
        import os

        os.makedirs(d1)
        os.makedirs(d2)
        _z_dependent_pool(d1, n=4000)
        _z_independent_pool_constant_horizon(d2)
        for directory, expect_equal in ((d1, False), (d2, True)):
            dp_off = _build(directory)
            dp_on = _build(directory, pdet_z_resolved=True)
            D_off = precompute_completion_denominator(  # noqa: N806
                [0.73], dp_off, Omega_m=0.2726, Omega_DE=0.7274
            )[0.73]
            D_on = precompute_completion_denominator(  # noqa: N806
                [0.73], dp_on, Omega_m=0.2726, Omega_DE=0.7274
            )[0.73]
            if expect_equal:
                assert D_on == pytest.approx(D_off, rel=1e-9)
            else:
                assert abs(D_on - D_off) / D_off > 0.01

    def test_beta_gbar_consumer_runs_and_scales(self, tmp_path: object) -> None:
        from darksiren_emri.bayesian_inference.bayesian_statistics import (
            precompute_completion_denominator,
            precompute_missing_completion_denominator,
        )

        d = str(tmp_path)
        _z_dependent_pool(d, n=3000)
        dp = _build(d, pdet_z_resolved=True)
        comp = _ConstCompleteness(0.5)
        D = precompute_completion_denominator(  # noqa: N806
            [0.73], dp, Omega_m=0.2726, Omega_DE=0.7274
        )[0.73]
        beta_gbar = precompute_missing_completion_denominator(
            [0.73],
            dp,
            comp,  # type: ignore[arg-type]
        )[0.73]
        # f_bar = 0.5 -> beta_Gbar = 0.5 D exactly (same integrand, same nodes).
        assert beta_gbar == pytest.approx(0.5 * D, rel=1e-10)
