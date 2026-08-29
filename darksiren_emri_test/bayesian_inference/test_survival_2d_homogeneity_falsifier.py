r"""[P3-2D twin adoption] Falsifier (i) -- the S_4D-homogeneity / S-bar-phi
double-weight regression test (charter node B7.2-pre, launched under rows
#222/#223).

Spec: results/campaign51_20260728/realistic_20260729/fanout1_20260829/
PROPOSAL_2D_TWIN_ADOPTION_20260829.md SS1.5 ("The structural asymmetry this
closes") and SS6.1(i) ("Homogeneity (zero compute, unit-test scale)"); also
registered as regression item R3 in SS8 ("NEW -- K-flat 2D homogeneity").

The proposal's degree-bookkeeping argument (SS1.5): writing S_4D -> c*S_4D for
a uniform rescaling (c in (0, 1], applied through a wrapped accessor -- never
a real physics change, just an instrument on the TEST side), the with-BH
mixture's combined likelihood is

    coded:  combined_wbh(c) = (T_cat*c^0 + T_comp*c^1) / (D~*c^1)   -- NOT
                                                                        homogeneous
    twin:   combined_wbh(c) = (T_cat*c^1 + T_comp*c^1) / (D~*c^1)   -- homogeneous
                                                                        of degree 0

because the coded with-BH catalogue numerator (``T_cat``, "off") never reads
the with-BH survival accessor at all (Gray 2020 Eq. A.10 convention,
``bayesian_statistics.py:6812-6821`` -- ``_cat_surv_2d_on`` gates the ONLY
multiplication by ``_mz_sel_2d_expectation``/``_mz_sel_2d_expectation_batch``,
``:6824-6837``), while the twin ("mz_sel") multiplies its own mass integrand
by ``E[S_4D]`` and that expectation is an EXACT linear functional of the
accessor's return value (``:6172-6176``: ``(S_4D @ weights)/sqrt(pi)``) --
confirmed exact-scaling precedent for the sibling completion function at
``test_selection_fusion.py::test_constant_s_is_exact_scaling`` (rtol=5e-15).
The completion leg (``T_comp``, via ``completion_mass_factor_g_sel``,
``:2268-2380``) and the Sigma^4D-style per-row point-query denominator
(``D~`` proxy, SS1.1) are architecturally IDENTICAL for both
``catalogue_numerator_survival_2d`` values and consume the SAME shared
accessor, so wrapping it scales both of them by ``c`` regardless of mode.

This module builds ``T_cat``/``T_comp``/``D~`` directly from the production
kernels (``single_host_likelihood``, ``completion_mass_factor_g_sel``, and a
literal Sigma^4D point-query sum, SS1.1) on the ``test_catalogue_numerator_
survival_2d.py`` single-event fixtures (``_DETECTIONS[0]``, ``_HOSTS``),
NOT the full ``BayesianStatistics.p_Di`` multi-candidate/Path-A machinery
(out of scope for a zero-compute unit-test-scale falsifier; the path-A
weights ``beta_G_phi``/``Sigma^phi`` are established elsewhere as
S_4D-scaling-invariant, SS1.5, ratio degree 0, and are elided here as 1.0).

CPU-only; no GPU, no real pool; no galaxy catalogue.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri_test.bayesian_inference.test_catalogue_numerator_survival_2d import (
    _BASE_KW,
    _HOSTS,
)
from darksiren_emri_test.bayesian_inference.test_kernel_parity import (
    _DETECTIONS,
    _StubDetectionProbability,
)

_EVENT = _DETECTIONS[0]
_H = float(_BASE_KW["h"])
_C_VALUES: list[float] = [1.0, 0.4, 0.15]


class _ScaledWithBHSurvival:
    """Wraps :class:`_StubDetectionProbability`, rescaling ONLY
    ``detection_probability_with_bh_mass_interpolated``'s return value by a
    constant ``c`` -- the "wrapped accessor" falsifier (i) probe (SS6.1(i)).
    Every other accessor method (the without-BH channel, the outside-grid
    diagnostic, ``_get_or_build_grid``) delegates unchanged via
    ``__getattr__``, since falsifier (i) is a pure with-BH-survival probe and
    the without-BH channel is architecturally untouched by the flag (A10).
    """

    def __init__(self, inner: _StubDetectionProbability, c: float) -> None:
        self._inner = inner
        self._c = c

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        s = self._inner.detection_probability_with_bh_mass_interpolated(d_L, M_z, phi, theta, h)
        return np.asarray(self._c * np.asarray(s, dtype=np.float64), dtype=np.float64)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _install_worker_globals(c: float) -> None:
    """Single-detection worker state (mirrors
    ``test_catalogue_numerator_survival_2d._install_worker_globals``), with
    the with-BH survival accessor wrapped to scale by ``c``."""
    d = _EVENT
    bs.det_index_to_slot = {0: 0}
    bs.det_d_L_arr = np.array([d["d_L"]])
    bs.det_d_L_unc_arr = np.array([d["d_L_unc"]])
    bs.det_M_arr = np.array([d["M"]])
    bs.det_phi_arr = np.array([d["phi"]])
    bs.det_theta_arr = np.array([d["theta"]])

    cov3 = np.diag([d["sig_phi"] ** 2, d["sig_theta"] ** 2, d["sig_dl_frac"] ** 2])
    cov4 = np.diag(
        [d["sig_phi"] ** 2, d["sig_theta"] ** 2, d["sig_dl_frac"] ** 2, d["sig_mz_frac"] ** 2]
    )
    bs.means_3d = np.array([[d["phi"], d["theta"], 1.0]])
    bs.cov_inv_3d = np.array([np.linalg.inv(cov3)])
    bs.log_norm_3d = np.array([-0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(cov3)[1])])
    bs.means_4d = np.array([[d["phi"], d["theta"], 1.0, 1.0]])
    bs.cov_inv_4d = np.array([np.linalg.inv(cov4)])
    bs.log_norm_4d = np.array([-0.5 * (4 * np.log(2 * np.pi) + np.linalg.slogdet(cov4)[1])])
    bs.sigma2_cond_arr = np.array([d["sig_mz_frac"] ** 2])
    bs.proj_arr = np.array([np.zeros(3)])
    bs.proj_d_L_to_M_arr = np.array([0.0])
    bs.sigma_cond_M_arr = np.array([np.sqrt(d["sig_mz_frac"] ** 2)])
    bs.detection_probability = _ScaledWithBHSurvival(_StubDetectionProbability(), c)
    bs.completeness_model = None


def _with_bh_catalogue_term(mode: str, c: float, center: str = "unset") -> float:
    """``T_cat`` proxy: sum over the three synthetic hosts (uniform weight
    ``w_g = 1`` -- the real rate weight ``w_g`` is architecturally untouched
    by ``catalogue_numerator_survival_2d``, SS1.1/A10, so a uniform stand-in
    preserves exactly the degree property under test) of the with-BH
    catalogue numerator column (``single_host_likelihood_numerator_with_bh_
    mass``, ``bayesian_statistics.py:6231-6725`` return index 2)."""
    _install_worker_globals(c)
    total = 0.0
    for host in _HOSTS:
        kw = dict(_BASE_KW)
        kw.update(host)
        kw["evaluate_with_bh_mass"] = True
        kw["catalogue_numerator_survival_2d"] = mode
        kw["catalogue_numerator_survival_2d_center"] = center
        row = bs.single_host_likelihood(**kw)
        total += row[2]
    return total


def _completion_term_proxy(c: float, n_z: int = 5) -> float:
    """``T_comp`` proxy: ``completion_mass_factor_g_sel``
    (``bayesian_statistics.py:2268-2380``, the FUSED completion mass density
    ``g_sel,prod(z;h)``) integrated (trapezoid) over a small synthetic z-grid
    around the event's own host redshifts, using the SAME wrapped with-BH
    survival accessor falsifier (i) probes. Not a reproduction of
    production's exact per-event ``B_num_wbh`` integral (that needs the full
    event z-quadrature + population-weight machinery, out of scope) -- only
    the object's c-scaling matters here, and ``completion_mass_factor_g_sel``
    is EXACTLY linear in its ``s_query`` return value (confirmed precedent:
    ``test_selection_fusion.py::test_constant_s_is_exact_scaling``,
    rtol=5e-15), and the leg is architecturally IDENTICAL for both
    ``catalogue_numerator_survival_2d`` values (rows #117-#118)."""
    stub = _ScaledWithBHSurvival(_StubDetectionProbability(), c)

    def s_query(
        d_L_gpc: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        phi = np.full_like(d_L_gpc, _EVENT["phi"])
        theta = np.full_like(d_L_gpc, _EVENT["theta"])
        return np.asarray(
            stub.detection_probability_with_bh_mass_interpolated(d_L_gpc, M_z, phi, theta, _H),
            dtype=np.float64,
        )

    z_nodes = np.linspace(0.05, 0.15, n_z)
    d_L_gpc = np.asarray(bs.dist_vectorized(z_nodes, h=_H), dtype=np.float64)
    d_L_fraction = np.asarray(d_L_gpc / _EVENT["d_L"], dtype=np.float64)
    g_sel = bs.completion_mass_factor_g_sel(
        z_nodes,
        d_L_gpc,
        d_L_fraction,
        float(_EVENT["M"]),
        0.0,  # proj_d_L_to_M -- decorrelated, matches the diagonal-covariance fixture
        math.sqrt(_EVENT["sig_mz_frac"] ** 2),
        s_query=s_query,
    )
    return float(np.trapezoid(g_sel, x=z_nodes))


def _sigma4d_denominator_proxy(c: float) -> float:
    """``D~`` proxy: the Sigma^4D-style per-row POINT query (SS1.1:
    ``S_4D(d_L(z_g;h), M_g(1+z_g))``, ``precompute_global_catalog_selection``
    ``:2965-2983``), summed (uniform weight) over the three synthetic hosts
    directly against the wrapped accessor -- not through the full
    ``GalaxyCatalogueHandler``-backed ``precompute_global_catalog_selection``
    (out of scope), but the same point-query formula it is built from."""
    stub = _ScaledWithBHSurvival(_StubDetectionProbability(), c)
    total = 0.0
    for host in _HOSTS:
        d_L = np.asarray(bs.dist_vectorized(np.array([host["host_z"]]), h=_H), dtype=np.float64)
        M_det = np.array([host["host_M"] * (1.0 + host["host_z"])])
        s = stub.detection_probability_with_bh_mass_interpolated(
            d_L,
            M_det,
            np.array([host["host_phiS"]]),
            np.array([host["host_qS"]]),
            _H,
        )
        total += float(s[0])
    return total


def _combined_with_bh_proxy(mode: str, c: float, center: str = "eff") -> float:
    """``combined_wbh(c) = (T_cat(mode, c) + T_comp(c)) / D~(c)`` -- the
    proposal's SS1.5 boxed form, with ``beta_G_phi``/``Sigma^phi`` (both
    established S_4D-scaling-invariant in ratio, SS1.5) elided as 1.0."""
    t_cat = _with_bh_catalogue_term(mode, c, center=center if mode == "mz_sel" else "unset")
    t_comp = _completion_term_proxy(c)
    d_tilde = _sigma4d_denominator_proxy(c)
    return (t_cat + t_comp) / d_tilde


# ===========================================================================
# Falsifier (i): the twin's combined_wbh is homogeneous of degree 0 in a
# uniform S_4D rescaling; the coded one is not.
# ===========================================================================
def test_falsifier_i_twin_combined_wbh_invariant_under_s4d_rescaling() -> None:
    """SS6.1(i): under S_4D -> c*S_4D, the twin's combined_wbh must be
    invariant to <= 1e-10 relative. PASS at HEAD is the proposal's
    registered prediction (SS4 "S_4D -> c (constant)" limiting case)."""
    baseline = _combined_with_bh_proxy("mz_sel", 1.0)
    for c in _C_VALUES[1:]:
        scaled = _combined_with_bh_proxy("mz_sel", c)
        rel_dev = abs(scaled - baseline) / abs(baseline)
        assert rel_dev <= 1e-10, (
            f"twin combined_wbh not invariant at c={c}: baseline={baseline!r}, "
            f"scaled={scaled!r}, rel_dev={rel_dev:.3e} -- the SS1.5 degree "
            "bookkeeping is wrong (some with-BH term has degree != 1); "
            "proposal RETURNS per SS6.1(i)"
        )


def test_falsifier_i_coded_combined_wbh_not_invariant_under_s4d_rescaling() -> None:
    """Two-sided by construction (SS6.1(i)): the coded arrangement's
    with-BH catalogue numerator never reads the survival accessor
    (``_cat_surv_2d_on`` gate, ``bayesian_statistics.py:6812``), so its
    combined_wbh is NOT homogeneous under the same rescaling -- this is the
    structural asymmetry the proposal's adoption argument (SS1.5) rests on,
    reproduced here as the control arm of the same probe."""
    baseline = _combined_with_bh_proxy("off", 1.0)
    for c in _C_VALUES[1:]:
        scaled = _combined_with_bh_proxy("off", c)
        rel_dev = abs(scaled - baseline) / abs(baseline)
        assert rel_dev > 1e-3, (
            f"coded combined_wbh unexpectedly near-invariant at c={c} "
            f"(rel_dev={rel_dev:.3e}) -- the SS1.5 asymmetry central to the "
            "adoption argument failed to reproduce"
        )


# ===========================================================================
# A15: the falsifier must be CAPABLE of failing. Simulate a bookkeeping
# defect that inserts the survival factor twice (the real-world pattern of
# the S-bar-phi double-weight, SS5) and confirm the homogeneity probe catches
# it -- proving the test discriminates, not merely confirms.
# ===========================================================================
def test_falsifier_i_detects_double_applied_survival_bookkeeping_defect() -> None:
    """Deliberately break bookkeeping: feed the with-BH catalogue term's OWN
    accessor ``c**2`` (survival applied twice) while the completion leg and
    the Sigma^4D-style denominator still see the correct ``c**1``. Per
    SS1.5's degree argument this makes ``T_cat`` degree 2 while ``T_comp``/
    ``D~`` stay degree 1, so ``combined_wbh`` is no longer homogeneous of
    degree 0 -- the same failure signature SS6.1(i) says a wrong degree
    would produce. If this assertion ever stopped failing, the probe above
    would be blind to this defect class."""

    def combined_wbh_double_survival(c: float) -> float:
        t_cat_double = _with_bh_catalogue_term("mz_sel", c * c, center="eff")
        t_comp = _completion_term_proxy(c)
        d_tilde = _sigma4d_denominator_proxy(c)
        return (t_cat_double + t_comp) / d_tilde

    baseline = combined_wbh_double_survival(1.0)
    c = 0.4
    broken = combined_wbh_double_survival(c)
    rel_dev = abs(broken - baseline) / abs(baseline)
    assert rel_dev > 1e-3, (
        "the double-survival bookkeeping defect should break homogeneity, "
        f"but the probe did not detect it (rel_dev={rel_dev:.3e}) -- "
        "falsifier (i) would be blind to this defect class"
    )


def test_falsifier_i_completion_and_sigma4d_proxies_scale_exactly_with_c() -> None:
    """Sanity/A11 check on the two supporting objects: both ``T_comp`` and
    the ``D~`` proxy must scale EXACTLY linearly in ``c`` (the property SS1.5
    assumes for them, established independently at
    ``test_selection_fusion.py::test_constant_s_is_exact_scaling`` for the
    completion leg) -- if this ever failed, the two falsifier tests above
    would be measuring the wrong thing."""
    t_comp_1 = _completion_term_proxy(1.0)
    d_tilde_1 = _sigma4d_denominator_proxy(1.0)
    for c in _C_VALUES[1:]:
        assert _completion_term_proxy(c) == pytest.approx(c * t_comp_1, rel=1e-10)
        assert _sigma4d_denominator_proxy(c) == pytest.approx(c * d_tilde_1, rel=1e-12)
