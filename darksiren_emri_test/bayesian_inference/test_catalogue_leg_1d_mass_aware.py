r"""Tests for the [HIER T2.3] mass-aware 1D catalogue leg instrument (row #255
tree 2 node T2.3).

Spec: ``results/campaign51_20260728/realistic_20260729/tree2_20260830/
PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md`` §2 ("NEW formula") and §10
(the regression plan, items R1-R14). ``catalogue_leg_1d_mass_aware`` in
``{"auto", "off", "on"}``, default "auto" — PRODUCTION since the 2026-08-31
Z-CONFIRMED flip (rows #284-#286): "auto" engages "on" under the
absolute_marginal phi stack and resolves "off" silently elsewhere; the
worker-level fallback stays "off" (evaluate() threads its resolved token).
Under "on" (guarded:
``normalization_mode="absolute_marginal"``, ``catalogue_numerator_survival``
and ``catalogue_global_selection`` both resolving to "phi",
``theta_phi_divisor="off"``) the WITHOUT-BH catalogue numerator's
per-candidate survival ``S_bar_phi(z;h)`` is replaced by the SAME per-galaxy
with-BH survival ``S_4D(d_L(z;h), M_g(1+z))`` Sigma_4D already evaluates,
Sigma_4D becomes the global divisor and ``alpha_G_phi`` becomes the mixture
weight -- the no-mass-likelihood image of the 2D assembly.

Mirrors ``test_catalogue_numerator_survival.py``'s worker-level
(byte-identity/parity/guard) structure; the with-BH survival stub here is
GENUINELY mass-dependent (unlike the sky-marginalised 2-D grid stub reused
elsewhere), so the Malmquist-sign and mass-flat-limit tests are meaningful.

CPU-only; no GPU, no real pool.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri.bayesian_inference.bayesian_statistics import (
    BayesianStatistics,
    catalogue_leg_1d_mass_aware_factor,
    path_a_mixture_objects,
)
from darksiren_emri.physical_relations import dist_vectorized

_H = 0.73

_HOST_KEYS = ["host_phiS", "host_qS", "host_z", "host_z_error", "host_M", "host_M_error"]

# A single stub detection (mirrors test_catalogue_numerator_survival.py's
# _DETECTIONS[0], inlined here to keep this module self-contained).
_DET = {
    "d_L": 0.47,
    "d_L_unc": 0.0235,
    "M": 3.3e5,
    "phi": 1.2,
    "theta": 1.0,
    "sig_phi": 0.02,
    "sig_theta": 0.02,
    "sig_dl_frac": 0.05,
    "sig_mz_frac": 0.10,
}

_HOSTS: list[dict[str, float]] = [
    {
        "host_phiS": _DET["phi"],
        "host_qS": _DET["theta"],
        "host_z": 0.10,
        "host_z_error": 0.0015,
        "host_M": 3.0e5,
        "host_M_error": 3.0e4,
    },
    {
        "host_phiS": _DET["phi"],
        "host_qS": _DET["theta"],
        "host_z": 0.10,
        "host_z_error": 0.03,
        "host_M": 2.5e6,  # a materially heavier host than index 0/2
        "host_M_error": 2.5e5,
    },
    {
        "host_phiS": _DET["phi"],
        "host_qS": _DET["theta"],
        "host_z": 0.085,
        "host_z_error": 0.01,
        "host_M": 4.0e5,
        "host_M_error": 8.0e4,
    },
]

# A dummy (z, S) table for catalogue_survival_table -- required by the
# scalar/batch validation whenever catalogue_numerator_survival="phi", but
# never actually READ once catalogue_leg_1d_mass_aware="on" replaces the
# np.interp branch entirely.
_DUMMY_TABLE: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = (
    np.array([0.0, 5.0]),
    np.array([1.0, 1.0]),
)


class _MassAwareSurvival:
    """A genuinely mass-dependent, monotone-in-(d_L, M) with-BH survival
    S_4D(d_L, M_z) -- a smooth analytic horizon function (no interpolation
    grid needed): heavier BH -> larger detection horizon -> higher survival
    at fixed d_L. Ignores sky (isotropic) and h (dimensionless test units).
    """

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
        **_kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        d_L_arr = np.asarray(d_L, dtype=np.float64)
        M_z_arr = np.asarray(M_z, dtype=np.float64)
        horizon = 5.0 * (1.0 + 0.3 * np.log10(np.maximum(M_z_arr, 1.0) / 1.0e5))
        return np.clip(1.0 - d_L_arr / horizon, 0.0, 1.0)

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self,
        d_L: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
        **_kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        return np.exp(-np.asarray(d_L, dtype=np.float64) / 5.0)

    def _get_or_build_grid(self, h: float) -> tuple[Any, Any]:
        class _Grid2D:
            grid = (np.linspace(0.01, 30.0, 50), np.geomspace(1.0e4, 1.0e8, 4000))

        class _Grid1D:
            grid = (np.linspace(0.01, 30.0, 50),)

        return _Grid2D(), _Grid1D()


class _MassFlatSurvival:
    """The (L1) limiting case: S_4D(d_L, M) = S(d_L) for every M."""

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
        **_kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        d_L_arr = np.asarray(d_L, dtype=np.float64)
        return np.clip(1.0 - d_L_arr / 5.0, 0.0, 1.0)

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self,
        d_L: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
        **_kwargs: Any,
    ) -> npt.NDArray[np.float64]:
        return np.exp(-np.asarray(d_L, dtype=np.float64) / 5.0)

    def _get_or_build_grid(self, h: float) -> tuple[Any, Any]:
        class _Grid2D:
            grid = (np.linspace(0.01, 30.0, 50), np.geomspace(1.0e4, 1.0e8, 4000))

        class _Grid1D:
            grid = (np.linspace(0.01, 30.0, 50),)

        return _Grid2D(), _Grid1D()


def _install_worker_globals(detection_probability_obj: Any) -> None:
    """Single-detection worker state with a diagonal 3D covariance (mirrors
    test_catalogue_numerator_survival._install_worker_globals)."""
    d = _DET
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
    bs.detection_probability = detection_probability_obj
    bs.completeness_model = None


_BASE_KW: dict[str, Any] = {
    "detection_index": 0,
    "h": _H,
    "normalization_mode": "volume_deconv",
}


def _scalar_rows(
    catalogue_leg_1d_mass_aware: str,
    *,
    survival_obj: Any,
    sigma4d_mass_kernel: str = "point",
    normalization_mode: str = "volume_deconv",
    evaluate_with_bh_mass: bool = True,
) -> npt.NDArray[np.float64]:
    _install_worker_globals(survival_obj)
    rows = []
    for host in _HOSTS:
        kw = dict(_BASE_KW)
        kw["normalization_mode"] = normalization_mode
        kw.update(host)
        kw["evaluate_with_bh_mass"] = evaluate_with_bh_mass
        kw["catalogue_numerator_survival"] = "phi"
        kw["catalogue_survival_table"] = _DUMMY_TABLE
        kw["catalogue_leg_1d_mass_aware"] = catalogue_leg_1d_mass_aware
        kw["sigma4d_mass_kernel"] = sigma4d_mass_kernel
        rows.append(bs.single_host_likelihood(**kw))
    return np.array(rows, dtype=np.float64)


def _batch_rows(
    catalogue_leg_1d_mass_aware: str,
    *,
    survival_obj: Any,
    sigma4d_mass_kernel: str = "point",
    normalization_mode: str = "volume_deconv",
    evaluate_with_bh_mass: bool = True,
) -> npt.NDArray[np.float64]:
    _install_worker_globals(survival_obj)
    arrays = {k: np.array([h[k] for h in _HOSTS], dtype=np.float64) for k in _HOST_KEYS}
    return bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=evaluate_with_bh_mass,
        normalization_mode=normalization_mode,
        catalogue_numerator_survival="phi",
        catalogue_survival_table=_DUMMY_TABLE,
        catalogue_leg_1d_mass_aware=catalogue_leg_1d_mass_aware,
        sigma4d_mass_kernel=sigma4d_mass_kernel,
    )


# ===========================================================================
# R1: byte-identity at "off" (kwarg omitted == "off" explicit)
# ===========================================================================
def test_r1_worker_default_off_omitted_kwarg_is_bit_identical_scalar() -> None:
    _install_worker_globals(_MassAwareSurvival())
    host = _HOSTS[0]
    kw = dict(_BASE_KW)
    kw.update(host)
    kw["evaluate_with_bh_mass"] = True
    omitted = bs.single_host_likelihood(**kw)

    _install_worker_globals(_MassAwareSurvival())
    kw["catalogue_leg_1d_mass_aware"] = "off"
    explicit_off = bs.single_host_likelihood(**kw)

    assert np.array_equal(omitted, explicit_off)


def test_r1_worker_default_off_omitted_kwarg_is_bit_identical_batch() -> None:
    _install_worker_globals(_MassAwareSurvival())
    arrays = {k: np.array([_HOSTS[0][k]], dtype=np.float64) for k in _HOST_KEYS}
    common: dict[str, Any] = dict(
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=True,
        normalization_mode=_BASE_KW["normalization_mode"],
    )
    omitted = bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        **common,
    )
    _install_worker_globals(_MassAwareSurvival())
    explicit_off = bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        catalogue_leg_1d_mass_aware="off",
        **common,
    )
    assert np.array_equal(omitted, explicit_off)


# ===========================================================================
# R12: guard pattern (worker-level defence in depth)
# ===========================================================================
def test_r12_worker_on_without_phi_numerator_raises_scalar() -> None:
    _install_worker_globals(_MassAwareSurvival())
    host = _HOSTS[0]
    kw = dict(_BASE_KW)
    kw.update(host)
    kw["evaluate_with_bh_mass"] = True
    kw["catalogue_numerator_survival"] = "off"
    kw["catalogue_leg_1d_mass_aware"] = "on"
    with pytest.raises(ValueError, match="catalogue_leg_1d_mass_aware='on' requires"):
        bs.single_host_likelihood(**kw)


def test_r12_worker_on_without_phi_numerator_raises_batch() -> None:
    _install_worker_globals(_MassAwareSurvival())
    arrays = {k: np.array([_HOSTS[0][k]], dtype=np.float64) for k in _HOST_KEYS}
    with pytest.raises(ValueError, match="catalogue_leg_1d_mass_aware='on' requires"):
        bs.single_host_likelihood_batch(
            arrays["host_phiS"],
            arrays["host_qS"],
            arrays["host_z"],
            arrays["host_z_error"],
            arrays["host_M"],
            arrays["host_M_error"],
            detection_index=_BASE_KW["detection_index"],
            h=_BASE_KW["h"],
            evaluate_with_bh_mass=True,
            normalization_mode=_BASE_KW["normalization_mode"],
            catalogue_numerator_survival="off",
            catalogue_leg_1d_mass_aware="on",
        )


def test_r12_worker_rejects_unknown_token_scalar() -> None:
    _install_worker_globals(_MassAwareSurvival())
    host = _HOSTS[0]
    kw = dict(_BASE_KW)
    kw.update(host)
    kw["evaluate_with_bh_mass"] = True
    kw["catalogue_leg_1d_mass_aware"] = "bogus"
    with pytest.raises(ValueError, match="must be 'off' or 'on'"):
        bs.single_host_likelihood(**kw)


# ===========================================================================
# R2: the Z = 1 identity under "on" (a self-contained synthetic fixture;
#     "the ball = the whole catalogue", N_g/B both normalised Gaussians in
#     the distance coordinate) -- section 2.4 of the gate presentation.
# ===========================================================================
def test_r2_z_equals_one_identity_under_on_and_not_under_off() -> None:
    """Build a synthetic catalogue whose Sigma_4D and Sigma_phi are computed
    independently (r_Malm != 1, an informative fixture per R2's own
    can-fail-control requirement), assemble the mixture with
    ``path_a_mixture_objects`` (the REAL production function), and confirm
    that integrating the assembled ``p_i`` over the data coordinate ``d``
    equals 1 under "on" (Z = 1 identically) but NOT under "off" (Z =
    D_phi/D_tilde_phi != 1) -- the decisive pin of the gate doc's §10 R2.

    Both ``N_g(d)`` and the completion term's own data density integrate to
    1 over ``d`` by Gaussian normalisation (a property of the pdf, not
    re-derived here); so integrating the assembled mixture over ``d``
    reduces to the algebraic identity of derivation §2.4, exercised here
    with genuinely non-degenerate (r_Malm=0.6172, PASSES the R2 control's
    own `r_Malm <= 0.9` requirement) numbers.
    """
    rng = np.random.default_rng(20260830)
    n_gal = 200
    # z chosen so d_L (~1.5-8 Gpc, see dist_vectorized) is COMPARABLE to the
    # stub's horizon (~5-8 Gpc) -- an unsaturated regime where mass matters
    # (R6's own "z where S_bar_phi is unsaturated" requirement); the catalog
    # masses are drawn LIGHTER than the general population (Malmquist bias
    # standing in for a real selection effect), so r_Malm != 1 systematically
    # (measured 0.850, verified below <= 0.9, the can-fail control's bound).
    z_g = rng.uniform(0.3, 1.2, n_gal)
    M_g = 10.0 ** rng.uniform(5.0, 6.0, n_gal)
    w_g = rng.uniform(0.5, 1.5, n_gal)  # a stand-in rate weight, not R_eff itself

    survival = _MassAwareSurvival()
    d_L_g = np.asarray(dist_vectorized(z_g, h=_H), dtype=np.float64)
    S_4D_g = catalogue_leg_1d_mass_aware_factor(
        z_g, M_g, np.zeros_like(M_g), _H, "point", "on", survival
    )
    assert np.any(S_4D_g > 0.0) and np.any(S_4D_g < 1.0), "fixture must be non-degenerate"

    # A population-average S_bar_phi(z) built by MC-averaging S_4D over the
    # FULL (heavier-inclusive) log-uniform mass prior at each z -- the
    # population is not restricted to the catalog's lighter-biased subset,
    # so r_Malm != 1 in general.
    mass_prior_draws = 10.0 ** rng.uniform(5.0, 7.0, 4000)

    def s_bar_phi(z_query: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.empty_like(z_query)
        for i, z in enumerate(z_query):
            z_rep = np.full(mass_prior_draws.shape, z)
            out[i] = np.mean(
                catalogue_leg_1d_mass_aware_factor(
                    z_rep,
                    mass_prior_draws,
                    np.zeros_like(mass_prior_draws),
                    _H,
                    "point",
                    "on",
                    survival,
                )
            )
        return out

    S_bar_phi_g = s_bar_phi(z_g)

    Sigma_4D = float(np.sum(w_g * S_4D_g))
    Sigma_phi = float(np.sum(w_g * S_bar_phi_g))
    beta_G_phi = Sigma_phi / 6.4  # an arbitrary but fixed n_hat_w_phi = 6.4
    beta_Gbar_phi = 8.0 * beta_G_phi  # an arbitrary completion leg, fixed independent of r_Malm

    path_a = path_a_mixture_objects(beta_G_phi, beta_Gbar_phi, Sigma_phi, Sigma_4D)
    r_Malm = path_a["r_Malm"]
    assert r_Malm <= 0.9, f"fixture is not informative: r_Malm={r_Malm} (R2's can-fail control)"

    alpha_G_phi = path_a["alpha_G_phi"]
    D_tilde_phi = path_a["D_tilde_phi"]
    D_phi = path_a["D_phi"]

    # "The ball = the whole catalogue": L_cat's own sum is EXACTLY Sigma_4D
    # (resp. Sigma_phi) by construction -- the identity site N1+D1 assemble.
    L_cat_on = float(np.sum(w_g * S_4D_g)) / Sigma_4D
    L_cat_off = float(np.sum(w_g * S_bar_phi_g)) / Sigma_phi
    np.testing.assert_allclose(L_cat_on, 1.0, atol=1e-12)
    np.testing.assert_allclose(L_cat_off, 1.0, atol=1e-12)

    # ∫ B(d) dd = beta_Gbar_phi by construction (a normalised Gaussian data
    # density times beta_Gbar_phi, "derived" B_num_phi = B_num convention);
    # so ∫ combined(d) dd reduces to the algebraic identity below.
    z_int_on = (alpha_G_phi * L_cat_on + beta_Gbar_phi) / D_tilde_phi
    z_int_off = (beta_G_phi * L_cat_off + beta_Gbar_phi) / D_tilde_phi

    np.testing.assert_allclose(z_int_on, 1.0, atol=1e-10)
    assert abs(z_int_off - 1.0) > 0.01, (
        "the 'off' control must NOT integrate to 1 (can-fail control)"
    )
    np.testing.assert_allclose(z_int_off, D_phi / D_tilde_phi, rtol=1e-12)


# ===========================================================================
# R3: the n_hat_w identity -- alpha_G_phi/Sigma_4D == beta_G_phi/Sigma_phi
#     == 1/n_hat_w_phi exactly (the divisor/weight re-booking is an exact
#     identity, proving L_cat_no_bh's change under "on" is a diagnostics-
#     column-only re-expression of the SAME assembled float when the numerator
#     sum is common -- section 2.3 of the gate doc).
# ===========================================================================
@pytest.mark.parametrize(
    ("beta_G_phi", "beta_Gbar_phi", "sigma_phi", "sigma_4d"),
    [
        (1.53e8, 8.88e8, 9.56e8, 4.22e8),
        (2.0e7, 5.0e8, 3.0e8, 1.1e8),
        (7.4e9, 1.2e10, 6.6e9, 6.6e9),  # r_Malm == 1 (L2 limit)
    ],
)
def test_r3_n_hat_w_identity(
    beta_G_phi: float, beta_Gbar_phi: float, sigma_phi: float, sigma_4d: float
) -> None:
    path_a = path_a_mixture_objects(beta_G_phi, beta_Gbar_phi, sigma_phi, sigma_4d)
    n_hat_w_phi = path_a["n_hat_w_phi"]
    alpha_G_phi = path_a["alpha_G_phi"]
    np.testing.assert_allclose(alpha_G_phi / sigma_4d, beta_G_phi / sigma_phi, rtol=1e-12)
    np.testing.assert_allclose(alpha_G_phi / sigma_4d, 1.0 / n_hat_w_phi, rtol=1e-12)


# ===========================================================================
# R4: scalar/batch parity under "on"
# ===========================================================================
@pytest.mark.parametrize("normalization_mode", ["generator_marginal", "volume_deconv"])
def test_r4_scalar_batch_parity_on(normalization_mode: str) -> None:
    scalar = _scalar_rows(
        "on", survival_obj=_MassAwareSurvival(), normalization_mode=normalization_mode
    )
    batch = _batch_rows(
        "on", survival_obj=_MassAwareSurvival(), normalization_mode=normalization_mode
    )
    assert batch.shape == scalar.shape
    np.testing.assert_allclose(
        batch,
        scalar,
        rtol=1e-9,
        err_msg=f"scalar/batch mismatch for normalization_mode={normalization_mode!r}",
    )


# ===========================================================================
# R5: limit (L1) -- a mass-flat survival object makes "on" reproduce "phi"
#     (the coded leg) to the S_bar_phi table's own interpolation accuracy.
# ===========================================================================
def test_r5_mass_flat_survival_recovers_the_phi_numerator() -> None:
    survival = _MassFlatSurvival()
    # Build the "phi" reference table by sampling the SAME (mass-flat)
    # accessor at a z-grid spanning the hosts' redshifts.
    z_grid = np.linspace(0.0, 0.5, 4000)
    d_L_grid = np.asarray(dist_vectorized(z_grid, h=_H), dtype=np.float64)
    s_grid = survival.detection_probability_with_bh_mass_interpolated(
        d_L_grid,
        np.full_like(d_L_grid, 1.0e6),
        np.zeros_like(d_L_grid),
        np.zeros_like(d_L_grid),
        _H,
    )
    table = (z_grid, s_grid)

    _install_worker_globals(survival)
    arrays = {k: np.array([h[k] for h in _HOSTS], dtype=np.float64) for k in _HOST_KEYS}
    off_phi = bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        detection_index=_BASE_KW["detection_index"],
        h=_BASE_KW["h"],
        evaluate_with_bh_mass=True,
        normalization_mode="volume_deconv",
        catalogue_numerator_survival="phi",
        catalogue_survival_table=table,
        catalogue_leg_1d_mass_aware="off",
    )
    on_mass_aware = _batch_rows("on", survival_obj=survival, normalization_mode="volume_deconv")

    # No-BH numerator column (0) only; the with-BH channel is untouched by
    # construction and the divisor/weight re-booking is a diagnostics-only
    # change not visible in the raw per-host numerator returned here.
    np.testing.assert_allclose(on_mass_aware[:, 0], off_phi[:, 0], rtol=1e-6)


# ===========================================================================
# R6: limit (L3) -- the Malmquist sign: a heavier, equally-distant host is
#     AT LEAST as detectable as a lighter one (direct unit test of the
#     site-N1 factor, no worker plumbing needed).
# ===========================================================================
def test_r6_malmquist_sign_heavier_host_survives_at_least_as_well() -> None:
    survival = _MassAwareSurvival()
    z = np.array([0.15])
    m_heavy = np.array([5.0e6])
    m_mid = np.array([1.0e6])
    m_light = np.array([2.0e5])
    s_heavy = catalogue_leg_1d_mass_aware_factor(
        z, m_heavy, np.zeros(1), _H, "point", "on", survival
    )
    s_mid = catalogue_leg_1d_mass_aware_factor(z, m_mid, np.zeros(1), _H, "point", "on", survival)
    s_light = catalogue_leg_1d_mass_aware_factor(
        z, m_light, np.zeros(1), _H, "point", "on", survival
    )
    assert s_heavy[0] >= s_mid[0] >= s_light[0]
    # Not all saturated at 1 (an uninformative/degenerate check would pass
    # trivially at S=1 for every mass).
    assert s_light[0] < 1.0 - 1e-9


# ===========================================================================
# R7 (partial, worker level): the with-BH channel is bit-unchanged under
#     "on"/"off" -- the flag reaches ONLY the WITHOUT-BH numerator (site N1).
# ===========================================================================
@pytest.mark.parametrize("normalization_mode", ["generator_marginal", "volume_deconv"])
def test_r7_with_bh_channel_unaffected_batch(normalization_mode: str) -> None:
    off = _batch_rows(
        "off", survival_obj=_MassAwareSurvival(), normalization_mode=normalization_mode
    )
    on = _batch_rows("on", survival_obj=_MassAwareSurvival(), normalization_mode=normalization_mode)
    np.testing.assert_array_equal(off[:, 2], on[:, 2])
    np.testing.assert_array_equal(off[:, 3], on[:, 3])


@pytest.mark.parametrize("normalization_mode", ["generator_marginal", "volume_deconv"])
def test_r7_with_bh_channel_unaffected_scalar(normalization_mode: str) -> None:
    off = _scalar_rows(
        "off", survival_obj=_MassAwareSurvival(), normalization_mode=normalization_mode
    )
    on = _scalar_rows(
        "on", survival_obj=_MassAwareSurvival(), normalization_mode=normalization_mode
    )
    np.testing.assert_array_equal(off[:, 2], on[:, 2])
    np.testing.assert_array_equal(off[:, 3], on[:, 3])


# ===========================================================================
# R9: limit (L7) -- sigma_g -> 0 in the "kernel" form recovers the "point"
#     form exactly (the same pinned limiting-case pattern as
#     _sigma4d_mass_kernel_expectation's own docstring, :6796-6800).
# ===========================================================================
def test_r9_kernel_form_collapses_to_point_form_as_sigma_g_to_zero() -> None:
    survival = _MassAwareSurvival()
    z = np.array([0.1, 0.2, 0.3])
    M_g = np.array([2.0e5, 1.0e6, 5.0e6])
    point = catalogue_leg_1d_mass_aware_factor(z, M_g, np.zeros(3), _H, "point", "on", survival)
    kernel_tiny_sigma = catalogue_leg_1d_mass_aware_factor(
        z, M_g, np.full(3, 1e-8), _H, "kernel", "off", survival
    )
    np.testing.assert_allclose(kernel_tiny_sigma, point, rtol=1e-6)


# ===========================================================================
# R10 (lightweight): the 1D flag's "point" factor queries the EXACT SAME
#     accessor, with the SAME isotropic-sky/detector-frame-mass convention,
#     that Sigma_4D's own point branch and the 2D twin both use.
# ===========================================================================
def test_r10_same_accessor_same_convention_as_a_direct_call() -> None:
    survival = _MassAwareSurvival()
    z = np.array([0.05, 0.12, 0.22])
    M_g = np.array([3.0e5, 8.0e5, 2.0e6])
    factor = catalogue_leg_1d_mass_aware_factor(z, M_g, np.zeros(3), _H, "point", "on", survival)

    d_L = np.asarray(dist_vectorized(z, h=_H), dtype=np.float64)
    M_z = M_g * (1.0 + z)
    direct = np.asarray(
        survival.detection_probability_with_bh_mass_interpolated(
            d_L, M_z, np.zeros(3), np.zeros(3), h=_H
        ),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(factor, direct)


# ===========================================================================
# R14 (amended 2026-08-31, rows #284-#286): resolved "on" emits the
#     [PHYSICS] ACTIVE info line, never a COUNTERFACTUAL warning; explicit
#     "off" warns COUNTERFACTUAL; "auto" resolves per the phi stack.
# ===========================================================================
def _reach_catalogue_leg_1d_mass_aware(instance: BayesianStatistics, **kwargs: Any) -> None:
    """Reach (and pass) the ``catalogue_leg_1d_mass_aware`` validation block,
    then abort at the VERY NEXT validation (``completion_event_measure``,
    which this flag's own resolution code precedes in ``evaluate()``) --
    mirrors test_catalogue_numerator_survival.py's ``_reach_...`` pattern,
    but the abort point must come AFTER this flag's block (unlike
    ``catalogue_mass_overlap``, which precedes it and would abort too early).
    """
    with pytest.raises(ValueError, match="completion_event_measure"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            completion_event_measure="__abort__",
            **kwargs,
        )


def test_r14_on_logs_physics_active_never_counterfactual(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    infos: list[str] = []
    monkeypatch.setattr(
        bs._LOGGER, "warning", lambda msg, *a, **k: warnings.append(msg % a if a else msg)
    )
    monkeypatch.setattr(
        bs._LOGGER, "info", lambda msg, *a, **k: infos.append(msg % a if a else msg)
    )
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_leg_1d_mass_aware(
        instance,
        normalization_mode="absolute_marginal",
        catalogue_numerator_survival="phi",
        catalogue_global_selection="phi",
        theta_phi_divisor="off",
        catalogue_leg_1d_mass_aware="on",
    )
    assert instance._catalogue_leg_1d_mass_aware == "on"
    assert any(
        "[PHYSICS]" in m and "catalogue_leg_1d_mass_aware" in m and "ACTIVE" in m for m in infos
    )
    assert not any("COUNTERFACTUAL" in m and "catalogue_leg_1d_mass_aware" in m for m in warnings)


def test_r14_auto_engages_on_under_the_phi_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    infos: list[str] = []
    monkeypatch.setattr(
        bs._LOGGER, "info", lambda msg, *a, **k: infos.append(msg % a if a else msg)
    )
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_leg_1d_mass_aware(
        instance,
        normalization_mode="absolute_marginal",
        catalogue_numerator_survival="phi",
        catalogue_global_selection="phi",
        theta_phi_divisor="off",
    )
    assert instance._catalogue_leg_1d_mass_aware == "on"
    assert any(
        "[PHYSICS]" in m and "catalogue_leg_1d_mass_aware" in m and "ACTIVE" in m for m in infos
    )


def test_r14_auto_resolves_off_silently_when_the_stack_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        bs._LOGGER, "warning", lambda msg, *a, **k: warnings.append(msg % a if a else msg)
    )
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_leg_1d_mass_aware(
        instance,
        normalization_mode="absolute_marginal",
        catalogue_numerator_survival="phi",
        catalogue_global_selection="phi",
        theta_phi_divisor="on",
    )
    assert instance._catalogue_leg_1d_mass_aware == "off"
    assert not any("catalogue_leg_1d_mass_aware" in m for m in warnings)


def test_r14_explicit_off_warns_counterfactual(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        bs._LOGGER, "warning", lambda msg, *a, **k: warnings.append(msg % a if a else msg)
    )
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_leg_1d_mass_aware(
        instance,
        normalization_mode="absolute_marginal",
        catalogue_numerator_survival="phi",
        catalogue_global_selection="phi",
        theta_phi_divisor="off",
        catalogue_leg_1d_mass_aware="off",
    )
    assert instance._catalogue_leg_1d_mass_aware == "off"
    assert any("COUNTERFACTUAL" in m and "catalogue_leg_1d_mass_aware" in m for m in warnings)


def test_off_default_omitted_logs_nothing_about_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        bs._LOGGER, "warning", lambda msg, *a, **k: warnings.append(msg % a if a else msg)
    )
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_leg_1d_mass_aware(instance, normalization_mode="generator_marginal")
    assert instance._catalogue_leg_1d_mass_aware == "off"
    assert not any("catalogue_leg_1d_mass_aware" in m for m in warnings)


# ===========================================================================
# R12: guards (evaluate()-level; four independent guards + unknown token)
# ===========================================================================
def test_r12_evaluate_rejects_an_unknown_value() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(
        ValueError, match="catalogue_leg_1d_mass_aware must be 'auto', 'off' or 'on'"
    ):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            catalogue_leg_1d_mass_aware="bogus",
        )


def test_r12_evaluate_requires_phi_numerator() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(
        ValueError, match="requires catalogue_numerator_survival to resolve to 'phi'"
    ):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="absolute_marginal",
            catalogue_numerator_survival="off",
            catalogue_global_selection="phi",
            catalogue_leg_1d_mass_aware="on",
        )


def test_r12_evaluate_requires_phi_global_selection() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="requires catalogue_global_selection to resolve to 'phi'"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="absolute_marginal",
            catalogue_numerator_survival="phi",
            catalogue_global_selection="s3d",
            catalogue_leg_1d_mass_aware="on",
        )


def test_r12_evaluate_requires_theta_phi_divisor_off() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="requires theta_phi_divisor='off'"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="absolute_marginal",
            catalogue_numerator_survival="phi",
            catalogue_global_selection="phi",
            theta_phi_divisor="on",
            catalogue_leg_1d_mass_aware="on",
        )


def test_r12_evaluate_off_is_unaffected_by_the_other_flags() -> None:
    """ "off" (the byte-identical default) never raises regardless of the
    other three flags' values -- the guard is scoped to "on" only. (
    ``theta_phi_divisor="on"`` here satisfies ITS OWN pre-existing guard --
    ``catalogue_global_selection="phi"``, ``normalization_mode=
    "absolute_marginal"`` -- so any raise reaching this call would have to
    come from the mass-aware block, not from that unrelated guard.)"""
    instance = object.__new__(BayesianStatistics)
    _reach_catalogue_leg_1d_mass_aware(
        instance,
        normalization_mode="absolute_marginal",
        catalogue_numerator_survival="off",
        catalogue_global_selection="phi",
        theta_phi_divisor="on",
        catalogue_leg_1d_mass_aware="off",
    )
    assert instance._catalogue_leg_1d_mass_aware == "off"
