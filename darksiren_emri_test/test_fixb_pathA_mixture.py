r"""Tests for the path-(A) joint C9+C8 mass-consistent mixture (Fix B).

Spec: ``.planning/derivation-2dbias-fix-20260803/FIXB_PATHA_PACKAGE.md`` §3-§5
(author-approved 2026-08-04). The change replaces the separately fitted
mass-blind survival ``S_3D`` by the phi-marginal of the SAME with-BH survival,

    S_bar_phi(z;h) = INTEGRAL phi(log10 M) S_4D(d_L(z;h), M(1+z)) dlog10 M,

in all three selection slots (``beta_Gbar``'s integrand, ``n_hat_w``'s
numerator ``Sigma^phi`` and its denominator ``beta_G^phi``), re-derives the 1D
channel's ``D^phi = beta_G^phi + beta_Gbar^phi``, assembles the mixture from
``alpha_G^phi = beta_G^phi r_Malm`` and ``D~^phi = alpha_G^phi +
beta_Gbar^phi``, and gives the 2D completion leg its own numerator
``B_num_wbh`` with the mass density ``g_i`` INSIDE the quadrature ((N8)).

Covered here:

* **gate (i)** measure invariance post-fix — the 2D channel is exactly
  homogeneous of degree one in the ``x_M``-measure, at machine zero and
  h-independently, so ``dMAP/dlnC = 0``.
* **gate (iii-a)** ``generator_marginal`` is untouched even when the
  phi-convention tables exist.
* **T8** the tower identity ``r_phi == 1`` by construction: the survival that
  enters ``Sigma^phi``/``beta^phi`` IS the phi-marginal of the ``S_4D`` that
  enters ``Sigma^4D``.
* **T9** the ``Sigma^4D`` mass-band shares are logged.
* **T10** the two alphas (``generator_marginal``'s draw-side ``a_cat`` and path
  (A)'s ``alpha_G^phi``) are distinct objects — the F5/F12 attribution gap
  stays open and is documented, not silently unified.
* **L4** (``s = 0``): a flat-in-log mass density makes ``g_i`` exactly z- and
  h-independent, so its tilt vanishes.
* **L5** (``sigma_Mz -> 0``): ``g_i`` stays finite and non-zero (a dark host's
  mass is never "measured") and collapses to the point evaluation.
* **falsifier (c)**: g-inside vs g-at-the-event-redshift coincide as the
  completion window collapses; the measured tolerance of record (0.05 nats
  tilt / 1e-4 in h) is quoted with it.
* the mixture anchors of record at ``h = 0.73`` (delivered convention primary
  per author decision D2).

CPU-only; the pool-backed anchor reproduction is marked ``slow``.
"""

import math
import os
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri.bayesian_inference.bayesian_statistics import (
    BayesianStatistics,
    completion_mass_factor_g,
    dark_mass_density_per_mass,
    path_a_mixture_objects,
    precompute_phi_marginal_survival,
    precompute_phi_selection_integrals,
    rescore_class_share_joint_selection,
)
from darksiren_emri.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from darksiren_emri.emri_rate import R_eff_per_mbh

_H = 0.73

# ---------------------------------------------------------------------------
# Anchors of record — FIXB_PATHA_PACKAGE.md §5, measured on the staged
# catalogues of record (sha256-verified) and the production injection pool.
#
# CONVENTION (author decision D2, 2026-08-04): the DELIVERED-catalogue values
# are the PRIMARY pins; the truth (generator-parent) values are recorded as
# secondary/informational. Promotion path: promote the truth-convention values
# to primary once the truth-convention Sigma^4D(h) is measured at all 41 h on
# the D1-remedied rerun (until then its h-curve is a 3-anchor quadratic model).
# ---------------------------------------------------------------------------
_BETA_G_PHI_073 = 1.533228e8  # catalogue-free quadrature
_BETA_GBAR_PHI_073 = 8.884038e8  # catalogue-free quadrature
_D_PHI_073 = 1.041727e9  # = beta_G_phi + beta_Gbar_phi
_SIGMA_PHI_073_DELIVERED = 9.562370e8  # staged observed catalogue (PRIMARY)
_SIGMA_PHI_073_TRUTH = 9.808671e8  # staged parent catalogue (secondary)
_SIGMA_4D_073_DELIVERED = 4.221903e8  # production logs, exact (PRIMARY)
_SIGMA_4D_073_TRUTH = 3.754526e8  # candc staged parent (secondary)
_R_MALM_073_DELIVERED = 0.4415122  # PRIMARY
_R_MALM_073_TRUTH = 0.3827762  # secondary
_W_TILDE_G_073_DELIVERED = 0.070802  # PRIMARY
_W_TILDE_G_073_TRUTH = 0.061967  # secondary
# RETIRED (mixed-catalogue artifact: production Sigma^4D over a LOCAL-baseline
# Sigma^phi) — kept here so the retirement is greppable, never asserted as a
# production pin. FIXB_PATHA_PACKAGE.md §7.9.
_R_MALM_073_RETIRED_MIXED = 0.4304
_W_TILDE_G_073_RETIRED_MIXED = 0.069143

_POOL_OF_RECORD = (
    "results/campaign51_20260728/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728"
)


# ===========================================================================
# Mixture arithmetic and the anchors of record
# ===========================================================================
def test_path_a_mixture_objects_anchor_delivered_convention() -> None:
    """The delivered-convention anchors of record reproduce (PRIMARY pins)."""
    obj = path_a_mixture_objects(
        _BETA_G_PHI_073,
        _BETA_GBAR_PHI_073,
        _SIGMA_PHI_073_DELIVERED,
        _SIGMA_4D_073_DELIVERED,
    )
    assert obj["D_phi"] == pytest.approx(_D_PHI_073, rel=1e-6)
    assert obj["r_Malm"] == pytest.approx(_R_MALM_073_DELIVERED, rel=1e-5)
    assert obj["w_tilde_G"] == pytest.approx(_W_TILDE_G_073_DELIVERED, rel=1e-4)
    # alpha_G^phi = Sigma^4D/n_hat_w^phi = beta_G^phi r_Malm (two routes, one value)
    assert obj["alpha_G_phi"] == pytest.approx(_BETA_G_PHI_073 * obj["r_Malm"], rel=1e-12)
    assert obj["D_tilde_phi"] == pytest.approx(obj["alpha_G_phi"] + _BETA_GBAR_PHI_073, rel=1e-12)
    assert obj["n_hat_w_phi"] == pytest.approx(
        _SIGMA_PHI_073_DELIVERED / _BETA_G_PHI_073, rel=1e-12
    )


def test_path_a_mixture_objects_anchor_truth_convention_secondary() -> None:
    """The truth-convention anchors reproduce (SECONDARY, informational).

    Promotion path (author decision D2): these become primary once the
    truth-convention Sigma^4D(h) is measured at all 41 h on the D1-remedied
    rerun; today its h-curve between 0.60/0.73/0.86 is a modeled quadratic.
    """
    obj = path_a_mixture_objects(
        _BETA_G_PHI_073, _BETA_GBAR_PHI_073, _SIGMA_PHI_073_TRUTH, _SIGMA_4D_073_TRUTH
    )
    assert obj["r_Malm"] == pytest.approx(_R_MALM_073_TRUTH, rel=1e-4)
    assert obj["w_tilde_G"] == pytest.approx(_W_TILDE_G_073_TRUTH, rel=1e-3)


def test_retired_mixed_catalogue_values_are_not_reproduced() -> None:
    """The retired mixed-catalogue r_Malm/w~_G are NOT what the code computes.

    r_Malm = 0.4304 / w~_G = 0.069143 came from pairing the production
    Sigma^4D with a LOCAL-baseline Sigma^phi. Because both sums are now taken
    on the catalogue the run loads (decision D2, enforced by sharing
    ``precompute_global_catalog_selection``), that pairing cannot recur.
    """
    obj = path_a_mixture_objects(
        _BETA_G_PHI_073,
        _BETA_GBAR_PHI_073,
        _SIGMA_PHI_073_DELIVERED,
        _SIGMA_4D_073_DELIVERED,
    )
    assert abs(obj["r_Malm"] - _R_MALM_073_RETIRED_MIXED) > 0.005
    assert abs(obj["w_tilde_G"] - _W_TILDE_G_073_RETIRED_MIXED) > 1e-3


def test_degenerate_legs_do_not_raise() -> None:
    """A non-positive leg yields zeros/NaN instead of aborting the h grid."""
    obj = path_a_mixture_objects(0.0, 0.0, 0.0, 0.0)
    assert obj["r_Malm"] == 0.0
    assert obj["alpha_G_phi"] == 0.0
    assert math.isnan(obj["w_tilde_G"])


# ===========================================================================
# Monitored gate (ii) under the joint selection S_and (author decision D1)
# ===========================================================================
def test_gate_ii_rescore_reproduces_the_measurement_of_record() -> None:
    """share 0.07280503 -> 0.0542477 under S_and (z = -0.48), k=164/N=3135."""
    share_snr_only = 0.07280503210939372
    share_and = rescore_class_share_joint_selection(share_snr_only)
    assert share_and == pytest.approx(0.05424771805090346, rel=5e-5)
    k, n = 164, 3135
    expected = n * share_and
    z = (k - expected) / math.sqrt(expected * (1.0 - share_and))
    assert z == pytest.approx(-0.478, abs=0.01)
    # Monitored only: scored WITHOUT the p0 window the pipeline actually
    # applied, the same statistic sits at -4.4 sigma.
    expected_snr = n * share_snr_only
    z_snr = (k - expected_snr) / math.sqrt(expected_snr * (1.0 - share_snr_only))
    assert z_snr < -4.0


def test_gate_ii_rescore_edge_cases() -> None:
    """Degenerate shares pass through unchanged."""
    assert rescore_class_share_joint_selection(0.0) == 0.0
    assert rescore_class_share_joint_selection(1.0) == 1.0
    assert rescore_class_share_joint_selection(0.5, 1.0) == pytest.approx(0.5)


# ===========================================================================
# phi: the ONE mass density, imported from the generator
# ===========================================================================
def test_phi_is_normalised_and_supported_on_the_babak_band() -> None:
    """INTEGRAL phi(M) dM = 1 on [1e4, 1e7]; exactly zero outside."""
    M = np.geomspace(M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX, 20001)
    total = float(np.trapezoid(dark_mass_density_per_mass(M), M))
    assert total == pytest.approx(1.0, rel=2e-5)
    off_band = dark_mass_density_per_mass(
        np.array([0.5 * M_SOURCE_FRAME_MIN, 2.0 * M_SOURCE_FRAME_MAX])
    )
    assert np.all(off_band == 0.0)


def test_phi_matches_the_generator_density_up_to_normalisation() -> None:
    """phi is the density _draw_dark_masses samples — never re-typed."""
    from darksiren_emri.dark_siren_injection import (
        dark_mass_log10_density_unnormalised,
    )

    M = np.geomspace(2e4, 5e6, 17)
    ratio = (
        dark_mass_density_per_mass(M)
        * (M * math.log(10.0))
        / (dark_mass_log10_density_unnormalised(M))
    )
    assert np.allclose(ratio, ratio[0], rtol=1e-12)


# ===========================================================================
# T8 — the tower identity r_phi == 1 by construction
# ===========================================================================
class _StubWithBhPdet:
    """S_4D(d_L, M_z) = exp(-d_L/3) * sigmoid(log10 M_z) — smooth and separable."""

    def __init__(self, dl_max: float = 8.0) -> None:
        self._dl_max = dl_max

    def get_dl_max(self, h: float) -> float:
        return self._dl_max

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        *,
        h: float,
    ) -> npt.NDArray[np.float64]:
        d = np.asarray(d_L, dtype=np.float64)
        m = np.log10(np.asarray(M_z, dtype=np.float64))
        return np.exp(-d / 3.0) / (1.0 + np.exp(-(m - 5.5)))


def _constant_completeness(f_const: float) -> MagicMock:
    mock = MagicMock()
    del mock.f_pixels
    del mock.pixel_centers
    mock.f_bar.side_effect = lambda z, h: np.full_like(np.asarray(z, dtype=np.float64), f_const)
    return mock


def test_T8_r_phi_is_one_by_construction() -> None:
    """S_bar_phi IS the phi-marginal of the S_4D that Sigma^4D evaluates.

    Post-change there is no separately fitted S_3D in any path-(A) slot, so the
    tower identity holds identically: recomputing the contraction directly from
    the with-BH accessor reproduces the tabulated S_bar_phi, i.e. r_phi == 1
    up to quadrature/interpolation error (the instrument of record runs at
    2.5e-5 MC; here it is exact by construction).
    """
    p_det = _StubWithBhPdet()
    table = precompute_phi_marginal_survival([_H], p_det, z_max_cap=1.5)  # type: ignore[arg-type]
    z_grid, s_phi = table[_H]

    log10_M, M_grid, phi, _ = bs._phi_dark_mass_log10_grid()
    from darksiren_emri.physical_relations import dist_vectorized

    d_L = np.asarray(dist_vectorized(z_grid, h=_H), dtype=np.float64)
    s_4d = p_det.detection_probability_with_bh_mass_interpolated(
        np.repeat(d_L[:, None], M_grid.size, axis=1),
        M_grid[None, :] * (1.0 + z_grid[:, None]),
        np.zeros(1),
        np.zeros(1),
        h=_H,
    )
    direct = np.trapezoid(s_4d * phi[None, :], log10_M, axis=1)
    assert np.allclose(s_phi, direct, rtol=1e-14, atol=0.0)
    # S_bar_phi is a probability.
    assert np.all(s_phi >= 0.0) and np.all(s_phi <= 1.0)
    # Monotone decreasing in z (S_4D decreases in d_L; the mass lift only helps
    # weakly here) — a sanity property of a survival function.
    assert s_phi[0] > s_phi[-1]


def test_phi_selection_integrals_partition_exactly() -> None:
    """beta_G^phi + beta_Gbar^phi = D^phi and the f-weights partition."""
    p_det = _StubWithBhPdet()
    table = precompute_phi_marginal_survival([_H], p_det, z_max_cap=1.5)  # type: ignore[arg-type]
    comp = _constant_completeness(0.3)
    beta_G_phi, beta_Gbar_phi = precompute_phi_selection_integrals([_H], table, comp)
    z_grid, s_phi = table[_H]
    p_pop = np.asarray(bs._redshift_population_weight(z_grid, _H), dtype=np.float64)
    total = float(np.trapezoid(s_phi * p_pop, z_grid))
    assert beta_G_phi[_H] + beta_Gbar_phi[_H] == pytest.approx(total, rel=1e-12)
    # Constant completeness: beta_G^phi = f D^phi exactly.
    assert beta_G_phi[_H] == pytest.approx(0.3 * total, rel=1e-12)


def test_sigma_phi_uses_the_same_rows_and_weights_as_sigma_4d() -> None:
    """Sigma^phi and Sigma^4D share catalogue rows, eligibility and w_g (D2)."""
    import pandas as pd

    from darksiren_emri.bayesian_inference.bayesian_statistics import (
        precompute_global_catalog_selection,
    )
    from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns

    z = np.array([0.05, 0.2, 0.9, 3.0])  # last row is beyond every z_max
    M = np.array([3e5, 2e6, 5e6, 1e6])
    catalog = pd.DataFrame(
        {
            InternalCatalogColumns.REDSHIFT: z,
            InternalCatalogColumns.BH_MASS: M,
        }
    )
    handler = MagicMock()
    handler.reduced_galaxy_catalog = catalog

    p_det = _StubWithBhPdet()
    table = precompute_phi_marginal_survival([_H], p_det, z_max_cap=1.5)  # type: ignore[arg-type]
    sigma_phi = precompute_global_catalog_selection(
        h_values=[_H],
        galaxy_catalog=handler,
        detection_probability_obj=p_det,  # type: ignore[arg-type]
        with_bh_mass=False,
        z_max_cap=1.5,
        phi_survival_table=table,
    )
    z_grid, s_phi = table[_H]
    z_max = min(1.5, float(z_grid[-1]))
    eligible = z < z_max
    w_g = np.asarray(R_eff_per_mbh(M[eligible]), dtype=np.float64) / (1.0 + z[eligible])
    expected = float(np.sum(w_g * np.interp(z[eligible], z_grid, s_phi)))
    assert sigma_phi[_H] == pytest.approx(expected, rel=1e-12)


def test_phi_survival_table_rejects_the_wrong_channel() -> None:
    """Sigma^phi is mass-blind: with_bh_mass/smear combinations are refused."""
    from darksiren_emri.bayesian_inference.bayesian_statistics import (
        precompute_global_catalog_selection,
    )

    with pytest.raises(ValueError, match="mass-blind"):
        precompute_global_catalog_selection(
            h_values=[_H],
            galaxy_catalog=MagicMock(),
            detection_probability_obj=_StubWithBhPdet(),  # type: ignore[arg-type]
            with_bh_mass=True,
            phi_survival_table={_H: (np.linspace(1e-6, 1.0, 3), np.ones(3))},
        )


# ===========================================================================
# g_i — the 2D completion leg's mass density ((N8))
# ===========================================================================
def _g_at(
    z: npt.NDArray[np.float64],
    d_L_frac: npt.NDArray[np.float64],
    *,
    det_M_z: float = 1.0e6,
    proj: float = 0.3,
    sigma: float = 0.1,
) -> npt.NDArray[np.float64]:
    return completion_mass_factor_g(z, d_L_frac, det_M_z, proj, sigma)


def test_g_i_is_a_positive_density_in_x_M() -> None:
    """g_i > 0 on band and integrates the mass kernel against phi."""
    z = np.linspace(0.05, 0.4, 9)
    g = _g_at(z, 1.0 + 0.1 * (z - z.mean()))
    assert np.all(g > 0.0)
    assert np.all(np.isfinite(g))


def test_L5_sigma_Mz_to_zero_is_finite_and_the_point_evaluation() -> None:
    """L5: sigma_cond -> 0 leaves g_i finite, non-zero and = phi_x(mu_cond)."""
    z = np.linspace(0.05, 0.4, 7)
    d_L_frac = 1.0 + 0.05 * (z - z.mean())
    det_M_z = 1.0e6
    proj = 0.3
    g_small = _g_at(z, d_L_frac, det_M_z=det_M_z, proj=proj, sigma=1e-9)
    scale = det_M_z / (1.0 + z)
    mu = 1.0 + proj * (d_L_frac - 1.0)
    point = dark_mass_density_per_mass(mu * scale) * scale
    assert np.all(g_small > 0.0)
    assert np.allclose(g_small, point, rtol=1e-8)


def test_L4_flat_log_mass_density_makes_g_i_z_and_h_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L4 (s = 0): the g_i tilt vanishes exactly for a flat-in-log phi.

    With ``phi(M) = 1/(M ln(hi/lo))`` (log-slope s = 0), ``phi_x(x;z) =
    phi(x M_z/(1+z)) M_z/(1+z) = 1/(x ln(hi/lo))`` — the ``z``-dependent mass
    scale cancels identically, so ``g_i`` is the same at every ``z`` (hence at
    every ``h``) and contributes no tilt to the posterior. Verified 0.00 of
    record; here it is exact.
    """
    lo, hi = M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX
    norm = math.log(hi / lo)

    def _flat_log_phi(M: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        M_arr = np.asarray(M, dtype=np.float64)
        inside = (M_arr >= lo) & (M_arr <= hi)
        safe = np.where(inside, M_arr, lo)
        return np.asarray(np.where(inside, 1.0 / (safe * norm), 0.0), dtype=np.float64)

    monkeypatch.setattr(bs, "dark_mass_density_per_mass", _flat_log_phi)
    z = np.linspace(0.02, 0.5, 11)
    # mu_cond frozen to 1 isolates the mass-scale tilt from the d_L projection.
    g = _g_at(z, np.ones_like(z), sigma=0.05)
    assert np.allclose(g, g[0], rtol=1e-12)


def test_falsifier_c_g_inside_collapses_to_g_at_event_redshift() -> None:
    """Falsifier (c): the inside/outside forms coincide as the window closes.

    The g-inside form (N8) and the rejected outside-quadrature form
    ``g_frac,i = phi_x(1; z_i)`` differ only through the ``z``-variation of
    ``mu_cond`` and of the mass scale across the completion window. As the
    window collapses to the event redshift the two coincide; on the delivered
    event set the residual is pinned at <= 0.05 nats of tilt / 1e-4 in h
    (FIXB_PATHA_PACKAGE.md §5 R16, GINSIDE_COUNTERFACTUAL.md).
    """
    z_i = 0.25
    det_M_z = 1.0e6
    scale_i = det_M_z / (1.0 + z_i)
    g_out = float(dark_mass_density_per_mass(np.array([scale_i]))[0] * scale_i)
    for half_width in (1e-2, 1e-3, 1e-4):
        z = np.linspace(z_i - half_width, z_i + half_width, 9)
        g_in = _g_at(z, np.ones_like(z), det_M_z=det_M_z, sigma=1e-9)
        assert float(np.mean(g_in)) == pytest.approx(g_out, rel=20.0 * half_width)


# ===========================================================================
# p_Di assembly: gate (i), gate (iii-a), T10
# ===========================================================================
def _run_p_Di_phi(
    *,
    f_const: float = 0.5,
    D_h: float = 1.520637e9,
    beta_Gbar: float = 1.335874e9,
    global_no_bh: float = 1.075654e9,
    global_with_bh: float = 4.221903e8,
    beta_G_phi: float = _BETA_G_PHI_073,
    beta_Gbar_phi: float = _BETA_GBAR_PHI_073,
    sigma_phi: float = _SIGMA_PHI_073_DELIVERED,
    use_phi_selection: bool = True,
    norm_mode: str = "absolute_marginal",
    with_bh_numerator_scale: float = 1.0,
    h: float = _H,
    # Completion-leg normalization convention (docs/derivations/
    # bscale_completion_normalization.md §6/§7, ledger rows #130-#131).
    # "derived" (default) matches the production default since the
    # [PHYSICS] change; explicit "legacy" reproduces the retracted
    # beta_Gbar^phi/beta_Gbar transfer for the historical-arithmetic pin.
    completion_b_scale: str = "derived",
) -> dict[str, Any]:
    """Run ``p_Di`` with the path-(A) tables installed; return its diagnostic row."""
    instance = object.__new__(BayesianStatistics)
    instance.h = h
    instance._normalization_mode = norm_mode
    instance._completion_b_scale = completion_b_scale
    instance.catalog_only = False
    instance.posterior_data = {}
    instance.posterior_data_with_bh_mass = {
        "galaxy_likelihoods": {},
        "additional_galaxies_without_bh_mass": {},
    }
    instance._diagnostic_rows = []

    mock_detection = MagicMock()
    mock_detection.d_L = 1.0
    mock_detection.d_L_uncertainty = 0.1
    mock_detection.phi = 0.5
    mock_detection.theta = 0.5
    mock_detection.M = 1.0e6
    instance.detection = mock_detection

    instance._det_index_to_slot = {0: 0}
    instance._means_3d = np.array([[0.5, 0.5, 1.0]])
    instance._cov_inv_3d = np.array([np.eye(3)])
    instance._log_norm_3d = np.array([0.0])
    instance._det_d_L = np.array([1.0])

    instance._D_h_table = {h: D_h}
    instance._beta_Gbar_table = {h: beta_Gbar}
    instance._beta_G_table = {h: D_h - beta_Gbar}
    instance._global_cat_denom_no_bh = {h: global_no_bh}
    instance._global_cat_denom_with_bh = {h: global_with_bh}
    instance._use_phi_selection = use_phi_selection
    instance._beta_G_phi_table = {h: beta_G_phi} if use_phi_selection else {}
    instance._beta_Gbar_phi_table = {h: beta_Gbar_phi} if use_phi_selection else {}
    instance._global_cat_selection_phi = {h: sigma_phi} if use_phi_selection else {}
    instance._proj_d_L_to_M = np.array([0.3])
    instance._sigma_cond_M = np.array([0.1])

    mock_pool = MagicMock()
    mock_pool._processes = 1
    s = with_bh_numerator_scale
    mock_pool.starmap.side_effect = [
        [np.array([[0.5, 0.3, 0.4 * s, 0.2]])],
        [np.array([[0.3, 0.2]])],
    ]

    mock_completeness = MagicMock()
    mock_completeness.ang2pix.return_value = 0
    mock_completeness.f_k.side_effect = lambda z, k, h: np.full_like(
        np.asarray(z, dtype=np.float64), f_const
    )

    mock_p_det = MagicMock()
    mock_p_det.get_dl_max.return_value = 10.0

    host = MagicMock()
    host.M, host.z, host.catalog_index = 1e6, 0.1, 0
    host_with_bh = MagicMock()
    host_with_bh.M, host_with_bh.z, host_with_bh.catalog_index = 1e6, 0.1, 1

    BayesianStatistics.p_Di(
        instance,
        possible_host_galaxies=[host],
        possible_host_galaxies_with_bh_mass=[host_with_bh],
        detection_index=0,
        pool=mock_pool,
        completeness=mock_completeness,
        detection_probability_obj=mock_p_det,
    )
    return instance._diagnostic_rows[0]


def test_path_a_assembly_matches_the_packaged_formulas() -> None:
    """1D: (beta_G^phi L1 + B^phi)/D~^phi; 2D: (alpha_G^phi L2 + B^phi g)/D~^phi.

    Pins the FIXB_PATHA_PACKAGE.md §3.2 (2026-08-04) arithmetic verbatim,
    including its ``B_scale = beta_Gbar^phi/beta_Gbar`` transfer factor --
    now retracted as an un-derived defect (docs/derivations/
    bscale_completion_normalization.md §6, ledger rows #130-#131) and no
    longer the production default. Per the memo's §7 regression-test
    clause ("keep an instrument flag --completion_b_scale legacy ... to
    preserve counterfactual reproducibility of the historical runs"), this
    historical-arithmetic pin is kept ALIVE by running it explicitly under
    ``completion_b_scale="legacy"`` rather than silently changing the
    expected numbers; the new default's numerics are covered separately by
    ``darksiren_emri_test/bayesian_inference/test_completion_b_scale.py``.
    """
    row = _run_p_Di_phi(completion_b_scale="legacy")
    obj = path_a_mixture_objects(
        _BETA_G_PHI_073, _BETA_GBAR_PHI_073, _SIGMA_PHI_073_DELIVERED, 4.221903e8
    )
    B_scale = _BETA_GBAR_PHI_073 / 1.335874e9
    B_phi = row["B_num"] * B_scale
    B_wbh_phi = row["B_num_wbh"] * B_scale
    assert row["combined_no_bh"] == pytest.approx(
        (_BETA_G_PHI_073 * row["L_cat_no_bh"] + B_phi) / obj["D_tilde_phi"], rel=1e-12
    )
    assert row["combined_with_bh"] == pytest.approx(
        (obj["alpha_G_phi"] * row["L_cat_with_bh"] + B_wbh_phi) / obj["D_tilde_phi"],
        rel=1e-12,
    )
    # The operative weight moved: w~_G, not the legacy beta_G/D = 0.1215039.
    assert row["w_G"] == pytest.approx(obj["w_tilde_G"], rel=1e-12)
    assert row["w_tilde_G"] == pytest.approx(obj["w_tilde_G"], rel=1e-12)
    assert row["w_G_legacy"] == pytest.approx(0.1215039, rel=1e-5)
    assert row["r_Malm"] == pytest.approx(_R_MALM_073_DELIVERED, rel=1e-5)
    # g_i is a real, non-trivial factor on the 2D completion leg only.
    assert 0.0 < row["g_frac"] < 1.0
    assert row["B_num_wbh"] == pytest.approx(row["B_num"] * row["g_frac"], rel=1e-12)


def test_gate_i_2d_channel_is_exactly_homogeneous_in_the_mass_measure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate (i): x_M -> x_M/C rescales the whole 2D channel by 1/C, at every h.

    Both 2D numerator legs are densities in x_M — the catalogue leg through
    ``mz_integral`` and the completion leg through ``g_i`` — so a change of the
    mass measure multiplies ``p_i^2D`` by one h-INDEPENDENT constant and cancels
    out of the posterior: ``dMAP/dlnC = 0`` exactly (measured 0.0 on the
    path-(A) recombination, ``pathA_recomb_results.json:gate_i_spotcheck``).
    """
    C = math.e
    base = _run_p_Di_phi()
    base_hi = _run_p_Di_phi(h=0.81)
    real_g = bs.dark_mass_density_per_mass
    monkeypatch.setattr(
        bs,
        "dark_mass_density_per_mass",
        lambda M: real_g(M) / C,
    )
    scaled = _run_p_Di_phi(with_bh_numerator_scale=1.0 / C)
    scaled_hi = _run_p_Di_phi(h=0.81, with_bh_numerator_scale=1.0 / C)
    # 2D channel: exactly 1/C, at machine precision.
    assert scaled["combined_with_bh"] / base["combined_with_bh"] == pytest.approx(
        1.0 / C, rel=1e-14
    )
    # ... and h-independently, so the MAP does not move.
    assert scaled_hi["combined_with_bh"] / base_hi["combined_with_bh"] == pytest.approx(
        scaled["combined_with_bh"] / base["combined_with_bh"], rel=1e-14
    )
    # 1D channel untouched by the mass measure (gate (iv)).
    assert scaled["combined_no_bh"] == base["combined_no_bh"]


def test_gate_iiia_generator_marginal_ignores_the_phi_tables() -> None:
    """Gate (iii-a): generator_marginal keeps the LEGACY assembly byte-for-byte."""
    from darksiren_emri_test.test_generator_marginal_mode import _run_p_Di_gen

    legacy = _run_p_Di_gen(
        norm_mode="generator_marginal",
        with_hosts=True,
        f_const=0.5,
        D_h=1.0e9,
        beta_Gbar=0.5e9,
        global_no_bh=2.0,
        global_with_bh=1.5,
        W_cat=1.0e9,
        V_f=2.0e8,
    )
    n_hat_w = 1.0e9 / 2.0e8
    a_cat = 1.5 / n_hat_w
    D_gen = a_cat + 0.5e9
    assert legacy["combined_with_bh"] == pytest.approx(
        (legacy["L_cat_with_bh"] + legacy["B_num"]) / D_gen, rel=1e-12
    )
    # The path-(A) tables are absent in that mode, so no phi diagnostics leak in.
    assert math.isnan(legacy.get("w_tilde_G", float("nan")))


def test_T10_the_two_alphas_are_distinct_objects() -> None:
    """T10: generator_marginal's a_cat and path (A)'s alpha_G^phi differ.

    ``a_cat = Sigma_glob_sel/n_hat_w`` (draw-side calibration, W_cat/V_f) and
    ``alpha_G^phi = Sigma^4D/n_hat_w^phi`` (selection-side, Sigma^phi/beta_G^phi)
    are DIFFERENT estimands. The F5/F12 attribution gap between them (delivered
    generator_marginal implying W_cat = 1.4048e9) remains OPEN and is unaffected
    by path (A) — this test documents the distinction rather than papering over
    it.
    """
    W_cat, V_f = 1.2493e9, 2.3237e8
    a_cat = _SIGMA_4D_073_DELIVERED / (W_cat / V_f)
    obj = path_a_mixture_objects(
        _BETA_G_PHI_073,
        _BETA_GBAR_PHI_073,
        _SIGMA_PHI_073_DELIVERED,
        _SIGMA_4D_073_DELIVERED,
    )
    assert a_cat != pytest.approx(obj["alpha_G_phi"], rel=0.05)


def test_T9_sigma_4d_mass_band_shares_are_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T9: the Sigma^4D in-band/above/below/clamped shares appear in the log."""
    import logging

    import pandas as pd

    from darksiren_emri.bayesian_inference.bayesian_statistics import (
        precompute_global_catalog_selection,
    )
    from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns

    class _Stub(_StubWithBhPdet):
        def _grid_support(
            self,
        ) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
        ]:
            edges = np.geomspace(1e4, 1e7, 41)
            centers = np.sqrt(edges[:-1] * edges[1:])
            return edges, centers, edges, centers

    catalog = pd.DataFrame(
        {
            InternalCatalogColumns.REDSHIFT: np.array([0.05, 0.2, 0.5]),
            InternalCatalogColumns.BH_MASS: np.array([5e3, 2e6, 5e7]),
        }
    )
    handler = MagicMock()
    handler.reduced_galaxy_catalog = catalog
    with caplog.at_level(logging.INFO):
        precompute_global_catalog_selection(
            h_values=[_H],
            galaxy_catalog=handler,
            detection_probability_obj=_Stub(),  # type: ignore[arg-type]
            with_bh_mass=True,
            z_max_cap=1.5,
        )
    assert any("mass-band shares" in rec.message for rec in caplog.records)


# ===========================================================================
# Anchor reproduction on the production pool (slow, data-gated)
# ===========================================================================
@pytest.mark.slow
def test_beta_phi_anchors_on_the_production_pool() -> None:
    """beta_G^phi/beta_Gbar^phi/D^phi at h = 0.73 on the pool of record."""
    from darksiren_emri.bayesian_inference.simulation_detection_probability import (
        SimulationDetectionProbability,
    )
    from darksiren_emri.constants import HOST_DRAW_Z_MAX, SNR_THRESHOLD
    from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build

    # A checkout without the campaign results (or a git worktree, whose
    # results/ lives in the main clone) can point at the pool explicitly.
    pool = os.environ.get("MTC_INJECTION_POOL_OF_RECORD", _POOL_OF_RECORD)
    if not os.path.isdir(pool):
        pytest.skip(f"injection pool of record not staged: {pool}")
    p_det = SimulationDetectionProbability(
        injection_data_dir=pool,
        snr_threshold=SNR_THRESHOLD,
        dl_bins=60,
        mass_bins=40,
        estimator="local_linear",
        expected_z_max=HOST_DRAW_Z_MAX,
        pdet_z_resolved=True,
        pdet_wbh_z_resolved=False,
    )
    table = precompute_phi_marginal_survival([_H], p_det, z_max_cap=HOST_DRAW_Z_MAX)
    beta_G_phi, beta_Gbar_phi = precompute_phi_selection_integrals(
        [_H], table, from_cache_or_build()
    )
    assert beta_G_phi[_H] == pytest.approx(_BETA_G_PHI_073, rel=1e-5)
    assert beta_Gbar_phi[_H] == pytest.approx(_BETA_GBAR_PHI_073, rel=1e-5)
    assert beta_G_phi[_H] + beta_Gbar_phi[_H] == pytest.approx(_D_PHI_073, rel=1e-5)
