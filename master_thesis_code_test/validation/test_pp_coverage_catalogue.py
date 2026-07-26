"""Tests for the catalogue / impostor-ball universe of the P-P/coverage harness.

Covers the 2026-07-26 extension of ``master_thesis_code/validation/pp_coverage.py``
that replaces the one-candidate-per-event generative model with a discrete frozen
galaxy catalogue plus hard sky-localization balls, and the three estimators that
operate on it (``lcat``, ``absolute``, ``generator_marginal``).

Derivation: ``results/pp_impostor_harness_20260726/DERIVATION_HARNESS_ANALOG.md``.
All tests are CPU-only and fast (no ``gpu`` / ``slow`` marker).
"""

import dataclasses

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats

from master_thesis_code.validation import pp_coverage as pp
from master_thesis_code.validation.pp_coverage import (
    Z_MAX_POP,
    Z_MIN,
    PPCoverageConfig,
    _build_catalogue,
    _norm_pdf,
    _random_unit_vectors,
    _sample_detected_redshifts,
    _smeared_catalogue_density,
    comoving_amplitude_of_z,
    detection_probability,
    galaxy_number_weight_of_z,
    host_rate_weight_of_z,
    population_weight_of_z,
    run_coverage,
)

TINY_CAT = PPCoverageConfig(
    n_realizations=3,
    n_events=25,
    injected_truths=[0.72],
    seed=20260726,
    kernel="volume",
    catalogue_mode=True,
    z_support=0.30,
    mixture_mode="generator_marginal",
    n_galaxies=40_000,
    sky_frac=4.0e-4,
)


# ---------------------------------------------------------------------------
# Generative-model consistency
# ---------------------------------------------------------------------------


def test_galaxy_number_weight_times_rate_weight_is_population_weight() -> None:
    """n_gal(z) * w(z) == w_pop(z) identically (derivation Sec 1, G1/G2)."""
    z = np.linspace(Z_MIN, Z_MAX_POP, 500)
    np.testing.assert_allclose(
        galaxy_number_weight_of_z(z) * host_rate_weight_of_z(z),
        population_weight_of_z(z),
        rtol=1e-14,
        atol=0.0,
    )


def test_catalogue_host_draw_matches_continuum_detected_population() -> None:
    """Catalogue-mode detected hosts share the continuum harness's z density.

    Derivation Sec 1 (G4): drawing hosts from the discrete catalogue with rate
    weight w(z) = 1/(1+z) and accepting with p_det gives the density
    n_gal * w * p_det = w_pop * p_det, which is exactly what
    ``_sample_detected_redshifts`` samples.
    """
    config = dataclasses.replace(TINY_CAT, n_galaxies=200_000)
    h_grid = config.h_grid()
    catalogue = _build_catalogue(config, h_grid, np.random.default_rng(1))
    rng = np.random.default_rng(2)
    h_true = 0.72
    p = host_rate_weight_of_z(catalogue.z_true) * detection_probability(
        comoving_amplitude_of_z(catalogue.z_true) / h_true,
        config.d50_gpc,
        config.w_pdet_gpc,
    )
    p = p / p.sum()
    drawn = catalogue.z_true[rng.choice(catalogue.z_true.size, size=8000, p=p)]
    reference = _sample_detected_redshifts(
        h_true, 8000, np.random.default_rng(3), d50=config.d50_gpc, w_pdet=config.w_pdet_gpc
    )
    assert stats.ks_2samp(drawn, reference).pvalue > 0.01


def test_cap_perturbation_places_host_uniformly_in_cap() -> None:
    """The cap centre draw makes the flat in-cap sky likelihood exact.

    If the centre is uniform in the cap about the host, the host is uniform in
    the cap about the centre; the test checks the resulting cos(psi)
    distribution is uniform on [cos theta_c, 1] (derivation Sec 1, G5).
    """
    rng = np.random.default_rng(5)
    sky_frac = 0.02
    cos_theta_c = 1.0 - 2.0 * sky_frac
    axis = _random_unit_vectors(20_000, rng)
    centre = pp._perturb_within_cap(axis, cos_theta_c, rng)
    cos_psi = np.sum(axis * centre, axis=1)
    assert cos_psi.min() >= cos_theta_c - 1e-9
    uniform = (cos_psi - cos_theta_c) / (1.0 - cos_theta_c)
    assert stats.kstest(uniform, "uniform").pvalue > 0.01


# ---------------------------------------------------------------------------
# Precompute correctness
# ---------------------------------------------------------------------------


def test_smeared_catalogue_density_matches_direct_sum() -> None:
    """The binned/convolved Khat(z) reproduces the direct per-galaxy sum.

    Derivation Sec 6 item 7: binning at sigma_z/16 costs O((sigma_z/16)^2).
    """
    rng = np.random.default_rng(7)
    sigma_z = 0.035
    z_obs = np.clip(rng.uniform(Z_MIN, 0.5, 300), Z_MIN, None)
    inv_norm = rng.uniform(0.5, 1.5, 300)
    z_eval = np.linspace(Z_MIN, Z_MAX_POP, 400)
    fast = _smeared_catalogue_density(z_obs, inv_norm, sigma_z, z_eval)
    direct: npt.NDArray[np.float64] = np.sum(
        _norm_pdf(z_eval[None, :], z_obs[:, None], sigma_z) * inv_norm[:, None], axis=0
    )
    assert float(np.max(np.abs(fast - direct))) / float(np.max(direct)) < 1e-3


def test_n_hat_w_recovers_analytic_galaxy_density() -> None:
    """W_cat/V_f equals N_gal / int n_gal dz up to the above-edge leak.

    Derivation Sec 5.1: the deficit is the untruncated per-galaxy posterior
    leaking above z_support, and must vanish as sigma_z -> 0.
    """
    zint = np.linspace(Z_MIN, Z_MAX_POP, 4000)
    analytic_scale = np.trapezoid(galaxy_number_weight_of_z(zint), zint)
    deficits: list[float] = []
    for sigma_z in (0.035, 0.002):
        config = dataclasses.replace(TINY_CAT, n_galaxies=300_000, sigma_z=sigma_z, z_support=0.30)
        catalogue = _build_catalogue(config, config.h_grid(), np.random.default_rng(11))
        expected = config.n_galaxies / float(analytic_scale)
        deficits.append(abs(catalogue.n_hat_w / expected - 1.0))
    assert deficits[0] < 0.05, "sigma_z=0.035 leak larger than the documented few percent"
    assert deficits[1] < deficits[0], "the above-edge leak must shrink as sigma_z -> 0"
    assert deficits[1] < 0.01


def test_balls_contain_genuine_impostors() -> None:
    """The extension's raison d'etre: balls with multiple, often wrong, candidates."""
    config = dataclasses.replace(TINY_CAT, n_galaxies=80_000, sky_frac=1.0e-3)
    out = run_coverage(config)
    res = out["results"]["0.7200"]
    assert res["mean_ball_size"] > 1.0
    assert res["impostor_fraction"] > 0.2
    assert 0.0 < res["host_in_ball_fraction"] < 1.0
    assert res["completion_fraction"] > 0.0


# ---------------------------------------------------------------------------
# Derived identities and limiting cases
# ---------------------------------------------------------------------------


def test_wpop_normalization_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    """p_i is homogeneous of degree zero under w_pop -> c * w_pop.

    Derivation Sec 2.4: all four terms of p_i = (A_i + B_num)/D_gen are
    homogeneous of degree one in w_pop, so any multiplicative function of h
    (in particular production's h^-3 comoving-volume factor) cancels.
    """
    baseline = run_coverage(TINY_CAT)
    monkeypatch.setattr(pp, "_W_POP", pp._W_POP * 3.7)
    scaled = run_coverage(TINY_CAT)
    assert scaled["results"] == baseline["results"]


def test_complete_catalogue_absolute_equals_generator_marginal() -> None:
    """At z_support >= Z_MAX_POP the two absolute-mass modes are algebraically identical.

    Derivation Sec 5.3: beta_Gbar = 0 and B_num = 0, so both reduce to
    (Sum_ball w N)/(Sigma_glob * sky_frac).
    """
    base = dataclasses.replace(TINY_CAT, z_support=Z_MAX_POP, n_galaxies=60_000)
    absolute = run_coverage(dataclasses.replace(base, mixture_mode="absolute"))
    generator = run_coverage(dataclasses.replace(base, mixture_mode="generator_marginal"))
    a = absolute["results"]["0.7200"]
    g = generator["results"]["0.7200"]
    assert a["map_mean"] == pytest.approx(g["map_mean"], rel=0.0, abs=1e-12)
    assert a["coverage"] == g["coverage"]
    assert a["completion_fraction"] == pytest.approx(0.0, abs=1e-12)


def test_empty_balls_reduce_to_pure_completion_branch() -> None:
    """sky_frac -> 0 leaves exactly the host (if catalogued) in the ball.

    A catalogued host is ALWAYS inside its own cap by construction (derivation
    Sec 1, G5), so the vanishing-cap limit is the perfect-association universe:
    ball size 1 for catalogued hosts, empty for uncatalogued ones. Derivation
    Sec 2.3: the empty ball gives Sum_ball w N = 0, so p_i = B_num/Den emerges
    as a continuous limit rather than a separate branch.
    """
    config = dataclasses.replace(TINY_CAT, sky_frac=1.0e-9, n_galaxies=20_000)
    res = run_coverage(config)["results"]["0.7200"]
    assert res["empty_ball_fraction"] == pytest.approx(res["completion_fraction"])
    assert res["host_in_ball_fraction"] == pytest.approx(1.0 - res["completion_fraction"])
    assert res["mean_ball_size"] == pytest.approx(1.0 - res["completion_fraction"])
    assert res["impostor_fraction"] == pytest.approx(0.0)
    assert np.isfinite(res["map_mean"])
    assert 0.0 <= res["coverage"]["68"] <= 1.0


def test_option_a_ratio_is_order_unity() -> None:
    """n_bar_w = Sigma_glob/beta_G tracks the generator-consistent n_hat_w.

    Derivation Sec 5.5: the harness catalogue is drawn from exactly the modelled
    density, so Option A holds up to the sigma_z asymmetry and the above-edge
    leak — a documented harness limitation, asserted here so a regression that
    broke the normalization would be caught.
    """
    config = dataclasses.replace(TINY_CAT, n_galaxies=300_000)
    h_grid = config.h_grid()
    catalogue = _build_catalogue(config, h_grid, np.random.default_rng(13))
    zbg = np.linspace(Z_MIN, min(float(config.z_support or Z_MAX_POP), Z_MAX_POP), 3000)
    beta_G = np.trapezoid(
        detection_probability(
            comoving_amplitude_of_z(zbg)[:, None] / h_grid[None, :],
            config.d50_gpc,
            config.w_pdet_gpc,
        )
        * population_weight_of_z(zbg)[:, None],
        zbg,
        axis=0,
    )
    ratio = (catalogue.sigma_glob / beta_G) / catalogue.n_hat_w
    assert 0.8 < float(ratio.min()) and float(ratio.max()) < 1.2


def test_determinism_same_seed() -> None:
    """Catalogue mode is fully deterministic given the seed."""
    assert run_coverage(TINY_CAT)["results"] == run_coverage(TINY_CAT)["results"]


def test_resample_catalogue_changes_results() -> None:
    """Per-realization catalogues are a different (independent-universe) experiment."""
    shared = run_coverage(TINY_CAT)["results"]["0.7200"]
    resampled = run_coverage(
        dataclasses.replace(TINY_CAT, resample_catalogue_per_realization=True)
    )["results"]["0.7200"]
    assert shared != resampled


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_catalogue_mode_requires_z_support() -> None:
    with pytest.raises(ValueError, match="requires z_support"):
        run_coverage(dataclasses.replace(TINY_CAT, z_support=None))


def test_catalogue_only_modes_rejected_without_catalogue_mode() -> None:
    for mode in ("lcat", "generator_marginal"):
        with pytest.raises(ValueError, match="catalogue-mode only"):
            run_coverage(dataclasses.replace(TINY_CAT, catalogue_mode=False, mixture_mode=mode))


def test_catalogue_mode_rejects_non_catalogue_modes() -> None:
    for mode in ("two_branch", "gray", "conditioned", "exact"):
        with pytest.raises(ValueError, match="catalogue_mode=True supports"):
            run_coverage(dataclasses.replace(TINY_CAT, mixture_mode=mode))


def test_catalogue_mode_rejects_bare_kernel() -> None:
    with pytest.raises(ValueError, match="kernel='bare' is"):
        run_coverage(dataclasses.replace(TINY_CAT, kernel="bare"))


def test_all_three_catalogue_estimators_run_and_differ() -> None:
    """lcat / absolute / generator_marginal are distinct estimators on one universe."""
    maps = {}
    for mode in ("lcat", "absolute", "generator_marginal"):
        res = run_coverage(dataclasses.replace(TINY_CAT, mixture_mode=mode))["results"]["0.7200"]
        assert np.isfinite(res["map_mean"])
        maps[mode] = res["map_mean"]
    assert maps["lcat"] != maps["generator_marginal"]


def test_non_catalogue_modes_unaffected() -> None:
    """The pre-existing single-candidate golden pin is untouched by this extension."""
    config = PPCoverageConfig(
        n_realizations=6,
        n_events=30,
        injected_truths=[0.72],
        seed=20260711,
        kernel="volume",
        z_support=0.30,
        mixture_mode="absolute",
    )
    out = run_coverage(config)
    assert np.isfinite(out["results"]["0.7200"]["map_mean"])
    assert out["config"]["catalogue_mode"] is False
