"""Tests for the mass channel of the P-P/coverage harness ([A3], 2026-08-17).

Covers the ledger-row-#120 item-2 (D-2) extension of
``darksiren_emri/validation/pp_coverage.py``:

* the second (mass) observable on top of ``catalogue_mode`` — mass-dependent
  survival ``S_4D``, the phi-marginal survival ``S_bar_phi(z;h)``, the
  per-candidate Gaussian mass overlap, and the completion-leg mass factors
  ``completion_mass_factor_g`` / ``completion_mass_factor_g_sel``;
* the ``selection_cell`` switch mirroring production's
  ``selection_in_completion_numerator`` (off / 1d / 2d / fused);
* the three-way noise-model cell (const / model / production) from the Q-0
  audit;
* the byte-identity guarantee for every pre-existing mode.

All tests are CPU-only and fast (no ``gpu`` / ``slow`` marker).
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from darksiren_emri.validation.pp_coverage import (
    M_REF_MSUN,
    M_SOURCE_MAX,
    M_SOURCE_MIN,
    PPCoverageConfig,
    completion_mass_factor_g,
    completion_mass_factor_g_sel,
    dark_mass_density_per_mass,
    detection_probability,
    main,
    phi_marginal_survival_table,
    run_coverage,
    survival_with_mass,
)

# A tiny mass-bearing cell: catalogue/impostor-ball universe + mass channel.
TINY_MASS = PPCoverageConfig(
    n_realizations=2,
    n_events=20,
    injected_truths=[0.72],
    seed=20260817,
    kernel="volume",
    catalogue_mode=True,
    z_support=0.30,
    mixture_mode="absolute",
    n_galaxies=40_000,
    sky_frac=4.0e-4,
    h_step=0.02,
    mass_channel=True,
    mass_horizon_index=0.25,
    selection_cell="fused",
    n_z_survival=400,
    n_mass_quad=200,
)


# ---------------------------------------------------------------------------
# Mass model and survival
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slope", [0.0, 0.5, -0.3])
def test_phi_is_normalized_and_compactly_supported(slope: float) -> None:
    """``phi(M)`` integrates to one on its support and vanishes outside it."""
    ln_m = np.linspace(np.log(M_SOURCE_MIN), np.log(M_SOURCE_MAX), 20_000)
    m = np.exp(ln_m)
    total = float(np.trapezoid(dark_mass_density_per_mass(m, slope) * m, ln_m))
    # The residual is the trapezoid rule's half-bin loss at each edge of the
    # compact support (~1/n_grid), not a normalization error.
    assert total == pytest.approx(1.0, rel=1e-4)
    outside = dark_mass_density_per_mass(np.array([0.1 * M_SOURCE_MIN, 10.0 * M_SOURCE_MAX]), slope)
    np.testing.assert_array_equal(outside, np.zeros(2))


def test_survival_reduces_to_mass_blind_pdet_at_zero_horizon_index() -> None:
    """``mass_horizon_index = 0`` is the exact mass-blind ``p_det`` limit."""
    d_l = np.linspace(0.05, 4.0, 50)
    m_z = np.geomspace(M_SOURCE_MIN, M_SOURCE_MAX, 50)
    np.testing.assert_array_equal(
        survival_with_mass(d_l, m_z, mass_horizon_index=0.0),
        detection_probability(d_l),
    )


def test_survival_is_monotone_in_mass_and_bounded() -> None:
    """A heavier detector-frame BH is louder: S rises with M_z, stays in [0, 1]."""
    d_l = np.full(40, 2.0)
    m_z = np.geomspace(M_SOURCE_MIN, M_SOURCE_MAX, 40)
    s = survival_with_mass(d_l, m_z, mass_horizon_index=0.3)
    assert np.all(np.diff(s) >= 0.0)
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_phi_marginal_survival_matches_pdet_at_zero_horizon_index() -> None:
    """``S_bar_phi(z;h)`` collapses to ``p_det(A(z)/h)`` when S is mass-blind."""
    h_grid = np.array([0.62, 0.72, 0.84])
    z_grid, s_tab = phi_marginal_survival_table(
        h_grid, mass_horizon_index=0.0, n_z=300, n_mass_quad=200
    )
    from darksiren_emri.validation.pp_coverage import comoving_amplitude_of_z

    expected = detection_probability(comoving_amplitude_of_z(z_grid)[:, None] / h_grid[None, :])
    np.testing.assert_allclose(s_tab, expected, rtol=1e-10, atol=1e-12)


def test_phi_marginal_survival_is_h_dependent_and_decreasing_in_z() -> None:
    """The table carries genuine h dependence and falls with redshift."""
    h_grid = np.array([0.62, 0.84])
    _z, s_tab = phi_marginal_survival_table(
        h_grid, mass_horizon_index=0.25, n_z=200, n_mass_quad=150
    )
    assert not np.allclose(s_tab[:, 0], s_tab[:, 1])
    assert np.all(np.diff(s_tab, axis=0) <= 1e-12)


# ---------------------------------------------------------------------------
# Completion-leg mass factor g / g_sel
# ---------------------------------------------------------------------------


def _g_inputs() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Return ``(z_nodes, d_L_gpc, det_M_z)`` on a small (nz, nh) node block."""
    z = np.linspace(0.05, 0.5, 12)[:, None] * np.ones((1, 4))
    h = np.array([0.62, 0.70, 0.78, 0.84])[None, :]
    from darksiren_emri.validation.pp_coverage import comoving_amplitude_of_z

    d_l = comoving_amplitude_of_z(z[:, 0])[:, None] / h
    return z, d_l, 5.0e5


def test_g_sel_factorizes_when_survival_is_mass_independent() -> None:
    """``alpha_M = 0`` makes S constant in ``x_M``: ``g_sel == g * p_det(d_L)``."""
    z, d_l, det_m_z = _g_inputs()
    frac = d_l / d_l[0, 0]
    g = completion_mass_factor_g(z, frac, det_m_z, 0.4, 0.08)
    g_sel = completion_mass_factor_g_sel(z, d_l, frac, det_m_z, 0.4, 0.08, mass_horizon_index=0.0)
    np.testing.assert_allclose(g_sel, g * detection_probability(d_l), rtol=1e-12, atol=0.0)


def test_g_sel_sigma_cond_to_zero_limit() -> None:
    """``sigma_cond -> 0`` gives ``g_i * S(mu_cond * M_z,det)`` (row #118 / MAJOR-1)."""
    z, d_l, det_m_z = _g_inputs()
    frac = d_l / d_l[0, 0]
    proj = 0.4
    tiny = 1e-8
    g = completion_mass_factor_g(z, frac, det_m_z, proj, tiny)
    g_sel = completion_mass_factor_g_sel(z, d_l, frac, det_m_z, proj, tiny, mass_horizon_index=0.25)
    mu_cond = 1.0 + proj * (frac - 1.0)
    s_point = survival_with_mass(d_l, mu_cond * det_m_z, mass_horizon_index=0.25)
    np.testing.assert_allclose(g_sel, g * s_point, rtol=1e-6, atol=0.0)


def test_g_is_recomputed_at_every_h_grid_point() -> None:
    """``g``/``g_sel`` genuinely vary along the h axis and match a per-h call.

    The [A3] "never frozen, never elided" criterion: the h axis is an array
    axis of the node block, so each column is its own contraction.
    """
    z, d_l, det_m_z = _g_inputs()
    frac = d_l / d_l[0, 0]
    g_sel = completion_mass_factor_g_sel(z, d_l, frac, det_m_z, 0.4, 0.08, mass_horizon_index=0.25)
    assert not np.allclose(g_sel[:, 0], g_sel[:, -1])
    for j in (0, 2, 3):
        column = completion_mass_factor_g_sel(
            z[:, j], d_l[:, j], frac[:, j], det_m_z, 0.4, 0.08, mass_horizon_index=0.25
        )
        np.testing.assert_allclose(g_sel[:, j], column, rtol=0.0, atol=0.0)


def test_g_is_a_density_in_x_m_of_order_the_mass_prior() -> None:
    """``g`` is positive and finite where the phi support is reached."""
    z, d_l, det_m_z = _g_inputs()
    frac = d_l / d_l[0, 0]
    g = completion_mass_factor_g(z, frac, det_m_z, 0.0, 0.10)
    assert np.all(np.isfinite(g))
    assert np.all(g >= 0.0)
    assert float(np.max(g)) > 0.0


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_mass_channel_requires_catalogue_mode() -> None:
    cfg = dataclasses.replace(TINY_MASS, catalogue_mode=False, mixture_mode="absolute")
    with pytest.raises(ValueError, match="requires catalogue_mode"):
        run_coverage(cfg)


def test_selection_cell_requires_mass_channel() -> None:
    cfg = dataclasses.replace(TINY_MASS, mass_channel=False, selection_cell="fused")
    with pytest.raises(ValueError, match="requires mass_channel"):
        run_coverage(cfg)


def test_unknown_selection_cell_rejected() -> None:
    cfg = dataclasses.replace(TINY_MASS, selection_cell="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="selection_cell must be"):
        run_coverage(cfg)


# ---------------------------------------------------------------------------
# End-to-end behaviour of the mass-bearing harness
# ---------------------------------------------------------------------------


def test_mass_run_reports_both_channels_with_finite_statistics() -> None:
    """A mass-bearing run emits the 1D block plus the nested 2D block."""
    res = run_coverage(TINY_MASS)["results"]["0.7200"]
    assert "mass_channel_2d" in res
    two_d = res["mass_channel_2d"]
    for block in (res, two_d):
        assert set(block["coverage"]) == {"50", "68", "90"}
        assert all(0.0 <= v <= 1.0 for v in block["coverage"].values())
        assert np.isfinite(block["map_mean"])
    assert res["mean_ball_size"] > 0.0


def test_mass_run_determinism_same_seed() -> None:
    assert run_coverage(TINY_MASS) == run_coverage(TINY_MASS)


def test_selection_cell_decomposition_is_channel_local() -> None:
    """'2d' touches ONLY the 2D channel and '1d' ONLY the 1D channel.

    Mirrors production's counterfactual decomposition: [P1] lives in the mass
    integral (2D leg) and [P2] in the 1D completion numerator, so the paired
    cells must leave the other channel bit-identical on a shared seed.
    """
    off = run_coverage(dataclasses.replace(TINY_MASS, selection_cell="off"))["results"]["0.7200"]
    only_1d = run_coverage(dataclasses.replace(TINY_MASS, selection_cell="1d"))["results"]["0.7200"]
    only_2d = run_coverage(dataclasses.replace(TINY_MASS, selection_cell="2d"))["results"]["0.7200"]
    fused = run_coverage(dataclasses.replace(TINY_MASS, selection_cell="fused"))["results"][
        "0.7200"
    ]

    def one_d(block: dict) -> dict:
        return {k: v for k, v in block.items() if k != "mass_channel_2d"}

    assert one_d(off) == one_d(only_2d)
    assert one_d(only_1d) == one_d(fused)
    assert off["mass_channel_2d"] == only_1d["mass_channel_2d"]
    assert only_2d["mass_channel_2d"] == fused["mass_channel_2d"]
    # And the cells are not all the same run.
    assert one_d(off) != one_d(only_1d)


def test_mass_channel_composes_with_all_catalogue_estimators() -> None:
    """The mass channel runs under lcat / absolute / generator_marginal."""
    seen = []
    for mode in ("lcat", "absolute", "generator_marginal"):
        res = run_coverage(dataclasses.replace(TINY_MASS, mixture_mode=mode))["results"]["0.7200"]
        assert np.isfinite(res["mass_channel_2d"]["map_mean"])
        seen.append(res["mass_channel_2d"]["map_mean"])
    assert len(seen) == 3


def test_production_n_scale_run_is_finite() -> None:
    """Production-N capability ([A3] criterion (ii)): a 1-realization N=400 cell."""
    cfg = dataclasses.replace(TINY_MASS, n_events=400, n_realizations=1, n_galaxies=60_000)
    res = run_coverage(cfg)["results"]["0.7200"]
    assert np.isfinite(res["map_mean"])
    assert np.isfinite(res["mass_channel_2d"]["map_mean"])


def test_event_chunk_does_not_change_results() -> None:
    """The vectorization block size is a performance knob only."""
    a = run_coverage(dataclasses.replace(TINY_MASS, event_chunk=4))
    b = run_coverage(dataclasses.replace(TINY_MASS, event_chunk=64))
    assert a["results"]["0.7200"]["map_mean"] == b["results"]["0.7200"]["map_mean"]
    assert (
        a["results"]["0.7200"]["mass_channel_2d"]["map_mean"]
        == b["results"]["0.7200"]["mass_channel_2d"]["map_mean"]
    )


def test_mass_horizon_index_changes_the_venue() -> None:
    """A mass-blind cell (alpha_M = 0) and a mass-bearing one differ."""
    blind = run_coverage(dataclasses.replace(TINY_MASS, mass_horizon_index=0.0))["results"][
        "0.7200"
    ]
    bearing = run_coverage(TINY_MASS)["results"]["0.7200"]
    assert blind["map_mean"] != bearing["map_mean"]


# ---------------------------------------------------------------------------
# Noise-model cells (Q-0 audit)
# ---------------------------------------------------------------------------


def test_no_scatter_keeps_the_random_stream_aligned() -> None:
    """The discarded draw keeps the sky/host draws identical to the scatter cell.

    ``gw_measurement_scatter=False`` removes the measurement noise but still
    consumes its RNG draw, so every downstream generative quantity (cap
    centres, hence ball occupancy) is untouched — which is what makes the
    scatter/no-scatter A/B a paired comparison.
    """
    scatter = run_coverage(TINY_MASS)["results"]["0.7200"]
    clean = run_coverage(dataclasses.replace(TINY_MASS, gw_measurement_scatter=False))["results"][
        "0.7200"
    ]
    assert scatter["mean_ball_size"] == clean["mean_ball_size"]
    assert scatter["host_in_ball_fraction"] == clean["host_in_ball_fraction"]
    assert scatter["completion_fraction"] == clean["completion_fraction"]
    # ... while the inference result does move (the observables changed).
    assert scatter["map_mean"] != clean["map_mean"] or scatter["map_std"] != clean["map_std"]


def test_noise_model_cells_are_three_distinct_venues() -> None:
    """const / model / production are three different cells of the #66/#67 ladder."""
    fine = dataclasses.replace(TINY_MASS, h_step=0.004)
    const = run_coverage(fine)["results"]["0.7200"]
    model = run_coverage(dataclasses.replace(fine, sigma_dl_model_in_likelihood=True))["results"][
        "0.7200"
    ]
    production = run_coverage(dataclasses.replace(fine, gw_measurement_scatter=False))["results"][
        "0.7200"
    ]
    assert const != model
    assert const != production
    assert model != production


def test_noise_model_cli_maps_to_the_three_cells(tmp_path: Path) -> None:
    """``--noise-model`` sets the (sigma-model, scatter) pair for each cell."""
    expected = {
        "const": (False, True),
        "model": (True, True),
        "production": (False, False),
    }
    for cell, (sig_model, scatter) in expected.items():
        out = tmp_path / f"{cell}.json"
        main(
            [
                "--n-realizations",
                "1",
                "--n-events",
                "5",
                "--truths",
                "0.72",
                "--noise-model",
                cell,
                "--output",
                str(out),
            ]
        )
        cfg = json.loads(out.read_text())["config"]
        assert cfg["sigma_dl_model_in_likelihood"] is sig_model
        assert cfg["gw_measurement_scatter"] is scatter


def test_mass_cli_flags_thread_into_config(tmp_path: Path) -> None:
    out = tmp_path / "mass.json"
    main(
        [
            "--n-realizations",
            "1",
            "--n-events",
            "10",
            "--truths",
            "0.72",
            "--catalogue-mode",
            "--z-support",
            "0.30",
            "--mixture-mode",
            "absolute",
            "--n-galaxies",
            "20000",
            "--sky-frac",
            "4e-4",
            "--mass-channel",
            "--mass-horizon-index",
            "0.25",
            "--selection-cell",
            "fused",
            "--n-hermite",
            "16",
            "--event-chunk",
            "8",
            "--output",
            str(out),
        ]
    )
    payload = json.loads(out.read_text())
    cfg = payload["config"]
    assert cfg["mass_channel"] is True
    assert cfg["mass_horizon_index"] == 0.25
    assert cfg["selection_cell"] == "fused"
    assert cfg["n_hermite"] == 16
    assert cfg["event_chunk"] == 8
    assert "mass_channel_2d" in payload["results"]["0.7200"]


# ---------------------------------------------------------------------------
# Byte-identity of the pre-existing modes ([A3] regression guarantee)
# ---------------------------------------------------------------------------

# Recorded from the harness at commit 07bbecc9 (before the mass-channel
# extension), 2026-08-17. Any drift in these numbers means an existing mode
# stopped being bit-identical.
GOLDEN_CONTINUUM = {
    "map_mean": 0.7200000000000001,
    "map_std": 0.0,
    "dlogL_dh_host_mean": 0.2133078673865659,
}
GOLDEN_CATALOGUE = {
    "map_mean": 0.7600000000000001,
    "map_std": 0.016329931618554536,
    "completion_fraction": 0.30666666666666664,
    "empty_ball_fraction": 0.12,
    "mean_ball_size": 1.32,
    "host_in_ball_fraction": 0.6933333333333334,
    "impostor_fraction": 0.47743821920392376,
}


def test_pre_mass_continuum_mode_is_byte_identical() -> None:
    cfg = PPCoverageConfig(
        n_realizations=3, n_events=25, injected_truths=[0.72], seed=20260817, h_step=0.02
    )
    res = run_coverage(cfg)["results"]["0.7200"]
    for key, value in GOLDEN_CONTINUUM.items():
        assert res[key] == value, f"{key}: {res[key]!r} != {value!r}"
    assert "mass_channel_2d" not in res


def test_pre_mass_catalogue_mode_is_byte_identical() -> None:
    cfg = PPCoverageConfig(
        n_realizations=3,
        n_events=25,
        injected_truths=[0.72],
        seed=20260726,
        kernel="volume",
        catalogue_mode=True,
        z_support=0.30,
        mixture_mode="absolute",
        n_galaxies=40_000,
        sky_frac=4.0e-4,
        h_step=0.02,
    )
    res = run_coverage(cfg)["results"]["0.7200"]
    for key, value in GOLDEN_CATALOGUE.items():
        assert res[key] == value, f"{key}: {res[key]!r} != {value!r}"
    assert "mass_channel_2d" not in res


def test_mass_reference_scale_is_the_documented_constant() -> None:
    """Guard the mass units the horizon rescaling is anchored on."""
    assert M_REF_MSUN == 1.0e6
    assert (M_SOURCE_MIN, M_SOURCE_MAX) == (1.0e4, 1.0e7)
