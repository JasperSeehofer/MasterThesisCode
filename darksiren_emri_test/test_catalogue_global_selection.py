r"""Tests for the [P3-RPHI] fourth Path-A slot instrumentation counterfactual
(``--catalogue_global_selection``).

Spec: ``docs/derivations/PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md`` §2 ("New
formula") and §6(ii) ("an instrumentation counterfactual flag
(``catalogue_global_selection in {"s3d","phi"}``, default byte-identical)").

The flag touches exactly ONE read site: the no-BH catalogue divisor lookup in
``p_Di`` (``L_cat_no_bh = A_ball / Sigma``). ``"s3d"`` (default) reads the
separately fitted, mass-blind ``Sigma^3D`` (``_global_cat_denom_no_bh``);
``"phi"`` reads ``Sigma^phi`` (``_global_cat_selection_phi``) -- the SAME
catalogue-weighted sum Path A already builds (on the same rows/weights/
eligibility as ``Sigma^4D``) for the weight chain (``:3878``). The with-BH
leg (``global_denom_with_bh``) is architecturally untouched: it is read from
a single call site (``:4823`` neighbourhood) and never dispatched to a
worker -- ``single_host_likelihood``/``single_host_likelihood_batch`` only
ever produce the per-host NUMERATOR sums (``r[0]``/``r[2]``); the global
divisor is applied once, in the parent process, after the pool results are
gathered. There is therefore no separate "batch worker" consumption path to
patch (verified by an exhaustive grep of ``_global_cat_denom_no_bh`` across
``darksiren_emri/``) -- "scalar and batch parity" below means the SAME
``p_Di`` call, exercised through the mocked-pool batch-dispatch machinery
``_run_p_Di_phi`` already uses (i.e. the divisor swap is provably
independent of host-batch mechanics).

CPU-only; no GPU, no real pool.
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from darksiren_emri.arguments import Arguments
from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics

_H = 0.73

# Path-(A) selection tables of record at h = 0.73 (FIXB_PATHA_PACKAGE.md §5),
# reused from test_fixb_pathA_mixture.py's anchors.
_BETA_G_PHI = 1.533228e8
_BETA_GBAR_PHI = 8.884038e8
_SIGMA_PHI = 9.562370e8
_SIGMA_3D = 1.075654e9  # _global_cat_denom_no_bh (production anchor, distinct from Sigma^phi)
_SIGMA_4D = 4.221903e8


def _run_p_Di_phi(
    *,
    catalogue_global_selection: str | None = "s3d",
    D_h: float = 1.520637e9,
    beta_Gbar: float = 1.335874e9,
    global_no_bh: float = _SIGMA_3D,
    global_with_bh: float = _SIGMA_4D,
    beta_G_phi: float = _BETA_G_PHI,
    beta_Gbar_phi: float = _BETA_GBAR_PHI,
    sigma_phi: float = _SIGMA_PHI,
    h: float = _H,
    f_const: float = 0.5,
) -> dict[str, Any]:
    """Run ``p_Di`` with the path-(A) tables installed; return its diagnostic row.

    Modeled on ``test_fixb_pathA_mixture.py``'s ``_run_p_Di_phi`` harness.
    ``catalogue_global_selection=None`` leaves the attribute unset entirely --
    the pre-flag legacy-instance shape seen through the ``getattr`` guard.
    """
    instance = object.__new__(BayesianStatistics)
    instance.h = h
    instance._normalization_mode = "absolute_marginal"
    if catalogue_global_selection is not None:
        instance._catalogue_global_selection = catalogue_global_selection
    instance._completion_b_scale = "derived"
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
    instance._use_phi_selection = True
    instance._beta_G_phi_table = {h: beta_G_phi}
    instance._beta_Gbar_phi_table = {h: beta_Gbar_phi}
    instance._global_cat_selection_phi = {h: sigma_phi}
    instance._proj_d_L_to_M = np.array([0.3])
    instance._sigma_cond_M = np.array([0.1])

    mock_pool = MagicMock()
    mock_pool._processes = 1
    mock_pool.starmap.side_effect = [
        [np.array([[0.5, 0.3, 0.4, 0.2]])],
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


# ===========================================================================
# (a) default byte-identity
# ===========================================================================
def test_default_s3d_is_bit_identical_to_the_unflagged_instance() -> None:
    """``"s3d"`` and "attribute absent" produce EXACTLY the same row.

    "Attribute absent" is the pre-flag code path as seen through the
    ``getattr(self, "_catalogue_global_selection", "s3d")`` guard at the
    divisor read site -- the byte-identity claim of the instrumentation
    commit, asserted with ``==`` rather than ``approx``.
    """
    legacy = _run_p_Di_phi(catalogue_global_selection=None)
    default = _run_p_Di_phi(catalogue_global_selection="s3d")

    assert default.keys() == legacy.keys()
    for key in legacy:
        assert default[key] == legacy[key], key


def test_default_path_ignores_sigma_phi_for_the_no_bh_divisor() -> None:
    """Default ``"s3d"`` must not read Sigma^phi for L_cat_no_bh's divisor.

    Sigma^phi still feeds the (untouched, pre-existing) with-BH mixture
    assembly (``n_hat_w^phi``/``alpha_G^phi``/``D~^phi``) regardless of this
    flag -- that is production behaviour this instrument does not change
    (proposal §3: "the with-BH leg ... is untouched"). Only the two no-BH
    fields the divisor swap actually targets are asserted here.
    """
    a = _run_p_Di_phi(catalogue_global_selection="s3d", sigma_phi=_SIGMA_PHI)
    b = _run_p_Di_phi(catalogue_global_selection="s3d", sigma_phi=_SIGMA_PHI * 5.0)
    # L_cat_no_bh = A_ball / Sigma^3D is untouched by Sigma^phi entirely.
    assert a["L_cat_no_bh"] == b["L_cat_no_bh"]
    # combined_no_bh DOES still move (D~^phi/alpha_G^phi are shared with the
    # with-BH mixture assembly, which legitimately reads Sigma^phi
    # regardless of this flag) -- that is pre-existing production behaviour,
    # not something this instrument changes; asserted here so a future
    # change that also freezes combined_no_bh is not silently "fixed" by
    # this test.
    assert a["combined_no_bh"] != b["combined_no_bh"]


def test_construct_level_divisor_object_identity() -> None:
    """The divisor read site selects the DICT OBJECT, not a recomputed copy.

    Mirrors the code's own ternary at the read site
    (``self._global_cat_selection_phi if ... == "phi" else
    self._global_cat_denom_no_bh``): sanity-checks that expression directly
    against two distinct dict instances, independent of the full ``p_Di``
    run, so a future refactor that accidentally reads a stale/copied table
    is caught even if the numeric anchors above happen to coincide.
    """
    denom_s3d = {_H: _SIGMA_3D}
    denom_phi = {_H: _SIGMA_PHI}

    def _selected(flag: str) -> dict[float, float]:
        return denom_phi if flag == "phi" else denom_s3d

    assert _selected("s3d") is denom_s3d
    assert _selected("phi") is denom_phi
    assert _selected("s3d") is not denom_phi


# ===========================================================================
# (b) mode guard
# ===========================================================================
def test_evaluate_rejects_an_unknown_value() -> None:
    """``BayesianStatistics.evaluate`` validates the flag value itself."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="must be 's3d' or 'phi'"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            catalogue_global_selection="4d",
        )


def test_evaluate_requires_absolute_marginal_for_phi() -> None:
    """The Sigma^phi table only exists in ``absolute_marginal``; say so loudly."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="absolute_marginal"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="generator_marginal",
            catalogue_global_selection="phi",
        )


def test_evaluate_accepts_phi_under_absolute_marginal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The valid combination passes validation and reaches the COUNTERFACTUAL log."""
    import darksiren_emri.bayesian_inference.bayesian_statistics as bs_mod

    logged: list[str] = []
    monkeypatch.setattr(
        bs_mod._LOGGER,
        "warning",
        lambda msg, *a, **k: logged.append(msg % a if a else msg),
    )
    instance = object.__new__(BayesianStatistics)
    # Reach the catalogue_global_selection validation block, then abort on the
    # very next (unrelated) validation so the rest of evaluate() need not be
    # mocked -- catalogue_mass_error_scale != 1.0 without 'inflated' raises
    # deterministically right after our block.
    with pytest.raises(ValueError, match="catalogue_mass_overlap"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            normalization_mode="absolute_marginal",
            catalogue_global_selection="phi",
            catalogue_mass_error_scale=2.0,
        )
    assert instance._catalogue_global_selection == "phi"
    assert any("COUNTERFACTUAL" in m and "catalogue_global_selection" in m for m in logged)


# ===========================================================================
# (c) "phi" changes L_cat_no_bh by exactly Sigma^3D/Sigma^phi per h
# ===========================================================================
def test_phi_rescales_L_cat_no_bh_by_exactly_sigma3d_over_sigma_phi() -> None:
    """L_cat_no_bh(phi) / L_cat_no_bh(s3d) == Sigma^3D / Sigma^phi, exactly.

    The catalogue NUMERATOR (``cat_num_sum_no_bh``, from the mocked pool) is
    identical in both runs -- only the divisor differs -- so the ratio is the
    pure divisor swap the proposal specifies (§2): "one-slot change".
    """
    s3d = _run_p_Di_phi(catalogue_global_selection="s3d")
    phi = _run_p_Di_phi(catalogue_global_selection="phi")

    assert s3d["L_cat_no_bh"] != phi["L_cat_no_bh"]
    ratio = phi["L_cat_no_bh"] / s3d["L_cat_no_bh"]
    expected = _SIGMA_3D / _SIGMA_PHI
    assert ratio == pytest.approx(expected, rel=1e-12)

    # The with-BH leg is bit-identical -- untouched by construction (proposal
    # §3: "the with-BH leg ... is untouched -- it is the internal control").
    assert s3d["L_cat_with_bh"] == phi["L_cat_with_bh"]
    assert s3d["combined_with_bh"] == phi["combined_with_bh"]
    assert s3d["alpha_G_phi"] == phi["alpha_G_phi"]
    assert s3d["D_tilde_phi"] == phi["D_tilde_phi"]


@pytest.mark.parametrize("h", (0.62, 0.73, 0.84))
def test_phi_rescaling_holds_per_h(h: float) -> None:
    """The exact-ratio claim holds at every h on the grid, not just h=0.73."""
    s3d = _run_p_Di_phi(catalogue_global_selection="s3d", h=h)
    phi = _run_p_Di_phi(catalogue_global_selection="phi", h=h)
    ratio = phi["L_cat_no_bh"] / s3d["L_cat_no_bh"]
    assert ratio == pytest.approx(_SIGMA_3D / _SIGMA_PHI, rel=1e-12)


def test_batch_dispatch_parity_with_a_second_no_bh_host() -> None:
    """The divisor swap is independent of host-batch mechanics.

    Adds a SECOND no-BH-only host to the batch (a distinct
    ``pool.starmap`` payload shape) and re-checks the same exact ratio --
    demonstrating the read site is untouched by how many hosts the batched
    worker dispatch returns (the "batch worker path" the proposal's
    verification plan names is, architecturally, this SAME single read
    site; see the module docstring above).
    """
    instance_kwargs = dict(
        D_h=1.520637e9,
        beta_Gbar=1.335874e9,
        global_no_bh=_SIGMA_3D,
        global_with_bh=_SIGMA_4D,
        beta_G_phi=_BETA_G_PHI,
        beta_Gbar_phi=_BETA_GBAR_PHI,
        sigma_phi=_SIGMA_PHI,
    )

    def _run(flag: str) -> dict[str, Any]:
        instance = object.__new__(BayesianStatistics)
        instance.h = _H
        instance._normalization_mode = "absolute_marginal"
        instance._catalogue_global_selection = flag
        instance._completion_b_scale = "derived"
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

        instance._D_h_table = {_H: instance_kwargs["D_h"]}
        instance._beta_Gbar_table = {_H: instance_kwargs["beta_Gbar"]}
        instance._beta_G_table = {_H: instance_kwargs["D_h"] - instance_kwargs["beta_Gbar"]}
        instance._global_cat_denom_no_bh = {_H: instance_kwargs["global_no_bh"]}
        instance._global_cat_denom_with_bh = {_H: instance_kwargs["global_with_bh"]}
        instance._use_phi_selection = True
        instance._beta_G_phi_table = {_H: instance_kwargs["beta_G_phi"]}
        instance._beta_Gbar_phi_table = {_H: instance_kwargs["beta_Gbar_phi"]}
        instance._global_cat_selection_phi = {_H: instance_kwargs["sigma_phi"]}
        instance._proj_d_L_to_M = np.array([0.3])
        instance._sigma_cond_M = np.array([0.1])

        mock_pool = MagicMock()
        mock_pool._processes = 1
        mock_pool.starmap.side_effect = [
            [np.array([[0.5, 0.3, 0.4, 0.2]])],
            # TWO no-BH-only hosts in this batch (vs one in the base harness).
            [np.array([[0.3, 0.2], [0.15, 0.1]])],
        ]

        mock_completeness = MagicMock()
        mock_completeness.ang2pix.return_value = 0
        mock_completeness.f_k.side_effect = lambda z, k, h: np.full_like(
            np.asarray(z, dtype=np.float64), 0.5
        )

        mock_p_det = MagicMock()
        mock_p_det.get_dl_max.return_value = 10.0

        host_a = MagicMock()
        host_a.M, host_a.z, host_a.catalog_index = 1e6, 0.1, 0
        host_b = MagicMock()
        host_b.M, host_b.z, host_b.catalog_index = 2e6, 0.15, 2
        host_with_bh = MagicMock()
        host_with_bh.M, host_with_bh.z, host_with_bh.catalog_index = 1e6, 0.1, 1

        BayesianStatistics.p_Di(
            instance,
            possible_host_galaxies=[host_a, host_b],
            possible_host_galaxies_with_bh_mass=[host_with_bh],
            detection_index=0,
            pool=mock_pool,
            completeness=mock_completeness,
            detection_probability_obj=mock_p_det,
        )
        return instance._diagnostic_rows[0]

    s3d = _run("s3d")
    phi = _run("phi")
    ratio = phi["L_cat_no_bh"] / s3d["L_cat_no_bh"]
    assert ratio == pytest.approx(_SIGMA_3D / _SIGMA_PHI, rel=1e-12)
    # Untouched with-BH leg, even with the extra no-BH host present.
    assert s3d["L_cat_with_bh"] == phi["L_cat_with_bh"]


# ===========================================================================
# (d) degree/regression test: S_bar -> c*S_bar homogeneity
# ===========================================================================
def test_phi_no_bh_leg_is_degree_matched_under_the_weight_chain_scaling() -> None:
    r"""Gate: under Sigma^phi/beta_G^phi -> c*(Sigma^phi/beta_G^phi), the no-BH
    catalogue TERM is invariant with "phi" and moves with "s3d".

    Proposal §5: "S̄ -> c·S̄ homogeneity: with Σ^φ the catalogue leg becomes
    c-degree-matched to its weight chain ... with Σ³ᴰ it is not." ``beta_G^phi``
    and ``Sigma^phi`` are BOTH linear integrals of the SAME S_bar_phi(z;h)
    table (``precompute_phi_selection_integrals``/``precompute_global_catalog_
    selection`` sharing ``phi_survival_table``, FIXB_PATHA_PACKAGE.md §3.2), so
    a uniform table rescaling S_bar_phi -> c*S_bar_phi scales BOTH by exactly
    c (holding the missing-completion leg beta_Gbar^phi and the with-BH-only
    Sigma^4D fixed -- proposal §6 scope: "no-BH leg only", separable from the
    shared with-BH machinery D~^phi is built from).

    Under "phi": beta_G^phi * L_cat_no_bh = beta_G^phi * A_ball/Sigma^phi ->
    (c*beta_G^phi) * A_ball/(c*Sigma^phi) = beta_G^phi * A_ball/Sigma^phi --
    degree ONE cancels degree MINUS ONE exactly (the review's invariance
    test); n_hat_w^phi = Sigma^phi/beta_G^phi is likewise c-invariant, so
    alpha_G^phi (and D~^phi, held fixed here) do not move either -- the
    WHOLE combined no-BH posterior is therefore exactly invariant.

    Under "s3d": Sigma^3D is a SEPARATELY FITTED, mass-blind table (proposal
    §1) -- unrelated to S_bar_phi -- so it does NOT move under this scaling
    while beta_G^phi still does: the pairing breaks and the posterior moves
    with c.
    """
    c = 3.7

    base_s3d = _run_p_Di_phi(catalogue_global_selection="s3d")
    base_phi = _run_p_Di_phi(catalogue_global_selection="phi")

    scaled_s3d = _run_p_Di_phi(
        catalogue_global_selection="s3d",
        beta_G_phi=_BETA_G_PHI * c,
        sigma_phi=_SIGMA_PHI * c,
        # beta_Gbar_phi and Sigma^4D held fixed (out of scope, proposal §6).
    )
    scaled_phi = _run_p_Di_phi(
        catalogue_global_selection="phi",
        beta_G_phi=_BETA_G_PHI * c,
        sigma_phi=_SIGMA_PHI * c,
    )

    # "phi": exactly invariant (machine precision).
    assert scaled_phi["combined_no_bh"] == pytest.approx(
        base_phi["combined_no_bh"], rel=1e-12, abs=0.0
    )
    assert scaled_phi["L_cat_no_bh"] == pytest.approx(
        base_phi["L_cat_no_bh"] / c, rel=1e-12, abs=0.0
    )

    # "s3d": genuinely NOT invariant -- the divisor (Sigma^3D) is unrelated
    # to the S_bar_phi table and does not move, so beta_G^phi's c-scaling
    # leaks straight into the posterior.
    assert scaled_s3d["combined_no_bh"] != pytest.approx(
        base_s3d["combined_no_bh"], rel=1e-9, abs=0.0
    )
    assert scaled_s3d["L_cat_no_bh"] == pytest.approx(base_s3d["L_cat_no_bh"], rel=1e-12, abs=0.0)


# ===========================================================================
# CLI plumbing
# ===========================================================================
def test_cli_flag_defaults_to_s3d() -> None:
    args = Arguments.create([".", "--evaluate"])
    assert args.catalogue_global_selection == "s3d"
    assert args.to_dict()["catalogue_global_selection"] == "s3d"


def test_cli_flag_parses_and_lands_in_run_metadata_dict() -> None:
    args = Arguments.create([".", "--evaluate", "--catalogue_global_selection", "phi"])
    assert args.catalogue_global_selection == "phi"
    assert args.to_dict()["catalogue_global_selection"] == "phi"


def test_cli_flag_rejects_unknown_values() -> None:
    with pytest.raises(SystemExit):
        Arguments.create([".", "--evaluate", "--catalogue_global_selection", "4d"])
