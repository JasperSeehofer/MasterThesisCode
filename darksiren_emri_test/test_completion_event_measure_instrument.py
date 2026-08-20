"""Regression tests for the B-DEN falsifier instrument
(``--completion_event_measure {ratio,data}``).

``docs/derivations/completion_numerator_data_measure.md`` §2/§6 (author
approval 2026-08-20; AMENDMENT A-5,
results/prod2d_closure_20260818/PREREGISTRATION_1D_CORRESPONDENCE.md): the
completion numerator's GW event term ``p_gw`` is currently a density in the
dimensionless distance ratio ``d_L(z;h)/d_L,det`` (``"ratio"``,
``bayesian_statistics.py`` ~:4904-4950), which does NOT integrate to 1 over
the observable ``d_L,det`` -- it is proportional to ``d_L(z;h)`` (memo §2).
``"data"`` re-expresses the SAME Gaussian measurement model as a density in
the observable, ``N(d_L,det; d_L(z;h), sigma_frac*d_L(z;h))``, which DOES
integrate to ~1.

Gates pinned here (the memo §6 gate presentation):
  (i)   default "ratio" is byte-identical to the pre-flag path.
  (ii)  sigma_frac -> 0: both forms collapse to the same pinned single-event
        likelihood shape (delta at d_L = d_L,det).
  (iii) the memo §2 defect itself: integral dd_L,det of the "data" form is
        ~1; the SAME integral of the "ratio" form is ~d_L (NOT 1) -- both
        asserted, since the contrast IS the defect.
  (iv)  an unknown mode is a hard error, never a silent fallback.

CPU-only; no GPU, no pool (the completion quadrature runs in the parent
process -- ``p_Di`` uses the worker pool only for the catalogue legs).
"""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from darksiren_emri.arguments import Arguments
from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics

_H_GRID = (0.62, 0.73, 0.84)

_Z_TABLE = np.linspace(1e-6, 1.0, 2001, dtype=np.float64)


def _run_p_Di(
    *,
    h: float,
    event_measure: str | None,
    sigma_frac: float = 0.10,
) -> dict[str, Any]:
    """Run ``p_Di``'s completion-leg quadrature at ``h``; return its diagnostic row.

    Harness mirrors ``test_selection_in_completion_numerator.py``'s
    ``_run_p_Di``: ``object.__new__`` installs only the attributes ``p_Di``
    touches. ``event_measure`` of ``None`` leaves the attribute unset
    entirely, which is how a legacy (pre-flag) instance looks to the
    ``getattr`` guard.
    """
    instance = object.__new__(BayesianStatistics)
    instance.h = h
    instance._normalization_mode = "volume_deconv"
    instance.catalog_only = False
    instance.posterior_data = {}
    instance.posterior_data_with_bh_mass = {
        "galaxy_likelihoods": {},
        "additional_galaxies_without_bh_mass": {},
    }
    instance._diagnostic_rows = []
    if event_measure is not None:
        instance._completion_event_measure = event_measure

    mock_detection = MagicMock()
    mock_detection.d_L = 1.0
    mock_detection.d_L_uncertainty = 0.1
    mock_detection.phi = 0.5
    mock_detection.theta = 0.5
    mock_detection.M = 1.0e6
    instance.detection = mock_detection

    instance._det_index_to_slot = {0: 0}
    instance._means_3d = np.array([[0.5, 0.5, 1.0]])
    instance._cov_inv_3d = np.array([np.linalg.inv(np.diag([1.0, 1.0, sigma_frac**2]))])
    instance._log_norm_3d = np.array([0.0])
    instance._det_d_L = np.array([1.0])

    instance._D_h_table = {h: 1.520637e9}
    instance._beta_Gbar_table = {h: 1.335874e9}
    instance._beta_G_table = {h: 1.520637e9 - 1.335874e9}
    instance._global_cat_denom_no_bh = {h: 1.075654e9}
    instance._global_cat_denom_with_bh = {h: 4.221903e8}
    instance._use_phi_selection = False
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
        np.asarray(z, dtype=np.float64), 0.5
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
    row: dict[str, Any] = instance._diagnostic_rows[0]
    return row


# ===========================================================================
# (i) default "ratio" byte-identity
# ===========================================================================
@pytest.mark.parametrize("h", _H_GRID)
def test_ratio_is_bit_identical_to_the_unflagged_instance(h: float) -> None:
    """ "ratio" and "attribute absent" produce EXACTLY the same row."""
    legacy = _run_p_Di(h=h, event_measure=None)
    ratio = _run_p_Di(h=h, event_measure="ratio")

    assert ratio.keys() == legacy.keys()
    for key in legacy:
        np.testing.assert_equal(ratio[key], legacy[key], err_msg=key)


@pytest.mark.parametrize("h", _H_GRID)
def test_data_mode_actually_moves_b_num(h: float) -> None:
    """Sanity: the "data" cell is not a silent no-op."""
    ratio = _run_p_Di(h=h, event_measure="ratio")
    data = _run_p_Di(h=h, event_measure="data")
    assert data["B_num"] != ratio["B_num"]


# ===========================================================================
# (ii) sigma_frac -> 0 collapses both forms onto the same pinned shape
# ===========================================================================
@pytest.mark.parametrize("h", _H_GRID)
def test_sigma_frac_to_zero_collapses_ratio_and_data_forms(h: float) -> None:
    """As sigma_frac -> 0 both event-term forms pin to the same delta at
    d_L(z;h) = d_L,det -- the single-event likelihood SHAPE in z converges
    (memo §6 limiting case). Compared via the completion numerator B_num,
    which is dominated entirely by the (very narrow) event term at small
    sigma_frac.
    """
    tiny = 1.0e-4
    ratio = _run_p_Di(h=h, event_measure="ratio", sigma_frac=tiny)
    data = _run_p_Di(h=h, event_measure="data", sigma_frac=tiny)
    assert ratio["B_num"] == pytest.approx(data["B_num"], rel=1e-3)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="completion_event_measure"):
        BayesianStatistics.evaluate(
            object.__new__(BayesianStatistics),
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            completion_event_measure="bogus",
        )


# ===========================================================================
# (iii) the memo §2 defect: integral dd_L,det of "data" ~= 1; of "ratio" ~= d_L
# ===========================================================================
def _p_gw_ratio(d_hat: npt.NDArray[np.float64], d_L: float, sigma_frac: float) -> Any:
    return norm.pdf(d_L / d_hat, loc=1.0, scale=sigma_frac)


def _p_gw_data(d_hat: npt.NDArray[np.float64], d_L: float, sigma_frac: float) -> Any:
    return norm.pdf(d_hat, loc=d_L, scale=sigma_frac * d_L)


@pytest.mark.parametrize("d_L,sigma_frac", [(1.0, 0.10), (2.5, 0.05), (0.5, 0.20)])
def test_data_form_integrates_to_one_over_d_hat(d_L: float, sigma_frac: float) -> None:
    """Direct numerical check of the memo §2/§6 claim: the "data" density
    integrates to ~1 over the observable, at a representative (d_L, sigma_frac).
    """
    integral, _ = quad(_p_gw_data, 1.0e-6, np.inf, args=(d_L, sigma_frac), limit=200)
    assert integral == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("d_L,sigma_frac", [(1.0, 0.10), (2.5, 0.05), (0.5, 0.20)])
def test_ratio_form_integrates_to_d_l_not_one(d_L: float, sigma_frac: float) -> None:
    """The SAME integral of the current ("ratio") form is proportional to
    d_L, NOT 1 -- this contrast IS the memo §2 defect the "data" flag fixes.

    Memo §2: integral dd_hat N(d_L/d_hat; 1, sigma) = d_L * (1 + 3*sigma^2 +
    O(sigma^4)) (delta-method expansion of E[1/U^2], U ~ N(1,sigma)). Checked
    to leading order (loose tolerance -- the higher-order terms are not
    negligible at sigma_frac=0.20) and, decisively, that it is NOT ~1 in
    units of d_L.
    """
    integral, _ = quad(_p_gw_ratio, 1.0e-6, np.inf, args=(d_L, sigma_frac), limit=200)
    leading_order = d_L * (1.0 + 3.0 * sigma_frac**2)
    assert integral == pytest.approx(leading_order, rel=0.05)
    # The decisive contrast: in units of d_L this is NOT ~1 -- it moves with
    # sigma_frac, unlike the "data" form's integral (previous test).
    assert integral / d_L != pytest.approx(1.0, rel=5e-3)


# ===========================================================================
# CLI plumbing
# ===========================================================================
def test_cli_flag_defaults_to_ratio() -> None:
    args = Arguments.create([".", "--evaluate"])
    assert args.completion_event_measure == "ratio"
    assert args.to_dict()["completion_event_measure"] == "ratio"


def test_cli_flag_parses_and_lands_in_run_metadata_dict() -> None:
    args = Arguments.create([".", "--evaluate", "--completion_event_measure", "data"])
    assert args.completion_event_measure == "data"
    assert args.to_dict()["completion_event_measure"] == "data"


def test_cli_flag_rejects_unknown_values() -> None:
    with pytest.raises(SystemExit):
        Arguments.create([".", "--evaluate", "--completion_event_measure", "bogus"])
