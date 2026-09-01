r"""[HIER] theta-consistent candidate z-window plumbing regression gates (row
#255 tree 2 node T1.3-zwin, results/campaign51_20260728/realistic_20260729/
tree2_20260830/PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md).

``BayesianStatistics.evaluate()`` gains two new instrument flags
(``theta_zwindow``/``z_window_k``) that are validated/stored and passed
through to ``GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree`` at
the single call site. Defaults ("off"/1.0) are byte-identical.

Gates encoded here (section 7 of the presentation, the plumbing half --
the handler-level mask itself is pinned in
``darksiren_emri_test/test_theta_zwindow.py``):
R5  -- guards (invalid theta_zwindow token raises at evaluate()).
R7  -- CLI/evaluate() plumbing defaults are byte-identical.

Plus a unit test of the driver's PA-HIER-32(d) scorer arithmetic
(``hier_s0_driver.compute_scores``'s corrected ``score_s = score_lns -
Es_null_det``), loaded directly from its results/ path (no test package
there -- see the module-loading shim below).
"""

import importlib.util
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics

# ===========================================================================
# Guards
# ===========================================================================


def test_evaluate_rejects_invalid_theta_zwindow_token() -> None:
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="theta_zwindow must be 'off' or 'on'"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            theta_zwindow="maybe",
        )


def test_theta_zwindow_stored_before_a_later_validation_raises() -> None:
    """Plumbing reach test (mirrors test_theta_phi_divisor.py's
    ``test_sky_cone_k_is_stored_before_a_later_validation_raises``): a
    deliberate abort at a LATER guard still proves theta_zwindow/z_window_k
    reached ``self._theta_zwindow``/``self._z_window_k``."""
    instance = object.__new__(BayesianStatistics)
    with pytest.raises(ValueError, match="theta_phi_divisor must be 'off' or 'on'"):
        BayesianStatistics.evaluate(
            instance,
            galaxy_catalog=MagicMock(),
            cosmological_model=MagicMock(),
            h_value=0.73,
            theta_zwindow="on",
            z_window_k=4.0,
            theta_phi_divisor="bogus",
        )
    assert instance._theta_zwindow == "on"
    assert instance._z_window_k == 4.0


# ===========================================================================
# R7 -- CLI / evaluate() plumbing defaults
# ===========================================================================


def test_default_instrument_attributes_are_byte_identical() -> None:
    """Class-level defaults: 'off' / 1.0 -- the pre-flag literals.

    ``object.__new__`` (not ``BayesianStatistics()``) so this does not
    require a CWD with ``simulations/prepared_cramer_rao_bounds.csv``."""
    instance = object.__new__(BayesianStatistics)
    assert instance._theta_zwindow == "off"
    assert instance._z_window_k == 1.0


def test_arguments_defaults_are_byte_identical() -> None:
    from darksiren_emri.arguments import Arguments

    args = Arguments.create([".", "--evaluate"])
    assert args.theta_zwindow == "off"
    assert args.z_window_k == 1.0


def test_arguments_theta_zwindow_and_z_window_k_forward() -> None:
    from darksiren_emri.arguments import Arguments

    args = Arguments.create([".", "--evaluate", "--theta_zwindow", "on", "--z_window_k", "4.0"])
    assert args.theta_zwindow == "on"
    assert args.z_window_k == 4.0


def test_arguments_rejects_invalid_theta_zwindow_choice() -> None:
    from darksiren_emri.arguments import Arguments

    with pytest.raises(SystemExit):
        Arguments.create([".", "--evaluate", "--theta_zwindow", "bogus"])


# ===========================================================================
# The driver's PA-HIER-32(d) scorer arithmetic (score_lns / score_s /
# score_s_raw), loaded directly from its results/ path.
# ===========================================================================

_DRIVER_PATH = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "campaign51_20260728"
    / "realistic_20260729"
    / "fanout1_20260829"
    / "hier_s0_driver.py"
)


def _load_driver_module() -> object:
    spec = importlib.util.spec_from_file_location("hier_s0_driver_under_test", _DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver() -> object:
    if not _DRIVER_PATH.is_file():
        pytest.skip(f"driver script not found at {_DRIVER_PATH}")
    return _load_driver_module()


def _node_result(driver: object, node: str, theta_b: float, theta_s: float, ln_l: pd.DataFrame):  # type: ignore[no-untyped-def]
    return driver.NodeResult(  # type: ignore[attr-defined]
        node=node,
        theta_b=theta_b,
        theta_s=theta_s,
        seed=1,
        diag_csv="unused",
        elapsed_s=0.0,
        n_events=len(ln_l),
        ln_l=ln_l,
    )


def test_compute_scores_score_lns_and_raw_use_the_registered_denominators(driver: object) -> None:
    """score_s_raw's denominator is (sqrt2 - 1/sqrt2); score_lns's is ln(2)
    -- SAME numerator (s_plus - s_minus), both hand-verified against the
    prereg's own quoted forms (PA-HIER-4 / PREREGISTRATION_HIER_
    HTHETA_20260826.md lines ~843-867)."""
    truth = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.0, 0.0], "ln_L_with_bh": [0.0, 0.0]}
    )
    b_plus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.1, 0.2], "ln_L_with_bh": [0.1, 0.2]}
    )
    b_minus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [-0.1, -0.2], "ln_L_with_bh": [-0.1, -0.2]}
    )
    s_plus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.30, 0.40], "ln_L_with_bh": [0.30, 0.40]}
    )
    s_minus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.10, 0.20], "ln_L_with_bh": [0.10, 0.20]}
    )

    all_nodes = {
        "truth": [_node_result(driver, "truth", 0.0, 1.0, truth)],
        "b_plus": [_node_result(driver, "b_plus", 0.02, 1.0, b_plus)],
        "b_minus": [_node_result(driver, "b_minus", -0.02, 1.0, b_minus)],
        "s_plus": [_node_result(driver, "s_plus", 0.0, math.sqrt(2.0), s_plus)],
        "s_minus": [_node_result(driver, "s_minus", 0.0, 1.0 / math.sqrt(2.0), s_minus)],
    }
    scores = driver.compute_scores(all_nodes, seeds=(1,))  # type: ignore[attr-defined]
    channel = scores["ln_L_no_bh"]

    # score_s_raw: per-event (s_plus - s_minus) / (sqrt2 - 1/sqrt2); both
    # events have the SAME 0.20 numerator here, so mean == per-event value.
    expected_raw_per_event = 0.20 / (math.sqrt(2.0) - 1.0 / math.sqrt(2.0))
    assert channel["score_s_raw"]["mean"] == pytest.approx(expected_raw_per_event)
    assert channel["score_s_raw"]["n_pooled"] == 2

    # score_lns: SAME numerator, denominator ln(2).
    expected_lns_per_event = 0.20 / math.log(2.0)
    assert channel["score_lns"]["mean"] == pytest.approx(expected_lns_per_event)
    assert channel["score_lns"]["n_pooled"] == 2

    # No es_null_det column anywhere -> the corrected score_s is reported
    # unavailable, NEVER silently equal to score_lns.
    assert channel["score_s_available"] is False
    assert channel["score_s"]["n_pooled"] == 0
    assert math.isnan(channel["score_s"]["mean"])

    # score_b unaffected by any of this (PA-HIER-32(d): "score_b is not
    # affected").
    expected_b_per_event = 0.20 / 0.04  # (b_plus - b_minus)/0.04, event 0
    assert channel["score_b"]["n_pooled"] == 2


def test_compute_scores_score_s_corrected_subtracts_es_null_det(driver: object) -> None:
    """score_s = score_lns - Es_null_det, per event, THEN pooled -- verified
    against an independent by-hand computation (PA-HIER-32(d)'s registered
    form, quoted in this module's docstring)."""
    s_plus = pd.DataFrame(
        {
            "event_idx": [0, 1, 2],
            "ln_L_no_bh": [0.30, 0.55, 0.40],
            "ln_L_with_bh": [0.30, 0.55, 0.40],
            "es_null_det": [0.02, 0.03, 0.01],
        }
    )
    s_minus = pd.DataFrame(
        {
            "event_idx": [0, 1, 2],
            "ln_L_no_bh": [0.10, 0.20, 0.15],
            "ln_L_with_bh": [0.10, 0.20, 0.15],
        }
    )
    b_plus = pd.DataFrame(
        {"event_idx": [0, 1, 2], "ln_L_no_bh": [0.1, 0.1, 0.1], "ln_L_with_bh": [0.1, 0.1, 0.1]}
    )
    b_minus = pd.DataFrame(
        {
            "event_idx": [0, 1, 2],
            "ln_L_no_bh": [-0.1, -0.1, -0.1],
            "ln_L_with_bh": [-0.1, -0.1, -0.1],
        }
    )

    all_nodes = {
        "truth": [],
        "b_plus": [_node_result(driver, "b_plus", 0.02, 1.0, b_plus)],
        "b_minus": [_node_result(driver, "b_minus", -0.02, 1.0, b_minus)],
        "s_plus": [_node_result(driver, "s_plus", 0.0, math.sqrt(2.0), s_plus)],
        "s_minus": [_node_result(driver, "s_minus", 0.0, 1.0 / math.sqrt(2.0), s_minus)],
    }
    scores = driver.compute_scores(all_nodes, seeds=(1,))  # type: ignore[attr-defined]
    channel = scores["ln_L_no_bh"]

    assert channel["score_s_available"] is True
    numerators = [0.30 - 0.10, 0.55 - 0.20, 0.40 - 0.15]
    es_null_det = [0.02, 0.03, 0.01]
    expected_score_s = [n / math.log(2.0) - e for n, e in zip(numerators, es_null_det, strict=True)]
    expected_mean = sum(expected_score_s) / len(expected_score_s)
    assert channel["score_s"]["mean"] == pytest.approx(expected_mean)
    assert channel["score_s"]["n_pooled"] == 3
    # score_s_raw is STILL computed and reported (continuity), unaffected
    # by the Es_null_det correction.
    assert channel["score_s_raw"]["n_pooled"] == 3
    assert channel["score_s_raw"]["mean"] != pytest.approx(channel["score_s"]["mean"])


def test_compute_scores_handles_the_registered_p1_node_list_with_no_b_axis(driver: object) -> None:
    """The registered P1 arm's own node list is {truth, s_plus, s_minus} --
    NO b_plus/b_minus at all (PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md
    section 5.6: "b-axis: NOT re-run under P1", T1.2's own certification
    stands). Before this node's per-axis relaxation, compute_scores
    unconditionally required all 4 of b_plus/b_minus/s_plus/s_minus and
    would have raised ValueError on exactly this shape -- this is the
    registered arm's own node dict, not a hypothetical."""
    truth = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.0, 0.0], "ln_L_with_bh": [0.0, 0.0]}
    )
    s_plus = pd.DataFrame(
        {
            "event_idx": [0, 1],
            "ln_L_no_bh": [0.30, 0.40],
            "ln_L_with_bh": [0.30, 0.40],
            "es_null_det": [0.02, 0.02],
        }
    )
    s_minus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.10, 0.20], "ln_L_with_bh": [0.10, 0.20]}
    )
    all_nodes = {
        "truth": [_node_result(driver, "truth", 0.0, 1.0, truth)],
        "s_plus": [_node_result(driver, "s_plus", 0.0, math.sqrt(2.0), s_plus)],
        "s_minus": [_node_result(driver, "s_minus", 0.0, 1.0 / math.sqrt(2.0), s_minus)],
    }
    # No "b_plus"/"b_minus" keys at all -- exactly the dict shape run_arm's
    # {n: [] for n in nodes} comprehension produces for --nodes
    # truth,s_plus,s_minus.
    scores = driver.compute_scores(all_nodes, seeds=(1,))  # type: ignore[attr-defined]
    channel = scores["ln_L_no_bh"]
    assert channel["score_b_available"] is False
    assert channel["score_b"]["n_pooled"] == 0
    assert math.isnan(channel["score_b"]["mean"])
    assert channel["score_s_available"] is True
    assert channel["score_s"]["n_pooled"] == 2
    assert channel["score_s_raw"]["n_pooled"] == 2


def test_gate_eng_handles_a_b_only_node_dict_with_no_truth_node(driver: object) -> None:
    """runner-11's 8-cell [HIER] b-node pair (seeds 900101-900104 x
    b_plus/b_minus, zwin-on zk4, divisor-on) is a b-only run with NO
    "truth" node produced at all -- gate_eng previously indexed
    ``all_nodes["truth"]`` unconditionally (hier_s0_driver.py, pre-fix)
    and raised ``KeyError`` for exactly this node dict, crashing run_arm's
    unconditional ``gate_eng(all_nodes)`` call at the b/s "axis ready" gate
    (rows #280/#287; mirrors the 2026-08-30 addendum's per-axis
    relaxation of compute_scores/run_arm/score_only_payload for the SAME
    missing-node shape, here applied to gate_eng). It must degrade to
    ``eng_available=False`` per off-truth node instead of raising."""
    b_plus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.1, 0.2], "ln_L_with_bh": [0.1, 0.2]}
    )
    b_minus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [-0.1, -0.2], "ln_L_with_bh": [-0.1, -0.2]}
    )
    # No "truth" key at all -- exactly the dict shape a b-only
    # --nodes b_plus,b_minus run produces.
    all_nodes = {
        "b_plus": [_node_result(driver, "b_plus", 0.02, 1.0, b_plus)],
        "b_minus": [_node_result(driver, "b_minus", -0.02, 1.0, b_minus)],
    }

    eng = driver.gate_eng(all_nodes)  # type: ignore[attr-defined]

    for node in ("b_plus", "b_minus", "s_plus", "s_minus"):
        assert eng[node]["eng_available"] is False
        assert eng[node]["per_seed_fraction_moved"] == []
        assert math.isnan(eng[node]["mean_fraction_moved"])
        assert eng[node]["pass"] is False
    assert "truth" not in eng


def test_compute_scores_raises_on_broken_axis_pair(driver: object) -> None:
    """A LONE b_plus with no b_minus (a genuinely broken pair, as opposed to
    an axis simply never requested) must still raise -- the relaxation only
    tolerates a WHOLLY absent axis, never a half-present one."""
    b_plus = pd.DataFrame({"event_idx": [0], "ln_L_no_bh": [0.1], "ln_L_with_bh": [0.1]})
    all_nodes = {"truth": [], "b_plus": [_node_result(driver, "b_plus", 0.02, 1.0, b_plus)]}
    with pytest.raises(ValueError, match="incomplete node pair"):
        driver.compute_scores(all_nodes, seeds=(1,))  # type: ignore[attr-defined]


def test_compute_scores_raises_when_neither_axis_present(driver: object) -> None:
    all_nodes: dict = {"truth": []}
    with pytest.raises(ValueError, match="both axes are incomplete"):
        driver.compute_scores(all_nodes, seeds=(1,))  # type: ignore[attr-defined]


def test_compute_es_null_det_closed_form_matches_delta_limit(driver: object) -> None:
    """s -> 0 (a near-delta kernel): the +/- k sigma window collapses
    around z_g, and (since S_bar_phi/completeness is locally ~constant
    there) the secant of ln(kernel) w.r.t. s at a fixed evaluation point
    approaches the analytic Gaussian-only value -- a coarse but real
    consistency check on :func:`_es_null_det_closed_form`'s wiring, run at
    a small n_grid for speed."""
    import numpy as np

    class _FlatCompleteness:
        def f_k(self, z: object, k: int, h: float) -> object:
            return np.ones_like(np.asarray(z, dtype=float))

    z_g = np.array([0.10])
    sigma_g = np.array([0.01])
    host_pixels = np.array([0], dtype=np.int64)
    es = driver._es_null_det_closed_form(  # type: ignore[attr-defined]
        z_g, sigma_g, host_pixels, _FlatCompleteness(), h=0.73, n_grid=801
    )
    assert es.shape == (1,)
    assert np.isfinite(es[0])
    # PA-HIER-32(d): E[score_lns | truth]_unweighted is small and POSITIVE
    # (+0.0455 +/- 0.0005 per unit s, measured on the real GLADE catalogue
    # with survival weighting; this flat-completeness single-host toy is not
    # expected to reproduce that number, only its sign and rough magnitude
    # under a comoving-volume-weighted kernel with no survival curvature).
    assert es[0] > 0.0
    assert es[0] < 1.0


def test_es_null_det_closed_form_uses_the_ln2_secant_denominator(driver: object) -> None:
    """MUST_FIX from T1_3_ZWINDOW_VERIFIER_REPORT.md item 3: PA-HIER-32(d)
    defines ``Es_null_det_i`` as "the closed-form expectation of
    ``score_lns_i``" -- i.e. the secant's denominator must be ``ln(2)``
    (``score_lns``'s own denominator), NOT ``sqrt2 - 1/sqrt2``
    (``score_s_raw``'s denominator, what the pre-fix code used in error).

    Independently re-derives the RAW-denominator form using the SAME
    per-host kernel/window machinery (:func:`driver._es_null_det_kernel`),
    changing only the secant's denominator, and pins the registered ratio
    between the two forms: ``Es_null_det(ln2 form) = Es_null_det(raw form) *
    (sqrt2 - 1/sqrt2) / ln(2)`` -- the SAME per-host weighted-average
    numerator divided by a smaller denominator (``ln(2) = 0.69315 <
    sqrt2 - 1/sqrt2 = 0.70711``) is larger in magnitude; the verifier's own
    reproduction (MUST_FIX item 3) found the pre-fix (raw-denominator) code
    undercounted the registered ln(2)-denominator value by exactly this
    ``(sqrt2-1/sqrt2)/ln(2) = 1.02014`` factor."""
    import numpy as np

    class _FlatCompleteness:
        def f_k(self, z: object, k: int, h: float) -> object:
            return np.ones_like(np.asarray(z, dtype=float))

    z_g = np.array([0.10])
    sigma_g = np.array([0.01])
    host_pixels = np.array([0], dtype=np.int64)
    h = 0.73
    n_grid = 801
    completeness = _FlatCompleteness()

    es_ln2 = driver._es_null_det_closed_form(  # type: ignore[attr-defined]
        z_g, sigma_g, host_pixels, completeness, h=h, n_grid=n_grid
    )

    # Independent re-derivation of the RAW-secant form (denominator
    # sqrt2 - 1/sqrt2), reusing the driver's own kernel helper so only the
    # denominator differs from the registered (fixed) function above.
    sqrt2 = math.sqrt(2.0)
    denom_raw = sqrt2 - 1.0 / sqrt2
    zg, sg = float(z_g[0]), float(sigma_g[0])
    pix = host_pixels[0:1]
    window_sigma = 4.0
    z_floor = 1e-6
    lo0 = max(zg - window_sigma * sg, z_floor)
    hi0 = zg + window_sigma * sg
    zz = np.linspace(lo0, hi0, n_grid)
    kernel = driver._es_null_det_kernel  # type: ignore[attr-defined]
    k0 = kernel(0.0, 1.0, zz, zg, sg, pix, completeness, h, n_grid)
    ln_plus = np.log(
        np.clip(kernel(0.0, sqrt2, zz, zg, sg, pix, completeness, h, n_grid), 1e-300, None)
    )
    ln_minus = np.log(
        np.clip(kernel(0.0, 1.0 / sqrt2, zz, zg, sg, pix, completeness, h, n_grid), 1e-300, None)
    )
    secs_raw = (ln_plus - ln_minus) / denom_raw
    window_minus = (zz >= max(zg - window_sigma * sg / sqrt2, z_floor)) & (
        zz <= zg + window_sigma * sg / sqrt2
    )
    weight = np.where(window_minus, k0, 0.0)
    weight_sum = np.trapezoid(weight, zz)
    es_raw = float(np.trapezoid(weight * secs_raw, zz) / weight_sum)

    assert es_ln2[0] == pytest.approx(es_raw * denom_raw / math.log(2.0), rel=1e-9)
    # Equivalently, in the verifier's own direction: the pre-fix raw form is
    # 1/1.02014...x SMALLER in magnitude than the registered ln(2) form.
    assert es_raw == pytest.approx(es_ln2[0] * math.log(2.0) / denom_raw, rel=1e-9)


def test_score_s_equals_score_lns_minus_es_null_det_by_construction(driver: object) -> None:
    """PA-HIER-32(d): ``score_s_i = score_lns_i - Es_null_det_i`` per event,
    then pooled -- so by linearity ``mean(score_s) == mean(score_lns) -
    mean(Es_null_det)`` exactly. Distinct from
    ``test_compute_scores_score_s_corrected_subtracts_es_null_det`` above
    (which hand-derives the per-event arithmetic): this test checks the
    pooled-statistic identity the verifier's MUST_FIX item asked for."""
    s_plus = pd.DataFrame(
        {
            "event_idx": [0, 1, 2],
            "ln_L_no_bh": [0.30, 0.55, 0.40],
            "ln_L_with_bh": [0.30, 0.55, 0.40],
            "es_null_det": [0.02, 0.03, 0.01],
        }
    )
    s_minus = pd.DataFrame(
        {
            "event_idx": [0, 1, 2],
            "ln_L_no_bh": [0.10, 0.20, 0.15],
            "ln_L_with_bh": [0.10, 0.20, 0.15],
        }
    )
    all_nodes = {
        "truth": [],
        "s_plus": [_node_result(driver, "s_plus", 0.0, math.sqrt(2.0), s_plus)],
        "s_minus": [_node_result(driver, "s_minus", 0.0, 1.0 / math.sqrt(2.0), s_minus)],
    }
    scores = driver.compute_scores(all_nodes, seeds=(1,))  # type: ignore[attr-defined]
    channel = scores["ln_L_no_bh"]

    assert channel["score_s_available"] is True
    mean_es_null_det = sum([0.02, 0.03, 0.01]) / 3
    assert channel["score_s"]["mean"] == pytest.approx(
        channel["score_lns"]["mean"] - mean_es_null_det
    )


# ===========================================================================
# PA-HIER-33 (ratified rows #278/#280 via the T1.4 Richardson falsifier,
# row #275; row #290 "b-pahier33-scorer" build task): the arm's-own-null
# Bartlett-identity scorer, `driver.compute_es_null_arm` /
# `compute_scores`'s `score_pahier33` key. Registration-form rule:
# PREREGISTRATION_HIER_HTHETA_20260826.md section 5 ("Rule"), reproduced
# verbatim in this module's docstring comment above `compute_es_null_arm`.
# ===========================================================================


def test_compute_es_null_arm_matches_hand_derivation(driver: object) -> None:
    """Hand-derive Es_null^{(arm)} = (Delta^2/6)*(-3<l'l''> - <l'^3>) for a
    tiny 3-event toy and compare against the driver's implementation
    (bootstrap disabled, n_bootstrap=0, for a pure point-estimate check)."""
    delta = math.log(2.0) / 2.0
    l0 = np.array([0.0, 0.1, -0.2])
    l_plus = np.array([0.30, 0.55, 0.10])
    l_minus = np.array([0.10, 0.20, -0.05])

    lprime = (l_plus - l_minus) / (2.0 * delta)
    ldbl = (l_plus - 2.0 * l0 + l_minus) / (delta**2)
    expected = (delta**2 / 6.0) * (-3.0 * np.mean(lprime * ldbl) - np.mean(lprime**3))

    es_null, bootstrap_sd = driver.compute_es_null_arm(  # type: ignore[attr-defined]
        l_plus, l0, l_minus, delta=delta, n_bootstrap=0
    )
    assert es_null == pytest.approx(expected)
    assert bootstrap_sd == 0.0


def test_compute_es_null_arm_default_delta_is_ln_sqrt2(driver: object) -> None:
    """Default delta = ln(sqrt(2)) -- PA-HIER-4's registered s-node grid
    (s = sqrt(2), 1/sqrt(2)), so 2*delta == ln(2) and l' is IDENTICALLY
    score_lns's own denominator."""
    assert driver._DELTA == pytest.approx(math.log(2.0) / 2.0)  # type: ignore[attr-defined]
    assert 2.0 * driver._DELTA == pytest.approx(math.log(2.0))  # type: ignore[attr-defined]


def test_compute_es_null_arm_bootstrap_sd_is_finite_and_nonnegative(driver: object) -> None:
    rng = np.random.default_rng(42)
    n = 50
    l0 = rng.normal(size=n)
    l_plus = l0 + rng.normal(scale=0.1, size=n) + 0.05
    l_minus = l0 - rng.normal(scale=0.1, size=n) - 0.05
    es_null, bootstrap_sd = driver.compute_es_null_arm(  # type: ignore[attr-defined]
        l_plus, l0, l_minus, n_bootstrap=200, bootstrap_seed=1
    )
    assert math.isfinite(es_null)
    assert math.isfinite(bootstrap_sd)
    assert bootstrap_sd >= 0.0


def test_compute_es_null_arm_requires_matching_shapes(driver: object) -> None:
    with pytest.raises(ValueError, match="must share one shape"):
        driver.compute_es_null_arm(  # type: ignore[attr-defined]
            np.array([0.1, 0.2]), np.array([0.0]), np.array([-0.1, -0.2])
        )


def test_compute_es_null_arm_returns_nan_below_two_finite_events(driver: object) -> None:
    es_null, bootstrap_sd = driver.compute_es_null_arm(  # type: ignore[attr-defined]
        np.array([0.1]), np.array([0.0]), np.array([-0.1]), n_bootstrap=10
    )
    assert math.isnan(es_null)
    assert math.isnan(bootstrap_sd)


def test_seed_clustered_sem_matches_hand_derivation(driver: object) -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    seeds = pd.Series([1, 1, 2, 2, 3, 3])
    sem, n_seeds = driver._seed_clustered_sem(values, seeds)  # type: ignore[attr-defined]
    per_seed_means = np.array([1.5, 3.5, 5.5])
    expected_sem = float(np.std(per_seed_means, ddof=1) / math.sqrt(3))
    assert n_seeds == 3
    assert sem == pytest.approx(expected_sem)


def test_seed_clustered_sem_nan_below_two_seeds(driver: object) -> None:
    sem, n_seeds = driver._seed_clustered_sem(  # type: ignore[attr-defined]
        pd.Series([1.0, 2.0]), pd.Series([1, 1])
    )
    assert n_seeds == 1
    assert math.isnan(sem)


def test_compute_scores_score_pahier33_requires_truth_and_s_axis(driver: object) -> None:
    """score_pahier33 is available only when BOTH the s-axis (s_plus/
    s_minus) AND the truth node are present -- neither has_s alone (as
    score_lns/score_s already require) nor has_truth alone is sufficient."""
    truth = pd.DataFrame(
        {"event_idx": [0, 1, 2], "ln_L_no_bh": [0.0, 0.1, -0.2], "ln_L_with_bh": [0.0, 0.1, -0.2]}
    )
    s_plus = pd.DataFrame(
        {
            "event_idx": [0, 1, 2],
            "ln_L_no_bh": [0.30, 0.55, 0.10],
            "ln_L_with_bh": [0.30, 0.55, 0.10],
        }
    )
    s_minus = pd.DataFrame(
        {
            "event_idx": [0, 1, 2],
            "ln_L_no_bh": [0.10, 0.20, -0.05],
            "ln_L_with_bh": [0.10, 0.20, -0.05],
        }
    )
    all_nodes = {
        "truth": [_node_result(driver, "truth", 0.0, 1.0, truth)],
        "s_plus": [_node_result(driver, "s_plus", 0.0, math.sqrt(2.0), s_plus)],
        "s_minus": [_node_result(driver, "s_minus", 0.0, 1.0 / math.sqrt(2.0), s_minus)],
    }
    scores = driver.compute_scores(all_nodes, seeds=(1,))  # type: ignore[attr-defined]
    channel = scores["ln_L_no_bh"]

    assert channel["score_pahier33_available"] is True
    assert channel["score_pahier33"]["n_pooled"] == 3

    delta = math.log(2.0) / 2.0
    l0 = np.array([0.0, 0.1, -0.2])
    l_plus = np.array([0.30, 0.55, 0.10])
    l_minus = np.array([0.10, 0.20, -0.05])
    es_null_expected, _ = driver.compute_es_null_arm(  # type: ignore[attr-defined]
        l_plus, l0, l_minus, delta=delta, n_bootstrap=0
    )
    mean_lns_expected = float(np.mean((l_plus - l_minus) / math.log(2.0)))
    assert channel["es_null_arm"]["value"] == pytest.approx(es_null_expected)
    assert channel["score_pahier33"]["mean"] == pytest.approx(mean_lns_expected - es_null_expected)


def test_compute_scores_score_pahier33_unavailable_without_truth(driver: object) -> None:
    """The s-axis alone (score_lns/score_s's own requirement) is NOT
    sufficient for score_pahier33 -- it additionally needs the truth node's
    l_i(0), which score_lns/score_s never require."""
    s_plus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.30, 0.40], "ln_L_with_bh": [0.30, 0.40]}
    )
    s_minus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.10, 0.20], "ln_L_with_bh": [0.10, 0.20]}
    )
    all_nodes = {
        "truth": [],
        "s_plus": [_node_result(driver, "s_plus", 0.0, math.sqrt(2.0), s_plus)],
        "s_minus": [_node_result(driver, "s_minus", 0.0, 1.0 / math.sqrt(2.0), s_minus)],
    }
    scores = driver.compute_scores(all_nodes, seeds=(1,))  # type: ignore[attr-defined]
    channel = scores["ln_L_no_bh"]
    assert channel["score_pahier33_available"] is False
    assert channel["score_pahier33"]["n_pooled"] == 0
    assert math.isnan(channel["score_pahier33"]["mean"])


def test_compute_scores_score_pahier33_unavailable_on_b_only_node_dict(driver: object) -> None:
    """runner-11's own 8-cell b-node pair shape (row #287): neither the
    s-axis nor truth is present -- score_pahier33 must degrade, never
    raise, exactly like score_b_available/score_s_available/gate_eng
    already do for this node-dict shape."""
    b_plus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [0.1, 0.2], "ln_L_with_bh": [0.1, 0.2]}
    )
    b_minus = pd.DataFrame(
        {"event_idx": [0, 1], "ln_L_no_bh": [-0.1, -0.2], "ln_L_with_bh": [-0.1, -0.2]}
    )
    all_nodes = {
        "b_plus": [_node_result(driver, "b_plus", 0.02, 1.0, b_plus)],
        "b_minus": [_node_result(driver, "b_minus", -0.02, 1.0, b_minus)],
    }
    scores = driver.compute_scores(all_nodes, seeds=(1,))  # type: ignore[attr-defined]
    channel = scores["ln_L_no_bh"]
    assert channel["score_pahier33_available"] is False
    assert channel["score_pahier33"]["n_pooled"] == 0


# ===========================================================================
# iiib (CoR-P production) venue path -- S0-B's precondition (row #290
# "b-pahier33-scorer" build task; PA-HIER-31 sec 1/(d)/(g)).
# ===========================================================================


def test_config_choices_includes_iiib(driver: object) -> None:
    assert driver.CONFIG_CHOICES == ("b0i", "ft", "iiib")  # type: ignore[attr-defined]


def test_build_iiib_venue_rejects_non_identity_sigma_z_scale(
    driver: object, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="sigma_z_scale must be 1.0"):
        driver.build_iiib_venue(tmp_path, 900101, sigma_z_scale=1.5)  # type: ignore[attr-defined]


def test_build_iiib_venue_stops_on_crb_pin_mismatch(
    driver: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver.c1d, "check_crb_pin", lambda: False)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="CRB CSV pin mismatch"):
        driver.build_iiib_venue(tmp_path, 900101)  # type: ignore[attr-defined]


def test_build_iiib_venue_stops_on_reduced_catalogue_pin_mismatch(
    driver: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver.c1d, "check_crb_pin", lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setattr(driver.c1d, "check_reduced_catalogue_pin", lambda: False)  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="reduced-catalogue pin mismatch"):
        driver.build_iiib_venue(tmp_path, 900101)  # type: ignore[attr-defined]


def test_build_iiib_venue_loads_the_real_pinned_inputs_when_pins_pass(
    driver: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No mirror realization is drawn -- events/handler come from the two
    pinned loaders, called with the pinned paths, verbatim (dataset
    pinning discipline: this test never touches the REAL multi-GB files,
    it substitutes a tiny fixture CSV and a sentinel handler and checks
    the wiring, not the real data)."""
    fake_crb_path = tmp_path / "fake_crb.csv"
    fake_events = pd.DataFrame({"event_idx": [0, 1], "some_col": [1.0, 2.0]})
    fake_events.to_csv(fake_crb_path, index=False)

    sentinel_handler = object()
    calls: list[str] = []

    def _fake_loader(catalogue_path: str) -> object:
        calls.append(catalogue_path)
        assert catalogue_path == driver.c1d.REDUCED_CATALOGUE_PATH  # type: ignore[attr-defined]
        return sentinel_handler

    monkeypatch.setattr(driver.c1d, "check_crb_pin", lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setattr(driver.c1d, "check_reduced_catalogue_pin", lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setattr(driver.c1d, "CRB_CSV_PATH", str(fake_crb_path))  # type: ignore[attr-defined]
    monkeypatch.setattr(driver.c1d, "_load_galaxy_catalog_handler", _fake_loader)  # type: ignore[attr-defined]

    events, handler = driver.build_iiib_venue(tmp_path, 900101)  # type: ignore[attr-defined]

    pd.testing.assert_frame_equal(events, fake_events)
    assert handler is sentinel_handler
    assert calls == [driver.c1d.REDUCED_CATALOGUE_PATH]  # type: ignore[attr-defined]


def test_build_venue_dispatches_iiib(
    driver: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = (pd.DataFrame({"event_idx": [0]}), object())
    called_with: dict[str, object] = {}

    def _fake_build_iiib_venue(work_root: Path, seed: int, sigma_z_scale: float = 1.0) -> object:
        called_with["work_root"] = work_root
        called_with["seed"] = seed
        called_with["sigma_z_scale"] = sigma_z_scale
        return sentinel

    monkeypatch.setattr(driver, "build_iiib_venue", _fake_build_iiib_venue)
    result = driver._build_venue("iiib", tmp_path, 900101, 1.0)  # type: ignore[attr-defined]
    assert result is sentinel
    assert called_with == {"work_root": tmp_path, "seed": 900101, "sigma_z_scale": 1.0}


def test_build_venue_rejects_unknown_config(driver: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config must be one of"):
        driver._build_venue("bogus", tmp_path, 900101, 1.0)  # type: ignore[attr-defined]
