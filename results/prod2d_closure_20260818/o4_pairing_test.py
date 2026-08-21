r"""Pre-check O4 (registered 2026-08-21, "PRE-CHECK O4 -- REGISTRATION" block,
below the C-SG prereg freeze line in ``PREREGISTRATION_SELFGEN_CONTROL.md``).

Question (A19): is the banked matched-channel score (S_bar_15 = -0.1173 +-
0.0194 realized) owned by the DOMAIN-AND-QUADRATURE PAIRING of the two
production implementations that must satisfy the same normalization identity
-- ``B_num`` (per-event, 50-node Gauss-Legendre over the h-dependent window
``[z(d_hat-4*sigma;h), min(z(d_hat+4*sigma;h), REDSHIFT_UPPER_LIMIT)]``) vs
``beta_Gbar_phi`` (1500-node trapezoid over ``[1e-6, min(z_max(h),
REDSHIFT_UPPER_LIMIT)]``, no interpolation at all since it evaluates directly
on its own quadrature grid) -- or does it survive alignment (a deeper math
defect in the estimator)?

**CORRECTED PREMISE (2026-08-21, post-launch adversarial finding, verified
against source before this fix -- see the BLOCKER item this docstring
resolves).** The prereg's Design table (and an earlier revision of this
docstring) described ``B_num`` as carrying an ``S_bar_phi`` factor via
endpoint-clamped ``np.interp``. That description is the 'fused'/'1d'
completion-numerator cell (``bayesian_statistics.py:4992-5007``,
``completion_numerator_integrand_sel_1d``), selected only when
``_sel_1d`` is True (``:4988-4990``: ``_sel_cell in ("1d", "fused")``). The
banked C-SG reference this pre-check falsifies did NOT run that cell:
production's own flags fix ``--selection_in_completion_numerator=off``
(``correspondence_1d.py:302-310``, ``PRODUCTION_FLAGS``); the mirror driver
defaults to that value (``correspondence_1d.py:1732-1733``,
``run_mirror_seed_inprocess``); and neither ``correspondence_1d.
ARM_SELECTION_CELL`` (``:452-462``, no ``csg*`` key) nor
``selfgen_control.run_csg_arm_seed`` (``:1437-1443``, calls
``run_mirror_seed_inprocess`` without overriding the parameter) ever
overrides it for any C-SG arm. Under ``_sel_cell == "off"``, ``_sel_1d`` is
False, so ``_completion_numerators`` (``bayesian_statistics.py:5157-5179``)
takes the ``else`` branch and calls the PLAIN ``completion_numerator_integrand``
(``:4904-4969``) -- no ``S_bar_phi`` factor at all. Meanwhile
``beta_Gbar_phi`` is built by ``precompute_phi_selection_integrals``
whenever ``normalization_mode == "absolute_marginal"``
(``bayesian_statistics.py:3800-3821``, unconditional on ``_sel_cell``) and
its integrand DOES carry ``S_bar_phi`` by construction (``:2065-2066``,
``s_phi`` factor). So the banked ``S_bar_15 = -0.1173`` is
``B_num(no S_bar_phi) / beta_Gbar_phi(with S_bar_phi)`` -- confirmed by the
log line at ``bayesian_statistics.py:3390-3395`` ("no survival factor in
either completion leg" under this cell). :func:`b_num_arm_a` and
:func:`b_num_arm_a1_window_to_full` / :func:`b_num_arm_a2_gl50_to_trapz1500`
/ :func:`b_num_arm_a3_clamp_to_zeroext` below no longer multiply the base
integrand by ``S_bar_phi`` for exactly this reason -- doing so would silently
test a 'fused'-semantics numerator against the 'off'-semantics banked
reference, an unregistered normalization-mode flip layered on top of the
domain/quadrature axis O4 exists to isolate. Arm A3 ("clamp to
zero-extension") is now DEGENERATE under 'off' semantics: there is no
``S_bar_phi`` lookup left to clamp or zero-extend, so A3 is mathematically
identical to the production-window/GL-50 baseline with no alignment applied
at all. It is kept (rather than silently dropped, per the registered
factorial design) and its output is flagged ``degenerate`` so the Run-phase
report can see this explicitly; A3's REPORTED-ONLY score carries no
independent information beyond what arm P already establishes.

**Arms** (see the prereg's Design table):

- **P (replica)** -- production code path, unmodified: regenerate each F
  seed's event set via ``draw_csg_realization`` (deterministic) and evaluate
  it with the REAL production leaf functions (``selfgen_control.
  run_csg_arm_seed`` -> ``correspondence_1d.run_mirror_seed_inprocess`` ->
  ``BayesianStatistics.evaluate()``) -- no reimplementation. GATE R4 compares
  the resulting per-event ``B_num(h)`` column bit-exactly against the banked
  diagnostics.
- **A (aligned, primary)** -- full common domain
  ``[1e-6, min(z_max(h), REDSHIFT_UPPER_LIMIT)]``, the SAME 1500-node
  trapezoid grid ``beta_Gbar_phi`` uses (evaluated directly ON that grid, so
  there is no interpolation/clamp decision to make at all -- the strongest
  form of "no clamp"). GATE T4 checks ``beta_Gbar_phi`` at production
  settings reproduces the column-derived ``D_tilde_phi - alpha_G_phi``.
- **A1-A3 (factorial, REPORTED-ONLY)** -- one alignment component at a time,
  the other two held at production settings:
    - A1 "window -> full-domain only": full domain, GL-50 quadrature,
      production (clamped) ``S_bar_phi`` lookup.
    - A2 "GL50 -> trapezoid1500 only": production per-event window,
      1500-node trapezoid (a fresh linspace over that window), production
      (clamped) ``S_bar_phi`` lookup.
    - A3 "clamp -> zero-extension only": production per-event window, GL-50
      quadrature, zero-extension ``S_bar_phi`` lookup (``np.interp(...,
      left=0.0, right=0.0)``).

**REGISTERED CORRECTION (verified against source, disclosed per the launch
task's "verify anything load-bearing yourself" instruction).** The prereg
text and the adversarial review both state the domain cap as literal
``1.55``. Verified: ``cosmological_model.max_redshift`` defaults to
``HOST_DRAW_Z_MAX`` (``constants.py:111`` = ``1.5``; ``cosmological_model.py:
199-201`` sets ``self.max_redshift = 1.5`` whenever no
``max_redshift_override`` is passed), and no call site in
``selfgen_control.py``/``correspondence_1d.py`` ever passes
``max_redshift_override`` (grep-verified). The pinned production venue that
built the CRB donor pool records ``"max_redshift": null``
(``results/prod2d_closure_20260818/postfix_baseline/iiib/run_metadata_0.json:
44``). So the value every real ``evaluate()`` call in this pipeline actually
uses is **1.5**, not 1.55. :data:`REDSHIFT_UPPER_LIMIT` below is set to
``HOST_DRAW_Z_MAX`` (1.5) accordingly -- using the prereg's stated 1.55
instead would inject an artificial ~0.05 domain mismatch that has nothing to
do with the domain-and-quadrature axis under test, contaminating arm A.

**HARD CONSTRAINTS (launch task):**

1. Never end a turn to wait on an untracked process (Bash calls in this
   harness are synchronous/blocking; no background-and-forget).
2. Every load-bearing claim in the docstrings/comments below cites file:line.
3. The O4 statistic (``S_bar(O4-A)`` and the A1-A3 REPORTED-ONLY scores) is
   only computed in the "Run phase" (:func:`main`, AFTER both gates pass for
   every requested seed) -- ``--gates-only`` stops before any aligned number
   is computed at all, for exactly this reason.

Usage:
    # Gate-only plumbing proof (no O4 statistic computed):
    uv run python results/prod2d_closure_20260818/o4_pairing_test.py \
        --seeds 910101 --gates-only

    # Full registered run (all 15 F seeds, computes S_bar(O4-A) and bands it):
    uv run python results/prod2d_closure_20260818/o4_pairing_test.py
"""

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import fixed_quad
from scipy.stats import norm

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    precompute_phi_marginal_survival,
    precompute_phi_selection_integrals,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import HOST_DRAW_Z_MAX
from darksiren_emri.datamodels.detection import Detection
from darksiren_emri.physical_relations import (
    comoving_volume_element,
    dist_to_redshift,
    dist_vectorized,
)
from darksiren_emri.validation import selfgen_control as csg
from darksiren_emri.validation.correspondence_1d import CRB_CSV_PATH
from darksiren_emri.validation.selfgen_control import (
    CsgCompletenessModel,
    CsgDetectionProbabilityModel,
    build_csg_selection_objects,
    draw_csg_realization,
    score_at_h_gen,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results/prod2d_closure_20260818"
BANKED_DIAG_DIR = RESULTS_DIR / "csg_pilot_20260821" / "diagnostics"

ARM: str = "csgf"
H_GEN: float = 0.73
H_LO: float = 0.725
H_HI: float = 0.735
F_SEEDS: tuple[int, ...] = tuple(range(910101, 910101 + 15))

# See the module docstring's "REGISTERED CORRECTION" note.
REDSHIFT_UPPER_LIMIT: float = HOST_DRAW_Z_MAX

# GATE R4 (prereg PRE-CHECK O4, GATE R4): bit-exact, or <= 1e-12 relative if
# the call cannot be made bit-identical (production's per-event dispatch goes
# through a multiprocessing pool -- _starmap_host_batches,
# bayesian_statistics.py:4636-4662 -- whose float-summation order over
# candidate hosts is not guaranteed identical run-to-run; the registered
# fallback tolerance absorbs that, nothing else).
GATE_R4_RTOL: float = 1.0e-12
# GATE T4 (prereg PRE-CHECK O4, GATE T4): 2e-6 relative, the same tolerance
# GATE T (decompose_matched_channel.py:87 GATE_T_TOL, selfgen_control.py:758
# gate_t_h_only's default) uses for alpha_G_phi/D_tilde_phi h-onlyness.
GATE_T4_TOL: float = 2.0e-6

# Bands (prereg PRE-CHECK O4, "Bands" table).
BAND_PAIRING_OWNS: float = 0.0373
S_REGISTERED: float = -0.1173
SEM_REGISTERED: float = 0.0194
BAND_DEFECT_HARDENED: float = 3.0 * SEM_REGISTERED

# Production constants reproduced here because B_num's integrand is a nested
# closure inside BayesianStatistics.p_Di, not a module-level export
# (correspondence_1d.py:186-198's own disclosed scope limitation; recon
# confirmed no leaf B_num callable exists). integration_limit_sigma_multiplier
# and FIXED_QUAD_N=50 are bayesian_statistics.py:4866-4867.
INTEGRATION_LIMIT_SIGMA_MULTIPLIER: float = 4.0
PRODUCTION_QUAD_N: int = 50
ALIGNED_TRAPZ_N: int = 1500  # matches _S_PHI_Z_GRID_POINTS (bayesian_statistics.py:1754,1979)

FactorialArm = str  # "A" | "A1_window_to_full" | "A2_gl50_to_trapz1500" | "A3_clamp_to_zeroext"


@dataclasses.dataclass(frozen=True)
class EventGeometry:
    """Per-event quantities the completion-numerator integrand needs, extracted
    once per event and reused across every h-node and every arm.

    Mirrors the per-event closure state ``p_Di`` builds at
    ``bayesian_statistics.py:4868-4902`` (``_comp_det_d_L``,
    ``_comp_sigma_dLfrac``, ``_event_pixel``) plus the window inputs
    ``_completion_numerators`` reads directly off ``self.detection``
    (``:5130-5136``).
    """

    event_idx: int
    d_hat: float
    d_L_uncertainty: float
    theta: float
    sigma_dLfrac: float
    pixel: int


def _sigma_dLfrac(det: Detection) -> float:
    r"""Reproduce ``p_Di``'s marginal sigma on ``d_L_fraction`` (``bayesian_statistics.py:
    3960-3979`` builds ``cov_3d``; ``:4032`` sets ``cov_inv_3d = pinv(cov_3d)``;
    ``:4880-4882`` sets ``_comp_sigma_dLfrac = sqrt(inv(cov_inv_3d)[2, 2])``).

    For an invertible 3x3 matrix ``pinv(A) == inv(A)`` and ``inv(inv(A)) == A``
    to machine precision, so this reconstructs ``cov_3d[2, 2]`` (i.e.
    ``(d_L_uncertainty / d_L)**2``) up to a ~1e-14 relative round-trip error --
    far below every O4 tolerance (2e-6, 0.0373). The full 3x3 build (rather
    than the ``d_L_uncertainty / d_L`` shortcut) is kept here anyway so this
    function is a literal, auditable copy of the production recipe rather than
    an argued-equivalent simplification.
    """
    cov_3d = np.array(
        [
            [
                det.phi_error**2,
                det.theta_phi_covariance,
                det.d_L_phi_covariance / det.d_L,
            ],
            [
                det.theta_phi_covariance,
                det.theta_error**2,
                det.d_L_theta_covariance / det.d_L,
            ],
            [
                det.d_L_phi_covariance / det.d_L,
                det.d_L_theta_covariance / det.d_L,
                det.d_L_uncertainty**2 / det.d_L**2,
            ],
        ]
    )
    cov_inv_3d = np.linalg.pinv(cov_3d)
    cov_3d_recon = np.linalg.inv(cov_inv_3d)
    return float(np.sqrt(cov_3d_recon[2, 2]))


def event_geometries(rows: pd.DataFrame, completeness: CsgCompletenessModel) -> list[EventGeometry]:
    """Extract one :class:`EventGeometry` per row of a C-SG realization.

    Row order is assumed to equal the diagnostics CSV's ``event_idx``
    (``bayesian_statistics.py:5366``: ``"event_idx": detection_index``, with
    ``detection_index`` the 0-based position production's ``evaluate()``
    reads off the freshly-written, freshly-re-read CRB CSV -- the same order
    ``rows`` is in after ``draw_csg_realization``'s
    ``.reset_index(drop=True)``, ``selfgen_control.py:611``). GATE R4
    empirically proves this alignment: a misalignment would show up as a
    GATE R4 failure, not silently.
    """
    out: list[EventGeometry] = []
    for i, (_, row) in enumerate(rows.iterrows()):
        det = Detection(row)
        pixel = int(completeness.ang2pix(det.phi, det.theta))
        out.append(
            EventGeometry(
                event_idx=i,
                d_hat=det.d_L,
                d_L_uncertainty=det.d_L_uncertainty,
                theta=det.theta,
                sigma_dLfrac=_sigma_dLfrac(det),
                pixel=pixel,
            )
        )
    return out


def _base_integrand(
    z: npt.NDArray[np.float64],
    h_eval: float,
    geo: EventGeometry,
    completeness: CsgCompletenessModel,
) -> npt.NDArray[np.float64]:
    """``(1 - f_k) * p_gw * dVc / (1 + z)`` -- literal copy of
    ``completion_numerator_integrand`` (``bayesian_statistics.py:4904-4969``),
    the 'ratio' event measure (production default; C-SG never overrides
    ``completion_event_measure``). NOT the ``S_bar_phi``-multiplied form --
    callers apply the arm's own S_bar_phi lookup on top.
    """
    d_l = np.asarray(dist_vectorized(z, h=h_eval), dtype=np.float64)
    d_l_fraction = d_l / geo.d_hat
    p_gw = (
        norm.pdf(d_l_fraction, loc=1.0, scale=geo.sigma_dLfrac)
        * math.sin(geo.theta)
        / (4.0 * math.pi)
    )
    d_vc = np.atleast_1d(np.asarray(comoving_volume_element(z, h=h_eval), dtype=np.float64))
    f_z = np.clip(np.asarray(completeness.f_k(z, geo.pixel, h_eval), dtype=np.float64), 0.0, 1.0)
    return np.asarray((1.0 - f_z) * p_gw * d_vc / (1.0 + z), dtype=np.float64)


# NOTE: no S_bar_phi lookup helpers here. An earlier revision defined
# `_s_bar_clamped`/`_s_bar_zero_extended` and applied them inside every arm's
# B_num integrand (module docstring "CORRECTED PREMISE" explains why that was
# wrong): the banked reference's real B_num, computed under production's
# `selection_in_completion_numerator="off"`, never queries S_bar_phi at all
# (bayesian_statistics.py:5157-5179's `else` branch calls the plain
# `completion_numerator_integrand`, `:4904-4969`). Removed rather than left
# dead so this file cannot silently regain the confound on a future edit.


def _production_window(
    geo: EventGeometry, h_eval: float, redshift_upper_limit: float
) -> tuple[float, float]:
    """Literal copy of ``_completion_numerators``'s window
    (``bayesian_statistics.py:5129-5148``): ``[z(d_hat - 4*sigma_dL; h),
    min(z(d_hat + 4*sigma_dL; h), redshift_upper_limit)]``, floored at 1e-6.
    """
    z_upper = dist_to_redshift(
        geo.d_hat + INTEGRATION_LIMIT_SIGMA_MULTIPLIER * geo.d_L_uncertainty, h=h_eval
    )
    z_lower = dist_to_redshift(
        geo.d_hat - INTEGRATION_LIMIT_SIGMA_MULTIPLIER * geo.d_L_uncertainty, h=h_eval
    )
    z_lower = max(z_lower, 1.0e-6)
    z_upper = min(z_upper, redshift_upper_limit)
    return z_lower, z_upper


def b_num_arm_a(
    geo: EventGeometry,
    h_eval: float,
    completeness: CsgCompletenessModel,
    z_grid: npt.NDArray[np.float64],
    s_grid: npt.NDArray[np.float64],
) -> float:
    """Arm A (aligned, primary): full domain ``[1e-6, min(z_max(h),
    REDSHIFT_UPPER_LIMIT)]`` = ``[z_grid[0], z_grid[-1]]`` by construction of
    ``z_grid`` (``precompute_phi_marginal_survival``,
    ``bayesian_statistics.py:1975-1979``), 1500-node trapezoid ON that SAME
    grid -- the domain/quadrature axis ``beta_Gbar_phi`` uses.

    NO ``S_bar_phi`` factor (module-docstring "CORRECTED PREMISE"): the
    banked reference's real ``B_num`` was computed under
    ``selection_in_completion_numerator="off"``
    (``bayesian_statistics.py:5171-5179``, the plain
    ``completion_numerator_integrand``, no survival factor); multiplying by
    ``s_grid`` here would silently test 'fused' semantics against an 'off'
    reference. ``s_grid`` is accepted for signature symmetry with the other
    arms / :func:`build_aligned_tables` but is intentionally unused.
    """
    del s_grid  # not applied -- see module docstring "CORRECTED PREMISE"
    integrand = _base_integrand(z_grid, h_eval, geo, completeness)
    return float(np.trapezoid(integrand, z_grid))


def b_num_arm_a1_window_to_full(
    geo: EventGeometry,
    h_eval: float,
    completeness: CsgCompletenessModel,
    z_grid: npt.NDArray[np.float64],
    s_grid: npt.NDArray[np.float64],
) -> float:
    """A1 (factorial): domain -> full (``[z_grid[0], z_grid[-1]]``); quadrature
    stays at production settings (GL-50). NO ``S_bar_phi`` factor (module
    docstring "CORRECTED PREMISE" -- the production 'off' numerator this arm
    isolates the domain change from never queries ``S_bar_phi``, so there is
    no clamp to exercise or avoid). ``s_grid`` is accepted for signature
    symmetry and intentionally unused.
    """
    del s_grid  # not applied -- see module docstring "CORRECTED PREMISE"

    def integrand(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _base_integrand(z, h_eval, geo, completeness)

    return float(fixed_quad(integrand, float(z_grid[0]), float(z_grid[-1]), n=PRODUCTION_QUAD_N)[0])


def b_num_arm_a2_gl50_to_trapz1500(
    geo: EventGeometry,
    h_eval: float,
    completeness: CsgCompletenessModel,
    z_grid: npt.NDArray[np.float64],
    s_grid: npt.NDArray[np.float64],
    redshift_upper_limit: float,
) -> float:
    """A2 (factorial): quadrature -> 1500-node trapezoid (on a FRESH linspace
    over production's own per-event window); domain stays at production
    settings (per-event window). NO ``S_bar_phi`` factor (module docstring
    "CORRECTED PREMISE" -- the production 'off' numerator this arm isolates
    the quadrature change from never queries ``S_bar_phi``, so there is no
    clamp to exercise). ``z_grid``/``s_grid`` are accepted for signature
    symmetry and intentionally unused.
    """
    del z_grid, s_grid  # not applied -- see module docstring "CORRECTED PREMISE"
    z_lo, z_hi = _production_window(geo, h_eval, redshift_upper_limit)
    if z_lo >= z_hi:
        return 0.0
    z_local = np.linspace(z_lo, z_hi, ALIGNED_TRAPZ_N)
    base = _base_integrand(z_local, h_eval, geo, completeness)
    return float(np.trapezoid(base, z_local))


def b_num_arm_a3_clamp_to_zeroext(
    geo: EventGeometry,
    h_eval: float,
    completeness: CsgCompletenessModel,
    z_grid: npt.NDArray[np.float64],
    s_grid: npt.NDArray[np.float64],
    redshift_upper_limit: float,
) -> float:
    """A3 (factorial): clamp -> zero-extension; domain and quadrature stay at
    production settings (per-event window, GL-50).

    DEGENERATE under the corrected premise (module docstring "CORRECTED
    PREMISE"): the production 'off' numerator never queries ``S_bar_phi``,
    so there is no clamp/zero-extension behaviour left to swap -- this arm
    is mathematically identical to the production-window/GL-50 baseline with
    no alignment applied (i.e. to arm P's own integrand). Kept (not silently
    dropped) per the registered factorial design; callers should treat its
    REPORTED-ONLY score as carrying no independent information. ``z_grid``/
    ``s_grid`` are accepted for signature symmetry and intentionally unused.
    """
    del z_grid, s_grid  # not applied -- see module docstring "CORRECTED PREMISE"
    z_lo, z_hi = _production_window(geo, h_eval, redshift_upper_limit)
    if z_lo >= z_hi:
        return 0.0

    def integrand(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return _base_integrand(z, h_eval, geo, completeness)

    return float(fixed_quad(integrand, z_lo, z_hi, n=PRODUCTION_QUAD_N)[0])


def build_aligned_tables(
    completeness: CsgCompletenessModel,
    detection_probability: CsgDetectionProbabilityModel,
) -> tuple[
    dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    dict[float, float],
]:
    """``S_bar_phi`` table and ``beta_Gbar_phi(h)`` at ``h in {H_LO, H_HI}``, via
    the REAL production module-level functions
    :func:`precompute_phi_marginal_survival` / :func:`precompute_phi_selection_integrals`
    (no reimplementation) at ``z_max_cap=REDSHIFT_UPPER_LIMIT``. Identical for
    every seed of a given arm (depends only on ``h`` and the shared
    completeness/detection-probability objects), so callers compute this ONCE
    per script invocation, not per seed.
    """
    # build_csg_selection_objects (selfgen_control.py:394-449) returns the real
    # concrete SimulationDetectionProbability behind the narrower structural
    # CsgDetectionProbabilityModel Protocol (selfgen_control.py:241-249's own
    # docstring: "the one production instance satisfies it directly"); cast
    # back to the concrete type precompute_phi_marginal_survival's signature
    # requires -- not a reimplementation, just a type-narrowing of the same
    # object.
    detection_probability_concrete = cast(SimulationDetectionProbability, detection_probability)
    phi_table = precompute_phi_marginal_survival(
        [H_LO, H_HI], detection_probability_concrete, z_max_cap=REDSHIFT_UPPER_LIMIT
    )
    _beta_g_phi, beta_gbar_phi = precompute_phi_selection_integrals(
        [H_LO, H_HI], phi_table, completeness
    )
    return phi_table, beta_gbar_phi


def run_gate_r4(
    work_root: Path, out_dir: Path, seed: int, arm: str, n_events: int
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Arm P (replica): regenerate one seed via the REAL, unmodified production
    generator + evaluate() (``selfgen_control.run_csg_arm_seed`` -- the SAME
    function that produced ``csg_pilot_20260821``, called verbatim, no
    reimplementation) and bit-compare its ``B_num`` diagnostics column
    against the banked reference.

    Returns:
        ``(gate_result, fresh_diagnostics_df, run_csg_arm_seed_record)``.
    """
    t0 = time.time()
    record_path = csg.run_csg_arm_seed(work_root, arm, seed, out_dir, n_events=n_events)
    elapsed = time.time() - t0
    record = json.loads(record_path.read_text())
    fresh_csv = Path(str(record["diagnostics_csv"]))
    fresh = pd.read_csv(fresh_csv)

    banked_csv = BANKED_DIAG_DIR / f"{arm}_seed{seed}" / "event_likelihoods.csv"
    if not banked_csv.is_file():
        return (
            {
                "seed": seed,
                "pass": False,
                "reason": f"banked diagnostics not found: {banked_csv}",
                "elapsed_s": elapsed,
            },
            fresh,
            record,
        )
    banked = pd.read_csv(banked_csv)
    merged = fresh.merge(
        banked[["event_idx", "h", "B_num"]],
        on=["event_idx", "h"],
        suffixes=("_fresh", "_banked"),
        how="outer",
        indicator=True,
    )
    key_mismatch = bool((merged["_merge"] != "both").any())
    b_fresh = merged["B_num_fresh"].to_numpy(dtype=np.float64)
    b_banked = merged["B_num_banked"].to_numpy(dtype=np.float64)
    exact = bool(np.array_equal(b_fresh, b_banked))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(b_fresh - b_banked) / np.maximum(np.abs(b_banked), np.finfo(float).tiny)
    max_rel = float(np.nanmax(rel)) if rel.size else float("nan")
    ok = (not key_mismatch) and (exact or max_rel <= GATE_R4_RTOL)
    return (
        {
            "seed": seed,
            "pass": ok,
            "key_mismatch": key_mismatch,
            "bit_exact": exact,
            "max_rel_err": max_rel,
            "tol": GATE_R4_RTOL,
            "n_rows_compared": int(len(merged)),
            "fresh_csv": str(fresh_csv),
            "banked_csv": str(banked_csv),
            "fallback_justification": (
                None
                if exact
                else (
                    "not bit-exact (max_rel_err="
                    f"{max_rel:.3e}); production evaluate() dispatches per-host "
                    "likelihood terms through a multiprocessing pool "
                    "(_starmap_host_batches, bayesian_statistics.py:4636-4662) "
                    "whose float-summation order is not guaranteed run-to-run "
                    "identical, so the registered <=1e-12 relative fallback is "
                    "used per the O4 GATE R4 spec."
                )
            ),
            "elapsed_s": elapsed,
            "reference": (
                "PREREGISTRATION_SELFGEN_CONTROL.md PRE-CHECK O4, GATE R4; "
                "regenerated via selfgen_control.run_csg_arm_seed (unmodified)"
            ),
        },
        fresh,
        record,
    )


def run_gate_t4(fresh_diag: pd.DataFrame, beta_gbar_phi: dict[float, float]) -> dict[str, Any]:
    """GATE T4: ``beta_Gbar_phi(h)`` at production settings reproduces the
    column-derived ``D_tilde_phi(h) - alpha_G_phi(h)`` to 2e-6 relative, for
    ``h in {H_LO, H_HI}``. ``alpha_G_phi``/``D_tilde_phi`` are h-only (GATE T,
    ``bayesian_statistics.py:5120-5192`` closes over ``self.h`` only; verified
    to <= 2e-6 by ``decompose_matched_channel.py``'s own GATE T on B-SEL), so
    any single event row suffices -- ``event_idx == 0`` is used.
    """
    rows_by_h = fresh_diag[fresh_diag["event_idx"] == 0].set_index("h")
    per_h: dict[str, Any] = {}
    ok_all = True
    for h in (H_LO, H_HI):
        if h not in rows_by_h.index:
            per_h[str(h)] = {"pass": False, "reason": f"h={h} not present in diagnostics CSV"}
            ok_all = False
            continue
        col_val = float(rows_by_h.loc[h, "D_tilde_phi"] - rows_by_h.loc[h, "alpha_G_phi"])
        mine = beta_gbar_phi[h]
        rel = abs(mine - col_val) / max(abs(col_val), float(np.finfo(float).tiny))
        ok = rel <= GATE_T4_TOL
        ok_all = ok_all and ok
        per_h[str(h)] = {
            "beta_gbar_phi_mine": mine,
            "column_derived_D_tilde_minus_alpha": col_val,
            "rel_err": rel,
            "tol": GATE_T4_TOL,
            "pass": ok,
        }
    return {
        "pass": ok_all,
        "per_h": per_h,
        "reference": "PREREGISTRATION_SELFGEN_CONTROL.md PRE-CHECK O4, GATE T4",
    }


def compute_factorial_scores(
    rows: pd.DataFrame,
    completeness: CsgCompletenessModel,
    phi_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    beta_gbar_phi: dict[float, float],
    seed: int,
) -> dict[FactorialArm, dict[str, Any]]:
    """Per-seed matched score under arm A and the A1-A3 factorial cells, via
    the pre-committed scorer :func:`selfgen_control.score_at_h_gen` (central
    difference at ``H_LO``/``H_HI`` bracketing ``H_GEN``, same combine
    convention as every other C-SG channel score).
    """
    geos = event_geometries(rows, completeness)
    z_lo_grid, s_lo_grid = phi_table[H_LO]
    z_hi_grid, s_hi_grid = phi_table[H_HI]

    def _matched(b_num_fn_lo: Any, b_num_fn_hi: Any) -> npt.NDArray[np.float64]:
        b_lo = np.array([b_num_fn_lo(geo) for geo in geos], dtype=np.float64)
        b_hi = np.array([b_num_fn_hi(geo) for geo in geos], dtype=np.float64)
        matched_lo = b_lo / beta_gbar_phi[H_LO]
        matched_hi = b_hi / beta_gbar_phi[H_HI]
        return np.column_stack([matched_lo, np.full_like(matched_lo, np.nan), matched_hi])

    variants: dict[FactorialArm, npt.NDArray[np.float64]] = {
        "A": _matched(
            lambda g: b_num_arm_a(g, H_LO, completeness, z_lo_grid, s_lo_grid),
            lambda g: b_num_arm_a(g, H_HI, completeness, z_hi_grid, s_hi_grid),
        ),
        "A1_window_to_full": _matched(
            lambda g: b_num_arm_a1_window_to_full(g, H_LO, completeness, z_lo_grid, s_lo_grid),
            lambda g: b_num_arm_a1_window_to_full(g, H_HI, completeness, z_hi_grid, s_hi_grid),
        ),
        "A2_gl50_to_trapz1500": _matched(
            lambda g: b_num_arm_a2_gl50_to_trapz1500(
                g, H_LO, completeness, z_lo_grid, s_lo_grid, REDSHIFT_UPPER_LIMIT
            ),
            lambda g: b_num_arm_a2_gl50_to_trapz1500(
                g, H_HI, completeness, z_hi_grid, s_hi_grid, REDSHIFT_UPPER_LIMIT
            ),
        ),
        "A3_clamp_to_zeroext": _matched(
            lambda g: b_num_arm_a3_clamp_to_zeroext(
                g, H_LO, completeness, z_lo_grid, s_lo_grid, REDSHIFT_UPPER_LIMIT
            ),
            lambda g: b_num_arm_a3_clamp_to_zeroext(
                g, H_HI, completeness, z_hi_grid, s_hi_grid, REDSHIFT_UPPER_LIMIT
            ),
        ),
    }
    out: dict[FactorialArm, dict[str, Any]] = {}
    for name, vals in variants.items():
        s = score_at_h_gen(vals, H_GEN, (H_LO, H_GEN, H_HI))
        s["seed"] = seed
        out[name] = s
    return out


def apply_bands(s_bar: float) -> tuple[str, float | None]:
    """The registered O4 bands (PRE-CHECK O4, "Bands" table), applied ONLY to
    arm A's fleet statistic.
    """
    if abs(s_bar) <= BAND_PAIRING_OWNS:
        return "PAIRING-OWNS-IT", None
    if abs(s_bar - S_REGISTERED) <= BAND_DEFECT_HARDENED:
        return "DEFECT-HARDENED", None
    owned_fraction = 1.0 - s_bar / S_REGISTERED
    return "PAIRING-PARTIAL", owned_fraction


def _write_output(path: str, obj: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in F_SEEDS),
        help="Comma-separated seeds (default: all 15 registered F seeds).",
    )
    ap.add_argument("--arm", type=str, default=ARM)
    ap.add_argument(
        "--work-root",
        type=str,
        default=str(RESULTS_DIR / "o4_pairing_test_work"),
        help="Scratch directory for the Arm-P regeneration (run_csg_arm_seed).",
    )
    ap.add_argument("--out", type=str, default=str(RESULTS_DIR / "o4_pairing_test_output.json"))
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument(
        "--gates-only",
        action="store_true",
        help=(
            "Run GATE R4 + GATE T4 for the requested seed(s) and stop -- never "
            "compute arm A / A1-A3 or the O4 statistic. Use for plumbing proof "
            "runs (per the launch task, only ONE seed may be proved this way "
            "before the author authorizes the full fleet Run phase)."
        ),
    )
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arm = args.arm
    if arm not in csg.CSG_H_GEN:
        print(f"unknown arm {arm!r}; registered arms: {sorted(csg.CSG_H_GEN)}", file=sys.stderr)
        return 2
    h_gen = csg.CSG_H_GEN[arm]
    if arm != ARM or h_gen != H_GEN:
        print(
            f"NOTE: O4 is registered for arm={ARM!r} (h_gen={H_GEN}) only; "
            f"running arm={arm!r} (h_gen={h_gen}) is OUT OF REGISTERED SCOPE "
            "-- gates and scores will still run, but this is not the "
            "registered measurement.",
            file=sys.stderr,
        )

    work_root = Path(args.work_root)
    out_dir = work_root / "run_csg_arm_seed_out"
    work_root.mkdir(parents=True, exist_ok=True)

    completeness, detection_probability = build_csg_selection_objects(h_gen=h_gen)
    donor_rows = pd.read_csv(CRB_CSV_PATH)
    phi_table, beta_gbar_phi = build_aligned_tables(completeness, detection_probability)

    gate_r4_rows: list[dict[str, Any]] = []
    gate_t4_rows: list[dict[str, Any]] = []
    rows_by_seed: dict[int, pd.DataFrame] = {}

    for seed in seeds:
        print(f"=== seed {seed}: Arm P (production replica, run_csg_arm_seed) ===", flush=True)
        r4, fresh, _record = run_gate_r4(work_root, out_dir, seed, arm, args.n_events)
        gate_r4_rows.append(r4)
        print(json.dumps(r4, indent=2))
        if not r4["pass"]:
            print(
                f"GATE R4 FAILED for seed {seed} -- STOP before any aligned number is computed.",
                file=sys.stderr,
            )
            _write_output(
                args.out,
                {"gate_r4": gate_r4_rows, "gate_t4": gate_t4_rows, "status": "GATE_R4_FAILED"},
            )
            return 1

        t4 = run_gate_t4(fresh, beta_gbar_phi)
        t4["seed"] = seed
        gate_t4_rows.append(t4)
        print(json.dumps(t4, indent=2))
        if not t4["pass"]:
            print(
                f"GATE T4 FAILED for seed {seed} -- STOP before any aligned number is computed.",
                file=sys.stderr,
            )
            _write_output(
                args.out,
                {"gate_r4": gate_r4_rows, "gate_t4": gate_t4_rows, "status": "GATE_T4_FAILED"},
            )
            return 1

        if not args.gates_only:
            # Deterministic re-draw of the SAME realization (pure function of
            # (seed, arm) given the same completeness/detection_probability/
            # donor_rows -- selfgen_control.py:488-492) to recover the
            # per-event geometry (d_hat, sigma_dL, sky) GATE R4 just proved
            # matches the banked B_num bit-exactly.
            rows, _draw_diag = draw_csg_realization(
                seed, arm, args.n_events, completeness, detection_probability, donor_rows
            )
            rows_by_seed[seed] = rows

    if args.gates_only:
        print(
            "--gates-only: stopping after GATE R4 + GATE T4 for "
            f"{len(seeds)} seed(s). Arm A / A1-A3 / the O4 statistic were NOT "
            "computed (hard constraint 3)."
        )
        _write_output(
            args.out,
            {
                "gate_r4": gate_r4_rows,
                "gate_t4": gate_t4_rows,
                "status": "GATES_ONLY_PASS",
                "note": "arm A / A1-A3 / the O4 statistic were not run (--gates-only)",
            },
        )
        return 0

    # ---- Run phase: GATE R4 + GATE T4 passed for every requested seed.
    # Only now may the O4 statistic be computed (hard constraint 3). ----
    per_seed_scores: dict[FactorialArm, list[dict[str, Any]]] = {
        "A": [],
        "A1_window_to_full": [],
        "A2_gl50_to_trapz1500": [],
        "A3_clamp_to_zeroext": [],
    }
    for seed in seeds:
        scored = compute_factorial_scores(
            rows_by_seed[seed], completeness, phi_table, beta_gbar_phi, seed
        )
        for name, s in scored.items():
            per_seed_scores[name].append(s)

    fleet: dict[str, Any] = {}
    for name, seed_list in per_seed_scores.items():
        means = np.array(
            [s["mean_score"] for s in seed_list if s["mean_score"] is not None],
            dtype=np.float64,
        )
        n = int(means.size)
        s_bar = float(means.mean()) if n else None
        sem = float(means.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        band, owned_fraction = (
            apply_bands(s_bar) if (name == "A" and s_bar is not None) else (None, None)
        )
        fleet[name] = {
            "n_seeds": n,
            "S_bar": s_bar,
            "sem_seeds": sem,
            "per_seed": seed_list,
            "band_fired": band,
            "owned_fraction": owned_fraction,
            "reference": (
                "PREREGISTRATION_SELFGEN_CONTROL.md PRE-CHECK O4, Statistic + Bands "
                "(bands applied only to arm A)"
            ),
        }

    output = {
        "registered_in": (
            "results/prod2d_closure_20260818/PREREGISTRATION_SELFGEN_CONTROL.md, "
            "PRE-CHECK O4 -- REGISTRATION (2026-08-21, ledger row #153 item 4)"
        ),
        "arm": arm,
        "h_gen": h_gen,
        "h_lo": H_LO,
        "h_hi": H_HI,
        "n_seeds_requested": len(seeds),
        "redshift_upper_limit_used": REDSHIFT_UPPER_LIMIT,
        "redshift_upper_limit_note": (
            "The prereg/adversarial-review text states this cap as literal 1.55; "
            "verified against source (constants.py:111 HOST_DRAW_Z_MAX=1.5, "
            "cosmological_model.py:199-201 default max_redshift=HOST_DRAW_Z_MAX, "
            "postfix_baseline/iiib/run_metadata_0.json:44 'max_redshift': null, no "
            "max_redshift_override call site in selfgen_control.py/"
            "correspondence_1d.py) -- every real evaluate() call in this pipeline "
            "actually uses 1.5. Used here as 1.5 (a disclosed spec deviation)."
        ),
        "gate_r4": gate_r4_rows,
        "gate_t4": gate_t4_rows,
        "fleet": fleet,
    }
    _write_output(args.out, output)
    print(
        json.dumps(
            {
                "status": "OK",
                "out": args.out,
                **{name: v["S_bar"] for name, v in fleet.items()},
                "band_fired_A": fleet["A"]["band_fired"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
