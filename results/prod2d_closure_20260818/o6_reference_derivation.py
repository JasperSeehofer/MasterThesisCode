r"""Registered reference instrument for CONFIRMATION RUN O6
(``PREREGISTRATION_SELFGEN_CONTROL.md``, "CONFIRMATION RUN O6 -- REGISTRATION",
2026-08-21, ledger row #157 item 2; A18 "reference" note).

Computes, for seed 910101 ONLY, from BANKED inputs and a deterministic
(zero-``evaluate()``) realization redraw:

- ``r_prod(910101)`` -- the PRIMARY registered reference: a harness replica of
  the FUSED-cell completion numerator at PRODUCTION settings (per-event
  window, GL-50 quadrature, ``S_bar_phi`` applied via endpoint-clamped
  ``np.interp`` -- "a literal convention copy of
  ``completion_numerator_integrand_sel_1d``",
  ``bayesian_statistics.py:4992-5007``), divided by the SAME
  ``beta_Gbar_phi`` every C-SG channel uses, scored by the committed
  :func:`~darksiren_emri.validation.selfgen_control.score_at_h_gen`.
- ``r_A(910101)`` -- REPORTED-ONLY companion: O4's aligned arm A (full-domain
  1500-node trapezoid on ``beta_Gbar_phi``'s own grid) WITH ``S_bar_phi``
  restored (the orchestrator's independent re-derivation of the A20 review's
  restored-arm value for this seed).

This instrument makes NO ``BayesianStatistics.evaluate()`` call anywhere --
it only re-derives the ``B_num``/``beta_Gbar_phi`` numbers from the pinned
production leaf functions (``precompute_phi_marginal_survival``,
``precompute_phi_selection_integrals``) and a deterministic event-set redraw
(``draw_csg_realization``, pure function of ``(seed, arm)`` given the shared
completeness/detection_probability/donor_rows -- ``selfgen_control.py:
488-492``), exactly as :mod:`o4_pairing_test`'s Run phase does. Costing line
(A6/A17, per the O6 registration): < 5 min wall, < 2 GB RSS, local.

Mechanics are reused from the committed O4 instrument
(``o4_pairing_test.py``, commit ``bfe4d09c``) by import, per the launch
task's "copy structure/idioms" instruction and ``o4_merge_shards.py``'s own
``sys.path.insert`` + ``import o4_pairing_test as o4`` pattern
(``o4_merge_shards.py:23-25``) -- no reimplementation of
:func:`o4.event_geometries` / :func:`o4.build_aligned_tables` /
:func:`o4._base_integrand` / :func:`o4._production_window`.

**FLAGGED SPEC AMBIGUITY (A21 -- disclosed, not silently resolved):** the O6
registration's ``r_prod`` description does not explicitly say whether the
per-event ``B_num`` diagnostic comparison (this instrument's "diagnostic
only" section) should compare against the banked seed's OWN off-cell column
or against some other reference. This instrument compares against the banked
``csgf_seed910101/event_likelihoods.csv`` ``B_num`` column (the only banked
per-event numerator for this seed) per the launch task's explicit
instruction; this is diagnostic/informational only and does not feed
``r_prod`` or ``r_A`` themselves, so it cannot contaminate the registered
statistic even if the choice were later judged wrong.

Usage:
    uv run python results/prod2d_closure_20260818/o6_reference_derivation.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.integrate import fixed_quad

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o4_pairing_test as o4  # noqa: E402

from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402
from darksiren_emri.validation.correspondence_1d import CRB_CSV_PATH  # noqa: E402
from darksiren_emri.validation.selfgen_control import (  # noqa: E402
    build_csg_selection_objects,
    draw_csg_realization,
    score_at_h_gen,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results/prod2d_closure_20260818"

SEED: int = 910101
ARM: str = "csgf"
N_EVENTS: int = 200

REGISTRATION_SECTION: str = (
    "results/prod2d_closure_20260818/PREREGISTRATION_SELFGEN_CONTROL.md, "
    "CONFIRMATION RUN O6 -- REGISTRATION (2026-08-21, ledger row #157 item 2), "
    "'Registered reference (A18)' paragraph"
)


def _matched_matrix(
    b_num_lo: npt.NDArray[np.float64],
    b_num_hi: npt.NDArray[np.float64],
    beta_gbar_phi: dict[float, float],
) -> npt.NDArray[np.float64]:
    """Build the ``[matched_lo, nan, matched_hi]`` column stack
    :func:`~darksiren_emri.validation.selfgen_control.score_at_h_gen` expects,
    identical in shape to :func:`o4.compute_factorial_scores`'s ``_matched``
    closure.
    """
    matched_lo = b_num_lo / beta_gbar_phi[o4.H_LO]
    matched_hi = b_num_hi / beta_gbar_phi[o4.H_HI]
    return np.column_stack([matched_lo, np.full_like(matched_lo, np.nan), matched_hi])


def r_prod_b_num(
    geo: o4.EventGeometry,
    h_eval: float,
    completeness: o4.CsgCompletenessModel,
    z_grid: npt.NDArray[np.float64],
    s_grid: npt.NDArray[np.float64],
) -> float:
    r"""r_prod's per-event completion numerator at production settings: the
    per-event window (:func:`o4._production_window`), GL-50 quadrature
    (:data:`o4.PRODUCTION_QUAD_N`), ``S_bar_phi`` applied via endpoint-clamped
    ``np.interp`` on the aligned ``(z_grid, s_grid)`` table -- a literal
    convention copy of ``completion_numerator_integrand_sel_1d``
    (``bayesian_statistics.py:4992-5007``): ``np.interp``'s default
    ``left``/``right`` (clamp to the nearest table endpoint) IS the
    registered clamp convention there, not an approximation of it.

    Returns 0.0 if the production window is degenerate (``z_lo >= z_hi``),
    matching :func:`o4.b_num_arm_a2_gl50_to_trapz1500`'s degenerate-window
    convention.
    """
    z_lo, z_hi = o4._production_window(geo, h_eval, o4.REDSHIFT_UPPER_LIMIT)
    if z_lo >= z_hi:
        return 0.0

    def integrand(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        base = o4._base_integrand(z, h_eval, geo, completeness)
        s_bar = np.interp(z, z_grid, s_grid)
        return np.asarray(base * s_bar, dtype=np.float64)

    return float(fixed_quad(integrand, z_lo, z_hi, n=o4.PRODUCTION_QUAD_N)[0])


def r_a_b_num(
    geo: o4.EventGeometry,
    h_eval: float,
    completeness: o4.CsgCompletenessModel,
    z_grid: npt.NDArray[np.float64],
    s_grid: npt.NDArray[np.float64],
) -> float:
    r"""r_A's per-event completion numerator: O4's aligned arm A domain/
    quadrature (full common domain ``[z_grid[0], z_grid[-1]]``, the SAME
    1500-node trapezoid grid ``beta_Gbar_phi`` uses) WITH ``S_bar_phi``
    restored -- ``o4.b_num_arm_a`` deliberately drops the ``S_bar_phi``
    factor (its "CORRECTED PREMISE" docstring note, since it falsifies the
    OFF-cell banked reference); this REPORTED-ONLY companion restores it,
    reproducing the A20 review's restored-arm reading for this seed.
    """
    base = o4._base_integrand(z_grid, h_eval, geo, completeness)
    integrand = base * s_grid
    return float(np.trapezoid(integrand, z_grid))


def compute_reference(
    geos: list[o4.EventGeometry],
    completeness: o4.CsgCompletenessModel,
    phi_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    beta_gbar_phi: dict[float, float],
) -> dict[str, Any]:
    """Compute ``r_prod`` and ``r_A`` (+ per-h B_num diagnostics) for one seed's geometries."""
    z_lo_grid, s_lo_grid = phi_table[o4.H_LO]
    z_hi_grid, s_hi_grid = phi_table[o4.H_HI]

    b_num_prod: dict[float, npt.NDArray[np.float64]] = {}
    b_num_a: dict[float, npt.NDArray[np.float64]] = {}
    for h_eval, z_grid, s_grid in (
        (o4.H_LO, z_lo_grid, s_lo_grid),
        (o4.H_HI, z_hi_grid, s_hi_grid),
    ):
        b_num_prod[h_eval] = np.array(
            [r_prod_b_num(g, h_eval, completeness, z_grid, s_grid) for g in geos],
            dtype=np.float64,
        )
        b_num_a[h_eval] = np.array(
            [r_a_b_num(g, h_eval, completeness, z_grid, s_grid) for g in geos],
            dtype=np.float64,
        )

    matrix_prod = _matched_matrix(b_num_prod[o4.H_LO], b_num_prod[o4.H_HI], beta_gbar_phi)
    matrix_a = _matched_matrix(b_num_a[o4.H_LO], b_num_a[o4.H_HI], beta_gbar_phi)

    score_prod = score_at_h_gen(matrix_prod, o4.H_GEN, (o4.H_LO, o4.H_GEN, o4.H_HI))
    score_a = score_at_h_gen(matrix_a, o4.H_GEN, (o4.H_LO, o4.H_GEN, o4.H_HI))

    return {
        "r_prod": score_prod,
        "r_A": score_a,
        "b_num_prod_per_h": {str(h): b_num_prod[h].tolist() for h in (o4.H_LO, o4.H_HI)},
        "b_num_A_per_h": {str(h): b_num_a[h].tolist() for h in (o4.H_LO, o4.H_HI)},
    }


def b_num_ratio_diagnostics(
    b_num_prod_per_h: dict[str, list[float]], banked: pd.DataFrame
) -> dict[str, Any]:
    """Diagnostic-only summary stats of ``r_prod``'s per-event B_num vs the
    banked off-cell ``B_num`` column, per h. Does not feed any registered
    statistic (A18 note: the primary/secondary statistics subtract only
    ``r_prod``/0, never this diagnostic).
    """
    out: dict[str, Any] = {}
    for h_str, values in b_num_prod_per_h.items():
        h = float(h_str)
        banked_h = banked[np.isclose(banked["h"], h)].sort_values("event_idx")
        banked_b_num = banked_h["B_num"].to_numpy(dtype=np.float64)
        prod_b_num = np.asarray(values, dtype=np.float64)
        if banked_b_num.size != prod_b_num.size:
            out[h_str] = {
                "pass": False,
                "reason": (
                    f"row-count mismatch: banked={banked_b_num.size}, prod={prod_b_num.size}"
                ),
            }
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = prod_b_num / np.where(banked_b_num != 0.0, banked_b_num, np.nan)
        finite = ratio[np.isfinite(ratio)]
        out[h_str] = {
            "n_events": int(prod_b_num.size),
            "n_finite_ratio": int(finite.size),
            "ratio_mean": float(finite.mean()) if finite.size else None,
            "ratio_std": float(finite.std(ddof=1)) if finite.size > 1 else None,
            "ratio_min": float(finite.min()) if finite.size else None,
            "ratio_max": float(finite.max()) if finite.size else None,
        }
    return out


def main() -> int:
    completeness, detection_probability = build_csg_selection_objects(h_gen=o4.H_GEN)
    donor_rows = pd.read_csv(CRB_CSV_PATH)

    banked_csv = o4.BANKED_DIAG_DIR / f"{ARM}_seed{SEED}" / "event_likelihoods.csv"
    if not banked_csv.is_file():
        print(f"banked diagnostics not found: {banked_csv}", file=sys.stderr)
        return 1
    banked = pd.read_csv(banked_csv)

    # Deterministic redraw (pure function of (seed, arm) given the shared
    # completeness/detection_probability/donor_rows, selfgen_control.py:
    # 488-492) -- NOT an evaluate() call. GATE R4 (o4_pairing_test.py) already
    # proved this redraw reproduces the banked B_num column bit-exactly for
    # the F seeds it covers; this instrument does not re-run that gate (it is
    # zero-evaluate() by construction and has no B_num of its own to gate --
    # the banked-vs-r_prod ratio below is diagnostic, not a gate).
    rows, _draw_diag = draw_csg_realization(
        SEED, ARM, N_EVENTS, completeness, detection_probability, donor_rows
    )
    geos = o4.event_geometries(rows, completeness)
    phi_table, beta_gbar_phi = o4.build_aligned_tables(completeness, detection_probability)

    result = compute_reference(geos, completeness, phi_table, beta_gbar_phi)
    b_num_diag = b_num_ratio_diagnostics(result["b_num_prod_per_h"], banked)

    output: dict[str, Any] = {
        "registered_in": REGISTRATION_SECTION,
        "instrument": "o6_reference_derivation.py",
        "seed": SEED,
        "arm": ARM,
        "h_gen": o4.H_GEN,
        "h_lo": o4.H_LO,
        "h_hi": o4.H_HI,
        "redshift_upper_limit_used": o4.REDSHIFT_UPPER_LIMIT,
        "production_quad_n": o4.PRODUCTION_QUAD_N,
        "aligned_trapz_n": o4.ALIGNED_TRAPZ_N,
        "integration_limit_sigma_multiplier": o4.INTEGRATION_LIMIT_SIGMA_MULTIPLIER,
        "zero_evaluate_note": (
            "This instrument never calls BayesianStatistics.evaluate(); B_num/"
            "beta_Gbar_phi are re-derived from the pinned production leaf "
            "functions (precompute_phi_marginal_survival, "
            "precompute_phi_selection_integrals) and a deterministic "
            "draw_csg_realization redraw, exactly as o4_pairing_test.py's Run "
            "phase does."
        ),
        "r_prod_910101": {
            **result["r_prod"],
            "reference": (
                "PREREGISTRATION_SELFGEN_CONTROL.md CONFIRMATION RUN O6, "
                "'Registered reference (A18)' -- r_prod(910101), the PRIMARY "
                "registered reference O6's S(F6) is scored against."
            ),
            "subtracts": None,
        },
        "r_A_910101_REPORTED_ONLY": {
            **result["r_A"],
            "reference": (
                "PREREGISTRATION_SELFGEN_CONTROL.md CONFIRMATION RUN O6, "
                "'Registered reference (A18)' -- r_A(910101), REPORTED-ONLY "
                "companion (O4's aligned arm A with S_bar_phi restored); "
                "review fleet numbers: restored S_bar_15 = +0.007604 +- "
                "0.018361, per-seed shift +0.124925, sd 0.004625."
            ),
            "subtracts": None,
        },
        "b_num_ratio_diagnostic_vs_banked_off_cell": {
            "per_h": b_num_diag,
            "note": (
                "Diagnostic only (r_prod-numerator vs banked OFF-cell B_num "
                "column, seed 910101): the banked column never applies "
                "S_bar_phi (production 'off' semantics), so a ratio != 1 is "
                "EXPECTED and is the mechanism's own signature, not an error. "
                "Does not feed r_prod, r_A, or any registered band."
            ),
        },
        "git_commit": c1d._git_commit(),
    }

    out_path = RESULTS_DIR / "o6_reference_derivation_output.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"=== O6 reference derivation, seed {SEED} ===")
    print(f"r_prod(910101) mean_score = {result['r_prod']['mean_score']}")
    print(f"r_A(910101)    mean_score = {result['r_A']['mean_score']}  (REPORTED-ONLY)")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
