"""Test 12 (Phase 45 Step 1d): h-independence of empirical p_det asymptote.

Re-runs T10's d_L → 0 detection-rate measurement, but iterates over EVERY
h_inj group present in ``simulations/injections/`` (not just 0.73). The goal
is to verify the load-bearing claim of Plan 45-02:

    The empirical p_det asymptote at d_L → 0 is h_inj-independent within
    statistical noise (max-min spread < 0.10).

If verified, the planner may safely prepend a single h-independent scalar
``p_max_empirical`` to the production interpolator's grid in
``_build_grid_1d``. If rejected (spread ≥ 0.10), the single-scalar fix is
unsound and Plan 45-02 must escalate to a per-h_inj anchor.

Outputs (under ``scripts/bias_investigation/outputs/phase45/``):

* ``p_max_h_independence.json`` — per-group counts/CIs, pooled estimate +
  Wilson 95% CI, max-min spread, sensitivity check at d_L < 0.15 Gpc, and
  two recommended scalars (point estimate and Wilson lower bound).

Probability values are dimensionless and bounded to [0, 1] by construction
(counts ratio). No production code is modified.

Run::

    uv run python scripts/bias_investigation/test_12_p_max_h_independence.py

References
----------
* T10 single-h_inj result: ``pdet_asymptote.json`` (h_inj=0.73 row must
  reproduce here as a limiting-case cross-check).
* Wilson score CI: Brown, Cai, DasGupta (2001) Stat. Sci. 16:101–133;
  ``astropy.stats.binom_conf_interval(..., interval="wilson")``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from astropy.stats import binom_conf_interval  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.grid_quality import load_injection_data
from master_thesis_code.constants import SNR_THRESHOLD

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "bias_investigation" / "outputs" / "phase45"
INJECTION_DIR = PROJECT_ROOT / "simulations" / "injections"

DL_THRESHOLD_GPC: float = 0.10  # headline anchor band
DL_SENSITIVITY_GPC: float = 0.15  # sensitivity cross-check (matches T10)

# Acceptance gates
H_SPREAD_LIMIT: float = 0.10  # claim-h-independence pass condition
PHASE_44_H_SPREAD_BUDGET: float = 0.20  # regression test budget; for traceability
POOLED_CI_LOWER_GATE: float = 0.70  # claim-pooled-anchor-derivation pass condition
PER_GROUP_MIN_N: int = 5  # exclude tiny groups from spread calculation

CONFIDENCE_LEVEL: float = 0.95


# ---------------------------------------------------------------------------
# Core counting + Wilson CI helper (mirrors T10's empirical_asymptote)
# ---------------------------------------------------------------------------
def _empirical_rate_with_ci(
    df: pd.DataFrame,
    dl_threshold: float,
    snr_threshold: float,
) -> dict[str, Any]:
    """Return n_total, n_detected, p_hat, Wilson 95% CI in [0, dl_threshold] Gpc.

    Args:
        df: Injection events for a single h_inj group. Must contain columns
            ``luminosity_distance`` (Gpc) and ``SNR`` (dimensionless).
        dl_threshold: Upper bound of d_L band (Gpc).
        snr_threshold: SNR detection threshold (dimensionless).

    Returns:
        Dict with keys n_total, n_detected, p_hat, ci_lower, ci_upper.
        For n_total == 0 the rate fields are NaN.

    Notes:
        * p_hat = n_detected / n_total ∈ [0, 1] by construction (probability
          is dimensionless; counts ratio cannot exceed 1).
        * Wilson 95% CI via astropy uses the score interval, which is well-
          defined at k=0 and k=n (unlike Wald).
    """
    band_mask = df["luminosity_distance"] < dl_threshold
    n_total = int(band_mask.sum())
    n_det = int(((df["SNR"] >= snr_threshold) & band_mask).sum())
    if n_total == 0:
        return {
            "n_total": 0,
            "n_detected": 0,
            "p_hat": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    if n_det > n_total:  # invariant guard
        msg = f"n_detected={n_det} exceeds n_total={n_total} (band={dl_threshold} Gpc)"
        raise AssertionError(msg)
    p_hat = n_det / n_total
    ci = binom_conf_interval(n_det, n_total, confidence_level=CONFIDENCE_LEVEL, interval="wilson")
    return {
        "n_total": n_total,
        "n_detected": n_det,
        "p_hat": float(p_hat),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
    }


def _per_group_rates(
    groups: dict[float, pd.DataFrame],
    dl_threshold: float,
    snr_threshold: float,
) -> list[dict[str, Any]]:
    """Compute the per-group rate dict for every h_inj key, sorted by h_inj."""
    rows: list[dict[str, Any]] = []
    for h_inj in sorted(groups):
        df = groups[h_inj]
        row = _empirical_rate_with_ci(df, dl_threshold, snr_threshold)
        row["h_inj"] = float(h_inj)
        # Re-order keys so h_inj comes first when serialised
        rows.append(
            {
                "h_inj": row["h_inj"],
                "n_total": row["n_total"],
                "n_detected": row["n_detected"],
                "p_hat": row["p_hat"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
            }
        )
    return rows


def _pooled_rate(
    per_group: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pool counts across all h_inj groups and recompute Wilson CI."""
    n_total_pooled = int(sum(r["n_total"] for r in per_group))
    n_det_pooled = int(sum(r["n_detected"] for r in per_group))
    if n_total_pooled == 0:
        return {
            "n_total": 0,
            "n_detected": 0,
            "p_hat": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    p_hat = n_det_pooled / n_total_pooled
    ci = binom_conf_interval(
        n_det_pooled,
        n_total_pooled,
        confidence_level=CONFIDENCE_LEVEL,
        interval="wilson",
    )
    return {
        "n_total": n_total_pooled,
        "n_detected": n_det_pooled,
        "p_hat": float(p_hat),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
    }


def _spread_max_minus_min(
    per_group: list[dict[str, Any]],
    min_n: int = PER_GROUP_MIN_N,
) -> float:
    """Max-min spread of per-group p_hat over groups with n_total >= min_n.

    Returns NaN if no group meets the count threshold (degenerate case;
    caller should not pass acceptance in that scenario).
    """
    eligible = [r["p_hat"] for r in per_group if r["n_total"] >= min_n]
    if len(eligible) < 2:
        return float("nan")
    return float(max(eligible) - min(eligible))


def _print_per_group_table(per_group: list[dict[str, Any]], dl_threshold: float) -> None:
    """Console-friendly table of per-group rates."""
    print(f"  d_L < {dl_threshold} Gpc")
    print(
        f"  {'h_inj':>7s}  {'n_tot':>5s}  {'n_det':>5s}  {'p_hat':>6s}  "
        f"{'ci_low':>7s}  {'ci_up':>7s}"
    )
    for row in per_group:
        if row["n_total"] == 0:
            print(
                f"  {row['h_inj']:7.2f}  {row['n_total']:5d}  {row['n_detected']:5d}  "
                f"{'  nan':>6s}  {'    nan':>7s}  {'    nan':>7s}"
            )
        else:
            print(
                f"  {row['h_inj']:7.2f}  {row['n_total']:5d}  {row['n_detected']:5d}  "
                f"{row['p_hat']:6.3f}  {row['ci_lower']:7.3f}  {row['ci_upper']:7.3f}"
            )


def main() -> int:
    """Run the h-independence diagnostic and emit the JSON artifact.

    Exit codes:
        0 — h-independence pass (spread < H_SPREAD_LIMIT) AND pooled CI lower
            >= POOLED_CI_LOWER_GATE.
        1 — at least one acceptance gate failed; planner must escalate before
            Plan 45-02 proceeds.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convention assertion: SNR threshold must equal 20 (project lock).
    if float(SNR_THRESHOLD) != 20.0:
        msg = f"SNR_THRESHOLD={SNR_THRESHOLD} != 20.0 (project constant drift)"
        raise AssertionError(msg)

    print(f"Loading injection campaign from {INJECTION_DIR}")
    groups = load_injection_data(INJECTION_DIR)
    print(f"Loaded {len(groups)} h_inj groups: {sorted(groups)}")
    print(f"SNR threshold = {SNR_THRESHOLD}")

    # === Headline: d_L < DL_THRESHOLD_GPC =====================================
    print(f"\n=== Per-group empirical rate (d_L < {DL_THRESHOLD_GPC} Gpc) ===")
    per_group = _per_group_rates(groups, DL_THRESHOLD_GPC, float(SNR_THRESHOLD))
    _print_per_group_table(per_group, DL_THRESHOLD_GPC)

    pooled = _pooled_rate(per_group)
    print(
        f"\n  Pooled: n_tot={pooled['n_total']}, n_det={pooled['n_detected']}, "
        f"p_hat={pooled['p_hat']:.4f} "
        f"[{pooled['ci_lower']:.4f}, {pooled['ci_upper']:.4f}]"
    )

    spread = _spread_max_minus_min(per_group)
    print(f"  Spread max(p_hat) - min(p_hat) over groups with n>={PER_GROUP_MIN_N}: {spread:.4f}")

    # === Sensitivity: d_L < DL_SENSITIVITY_GPC ================================
    print(f"\n=== Sensitivity check (d_L < {DL_SENSITIVITY_GPC} Gpc) ===")
    per_group_sens = _per_group_rates(groups, DL_SENSITIVITY_GPC, float(SNR_THRESHOLD))
    _print_per_group_table(per_group_sens, DL_SENSITIVITY_GPC)
    pooled_sens = _pooled_rate(per_group_sens)
    spread_sens = _spread_max_minus_min(per_group_sens)
    print(
        f"\n  Sensitivity pooled: n_tot={pooled_sens['n_total']}, "
        f"n_det={pooled_sens['n_detected']}, p_hat={pooled_sens['p_hat']:.4f} "
        f"[{pooled_sens['ci_lower']:.4f}, {pooled_sens['ci_upper']:.4f}]"
    )
    print(f"  Sensitivity spread: {spread_sens:.4f}")

    # === Acceptance gates =====================================================
    h_independence_pass = (
        not (spread != spread)  # i.e. not NaN
        and spread < H_SPREAD_LIMIT
    )
    pooled_ci_lower_pass = (
        not (pooled["ci_lower"] != pooled["ci_lower"])
        and pooled["ci_lower"] >= POOLED_CI_LOWER_GATE
    )

    # === Recommended scalars ==================================================
    # Point estimate: pooled p_hat (most aggressive lift)
    # Conservative: Wilson 95% lower bound (RESEARCH.md §4a-(ii) default)
    recommended_point = round(pooled["p_hat"], 4)
    recommended_conservative = round(pooled["ci_lower"], 4)

    # === Bounds invariants (defensive) ========================================
    # Tolerance handles the floating-point edge case where Wilson CI at k=n
    # returns ci_upper ≈ 1 - 1 ULP (e.g. 0.9999999999999999) while p_hat = 1.0.
    fp_tol = 1e-12
    for row in per_group + per_group_sens + [pooled, pooled_sens]:
        if row["n_total"] == 0:
            continue
        if not (-fp_tol <= row["p_hat"] <= 1.0 + fp_tol):
            msg = f"p_hat={row['p_hat']} out of [0, 1]"
            raise AssertionError(msg)
        if not (-fp_tol <= row["ci_lower"] <= 1.0 + fp_tol):
            msg = f"ci_lower={row['ci_lower']} out of [0, 1]"
            raise AssertionError(msg)
        if not (-fp_tol <= row["ci_upper"] <= 1.0 + fp_tol):
            msg = f"ci_upper={row['ci_upper']} out of [0, 1]"
            raise AssertionError(msg)
        if not (row["ci_lower"] - fp_tol <= row["p_hat"] <= row["ci_upper"] + fp_tol):
            msg = (
                f"CI ordering violated: ci_lower={row['ci_lower']} "
                f"<= p_hat={row['p_hat']} <= ci_upper={row['ci_upper']}"
            )
            raise AssertionError(msg)

    # === Assemble JSON artifact ===============================================
    summary: dict[str, Any] = {
        "snr_threshold": float(SNR_THRESHOLD),
        "dl_threshold_gpc": DL_THRESHOLD_GPC,
        "per_group": per_group,
        "pooled": pooled,
        "spread_max_minus_min": spread,
        "sensitivity_dl_lt_0p15": {
            "dl_threshold_gpc": DL_SENSITIVITY_GPC,
            "per_group": per_group_sens,
            "pooled": pooled_sens,
            "spread_max_minus_min": spread_sens,
        },
        "recommended_p_max_empirical": recommended_point,
        "recommended_p_max_empirical_conservative": recommended_conservative,
        "phase_44_h_spread_budget": PHASE_44_H_SPREAD_BUDGET,
        "anchor_h_independence_pass": bool(h_independence_pass),
        "pooled_ci_lower_pass": bool(pooled_ci_lower_pass),
        "h_spread_limit": H_SPREAD_LIMIT,
        "pooled_ci_lower_gate": POOLED_CI_LOWER_GATE,
        "per_group_min_n": PER_GROUP_MIN_N,
        "confidence_level": CONFIDENCE_LEVEL,
    }

    out_json = OUTPUT_DIR / "p_max_h_independence.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    # === Final recommendation block ===========================================
    print("\n=== Recommended scalars for Plan 45-02 ===")
    print(f"  recommended_p_max_empirical              = {recommended_point}")
    print(f"  recommended_p_max_empirical_conservative = {recommended_conservative}")
    print(f"  h-independence spread (must be < {H_SPREAD_LIMIT}) = {spread:.4f}")
    print(f"  pooled CI lower (must be >= {POOLED_CI_LOWER_GATE}) = {pooled['ci_lower']:.4f}")

    if not h_independence_pass:
        print(
            f"\nWARNING: h-independence FAILED — spread={spread:.4f} >= {H_SPREAD_LIMIT}. "
            "Plan 45-02 single-scalar fix is unsound; escalate to per-h_inj anchor."
        )
    if not pooled_ci_lower_pass:
        print(
            f"\nWARNING: pooled CI lower={pooled['ci_lower']:.4f} < {POOLED_CI_LOWER_GATE}. "
            "Pooled empirical asymptote does not clear the current interpolator value."
        )

    if h_independence_pass and pooled_ci_lower_pass:
        print("\nAll acceptance gates PASS.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
