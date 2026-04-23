"""VERIFY-02 abort diagnostic writer.

Invoked only when the Task 2 comparison driver exited 1 (ABORT gate fired). Emits the
D-10 candidate-cause diagnostic markdown with the exact 7-row cause table, full
log-posterior side-by-side curves, and the 4-option triage list.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", required=True, help="Path to verify02_comparison_{TS}.json")
    ap.add_argument("--baseline", required=True, help="Path to v2.1 archived combined_posterior.json")
    ap.add_argument("--current", required=True, help="Path to v2.2 current combined_posterior.json")
    ap.add_argument("--out", required=True, help="Output MD path")
    args = ap.parse_args()

    comp = json.loads(Path(args.comparison).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    current = json.loads(Path(args.current).read_text())

    ts = comp.get("ts", "")
    shift_pct = comp["delta_map_relative"] * 100.0
    baseline_map = comp["baseline_map_h"]
    current_map = comp["current_map_h"]
    signed_delta = current_map - baseline_map
    threshold_val = comp["abort_threshold"]

    bias_v21 = comp.get("bias_percent_v2_1", float("nan"))
    bias_v22 = comp.get("bias_percent_v2_2", float("nan"))
    ciw_v21 = comp.get("ci_width_v2_1", float("nan"))
    ciw_v22 = comp.get("ci_width_v2_2", float("nan"))
    ks_p = comp.get("ks_pvalue", float("nan"))

    # Extract log_posterior arrays from the BaselineSnapshot JSONs. The
    # combined_posterior.json schema has h_values and log_posteriors arrays.
    b_h = baseline.get("h_values", [])
    b_lp = baseline.get("log_posteriors", [])
    c_h = current.get("h_values", [])
    c_lp = current.get("log_posteriors", [])

    # Align on baseline h grid (v2.2 may have additional h-values but baseline is canonical here)
    b_map_lp = dict(zip(b_h, b_lp))
    c_map_lp = dict(zip(c_h, c_lp))
    aligned_h = [h for h in b_h if h in c_map_lp]

    curves_rows = []
    for h in aligned_h:
        lp1 = b_map_lp[h]
        lp2 = c_map_lp[h]
        delta = lp2 - lp1
        curves_rows.append(f"| {h:.4f} | {lp1:+.6e} | {lp2:+.6e} | {delta:+.6e} |")
    curves_table = "\n".join(curves_rows) if curves_rows else "| (no aligned h-values) | - | - | - |"

    # Candidate-cause table — LOCKED at plan time; 7 rows, exactly these causes and phases.
    cause_table = """| Fix | Phase | Code path | Expected shift direction | Diagnostic toggle |
|-----|-------|-----------|--------------------------|-------------------|
| PE-02 per-param epsilon | 37 | `parameter_estimation.py` Fisher loop | CRB scale change → CI width shift → MAP may drift | Re-run with old uniform epsilon=1e-6 |
| PE-01 h-threading | 37 | `parameter_space.py:148` `set_host_galaxy_parameters` | Host d_L recomputed at injected h; shifts Fisher input | Pin h=0.73 in `set_host_galaxy_parameters` default |
| STAT-01 L_cat (1/N)Σ(N_g/D_g) | 38 | `bayesian_statistics.py:929–944` | Catalog weighting change; per-event likelihood scale | Revert to ΣN_g/ΣD_g |
| STAT-03 P_det zero-fill | 38 | `bayesian_statistics.py:119`, `simulation_detection_probability.py:454–463` | P_det floor at grid edges; denominator symmetry change | Revert to NN-fill numerator |
| COORD-02/02b BallTree embedding | 36 | `galaxy_catalogue/handler.py:286–291` | Sky-match distribution change across events | Re-run with old latitude embedding |
| COORD-03 frame rotation | 36 | `galaxy_catalogue/handler.py:486–492` | Up to 23.4° sky shift per event; localization change | Remove astropy BarycentricTrueEcliptic rotation |
| COORD-04 eigenvalue sky radius | 36 | `parameter_estimation.py` sky ellipse | Localization region shape; catalog host count | Revert to old isotropic radius |"""

    triage = """1. Rollback a specific v2.2 fix and re-run VERIFY-02 (identifies causal fix)
2. Open a diagnostic phase 40.1 to per-fix-toggle investigate (requires feature flags — deferred per D-11)
3. Accept the shift as a new v2.2 baseline and update `project_bias_audit.md` memory (if physics-argued)
4. Pause and consult with collaborators"""

    md = f"""# Phase 40 Abort — VERIFY-02 Gate Fired ({ts})

**Shift vs v2.1:** {shift_pct:.2f}% of true h (0.73)
**Baseline MAP (v2.1):** {baseline_map:.4f}
**Current MAP (v2.2):** {current_map:.4f}
**Signed delta:** {signed_delta:+.4f}
**Abort threshold (D-03 #1):** {threshold_val * 100.0:.2f}% (5%)

## Observed Shift

| Metric        | v2.1      | v2.2      | Delta         |
|---------------|-----------|-----------|---------------|
| MAP h         | {baseline_map:.4f} | {current_map:.4f} | {signed_delta:+.4f} |
| |ΔMAP|/0.73   | -         | -         | {shift_pct:.2f}% |
| bias_percent  | {bias_v21} | {bias_v22:+.2f}% | - |
| CI width      | {ciw_v21} | {ciw_v22:.4f} | - |
| KS p-value    | -         | {ks_p:.4g} | - |

## Candidate-Cause Table

{cause_table}

## Full log-posterior curves

| h | log_post_v2.1 | log_post_v2.2 | Δ |
|---|---------------|---------------|---|
{curves_table}

## Triage Options

{triage}

---

See also: `.planning/debug/verify_gate_{ts}.md` (top-level index)
"""
    Path(args.out).write_text(md)
    print(f"Abort diagnostic written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
