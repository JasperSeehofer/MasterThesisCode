"""C-SG v3 fleet readout — committed BEFORE the 42-seed fleet (job 6420343) reports.

Applies the FROZEN bands of record (`csg_pilot_bands_output.json`, published
with the pilot per the prereg's band-formula registration) to the full 46-seed
fleet. No threshold in this file is derived from fleet data.

Statistics (matched channel primary, per the v3 design change):
- S_bar_15 = mean over the 15 C-SG-F seeds of the per-seed mean per-event
  matched score at h_gen; tri-band SELF-CONSISTENT / INTERNAL-DEFECT / MIXED.
- bias_15 = mean matched mean_h - 0.73 over the F seeds; confirmatory tri-band.
- GATE S: OLS mean_h(seed) = b + s*h_gen over all F + delta seeds;
  CONTROL-VALID |s_hat-1| <= 3*SE; CONTROL-INERT if CI(s_hat) contains 0.
- BAND R: |mean_F - mean_E| vs the frozen band_r_abs_max.
- GATE V roll-up under GATE V AMENDMENT 1 (span >= 1 nat,
  sigma_h <= 0.9*sigma_prior, re-evaluated from recorded fields so pilot JSONs
  banked under the superseded v2 thresholds score identically).
- Secondaries (REPORTED-ONLY): full/pure channel per-arm fleet means, the
  full-vs-matched offset (C-SG's impostor+composition mismatch, to compare
  against B-SEL's structure), per-arm sd/SEM tables.

Disclosure: the 4 pilot seeds are part of the 15 F seeds by the registered
design; the frozen bands use the pilot only through sigma_hat (a scale, not a
location), so no location statistic is reused.

The printed verdict is the registered band outcome. The BAND C branch
comparison is a fresh author [RULE]; this script records, it does not rule.

Usage:
    uv run python results/prod2d_closure_20260818/csg_fleet_readout.py --out-dir <dir>
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
H_TRUE = 0.73
ARMS: dict[str, tuple[float, list[int]]] = {
    "csgf": (0.73, list(range(910101, 910116))),
    "csge": (0.73, list(range(910101, 910116))),
    "csgdm": (0.68, list(range(910101, 910109))),
    "csgdp": (0.78, list(range(910101, 910109))),
}


def amended_gate_v_pass(gv: dict[str, Any]) -> bool:
    return bool(gv["span_nats"] >= 1.0 and gv["sigma_h"] <= 0.9 * gv["sigma_prior"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Directory holding <arm>_seed<seed>.json")
    ap.add_argument("--bands", default=str(HERE / "csg_pilot_bands_output.json"))
    ap.add_argument("--out", default=str(HERE / "csg_fleet_readout_output.json"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    bands = json.loads(Path(args.bands).read_text())["bands_of_record"]

    arms: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    gate_v_failures: list[dict[str, Any]] = []
    for arm, (h_gen, seeds) in ARMS.items():
        recs = []
        for seed in seeds:
            p = out_dir / f"{arm}_seed{seed}.json"
            if not p.is_file():
                missing.append(f"{arm}_seed{seed}")
                continue
            r = json.loads(p.read_text())
            gv_ok = amended_gate_v_pass(r["gate_v"])
            if not gv_ok:
                gate_v_failures.append({"arm": arm, "seed": seed, "gate_v": r["gate_v"]})
            recs.append(r)
        if not recs:
            continue
        stats: dict[str, Any] = {"h_gen": h_gen, "n": len(recs)}
        for ch in ("matched", "pure", "full"):
            means = np.array([r["channel_scores"][ch]["mean_h"] for r in recs], dtype=np.float64)
            scores = np.array(
                [r["score_at_h_gen"][ch]["mean_score"] for r in recs], dtype=np.float64
            )
            stats[ch] = {
                "mean_h": float(means.mean()),
                # CORRECTION (2026-08-21 adversarial review FATAL-1, ledger row
                # #152): bias is measured against the ARM'S OWN h_gen, not the
                # global H_TRUE — the first release of this scorer subtracted
                # 0.73 for the delta arms too, and the wrong numbers reached
                # the readout report and row #151 item 3 before being caught.
                "bias": float(means.mean() - h_gen),
                "bias_vs_073_SUPERSEDED": float(means.mean() - H_TRUE),
                "sd": float(means.std(ddof=1)) if means.size > 1 else None,
                "sem": float(means.std(ddof=1) / np.sqrt(means.size)) if means.size > 1 else None,
                "mean_score_at_h_gen": float(scores.mean()),
                "score_sd": float(scores.std(ddof=1)) if scores.size > 1 else None,
                "per_seed_mean_h": [float(x) for x in means],
            }
        arms[arm] = stats

    f = arms["csgf"]["matched"]
    s_bar = float(f["mean_score_at_h_gen"])
    bias = float(f["bias"])
    n_f = int(arms["csgf"]["n"])

    # Registered tri-band (frozen numbers).
    def triband(value: float, self_max: float, defect_max: float) -> str:
        if abs(value) <= self_max:
            return "ESTIMATOR-SELF-CONSISTENT"
        if value <= defect_max:
            return "INTERNAL-DEFECT"
        return "MIXED"

    band_score = triband(
        s_bar, bands["score_self_consistent_abs_max"], bands["score_internal_defect_max"]
    )
    band_bias = triband(
        bias, bands["bias_self_consistent_abs_max"], bands["bias_internal_defect_max"]
    )
    band_c = band_score if band_score == band_bias else "MIXED"

    # GATE S over F + delta seeds (matched channel).
    xs, ys = [], []
    for arm in ("csgf", "csgdm", "csgdp"):
        if arm not in arms:
            continue
        for m in arms[arm]["matched"]["per_seed_mean_h"]:
            xs.append(arms[arm]["h_gen"])
            ys.append(m)
    x = np.array(xs)
    y = np.array(ys)
    sxx = float(((x - x.mean()) ** 2).sum())
    s_hat = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    b_hat = float(y.mean() - s_hat * x.mean())
    resid = y - (b_hat + s_hat * x)
    dof = max(len(xs) - 2, 1)
    se_slope = float(np.sqrt((resid**2).sum() / dof / sxx))
    gate_s = (
        "CONTROL-INERT-STOP"
        if abs(s_hat) <= 3.0 * se_slope
        else ("CONTROL-VALID" if abs(s_hat - 1.0) <= 3.0 * se_slope else "CONTROL-SLOPE-ANOMALY")
    )
    intercept_bias = float(b_hat + (s_hat - 1.0) * H_TRUE)

    band_r_gap = (
        abs(arms["csgf"]["matched"]["mean_h"] - arms["csge"]["matched"]["mean_h"])
        if "csge" in arms
        else None
    )
    band_r = (
        None
        if band_r_gap is None
        else ("CONSISTENT" if band_r_gap <= bands["band_r_abs_max"] else "SIGMA-SENSITIVE")
    )

    out = {
        "registered_in": "csg_fleet_readout.py (committed pre-fleet-data) + frozen bands",
        "missing_seeds": missing,
        "gate_v_failures_amended": gate_v_failures,
        "n_f_seeds": n_f,
        "S_bar_15": s_bar,
        "bias_15": bias,
        "bands_applied": bands,
        "band_on_score": band_score,
        "band_on_bias": band_bias,
        "BAND_C": band_c,
        "band_c_note": "fresh author [RULE] — recorded, not ruled",
        # Added 2026-08-21 post-review (MAJOR-1/MAJOR-2, ledger row #152): the
        # REALIZED F-arm scatter, which is 1.56x the pilot's sigma_hat. The
        # frozen bands are NOT retuned (anti-tuning); these fields quantify the
        # verdict's margin under the realized scatter.
        "realized_scatter": {
            "score_sd": f["score_sd"],
            "score_sem": (f["score_sd"] / np.sqrt(n_f)) if f["score_sd"] else None,
            "sigma_from_zero": (abs(s_bar) / (f["score_sd"] / np.sqrt(n_f)))
            if f["score_sd"]
            else None,
            "sigma_past_defect_edge": (
                (abs(s_bar) - abs(bands["score_internal_defect_max"]))
                / (f["score_sd"] / np.sqrt(n_f))
            )
            if f["score_sd"]
            else None,
            "n_adequacy_on_realized": (
                abs(bands["score_internal_defect_max"]) / (f["score_sd"] / np.sqrt(n_f))
            )
            if f["score_sd"]
            else None,
        },
        "gate_s": {
            "s_hat": s_hat,
            "se_slope": se_slope,
            "intercept_bias_estimate": intercept_bias,
            "verdict": gate_s,
            "n_points": len(xs),
        },
        "band_r": {
            "gap_F_vs_E": band_r_gap,
            "threshold": bands["band_r_abs_max"],
            "verdict": band_r,
        },
        "per_arm": arms,
        "reference": {
            "S_REF": -0.1932,
            "B_REF": -0.0846,
            "note": "B-SEL matched-channel references (12 banked seeds, rows #149-#150)",
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "missing": len(missing),
                "gate_v_failures": len(gate_v_failures),
                "S_bar_15": s_bar,
                "bias_15": bias,
                "band_on_score": band_score,
                "band_on_bias": band_bias,
                "BAND_C": band_c,
                "gate_s": gate_s,
                "s_hat": s_hat,
                "band_r": band_r,
                "out": args.out,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
