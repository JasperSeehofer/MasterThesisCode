#!/usr/bin/env python
"""Independent stage-5 readout scorer for A-FULL (PREREGISTRATION_A_FULL_STAGE5.md).

Pre-committed BEFORE the A-FULL arm data exists (analysis-code-freeze discipline, prereg
§6 item 2), mirroring score_stage3.py's own discipline. Recomputes every statistic from the
raw per-seed ``ln_post_1d`` / ``ln_post_2d`` vectors of the arm result file; the file's
``aggregate`` block and stored per-seed scalar fields (``map_*``, ``hpd*_*``) are never read as
inputs -- MAP, bias, HPD coverage, and the tilt are all recomputed from the raw posterior grid,
exactly as score_stage3.py does.

Implements the five registered decision statistics (prereg §4):
    DS-F1 (PRIMARY, branch-carrying): seed-mean 1D venue tilt T at h_true=0.730, grid-neighbour
        central difference. PASS iff mean in [-131.5, +192.7]. 2D tilt also reported (no band).
    DS-F2 (WEAK, report-only): MAP bias mean +/- SE, both channels. Expectation |b(1D)| <= 0.003,
        reported met/not-met, non-adjudicating.
    DS-F3 (branch-carrying jointly with DS-F1): HPD50/68/90 coverage fractions, both channels.
        RESTORED iff (1D) hpd50 in [0.20,0.80] AND hpd68 in [0.40,0.96] AND hpd90 in [0.72,1.00].
    DS-F4 (descriptive, no band): T(2D) - T(1D), paired mean +/- paired SE.
    DS-F5 (descriptive): per-seed T scatter, railed_low/high counts, non-finite counts; any
        rail or non-finite event triggers the STOP flag.

Branch determination (prereg §5) is PRESENTED, NOT ADJUDICATED -- the author rules on the
five-way branch table; this script only reports which entry the measured pattern points to.

Mechanics dry-run (prereg §6 item 2): run with ``--input`` pointed at the committed
AJREN_h0p730_results_seeds0_25.json (same per-seed schema) before the AFULL data exists, to
verify the scorer runs end-to-end. DS verdicts on that input are expected to be FAIL /
not-restored -- that is the point of the dry run, mechanics only, not a physics check.

Emits ``score_stage5_output.json`` next to this file and a compact stdout summary block.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
H_TRUE = 0.730
HPD_LEVELS = (0.50, 0.68, 0.90)

# ---- registered pins (PREREGISTRATION_A_FULL_STAGE5.md §4) ----------------
DS_F1_LO = -131.5
DS_F1_HI = 192.7
DS_F2_EXPECT_ABS_BIAS = 0.003
DS_F3_BANDS: dict[int, tuple[float, float]] = {
    50: (0.20, 0.80),
    68: (0.40, 0.96),
    90: (0.72, 1.00),
}
DS_F4_REFERENCE = {"center": 129.0, "half_width": 24.0}  # coded form's excess, descriptive only

DEFAULT_INPUT = HERE / "AFULL_h0p730_results_seeds0_25.json"


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


# ── recomputation kernel (ports of score_stage3.py's, verbatim) ────────────
def posterior_readout(
    h_grid: NDArray[np.float64], ln_post: NDArray[np.float64]
) -> dict[str, float]:
    i = int(np.argmax(ln_post))
    return {
        "map_index": float(i),
        "map": float(h_grid[i]),
        "railed_low": float(i == 0),
        "railed_high": float(i == len(h_grid) - 1),
    }


def hpd_contains(
    h_grid: NDArray[np.float64], post: NDArray[np.float64], h_true: float, level: float
) -> bool:
    dh = np.gradient(h_grid)
    mass = post * dh
    order = np.argsort(post)[::-1]
    csum = np.cumsum(mass[order])
    k = min(int(np.searchsorted(csum, level)), order.size - 1)
    thresh = float(post[order[k]])
    p_true = float(np.interp(h_true, h_grid, post))
    return bool(p_true >= thresh)


def pp_readout(
    h_grid: NDArray[np.float64], ln_post: NDArray[np.float64], h_true: float
) -> dict[str, float]:
    p = np.exp(ln_post - float(np.max(ln_post)))
    norm_c = float(np.trapezoid(p, h_grid))
    post = p / norm_c
    r: dict[str, float] = {}
    for lv in HPD_LEVELS:
        r[f"hpd{int(round(lv * 100))}"] = float(hpd_contains(h_grid, post, h_true, lv))
    return r


# ── DS-F1/F4: grid-neighbour central-difference tilt ────────────────────────
def tilt_at_truth(
    h_grid: NDArray[np.float64], ln_post: NDArray[np.float64], h_true: float
) -> float:
    """Grid-neighbour central difference of ln_post at h_true (prereg §4 DS-F1).

    Ported verbatim from ``l4_afull_premeasure.py``'s pre-measurement construction:
    ``i_true = argmin(|h_grid - h_true|)``, neighbours ``i_true -/+ 1``, slope
    ``(ln_post[i_hi] - ln_post[i_lo]) / (h_grid[i_hi] - h_grid[i_lo])``.
    """
    i_true = int(np.argmin(np.abs(h_grid - h_true)))
    i_lo, i_hi = i_true - 1, i_true + 1
    dh = float(h_grid[i_hi] - h_grid[i_lo])
    return float((ln_post[i_hi] - ln_post[i_lo]) / dh)


# ── per-seed, per-channel recomputation ─────────────────────────────────────
def score_channel(
    h_grid: NDArray[np.float64], recs: list[dict[str, Any]], ch: str
) -> dict[str, Any]:
    """Recompute T, MAP bias, and HPD coverage per seed for one channel (1d/2d)."""
    seeds: list[int] = []
    tvals: list[float] = []
    biases: list[float] = []
    rl: list[float] = []
    rh: list[float] = []
    hpd: dict[int, list[float]] = {50: [], 68: [], 90: []}
    nonfinite = 0
    nonfinite_seeds: list[int] = []
    for r in recs:
        ln = np.array(r[f"ln_post_{ch}"], dtype=np.float64)
        if not np.all(np.isfinite(ln)):
            nonfinite += 1
            nonfinite_seeds.append(int(r["seed"]))
            continue
        pr = posterior_readout(h_grid, ln)
        pp = pp_readout(h_grid, ln, H_TRUE)
        t = tilt_at_truth(h_grid, ln, H_TRUE)
        seeds.append(int(r["seed"]))
        tvals.append(t)
        biases.append(pr["map"] - H_TRUE)
        rl.append(pr["railed_low"])
        rh.append(pr["railed_high"])
        for lv in (50, 68, 90):
            hpd[lv].append(pp[f"hpd{lv}"])
    n = len(tvals)
    if n == 0:
        return {"n": 0, "status": "NO FINITE SEEDS", "nonfinite": nonfinite}

    def _mean_se(vals: list[float]) -> tuple[float, float]:
        m = sum(vals) / len(vals)
        sd = st.stdev(vals) if len(vals) > 1 and len(set(vals)) > 1 else 0.0
        return m, sd / math.sqrt(len(vals))

    t_mean, t_se = _mean_se(tvals)
    b_mean, b_se = _mean_se(biases)
    return {
        "n": n,
        "nonfinite": nonfinite,
        "nonfinite_seeds": nonfinite_seeds,
        "seeds": seeds,
        "T_vals": tvals,
        "T_mean": t_mean,
        "T_se": t_se,
        "bias_mean": b_mean,
        "bias_se": b_se,
        "hpd50_cov": sum(hpd[50]) / n,
        "hpd68_cov": sum(hpd[68]) / n,
        "hpd90_cov": sum(hpd[90]) / n,
        "railed_low_frac": sum(rl) / n,
        "railed_high_frac": sum(rh) / n,
        "railed_low_n": int(sum(rl)),
        "railed_high_n": int(sum(rh)),
    }


# ── DS-F1: 1D (primary, branch-carrying) + 2D (reported) tilt ──────────────
def score_ds_f1(ch1d: dict[str, Any], ch2d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"band_lo": DS_F1_LO, "band_hi": DS_F1_HI}
    if ch1d.get("n", 0) == 0:
        out["1d"] = {"status": "NO FINITE SEEDS"}
        out["pass"] = False
    else:
        m, se = ch1d["T_mean"], ch1d["T_se"]
        out["1d"] = {"mean": m, "se": se, "n": ch1d["n"]}
        out["pass"] = bool(DS_F1_LO <= m <= DS_F1_HI)
    if ch2d.get("n", 0) == 0:
        out["2d"] = {"status": "NO FINITE SEEDS"}
    else:
        out["2d"] = {"mean": ch2d["T_mean"], "se": ch2d["T_se"], "n": ch2d["n"]}
    return out


# ── DS-F2 (WEAK, report-only): MAP bias, both channels ──────────────────────
def score_ds_f2(ch1d: dict[str, Any], ch2d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"expect_abs_bias_1d_le": DS_F2_EXPECT_ABS_BIAS}
    for key, ch in (("1d", ch1d), ("2d", ch2d)):
        if ch.get("n", 0) == 0:
            out[key] = {"status": "NO FINITE SEEDS"}
        else:
            out[key] = {"mean": ch["bias_mean"], "se": ch["bias_se"], "n": ch["n"]}
    if ch1d.get("n", 0) > 0:
        out["met"] = bool(abs(ch1d["bias_mean"]) <= DS_F2_EXPECT_ABS_BIAS)
    else:
        out["met"] = None
    out["status"] = "WEAK, non-branch-carrying (report-only)"
    return out


# ── DS-F3: coverage restoration ─────────────────────────────────────────────
def score_ds_f3(ch1d: dict[str, Any], ch2d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"bands": DS_F3_BANDS}
    for key, ch in (("1d", ch1d), ("2d", ch2d)):
        if ch.get("n", 0) == 0:
            out[key] = {"status": "NO FINITE SEEDS"}
        else:
            out[key] = {
                "hpd50": ch["hpd50_cov"],
                "hpd68": ch["hpd68_cov"],
                "hpd90": ch["hpd90_cov"],
                "n": ch["n"],
            }
    if ch1d.get("n", 0) > 0:
        c = out["1d"]
        lo50, hi50 = DS_F3_BANDS[50]
        lo68, hi68 = DS_F3_BANDS[68]
        lo90, hi90 = DS_F3_BANDS[90]
        out["restored"] = bool(
            (lo50 <= c["hpd50"] <= hi50)
            and (lo68 <= c["hpd68"] <= hi68)
            and (lo90 <= c["hpd90"] <= hi90)
        )
    else:
        out["restored"] = False
    return out


# ── DS-F4 (descriptive, no band): T(2D) - T(1D), paired ─────────────────────
def score_ds_f4(ch1d: dict[str, Any], ch2d: dict[str, Any]) -> dict[str, Any]:
    if ch1d.get("n", 0) == 0 or ch2d.get("n", 0) == 0:
        return {"status": "NO FINITE SEEDS", "reference": DS_F4_REFERENCE}
    s1 = {s: t for s, t in zip(ch1d["seeds"], ch1d["T_vals"], strict=True)}
    s2 = {s: t for s, t in zip(ch2d["seeds"], ch2d["T_vals"], strict=True)}
    common = sorted(set(s1) & set(s2))
    diffs = [s2[s] - s1[s] for s in common]
    n = len(diffs)
    if n == 0:
        return {"status": "NO PAIRED SEEDS", "reference": DS_F4_REFERENCE}
    mean = sum(diffs) / n
    sd = st.stdev(diffs) if n > 1 and len(set(diffs)) > 1 else 0.0
    se = sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "reference": DS_F4_REFERENCE,
        "status": "descriptive, no band",
    }


# ── DS-F5 (descriptive): scatter, rails, non-finite, STOP flag ──────────────
def score_ds_f5(ch1d: dict[str, Any], ch2d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    stop = False
    for key, ch in (("1d", ch1d), ("2d", ch2d)):
        if ch.get("n", 0) == 0:
            out[key] = {"status": "NO FINITE SEEDS"}
            stop = True
            continue
        tvals = ch["T_vals"]
        sd = st.stdev(tvals) if len(tvals) > 1 and len(set(tvals)) > 1 else 0.0
        rl_n, rh_n, nf = ch["railed_low_n"], ch["railed_high_n"], ch["nonfinite"]
        out[key] = {
            "T_min": min(tvals),
            "T_max": max(tvals),
            "T_sd": sd,
            "railed_low_n": rl_n,
            "railed_high_n": rh_n,
            "nonfinite_n": nf,
            "nonfinite_seeds": ch["nonfinite_seeds"],
        }
        if rl_n > 0 or rh_n > 0 or nf > 0:
            stop = True
    out["stop"] = stop
    return out


# ── branch determination (§5; PRESENTED, NOT ADJUDICATED) ──────────────────
def determine_branch(
    input_present: bool,
    ds_f1: dict[str, Any] | None,
    ds_f3: dict[str, Any] | None,
    ds_f5: dict[str, Any] | None,
) -> dict[str, Any]:
    if not input_present:
        return {
            "status": "NOT PRESENTED",
            "reason": "execution-completeness clause (§5): input file missing",
        }
    assert ds_f1 is not None and ds_f3 is not None and ds_f5 is not None

    if ds_f5.get("stop"):
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "5. OTHER / confounded",
            "fired_by": "DS-F5 STOP: railed posterior, non-finite ln_post, or 0 finite seeds",
        }

    if ds_f1.get("1d", {}).get("status") == "NO FINITE SEEDS":
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "5. OTHER / confounded",
            "fired_by": "DS-F1 1D channel has no finite seeds",
        }

    ds_f1_pass = bool(ds_f1["pass"])
    ds_f3_restored = bool(ds_f3["restored"])
    t_mean = ds_f1["1d"]["mean"]

    if ds_f1_pass and ds_f3_restored:
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "1. DS-F1 PASS + DS-F3 RESTORED (M-OWNED-CLOSED candidate)",
            "fired_by": f"T(1D)={t_mean:+.1f} in band, coverage restored on all three HPD levels",
        }
    if ds_f1_pass and not ds_f3_restored:
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "2. DS-F1 PASS + DS-F3 NOT restored (width/curvature channel leads)",
            "fired_by": f"T(1D)={t_mean:+.1f} in band, coverage NOT restored",
        }
    if t_mean > DS_F1_HI:
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "3. DS-F1 FAIL high (positive term still missing)",
            "fired_by": f"T(1D)={t_mean:+.1f} > +{DS_F1_HI}",
        }
    if t_mean < DS_F1_LO:
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "4. DS-F1 FAIL low (over-correction)",
            "fired_by": f"T(1D)={t_mean:+.1f} < {DS_F1_LO}",
        }
    return {
        "status": "PRESENTED, NOT ADJUDICATED — the author rules",
        "branch": "5. OTHER / confounded",
        "fired_by": "pattern of verdicts does not match branches 1-4",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="stage-5 arm result JSON (default: AFULL_h0p730_results_seeds0_25.json); "
        "pass the AJREN file for the prereg §6 item 2 mechanics dry-run",
    )
    ap.add_argument("--out", type=Path, default=HERE / "score_stage5_output.json")
    args = ap.parse_args(argv)

    P = print
    out: dict[str, Any] = {}

    input_present = args.input.exists()
    out["inputs"] = {"input_path": str(args.input), "input_present": input_present}

    P("=== INPUTS ===")
    P(
        f"  {args.input.name}: "
        f"{'FOUND' if input_present else 'MISSING (analysis-code-freeze: expected before data exists)'}"
    )
    P()

    ds_f1: dict[str, Any] | None = None
    ds_f2: dict[str, Any] | None = None
    ds_f3: dict[str, Any] | None = None
    ds_f4: dict[str, Any] | None = None
    ds_f5: dict[str, Any] | None = None

    if input_present:
        d = load(args.input)
        grid = np.array(d["config"]["h_grid"], dtype=np.float64)
        recs = d["per_seed"]
        ch1d = score_channel(grid, recs, "1d")
        ch2d = score_channel(grid, recs, "2d")

        ds_f1 = score_ds_f1(ch1d, ch2d)
        out["ds_f1"] = ds_f1
        P("=== DS-F1: 1D venue tilt at truth (PRIMARY, branch-carrying) ===")
        if "mean" in ds_f1["1d"]:
            P(
                f"  1D: T_mean={ds_f1['1d']['mean']:+.1f} SE={ds_f1['1d']['se']:.1f} "
                f"N={ds_f1['1d']['n']} band=[{DS_F1_LO:+.1f},{DS_F1_HI:+.1f}] "
                f"-> {'PASS' if ds_f1['pass'] else 'FAIL'}"
            )
        else:
            P("  1D: NO FINITE SEEDS")
        if "mean" in ds_f1["2d"]:
            P(
                f"  2D (reported, no band): T_mean={ds_f1['2d']['mean']:+.1f} SE={ds_f1['2d']['se']:.1f}"
            )
        else:
            P("  2D: NO FINITE SEEDS")
        P()

        ds_f2 = score_ds_f2(ch1d, ch2d)
        out["ds_f2"] = ds_f2
        P("=== DS-F2: MAP bias (WEAK, report-only) ===")
        for key in ("1d", "2d"):
            c = ds_f2[key]
            if "mean" in c:
                P(f"  {key}: bias_mean={c['mean']:+.4f} SE={c['se']:.4f} N={c['n']}")
            else:
                P(f"  {key}: NO FINITE SEEDS")
        P(
            f"  expectation |b(1D)| <= {DS_F2_EXPECT_ABS_BIAS}: {'MET' if ds_f2['met'] else 'NOT MET' if ds_f2['met'] is not None else 'N/A'}"
        )
        P()

        ds_f3 = score_ds_f3(ch1d, ch2d)
        out["ds_f3"] = ds_f3
        P("=== DS-F3: coverage restoration ===")
        for key in ("1d", "2d"):
            c = ds_f3[key]
            if "hpd50" in c:
                P(
                    f"  {key}: hpd50/68/90={c['hpd50']:.3f}/{c['hpd68']:.3f}/{c['hpd90']:.3f} N={c['n']}"
                )
            else:
                P(f"  {key}: NO FINITE SEEDS")
        P(f"  RESTORED (1D bands): {ds_f3['restored']}")
        P()

        ds_f4 = score_ds_f4(ch1d, ch2d)
        out["ds_f4"] = ds_f4
        P("=== DS-F4: T(2D) - T(1D) (descriptive, no band) ===")
        if "mean" in ds_f4:
            ref = ds_f4["reference"]
            P(
                f"  mean={ds_f4['mean']:+.1f} SE={ds_f4['se']:.1f} N={ds_f4['n']} "
                f"(coded-form reference: {ref['center']:+.1f} +/- {ref['half_width']:.1f})"
            )
        else:
            P(f"  {ds_f4['status']}")
        P()

        ds_f5 = score_ds_f5(ch1d, ch2d)
        out["ds_f5"] = ds_f5
        P("=== DS-F5: per-seed T scatter, rails, non-finite (descriptive) ===")
        for key in ("1d", "2d"):
            c = ds_f5[key]
            if "T_min" in c:
                P(
                    f"  {key}: T in [{c['T_min']:+.1f},{c['T_max']:+.1f}] sd={c['T_sd']:.1f} "
                    f"railed_low={c['railed_low_n']} railed_high={c['railed_high_n']} "
                    f"nonfinite={c['nonfinite_n']}"
                )
            else:
                P(f"  {key}: {c['status']}")
        if ds_f5["stop"]:
            P("  *** STOP: rail or non-finite event detected (prereg §6 item 4) ***")
        P()
    else:
        out["ds_f1"] = None
        out["ds_f2"] = None
        out["ds_f3"] = None
        out["ds_f4"] = None
        out["ds_f5"] = None

    branch = determine_branch(input_present, ds_f1, ds_f3, ds_f5)
    out["branch"] = branch
    P("=== BRANCH (§5) ===")
    P("  PRESENTED, NOT ADJUDICATED — the author rules")
    P(f"  status: {branch['status']}")
    if "branch" in branch:
        P(f"  branch: {branch['branch']}")
    P(f"  reason: {branch.get('reason', branch.get('fired_by'))}")
    P()

    args.out.write_text(json.dumps(out, indent=1))
    P(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
