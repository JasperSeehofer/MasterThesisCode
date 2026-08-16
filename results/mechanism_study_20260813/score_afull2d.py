#!/usr/bin/env python
"""Independent A-FULL-2D readout scorer (PREREGISTRATION_A_FULL_2D.md).

Pre-committed BEFORE the AFULL2D arm data exists (analysis-code-freeze discipline, prereg §6
item 2), mirroring ``score_stage5.py``'s own discipline (which mirrors ``score_stage3.py``).
Recomputes every statistic from the raw per-seed ``ln_post_1d`` / ``ln_post_2d`` vectors of the
arm result file; the file's ``aggregate`` block and stored per-seed scalar fields (``map_*``,
``hpd*_*``) are never read as inputs -- MAP, bias, HPD coverage, and the tilt are all recomputed
from the raw posterior grid.

Implements the five registered decision statistics (prereg §4):
    DS-G1 (PRIMARY, branch-carrying): paired 2D-1D excess at truth (mean over seeds of T2 - T1,
        grid-neighbour central difference at h_true, k=20/22, raw ln_post vectors). PASS iff mean
        in [-15.7, -7.8].
    DS-G2 (secondary, non-branch-carrying): tilts at truth. T(2D) in [-143.8, +181.6] and T(1D)
        inside the stage-5 DS-F1 band [-131.5, +192.7]; both reported, neither branch-forcing.
    DS-G3 (2D coverage, branch-carrying jointly with DS-G1): binomial bands at nominal on the
        2D-channel posterior. RESTORED iff hpd50 in [0.20,0.80] AND hpd68 in [0.40,0.96] AND
        hpd90 in [0.72,1.00].
    DS-G4 (1D invariance, branch-carrying as STOP): c1 bit-identity -- checked separately
        post-run (the pre-submission gate, prereg §6 item 1), NOT recomputed by this scorer from
        the arm JSON alone (it has no a_full comparison column); this scorer only notes the
        check's status if supplied via --c1-bit-identity-max-diff.
    DS-G5 (specificity, descriptive): 2D MAP bias, per-seed T scatter, zero-rail/NaN counts. Any
        rail or non-finite event triggers the STOP flag.

Branch determination (prereg §5) is PRESENTED, NOT ADJUDICATED -- the author rules on the
five-way branch table; this script only reports which entry the measured pattern points to.

Mechanics dry-run (prereg §6 item 2): run with ``--input`` pointed at the committed
AFULL_h0p730_results_seeds0_25.json (schema-identical; 2D fields exercised) before the AFULL2D
data exists, to verify the scorer runs end-to-end. DS verdicts on that input read whatever the
a_full data reads -- that is the point of the dry run, mechanics only, not a physics check (the
bands are calibrated to the a_full_gsel mirror pre-measurement, not to a_full).

Emits ``score_afull2d_output.json`` next to this file and a compact stdout summary block.
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

# ---- registered pins (PREREGISTRATION_A_FULL_2D.md §4) --------------------
DS_G1_LO = -15.7
DS_G1_HI = -7.8
DS_G2_T2_LO = -143.8
DS_G2_T2_HI = 181.6
DS_G2_T1_LO = -131.5  # stage-5 DS-F1 band, quoted (prereg §4 DS-G2)
DS_G2_T1_HI = 192.7
DS_G3_BANDS: dict[int, tuple[float, float]] = {
    50: (0.20, 0.80),
    68: (0.40, 0.96),
    90: (0.72, 1.00),
}
DS_G1_MIRROR_REFERENCE = {"mean": -11.740, "se": 1.038, "per_seed_sd": 4.019, "n_mirror": 15}

DEFAULT_INPUT = HERE / "AFULL2D_h0p730_results_seeds0_25.json"


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


# ── recomputation kernel (ports of score_stage5.py's, verbatim) ────────────
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


# ── DS-G1/G2: grid-neighbour central-difference tilt ────────────────────────
def tilt_at_truth(
    h_grid: NDArray[np.float64], ln_post: NDArray[np.float64], h_true: float
) -> float:
    """Grid-neighbour central difference of ln_post at h_true (prereg §4 DS-G1/DS-G2).

    Ported verbatim from ``l6_der2_gsel_premeasure.py``'s / ``score_stage5.py``'s
    construction: ``i_true = argmin(|h_grid - h_true|)``, neighbours ``i_true -/+ 1``,
    slope ``(ln_post[i_hi] - ln_post[i_lo]) / (h_grid[i_hi] - h_grid[i_lo])`` -- the
    same k=20/22 (h=0.725/0.735) central difference the premeasure used when the
    canonical 41-point grid is passed (registered convention, prereg §2).
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


# ── DS-G1: PRIMARY, branch-carrying -- paired 2D-1D excess at truth ────────
def score_ds_g1(ch1d: dict[str, Any], ch2d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "band_lo": DS_G1_LO,
        "band_hi": DS_G1_HI,
        "mirror_reference": DS_G1_MIRROR_REFERENCE,
    }
    if ch1d.get("n", 0) == 0 or ch2d.get("n", 0) == 0:
        out["status"] = "NO FINITE SEEDS"
        out["pass"] = False
        return out
    s1 = {s: t for s, t in zip(ch1d["seeds"], ch1d["T_vals"], strict=True)}
    s2 = {s: t for s, t in zip(ch2d["seeds"], ch2d["T_vals"], strict=True)}
    common = sorted(set(s1) & set(s2))
    diffs = [s2[s] - s1[s] for s in common]
    n = len(diffs)
    if n == 0:
        out["status"] = "NO PAIRED SEEDS"
        out["pass"] = False
        return out
    mean = sum(diffs) / n
    sd = st.stdev(diffs) if n > 1 and len(set(diffs)) > 1 else 0.0
    se = sd / math.sqrt(n)
    out.update({"n": n, "mean": mean, "sd": sd, "se": se})
    out["pass"] = bool(DS_G1_LO <= mean <= DS_G1_HI)
    return out


# ── DS-G2 (secondary, non-branch-carrying): T(2D) and T(1D) at truth ───────
def score_ds_g2(ch1d: dict[str, Any], ch2d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "band_2d": [DS_G2_T2_LO, DS_G2_T2_HI],
        "band_1d": [DS_G2_T1_LO, DS_G2_T1_HI],
        "status": "secondary, non-branch-carrying (report-only)",
    }
    if ch2d.get("n", 0) == 0:
        out["2d"] = {"status": "NO FINITE SEEDS"}
        out["pass_2d"] = None
    else:
        out["2d"] = {"mean": ch2d["T_mean"], "se": ch2d["T_se"], "n": ch2d["n"]}
        out["pass_2d"] = bool(DS_G2_T2_LO <= ch2d["T_mean"] <= DS_G2_T2_HI)
    if ch1d.get("n", 0) == 0:
        out["1d"] = {"status": "NO FINITE SEEDS"}
        out["pass_1d"] = None
    else:
        out["1d"] = {"mean": ch1d["T_mean"], "se": ch1d["T_se"], "n": ch1d["n"]}
        out["pass_1d"] = bool(DS_G2_T1_LO <= ch1d["T_mean"] <= DS_G2_T1_HI)
    return out


# ── DS-G3: 2D-channel coverage restoration ──────────────────────────────────
def score_ds_g3(ch2d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"bands": DS_G3_BANDS}
    if ch2d.get("n", 0) == 0:
        out["2d"] = {"status": "NO FINITE SEEDS"}
        out["restored"] = False
        return out
    c = {
        "hpd50": ch2d["hpd50_cov"],
        "hpd68": ch2d["hpd68_cov"],
        "hpd90": ch2d["hpd90_cov"],
        "n": ch2d["n"],
    }
    out["2d"] = c
    lo50, hi50 = DS_G3_BANDS[50]
    lo68, hi68 = DS_G3_BANDS[68]
    lo90, hi90 = DS_G3_BANDS[90]
    out["restored"] = bool(
        (lo50 <= c["hpd50"] <= hi50)
        and (lo68 <= c["hpd68"] <= hi68)
        and (lo90 <= c["hpd90"] <= hi90)
    )
    return out


# ── DS-G4: 1D invariance -- checked separately, note only ──────────────────
def score_ds_g4(c1_bit_identity_max_diff: float | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": (
            "checked separately post-run (prereg §6 item 1 pre-submission gate + "
            "DS-G4 note); this scorer does not recompute c1 bit-identity from the "
            "arm JSON alone (no a_full comparison column in the schema)"
        ),
    }
    if c1_bit_identity_max_diff is None:
        out["max_diff"] = None
        out["pass"] = None
    else:
        out["max_diff"] = c1_bit_identity_max_diff
        out["pass"] = bool(c1_bit_identity_max_diff == 0.0)
    return out


# ── DS-G5 (descriptive): scatter, rails, non-finite, STOP flag ──────────────
def score_ds_g5(ch1d: dict[str, Any], ch2d: dict[str, Any]) -> dict[str, Any]:
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
            "bias_mean": ch["bias_mean"],
            "bias_se": ch["bias_se"],
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
    ds_g1: dict[str, Any] | None,
    ds_g3: dict[str, Any] | None,
    ds_g5: dict[str, Any] | None,
) -> dict[str, Any]:
    if not input_present:
        return {
            "status": "NOT PRESENTED",
            "reason": "execution-completeness clause (§6): input file missing",
        }
    assert ds_g1 is not None and ds_g3 is not None and ds_g5 is not None

    if ds_g5.get("stop"):
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "5. OTHER / confounded",
            "fired_by": "DS-G5 STOP: railed posterior, non-finite ln_post, or 0 finite seeds",
        }

    if ds_g1.get("status") in ("NO FINITE SEEDS", "NO PAIRED SEEDS"):
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "5. OTHER / confounded",
            "fired_by": f"DS-G1: {ds_g1['status']}",
        }

    ds_g1_pass = bool(ds_g1["pass"])
    ds_g3_restored = bool(ds_g3["restored"])
    mean = ds_g1["mean"]

    if ds_g1_pass and ds_g3_restored:
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "1. DS-G1 PASS + DS-G3 RESTORED (M-OWNED-CLOSED candidate)",
            "fired_by": f"excess(2D-1D)={mean:+.1f} in band, 2D coverage restored",
        }
    if ds_g1_pass and not ds_g3_restored:
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "2. DS-G1 PASS + DS-G3 NOT restored (width channel leads)",
            "fired_by": f"excess(2D-1D)={mean:+.1f} in band, 2D coverage NOT restored",
        }
    if mean > DS_G1_HI:
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "3. DS-G1 FAIL high (under-cancelled)",
            "fired_by": f"excess(2D-1D)={mean:+.1f} > {DS_G1_HI}",
        }
    if mean < DS_G1_LO:
        return {
            "status": "PRESENTED, NOT ADJUDICATED — the author rules",
            "branch": "4. DS-G1 FAIL low (over-corrected)",
            "fired_by": f"excess(2D-1D)={mean:+.1f} < {DS_G1_LO}",
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
        help="AFULL2D arm result JSON (default: AFULL2D_h0p730_results_seeds0_25.json); "
        "pass the committed AFULL file for the prereg §6 item 2 mechanics dry-run",
    )
    ap.add_argument(
        "--c1-bit-identity-max-diff",
        type=float,
        default=None,
        help="DS-G4 max |ln1_gsel - ln1_afull| from the separate post-run check (optional)",
    )
    ap.add_argument("--out", type=Path, default=HERE / "score_afull2d_output.json")
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

    ds_g1: dict[str, Any] | None = None
    ds_g2: dict[str, Any] | None = None
    ds_g3: dict[str, Any] | None = None
    ds_g4: dict[str, Any] | None = None
    ds_g5: dict[str, Any] | None = None

    if input_present:
        d = load(args.input)
        grid = np.array(d["config"]["h_grid"], dtype=np.float64)
        recs = d["per_seed"]
        ch1d = score_channel(grid, recs, "1d")
        ch2d = score_channel(grid, recs, "2d")

        ds_g1 = score_ds_g1(ch1d, ch2d)
        out["ds_g1"] = ds_g1
        P("=== DS-G1: paired 2D-1D excess at truth (PRIMARY, branch-carrying) ===")
        if "mean" in ds_g1:
            P(
                f"  excess: mean={ds_g1['mean']:+.1f} SE={ds_g1['se']:.2f} sd={ds_g1['sd']:.1f} "
                f"N={ds_g1['n']} band=[{DS_G1_LO:+.1f},{DS_G1_HI:+.1f}] "
                f"-> {'PASS' if ds_g1['pass'] else 'FAIL'}"
            )
            P(
                f"  mirror reference: {DS_G1_MIRROR_REFERENCE['mean']:+.2f} +/- "
                f"{DS_G1_MIRROR_REFERENCE['se']:.2f} (N={DS_G1_MIRROR_REFERENCE['n_mirror']})"
            )
        else:
            P(f"  {ds_g1.get('status', 'UNKNOWN')}")
        P()

        ds_g2 = score_ds_g2(ch1d, ch2d)
        out["ds_g2"] = ds_g2
        P("=== DS-G2: tilts at truth (secondary, non-branch-carrying) ===")
        for key in ("1d", "2d"):
            c = ds_g2[key]
            if "mean" in c:
                lo, hi = ds_g2[f"band_{key}"]
                p = ds_g2[f"pass_{key}"]
                P(
                    f"  {key}: T_mean={c['mean']:+.1f} SE={c['se']:.1f} N={c['n']} "
                    f"band=[{lo:+.1f},{hi:+.1f}] -> {'PASS' if p else 'FAIL'}"
                )
            else:
                P(f"  {key}: NO FINITE SEEDS")
        P()

        ds_g3 = score_ds_g3(ch2d)
        out["ds_g3"] = ds_g3
        P("=== DS-G3: 2D-channel coverage restoration ===")
        c = ds_g3["2d"]
        if "hpd50" in c:
            P(f"  2D: hpd50/68/90={c['hpd50']:.3f}/{c['hpd68']:.3f}/{c['hpd90']:.3f} N={c['n']}")
        else:
            P("  2D: NO FINITE SEEDS")
        P(f"  RESTORED: {ds_g3['restored']}")
        P()

        ds_g4 = score_ds_g4(args.c1_bit_identity_max_diff)
        out["ds_g4"] = ds_g4
        P("=== DS-G4: 1D invariance (c1 bit-identity to a_full) ===")
        P(f"  {ds_g4['status']}")
        if ds_g4["max_diff"] is not None:
            P(f"  max_diff={ds_g4['max_diff']:.3e} -> {'PASS' if ds_g4['pass'] else 'STOP'}")
        P()

        ds_g5 = score_ds_g5(ch1d, ch2d)
        out["ds_g5"] = ds_g5
        P("=== DS-G5: per-seed T scatter, MAP bias, rails, non-finite (descriptive) ===")
        for key in ("1d", "2d"):
            c = ds_g5[key]
            if "T_min" in c:
                P(
                    f"  {key}: T in [{c['T_min']:+.1f},{c['T_max']:+.1f}] sd={c['T_sd']:.1f} "
                    f"bias={c['bias_mean']:+.4f}+/-{c['bias_se']:.4f} "
                    f"railed_low={c['railed_low_n']} railed_high={c['railed_high_n']} "
                    f"nonfinite={c['nonfinite_n']}"
                )
            else:
                P(f"  {key}: {c['status']}")
        if ds_g5["stop"]:
            P("  *** STOP: rail or non-finite event detected (prereg §6 item 4) ***")
        P()
    else:
        out["ds_g1"] = None
        out["ds_g2"] = None
        out["ds_g3"] = None
        out["ds_g4"] = None
        out["ds_g5"] = None

    branch = determine_branch(input_present, ds_g1, ds_g3, ds_g5)
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
