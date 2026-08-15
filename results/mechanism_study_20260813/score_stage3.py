#!/usr/bin/env python
"""Independent stage-3 readout scorer for A-JREN / A-REN (PREREGISTRATION_A_JREN_STAGE3.md).

Written and committed BEFORE the arm data exists (analysis-code-freeze discipline, thread 17,
mirroring ``score_m2prime_stage2.py``'s own discipline for stage 2). Recomputes every statistic
from the raw per-seed ``ln_post_1d`` / ``ln_post_2d`` vectors of an arbitrary stage-3 arm result
file; the stored per-seed scalars and the ``aggregate`` block are never used as inputs, only as
optional cross-checks.

Generalization over ``score_m2prime_stage2.py``: this scorer takes ONE arm file at a time (not a
fixed AM2P/ANULL pair) via ``--arm``/``--which``, so it applies to A-JREN (this stage's primary
arm, run first per the F1 ordering change) OR the conditional A-REN, without a code fork. DS-M1 is
identical in form to the stage-2 registration (verbatim edges, same ``b_ref``); DS-J1 (A-JREN only)
adds the coverage-restoration check (HPD90 >= 0.60 both channels) on top of DS-M1's classes. The
two F2 expectation windows (TBD-filled in the registration finalization block) are reported as
WEAK, non-branch-carrying context — never as a branch input, per the commission D1-03 bar carried
into this stage's prereg front matter.

Inputs (paths, all optional except MN0X which must be present):
    --arm    <ARM>_h0p730_results_seeds0_25.json  (A-JREN or A-REN, DS-M1 [+ DS-J1 if A-JREN])
    --which  ajren | aren                          (selects the F2 window and DS-J1 applicability)
    --mn0x   MN0X_h0p730_results_seeds0_100.json   (committed paired reference for b_ref)

The arm file need not exist at commit time: this script fails cleanly (reports the file as
MISSING) rather than raising, and the branch is withheld until the arm is present (§5
execution-completeness clause, carried).

Emits ``score_stage3_output.json`` next to this file and a compact stdout table.
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

# ---- registered pins (PREREGISTRATION_A_JREN_STAGE3.md §4, carried verbatim
# from PREREGISTRATION_M2PRIME_ABLATION.md §4) ------------------------------
B_REF = 0.037250  # MN0X, N=100, committed (§4 DS-M1 TERM-INNOCENT anchor)
IN_BAND = 0.010  # DS-M1 TERM-OWNS/TERM-PARTIAL edge
DEFECT = 0.030  # DS-M1 TERM-PARTIAL/TERM-INNOCENT edge
NULL_TOL = 0.004  # DS-M1 TERM-INNOCENT proximity-to-b_ref edge
HPD90_OWNS = 0.60  # DS-M1 TERM-OWNS coverage conjunct; also the DS-J1 coverage-restoration bar

# ---- F2 expectation windows (registration finalization block, WEAK,
# non-branch-carrying, two-sided) -------------------------------------------
F2_WINDOWS: dict[str, dict[str, float]] = {
    "aren": {"center": 0.0354, "half_width": 0.006},
    "ajren": {"center": 0.0173, "half_width": 0.012},
}

DEFAULT_MN0X = HERE / "MN0X_h0p730_results_seeds0_100.json"


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


# ── recomputation kernel (ports of score_m2prime_stage2.py's, verbatim) ────
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
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(h_grid))])
    pit = float(np.interp(h_true, h_grid, cum))
    mean = float(np.trapezoid(post * h_grid, h_grid))
    var = float(np.trapezoid(post * h_grid**2, h_grid)) - mean**2
    r = {"pit": pit, "post_sd": math.sqrt(max(var, 0.0))}
    for lv in HPD_LEVELS:
        r[f"hpd{int(round(lv * 100))}"] = float(hpd_contains(h_grid, post, h_true, lv))
    return r


def ks_distance(pits: list[float]) -> float:
    q = np.sort(np.asarray(pits, dtype=np.float64))
    n = q.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(np.max(np.maximum(i / n - q, q - (i - 1.0) / n)))


def ks_critical(n: int) -> dict[str, float]:
    """Asymptotic Kolmogorov critical values D_alpha = c(alpha)/sqrt(n)."""
    if n <= 0:
        return {"D_95": float("nan"), "D_99": float("nan")}
    return {"D_95": 1.36 / math.sqrt(n), "D_99": 1.63 / math.sqrt(n)}


def classify(bias: float, hpd90: float, b_ref: float) -> str:
    if abs(bias) <= IN_BAND and hpd90 >= HPD90_OWNS:
        return "TERM-OWNS"
    if IN_BAND < abs(bias) < DEFECT:
        return "TERM-PARTIAL"
    if abs(bias) >= DEFECT and abs(bias - b_ref) <= NULL_TOL:
        return "TERM-INNOCENT"
    return "OTHER"


# ── DS-M1: one arm file, both channels ──────────────────────────────────────
def score_ds_m1(d: dict[str, Any]) -> dict[str, Any]:
    """DS-M1 (headline, verbatim from the stage-2 registration, §4).

    Args:
        d: A loaded stage-3 (or MN0X) result JSON with ``config.h_grid`` and
            ``per_seed`` records carrying ``ln_post_1d``/``ln_post_2d``.

    Returns:
        Per-channel bias/coverage/classification dict, keyed ``"1d"``/``"2d"``.
    """
    grid = np.array(d["config"]["h_grid"], dtype=np.float64)
    recs = d["per_seed"]
    out: dict[str, Any] = {"n_records": len(recs), "seeds": sorted(r["seed"] for r in recs)}
    for ch in ("1d", "2d"):
        maps: list[float] = []
        pits: list[float] = []
        sds: list[float] = []
        rl: list[float] = []
        rh: list[float] = []
        hpd: dict[int, list[float]] = {50: [], 68: [], 90: []}
        nonfinite = 0
        for r in recs:
            ln = np.array(r[f"ln_post_{ch}"], dtype=np.float64)
            if not np.all(np.isfinite(ln)):
                nonfinite += 1
                continue
            pr = posterior_readout(grid, ln)
            pp = pp_readout(grid, ln, H_TRUE)
            maps.append(pr["map"])
            rl.append(pr["railed_low"])
            rh.append(pr["railed_high"])
            pits.append(pp["pit"])
            sds.append(pp["post_sd"])
            for lv in (50, 68, 90):
                hpd[lv].append(pp[f"hpd{lv}"])
        n = len(maps)
        if n == 0:
            out[ch] = {"n": 0, "status": "NO FINITE SEEDS"}
            continue
        bias_list = [m - H_TRUE for m in maps]
        mean_bias = sum(bias_list) / n
        sd = st.stdev(bias_list) if n > 1 and len(set(bias_list)) > 1 else 0.0
        se = sd / math.sqrt(n)
        hpd90_cov = sum(hpd[90]) / n
        ks = ks_distance(pits)
        crit = ks_critical(n)
        out[ch] = {
            "n": n,
            "nonfinite": nonfinite,
            "mean_bias": mean_bias,
            "se": se,
            "sd": sd,
            "post_sd_median": st.median(sds),
            "hpd50_cov": sum(hpd[50]) / n,
            "hpd68_cov": sum(hpd[68]) / n,
            "hpd90_cov": hpd90_cov,
            "pit_ks_D": ks,
            "pit_ks_D95": crit["D_95"],
            "pit_ks_D99": crit["D_99"],
            "pit_ks_status": "FAIL" if ks > crit["D_95"] else "PASS",
            "railed_low_frac": sum(rl) / n,
            "railed_high_frac": sum(rh) / n,
            "class": classify(mean_bias, hpd90_cov, B_REF),
        }
    return out


# ── DS-J1: coverage-restoration check (A-JREN only) ─────────────────────────
def score_ds_j1(ds_m1: dict[str, Any]) -> dict[str, Any]:
    """DS-J1 (§4): DS-M1 applied to the joint arm PLUS a coverage-restoration read.

    Registered purpose (prereg §4): does the joint arm's HPD90 return to
    ``>= 0.60`` in BOTH channels — the *interval*, not just the *point*,
    calibrating — since A-JREN's registered role is testing whether the
    located terms jointly restore calibration, not merely reduce ``|b|``.

    Args:
        ds_m1: The output of :func:`score_ds_m1` for the A-JREN arm.

    Returns:
        Per-channel and combined coverage-restoration verdicts.
    """
    out: dict[str, Any] = {}
    both_restored = True
    for ch in ("1d", "2d"):
        c = ds_m1.get(ch, {})
        hpd90 = c.get("hpd90_cov")
        restored = bool(hpd90 is not None and hpd90 >= HPD90_OWNS)
        out[ch] = {"hpd90_cov": hpd90, "coverage_restored": restored}
        both_restored = both_restored and restored
    out["coverage_restored_both_channels"] = both_restored
    return out


# ── F2 expectation window comparison (WEAK, non-branch-carrying, reported) ──
def score_f2_window(which: str, ds_m1: dict[str, Any]) -> dict[str, Any]:
    """Compare the measured 1D bias against the registered F2 expectation window.

    Registration finalization block F2 (2026-08-15): the window's sole
    purpose is legibility of surprise — "the branch reads DS-M1 classes
    only" (prereg §4) — so this is reported, never adjudicated as a branch
    input, per the commission D1-03 bar carried into this stage's prereg
    front matter.

    Args:
        which: ``"aren"`` or ``"ajren"`` — selects the registered window.
        ds_m1: The output of :func:`score_ds_m1` for the arm in question.

    Returns:
        The window, the measured bias, and a WEAK, non-branch-carrying read
        (``"BELOW"``/``"INSIDE"``/``"ABOVE"``/``"NO FINITE SEEDS"``).
    """
    win = F2_WINDOWS[which]
    lo = win["center"] - win["half_width"]
    hi = win["center"] + win["half_width"]
    c = ds_m1.get("1d", {})
    bias = c.get("mean_bias")
    if bias is None:
        return {
            "which": which,
            "window_center": win["center"],
            "window_half_width": win["half_width"],
            "window_lo": lo,
            "window_hi": hi,
            "measured_1d_bias": None,
            "read": "NO FINITE SEEDS",
            "status": "WEAK, non-branch-carrying",
        }
    if bias < lo:
        read = "BELOW"
    elif bias > hi:
        read = "ABOVE"
    else:
        read = "INSIDE"
    return {
        "which": which,
        "window_center": win["center"],
        "window_half_width": win["half_width"],
        "window_lo": lo,
        "window_hi": hi,
        "measured_1d_bias": bias,
        "read": read,
        "status": "WEAK, non-branch-carrying",
    }


# ── branch determination (§5, split-precedence + execution-completeness) ───
def determine_branch(
    which: str,
    arm_present: bool,
    ds_m1: dict[str, Any] | None,
    ds_j1: dict[str, Any] | None,
    validity_ok: bool,
) -> dict[str, Any]:
    """The §5 branch call for one arm, carried verbatim from the stage-2 form.

    Args:
        which: ``"aren"`` or ``"ajren"``.
        arm_present: Whether the arm's result file was found.
        ds_m1: :func:`score_ds_m1` output for the arm, or ``None`` if absent.
        ds_j1: :func:`score_ds_j1` output (A-JREN only), or ``None``.
        validity_ok: The §6 validity gate (MN0X cross-check exact match).

    Returns:
        The branch verdict dict (never self-adjudicated — presented raw).
    """
    if not arm_present:
        return {
            "status": "NOT PRESENTED",
            "reason": f"execution-completeness clause (§5): missing arm: {which.upper()}",
        }

    assert ds_m1 is not None

    if not validity_ok:
        return {
            "status": "PRESENTED, NOT ADJUDICATED",
            "branch": "1. STUDY-CONFOUNDED",
            "fired_by": "a §6 validity check failed (MN0X cross-check)",
        }

    cls_1d = ds_m1.get("1d", {}).get("class")
    cls_2d = ds_m1.get("2d", {}).get("class")

    if cls_1d != cls_2d:
        return {
            "status": "PRESENTED, NOT ADJUDICATED",
            "branch": "5. OTHER / SPLIT",
            "fired_by": f"split-precedence: 1D class {cls_1d} != 2D class {cls_2d}",
        }

    if which == "aren":
        branch_map = {
            "TERM-OWNS": "2. REN-OWNS",
            "TERM-PARTIAL": "3. REN-PARTIAL",
            "TERM-INNOCENT": "4. REN-INNOCENT",
        }
        return {
            "status": "PRESENTED, NOT ADJUDICATED",
            "branch": branch_map.get(cls_1d, "5. OTHER / SPLIT"),
            "fired_by": f"A-REN class {cls_1d} (both channels)",
        }

    # A-JREN: diagnostic role only (prereg §4/§7) -- no point-prediction
    # branch of its own; reports the DS-M1 class alongside the DS-J1
    # coverage-restoration flag rather than routing through the REN-* map.
    restored = ds_j1["coverage_restored_both_channels"] if ds_j1 is not None else None
    return {
        "status": "PRESENTED, NOT ADJUDICATED",
        "branch": "diagnostic (no A-JREN-only branch; informs REN-PARTIAL's completeness gate, §5)",
        "fired_by": f"A-JREN class {cls_1d} (both channels), coverage_restored_both_channels={restored}",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", type=Path, required=True, help="stage-3 arm result JSON")
    ap.add_argument(
        "--which",
        choices=("ajren", "aren"),
        required=True,
        help="which registered arm --arm is (selects the F2 window and DS-J1 applicability)",
    )
    ap.add_argument("--mn0x", type=Path, default=DEFAULT_MN0X)
    ap.add_argument("--out", type=Path, default=HERE / "score_stage3_output.json")
    args = ap.parse_args(argv)

    P = print
    out: dict[str, Any] = {}

    if not args.mn0x.exists():
        P(f"FATAL: committed reference MN0X file not found: {args.mn0x}")
        return 2
    d_mn0x = load(args.mn0x)

    arm_present = args.arm.exists()
    out["inputs"] = {
        "arm_path": str(args.arm),
        "arm_present": arm_present,
        "which": args.which,
        "mn0x_path": str(args.mn0x),
        "mn0x_present": True,
    }

    P("=== INPUTS ===")
    P(
        f"  {args.which.upper()} {args.arm.name}: "
        f"{'FOUND' if arm_present else 'MISSING (analysis-code-freeze: expected before data exists)'}"
    )
    P(f"  MN0X {args.mn0x.name}: FOUND (committed paired reference)")
    P()

    # -- MN0X cross-check only (never trusted as an input to any class edge) --
    ds_m1_mn0x = score_ds_m1(d_mn0x)
    b_mn0x = ds_m1_mn0x["1d"]["mean_bias"]
    P(
        "=== MN0X CROSS-CHECK (recomputed vs registered b_ref=+0.037250; not an input to any class) ==="
    )
    P(f"  recomputed 1D bias = {b_mn0x:+.6f}  |delta from b_ref| = {abs(b_mn0x - B_REF):.2e}")
    out["mn0x_cross_check"] = ds_m1_mn0x

    ds_m1_arm: dict[str, Any] | None = None
    ds_j1: dict[str, Any] | None = None
    f2: dict[str, Any] | None = None

    if arm_present:
        d_arm = load(args.arm)
        ds_m1_arm = score_ds_m1(d_arm)
        out["ds_m1"] = ds_m1_arm
        P(f"=== DS-M1: {args.which.upper()} ===")
        for ch in ("1d", "2d"):
            c = ds_m1_arm[ch]
            if c.get("n", 0) == 0:
                P(f"  {ch}: NO FINITE SEEDS")
                continue
            P(
                f"  {ch}: N={c['n']} bias={c['mean_bias']:+.6f} SE={c['se']:.6f} "
                f"hpd50/68/90={c['hpd50_cov']:.3f}/{c['hpd68_cov']:.3f}/{c['hpd90_cov']:.3f} "
                f"KS_D={c['pit_ks_D']:.4f} ({c['pit_ks_status']}) post_sd_med={c['post_sd_median']:.6f} "
                f"rails={c['railed_low_frac'] + c['railed_high_frac']:.3f} nonfin={c['nonfinite']} "
                f"-> {c['class']}"
            )
        P()

        f2 = score_f2_window(args.which, ds_m1_arm)
        out["f2_window"] = f2
        P("=== F2 expectation window (WEAK, non-branch-carrying) ===")
        P(
            f"  window: [{f2['window_lo']:+.4f}, {f2['window_hi']:+.4f}] "
            f"(center {f2['window_center']:+.4f} +/- {f2['window_half_width']:.4f})"
        )
        P(f"  measured 1D bias: {f2['measured_1d_bias']}  -> {f2['read']}")
        P()

        if args.which == "ajren":
            ds_j1 = score_ds_j1(ds_m1_arm)
            out["ds_j1"] = ds_j1
            P("=== DS-J1: coverage-restoration check (A-JREN only) ===")
            for ch in ("1d", "2d"):
                r = ds_j1[ch]
                P(f"  {ch}: hpd90_cov={r['hpd90_cov']} -> restored={r['coverage_restored']}")
            P(f"  both channels restored: {ds_j1['coverage_restored_both_channels']}")
            P()
        else:
            out["ds_j1"] = None  # DS-J1 is A-JREN-only (prereg §4)
    else:
        out["ds_m1"] = None
        out["f2_window"] = None
        out["ds_j1"] = None

    validity_ok = True
    if ds_m1_mn0x["1d"].get("n", 0) > 0:
        validity_ok = abs(b_mn0x - B_REF) <= 1e-9  # exact-record cross-check on committed data
    out["validity_ok"] = validity_ok

    branch = determine_branch(args.which, arm_present, ds_m1_arm, ds_j1, validity_ok)
    out["branch"] = branch
    P("=== BRANCH (§5; PRESENTED, NOT ADJUDICATED — author call only) ===")
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
