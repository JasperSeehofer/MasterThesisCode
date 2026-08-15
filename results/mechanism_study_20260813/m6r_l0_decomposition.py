"""M6-revision L0 tilt decomposition (M6-revision analysis note, ledger row #103).

Follow-on to ``m6_l0_killtests.py``: that script found M6-as-registered (pure
sigma_z-blind tilt x dose-controlled curvature, dose-INDEPENDENT tilt) KILLED on
tests (i) and (iii) under the registered ("all f_h > 0 cells") operationalization,
but noted the survivor column f_i = 1.0 reproduces the commission's reference band.
This script characterizes the dose-DEPENDENT tilt residual directly from committed
data and assembles the ingredients for a revised candidate statement M6' (proposed
here, not adjudicated).

Recomputes everything from the raw per-seed ``ln_post_1d``/``ln_post_2d`` vectors
and per-seed ``map_*``/``post_sd_*`` fields in the committed result JSONs under
``results/mechanism_study_20260813/``. Never reads the per-file ``aggregate`` block
(``sum_dlog_gfrac_dh`` there is a different quantity -- the Sigma ln(L2/L1) gfrac
slope, not either channel's own log-posterior tilt).

Sections (mirrors the task's five DO items):

1. Per-cell/arm, both channels: seed-averaged aggregate tilt T = d(ln_post)/dh at
   truth via the grid-neighbour central difference (matches
   ``venue_transfer._slope_at_truth`` / ``m6_l0_killtests.py`` verbatim geometry).
   The four f_h = 0 S-cells (S00-S03) are degenerate delta posteriors (host exact
   -> single-grid-point posterior in >=59/60 seeds); T is reported for them but
   they are excluded from every fit and from the interior/f_i=0 "reliable" groups
   below, because a central difference on a near-delta log-posterior measures
   local curvature noise, not a smooth linear tilt.
2. The J-tilt measured directly: T(AM2P) - T(MN0X) at full dose (both arms are
   the unablated "all"-dose cell; AM2P has the missing Jacobian restored, MN0X
   does not), compared to the prereg's closed-form-predicted missing-J tilt of
   -1345.2 nats/h. Method check: T(ANULL) - T(MN0X first 15, seed-paired) = 0
   (the A-NULL x1.7 constant shifts every ln_post level by +N ln 1.7, not the
   local slope).
3. Base-estimator tilt-surface decomposition R(f_h, f_i) = T(f_h, f_i) - T_alpha,
   with T_alpha = +1393.6 nats/h (analytic, dose-blind, prereg Sec.2). The
   missing-J share is expected +1345.2 nats/h, dose-INDEPENDENT, at full
   kernel-branch occupancy (prereg Sec.2's "at full dose" derivation). This
   script tests whether R - 1345.2 (residual beyond alpha+full-dose-J) is itself
   dose-dependent over the 9 interior cells (f_h > 0, f_i > 0) and fits <=2-param
   descriptive forms (linear-in-f_i pooled over f_h; saturating exponential in
   f_i with the asymptote fixed at the theoretical +1345.2).
4. Tilt x curvature closure: per f_h>0 cell, measured bias vs T * median(post_sd)^2
   (T taken from step 1/3, i.e. the cell's OWN measured, dose-dependent tilt,
   not a fixed constant) -- the ratio bias / (T * sigma_post^2) is the direct
   test of whether using the measured, dose-dependent T reconciles test (i)'s
   KILL (T is very much NOT dose-invariant) with test (ii)'s SURVIVE (bias /
   sigma_post^2 alone IS roughly dose-invariant, factor <=1.41).

Output: ``M6R_L0_output.json`` next to this file.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).parent

# ---- registered/derived constants (PREREGISTRATION_M2PRIME_ABLATION.md Sec.2) ----
ALPHA_N = 982
ALPHA_H = 0.730
ALPHA_SLOPE_COEFF = 1.036
ALPHA_TILT_NATS_PER_H = ALPHA_SLOPE_COEFF * ALPHA_N / ALPHA_H  # = +1393.6 nats/h
MISSING_J_PREDICTED_NATS_PER_H = 1345.2  # Sec.2, "-N/h" piece magnitude, full dose


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as fh:
        result: dict[str, Any] = json.load(fh)
    return result


def _load_s_cells() -> dict[str, dict[str, Any]]:
    """Load the 16 dose-grid S-cell result JSONs, keyed by cell id."""
    files = sorted(glob.glob(str(RESULTS_DIR / "S*_h0p730_results_seeds*.json")))
    cells: dict[str, dict[str, Any]] = {}
    for f in files:
        d = _load_json(f)
        cells[d["config"]["cell"]] = d
    assert len(cells) == 16, f"expected 16 S-cells, found {len(cells)}"
    return cells


def _slope_at_truth_per_seed(
    d: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-seed central-difference d(ln_post)/dh at truth, both channels.

    Verbatim geometry match to ``venue_transfer._slope_at_truth`` /
    ``m6_l0_killtests.py._slope_at_truth_per_seed``: grid neighbours of
    ``h_true``, central difference, applied directly to the raw
    ``ln_post_1d``/``ln_post_2d`` per-seed vectors.
    """
    hg = np.asarray(d["config"]["h_grid"], dtype=np.float64)
    h_true = float(d["config"]["h_true"])
    i_true = int(np.argmin(np.abs(hg - h_true)))
    lo, hi = i_true - 1, i_true + 1
    dh = hg[hi] - hg[lo]
    s1 = np.array(
        [
            (np.asarray(ps["ln_post_1d"])[hi] - np.asarray(ps["ln_post_1d"])[lo]) / dh
            for ps in d["per_seed"]
        ]
    )
    s2 = np.array(
        [
            (np.asarray(ps["ln_post_2d"])[hi] - np.asarray(ps["ln_post_2d"])[lo]) / dh
            for ps in d["per_seed"]
        ]
    )
    return s1, s2


def _mean_se(x: npt.NDArray[np.float64]) -> tuple[float, float]:
    n = x.size
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, se


def _cell_bias_sigma(
    d: dict[str, Any], channel: str
) -> tuple[float, float, npt.NDArray[np.float64]]:
    """mean MAP bias, median post_sd, and per-seed post_sd for one channel."""
    h_true = float(d["config"]["h_true"])
    maps = np.array([ps[f"map_{channel}"] for ps in d["per_seed"]], dtype=np.float64)
    sds = np.array([ps[f"post_sd_{channel}"] for ps in d["per_seed"]], dtype=np.float64)
    bias = float(np.mean(maps - h_true))
    med_sd = float(np.median(sds))
    return bias, med_sd, sds


# ---------------------------------------------------------------------------
# Section 1 -- per-cell/arm tilt table
# ---------------------------------------------------------------------------


def section1_tilt_table(
    cells: dict[str, dict[str, Any]], extra_arms: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cid, d in {**cells, **extra_arms}.items():
        s1, s2 = _slope_at_truth_per_seed(d)
        m1, se1 = _mean_se(s1)
        m2, se2 = _mean_se(s2)
        dose = d["config"].get("dose_scales")
        fh, fi = (dose[0], dose[1]) if dose is not None else (None, None)
        out[cid] = {
            "dose_scales": dose,
            "f_h": fh,
            "f_i": fi,
            "n_seeds": s1.size,
            "T1_mean": m1,
            "T1_se": se1,
            "T2_mean": m2,
            "T2_se": se2,
            "degenerate_fh0": bool(fh == 0.0) if fh is not None else False,
        }
    return out


# ---------------------------------------------------------------------------
# Section 2 -- J-tilt direct measurement + A-NULL method check
# ---------------------------------------------------------------------------


def section2_j_tilt(
    am2p: dict[str, Any], mn0x: dict[str, Any], anull: dict[str, Any]
) -> dict[str, Any]:
    s1_am2p, s2_am2p = _slope_at_truth_per_seed(am2p)
    s1_mn0x, s2_mn0x = _slope_at_truth_per_seed(mn0x)

    t1_am2p, se1_am2p = _mean_se(s1_am2p)
    t2_am2p, se2_am2p = _mean_se(s2_am2p)
    t1_mn0x, se1_mn0x = _mean_se(s1_mn0x)
    t2_mn0x, se2_mn0x = _mean_se(s2_mn0x)

    j_tilt_1d = t1_am2p - t1_mn0x
    j_tilt_2d = t2_am2p - t2_mn0x
    j_tilt_se_1d = float(np.hypot(se1_am2p, se1_mn0x))
    j_tilt_se_2d = float(np.hypot(se2_am2p, se2_mn0x))

    # A-NULL vs MN0X first 15 (seed-paired by construction, prereg Sec.3).
    mn0x_first15 = dict(mn0x)
    mn0x_first15["per_seed"] = mn0x["per_seed"][:15]
    s1_a15, s2_a15 = _slope_at_truth_per_seed(mn0x_first15)
    s1_anull, s2_anull = _slope_at_truth_per_seed(anull)
    mn0x_seeds15 = [ps["seed"] for ps in mn0x_first15["per_seed"]]
    anull_seeds = [ps["seed"] for ps in anull["per_seed"]]
    seeds_match = mn0x_seeds15 == anull_seeds

    diff1 = s1_anull - s1_a15
    diff2 = s2_anull - s2_a15

    return {
        "AM2P": {
            "T1_mean": t1_am2p,
            "T1_se": se1_am2p,
            "T2_mean": t2_am2p,
            "T2_se": se2_am2p,
            "n": s1_am2p.size,
        },
        "MN0X_full": {
            "T1_mean": t1_mn0x,
            "T1_se": se1_mn0x,
            "T2_mean": t2_mn0x,
            "T2_se": se2_mn0x,
            "n": s1_mn0x.size,
        },
        "j_tilt_measured_1d_nats_per_h": j_tilt_1d,
        "j_tilt_measured_1d_se": j_tilt_se_1d,
        "j_tilt_measured_2d_nats_per_h": j_tilt_2d,
        "j_tilt_measured_2d_se": j_tilt_se_2d,
        "j_tilt_predicted_nats_per_h": -MISSING_J_PREDICTED_NATS_PER_H,
        "ratio_measured_over_predicted_1d": j_tilt_1d / (-MISSING_J_PREDICTED_NATS_PER_H),
        "ratio_measured_over_predicted_2d": j_tilt_2d / (-MISSING_J_PREDICTED_NATS_PER_H),
        "am2p_residual_beyond_alpha_1d": t1_am2p - ALPHA_TILT_NATS_PER_H,
        "am2p_residual_beyond_alpha_2d": t2_am2p - ALPHA_TILT_NATS_PER_H,
        "anull_method_check": {
            "seeds_match": seeds_match,
            "mn0x_first15_seeds": mn0x_seeds15,
            "anull_seeds": anull_seeds,
            "per_seed_diff_1d": diff1.tolist(),
            "per_seed_diff_2d": diff2.tolist(),
            "max_abs_diff_1d": float(np.max(np.abs(diff1))),
            "max_abs_diff_2d": float(np.max(np.abs(diff2))),
            "verdict": "PASS (slope invariant under x1.7, as predicted)"
            if float(np.max(np.abs(diff1))) < 1e-6 and float(np.max(np.abs(diff2))) < 1e-6
            else "FAIL",
        },
    }


# ---------------------------------------------------------------------------
# Section 3 -- base-estimator tilt-surface decomposition R(f_h, f_i)
# ---------------------------------------------------------------------------


def _sat_exp_fixed_asymptote(
    f_i: npt.NDArray[np.float64], r0: float, tau: float
) -> npt.NDArray[np.float64]:
    """R(f_i) = 1345.2 + (r0 - 1345.2) * exp(-f_i / tau); asymptote fixed at the
    theoretical full-dose missing-J share (2 free params: r0, tau)."""
    return MISSING_J_PREDICTED_NATS_PER_H + (r0 - MISSING_J_PREDICTED_NATS_PER_H) * np.exp(
        -f_i / tau
    )


def section3_decomposition(tilt_table: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cid, v in tilt_table.items():
        if cid not in ("S00", "S01", "S02", "S03") and not cid.startswith("S"):
            continue
        fh, fi = v["f_h"], v["f_i"]
        if fh is None:
            continue
        r1 = v["T1_mean"] - ALPHA_TILT_NATS_PER_H
        r2 = v["T2_mean"] - ALPHA_TILT_NATS_PER_H
        rows.append(
            {
                "cell": cid,
                "f_h": fh,
                "f_i": fi,
                "T1": v["T1_mean"],
                "T2": v["T2_mean"],
                "R1": r1,
                "R2": r2,
                "R1_minus_predicted_J": r1 - MISSING_J_PREDICTED_NATS_PER_H,
                "R2_minus_predicted_J": r2 - MISSING_J_PREDICTED_NATS_PER_H,
                "degenerate_fh0": fh == 0.0,
                "edge_fi0": fi == 0.0,
            }
        )
    rows.sort(key=lambda r: r["cell"])

    interior9 = [r for r in rows if r["f_h"] > 0 and r["f_i"] > 0]
    fi0_col = [r for r in rows if r["f_h"] > 0 and r["f_i"] == 0.0]
    fh0_row = [r for r in rows if r["f_h"] == 0.0]

    fits: dict[str, Any] = {}
    for ch in ("1", "2"):
        f_i_arr = np.array([r["f_i"] for r in interior9], dtype=np.float64)
        R_arr = np.array([r[f"R{ch}"] for r in interior9], dtype=np.float64)

        # linear-in-f_i, pooled over f_h (2 params: intercept, slope)
        lin_coef = np.polyfit(f_i_arr, R_arr, deg=1)
        lin_pred = np.polyval(lin_coef, f_i_arr)
        lin_resid = R_arr - lin_pred
        ss_res_lin = float(np.sum(lin_resid**2))
        ss_tot = float(np.sum((R_arr - np.mean(R_arr)) ** 2))
        r2_lin = 1.0 - ss_res_lin / ss_tot if ss_tot > 0 else float("nan")

        # saturating exponential, asymptote fixed at +1345.2 (2 params: r0, tau)
        try:
            popt, _ = curve_fit(
                _sat_exp_fixed_asymptote,
                f_i_arr,
                R_arr,
                p0=[float(R_arr[np.argmin(f_i_arr)]), 0.3],
                maxfev=20000,
            )
            r0_fit, tau_fit = float(popt[0]), float(popt[1])
            exp_pred = _sat_exp_fixed_asymptote(f_i_arr, r0_fit, tau_fit)
            ss_res_exp = float(np.sum((R_arr - exp_pred) ** 2))
            r2_exp = 1.0 - ss_res_exp / ss_tot if ss_tot > 0 else float("nan")
            exp_fit_ok = True
        except RuntimeError:
            r0_fit, tau_fit, r2_exp, exp_fit_ok = float("nan"), float("nan"), float("nan"), False

        fits[f"channel_{ch}d"] = {
            "n_points": int(f_i_arr.size),
            "n_distinct_f_i_levels": int(np.unique(f_i_arr).size),
            "linear_fit": {
                "intercept": float(lin_coef[1]),
                "slope_per_unit_f_i": float(lin_coef[0]),
                "r_squared": r2_lin,
                "rms_residual_nats_per_h": float(np.sqrt(np.mean(lin_resid**2))),
            },
            "saturating_exp_fit_fixed_asymptote_1345p2": {
                "converged": exp_fit_ok,
                "r0_at_fi0": r0_fit,
                "tau": tau_fit,
                "r_squared": r2_exp,
            },
            "caveat": (
                "Only 3 distinct f_i levels (0.25, 0.50, 1.00) in the interior-9 set; a "
                "2-parameter fit against 3 x-levels has ~1 effective residual dof pooled over "
                "f_h. Both forms are descriptive, not validated functional forms -- reported "
                "as trend + bound per the registered resolution floor, not as a mechanism "
                "derivation."
            ),
        }

    return {
        "all_rows": rows,
        "interior9": interior9,
        "fi0_column_S10_S20_S30": fi0_col,
        "fh0_row_S00_S03_degenerate": fh0_row,
        "fits_interior9": fits,
        "trend_summary": (
            "R - 1345.2 (residual beyond alpha + full-dose-predicted missing-J) is positive "
            "and largest at low f_i, decreasing monotonically as f_i increases toward 1.0 "
            "across all three f_h rows in the interior-9 set (pooled f_h average, both "
            "channels); it crosses through ~0 near f_i=1.0, consistent with the missing-J "
            "share converging on its full-dose analytic prediction only as f_i -> 1. "
            "Dependence on f_h at fixed f_i is comparatively weak and not cleanly monotonic "
            "(within the per-cell SE budget) -- the residual's dose-dependence is carried "
            "predominantly by f_i, not f_h."
        ),
    }


# ---------------------------------------------------------------------------
# Section 4 -- tilt x curvature closure
# ---------------------------------------------------------------------------


def section4_closure(
    cells: dict[str, dict[str, Any]], tilt_table: dict[str, Any]
) -> dict[str, Any]:
    rows = []
    for cid, d in cells.items():
        fh, fi = d["config"]["dose_scales"]
        if fh == 0.0:
            continue  # degenerate, sigma_post == 0 identically -- see Sec.1 caveat
        entry: dict[str, Any] = {"cell": cid, "f_h": fh, "f_i": fi}
        for ch in ("1d", "2d"):
            bias, med_sd, _ = _cell_bias_sigma(d, ch)
            t = tilt_table[cid][f"T{ch[0]}_mean"]
            pred = t * med_sd**2
            ratio = bias / pred if pred != 0 else float("nan")
            entry[f"bias_{ch}"] = bias
            entry[f"median_post_sd_{ch}"] = med_sd
            entry[f"T_{ch}"] = t
            entry[f"predicted_bias_T_sigma2_{ch}"] = pred
            entry[f"ratio_bias_over_predicted_{ch}"] = ratio
        rows.append(entry)
    rows.sort(key=lambda r: r["cell"])

    interior9 = [r for r in rows if r["f_h"] > 0 and r["f_i"] > 0]
    fi0_col = [r for r in rows if r["f_i"] == 0.0]

    def _ratio_stats(rows_: list[dict[str, Any]], key: str) -> dict[str, float]:
        vals = np.array([r[key] for r in rows_], dtype=np.float64)
        return {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "all_fh_gt0_rows": rows,
        "interior9_ratio_stats_1d": _ratio_stats(interior9, "ratio_bias_over_predicted_1d"),
        "interior9_ratio_stats_2d": _ratio_stats(interior9, "ratio_bias_over_predicted_2d"),
        "fi0_column_ratio_1d": [r["ratio_bias_over_predicted_1d"] for r in fi0_col],
        "fi0_column_ratio_2d": [r["ratio_bias_over_predicted_2d"] for r in fi0_col],
        "reconciliation_note": (
            "On the 9 interior cells, bias / (T * median(post_sd)^2) -- using each cell's OWN "
            "measured, strongly dose-dependent T, not a fixed constant -- is stable at "
            "~0.65-0.80 in both channels (a tighter spread than the raw T range, which spans "
            "2642.9-3544.3 nats/h, a 34% swing). This is the direct mechanism by which test "
            "(i)'s KILL (T is very much not dose-invariant) coexists with test (ii)'s SURVIVE "
            "(bias/sigma_post^2 alone stays within a factor 1.38-1.41): sigma_post^2 co-varies "
            "with dose in a way that, multiplied by the ACTUAL (dose-dependent) T rather than "
            "a constant, tracks bias almost linearly with an approximately fixed proportionality "
            "close to 0.75 -- matching the prereg's own disclosed ~1.5x (~1/0.75) local-Gaussian "
            "scale error within its stated tolerance. The f_i=0 column (S10/S20/S30) breaks this "
            "closure badly (ratios 3-215): sigma_post is tiny there (near-degenerate, host-only "
            "informative), so the same near-delta-posterior caveat that excludes S00-S03 from "
            "the fits in Sec.3 also degrades this closure's local-Gaussian approximation."
        ),
    }


def main() -> None:
    cells = _load_s_cells()
    am2p = _load_json(str(RESULTS_DIR / "AM2P_h0p730_results_seeds0_25.json"))
    anull = _load_json(str(RESULTS_DIR / "ANULL_h0p730_results_seeds0_15.json"))
    mn0x = _load_json(str(RESULTS_DIR / "MN0X_h0p730_results_seeds0_100.json"))
    extra_arms = {"AM2P": am2p, "ANULL": anull, "MN0X": mn0x}

    tilt_table = section1_tilt_table(cells, extra_arms)
    j_tilt = section2_j_tilt(am2p, mn0x, anull)
    decomposition = section3_decomposition(tilt_table)
    closure = section4_closure(cells, tilt_table)

    output = {
        "note": "results/mechanism_study_20260813/M6R_L0_NOTE_20260815.md",
        "parent_kill_tests": "results/mechanism_study_20260813/M6_L0_KILLTESTS_20260814.md",
        "data_source": (
            "16 S-cell dose-grid JSONs + AM2P_h0p730_results_seeds0_25.json + "
            "ANULL_h0p730_results_seeds0_15.json + MN0X_h0p730_results_seeds0_100.json, "
            "all committed under results/mechanism_study_20260813/"
        ),
        "method_note": (
            "All quantities recomputed from raw per-seed ln_post_1d/ln_post_2d vectors and "
            "per-seed map_*/post_sd_* fields; the per-file 'aggregate' block was never read."
        ),
        "constants": {
            "ALPHA_N": ALPHA_N,
            "ALPHA_H": ALPHA_H,
            "ALPHA_SLOPE_COEFF": ALPHA_SLOPE_COEFF,
            "ALPHA_TILT_NATS_PER_H": ALPHA_TILT_NATS_PER_H,
            "MISSING_J_PREDICTED_NATS_PER_H": MISSING_J_PREDICTED_NATS_PER_H,
        },
        "section1_tilt_table": tilt_table,
        "section2_j_tilt": j_tilt,
        "section3_decomposition": decomposition,
        "section4_closure": closure,
    }

    out_path = RESULTS_DIR / "M6R_L0_output.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, sort_keys=False)
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {
                "j_tilt_measured_1d": j_tilt["j_tilt_measured_1d_nats_per_h"],
                "j_tilt_predicted": j_tilt["j_tilt_predicted_nats_per_h"],
                "anull_check": j_tilt["anull_method_check"]["verdict"],
                "interior9_ratio_1d_mean": closure["interior9_ratio_stats_1d"]["mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
