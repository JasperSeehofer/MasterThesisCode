"""ADVERSARIAL ADJUDICATION of A1_READOUT.md and SCAN_READOUT.md.

Fully independent re-implementation: no import of score_a1.py or score_2d_scan.py.
Every posterior-derived statistic is recomputed from the raw 41-point ln_post
vectors (grid argmax, trapezoid normalisation, PIT, HPD, edge mass, post_sd).
Read-only on every .md file. Writes only adjudicate_mechanism_study_output.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H_TRUE = 0.730
BASE = 20260808
FRACS = (0.0, 0.25, 0.5, 1.0)
OUT: dict = {}


# ------------------------------------------------------------------ raw layer
def load(stem: str) -> dict:
    return json.loads((HERE / f"{stem}.json").read_text())


def grid(d: dict) -> np.ndarray:
    return np.asarray(d["config"]["h_grid"], dtype=float)


def map_from_lnpost(hg: np.ndarray, lp: np.ndarray) -> float:
    """Grid-argmax MAP, independent implementation."""
    return float(hg[int(np.argmax(lp))])


def map_refined_from_lnpost(hg: np.ndarray, lp: np.ndarray) -> float:
    i = int(np.argmax(lp))
    if i == 0 or i == hg.size - 1:
        return float(hg[i])
    x0, x1, x2 = hg[i - 1], hg[i], hg[i + 1]
    y0, y1, y2 = lp[i - 1], lp[i], lp[i + 1]
    d1 = (y1 - y0) / (x1 - x0)
    d2 = (y2 - y1) / (x2 - x1)
    curv = (d2 - d1) / (0.5 * (x2 - x0))
    if curv >= 0.0:
        return float(hg[i])
    return float(min(max(0.5 * (x0 + x1) - d1 / curv, hg[0]), hg[-1]))


def post_summary(hg: np.ndarray, lp: np.ndarray) -> dict:
    """PIT / HPD50,68,90 / post_sd / edge_mass / posterior mean, from ln_post."""
    if not np.all(np.isfinite(lp)):
        return {k: float("nan") for k in
                ("pit", "hpd50", "hpd68", "hpd90", "post_sd", "edge_mass", "mean")}
    p = np.exp(lp - lp.max())
    dens = p / np.trapezoid(p, hg)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(hg))])
    pit = float(np.interp(H_TRUE, hg, cum))
    mean = float(np.trapezoid(dens * hg, hg))
    var = float(np.trapezoid(dens * hg ** 2, hg)) - mean ** 2
    edge = float(cum[1] + (cum[-1] - cum[-2]))
    res = {"pit": pit, "post_sd": math.sqrt(max(var, 0.0)), "edge_mass": edge,
           "mean": mean}
    dh = np.gradient(hg)
    mass = dens * dh
    order = np.argsort(dens)[::-1]
    csum = np.cumsum(mass[order])
    p_true = float(np.interp(H_TRUE, hg, dens))
    for lv in (0.50, 0.68, 0.90):
        k = min(int(np.searchsorted(csum, lv)), order.size - 1)
        res[f"hpd{int(round(lv * 100))}"] = float(p_true >= float(dens[order[k]]))
    return res


def ks_uniform(vals: list[float]) -> float:
    q = np.sort(np.asarray(vals, dtype=float))
    n = q.size
    i = np.arange(1, n + 1, dtype=float)
    return float(np.max(np.maximum(i / n - q, q - (i - 1.0) / n)))


def cell_stats(d: dict, ch: str) -> dict:
    """All per-cell statistics, recomputed from ln_post vectors only."""
    hg = grid(d)
    recs = d["per_seed"]
    maps, refs, psd, pits, h50, h68, h90, edges = [], [], [], [], [], [], [], []
    dev_map = dev_psd = dev_pit = dev_ref = 0.0
    nonfinite = 0
    for r in recs:
        lp = np.asarray(r[f"ln_post_{ch}"], dtype=float)
        if not np.all(np.isfinite(lp)):
            nonfinite += 1
        m = map_from_lnpost(hg, lp)
        mr = map_refined_from_lnpost(hg, lp)
        s = post_summary(hg, lp)
        maps.append(m)
        refs.append(mr)
        psd.append(s["post_sd"])
        pits.append(s["pit"])
        h50.append(s["hpd50"])
        h68.append(s["hpd68"])
        h90.append(s["hpd90"])
        edges.append(s["edge_mass"])
        dev_map = max(dev_map, abs(m - r[f"map_{ch}"]))
        dev_ref = max(dev_ref, abs(mr - r[f"map_{ch}_refined"]))
        dev_psd = max(dev_psd, abs(s["post_sd"] - r[f"post_sd_{ch}"]))
        dev_pit = max(dev_pit, abs(s["pit"] - r[f"pit_{ch}"]))
    a = np.asarray(maps)
    n = a.size
    bias = float(a.mean() - H_TRUE)
    sd1 = float(a.std(ddof=1)) if n > 1 else 0.0
    sd0 = float(a.std(ddof=0))
    return {
        "n": n,
        "bias": bias,
        "sd_ddof1": sd1,
        "sd_ddof0": sd0,
        "se_ddof1": sd1 / math.sqrt(n),
        "se_ddof0": sd0 / math.sqrt(n),
        "bias_refined": float(np.mean(refs) - H_TRUE),
        "post_sd_median": float(np.median(psd)),
        "n_distinct_map": int(len(set(maps))),
        "map_min": float(a.min()),
        "map_max": float(a.max()),
        "rails_low": float(np.mean([r[f"railed_low_{ch}"] for r in recs])),
        "rails_high": float(np.mean([r[f"railed_high_{ch}"] for r in recs])),
        "rails_low_recomputed": float(np.mean([m == hg[0] for m in maps])),
        "rails_high_recomputed": float(np.mean([m == hg[-1] for m in maps])),
        "nonfinite_seeds": nonfinite,
        "hpd50": float(np.mean(h50)),
        "hpd68": float(np.mean(h68)),
        "hpd90": float(np.mean(h90)),
        "pit_max": float(np.max(pits)),
        "ks_D": ks_uniform(pits),
        "edge_loaded_frac": float(np.mean([e > 0.01 for e in edges])),
        "maxdev_vs_stored": {"map": dev_map, "map_refined": dev_ref,
                             "post_sd": dev_psd, "pit": dev_pit},
    }


def validity(d: dict, expect_seeds: list[int]) -> dict:
    recs = d["per_seed"]
    seeds = [r["seed"] for r in recs]
    return {
        "n": len(recs),
        "seeds_exact": seeds == expect_seeds,
        "seeds_sorted_exact": sorted(seeds) == sorted(expect_seeds),
        "seed_first": seeds[0], "seed_last": seeds[-1],
        "n_unique": len(set(seeds)),
        "K_sum_values": sorted({r["K_sum"] for r in recs}),
        "n_events_values": sorted({r["n_events"] for r in recs}),
        "n_events_run_values": sorted({r["n_events_run"] for r in recs}),
        "horizon_drop_max": max(r["n_horizon_dropped"] for r in recs),
        "f_incl_values": sorted({r["f_incl"] for r in recs}),
        "pin_pass": d["pin_integrity"]["pass"],
        "pin_all_match": all_match(d["pin_integrity"]),
        "git_commit": d["git_commit"][:10],
        "git_dirty": d["git_dirty"],
        "import_path_clean": d["import_path_clean"],
        "dirt_import_path": d["dirt_inventory"].get("import_path"),
        "allow_dirty": d["allow_dirty"],
        "smoke": d.get("smoke"),
        "sigma_bar_mean": float(np.mean([r["sigma_z_mean_pairs"] for r in recs])),
        "sigma_bar_min": float(np.min([r["sigma_z_mean_pairs"] for r in recs])),
        "sigma_bar_max": float(np.max([r["sigma_z_mean_pairs"] for r in recs])),
        "wall_s": d["wall_time_s"], "workers": d["workers"],
        "cpu_h_per_seed": d["wall_time_s"] * d["workers"] / 3600.0 / len(recs),
    }


def all_match(pin: dict) -> bool:
    ok = True
    for k, v in pin.items():
        if isinstance(v, dict):
            if "match" in v:
                m = v["match"]
                ok &= all(m.values()) if isinstance(m, dict) else bool(m)
    return bool(ok)


# =====================================================================  PART A
REF_1D, REF_SE = 0.037237, 0.000230
REF_2D = 0.039713
WINDOW = 0.002
PRED, PRED_SE = 0.0368515, 0.000564

mn0x = load("MN0X_h0p730_results_seeds0_100")
mn0 = load("MN0_h0p730_results_seeds0_15")
meh = load("MEH_h0p730_results_seeds0_15")
mei = load("MEI_h0p730_results_seeds0_15")

A: dict = {}
A["validity"] = validity(mn0x, list(range(BASE + 50000, BASE + 50100)))
A["validity"]["MN0_subset"] = {r["seed"] for r in mn0["per_seed"]} <= {
    r["seed"] for r in mn0x["per_seed"]}
A["validity"]["overlap_MEH"] = sorted(
    {r["seed"] for r in mn0x["per_seed"]} & {r["seed"] for r in meh["per_seed"]})
A["validity"]["overlap_MEI"] = sorted(
    {r["seed"] for r in mn0x["per_seed"]} & {r["seed"] for r in mei["per_seed"]})
A["validity"]["MEH_first"] = min(r["seed"] for r in meh["per_seed"])
A["validity"]["top_level_seeds_match"] = sorted(mn0x["seeds"]) == list(
    range(BASE + 50000, BASE + 50100))

for ch in ("1d", "2d"):
    A[f"MN0X_{ch}"] = cell_stats(mn0x, ch)
    A[f"MN0_{ch}"] = cell_stats(mn0, ch)

b1 = A["MN0X_1d"]["bias"]
b2 = A["MN0X_2d"]["bias"]
se1 = A["MN0X_1d"]["se_ddof1"]
se1p = A["MN0X_1d"]["se_ddof0"]
sed = math.hypot(se1, REF_SE)
A["A1_rule"] = {
    "mean_bias_1d": b1, "abs_delta": abs(b1 - REF_1D), "window": WINDOW,
    "margin": WINDOW - abs(b1 - REF_1D),
    "times_inside": WINDOW / abs(b1 - REF_1D),
    "verdict": "A1-PASS" if abs(b1 - REF_1D) <= WINDOW else "A1-FAIL",
    "se_ddof1": se1, "se_ddof0": se1p, "sd_ddof1": A["MN0X_1d"]["sd_ddof1"],
    "se_diff": sed, "delta_in_sigma_diff": (b1 - REF_1D) / sed,
    "window_in_sigma_arm": WINDOW / se1, "window_in_sigma_diff": WINDOW / sed,
    "mean_bias_2d": b2, "abs_delta_2d": abs(b2 - REF_2D),
    "bias_refined_1d": A["MN0X_1d"]["bias_refined"],
    "bias_refined_2d": A["MN0X_2d"]["bias_refined"],
    "verdict_refined": "A1-PASS" if abs(A["MN0X_1d"]["bias_refined"] - REF_1D)
    <= WINDOW else "A1-FAIL",
}
A["prediction"] = {
    "predicted": PRED, "pred_se": PRED_SE, "measured": b1,
    "residual": b1 - PRED, "n_sigma": (b1 - PRED) / PRED_SE,
    "inside_68": 0.03629 <= b1 <= 0.03741,
}

# fresh-85 arithmetic (§5 falsification)
mn0_seeds = {r["seed"] for r in mn0["per_seed"]}
fresh = {"config": mn0x["config"],
         "per_seed": [r for r in mn0x["per_seed"] if r["seed"] not in mn0_seeds]}
incl = {"config": mn0x["config"],
        "per_seed": [r for r in mn0x["per_seed"] if r["seed"] in mn0_seeds]}
f1 = cell_stats(fresh, "1d")
i1 = cell_stats(incl, "1d")
FRESH_SE_REG = 0.0061154 / math.sqrt(85)
A["fresh_85"] = {
    "n": f1["n"], "mean_bias": f1["bias"], "se_ddof1": f1["se_ddof1"],
    "se_ddof0": f1["se_ddof0"],
    "delta_vs_reference": f1["bias"] - REF_1D,
    "sigma_vs_reference_registered_se": (f1["bias"] - REF_1D) / FRESH_SE_REG,
    "fail_threshold": 0.0353376,
    "above_threshold": f1["bias"] - 0.0353376,
    "sigma_above_threshold": (f1["bias"] - 0.0353376) / FRESH_SE_REG,
}
A["included_15"] = {
    "mean_bias": i1["bias"], "se_ddof0": i1["se_ddof0"], "se_ddof1": i1["se_ddof1"],
    "equals_parent_record": abs(i1["bias"] - 0.034667) < 5e-7,
    "diff_15_minus_85": i1["bias"] - f1["bias"],
    "se_diff_ddof0": math.hypot(i1["se_ddof0"], f1["se_ddof0"]),
    "sigma_ddof0": (i1["bias"] - f1["bias"]) / math.hypot(i1["se_ddof0"], f1["se_ddof0"]),
    "sigma_ddof1": (i1["bias"] - f1["bias"]) / math.hypot(i1["se_ddof1"], f1["se_ddof1"]),
}

# ---- A1-DET, independent implementation over ALL shared fields -------------
xb = {r["seed"]: r for r in mn0x["per_seed"]}
ob = {r["seed"]: r for r in mn0["per_seed"]}
shared = sorted(set(xb) & set(ob))
keys_x = set(mn0x["per_seed"][0])
keys_o = set(mn0["per_seed"][0])
det = {
    "n_shared": len(shared),
    "keys_only_in_MN0X": sorted(keys_x - keys_o),
    "keys_only_in_MN0": sorted(keys_o - keys_x),
    "n_shared_keys": len(keys_x & keys_o),
    "n_compared_excluding_cell": len(keys_x & keys_o) - 1,
    "list_fields": sorted(k for k in keys_x & keys_o
                          if isinstance(mn0x["per_seed"][0][k], list)),
}
worst, worst_field, mismatch = 0.0, None, []
n_scalar_vals = 0
for s in shared:
    a, b = xb[s], ob[s]
    for f in sorted((keys_x & keys_o) - {"cell"}):
        va, vb = a[f], b[f]
        pairs = list(zip(va, vb)) if isinstance(va, list) else [(va, vb)]
        for u, v in pairs:
            n_scalar_vals += 1
            if isinstance(u, str) or isinstance(v, str):
                if u != v:
                    mismatch.append((s, f, u, v))
                continue
            if u == v:
                continue
            rel = abs(u - v) / max(abs(u), abs(v), 1e-300)
            if rel > worst:
                worst, worst_field = rel, f
det["max_rel_dev"] = worst
det["worst_field"] = worst_field
det["string_mismatches"] = mismatch
det["n_value_comparisons"] = n_scalar_vals
det["maps_exactly_equal"] = all(
    xb[s]["map_1d"] == ob[s]["map_1d"] and xb[s]["map_2d"] == ob[s]["map_2d"]
    for s in shared)
det["lnpost_bitidentical"] = all(
    xb[s][k] == ob[s][k] for s in shared for k in ("ln_post_1d", "ln_post_2d"))
det["cell_labels"] = [xb[shared[0]]["cell"], ob[shared[0]]["cell"]]
det["commit_MN0X"] = mn0x["git_commit"][:10]
det["commit_MN0"] = mn0["git_commit"][:10]
det["cross_commit"] = mn0x["git_commit"] != mn0["git_commit"]
det["verdict"] = "PASS" if (worst <= 1e-12 and det["maps_exactly_equal"]) else "FAIL"
A["A1_DET"] = det

# ---- companion statistics + quantisation ----------------------------------
hg = grid(mn0x)
fine = [float(x) for x in hg]
A["grid"] = {
    "n": len(fine), "min": fine[0], "max": fine[-1],
    "unique_spacings": sorted({round(fine[i + 1] - fine[i], 6)
                               for i in range(len(fine) - 1)}),
    "uniform": len({round(fine[i + 1] - fine[i], 6)
                    for i in range(len(fine) - 1)}) == 1,
    "all_MN0X_maps_in_0.005_region": bool(
        A["MN0X_1d"]["map_min"] >= 0.655 and A["MN0X_1d"]["map_max"] <= 0.80),
}
A["quantisation"] = {
    "mean_bias_in_ticks_of_5e-5": b1 / 5e-5,
    "reference_in_ticks_of_5e-5": REF_1D / 5e-5,
    "max_offset": 0.005 / 200,
    "abs_delta_below_max_offset": abs(b1 - REF_1D) < 0.005 / 200,
}
A["companion"] = {
    "post_sd_median_1d": A["MN0X_1d"]["post_sd_median"],
    "post_sd_median_2d": A["MN0X_2d"]["post_sd_median"],
    "bias_over_post_sd_1d": b1 / A["MN0X_1d"]["post_sd_median"],
    "bias_over_post_sd_2d": b2 / A["MN0X_2d"]["post_sd_median"],
    "hpd_1d": [A["MN0X_1d"]["hpd50"], A["MN0X_1d"]["hpd68"], A["MN0X_1d"]["hpd90"]],
    "hpd_2d": [A["MN0X_2d"]["hpd50"], A["MN0X_2d"]["hpd68"], A["MN0X_2d"]["hpd90"]],
    "ks_D_1d": A["MN0X_1d"]["ks_D"], "ks_D_2d": A["MN0X_2d"]["ks_D"],
    "pit_max_1d": A["MN0X_1d"]["pit_max"], "pit_max_2d": A["MN0X_2d"]["pit_max"],
}
OUT["A1"] = A


# =====================================================================  PART B
SIG_CELL = 0.001579
PER_SEED_SD = 0.0061154
SE_S23_REG = 0.00061154
DEAD_15 = 0.004737
S23_INT, S23_THR = 0.01150132, 0.00783208
S13_INT, S13_THR = 0.0095703, 0.0000963
SBAR_MN0 = 0.041813
HOST_SHARE = 8.2265e-4

B: dict = {}
cells: dict[str, dict] = {}
for h in range(4):
    for i in range(4):
        nm = f"S{h}{i}"
        n = 100 if nm == "S23" else 15
        d = load(f"{nm}_h0p730_results_seeds0_{n}")
        exp = [BASE + 51000 + 100 * (4 * h + i) + j for j in range(n)]
        v = validity(d, exp)
        pred = SBAR_MN0 * (FRACS[h] * HOST_SHARE + FRACS[i] * (1 - HOST_SHARE))
        cells[nm] = {
            "f_h": FRACS[h], "f_i": FRACS[i], "N_registered": n,
            "validity": v,
            "dose_scales": d["config"]["dose_scales"],
            "dose_ok": list(d["config"]["dose_scales"]) == [FRACS[h], FRACS[i]],
            "sbar_pred": pred, "sbar_meas": v["sigma_bar_mean"],
            "sbar_relerr": (abs(v["sigma_bar_mean"] - pred) / pred) if pred else 0.0,
            "sbar_tolerance": 0.10 if FRACS[i] == 0.0 else 0.02,
            "1d": cell_stats(d, "1d"), "2d": cell_stats(d, "2d"),
        }
        cells[nm]["sbar_in_tol"] = (
            cells[nm]["sbar_relerr"] <= cells[nm]["sbar_tolerance"]
            if pred else v["sigma_bar_mean"] == 0.0)
B["cells"] = cells

b = {k: v["1d"]["bias"] for k, v in cells.items()}
se = {k: v["1d"]["se_ddof1"] for k, v in cells.items()}
se0 = {k: v["1d"]["se_ddof0"] for k, v in cells.items()}
b2d = {k: v["2d"]["bias"] for k, v in cells.items()}
se2d = {k: v["2d"]["se_ddof1"] for k, v in cells.items()}

# ---- DS-D4 / pin ----------------------------------------------------------
B["DS_D4"] = {
    nm: {"bias": b[nm], "sd": cells[nm]["1d"]["sd_ddof1"],
         "distinct_maps": cells[nm]["1d"]["n_distinct_map"],
         "post_sd_median": cells[nm]["1d"]["post_sd_median"],
         "exact_zero": b[nm] == 0.0 and cells[nm]["1d"]["sd_ddof1"] == 0.0}
    for nm in ("S00", "S01", "S02", "S03")}
B["DS_D4"]["class"] = "PIN-BINARY" if all(
    B["DS_D4"][nm]["exact_zero"] for nm in ("S00", "S01", "S02", "S03")
) else "PIN-GRADED"
B["S00_exactly_zero"] = b["S00"] == 0.0

# ---- corner cross-checks --------------------------------------------------
def corner(cell, ref, ref_se):
    s = se[cell]
    tol = 3 * math.sqrt(s ** 2 + ref_se ** 2)
    d = b[cell] - ref
    return {"measured": b[cell], "se": s, "ref": ref, "ref_se": ref_se,
            "delta": d, "tolerance": tol, "ratio": abs(d) / tol,
            "sigma": abs(d) / math.sqrt(s ** 2 + ref_se ** 2),
            "verdict": "PASS" if abs(d) <= tol else "CROSS-CHECK-FAILED"}


B["corners"] = {
    "S33_vs_MN0": corner("S33", 0.034667, 0.001579),
    "S30_vs_MEH": corner("S30", 0.004000, 0.000535),
    "S03_vs_MEI": {"measured": b["S03"], "sd": cells["S03"]["1d"]["sd_ddof1"],
                   "verdict": "PASS (exact)" if b["S03"] == 0.0
                   else "CROSS-CHECK-DISCREPANT"},
    "S00_anchor": {"measured": b["S00"],
                   "verdict": "PASS (exact)" if b["S00"] == 0.0
                   else "SCAN-CONFOUNDED"},
}
# parent arms recomputed from their own raw records
B["parent_arms_recomputed"] = {
    nm: {"1d": cell_stats(load(f"{nm}_h0p730_results_seeds0_15"), "1d"),
         "2d": cell_stats(load(f"{nm}_h0p730_results_seeds0_15"), "2d")}
    for nm in ("MN0", "MEH", "MEI")}

# ---- DS-D2 ----------------------------------------------------------------
def dsd2(bb, ss):
    r = {}
    for h in range(1, 4):
        for i in range(1, 4):
            nm, a_, c_ = f"S{h}{i}", f"S{h}0", f"S0{i}"
            D = bb[nm] - bb[a_] - bb[c_] + bb["S00"]
            SE_D = math.sqrt(ss[nm] ** 2 + ss[a_] ** 2 + ss[c_] ** 2 + ss["S00"] ** 2)
            cls = ("NON-ADDITIVE" if abs(D) >= 3 * SE_D else
                   "ADDITIVE-CONSISTENT" if abs(D) < 2 * SE_D else "AMBIGUOUS")
            r[nm] = {"D": D, "SE_D": SE_D, "sigma": abs(D) / SE_D, "class": cls}
    return r


B["DS_D2_1d"] = dsd2(b, se)
B["DS_D2_2d"] = dsd2(b2d, se2d)
D11 = B["DS_D2_1d"]["S33"]["D"]
B["DS_D2_prediction_check"] = {
    "registered_D11_prediction": 0.030667,
    "measured_D11": D11,
    "delta": D11 - 0.030667,
    "se_of_difference": math.hypot(B["DS_D2_1d"]["S33"]["SE_D"], 0.001579),
    "sigma": (D11 - 0.030667) / math.hypot(B["DS_D2_1d"]["S33"]["SE_D"], 0.001579),
}

# ---- DS-D3 ----------------------------------------------------------------
def shape(v, hi, lo):
    return ("SHAPE-INTERACTION" if v >= hi else
            "SHAPE-THRESHOLD" if v <= lo else "SHAPE-UNDECIDED")


B["DS_D3"] = {
    "S23_1d": {"b": b["S23"], "se_realized": se["S23"], "se_registered": SE_S23_REG,
               "class": shape(b["S23"], S23_INT, S23_THR),
               "dist_above_INT": b["S23"] - S23_INT,
               "in_realized_SE": (b["S23"] - S23_INT) / se["S23"],
               "in_registered_SE": (b["S23"] - S23_INT) / SE_S23_REG,
               "vs_H_INT_0.017333": b["S23"] - 0.017333,
               "vs_H_INT_sigma": (b["S23"] - 0.017333) / se["S23"],
               "vs_H_THRESH_0.002": b["S23"] - 0.002,
               "vs_H_THRESH_sigma": (b["S23"] - 0.002) / se["S23"]},
    "S23_2d": {"b": b2d["S23"], "class": shape(b2d["S23"], S23_INT, S23_THR)},
    "S13_1d": {"b": b["S13"], "se": se["S13"],
               "class": shape(b["S13"], S13_INT, S13_THR),
               "dist_above_INT_in_SE": (b["S13"] - S13_INT) / se["S13"]},
    "S13_2d": {"b": b2d["S13"], "class": shape(b2d["S13"], S13_INT, S13_THR)},
}

# ---- DS-D5 ----------------------------------------------------------------
def dsd5(anchor0, anchor1, tag):
    r = {}
    for f, nm in ((0.25, "S31"), (0.5, "S32")):
        pred = anchor0 + (anchor1 - anchor0) * f
        d = b[nm] - pred
        cls = ("SUPER-LINEAR" if d >= DEAD_15 else
               "SUB-LINEAR" if d <= -DEAD_15 else "LINEAR-CONSISTENT")
        r[nm] = {"pred": pred, "delta": d, "in_cell_SE": d / se[nm],
                 "class": cls, "edge": DEAD_15}
    r["anchors"] = [anchor0, anchor1, tag]
    return r


B["DS_D5_registered_line"] = dsd5(0.004000, 0.034667, "registered")
B["DS_D5_self_anchored"] = dsd5(b["S30"], b["S33"], "self")

# ---- DS-D6 ----------------------------------------------------------------
B["DS_D6"] = {nm: {"R_dose": b[nm] / (c["f_i"] * SBAR_MN0),
                   "banded": nm == "S33",
                   "in_band": (0.75 <= b[nm] / (c["f_i"] * SBAR_MN0) <= 1.25)
                   if nm == "S33" else None}
              for nm, c in cells.items() if c["f_i"] > 0}
B["DS_D6"]["MN0_anchor"] = 0.034667 / SBAR_MN0

# ---- bilinearity residuals ------------------------------------------------
def bilinear(I, tag):
    r = {"I": I, "tag": tag, "cells": {}}
    for h in range(1, 4):
        for i in range(1, 4):
            nm = f"S{h}{i}"
            D = B["DS_D2_1d"][nm]["D"]
            SE_D = B["DS_D2_1d"][nm]["SE_D"]
            pred = I * FRACS[h] * FRACS[i]
            res = D - pred
            r["cells"][nm] = {"prod": FRACS[h] * FRACS[i], "D": D, "pred": pred,
                              "resid": res, "sigma": res / SE_D,
                              "over3sigma": abs(res) >= 3 * SE_D,
                              "not_evaluable_reg": nm in ("S11", "S12", "S21")}
    r["all_positive"] = all(v["resid"] > 0 for v in r["cells"].values())
    return r


B["bilinear_registered_I"] = bilinear(0.030667, "registered I=0.030667")
B["bilinear_self_I"] = bilinear(D11, "self-anchored I=D(1,1) this scan")
# registered §6 item 1 threshold: product >= 0.0050017/0.030667
B["not_evaluable_threshold_product"] = 0.0050017 / 0.030667
B["not_evaluable_cells_derived"] = sorted(
    nm for nm in B["bilinear_registered_I"]["cells"]
    if B["bilinear_registered_I"]["cells"][nm]["prod"] < 0.0050017 / 0.030667)

# ---- H-THRESH test --------------------------------------------------------
B["H_THRESH_test"] = {
    "f_star": 0.022 / 0.041813,
    "S13": {"measured": b["S13"], "prediction": 0.001000,
            "excess": b["S13"] - 0.001, "sigma": (b["S13"] - 0.001) / se["S13"]},
    "S23": {"measured": b["S23"], "prediction": 0.002000,
            "excess": b["S23"] - 0.002, "sigma": (b["S23"] - 0.002) / se["S23"]},
}

# ---- steps / slopes -------------------------------------------------------
def step(a_, c_):
    d = b[a_] - b[c_]
    s = math.hypot(se[a_], se[c_])
    cls = ("RESOLVED" if abs(d) >= 3 * s else
           "MARGINAL" if abs(d) >= 2 * s else "UNRESOLVED")
    return {"delta": d, "se": s, "sigma": d / s, "class": cls}


B["steps"] = {
    "row_fh1": {"S31-S30": step("S31", "S30"), "S32-S31": step("S32", "S31"),
                "S33-S32": step("S33", "S32")},
    "row_fh05": {"S21-S20": step("S21", "S20"), "S22-S21": step("S22", "S21"),
                 "S23-S22": step("S23", "S22")},
    "row_fh025": {"S11-S10": step("S11", "S10"), "S12-S11": step("S12", "S11"),
                  "S13-S12": step("S13", "S12")},
    "col_fi0": {"S10-S00": step("S10", "S00"), "S20-S10": step("S20", "S10"),
                "S30-S20": step("S30", "S20")},
}
d1, s1 = b["S31"] - b["S30"], math.hypot(se["S31"], se["S30"])
d2, s2 = b["S32"] - b["S31"], math.hypot(se["S32"], se["S31"])
d3, s3 = b["S33"] - b["S32"], math.hypot(se["S33"], se["S32"])
se_2nd = math.sqrt(se["S30"] ** 2 + 4 * se["S31"] ** 2 + se["S32"] ** 2)
m1_, sm1 = d1 / 0.25, s1 / 0.25
m2_, sm2 = d2 / 0.25, s2 / 0.25
m3_, sm3 = d3 / 0.50, s3 / 0.50
B["row_fh1_nonlinearity"] = {
    "second_difference": d2 - d1, "se": se_2nd, "sigma": (d2 - d1) / se_2nd,
    "m1": m1_, "m2": m2_, "m3": m3_,
    "m2-m1": m2_ - m1_, "m2-m1_sigma": (m2_ - m1_) / math.hypot(sm1, sm2),
    "m3-m2": m3_ - m2_, "m3-m2_sigma": (m3_ - m2_) / math.hypot(sm2, sm3),
}
dip = (b["S22"] - b["S21"]) + (b["S12"] - b["S11"])
dip_se = math.sqrt(se["S22"] ** 2 + se["S21"] ** 2 + se["S12"] ** 2 + se["S11"] ** 2)
B["dip"] = {"S22-S21": step("S22", "S21"), "S12-S11": step("S12", "S11"),
            "pooled": dip, "pooled_se": dip_se, "pooled_sigma": dip / dip_se}
B["saturation_ratios"] = {
    f"f_h={FRACS[h]}": {f"f_i={FRACS[i]}":
                        (b[f'S{h}{i}'] - b[f'S{h}0']) / (b[f'S{h}3'] - b[f'S{h}0'])
                        for i in (1, 2, 3)} for h in (1, 2, 3)}
B["resolution_levels"] = {
    "registered_floor_3sigma": 3 * math.sqrt(2) * SIG_CELL,
    "levels": 0.034667 / (3 * math.sqrt(2) * SIG_CELL),
}

# ---- 1D/2D -----------------------------------------------------------------
B["channel_gap"] = {nm: b[nm] - b2d[nm] for nm in cells}
B["classification_agreement"] = {
    "row0_zero_2d": all(b2d[nm] == 0.0 for nm in ("S00", "S01", "S02", "S03")),
    "DSD2_S33_2d": B["DS_D2_2d"]["S33"],
    "DSD3_S23_2d": B["DS_D3"]["S23_2d"],
}

# ---- cost / aborts ---------------------------------------------------------
B["cost"] = {nm: c["validity"]["cpu_h_per_seed"] for nm, c in cells.items()}
B["cost_total_cpu_h"] = sum(
    c["validity"]["wall_s"] * c["validity"]["workers"] / 3600.0 for c in cells.values())
B["cost_max_per_seed"] = max(B["cost"].values())
B["abort_e_fires"] = B["cost_max_per_seed"] > 2 * 0.969
B["rails_any"] = any(c[ch]["rails_low"] + c[ch]["rails_high"] > 0
                     for c in cells.values() for ch in ("1d", "2d"))
B["nonfinite_any"] = any(c[ch]["nonfinite_seeds"] > 0
                         for c in cells.values() for ch in ("1d", "2d"))

# ---- branch tree, registered order ----------------------------------------
br1 = {
    "b_S00_nonzero": b["S00"] != 0.0,
    "validity_fail": not (all(c["validity"]["pin_pass"] for c in cells.values())
                          and all(c["validity"]["K_sum_values"] == [1193703]
                                  for c in cells.values())
                          and all(c["validity"]["seeds_exact"] for c in cells.values())
                          and not B["rails_any"] and not B["nonfinite_any"]),
    "dosing_out_of_tol": not all(c["sbar_in_tol"] for c in cells.values()),
    "A1_FAIL": abs(b1 - REF_1D) > WINDOW,
}
br1["fires"] = any(br1.values())
B["branch_tree"] = {
    "1_SCAN_CONFOUNDED": br1,
    "2_INTERACTION_BILINEAR": {
        "DSD2_S33_NON_ADDITIVE": B["DS_D2_1d"]["S33"]["class"] == "NON-ADDITIVE",
        "DSD3_S23_SHAPE_INTERACTION":
            B["DS_D3"]["S23_1d"]["class"] == "SHAPE-INTERACTION",
        "fires": (B["DS_D2_1d"]["S33"]["class"] == "NON-ADDITIVE"
                  and B["DS_D3"]["S23_1d"]["class"] == "SHAPE-INTERACTION")},
    "3_INTERACTION_THRESHOLD": {
        "would_fire": (B["DS_D2_1d"]["S33"]["class"] == "NON-ADDITIVE"
                       and B["DS_D3"]["S23_1d"]["class"] == "SHAPE-THRESHOLD")},
    "4_ADDITIVE": {"would_fire": B["DS_D2_1d"]["S33"]["class"] != "NON-ADDITIVE"},
    "5_UNDECIDED_conditions": {
        "SHAPE_UNDECIDED": B["DS_D3"]["S23_1d"]["class"] == "SHAPE-UNDECIDED",
        "1D_2D_split": False,
        "PIN_GRADED": B["DS_D4"]["class"] == "PIN-GRADED",
        "fimp0_column_negative": any(b[nm] < 0 for nm in ("S10", "S20", "S30")),
        "resolved_nonbilinear_nonthreshold": (
            any(v["over3sigma"] for nm, v in
                B["bilinear_registered_I"]["cells"].items()
                if not v["not_evaluable_reg"])
            and B["DS_D3"]["S13_1d"]["class"] != "SHAPE-THRESHOLD"),
    },
}
OUT["SCAN"] = B

(HERE / "adjudicate_mechanism_study_output.json").write_text(
    json.dumps(OUT, indent=1, default=str) + "\n")
print(json.dumps(OUT, indent=1, default=str))
