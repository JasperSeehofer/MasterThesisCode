#!/usr/bin/env python
"""Independent scorer for the PARENT mechanism-isolation study.

Recomputes every posterior-derived statistic from the raw per-seed ``ln_post_1d`` /
``ln_post_2d`` vectors of the three registered split-dose arms (MN0 / MEH / MEI) plus
the Amendment-A1 null MN0X. No ``aggregate`` block and no upstream extraction is trusted;
the stored per-seed scalars are *compared against*, never used as inputs.

Read-only on every registered .md. Emits MECHANISM_ISOLATION_READOUT.json alongside.
"""

from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
H_TRUE = 0.730
HPD_LEVELS = (0.50, 0.68, 0.90)

# ---- registered constants (parent §3 / §2 / §5, amendment A1) ---------------
REF_1D = 0.037237          # campaign decision cell T-c(0.730) N=400, 1D grid-argmax bias
REF_1D_SE = 0.000230
REF_2D = 0.039713
VM1_WINDOW = 0.002
IN_BAND = 0.010            # DS-M1 in-band edge
DEFECT = 0.030             # DS-M1 DEFECT edge
NULL_TOL = 0.004           # DS-M1 TERM-INNOCENT proximity-to-null edge
HPD90_OWNS = 0.60          # DS-M1 TERM-OWNS coverage conjunct
DSM5_IMP_MIN = 0.030       # DS-M5: E1-imp bias >= 0.030
DSM5_HOST_MAX = 0.012      # DS-M5: E1-host bias <= 0.012
# DS-M2 registered 2-sigma bands (binomial, N = 25 rows)
DSM2_BANDS = {"hpd50": (0.500, 0.200), "hpd68": (0.680, 0.187), "hpd90": (0.900, 0.120)}
# L0 toy split-dose values (parent §7, K = 50): total / impostors-only / host-only
TOY = {"all": 0.0334, "impostors": 0.0247, "host": 0.0062}

ARMS = {
    "MN0": "MN0_h0p730_results_seeds0_15.json",
    "MEH": "MEH_h0p730_results_seeds0_15.json",
    "MEI": "MEI_h0p730_results_seeds0_15.json",
    "MN0X": "MN0X_h0p730_results_seeds0_100.json",
}
DATA = {k: json.loads((RUN / v).read_text()) for k, v in ARMS.items()}
out: dict = {"arms": {}, "validity": {}, "scorecard": {}, "dsm5": {}, "branch": {}}


# ── recomputation kernel (ports of posterior_readout / pp_readout) ───────────
def posterior_readout(h_grid: np.ndarray, ln_post: np.ndarray) -> dict:
    i = int(np.argmax(ln_post))
    p = np.exp(ln_post - ln_post[i])
    norm_p = float(np.trapezoid(p, h_grid))
    mean = float(np.trapezoid(p * h_grid, h_grid) / norm_p) if norm_p > 0.0 else float("nan")
    map_grid = float(h_grid[i])
    map_ref = map_grid
    if 0 < i < len(h_grid) - 1:
        x0, x1, x2 = h_grid[i - 1], h_grid[i], h_grid[i + 1]
        y0, y1, y2 = ln_post[i - 1], ln_post[i], ln_post[i + 1]
        d1 = (y1 - y0) / (x1 - x0)
        d2 = (y2 - y1) / (x2 - x1)
        curv = (d2 - d1) / (0.5 * (x2 - x0))
        if curv < 0.0:
            map_ref = float(np.clip(0.5 * (x0 + x1) - d1 / curv, h_grid[0], h_grid[-1]))
    return {"map": map_grid, "map_refined": map_ref, "mean": mean,
            "railed_low": float(i == 0), "railed_high": float(i == len(h_grid) - 1)}


def hpd_contains(h_grid, post, h_true, level) -> bool:
    dh = np.gradient(h_grid)
    mass = post * dh
    order = np.argsort(post)[::-1]
    csum = np.cumsum(mass[order])
    k = min(int(np.searchsorted(csum, level)), order.size - 1)
    thresh = float(post[order[k]])
    p_true = float(np.interp(h_true, h_grid, post))
    return p_true >= thresh


def pp_readout(h_grid: np.ndarray, ln_post: np.ndarray, h_true: float) -> dict:
    p = np.exp(ln_post - float(np.max(ln_post)))
    norm_c = float(np.trapezoid(p, h_grid))
    post = p / norm_c
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(h_grid))])
    pit = float(np.interp(h_true, h_grid, cum))
    mean = float(np.trapezoid(post * h_grid, h_grid))
    var = float(np.trapezoid(post * h_grid**2, h_grid)) - mean**2
    r = {"pit": pit, "post_sd": math.sqrt(max(var, 0.0)),
         "edge_mass": float(cum[1] + (cum[-1] - cum[-2]))}
    for lv in HPD_LEVELS:
        r[f"hpd{int(round(lv * 100))}"] = float(hpd_contains(h_grid, post, h_true, lv))
    return r


def ks_distance(pits) -> float:
    q = np.sort(np.asarray(pits, dtype=np.float64))
    n = q.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(np.max(np.maximum(i / n - q, q - (i - 1.0) / n)))


def support_stats(h_grid: np.ndarray, ln_post: np.ndarray) -> dict:
    """How many grid points actually hold posterior mass."""
    p = np.exp(ln_post - float(np.max(ln_post)))
    dh = np.gradient(h_grid)
    m = p * dh
    m = m / m.sum()
    nz = m[m > 0]
    ent = float(-(nz * np.log(nz)).sum())
    return {"n_points_mass_ge_1e-6": int((m >= 1e-6).sum()),
            "n_points_mass_ge_1e-3": int((m >= 1e-3).sum()),
            "max_point_mass": float(m.max()),
            "effective_support": float(math.exp(ent))}


# ── per-arm recomputation ───────────────────────────────────────────────────
def score_arm(name: str) -> dict:
    d = DATA[name]
    grid = np.array(d["config"]["h_grid"], dtype=np.float64)
    recs = d["per_seed"]
    res: dict = {"n": len(recs), "seeds": sorted(r["seed"] for r in recs),
                 "git_commit": d.get("git_commit"), "instrument": d.get("instrument"),
                 "dose_target": d["config"].get("dose_target"),
                 "sigma_mode": d["config"].get("sigma_mode"),
                 "channels": {}, "recompute_max_rel_dev": {}}
    # pins
    res["K_sum_values"] = sorted({r["K_sum"] for r in recs})
    res["n_events_values"] = sorted({r["n_events"] for r in recs})
    res["n_events_run_values"] = sorted({r["n_events_run"] for r in recs})
    res["f_incl_values"] = sorted({r["f_incl"] for r in recs})
    res["n_horizon_dropped_max"] = max(r["n_horizon_dropped"] for r in recs)
    res["sigma_z_mean_pairs_mean"] = sum(r["sigma_z_mean_pairs"] for r in recs) / len(recs)
    res["sigma_z_mean_pairs_min"] = min(r["sigma_z_mean_pairs"] for r in recs)
    res["sigma_z_mean_pairs_max"] = max(r["sigma_z_mean_pairs"] for r in recs)
    res["pin_integrity_pass"] = d["pin_integrity"]["pass"]
    res["pin_crb_md5"] = d["pin_integrity"]["crb_csv_md5"]["value"]
    res["pin_frozeng_md5"] = d["pin_integrity"]["frozeng_emit_md5"]["value"]
    res["import_path_clean"] = d.get("import_path_clean")
    res["allow_dirty"] = d.get("allow_dirty")
    res["git_dirty"] = d.get("git_dirty")
    res["smoke"] = d.get("smoke")
    res["dirt_import_path"] = d.get("dirt_inventory", {}).get("import_path")
    res["preregistration"] = d.get("preregistration")

    for ch in ("1d", "2d"):
        maps, refs, means, pits, sds, edges, rl, rh = [], [], [], [], [], [], [], []
        hpd = {50: [], 68: [], 90: []}
        worst = 0.0
        worst_field = None
        nonfinite = 0
        sup = []
        for r in recs:
            ln = np.array(r[f"ln_post_{ch}"], dtype=np.float64)
            if not np.all(np.isfinite(ln)):
                nonfinite += 1
                continue
            pr = posterior_readout(grid, ln)
            pp = pp_readout(grid, ln, H_TRUE)
            sup.append(support_stats(grid, ln))
            maps.append(pr["map"]); refs.append(pr["map_refined"]); means.append(pr["mean"])
            rl.append(pr["railed_low"]); rh.append(pr["railed_high"])
            pits.append(pp["pit"]); sds.append(pp["post_sd"]); edges.append(pp["edge_mass"])
            for lv in (50, 68, 90):
                hpd[lv].append(pp[f"hpd{lv}"])
            # compare recomputation against the stored scalars
            for key, val in ((f"map_{ch}", pr["map"]), (f"map_{ch}_refined", pr["map_refined"]),
                             (f"mean_{ch}", pr["mean"]), (f"post_sd_{ch}", pp["post_sd"]),
                             (f"pit_{ch}", pp["pit"]), (f"edge_mass_{ch}", pp["edge_mass"]),
                             (f"railed_low_{ch}", pr["railed_low"]),
                             (f"railed_high_{ch}", pr["railed_high"]),
                             (f"hpd50_{ch}", pp["hpd50"]), (f"hpd68_{ch}", pp["hpd68"]),
                             (f"hpd90_{ch}", pp["hpd90"])):
                stored = r[key]
                if stored == val:
                    continue
                dev = abs(stored - val) / max(abs(stored), abs(val), 1e-300)
                if dev > worst:
                    worst, worst_field = dev, key
        n = len(maps)
        bias = [m - H_TRUE for m in maps]
        mean_b = sum(bias) / n
        sd1 = st.stdev(bias) if n > 1 and len(set(bias)) > 1 else 0.0
        sd0 = math.sqrt(sum((x - mean_b) ** 2 for x in bias) / n)
        res["channels"][ch] = {
            "n": n,
            "mean_bias": mean_b,
            "sd_sample": sd1, "sd_pop": sd0,
            "se_sample": sd1 / math.sqrt(n), "se_pop": sd0 / math.sqrt(n),
            "map_values": sorted(set(maps)),
            "n_distinct_maps": len(set(maps)),
            "mean_bias_refined": sum(refs) / n - H_TRUE,
            "post_sd_median": st.median(sds),
            "bias_over_post_sd": (mean_b / st.median(sds)) if st.median(sds) > 0 else float("inf"),
            "hpd50_cov": sum(hpd[50]) / n, "hpd68_cov": sum(hpd[68]) / n,
            "hpd90_cov": sum(hpd[90]) / n,
            "pit_ks_D": ks_distance(pits),
            "pit_max": max(pits), "pit_min": min(pits),
            "railed_low_frac": sum(rl) / n, "railed_high_frac": sum(rh) / n,
            "edge_loaded_frac": sum(e > 0.01 for e in edges) / n,
            "nonfinite_ln_post_seeds": nonfinite,
            "support_max_point_mass_median": st.median([s["max_point_mass"] for s in sup]),
            "support_eff_median": st.median([s["effective_support"] for s in sup]),
            "support_npts_1e6_median": st.median([s["n_points_mass_ge_1e-6"] for s in sup]),
            "support_npts_1e6_max": max(s["n_points_mass_ge_1e-6"] for s in sup),
        }
        res["recompute_max_rel_dev"][ch] = {"max_rel_dev": worst, "worst_field": worst_field}
    return res


for a in ARMS:
    out["arms"][a] = score_arm(a)

# ── DS-M1 classification ────────────────────────────────────────────────────
b_n0 = out["arms"]["MN0"]["channels"]["1d"]["mean_bias"]


def classify(b: float, hpd90: float, b_null: float) -> str:
    if abs(b) <= IN_BAND and hpd90 >= HPD90_OWNS:
        return "TERM-OWNS"
    if IN_BAND < abs(b) < DEFECT:
        return "TERM-PARTIAL"
    if abs(b) >= DEFECT and abs(b - b_null) <= NULL_TOL:
        return "TERM-INNOCENT"
    return "OTHER"


out["scorecard"]["DS_M1"] = {}
for a in ("MN0", "MEH", "MEI", "MN0X"):
    for ch in ("1d", "2d"):
        c = out["arms"][a]["channels"][ch]
        null_ref = out["arms"]["MN0"]["channels"][ch]["mean_bias"]
        out["scorecard"]["DS_M1"][f"{a}_{ch}"] = {
            "bias": c["mean_bias"], "se": c["se_sample"], "hpd90": c["hpd90_cov"],
            "abs_b": abs(c["mean_bias"]), "abs_b_minus_null": abs(c["mean_bias"] - null_ref),
            "class": classify(c["mean_bias"], c["hpd90_cov"], null_ref),
        }

# ── DS-M2 coverage vs registered bands ──────────────────────────────────────
out["scorecard"]["DS_M2"] = {}
for a in ("MN0", "MEH", "MEI", "MN0X"):
    for ch in ("1d", "2d"):
        c = out["arms"][a]["channels"][ch]
        row = {}
        for lv, (mid, half) in DSM2_BANDS.items():
            v = c[f"{lv}_cov"]
            row[lv] = {"value": v, "band": [mid - half, mid + half],
                       "inside": (mid - half) <= v <= (mid + half)}
        out["scorecard"]["DS_M2"][f"{a}_{ch}"] = row

# ── DS-M5 — the registered confrontation ────────────────────────────────────
imp = out["arms"]["MEI"]["channels"]["1d"]
host = out["arms"]["MEH"]["channels"]["1d"]
null = out["arms"]["MN0"]["channels"]["1d"]
nullx = out["arms"]["MN0X"]["channels"]["1d"]
se_sum = math.hypot(host["se_sample"], imp["se_sample"])
sum_split = host["mean_bias"] + imp["mean_bias"]
nonadd = null["mean_bias"] - sum_split
se_nonadd = math.hypot(null["se_sample"], se_sum)
nonadd_x = nullx["mean_bias"] - sum_split
se_nonadd_x = math.hypot(nullx["se_sample"], se_sum)
out["dsm5"] = {
    "cond_imp_ge_0.030": {"measured": imp["mean_bias"], "required": DSM5_IMP_MIN,
                          "satisfied": imp["mean_bias"] >= DSM5_IMP_MIN,
                          "shortfall": DSM5_IMP_MIN - imp["mean_bias"],
                          "shortfall_in_MN0_se": (DSM5_IMP_MIN - imp["mean_bias"]) / null["se_sample"],
                          "arm_se": imp["se_sample"]},
    "cond_host_le_0.012": {"measured": host["mean_bias"], "required": DSM5_HOST_MAX,
                           "satisfied": host["mean_bias"] <= DSM5_HOST_MAX,
                           "margin": DSM5_HOST_MAX - host["mean_bias"],
                           "margin_in_arm_se": (DSM5_HOST_MAX - host["mean_bias"]) / host["se_sample"]},
    "M5prime_confirmed": bool(imp["mean_bias"] >= DSM5_IMP_MIN and host["mean_bias"] <= DSM5_HOST_MAX),
    "split_direction": {"host_minus_imp": host["mean_bias"] - imp["mean_bias"],
                        "predicted_sign": "imp >> host", "measured_sign": "host > imp"},
    "non_additivity_vs_MN0": {"sum_split": sum_split, "se_sum": se_sum,
                              "null": null["mean_bias"], "se_null": null["se_sample"],
                              "residual": nonadd, "se_residual": se_nonadd,
                              "sigma": nonadd / se_nonadd},
    "non_additivity_vs_MN0X": {"null": nullx["mean_bias"], "se_null": nullx["se_sample"],
                               "residual": nonadd_x, "se_residual": se_nonadd_x,
                               "sigma": nonadd_x / se_nonadd_x},
    "toy_vs_instrument": {
        "MN0": {"toy": TOY["all"], "instrument": null["mean_bias"],
                "sign_agrees": True},
        "MEH": {"toy": TOY["host"], "instrument": host["mean_bias"],
                "sign_agrees": (TOY["host"] > 0) == (host["mean_bias"] > 0)},
        "MEI": {"toy": TOY["impostors"], "instrument": imp["mean_bias"],
                "instrument_is_exactly_zero": imp["mean_bias"] == 0.0,
                "sign_agrees_literal": None},
    },
    "fraction_of_null_carried": {
        "host_over_MN0": host["mean_bias"] / null["mean_bias"],
        "imp_over_MN0": imp["mean_bias"] / null["mean_bias"],
        "sum_over_MN0": sum_split / null["mean_bias"],
    },
}

# ── V-M1 via A1 ─────────────────────────────────────────────────────────────
out["validity"]["V_M1"] = {
    "MN0_N15": {"bias": null["mean_bias"], "abs_delta": abs(null["mean_bias"] - REF_1D),
                "window": VM1_WINDOW, "passes": abs(null["mean_bias"] - REF_1D) <= VM1_WINDOW},
    "MN0X_N100": {"bias": nullx["mean_bias"], "se": nullx["se_sample"],
                  "abs_delta": abs(nullx["mean_bias"] - REF_1D), "window": VM1_WINDOW,
                  "passes": abs(nullx["mean_bias"] - REF_1D) <= VM1_WINDOW,
                  "times_inside": VM1_WINDOW / abs(nullx["mean_bias"] - REF_1D),
                  "sigma_diff": (nullx["mean_bias"] - REF_1D)
                                / math.hypot(nullx["se_sample"], REF_1D_SE)},
    "MN0X_2d": {"bias": out["arms"]["MN0X"]["channels"]["2d"]["mean_bias"],
                "ref": REF_2D,
                "abs_delta": abs(out["arms"]["MN0X"]["channels"]["2d"]["mean_bias"] - REF_2D)},
}

# ── seed-block integrity ────────────────────────────────────────────────────
BASE = 20260808
blocks = {"MN0": (50000, 15), "MEH": (50100, 15), "MEI": (50200, 15), "MN0X": (50000, 100)}
sb = {}
for a, (off, n) in blocks.items():
    exp = list(range(BASE + off, BASE + off + n))
    got = out["arms"][a]["seeds"]
    sb[a] = {"registered_first": exp[0], "registered_last": exp[-1],
             "realized_first": got[0], "realized_last": got[-1],
             "exact_match": got == exp, "n": len(got), "n_unique": len(set(got))}
sets = {a: set(out["arms"][a]["seeds"]) for a in ARMS}
sb["overlaps"] = {
    "MN0_MEH": sorted(sets["MN0"] & sets["MEH"]),
    "MN0_MEI": sorted(sets["MN0"] & sets["MEI"]),
    "MEH_MEI": sorted(sets["MEH"] & sets["MEI"]),
    "MN0X_MEH": sorted(sets["MN0X"] & sets["MEH"]),
    "MN0X_MEI": sorted(sets["MN0X"] & sets["MEI"]),
    "MN0_subset_MN0X": sets["MN0"] <= sets["MN0X"],
}
sb["inside_registered_decade_50000_50999"] = all(
    BASE + 50000 <= s <= BASE + 50999 for a in ARMS for s in out["arms"][a]["seeds"])
out["validity"]["seed_blocks"] = sb

# ── V-M5 artifact presence ──────────────────────────────────────────────────
vm5 = sorted(p.name for p in RUN.glob("VM5_GOLDEN_*.md"))
out["validity"]["V_M5"] = {"golden_readout_files_found": vm5,
                           "status": "CLOSED" if vm5 else "OPEN (D-A1-2 carried)",
                           "other_vm5_artifacts": sorted(p.name for p in RUN.glob("*vm5*"))
                           + sorted(p.name for p in RUN.glob("*VM5*"))}

# ── abort criteria ──────────────────────────────────────────────────────────
out["validity"]["abort"] = {}
for a in ("MN0", "MEH", "MEI"):
    arm = out["arms"][a]
    nf = sum(arm["channels"][ch]["nonfinite_ln_post_seeds"] for ch in ("1d", "2d"))
    out["validity"]["abort"][a] = {
        "a_nonfinite_seed_frac": nf / arm["n"],
        "a_fires": (nf / arm["n"]) > 0.01,
        "b_horizon_dropped_max": arm["n_horizon_dropped_max"],
        "b_fires": arm["n_horizon_dropped_max"] > 0.05 * 982,
        "rails_1d": [arm["channels"]["1d"]["railed_low_frac"], arm["channels"]["1d"]["railed_high_frac"]],
        "rails_2d": [arm["channels"]["2d"]["railed_low_frac"], arm["channels"]["2d"]["railed_high_frac"]],
    }
out["validity"]["abort"]["d_toy_vs_instrument_sign"] = out["dsm5"]["toy_vs_instrument"]

# ── branch evaluation, registered order ─────────────────────────────────────
classes_1d = {a: out["scorecard"]["DS_M1"][f"{a}_1d"]["class"] for a in ("MN0", "MEH", "MEI")}
owns = [a for a, c in classes_1d.items() if c == "TERM-OWNS"]
out["branch"] = {
    "classes_1d": classes_1d,
    "classes_2d": {a: out["scorecard"]["DS_M1"][f"{a}_2d"]["class"] for a in ("MN0", "MEH", "MEI")},
    "n_term_owns_1d": len(owns),
    "term_owns_arms_1d": owns,
    "branch1_leg_VM1": "not satisfied (A1-PASS at N=100)",
    "branch1_leg_validity": "V-M1/V-M2/V-M3/V-M4 pass; V-M5 " + out["validity"]["V_M5"]["status"],
    "branch2_SINGLE_OWNER": len(owns) == 1,
    "branch3_MULTI_TERM": len(owns) >= 2,
    "branch4_NO_OWNER": len(owns) == 0,
}


# ── extra structural diagnostics (posterior collapse, MAP histograms) ───────
def structure(name: str) -> dict:
    d = DATA[name]
    grid = np.array(d["config"]["h_grid"], dtype=np.float64)
    recs = d["per_seed"]
    i_true = int(np.argmin(np.abs(grid - H_TRUE)))
    row = {}
    for ch in ("1d", "2d"):
        maps, gaps, truegap, rng, pits = [], [], [], [], []
        for r in recs:
            ln = np.array(r[f"ln_post_{ch}"], dtype=np.float64)
            i = int(np.argmax(ln))
            maps.append(float(grid[i]))
            srt = np.sort(ln)[::-1]
            gaps.append(float(srt[0] - srt[1]))
            truegap.append(float(ln[i_true] - max(ln[i_true - 1], ln[i_true + 1])))
            rng.append(float(ln.max() - ln.min()))
            pits.append(r[f"pit_{ch}"])
        row[ch] = {
            "map_histogram": {str(v): maps.count(v) for v in sorted(set(maps))},
            "ln_post_top_gap_median": st.median(gaps),
            "ln_post_top_gap_min": min(gaps), "ln_post_top_gap_max": max(gaps),
            "ln_post_gap_at_h_true_median": st.median(truegap),
            "ln_post_dynamic_range_median": st.median(rng),
            "pit_min": min(pits), "pit_max": max(pits), "pit_median": st.median(pits),
        }
    return row


out["structure"] = {a: structure(a) for a in ARMS}

# ── companion R_dose (reported UNBANDED; denominator ill-defined on split arms)
out["scorecard"]["R_dose_companion"] = {
    a: {"bias_1d": out["arms"][a]["channels"]["1d"]["mean_bias"],
        "sigma_z_mean_pairs": out["arms"][a]["sigma_z_mean_pairs_mean"],
        "R_dose_pairwise": (out["arms"][a]["channels"]["1d"]["mean_bias"]
                            / out["arms"][a]["sigma_z_mean_pairs_mean"])}
    for a in ARMS}

# ── DS-M3 / DS-M4 evaluability ─────────────────────────────────────────────
out["scorecard"]["DS_M3"] = {
    "registered": "each surviving arm re-run at flat doses 0.011 and 0.035, 5 seeds each; "
                  "residual R_dose must fall below 0.25 at both doses",
    "flat_dose_arms_present": [a for a in ARMS if DATA[a]["config"].get("sigma_mode") != "glade"],
    "verdict": "NOT EVALUABLE — no flat-dose arm was run in this study "
               "(all four arms are sigma_mode='glade'); arm E3 was never run",
}
out["scorecard"]["DS_M4"] = {
    "registered_arm": "A-M5b (W1)",
    "arm_run": False,
    "verdict": "NOT EVALUABLE — arm withdrawn at registration (parent §2); "
               "seeds +46000..46399 reserved and unconsumed",
    "L0_toy_deltas_vs_0.004_edge": {
        "rate_shaped_weights": 0.02 * TOY["all"],
        "oracle_weights_true_z": 0.01 * TOY["all"],
        "w_pop_inside_integral": 0.28 * TOY["all"],
        "window_renormalisation": 0.22 * TOY["all"],
    },
}

# ── MN0 2D against its OWN campaign reference ──────────────────────────────
out["validity"]["V_M1_2d_note"] = {
    "MN0_2d": out["arms"]["MN0"]["channels"]["2d"]["mean_bias"],
    "campaign_2d_reference": REF_2D,
    "abs_delta": abs(out["arms"]["MN0"]["channels"]["2d"]["mean_bias"] - REF_2D),
    "inside_window": abs(out["arms"]["MN0"]["channels"]["2d"]["mean_bias"] - REF_2D) <= VM1_WINDOW,
    "note": "the parent operational record compared MN0's 2D value against the 1D reference "
            "0.037237; against the 2D reference 0.039713 it is also outside +-0.002. "
            "V-M1 is registered on the 1D channel; no verdict depends on this.",
}

(RUN / "MECHANISM_ISOLATION_READOUT.json").write_text(json.dumps(out, indent=1, default=str))

# ── console summary ─────────────────────────────────────────────────────────
print("=== ARMS (1D) ===")
for a in ARMS:
    c = out["arms"][a]["channels"]["1d"]
    print(f"{a:5s} N={c['n']:3d} bias={c['mean_bias']:+.6f} sd={c['sd_sample']:.6f} "
          f"SE={c['se_sample']:.6f} maps={c['map_values']} hpd50/68/90="
          f"{c['hpd50_cov']:.3f}/{c['hpd68_cov']:.3f}/{c['hpd90_cov']:.3f} "
          f"post_sd_med={c['post_sd_median']:.6f} KS={c['pit_ks_D']:.4f} "
          f"suppmax={c['support_max_point_mass_median']:.6f} npts={c['support_npts_1e6_median']}")
print("=== ARMS (2D) ===")
for a in ARMS:
    c = out["arms"][a]["channels"]["2d"]
    print(f"{a:5s} bias={c['mean_bias']:+.6f} SE={c['se_sample']:.6f} "
          f"hpd90={c['hpd90_cov']:.3f} post_sd_med={c['post_sd_median']:.6f} "
          f"maps={c['map_values']}")
print("=== recompute vs stored ===")
for a in ARMS:
    print(a, out["arms"][a]["recompute_max_rel_dev"])
print("=== DS-M1 ===")
for k, v in out["scorecard"]["DS_M1"].items():
    print(f"{k:9s} b={v['bias']:+.6f} |b|={v['abs_b']:.6f} |b-bN0|={v['abs_b_minus_null']:.6f} "
          f"hpd90={v['hpd90']:.3f} -> {v['class']}")
print("=== DS-M5 ===")
print(json.dumps(out["dsm5"], indent=1))
print("=== BRANCH ===")
print(json.dumps(out["branch"], indent=1))
print("=== V-M5 ===", out["validity"]["V_M5"])
print("=== seeds ===", json.dumps(out["validity"]["seed_blocks"], indent=1))
print("=== pins ===")
for a in ARMS:
    r = out["arms"][a]
    print(a, r["git_commit"], r["dose_target"], r["K_sum_values"], r["n_events_values"],
          r["f_incl_values"], f"sigmaz={r['sigma_z_mean_pairs_mean']:.6f}",
          "pin_ok", r["pin_integrity_pass"], "clean", r["import_path_clean"],
          "allow_dirty", r["allow_dirty"], "smoke", r["smoke"], "dirt", r["dirt_import_path"])
print("=== abort ===", json.dumps(out["validity"]["abort"], indent=1))
print("=== structure ===", json.dumps(out["structure"], indent=1))
print("=== R_dose/DS-M3/DS-M4/2D-note ===", json.dumps(
    {"R": out["scorecard"]["R_dose_companion"], "M3": out["scorecard"]["DS_M3"],
     "M4": out["scorecard"]["DS_M4"], "vm1_2d": out["validity"]["V_M1_2d_note"]}, indent=1))
print("=== DS-M2 ===", json.dumps(out["scorecard"]["DS_M2"], indent=1))
