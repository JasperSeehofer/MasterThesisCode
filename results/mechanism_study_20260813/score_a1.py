"""Independent scorer for Amendment A1 (arm MN0X, V-M1 null at N=100).

Recomputes every quantity from the raw per-seed records; trusts no aggregate block.
Read-only on all registered .md files. Emits A1_READOUT.json alongside.
"""

import json
import math
import statistics as st
from pathlib import Path

RUN = Path(__file__).resolve().parent
MN0X = json.loads((RUN / "MN0X_h0p730_results_seeds0_100.json").read_text())
MN0 = json.loads((RUN / "MN0_h0p730_results_seeds0_15.json").read_text())
MEH = json.loads((RUN / "MEH_h0p730_results_seeds0_15.json").read_text())
MEI = json.loads((RUN / "MEI_h0p730_results_seeds0_15.json").read_text())

REF = 0.037237          # campaign decision cell T-c(0.730) N=400 1D grid-argmax bias
REF_SE = 0.000230
WINDOW = 0.002          # registered V-M1 window, unchanged
PRED = 0.03685          # registered point prediction
PRED_SE = 0.000564
BASE = 20260808
H_TRUE = 0.730

out = {}

# ---------------------------------------------------------------- seed block
ps = MN0X["per_seed"]
seeds = [r["seed"] for r in ps]
expected = list(range(BASE + 50000, BASE + 50100))
out["n_per_seed_records"] = len(ps)
out["n_unique_seeds"] = len(set(seeds))
out["seeds_equal_registered_block"] = sorted(seeds) == expected
out["seed_min"], out["seed_max"] = min(seeds), max(seeds)
out["json_seeds_field_matches"] = sorted(MN0X["seeds"]) == expected
meh = {r["seed"] for r in MEH["per_seed"]}
mei = {r["seed"] for r in MEI["per_seed"]}
out["overlap_with_MEH"] = sorted(set(seeds) & meh)
out["overlap_with_MEI"] = sorted(set(seeds) & mei)
out["MEH_first_seed"] = min(meh)
out["MN0_seeds_subset_of_MN0X"] = {r["seed"] for r in MN0["per_seed"]} <= set(seeds)

# ---------------------------------------------------------------- per-channel
def channel(recs, ch):
    maps = [r[f"map_{ch}"] for r in recs]
    b = [m - H_TRUE for m in maps]
    n = len(b)
    mean = sum(b) / n
    sd0 = math.sqrt(sum((x - mean) ** 2 for x in b) / n)         # population sd (aggregate convention)
    sd1 = st.stdev(b)                                            # sample sd, ddof=1
    return {
        "n": n,
        "mean_bias": mean,
        "sd_pop": sd0,
        "sd_sample": sd1,
        "se_pop": sd0 / math.sqrt(n),
        "se_sample": sd1 / math.sqrt(n),
        "map_min": min(maps),
        "map_max": max(maps),
        "post_sd_median": st.median([r[f"post_sd_{ch}"] for r in recs]),
        "railed_low_frac": sum(bool(r[f"railed_low_{ch}"]) for r in recs) / n,
        "railed_high_frac": sum(bool(r[f"railed_high_{ch}"]) for r in recs) / n,
        "nonfinite_ln_post_seeds": sum(
            any(not math.isfinite(v) for v in r[f"ln_post_{ch}"]) for r in recs
        ),
        "edge_loaded_frac": sum(r[f"edge_mass_{ch}"] > 0.01 for r in recs) / n,
        "hpd90_cov": sum(r[f"hpd90_{ch}"] for r in recs) / n,
        "hpd68_cov": sum(r[f"hpd68_{ch}"] for r in recs) / n,
        "hpd50_cov": sum(r[f"hpd50_{ch}"] for r in recs) / n,
    }

for ch in ("1d", "2d"):
    out[f"MN0X_{ch}"] = channel(ps, ch)
    out[f"MN0_{ch}"] = channel(MN0["per_seed"], ch)

# ---------------------------------------------------------------- pins
out["K_sum_values"] = sorted({r["K_sum"] for r in ps})
out["K_sum_pin_ok"] = out["K_sum_values"] == [1193703]
out["n_events_values"] = sorted({r["n_events"] for r in ps})
out["n_events_run_values"] = sorted({r["n_events_run"] for r in ps})
out["n_horizon_dropped_max"] = max(r["n_horizon_dropped"] for r in ps)
out["f_incl_values"] = sorted({r["f_incl"] for r in ps})
out["sigma_z_mean_pairs_mean"] = sum(r["sigma_z_mean_pairs"] for r in ps) / len(ps)
out["sigma_z_mean_pairs_min"] = min(r["sigma_z_mean_pairs"] for r in ps)
out["sigma_z_mean_pairs_max"] = max(r["sigma_z_mean_pairs"] for r in ps)
out["pin_integrity_pass"] = MN0X["pin_integrity"]["pass"]
out["pin_crb_md5"] = MN0X["pin_integrity"]["crb_csv_md5"]["value"]
out["pin_frozeng_md5"] = MN0X["pin_integrity"]["frozeng_emit_md5"]["value"]
out["provenance"] = {
    k: MN0X.get(k)
    for k in ("instrument", "preregistration", "git_commit", "git_dirty",
              "import_path_clean", "allow_dirty", "smoke", "workers", "grain",
              "wall_time_s", "wall_time_per_seed_s")
}
out["dirt_import_path"] = MN0X.get("dirt_inventory", {}).get("import_path")
out["config_matches_MN0"] = {
    k: (MN0X["config"].get(k), MN0["config"].get(k))
    for k in MN0X["config"]
    if MN0X["config"].get(k) != MN0["config"].get(k)
}

# ---------------------------------------------------------------- A1-DET
shared = sorted({r["seed"] for r in MN0["per_seed"]} & set(seeds))
x_by = {r["seed"]: r for r in ps}
o_by = {r["seed"]: r for r in MN0["per_seed"]}
det = {"shared_seeds": shared, "n_shared": len(shared), "per_seed": {}}
worst = 0.0
map_fields = ("map_1d", "map_2d")
all_maps_equal = True
for s in shared:
    a, b = x_by[s], o_by[s]
    # 'cell' is the arm label, not a value: MN0X vs MN0 by construction. Compared
    # separately and reported, never scored as a values-golden mismatch.
    label_fields = {"cell"}
    fields = sorted((set(a) & set(b)) - label_fields)
    det["n_fields_compared"] = len(fields)
    det["label_fields_excluded"] = sorted(label_fields)
    det["label_field_values"] = {f: [a[f], b[f]] for f in sorted(label_fields)}
    seed_worst, worst_field = 0.0, None
    for f in fields:
        va, vb = a[f], b[f]
        if isinstance(va, list):
            pairs = list(zip(va, vb))
        else:
            pairs = [(va, vb)]
        for u, v in pairs:
            if isinstance(u, str) or isinstance(v, str):
                if u != v:
                    seed_worst, worst_field = float("inf"), f
                continue
            if u == v:
                continue
            d = abs(u - v) / max(abs(u), abs(v), 1e-300)
            if d > seed_worst:
                seed_worst, worst_field = d, f
    for f in map_fields:
        if a[f] != b[f]:
            all_maps_equal = False
    det["per_seed"][str(s)] = {
        "max_rel_dev": seed_worst,
        "worst_field": worst_field,
        "map_1d_equal": a["map_1d"] == b["map_1d"],
        "map_2d_equal": a["map_2d"] == b["map_2d"],
        "map_1d": a["map_1d"],
    }
    worst = max(worst, seed_worst)
det["max_rel_dev_over_all_shared_seeds"] = worst
det["all_maps_exactly_equal"] = all_maps_equal
det["rtol"] = 1e-12
det["verdict"] = "PASS" if (worst <= 1e-12 and all_maps_equal) else "FAIL"
out["A1_DET"] = det

# ---------------------------------------------------------------- decision rule
m1 = out["MN0X_1d"]["mean_bias"]
m2 = out["MN0X_2d"]["mean_bias"]
delta = m1 - REF
out["A1_rule"] = {
    "mean_bias_1d": m1,
    "reference": REF,
    "delta": delta,
    "abs_delta": abs(delta),
    "window": WINDOW,
    "margin": WINDOW - abs(delta),
    "times_inside_window": WINDOW / abs(delta) if delta else float("inf"),
    "verdict": "A1-PASS" if abs(delta) <= WINDOW else "A1-FAIL",
    "mean_bias_2d": m2,
    "abs_delta_2d_vs_1d_ref": abs(m2 - REF),
    "delta_2d_vs_2d_campaign": m2 - 0.039713,
}
se_diff = math.hypot(out["MN0X_1d"]["se_pop"], REF_SE)
out["A1_rule"]["se_pop_1d"] = out["MN0X_1d"]["se_pop"]
out["A1_rule"]["se_diff_vs_reference"] = se_diff
out["A1_rule"]["delta_in_sigma_diff"] = delta / se_diff
out["A1_rule"]["window_in_sigma_arm"] = WINDOW / out["MN0X_1d"]["se_pop"]
out["A1_rule"]["window_in_sigma_diff"] = WINDOW / se_diff

out["prediction_check"] = {
    "registered_prediction": PRED,
    "registered_prediction_se": PRED_SE,
    "measured": m1,
    "residual": m1 - PRED,
    "n_sigma_vs_prediction": (m1 - PRED) / PRED_SE,
    "inside_68pct_interval": 0.03629 <= m1 <= 0.03741,
}

# fresh-seed-only mean (the 85 seeds not in MN0) — falsification arithmetic of §5
fresh = [r for r in ps if r["seed"] not in o_by]
out["fresh_85"] = channel(fresh, "1d")
out["fresh_85"]["delta_vs_reference"] = out["fresh_85"]["mean_bias"] - REF
out["fresh_85"]["n_sigma_vs_reference"] = (
    (out["fresh_85"]["mean_bias"] - REF) / (0.0061154 / math.sqrt(85))
)

# grid quantisation check (§3 iii)
out["quantisation"] = {
    "mean_x_100_over_0.005": m1 * 100 / 0.005,
    "mean_2d_x_100_over_0.005": m2 * 100 / 0.005,
    "reference_in_ticks_at_N100": REF * 100 / 0.005,
}

(RUN / "A1_READOUT.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))

# ---------------------------------------------------------- extras (appended)
import math as _m


def _ks(pits):
    p = sorted(pits)
    n = len(p)
    return max(
        max(abs((i + 1) / n - v), abs(v - i / n)) for i, v in enumerate(p)
    )


extra = {}
for ch in ("1d", "2d"):
    extra[f"ks_D_{ch}"] = _ks([r[f"pit_{ch}"] for r in ps])
    extra[f"pit_max_{ch}"] = max(r[f"pit_{ch}"] for r in ps)
    extra[f"bias_over_post_sd_median_{ch}"] = (
        out[f"MN0X_{ch}"]["mean_bias"] / out[f"MN0X_{ch}"]["post_sd_median"]
    )

# 15 included vs 85 fresh — is the N=15 shortfall a fluctuation of this block?
m15 = out["MN0_1d"]["mean_bias"]
m85 = out["fresh_85"]["mean_bias"]
se15 = out["MN0_1d"]["se_pop"]
se85 = out["fresh_85"]["se_pop"]
extra["subset_15_vs_85"] = {
    "mean_15": m15,
    "mean_85": m85,
    "difference": m15 - m85,
    "se_difference": _m.hypot(se15, se85),
    "n_sigma": (m15 - m85) / _m.hypot(se15, se85),
}
# §5 registered falsification arithmetic, checked against the realized fresh mean
extra["fail_threshold_fresh_mean"] = 0.0353376
extra["fresh_mean_above_fail_threshold"] = m85 - 0.0353376
extra["fresh_mean_sigma_above_fail_threshold"] = (m85 - 0.0353376) / (0.0061154 / _m.sqrt(85))
# quantisation floor at N=100
extra["quantisation_tick_N100"] = 0.005 / 100
extra["max_quantisation_offset_N100"] = 0.005 / 200
extra["abs_delta_below_quantisation_floor"] = out["A1_rule"]["abs_delta"] < 0.005 / 200
# campaign 2D reference
extra["campaign_2d_reference"] = 0.039713
extra["abs_delta_2d"] = abs(out["MN0X_2d"]["mean_bias"] - 0.039713)
out["extra"] = extra
(RUN / "A1_READOUT.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(extra, indent=1))
