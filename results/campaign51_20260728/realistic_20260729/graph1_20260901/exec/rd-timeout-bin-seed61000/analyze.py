import sys, re, json, ast
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, "/home/jasper/Repositories/darksiren-emri")
from darksiren_emri.physical_relations import dist_to_redshift
import darksiren_emri.constants as const

BASE = Path("/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/seed61000")
FETCH = BASE / "cluster_logs_fetch_20260904"
LOGS = FETCH / "logs"
OUTDIR = Path("/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/rd-timeout-bin-seed61000")
OUTDIR.mkdir(parents=True, exist_ok=True)
S3000_DIR = OUTDIR.parent / "rd-timeout-bin-seed3000"

# ---------------------------------------------------------------------------
# 0. Existence contract.
# ---------------------------------------------------------------------------
import hashlib
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

kept_csv = BASE / "prepared_cramer_rao_bounds.csv"
KEPT_MD5_PIN = "9a1f2a14384a9281c97ca3be312ddaab"
kept_md5_actual = md5(kept_csv)
assert kept_md5_actual == KEPT_MD5_PIN, f"CRB md5 MISMATCH: {kept_md5_actual} != pin {KEPT_MD5_PIN}"

manifest = FETCH.parent / "cluster_logs_fetch_20260904_MANIFEST.md5"
manifest_lines = manifest.read_text().splitlines()
n_manifest = len(manifest_lines)

sim_err_files = sorted(LOGS.glob("simulate_6088772_*.err"))
sim_out_files = sorted(LOGS.glob("simulate_6088772_*.out"))
n_sim_err = len(sim_err_files)
n_sim_out = len(sim_out_files)

# ---------------------------------------------------------------------------
# 1. Parse timeout log lines (both stages) from the simulate array's stderr.
#    (Confirmed empirically: timeout param-dict records land in
#    logs/simulate_6088772_<idx>.err, NOT in .out or the top-level
#    master_thesis_code_*.log app files, which are dominated by mixed
#    simulate+evaluate runs sharing the default h=0.73 filename suffix.)
# ---------------------------------------------------------------------------
SNR_PAT = re.compile(r"Waveform/SNR computation timed out \(>90s\)\. Skipping event\.\.\. params=(\{.*\})\s*$")
CRB_PAT = re.compile(r"Cram\S+r-Rao bound computation timed out \(>90s\)\. Skipping event\.\.\. params=(\{.*\})\s*$")

def parse_params(blob: str) -> dict:
    blob2 = re.sub(r"np\.float64\(([^)]*)\)", r"\1", blob)
    return ast.literal_eval(blob2)

records = []
for lf in sim_err_files:
    idx = int(re.search(r"simulate_6088772_(\d+)\.err", lf.name).group(1))
    text = lf.read_text(errors="replace")
    for line in text.splitlines():
        m = SNR_PAT.search(line)
        if m:
            p = parse_params(m.group(1)); p["stage"] = "snr"; p["task_idx"] = idx; records.append(p); continue
        m = CRB_PAT.search(line)
        if m:
            p = parse_params(m.group(1)); p["stage"] = "crb"; p["task_idx"] = idx; records.append(p)

timeouts = pd.DataFrame(records)
n_snr_to = int((timeouts.stage == "snr").sum())
n_crb_to = int((timeouts.stage == "crb").sum())

# Non-timeout skip categories seen (descriptive; no per-event params logged for these).
err_blob_sample = "\n".join(f.read_text(errors="replace") for f in sim_err_files[:100])
other_errors = {}
for tag in ["ParameterOutOfBoundsError", "ZeroDivisionError", "EllipticK", "Brent", "SeparatrixSigns",
            "LinAlgError", "Warning"]:
    other_errors[tag] = len(re.findall(tag, err_blob_sample))
n_skip_tally_lines = sum(1 for f in sim_err_files for line in f.read_text(errors="replace").splitlines()
                          if "Skip tally" in line)

# ---------------------------------------------------------------------------
# 2. Kept (successful, prepared CRB) population.
# ---------------------------------------------------------------------------
kept = pd.read_csv(kept_csv)
n_kept = len(kept)

z_kept = kept["luminosity_distance"].apply(lambda d: dist_to_redshift(float(d), h=0.73))
z_summary = {
    "n": int(len(z_kept)),
    "min": float(z_kept.min()),
    "median": float(z_kept.median()),
    "p95": float(z_kept.quantile(0.95)),
    "max": float(z_kept.max()),
    "HOST_DRAW_Z_MAX": float(const.HOST_DRAW_Z_MAX),
    "depth_fraction_of_zmax": float(z_kept.max() / const.HOST_DRAW_Z_MAX),
    "frac_events_z_gt_0.9_zmax": float((z_kept > 0.9 * const.HOST_DRAW_Z_MAX).mean()),
}

# ---------------------------------------------------------------------------
# 3. Bin edges. p0's bound changed between the two runs (seed3000 used the
#    now-RETIRED [10,16] snapshot-mode prior; production seed61000 draws p0
#    via the unclamped plunge-window convention, HIGHM_AUDIT.md item 1,
#    2026-07-28) -- observed p0 support here is [3.68, 87.22], not [10,16].
#    M's declared support also differs (detector-frame lift by (1+z) raises
#    the upper edge). Because the injected ranges genuinely differ, we do
#    NOT reuse seed3000's numeric edges as the primary table (would silently
#    truncate/distort the seed61000 p0 tail); instead we apply the SAME BLIND
#    quantile/log rule fresh to the seed61000 union population, and ALSO
#    report a secondary comparison table using seed3000's original M edges
#    verbatim (for direct cross-run comparability on the M axis specifically,
#    where the rule -- not the domain -- is what needs to match).
#    Frozen BEFORE any timeout rate/count is inspected.
# ---------------------------------------------------------------------------
pop_M = pd.concat([kept["M"], timeouts["M"]], ignore_index=True).astype(float)
pop_e0 = pd.concat([kept["e0"], timeouts["e0"]], ignore_index=True).astype(float)
pop_p0 = pd.concat([kept["p0"], timeouts["p0"]], ignore_index=True).astype(float)

N_BINS = 5
M_edges = np.logspace(np.log10(pop_M.min()), np.log10(pop_M.max()), N_BINS + 1)
M_edges[0] *= 0.999999; M_edges[-1] *= 1.000001
e0_edges = np.quantile(pop_e0, np.linspace(0, 1, N_BINS + 1))
e0_edges[0] *= 0.999999; e0_edges[-1] *= 1.000001
p0_edges = np.quantile(pop_p0, np.linspace(0, 1, N_BINS + 1))
p0_edges[0] *= 0.999999; p0_edges[-1] *= 1.000001

# seed3000's frozen M edges, reused verbatim for the direct comparison table.
s3000_edges = json.loads((S3000_DIR / "design_gate_bin_edges.json").read_text())
M_edges_s3000 = np.array(s3000_edges["M_edges"])

design_gate = {
    "rule_primary": "Same blind rule as seed3000 (M: log-spaced 5 bins equal-width in log10 M; "
                    "e0, p0: quantile/quintile 5 bins), applied FRESH to the seed61000 union(kept, "
                    "timeout) population -- NOT a verbatim reuse of seed3000's numeric edges, because "
                    "the p0 prior changed (seed3000 = retired [10,16] snapshot-mode bound; seed61000 "
                    "= production plunge-window convention, unclamped upper p0) and M's detector-frame "
                    "support differs. Frozen BEFORE any timeout rate/count is computed or inspected.",
    "rule_secondary_M_only": "seed3000's original M edges reused VERBATIM (not re-derived) for a "
                    "direct cross-run comparison table on the M axis only (state so).",
    "population_source": "union of prepared_cramer_rao_bounds.csv kept events (n=%d) and simulate_6088772_*.err "
                          "timeout log records (n=%d, both stages)" % (n_kept, len(timeouts)),
    "seed61000_M_edges": M_edges.tolist(),
    "seed61000_e0_edges": e0_edges.tolist(),
    "seed61000_p0_edges": p0_edges.tolist(),
    "seed3000_M_edges_reused_for_comparison": M_edges_s3000.tolist(),
    "seed3000_p0_edges_for_reference_NOT_reused": s3000_edges["p0_edges"],
    "p0_support_note": {
        "seed3000_p0_range_pop": [float(0.05), None],  # not applicable; see p0 below
        "seed61000_observed_p0_range_kept": [float(kept.p0.min()), float(kept.p0.max())],
        "seed61000_observed_p0_range_timeouts": [float(timeouts.p0.min()), float(timeouts.p0.max())],
        "reason": "seed3000 p0 prior = retired SNAPSHOT-mode bound [10,16] (few's Pn5AAK input domain); "
                  "seed61000 (production, post 2026-07-28 flip) draws p0 via the plunge-window convention "
                  "with no upper clamp -- HIGHM_AUDIT.md item 1.",
    },
}
(OUTDIR / "design_gate_bin_edges.json").write_text(json.dumps(design_gate, indent=2))
print("DESIGN GATE FROZEN (bin edges, before any rate is read):")
print(json.dumps(design_gate, indent=2))

# ---------------------------------------------------------------------------
# 4. Rates.
# ---------------------------------------------------------------------------
def garwood_interval(k: int, conf: float = 0.95):
    alpha = 1 - conf
    lo = 0.0 if k == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * k)
    hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1))
    return lo, hi

def bin_index(values, edges):
    return np.digitize(values, edges[1:-1], right=False)

def rate_table_1d(edges, kept_vals, to_vals):
    n_bins = len(edges) - 1
    kept_bin = bin_index(kept_vals, edges)
    to_bin = bin_index(to_vals, edges)
    rows = []
    for b in range(n_bins):
        n_k = int((kept_bin == b).sum())
        n_t = int((to_bin == b).sum())
        denom = n_k + n_t
        rate = n_t / denom if denom > 0 else float("nan")
        lo, hi = garwood_interval(n_t)
        rate_lo = lo / denom if denom > 0 else float("nan")
        rate_hi = hi / denom if denom > 0 else float("nan")
        se = (rate_hi - rate_lo) / (2 * 1.959964) if denom > 0 else float("nan")
        rows.append({"bin": b, "edge_lo": float(edges[b]), "edge_hi": float(edges[b + 1]),
                     "n_kept": n_k, "n_timeout": n_t, "denom": denom, "rate": rate,
                     "rate_garwood_lo95": rate_lo, "rate_garwood_hi95": rate_hi,
                     "se_approx_from_garwood": se})
    df = pd.DataFrame(rows)
    max_grad, pair = 0.0, None
    for i in range(n_bins - 1):
        j = i + 1
        if df.loc[i, "denom"] == 0 or df.loc[j, "denom"] == 0:
            continue
        se_i, se_j = df.loc[i, "se_approx_from_garwood"], df.loc[j, "se_approx_from_garwood"]
        combined = np.sqrt(se_i**2 + se_j**2)
        if combined == 0 or np.isnan(combined):
            continue
        grad = abs(df.loc[i, "rate"] - df.loc[j, "rate"]) / combined
        if grad > max_grad:
            max_grad, pair = grad, (i, j)
    return df, float(max_grad), pair

snr_numerator = timeouts[timeouts.stage == "snr"]
snr_not_to = pd.concat([kept, timeouts[timeouts.stage == "crb"]], ignore_index=True)

axis_tables = {}
for axis, edges in [("M", M_edges), ("e0", e0_edges), ("p0", p0_edges)]:
    kv = snr_not_to[axis].astype(float).values
    tv = snr_numerator[axis].astype(float).values
    df, max_grad, pair = rate_table_1d(edges, kv, tv)
    axis_tables[axis] = {"table": df, "max_gradient_sigma": max_grad, "max_gradient_pair": pair}
    print(f"\n=== SNR-stage timeout rate by {axis} (seed61000-native edges) ===")
    print(df.to_string(index=False))
    print(f"max adjacent-bin gradient: {max_grad:.3f} sigma (bins {pair})")

# secondary: M axis using seed3000's verbatim edges (direct comparison table)
kv_M = snr_not_to["M"].astype(float).values
tv_M = snr_numerator["M"].astype(float).values
df_M_s3000edges, max_grad_M_s3000edges, pair_M_s3000edges = rate_table_1d(M_edges_s3000, kv_M, tv_M)
print("\n=== SNR-stage timeout rate by M (seed3000's verbatim edges, comparison table) ===")
print(df_M_s3000edges.to_string(index=False))

# 2-D (M, p0) grid, seed61000-native edges
M_bin_kept = bin_index(snr_not_to["M"].astype(float).values, M_edges)
p0_bin_kept = bin_index(snr_not_to["p0"].astype(float).values, p0_edges)
M_bin_to = bin_index(snr_numerator["M"].astype(float).values, M_edges)
p0_bin_to = bin_index(snr_numerator["p0"].astype(float).values, p0_edges)
cell_rows = []
for mi in range(N_BINS):
    for pi in range(N_BINS):
        n_k = int(((M_bin_kept == mi) & (p0_bin_kept == pi)).sum())
        n_t = int(((M_bin_to == mi) & (p0_bin_to == pi)).sum())
        denom = n_k + n_t
        rate = n_t / denom if denom > 0 else float("nan")
        lo, hi = garwood_interval(n_t)
        rate_lo = lo / denom if denom > 0 else float("nan")
        rate_hi = hi / denom if denom > 0 else float("nan")
        se = (rate_hi - rate_lo) / (2 * 1.959964) if denom > 0 else float("nan")
        cell_rows.append({"M_bin": mi, "p0_bin": pi, "n_kept": n_k, "n_timeout": n_t, "denom": denom,
                           "rate": rate, "rate_garwood_lo95": rate_lo, "rate_garwood_hi95": rate_hi,
                           "se_approx": se})
cell_df = pd.DataFrame(cell_rows)
print("\n=== SNR-stage timeout rate, 2-D (M, p0) grid (seed61000-native edges) ===")
print(cell_df.to_string(index=False))

max_grad_2d, max_grad_2d_pair = 0.0, None
grid = {(r.M_bin, r.p0_bin): r for r in cell_df.itertuples()}
for (mi, pi), r in grid.items():
    if r.denom == 0:
        continue
    for dm, dp in [(1, 0), (0, 1)]:
        nb = grid.get((mi + dm, pi + dp))
        if nb is None or nb.denom == 0:
            continue
        combined = np.sqrt(r.se_approx**2 + nb.se_approx**2)
        if combined == 0 or np.isnan(combined):
            continue
        grad = abs(r.rate - nb.rate) / combined
        if grad > max_grad_2d:
            max_grad_2d, max_grad_2d_pair = grad, ((mi, pi), (mi + dm, pi + dp))
print(f"\nmax adjacent-cell gradient (2D M x p0 grid): {max_grad_2d:.3f} sigma (cells {max_grad_2d_pair})")

# CRB-stage (n=2, descriptive only)
crb_to_events = timeouts[timeouts.stage == "crb"][["M", "e0", "p0", "task_idx"]].to_dict("records")
crb_denom = n_kept + n_crb_to
crb_rate = n_crb_to / crb_denom
crb_lo, crb_hi = garwood_interval(n_crb_to)

# Aggregate SNR-stage
n_snr_kept = len(snr_not_to)
snr_denom = n_snr_kept + n_snr_to
snr_rate = n_snr_to / snr_denom
snr_lo, snr_hi = garwood_interval(n_snr_to)

overall_max_grad = max(axis_tables["M"]["max_gradient_sigma"], axis_tables["e0"]["max_gradient_sigma"],
                        axis_tables["p0"]["max_gradient_sigma"], max_grad_2d)

print("\n=== SUMMARY ===")
print(f"SNR-stage: {n_snr_to}/{snr_denom} = {snr_rate:.5f}, Garwood95 rate CI "
      f"[{snr_lo/snr_denom:.5f},{snr_hi/snr_denom:.5f}]")
print(f"CRB-stage: {n_crb_to}/{crb_denom} = {crb_rate:.5f} (n=2, descriptive)")
print(f"Max gradient across all axes/grid: {overall_max_grad:.3f} sigma")
print(f"z-depth: {json.dumps(z_summary, indent=2)}")

# ---------------------------------------------------------------------------
# 5. seed3000 vs seed61000 direct comparison (per seed3000's own M bins).
# ---------------------------------------------------------------------------
s3000_M_table = pd.read_csv(S3000_DIR / "rate_table_M.csv")
comparison_rows = []
for b in range(N_BINS):
    r3000 = s3000_M_table.iloc[b]
    r61000 = df_M_s3000edges.iloc[b]
    se_combined = np.sqrt(r3000["se_approx_from_garwood"] ** 2 + r61000["se_approx_from_garwood"] ** 2)
    diff_sigma = (abs(r3000["rate"] - r61000["rate"]) / se_combined) if se_combined > 0 and not np.isnan(se_combined) else float("nan")
    comparison_rows.append({
        "M_bin": b, "edge_lo": r3000["edge_lo"], "edge_hi": r3000["edge_hi"],
        "seed3000_n_kept": int(r3000["n_kept"]), "seed3000_n_timeout": int(r3000["n_timeout"]),
        "seed3000_rate": r3000["rate"],
        "seed61000_n_kept": int(r61000["n_kept"]), "seed61000_n_timeout": int(r61000["n_timeout"]),
        "seed61000_rate": r61000["rate"],
        "diff_sigma": diff_sigma,
    })
comparison_df = pd.DataFrame(comparison_rows)
print("\n=== DIRECT COMPARISON: seed3000 vs seed61000, per seed3000's frozen M bins ===")
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Selection-effect note (FACTS only): mass lost to timeouts per bin vs
#    the fraction of the kept (inference) population in each bin.
# ---------------------------------------------------------------------------
sel_rows = []
for _, row in axis_tables["M"]["table"].iterrows():
    sel_rows.append({
        "M_bin": int(row["bin"]), "edge_lo": row["edge_lo"], "edge_hi": row["edge_hi"],
        "frac_of_bin_lost_to_timeout": row["rate"],
        "frac_of_kept_population_in_bin": row["n_kept"] / n_kept,
    })
selection_df = pd.DataFrame(sel_rows)
print("\n=== Selection-effect note: fraction lost to timeout vs fraction of kept (H0-inference) population ===")
print(selection_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Persist.
# ---------------------------------------------------------------------------
def band_call(max_grad):
    return "NEW_SYSTEMATIC_CANDIDATE (>3sigma)" if max_grad > 3.0 else "NON_ISSUE_WITHIN_POISSON_BOUND"

result = {
    "node": "rd-timeout-bin-seed61000",
    "existence_contract": {
        "prepared_cramer_rao_bounds_csv": {"status": "PRESENT", "n_rows": n_kept,
                                            "md5_pin": KEPT_MD5_PIN, "md5_actual": kept_md5_actual,
                                            "md5_match": kept_md5_actual == KEPT_MD5_PIN},
        "manifest_md5_file": {"status": "PRESENT", "n_lines": n_manifest},
        "fetched_files_total": 2194,
        "simulate_array_job": "6088772 (100 tasks, seed 61000+task_idx, sim_steps=40/task = 4000 draws)",
        "simulate_err_files": {"status": "PRESENT", "n": n_sim_err},
        "simulate_out_files": {"status": "PRESENT", "n": n_sim_out},
        "n_snr_stage_timeout_records": n_snr_to,
        "n_crb_stage_timeout_records": n_crb_to,
        "n_skip_tally_lines_found": n_skip_tally_lines,
        "other_skip_category_string_counts_first100_err_files": other_errors,
        "other_top_level_dirs_in_fetch_NOT_used": ["real_r1..r5", "sig0_control", "zoom",
            "estimatorB_2x2", "top-level master_thesis_code_*.log (mixed simulate+evaluate, "
            "default h=0.73 filename suffix on both stages -- ambiguous, not used)"],
    },
    "design_gate": design_gate,
    "snr_stage": {
        "aggregate": {"n_timeout": n_snr_to, "denom": snr_denom, "rate": snr_rate,
                      "garwood95_count_ci": [snr_lo, snr_hi],
                      "garwood95_rate_ci": [snr_lo / snr_denom, snr_hi / snr_denom]},
        "by_axis_native_edges": {
            axis: {"table": axis_tables[axis]["table"].to_dict("records"),
                   "max_gradient_sigma": axis_tables[axis]["max_gradient_sigma"],
                   "max_gradient_pair_bins": axis_tables[axis]["max_gradient_pair"],
                   "band_call": band_call(axis_tables[axis]["max_gradient_sigma"])}
            for axis in ["M", "e0", "p0"]
        },
        "M_axis_seed3000_edges_comparison_table": df_M_s3000edges.to_dict("records"),
        "grid_2d_M_p0_native_edges": {"table": cell_df.to_dict("records"),
                                       "max_gradient_sigma": max_grad_2d,
                                       "max_gradient_pair_cells": max_grad_2d_pair,
                                       "band_call": band_call(max_grad_2d)},
    },
    "crb_stage_descriptive": {"n_timeout": n_crb_to, "denom": crb_denom, "rate": crb_rate,
                               "garwood95_count_ci": [crb_lo, crb_hi],
                               "garwood95_rate_ci": [crb_lo / crb_denom, crb_hi / crb_denom],
                               "events": crb_to_events,
                               "note": "n=2, not binnable; reported descriptively only"},
    "population_depth_z": z_summary,
    "overall_max_gradient_sigma": overall_max_grad,
    "overall_band_call": band_call(overall_max_grad),
    "seed3000_vs_seed61000_comparison": comparison_df.to_dict("records"),
    "selection_effect_note": selection_df.to_dict("records"),
    "gaps": [
        "Raw injected-population files were explicitly excluded from the fetch (OPS_RECORD step 4: "
        "'NOT injections/posteriors') -- bin edges use the union of kept + timeout parameters as the "
        "population proxy, same convention as seed3000, disclosed not substituted silently.",
        "p0's declared prior differs between the two runs: seed3000 used the retired [10,16] "
        "SNAPSHOT-mode bound; seed61000 (production) draws p0 via the unclamped plunge-window "
        "convention (HIGHM_AUDIT.md item 1, 2026-07-28). Observed seed61000 p0 support is "
        f"[{float(pop_p0.min()):.2f}, {float(pop_p0.max()):.2f}], not [10,16]. seed3000's M edges "
        "(not its p0 edges) are reused verbatim for the cross-run comparison table; p0/e0 tables use "
        "fresh seed61000-native edges only.",
        "Zero of the 100 simulate_6088772 tasks logged a final 'Skip tally' summary line (all were "
        "cancelled at walltime before reaching it) -- worse completeness than seed3000 (33/100); does "
        "not affect the binned rate (built from raw per-event records), disclosed as a caveat.",
        "Denominator excludes all non-timeout skip categories (ParameterOutOfBoundsError, "
        "ZeroDivisionError, ...) -- no per-event params logged for those (G9 instruments only the two "
        "timeout call sites). Reported rate is conditional on {timed out, fully succeeded}, not of "
        "every attempted draw.",
        "CRB-stage timeout sample (n=2) is descriptive only, not binned.",
        "The fetch directory also contains real_r1..r5, sig0_control, zoom, estimatorB_2x2 subtrees "
        "and top-level master_thesis_code_*.log files (mixed simulate+evaluate app logs, ambiguous by "
        "filename alone) belonging to the same run_20260729_seed61000 workspace directory but NOT to "
        "the seed61000 simulate array (job 6088772) analyzed here -- excluded from this read.",
    ],
}
(OUTDIR / "READ_RECORD.json").write_text(json.dumps(result, indent=2, default=str))
axis_tables["M"]["table"].to_csv(OUTDIR / "rate_table_M.csv", index=False)
axis_tables["e0"]["table"].to_csv(OUTDIR / "rate_table_e0.csv", index=False)
axis_tables["p0"]["table"].to_csv(OUTDIR / "rate_table_p0.csv", index=False)
cell_df.to_csv(OUTDIR / "rate_table_2d_M_p0.csv", index=False)
df_M_s3000edges.to_csv(OUTDIR / "rate_table_M_seed3000edges_comparison.csv", index=False)
comparison_df.to_csv(OUTDIR / "comparison_seed3000_vs_seed61000.csv", index=False)
selection_df.to_csv(OUTDIR / "selection_effect_note.csv", index=False)
print("\nWrote:", OUTDIR / "READ_RECORD.json")
