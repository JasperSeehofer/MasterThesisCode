import sys, re, json, ast
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, "/home/jasper/Repositories/darksiren-emri")
from darksiren_emri.physical_relations import dist_to_redshift
import darksiren_emri.constants as const

ARCHIVE = Path("/home/jasper/Repositories/darksiren-emri/results/_archive/run_20260707_seed3000")
OUTDIR = Path("/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/rd-timeout-bin-seed3000")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Parse timeout log lines (both stages) with full parameter dicts.
# ---------------------------------------------------------------------------
SNR_PAT = re.compile(r"Waveform/SNR computation timed out \(>90s\)\. Skipping event\.\.\. params=(\{.*\})\s*$")
CRB_PAT = re.compile(r"Cram\S+r-Rao bound computation timed out \(>90s\)\. Skipping event\.\.\. params=(\{.*\})\s*$")

def parse_params(blob: str) -> dict:
    # params dict text contains np.float64(...) wrappers -- neutralize for literal_eval.
    blob2 = re.sub(r"np\.float64\(([^)]*)\)", r"\1", blob)
    return ast.literal_eval(blob2)

log_files = sorted(ARCHIVE.rglob("*.log"))
assert len(log_files) == 99, f"expected 99 logs (recursive), found {len(log_files)}"

records = []
for lf in log_files:
    text = lf.read_text(errors="replace")
    for line in text.splitlines():
        m = SNR_PAT.search(line)
        if m:
            p = parse_params(m.group(1))
            p["stage"] = "snr"
            p["log_file"] = lf.name
            records.append(p)
            continue
        m = CRB_PAT.search(line)
        if m:
            p = parse_params(m.group(1))
            p["stage"] = "crb"
            p["log_file"] = lf.name
            records.append(p)

timeouts = pd.DataFrame(records)
n_snr_to = int((timeouts.stage == "snr").sum())
n_crb_to = int((timeouts.stage == "crb").sum())
assert len(timeouts) == 1198, f"expected 1198 timeout records, parsed {len(timeouts)}"
assert n_snr_to == 1196 and n_crb_to == 2, (n_snr_to, n_crb_to)

# ---------------------------------------------------------------------------
# 2. Kept (successful, CRB-CSV) population -- the "made it through" denominator.
# ---------------------------------------------------------------------------
crb_csv = ARCHIVE / "simulations" / "cramer_rao_bounds.csv"
kept = pd.read_csv(crb_csv)
assert len(kept) == 3325, f"expected 3325 CRB rows, found {len(kept)}"

# Redshift of every kept event (h=0.73, fiducial default cosmology) -- population depth.
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
# 3. FREEZE bin edges BLIND -- population quantiles / log-spacing, computed
#    from the (M, e0, p0) support alone, before any timeout rate is read.
#    Population = kept UNION timeout (both stages), i.e. every event whose
#    parameters are actually known -- the only honest local proxy for the
#    "injected distribution" (raw injection CSVs are unreachable: they are
#    dangling symlinks to a cluster-only workspace path, see gaps).
# ---------------------------------------------------------------------------
pop_M = pd.concat([kept["M"], timeouts["M"]], ignore_index=True).astype(float)
pop_e0 = pd.concat([kept["e0"], timeouts["e0"]], ignore_index=True).astype(float)
pop_p0 = pd.concat([kept["p0"], timeouts["p0"]], ignore_index=True).astype(float)

N_BINS = 5
# M: log-spaced (equal-width in log10 M) over the observed support.
M_edges = np.logspace(np.log10(pop_M.min()), np.log10(pop_M.max()), N_BINS + 1)
M_edges[0] *= 0.999999  # guard against a min-value event landing left of bin 0
M_edges[-1] *= 1.000001
# e0, p0: quantile (quintile) bins over the observed support.
e0_edges = np.quantile(pop_e0, np.linspace(0, 1, N_BINS + 1))
e0_edges[0] *= 0.999999
e0_edges[-1] *= 1.000001
p0_edges = np.quantile(pop_p0, np.linspace(0, 1, N_BINS + 1))
p0_edges[0] *= 0.999999
p0_edges[-1] *= 1.000001

design_gate = {
    "rule": "M: log-spaced 5 bins over union(kept,timeout) support (equal width in log10 M). "
            "e0, p0: quantile (quintile) 5 bins over the same union population. "
            "Frozen BEFORE any timeout rate/count is computed or inspected.",
    "population_source": "union of CRB-CSV kept events (n=%d) and timeout log records (n=%d, both stages)"
                          % (len(kept), len(timeouts)),
    "M_edges": M_edges.tolist(),
    "e0_edges": e0_edges.tolist(),
    "p0_edges": p0_edges.tolist(),
}
(OUTDIR / "design_gate_bin_edges.json").write_text(json.dumps(design_gate, indent=2))
print("DESIGN GATE FROZEN (bin edges, before any rate is read):")
print(json.dumps(design_gate, indent=2))

# ---------------------------------------------------------------------------
# 4. Read the rates. Denominator per stage = events with KNOWN parameters
#    that entered that stage: kept (CRB-CSV, entered+passed every stage) plus
#    the timeout events at-or-after that stage (snr-stage denominator also
#    includes crb-stage timeouts, since they passed the snr stage before
#    failing at crb; crb-stage denominator is kept + crb-stage timeouts only).
#    Other skip categories (ZeroDivisionError, ParameterOutOfBoundsError, ...)
#    have NO per-event parameters logged (G9 instrumented only the timeout
#    sites) so they cannot be placed in a bin -- excluded by construction,
#    disclosed under gaps.
# ---------------------------------------------------------------------------
def garwood_interval(k: int, conf: float = 0.95):
    """Exact Poisson (Garwood) CI on a count k."""
    alpha = 1 - conf
    lo = 0.0 if k == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * k)
    hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1))
    return lo, hi

def bin_index(values, edges):
    # np.digitize with right-inclusive top bin
    idx = np.digitize(values, edges[1:-1], right=False)
    return idx  # 0..N_BINS-1

def rate_table_1d(axis_name, edges, kept_vals, to_vals):
    kept_bin = bin_index(kept_vals, edges)
    to_bin = bin_index(to_vals, edges)
    rows = []
    for b in range(N_BINS):
        n_kept = int((kept_bin == b).sum())
        n_to = int((to_bin == b).sum())
        denom = n_kept + n_to
        rate = n_to / denom if denom > 0 else float("nan")
        lo, hi = garwood_interval(n_to)
        rate_lo = lo / denom if denom > 0 else float("nan")
        rate_hi = hi / denom if denom > 0 else float("nan")
        se_approx = (rate_hi - rate_lo) / (2 * 1.959964) if denom > 0 else float("nan")
        rows.append({
            "bin": b, "edge_lo": float(edges[b]), "edge_hi": float(edges[b + 1]),
            "n_kept": n_kept, "n_timeout": n_to, "denom": denom,
            "rate": rate, "rate_garwood_lo95": rate_lo, "rate_garwood_hi95": rate_hi,
            "se_approx_from_garwood": se_approx,
        })
    df = pd.DataFrame(rows)
    # max adjacent-bin gradient in sigma (only over bins with denom>0 on both sides)
    max_grad = 0.0
    max_grad_pair = None
    for i in range(N_BINS - 1):
        j = i + 1
        if df.loc[i, "denom"] == 0 or df.loc[j, "denom"] == 0:
            continue
        se_i, se_j = df.loc[i, "se_approx_from_garwood"], df.loc[j, "se_approx_from_garwood"]
        combined_se = np.sqrt(se_i**2 + se_j**2)
        if combined_se == 0 or np.isnan(combined_se):
            continue
        grad = abs(df.loc[i, "rate"] - df.loc[j, "rate"]) / combined_se
        if grad > max_grad:
            max_grad = grad
            max_grad_pair = (i, j)
    return df, float(max_grad), max_grad_pair

# ---- SNR-stage: numerator = snr-stage timeouts ONLY (1196). "Did not time out
#      at the snr stage" population = kept (passed both stages) UNION the 2
#      crb-stage timeouts (they passed the snr stage, then failed at crb) ----
snr_numerator = timeouts[timeouts.stage == "snr"]
snr_not_to = pd.concat([kept, timeouts[timeouts.stage == "crb"]], ignore_index=True)
axis_tables = {}
for axis, edges in [("M", M_edges), ("e0", e0_edges), ("p0", p0_edges)]:
    kv = snr_not_to[axis].astype(float).values
    tv = snr_numerator[axis].astype(float).values
    df, max_grad, pair = rate_table_1d(axis, edges, kv, tv)
    axis_tables[axis] = {"table": df, "max_gradient_sigma": max_grad, "max_gradient_pair": pair}
    print(f"\n=== SNR-stage timeout rate by {axis} ===")
    print(df.to_string(index=False))
    print(f"max adjacent-bin gradient: {max_grad:.3f} sigma (bins {pair})")

# ---- 2-D (M, p0) cell (snr-stage numerator vs snr-stage "not timed out" population) ----
M_bin_kept = bin_index(snr_not_to["M"].astype(float).values, M_edges)
p0_bin_kept = bin_index(snr_not_to["p0"].astype(float).values, p0_edges)
M_bin_to = bin_index(snr_numerator["M"].astype(float).values, M_edges)
p0_bin_to = bin_index(snr_numerator["p0"].astype(float).values, p0_edges)

cell_rows = []
for mi in range(N_BINS):
    for pi in range(N_BINS):
        n_kept = int(((M_bin_kept == mi) & (p0_bin_kept == pi)).sum())
        n_to = int(((M_bin_to == mi) & (p0_bin_to == pi)).sum())
        denom = n_kept + n_to
        rate = n_to / denom if denom > 0 else float("nan")
        lo, hi = garwood_interval(n_to)
        rate_lo = lo / denom if denom > 0 else float("nan")
        rate_hi = hi / denom if denom > 0 else float("nan")
        se = (rate_hi - rate_lo) / (2 * 1.959964) if denom > 0 else float("nan")
        cell_rows.append({"M_bin": mi, "p0_bin": pi, "n_kept": n_kept, "n_timeout": n_to,
                           "denom": denom, "rate": rate, "rate_garwood_lo95": rate_lo,
                           "rate_garwood_hi95": rate_hi, "se_approx": se})
cell_df = pd.DataFrame(cell_rows)
print("\n=== SNR-stage timeout rate, 2-D (M, p0) grid ===")
print(cell_df.to_string(index=False))

# max gradient over adjacent 2D cells (4-neighbour)
max_grad_2d = 0.0
max_grad_2d_pair = None
grid = {(r.M_bin, r.p0_bin): r for r in cell_df.itertuples()}
for (mi, pi), r in grid.items():
    if r.denom == 0:
        continue
    for (dm, dp) in [(1, 0), (0, 1)]:
        nb = grid.get((mi + dm, pi + dp))
        if nb is None or nb.denom == 0:
            continue
        se_c = r.se_approx
        se_n = nb.se_approx
        combined = np.sqrt(se_c**2 + se_n**2)
        if combined == 0 or np.isnan(combined):
            continue
        grad = abs(r.rate - nb.rate) / combined
        if grad > max_grad_2d:
            max_grad_2d = grad
            max_grad_2d_pair = ((mi, pi), (mi + dm, pi + dp))
print(f"\nmax adjacent-cell gradient (2D M x p0 grid): {max_grad_2d:.3f} sigma (cells {max_grad_2d_pair})")

# ---- CRB-stage (n=2, descriptive only -- not binnable) ----
crb_to_events = timeouts[timeouts.stage == "crb"][["M", "e0", "p0", "log_file"]].to_dict("records")
n_crb_kept = len(kept)  # CRB-CSV rows = passed BOTH stages
crb_denom = n_crb_kept + n_crb_to
crb_rate = n_crb_to / crb_denom
crb_lo, crb_hi = garwood_interval(n_crb_to)
print(f"\n=== CRB-stage timeouts (descriptive, n={n_crb_to}, not binnable) ===")
print(f"events: {crb_to_events}")
print(f"aggregate rate: {n_crb_to}/{crb_denom} = {crb_rate:.5f}, "
      f"Garwood95 count CI [{crb_lo:.3f},{crb_hi:.3f}] -> rate CI "
      f"[{crb_lo/crb_denom:.5f},{crb_hi/crb_denom:.5f}]")

# ---------------------------------------------------------------------------
# 5. Aggregate rate + verdict
# ---------------------------------------------------------------------------
n_snr_kept = len(snr_not_to)  # kept (3325) + crb-stage timeouts (2), both "passed" the snr stage
snr_denom = n_snr_kept + n_snr_to
snr_rate = n_snr_to / snr_denom
snr_lo, snr_hi = garwood_interval(n_snr_to)

overall_max_grad = max(
    axis_tables["M"]["max_gradient_sigma"],
    axis_tables["e0"]["max_gradient_sigma"],
    axis_tables["p0"]["max_gradient_sigma"],
    max_grad_2d,
)

print("\n=== SUMMARY ===")
print(f"SNR-stage: {n_snr_to}/{snr_denom} = {snr_rate:.5f}, Garwood95 rate CI "
      f"[{snr_lo/snr_denom:.5f},{snr_hi/snr_denom:.5f}]")
print(f"CRB-stage: {n_crb_to}/{crb_denom} = {crb_rate:.5f} (n=2, descriptive)")
print(f"Max gradient across all axes/grid: {overall_max_grad:.3f} sigma")
print(f"z-depth: {json.dumps(z_summary, indent=2)}")

# ---------------------------------------------------------------------------
# 6. Persist full output.
# ---------------------------------------------------------------------------
def band_call(max_grad):
    return "NEW_SYSTEMATIC_CANDIDATE (>3sigma)" if max_grad > 3.0 else "NON_ISSUE_WITHIN_POISSON_BOUND"

result = {
    "node": "rd-timeout-bin-seed3000",
    "source_paths": {
        "logs_dir": str(ARCHIVE),
        "n_log_files": len(log_files),
        "crb_csv": str(crb_csv),
        "n_crb_rows": len(kept),
        "n_timeout_records": len(timeouts),
        "n_snr_stage_timeouts": n_snr_to,
        "n_crb_stage_timeouts": n_crb_to,
        "commit": "a545c0eb (run_metadata_16.json, run_metadata_combine.json)",
    },
    "design_gate": design_gate,
    "snr_stage": {
        "aggregate": {"n_timeout": n_snr_to, "denom": snr_denom, "rate": snr_rate,
                      "garwood95_count_ci": [snr_lo, snr_hi],
                      "garwood95_rate_ci": [snr_lo / snr_denom, snr_hi / snr_denom]},
        "by_axis": {
            axis: {
                "table": axis_tables[axis]["table"].to_dict("records"),
                "max_gradient_sigma": axis_tables[axis]["max_gradient_sigma"],
                "max_gradient_pair_bins": axis_tables[axis]["max_gradient_pair"],
                "band_call": band_call(axis_tables[axis]["max_gradient_sigma"]),
            }
            for axis in ["M", "e0", "p0"]
        },
        "grid_2d_M_p0": {
            "table": cell_df.to_dict("records"),
            "max_gradient_sigma": max_grad_2d,
            "max_gradient_pair_cells": max_grad_2d_pair,
            "band_call": band_call(max_grad_2d),
        },
    },
    "crb_stage_descriptive": {
        "n_timeout": n_crb_to, "denom": crb_denom, "rate": crb_rate,
        "garwood95_count_ci": [crb_lo, crb_hi],
        "garwood95_rate_ci": [crb_lo / crb_denom, crb_hi / crb_denom],
        "events": crb_to_events,
        "note": "n=2, not binnable at 5 bins/axis; reported descriptively only",
    },
    "population_depth_z": z_summary,
    "overall_max_gradient_sigma": overall_max_grad,
    "overall_band_call": band_call(overall_max_grad),
    "gaps": [
        "Raw injected-population files (results/_archive/run_20260707_seed3000/simulations/injections/*.csv) "
        "are UNREACHABLE: dangling symlinks to a cluster-only workspace path "
        "(/pfs/work9/workspace/scratch/st_ac147838-emri/injection_pool_depth15_50k/), not present on this "
        "machine. Bin edges therefore use the union of kept (CRB-CSV) and timeout-log parameters as the "
        "population proxy, not the raw injected draw -- disclosed, not substituted silently.",
        "Denominator for both stages = kept (CRB-CSV rows) + timeout events with known params. Other skip "
        "categories seen in the 33 available 'Skip tally' summary lines (snr:ZeroDivisionError=1151, "
        "snr:Warning, crb:ParameterOutOfBoundsError=32, crb:ZeroDivisionError=12, etc., aggregate "
        "34480 attempts / 1320 successful over the 33 tasks that logged a final tally) have NO per-event "
        "parameters logged (G9 instrumented only the two timeout call sites) and cannot be placed in a bin; "
        "excluded from rate/denominator by construction. This rate answers 'of events that either timed out "
        "or fully succeeded, what fraction timed out', not 'of every attempted draw'.",
        "Only 33 of ~100 simulate array tasks reached the final 'Skip tally' log line (most were still "
        "running / hit the walltime before printing it), so the full per-task attempt/success accounting "
        "is incomplete; this does not affect the timeout-vs-kept binned rate above (which uses raw event "
        "records, not the tally line) but is disclosed as a completeness caveat on the archive.",
        "run_metadata_*.json for simulate task indices 0-40 has been overwritten by a LATER evaluate-stage "
        "array run reusing the same run_metadata_<index>.json filenames (same working directory): only "
        "task indices 41-99 (59 of ~100 files) retain original simulate-stage SLURM metadata. This blocks "
        "a strict per-task-id cross-check of the g-population gate beyond 'same archive, same commit, same "
        "seed, same simulate run' -- disclosed; the global rate computation above does not depend on it.",
        "CRB-stage timeout count (n=2) is far too small to bin into 5x5x5 cells; reported descriptively only.",
    ],
}
(OUTDIR / "READ_RECORD.json").write_text(json.dumps(result, indent=2, default=str))
axis_tables["M"]["table"].to_csv(OUTDIR / "rate_table_M.csv", index=False)
axis_tables["e0"]["table"].to_csv(OUTDIR / "rate_table_e0.csv", index=False)
axis_tables["p0"]["table"].to_csv(OUTDIR / "rate_table_p0.csv", index=False)
cell_df.to_csv(OUTDIR / "rate_table_2d_M_p0.csv", index=False)
print("\nWrote:", OUTDIR / "READ_RECORD.json")
