#!/usr/bin/env python3
"""Item 6 (B4.2 KW-Q1) independent verifier re-derivation.

Re-derives S(s), R, GATE I, GATE ENG, and the A14 falsifier share DIRECTLY
from the raw per-node event_likelihoods.csv files and the frozen q1
membership artifacts -- WITHOUT importing kwq1_score.py, hier_s0_driver.py,
or the record's own re-derivation script. This is a from-scratch third
implementation, deliberately not sharing code with either the scorer or the
readout record's re-derivation.

Sources (opened directly, not restated from any record):
  - kwq1_registered_run/s0a_seed<seed>/node_<node>_ft_sites2.2_nosmear/
      simulations/diagnostics/event_likelihoods.csv  (4 seeds x 3 nodes)
  - b4_imp_stage1_events.csv  (arm == "ft" rows; z_true membership)
  - b4_imp_stage1_forecast.json  (covariates.ft.z_true.edges)
  - kwq1_registered_run/kwq1_score_output.json  (comparison target only,
    not used as an input to any computed number)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/fanout1_20260829")
RUN = ROOT / "kwq1_registered_run"
SEEDS = [900101, 900102, 900103, 900104]
NODES = ["s_minus", "truth", "s_plus"]  # s = 1/sqrt2, 1, sqrt2
SUFFIX = "_ft_sites2.2_nosmear"
H_LO, H_HI = 0.725, 0.735
GATE_I_TOL = 2.0e-6
GATE_ENG_MIN = 0.99
FALSIFIER_MIN_Q1 = 0.50
BAND_OWNS, BAND_INERT = 0.5, 0.2


def node_csv(seed: int, node: str) -> Path:
    return RUN / f"s0a_seed{seed}" / f"node_{node}{SUFFIX}" / "simulations" / "diagnostics" / "event_likelihoods.csv"


def per_event_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    lo = df[np.isclose(df["h"], H_LO)].sort_values("event_idx").reset_index(drop=True)
    hi = df[np.isclose(df["h"], H_HI)].sort_values("event_idx").reset_index(drop=True)
    m = lo.merge(hi, on="event_idx", suffixes=("_lo", "_hi"))

    cat_lo = (m["alpha_G_phi_lo"] / m["r_Malm_lo"]) * m["L_cat_no_bh_lo"] / m["D_tilde_phi_lo"]
    cat_hi = (m["alpha_G_phi_hi"] / m["r_Malm_hi"]) * m["L_cat_no_bh_hi"] / m["D_tilde_phi_hi"]
    comp_lo = m["B_num_lo"] / m["D_tilde_phi_lo"]
    comp_hi = m["B_num_hi"] / m["D_tilde_phi_hi"]
    full_lo = m["combined_no_bh_lo"].to_numpy(float)
    full_hi = m["combined_no_bh_hi"].to_numpy(float)

    # GATE I: assembly identity cat_term + comp_term == full, both h.
    scale_lo = np.maximum(np.abs(full_lo), np.finfo(float).tiny)
    scale_hi = np.maximum(np.abs(full_hi), np.finfo(float).tiny)
    rel_lo = np.abs(cat_lo.to_numpy(float) + comp_lo.to_numpy(float) - full_lo) / scale_lo
    rel_hi = np.abs(cat_hi.to_numpy(float) + comp_hi.to_numpy(float) - full_hi) / scale_hi
    gate_i_max_rel = float(np.nanmax(np.concatenate([rel_lo, rel_hi])))

    pure_lo = np.clip(full_lo - cat_lo.to_numpy(float), 0.0, None)
    pure_hi = np.clip(full_hi - cat_hi.to_numpy(float), 0.0, None)

    def secant(a, b):
        out = np.full(a.shape[0], np.nan)
        ok = (a > 0.0) & (b > 0.0)
        out[ok] = (np.log(b[ok]) - np.log(a[ok])) / (H_HI - H_LO)
        return out

    s_full = secant(full_lo, full_hi)
    s_pure = secant(pure_lo, pure_hi)
    s_imp = s_full - s_pure

    out = pd.DataFrame({
        "event_idx": m["event_idx"],
        "s_imp": s_imp,
        "L_cat_no_bh_lo": m["L_cat_no_bh_lo"].to_numpy(float),
    })
    out.attrs["gate_i_max_rel"] = gate_i_max_rel
    return out


def load_quartiles():
    forecast = json.loads((ROOT / "b4_imp_stage1_forecast.json").read_text())
    e0, e1, e2 = forecast["covariates"]["ft"]["z_true"]["edges"]
    ev = pd.read_csv(ROOT / "b4_imp_stage1_events.csv")
    ev = ev[(ev["arm"] == "ft") & (ev["seed"].isin(SEEDS))]
    z = ev["z_true"].to_numpy(float)
    q = np.where(z < e0, 1, np.where(z < e1, 2, np.where(z < e2, 3, 4)))
    pairs_by_q = {k: set() for k in (1, 2, 3, 4)}
    for seed, ei, qk in zip(ev["seed"].to_numpy(int), ev["event_idx"].to_numpy(int), q):
        pairs_by_q[int(qk)].add((int(seed), int(ei)))
    return pairs_by_q, float(e0)


def pooled_mean(frames_by_seed, pairs):
    vals = []
    for seed, frame in frames_by_seed.items():
        keep = [(seed, ei) in pairs for ei in frame["event_idx"]]
        sub = frame.loc[keep, "s_imp"].to_numpy(float)
        vals.extend(sub[np.isfinite(sub)].tolist())
    arr = np.array(vals, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return mean, sem, int(arr.size)


def main():
    frames = {node: {} for node in NODES}
    gate_i_max_rel_all = 0.0
    for node in NODES:
        for seed in SEEDS:
            p = node_csv(seed, node)
            assert p.is_file(), f"missing {p}"
            f = per_event_frame(p)
            gate_i_max_rel_all = max(gate_i_max_rel_all, f.attrs["gate_i_max_rel"])
            frames[node][seed] = f

    pairs_by_q, edge_q1 = load_quartiles()
    q1_pairs = pairs_by_q[1]

    S = {}
    per_seed_S = {node: {} for node in NODES}
    for node in NODES:
        mean, sem, n = pooled_mean(frames[node], q1_pairs)
        S[node] = (mean, sem, n)
        for seed in SEEDS:
            m2, _, _ = pooled_mean({seed: frames[node][seed]}, q1_pairs)
            per_seed_S[node][seed] = m2

    R_pooled = (S["s_plus"][0] - S["s_minus"][0]) / abs(S["truth"][0])

    per_seed_R = {}
    for seed in SEEDS:
        r = (per_seed_S["s_plus"][seed] - per_seed_S["s_minus"][seed]) / abs(per_seed_S["truth"][seed])
        per_seed_R[seed] = r

    # GATE ENG: fraction of active rows (L_cat_no_bh_lo > 0 on truth) where
    # L_cat_no_bh_lo differs between s_minus and s_plus.
    n_active = 0
    n_differ = 0
    for seed in SEEDS:
        t = frames["truth"][seed][["event_idx", "L_cat_no_bh_lo"]]
        sm = frames["s_minus"][seed][["event_idx", "L_cat_no_bh_lo"]].rename(columns={"L_cat_no_bh_lo": "sm"})
        sp = frames["s_plus"][seed][["event_idx", "L_cat_no_bh_lo"]].rename(columns={"L_cat_no_bh_lo": "sp"})
        m = t.merge(sm, on="event_idx").merge(sp, on="event_idx")
        active = m["L_cat_no_bh_lo"].to_numpy(float) > 0.0
        differ = m["sm"].to_numpy(float) != m["sp"].to_numpy(float)
        n_active += int(active.sum())
        n_differ += int((differ & active).sum())
    gate_eng_frac = n_differ / n_active if n_active else float("nan")

    # Falsifier: q1 share of total s_imp (all quartiles) at truth (s=1).
    quartile_sums = {}
    for qk, pairs in pairs_by_q.items():
        mean_q, _, n_q = pooled_mean(frames["truth"], pairs)
        quartile_sums[qk] = mean_q * n_q if np.isfinite(mean_q) else 0.0
    total = sum(quartile_sums.values())
    q1_share = quartile_sums[1] / total

    if abs(R_pooled) >= BAND_OWNS:
        band = "KERNEL-WIDTH-OWNS"
    elif abs(R_pooled) <= BAND_INERT:
        band = "KERNEL-WIDTH-INERT"
    else:
        band = "MIXED"

    print("=== Item 6 (B4.2 KW-Q1) independent re-derivation ===")
    print(f"q1 edge (z_true <): {edge_q1!r}")
    print(f"q1 n pooled pairs: {S['truth'][2]}")
    for node in NODES:
        print(f"S({node}) = {S[node][0]!r}  sem={S[node][1]!r}  n={S[node][2]}")
    print(f"R_pooled = {R_pooled!r}")
    print(f"per-seed R: {per_seed_R}")
    print(f"GATE I max_rel = {gate_i_max_rel_all!r} (tol {GATE_I_TOL}) -> {'PASS' if gate_i_max_rel_all <= GATE_I_TOL else 'FAIL'}")
    print(f"GATE ENG: n_active={n_active} n_differ={n_differ} frac={gate_eng_frac!r} (thr {GATE_ENG_MIN}) -> {'PASS' if gate_eng_frac >= GATE_ENG_MIN else 'FAIL'}")
    print(f"Falsifier q1 share of total s_imp at truth = {q1_share!r} (floor {FALSIFIER_MIN_Q1}) -> "
          f"{'NOT withdrawn' if q1_share >= FALSIFIER_MIN_Q1 else 'WITHDRAWN'}")
    print(f"BAND: |R|={abs(R_pooled):.6f} -> {band}")

    # Comparison against the scorer's own JSON (read for comparison only, not
    # used as an input to any of the numbers above).
    ref = json.loads((RUN / "kwq1_score_output.json").read_text())
    print()
    print("=== Comparison to kwq1_score_output.json (reference, not an input) ===")
    print(f"R: mine={R_pooled!r}  scorer={ref['R']!r}  match={np.isclose(R_pooled, ref['R'], rtol=0, atol=1e-9)}")
    for node in NODES:
        print(f"S({node}): mine={S[node][0]!r} scorer={ref['S_by_node'][node]['S']!r} "
              f"match={np.isclose(S[node][0], ref['S_by_node'][node]['S'], atol=1e-9)}")
    print(f"gate_i_max_rel: mine={gate_i_max_rel_all!r} scorer={ref['gate_i_max_rel']!r}")
    print(f"gate_eng frac: mine={gate_eng_frac!r} scorer={ref['gate_eng']['fraction_L_cat_differs_across_s']!r}")
    print(f"falsifier q1_share: mine={q1_share!r} scorer={ref['falsifier_q1_share_of_total_s_imp_at_truth']!r}")


if __name__ == "__main__":
    main()
