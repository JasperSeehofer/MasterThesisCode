import pandas as pd
import numpy as np
import json

ROOT = "/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729"

SEEDS = [900101, 900102, 900103, 900104]

def load(seed):
    zwin = pd.read_csv(f"{ROOT}/tree2_20260830/hier_s0_zwin_run/s0a_seed{seed}/node_truth_sites2.2_nosmear_divisor_zwin_zk4/simulations/diagnostics/event_likelihoods.csv")
    t12  = pd.read_csv(f"{ROOT}/tree2_20260830/hier_s0_recert_run/s0a_seed{seed}/node_truth_sites2.2_nosmear_divisor/simulations/diagnostics/event_likelihoods.csv")
    bc   = pd.read_csv(f"{ROOT}/p3_b0_work/bc_{seed}_work/seed{seed}/simulations/diagnostics/event_likelihoods.csv")
    bc73 = bc[np.isclose(bc["h"], 0.73, rtol=1e-9, atol=1e-12)].drop_duplicates(subset="event_idx", keep="last")
    return zwin, t12, bc73

def ln_l(df):
    out = df[["event_idx"]].copy()
    for col, name in (("combined_no_bh","ln_L_no_bh"),("combined_with_bh","ln_L_with_bh")):
        v = df[col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = np.where(v>0.0, np.log(v), np.nan)
    return out

AUX_COLS = ["w_G","w_G_legacy","w_tilde_G","alpha_G_phi","r_Malm","D_tilde_phi","g_frac","L_comp","B_num","B_num_wbh","den_log_term"]

results = {}
ALL_NO_ADD_ABSDIFFS_NOBH = []
ALL_NO_ADD_ABSDIFFS_WBH = []
PRE_WINDOW_FLOOR_NOBH = {}  # per-seed E19 floor, recomputed vs bc

for seed in SEEDS:
    zwin, t12, bc73 = load(seed)

    # merge zwin vs t12 on event_idx (inner)
    m = zwin.merge(t12, on="event_idx", suffixes=("_zwin","_t12"), how="outer", indicator=True)
    only_zwin = (m["_merge"]=="left_only").sum()
    only_t12 = (m["_merge"]=="right_only").sum()
    both = (m["_merge"]=="both").sum()

    m = zwin.merge(t12, on="event_idx", suffixes=("_zwin","_t12"))

    lnL_zwin = ln_l(zwin).set_index("event_idx")
    lnL_t12 = ln_l(t12).set_index("event_idx")
    common = lnL_zwin.index.intersection(lnL_t12.index)
    dln_nobh = lnL_zwin.loc[common,"ln_L_no_bh"] - lnL_t12.loc[common,"ln_L_no_bh"]
    dln_wbh  = lnL_zwin.loc[common,"ln_L_with_bh"] - lnL_t12.loc[common,"ln_L_with_bh"]

    # candidate-added classification via L_cat_no_bh / L_cat_with_bh exact identity
    Lcat_zwin = zwin.set_index("event_idx")[["L_cat_no_bh","L_cat_with_bh"]]
    Lcat_t12 = t12.set_index("event_idx")[["L_cat_no_bh","L_cat_with_bh"]]
    Lcat_diff_nobh = (Lcat_zwin.loc[common,"L_cat_no_bh"] - Lcat_t12.loc[common,"L_cat_no_bh"])
    Lcat_diff_wbh = (Lcat_zwin.loc[common,"L_cat_with_bh"] - Lcat_t12.loc[common,"L_cat_with_bh"])

    added_mask = (Lcat_diff_nobh.abs() > 0.0) | (Lcat_diff_wbh.abs() > 0.0)
    n_added = int(added_mask.sum())
    n_no_added = int((~added_mask).sum())

    # direction check: L_cat should only INCREASE when candidates are added (window widens)
    n_decreased_nobh = int((Lcat_diff_nobh[added_mask] < 0).sum())
    n_decreased_wbh = int((Lcat_diff_wbh[added_mask] < 0).sum())

    # aux-column identity check (are non-candidate terms bit-identical?)
    aux_z = zwin.set_index("event_idx")[AUX_COLS].loc[common]
    aux_t = t12.set_index("event_idx")[AUX_COLS].loc[common]
    aux_diff = (aux_z - aux_t).abs()
    aux_max_abs_per_col = aux_diff.max().to_dict()
    aux_all_identical = bool((aux_diff.to_numpy() == 0.0).all())
    n_aux_nonzero_events = int((aux_diff.to_numpy(dtype=float).sum(axis=1) > 0).sum())

    # E19 floor: recompute zwin vs bc and t12 vs bc gate parity (same methodology as driver)
    lnL_bc = ln_l(bc73).set_index("event_idx")
    common_bc_t12 = lnL_t12.index.intersection(lnL_bc.index)
    a = lnL_t12.loc[common_bc_t12,"ln_L_no_bh"].to_numpy(float)
    b = lnL_bc.loc[common_bc_t12,"ln_L_no_bh"].to_numpy(float)
    fin = np.isfinite(a) & np.isfinite(b)
    abs_diff = np.abs(a[fin]-b[fin])
    rel_diff = abs_diff/np.maximum(np.abs(b[fin]),1e-300)
    t12_vs_bc_max_rel_nobh = float(np.max(rel_diff)) if fin.sum()>0 else float('nan')
    t12_vs_bc_max_abs_nobh = float(np.max(abs_diff)) if fin.sum()>0 else float('nan')

    common_bc_zwin = lnL_zwin.index.intersection(lnL_bc.index)
    a2 = lnL_zwin.loc[common_bc_zwin,"ln_L_no_bh"].to_numpy(float)
    b2 = lnL_bc.loc[common_bc_zwin,"ln_L_no_bh"].to_numpy(float)
    fin2 = np.isfinite(a2) & np.isfinite(b2)
    abs_diff2 = np.abs(a2[fin2]-b2[fin2])
    rel_diff2 = abs_diff2/np.maximum(np.abs(b2[fin2]),1e-300)
    zwin_vs_bc_max_rel_nobh = float(np.max(rel_diff2)) if fin2.sum()>0 else float('nan')
    zwin_vs_bc_max_abs_nobh = float(np.max(abs_diff2)) if fin2.sum()>0 else float('nan')

    a3 = lnL_t12.loc[common_bc_t12,"ln_L_with_bh"].to_numpy(float)
    b3 = lnL_bc.loc[common_bc_t12,"ln_L_with_bh"].to_numpy(float)
    fin3 = np.isfinite(a3) & np.isfinite(b3)
    abs_diff3 = np.abs(a3[fin3]-b3[fin3])
    rel_diff3 = abs_diff3/np.maximum(np.abs(b3[fin3]),1e-300)
    t12_vs_bc_max_rel_wbh = float(np.max(rel_diff3)) if fin3.sum()>0 else float('nan')

    a4 = lnL_zwin.loc[common_bc_zwin,"ln_L_with_bh"].to_numpy(float)
    b4 = lnL_bc.loc[common_bc_zwin,"ln_L_with_bh"].to_numpy(float)
    fin4 = np.isfinite(a4) & np.isfinite(b4)
    abs_diff4 = np.abs(a4[fin4]-b4[fin4])
    rel_diff4 = abs_diff4/np.maximum(np.abs(b4[fin4]),1e-300)
    zwin_vs_bc_max_rel_wbh = float(np.max(rel_diff4)) if fin4.sum()>0 else float('nan')

    results[seed] = {
        "n_zwin_only": int(only_zwin), "n_t12_only": int(only_t12), "n_common": int(both),
        "n_events_no_added": n_no_added,
        "n_events_added": n_added,
        "n_decreased_nobh_among_added": n_decreased_nobh,
        "n_decreased_wbh_among_added": n_decreased_wbh,
        "dln_nobh_no_added_max_abs": float(dln_nobh[~added_mask].abs().max()) if n_no_added>0 else None,
        "dln_wbh_no_added_max_abs": float(dln_wbh[~added_mask].abs().max()) if n_no_added>0 else None,
        "dln_nobh_added_min": float(dln_nobh[added_mask].min()) if n_added>0 else None,
        "dln_nobh_added_max": float(dln_nobh[added_mask].max()) if n_added>0 else None,
        "dln_wbh_added_min": float(dln_wbh[added_mask].min()) if n_added>0 else None,
        "dln_wbh_added_max": float(dln_wbh[added_mask].max()) if n_added>0 else None,
        "aux_columns_all_identical_zwin_vs_t12": aux_all_identical,
        "aux_max_abs_per_col": {k: float(v) for k,v in aux_max_abs_per_col.items()},
        "n_events_with_aux_col_change": n_aux_nonzero_events,
        "recomputed_t12_vs_bc_max_rel_nobh": t12_vs_bc_max_rel_nobh,
        "recomputed_t12_vs_bc_max_abs_nobh": t12_vs_bc_max_abs_nobh,
        "recomputed_zwin_vs_bc_max_rel_nobh": zwin_vs_bc_max_rel_nobh,
        "recomputed_zwin_vs_bc_max_abs_nobh": zwin_vs_bc_max_abs_nobh,
        "recomputed_t12_vs_bc_max_rel_wbh": t12_vs_bc_max_rel_wbh,
        "recomputed_zwin_vs_bc_max_rel_wbh": zwin_vs_bc_max_rel_wbh,
    }
    PRE_WINDOW_FLOOR_NOBH[seed] = t12_vs_bc_max_rel_nobh

print(json.dumps(results, indent=2))
print("---FLOOR---")
print(json.dumps(PRE_WINDOW_FLOOR_NOBH, indent=2))

print("=== SPOT CHECK: no-added-candidate events, raw values ===")
for seed in SEEDS:
    zwin, t12, bc73 = load(seed)
    Lcat_zwin = zwin.set_index("event_idx")[["L_cat_no_bh","L_cat_with_bh","combined_no_bh","combined_with_bh"]]
    Lcat_t12 = t12.set_index("event_idx")[["L_cat_no_bh","L_cat_with_bh","combined_no_bh","combined_with_bh"]]
    common = Lcat_zwin.index.intersection(Lcat_t12.index)
    diff_nobh = (Lcat_zwin.loc[common,"L_cat_no_bh"] - Lcat_t12.loc[common,"L_cat_no_bh"])
    diff_wbh = (Lcat_zwin.loc[common,"L_cat_with_bh"] - Lcat_t12.loc[common,"L_cat_with_bh"])
    no_added = common[(diff_nobh.abs()==0.0) & (diff_wbh.abs()==0.0)]
    print(f"seed {seed}: no_added event_idx = {list(no_added)}")
    for ei in list(no_added)[:2]:
        print(f"   event {ei}: zwin combined_no_bh={Lcat_zwin.loc[ei,'combined_no_bh']!r} t12={Lcat_t12.loc[ei,'combined_no_bh']!r} equal={Lcat_zwin.loc[ei,'combined_no_bh']==Lcat_t12.loc[ei,'combined_no_bh']}")
