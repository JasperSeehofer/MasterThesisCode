#!/usr/bin/env python3
"""Diagnostics on S_4D structure + corrected invariance + joint_r1 off-grid response."""
import json
import numpy as np, pandas as pd

RT = "/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729"
DUMP = f"{RT}/tree2_20260830/t2_2b_arm_b_run"
OUT = f"{RT}/graph1_20260901/exec/dv-jr1-transform"
res = {}

pc = pd.read_csv(f"{DUMP}/off/candidate_dump/per_candidate_h_0_73.csv")
pe = pd.read_csv(f"{DUMP}/off/candidate_dump/per_event_h_0_73.csv")

# --- 1. determinism: is s_4d a function of (z, lnM) alone? round keys tightly
d = pc[(pc.s_4d_zg_mg > 0)].copy()
d["lnS"] = np.log(d.s_4d_zg_mg)
d["zk"] = (d.z_g / 5e-4).round().astype(int)
d["mk"] = (np.log(d.M_g) / 5e-3).round().astype(int)
g = d.groupby(["zk", "mk"])["lnS"].agg(["std", "count"])
g = g[g["count"] > 1]
res["determinism"] = dict(n_multi_cells=int(len(g)),
    std_median=float(g["std"].median()), std_p95=float(g["std"].quantile(0.95)),
    std_max=float(g["std"].max()))

# --- 2. gradient scales of lnS in z and lnM (finite diff on tight pairs)
d2 = d.sample(n=min(300000, len(d)), random_state=1)
# local regression via small neighborhoods is heavy; instead estimate via sorted diffs at fixed other-coord bins
def grad(df, key_fix, w_fix, key_var, vname):
    df = df.copy()
    df["fk"] = (df[key_fix] / w_fix).round().astype(int)
    out = []
    for _, sub in df.groupby("fk"):
        if len(sub) < 10: continue
        sub = sub.sort_values(vname)
        dv = np.diff(sub[vname].to_numpy()); dl = np.diff(sub["lnS"].to_numpy())
        m = dv > 1e-6
        if m.sum(): out.append(np.median(np.abs(dl[m] / dv[m])))
    return float(np.median(out))
d2["lnM"] = np.log(d2.M_g)
res["grad_lnS_dlnM_median"] = grad(d2, "z_g", 2e-3, "lnM", "lnM")
res["grad_lnS_dz_median"] = grad(d2, "lnM", 5e-2, "z_g", "z_g")

# --- 3. corrected invariance: per-event on vs off pure columns + L_cat re-booking
peo = pd.read_csv(f"{DUMP}/on/candidate_dump/per_event_h_0_73.csv")
j = pe.merge(peo, on="event_idx", suffixes=("_off", "_on"))
res["pure_invariance"] = dict(
    max_rel_B=float(np.nanmax(np.abs(j.B_num_on / j.B_num_off - 1))),
    max_rel_D=float(np.nanmax(np.abs(j.D_tilde_phi_on / j.D_tilde_phi_off - 1))))
# per-event L_cat identity: L_on = sum w N_on / Sigma_4D ; L_off = sum w N_off / Sigma_phi
pco = pd.read_csv(f"{DUMP}/on/candidate_dump/per_candidate_h_0_73.csv",
                  usecols=["event_idx","catalog_index","w_g","N_g_used"])
num_on = pco.assign(x=pco.w_g*pco.N_g_used).groupby("event_idx")["x"].sum()
num_off = pc.assign(x=pc.w_g*pc.N_g_used).groupby("event_idx")["x"].sum()
LON = j.set_index("event_idx")["L_cat_no_bh_on"]; LOFF = j.set_index("event_idx")["L_cat_no_bh_off"]
ci = pd.DataFrame({"non": num_on, "noff": num_off, "lon": LON, "loff": LOFF}).dropna()
ci = ci[(ci.lon>0)&(ci.loff>0)&(ci.non>0)&(ci.noff>0)]
sig_phi_impl = (ci.noff/ci.loff)
sig_4d_impl = (ci.non/ci.lon)
res["rebooking"] = dict(n=int(len(ci)),
    sigma_phi_implied_rel_spread=float(sig_phi_impl.std()/sig_phi_impl.mean()),
    sigma_4d_implied_rel_spread=float(sig_4d_impl.std()/sig_4d_impl.mean()),
    sigma_phi_implied=float(sig_phi_impl.mean()), sigma_4d_implied=float(sig_4d_impl.mean()),
    ratio_alpha=float(sig_4d_impl.mean()/sig_phi_impl.mean()))

# --- 4. joint_r1 off grid: full 41-node posterior + per-event secant scores
el = pd.read_csv(f"{RT}/headreadout_20260827/joint_r1/event_likelihoods.csv")
hs = np.sort(el.h.unique())
res["jr1_grid"] = dict(n_nodes=int(len(hs)), h_min=float(hs[0]), h_max=float(hs[-1]))
piv = el.pivot(index="event_idx", columns="h", values="combined_no_bh")
lnL = np.log(piv).sum(axis=0).reindex(hs)
# trapezoid weights, flat prior
w = np.zeros(len(hs)); w[1:-1] = (hs[2:] - hs[:-2]) / 2; w[0] = (hs[1]-hs[0])/2; w[-1] = (hs[-1]-hs[-2])/2
lw = lnL.to_numpy() - lnL.max()
post = np.exp(lw) * w; post /= post.sum()
res["jr1_off_posterior"] = dict(MAP=float(hs[np.argmax(lnL.to_numpy())]),
    mean=float(np.sum(post*hs)), floor_mass=float(post[0]),
    lnL_073_secant=float((lnL.loc[0.735]-lnL.loc[0.725])/0.01))
# per-event scores at 0.725/0.735 secant
pf = el.pivot(index="event_idx", columns="h", values="combined_no_bh")
pB = el.pivot(index="event_idx", columns="h", values="B_num")
pD = el.pivot(index="event_idx", columns="h", values="D_tilde_phi")
s_full = (np.log(pf[0.735]) - np.log(pf[0.725]))/0.01
s_pure = (np.log(pB[0.735]/pD[0.735]) - np.log(pB[0.725]/pD[0.725]))/0.01
s_imp = s_full - s_pure
host = pe.set_index("event_idx")["host_galaxy_index"].reindex(s_full.index)
dark = host < 0
res["jr1_scores"] = dict(
    n=int(len(s_full)), n_dark=int(dark.sum()), n_incat=int((~dark).sum()),
    sum_pure=float(s_pure.sum()),
    dark_simp_sum=float(s_imp[dark].sum()), dark_simp_mean=float(s_imp[dark].mean()),
    incat_simp_sum=float(s_imp[~dark].sum()), incat_simp_mean=float(s_imp[~dark].mean()),
    pooled_simp_mean=float(s_imp.mean()))
# same for iiib T2.2b off (consistency vs BAND_REDERIVATION)
elо = None
json.dump(res, open(f"{OUT}/jr1_diag_response.json","w"), indent=1)
print(json.dumps(res, indent=1))
