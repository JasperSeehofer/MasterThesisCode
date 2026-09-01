#!/usr/bin/env python3
"""Final stage: slab-interp smearing -> jr1 transform + R_K + h-stability + posterior band scan."""
import json
import numpy as np, pandas as pd

RT = "/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729"
DUMP = f"{RT}/tree2_20260830/t2_2b_arm_b_run"
OUT = f"{RT}/graph1_20260901/exec/dv-jr1-transform"
H_NODES = [0.725, 0.73, 0.735]
ZSLAB = 0.002
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(41)
GH_W = GH_W / np.sqrt(2*np.pi)
RNG = np.random.default_rng(20260902)
res = {"convention": "slab-interp: z-slabs of 0.002, raw-point lnS-vs-lnM interp (dup-averaged), flat ends"}

def build_slabs(pc):
    d = pc[pc.s_4d_zg_mg > 0]
    sl = {}
    zi = np.floor(d.z_g.to_numpy()/ZSLAB).astype(int)
    df = pd.DataFrame({"zi": zi, "lnM": np.log(d.M_g.to_numpy()), "lnS": np.log(d.s_4d_zg_mg.to_numpy())})
    for u, sub in df.groupby("zi"):
        g = sub.groupby(sub.lnM.round(4))["lnS"].mean()
        sl[u] = (g.index.to_numpy(), g.to_numpy())
    return sl

def slab_eval(sl, keys_sorted, z, lnM):
    zi = np.floor(np.asarray(z)/ZSLAB).astype(int)
    out = np.empty(np.shape(lnM), float); clamped = np.zeros(np.shape(lnM), bool)
    for u in np.unique(zi):
        m = zi == u
        k = u if u in sl else keys_sorted[np.argmin(np.abs(keys_sorted - u))]
        xs, ys = sl[k]
        q = np.asarray(lnM)[m]
        clamped[m] = (q < xs[0]) | (q > xs[-1])
        out[m] = np.interp(q, xs, ys)
    return out, clamped

def smear_rows(sl, ks, z, lnM, sig):
    acc = np.zeros(len(lnM)); clw = np.zeros(len(lnM))
    for x, w in zip(GH_X, GH_W):
        ls, cl = slab_eval(sl, ks, z, lnM + sig*x)
        acc += w*np.exp(ls); clw += w*cl
    return acc, clw

pe = pd.read_csv(f"{DUMP}/off/candidate_dump/per_event_h_0_73.csv")
dark_ev = set(pe[pe.host_galaxy_index < 0].event_idx)

for h in H_NODES:
    tag = str(h).replace(".", "_")
    pc = pd.read_csv(f"{DUMP}/off/candidate_dump/per_candidate_h_{tag}.csv")
    sl = build_slabs(pc); ks = np.array(sorted(sl.keys()))
    node = {}
    # fidelity on ALL rows (weighted by w*N) and on dark rows
    zc = pc.z_g.to_numpy(); lnMc = np.log(pc.M_g.to_numpy())
    s4 = pc.s_4d_zg_mg.to_numpy(); sb = pc.s_bar_phi_zg.to_numpy()
    wN = (pc.w_g*pc.N_g_used).to_numpy()
    ok = (s4>0)&(sb>0)&np.isfinite(wN)&(wN>=0)
    p0, _ = slab_eval(sl, ks, zc[ok], lnMc[ok])
    p0 = np.exp(p0)
    node["fidelity_wN_aggregate"] = float(np.sum(wN[ok]*p0)/np.sum(wN[ok]*s4[ok]))
    node["fidelity_lnresid_p95"] = float(np.quantile(np.abs(np.log(p0/s4[ok])), 0.95))
    # true hosts
    th = pc[pc.is_true_host == True]
    z_t = th.z_g.to_numpy(); lnM_t = np.log(th.M_g.to_numpy())
    sig_t = np.clip((th.M_err_g/th.M_g).to_numpy(), 1e-3, 3.0)
    sb_t = th.s_bar_phi_zg.to_numpy(); s4_t = th.s_4d_zg_mg.to_numpy()
    node["delta_law_median"] = float(np.median(s4_t/sb_t))
    pt0, _ = slab_eval(sl, ks, z_t, lnM_t)
    S0 = np.exp(pt0)
    node["truehost_surface_fid_median"] = float(np.median(S0/s4_t))
    Es, clw = smear_rows(sl, ks, z_t, lnM_t, sig_t)
    K = Es/S0  # same-surface ratio: profile bias cancels
    node["K_median"] = float(np.median(K)); node["K_mean"] = float(np.mean(K))
    node["K_p10"] = float(np.quantile(K, 0.10)); node["K_p90"] = float(np.quantile(K, 0.90))
    node["truehost_clamp_w_mean"] = float(np.mean(clw))
    node["T_expectation_median"] = float(np.median(s4_t/sb_t * K))
    # realized-median MC (transform as it will be measured on one realization)
    meds = np.empty(4000)
    for i in range(4000):
        eps = RNG.standard_normal(len(z_t))
        ls, _ = slab_eval(sl, ks, z_t, lnM_t + sig_t*eps)
        meds[i] = np.median(np.exp(ls)/S0 * (s4_t/sb_t))
    node["T_realized_median_q"] = [float(np.quantile(meds, q)) for q in (0.025, 0.05, 0.5, 0.95, 0.975)]
    # per-host realized-ratio spread (one-draw distribution, pooled)
    eps = RNG.standard_normal((200, len(z_t)))
    allr = []
    for e in eps:
        ls, _ = slab_eval(sl, ks, z_t, lnM_t + sig_t*e)
        allr.append(np.exp(ls)/S0 * (s4_t/sb_t))
    allr = np.concatenate(allr)
    node["per_host_realized_ratio_q"] = [float(np.quantile(allr, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    # dark-class R_K
    dm = pc.event_idx.isin(dark_ev).to_numpy() & ok
    zd = zc[dm]; lMd = lnMc[dm]
    sgd = np.clip((pc.M_err_g/pc.M_g).to_numpy()[dm], 1e-3, 3.0)
    wNd = wN[dm]; sbd = sb[dm]
    pd0, _ = slab_eval(sl, ks, zd, lMd); pd0 = np.exp(pd0)
    Ed, clwd = smear_rows(sl, ks, zd, lMd, sgd)
    node["RK_dark"] = float(np.sum(wNd*Ed)/np.sum(wNd*pd0))
    node["rho_surv_point"] = float(np.sum(wNd*pd0)/np.sum(wNd*sbd))
    node["rho_surv_smeared"] = float(np.sum(wNd*Ed)/np.sum(wNd*sbd))
    node["dark_clamp_w_mean"] = float(np.mean(clwd))
    res[str(h)] = node
    print(h, json.dumps(node, indent=1), flush=True)

# ---------- posterior band scan on the joint_r1 banked off grid ----------
el = pd.read_csv(f"{RT}/headreadout_20260827/joint_r1/event_likelihoods.csv")
hs = np.sort(el.h.unique())
piv = el.pivot(index="event_idx", columns="h", values="combined_no_bh")
lnL = np.log(piv).sum(axis=0).reindex(hs).to_numpy()
pB = el.pivot(index="event_idx", columns="h", values="B_num")
pD = el.pivot(index="event_idx", columns="h", values="D_tilde_phi")
s_full = (np.log(piv[0.735]) - np.log(piv[0.725]))/0.01
s_pure = (np.log(pB[0.735]/pD[0.735]) - np.log(pB[0.725]/pD[0.725]))/0.01
s_imp = s_full - s_pure
host = pe.set_index("event_idx")["host_galaxy_index"].reindex(s_imp.index)
dark = (host < 0).to_numpy()
dark_sum = float(s_imp[dark].sum()); incat_sum = float(s_imp[~dark].sum())
pure_dark_sum = float(s_pure[dark].sum())
w = np.zeros(len(hs)); w[1:-1] = (hs[2:]-hs[:-2])/2; w[0]=(hs[1]-hs[0])/2; w[-1]=(hs[-1]-hs[-2])/2
def moments(l):
    lw = l - l.max(); p = np.exp(lw)*w; p /= p.sum()
    return float(hs[np.argmax(l)]), float(np.sum(p*hs)), float(p[0])
scan = []
IIIB = dict(rho=0.2604, dpr=216.903, ddp=-905.03, dark_off=-291.16)
for rho in (0.20, 0.23, 0.2604, 0.30, 0.34):
    d1 = (rho-1.0)*dark_sum + 1.7  # in-cat term scaled from iiib +1.55 x (179.5/164.5)
    for mode in ("lin", "quad"):
        d2 = IIIB["ddp"]*(d1/IIIB["dpr"]) if mode == "quad" else 0.0
        lon = lnL + d1*(hs-0.73) + 0.5*d2*(hs-0.73)**2
        m, mn, fl = moments(lon)
        scan.append(dict(rho=rho, mode=mode, MAP=m, mean=mn, floor=fl, dpr=d1))
res["jr1_band_scan"] = dict(dark_simp_off_sum=dark_sum, incat_simp_off_sum=incat_sum,
    dark_pure_off_sum=pure_dark_sum, off=dict(zip(("MAP","mean","floor"), moments(lnL))), scan=scan)
json.dump(res, open(f"{OUT}/jr1_transform_final.json","w"), indent=1)
print(json.dumps(res["jr1_band_scan"], indent=1))
