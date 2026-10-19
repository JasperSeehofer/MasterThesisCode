#!/usr/bin/env python3
"""dv-jr1-transform numerical engine (graph-1 wave-1; analysis-only, CPU, no repo code edits).

Derives the joint_r1 T2.2b-equivalent true-host transform under the venue's log-normal
realized-forward mass law, from the T2.2b iiib candidate dumps (S_4D is venue-independent:
same injection pool / detection model; only the query masses differ by venue).
"""
import json, sys
import numpy as np
import pandas as pd

RT = "/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729"
DUMP = f"{RT}/tree2_20260830/t2_2b_arm_b_run"
OUT = f"{RT}/graph1_20260901/exec/dv-jr1-transform"
H_NODES = [0.725, 0.73, 0.735]
RNG = np.random.default_rng(20260902)

def load_pc(arm, h):
    tag = str(h).replace(".", "_")
    return pd.read_csv(f"{DUMP}/{arm}/candidate_dump/per_candidate_h_{tag}.csv")

def load_pe(arm, h):
    tag = str(h).replace(".", "_")
    return pd.read_csv(f"{DUMP}/{arm}/candidate_dump/per_event_h_{tag}.csv")

# ---------- surface: ln S_4D(z, lnM), isotone in lnM per z-bin ----------
def build_surface(pc, z_bin, m_bin):
    z = pc["z_g"].to_numpy(); lnM = np.log(pc["M_g"].to_numpy())
    s = pc["s_4d_zg_mg"].to_numpy()
    ok = np.isfinite(z) & np.isfinite(lnM) & np.isfinite(s) & (s > 0)
    z, lnM, s = z[ok], lnM[ok], np.log(s[ok])
    zi = np.floor(z / z_bin).astype(int); mi = np.floor(lnM / m_bin).astype(int)
    df = pd.DataFrame({"zi": zi, "mi": mi, "ls": s})
    g = df.groupby(["zi", "mi"])["ls"].median().reset_index()
    surf = {}
    for zid, sub in g.groupby("zi"):
        sub = sub.sort_values("mi")
        mids = sub["mi"].to_numpy() * m_bin + m_bin / 2
        vals = np.maximum.accumulate(sub["ls"].to_numpy())  # isotone in lnM
        surf[zid] = (mids, vals)
    return surf

def surf_eval(surf, z_bin, z, lnM):
    """Evaluate ln S_4D at (z, lnM) arrays; nearest z-bin fallback; flat lnM extension."""
    zi = np.floor(np.asarray(z) / z_bin).astype(int)
    lnM = np.asarray(lnM, dtype=float)
    out = np.empty(lnM.shape); clamped = np.zeros(lnM.shape, bool)
    keys = np.array(sorted(surf.keys()))
    for u in np.unique(zi):
        m = zi == u
        k = u if u in surf else keys[np.argmin(np.abs(keys - u))]
        mids, vals = surf[k]
        x = lnM[m]
        clamped[m] = (x < mids[0]) | (x > mids[-1])
        out[m] = np.interp(x, mids, vals)
    return out, clamped

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(41)  # probabilists': int f(x) N(x)dx = sum w/sqrt(2pi) f(x)
GH_W = GH_W / np.sqrt(2 * np.pi)

def smear(surf, z_bin, z, lnM, sig):
    """E_eps[S_4D(z, lnM + sig*eps)] via 41-node Gauss-Hermite on the surface."""
    z = np.asarray(z); lnM = np.asarray(lnM); sig = np.asarray(sig)
    n = len(lnM); acc = np.zeros(n); clfrac = np.zeros(n)
    for x, w in zip(GH_X, GH_W):
        ls, cl = surf_eval(surf, z_bin, z, lnM + sig * x)
        acc += w * np.exp(ls); clfrac += w * cl
    return acc, clfrac

res = {"h_nodes": H_NODES}

# ---------- A. baseline delta-law transform (verify vs run record) ----------
res["A_delta_law_truehost"] = {}
pc73 = load_pc("off", 0.73)
for h in H_NODES:
    pc = pc73 if h == 0.73 else load_pc("off", h)
    th = pc[pc["is_true_host"] == True]
    r = (th["s_4d_zg_mg"] / th["s_bar_phi_zg"]).to_numpy()
    res["A_delta_law_truehost"][str(h)] = dict(n=int(len(r)), median=float(np.median(r)),
        mean=float(np.mean(r)), min=float(np.min(r)), max=float(np.max(r)))
    if h != 0.73: del pc

# ---------- B/C. joint_r1 transform: log-normal smearing at the true hosts ----------
SURF_CFGS = {"base": (0.005, 0.20), "coarse": (0.010, 0.30), "fine": (0.0025, 0.15)}
res["BC_jr1_transform"] = {}
for h in H_NODES:
    pc = pc73 if h == 0.73 else load_pc("off", h)
    th = pc[pc["is_true_host"] == True].copy()
    z_t = th["z_g"].to_numpy(); lnM_t = np.log(th["M_g"].to_numpy())
    sig_t = np.clip((th["M_err_g"] / th["M_g"]).to_numpy(), 1e-3, 3.0)
    sbar_t = th["s_bar_phi_zg"].to_numpy(); s4d_t = th["s_4d_zg_mg"].to_numpy()
    node = {}
    for name, (zb, mb) in SURF_CFGS.items():
        surf = build_surface(pc, zb, mb)
        # profile fidelity at sigma->0: surface value vs the dumped point value
        ls0, _ = surf_eval(surf, zb, z_t, lnM_t)
        fid = np.exp(ls0) / s4d_t
        # expectation form
        Es, clf = smear(surf, zb, z_t, lnM_t, sig_t)
        K_t = Es / s4d_t
        T_exp = np.median(Es / sbar_t)
        # realized-median predictive band (one lognormal draw per host, seed-900001 analogue)
        meds = np.empty(4000)
        for i in range(4000):
            eps = RNG.standard_normal(len(z_t))
            ls, _ = surf_eval(surf, zb, z_t, lnM_t + sig_t * eps)
            meds[i] = np.median(np.exp(ls) / sbar_t)
        node[name] = dict(
            fidelity_median=float(np.median(fid)),
            fid_p10=float(np.quantile(fid, 0.1)), fid_p90=float(np.quantile(fid, 0.9)),
            K_median=float(np.median(K_t)), K_mean=float(np.mean(K_t)),
            T_expectation_median=float(T_exp),
            T_realized_median_q025=float(np.quantile(meds, 0.025)),
            T_realized_median_q50=float(np.quantile(meds, 0.5)),
            T_realized_median_q975=float(np.quantile(meds, 0.975)),
            clamp_weight_mean=float(np.mean(clf)),
            sigma_t_median=float(np.median(sig_t)))
    res["BC_jr1_transform"][str(h)] = node
    if h != 0.73: del pc

# ---------- D. limiting case sigma -> 0 ----------
surf, (zb, mb) = build_surface(pc73, *SURF_CFGS["base"]), SURF_CFGS["base"]
th = pc73[pc73["is_true_host"] == True]
z_t = th["z_g"].to_numpy(); lnM_t = np.log(th["M_g"].to_numpy())
sbar_t = th["s_bar_phi_zg"].to_numpy()
Es0, _ = smear(surf, zb, z_t, lnM_t, np.full(len(z_t), 1e-6))
res["D_sigma0_limit"] = dict(
    T_sigma0_median=float(np.median(Es0 / sbar_t)),
    delta_law_median=res["A_delta_law_truehost"]["0.73"]["median"])

# ---------- E. on/off invariance check (site N1 factor identity, h=0.73) ----------
on73 = load_pc("on", 0.73)
j = pc73.merge(on73, on=["event_idx", "catalog_index"], suffixes=("_off", "_on"))
ok = (j["N_g_used_off"] > 0) & (j["N_g_used_on"] > 0) & (j["s_bar_phi_zg_off"] > 0) & (j["s_4d_zg_mg_off"] > 0)
j = j[ok]
ratio_meas = (j["N_g_used_on"] / j["N_g_used_off"]).to_numpy()
ratio_pred = (j["s_4d_zg_mg_off"] / j["s_bar_phi_zg_off"]).to_numpy()
rr = np.log(ratio_meas) - np.log(ratio_pred)
res["E_invariance"] = dict(n=int(len(j)), median_abs_lnresid=float(np.median(np.abs(rr))),
    p95_abs_lnresid=float(np.quantile(np.abs(rr), 0.95)), max_abs_lnresid=float(np.max(np.abs(rr))))
del on73, j

# ---------- F. dark-class smearing factor R_K ----------
pe = load_pe("off", 0.73)
hostmap = pe.set_index("event_idx")["host_galaxy_index"]
dark_ev = set(pe[pe["host_galaxy_index"] < 0]["event_idx"])
res["F_dark_RK"] = {}
dk = pc73[pc73["event_idx"].isin(dark_ev)].copy()
w = (dk["w_g"] * dk["N_g_used"]).to_numpy()
zc = dk["z_g"].to_numpy(); lnMc = np.log(dk["M_g"].to_numpy())
sigc = np.clip((dk["M_err_g"] / dk["M_g"]).to_numpy(), 1e-3, 3.0)
s4c = dk["s_4d_zg_mg"].to_numpy(); sbc = dk["s_bar_phi_zg"].to_numpy()
okc = np.isfinite(w) & (w > 0) & np.isfinite(s4c) & np.isfinite(sbc) & (sbc > 0)
w, zc, lnMc, sigc, s4c, sbc = w[okc], zc[okc], lnMc[okc], sigc[okc], s4c[okc], sbc[okc]
for name, (zb2, mb2) in SURF_CFGS.items():
    surf2 = build_surface(pc73, zb2, mb2)
    Ec, clf = smear(surf2, zb2, zc, lnMc, sigc)
    # surface-fidelity-corrected point value (same surface for num and denom kills profile bias)
    p0, _ = surf_eval(surf2, zb2, zc, lnMc)
    p0 = np.exp(p0)
    rho_point = float(np.sum(w * p0) / np.sum(w * sbc))
    rho_smear = float(np.sum(w * Ec) / np.sum(w * sbc))
    res["F_dark_RK"][name] = dict(rho_point_surface=rho_point, rho_smeared=rho_smear,
        R_K=rho_smear / rho_point,
        rho_point_dumpvals=float(np.sum(w * s4c) / np.sum(w * sbc)),
        clamp_weight_mean=float(np.mean(clf)), n=int(len(w)),
        sigma_c_median=float(np.median(sigc)))

json.dump(res, open(f"{OUT}/jr1_transform_stage1.json", "w"), indent=1)
print(json.dumps(res, indent=1))
