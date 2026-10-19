"""T1.3 zero-compute instrument (row #255, tree 2 node T1.3).

(1) Per-event s-secants from the T1.2 re-certification CSVs (theta_phi_divisor=on), joined to
    the forensic's f7_events.csv (true-host z_g, sigma_g, ball bounds, n_cand, class, c_i,
    Es_null_det), characterised by class / candidate count / window half-width in sigma_g units.
(2) A capture-fraction model of the candidate ball's s-response under the FIXED (+/-1 sigma_g)
    vs the THETA-CONSISTENT (+/-k s sigma_g) z-window at k in {1, 4}, evaluated on the drawn
    hosts (a sample from w_g S_tilde_g, the weighting the score identity needs), calibrated
    against the measured residual.
No evaluate() call; no source edits; foreground; < 60 s.
"""
import json
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = "/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729"
REC = f"{ROOT}/tree2_20260830/hier_s0_recert_run"
F7 = f"{ROOT}/fanout1_20260829/b1_1_forensic_work/f7_events.csv"
OUT = f"{ROOT}/tree2_20260830/t1_3_gate_work/t13_out.json"
SEEDS = [900101, 900102, 900103, 900104]
NODES = ["truth", "b_plus", "b_minus", "s_plus", "s_minus"]
SQ2 = math.sqrt(2.0)
LN2 = math.log(2.0)
LIN = SQ2 - 1.0 / SQ2  # 0.70711

def load_node(seed, node):
    p = f"{REC}/s0a_seed{seed}/node_{node}_sites2.2_nosmear_divisor/simulations/diagnostics/event_likelihoods.csv"
    df = pd.read_csv(p)
    df = df[np.isclose(df["h"], 0.73)]
    df = df.drop_duplicates(subset=["event_idx"], keep="last")
    return df.set_index("event_idx")

rows = []
for seed in SEEDS:
    d = {n: load_node(seed, n) for n in NODES}
    idx = d["truth"].index
    for n in NODES:
        assert set(d[n].index) == set(idx), (seed, n)
    for e in idx:
        ln = {n: math.log(d[n].loc[e, "combined_no_bh"]) for n in NODES}
        lcat = d["truth"].loc[e, "L_cat_no_bh"]
        rows.append(dict(seed=seed, event_idx=int(e),
            sb=(ln["b_plus"] - ln["b_minus"]) / 0.04,
            ss_lin=(ln["s_plus"] - ln["s_minus"]) / LIN,
            ss_lns=(ln["s_plus"] - ln["s_minus"]) / LN2,
            L_cat_truth=lcat, dark=(lcat == 0.0)))
r = pd.DataFrame(rows)
f7 = pd.read_csv(F7)
m = r.merge(f7[["seed", "event_idx", "z_true", "z_g", "sigma_g", "z_GW", "sig_zGW", "z_min_ball", "z_max_ball",
                "n_cand", "pi_true", "c_nb", "sky_in", "z_in", "recovered", "sig_dL_rel", "Es_null_det",
                "ss_nb", "tw_ss", "mu_k", "sd_k"]], on=["seed", "event_idx"], how="left", validate="1:1")
assert len(m) == 461, len(m)
m["hw_sig"] = (m["z_max_ball"] - m["z_min_ball"]) / 2.0 / m["sigma_g"]
m["zeta"] = m["z_g"] / m["sigma_g"]
m["f_lo"] = m["z_min_ball"] / m["z_GW"]
m["f_hi"] = m["z_max_ball"] / m["z_GW"]
# PA-HIER-32 debiased statistic; two conventions for the combined-channel expectation
m["ss_deb_raw"] = m["ss_lns"] - m["Es_null_det"]
m["ss_deb_c"] = m["ss_lns"] - m["c_nb"] * m["Es_null_det"]

def stat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x); mu = x.mean(); sem = x.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return dict(n=int(n), mean=float(mu), sem=float(sem), Z=float(mu / sem) if n > 1 else float("nan"))

out = {"pooled_recert": {k: stat(m[k]) for k in ["sb", "ss_lin", "ss_lns", "ss_deb_raw", "ss_deb_c"]},
       "Es_null_det": {"mean_unweighted": float(m["Es_null_det"].mean()), "mean_c_weighted": float((m["c_nb"] * m["Es_null_det"]).mean())},
       "check_vs_s0a_score_md": {"score_b_expected": -0.28878240960372603, "score_s_lin_expected": -0.07195958393659582}}
out["per_seed"] = {int(s): {k: stat(g[k]) for k in ["sb", "ss_lin", "ss_deb_c"]} for s, g in m.groupby("seed")}
# cross-check: banked (divisor off) per-event ss_nb from f7 vs recert (divisor on): the difference is the per-node rho term only
out["bank_vs_recert_ss"] = {"bank_ss_nb_mean": float(m["ss_nb"].mean()), "recert_ss_lin_mean": float(m["ss_lin"].mean()),
                            "corr": float(np.corrcoef(m["ss_nb"], m["ss_lin"])[0, 1])}
# classes
cls = {}
mm = m[~m["dark"]]
cls["all_nondark"] = {k: stat(mm[k]) for k in ["ss_lin", "ss_deb_c"]}
cls["dark"] = {k: stat(m[m["dark"]][k]) for k in ["ss_lin"]}
cls["recovered"] = {k: stat(mm[mm["recovered"] == True][k]) for k in ["ss_lin", "ss_deb_c"]}
cls["not_recovered"] = {k: stat(mm[mm["recovered"] == False][k]) for k in ["ss_lin", "ss_deb_c"]}
cls["sky_out"] = {k: stat(mm[mm["sky_in"] == False][k]) for k in ["ss_lin", "ss_deb_c"]}
cls["z_out"] = {k: stat(mm[mm["z_in"] == False][k]) for k in ["ss_lin", "ss_deb_c"]}
out["by_class"] = cls
# by candidate count quartile, by window half-width quartile, by zeta quartile, by c quartile
def by_quartile(col):
    q = pd.qcut(mm[col], 4, labels=False, duplicates="drop")
    res = {}
    for qi, g in mm.groupby(q):
        res[f"q{int(qi)+1}"] = dict(range=[float(g[col].min()), float(g[col].max())], median=float(g[col].median()),
                                   ss_lin=stat(g["ss_lin"]), ss_deb_c=stat(g["ss_deb_c"]), c_mean=float(g["c_nb"].mean()))
    return res
out["by_n_cand_quartile"] = by_quartile("n_cand")
out["by_hw_sig_quartile"] = by_quartile("hw_sig")
out["by_zeta_quartile"] = by_quartile("zeta")
out["by_c_quartile"] = by_quartile("c_nb")
out["by_sig_dL_rel_quartile"] = by_quartile("sig_dL_rel")
out["corr"] = {c: float(np.corrcoef(mm[c], mm["ss_deb_c"])[0, 1]) for c in ["hw_sig", "zeta", "c_nb", "sig_dL_rel", "pi_true"]}
out["corr"]["log_n_cand"] = float(np.corrcoef(np.log(mm["n_cand"].clip(1)), mm["ss_deb_c"])[0, 1])

# ---- (2) capture-fraction model on the drawn hosts ----
def capture(zg, sg, f_lo, f_hi, s, k, consistent, b=0.0):
    """P_{z_s ~ N(zg_th, s sg)}( f_lo z_s - k sig_sel <= zg_th <= f_hi z_s + k sig_sel ),
    sig_sel = s*sg (theta-consistent) or sg (fixed); zg_th = zg + b(1+zg). sigma_pv = 0 (constants.py:95)."""
    zg_th = zg + b * (1.0 + zg)
    # galaxy-side selection term: FIXED = listed z_g +/- k sigma_g (never moves under theta);
    # CONSISTENT = transformed centre z_g^theta +/- k s sigma_g (moves with the kernel).
    if consistent:
        z_lo = (zg_th - k * s * sg) / f_hi
        z_hi = (zg_th + k * s * sg) / f_lo
    else:
        z_lo = (zg - k * sg) / f_hi
        z_hi = (zg + k * sg) / f_lo
    w = s * sg
    return norm.cdf((z_hi - zg_th) / w) - norm.cdf((z_lo - zg_th) / w)

model = {}
sub = mm.copy()
for name, cons in [("fixed", False), ("consistent", True)]:
    for k in [1.0, 2.0, 3.0, 4.0]:
        C = {}
        for lab, s in [("s_minus", 1 / SQ2), ("s1", 1.0), ("s_plus", SQ2)]:
            c = capture(sub["z_g"].values, sub["sigma_g"].values, sub["f_lo"].values, sub["f_hi"].values, s, k, cons)
            C[lab] = c
        Cm = {lab: float(np.mean(c)) for lab, c in C.items()}
        Cc = {lab: float(np.mean(sub["c_nb"].values * c) / np.mean(sub["c_nb"].values)) for lab, c in C.items()}
        sec_lns = (math.log(Cm["s_plus"]) - math.log(Cm["s_minus"])) / LN2
        sec_lns_c = (math.log(Cc["s_plus"]) - math.log(Cc["s_minus"])) / LN2
        # b-axis at s=1, k: capture secant in b
        Cb = {}
        for lab, b in [("b_minus", -0.02), ("b_plus", 0.02)]:
            Cb[lab] = float(np.mean(capture(sub["z_g"].values, sub["sigma_g"].values, sub["f_lo"].values, sub["f_hi"].values, 1.0, k, cons, b=b)))
        sec_b = (math.log(Cb["b_plus"]) - math.log(Cb["b_minus"])) / 0.04
        model[f"{name}_k{k:g}"] = dict(C_mean=Cm, C_cweighted=Cc, secant_lnC_lns=sec_lns, secant_lnC_lns_cweighted=sec_lns_c,
                                      Cb_mean=Cb, secant_lnC_b=sec_b,
                                      registered_shift_lns=float(np.mean(sub["c_nb"]) * sec_lns),
                                      registered_shift_b=float(np.mean(sub["c_nb"]) * sec_b))
out["capture_model"] = model
out["capture_model_notes"] = {
    "c_nb_mean": float(np.mean(sub["c_nb"])),
    "kernel": "bare Gaussian N(z; z_g^theta, s sigma_g), sigma_pv = 0; C7 tilt and S_bar_phi weighting ignored (disclosed)",
    "envelope": "per-event f_lo = z_min_ball/z_GW, f_hi = z_max_ball/z_GW from f7_events.csv (E6-exact ball bounds)",
}
# what the flag alone (consistent, k=1) predicts relative to the measured T1.2 residual
meas = out["pooled_recert"]["ss_deb_c"]["mean"]
meas_lin = out["pooled_recert"]["ss_lin"]["mean"]
pred = {}
for cfg in ["consistent_k1", "consistent_k2", "consistent_k3", "consistent_k4", "fixed_k4"]:
    delta = model[cfg]["registered_shift_lns"] - model["fixed_k1"]["registered_shift_lns"]
    pred[cfg] = dict(delta_lns=delta, predicted_ss_deb_c=meas + delta, predicted_Z=(meas + delta) / out["pooled_recert"]["ss_deb_c"]["sem"],
                     removed_fraction_of_fixed_term=float(1.0 - model[cfg]["registered_shift_lns"] / model["fixed_k1"]["registered_shift_lns"]))
out["prediction"] = dict(measured_T12_ss_deb_c=meas, measured_T12_ss_lin=meas_lin, model_fixed_k1_registered_term=model["fixed_k1"]["registered_shift_lns"], by_config=pred)
# residual by hw_sig quartile predicted by the model (fixed k=1): does the model's per-event capture secant track the measured residual by quartile?
sub["cap_sec_fixed1"] = (np.log(capture(sub["z_g"].values, sub["sigma_g"].values, sub["f_lo"].values, sub["f_hi"].values, SQ2, 1.0, False))
                         - np.log(capture(sub["z_g"].values, sub["sigma_g"].values, sub["f_lo"].values, sub["f_hi"].values, 1 / SQ2, 1.0, False))) / LN2
q = pd.qcut(sub["hw_sig"], 4, labels=False)
out["model_vs_measured_by_hw_sig_quartile"] = {f"q{int(qi)+1}": dict(hw_sig_median=float(g["hw_sig"].median()),
    model_c_x_capsec_mean=float((g["c_nb"] * g["cap_sec_fixed1"]).mean()), measured_ss_deb_c=stat(g["ss_deb_c"])) for qi, g in sub.groupby(q)}
json.dump(out, open(OUT, "w"), indent=1, default=float)
# ---- (3) candidate-count growth estimate for z_window_k = 4 at sky_cone_k = 1.5 ----
def extent_count(zmin, zmax, sg, k):
    lo = np.maximum(zmin - k * sg, 0.0); hi = zmax + k * sg
    return (hi**3 - lo**3) / 3.0  # integral of z^2 dz: comoving-volume proxy for the count in the z-slab
g1 = extent_count(sub["z_min_ball"].values, sub["z_max_ball"].values, sub["sigma_g"].values, 1.0)
g4 = extent_count(sub["z_min_ball"].values, sub["z_max_ball"].values, sub["sigma_g"].values, 4.0)
g2 = extent_count(sub["z_min_ball"].values, sub["z_max_ball"].values, sub["sigma_g"].values, 2.0)
ratio4 = g4 / g1; ratio2 = g2 / g1
lin4 = ((sub["z_max_ball"] - sub["z_min_ball"]) + 8 * sub["sigma_g"]) / ((sub["z_max_ball"] - sub["z_min_ball"]) + 2 * sub["sigma_g"])
out["count_growth_k4_sky15"] = dict(z2_weighted_ratio_median=float(np.median(ratio4)), z2_weighted_ratio_mean=float(np.mean(ratio4)),
    z2_weighted_ratio_q25_q75=[float(np.percentile(ratio4, 25)), float(np.percentile(ratio4, 75))],
    linear_extent_ratio_median=float(np.median(lin4)),
    n_cand_weighted_ratio=float(np.sum(sub["n_cand"] * ratio4) / np.sum(sub["n_cand"])),
    k2_z2_weighted_ratio_median=float(np.median(ratio2)))
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("COUNT_GROWTH", json.dumps(out["count_growth_k4_sky15"]))


# ---- (4) tilted-kernel variant: source z_s ~ N(mu_k + b(1+z_g), s*sd_k) (the C7-core kernel moments of f7,
#          E15 tilt included to first order); selection term unchanged (listed z_g +/- k sigma_g, or the
#          theta-transformed z_g^theta +/- k s sigma_g). Bare variant above is retained for comparison.
def capture_tilted(zg, sg, muk, sdk, f_lo, f_hi, s, k, consistent, b=0.0):
    zg_th = zg + b * (1.0 + zg)
    mu = muk + b * (1.0 + zg)
    w = s * sdk
    if consistent:
        z_lo = (zg_th - k * s * sg) / f_hi; z_hi = (zg_th + k * s * sg) / f_lo
    else:
        z_lo = (zg - k * sg) / f_hi; z_hi = (zg + k * sg) / f_lo
    return norm.cdf((z_hi - mu) / w) - norm.cdf((z_lo - mu) / w)

tm = {}
A = (sub["z_g"].values, sub["sigma_g"].values, sub["mu_k"].values, sub["sd_k"].values, sub["f_lo"].values, sub["f_hi"].values)
cbar = float(np.mean(sub["c_nb"]))
for name, cons in [("fixed", False), ("consistent", True)]:
    for k in [1.0, 2.0, 3.0, 4.0]:
        Cs = {lab: float(np.mean(capture_tilted(*A, s, k, cons))) for lab, s in [("s_minus", 1 / SQ2), ("s1", 1.0), ("s_plus", SQ2)]}
        Cb = {lab: float(np.mean(capture_tilted(*A, 1.0, k, cons, b=b))) for lab, b in [("b_minus", -0.02), ("b_plus", 0.02)]}
        sec_s = (math.log(Cs["s_plus"]) - math.log(Cs["s_minus"])) / LN2
        sec_b = (math.log(Cb["b_plus"]) - math.log(Cb["b_minus"])) / 0.04
        tm[f"{name}_k{k:g}"] = dict(C=Cs, Cb=Cb, secant_lnC_lns=sec_s, secant_lnC_b=sec_b, registered_shift_lns=cbar * sec_s, registered_shift_b=cbar * sec_b)
out["capture_model_tilted"] = tm
predt = {}
for cfg in ["consistent_k1", "consistent_k2", "consistent_k3", "consistent_k4", "fixed_k4"]:
    ds = tm[cfg]["registered_shift_lns"] - tm["fixed_k1"]["registered_shift_lns"]
    db = tm[cfg]["registered_shift_b"] - tm["fixed_k1"]["registered_shift_b"]
    predt[cfg] = dict(delta_lns=ds, predicted_ss_deb_c=meas + ds, predicted_Z=(meas + ds) / out["pooled_recert"]["ss_deb_c"]["sem"],
                      removed_fraction=float(1 - tm[cfg]["registered_shift_lns"] / tm["fixed_k1"]["registered_shift_lns"]),
                      delta_score_b=db, predicted_score_b=out["pooled_recert"]["sb"]["mean"] + db,
                      predicted_Z_b=(out["pooled_recert"]["sb"]["mean"] + db) / out["pooled_recert"]["sb"]["sem"])
out["prediction_tilted"] = dict(model_fixed_k1_registered_term_s=tm["fixed_k1"]["registered_shift_lns"], model_fixed_k1_registered_term_b=tm["fixed_k1"]["registered_shift_b"], by_config=predt)
# tilted model by hw_sig quartile (fixed k=1), per-event c x capture secant
cs = (np.log(capture_tilted(*A, SQ2, 1.0, False)) - np.log(capture_tilted(*A, 1 / SQ2, 1.0, False))) / LN2
sub["cap_sec_tilted_fixed1"] = cs
out["tilted_model_vs_measured_by_hw_sig_quartile"] = {f"q{int(qi)+1}": dict(model=float((g["c_nb"] * g["cap_sec_tilted_fixed1"]).mean()), measured=float(g["ss_deb_c"].mean()), sem=float(g["ss_deb_c"].std(ddof=1) / math.sqrt(len(g)))) for qi, g in sub.groupby(q)}
# E9 comparison: catalogue-leg (unweighted) b-secant change fixed_k1 -> enlarged (fixed k=4) measured -0.77 (1.94 -> 1.17)
out["E9_b_check"] = dict(bare_model_catleg_delta_b=model["fixed_k4"]["secant_lnC_b"] - model["fixed_k1"]["secant_lnC_b"],
                         tilted_model_catleg_delta_b=tm["fixed_k4"]["secant_lnC_b"] - tm["fixed_k1"]["secant_lnC_b"],
                         measured_E9_catleg_delta_b=1.17 - 1.94, measured_E9_catleg_delta_b_err=0.7,
                         bare_model_catleg_delta_s=model["fixed_k4"]["secant_lnC_lns"] - model["fixed_k1"]["secant_lnC_lns"],
                         tilted_model_catleg_delta_s=tm["fixed_k4"]["secant_lnC_lns"] - tm["fixed_k1"]["secant_lnC_lns"],
                         measured_E9_catleg_delta_s_lin=0.019 + 0.052, note="E9 twin: sky 3.0 AND z +/-4 sigma_g; secants in the linear /0.70711 form; lns form = x1.0201")
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("TILTED")
for k, v in tm.items():
    print(k, "C(s-,1,s+)=", [round(v["C"][x], 4) for x in ["s_minus", "s1", "s_plus"]], "sec_s %.4f reg %.4f | Cb=%s sec_b %.3f reg_b %.3f" % (v["secant_lnC_lns"], v["registered_shift_lns"], [round(v["Cb"][x], 4) for x in ["b_minus", "b_plus"]], v["secant_lnC_b"], v["registered_shift_b"]))
print(json.dumps(out["prediction_tilted"], indent=0))
print(json.dumps(out["tilted_model_vs_measured_by_hw_sig_quartile"], indent=0))
print(json.dumps(out["E9_b_check"], indent=0))
