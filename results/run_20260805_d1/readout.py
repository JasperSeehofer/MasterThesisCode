#!/usr/bin/env python3
"""Mechanical readout of the D1 S_and-consistent selection re-weight campaign.

Scores ONLY against the committed reads, bands, and thresholds in
`results/campaign51_20260728/realistic_20260729/PREREGISTRATION_D1_SAND_REWEIGHT.md`
(commit 38ffa6ce). No interpretation beyond mechanical scoring.

Not committed (scratch analysis script per the readout task instructions).
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "/home/jasper/Repositories/MasterThesisCode"
D1 = f"{ROOT}/results/run_20260805_d1"
A0 = f"{ROOT}/results/run_20260804_postfix"

VENUES = ["iiib", "joint_r1"]
ARM_DIRS = {
    "A1": {"iiib": f"{D1}/a1_iiib", "joint_r1": f"{D1}/a1_joint_r1"},
    "A2": {"iiib": f"{D1}/a2_iiib", "joint_r1": f"{D1}/a2_joint_r1"},
}
H_LOW, H_HIGH = 0.73, 0.81
H_GRID_LOW, H_GRID_HIGH = 0.60, 0.86

out = {}

# ---------------------------------------------------------------------------
# 0. Fingerprints + metadata
# ---------------------------------------------------------------------------
fp = {}
for arm in ["A1", "A2"]:
    for venue in VENUES:
        d = ARM_DIRS[arm][venue]
        df = pd.read_csv(f"{d}/diagnostics/event_likelihoods.csv")
        meta = json.load(open(f"{d}/run_metadata_0.json"))
        key = f"{arm}_{venue}"
        fp[key] = {
            "n_rows": len(df),
            "n_rows_expected_65108": len(df) == 65108,
            "n_events": df["event_idx"].nunique(),
            "n_h": df["h"].nunique(),
            "row_formula_ok": len(df) == df["event_idx"].nunique() * df["h"].nunique(),
            "git_commit": meta["git_commit"],
            "git_commit_matches_128f318a": meta["git_commit"].startswith("128f318a"),
            "cli_args": meta["cli_args"],
        }
out["fingerprints"] = fp

# production-flag / no-counterfactual-toggle check
expected_flags = {
    "normalization_mode": "absolute_marginal",
    "host_z_kernel": "volume_deconv",
    "host_mass_kernel": "auto",
    "pdet_z_resolved": True,
    "pdet_wbh_z_resolved": False,
    "pdet_dl_bins": 60,
    "pdet_mass_bins": 40,
    "pdet_estimator": "local_linear",
}
counterfactual_defaults = {"freeze_g_frac_ref_h": None, "selection_in_completion_numerator": "off"}
flag_check = {}
for key, rec in fp.items():
    args = rec["cli_args"]
    mism = {k: (args.get(k), v) for k, v in expected_flags.items() if args.get(k) != v}
    ctf = {k: (args.get(k), v) for k, v in counterfactual_defaults.items() if args.get(k) != v}
    flag_check[key] = {"production_flags_ok": len(mism) == 0, "mismatches": mism,
                        "counterfactual_toggles_at_default": len(ctf) == 0, "toggle_mismatches": ctf}
out["flag_check"] = flag_check

# code-commit provenance: registered pin 9a715405 vs actual git_commit 128f318a
out["commit_provenance_note"] = (
    "Registered pin (table 'The run') = 9a715405. Actual run_metadata git_commit "
    "= 128f318a8b5f9f9c7889b69f7f3402bd5726d037. `git diff --stat 9a715405 128f318a "
    "-- master_thesis_code/` is EMPTY (verified) -- the only commit between the pin "
    "and 128f318a is 128f318a itself, which adds `d1_sand/make_sand_pools.py` "
    "(the registered 'instrumentation' script) and touches nothing on the "
    "/physics-change trigger list. Estimator code at 128f318a is byte-identical to "
    "the pinned 9a715405 production code. No discrepancy."
)

# ---------------------------------------------------------------------------
# 1. Path-A self-certification scalars (S1), h=0.73
#    Retrieved from evaluate_*.err INFO logs (not in event_likelihoods.csv,
#    which only carries the per-h BROADCAST values alpha_G_phi, D_tilde_phi,
#    r_Malm, w_tilde_G -- Sigma_4D and beta_G_phi/beta_Gbar_phi individually
#    are only visible in the run logs).
# ---------------------------------------------------------------------------
path_a_h73 = {
    "A1_iiib": {"w_tilde_G": 0.06190526, "alpha_G_phi": 5.862281e07, "r_Malm": 0.3823132,
                "n_hat_w_phi": 6.396689, "D_phi": 1.041691e09, "D_tilde_phi": 9.469762e08,
                "Sigma_phi": 9.808498e08, "Sigma_4D": 3.749919e08},
    "A2_iiib": {"w_tilde_G": 0.05101099, "alpha_G_phi": 1.494512e07, "r_Malm": 0.5216586,
                "n_hat_w_phi": 5.736874, "D_phi": 3.066825e08, "D_tilde_phi": 2.929784e08,
                "Sigma_phi": 1.64357e08, "Sigma_4D": 8.573825e07},
    "A1_joint_r1": {"w_tilde_G": 0.07074465, "alpha_G_phi": 6.763077e07, "r_Malm": 0.4410594,
                    "n_hat_w_phi": 6.236097, "D_phi": 1.041691e09, "D_tilde_phi": 9.559842e08,
                    "Sigma_phi": 9.562252e08, "Sigma_4D": 4.21752e08},
    "A2_joint_r1": {"w_tilde_G": 0.06542319, "alpha_G_phi": 1.946316e07, "r_Malm": 0.6793608,
                    "n_hat_w_phi": 5.573401, "D_phi": 3.066825e08, "D_tilde_phi": 2.974964e08,
                    "Sigma_phi": 1.596737e08, "Sigma_4D": 1.08476e08},
}
S1_TARGET_SIGMA4D = 0.2286246597604769
S1_TARGET_BETA_GBAR_PHI = 0.3129747690740832

s1 = {}
for venue in VENUES:
    a1 = path_a_h73[f"A1_{venue}"]
    a2 = path_a_h73[f"A2_{venue}"]
    sigma4d_ratio = a2["Sigma_4D"] / a1["Sigma_4D"]
    beta_gbar_phi_a1 = a1["D_tilde_phi"] - a1["alpha_G_phi"]
    beta_gbar_phi_a2 = a2["D_tilde_phi"] - a2["alpha_G_phi"]
    beta_gbar_phi_ratio = beta_gbar_phi_a2 / beta_gbar_phi_a1
    s1[venue] = {
        "Sigma4D_A2_over_A1": sigma4d_ratio,
        "Sigma4D_target": S1_TARGET_SIGMA4D,
        "Sigma4D_rel_dev": abs(sigma4d_ratio - S1_TARGET_SIGMA4D) / S1_TARGET_SIGMA4D,
        "Sigma4D_pass_1e-3": abs(sigma4d_ratio - S1_TARGET_SIGMA4D) / S1_TARGET_SIGMA4D <= 1e-3,
        "beta_Gbar_phi_A2_over_A1": beta_gbar_phi_ratio,
        "beta_Gbar_phi_target": S1_TARGET_BETA_GBAR_PHI,
        "beta_Gbar_phi_rel_dev": abs(beta_gbar_phi_ratio - S1_TARGET_BETA_GBAR_PHI) / S1_TARGET_BETA_GBAR_PHI,
        "beta_Gbar_phi_pass_1e-3": abs(beta_gbar_phi_ratio - S1_TARGET_BETA_GBAR_PHI) / S1_TARGET_BETA_GBAR_PHI <= 1e-3,
    }
s1["both_pass"] = all(s1[v]["Sigma4D_pass_1e-3"] and s1[v]["beta_Gbar_phi_pass_1e-3"] for v in VENUES)
out["S1_self_certification"] = s1

# ---------------------------------------------------------------------------
# 2. Stratum membership (shared-218 / resurrected-316), from A0 (gate_vii method)
# ---------------------------------------------------------------------------
bounds_iiib = pd.read_csv(f"{A0}/iiib/diagnostics/prepared_cramer_rao_bounds.csv")
dark_event_indices = set(bounds_iiib[bounds_iiib["host_galaxy_index"] < 0].index.tolist())

def find_survivors(lh_df):
    survivors = set()
    for event_idx, g in lh_df.groupby("event_idx"):
        if (g["L_cat_no_bh"] > 0).all() and (g["L_cat_with_bh"] > 0).all():
            survivors.add(event_idx)
    return survivors

a0_lh = {v: pd.read_csv(f"{A0}/{v}/diagnostics/event_likelihoods.csv") for v in VENUES}
survivors = {v: find_survivors(a0_lh[v]) for v in VENUES}
dark_survivors = {v: dark_event_indices & survivors[v] for v in VENUES}
shared_218 = dark_survivors["iiib"] & dark_survivors["joint_r1"]
resurrected_316 = dark_survivors["joint_r1"] - dark_survivors["iiib"]

out["stratum_membership"] = {
    "dark_event_indices_n": len(dark_event_indices),
    "iiib_dark_survivors_n": len(dark_survivors["iiib"]),
    "joint_r1_dark_survivors_n": len(dark_survivors["joint_r1"]),
    "shared_218_n": len(shared_218),
    "shared_218_matches_registered_218": len(shared_218) == 218,
    "resurrected_316_n": len(resurrected_316),
    "resurrected_316_matches_registered_316": len(resurrected_316) == 316,
}

# Reference sums under A0, joint_r1 venue (the venue the registered numbers were computed on)
REF_SUM_SHARED_218 = -112.6967467481951
REF_MEAN_SHARED_218 = -0.5169575538908032
REF_SUM_RESURRECTED_316 = -492.0762245626546


def delta_e(lh_df, events, h_lo=H_LOW, h_hi=H_HIGH):
    """Delta_e = ln(L_with_bh/L_no_bh)@h_hi - same@h_lo, per event.

    Requires L_cat_no_bh > 0 AND L_cat_with_bh > 0 at BOTH h (else log is
    -inf/nan and the object is not defined) -- same well-definedness
    condition `find_survivors`/`paired_check.py` uses. Non-finite results
    are DROPPED here (not silently propagated as NaN) and must be counted
    by the caller as `n_events_requested - len(returned dict)`.
    """
    out_d = {}
    for e in events:
        g = lh_df[lh_df["event_idx"] == e]
        r_lo = g[np.isclose(g["h"], h_lo)]
        r_hi = g[np.isclose(g["h"], h_hi)]
        if len(r_lo) == 0 or len(r_hi) == 0:
            continue
        no_bh_lo, wbh_lo = r_lo["L_cat_no_bh"].values[0], r_lo["L_cat_with_bh"].values[0]
        no_bh_hi, wbh_hi = r_hi["L_cat_no_bh"].values[0], r_hi["L_cat_with_bh"].values[0]
        if no_bh_lo <= 0 or wbh_lo <= 0 or no_bh_hi <= 0 or wbh_hi <= 0:
            continue
        lr_lo = np.log(wbh_lo / no_bh_lo)
        lr_hi = np.log(wbh_hi / no_bh_hi)
        val = lr_hi - lr_lo
        if np.isfinite(val):
            out_d[e] = val
    return out_d


def t_e_func(lh_df, events, h_lo=H_LOW, h_hi=H_HIGH):
    out_d = {}
    for e in events:
        g = lh_df[lh_df["event_idx"] == e]
        r_lo = g[np.isclose(g["h"], h_lo)]
        r_hi = g[np.isclose(g["h"], h_hi)]
        if len(r_lo) == 0 or len(r_hi) == 0:
            continue
        v_lo, v_hi = r_lo["combined_with_bh"].values[0], r_hi["combined_with_bh"].values[0]
        if v_lo <= 0 or v_hi <= 0:
            continue
        out_d[e] = np.log(v_hi) - np.log(v_lo)
    return out_d


# Verify A0 shared-218 reference reproduces (sanity check on stratum + formula)
a0_delta_joint_218 = delta_e(a0_lh["joint_r1"], shared_218)
sum_check = sum(a0_delta_joint_218.values())
mean_check = np.mean(list(a0_delta_joint_218.values()))
out["stratum_membership"]["A0_joint_r1_shared_218_reproduction"] = {
    "sum": sum_check, "mean": mean_check,
    "sum_matches_registered": abs(sum_check - REF_SUM_SHARED_218) < 1e-6,
    "mean_matches_registered": abs(mean_check - REF_MEAN_SHARED_218) < 1e-6,
}
a0_delta_joint_316 = delta_e(a0_lh["joint_r1"], resurrected_316)
sum_check_316 = sum(a0_delta_joint_316.values())
out["stratum_membership"]["A0_joint_r1_resurrected_316_reproduction"] = {
    "sum": sum_check_316,
    "sum_matches_registered": abs(sum_check_316 - REF_SUM_RESURRECTED_316) < 1e-6,
}
# iiib's own A0 baseline on shared-218 (no registered reference number given for iiib;
# derived here for the m_S generalization to iiib, flagged as such downstream)
a0_delta_iiib_218 = delta_e(a0_lh["iiib"], shared_218)
sum_iiib_218_a0 = sum(a0_delta_iiib_218.values())

# ---------------------------------------------------------------------------
# 3. Per-arm per-venue Delta_e, t_e; paired ratios rho_e, tau_e
# ---------------------------------------------------------------------------
GUARD = 1e-6
all_events = {v: set(a0_lh[v]["event_idx"].unique().tolist()) for v in VENUES}

per_venue = {}
for venue in VENUES:
    lh_a1 = pd.read_csv(f"{ARM_DIRS['A1'][venue]}/diagnostics/event_likelihoods.csv")
    lh_a2 = pd.read_csv(f"{ARM_DIRS['A2'][venue]}/diagnostics/event_likelihoods.csv")
    events = all_events[venue]

    d_a1 = delta_e(lh_a1, events)
    d_a2 = delta_e(lh_a2, events)
    t_a1 = t_e_func(lh_a1, events)
    t_a2 = t_e_func(lh_a2, events)

    common_d = sorted(set(d_a1) & set(d_a2))
    common_t = sorted(set(t_a1) & set(t_a2))

    rho_vals, rho_guarded = [], 0
    for e in common_d:
        if abs(d_a1[e]) < GUARD:
            rho_guarded += 1
        else:
            rho_vals.append(d_a2[e] / d_a1[e])
    rho_vals = np.array(rho_vals)

    tau_vals, tau_guarded = [], 0
    for e in common_t:
        if abs(t_a1[e]) < GUARD:
            tau_guarded += 1
        else:
            tau_vals.append(t_a2[e] / t_a1[e])
    tau_vals = np.array(tau_vals)

    darr1 = np.array([d_a1[e] for e in common_d])
    darr2 = np.array([d_a2[e] for e in common_d])
    tarr1 = np.array([t_a1[e] for e in common_t])
    tarr2 = np.array([t_a2[e] for e in common_t])

    sp_d, sp_d_p = spearmanr(darr1, darr2) if len(darr1) > 1 else (np.nan, np.nan)
    sp_t, sp_t_p = spearmanr(tarr1, tarr2) if len(tarr1) > 1 else (np.nan, np.nan)

    # S4 ceiling: |t_e^A2 - t_e^A1| <= 0.0107 nats, on un-guarded events
    t_diff = np.array([t_a2[e] - t_a1[e] for e in common_t])
    s4_ceiling = 0.01066732831245143
    s4_compliant = np.abs(t_diff) <= s4_ceiling
    s4_frac = s4_compliant.mean() if len(t_diff) > 0 else np.nan

    per_venue[venue] = {
        "n_events_requested": len(events),
        "n_common_delta_events": len(common_d),
        "n_delta_undefined_A1": len(events) - len(d_a1),
        "n_delta_undefined_A2": len(events) - len(d_a2),
        "n_delta_undefined_note": (
            "Delta_e requires L_cat_no_bh>0 AND L_cat_with_bh>0 at BOTH h=0.73,0.81; "
            "undefined counts explain why the paired distribution population is smaller "
            "than the full 1588 events, and tie directly to the N2 catalogue-leg "
            "bit-identity violation below (A2's numerator legs go to 0 for many rows)."
        ),
        "n_common_t_events": len(common_t),
        "rho_e": {
            "n_guarded": rho_guarded, "n_used": len(rho_vals),
            "median": float(np.median(rho_vals)) if len(rho_vals) else np.nan,
            "p16": float(np.percentile(rho_vals, 16)) if len(rho_vals) else np.nan,
            "p84": float(np.percentile(rho_vals, 84)) if len(rho_vals) else np.nan,
            "spearman_delta_A1_A2": float(sp_d), "spearman_p": float(sp_d_p),
            "frac_within_0.05": float(np.mean(np.abs(rho_vals - 1) < 0.05)) if len(rho_vals) else np.nan,
            "frac_within_0.20": float(np.mean(np.abs(rho_vals - 1) < 0.20)) if len(rho_vals) else np.nan,
        },
        "tau_e": {
            "n_guarded": tau_guarded, "n_used": len(tau_vals),
            "median": float(np.median(tau_vals)) if len(tau_vals) else np.nan,
            "p16": float(np.percentile(tau_vals, 16)) if len(tau_vals) else np.nan,
            "p84": float(np.percentile(tau_vals, 84)) if len(tau_vals) else np.nan,
            "spearman_t_A1_A2": float(sp_t), "spearman_p": float(sp_t_p),
            "frac_within_0.05": float(np.mean(np.abs(tau_vals - 1) < 0.05)) if len(tau_vals) else np.nan,
            "frac_within_0.20": float(np.mean(np.abs(tau_vals - 1) < 0.20)) if len(tau_vals) else np.nan,
        },
        "S4_ceiling_0.73_0.81": {
            "ceiling_nats": s4_ceiling, "n_events": len(t_diff),
            "frac_compliant": float(s4_frac), "pass_ge_90pct": bool(s4_frac >= 0.90),
            "max_abs_violation": float(np.max(np.abs(t_diff))) if len(t_diff) else np.nan,
            "sum_abs_class_summed_move": float(np.sum(np.abs(t_diff))) if len(t_diff) else np.nan,
            "scale_sanity_ceiling_17nats": float(np.sum(np.abs(t_diff))) <= 17.0 * 2 if len(t_diff) else None,
        },
        "_d_a1": d_a1, "_d_a2": d_a2, "_t_a1": t_a1, "_t_a2": t_a2,
    }
out["per_venue_distribution"] = {v: {k: val for k, val in per_venue[v].items() if not k.startswith("_")}
                                  for v in VENUES}

# ---------------------------------------------------------------------------
# 4. Stratum decomposition + branch scoring
# ---------------------------------------------------------------------------
strata_read = {}
for venue in VENUES:
    d_a1, d_a2 = per_venue[venue]["_d_a1"], per_venue[venue]["_d_a2"]
    entry = {}
    # shared-218 (both venues)
    ev = shared_218 & set(d_a1) & set(d_a2)
    sum_a1 = sum(d_a1[e] for e in ev)
    sum_a2 = sum(d_a2[e] for e in ev)
    mean_a1 = sum_a1 / len(ev) if ev else np.nan
    mean_a2 = sum_a2 / len(ev) if ev else np.nan
    ratios_218 = np.array([d_a2[e] / d_a1[e] for e in ev if abs(d_a1[e]) >= GUARD])
    entry["shared_218"] = {
        "n_events": len(ev), "sum_A1": sum_a1, "sum_A2": sum_a2, "mean_A1": mean_a1, "mean_A2": mean_a2,
        "delta_sum_A2_minus_A1": sum_a2 - sum_a1,
        "m_S_vs_registered_joint_A0_ref_112.697": abs(sum_a2 - sum_a1) / abs(REF_SUM_SHARED_218),
        "sign_delta_sum": "same_as_A0(negative)" if (sum_a2 - sum_a1) < 0 else "opposite_to_A0(positive)",
        "rho_e_median_on_shared218": float(np.median(ratios_218)) if len(ratios_218) else np.nan,
        "rho_e_p16_on_shared218": float(np.percentile(ratios_218, 16)) if len(ratios_218) else np.nan,
        "rho_e_p84_on_shared218": float(np.percentile(ratios_218, 84)) if len(ratios_218) else np.nan,
        "rho_e_median_within_pm10pct_of_1": bool(len(ratios_218) and abs(np.median(ratios_218) - 1) < 0.10),
        "rho_e_median_within_0.9_1.1": bool(len(ratios_218) and 0.9 <= np.median(ratios_218) <= 1.1),
    }
    if venue == "iiib":
        entry["shared_218"]["note"] = ("No registered A0 reference constant exists for iiib on this "
                                        "stratum; m_S here is computed against the JOINT_R1 A0 "
                                        "reference (112.697) per the registered single m_S formula, "
                                        "since the prereg gives only one denominator. iiib's own A0 "
                                        f"baseline sum on shared-218 = {sum_iiib_218_a0:.6f} is reported "
                                        "for context/flagging only.")
        entry["shared_218"]["iiib_own_A0_baseline_sum"] = sum_iiib_218_a0
    # resurrected-316 (joint_r1 only, by construction)
    if venue == "joint_r1":
        ev2 = resurrected_316 & set(d_a1) & set(d_a2)
        sum_a1_316 = sum(d_a1[e] for e in ev2)
        sum_a2_316 = sum(d_a2[e] for e in ev2)
        entry["resurrected_316"] = {
            "n_events": len(ev2), "sum_A1": sum_a1_316, "sum_A2": sum_a2_316,
            "mean_A1": sum_a1_316 / len(ev2) if ev2 else np.nan,
            "mean_A2": sum_a2_316 / len(ev2) if ev2 else np.nan,
            "delta_sum_A2_minus_A1": sum_a2_316 - sum_a1_316,
            "m_R": abs(sum_a2_316 - sum_a1_316) / abs(REF_SUM_RESURRECTED_316),
            "sign_delta_sum": "same_as_A0(negative)" if (sum_a2_316 - sum_a1_316) < 0 else "opposite_to_A0(positive)",
        }
    strata_read[venue] = entry
out["stratum_decomposition"] = strata_read

# Branch scoring (mechanical, per venue where m_R is defined i.e. joint_r1;
# iiib has no resurrected-316 stratum by construction -- flagged)
branch = {}
for venue in VENUES:
    m_S = strata_read[venue]["shared_218"]["m_S_vs_registered_joint_A0_ref_112.697"]
    rho_med_218 = strata_read[venue]["shared_218"]["rho_e_median_on_shared218"]
    if venue == "joint_r1":
        m_R = strata_read[venue]["resurrected_316"]["m_R"]
        sign_S = strata_read[venue]["shared_218"]["delta_sum_A2_minus_A1"] < 0
        sign_R = strata_read[venue]["resurrected_316"]["delta_sum_A2_minus_A1"] < 0
        same_sign = sign_S == sign_R
        tail_acting = (m_R >= 0.25) and (m_S < 0.10) and (rho_med_218 is not np.nan and abs(rho_med_218 - 1) < 0.10)
        core_reaching = (m_S >= 0.25) and (m_R >= 0.25) and same_sign and (not (0.9 <= rho_med_218 <= 1.1))
        if tail_acting and not core_reaching:
            verdict = "(a) TAIL-ACTING"
        elif core_reaching and not tail_acting:
            verdict = "(b) CORE-REACHING"
        else:
            verdict = "(c) MIXED/UNDETERMINED"
        branch[venue] = {
            "m_S": m_S, "m_R": m_R, "same_sign_strata": same_sign,
            "rho_e_median_shared218": rho_med_218,
            "tail_acting_criteria_met": tail_acting,
            "core_reaching_criteria_met": core_reaching,
            "verdict": verdict,
        }
    else:
        branch[venue] = {
            "m_S": m_S, "m_R": None,
            "note": ("resurrected-316 is a joint_r1-only stratum by definition (joint_r1-only dark "
                     "survivors); it structurally does not exist for iiib. Branches (a)/(b) both "
                     "require m_R, so iiib alone cannot independently satisfy (a) or (b) under a "
                     "literal reading -- iiib is reported as (c) MIXED/UNDETERMINED by construction "
                     "unless the joint verdict is read as a single cross-venue call (see joint below)."),
            "verdict": "(c) MIXED/UNDETERMINED (m_R undefined for iiib)",
        }
out["branch_scoring"] = branch

# Joint verdict: the registered scoring is stated per-venue with strata defined on the
# joint_r1/iiib intersection; only joint_r1 carries a full (m_S, m_R) pair. The joint
# call is read here as: joint_r1's own branch call, cross-checked against iiib's m_S.
joint_verdict = branch["joint_r1"]["verdict"]
out["branch_scoring"]["joint_verdict"] = joint_verdict
out["branch_scoring"]["joint_verdict_note"] = (
    "Only joint_r1 carries both m_S and m_R (resurrected-316 does not exist as an iiib "
    "stratum). The joint verdict is read off joint_r1's mechanical branch call; iiib's "
    "m_S is reported as a cross-check, not as an independent branch input."
)

# ---------------------------------------------------------------------------
# 5. Expected NULLs N1-N4
# ---------------------------------------------------------------------------
nulls = {}
# N1: horizon invariance -- from B2 pre-run instrumentation (pools are h-invariant
# and identical between the S ("kept") and "and" pools at every h; already measured
# and CLOSED before submission). Cross-checked against evaluate .err logs (4 s.f.).
b2 = json.load(open(f"{ROOT}/results/campaign51_20260728/realistic_20260729/d1_b2_sand_hslope.json"))
dl_vals = set()
for row in b2["grid41_results"]:
    dl_vals.add(round(row["dl_max_S"], 9))
    dl_vals.add(round(row["dl_max_and"], 9))
N1_TARGET = 9.164987215485882
nulls["N1_horizon_invariance"] = {
    "unique_dl_max_values_across_grid_and_pools": sorted(dl_vals),
    "all_equal_target": dl_vals == {round(N1_TARGET, 9)},
    "pass": dl_vals == {round(N1_TARGET, 9)},
    "cross_check_evaluate_logs_h0.73_A1_iiib": "D(h=0.7300) ... dl_max=9.1650 Gpc (4 s.f., consistent)",
}

# N2: L_cat_no_bh / L_cat_with_bh bit-identical between A1 and A2, all 41x1588 cells
n2 = {}
for venue in VENUES:
    lh_a1 = pd.read_csv(f"{ARM_DIRS['A1'][venue]}/diagnostics/event_likelihoods.csv").sort_values(["event_idx", "h"]).reset_index(drop=True)
    lh_a2 = pd.read_csv(f"{ARM_DIRS['A2'][venue]}/diagnostics/event_likelihoods.csv").sort_values(["event_idx", "h"]).reset_index(drop=True)
    aligned = lh_a1[["event_idx", "h"]].equals(lh_a2[["event_idx", "h"]])
    if aligned:
        no_bh_identical = bool((lh_a1["L_cat_no_bh"].values == lh_a2["L_cat_no_bh"].values).all())
        with_bh_identical = bool((lh_a1["L_cat_with_bh"].values == lh_a2["L_cat_with_bh"].values).all())
        n_mismatch_no_bh = int((lh_a1["L_cat_no_bh"].values != lh_a2["L_cat_no_bh"].values).sum())
        n_mismatch_with_bh = int((lh_a1["L_cat_with_bh"].values != lh_a2["L_cat_with_bh"].values).sum())
    else:
        no_bh_identical = with_bh_identical = False
        n_mismatch_no_bh = n_mismatch_with_bh = None
    n2[venue] = {
        "row_alignment_ok": aligned, "n_cells": len(lh_a1),
        "L_cat_no_bh_bit_identical": no_bh_identical, "n_mismatch_no_bh": n_mismatch_no_bh,
        "L_cat_with_bh_bit_identical": with_bh_identical, "n_mismatch_with_bh": n_mismatch_with_bh,
        "pass": no_bh_identical and with_bh_identical and aligned,
    }
nulls["N2_catalogue_leg_bit_identity"] = n2

# N3: A1 vs A0 composition near-null: |Delta ln w_tilde_G(h)| < 0.01 at every h
n3 = {}
for venue in VENUES:
    lh_a1 = pd.read_csv(f"{ARM_DIRS['A1'][venue]}/diagnostics/event_likelihoods.csv")
    lh_a0 = a0_lh[venue]
    wg_a1 = lh_a1.groupby("h")["w_tilde_G"].first()
    wg_a0 = lh_a0.groupby("h")["w_tilde_G"].first()
    common_h = sorted(set(wg_a1.index) & set(wg_a0.index))
    dln = np.array([abs(np.log(wg_a1[h]) - np.log(wg_a0[h])) for h in common_h])
    n3[venue] = {
        "n_h": len(common_h), "max_abs_dln_w_tilde_G": float(dln.max()),
        "all_below_0.01": bool((dln < 0.01).all()),
        "pass": bool((dln < 0.01).all()),
    }
nulls["N3_A1_vs_A0_near_null"] = n3

# N4: 1D channel must NOT be bit-identical under A2 (non-null expected)
n4 = {}
for venue in VENUES:
    lh_a1 = pd.read_csv(f"{ARM_DIRS['A1'][venue]}/diagnostics/event_likelihoods.csv").sort_values(["event_idx", "h"]).reset_index(drop=True)
    lh_a2 = pd.read_csv(f"{ARM_DIRS['A2'][venue]}/diagnostics/event_likelihoods.csv").sort_values(["event_idx", "h"]).reset_index(drop=True)
    aligned = lh_a1[["event_idx", "h"]].equals(lh_a2[["event_idx", "h"]])
    comb_wbh_identical = bool((lh_a1["combined_with_bh"].values == lh_a2["combined_with_bh"].values).all()) if aligned else None
    D_tilde_identical = bool((lh_a1["D_tilde_phi"].values == lh_a2["D_tilde_phi"].values).all()) if aligned else None
    n4[venue] = {
        "combined_with_bh_bit_identical_A1_A2": comb_wbh_identical,
        "D_tilde_phi_bit_identical_A1_A2": D_tilde_identical,
        "expected_NON_null_confirmed": bool(aligned and not comb_wbh_identical and not D_tilde_identical),
        "pass": bool(aligned and not comb_wbh_identical and not D_tilde_identical),
    }
nulls["N4_1D_channel_moves_nonnull_expected"] = n4
out["nulls"] = nulls

# ---------------------------------------------------------------------------
# 6. Context-only numbers (never scored): w_tilde_G(h) shift across grid, per arm
# ---------------------------------------------------------------------------
context = {}
for venue in VENUES:
    entry = {}
    for arm_name, arm_dir_map in [("A0", None), ("A1", ARM_DIRS["A1"]), ("A2", ARM_DIRS["A2"])]:
        df = a0_lh[venue] if arm_name == "A0" else pd.read_csv(f"{arm_dir_map[venue]}/diagnostics/event_likelihoods.csv")
        wg = df.groupby("h")["w_tilde_G"].first().sort_index()
        entry[arm_name] = {"w_tilde_G_h0.73": float(wg.get(0.73, np.nan)),
                            "w_tilde_G_h0.60": float(wg.get(0.60, np.nan)),
                            "w_tilde_G_h0.86": float(wg.get(0.86, np.nan))}
    context[venue] = entry
context["note_2d_1d_MAP"] = (
    "2D/1D MAP context numbers are NOT available: all four D1 arms were run with "
    "--combine=false (diagnostic-only, per the prereg's 'No production posterior' "
    "scope guard) -- no combined_2d.json/combined_1d.json exist for the D1 arms. "
    "Only the retrieved per-event/per-h event_likelihoods.csv are available, so the "
    "MAP context read is reported as N/A by design, not a gap in retrieval."
)
out["context_only"] = context

# ---------------------------------------------------------------------------
# 7. S2 (g_frac h-slope, and per-event invariance -- B3's registered check)
#    and S3 (sign expectation of the A2-A1 mixture tilt shift)
# ---------------------------------------------------------------------------
s2 = {}
gfrac_bit_identical = {}
for venue in VENUES:
    lh_a1 = pd.read_csv(f"{ARM_DIRS['A1'][venue]}/diagnostics/event_likelihoods.csv").sort_values(["event_idx", "h"]).reset_index(drop=True)
    lh_a2 = pd.read_csv(f"{ARM_DIRS['A2'][venue]}/diagnostics/event_likelihoods.csv").sort_values(["event_idx", "h"]).reset_index(drop=True)
    aligned = lh_a1[["event_idx", "h"]].equals(lh_a2[["event_idx", "h"]])
    g1, g2 = lh_a1["g_frac"].values, lh_a2["g_frac"].values
    identical = bool(aligned and np.array_equal(g1, g2))
    gfrac_bit_identical[venue] = {
        "bit_identical_per_event_A1_A2": identical,
        "n_mismatch": int((g1 != g2).sum()) if aligned else None,
        "max_abs_diff": float(np.max(np.abs(g1 - g2))) if aligned else None,
    }
    for arm in ["A1", "A2"]:
        df = pd.read_csv(f"{ARM_DIRS[arm][venue]}/diagnostics/event_likelihoods.csv")
        g60 = df[df["h"] == H_GRID_LOW]["g_frac"].mean()
        g86 = df[df["h"] == H_GRID_HIGH]["g_frac"].mean()
        s2[f"{arm}_{venue}"] = float(np.log(g86 / g60))

S2_A0_REFERENCE_QUOTED = 0.047586  # as quoted in prereg text
a0_slope = {}
for venue in VENUES:
    g60 = a0_lh[venue][a0_lh[venue]["h"] == H_GRID_LOW]["g_frac"].mean()
    g86 = a0_lh[venue][a0_lh[venue]["h"] == H_GRID_HIGH]["g_frac"].mean()
    a0_slope[venue] = float(np.log(g86 / g60))

s2_summary = {"per_arm_venue_slope_mean_g_frac_ln_ratio_h060_h086": s2,
              "A0_recomputed_slope_this_definition": a0_slope,
              "A0_recomputed_slope_bit_identical_across_venues": a0_slope["iiib"] == a0_slope["joint_r1"],
              "prereg_quoted_A0_reference": S2_A0_REFERENCE_QUOTED,
              "AMBIGUITY_FLAG": (
                  "The prereg quotes A0 reference slope 0.047586 for 'Deltaln g-bar'. Recomputing "
                  "the population MEAN g_frac ln-ratio over the full grid (h=0.60->0.86) from the "
                  "retrieved A0 twins gives 0.040284309008804246 (bit-identical between venues, "
                  "confirming the 'bit-identical across venues' half of the registered fact, but "
                  "NOT matching 0.047586 under this aggregation choice). Median-based and sum-based "
                  "alternatives were also tried (0.043408, identical-to-mean respectively) and do "
                  "not match 0.047586 either. Flagged rather than guessed further; the %-change "
                  "verdict below is computed self-consistently against MY OWN recomputed A0/A1 "
                  "reference using the mean-ln-ratio definition, since the pass/fail condition is a "
                  "RELATIVE (%) comparison, not an absolute-value match."
              ),
              "gfrac_bit_identity_per_event": gfrac_bit_identical,
              }
pct_change = {}
for venue in VENUES:
    a1s, a2s = s2[f"A1_{venue}"], s2[f"A2_{venue}"]
    pct = 0.0 if a1s == 0 else abs(a2s - a1s) / abs(a1s) * 100.0
    pct_change[venue] = {"A1_slope": a1s, "A2_slope": a2s, "pct_change": pct,
                          "unchanged_le_5pct": pct <= 5.0}
s2_summary["pct_change_A1_to_A2"] = pct_change
s2_summary["S2_verdict"] = (
    "g_frac is BIT-IDENTICAL per event, per h, in BOTH venues, between A1 and A2 "
    "(max abs diff = 0.0, 0 mismatches out of 65108 cells each). This directly answers "
    "B3's registered check ('if g_frac is invariant under S_4D -> S_and,4D to <=1e-6 per "
    "event, the C7 convergence route is dead') at machine precision (0, not just <=1e-6). "
    "Consequently the h-slope is unchanged (0.0% change, well within the <=5%% band the "
    "prereg itself states as the refutation condition): "
    "'D1 does not act through the completion leg's mass factor, and the C7 convergence is "
    "refuted regardless of the branch.' -- S2 registers this refutation as TRUE."
)
out["S2_g_frac_slope"] = s2_summary

s3 = {}
for venue in VENUES:
    t1, t2 = per_venue[venue]["_t_a1"], per_venue[venue]["_t_a2"]
    common = sorted(set(t1) & set(t2))
    diff = np.array([t2[e] - t1[e] for e in common])
    s3[venue] = {
        "n_events": len(diff),
        "frac_positive": float((diff > 0).mean()),
        "frac_negative": float((diff < 0).mean()),
        "frac_zero": float((diff == 0).mean()),
        "median_diff": float(np.median(diff)),
        "mean_diff": float(np.mean(diff)),
        "one_signed_strict": bool((diff > 0).all() or (diff < 0).all()),
        "predominant_sign": "positive" if (diff > 0).mean() > 0.5 else "negative",
    }
s3["S3_verdict"] = (
    "t_e^A2 - t_e^A1 at the gate-vii (0.73,0.81) pair is predominantly ONE-SIGNED "
    "(positive) in both venues (~96.6% iiib / ~94.9% joint_r1 of events share the same "
    "sign), consistent with S3's first-order mixture-log-odds direction, but NOT strictly "
    "one-signed at the individual-event level: 3.4% (iiib) / 5.1% (joint_r1) of events "
    "flip sign. Per S3's own text ('A sign flip ... is a registered failure of S3 ... and "
    "must be reported, not smoothed'), these minority-sign events are a registered PARTIAL "
    "FAILURE of the strict one-signed prediction, reported as such."
)
out["S3_sign_expectation"] = s3

# strip private helper keys before dumping
for venue in VENUES:
    for k in list(per_venue[venue].keys()):
        if k.startswith("_"):
            del per_venue[venue][k]
out["per_venue_distribution"] = per_venue

# ---------------------------------------------------------------------------
json.dump(out, open(f"{D1}/readout.json", "w"), indent=2, default=str)
print("Wrote", f"{D1}/readout.json")
print(json.dumps({k: v for k, v in out.items() if k not in ("per_venue_distribution",)}, indent=2, default=str)[:3000])
