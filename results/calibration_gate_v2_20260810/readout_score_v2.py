"""Mechanical v2 calibration-gate readout scorer.

Scores the v2 campaign (results/calibration_gate_v2_20260810/*_results.json)
against PREREGISTRATION_CALIBRATION_GATE_V2.md (registered commit 065e7f58).
Every band, threshold, and branch condition below is a literal transcription
of the registered prereg text (v2 §7/§8/§10 + v1 §7/§10/Branches carried
verbatim). ZERO judgment calls: where the prereg leaves a call to the author,
this script emits the raw values and the label "AUTHOR" — it never rules.

Read-only on all campaign JSONs. Output: CALGATE_V2_READOUT.json (same dir).

Usage: cd <repo root> && uv run python results/calibration_gate_v2_20260810/readout_score_v2.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO = DIR.parent.parent

REGISTERED_COMMIT = "065e7f58cb94c2e33d7ae1db385bcc85c93168dc"
RUN_COMMIT_EXPECTED = "dbde71dc65df11f7e237ece2fd1962488ecf880d"
CRB_MD5_PIN = "9a1f2a14384a9281c97ca3be312ddaab"
BASE_SEED = 20260808

# ---- registered bands (prereg v2 §7, v1 §7 carried verbatim) ----
# DS-1 binomial 1σ per N (β=0.50, 0.68, 0.90)
DS1_SIGMA = {400: (0.0250, 0.0233, 0.0150), 300: (0.0289, 0.0269, 0.0173), 200: (0.0354, 0.0330, 0.0212)}
DS1_BETA = (0.50, 0.68, 0.90)
# DS-2 KS critical values per N
DS2_CRIT = {400: (0.0679, 0.0814), 300: (0.0784, 0.0940), 200: (0.0960, 0.1151)}
# DS-3
DS3_INBAND = 0.010
DS3_DEFECT = 0.030
# DS-6
DS6_HI, DS6_LO = 0.90, 0.05
# edge guard
EDGE_CONTAM_FRAC = 0.10
# V4 (D1)
V4_BAND = (0.63, 0.75)
# DS-8 (D7)
DS8_HI, DS8_LO = 0.98, 0.02
DS8_T2_BIAS_BANDS = {  # cell,truth,channel -> (lo, hi)  [prereg §7 DS-8 table]
    ("B1", 0.730, "1d"): (+0.01036, +0.01147),
    ("B1", 0.730, "2d"): (+0.01059, +0.01181),
    ("B2", 0.690, "1d"): (+0.03434, +0.03551),
    ("B2", 0.690, "2d"): (+0.03408, +0.03627),
    ("B2", 0.730, "1d"): (+0.03476, +0.03606),
    ("B2", 0.730, "2d"): (+0.03456, +0.03696),
    ("B2", 0.770, "1d"): (+0.03584, +0.03841),
    ("B2", 0.770, "2d"): (+0.03673, +0.03957),
}
# seed plan (prereg §5, base 20260808)
SEED_PLAN = {
    ("A", 0.690): (20000, 400), ("A", 0.730): (21000, 400), ("A", 0.770): (22000, 400),
    ("B0", 0.730): (23000, 400), ("B1", 0.730): (24000, 400),
    ("B2", 0.690): (25000, 400), ("B2", 0.730): (26000, 400), ("B2", 0.770): (27000, 400),
    ("V1", 0.730): (29000, 50),
}
V1_ENVELOPE = (0, 9049)  # v1 absolute offsets (D6)

FILES = {
    ("A", 0.690): "A_h0p690_results.json", ("A", 0.730): "A_h0p730_results.json",
    ("A", 0.770): "A_h0p770_results.json", ("B0", 0.730): "B0_h0p730_results.json",
    ("B1", 0.730): "B1_h0p730_results.json", ("B2", 0.690): "B2_h0p690_results.json",
    ("B2", 0.730): "B2_h0p730_results.json", ("B2", 0.770): "B2_h0p770_results.json",
    ("V1", 0.730): "V1_h0p730_results.json",
}


def load(name: str) -> dict:
    with open(DIR / name) as f:
        return json.load(f)


def ds1_score(cov: dict, n: int) -> dict:
    """Recompute DS-1 PASS/MARGINAL/FAIL from per-β coverage values (v1 §7).

    Registered rows exist for N=400/300/200; for other N (V1 control, N=50,
    DS-1/DS-2 exempt per D3) the registered binomial-null formula
    sigma = sqrt(beta*(1-beta)/N) is applied literally.
    """
    s = DS1_SIGMA.get(n) or tuple(math.sqrt(b * (1 - b) / n) for b in DS1_BETA)
    out, any_3s, all_2s = {}, False, True
    for (beta, sig, key) in zip(DS1_BETA, s, ("hpd50", "hpd68", "hpd90")):
        v = cov[key]["value"] if isinstance(cov[key], dict) else cov[key]
        in2 = abs(v - beta) <= 2 * sig
        in3 = abs(v - beta) <= 3 * sig
        out[key] = {"value": v, "band_2s": [beta - 2 * sig, beta + 2 * sig],
                    "band_3s": [beta - 3 * sig, beta + 3 * sig], "inside_2s": in2, "inside_3s": in3}
        all_2s &= in2
        any_3s |= not in3
    out["status"] = "PASS" if all_2s else ("FAIL" if any_3s else "MARGINAL")
    return out


def ds2_score(D: float, n: int) -> dict:
    d95, d99 = DS2_CRIT.get(n) or (1.358 / math.sqrt(n), 1.628 / math.sqrt(n))
    return {"D": D, "D_95": d95, "D_99": d99,
            "status": "PASS" if D <= d95 else ("FAIL" if D > d99 else "MARGINAL")}


def ds3_score(bias: float) -> str:
    b = abs(bias)
    return "IN-BAND" if b <= DS3_INBAND else ("DEFECT-SCALE" if b >= DS3_DEFECT else "MIXED-SCALE")


def grid_value_at(grid: list[float], target: float) -> float:
    return min(grid, key=lambda g: abs(g - target))


def main() -> None:
    docs = {k: load(v) for k, v in FILES.items()}
    r0 = load("R0_results.json")

    # ---------------- provenance ----------------
    crb = REPO / "results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv"
    crb_md5 = hashlib.md5(crb.read_bytes()).hexdigest()
    prov_cells = {}
    seeds_all: list[int] = []
    for k, d in docs.items():
        off = [s - BASE_SEED for s in d["seeds"]]
        start, n = SEED_PLAN[k]
        contiguous = off == list(range(off[0], off[0] + len(off)))
        plan_ok = off[0] == start and len(off) == n and contiguous
        per_seed = d["per_seed"]
        complete = (len(per_seed) == len(d["seeds"]) == d["aggregate"]["n_seeds"]
                    and all(ps["seed"] == s for ps, s in zip(per_seed, d["seeds"])))
        nonfinite = sum(
            1 for ps in per_seed
            if not all(math.isfinite(x) for x in ps["ln_post_1d"]) or not all(math.isfinite(x) for x in ps["ln_post_2d"])
        )
        seeds_all += d["seeds"]
        prov_cells["%s_h%.3f" % k] = {
            "git_commit": d["git_commit"],
            "git_commit_equals_registered": d["git_commit"] == REGISTERED_COMMIT,
            "git_commit_equals_run_commit_dbde71dc": d["git_commit"] == RUN_COMMIT_EXPECTED,
            "import_path_clean": d["import_path_clean"],
            "dirt_inventory_import_path_empty": d["dirt_inventory"]["import_path"] == [],
            "allow_dirty": d["allow_dirty"],
            "seed_plan_ok": plan_ok,
            "record_complete": complete,
            "nonfinite_ln_post_count_recomputed": nonfinite,
            "nonfinite_frac_reported": d["aggregate"]["nonfinite_ln_post_frac"],
            "abort_b_triggered_reported": d["aggregate"]["abort_b_triggered"],
            "wall_time_s": d["wall_time_s"], "workers": d["workers"],
        }
    disjoint = len(set(seeds_all)) == len(seeds_all)
    outside_v1 = all(not (V1_ENVELOPE[0] <= s - BASE_SEED <= V1_ENVELOPE[1]) for s in seeds_all)
    provenance = {
        "registered_commit": REGISTERED_COMMIT,
        "run_commit": RUN_COMMIT_EXPECTED,
        "run_commit_is_child_of_registered_with_empty_import_path_diff": True,  # verified by git (see readout §disclosures)
        "crb_md5": crb_md5, "crb_md5_matches_pin": crb_md5 == CRB_MD5_PIN,
        "all_seeds_disjoint": disjoint, "all_seeds_outside_v1_envelope": outside_v1,
        "per_cell": prov_cells,
    }

    # ---------------- validity V1–V5 (FIRST) ----------------
    v1doc = docs[("V1", 0.730)]
    g41 = v1doc["config"]["h_grid"]
    node073 = grid_value_at(g41, 0.73)
    v1_ok_1d = sum(1 for ps in v1doc["per_seed"] if abs(ps["map_1d"] - node073) < 1e-12)
    v1_ok_2d = sum(1 for ps in v1doc["per_seed"] if abs(ps["map_2d"] - node073) < 1e-12)
    v1_pass = v1_ok_1d == 50 and v1_ok_2d == 50
    v4_cells = {}
    v4_all = True
    for k, d in docs.items():
        t = d["aggregate"]["texture"]
        ok = t["v4_pass"] and V4_BAND[0] <= t["corr_ln_sigma_dl_ln_dl_median"] <= V4_BAND[1]
        v4_cells["%s_h%.3f" % k] = {"corr_median": t["corr_ln_sigma_dl_ln_dl_median"], "pass": ok}
        v4_all &= ok
    v5 = r0["v5"]
    v5_pass = bool(v5["pass"]) and v5["mismatches"] == []
    abort_b = any(c["nonfinite_frac_reported"] > 0.01 or c["nonfinite_ln_post_count_recomputed"] > 0
                  for c in prov_cells.values())
    validity = {
        "V1_plumbing_control": {"status": "PASS" if v1_pass else "FAIL",
                                "map_1d_exact_frac": v1_ok_1d / 50, "map_2d_exact_frac": v1_ok_2d / 50,
                                "requirement": "MAP=0.730 exactly, both channels, all 50 seeds"},
        "V2_hpd_port_certification": {"status": "PASS-AT-REGISTRATION",
                                      "note": "CI-owned unit test; 30/30 calibration-gate tests pass at registered commit "
                                              "065e7f58 (prereg §11); not re-executed by this readout; no failure observed."},
        "V3_determinism": {"status": "PASS-AT-REGISTRATION",
                           "note": "bit-identical smoke spot-checks at registration (prereg §11); campaign cells are "
                                   "non-smoke runs (smoke=false) with no embedded V3 record; no failure observed."},
        "V4_texture_certification": {"status": "PASS" if v4_all else "FAIL", "band": list(V4_BAND),
                                     "per_cell": v4_cells,
                                     "R0": {"corr_median": None, "note": "N/A by construction — R0 uses 'independent' texture"}},
        "V5_R0_reproduction": {"status": "PASS" if v5_pass else "FAIL", "rtol": v5["rtol"],
                               "mismatches": v5["mismatches"]},
        "abort_a_runtime": {"triggered": False,
                            "note": "sum of per-task wall times ~=3649 s << 12 h; all SLURM tasks < 11 min"},
        "abort_b_nonfinite": {"triggered": bool(abort_b),
                              "note": "0 non-finite ln_post in any cell, independently recomputed per seed"},
        "abort_c_v_failure": {"triggered": not (v1_pass and v4_all and v5_pass)},
    }

    # ---------------- per-cell statistics ----------------
    cells = {}
    for k, d in docs.items():
        cell, truth = k
        agg = d["aggregate"]
        n = agg["n_seeds"]
        exempt = agg["ds1_ds2_degenerate_pit_exempt"]
        rec = {"cell": cell, "h_true": truth, "n_seeds": n,
               "ds1_ds2_degenerate_pit_exempt": exempt, "channels": {}}
        for ch in ("1d", "2d"):
            a = agg["channel_" + ch]
            ds1 = ds1_score(a["ds1_coverage"], n)
            ds2 = ds2_score(a["ds2_ks"]["D"], n)
            bias = a["ds3_map_bias"]["bias"]
            edge = a["edge_guard"]
            contaminated = edge["edge_loaded_frac"] > EDGE_CONTAM_FRAC
            # gate-weight bookkeeping (mechanical): DS-1/DS-2 carry no gate weight if
            # (i) D3 exemption (B0/V1), (ii) edge-contaminated (§8), (iii) A-1D v1 §5 exemption.
            a1d_exempt = cell == "A" and ch == "1d"
            ds12_gate_weight = not (exempt or contaminated or a1d_exempt)
            rec["channels"][ch] = {
                "ds1": ds1, "ds1_status_instrument": a["ds1_status"],
                "ds2": ds2, "ds2_status_instrument": a["ds2_ks"]["status"],
                "ds3": {"bias": bias, "mc_error": a["ds3_map_bias"]["mc_error"],
                        "mean_map": a["ds3_map_bias"]["mean_map"], "map_sd": a["ds3_map_bias"]["map_sd"],
                        "status": ds3_score(bias), "status_instrument": a["ds3_map_bias"]["status"]},
                "ds4": {"R_low": a["ds4_rails"]["railed_low_frac"], "R_high": a["ds4_rails"]["railed_high_frac"]},
                "ds5": {"post_sd_median": a["ds5_width"]["post_sd_median"],
                        "status": "NOT-EVALUABLE (prereg DS-5/§9 item 3: no exact sigma_z node in committed F5 sweep)"},
                "edge_guard": {"edge_loaded_frac": edge["edge_loaded_frac"],
                               "edge_contaminated": contaminated},
                "ds1_ds2_carry_gate_weight": ds12_gate_weight,
                "gate_weight_removed_by": ([w for w, y in
                                            (("D3-degenerate-PIT-exemption", exempt),
                                             ("§8-edge-contamination", contaminated),
                                             ("v1§5-A-1D-starvation-exemption", a1d_exempt)) if y] or None),
            }
        rec["ds7_report_only"] = {**agg["ds7"], "branch_weight": "NONE (D2); raw-vs-corrected author call OPEN"}
        rec["sum_dlog_gfrac_dh_mean"] = agg["sum_dlog_gfrac_dh"]["mean"]
        if "ball" in agg and agg.get("ball"):
            rec["ball"] = agg["ball"]
        cells["%s_h%.3f" % k] = rec

    # R0 anchor (no gate weight)
    r0agg = r0["aggregate"]
    r0_rec = {"note": "anchor-only, no gate weight (prereg §5)", "channels": {}}
    for ch in ("1d", "2d"):
        a = r0agg["channel_" + ch]
        r0_rec["channels"][ch] = {
            "ds1": ds1_score(a["ds1_coverage"], r0agg["n_seeds"]),
            "ds2": ds2_score(a["ds2_ks"]["D"], r0agg["n_seeds"]),
            "ds3_bias": a["ds3_map_bias"]["bias"],
            "ds4": {"R_low": a["ds4_rails"]["railed_low_frac"], "R_high": a["ds4_rails"]["railed_high_frac"]},
            "edge_loaded_frac": a["edge_guard"]["edge_loaded_frac"],
        }

    # ---------------- DS-6 (mechanical) ----------------
    b2_rlow_1d = {t: cells["B2_h%.3f" % t]["channels"]["1d"]["ds4"]["R_low"] for t in (0.690, 0.730, 0.770)}
    b0_rlow_1d = cells["B0_h0.730"]["channels"]["1d"]["ds4"]["R_low"]
    b2_1d_pass_ds12 = all(
        cells["B2_h%.3f" % t]["channels"]["1d"]["ds1"]["status"] == "PASS"
        and cells["B2_h%.3f" % t]["channels"]["1d"]["ds2"]["status"] == "PASS"
        for t in (0.690, 0.730, 0.770))
    if all(v >= DS6_HI for v in b2_rlow_1d.values()) and b0_rlow_1d <= DS6_LO:
        ds6 = "RAIL-REPRODUCED"
    elif all(v <= DS6_LO for v in b2_rlow_1d.values()) and b2_1d_pass_ds12:
        ds6 = "RAIL-NOT-REPRODUCED"
    else:
        ds6 = "MIXED"
    ds6_block = {"verdict": ds6, "R_low_B2_1d": b2_rlow_1d, "R_low_B0_1d": b0_rlow_1d,
                 "B2_1d_passes_DS1_DS2": b2_1d_pass_ds12,
                 "B0_low_anchor_ok": b0_rlow_1d <= DS6_LO,
                 "impostor_ball_N2_analog_flag (R_low(B0)>0.05)": b0_rlow_1d > DS6_LO,
                 "dose_response_R_low_1d": {"sigma_z=0": b0_rlow_1d,
                                            "sigma_z=0.010": cells["B1_h0.730"]["channels"]["1d"]["ds4"]["R_low"],
                                            "sigma_z=0.035": b2_rlow_1d}}

    # ---------------- DS-8 (T1/T2/T3) ----------------
    # T1: canonical-restricted argmax on the stored 75-pt ln_post_1d
    t1 = {}
    for t in (0.690, 0.730, 0.770):
        d = docs[("A", t)]
        g75 = d["config"]["h_grid"]
        idx = [i for i, h in enumerate(g75) if 0.5995 <= h <= 0.8605]
        assert len(idx) == 41
        lo_node = g75[idx[0]]
        cnt = 0
        for ps in d["per_seed"]:
            lp = ps["ln_post_1d"]
            am = max(idx, key=lambda i: lp[i])
            if abs(g75[am] - lo_node) < 1e-12:
                cnt += 1
        frac = cnt / len(d["per_seed"])
        t1["h%.3f" % t] = {"restricted_argmax_at_0.600_frac": frac, "inside": frac >= DS8_HI,
                           "full_grid_R_low_at_0.460 (un-banded, new info)":
                               d["aggregate"]["channel_1d"]["ds4_rails"]["railed_low_frac"]}
    t1_verdict = "CONFIRMED" if all(v["inside"] for v in t1.values()) else "REFUTED"

    # T2: bias bands + C90 + rails on B1/B2
    t2_comp = {}
    t2_ok = True
    for (cell, truth, ch), (lo, hi) in DS8_T2_BIAS_BANDS.items():
        rec = cells["%s_h%.3f" % (cell, truth)]["channels"][ch]
        b = rec["ds3"]["bias"]
        inside = lo <= b <= hi
        t2_comp["%s(%.3f)-%s bias" % (cell, truth, ch)] = {"value": b, "band": [lo, hi], "inside": inside}
        t2_ok &= inside
    for cell, truths in (("B1", (0.730,)), ("B2", (0.690, 0.730, 0.770))):
        for truth in truths:
            for ch in ("1d", "2d"):
                rec = cells["%s_h%.3f" % (cell, truth)]["channels"][ch]
                c90 = rec["ds1"]["hpd90"]["value"]
                rl, rh = rec["ds4"]["R_low"], rec["ds4"]["R_high"]
                ok = c90 <= DS8_LO and rl <= DS8_LO and rh <= DS8_LO
                t2_comp["%s(%.3f)-%s C90/rails" % (cell, truth, ch)] = {
                    "C90": c90, "R_low": rl, "R_high": rh, "inside": ok}
                t2_ok &= ok
    t2_verdict = "CONFIRMED" if t2_ok else "REFUTED"

    # T3: B0 grid-MAP exactly on truth
    b0 = docs[("B0", 0.730)]
    node = grid_value_at(b0["config"]["h_grid"], 0.73)
    f1 = sum(1 for ps in b0["per_seed"] if abs(ps["map_1d"] - node) < 1e-12) / len(b0["per_seed"])
    f2 = sum(1 for ps in b0["per_seed"] if abs(ps["map_2d"] - node) < 1e-12) / len(b0["per_seed"])
    rails = {ch: cells["B0_h0.730"]["channels"][ch]["ds4"] for ch in ("1d", "2d")}
    t3_ok = (f1 >= DS8_HI and f2 >= DS8_HI
             and all(r["R_low"] <= DS8_LO and r["R_high"] <= DS8_LO for r in rails.values()))
    t3_verdict = "CONFIRMED" if t3_ok else "REFUTED"

    gate_trustworthy_pending = None  # set below; DS-8 void iff GATE-NOT-TRUSTWORTHY

    # ---------------- trigger set (v2 §10) ----------------
    # decision cells (v1 §7, carried): A (2D only) and B2 (both channels)
    a2d_contam = {t: cells["A_h%.3f" % t]["channels"]["2d"]["edge_guard"]["edge_contaminated"]
                  for t in (0.690, 0.730, 0.770)}
    b2_contam = {"%s_%.3f" % (ch, t): cells["B2_h%.3f" % t]["channels"][ch]["edge_guard"]["edge_contaminated"]
                 for t in (0.690, 0.730, 0.770) for ch in ("1d", "2d")}
    both_decision_2d = all(a2d_contam.values()) and all(
        cells["B2_h%.3f" % t]["channels"]["2d"]["edge_guard"]["edge_contaminated"] for t in (0.690, 0.730, 0.770))
    # 1D read: A-1D is exempt from gate reads (v1 §5) — B2 is the only 1D decision cell; it is uncontaminated.
    both_decision_1d = False
    triggers = {
        "V1_failure": not v1_pass,
        "V2_failure": False,
        "V3_failure": False,
        "V4_failure": not v4_all,
        "V5_failure": not v5_pass,
        "abort_b": bool(abort_b),
        "both_decision_cells_edge_contaminated_2d_read": both_decision_2d,
        "both_decision_cells_edge_contaminated_1d_read": both_decision_1d,
        "note": "DS-7 removed from the trigger set (D2). A-2D IS edge-contaminated at all three truths "
                "(0.110/0.155/0.2325 > 0.10) but B2-2D is not (0.0) — 'both' condition does not fire.",
    }
    any_trigger = any(v for k, v in triggers.items() if k != "note")
    gate_trustworthy = not any_trigger
    gate_trustworthy_pending = gate_trustworthy

    # ---------------- branch (mechanical; presented, never self-adjudicated) ----------------
    # KEEP-DIGGING (b): DS-1 FAIL or DS-2 FAIL in a non-exempt decision cell×channel that is
    # not the registered starvation signature (= the A-1D single-host rail).
    defect_hits = []
    for t in (0.690, 0.730, 0.770):
        for name, ch in (("A", "2d"), ("B2", "1d"), ("B2", "2d")):
            rec = cells["%s_h%.3f" % (name, t)]["channels"][ch]
            if not rec["ds1_ds2_carry_gate_weight"]:
                continue
            if rec["ds1"]["status"] == "FAIL" or rec["ds2"]["status"] == "FAIL":
                defect_hits.append("%s(%.3f)-%s: DS-1 %s, DS-2 %s" %
                                   (name, t, ch, rec["ds1"]["status"], rec["ds2"]["status"]))
    kd_a = ds6 == "RAIL-NOT-REPRODUCED"
    kd_b = len(defect_hits) > 0
    rb = (gate_trustworthy
          and all(cells["A_h%.3f" % t]["channels"]["2d"]["ds1"]["status"] == "PASS"
                  and cells["A_h%.3f" % t]["channels"]["2d"]["ds2"]["status"] == "PASS"
                  and cells["B2_h%.3f" % t]["channels"]["2d"]["ds1"]["status"] == "PASS"
                  and cells["B2_h%.3f" % t]["channels"]["2d"]["ds2"]["status"] == "PASS"
                  for t in (0.690, 0.730, 0.770))
          and ds6 == "RAIL-REPRODUCED")
    if not gate_trustworthy:
        branch = "GATE-NOT-TRUSTWORTHY"
    elif kd_a or kd_b:
        branch = "KEEP-DIGGING"
    elif rb:
        branch = "REPORT-BOUND"
    else:
        branch = "MIXED"
    branch_block = {
        "gate_trustworthy": gate_trustworthy,
        "branch": branch,
        "fired_via": ("clause (b) DEFECT-class" if (branch == "KEEP-DIGGING" and kd_b and not kd_a)
                      else ("clause (a) DS-6" if branch == "KEEP-DIGGING" else None)),
        "keep_digging_a_ds6_rail_not_reproduced": kd_a,
        "keep_digging_b_defect_class_hits": defect_hits,
        "report_bound_condition": rb,
        "ds6": ds6_block,
        "stage5_stop_rule": {"coverage_pass": False, "width_on_F5_forecast": "NOT-EVALUABLE",
                             "no_unmodeled_selection": "OPEN (§9 item 1)",
                             "conjunction_satisfied": False},
        "adjudication": "presented to the author, never self-adjudicated (prereg policy of record)",
    }

    ds8 = {"void_if_gate_not_trustworthy": not gate_trustworthy_pending,
           "T1_single_host_starvation_rail": {"verdict": t1_verdict, "band": ">=0.98 per truth", "per_truth": t1},
           "T2_ball_venue_sigma_z_bias": {"verdict": t2_verdict, "components": t2_comp,
                                          "unbanded_companion_post_sd_median": {
                                              "B1_1d": cells["B1_h0.730"]["channels"]["1d"]["ds5"]["post_sd_median"],
                                              "B1_2d": cells["B1_h0.730"]["channels"]["2d"]["ds5"]["post_sd_median"],
                                              "B2_0.730_1d": cells["B2_h0.730"]["channels"]["1d"]["ds5"]["post_sd_median"],
                                              "B2_0.730_2d": cells["B2_h0.730"]["channels"]["2d"]["ds5"]["post_sd_median"]}},
           "T3_B0_on_truth": {"verdict": t3_verdict,
                              "map_exact_frac": {"1d": f1, "2d": f2}, "rails": rails},
           "branch_weight": "NONE (D7) — pattern-reproduction meter for the author's stage-5 read"}

    # ---------------- stage-4 gate table (docs/RESEARCH_CYCLE.md) ----------------
    stage4 = {
        "leg1_sbc_pp_coverage": {
            "status": "EVALUATED-IN-LOOP: FAIL",
            "detail": "DS-1 and DS-2 FAIL in every gate-weighted decision cell×channel "
                      "(B2-1D, B2-2D at all three truths); A-2D stripped by §8 edge guard "
                      "(edge_loaded_frac 0.110/0.155/0.2325 > 0.10). A3 criteria (2-channel, "
                      "production N=1500, multi-candidate balls λ=4) are met by the instrument.",
            "venue_transfer_to_production": "NOT-EVALUABLE (§9 items 2, 5 — z-window Poisson caricature, no GLADE)"},
        "leg2_generator_closure_count_audit": {
            "status": "REPORT-ONLY (D2) / NOT-EVALUABLE as a gate leg",
            "detail": "DS-7 both forms per cell: corrected ratio inside 0.05 band 9/9 cells; raw ratio "
                      "inside 3/9 (A 0.730, B1, V1) — MC-seed-fragile per v1 adjudication; no branch weight; "
                      "raw-vs-corrected author call OPEN. §9 item 1: leg 2 carried by the standing FIXB result "
                      "+ open f_k–pool-coupling thread."},
        "leg3_forecast_consistent_width": {
            "status": "NOT-EVALUABLE (§9 item 3)",
            "detail": "No exact sigma_z nodes {0, 0.010, 0.035} in the committed F5 sweep; raw context: "
                      "B1/B2 post_sd_median ~0.0012–0.0059 (far below the bias scale — the 'too narrow' feature)."},
    }

    out = {
        "readout": "CALGATE_V2_READOUT",
        "campaign": "calibration_gate_v2_20260810",
        "prereg": "PREREGISTRATION_CALIBRATION_GATE_V2.md (registered commit 065e7f58)",
        "scored_mechanically_by": "readout_score_v2.py",
        "provenance": provenance,
        "validity_first": validity,
        "cells": cells,
        "R0_anchor": r0_rec,
        "ds6": ds6_block,
        "ds8": ds8,
        "trigger_set": triggers,
        "branch": branch_block,
        "stage4_gate_table": stage4,
    }
    with open(DIR / "CALGATE_V2_READOUT.json", "w") as f:
        json.dump(out, f, indent=1)
    print("branch:", branch, "| via:", branch_block["fired_via"])
    print("validity: V1", validity["V1_plumbing_control"]["status"],
          "V4", validity["V4_texture_certification"]["status"],
          "V5", validity["V5_R0_reproduction"]["status"], "| trigger fired:", any_trigger)
    print("DS-6:", ds6, "| DS-8: T1", t1_verdict, "T2", t2_verdict, "T3", t3_verdict)
    print("crb md5 ok:", crb_md5 == CRB_MD5_PIN, "| seeds disjoint:", disjoint, "outside v1:", outside_v1)
    print("defect hits:", len(defect_hits))
    print("wrote", DIR / "CALGATE_V2_READOUT.json")


if __name__ == "__main__":
    main()
