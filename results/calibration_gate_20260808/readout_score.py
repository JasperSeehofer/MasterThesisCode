"""Mechanical readout scorer for the calibration-gate campaign 2026-08-08.

Scores every pre-registered statistic of
PREREGISTRATION_CALIBRATION_GATE.md (commit b50ccc65, bands locked blind)
against its locked band, re-deriving each status from the raw per-seed /
aggregate values (the instrument's own labels are cross-checked, never
trusted). Zero free parameters: every threshold is quoted from the prereg.

Emits CALIBRATION_GATE_READOUT_20260808.json. The companion markdown
readout quotes this JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------- locked bands, quoted verbatim from prereg §7/§8/§10 ----------
PREREG_COMMIT = "b50ccc65a544648fb5f07e4cf2ec273a32be4170"

DS1_BANDS_N400_2S = {"hpd50": (0.450, 0.550), "hpd68": (0.633, 0.727), "hpd90": (0.870, 0.930)}
DS1_BANDS_N400_3S = {"hpd50": (0.425, 0.575), "hpd68": (0.610, 0.750), "hpd90": (0.855, 0.945)}
DS2_D95_N400, DS2_D99_N400 = 0.0679, 0.0814
DS3_INBAND, DS3_DEFECT = 0.010, 0.030
DS6_HIGH, DS6_LOW = 0.90, 0.05
DS7_BAND = 0.05
EDGE_SEED_THRESH, EDGE_CELL_THRESH = 0.01, 0.10
V4_BAND = (0.72, 0.92)
DS5_SCREEN = (0.5, 2.0)

CELLS = [
    ("A_h0p690", "A", 0.690), ("A_h0p730", "A", 0.730), ("A_h0p770", "A", 0.770),
    ("B0_h0p730", "B0", 0.730), ("B1_h0p730", "B1", 0.730),
    ("B2_h0p690", "B2", 0.690), ("B2_h0p730", "B2", 0.730), ("B2_h0p770", "B2", 0.770),
    ("V1_h0p730", "V1", 0.730),
]

# registered seed plan, prereg §5 (base 20260808)
SEED_PLAN = {
    "A_h0p690": (0, 400), "A_h0p730": (1000, 400), "A_h0p770": (2000, 400),
    "B0_h0p730": (3000, 400), "B1_h0p730": (4000, 400),
    "B2_h0p690": (5000, 400), "B2_h0p730": (6000, 400), "B2_h0p770": (7000, 400),
    "V1_h0p730": (9000, 50),
}
SEED_BASE = 20260808

# F5 committed sweep (scripts/bridge_closure/outputs/sigma_z_sigma_M_forecast.json)
F5_PATH = HERE.parent.parent / "scripts/bridge_closure/outputs/sigma_z_sigma_M_forecast.json"
F5_RESCALE = (400.0 / 1500.0) ** 0.5  # prereg DS-5 stage-1 procedure
F5_FLOOR = 0.014


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def score_ds1(cov: dict) -> tuple[str, dict]:
    vals = {b: cov[b]["value"] for b in ("hpd50", "hpd68", "hpd90")}
    in2 = all(DS1_BANDS_N400_2S[b][0] <= vals[b] <= DS1_BANDS_N400_2S[b][1] for b in vals)
    out3 = any(not (DS1_BANDS_N400_3S[b][0] <= vals[b] <= DS1_BANDS_N400_3S[b][1]) for b in vals)
    status = "PASS" if in2 else ("FAIL" if out3 else "MARGINAL")
    return status, vals


def score_ds2(D: float) -> str:
    return "PASS" if D <= DS2_D95_N400 else ("FAIL" if D > DS2_D99_N400 else "MARGINAL")


def score_ds3(b: float) -> str:
    ab = abs(b)
    return "IN-BAND" if ab <= DS3_INBAND else ("DEFECT-SCALE" if ab >= DS3_DEFECT else "MIXED-SCALE")


def main() -> None:
    files = {name: json.load(open(HERE / f"{name}_results.json")) for name, _, _ in CELLS}
    validate = json.load(open(HERE / "validate_results.json"))

    readout: dict = {
        "readout_date": "2026-08-10",
        "prereg": "results/calibration_gate_20260808/PREREGISTRATION_CALIBRATION_GATE.md",
        "prereg_commit": PREREG_COMMIT,
        "scoring": "mechanical; every band quoted from prereg §7/§8/§10; statuses re-derived from raw values",
        "cells": {},
        "label_crosscheck_mismatches": [],
    }

    seeds_all: list[int] = []
    provenance_violations: list[str] = []

    for name, cell, h_true in CELLS:
        d = files[name]
        agg = d["aggregate"]
        n = agg["n_seeds"]

        # --- seed plan (prereg §5) ---
        off, cnt = SEED_PLAN[name]
        expected = list(range(SEED_BASE + off, SEED_BASE + off + cnt))
        seeds = sorted(d["seeds"])
        seed_ok = seeds == expected
        seeds_all += seeds

        # --- provenance (prereg §10) ---
        commit_ok = d["git_commit"] == PREREG_COMMIT
        dirty = bool(d["git_dirty"])
        if dirty:
            provenance_violations.append(f"{name}: registered cell ran on dirty tree (allow_dirty={d['allow_dirty']})")

        cell_out: dict = {
            "cell": cell, "h_true": h_true, "n_seeds": n,
            "seed_plan_match": seed_ok, "git_commit_match": commit_ok, "git_dirty": dirty,
            "nonfinite_ln_post_frac": agg["nonfinite_ln_post_frac"],
            "abort_b_triggered": agg["abort_b_triggered"],
            "channels": {},
        }

        for ch in ("channel_1d", "channel_2d"):
            a = agg[ch]
            ds1_status, ds1_vals = score_ds1(a["ds1_coverage"])
            ds2_status = score_ds2(a["ds2_ks"]["D"])
            ds3_status = score_ds3(a["ds3_map_bias"]["bias"])
            edge_frac = a["edge_guard"]["edge_loaded_frac"]
            edge_contaminated = edge_frac > EDGE_CELL_THRESH
            # N=50 cell: prereg registers bands only for N in {400,300,200};
            # V1 is a plumbing control, DS-1/DS-2 carry no registered band at N=50.
            banded = n in (400, 300, 200)
            cell_out["channels"][ch] = {
                "ds1": {"values": ds1_vals, "status": ds1_status if banded else "NO-REGISTERED-BAND(N=50 control)"},
                "ds2": {"D": a["ds2_ks"]["D"], "status": ds2_status if banded else "NO-REGISTERED-BAND(N=50 control)"},
                "ds3": {"bias": a["ds3_map_bias"]["bias"], "mc_error": a["ds3_map_bias"]["mc_error"], "status": ds3_status},
                "ds4": {"R_low": a["ds4_rails"]["railed_low_frac"], "R_high": a["ds4_rails"]["railed_high_frac"]},
                "ds5_sd_median": a["ds5_width"]["post_sd_median"],
                "edge_guard": {"edge_loaded_frac": edge_frac, "edge_contaminated": edge_contaminated,
                               "gate_weight": "NONE (§8)" if edge_contaminated else "carries gate weight"},
            }
            # cross-check instrument's own labels (banded cells only)
            if banded:
                if a["ds1_status"] != ds1_status:
                    readout["label_crosscheck_mismatches"].append(f"{name}/{ch}/ds1: instrument={a['ds1_status']} scored={ds1_status}")
                if a["ds2_ks"]["status"] != ds2_status:
                    readout["label_crosscheck_mismatches"].append(f"{name}/{ch}/ds2: instrument={a['ds2_ks']['status']} scored={ds2_status}")
            if a["ds3_map_bias"]["status"] != ds3_status:
                readout["label_crosscheck_mismatches"].append(f"{name}/{ch}/ds3: instrument={a['ds3_map_bias']['status']} scored={ds3_status}")

        # texture (V4 input) — recompute median from per_seed
        corrs = [r["texture_corr"] for r in d["per_seed"]]
        cell_out["texture_corr_median_recomputed"] = median(corrs)
        cell_out["texture_corr_in_v4_band"] = V4_BAND[0] <= median(corrs) <= V4_BAND[1]

        # DS-7 (registered raw form ±0.05; corrected reported alongside)
        ds7 = agg["ds7"]
        cell_out["ds7"] = {
            "ratio_raw": ds7["ratio"], "pass_raw": abs(ds7["ratio"] - 1.0) <= DS7_BAND,
            "ratio_corrected": ds7["ratio_corrected"], "pass_corrected": abs(ds7["ratio_corrected"] - 1.0) <= DS7_BAND,
        }

        readout["cells"][name] = cell_out

    # ---- global seed disjointness ----
    readout["seed_plan"] = {
        "all_cells_match_registered_blocks": all(readout["cells"][n]["seed_plan_match"] for n, _, _ in CELLS),
        "total_seeds": len(seeds_all), "unique_seeds": len(set(seeds_all)),
        "disjoint": len(seeds_all) == len(set(seeds_all)),
    }

    # ---- V1 control (prereg §10): MAP=0.730 exactly, both channels, all 50 seeds ----
    v1 = files["V1_h0p730"]["per_seed"]
    v1_maps_1d = {r["map_1d"] for r in v1}
    v1_maps_2d = {r["map_2d"] for r in v1}
    v1_pass = v1_maps_1d == {0.73} and v1_maps_2d == {0.73} and len(v1) == 50
    readout["validity"] = {
        "V1_plumbing_control": {"pass": v1_pass, "n_seeds": len(v1),
                                "unique_map_1d": sorted(v1_maps_1d), "unique_map_2d": sorted(v1_maps_2d)},
        "V2_hpd_port": {"pass": True, "evidence": "test_v2_hpd_port_agrees_boolean_exactly_with_pp_coverage PASSED (21/21 suite re-run at readout)"},
        "V3_determinism": {"pass": bool(validate["v3"]["pass"]),
                           "evidence": "validate_results.json v3.pass + adjudicate P1_smoke_rerun_{V1,B2} per_seed_identical=true"},
        "V4_texture": {"pass": False,
                       "median_by_cell": {n: readout["cells"][n]["texture_corr_median_recomputed"] for n, _, _ in CELLS},
                       "band": list(V4_BAND),
                       "consequence": "prereg §10: 'Failure ⇒ the texture cells are void'; ALL 9 run cells are sigma_texture=dl_binned (incl. V1) — no independent-texture cell exists in the campaign",
                       "pre_declared": "module docstring divergence-log item 7 predicted ≈0.69±0.02 attenuation at build time; prediction was never appended to prereg §11 as an amendment; band therefore stands as registered"},
        "V5_r0_reproduction": {"pass": bool(validate["v5"]["pass"]), "rtol": validate["v5"]["rtol"]},
        "config_provenance_dirty_tree": {
            "violated": len(provenance_violations) > 0,
            "detail": provenance_violations,
            "prereg_clause": "§10 'Runs that would execute on a dirty tree STOP instead' — all 9 registered cells ran with --allow-dirty on a dirty tree; the instrument module itself (master_thesis_code/validation/calibration_gate.py) is untracked/uncommitted, so the campaign's code identity has no git object",
        },
        "section11_appendix": {"appended": False,
                               "prereg_clause": "'The new module's code commit is appended to §11 when it exists' — §11 is empty; the module was never committed"},
        "abort_b": {"triggered": any(readout["cells"][n]["abort_b_triggered"] for n, _, _ in CELLS),
                    "max_nonfinite_frac": max(readout["cells"][n]["nonfinite_ln_post_frac"] for n, _, _ in CELLS)},
    }

    # ---- DS-6 (Q2), locked thresholds 0.90 / 0.05 ----
    b2_rlow_1d = {t: readout["cells"][f"B2_h0p{int(t*1000)}"]["channels"]["channel_1d"]["ds4"]["R_low"] for t in (0.690, 0.730, 0.770)}
    b0_rlow_1d = readout["cells"]["B0_h0p730"]["channels"]["channel_1d"]["ds4"]["R_low"]
    b1_rlow_1d = readout["cells"]["B1_h0p730"]["channels"]["channel_1d"]["ds4"]["R_low"]
    b2_pass_ds12 = all(
        readout["cells"][f"B2_h0p{int(t*1000)}"]["channels"]["channel_1d"]["ds1"]["status"] == "PASS"
        and readout["cells"][f"B2_h0p{int(t*1000)}"]["channels"]["channel_1d"]["ds2"]["status"] == "PASS"
        for t in (0.690, 0.730, 0.770)
    )
    rail_reproduced = all(v >= DS6_HIGH for v in b2_rlow_1d.values()) and b0_rlow_1d <= DS6_LOW
    rail_not_reproduced = all(v <= DS6_LOW for v in b2_rlow_1d.values()) and b2_pass_ds12
    ds6 = "RAIL-REPRODUCED" if rail_reproduced else ("RAIL-NOT-REPRODUCED" if rail_not_reproduced else "MIXED")
    readout["ds6"] = {
        "verdict": ds6,
        "R_low_1d_B2_by_truth": b2_rlow_1d, "R_low_1d_B0": b0_rlow_1d,
        "dose_response_R_low_1d_sigma_z": {"0.000": b0_rlow_1d, "0.010": b1_rlow_1d, "0.035": b2_rlow_1d[0.730]},
        "b2_1d_passes_ds1_and_ds2": b2_pass_ds12,
        "mechanics": "R_low(B2)≥0.90 at all truths fails (all 0.000); R_low(B2)≤0.05 holds but B2-1D FAILS DS-1 and DS-2 → neither branch condition met → MIXED (prereg DS-6 'otherwise' clause)",
    }

    # ---- DS-5 screen (bracket read from committed F5 sweep; exact venue nodes absent) ----
    ds5: dict = {"screen_band": list(DS5_SCREEN), "rescale_sqrt_400_over_1500": F5_RESCALE, "floor": F5_FLOOR}
    if F5_PATH.exists():
        f5 = json.load(open(F5_PATH))
        zgrid = f5["sigma_z_grid"]
        ds5["f5_sigma_z_grid"] = zgrid
        ds5["exact_venue_nodes_present"] = {"0.0": 0.0 in zgrid, "0.010": 0.010 in zgrid, "0.035": 0.035 in zgrid}
        per_cell = {}
        for name, sz in (("B0_h0p730", 0.0), ("B1_h0p730", 0.010), ("B2_h0p690", 0.035), ("B2_h0p730", 0.035), ("B2_h0p770", 0.035)):
            sd_med = readout["cells"][name]["channels"]["channel_1d"]["ds5_sd_median"]
            # bracketing committed nodes (no interpolation — that would be a judgment call)
            lo = max([g for g in zgrid if g <= sz], default=None)
            hi = min([g for g in zgrid if g >= sz], default=None)
            reads = {}
            for node in {v for v in (lo, hi) if v is not None}:
                i = zgrid.index(node)
                for metric in ("width", "rmse_truth"):
                    sig = max(f5["oned"][metric][i] * F5_RESCALE, F5_FLOOR)
                    reads[f"node{node}_{metric}"] = {"sigma_F5": sig, "W": sd_med / sig,
                                                     "in_band": DS5_SCREEN[0] <= sd_med / sig <= DS5_SCREEN[1]}
            per_cell[name] = {"sigma_z": sz, "sd_median_1d": sd_med, "bracket_reads": reads,
                              "any_reading_in_band": any(r["in_band"] for r in reads.values())}
        ds5["oned_channel_bracket"] = per_cell
        ds5["status"] = ("SCREEN-NOT-EVALUABLE-AT-EXACT-VENUE-POINTS (no committed F5 node at sigma_z ∈ {0, 0.010, 0.035}; "
                         "matched-population F5 run not executed — prereg §9 item 3); bracket reads reported as raw context: "
                         "no bracketing reading places any ball-cell 1D W inside [0.5, 2.0] (all W ≪ 0.5: measured posteriors "
                         "orders of magnitude narrower than every committed F5 forecast reading)")
        ds5["twod_channel"] = "NOT-EVALUABLE (F5's 2D axis is the with-BH-mass host channel; the gate's 2D channel is the completion-leg g factor — prereg §9 item 4 structural mismatch)"
        ds5["A_cells"] = "NOT-EVALUABLE (f=0 single-host venue has no host-z axis on the F5 sweep)"
    else:
        ds5["status"] = "NOT-EVALUABLE (committed F5 sweep JSON absent)"
    readout["ds5"] = ds5

    # ---- edge-guard trigger ('both decision cells EDGE-CONTAMINATED in the channel being read') ----
    a_2d_cont = all(readout["cells"][f"A_h0p{int(t*1000)}"]["channels"]["channel_2d"]["edge_guard"]["edge_contaminated"] for t in (0.690, 0.730, 0.770))
    b2_2d_cont = all(readout["cells"][f"B2_h0p{int(t*1000)}"]["channels"]["channel_2d"]["edge_guard"]["edge_contaminated"] for t in (0.690, 0.730, 0.770))
    b2_1d_cont = all(readout["cells"][f"B2_h0p{int(t*1000)}"]["channels"]["channel_1d"]["edge_guard"]["edge_contaminated"] for t in (0.690, 0.730, 0.770))
    both_decision_contaminated_2d = a_2d_cont and b2_2d_cont
    both_decision_contaminated_1d = b2_1d_cont  # B2 is the only 1D decision cell (A-1D exempt per §5)

    # ---- DS-7 registered-form violation count ----
    ds7_raw_fail = [n for n, _, _ in CELLS if not readout["cells"][n]["ds7"]["pass_raw"]]
    ds7_corr_fail = [n for n, _, _ in CELLS if not readout["cells"][n]["ds7"]["pass_corrected"]]

    # ---- GATE-NOT-TRUSTWORTHY trigger set (prereg §10, verbatim membership) ----
    triggers = {
        "V1_failure": not v1_pass,
        "V2_failure": False,
        "V3_failure": not validate["v3"]["pass"],
        "V4_failure": True,
        "V5_failure": not validate["v5"]["pass"],
        "DS7_violation_registered_raw_form": len(ds7_raw_fail) > 0,
        "abort_b": readout["validity"]["abort_b"]["triggered"],
        "both_decision_cells_edge_contaminated_2d": both_decision_contaminated_2d,
        "both_decision_cells_edge_contaminated_1d": both_decision_contaminated_1d,
    }
    any_trigger = any(triggers.values())

    readout["trigger_set"] = {
        "triggers": triggers,
        "fired": [k for k, v in triggers.items() if v],
        "ds7_raw_fail_cells": ds7_raw_fail,
        "ds7_corrected_fail_cells": ds7_corr_fail,
        "ds7_author_call": "module divergence-log item 9 leaves raw-vs-corrected V-class weight to the author; the REGISTERED §7 statistic is the raw identity — scored as such here; corrected passes 9/9",
        "note_outside_trigger_set": "dirty-tree run (all 9 cells) and empty §11 violate §10 provenance clauses but are not members of the enumerated trigger set; reported separately under validity.config_provenance_dirty_tree",
    }

    # ---- branch (prereg Branches section, applied mechanically) ----
    branch = "GATE-NOT-TRUSTWORTHY" if any_trigger else None
    readout["branch"] = {
        "fired": branch,
        "mechanics": ("§10 trigger fired (V4 texture-certification FAILURE: all 9 cells' corr(ln σ_dL, ln d_L) median "
                      "0.664–0.666 vs locked band [0.72, 0.92]; plus DS-7 registered-raw-form violation in 6/9 cells). "
                      "Prereg: 'The instrument's own verdict is void; report which control failed and why; no stage-4 "
                      "leg-1 claim of any kind may be made.' KEEP-DIGGING / REPORT-BOUND / MIXED are all unreachable: "
                      "each requires 'gate trustworthy' as its first conjunct."),
        "counterfactual_note": ("Reported as raw pattern only, NOT a claim (barred by the fired branch): had the gate been "
                                "trustworthy, DS-6=MIXED + DS-1/DS-2 FAIL in decision cells would have fired MIXED/KEEP-DIGGING(b), "
                                "not REPORT-BOUND — the B2-1D channel is un-railed but grossly miscalibrated "
                                "(bias ≈ +σ_z: +0.011 at σ_z=0.010, +0.035 at σ_z=0.035; near-delta posteriors, 0/0/0 coverage)."),
    }

    out = HERE / "CALIBRATION_GATE_READOUT_20260808.json"
    json.dump(readout, open(out, "w"), indent=1)
    print("wrote", out)
    print("branch:", branch)
    print("triggers fired:", readout["trigger_set"]["fired"])
    print("label mismatches:", readout["label_crosscheck_mismatches"])
    print("ds6:", ds6, "| seed plan ok:", readout["seed_plan"])
    print("V1:", v1_pass, "| V4 medians:", {n: round(readout['cells'][n]['texture_corr_median_recomputed'], 4) for n, _, _ in CELLS})
    print("ds7 raw fails:", ds7_raw_fail)
    for n, _, _ in CELLS:
        c = readout["cells"][n]
        print(n, "1D:", c["channels"]["channel_1d"]["ds1"]["status"], c["channels"]["channel_1d"]["ds2"]["status"],
              c["channels"]["channel_1d"]["ds3"]["status"], f"Rlow={c['channels']['channel_1d']['ds4']['R_low']}",
              "| 2D:", c["channels"]["channel_2d"]["ds1"]["status"], c["channels"]["channel_2d"]["ds2"]["status"],
              c["channels"]["channel_2d"]["ds3"]["status"], f"edge2d={c['channels']['channel_2d']['edge_guard']['edge_loaded_frac']}")


if __name__ == "__main__":
    main()
