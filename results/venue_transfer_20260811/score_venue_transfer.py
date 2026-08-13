#!/usr/bin/env python3
"""score_venue_transfer.py — MECHANICAL scorer for the VENUE-TRANSFER READ.

Reads the 49 registered chunk JSONs in this directory DIRECTLY (it does not
trust collect_raw.json), recomputes every statistic, and scores them against
the bands locked in PREREGISTRATION_VENUE_TRANSFER.md (registered e77eecad).

Read-only on all inputs. Emits VENUE_TRANSFER_READOUT.json next to itself.
It computes the branch that the registered decision tree fires; it does not
adjudicate — the author rules.

Run:  cd <repo root> && uv run python results/venue_transfer_20260811/score_venue_transfer.py
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# ----------------------------------------------------------------------------
# BANDS — verbatim from the registered prereg. Locked at commit e77eecad;
# not adjustable after any readout (prereg "Anti-tuning").
# ----------------------------------------------------------------------------

# DS-VT1 §7 HPD coverage nulls, per N row: (2sigma_lo, 2sigma_hi, 3sigma_lo, 3sigma_hi)
DS_VT1: dict[int, dict[str, tuple[float, float, float, float]]] = {
    400: {
        "hpd50": (0.450, 0.550, 0.425, 0.575),
        "hpd68": (0.633, 0.727, 0.610, 0.750),
        "hpd90": (0.870, 0.930, 0.855, 0.945),
    },
    200: {
        "hpd50": (0.429, 0.571, 0.394, 0.606),
        "hpd68": (0.614, 0.746, 0.581, 0.779),
        "hpd90": (0.858, 0.942, 0.836, 0.964),
    },
    100: {
        "hpd50": (0.400, 0.600, 0.350, 0.650),
        "hpd68": (0.587, 0.773, 0.540, 0.820),
        "hpd90": (0.840, 0.960, 0.810, 0.990),
    },
}
# DS-VT2 §7 KS: (PASS edge D<=, FAIL edge D>)
DS_VT2 = {400: (0.0679, 0.0814), 200: (0.0960, 0.1151), 100: (0.1358, 0.1628)}
# DS-VT3 §7
BIAS_IN_BAND = 0.010          # |b| <= 0.010 -> in-band
BIAS_DEFECT = 0.030           # |b| >= 0.030 -> DEFECT-scale
R_DOSE_BAND = (0.75, 1.25)
# DS-VT4 §7 collapse-pattern rail bands per N row
RAIL_BAND = {400: 0.02, 200: 0.04, 100: 0.08}
RAIL_EMERGENT = 0.90          # pre-named distinct pattern, decision cells
# §8 edge-contamination guard
EDGE_MASS_LOADED = 0.01
EDGE_CONTAM_FRAC = 0.10
# §10 V-T1 T-0 anchor edges
VT1_BIAS_OK = 0.010
VT1_BIAS_HARD = 0.030
VT1_RAIL_HARD = 0.05
# §10 abort criteria
ABORT_B_NONFINITE = 0.01      # >1% of a cell's seeds
ABORT_D_HORIZON = 0.05        # >5% of the pinned set
ABORT_A_CPU_ANCHOR_H = 4.33   # derived CPU-h/seed, heavy cells
ABORT_A_FACTOR = 2.0
N_PINNED_EVENTS = 982         # VT-D5
K_SUM_PIN = 1_193_703         # VT-D2 census pin (real_k balls only)

# §5 cell matrix: cell -> (prereg name, h_true, glob, expected N, seed block)
BASE_SEED = 20260808
CELLS: dict[str, dict[str, Any]] = {
    "T0": dict(name="T-0", h=0.730, glob="T0_h0p730_results_seeds*.json", n=200,
               off=(40000, 40199), balls="real_k", sigma="zero", role="anchor"),
    "Ta": dict(name="T-a", h=0.730, glob="Ta_h0p730_results_seeds*.json", n=200,
               off=(41000, 41199), balls="poisson4", sigma="flat0.035", role="ladder"),
    "Tb": dict(name="T-b", h=0.730, glob="Tb_h0p730_results_seeds*.json", n=200,
               off=(42000, 42199), balls="real_k", sigma="flat0.035", role="ladder"),
    "Tc_h0p690": dict(name="T-c(0.690)", h=0.690, glob="Tc_h0p690_results_seeds*.json", n=200,
                      off=(43000, 43199), balls="real_k", sigma="glade", role="decision-wing"),
    "Tc_h0p730": dict(name="T-c(0.730)", h=0.730, glob="Tc_h0p730_results_seeds*.json", n=400,
                      off=(44000, 44399), balls="real_k", sigma="glade", role="DECISION"),
    "Tc_h0p770": dict(name="T-c(0.770)", h=0.770, glob="Tc_h0p770_results_seeds*.json", n=200,
                      off=(45000, 45199), balls="real_k", sigma="glade", role="decision-wing"),
}
DECISION_CELLS = ["Tc_h0p690", "Tc_h0p730", "Tc_h0p770"]

# committed v2 B2(0.730) baseline — DS-VT5 ladder rung 0 (quotable per R2)
V2_B2 = {
    "source": "results/calibration_gate_v2_20260810/B2_h0p730_results.json (committed 64abd5f6)",
    "n": 400, "h_true": 0.730,
    "1d": dict(hpd50=0.0, hpd68=0.0, hpd90=0.0, ks_D=1.0, bias_argmax=0.0352625,
               rail_low=0.0, rail_high=0.0, R_dose=1.0075, sigma_bar=0.035),
    "2d": dict(hpd50=0.0, hpd68=0.0, hpd90=0.0, ks_D=0.9999999956170457, bias_argmax=0.0357375,
               rail_low=0.0, rail_high=0.0, R_dose=1.0211, sigma_bar=0.035),
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def ks_uniform(u: list[float]) -> float:
    """One-sample KS statistic of u against U(0,1)."""
    n = len(u)
    if n == 0:
        return float("nan")
    s = sorted(u)
    return max(max((i + 1) / n - s[i], s[i] - i / n) for i in range(n))


def mean(x: list[float]) -> float:
    return sum(x) / len(x)


def sd(x: list[float]) -> float:
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else 0.0


def median(x: list[float]) -> float:
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def n_row(n: int) -> int:
    """Map a realized N onto the registered per-N band row (exact rows only)."""
    if n in (400, 200, 100):
        return n
    raise SystemExit(f"realized N={n} has no registered band row (400/200/100 only)")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
chunks: dict[str, list[dict[str, Any]]] = {}
for cell, spec in CELLS.items():
    files = sorted(HERE.glob(spec["glob"]))
    chunks[cell] = [json.loads(f.read_text()) for f in files]
    spec["files"] = [f.name for f in files]

n_chunk_files = sum(len(v) for v in chunks.values())

# ---------------- provenance (V-T4 + registered-commit chain) ----------------
commit_counts: dict[str, int] = {}
import_clean, smoke_false, allow_dirty_false, pin_pass = [], [], [], []
dirt_import_nonempty = []
workers_seen: dict[str, int] = {}
for cell, cl in chunks.items():
    for c in cl:
        sha = c["git_commit"][:8]
        commit_counts[sha] = commit_counts.get(sha, 0) + 1
        import_clean.append(bool(c["import_path_clean"]))
        smoke_false.append(c["smoke"] is False)
        allow_dirty_false.append(c["allow_dirty"] is False)
        pin_pass.append(bool(c["pin_integrity"]["pass"]))
        if c["dirt_inventory"]["import_path"]:
            dirt_import_nonempty.append(c["git_commit"][:8])
        workers_seen[str(c["workers"])] = workers_seen.get(str(c["workers"]), 0) + 1

commits = sorted(commit_counts)
older, newer = "2ece8801", "e93f3068"
anc = subprocess.run(["git", "merge-base", "--is-ancestor", older, newer], cwd=REPO).returncode == 0
imp_diff_old = git("diff", "--stat", older, newer, "--", "master_thesis_code/", "master_thesis_code_test/")
imp_diff_new = git("diff", "--stat", older, newer, "--", "darksiren_emri/", "darksiren_emri_test/")
full_diff = git("diff", "--name-only", older, newer)
prereg_hunks = git("diff", "-U0", older, newer, "--",
                   "results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md")
prereg_hunks = [ln for ln in prereg_hunks.splitlines() if ln.startswith("@@")]
rename_desc = git("log", "-1", "--format=%h %s", "227e7a32")
rename_after = subprocess.run(["git", "merge-base", "--is-ancestor", newer, "227e7a32"],
                              cwd=REPO).returncode == 0

provenance = {
    "n_chunk_files": n_chunk_files,
    "commit_counts": commit_counts,
    "registered_instrument_commit": "2ece8801",
    "registered_prereg_commit": "e77eecad",
    "commit_chain": {
        "older": older, "newer": newer,
        "ancestor_verified": anc,
        "import_path_diff_old_names_empty": imp_diff_old == "",
        "import_path_diff_new_names_empty": imp_diff_new == "",
        "files_touched_between": full_diff.splitlines(),
        "prereg_diff_hunks": prereg_hunks,
        "prereg_diff_is_pure_append_below_sec11": all(
            h.startswith("@@ -607,0") or h.startswith("@@ -60") for h in prereg_hunks),
    },
    "all_import_path_clean": all(import_clean),
    "all_smoke_false": all(smoke_false),
    "all_allow_dirty_false": all(allow_dirty_false),
    "all_pin_integrity_pass": all(pin_pass),
    "chunks_with_dirty_import_path": dirt_import_nonempty,
    "workers_field_counts": workers_seen,
    "rename_disclosure": {
        "rename_commit": rename_desc,
        "rename_postdates_campaign": rename_after,
        "note": ("V-T4's registered clean-rule wording names master_thesis_code/ and "
                 "master_thesis_code_test/; the package was renamed to darksiren_emri AFTER the "
                 "campaign, so every chunk's import_path_clean flag was correctly evaluated "
                 "against the then-current path. Disclosure, not defect."),
    },
}

# ---------------- seed plan (VT-D7) ----------------
seed_plan: dict[str, Any] = {"per_cell": {}, "cross_cell": {}}
all_seeds: list[int] = []
for cell, spec in CELLS.items():
    seeds = [s["seed"] for c in chunks[cell] for s in c["per_seed"]]
    exp = set(range(BASE_SEED + spec["off"][0], BASE_SEED + spec["off"][1] + 1))
    got = set(seeds)
    all_seeds += seeds
    seed_plan["per_cell"][cell] = {
        "prereg_name": spec["name"],
        "expected_n": spec["n"], "realized_n": len(seeds), "realized_unique": len(got),
        "expected_block": [min(exp), max(exp)],
        "missing": sorted(exp - got), "extra": sorted(got - exp),
        "duplicates_within_cell": len(seeds) - len(got),
        "exact_match": got == exp and len(seeds) == spec["n"],
    }
v1 = set(range(BASE_SEED, BASE_SEED + 9050))
v2 = set(range(BASE_SEED + 20000, BASE_SEED + 29050))
w1 = set(range(BASE_SEED + 46000, BASE_SEED + 46400))
o2 = set(range(BASE_SEED + 47000, BASE_SEED + 47400))
seed_plan["cross_cell"] = {
    "total_seeds": len(all_seeds), "total_unique": len(set(all_seeds)),
    "cross_cell_duplicates": len(all_seeds) - len(set(all_seeds)),
    "v1_collisions": len(set(all_seeds) & v1), "v2_collisions": len(set(all_seeds) & v2),
    "W1_reserved_collisions": len(set(all_seeds) & w1),
    "O2_reserved_collisions": len(set(all_seeds) & o2),
    "registered_total": 1400,
    "pass": (len(all_seeds) == 1400 and len(set(all_seeds)) == 1400
             and not (set(all_seeds) & (v1 | v2 | w1 | o2))
             and all(v["exact_match"] for v in seed_plan["per_cell"].values())),
}

# ---------------- per-cell / per-channel statistics ----------------
stats: dict[str, Any] = {}
for cell, spec in CELLS.items():
    rows = [s for c in chunks[cell] for s in c["per_seed"]]
    h = spec["h"]
    n = len(rows)
    cell_out: dict[str, Any] = {
        "prereg_name": spec["name"], "role": spec["role"], "h_true": h, "n_seeds": n,
        "n_row_used": n_row(n),
        "balls": sorted({s["balls"] for s in rows}),
        "sigma_mode": sorted({s["sigma_mode"] for s in rows}),
        "n_events_all_982": all(s["n_events"] == s["n_events_run"] == N_PINNED_EVENTS for s in rows),
        "f_incl_all_1": all(s["f_incl"] == 1.0 for s in rows),
    }
    # abort (b) non-finite ln_post, per channel
    for ch in ("1d", "2d"):
        bad = sum(1 for s in rows
                  if not all(math.isfinite(v) for v in s[f"ln_post_{ch}"]))
        cell_out[f"abort_b_nonfinite_{ch}"] = {
            "count": bad, "fraction": bad / n, "stop_triggered": bad / n > ABORT_B_NONFINITE}
    # abort (d) horizon drops
    drops = [s["n_horizon_dropped"] for s in rows]
    cell_out["abort_d_horizon"] = {
        "max_dropped": max(drops), "max_fraction_of_982": max(drops) / N_PINNED_EVENTS,
        "n_seeds_over_5pct": sum(1 for d in drops if d / N_PINNED_EVENTS > ABORT_D_HORIZON),
        "stop_triggered": max(drops) / N_PINNED_EVENTS > ABORT_D_HORIZON}
    # K_sum pin (real_k cells only; T-a is poisson4 by registered design VT-D2)
    ksums = [s["K_sum"] for s in rows]
    if spec["balls"] == "real_k":
        cell_out["K_sum_pin"] = {"applicable": True, "pin": K_SUM_PIN,
                                 "all_match": all(k == K_SUM_PIN for k in ksums),
                                 "n_mismatch": sum(1 for k in ksums if k != K_SUM_PIN),
                                 "min_max": [min(ksums), max(ksums)]}
    else:
        cell_out["K_sum_pin"] = {"applicable": False, "pin": None,
                                 "min_max": [min(ksums), max(ksums)], "distinct": len(set(ksums)),
                                 "note": "poisson4 balls (VT-D2/§5): K_sum is a per-seed Poisson draw, no pin"}
    # wall/CPU disclosure
    walls = [c["wall_time_per_seed_s"] for c in chunks[cell]]
    cell_out["wall_time_per_seed_s"] = {"min": min(walls), "max": max(walls), "median": median(walls)}

    sig = [s["sigma_z_mean_pairs"] for s in rows]
    sigma_bar = mean(sig)
    cell_out["sigma_bar_pairs"] = sigma_bar
    cell_out["sigma_z_median_pairs_mean"] = mean([s["sigma_z_median_pairs"] for s in rows])
    cell_out["frac_pairs_sigma_lt_5e-3_mean"] = mean([s["frac_pairs_sigma_lt_5e-3"] for s in rows])

    for ch in ("1d", "2d"):
        cov = {lvl: mean([s[f"{lvl}_{ch}"] for s in rows]) for lvl in ("hpd50", "hpd68", "hpd90")}
        pit = [s[f"pit_{ch}"] for s in rows]
        D = ks_uniform(pit)
        maps = [s[f"map_{ch}"] for s in rows]
        mapsr = [s[f"map_{ch}_refined"] for s in rows]
        b_arg = mean(maps) - h
        b_ref = mean(mapsr) - h
        rl = mean([s[f"railed_low_{ch}"] for s in rows])
        rh = mean([s[f"railed_high_{ch}"] for s in rows])
        em = [s[f"edge_mass_{ch}"] for s in rows]
        edge_loaded = sum(1 for v in em if v > EDGE_MASS_LOADED) / n
        cell_out[ch] = {
            "hpd": cov,
            "pit_ks_D": D,
            "bias_argmax": b_arg, "bias_argmax_se": sd(maps) / math.sqrt(n),
            "bias_refined": b_ref, "bias_refined_se": sd(mapsr) / math.sqrt(n),
            "R_dose_argmax": (b_arg / sigma_bar) if sigma_bar > 0 else None,
            "R_dose_refined": (b_ref / sigma_bar) if sigma_bar > 0 else None,
            "rail_low": rl, "rail_high": rh,
            "post_sd_median": median([s[f"post_sd_{ch}"] for s in rows]),
            "edge_loaded_fraction": edge_loaded,
            "edge_contaminated": edge_loaded > EDGE_CONTAM_FRAC,
            "max_edge_mass": max(em),
        }
    stats[cell] = cell_out

# ---------------- V-T checks ----------------
vfull = json.loads((HERE / "validate_results_full.json").read_text())
vnov = json.loads((HERE / "validate_results_novt5.json").read_text())

t0 = stats["T0"]
vt1_members = []
for ch in ("1d", "2d"):
    b = abs(t0[ch]["bias_argmax"])
    br = abs(t0[ch]["bias_refined"])
    vt1_members.append({
        "channel": ch,
        "bias_argmax": t0[ch]["bias_argmax"], "bias_refined": t0[ch]["bias_refined"],
        "rail_low": t0[ch]["rail_low"], "rail_high": t0[ch]["rail_high"],
        "bias_ok_le_0.010": max(b, br) <= VT1_BIAS_OK,
        "anchor_marginal_zone": VT1_BIAS_OK < max(b, br) < VT1_BIAS_HARD,
        "hard_trigger": (max(b, br) >= VT1_BIAS_HARD
                         or t0[ch]["rail_low"] > VT1_RAIL_HARD
                         or t0[ch]["rail_high"] > VT1_RAIL_HARD),
    })
vt1_pass = all(m["bias_ok_le_0.010"] and not m["hard_trigger"] for m in vt1_members)

vt = {
    "V-T1": {"verdict": "PASS" if vt1_pass else "FAIL",
             "hard_trigger_fired": any(m["hard_trigger"] for m in vt1_members),
             "anchor_marginal": any(m["anchor_marginal_zone"] for m in vt1_members),
             "per_channel": vt1_members,
             "evidence": "T-0 cell, 200 seeds, 8 chunk JSONs (recomputed here from per_seed records)",
             "exemption": "DS-VT1/DS-VT2 not scored on T-0 (degenerate-PIT exemption, VT-D8)"},
    "V-T2": {"verdict": "PASS" if vfull["v_t2"]["pass"] else "FAIL",
             "detail": vfull["v_t2"],
             "evidence": "validate_results_full.json v_t2 (dev box, instrument 2ece8801); "
                         "corroborated by identical K_sum pin match across workers=64 and workers=25 chunks"},
    "V-T3": {"verdict": "PASS" if (vfull["v_t3"]["pass"] and all(pin_pass)) else "FAIL",
             "validate_pass": vfull["v_t3"]["pass"],
             "per_chunk_pin_integrity_all_pass": all(pin_pass),
             "n_chunks_checked": n_chunk_files,
             "evidence": "validate_results_full.json v_t3 + pin_integrity.pass in all 49 chunk JSONs"},
    "V-T4": {"verdict": "PASS" if (all(import_clean) and all(allow_dirty_false)
                                   and all(smoke_false) and not dirt_import_nonempty) else "FAIL",
             "all_import_path_clean": all(import_clean),
             "all_allow_dirty_false": all(allow_dirty_false),
             "all_smoke_false": all(smoke_false),
             "chunks_with_import_path_dirt": dirt_import_nonempty,
             "evidence": "import_path_clean / allow_dirty / smoke / dirt_inventory embedded in all 49 chunk JSONs",
             "disclosure": provenance["rename_disclosure"]["note"]},
    "V-T5": {"verdict": "PASS" if vfull["v_t5"]["pass"] else "FAIL",
             "detail": {k: vfull["v_t5"][k] for k in ("pass", "committed_json", "seeds")},
             "n_seeds_bit_identical": sum(1 for p in vfull["v_t5"]["per_seed"] if p["pass"]),
             "evidence": "validate_results_full.json v_t5 (3/3 seeds bit-identical to committed v2 B2(0.730))",
             "sequencing_disclosure": "the earlier launch-phase validate (validate_results_novt5.json) "
                                      "skipped V-T5 (v_t5.pass=null); prereg §11 addendum 1 logs the "
                                      "compliance-order deviation, PENDING AUTHOR RATIFICATION"},
}

abort = {
    "(a) smoke CPU > 2x 4.33 CPU-h/seed": {
        "triggered": False,
        "registered_anchor_CPU_h_per_seed": ABORT_A_CPU_ANCHOR_H,
        "trip_point_CPU_h_per_seed": ABORT_A_CPU_ANCHOR_H * ABORT_A_FACTOR,
        "measured_uncontended_CPU_h_per_seed": 3.79,
        "ratio_to_anchor": round(3.79 / ABORT_A_CPU_ANCHOR_H, 3),
        "evidence": "prereg §11 note 1 (array 6252702 task 28: 94.63 CPU-h / 25 seeds)",
        "consequence": "no N-floor fallback stage invoked; all cells ran at full registered N",
    },
    "(b) non-finite ln_post > 1% of any cell": {
        "triggered": any(stats[c][f"abort_b_nonfinite_{ch}"]["stop_triggered"]
                         for c in CELLS for ch in ("1d", "2d")),
        "max_fraction_seen": max(stats[c][f"abort_b_nonfinite_{ch}"]["fraction"]
                                 for c in CELLS for ch in ("1d", "2d")),
        "evidence": "recomputed from the 41-point ln_post_1d/ln_post_2d vectors of all 1400 seeds",
    },
    "(c) any V-T failure": {
        "triggered": any(v["verdict"] == "FAIL" for v in vt.values()),
        "per_check": {k: v["verdict"] for k, v in vt.items()},
    },
    "(d) horizon-drop guard > 5%": {
        "triggered": any(stats[c]["abort_d_horizon"]["stop_triggered"] for c in CELLS),
        "max_fraction_seen": max(stats[c]["abort_d_horizon"]["max_fraction_of_982"] for c in CELLS),
        "evidence": "n_horizon_dropped on all 1400 seeds",
    },
}

# §8 edge guard table
edge_guard = {c: {ch: {"edge_loaded_fraction": stats[c][ch]["edge_loaded_fraction"],
                       "max_edge_mass": stats[c][ch]["max_edge_mass"],
                       "EDGE_CONTAMINATED": stats[c][ch]["edge_contaminated"]}
                  for ch in ("1d", "2d")} for c in CELLS}

# VENUE-CONFOUNDED trigger set (§10), member by member
trigger_set = [
    {"member": "V-T2 failure", "fired": vt["V-T2"]["verdict"] == "FAIL"},
    {"member": "V-T3 failure", "fired": vt["V-T3"]["verdict"] == "FAIL"},
    {"member": "V-T4 failure", "fired": vt["V-T4"]["verdict"] == "FAIL"},
    {"member": "V-T5 failure", "fired": vt["V-T5"]["verdict"] == "FAIL"},
    {"member": "abort (b) non-finite ln_post > 1%",
     "fired": abort["(b) non-finite ln_post > 1% of any cell"]["triggered"]},
    {"member": "abort (d) horizon drop > 5%",
     "fired": abort["(d) horizon-drop guard > 5%"]["triggered"]},
    {"member": "V-T1 T-0 hard trigger (|bias|>=0.030 or rail>0.05)", "fired": vt["V-T1"]["hard_trigger_fired"]},
    {"member": "decision cell EDGE-CONTAMINATED in the channel being read (1D, VT-D6)",
     "fired": any(stats[c]["1d"]["edge_contaminated"] for c in DECISION_CELLS)},
    {"member": "decision cell EDGE-CONTAMINATED, 2D secondary channel",
     "fired": any(stats[c]["2d"]["edge_contaminated"] for c in DECISION_CELLS)},
]
venue_confounded = any(m["fired"] for m in trigger_set)


# ---------------- DS-VT1..DS-VT4 scoring + classification ----------------
def score_channel(cell: str, ch: str) -> dict[str, Any]:
    s = stats[cell][ch]
    n = stats[cell]["n_seeds"]
    row = n_row(n)
    exempt = cell == "T0"  # VT-D8 degenerate-PIT exemption

    ds1 = {}
    for lvl in ("hpd50", "hpd68", "hpd90"):
        lo2, hi2, lo3, hi3 = DS_VT1[row][lvl]
        v = s["hpd"][lvl]
        ds1[lvl] = {"value": v, "band_2sigma": [lo2, hi2], "band_3sigma": [lo3, hi3],
                    "inside_2sigma": lo2 <= v <= hi2, "inside_3sigma": lo3 <= v <= hi3}
    ds1_all_in_3sigma = all(d["inside_3sigma"] for d in ds1.values())
    ds1_status = "EXEMPT (VT-D8 degenerate PIT)" if exempt else (
        "PASS" if ds1_all_in_3sigma else "FAIL")

    p_edge, f_edge = DS_VT2[row]
    D = s["pit_ks_D"]
    ds2_status = "EXEMPT (VT-D8 degenerate PIT)" if exempt else (
        "PASS" if D <= p_edge else ("FAIL" if D > f_edge else "MARGINAL"))

    b = s["bias_argmax"]
    ds3_status = ("IN-BAND" if abs(b) <= BIAS_IN_BAND else
                  ("DEFECT-SCALE" if abs(b) >= BIAS_DEFECT else "ATTENUATED (0.010<|b|<0.030)"))
    Rd = s["R_dose_argmax"]
    ds3_rdose = None if Rd is None else {
        "value": Rd, "band": list(R_DOSE_BAND),
        "in_band": R_DOSE_BAND[0] <= Rd <= R_DOSE_BAND[1]}

    rb = RAIL_BAND[row]
    ds4 = {"rail_low": s["rail_low"], "rail_high": s["rail_high"], "collapse_band_le": rb,
           "rail_low_in_band": s["rail_low"] <= rb, "rail_high_in_band": s["rail_high"] <= rb,
           "RAIL_EMERGENT": (cell in DECISION_CELLS
                             and max(s["rail_low"], s["rail_high"]) >= RAIL_EMERGENT)}

    # mechanical classification (prereg §7)
    rails_ok = ds4["rail_low_in_band"] and ds4["rail_high_in_band"]
    collapse = (s["hpd"]["hpd90"] <= rb and rails_ok and b >= BIAS_DEFECT
                and Rd is not None and R_DOSE_BAND[0] <= Rd <= R_DOSE_BAND[1])
    calibrated = (ds1_all_in_3sigma and ds2_status == "PASS"
                  and abs(b) <= BIAS_IN_BAND and rails_ok)
    if exempt:
        cls = "ANCHOR (not classified — DS-VT1/DS-VT2 exempt, VT-D8; scored on DS-3/DS-4 only)"
    elif collapse:
        cls = "COLLAPSE-REPRODUCED"
    elif calibrated:
        cls = "CALIBRATED"
    else:
        cls = "OTHER"

    return {"n": n, "n_row": row,
            "DS-VT1": {"levels": ds1, "status": ds1_status},
            "DS-VT2": {"D": D, "pass_edge_le": p_edge, "fail_edge_gt": f_edge, "status": ds2_status},
            "DS-VT3": {"bias_argmax": b, "bias_argmax_se": s["bias_argmax_se"],
                       "bias_refined": s["bias_refined"], "bias_refined_se": s["bias_refined_se"],
                       "sigma_bar_pairs": stats[cell]["sigma_bar_pairs"],
                       "status": ds3_status, "R_dose": ds3_rdose,
                       "R_dose_refined": s["R_dose_refined"]},
            "DS-VT4": ds4,
            "delta_narrow_companion_unbanded": {
                "post_sd_median": s["post_sd_median"],
                "ratio_bias_over_post_sd": (b / s["post_sd_median"]) if s["post_sd_median"] else None,
                "v2_committed_reference_range": [0.0012, 0.0059]},
            "classification": cls}


scored = {c: {ch: score_channel(c, ch) for ch in ("1d", "2d")} for c in CELLS}

# ---------------- DS-VT5 ladder ----------------
ladder = [
    {"rung": 0, "arm": "v2 B2(0.730) [committed baseline, quotable per R2]", "N": 400,
     "axes": "gate caricature: synthetic universe, Poisson λ=4 balls, flat σ_z=0.035",
     "1d": {"classification": "COLLAPSE-REPRODUCED (committed v2 record)",
            "bias_argmax": V2_B2["1d"]["bias_argmax"], "hpd90": 0.0, "R_dose": V2_B2["1d"]["R_dose"],
            "rails": [0.0, 0.0]},
     "2d": {"classification": "COLLAPSE-REPRODUCED (committed v2 record)",
            "bias_argmax": V2_B2["2d"]["bias_argmax"], "hpd90": 0.0, "R_dose": V2_B2["2d"]["R_dose"],
            "rails": [0.0, 0.0]}},
]
for rung, cell, axis in ((1, "Ta", "+ real event population (axis a)"),
                         (2, "Tb", "+ real ball multiplicity, real K_i (axis b-multiplicity)"),
                         (3, "Tc_h0p730", "+ real GLADE heterogeneous σ_z (axis c) — DECISION CELL")):
    ladder.append({
        "rung": rung, "arm": CELLS[cell]["name"], "N": stats[cell]["n_seeds"], "axes": axis,
        "1d": {"classification": scored[cell]["1d"]["classification"],
               "bias_argmax": scored[cell]["1d"]["DS-VT3"]["bias_argmax"],
               "hpd90": stats[cell]["1d"]["hpd"]["hpd90"],
               "R_dose": scored[cell]["1d"]["DS-VT3"]["R_dose"]["value"],
               "rails": [stats[cell]["1d"]["rail_low"], stats[cell]["1d"]["rail_high"]]},
        "2d": {"classification": scored[cell]["2d"]["classification"],
               "bias_argmax": scored[cell]["2d"]["DS-VT3"]["bias_argmax"],
               "hpd90": stats[cell]["2d"]["hpd"]["hpd90"],
               "R_dose": scored[cell]["2d"]["DS-VT3"]["R_dose"]["value"],
               "rails": [stats[cell]["2d"]["rail_low"], stats[cell]["2d"]["rail_high"]]},
    })
killing_axis = next((r["axes"] for r in ladder[1:]
                     if r["1d"]["classification"] != "COLLAPSE-REPRODUCED"), None)
ladder_note_Ta = {
    "T-a vs committed v2 B2(0.730) (raw context only, no band carries — pre-stated DS-VT5)": {
        "v2_1d_bias": V2_B2["1d"]["bias_argmax"], "T-a_1d_bias": scored["Ta"]["1d"]["DS-VT3"]["bias_argmax"],
        "delta_1d": scored["Ta"]["1d"]["DS-VT3"]["bias_argmax"] - V2_B2["1d"]["bias_argmax"],
        "v2_2d_bias": V2_B2["2d"]["bias_argmax"], "T-a_2d_bias": scored["Ta"]["2d"]["DS-VT3"]["bias_argmax"],
        "delta_2d": scored["Ta"]["2d"]["DS-VT3"]["bias_argmax"] - V2_B2["2d"]["bias_argmax"],
    }
}

# ---------------- branch tree, checked in the registered order ----------------
tc_1d = [scored[c]["1d"]["classification"] for c in DECISION_CELLS]
tc_2d = [scored[c]["2d"]["classification"] for c in DECISION_CELLS]
transfer_confirmed = all(x == "COLLAPSE-REPRODUCED" for x in tc_1d)
transfer_refuted = scored["Tc_h0p730"]["1d"]["classification"] == "CALIBRATED"
split_1d_2d = (all(x == "COLLAPSE-REPRODUCED" for x in tc_1d)
               != all(x == "COLLAPSE-REPRODUCED" for x in tc_2d))
rail_emergent = any(scored[c][ch]["DS-VT4"]["RAIL_EMERGENT"] for c in DECISION_CELLS for ch in ("1d", "2d"))

order = [
    {"n": 1, "branch": "VENUE-CONFOUNDED", "condition": "any trigger-set member fires",
     "fires": venue_confounded},
    {"n": 2, "branch": "TRANSFER-CONFIRMED",
     "condition": "T-c 1D COLLAPSE-REPRODUCED at all three truths (0.730 @N=400 rows, wings @N=200 rows)",
     "fires": (not venue_confounded) and transfer_confirmed,
     "detail": dict(zip(DECISION_CELLS, tc_1d))},
    {"n": 3, "branch": "TRANSFER-REFUTED", "condition": "T-c(0.730) 1D CALIBRATED",
     "fires": (not venue_confounded) and (not transfer_confirmed) and transfer_refuted},
    {"n": 4, "branch": "MIXED", "condition": "anything else (attenuated bias, partial coverage failure, "
                                             "RAIL-EMERGENT, 1D/2D split, wings disagreeing with centre)",
     "fires": (not venue_confounded) and (not transfer_confirmed) and (not transfer_refuted)},
]
fired = next(b["branch"] for b in order if b["fires"])

branch = {
    "checked_in_registered_order": order,
    "branch_fired": fired,
    "headline_channel": "1D (VT-D6)",
    "headline_classification_decision_cell_1d": scored["Tc_h0p730"]["1d"]["classification"],
    "secondary_2d_classification_decision_cell": scored["Tc_h0p730"]["2d"]["classification"],
    "wings_1d": dict(zip(DECISION_CELLS, tc_1d)),
    "wings_2d": dict(zip(DECISION_CELLS, tc_2d)),
    "1d_2d_split": split_1d_2d,
    "RAIL_EMERGENT_fired": rail_emergent,
    "adjudication": "NOT SELF-ADJUDICATED — presented to the author (prereg model/effort policy).",
}


# ---------------- §9 NOT-EVALUABLE registry (carried) ----------------
not_evaluable = [
    {"row": 1, "item": "Estimator code-path identity (axis d)",
     "status": "NOT-EVALUABLE — carried", "detail": "the gate mirror, not BayesianStatistics; "
     "certification chain V-T5 + T-0/T-a anchors (all PASS); any estimator fix routes /physics-change (R6)"},
    {"row": 2, "item": "volume_deconv kernel form",
     "status": "NOT-EVALUABLE — carried", "detail": "O2 arm reserved (+47000..47399), NOT BUILT; "
     "zero seeds realized in that block (verified)"},
    {"row": 3, "item": "Per-galaxy rate weights R_eff(M_g)/(1+z_g)",
     "status": "NOT-EVALUABLE — carried", "detail": "W1 arm reserved (+46000..46399), NOT BUILT; "
     "zero seeds realized in that block (verified). VT-D2 bracketing argument stands; author may order it post-read"},
    {"row": 4, "item": "f_incl < 1 / empty-ball events / completeness",
     "status": "NOT-EVALUABLE — carried", "detail": "the 606 zero-ball events excluded (VT-D5); "
     "read is conditional on host-in-ball, f_incl = 1.0 on all 1400 seeds (verified)"},
    {"row": 5, "item": "Window-interior n(z) shape (GLADE clustering + completeness roll-off inside W_i)",
     "status": "NOT-EVALUABLE — carried", "detail": "impostors stay w_pop|W; concentration bracket VT-D2"},
    {"row": 6, "item": "Sky-cone geometry / per-event sky selection",
     "status": "NOT-EVALUABLE — carried", "detail": "no sky in the mirror (v2 §9 item 5 residue)"},
    {"row": 7, "item": "With-BH-subset 2D ball realism",
     "status": "NOT-EVALUABLE — carried", "detail": "VT-D6 convention: 2D applies g_i over the SAME 1D ball; "
     "production's 2D ball is the with-BH subset (1294/1588 empty). The 2D verdict is secondary"},
    {"row": 8, "item": "DS-5 width-vs-F5 fine read",
     "status": "NOT-EVALUABLE — carried", "detail": "matched-population F5 run remains the registered follow-up "
     "(v2 §9 item 3). The delta-narrow companion is reported un-banded only"},
    {"row": "VT-D8", "item": "DS-7 generator closure",
     "status": "N/A in this venue", "detail": "no accept/reject generator (VT-D1); the R5 OPEN form call is untouched"},
]

# ---------------- disclosures ----------------
disclosures = [
    {"id": "D-VT-1", "class": "PENDING AUTHOR RATIFICATION (prereg §11 note 1, 2026-08-11)",
     "item": "Array 6252702 runtime blowout + resubmission",
     "detail": "49 tasks at --time=04:00:00 -> 10 COMPLETED / 39 TIMEOUT with no partial output. "
               "Root cause operational: mp.Pool parallelizes over SEEDS, so a 25-seed chunk cannot finish "
               "faster than one seed's single-process wall (~3.8-3.95 h). Seeds, seed->cell map, chunking, "
               "bands, statistics and instrument commit untouched; only --time/--cpus-per-task changed. "
               "NON-STATISTICAL. Abort (a) does NOT trip (3.79 vs 8.66 CPU-h/seed trip point); no N-floor "
               "fallback invoked."},
    {"id": "D-VT-2", "class": "PENDING AUTHOR RATIFICATION (prereg §11 addendum 1, 2026-08-12)",
     "item": "V-T5 compliance-order / sequencing deviation",
     "detail": "The prereg required the full §11 validity evidence BEFORE the campaign. The launch-phase "
               "validate skipped V-T5 (validate_results_novt5.json, v_t5.pass=null); the full run "
               "(validate_results_full.json, V-T5 PASS, 3/3 seeds bit-identical to committed v2 B2(0.730)) "
               "completed only after the first (partially timed-out) array. No statistical content; the check "
               "itself PASSES."},
    {"id": "D-VT-3", "class": "PENDING AUTHOR RATIFICATION (prereg §11 addendum 2, 2026-08-12)",
     "item": "Second straggler resubmission — contention",
     "detail": "Array 6253922 (39 tasks, 9 h, 25 cores): 17 COMPLETED / 22 TIMEOUT. Packed 25-core tasks run "
               "~1.6-1.9x slower than the uncontended 64-core reference (memory-bandwidth contention). "
               "Remaining 22 resubmitted as array 6259842 (--time=24:00:00), ALL 22 COMPLETED "
               "(sacct-verified 2026-08-13, zero FAILED/TIMEOUT). NON-STATISTICAL."},
    {"id": "D-VT-4", "class": "operational, result-invariant",
     "item": "workers grain 64 -> 25 across resubmissions",
     "detail": "10 first-wave chunks embed workers=64; 39 resubmitted chunks embed workers=25 (verified in the "
               "chunk JSONs). Result-invariance is certified by V-T2 (Pool maps seed->record deterministically) "
               "and corroborated here: every real_k seed at BOTH worker counts reproduces K_sum = 1,193,703 exactly, "
               "and the seed->cell map is unchanged."},
    {"id": "D-VT-5", "class": "naming shift, not a defect",
     "item": "Post-campaign package/repo rename vs the V-T4 import-path wording",
     "detail": "V-T4 names master_thesis_code/ and master_thesis_code_test/ as the import path. AFTER the campaign "
               "the package was renamed to darksiren_emri (commit 227e7a32, verified a descendant of both campaign "
               "commits) and the repo dir MasterThesisCode -> darksiren-emri. Every chunk's import_path_clean flag "
               "was therefore evaluated against the then-current (old) name at run time and is a correct clean-rule "
               "evaluation of the repo state at run time. The import-path diff 2ece8801..e93f3068 is empty under BOTH "
               "the old and the new path names (re-verified here)."},
    {"id": "D-VT-6", "class": "registered-commit chain",
     "item": "Two run commits (2ece8801 x10 chunks, e93f3068 x39 chunks)",
     "detail": "2ece8801 is the registered instrument commit; e93f3068 is a descendant with an EMPTY import-path diff "
               "(the R1-ratified D-4/D-5 pattern). The 4 files differing between them are markdown under results/ only. "
               "The prereg file itself is one of them: the diff is a PURE APPEND at line 608 (hunk @@ -607,0 +608,48 @@) "
               "into §11, i.e. VT-D0(ii) 'leaves every line above the §11 appendix unmodified' holds. The on-disk prereg "
               "above §11 is byte-identical (md5 388edd11254903d216fb16b1dbd476cb) to the registered e77eecad version."},
    {"id": "D-VT-7", "class": "statistic definition",
     "item": "MAP-bias statistic: grid-argmax (registered) vs refined",
     "detail": "DS-VT3 registers GRID-ARGMAX bias, and the committed v2 anchors (+0.035263 / +0.035737) are grid-argmax "
               "(mean_map - h_true). This scorer uses grid-argmax as PRIMARY and reports the refined-argmax companion "
               "alongside. Every DS-VT3 status and every cell classification is IDENTICAL under both definitions "
               "(max |difference| across all 12 cell x channel entries < 0.0002). The upstream extraction "
               "(collect_raw.json) reported the refined variant."},
    {"id": "D-VT-8", "class": "registered design, not an integrity failure",
     "item": "T-a is exempt from the K_sum = 1,193,703 census pin",
     "detail": "T-a runs balls='poisson4' by registered design (§5 / VT-D2), so its K_sum is a per-seed Poisson-lambda=4 "
               "draw (realized range 4757-5128, 137 distinct values), not a fixed pin. The five real_k cells match the "
               "pin exactly on all 1200 of their seeds. An earlier draft of the upstream extraction script applied the "
               "pin to T-a and produced a spurious 'mismatch'; corrected before finalizing."},
    {"id": "D-VT-9", "class": "read-the-numbers-this-way",
     "item": "PIT is saturated, not merely failing",
     "detail": "In every dosed cell the per-seed PIT values sit at ~1e-20 or below, so the KS statistic saturates at "
               "D ~ 1.0 (reported to 1e-12 precision). This is a total coverage failure, not a marginal one: the "
               "posterior never contains the truth on ANY seed (HPD50/68/90 = 0/0/0 on all 1400 dosed seeds). "
               "T-0's PIT is identically 0.5 on all 200 seeds (degenerate) — hence the VT-D8 exemption; its "
               "HPD 1.000/1.000/1.000 is likewise degenerate and carries no coverage information."},
    {"id": "D-VT-10", "class": "read-the-numbers-this-way",
     "item": "Realized dose sigma_bar exceeded the pre-registered prediction slightly",
     "detail": "VT-D3 predicted realized pair-mean dose sigma_bar in ~[0.039, 0.041]; the realized T-c values are "
               "0.041452 / 0.041775 / 0.042082 (0.690/0.730/0.770) — just above the top of the predicted window. "
               "R_dose is computed against the REALIZED sigma_bar as registered, so the band comparison is unaffected; "
               "a larger sigma_bar mechanically LOWERS R_dose (i.e. this works against, not for, the CONFIRMED call)."},
    {"id": "D-VT-11", "class": "operational context",
     "item": "Total campaign compute and per-seed cost under contention",
     "detail": "Summed chunk wall time across the 49 retained chunks = 303.1 h. Median wall per seed: T-0 8 s, "
               "T-a 1 s, T-b 1172 s, T-c 1007-1128 s (25-seed chunks at 25 cores => ~1 seed/core, so wall/seed here is a "
               "contended per-seed CPU proxy of ~7-8 CPU-h vs the 3.79 CPU-h uncontended measurement). The registered "
               "abort-(a) evidence is the UNCONTENDED 3.79 CPU-h/seed figure recorded in §11."},
    {"id": "D-VT-12", "class": "scope",
     "item": "No production posterior exists in this read",
     "detail": "Prereg §0: every posterior here is a synthetic-universe diagnostic quotable only against its own truth. "
               "The read is conditional (fixed-design) frequentist coverage over noise + ball + sigma_z randomness at "
               "fixed truth (VT-D1), on the 982 nonempty-ball events (VT-D5)."},
]

# ---------------- formulation awaiting the author's ruling ----------------
formulation = {
    "status": "AWAITING AUTHOR RULING — nothing below is adjudicated here",
    "1D_rail_account": {
        "question": "starvation vs the sigma_z-dosed co-candidate: which owns the production 1D behaviour?",
        "what_this_read_settles": "Under the branch the tree fires (TRANSFER-CONFIRMED), the sigma_z-dosed coverage "
            "DEFECT is no longer confined to the v2 caricature: it survives the real detected event population, the "
            "real per-event ball multiplicities (SigmaK = 1,193,703 vs the caricature's ~7,500) and the real "
            "heterogeneous GLADE per-galaxy sigma_z including the spec-z tail, at all three truths and in both channels.",
        "what_this_read_does_NOT_settle": "This venue produced ZERO rails (0/1400 seeds, both channels, every cell). "
            "The production 1D railing SHAPE (A-1D starvation rail 400/400, R2) is therefore NOT reproduced here and "
            "is NOT explained by this mechanism. The two accounts remain compatible-not-competing exactly as R3 "
            "registered them: starvation owns the railing shape; the dosed coverage collapse is the candidate for what "
            "the estimator does underneath (a uniform +~sigma_z MAP displacement with delta-narrow posteriors).",
        "pre_named_pattern_check": "RAIL-EMERGENT (any rail fraction >= 0.90 in a decision cell) did NOT fire.",
    },
    "paper_47_hold_reason": {
        "current": "R6: 'P-P leg FAILED - coverage DEFECT; fix routes through /physics-change'",
        "mechanical_consequence_of_the_fired_branch": "Under TRANSFER-CONFIRMED the prereg pre-states: paper #47's hold "
            "reason STANDS as upgraded by R6, with the transfer leg now EVALUATED-CONFIRMED (it was a NOT-EVALUABLE row "
            "before this campaign). Nothing here lifts or weakens the hold.",
        "author_decision_required": "whether the hold wording is amended to cite the venue-transfer evidence, and "
            "whether the DEFECT is quotable as a production-mechanism CANDIDATE (prereg wording) or something stronger.",
    },
    "physics_change_intake_PREPARED_NOT_OPENED": {
        "gate": "AUTHOR-GATED. Prereg §0 and branch 2: a TRANSFER-CONFIRMED escalation routes through /physics-change "
            "intake on the estimator's photo-z handling. This readout PREPARES the intake; it does NOT open it and "
            "touches no production physics file.",
        "target_of_record": "the estimator's photo-z handling — the per-candidate redshift-kernel treatment in the "
            "H0 likelihood (production analogue of the mirror's bare N(z; z_obs,k, sigma_z,k) x distance-likelihood "
            "form, prereg §4 step 5).",
        "prepared_intake_skeleton": {
            "old_form": "bare per-candidate Gaussian redshift kernel N(z; z_obs,k, sigma_z,k) multiplying the distance "
                "likelihood, equal candidate prior, GL-50 on the +-5 sigma_k clip (prereg §4.5, verbatim gate math)",
            "measured_symptom": "uniform positive MAP displacement of magnitude ~ +1 x sigma_bar (R_dose 0.88-0.98 "
                "across truths and channels) with delta-narrow posteriors (bias/post_sd ~ 8.5-10) and 0/400 HPD "
                "coverage at every level - i.e. the estimator is confidently wrong by about one photo-z sigma.",
            "candidate_axes_named_by_the_prereg_but_NOT_measured_here": [
                "kernel FORM: volume_deconv (production's resolved kernel per m1_kernel_delta_check.json) - "
                "O2 arm reserved (+47000), NOT BUILT, §9 row 2",
                "per-galaxy rate weights R_eff(M_g)/(1+z_g) - W1 arm reserved (+46000), NOT BUILT, §9 row 3",
                "completeness / out-of-catalogue term for the 606 empty-ball events - §9 row 4",
            ],
            "required_before_any_edit": "the full /physics-change protocol: derivation, dimensional analysis, "
                "limiting-case check, literature reference, regression test, PHYSICS-GATE-LEDGER row.",
            "state": "PREPARED, NOT OPENED - awaiting the author's explicit order.",
        },
    },
    "immediate_author_calls_this_readout_asks_for": [
        "ratify or reject the three §11 deviation notes (D-VT-1/2/3)",
        "rule on the branch the tree fired (the readout does not adjudicate)",
        "decide whether to order the reserved W1 (rate weights) and/or O2 (volume_deconv) arms",
        "decide whether to open the /physics-change intake on the estimator's photo-z handling",
    ],
}

out = {
    "readout": "VENUE-TRANSFER READ — mechanical scoring against the registered prereg",
    "scorer": "results/venue_transfer_20260811/score_venue_transfer.py",
    "prereg": "results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md (registered e77eecad)",
    "instrument_commit": "2ece8801",
    "inputs": {"n_chunk_files": n_chunk_files, "n_seeds": len(all_seeds),
               "collect_raw_cross_check": "results/venue_transfer_20260811/collect_raw.json "
                                          "(independent extraction; this scorer re-derives from the chunk JSONs)"},
    "validity": {"V_T_checks": vt, "abort_criteria": abort,
                 "edge_contamination_guard_sec8": edge_guard,
                 "VENUE_CONFOUNDED_trigger_set": trigger_set,
                 "VENUE_CONFOUNDED": venue_confounded},
    "provenance": provenance,
    "seed_plan": seed_plan,
    "raw_statistics": stats,
    "scored": scored,
    "DS_VT5_ladder": {"ladder": ladder, "killing_axis": killing_axis,
                      "T_a_vs_v2_raw_context": ladder_note_Ta},
    "branch": branch,
    "NOT_EVALUABLE_registry_sec9": not_evaluable,
    "disclosures": disclosures,
    "formulation_awaiting_author_ruling": formulation,
}
(HERE / "VENUE_TRANSFER_READOUT.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"branch_fired": fired, "venue_confounded": venue_confounded,
                  "seed_plan_pass": seed_plan["cross_cell"]["pass"],
                  "vt": {k: v["verdict"] for k, v in vt.items()},
                  "tc_1d": tc_1d, "tc_2d": tc_2d, "killing_axis": killing_axis,
                  "classifications": {c: {ch: scored[c][ch]["classification"] for ch in ("1d", "2d")}
                                      for c in CELLS}}, indent=1))
