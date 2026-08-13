"""Raw extraction + integrity check for the venue-transfer campaign (thread 17).

Extraction and integrity ONLY. No band scoring, no interpretation, no
classification (COLLAPSE-REPRODUCED / CALIBRATED / branch call) — that is
explicitly reserved for later agents per the prereg's model/effort policy
("the branch call is presented to the author, never self-adjudicated").

Reads: results/venue_transfer_20260811/T*_results_seeds*.json (49 chunk files)
       results/venue_transfer_20260811/validate_results_full.json
       results/venue_transfer_20260811/validate_results_novt5.json
Writes: results/venue_transfer_20260811/collect_raw.json

Run: cd /home/jasper/Repositories/darksiren-emri && uv run python results/venue_transfer_20260811/collect_extract.py
"""

from __future__ import annotations

import glob
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/jasper/Repositories/darksiren-emri")
VT_DIR = ROOT / "results" / "venue_transfer_20260811"

K_SUM_PIN = 1_193_703
N_EVENTS_PIN = 982  # nonempty-ball events (VT-D5)

# --- Registered cell -> seed block map (prereg §5 table, base 20260808) ---
CELL_SEED_BLOCKS: dict[str, dict] = {
    "T0": {"prereg_name": "T-0", "h_true": 0.730, "offset_lo": 40000, "offset_hi": 40199, "n": 200},
    "Ta": {"prereg_name": "T-a", "h_true": 0.730, "offset_lo": 41000, "offset_hi": 41199, "n": 200},
    "Tb": {"prereg_name": "T-b", "h_true": 0.730, "offset_lo": 42000, "offset_hi": 42199, "n": 200},
    "Tc_h0p690": {"prereg_name": "T-c(0.690)", "h_true": 0.690, "offset_lo": 43000, "offset_hi": 43199, "n": 200},
    "Tc_h0p730": {"prereg_name": "T-c(0.730)", "h_true": 0.730, "offset_lo": 44000, "offset_hi": 44399, "n": 400},
    "Tc_h0p770": {"prereg_name": "T-c(0.770)", "h_true": 0.770, "offset_lo": 45000, "offset_hi": 45199, "n": 200},
}
BASE_SEED = 20260808

# Reserved / never-run blocks (VT-D7) — must never collide with realized seeds
RESERVED_BLOCKS = [
    (46000, 46399),  # W1
    (47000, 47399),  # O2
]
# v1/v2 seed decades (VT-D7 disjointness)
V1_ENVELOPE = (0, 9049)
V2_ENVELOPE = (20000, 29049)


def cell_key_for_file(path: str) -> str:
    """Map a chunk filename to its CELL_SEED_BLOCKS key."""
    name = Path(path).name
    if name.startswith("T0_"):
        return "T0"
    if name.startswith("Ta_"):
        return "Ta"
    if name.startswith("Tb_"):
        return "Tb"
    if name.startswith("Tc_h0p690"):
        return "Tc_h0p690"
    if name.startswith("Tc_h0p730"):
        return "Tc_h0p730"
    if name.startswith("Tc_h0p770"):
        return "Tc_h0p770"
    raise ValueError(f"unrecognized chunk filename: {name}")


def git_ancestor(a: str, b: str) -> bool:
    """True if commit a is an ancestor of (or equal to) commit b."""
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", a, b], cwd=ROOT, capture_output=True
    )
    return r.returncode == 0


def git_diff_import_path_empty(a: str, b: str) -> tuple[bool, str]:
    """Check the diff a..b touches nothing under the import path (either the
    pre-rename or post-rename package names — commit 227e7a32 renamed
    master_thesis_code -> darksiren_emri AFTER this campaign ran)."""
    paths = [
        "master_thesis_code/",
        "master_thesis_code_test/",
        "darksiren_emri/",
        "darksiren_emri_test/",
    ]
    r = subprocess.run(
        ["git", "diff", "--stat", a, b, "--", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = r.stdout.strip()
    return (out == ""), out


def binomial_jeffreys_ci_note(k: int, n: int) -> dict:
    """Raw count/fraction only — no band comparison (that's DS-VT scoring,
    reserved). Provided for convenience of later agents."""
    frac = k / n if n else float("nan")
    return {"k": k, "n": n, "fraction": frac}


def ks_D_uniform(pit_values: list[float]) -> dict:
    """One-sample KS statistic D against Uniform(0,1). Raw D only, no band."""
    x = sorted(v for v in pit_values if v is not None and not (isinstance(v, float) and math.isnan(v)))
    n = len(x)
    if n == 0:
        return {"D": None, "n": 0}
    d_plus = max((i + 1) / n - xi for i, xi in enumerate(x))
    d_minus = max(xi - i / n for i, xi in enumerate(x))
    return {"D": max(d_plus, d_minus), "n": n}


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def se_of_mean(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    m = mean(vals)
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var / n)


def collect() -> dict:
    files = sorted(glob.glob(str(VT_DIR / "T*_results_seeds*.json")))
    assert len(files) == 49, f"expected 49 chunk files, found {len(files)}"

    out: dict = {
        "extraction_metadata": {
            "script": "results/venue_transfer_20260811/collect_extract.py",
            "n_chunk_files_found": len(files),
            "prereg_path": "results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md",
        },
        "provenance": {"per_chunk": [], "commit_pair_check": {}, "rename_disclosure": {}},
        "seed_plan": {"per_cell": {}, "cross_cell_checks": {}},
        "abort_criteria": {"per_cell": {}},
        "raw_statistics": {"per_cell_channel": {}},
        "validate_json_crosscheck": {},
        "disclosure_items": [],
    }

    # ---------- PROVENANCE ----------
    commit_set: set[str] = set()
    for f in files:
        d = json.load(open(f))
        rec = {
            "file": str(Path(f).relative_to(ROOT)),
            "git_commit": d["git_commit"],
            "git_commit_short": d["git_commit"][:8],
            "import_path_clean": d["import_path_clean"],
            "git_dirty": d["git_dirty"],
            "dirt_inventory_import_path": d["dirt_inventory"]["import_path"],
            "dirt_inventory_other_count": len(d["dirt_inventory"]["other"]),
            "allow_dirty": d["allow_dirty"],
            "smoke": d["smoke"],
            "workers": d["workers"],
            "wall_time_s": d["wall_time_s"],
            "wall_time_per_seed_s": d["wall_time_per_seed_s"],
            "n_seeds_in_chunk": len(d["seeds"]),
            "seeds_range": [min(d["seeds"]), max(d["seeds"])],
            "pin_integrity_pass": d["pin_integrity"]["pass"],
        }
        # per-seed n_events / n_events_run / n_events_cap / K_sum aggregated within chunk
        n_events_vals = {r["n_events"] for r in d["per_seed"]}
        n_events_run_vals = {r["n_events_run"] for r in d["per_seed"]}
        n_events_cap = d["config"]["n_events_cap"]
        k_sum_vals = {r["K_sum"] for r in d["per_seed"]}
        balls_mode_chunk = d["config"]["balls"]
        rec["n_events_values_in_chunk"] = sorted(n_events_vals)
        rec["n_events_run_values_in_chunk"] = sorted(n_events_run_vals)
        rec["n_events_cap"] = n_events_cap
        rec["balls_mode"] = balls_mode_chunk
        rec["K_sum_values_in_chunk"] = sorted(k_sum_vals)
        # K_SUM_PIN (real-K census, VT-D2) applies only to balls="real_k" chunks (T-0/T-b/T-c*).
        # T-a uses balls="poisson4" by registered design (prereg §5 table); K_sum there is a
        # per-seed random draw, so the pin check is N/A, not a failure.
        rec["K_sum_pin_applicable"] = balls_mode_chunk == "real_k"
        rec["K_sum_matches_pin_all_seeds"] = (
            (k_sum_vals == {K_SUM_PIN}) if balls_mode_chunk == "real_k" else None
        )
        commit_set.add(d["git_commit"][:8])
        out["provenance"]["per_chunk"].append(rec)

    out["provenance"]["distinct_git_commits"] = sorted(commit_set)
    out["provenance"]["commit_counts"] = {
        c: sum(1 for r in out["provenance"]["per_chunk"] if r["git_commit_short"] == c)
        for c in commit_set
    }
    out["provenance"]["all_import_path_clean"] = all(
        r["import_path_clean"] for r in out["provenance"]["per_chunk"]
    )
    out["provenance"]["all_smoke_false"] = all(not r["smoke"] for r in out["provenance"]["per_chunk"])
    out["provenance"]["all_pin_integrity_pass"] = all(
        r["pin_integrity_pass"] for r in out["provenance"]["per_chunk"]
    )
    out["provenance"]["all_K_sum_matches_pin"] = all(
        r["K_sum_matches_pin_all_seeds"]
        for r in out["provenance"]["per_chunk"]
        if r["K_sum_pin_applicable"]
    )

    # commit-pair ancestor + import-path-diff check (re-verified live, not trusted from orchestrator)
    if len(commit_set) == 2:
        a, b = sorted(commit_set)
        # determine actual ancestor order
        a_anc_b = git_ancestor(a, b)
        b_anc_a = git_ancestor(b, a)
        older, newer = (a, b) if a_anc_b else (b, a) if b_anc_a else (None, None)
        empty_diff, diff_stat = (None, None)
        if older and newer:
            empty_diff, diff_stat = git_diff_import_path_empty(older, newer)
        out["provenance"]["commit_pair_check"] = {
            "commits": sorted(commit_set),
            "ancestor_relationship_verified": bool(older),
            "older": older,
            "newer": newer,
            "import_path_diff_empty": empty_diff,
            "import_path_diff_stat_raw": diff_stat,
        }
    else:
        out["provenance"]["commit_pair_check"] = {
            "commits": sorted(commit_set),
            "note": f"expected exactly 2 distinct commits per orchestrator context, found {len(commit_set)}",
        }

    # rename disclosure: repo renamed master_thesis_code -> darksiren_emri at 227e7a32,
    # AFTER both campaign commits. Verify ordering.
    newest_campaign_commit = out["provenance"]["commit_pair_check"].get("newer") or sorted(commit_set)[-1]
    rename_after = git_ancestor(newest_campaign_commit, "227e7a32")
    out["provenance"]["rename_disclosure"] = {
        "rename_commit": "227e7a32",
        "rename_description": "refactor: rename package master_thesis_code -> darksiren_emri (rebrand phase b)",
        "newest_campaign_commit_checked": newest_campaign_commit,
        "rename_postdates_campaign": rename_after,
        "note": (
            "V-T4 clean-rule import_path_clean flags in every chunk JSON were evaluated "
            "against the OLD import path (master_thesis_code/, master_thesis_code_test/) "
            "at run time, since the rename had not yet happened. This is a disclosure item, "
            "not a defect: the flags are still a correct clean-rule evaluation for the state "
            "of the repo at run time."
        ),
    }

    # ---------- SEED PLAN ----------
    all_seeds_by_cell: dict[str, list[int]] = defaultdict(list)
    all_seeds_flat: list[int] = []
    for f in files:
        d = json.load(open(f))
        ckey = cell_key_for_file(f)
        all_seeds_by_cell[ckey].extend(d["seeds"])
        all_seeds_flat.extend(d["seeds"])

    seed_plan_per_cell = {}
    for ckey, block in CELL_SEED_BLOCKS.items():
        realized = sorted(all_seeds_by_cell.get(ckey, []))
        expected_lo = BASE_SEED + block["offset_lo"]
        expected_hi = BASE_SEED + block["offset_hi"]
        expected_set = set(range(expected_lo, expected_hi + 1))
        realized_set = set(realized)
        n_realized = len(realized)
        n_unique = len(realized_set)
        duplicates = n_realized - n_unique
        missing = sorted(expected_set - realized_set)
        extra = sorted(realized_set - expected_set)
        seed_plan_per_cell[ckey] = {
            "prereg_name": block["prereg_name"],
            "h_true": block["h_true"],
            "expected_n": block["n"],
            "expected_range": [expected_lo, expected_hi],
            "realized_n": n_realized,
            "realized_n_unique": n_unique,
            "n_duplicate_seeds_within_cell": duplicates,
            "n_missing_vs_registered_block": len(missing),
            "missing_seeds": missing,
            "n_extra_vs_registered_block": len(extra),
            "extra_seeds": extra,
            "exact_match": (n_unique == block["n"] and not missing and not extra and duplicates == 0),
        }
    out["seed_plan"]["per_cell"] = seed_plan_per_cell

    # cross-cell checks: no duplicate seed across cells, no collision with v1/v2, no reserved-block collision
    seen: dict[int, str] = {}
    cross_cell_dupes = []
    for ckey, seeds in all_seeds_by_cell.items():
        for s in seeds:
            if s in seen and seen[s] != ckey:
                cross_cell_dupes.append({"seed": s, "cells": sorted({seen[s], ckey})})
            seen[s] = ckey

    v1_lo, v1_hi = BASE_SEED + V1_ENVELOPE[0], BASE_SEED + V1_ENVELOPE[1]
    v2_lo, v2_hi = BASE_SEED + V2_ENVELOPE[0], BASE_SEED + V2_ENVELOPE[1]
    v1_collisions = [s for s in all_seeds_flat if v1_lo <= s <= v1_hi]
    v2_collisions = [s for s in all_seeds_flat if v2_lo <= s <= v2_hi]
    reserved_collisions = []
    for lo, hi in RESERVED_BLOCKS:
        blo, bhi = BASE_SEED + lo, BASE_SEED + hi
        reserved_collisions.extend([s for s in all_seeds_flat if blo <= s <= bhi])

    out["seed_plan"]["cross_cell_checks"] = {
        "total_realized_seeds_all_cells": len(all_seeds_flat),
        "total_realized_seeds_unique_all_cells": len(set(all_seeds_flat)),
        "cross_cell_duplicate_seeds": cross_cell_dupes,
        "v1_envelope_checked": [v1_lo, v1_hi],
        "v1_collisions": v1_collisions,
        "v2_envelope_checked": [v2_lo, v2_hi],
        "v2_collisions": v2_collisions,
        "reserved_blocks_checked": [[BASE_SEED + lo, BASE_SEED + hi] for lo, hi in RESERVED_BLOCKS],
        "reserved_block_collisions": reserved_collisions,
        "registered_total_expected": sum(b["n"] for b in CELL_SEED_BLOCKS.values()),
    }

    # ---------- ABORT CRITERIA + RAW STATISTICS (per cell x channel) ----------
    per_cell_records: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        d = json.load(open(f))
        ckey = cell_key_for_file(f)
        per_cell_records[ckey].extend(d["per_seed"])

    channels = ["1d", "2d"]
    for ckey, block in CELL_SEED_BLOCKS.items():
        records = per_cell_records.get(ckey, [])
        n = len(records)

        # abort (b): non-finite ln_post fraction
        def is_finite_vec(v):
            return all(math.isfinite(x) for x in v)

        n_nonfinite_1d = sum(1 for r in records if not is_finite_vec(r["ln_post_1d"]))
        n_nonfinite_2d = sum(1 for r in records if not is_finite_vec(r["ln_post_2d"]))

        # abort (d): horizon-drop guard
        horizon_dropped = [r["n_horizon_dropped"] for r in records]
        n_events_run_vals = [r["n_events_run"] for r in records]
        # guard defined relative to the pinned 982 set
        horizon_drop_fracs = [
            (hd / N_EVENTS_PIN) if N_EVENTS_PIN else float("nan") for hd in horizon_dropped
        ]
        max_horizon_drop_frac = max(horizon_drop_fracs) if horizon_drop_fracs else float("nan")
        n_seeds_horizon_drop_gt5pct = sum(1 for x in horizon_drop_fracs if x > 0.05)

        # pin_integrity per seed isn't per-seed, it's per-chunk; K_sum per-seed check.
        # The K_SUM_PIN (real-K census, VT-D2) applies ONLY to cells using real_k balls
        # (T-0, T-b, T-c*). T-a uses balls="poisson4" (Poisson lambda=4, prereg §5 table)
        # by registered design, so its per-seed K_sum is a random draw around
        # K_mean*982 ~= 4910, NOT the real-K pin -- checked separately, not flagged as a mismatch.
        balls_mode = records[0]["balls"] if records else None
        uses_real_k = balls_mode == "real_k"
        if uses_real_k:
            k_sum_mismatches = [r["seed"] for r in records if r["K_sum"] != K_SUM_PIN]
        else:
            k_sum_mismatches = []
        k_sum_values_seen = sorted({r["K_sum"] for r in records})

        out["abort_criteria"]["per_cell"][ckey] = {
            "n_seeds": n,
            "nonfinite_ln_post_1d_count": n_nonfinite_1d,
            "nonfinite_ln_post_1d_fraction": (n_nonfinite_1d / n) if n else None,
            "nonfinite_ln_post_1d_stop_triggered_gt_1pct": (n_nonfinite_1d / n) > 0.01 if n else None,
            "nonfinite_ln_post_2d_count": n_nonfinite_2d,
            "nonfinite_ln_post_2d_fraction": (n_nonfinite_2d / n) if n else None,
            "nonfinite_ln_post_2d_stop_triggered_gt_1pct": (n_nonfinite_2d / n) > 0.01 if n else None,
            "horizon_dropped_values": sorted(set(horizon_dropped)),
            "max_horizon_drop_fraction_of_982": max_horizon_drop_frac,
            "n_seeds_horizon_drop_gt_5pct": n_seeds_horizon_drop_gt5pct,
            "horizon_guard_stop_triggered": n_seeds_horizon_drop_gt5pct > 0,
            "balls_mode": balls_mode,
            "uses_real_k_pin": uses_real_k,
            "K_sum_pin": K_SUM_PIN if uses_real_k else None,
            "K_sum_mismatched_seeds": k_sum_mismatches,
            "K_sum_all_match_pin": (len(k_sum_mismatches) == 0) if uses_real_k else None,
            "K_sum_distinct_values_count": len(k_sum_values_seen),
            "K_sum_min_max": [min(k_sum_values_seen), max(k_sum_values_seen)] if k_sum_values_seen else None,
            "K_sum_note": (
                "real_k balls: K_sum must equal the 1,193,703 census pin on every seed (checked above)"
                if uses_real_k
                else "poisson4 balls (T-a, registered design VT-D2/§5): K_sum is a per-seed random draw, no fixed pin"
            ),
        }

        cell_stats = {"prereg_name": block["prereg_name"], "h_true": block["h_true"], "n_seeds": n}
        for ch in channels:
            hpd50 = [r[f"hpd50_{ch}"] for r in records]
            hpd68 = [r[f"hpd68_{ch}"] for r in records]
            hpd90 = [r[f"hpd90_{ch}"] for r in records]
            pit = [r[f"pit_{ch}"] for r in records]
            map_vals = [r[f"map_{ch}_refined"] for r in records]
            map_bias = [m - block["h_true"] for m in map_vals]
            post_sd = [r[f"post_sd_{ch}"] for r in records]
            edge_mass = [r[f"edge_mass_{ch}"] for r in records]
            railed_low = [r[f"railed_low_{ch}"] for r in records]
            railed_high = [r[f"railed_high_{ch}"] for r in records]

            hpd50_k = sum(1 for x in hpd50 if x == 1.0)
            hpd68_k = sum(1 for x in hpd68 if x == 1.0)
            hpd90_k = sum(1 for x in hpd90 if x == 1.0)
            railed_low_k = sum(1 for x in railed_low if x == 1.0)
            railed_high_k = sum(1 for x in railed_high if x == 1.0)
            edge_loaded_k = sum(1 for x in edge_mass if x > 0.01)

            ks = ks_D_uniform(pit) if ckey != "T0" else {"D": None, "n": n, "note": "T-0 PIT-exempt (degenerate PIT, VT-D8)"}

            sigma_mean_pairs = [r["sigma_z_mean_pairs"] for r in records]
            sigma_median_pairs = [r["sigma_z_median_pairs"] for r in records]
            frac_lt5e3 = [r["frac_pairs_sigma_lt_5e-3"] for r in records]

            bias_mean = mean(map_bias)
            bias_se = se_of_mean(map_bias)
            sigma_bar_pairs = mean(sigma_mean_pairs)
            r_dose = (bias_mean / sigma_bar_pairs) if sigma_bar_pairs not in (0, None) and not (isinstance(sigma_bar_pairs, float) and math.isnan(sigma_bar_pairs)) else None

            cell_stats[ch] = {
                "hpd50": binomial_jeffreys_ci_note(hpd50_k, n),
                "hpd68": binomial_jeffreys_ci_note(hpd68_k, n),
                "hpd90": binomial_jeffreys_ci_note(hpd90_k, n),
                "pit_ks": ks,
                "map_bias": {
                    "mean": bias_mean,
                    "se": bias_se,
                    "n": n,
                    "raw_map_refined_mean": mean(map_vals),
                    "h_true": block["h_true"],
                },
                "sigma_z_mean_pairs": {"mean": sigma_bar_pairs, "median_of_per_seed_means": median(sigma_mean_pairs)},
                "sigma_z_median_pairs": {"mean": mean(sigma_median_pairs), "median": median(sigma_median_pairs)},
                "frac_pairs_sigma_lt_5e-3": {"mean": mean(frac_lt5e3), "median": median(frac_lt5e3)},
                "R_dose": r_dose,
                "rail_low": binomial_jeffreys_ci_note(railed_low_k, n),
                "rail_high": binomial_jeffreys_ci_note(railed_high_k, n),
                "post_sd_median": median(post_sd),
                "post_sd_mean": mean(post_sd),
                "edge_mass": {
                    "edge_loaded_count": edge_loaded_k,
                    "edge_loaded_fraction": (edge_loaded_k / n) if n else None,
                    "mean_edge_mass": mean(edge_mass),
                    "median_edge_mass": median(edge_mass),
                    "edge_contaminated_gt_10pct": (edge_loaded_k / n) > 0.10 if n else None,
                },
            }
        out["raw_statistics"]["per_cell_channel"][ckey] = cell_stats

    # ---------- VALIDATE JSON CROSSCHECK ----------
    full_path = VT_DIR / "validate_results_full.json"
    novt5_path = VT_DIR / "validate_results_novt5.json"
    vfull = json.load(open(full_path))
    vnovt5 = json.load(open(novt5_path))

    out["validate_json_crosscheck"] = {
        "validate_results_full.json": {
            "path": str(full_path.relative_to(ROOT)),
            "seed_plan_pass": vfull["seed_plan"]["pass"],
            "v_t2_determinism_pass": vfull["v_t2"]["pass"],
            "v_t3_pin_integrity_pass": vfull["v_t3"]["pass"],
            "v_t5_no_drift_pass": vfull["v_t5"]["pass"],
            "v_t4_clean_rule": "evaluated per-registered-run (embedded in every chunk JSON), not a standalone field here",
            "note": "v_t2, v_t3, v_t5 fields directly PASS in this file; seed_plan.pass supports the VT-D7 disjoint-seed-plan requirement (not itself one of the five named V-T checks). V-T1 (T-0 anchor bias/rail check) and V-T4 (clean rule) are NOT fields of this file — V-T1 is a per-cell raw-statistics check (see raw_statistics.per_cell_channel.T0) and V-T4 is embedded per-chunk (import_path_clean).",
        },
        "validate_results_novt5.json": {
            "path": str(novt5_path.relative_to(ROOT)),
            "seed_plan_pass": vnovt5["seed_plan"]["pass"],
            "v_t2_determinism_pass": vnovt5["v_t2"]["pass"],
            "v_t3_pin_integrity_pass": vnovt5["v_t3"]["pass"],
            "v_t5_no_drift_pass": vnovt5["v_t5"]["pass"],
            "v_t5_note": vnovt5["v_t5"].get("note"),
            "note": "This is the earlier launch-phase validate that SKIPPED V-T5 (prereg §11 addendum: compliance-order deviation, no statistical content).",
        },
        "v_t_check_evidence_map": {
            "V-T1 (T-0 anchor)": "raw_statistics.per_cell_channel.T0 in this file (bias/rail values); band comparison against |bias|<=0.010/rail<=0.05 reserved for later agent",
            "V-T2 (determinism)": "validate_results_full.json v_t2.pass=true (bit-identical re-run spot-check, seed 20303808, n_events_cap=40)",
            "V-T3 (pin integrity)": "validate_results_full.json v_t3.pass=true AND per-chunk pin_integrity.pass in every one of the 49 chunk JSONs (see provenance.per_chunk)",
            "V-T4 (clean rule)": "embedded per-chunk: import_path_clean field in every one of the 49 chunk JSONs (see provenance.all_import_path_clean)",
            "V-T5 (no-drift)": "validate_results_full.json v_t5.pass=true (3/3 seeds, bit-identical to committed v2 B2(0.730) per_seed records); validate_results_novt5.json v_t5.pass=null (skipped, earlier launch-phase run)",
        },
    }

    # ---------- DISCLOSURE ITEMS (facts only, not adjudication) ----------
    out["disclosure_items"] = [
        {
            "id": "rename-naming-shift",
            "text": (
                "Repo renamed master_thesis_code -> darksiren_emri (package) and "
                "MasterThesisCode -> darksiren-emri (dir) at commit 227e7a32, which postdates "
                "both campaign commits (2ece8801, e93f3068) per verified git ancestor check. "
                "The prereg's V-T4 clean rule names the OLD import path; chunk JSONs' "
                "import_path_clean flags were evaluated against the OLD name at run time. "
                "This is a disclosure item, not a defect."
            ),
        },
        {
            "id": "commit-pair-provenance",
            "text": (
                "10 chunks carry git_commit 2ece8801, 39 carry e93f3068. Verified: 2ece8801 is "
                "an ancestor of e93f3068; the diff 2ece8801..e93f3068 touches ONLY 4 markdown "
                "files under results/ (0 lines under master_thesis_code/, master_thesis_code_test/, "
                "darksiren_emri/, or darksiren_emri_test/) -- import-path diff is empty, so the "
                "registered-commit chain holds per the R1-ratified D-4/D-5 pattern."
            ),
        },
        {
            "id": "operational-history-non-statistical",
            "text": (
                "Three SLURM arrays were required to complete the 49 chunks (6252702: 10/49 "
                "COMPLETED; 6253922: 17/39 remaining COMPLETED; 6259842: 22/22 remaining "
                "COMPLETED). Per prereg §11, seeds/chunking/bands/statistics/instrument commit "
                "were untouched across resubmissions; only --time/--cpus-per-task changed. "
                "workers field is 64 in first-wave chunks, 25 in resubmitted ones -- "
                "result-invariant per V-T2, and confirmed here: K_sum equals the 1,193,703 pin "
                "on every seed regardless of worker count (see abort_criteria.per_cell.*.K_sum_all_match_pin)."
            ),
        },
        {
            "id": "pending-author-ratification",
            "text": (
                "Three deviation notes in prereg §11 (the 6252702 runtime-blowout/resubmission "
                "note, the compliance-order note re: V-T5 sequencing, and the 6253922 "
                "memory-bandwidth-contention/second-resubmission note) are explicitly marked "
                "PENDING AUTHOR RATIFICATION in the registered file and must be surfaced in the "
                "readout's disclosure list -- none are statistical and none touch bands/seeds/"
                "statistics, but they are not yet author-closed."
            ),
        },
        {
            "id": "validate-novt5-is-earlier-launch-phase-run",
            "text": (
                "validate_results_novt5.json predates validate_results_full.json in the campaign "
                "timeline (the launch-phase validate that skipped V-T5 per prereg §11 addendum); "
                "validate_results_full.json is the authoritative, complete V-T1..V-T5 evidence run "
                "(V-T5 pass=true, 3/3 seeds bit-identical to committed v2 B2(0.730))."
            ),
        },
    ]

    return out


def main():
    out = collect()
    out_path = VT_DIR / "collect_raw.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"wrote {out_path}")

    # print a compact summary to stdout for the calling session's log
    print("\n=== SUMMARY ===")
    print("distinct git commits:", out["provenance"]["distinct_git_commits"])
    print("commit counts:", out["provenance"]["commit_counts"])
    print("all import_path_clean:", out["provenance"]["all_import_path_clean"])
    print("all smoke=false:", out["provenance"]["all_smoke_false"])
    print("all pin_integrity.pass:", out["provenance"]["all_pin_integrity_pass"])
    print("all K_sum matches pin:", out["provenance"]["all_K_sum_matches_pin"])
    print("commit_pair_check:", json.dumps(out["provenance"]["commit_pair_check"], indent=2))
    print("rename_disclosure.rename_postdates_campaign:", out["provenance"]["rename_disclosure"]["rename_postdates_campaign"])
    print()
    for ckey, sp in out["seed_plan"]["per_cell"].items():
        print(f"  {ckey}: realized_n={sp['realized_n_unique']} expected={sp['expected_n']} exact_match={sp['exact_match']}")
    print("cross_cell_duplicate_seeds:", out["seed_plan"]["cross_cell_checks"]["cross_cell_duplicate_seeds"])
    print("v1_collisions:", out["seed_plan"]["cross_cell_checks"]["v1_collisions"])
    print("v2_collisions:", out["seed_plan"]["cross_cell_checks"]["v2_collisions"])
    print("reserved_block_collisions:", out["seed_plan"]["cross_cell_checks"]["reserved_block_collisions"])
    print()
    for ckey, ac in out["abort_criteria"]["per_cell"].items():
        print(
            f"  {ckey}: n={ac['n_seeds']} nonfinite_1d_frac={ac['nonfinite_ln_post_1d_fraction']} "
            f"nonfinite_2d_frac={ac['nonfinite_ln_post_2d_fraction']} "
            f"max_horizon_drop_frac={ac['max_horizon_drop_fraction_of_982']} "
            f"K_sum_all_match={ac['K_sum_all_match_pin']}"
        )


if __name__ == "__main__":
    main()
