import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset")

# ---- covariate table (blind), 8 synthetic events ----
event_idx = list(range(8))
# S (iiib_2d) = {0,1,2}: constructed to have HIGH z_gw and be truth-hosted (C1) sometimes.
c1 = [True, False, False, True, False, False, True, False]           # in_catalog
c2 = [True, True, False, False, False, True, False, False]           # hosted_exact
c3 = [True, True, True, False, False, True, False, False]            # hosted_rel
c3c = [-1.0, -0.5, -2.0, -6.0, -7.0, -1.5, -8.0, -9.0]                # log10_f_cat
c4 = [0.90, 0.85, 0.80, 0.30, 0.25, 0.20, 0.15, 0.10]                 # z_gw -- S high, B low
c5 = [1.2, 1.1, 1.0, 0.5, 0.4, 0.6, 0.3, 0.2]                         # log10_sky_area
c6 = [0.9, 0.8, np.nan, 0.5, 0.6, 0.4, np.nan, 0.3]                   # mass_window_retention
c7 = [1.0, 0.9, 0.8, 0.3, 0.2, 0.4, 0.1, 0.05]                        # log10_n_cand_1d
c8 = [False, np.nan, np.nan, True, np.nan, np.nan, True, np.nan]      # cone_outside (C1 only)
c10 = [5.5, 5.4, 5.3, 5.0, 4.9, 5.1, 4.8, 4.7]                        # log10_M
c10b = [False] * 8                                                    # n<10 -> NOT-TESTED gate
c11 = [1.5, 1.4, 1.3, 0.9, 0.8, 1.0, 0.7, 0.6]                        # log10_snr, reported-only

df = pd.DataFrame(
    {
        "event_idx": event_idx,
        "C1": c1, "C2": c2, "C3": c3, "C3c": c3c, "C4": c4, "C5": c5,
        "C6": c6, "C7": c7, "C8": c8, "C10": c10, "C10b": c10b, "C11": c11,
    }
)
table_path = OUT / "SYNTH_covariate_table_blind.csv"
df.to_csv(table_path, index=False)
table_sha256 = hashlib.sha256(table_path.read_bytes()).hexdigest()
print("table sha256:", table_sha256)

# ---- influence vectors: 4 families' d_e/rank/in_S + primary logL_h* columns ----
h_grid = np.array([0.60, 0.665, 0.73, 0.795, 0.86])

# primary family iiib_2d: S = {0,1,2} (k=3). Rank by influence descending: 0>1>2>rest.
d_e_iiib2d = [0.02, 0.015, 0.01, 0.004, 0.003, 0.002, 0.001, 0.0005]
in_s_iiib2d = [True, True, True, False, False, False, False, False]

# replicate iiib_1d: k=3, SAME direction as primary for C4 (consistency)
d_e_iiib1d = [0.018, 0.013, 0.009, 0.005, 0.002, 0.001, 0.0008, 0.0003]
in_s_iiib1d = [True, True, True, False, False, False, False, False]

# replicate jr1_2d: k=3, also consistent
d_e_jr12d = [0.016, 0.011, 0.008, 0.004, 0.003, 0.0015, 0.0009, 0.0004]
in_s_jr12d = [True, True, True, False, False, False, False, False]

# replicate jr1_1d: k=2, only events 0,1 in S -- still same direction (subset)
d_e_jr11d = [0.02, 0.017, 0.003, 0.002, 0.0018, 0.001, 0.0007, 0.0002]
in_s_jr11d = [True, True, False, False, False, False, False, False]

infl = pd.DataFrame(
    {
        "event_idx": event_idx,
        "iiib_2d_d_e": d_e_iiib2d, "iiib_2d_rank": np.argsort(np.argsort(-np.array(d_e_iiib2d))) + 1,
        "iiib_2d_in_S": in_s_iiib2d,
        "iiib_1d_d_e": d_e_iiib1d, "iiib_1d_rank": np.argsort(np.argsort(-np.array(d_e_iiib1d))) + 1,
        "iiib_1d_in_S": in_s_iiib1d,
        "jr1_2d_d_e": d_e_jr12d, "jr1_2d_rank": np.argsort(np.argsort(-np.array(d_e_jr12d))) + 1,
        "jr1_2d_in_S": in_s_jr12d,
        "jr1_1d_d_e": d_e_jr11d, "jr1_1d_rank": np.argsort(np.argsort(-np.array(d_e_jr11d))) + 1,
        "jr1_1d_in_S": in_s_jr11d,
    }
)

# Primary-family (iiib_2d) per-event log-likelihood at each h-grid node, constructed so
# that events {0,1,2} (S, high z_gw) pull the posterior toward LOW h and the bulk toward
# h=0.73 -- i.e. removing the {0,1,2} stratum should move mean_h UP toward truth 0.73.
rng = np.random.default_rng(20260904)
logl = np.zeros((8, 5))
truth_idx = 2  # h=0.73
for e in range(8):
    if e in (0, 1, 2):
        # peaked at h=0.60 (index 0) -- these events pull the sample off-truth
        base = np.array([3.0, 1.0, -1.0, -3.0, -5.0])
    else:
        # peaked at h=0.73 (truth) -- the bulk
        base = np.array([-2.0, 1.0, 3.0, 1.0, -2.0])
    logl[e] = base + rng.normal(0, 0.05, size=5)
for i, col in enumerate(h_grid):
    infl[f"logL_h{col:.6f}"] = logl[:, i]

infl_path = OUT / "SYNTH_influence_vectors.csv"
infl.to_csv(infl_path, index=False)

print("wrote", table_path, infl_path)
print(json.dumps({"table_sha256": table_sha256}, indent=2))

# ===========================================================================
# FIX 2 (BUILD_RECORD_B3.md "FIX 2") -- deliberate exercises of DESIGN_GATE_
# formula.md Findings A, B, C, D. Each is a direct call into the fixed
# `offset_subset_reads` functions on hand-built synthetic data sized just
# large enough to isolate the bug the finding described (Exercise 1/2's
# n=8 table never reaches the materiality function's binary branch, the
# NOT-TESTED disposition branch, or the null-draw rail path at all -- that
# coverage gap is exactly what DESIGN_GATE_formula.md flagged). Every
# assertion below fails loudly (AssertionError) if the fix regresses.
# ===========================================================================
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offset_subset_reads as osr  # noqa: E402

fix2_record: dict = {}

# --- Finding A: binary materiality stratum must use the OR-enrichment
# direction (sep.effect >= 1.0), not the raw within-S majority. Built so the
# two rules DISAGREE: True is enriched in S relative to bulk (OR > 1) but is
# a MINORITY within S itself (s_bool.mean() < 0.5) -- exactly the C1
# in_catalog shape DESIGN_GATE_formula.md pointed at (~13% of S, OR=3).
n_total_a = 30
event_order_a = np.arange(n_total_a, dtype=np.float64)
covA = np.array([True, True, False, False, False, False, True, True] + [False] * 22)
# S = events 0-5 (k=6): True count = 2/6 = 0.333 (MINORITY within S).
# Bulk = events 6-29 (24): True count = 2/24 = 0.0833 -> OR > 1 (True enriched in S vs bulk).
table_a = pd.DataFrame({"event_idx": event_order_a, "C_A": covA}).set_index("event_idx", drop=False)
s_idx_a = pd.Index(np.arange(6, dtype=np.float64))
a_true_s, a_false_s = int(covA[:6].sum()), 6 - int(covA[:6].sum())
a_true_b, a_false_b = int(covA[6:].sum()), 24 - int(covA[6:].sum())
or_a, p_a = osr._binary_or(a_true_s, a_false_s, a_true_b, a_false_b)
assert or_a > 1.0, "exercise miswired: OR must be >1 (True enriched) for this test to be meaningful"
sep_a = osr.SeparationResult(
    covariate="C_A", kind="binary", n_s=6, n_b=24, n_nan=0,
    effect_name="OR", effect=or_a, p_raw=p_a, p_holm=0.02, holm_significant=True,
    band_pass=True, verdict="SEPARATES",
)
h_grid_ac = np.array([0.60, 0.665, 0.73, 0.795, 0.86])
rng_a = np.random.default_rng(11)
logl_a = np.zeros((n_total_a, 5)) + rng_a.normal(0, 0.01, size=(n_total_a, 5))
mat_a = osr.materiality_for_covariate(
    "C_A", sep_a, table_a, event_order_a, logl_a, h_grid_ac, s_idx_a,
    decile=0.10, t_mat=0.008, null_draws=200, null_seed=11,
)
assert mat_a is not None
n_true_total = int(covA.sum())  # 4 (2 in S + 2 in bulk)
n_false_total = n_total_a - n_true_total  # 26
# Old (buggy) majority-of-S rule would have picked enriched_level=False (since
# s_bool.mean()=0.333<0.5) and frozen the stratum at the 26 False rows. The
# fixed OR-direction rule must pick True (n_stratum == 4), the opposite.
assert mat_a.n_stratum == n_true_total, (mat_a.n_stratum, n_true_total, n_false_total)
assert "True" in mat_a.stratum_rule and str(round(or_a, 6)) in mat_a.stratum_rule or "OR=" in mat_a.stratum_rule
fix2_record["finding_A"] = {
    "or_a": or_a,
    "s_bool_mean_within_S": a_true_s / 6,
    "old_buggy_enriched_level_would_be": bool(a_true_s / 6 >= 0.5),
    "fixed_enriched_level": bool(or_a >= 1.0),
    "n_stratum_fixed": mat_a.n_stratum,
    "n_stratum_old_buggy_would_be": n_false_total,
    "stratum_rule": mat_a.stratum_rule,
}
print(
    f"FIX 2 / Finding A: OR={or_a:.4f}, s_bool.mean(S)={a_true_s / 6:.3f} (<0.5, minority) -> "
    f"fixed n_stratum={mat_a.n_stratum} (old buggy rule would give n_stratum={n_false_total})"
)

# --- Finding B: disposition_for must return INTERMEDIATE (not
# DIFFUSE-IN-COVARIATES) when nothing SEPARATES but C10b (or C8) is
# NOT-TESTED -- REGISTRATION_DRAFT.md §5's disposition table, verbatim:
# "C8 or C10b NOT-TESTED and no other covariate separates" -> INTERMEDIATE.
primary_b: dict[str, osr.SeparationResult] = {}
for cov in osr.HOLM_FAMILY:
    if cov == "C10b":
        primary_b[cov] = osr.SeparationResult(
            covariate=cov, kind="binary", n_s=0, n_b=0, n_nan=0, effect_name="OR",
            effect=float("nan"), p_raw=float("nan"), p_holm=None, holm_significant=None,
            band_pass=False, verdict="NOT-TESTED",
        )
    else:
        primary_b[cov] = osr.SeparationResult(
            covariate=cov, kind=osr.COVARIATE_TYPE[cov], n_s=6, n_b=24, n_nan=0,
            effect_name="OR" if osr.COVARIATE_TYPE[cov] == "binary" else "AUC",
            effect=1.0, p_raw=0.9, p_holm=1.0, holm_significant=False,
            band_pass=False, verdict="NULL",
        )
disp_b, named_b = osr.disposition_for(primary_b, {}, {})
assert (disp_b, named_b) == ("INTERMEDIATE", []), (disp_b, named_b)
fix2_record["finding_B"] = {"disposition": disp_b, "named_covariates": named_b}
print(f"FIX 2 / Finding B: all-NULL + C10b NOT-TESTED -> disposition = {disp_b!r} (was DIFFUSE-IN-COVARIATES pre-fix)")

# --- Finding C: NaN must be excluded from the decile-tail stratum (and the
# valid-n denominator used to size it), not swept to the "top" rank by
# na_option="bottom". 5 NaN rows + 15 real increasing values; old code would
# put the 2 NaN rows in the "top decile" (na_option="bottom" ranks NaN
# highest under an ascending rank), the fix must put the two HIGHEST REAL
# values (indices 18, 19) there instead.
n_total_c = 20
event_order_c = np.arange(n_total_c, dtype=np.float64)
covC = np.concatenate([np.full(5, np.nan), np.arange(15, dtype=np.float64)])
table_c = pd.DataFrame({"event_idx": event_order_c, "C_C": covC}).set_index("event_idx", drop=False)
s_idx_c = pd.Index(np.arange(4, dtype=np.float64))
sep_c = osr.SeparationResult(
    covariate="C_C", kind="continuous", n_s=4, n_b=11, n_nan=5, effect_name="AUC",
    effect=0.9, p_raw=0.01, p_holm=0.02, holm_significant=True, band_pass=True, verdict="SEPARATES",
)
logl_c = np.zeros((n_total_c, 5)) + np.random.default_rng(13).normal(0, 0.01, size=(n_total_c, 5))
mat_c = osr.materiality_for_covariate(
    "C_C", sep_c, table_c, event_order_c, logl_c, h_grid_ac, s_idx_c,
    decile=0.10, t_mat=0.008, null_draws=200, null_seed=13,
)
assert mat_c is not None
assert mat_c.n_missing == 5, mat_c.n_missing
valid_n_c = n_total_c - 5  # 15
expected_n_tail_c = max(1, round(valid_n_c * 0.10))  # round(1.5) -> 2
assert mat_c.n_stratum == expected_n_tail_c, (mat_c.n_stratum, expected_n_tail_c)
fix2_record["finding_C"] = {
    "n_missing": mat_c.n_missing,
    "valid_n": valid_n_c,
    "n_stratum": mat_c.n_stratum,
    "stratum_rule": mat_c.stratum_rule,
    "old_buggy_stratum_would_include_NaN_rows": True,
}
print(f"FIX 2 / Finding C: n_missing={mat_c.n_missing}, valid_n={valid_n_c}, "
      f"n_stratum={mat_c.n_stratum} (all real values, no NaN row swept in)")

# --- Finding D: null-draw MAP rail must be tracked and wired into a
# g-censoring gate red. 4 events, one (event 3) carries ALL of the
# information (peaked hard at h=0.73); the other three are flat/uninformative.
# n_stratum forced to 3 (decile=0.75) so a random size-3-of-4 draw excludes
# event 3 with probability 1/4 (not rail) and includes it with probability
# 3/4 (rail: the size-3 draw always contains it when n_stratum=3, and
# whenever the draw INCLUDES event 3, removing it leaves an uninformative,
# flat leave-out logpost that rails to the grid boundary) -> exactly a
# forced rail red (rail fraction ~=0.75 >= CENSORING_NULL_RAIL_RED_FRACTION).
n_total_d = 4
event_order_d = np.arange(n_total_d, dtype=np.float64)
covD = np.array([1.0, 2.0, 3.0, 4.0])
table_d = pd.DataFrame({"event_idx": event_order_d, "C_D": covD}).set_index("event_idx", drop=False)
s_idx_d = pd.Index(np.arange(2, dtype=np.float64))
h_grid_d = np.array([0.60, 0.665, 0.73, 0.795, 0.86])
logl_d = np.zeros((n_total_d, 5))
logl_d[3] = np.array([-100.0, -100.0, 100.0, -100.0, -100.0])  # event 3 alone carries all signal
sep_d = osr.SeparationResult(
    covariate="C_D", kind="continuous", n_s=2, n_b=2, n_nan=0, effect_name="AUC",
    effect=1.0, p_raw=0.01, p_holm=0.02, holm_significant=True, band_pass=True, verdict="SEPARATES",
)
mat_d = osr.materiality_for_covariate(
    "C_D", sep_d, table_d, event_order_d, logl_d, h_grid_d, s_idx_d,
    decile=0.75, t_mat=0.008, null_draws=400, null_seed=7,
)
assert mat_d is not None
assert mat_d.n_stratum == 3, mat_d.n_stratum
assert 0.65 <= mat_d.null_rail_fraction <= 0.85, mat_d.null_rail_fraction  # expected ~0.75
assert mat_d.censoring_gate_red is True, mat_d.censoring_gate_red
assert not mat_d.map_rail_full  # full-sample MAP interior (0.73), only the NULL draws rail
# Replicate build_report's wiring (offset_subset_reads.py build_report()) to confirm the
# gate actually reaches the disposition, not just the MaterialityResult field:
censoring_red_covariates = [cov for cov, m in {"C_D": mat_d}.items() if m.censoring_gate_red]
assert censoring_red_covariates == ["C_D"]
fix2_record["finding_D"] = {
    "n_stratum": mat_d.n_stratum,
    "null_rail_fraction": mat_d.null_rail_fraction,
    "censoring_gate_red": mat_d.censoring_gate_red,
    "map_rail_full": mat_d.map_rail_full,
    "would_set_instrument_note_in_build_report": bool(censoring_red_covariates),
}
print(f"FIX 2 / Finding D: null_rail_fraction={mat_d.null_rail_fraction:.3f}, "
      f"censoring_gate_red={mat_d.censoring_gate_red} (forced rail red)")

fix2_out_path = OUT / "SYNTH_fix2_output.json"
fix2_out_path.write_text(json.dumps(fix2_record, indent=2, default=str))
print("wrote", fix2_out_path)
print("FIX 2: all assertions passed (A, B, C, D)")

# ===========================================================================
# FIX 3 (BUILD_RECORD_B3.md "FIX 3") -- deliberate exercises of
# DESIGN_GATE_formula_rev2.md SSB (2D/1D disagreement), SSC (g-population join
# completeness), SSD (WEAK keyed to raw p instead of Holm-adjusted p), SSE
# (reported-only secondaries absent). Every assertion below fails loudly
# (AssertionError) if a fix regresses.
# ===========================================================================

fix3_record: dict = {}

# --- (1) SS5 INTERMEDIATE trigger: "primary 2D and 1D iiib families disagree
# in disposition". Build iiib_2d (as primary) with one covariate C1 SEPARATES
# + MATERIAL + replicate-consistent (-> raw disposition SUBSET-IDENTIFIED on
# its own), and iiib_1d (as an alternate primary, its own separation results)
# with NOTHING separating (-> DIFFUSE-IN-COVARIATES on its own). The two raw
# dispositions disagree, so build_report()'s wiring must downgrade the FINAL
# disposition to INTERMEDIATE even though iiib_2d alone would have banked
# SUBSET-IDENTIFIED.
sep_2d_c1 = osr.SeparationResult(
    covariate="C1", kind="binary", n_s=82, n_b=1506, n_nan=0, effect_name="OR",
    effect=5.0, p_raw=0.001, p_holm=0.01, holm_significant=True, band_pass=True,
    verdict="SEPARATES",
)
primary_2d = {"C1": sep_2d_c1}
mat_2d_c1 = osr.MaterialityResult(
    covariate="C1", stratum_rule="binary level == True", n_stratum=82,
    delta_strat=0.02, delta_s_oracle=0.03, captured_fraction=0.667,
    null_percentile=99.8, null_ci99=(-0.005, 0.006), material=True,
    map_rail_full=False, map_rail_stratum=False, null_rail_fraction=0.02,
    censoring_gate_red=False, n_missing=0,
)
materiality_2d = {"C1": mat_2d_c1}


def _replicate_sep_entry(verdict: str, effect: float) -> osr.SeparationResult:
    return osr.SeparationResult(
        covariate="C1", kind="binary", n_s=50, n_b=900, n_nan=0, effect_name="OR",
        effect=effect, p_raw=0.001, p_holm=0.01, holm_significant=True,
        band_pass=True, verdict=verdict,
    )


replicate_sep_2d = {
    "iiib_1d": {"C1": _replicate_sep_entry("SEPARATES", 4.0)},
    "jr1_2d": {"C1": _replicate_sep_entry("SEPARATES", 3.5)},
    "jr1_1d": {"C1": _replicate_sep_entry("NULL", 1.0)},
}
disposition_2d, named_2d = osr.disposition_for(primary_2d, materiality_2d, replicate_sep_2d)
assert disposition_2d == "SUBSET-IDENTIFIED", disposition_2d
assert named_2d == ["C1"], named_2d

primary_1d = {"C1": osr.SeparationResult(
    covariate="C1", kind="binary", n_s=94, n_b=1494, n_nan=0, effect_name="OR",
    effect=1.1, p_raw=0.8, p_holm=1.0, holm_significant=False, band_pass=False,
    verdict="NULL",
)}
disposition_1d, named_1d = osr.disposition_for(
    primary_1d, {}, {}, replicate_families=("iiib_2d", "jr1_2d", "jr1_1d"),
)
assert disposition_1d == "DIFFUSE-IN-COVARIATES", disposition_1d

# Replicate build_report()'s wiring exactly: raw disposition_2d is downgraded
# to INTERMEDIATE when the two whole-family dispositions disagree.
families_agree = disposition_2d == disposition_1d
final_disposition = disposition_2d if families_agree else "INTERMEDIATE"
assert families_agree is False
assert final_disposition == "INTERMEDIATE", final_disposition
fix3_record["finding_2d_1d_disagree"] = {
    "iiib_2d_raw_disposition": disposition_2d,
    "iiib_1d_disposition": disposition_1d,
    "agrees_with_primary": families_agree,
    "final_disposition": final_disposition,
}
print(
    f"FIX 3 / SS5 2D-vs-1D disagreement: iiib_2d raw={disposition_2d!r}, "
    f"iiib_1d={disposition_1d!r} -> final disposition={final_disposition!r} "
    "(would have banked SUBSET-IDENTIFIED without this trigger)"
)

# --- (2) SS6 g-population: an unmatched event_idx on each side must be
# disclosed (0 unmatched required) and routed into INSTRUMENT / NO-READ, not
# silently dropped by a pandas .intersection() join.
table_join = pd.DataFrame({"event_idx": [0, 1, 2, 3, 4], "C1": [True] * 5}).set_index(
    "event_idx", drop=False
)
infl_join = pd.DataFrame({"event_idx": [0, 1, 2, 3, 5], "iiib_2d_in_S": [True] * 5}).set_index(
    "event_idx", drop=False
)
join_info = osr.check_join_completeness(table_join, infl_join)
assert join_info["n_unmatched_table_only"] == 1, join_info  # event_idx 4
assert join_info["n_unmatched_influence_only"] == 1, join_info  # event_idx 5
assert join_info["unmatched_table_only_event_idx"] == [4], join_info
assert join_info["unmatched_influence_only_event_idx"] == [5], join_info
assert join_info["join_complete"] is False, join_info
# Replicate build_report()'s wiring: an incomplete join must produce a
# non-None instrument_note mentioning "g-population RED".
instrument_note_join = None
if not join_info["join_complete"]:
    join_note = (
        "g-population RED: table/influence join incomplete -- "
        f"{join_info['n_unmatched_table_only']} table row(s) with no influence match, "
        f"{join_info['n_unmatched_influence_only']} influence row(s) with no table "
        "match (0 unmatched required, SS6 g-population)"
    )
    instrument_note_join = join_note
assert instrument_note_join is not None and "g-population RED" in instrument_note_join
fix3_record["finding_g_population_join"] = {
    **join_info,
    "instrument_note": instrument_note_join,
}
print(
    f"FIX 3 / SS6 g-population: {join_info['n_unmatched_table_only']} unmatched table row(s), "
    f"{join_info['n_unmatched_influence_only']} unmatched influence row(s) -> "
    "routed to INSTRUMENT / NO-READ"
)

# --- (3) SS4.1 WEAK must key on Holm-adjusted significance, not raw p. m=10
# family, sorted raw p-values [0.001, 0.006, 0.011, 0.02, 0.03, 0.04, 0.3,
# 0.4, 0.5, 0.6] (DESIGN_GATE_formula_rev2.md SSD's own hand-check), every
# covariate's effect deliberately OUTSIDE the practical-null band pass
# (band_pass=False for all) to isolate the significance question. Holm at
# rank 2 (p_raw=0.006): p_holm = max(0.01, 9*0.006) = 0.054 >= alpha=0.05 ->
# holm_significant=False, even though p_raw=0.006 < alpha=0.05 -- the
# registered verdict is NULL, never WEAK.
holm_family_10 = ("C1", "C2", "C3", "C3c", "C4", "C5", "C6", "C7", "C8", "C10")
p_raws_sorted = [0.001, 0.006, 0.011, 0.02, 0.03, 0.04, 0.3, 0.4, 0.5, 0.6]
results_weak: dict[str, osr.SeparationResult] = {}
for cov, p_raw in zip(holm_family_10, p_raws_sorted, strict=True):
    kind = osr.COVARIATE_TYPE[cov]
    results_weak[cov] = osr.SeparationResult(
        covariate=cov, kind=kind, n_s=82, n_b=1506, n_nan=0,
        effect_name="OR" if kind == "binary" else "AUC",
        effect=1.5 if kind == "binary" else 0.55, p_raw=p_raw, p_holm=None,
        holm_significant=None, band_pass=False, verdict="NULL",
    )
osr.holm_correct(results_weak, alpha=0.05, auc_band=0.20, or_band=3.0)
r_rank0 = results_weak["C1"]  # p_raw=0.001 -> p_holm=0.01 -> holm_significant=True -> WEAK
r_rank1 = results_weak["C2"]  # p_raw=0.006 -> p_holm=0.054 -> holm_significant=False -> NULL
assert abs(r_rank0.p_holm - 0.01) < 1e-12, r_rank0.p_holm
assert r_rank0.holm_significant is True, r_rank0.holm_significant
assert r_rank0.verdict == "WEAK", r_rank0.verdict
assert abs(r_rank1.p_holm - 0.054) < 1e-12, r_rank1.p_holm
assert r_rank1.holm_significant is False, r_rank1.holm_significant
assert r_rank1.verdict == "NULL", (
    r_rank1.verdict,
    "raw p=0.006 < alpha=0.05 but Holm-adjusted p=0.054 >= alpha -> must be NULL, not WEAK",
)
fix3_record["finding_weak_holm"] = {
    "C1_p_raw": r_rank0.p_raw, "C1_p_holm": r_rank0.p_holm,
    "C1_holm_significant": r_rank0.holm_significant, "C1_verdict": r_rank0.verdict,
    "C2_p_raw": r_rank1.p_raw, "C2_p_holm": r_rank1.p_holm,
    "C2_holm_significant": r_rank1.holm_significant, "C2_verdict": r_rank1.verdict,
}
print(
    f"FIX 3 / SS4.1 WEAK-vs-Holm: C1 p_raw={r_rank0.p_raw} p_holm={r_rank0.p_holm:.3f} "
    f"holm_sig={r_rank0.holm_significant} -> {r_rank0.verdict}; "
    f"C2 p_raw={r_rank1.p_raw} p_holm={r_rank1.p_holm:.3f} "
    f"holm_sig={r_rank1.holm_significant} -> {r_rank1.verdict} (raw-significant, Holm-not)"
)

# --- (4) Reported-only secondaries: Spearman rho(d_e, continuous covariate),
# C1/C2/C3 class composition of S, C1-vs-C2/C3 truth-disagreement 2x2.
table_sec = pd.DataFrame(
    {
        "event_idx": [0, 1, 2, 3, 4],
        "C1": [True, True, False, False, False],
        "C2": [True, False, False, True, False],
        "C3": [False, False, True, True, False],
        "C4": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
).set_index("event_idx", drop=False)
infl_sec = pd.DataFrame(
    {"event_idx": [0, 1, 2, 3, 4], "iiib_2d_d_e": [5.0, 4.0, 3.0, 2.0, 1.0]}
).set_index("event_idx", drop=False)

spearman_out = osr.spearman_secondaries(table_sec, infl_sec, "iiib_2d")
# C4 is perfectly (anti-)monotonic with d_e by construction -> rho == -1.0 exactly.
assert abs(spearman_out["C4"]["rho"] - (-1.0)) < 1e-9, spearman_out["C4"]
assert spearman_out["C4"]["n"] == 5, spearman_out["C4"]

s_index_sec = pd.Index([0, 1])
class_comp_out = osr.class_composition_counts(table_sec, s_index_sec)
assert class_comp_out["C1"] == {"n_true": 2, "n_false": 0, "n_nan": 0}, class_comp_out["C1"]
assert class_comp_out["C2"] == {"n_true": 1, "n_false": 1, "n_nan": 0}, class_comp_out["C2"]
assert class_comp_out["C3"] == {"n_true": 0, "n_false": 2, "n_nan": 0}, class_comp_out["C3"]

truth_out = osr.truth_disagreement_tables(table_sec)
assert truth_out["C2"] == {
    "C1_true_and_cov_true": 1, "C1_true_and_cov_false": 1,
    "C1_false_and_cov_true": 1, "C1_false_and_cov_false": 2,
}, truth_out["C2"]
assert truth_out["C3"] == {
    "C1_true_and_cov_true": 0, "C1_true_and_cov_false": 2,
    "C1_false_and_cov_true": 2, "C1_false_and_cov_false": 1,
}, truth_out["C3"]

fix3_record["finding_secondaries"] = {
    "spearman_C4_rho": spearman_out["C4"]["rho"],
    "class_composition_S": class_comp_out,
    "truth_disagreement_2x2": truth_out,
}
print(
    f"FIX 3 / secondaries: spearman(d_e, C4).rho={spearman_out['C4']['rho']:.3f} (exact -1.0); "
    f"class_composition_S={class_comp_out}; truth_disagreement(C1 vs C2/C3)={truth_out}"
)

fix3_out_path = OUT / "SYNTH_fix3_output.json"
fix3_out_path.write_text(json.dumps(fix3_record, indent=2, default=str))
print("wrote", fix3_out_path)
print("FIX 3: all assertions passed (2D/1D disagreement, g-population join, WEAK-vs-Holm, secondaries)")

# ===========================================================================
# FIX 4 (BUILD_RECORD_B3.md "FIX 4") -- PIN CORRECTIONS item 1 / DESIGN_GATE_
# formula_rev3.md finding 4.2: the real, BUILT covariate/influence files use a
# suffixed-id, per-venue schema, not the draft's bare-id combined-file one.
# This exercises `load_table`/`load_influence`/`detect_venue` (the schema
# adapter added in fix round 4) against a HAND-BUILT fixture using that REAL
# schema -- and separately confirms a missing registered column raises
# INSTRUMENT-DEFECT (never a silent skip), on both the covariate and the
# influence side. Every assertion below fails loudly if the fix regresses.
# ===========================================================================

fix4_record: dict = {}

# --- (1) Happy path: a small table/influence pair using the REAL suffixed
# schema (covariate_table_iiib.csv / influence_iiib.csv headers, confirmed via
# `head -1` on the committed files) loads cleanly through `load_table`/
# `load_influence`, maps onto the registered bare C1..C11 ids, and derives
# `{family}_in_S` from the top-k rank over `influence_2D`/`influence_1D` --
# never from a `_in_S` column, which the real files do not carry.
n4 = 10
event_idx4 = list(range(n4))
real_schema_table = pd.DataFrame(
    {
        "event_idx": event_idx4,
        "C1_in_catalog": [True, True, False, False, False, False, False, False, False, False],
        "C2_hosted_exact": [True] * 3 + [False] * 7,
        "C3_hosted_rel": [True] * 2 + [False] * 8,
        "C3c_log10_f_cat": [-0.5, -0.6, -0.7, -1.0, -1.2, -1.5, -2.0, -2.5, -3.0, -3.5],
        "C3c_censored": [False] * n4,
        "C4_z_gw": [float(i) for i in range(n4)],
        "C5_log10_sky_area": [0.1 * i for i in range(n4)],
        "C6_mass_window_retention": [0.9] * n4,
        "C7_log10_n_cand_1d": [1.0] * n4,
        "C8_cone_outside": [False, True] + [None] * 8,
        "C10_log10_M": [5.0 + 0.01 * i for i in range(n4)],
        "C10b_low_M_timeout_bins12": [False] * n4,
        "C11_log10_snr": [1.0] * n4,
    }
)
real_table_path = OUT / "SYNTH_real_schema_covariate_table_iiib.csv"
real_schema_table.to_csv(real_table_path, index=False)
real_table_sha256 = hashlib.sha256(real_table_path.read_bytes()).hexdigest()

# influence_2D/influence_1D ARE the directional d_e (BUILD_RECORD_B2.md), strictly
# decreasing here so top-k is unambiguous by construction; k=3 -> S={0,1,2}.
real_schema_influence = pd.DataFrame(
    {
        "event_idx": event_idx4,
        "influence_2D": [float(n4 - i) for i in range(n4)],
        "influence_1D": [float(n4 - i) * 0.9 for i in range(n4)],
        "rank": list(range(1, n4 + 1)),
    }
)
real_influence_path = OUT / "SYNTH_real_schema_influence_iiib.csv"
real_schema_influence.to_csv(real_influence_path, index=False)

loaded_table = osr.load_table(real_table_path)
assert list(loaded_table["C1"]) == list(real_schema_table["C1_in_catalog"]), "C1 mapping failed"
assert list(loaded_table["C4"]) == list(real_schema_table["C4_z_gw"]), "C4 mapping failed"
for bare, real in osr.COVARIATE_COLUMN_MAP.items():
    assert bare in loaded_table.columns, f"{bare} missing after schema mapping"

venue4 = osr.detect_venue(real_table_path, real_influence_path)
assert venue4 == "iiib", venue4
family_k4 = {"iiib_2d": 3, "iiib_1d": 3, "jr1_2d": 3, "jr1_1d": 3}
loaded_influence = osr.load_influence(real_influence_path, venue4, family_k4)[0]
s_2d = set(loaded_influence.index[loaded_influence[osr.family_in_s_col("iiib_2d")].astype(bool)].tolist())
assert s_2d == {0, 1, 2}, s_2d
s_1d = set(loaded_influence.index[loaded_influence[osr.family_in_s_col("iiib_1d")].astype(bool)].tolist())
assert s_1d == {0, 1, 2}, s_1d
# jr1 families were NOT requested for this venue -- their columns must not appear.
assert osr.family_in_s_col("jr1_2d") not in loaded_influence.columns
fix4_record["happy_path"] = {
    "venue": venue4,
    "iiib_2d_S": sorted(s_2d),
    "iiib_1d_S": sorted(s_1d),
    "table_columns_after_mapping": [c for c in osr.COVARIATE_COLUMN_MAP if c in loaded_table.columns],
}
print(f"FIX 4 / happy path: venue={venue4!r}, iiib_2d S={sorted(s_2d)}, iiib_1d S={sorted(s_1d)}, "
      "all 12 registered covariates present after schema mapping")

# --- (2) Missing registered covariate column -> INSTRUMENT-DEFECT, never a
# silent `continue` (this is the exact silent-DIFFUSE-IN-COVARIATES failure
# mode DESIGN_GATE_formula_rev3.md finding 4.2 named).
bad_table = real_schema_table.drop(columns=["C4_z_gw"])
bad_table_path = OUT / "SYNTH_bad_schema_covariate_table.csv"
bad_table.to_csv(bad_table_path, index=False)
try:
    osr.load_table(bad_table_path)
    raise AssertionError("load_table must raise InstrumentDefectError on a missing registered column")
except osr.InstrumentDefectError as exc:
    assert "C4" in exc.message and "C4_z_gw" in exc.message, exc.message
    assert exc.detail["missing_covariate_columns"] == [{"covariate": "C4", "expected_column": "C4_z_gw"}], exc.detail
    fix4_record["missing_covariate_column"] = {"message": exc.message, "detail": exc.detail}
    print(f"FIX 4 / missing covariate column: InstrumentDefectError raised as required: {exc.message}")

# --- (3) Missing influence base column -> INSTRUMENT-DEFECT, never a silent skip.
bad_influence = real_schema_influence.drop(columns=["influence_1D"])
bad_influence_path = OUT / "SYNTH_bad_schema_influence.csv"
bad_influence.to_csv(bad_influence_path, index=False)
try:
    osr.load_influence(bad_influence_path, "iiib", family_k4)
    raise AssertionError("load_influence must raise InstrumentDefectError on a missing influence column")
except osr.InstrumentDefectError as exc:
    assert exc.detail["missing_influence_columns"] == ["influence_1D"], exc.detail
    fix4_record["missing_influence_column"] = {"message": exc.message, "detail": exc.detail}
    print(f"FIX 4 / missing influence column: InstrumentDefectError raised as required: {exc.message}")

# --- (4) Ambiguous / mismatched venue (e.g. an iiib table paired with a
# joint_r1 influence file) -> INSTRUMENT-DEFECT, never a silent guess.
try:
    osr.detect_venue(Path("covariate_table_iiib.csv"), Path("influence_joint_r1.csv"))
    raise AssertionError("detect_venue must raise InstrumentDefectError on a venue mismatch")
except osr.InstrumentDefectError as exc:
    fix4_record["venue_mismatch"] = {"message": exc.message}
    print(f"FIX 4 / venue mismatch: InstrumentDefectError raised as required: {exc.message}")

# --- (5) `--dry-run` on the REAL, committed per-venue files never writes an
# output file and never touches a registered aggregate -- confirmed via the
# CLI itself (subprocess), mirroring the launch-block invocation, for both
# venues; asserts exit 0 and the exact "1588/1588 joined" row counts.
import subprocess  # noqa: E402

covariate_sha = {}
for line in (OUT / "covariate_table.sha256").read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    covariate_sha[name.strip()] = digest

for venue_name, table_name, influence_name, expected_k in (
    ("iiib", "covariate_table_iiib.csv", "influence_iiib.csv", {"iiib_2d": 82, "iiib_1d": 94}),
    ("joint_r1", "covariate_table_joint_r1.csv", "influence_joint_r1.csv", {"jr1_2d": 72, "jr1_1d": 46}),
):
    out_path = OUT / f"SYNTH_should_not_exist_{venue_name}.json"
    if out_path.exists():
        out_path.unlink()
    result = subprocess.run(
        [
            sys.executable, str(Path(__file__).resolve().parent / "offset_subset_reads.py"),
            "--table", str(OUT / table_name), "--table-sha256", covariate_sha[table_name],
            "--influence", str(OUT / influence_name), "--out", str(out_path), "--dry-run",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (venue_name, result.returncode, result.stdout, result.stderr)
    assert "1588 table rows / 1588 influence rows joined" in result.stdout, result.stdout
    assert "unmatched table-only=0, unmatched influence-only=0; join_complete=True" in result.stdout, result.stdout
    for fam, k in expected_k.items():
        assert f"family {fam}: k={k}" in result.stdout, (fam, k, result.stdout)
    assert not out_path.exists(), f"--dry-run must never write {out_path}"
    fix4_record[f"dry_run_{venue_name}"] = {"returncode": result.returncode, "stdout": result.stdout}
    print(f"FIX 4 / --dry-run ({venue_name}, REAL inputs): exit 0, 1588/1588 joined, k={expected_k}, no file written")

fix4_out_path = OUT / "SYNTH_fix4_output.json"
fix4_out_path.write_text(json.dumps(fix4_record, indent=2, default=str))
print("wrote", fix4_out_path)
print("FIX 4: all assertions passed (real-schema mapping, missing-column INSTRUMENT-DEFECT x2, venue mismatch, real-input --dry-run x2)")
