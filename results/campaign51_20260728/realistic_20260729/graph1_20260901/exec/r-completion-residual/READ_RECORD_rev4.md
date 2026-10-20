# READ_RECORD_rev4.md — r-completion-residual, REGISTRATION_DRAFT.md §7 launch, REAL mode

Disjoint reader for m-completion-residual, launched under docket 2.2 after the GREEN gate
`DESIGN_GATE_rev4_computability.md`. This record is **VERDICT-FREE**: it reports the registered read's
outputs and the disposition table's mechanical trigger evaluation only — no ruling, promotion, or
recommendation. Every number below is machine-copied from the script's own JSON output; none is
independently recomputed by this reader.

## 0. Command executed (from repo root, exactly once)

Launch block taken verbatim from `REGISTRATION_DRAFT.md` §7, with the `--out` path substituted per the
orchestrator's launch instruction (`completion_result_rev4_read.json` in place of the draft's
`completion_residual_result.json`); `[--dry-run]` omitted (REAL mode). No other flag, value, or ordering
changed. The script itself was not modified.

```
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_residual_reads.py \
  --production-csv results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv \
  --production-crb results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
  --replicate-csv results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv \
  --harness-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip \
  --population 200 --h-lo 0.725 --h-hi 0.735 --h-true 0.73 \
  --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
  --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_result_rev4_read.json
```

**Working directory:** `/home/jasper/Repositories/darksiren-emri` (repo root). **Invocation count:** 1
(single run, no retries). **Exit code: 0.** stdout: `wrote results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_result_rev4_read.json`
(no stderr). **`mode` field in output JSON:** `"real"`.

**Pre-flight existence contract** (checked before launch, all inputs present):

| input | status |
|---|---|
| `--production-csv` | EXISTS |
| `--production-crb` | EXISTS; md5 `9a1f2a14384a9281c97ca3be312ddaab` — matches `--crb-md5` and the pinned value in `DESIGN_GATE_rev4_computability.md` §1 |
| `--replicate-csv` | EXISTS |
| `--harness-root` | EXISTS; `ls | grep -c universe_seed.*_S.json` = 67 |
| `completion_residual_reads.py` (unmodified, not edited by this reader) | EXISTS |

## 1. NO-READ status

`NO_READ` (top level): **False**. `gates.NO_READ.no_read`: **False**. `gates.NO_READ.triggers`: **[]** (empty
list — no trigger fired). `gates.gates_green`: **True**. The read was banked; no NO-READ occurred, so
nothing further is reported under that branch.

## 2. Gate results — both venues

### 2.1 g-population

| field | production | replicate |
|---|---|---|
| n_h_nodes | 41 | 41 |
| rows_per_h_node (uniform value) | 1588 | 1588 |
| rows_per_h_uniform | True | True |
| n_rows_total | 65108 | 65108 |
| n_crb_rows | 1590 | 1590 |
| missing_event_idx | [1203, 1356] | [1203, 1356] |
| join_gate_green | True | True |
| n_in_catalogue_scored | 76 | 76 |
| n_dark_scored | 1512 | 1512 |
| in_catalogue_matches_expected | True | True |
| dark_matches_expected | True | True |

Both venues identical on every field. No diff between production and replicate populations.

### 2.2 g-znorm

| field | production | replicate |
|---|---|---|
| all_h_nodes_uniform | True | True |
| n_h_nodes_checked | 41 | 41 |
| max_nunique | 1 | 1 |

### 2.3 g-closure (per-event identity)

| field | production | replicate |
|---|---|---|
| n_events | 1588 | 1588 |
| max_closure_residual | 2.5579538487363607e-13 | 9.836575998178887e-14 |
| n_violations | 0 | 0 |
| gclosure_green | True | True |

### 2.4 g-closure (class closure, S_all = π_G·S_G + π_Ḡ·S_dark)

| field | production | replicate |
|---|---|---|
| n_total | 1588 | 1588 |
| n_dark | 1512 | 1512 |
| n_catalogue | 76 | 76 |
| pi_G | 0.04785894206549118 | 0.04785894206549118 |
| pi_Gbar | 0.9521410579345088 | 0.9521410579345088 |
| S_G | 1.207935464057304 | 1.0502771840482537 |
| S_dark | -0.11420262006209346 | -0.1064993120995181 |
| S_all | -0.050926490091643725 | -0.05113721278765999 |
| reconstructed_S_all | -0.05092649009164371 | -0.05113721278766 |
| class_closure_residual | 1.3877787807814457e-17 | 1.3877787807814457e-17 |
| class_closure_green | True | True |

### 2.5 g-byte-id (instrument, harness)

- `n_checkpoint_files_globbed`: 67; `n_checkpoints_matched_population`: 67; `n_checkpoints_expected`: 67;
  `byte_id_count_green`: **True**.
- `n_dark_means_present`: 67; `seed_min`/`seed_max`: 901000 / 901066.
- `mean_of_dark_full_score_means` (informational, all 67 checkpoint `score_at_truth.no_bh.dark.mean`
  values averaged): **0.008215870005381617**; `sem_of_dark_full_score_means`: **0.006314188695650197**
  (these two reproduce `T_full_harn_informational` / `SE_full_harn_informational` at top level exactly).
- `resolved_flags_internally_consistent`: True; `n_distinct_resolved_flags_blocks`: 1.

### 2.6 g-harness-universes (per-universe g-closure + g-znorm, all 67)

`n_universes_checked`: 67; `n_universes_expected`: 67; `count_matches_expected`: **True**;
`all_universes_gclosure_gznorm_green`: **True**. Independently re-checked by this reader against the
`per_universe` array: 0 of 67 entries have `universe_green: false`; max per-universe `max_closure_residual`
across all 67 = **2.1749269052406817e-13**. Sample (seed 901000): `gclosure_green: True`,
`max_closure_residual: 1.1546319456101628e-14`, `gznorm_green: True`, `universe_green: True`.

### 2.7 g-resolved-flags (harness ↔ production equality, FIX 4/F1)

`flag_names_compared` (13 tokens): `normalization_mode, catalogue_global_selection,
selection_in_completion_numerator, catalogue_numerator_survival, catalogue_numerator_survival_2d,
mass_filter_sigma, mass_filter_geometry, mass_filter_k, theta_b, theta_s, theta_sites,
theta_phi_divisor, theta_zwindow`.

`n_checkpoints_matched_population`: 67; `n_checkpoints_mismatched`: **0**;
`resolved_flags_equality_green`: **True**; `differing_keys`: **[]** (empty — no diffs to report verbatim).
Independently re-checked by this reader against the 67-entry `per_checkpoint` array: 0 of 67 have
`match: false`; every `diffs` object is `{}`.

`production_registered_flags` (the CoR-P CLI values every checkpoint was compared against):
```
normalization_mode: absolute_marginal
catalogue_global_selection: phi
selection_in_completion_numerator: fused
catalogue_numerator_survival: phi
catalogue_numerator_survival_2d: mz_sel
mass_filter_sigma: symmetric
mass_filter_geometry: linear
mass_filter_k: 1.5
theta_b: 0.0
theta_s: 1.0
theta_sites: all
theta_phi_divisor: off
theta_zwindow: off
```

### 2.8 g-rail-fraction-disclosure (disclosure-only, `disposition_role: None`)

`disclosure_threshold`: 0.1.

| channel | n_checkpoints | n_at_rail | rail_fraction | above_disclosure_threshold |
|---|---|---|---|---|
| no_bh | 67 | 10 | 0.14925373134328357 (14.9%) | True |
| with_bh | 67 | 14 | 0.208955223880597 (20.9%) | True |

`production_map_source.available`: **False** — script's own note: "REGISTRATION_DRAFT.md §5 names no
production/replicate map_h source for this script; the draft's 'production MAP (0.665, interior)' figure
is quoted prose, not a file/column this read consumes — disclosed as unavailable rather than hardcoded
from that prose figure." `disposition_role`: `None` (confirms this gate is disclosure-only, consistent
with `DESIGN_GATE_rev4_computability.md` §3's finding that it is excluded from `gates_green`/`triggers`).

### 2.9 g-precision (disclosure-only, non-gating, FIX 4/F4)

| h-node | venue | source | full_precision_value | csv_derived_value | relative_diff | within_tolerance (1e-3) |
|---|---|---|---|---|---|---|
| 0.725 | production | `seed901013_S/selection_tables_h_0_72.json` | 898273432.2933638 | 893324910.0 | 0.005508926475460637 | **False** |
| 0.735 | production | `seed901013_S/selection_tables_h_0_73.json` | 888403798.0710543 | 883510540.0 | 0.005507921152159435 | **False** |
| 0.725 | replicate | `seed901013_S/selection_tables_h_0_72.json` | 898273432.2933638 | 893324850.0 | 0.005508993270267032 | **False** |
| 0.735 | replicate | `seed901013_S/selection_tables_h_0_73.json` | 888403798.0710543 | 883510550.0 | 0.005507909896016652 | **False** |

`any_full_precision_source_found`: True (both venues). Not in `gates_green`/`triggers` (confirmed absent
from the `NO_READ.triggers` list, which is empty).

### 2.10 t0-mean-h anchor

`computed`: **0.6669869414473403**; `target_display_precision` (6-dp anchor): 0.666987;
`computed_rounded_to_6dp`: 0.666987; `abs_diff`: 5.855265972076751e-08; `tolerance`: 1e-09;
`reproduces_to_tolerance`: **True**; `reproduction_basis`: "round(computed, 6) == displayed anchor (source
carries no finer precision)"; `n_h_grid`: 41.

### 2.11 Catalogue md5 of record

`expected`: `c52c13b5cab61f6b3f04bbe202550969` — script's own note: not independently re-hashed by this
script (it consumes only the CSVs the catalogue already fed into); recorded for provenance per the
CLAUDE.md dataset-pinning convention.

### 2.12 Anchors block (as reported)

`production_rows`: 65108; `production_n_h_nodes`: 41; `crb_rows`: 1590; `n_dark_scored`: 1512;
`n_in_catalogue_scored`: 76; `harness_checkpoints_matched`: 67; `h_stencil`: [0.725, 0.735]; `h_true`: 0.73.

## 3. Every registered intermediate (§2.4 of the draft)

| symbol | value | source field |
|---|---|---|
| N_dark_prod (per-class n, dark) | 1512 | `N_dark_prod` |
| n_catalogue (per-class n) | 76 | `class_closure.n_catalogue` |
| T_prod | **-0.19663662072454366** | `T_prod` |
| SE_prod | **0.01943993234568931** | `SE_prod` |
| Z_prod | **-10.115087708530357** | `Z_prod` |
| n_universes_harn | 67 | `n_universes_harn` |
| T_harn | **-0.05054134388858336** | `T_harn` |
| SE_harn | **0.007321725821419925** | `SE_harn` |
| Z_harn | **-6.9029276869017915** | `Z_harn` |
| T_full_harn_informational (harness FULL-score, informational, NOT Z_harn's input) | 0.008215870005381617 | `T_full_harn_informational` |
| SE_full_harn_informational | 0.006314188695650197 | `SE_full_harn_informational` |
| ρ (T_harn/T_prod, evaluated since \|Z_prod\|>3) | **0.25702915205903415** | `rho` |
| δh_M (REPORTED-ONLY, verdict_bearing: False) | -0.09132334447033695 | `delta_h_M.delta_h_M` |
| δh_M inputs: N_Ḡ | 1512 | `delta_h_M.N_Gbar` |
| δh_M inputs: σ_h,1D | 0.017526 | `delta_h_M.sigma_h_1D` |
| δh_M inputs: I_1D | 3255.6250787779877 | `delta_h_M.I_1D` |

### 3.1 Per-universe S_M summary (67 harness universes, matched-channel)

Computed by this reader directly from the script's own `harness_matched_channel_detail.per_universe`
array (67 entries, `S_M_universe` field), not recomputed independently:

- n = 67; seeds 901000–901066 (all present, all `available: true`)
- min = -0.19006358310199747; max = 0.08828914553745193
- mean = -0.05054134388858336 (matches `T_harn` exactly)
- sample SD (ddof=1) = 0.05993090874724968 → SE = SD/√67 = 0.007321725821419925 (matches `SE_harn` exactly)
- per-universe dark event counts range from 167 (seed 901000) to 173 (seed 901066) in the two sampled
  entries; all 67 `n_dark` values are present in the JSON (not individually re-quoted here)

## 4. Disposition table — three-valued outcome of EACH row (§4 of the draft, VERDICT-FREE)

Evaluated against the actual registered numbers: |Z_harn| = 6.9029276869017915, |Z_prod| =
10.115087708530357, ρ = 0.25702915205903415.

| disposition | trigger (as drafted) | evaluated against this run | fired? |
|---|---|---|---|
| ILLEGITIMATE | \|Z_harn\| > 3 AND ρ ≥ 0.5 | 6.903 > 3 → **True**; 0.257 ≥ 0.5 → **False** | **NOT FIRED** |
| FLOOR-CONSISTENT | \|Z_harn\| ≤ 3 AND \|Z_prod\| ≤ 3 | 6.903 ≤ 3 → **False** | **NOT FIRED** |
| INTERMEDIATE (a) harness-clean, production-displaced | \|Z_harn\| ≤ 3 AND \|Z_prod\| > 3 | 6.903 ≤ 3 → **False** | **NOT FIRED** |
| INTERMEDIATE (b) partial | \|Z_harn\| > 3 AND 0.2 < ρ < 0.5 | 6.903 > 3 → **True**; 0.2 < 0.257 < 0.5 → **True** | **FIRED** |
| INTERMEDIATE (c) minor-illegitimate | \|Z_harn\| > 3 AND ρ ≤ 0.2 | 6.903 > 3 → **True**; 0.257 ≤ 0.2 → **False** | **NOT FIRED** |
| INTERMEDIATE (d) HARNESS-ONLY-SIGNAL | \|Z_harn\| > 3 AND \|Z_prod\| ≤ 3 (ρ undefined) | 6.903 > 3 → **True**; 10.115 ≤ 3 → **False** | **NOT FIRED** |
| NO-READ | g-closure red, JOIN gate red, byte-id red, g-population red, g-znorm red | all gates green (§1–§2 above); `NO_READ.triggers` = [] | **NOT FIRED** |

`disposition` field reported by the script: **`"INTERMEDIATE (b) partial"`** — the sole row whose trigger
condition evaluates True against the run's own Z_harn, Z_prod, ρ.

Per the draft's own row text for INTERMEDIATE (b) (quoted verbatim, claim-writeback and stage-5-action
columns, reported here as the draft defines them, not as a ruling by this reader): claim writeback —
"both claims partial; split quoted"; stage-5 action — "fresh RULE: replication cell R (§6) within the cap,
or park with the split."

This record makes no determination as to which of these the author should choose; that choice, and the
disposition's ratification, is explicitly routed back to the author as a fresh RULE per the draft's own
binding language (§4 header: "every row returns as a fresh RULE — nothing self-ratifies").

## 5. Output artifact

Full JSON: `results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_result_rev4_read.json`
(single run, `mode: "real"`, exit code 0, written once).
