# g-c0-baseline gate record — m-head-rebaseline C0-prime

Instrument (graph spec §2, decisions row 4; g-c0-baseline row of the gate/instrument table,
`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` line 243): **band = max_abs = 0 on shared columns; md5
match.** Verdict-free beyond the stamp itself — numbers below are banked, not interpreted. A red
STOPs every downstream delta-read per the graph spec (line 85) and specifically blocks
`m-t5-armS`/`m-t5-armR`'s comparand check.

## STAMP

| venue | stamp |
|---|---|
| iiib | **RED** |
| joint_r1 | **RED** |

Both venues fail on `md5 match`; the fallback `max_abs = 0` characterization also fails (nonzero
on several columns/leaves, detailed below). `fisher_quality.csv` is the sole file that is
byte-identical (md5 match, max_abs 0) in both venues.

## 1. SLURM completion evidence (sacct)

```
JobID|JobName|State|ExitCode|Elapsed|NodeList
6764460_0|graph1-c0prime-headrebaseline|COMPLETED|0:0|00:06:43|uc2n853
6764460_0.batch|batch|COMPLETED|0:0|00:06:43|uc2n853
6764460_0.extern|extern|COMPLETED|0:0|00:06:43|uc2n853
6764460_1|graph1-c0prime-headrebaseline|COMPLETED|0:0|00:06:44|uc2n853
6764460_1.batch|batch|COMPLETED|0:0|00:06:44|uc2n853
6764460_1.extern|extern|COMPLETED|0:0|00:06:44|uc2n853
```

Both array tasks: `State=COMPLETED`, `ExitCode=0:0`. Job feedback (`slurm-6764460_0.out`):
`CPU Efficiency 23.33%`, `Memory Efficiency 43.65%` — no OOM/timeout markers, `State: COMPLETED
(exit code 0)`.

Per-task stdout (`logs/graph1_c0prime_headrebaseline_task{0,1}_*.out`) both end with the clean
`=== done: graph1-c0prime-headrebaseline task=N venue=<venue> h=0.730 ===` marker preceded by
`dataset pins OK` lines. `.err` tails are matplotlib/fonttools glyph-subsetting log noise (PDF
figure generation), not error output — greps for `error|traceback|exception|fail` over the full
`.log` files returned only benign matches (a DataFrame column literally named
`STELLAR_MASS_ABSOULTE_ERROR`, and one `quality filtering` info line).

Commit pin verified both tasks: `GIT_COMMIT_AT_RUN.txt` and `provenance_*.json` both report
`1ec9514dd1808c48b18c0792dce558e5bba0f116` (matches the LAUNCH_RECORD's submission commit; ancestor
of the flip `5e7fda16`). Dataset pins confirmed OK in-script for both tasks (CRB md5
`9a1f2a14384a9281c97ca3be312ddaab`, catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`; joint_r1
additionally confirmed its observed-catalogue sha256 pin).

**Conclusion: clean completion, rc=0, both tasks — reachable and confirmed, not merely
"COMPLETED".**

## 2. Retrieval integrity (rsync -L + md5 manifest, both ends)

Retrieved `posteriors/`, `posteriors_with_bh_mass/`, `diagnostics/`, `fisher_quality.csv` from
`$WS/run_20260902_graph1_c0prime_headrebaseline_{iiib,joint_r1}/simulations/` to
`c0prime_eval/{iiib,joint_r1}/` via `rsync -avL`. (`cramer_rao_bounds.csv` /
`prepared_cramer_rao_bounds.csv` are symlinks into the shared, already checksum-pinned input
dataset `run_20260729_seed61000` — the standing rsync-symlink gotcha — and are excluded from this
gate's file set as inputs, not this task's outputs.)

md5 manifest generated on the cluster (source) and independently re-computed on the local copy
after transfer; diffed — **identical, zero transfer corruption**, both venues:

```
iiib:
1c603309b5f139b52e02d0f12571ed4e  posteriors/h_0_73.json
228f12b0f086942fcfc80fbafdc1388f  diagnostics/event_likelihoods.csv
32c9f3a1b60c37616fb360bb3d6b5baa  fisher_quality.csv
abf242ed8747ba5a11b8a8ac84778460  posteriors_with_bh_mass/h_0_73.json

joint_r1:
32c9f3a1b60c37616fb360bb3d6b5baa  fisher_quality.csv
81ae557e5a378479f655d59cecb6e1b3  posteriors_with_bh_mass/h_0_73.json
8ac1f2a4b461d681353da252652457f3  posteriors/h_0_73.json
997f2b542b1f622d600a388e28e29b03  posteriors/realization_provenance.json
997f2b542b1f622d600a388e28e29b03  posteriors_with_bh_mass/realization_provenance.json
a7ca893699a71acf3a074cc36a14d5de  diagnostics/event_likelihoods.csv
```

(`realization_provenance.json` exists only for joint_r1, per its `--observed_catalogue` sha256
STOP-gate; not part of the shared-column comparison.)

## 3. Reference (wave-3 banked comparand)

LAUNCH_RECORD names the comparand as `wave3_20260830/{iiib,joint_r1}` task-21 outputs (h=0.730,
seed 777021). Located locally, already banked in-repo (not re-pulled from cluster):

- `results/campaign51_20260728/realistic_20260729/wave3_20260830/iiib/simulations/{posteriors,posteriors_with_bh_mass,diagnostics}/...`
- `results/campaign51_20260728/realistic_20260729/wave3_20260830/joint_r1/simulations/{posteriors,posteriors_with_bh_mass,diagnostics}/...`

`run_metadata_21.json` (iiib) confirms `random_seed=777021`, `h_value=0.73`, `simulation_index=21`
— matches the C0-prime task's seed/h exactly. Reference commit at wave-3 time was
`1e092e82a7fea45fd20c23dfdbc2b96e562be322` (pre-flip-repair HEAD used for that campaign;
recorded here as evidence only, not interpreted).

`event_likelihoods.csv` in the wave-3 reference is a 65108-row file (41 h-values x 1588 events,
one row per event per h in the full HEAD grid, banked across the whole campaign) — filtered to
`h == 0.73` (1588 rows) before comparison, row-aligned on `event_idx`, matching the C0-prime
output's own 1588-row single-h file shape exactly.

## 4. File-by-file md5 table (C0-prime vs wave-3 reference)

| venue | file | C0-prime md5 | wave-3 ref md5 | match |
|---|---|---|---|---|
| iiib | posteriors/h_0_73.json | 1c603309b5f139b52e02d0f12571ed4e | 563ef45b0598dcfc8f5c9ba19efbf9fd | NO |
| iiib | posteriors_with_bh_mass/h_0_73.json | abf242ed8747ba5a11b8a8ac84778460 | 637b5cd21fac54b86e3734d060496947 | NO |
| iiib | diagnostics/event_likelihoods.csv | 228f12b0f086942fcfc80fbafdc1388f | 704e042570a1d90eb05f575ea53ee18e | NO |
| iiib | fisher_quality.csv | 32c9f3a1b60c37616fb360bb3d6b5baa | 32c9f3a1b60c37616fb360bb3d6b5baa | YES |
| joint_r1 | posteriors/h_0_73.json | 8ac1f2a4b461d681353da252652457f3 | 681364526966e835696946c4733456bb | NO |
| joint_r1 | posteriors_with_bh_mass/h_0_73.json | 81ae557e5a378479f655d59cecb6e1b3 | f7ad1a7df61bf4f0e0f3f6cb25c0b14a | NO |
| joint_r1 | diagnostics/event_likelihoods.csv | a7ca893699a71acf3a074cc36a14d5de | 0913b4eb7e1232a119b1b6237fab2ea8 | NO |
| joint_r1 | fisher_quality.csv | 32c9f3a1b60c37616fb360bb3d6b5baa | 32c9f3a1b60c37616fb360bb3d6b5baa | YES |

Note: `fisher_quality.csv` is identical across both venues too (same md5 for iiib and joint_r1) —
consistent with it being derived from the shared, dataset-pinned CRB input rather than
venue/evaluate-specific output.

## 5. max_abs characterization on shared columns (md5 mismatch cases)

Structural check first (both venues, both `posteriors` and `posteriors_with_bh_mass`): key sets
and array lengths are identical between C0-prime and wave-3 reference at every level of the nested
JSON (no `KEY_MISMATCH`, no `LEN_MISMATCH` issues) — the divergence is purely numerical, not
structural.

### `posteriors/h_0_73.json` (per-event scalar; `h` key excluded as non-per-event)

| venue | n_events compared | max_abs | at event_idx |
|---|---|---|---|
| iiib | 1588 | 0.011987371958155815 | 249 |
| joint_r1 | 1588 | 0.01950658158524865 | 889 |

### `posteriors_with_bh_mass/h_0_73.json` (nested `galaxy_likelihoods` / `additional_galaxies_without_bh_mass` arrays included)

| venue | n_numeric_leaves compared | max_abs | at path |
|---|---|---|---|
| iiib | 8,355,780 | 216544.26303892955 | `.galaxy_likelihoods.889[0][1][0]` (new=18133323.856, ref=17916779.593) |
| joint_r1 | 7,773,065 | 987610.0823674798 | `.additional_galaxies_without_bh_mass.889[0][1][0]` (new=..., ref=...) |

### `diagnostics/event_likelihoods.csv` (1588 rows, h=0.73 subset, aligned on `event_idx`)

19 shared columns; column-wise max_abs (nonzero columns only — 13 of 19 columns are exact 0.0,
including `B_num`, `B_num_wbh`, `D_tilde_phi`, `alpha_G_phi`, `L_comp`, `den_log_term`, `g_frac`,
`h`, `r_Malm`, `w_G`, `w_G_legacy`, `w_tilde_G`, `event_idx`):

| venue | column | max_abs |
|---|---|---|
| iiib | L_cat_no_bh | 4.845906204431403 |
| iiib | num_log_term_no_bh | 0.8587279268427146 |
| iiib | num_log_term_with_bh | 0.17719762362510494 |
| iiib | combined_no_bh | 0.011987371958155801 |
| iiib | L_cat_with_bh | 0.0047554377123987 |
| iiib | combined_with_bh | 0.00029467945313679995 |
| joint_r1 | L_cat_no_bh | 12.634064809150482 |
| joint_r1 | num_log_term_no_bh | 1.2805663558362763 |
| joint_r1 | num_log_term_with_bh | 0.17342658274017353 |
| joint_r1 | combined_no_bh | 0.01950658158524865 |
| joint_r1 | L_cat_with_bh | 0.0035793250588652004 |
| joint_r1 | combined_with_bh | 0.0002534242715218 |

Cross-check (evidence only): `event_likelihoods.csv`'s `combined_no_bh` column max_abs
(0.011987371958155801 iiib / 0.01950658158524865 joint_r1) matches the `posteriors/h_0_73.json`
per-event max_abs to full float precision in both venues, consistent with `posteriors/h_0_73.json`
storing `combined_no_bh` per event.

**Overall max_abs across all shared-column files, both venues: nonzero.** The
`max_abs = 0` band is not met by either venue on any of the three non-`fisher_quality.csv` files.

## Summary

| venue | stamp | md5 match | max_abs = 0 |
|---|---|---|---|
| iiib | RED | 3/4 files NO | 3/4 files nonzero (see table) |
| joint_r1 | RED | 3/4 files NO | 3/4 files nonzero (see table) |

Per the graph spec: this RED **STOPs every downstream delta-read** against the m-head-rebaseline
comparand (m-s3-postflip-coverage, v-falsifier-ii-classG, m-joint-r1-mass-aware,
m-t5-armS/m-t5-armR, r-completion-residual) and specifically blocks `m-t5-armR`'s launch. No
interpretation of the mismatch is offered here — the numbers above are banked as evidence for the
author's next [RULE].
