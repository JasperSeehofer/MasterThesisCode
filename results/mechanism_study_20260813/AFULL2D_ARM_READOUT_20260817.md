# A-FULL-2D arm readout — 20260817

**Status: PRESENTED, NOT ADJUDICATED — branch determination returns to the author.**
This document executes the readout mechanics registered in
`PREREGISTRATION_A_FULL_2D.md` (registering commit `d50de222`). It performs no
branch selection beyond restating exactly what `score_afull2d.py` printed.

## 1. What ran

- **Job**: bwUniCluster job `6336352`, SLURM array `0-24`, 25/25 tasks `COMPLETED`, exit code `0:0` for every task.
- **Commit**: `d50de222606a581bb0a91586b04bd6898ce39548` (registering commit), confirmed identical across all 25 tasks' `.out` banners and JSON `git_commit` fields. `git_dirty=true` on every task (expected — dirt_inventory recorded, consistent across tasks, `import_path_clean=true`).
- **Seeds**: exactly `20315108..20315132` (25 seeds, offsets `+54300..+54324` on base `20260808`), one seed per array task.
- **Estimator**: `estimator_variant = a_full_gsel`, `cell = AFULL2D`, `h_true = 0.730`, full 41-point canonical `h_grid`, `dose_target = all`, `balls = real_k`, `sigma_mode = glade`, `chunk_pairs = 16384`. `pin_integrity.pass = true` on every task (CRB CSV md5, frozeng emit md5 both match their pins).

### CPU-h vs budget

| Metric | Value | Ceiling | Central estimate |
|---|---|---|---|
| Σ Elapsed × AllocCPUS (42 cpus/task) | **499.0 CPU-h** | 300 CPU-h | ~205 CPU-h |
| Σ TotalCPU (sacct, `.batch` steps) | **406.5 CPU-h** | 300 CPU-h | ~205 CPU-h |
| Total wall-clock (sum of 25 task Elapsed) | 11.88 h | — | — |
| Mean CPU efficiency (from `.out` seff banners) | ~81% (range 74.8–91.1%) | — | — |

Both realized-CPU-h metrics **exceed the 300 CPU-h ceiling and the ~205 CPU-h central
estimate** (by ~66% and ~35% respectively on the two metrics). Per-task wall time ranged
21:11–43:59 (mm:ss); tasks 15 and 16 were the outliers (43:59, 42:59 wall; 23:01:05,
23:27:26 TotalCPU) — noted here as fact, not diagnosed further (out of scope for a readout).

## 2. Retrieval integrity

Retrieved via `rsync` from `~/darksiren-emri/results/mechanism_study_20260813/` to
`results/mechanism_study_20260813/afull2d_staged/` (25 per-seed JSONs `AFULL2D_h0p730_results_seeds{0..24}_1.json`, plus 25 `.out`/`.err` log pairs under `afull2d_staged/logs/`).

- File count: **25/25** ✓
- Seed set: `{20315108, ..., 20315132}` exactly, no missing, no extras ✓
- Every `per_seed` entry: `ln_post_1d` and `ln_post_2d` each length **41**, all values finite (checked with `math.isfinite`) ✓
- `config` block: byte-identical (via `json.dumps(sort_keys=True)`) across all 25 files ✓ — `estimator_variant=a_full_gsel`, full `h_grid` confirmed identical

## 3. Combine

`results/mechanism_study_20260813/AFULL2D_h0p730_results_seeds0_25.json` was built by
concatenating the 25 tasks' `per_seed` lists in ascending seed order (task 0's metadata —
`instrument`, `config`, `git_commit`, etc. — used as the combined file's top-level metadata,
after verifying it is identical across all 25 source files, matching the pattern used for the
prior `AFULL_h0p730_results_seeds0_25.json` / `AJREN_...` / `AM2P_...` combined arms). The
combined file's single-seed `aggregate` block is retained from task 0 but is not meaningful for
the 25-seed set — `score_afull2d.py` recomputes every DS statistic from the raw per-seed
`ln_post_1d`/`ln_post_2d` vectors, never reading `aggregate` or the per-seed scalar fields.

## 4. Scorer output (verbatim)

```
=== INPUTS ===
  AFULL2D_h0p730_results_seeds0_25.json: FOUND

=== DS-G1: paired 2D-1D excess at truth (PRIMARY, branch-carrying) ===
  excess: mean=-11.8 SE=0.61 sd=3.0 N=25 band=[-15.7,-7.8] -> PASS
  mirror reference: -11.74 +/- 1.04 (N=15)

=== DS-G2: tilts at truth (secondary, non-branch-carrying) ===
  1d: T_mean=+14.2 SE=30.1 N=25 band=[-131.5,+192.7] -> PASS
  2d: T_mean=+2.4 SE=30.1 N=25 band=[-143.8,+181.6] -> PASS

=== DS-G3: 2D-channel coverage restoration ===
  2D: hpd50/68/90=0.520/0.760/0.960 N=25
  RESTORED: True

=== DS-G4: 1D invariance (c1 bit-identity to a_full) ===
  checked separately post-run (prereg §6 item 1 pre-submission gate + DS-G4 note); this scorer does not recompute c1 bit-identity from the arm JSON alone (no a_full comparison column in the schema)

=== DS-G5: per-seed T scatter, MAP bias, rails, non-finite (descriptive) ===
  1d: T in [-264.9,+286.2] sd=150.7 bias=+0.0008+/-0.0014 railed_low=0 railed_high=0 nonfinite=0
  2d: T in [-276.5,+277.3] sd=150.4 bias=+0.0006+/-0.0013 railed_low=0 railed_high=0 nonfinite=0

=== BRANCH (§5) ===
  PRESENTED, NOT ADJUDICATED — the author rules
  status: PRESENTED, NOT ADJUDICATED — the author rules
  branch: 1. DS-G1 PASS + DS-G3 RESTORED (M-OWNED-CLOSED candidate)
  reason: excess(2D-1D)=-11.8 in band, 2D coverage restored
```

Full machine-readable output: `results/mechanism_study_20260813/score_afull2d_output.json`.

## 5. DS-G4 bit-identity check (prereg §4)

The scorer does not compute DS-G4 from the arm JSON alone (no `a_full` comparison column in
the schema), so it was reproduced locally and independently, per the prereg's explicit
instruction:

- Built the venue context from the **arm's own retrieved config** (`crb_reference_csv`,
  `frozeng_emit_json`, `pruned_catalogue_csv`, `injection_data_dir`, `chunk_pairs`, `h_grid`,
  etc. — read directly out of the combined arm JSON, not re-derived).
- For seeds **20315108** and **20315120** (2 of the 25 arm seeds), drew the identical seed
  realization (`venue_transfer._draw_seed_realization`) and ran
  `venue_transfer.log_channel_posteriors_ball_sigma_vector` with
  `estimator_variant="a_full"` over the **full 41-point `h_grid`** — the same per-h body
  (`_channel_terms_at_h`) the arm's `a_full_gsel` cell uses, called with the sibling variant.
- Compared the resulting `ln_post_1d` vector against the arm's retrieved `ln_post_1d`
  (computed under `a_full_gsel`) for the same seed.

Result:

```
seed 20315108: n_points=41 max|diff|=0.000e+00 -> BIT-IDENTICAL
seed 20315120: n_points=41 max|diff|=0.000e+00 -> BIT-IDENTICAL

OVERALL DS-G4: PASS (bit-identical)
```

Script: `results/mechanism_study_20260813/ds_g4_bit_identity_check.py`; output:
`results/mechanism_study_20260813/ds_g4_bit_identity_check_output.json`.

## 6. Log anomalies

- **`.err` logs**: every one of the 25 tasks logs exactly one `WARNING` — the known
  2-code-revision injection-pool note (`Injection pool spans 2 code revisions (a9f29e82,
  f6449051) — legitimate for straggler resubmits after a non-physics fix, but verify none of
  them changed SNR semantics.`), 25/25 occurrences, no other distinct warning text. No
  `ERROR`/`Traceback`/`Exception` strings found in any `.out` or `.err` file across all 25 tasks.
- **`.out` logs**: clean module-load banners, one `afull2d task=N ...` identification line per
  task (commit confirmed `d50de222...` on every banner), and the SLURM `seff` accounting
  footer. No warnings or errors.

## 7. Combined-file / retrieval provenance

| Item | Path |
|---|---|
| Staged per-seed JSONs | `results/mechanism_study_20260813/afull2d_staged/AFULL2D_h0p730_results_seeds{0..24}_1.json` |
| Staged logs | `results/mechanism_study_20260813/afull2d_staged/logs/afull2d_{0..24}.{out,err}` |
| Combined arm JSON | `results/mechanism_study_20260813/AFULL2D_h0p730_results_seeds0_25.json` |
| Scorer output | `results/mechanism_study_20260813/score_afull2d_output.json` |
| DS-G4 check script + output | `results/mechanism_study_20260813/ds_g4_bit_identity_check.py`, `..._output.json` |

---

**This document is PRESENTED, NOT ADJUDICATED.** The branch table in
`PREREGISTRATION_A_FULL_2D.md` §5 is the author's to apply against the numbers above; no
branch has been selected or endorsed by this readout.
