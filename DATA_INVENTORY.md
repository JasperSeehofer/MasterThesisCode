# Data Inventory

Tracks all simulation datasets and their evaluation status.
**Update this file whenever a new dataset is generated or a pipeline change is applied.**

---

## ⚠️ ALL PRIOR DATA RETIRED — 2026-06-20 (fresh production run in progress)

As of **2026-06-20** the two bundled physics fixes are **merged into `main` (`af6014d`)**:
- `[PHYSICS]` redshifted-mass convention (Design B, `0099ce2`) — **RE-SIMULATE tier**: changes the
  injected waveform/SNR/Fisher AND the injection generation. Invalidates **every CRB and every injection
  pool** ever produced (all source-frame; cannot be back-corrected).
- `[PHYSICS]` L_cat Gray A.9/A.10 ratio-of-sums (`816f904`) — **RE-EVALUATE tier**: invalidates all posteriors.

**Therefore every dataset listed below — all CRBs, the p_det injection pool(s), all `posteriors{,_with_bh_mass}/`,
all diagnostics and figures — is STALE and RETIRED.** Do **not** use any of it for results. A fresh
production run (regenerating injections AND events with the merged code) supersedes everything:
- Scope: 4 independent seeds @ h_true=0.73 + 2 closure truths (0.67, 0.77); validate one seed end-to-end first.
- Local stale working dirs archived under `simulations/_RETIRED_20260620_pre_massfix_lcat/`.
- New entries will be appended to the Dataset Registry + Evaluation Log as the campaign lands.

### Cluster workspace lifetime (`emri`)

| checked | expiry | action |
|---|---|---|
| 2026-07-25 | ~~2026-08-31~~ → **2026-09-23** (`ws_extend emri 60`; **0 extensions left — LAST one used**) | no further extensions possible: copy all final results off the workspace to persistent storage BEFORE 2026-09-23 |

### De-rail evidence backup (2026-07-02)

The 2026-07-01 de-rail demonstration ran in ephemeral `/tmp/seed600_local/` (not in git).
Durable copies now exist:
- **Repo** [`thinkpad`, ✅ present 2026-09-02, 776 KB]: `results/commission_20260701/redteam/posteriors_per_mode/`
  — per-mode combined posteriors (prod 0.86 / prod_global 0.60 / local_ratio 0.73 / volume_deconv 0.73
  / catonly 0.73) + `crux_results{,_fixed}.json` (commit `1f0e371`). **This is in git**, so it is the
  one genuinely durable piece of the de-rail evidence.
- **Home** [`thinkpad`, ⚠️ **VERIFIED ABSENT 2026-09-02**]: `~/data-backups/seed600_local_derail_20260702/`
  (was 3.8 GB: full working dirs incl. the 474 MB with-BH-mass per-event posteriors, the 494-event CRB
  subsample, the fixed 8-col catalogue copy). The directory `~/data-backups/` does not exist. The
  "durable copies now exist" claim above therefore holds **only** for the small in-git repo copy; the
  full working dirs are gone. Not recovered by the 2026-09-02 dedup — this is a separate, earlier loss.
- **⚠ Ω_m era mismatch (registered 2026-07-10):** the underlying seed600 CRBs were simulated at
  Ω_m = 0.25 (pre-G11) but every post-`bdf5339` evaluation infers at Ω_m = 0.2726 → the venue is
  biased LOW ≈0.3–0.8% (z-graded). **A/B-code-comparison venue only** — see the 2026-07-10
  provenance row in the Evaluation Log and `.planning/BIAS-INVESTIGATION-20260710.md` §1.

---

## Pipeline Change Checklist

When any trigger file changes, mark affected datasets as stale and re-run the
corresponding tier before reporting results.

| Tier | Trigger files / changes | Action required |
|------|------------------------|-----------------|
| **Re-simulate** (CRBs invalid) | `LISA_configuration.py` PSD formula · SNR threshold in `constants.py` · Fisher stencil or ε in `parameter_estimation.py` · waveform params passed to `few` | Re-run GPU simulation on cluster; old CRBs are **stale** |
| **Re-prepare** (prepared CSV invalid) | `scripts/prepare_detections.py` · sampling method · SNR pre-filter applied at prepare time | Re-run `prepare_detections.py` on raw CRBs |
| **Re-migrate** (coord frame invalid) | `galaxy_catalogue/handler.py` BallTree frame or angle convention · `scripts/migrate_crb_to_ecliptic.py` transform | Re-apply migration; mark `_coord_frame` entries stale |
| **Re-evaluate** (posterior invalid) | `bayesian_inference/bayesian_statistics.py` · `bayesian_inference/simulation_detection_probability.py` (p_det grid build OR extrapolation policy) · `single_host_likelihood` · D(h) normalisation · injection files · h-grid in `cluster/evaluate.sbatch` | Re-run `evaluate.sbatch`; old `posteriors/` are **stale** |
| **Re-figure** (figures invalid) | Any plotting code · `--generate_figures` pipeline | Re-run `--generate_figures`; old PDFs are **stale** |

---

## ⚠️ Injection pools RETIRED — 2026-07-03 (depth-1.5 campaign prep)

All pre-dt² / z_cut = 0.5 injection pools are **RETIRED** by the issue-#20 depth change:
local `simulations/injections/` (80 files) moved to
`simulations/injections_RETIRED_predt2_zcut0p5_20260703/`; cluster pools
(`seed43000_Mz`, `seed700`) marked retired in `cluster/datasets.yaml`. The campaign
regenerates a single-h (h_ref = 0.73) pool at z_cut = 1.5 with the **same filenames** —
never mix the eras. Guards: injection rows now carry `z_cut` + `code_rev` provenance
columns; `SimulationDetectionProbability` rejects shallow/mixed pools
(`expected_z_max`, readiness sweep A2-STALE-POOL-GATE); `--evaluate` hard-fails below
95% P_det grid coverage (`--allow_low_pdet_coverage` to override deliberately).

---

## Phase-2 Campaign (2026-07-03 → , tag `campaign-phase2-base` = `b6bf57d`)

| Item | Value |
|---|---|
| **Injection pool** | `$WS/injection_pool_depth15_50k` — 500 files / 50 000 events, z_cut = 1.5, single h_ref = 0.73, provenance-stamped (see `cluster/datasets.yaml` `depth15_campaign`) |
| **Design** | 4 seeds @ h_true = 0.73 (BASE_SEED 1000/2000/3000/4000) + closure 0.67 (5000) / 0.77 (6000); `--tasks 100 --steps 40` (~4k detections/seed target); volume_deconv; 41-value hybrid h-grid; per-task eval seeds `SEED·1000 + task` |
| **Smoke** | run_20260703_seed900 (jobs 5740080-83) — sim/merge validated; prescreen audit 543 pairs → quick gate DISABLED (`b6bf57d`); anchors: ~42 s/detection (GPU), injections 3-6.6 s/event |
| **Submitted** | seed1000: jobs 5743694-97 (2026-07-03 ~12:45Z). Remaining seeds staggered against the ~300-job submit cap |
| **Criterion** | pre-registered in `.planning/CAMPAIGN-PREP-PHASE2.md` §4b BEFORE submission |

---

## Galaxy Catalogue (reduced GLADE+)

The single on-disk input the whole pipeline shares — previously untracked here.

| Property | Value |
|----------|-------|
| **File** | `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` (headerless, **1.68 GB**, 22 641 048 rows) |
| **Schema** | **8 columns** (order = `_reduced_catalog_column_names()`): RA_deg, Dec_deg, B_mag, **z_cmb**, z_error (**per-PV-class since 2026-07-27**, issue #40b RATIFIED: corrected rows = meas ⊕ catalogue σ_tot ⊕ (1+z)·150 km/s/c; uncorrected/flag-null rows = meas ⊕ (1+z)·500 km/s/c; NO 0.0015 floor; runtime `SIGMA_V_PEC_KM_S` retired to 0.0), stellar_mass, stellar_mass_err, z_flag (1=photo-z, 3=spec-z; trailing) |
| **Frame** | **z_cmb** (CMB frame) since `18e9608` (2026-07-02 rebuild; 99.9% rows shifted, median \|Δz\| 6e-4 — `.planning/gate/GATE_SIGNOFF.md:27`) |
| **Depth** | **Full-depth** (no z cut in the writer; max z ≈ 7.03). Effective load-time depth = `Model1CrossCheck.max_redshift` = 1.5 via `_get_pruned_galaxy_catalog`. `GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT` is documentation-only |
| **Rebuild** | `results/commission_20260701/scratch/rebuild_catalog.py` from repo root; **move the old CSV aside first** (writer appends, `mode="a"`); ~77 s full GLADE+ pass on the dev box |
| **Source** | `darksiren_emri/galaxy_catalogue/GLADE+.txt` (6.4 GB, dev box ONLY — cluster cannot rebuild; staging is rsync of the reduced CSV per `/cluster` skill) |
| **Superseded** | `.pre40b_20260727` (double-counted PV widths, 2026-07-02 build), `.zhelio_20260702` (z_helio 8-col, 2026-07-01), `.stale6col_mar28` (6-col) — backups next to the live file, RETIRED |
| **Coupled artifact** | `m_th_map_nside32.npy` (frozen per-pixel m_th, C1: byte-identical on injection + inference sides). Built from the full flag-{1,3} catalogue; MUST be regenerated atomically on both sides if the CSV content ever changes. **2026-07-27 z_error-only rebuild: map re-derived from the new CSV and verified BYTE-IDENTICAL** (depends only on B_mag + sky position), so the frozen artifact stands |

---

## Dataset Registry

### phase45-seed200-20260501 *(current canonical, post-Tier-3 fix)*

| Property | Value |
|----------|-------|
| **Location (cluster)** [`bwuni`] | `/pfs/work9/workspace/scratch/st_ac147838-emri/run_phase45_20260501/simulations/` |
| **Location (local)** | `simulations/cluster_run_phase45_20260501/cramer_rao_bounds.csv` |
| **Device** | `thinkpad` — ⚠️ **VERIFIED ABSENT 2026-09-02** (the whole `simulations/` tree is gone from the dev box; reclaimed at some point without a ledger row). Copy of record is `bwuni` only, and that workspace expires 2026-09-23 |
| **Lineage** | Same population as `prod-seed200-20260401`; CRB rsynced into Phase 45 workspace 2026-05-01 19:44 (no new GPU sims) |
| **Simulation date (origin)** | 2026-04-01 / 2026-04-02 |
| **Git commit (simulation)** | `a56e30de` — v1.3 milestone roadmap |
| **SLURM tasks (origin)** | 99 GPU tasks, `gpu_h100`, BASE_SEED=200 |
| **SNR threshold (simulation)** | 15 |
| **Total CRB rows** | 4 497 |
| **Rows with SNR ≥ 20** | 424 *(used by evaluation at runtime)* |
| **Confusion noise (Phase 9)** | ✅ included |
| **5-point stencil (Phase 10)** | ✅ included |
| **Coordinate frame** | ✅ ecliptic (Phase 43 migration) |
| **Ecliptic migration** | ✅ applied |
| **Evaluation status** | ✅ evaluated post-Tier-3 fix (commit `d6b784c`, 2026-05-04): 1D MAP=0.7400 z=+1.4σ, 2D MAP=0.7400 z=+1.97σ |
| **Verification status** | 🔄 Multi-truth bias-vs-h_true sweep in progress (commit `b110ba7`, 2026-05-04) |

**Status:** Active reference dataset for the bias-resolution paper. The h_true=0.65 closure test and the multi-truth panel both rescale this CRB.

---

### sim-seed300-extension-20260504 *(in flight — extension to ~1000 events)*

| Property | Value |
|----------|-------|
| **Location (cluster)** [`bwuni`] | `/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260504_seed300_extension/` |
| **Submission** | 2026-05-04 (job `4216105` failed, replaced by `4216323`, gpu_h100) |
| **Git commit (simulation)** | `b110ba7` — verification scaffold |
| **SLURM tasks** | 50 GPU tasks, `gpu_h100`, BASE_SEED=300 (per-task seed = 300 + array_id) |
| **Sim steps per task** | 10 000 (walltime-capped at 1 h) |
| **Observed yield (partial, 2026-05-04 19:30)** | 17 of 50 tasks complete → 500 SNR≥20 events (~30 events/task, far above 9.4% extrapolation; remaining 33 tasks running/pending) |
| **Confusion noise (Phase 9)** | ✅ included |
| **5-point stencil (Phase 10)** | ✅ included |
| **Coordinate frame** | ✅ `ecliptic_BarycentricTrue_J2000` — **natively ecliptic** (sim ran at `b110ba7`, post Phase 36 catalog rotation `b460297` 2026-04-22). `_coord_frame`/`_cov_frame` markers added by hand, **no rotation applied** |
| **Status** | 🟡 partial — 17/50 tasks merged into phase46-merged-20260504; remaining tasks may land later |

**Goal:** When merged with `phase45-seed200-20260501`, target ~600–650 SNR≥20 events for tighter σ_boot in the multi-truth panel and better-resolved D(h). Continued extension may follow.

**Merge realised:** local concat of 17 per-task CSVs (rsynced 2026-05-04 19:30) → tagged with `_coord_frame`/`_cov_frame` (no rotation — sim already ecliptic) → concatenated with Phase 45 CRB into `phase46-merged-20260504` (see below).

**⚠️ Migration foot-gun caught 2026-05-04:** an initial pass ran `migrate_crb_to_ecliptic.py` on this CSV, which double-rotated phiS/qS by ~obliquity. The script's idempotency guard relies solely on the `_coord_frame` marker; it cannot detect that a *new* simulation already wrote ecliptic natively. The mistake was reverted from `.bak_equatorial`. **Action item:** harden `migrate_crb_to_ecliptic.py` to refuse migration when the source commit is post-Phase-36 (or require an explicit `--legacy-equatorial` flag for pre-Phase-36 archives).

---

### phase46-merged-20260504 *(current canonical for multi-truth verification)*

| Property | Value |
|----------|-------|
| **Location (local)** | `simulations/cluster_run_phase46_merged_20260504/cramer_rao_bounds.csv` |
| **Device** | `thinkpad` — ⚠️ **VERIFIED ABSENT 2026-09-02** (`simulations/` tree gone). No cluster row was ever recorded for the merged CSV → **this dataset may exist nowhere**; it is a local concat, so it is reconstructible from its two parents if those survive |
| **Lineage** | `phase45-seed200-20260501` (4497 rows, 424 SNR≥20) ⊕ `sim-seed300-extension-20260504` partial (500 rows, 500 SNR≥20) |
| **Total CRB rows** | 4 997 |
| **Rows with SNR ≥ 20** | **924** (~2.18× over Phase 45 alone) |
| **Coordinate frame** | ✅ `ecliptic_BarycentricTrue_J2000` — Phase 45 half: rotated by `migrate_crb_to_ecliptic.py` (origin commit predates Phase 36, so equatorial → ecliptic was correct). seed=300 half: **natively ecliptic** (origin commit post Phase 36), markers added without rotation. |
| **Schema** | 126 cols (matches Phase 45) |
| **Constructed** | 2026-05-04 ~19:30 local; commit (pending) |
| **Used by** | Multi-truth panel (`run_multi_truth_sweep.sh` via `INPUT_CRB` env var; default points here) |

**Notes:**
- The seed=300 extension job (`4216323`) is still in flight — at the time of merge, 17/50 tasks had flushed CSVs. A second merge can be performed later if tighter σ_boot is desired.
- Injections directory is **not** merged — the panel orchestrator symlinks Phase 45's `injections/` for D(h) estimation. This is scientifically valid: D(h) is population-level, and seed=300 follows the same population as Phase 45, so reusing Phase 45 injections is consistent.
- Per-task CSVs preserved at `simulations/cluster_run_seed300_extension_20260504/simulations/cramer_rao_bounds_simulation_*.csv` for reproducibility.

---

### prod-seed200-20260401 *(superseded — see phase45-seed200-20260501)*

Historical record of the original simulation campaign. Same data lives in `phase45-seed200-20260501` with the post-Tier-3 evaluation results attached.

---

### prod-seed200000-20260409

| Property | Value |
|----------|-------|
| **Location (cluster)** [`bwuni`] | `/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260409_seed200000/` |
| **Simulation date** | 2026-04-09 / 2026-04-11 |
| **Git commit (simulation)** | `d247f8ba` — Phase 34 |
| **SLURM tasks** | 15 GPU tasks |
| **SNR threshold (simulation)** | 15 |
| **Total prepared rows** | 303 |
| **Rows with SNR ≥ 20** | ~50 (not counted) |
| **Confusion noise (Phase 9)** | ✅ included |
| **5-point stencil (Phase 10)** | ✅ included |
| **Coordinate frame** | ⬜ equatorial ICRS |
| **Ecliptic migration** | ⬜ NOT YET applied |
| **Evaluation status** | ⬜ Not evaluated post-migration |

**Status:** Superseded by `prod-seed200-20260401` (smaller, same physics era). Migrate only if merging datasets.

---

### local-phase43-verification *(current local canonical)*

| Property | Value |
|----------|-------|
| **Location (local)** | `simulations/prepared_cramer_rao_bounds.csv` |
| **Device** | `thinkpad` — ⚠️ **VERIFIED ABSENT 2026-09-02** (`simulations/` tree gone). Regenerable via `prepare_detections.py` from `prod-seed200-20260401` if that CRB survives on `bwuni` |
| **Origin** | Partial rsync of `prod-seed200-20260401` (542 of 4 497 rows) |
| **SNR threshold (simulation)** | 15 |
| **Total rows** | 542 |
| **Rows with SNR ≥ 20** | 60 *(used by evaluation)* |
| **Confusion noise** | ✅ |
| **5-point stencil** | ✅ |
| **Coordinate frame** | ✅ `ecliptic_BarycentricTrue_J2000` (Phase 43 migration) |
| **Ecliptic migration** | ✅ Applied 2026-04-27, commit `a2df67bf` |
| **Evaluation (h=0.73 single point)** | ✅ MAP = 0.730 (Phase 43-03 verification) |
| **Full h-sweep (38 points)** | ⬜ Not yet run post-Phase-43 |
| **Figures regenerated** | ⬜ Not yet (existing PDFs dated 2026-04-24, pre-Phase-43) |

**Status:** Valid for single-point verification. Superseded by `prod-seed200-20260401` once migrated (7× more SNR≥20 events).

---

### small-validation-runs (archive)

Runs `run_20260328_seed100_v3` (19 rows), `run_20260330_seed100` (22 rows), `run_20260330_seed75` (42 rows) — pre-Phase-9/10, equatorial, no evaluation planned. Keep as historical record only.

---

### closure-verification-runs *(post-Tier-3 verification CRBs, rescaled from phase45)*

Each closure run rescales `phase45-seed200-20260501` to a different h_true via `scripts/bias_investigation/test_23_rescale_crb_to_h_true.py`. Distance and SNR transform as `d_L_new = (h_old/h_new) · d_L_old`, `SNR_new = (h_new/h_old) · SNR_old`; events with SNR_new < 20 are dropped.

| Workdir | h_true | N_events post-rescale | Source CRB | Status |
|---------|--------|----------------------|------------|--------|
| `simulations/closure_h0p65/` | 0.65 | 251 (424 → 251 after SNR filter) | phase45-seed200 | ✅ smoke 2026-05-04 (cluster job 4210104) |
| `simulations/closure_h0p73/` | 0.73 | 424 (identity rescale) | phase45-seed200 | ✅ smoke 2026-05-04 (cluster job 4213944) |

Posteriors landed at:
- `simulations/cluster_run_closure_h0p65_finegrid/posteriors{,_with_bh_mass}/h_*.json` (21-pt grid 0.6000–0.7000)
- `simulations/cluster_run_closure_h0p73_finegrid/posteriors{,_with_bh_mass}/h_*.json` (21-pt grid 0.6800–0.7800)

Aggregated by `scripts/bias_investigation/test_24_multi_truth_bias_sweep.py` →
- `scripts/bias_investigation/outputs/phase45/multi_truth_sweep.{json,png}`

**Smoke verdict (2 truths):** all four panel checks PASS — weighted mean bias |z|≤2, sign distribution random (sign FLIP between truths), no boundary-rail, per-event pos_frac dispersion not suspicious.

**Pending:** full 7-truth panel (`{0.60, 0.65, 0.70, 0.73, 0.75, 0.80, 0.85}`) on cpu_il post-20:00.

---

### Post-2026-07-28 fleet backfill (2026-08-27)

`cluster/WORKSPACE_ARCHIVAL_TRIAGE_20260827.md` found ~30 workspace run directories
(~250 GB) created since this file's last update (2026-07-26) that were never entered
here or in `cluster/datasets.yaml` — essentially the entire post-2026-07-28 fleet
(csg_pilot, o4_shards, the p3_2d / b0-identity / bat / cf / massab families,
seed61000-65000, `realizations_20260729`). Backfilled below from each directory's
`run_metadata*.json` where present; **where no metadata file was found, or where
the metadata gives only mechanical fields (git_commit/timestamp/args) and not the
scientific "why", this is marked UNVERIFIED PROVENANCE rather than guessed** — the
author should confirm before these are cited as evidence. Full per-directory detail
(sizes, local-mirror status, archival priority) is in the triage doc; this section
is the durable registry entry, not a duplicate of the triage.

Structured entries live in `cluster/datasets.yaml` under `campaigns_20260728_plus:`
(the two files are a matched pair — see that file's header comment). Summary:

| Directory / family | git_commit | Provenance | Notes |
|---|---|---|---|
| `run_20260727_massab_{ablApp,ablAprime,cellApp,cellBpp,n200Aprime,zmzApp,zmzBpp,zmzGridOnly}` | `fe0ca3e`/`bb24b71`/`608426b` (3 distinct, per-dir) | UNVERIFIED beyond run_metadata | Mass-ablation cell study, 8 small dirs. No claim/ledger row identified. |
| `pp_fullpower_20260727` | — (no run_metadata found) | UNVERIFIED | Name suggests `validation/pp_coverage.py` full-power run; not confirmed. |
| `realizations_20260729` | `7b30d1f` | UNVERIFIED beyond run_metadata_realization.json | Realization sidecars for the seed6x000 fleet below; cf. `cluster/SKILL.md` gotcha #10 (fragile/path-sensitive). |
| `run_20260729_seed{61000,62000,63000,64000_h0p67,65000_h0p77}` | `03cfe80` (61000/62000/63000) / `7b30d1f` (64000) / none found (65000) | UNVERIFIED beyond run_metadata | "campaign51" closure-truth seeds (0.67/0.77); local partial mirror at `results/campaign51_20260728/`. |
| `csg_pilot_20260821` | bounded by commit chain (readout §11: ...e5bd5bf0, dae957d6, 3d385152, f59a6f48) | RECOVERED (no run_metadata, but see note) | Matches the C-SG campaign (session memory rows #145-#157). `CAMPAIGN_READOUT_REPORT_CSG_20260821.md` §11 lists the exact per-stage commit chain; SLURM jobs 6415588 (pilot)/6420343 (fleet) confirmed still in `sacct` 2026-08-27. Distilled result in `results/prod2d_closure_20260818/csg_pilot_bands_output.json`. |
| `o4_shards_20260821` | bounded to `bfe4d09c` (scorer, 4.5h before submit) — not exactly pinned | PARTIALLY RECOVERABLE | Same-day O4 pre-check. SLURM job 6441957 confirmed in `sacct` 2026-08-27 (submit 17:46, 11/11 COMPLETED). No a22_stamp in shard JSONs (campaign predates the A22 convention, adopted same day at commit `1e862398`), so the exact tree state at run start is not independently verified, only bounded by the scorer's commit time. |
| `p3_b0_identity_fleet_20260823` | `3bd6b564bc900751fcbed8df16fb5fad3b275edb` | FULLY RECOVERABLE | `results/campaign51_20260728/realistic_20260729/p3_b0_work/retrieval_manifest_20260824.json` records branch/tag/commit/SLURM-job-history/timestamp; independently confirmed by per-seed `a22_stamp` in every `bc_/bt_*_meta.json`. Backs the ratified "b0 identity: UNDISCRIMINATING" verdict. Previously mislabeled UNVERIFIED — the recovery data was already in the repo, just not cross-referenced here. |
| `p3_2d_fleet_20260825` | `fb4ac4eea8bb415e38d542f6f458b3dd259060f0` | FULLY RECOVERABLE | `p3_2d_fleet_submission_20260825.json` (job_id 6708698, array 0-23, submit timestamp) + per-seed `a22_stamp` in every `bc_/bt_*_meta.json`, cross-verified to the same commit. Part of the `[P3-2D]` thread (rows #198-#211, PARKED). At least one shard FAILED (`bc_900115_work_FAILED_6708698_14`). |
| `p3_2d_rhs2_20260826` | `7e4f1c64` | FULLY RECOVERABLE | `p3_2d_rhs2_submission_20260826.json` records commit/job_id (6709953, array 0-31)/seed plan/submit timestamp directly. The RHS-suspect work (rows #209-#211, "RHS unlinked-mass suspect REFUTED"). Local sync of a subset appears in-flight under `results/campaign51_20260728/realistic_20260729/ca_rhs_work/`. |
| `p3_2d_fleet_repair_20260827` | `d04d9dc9` (branch `fix/p32d-classg-venue-repair`, tag `p32d-repair-4af1baec`) | FULLY RECOVERABLE | `[P3-2D]` class-G venue repair fleet, job 6723958 (array 0-23, partition `cpu_il`, 24/24 COMPLETED 2026-08-27); seeds 900101-900124, arms bc+bt (48 arm-seed pairs). First fleet with `write_provenance.sh` wired — per-task `provenance_*.json` present. Produced under `PREREGISTRATION_P3_2D_REPAIR_20260827.md` REGISTERED DESIGN v2 (submission record filled commit `bbfdd2e0`); read out 2026-08-28 via `stage_lhs2d` — `results/campaign51_20260728/realistic_20260729/P3_2D_REPAIR_READOUT_20260828.md` (same results dir as the prereg). |
| `run_20260827_headreadout_iiib` | `d04d9dc9` | FULLY RECOVERABLE | Production H0 readout at HEAD, `iiib` venue (true reduced GLADE+ catalogue). Jobs 6724169 (smoke, array 21) + 6725283 (full, array 0-40), both COMPLETED 2026-08-27. `EVAL_SEED` 777000; config `absolute_marginal`/`volume_deconv`/`fused`/`phi`; CRB set `run_20260729_seed61000` (md5 `9a1f2a14384a9281c97ca3be312ddaab`). 41 posteriors both channels + `diagnostics/event_likelihoods.csv` (65108 rows). Registration: `results/campaign51_20260728/realistic_20260729/MEASUREMENT_HEAD_READOUT_20260827.md`. |
| `run_20260827_headreadout_joint_r1` | `d04d9dc9` | FULLY RECOVERABLE | Same production H0 readout as `run_20260827_headreadout_iiib` above but `joint_r1` venue (`observed_catalogue_seed900001.csv`, sha256 `e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751`). Jobs 6724170 (smoke) + 6725284 (full array), both COMPLETED 2026-08-27. Same `EVAL_SEED`/config/CRB set. Registration: `results/campaign51_20260728/realistic_20260729/MEASUREMENT_HEAD_READOUT_20260827.md`. |
| `d1_sand` | — (no run_metadata found) | UNVERIFIED | Possibly related to `d1_a1`/`d1_a2` below; "sand" suggests a sandbox run. Not confirmed. |
| `run_2026080{4,5}_{frozeng,postfix,d1_a1,d1_a2,n2sel1d}_{iiib,joint_r1}[_smoke]`, `run_20260817_fusioncf_{fused,off}_{iiib,joint_r1}`, `run_20260819_{cf_v0,cf_v1,cf_v2k05,cf_v2k2,postfix_baseline}_{iiib,joint_r1}`, `run_20260820_bat_{eoff,jker,v0}_{iiib,joint_r1}` | one commit per named sub-campaign (see `datasets.yaml`) | UNVERIFIED beyond run_metadata | ~32 dirs, ~3.3G each, CPU evaluate-only re-eval sweeps (not fresh GPU sims). `iiib`/`joint_r1` = paired A/B arms per named study. Local mirror per the triage: `frozeng`, `fusioncf_{fused,off}` ALREADY-LOCAL; `postfix`, `d1_a1`, `d1_a2`, `n2sel1d` MUST-ARCHIVE (mostly missing locally); `cf_v*`, `bat_*` not spot-checked. |
| `run_20260820_corr_arms` | — (no run_metadata found) | UNVERIFIED | Likely adjacent to the `bat_*`/`cf_v*` family above. |
| `run_prod2d_20260818` | — (no run_metadata found) | UNVERIFIED | Name matches the local `results/prod2d_closure_20260818/` tree (2D venue thread CLOSED, row #116/#119); likely the raw run backing that closure. |
| `run_20260829_wave2_c0_iiib` | `ff230621` | FULLY RECOVERABLE | Wave-2 charter node C0 — shared baseline reproduction gate, `iiib` venue, h=0.730 only (task 21, seed 777021). Job 6738998 (1 task, COMPLETED 2026-08-29). CoR-P config plus explicit `mass_filter_geometry=linear`/`mass_filter_k=1.5`; CRB set `run_20260729_seed61000` (md5 `9a1f2a14384a9281c97ca3be312ddaab`). Gate PASS bit-identical vs the banked `run_20260827_headreadout_iiib` (`d04d9dc9`): max_abs 0.000 on all 14 non-trivial numeric `event_likelihoods.csv` columns at h=0.73 (1588 rows); both posteriors md5-identical. No fallback triggered — banked HEAD readout reused as zero-compute baseline for C3/C4 and the C1 truth node. Registration: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_C0_BASELINE_GATE_20260829.md` §13. |
| `run_20260829_wave2_c3_iiib` | `ff230621` | FULLY RECOVERABLE | Wave-2 charter node B5.2 (arm C3) — log-k3 mass-window counterfactual, `iiib` venue, arm T `mass_filter_geometry=log`/`mass_filter_k=3.0`, H4 grid (h=0.660/0.665/0.670/0.730). Job 6738999 (array 0-3, all COMPLETED 2026-08-29). CRB set `run_20260729_seed61000` (md5 `9a1f2a14384a9281c97ca3be312ddaab`); baseline B = the banked `run_20260827_headreadout_iiib` (`d04d9dc9`), reused per the C0 gate PASS. Independent readout (2026-08-29): R6 (1D bit-identity) PASS (max rel diff 2.667e-14); R2 (engagement) PASS (0.9684); R5 (stencil validity) PASS; primary reading `Δmean_h,pred = +0.0035225` — INTERMEDIATE (between IMMATERIAL ≤0.003 and `T_mat=0.008`), REPORTED not adjudicated; R1 retention falsifier FALSIFIED (production true-host retention 66/76 identical arm vs baseline; the window's 621/1588-event candidate collapse falls exclusively on dark/impostor-class events); measured cost 4.97 CPU-h vs 44–137 CPU-h estimated. Retrieved to `results/campaign51_20260728/realistic_20260729/wave2_20260829/c3/`. Registration: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md` §13; readout: `fanout1_20260829/B5_2_WIN_K3_READOUT_RECORD.md`, `fanout1_20260829/b5_2_readout.json`; `BIAS_HISTORY_LEDGER.md` row #247. |
| `run_20260829_wave2_c4_iiib` | `ff230621` | PARTIALLY RECOVERABLE — diagnostics CSV retrieved; provenance extras pending (SSH outage) | Wave-2 charter node B7.2 (arm C4) — PROD-CF-2D twin flip (`catalogue_numerator_survival_2d=mz_sel`, `center=eff`), `iiib` venue, H4 grid (h=0.660/0.665/0.670/0.730). Jobs `6739000` (task 0) + `6739001` (tasks 1–3), all COMPLETED 2026-08-29. CRB set `run_20260729_seed61000` (md5 `9a1f2a14384a9281c97ca3be312ddaab`); baseline B = the banked `run_20260827_headreadout_iiib` (`d04d9dc9`), reused per the C0 gate PASS. Independent readout (2026-08-29): R1 PASS (0/6352 violations); R2 PASS (982/982 engaged, fraction 1.0); R6 PASS (1D max_abs 0.0 at all H4 nodes); primary reading `Δmean_h,pred = +0.0025057` via the Δℓ′(0.665)/I_HEAD stencil — IMMATERIAL-PREDICTED (≤ T_mat/2=0.004); STEP-2 overhead ≈0.99× (task-0 385s vs C0 baseline 388s); measured cost 6.8 CPU-h vs 59.7–105 CPU-h estimated. **Partial local retrieval**: only `simulations/diagnostics/event_likelihoods.csv` was retrieved before the cluster SSH `ControlMaster` session expired mid-transfer; `run_metadata_*.json`, `logs/`, the 2 `posteriors_with_bh_mass` JSONs for h=0.67/0.73, and a local `GIT_COMMIT_AT_RUN.txt` copy remain on the cluster pending retrieval — none of the gate/reading numbers above depend on the missing files. Retrieved to `results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/`. Registration/readout: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §15; `fanout1_20260829/B7_2_TWIN_CF_READOUT_RECORD.md` §6; `fanout1_20260829/b7_2_readout.json`; `BIAS_HISTORY_LEDGER.md` row #248. |

**Action item for the author:** every UNVERIFIED row above needs either (a) a pointer
to the claim card / ledger row it fed, or (b) a ruling that it is disposable/superseded,
before the 2026-09-23 workspace deadline.

**2026-08-27 recovery pass (provenance-defect fix).** `run_metadata*.json` is written
only by `main.py`'s entry point; every campaign in the C-SG/O4/P3-2D/b0-identity family
above bypassed it by calling a bespoke driver script directly (root cause, and the
forward fix `cluster/write_provenance.sh`, are in `cluster/SKILL.md` gotcha #12). Before
concluding the record was lost, this pass checked for campaign-carried provenance in
other forms — per-seed `a22_stamp` sidecars (an author-adopted convention, commit
`1e862398`, 2026-08-21, predating this fix and independent of it), submission JSONs, a
retrieval manifest, and campaign readout docs' own commit citations — and cross-checked
against live `sacct` history on the cluster (job records for 2026-08-20 through
2026-08-26 were all still present as of this pass). Result: **`p3_b0_identity_fleet_20260823`,
`p3_2d_fleet_20260825`, `p3_2d_rhs2_20260826`, and `csg_pilot_20260821` are FULLY
RECOVERABLE** (exact git commit + SLURM job IDs + timestamps, from sources already in
the repo, just not previously cross-referenced into this registry); **`o4_shards_20260821`
is PARTIALLY RECOVERABLE** (job ID and timing recovered from `sacct`, but the exact
commit is bounded to a ~4.5h window rather than pinned, since it predates the a22_stamp
convention adopted later that same day). The other UNVERIFIED rows in this table
(`massab_*`, `pp_fullpower_20260727`, `d1_sand`, `run_20260820_corr_arms`,
`run_prod2d_20260818`) were not part of this pass's scope and remain as before — they
were not named in the provenance-defect task and were not checked for sidecar recovery.

---

## Injection Data

> **2026-06-21 — the table below describes the RETIRED (source-frame) pool; superseded.** The fresh
> M_z-convention pool is on the cluster at `injection_20260620-213449_seed43000/simulations/injections`
> (560 files, **504,000 events**, 7 h-nodes 0.60–0.90 @ 72k each, M_z validated). **Methodology note:**
> the detection-horizon survival p_det is **h-invariant** (`d_hor=SNR·d_L/thr` cancels h; `(M,z)` drawn
> from the h-free rate model; `M_z=M·(1+z)` h-free) — so it pools all h and a **single h suffices**.
> Verified the 7 h-nodes are independent (0 shared `(z,M)` across h) → pooling genuinely smooths p_det;
> ~42% within-pool `(z,M)` repeats are emcee MCMC autocorrelation (correct density representation — do
> NOT deduplicate), so effective sample size < 504k but the estimate is unbiased. `submit_injection.sh`
> default changed to single-h (`0.73`) accordingly; multi-h is optional and only adds pooled samples.

| Property | Value |
|----------|-------|
| **Location** | `simulations/injections/injection_h_<val>_task_<N>.csv` |
| **h values** | 0.60, 0.65, 0.70, 0.73, 0.80, 0.85, 0.90 (7 values, ~40 tasks each) |
| **Total files** | 262 |
| **Total rows** | ~165 000 (pooled by P_det KDE) |
| **Coordinate frame** | ⬜ equatorial ICRS *(not yet migrated; injection data is used for P_det KDE, not host-matching — assess whether migration is needed before next evaluation)* |
| **Used by** | `SimulationDetectionProbability` at evaluation time |

---

## Evaluation Log

| Date | Dataset | Git commit | h-grid | SNR≥20 events | 1D MAP | 2D MAP | Notes |
|------|---------|-----------|--------|--------------|--------|--------|-------|
| 2026-04-24 | local-phase43-verification (pre-fix) | pre-`a2df67b` | 38-pt | 60 | 0.860 | — | Equatorial frame — biased |
| 2026-04-27 | local-phase43-verification | `a2df67bf` | single h=0.73 | 60 | 0.730 | — | Phase 43-03 verification only |
| 2026-05-04 | phase45-seed200-20260501 (Tier 3 fix) | `d6b784c` | 38-pt + closure h=0.65 | 424 (h=0.73) / 251 (h=0.65) | 0.7400 (z=+1.4σ) | 0.7400 (z=+1.97σ) | Production result post-Tier-3 fix; closure h=0.65 PASSED |
| 2026-05-04 (smoke) | closure-verification (multi-truth) | `b110ba7` | 21-pt fine | 243 (h=0.65) / 408 (h=0.73) | 0.6555 / 0.7233 | 0.6558 / 0.7285 | Smoke for 2-truth panel; sign flip — **production seed-dependence ~0.02** |
| 2026-05-05 (partial) | phase46-merged-20260504 (4/7 truths) | pre-fix | 11–21-pt fine each | 1549 SNR≥20 | 1D z=+0.6→+3.8 across truths | 2D z=+37 (h=0.73), +55 (h=0.60) | Partial 4-truth panel; 2D structural residual uncovered → motivated 1D/2D principled-extrapolation fix below |
| 2026-05-05 (audit) | phase46-merged-20260504 — 2D p_det edge behaviour | (diagnostic) | 21-pt fine ×4 truths | 1549 | — | — | `test_26` confirms 6–12% events fall below dl_min(2D); raw scipy linear-extrapolation drifts negative & clips to ≈0 vs principled 1; mechanism for 2D z=37/55 |
| **2026-05-05 (pipeline change — Re-evaluate)** | `[PHYSICS]` 1D + 2D detection probabilities → principled monotonic-asymptotic extrapolation | `2b33cad` | n/a | n/a | n/a | n/a | Replaces Phase 45 Plan 45-02/04 anchor scheme (1D Wilson 95% LB + intermediate empirical anchor) with a principled bridge from (dl_min, p_edge) to (0, 1) and slope-matched + clamped suppressing-face extrapolation. **All `posteriors{,_with_bh_mass}/` produced before this commit are stale** for any conclusion that depends on absolute p_det values near the d_L grid edges. See `.planning/2D-CHANNEL-AUDIT-20260505.md` for the audit + rationale. |
| 2026-05-05 (post-bridge fix) | h=0.73 phase46-merged closure (job 4229895) | `2b33cad` | 21-pt fine | 1473 | 0.7309 (z=+0.19σ PASS) | 0.7441 (z=+3.60σ FAIL) | 1D fully closed; 2D residual remains. σ_boot widened 6.5× (0.0006→0.0039), confirming pre-fix tightness was a discontinuity symptom. 2D bias 16× larger than 1D — violates info monotonicity → motivated H3 fix below. |
| **2026-05-05 (pipeline change — Re-evaluate)** | `[PHYSICS]` H3 fix: numerator p_det observation→hypothesis + 2D grid M_z axis | `f01595c` | n/a | n/a | n/a | n/a | Two coupled changes: (1) numerator passes `host_M·(1+z)` (hypothesis) instead of `_det_M` (observation); (2) 2D grid axis is built in observer-frame `M_z = M_source·(1+z_inj)` to match query coordinate. Removes Phase 14's "approximation, not a bug" mismatch. **All `posteriors_with_bh_mass/` produced before this commit are stale.** 1D `posteriors/` unaffected. See `docs/H0_BIAS_RESOLUTION.md` §3.15 (planned) and §4.7 (planned-state record). |
| 2026-05-06 (post-H3 fix) | h=0.73 phase46-merged closure (job 4252817; original 4250797 cancelled, fairshare-starved on cpu_il) | `f01595c` | 21-pt fine | 1473 | **0.7309 (z=+0.18σ PASS ✅)** | **0.7307 (z=+0.20σ PASS ✅)** | **G_H3b PASS**: 2D z ≤ 2σ AND 2D bias (+0.0007) ≤ 1D bias (+0.0009). σ_boot 0.0047 (1D), 0.0037 (2D). 2D bias dropped 20× from +0.0141 (post-bridge) to +0.0007 (post-H3). Info monotonicity restored. |
| **2026-06-20 (pipeline change — RE-SIMULATE)** | `[PHYSICS]` redshifted-mass convention (Design B): inject detector-frame `M_z = M_source·(1+z)` into FEW | branch `physics/mass-redshift-convention` (NOT merged) | n/a | n/a | n/a | n/a | The sim injected source-frame `M` into FEW, which expects detector-frame `M_z`. Fix lifts the mass once at injection (`parameter_space.set_host_galaxy_parameters`, `main.py:injection_campaign`); the CSV "M" column now stores `M_z`, so the p_det grid no longer re-lifts (`simulation_detection_probability.py:265`) and the inference's `det.M = M_z` / `/(1+z)` filter / `host_M·(1+z)` lift become exactly correct. Dead `_map_BH_masses_to_redshifted_masses` removed. **ALL existing CRBs AND injection sets (incl. canonical seed400) are STALE for this change — they were generated source-frame and CANNOT be back-corrected** (the Fisher covariance was computed at the wrong mass). Must be regenerated; **bundle into the keystone multi-seed re-sim campaign.** Correctness fix — expected H0 effect small and possibly adverse (see `project_mass_convention_defect` memory; `docs/H0_BIAS_RESOLUTION.md`). Do NOT run `--evaluate` with this branch against old data (would treat source-frame M as M_z). |
| 2026-05-06 (post-H3 regression) | h=0.73 Phase 45 412 events (job 4250798) | `f01595c` | 21-pt fine | 412 | 0.7425 (z=+2.40σ) | 0.7418 (z=+3.20σ) | 1D-2D asymmetry resolved (Δ=0.0007); residual +0.0125 bias attributable to combined bridge+H3 effects on small dataset, within seed-dependent MAP scale of 0.02. |
| **2026-05-07 (Phase 48 production, pre-F1)** | h=0.73 phase46-merged fine-grid sweep (jobs `4271862` + `4344777`) | `8292359` | 63-pt non-uniform (Δh=0.001 dense core [0.710, 0.750] + Δh=0.010 wings [0.600, 0.860]) | 1473 | **0.7324 (z=+1.16σ PASS ✅)** | **0.7322 (z=+0.97σ PASS ✅)** | Initial pre-F1 verdict — superseded by user-detected coherent-noise issue (spiky combined posterior; debug session `.planning/debug/posterior-noisy-peak.md`). σ_boot 0.0021 (1D), 0.0022 (2D) — ~2× tighter than R1 21-pt. R1's 21-pt MAP (~0.7308) was Δh-resolution-limited. Info monotonicity preserved. Two-submission recovery: job `4271862` TIMEOUT'd at 30:00 with per-h walltime 2× over plan, sbatch hardened idempotent (`7b24b98` skip-if-output + `8292359` opt-in archive gate); `4344777` filled remaining 22/63 h-values. Verdict: `scripts/bias_investigation/outputs/phase46_merged/h3_production_sweep_verdict.json`. **STALE under post-F1 pipeline** — see 2026-05-14 row below. |
| **2026-05-14 (Phase 49 F1 — PARTIAL)** | h=0.73 phase46-merged fine-grid sweep, post-F1 (`[PHYSICS]` `87ea7a8`) (job `4662333`) | `ef3d2c3` | 63-pt non-uniform same as Phase 48 | 1473 | **0.7378 (z=+2.02σ — bias +0.0054 worse vs pre-F1)** | **0.7378 (σ_boot collapsed to ≈0 → bootstrap pinned on single h=0.738 bin)** | **F1 PARTIAL — does not fully resolve the spikiness.** The rising flank 0.730→0.738 is now mostly monotonic/smooth (one noise mechanism removed), but a single huge discontinuity remains at h=0.738→0.739 (1D drops 16×, 2D drops 32×). MAP shifted +0.0056 further from truth than pre-F1. F1 fixed bin-edge drift (one mechanism); a second mechanism — **SNR-threshold integer crossings** of individual injections — remains independent of bin edges and is the suspected dominant residual noise source. The Farr 2019 fixed-injection + analytic-reweighting form (gwcosmo / ICAROGW production pipelines) would eliminate both mechanisms; queued as F4 in the debug session. Verdict: `scripts/bias_investigation/outputs/phase46_merged/F1_post_fix_verdict_PARTIAL.json`. Pre-F1 posteriors archived on cluster at `run_production_h0p73_20260506/simulations/archive/production_h0.73_20260512_175829/`. **Not paper-grade.** No figure refresh from these posteriors. |
| 2026-05-04 (merge) | phase46-merged-20260504 | pending commit | n/a (CRB construction) | 924 (424+500) | — | — | Phase 45 ⊕ seed=300 partial (17/50 tasks); ~2.18× event count for tighter σ_boot |
| *(pending)* | phase45 + full sim-seed300 merged | next | 38-pt | ~1100+ | — | — | After remaining seed=300 tasks land (~later tonight) |
| **2026-06-20 (BRANCHES MERGED + ALL DATA RETIRED)** | mass-convention `0099ce2` + L_cat-gray `816f904` merged to main | `af6014d` | n/a | n/a | n/a | n/a | Both physics fixes merged (`/check` green: 569 pass, ruff+mypy clean). Stale multi-seed campaign (seeds 500/600/700/800, jobs 5084023–5084038) **cancelled** — it predated both fixes + reused source-frame injections. **All prior CRBs/injections/posteriors RETIRED** (see banner). Fresh run: regenerate injections + events with merged code; 4 seeds @0.73 + closure 0.67/0.77; validate one seed end-to-end first. |
| **2026-07-10 (seed1000 local combine — RAILED, campaign NO-GO)** | Phase-2 `run_20260703_seed1000` (3,470 CRB rows; depth15 pool via now-broken symlinks; cluster eval jobs 5743696+) | eval `b233375`; combine at a `b233375` worktree (`combine_local_20260710.py`; D(h) diagnostic reconstructed from eval logs, ~1e-4) | 40-pt 0.60–0.86 (h=0.705 hole: task 16 hung) | 3,454 evaluated; **only 1,462 with likelihoods** | **0.6000 (lower grid EDGE)** | **0.6000 (lower grid EDGE)** | First campaign posterior — **unusable as H₀ measurement**. 58% of events silently zero-host-dropped (issue #29); effective mass-pruned catalogue is 99.98% z<0.3 (issue #30). Verified diagnosis: `FINDINGS_COMBINE_20260710.md` (in the run dir). **Do NOT relaunch seeds 2000–6000 until re-validated with the fixes below.** |
| **2026-07-10 (pipeline change — Re-evaluate)** | `[PHYSICS]` `8db6c6e` zero-host pure-completion fallback (#29) + `f29a5e7` selection-integral z-caps (#30 groundwork), branch `physics/zero-host-completion-fallback` | `8db6c6e`+`f29a5e7` | n/a | n/a | n/a | n/a | Zero-host events now contribute `p_i = B_num/D` (Gray Eqs. 29+32; was: silent drop since 2024). **All `posteriors{,_with_bh_mass}/` from venues with ANY zero-host events are STALE for this change** — in practice every depth-1.5 campaign venue (seed900: 60% drops; seed1000: 58%) and marginally seed600-era venues (few drops). Shallow seed400 perf venues unaffected in the hosts-present values (fallback adds events, never changes them; pipeline+kernel goldens unchanged except the documented synthetic-fixture cap re-pin). Deep-venue validation = seed1000 re-eval on cluster return, THEN campaign relaunch. |
| **2026-07-10 (provenance note — seed600 Ω_m era mismatch)** | `run_20260628_seed600` CRBs (evidence-locker seed600; PV-test + de-rail venues) | sim era pre-G11 | n/a | n/a | n/a | n/a | seed600 was **simulated at Ω_m = 0.25** (pre-`bdf5339`); all post-G11 evaluations infer at **Ω_m = 0.2726** → generation-vs-inference mismatch biases the venue LOW by ≈0.3–0.8% (z-graded). seed600 is therefore an **A/B-code-comparison venue only**; do not quote absolute closure residuals from it without the era term. The only Ω_m-consistent closure venues are the Phase-2 campaign seeds. See `.planning/BIAS-INVESTIGATION-20260710.md` §1. |
| **2026-07-19/25 (EXP-40 re-eval — RAIL PERSISTS, fallback exonerated)** | `run_20260719_seed1000_exp40` (seed1000 CRB via symlinks; #29 fallback ACTIVE) | eval `ba2b381` (main, post-consolidation) | 41-pt 0.60–0.86 (full, no hole) | 3,454/3,454 used (0 excluded) | **0.6000 (EDGE)** | **0.6000 (EDGE)** | EXP-40 verdict: #29 fallback does NOT de-rail. 57.7% of events completion-governed but that branch is nearly h-inert (59 log-e grid span, slightly *pro*-rail). Measured tilt decomposition: 16.3% global w_G + **82.3% host-found L_cat** + 1.4% fallback. z≤0.3 subsets also rail. Analysis: `results/campaign_phase2_runs/run_20260719_seed1000_exp40/FINDINGS_EXP40_20260725.md`; issue #30 comment 2026-07-25. 2D `n_events_empty=2` is a combine bookkeeping artifact (issue #36). |
| **2026-07-25 (pipeline change — Re-evaluate, deep venues)** | `[PHYSICS]` `7d3573d` B_num analysis-depth cap + `276c8c7` `--max_redshift` CLI (branch `feat/max-redshift-cli`) | `7d3573d`+`276c8c7` | n/a | n/a | n/a | n/a | B_num completion-numerator upper limit now `min(z_upper, redshift_upper_limit)` — domain-matched to D(h)/β_Ḡ/Σ_global (Gray Eq. 32; the one sibling integral f29a5e7 missed). At the default depth 1.5 the cap can bind only for the farthest events (4σ window beyond 1.5, where the population has no support) → deep-venue posteriors with such events are marginally stale; shallow venues byte-identical. `--max_redshift`/`MAX_REDSHIFT` env expose the knob for truncation studies (evaluate-only; WARNs if shallower than HOST_DRAW_Z_MAX). |
| **2026-07-25 (z_cut truncation scan — ALL RAIL; issue #30 option (b) DEAD)** | `run_20260725_seed1000_zcut{02,03,05}` (same seed1000 CRB, `--max_redshift` 0.2/0.3/0.5, consistent D/β_Ḡ/w_G/B_num truncation) | eval `276c8c7` | 41-pt 0.60–0.86 each | n_used 1,194 / 1,768 / 3,251 of 3,454 | **0.6000 (EDGE, all three)** | **0.6000 (EDGE, all three)** | Depth truncation does NOT de-rail at any tested depth — option (b) empirically dead. Sharp contrast: the untruncated z≤0.2 *subset* closed at 0.7292, but the *truncated* z_cut=0.2 re-eval rails → the rail lives in the h-dependence of the truncated selection/normalization structure (w_G=β_G/D) interacting with L_cat, not in the deep events. Primary path forward: full Gray-mixture / L_cat h-dependence estimator work (issue #30 escape hatch). Jobs 6038717–6038722. |
| **2026-07-25 (pipeline change — Re-evaluate, opt-in)** | `[PHYSICS]` `49b9ade` `absolute_marginal` normalization mode: absolute-mass per-event host marginal (issue #30 Variant 1, `DERIVATION_ESTIMATOR_REDESIGN.md`) | `49b9ade` | n/a | n/a | n/a | n/a | Opt-in via `--normalization_mode absolute_marginal`; production default `volume_deconv` unchanged. Replaces the self-normalized ratio-of-sums `L_cat` catalogue term with the absolute-mass marginal `A_i = Σ_ball w_g N_g / n̄_w`, `n̄_w = Σ_glob/β_G`, removing the host-misassociation mechanism identified in §3.21 (`D1_EMPIRICAL_DECOMPOSITION.md`/`D2_STRUCTURAL_AUDIT.md`). **Does not close the deep venue** — seed1000 probe rails HIGH to h=0.86 (row below), exposing a second calibration defect closed by `8fbb21e` (row below). No existing production posteriors depend on this mode (opt-in only); `catalog_only` and `volume_deconv` byte-identical. |
| **2026-07-25 (pipeline change — Re-evaluate, opt-in, overnight-autonomy grant)** | `[PHYSICS]` `f9c58f4` opt-in σ_z-smeared `Σ_glob`: num/denom kernel symmetry (issue #30 R4) | `f9c58f4` | n/a | n/a | n/a | n/a | Opt-in via `--smear_global_selection`; production default unchanged. Symmetrizes the σ_z kernel between the in-catalogue numerator (already smeared) and the global catalogue selection sum `Σ_glob` (previously point-evaluated). Measured effect: removes only ~20% of the `n̄_w` residual slope defect (+0.067/h of the +0.38/h target) — superseded/moot once `8fbb21e` (below) removes `n̄_w` entirely; retained as a diagnostic flag. |
| **2026-07-26 (pipeline change — Re-evaluate; deep-venue posteriors from older estimators SUPERSEDED for production claims)** | `[PHYSICS]` `8fbb21e` `generator_marginal` normalization mode: generator-consistent selection normalization (E1 FIX-3, `DERIVATION_GENERATOR_CONSISTENT_NORM.md`) | `8fbb21e` | n/a | n/a | n/a | n/a | Opt-in via `--normalization_mode generator_marginal`; production default `volume_deconv` unchanged pending the multi-seed + seed600 gates (§3.22). Replaces the approximate calibration `n̄_w = Σ_glob/β_G` with the generator-exact `n̂_w = W_cat/V_f(h)` (derived from the injection generator's own `Bernoulli(F)` channel split) and `D` with `D_gen = Σ_glob/n̂_w + β_Ḡ`. The packet's own pre-registered prediction (gap ≈+52…+92 ln, still railed HIGH) was falsified favorably by the seed1000 probe: **1D and 2D MAP = 0.7300 = truth**, attributed to the accompanying point/point σ_z pairing decision sharpening catalogue-matched events. **Any deep-venue posterior produced under `volume_deconv` or `absolute_marginal` is superseded for production H0 claims as of this commit**, pending the standing multi-seed χ² and seed600 third-arm gates (§3.22). |
| **2026-07-26 (pipeline change — Re-evaluate, gated)** | `[PHYSICS]` `a608c4f` z-resolved detection survival `S(d_L\|z)` (E1 FIX-2, `DERIVATION_ZRESOLVED_SURVIVAL.md`) | `a608c4f` | n/a | n/a | n/a | n/a | Opt-in via `--pdet_z_resolved`; replaces the pooled detection-horizon survival `S(d_L)` with the z-conditional `S(d_L\|z)` (kernel in `u=ln(1+z)`, Scott/Abramson-adaptive) inside `D`, `β_Ḡ`, `Σ_glob`, and per-host `D_g`; `B_num` and all numerators untouched (no p_det in the numerator, MFG convention). Refines an earlier binning-artifact estimate (`dlog D_zres/dh` −0.56 → −1.26). Stacked on `generator_marginal`, preserves the truth peak (row above) and deepens the anti-0.86 margin; alone on `absolute_marginal`, reproduces its own pre-registered −69 ln HIGH→interior prediction to within 0.3 ln (−68.75 ln measured). |
| **2026-07-26 (seed1000 deep-venue probe suite, estimator arc summary)** | `results/lcat_h_dependence_20260725/{v1_probe,v1_probe_genmarg,zres_probe_20260726,densecore_probe}/` (seed1000 CRB, 3,454 events, physics-floor combine) | `49b9ade`→`8fbb21e`+`a608c4f` | 7-pt 0.60–0.86 (dense-core: 13-pt around 0.73) | 3,454/3,454 (all probes) | see Notes | see Notes | Arc: `absolute_marginal` (V1) **0.8600 RAIL** (both channels) → `generator_marginal` (FIX-3) **0.7300 = truth** (both channels; gap 0.73→0.86 = −898.8 ln 1D / −735.4 ln 2D) → `generator_marginal`+`--pdet_z_resolved` (FIX-2 stacked) **0.7300 = truth** (both channels; gap deepens to −994.8/−831.4 ln) → dense-core 13-pt sub-grid refinement **MAP ≈ 0.7304 both channels, curvature σ ≈ 0.00025–0.0003** (⚠ width UNVALIDATED — pending pre-registered multi-seed χ² test, `RUNBOOK_NEXT_SESSION.md`; do not quote as a confidence interval). See `docs/H0_BIAS_RESOLUTION.md` §3.21/§3.22. |
| **2026-07-26 (seed600 shallow-venue A/B, must-not-change gate)** | `run_20260628_seed600` (3,355 events, `--allow_low_pdet_coverage`; Ω_m era mismatch — A/B-comparison venue only, see 2026-07-10 provenance row) | `c87caba`→ | n/a | 3,353/3,355 (all arms) | vdeconv 0.7450 / absmarg 0.7750 / genmarg+zres 0.7550* | vdeconv 0.7550 / absmarg 0.8600 RAIL / genmarg+zres 0.7550* | Registered in `SEED600_GATE_REGISTRATION.md`. Arm 1 (`volume_deconv`, production default) and Arm 2 (`absolute_marginal`, V1) measured: V1-alone **FAILS** the shallow must-not-change gate (1D +0.030, 2D rails to 0.86 — the with-BH mass-composition defect `8fbb21e` later removes). *Arm 3 (`generator_marginal`+`--pdet_z_resolved`, the production candidate) numbers shown are as reported by the concurrent verdict session (1D +0.010 vs +0.012 tolerance PASS; 2D +0.000 PASS; n_used deviation −1/−2 events under diagnosis) — **at the time this row was written `SEED600_GATE_REGISTRATION.md` itself had not yet had the verdict appended below its registration text**; treat the Arm-3 numbers here as provisional and consult that file directly before quoting. |
| **2026-07-26 (five-seed production-stack campaign — PASS on valid-4 basis; CANONICAL config declared)** | `run_20260726_seed{1000,900,2000,3000,90000}_prodstack` (source CRBs: `run_20260703_seed1000`, `run_20260703_seed900` **[INVALID — see below]**, `run_20260707_seed2000`, `run_20260707_seed3000`, `run_20260707_seed90000`) | eval+combine `6dae9d3`; env flags `--normalization_mode generator_marginal --pdet_z_resolved` (pre-dates the `ce6338e` default flip — set explicitly per run) | 41-pt 0.60–0.86 | seed1000 3,454/3,454; seed2000 3,254/3,254; seed3000 3,314/3,314; seed90000 20/20; seed900 20/20 (**invalid pool, see below**) | seed1000 0.7304; seed2000 0.7300; seed3000 0.7297; seed90000 0.7287; seed900 **0.86 RAIL (invalid)** | seed1000 0.7304; seed2000 0.7301; seed3000 0.7298; seed90000 0.7296; seed900 0.8547 (invalid) | Jobs 6044799–6044808 (eval+combine ×5). **Registered 5-seed test: QUALIFIED FAIL** (seed900 rails, criterion 3) — root cause diagnosed as invalid injection-pool provenance (see seed900 row below), not an estimator defect. **Valid-4 (1000/2000/3000/90000) readout, author-ratified 2026-07-26: PASS all criteria** — bias −0.00030±0.00035 (base) / −0.00003±0.00018 (bh_mass), width χ²=8.0 (base, marginal)/3.7 (bh_mass) both VALID, sanity PASS. Full readout + author ratification: `results/lcat_h_dependence_20260725/MULTISEED_READOUT_20260726.md`. **Campaign NO-GO LIFTED on the valid-4 basis**; merge to `main` additionally gated on an independent redteam (math+physics+anti-tuning) review, `results/redteam_20260726/` pending. **`generator_marginal` + `--pdet_z_resolved` declared the CANONICAL production configuration as of `[PHYSICS]` `ce6338e`** (defaults flipped to match; `--no-pdet_z_resolved` for legacy pooled), superseding `volume_deconv` for all production H0 claims. Per the 5-tier Pipeline Change Checklist: the `ce6338e` default flip triggers **no dataset staleness** — every 2026-07-26 prodstack run above already used these exact settings via explicit CLI/env flags, so no re-run is required on account of the flip itself. |
| **2026-07-26 (seed900 injection pool — DEFECTIVE, dataset-registry flag)** | `$WS/run_20260703_seed900/simulations/injections` | n/a (input-data defect, not a code change) | n/a | n/a | n/a | n/a | **Mark DEFECTIVE.** The `injections` symlink at this path points at a bespoke one-off pool `injection_20260703-112746_seed46910/` (4 task CSVs, ≈204 injections) instead of the canonical `injection_pool_depth15_50k/` (500 task CSVs, 50,000 injections) used by every other Phase-2 seed. Consequence: the z-resolved survival estimator built from it is severely undersampled (node ESS min/median 6/55; 418/726 sky-band cells, 57.6%, below the ESS floor — cf. seed90000 on the canonical pool: ESS min/median 211/3944, 0/726 below floor), which rails the seed900 posterior to h=0.86 (see campaign row above). **Do not evaluate anything against this symlink until re-pointed.** Fix in flight: `run_20260726_seed900_fixpool` re-points at `injection_pool_depth15_50k` (generator-consistency confirmed), eval job 6051189, combine job 6051190, canonical pool relinked and provenance-verified (500 CSVs, era-consistent). This restores the registered n=5 multi-seed test (non-blocking — the campaign verdict already stands on the valid-4 basis above). |
| **2026-08-31 (wave-3 blind HEAD readout + C0′ gate — A14 NOT MATERIAL; row #283)** | `run_20260830_wave3_headreadout_{iiib,joint_r1}` + `run_20260830_wave3_c0prime_off_{iiib,joint_r1}` (CRB `run_20260729_seed61000`, md5 `9a1f2a14…`; catalogue `c52c13b5…`; joint_r1 realization sha256 `e8f7ab31…`) | eval `1e092e82` | 41-pt 0.60–0.86 (gate: h=0.730 only) | 1588/1588 both venues | 1D 0.600 (both, unchanged) | 2D 0.665 (iiib; joint_r1 0.660→0.665) | Jobs 6746274–6746276 (84 tasks, all COMPLETED, ~6.5 min/task). C0′ gate PASS **bit-identical** both venues (row #281) → banked 2026-08-27 readouts certified as the A14 baseline. Blind readout: **2D Δmean_h +0.002127 (iiib) / +0.003519 (joint_r1), both ≤ T_mat = 0.008 → adoption NOT MATERIAL; 1D exact-zero.** A4 returns to the author pending falsifier (ii). Retrieved to `realistic_20260729/wave3_20260830/`; archive program running (`results/_archive/archive_run_wave2.sh`, wave-2+3 blocks). Also this date: the seed61000 injection pool (707 files) + raw CRB (`a1c34a46…`) staged LOCALLY under `realistic_20260729/seed61000/simulations/` (md5-manifest-verified) for T2.2b — local copies are consumers of the same pins. |

---

## Local Storage Register (added 2026-09-02)

**Why this section exists:** on 2026-09-02 the dev box hit a low-disk warning with **1.0 GB free
of 931 GB**. This register tracks what run data physically lives on the dev box, what is
regenerable, and what has no second copy anywhere. Update it whenever a campaign is retrieved
from the cluster or a local working dir is deleted.

### Device Registry — every location gets a device tag

**Convention (adopted 2026-09-02):** every path in this file carries a `device` tag naming the
physical machine or medium it lives on. A path without a device tag is not an inventory entry, it
is a rumour. Tag the *device*, not the *role* — "backup" is a claim about redundancy that only
holds if two rows name two different devices.

| Tag | Device | Identity | Capacity / free | Backed up? |
|---|---|---|---|---|
| `thinkpad` | dev laptop (the machine this repo is checked out on) | hostname `thinkpadseehofer`, machine-id `4140d149…`, NVMe `UMIS RPETJ1T24MHP2QDQ` s/n `SS1D71506X1LC51B069S`, rootfs UUID `2786115f-…` | 931 GB single partition · 160 GB free (2026-09-02) | **NO** |
| `bwuni` | bwUniCluster 3.0 workspace `emri` | `/pfs/work9/workspace/scratch/st_ac147838-emri/` | scratch | **NO** — and **expires 2026-09-23, 0 extensions left** |
| `ext-1` | *(not yet acquired)* | — | — | — |

**⚠️ There is currently exactly ONE local device.** `thinkpad` has a single NVMe with a single
partition carrying `/` and `/home` both. So today, "we have it in two places locally" is never
true — `results/_archive/` vs `~/emri-archive/`, or repo vs `~/data-backups/`, are two directories
on one disk. This is precisely the illusion that the 2026-09-02 dedup dissolved (161 GB recovered
by deleting a "second copy" that protected nothing). **Redundancy starts existing when `ext-1`
exists.**

**Tagging drift found on adoption (2026-09-02).** Applying device tags to the pre-existing entries
immediately surfaced three rows asserting local copies that are gone — the whole `simulations/`
tree and `~/data-backups/`. None had a ledger row recording the deletion. This is the argument for
the convention: an untagged path silently decays from *fact* to *belief*, and you only find out
when you go looking for the data.

### ⚠️ The single-filesystem finding (READ THIS FIRST)

`/` and `/home` are **the same partition** (`/dev/nvme0n1p3`, 931 GB). There is therefore
**no local redundancy of any kind**: `~/emri-archive/` is *not* a backup of
`results/_archive/`, it sits on the same physical disk. One NVMe failure loses every
byte in this register simultaneously.

```
$ df -h /
/dev/nvme0n1p3  931G  725G  160G  82% /     # after the 2026-09-02 dedup below
```

### 2026-09-02 dedup action (space recovered: 161 GB, zero data loss)

`results/_archive/` and `~/emri-archive/` held **byte-identical duplicate copies** of four
seed600 evidence-locker venues (separate inodes — real duplication, not hardlinks).
Verified before deletion: full `name+size` manifest match on all four trees (948 / 388 / 262 /
389 files) plus `md5sum` agreement on the three largest files of each tree. The only delta was a
leftover partial-write temp file `posteriors_with_bh_mass/.h_0_705.json.xTV1f2` (158 MB) in the
repo copy — junk, not data.

The **repo-side copies were deleted**; `~/emri-archive/` is retained as the copy of record:

| Venue (now only at `~/emri-archive/`) | Device | Size | Status |
|---|---|---|---|
| `run_20260628_seed600` | `thinkpad` (**only**) | 56 GB | evidence locker; **A/B-comparison venue only** (Ω_m era mismatch, see 2026-07-10 provenance row) |
| `run_20260726_seed600_ab_absmarg` | `thinkpad` (**only**) | 35 GB | seed600 gate arm 2 (`absolute_marginal`) |
| `run_20260726_seed600_ab_genmarg_zres` | `thinkpad` (**only**) | 35 GB | seed600 gate arm 3 (production candidate) |
| `run_20260726_seed600_ab_vdeconv` | `thinkpad` (**only**) | 35 GB | seed600 gate arm 1 (`volume_deconv`) |

Free space went 1.0 GB → **160 GB**, past the 50 GB floor, without discarding any science data.

### What is on the dev box now (~526 GB project-related)

| Path | Device | Size | Regenerable? | Second copy? | Retention call |
|---|---|---|---|---|---|
| `results/campaign51_20260728/realistic_20260729/` | `thinkpad` | 200 GB | from seed + cluster | cluster ws (**expires 2026-09-23**) | **HOT — do not touch.** Backs the live branch and rows #315–#317 |
| `results/_archive/` (post-dedup) | `thinkpad` | 119 GB | from seed | partly cluster | warm; oldest seed1000/2000/3000 blocks are the first cull candidates |
| `~/emri-archive/` | `thinkpad` | 159 GB | expensive (GPU re-sim) | **NONE** | **sole copy — highest-priority evacuation target** |
| `darksiren_emri/galaxy_catalogue/GLADE+.txt` | `thinkpad` | 6.0 GB | re-downloadable from GLADE+ upstream | upstream | keep (needed to rebuild the reduced catalogue) |
| `results/prod2d_closure_20260818/` | `thinkpad` | 14 GB | from seed | cluster ws (expiring) | warm |
| `results/run_20260817_fusion_counterfactual/` | `thinkpad` | 13 GB | from seed | cluster ws (expiring) | warm; thread CLOSED (row #119) → cull candidate |
| `results/run_20260804_frozeng/` | `thinkpad` | 6.5 GB | from seed | cluster ws (expiring) | warm |
| `results/run_20260805_n2sel1d/` | `thinkpad` | 4.6 GB | from seed | cluster ws (expiring) | warm |
| `results/run_20260620_seed500_phase50/` | `thinkpad` | 2.3 GB | n/a — **pre-2026-06-20 RETIRED era** | — | **cold — safe to delete** (retired by the mass-convention + L_cat banner above) |

Size is dominated by `posteriors_with_bh_mass/` per-h JSONs (~150–250 MB each × 41 h-values per
venue); the CRB CSVs and verdict JSONs that actually carry the claims are a rounding error by
comparison.

### Storage decision put to the author

Three deadlines collide: the disk is 82% full even after the dedup, the cluster workspace
`emri` **expires 2026-09-23 with 0 extensions left**, and 159 GB currently exists in exactly one
place. Recommendation:

1. **Buy a ≥2 TB external SSD/HDD.** Total project footprint is ~526 GB and the campaign51 tree
   grows with every wave; 1 TB would be tight within months. This is the only option that solves
   the cluster-expiry deadline and the no-backup finding at the same time.
2. **First evacuation, in priority order:** `~/emri-archive/` (159 GB, no second copy) →
   everything that must come off the cluster before 2026-09-23 → `results/_archive/` warm blocks.
3. **Pulling to another machine is not a substitute** unless that machine has ≥600 GB spare and
   is itself backed up; otherwise it just relocates the single-point-of-failure.

### Off-device storage options (assessed 2026-09-02)

The question "should this go on a big online filesystem instead?" resolves differently for the
three classes of online storage — the instinct that online storage is "usually used for other
things" is right about consumer drives and wrong about research archives.

| Option | Fit for this data | Verdict |
|---|---|---|
| **Institutional research archive** (KIT LSDF / bwDataArchive-class tape+disk for BW research data) | Purpose-built for exactly this: write-once, read-rarely, multi-year retention of research outputs; typically free at project scale; off-site by construction | ✅ **Best durable copy.** Also the intended answer to "copy to persistent storage before the workspace expires" |
| **Object storage** (B2 / Wasabi / S3 Glacier-class) | Technically fine; ~500 GB–1 TB is cheap to store. Watch egress pricing — free/cheap egress tiers matter because a restore pulls the whole 500 GB | 🟡 Workable fallback if no institutional option |
| **Consumer sync drive** (Dropbox / Drive / OneDrive / bwSync&Share) | Poor fit: sync clients degrade badly on the hundreds of thousands of small per-h JSONs; the client wants to mirror data back onto the already-full laptop; quotas are far below 500 GB | ❌ **Not this.** This is the class that is "used for other things" — active documents and sharing, not bulk archive |

**Online storage is complementary to the external disk, not a substitute.** They fail differently:
a local disk gives fast restores but shares a desk (and a theft/fire/coffee radius) with the
laptop; an archive is off-site and durable but slow to pull 500 GB back. Holding both, plus the
working copy on `thinkpad`, is the standard 3-2-1 arrangement (3 copies, 2 media, 1 off-site) and
is the recommendation.

**⚠️ Open action — identify the institutional archive.** `cluster/README.md:185` instructs
"copy to persistent storage before expiration" but **never names what that storage is**, and no
device tag for it exists in the Device Registry. With the `emri` workspace expiring 2026-09-23,
resolving this is time-critical: ask the bwHPC/KIT support desk what long-term research-data
archive the account is entitled to, then add it to the Device Registry as its own tag.

**Cheap wins available without buying anything** (~16 GB, author sign-off required — none are
deletable by an agent unilaterally since `results/` is fully gitignored and therefore
unrecoverable):
`run_20260620_seed500_phase50` (2.3 GB, retired era) · `run_20260817_fusion_counterfactual`
(13 GB, thread closed at row #119) · stray `.json.*` partial-write temp files across
`posteriors_with_bh_mass/` dirs.
