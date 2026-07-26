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
- **Repo:** `results/commission_20260701/redteam/posteriors_per_mode/` — per-mode combined
  posteriors (prod 0.86 / prod_global 0.60 / local_ratio 0.73 / volume_deconv 0.73 / catonly 0.73)
  + `crux_results{,_fixed}.json` (commit `1f0e371`).
- **Home:** `~/data-backups/seed600_local_derail_20260702/` (3.8 GB: full working dirs incl. the
  474 MB with-BH-mass per-event posteriors, the 494-event CRB subsample, the fixed 8-col catalogue copy).
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
| **File** | `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv` (headerless, **1.68 GB**, 22 641 048 rows) |
| **Schema** | **8 columns** (order = `_reduced_catalog_column_names()`): RA_deg, Dec_deg, B_mag, **z_cmb**, z_error (PV-correction error folded in quadrature, 0.0015 floor), stellar_mass, stellar_mass_err, z_flag (1=photo-z, 3=spec-z; trailing) |
| **Frame** | **z_cmb** (CMB frame) since `18e9608` (2026-07-02 rebuild; 99.9% rows shifted, median \|Δz\| 6e-4 — `.planning/gate/GATE_SIGNOFF.md:27`) |
| **Depth** | **Full-depth** (no z cut in the writer; max z ≈ 7.03). Effective load-time depth = `Model1CrossCheck.max_redshift` = 1.5 via `_get_pruned_galaxy_catalog`. `GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT` is documentation-only |
| **Rebuild** | `results/commission_20260701/scratch/rebuild_catalog.py` from repo root; **move the old CSV aside first** (writer appends, `mode="a"`); ~77 s full GLADE+ pass on the dev box |
| **Source** | `master_thesis_code/galaxy_catalogue/GLADE+.txt` (6.4 GB, dev box ONLY — cluster cannot rebuild; staging is rsync of the reduced CSV per `/cluster` skill) |
| **Superseded** | `.zhelio_20260702` (z_helio 8-col, 2026-07-01), `.stale6col_mar28` (6-col) — backups next to the live file, RETIRED |
| **Coupled artifact** | `m_th_map_nside32.npy` (frozen per-pixel m_th, C1: byte-identical on injection + inference sides). Built from the full flag-{1,3} catalogue → **unchanged by the 2026-07-03 depth constants** (no CSV rebuild occurred); MUST be regenerated atomically on both sides if the CSV content ever changes |

---

## Dataset Registry

### phase45-seed200-20260501 *(current canonical, post-Tier-3 fix)*

| Property | Value |
|----------|-------|
| **Location (cluster)** | `/pfs/work9/workspace/scratch/st_ac147838-emri/run_phase45_20260501/simulations/` |
| **Location (local)** | `simulations/cluster_run_phase45_20260501/cramer_rao_bounds.csv` |
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
| **Location (cluster)** | `/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260504_seed300_extension/` |
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
| **Location (cluster)** | `/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260409_seed200000/` |
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
