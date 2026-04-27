# Data Inventory

Tracks all simulation datasets and their evaluation status.
**Update this file whenever a new dataset is generated or a pipeline change is applied.**

---

## Pipeline Change Checklist

When any trigger file changes, mark affected datasets as stale and re-run the
corresponding tier before reporting results.

| Tier | Trigger files / changes | Action required |
|------|------------------------|-----------------|
| **Re-simulate** (CRBs invalid) | `LISA_configuration.py` PSD formula · SNR threshold in `constants.py` · Fisher stencil or ε in `parameter_estimation.py` · waveform params passed to `few` | Re-run GPU simulation on cluster; old CRBs are **stale** |
| **Re-prepare** (prepared CSV invalid) | `scripts/prepare_detections.py` · sampling method · SNR pre-filter applied at prepare time | Re-run `prepare_detections.py` on raw CRBs |
| **Re-migrate** (coord frame invalid) | `galaxy_catalogue/handler.py` BallTree frame or angle convention · `scripts/migrate_crb_to_ecliptic.py` transform | Re-apply migration; mark `_coord_frame` entries stale |
| **Re-evaluate** (posterior invalid) | `bayesian_inference/bayesian_statistics.py` · `single_host_likelihood` · D(h) normalisation · injection files · h-grid in `cluster/evaluate.sbatch` | Re-run `evaluate.sbatch`; old `posteriors/` are **stale** |
| **Re-figure** (figures invalid) | Any plotting code · `--generate_figures` pipeline | Re-run `--generate_figures`; old PDFs are **stale** |

---

## Dataset Registry

### prod-seed200-20260401 *(canonical large set)*

| Property | Value |
|----------|-------|
| **Location (cluster)** | `/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260401_seed200/` |
| **Location (local)** | `simulations/prepared_cramer_rao_bounds.csv` *(partial — 542 of 4 497 rows)* |
| **Simulation date** | 2026-04-01 / 2026-04-02 |
| **Git commit (simulation)** | `a56e30de` — v1.3 milestone roadmap |
| **SLURM tasks** | 99 GPU tasks, `gpu_h100`, seed 200 |
| **SNR threshold (simulation)** | 15 |
| **Total prepared rows** | 4 497 |
| **Rows with SNR ≥ 20** | 424 *(used by evaluation at runtime)* |
| **Confusion noise (Phase 9)** | ✅ included |
| **5-point stencil (Phase 10)** | ✅ included |
| **Coordinate frame** | ⬜ equatorial ICRS *(needs `migrate_crb_to_ecliptic.py`)* |
| **Ecliptic migration** | ⬜ NOT YET applied |
| **Evaluation status** | ⬜ Not evaluated post-migration |

**Next action:** run `scripts/migrate_crb_to_ecliptic.py` on cluster, then submit `cluster/evaluate.sbatch` with `RUN_DIR` pointing to this workspace.

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

## Injection Data

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

| Date | Dataset | Git commit | h-grid | SNR≥20 events | MAP | Notes |
|------|---------|-----------|--------|--------------|-----|-------|
| 2026-04-24 | local-phase43-verification (pre-fix) | pre-`a2df67b` | 38-pt | 60 | 0.860 | Equatorial frame — biased |
| 2026-04-27 | local-phase43-verification | `a2df67bf` | single h=0.73 | 60 | 0.730 | Phase 43-03 verification only |
| *(pending)* | prod-seed200-20260401 (post-migration) | HEAD | 38-pt | 424 | — | **Next evaluation target** |
