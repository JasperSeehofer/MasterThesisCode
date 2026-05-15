# Handoff — Phase 50 re-simulation campaign IN FLIGHT (2026-05-16 evening)

## TL;DR

The CRB two-population diagnosis (handoff §A from
`HANDOFF-PLOTTING-OVERHAUL-20260516.md`) is **resolved**: the row-424
boundary in `simulations/cluster_run_production_h0p73_20260506/simulations/prepared_cramer_rao_bounds.csv`
was the seed200/seed300 concatenation seam, caused by per-task emcee
under-mixing (τ_ACT ≈ 33 steps but burn_in_steps=1000 ≈ 30·τ_ACT, below
the 50·τ_ACT safety margin). Sampler fix shipped in commit `991333a`
(nwalkers=20→50, burn_in_steps=1000→10000); §3.16 doc entry in commit
`ac8eddc`; full forensics in the previous handoff §A.

**Phase 50 re-simulation campaign is submitted and running on
bwUniCluster as of this handoff.** Goals:

1. Replace the heterogeneous `seed200 ⊕ seed300_extension` production
   CRB with a single homogeneous campaign drawn from the well-mixed
   sampler.
2. Bundle handoff §B (superdense h-grid Δh=0.0005 in [0.720, 0.740],
   83 total h-values) into the same campaign so figures regenerate
   in one pass.
3. Bundle handoff §F (F4 Nadaraya-Watson p_det estimator) into the
   evaluation stage since main is on F4 (commit `d1087f1`) and the
   production CRB was generated against F1.

The next session's primary work is **harvest + analysis + figure
regen** once the four-job chain on the cluster completes.

---

## Cluster state (as of 2026-05-16T18:30 CEST)

Run directory: `/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260516_seed400_phase50/`

| Job ID | Stage | Partition | Walltime | Dependency |
|---|---|---|---|---|
| `4718691` | simulate (50-task GPU array) | `gpu_h100` | 1 h/task | none |
| `4718692` | merge | `cpu_il` | 30 min | `afterany:4718691` |
| `4718693` | evaluate (9-task array, 83-pt grid) | `cpu_il` | 45 min/task | `afterok:4718692` |
| `4718694` | combine | `cpu_il` | 10 min | `afterok:4718693` |

Submission script: `cluster/submit_resimulate_phase50.sh` (defaults
`--tasks 50 --steps 35 --seed 400`).
Evaluate sbatch: `cluster/evaluate_production_h0p73_superdense.sbatch`.
Campaign metadata file: `$RUN_DIR/campaign_metadata.json` (captured at
submission with git commit hash, params, purpose).

**Monitor:**
```bash
ssh bwunicluster 'sacct -j 4718691,4718692,4718693,4718694 --format=JobID,State,Elapsed,ExitCode'
ssh bwunicluster 'squeue -u $USER'
```

**Expected wall-clock:** ~6–12 h end-to-end, dominated by `gpu_h100`
queue depth. Sim+merge ~3–6 h, evaluate ~30–60 min, combine ~5 min.

**Expected output sizes:**
- ~1750 events (50 tasks × ~35 SNR≥20 events/task at the fix's
  well-mixed sampler — slightly more than seed300's ~21/task because
  seed300 was bottlenecked by the under-mixed mass library)
- 83 × 2 = 166 posterior JSONs (`posteriors/h_*.json` + `posteriors_with_bh_mass/h_*.json`)
- `combined_posterior.json` + `combined_posterior_with_bh_mass.json`
- `prepared_cramer_rao_bounds.csv` (~12 MB)

---

## What's bundled in this campaign

### Sampler fix (commit `991333a`, handoff §A resolution)
- `cosmological_model.py:setup_emri_events_sampler`:
  `nwalkers = 20 → 50`, `burn_in_steps = 1000 → 10000`.
- Local CPU verification: cross-seed median-M ratio 1.02 (was 2.0 on
  production CRB); cross-seed log10(M) std ≈ 0.35 (well-mixed).
- This campaign is the first end-to-end GPU test of the fix.

### Superdense h-grid (handoff §B)
- 83 h-values: 41 dense Δh=0.001 in [0.710, 0.750] (Phase 48
  resolution) + **20 new super-dense Δh=0.0005 mid-points in
  (0.720, 0.740)** + 22 wing Δh=0.010 points spanning [0.600, 0.860].
- Truth h=0.730 sits on the dense core.
- σ_boot post-H3-fix is ~0.0037 (2D); Δh_super = 0.0005 ≈ σ_boot/7,
  comfortably below the resolution floor that hit `paper_m_z_improvement.pdf`
  at HDI68 = 0.001.

### F4 Nadaraya-Watson p_det (commit `d1087f1`, handoff §F resolution)
- Was on main but production CRB was evaluated under F1 (commit `87ea7a8`).
- Re-simulation evaluates fresh, so this campaign delivers the F4
  comparison point that handoff §F asked for.
- F4 is meant to close the spiky-posterior issue diagnosed in
  Phase 49 (see `.planning/HANDOFF-PHASE49-MECHANISM-VERIFY-20260514.md`).

---

## Post-campaign workflow (when all four jobs land OK)

### 1. Verify simulation health (run before harvest)
```bash
ssh bwunicluster 'sacct -j 4718691 --format=JobID,State,Elapsed,ExitCode | head -55'
# expect: 50 COMPLETED entries (TIMEOUT on a few is OK — afterany tolerates)
ssh bwunicluster 'ls /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260516_seed400_phase50/simulations/cramer_rao_bounds_simulation_*.csv 2>/dev/null | wc -l'
# expect: 40+ per-task CSVs (allowing some timeouts)
```

### 2. Verify the sampler fix worked
Goal: confirm cross-task M-library is homogeneous (no row-N boundary).
```bash
# On cluster: per-task M medians should all be similar (~ R_emri peak ≈ 2.5e5)
ssh bwunicluster '
  for f in /pfs/work9/workspace/scratch/st_ac147838-emri/run_20260516_seed400_phase50/simulations/cramer_rao_bounds_simulation_*.csv; do
    echo -n "$(basename $f): "
    awk -F, "NR>1 {sum+=\$1; n++} END {if(n>0) printf \"med-approx-M %.0f n=%d\n\", sum/n, n}" "$f"
  done | head -20
'
```
Expected: all task means ≈ 1.5e5–3e5 with no >2× outliers. If you see
the seed200-style heavy bias (M≈4.6e5 in some tasks), the fix didn't
fully resolve and we need to investigate further.

### 3. Rsync results local
```bash
RUN_DIR=/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260516_seed400_phase50
mkdir -p simulations/cluster_run_phase50_20260516
rsync -avz bwunicluster:$RUN_DIR/simulations/ simulations/cluster_run_phase50_20260516/
# But strip the 2D galaxy_likelihoods to save bandwidth (per the prior handoff §C lesson):
rsync -avz --include="h_*.json" --include="combined_*.json" \
    bwunicluster:$RUN_DIR/simulations/posteriors/ simulations/cluster_run_phase50_20260516/posteriors/
# Repeat for posteriors_with_bh_mass after applying the strip script on cluster (if 2D files balloon).
```

### 4. Compute H0 MAP + σ_boot on the new CRB
Use the canonical loader directly:
```bash
make validate-figures   # produces canonical_combined.json + prints discrete/continuous MAP
```
Or invoke `master_thesis_code/plotting/_helpers.py::load_canonical_combined_posterior`
on `simulations/cluster_run_phase50_20260516/posteriors/` and
`posteriors_with_bh_mass/` to get MAP/σ_boot.

### 5. Compare against production (regression check)
| Metric | Production (1549) | Phase 50 (target ~1750) | Acceptance |
|---|---|---|---|
| 1D MAP | 0.7322 (Phase 48) | should be in [0.728, 0.736] | |MAP_50 − MAP_prod| ≤ σ_boot_50 + σ_boot_prod |
| 2D MAP | 0.7320 | should be in [0.728, 0.736] | same |
| σ_boot (2D) | 0.0022 | expected smaller (more events) | |
| HDI68 (2D) | 0.001 (grid floor) | should drop to <0.001 | super-dense breaks the floor |
| Convergence elbow at N≈420 | YES (artifact) | NO (homogeneous campaign) | inspect `paper_m_z_improvement.pdf` |
| Cross-task M-library variance | high (handoff §A) | low (sampler fix) | per-task medians within factor 1.5 |

### 6. Regenerate figures + interactive
```bash
make regen-figures        # 26 static figures
make regen-interactives   # 8 Plotly HTMLs
```
Then republish to GH Pages via the existing CI workflow (push to main
of the `interactive/` paths triggers the `pages` workflow per CI config).

### 7. Update `docs/H0_BIAS_RESOLUTION.md`
Add a Phase 50 results table to the Executive Summary at line 24
(currently has rows for pre-Tier-3 / post-Tier-3 / post-bridge /
post-H3 / Phase 48). Also fix the stale "12 confirmed bias sources"
text on line 28 (catalogue is at 16 entries after §3.16).

### 8. Optional: dual-table in §3.16 (pre-fix vs post-fix)
Mirror the §3.13 dual-table pattern: row 1 = production CRB
(heterogeneous, row-424 boundary, MAP=0.7322), row 2 = Phase 50
(homogeneous, no boundary, MAP=X.XXXX). Demonstrates the fix end-to-end.

---

## Open items not blocked by the campaign

- **Pin down the ~75 unaccounted production-CRB events** (rows 1475–1549
  in the current production prepared CRB). Likely a third small extension
  or rerun. Not paper-blocking, but worth resolving before submission.
  Hint: check cluster `archive` dirs and `simulations/` symlinks; the
  metadata trail is in `simulations/cluster_run_production_h0p73_20260506/simulations/archive/`.
- **`/wiki-debrief`** for the four reusable lessons surfaced this session:
  burn-in vs usage pattern; selection-effect amplification; cluster-side
  forensics via per-source awk `value_counts`; predict-and-check disproof
  pattern. User-triggered; run in this project session, not the vault.
- **Fisher frame mismatch** (memory `project_fisher_frame_mismatch.md`)
  — sky position is ecliptic but Fisher covariance is still equatorial.
  Pre-existing; not paper-blocking but worth a `/physics-change` pass
  before submission.
- **Co-author meeting follow-ups** from `docs/coauthor_meeting_2026_05_15.md`.

---

## If the campaign FAILS or returns unexpected results

### Sim job 4718691 timeouts on >10% of tasks
Resubmit just the failed array tasks with `afterany` dependency unchanged.
The merge stage uses `afterany` deliberately to tolerate partial sim
completion. ~~no rerun needed unless event count drops below ~1500.~~

### Per-task M-library still bi-modal after fix
Indicates the under-mixing diagnosis was incomplete and another mechanism
is at play. Investigation candidates:
1. emcee chain post-burn-in autocorrelation within batches (currently
   no thinning in `sample_emri_events`). Fix: add `thin_by=10`.
2. Per-task RNG state propagation to other parts of the pipeline
   (galaxy_catalog sampling, SNR threshold pre-screen).
3. Numerical issues in `R_emri` or `dN_dz_of_mass` at the chain
   boundary (M=1.2e5 piecewise kink).

### F4 posteriors look discontinuous or worse than F1
Document in handoff §F follow-up. The F1-vs-F4 comparison was the
explicit decision point from the prior handoff — getting a clean F4
result here is one of the campaign's deliverables. If F4 underperforms,
fall back to F1 by `git checkout 87ea7a8 -- master_thesis_code/bayesian_inference/`
and re-evaluating (keeps the new CRBs, swaps the estimator).

### Combined posterior MAP shifts >2σ from production 0.7322
This would be a SIGNIFICANT finding — re-simulation revealed a hidden
bias in the production result. Pause figure regen, dump full analysis
to a new debug session, and write up the implication for the paper.
Most likely cause if this happens: F4 vs F1 produces a measurable shift,
in which case we need to decide which estimator the paper reports.

---

## Memory + reference pointers

- `memory/project_crb_two_population.md` — sampler under-mixing diagnosis (concise)
- `.planning/HANDOFF-PLOTTING-OVERHAUL-20260516.md` §A — full forensics (long)
- `docs/H0_BIAS_RESOLUTION.md` §3.16 — catalogued bias-resolution entry
- `cluster/submit_resimulate_phase50.sh` — campaign submission (this session)
- `cluster/evaluate_production_h0p73_superdense.sbatch` — 83-pt h-grid
- `master_thesis_code/cosmological_model.py:setup_emri_events_sampler` — sampler-fix site
