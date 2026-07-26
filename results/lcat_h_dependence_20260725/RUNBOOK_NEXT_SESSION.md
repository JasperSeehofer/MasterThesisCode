# Runbook — post-closure validation campaign (written 2026-07-26)

Context handoff for a fresh session. Everything below is mechanical: submit, monitor,
read out against the pre-registered criteria. Full scientific chain:
`OVERNIGHT_REPORT_20260726.md` → `DERIVATION_GENERATOR_CONSISTENT_NORM.md` →
`DERIVATION_ZRESOLVED_SURVIVAL.md` → issue #30 comments (2026-07-25/26).

## State at handoff

- Branch `physics/absolute-mass-marginal` (local + origin + cluster all @ `ca0ee3c`+).
  Production candidate config: `--normalization_mode generator_marginal --pdet_z_resolved`.
  1017+ tests green. All modes flag-gated; repo defaults untouched (`volume_deconv`).
- seed1000 (deep venue, 3454 ev): 7-pt probes → MAP 0.73 both channels
  (`v1_probe_genmarg/PROBE_RESULTS.md`, `zres_probe_20260726/PROBE_RESULTS.md`);
  13-pt dense core (`densecore_probe/`) → sub-grid parabolic **MAP ≈ 0.7304 both
  channels, curvature σ ≈ 0.00025–0.0003** (1D lnP: −206/−151 at ±0.005).
  ⚠ Width caveat: curvature width only; history of over-tight widths as symptoms
  (H0_BIAS_RESOLUTION §3.14); validated only by the multi-seed χ² below.
- seed600 shallow A/B (jobs 6043672/3, combines 6043805/6): vdeconv 0.745/0.755;
  absolute_marginal 0.775/**0.86 RAIL** → V1-alone fails shallow (recorded in
  `SEED600_GATE_REGISTRATION.md`). Third arm (production stack, jobs 6044088/9)
  = the registered gate; verdict appended to that file when read out.
- Venue inputs pre-staged locally: `results/campaign_phase2_runs/run_*_seed{600,900,1000,2000,3000,90000}/simulations/`.
- Cluster workspace expires **2026-09-23** (last extension used).

## Pre-registered multi-seed criteria (register BEFORE readout — this section IS the registration)

Five deep venues: seeds 900, 1000, 2000, 3000, 90000 (era-consistent Phase-2 CRBs).
Config: `generator_marginal + --pdet_z_resolved`, 41-pt grid, physics-floor combine.
1. **Bias**: the ensemble mean MAP minus 0.73, tested against the ensemble scatter
   (t-test, 5 seeds). PASS if |mean bias| < 2·SEM.
2. **Width validity**: χ² = Σ((MAP_s − mean)/σ_s)² over seeds vs χ²(4 dof). If the
   per-venue curvature σ underestimates (χ² ≫ 9.5), report the empirical seed scatter
   as THE uncertainty and flag the curvature width as invalid — do not quote it.
3. **Per-venue sanity**: interior MAP, no rails, n_used accounting clean, both channels.
4. NO estimator changes in response to these numbers without a fresh /physics-change
   cycle — this is measurement, not tuning.

## Exact commands

Cluster (after `ssh bwunicluster`, repo @ branch, preflight READY):
```bash
WS=$(ws_find emri)
# (a) seed1000 full 41-pt grid, production stack:
R=$WS/run_2026MMDD_seed1000_fullgrid; mkdir -p $R/simulations $R/logs
SRC=$WS/run_20260703_seed1000/simulations
for f in injections cramer_rao_bounds.csv prepared_cramer_rao_bounds.csv prepared_cramer_rao_bounds.meta.json; do ln -sfn $SRC/$f $R/simulations/$f; done
JE=$(sbatch --parsable --array=0-40 --time=03:00:00 \
  --output=$R/logs/evaluate_%A_%a.out --error=$R/logs/evaluate_%A_%a.err \
  --export=ALL,RUN_DIR=$R,EVAL_SEED=1000,NORMALIZATION_MODE=generator_marginal,PDET_Z_RESOLVED=yes \
  cluster/evaluate.sbatch)
sbatch --dependency=afterok:$JE --output=$R/logs/combine_%j.out --error=$R/logs/combine_%j.err \
  --export=ALL,RUN_DIR=$R cluster/combine.sbatch
# (b) same pattern per multi-seed venue (SRC per seed; EVAL_SEED per seed):
#     seed900:  run_20260703_seed900   | seed2000: run_20260707_seed2000
#     seed3000: run_20260707_seed3000  | seed90000: run_20260707_seed90000
# NOTE: all five are DEEP pools — no ALLOW_LOW_PDET_COVERAGE needed (that env is
# ONLY for the shallow seed600 venue).
```
Monitor: watch the EVAL arrays for FAILED tasks AND `DependencyNeverSatisfied`
zombie combines (both bit this session), not just combine terminal states.
Retrieval: rsync per-run `simulations/posteriors*/combined_posterior.json` + logs
(EXCLUDE per-h 2D JSONs — ~150 MB × 41 each; that filled the disk once already;
`df -h` + remote `du -sh` before any bulk pull; `find <dir> -xtype l` after).

## Decision tree after multi-seed readout

- All criteria PASS → post closure verdict on issue #30, close it; update
  DATA_INVENTORY (new canonical config), STATE.md, ROADMAP; merge PR chain
  (#37 then a PR for physics/absolute-mass-marginal); campaign NO-GO lifts;
  paper RESULT-PENDING numbers become fillable.
- Bias PASS but width χ² FAIL → uncertainty = empirical seed scatter; paper quotes
  that; investigate width model as a separate (non-blocking) issue.
- Bias FAIL → new mechanism hunt; the per-mechanism instrumentation in
  `D1_EMPIRICAL_DECOMPOSITION` scripts + `E1` scripts is the starting toolkit.
- seed600 third-arm gate FAIL → production adoption blocked; first suspect per
  derivation risk 4 (low-z weak-candidate events under the sharper numerator);
  diagnose BEFORE the multi-seed submission if it failed by > the tolerance.

## Open ends (not blocking, tracked)

- Real-data mode semantics: point/point is generator-exact for THIS mock only;
  real GLADE+ requires the photo-z kernel — mode choice must be re-derived for
  real data (paper methods section note).
- `f9c58f4` smearing flag: retained diagnostic; moot under generator_marginal.
- P–P harness impostor-capable extension (worktree branch @ 7c513dd) — needed
  before the harness can gate catalogue-association estimators.
- Issues #23/#24/#25/#26/#27 remain open; #36 (combine n_events_empty cosmetic).
- Old empty injection dirs on the workspace (2026-03-31 era) can be cleaned.

## SUBMITTED 2026-07-26 (this session — read out only)

Five-seed production-stack campaign (generator_marginal + --pdet_z_resolved, 41-pt grid):

| Seed | RUN_DIR | eval job | combine job |
|---|---|---|---|
| 1000 | run_20260726_seed1000_prodstack | 6044799 | 6044800 |
| 900 | run_20260726_seed900_prodstack | 6044801 | 6044802 |
| 2000 | run_20260726_seed2000_prodstack | 6044803 | 6044804 |
| 3000 | run_20260726_seed3000_prodstack | 6044805 | 6044806 |
| 90000 | run_20260726_seed90000_prodstack | 6044807 | 6044808 |

Code @ 6dae9d3. Read out combined_posterior.json per run/channel against the
pre-registered multi-seed criteria above (bias t-test; width χ²; per-venue sanity).
seed600 gate verdict: SEED600_GATE_REGISTRATION.md (MAP PASS; n_used deviation =
diagnosed benign risk-4 edge case, 3/3355 shallow-only — deep venues measured
0 hard zeros on 3454 events). Watch for FAILED eval tasks and zombie
DependencyNeverSatisfied combines, not just combine states.
