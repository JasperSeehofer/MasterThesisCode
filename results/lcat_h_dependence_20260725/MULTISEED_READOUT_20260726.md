# Multi-seed production-stack readout — 2026-07-26

Readout of the five-seed campaign (jobs 6044799–6044808, code @ `6dae9d3`,
config `generator_marginal + --pdet_z_resolved`, 41-pt grid 0.60–0.86,
physics-floor combine) against the pre-registered criteria in
`RUNBOOK_NEXT_SESSION.md` §"Pre-registered multi-seed criteria". Criteria were
registered before submission; no estimator changes were made in response to
these numbers.

## Job outcome

All 5×41 eval array tasks COMPLETED (exit 0:0); all 5 combines COMPLETED; no
FAILED tasks, no DependencyNeverSatisfied zombies. Results retrieved to
`results/campaign_phase2_runs/run_20260726_seed*_prodstack/` (combined
posteriors + logs only; per-h 2D JSONs excluded).

## Per-venue results

Sub-grid MAP = parabolic fit to lnP through the peak and its two neighbours;
σ = curvature width from the same fit.

| Seed | n_events | base MAP | base σ | bh_mass MAP | bh_mass σ | Sanity |
|---|---|---|---|---|---|---|
| 1000 | 3454 | 0.7304 | 0.00026 | 0.7304 | 0.00027 | PASS |
| 2000 | 3254 | 0.7300 | 0.00026 | 0.7301 | 0.00026 | PASS |
| 3000 | 3314 | 0.7297 | 0.00023 | 0.7298 | 0.00023 | PASS |
| 90000 | 20 | 0.7287 | 0.00326 | 0.7296 | 0.00297 | PASS |
| 900 | 20 | **0.86 RAIL** (grid edge) | — | 0.8547 | 0.01115 | **FAIL (crit. 3)** |

n_used accounting clean everywhere: n_used = n_total, n_excluded = 0,
n_empty = 0 (base) / 2 (bh_mass channel, all venues — the known constant
2-event bh_mass exclusion, cf. seed600 diagnosis).

## Seed900 rail — diagnosed cause: invalid injection-pool provenance

- `run_20260703_seed900/simulations/injections` symlinks to a bespoke one-off
  pool `injection_20260703-112746_seed46910/` (4 task CSVs ≈ 204 injections),
  **not** the canonical `injection_pool_depth15_50k/` (500 task CSVs, 50k
  injections) used by seeds 1000/2000/3000/90000.
- Its z-resolved survival build logs: `node ESS min/median = 6/55`,
  **418/726 sky-band cells (57.6%) below the ESS floor** — severely
  undersampled completion estimator. Reference seed90000 (same 20-event venue
  size, canonical pool): ESS min/median = 211/3944, 0/726 cells below floor,
  MAP interior at 0.7287.
- Warning-volume comparison rules out the alternative explanation: seed90000
  has *more* "quadrature weight outside P_det grid" warnings at high h
  (400 vs 222 at h=0.86) yet does not rail. The discriminator is the
  undersampled survival estimator, not out-of-grid quadrature.
- Conclusion: the seed900 point is an **invalid measurement of the estimator**
  (defective venue input), not evidence of estimator bias. It rails HIGH the
  same way the deep-venue Option-A composition violation did — an unreliable
  completion correction, here from undersampling.

## Pre-registered criteria

**Registered 5-seed set, base channel: NOT COMPUTABLE as registered** —
seed900 rails (criterion 3 FAIL), so the 5-seed bias/width statistics cannot
be formed on the base channel. Verdict on the registered set: **QUALIFIED
FAIL — one venue invalid for a diagnosed input-provenance defect unrelated to
the estimator under test.**

Supplementary readout on the 4 provenance-valid seeds (1000/2000/3000/90000):

| Criterion | base | bh_mass |
|---|---|---|
| 1. Bias: mean − 0.73 | −0.00030 ± 0.00035 (SEM); t = −0.85, p = 0.46 → **PASS** | −0.00003 ± 0.00018; t = −0.15, p = 0.89 → **PASS** |
| 2. Width χ² (3 dof, crit 9.3) | 8.0 → **curvature widths VALID** (marginal) | 3.7 → **VALID** |
| 3. Sanity | all interior, no rails, accounting clean → **PASS** | same → **PASS** |

Empirical seed scatter (std of MAPs): 0.00071 (base), 0.00036 (bh_mass).
Note the base-channel χ² = 8.0 is close to the 9.3 cut — driven by seed90000
(20 events, MAP 0.7287, 0.4σ_scatter low but 0.5σ_s ≈ within its own width);
with n=4 this is not evidence of width failure, but the width claim should be
re-checked when seed900 is re-run (n=5 restores the registered test).

For the registered 5-seed bh_mass channel (where seed900 is interior at
0.8547): bias PASS only vacuously (SEM inflated to 0.025 by the invalid
point); width χ² = 30137 → catastrophically inconsistent — consistent with
seed900 being a defective outlier, not with a width-model failure (the 4-seed
χ² = 3.7).

## Disposition (per decision tree)

- Not the clean "All criteria PASS" branch: **issue #30 stays open; no PR
  merges; campaign NO-GO not lifted** pending author decision.
- Author decision required: re-point `run_20260703_seed900/simulations/injections`
  at the canonical `injection_pool_depth15_50k` pool (generator-consistency of
  that pool with the seed900 venue generation config must be confirmed) and
  re-run the seed900 eval+combine, restoring the registered 5-seed test.
  This is an input-data fix, not an estimator change — but because it follows
  a look at the data, it is recorded here and both readouts (registered-5 and
  valid-4) will be reported regardless of the re-run outcome.
- On the 4 provenance-valid venues the production stack shows **no detectable
  bias at the 4×10⁻⁴ level** and **valid curvature widths** — the first
  configuration in the campaign history to pass both.

## Log-hygiene note (non-blocking)

"quadrature weight outside P_det grid" warning spam scales with h-grid index
(≈5×10⁵ lines in seed1000 task 40; logs 0.8–1.3 GB/run). Consider a
once-per-event or counter-summary emission before the next large campaign.
Local disk after retrieval: 3.0 GB free (99%) — clean before any further pulls.
