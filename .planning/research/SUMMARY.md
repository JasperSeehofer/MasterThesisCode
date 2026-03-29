# Project Research Summary

**Project:** v1.2 Production Campaign & Physics Corrections
**Domain:** EMRI gravitational wave parameter estimation — physics corrections and production HPC campaign
**Researched:** 2026-03-29
**Confidence:** HIGH

## Executive Summary

This milestone corrects two known physics errors in the EMRI simulation pipeline — an O(epsilon) forward-difference Fisher matrix derivative and a missing galactic confusion noise term in the LISA PSD — then runs a production-scale simulation campaign (100+ tasks) and evaluates the full H0 posterior over [0.6, 0.9]. All four deliverables are implementable with the existing stack and infrastructure: no new Python packages are needed, and the cluster scripts established in v1.1 scale directly. The primary constraint is sequencing: physics corrections must precede the production campaign, or the campaign wastes GPU hours producing scientifically invalid Cramér-Rao bounds.

The recommended approach is strictly sequential: (1) add galactic confusion noise to the PSD (self-contained additive term, lower risk), (2) wire the existing `five_point_stencil_derivative()` method into `compute_fisher_information_matrix()` (method already exists at lines 187-243, needs integration and cleanup), (3) run a small validation campaign to calibrate timeouts and check detection rates, (4) run the full 100-task production campaign, (5) evaluate the H0 posterior sweep using a new SLURM array job over h-values in [0.6, 0.9]. The H0 sweep requires no changes to `BayesianStatistics` — only a new `cluster/sweep_h0.sbatch` and `scripts/combine_posteriors.py`.

The key risks are: the 30-second CRB timeout in `main.py` will fire on nearly every event after the 4x waveform increase from the 5-point stencil (must be raised to 120-300s before any campaign run); the improved stencil may expose Fisher matrix ill-conditioning that forward-difference was inadvertently smoothing (condition number monitoring needed); confusion noise will reduce detection yield significantly (campaign size must be calibrated after physics corrections are validated, not before). Running the production campaign before completing both physics fixes is the single most costly mistake to avoid.

## Key Findings

### Recommended Stack

No new dependencies are required for v1.2. The existing stack — NumPy/CuPy for array math, SciPy for integration, pandas for CSV I/O, the `few` waveform generator, and the bwUniCluster SLURM infrastructure — handles all four features. The `five_point_stencil_derivative()` method (parameter_estimation.py:187-243) already implements the correct O(epsilon^4) formula using CuPy array arithmetic. The galactic confusion noise constants (constants.py:74-81) already cite Babak et al. (2023) arXiv:2303.15929 Eq. 17 and need only to be imported and used in `LISA_configuration.py`. The H0 sweep requires only a new shell script and a ~50-line Python combiner.

**Core technologies:**
- `numpy` / `cupy-cuda12x`: array computation for stencil derivatives and PSD — already used everywhere, no changes to imports needed
- `few` (fastemriwaveforms): EMRI waveform generation — 4x more calls per Fisher matrix with 5-point stencil; GPU acceleration absorbs the increase within SLURM walltime limits
- SLURM array jobs: parallelism for both the simulation campaign and the H0 sweep — pattern already proven in v1.1
- `scipy.stats` / `scipy.integrate`: unchanged; H0 sweep evaluates the same posterior code per h-value

**Configuration changes required (not new dependencies):**
- `main.py` line 332: `signal.alarm(30)` must increase to at least 120s for the CRB phase — without this, the stencil upgrade produces near-zero CRB detections due to timeout
- `LISA_configuration.py`: `power_spectral_density_a_channel()` gains `T_obs=5.0` parameter; PSD cache key becomes `(n, T_obs)` to correctly handle the 1-year SNR pre-check vs 5-year full computation

### Expected Features

**Must have (table stakes — blocks thesis without these):**
- 5-point stencil Fisher derivatives — O(epsilon) forward-diff produces inaccurate CRBs; Vallisneri (2008) explicitly warns against it; the 10% d_L threshold was already raised from 5% as a workaround; method already exists at lines 187-243 and needs wiring
- Galactic confusion noise in PSD — dominates LISA sensitivity at 0.1-3 mHz where EMRIs live; omitting it systematically overestimates SNR; constants already defined in constants.py:74-81
- Production simulation campaign (100 tasks, 50 steps each) — the v1.1 smoke test (3 tasks, 20 detections) is statistically insufficient for H0 inference; need O(1000+) detections
- Full H0 posterior sweep [0.6, 0.9] — current code evaluates at single h-value (0.73); a posterior requires likelihood over a grid

**Should have (improves thesis quality):**
- Tighten d_L error threshold from 10% to 5% — the 10% threshold was a forward-diff workaround; evaluate after stencil is validated and campaign data analyzed
- Campaign diagnostic plots — SNR distribution, d_L error distribution, detection rate vs redshift using existing `plotting/` infrastructure
- Posterior combination script with credible intervals — automate assembling per-h JSONs into normalized H0 posterior

**Defer to future work:**
- wCDM dark energy model fix — changes the physics model substantially; invalidates comparisons with prior results
- Planck 2018 cosmology update (Omega_m=0.3153, H=0.6736) — would invalidate all prior validation
- Galaxy redshift uncertainty fix — LOW priority, affects z > 0.14 near detection horizon only
- Pipeline A (BayesianInference) hardcoded 10% sigma — dev cross-check only; Pipeline B is production

### Architecture Approach

All v1.2 work is in-place modification of existing components — no new architectural layers, no new data pipelines, no new module boundaries. The two physics changes are localized: the stencil swap is entirely within `compute_fisher_information_matrix()` (one call site replaced with a loop); the confusion noise is entirely within `LISA_configuration.py` (one new method, one modified method). The H0 sweep adds two small files to existing directories. The existing boundary between `parameter_estimation.py` (computation) and `LISA_configuration.py` (noise model) is correct and must not be crossed.

**Modified components:**
1. `parameter_estimation.py:compute_fisher_information_matrix()` — replace single `finite_difference_derivative()` call with per-parameter loop over `five_point_stencil_derivative()`; update `_get_cached_psd` cache key to `(n, T_obs)`
2. `LISA_configuration.py:power_spectral_density_a_channel()` — add `S_conf(f, T_obs)` static method and sum it into PSD; thread `T_obs` through the call chain
3. `main.py` line 332 — increase `signal.alarm(30)` to 120-300s for CRB phase

**New files:**
1. `cluster/sweep_h0.sbatch` — SLURM array job mapping task IDs to h-values in [0.6, 0.9]
2. `scripts/combine_posteriors.py` — read per-h JSON results, normalize, compute credible intervals

### Critical Pitfalls

1. **30-second CRB timeout fires on every event after stencil upgrade** — forward-diff needs ~15 waveforms (7-30s on H100); 5-point needs ~56 (28-112s). Increase `signal.alarm(30)` to at least 120s before running any campaign. Failure to do this causes near-total loss of CRB detections with no error message — events are silently skipped.

2. **Physics corrections must precede the production campaign** — running 100+ tasks with the broken forward-diff Fisher matrix wastes 100+ GPU-hours on scientifically invalid CRBs that must be regenerated. There is no shortcut: validate both fixes first with a 3-5 task verification run.

3. **`afterok` SLURM dependency kills the entire pipeline on any single task failure** — at 100 tasks with ~5% per-task failure rate, P(at least one failure) > 99%. Change `submit_pipeline.sh` merge dependency to `afterany` so merge runs even if some tasks fail.

4. **Confusion noise reduces detection yield; campaign size must be calibrated after PSD fix** — confusion noise dominates 0.1-3 mHz and will decrease SNR across the EMRI band. Run a 5-task test with confusion noise before committing to 100-task campaign parameters.

5. **Better derivatives may expose Fisher matrix ill-conditioning** — the forward-difference implicitly smoothed near-degeneracies between phase parameters (Phi_phi0, Phi_theta0, Phi_r0). After switching to 5-point, log condition numbers; if `cond(F) > 1e12`, flag the event. Replace deprecated `np.matrix(...).I` with `np.linalg.inv()`.

## Implications for Roadmap

The dependency chain is strictly linear: physics corrections -> validation -> production campaign -> H0 sweep. Parallelizing any of these risks producing invalid data or wasting cluster resources.

### Phase 1: Galactic Confusion Noise in PSD
**Rationale:** Lower-risk physics fix; self-contained additive term; constants already defined in codebase. Implement first to establish the corrected noise floor before any further numerical work. Independent of the stencil change, enabling sequential validation of each physics fix.
**Delivers:** Corrected LISA PSD including galactic confusion foreground for T_obs=5yr; PSD cache keyed on `(n, T_obs)`; regression test confirming PSD with confusion > PSD without at 1 mHz.
**Addresses:** Table-stakes feature "Galactic confusion noise in PSD"; Babak et al. (2023) Eq. 17.
**Avoids:** Pitfall 5 (SNR drop not quantified before campaign sizing), Pitfall 6 (T_obs dependence in cache), Pitfall 13 (cache not invalidated).
**Research flag:** Standard pattern — additive noise term with documented formula. No additional research needed during planning.

### Phase 2: Five-Point Stencil Fisher Derivatives
**Rationale:** Higher-complexity physics fix that must follow confusion noise so the two changes can be independently validated. The method already exists; this phase wires it in, handles the signature mismatch, increases the CRB timeout, and replaces `print()` with logging.
**Delivers:** O(epsilon^4) Fisher matrix derivatives replacing O(epsilon) forward-diff; CRB timeout increased to 120-300s; condition number logging; `print()` replaced with `_LOGGER`; regression test comparing forward-diff vs stencil derivatives on identical parameter set.
**Addresses:** Known Bug 4 (Fisher matrix uses forward-diff); Vallisneri (2008) recommendation.
**Avoids:** Pitfall 3 (timeout kills events), Pitfall 4 (ill-conditioning), Pitfall 2 (sign convention drift), Pitfall 12 (np.matrix deprecation).
**Research flag:** Standard pattern — method already correct; wiring and cleanup only. Physics Change Protocol required.

### Phase 3: Validation Campaign (3-5 Tasks)
**Rationale:** Before committing 100+ GPU-hours to a production campaign, verify that physics corrections produce valid results: detection rates are not catastrophically reduced, Fisher matrices are well-conditioned, d_L error distributions are as expected, and per-task wall time fits within SLURM limits.
**Delivers:** Empirical detection rate with corrected physics; condition number distribution; baseline comparison against v1.1 smoke test; calibrated `--time` limit for Phase 4.
**Addresses:** Campaign sizing risk; Pitfall 1 (stencil bounds check yield reduction), Pitfall 11 (d_L threshold recalibration timing).
**Avoids:** Anti-Pattern 2 (production campaign before physics validation).
**Research flag:** No research needed — operational validation run.

### Phase 4: Production Simulation Campaign (100+ Tasks)
**Rationale:** With corrected physics validated and timeouts calibrated, run the campaign needed for statistically meaningful H0 inference. Parameters: 100 tasks, 50 steps each, seed distinct from smoke test.
**Delivers:** O(1000-2000) usable CRB detections in `simulations/cramer_rao_bounds.csv`; `run_metadata.json` with git commit and seed; per-task CSVs merged by `emri-merge`.
**Addresses:** Table-stakes feature "Production simulation campaign"; expected 2000-3000 detections above SNR>20 with 40-60% detection rate (lower with confusion noise — calibrated in Phase 3).
**Avoids:** Pitfall 9 (symlink collision — no simultaneous campaigns), Pitfall 10 (change `afterok` to `afterany`), Pitfall 15 (workspace expiration — copy results immediately after merge).
**Research flag:** No research needed — operational scaling of proven infrastructure.

### Phase 5: H0 Posterior Sweep [0.6, 0.9]
**Rationale:** After the production campaign provides sufficient CRBs, evaluate the Bayesian posterior over a grid of H0 values. SLURM array approach (one task per h-value) is strictly superior to an in-process loop: parallel execution, no changes to BayesianStatistics, proven infrastructure pattern.
**Delivers:** `cluster/sweep_h0.sbatch` array job; `scripts/combine_posteriors.py`; normalized H0 posterior JSON with MAP estimate and 68%/95% credible intervals; posterior plot.
**Addresses:** Table-stakes feature "Full H0 posterior sweep [0.6, 0.9]"; recommended Option A from STACK.md and ARCHITECTURE.md.
**Avoids:** Pitfall 8 (redundant setup per h-value via SLURM parallelism), Pitfall 14 (filename collision — use 4+ decimal places in h-value filenames), Anti-Pattern 3 (in-process H0 loop).
**Research flag:** The `combine_posteriors.py` normalization and credible interval computation is straightforward (standard numpy/scipy operations), but a quick review of Bayesian posterior combination patterns is recommended.

### Phase 6: d_L Threshold Recalibration (Optional)
**Rationale:** After Phase 4 results are available, analyze the distribution of `d_L_uncertainty / d_L`. If the median drops well below 10% with the stencil upgrade, tighten `FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD` from 0.10 to 0.05 to improve H0 posterior quality.
**Delivers:** Updated threshold in `bayesian_statistics.py:58` with justified value from empirical data; re-run of Phase 5 if threshold changes materially.
**Addresses:** FEATURES.md "Should have" — tightened d_L threshold.
**Avoids:** Pitfall 11 (premature tightening before empirical validation).
**Research flag:** No research needed — data-driven decision from Phase 4 outputs.

### Phase Ordering Rationale

- **Confusion noise before stencil:** Lower risk, self-contained, enables independent validation. Bundling both physics changes in one commit makes it impossible to attribute result changes to either fix.
- **Both physics fixes before production campaign:** The dependency chain is explicit; campaign results are scientifically invalid with the known-wrong physics. This is the single most important sequencing constraint.
- **Validation campaign before production:** Calibrates timeout, detection rate, storage requirements. Prevents discovering configuration problems after wasting 100+ GPU-hours.
- **H0 sweep after campaign:** Sweep operates on the campaign's CSV output; no other dependency.
- **d_L threshold last:** Data-driven decision requiring campaign results.

### Research Flags

Phases needing deeper research during planning:
- **Phase 5 (H0 sweep):** Confirm that combining per-h log-likelihoods and computing credible intervals is correct — standard Bayesian combination but worth a 15-minute review before implementation.

Phases with standard patterns (no research-phase needed):
- **Phase 1 (confusion noise):** Formula documented in Babak (2023) Eq. 17; constants already in codebase; additive implementation.
- **Phase 2 (5-point stencil):** Method already exists and is correct; wiring and cleanup only.
- **Phase 3 (validation campaign):** Operational run; no new patterns.
- **Phase 4 (production campaign):** Proven infrastructure; scaling only.
- **Phase 6 (d_L threshold):** Single-line change based on data analysis.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Direct codebase inspection confirms all needed components exist; zero new dependencies required |
| Features | HIGH | All four table-stakes features are defined with specific file/line references; implementation paths clear |
| Architecture | HIGH | Changes are localized; component boundaries correct; no new architectural layers needed |
| Pitfalls | HIGH | Based on direct codebase inspection + Vallisneri (2008) + Babak (2023); critical pitfalls include specific line numbers |

**Overall confidence: HIGH**

### Gaps to Address

- **Confusion noise formula source ambiguity:** constants.py cites arXiv:2303.15929 but PITFALLS.md notes the canonical parameterization may be Babak et al. (2021) arXiv:2108.01167. Verify formula against both sources during Phase 1 before writing any code. The physics is the same; the citation must be correct.
- **Validation campaign detection rate:** Actual yield with corrected physics is unknown until Phase 3 runs. FEATURES.md estimates 40-60% above SNR>20, but confusion noise may reduce this further. The 100-task campaign size is a starting point, not a firm number.
- **Fisher matrix ill-conditioning prevalence:** The risk (Pitfall 4) is real but its frequency on actual EMRI parameter draws is unknown. Phase 3 must explicitly log condition numbers to determine whether Tikhonov regularization is needed for Phase 4.
- **H0 sweep per-task wall time at production scale:** Estimated at 2-5 minutes with 16 CPUs, but this comes from single-h tests on smaller detection catalogs. Run a test evaluation on Phase 4 output before submitting the 31-task sweep.

## Sources

### Primary (HIGH confidence)
- [Vallisneri (2008), arXiv:gr-qc/0703086](https://arxiv.org/abs/gr-qc/0703086) — Fisher matrix numerical derivatives, 5-point stencil recommendation, ill-conditioning warnings
- [Babak et al. (2023), arXiv:2303.15929](https://arxiv.org/abs/2303.15929) — LISA PSD with galactic confusion noise, Eq. 17; parameterization used in constants.py
- Direct codebase inspection — `parameter_estimation.py` (lines 140-243, 335-365), `LISA_configuration.py`, `constants.py:74-81`, `bayesian_statistics.py:58`, `main.py:332`, `cluster/submit_pipeline.sh`

### Secondary (MEDIUM confidence)
- [Babak et al. (2021), arXiv:2108.01167](https://arxiv.org/abs/2108.01167) — alternative canonical reference for LISA confusion noise parameterization; may be the actual source for constants.py coefficients
- [Robson, Cornish & Liu (2019), arXiv:1803.01944](https://arxiv.org/abs/1803.01944) — LISA sensitivity curves and confusion noise background parameterization
- v1.1 smoke test results (PROJECT.md) — 20 detections from 30 events, 18 passed d_L filter; ~15-20 min per 10-step task on H100

### Tertiary (LOW confidence)
- SLURM per-task failure rate estimate (~5%) — heuristic from general HPC experience; actual rate on bwUniCluster depends on node stability and GPU partition availability

---
*Research completed: 2026-03-29*
*Ready for roadmap: yes*
