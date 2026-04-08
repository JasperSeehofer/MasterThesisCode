---
status: diagnosed
trigger: "H0 posterior results worsen when combining detection runs from different configurations"
created: 2026-04-08T00:00:00Z
updated: 2026-04-08T00:00:00Z
---

## Current Focus

hypothesis: Multiple independent root causes contribute to H0 bias, not simple data mixing
test: Traced full pipeline from data loading through posterior combination
expecting: Identify concrete causes of "results worsen with more detections"
next_action: Present findings

## Symptoms

expected: H0 posterior should improve (narrow) with more detections
actual: Results get WORSE with more detections when combining runs
errors: No crash - numerically degraded results
reproduction: Combine CSV outputs from different detection/injection runs and run --evaluate
started: After combining data from different runs

## Eliminated

- hypothesis: CSV files from different runs have incompatible schemas
  evidence: All cramer_rao_bounds.csv files across run_v12_diagnostic, run_v12_validation, run_v12_bhmass_fix, run_v12_analytic_mz share identical column headers
  timestamp: 2026-04-08

- hypothesis: SNR threshold filtering is missing from evaluation pipeline
  evidence: bayesian_statistics.py:186-188 explicitly filters `self.cramer_rao_bounds["SNR"] >= SNR_THRESHOLD` at evaluation time. Added in commit 97180a3.
  timestamp: 2026-04-08

- hypothesis: Data files are directly concatenated without any filtering
  evidence: The evaluation pipeline reads a single prepared_cramer_rao_bounds.csv (not multiple), and applies SNR + quality filters at runtime
  timestamp: 2026-04-08

## Evidence

- timestamp: 2026-04-08
  checked: SNR_THRESHOLD history in constants.py
  found: SNR_THRESHOLD was lowered from 20 to 15 at commit 565cab7 for validation, then restored to 20 at commit cf3cd89. Production campaign likely ran at SNR_THRESHOLD=15 (4497 detections), evaluation now filters at 20 (keeping ~417-534 well-localized detections).
  implication: SNR threshold mixing is handled correctly at evaluation time -- this is NOT the root cause.

- timestamp: 2026-04-08
  checked: Combined posterior results in cluster_results/eval_corrected_full/
  found: MAP h=0.66 (without BH mass), MAP h=0.68 (with BH mass), true h=0.73. 534 total events, 531 used (physics-floor strategy). The posterior is extremely peaked -- h=0.66 has 0.948 probability mass, with values at h=0.73 at 4.8e-65.
  implication: This is a systematic bias, not a statistical fluctuation. The posterior concentrates on the WRONG value.

- timestamp: 2026-04-08
  checked: Recent P_det extrapolation fix (commit 44d5358)
  found: fill_value changed from 0.0 to None (nearest-neighbor extrapolation). Previously, 44% of events lost completeness correction because P_det=0 outside grid. Fix reduced bias from -9.2% to -6.9% (h=0.663 to h=0.680).
  implication: P_det grid coverage was a major contributor but not the only one. Bias remains at ~6.8%.

- timestamp: 2026-04-08
  checked: Completeness correction implementation (bayesian_statistics.py:668-778)
  found: Gray et al. (2020) Eq. 9 implemented as p_i = f_i * L_cat + (1 - f_i) * L_comp. When L_comp denominator is zero, falls back to L_cat only (f_i=1). The completion term integrates P_det * p_GW * dVc/dz in numerator and P_det * dVc/dz in denominator.
  implication: If P_det is systematically wrong or the completeness function is biased, this propagates into every event's likelihood.

- timestamp: 2026-04-08
  checked: GLADE completeness digitization (glade_completeness.py)
  found: Completeness is digitized from Dalya et al. (2022) Fig. 2 at fiducial h=0.73. When evaluating at h=0.66, the same redshift maps to a smaller d_L, giving HIGHER completeness (catalog appears more complete). This means f_i is larger at lower h, favoring catalog term over completion term.
  implication: The h-dependence of completeness is physically correct (Gray et al. 2020 Sec. II.3.1), but combined with biased P_det or L_cat, it can amplify the bias.

- timestamp: 2026-04-08
  checked: How "results worsen with more detections" manifests
  found: The posterior combination multiplies per-event likelihoods (log-space sum). If each event has a systematic bias toward lower h (e.g., L(h=0.66) > L(h=0.73) for most events), more events AMPLIFIES the bias. The combined posterior at h=0.73 is 10^(-65) relative to h=0.66 with 534 events.
  implication: "Getting worse with more detections" is the expected behavior when there is a PER-EVENT systematic bias. The bias is not from combining runs -- it is intrinsic to the likelihood computation.

- timestamp: 2026-04-08
  checked: Known bias audit findings (memory: project_bias_audit.md)
  found: Multiple known issues documented: (1) allow_singular=True in MVN [MEDIUM, not fixed], (2) fixed galaxy search window across h values [LOW, not fixed], (3) P_det was recently fixed but residual bias remains at ~6.8%.
  implication: The residual bias is from known unfixed issues in the likelihood computation, not from data mixing.

- timestamp: 2026-04-08
  checked: Fiducial cosmology mismatch
  found: constants.py uses H=0.73, TRUE_HUBBLE_CONSTANT=0.7, OMEGA_M=0.25 (WMAP-era). Simulations inject at h=0.73 but Bayesian inference uses TRUE_HUBBLE_CONSTANT=0.7 as reference. LamCDMScenario fiducial may not match injection.
  implication: Potential source of systematic bias if injection and evaluation cosmologies are inconsistent. Need to verify LamCDMScenario fiducial values.

## Resolution

root_cause: The "results worsening with more detections" is NOT caused by data mixing or SNR threshold inconsistency. It is caused by a per-event systematic bias in the H0 likelihood that gets amplified when combining more events. The bias has multiple contributing factors: (1) P_det grid extrapolation was returning 0 for 44% of events (partially fixed in commit 44d5358, reduced bias from -9.2% to -6.9%), (2) remaining unfixed issues in the likelihood computation (allow_singular=True in MVN, fixed galaxy search window across h values), (3) possible fiducial cosmology mismatch between injection (h=0.73) and evaluation (TRUE_HUBBLE_CONSTANT=0.7). Each event's likelihood systematically favors h~0.66 over h=0.73, so multiplying N event likelihoods makes the posterior (0.66/0.73)^N more peaked at the wrong value.
fix:
verification:
files_changed: []
