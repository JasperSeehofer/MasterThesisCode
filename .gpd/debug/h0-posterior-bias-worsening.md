---
session_id: h0-posterior-bias-worsening
status: investigating
created: 2026-04-08
last_updated: 2026-04-08
symptom: "H0 posterior bias worsens with more detections; MAP at h=0.66 vs true h=0.73"
current_focus: "Completion term L_comp dominance due to low catalog completeness"
eliminated: []
root_cause: null
---

## Current Focus

hypothesis: The completion term L_comp dominates per-event likelihoods due to low GLADE completeness (f_i ~ 0.21 for EMRIs), and L_comp carries a systematic low-h bias from the comoving volume element dVc/dz
test: Analyze the h-dependence of L_comp analytically and compare individual event likelihoods across h values
expecting: L_comp should show slight preference for lower h due to volume effects, which accumulates over N events
next_action: Quantify the L_comp bias mechanism and check whether the selection effect denominator is correctly computed

## Symptoms

expected: MAP posterior at h=0.73 (true injection value), improving precision with more detections
actual: MAP posterior at h=0.66 (without BH mass) and h=0.68 (with BH mass); 531/527 events used; true h=0.73 receives probability ~1e-65
errors: No numerical errors, but posterior converges to WRONG value
reproduction: Run evaluation pipeline with cluster_results/eval_corrected_full data, combine with physics-floor strategy
context: Recent P_det fix (commit 44d5358) reduced bias from -9.2% to -6.9% (mean h 0.663->0.680), suggesting P_det was one factor but not the complete story

## Eliminated

(none yet)

## Evidence

- timestamp: 2026-04-08T10:00
  checked: "Per-event likelihood variation across h values"
  found: "Most events have nearly flat likelihoods: event 2 varies <0.1% across h=0.6-0.86 (4722-4726). Some events show U-shaped anti-correlation: event 78 gives 3886 at h=0.6, 1820 at h=0.73, 3650 at h=0.86 -- DISFAVORING true h."
  implication: "Per-event likelihoods are dominated by an h-insensitive component (completion term) with a slight bias"

- timestamp: 2026-04-08T10:05
  checked: "GLADE completeness at EMRI distances"
  found: "GLADE completeness saturates at ~21.3% beyond 796 Mpc. Typical EMRIs are at several Gpc. So f_i ~ 0.213, giving completion term (1-f_i)=0.787 weight."
  implication: "79% of the per-event likelihood comes from L_comp, which is nearly h-flat. Only 21% comes from L_cat, which carries h information."

- timestamp: 2026-04-08T10:10
  checked: "Combined posterior results"
  found: "534 total events, 531 used. MAP at h=0.66 (no BH mass) and h=0.68 (with BH mass). True h=0.73 gets probability ~1e-65."
  implication: "The bias is massive and grows sharper with more events, consistent with a systematic per-event bias being amplified by the product over events."

- timestamp: 2026-04-08T10:15
  checked: "P_det fix commit 44d5358"
  found: "Changed fill_value from 0.0 to None (nearest-neighbor extrapolation). This reduced L_comp fallbacks from 702 to 0 and improved mean h from 0.663 to 0.680."
  implication: "P_det was causing some events to lose completeness correction entirely. Fix helped but did not resolve the fundamental bias."

## Resolution

root_cause: |
  MULTIPLE CONTRIBUTING FACTORS identified, ordered by severity:

  1. COMPLETION TERM SYSTEMATIC BIAS (primary):
     The completion term L_comp = integral[p_GW * P_det * dVc/dz dz] / integral[P_det * dVc/dz dz]
     has a systematic h-dependence that biases toward lower h values. At lower h, the
     implied redshift z_peak(h) for a given d_L is lower, where dVc/dz has different shape.
     The p_GW modulation in the numerator causes incomplete cancellation of dVc/dz in the
     ratio, creating a small but systematic per-event bias.

  2. LOW COMPLETENESS AMPLIFICATION:
     GLADE completeness is ~21% at typical EMRI distances (>800 Mpc). This gives the
     biased L_comp 79% weight in the formula p = f_i * L_cat + (1-f_i) * L_comp.
     Only 21% of each event's likelihood carries actual h-discriminating information.

  3. EXPONENTIAL BIAS ACCUMULATION:
     With 531 events, a small per-event bias compounds: if each event has a bias factor
     of (1+epsilon) at h=0.66 vs h=0.73, the product gives (1+epsilon)^531 ~ exp(531*epsilon).
     Even epsilon=0.01 gives a factor of ~200.

  4. P_DET NEAREST-NEIGHBOR EXTRAPOLATION (contributing):
     The recent fix (44d5358) changed fill_value from 0.0 to None (nearest-neighbor).
     This prevents L_comp fallback but may introduce systematic P_det overestimation
     at the grid boundary, where P_det is floored at the boundary bin value rather
     than dropping to zero.

  VERIFICATION NEEDED: Run the code with diagnostic output to quantify L_cat vs L_comp
  separately for representative events across h values.

correction: |
  RECOMMENDATIONS (require user decision):
  
  1. INVESTIGATE: Add diagnostic logging to separate L_cat and L_comp for a few events
     across h values. This will confirm which term carries the bias.
  
  2. IF L_COMP IS BIASED: Consider either:
     a) Using a proper source population prior p(z|H0) instead of dVc/dz in the completion
        term. The EMRI rate doesn't simply scale with comoving volume.
     b) Implementing the completion term with a rate-weighted prior that accounts for the
        actual EMRI formation rate as a function of redshift.
     c) At minimum, normalize dVc/dz properly so the prior integrates to 1.
  
  3. IF P_DET EXTRAPOLATION IS BIASED: Extend the injection grid to cover the full 4-sigma
     integration range for all events, or use a physical model for P_det beyond the grid edge.
  
  4. CROSS-CHECK: Compare individual event posteriors (L_cat only, no completion term,
     f_i forced to 1.0) to verify that L_cat itself is unbiased.

verification: Not yet verified -- requires running code with diagnostics
files_changed: []
