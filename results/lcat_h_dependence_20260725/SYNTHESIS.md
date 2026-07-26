# L_cat h-dependence investigation — synthesis (2026-07-25)

Two independent agents: D2 structural audit (derivation vs Gray 2020 A9 / Gair 2023 /
gwcosmo v2, equations read from arXiv TeX) and D1 empirical decomposition (12 events
instrumented across the 41-value h-grid; validated against shipped cluster diagnostics
to ≤4.5e-13). D2's ranked predictions were tested by D1.

## Verdict: the rail is a HOST-MISASSOCIATION effect, not a normalization defect

- **Dominant factor (91–100% of d ln L_cat/dh per rail event):** the numerator
  GW-likelihood × host-redshift overlap. Every rail event's rate-weighted candidate
  ball has preferred h*_g = 0.73·d_L(z_g;0.73)/d_L_det with median ≈ 0.42–0.48 —
  i.e. the ball contains only FOREGROUND galaxies relative to the GW distance, so
  ln L_cat rises monotonically toward the low-h grid edge. Non-rail events have
  h*_med ∈ [0.60, 0.89]. A zero-free-parameter overlap model predicts the measured
  slopes' sign 11/12 and magnitude to ~20–45%.
- **D2's top candidate (ball-local selection denominator Σ_ball w_g D_g instead of
  the references' full-catalog sum) is REAL but SECONDARY:** measured share 1–14%,
  depth-graded, opposite sign; surgical local→global swap softens the deepest event
  only −54.6 → −50.6 ln/h (≤9%). P1 refuted (r = −0.31/−0.40 vs predicted ≳0.8);
  P2 refuted (measured mean −12.1 vs predicted +3.37, wrong sign); P3 not supported.
- **volume_deconv kernel: exactly h-invariant** (Z_g ∝ h⁻³ to 1e-15; normalized
  host-z prior h-invariant to 1.5e-15 — cancels identically in flat ΛCDM).
  Normalization-mode swaps (volume_deconv / local_ratio / full global over 9.06M
  galaxies) leave the rail slope essentially unchanged.
- **Why even shallow events rail:** photo-z slack (±σ window) + bottom-heavy dN/dz
  admits foreground impostors at all depths, not just beyond the catalogue mass-prune
  horizon — this is why host-found z ≤ 0.3 subsets rail too. The z ≤ 0.2 subset
  "closure" at 0.7292 is not host identification succeeding: at low z the photo-z
  width ≫ GW precision blows up σ_eff and *flattens* the overlap term.
- **Consistency with the full evidence chain:** explains the 82% L_cat tilt share
  (EXP-40), the z-grading, the failure of the #29 fallback (fallback events are
  h-inert; the damage is in host-FOUND events), and the failure of depth truncation
  (z_cut 0.2/0.3/0.5 all rail — truncation does not change which hosts carry
  numerator weight).

## Structural mechanism (why the estimator lets impostors dominate)

L_cat is a SELF-NORMALIZED ratio-of-sums over the candidate ball. When the ball
contains no plausible true host, the ratio still produces an O(1) likelihood whose
h-shape follows the impostors — the event never gets to say "my catalogue candidates
are collectively implausible, defer to the completion term". The relative weight of
catalogue vs dark hypotheses (β_G·L_cat vs B_num inside p_i = (β_G L_cat + B_num)/D)
uses the GLOBAL completeness weight, not per-event candidate plausibility.

**Falsifiable scoping statement (D1 §7):** only interventions that change *which
hosts carry numerator weight* — a Gray-style per-event membership mixture /
catalogue-vs-dark odds, or absolute (non-self-normalized) catalogue mass — can
de-rail; interventions on normalizations or selection integrals cannot. The z_cut
re-eval scan (all rail) already confirms the negative half of this statement.

## Follow-ups (secondary, tracked)

1. Σ_ball vs Σ_full selection denominator (D2's finding): real discrepancy vs all
   three references, 1–14% of the tilt; fix opportunistically with the main redesign.
2. The 2026-07-02 rejection of the global normalization mode predates the #29 fix —
   historical result is confounded; no action needed if the membership-mixture
   redesign supersedes the mode choice.
3. D2 bonus: σ_z = 0.013(1+z)³ (Known Bug 9, "no reference") is actually the
   Gair/TH21 toy model — the reference exists.

## Artifacts

- D1: `D1_EMPIRICAL_DECOMPOSITION.md` + scripts + JSONs (this dir)
- D2: `D2_STRUCTURAL_AUDIT.md` (this dir)
- Evidence chain: issue #30 (2026-07-25 comments), 
  `results/campaign_phase2_runs/run_20260719_seed1000_exp40/FINDINGS_EXP40_20260725.md`
