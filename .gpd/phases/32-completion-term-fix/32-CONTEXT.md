# Phase 32: Completion Term Fix - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the systematic low-h bias in the H0 posterior caused by the completion term L_comp in the Gray et al. (2020) completeness-corrected likelihood. The current implementation shows MAP h=0.66-0.68 (true h=0.73), with bias worsening as more events are combined. The completion term dominates 79% of per-event likelihoods due to low GLADE completeness (~21% at EMRI distances).

This phase fixes the L_comp implementation and validates the correction. It does NOT change the redshift prior model (dVc/dz vs rate-weighted) — that is a separate modelling decision requiring consistent treatment in both catalog and completion terms.

</domain>

<contract_coverage>
## Contract Coverage

- **Claim / deliverable:** L_comp denominator integrates over the full detectable volume per Gray et al. (2020) Eq. 32, with precomputed D(h) table
- **Acceptance signal:** MAP H0 shifts closer to h=0.73 after the fix; bias-vs-N convergence plot shows bias shrinking as ~1/sqrt(N)
- **False progress to reject:** MAP shift that comes from a normalization bug rather than correct physics; bias that improves at one N but worsens at larger N

</contract_coverage>

<user_guidance>
## User Guidance To Preserve

- **User-stated observables:** (1) MAP shift toward h=0.73 for both "with BH mass" and "without BH mass" channels, (2) bias-vs-N convergence curve, (3) per-event L_comp decomposition across h values
- **User-stated deliverables:** All three validation outputs: MAP comparison, convergence plot, per-event decomposition
- **Must-have references / prior outputs:** Gray et al. (2020), arXiv:1908.06050, Eqs. 9, 31-32. Catalog-only diagnostic (--catalog_only flag) as baseline comparison. Existing debug investigation in `.gpd/debug/h0-posterior-bias-worsening.md`
- **Stop / rethink conditions:** (1) Fix makes bias worse (>-10%), (2) >10% of events produce zero or NaN L_comp, (3) catalog-only baseline disagrees qualitatively with expectations

</user_guidance>

<decisions>
## Methodological Decisions

### L_comp redshift prior

- **Decision:** Keep dVc/dz (uniform in comoving volume) as the completion term prior for now
- **Rationale:** Changing to a rate-weighted prior R(z)·dVc/dz would require consistent treatment in both catalog and completion terms — a larger modelling change. The bias may not originate from the prior choice.
- **Action:** Run diagnostics first (catalog-only comparison + per-event L_comp/L_cat decomposition) to quantify how much bias comes from L_comp's h-dependence vs other sources
- **Disconfirming signal:** If diagnostics show L_comp is h-flat, the bias source is elsewhere

### Integration limits and normalization

- **Decision:** Extend the L_comp denominator to the full detectable volume, per Gray et al. (2020) Eq. 32. The numerator stays local (4-sigma d_L window where p_GW has support).
- **Rationale:** The current implementation restricts both numerator and denominator to the same 4-sigma window. Gray et al. (2020) specifies the denominator as an integral over the full detectable volume. Local normalization may not properly cancel h-dependent volume effects.
- **Implementation:** Precompute denominator D(h) once per h-value (does not depend on the event, only on P_det and dVc/dz). Reuse for all events. Major speedup.
- **Verification:** Add a check that the numerator integrand is negligible at the 4-sigma boundary, confirming the local approximation is valid for the numerator.
- **Research needed:** Cross-check exact integration limits against Gray et al. (2020) before implementing.

### P_det boundary behavior

- **Decision:** Truncate P_det to 0 beyond the injection grid edge (revert fill_value to 0.0 for the full-volume denominator integral). In the numerator (local 4-sigma window), flag events where the injection grid doesn't cover the integration range.
- **Rationale:** Beyond the injection grid, EMRIs are truly not detectable — P_det=0 is physically correct for the full-volume integral. Nearest-neighbor extrapolation overestimates detectability. For the numerator, the grid likely covers the 4-sigma window, but events where it doesn't should be logged rather than silently zeroed.
- **Approximation note:** Truncation underestimates detectability near the grid edge. Future improvement: extend the injection grid to cover the full d_L range. Document as a known systematic in the paper.

### Agent's Discretion

- Quadrature method and number of points for the full-volume denominator integral
- Implementation of the precomputed D(h) table (interpolation, caching strategy)
- Diagnostic output format for per-event L_comp decomposition
- Whether to use the existing `--catalog_only` flag for the clean baseline or create a separate diagnostic mode

</decisions>

<assumptions>
## Physical Assumptions

- GLADE completeness f(z, h) is correct as implemented: ~21% at typical EMRI distances (>800 Mpc) | If f(z) is wrong, both L_cat weight and L_comp weight are wrong — the fix won't help
- P_det from injection campaign is accurate within the grid: KDE-based, SNR-rescaled, validated in Phase 20 | If P_det is systematically biased, L_comp inherits the bias regardless of integration limits
- The 4-sigma d_L window captures effectively all of p_GW's support: Gaussian measurement model with Fisher-matrix-derived uncertainties | If tails are heavier than Gaussian (e.g., multimodal posteriors), the local numerator approximation breaks
- dVc/dz is the appropriate uninformative redshift prior for uncataloged hosts | If EMRIs have a strongly z-dependent rate that matters, this prior is wrong — but changing it is out of scope (requires consistent treatment in both terms)

</assumptions>

<limiting_cases>
## Expected Limiting Behaviors

- When f_i -> 1 (complete catalog): L_comp drops out, result must match catalog-only mode exactly
- When f_i -> 0 (empty catalog): Result is entirely from L_comp; should still be unbiased if L_comp is correct
- When N_events -> infinity: Bias should shrink as ~1/sqrt(N), not grow. Growing bias signals a per-event systematic.
- When denominator window matches numerator window: Must recover current implementation's results (regression check)
- When P_det = constant: L_comp reduces to integral of p_GW * dVc/dz, normalized by integral of dVc/dz. The h-dependence should come only from the d_L(z, h) mapping.

</limiting_cases>

<anchor_registry>
## Active Anchor Registry

- Gray et al. (2020), arXiv:1908.06050, Eqs. 9, 31-32
  - Why it matters: Defines the completeness-corrected likelihood formula. Integration limits and normalization must match.
  - Carry forward: planning, execution, verification
  - Required action: read, compare, cite

- Catalog-only baseline (--catalog_only flag, full 531-event production data)
  - Why it matters: Isolates L_cat contribution. Catalog-only showed -17.8% bias (60 events) vs -6.8% with completeness (531 events), confirming completeness helps. Need clean re-run on full data.
  - Carry forward: execution, verification
  - Required action: run, compare

- Debug investigation: `.gpd/debug/h0-posterior-bias-worsening.md`
  - Why it matters: Documents the root cause analysis: L_comp dominance, low completeness amplification, exponential bias accumulation
  - Carry forward: planning, execution
  - Required action: read

- Production results: cluster_results/eval_corrected_full
  - Why it matters: Current baseline — MAP h=0.66/0.68, 531/527 events, bias -9.6%/-6.8%
  - Carry forward: execution, verification
  - Required action: compare (before vs after fix)

- P_det fix commit 44d5358
  - Why it matters: Changed fill_value from 0.0 to None. Reduced bias from -9.2% to -6.9%. This phase partially reverts that decision for the denominator.
  - Carry forward: execution
  - Required action: read, understand interaction

</anchor_registry>

<skeptical_review>
## Skeptical Review

- **Weakest anchor:** The assumption that extending the denominator to full volume will reduce the bias. The bias could be in L_cat itself (MVN allow_singular, fixed galaxy search window) rather than L_comp normalization.
- **Unvalidated assumptions:** That the injection grid covers the 4-sigma window for most events (needs checking). That precomputed D(h) is accurate enough (quadrature convergence).
- **Competing explanation:** The bias could come from the catalog term's selection effect denominator, not the completion term at all. The catalog-only diagnostic showing -17.8% bias suggests L_cat itself is biased.
- **Disconfirming check:** If the fix doesn't change the MAP by more than 0.01, the completion term normalization was not the dominant bias source.
- **False progress to reject:** A MAP shift that comes from breaking the f_i -> 1 limiting case (catalog-only regression). Any fix that improves one channel but worsens the other.

</skeptical_review>

<deferred>
## Deferred Ideas

- Rate-weighted redshift prior R(z) · dVc/dz for L_comp — requires consistent treatment in both catalog and completion terms, own phase
- Extended injection grid to cover full d_L range — eliminates P_det truncation approximation, requires new cluster run
- MVN allow_singular=True investigation — may contribute to L_cat bias, separate diagnostic
- Fixed galaxy search window across h values — known issue from bias audit, separate fix

</deferred>

---

_Phase: 32-completion-term-fix_
_Context gathered: 2026-04-08_
