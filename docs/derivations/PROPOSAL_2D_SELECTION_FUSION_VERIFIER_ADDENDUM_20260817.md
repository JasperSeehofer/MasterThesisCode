# VERIFIER ADDENDUM — PROPOSAL_2D_SELECTION_FUSION_20260817 (row #117 item 5)

**Date:** 2026-08-17 · **Verifier:** independent xhigh adversarial agent (read-only), per the
ratified item 5 (row #117) · **Object:** `PROPOSAL_2D_SELECTION_FUSION_20260817.md` (`298c4963`)
· **Read of record:** the proposal, L6-DER3, L6-DER2 + its verifier addendum, the A-FULL-2D arm
readout, the N-2/T3′ draft, and the live code (`bayesian_statistics.py` :1859, :2012, :4230–4500,
~:5200, :3172–3199; `simulation_detection_probability.py` :2018–2140).

## VERDICT: GO-WITH-AMENDMENTS

Item-1 implementation may begin; the presentation gate must incorporate MAJOR-1..4. No finding
overturns the derivation transfer, the pairing constraint, or the decision-table structure. The
directed hunt for an h-frozen survival table came back clean: `_phi_survival_table` is keyed per
h (:3180), `S_4D` grids rebuild per h with pool SNR rescaling
(`simulation_detection_probability.py:656-659`), and the fused query passes node `d_L(z;h_eval)`
— h-dependence enters correctly.

## MAJOR findings

- **MAJOR-1 — regime statement inverted.** [P1]'s "production exercises the broad-σ_cond regime"
  (inherited from L6-DER3 §5 V4) is factually wrong: measured on the production CRB reference
  (`results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv`, 1590 rows),
  fractional σ_M/M p5/p50/p95 = 2.5e-8 / 8.8e-8 / 3.0e-7. The "broad σ_M ~ 60–200%" figure is
  the CATALOGUE leg's host stellar-mass error, not the GW conditional `_sigma_cond_M` that sets
  g_i's Hermite window. Production's completion leg is in the venue's sharp-likelihood limit:
  `g_sel,prod ≈ g_i · S_4D(d_L(z;h), μ_cond·M_z,det)`. [P1] stays the correct form (A3 still
  forbids assuming magnitude), but the expected production action is 1D-dominated ([P2]) with
  [P1] formally-correct-but-possibly-near-inert. **Amendment:** correct the regime statement;
  require the [P5-3] counterfactual to decompose 1D and 2D contributions separately.
- **MAJOR-2 — S≡1 bit-exactness unachievable vs the production default; hidden quadrature flip.**
  The production call site (:4344) uses `completion_mass_factor_g` with no kwargs ⇒ Route-1
  `adaptive=True` (default since 2026-08-12), and at measured σ_cond the fast n=8 path is the
  common path. A pinned non-adaptive n=64 `g_sel` at S≡1 differs at the ~1e-15 level from the
  production default on nearly every row (the L6-DER2 addendum measured 1.1e-15 for exactly this
  comparison). As written, [P5]-1 bundles TWO changes (S-fusion + adaptive→pinned flip) and a
  perf regression on the hot path. **Amendment:** (i) restate the unit test as S≡1 recovery of
  `g_i(adaptive=False, n=64)` bit-exactly + a recorded pinned-vs-adaptive bound; (ii) present
  the quadrature choice as its own author-ruled line (pinned n=64 vs keep Route-1 with S per
  node — both defensible); (iii) state the measured runtime delta in the presentation gate.
- **MAJOR-3 — [P3] mixture-skew direction inverted.** With [P2] on, `B_num` acquires S̄_φ<1
  while the catalogue numerator is unweighted ⇒ the S̄-free catalogue leg is OVER-weighted
  (up-weighted) relative to completion, not "DOWN-weighted" as the proposal states. L6-DER3 §4
  states no direction; the error is proposal-only. Deferral structure survives (the
  counterfactual measures the skew regardless). **Amendment:** correct in the reviewable
  artifact; re-present decision-table row 2's basis to the author.
- **MAJOR-4 — [P4] scope contradiction.** V2's materiality lives in the catalogue leg's broad
  host-mass Gaussian (`sigma_gal_frac`, :5238) — the leg [P3] defers. In item 1's completion
  scope the relevant width is σ_cond ~ 1e-7, where the L6-DER2 addendum's bound applies
  (ratio-vs-density deviation ≲ 1e-6 from a tilt-neutral per-event constant). Folding the V2
  ruling into item 1 either settles it where it is provably immaterial or forces a cross-leg
  change item 1 does not authorize (and gate (i) addability forbids a completion-only measure
  change). **Amendment:** the presentation gate must rule explicitly: retain the ratio
  convention in both legs + record V2 as a tracked systematic + re-open with [P3]/row #110,
  OR widen the ruling to both legs. Silence is not available.

## MINOR findings

- **MINOR-1** — [P2] default-on breaks `--freeze_g_frac_ref_h` for off-grid reference h:
  `completion_numerator_integrand_sel_1d` raises ValueError when `h_ref ∉ _phi_survival_table`
  (:4300–4306; table built only on `_h_list`, :3180–3184). Tabulate S̄_φ at the freeze h (or
  document the constraint) + test.
- **MINOR-2** — the :4352 φ-support warning will fire mislabeled on S-induced zeros
  (beyond-horizon nodes). Distinguish φ-support-exit zeros from S=0 zeros; add the regression
  test L6-DER3 §5 requires (missing from [P5]).
- **MINOR-3** — `completion_mass_factor_g` is imported by `validation/calibration_gate.py:767`,
  `validation/closed_loop_gfrac.py:524`, `validation/venue_transfer.py:1346`, and tests. The
  fused object must be a new function or strictly-additive extension; name these paths in
  [P5]-1's byte-identity list.
- **MINOR-4** — the 1D flip rides on an unadjudicated DRAFT measurement
  (`results/run_20260805_n2sel1d/`, +30.9 nats/h at h=0.73, status DRAFT — uncited), and the
  N-2 draft's honest gap 5 (#66/#67: selection-inside only calibrated when PAIRED with the
  noise-model companion — "the single most likely way this correction disappoints") is
  uncarried. Cite both, with status, in the presentation gate. DS-G3 covers calibration
  in-venue only; [P5-3] measures magnitude, not calibration.
- **MINOR-5** — the fused numerator inherits (and extends into the numerator) the pool-ψ
  prior-weighted survival approximation (exact form wants the extrinsic LIKELIHOOD-weighted
  survival; `S_4D` is the prior-marginal — N-2 §1 honest gap 3; sky part bounded 1.000202 by
  gate (ii-e)). Not introduced by [P1]; record as part of the pool-vs-model residual class
  (consistent with the venue's r=0.85 seed-correlated −11.7-class residual).
- **MINOR-6** — guard `np.log10(M_z)` (:2087) against non-positive Hermite node masses under
  `wbh_z_resolved` (NaN·0 ≠ 0); unreachable at measured σ_cond but one line + test.

## Checks performed (A–H)

| Surface | Verdict |
|---|---|
| A. Derivation transfer (latent-thresholded → selected prior; production p_det construction) | SOUND (MINOR-5 recorded) |
| B. Pairing constraint (P1⊕P2 spurious term; P3 skew existence) | SOUND (direction AMENDED, MAJOR-3) |
| C. Formula faithfulness ("Old" vs code; S-query availability at call site) | SOUND (adaptive-default omission → MAJOR-2) |
| D. Measure/dimensional (gate (i) addability; V2 class claim) | AMENDED (MAJOR-4) |
| E. Limiting cases (S≡1 bit-exactness; σ_cond→0) | AMENDED (MAJOR-2; the "limit" is the operating point, MAJOR-1) |
| F. Numerics (n_hermite=64 in production regime) | SOUND after MAJOR-1 (pin conservative; perf cost → MAJOR-2) |
| G. Verification-plan sufficiency + h-dependence hunt | SOUND with additions (h-freeze ABSENT; add MAJOR-1 decomposition, MINOR-1/2/3 tests) |
| H. Interaction claims (φ-support warning; g_frac/freeze counterfactual) | AMENDED (MINOR-1, MINOR-2) |

**Not verifiable from the repo alone:** MFG (2019) Eqs. (5)–(7) and Gray et al. (2020) Eq.
(A.10) against the published papers; venue premeasure/arm script internals (taken at the
committed readouts' word); the exact production fraction taking the Route-1 n=8 fast path
(inferred from the σ_cond distribution).

## Orchestrator verification note (same date)

The verifier measured the MARGINAL σ_M/M and bounded the conditional by it. The conditional was
then computed directly from the same CSV (Bishop 2.81–2.82 on the fractional (d_L, M_z) block,
matching the :3436–3441 construction): **σ_cond p5/p50/p95 = 2.468e-8 / 8.796e-8 / 2.990e-7,
max 2.73e-6, n=1590; median conditional/marginal ratio 0.99992.** MAJOR-1 holds at the
conditional level exactly, closing the verifier's stated gap.

*Append-only from its commit.*
