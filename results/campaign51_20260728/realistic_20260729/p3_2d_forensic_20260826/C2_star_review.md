# Adversarial review — C₂* blind re-derivation + venue-drift adjudication (PA-2D-9 forensic, row #207)

**Un-blind reviewer, 2026-08-26.** Inputs: `C2_star_rederivation.md` (the blind derivation),
the PA-2D-9 frozen numbers, the 24-seed bt/bc fleet frames
(`p3_2d_fleet_20260825/`), the committed code at the fleet's stamped commit lineage
(`bayesian_statistics.py`, `correspondence_1d.py`, `p3_2d_fleet.py`, `ca_rhs_scorer.py`,
`p3_2d_companion*` artifacts), `PREREGISTRATION_P3_2D_20260825.md` +
`A20_REVIEW_P3_2D_DESIGN_20260825.md`. Companion compute artifacts:
`venue_drift_adjudication.py`, `venue_drift_per_seed.csv`, `venue_drift_adjudication.json`,
`phi_table.npz` (this scratchpad). No production edits; no commits.

---

## Task 1 — Verdict on the derivation: **AGREE, with one completeness criticism**

Step-by-step against the code:

1. **α_G_φ = β_G_φ·Σ^4D/Σ^φ** — verified at `path_a_mixture_objects`
   (`bayesian_statistics.py` :2455–2459: `n_hat_w_phi = sigma_phi/beta_G_phi`,
   `alpha_G_phi = sigma_4d/n_hat_w_phi`). Numerically confirmed against a fleet frame:
   CSV `alpha_G_phi = 5.868831e7` = 153322758.6 × 0.382776. ✓
2. **With-BH global divisor = Σ^4D** — verified at :5013–5022/:5098–5106: the `[P3-RPHI]`
   "phi" slot swaps ONLY `global_denom_no_bh` to Σ^φ; `global_denom_with_bh` stays
   `_global_cat_denom_with_bh` (Σ^4D). The **Σ^4D cancellation** in
   A₂ = α_G_φ·L_cat_with_bh = (β_G_φ/Σ^φ)·Σ_g w_g N_g^{2D} is therefore mechanical; the
   prereg additionally banks it verified ≤6.9e-8 on all 24 artifacts. **Σ^φ is the surviving
   divisor — it belongs in C₂*'s denominator for the estimator as implemented.** ✓
3. **No S̄_φ in the with-BH numerator** — verified: the twin S̄_φ factor (`_cat_surv_on`,
   :6362–6368) multiplies the WITHOUT-BH integrand only; the 2D twin (`_cat_surv_2d_on`,
   :6620–6646) inserts `_mz_sel_2d_expectation` (S_4D inside the product-Gaussian mass
   quadrature), no S̄_φ(z). ✓
4. **b₂ = β̄_Ḡ_φ (tower identity)** — the B_num_wbh integrand (:5282–5340) contracts
   `completion_mass_factor_g_sel` built on the SAME `precompute_phi_marginal_survival`
   S_4D machinery; ∫∫ φ_pop·p(M̂|M(1+z))·S_4D dM dM̂ = S̄_φ(z) gives
   ∫B₂ = ∫(1−f̄)w_pop S̄_φ = β̄_Ḡ_φ (:2087–2088 table). Accepted (gate (ii-e) bounds carried). ✓
5. **∫U₂ = Σ̃^4D** — the companion's object (`p3_2d_companion_v2.json`, arbiter-adjudicated to
   ≤6.5e-10) is the same kernel × Eddington-shifted-Gaussian × S_4D contraction. ✓
6. **Arithmetic** — C₂* = 153322777.12146157×348078892.5018141/(980867125.6740596×888403790.0)
   = 0.06124403326364123, reproduced exactly. ✓
7. **The flagged class-G venue drift** — verified in code: `_draw_2d_accepted_latents`
   (`correspondence_1d.py` :1605–1760) draws z_true ∝ k̄_g·S̄_φ
   (`_draw_kernel_survival_redshifts` :1440–1502, 1D law unchanged) AND layers
   Bernoulli(S_4D) on top, while the mz_sel numerator carries no per-candidate S̄_φ. The
   S̃_φ,g host weight cancels against the z-conditional's normalizer, so the accepted-event
   law is exactly (model class-G law) × S̄_φ(z_ev), renormalized. The flag is REAL. ✓

**Verdict on the AGREE:** RATIFY. C₂* = β_G_φ·Σ̃^4D/(Σ^φ·β̄_Ḡ_φ) is the exact constant for
the model-side pairing; Σ^φ (not Σ^4D) belongs in the denominator. The empirical ×r₂
"reconciliation" is NOT evidence for a Σ^4D pairing — see Task 2: it is numerology
(§ below).

**Completeness criticism (the one gap):** the derivation's §2 asserts the completion-class
venue generates "M̂|z from g_sel(·)/S̄_φ(z) — the S̄_φ cancels … ḡ₂(d) = B_num_wbh(d)/β̄_Ḡ_φ
pointwise." That is true of the *mixture's own predictive*, but the derivation presents it
as if the implemented `population_selected` venue realizes it. It does not: the venue draws
z ∝ w_pop(1−f̄)S̄_φ (z-marginal exact ✓) but assigns the event's **M̂_z from the SNR-weighted
donor injection row, unlinked to the drawn z** — the completion side never received the
mass-law extension the class-G side got (the prereg's M2-LINK gate is class-G-only). The
derivation applied venue-vs-model scrutiny to G but not to Ḡ. By the quantitative
elimination below, the Ḡ-side mismatch is the prime suspect for the dominant, still
unattributed factor.

---

## Task 2 — Quantitative adjudication of the flagged mechanism: **REFUTED as the CONTROL-FAIL's cause** (real, but ~7× too small)

### The mechanism's own algebra (derived, not assumed)

Venue accepted-event law = model law × S̄_φ(z_ev), renormalized by
Σ̃^{φ4D}/Σ̃^4D = ⟨S̄_φ⟩_{model,1} (the both-survivals contraction over the model's
S_4D-accepted measure). Hence for any per-event weight ω(w₂)·1_F0 (a function of the data):

    LHS_obs/LHS_model = ⟨S̄_φ⟩_{model,ω} / ⟨S̄_φ⟩_{model,1}
                      = E_b[ω]·E_b[1/S̄_φ] / E_b[ω/S̄_φ]          ≡ R_pred(ω)

with E_b over the banked venue-accepted events (all 200/seed; ω = 0 off F-0) — the model
measure recovered by 1/S̄_φ harmonic reweighting of the banked draws. **The relevant
weighting is NOT a bare arithmetic (1−w₂)-weighted mean of S̄_φ**: the S̄_φ tilt appears in
both the event weight and the normalization, so the deficit is a *ratio of two S̄_φ means*
(weighted / unweighted), each a harmonic-type mean under the banked measure.
ω_identity = (1−w₂); ω_BR = (1−w₂)/(1+(r₂−1)w₂) (F9 form). The F-0 filter, w₂
conventions (LIVE = L_cat_with_bh>0), event_idx→prepared-row mapping (0-based), and the
frozen LHS were all replicated exactly before prediction (my per-seed LHS mean/SEM matches
PA-2D-9 to <1e-8: 0.0050077/0.00011615 and 0.00332207/9.164e-5).

Machinery: S̄_φ from the SAME committed builder the venue uses
(`c1d.build_b0i_2d_selection_objects(h_true=0.73)` → `phi_table.npz`, 1500 nodes),
endpoint-clamped `np.interp` (the :6368 convention); z_ev = the banked `z_true` column of
each seed's `prepared_cramer_rao_bounds.csv` (200 venue-accepted events/seed, 24 bt seeds).

### Predicted vs observed

| quantity | mechanism prediction (per-seed mean ± SEM; pooled) | observed (frozen PA-2D-9) | verdict |
|---|---|---|---|
| LHS₂/RHS₂ (identity, B-T/twin) | **0.86473 ± 0.00511** (pooled 0.86288) | **0.34505 ± 0.01342** | FAILS by ~36σ |
| LHS₂,BR/RHS₂,BR | **0.84024 ± 0.00626** (pooled 0.83769) | **0.36575 ± 0.01390** | FAILS by ~33σ |
| crude ×r₂ reconciliation (ratio 1/r₂ = 0.38278) | — | id: −2.81σ; BR: −1.22σ | closer, but see below |
| drift × 1/r₂ combined | id 0.33100; BR 0.32162 | id +1.05σ ✓; BR +3.17σ ✗ | no constant+drift combo fits both arms |

Diagnostics: ⟨S̄_φ⟩ arithmetic (1−w₂)1_F0-weighted = 0.7812±0.0044 vs all-accepted
0.8896±0.0012 (weighted-mean z 0.183 vs 0.119 — the completion weight does sit at the
high-z, low-S̄_φ end, exactly the mechanism's sign); min S̄_φ over all 4800 accepted events
= 0.481, so the harmonic estimator has no heavy tail. Venue-corrected LHS
(÷R_pred): identity 0.005791±0.000139 (residual vs RHS −0.008722±0.000474, ~4.6× the ε₂
band); BR 0.003954±0.000113 (residual −0.005129±0.000263, ~2.7× band).

**Conclusion:** the flagged double-S̄_φ class-G venue drift is REAL and now MEASURED — a
13.5% (identity) / 16.0% (BR) deficit, incidentally matching the A20 F9 line's
"review-computed venue-drift reference" −13.6% — but it explains only ~0.865 of the
observed ~0.345. It does **not** reproduce the observed ratios and is far worse than the
crude ×r₂ reconciliation. **The prediction test fails.**

### Why the ×r₂ reconciliation is numerology, and where the residual factor lives

- ×r₂ predicts 0.38278 for BOTH arms; the arms sit at 0.34505 (−2.81σ) and 0.36575
  (−1.22σ). Its "in band" status (residuals 1.43e-3, 4.0e-4 vs ε₂ = 1.914e-3) rides on the
  ε₂ anchor, not the measurement errors (the identity residual is 2.6σ_comb). No single
  constant fits both arms, and combining it with the measured drift makes BR worse (+3.17σ).
- C₂* is a **self-consistency** constant: since Σ^4D cancels identically in the implemented
  A₂ (item 2 above), a Σ^φ↔Σ^4D "wrong pairing" story has no mechanism — the identity is
  derived FROM the implemented estimator, so a pure constant cannot be off unless one of the
  contractions (Σ̃^4D, β̄_Ḡ_φ — both independently arbitered) or a venue-model pairing is
  wrong. Landing near 1/r₂ is a coincidence of magnitudes, and banking it as an attribution
  would paint over an unexplained ×~0.40 with a meaningful-looking constant (exactly the F9
  Jensen-gap warning's territory: scalar transforms of RHS are not the control).
- Eliminating the class-G side (fully measured above) localizes the residual to the
  **RHS/completion side**: the required inflation is X_id = R_pred/obs = **2.506** and
  X_BR = **2.297** — same sign, same scale, ~9% weight-dependence, which a *law* mismatch
  produces and a constant cannot. Prime suspect (Task 1 criticism): `population_selected`
  draws the completion events' M̂_z from donor injection rows unlinked to the drawn z, while
  ḡ₂ = B₂/β̄_Ḡ_φ requires M̂|z ~ g_sel(z,·)/S̄_φ(z). The RHS₂ instrument scores these draws
  through the production w₂ pipeline, so E_q[w₂·1_F0] ≠ E_ḡ₂[w₂·1_F0] with an
  arm-weight-dependent factor. UNVERIFIED quantitatively — the RHS per-draw frames were not
  retrieved from the cluster (task JSONs only), so this is a hypothesis with the right
  structure, not a measurement. It is cheap to test (below).

Also computed: the F8 coherence clause **holds** on the frozen numbers —
|(T_w₂(B-C) − D_C₂) − κ̂₂·T_w₂(B-T)| = 6.0e-4 ≤ band 1.914e-3 — i.e. the bt/bc arms are
displaced coherently (common-mode), consistent with a venue-side mechanism and inconsistent
with an arrangement-side (twin-vs-coded) defect.

---

## Task 3 — Verdict map and honest re-scoring

**Map consequence.** The registered C-A map banked CONTROL-FAIL on the B2-R gate. With the
mechanism evidence — (i) B-C/B-T coherence holds within band (common-mode), (ii) a measured,
real class-G venue drift (0.865/0.840), (iii) the residual ×~0.40 localized by elimination
to the completion-side venue law, (iv) no scorer-side defect found (w₂ assembly is the same
CSV pipeline on both sides; α_G_φ verified) — the failure is **venue-side, common-mode
across arms**: the map's honest reading is **VENUE-MISSPEC**, not CONTROL-FAIL-unattributed.
However, PA-2D-9's CONTROL-FAIL was banked under the registered gate, and re-mapping a
banked verdict is a **[RULE]** — it returns to the author with this evidence; I do not
re-score it unilaterally. Strictly: VENUE-MISSPEC with the class-G component ATTRIBUTED
(measured) and the completion-side component LOCALIZED-BUT-UNCONFIRMED.

**Re-scoring options, strictly assessed:**

- **(a) Venue-corrected identity (divide banked side by the measured S̄_φ factor): REJECT.**
  Two independent grounds. Strictness: it is a post-hoc correction — the correction term was
  chosen after seeing the fail, it changes the registered banked statistic, and it is not
  registered-form-preserving (the registered form fixes LHS₂ = (C₂*/200)·Σ_acc(1−w₂) with a
  frozen constant; an event-weighted empirical divisor is a new estimator). Sufficiency: it
  does not work anyway — the corrected LHS (0.005791/0.003954) still misses RHS by 4.6×/2.7×
  the band. The same strictness verdict applies a fortiori to the ×r₂ reconciliation: also
  post-hoc, also not form-preserving, and additionally attribution-free.
- **(b) Fix the venue and re-run: the RIGHT fix, but SEQUENCED — costed.**
  (b1) Class-G: remove the double survival weight in the 2D branch (z-draw from k̄_g·w_pop
  without the S̄_φ factor when the Bernoulli(S_4D) layer is active, or keep the z-draw and
  drop the Bernoulli in favor of an S̃-reweighting — one of the two, not both); harness-only
  (`correspondence_1d.py`), fleet re-run ≈ 24 seeds × 2 arms × ~2 min evaluate ≈ **2–4 CPU-h
  local/cluster**. (b2) Completion side: mass-law extension for `population_selected`
  (M̂|z from the g_sel conditional — the exact Ḡ analog of what the class-G side already
  got) + RHS₂ re-run ≈ **~40 CPU-h cluster** (the PA-2D-8 realized costing). **Before (b2)
  spends 40 CPU-h: a confirmation instrument** — re-run 2–3 RHS chunks locally with
  per-draw (z, M̂_z, w₂) banked (~5 min/chunk/arm) and compute the X factor directly against
  a z-linked-M̂ counterfactual draw; if X ≈ 2.3–2.5 lands, the attribution is confirmed and
  (b) is justified end-to-end. All of (b) requires fresh [DO]s: new instruments, new runs.
- **(c) Report the 2D identity as VENUE-MISSPEC-bounded: ADOPT NOW.** Zero-compute, honest,
  and preserves everything banked: the frozen numbers are pairing-independent measurements
  (PA-2D-9's own note); the verdict line becomes "2D bounded identity: VENUE-MISSPEC
  [pending author RULE on the re-map from CONTROL-FAIL] — class-G drift measured at
  0.86473±0.00511 (identity) / 0.84024±0.00626 (BR); residual common-mode factor
  X_id = 2.506 / X_BR = 2.297 localized to the completion-class venue law, unconfirmed;
  no TWIN2 verdict issued." C₂* itself stands as derived.

**Recommendation: (c) now, (b) next, (a) never.** Report VENUE-MISSPEC-bounded with the
measured drift banked as the attributed component and the completion-side hypothesis
registered as the next falsifiable claim; put the CONTROL-FAIL→VENUE-MISSPEC re-map and the
(b1)+(confirmation)+(b2) sequence to the author as [RULE] + [DO]s. Do not bank the ×r₂
reconciliation as anything but a rejected fit.

---

## Numbers appendix

- r₂ = Σ^φ/Σ^4D = 2.6124925 (verified from the banked selection tables).
- Observed ratios: identity 0.34505±0.01342; BR 0.36575±0.01390 (SEs propagated from the
  frozen SEMs).
- R_pred per-seed vectors + diagnostics: `venue_drift_per_seed.csv`; summary:
  `venue_drift_adjudication.json`.
- Replication: my recomputed LHS₂ and LHS₂,BR match the frozen values to <1e-8 (mapping:
  diagnostics `event_idx` = 0-based prepared-CSV row; F-0 = σ_dL/d̂<0.10 ∧ SNR≥20 verified
  to reproduce the 84/200 acceptance of seed 900101 exactly).
- F8 coherence residual on frozen numbers: −6.01e-4, within band 1.914e-3.
