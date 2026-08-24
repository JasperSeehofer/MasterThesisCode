# CLAIM INTAKE — a finite-moment identity statistic for catalogue-class calibration, and the M_Ḡ common-mode question

**STATUS: DRAFT — stage-0 claim intake (research-cycle stage 0). Returns to the orchestrator/author. Registers NOTHING; no verdict, band, or arm ruling is banked by this document.**

**Date:** 2026-08-24 · **Grant:** row #178 items 3+4 (A20 b0-verdict review RETURNS (3) and (4))
· **Inputs read:** `A20_REVIEW_B0_VERDICT_20260824.md` (Findings 2, 3, 5, 6, RETURNS),
`PREREGISTRATION_B0_IDENTITY_20260823.md` (§1, PA-1, PA-4, PA-6, PA-11–PA-16, BAND FREEZE,
VERDICT), `docs/derivations/PROPOSAL_P3_MIXTURE_WEIGHT_20260822.md` §1 + Appendices,
`docs/derivations/completion_numerator_data_measure.md`, `bayesian_statistics.py` (B_num
`:5009-5129`, catalogue numerator `:6057-6093`, filter `:386/:3798-3805/:5540-5554`),
`correspondence_1d.py` (b0i draw `:1547-1760`, kernels `:1132-1391`).
**Compute disclosed:** zero `evaluate()` calls. All numbers below are (a) quoted banked values
[DOC], (b) recomputed zero-compute from the 24 banked `p3_b0_work/*/event_likelihoods.csv` +
per-seed `selection_tables_h_0_73.json` + `prepared_cramer_rao_bounds.csv` truth columns
[LOCAL], or (c) computed by local zero-`evaluate()` model-side passes (one
`build_bsel_selection_objects(0.73)` table build — the same construction the banked runs used —
plus trapezoid quadratures, one chunked `kernel_smeared_survival` pass with a doctored value
table, and one 2×10⁵-draw acceptance MC) [LOCAL]. Derivations and judgment calls are [INFER].

---

## 0. Executive summary (numbered claims; each substantiated below)

- **F-0 (NEW venue finding, decision-relevant for everything downstream).** The b0i venue
  draws 200 events/seed, but production's own analysis-intake filter
  `distance_relative_error < 0.10` (`FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD`,
  `bayesian_statistics.py:386`, applied at `:3798-3805` via `use_detection` `:5540-5554`)
  silently removes **41.8% of the drawn events** (fleet: 1397 of 2400 rows survive) before any
  identity statistic is computed. The cut boundary is exact in the banked data (seed 900101:
  kept max σ/d̂ = 0.09934, dropped min = 0.10016) [LOCAL]. This conditioning is
  **class-asymmetric** — it preferentially removes low-z, high-survival catalogue-class events
  (dropped: z_true median 0.138, S̃_host median 0.869; kept: 0.173, 0.799) — and it was **not
  carried into the b0 identity registration's targets** (§7's blindness list names four
  instruments; the survival-time filter is not among them; the in-repo D-1/AMENDMENT-A-6
  machinery knew the mechanism for B-SEL, where it removes only ~10%). The exact conditioned
  identity targets are derived in §2.5: **B-T ≈ 1.59, B-C ≈ 0.52 — not 1 and not ⟨S̄_φ⟩-anything**.
- **Part B, M_Ḡ:** for the registered odds statistic, **M_Ḡ = β̄_Ḡ_φ is exact** (up to
  erf(4/√2)-window ≈ 6×10⁻⁵, O(σ_frac²), and sky–distance Fisher cross-terms): the
  `completion_numerator_data_measure.md` d_L(z)-measure factor is real for ∫B_num dd̂ but
  **cancels event-by-event in the odds ratio** because both legs carry the same
  ratio-measure GW kernel and the identity `N(u;1,σ_dL/d̂) = d̂·N(d̂;d_L,σ_dL)` holds
  pointwise (§2.2). **The ×2.34 B-C miss is NOT an M_Ḡ common-mode.**
- **Part B, the ×2.34:** it decomposes as **(i) a PA-4 prediction error** — the correct
  sharp-GW B-C target is the completion-class-weighted ⟨S̄_φ²⟩/⟨S̄_φ⟩ = **0.3532** [LOCAL],
  not ⟨S̄_φ⟩_w = 0.7851 (factor 2.223 of the 2.343) — **times (ii) the F-0 conditioning**
  (0.3532 → 0.522) **times (iii) a residual deficit factor ≈ 0.64** (B-C) that is two-sidedly
  attributable to heavy-tail truncation and/or genuine displacement, ordered consistently with
  B-T's residual ≈ 0.41 (B-T's tail is fatter). A single common-mode rescale of C* (or M_Ḡ)
  **cannot** reconcile both arms (required γ: 0.64 for B-C vs 0.41 for B-T) — the §7(ii)
  common-mode hypothesis is refuted as a full explanation (§2.6).
- **Part A:** three candidate families derived and ranked. Leading: **C-A, the
  bounded-transform mixture identity** `C*·E_{q_G}[φ(d)·R] = E_{q_Ḡ}[φ(d)]` for bounded φ
  (primary member φ = w), with provably finite variance, an exact conditioned form under F-0,
  and a model-computable right side — distinct from both refuted alternatives (§3.1).
  **C-B, the paired catalogue-leg log-likelihood-ratio Λ**, is fully zero-compute on the
  banked CSVs and was probed here: Λ̄ = **−0.02516 ± 0.00454, negative in 12/12 seeds**,
  bounded per event in [−0.99, +0.2542] — and the probe *itself* surfaced F-0 (the −0.025 is
  quantitatively accounted for by KL(+0.019) + the F-0 acceptance tilt (−0.041) + kernel-smear
  (−0.012) ≈ −0.034 model vs −0.025 measured, §3.2). The 12 banked pairs decide C-B
  **zero-`evaluate()`** once its null is pinned by two more doctored-table passes (~1 CPU-h).
  The 12 banked pairs also already resolve the two arms' bounded-statistic separation at
  **51σ paired** (C-A's left sides: 0.07171 vs 0.06317; paired Δ = +0.008544 ± 0.000166,
  12/12 positive) — the verdict additionally needs the one-time model-side right side
  (~2–4 CPU-h, §3.1).

---

## 1. Objects and notation (venue b0i, fused cell, Σ^φ slot, no-BH channel, h = h_true = 0.73)

Per event with data d = (d̂, Ω̂) (measured distance, measured sky) and donor context ξ (the
SNR-weighted resampled Fisher row: σ_dL, sky covariance) [DOC `correspondence_1d.py:1614-1760`]:

- Catalogue leg (`absolute_marginal` + Path A): `A(d) = β_G_φ·L_cat(d)`, with
  `L_cat = (1/Σ^φ)·Σ_{g∈ball} w_g N_g(d)` (global divisor = Σ^φ under
  `catalogue_global_selection="phi"` [DOC :4896-4905]), and per candidate
  `N_g = ∫ ρ_g(z)·G3(φ_g,θ_g,u(z))·[S̄_φ(z)]_{twin only} dz`, where
  `ρ_g(z) ∝ N(z;z_g,σ_eff)·w_pop(z)·f_k(z)` normalized (Z_g convention), `u(z) = d_L(z;h)/d̂`,
  and G3 is the donor row's 3-D Gaussian density in (φ, θ, u) [DOC :6057-6093]. The coded arm
  (B-C) omits the per-candidate S̄_φ factor; the twin (B-T) includes it (`:6087-6093`,
  un-renormalized — the kernel stays unit-mass without S̄).
- Completion leg: `B_num(d) = ∫ dz (1−f_k(z;pix(Ω̂)))·N(u;1,σ_frac)·(sinθ_det/4π)·w_pop(z)·S̄_φ(z)`
  ('ratio' event measure + fused cell) [DOC :5009-5129]. `w_pop = dV_c/dz/(1+z)`.
- Odds ratio (PA-3): `R(d) = B_num/(β_G_φ·L_cat)`; responsibility `w = A/(A+B_num)`;
  `R = (1−w)/w`.
- Constants at 0.73 [DOC, verified [LOCAL]]: β_G_φ = 1.5332276×10⁸, β̄_Ḡ_φ = 8.8840380×10⁸
  (per-seed `selection_tables_h_0_73.json`; my trapezoid quadrature on the freshly built
  S̄_φ-table grid reproduces both to all printed digits), Σ_w = 1.2493011×10⁹,
  Σ^φ = 9.8086713×10⁸, Σ̃^φ = 9.6887184×10⁸, ρ = Σ̃^φ/Σ^φ = 0.9877707,
  ⟨S̄_φ⟩_w = Σ^φ/Σ_w = 0.7851327, C* = β_G_φ·ρ/β̄_Ḡ_φ = 0.1704718 (recomputed 0.1704717 ✓).
- Generator (b0i, PA-2/PA-11): hosts ∝ w_g·S̃_φ,g; z_true ~ k_g(z)S̄_φ(z)/S̃_φ,g with
  k_g = the estimator's own kernel; **d̂ = d_L(z_true;0.73) + σ_dL·ε, ε~N(0,1)** (additive,
  donor-row σ_dL, clip at 10⁻⁶) and sky = host position + the donor 2×2 Gaussian, drawn
  **independently** of the distance noise [DOC `draw_realization` (c)].
- **F-0:** the evaluated event set is the drawn set conditioned on
  `acc = {σ_dL < 0.10·d̂}` (+ SNR ≥ 20, which no b0i draw fails: dropped rows all have
  SNR ≥ 20.1) [LOCAL, seed-900101 forensics §2.5].

Completion-class ("μ") weighting used throughout: μ(z) ∝ (1−f̄(z))·S̄_φ(z)·w_pop(z),
normalized by β̄_Ḡ_φ. Catalogue-class ("ω") weighting: ω(g,z) ∝ w_g·k_g(z), normalized by Σ_w.

---

## 2. PART B — ∫B_num dd under the 'ratio' measure; is M_Ḡ = β̄_Ḡ_φ exact?

### 2.1 The data-measure factor is real for B_num alone [INFER, confirming DOC]

Re-deriving `completion_numerator_data_measure.md` §2 at the b0i source: with u = d_L/d̂,
∫dd̂ N(u;1,σ) = d_L(z;h)·E_{N(1,σ)}[u⁻²] ≈ d_L(z;h)(1+3σ²). Hence
∫dd B_num = ∫dz (1−f̄)S̄_φ w_pop·d_L(z;h)·(1+O(σ²)) ≈ β̄_Ḡ_φ·⟨d_L⟩_μ ≠ β̄_Ḡ_φ.
Taken alone this looks like exactly the feared common-mode: a ⟨d_L⟩-class factor on M_Ḡ.

### 2.2 …but it cancels exactly in the odds statistic — M_Ḡ = β̄_Ḡ_φ is exact for R [INFER]

The venue's noise convention makes the cancellation *pointwise*, not merely to leading order.
The estimator's per-event distance kernel, at the event's own datum d̂ with
σ_frac = σ_dL/d̂ (the covariance row is divided by d̂², `:5978-5982` "matches covariance
σ²/d_L_measured²"), satisfies the **exact identity**

    N(d_L(z)/d̂ ; 1, σ_dL/d̂)  =  d̂ · N(d̂ ; d_L(z), σ_dL)          (Gaussian symmetry)

for every z and every d̂. The same single factor d̂ (an event constant — NOT d_L(z)) therefore
multiplies the GW kernel of **both** legs, catalogue and completion. In R = B_num/(β_G_φ L_cat)
it cancels identically. Consequently, with p_gen the generator's data density and the twin's
(g,z)-mixture exactly proportional to the generator's (PA-2/PA-11 alignment):

    A_BT(d) = β_G_φ·ρ · d̂ · p_gen(d)   and   B_num(d) = d̂ · B̃(d),
    ∫ B̃(d) dd = ∫dz (1−f̄) S̄_φ w_pop · [∫ N(d̂; d_L(z), σ_dL) dd̂] = β̄_Ḡ_φ · (1 − ε),

so the **unconditioned** twin identity target is E[R]·C* = 1 exactly, with
ε ≤ 1 − erf(4/√2) ≈ 6.3×10⁻⁵ (the numerator's ±4σ z-window), plus O(σ_frac²) from the
u↔1/u reciprocal asymmetry (per-event σ_frac medians ≈ 0.02–0.03 ⇒ ≤ 10⁻³), plus an
unquantified-but-structurally-common term from the donor Fisher **sky–distance cross-terms**
(the generator draws sky and distance noise independently; the estimator's G3 carries the
donor's full 3×3 covariance — identical in both arms, so it cancels in *paired* statistics
like §3.2's Λ but enters class-mass normalizations at second order in the correlations).

**Verdict on the §7(ii) question as posed: M_Ḡ = β̄_Ḡ_φ carries no ⟨d_L⟩-type common-mode
for this statistic.** The `data_measure` memo's factor lives in score/shape statistics (where
the numerator is ratioed against the d-independent D̃_φ), not in the odds ratio.

### 2.3 The corrected B-C point prediction: the PA-4 ⟨S̄_φ⟩_w was the wrong weighting [INFER+LOCAL]

For the coded arm the same computation gives (sharp-GW limit; the GW kernel is 10²–10⁴ times
narrower than the host kernels, so S̄_φ(z) factors out of both candidate sums at the pinned
z* = z*(d̂), *independently of ball membership and kernel shapes*):

    E_{q_G}[R_BC]·C*  =  ∫ B̃(d)·W(d) dd / β̄_Ḡ_φ  →  ⟨S̄_φ²⟩_μ0/⟨S̄_φ⟩_μ0
                       =  ∫(1−f̄)S̄_φ² w_pop dz / ∫(1−f̄)S̄_φ w_pop dz,

where W(d) = the coded catalogue leg's posterior-mean survival ≈ S̄_φ(z*(d̂)) and the
rearrangement moves the measure onto the **completion** intensity — the miss in PA-4 was
predicting with the *catalogue*-class mean ⟨S̄_φ⟩_w = 0.785 where the derivation forces the
*completion*-class-weighted second-moment ratio. Computed on the freshly built (banked-identical)
tables [LOCAL]:

    ∫(1−f̄)S̄_φ²w_pop = 3.1378960×10⁸  ⇒  corrected B-C target = 0.35321   (unconditioned)

vs measured clean-11 B-C I+1 = 0.33482 ± 0.04151 [LOCAL, reproduces the review's numbers].
The ×2.343 miss therefore contains a derivable factor 0.78513/0.35321 = **2.223** that is a
**prediction error, not an arrangement or mass defect**. (The per-event sharp-pin premise is
verified directly in §3.2: mean |lnW̃ − lnS̄(z_true)| residual = +0.0002, sd 0.023, over the
1157 paired-live events of the 10 locally-mirrored seeds.)

### 2.4 Support/heavy-tail structure is the price of this identity [INFER]

The rearrangements in 2.2/2.3 are importance-sampling identities: they hold iff catalogue-class
draws reach everywhere B̃ has mass. They do (the catalogue spans z < 1.5 ⊇ the completion
window z ≤ 1.326), but with vanishing density at depth — which is *exactly* the banked
k̂ ≈ 1–2.7 pathology (review Finding 2/6) and seed-900108's w ≈ 2.3×10⁻⁵ event (Finding 3):
the identity's mean arrives in rare, huge, legitimate lumps. This is a property of the
**estimand**, not of any defect — the motivation for Part A.

### 2.5 F-0: the unregistered conditioning, and the conditioned targets [LOCAL+INFER]

Forensics (seed 900101, [LOCAL]): the drawn realization has 200 rows; 106 reach the
diagnostics. Dropped ⇔ σ_dL/d̂ ≥ 0.10 with a clean boundary (kept max 0.09934, dropped min
0.10016); all dropped rows pass SNR ≥ 20 and the p0 window. Mechanism: the mirror pairs donor
Fisher rows (absolute σ_dL from production events) with re-drawn distances; a low-z catalogue
host paired with a modest-SNR donor fails the ratio cut. In production the donor pool is
post-filter at its own distances, so the filter is nearly inactive there; in the *deep* B-SEL
venue it removes ~10% (the disclosed D-1/A-6 number); in the *shallow* catalogue-class b0i
venue it removes **41.8%** (1397/2400) — and asymmetrically in z.

Conditioning the identity (exact, no independence assumptions; 1_acc = 1{σ_dL < 0.1·d̂}):

    E[R_BT | acc]·C* = E_ξ[P_Ḡ(acc|ξ)] / E_ξ[P_G(acc|ξ)]  ≡  P̄_Ḡ / P̄_G,
    E[R_BC | acc]·C* = E_ξ E_μ[S̄_φ(z)·1_acc] / P̄_G,

with P̄_G directly measured (fleet 1397/2400 = 0.5821) and P̄_Ḡ, E_μ[S̄·1_acc] computed by a
2×10⁵-draw model-side MC (donors SNR-weighted from the pinned production CRB pool
`CRB_CSV_PATH`; completion z ~ μ; d̂ = d_L(z)+σε) [LOCAL]:

    P̄_Ḡ = 0.9259,  E_μ[S̄_φ·1_acc] = 0.3070,  E_μ[S̄_φ | acc] = 0.3316,
    (validation: the same MC on the class-G side predicts P_G = 0.6245 vs realized 0.5821 —
     a ~7% model error from point-evaluated class-G weights and no kernel smear, disclosed)

    ⇒ conditioned targets (P̄_G = 0.5821):  **B-T: 1.591 · B-C: 0.5274**
       (with the MC's own P_G = 0.6245:      B-T: 1.483 · B-C: 0.4915)

**Implications, two-sided:**
1. The registered PA-4 targets (B-T → 1, B-C → ⟨S̄_φ⟩) were doubly mis-specified for the venue
   as executed: wrong weighting (§2.3) *and* un-conditioned (F-0).
2. Measured-to-conditioned-target ratios: B-C 0.3348/0.527 = **0.64 ± 0.08**;
   B-T 0.6504/1.591 = **0.41 ± 0.06**. Both below 1, ordered exactly as the arms' tail
   fatness predicts (B-T needs a mean of 9.33 per unit C*, so more of its identity mass sits
   in the un-realized far tail; the banked truncated-share curves [LOCAL] show B-C's mass
   saturating by R ≤ 500 at 0.337 ± 0.038 while B-T is still climbing: 0.576 ± 0.081 at 500,
   0.641 ± 0.088 at 10³–10⁴, the remainder arriving only via 900108-class events).
3. **The banked "B-T closer to calibrated odds in 11/11 clean seeds" REPORTED-ONLY hint is not
   robust to F-0**: against the *conditioned* targets, B-C sits nearer its target than B-T at
   face value (|0.335−0.527| < |0.650−1.591|), and neither comparison is meaningful until the
   per-arm far-tail deficits are computed model-side. The direction hint should not be quoted
   without this caveat. (The paired secondary 3 Δmean_h and the twin's B-SEL Δ̄ are paired
   same-event statistics; F-0 cancels there to first order.)

### 2.6 What C* would have to be — simultaneous consistency, with Finding 5's deficit bracketed [INFER+LOCAL]

Let γ rescale C* (equivalently a common-mode M_Ḡ error) and τ_a ∈ (0,1] be arm a's
finite-sample tail-retention factor (Finding 5(i)). Simultaneous consistency requires:

    B-C: 0.3348 = γ·τ_C·(target_C)      B-T: 0.6504 = γ·τ_T·(target_T)

- Under the **registered PA-4 targets** (1, 0.785): γτ_C = 0.427, γτ_T = 0.650 ⇒ τ_T/τ_C =
  1.52 > 1 — impossible (B-T cannot retain *more* tail than B-C; its ratios are strictly
  larger event-by-event). **PA-4's constant is refuted by simultaneity alone.**
- Under the **conditioned corrected targets** (1.591, 0.527): γτ_C = 0.64, γτ_T = 0.41,
  τ_C/τ_T = 1.55 — the *right* ordering and a plausible magnitude (Finding 5's unconditioned
  bracket: B-T deficit ≈ 35% of identity mass at the raw target; conditioned it becomes ≈ 59%;
  B-C's smaller and saturating-by-τ=500 share is consistent with τ_C near 0.64·γ⁻¹).
  With τ's free, γ is unidentifiable in [≈0.64, 1] from these two numbers alone; **no value of
  γ (equivalently no C* rescale) fits both arms with τ_C = τ_T** — a common-mode M_Ḡ error is
  refuted as the *sole* explanation (it echoes Finding 5's "multiplicative, cannot create the
  1.94 ratio pattern"), and is unnecessary once (§2.3) + (F-0) + arm-ordered tail deficits are
  in the model. Deciding between γ ≈ 1 ("twin calibrated; tails own the residual") and γ < 1
  ("genuine residual displacement") is exactly what the Part A candidates are for.

### 2.7 What would make Part B wrong

- The conditioned targets carry the acceptance-MC's model error (point-eval class-G law, no
  kernel smear in the acceptance, no donor–host correlation beyond independence): the P_G
  cross-check bounds it at ~7% relative; a kernel-smeared acceptance companion (zero-evaluate,
  §3.2's machinery) would pin it to ≲1%.
- The exact-cancellation claim (2.2) assumes the donor 3×3 covariance's sky–distance
  cross-terms are second-order; a donor pool with O(1) distance–sky correlations would leave a
  residual class-mass factor. Bounded empirically by the §3.2 per-event residual (sd 0.023,
  mean +0.0002) — any such term is ≲10⁻³ on the log scale for the realized pool.
- The sharp-GW factor-out in 2.3 fails for events with σ_frac ~ O(1); F-0 removes exactly
  those (kept max 0.099), so post-conditioning the premise is safe — but this also means 2.3's
  unconditioned form must never be quoted alone (it was, above, only as a decomposition step).
- If `event_idx` did not index the realization rows, the truth-matching in §3.2 would be
  invalid; verified: max event_idx = 199 on 200-row tables, and the per-event residual's
  tightness (sd 0.023) is itself proof of correct alignment.
- The 2.5 acceptance model treats the filter as the only drop mechanism; SNR/p0 were checked
  inactive on seed 900101's 94 drops [LOCAL], not on all seeds (same code path, low risk).

---

## 3. PART A — candidate finite-moment identity statistics (derived, probed, ranked)

The refuted estimand E_{q_G}[R]·C* − 1 is the φ ≡ 1 member of an exact family: **for any
bounded measurable φ(d),**

    C* · E_{q_G}[ φ(d)·R(d) ]  =  E_{q_Ḡ}[ φ(d) ],        q_Ḡ ≡ B̃/β̄_Ḡ_φ,
    conditioned (F-0):  C* · E[ φR | acc ] · P̄_G  =  E_ξ[ ∫ B̃ φ 1_acc dd ] / β̄_Ḡ_φ.

φ ≡ 1 makes the left integrand unbounded (k̂ ≈ 1–2.7, no variance) and the right side trivial
(= 1). Every candidate below chooses φ to bound the left side, at the price of a right side
that must be **model-computed** — which is possible *without any venue Ḡ-class draws*, because
q_Ḡ is an analytically specified density (the completion predictive): this is the decisive
difference from the review's D-4 refutation of the reciprocal form (which needed the venue to
*realize* Ḡ-class events, structurally impossible in the catalogue-conditioned venue). The
per-seed median stays refuted (PA-6c: different functional); no candidate below is a median.

### 3.1 C-A (leading): the bounded-transform mixture identity, member φ = w

**Statistic (per arm, per seed, live rows, h = 0.73):**
`T_w(a) = C*·mean_e[(1−w_e)] · P̄_G − RHS_w(a)`, with
`RHS_w(a) = E_ξ[∫B̃·w_a·1_acc dd]/β̄_Ḡ_φ` computed once, model-side, by scoring synthetic
completion-predictive events (z ~ μ, donor ξ ~ SNR-weighted pool, d̂ = d_L(z)+σε, isotropic
sky; keep acc-passers) through the estimator's own per-event `L_cat`/`B_num` machinery at the
single node h = 0.73 — one draw set, scored under both arms' flags.

- **Exact identity:** φ = w ⇒ φR = w·R = (1−w) ∈ [0,1]. For a correctly arranged mixture the
  conditioned identity above holds exactly (same proof as §2.2; no new assumptions).
- **Moments — proved, not diagnosed:** both sides are means of [0,1]-bounded variables.
  Var ≤ 1/(4n) unconditionally; PSIS/k̂ machinery unnecessary; **the A21-B0-C band-vacuity
  mechanism cannot recur** (one event moves a seed mean by ≤ C*/n ≈ 1.5×10⁻³, vs the
  registered statistic where one event moved a seed by +62 to +98).
- **Banked leverage [LOCAL, zero-compute]:** left sides on the 12 banked pairs:
  C*·mean[1−w]: **B-T 0.07171 ± 0.00129, B-C 0.06317 ± 0.00118** (fleet SEM over seeds);
  paired Δ(B-T − B-C) = **+0.008544 ± 0.000166 (paired seed-level SEM), positive 12/12** —
  the two arrangements are separated at **~51σ by the banked data alone**, with bounded
  statistics, monster seed included (900108 is no outlier in this statistic: 0.0781/0.0691,
  mid-pack). (The arm-vs-arm separation is not itself a correctness verdict — it fixes the
  fork's *resolvability* in this statistic; correctness is LHS-vs-RHS per arm.)
- **Discrimination direction/magnitude [INFER]:** under the twin-correct hypothesis, B-T's
  T_w = 0 and B-C's sits displaced by a derived amount (its RHS differs: scoring w_BC over
  q_Ḡ upweights S̄-poor regions where w_BC > w_BT); under coded-correct the pattern inverts.
  The B-R-style control comes free: rescoring the twin columns by R(0.73) = 1.5155… (Finding 4
  [DOC]) shifts the left side deterministically — an at-predicted-value scorer check.
- **What breaks it under the coded arrangement:** the same §2.3 mechanism, now bounded — the
  coded leg's excess responsibility at S̄-poor z* produces a positive LHS−RHS displacement of
  order (1−⟨S̄⟩-type) × the LHS scale ~ 0.01–0.02, i.e. **≥ 8× the 0.0012 seed-level
  resolution** — the A17 ≥5× leverage threshold passes on banked numbers.
- **Blindness:** (i) inherits the PA-2 generator premise (as did the registered test); (ii) a
  defect in B̃'s *shape* (inside `precompute_phi_marginal_survival` or the completion
  integrand) enters LHS and RHS coherently — partially blind, exactly the §7(i) four-instrument
  blind spot, unchanged; (iii) the RHS's synthetic-event generator is new code — its
  acceptance/kernels must be gate-checked against the venue's own (an E-gate analog).
- **Refute-by clause (draft, for the eventual registration — not registered here):** with
  RHS pinned at SE ≤ 0.002 (≈3–5k accepted synthetic events), the twin arrangement's
  calibration is REFUTED if |T_w(B-T)| > max(3σ_comb, 0.005) with sign reported; the coded arm
  must land at its derived displacement within 3σ_comb (scorer/venue control, B-R-style);
  T_w(B-T) inside band with the coded arm mis-landing ⇒ VENUE-MISSPEC (F-0-model first
  suspect). Trim twins are unnecessary; the robustness twin is the second family member
  (winsorized C-TCI below) — band agreement required as in §4 of the b0 registration.
- **Zero-compute decision?** The LHS and the 4.6σ arm separation: yes, already banked. The
  absolute verdict: **no** — it needs the one-time RHS computation.
- **Costing line:** RHS: ~5k synthetic events × 2 arm-scorings at one h-node ≈ **2–4 CPU-h**
  (reusing the b0i venue's per-event machinery; no fleet, no new seeds); everything else
  zero-compute from the banked CSVs. Total ≤ 4 CPU-h + review.

**C-TCI (same family, hard-indicator member — discharges the task's "trimmed/winsorized with
the truncation bias derived" direction):** φ = 1{R ≤ τ} gives
`C*·E[R·1{R≤τ}] = 1 − Q_Ḡ(τ)` with `Q_Ḡ(τ) = P_{q_Ḡ}(R > τ)` — **the truncation correction
is exactly a completion-predictive tail probability**, computable from the same synthetic
event set (count R > τ). It is NOT ignorable: from the banked left sides, under twin-correct
the implied Q_Ḡ(τ=100) ≈ 1 − 0.438/1.591-scaled ≈ 0.5–0.7 — the review's trimmed reads
(−0.67, −0.80, −0.92) discarded over half the identity's mass, which is why PA-6(c)-era trims
mislead. Variance bound: C*τ/√n ⇒ τ ∈ [10², 10³] gives seed-level SEM 0.02–0.09 [LOCAL,
measured on banked vectors]. Registered role: robustness twin to φ = w, τ-profile reported.

### 3.2 C-B: the paired catalogue-leg log-LR Λ (zero-compute; probed here; found F-0)

**Statistic:** per paired-live event (same seed ⇒ same realization, verified pairing),
`Λ_e = ln(L_cat^BT_e / L_cat^BC_e) + ln(Σ_w/Σ̃^φ)`, center = +0.2542072 [LOCAL from banked
Σ's]. Exact structure: the two normalized class-G predictive densities g_T ∝ A_BT/d̂,
g_C ∝ A_BC/d̂ have masses β_G_φρ and β_G_φΣ_w/Σ^φ (both derived in §2.2's frame), so
Λ_e = ln(g_T/g_C)(d_e) and, **if the realized generator law equals g_T**,
E[Λ] = KL(g_T‖g_C) ≥ 0, while a coded-law generator gives E[Λ] = −KL(g_C‖g_T) ≤ 0 — a
Neyman–Pearson-grade sign-separated discriminator of the fork, fully **zero-compute** on the
banked CSVs (β_G_φ, D̃_φ, B_num all cancel: it reads only the two L_cat columns).

- **Moments — bounded:** L^BT ≤ L^BC per event (S̄_φ ≤ 1) ⇒ Λ_e ≤ 0.2542 exactly (observed
  max +0.2411 ✓); below, bounded by −sup|ln S̄_φ| on the reachable window; observed pooled
  range [−0.989, +0.241], per-seed sd 0.17–0.22, no heavy tail (the 900108 monster is
  mid-pack: seed Λ̄ = −0.036). Finite variance is empirical-plus-structural here (the hard
  lower bound depends on S̄_φ's floor on the realized support — state as a bound-in-venue).
- **Probe result [LOCAL, all 12 banked pairs]:** **Λ̄ = −0.02516 ± 0.00454 (12/12 seeds
  negative; clean-11: −0.02417 ± 0.00486).** Naively (pre-F-0) this *violates* the KL ≥ 0
  floor at 5.5σ. Full accounting, each term computed this session [LOCAL]:
  - per-event sharp-pin check: lnW̃_e vs lnS̄_φ(z_true,e) (truth from the mirrored
    `prepared_cramer_rao_bounds.csv`, 10 locally-mirrored seeds, 1157 events):
    mean residual **+0.0002**, sd 0.023 — the statistic reads S̄_φ(z_true) essentially exactly;
  - intended-law moment (kernel-smeared, chunked doctored-table pass over the full 20.8M-row
    pool): L₁ = E_intended[lnS̄_φ] = **−0.23565** ⇒ KL(g_T‖g_C) = +0.01856 ≥ 0 ✓ (theorem
    holds for the *intended* law);
  - realized draws: E[lnS̄_φ(z_true)] = **−0.27942 ± 0.0056** — 0.044 below the intended law
    ⇒ the realized law ≠ intended law ⇒ **F-0 found** (§2.5); the acceptance MC reproduces
    the tilt: point-eval −0.2238 → acc-conditioned −0.2644 (Δ = −0.0406), plus the
    point→smeared shift −0.0118, total −0.276 vs measured −0.2794 (residual 0.003 ≈ 0.5σ);
  - eliminated candidate owners [LOCAL]: draw-weight concentration (max p = 1.2×10⁻⁷ ⇒
    without-replacement bias ~10⁻⁵), pool-eligibility mismatch (weight share beyond the table
    grid = 7.5×10⁻⁷), PV term (σ_V = 0, no-op), M ≤ 0 rows (none).
- **Null and decision capacity:** conditioned twin-null ≈ KL − acc-tilt ≈ −0.022 ± ~0.010
  (model error dominated by the point-eval acceptance MC); measured −0.0252 ± 0.0045 —
  **consistent with twin-correct under the venue as executed**. Coded-null ≈ −KL' + same tilt
  ≈ −0.06 ± 0.01. The two nulls separate at ~3σ *today* and at >5σ once the null is pinned by
  the acceptance-weighted smeared companions (two more doctored-table passes + the smeared
  acceptance function A(z) on the grid — **zero `evaluate()`, ≈1 CPU-h**). **Answer to the
  task's leverage question: yes — the 12 banked seed-pairs decide C-B zero-compute once that
  1-CPU-h null-pinning pass is done; no new venue runs are needed.**
- **Blindness:** Λ tests *only* the catalogue-leg z-weighting fork — it is structurally blind
  to the completion leg, D̃_φ, and the mixture weights (complement of C-A, which tests the
  mixture); it inherits the PA-2 premise (the generator *is* the twin's law by construction,
  so a "twin-favoring" verdict is partially by construction — the informative failure mode is
  the other direction, and the magnitude-vs-KL-prediction check is what makes PASS non-vacuous).
- **What would make it wrong:** the centering constant is venue-level (Σ_w, Σ̃^φ) — a
  draw-vs-companion kernel mismatch shifts the null coherently (this session's probes bounded
  the known candidates to ≲10⁻⁵ except F-0, now modeled); the sharp-pin premise fails if a
  rerun admits σ_frac ~ O(1) events (F-0 currently excludes them); the lower bound on Λ_e
  (hence variance) is venue-empirical, not universal.

### 3.3 C-C: PIT/rank calibration of w under the model's own conditioned class-G predictive

Compute model-side the predicted CDF F_a of w_a under (q_G, acc) per arm (same synthetic-event
machinery as C-A's RHS but drawing from the *class-G* side, which the b0i generator already
implements); score venue events' u_e = F_a(w_e); test uniformity (mean-PIT ± band; AD as
reported-only). Bounded by construction; tests the whole *distribution* rather than one moment
(catches compensating-defect patterns C-A's single moment can miss — a direct answer to §7(ii)
blindness (ii)). Costlier (~5–10 CPU-h: per-event predictive CDFs need ~10× the synthetic
events) and its band calibration under 12-seed clustering needs design work. Ranked third;
natural *second* measurement if C-A/C-B disagree.

### 3.4 Ranking and non-reproposal check

| rank | candidate | tests | variance | venue cost | zero-compute decision? |
|---|---|---|---|---|---|
| 1 | C-A φ=w (+C-TCI twin) | mixture calibration (the original estimand) | proved ≤ 1/4n | RHS 2–4 CPU-h, once | LHS + 51σ paired arm separation: yes; verdict: after RHS |
| 2 | C-B Λ | the catalogue-leg fork | bounded in venue | ~1 CPU-h null pinning | **yes** (12 banked pairs) |
| 3 | C-C PIT | distributional calibration | bounded | 5–10 CPU-h | no |

Neither refuted alternative is re-proposed: the per-seed median (PA-6c) targets a different
functional and appears nowhere; the reciprocal form (D-4) needed venue Ḡ-draws — C-A's right
side replaces those with model-side integration of a *bounded* function under the analytically
specified completion predictive, which is new derivation content, not a re-proposal.

---

## 4. RETURNS to the orchestrator/author (nothing registered)

1. **F-0** (§2.5) — the σ/d̂ < 0.10 intake filter conditions the b0i identity venue at the
   41.8% level, class-asymmetrically, and was not in the registration's targets or §7
   blindness list. Decision needed: (a) treat as a venue bookkeeping term via the conditioned
   targets (as derived here) in any rerun/re-scoring, or (b) amend the venue (e.g., donor
   re-pairing or filter-aware draw) — an A21-class premise question for the family that
   A21-B0-C already binds. Also: the banked REPORTED-ONLY "B-T closer 11/11" quotation should
   carry the F-0 caveat from now on (§2.5 item 3).
2. **The mass-derivation question (review RETURN 1):** M_Ḡ = β̄_Ḡ_φ is exact for the odds
   statistic (§2.2); the ×2.34 decomposes into PA-4-weighting (×2.223, derived+computed) ×
   conditioning × ordered tail deficits (§2.3–2.6); no C* rescale fits both arms. The C*
   *derivation* stands; its PA-4 *point predictions* were wrong.
3. **Candidate ranking (§3.4)** with C-A as the proposed next centerpiece statistic and C-B as
   the free corroborator; registration drafts to be authored only after the author rules on
   F-0's disposition (the targets depend on it).
4. **Production-side note (scope-limited):** the 0.10 filter is production's own intake and is
   *not* modeled in `simulation_detection_probability`/S̄_φ; in production it is nearly
   inactive (the donor pool is post-filter at its own distances), but any future venue or
   population where it bites is exposed to the same unmodeled-selection term — a one-line
   check ("fraction dropped at intake") is worth adding to run metadata.
5. Numbers bank-ready if the author wants them quoted (all [LOCAL], zero-evaluate, this doc):
   the corrected targets (0.35321 unconditioned; 0.527/1.591 conditioned at P̄_G = 0.5821),
   the moment set (β̄_Ḡ = 1.06778×10¹⁰, ⟨S̄⟩_μ0 = 0.08320, ⟨S̄²⟩/⟨S̄⟩_μ0 = 0.35321,
   L₁ = −0.23565, KL = +0.01856), the Λ̄ probe (−0.02516 ± 0.00454, 12/12), the C-A left
   sides (0.07171/0.06317 ± 0.0013, Δ 12/12 positive), and the F-0 forensics.

*(Draft ends. STATUS: DRAFT — returns to the orchestrator/author; register NOTHING.)*
