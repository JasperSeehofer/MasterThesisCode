# A20-STYLE PRE-EXECUTION DESIGN REVIEW — C-A registration (banked verbatim, 2026-08-24)

**Reviewer:** clean-context adversarial agent (inherit-model, xhigh). **Verdict: BLOCKED** (Finding 1 FATAL — the P̄_G conditioning factor dropped in the prereg encoding, a guaranteed ~19σ spurious TWIN-MISCALIBRATED; Findings 2–4, 6, 7 amendment-class with exact replacement text). **Disposition:** PA-CA-1…PA-CA-9 folded before commit; NO instrument has run for the registered statistic.

# ADVERSARIAL PRE-EXECUTION DESIGN REVIEW — PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md (round 1)

All decisive numbers below re-derived independently from the 24 banked CSVs + selection JSONs (my recomputation reproduces every quoted banked value exactly: all-rows LHS B-T 0.07279±0.00136, B-C 0.06435±0.00124, paired Δ +0.008447, 12/12; n_kept = 106,120,105,130,111,124,112,121,108,131,116,113; 15 dead rows fleet-wide; all 24 metas `catalogue_global_selection:"phi"`, cell `fused`, 12 phi/12 off).

---

## FINDING 1 (Item A) — the conditioned identity is mis-encoded: §1's T_w drops the P̄_G factor. FATAL as written. VERDICT: **AMEND**

The unconditioned member is exact and faithfully the adjudicated C-A: φ=w ⇒ φ·R = w·(1−w)/w = 1−w, and substituting p_gen = A_BT/(β_G_φ·ρ·d̂), B_num = d̂·B̃ gives C*·E_G[1−w] = E_Ḡ[w] with C* = β_G_φρ/β̄_Ḡ_φ = 0.1704718 (recomputed ✓). **But the exact F-0-conditioned identity is**

  C*·E_G[(1−w)·1_acc] = E_Ḡ[w·1_acc]  ⇔  C*·E_G[1−w | acc]·**P̄_G** = **P̄_Ḡ**·E_Ḡ[w | acc].

Acceptance depends on the **drawn class's** (z,d̂,ξ) law: P̄_G = 0.5821 ≠ P̄_Ḡ = 0.9269 — the very asymmetry the adjudication computed (§1d). The prereg's §1 statistic `T_w = C*·E_G-banked[1−w] − RHS_model` with §2's scorer "accumulat[ing] E_Ḡ[w]" over accepted synthetic draws conditions each side on **its own** acceptance event, which breaks the identity by the factor P̄_Ḡ/P̄_G = 1.592. Quantified: under twin-correct, LHS_all = 0.07279 while E_Ḡ[w|acc] = 0.0457 (or E_Ḡ[w·1_acc] = 0.0423) ⇒ T_w(B-T) = **+0.027 to +0.031 ≈ 19–22σ_comb — a guaranteed false TWIN-MISCALIBRATED verdict**. The claim doc §3.1 carried the ·P̄_G explicitly; the adjudication §3a/§4 confirmed that form; the prereg's encoding dropped it. This is exactly "the place a wrong constant hides."

Also unregistered and load-bearing: the operational definition of w. The CSVs' `w_G`/`w_tilde_G` columns are **NOT** the identity's w (max |w_G − w| = 0.935 at h=0.73, verified) — a scorer reading them produces garbage.

**Replacement text for §1's verdict-statistic sentence + conventions (i):**

> **Verdict statistic per arm a (drawn-count normalization; exact under F-0):** T_w(a) = LHS(a) − RHS_model(a) at h = H_TRUE = 0.73, with per-seed LHS_s(a) = (C\*/200)·Σ_{accepted rows e}(1−w_e), fleet mean over 12 seeds; RHS_model(a) = (1/N_syn)·Σ_{ALL synthetic draws j} w_a(d_j)·1_acc(d_j) — normalized by ALL completion-class draws with the F-0 filter applied exactly per draw (σ_dL < 0.10·d̂; the donor pool's SNR ≥ 20 holds by construction, so the SNR clause is inactive and no acceptance MODEL enters the verdict statistic), never the accepted-draw conditional mean. The identity C\*·E_G[(1−w)·1_acc] = E_Ḡ[w·1_acc] is exact (adjudication §3a; equivalently C\*·E[1−w|acc]·P̄_G = P̄_Ḡ·E_Ḡ[w|acc] — the 1d class asymmetry P̄_G = 0.5821 vs P̄_Ḡ = 0.9269 is carried by construction, with the realized per-seed n_kept/200 internal to LHS_s). Both summands are [0, max(1,C\*)]-bounded; variance ≤ C\*²/4·200 per seed. **Banked LHS under this normalization: B-T 0.04233 ± 0.00108, B-C 0.03741 ± 0.00095 (12-seed SEM), paired Δ = +0.004919 ± 0.000146, positive 12/12.** Dead rows INCLUDED (w=0 ⇒ 1−w=1; 15 rows fleet-wide). **w registered operationally:** per CSV row at the h = 0.73 node, w_e = A_e/(A_e+B_e), A_e = β_G_φ(0.73)·`L_cat_no_bh`_e with β_G_φ from the per-seed `selection_tables_h_0_73.json`, B_e = `B_num`_e; the CSV's `w_G`/`w_tilde_G` columns are NOT this w and must not be read.

The T_w construction itself (banked LHS minus model RHS, per arm) is the right verdict statistic — LHS-vs-RHS tests calibration; the 51σ arm separation alone does not — **CONFIRMED once amended as above**.

## FINDING 2 (Item B) — D_C is derivable and non-circular, but the prereg registers no formula. VERDICT: **AMEND**

Derivation: the generator law is the twin predictive (PA-2), so for the coded arm, substituting p_gen = A_BT/(β_G_φρd̂) into C*·E_G[(1−w_BC)·1_acc] yields **E_Ḡ[W̃·w_BC·1_acc]** with W̃ = A_BT/A_BC = L_cat^BT/L_cat^BC (same β_G_φ, same Σ^φ divisor in both arms; W̃ ≤ 1). Hence

  **D_C = E_Ḡ[(W̃ − 1)·w_BC·1_acc]** — every factor model-side, accumulated on the SAME synthetic set scored under both arms' flags. No circularity. Sign prediction D_C < 0, consistent with banked LHS(B-C) < LHS(B-T) (0.03741 < 0.04233 ✓). The criterion collapses: |T_w(B-C) − D_C| ≤ band ⇔ **|LHS(B-C) − E_Ḡ[W̃·w_BC·1_acc]| ≤ band** — register the collapsed form (one MC error; no RHS/D_C correlation bookkeeping). Convention: synthetic rows with L_cat^BC = 0 contribute 0.

**Verdict-map reachability defect (same item):** for any generator deviation δ(d), T_w(B-T) = E_Ḡ[(δ−1)w_BT·1_acc] and T_w(B-C) − D_C = E_Ḡ[(δ−1)W̃w_BC·1_acc], with W̃w_BC = A_BT/(A_BC+B) comparable to w_BT (ratio (A_BT+B)/(A_BC+B) ∈ [~0.6,1]). A genuine twin miscalibration therefore drags B-C off D_C by a proportional amount (exactly κ̂·T_w(B-T) for a common mass rescale), so §4's TWIN-MISCALIBRATED cell ("with B-C at its derived value") is unreachable for any violation ≳2× the band and mis-routes to VENUE-MISSPEC. **Replacement:** register the pre-frozen coherence slope κ̂ = E_Ḡ[W̃w_BC·1_acc]/E_Ḡ[w_BT·1_acc]; TWIN-MISCALIBRATED ⇔ |T_w(B-T)| > band AND |(T_w(B-C) − D_C) − κ̂·T_w(B-T)| ≤ max(3σ_comb, 0.005); VENUE-MISSPEC ⇔ the B-C deviation inconsistent with that coherence relation.

## FINDING 3 (Item C) — GATE ACC is not scoreable as written. VERDICT: **AMEND**

Three holes: (a) "central binomial bands" has no coverage level — at per-seed 95%, a **correct** model passes all 12 only ~54% of the time; (b) the binomial p is unspecified (common p̄ vs per-seed p̄_s) — realized counts are overdispersed (sd 8.84 vs binomial(200,0.5821) sd 6.98; χ²₁₁ = 17.6, p ≈ 0.09, recomputed), so a common-p̄ band is materially narrow if the class-G draw context varies by seed; (c) "within its stated error of 0.5821" names no error. What must improve: the current point-eval no-smear model (P_G 0.6245, +7%) centers bands at ~124.9, expelling counts 105/106/108/111 and failing the fleet clause at ~6σ — the scorer must replay the class-G b0i draw law kernel-smeared with the exact per-draw filter. **Replacement:**

> **GATE ACC:** the scorer replays the class-G b0i draw law (kernel-smeared z; per-seed donor context as drawn — state whether p̄ is seed-invariant) and applies the exact F-0 filter per draw. PASS iff (i) every realized n_kept,s lies in the central binomial(200, p̄_s) band at per-seed coverage 99.6% (joint false-STOP ≈ 5% over 12 seeds), AND (ii) |P_model − 0.5821| ≤ 2σ_P, σ_P² = 0.5821·0.4179/2400 + SE_model². Registered overdispersion note: realized seed sd 8.84 vs common-p 6.98 (p ≈ 0.09); a clause-(i) failure with clause (ii) passing is a FINDING on the draw law's seed conditioning — STOP, A21-amend. With the Finding-1 normalization, NO acceptance-model number enters T_w; GATE ACC is venue-fidelity only, and §5(iv)'s "residual acceptance-model error budget in the verdict" is restated accordingly.

## FINDING 4 (Item D) — the B-R predicted value is not any transform of the scalar RHS(twin); the one-liner is unexecutable. VERDICT: **AMEND**

Under A → r·A (r = R(0.73) = 1.515548762178686, the Σ³ᴰ/Σ^φ rescore ratio), per-row exactly: 1−w′ = (1−w)/(1+(r−1)w). The rescale does **not** commute with the expectation (Jensen): candidate scalar readings of "its derived transform of RHS(twin)" differ from the correct value by up to ~0.008 > the 0.005 tolerance — the gate as written can be passed or failed by choice of reading. **Replacement:**

> **GATE B-R:** banked side (deterministic, zero-compute): LHS_BR = (C\*/200)·Σ_acc(1−w_e)/(1+(r−1)w_e) over the 12 bt frames — **banked value 0.03571 ± 0.00093** (computed this review; freeze at registration). Model side: RHS_BR = E_Ḡ[w/(1+(r−1)w)·1_acc], a NEW expectation of the transformed integrand accumulated on the same synthetic set (identity: C\*·E_G[(1−w′)·1_acc] = E_Ḡ[w/(1+(r−1)w)·1_acc], same substitution as §2.2). PASS iff |LHS_BR − RHS_BR| ≤ 0.005. C\* stays un-rescaled on both sides (registered convention; the C\*′ = rC\* alternative differs by ×1.52 — state once).

Design synergy worth registering: an Σ-slot-skewed scorer mis-scales A by exactly 1/r, so B-R + RHS-F jointly pin the Finding-6 hazard.

## FINDING 5 (Item A/E) — dead-row convention and banked-LHS quotes: **CONFIRMED** (with the Finding-1 renormalization)

Dead-row inclusion is coherent both sides (LHS: 1−w = 1; RHS: empty-ball synthetic rows contribute w = 0 — symmetric), faithfully encodes adjudication §3a, and my recount confirms the effect (~1.1% of accepted rows, 15 fleet-wide, live→all shift 0.07171→0.07279 ✓). The quoted numbers 0.07279/0.06435/+0.008447 are exactly reproducible — but must be **replaced** by the drawn-count-normalized set (0.04233/0.03741/+0.004919) per Finding 1.

## FINDING 6 — the Σ^φ comparand-skew hazard is real, one gate away from FATAL. VERDICT: **AMEND (tighten A22 clause)**

Verified: all 24 banked metas ran `catalogue_global_selection:"phi"` explicitly at 71b52e9c (pre-adoption). Current HEAD: `evaluate()`'s "auto" resolves to phi under absolute_marginal — but the **bare class fallback is "s3d"** (`bayesian_statistics.py:3231` and `:3286`, consumed via `getattr(self, "_catalogue_global_selection", "s3d")` at `:4900`), and the code comment says this fallback is live precisely for "pre-evaluate()/object.__new__ harnesses" — the established instrument pattern the RHS scorer will use. An s3d-defaulting scorer computes L_cat smaller by ×1/1.5155, skewing RHS by ~0.007 ≈ 5σ_comb. GATE RHS-F (≤1e-6) catches it **if run per-arm before accumulation**. Replacements: (a) §3 A22 clause: "both flag values" → "all THREE resolved flag values — completion_cell='fused', catalogue_numerator_survival (per arm), catalogue_global_selection='phi' — stamped as RESOLVED values, never 'auto'; the scorer constructor passes catalogue_global_selection='phi' EXPLICITLY"; (b) GATE RHS-F: "run on BOTH arms (≥1 full seed per arm; all 24 CSVs recommended), h = 0.73 rows, BEFORE any RHS accumulation; dead rows reproduce as exact zeros."

## FINDING 7 (Item E) — bands, C-TCI, C-B, costing, manifest. VERDICT: **AMEND (five sub-items)**

- **7a Bands:** replace SEM quotes per Finding 1 (1.08e-3/0.95e-3; σ_comb(B-T) ≈ 1.19e-3, anchor ≈ 4.2σ — improves). State the 12-seed convention: normal quantiles or t₁₁ (3σ → 3.35) — pick one.
- **7b C-TCI:** "winsorized" contradicts the adjudicated derivation (hard indicator). Register ONE member — recommended the indicator: (C\*/200)Σ_acc R_e·1{R_e≤τ} vs E_Ḡ[1{R≤τ}·1_acc]; τ grid {30, 100, 300, 1000}; per-τ σ (seed SEM ⊕ RHS count error, ~0.02–0.09). State the power asymmetry: per-τ bands are 4–18× the 0.005 anchor, so agreement is weak evidence and only a >3σ_τ discrepancy is a falsifier. Dead rows auto-consistent (R = ∞ excluded both sides).
- **7c C-B contradiction:** §2 "REPORTED-WITH-VERDICT, not verdict-bearing" vs §6 using C-B as a falsifier of TWIN-CALIBRATED. Fix: C-B enters §6 only if its pinned coded-null and twin-null separate ≥3σ_null AND the measured Λ̄ = −0.02516 ± 0.00454 lies ≥3σ closer to the coded-null; otherwise REPORT-ONLY.
- **7d Costing:** SE ≤ 5e-4 with sd_Ḡ(w·1_acc) plausibly 0.1–0.2 (venue-side sd(w) = 0.31 measured) needs N ≈ 4×10⁴–1.6×10⁵ accepted draws ≈ 20–90 CPU-h at the claim's per-event rate — the "~10 CPU-h" is 2–9× optimistic. Register the fallback: SE ≤ 1e-3 acceptable (σ_comb 1.19e-3 → 1.47e-3, anchor ≥ 3.4σ, still LHS-limited); cap N; state realized SE at the σ freeze.
- **7e Manifest:** §5's "sha256-manifested" is FALSE for 4 of 24 CSVs — `retrieval_manifest_20260824.json` covers the cluster retrieval only; **seeds 900102 and 900103 are absent**. Generate the complete 24-CSV sha256 manifest in the registration commit.

## FINDING 8 — k̂ invariant, GATE W, invariants block, falsifier structure. VERDICT: **CONFIRMED**

k̂-as-finding on a bounded summand is coherent; W-B0 closure reuse well-defined; C\* = 0.1704718 and R(0.73) recomputed ✓; the "one venue, one h" blindness items faithful to the adjudication; §6's GATE-ACC-on-re-audit falsifier executable once Finding 3 lands.

---

# OVERALL VERDICT: **BLOCKED**

Finding 1 is fatal as written (a guaranteed ~+0.03 ≈ 19σ spurious twin-miscalibration signal from the P̄_Ḡ/P̄_G conditioning mismatch, plus an unregistered w definition beside decoy CSV columns); Findings 2, 3, 4 are unexecutable-gate/underived-comparand class; 6 and 7 are pre-commit tightenings. All replacement text is supplied above; with Findings 1–4, 6, 7 folded in, the design is sound — the corrected statistic is exact, model-free in P̄_G, bounded, and its banked side is already in hand (B-T 0.04233 ± 0.00108, B-C 0.03741 ± 0.00095, Δ +0.004919, 12/12; B-R banked 0.03571 ± 0.00093). REGISTRATION-READY after amendment, second-round review recommended on the amended text only.

Files: subject `/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md`; evidence re-derived from `/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/p3_b0_work/{bt,bc}_9001*/seed*/simulations/diagnostics/event_likelihoods.csv` + `selection_tables_h_0_73.json`; code slots `darksiren_emri/bayesian_inference/bayesian_statistics.py:386,3231,3286,3490-3519,4896-4905,5540-5554,6057-6093`; manifest gap `p3_b0_work/retrieval_manifest_20260824.json`.