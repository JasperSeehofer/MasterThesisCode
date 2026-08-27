# Stage-L fresh search — [P3-2D] STUCK residual (2026-08-27)

**Searcher independence statement.** I read exactly one project document:
`STUCK_P3_2D_SYMPTOM_CARD_20260826.md`. I read no ledger, claim, pre-registration, runbook,
handoff, adjudication, review, forensic or retrospective file, and no git log. Everything else
below comes from (a) the pipeline and harness **source code**, (b) **banked numeric artifacts**
(JSON/CSV outputs, which are data, not opinion), and (c) the published literature. Where I wanted
a forbidden file I say so explicitly (§6).

Files I read as evidence:
`darksiren_emri/validation/correspondence_1d.py`,
`darksiren_emri/bayesian_inference/bayesian_statistics.py`,
`results/campaign51_20260728/realistic_20260729/p3_2d_companion.py`,
`.../p3_2d_fleet.py`, `.../p3_b0_identity_test.py`, `.../ca_rhs_scorer.py`, and banked outputs
under `.../ca_rhs_work2d/`, `.../p3_2d_work/`, `.../p3_2d_rhs2_20260826/`, `.../ca_rhs_work/`.

---

## 1. The symptom, restated

### 1.1 As the card states it

Two estimates of the same quantity disagree by a stable multiplicative factor:

- **Banked side:** a per-seed statistic computed over ~**24** realizations of a 2-D
  (distance, mass) mirror venue, each realization being a set of *accepted* events.
- **Model side:** the matching expectation over ~**25,600** synthetic draws from the model's
  completion-class law.
- **Ratio banked/model = 0.345 ± 0.013** (primary) and **0.366 ± 0.014** under a registered
  nonlinear rescale of the same objects. Equivalently a **×2.5–2.9 deficit** of the banked side,
  **~5×** the pre-registered acceptance band.
- The deficit is **arm-independent** and therefore **common-mode**.
- The **normalization constant** joining the two sides was re-derived blind and is arithmetically
  exact (two algebraic forms agree to float round-off); its only nontrivial input is a
  **20.8M-row zero-evaluate contraction**, itself arbitered to ~1e-9.
- **Mass-observable linkage on the model side is verified**: two counterfactual re-scorings that
  alter the mass assignment move the statistic by **×0.05** (independent-host swap) and
  **×0.9997** (own-mass re-redshifting). "Neither reproduces ×2.5."
- A **real, measured selection double-weighting** exists in the empirical venue's draw law
  (**13.5–16% tilt**) — sign-correct but **~7× too small**; the fix is authorized but unrun.
- The **1-D member** of the same identity family **closes in band** in the same codebase. The 2-D
  extension added: a latent mass dimension, a two-stage accepted-latent draw (weighted z-draw +
  Bernoulli survival thinning + whole-event rejection), and the new analytic contraction.
- **Per-seed scatter is small** (SEMs 2–3% of the means): "not a variance problem."

### 1.2 What the card does not say, that I recovered from code and artifacts

The card is written at an abstraction that hides the two facts I found most decision-relevant.
For the record, the objects are:

- **Identity:** `C* · E_{q_G}[(1-w)·1_acc] = E_{q_Ḡ}[w·1_acc]`, the exact class-odds identity for a
  two-component mixture `p(d) = M_G q_G(d) + M_Ḡ q_Ḡ(d)` with
  `w(d) = M_G q_G/(M_G q_G + M_Ḡ q_Ḡ)` and `C* = M_G/M_Ḡ`. (Derivable in two lines; holds with any
  common bounded `1_acc(d)`.)
- **2-D instance** (`p3_2d_fleet.py --stage lhs2d`, `ca_rhs_scorer.py --stage rhs2`):
  - LHS2 per seed `= (C2*/200)·Σ_accepted (1 - w2_e)`, with
    `w2_e = α_G^φ·L_cat_with_bh / (α_G^φ·L_cat_with_bh + B_num_wbh)`;
    200 = the number of accepted latents handed to the pipeline per seed (venue `b0i2d`).
  - `C2* = β_G^φ · Σ̃^4D / (Σ^φ · β_Ḡ^φ)`; banked `C2* = 0.06124403`
    (`ca_rhs_work2d/p3_2d_companion_v2.json`), `Σ̃^4D = 3.48079e8` over `n_eligible = 20,834,132`
    — that is the 20.8M-row contraction.
  - RHS2 `= E_Ḡ[w2·1_acc]`, drawn-count normalized, over `host_mode="population_selected"` draws;
    banked per-task values ≈ **0.0151 ± 0.0015** per 800 draws (32 tasks × 800 = 25,600).
- **Selection-table scalars at h=0.73:** `β_G^φ=1.53323e8`, `β_Ḡ^φ=8.88404e8`, `Σ^φ=9.80867e8`,
  `Σ^4D=3.75453e8`, **`r_Malm = Σ^4D/Σ^φ = 0.3827762`**, so **`1/r_Malm = 2.6124925`**.
- **Acceptance fractions differ enormously between the two sides**: class-G venue ≈ **84/200 = 42%**
  F-0-accepted (pilot seed 900101); class-Ḡ ≈ **177–190/200 = 89–95%**.

**Two facts I measured directly that the card omits and that reframe the whole search:**

**(F-A) The model side (RHS2) is a *tail-dominated* functional.** Over 60 banked class-Ḡ chunks
(12,000 drawn, 11,239 accepted, scored with the with-BH columns):

| statistic | value |
|---|---|
| drawn-count mean (RHS2-style) | 0.01042 |
| `L_cat_with_bh == 0` (no mass-compatible candidate) | **73%** of accepted rows |
| top 100 events (0.89% of accepted) share of the mean | **56%** |
| top 50 (0.44%) share | 33% |
| events with `w2 > 0.5` | 89 of 11,239 (0.79%) |

**(F-B) That tail lives at the low-distance edge of the completion-class law.**
`w2`-weighted mean `d_L` = **0.8** vs unweighted **2.6** (same units, ~Gpc); the `w2>0.5` events
have median `d_L` 0.6 against a population median of 2.5. So RHS2 is set almost entirely by the
rare "impostor coincidence" events in the innermost ~1% of the completion-class draw — precisely
where `1 - f̄(z)` is smallest and hardest to model.

This is the single most important structural difference from the 1-D member: in 1-D,
`L_cat_no_bh` is nonzero for most events and `E_Ḡ[w]` is a well-conditioned mean; in 2-D the
with-BH mass factor zeroes 73% of the events and hands the mean to a ~1% tail. **A defect in the
class-Ḡ law that is first-order harmless in 1-D can be amplified by O(1)–O(10) in 2-D purely
through leverage.** That alone can explain "1-D closes, 2-D does not" without any new 2-D-specific
bug — and it means the previous search strategy (hunt for one big new defect) may be looking for
an object that does not exist.

---

## 2. Candidate mechanisms

Each entry: mechanism · why this symptom (sign/magnitude) · cheapest discriminator · confidence
and refutation.

Sign convention: the banked (LHS2) side is **too small** by 2.5–2.9, equivalently the model side
(RHS2) is **too large** by the same factor, or `C2*` is too small by it.

---

### C1 — Residual `S̄_φ(z)` factor in the 2-D accepted-latent draw law *(LHS side, code-derived exactly)*

**Mechanism.** The `b0i2d` generator draws hosts `∝ w_g · S̃_φ,g`
(`catalogue_selected_host_draw_weights`, `correspondence_1d.py:1341`), then `z_true` per host from
`k_g(z)·w_pop_eff(z)·S̄_φ(z)` (`_draw_kernel_survival_redshifts`, `:1440`), then a latent mass
`M ~ N(M_eff, σ_M)`, then accepts the whole triple with `Bernoulli(S_4D(d_L(z), M(1+z)))`
(`_draw_2d_accepted_latents`, `:1605`). Because the per-host `z` draw is normalized *within* the
host's window, `S̃_φ,g` cancels exactly, and the realized joint law is

```
q_real(g, z, M)  ∝  w_g · k_g(z) w_pop_eff(z) · S̄_φ(z) · p_gal(M|g) · S_4D(z, M)
```

while the target law implied by `Σ̃^4D = Σ_g w_g S̃_4D,g` (the companion contracts
`∫ k_g w_pop_eff · E_M[S_4D] dz`, `p3_2d_companion.py`) is

```
q_target(g, z, M) ∝  w_g · k_g(z) w_pop_eff(z) ·          p_gal(M|g) · S_4D(z, M)
```

The ratio is **exactly `S̄_φ(z_true)`**, with no per-host constants surviving. This is a clean,
structural, 2-D-only double-application of the φ-marginal survival: the 1-D venue applies `S̄_φ`
*once* (there is no Bernoulli stage), which is why the 1-D member closes.

**Why this symptom.** `S̄_φ(z)` decreases with `z`, so the venue over-populates low `z`. Low-`z`
accepted events are better localized and better catalogue-supported ⇒ `w2` larger ⇒ `(1-w2)`
smaller ⇒ **LHS2 too small. Sign correct.** Magnitude is set by the dynamic range of `S̄_φ` across
the accepted `z` range; a factor of a few is entirely plausible and is *not* bounded by the
"13.5–16% tilt" the card reports, because that number appears to measure a distributional tilt,
not the importance-reweighted change in the statistic (see C1′).

**Cheapest discriminator (zero `evaluate()`).** Every seed's `prepared_cramer_rao_bounds.csv`
banks `z_true` for **all 200** drawn latents (columns verified present: `z_true`, `M_true`,
`M_z_true`, `M_z_obs`, `s4d_at_truth`, `host_galaxy_index`, `s_tilde_phi_host`). With the cached
`phi_survival_table` compute

```
LHS2_corrected = C2* · Σ_accepted (1-w2_e)/S̄_φ(z_e)  /  Σ_all-200 1/S̄_φ(z_e)
```

(self-normalized importance reweighting from `q_real` to `q_target`). If this moves LHS2 by
≈2.5–2.9×, C1 is the mechanism. Cost: one pandas pass per seed; minutes.

**Confidence: HIGH (that the mismatch exists — it is a code identity; MEDIUM-HIGH that it carries
the full factor).** Refuted if the reweight moves LHS2 by ≲20%, i.e. if `S̄_φ` is nearly constant
over the accepted `z` support.

**C1′ (a warning about the "13.5–16%" number).** A double-weighting of this shape has *two*
distinct magnitudes: the shift it induces in a marginal distribution (small if `S̄_φ` is smooth),
and the change it induces in a *tail-weighted* statistic (potentially large). If the banked
13.5–16% was measured as the former, the "~7× too small" conclusion is an artifact of the
measurement functional, not evidence against the mechanism. **Measure the tilt as
`ΔLHS2/LHS2` under the importance reweight, never as a quantile shift.**

---

### C2 — RHS2 is tail-dominated, so any small class-Ḡ law defect is amplified *(model side, frame-level)*

**Mechanism.** Not a single bug but a leverage statement, established empirically above (F-A/F-B):
~1% of class-Ḡ draws carry >50% of RHS2, and they are the low-`d_L` impostor coincidences. The
class-Ḡ draw law (`draw_selected_population_redshifts` + `draw_isotropic_sky`) is an
*approximation* of the estimator's completion predictive in at least three ways, each of which is
small in `L¹` but concentrated exactly in that tail:

- **(a) sky/completeness factorization.** `selected_population_z_weights`
  (`correspondence_1d.py:895`) weights `z` by the **sky-averaged** deficit `1 - f̄(z)`, and the sky
  is then drawn **isotropically and independently** (`draw_isotropic_sky`). The estimator's own
  completion term is **per-pixel** (`f_k` at the event's pixel). The correct joint law couples sky
  to `z` through `1 - f_k(pixel, z)`. Because pixel completeness and catalogue galaxy density are
  strongly positively correlated, `(1-f_k)` and the impostor rate are **negatively correlated**;
  drawing sky isotropically ignores that covariance and therefore **over-samples the directions
  where impostors are common**. ⇒ RHS2 too large ⇒ **deficit. Sign correct.**
- **(b) low-`z` end of `1-f̄`.** RHS2's mass sits where `1-f̄(z)` is smallest; a modest relative
  error in `f̄` near `z ≲ 0.15` rescales the tail rate directly, and hence RHS2, with almost no
  visible effect on the `z`-marginal as a whole (which is exactly the object the existing F10(c)
  consistency gate checks — a gate that by construction cannot see this).
- **(c) localization transplant.** `MirrorUniverseGenerator` **recenters** a donor Fisher row's
  sky covariance at the new position rather than rotating it (documented "registered design
  choice, flagged for review"). LISA sky-localization area is strongly ecliptic-latitude
  dependent, so the transplanted error volume is inconsistent with the new sky position — and the
  ball volume is precisely what sets the impostor rate that RHS2 lives on.

**Why 1-D closes anyway.** In 1-D the same defects exist but `E_Ḡ[w]` is not tail-dominated
(`L_cat_no_bh` is nonzero for most events), so the same relative law error produces a
correspondingly small statistic error. This candidate is the only one on my list that explains
"1-D in band, 2-D 2.9× off" **without** requiring any new 2-D-specific defect.

**Cheapest discriminator (zero `evaluate()`, all inputs already banked).**
1. **Leverage profile:** recompute RHS2 restricted to `d_L > d_cut` for a ladder of cuts, and plot
   the cumulative share; then the same for LHS2. If ≥50% of RHS2 sits in <1% of draws (already
   shown), publish that as a standing caveat on the instrument.
2. **Per-pixel reweight:** multiply each banked class-Ḡ event by
   `(1 - f_k(pixel_e, z_e)) / (1 - f̄(z_e))`, self-normalized, and recompute RHS2. `phiS`, `qS` are
   banked in every chunk's `prepared_cramer_rao_bounds.csv`; `z` is recoverable by the existing
   byte-identical replay (`_replay_completion_host_z`, already implemented). If RHS2 drops by
   ~2–3×, (a) is the mechanism.

**Confidence: HIGH for the leverage statement (measured); MEDIUM for (a) carrying the factor.**
Refuted if the per-pixel reweight moves RHS2 by <10% and the leverage profile shows the tail is
insensitive to the sky law.

---

### C3 — Class-Ḡ synthetic events carry the *donor row's own* mass observable *(model side, 2-D-specific)*

**Mechanism.** In `MirrorUniverseGenerator.draw_realization`, `host_mode="catalogue_selected_2d"`
performs the "monster event fix": the observed `(d̂, M̂_z)` is drawn from the donor's 2×2
covariance **centered at the host's latent `(d_L_true, M_z_true)`**, *replacing the donor row's own
mass*. `host_mode="population_selected"` — the class-Ḡ law used by RHS2 — does **not**: it
redraws `z` and the sky, but the mass columns stay at the donor's values. `ca_rhs_scorer.py`
states this explicitly as a scope limitation ("the completion class's OWN mass-law extension is
NOT attempted here"). So the two sides of a *with-BH* identity are generated under
qualitatively different mass constructions: the class-G side has a latent mass drawn from the
host's own prior and re-redshifted to the event's `z`; the class-Ḡ side has a mass copied from a
real, catalogue-hosted production injection at a *different* redshift, with no completion-class
mass prior anywhere in the chain.

**Why this symptom.** A donor mass is by construction a catalogue-typical mass, which makes
mass-coincidence with catalogue candidates more likely than for a genuine dark event drawn from
the population mass function ⇒ `L_cat_with_bh` non-zero more often ⇒ `w2` larger ⇒ **RHS2 too
large ⇒ deficit. Sign correct.** In 1-D the no-BH channel never reads the mass observable, so this
defect is exactly invisible there — it satisfies the "why does 1-D close" filter cleanly.

**On the card's two counterfactuals.** I read the card's ×0.05 / ×0.9997 result as *implicating*
this axis, not exonerating it:

- The needed correction, **0.345**, lies **strictly between** the two counterfactuals
  (0.05 < 0.345 < 0.9997). Two constructions that bracket the target do not clear the axis; they
  show the axis has enough dynamic range to contain it and that neither construction is the right
  one.
- A ×0.05 swing under an independent-host swap says the statistic is **20× sensitive** to the mass
  assignment. An axis with 20× leverage is the last axis one should retire on the grounds that two
  arbitrary alternatives missed the target.
- The ×0.9997 result (re-redshifting the donor's own mass, a ≲30% shift in `M̂_z`) is itself
  informative and slightly surprising: it implies `L_cat_with_bh` is *insensitive* to O(30%) mass
  shifts, i.e. the effective mass-matching width is the **catalogue** `σ_M` (60–200%), not the
  Fisher `σ_M` (p50 ~1e-7 quoted in `p3_2d_companion.py`). That is worth stating as a standing
  fact about the instrument, and it means the right counterfactual is not a mass *shift* but a
  mass *law* replacement.

**Cheapest discriminator.** Two-sample comparison of the mass-observable marginal
(`M̂_z`, and `σ_M/M̂_z`) between the banked b0i2d accepted events and the banked class-Ḡ events —
i.e. **the mass-marginal analogue of the F10(c) `z`-marginal gate that was never built**. Then:
re-score one RHS2 chunk with `M̂_z` replaced by a draw from the completion-class mass law
(`φ(M)·S_4D`-weighted, redshifted to the event's own `z`). Both are cheap; the second needs one
chunk's `evaluate()` (~minutes).

**Confidence: MEDIUM-HIGH.** Refuted if the mass-observable marginals already agree, or if the
law-replacement re-score moves RHS2 by <20%.

---

### C4 — `C2*` assembly premise: the α↔β / `Σ^φ`↔`Σ^4D` pairing *(the ×2.6125 numerology)*

**Mechanism.** The card certifies the constant's *arithmetic*, not its *premise*. The observed
deficit **2.5–2.9 straddles `1/r_Malm = Σ^φ/Σ^4D = 2.6124925`** — a number the codebase itself
hardcodes as `R2_REGISTERED` in `ca_rhs_scorer.py`, described there as guarding "the α↔β mix-up
class, the most likely 2D scorer defect". Concretely: the banked
`C2* = β_G^φ Σ̃^4D/(Σ^φ β_Ḡ^φ) = 0.06124`; the alternative assembly
`β_G^φ Σ̃^4D/(Σ^4D β_Ḡ^φ) = 0.15999` differs by exactly `1/r_Malm`, and
`C2*·2.9 = 0.1775`, `C2*·2.73 = 0.1673` — both within ~11% and ~4% of that alternative. If the
remaining 4–11% is the (independently measured, sign-correct) selection tilt, the two effects
compose to close the identity: `2.6125 × 1.135–1.16 = 2.96–3.03` against a needed 2.73–2.90.

**My own code check partially exonerates the banked assembly**, and I report it as such rather
than suppress it: `bayesian_statistics.py:5022` takes `global_denom_with_bh` from
`_global_cat_denom_with_bh`, which `:4077` identifies as **`Σ^4D`**, and
`path_a_mixture_objects` sets `α_G^φ = β_G^φ·Σ^4D/Σ^φ`. Hence
`α_G^φ·L_cat_with_bh` carries `β_G^φ/Σ^φ × (local with-BH numerator)`, the `Σ^4D` cancels, and
`M_G = β_G^φ Σ̃^4D/Σ^φ` — i.e. the banked `C2*` is the self-consistent one. **Two caveats keep
this candidate alive:** (i) the identity `∫L_cat_with_bh dd = Σ̃^4D/Σ^4D` holds only for the
**twin** arrangement (`catalogue_numerator_survival_2d="mz_sel"`, which inserts `S_4D` in the
catalogue numerator); the **coded** arm (`"off"`) integrates to `Σ_g w_g/Σ^4D`, a different
constant — so a single `C2*` cannot be correct for both arms, and "arm-independence" of the
deficit is therefore a *finding to explain*, not a reassurance; (ii) the coincidence of the
measured ratio with the project's own deliberately-wrong control constant is too sharp to leave
unremarked.

**Cheapest discriminator.** State, in one line, whether the measured 0.345/0.366 is or is not
consistent with the registered B2-R2 control's predicted value `1/2.6124925 = 0.38278` — the
transform arm at 0.366 ± 0.014 is **1.2σ** from it and the primary at 0.345 ± 0.013 is 2.9σ. If
the twin arm is sitting at the control's own predicted failure value, that is a
mechanism-bearing fact, not noise. Free.

**Confidence: LOW-MEDIUM that the constant is wrong; HIGH that the coincidence must be
explicitly dispositioned.** Refuted by the code trace above plus an explicit derivation of
`∫L_cat_with_bh dd` for each arrangement.

---

### C5 — Degenerate rows: `B_num_wbh == 0` pins `w2 = 1`, and `0/0` rows are dropped

**Mechanism / measurement.** In the pilot b0i2d twin seed (900101, 84 scored rows of 200 drawn):
**17 rows have `B_num_wbh == 0` exactly** (⇒ `w2 = 1`, `(1-w2) = 0`), 7 have
`L_cat_with_bh == 0`, and **1 row is `0/0` → NaN**. If the aggregation `nansum`s, that NaN row
contributes 0, whereas the class's own adjudicated convention for the 1-D member (dead rows are
bounded, `w=0`, contribute `1-w=1`) would give it 1. Each such row is a full unit of
`Σ(1-w2)` — against a seed total of `Σ(1-w2) = 23.7`, four mis-handled dead rows is a 17% effect.

**Why this symptom.** Sign correct on the LHS (drops contributions), and the *same* underflow on
the class-Ḡ side has the opposite, much larger leverage: any Ḡ event with `B_num_wbh` underflowing
to 0 is pinned at `w2 = 1` and, given `RHS2 ≈ 0.015`, **1.5% of such rows would account for the
entire RHS2**. In the 60 chunks I checked, `B_num_wbh == 0` never occurred on the Ḡ side and only
1 row had `w2 == 1`, so this is a small effect *there* — but it is exactly the kind of thing a
tail-dominated mean cannot tolerate, and it must be reported, not assumed.

**Cheapest discriminator.** Count, per side, rows with `B_num_wbh == 0`, `L_cat_with_bh == 0`,
and NaN `w2`; recompute both sides under each of the two dead-row conventions. Free (banked CSVs).

**Confidence: MEDIUM that it contributes 10–20%; LOW that it carries the factor.** Refuted by the
count itself.

---

### C6 — Mass-error transplant: donor `σ_M` applied to a replaced mass value *(LHS side, 2-D-only)*

**Mechanism.** On the class-G side the mass *value* is replaced by the host's latent
`M_z_true`, but the 2×2 covariance block used to scatter it is the **donor's own** — a Fisher
error computed at the donor's mass and distance, which may be dex away. The event's
`σ_M/M̂_z` is therefore not the error a real observation of that source would have. Both
`L_cat_with_bh` and `B_num_wbh` consume that width; the class-Ḡ side has no such transplant
(mass and error are both the donor's, mutually consistent). Asymmetric by construction.

**Why this symptom.** Sign not determinable a priori — a too-narrow width sharpens
mass-matching (raises `L_cat_with_bh`, raises `w2`, lowers LHS2, correct sign); a too-broad width
does the reverse. Magnitude is potentially O(1) because the transplant spans the catalogue's
mass range.

**Cheapest discriminator.** Compare the distribution of `σ_M/M̂_z` on the two sides; and regress
`(1-w2)` against `σ_M/M̂_z` on the class-G side. Free.

**Confidence: MEDIUM-LOW.** Refuted if `σ_M/M̂_z` distributions overlap and the regression is flat.

---

### C7 — Latent-mass clipping at `_M2D_MASS_FLOOR = 1 M_☉` vs the analytic `S(M≤0) := 0`

**Mechanism.** `_draw_2d_accepted_latents` draws `M ~ N(M_eff, σ_M)` and **clips** to 1 M_☉
(`correspondence_1d.py:1549, 1708`). The analytic contraction instead applies the guard
`S_4D(M≤0) := 0` (`p3_2d_companion.py`, `_mass_marginal_survival`). The banked companion reports
`sum_P_M_le_0 = 2.06e6` over `n_eligible = 20.83e6` — i.e. **~10% of the total Gaussian mass sits
at `M ≤ 0`** and is treated differently by the two sides. If the `S_4D` interpolator clamps below
its lowest mass node rather than returning 0, the venue accepts events the model assigns zero
weight.

**Why this symptom.** Wrong sign for the observed deficit: clipped events are anomalously
light, match catalogue hosts poorly, give small `w2` and large `(1-w2)` ⇒ LHS2 **too big**, not
too small. Magnitude bounded at ~10%.

**Cheapest discriminator.** Evaluate `S_4D(d_L, M = 1 M_☉)` at a few grid distances; if it is
numerically 0, C7 is closed. Seconds. Also count how many banked `M_true` sit at exactly 1.0.

**Confidence: LOW as a cause; HIGH that it should be recorded as a known convention mismatch.**

---

### C8 — Candidate-ball and `mass_filter_sigma` truncation vs the untruncated `Σ̃^4D`

**Mechanism.** `Σ̃^4D` sums over the **whole** eligible catalogue (20.83M rows). The per-event
`L_cat_with_bh` sums only over the BallTree candidate set intersected with the mass-filter window
— a truncation that has no counterpart in the contraction, so `∫L_cat_with_bh dd < Σ̃^4D/Σ^4D`.
The mass window is *new in 2-D*.

**Why this symptom.** The true `M_G` is smaller than `C2*` assumes ⇒ the correct `C2*` is smaller
⇒ LHS2 should be **smaller** still. **Wrong sign** — this makes the discrepancy worse, so it
cannot be the cause, but it *bounds* how much of the residual any other mechanism must supply.

**Cheapest discriminator.** For a sample of events, compute the fraction of `Σ̃^4D` captured by
the event's own candidate set. Cheap; also directly useful as a caveat.

**Confidence: HIGH that the effect exists; HIGH that it has the wrong sign.**

---

### C9 — Selection applied at true parameters *and* again at the data level

**Mechanism.** The venue thins by `Bernoulli(S_4D(θ_true))` and then applies the F-0 data cut
(`σ_dL/d̂ < 0.10` and `SNR ≥ 20`). Mandel, Farr & Gair (2019) are explicit that detection is "a
property purely of the data" and that selection should be marginalized over data realizations
rather than applied to true parameters. The identity survives a *common* `1_acc(d)` on both sides,
so this is formally safe — **but** the two sides' acceptance rates differ hugely (42% vs ~90%),
which means the acceptance operator is doing very different work on the two classes, and any
residual dependence of `1_acc` on the donor's identity (its `σ_dL`, `SNR`, transplanted rather
than recomputed at the new placement) breaks the "same function of the data" premise.

**Why this symptom.** Sign indeterminate; magnitude potentially O(1) because it re-weights the
low-`d_L` region that C2 shows owns RHS2.

**Cheapest discriminator.** Verify that `1_acc` is computable from the event row alone with the
identical formula on both sides (it is, by code) *and* that the donor-row draw is statistically
identical on both sides (SNR-weighted, same pool, same without-replacement scheme). Then check
whether `1_acc` is independent of class given `(d̂, SNR, σ_dL)`. Free by inspection.

**Confidence: LOW-MEDIUM.** Refuted by the inspection above.

---

### C10 — Draw-side quadrature never arbitered against the same kink structure the model side needed

**Mechanism.** The model side discovered (and fixed) that `S̄_4D(d_L(z))` is piecewise-linear with
kinks at 60 `dl_centers` grid edges, and that a flat GL-50 rule under-resolved them by 3.8e-4;
it now uses segment-aware GL-16 with breakpoints. The **draw** side builds each host's `z_true`
density on a **401-node uniform grid** over the ±4σ window (`_B0I_ZTRUE_GRID_N = 401`) and
inverse-CDF-samples via a trapezoid CDF with linear interpolation — the same kink structure, never
arbitered. The class-Ḡ `z` draw uses 4001 uniform nodes over `[1e-6, HOST_DRAW_Z_MAX]`.

**Why this symptom.** Sign indeterminate; the model-side evidence suggests the raw magnitude is
~1e-4, far too small — **unless** it lands in the tail that owns RHS2 (C2), where a 1e-3-level
probability misplacement is a percent-level change in the impostor rate.

**Cheapest discriminator.** Redraw one seed with `n_grid = 4010` / `40010` and compare the
realized `z_true` empirical CDF and the resulting LHS2. Cheap (no `evaluate()` for the CDF part).

**Confidence: LOW.** Refuted if the refined-grid draw reproduces the same `z_true` distribution.

---

### C11 — Host-pool eligibility set vs the contraction's eligibility mask

**Mechanism.** `Σ̃^4D` sums rows satisfying `z < z_max(h) & isfinite(M) & M > 0`
(`n_eligible = 20,834,132`, banked). The venue's `HostPool` comes from `_host_pool_from_handler`
and may apply a different (or no) mask; hosts outside the contraction's eligible set would be
drawable but carry no model-side mass, or vice versa. Any mismatch is a pure O(1) normalization
slip of exactly the kind rung 4 of the card's ladder describes.

**Cheapest discriminator.** Print `pool.n` and the count of `pool` rows satisfying the
contraction's mask; assert equality. Seconds (one catalogue load). **This should be a permanent
assertion in the harness, not a one-off check.**

**Confidence: LOW-MEDIUM** (nothing in the code told me they differ; nothing told me they agree
either — no assertion exists). Refuted by the count.

---

### C12 — Hosts drawn **with** replacement in 2-D, **without** in 1-D

**Mechanism.** `_draw_2d_accepted_latents` uses `replace=True` over a 2.3e7-row pool
(documented, "negligible-probability simplification"); the 1-D `catalogue_selected` branch uses
`replace=False`. For n=200 from 2e7 this is a ~1e-6 effect.

**Confidence: negligible.** Listed for completeness; refuted by the birthday bound.

---

## 3. Literature sweep

I searched for published treatment of (i) consistency identities between an accepted mock sample
and a model-side expectation in two-class catalogue/completion mixtures, (ii) which normalization
each side must carry, (iii) whole-event-rejection samplers realizing a different joint law than
the per-draw target.

**What the field does say.**

- **Mandel, Farr & Gair (2019), arXiv:1809.02063** — the canonical statement that the selection
  normalization `α(λ)` is "the fraction of events in the Universe that would be detected for a
  particular population model" and must be computed under the *same* population and detection
  model as every numerator; that detection is "a property purely of the data"; and that "any
  incorrect analysis … will lead to a bias in the result", with the crucial asymptotic warning
  that "the bias remains constant while the uncertainty decreases like the square root of the
  number of events". (Verified by fetching the paper's HTML; quotes above are from that fetch.)
  This is directly on point for C9 (parameter-level vs data-level selection) and for the general
  frame — a stable O(1) offset with shrinking SEM is exactly the signature the card describes.
- **Gray et al. (2020), arXiv:1908.06050** and **Gray et al. (2022), MNRAS 512, 1127
  (arXiv:2111.04629)** — the in-catalogue/out-of-catalogue partition and its pixelated
  completeness treatment. The 2022 paper's central methodological point is precisely that
  *uniform completeness within a GW sky area is not adequate* and that completeness must be
  computed line-of-sight by line-of-sight. That is a published warning aimed squarely at C2(a):
  our class-Ḡ generator uses a **sky-averaged** `1-f̄(z)` with an **isotropic** sky, i.e. the
  approximation this literature exists to reject — applied on the *generator* side rather than the
  estimator side, where nobody has audited it.
- **gwcosmo validation (arXiv:2605.23538)** — validates by "recovering the injected
  hyperparameters of a simulated population of 2000 GW events … to demonstrate the
  self-consistency of our method", plus GPU/CPU KL comparisons. Confirmed by fetch.
- **Essick & Farr (2022) / Talbot & Golomb (arXiv:2110.13091, 2204.00461)** and follow-ups
  (arXiv:2502.12156, arXiv:2606.14229) — precision requirements for Monte-Carlo sums inside
  hierarchical inference, and the finding that the selection-function variance contribution
  *grows quadratically with population size*. This literature is the closest published relative of
  **F-A/C2** (my tail-domination finding): the field knows that MC estimates of selection-type
  integrals are effective-sample-size limited and tail-sensitive, and prescribes ESS/variance
  thresholds. **The card's "SEMs are 2–3%, so this is not a variance problem" is exactly the
  reasoning this literature warns against** — a bounded functional (`w ∈ [0,1]`) can have a small,
  honest SEM while being dominated by a 1% tail, and a small SEM says nothing about whether the
  *tail's law* is right. Report the effective sample size, not the SEM.
- **arXiv:2302.10621 ("The Dark Side of Using Dark Sirens…")** — the hazard that catalogue
  galaxies at the inferred distance/sky position are physically uncorrelated with the true source,
  producing large cosmological bias. This is the published name for the impostor population that
  I find carries RHS2.

**Absences (findings in their own right).**

1. **No published treatment of a class-odds consistency identity of the form
   `C*·E_G[1-w] = E_Ḡ[w]` used as a validation gate for a dark-siren catalogue/completion
   mixture.** The identity is elementary and correct, but the field validates by
   parameter recovery (P–P plots, injected-hyperparameter recovery), not by class-mass
   identities. There is therefore **no literature on which normalization (drawn-count,
   accepted-count, class-count) each side must carry** — the card's central "Sought" question is
   unanswered in the published record. This project is ahead of the literature here, which also
   means there is no external error catalogue to consult.
2. **The gwcosmo validation paper does not validate the normalization *between* the in-catalogue
   and out-of-catalogue mixture terms as a separate closure test** (confirmed by fetch). The
   mixture-normalization closure that [P3-2D] is attempting has, as far as I can find, no
   published precedent.
3. **No literature found on restart/whole-event-rejection samplers realizing a different joint law
   than the per-draw target in a mock-catalogue context.** Three searches returned only generic
   rejection-sampling and diffusion-restart material. The statistical fact is textbook (joint
   rejection sampling of the whole tuple is unbiased for the joint target — and I verified that
   `_draw_2d_accepted_latents` does redraw the whole tuple, so **no restart bias exists here**);
   the *applied* failure mode the card asks about is not a documented one. The real defect in this
   sampler is not restart bias but the residual `S̄_φ(z)` factor (C1), which is a
   proposal/target mismatch, not a rejection artifact.
4. **No literature on tail-domination of class-membership expectations in dark-siren mixtures.**
   The nearest analogue is the MC-precision literature above, which addresses selection integrals,
   not class-weight expectations.

**Sources:**
- [Mandel, Farr & Gair (2019), arXiv:1809.02063](https://arxiv.org/html/1809.02063v2)
- [Gray et al. (2022), pixelated catalogue incompleteness, MNRAS](https://doi.org/10.1093/mnras/stac366) · [arXiv:2111.04629](https://arxiv.org/pdf/2111.04629)
- [Scalable Dark Siren Cosmology with gwcosmo (arXiv:2605.23538)](https://arxiv.org/html/2605.23538)
- [Precision Requirements for Monte Carlo Sums within Hierarchical Bayesian Inference (arXiv:2204.00461)](https://arxiv.org/pdf/2204.00461)
- [Growing pains: likelihood uncertainty in hierarchical Bayesian inference, MNRAS 526, 3495](https://academic.oup.com/mnras/article/526/3/3495/7285822)
- [Sampling the full hierarchical population posterior (arXiv:2502.12156)](https://arxiv.org/pdf/2502.12156)
- [The Dark Side of Using Dark Sirens (arXiv:2302.10621)](https://arxiv.org/html/2302.10621)
- [Implementing a Robust Test of Galaxy Catalogue Completeness (arXiv:2502.14164)](https://arxiv.org/pdf/2502.14164)

---

## 4. Ranking

Score = P(cause) × cheapness. All top entries are zero- or near-zero-`evaluate()`.

| # | Candidate | P(cause) | Cost | Sign OK? | Explains "1-D closes"? |
|---|---|---|---|---|---|
| 1 | **C1** residual `S̄_φ(z)` in the 2-D draw law | high | free (banked `z_true`) | ✓ | ✓ (2-D-only stage) |
| 2 | **C2** RHS2 tail-domination amplifying small class-Ḡ law defects (esp. isotropic sky × sky-averaged `f̄`) | high (frame) / medium (specific) | free | ✓ | ✓ (leverage, not a new bug) |
| 3 | **C3** class-Ḡ mass observable is the donor's, no completion mass law | medium-high | free → 1 chunk | ✓ | ✓ (no-BH never reads mass) |
| 4 | **C4** `C2*` assembly premise / the ×2.6125 coincidence | low-medium | free | ✓ | ✓ |
| 5 | **C5** dead/degenerate row convention (`B_num_wbh==0`, `0/0`) | medium (partial) | free | ✓ | ✓ |
| 6 | **C6** donor `σ_M` transplanted onto a replaced mass | medium-low | free | ? | ✓ |
| 7 | **C11** pool eligibility vs contraction mask | low-medium | seconds | ? | ✓ |
| 8 | **C8** ball/mass-window truncation vs untruncated `Σ̃^4D` | high (exists) | cheap | ✗ (wrong sign) | ✓ |
| 9 | **C7** mass clip at 1 M_☉ vs `S(M≤0):=0` | low | seconds | ✗ | ✓ |
| 10 | **C9** parameter-level + data-level double selection | low-medium | free | ? | ✗ |
| 11 | **C10** draw-side quadrature never arbitered | low | cheap | ? | partial |
| 12 | **C12** with- vs without-replacement host draw | negligible | free | – | – |

### The single cheapest decisive discriminator

> **Importance-reweight the banked b0i2d latents by `1/S̄_φ(z_true)` and recompute LHS2.**
>
> `LHS2_corrected = C2* · Σ_accepted (1-w2_e)/S̄_φ(z_e) ÷ Σ_all-200 1/S̄_φ(z_e)`
>
> Every input is already on disk: `z_true` is banked for **all 200** drawn latents in each seed's
> `prepared_cramer_rao_bounds.csv` (verified), `w2_e` from `event_likelihoods.csv`, `S̄_φ` from the
> cached `phi_survival_table`. No `evaluate()` call, no cluster job, one pandas pass per seed.
>
> This is decisive because C1 is not a hypothesis about the physics — it is an **exact algebraic
> identity between the sampler and the contraction**: the realized law is the target law times
> `S̄_φ(z)`, with every per-host normalizer cancelling. The reweight is therefore the *correct*
> statistic, not an approximation to it. If it lands in band, the thread closes. If it moves LHS2
> by ≲20%, C1 is quantitatively refuted and the search moves to the model side with the leverage
> frame (C2) already established.

**Mandatory companion (also free), because it changes how every subsequent number is read:**
publish the **effective sample size / tail-share profile of RHS2** (§1.2 F-A/F-B) and the per-pixel
completeness reweight `(1-f_k)/(1-f̄)`. The instrument's model side is a 1%-tail functional; that
fact belongs on the symptom card and in every future band.

---

## 5. Two methodological cautions for whoever picks this up

1. **"Small SEM ⇒ not a variance problem" is a non-sequitur for a bounded, tail-dominated
   functional.** `w2 ∈ [0,1]`, so the CLT applies and the 2–3% SEM is honest — and completely
   uninformative about whether the 1% of draws that carry 56% of the mean were generated under the
   right law. Report ESS, not SEM.
2. **Bracketing counterfactuals implicate an axis; they do not exonerate it.** The card's two mass
   counterfactuals returned ×0.05 and ×0.9997 and the needed factor is ×0.345 — strictly between
   them. The correct reading is "this axis has ≥20× leverage and neither construction was the
   right one", not "the axis is cleared."

---

## 6. Feedback on the symptom card — what was ambiguous or under-specified

The card is unusually disciplined about *not* leaking hypotheses, and the abstraction ladder is
genuinely useful for literature vocabulary. The costs of that discipline:

1. **The statistic is never defined.** Not one formula appears. A fresh searcher cannot tell
   whether the "bounded-identity comparison" is a ratio of means, a paired difference, a log-odds,
   or a moment. I had to reconstruct `C*·E_G[1-w] = E_Ḡ[w]` from source. Giving the identity in
   one line would have leaked no hypothesis whatsoever and saved a large fraction of my budget.
2. **Direction is not stated.** "banked/model = 0.345" leaves the reader to infer which side is
   the numerator and which side is "wrong". Say "the empirical side is 2.9× *smaller*".
3. **No absolute values.** Only the ratio is given. `LHS2 ≈ 0.0073`, `RHS2 ≈ 0.0151` are decisive
   context: they tell you immediately that both sides are small numbers built from a heavy tail.
   Withholding them hid the single most important structural property of the instrument.
4. **"Arm-independent (two estimator arrangements give 0.345/0.366 alike)" contradicts the earlier
   clause** that 0.366 is "a registered nonlinear rescale transform of the same objects". Those are
   different things — two *arms* vs one arm under two *transforms*. As written I could not tell
   which, and it matters (the twin and coded arms do **not** share the same class-odds constant,
   §C4).
5. **"~24 accepted-event realizations"** reads as 24 events. It is 24 *seeds* × 200 drawn latents
   each, of which ~40% survive F-0. The drawn/accepted distinction is load-bearing for the
   normalization question the card itself poses in "Sought".
6. **"~5× the pre-registered band" without the band.** Unfalsifiable as posed.
7. **The "13.5–16% tilt" is reported without saying what functional it is a tilt *of*.** Because
   the same mechanism can produce a 15% distributional tilt and a 250% statistic tilt (§C1′), the
   "~7× too small" verdict cannot be evaluated by a reader. Also: "the fix is authorized but
   unrun" without saying *what the fix is* means a searcher cannot tell whether their leading
   candidate is already the authorized fix — I still do not know whether C1 is that fix or a
   different one.
8. **The two mass counterfactuals are reported without their constructions.** "Independent-host
   swap" and "own-mass re-redshifting" are named, not defined; ×0.05 vs ×0.9997 is a 20,000×
   spread between two things called by similar names, which is impossible to interpret cold.
9. **No statement of which side each control constrained.** "Per-event mass-observable linkage on
   the model side is verified" — but the card elsewhere uses "model" for the 25,600-draw side,
   while the mass *linkage* (the "monster event fix") lives in the *venue* generator. I could not
   resolve this from the card alone.
10. **The contraction's arbitration is quoted as "1e-9"**; the banked artifact records a derived
    target of 2.916e-9 against a max deviation of 8.05e-10 — fine, but the card's round number
    hides that the original 1e-6 target was replaced by an arbiter-grounded one, which is itself a
    caution worth carrying (the first target was, in the instrument's own words, "unfalsifiable as
    posed").
11. **Nothing about acceptance rates.** 42% (class-G) vs ~90% (class-Ḡ) is a large asymmetry in
    the shared acceptance operator, and the identity's validity rests on `1_acc` being the same
    function of the data on both sides. A card for an acceptance-based identity should state them.

**Suggested minimal additions that leak no hypothesis:** the identity in symbols; both sides'
absolute values with n; the drawn vs accepted counts per seed; the acceptance rates per class; the
pre-registered band; the functional under which the 13.5–16% was measured; and a one-line
statement of the tail-share/ESS of each side.
