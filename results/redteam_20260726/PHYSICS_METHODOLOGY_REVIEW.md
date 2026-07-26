# Adversarial physics + statistical-methodology review — production stack `generator_marginal + --pdet_z_resolved`

**Date:** 2026-07-26 · **Reviewer role:** independent redteam (physics soundness + anti-tuning audit)
**Scope ordered by the author:** entire evaluation setup, and specifically whether the 2026-07-26
bias closure (4-seed, MAP = 0.7300 ± 0.0004 at truth h = 0.73) is the product of unscientific
tuning or forking-paths selection.
**Branch reviewed:** `physics/absolute-mass-marginal` @ `c5047ff`.
**Inputs read:** `OVERNIGHT_REPORT_20260726.md`, `DERIVATION_GENERATOR_CONSISTENT_NORM.md`,
`DERIVATION_ZRESOLVED_SURVIVAL.md`, `MULTISEED_READOUT_20260726.md`,
`SEED600_GATE_REGISTRATION.md`, `RUNBOOK_NEXT_SESSION.md`, `docs/H0_BIAS_RESOLUTION.md`
§3.14–§3.22, `DATA_INVENTORY.md`, `master_thesis_code/bayesian_inference/*.py`,
`master_thesis_code/galaxy_catalogue/{handler,pixel_completeness}.py`,
`master_thesis_code/validation/pp_coverage.py`.
**New measurements made by this review** (all from committed artifacts; no pipeline re-run):
per-event decomposition of the seed1000 dense-core posterior
(`results/lcat_h_dependence_20260725/densecore_probe/simulations/diagnostics/event_likelihoods.csv`,
3454 events × 13 h-nodes) cross-matched against
`results/campaign_phase2_runs/run_20260703_seed1000/simulations/prepared_cramer_rao_bounds.csv`.

---

## 0. Verdict

**SOUND-WITH-CAVEATS — with one interpretation claim that must be retracted or rescoped before
it appears in the paper.**

Two separable judgements:

1. **Anti-tuning audit: PASS, with one identified forking path (F-1 below).** I found no
   evidence of truth-tuning. There are no free constants in FIX-2/FIX-3; every calibration
   quantity is a deterministic functional of the catalogue, the completeness cache, or the
   injection pool; the pre-registration discipline is real, is time-stamped ahead of the
   readouts, and — decisively — *records its own failures* (V1 failing the seed600 gate, FIX-3's
   own quantitative prediction being falsified, criteria 3/4 of the seed600 gate failing, the
   registered 5-seed test returning QUALIFIED FAIL). A pipeline that was being tuned to truth
   would not have left that trail. I additionally ran an independent calibration test the
   project has not run: the per-event pulls of the events that actually carry the measurement
   are `mean = +0.06, std = 0.94` over 133 events (expected `N(0,1)`) — the estimator is
   internally unbiased **and correctly calibrated** on the informative subset. That is a
   genuinely strong result.

2. **Physical interpretation: UNSOUND AS STATED.** The claim "no detectable bias at the
   4×10⁻⁴ level" and the associated `σ_h ≈ 3×10⁻⁴` are attached to the wrong physical object.
   The seed1000 measurement is **not** a 3454-event dark-siren measurement. It is, to 100 %
   of the posterior curvature, a **≈130-event bright-siren measurement**: ~3.9 % of the events
   have their true host uniquely and exactly identified because the mock's catalogue redshifts
   carry *zero* scatter, and the production estimator was changed (`generator_marginal`,
   point/point σ_z pairing) to exploit that exactness. The remaining 96.1 % of events —
   i.e. the entire dark-siren / completeness-correction machinery the thesis and paper are
   about — contribute **net negative** curvature at truth and would not close on their own.
   The closure therefore validates the *host-association* leg of the estimator and the *code's
   consistency with its own generator*; it does **not** validate the selection-, completeness-,
   or completion-modelling, and it does not forecast LISA's dark-siren H₀ capability.

`recommendation_ceiling`: **major revision** for any manuscript text that presents the 4-seed
closure as validation of the dark-siren estimator or quotes σ_h ≈ 3×10⁻⁴ as a LISA forecast.

---

## 1. The decisive measurement (new, made by this review)

From the 13-point dense-core diagnostics (`h ∈ [0.700, 0.760]`, step 0.005, seed1000, no-BH
channel; `combined_no_bh` is the per-event marginal `p_i(h)`), define per event
`slope_i = d ln p_i/dh` and `curv_i = −d² ln p_i/dh²` at h = 0.730 by 3-point finite
differences. Reproduces the shipped readout exactly: `Σ curv = 1.427×10⁷ → σ = 2.647×10⁻⁴`,
`MAP = 0.73 + Σslope/Σcurv = 0.730387` (shipped: 0.7304, σ 0.00026).

| Subset | n | Σ slope | Σ curvature | share of curvature | MAP if used alone |
|---|---|---|---|---|---|
| all events | 3454 | +5521 | 1.427×10⁷ | 100 % | **0.73039** |
| "golden" (`curv_i > 10⁴`) | **133** | +6004 | 1.433×10⁷ | **100.4 %** | **0.73042** |
| all others | 3321 | −483 | **−5.5×10⁴** (negative) | −0.4 % | undefined (local *minimum*) |
| pure-completion fallback (`A_i = 0`) | 1996 | −591 | +6.1×10³ | 0.04 % | 0.633 |

- The **top 100 events carry 96.6 % of the total curvature**; the top 20 carry 56.6 %; the
  single largest event carries 8.6 %.
- Peak height above the plateau (ln p at 0.730 minus the mean at 0.700/0.760) totals 883 ln,
  of which the top 100 events supply 92.5 %; only 131 events supply more than 1 ln each.
- **Identification of the golden events as exact host identifications.** Cross-matching
  `event_idx` to the prepared CRB rows, each golden event's measured per-event width equals
  `0.73 × σ_dL/d_L` to a few percent (`corr(log σ_measured, log 0.73·σ_dL/d_L) = 0.95`; e.g.
  measured 0.00090 / predicted 0.00087, 0.00127/0.00127, 0.00143/0.00143). Since
  `d_L(z;h) ∝ 1/h` at fixed z and fixed Ω_m, `σ_h/h = σ_dL/d_L` is *exactly* the
  zero-host-redshift-error (bright-siren) limit. The golden events are therefore delta-function
  host identifications, no more and no less.
- **Their per-event pulls are clean:** `(MAP_i − 0.73)/σ_i` over the 133 golden events has
  mean +0.062, std 0.944, one |pull| > 3, none > 5. Inverse-variance-weighted mean pull +0.16.
  This is the strongest single piece of evidence that the closure is not an artifact and not
  tuned — and it is a test the project should adopt.
- **Where they live:** golden events are at `z ∈ [0.011, 0.166]`, median **z = 0.056**,
  `d_L ∈ [0.046, 0.766] Gpc`, median relative distance error 0.54 %. The venue's median event
  is at `d_L = 2.23 Gpc` (z ≈ 0.4). **The "deep venue" result is carried entirely by its
  shallow tail.**

Everything below follows from this decomposition.

---

## 2. Severity-ranked findings

### P-1 [CRITICAL — interpretation] The headline claim attributes a bright-siren result to a dark-siren estimator

`MULTISEED_READOUT_20260726.md` §Disposition: *"On the 4 provenance-valid venues the production
stack shows no detectable bias at the 4×10⁻⁴ level"*. Scoped as written, this is a statement
about the whole 3454-event dark-siren analysis. Physically it is a statement about ~130 events
whose likelihood is `p(d_L^obs | d_L(z_g;h))` with `z_g` known exactly, times a shared
`1/D_gen(h)^N` factor. The 96 % of events that exercise the completeness correction, the
completion integral `B_num`, the `β_Ḡ` selection integral and the catalogue-vs-dark mixture
contribute **net negative curvature** at truth.

Consequence for the physics claim: the bias test has **no statistical power against systematics
in the dark-siren machinery**. Quantitatively — a residual shared per-event tilt `t` (in
ln/h/event) displaces the MAP by `N·t/Σcurv = 3454·t/1.427×10⁷ = 2.4×10⁻⁴·t`. To move the MAP
by the 4×10⁻⁴ detection threshold requires `t ≈ 1.7/h per event`. For reference, the shared
normalization tilt itself is `−d ln D_gen/dh ≈ +1.45/h`, and the pre-fix estimators railed on
tilts of this same order. **The golden events do not remove the systematic; they out-vote it.**
The estimator's ability to absorb a ±1.5/h shared tilt with a ≤3×10⁻⁴ MAP shift is exactly why
FIX-3 and FIX-2 both "closed" despite FIX-3's own derivation predicting a persistent rail.

Required action: rescope every closure statement to "the production stack recovers h to
3×10⁻⁴ on this mock; the recovery is dominated by ~4 % of events with exactly-known host
redshifts, and therefore constitutes a validation of the host-association and normalization
*assembly*, not of the completeness/selection modelling."

### P-2 [CRITICAL — regime of validity] The point/point σ_z pairing silently deletes physically motivated host-redshift uncertainty, and it is the sole carrier of the result

`bayesian_statistics.py:3067–3079` (`_use_generator_point`) point-evaluates `N_g` at the
catalogue `z_g`, dropping (a) the catalogue redshift measurement error and (b) the peculiar-
velocity dispersion `σ_z,pv = (1+z)·σ_v/c` with `SIGMA_V_PEC_KM_S = 200` (`constants.py:83`).
The latter was itself a *physics-approved* term (issue #16, author decision 2026-07-03, Davis
et al. 2011; Mastrogiovanni et al. 2023; Laghi et al. 2021). It is now switched off on the
production path.

The justification given — the mock generator applies no σ_z, so point/point is generator-exact
— is *correct for this mock* and is honestly flagged in three places
(`RUNBOOK_NEXT_SESSION.md` "Open ends", §3.22 "Standing caveats", `DERIVATION_..._NORM.md`
§4.3). The quarantine is real. **But its consequence is under-stated by orders of magnitude.**
Measured on the events that carry the result (median z = 0.056):

| neglected host-z error term | median size, relative to z | vs. retained σ_dL/d_L (0.54 %) |
|---|---|---|
| peculiar velocity, 200 km/s | **1.25 %** | 2.3× larger |
| GLADE+ z-error floor used by the parser (0.0015) | **2.67 %** | 4.9× larger |

Restoring them (still in the optimistic "host known" limit) degrades the combined width over
the golden set from `1.9×10⁻⁴` to `6.3×10⁻⁴` (PV only, ×3.3) or `1.3×10⁻³` (PV + z-floor,
×6.8). And that is a floor on the degradation: with `σ_z ≈ 0.0015–0.003`, the ±4σ GW window at
z ≈ 0.05 admits many catalogue galaxies, so the golden events *cease to be golden* and the
analysis reverts to a genuine dark-siren problem with impostor-dominated balls — the exact
regime in which the estimator previously railed.

Required action: the paper must not quote σ_h ≈ 3×10⁻⁴ (or `H₀` to 0.04 %) as a LISA/EMRI
dark-siren forecast. It is a mock-internal recovery figure. A realism ablation (§4, T-3) is
needed before any precision statement.

### P-3 [HIGH — statistics] The quoted width is not resolved by any grid that was ever evaluated

`σ = 2.6×10⁻⁴` is obtained by fitting a parabola through the peak node and its two neighbours
at spacing `Δh = 0.005` — i.e. through points **19σ away from the claimed peak**, where the
log-posterior is already −206 and −151. The 41-point production grid has spacing 0.0065 = 25σ.
The posterior has never been evaluated anywhere inside ±5σ of its own claimed peak. The
measured curve is a sharp cusp riding on a broad plateau (−877 at h = 0.700, −888 at 0.760,
0 at 0.730), not a parabola; a 3-point curvature on such a shape has no guaranteed relation
to the true width, and the sub-grid MAP offset "+0.0004" is likewise an extrapolation
artifact of order the interpolation error.

This also explains the empirical tension the readout flags: seed scatter (7.1×10⁻⁴) is 2.7×
the curvature width, and the base-channel width χ² = 8.0 on 3 dof is **p = 0.046** — reported
as "VALID (marginal)" when it is, at face value, a marginal *rejection*. (Restricting to the
three informative deep venues, χ² = 4.1 on 2 dof, p = 0.13 — so the width model is not clearly
broken, but it is not established either.)

Required action: evaluate a grid with spacing ≤ 1×10⁻⁴ over `h ∈ [0.7295, 0.7315]` on at least
two seeds before any width is quoted; otherwise quote the empirical seed scatter only, per
the runbook's own criterion-2 branch.

### F-1 [HIGH — forking paths] The pre-registration protected the terms that did not matter and left the decisive lever ungated

This is the one place where the discipline failed, and it should be recorded as such.

`DERIVATION_GENERATOR_CONSISTENT_NORM.md` pre-registered a quantitative gate (§9.2): predicted
full-mixture gap `0.73 → 0.86` of **+92 ln (3D) / +52 ln (4D)**, with the explicit falsification
clause *"a measured gap far BELOW the prediction falsifies the §6.4 attribution"*. The measured
gap was **−899 ln**. That is a ~1000 ln, sign-flipped miss of a pre-registered prediction — the
strongest possible failure of the packet's own physical model of what it was doing.

The packet contained a second, *binary* modelling choice (§7.2: σ_z pairing, point/point vs
kernel/smeared) which it assessed as **immaterial**: *"measured consequence inside `D_gen`:
≲0.013/h — immaterial; decide for documentation coherence, not effect."* That assessment
considered only the choice's effect on `Σ_glob` inside `D_gen` and **not** its effect on the
numerator `N_g`. Its actual effect on the numerator is the entire result (P-1/P-2): it converts
~130 events from broad kernel-smeared likelihoods into delta-function host identifications,
changing the A-dominated per-event slope at truth from a predicted −0.23 ± 0.05 to a measured
−2.87 (a fact §3.22 records honestly).

So the chain is: a free binary choice → declared immaterial on an incomplete analysis →
therefore not pre-registered as a gated variant → and it is the *only* thing that closed the
bias. **This is a forking path even though nobody forked deliberately.** The good-faith
evidence is strong (the packet *predicted against itself*, and the closure was a surprise that
falsified the author's own hypothesis — the opposite signature of p-hacking). But
pre-registration does not confer protection on a claim when the decisive degree of freedom was
outside the registered set.

Required action: re-register the σ_z pairing as a first-class, gated estimator variant with
both arms run and reported (point/point *and* kernel), and state plainly in the paper which
arm produces which result. The kernel arm is the one that transfers to real data.

### P-4 [HIGH — regime of validity] "Generator-exact" makes the closure partly tautological, and FIX-3's normalization does not transfer to real data

`DERIVATION_GENERATOR_CONSISTENT_NORM.md` §3.3 states it explicitly: *"the generator IS the
reference"*. `n̂_w = W_cat/V_f(h)` and `D_gen = Σ_glob/n̂_w + β_Ḡ` are read off
`draw_mixture_hosts` / `draw_rate_weighted_hosts` / `compute_global_catalog_fraction`. This is
methodologically legitimate and I have no objection to it as a *code-correctness* exercise —
recovering truth when the estimator is the analytic inverse of the generator is a necessary
condition, and failing it (as the pipeline did for months) is diagnostic.

But it must be scoped as such. Three consequences:

1. Closure under a generator-exact estimator is **weak evidence for physical correctness**: it
   tests that the two code paths agree, not that either models a real universe.
2. The mock universe is itself not astrophysically self-consistent. The in-catalogue channel
   draws hosts ∝ `w_g` from the *mass-pruned* catalogue while the mixture fraction `F = 0.0175`
   is set by the *luminosity* completeness `f̄` of the *full* 22.6 M-row catalogue
   (`build_m_th_map` docstring: completeness is "catalog DEPTH (mass-independent) … NOT the
   mass-pruned subset"). The two selections are correlated (BH mass is derived from stellar
   mass, which correlates with B-band luminosity), so the mock's total host population is not
   the intended `p_pop`, and E1's own measurement confirms it: the catalogue's detected rate
   weight exceeds the constant-comoving-density (Option-A) prediction by **×1.334** in value
   and −0.39/h in slope. FIX-3 handles this by *matching the generator's idiosyncrasy* rather
   than by fixing the population model.
3. **`n̂_w = W_cat/V_f` has no real-data analogue.** For GLADE+ there is no `W_cat` draw
   normalizer and no known `F`; a real analysis is forced back to a Gray-2020-style
   Option-A-like assumption — i.e. back to the ×1.334 / −0.39/h inconsistency that FIX-3 was
   written to remove. The production estimator's central normalization is therefore
   **mock-only**. This is not currently stated anywhere; the "Open ends" caveat covers only the
   σ_z pairing.

### P-5 [MEDIUM] Selection modelling: consistent within the mock; two residual physical gaps

Positive findings (checked, no objection):

- **Malmquist / measurement-noise selection is handled correctly and is, in this mock, an
  easier problem than in reality.** Detection thresholds on the *true* SNR (oracle selection),
  while the observed `d_L` is Fisher-scattered *after* detection. Because the selection does
  not depend on the noise, `p(x|θ)` is unbiased and the correct normalization is
  `α(h) = E_θ[P_det(θ)]` — which is what `D_gen` computes. The residual oracle-vs-observed
  correction is measured second-order (E1 s6, +0.008). MFG (2019) convention (`P(det|x)=1` in
  numerators, one `α(h)` divisor) is applied consistently: no `p_det` appears in `N_g` or
  `B_num`, exactly one appears in the denominator. Verified in the code, `bayesian_statistics.py`
  consumer table of `DERIVATION_ZRESOLVED_SURVIVAL.md` §5.1.
- **Numerator/normalization domain coherence** is enforced (`W_cat` and `V_f` share the draw
  depth; the #30 depth cap moves them together; `B_num`'s upper limit was domain-matched in
  `7d3573d`). The dimensional analysis in FIX-3 §3.4 is correct and it correctly identifies
  E1's earlier `F·Σ_glob + (1−F)·β_Ḡ` as dimensionally inconsistent — a good catch.
- **FIX-2 is the strongest physics in the arc.** The pooled-survival conditioning error is a
  real, textbook error (`P(det|z,h)` requires `p(θ|z)`, not the z-marginal), the mechanism is
  identified (detector-frame mass lift, median horizon 0.89→1.59 Gpc), the estimator is
  bandwidth-insensitive over ×4, the earlier −0.56 estimate was correctly diagnosed as a
  binning artifact, ESS diagnostics are honest, and its isolated pre-registered prediction
  (−69 ln) was hit to 0.3 ln. This is a clean, verified physics fix. It is also — per §1 — very
  nearly irrelevant to the final MAP.

Residual gaps:

- **Catalogue BH masses and rate weights `w_g` are held h-invariant.** For real data,
  `M_* ∝ h⁻²` from flux, so `M_BH ∝ h⁻²·¹` (Reines & Volonteri 2015 slope 1.05) and
  `w_g = R_eff(M_g)/(1+z_g)` would carry a strong h-dependence through the steep EMRI rate–mass
  relation. Mock-internally consistent (generator and estimator share the frozen catalogue), so
  not a bias here; but it is an unmodelled h-channel that must be reinstated for real data and
  for the with-BH channel's interpretation.
- **The completeness `f(z,h)` is exactly h-independent** by the `M_* + 5log₁₀h` convention
  (`pixel_completeness.py:220–227`). That is the standard convention and is correctly
  implemented — I checked the cancellation. Worth one sentence in the methods section, because
  the reader's naive expectation (higher h → nearer → more complete) is wrong here.

### P-6 [MEDIUM] The seed600 shallow-venue offset is dismissed on an unquantified caveat

seed600 (p_det ≈ 1, catalogue-dominated — the regime closest to a *real* low-z dark-siren
analysis) gives MAP 0.745 (`volume_deconv`) and 0.755 (production stack) against truth 0.73:
**+2 % to +3.4 %**. This is excluded from absolute claims on an "Ω_m era mismatch" caveat that
is asserted but never quantified. I quantified it with the repo's own `dist()`: over
z ≤ 0.5, moving Ω_m from 0.2726 to 0.30/0.315 changes `d_L` by only 0.1–1.5 % (0.2 %/0.6 % at
z = 0.1; 0.4 %/1.0 % at z = 0.2), i.e. |Δh| ≲ 0.005 where seed600's information lives.
**The caveat under-explains the observed +0.015…+0.025 by a factor 3–5.** Either the residual
is real and unexplained, or the era mismatch is larger than Ω_m alone. Since the production
stack is *worse* than the superseded estimator on this venue (+0.025 vs +0.015), this deserves
an explicit diagnosis, not a caveat.

Related: the seed600 gate's criteria 3 and 4 **failed** as registered (3 new zero-likelihood
events). The diagnosis is excellent, complete, and quantitatively bounded (0.09 % of events,
z < 0.01, coarse-ball window vs GW window divergence under the sharp numerator), and it matches
the pre-registered risk 4 — which is a *point in the project's favour*. My only objection is the
framing "recommend conditional adoption … do not treat criteria 3/4 as blocking": a
pre-registered gate that fails should be recorded as failed and re-registered with the corrected
criterion, not reinterpreted. The follow-up the file itself proposes (quantify on the
production-depth venue) is the right action — and per §1, those low-z, tight-σ, high-SNR events
are precisely the *golden* population, so a mechanism that zeroes them is not a measure-zero
concern on deep venues either. Note the readout's claim "deep venues measured 0 hard zeros on
3454 events" is consistent with the diagnostics I read (n_empty = 0), so this is a
watch-item, not an active defect.

### P-7 [MEDIUM] The post-hoc seed900 exclusion: defensible, disclosed, but not airtight — and its failure mode is informative

The justification is strong by the standards that matter:
- The defect is in the **input**, not the posterior: `run_20260703_seed900/simulations/injections`
  symlinks a bespoke ~204-injection pool instead of the canonical `injection_pool_depth15_50k`.
- The diagnostic is **independent of the result**: 418/726 sky-band cells (57.6 %) below the ESS
  floor; node ESS min/median 6/55 vs 211/3944 for the canonical pool.
- There is a **matched control**: seed90000, same 20-event venue size, canonical pool, does not
  rail (MAP 0.7287 interior).
- The alternative explanation (out-of-grid quadrature) is ruled out by a counter-example.
- It is disclosed as post-hoc, both readouts are to be reported, and a re-run is ordered.

That is close to best practice. Two residual objections:

1. It is still an exclusion **discovered after seeing the rail**. The airtight version is a
   *provenance precondition applied to all venues before any readout* (an automated check that
   every venue's injection symlink resolves to the canonical pool and that the survival build
   reports 0 cells below the ESS floor). That check should exist in the pipeline, be run
   blind, and be cited — then the exclusion is a precondition, not a post-hoc judgement.
2. **The more interesting reading is being missed.** seed900 shows that when the completion
   estimator degrades and there are no golden events to out-vote it, the estimator rails HIGH
   — the same failure mode as the pre-FIX deep venue. Read alongside §1, this is evidence for
   the P-1 finding: the production stack's stability is supplied by the golden subset, not by
   the selection model. Do not file seed900 purely as a data defect; file it as a
   sensitivity datum.

### P-8 [LOW-MEDIUM] The completion channel remains unvalidated in both directions

The fallback (`A_i = 0`, 1996 events) subset peaks at ≈ 0.633 on the production stack (measured
above; consistent with the documented 0.61/0.647 fallback statistics). E1's argument that a
fallback-only closure is *not a theorem* (membership is h-informative) is correct and I accept
it — this is not by itself evidence of bias. But the converse must be accepted too: **the
full-mixture closure is not evidence that the completion channel is unbiased**, because the
full-mixture closure is carried by events whose likelihood is insensitive to it. The completion
term is currently supported by (i) E1's self-consistency MC against its own generator density —
again a code-consistency test — and (ii) nothing else. An impostor-capable P–P harness, or the
conditioned subset statistic, is the only way to close this.

### P-9 [LOW] Fixed Ω_m; single-parameter inference

Both generator and estimator fix Ω_m = 0.2726, Ω_de = 1 − Ω_m, w = −1 (and `physical_relations`
still ignores `w_0`, `w_a` — GitHub #4). The reported σ_h is therefore conditional on an exactly
known expansion history. This is a legitimate mock design (§CLAUDE.md G11) but must be stated
with the result, since the quoted precision (0.04 %) is far below the level at which Ω_m
uncertainty would matter for events at z ≈ 0.4 (a 3 % Ω_m error moves d_L by ~1 % at z = 0.5).

---

## 3. Direct answers to the five questions posed

**Q1 — is the mock-exact shortcut cleanly quarantined, and is σ_h ≈ 3×10⁻⁴ physically
plausible?** Quarantined *in words* (three places), **not quarantined in effect**: it is the
mechanism that produces the result (P-2, F-1). The value is not "double counting" — it is
arithmetically exactly right for what it is. Perfect host identification for *all* 3470 events
would give σ_h = 1.97×10⁻⁴; the measured 2.6×10⁻⁴ is 1.3× that, and the golden subset alone
gives 1.9×10⁻⁴ before the shared-normalization dilution. No information is created from
nothing. But the *physical* reading — "3454 LISA dark sirens measure H₀ to 0.04 %" — is wrong
by construction: ~130 events act as bright sirens, and in reality they would not.

**Q2 — selection modelling consistency.** Numerator/denominator selection is consistent and
correctly MFG-normalized; the injection-pool survival estimator is sound and FIX-2 fixes a real
conditioning error; domains are matched. Gray et al. (2020) is *specialized*, not reproduced:
FIX-3 deliberately replaces the Gray `β_G` catalogue-side selection integral with a
generator-read normalizer that has no real-data analogue (P-4). The `f→1` limit does return the
Gray A9 form, which is the right consistency check and it passes. Depth statement: the
catalogue supports 9,060,008 galaxies out to z < 0.992 (not z ≈ 0.3) — but the *information*
ends at z ≈ 0.17 (golden-event maximum), so any "effective catalogue depth" statement in the
paper should be phrased in terms of where the H₀ information lives, not where rows exist.

**Q3 — do the pre-registrations protect the closure claim?** Partially. They protect against
deliberate truth-tuning and they demonstrably did their job (recorded failures, falsified
predictions, honest QUALIFIED FAIL). They did **not** protect the specific claim, because the
decisive degree of freedom (σ_z pairing) was assessed as immaterial and left ungated (F-1).
Decisive additional tests: §4 T-1 (alternative-truth mock), T-2 (golden ablation), T-3 (realism
ablation). The seed900 exclusion is defensible and disclosed but not airtight (P-7).

**Q4 — would it close at a different truth?** I found **no numerical anchor to 0.73**. The
completeness map `m_th` is apparent-magnitude-based and h-free; `f(z,h)` is h-invariant by the
`M_*+5log h` convention (verified in code); `F = 0.017537` is h-constant; `V_f ∝ h⁻³` exactly;
`W_cat` is an h-free catalogue sum; `d_hor_k` is h-free; the grid is symmetric about neither
0.73 nor anything else in a way that matters. Structurally, the golden mechanism solves
`d_L(z_g;h) = d_L^obs` and is anchored to the *data*, so a mock generated at h = 0.70 should
peak at 0.70. **Prediction: it closes at a different truth — and that is exactly why the test
is worth running, because a *failure* would reveal an anchor nobody has found by inspection,
and a *pass* is the strongest anti-tuning evidence available.** Two residual anchors to check
in that test: the injection pool's `(z_k, d_hor_k)` pairing (assigned at h_inj = 0.73) and the
catalogue's h-frozen `M_g`/`w_g` (P-5). Concrete specification: §4 T-1.

**Q5 — is MULTISEED_READOUT correctly scoped?** Mock-internal validity: yes, and the document
is admirably careful about provenance, disclosure and the width caveat. Physical scoping: **no**
— the sentence "the production stack shows no detectable bias at the 4×10⁻⁴ level" needs the
qualifier that the test's power is concentrated in ~4 % of events and that the dark-siren
components are untested by it (P-1). Real-data readiness is correctly *not* claimed, and the
"Open ends" note is honest — but the note understates the size of the transfer gap (P-2, P-4).

---

## 4. Ranked decisive follow-up tests

**T-1 [DECISIVE, must-run before any closure claim] — alternative-truth mock.**
Regenerate a full venue at `h_inj = 0.68` (or, better, at a value drawn blind by a script and
sealed until readout) with the same catalogue, same completeness cache, same population model,
and a freshly generated injection pool at that cosmology. Run the *unchanged* production stack
on a 41-point grid spanning the new truth. Pre-register: MAP within 2×(empirical seed scatter)
of 0.68, interior, plus the T-4 pull test. Cost: one simulation campaign + one eval array —
the same cost as one existing seed. This is the only test that can rule out "the estimator was
assembled, over 20+ variants, until it pointed at 0.73".
*Cheap partial (hours, not days, but NOT a substitute):* rescale `luminosity_distance` and its
CRB error by 0.73/0.68 in a copy of `prepared_cramer_rao_bounds.csv` and re-evaluate. Because
`d_L ∝ 1/h` at fixed z, this is exact for the golden mechanism but leaves the selection
modelling matched to the h = 0.73 detected set. A pass confirms the §1 mechanism; it does *not*
test the selection model. Report both, and label the cheap one as a mechanism check.

**T-2 [DECISIVE for the dark-siren claim] — golden-event ablation, properly conditioned.**
Re-run the combine on the complement of the informative subset and see whether the dark-siren
machinery closes on its own. Because subset membership is h-informative (E1's FIX-1 warning is
correct and must be respected), the subset must be defined by an **h-independent** criterion —
e.g. "events with no catalogue galaxy within 4σ_dL of `d_L^obs` at *any* h in the grid", or
simply `z_obs > 0.2` — and scored with the membership-conditioned statistic
`Σ log p_i − N·log P_subset(h)`. Pre-register the expected result. If the complement rails (my
prediction from §1: it does), the paper must say so: the completeness-corrected dark-siren
estimator is *not yet* demonstrated unbiased, and the closure is a host-identification result.

**T-3 [DECISIVE for every precision claim] — photo-z/peculiar-velocity realism ablation.**
Re-run seed1000 in `absolute_marginal` (kernel σ_z pairing) with `σ_z_pv` restored *and* with a
GLADE-realistic catalogue z-error floor injected into the *generator* as well (so the mock
stays self-consistent: scatter the host's observed catalogue z, keep the true z for the
waveform). Report MAP and width. Expected: σ_h degrades by ≥3–7× (§P-2) and the closure must
be re-established under the *transferable* estimator. This is the number the paper should
quote as the EMRI dark-siren forecast. Without it, no precision statement should appear.

**T-4 [CHEAP, run immediately — costs one script] — per-event pull calibration, promoted to a
standing gate.** Formalize the test I ran in §1: on every venue, extract per-event `(MAP_i,
σ_i)` from the dense-core diagnostics for all events with `curv_i > 0` above a fixed threshold,
and test the pull distribution against `N(0,1)` (KS + mean + std). Seed1000 passes
(0.06 ± 0.94, n = 133). This is a far more powerful, far cheaper bias/calibration test than the
4-seed t-test on MAPs (which has 3 effective degrees of freedom), and it is immune to the
forking-paths critique because it is a *within-venue* calibration statement. Report it per seed
and per channel.

**T-5 [HIGH VALUE] — resolve the peak.** A ≤1×10⁻⁴-spaced grid over `h ∈ [0.7295, 0.7315]` on
seeds 1000 and 2000, both channels. Settles P-3 and makes the width quotable (or kills it).
Trivially cheap relative to a 41-point grid.

**T-6 — impostor-capable P–P coverage.** Finish the multi-galaxy-catalogue extension of
`validation/pp_coverage.py` (worktree branch @ `7c513dd` already has `mixture_mode="absolute"`;
the honest null is already recorded in `OVERNIGHT_REPORT_20260726.md`). Only a synthetic
universe with impostor balls *and* a generative photo-z kernel can test the completion channel
and the association estimator jointly under many truths. This is the instrument that would have
caught §3.21's misassociation mechanism before it cost months.

**T-7 — provenance precondition, automated.** Turn the seed900 diagnosis into a blind gate:
assert every venue's injection symlink resolves to the canonical pool and that the survival
build reports 0 (band, node) cells below the ESS floor, *before* any posterior is read.
Then re-run seed900 (already ordered) and report the registered n = 5 test alongside the
valid-4 result.

**T-8 — quantify the seed600 residual.** Either explain the +0.015…+0.025 shallow-venue offset
(P-6: the Ω_m caveat covers ≲0.005) or re-run seed600 with an era-consistent CRB set. Since
seed600 is the venue most like a real low-z analysis, an unexplained 2–3 % offset there is a
paper-level concern.

---

## 5. What the paper may and may not say (concrete)

**May say:**
- "The generator-consistent estimator recovers the injected H₀ on four independent mock venues
  with no bias detectable at the 4×10⁻⁴ level, and with per-event pulls consistent with
  `N(0,1)` (0.06 ± 0.94, n = 133)."
- "The z-resolved detection survival corrects a +30–45 % conditioning error in the pooled
  survival estimator across 0.1 ≤ z ≤ 0.65, verified against the injection pool's own stratum
  detection rates."
- "Host misassociation in a self-normalized candidate-ball likelihood was identified as the
  dominant bias mechanism and is removed by the absolute-mass marginal."

**May not say (without T-1/T-2/T-3):**
- Any statement of the form "LISA EMRI dark sirens constrain H₀ to ~0.04 %".
- "The completeness correction / selection modelling is validated / unbiased."
- "3454 events yield σ_h = 3×10⁻⁴" without stating that ~4 % of events supply ~100 % of the
  information and that they act as bright sirens because the mock's catalogue redshifts are
  exact.
- Any width at all until T-5 resolves the peak.

---

## 6. Reproduction of this review's measurements

All from committed artifacts, CPU-only, seconds:

```python
# per-event decomposition (Section 1)
d = pd.read_csv("results/lcat_h_dependence_20260725/densecore_probe/"
                "simulations/diagnostics/event_likelihoods.csv")
p = d.pivot(index="event_idx", columns="h", values="combined_no_bh")
h = np.asarray(p.columns, float); ln = np.log(np.where(p.values > 0, p.values, np.nan))
i = list(h).index(0.730)
slope = (ln[:, i+1] - ln[:, i-1]) / 0.010
curv  = -(ln[:, i-1] - 2*ln[:, i] + ln[:, i+1]) / 0.005**2
# sigma = 1/sqrt(curv.sum()); MAP = 0.73 + slope.sum()/curv.sum()
# golden = curv > 1e4  -> n=133, carries 100.4% of curv.sum()
# pulls  = (slope[golden]/curv[golden]) * sqrt(curv[golden])  -> mean .062, std .944
```
Cross-match to `run_20260703_seed1000/simulations/prepared_cramer_rao_bounds.csv` by
`event_idx` (positional row index; 3454 of 3470 rows evaluated) for `d_L`, `σ_dL`, and the
`σ_h = 0.73·σ_dL/d_L` comparison. Ω_m sensitivity via `master_thesis_code.physical_relations.dist`.

No files outside `results/redteam_20260726/` were modified by this review; no jobs were
submitted; no pipeline code was changed.
