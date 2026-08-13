# M1 — "bare kernel" (missing comoving-volume/rate prior) — mechanism study

**Status:** analysis only, read-only on `darksiren_emri/`. No code edited, no `/physics-change`
gate opened, nothing committed. Investigator note for the author's mechanism ledger on the
1,400-seed venue-transfer defect (`results/venue_transfer_20260811/VENUE_TRANSFER_READOUT.md`):
bias = +1×σ_z in h, 0/400 coverage, width ~8.5× smaller than the displacement.

**VERDICT: M1 REFUTED AS SOLE MECHANISM — and on stronger grounds than the orchestrator's prior
(wrong scaling AND wrong sign) — but SURVIVES as a plausible compounding secondary correction
that reproduces the observed R_dose drift (1.07 → 1.01 → 0.88–0.95) when superposed on an
unidentified dominant linear-in-σ_z, positive-sign mechanism.**

---

## 1. What `pp_coverage.py`'s `bare` vs `volume` switch actually computes

`darksiren_emri/validation/pp_coverage.py` is an independent from-scratch reimplementation of the
dark-siren estimator (not import-linked to production) built by the 2026-07-01 verification
commission specifically to test this defect. It carries a `Literal["bare", "volume"]` kernel
switch (`PPCoverageConfig.kernel`, line 546) that is structurally the same object as the
estimator under investigation here.

**The kernel code (lines 868–872):**

```python
kernel_z = _norm_pdf(zq, float(z_gal[i]), sigma_z)  # (nz,)   <- bare, always computed
if config.kernel == "volume":
    kernel_z = kernel_z * _inference_population_weight(zq, config.inference_wpop_tilt)
    kernel_z = kernel_z / max(float(np.trapezoid(kernel_z, zq)), 1e-300)
num = (wq * kernel_z) @ pGW  # (nh,)
```

- **`bare`** (line 868 only): `kernel_z = N(z; z_gal, σ_z)` — a plain Gaussian in z, unweighted.
  This is *exactly* the object M1 accuses: `venue_transfer.py:1136` computes
  `kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])` — the identical bare-Gaussian
  form, no population weight, no renormalization by any Z_g.
- **`volume`** (lines 869–871, the default per line 546): multiplies the bare Gaussian by
  `_inference_population_weight(z, tilt) = w_pop(z)·exp(tilt·z)` (line 289, tilt=0 by default
  ⇒ exactly `w_pop(z) ∝ dV_c/dz / (1+z)`, defined at line 238 and computed from the flat-ΛCDM
  tables built at lines 197–211), then **renormalizes the product kernel to unit z-integral**
  (`kernel_z /= ∫kernel_z dz`, line 871). This is precisely M1's prescribed fix:
  `p(z_true|z_obs) ∝ N(z_obs; z_true, σ_z) · p_pop(z_true)`, properly normalized.

The module docstring (lines 24–28) states the pre-registered finding this switch was built to
confirm: *"with photo-z scatter σ_z ≈ 0.035 the bare-Gaussian host-z kernel carries a fixed
`~ -σ_z²·d ln(dV_c/dz)/dz` (Eddington/Malmquist-in-z) **low** bias in H0 that collapses coverage to
~0–3%, while the volume-weighted kernel is calibrated (coverage ≈ nominal, bias ≈ 0)."**

### 1.1 Committed results comparing `bare` vs `volume`

**`results/commission_20260701/scratch/d2/NOTE_calibration_findings.md`** (the origin note, plus
`coverage_results.json`) — the *cleanest* controlled test, run by the original independent
investigator (d2) who wrote pp_coverage's scratch predecessor before the module existed:

- **RESULT 1** (single host, f≈1, no completion term, 120 realizations × 250 events, three
  truths): FLAT/bare numerator coverage **collapses to 0.02–0.08 at all levels** with a **fixed
  −0.022 to −0.025 low bias** in H0 at every truth (0.66/0.72/0.78). VOLUME numerator: coverage
  ≈ nominal (0.53–0.88 across levels), bias ≈ −0.002 to −0.003 (essentially calibrated).
- **RESULT 2** (σ_z scan, h_true=0.72, FLAT/bare numerator only): MAP bias vs σ_z = **−0.0016 /
  −0.0064 / −0.023 / −0.046** at σ_z = 0.005 / 0.015 / 0.035 / 0.050 — this is the explicit
  empirical basis for the commission's `~σ_z²` claim; VOLUME stays ≈ −0.002 at every σ_z (flat,
  i.e. calibrated across the whole range).
- **RESULT 3** (full estimator with completion term + interlopers, H0_true=0.72, 120 realizations
  × ~160 events): bare-numerator production analog (`A_prod`) carries **−0.038** bias at σ_z=0.035
  vs **−0.013** for the volume-corrected (`B_corr`); coverage 0.00→0.40/0.54/0.82.

`results/pp_coverage_sigmaz_scan_20260703/SUMMARY.md` extends the comparison to order-unity
σ_z/z (σ_z ∈ {0.10, 0.15, 0.25}, well outside the venue's 0.011–0.042 regime): bare **rails to the
grid edge in ~100% of realizations at every σ_z ≥ 0.10** (bias −0.02 to −0.24, growing with σ_z,
zero coverage throughout); volume stays near-nominal out to σ_z/z ≈ 0.5–0.8 then degrades. Because
of the rail saturation this campaign is **not usable** for a clean small-σ_z scaling check (the
"MAP bias" there is dominated by the h-grid boundary, not the underlying mechanism) — flagged as a
caveat, not used quantitatively below.

**Conclusion of §1: the bare/volume switch is real, load-bearing, and has already been validated
in a controlled synthetic — but every single comparison on record shows the bare kernel biasing
H0 LOW. This sign is the crux of §4.**

---

## 2. Derivation: the shift a missing p_pop(z) prior induces

### 2.1 Setup

Single candidate/host redshift nuisance parameter z, observed via a noisy photo-z
`z_obs = z_true + N(0, σ_z)` (constraint (c): zero-mean, generator and estimator share σ_z). The
true redshift has population prior density `p_pop(z_true) ∝ (dV_c/dz)·R(z)/(1+z)` (M1's claimed
correct object; `dV_c/dz` is the comoving volume element, `R(z)` the EMRI rate, `1/(1+z)` the
observer-frame time-dilation of the rate — exactly `w_pop(z)` as coded at `pp_coverage.py:238`).

The event's z-marginal likelihood contribution, correctly done, is

```
L_correct(z_obs) ∝ ∫ dz_true  p(z_obs | z_true) · p(d_obs | z_true, h) · p_pop(z_true)
```

i.e. `p_pop` enters as the **prior on the true host redshift**, exactly as it enters every other
term in the estimator (the selection denominator `D(h)`/`α(h)` and completion terms all already
carry it — that asymmetry is the defect). The bare kernel instead treats
`N(z; z_obs, σ_z)` as if it *were* `p(z_true | z_obs)` directly, omitting `p_pop`.

### 2.2 Expansion of E[z_true | z_obs]

Let `λ(z) ≡ d ln p_pop(z)/dz`, the local log-slope of the population prior. For `σ_z` small
compared to the curvature scale of `ln p_pop`, Taylor-expand around `z_obs`:

```
p_pop(z) ≈ p_pop(z_obs) · exp[λ(z_obs)·(z − z_obs)]
```

The correct posterior (Bayes, using the symmetric Gaussian `N(z_obs; z, σ_z) = N(z; z_obs, σ_z)`):

```
p(z_true | z_obs) ∝ N(z; z_obs, σ_z) · exp[λ·(z − z_obs)]
                  ∝ exp[ −(z−z_obs)²/(2σ_z²) + λ(z−z_obs) ]
                  = exp[ −(1/2σ_z²)·( (z − z_obs − σ_z²λ)² − σ_z⁴λ² ) ]
```

Completing the square shows this is **still Gaussian**, with the *same* variance σ_z² to leading
order, but a **shifted mean**:

```
E[z_true | z_obs]_correct = z_obs + σ_z² · λ(z_obs) + O(σ_z⁴, curvature of λ)
```

The bare kernel's implicit mean is `E[z | z_obs]_bare = z_obs` exactly (a bare Gaussian is
symmetric around its own center — no shift, by construction, regardless of σ_z).

**The missing-prior shift is therefore**

```
Δz_M1(z_obs) ≡ E[z_true|z_obs]_correct − E[z|z_obs]_bare = σ_z² · λ(z_obs)
```

**QUADRATIC in σ_z**, with sign set by the sign of λ(z_obs) — i.e. this is the standard
Eddington/Malmquist bias for a Gaussian-scattered observable drawn against a sloped population
prior (Eddington 1913/1940; textbook form e.g. Teerikorpi 1997).

### 2.3 Sign, evaluated on the venue's actual population

`λ(z)` was evaluated numerically from `pp_coverage.py`'s own flat-ΛCDM tables
(`population_weight_of_z`, i.e. `w_pop(z) ∝ dV_c/dz/(1+z)`, and the bare `dV_c/dz` variant used in
the module's headline formula):

| z | λ = d ln w_pop/dz | λ = d ln(dV_c/dz)/dz |
|---|---|---|
| 0.30 | 5.09 | 5.86 |
| 0.50 | **2.26** | **2.93** |
| 0.70 | 1.15 | 1.74 |
| 1.00 | 0.40 | 0.91 |

**λ is positive throughout the entire redshift range the venue's pinned population occupies.**
The venue's 982-event population (`results/campaign51_20260728/realistic_20260729/seed61000/
prepared_cramer_rao_bounds.csv`, CRB CSV md5 `9a1f2a14…`, the same file V-T3-pinned by the
venue-transfer campaign) has, decoded via `pp_coverage.z_of_comoving_amplitude(d_L·h)` at
h=0.73: **z median = 0.494, mean = 0.490, IQR [0.35, 0.63], full range [0.016, 1.13]** — squarely
in the λ ≈ 1.2–5 zone, never near the (much higher-z) turnover where λ would flip sign.

### 2.4 Direction of the induced H0 bias

Because `p_pop` is increasing near the venue's typical z (λ>0), the correct posterior mean is
**higher** than z_obs — more true hosts scatter *up* into a given z_obs from the larger reservoir
of population density below/around it than scatter down from above (standard Eddington bias:
under-dense high-z tail, over-dense low-z reservoir feeding it). The bare kernel, lacking this
correction, uses a systematically **too-low** effective z for every host redshift marginalization.

Since `d_L(z,h) = A(z)/h` with A(z) monotonically increasing, and the GW-distance term pins
`z_GW(h)` (the redshift implied by the trial h and the observed d_L) to increase with h: an
estimator whose z-kernel is centered too low (bare) needs a **lower** h to make `z_GW(h)` match
that too-low kernel center than an estimator with the correct (upward-shifted) kernel. **M1
therefore predicts the bare kernel biases H0 LOW** — exactly the sign `pp_coverage`'s controlled
experiment measured (§1.1: −0.022 to −0.046 across σ_z, always negative, never once positive in
any committed run).

---

## 3. Magnitude vs the observed +0.037

Using σ_z ≈ 0.035–0.042 (the venue's dose range) and λ(z≈0.49) ≈ 2.3–2.9:

```
Δz_M1 ≈ σ_z² · λ ≈ (0.038)² × 2.5 ≈ 0.0036
```

Propagating through `d ln A/dz` at z≈0.49 (from the same table) gives `Δh_M1 ≈ h·(d ln A/dz)·Δz`,
of order **0.01–0.04** in h — i.e. **M1's predicted magnitude is the same order as the observed
total defect** (+0.035 to +0.041). This is corroborated directly (not just by propagation) by the
`pp_coverage` empirical numbers already in hand: bare-kernel bias at σ_z=0.035 is −0.023 (clean
single-host test) to −0.038 (full estimator with completion + interlopers) — **essentially
identical in magnitude to the observed venue-transfer defect, but with the opposite sign.**

**This is the load-bearing result of this note: M1, taken as the sole mechanism, gets the
magnitude right and the sign wrong.**

---

## 4. Falsification against the three constraints

### (a) Vanishes identically at σ_z = 0 — PASSES (trivially, non-discriminating)

`Δz_M1 = σ_z²λ` vanishes exactly at σ_z=0, consistent with the T-0 anchor (200/200 seeds exactly
on truth). This is necessary but not diagnostic — essentially any physically sensible mechanism
tied to photo-z error vanishes at σ_z=0, so T-0 cannot distinguish M1 from other candidates.

### (b) Linear in σ_z — FAILS as sole explanation, as expected; the residual pattern is interesting

M1's leading-order term is **quadratic** in σ_z, not linear. Checked directly against the RESULT-2
σ_z scan (§1.1): bias = −0.0016 / −0.0064 / −0.023 / −0.046 at σ_z = 0.005/0.015/0.035/0.050. A
log-log slope between successive points is 1.26 → 1.51 → 1.95 (not a clean 2.0 throughout — likely
a finite-N=120-realization noise floor at the smallest σ_z where the bias itself is only ~0.2× the
per-realization MAP scatter — but trending toward 2 and consistent with quadratic, never close to
1). **This directly contradicts the observed near-linear R_dose behavior if M1 is treated as the
sole driver** — confirming the orchestrator's prior expectation.

**However**, the observed R_dose *drift itself* (1.07 at dose ≈0.010–0.011 → 1.01 at dose 0.035 →
0.88–0.95 at the GLADE-mix dose ≈0.0415–0.0421 — venue-transfer readout §2.3, and the matched
calibration-gate-v2 B1/B2 cells) is exactly what a **linear-plus-negative-quadratic** compound
predicts: if `bias(σ) = a·σ + b·σ²` with a>0 (dominant, unidentified, linear, positive — the
"+1×σ_z" signature itself) and b<0 (M1's own quadratic, negative-in-H0 by §2.4), then
`R_dose(σ) = bias/σ = a + b·σ` is **linearly decreasing in σ** — precisely the observed shape.

A 2-parameter least-squares fit of `R_dose = a + b·σ` across all seven measured 1D-argmax cells
(B1/T-a/T-b/B2(0.730)/T-c(0.690/0.730/0.770), sources: `calibration_gate_v2_20260810/
PREREGISTRATION_CALIBRATION_GATE_V2.md` lines 341–348 and `venue_transfer_20260811/
VENUE_TRANSFER_READOUT.md` §2.3) gives:

```
a ≈ 1.15   b ≈ −5.29  (residuals up to ±0.06 — a rough fit, not a precision one)
```

`|b|` (the fitted quadratic-in-σ contribution to R_dose, i.e. the coefficient of the σ² term in
`bias(σ)` after dividing by σ) is compared to `λ(z≈0.49) ≈ 2.3–2.9` from §2.3: same order of
magnitude (factor ≈ 1.8–2.3×), not an exact match — expected, since the fitted `b` folds in the
*actual* multi-candidate ball structure (K_i averaging, the `_IMPOSTOR_KERNEL_WINDOW` truncation,
the GW-distance kernel width) that the single-host clean derivation of §2 does not model, and the
population weight the estimator would need to add is evaluated against the z **distribution of
in-ball candidates**, not the raw z_true population — plausibly a steeper effective λ than the
raw `dV_c/dz/(1+z)` slope. **Order-of-magnitude consistency, not proof.**

### (c) Not misspecification — PASSES, not applicable to M1

M1's mechanism is a missing-prior defect in the estimator's *own* z-marginal — it requires no
generator/estimator σ_z mismatch to operate (unlike a hypothetical M-misspecification candidate).
Constraint (c) rules out one *class* of alternative mechanisms, not M1 itself; M1 is unaffected by
it either way.

---

## 5. Compounding assessment (the honest, load-bearing conclusion)

M1 cannot be the sole mechanism: it predicts the **wrong sign** (LOW H0, not the observed HIGH)
and the **wrong leading-order scaling** (quadratic, not linear) when taken alone. Both of these are
independently confirmed — the sign by every committed `bare`-vs-`volume` comparison on record
(§1.1, always negative for bare), the scaling by the RESULT-2 σ_z scan (§4b).

But M1 is a **real, independently-verified estimator defect** (not a hypothetical): the exact code
object it accuses (`venue_transfer.py:1136`'s `norm.pdf(z_nodes, loc=zo, scale=so)`) is byte-for-
byte structurally identical to the object `pp_coverage.py`'s `bare` mode reimplements and the
commission already demonstrated is uncalibrated, with a magnitude (§3: ~0.02–0.04 in h at
σ_z=0.035) comparable to the observed total defect. Its predicted sign is a **compounding,
opposing correction**, not the driver — and the observed R_dose drift (declining with dose, §4b)
is the qualitative signature such a compounding correction would leave on a dominant positive
linear mechanism.

**Reading:** the venue-transfer defect is very plausibly the sum of (i) an as-yet-unidentified
dominant mechanism, linear in σ_z, positive-signed, of magnitude ≈ +(1.1–1.2)×σ_z at small dose
(matching the near-exact "+1×σ_z" finding at the lowest doses), **compounded with** (ii) M1's
quadratic, negative-signed correction of order σ_z²·λ ≈ 0.02–0.04 at σ_z≈0.035–0.042, which eats a
growing fraction of (i) as the dose rises — producing the observed decline of R_dose from ~1.07
toward ~0.88–0.95. M1 is therefore a **secondary, subtractive fingerprint riding on a larger
unidentified linear driver**, not the primary defect.

---

## 6. Cheapest decisive experiment

`pp_coverage.py`'s existing `kernel="bare"/"volume"` switch (§1) already isolates M1 for
~free — no new code is needed to test M1's *existence and sign in a matched synthetic*, only to
test it **on the actual venue structure** (982-event pinned population, real K_i multiplicity,
GLADE-heterogeneous σ_z). Per constraint-of-record ("per-seed bias is ~7σ in a single seed"),
N≈10–25 seeds is enough to resolve a mean shift at high confidence.

**Proposed test (not run here — analysis-only mandate):** in a scratch/test-only copy of
`_channel_terms_at_h` (never touching the production file), apply the identical 2-line
`pp_coverage`-style patch to the bare kernel at line 1136 — multiply
`kern = norm.pdf(z_nodes, loc=zo, scale=so)` by `w_pop(z_nodes)` (the same `dV_c/dz/(1+z)` object,
already implemented and tested in `pp_coverage.py`) and renormalize the z-integral — then rerun
N=10–25 matched seeds at the T-c (GLADE-mix, σ̄≈0.0415–0.042) dose, holding everything else fixed
(same seeds, same K_i ball structure, same α(h) normalization).

**The decisive, falsifiable prediction from §5:** if M1 is genuinely a *negative* compounding
correction riding on a larger positive driver, adding the volume prior should make the residual
bias **larger and more positive** (not smaller/closer to zero) — the opposite of what "fixing a
bug" naively suggests, and a clean way to adjudicate the compounding-vs-irrelevant question in one
cheap run. If instead the bias shrinks toward zero, M1 is not compounding as modeled here and this
note's sign/mechanism analysis needs revisiting.

---

## Sources

- `darksiren_emri/validation/pp_coverage.py` — lines 1–171 (docstring), 197–270 (cosmology tables,
  `population_weight_of_z`, `galaxy_number_weight_of_z`), 289–332 (`_inference_population_weight`),
  421–460, 536–546 (`kernel` field), 868–872 (bare/volume kernel switch).
- `darksiren_emri/validation/venue_transfer.py` — lines 1099–1190 (`_channel_terms_at_h`), in
  particular line 1136 (`kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])`, the bare
  kernel M1 accuses) and 1163–1173 (`L1`, `ln1_k`, `α(h)` normalization).
- `results/commission_20260701/scratch/d2/NOTE_calibration_findings.md` +
  `coverage_results.json` — origin controlled comparison (RESULT 1/2/3).
- `results/pp_coverage_sigmaz_scan_20260703/SUMMARY.md` — order-unity σ_z/z extension (not used
  quantitatively; rail-saturated).
- `results/venue_transfer_20260811/VENUE_TRANSFER_READOUT.md` — §1 (T-0 anchor), §2.3 (DS-VT3
  bias/R_dose table), §3 (ablation ladder).
- `results/calibration_gate_v2_20260810/PREREGISTRATION_CALIBRATION_GATE_V2.md` (lines 246–348) +
  `CALGATE_V2_READOUT.md` (lines 66–120) — B1 (σ_z=0.010) / B2 (σ_z=0.035) dose cells.
- `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv` — the
  V-T3-pinned 982-event population (md5 `9a1f2a14384a9281c97ca3be312ddaab`); z decoded via
  `pp_coverage.z_of_comoving_amplitude(luminosity_distance · h)` at h=0.73: median 0.494, IQR
  [0.35, 0.63], range [0.016, 1.13].
