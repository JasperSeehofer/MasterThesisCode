# Catalogue Data-Usage Check & Likelihood-vs-Posterior Fork

**Task:** Option 3 — a CHECK before prototyping. No code changes, no derivation.
**Date:** 2026-06-30
**Scope:** Confirm we are USING GLADE+ data correctly, and SETTLE the load-bearing
posterior-vs-likelihood interpretation of the per-host redshift PDF
`N(z; z_g, σ_z)`. This decides the concrete form of the Hint-4 fix.

**Inputs synthesised:** (a) literature provenance, (b) in-repo code audit
(spot-verified live against `handler.py` and `bayesian_statistics.py`), (c)
adversarial verification of the interpretation against the three primary PDFs.

**Bottom line (one sentence):** GLADE+ flag-1 photo-z is an empirical
neural-network estimate from broadband colours — **not** distance-derived and
**not** cosmology-circular; flag-2 (the genuinely circular `d_L→z` class) is
**correctly excluded**; the per-host `N(z; z_g, σ_z)` must be treated as a
**LIKELIHOOD** (multiply by `p_bg ∝ dV_c/dz` exactly once), so the Hint-4 fix is
**ADD one dV_c** to the in-catalogue host numerator (net 0 → 1).

---

## 1. Photo-z mechanism & the circularity question (the user's question)

### 1.1 How a GLADE+ flag-1 redshift is actually produced

GLADE+ flag-1 photometric redshifts come overwhelmingly from two empirical
machine-learning catalogues that GLADE+ ingests, **not** from SED-template fitting
with a cosmological/luminosity prior:

- **WISE×SuperCOSMOS (WISE×SCOSPZ, Bilicki et al. 2016, ApJS 225, 5)** — the
  dominant source. Method: the **ANNz** artificial-neural-network package, an
  **empirical regression trained on GAMA-II spectroscopic redshifts** (~193,500
  overlapping galaxies). Inputs: 4 broadband fluxes/colours — WISE **W1 (3.4 µm)**
  and **W2 (4.6 µm)** mid-IR + SuperCOSMOS **B_J, R_F** optical (photographic
  plates). Performance: scatter `σ_z/(1+z) ≈ 0.033`, systematic `|Δz| ~ 1e-3`,
  ~3% outliers.
- **2MPZ (Bilicki et al. 2014)** — same ANN family, trained on spec-z, using
  2MASS J,H,Ks + WISE + SuperCOSMOS.
- GLADE+ also folds in SDSS-DR16Q and other catalogue redshifts.

Mechanism: **broadband colour → empirical neural-network regression → redshift**
— the same wavelength-stretch physics as low-resolution spectroscopy, exactly as
the project's physics background assumed.

### 1.2 Is the photo-z derived from luminosity distance? Is it circular?

**NO — flag-1 photo-z is NOT distance-derived and is NOT circular for a
dark-siren H0 measurement.** [CONFIDENCE: HIGH]

Decisive evidence (verbatim from Dalya+2022, arXiv:2110.06184, Sec. 4, col 35,
verified via ar5iv/pdftotext):

- flag **1** = "measured **photometric** redshift, **from which** we have
  calculated its luminosity distance" → the direction is **`z → d_L`** (z is the
  primary measurement, d_L is the downstream product).
- flag **2** = "measured **luminosity distance**, from which we have calculated
  its redshift" → direction **`d_L → z`**. **This is the genuinely circular
  class** for dark sirens: it feeds a cosmology-assumed distance back into the
  cosmology measurement.
- flag **3** = "measured **spectroscopic** redshift" (`z → d_L`).
- flag **0** = no measured redshift or distance.

So for the retained classes `{1,3}` the redshift **never passes through `d_L`**.
The reverse direction (`d_L → z`) is exactly flag 2, which we drop.

**Cosmology dependence: negligible.** The dominant flag-1 engine (ANNz/ANN) is an
empirical regression on spec-z — **no luminosity prior, no cosmological model** in
the photo-z estimate itself (unlike Bayesian SED-template codes such as BPZ that
apply a `P(z,T,m)` HB/luminosity prior). The only cosmology in GLADE+ is the fixed
flat ΛCDM (`H0=67.66, Ω_m=0.3111`) used to convert `z → d_L` for the catalogue's
distance column — a downstream product **not** used as our dark-siren redshift.

**Adversarial correction (kept):** "fully prior-free" is too strong. Empirical ANN
photo-z inherit an **implicit redshift prior from their spectroscopic training
set's n(z)** (a standard ML photo-z effect). However that implicit prior is **not a
comoving-volume prior**, so it does **not** affect the dV_c bookkeeping. The
"modulo weak template/training dependence" caveat should be retained, but it does
not change any conclusion below.

### 1.3 Flag-2 exclusion status — VERIFIED

- **Code logic:** `handler.py:297-300` admits a row only if
  `REDSHIFT_FLAG == 1 OR REDSHIFT_FLAG == 3`. Any flag-2 (or flag-0) row fails both
  clauses and is filtered out at parse time. Confirmed live this session.
- **Flag semantics in code** match Dalya+2022 exactly: `handler.py:149-150`
  (`0=none, 1=PHOTOMETRIC, 2=lum. distance, 3=SPECTROSCOPIC`) and the comment block
  at `:284-287`. `CatalogueColumns.REDSHIFT_FLAG = 34` (0-based) = GLADE+ raw col 35.
- **Caveat (provenance, not a bug):** the flag column is dropped from the reduced
  CSV (`handler.py:310-315`), so the on-disk 7-column `reduced_galaxy_catalogue.csv`
  has no flag column. Flag-2 exclusion therefore **cannot be re-proven from the CSV
  alone** — it relies on the CSV having been generated by this exact parse. The
  7-column schema is consistent with the current enum, so this is the most likely
  provenance but is an assumption. [CONFIDENCE: MEDIUM]

**Conclusion for §1:** The circularity concern is **real but already correctly
handled.** The railing is a **separate problem** — the *magnitude* of the photo-z
error (`σ_z ~ 0.035 ≈ 17× σ_GW`), not circularity.

---

## 2. The fork: POSTERIOR vs LIKELIHOOD — verdict & reasoning

### 2.1 Verdict: **LIKELIHOOD** [CONFIDENCE: HIGH on the practical conclusion; MEDIUM-HIGH on the strict classification]

For dark-siren use with GLADE+ flag-1 photo-z, treat `N(z; z_g, σ_z)` as a
**measurement likelihood**. To obtain a distribution over the *true* redshift,
multiply by the redshift prior `p_bg(z) ∝ dV_c/dz` and renormalise:

```
p_red(z) = N(z; z_g, σ_z) · p_bg(z) / Z_g ,   p_bg(z) ∝ (dV_c/dz)/(1+z),
Z_g = ∫ N(z; z_g, σ_z) · p_bg(z) dz
```

i.e. **the comoving-volume factor is applied exactly ONCE.** (Hitchhiker/Gair+2023
Eq. 16 prescription.)

### 2.2 Decisive reasoning

1. **The reported value is a measurement, not a posterior.** Dalya+2022 labels
   flag-1 a "measured photometric redshift" produced by an ANNz regression and
   **never states** that any redshift/comoving-volume prior was folded into `z_g`.
   ANNz outputs a regression point estimate + scatter — no `p(z)` prior applied.
   So a prior **must be supplied** to get a true-z distribution.
2. **The prior is NOT negligible in this regime.** GWcosmo's posterior shortcut is
   valid only where "the choice of prior becomes irrelevant," which (by their own
   footnote 10) holds only as `σ_z → 0` (spectroscopic). GLADE+ flag-1 has
   `σ_z/(1+z) ≈ 0.033` (`σ_z/z ~ 0.7` at z~0.05) — **firmly outside** that limit,
   where the prior **dominates** the posterior. The shortcut is invalid exactly here.
3. **The correct construction is published.** Gair+2023 Eq. (16) gives precisely
   `p_red = L·p_bg/Z` with `p_bg` uniform in comoving volume — dV_c applied once.

### 2.3 Literature convention (the two horns)

| Source | Interpretation | What they actually say | Validity |
|---|---|---|---|
| **GWcosmo** (Gray et al. 2023, arXiv:2308.02281), Eq. 2.9 + footnote 10 | **POSTERIOR** (explicit simplification) | "the assumption is made that the galaxy measurements … are posteriors." FN10: "If the measured redshift is actually a likelihood, a prior ought to be applied … a uniform in comoving volume prior might be sensible … In the limit σ̂_k → 0 … the choice of prior will become irrelevant." | Safe only for spec-z; the field's flagship code adopts it operationally because GWTC-3 weight sits with spec-z hosts. |
| **Hitchhiker** (Gair et al. 2023, arXiv:2212.08694), Eq. 16-17 + "Inconsistency 1" | **LIKELIHOOD** | `p_red(z\|ẑ) = L_red(ẑ\|z) p_bg(z) / ∫ L_red p_bg dz`, `p_bg` uniform in comoving volume (**dV_c once**). Sec 4.2: applying an extra dV_c weight to an already-dV_c-distributed sample makes events follow z⁴ not z² and "biases towards lower values of H0." | The physically correct foundation; required for large σ_z. |

**Reconciliation:** the two papers stake out the two horns; together they
prescribe **"dV_c counted exactly once."** For GLADE+ photo-z (large σ_z) adopt the
Hitchhiker likelihood treatment and avoid the Inconsistency-1 double-count at
re-injection. GWcosmo's posterior shortcut is inapplicable here **precisely
because σ_z is large.**

### 2.4 Where adversarial verification corrected / sharpened the claims

- **CONFIRMED, all six load-bearing claims** verified verbatim against the actual
  PDFs (Dalya+2022 col-35; GWcosmo Eq. 2.9 + FN10; Gair+2023 Eq. 16 + Inconsistency 1).
- **Sharpened:** ANN photo-z are not a *pristine* likelihood mean — they carry an
  implicit **training-set** n(z) prior. But that prior is **not** dV_c, so
  "apply dV_c once" is unaffected.
- **Sharpened:** "cosmology-independent" → photo-z assume **no cosmology**, but do
  inherit a training-set redshift prior. Keep the caveat; conclusion stands.
- **Attribution fix:** Hitchhiker's "Inconsistency 1" is specifically the
  **GW-event-to-galaxy assignment / re-injection** double-count (extra weight on an
  already-distributed sample), **not** "numerator dV_c applied twice." This
  *strengthens* the "count dV_c once across numerator + denominator + re-injection"
  rule.
- **Bias direction CONFIRMED** from the primary source (double-count → H0 low);
  the link to the project's specific **0.60 low rail** is a project-internal,
  directionally-consistent diagnostic — treat the **direction** as confirmed and the
  **specific-rail attribution** as plausible, not proven.

---

## 3. What our code ACTUALLY does (file:line) — and is it correct?

All pointers spot-verified live against the working tree this session.

### 3.1 Flags & columns — CORRECT

| Item | Code | Status |
|---|---|---|
| Keep flag `{1,3}`, drop `{0,2}` | `handler.py:297-300` | **Correct** (flag-2 circular class excluded) |
| Flag semantics documented | `handler.py:149-150`, `:284-287` | **Correct** (matches Dalya+2022) |
| Redshift = raw 0-based col 27 (`z_helio`) | `handler.py:146` | Reads heliocentric z — **see §3.3** |
| Redshift error = quad-sum of meas-err (col 31) + pec-vel-err (col 30, null→0.0015) | `handler.py:302-308` | Functionally ok; hardcoded 0.0015 uncited |
| Flag + pec-vel-err columns dropped from reduced CSV | `handler.py:310-315` | Correct (but blocks CSV re-audit, §1.3) |
| Per-host PDF = `norm(host_z, host_z_error)` | `bayesian_statistics.py:1623` | **Bare Gaussian — the fork lives here** |

### 3.2 The net dV_c count — the crux

Verified the assembled ratio `p_i = (β_G·L_cat + B_num) / D(h)`
(`bayesian_statistics.py:1503`):

| Term | dV_c present? | Code |
|---|---|---|
| In-cat numerator `N_g = ∫ p_GW(z)·N(z;z_g,σ_z) dz` | **NO** (bare Gaussian × GW MVN) | `:1623`, `:1646-1647` |
| Global selection denom `D_g = w_g·P_det(z_g)` | **NO** (evaluated at point z_g, no smearing) | `:472-490` |
| Population denom `D(h) = ∫ P_det/(1+z) dV_c dz` | **YES** (1×) | `:248-260` |
| `β_Gbar = ∫ (1-f) P_det/(1+z) dV_c dz` | **YES** (1×) | `:352-372` |
| Out-of-cat completion `B_num = ∫ (1-f) p_GW/(1+z) dV_c dz` | **YES** (1×, explicit true-z prior) | `:1462-1480` |

In `β_G·L_cat/D(h)` the `β_G` dV_c **cancels** the `D(h)` dV_c, so the host's
`N(z;z_g,σ_z)` carries **net 0 dV_c**. The dark branch `B_num/D(h)` likewise
cancels, but `B_num` carries an **explicit dV_c** multiplying `p_GW(z)` as a genuine
true-z prior (**net 1**).

**This asymmetry IS the GWcosmo footnote-10 / Hitchhiker Inconsistency-1 fork made
concrete in our code:**

- In-cat host redshift term: **net dV_c = 0** → an implicit **POSTERIOR** reading.
- Dark / out-of-cat host true-z prior: **net dV_c = 1** → a **LIKELIHOOD×volume** reading.
- **The two host branches use different redshift-prior conventions for the same
  physical quantity.** This is internally inconsistent.

### 3.3 Is the current usage correct as-is? **NO — two defects.**

1. **dV_c fork (primary, the Hint-4 target):** Given the §2 verdict (LIKELIHOOD),
   the in-cat numerator is **MISSING one dV_c.** It should become
   `∝ N(z;z_g,σ_z)·(dV_c/dz)/(1+z)` (the regularised `p_red = N·p_bg/Z_g`) to match
   `B_num`. Current state = posterior-no-dV_c = correct *only* if the catalogue value
   were a posterior, which the literature says it is not in this σ_z regime.
2. **Numerator/denominator photo-z asymmetry (secondary):** the in-cat numerator
   marginalises the host photo-z via a Gaussian integral (`:1641-1646`), but the
   global selection denominator uses `P_det` evaluated at the single point `z_g`
   with **no** Gaussian smearing (`:472-490`). The same σ_z is treated
   inconsistently between numerator and denominator — a second consistency defect
   that the dV_c fix alone does **not** resolve.

> **Cross-check with the project's negative result:** the bridge investigation
> showed a *numerator-only* fix does not de-rail. That is consistent with §3.2-3.3:
> the dV_c-once rule and the σ_z smearing must be applied **consistently across
> numerator, selection denominator, and re-injection** (ensemble coherence), not
> just the numerator.

---

## 4. Concrete implication for the Hint-4 form

**Hint-4 = ADD a dV_c** (net 0 → 1) to the in-catalogue host numerator.

- Replace the bare Gaussian host PDF
  `N(z; z_g, σ_z)` (`bayesian_statistics.py:1623/1646`) with the **regularised
  posterior**
  `p_red(z) = N(z; z_g, σ_z) · (dV_c/dz)/(1+z) / Z_g`,
  `Z_g = ∫ N·(dV_c/dz)/(1+z) dz`.
- This makes the in-cat host's redshift prior **match `B_num`** (both carry exactly
  one dV_c), removing the branch asymmetry of §3.2.
- **Direction of effect:** adding the rising `dV_c` up-weights higher z, which
  pulls H0 **off the low (0.60) rail** — directionally consistent with the
  Hitchhiker result that a double-counted (or here, a *missing*-then-restored) dV_c
  biases H0 low.
- **NOT** "remove a dV_c": removal would correspond to adopting the POSTERIOR
  reading, which the σ_z >> σ_GW regime invalidates.
- **NOT** "current treatment already correct": the in-cat branch is internally
  inconsistent with the dark branch and uses the wrong (posterior) reading for
  large-σ_z photo-z.

**Critical caveat:** Hint-4 (dV_c once in the numerator) is the **likelihood-side
ingredient only.** The rigorous negative result requires the dV_c-once rule be
enforced **consistently across numerator + selection denominator + re-injection.**
Hint-4 is necessary but **not sufficient** alone.

---

## 5. Recommended setup for the option-1 bridge prototype

1. **Interpretation = LIKELIHOOD.** Treat `N(z; z_g, σ_z)` as a measurement; build
   `p_red(z) = N · p_bg / Z_g` with `p_bg ∝ (dV_c/dz)/(1+z)`, dV_c applied **once**.
2. **Apply dV_c consistently** across (a) in-cat numerator, (b) selection
   denominator, (c) re-injection / host-draw. Do **not** add a *second* dV_c at
   injection (that is Hitchhiker Inconsistency 1 → H0 low).
3. **Symmetrise photo-z marginalisation:** smear `P_det` over the same host
   `N(z;z_g,σ_z)` in the selection denominator (`:472-490`) that the numerator uses
   (`:1641-1646`), so σ_z is handled identically on both sides.
4. **Keep flags `{1,3}`, keep excluding flag 2** — no change; this is already
   correct and non-circular.
5. **Handle the z<0 truncation explicitly:** with σ_z ≈ 0.033 and z~0.002, the bare
   Gaussian has large mass at z<0; the `p_bg ∝ dV_c` factor naturally suppresses
   z→0 and removes most of the unphysical mass — make this an intended feature of
   `p_red`, not an ad-hoc rejection (`handler.py:127-135`).
6. **Validate by railing direction:** success criterion = the 0.60 low rail is
   cancelled and the posterior recovers the injected H0 = 0.73 (not the 0.87 high
   rail of the over-cleaned numerator). The truth sitting *between* the two rails is
   the signature that the normalisation/comoving-volume counting is the lever.

---

## 6. Other data-usage issues & open questions

### 6.1 Other data-usage issues (from the code audit)

- **FRAME (potential systematic):** the pipeline reads **heliocentric** z (raw col
  27), not CMB-frame `z_cmb` (raw col 28). **No peculiar-velocity correction** is
  applied to the redshift *value* — only the pec-vel *error* (null→0.0015) is added
  in quadrature (`handler.py:302-308`). This leaves an **uncorrected coherent
  offset** in `cz = H0·d_L`, a systematic (not just noise) for an H0 measurement.
  Helio/CMB determination is MEDIUM-HIGH (physics-based dipole inference for
  NGC4736; GLADE+ readme not fetchable in-session — confirm against Dalya+2022
  before finalising).
- **Num/denom σ_z asymmetry** (§3.3 defect 2) — `bayesian_statistics.py:1641-1646`
  vs `:472-490`.
- **σ_z is a near-constant ~0.033 absolute floor** (sampled flag-1 rows: z~0.002,
  z_err~0.0331, σ_z/z ~ 16); ~12.7% of first 200k rows have z_err>0.02. Drives the
  z<0 mass / truncation issue.
- **Hardcoded pec-vel-error default 0.0015** (`handler.py:302`) — no citation
  (relates to Known Bug #9).
- **Stellar-mass → BH-mass relation** (`α=7.45·ln10, β=1.05`; `handler.py:30-33`,
  `:1033`) applied to all hosts; strong model assumption driving the mass-channel
  selection weight (not redshift-related but affects host weighting).
- **No k-correction** on apparent B-mag (raw col 10) used for per-pixel
  completeness `m_th`; minor at low z.
- **Wide host windows:** pruning uses `z ± z_error` (`handler.py:230-234`,
  `:430-438`); with σ_z~0.033 these admit many spurious candidate hosts.

### 6.2 Open questions

1. **CMB vs heliocentric frame:** confirm against Dalya+2022 readme which redshift
   column is appropriate, and whether a peculiar-velocity / solar-motion correction
   to the *value* (not just the error) should be added. Currently uncorrected →
   coherent H0 bias.
2. **Reduced-CSV provenance:** flag-2 exclusion cannot be re-proven from the
   on-disk CSV (flag column discarded). Should the parse retain the flag column (or
   write a provenance hash) so the exclusion is auditable?
3. **Strict classification certainty:** GLADE+ never explicitly labels its
   redshifts likelihood-vs-posterior; the LIKELIHOOD verdict is a strong inference
   from how ANNz works + absence of any prior statement, not a verbatim catalogue
   claim (MEDIUM-HIGH). Does the implicit ANN training-set n(z) prior need any
   correction beyond the dV_c term? (Believed negligible — it is not a dV_c prior.)
4. **Sufficiency of Hint-4 alone:** confirm in the bridge prototype that the
   dV_c-once fix **plus** denominator smearing **plus** re-injection consistency
   together de-rail, since the numerator-only fix is known to fail.
5. **Specific-rail attribution:** the 0.60 low rail ↔ uncounted comoving-volume
   prior link is project-internal; verify quantitatively that restoring the
   numerator dV_c moves the rail by the predicted amount.

---

## Sources

**Primary literature**
- Dalya et al. 2022, GLADE+, arXiv:2110.06184 (MNRAS) — col-35 flag definitions
  (1=photometric z→d_L, 2=d_L→z, 3=spectroscopic z→d_L); photo-z from ANN, bands
  B_J,R_F,W1,W2, σ_z/(1+z)~0.033.
- Bilicki et al. 2016, WISE×SCOSPZ, ApJS 225, 5 — ANNz empirical NN trained on
  GAMA-II spec-z.
- Gray/Gair et al. 2023, GWcosmo LOS redshift prior, arXiv:2308.02281 (JCAP) —
  Eq. 2.9 + footnote 10 (POSTERIOR assumption; prior irrelevant only as σ→0).
- Gair et al. 2023, "Hitchhiker's Guide to the Galaxy Catalog Approach", arXiv:2212.08694 —
  Eq. 16-17 regularised posterior (dV_c once); "Inconsistency 1" double-count → H0 low.

**Project source (spot-verified live 2026-06-30)**
- `master_thesis_code/galaxy_catalogue/handler.py:146-152, 284-300, 302-315, 330-349, 66-67, 127-135`
- `master_thesis_code/bayesian_inference/bayesian_statistics.py:1623, 1641-1647, 248-260, 352-372, 472-490, 1462-1480, 1503`
- Project memory: `h0-railing-rootcause-photoz` (0.60 low rail / 0.87 high rail / 0.73 truth between).
- Companion: `.planning/derivation-photoz-incatalog/COMPARISON.md`, `GAP-ANALYSIS.md`.
