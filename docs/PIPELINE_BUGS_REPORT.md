# Pipeline Bugs & Errors Report — Photo-z / Frame Investigation (June 2026)

**Date:** 2026-06-30
**Branch:** `physics/photoz-joint-normalisation`
**Scope:** Bugs and errors in the LISA EMRI dark-siren H₀ inference pipeline uncovered while
investigating the H₀ posterior *railing* (the seed-600 posterior collapsing to the grid edge,
H₀/100 → 0.86, +18%). Companion research artifacts:
`.planning/derivation-photoz-incatalog/` and `scripts/bridge_closure/`.

---

## Summary

The investigation that began as "why does H₀ rail to 0.86?" surfaced one **data-usage root cause**
and a cluster of related **redshift-handling defects** in the galaxy-catalogue → likelihood path.
The root cause is that the GLADE+ host catalogue is dominated by **photometric** redshifts
(flag 1, ~62% of rows) whose error σ_z ≈ 0.035 is ~17× the EMRI GW redshift precision; convolving
this broad photo-z PDF inside the in-catalogue likelihood washes out the sharp GW distance, so the
posterior tracks the catalogue redshift-density gradient instead of the data and rails. A bridge
study isolated this ingredient (every other ingredient recovers the truth; only the photo-z
convolution and an actively-broken host↔sky shuffle rail), and a full normalisation search plus a
hierarchical candidate (`D_sm`) showed the channel is **information-starved**: no normalisation
recovers a peaked H₀ at this σ_z/z. Alongside the root cause we found and quantified a genuine
**heliocentric-vs-CMB frame** systematic (fixed this session, `[PHYSICS]` commit `7021f6f`,
issue #15, PR #17), an uncorrected **host peculiar-velocity value** (filed as issue #16), and two
**internal-consistency defects** in the dV_c (comoving-volume) bookkeeping and the
numerator/denominator photo-z smearing. Two pre-existing `CLAUDE.md` "Known Bugs" (#9 redshift-error
scaling, and the uncited hardcoded pec-vel-error default) were touched and are re-flagged here.

### Severity table

| # | Bug | File:line | Severity | Status |
|---|-----|-----------|----------|--------|
| 1 | GLADE flag-1 = **photometric** redshifts (σ_z ≈ 0.035) → in-catalogue railing (root cause) | `handler.py:297-300`, `:149-150` | **CRITICAL** (paper-blocker) | Root-caused; channel **information-starved**; recommend spec-z forecast arm + characterized limitation |
| 2 | **Heliocentric** redshift fed to `d_L(z;H₀)` instead of CMB-frame (solar dipole uncorrected) | `handler.py:146`, `:66` | LOW–MODERATE (coherent, citable) | **FIXED** `[PHYSICS]` `7021f6f`, issue #15, PR #17 |
| 3 | Host **peculiar velocity** never value-corrected (only error inflated) | `handler.py:302-307` | LOW–MODERATE (≤~1.3% residual bulk flow) | **OPEN**, issue #16 (`physics`) |
| 4 | **dV_c branch inconsistency**: in-cat numerator net-0 dV_c vs dark branch net-1 | `bayesian_statistics.py:1623/1646`, `:1462-1480`, `:1503` | HIGH (internal inconsistency) | Characterized; numerator-only fix **insufficient** |
| 5 | **Numerator/denominator photo-z smearing asymmetry** (num convolves σ_z, denom does not) | `bayesian_statistics.py:1641-1646` vs `:472-490` | HIGH (consistency defect; railing mechanism) | Characterized; `D_sm` candidate de-biases but does not recover a peak |
| — | Pre-existing #9: redshift-error scaling `0.013·(1+z)³` (Hitchhiker form) | `datamodels/galaxy.py:64` | LOW | Pre-existing; re-flagged |
| — | Uncited hardcoded pec-vel-error default `0.0015` | `handler.py:302` | LOW | Pre-existing; re-flagged (relates to #9) |

All railing magnitudes below are quoted as H₀/100 on the inference grid; "truth" = injected
H₀/100 = 0.73.

---

## Bug 1 — GLADE flag-1 redshifts are PHOTOMETRIC (root cause of the railing)

### Motivation
The seed-600 H₀ posterior rails to the upper grid edge (MAP 0.86, +0.13 ≈ +18%) and is
sky-position-*independent* (it reproduces in a no-sky closure), ruling out the measurement side,
the n(z) shape, the sky/Fisher covariance, the completion term, the survival p_det, and the
candidate-selection radius. A "bridge" study added one real-pipeline ingredient at a time to the
closure (which recovers 0.73 with spectroscopic σ_z) until the railing appeared.

### What we found
The decisive ingredient is the **host-redshift photo-z convolution**. The parser keeps GLADE+
measurement flags `{1, 3}`:

```
master_thesis_code/galaxy_catalogue/handler.py:297-300
    chunk = chunk[
        (chunk[CatalogueColumns.REDSHIFT_FLAG.name] == 1)
        | (chunk[CatalogueColumns.REDSHIFT_FLAG.name] == 3)
    ]
```

The GLADE+ flag semantics (Dálya et al. 2022, arXiv:2110.06184, raw col 35; documented at
`handler.py:149-150`) are **1 = photometric, 2 = luminosity-distance, 3 = spectroscopic**. The
column *indices are all correct* and the flag-2 (cosmology-circular `d_L→z`) class is *correctly
excluded* — this is **not a parse/index bug**. It is a **data-usage finding**: flag-1 photometric
hosts dominate the retained catalogue (~62% of rows) and carry σ_z ≈ 0.035, ~17× the EMRI GW
redshift precision σ_z^GW ≈ 0.037·z ≈ 0.002 at z ≈ 0.05 (σ_z/z ≈ 0.7). The in-catalogue likelihood
marginalises this broad PDF, `N_g = ∫ p_GW(d_L(z,h))·𝒩(z; z_g, σ_z) dz`, so the candidate sum
tracks the catalogue redshift-density over the σ_z window rather than the sharp GW distance, driving
H₀ to the grid edge.

### Result
**Root cause confirmed and reproduced.** A full normalisation search (every numerator-only kernel)
and a hierarchical global photo-z-smeared selection candidate `D_sm` were tested. The conclusion is
that the in-catalogue photometric channel at GLADE's regime is **information-starved**: `D_sm`
de-biases the global density gradient and passes the σ_z→0 gate (~0.74), but multi-seed it is
bimodal/scattered (std ~0.1) and the scatter does **not** shrink with event count; per-seed
posteriors (n_ev = 2000) peak at 0.64/0.64/0.69/0.87 — **never 0.73**. No tested normalisation
recovers a peaked H₀. **Recommendation:** pivot the H₀ headline to the spectroscopic forecast arm
(self-consistent spec-z hosts recover h ≈ 0.725) plus a rigorously-characterized GLADE-photometric
limitation. A warning caveat is already documented in-code at `handler.py:288-296`.

### Displaying the result

σ_z split by flag (sampled GLADE+ rows):

| flag | meaning | σ_z median | σ_z 90th pct |
|------|---------|-----------|--------------|
| 1 | photometric | **0.0346** | 0.0482 |
| 3 | spectroscopic | **0.0017** | 0.0036 |

Bridge "rung" ladder — swap one ingredient synthetic → real; only photo-z (and an actively-wrong
sky shuffle) rail:

| Rung | Ingredient | MAP | bias | rails? |
|------|-----------|-----|------|--------|
| R0 | synthetic baseline | 0.735 | +0.005 | no |
| A | real σ_dL distribution (N=3361) | 0.729 | −0.001 | no |
| B | real GLADE n(z) shape | 0.734 | +0.004 | no |
| C-real | real catalogue + sky + 3-D MVN | 0.725 | −0.005 | no |
| C-iso | **host sky positions shuffled** | 0.855 | +0.125 | **yes** |
| D | + real pixelated f_k + B_num | 0.735 | +0.005 | no |
| E | + real survival p_det | 0.725 | −0.005 | no |
| F | fully faithful (delta-z) | 0.735 | +0.005 | no |
| **G** | **+ host-z convolution at real σ_z = 0.035** | **0.857** | **+0.127** | **yes** |

σ_z sweep at fully-faithful Rung G (bias is a sensitive, sign-changing function of σ_z):

| σ_z | MAP | bias |
|-----|-----|------|
| 0 (delta-z) | 0.725 | −0.005 |
| ≈0.002 (spec-z) | 0.600 | −0.130 |
| ≈0.009 | 0.870 | +0.140 |
| ≈0.018 | 0.870 | +0.140 |
| **≈0.035 (real GLADE)** | **0.857** | **+0.127** |

Hierarchical `D_sm` candidate (clean rung_I closure, truth 0.73):

| Test | Result |
|------|--------|
| Gate σ_z = 0.002, multi-seed | median ~0.73 — PASS |
| Single-seed σ_z = 0.035 | 0.693 interior — a favourable draw, not a win |
| Multi-seed σ_z = 0.035, n_ev = 250 | 6 interior / 4 rail-up (0.87) / 2 rail-down (0.60); std 0.11 |
| n_ev = 2000 | std 0.097 — did **not** shrink |
| Per-seed posterior peaks (n_ev = 2000) | 0.64, 0.64, 0.69, 0.87 — never 0.73; multimodal |

Refs: `scripts/bridge_closure/BRIDGE-FINDINGS.md`,
`.planning/derivation-photoz-incatalog/{NORMALISATION-FIX.md, INCREMENT3-DSM-VERDICT.md}`,
commit `ee98f71`; figures in `scripts/bridge_closure/outputs/rung{A..I}*.pdf`.

---

## Bug 2 — Heliocentric redshift fed to `d_L(z; H₀)` instead of CMB-frame

### Motivation
`cz = H₀·d_L` at low z, so any coherent redshift offset maps directly to an H₀ offset
(δH₀/H₀ = δz/z). The solar-motion dipole (v_sun ≈ 369.8 km/s) injects exactly such an offset if the
heliocentric redshift is used without frame correction — a *coherent, direction-dependent* bias that
a referee will expect to see controlled in any GW H₀ analysis.

### What we found
The parser reads GLADE+ raw 0-based **column 27 = z_helio** (heliocentric) and assigns it directly
to the host redshift that flows into the luminosity-distance relation:

```
master_thesis_code/galaxy_catalogue/handler.py:146   REDSHIFT = 27   # z_helio
master_thesis_code/galaxy_catalogue/handler.py:66    HostGalaxy.z  <- REDSHIFT
```

The adjacent **CMB-frame** redshift (raw 0-based **column 28 = z_cmb**, already populated by GLADE+)
is never read, and no frame/value correction is applied. The solar dipole is left entirely in
`cz = H₀·d_L`.

### Result
**FIXED** this session. The one-line value fix reads z_cmb (0-based col 28) instead of z_helio
(col 27), routed through `/physics-change` and committed `[PHYSICS]` `7021f6f`. Tracked as GitHub
issue #15; PR #17 merges the frame fix to `main`. This systematic is **orthogonal to the railing**
(see comparison below) — fixing it does not touch the railing, and it was confirmed not to cause it.

### Displaying the result

| Quantity | Value |
|----------|-------|
| v_sun / c | 1.2335×10⁻³ |
| Per-event envelope at z = 0.05 (apex/anti-apex) | ±2.47% |
| Net ensemble bias (detected-host sample, N = 3375, rigorous per-event mean) | **+0.151%** |
| Net ensemble bias (simplified `β·⟨cos⟩/z_typ`, z_typ = 0.046) | +0.065% |
| Full-catalogue cross-check bound (200k sample, FK5→Galactic) | +0.33% |

Frame systematic vs railing — orthogonal degrees of freedom:

| Property | Frame systematic | H₀ railing |
|----------|------------------|------------|
| Magnitude | net +0.15% (per-event 2.47%) | ~+18% |
| Sky dependence | direction-dependent (∝ cos θ_apex) | sky-INDEPENDENT |
| Reproduces in no-sky closure? | No | **Yes** |
| Mechanism | uncorrected solar dipole in cz | normalisation / prior domination |

Net frame bias is ~120× smaller than the railing (18 / 0.151 ≈ 119).
Ref: `.planning/derivation-photoz-incatalog/FRAME-SYSTEMATIC.md`, commit `c42f558` (verification),
`7021f6f` (fix).

> Execution-time caveat recorded at fix time: confirm z_cmb's positional index survived the catalogue
> reduction step in `reduced_galaxy_catalogue.csv` before relying on col 28 (the field *identity* is
> certain; the reduced-file *position* was the one item to verify).

---

## Bug 3 — Host peculiar velocity never value-corrected (only error inflated)

### Motivation
Removing the solar dipole (Bug 2) leaves the host galaxy's *own* peculiar velocity in z_cmb. A
coherent large-scale bulk flow (~150–250 km/s) leaves a residual direction-dependent H₀ bias if the
detected-host distribution is anisotropic; the random component is noise that should be marginalised,
not ignored. Error inflation is **not** a substitute for a value correction of a coherent offset.

### What we found
The code applies only a partial version of an error-marginalisation: the GLADE+ PV-correction *error*
column (raw 0-based col 30, null → 0.0015) is added in quadrature into the redshift error, but the
redshift **value** is never PV-corrected, and the column is treated as a generic error term rather
than the uncertainty of an actually-applied correction:

```
master_thesis_code/galaxy_catalogue/handler.py:302-307
    chunk = chunk.fillna({REDSHIFT_PECULIAR_VELOCITY_ERROR.name: 0.0015})
    REDSHIFT_MEASUREMENT_ERROR = sqrt(meas_err**2 + pec_vel_err**2)
```

### Result
**OPEN** — filed as GitHub issue #16 (`physics`), lower priority than the frame fix. Recommended
two-step convention: (1) frame correction via z_cmb (Bug 2, done), then (2) treat the host PV either
by value-correction with a reconstructed PV field (linear theory / 2M++; GLADE+ provides a
PV-corrected redshift flagged at raw col 29 with its error in col 30) **or** by marginalisation as
an added uncertainty (σ_v ~ 150–500 km/s, commonly ~200).

### Displaying the result

| Term | Magnitude |
|------|-----------|
| Residual bulk flow (~200 km/s), if host distribution anisotropic | up to ~1.3% direction-dependent |
| Random PV scatter σ_v ~ 300 km/s → σ_z = 1.0×10⁻³ → per-event at z = 0.05 | 2.00% |
| …averaged over detected ensemble (1/√N, N = 3375) | 0.034% |

**Key physics point:** the solar-motion frame offset is a coherent, direction-dependent bias, not
zero-mean noise — inflating the error neither removes nor accounts for it.
Ref: `.planning/derivation-photoz-incatalog/FRAME-SYSTEMATIC.md §3`.

---

## Bug 4 — dV_c branch inconsistency (in-catalogue numerator net-0 vs dark branch net-1)

### Motivation
The comoving-volume factor `dV_c/dz` is the redshift prior on a host's true z and must be counted
*exactly once* across numerator, selection denominator, and re-injection (Gair et al. 2023,
"Hitchhiker's Guide", arXiv:2212.08694, Eq. 16; "Inconsistency 1"). A double-count makes events
follow z⁴ instead of z² and biases H₀ low — the exact direction of the standard pipeline's low rail.

### What we found
The two host branches use **different redshift-prior conventions for the same physical quantity**.
Tracing the assembled per-event ratio `p_i = (β_G·L_cat + B_num) / D(h)`
(`bayesian_statistics.py:1503`):

| Term | dV_c present? | Code |
|------|---------------|------|
| In-cat numerator `N_g = ∫ p_GW(z)·𝒩(z;z_g,σ_z) dz` | **NO** (bare Gaussian × GW MVN) | `:1623`, `:1646-1647` |
| Global selection denom `D_g = w_g·P_det(z_g)` | **NO** (point eval, no smearing) | `:472-490` |
| Population denom `D(h) = ∫ P_det/(1+z) dV_c dz` | YES (1×) | `:248-260` |
| `β_Ḡ = ∫ (1−f) P_det/(1+z) dV_c dz` | YES (1×) | `:352-372` |
| Out-of-cat completion `B_num = ∫ (1−f) p_GW/(1+z) dV_c dz` | YES (1×, explicit true-z prior) | `:1462-1480` |

In `β_G·L_cat/D(h)` the β_G dV_c cancels the D(h) dV_c, so the host's `𝒩(z;z_g,σ_z)` carries **net
0 dV_c** (an implicit POSTERIOR reading). The dark branch `B_num/D(h)` cancels too, but B_num carries
an **explicit dV_c** multiplying p_GW(z) as a genuine true-z prior (**net 1**, a LIKELIHOOD×volume
reading). This is the GWcosmo footnote-10 / Hitchhiker Inconsistency-1 fork made concrete and
internally inconsistent.

### Result
Characterized. The literature verdict for GLADE flag-1 photo-z (large σ_z) is **LIKELIHOOD**: the
in-cat numerator is *missing* one dV_c and should become the regularised posterior
`p_red(z) = 𝒩(z;z_g,σ_z)·(dV_c/dz)/(1+z) / Z_g` to match B_num. **However**, this is necessary but
**not sufficient**: a numerator-only fix does not de-rail (it removes the low rail but overshoots to
the high rail). dV_c-once must be enforced consistently across numerator **and** selection denominator
**and** re-injection.

### Displaying the result

| Construction | numerator effective prior | σ_z = 0.002 | σ_z = 0.035 | verdict |
|---|---|---|---|---|
| STANDARD (current code) | doubly-smeared dV_c | 0.7438 peaked | **0.6000** rail DOWN | effective double-count → low |
| Angle A/C (per-galaxy posterior `𝒩·p_bg/Z_g`) | clean dV_c | 0.7478 peaked | **0.8700** rail UP | disqualified |
| Angle B (global de-count) | clean dV_c | 0.7439 peaked | **0.8700** rail UP | disqualified |
| Local same-kernel (consistent denom) | any | 0.8700 rail | 0.8700 rail | gate FAIL |

Truth (0.73) sits strictly **between** the two rails: no pure numerator-kernel choice lands on it.
Ref: `.planning/derivation-photoz-incatalog/{CATALOG-INTERPRETATION.md §3, NORMALISATION-FIX.md}`,
commit `bd66f5b`.

---

## Bug 5 — Numerator/denominator photo-z smearing asymmetry

### Motivation
For an unbiased catalogue dark-siren likelihood the same host redshift kernel must appear in the
numerator (the host's contribution) and in the selection denominator (the normalisation). Treating
σ_z differently on the two sides introduces a density-gradient bias that is negligible for
spectroscopic hosts but dominant for photo-z.

### What we found
The in-cat numerator marginalises the host photo-z via a Gaussian integral, but the global selection
denominator evaluates `P_det` at the single point z_g with **no** smearing:

```
numerator  : bayesian_statistics.py:1641-1646   ∫ p_GW · 𝒩(z;z_g,σ_z) dz
denominator: bayesian_statistics.py:472-490      D_g ≈ w_g · P_det(z_g)   (no convolution)
```

The same σ_z is handled inconsistently between numerator and denominator. This asymmetry is the
mechanism by which the photo-z error propagates into the railing: the numerator tracks the smoothed
catalogue density n_smooth while the denominator uses the un-smoothed n(z).

### Result
Characterized. The symmetrised candidate — a **global photo-z-smeared selection** that puts the same
kernel in the denominator, `D_sm(h) = Σ_g w_g ∫ p_det^GW(d_L(z,h))·p_red(z|z_cat_g) dz` — was derived,
gate-proven (σ_z→0 → standard), and prototyped. It **de-biases the global density gradient but does
not recover a peaked H₀** (see Bug 1's `D_sm` table). The deeper obstruction: with `p_det ≈ 1` across
the in-catalogue range there is *no local selection gradient*, so a single global per-h scalar cannot
track the local numerator gradient at z*(h), and with σ_z ≫ σ_z^GW there is no per-event localisation
to track. This is the structural face of the information-starvation conclusion in Bug 1.

### Displaying the result
The `D_sm` lever is real and deterministic — `d/dh log(D_sm/D)@0.73 ≈ +0.19` (stable already at
n_gal = 12k, → ±0.007 at 400k, D_sm/D ≈ 0.920) — and it cancels the standard low rail, but the
residual discrete-catalogue + photo-z noise yields a flat/multimodal posterior with no peak. The
apparent `E[h] ≈ 0.735` is an artifact (the grid midpoint of [0.60, 0.87] is 0.735); the *shape*
(flat/multimodal) is the real readout.
Ref: `.planning/derivation-photoz-incatalog/{DERIVATION-HIERARCHICAL.md, INCREMENT3-DSM-VERDICT.md}`,
commits `415500b`, `5ef8c6e`, `a8cbab0`.

---

## Pre-existing `CLAUDE.md` "Known Bugs" touched by this work

- **Known Bug #9 — non-standard redshift-error scaling.** `datamodels/galaxy.py:64` uses
  `0.013·(1+z)³` for the galaxy redshift uncertainty (the "Hitchhiker" form), which has no standard
  reference; conventional forms scale as `(1+z)`. Not the railing driver (the railing is set by the
  GLADE per-host σ_z, Bug 1), but it remains an uncited modelling choice on the redshift error and
  should be reconciled with whatever σ_z convention the spec-z forecast arm adopts. Severity LOW.

- **Uncited hardcoded pec-vel-error default `0.0015`.** `handler.py:302`
  (`fillna({REDSHIFT_PECULIAR_VELOCITY_ERROR: 0.0015})`) injects an uncited constant into the
  redshift-error quadrature for rows with a null PV-error. Related to #9 (both are unreferenced
  redshift-error inputs) and to Bug 3 (the same column whose *value* correction is missing).
  Severity LOW; cite or replace when the PV treatment (issue #16) is implemented.

---

## Cross-reference index

| Artifact | Path |
|----------|------|
| Bridge findings (rung ladder + root cause) | `scripts/bridge_closure/BRIDGE-FINDINGS.md` |
| Bridge figures + JSON | `scripts/bridge_closure/outputs/rung{A..I}*.{pdf,json}` |
| Method-by-method literature comparison | `.planning/derivation-photoz-incatalog/COMPARISON.md` |
| Ours-vs-theirs gap analysis | `.planning/derivation-photoz-incatalog/GAP-ANALYSIS.md` |
| Likelihood-vs-posterior / dV_c bookkeeping | `.planning/derivation-photoz-incatalog/CATALOG-INTERPRETATION.md` |
| Frame systematic (quantified) | `.planning/derivation-photoz-incatalog/FRAME-SYSTEMATIC.md` |
| Hierarchical / `D_sm` derivation | `.planning/derivation-photoz-incatalog/DERIVATION-HIERARCHICAL.md` |
| `D_sm` verdict + canonical numbers | `.planning/derivation-photoz-incatalog/INCREMENT3-DSM-VERDICT.md` |
| Numerator-only negative result | `.planning/derivation-photoz-incatalog/NORMALISATION-FIX.md` |
| Cumulative bias catalog (phases 9–48) | `docs/H0_BIAS_RESOLUTION.md` |

| Commit | Subject |
|--------|---------|
| `ee98f71` | root-cause the H₀ railing to photometric host redshift error |
| `0bd1f73` | photo-z method comparison + gap analysis |
| `bd66f5b` | catalog data-usage check + likelihood-vs-posterior verdict |
| `c42f558` | verify + quantify heliocentric-vs-CMB frame systematic |
| `415500b` | derive hierarchical / global-smeared-selection candidate |
| `5ef8c6e` | prototype global photo-z-smeared selection `D_sm` |
| `a8cbab0` | increment-3 verdict — `D_sm` de-biases but channel info-starved |
| `7021f6f` | `[PHYSICS]` heliocentric → CMB-frame redshift fix |

| GitHub | Item |
|--------|------|
| issue #15 | Heliocentric-vs-CMB frame fix (Bug 2) |
| issue #16 | Host peculiar-velocity value correction (Bug 3) |
| PR #17 | Frame fix → `main` |
