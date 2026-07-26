# GLADE+ Peculiar-Velocity Double-Count Audit

**Date:** 2026-07-26
**Question:** Does the pipeline double-count peculiar-velocity (PV) uncertainty in the
host-redshift kernel, and what residual σ_v is scientifically defensible?
**Verdict (short):** **YES — a structured, partial double-count exists**, in two distinct
forms depending on whether the host is BORG-PV-corrected. It is numerically negligible for
photo-z hosts (0.02% kernel-width inflation) and moderate (~5–10% width inflation) for
spec-z hosts. No H₀ *bias* is implied in mock mode (generator draws hosts at catalogue z
verbatim; PV terms are pure width terms there). The issue is real-data-mode correctness.

All claims below are tagged **[VERIFIED: source]** or **[INFERRED]**. No code was modified.

---

## 1. Layer 1 — What GLADE+ actually reports (Dálya et al. 2022, arXiv:2110.06184)

Read via ar5iv full text (2026-07-26).

### 1.1 Column semantics (paper Table; 1-based paper numbering vs repo 0-based)

| Paper col (1-based) | Repo 0-based | Content (paper wording) |
|---|---|---|
| 28 | 27 | Heliocentric redshift z_helio |
| 29 | **28** | CMB-frame redshift z_cmb — the repo's `REDSHIFT` |
| 30 | 29 | **PV-correction flag**: "0 if the CMB frame redshift and luminosity distance values given in columns 29 and 33 are not corrected for the peculiar velocity, 1 if they are corrected values" — **NOT parsed by the repo** |
| 31 | **30** | "Error of redshift from peculiar velocity estimation" — the repo's `REDSHIFT_PECULIAR_VELOCITY_ERROR` |
| 32 | **31** | "Measurement error of heliocentric redshift" — the repo's `REDSHIFT_MEASUREMENT_ERROR` |
| 35 | **34** | Redshift/distance flag: 0=none, 1=photometric z, 2=luminosity distance, 3=spectroscopic z — the repo's `REDSHIFT_FLAG` |

**[VERIFIED: paper column table via ar5iv; repo mapping at `handler.py:143-166` matches
exactly under the 1-based→0-based shift.]**

Key semantic facts:

- **(a) PV corrections** are applied to z_cmb (col 29) for galaxies **cross-matched with
  2M++, z ≤ 0.05, with B-band magnitude** (paper §2.2), using the BORG forward model
  (256³ grid, 2.64 Mpc/h resolution — linear + partially non-linear velocity field).
  Correction status is recorded in the flag col 30 (1-based). **[VERIFIED: §2.2 quotes]**
- **(b) Error columns are separate**: col 31 (1-based) carries the PV *uncertainty*
  in redshift units; col 32 (1-based) carries the *measurement* error of z_helio only.
  The catalogue does **not** pre-combine them — combination is left to the user.
  **[VERIFIED: column table wording]** That the repo must (and does) combine them itself
  is therefore correct in principle.
- **(c) Residual dispersion after BORG:** the paper defines the total PV variance as
  **σ_tot² = σ_borg² + σ_vir²** (§2.2), where σ_vir is the *non-linear virial* term from
  their Eq. (1): σ_vir = 476 g_v (Δ_nl(z)E(z)²)^{1/6} (M_h/10¹⁵ M_⊙ h⁻¹)^{1/3}, per-galaxy
  via a halo-mass–luminosity estimate. **No single km/s residual value is quoted.**
  **[VERIFIED: §2.2 via ar5iv]**
  ⇒ **Critical consequence:** for PV-corrected galaxies, the catalogue's col-31 PV error
  is *designed to already include the non-linear (virial) residual*. It is σ_tot, not just
  the linear-reconstruction (BORG posterior) error.
- **(d) Flags:** flag=1 photometric (σ_z/(1+z) ≈ 0.033 for WISE×SCOSPZ, paper §2),
  flag=3 spectroscopic (no explicit σ value quoted in the paper for spec-z).
  **[VERIFIED: flag semantics + 0.033; spec-z error value NOT stated in paper — the
  repo's σ_z ≈ 0.0017 for flag-3 is an empirical catalogue statistic, not a paper quote.]**
- **(e) Uncorrected galaxies (z > 0.05 / no B mag / no 2M++ match):** the paper states
  **no default PV error**; col 31 is presumed null for them. **[INFERRED: null pattern not
  verifiable locally — raw `GLADE+.txt` is not on this machine, only the post-quadrature
  reduced CSV.]**

## 2. Layer 2 — Repo parsing (`master_thesis_code/galaxy_catalogue/handler.py`)

- `CatalogueColumns` (lines 143–166): `REDSHIFT=28` (z_cmb ✓), `REDSHIFT_PECULIAR_VELOCITY_ERROR=30`
  (= paper col 31, PV error ✓), `REDSHIFT_MEASUREMENT_ERROR=31` (= paper col 32, z_helio
  measurement error ✓), `REDSHIFT_FLAG=34` (= paper col 35 ✓). Column identification is
  **correct**. **[VERIFIED: file read + paper table]**
- The PV-correction **flag (0-based col 29) is not read** — the parse cannot distinguish
  BORG-corrected from uncorrected galaxies. **[VERIFIED: absent from `CatalogueColumns`]**
- `parse_to_reduced_catalog` line 344: `fillna({REDSHIFT_PECULIAR_VELOCITY_ERROR: 0.0015})`
  — every galaxy with a null PV-error column gets σ_z = 0.0015 (≈ 428 km/s at z=0.05 with
  the (1+z) convention; ≈ 450 km/s without). **Uncited constant** (commit `0175a55`: "GLADE
  now uses rms peculiar velocity error on redshift if none is given"; re-flagged uncited in
  `docs/PIPELINE_BUGS_REPORT.md:335`). **[VERIFIED]**
- Lines 347–350: stored `z_error = sqrt(measurement_error² + pv_error²)`. **So yes — the
  parse-time quadrature already folds catalogue-reported PV uncertainty (σ_tot for
  corrected galaxies, the 0.0015 stand-in for everyone else) into the stored `z_error`.**
  The PV column is then dropped (line 356); downstream code cannot un-mix it. **[VERIFIED]**

## 3. Layer 3 — Likelihood residual term

- `bayesian_statistics.py:3196-3197` (scalar kernel; identically at ~3645–3648 batched and
  ~4065): `sigma_z_pv = (1+z_g)·SIGMA_V_PEC_KM_S/c`, added **in quadrature on top of** the
  stored `host_z_error`. Applied once per kernel (the comment block at 3180–3195 correctly
  argues no *internal* double-count within the likelihood). **[VERIFIED]**
- `constants.py:71-83`: `SIGMA_V_PEC_KM_S = 200.0`, issue #16 decision 2026-07-03.
  (1+z) factor: Davis et al. 2011 arXiv:1012.2912 Eqs. (1)/(A1) ✓ standard. Value: cites
  Fishbach et al. 2019 arXiv:1807.05667 and Chen et al. 2018; Laghi et al. 2021 (500 km/s)
  kept as a systematics row. The comment explicitly claims the 200 km/s is "the residual
  (uncorrected/nonlinear) dispersion on top of" the catalogue PV error. **[VERIFIED: file]**
- Issue #16 (CLOSED) offered options (a) value-correct or (b) marginalise σ_v ≈ 150–500
  km/s; the implemented commit is `8568d9f` "[PHYSICS] marginalize residual host peculiar
  velocity into the host-z kernel (issue #16)". **[VERIFIED: `gh issue view 16`, git log]**
- Fishbach et al. 2019: "we assign a 200 km/s Gaussian uncertainty to the 'Hubble
  velocity' of each galaxy" — **[VERIFIED via ar5iv]**. Note their 200 km/s is a
  *stand-alone* PV budget for catalogue redshifts, not a term added on top of a
  catalogue-reported σ_tot.

## 4. Reconciliation — is there a double-count?

**Yes, in both branches, with different structure:**

| Host class | Parse-time z_error contains | Likelihood adds | Double-count? |
|---|---|---|---|
| **BORG-corrected** (z<0.05, 2M++, B-mag; the EMRI venue) | σ_meas ⊕ σ_tot, where σ_tot = σ_borg ⊕ **σ_vir (non-linear residual)** | (1+z)·200/c | **Yes** — the "non-linear residual" rationale in `constants.py` is exactly what σ_vir already covers per GLADE+ §2.2. The 200 km/s re-adds it. |
| **Uncorrected** (col 31 null) | σ_meas ⊕ **0.0015** (≈ 430–450 km/s ≈ a *full* PV dispersion stand-in) | (1+z)·200/c | **Yes** — the floor already represents the full uncorrected PV; adding 200 km/s inflates it again. Combined ≈ sqrt(0.0015² + 7.0e-4²) = 1.66e-3 ⇔ ~470 km/s — *accidentally* close to Laghi's 500 km/s, so numerically defensible but structurally a double-count. |

**Caveat on severity:** the double-count is a *width inflation*, not a bias mechanism.
In mock mode it is inert for the point/δ-kernel numerator (bypasses `host_z_error_eff` for
N_g) and second-order for windows/Z_g. It matters for real-data mode and for the #40b
kernel derivation (`docs/derivations/hostz_pv_photoz_kernel.md` §2 flagged exactly this
risk; this audit resolves it: **the risk is real**).

## 5. What residual σ_v is scientifically defensible (real-data mode)

- **Carrick et al. 2015 (arXiv:1504.04627), §4.2.1:** "We take σ_v = 150 km s⁻¹ based on
  the tests discussed in Appendix A" — residual scatter about *linear-theory 2M++
  predictions* (the same 2M++ density field GLADE+'s BORG correction cross-matches
  against). **[VERIFIED via ar5iv]** This is the canonical post-linear-correction
  residual, and it is *of the same nature as* GLADE+'s σ_vir term.
- **Fishbach et al. 2019: 200 km/s** — but as the *entire* PV budget for
  frame-corrected-only redshifts, not a residual on top of a reported σ_tot. **[VERIFIED]**
- **Laghi et al. 2021: σ_z = 0.0015 ⇔ ~500 km/s** total PV budget with *no* correction
  (their Eq. 2.8/2.10 context, per repo derivation note). **[VERIFIED at the level of the
  repo's derivation skeleton; eq. numbers from `hostz_pv_photoz_kernel.md` §2.]**

Defensible assignment, per host class:

1. **BORG-corrected hosts:** residual on top of catalogue σ_tot should be **0** if GLADE+'s
   σ_borg ⊕ σ_vir error model is trusted, or a conservative **~150 km/s** (Carrick) if one
   doubts per-galaxy σ_vir coverage (halo-mass mis-estimates, BORG boundary effects). A
   blanket 200 km/s on top of σ_tot is not first-principles derivable.
2. **Uncorrected hosts:** exactly **one** full-PV term of **300–500 km/s** (Fishbach 200
   applies only after a value correction they performed; with *no* correction Laghi's 500
   is the honest number), (1+z)-scaled, applied **once** — either at parse (replacing the
   uncited 0.0015) or in the likelihood, not both.

## 6. Does it matter for photo-z hosts? — No.

σ_z,pv(200 km/s) / σ_z,photo with σ_photo = 0.033(1+z) (paper value; repo uses ≈0.035):

| z | σ_z,pv(200) | σ_z,photo | ratio | quadrature width inflation |
|---|---|---|---|---|
| 0.05 | 7.00e-4 | 0.0347 | **0.020** | **+0.020%** |
| 0.10 | 7.34e-4 | 0.0363 | **0.020** | **+0.020%** |

(The ratio is z-independent because both scale as (1+z): ratio = (σ_v/c)/0.033 ≈ 0.0202.)
Even the 0.0015 floor is only 4% of σ_photo (+0.09% in quadrature). **For flag-1 hosts the
entire PV question is numerically irrelevant.** It matters only for flag-3 spec-z hosts
(σ_meas ≈ 0.0017): there, floor + 200 km/s inflates the kernel from 1.7e-3 to 2.37e-3
(**+40%** over bare σ_meas; +4.7% is the marginal effect of the 200 km/s on top of the
already-folded floor) — directly relevant to the golden-event widths (P-2 measured the
PV/floor terms at 2.3–4.9× the retained σ_dL budget).

## 7. Recommendation

**Keep the parse-time quadrature of the genuine catalogue PV-error column (it consumes
GLADE+'s intended σ_tot correctly), but for real-data mode (i) parse the PV-correction
flag (0-based col 29) and stop adding `SIGMA_V_PEC_KM_S = 200` on top of σ_tot for
BORG-corrected hosts (set the residual to 0, or 150 km/s per Carrick 2015 §4.2.1 as a
conservative systematics row), and (ii) for uncorrected hosts replace the uncited
parse-time 0.0015 fill with a single, cited, (1+z)-scaled full-PV term of ~500 km/s
(Laghi 2021) applied in exactly one place — eliminating both double-counts.** Any change
is a physics change (`/physics-change` gate, `constants.py` + `handler.py` +
`bayesian_statistics.py` are trigger files) and interacts with the #40b real-data kernel
derivation, which should absorb this audit as its §3.1 double-counting answer.

## 8. Verified-vs-inferred ledger

| Claim | Status |
|---|---|
| GLADE+ col semantics (28–35, 1-based) & flag meanings | VERIFIED (paper table, ar5iv) |
| BORG correction scope z≤0.05 ∩ 2M++ ∩ B-mag; σ_tot² = σ_borg² + σ_vir²; σ_vir = Eq. (1) non-linear virial term | VERIFIED (paper §2.2) |
| Repo column mapping and quadrature (`handler.py:143-166, 344-356`) | VERIFIED (file read) |
| Likelihood residual term (`bayesian_statistics.py:3196-3197, 3645-3648`; `constants.py:71-83`) | VERIFIED (file read) |
| Issue #16 decision context | VERIFIED (`gh issue view 16`, commit `8568d9f`) |
| Carrick 2015 σ_v = 150 km/s residual (§4.2.1) | VERIFIED (ar5iv) |
| Fishbach 2019 200 km/s assignment | VERIFIED (ar5iv) |
| Col 31 null ⇔ PV-correction flag = 0 (which rows get the 0.0015 fill) | INFERRED — raw GLADE+.txt not on this machine; verify against the cluster copy before implementing |
| Laghi 2021 Eq. (2.8)/(2.10), 500 km/s | VERIFIED only via repo derivation note, not re-opened here |
| GLADE+ paper stating a single residual km/s value | NOT FOUND — paper gives per-galaxy σ_vir formula, no scalar |
| flag-3 spec σ_z ≈ 0.0017 | NOT in paper — repo empirical catalogue statistic |
