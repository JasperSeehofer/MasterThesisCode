# Frame Systematic Verification: Heliocentric vs CMB-Frame Redshift in the GLADE+ Host Lookup

**Status:** VERIFICATION ONLY — no code changed.
**Date:** 2026-06-30
**Scope:** Confirm whether the catalogue parser feeds the *heliocentric* redshift into the
dark-siren `d_L(z; H0)` relation without any frame / peculiar-velocity correction, and quantify
the resulting coherent H0 systematic.

---

## 1. Verdict and the Exact Column Fix

### Verdict: THE BUG IS REAL (confidence: HIGH)

The pipeline uses the **heliocentric** redshift *value* as the host redshift that enters
`dist(z) -> d_L(z; H0)`. It never reads the adjacent CMB-frame redshift column, and it never
applies any frame conversion or peculiar-velocity *value* correction. The solar-motion dipole
(v_sun ~ 369.8 km/s) is left entirely uncorrected in `cz = H0 d_L`.

- `CatalogueColumns.REDSHIFT = 27` (0-based) is declared at `handler.py:146`, assigned to
  `HostGalaxy.z` at `handler.py:66`, and that `z` flows into the luminosity-distance relation.
- The peculiar-velocity column (0-based 30) is used ONLY to inflate the redshift *error* in
  quadrature (`handler.py:302-307`, null -> 0.0015). It never corrects the redshift value.
- No CMB-frame redshift column is referenced anywhere in the code.

### The Fix: read 0-based column 28 (z_cmb) instead of 0-based column 27 (z_helio)

The CMB-frame redshift is GLADE+ 1-based column 29 (= 0-based **28**), immediately adjacent to the
heliocentric column the code currently reads. In the raw GLADE+ file it is already populated by
Dalya et al. (no extra computation needed): the one-line change is `REDSHIFT = 28`.

> **Caveat (must verify at execution time, NOT assumed):** whether the *reduced* on-disk file
> `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv` preserved the original
> raw-file column positions. If the reduction step re-exported a named subset of columns, the
> z_cmb column may not sit at position 28 (or may have been dropped). The field *identities* below
> are certain; the positional index in the reduced file is the one item to confirm before editing.

### Verified GLADE+ (40-column) Column Map

GLADE+ documents columns 1-based; the code is 0-based, so every code index = documented index - 1.

| 1-based | 0-based | Field | Code constant | Status |
|--------:|--------:|-------|---------------|--------|
| 9  | 8  | RA (deg, J2000) | `RIGHT_ASCENSION = 8` | OK |
| 10 | 9  | Dec (deg, J2000) | `DECLINATION = 9` | OK |
| 11 | 10 | apparent B magnitude | `APPARENT_B_MAG = 10` | OK |
| **28** | **27** | **z_helio (heliocentric redshift)** | **`REDSHIFT = 27`** | **CURRENT host z (the bug)** |
| **29** | **28** | **z_cmb (CMB-frame redshift)** | **— (not read)** | **THE FIX TARGET** |
| 30 | 29 | PV-correction flag (0=uncorrected, 1=PV-corrected) | — (not read) | available |
| 31 | 30 | v_err (error from PV estimation) | `REDSHIFT_PECULIAR_VELOCITY_ERROR = 30` | OK (error-only use) |
| 32 | 31 | z_err (measurement error of z_helio) | `REDSHIFT_MEASUREMENT_ERROR = 31` | OK |
| 33 | 32 | d_L (Mpc) | — (not read) | available |
| 35 | 34 | dist/redshift-type flag (measurement type) | `REDSHIFT_FLAG = 34` | OK (photometric selection) |
| 36 | 35 | M* (10^10 Msun) | `STELLAR_MASS = 35` | OK |
| 37 | 36 | M* error | `STELLAR_MASS_ABSOULTE_ERROR = 36` | OK |

**Index-mapping confidence:**
- **CONFIRMED (certain):** z_helio = 0-based 27, z_cmb = 0-based 28, and all seven checked code
  indices reproduce the documented GLADE+ fields with zero discrepancies. A coincidental match
  across seven independent indices is effectively impossible, so the 0-based interpretation is
  certain. Two independent authoritative sources (Dalya et al. 2022 Sec. 4; official GLADE+
  website) give identical 40-column layouts.
- **ASSUMED (verify at execution):** that `reduced_galaxy_catalogue.csv` retained raw positional
  ordering so that z_cmb lands at 0-based 28 in the reduced file specifically.
- **Do not confuse the two flag columns:** the PV-correction flag (1-based 30) is distinct from the
  dist/measurement-type flag (1-based 35). `REDSHIFT_FLAG = 34` (0-based) is the *latter*
  (measurement type), used for photometric-row selection at `handler.py:298-299`.

---

## 2. Quantified H0 Bias

Because `cz = H0 d_L` at low z, a redshift offset maps directly to an H0 offset: `delta_H0/H0 =
delta_z/z`. The heliocentric-frame error injects `delta_z = (v_sun/c) cos(theta_apex)`, where
theta_apex is the angle between the line of sight and the solar apex (galactic l,b = 264.0, 48.3).

- `v_sun/c = 369.8 / 299792.458 = 1.2335e-3`.

### Per-event (worst case)

At z = 0.05 (cz ~ 15000 km/s), along the apex / anti-apex line of sight:

`delta_H0/H0 = (v_sun/c)/z = 1.2335e-3 / 0.05 = 0.0247 = +/-2.47%`

This is the per-event *envelope* (+2.47% apex hemisphere, -2.47% anti-apex), not the ensemble bias.

### Ensemble (net) bias — DETECTED-HOST sample

Host sky positions taken from the actual detected-host run output
`simulations/cramer_rao_bounds.csv` (N = 3375; ecliptic qS, phiS), frame
`ecliptic_BarycentricTrue_J2000` confirmed via the migration JSON `coord_frame` field. Converted
to Galactic with astropy; redshifts from luminosity_distance inverted through
`FlatLambdaCDM(H0=73, Om0=0.25)` (matching project WMAP-era constants).

- `<cos theta_apex> = +0.0245` (std 0.612, N = 3375) — near-zero mean confirms the solar dipole
  **largely sky-averages out** over the detected sample.
- Net ensemble H0 bias (rigorous per-event average `mean(beta*cos/z)`): **+0.151%**.
- Simplified form `(beta*<cos>)/z_typ` (z_typ = 0.046): +0.065%.

**Sky-averaging caveat:** the dipole is zero-mean over an isotropic sky, so the net ensemble bias
(+0.15%) is far below the per-event envelope (2.47%). The residual non-zero net comes from the
mild anisotropy of the detected-host distribution. The net is therefore **sensitive to the actual
sky selection** of detected events.

**Full-catalogue cross-check bound (labelled, NOT the detected sample):**
`reduced_galaxy_catalogue.csv` (on-disk EQUATORIAL J2000, verified RA 0.0005-360 deg,
Dec -89.88..+89.78), 200k random sample FK5(J2000)->Galactic: `<cos> = +0.135` => **+0.33%**.
This is a *bound*, not the detection bias — the full catalogue has a different sky/redshift
distribution than the detections.

---

## 3. Peculiar-Velocity Recommendation

Use a two-step convention (the frame conversion and the PV treatment are distinct):

1. **Remove solar motion (the missing VALUE correction):** read z_cmb (0-based col 28) instead of
   z_helio. This subtracts the solar-system dipole. This is the primary fix.
2. **Treat the host's own peculiar velocity** (still present in z_cmb), by EITHER:
   - **(a) Value correction** with a reconstructed PV field (linear theory / 2M++). GLADE+ already
     provides a PV-corrected redshift for nearby galaxies, flagged by 0-based col 29 with its
     correction uncertainty in 0-based col 30; OR
   - **(b) Marginalize** as added uncertainty: sigma_v ~ 150-500 km/s (commonly ~200 km/s),
     converted to a redshift error and added in quadrature to cz.

**What the code does today:** only a partial version of (b) — it adds the GLADE+ PV-correction
*error* (col 30) in quadrature into the redshift error, but (i) never applies the frame/value
correction, and (ii) treats that column as a generic error term rather than the uncertainty of an
actually-applied PV correction.

**Key physics point:** error inflation is NOT a substitute for using the CMB-frame redshift value.
The solar-motion frame offset is a *coherent, direction-dependent bias*, not zero-mean noise.
Inflating the error neither removes nor accounts for it.

**Residual-bias warning:** the heliocentric->CMB step removes ONLY the solar dipole. A coherent
large-scale bulk flow (~150-250 km/s; take 200) leaves up to ~1.3% direction-dependent residual
bias if the host distribution is anisotropic. The random PV scatter (sigma_v ~ 300 km/s =>
sigma_z = 1.0e-3 => 2.00% per-event at z = 0.05) averages down as 1/sqrt(N): 2.00%/sqrt(3375) =
0.034% over the detected ensemble.

---

## 4. Separation from the H0 Railing — EXPLICIT CONFIRMATION

**The frame systematic is SEPARATE from, and cannot cause, the H0 railing (H0 -> 0.86).**

| Property | Frame systematic | H0 railing |
|----------|------------------|------------|
| Magnitude | net +0.15% (worst-case per-event 2.47%) | ~+18% |
| Sky dependence | intrinsically direction-dependent (scales with cos theta_apex) | sky-position-INDEPENDENT |
| Reproduces in no-sky closure? | No (vanishes without sky directions) | YES (reproduced) |
| Mechanism | uncorrected solar dipole in cz | normalisation / prior domination |

- The net frame bias (+0.151%) is ~120x smaller than the +18% railing (18/0.151 ~ 119).
- The railing is reproduced in a **no-sky closure** — it is sky-position-independent, a
  normalisation/prior-domination problem. The frame error is intrinsically direction-dependent.
- They are **orthogonal degrees of freedom:** the frame error cannot generate the railing, and
  removing the railing does not remove the frame systematic.

---

## 5. Severity and Recommended Action

**Severity: LOW-to-MODERATE for the final H0 result, but it is a genuine, citable systematic.**

- The *net ensemble* bias on the detected sample is small (+0.15%), well below the statistical
  H0 uncertainty expected from 3375 dark sirens, because the solar dipole largely sky-averages.
- However: (i) it is a *coherent, known-direction* systematic that a referee will expect to see
  controlled or at least bounded; (ii) the per-event envelope is 2.47%, so any sky-selection
  anisotropy or smaller detected sample raises the net; (iii) the residual bulk-flow term (up to
  ~1.3%) is comparable to plausible statistical errors. It is a "correctness" issue (wrong frame),
  cheap to fix, and standard practice in every GW H0 analysis.

**Recommended action: FIX NOW (one-line value fix) + FILE ISSUE for the full PV treatment.**

1. **Fix now (after the one execution-time check):** confirm z_cmb's position in
   `reduced_galaxy_catalogue.csv`, then read 0-based col 28 (z_cmb) instead of col 27 (z_helio).
   This is a physics change (redshift value feeding `d_L(z; H0)`): route through `/physics-change`,
   prefix the commit `[PHYSICS]`. If z_cmb was dropped during reduction, regenerate the reduced
   catalogue retaining col 28 first.
2. **File issue** for the second-order PV treatment (host peculiar velocity value-correction vs
   marginalization, and the residual bulk-flow bound), tagged `physics`. Lower priority than the
   frame fix.
3. **Detected-host data:** already available and used (`simulations/cramer_rao_bounds.csv`); no
   additional data is needed to quantify the systematic.

---

## Sources

- GLADE+ official column description: https://glade.elte.hu/
- Dalya et al. 2022, "GLADE+: an extended galaxy catalogue...", MNRAS 514, 1403,
  arXiv:2110.06184, Sec. 4 (column table); ar5iv: https://ar5iv.labs.arxiv.org/html/2110.06184
- Code under audit: `master_thesis_code/galaxy_catalogue/handler.py`
  (CatalogueColumns lines 139-152; REDSHIFT used :66, :146; PV-error quadrature :302-307;
  photometric-flag selection :298-299)
- Gray et al. 2020 (gwcosmo dark-siren / completeness; project-cited) — CMB-frame + PV convention
- Howlett & Davis 2020 (arXiv:1909.00587); Nicolaou et al. 2020 (arXiv:1909.09609) — PV treatment
- Detected-host sky/redshift: `simulations/cramer_rao_bounds.csv` (+ `.migration.json`,
  coord_frame = ecliptic_BarycentricTrue_J2000)
- Cross-check bound: `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv`
  (on-disk EQUATORIAL J2000)
- astropy.coordinates (FK5 J2000, BarycentricTrueEcliptic, Galactic);
  astropy.cosmology.FlatLambdaCDM(H0=73, Om0=0.25)
- Solar/CMB dipole: v_sun = 369.8 km/s toward galactic l = 264.0, b = 48.3 (Planck 2018)
