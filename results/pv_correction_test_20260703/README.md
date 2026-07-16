# PV-uncorrected galaxy-catalogue variant (issue #16, handoff §7b)

**Date:** 2026-07-03 · **Branch:** `physics/campaign-depth-pv` · **Builder:** `build_uncorrected_variant.py`

## What this is

`reduced_galaxy_catalogue_noPVcorr.csv` is a variant of the live reduced GLADE+ catalogue
(`master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv`, 8 headerless columns:
`RA, Dec, B_mag, REDSHIFT, REDSHIFT_MEASUREMENT_ERROR, STELLAR_MASS, STELLAR_MASS_ABSOULTE_ERROR,
REDSHIFT_FLAG`) in which the **peculiar-velocity value-correction has been removed** from the
redshift column:

- GLADE+ raw col 28 (0-based) is z_cmb; where raw col 29 (flag2) == 1 it is additionally
  PV-corrected (Dálya et al. 2022, arXiv:2110.06184). The live catalogue uses col 28 as-is.
- In the variant, for the **709,117 flag2==1 rows** (3.13% of the catalogue; all have finite
  z_helio) the redshift is replaced by the pure heliocentric→CMB **frame-only** transform of
  raw col 27 (z_helio). All **flag2==0 rows are byte-identical** to the live catalogue
  (verified by full-file diff: exactly 709,117 differing lines, each differing **only** in the
  redshift column).
- Everything else replicates `GalaxyCatalogueHandler.parse_to_reduced_catalog`
  (handler.py:311–369) exactly: `z_flag ∈ {1, 3}` filter, PV-error (raw col 30) NaN→0.0015 on
  all rows and folded in quadrature into raw col 31 as `REDSHIFT_MEASUREMENT_ERROR`
  (the error handling is deliberately **unchanged** — only the redshift *value* differs),
  integer flag, same column order, headerless CSV.

## Validated frame transform

Validated on the 21,925,647 flag2==0 rows with finite (z_helio, z_cmb), where GLADE+'s z_cmb
must equal the pure frame transform. Convention selected by smallest median
|z_cmb_reconstructed − z_cmb_GLADE| among five candidates (see `stats.json`):

**Chosen: `mult_plus` — (1 + z_cmb) = (1 + z_helio) · (1 + (v_sun/c)·cosθ)**,
θ = angle between the galaxy direction and the solar apex, Planck 2018 dipole
(v_sun = 369.82 km/s toward galactic l = 264.021°, b = 48.253°, arXiv:1807.06205;
apex → ICRS RA 167.942°, Dec −6.944° via astropy).

| convention | median | p99 | max |
|---|---|---|---|
| **mult_plus (chosen)** | **5.28e-05** | **1.98e-04** | 5.77e-03 |
| mult_sr_plus (SR γ) | 5.29e-05 | 1.98e-04 | 5.78e-03 |
| add_plus (z_cmb = z_helio + βcosθ) | 6.64e-05 | 1.77e-03 | 1.44e-02 |
| add_minus | 1.52e-03 | 3.68e-03 | 1.69e-02 |
| mult_minus | 1.66e-03 | 5.42e-03 | 2.55e-02 |

The multiplicative structure is decisive: the additive form fails at high z (p99 1.8e-3,
the β·z tail of the photo-z quasars), while `mult_plus` stays ≤2e-4 at p99 across the full
z range. Sign/direction flips are ruled out at the 30× level. The best-convention median
(5.3e-5) passes the <1e-4 acceptance gate.

### Residual-floor characterization (sample diagnostics, first 4M raw rows)

The residual is not "tiny" (<1e-5), so it was chased down rather than assumed away:

- Not storage rounding: z_cmb is stored to 6 decimals for flag2==0 rows (quantum 5e-7).
- Not an uncorrected subpopulation: only 0.01% of flag2==0 rows have z_cmb == z_helio.
- Per-row implied dipole (spec-z, 0.1<z<0.4, |cosθ|>0.6): 100% within v ∈ [330, 355) km/s
  under the multiplicative form (median 350.4) — vs additive-implied median 395. Direction
  fits l = 264.1–264.2°, b = 48.14–48.18° (within 0.15° of Planck) in every subpopulation.
- The *effective amplitude* GLADE+ used drifts with subpopulation: ~363.4 km/s (spec-z
  z<0.05, post-fit residual median 2.8e-6), ~355.5 (spec-z all), ~343–347 km/s (photo-z).
  I.e. GLADE+ applied a multiplicative dipole toward the Planck direction with a small
  (~2–7%), z-dependent amplitude deficit relative to 369.82 km/s, unexplained (possibly a
  heterogeneous upstream conversion across source catalogues). Fixsen-96 parameters
  (371 km/s, l=264.14, b=48.26) fit slightly *worse* than Planck.

**Impact bound:** for the flagged rows (all z<0.11, spec-z-like — the population where the
low-z spec-z fit applies), the amplitude ambiguity (369.82 vs ~363 km/s) shifts the
frame-only reconstruction by ≤|Δv|/c ≈ 2.1e-5 — a factor ~30 below the median PV correction
being removed (6.3e-4). The documented, reproducible Planck values are therefore used.

## PV-correction magnitude removed (flag2==1 rows)

|z_GLADE_PV-corrected − z_frame-only| over the 709,117 flagged rows:

- median **6.30e-4** · p90 **1.89e-3** · p99 **3.11e-3** · max **5.19e-2**
- flagged-row redshift range: z_cmb ∈ [−3.18e-4, 0.109] (i.e. PV corrections live entirely
  at low z, exactly where dark-siren H0 information concentrates)

Example (raw row 2, NGC4548, Virgo): live PV-corrected z = 0.0035733 → variant frame-only
z = 0.0051941 (Δz = 1.62e-3 ≈ 486 km/s), hand-verified to the last printed digit.

## Verification summary

- Row parity: filtered raw stream = live reduced CSV = variant = **22,641,048** rows (exact).
- Full-file diff: 709,117 differing lines == n_flagged; all diffs confined to the redshift
  column; flag2==0 rows byte-identical.
- Production-reader smoke test: variant loads with the 8 `_reduced_catalog_column_names()`
  columns, correct dtypes, `REDSHIFT_FLAG ∈ {1, 3}`.
- Deterministic: no randomness anywhere; full provenance in `stats.json`; run log in
  `build.log` (86.5 s total on the dev machine).

## Files

- `build_uncorrected_variant.py` — two-pass builder (PASS 1 validation w/ abort gates,
  PASS 2 write). Refuses to run if the output CSV exists (the writer appends).
- `reduced_galaxy_catalogue_noPVcorr.csv` — the variant (1.7 GB, 22,641,048 rows, headerless).
- `stats.json` — machine-readable validation + build stats.
- `build.log` — run log.

## Intended usage (isolated PV value-correction impact test)

Run the production Bayesian evaluation (`--evaluate`, Pipeline B) on the **frozen seed600
event set** twice — once against the live catalogue, once against this variant — with
everything else identical (same events, same CRB, same settings). The catalogues are
identical except for the redshift value on the 709,117 PV-corrected low-z rows, so the
difference between the two H0 posteriors **isolates the impact of the GLADE+ peculiar-velocity
value-correction at low z**. Swap mechanics: point the evaluation at this CSV (symlink or
temporary copy into the expected `REDUCED_CATALOGUE_FILE_PATH`); do **not** overwrite the live
file — restore the original afterwards. Per issue #16, this runs locally, in parallel with the
campaign's inference-side σ_v marginalization; it does not gate the campaign.
