# [P3-MKER] round 2 — Measurer B — independent readmission test for candidate 6791151

Seed 900121, event 20 (0-based). Independent route: direct chunked/raw reads of
`darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`, hand re-implementation of
`_empiric_stellar_mass_to_BH_mass_relation` (handler.py:1368-1382) and of the mass/redshift
prune (`_mass_redshift_prune_mask`, handler.py:215-251) — no call into
`GalaxyCatalogueHandler` itself. Cosmology-utility functions (`get_redshift_outer_bounds`,
`dist_to_redshift`) were called directly from `physical_relations.py` since they are not the
object under test.

Dataset pin: file `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`, size
1,681,954,844 bytes, `md5sum` = `c52c13b5cab61f6b3f04bbe202550969` — re-verified in this
session (matches the orchestrator-supplied pin; no mismatch, no stop condition triggered).
`free -g`: 30 GB total / ~21 GB available before starting — well within budget for a 4-column
(REDSHIFT, REDSHIFT_MEASUREMENT_ERROR, STELLAR_MASS, STELLAR_MASS_ABSOULTE_ERROR), 22,641,048-row
`usecols` read (~723 MB as float64), so no chunking was required for the vectorized full-column
pass; the three individual row cross-checks additionally used a minimal `skiprows`/`nrows` window
read and a raw `sed -n` line read (two more independent routes, see §1).

## 1. Index semantics (established first, cross-checked three ways)

Reduced-catalogue on-disk column order (headerless CSV, `_reduced_catalog_column_names()`,
handler.py:193-212): `RIGHT_ASCENSION(0), DECLINATION(1), APPARENT_B_MAG(2), REDSHIFT(3),
REDSHIFT_MEASUREMENT_ERROR(4), STELLAR_MASS(5), STELLAR_MASS_ABSOULTE_ERROR(6), REDSHIFT_FLAG(7)`.

`GalaxyCatalogueHandler.__init__` (handler.py:267-356) applies, in order:
1. `_map_stellar_masses_to_BH_masses()` — overwrites the `STELLAR_MASS`/`STELLAR_MASS_ABSOULTE_ERROR`
   columns in place with the *computed* BH mass / BH mass error (handler.py:1136-1142).
2. `_remove_galaxies_without_mass_information()` — drops rows where the computed BH mass is NaN
   (handler.py:1131-1134).
3. `_get_pruned_galaxy_catalog(M_min, M_max, z_max)` — applies `_mass_redshift_prune_mask` with
   `M_min=M_SOURCE_FRAME_MIN=1e4`, `M_max=M_SOURCE_FRAME_MAX=1e7` (constants.py:125-126, passed
   from `main.py:154-157`), `z_max=cosmological_model.max_redshift=1.5` (default, no
   `--max_redshift` override for this fleet; `cosmological_model.py:199-200`).
4. `setup_galaxy_catalog_balltree()` calls `self.reduced_galaxy_catalog.reset_index()`
   (handler.py:555) — THIS is what fixes "catalog_index": pandas boolean-mask filtering in steps
   2-3 preserves the original 0-based row-label values, so `reset_index()` writes those preserved
   original-row labels into a new `"index"` column while resetting the frame's own positional
   index to `0..M-1`. All downstream candidate lookups (`.iloc[indices]` off the BallTree query,
   the mass-filter mask) operate on this **post-prune, reset positional frame**. So "catalog_index"
   as used in the banked posteriors = **position in the reduced_galaxy_catalog frame AFTER the BH-mass
   mapping + mass/redshift prune + reset_index()** — NOT a raw CSV row number.

**Cross-check (mandatory, both candidates reproduced exactly):**

Reimplemented the full pipeline independently (vectorized full-catalogue pass, own code, not the
handler): computed `BH_mass`/`BH_mass_error` for all 22,641,048 rows from raw
`STELLAR_MASS`/`STELLAR_MASS_ABSOULTE_ERROR`, built the keep-mask
(`has_mass & (BH_mass+BH_mass_error>=1e4) & (BH_mass-BH_mass_error<=1e7) & (REDSHIFT-REDSHIFT_ERROR<=1.5)`),
took `orig_idx = np.flatnonzero(keep)` (21,753,847→20,834,171 rows survive: mass-info drop then
mass/z-band prune), and read off `orig_idx[reset_frame_position]` to get the raw CSV row:

| reset-frame position | raw CSV row (0-based) | raw `STELLAR_MASS` | raw `STELLAR_MASS_ABSOULTE_ERROR` | computed `BH_mass` | computed `BH_mass_error` (0.24-dex-only, as coded) |
|---|---|---|---|---|---|
| 6791158 | 7351457 | 0.3 | 0.6 | **709540.709** | **1570331.165** |
| 6791138 | 7351437 | 0.3 | 0.3 | **709540.709** | **894866.276** |

Both reproduce the task's given cross-check values (`host_M=709540.709` for both; `host_M_error`
1570331.165 / 894866.276 for σ_ratio 2.00 / 1.00) **exactly**, to the printed precision. Raw values
independently re-confirmed with a plain `sed -n '<line>p'` read of the file (file line = 0-based
row + 1, no header):

```
$ sed -n '7351438p' reduced_galaxy_catalogue.csv   # row 6791138 (0-based 7351437)
98.1206114,-64.2284019,18.368592,0.0574429999999999,0.0349293521910955,0.3,0.3,1
$ sed -n '7351458p' reduced_galaxy_catalogue.csv   # row 6791158 (0-based 7351457)
97.8901179,-64.2698205,17.330164,0.0314027952049768,0.034048737858373264,0.3,0.6,1
```

→ **Index-semantics verdict: CONFIRMED.** `index_crosscheck_passed = true`. Proceeding with this
interpretation for candidate 6791151.

## 2. Candidate 6791151 — raw and derived properties

Reset-frame position 6791151 → raw CSV row **7351450** (0-based) → file line **7351451** (1-based).
Raw sed read:

```
$ sed -n '7351451p' reduced_galaxy_catalogue.csv
98.1044995,-64.2752005,19.266656,0.052818,0.03477655735905269,0.1,0.1,1
```

| Column (`InternalCatalogColumns`) | on-disk name | value |
|---|---|---|
| `PHI_S` (RA, pre-rotation) | `RIGHT_ASCENSION` | 98.1044995 deg |
| `THETA_S` (Dec, pre-rotation) | `DECLINATION` | -64.2752005 deg |
| `B_MAG` | `APPARENT_B_MAG` | 19.266656 |
| `REDSHIFT` | `REDSHIFT` | **0.052818** |
| `REDSHIFT_ERROR` | `REDSHIFT_MEASUREMENT_ERROR` | **0.03477655735905269** |
| raw stellar mass (pre-map) | `STELLAR_MASS` | 0.1 (× 1e10 M☉) |
| raw stellar mass error (pre-map) | `STELLAR_MASS_ABSOULTE_ERROR` | 0.1 (× 1e10 M☉) |
| `REDSHIFT_FLAG` | `REDSHIFT_FLAG` | 1 (photometric) |

Derived (`_empiric_stellar_mass_to_BH_mass_relation`, handler.py:1368-1382, reimplemented from
scratch):

```
BH_mass = exp(alpha + beta*ln(M_*/10))
BH_mass_error = BH_mass * sqrt(sigma_int^2 + d_alpha^2 + (ln(M_*/10)*d_beta)^2 + (beta/M_* * M_*_err)^2)
  alpha=7.45*ln(10), beta=1.05, d_alpha=0.08*ln(10), d_beta=0.11, sigma_int=0.24*ln(10)
```

with `M_* = 0.1`, `M_*_err = 0.1`:

- `InternalCatalogColumns.BH_MASS` = **223872.11385683485** M☉
- `InternalCatalogColumns.BH_MASS_ERROR` (current, 0.24-dex-only budget) = **291758.99489010876** M☉

## 3. Window test under the CURRENT (0.24-dex-only) budget

Event inputs, seed 900121 event 20 (0-based), from
`p3_2d_fleet_20260825/bt_900121_work/seed900121/simulations/prepared_cramer_rao_bounds.csv` row 20
(`.iloc[20]`; arm-cross-checked against `bc_900121_work/.../prepared_cramer_rao_bounds.csv` row 20
— `M`, `delta_M_delta_M`, `luminosity_distance` are bit-identical between arms, confirming the
window is arm-independent as the claim card states):

- `M` (= `M_z_obs`, `Detection.M`) = **1333246.127516857** M☉
- `delta_M_delta_M` → `M_z_sigma = Detection.M_uncertainty = sqrt(delta_M_delta_M)` = **0.005188122836068134** M☉
- `luminosity_distance` (`Detection.d_L`) = 0.2831422160233205 Gpc
- `delta_luminosity_distance_delta_luminosity_distance` → `Detection.d_L_uncertainty` = 0.0014316570944745673 Gpc

Redshift-range bounds (`bayesian_statistics.py:4669-4679`, `get_redshift_outer_bounds` called with
`h_min=0.6, h_max=0.86` from `LamCDMScenario.h` limits (`cosmological_model.py:388-389`),
`Omega_m_min=0.04, Omega_m_max=0.5` (`cosmological_model.py:396-397`), `sigma_multiplier=2.0`,
result capped at `redshift_upper_limit = cosmological_model.max_redshift = 1.5`, not binding here):

- `z_min = 0.05356499027434118`
- `z_max = 0.07776556271743075` (below the 1.5 cap)

(Redshift *filter*, handler.py:637-645, sanity-checked and passes: `z_min ≤ REDSHIFT+REDSHIFT_ERROR`
→ `0.0536 ≤ 0.0876` ✓; `z_max ≥ REDSHIFT−REDSHIFT_ERROR` → `0.0778 ≥ 0.0180` ✓ — 6791151 is a
genuine redshift/sky candidate that is excluded specifically by the *mass* filter, matching the
claim card's characterization.)

Mass filter (`mass_filter_mask`, handler.py:663-673), `sigma_multiplier = 1.5`,
`mass_filter_sigma = "symmetric"` (⇒ `_bh_mass_error_multiplier = 1.5`, confirmed from
`bt_900121_meta.json`'s `a22_stamp.mass_filter_sigma: "symmetric"`):

```
cond1:  (M_z − 1.5·σ_Mz) / (1+z_max)  ≤  BH_MASS + 1.5·BH_MASS_ERROR
cond2:  BH_MASS − 1.5·BH_MASS_ERROR   ≤  (M_z + 1.5·σ_Mz) / (1+z_min)
PASS (readmitted) ⟺ cond1 AND cond2
```

Numbers substituted (current, 0.24-dex-only budget, `BH_MASS_ERROR = 291758.995`):

```
cond1:  1237046.502  ≤  223872.114 + 1.5·291758.995 = 661510.606   →  FALSE
cond2:  223872.114 − 1.5·291758.995 = −213766.378  ≤  1265461.692  →  TRUE
```

`cond1` is FALSE ⇒ **candidate 6791151 FAILS the window under the current budget** (confirmed, as
stated in the task).

## 4. Window test under the FULL (0.55-dex) R&V15 budget

Per the task: add the excluded 0.50-dex virial-measurement component in quadrature, i.e. add
`(0.50·ln10)^2` inside the same sqrt (leaving every other term, including `sigma_int=0.24·ln10`,
untouched):

```
BH_mass_error_inflated = BH_mass * sqrt( sigma_int^2 + sigma_meas^2 + d_alpha^2
                                          + (ln(M_*/10)*d_beta)^2 + (beta/M_* * M_*_err)^2 )
  sigma_meas = 0.50*ln(10)
```

Reduction check: setting `sigma_meas → 0` in this expression reproduces the current-budget formula
term-for-term (same function, one added term set to zero) — confirmed by construction, and the two
values below are computed from literally the same Python expression with `sigma_meas_term_sq` set
to `0.0` vs `sigma_meas**2`.

- `BH_mass_error_inflated` = **389299.8873277455** M☉ (ratio to current: ×1.3343)
- Sanity: `sqrt(0.24² + 0.50²) = 0.55462 dex` ≈ the literature 0.55 dex total (small residual from
  0.24/0.50 not being exact to more digits) ✓

Numbers substituted (inflated, `BH_MASS_ERROR = 389299.887`):

```
cond1:  1237046.502  ≤  223872.114 + 1.5·389299.887 = 807821.945   →  FALSE
cond2:  223872.114 − 1.5·389299.887 = −360077.717  ≤  1265461.692  →  TRUE
```

`cond1` is STILL FALSE ⇒ **candidate 6791151 is NOT readmitted even under the full 0.55-dex R&V15
budget.** Not close, either: the GW-side floor `1,237,047` M☉ is more than 3× the inflated host
mass ceiling `807,822` M☉ (host mass `223,872` M☉ vs. required floor `1,237,047` M☉ is a ~5.5×
mass mismatch even before subtracting the error term).

## 5. Margin — how far off is it?

Binding constraint is `cond1` in both cases (host mass is far below the GW-side floor; `cond2` is
never close). Solve `cond1` for the `BH_MASS_ERROR` at which it just turns into equality:

```
required_BH_MASS_ERROR = ( (M_z − 1.5σ_Mz)/(1+z_max) − BH_MASS ) / 1.5
                        = (1237046.502 − 223872.114) / 1.5
                        = 675449.592 M☉
```

- **Factor over the CURRENT (0.24-dex-only) `BH_MASS_ERROR`:** `675449.592 / 291758.995` = **2.315×**
- **Factor STILL needed beyond the FULL 0.55-dex-inflated `BH_MASS_ERROR`:** `675449.592 / 389299.887` = **1.735×**
- Expressed as an equivalent additional ln-space scatter term (beyond the current 0.24-dex budget,
  same formula): solving `(required_err/BH_mass)^2 = sigma_int^2 + d_alpha^2 + (ln(M_*/10)d_beta)^2
  + (beta/M_* M_*_err)^2 + sigma_extra^2` for `sigma_extra` gives **`sigma_extra ≈ 1.182 dex`** — i.e.
  the mass-relation scatter would need to carry roughly `sqrt(0.24² + 1.18²) ≈ 1.20 dex` of total
  budget for 6791151 to clear the window, more than double the full literature R&V15 total of 0.55
  dex and with no citable justification for a term that large.

## 6. Answer to the decisive question

**NO.** With the full 0.55-dex R&V15 budget (adding the previously-excluded 0.50-dex virial
measurement component in quadrature to the existing 0.24-dex intrinsic-scatter budget), candidate
6791151 is **still excluded** by the eligibility window for seed 900121 event 20 — `cond1` remains
false by a wide margin (required `BH_MASS_ERROR` is **2.315×** the current value and **1.735×** the
already-inflated value; equivalently ~1.18 additional dex of scatter, more than double the entire
literature budget). The 0.50-dex measurement-error omission is real and worth fixing on its own
merits (raises `BH_MASS_ERROR` by a factor 1.334×), but it is **not sufficient, on its own, to
explain or reverse this exclusion**. Per the task framing: this is a genuine mass mismatch — the
exhibit does not survive as a "window defect with a demonstrated fix" via this lever; part (b)'s
scope narrows to the window's own ε-derivation, independent of the 0.50-dex question.
