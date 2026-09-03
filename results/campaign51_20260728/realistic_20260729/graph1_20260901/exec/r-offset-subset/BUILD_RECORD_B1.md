# BUILD_RECORD_B1 — r-offset-subset blind covariate table (Builder B1)

Implements REGISTRATION_DRAFT.md §2 (C1–C11) + §3 Phase A + §6 gates G-1/G-2(vi)/G-3a/g-population/g-precision. This builder never opened `exec/rd-2d-bootstrap-jackknife/` or any `influence_*.csv` in this directory.

## G-1 pins

- `g1_crb_pin`: passed=True
- `g1_catalogue_pin`: passed=True
- `g1_csv_pin_iiib`: passed=True
- `g1_csv_pin_jr1`: passed=True
- `g1_dark_class_pin`: passed=True
- `g1_git_commit_pin`: passed=True
- **G-1 overall: GREEN**

## G-2 (vi) cone-radius anchor (R-MKER-6, re-run as this file's own byte-id)

- found_radius = 0.0014956979545757095
- expected_radius = 0.0014956979545757095
- |diff| = 0.0 < tol 1e-15
- **passed: True**

## G-3a decisive gate: 606-line "no catalog results" = exact-zero L_cat_no_bh

- **iiib**: n_no_catalog (log) = 606, n_exact_zero (L_cat_no_bh==0 at h=0.73) = 606, set-equality passed = **True**
- **joint_r1**: n_no_catalog (log) = 493, n_exact_zero (L_cat_no_bh==0 at h=0.73) = 493, set-equality passed = **True**

## Population / row counts

- production CRB total rows: 1590
- scored set size (gaps (1203, 1356)): 1588 (iiib), 1588 (joint_r1)

## Class-label counts (R8 table cross-check, g-precision)

- **iiib**: exact dark/hosted = 606/982; relative dark/hosted = 1241/347; C3c censored (floor applied) = 606; C6 NaN (n_1D==0) = 606
- **joint_r1**: exact dark/hosted = 493/1095; relative dark/hosted = 967/621; C3c censored (floor applied) = 493; C6 NaN (n_1D==0) = 493

## Column definitions (exact, as implemented)

| id | column | definition | source |
|---|---|---|---|
| C1 | `C1_in_catalog` | CRB `in_catalog` | CRB |
| C2 | `C2_hosted_exact` | NOT `is_dark_exact(L_cat_no_bh)` at h=h_true (`dark_class.py`) | event_likelihoods.csv |
| C3 | `C3_hosted_rel` | NOT `is_dark_relative(L_cat_no_bh, combined_no_bh, 1e-6)` at h=h_true | event_likelihoods.csv |
| C3c | `C3c_log10_f_cat` | log10(L_cat_no_bh/combined_no_bh); censored floor -320.0 where L_cat_no_bh==0 or combined_no_bh==0 (`C3c_censored` flag) | event_likelihoods.csv |
| C4 | `C4_z_gw` | `dist_to_redshift(luminosity_distance, h=h_true)` | CRB |
| C5 | `C5_log10_sky_area` | log10(pi * cone_radius(qS, phi_var, theta_var, cov, k)^2), k=1.5, `cone_radius` reused from `cone_loss_reads.py` | CRB |
| C6 | `C6_mass_window_retention` | n_2D/n_1D from "possible hosts found n_1D/n_2D"; NaN if n_1D==0 | log |
| C7 | `C7_log10_n_cand_1d` | log10(1 + n_1D) | log |
| C8 | `C8_cone_outside` | chord > radius (r-cone-loss OUT flag); NaN for non-in_catalog rows | CRB + catalogue |
| C9 | (alias of C1) | class G == in_catalog on production; no separate column written | — |
| C10 | `C10_log10_M` | log10(CRB `M`) | CRB |
| C10b | `C10b_low_M_timeout_bins12` | CRB `M` < 169568.12917853205 | CRB |
| C11 | `C11_log10_snr` | log10(CRB `SNR`) | CRB |

## Missing-value counts per column

**iiib**:
- `C8_cone_outside`: 1512
- `C6_mass_window_retention`: 606
**joint_r1**:
- `C8_cone_outside`: 1512
- `C6_mass_window_retention`: 493

## C10b testability (n >= 10 rule, disclosed for Phase C)

- iiib: n C10b=True = 5 (NOT-TESTED, n<10)
- joint_r1: n C10b=True = 5 (NOT-TESTED, n<10)

## Output files

- `covariate_table_iiib.csv`: sha256 `90c92026bb7fecff46e5a55e1e2c67a33b424e4b71611ee0d0854576b189f7b0`
- `covariate_table_joint_r1.csv`: sha256 `fc2eebe7fa66afbe2e35b0dd09c889be790511a6f5dabce3338969c849fcdf3a`
- both hashes also recorded in `covariate_table.sha256`

