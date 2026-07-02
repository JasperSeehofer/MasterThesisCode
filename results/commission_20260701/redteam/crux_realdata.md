# Crux experiment — real seed600 dataset: does the H0 pipeline recover the injected value?

Independent red-team run. Dataset: `/tmp/seed600_local/simulations/` (seed600, 3375 detected
EMRI events; per-event Cramér–Rao covariances with measurement scatter). Injected/fiducial
H0 determined independently below.

## Injected / fiducial H0

- Injection pool filenames: `injection_h_0p73_task_*.csv` → sampled at **h = 0.73**.
- `prepared_cramer_rao_bounds.meta.json`: `sampling_method = multivariate_normal`, `seed = 1000599`,
  `input_rows = 3375` (the prepared CRB carries MVN measurement scatter around the Fisher mean).
- `constants.py`: `H = 0.73` (fiducial Hubble constant used by the simulation).
- ⇒ **Injected H0 = 0.73** (i.e. H0 = 73 km/s/Mpc). A well-behaved posterior should peak near 0.73.

## Method / wiring (faithfulness)

The `--evaluate` CLI reads its inputs from **cwd-relative** paths, not from the `<working_dir>`
argument:
- `simulations/prepared_cramer_rao_bounds.csv`  (per-event measurements + covariances, WITH scatter)
- `simulations/cramer_rao_bounds.csv`           (loaded as `true_cramer_rao_bounds`; not used in the posterior)
- `simulations/injections/`                      (`SimulationDetectionProbability` injection pool)
- `./master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv`  (GLADE+ reduced catalogue)
- package-relative `m_th_map_nside32.npy`        (HEALPix completeness cache)
It writes `simulations/posteriors/h_XX.json` and `simulations/posteriors_with_bh_mass/h_XX.json`
(cwd-relative). `--combine` (`combine_posteriors`, strategy `physics-floor`) sums per-event
`log L_i(h)` across events → softmax over the h-grid (**flat prior**; `combine_log_space` correctly
does NOT re-apply an outer `−N·log D(h)` — D(h) already lives inside each per-event `L_comp`).

### BLOCKER found and resolved (faithfulness caveat #1)

The on-disk `reduced_galaxy_catalogue.csv` in the repo has **only 6 columns**
`(RA, Dec, z, z_err, stellar_mass, stellar_mass_err)`, but the current reader
(`_reduced_catalog_column_names`) expects **8**
`(RA, Dec, APPARENT_B_MAG, z, z_err, stellar_mass, stellar_mass_err, REDSHIFT_FLAG)`.
Pandas fills the 8 names left-to-right onto the 6 data columns, so `STELLAR_MASS_ABSOULTE_ERROR`
(= internal `BH_MASS_ERROR`) becomes **NaN for every row**. The prune mask
`BH_MASS ± BH_MASS_ERROR ∈ [M_min, M_max]` is then NaN⇒False everywhere → the galaxy catalogue
is **emptied** → `ZeroDivisionError` in `_show_catalog_information`. The current package therefore
**cannot load its own committed reduced catalogue**; the original seed600 posteriors (38 h-values,
3335 events, per the combine logs) were produced elsewhere (cluster) with a correct 8-column
catalogue and rsynced back. This local catalogue file is not git-tracked.

Resolution (faithful): I streamed a column-aligned 8-column catalogue
(`awk`: insert a dummy `APPARENT_B_MAG` at col 3, append `REDSHIFT_FLAG=3`), preserving the real
`z / z_err / stellar_mass / stellar_mass_err` columns byte-for-byte. This is faithful for the
evaluate likelihood because (a) `APPARENT_B_MAG` is consumed ONLY by the m_th-map builder
(`pixel_completeness.py`), which is bypassed at evaluate time — `from_cache_or_build()` `np.load`s
the existing `m_th_map_nside32.npy` cache; and (b) `REDSHIFT_FLAG` is used only in the raw-catalogue
writer, never in the likelihood. After the fix the catalogue prunes to **9,060,017 galaxies** with
all `BH_MASS_ERROR` finite (z∈[0.003,1.33]). Runs use a shadow-symlink overlay so the repo's
committed file is untouched.

### Faithfulness confirmations (from the evaluate logs)

- Loaded **3375** detections; SNR≥20 keeps all 3375; d_L-relative-error<0.1 quality filter → **3355** events.
- Injection P_det: "Pooled 72000 injection events from 80 files (h=0.73)"; 6 equal-|sin β| sky bands;
  **P_det grid coverage 100%** (3355/3355 within 4σ d_L bounds). Estimator `local_linear` (F4-v2 default).
- Partition norm at h=0.73: `w_G = beta_G/D(h) = 0.8175`.
- **Diagnostic:** 3331/3355 events emit "numerator=0.000" (>5% quadrature weight outside P_det grid);
  i.e. the in-catalogue completion numerator is ~0 for ~99% of events, yet per-event `L_i(h)` is
  finite and positive (values O(10–10^3)). The constraint is carried by the completion/normalisation
  terms far more than by in-catalogue host matches.

### Compute note

Per-h evaluate is dispatch-bound (a serial loop over 3355 detections, each doing a small
`pool.starmap` over candidate hosts), ~5 min/h regardless of worker count. To stay faithful while
fitting the session, I drove the **production** `BayesianStatistics().evaluate()` per h in-process
(single catalogue load, exact CLI defaults: pdet_dl_bins=60, pdet_mass_bins=40,
estimator=local_linear, fisher_cond_threshold=1e16), split across 4 processes (prod/catonly ×
two h-subsets). This is the same per-event definition the CLI uses; only the catalogue load is
shared across h. Combined with `combine_posteriors` (physics-floor) and cross-checked with the
canonical raw Σ log L combiner.

## Results

Production `BayesianStatistics().evaluate()`, 494-event seed600 subsample, 7-point H0 grid
[0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86], `combine_posteriors` (physics-floor). Injected H0 = 0.73.

### Original crux (commission-base, pre-fix) — `crux_results.json`

| mode | MAP | mean | edge mass | railed |
|---|---|---|---|---|
| production (default) | **0.86** | 0.860 | 1.000 | yes (upper edge) |
| `--catalog_only` | **0.73** | 0.737 | 3.7e-5 | no (peaked: 0.77@0.73, 0.22@0.76) |

**Same real data, same events, same catalogue** — dropping the completion term / switching to the
local self-normalized ratio moves the mode from the 0.86 grid edge to a clean peak at the injected 0.73.
The rail is normalization-driven, not information-starvation. (The full stored 3375-event result also rails to 0.86.)

### De-rail progression (branch `physics/derail-completion-4pi` + `normalization_mode`) — `derail_matrix_results.json`

Each fix is applied to the production path and re-run on the identical data:

| step | in-catalogue normalization | completion | MAP | mean | posterior | railed |
|---|---|---|---|---|---|---|
| pre-4π | global-denominator single ratio | peak sky-density B_num | **0.86** | 0.860 | 100% @ 0.86 | ↑ rail |
| 4π only (`prod_global`) | global-denominator single ratio | **1/(4π) B_num** (cb16142) | **0.60** | 0.600 | 100% @ 0.60 | ↓ rail |
| fix #2 (`local_ratio`) | **local ratio-of-sums** (Gray A.9/A.10) | 1/(4π) B_num | **0.73** | 0.730 | 98% @ 0.73 | no (peaked) |
| fix #1 (`volume_deconv`) | local ratio + **dV_c/(1+z) host-z prior** | 1/(4π) B_num | **0.73** | 0.740 | 68% @ 0.73, 31% @ 0.76 | no (peaked) |

- The **1/(4π) completion fix alone flips the rail 0.86 → 0.60** (upper → lower edge): it removes the
  ~1640x-inflated completion term (confirming that defect) but exposes the still-uncorrected
  in-catalogue normalization — necessary but **not sufficient** (the sign-flip mechanism, §4).
- **Either principled normalization fix (#2 local ratio, or #1 local ratio + volume-prior host-z) de-rails
  to a peaked interior posterior recovering the injected 0.73.** `volume_deconv`'s mean sits +0.010 above
  `local_ratio` (0.740 vs 0.730) — the volume prior removes the bare-Gaussian numerator's Eddington-in-z
  low bias (D2 measured -2-3% synthetically, smaller here since most GLADE hosts are near-spec-z).
