# Driver-vs-production iiib venue forensics — code-level field diff

Scope: pure code read + one data test. No runs launched. Comparands: S0-B truth
node (`hier_s0_driver.py`, commit 081b1f28, `results/.../graph1_20260901/
retrieved/s0b_run_20260902/s0a_seed900101/node_truth_iiib_sites2.2_nosmear/`)
vs R4b (`retrieved/r4b_comparand_sites22_2doff_20260904/`, production CLI
`python -m darksiren_emri --evaluate`).

## 1-2. Field-by-field table (driver `build_iiib_venue`/`run_theta_node` vs
production `--evaluate` resolution for R4b)

| Field | Driver (S0-B truth node) | Production (R4b `run_metadata.json`) | Verdict |
|---|---|---|---|
| `theta_b`, `theta_s` | (0.0, 1.0) — `THETA_NODES["truth"]` | (0.0, 1.0) | SAME |
| `theta_sites` | `"2.2"` (CLI `--theta-sites 2.2`) | `"2.2"` | SAME |
| `smear_global_selection` | `False` (`--smear off` → `_resolve_smear`) | `false` | SAME |
| `selection_in_completion_numerator` | `"fused"` (`IIIB_COMPLETION_CELL`) | `"fused"` | SAME |
| `completion_event_measure` | `"ratio"` (`IIIB_EVENT_MEASURE`) | `"ratio"` | SAME |
| `catalogue_numerator_survival` | `"phi"` (`IIIB_CATALOGUE_NUMERATOR_SURVIVAL`, explicit) | no CLI flag exists (`main.py`/`arguments.py` grep: zero hits) → `bs.evaluate()` default `"auto"` → resolves `"phi"` under `absolute_marginal` | SAME (by resolution) |
| `catalogue_global_selection` | `"phi"` (`IIIB_CATALOGUE_GLOBAL_SELECTION`, explicit) | `"phi"` | SAME |
| `catalogue_numerator_survival_2d` / `_center` | `"off"` / `"unset"` (hardcoded `cat_num_surv_2d_kwargs`, all configs) | `"off"` / `"unset"` | SAME — rules out the earlier mz_sel-default hypothesis; R4b already pins the counterfactual |
| `catalogue_leg_1d_mass_aware` | `"off"` (driver's own default, `common_kwargs`) | `"off"` (explicit CLI) | SAME — R4b explicitly pins the pre-flip counterfactual, matching the driver |
| `mass_filter_geometry` / `mass_filter_k` | `"linear"` / `1.5` (`IIIB_MASS_FILTER_*`, explicit) | `"linear"` / `1.5` | SAME |
| `mass_filter_sigma` | `"symmetric"` (never overridden; not exposed on driver CLI or production CLI) | `"symmetric"` (evaluate() default; `main.py` never forwards this kwarg either) | SAME |
| `theta_phi_divisor` / `sky_cone_k` | `"off"` / `1.5` (driver defaults, not passed on `--nodes` CLI) | `"off"` / `1.5` | SAME |
| `theta_zwindow` / `z_window_k` | `"off"` / `1.0` (driver defaults) | `"off"` / `1.0` | SAME |
| `normalization_mode` / `host_z_kernel` | `"absolute_marginal"` / `"volume_deconv"` (hardcoded from `c1d.PRODUCTION_FLAGS`, all configs) | `"absolute_marginal"` / `"volume_deconv"` | SAME |
| `eddington_m`, `sigma4d_mass_kernel`, `host_mass_kernel`, `catalogue_mass_error_scale` | **not forwarded at all** by `run_mirror_seed_inprocess` — `bs.evaluate()` internal defaults (`"on"`, `"point"`, `"auto"`, `1.0`) apply | `"on"`, `"point"`, `"auto"`, `1.0` (CLI defaults, unset) | SAME (coincidentally — driver never plumbs these, defaults happen to match) |
| `catalogue_mass_overlap`, `completion_b_scale` | `"production"` / `"derived"` (hardcoded from `c1d.PRODUCTION_FLAGS`) | `"production"` / `"derived"` | SAME |
| `pdet_z_resolved` | `True` (hardcoded in `run_mirror_seed_inprocess`) | `true` | SAME |
| `allow_low_pdet_coverage` | **`True`** (`run_mirror_seed_inprocess` signature default, line 3209 — never overridden by the driver's `common_kwargs`) | `false` (CLI default) | **DIFFERS**, but this only relaxes the stale-injection-pool *validation gate* (`allow_shallow_pool` in `SimulationDetectionProbability`, a raise-or-not check on `expected_z_max` coverage) and the ≥95%-coverage hard-raise; it does not touch `p_det` grid values or any per-candidate quantity. Both runs' pool passes the strict (95%/full-depth) check anyway since R4b ran to completion under `False`. **Cannot change WHICH candidates enter or their per-candidate value.** |
| GalaxyCatalogueHandler `M_min`/`M_max`/`z_max` | `M_SOURCE_FRAME_MIN`=1e4, `M_SOURCE_FRAME_MAX`=1e7, `z_max=HOST_DRAW_Z_MAX=1.5` (`c1d._load_galaxy_catalog_handler`) | same constants, `z_max=cosmological_model.max_redshift` = 1.5 (no `--max_redshift` override; R4b `max_redshift: null`) | SAME |
| CRB CSV / reduced catalogue | `c1d.CRB_CSV_PATH` / `c1d.REDUCED_CATALOGUE_PATH`, md5-pinned | same production input files (R4b working dir was seeded from the same pinned copies) | SAME (by construction/pin) |
| `h_values` passed to `bs.evaluate()` | **`(0.73,)`** — single value (`h_values=(H_GEN,)`, `H_GEN=H_TRUE=0.73`) | **`None`** (CLI `--h_values` unset; `main.py` passes `h_values=None` straight through) | DIFFERS in shape, but not decisive by itself (see next row) |
| **`h_bounds`, hence `cosmological_model.h.lower_limit` fed into `get_redshift_outer_bounds(h_min=...)`** | **`H_BOUNDS = (0.50, 0.86)`**, forwarded unconditionally to `run_mirror_seed_inprocess` for every node (line ~94-95, `common_kwargs`). Inside `run_mirror_seed_inprocess` (line 3470-3473): `eff_lo = h_bounds[0] = 0.50` (h_bounds is not None, so `h_values` is ignored for this purpose) → `bs.cosmological_model.h.lower_limit = min(0.6, 0.50) = 0.50` before `bs.evaluate()` is called. | **No widening at all.** `main.py`'s `evaluate()` function calls `BayesianStatistics()` → `bs.evaluate(...)` directly; it never touches `bs.cosmological_model.h.lower_limit`/`upper_limit`. `BayesianStatistics.__init__` sets `self.cosmological_model = LamCDMScenario()`, whose `h` parameter defaults to `lower_limit=0.6, upper_limit=0.86` (`cosmological_model.py:393-401`) — **stays 0.6, never 0.50**, for a single-h `--evaluate` run like R4b. | **DECISIVE DIFFER.** `h_min` is passed straight into `get_redshift_outer_bounds(distance, distance_error, h_min=..., h_max=..., sigma_multiplier=2.0)` (`bayesian_statistics.py:5729-5737`), whose `z_min = dist_to_redshift(d_L − 3σ, h_min)` (`physical_relations.py:563`) **feeds `get_possible_hosts_from_ball_tree`'s candidate window directly** — this is precisely a "which candidates enter" field, not a per-candidate-value field. Lower `h_min` ⇒ lower (more inclusive) `z_min` for the same observed distance ⇒ the driver's window is systematically wider at the low-z edge than production's. |

`B_num`, `D_tilde_phi`, `alpha_G_phi`, `den_log_term` are unaffected because
they are computed from the completion/global-normalization legs, which do not
re-run the ball-tree query per candidate window — consistent with the
established finding.

## 3. Verdict

One field explains the whole observed pattern: **`h_bounds` / the resulting
`cosmological_model.h.lower_limit` passed to `get_redshift_outer_bounds`.**
Driver = 0.50 (from module-level `H_BOUNDS = (0.50, 0.86)`, justified in the
driver's own comments as "a single-h caller reproducing a **full-grid** run's
L_cat must pass `h_bounds=(min(grid), max(grid))`" — but S0-B's registered
run is a **single-h (0.73-only)** evaluation being compared against a
**single-h** production run, so that full-grid justification does not apply
to this comparand pair). Production (R4b, and R4, both single-h) never
widens past the class default 0.6. This is a **candidate-window** field (item
3's "which candidates enter"), not a per-candidate-value field.

## 4. Data test (S0-B truth node vs R4b `event_likelihoods.csv`, 1588 events)

- `B_num`, `D_tilde_phi`, `alpha_G_phi`, `den_log_term`: 0 events differ (max
  rel ≈ 1.8e-14, float noise) — confirms the established finding.
- `L_cat_no_bh`: **1002/1588** events differ (this run pair; the task's
  quoted 1083/1588 is evidently from a different driver/comparand snapshot —
  same qualitative pattern).
  - **0** events: driver = 0, R4b ≠ 0 (driver never *drops* a candidate R4b
    has).
  - **157** events: driver ≠ 0, R4b = 0 — driver picks up candidates R4b's
    window excludes. Median event z = 0.69 (z ∈ [0.18, 0.89]), median
    candidate count (`C7_log10_n_cand_1d`) = 0 (i.e. exactly 1 candidate) —
    consistent with a single candidate sitting just inside the driver's
    wider low-z-edge window and just outside production's.
  - **906** events: both nonzero but values differ (ratio driver/R4b:
    median 1.011, 75th pct 1.033, but a heavy right tail to 4.7e7) —
    consistent with *partial*-window differences (an extra faint/edge
    candidate added to the driver's sum, or an integration boundary shift)
    rather than a uniform multiplicative factor.
- Covariate split (`covariate_table_iiib.csv`): the "differ" group has median
  `log10(n_candidates)` = 1.91 (~81 candidates/event) vs the "same" group's
  median 0.0 (1 candidate/event) — **events with dense candidate fields are
  the ones that differ**, exactly as expected for a candidate-window-width
  effect (more candidates ⇒ higher chance one sits in the
  [z_min(h=0.60), z_min(h=0.50)) sliver the driver's wider bound opens up).
  No comparable split was found on sky-cone area (`C5_log10_sky_area`),
  ruling out the sky-cone flag as the driver.

## 5. Discriminating single-job test

No CLI flag exists on either side to toggle this (the driver's `H_BOUNDS` is
a module constant, not exposed via `argparse`; production has no `h_bounds`
concept at all — a single-h `--evaluate` run simply never widens
`cosmological_model.h.lower_limit`). The test is a **driver code change**:
re-run the S0-B truth node (`--config iiib --theta-sites 2.2 --smear off
--seeds 900101 --nodes truth`) with `H_BOUNDS` temporarily set to `(0.60,
0.86)` instead of `(0.50, 0.86)` (or, equivalently, with `h_bounds=None`
passed through so `run_mirror_seed_inprocess` derives `eff_lo` from
`h_values=(0.73,)` alone, which floors below 0.6 only via the `min(...,
eff_lo)` call — passing `h_bounds=(0.6, 0.86)` explicitly is the cleaner
one-line change). If `L_cat_no_bh`/`combined_no_bh` become byte-identical to
R4b under that change, `h_bounds` is confirmed as the sole discriminator; no
other field in the table above is a candidate. Not implemented here (no fix,
per task scope).
