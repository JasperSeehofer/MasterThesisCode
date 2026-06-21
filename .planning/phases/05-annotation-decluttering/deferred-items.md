# Deferred Items — Phase 5 (Annotation & Decluttering Rollout)

Out-of-scope discoveries surfaced during Phase 5 execution. NOT fixed here
(scope boundary: only auto-fix issues DIRECTLY caused by this plan's changes).

## DEFER-05-01 — fig09 detection-efficiency raises on empty bin (pre-existing)

- **Where:** `master_thesis_code/plotting/convergence_plots.py:349`
  (`plot_detection_efficiency` -> `astropy.stats.binom_conf_interval`).
- **Symptom:** During figure regen against `simulations/_archive_v2_1_baseline`,
  `fig09_detection_efficiency` fails with `ValueError: n must be positive` from
  `binom_conf_interval` when a redshift bin has zero injections (`n_inj == 0`).
- **Cause:** `binom_conf_interval` is passed the full `n_inj` array including
  zero-count bins; astropy rejects `n <= 0`. The code already builds a `mask`
  for `n_inj > 0` but applies it only AFTER the CI call (sets masked CI to NaN),
  so the call itself still receives the zero entries.
- **Pre-existing:** line introduced in commit 7dc1421f (2026-04-02), untouched by
  Phase 5. `plot_detection_efficiency` was NOT modified by any Phase-5 task (this
  file's Phase-5 edits were limited to `plot_h0_convergence` + the import block).
- **Not caused by this plan; not in scope.** Suggested future fix: pass only the
  `mask`-selected positive bins into `binom_conf_interval` and scatter the result
  back, or guard with `n_inj.clip(min=1)` + post-mask to NaN. Reproduces only on
  data dirs whose injection/detection arrays produce an empty redshift bin.
