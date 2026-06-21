# Deferred Items — Phase 03 (New Static Figures)

Out-of-scope discoveries logged during execution. NOT fixed (Phase 3 is
plotting-only for fig21/fig22; these are pre-existing failures in other
figure paths).

## DI-03-01: `binom_conf_interval` "n must be positive" on seed200

- **Where:** `master_thesis_code/plotting/convergence_plots.py:332` (fig09
  detection-efficiency path), via `astropy.stats.binom_conf_interval`.
- **Trigger:** `--generate_figures results/figures_seed200` — one figure fails
  ("1 failed" in the manifest summary) because a detection-efficiency bin has
  `n == 0` injections, which `binom_conf_interval` rejects.
- **Scope:** PRE-EXISTING, unrelated to fig21/fig22 (Phase 3). Reproduces on
  `main` without any of this phase's changes.
- **Disposition:** Deferred. A future Phase-3-adjacent fix should guard the
  empty-bin case before calling `binom_conf_interval` (mask `n == 0` bins to
  NaN, mirroring the `plot_pdet_surface` zero-count guard).
