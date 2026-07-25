# Project State

A short, human-readable snapshot of where the project is, for continuity across sessions and
machines. Detailed, ephemeral working notes are kept out of the repository by design (see the
`.gitignore` note on internal planning state); this file is the curated, durable surface.

**Last updated:** 2026-07-25

## What works today

- End-to-end EMRI simulation → SNR + Cramér–Rao bounds (GPU / cluster), and the CPU Bayesian H₀
  inference pipeline over the GLADE+ catalogue with completeness correction.
- Full CPU test suite green; `ruff` + `mypy` clean; docs and interactive figures deploy to GitHub Pages.
- `main` reflects the current, soundness-verified pipeline, including: zero-host pure-completion
  fallback, deep (z ≤ 1.5) population support, peculiar-velocity marginalization in the host-z
  kernel, an exact semi-analytic 2D denominator, and a value-preserving batched/fused likelihood
  evaluation (~3.8× faster).

## Current focus

- **Deep-venue rail (issue #30)** — the EXP-40 re-evaluation (`run_20260719_seed1000_exp40`,
  main @ ba2b381, #29 fallback active) confirmed the seed1000 posterior still rails at the lower
  grid edge (MAP h=0.60, both channels). Post-fix diagnostics re-attribute the rail: ~82% of the
  tilt is the host-found `L_cat` term (the 57.7% completion-fallback events are nearly h-inert),
  and z ≤ 0.3 subsets still rail — weakening pure depth truncation, strengthening the
  L_cat/Gray-mixture estimator path. See
  `results/campaign_phase2_runs/run_20260719_seed1000_exp40/FINDINGS_EXP40_20260725.md`.
- **In flight:** consistently truncated re-evals (`--max_redshift`, with the B_num completion
  numerator domain-matched to D(h) — `[PHYSICS]` 7d3573d + 276c8c7 on `feat/max-redshift-cli`)
  at z_cut ∈ {0.2, 0.3, 0.5} on the same seed1000 CRB (`run_20260725_seed1000_zcut*`).
  Campaign relaunch remains NO-GO until a truncated eval closes or the estimator is upgraded.

## Known open questions (tracked honestly)

- Whether the H₀ posterior peak is fully information-driven versus partly a normalization effect in
  the deepest-incompleteness regime — under investigation (`docs/H0_BIAS_RESOLUTION.md`).
- Two alternative host-kernel normalization modes (`volume_trunc`, `mass_trunc`) were implemented
  and empirically rejected; they are retained as documented, non-default experimental modes.
- Standing modelling choices are documented in `docs/source/limitations.rst` (flat-ΛCDM distance
  integrals, cosmology-constant vintage matched to the mock universe, redshift-uncertainty scaling).

## Next

- Read out the z_cut-truncated re-evals: if even truncated z_cut=0.3 rails, depth truncation is
  dead at the effective catalogue depth and the Gray-mixture / L_cat-h-dependence estimator
  upgrade becomes the primary path; if a z_cut closes near 0.73, decide the Paper-B depth framing
  (issue #30 decision D1) and relaunch the campaign seeds accordingly.
- Workspace note: `ws_extend` used the **last** available extension (expires 2026-09-23) — copy
  finals off before then.
