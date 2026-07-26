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
- **z_cut truncation scan (2026-07-25): ALL RAIL.** Consistently truncated re-evals
  (`--max_redshift` with B_num domain-matched to D(h) — `[PHYSICS]` 7d3573d + 276c8c7 on
  `feat/max-redshift-cli`) at z_cut ∈ {0.2, 0.3, 0.5} on the same seed1000 CRB all rail at
  h = 0.60 in both channels — **depth truncation (issue #30 option b) is empirically dead**.
  The untruncated z ≤ 0.2 *subset* closes at 0.729 while the truncated z_cut = 0.2 re-eval
  rails, isolating the rail in the h-dependence of the truncated selection/normalization
  structure (w_G = β_G/D) interacting with L_cat. Campaign relaunch remains NO-GO.

## Known open questions (tracked honestly)

- Whether the H₀ posterior peak is fully information-driven versus partly a normalization effect in
  the deepest-incompleteness regime — under investigation (`docs/H0_BIAS_RESOLUTION.md`).
- Two alternative host-kernel normalization modes (`volume_trunc`, `mass_trunc`) were implemented
  and empirically rejected; they are retained as documented, non-default experimental modes.
- Standing modelling choices are documented in `docs/source/limitations.rst` (flat-ΛCDM distance
  integrals, cosmology-constant vintage matched to the mock universe, redshift-uncertainty scaling).

## Next

- **Rail mechanism identified (2026-07-25): host misassociation.** Two independent
  investigations (empirical per-event decomposition, validated to ≤4.5e-13 against cluster
  diagnostics; structural audit vs Gray A9/Gair 2023/gwcosmo v2) show 91–100% of each rail
  event's tilt is the numerator GW-likelihood × host-z overlap: candidate balls contain only
  foreground galaxies (preferred h* ≈ 0.42–0.48, below the grid). volume_deconv is exonerated
  (exactly h-invariant); the ball-local selection denominator is a real-but-secondary
  discrepancy vs the references (1–14%). See `results/lcat_h_dependence_20260725/SYNTHESIS.md`.
- Decide and implement the **estimator redesign** (author physics decision, /physics-change):
  the fix domain is per-event catalogue-vs-dark weighting (Gray membership mixture /
  non-self-normalized catalogue mass) so impostor-only balls defer to the completion term.
  Campaign relaunch stays NO-GO until a redesigned estimator closes on the deep venue.
- Workspace note: `ws_extend` used the **last** available extension (expires 2026-09-23) — copy
  finals off before then.
