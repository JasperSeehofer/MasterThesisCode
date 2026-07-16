# Project State

A short, human-readable snapshot of where the project is, for continuity across sessions and
machines. Detailed, ephemeral working notes are kept out of the repository by design (see the
`.gitignore` note on internal planning state); this file is the curated, durable surface.

**Last updated:** 2026-07-16

## What works today

- End-to-end EMRI simulation → SNR + Cramér–Rao bounds (GPU / cluster), and the CPU Bayesian H₀
  inference pipeline over the GLADE+ catalogue with completeness correction.
- Full CPU test suite green; `ruff` + `mypy` clean; docs and interactive figures deploy to GitHub Pages.
- `main` reflects the current, soundness-verified pipeline, including: zero-host pure-completion
  fallback, deep (z ≤ 1.5) population support, peculiar-velocity marginalization in the host-z
  kernel, an exact semi-analytic 2D denominator, and a value-preserving batched/fused likelihood
  evaluation (~3.8× faster).

## Current focus

- **Empirical H₀ closure** on the deep injection pool — a cluster run to confirm the pipeline's
  H₀ MAP / bias / coverage on the z ≤ 1.5 population. The code is verified sound; the empirical
  result is data-gated on cluster availability.

## Known open questions (tracked honestly)

- Whether the H₀ posterior peak is fully information-driven versus partly a normalization effect in
  the deepest-incompleteness regime — under investigation (`docs/H0_BIAS_RESOLUTION.md`).
- Two alternative host-kernel normalization modes (`volume_trunc`, `mass_trunc`) were implemented
  and empirically rejected; they are retained as documented, non-default experimental modes.
- Standing modelling choices are documented in `docs/source/limitations.rst` (flat-ΛCDM distance
  integrals, cosmology-constant vintage matched to the mock universe, redshift-uncertainty scaling).

## Next

- Run the deep-pool cluster closure job; fold the result into the manuscript (in preparation).
