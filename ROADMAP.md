# Roadmap

High-level development arc of the EMRI dark-siren H₀ inference pipeline. For scientific
assumptions and known limitations see [`docs/source/limitations.rst`](docs/source/limitations.rst)
and the published [documentation](https://jasperseehofer.github.io/MasterThesisCode/); the detailed
H₀-bias investigation is in [`docs/H0_BIAS_RESOLUTION.md`](docs/H0_BIAS_RESOLUTION.md).

**Status legend:** ✅ done · 🔄 in progress · 🔬 cluster/data-gated · 📝 planned

## Phases

- ✅ **EMRI simulation pipeline** — GPU-accelerated waveform generation (`fastemriwaveforms`), LISA
  TDI response and PSD (including galactic confusion noise), SNR, and Fisher-matrix Cramér–Rao
  bounds (five-point stencil derivatives).
- ✅ **Bayesian H₀ inference pipeline** — dark-siren posterior over H₀ from per-event Cramér–Rao
  bounds and the GLADE+ galaxy catalogue with completeness correction (Gray et al. 2020).
- ✅ **Pipeline correctness** — coordinate-frame handling (equatorial→ecliptic), redshifted-mass
  convention, catalogue likelihood aligned to Gray Eq. A.9/A.10, per-source distance errors.
- ✅ **Selection function** — detection-horizon survival estimator for `p_det` (bandwidth-free,
  h-invariant), replacing the earlier kernel form.
- ✅ **H₀ bias investigation & normalization** — completeness-normalization modes, host-z kernel,
  and peculiar-velocity marginalization; resolved by the `generator_marginal` +
  `--pdet_z_resolved` estimator redesign (`[PHYSICS]` `8fbb21e`/`a608c4f`, defaults flipped in
  `ce6338e`). See `H0_BIAS_RESOLUTION.md` §3.21–3.22.
- ✅ **Multi-seed campaign & empirical closure** (2026-07-26, valid-4 basis) — deep (z ≤ 1.5)
  five-seed production-stack campaign passed all pre-registered bias/width/sanity criteria on
  seeds 1000/2000/3000/90000 (seed900 dropped, author-ratified, for a diagnosed injection-pool
  provenance defect; non-blocking fixpool re-run in flight). Campaign NO-GO lifted; merge to
  main additionally gated on an independent adversarial (redteam) review, in progress. See
  `results/lcat_h_dependence_20260725/MULTISEED_READOUT_20260726.md`.
- 📝🔄 **Manuscript** — in preparation (2026); next up, blocked on the redteam verdict and the
  associated PR chain merging to `main`.

## Current focus

See [`STATE.md`](STATE.md).
