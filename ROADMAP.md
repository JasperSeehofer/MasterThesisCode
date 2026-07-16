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
- 🔄🔬 **H₀ bias investigation & normalization** — completeness-normalization modes, host-z kernel,
  and peculiar-velocity marginalization. The residual bias and its empirical closure are under
  active investigation; some questions are data-gated (see `H0_BIAS_RESOLUTION.md`).
- 🔬 **Multi-seed campaign & empirical closure** — deep (z ≤ 1.5) injection campaign on the GPU
  cluster to establish the H₀ MAP / bias / coverage. The current pipeline is verified sound in
  code; empirical closure is pending the cluster run.
- 📝 **Manuscript** — in preparation (2026).

## Current focus

See [`STATE.md`](STATE.md).
