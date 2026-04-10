# EMRI Bayesian H₀ Inference

[![CI](https://github.com/JasperSeehofer/MasterThesisCode/actions/workflows/ci.yml/badge.svg)](https://github.com/JasperSeehofer/MasterThesisCode/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jasperseehofer.github.io/MasterThesisCode/)
[![Interactive Figures](https://img.shields.io/badge/figures-interactive-56B4E9)](https://jasperseehofer.github.io/MasterThesisCode/interactive/)

Dark siren inference of the Hubble constant H₀ from Extreme Mass Ratio Inspiral (EMRI)
gravitational wave events detected by the LISA space detector, using Bayesian analysis
with the GLADE+ galaxy catalog.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install system packages required before `uv sync`:

- **GSL** (GNU Scientific Library) — required by `fastemriwaveforms` at build time
  - Arch/Manjaro: `sudo pacman -S gsl`
  - Ubuntu/Debian: `sudo apt install libgsl-dev`
- **CUDA 12 toolkit** — required on the GPU cluster only

### Set up the environment

Dev machine (CPU only):

```bash
uv sync --extra cpu --extra dev
```

GPU cluster (CUDA 12):

```bash
uv sync --extra gpu
```

## Usage

**EMRI simulation** — generates SNR and Cramér-Rao bounds:

```bash
uv run python -m master_thesis_code <working_dir> --simulation_steps N [--simulation_index I] [--log_level DEBUG]
```

**Bayesian inference** — evaluate Hubble constant posterior:

```bash
uv run python -m master_thesis_code <working_dir> --evaluate [--h_value 0.73]
```

**SNR analysis only**:

```bash
uv run python -m master_thesis_code <working_dir> --snr_analysis
```

**Injection campaign** — generate detection probability grid data:

```bash
uv run python -m master_thesis_code <working_dir> --injection_campaign --simulation_steps N [--seed 42]
```

**Reproducibility:** Pass `--seed <int>` to fix the NumPy random state. When omitted,
a random seed is chosen, logged, and recorded in `run_metadata.json` in the working
directory. Always pass `--seed` for production campaigns.

### Data Requirements

The evaluation pipeline (Pipeline B) requires the GLADE+ galaxy catalog as
`reduced_galaxy_catalogue.csv` in `galaxy_catalogue/`. See
[GLADE+](https://glade.elte.hu/) for the source catalog. Expected columns are
documented in `galaxy_catalogue/handler.py`.

## Running on HPC

This project runs on bwUniCluster 3.0 (KIT) as SLURM array jobs. The `cluster/` directory
contains all scripts for environment setup, job submission, and failure recovery.

See [`cluster/README.md`](cluster/README.md) for the complete guide covering:
- First-time cluster setup
- Submitting simulation campaigns
- Monitoring and retrieving results
- Troubleshooting common failures

## Running Tests

```bash
uv run pytest -m "not gpu"          # dev machine (CPU only)
uv run pytest                       # cluster (GPU available)
```

## Documentation

Build and open the HTML docs:

```bash
uv run make -C docs html
open docs/build/html/index.html     # macOS
xdg-open docs/build/html/index.html  # Linux
```

## Project Structure

| Module | Description |
|--------|-------------|
| `master_thesis_code/parameter_estimation/` | Waveform generation, Fisher matrix, SNR, Cramér-Rao bounds |
| `master_thesis_code/LISA_configuration.py` | LISA antenna patterns, PSD, frame transformations |
| `master_thesis_code/datamodels/` | `ParameterSpace`, `Galaxy`, `GalaxyCatalog`, `EMRIDetection`, `Detection` |
| `master_thesis_code/bayesian_inference/bayesian_inference.py` | Pipeline A (dev cross-check): scalar Gaussian likelihood, synthetic catalog |
| `master_thesis_code/bayesian_inference/bayesian_statistics.py` | Pipeline B (production): Fisher covariance, GLADE catalog, completeness correction |
| `master_thesis_code/bayesian_inference/detection_probability.py` | Detection probability: `SimulationDetectionProbability` (IS estimator from injection campaigns) |
| `master_thesis_code/physical_relations.py` | Cosmological distance functions |
| `master_thesis_code/constants.py` | Physical constants and simulation configuration |
| `master_thesis_code/cosmological_model.py` | EMRI event rate model, H₀ evaluation orchestration |
| `master_thesis_code/galaxy_catalogue/` | GLADE galaxy catalog interface (BallTree lookups) |
| `master_thesis_code/galaxy_catalogue/glade_completeness.py` | GLADE+ catalog completeness estimation $f(z, H_0)$ |
| `master_thesis_code/plotting/` | All visualization code (factory functions, style, helpers) |
| `analysis/` | Post-hoc analysis: grid quality, importance sampling, injection yield, validation |
| `scripts/` | Utility scripts for post-processing simulation output |
| `scripts/bias_investigation/` | H₀ posterior bias diagnostic scripts and findings |
| `derivations/` | Physics derivation notes (dark siren likelihood) |
| `interactive/` | Interactive Plotly HTML figures (posteriors, Fisher ellipses, sky map, M_z improvement explorer) |
| `paper/` | LaTeX paper source (REVTeX4-2 PRD format) |
| `master_thesis_code_test/` | Test suite (mirrors source layout) |

---

## Scientific Background and Known Limitations

### Project Narrative

EMRIs are systems in which a stellar-mass compact object (~10 M☉) spirals slowly into a
massive black hole (10⁴–10⁷ M☉) at the centre of a galaxy. The inspiral takes years to
decades and produces a rich, multi-harmonic gravitational-wave signal that encodes 14 source
parameters in its phase evolution. LISA, the planned ESA space-based GW detector, will observe
EMRIs in the millihertz band where their signals accumulate over the full mission lifetime.
The deep phase coherence that makes EMRI signals complex to model also makes them extremely
precise distance probes: the signal amplitude provides the luminosity distance $d_L$ to
~few percent accuracy, while the Fisher-matrix Cramér–Rao bounds on all 14 parameters are
computed simultaneously from the noise-weighted inner product of waveform derivatives.

The Hubble constant inference follows the **dark-siren (statistical) method**: unlike
binary neutron star mergers, EMRIs produce no detectable electromagnetic counterpart, so
the host galaxy and its spectroscopic redshift $z$ are not directly known.
Instead, EMRI detections are cross-matched against a galaxy catalog, and the posterior
$p(H_0 | \{d_L^{(i)}\})$ is obtained by marginalising each event's distance measurement
over the catalog redshift distribution, weighting by the LISA sky-localization probability
and correcting for Malmquist-type selection effects. With enough events (projected: tens to
hundreds per year of LISA observations), the dark-siren method can constrain $H_0$ to a
few percent, independent of the cosmic distance ladder.

---

### Key Equations

**Hubble function** (flat ΛCDM; $w_0 = -1$, $w_a = 0$):

$$E(z) = \frac{H(z)}{H_0} = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda}$$

**Luminosity distance** (Hogg 1999, Eq. 16):

$$d_L(z,H_0) = \frac{c(1+z)}{H_0} \int_0^z \frac{dz'}{E(z')}$$

Evaluated analytically via the Gauss hypergeometric function $\,{}_2F_1(1/3,\,1/2;\,4/3;\,-\Omega_m(1+z)^3/\Omega_\Lambda)$ for flat ΛCDM.

**LISA noise-weighted inner product** (Cutler & Flanagan 1994):

$$\langle h_1 \mid h_2 \rangle = 4\,\mathrm{Re}\sum_{\alpha \in \{A,E\}} \int_{f_\mathrm{min}}^{f_\mathrm{max}} \frac{\tilde{h}_1^\alpha(f)\,\tilde{h}_2^{\alpha*}(f)}{S_n^\alpha(f)}\,df$$

**Fisher information matrix** (Vallisneri 2008):

$$\Gamma_{ij} = \left\langle \frac{\partial h}{\partial \theta_i} \,\middle|\, \frac{\partial h}{\partial \theta_j} \right\rangle, \qquad \Sigma = \Gamma^{-1}$$

where $\Sigma_{ii}^{1/2}$ is the Cramér–Rao lower bound on parameter $\theta_i$.

**Signal-to-noise ratio:**

$$\rho = \sqrt{\langle h \mid h \rangle}$$

**H₀ likelihood** (per event; Chen et al. 2018):

$$\mathcal{L}(H_0) = \frac{\displaystyle\int p_\mathrm{GW}(\hat{d}_L \mid z,H_0)\,p_\mathrm{det}(z,H_0)\,p(z \mid \mathrm{cat})\,dz}{\displaystyle\int p_\mathrm{det}(z,H_0)\,p(z \mid \mathrm{cat})\,dz}$$

where $p_\mathrm{GW}$ is a Gaussian in $d_L$ with fractional width $\sigma/d_L$, and the
denominator corrects for Malmquist-type selection bias.

**Completeness-corrected likelihood** (Gray et al. 2020, Eq. 9):

$$p_i(H_0) = f(z, H_0)\,\mathcal{L}_\mathrm{cat} + \bigl(1 - f(z, H_0)\bigr)\,\mathcal{L}_\mathrm{comp}$$

where $f(z, H_0)$ is the GLADE+ catalog completeness fraction at redshift $z$,
$\mathcal{L}_\mathrm{cat}$ is the catalog term (sum over cataloged galaxies), and
$\mathcal{L}_\mathrm{comp}$ is the completion term integrating over uncataloged hosts
weighted by a comoving volume prior. Implemented in
`bayesian_inference/bayesian_statistics.py` with completeness from
`galaxy_catalogue/glade_completeness.py`.

For known limitations, model assumptions, verified components, and scientific references,
see the [documentation](https://jasperseehofer.github.io/MasterThesisCode/limitations.html).
For the H₀ posterior bias investigation timeline, see [`docs/H0_BIAS_RESOLUTION.md`](docs/H0_BIAS_RESOLUTION.md).

---

## Citation

If you use this code, please cite:

> [Paper reference TBD — will be updated upon arXiv submission]

See also [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.
