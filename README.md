# darksiren-emri

[![CI](https://github.com/JasperSeehofer/darksiren-emri/actions/workflows/ci.yml/badge.svg)](https://github.com/JasperSeehofer/darksiren-emri/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jasperseehofer.github.io/MasterThesisCode/)
[![Interactive Figures](https://img.shields.io/badge/figures-interactive-56B4E9)](https://jasperseehofer.github.io/MasterThesisCode/interactive/)

**End-to-end EMRI dark-siren H₀ inference — with the book that teaches it.**

darksiren-emri measures the Hubble constant H₀ from Extreme Mass Ratio Inspiral (EMRI)
gravitational-wave events detected by the LISA space observatory, using Bayesian analysis with
the GLADE+ galaxy catalogue and a completeness correction (Gray et al. 2020). It ships as
working, GPU-capable research code — and as
*[A Dark Siren Discovery Book](https://jasperseehofer.github.io/MasterThesisCode/book/)*, an
interactive, build-and-break narrative that walks a reader from "why does H₀ disagree with
itself" to a working estimator, including the wrong turns the project itself took along the way.
If you're new to dark sirens, start with the book. If you know the field, the pipeline below is
production code with a pre-registration discipline and a public validation record.

> **Development note.** This code is AI-*assisted* and human-*verified*. The author owns all
> scientific decisions; every change to physics is gated by a documented verification protocol
> (dimensional analysis, limiting-case checks, literature references, regression tests). See the
> `physics-change` protocol in [`CLAUDE.md`](CLAUDE.md).

## How this relates to other codes

The dark-siren / GW-population-inference community has excellent public tools already — this
project builds on their published methods and, in `docs/gates/`, on a line-by-line comparison
against their source. It doesn't replace them; it covers different ground.

| Code | What it's great at | Where darksiren-emri sits alongside it |
|---|---|---|
| [gwcosmo](https://git.ligo.org/lscsoft/gwcosmo) | The LVK production dark-siren pipeline for ground-based compact binaries — the reference implementation of the Gray et al. galaxy-catalogue method | darksiren-emri follows the same completeness-correction formalism, applied to LISA EMRIs instead of ground-based CBCs |
| [CHIMERA](https://github.com/CosmoStatGW/CHIMERA) | The most actively developed, best-documented, JAX/GPU-accelerated bright/dark/spectral-siren code, with the fullest reference docs in the field | darksiren-emri adds an EMRI waveform → Fisher-matrix layer upstream of the H₀ inference, and a narrative teaching book alongside the reference material |
| [icarogw](https://github.com/simone-mastrogiovanni/icarogw), [GWPopulation](https://github.com/ColmTalbot/gwpopulation) | General-purpose hierarchical population-inference toolkits — the right choice if you need flexible population models beyond a single H₀ estimator | darksiren-emri is narrower and more opinionated: one estimator, EMRI-specific, run to a published validation standard |
| [DarkSirensStat](https://github.com/CosmoStatGW/DarkSirensStat), [MGCosmoPop](https://github.com/CosmoStatGW/MGCosmoPop) | The methodological lineage CHIMERA grew from — modified-propagation and population-model extensions | Same formalism family; darksiren-emri's contribution is the LISA/EMRI branch, not a competing ground-based method |
| [StableEMRIFisher](https://github.com/perturber/StableEMRIFisher) | Focused, well-validated EMRI Fisher-matrix computation | darksiren-emri computes EMRI Fisher/CRB too, then carries the result all the way to a galaxy-catalogue H₀ posterior |

If your project needs ground-based dark sirens today, gwcosmo or CHIMERA are the mature choice.
If you're learning the field or working on LISA/EMRI dark sirens specifically, that's what
darksiren-emri is for.

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
uv run python -m darksiren_emri <working_dir> --simulation_steps N [--simulation_index I] [--log_level DEBUG]
```

**Bayesian inference** — evaluate Hubble constant posterior:

```bash
uv run python -m darksiren_emri <working_dir> --evaluate [--h_value 0.73]
```

**SNR analysis only**:

```bash
uv run python -m darksiren_emri <working_dir> --snr_analysis
```

**Injection campaign** — generate detection probability grid data:

```bash
uv run python -m darksiren_emri <working_dir> --injection_campaign --simulation_steps N [--seed 42]
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
| `darksiren_emri/parameter_estimation/` | Waveform generation, Fisher matrix, SNR, Cramér-Rao bounds |
| `darksiren_emri/LISA_configuration.py` | LISA antenna patterns, PSD, frame transformations |
| `darksiren_emri/datamodels/` | `ParameterSpace`, `Galaxy`, `GalaxyCatalog`, `EMRIDetection`, `Detection` |
| `darksiren_emri/bayesian_inference/bayesian_inference.py` | Pipeline A (dev cross-check): scalar Gaussian likelihood, synthetic catalog |
| `darksiren_emri/bayesian_inference/bayesian_statistics.py` | Pipeline B (production): Fisher covariance, GLADE catalog, completeness correction |
| `darksiren_emri/bayesian_inference/detection_probability.py` | Detection probability: `SimulationDetectionProbability` (IS estimator from injection campaigns) |
| `darksiren_emri/physical_relations.py` | Cosmological distance functions |
| `darksiren_emri/constants.py` | Physical constants and simulation configuration |
| `darksiren_emri/cosmological_model.py` | EMRI event rate model, H₀ evaluation orchestration |
| `darksiren_emri/galaxy_catalogue/` | GLADE galaxy catalog interface (BallTree lookups) |
| `darksiren_emri/galaxy_catalogue/glade_completeness.py` | GLADE+ catalog completeness estimation $f(z, H_0)$ |
| `darksiren_emri/plotting/` | All visualization code (factory functions, style, helpers) |
| `analysis/` | Post-hoc analysis: grid quality, importance sampling, injection yield, validation |
| `scripts/` | Utility scripts for post-processing simulation output |
| `scripts/bias_investigation/` | H₀ posterior bias diagnostic scripts and findings |
| `derivations/` | Physics derivation notes (dark siren likelihood) |
| `interactive/` | Interactive Plotly HTML figures (posteriors, Fisher ellipses, sky map, M_z improvement explorer) |
| `paper/` | LaTeX paper source (REVTeX4-2 PRD format) |
| `darksiren_emri_test/` | Test suite (mirrors source layout) |

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

A manuscript is **in preparation (2026)**. Until the arXiv submission, please cite the repository
directly via [`CITATION.cff`](CITATION.cff) (machine-readable citation metadata); the citation block
here will be updated with the paper reference on submission.
