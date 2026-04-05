# EMRI Bayesian H₀ Inference

[![CI](https://github.com/JasperSeehofer/MasterThesisCode/actions/workflows/ci.yml/badge.svg)](https://github.com/JasperSeehofer/MasterThesisCode/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://jasperseehofer.github.io/MasterThesisCode/)

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
| `master_thesis_code/bayesian_inference/` | `BayesianInference` — likelihood, posterior over H₀ |
| `master_thesis_code/physical_relations.py` | Cosmological distance functions |
| `master_thesis_code/constants.py` | Physical constants and simulation configuration |
| `master_thesis_code/cosmological_model.py` | EMRI event rate model, H₀ evaluation orchestration |
| `master_thesis_code/galaxy_catalogue/` | GLADE galaxy catalog interface (BallTree lookups) |
| `scripts/` | Utility scripts for post-processing simulation output |
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

**Luminosity distance** ([Hogg 1999](#hogg1999), Eq. 16):

$$d_L(z,H_0) = \frac{c(1+z)}{H_0} \int_0^z \frac{dz'}{E(z')}$$

Evaluated analytically via the Gauss hypergeometric function $\,{}_2F_1(1/3,\,1/2;\,4/3;\,-\Omega_m(1+z)^3/\Omega_\Lambda)$ for flat ΛCDM.

**LISA noise-weighted inner product** ([Cutler & Flanagan 1994](#cf1994)):

$$\langle h_1 \mid h_2 \rangle = 4\,\mathrm{Re}\sum_{\alpha \in \{A,E\}} \int_{f_\mathrm{min}}^{f_\mathrm{max}} \frac{\tilde{h}_1^\alpha(f)\,\tilde{h}_2^{\alpha*}(f)}{S_n^\alpha(f)}\,df$$

**Fisher information matrix** ([Vallisneri 2008](#vallisneri2008)):

$$\Gamma_{ij} = \left\langle \frac{\partial h}{\partial \theta_i} \,\middle|\, \frac{\partial h}{\partial \theta_j} \right\rangle, \qquad \Sigma = \Gamma^{-1}$$

where $\Sigma_{ii}^{1/2}$ is the Cramér–Rao lower bound on parameter $\theta_i$.

**Signal-to-noise ratio:**

$$\rho = \sqrt{\langle h \mid h \rangle}$$

**H₀ likelihood** (per event; [Chen et al. 2018](#chen2018)):

$$\mathcal{L}(H_0) = \frac{\displaystyle\int p_\mathrm{GW}(\hat{d}_L \mid z,H_0)\,p_\mathrm{det}(z,H_0)\,p(z \mid \mathrm{cat})\,dz}{\displaystyle\int p_\mathrm{det}(z,H_0)\,p(z \mid \mathrm{cat})\,dz}$$

where $p_\mathrm{GW}$ is a Gaussian in $d_L$ with fractional width $\sigma/d_L$, and the
denominator corrects for Malmquist-type selection bias.

---

### Model Assumptions

| Assumption | Where used | Notes |
|---|---|---|
| Flat ΛCDM ($w_0=-1$, $w_a=0$) | All distance integrals | `physical_relations.py`; wCDM infrastructure exists but is untested |
| Gaussian measurement noise on $d_L$ | GW likelihood | Width hardcoded to 10% of $d_L$ (see Limitation 6) |
| SNR threshold as detection proxy | `parameter_estimation.py`, `bayesian_inference.py` | Threshold = 20; detailed waveform-parameter dependence not captured |
| Uniform prior on $H_0$ | `bayesian_inference.py` | No prior declared; implicitly flat over $[H_\mathrm{min}, H_\mathrm{max}]$ |
| Synthetic galaxy catalog | Pipeline A | Galaxies drawn uniformly in log-mass and from comoving volume; GLADE used in Pipeline B |
| LISA mission duration 5 years | Waveform generation | Feeds directly into TDI response and sky-averaged sensitivity |

---

### Known Limitations

Items are ordered by severity. Each references the specific source location and carries a
status tag: **bug** (incorrect formula or logic), **design choice** (deliberate simplification),
or **pending fix** (acknowledged issue not yet addressed).

#### Limitation 1 — Comoving volume formula is wrong  `[bug · CRITICAL]`
**File:** `master_thesis_code/datamodels/galaxy.py:121`

```python
# Current (wrong)
cv_grid = 4 * np.pi * (SPEED_OF_LIGHT / h0) ** 3 * cumulative_integral**2

# Correct (Hogg 1999, Eq. 28)
cv_grid = (4/3) * np.pi * (SPEED_OF_LIGHT / h0) ** 3 * cumulative_integral**3
```

Two errors: the exponent on the integral is 2 instead of the correct 3, and the prefactor
is $4\pi$ instead of $4\pi/3$. The comoving volume therefore scales as $z^2$ at low
redshift instead of the correct $z^3$. Every downstream use — galaxy sampling, background
weight in the likelihood — draws from the wrong redshift distribution.

Reference: Hogg (1999), arXiv:astro-ph/9905116, Eq. (28).

---

#### Limitation 2 — Fisher matrix uses first-order forward difference, not five-point stencil  `[bug · HIGH]`
**File:** `master_thesis_code/parameter_estimation/parameter_estimation.py:336`

`compute_fisher_information_matrix()` calls `finite_difference_derivative()`, which
implements an $O(\varepsilon)$ forward difference. A correct `five_point_stencil_derivative()`
method implementing the $O(\varepsilon^4)$ formula

$$\frac{\partial h}{\partial\theta} \approx \frac{-h(\theta+2\varepsilon)+8h(\theta+\varepsilon)-8h(\theta-\varepsilon)+h(\theta+2\varepsilon)}{12\varepsilon}$$

exists in the same class but is never called from the Fisher computation.
The class docstring incorrectly claims a five-point stencil is used.
Cramér–Rao bounds are therefore less accurate than advertised; the bias grows as
$\varepsilon$ increases.

References: Vallisneri (2008), arXiv:gr-qc/0703086; Cutler & Flanagan (1994), PRD 49, 2658.

---

#### Limitation 3 — Galactic confusion noise absent from LISA PSD  `[bug · MEDIUM]`
**File:** `master_thesis_code/LISA_configuration.py` (PSD functions); `master_thesis_code/constants.py:77–83` (unused parameters)

`power_spectral_density_a_channel()` implements only the instrumental (OMS + test-mass)
noise. The galactic confusion-noise parameters `LISA_PSD_A`, `LISA_PSD_ALPHA`, `LISA_PSD_F2`,
etc. are defined in `constants.py` but never used. Galactic confusion noise dominates the
LISA sensitivity from ~0.1 to ~3 mHz — exactly where many EMRI signals peak — so computed
SNRs and Fisher bounds are systematically too optimistic.

Reference: Babak et al. (2023), arXiv:2303.15929, Eq. (17) and Table 1.

---

#### Limitation 4 — wCDM parameters $w_0$, $w_a$ silently ignored  `[bug · MEDIUM]`
**File:** `master_thesis_code/physical_relations.py:72`

`dist()` accepts `w_0` and `w_a` but passes them only to `lambda_cdm_analytic_distance()`,
which ignores them — the hypergeometric formula is exact only for flat ΛCDM ($w_0=-1$,
$w_a=0$). No warning is raised. Any caller supplying non-fiducial dark-energy parameters
silently receives ΛCDM distances. The correct general formula would require numerical
integration via `hubble_function()`, which already implements the full CPL parameterisation.

Reference: Hogg (1999), arXiv:astro-ph/9905116, Eq. (14–16).

---

#### Limitation 5 — GW likelihood distance uncertainty hardcoded at 10%  `[design choice · MEDIUM]`
**File:** `master_thesis_code/bayesian_inference/bayesian_inference.py`

The GW likelihood uses $\sigma_{d_L} = 0.1\,d_L$ for every event (via
`FRACTIONAL_LUMINOSITY_ERROR`). The simulation already computes the actual per-source
Cramér–Rao bound on $d_L$ (stored as `delta_luminosity_distance_delta_luminosity_distance`
in the CSV output). Using a source-by-source uncertainty from the Fisher matrix would make
nearby, well-localised events contribute more sharply to the $H_0$ posterior, as they
physically should.

---

#### Limitation 6 — Two Bayesian pipelines with inconsistent formulations  `[design choice · IMPORTANT]`
**Files:** `master_thesis_code/bayesian_inference/bayesian_inference.py` (Pipeline A);
`master_thesis_code/cosmological_model.py` (`BayesianStatistics`, Pipeline B)

Pipeline A (synthetic catalog) marginalises over a continuous redshift grid with a scalar
Gaussian likelihood on $d_L$ and a simplified selection correction.
Pipeline B (GLADE catalog) constructs a full multivariate Gaussian likelihood over
$(φ, θ, d_L/d_L^\mathrm{pred})$ using the actual Fisher-matrix covariance and a
KDE-based detection-probability estimate.
The two formulations are not mathematically equivalent and would yield different posteriors
on identical data. Pipeline B is the science-grade implementation; Pipeline A is a
development-only cross-check. This distinction is not documented in the code.

---

#### Limitation 7 — Outdated fiducial cosmological parameters  `[design choice · LOW]`
**File:** `master_thesis_code/constants.py:29–30`

| Parameter | Code | Planck 2018 |
|---|---|---|
| $\Omega_m$ | 0.25 | 0.3153 ± 0.0073 |
| $\Omega_\Lambda$ | 0.75 | 0.6847 ± 0.0073 |
| $h$ (simulation) | 0.73 | 0.6736 ± 0.0054 |

The WMAP-era values used here differ from the current Planck 2018 best fit by ~2σ in
$\Omega_m$ and ~1σ in $h$. For a simulation intended to represent realistic LISA science
the fiducial point should be updated.

Reference: Planck Collaboration (2018), arXiv:1807.06209, Table 2.

---

#### Limitation 8 — Galaxy redshift uncertainty has non-standard scaling  `[design choice · LOW]`
**File:** `master_thesis_code/datamodels/galaxy.py:64`

```python
redshift_uncertainty = min(0.013 * (1 + redshift) ** 3, 0.015)
```

The $(1+z)^3$ scaling grows rapidly and hits the cap of 0.015 at $z \approx 0.14$, so all
galaxies above that redshift are assigned identical uncertainty. Standard photometric
redshift errors scale as $\sigma_z \approx 0.05(1+z)$; spectroscopic errors as
$\sigma_z \approx 0.001(1+z)$. No reference for the cubic form is provided.

---

### What Is Mathematically Correct

The following components have been verified against their cited references:

- **Luminosity distance hypergeometric integral** (`physical_relations.py`): the form
  $\,{}_2F_1(1/3,1/2;4/3;-\Omega_m(1+z)^3/\Omega_\Lambda)$ is the correct analytic
  solution for flat ΛCDM. ✓
- **LISA instrumental PSD (A/E channels)** (`LISA_configuration.py`): matches Babak et al.
  (2023), arXiv:2303.15929, Eqs. (8)–(11), excluding galactic confusion noise. ✓
- **Noise-weighted inner product** (`parameter_estimation.py`): the factor-of-4 prefactor
  and one-sided PSD convention are correct; FFT normalisation is handled correctly via
  `trapz` over the frequency axis. ✓
- **Five-point stencil formula** (`five_point_stencil_derivative` in `parameter_estimation.py`):
  the coefficients $(-1, 8, -8, 1)/12\varepsilon$ are the correct $O(\varepsilon^4)$
  centred finite difference — the issue is only that this method is not called by the
  Fisher matrix computation (see Limitation 2). ✓
- **Bayesian selection-effects correction** (Pipeline A): the ratio
  $\text{numerator}/\text{denominator}$ where the denominator integrates
  $p_\mathrm{det}(z,H_0)\,p(z|\mathrm{cat})$ correctly implements the Loredo–Mandel
  selection-bias correction for the marginalised likelihood. ✓
- **Redshifted mass conversion**: $M_z = M(1+z)$ and its inverse are correctly implemented
  in `physical_relations.py`. ✓

---

### Bibliography

<a id="hogg1999"></a>
Hogg, D. W. (1999). *Distance measures in cosmology*. arXiv:astro-ph/9905116.

<a id="babak2023"></a>
Babak, S. et al. (2023). *LISA sensitivity and SNR calculations*. arXiv:2303.15929.

<a id="cf1994"></a>
Cutler, C. & Flanagan, É. E. (1994). Gravitational waves from merging compact binaries:
How accurately can one extract the binary's parameters from the inspiral waveform?
*Phys. Rev. D* **49**, 2658.

<a id="vallisneri2008"></a>
Vallisneri, M. (2008). Use and abuse of the Fisher information matrix in the assessment
of gravitational-wave parameter-estimation prospects. *Phys. Rev. D* **77**, 042001.
arXiv:gr-qc/0703086.

<a id="chen2018"></a>
Chen, H.-Y., Fishbach, M. & Holz, D. E. (2018). A two percent Hubble constant measurement
from standard sirens within five years. *Nature* **562**, 545–547. arXiv:1709.08079.

<a id="planck2018"></a>
Planck Collaboration (2020). Planck 2018 results VI: Cosmological parameters.
*Astron. Astrophys.* **641**, A6. arXiv:1807.06209.
