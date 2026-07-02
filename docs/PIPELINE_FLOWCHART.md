# Pipeline Flowchart — LISA EMRI Dark-Siren H₀ Inference

## Orientation

This codebase measures the Hubble constant H₀ from simulated LISA Extreme Mass-Ratio
Inspiral (EMRI) "dark siren" events. It is two independent pipelines joined by a CSV of
Cramér-Rao bounds (CRBs). **Pipeline 1** (GPU, on the cluster) samples EMRI events from a
cosmological rate model, resolves each to a GLADE+ host galaxy, builds a LISA waveform with
`few`, and computes the signal-to-noise ratio (SNR) and Fisher/Cramér-Rao parameter
covariance — writing one CSV row per detected event. **Pipeline 2** (CPU, multiprocessing)
reads those CRBs back, builds a per-event H₀ likelihood by marginalising over candidate host
galaxies with a galaxy-catalogue completeness correction (Gray et al. 2020) and a selection
denominator D(h), then combines the per-event posteriors into the final H₀ posterior. Both are
driven from a single CLI entry point, `master_thesis_code/main.py`, dispatched by the
`--simulation_steps`, `--injection_campaign`, `--evaluate`, and `--combine` flags parsed in
`arguments.py`.

### Legend

| Symbol | Meaning |
|--------|---------|
| 🟩 GPU / CUDA | Runs on the GPU partition (CuPy + `fastemriwaveforms-cuda12x`); guarded `cupy` fallback to NumPy on CPU |
| 🟦 CPU / MP | CPU only; uses a `multiprocessing` pool (forkserver/spawn) |
| ⬜ shared | Pure-Python/NumPy helper used by both pipelines |
| `file.py:symbol` | Owning source file and the function/class/method |
| ⟶ CSV / JSON | Data artefact written to / read from disk (the hand-off between pipelines) |

---

## Entry Point (shared)

```mermaid
flowchart TD
    CLI["python -m master_thesis_code &lt;dir&gt;<br/>__main__.py"] --> ARGS["Arguments.create() / validate()<br/>arguments.py"]
    ARGS --> MAIN["main()<br/>main.py:main"]
    MAIN --> MODEL["Model1CrossCheck(rng)<br/>cosmological_model.py"]
    MAIN --> CAT["GalaxyCatalogueHandler(M_min, M_max, z_max)<br/>galaxy_catalogue/handler.py"]
    MAIN -->|"--simulation_steps N"| P1["data_simulation()<br/>main.py:data_simulation"]
    MAIN -->|"--injection_campaign"| INJ["injection_campaign()<br/>main.py:injection_campaign"]
    MAIN -->|"--evaluate"| P2["evaluate()<br/>main.py:evaluate"]
    MAIN -->|"--combine"| COMB["combine_posteriors()<br/>bayesian_inference/posterior_combination.py"]
    MAIN -->|"--generate_figures"| FIG["generate_figures()<br/>main.py + plotting/"]
    MAIN -->|"--snr_analysis"| SNRA["snr_analysis()<br/>main.py:snr_analysis"]

    classDef gpu fill:#bbf7d0,stroke:#15803d,color:#000;
    classDef cpu fill:#bfdbfe,stroke:#1d4ed8,color:#000;
    classDef shared fill:#f1f5f9,stroke:#64748b,color:#000;
    class P1,INJ gpu;
    class P2,COMB cpu;
    class CLI,ARGS,MAIN,MODEL,CAT,FIG,SNRA shared;
```

`Model1CrossCheck` and `GalaxyCatalogueHandler` are constructed once and shared by whichever
pipeline the flags select. The simulation loop (Pipeline 1) and the injection campaign are
separate GPU run-modes; `--evaluate` is Pipeline 2.

---

## Pipeline 1 — EMRI Simulation (🟩 GPU / CUDA)

Produces the per-event Cramér-Rao bounds CSV. Driven by `main.py:data_simulation()`, looping
until `simulation_steps` events pass the SNR threshold.

```mermaid
flowchart TD
    START["data_simulation(simulation_steps, ...)<br/>main.py:data_simulation"] --> COMPL["from_cache_or_build()<br/>galaxy_catalogue/pixel_completeness.py"]
    COMPL --> FRAC["compute_global_catalog_fraction(F)<br/>dark_siren_injection.py"]
    FRAC --> PE["ParameterEstimation(PN5_AAK, use_gpu)<br/>parameter_estimation/parameter_estimation.py"]

    PE --> LOOP{"counter &lt; simulation_steps?"}

    subgraph SAMP["Event sampling + host resolution"]
        DRAW["draw_mixture_hosts(200, rng, ...)<br/>dark_siren_injection.py:draw_mixture_hosts"]
        DRAW --> RW["in-catalog (prob F):<br/>draw_rate_weighted_hosts()<br/>handler.py  P(g) ∝ R_eff(M_g)/(1+z_g)"]
        DRAW --> DK["out-of-catalog (prob 1-F):<br/>draw_dark_hosts() catalog_index = -1<br/>dark_siren_injection.py"]
        RW --> RATE["R_eff_per_mbh(M)<br/>emri_rate.py"]
    end

    LOOP -->|"next host"| SAMP
    SAMP --> SETP["randomize_parameters() +<br/>set_host_galaxy_parameters(host, h)<br/>datamodels/parameter_space.py (14 params)"]
    SETP --> PRESCREEN{"d_L &gt; 2.0 Gpc<br/>pre-screen?"}
    PRESCREEN -->|"yes"| LOOP
    PRESCREEN -->|"no"| QSNR["quick SNR (1-yr generator)<br/>compute_signal_to_noise_ratio(use_snr_check_generator=True)"]

    QSNR --> QGATE{"quick_snr ≥<br/>SNR_THRESHOLD·0.3?"}
    QGATE -->|"no"| LOOP
    QGATE -->|"yes"| FSNR["full SNR (5-yr waveform)<br/>compute_signal_to_noise_ratio()"]
    FSNR --> SGATE{"snr ≥ SNR_THRESHOLD (20)?"}
    SGATE -->|"no"| LOOP
    SGATE -->|"yes"| CRB["compute_Cramer_Rao_bounds()<br/>parameter_estimation.py"]

    CRB --> FISHER["compute_fisher_information_matrix()<br/>→ five_point_stencil_derivative()<br/>→ scalar_product_of_functions() (PSD inner product)"]
    FISHER --> SAVE["save_cramer_rao_bound(snr, host_index, in_catalog)<br/>buffered, flush every 5"]
    SAVE --> LOOP
    LOOP -->|"done"| OUT["⟶ cramer_rao_bounds*.csv<br/>(SNR + 14×14 covariance per event)"]

    subgraph WAVE["Waveform + noise (per SNR / derivative call)"]
        GEN["generate_lisa_response()<br/>few PN5_AAK + ResponseWrapper"]
        PSD["LisaTdiConfiguration.power_spectral_density()<br/>LISA_configuration.py (OMS + test-mass + confusion)"]
        GEN --> PSD
    end
    FSNR -.uses.-> WAVE
    CRB -.uses.-> WAVE

    classDef gpu fill:#bbf7d0,stroke:#15803d,color:#000;
    classDef shared fill:#f1f5f9,stroke:#64748b,color:#000;
    classDef data fill:#fde68a,stroke:#b45309,color:#000;
    class PE,QSNR,FSNR,CRB,FISHER,GEN,PSD gpu;
    class START,COMPL,FRAC,SAMP,DRAW,RW,DK,RATE,SETP shared;
    class OUT data;
```

**Notes**
- The event population is drawn by `Model1CrossCheck` (emcee MCMC over `(log₁₀M, z)` with
  `emri_distribution = dN_dz_of_mass(M, z) · R_emri(M)`; `cosmological_model.py`). In
  `data_simulation` the hosts are drawn directly from the catalogue via
  `draw_mixture_hosts` (in-catalogue rate-weighted + dark fraction); the emcee sampler is
  the population engine used by the `injection_campaign` run-mode.
- GPU is threaded explicitly via `use_gpu` into `ParameterEstimation`; the `_get_xp` /
  `_get_fft` helpers resolve CuPy vs NumPy once. `MemoryManagement` frees GPU blocks between
  steps. The 5-point stencil (`five_point_stencil_derivative`, default since Phase 10) is the
  O(ε⁴) Fisher-derivative method (Vallisneri 2008).

### Injection-campaign run-mode (🟩 GPU, SNR-only)

`main.py:injection_campaign()` reuses `Model1CrossCheck.sample_emri_events()` (emcee) and
`ParameterEstimation.compute_signal_to_noise_ratio()` but **skips the Fisher matrix**, storing
SNR for *all* sampled events (detected and sub-threshold) to `injections/injection_*.csv`.
That pool is the empirical input to the detection-probability estimator P_det in Pipeline 2.

---

## Pipeline 2 — Bayesian H₀ Inference (🟦 CPU / multiprocessing)

Driven by `main.py:evaluate()` → `BayesianStatistics.evaluate()`
(`bayesian_inference/bayesian_statistics.py`). Reads CRBs, builds per-event H₀ likelihoods,
writes per-event posterior JSONs.

```mermaid
flowchart TD
    EV["evaluate(galaxy_catalog, model, h, ...)<br/>bayesian_inference/bayesian_statistics.py:evaluate"] --> LOAD["⟶ load prepared_cramer_rao_bounds.csv"]
    LOAD --> FILT["SNR filter (≥20) + quality filter<br/>Detection / use_detection()"]
    FILT --> PDET["SimulationDetectionProbability(injections)<br/>simulation_detection_probability.py<br/>KDE P_det grid over (d_L, M_z)"]
    PDET --> GRID["_get_or_build_grid(h) + validate_coverage()"]

    GRID --> DEN["Selection denominator D(h)<br/>precompute_completion_denominator()"]
    DEN --> BETA["β_Ḡ(h) = ∫(1-f)·P_det·dV_c/(1+z)<br/>precompute_missing_completion_denominator()"]
    BETA --> BETAG["β_G(h) = D(h) − β_Ḡ(h)"]
    BETAG --> GCAT["global in-catalog selection Σ w_g D_g<br/>precompute_global_catalog_selection() (no-BH / with-BH)"]
    GCAT --> COMPL2["completeness f_k(z,Ω,h)<br/>pixel_completeness.from_cache_or_build()"]

    COMPL2 --> GAUSS["build per-event 3D/4D Gaussians<br/>(φ, θ, d_L [, M_z]) covariances + Fisher cond. flags"]
    GAUSS --> POOL["multiprocessing Pool (forkserver/spawn)<br/>child_process_init(...) per worker"]

    POOL --> PD["p_D(): loop detections<br/>bayesian_statistics.py:p_D"]
    PD --> HOSTS["get_possible_hosts_from_ball_tree()<br/>handler.py (sky Fisher ellipse + z + M_z cuts)"]
    HOSTS --> PDI["p_Di(): per-event likelihood<br/>bayesian_statistics.py:p_Di"]
    PDI --> SHL["pool.map → single_host_likelihood(z)<br/>numerator/denominator z-integrals per host"]
    SHL --> LIK["p_i = (β_G·L_cat + B_num) / D(h)<br/>catalog term + completion term"]
    LIK --> ACC["posterior_data[index].append(p_i)<br/>(with & without BH mass)"]
    ACC --> PD

    PD --> WRITE["⟶ posteriors/h_*.json<br/>⟶ posteriors_with_bh_mass/h_*.json<br/>⟶ diagnostics/event_likelihoods.csv"]
    WRITE --> CB["combine_posteriors(strategy)<br/>posterior_combination.py (--combine)"]
    CB --> FINAL["⟶ combined H₀ posterior<br/>(Σ log L_i, MAP + HDI)"]

    classDef cpu fill:#bfdbfe,stroke:#1d4ed8,color:#000;
    classDef shared fill:#f1f5f9,stroke:#64748b,color:#000;
    classDef data fill:#fde68a,stroke:#b45309,color:#000;
    class EV,FILT,PDET,GRID,DEN,BETA,BETAG,GCAT,COMPL2,GAUSS,POOL,PD,HOSTS,PDI,SHL,LIK,ACC,CB,FINAL cpu;
    class LOAD,WRITE data;
```

**Notes**
- `evaluate()` is called once per candidate `h_value`; the grid of H₀ values is built by
  running `--evaluate --h_value …` repeatedly (one JSON per h) and combining at the end.
- D(h) and the β / global-catalogue selection tables are **event-independent**, computed once
  per h before the worker pool spawns. The per-event work (host search + `single_host_likelihood`
  z-integrals) is what the pool parallelises.
- The per-event likelihood is the Gray et al. (2020) catalogue-mixture form: an in-catalogue
  term `L_cat` (sum over candidate hosts, rate-weighted, completeness-scaled by `f`) plus a
  completion/out-of-catalogue numerator `B_num` (weighted by `1−f`), all normalised by the
  selection denominator `D(h)`. Two variants are produced throughout: **without** and **with**
  the redshifted BH mass `M_z` as an extra discriminating dimension.

---

## Module Map — who owns what

```mermaid
flowchart LR
    subgraph ENTRY["Entry / config"]
        A1["main.py — CLI dispatch, data_simulation,<br/>injection_campaign, evaluate, generate_figures"]
        A2["arguments.py — Arguments (CLI flags, --use_gpu,<br/>--num_workers, --seed, --simulation_index)"]
        A3["constants.py — H=0.73, SNR_THRESHOLD=20,<br/>OMEGA_M=0.25, freq limits, paths"]
    end

    subgraph PHYS["Physics core (⬜ shared / 🟩 GPU)"]
        B1["physical_relations.py — dist(z;h), dist_to_redshift,<br/>hubble_function, comoving_volume_element, redshifted_mass"]
        B2["LISA_configuration.py — LisaTdiConfiguration:<br/>power_spectral_density, _confusion_noise, S_OMS, S_TM"]
        B3["parameter_estimation/parameter_estimation.py —<br/>ParameterEstimation: waveform, Fisher, SNR, CRB"]
        B4["datamodels/parameter_space.py — ParameterSpace<br/>(14 EMRI params); datamodels/detection.py — Detection"]
        B5["emri_rate.py — R_eff_per_mbh, R_EMRI,<br/>mbh_mass_function (Babak et al. 2017)"]
    end

    subgraph CATA["Galaxy catalogue"]
        C1["galaxy_catalogue/handler.py — GalaxyCatalogueHandler<br/>(GLADE+ BallTree, host search, rate-weighted draw)"]
        C2["galaxy_catalogue/pixel_completeness.py —<br/>PixelCompleteness (per-HEALPix m_th completeness)"]
        C3["galaxy_catalogue/parser.py / glade_completeness.py —<br/>raw GLADE+ parsing + completeness models"]
        C4["dark_siren_injection.py — draw_mixture_hosts,<br/>compute_global_catalog_fraction (in/out-of-catalog)"]
    end

    subgraph INFER["Bayesian inference (🟦 CPU)"]
        D1["bayesian_inference/bayesian_statistics.py —<br/>BayesianStatistics, p_D, p_Di, single_host_likelihood,<br/>D(h) + β + global-catalog precomputes"]
        D2["bayesian_inference/simulation_detection_probability.py —<br/>SimulationDetectionProbability (KDE P_det grid)"]
        D3["bayesian_inference/posterior_combination.py —<br/>combine_posteriors, load_posterior_jsons"]
        D4["bayesian_inference/bayesian_inference.py —<br/>Pipeline A dev cross-check (NOT used by --evaluate)"]
        D5["cosmological_model.py — Model1CrossCheck<br/>(emcee event sampling), LamCDM/DarkEnergy scenarios"]
    end

    subgraph OUT["Output"]
        E1["plotting/ — figure factories (bayesian_plots,<br/>evaluation_plots, model_plots, catalog_plots, ...)"]
        E2["memory_management.py — GPU memory; decorators.py — timers"]
    end

    A1 --> B3 & C1 & D1 & D5
    A3 --> B1 & B2 & B3
    B3 --> B2 & B4 & B1
    D5 --> B5 & C1
    C4 --> C1 & C2 & B5
    D1 --> C1 & C2 & D2 & B1
    D1 --> D3
    D2 -.reads.-> A1
    A1 --> E1

    classDef gpu fill:#bbf7d0,stroke:#15803d,color:#000;
    classDef cpu fill:#bfdbfe,stroke:#1d4ed8,color:#000;
    classDef shared fill:#f1f5f9,stroke:#64748b,color:#000;
    class B2,B3 gpu;
    class D1,D2,D3,D4 cpu;
    class A1,A2,A3,B1,B4,B5,C1,C2,C3,C4,D5,E1,E2 shared;
```

### Ownership table

| File | Owns | Pipeline |
|------|------|----------|
| `main.py` | CLI dispatch + both pipeline drivers (`data_simulation`, `injection_campaign`, `evaluate`, `generate_figures`) | both |
| `arguments.py` | `Arguments` CLI parsing (`--use_gpu`, `--num_workers`, `--seed`, `--simulation_index`, `--evaluate`, `--combine`) | both |
| `constants.py` | Physical/cosmological constants & config (`H=0.73`, `SNR_THRESHOLD=20`, `OMEGA_M=0.25`, frequency limits) | both |
| `cosmological_model.py` | `Model1CrossCheck` (emcee EMRI event sampler), `LamCDMScenario`, `DarkEnergyScenario` | 1 (sampling) |
| `emri_rate.py` | Per-MBH EMRI rate `R_eff_per_mbh`, volumetric `R_EMRI`, mass function | 1 + 2 (host weights) |
| `galaxy_catalogue/handler.py` | `GalaxyCatalogueHandler` — GLADE+ load, equatorial→ecliptic rotation, BallTree host search, rate-weighted host draw | 1 + 2 |
| `galaxy_catalogue/pixel_completeness.py` | `PixelCompleteness` — per-HEALPix-pixel magnitude-threshold completeness `f_k`, `f_bar` | 1 + 2 |
| `dark_siren_injection.py` | `draw_mixture_hosts`, `compute_global_catalog_fraction` — in-catalog vs dark host split | 1 |
| `datamodels/parameter_space.py` | `ParameterSpace` — the 14 EMRI parameters, randomisation, host→param mapping | 1 |
| `datamodels/detection.py` | `Detection` — one CRB row wrapped with sky-covariance accessors | 2 |
| `parameter_estimation/parameter_estimation.py` | `ParameterEstimation` — `few` waveform, Fisher matrix (5-pt stencil), `scalar_product_of_functions`, SNR, CRB, CSV write | 1 (GPU) |
| `LISA_configuration.py` | `LisaTdiConfiguration` — TDI PSD (`power_spectral_density`, OMS/test-mass/confusion noise) | 1 (GPU) |
| `physical_relations.py` | `dist(z;h)`, `dist_to_redshift`, `hubble_function`, `comoving_volume_element`, `redshifted_mass` | both |
| `bayesian_inference/bayesian_statistics.py` | `BayesianStatistics` (Pipeline B / production): `p_D`, `p_Di`, `single_host_likelihood`, D(h) + β + global-catalog precomputes | 2 (CPU) |
| `bayesian_inference/simulation_detection_probability.py` | `SimulationDetectionProbability` — KDE-based P_det grid + `RegularGridInterpolator` look-ups | 2 |
| `bayesian_inference/posterior_combination.py` | `combine_posteriors`, `load_posterior_jsons` — combine per-event JSONs into the H₀ posterior | 2 |
| `bayesian_inference/bayesian_inference.py` | Pipeline A dev cross-check (`BayesianInference`); **not** used by `--evaluate` | dev only |
| `memory_management.py` / `decorators.py` | GPU memory pool management; timing decorators | 1 |
| `plotting/` | All figures: factory functions (`data → (fig, ax)`) in topic modules; `_style.py` sets Agg + thesis mplstyle | output |

> Note: `CLAUDE.md` refers to a `bayesian_inference/detection_probability.py`; the live
> detection-probability class is `SimulationDetectionProbability` in
> `bayesian_inference/simulation_detection_probability.py`.

---

## The hand-off in one line

```
Pipeline 1 (GPU): events → hosts → waveform → Fisher → ⟶ cramer_rao_bounds*.csv
                                                          │
Pipeline 2 (CPU):  ⟶ read CSV → P_det + D(h) + per-host likelihood → ⟶ posteriors/*.json → combine → H₀
```
