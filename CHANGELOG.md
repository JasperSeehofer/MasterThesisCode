# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Changed
- **Inference is now deterministic (G4):** the with-BH-mass MC denominator draws from a
  per-host stream derived from `(base_seed, detection_index, host_z, host_M)`;
  `--seed` reaches the inference layer via `evaluate(..., base_seed=...)` (default 0).
  Previously unseeded ~1% MC noise made 2D-channel posteriors non-reproducible run-to-run.
- **Fisher condition-number gate (G10):** `kappa > FISHER_CONDITION_NUMBER_MAX = 1e14`
  now skips the event (was log-only); singular matrices are caught by the same gate.
- **Completion-term solid-angle Jacobian (G2a):** `B_num` sky marginal is
  `(sin θ_det/4π)·N(d_L_frac; 1, Σ[2,2])`; `volume_deconv` `Z_g`/`D_g` window clamped to z ≥ 0.
- **[PHYSICS] BREAKING — missing `dt²` DFT normalization restored in the LISA inner product**
  (`parameter_estimation.scalar_product_of_functions`). The raw `rfft` output was integrated
  against the physical PSD without the `h̃(f) = dt·X` correspondence, making every SNR exactly
  `dt=10`× too small and every Cramér–Rao σ 10× too large (the "SNR ≥ 20" catalogues were
  physical-SNR ≥ 200 populations confined to z ≤ 0.11, vs the Babak et al. 2017 M1 horizon
  z ≈ 1.5–3.8). Verified five ways (analytic monochromatic, FFT-free Parseval, broadband
  independent FFT, lisatools' `dt*rfft` convention, astrophysical horizon):
  `docs/derivations/G8_dt2_inner_product_derivation.md`. **All pre-fix SNR/CRB data remains
  RETIRED; the Phase-2 campaign runs with physical SNR semantics** (deeper population, more
  events; PRE_SCREEN_SNR_FACTOR and timeout budgets to be re-checked at the new scale).
- **[PHYSICS] BREAKING — library default `normalization_mode` flipped `'global'` → `'volume_deconv'`**
  (`BayesianStatistics.evaluate()`), aligning the library with the CLI default and the
  P–P-calibrated estimator (Gray et al. 2020 arXiv:1908.06050 Eqs. A.9/A.10 + volume-consistent
  host-z prior; verification report §7). Requesting `'global'` explicitly now emits a
  `UserWarning`: it is mis-calibrated for photometric-redshift catalogues (~0% coverage, posterior
  rails to the grid edge) and remains available only to reproduce the railed baseline.
  **Migration:** code calling `evaluate()` without `normalization_mode` now gets calibrated
  `volume_deconv` posteriors instead of railed `global` ones; pass `normalization_mode="global"`
  to reproduce pre-fix results.
- **[PHYSICS]** de-rail the in-catalogue H₀ likelihood normalization (commission
  `.planning/INDEPENDENT-VERIFICATION-REPORT-20260701.md` §7). `bayesian_statistics.py`: new
  `normalization_mode ∈ {global, local_ratio, volume_deconv}` on `evaluate()` /
  `single_host_likelihood`. The pre-fix `global` partition-norm single ratio
  `L_cat=(Σ_local w_g N_g)/(Σ_global w_g D_g)` pins the photo-z H₀ posterior to a grid edge on real
  seed600 data (MAP 0.86 pre-4π, 0.60 after the 1/(4π) completion fix `cb16142`). `local_ratio`
  reverts to the Gray A.9/A.10 local self-normalized ratio-of-sums (de-rails to a peaked 0.73);
  `volume_deconv` additionally deconvolves the host-z prior through the comoving-volume element
  `dV_c/(1+z)` (per-galaxy renormalized), consistent with `D(h)`. A from-scratch P–P/coverage test
  (`results/commission_20260701/scratch/d2/`) shows the bare-Gaussian host-z numerator is
  mis-calibrated (≈0% coverage, σ_z² Eddington-in-z bias) and the volume-weighted numerator is
  calibrated; the real-data de-rail matrix (`redteam/derail_matrix_results.json`, 0.86→0.60→0.73)
  and the +0.010 MAP shift agree. **Production CLI default → `volume_deconv`** (`--normalization_mode`);
  the library `BayesianStatistics.evaluate` default initially stayed `global` (superseded by the
  entry above: library default now also `volume_deconv`). Gray et al. (2020) arXiv:1908.06050
  Eqs. A.9 / A.10 / 33.
- **[PHYSICS]** sky-aware selection function (closes the p_sample≠p_comp sky/selection
  paper-blocker; audit `.planning/PSAMPLE-PCOMP-AUDIT-20260701.md` R1). The generator draws +
  SNR-selects an anisotropic real sky through the sky-dependent LISA response, but the inference
  selection evaluated an **isotropic** `p_det` (φ=θ=0). Now: `simulation_detection_probability.py`
  builds an **ecliptic-latitude-band** detection-horizon survival `p_det(d_L|β)` re-binned from the
  **existing** injections (no new campaign; LISA's annual orbit ⇒ azimuthal symmetry `R=R(β)`,
  Cutler 1998 arXiv:gr-qc/9703068); `bayesian_statistics.py` `D(h)`, `β_Ḡ`, and the global catalog
  denominator become per-pixel/per-band sums `(1/N_pix)Σ_k(…)p_det(Ω_k)` (Gray 2023 arXiv:2308.02281
  Eq. 2.3; GMV 2022 arXiv:2111.04629 Eq. 5; MFG 2019 arXiv:1809.02063 Eq. 6), with each catalog
  galaxy evaluated at its real ecliptic latitude via the **same flat per-band** survival (one shared
  `p_det(Ω)` object across all integrals — guardrail). The with-BH-mass 4D branch stays isotropic +
  flagged (statistics-starved). Isotropic limit recovers the old code bit-for-bit (regression T1=0.0);
  partition `D=β_G+β_Ḡ` (T2); anisotropic closure witnesses + removes the sky bias (T6, 75×). New
  `test_sky_selection.py` (T1–T8). **Measured H₀ impact ≲1%, sign-indeterminate** (GLADE ZoA is
  Galactic-plane-aligned ≈60° to the ecliptic ⇒ `Cov[p_det,p_sky]≈0`); reported as a bounded
  systematic, a formal-correctness / self-consistency closure. Derivation:
  `.planning/derivation-sky-selection/PHYSICS-CHANGE-PROTOCOL.md`.
- **[PHYSICS]** `galaxy_catalogue/handler.py`: corrected the host stellar-mass → BH-mass
  **error budget**. The relation is identified+cited as **Reines & Volonteri (2015)**
  (arXiv:1508.06274, Eq. 5; broad-line AGN M_BH–M_*,total; α=7.45, β=1.05) — *not* McConnell
  & Ma 2013. Two fixes to `BH_mass_error`: (1) **added the relation intrinsic scatter**
  ε₀ = 0.24 dex (`sigma_int`), previously omitted — it is the *dominant* term, so host-mass
  errors were ~3× too tight at the pivot (fractional CV 0.18 → 0.59), making the with-BH-mass
  (2-D) inference channel over-confident; (2) **fixed an operator-precedence bug** in the
  stellar-mass-error term (`beta / stellar_mass / 10` → `beta / stellar_mass`; d ln M_BH/d M_*
  = β/M_*, the 1e11 pivot is constant), which understated that term 100× in variance. Also
  corrected the (currently unused) inverse relation's M_BH-error term (β → 1/β) and added its
  scatter. Regression test `test_mass_relation.py`. This quantifies the σ_z/σ_M forecast: the
  realistic σ_M floor (≈60–200%) ≫ the ~1–2% the 2-D channel needs → no H₀ rescue. Follow-up:
  a log-normal host-mass model (the linear-Gaussian leaks ~5% to M<0 at this scatter).
  Ref: Reines & Volonteri (2015) arXiv:1508.06274 §4.1; Greene+2020 arXiv:1911.09678.
- **[PHYSICS]** `bayesian_inference/simulation_detection_probability.py`: replaced
  the local-linear / Nadaraya-Watson kernel-regression `p_det` estimator with the
  **detection-horizon survival function** `p_det(d_L) = P(d_hor ≥ d_L)`, with the
  per-injection horizon `d_hor = SNR · d_L / SNR_threshold`. Detection is a
  deterministic optimal-SNR threshold and SNR ∝ 1/d_L, so `p_det` is *exactly* the
  survival function of the (h-invariant) horizon distribution — monotone, `p(0)=1`
  and `p(d_L > max d_hor)=0` by construction, bandwidth-free in d_L, built once
  (no per-h regrid). The 2D channel conditions on observer-frame `M_z` with a
  Gaussian kernel in `log10 M_z` only. Fixes the kernel's far-tail overshoot
  (e.g. 1D `p_det(0.6 Gpc)` 0.0315 → 0.00055 where the truth → 0), which had
  inflated/steepened the selection denominator `D(h)` and biased the 1D MAP up by
  ~+0.02. Verified via the production `precompute_completion_denominator`: D(h)
  decline 0.73→0.76 = −0.87% (was −3.9% under local-linear; matches the survival
  prediction). Public API and the 6 `bayesian_statistics` call sites unchanged.
  Ref: Finn & Chernoff (1993) arXiv:gr-qc/9301003; Finn (1996) arXiv:gr-qc/9601048
  (`p_det = P(Θ > Θ_thr)`); Mandel–Farr–Gair (2019) arXiv:1809.02063;
  Gray et al. (2020) arXiv:1908.06050.
- **[PHYSICS]** `bayesian_inference/bayesian_statistics.py`: aligned the in-catalogue
  likelihood `L_cat` with Gray et al. (2020) **Eq. (A.9/A.10)** (verified against the
  paper appendix directly). (1) Removed the spurious `p_det` from the catalog-term
  numerators in **both** channels — `p_det = p(D_GW|…)` now appears only in the
  denominator `D_g`; an extra numerator `p_det` is the Mandel–Farr–Gair (2019)
  "most common mistake". (2) Changed the `L_cat` aggregation from the mean of per-galaxy
  self-normalized ratios `(1/N)Σ(N_g/D_g)` to the ratio of sums `(ΣN_g)/(ΣD_g)` with a
  single shared selection denominator. This **reverses the Phase-38 STAT-01 choice**,
  which was a misreading of Gray (its `test_l_cat_equivalence.py` had labeled the correct
  ratio-of-sums form "incorrect" — re-labeled). Empirically confirmed on seed400: halves
  the 1D MAP bias (0.750→0.740, +0.020→+0.010) and moves the 2D headline toward truth
  (0.7375→0.7350). Residual after fix (1D +0.010 / 2D +0.005) → completeness-weight
  `p(G|D,H0)` and/or single-seed scatter (multi-seed in flight). See
  `docs/H0_BIAS_RESOLUTION.md` §3.17. Ref: Gray et al. (2020) arXiv:1908.06050 Eq.
  A.9/A.10; Mandel, Farr & Gair (2019) arXiv:1809.02063.
- **[viz]** `plotting/bayesian_plots.py::plot_combined_posterior` (fig01)
  reworked to the "Observatory" grammar: the two mass-convention variants now
  share one blue separated by linestyle (solid Without M_z / dashed With M_z),
  with nested 50/68/95% HDI shading, a flat-prior overlay, Planck (pink) + SH0ES
  (cyan) reference bands as swatch labels, a km/s/Mpc secondary top axis, and a
  MAP title. `emri_thesis.mplstyle` default `image.cmap` viridis → cividis
  (Atlas: perceptually-uniform, CVD-safe).
- **[viz]** Full Observatory+Atlas restyle of the remaining figure suite (see
  `docs/VIZ_REDESIGN_PROPOSAL.md`): fig02 (per-event coloured by SNR via batlow,
  black combined headline), fig03 (fixed the clipped threshold annotation, grey
  hist + accent CDF), fig05 (sky → batlow), fig06 (offset notation off the ticks,
  truth crosshair), fig07 (analytic Fisher contours, KDE smoothing off), fig08
  (single CI-width-vs-N panel + Planck/SH0ES target bands + 1/√N guide, variants
  by linestyle), fig10 (PSD decomposition by linestyle), fig11 (fixed the d_L Gpc
  unit mislabel, direct-labelled h-curves), fig12 (locked-palette
  intrinsic/extrinsic groups + nested quantile markers), fig13 (single
  characteristic-strain figure with an example EMRI track), fig14 (mplot3d → 2D
  pairwise hexbin density). Retired the redundant fig15 campaign dashboard; the
  `main.py` manifest was re-pointed to the restyled factories.
- **[viz]** Interactives (`plotting/interactive.py`) restyled to Observatory+Atlas
  (flat prior + nested HDI + a HOPs Play button on the combined posterior, batlow
  sky map, Planck/SH0ES target-width bands on convergence, palette alignment) plus a
  NEW `interactive_h0_tension_explorer` (NF-8) — combined posterior vs Planck/SH0ES
  bands with an event-stacking slider. fig04/fig09 (detection yield / efficiency)
  unlocked: the generators now pool the injection CSVs
  (`<run>/simulations/injections`, z + SNR) for a real injected-vs-detected
  selection function (504k injected, SNR ≥ 20 detected) instead of gating to None.

### Fixed
- `__main__.py`: force a clean process exit (`logging.shutdown()` + flush +
  `os._exit(0)`) at the `python -m master_thesis_code` entrypoint. The
  `--generate_figures` command enables matplotlib LaTeX rendering
  (`text.usetex=True`) — the only command that does — whose `latex`/`dvipng`
  helper subprocesses left the interpreter blocked during teardown on the
  cluster: the combine job wrote all 15 figures in ~2 min, then idled ~43 min
  until SLURM killed it at walltime (`TIMEOUT`, e.g. job 5148384), wasting the
  node and poisoning the job exit state. Scoped to the CLI entrypoint so
  library/test callers of `main.main()` are unaffected (`os._exit` would
  otherwise terminate a hosting pytest process).

### Added
- **`master_thesis_code.validation` subpackage (G4b):** the 2026-07-01 commission's
  independent P–P/coverage calibration harness (investigator d2) promoted from
  `results/commission_20260701/scratch/d2/` into first-class, tested package code.
  `pp_coverage.py` runs a pure numpy/scipy synthetic-universe coverage test
  (flat-ΛCDM tables, Malmquist selection, single-host dark-siren H₀ estimator with
  switchable host-z kernel: 'bare' Gaussian vs calibrated 'volume' dV_c/dz/(1+z));
  intended for per-seed coverage runs during the Phase-2 campaign.
- New `plotting/validation_plots.py`: `plot_h0_forest` (NF-1, the H₀-in-context
  forest plot vs Planck 2018 / SH0ES / GWTC-3 dark sirens, now shipped as fig15)
  and `plot_pp_coverage` (NF-2, a P–P / coverage factory ready for an
  injection-recovery campaign).
- `cmcrameri` dependency — Crameri scientific colormaps (batlow / vik / romaO)
  for the Atlas field and validation figures.
- Visualization redesign — Observatory + Atlas foundation (see
  `docs/VIZ_REDESIGN_PROPOSAL.md`: 4 named design directions, a per-figure
  current→proposed table, 8 new-figure ideas). New `_colors.py` v2 tokens:
  `METHOD` (method→color map), `VARIANT_STYLE` (one hue + linestyle per
  variant), `PLANCK_BAND`/`SHOES_BAND`/`PRIOR`, and `SEQUENTIAL_CMAP`/
  `DIVERGING_CMAP`/`CYCLIC_CMAP` (optional `cmcrameri` batlow/vik/romaO with a
  cividis/RdBu/twilight fallback).
- Results gallery on the documentation site / GitHub Pages
  (`docs/source/results_gallery.rst`, linked from the `index.rst` toctree): a
  grouped, captioned gallery of the 15 production figures (fig01–fig15) from the
  `run_20260620_seed500_phase50` campaign (1385 detections, 83-point H₀ grid;
  combined MAP h=0.737 1D / 0.732 2D). Figures are committed as web PNGs under
  `docs/source/figures/` (with a `.gitignore` negation for the global `*.png`
  rule).
- Phase 48 production-sweep verdict
  (`scripts/bias_investigation/outputs/phase46_merged/h3_production_sweep_verdict.json`):
  1D MAP=0.7324 (σ_boot=0.0021, bias=+0.0024, z=+1.16σ) and 2D
  MAP=0.7322 (σ_boot=0.0022, bias=+0.0022, z=+0.97σ) on phase46-merged
  1473 events at h=0.73 across the 63-point non-uniform grid. PASS on
  all five Phase 48 stopping criteria except (3) Δh-sensitivity ≤0.001
  (observed 0.00188 2D / 0.00273 1D, comparable to σ_boot); R1's 21-pt
  parabolic refine was Δh-resolution-limited (Δh=0.005 sub-grid recovers
  R1's MAP ≈0.7308; full Δh=0.001 dense core resolves MAP to ~0.7322).
  Info monotonicity preserved (|2D bias| ≤ |1D bias|). Two-submission
  recovery (jobs `4271862` → TIMEOUT 41/63; `4344777` → COMPLETED 22/63
  remaining) documented in `.planning/phase-48-job-tracking.json`.
- Phase 48 production fine-grid h-sweep infrastructure for h=0.73:
  `cluster/evaluate_production_h0p73_dense.sbatch` runs a 63-point
  non-uniform grid (Δh=0.001 dense core across [0.710, 0.750] = 41
  points; Δh=0.010 wings across [0.600, 0.700] and [0.760, 0.860] = 11
  points each) on cpu_il, --array=0-6 stride-sliced, ~25 min wall.
  `scripts/bias_investigation/test_28_production_finegrid_analyze.py`
  reuses test_24's `load_per_h_likelihoods` + `parabolic_refine` and
  adds a Δh-sensitivity diagnostic that re-computes MAP under sub-grids
  {full 63-pt, dense-core-only, Δh=0.005, Δh=0.010} to confirm Δh is
  not the resolution-limiting factor for MAP/σ_boot. Δh_core ≈ σ_boot/4
  (post-H3-fix σ_boot ≈ 0.0037 2D / 0.0047 1D); 4-5 grid points within
  ±σ_boot of MAP keeps parabolic refine well-conditioned.
- Phase 47 H3 fix pre-implementation diagnostic
  `scripts/bias_investigation/test_27_m_coordinate_mismatch.py`: builds two
  `SimulationDetectionProbability` instances differing only in the M-axis
  coordinate convention (current source-frame `M` vs proposed observer-frame
  `M_z`), queries both per detection at sample integration z values, and
  reports per-event Δp_det distributions and a heuristic predicted MAP shift.
  On phase46-merged 1473 events at h=0.73 the proposed grid's M-axis extends
  1.49× the current max (matching the expected M_z/M_source ratio at typical
  z); 23% of events show |Δp_det| > 0.05 with mean Δp_det = -0.031
  (proposed retrieves the correct M_z bin instead of the heavier-source bin
  the current source-frame grid was hitting).
- Phase 35: coordinate-bug baseline audit (`scripts/audit_coordinate_bug.py` CLI +
  `.planning/audit_coordinate_bug.{md,json,png}` artifacts). Pre-fix baseline:
  42 CRB events — 0 in ±5° ecliptic-equator band (expected ~8.7% under isotropic
  prior). Phase 40 VERIFY-04 diffs the JSON sidecar post-fix.
- Phase 35: coordinate-frame test fixtures (`master_thesis_code_test/fixtures/coordinate.py`)
  with `equatorial_to_ecliptic_astropy`, `synthetic_catalog_builder`, `build_balltree` helpers.
- 2D-bias investigation diagnostic
  `scripts/bias_investigation/test_26_2d_pdet_edge_behavior.py`: classifies every
  (event × h_trial) cell on the phase46-merged CRB as in-grid or out-of-grid by
  direction relative to the 2D `p_det_with_bh_mass` grid; reports raw scipy
  extrapolation, production-clipped value, and principled-asymptote value
  side-by-side. Confirmed H2 mechanism: 6–12% of events at every truth fall in
  the d_L<dl_min direction, where the production code returns p_det ≈ 0 (mean)
  vs principled value 1.0 (saturated regime); 57 events cross the grid boundary
  at the h_trial=0.680→0.685 transition (Δh=0.005), causing per-event likelihood
  steps that drive spurious h-trial-dependence in the joint MAP.
- `.planning/2D-CHANNEL-AUDIT-20260505.md`: audit report tracking the 2D-bias
  investigation (Step 1a tier-3-fix sanity, Step 1b 2D p_det edge behaviour,
  Step 2 principled-extrapolation implementation, Step 2 follow-up 1D alignment).

### Changed
- `[PHYSICS]` Phase 49 F4 — Nadaraya-Watson kernel `p_det` estimator
  (`bayesian_inference/simulation_detection_probability.py:_build_grid_2d`,
  `_build_grid_1d`). Replaces the histogram form `p_det = N_det/N_total`
  with the smooth kernel-weighted estimator
  `p̂(d_L_q, M_q, h) = Σ_k K_k · 1[SNR_k(h)≥thr] / Σ_k K_k` evaluated at
  fixed grid centers. Kernels are Gaussian on `d_L` (linear, Gpc) and
  `log10(M_z)` (log-scale, dex); bandwidths from Scott's rule
  (`σ = bandwidth_scale · N^(-1/6) · std`) on the injection sample;
  truncated at 3σ in `d_L` via sorted `np.searchsorted` for O(N_inj)
  cost (~0.75s per grid build on 105 500 injections, 60×40 grid). New
  constructor parameter `bandwidth_scale: float = 1.0` (validated > 0)
  for tuning. Closes both residual spike mechanisms identified by
  `test_29`: A (injection d_L motion across fixed bin edges, 96% of
  pre-F4 Σ(Δp)²) and B (SNR-threshold integer crossings, 3.6%).
  **Estimator-level smoothness:** Σ(Δp_det)² over 48 queries × 30 Δh=0.0005
  steps dropped from 1.5434 (post-F1) to 0.0016 (post-F4) = 987×
  reduction; worst single-step |Δp| dropped from ≈0.05 to 0.0103.
  Public API unchanged (kernel selected automatically; `quality_flags`
  keys preserved with continuous-mass semantics under the new form).
  Verification: `scripts/bias_investigation/test_30_f4_estimator_smoothness.py`,
  output at
  `scripts/bias_investigation/outputs/phase46_merged/test_30_f4_smoothness.json`.
  Cluster validation of production posteriors pending. Plan and physics
  presentation at `.planning/PHASE-49-F4-PLAN.md`. Refs: Nadaraya (1964);
  Watson (1964); Scott (1992) Ch. 6; Farr (2019) arXiv:1904.10879 Sec III;
  Mandel-Farr-Gair (2019) arXiv:1809.02063 Eq. 18.
- Phase 49 F1 cluster validation (job `4662333`) — **PARTIAL fix**, not
  paper-grade. The h-stable bin-edge fix (commit `87ea7a8`) removed
  one of two coherent-noise mechanisms (the rising flank of the
  combined H₀ posterior is now mostly smooth and monotonic), but a
  single sharp discontinuity remains at h=0.738→0.739 (1D drops 16×,
  2D drops 32× across one Δh=0.001 bin) and MAP shifted +0.0056
  further from truth than pre-F1 (1D MAP 0.7324→0.7378). Suspected
  residual mechanism: SNR-threshold integer crossings of individual
  injections (bin-edge-independent). The Farr 2019 fixed-injection +
  analytic-reweighting form (gwcosmo / ICAROGW production pipelines)
  would eliminate both mechanisms; queued as F4 in
  `.planning/debug/posterior-noisy-peak.md`. Pre-F1 posteriors
  archived on cluster (`archive/production_h0.73_20260512_175829/`);
  post-F1 verdict at
  `scripts/bias_investigation/outputs/phase46_merged/F1_post_fix_verdict_PARTIAL.json`.
  **No figure refresh from these posteriors — not paper-ready.**
- Phase 48 figure refresh with production-sweep posteriors (1473 events,
  63-point non-uniform h-grid) + phase46-merged 1549-event CRB:
  `paper/figures/{h0_posterior_comparison,h0_posterior_kde,m_z_improvement,
  posterior_convergence,single_event_likelihoods,snr_distribution}.pdf`,
  thesis archive `results/figures_seed200/fig01–fig15.pdf`, GitHub Pages
  interactives `interactive/{combined_posterior,fisher_ellipses,h0_convergence,
  m_z_improvement,sky_map}.html` and updated `interactive/index.html`
  footer (data lineage: production-sweep 63-pt h-grid, MAP=0.732 at
  z≈+1σ for both channels). Paper PDF (`paper/main.pdf`, untracked) rebuilt.
- `[PHYSICS]` Phase 47 H3 fix — numerator p_det query
  observation→hypothesis + 2D grid M-axis observer-frame convention.
  Two coupled changes that together fix the residual 2D-channel structural
  bias remaining after the principled-bridge fix (commit `2b33cad`).
  (i) `bayesian_inference/simulation_detection_probability.py:_get_or_build_grid`:
  the 2D grid M-axis is now observer-frame `M_z = M_source · (1+z_inj)`
  (multiplied at grid construction time, one-line change). The grid axis
  thus matches the production query coordinate
  (numerator: `host_M·(1+z)`; denominator: `M·(1+z)`) instead of binning
  in source-frame `M_source` while queries pass `M_z` (a coordinate
  mismatch that put queries ~50% higher than the bin labels at typical
  z≈0.5).
  (ii) `bayesian_inference/bayesian_statistics.py:1304-1306` numerator
  integrand: changed the `p_det` query from `np.full_like(z, _det_M)`
  (the *observation* — detection's ML observer-frame mass, constant
  across the integration over candidate redshift z) to
  `host_M · (1.0 + z)` (the *hypothesis* — the host candidate's
  observer-frame mass at integration z). This matches the rest of the
  integrand's hypothesis convention (cf. `mu_gal_frac` line) and the
  denominator's `M·(1+z)` query. Phase 14's "approximation, not a bug"
  justification at L1298–1300 was valid only while σ_boot was wide
  enough (>~0.005) to mask the residual; under post-bridge σ_boot=0.0039
  on phase46-merged 1473 events it drove a +3.6σ structural residual.
  Removed the "known approximation" comment blocks at
  `bayesian_statistics.py:1298–1303` and `:1357–1359`, and the
  "this is a known approximation" note in
  `simulation_detection_probability.py:619–623`. **All
  `posteriors_with_bh_mass/` produced before this commit are stale.**
  References: Mandel, Farr & Gair (2019), arXiv:1809.02063 §2 (selection
  function evaluated at hypothesis); Maggiore (2008) Vol 1 §4.1.4
  (`M_z = M_source · (1+z)`); Babak et al. (2017), arXiv:1703.09722 §III.
  Property tests added (`TestPDetGridMassCoordinateFrame`):
  M-axis is observer-frame M_z; query at M_z lands in the expected
  built bin. Pre-implementation diagnostic and post-fix narrative in
  `docs/H0_BIAS_RESOLUTION.md` §3.15.
- `[PHYSICS]` `bayesian_inference/simulation_detection_probability.py`
  (`detection_probability_with_bh_mass_interpolated`, 2D channel): replaced raw
  scipy linear extrapolation + [0,1] clip with a principled
  monotonic-asymptotic scheme. Saturating face (d_L<dl_min): linear bridge
  from (dl_min, p_edge) to (0, 1) — C0 continuous at dl_min, reaches the
  asymptote p_det=1 at the unique natural physical scale d_L=0. Suppressing
  faces (d_L>dl_max, M_z>M_max, M_z<M_min): slope-matched linear extrapolation
  from the boundary, clamped to [0, p_edge] (Option A directional clamp).
  Corner cells: min of the two face extrapolations. Removes the
  ~h_trial-driven discontinuity that drove spurious h-dependence as hosts
  crossed the moving 2D grid boundary; replaces the 6–12% out-of-grid clipped
  ≈0 values with the principled saturated ≈1 in the d_L→0 regime. Property
  tests added (`TestDetectionProbabilityWithBHMassPrincipledExtrapolation`):
  in-grid contract, C0 continuity, Option A floor, suppressing-face decay,
  corner min-rule, vectorisation. Mechanism documented in
  `.planning/2D-CHANNEL-AUDIT-20260505.md` Step 1b. **All
  `posteriors_with_bh_mass/` produced before this commit are stale** for any
  conclusion that depends on absolute p_det values near the d_L grid edges.
- `[PHYSICS]` `bayesian_inference/simulation_detection_probability.py`
  (`detection_probability_without_bh_mass_interpolated_zero_fill`, 1D channel,
  alignment with 2D): replaced the Phase 45 Plan 45-02/04 anchor scheme
  (Wilson 95% LB at d_L=0 = 0.7931 + intermediate empirical anchor at
  d_L=0.05 = 1.0, deliberately fitted to "not overshoot truth on production
  posteriors") with the same principled bridge + slope-matched scheme as the
  2D channel. The Wilson 95% LB anchor was actively suppressing the
  empirical p̂(c_0)=1.0 produced by the augmented Phase 46 injection campaign
  (the opposite of its original lift purpose). Removed module constants
  `_P_MAX_EMPIRICAL_ANCHOR`, `_D_INTERMEDIATE_ANCHOR_GPC`,
  `_P_INTERMEDIATE_EMPIRICAL`. Removed anchor-prepending in `_build_grid_1d`
  (grid is now the raw histogram with bin centres in d_L). Function name
  retains the legacy `_zero_fill` suffix for backward compatibility with ~15
  call sites; new behaviour at d_L=0 is **1.0** (asymptote) instead of the
  old **0.7931**. **All `posteriors/` produced before this commit are stale.**
- `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py`:
  removed the entire `TestPhase45EmpiricalAnchor` class (13 anchor-specific
  tests; the scheme they tested no longer exists). Renamed and rewrote two
  `TestZeroFillBoundaryConvention` tests to test the bridge formula and the
  slope-matched-toward-0 above-dl_max behaviour. Added 8+7=15 new
  property-based tests covering the principled extrapolation in both
  channels.
- `scripts/bias_investigation/test_22_dh_double_count.py:254`: rename
  `D_term_per_h` → `D_term_per_h_legacy` to match the JSON schema produced
  by the post-Tier-3 audit script (one-line key fix; not a physics change).
- `scripts/bias_investigation/test_14_channel_audit.py`: deprecation header
  added — the script imports the now-deleted anchor constants and is
  superseded by the principled-extrapolation scheme.

### Fixed
- `[PHYSICS]` `bayesian_inference/simulation_detection_probability.py:708–713`
  (`detection_probability_without_bh_mass_interpolated_zero_fill`): removed
  spurious left-side cutoff that zeroed `p_det` for `d_L < dl_centers[0]`.
  Because `dl_centers[0] = dl_max(h)/120` scales as `1/h`, the cutoff was a
  moving threshold that biased every close event with `d_L ≈ c_0` toward
  `h_max`. Pre-fix at `d_L = 0.085 Gpc`: `p_det = 0` for `h ∈ [0.65, 0.83]`
  and `p_det = 0.59` at `h = 0.86`, driving a +145.7 log-unit shift across
  the 412-event production seed200 dataset and producing MAP = 0.860.
  Post-fix cluster re-eval (jobs 4160638/4160639) gives **MAP = 0.7650**
  on the same data, eliminating the +145.7 log-unit pathology and shifting
  the posterior 0.095 toward the true h=0.73. All 4 zero-handling
  strategies (naive/exclude/per-event-floor/physics-floor) now produce
  identical MAP, confirming no events are being suppressed by zero-handling
  logic. Below the first bin centre, the interpolator's existing
  `fill_value=None` (nearest-neighbour) returns the genuine first-bin
  injection statistic (≈ 0.55 at h = 0.73, n_total[0] = 312 injections in
  bin). Right-side cutoff above `dl_max` preserved. Function is shared
  across L_comp numerator, L_cat numerator/denominator, and D(h) denominator
  (STAT-03 invariant preserved). 4 regression tests added; full CPU suite
  557/557 pass. Residual +0.035 bias above truth is deferred to a follow-up
  phase (plan §8 fallback regime; hypothesis: p̂(c_0) underestimates true
  p_det → 1 at d_L → 0). Phase 44 — Eq. (A.19) in Gray et al. (2020),
  arXiv:1908.06050.
- `bayesian_inference/posterior_combination.py` (`combine_posteriors`, `combine_log_space`,
  `generate_comparison_table`): missing Gray et al. (2020) arXiv:1908.06050 Eq. A.19
  selection-function correction. The combine path was summing per-event log-likelihoods
  with no `−N·log D(h)` term, causing the joint posterior to grow monotonically with `h`
  and pin to the grid ceiling (MAP = 0.860). Fix subtracts `n_used · log D(h)` where
  `D(h)` is precomputed via `precompute_completion_denominator()`. Expected MAP ~0.730
  (matches `--evaluate` path, Phase 43). Paper-blocker resolved.
- `plotting/convergence_analysis.py`: missing per-event likelihood values (empty
  list in JSON, e.g. event 255 in `h_0_73.json`) now recorded as `nan` instead
  of `0.0`, and combined via `nansum` in log-space — eliminates the sharp dip to
  zero at h=0.73 in the representative combined posterior.
- `plotting/interactive.py` (`interactive_m_z_improvement`): layout crowding
  fixed — increased figure height (720→950 px), wider margins, repositioned
  dropdown above panel A (no longer overlaps panel B title), manual subplot
  titles placed via `add_annotation` before the frame loop (no auto-placement
  bleed into adjacent panels), increased vertical spacing (0.18→0.22).
- `docs/source/index.rst`: interactive figures link changed from `../interactive/`
  to `interactive/` — was routing to `/interactive/` (wrong domain root) instead
  of `/MasterThesisCode/interactive/`.

### Added
- `interactive/m_z_improvement.html`: committed pre-generated HTML for the M_z
  improvement explorer — now reachable from the GitHub Pages interactive index.
- `interactive/index.html`: added entry for the M_z improvement explorer.
  All five interactive figures are now linked from the index page.

### Fixed
- `.github/workflows/ci.yml`: Pages deploy step now copies all committed
  `interactive/*.html` files (not just `index.html`), so every interactive
  figure is live on GitHub Pages after each push to main.
- Refreshed `interactive/combined_posterior.html`, `sky_map.html`,
  `fisher_ellipses.html`, `h0_convergence.html` from latest `simulations/` data.

### Added
- `plotting/_metrics.py`: pure-function constraint-quality metrics for H₀ posteriors
  — HDI width, relative precision, KL info gain, Jensen–Shannon divergence (bits),
  MAP bias, and the paired-curve `effective_event_gain` interpolated in
  (log-width, log-N) space (exact for 1/√N power-law curves).
- `plotting/convergence_analysis.py`: `compute_m_z_improvement_bank()` paired-bootstrap
  aggregator (B=200, seed 20260410) that answers "does adding M_z tighten H₀, and
  by how much?" for varying numbers of detections. Emits
  `diagnostics/m_z_improvement_bank.json` (signature-validated cache) plus the
  static three-panel `plot_m_z_improvement_panels()` matplotlib figure.
- `plotting/interactive.py::interactive_m_z_improvement()`: Plotly three-panel
  explorer with a slider over N and a dropdown over metrics
  (HDI68 width / rel. precision / KL info gain / MAP bias), plus a live
  headline-numbers annotation. Wired into `generate_all_interactive()` so
  `--generate_interactive` emits `m_z_improvement.html`.
- `plotting/_helpers.py::compute_hdi_interval()`: minimal/highest-density credible
  interval (LIGO convention) — complements the existing equal-tailed helper.
- Paper figure `paper_m_z_improvement` registered in `main.py` figure manifest.
- 14 unit tests in `master_thesis_code_test/plotting/test_metrics.py` covering
  analytical limits of every new metric (Gaussian HDI = 2 × 0.99445σ,
  KL(uniform)=0, JSD self=0 and ≤1 bit, K(N)=1 for identical curves and =2 for
  1/√(2N) scaling, out-of-range → nan).
- `galaxy_catalogue/glade_completeness.py`: GLADE+ catalog completeness estimation
  $f(z, H_0)$ using galaxy counts in comoving volume shells (Phase 24).
- Completeness-corrected dark siren likelihood implementing Gray et al. (2020) Eq. 9,
  combining catalog and completion terms weighted by GLADE+ completeness (Phase 25).
- 23 tests for completeness estimation, 11 tests for completion term and combination
  formula (`test_glade_completeness.py`, `test_completion_term.py`).
- `scripts/bias_investigation/`: 7 diagnostic scripts and `FINDINGS.md` documenting
  root cause of H₀ posterior bias (GLADE catalog density gradient, not a formula bug).
- Docs badge in README linking to GitHub Pages.
- `CITATION.cff` for machine-readable citation metadata.
- GitHub issue templates (bug report, physics bug) and PR template.
- 14 new correctness tests for `BayesianInference` (TEST-3): likelihood peak location,
  detection probability monotonicity, selection effects, BH mass term, posterior
  positivity, and cross-H₀ consistency checks.
- 4 regression tests for truncnorm distribution correctness (STAT-3): PDF peak location,
  integration-to-one, correct `loc` for redshift and mass distributions.
- `bayesian_inference/simulation_detection_probability.py`: simulation-based P_det
  replacing KDE with injection-campaign histograms and importance-sampling weights
  (Phase 11.1).
- `bayesian_inference/posterior_combination.py`: log-space posterior combination with
  four strategies (log-sum, per-event floor, per-event nonzero-min, global floor)
  (Phase 21).
- `analysis/` module: post-hoc analysis scripts for grid quality, importance sampling,
  injection yield, sampling design, and validation (Phases 17–20).
- `interactive/` directory: five Plotly HTML figures — combined posterior, Fisher ellipses,
  H₀ convergence, sky map, index page (Phases 18–19).
- `derivations/dark_siren_likelihood.md`: first-principles derivation of the dark siren
  likelihood from Bayes' theorem (Phase 14).
- `paper/` directory: REVTeX4-2 PRD paper draft with 6 sections and 21 references
  (Phase 26).
- `docs/H0_BIAS_RESOLUTION.md`: dedicated chronological changelog of the H₀ posterior
  bias investigation and resolution (10 phases documented).
- `docs/source/limitations.rst`: known limitations, model assumptions, verified
  components, and bibliography moved from README to Sphinx docs.
- H₀ sweep capability: evaluation pipeline sweeps over H₀ grid points for posterior
  construction (Phase 13).
- `--injection_campaign` CLI flag for running SNR-only injection campaigns (Phase 11.1).
- `emri-merge-injections` entry point for merging injection CSV outputs (Phase 11.1).

### Fixed
- **[PHYSICS]** Comoving volume formula corrected to proper volume element $dV_c/dz$ with
  $1/E(z)$ factor; methods renamed `comoving_volume` → `comoving_volume_element`
  (PHYS-1, Issue #1).
- **[PHYSICS]** Galactic confusion noise integrated into LISA PSD via
  `LisaTdiConfiguration._confusion_noise()`, implementing Babak et al. (2023)
  Eq. (17) (PHYS-4, Issue #3).
- **[PHYSICS]** `GalaxyCatalog` truncnorm distributions (STAT-3): `truncnorm()` was created
  without `loc`/`scale` parameters in `setup_galaxy_mass_distribution`,
  `append_galaxy_to_galaxy_mass_distribution`, `setup_galaxy_distribution`, and
  `append_galaxy_to_galaxy_distribution`, defaulting to N(0,1) instead of the intended
  mass/redshift-space distributions. Also removed double normalization in
  `evaluate_galaxy_mass_distribution` — `truncnorm.pdf()` is already normalized.
- **[PHYSICS]** `single_host_likelihood` d_L fraction direction (STAT-4): the production
  likelihood in `bayesian_statistics.py` used `detection.d_L / d_L` (measured/model)
  instead of the correct `d_L / detection.d_L` (model/measured). The incorrect direction
  introduced a spurious `(d_L_measured/d_L_model)²` factor in the Gaussian exponent,
  biasing the H₀ posterior. Now consistent with `single_host_likelihood_integration_testing`.
- **[PHYSICS]** Fisher matrix now uses five-point stencil $O(\varepsilon^4)$ derivatives
  by default (`use_five_point_stencil=True`), replacing the $O(\varepsilon)$ forward
  difference (Phase 10, PHYS-3). Ref: Vallisneri (2008) arXiv:gr-qc/0703086.
- **[PHYSICS]** Spurious `/(1+z)` Jacobian removed from `single_host_likelihood`
  with-BH-mass numerator in `bayesian_statistics.py` (Phase 15).
- **[PHYSICS]** P_det grid extrapolation fix: `RegularGridInterpolator` `fill_value`
  changed from `0.0` to `None` (nearest-neighbor extrapolation), eliminating 702
  zero-likelihood completeness fallbacks (commit `44d5358`).
- Likelihood floor added to `single_host_likelihood` to prevent zero-product posterior
  collapse; per-event min-nonzero floor strategy (Phase 22).

### Changed
- Project framing updated from master thesis to paper publication stage.
- GitHub issues triaged: #1 and #3 closed as resolved, remaining issues labeled
  (`paper-blocker`, `design-choice`) and assigned to "Paper Submission" milestone.
- `.claude/skills/` directory with 6 custom skills for codified, repeatable workflows:
  - `physics-change`: enforces the 5-step Physics Change Protocol before any formula modification
  - `gpu-audit`: scans files for GPU/HPC compliance violations (guarded imports, xp pattern, vectorization)
  - `run-pipeline`: runs simulation/evaluation/SNR pipelines with correct flags and validates output
  - `check`: full quality gate (ruff lint + format + mypy + pytest) in one invocation
  - `known-bugs`: shows current status of all known physics/code bugs with priorities
  - `pre-commit-docs`: verifies CHANGELOG, TODO, CLAUDE.md, README consistency with staged changes
- Excalidraw MCP server configured (HTTP transport) for architecture diagram generation.
- `CLAUDE.md`: new "Skill-Driven Workflows" section with trigger rules table and
  physics-change trigger file list. Skills are mandatory workflow gates, not optional.
- `CLAUDE.md`: new "GitHub Integration" section — GSD/GPD workflows must keep GitHub
  issues, milestones, and labels in sync as work progresses.
- Detection probability estimation switched from KDE-based (`DetectionProbability` class)
  to simulation-based importance sampling estimator with injection campaign data
  (`SimulationDetectionProbability`, Phase 11.1).
- Posterior combination moved to log-space accumulation to prevent float64 underflow
  with 500+ events (Phase 21).
- `SimulationDetectionProbability` refactored to use SNR rescaling for h-dependent
  detection probability (commit `8161533`).
- README slimmed: known limitations, model assumptions, verification checklist, and
  bibliography moved to Sphinx docs (`docs/source/limitations.rst`).

---

## [2026-03-11] — plotting architecture refactor (Phases 1–4)

### Added
- `master_thesis_code/plotting/` subpackage: all visualization code now lives here.
  - `_style.py`: `apply_style()` sets Agg backend and loads the project style sheet.
  - `_helpers.py`: `get_figure()`, `save_figure()`, `make_colorbar()` utilities.
  - `emri_thesis.mplstyle`: single source of truth for all matplotlib rcParams
    (figure size, DPI, constrained layout, chunksize).
  - `simulation_plots.py`: factory functions for GPU usage, LISA PSD, noise components,
    and Cramér-Rao coverage plots. Also contains `PlottingCallback`.
  - `bayesian_plots.py`: factory functions for combined/event/subset posteriors,
    detection redshift distribution, and host galaxy count plots.
  - `evaluation_plots.py`: factory functions for Cramér-Rao heatmap, uncertainty violins,
    sky localization 3D scatter, detection contour, and generation time histogram.
  - `model_plots.py`: factory functions for EMRI distribution, rate, sampling, and
    detection probability grid plots.
  - `catalog_plots.py`: factory functions for BH mass distribution, redshift distribution,
    GLADE completeness, and comoving volume sampling plots.
  - `physical_relations_plots.py`: factory function for distance-redshift relation plot.
- `master_thesis_code/callbacks.py`: `SimulationCallback` Protocol class with five hooks
  (`on_simulation_start`, `on_snr_computed`, `on_detection`, `on_step_end`,
  `on_simulation_end`) and a `NullCallback` no-op implementation.
- `data_simulation()` now accepts an optional `callbacks: list[SimulationCallback]`
  parameter; hook calls inserted throughout the simulation loop.
- `--generate_figures <dir>` CLI argument in `arguments.py` (stub handler in `main.py`).
- `master_thesis_code_test/plotting/test_style.py`: 9 tests covering `apply_style`,
  `get_figure`, `save_figure`, and style sheet rcParams.

### Changed
- `main.py`: backend setup moved from inline `matplotlib.use("Agg")` to
  `from master_thesis_code.plotting import apply_style; apply_style()`.
- `memory_management.py`: removed `plot_GPU_usage()` method; added `time_series`,
  `memory_pool_gpu_usage`, `gpu_usage` properties for callback-based data access.
- `cosmological_model.py`: shrunk from ~3530 to ~1611 lines by extracting 7 plotting
  methods (~1900 lines): `plot_expected_detection_distribution`,
  `visualize_emri_distribution_sampling`, `visualize_emri_distribution`,
  `plot_detection_probability`, `plot_detection_fraction`, `visualize`,
  `visualize_galaxy_weights`.
- `parameter_estimation/evaluation.py`: removed `visualize()`,
  `visualize_detection_distribution()`, `evaluate_snr_analysis()` methods.
- `galaxy_catalogue/handler.py`: removed `visualize_galaxy_catalog()` method.
- `physical_relations.py`: removed `visualize()` function.
- Test coverage increased from 28.83% to 36.19%.

### Removed
- `master_thesis_code/bayesian_inference/scientific_plotter.py`: deleted entirely
  (dead `ScientificPlotter` wrapper class).
- `IS_PLOTTING_ACTIVATED` constant from `constants.py`.
- `if_plotting_activated` decorator from `decorators.py`.
- `__init__` plot side effects from `glade_completeness.py` (including module-level
  `asdf = GladeCatalogCompleteness()` instantiation), `detection_horizon.py`,
  `detection_distribution_simplified.py`, `emri_distribution.py`, `detection_fraction.py`.
- All `import matplotlib` statements from computation modules: `LISA_configuration.py`,
  `parameter_estimation.py`, `memory_management.py`, `galaxy.py`, `emri_detection.py`,
  `bayesian_inference.py`, `glade_completeness.py`, `handler.py`.
- Plotting methods removed from: `LISA_configuration.py` (`_visualize_lisa_configuration`),
  `parameter_estimation.py` (`_visualize_cramer_rao_bounds`),
  `galaxy.py` (`save_comoving_volume_sampling_plot`, `plot_comoving_volume`,
  `plot_galaxy_catalog`, `plot_galaxy_catalog_mass_distribution`),
  `emri_detection.py` (`plot_detection_distribution`, `plot_detection_sky_distribution`),
  `bayesian_inference.py` (`plot_gw_detection_probability`).

---

## [2026-03-11] — fix incomplete dist → luminosity_distance rename in scripts

### Fixed
- `scripts/prepare_detections.py`: column write `"dist"` → `"luminosity_distance"` so
  prepared CSVs produced by this script match the column name expected by the evaluation
  pipeline.
- `scripts/estimate_hubble_constant.py`: updated all column reads (`"dist"`,
  `"delta_dist_delta_dist"`) and dict keys (`"dist"`, `"dist_error"`) to use
  `"luminosity_distance"` / `"luminosity_distance_error"` /
  `"delta_luminosity_distance_delta_luminosity_distance"`.
- Patched existing simulation CSVs (`cramer_rao_bounds.csv`,
  `prepared_cramer_rao_bounds.csv`, `undetected_events.csv`) to rename the `dist` column
  to `luminosity_distance` so the evaluation pipeline can load them without error.

---

## [2026-03-11] — remove redundant binary data files from repo

### Changed
- `master_thesis_code/waveform_generator.py`: orbit file path changed from
  `"./lisa_files/esa-trailing-orbits.h5"` to bare `"esa-trailing-orbits.h5"`.
  `lisatools` resolves bare filenames against its own bundled `orbit_files/` directory,
  so the repo-local copy is no longer needed.
- `.gitignore`: added `few_data/` and `lisa_files/` to prevent accidental re-addition.

### Removed
- `few_data/` (~105 MB): 4 FEW waveform model binary files removed from git tracking.
  FEW auto-downloads its data to `~/.local/share/few/` on first use via its built-in
  `FileManager`; the repo-local copies were never registered as a search path.
- `lisa_files/` (~2.4 MB): 2 LISA orbit HDF5 files removed from git tracking.
  `lisatools` bundles all three orbit files inside the installed package; the
  relative-path workaround in `waveform_generator.py` is no longer needed.

---

## [2026-03-10] — physics & mathematics review (Phase 9)

### Added
- `README.md`: new top-level section "Scientific Background and Known Limitations" containing:
  - Two-paragraph project narrative (EMRIs as GW standard sirens, dark-siren H₀ method)
  - Key equations with references: Hubble function, luminosity distance, LISA inner product,
    Fisher matrix, SNR, and marginalised H₀ likelihood with selection-effects denominator
  - Model assumptions table (flat ΛCDM, Gaussian noise, SNR threshold, uniform H₀ prior,
    synthetic galaxy catalog, 5-year LISA mission)
  - Eight documented known limitations, each with file:line reference, impact description,
    and status tag (bug / design choice)
  - "What is mathematically correct" verification checklist for six core components
  - Bibliography with six key references (Hogg 1999, Babak 2023, Cutler & Flanagan 1994,
    Vallisneri 2008, Chen 2018, Planck 2018)
- `TODO.md`: physics fix items for all confirmed bugs (Issues 1–8), ordered by severity

### Changed
- `CLAUDE.md`: "Known Bugs to Be Aware Of" section updated with all eight confirmed issues
  from the physics review, with file:line references and fix descriptions

---

## [2026-03-10] — dev infrastructure & code health (Phase 8)

### Added
- `LICENSE`: MIT licence added so the project can legally be shared, forked, and cited.
- `CONTRIBUTING.md`: human-readable contribution guide covering env setup, branching,
  pre-commit, test commands, and the physics-change protocol.
- `.editorconfig`: enforces LF line endings, 4-space Python indent, UTF-8, and
  trailing-whitespace trimming across all editors.
- `.github/dependabot.yml`: weekly automated dependency-update PRs for both the `pip`
  ecosystem (uv lock file) and GitHub Actions.
- `pytest-cov` and `pytest-benchmark` added to `dev` extras in `pyproject.toml`.
- `[tool.coverage.run]` and `[tool.coverage.report]` sections in `pyproject.toml`:
  source is `master_thesis_code/`, test files omitted, gate at 25% (current: 36.19%).
- `addopts` in `[tool.pytest.ini_options]` now includes `--cov` and `--cov-report`
  so every `pytest` invocation reports coverage automatically.
- `pip-audit` added to `dev` extras; new CI step `pip-audit (security)` runs on every
  push to surface known CVEs in installed packages.
- CI step `Upload coverage report` uploads `coverage.xml` as a GitHub Actions artifact
  after the test run.
- `--seed` CLI argument in `arguments.py` (optional `int`; random value chosen and
  logged when omitted).
- `_write_run_metadata()` in `main.py`: writes `run_metadata.json` into the working
  directory at startup, recording `git_commit`, `timestamp`, `random_seed`, and all
  CLI arguments for simulation reproducibility.
- `master_thesis_code_test/test_benchmarks.py`: two `@pytest.mark.slow` benchmark
  tests — `BayesianInference.likelihood` for N=50 detections and
  `GalaxyCatalog.evaluate_galaxy_distribution` for a 500-galaxy catalog.

### Changed
- `main.py`: `main()` now seeds `numpy.random` from `arguments.seed` before any
  sampling begins, and calls `_write_run_metadata()`.
- `master_thesis_code/datamodels/galaxy.py`:
  `GalaxyCatalog.get_samples_from_comoving_volume` gains `save_plot: bool = False`
  parameter; the PNG side-effect is suppressed by default and only fires when the
  caller explicitly passes `save_plot=True`.
- `master_thesis_code/datamodels/parameter_space.py`:
  `ParameterSpace.dist` field and its `Parameter.symbol` both renamed to
  `luminosity_distance`. `_parameters_to_dict` key updated accordingly.
  This removes the Python name-shadowing of the imported `dist()` function.
- All CSV column names derived from the renamed field updated throughout the codebase:
  `"dist"` → `"luminosity_distance"`,
  `"delta_dist_delta_dist"` → `"delta_luminosity_distance_delta_luminosity_distance"`,
  and the four mixed cross-covariance column names in `datamodels/detection.py`,
  `cosmological_model.py`, `parameter_estimation/evaluation.py`.
- `master_thesis_code_test/datamodels/parameter_space_test.py`,
  `master_thesis_code_test/datamodels/test_detection.py`,
  `master_thesis_code_test/cosmological_model_test.py`: all test fixtures updated
  to use the new `luminosity_distance` column names.
- CI `pytest` step now runs `not gpu and not slow` (slow benchmarks excluded from
  the fast CI path).

---

## [2026-03-10] — code cleanup & quality improvement (Phases 1–7)

### Added
- `master_thesis_code/datamodels/galaxy.py`: extracted `Galaxy` and `GalaxyCatalog` classes
  from `bayesian_inference_mwe.py` into a focused datamodel module.
- `master_thesis_code/datamodels/emri_detection.py`: extracted `EMRIDetection` dataclass.
- `master_thesis_code/bayesian_inference/bayesian_inference.py`: extracted `BayesianInference`
  class and `dist_array` helper; `bayesian_inference_mwe.py` reduced to a thin re-export shim
  plus the `__main__` demonstration script.
- `master_thesis_code/datamodels/detection.py`: extracted `Detection` dataclass and
  `_sky_localization_uncertainty()` from the 3617-line `cosmological_model.py` monolith.
- `scripts/` directory with four utility scripts moved out of the package root:
  `prepare_detections.py`, `remove_detections_out_of_bounds.py`,
  `merge_cramer_rao_bounds.py`, `estimate_hubble_constant.py`.
- `master_thesis_code_test/test_constants.py`: 5 tests — flat universe (Ω_m + Ω_de ≈ 1),
  speed of light value, GPC_TO_MPC, KM_TO_M, RADIAN_TO_DEGREE.
- `master_thesis_code_test/datamodels/test_detection.py`: 6 tests for `Detection` construction,
  field parsing from `pd.Series`, relative distance error, sky localisation error.
- `master_thesis_code_test/datamodels/test_emri_detection.py`: 4 tests — regression for
  `float` fields when `use_measurement_noise=False`, sky angles preserved, noise path positive.

### Changed
- `master_thesis_code/constants.py` (Phase 1): `C` and `G` now derived from
  `astropy.constants` for traceability; added `TRUE_HUBBLE_CONSTANT`, `SPEED_OF_LIGHT_KM_S`,
  `GALAXY_REDSHIFT_ERROR_COEFFICIENT`, `LUMINOSITY_DISTANCE_THRESHOLD_GPC`,
  `FRACTIONAL_LUMINOSITY_ERROR`, `FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR`,
  `FRACTIONAL_MEASURED_MASS_ERROR`, `SKY_LOCALIZATION_ERROR`, and LISA hardware constants
  (`LISA_ARM_LENGTH`, `YEAR_IN_SEC`, `LISA_STEPS`, `LISA_DT`, PSD coefficients).
  Removed duplicate constants previously scattered across `bayesian_inference_mwe.py`,
  `galaxy_catalogue/handler.py`, and `LISA_configuration.py`.
- `master_thesis_code/physical_relations.py` (Phase 2): `hubble_function()` now accepts and
  returns `float | npt.NDArray[np.floating[Any]]`; uses `np.ndim(result) == 0` to decide
  whether to wrap in `float()`. Added `redshifted_mass()` and `redshifted_mass_inverse()`.
- `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py` (Phase 3): removed 12
  locally-duplicated constants and functions (`dist`, `lambda_cdm_analytic_distance`,
  `dist_to_redshift`, `redshifted_mass`, `redshifted_mass_inverse`); imports canonical
  versions from `constants.py` and `physical_relations.py`.
- `master_thesis_code/LISA_configuration.py` (Phase 1): removed 11 inline hardware constants;
  imports them from `constants.py`.
- `master_thesis_code/galaxy_catalogue/handler.py` (Phase 1): removed local `GPC_TO_MPC` and
  `RADIAN_TO_DEGREE`; imports from `constants.py`.
- `master_thesis_code/datamodels/galaxy.py` (Phase 6): unit comments added to all fields
  (`dimensionless`, `M_sun`, `rad`).
- `master_thesis_code/datamodels/detection.py` (Phase 6): unit comments added to all fields
  (`Gpc`, `M_sun`, `rad`, `dimensionless`).
- `master_thesis_code/bayesian_inference/bayesian_inference.py` (Phase 6):
  `luminosity_distance_threshold` field annotated `# Gpc`.

### Fixed (Phase 2 — four documented known bugs)
- **Bug 1** `hubble_function` ndarray crash: union return type prevents `float()` wrapping
  array results; `test_dist_derivative_positive` xfail removed and now passes.
- **Bug 2** `LISAConfiguration` staleness: `test_lisa_config_does_not_go_stale_after_randomize`
  xfail removed; test passes (fix was applied in a prior session).
- **Bug 3** `comoving_volume` hardcoded H₀: `GalaxyCatalog.__init__` now accepts `h0` param
  and passes it to `_build_comoving_volume_spline`; `test_comoving_volume_varies_with_hubble_constant`
  xfail removed and now passes.
- **Bug 4** `dist()` unit inconsistency (Mpc vs Gpc): removed the local `dist()` from
  `bayesian_inference_mwe.py`; all callers now use the canonical Gpc implementation in
  `physical_relations.py`; `luminosity_distance_threshold` updated 1550.0 Mpc → 1.55 Gpc.

### Removed
- Dead commented-out code blocks: multiprocessing derivatives stub in
  `parameter_estimation.py`, waveform-plotting block in `compute_signal_to_noise_ratio`,
  `# import statsmodels.api as sm` in `cosmological_model.py`, alternative galaxy
  distribution implementations in `galaxy.py`.
- `sys.exit()` calls in utility scripts replaced with `return` / natural function end.
- Duplicate `Galaxy`, `GalaxyCatalog`, `EMRIDetection`, `BayesianInference` class bodies
  from `bayesian_inference_mwe.py` (now delegated to the new datamodel/inference modules).

---

## [2026-03-10] — tests for HPC performance refactoring

### Added
- `master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py`: 7 new regression
  and correctness tests for the vectorized hot paths introduced in the HPC refactor —
  `dist_array` shape/dtype, element-wise agreement with scalar `dist()` to `1e-10`,
  strict monotonicity, zero-distance at z=0, comoving-volume spline accuracy vs direct
  trapezoid quadrature (<0.1% relative error at 20 redshifts), spline returns 0 at z=0, and
  `BayesianInference.likelihood()` returning a finite positive float (exercises the full
  vectorized numerator/denominator path).
- `master_thesis_code_test/parameter_estimation/parameter_estimation_test.py`: 3 CPU tests for
  the new buffered-CSV flush mechanism — empty buffer is a no-op (no file, no exception),
  explicit `flush_pending_results()` writes all 3 buffered rows to CSV, and
  `_crb_flush_interval=2` auto-flushes at the threshold with the remainder written on explicit
  flush. Plus 3 `@pytest.mark.gpu` tests: PSD cache identity (second call returns the same
  object), PSD cache shape `(n_channels, n_freqs_cropped)`, and Fisher matrix symmetry
  (mocked derivatives, asserts `np.allclose(F, F.T)`).
- `master_thesis_code_test/LISA_configuration_test.py`: 5 CPU tests (no GPU required) for
  the new `_get_xp()` numpy path — `power_spectral_density('A')`, `power_spectral_density('T')`,
  `S_OMS`, `S_TM` all positive with plain `np.logspace` input; channels A and E return
  identical PSD via `np.allclose`. Module-level `pytest.importorskip("cupy")` replaced with a
  `try/except` guard so the file is collected on CPU-only machines.

---

## [2026-03-10] — comprehensive test coverage & Python 3.13 fix

### Added
- `master_thesis_code_test/decorators_test.py`: 5 new tests for `if_plotting_activated`
  (disabled → returns `None`, enabled → passes return value through) and `timer_decorator`
  (return value, `__name__` preservation, function is actually called).
- `master_thesis_code_test/physical_relations_test.py`: 16 new tests covering `dist(0)==0`,
  monotonicity, float return type, `hubble_function` normalisation and positivity,
  `dist_to_redshift` at zero and round-trip (parametrised over z=0.5/1.0/2.0), vectorised
  shape/value consistency, `dist()` varying with `h` and approximate 1/H₀ scaling,
  mass-conversion algebra (both directions and round-trip), and error-propagation positivity.
  Two `dist_derivative` tests are `xfail` (known bug: `hubble_function` cannot accept ndarray).
- `master_thesis_code_test/datamodels/parameter_space_test.py`: 12 tests for `uniform`,
  `log_uniform`, `polar_angle_distribution`, `ParameterSpace` construction, per-parameter and
  bulk randomisation bounds, `_parameters_to_dict` keys/count/types/NaN safety, and
  `set_host_galaxy_parameters()` updating `dist`, `qS`, `phiS`, `M`.
- `master_thesis_code_test/LISA_configuration_test.py`: 7 tests — 1 CPU (instantiation), 6
  `@pytest.mark.gpu` (PSD positivity for A/E/T channels, A==E identity, S_OMS/S_TM/S_zz
  positivity). Plus `xfail` regression for Known Bug #1 (LISAConfiguration staleness).
- `master_thesis_code_test/parameter_estimation/parameter_estimation_test.py`: 2 CPU tests
  (CSV create/append via monkeypatched path) and 5 `@pytest.mark.gpu` tests (`scalar_product`
  positive-definiteness/symmetry, `_crop_frequency_domain` bounds and length, `_crop_to_same_length`).
- `master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py`: 15 new tests —
  `Galaxy` hashability (set deduplication, hash consistency, inequality), `redshifted_mass`/
  `redshifted_mass_inverse` algebra and round-trip, `dist`/`dist_to_redshift` in mwe module,
  `comoving_volume` positivity and monotonicity, `EMRIDetection.from_host_galaxy` tuple-comma
  regression (`use_measurement_noise=False` → float not tuple), truncnorm distribution type,
  `gw_detection_probability` near-zero and large-redshift bounds, posterior length matching.
  Plus `xfail` regression for Known Bug #3 (comoving_volume hardcoded H₀).
- `master_thesis_code_test/cosmological_model_test.py`: 12 tests — `gaussian` (peak, symmetry,
  positivity), `polynomial` (constant/linear/quadratic), `MBH_spin_distribution` range [0,1],
  and `Detection` dataclass (construction, field values, `get_relative_distance_error`,
  `get_skylocalization_error`, `convert_to_best_guess_parameters`).

### Fixed
- `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py`: `Galaxy` dataclass
  changed to `@dataclass(unsafe_hash=True)` so `Galaxy` instances can be used in sets and as
  dict keys. Fixes `test_add_unique_host_galaxies_from_catalog` (was the one failing test).
- `master_thesis_code/datamodels/parameter_space.py`: all 14 `Parameter` field defaults
  changed from bare mutable instances to `field(default_factory=lambda: Parameter(...))`.
  This is required by Python 3.13 (`dataclasses` now rejects mutable defaults that are not
  wrapped in `field()`). Removes 20 previously-skipped tests and makes `ParameterSpace`
  importable on Python 3.13 without error.

---

## [2026-03-10] — modern dev tooling: ruff, pre-commit, CI, mypy clean

### Added
- `ruff` and `pre-commit` added to `dev` dependency group in `pyproject.toml`.
- `[tool.ruff]` and `[tool.ruff.lint]` configuration in `pyproject.toml`: selects
  `E`, `F`, `I`, `UP`, `B`, `N` rules; line length 100; `target-version = "py313"`.
- `.pre-commit-config.yaml`: ruff (lint + format) and mypy run automatically on
  every `git commit`. mypy uses the project's local environment to avoid false positives.
- `.github/workflows/ci.yml`: CI pipeline runs ruff check, ruff format check, mypy,
  and pytest (CPU only, `not gpu`) on every push and pull request.
- `CLAUDE.md`: added "Dev Workflow" section documenting the linting commands and
  pre-commit usage.

### Changed
- `pyproject.toml`: `[tool.mypy]` `python_version` updated `"3.10"` → `"3.13"`;
  extended `ignore_missing_imports` overrides to cover `pandas`, `scipy`, `sklearn`,
  `mpl_toolkits`, `emcee`, and `tabulate`.
- All 39 source files and 10 test files brought to 0 mypy errors under strict settings
  (`disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`).
  Key changes across the codebase:
  - Removed all `from typing import List, Dict, Optional, Union`; replaced with
    native Python 3.10 syntax (`list[X]`, `dict[K,V]`, `X | None`, `X | Y`).
  - Removed `from __future__ import annotations` from `arguments.py` and
    `bayesian_inference_mwe.py` (per CLAUDE.md convention).
  - All public and private functions annotated with complete return types and
    parameter types.
  - `np.ndarray` → `npt.NDArray[np.float64]` / `npt.NDArray[np.floating[Any]]`
    throughout.
  - `np.trapz` → `np.trapezoid` (numpy 2.x rename).
  - Guarded `import cupy as cp` in `decorators.py`, `memory_management.py`,
    `LISA_configuration.py`, and `parameter_estimation.py` with
    `try/except ImportError` + `_CUPY_AVAILABLE` sentinel.
  - `decorators.py`: converted `TypeVar`-based generics to Python 3.12 type
    parameter syntax (`def f[F: Callable[..., Any]](func: F) -> F:`).
  - `lambda_cdm_analytic_distance` in `physical_relations.py` and
    `bayesian_inference_mwe.py`: removed `float()` cast (broke numpy 2.x when
    called with 1-D array from `fsolve`); `dist()` uses `np.asarray(...).flat[0]`
    for safe scalar extraction.
  - Fixed trailing-comma bug in `EMRIDetection.from_host_galaxy` that made
    `measured_luminosity_distance` and `measured_redshifted_mass` tuples instead
    of floats when `use_measurement_noise=False`.
  - Ruff-formatted all files to 100-character line length with isort ordering.

---

## [2026-03-09] — uv migration & Python 3.13 compatibility

### Added
- `pyproject.toml`: `[project]` section with dependency groups (`cpu`, `gpu`, `dev`),
  replacing the conda `environment.yml` as the authoritative dependency declaration.
- `uv.lock`: generated lock file replacing `conda-linux-64.lock`; commits exact resolved
  versions of all 150 transitive dependencies for bit-for-bit reproducibility.
- `.python-version`: pins to Python 3.13 (`fastlisaresponse` has no cp314 wheel yet).

### Changed
- Dependency manager switched from conda to [uv](https://docs.astral.sh/uv/).
  Motivation: faster installs, pure-pip workflow, `uv.lock` is simpler than
  `conda-lock`, and `fastemriwaveforms`/`fastlisaresponse` now ship cp313 wheels on PyPI.
- Updated `fastlisaresponse` 1.1.17 → 1.1.9 (latest stable with cp313 wheel; 1.1.17
  transitively pinned `numpy==1.26.0` via `lisaanalysistools`, which has no cp313 wheel).
- Updated `fastemriwaveforms` to 2.0.0rc1 (latest release; removes the numpy pin and
  ships a cp313 wheel on PyPI for the CPU variant).
- Scientific stack updated to current versions with cp313 wheels: numpy 2.4.3,
  scipy 1.17.1, matplotlib 3.10.8, pandas 3.0.1.
- `CLAUDE.md` Environment Setup section replaced with uv instructions.

### Fixed
- `BayesianInference` dataclass: `redshift_values` and `galaxy_distribution_at_redshifts`
  fields used bare `np.array([])` as defaults. Python 3.13 now explicitly rejects mutable
  defaults in dataclasses; replaced with `field(default_factory=lambda: np.array([]))`.

---

## [2025-05-08] — cosmological model & galaxy catalog refinements

### Changed
- `cosmological_model.py`: minor tuning of detection probability evaluation logic and
  integration limits in `BayesianStatistics`.
- `galaxy_catalogue/handler.py`: small adjustment to host-galaxy lookup parameters.

---

## [2025-05-04] — bugfix: detection probability (second round)

### Fixed
- `cosmological_model.py`: added plot of interpolated detection probability surface
  alongside the directly-evaluated one, to verify the interpolation is faithful.
  The root cause of the earlier divergence between the two was confirmed fixed.

---

## [2025-04-28] — bugfix: detection probability

### Fixed
- `cosmological_model.py`: phi boundary check was inverted (`phi >= 0` should be
  `phi < 0`); valid azimuth range is `[0, 2π)` so the out-of-range guard was
  accepting invalid values and rejecting valid ones.
- `cosmological_model.py`: `kde.evaluate(...)` returns a length-1 array, not a scalar;
  added `[0]` indexing so the detection probability is a float rather than an array,
  preventing silent broadcasting errors downstream.

### Added
- `cosmological_model.py`: `plot_detection_probability()` method for visual sanity-checking
  of the KDE-based detection probability over the (`d_L`, `M`, `φ`, `θ`) parameter space.

---

## [2025-04-30] — performance improvements & physical relations refactor

### Changed
- `physical_relations.py`: `dist()` now uses an analytic closed-form expression
  (`lambda_cdm_analytic_distance`) instead of the numerical `np.trapz` integral over
  redshift. Faster and avoids discretisation error.
- `physical_relations.py`: added `cached_dist()` with `@lru_cache(maxsize=1000)` so
  repeated calls at the same redshift/cosmology parameters hit the cache instead of
  recomputing the integral. Significant speedup for the inference loop.
- `cosmological_model.py`: extensive rework of the Bayesian inference evaluation loop;
  likelihood computation restructured around interpolated detection-probability functions
  rather than repeated KDE evaluation calls.
- `galaxy_catalogue/handler.py`: added `HostGalaxy.__eq__` and `__hash__` based on
  `catalog_index`, enabling deduplication of host candidates with a set; added
  `HostGalaxy.from_attributes()` classmethod for constructing instances without a
  full catalog row.

---

## [2025-04-25] — BallTree catalog lookups; inference via interpolated functions

### Changed
- `galaxy_catalogue/handler.py`: replaced linear-scan host-galaxy search with a
  scikit-learn `BallTree` on (φ, θ) sky coordinates. Lookup complexity drops from O(N)
  to O(log N) per query; dominant cost for large catalogs.
- `bayesian_inference/bayesian_inference_mwe.py`: inference now evaluates detection
  probability via interpolated functions over a precomputed grid instead of drawing
  Monte Carlo samples. Removes sample-size variance from the posterior and speeds up
  each likelihood call.
- `cosmological_model.py`: significant reduction in size (1 896 → ~300 lines) by
  removing dead evaluation scripts and consolidating the H₀ inference driver into
  `BayesianStatistics.evaluate()`.
- Tests in `test_bayesian_inference_mwe.py` updated to match the new function-based API.
