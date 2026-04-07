# Phase 27: Production Run & Figures - Research

**Researched:** 2026-04-07
**Domain:** Data pipeline, posterior analysis, publication figure generation
**Confidence:** HIGH

## Summary

Phase 27 is an operational/production phase, not a theoretical physics phase. The goal is to (1) run the completeness-corrected Bayesian evaluation on the cluster, (2) extract numerical results (MAP estimates, credible intervals, N_det, precision) from the combined posteriors, and (3) generate publication-quality figures and fill the 25 `\pending{}` and 4 `\todo{}` markers in the paper.

Existing cluster results in `cluster_results/eval_corrected_full/` contain posteriors from a **completeness-corrected** evaluation with 534 total events. However, the current combined posteriors show MAP at h=0.66 (without BH mass) and h=0.68 (with BH mass), both significantly biased low relative to h_true=0.73. This is the key physics finding: the completeness correction has not fully resolved the low-h bias. The production run may need to verify whether this is expected given the current injection campaign and grid resolution, or whether it indicates a remaining issue.

**Primary recommendation:** The existing `eval_corrected_full` results appear to be the production data (538 events per h-value, 23 h-values for "no mass" variant, 9 for "with mass"). The phase should focus on: (1) extracting numbers and computing credible intervals from these results, (2) generating the 15-figure manifest via `--generate_figures`, (3) creating the 4 paper-specific figures not in the manifest (posterior comparison, single-event likelihoods, convergence, SNR distribution), and (4) filling all `\pending{}` markers in the LaTeX source. A new cluster run is needed only if the with-BH-mass h-grid is too sparse (9 values vs 23).

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| `cluster_results/eval_corrected_full/` | prior artifact | Contains all production posterior data | read, extract numbers | plan, execution |
| `paper/sections/results.tex` | prior artifact | Has 25 `\pending{}` markers to fill | read, update | execution |
| Gray et al. (2020), arXiv:1908.06050, Eq. 9 | method | Completeness correction formula used in evaluation | cite | paper |
| Thesis baseline (h=0.712 no mass, h=0.742 with mass) | benchmark | P_det=1 comparison point | compare | results discussion |

**Missing or weak anchors:** The with-BH-mass variant has only 9 h-values (vs 23 for without-BH-mass) and one corrupted JSON file (`h_0_64.json`). This may produce a poorly resolved posterior that is not publication-quality. A re-run of the with-BH-mass evaluation on a finer grid may be needed.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Hubble parameter | h = H0/(100 km/s/Mpc), dimensionless | H0 in km/s/Mpc | Project standard |
| True value | h_true = 0.73 | -- | `constants.py` |
| Distance units | Gpc (code), Mpc (completeness lookup) | -- | `glade_completeness.py` |
| Figure format | PDF (publication), PNG (preview) | -- | `save_figure()` default |
| Figure widths | single=3.375in, double=7.0in (REVTeX) | -- | `_helpers.py` |
| Normalization | Peak-normalized (posterior plots) | Density-normalized | `bayesian_plots.py` |
| Zero-handling | physics-floor strategy | naive, exclude, per-event-floor | `posterior_combination.py` |

**CRITICAL: All equations and results below use these conventions.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| p(h\|data) = prod_i p_i(x_GW\|h) | Joint posterior (product of single-event likelihoods) | Gray et al. (2020) Eq. 6 | Combined in log-space by `combine_log_space()` |
| p_i = f_i * L_cat + (1-f_i) * L_comp | Completeness-corrected likelihood | Gray et al. (2020) Eq. 9 | Already computed; stored in per-event JSON files |
| CI width = h_84 - h_16 | 68% credible interval | Standard | Extract from combined posterior CDF |
| MAP = argmax p(h\|data) | Maximum a posteriori | Standard | Read from `combined_posterior.json` |
| Precision = CI_width / (2 * MAP) * 100 | Percentage precision | Convention | Compute for paper table |
| Bias = (MAP - h_true) / h_true * 100 | Percentage bias | Convention | Compute for paper table |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Log-space combination | Numerically stable product of likelihoods | `combine_log_space()` | Already implemented |
| CDF inversion for CI | Extract credible intervals from discrete posterior | `_credible_interval_width()` | `convergence_plots.py` |
| Trapezoidal integration | Normalize posterior density | `np.trapezoid()` | Standard |

### Approximation Schemes

No new approximations in this phase. All physics computations are complete; this phase extracts numbers and makes figures.

## Standard Approaches

### Approach 1: Extract from Existing Results (RECOMMENDED)

**What:** Use `cluster_results/eval_corrected_full/` data directly. Run `--combine` to regenerate `combined_posterior.json` with the physics-floor strategy. Then compute credible intervals and fill paper markers.

**Why standard:** The data is already computed. Re-running the cluster evaluation is expensive (~1h per h-value) and unnecessary unless the h-grid is too sparse.

**Key steps:**

1. Verify data integrity: check all JSON files load correctly, count events, identify corrupted files
2. Run `--combine` on `posteriors/` and `posteriors_with_bh_mass/` directories
3. Extract MAP, 68% CI, 95% CI from combined posteriors
4. Compute precision (%) and bias (%) for both variants
5. Generate figures via `--generate_figures`
6. Create paper-specific figures not in the manifest (posterior comparison overlay, single-event likelihoods)
7. Fill all `\pending{}` markers in `results.tex`, `main.tex`, `conclusions.tex`

**Known difficulties at each step:**

- Step 1: `posteriors_with_bh_mass/h_0_64.json` is corrupted (JSON decode error at char 108887040). This file needs to be excluded or regenerated.
- Step 2: The `--combine` flag writes output back to the same directory as input. The existing `combined_posterior.json` will be overwritten.
- Step 5: The `--generate_figures` function looks for `posteriors_without_bh_mass/` but the actual directory is `posteriors/`. This is a BUG in `main.py` line 724 that must be fixed before figure generation will work.
- Step 6: Several paper figures (posterior comparison overlay, single-event likelihoods) are not in the 15-figure manifest and need custom code.

### Approach 2: Re-run Cluster Evaluation (FALLBACK)

**What:** Submit a new evaluation run on the cluster with a denser h-grid, especially for the with-BH-mass variant.

**When to switch:** If the 9-point with-BH-mass grid is too coarse for a smooth posterior curve, or if the corrupted JSON cannot be recovered.

**Tradeoffs:** Requires cluster access (currently blocked by filesystem recovery). Takes ~11h total (11 h-values x 1h each). But produces consistent h-grids for both variants.

### Anti-Patterns to Avoid

- **Re-running the full simulation pipeline:** The EMRI simulation (SNR + CRB computation) is already done. Only the evaluation (`--evaluate`) and combination (`--combine`) steps are needed. Do NOT re-run `--simulation_steps`.
- **Manually editing JSON files:** Do not hand-edit posterior JSON files. Use `--combine` with the appropriate `--strategy` flag.
- **Using the wrong directory name:** The code has an inconsistency: `evaluate` writes to `posteriors/`, but `generate_figures` reads from `posteriors_without_bh_mass/`. Fix the code, do not rename directories.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| Combined posterior (no BH mass) | MAP h=0.66, 531 events used | `eval_corrected_full/combined_posterior.json` | Read directly, recompute CI |
| Combined posterior (with BH mass) | MAP h=0.68, 527 events used | `eval_corrected_full/combined_posterior_with_bh_mass.json` | Read directly, recompute CI |
| Thesis baseline (no mass) | h=0.712 +/- 0.028 (P_det=1) | `paper/sections/results.tex` line 134 | Comparison benchmark |
| Thesis baseline (with mass) | h=0.742 +/- 0.013 (P_det=1) | `paper/sections/results.tex` line 147 | Comparison benchmark |
| Total events | 534 total, 531/527 used (no/with mass) | `combined_posterior.json` | Paper table |
| Detection threshold | SNR >= 15 (paper) or 20 (code default) | `results.tex` line 26, `constants.py` | Verify consistency |
| Sigma(d_L)/d_L cut | < 0.10 | `results.tex` line 46 | Already applied in evaluate |

**Key insight:** The existing posteriors have 23 h-values for the no-mass variant (0.60 to 0.86) but only 9 h-values for the with-mass variant. The combined posteriors are extremely peaked (probability concentrated in 1-2 bins), which is a characteristic of having many events. The 68% CI extraction from such peaked distributions requires careful interpolation on the discrete grid.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| Per-event posterior JSONs | Individual p_i(h) for each detection | `posteriors/h_*.json` | 538 events x 23 h-values |
| Existing PNG plots | 8 diagnostic plots already generated | `eval_corrected_full/plots/` | May need reformatting for publication |
| `_credible_interval_width()` | Ready-made CI computation | `convergence_plots.py` line 26 | Needs h_values array and posterior array |
| `plot_combined_posterior()` | Ready-made posterior plot with Planck/SH0ES bands | `bayesian_plots.py` | Takes h_values, posterior, true_h |
| `plot_h0_convergence()` | Two-panel convergence plot | `convergence_plots.py` | Takes h_values, event_posteriors |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Gray et al. | Gray, Messenger, Veitch | 2020 | Completeness correction method | Already implemented; cite Eq. 9 |
| Dalya et al. | Dalya et al. | 2022 | GLADE+ completeness data | Digitized in `glade_completeness.py` |
| Babak et al. | Babak et al. | 2017 | EMRI parameter estimation benchmark | Table III validation |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| `posterior_combination.py` | `combine_posteriors()` | Combine per-event posteriors | Already built, tested |
| `bayesian_plots.py` | `plot_combined_posterior()`, `plot_event_posteriors()` | Main posterior figures | Factory functions, `(fig, ax)` out |
| `convergence_plots.py` | `plot_h0_convergence()` | Convergence diagnostic | Two-panel figure, CI width vs N |
| `main.py` | `generate_figures()` | 15-figure manifest | Full pipeline from data to PDF |
| matplotlib | `emri_thesis.mplstyle` | Consistent styling | Project-wide style sheet |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| `_helpers.py:get_figure()` | Create figures with REVTeX presets | All new figures |
| `_helpers.py:save_figure()` | Save as PDF, check size | All figure output |
| `_style.py:apply_style(use_latex=True)` | Enable LaTeX rendering for publication | Final production figures |
| `_colors.py` | Consistent color scheme (CYCLE, TRUTH, REFERENCE) | All plots |
| `np.trapezoid()` | Posterior normalization | CI computation |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| `--combine` on existing data | < 1 minute | I/O loading 538 x 23 JSON files | Already fast |
| `--generate_figures` | < 5 minutes | 15 figures, some with many curves | Already implemented |
| CI extraction | Seconds | Discrete grid interpolation | Use `np.interp` on CDF |
| New cluster evaluation (if needed) | ~11h wall time | 11 SLURM array tasks x 1h | Only if with-BH-mass grid too sparse |

**Installation / Setup:**
```bash
# No new packages needed. All tools are in the existing dependency tree.
uv sync --extra cpu --extra dev
```

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| Posterior normalization | Sum/integral = 1 | `np.trapezoid(posterior, h_values)` after normalization | Should be 1.0 to machine precision |
| MAP within grid | MAP estimate on grid | Check `map_h` is not at grid boundary | Should be interior point; boundary = suspicious |
| Event count consistency | Same events in both variants | Compare `n_events_total` across variants | 534 total for both |
| CI width positive | Meaningful constraint | `h_84 - h_16 > 0` | CI > 0 for any informative posterior |
| Bias direction | Completeness correction effect | Compare MAP with P_det=1 baseline | Should shift toward h_true=0.73 (TBD) |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| P_det = 1 baseline (no mass) | All events, no selection correction | h = 0.712 +/- 0.028 | Thesis |
| P_det = 1 baseline (with mass) | All events, no selection correction | h = 0.742 +/- 0.013 | Thesis |
| True injected value | -- | h = 0.73 | `constants.py` |
| N^{-1/2} scaling | Large N | CI width ~ 1/sqrt(N_det) | Statistical expectation |

### Red Flags During Computation

- **MAP at grid boundary (h=0.60 or h=0.86):** Indicates the posterior is not well-captured by the grid. Need to extend the h-range.
- **CI width = full grid width:** Posterior is uninformative, likely a bug in combination.
- **Posterior all zeros except one bin:** Over-concentration due to too many events with identical peaking. Check zero-handling strategy.
- **N_events_used << N_events_total:** Excessive exclusion by physics-floor strategy. Check for systematic issues.
- **with-BH-mass precision WORSE than without-BH-mass:** Physically impossible given the mass information adds constraining power. Indicates a bug.

## Common Pitfalls

### Pitfall 1: Directory Name Mismatch

**What goes wrong:** `--generate_figures` looks for `posteriors_without_bh_mass/` (line 724 of `main.py`) but the evaluation writes to `posteriors/`. All posterior-dependent figures will be skipped with "required data not found".
**Why it happens:** Naming convention changed at some point; `generate_figures` was not updated.
**How to avoid:** Fix `main.py` line 724 to use `"posteriors"` instead of `"posteriors_without_bh_mass"`, OR create a symlink, OR rename the directory.
**Warning signs:** "Skipping figXX: required data not found" in logs.
**Recovery:** Change the string in `main.py` and re-run.

### Pitfall 2: Extremely Peaked Posteriors

**What goes wrong:** With 531 events, the combined posterior is astronomically peaked. The `combined_posterior.json` shows values like 3.8e-241 at h=0.60 and 0.948 at h=0.66. The CDF is essentially a step function, making CI extraction numerically fragile.
**Why it happens:** Product of ~500 single-event likelihoods concentrates probability in 1-2 bins.
**How to avoid:** Use density-normalized posterior for CI extraction. Ensure interpolation is performed on the CDF, not the posterior directly.
**Warning signs:** CI width exactly equals one grid spacing (0.02), or CI width is zero.
**Recovery:** Interpolate the CDF at finer resolution between grid points before extracting quantiles.

### Pitfall 3: Corrupted JSON File

**What goes wrong:** `posteriors_with_bh_mass/h_0_64.json` has a JSON decode error at char ~109M. This will crash `load_posterior_jsons()` and prevent combination.
**Why it happens:** Likely truncated write during cluster job (timeout, disk issue).
**How to avoid:** Add try/except around individual file loading in `load_posterior_jsons()`, or exclude the file.
**Warning signs:** `JSONDecodeError: Expecting value: line 1 column 108887041`.
**Recovery:** Remove the corrupted file before running `--combine`. The h=0.64 point will be missing from the with-BH-mass posterior, but with 8 remaining points it may still be usable.

### Pitfall 4: SNR Threshold Inconsistency

**What goes wrong:** The paper says SNR_thr = 15 (line 26 of results.tex), but the code default in `constants.py` is SNR_THRESHOLD = 20, and `generate_figures` uses `snr >= 20.0` (line 843).
**Why it happens:** The threshold was changed at some point; paper and code diverged.
**How to avoid:** Verify which threshold was actually used in the production simulation. Check `constants.py` and the cluster job logs.
**Warning signs:** N_det in paper does not match N_det from data.
**Recovery:** Update the paper to match whichever threshold was actually used.

### Pitfall 5: LaTeX Rendering on Headless Systems

**What goes wrong:** `apply_style(use_latex=True)` requires a TeX installation. On systems without LaTeX, figures will fail to render.
**Why it happens:** The dev machine or CI environment may not have a full TeX installation.
**How to avoid:** Use `use_latex=False` for draft figures (mathtext fallback). Only enable LaTeX for final production figures on a machine with TeX installed.
**Warning signs:** `RuntimeError: Failed to process string with tex`.
**Recovery:** Run with `use_latex=False` and manually re-render on a TeX-equipped machine.

## Level of Rigor

**Required for this phase:** Numerical extraction with careful attention to discretization effects.

**Justification:** This is a production phase. The physics computations are complete. The rigor requirement is on accurate numerical extraction (CI computation, normalization) and figure quality (correct labels, consistent style, publication format).

**What this means concretely:**

- CI extraction must handle the extremely peaked posteriors correctly (interpolate CDF)
- All numbers in the paper must be traceable to specific JSON files
- Figures must use the `emri_thesis.mplstyle` and REVTeX single/double column widths
- PDF output at 300 DPI with vector elements (not rasterized except for dense scatter plots)

## State of the Art

Not applicable -- this is an operational phase, not a methods phase.

## Open Questions

1. **Is the low MAP (h=0.66/0.68) the correct production result, or does it indicate a remaining bug?**
   - What we know: The completeness correction was supposed to reduce the low-h bias. The thesis baseline (P_det=1) had MAP=0.712/0.742. The corrected result has MAP=0.66/0.68, which is *more* biased, not less.
   - What's unclear: Whether this is physically correct (completeness correction can shift the posterior in either direction depending on the data) or indicates an implementation issue (e.g., the P_det normalization is incorrect).
   - Impact on this phase: If the result is wrong, the paper numbers will be wrong. This is the most critical question.
   - Recommendation: Proceed with the current data but include a comparison table showing P_det=1 vs corrected results. Flag the bias direction as a discussion point.

2. **Is the 9-point h-grid for with-BH-mass sufficient?**
   - What we know: Only 9 h-values vs 23 for without-BH-mass. The combined posterior concentrates in 1-2 bins.
   - What's unclear: Whether 9 points gives a smooth enough curve for publication.
   - Impact on this phase: May need a cluster re-run with the full 23-point grid for the with-BH-mass variant.
   - Recommendation: Generate figures with the 9-point data first. If the posterior curve is too jagged, prioritize a cluster re-run for the with-BH-mass variant only.

3. **Should the paper report the MAP or the median?**
   - What we know: The paper currently uses MAP estimates. For extremely peaked posteriors, MAP and median are nearly identical.
   - What's unclear: Standard practice in GW cosmology papers.
   - Impact on this phase: Affects all numerical values in the paper.
   - Recommendation: Report MAP (consistent with thesis baseline) but also compute the median as a cross-check.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| Extract from existing data | Corrupted files, too few h-values | Re-run cluster evaluation | ~11h wall time, needs cluster access |
| `--generate_figures` | Directory name bug, missing data | Manual figure generation scripts | ~2h coding, but fragile |
| CDF interpolation for CI | Posterior too peaked for discrete grid | Fit a Gaussian to the posterior peak | Quick, but assumption-dependent |
| Physics-floor strategy | Too many events excluded | Switch to per-event-floor or exclude | Just change `--strategy` flag |

**Decision criteria:** If more than 10% of events are excluded by physics-floor, investigate the zero-likelihood pattern before switching strategies.

## Data Pipeline: Complete Sequence

The end-to-end pipeline from existing cluster results to paper is:

```
1. cluster_results/eval_corrected_full/posteriors/h_*.json  (23 files, 538 events each)
   cluster_results/eval_corrected_full/posteriors_with_bh_mass/h_*.json  (9 files, 540 events)
                              |
2. --combine --strategy physics-floor
                              |
3. combined_posterior.json  (h_values + normalized posterior array)
                              |
4. Extract: MAP, 68% CI, 95% CI, precision, bias
                              |
5. --generate_figures  (15-figure manifest -> PDF)
   + custom paper figures (posterior comparison, single-event, convergence)
                              |
6. Fill \pending{} markers in results.tex, main.tex, conclusions.tex
```

## Figure Mapping: Paper Needs vs Code Capabilities

| Paper Figure | `\todo{}` Location | Available Code | Data Source | Notes |
| --- | --- | --- | --- | --- |
| SNR distribution | results.tex:55 | `plot_snr_distribution()` in manifest (fig03) | CRB CSV (not in cluster_results) | Needs CRB data; may need to copy from cluster |
| H0 posterior comparison | results.tex:173 | `plot_combined_posterior()` -- needs overlay of BOTH variants | Combined posterior JSONs | Requires custom two-curve overlay plot |
| Single-event likelihoods | results.tex:198 | `plot_event_posteriors()` -- needs selection of 4 events | Per-event JSONs | Need to pick representative events at different z |
| Convergence vs N | results.tex:223 | `plot_h0_convergence()` in manifest (fig08) | Per-event posteriors | Already in manifest but paper wants specific format |

## Pending Marker Inventory

Total: 25 `\pending{}` + 4 `\todo{}` + 1 acknowledgments `\todo{}`

### Numerical values (from combined posteriors):
- N_det after sigma cut (line 48)
- h^(no mass) MAP +/- CI (line 132)
- h^(no mass) precision (line 138)
- h^(mass) MAP +/- CI (line 143)
- h^(mass) precision (line 149)
- N_det for figure caption (line 177)
- Subset size for convergence (line 213)
- N_subsets (line 227)
- Summary table: 10 values (lines 356-368)
- Abstract precision values (main.tex line 53)
- Conclusions precision (conclusions.tex)

### Qualitative results (require analysis):
- Realistic P_det results (line 250)
- Completeness correction effect (line 295)
- P_det normalization effect (line 339)

### Figures (require code + data):
- SNR distribution (line 55)
- H0 posterior comparison (line 173)
- Single-event likelihoods (line 198)
- Convergence plot (line 223)

## Sources

### Primary (HIGH confidence)

- Project codebase: `master_thesis_code/main.py` -- `generate_figures()` manifest, `--combine` entry point
- Project codebase: `master_thesis_code/bayesian_inference/posterior_combination.py` -- combination logic, strategies
- Project codebase: `master_thesis_code/plotting/bayesian_plots.py` -- all posterior plotting functions
- Project codebase: `master_thesis_code/plotting/convergence_plots.py` -- convergence diagnostics, CI extraction
- Cluster results: `cluster_results/eval_corrected_full/` -- production posterior data

### Secondary (MEDIUM confidence)

- Gray et al. (2020), arXiv:1908.06050 -- completeness correction formula, Eq. 9
- Dalya et al. (2022), arXiv:2110.06184 -- GLADE+ completeness data, Fig. 2
- REVTeX 4.2 documentation -- column width standards for figures

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH -- all computations are standard (CDF inversion, normalization, argmax)
- Standard approaches: HIGH -- data extraction pipeline is straightforward, tools exist
- Computational tools: HIGH -- all plotting and combination tools are already implemented and tested
- Validation strategies: HIGH -- comparison with thesis baseline is well-defined

**Research date:** 2026-04-07
**Valid until:** Indefinitely (operational phase, no expiring methodology)

## Caveats and Alternatives

1. **The low MAP bias may be correct.** The completeness correction adds a completion term that accounts for galaxies missing from the catalog. Depending on the shape of the completion likelihood, this can shift the posterior toward lower h if the completion term favors lower redshifts (i.e., more volume at lower h). The paper should discuss this carefully rather than assuming the correction must bring the posterior closer to h_true.

2. **The 15-figure manifest may produce more figures than the paper needs.** The manifest includes diagnostic figures (campaign dashboard, CRB coverage) that are not referenced in the paper's `\todo{}` markers. The planner should distinguish between paper figures (must be publication-quality) and supplementary figures (nice to have).

3. **The directory naming bug is a software issue, not a physics issue.** The fix is trivial (change one string), but it blocks the entire `--generate_figures` pipeline. This should be the first task in the phase.

4. **CRB CSV data may not be available locally.** The `--generate_figures` manifest requires CRB CSV files for SNR distribution, sky localization, and Fisher ellipse figures. These files are on the cluster, not in `cluster_results/eval_corrected_full/`. Several manifest figures will be skipped unless CRB data is copied from the cluster.
