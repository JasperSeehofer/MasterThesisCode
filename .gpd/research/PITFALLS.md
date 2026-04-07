# Visualization and Figure-Quality Pitfalls

**Physics Domain:** EMRI dark siren H0 inference -- publication figure preparation for PRD
**Researched:** 2026-04-07
**Context:** Upgrading matplotlib visualization pipeline for Physical Review D submission. Colorblind-safe, grayscale-safe, APS-compliant figures from discrete 15-point h-grid posteriors and 580MB per-galaxy JSON files.

## Critical Pitfalls

Mistakes that make figures unusable for review, mislead readers, or cause APS rejection.

### Pitfall 1: Type 3 Fonts in PDF Output Cause APS Rejection

**What goes wrong:** By default, matplotlib embeds Type 3 fonts in PDF output. APS production runs automated quality checks and rejects PDFs with Type 3 fonts. Adobe Acrobat also does not recognize Type 3 as embedded, which triggers submission system diagnostics.

**Why it happens:** matplotlib's default `pdf.fonttype = 3` uses bitmap-based font subsets. Type 3 fonts are technically valid but poorly supported by some PDF viewers and journal production workflows.

**Consequences:** Manuscript rejection at the submission stage with an opaque error about non-embedded fonts. This typically happens late in the submission process when time pressure is highest.

**Prevention:** Set in `emri_thesis.mplstyle` or at the top of `apply_style()`:
```
pdf.fonttype: 42
ps.fonttype: 42
```
This forces TrueType (Type 42) font embedding for both PDF and PostScript output. The existing `emri_thesis.mplstyle` does NOT currently set these -- this is a gap.

**Detection:** Run `pdffonts paper/figures/*.pdf` (from poppler-utils). Any line showing "Type 3" in the "type" column is a problem. Alternatively, open the PDF in Adobe Acrobat, go to File > Properties > Fonts, and check that all fonts say "(Embedded Subset)" with type TrueType or Type 1.

**References:**
- [Avoiding Type 3 fonts in matplotlib plots](http://phyletica.org/matplotlib-fonts/)
- [matplotlib docs: Fonts in Matplotlib](https://matplotlib.org/stable/users/explain/text/fonts.html)
- [APS Style Basics](https://journals.aps.org/authors/style-basics)

### Pitfall 2: Current Color Palette (tab10) Is Not Colorblind-Safe

**What goes wrong:** The current `_colors.py` CYCLE uses tab10 colors (`#1f77b4`, `#ff7f0e`, `#2ca02c`, `#d62728`, ...). The tab10 palette is NOT designed for color vision deficiency (CVD) accessibility. Specifically:
- `#2ca02c` (green, used for TRUTH) and `#d62728` (red, used for MEAN and "With Mz" variant) are nearly indistinguishable under deuteranopia (the most common form, affecting ~6% of males).
- In `paper_figures.py`, the H0 posterior comparison plots the "Without Mz" curve in blue (`CYCLE[0]`) and "With Mz" in red (`CYCLE[3]`). These two colors collapse to similar olive tones under protanopia.

**Why it happens:** tab10 was designed for screen-only distinguishability in its original Tableau context, not for print or CVD accessibility. The matplotlib community has moved away from it for publication work.

**Consequences:** ~8% of male readers (4.5% of all readers) cannot distinguish the two posterior variants in the key result figure. Referees with CVD may reject the paper on accessibility grounds. APS explicitly recommends "accessible color palettes."

**Prevention:** Replace the CYCLE with the Okabe-Ito palette (Wong, Nature Methods 2011):
```python
CYCLE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow (use sparingly, low contrast on white)
    "#999999",  # grey
]
```
For the two-curve posterior comparison, use blue (`#0072B2`) and vermillion (`#D55E00`) -- these are maximally separated under all three common CVD types (deuteranopia, protanopia, tritanopia).

For semantic colors:
- TRUTH: `#009E73` (bluish green) -- distinct from both blue and vermillion under CVD
- MEAN: `#D55E00` (vermillion)
- REFERENCE: `#999999` (grey) -- unchanged, grayscale-safe by definition

**Detection:** Use the `daltonize` Python package to simulate CVD on rendered figures:
```python
from daltonize import simulate_mpl
fig_deut = simulate_mpl(fig, color_deficit="d", copy=True)  # deuteranopia
fig_prot = simulate_mpl(fig, color_deficit="p", copy=True)  # protanopia
```
Or use `colorspacious` for pixel-level simulation. Alternatively, the Coblis online simulator (color-blindness.com/coblis-color-blindness-simulator/) works on exported PNGs.

**References:**
- Wong, B. (2011). "Points of view: Color blindness." Nature Methods, 8(6), 441. doi:10.1038/nmeth.1618
- [Okabe-Ito palette reference](https://conceptviz.app/blog/okabe-ito-palette-hex-codes-complete-reference)
- [daltonize](https://github.com/joergdietrich/daltonize)

### Pitfall 3: Connecting 15 Discrete h-Grid Points with Lines Implies Continuous Measurement

**What goes wrong:** The current `paper_figures.py` plots posteriors as `"o-"` (markers + connecting line). With only 15 h-grid points spanning [0.60, 0.90], the connecting line implies the posterior was evaluated continuously. A reviewer who notices the coarse grid may question whether the posterior shape is an artifact of interpolation or represents real structure.

**Why it happens:** The "o-" format string is a natural matplotlib default. With 15 points and a line connecting them, the visual impression is of a smooth curve evaluated at 15 sample points, when in reality the posterior was ONLY evaluated at those 15 points.

**Consequences:**
1. **Misleading shape:** Linear interpolation between 15 points can miss curvature. If the true posterior has a shoulder or asymmetry between grid points, the figure will not show it.
2. **False precision in CI:** The 68% CI is computed via CDF interpolation to a 1000-point fine grid (`h_fine = np.linspace(..., 1000)`). This interpolation assigns sub-grid precision that does not exist in the data, potentially over-reporting CI width precision.
3. **Reviewer distrust:** Experienced Bayesian reviewers will notice that `p(h | data)` is evaluated on only 15 points and may question whether the grid is adequate.

**Prevention:**
- **Always show markers** at the actual grid points (already done with `"o-"`, good).
- **Do NOT smooth or spline** the 15-point posterior. The honest representation is markers connected by straight lines (current approach) or a step function.
- **State the grid resolution** in the figure caption: "Posterior evaluated at 15 values of h in [0.60, 0.90]."
- **For CI calculation:** Report CIs rounded to the grid spacing (0.02 in h). Do not quote CI boundaries to three decimal places when the grid spacing is 0.02.
- **Consider a step-function representation** (bar plot or step plot) as an alternative that honestly communicates the discrete nature.

**Detection:** Check whether the quoted CI boundaries fall on grid points. If h_16 = 0.683 but the nearest grid points are 0.68 and 0.70, the reported boundary is an interpolation artifact.

### Pitfall 4: Credible Interval Calculation Using Raw cumsum Instead of Trapezoidal CDF

**What goes wrong:** In `bayesian_plots.py` line 99-101, the credible interval is computed as:
```python
cumsum = np.cumsum(normalized)
cumsum = cumsum / cumsum[-1]
```
This is a cumulative SUM, not a cumulative INTEGRAL. On a non-uniform h-grid (or even a uniform one), `np.cumsum` treats each posterior value as having equal weight regardless of the spacing between h points. The correct CDF is the cumulative trapezoidal integral: `CDF(h_i) = integral from h_0 to h_i of p(h) dh`.

In contrast, `paper_figures.py` lines 216-224 correctly use `np.trapezoid` for the CDF calculation. This inconsistency means the same posterior will produce different CI boundaries depending on which plotting function is called.

**Why it happens:** `np.cumsum` is the quick approximation. For a uniform grid it differs from trapezoidal integration by O(dh), which for 15 points with dh = 0.02 produces ~1% error in CI boundaries. For non-uniform grids the error is larger.

**Consequences:** Inconsistent CI widths reported in different figures. If the thesis shows a dashboard plot (using `bayesian_plots.py`) and a paper figure (using `paper_figures.py`), the CIs may differ by 1-2% in h, which is confusing.

**Prevention:** Use trapezoidal integration consistently. Replace the `cumsum` approach in `bayesian_plots.py` with the same `np.trapezoid`-based CDF as in `paper_figures.py`:
```python
cdf = np.zeros(len(h_values))
for i in range(1, len(h_values)):
    cdf[i] = cdf[i-1] + np.trapezoid(pn[i-1:i+1], h_values[i-1:i+1])
cdf /= cdf[-1]
```

**Detection:** Compute CIs both ways for the same posterior and compare. Any difference > 0.5% in h indicates a problem.

## Moderate Pitfalls

### Pitfall 5: Grayscale Collapse -- Lines Distinguished Only by Color

**What goes wrong:** When printed in grayscale (as many readers will do for PRD papers), lines distinguished only by color become indistinguishable. The current `paper_figures.py` uses blue solid line with circles (`"o-"`) vs red dashed line with squares (`"s--"`) for the two posterior variants. The marker+linestyle distinction is good, but other plots in the pipeline (e.g., `plot_event_posteriors` in `bayesian_plots.py`) plot hundreds of lines in the same color with alpha=0.3, which becomes a uniform grey wash in grayscale.

**Why it happens:** Color is the easiest visual variable to manipulate in matplotlib. Linestyle and marker cycling require explicit setup.

**Consequences:** Figures become unreadable in printed copies. Many institutional printers default to grayscale. Senior faculty who print papers to read them cannot distinguish data series.

**Prevention:**
- For the two-variant posterior comparison: the current `"o-"` vs `"s--"` approach provides redundant encoding (marker shape + linestyle). This is correct. Verify that the two semantic colors chosen also have distinct lightness values in grayscale (blue maps to ~40% grey, vermillion maps to ~55% grey -- sufficient contrast).
- For multi-line plots (event posteriors): use a sequential lightness ramp (light to dark) instead of alpha blending. Or use the `color_by` parameter with viridis (which is perceptually uniform and monotonic in lightness, so it degrades gracefully to grayscale).
- For histograms: use hatching patterns in addition to color fill. matplotlib supports: `'/'`, `'\\'`, `'|'`, `'-'`, `'+'`, `'x'`, `'o'`, `'O'`, `'.'`, `'*'`.
- **Test grayscale rendering** by applying `plt.style.use('grayscale')` temporarily and checking readability.

**Detection:** Export figure as PNG, convert to grayscale with PIL: `img.convert('L')`. Check if all data series remain distinguishable.

### Pitfall 6: APS Figure Size Mismatch -- Font Too Small After Scaling

**What goes wrong:** The current `_helpers.py` defines `"single": (3.375, 3.375 / 1.618)` which is correct for REVTeX single-column width (3.375 inches = 8.5 cm). However, the `emri_thesis.mplstyle` sets `font.size: 11` and `axes.labelsize: 12`. When the figure is rendered at 3.375 inches wide with 12pt axis labels, the labels are physically correct. But if APS production scales the figure down (common for two-column layouts), fonts shrink below the 2mm minimum capital height requirement.

**Why it happens:** APS requires that "the size of the smallest capital letters and numerals should be at least 2 mm." A 9pt font at 300 DPI has a capital height of ~2.1mm, which is borderline. If the figure is scaled to 80% of its original size during production, 9pt becomes 7.2pt (~1.7mm), which violates the requirement.

**Consequences:** APS production may reject the figure or request a revision. At minimum, small fonts cause readability problems.

**Prevention:**
- Use the `apply_style(use_latex=True)` path which sets `font.size: 10`, `axes.labelsize: 10`, `xtick.labelsize: 9`. These are the correct sizes for REVTeX.
- **Never go below 8pt** for any text element in a single-column figure.
- **Design at final size:** Always create figures at the exact output size (3.375" for single-column, 7.0" for double-column). Never create at a larger size and scale down.
- The `"double": (7.0, 7.0 / 1.618)` preset is correct for APS double-column width.
- Verify: render the figure, export at 300 DPI, measure the physical height of the smallest text in the PNG. It must exceed 2mm.

**References:**
- [APS Style Basics](https://journals.aps.org/authors/style-basics): "Smallest capital letters >= 2 mm, data points >= 1 mm diameter, linewidth >= 0.18 mm (0.5 pt)"
- [APS Author Guide for REVTeX 4.2](https://ctan.math.illinois.edu/macros/latex/contrib/revtex/aps/apsguide4-2.pdf)

### Pitfall 7: Loading Multiple 580MB JSON Files Exhausts Memory

**What goes wrong:** The `_load_per_event_with_mass_scalars` function in `paper_figures.py` already implements the tail-read optimization (reading only the last 300KB of each 580MB file). However, the `_load_per_event_no_mass` function loads ENTIRE JSON files with `json.load(fh)`. With 15 h-grid points, this loads 15 full JSON files into memory simultaneously (stored in `raw` dict). If the no-mass JSON files are also large, this can exhaust memory on a development machine.

**Why it happens:** The no-mass JSON files are smaller than the with-mass files (because they lack per-galaxy breakdowns), but they still contain per-event data for hundreds of events. The `raw` dict holds all 15 parsed JSON objects in memory simultaneously.

**Consequences:** Out-of-memory crash or severe swapping on machines with 16GB RAM. The crash happens during figure generation, not during the main simulation, so it may surprise the user.

**Prevention:**
- **Process one file at a time:** Instead of loading all 15 JSONs into `raw`, process each file immediately and discard the raw data:
  ```python
  for f in files:
      h = _h_from_file(f)
      with open(base / f) as fh:
          data = json.load(fh)
      # Extract per-event values immediately
      for eid in event_ids:
          events[eid].append(data.get(eid, [0.0])[0])
      del data  # free memory
  ```
- **For the with-mass files:** The tail-read approach in `_load_per_event_with_mass_scalars` is already correct and memory-efficient. Do not change it.
- **Memory monitoring:** Add a simple check before loading: `os.path.getsize(filepath)` and warn if total size exceeds available memory.
- **Do NOT use `ijson` for this case:** The JSON files are structured as dicts, and the tail-read regex approach is faster and simpler than streaming for extracting scalars from the end of the file.

**Detection:** Monitor RSS with `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` before and after loading. If RSS increases by more than 2GB, the loading is inefficient.

### Pitfall 8: Peak Normalization Hides Relative Posterior Widths

**What goes wrong:** Both `bayesian_plots.py` and `paper_figures.py` default to peak normalization (`posterior / max(posterior)`). When comparing two posteriors of different widths on the same axes, peak normalization makes them appear equally probable at their peaks. This hides the key scientific result: the narrower posterior (with-BH-mass) is more informative, but peak normalization makes both posteriors peak at 1.0.

**Why it happens:** Peak normalization is visually clean and avoids the issue of one posterior being orders of magnitude larger than another. It is standard practice. But for COMPARING posteriors, it removes the information about relative constraining power.

**Consequences:** A reader looking at the peak-normalized plot sees two curves both reaching 1.0 and may underestimate the significance of the width difference. The "area under the curve" meaning of probability is lost.

**Prevention:**
- For the main comparison figure: peak normalization is acceptable if the caption clearly states "peak-normalized for visual comparison" and the CI widths are reported quantitatively in the text.
- Consider an inset or separate panel showing the density-normalized (`mode="density"`) versions, where the narrower posterior is taller.
- At minimum, annotate the 68% CI width directly on the figure: `Delta h = X.XX` for each variant.
- Do NOT density-normalize both posteriors on the same axes if their widths differ by more than 5x -- the broader one becomes invisible.

**Detection:** If two posteriors are peak-normalized and their 68% CI widths differ by more than 3x, the visual comparison is misleading without annotation.

### Pitfall 9: Non-Reproducible Event Selection in Single-Event Likelihood Figure

**What goes wrong:** `_select_representative_events` in `paper_figures.py` selects 4 events based on likelihood width statistics. The selection is deterministic (no randomness), but it depends on the full set of loaded events and their ordering. If the input data changes (e.g., a few events are added or removed from the posterior directory), the "representative" events change, and the figure looks different.

**Why it happens:** The selection algorithm sorts events by width and picks fixed percentile positions (5th, 25th, 50th, 95th). This is deterministic for a fixed input set but fragile to data changes.

**Consequences:** A "before and after" comparison of figures from slightly different data sets shows different events, making it impossible to tell whether the difference is due to the data change or the event selection change.

**Prevention:**
- **Pin the selected event IDs** in the code or a config file once the initial selection is made. E.g., `REPRESENTATIVE_EVENTS = ["42", "117", "283", "491"]`.
- If using algorithmic selection, document the selection criteria in the figure caption and accept that the selection may change with different data.
- The convergence plot already uses a fixed seed (`seed=20260407`), which is good. Apply the same principle to event selection if randomness is involved.

**Detection:** Run the figure generation twice with the same data. If the selected events differ, there is a hidden source of non-determinism (e.g., dict ordering, filesystem ordering of glob results).

## Minor Pitfalls

### Pitfall 10: Missing `constrained_layout` Causes Label Clipping in Multi-Panel Figures

**What goes wrong:** The mplstyle sets `figure.constrained_layout.use: True`, but `paper_figures.py` creates some figures with `get_figure(figsize=(...))` which may not respect constrained layout for complex multi-panel setups (e.g., the 4x2 single-event likelihood grid). Axis labels, especially rotated y-labels like `"Peaked\n(event 42)"`, can be clipped.

**Prevention:** After creating multi-panel figures, call `fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04)` to add padding, or switch to `fig.tight_layout()` as a fallback. Test by exporting at final size and checking all labels are fully visible.

### Pitfall 11: Viridis Colormap on Scatter Plots Fails for Small Point Counts

**What goes wrong:** The SNR vs d_L scatter in `paper_figures.py` uses viridis with `s=8` markers and `alpha=0.6`. With few data points (<50), the color mapping shows few discrete colors, making the colorbar misleading (it implies a continuous range). With many overlapping points (>500), alpha blending makes all points look the same washed-out color.

**Prevention:**
- For <50 points: increase marker size to `s=20` and use `alpha=0.9`.
- For >500 points: use `rasterized=True` to avoid huge PDF file sizes, and consider a 2D histogram (`hist2d`) instead of a scatter plot.
- For the redshift colorbar: ensure the colorbar ticks match the actual data range, not a theoretical range.

### Pitfall 12: EPS Output Incompatible with Transparency

**What goes wrong:** If APS requests EPS format, any figure using alpha transparency (`alpha=0.12` for CI shading, `alpha=0.3` for fill_between, `alpha=0.6` for scatter) will render incorrectly. EPS does not support transparency; matplotlib fakes it by compositing against a white background, but this fails for overlapping transparent elements.

**Prevention:**
- **Prefer PDF** for all APS submissions. APS accepts PDF and it handles transparency correctly.
- If EPS is required: replace alpha-blended fills with solid fills using pre-mixed colors. E.g., instead of `alpha=0.12` blue on white, use `color="#dbe8f4"` (the composited result).
- Test: export to EPS, re-open in a viewer, and check that overlapping transparent regions look correct.

### Pitfall 13: Legend Placement Obscures Data Near the Peak

**What goes wrong:** `paper_figures.py` uses `legend(loc="upper right")`. For the H0 posterior comparison, the posterior peaks are typically near h=0.73 (right-of-center), and the legend box can overlap with the data. With the truth line, Planck band, and SH0ES band all near the peak, the upper-right corner is crowded.

**Prevention:** Use `legend(loc="upper left")` if the posteriors have low probability there, or move the legend outside the axes with `bbox_to_anchor=(1.02, 1)`. For paper figures, prefer annotating curves directly with `ax.text()` or `ax.annotate()` instead of using a legend box.

## Numerical Pitfalls

| Issue | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| `np.trapezoid` not available | `AttributeError` | matplotlib < 3.8 / numpy < 2.0 uses `np.trapz` | Check numpy version; use `np.trapz` as fallback or pin `numpy>=2.0` |
| Zero posterior at all grid points | Division by zero in normalization | All likelihoods below numerical precision | Add `if peak <= 0: return posterior` guard (already present in `_normalize_posterior`) |
| PDF file size > 10MB for scatter plots | Slow rendering, APS upload timeout | Vector rendering of thousands of scatter points | Use `rasterized=True` on scatter and fill operations |
| LaTeX rendering fails on headless CI | `RuntimeError: latex not found` | `text.usetex: True` without TeX installation | Default `use_latex=False`; only enable for final paper figures on a machine with TeX |
| Inconsistent DPI across figure types | Visual size mismatch when figures are placed side-by-side in paper | Different `dpi` settings for different figures | Always use `savefig.dpi: 300` from the mplstyle; never override per-figure |

## Convention and Notation Pitfalls

| Pitfall | Sources That Differ | Resolution |
|---------|-------------------|------------|
| h vs H_0 axis label | Some plots label x-axis as "$h$", others could use "$H_0$" | Use dimensionless $h$ consistently. State $H_0 = 100\,h$ km/s/Mpc in figure caption. Current code is consistent: `LABELS["h"]` returns `$h$`. |
| Peak-normalized vs density-normalized y-axis | `bayesian_plots.py` defaults to peak; paper figures also default to peak | State normalization in y-axis label. "Posterior (peak-normalized)" is explicit and correct (as in `paper_figures.py` line 231). |
| "Injected" vs "True" for reference line label | `paper_figures.py` uses "Injected"; `bayesian_plots.py` uses "True $h = 0.73$" | Use "Injected" in paper figures (physically precise: we injected this value into the simulation). Use "True" only in internal diagnostics. |
| Planck vs SH0ES reference values | `bayesian_plots.py` hardcodes Planck h=0.674 +/- 0.005, SH0ES h=0.73 +/- 0.01 | These are Planck 2018 (arXiv:1807.06209) and Riess et al. 2022 (arXiv:2112.04510) values. Verify against latest values before submission. |

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Replacing CYCLE colors | Breaking existing non-paper plots that depend on specific CYCLE indices | Use semantic named colors (TRUTH, MEAN, etc.) instead of CYCLE indices wherever possible. Only CYCLE-indexed code needs updating. |
| Adding linestyle cycling | Too many line styles become confusing (>4 distinct styles) | Limit to 4 standard styles: solid, dashed, dotted, dashdot. Use markers for additional distinction. |
| Testing colorblind safety | Daltonize may not be installed; adds a dev dependency | Add `daltonize` to the `dev` extras in `pyproject.toml`. Make the test a manual check, not a CI gate. |
| Merging two figure pipelines | Style inconsistency between `bayesian_plots.py` and `paper_figures.py` | Both already import from `_colors.py` and `_helpers.py`. Ensure the new palette is the single source of truth in `_colors.py`. |
| Memory during figure generation | Loading 15 with-mass JSON files (~580MB each) simultaneously | The tail-read approach in `paper_figures.py` is already correct. Do not regress by switching to full `json.load`. |
| CI precision reporting | Quoting CI to 3 decimal places from 15-point grid | Round CI boundaries to nearest grid point (0.02 spacing) or state interpolation method. |
| savefig format | Using PNG for paper submission | Always export PDF for vector quality. Use PNG only for rasterized supplementary material or web display. |

## Sources

- Wong, B. (2011). "Points of view: Color blindness." Nature Methods, 8(6), 441. doi:10.1038/nmeth.1618
- [Okabe-Ito palette complete reference](https://conceptviz.app/blog/okabe-ito-palette-hex-codes-complete-reference)
- [APS Style Basics](https://journals.aps.org/authors/style-basics)
- [APS Author Guide for REVTeX 4.2](https://ctan.math.illinois.edu/macros/latex/contrib/revtex/aps/apsguide4-2.pdf)
- [APS Journals Style Guide for Authors (November 2024)](https://res.cloudinary.com/apsphysics/image/upload/v1736779890/APS_Journals_Style_Guide_Authors_Nov2024_ua91lv.pdf)
- [matplotlib: Fonts in Matplotlib](https://matplotlib.org/stable/users/explain/text/fonts.html)
- [daltonize: CVD simulation for matplotlib](https://github.com/joergdietrich/daltonize)
- [colorspacious: color vision deficiency simulation](https://pypi.org/project/colorspacious/)
- [Ranocha: Coloring in Scientific Publications](https://ranocha.de/blog/colors/)
- Rougier, N.P. et al. (2014). "Ten Simple Rules for Better Figures." PLoS Comput Biol 10(9): e1003833. doi:10.1371/journal.pcbi.1003833
