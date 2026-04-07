---
phase: quick
plan: 260407-va0
subsystem: plotting / CI
tags: [plotly, interactive, github-pages, visualization, cli]
dependency_graph:
  requires: []
  provides: [interactive Plotly figures, --generate_interactive CLI flag, CI pages deployment]
  affects: [master_thesis_code/plotting, master_thesis_code/main.py, master_thesis_code/arguments.py, .github/workflows/ci.yml]
tech_stack:
  added: [plotly>=6.6.0 (dev extra)]
  patterns: [Plotly graph_objects factory functions, go.Scattergeo Mollweide projection, make_subplots two-panel layout, parametric ellipse from eigendecomposition]
key_files:
  created:
    - master_thesis_code/plotting/interactive.py
    - master_thesis_code_test/test_interactive.py
    - interactive/index.html
  modified:
    - master_thesis_code/arguments.py
    - master_thesis_code/main.py
    - .github/workflows/ci.yml
    - pyproject.toml
    - uv.lock
decisions:
  - "plotly added as dev extra (not core) -- not needed at simulation/inference runtime"
  - "generate_interactive_figures(data_dir) appends /interactive/ subdirectory for output, mirroring generate_figures() pattern"
  - "CI pages job gracefully skips figure generation when cluster_results/ absent (continue-on-error: true)"
  - "Ellipse parametric curve uses eigendecomposition of 2x2 cov sub-matrix directly -- _ellipse_params() import removed as unneeded"
metrics:
  duration: "~25 minutes"
  completed: "2026-04-07T20:44:26Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 3
  files_modified: 5
---

# Quick Task 260407-va0: Add Interactive Plotly Figures to GitHub Pages Summary

**One-liner:** 4 Plotly factory functions (posterior, sky map, Fisher ellipses, convergence) with CDN-hosted HTML output, `--generate_interactive` CLI flag, and CI pages deployment.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Add plotly dep and interactive.py with 4 Plotly figures | 8b47b5f | interactive.py, test_interactive.py, pyproject.toml, uv.lock |
| 2 | Wire CLI flag, update CI, create landing page | 33e1c86 | arguments.py, main.py, ci.yml, interactive/index.html |

## What Was Built

### interactive.py

Four Plotly factory functions, each returning `go.Figure`:

- `interactive_combined_posterior(h_values, posterior, true_h, ...)` — filled posterior trace, 68%/95% CI shading via `add_vrect`, Planck/SH0ES reference bands, truth vline, hover
- `interactive_sky_map(theta_s, phi_s, snr, ...)` — `go.Scattergeo` with Mollweide projection, viridis SNR colorscale, optional redshift/distance hover columns
- `interactive_fisher_ellipses(events, pairs, ...)` — parametric ellipses from eigendecomposition of 2x2 cov sub-matrices per pair, `make_subplots` columns, CYCLE colors per event
- `interactive_h0_convergence(h_values, event_posteriors, ...)` — two-panel `make_subplots`, log-sum-exp combined posteriors per subset size, CI width vs N with 1/sqrt(N) reference

Supporting helpers:
- `_strip_latex(label)` — strips `$`, `\mathrm{}`, `\bullet`, `\odot`, `\,` etc. for plain-text axis labels
- `_credible_interval_bounds()` / `_credible_interval_width()` — CDF-based CI extraction
- `generate_all_interactive(output_dir, data_dir)` — loads CRB CSVs and posterior JSONs, calls all 4 functions, writes HTML with `include_plotlyjs="cdn"`, gracefully skips on missing data

### CLI integration

`--generate_interactive <data_dir>` added to `arguments.py` and `main.py`. Fast-path block in `main()` calls `generate_interactive_figures(data_dir)` which appends `/interactive/` subdirectory for output.

### CI pages job

Restructured to checkout repo, install uv + deps, then: download docs and test-plot artifacts, copy `interactive/index.html`, optionally run `--generate_interactive` on `cluster_results/` (continue-on-error), upload and deploy.

### Landing page

`interactive/index.html` — clean serif HTML5 page, max-width 800px, links to all 4 figure files with descriptions. Note at bottom explains interaction model.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed invalid rgba color string for posterior fill**
- **Found during:** Task 1 verification (pytest)
- **Issue:** `CYCLE[0].replace("#", "rgba(") + "1a)"` produced `"rgba(E69F001a)"` — invalid Plotly color
- **Fix:** Manually parse hex to decimal RGB components and format as `f"rgba({r},{g},{b},0.15)"`
- **Files modified:** master_thesis_code/plotting/interactive.py
- **Commit:** 8b47b5f (included in same commit)

**2. [Rule 1 - Bug] Removed unused `_ellipse_params` import flagged by ruff**
- **Found during:** Task 1 ruff check
- **Issue:** Plan said "import `_ellipse_params` from fisher_plots for eigendecomposition" but the implementation performs eigendecomposition directly via `np.linalg.eigh` — import was unused
- **Fix:** Removed the import; kept direct eigendecomposition which is cleaner
- **Files modified:** master_thesis_code/plotting/interactive.py
- **Commit:** 8b47b5f (ruff --fix applied)

## Known Stubs

None — `generate_all_interactive` loads real data when present and skips gracefully when absent. No placeholder data is rendered.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundary changes. Interactive HTML files contain only simulation output data (accepted in threat model T-quick-02).

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| master_thesis_code/plotting/interactive.py exists | FOUND |
| master_thesis_code_test/test_interactive.py exists | FOUND |
| interactive/index.html exists | FOUND |
| Commit 8b47b5f exists | FOUND |
| Commit 33e1c86 exists | FOUND |
