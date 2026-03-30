---
phase: quick
plan: 260330-twe
title: Re-run h-value sweep evaluation with BH mass Gaussian index fix
status: ready
tasks: 3
---

# Quick Task 260330-twe: Re-run H-Value Sweep Evaluation

## Goal

Run Pipeline B evaluation for all 11 h-values (0.6 to 0.86, step 0.026) using the BH mass Gaussian index fix (commit ab77e70), then generate comparison report.

## Tasks

### Task 1: Set up evaluation directory

**Action:** Create `evaluation/run_v12_bhmass_fix/` with:
- Copy CRB data from `evaluation/run_v12_diagnostic/simulations/` (prepared_cramer_rao_bounds.csv, cramer_rao_bounds.csv)
- Symlink galaxy catalog: `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv` → absolute path to real file
- Create `simulations/posteriors/` and `simulations/posteriors_with_bh_mass/` directories

**Files:**
- evaluation/run_v12_bhmass_fix/ (new directory tree)

**Verify:** `ls evaluation/run_v12_bhmass_fix/simulations/prepared_cramer_rao_bounds.csv` exists
**Done:** Directory structure ready with data and symlinks

### Task 2: Run h-value sweep

**Action:** Run `uv run python -m master_thesis_code evaluation/run_v12_bhmass_fix --evaluate --h_value H` for each of the 11 h-values: 0.6, 0.626, 0.652, 0.678, 0.704, 0.73, 0.756, 0.782, 0.808, 0.834, 0.86

Run sequentially. Each should take ~30-60 seconds with 22 detections on CPU.

**Files:**
- evaluation/run_v12_bhmass_fix/simulations/posteriors/*.json (11 files)
- evaluation/run_v12_bhmass_fix/simulations/posteriors_with_bh_mass/*.json (11 files)

**Verify:** 11 JSON files exist in each posteriors directory
**Done:** All 22 posterior JSON files generated

### Task 3: Generate comparison report

**Action:** Run `uv run python scripts/compare_posterior_bias.py` (may need to modify to compare diagnostic vs bhmass_fix runs). Generate comparison report showing:
- Peak h-value for "without BH mass" and "with BH mass" paths
- Whether "with BH mass" path improved from h=0.600 to closer to h=0.73
- Comparison table and ASCII visualization

**Files:**
- evaluation/run_v12_bhmass_fix/comparison_report.md (new)

**Verify:** Report shows peak analysis for both paths
**Done:** Report generated with bias comparison
