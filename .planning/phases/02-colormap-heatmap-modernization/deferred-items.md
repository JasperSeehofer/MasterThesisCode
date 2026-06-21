# Deferred Items — Phase 02 (Colormap & Heatmap Modernization)

Out-of-scope discoveries logged during execution. NOT caused by this phase's
colormap/norm changes — pre-existing data-gated failures in the
`simulations/_archive_v2_1_baseline` render-verify dir. Do NOT fix here.

| Item | Where | Failure | Cause (pre-existing) |
|------|-------|---------|----------------------|
| `fig09_detection_efficiency` | archive baseline render | `ValueError: n must be positive` | empty/degenerate detection-efficiency data in this archive |
| `paper_single_event` | archive baseline render | `IndexError: list index out of range` | missing per-event detail data in this archive |
| `paper_convergence` | archive baseline render | `ValueError: Data has no positive values, and therefore cannot be log-scaled` | convergence series has no positive values in this archive |
| `fig16_catalog_completeness` | archive baseline render | skipped (required data not found) | catalog-completeness data absent |
| `fig20_pdet_surface` | archive baseline render | skipped (no injection CSVs in this dir) | render-verified directly against `simulations/archive/injections_partial_mar31_262files/*.csv` in Task 3 instead |

All five are independent of the cividis migration / explicit-norm work; the
recolored figures (fig01..fig08, fig10.., CRB heatmaps, sky map) all generated
without traceback.

## Pre-existing ruff errors in scripts/ (surfaced by `pre-commit run --all-files`)

`pre-commit run --all-files` (Task 4 final gate) flagged 2 ruff errors in a file
NOT touched by this phase: `scripts/quick_snr_calibration.py` — `F401` (unused
`OMEGA_DE`/`OMEGA_M`/`SNR_THRESHOLD` import, already `# noqa`-suppressed) and
`N818` (`_Timeout` exception should have an `Error` suffix). Pre-existing, in the
`scripts/` tree, unrelated to colormaps. Left untouched (scope boundary). The
phase commits use targeted per-file pre-commit / `uv run` gates that are green on
all Phase-2 files.
