---
name: run-pipeline
description: >
  Run the EMRI simulation or Bayesian evaluation pipeline with correct flags.
  Use when the user wants to run a simulation, evaluate posteriors, or test
  the full pipeline end-to-end.
disable-model-invocation: true
argument-hint: <sim|eval|snr> <working_dir> [--steps N] [--seed S] [--h_value V]
allowed-tools: Bash(uv run *), Read, Glob
---

## Pipeline Runner

Based on the first argument, run the appropriate pipeline:

### `sim` — EMRI Simulation
```bash
uv run python -m master_thesis_code <working_dir> --simulation_steps <N> [additional flags]
```
After completion:
1. Check exit code
2. Read `run_metadata.json` — report git_commit, seed, timestamp
3. Count rows in any generated CSV files (Cramer-Rao bounds)
4. Report how many events exceeded SNR threshold (grep logs or CSV)

### `eval` — Bayesian Inference
```bash
uv run python -m master_thesis_code <working_dir> --evaluate [--h_value <V>]
```
After completion:
1. Check for posterior JSON output in working_dir
2. Report H0 peak value and credible interval if available

### `snr` — SNR Analysis Only
```bash
uv run python -m master_thesis_code <working_dir> --snr_analysis
```

### Always:
- Warn if `--seed` is not provided (results won't be reproducible)
- Report wall-clock time
- Flag any Python warnings or errors in output
