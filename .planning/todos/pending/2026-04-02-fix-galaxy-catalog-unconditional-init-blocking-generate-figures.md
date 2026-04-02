---
created: 2026-04-02T20:28:03.924Z
title: Fix galaxy catalog unconditional init blocking generate-figures
area: general
files:
  - master_thesis_code/main.py:49
---

## Problem

`GalaxyCatalogueHandler` is initialized unconditionally at the top of `main()` (line 49), before any flag dispatch. This means `--generate_figures` (and other data-only flags like `--combine`) cannot run without a galaxy catalog present, even though they don't use it at all.

Discovered when running `generate_figures` locally against `h_sweep_20260401` campaign data — the call crashed with `FileNotFoundError` for `reduced_galaxy_catalogue.csv` before ever reaching the figure generation code.

Workaround: call `generate_figures()` directly in Python, bypassing `main()`.

## Solution

Guard the galaxy catalog init (and the `cosmological_model` that drives its limits) behind a check that at least one simulation/evaluation flag is set:

```python
needs_catalog = (
    arguments.simulation_steps > 0
    or arguments.evaluate
    or arguments.injection_campaign
    or arguments.snr_analysis
)
if needs_catalog:
    cosmological_model = Model1CrossCheck(rng=rng)
    galaxy_catalog = GalaxyCatalogueHandler(...)
```

Flags that are purely data-processing (`--generate_figures`, `--combine`) should be able to run standalone without any galaxy catalog or cosmological model.
