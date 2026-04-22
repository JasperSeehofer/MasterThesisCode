# Phase 37: Parameter Estimation Correctness - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-22
**Phase:** 37-parameter-estimation-correctness
**Areas discussed:** h-threading (PE-01), Epsilon structure (PE-02), SNR unification scope (PE-04)

---

## h-threading (PE-01)

| Option | Description | Selected |
|--------|-------------|----------|
| Required kwarg | `def set_host_galaxy_parameters(self, host_galaxy, h: float)` — h required, no default. main.py:394 passes h_value. | ✓ |
| Store h on ParameterSpace | Add h: float field to ParameterSpace dataclass; set_host_galaxy_parameters reads self.h. | |

**User's choice:** Required kwarg

**Notes:** User asked for clarification on why `h` is needed here — specifically whether this mixes up simulation and evaluation. Clarified:
- `set_host_galaxy_parameters` is simulation-side only; evaluation reads CRBs from CSV, never calls this function.
- `h` here is the **injected truth h** (h_inj), not the evaluation-side h sweep.
- The injection pipeline at `main.py:650-651` already works around the bug via `dist(sample.redshift, h=h_value)` with an explanatory comment — that workaround comment should be removed once PE-01 lands.
- For the standard simulation (h_inj=0.73), the bug was harmless because the hardcoded default matched the injected truth. Future injection campaigns at other h values would be incorrect without this fix.

---

## Epsilon structure (PE-02)

| Option | Description | Selected |
|--------|-------------|----------|
| Dict of absolute values | `DERIVATIVE_EPSILONS: dict[str, float]` keyed by param name in constants.py or ParameterSpace | |
| Per-Parameter field (emerged) | Set per-param `derivative_epsilon` values in each Parameter's default_factory lambda in ParameterSpace | ✓ |
| Relative scaling | `eps_i = max(floor, eps_rel × \|param_i\|)` at runtime | |

**User's choice:** Per-Parameter field on the existing `Parameter.derivative_epsilon` field

**Notes:** User pointed out that `derivative_epsilon` is already a field on the `Parameter` dataclass (not just `ParameterSpace`). Verification confirmed `parameter_estimation.py:163` already reads `derivative_epsilon = parameter.derivative_epsilon` per-instance. No structural change needed — just set appropriate non-default values in the `default_factory` lambdas inside `ParameterSpace`. The GPD executor will determine specific values (M, d_L particularly need larger epsilons) under the `/physics-change` protocol citing Vallisneri 2008.

---

## SNR unification scope (PE-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Fix the 15s + name the 0.3 | Update cosmological_model.py and paper_figures.py to SNR_THRESHOLD; add PRE_SCREEN_SNR_FACTOR = 0.3 | ✓ |
| Just fix the 15s | Update the literal values to 20.0; leave 0.3 unnamed | |

**User's choice:** Fix the 15s + name the 0.3

**Notes:** Four sites updated: constants.py (add PRE_SCREEN_SNR_FACTOR), cosmological_model.py:171 (int=15 → float=SNR_THRESHOLD), paper_figures.py:553 (15.0 → SNR_THRESHOLD), main.py:421 (0.3 → PRE_SCREEN_SNR_FACTOR). bayesian_plots.py:370 has 20.0 which already matches — left as Claude's discretion.

---

## Claude's Discretion

- Whether `bayesian_plots.py:370` imports `SNR_THRESHOLD` or keeps the matching `20.0` literal
- Exact field name for the COORD-05 idempotency flag on `GalaxyCatalogueHandler`
- Whether epsilon regression test goes in a new file or alongside existing tests
- Commit ordering for the hygiene fixes vs. physics fixes

## Deferred Ideas

- Automated per-run Vallisneri optimal-step formula — out of Phase 37 scope
- Re-evaluation of CRBs under new h-threading — Phase 40
