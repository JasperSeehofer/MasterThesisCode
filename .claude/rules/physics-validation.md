---
paths: ["**/physical_relations.py", "**/constants.py", "**/LISA_configuration.py", "**/parameter_estimation/parameter_estimation.py", "**/datamodels/galaxy.py", "**/bayesian_inference/**", "**/cosmological_model.py"]
description: Math/physics validation workflow — the physics-change protocol, before-writing gate, post-implementation checks, and [PHYSICS] commit convention
---

# Math/Physics Validation Workflow

Errors in physics formulas produce subtly wrong results with no crash. A strict protocol applies. This is the detail behind the `/physics-change` hard gate (see the Skill-Driven Workflows table in CLAUDE.md).

See [[scientific-computing-validation]] for the promoted, cross-project form of these patterns.

### What counts as a physics change

A change is a **physics change** if it touches any of:
- A formula (integrals, inner products, distance-redshift relations, posteriors, likelihoods)
- A physical or cosmological constant: `C`, `G`, `H`, `OMEGA_M`, `W_0`, `W_A`, `SNR_THRESHOLD`, `TRUE_HUBBLE_CONSTANT`, PSD coefficients in `LISA_configuration.py`, `derivative_epsilon` in `ParameterSpace`
- Waveform parameters passed to `few` or `ResponseWrapper`
- Frequency limits in `scalar_product_of_functions`
- Galaxy distribution or mass function model

A change is a **software change** if it is limited to: refactoring, type annotations, test additions, logging, or import cleanup — with no change to any computed numerical value. When in doubt, treat it as a physics change.

### Protocol — before writing any code, Claude presents

1. **Old formula** — exact expression, file:line
2. **New formula** — proposed replacement
3. **Reference** — citation (DOI/arXiv + equation number) or step-by-step derivation
4. **Dimensional analysis** — units of inputs and output, consistency check
5. **Limiting case** — at least one analytical limit where the result is known

The user approves or rejects. Claude then implements.

### Post-implementation checks

After implementing an approved change, Claude reports:
- Sign convention consistency
- Dimensional consistency
- A reference comment added directly above the changed line:
  ```python
  # Eq. (X.Y) in Author et al. (YYYY), arXiv:XXXX.XXXXX
  ```

### Git convention for physics changes

Prefix the commit subject line with `[PHYSICS]`:

```
[PHYSICS] fix luminosity distance prefactor in dist()
```
