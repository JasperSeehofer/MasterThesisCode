---
quick_task: 6
id: pe-02
title: "Per-parameter derivative_epsilon for 14 EMRI parameters in ParameterSpace"
status: completed
tasks_completed: 2
tasks_total: 2
commits:
  - hash: 7429c6e
    message: "[PHYSICS] PE-02: per-parameter derivative_epsilon for all 14 EMRI parameters"
  - hash: 16ce20f
    message: "[PHYSICS] PE-02: SC-3 regression tests for per-parameter derivative_epsilon"
files_written:
  - master_thesis_code/datamodels/parameter_space.py
  - master_thesis_code_test/test_parameter_space_h.py
---

## What Was Changed

All 14 `Parameter` factory lambdas in `ParameterSpace` (in
`master_thesis_code/datamodels/parameter_space.py`) were updated to carry an explicit
`derivative_epsilon` keyword argument. Previously every parameter used the class-level
default of `1e-6`, a uniform value that is orders of magnitude too small for large-scale
parameters (`M` ~ 10³–10⁶ solar masses, `d_L` ~ 1 Gpc) and marginally appropriate only
for unit-bounded dimensionless parameters.

### New epsilon values

| Parameter | Symbol | Unit | New epsilon | Old (uniform) | Factor change |
|-----------|--------|------|-------------|---------------|---------------|
| MBH mass | M | solar masses | 1.0 | 1e-6 | 10⁶× |
| CO mass | mu | solar masses | 0.01 | 1e-6 | 10⁴× |
| MBH spin | a | dimensionless | 1e-3 | 1e-6 | 10³× |
| Semi-latus rectum | p0 | dimensionless | 1e-3 | 1e-6 | 10³× |
| Eccentricity | e0 | dimensionless | 1e-4 | 1e-6 | 100× |
| Inclination cosine | x0 | dimensionless | 1e-4 | 1e-6 | 100× |
| Luminosity distance | luminosity_distance | Gpc | 1e-4 | 1e-6 | 100× |
| Sky polar angle | qS | radian | 1e-4 | 1e-6 | 100× |
| Sky azimuthal angle | phiS | radian | 1e-4 | 1e-6 | 100× |
| Spin polar angle | qK | radian | 1e-4 | 1e-6 | 100× |
| Spin azimuthal angle | phiK | radian | 1e-4 | 1e-6 | 100× |
| Azimuthal phase | Phi_phi0 | radian | 1e-4 | 1e-6 | 100× |
| Polar phase | Phi_theta0 | radian | 1e-4 | 1e-6 | 100× |
| Radial phase | Phi_r0 | radian | 1e-4 | 1e-6 | 100× |

The class-level default `derivative_epsilon: float = 1e-6` on `Parameter` is unchanged —
it remains the fallback for any `Parameter` constructed outside `ParameterSpace`.

## Physics Justification

**Reference:** Vallisneri (2008) arXiv:gr-qc/0703086 Eq. (A11)

For a 5-point central-difference stencil of order p=4, the optimal step size that
balances round-off error against truncation error is:

    h* ≈ ε_machine^(1/p) × |x|  =  (2.2×10⁻¹⁶)^(1/4) × |x|  ≈  3.3×10⁻⁴ × |x|

where `|x|` is the representative magnitude of the parameter. The chosen epsilons
approximate this rule applied to each parameter's scale:

- `M`: log-uniform over [10⁴, 10⁷] SM → geometric mean ≈ 3162 SM → h* ≈ 1.0 SM ✓
- `mu`: uniform over [1, 100] SM → midpoint ≈ 30 SM → h* ≈ 0.01 SM ✓
- `a`, `p0`: dimensionless, O(1) → h* ≈ 1e-3 ✓
- `e0`, `x0`, angles, `d_L`: dimensionless/angular/Gpc at O(0.1–1) → h* ≈ 1e-4 ✓

**Dimensional consistency check:**
- `M` epsilon has units of solar masses — same as the parameter ✓
- `luminosity_distance` epsilon is 1e-4 Gpc = 0.1 Mpc — same units as parameter ✓
- All angle epsilons are in radians — same units ✓

**Bounds safety check** (2 × epsilon << range width, to avoid ParameterOutOfBoundsError
in the 5-point stencil footprint):
- `M`: 2 × 1.0 = 2.0 SM << 9.99×10⁶ SM range ✓
- `mu`: 2 × 0.01 = 0.02 SM << 99 SM range ✓
- angles: 2 × 1e-4 << π range ✓
- All others: similarly safe ✓

## Parameters With Largest Change From Old 1e-6

The parameters that benefited most from this fix (in terms of absolute step-size ratio):

1. **M** (MBH mass): 10⁶× increase — previously at h/|x| = 1e-6/3162 ≈ 3×10⁻¹⁰,
   far into the round-off-dominated regime; now at h/|x| ≈ 3×10⁻⁴ (optimal)
2. **mu** (CO mass): 10⁴× increase — midpoint ~30 SM; old h/|x| ≈ 3×10⁻⁸
3. **a**, **p0**: 10³× increase

For the eight angular/phase parameters and `d_L`, the change is 100× — less dramatic
but still moves from the round-off floor into the optimal regime.

## Test Coverage (SC-3)

Two new tests added to `master_thesis_code_test/test_parameter_space_h.py`:

**`test_derivative_epsilon_per_parameter`** (structural):
- Instantiates `ParameterSpace()` and collects all 14 epsilons
- Asserts: count == 14, no epsilon == 1e-6 (old default), no epsilon == 0
- Asserts: ≥ 4 distinct values (actual: 4 — {1.0, 0.01, 1e-3, 1e-4})

**`test_fisher_determinant_stability`** (bounds):
- For each of the 14 parameters, checks:
  - `epsilon >= 1e-6 × representative_value` (round-off safety)
  - `epsilon <= 0.01 × range_width` (truncation safety / Taylor regime)
- Uses geometric mean as representative value for log-uniform `M`
  (arithmetic midpoint 5×10⁶ SM would reject the correct epsilon=1.0 SM)

Both tests are CPU-only (no `@pytest.mark.gpu`), no waveform generation required.

**Full suite result after change:** 521 passed, 6 skipped, 0 failed
(`uv run pytest -m "not gpu and not slow"`)

## Deviation Log

**Deviation (Rule 4 — missing component in test):** The plan's `test_fisher_determinant_stability`
used arithmetic midpoint for all parameters, but `M` is log-uniform — its representative
scale is the geometric mean (~3162 SM), not the arithmetic mean (~5×10⁶ SM). The test
initially failed for `M` with `1.0 < 5.005`. Fixed inline: test now uses geometric mean
when `upper_limit / lower_limit > 100`, matching the physics reasoning in the plan
frontmatter. No change to `parameter_space.py` was required.

## Contract Coverage

| Item | Status | Evidence |
|------|--------|----------|
| claim-per-param-set | PASS | All 14 lambdas carry explicit epsilon; ref comment present |
| claim-stability | PASS | test_fisher_determinant_stability passes for all 14 |
| deliv-parameter-space | DELIVERED | `master_thesis_code/datamodels/parameter_space.py` |
| deliv-test-epsilon | DELIVERED | `master_thesis_code_test/test_parameter_space_h.py` |
| test-epsilons-nonuniform | PASS | 4 distinct values, none == 1e-6 or 0 |
| test-fisher-stability | PASS | All 14 in valid Vallisneri regime |
| ref-vallisneri (must_surface) | CITED | Comment block + test docstrings + this summary |
| fp-zero-epsilon | CLEAR | No epsilon == 0 |
| fp-future-annotations | CLEAR | Not added |
| fp-bare-ndarray | CLEAR | No ndarray annotations added |
| fp-no-verify | CLEAR | Pre-commit hooks ran and passed |
| fp-param-count | CLEAR | Still exactly 14 parameters |
