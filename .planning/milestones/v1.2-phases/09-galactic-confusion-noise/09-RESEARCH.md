# Phase 9: Galactic Confusion Noise - Research

**Researched:** 2026-03-29
**Domain:** LISA noise model -- galactic binary confusion foreground
**Confidence:** HIGH

## Summary

The galactic confusion noise is an astrophysical noise foreground from unresolved white dwarf binaries in the Milky Way. It dominates the LISA noise budget in the 0.1--3 mHz band and must be included in the PSD for physically correct EMRI SNR estimates. The confusion noise decreases with observation time as more individual binaries are resolved and subtracted.

The codebase already defines all seven confusion noise coefficients in `constants.py:74-81` (amplitude A, spectral slope alpha, knee frequency f2, and four power-law fit coefficients a1, b1, ak, bk). These constants use the **LISA Data Challenge (LDC) parameterization**, which models the confusion noise with continuous observation-time dependence via power-law fits for transition and knee frequencies. The implementation requires adding a single method to `LisaTdiConfiguration` and calling it from `power_spectral_density_a_channel()`.

**Primary recommendation:** Add `S_c(f)` directly to the return value of `power_spectral_density_a_channel()`, following the LDC formula with the constants already defined in `constants.py`. Add `t_obs_years: float = 4.0` and `include_confusion_noise: bool = True` fields to the dataclass per CONTEXT.md decisions D-01 and D-04.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** T_obs (observation time for confusion noise model) is a field on `LisaTdiConfiguration` dataclass with default `t_obs_years: float = 4.0` (LISA nominal mission duration).
- **D-02:** T_obs for confusion noise is independent of waveform duration (controlled by `LISA_STEPS * LISA_DT`). The confusion noise level reflects cumulative foreground subtraction over the full mission, not the analysis window for a single source.
- **D-03:** Confusion noise is added only to the A/E-channel PSD, not the T-channel.
- **D-03a (Assumption):** The T-channel is a null channel for GW signals at long wavelengths and is used primarily for noise characterization. Omitting confusion noise from the T-channel has negligible impact on EMRI SNR calculations. This assumption should be revisited if T-channel data is used for signal extraction in future work.
- **D-04:** Add `include_confusion_noise: bool = True` field to `LisaTdiConfiguration` dataclass. Default is always-on for production. Tests can pass `False` to get the old (instrumental-only) PSD for comparison or regression testing.

### Claude's Discretion
- Implementation details of how S_c(f) is computed (helper method vs inline) are left to Claude
- Whether to refactor the existing S_OMS/S_TM static methods while touching the file is left to Claude's judgment (only if it simplifies the confusion noise addition)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PHYS-02 | LISA PSD includes galactic confusion noise term (Babak et al. 2023) | Formula verified from LDC source code and Cornish & Robson (2017/2019). Constants already in `constants.py:74-81`. Implementation adds S_c(f) to `power_spectral_density_a_channel()`. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Physics Change Protocol required:** `LISA_configuration.py` is a trigger file. Before writing code, must present old formula, new formula, reference, dimensional analysis, and limiting case. Wait for user approval.
- **`/physics-change` skill is mandatory gate** before any code modification.
- **Git commit prefix:** `[PHYSICS]` for the commit that modifies the PSD formula.
- **Reference comment:** Must add `# Eq. (X) in Author et al. (YYYY), arXiv:XXXX.XXXXX` above the confusion noise implementation.
- **GPU/CPU portability:** Must use `xp = _get_xp(frequencies)` pattern. No unconditional `cupy` imports.
- **Dataclass convention:** New fields with immutable defaults use `field(default=...)`. Since `float` and `bool` are immutable, bare defaults are fine (no `field(default_factory=...)` needed).
- **Type annotations:** All new methods need full type annotations. Use `npt.NDArray[np.float64]` for array types.
- **Pre-commit hooks:** ruff + mypy run on commit. Code must pass both.
- **Tests:** Must be runnable on CPU-only dev machine. GPU tests use `@pytest.mark.gpu`.

## Standard Stack

No new libraries needed. This phase modifies only existing code with existing dependencies.

### Core (already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (existing) | Array computation for PSD formula | Already used in `LISA_configuration.py` |

### Supporting (already installed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| cupy | (existing, optional) | GPU-portable array ops via `_get_xp()` | When `use_gpu=True` on cluster |

**No new dependencies to install.**

## Architecture Patterns

### Files to Modify

```
master_thesis_code/
├── LISA_configuration.py          # PRIMARY: add S_c method + modify PSD_A
└── constants.py                   # NO CHANGES (constants already defined)

master_thesis_code_test/
└── LISA_configuration_test.py     # Add confusion noise tests
```

### Pattern 1: Adding a Noise Component Method

The existing codebase uses static methods for noise components (`S_OMS`, `S_TM`). The confusion noise method should follow the same pattern but as an **instance method** (not static) because it depends on instance fields (`t_obs_years`, `include_confusion_noise`).

**What:** Add `_confusion_noise()` instance method that computes S_c(f) using the LDC formula.
**When to use:** Called from `power_spectral_density_a_channel()` when `self.include_confusion_noise` is True.
**Example:**
```python
# Eq. (14) in Robson, Cornish & Liu (2019), arXiv:1803.01944
# Using LDC parameterization with continuous T_obs dependence
def _confusion_noise(
    self, frequencies: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    xp = _get_xp(frequencies)
    t_obs_sec = self.t_obs_years * YEAR_IN_SEC
    # Transition frequency: f1 = 10^(a1 * log10(T_obs) + b1)
    f1 = 10.0 ** (LISA_PSD_A1 * xp.log10(t_obs_sec) + LISA_PSD_B1)
    # Knee frequency: fk = 10^(ak * log10(T_obs) + bk)
    fk = 10.0 ** (LISA_PSD_AK * xp.log10(t_obs_sec) + LISA_PSD_BK)
    return (  # type: ignore[no-any-return]
        LISA_PSD_A
        * frequencies ** (-7.0 / 3.0)
        * xp.exp(-(frequencies / f1) ** LISA_PSD_ALPHA)
        * 0.5
        * (1.0 + xp.tanh(-(frequencies - fk) / LISA_PSD_F2))
    )
```

### Pattern 2: Dataclass Field Addition

**What:** Add two new fields to `LisaTdiConfiguration`.
**When to use:** Follows existing dataclass conventions.
**Example:**
```python
@dataclass
class LisaTdiConfiguration:
    t_obs_years: float = 4.0
    include_confusion_noise: bool = True
```

### Anti-Patterns to Avoid
- **Adding S_c to T-channel:** Per D-03, confusion noise is A/E only. Do not modify `power_spectral_density_t_channel()`.
- **Using Cornish/Robson discrete parameter sets:** The constants in `constants.py` use the LDC continuous parameterization, not the discrete 4-year Table 1 values from Robson et al.
- **Coupling T_obs to waveform duration:** Per D-02, these are independent. The waveform duration is `LISA_STEPS * LISA_DT`; the confusion noise T_obs reflects mission-level foreground subtraction.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Confusion noise formula | Custom derivation | LDC parameterization with existing constants | Constants already vetted and defined in `constants.py:74-81` |
| Observation time unit conversion | Manual calculation | `YEAR_IN_SEC` constant already in `constants.py:70` | Avoids magic numbers |

## Common Pitfalls

### Pitfall 1: Wrong Citation
**What goes wrong:** The `constants.py:74` comment cites "arXiv:2303.15929 Eq. 17" but that paper (Hartwig, Lilley, Muratore & Pieroni 2023) is about SGWB reconstruction, not confusion noise. Its Eq. 17 is an SNR expression, not S_c(f).
**Why it happens:** The confusion noise model is widely used across LISA papers and the citation was likely a copy error.
**How to avoid:** The correct primary references are: (1) Cornish & Robson (2017), arXiv:1703.09858, Eq. (3) and Table 1 for the functional form; (2) Robson, Cornish & Liu (2019), arXiv:1803.01944, Eq. (14) for the same formula in the sensitivity curve context; (3) LISA Data Challenge toolbox for the LDC parameterization with continuous T_obs dependence. Since the user locked "Babak et al. (2023) arXiv:2303.15929 Eq. 17" as the reference, flag this discrepancy during physics-change protocol and let the user decide which citation to use.
**Warning signs:** If Eq. 17 from 2303.15929 doesn't look like a confusion noise formula, it's because it isn't.

### Pitfall 2: PSD Representation Mismatch
**What goes wrong:** Adding strain-level S_c(f) to TDI-level P_A(f) without understanding the representations produces wrong SNR values.
**Why it happens:** The existing `power_spectral_density_a_channel()` returns the TDI A-channel noise PSD with explicit `sin^2(2*pi*f*L/c)` transfer function terms. The confusion noise S_c(f) from Cornish/Robson is defined at the strain PSD level.
**How to avoid:** At the equal-arm-length approximation used by the codebase, adding S_c(f) directly to the A-channel PSD is the standard practice used by the LDC and multiple EMRI analysis codes. The confusion noise is isotropic and sky-averaged, so it enters each TDI channel uniformly. This is physically justified because the galactic foreground residual after source subtraction is effectively a diffuse background. The Robson/Cornish formulation (Eq. 1) adds S_c directly to S_n(f), and many EMRI papers follow this convention.
**Warning signs:** If S_c(1 mHz) is orders of magnitude different from the instrumental PSD at the same frequency, check unit consistency.

### Pitfall 3: Negative or Zero Frequencies
**What goes wrong:** `f^(-7/3)` and `log10(f)` blow up at f=0.
**Why it happens:** The PSD is called on frequency arrays that may include f=0 from `rfftfreq`.
**How to avoid:** The existing code already handles this -- the PSD is evaluated only on the cropped frequency range `[lower_idx:upper_idx]` which excludes DC. The confusion noise method should still guard against zero/negative frequencies defensively (or document that it expects positive frequencies only).

### Pitfall 4: T_obs Units
**What goes wrong:** Passing T_obs in years to `log10()` when the LDC formula expects seconds.
**Why it happens:** The dataclass field is in years (user-friendly), but the power-law coefficients a1, b1, ak, bk were fitted with T_obs in seconds.
**How to avoid:** Convert explicitly: `t_obs_sec = self.t_obs_years * YEAR_IN_SEC` before computing f1 and fk.

### Pitfall 5: Breaking Existing Test Expectations
**What goes wrong:** Enabling confusion noise by default changes the PSD values that existing tests validate.
**Why it happens:** D-04 sets `include_confusion_noise: bool = True` as default.
**How to avoid:** Existing tests use `LisaTdiConfiguration()` which will now include confusion noise. Since the existing tests only check `psd > 0` and `psd_A == psd_E`, they should pass with confusion noise enabled (PSD only gets larger). But check carefully -- `test_lisa_config_does_not_go_stale_after_randomize` doesn't use PSD at all, so it's fine.

## Code Examples

### Complete Implementation Pattern

Verified from LDC source code (`ldc.lisa.noise.noise.GalNoise.galshape`):

```python
# In LISA_configuration.py

from master_thesis_code.constants import (
    LISA_ARM_LENGTH as L,
    C,
    YEAR_IN_SEC,
    LISA_PSD_A,
    LISA_PSD_ALPHA,
    LISA_PSD_F2,
    LISA_PSD_A1,
    LISA_PSD_B1,
    LISA_PSD_AK,
    LISA_PSD_BK,
)

@dataclass
class LisaTdiConfiguration:
    t_obs_years: float = 4.0
    include_confusion_noise: bool = True

    # Eq. (14) in Robson, Cornish & Liu (2019), arXiv:1803.01944
    # LDC parameterization with continuous T_obs dependence
    def _confusion_noise(
        self, frequencies: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        xp = _get_xp(frequencies)
        t_obs_sec = self.t_obs_years * YEAR_IN_SEC
        f1 = 10.0 ** (LISA_PSD_A1 * xp.log10(t_obs_sec) + LISA_PSD_B1)
        fk = 10.0 ** (LISA_PSD_AK * xp.log10(t_obs_sec) + LISA_PSD_BK)
        return (  # type: ignore[no-any-return]
            LISA_PSD_A
            * frequencies ** (-7.0 / 3.0)
            * xp.exp(-(frequencies / f1) ** LISA_PSD_ALPHA)
            * 0.5
            * (1.0 + xp.tanh(-(frequencies - fk) / LISA_PSD_F2))
        )

    def power_spectral_density_a_channel(
        self, frequencies: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        xp = _get_xp(frequencies)
        instrumental = (
            8
            * xp.sin(2 * xp.pi * frequencies * L / C) ** 2
            * (
                self.S_OMS(frequencies) * (xp.cos(2 * xp.pi * frequencies * L / C) + 2)
                + 2
                * (3 + 2 * xp.cos(2 * xp.pi * frequencies * L / C)
                   + xp.cos(4 * xp.pi * frequencies * L / C))
                * self.S_TM(frequencies)
            )
        )
        if self.include_confusion_noise:
            instrumental = instrumental + self._confusion_noise(frequencies)
        return instrumental  # type: ignore[no-any-return]
```

### Test Pattern

```python
def test_confusion_noise_increases_psd_at_1mhz() -> None:
    """PSD at 1 mHz with confusion noise must exceed PSD without."""
    config_with = LisaTdiConfiguration(include_confusion_noise=True)
    config_without = LisaTdiConfiguration(include_confusion_noise=False)
    fs = np.array([1e-3])  # 1 mHz
    psd_with = config_with.power_spectral_density_a_channel(fs)
    psd_without = config_without.power_spectral_density_a_channel(fs)
    assert psd_with[0] > psd_without[0]

def test_confusion_noise_negligible_above_10mhz() -> None:
    """Above 10 mHz, confusion noise should be negligible."""
    config_with = LisaTdiConfiguration(include_confusion_noise=True)
    config_without = LisaTdiConfiguration(include_confusion_noise=False)
    fs = np.array([0.01])  # 10 mHz
    ratio = config_with.power_spectral_density_a_channel(fs)[0] / \
            config_without.power_spectral_density_a_channel(fs)[0]
    assert ratio < 1.01  # less than 1% difference

def test_confusion_noise_toggle_backward_compat() -> None:
    """include_confusion_noise=False gives identical results to old code."""
    config = LisaTdiConfiguration(include_confusion_noise=False)
    fs = np.logspace(-4, 0, 100)
    psd = config.power_spectral_density(fs, channel="A")
    assert np.all(psd > 0)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Instrumental noise only | Instrumental + galactic confusion foreground | Standard since Cornish & Robson 2017 | More realistic SNR estimates; lower detection yield in 0.1-3 mHz |
| Discrete parameter sets per T_obs | Continuous power-law T_obs dependence (LDC) | LDC toolbox ~2020+ | Allows arbitrary observation times without interpolation |
| Cornish/Robson 5-parameter fit (alpha, beta, kappa, gamma, fk) | LDC 3-parameter shape (A, alpha, f2) + 2 power-law fits (f1, fk from T_obs) | LDC standardization | Simpler formula; same physical content |

**Important note on formula discrepancy:**
- **Cornish & Robson (2017) / Robson et al. (2019) Eq. 14**: `S_c = A * f^(-7/3) * exp(-f^alpha + beta*f*sin(kappa*f)) * [1 + tanh(gamma*(fk-f))]` with `A = 9e-45` (two-channel) or `A = 1.8e-44` (single-channel)
- **LDC parameterization** (matching `constants.py`): `S_c = A * f^(-7/3) * exp(-(f/f1)^alpha) * 0.5 * (1 + tanh(-(f-fk)/f2))` with `A = 1.14e-44`, `alpha = 1.8`, `f2 = 0.31e-3`

These are **different functional forms** but describe the same physical phenomenon. The constants in `constants.py` use the LDC parameterization, which is the one to implement.

## Open Questions

1. **Citation discrepancy**
   - What we know: The `constants.py:74` comment cites "arXiv:2303.15929 Eq. 17" but that paper does not contain a confusion noise formula. The actual formula origin is Cornish & Robson (2017) arXiv:1703.09858 (or equivalently Robson et al. 2019 arXiv:1803.01944), with the LDC providing the continuous-T_obs parameterization.
   - What's unclear: Whether the user intentionally wants to cite 2303.15929 or if this was a citation error in the original code.
   - Recommendation: During the physics-change protocol, present this finding. The reference comment above the implementation should cite the actual formula source. Suggest using Cornish & Robson (2017) arXiv:1703.09858 Eq. (3) or Robson et al. (2019) arXiv:1803.01944 Eq. (14), with a note that the LDC continuous parameterization is used.

2. **Constant values not independently verified**
   - What we know: `LISA_PSD_A = 1.14e-44` does not exactly match the LDC source code value of `1.28265531e-44` (for 6-link, SNR=7 configuration). The Robson/Cornish papers use `A = 9e-45` (two-channel) or `1.8e-44` (single-channel).
   - What's unclear: The exact provenance of `1.14e-44`. It may be a different SNR threshold or link configuration.
   - Recommendation: Flag during physics-change protocol. Use the existing constants as-is (they were presumably chosen deliberately for this project's conventions). Document the discrepancy in a comment.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml [tool.pytest.ini_options]` |
| Quick run command | `uv run pytest master_thesis_code_test/LISA_configuration_test.py -m "not gpu" -v` |
| Full suite command | `uv run pytest -m "not gpu and not slow"` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PHYS-02a | PSD_A includes confusion noise term | unit | `uv run pytest master_thesis_code_test/LISA_configuration_test.py::test_confusion_noise_increases_psd_at_1mhz -x` | Wave 0 |
| PHYS-02b | PSD at 1 mHz larger with confusion noise | unit | `uv run pytest master_thesis_code_test/LISA_configuration_test.py::test_confusion_noise_increases_psd_at_1mhz -x` | Wave 0 |
| PHYS-02c | Existing CPU tests pass (no regression) | regression | `uv run pytest master_thesis_code_test/LISA_configuration_test.py -m "not gpu" -v` | Exists (7 tests) |
| PHYS-02d | Reference comment cites source | code review | Manual inspection | N/A |

### Sampling Rate
- **Per task commit:** `uv run pytest master_thesis_code_test/LISA_configuration_test.py -m "not gpu" -v`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow"`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `test_confusion_noise_increases_psd_at_1mhz` -- PSD with confusion noise > PSD without at 1 mHz
- [ ] `test_confusion_noise_negligible_above_10mhz` -- confusion noise effect < 1% at 10 mHz
- [ ] `test_confusion_noise_toggle_backward_compat` -- `include_confusion_noise=False` reproduces old PSD
- [ ] `test_confusion_noise_positive` -- S_c(f) > 0 for all frequencies in LISA band
- [ ] `test_t_obs_affects_confusion_noise` -- different T_obs values give different PSD at 1 mHz

None of these require new framework install -- existing pytest infrastructure covers all needs.

## Sources

### Primary (HIGH confidence)
- Robson, Cornish & Liu (2019), arXiv:1803.01944, Eq. (14) and Table 1 -- galactic confusion noise formula with discrete T_obs parameters
- Cornish & Robson (2017), arXiv:1703.09858, Eq. (3) and Table 1 -- original confusion noise fit for new LISA design
- LISA Data Challenge toolbox source code (`ldc.lisa.noise.noise.GalNoise.galshape`) -- LDC parameterization with continuous T_obs dependence
- eXtremeGravityInstitute/LISA_Sensitivity `LISA_tools.py:get_Sc_est()` -- reference implementation of Cornish/Robson formula

### Secondary (MEDIUM confidence)
- Existing `constants.py:74-81` -- confusion noise coefficients already defined, likely from LDC documentation
- arXiv:2303.15929 (Hartwig et al. 2023) -- cited in `constants.py` but does NOT contain confusion noise formula

### Tertiary (LOW confidence)
- The exact provenance of `LISA_PSD_A = 1.14e-44` is unclear -- does not match LDC source (`1.28e-44`) or Cornish/Robson (`9e-45` or `1.8e-44`)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, existing code patterns well understood
- Architecture: HIGH -- single-file modification with clear integration point
- Pitfalls: HIGH -- formula verified from multiple sources; representation mismatch and citation issues identified
- Constants: MEDIUM -- existing constant values don't exactly match any single source; likely project-specific choice

**Research date:** 2026-03-29
**Valid until:** 2026-04-29 (stable domain; LISA noise models change slowly)
