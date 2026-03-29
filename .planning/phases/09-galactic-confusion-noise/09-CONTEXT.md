# Phase 9: Galactic Confusion Noise - Context

**Gathered:** 2026-03-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Add the galactic foreground confusion noise term S_c(f) to the LISA A/E-channel PSD, following Babak et al. (2023) Eq. 17. The confusion noise constants are already defined in `constants.py:74-81` but are not yet used in `LISA_configuration.py`. This phase makes the PSD physically complete in the 0.1-3 mHz band where galactic binaries dominate.

</domain>

<decisions>
## Implementation Decisions

### Observation Time Parameter
- **D-01:** T_obs (observation time for confusion noise model) is a field on `LisaTdiConfiguration` dataclass with default `t_obs_years: float = 4.0` (LISA nominal mission duration).
- **D-02:** T_obs for confusion noise is independent of waveform duration (controlled by `LISA_STEPS * LISA_DT`). The confusion noise level reflects cumulative foreground subtraction over the full mission, not the analysis window for a single source.

### T-Channel PSD
- **D-03:** Confusion noise is added only to the A/E-channel PSD, not the T-channel.
- **D-03a (Assumption):** The T-channel is a null channel for GW signals at long wavelengths and is used primarily for noise characterization. Omitting confusion noise from the T-channel has negligible impact on EMRI SNR calculations. This assumption should be revisited if T-channel data is used for signal extraction in future work.

### Toggle / Backward Compatibility
- **D-04:** Add `include_confusion_noise: bool = True` field to `LisaTdiConfiguration` dataclass. Default is always-on for production. Tests can pass `False` to get the old (instrumental-only) PSD for comparison or regression testing.

### Claude's Discretion
- Implementation details of how S_c(f) is computed (helper method vs inline) are left to Claude
- Whether to refactor the existing S_OMS/S_TM static methods while touching the file is left to Claude's judgment (only if it simplifies the confusion noise addition)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Physics Reference
- Babak et al. (2023) arXiv:2303.15929, Eq. 17 -- defines the galactic confusion noise PSD model S_c(f) with time-dependent coefficients. This is THE reference for this phase.

### Codebase Files
- `master_thesis_code/LISA_configuration.py` -- target file; contains `LisaTdiConfiguration` dataclass with PSD methods
- `master_thesis_code/constants.py:74-81` -- already defines `LISA_PSD_A`, `LISA_PSD_ALPHA`, `LISA_PSD_F2`, `LISA_PSD_A1`, `LISA_PSD_B1`, `LISA_PSD_AK`, `LISA_PSD_BK`
- `master_thesis_code_test/LISA_configuration_test.py` -- existing PSD tests that must continue to pass

### Requirements
- `.planning/REQUIREMENTS.md` -- PHYS-02: LISA PSD includes galactic confusion noise term

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `constants.py:74-81`: All seven confusion noise coefficients from Babak et al. (2023) Eq. 17 are already defined and importable
- `_get_xp()` helper in `LISA_configuration.py`: GPU/CPU-portable array namespace dispatch, already used by all PSD methods
- `S_OMS()` and `S_TM()` static methods: pattern for noise component methods that the confusion noise can follow

### Established Patterns
- PSD methods use the `xp = _get_xp(frequencies)` pattern for GPU portability
- A/E channels share `power_spectral_density_a_channel()`; T-channel has its own method
- `LisaTdiConfiguration` is a `@dataclass` -- new fields use `field(default=...)` for immutable defaults

### Integration Points
- `power_spectral_density_a_channel()` is called by `power_spectral_density()` for both A and E channels
- `ParameterEstimation.compute_signal_to_noise_ratio()` calls the PSD through `self.lisa_configuration`
- No changes needed outside `LISA_configuration.py` and its test file (plus imports of new constants)

</code_context>

<specifics>
## Specific Ideas

No specific requirements -- open to standard approaches following Babak et al. (2023) Eq. 17 directly.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 09-galactic-confusion-noise*
*Context gathered: 2026-03-29*
