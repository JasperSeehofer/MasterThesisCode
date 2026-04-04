# Phase 9: Galactic Confusion Noise - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-29
**Phase:** 09-galactic-confusion-noise
**Areas discussed:** Observation time parameter, T-channel PSD update, Toggle/backward compat

---

## Observation Time Parameter

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcode 4 years | Use T_obs=4 years (LISA nominal mission). Simple constant in constants.py. | |
| Derive from existing constants | Compute from YEAR_IN_SEC and LISA_STEPS. Currently gives ~1 year. | |
| Make it a parameter | Add T_obs as a configurable parameter. More flexible. | Initial pick |

**User's choice:** Make it a parameter (initially), then refined to LisaTdiConfiguration field with default 4yr.

**Follow-up: Where should T_obs live?**

| Option | Description | Selected |
|--------|-------------|----------|
| LisaTdiConfiguration field | Add t_obs_years: float = 4.0 to the dataclass. Clean and local. | |
| CLI argument | Add --t_obs to arguments.py, thread through main.py. | Initial pick |

**User's note:** "As this also affects the waveform simulation it should be a CLI argument, or are these timespans independent of each other?"

**Claude's clarification:** Waveform T_obs (signal duration from few/fastlisaresponse) and confusion noise T_obs (mission duration for foreground subtraction) are independent concepts. You can observe a 1-year waveform snippet while assuming 4 years of confusion noise subtraction.

**Refined choice after clarification:**

| Option | Description | Selected |
|--------|-------------|----------|
| LisaTdiConfiguration field, default 4yr | Independent of waveform duration. No CLI change. Can be overridden per-instance. | ✓ |
| CLI argument anyway | Add --lisa_t_obs for visibility. | |
| Two CLI arguments | Separate --waveform_duration and --lisa_t_obs. | |

**Notes:** User agreed that since waveform duration and confusion noise T_obs are independent, a dataclass field is cleaner than a CLI argument.

---

## T-Channel PSD Update

| Option | Description | Selected |
|--------|-------------|----------|
| A/E only | Add confusion noise only to A/E channel PSD. T-channel is a null channel. Matches success criteria. | ✓ |
| All channels | Add to A, E, and T channels with proper transfer functions. | |
| Claude's discretion | Let Claude decide during implementation. | |

**User's choice:** A/E only

**Notes:** User requested tracking the assumption that T-channel confusion noise is negligible for EMRI SNR calculations. Claude confirmed this is physically sound -- T-channel is a null channel for GW signals at long wavelengths, used for noise characterization not signal extraction.

---

## Toggle / Backward Compatibility

| Option | Description | Selected |
|--------|-------------|----------|
| Boolean field on dataclass | Add include_confusion_noise: bool = True to LisaTdiConfiguration. Tests can pass False. | ✓ |
| Always on, no toggle | Just add it. Old behavior is wrong. Tests update. | |
| Separate method | Keep original method, add new method with confusion noise. | |

**User's choice:** Boolean field on dataclass

**Notes:** Default is always-on for production. Toggle enables comparison runs and regression testing.

---

## Claude's Discretion

- Implementation details of S_c(f) computation (helper method vs inline)
- Whether to refactor S_OMS/S_TM static methods while touching the file

## Deferred Ideas

None -- discussion stayed within phase scope.
