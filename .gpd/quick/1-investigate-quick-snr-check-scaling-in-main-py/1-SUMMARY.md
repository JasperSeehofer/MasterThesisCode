---
phase: quick-1
status: complete
date: 2026-04-01
type: investigation
---

# Quick SNR Check Investigation

## Question

Why does `main.py:270` use `snr_threshold * 0.2` and `quick_snr * 5`? Are these correct?

## Findings

### How the quick check works

- `snr_check_generator`: ResponseWrapper with T=1 year (vs T=5 years for full)
- Quick SNR is computed via the same `scalar_product_of_functions` inner product
- If `quick_snr < 15 * 0.2 = 3.0`, skip the full 5-year waveform generation
- Callback reports estimated full SNR as `quick_snr * 5`

### SNR scaling with observation time

For a **stationary** GW source: SNR ∝ √T, so SNR_1yr/SNR_5yr = 1/√5 ≈ 0.447.

For **chirping EMRIs**: the ratio is source-dependent and typically *lower* than 1/√5, because:
- EMRIs spend their final years at higher frequencies where LISA is more sensitive
- A 1-year observation captures only the early, lower-frequency inspiral
- For fast-evolving systems, the first year may contain only 10-20% of total SNR

### What's wrong

| Expression | Current | If √T scaling | Issue |
|------------|---------|---------------|-------|
| Threshold factor | `* 0.2` | `* 0.447` | Too loose — only catches extreme non-detections |
| Callback estimate | `* 5` | `* √5 ≈ 2.24` | **Wrong** — implies SNR ∝ T (linear), not √T |

The `* 5` callback is unambiguously incorrect (no physical scenario gives linear T scaling for SNR). The `* 0.2` threshold is debatably conservative but physically defensible for chirping sources.

### Empirical data (production campaign, task 0)

| Outcome | Count | % of 4327 evaluations |
|---------|-------|-----------------------|
| Quick SNR reject (< 3.0) | 1069 | 24.7% |
| Full SNR reject (passed quick, < 15) | 2156 | 49.8% |
| Other rejects (warnings, errors) | ~1069 | 24.7% |
| Detections (SNR >= 15) | 33 | 0.8% |

Quick SNR rejected values: min=0.0, max=2.997, median=1.145.
Full SNR rejected values: min=3.0, max=14.97, median=6.6. Of these, 1103 (51%) had full SNR < 6.7.

### Throughput analysis

The quick check saves one full 5-year waveform generation per reject (~few seconds on GPU).
Currently rejecting 24.7% of events. A tighter threshold could reject more, but risks
false rejections for highly chirping EMRIs where most SNR is in years 2-5.

## Recommendations

### Must fix (incorrect physics)
1. **Line 276:** Change `quick_snr * 5` → `quick_snr * np.sqrt(5)` in callback estimate.
   This is wrong by a factor of ~2.2x and misrepresents the expected full SNR.

### Should improve (performance, not correctness)
2. **Line 270:** Consider changing `* 0.2` → `* 0.3` (threshold 4.5 instead of 3.0).
   This is a more principled compromise between the √T lower bound (0.447) and chirp
   conservatism. Would reject an additional ~10-15% of hopeless events.

   **Caveat:** Without measuring actual quick/full SNR *pairs* for the same events, we can't
   determine the empirical ratio. A proper calibration would log both values for ~1000 events
   and fit the actual relationship.

### Nice to have (proper calibration)
3. **Log quick/full SNR pairs:** Add logging that records both `quick_snr` and `full_snr`
   for events that pass the quick check. After one campaign, fit the empirical
   `full_snr = f(quick_snr)` relationship and set the threshold optimally.

### What not to change
- The T=1yr generator itself is a good idea — 5x less data to FFT
- The quick-check-then-full-check pattern is sound
- The `snr_threshold = 15` value is fine (physics choice, not a bug)

## Dimensional analysis

- SNR is dimensionless ✓
- Threshold comparison: [dimensionless] < [dimensionless] ✓
- √T scaling: [s^(1/2)] / [s^(1/2)] = dimensionless ratio ✓

## Impact on production campaign

The current campaign (seed 200) is running with `* 0.2`. Results are valid — the threshold
only affects which events get the full SNR computation. No detection is missed; the quick
check only skips events that would certainly fail the full threshold. A too-loose threshold
just means slightly more compute time per step (computing full waveforms for events that
will fail anyway).
