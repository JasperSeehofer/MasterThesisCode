---
phase: quick-1
plan: 1
type: investigation
wave: 1
autonomous: true

must_haves:
  truths:
    - "Document how the quick SNR check works (T=1yr vs T=5yr)"
    - "Derive the correct SNR scaling factor between short and full observation"
    - "Identify all incorrect scaling factors in main.py"
    - "Propose corrected threshold and callback estimate"
    - "Assess impact on simulation throughput"
---

<objective>
Investigate the quick SNR pre-check in main.py:270 — determine whether the 0.2 threshold
factor and the *5 callback estimate are physically correct, and if not, derive the right values.
</objective>

<tasks>

<task type="auto">
  <name>Task 1: Analyze SNR scaling between T=1yr and T=5yr generators</name>
  <action>
  The quick SNR check uses a T=1yr ResponseWrapper while the full SNR uses T=5yr.

  **SNR scaling with observation time:**

  SNR² = 4 Re ∫ |h̃(f)|² / Sn(f) df

  For a quasi-monochromatic source, |h̃(f)|² ∝ T² (longer observation → narrower frequency bin
  with more power). But the frequency resolution is df = 1/T, so the integral sum has T bins.
  Net effect: SNR² ∝ T, hence SNR ∝ √T.

  However, EMRIs are NOT quasi-monochromatic — they chirp significantly over 5 years. A 1-year
  observation captures only the early, lower-frequency part of the inspiral. The SNR ratio
  SNR_1yr / SNR_5yr depends on the specific source parameters and is NOT simply 1/√5.

  For chirping sources, the actual ratio can vary from ~1/√5 (nearly monochromatic) to much
  lower (fast chirp where most SNR accumulates in final years).

  **Current code analysis:**
  - Line 270: threshold = snr_threshold * 0.2 = 15 * 0.2 = 3.0
  - Line 276: callback estimate = quick_snr * 5

  **If SNR ∝ √T:**
  - Correct ratio: SNR_1yr / SNR_5yr = 1/√5 ≈ 0.447
  - Correct threshold: 15 * 0.447 ≈ 6.7 (current 3.0 is too loose)
  - Correct callback estimate: quick_snr * √5 ≈ quick_snr * 2.24 (current *5 overestimates)

  **Key question:** Is 0.2 intentionally conservative (to avoid false rejections of chirping
  sources), or was it a mistake?

  Evaluate: how many events does the quick check currently reject vs how many would a
  tighter threshold reject? Look at the production campaign logs for statistics.
  </action>
</task>

<task type="auto">
  <name>Task 2: Write investigation summary with recommendations</name>
  <action>
  Document findings and recommendations in the summary file.
  </action>
</task>

</tasks>
