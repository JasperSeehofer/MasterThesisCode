# Campaign #51 pilot readout — detectability-verified-narrowing (2026-07-28)

> **[QUARANTINED same day — HIGHM_AUDIT.md]** The pilot's SNRs carry the
> confusion-noise TDI-transfer artifact (fixed in `49251f3`: suppression up
> to ~1100× above m ≈ 6.2) and the snapshot-p0 convention issue (audit item
> 1). The horizon table below is NOT the physical horizon; the narrowing-rule
> scoring is VOID (it would not reproduce on corrected data). What stands:
> the mixture-sampling mechanics (strata realized on target, zero
> tracebacks), the throughput measurement, the timeout profile, and the
> no-FEW-failure-wall finding to M_z = 10^7.37. The pilot re-runs after the
> initial-condition convention decision.

**Setup.** First campaign tranche = pilot: job 6070423, 60 A100 tasks × 100
events = 6,000 injections under the stratified mixture (`--injection_mixture`,
stack `8535ab2`), seeds 51000–51059, run
`injection_20260728-080420_seed51000`. Realized strata a/b/c =
2971/1536/1493 (target 3000/1500/1500). Zero tracebacks; 51 timeouts (0.85 %,
all skipped); ~3.5 min wall/task (100 steps). Detector-frame support covered:
m = log₁₀ M_z ∈ [4.02, 7.37]. Scored against the PRE-REGISTERED §5 rule of
`docs/campaign_redesign_51_design.md` (committed before submission).

## Measured horizon structure (the sizing-analysis expectation is OVERTURNED)

`SIZING_ANALYSIS.md` §7 predicted (from the capped 50k pool) that the d_hor
plateau "continues above 10⁶ with near-certainty" and expected NO narrowing.
Measured on the uncapped band: detections continue only briefly above the old
wall — the horizon then **collapses**:

| detector m bin | n | detections | max d_hor [Gpc] | p90 d_hor |
|---|---|---|---|---|
| 5.8–6.0 | 506 | 91 | 5.48 | 4.37 |
| 6.0–6.2 | 511 | 33 | 5.30 | 1.18 |
| 6.2–6.4 | 429 | 0 | 0.202 | 5.4e−3 |
| 6.4–6.6 | 455 | 0 | 2.5e−3 | 6.9e−4 |
| 6.6–6.8 | 479 | 0 | 2.5e−4 | 9.3e−5 |
| 6.8–7.0 | 497 | 0 | 9.1e−5 | 2.7e−5 |
| 7.0–7.4 | 172 | 0 | 2.2e−5 | 3.0e−6 |

Last detection at m = 6.143; 0 detections in 1,603 rows at m ≥ 6.4
(rule-of-three: P(det) ≤ 0.19 % at 95 % CL for that population). Five-decade
d_hor collapse over 0.6 dex — smooth, physical (fixed μ = 10: mass ratio and
in-band flux fall together), NOT a waveform artifact.

**Bias checks (both pass).** (i) Timeouts are LOW-mass (logged-param subset
n = 17: median M ≈ 1.8e5, max 1.37e6, 6 % above 1e6) — long many-cycle
inspirals, not high-m suppression. (ii) High-m SNRs: zero-fraction = 0
everywhere; medians decline smoothly 1.2e−1 → 1.5e−2 → 1.1e−3 → 1e−4 across
m ∈ [6.0, 7.4] — real signal evaluation, no FEW validity wall observed up to
M_z = 10^7.37 (the §8 FEW-validity open item closes: no systematic failure).

## Pre-registered rule scoring

Threshold ½·d_L(z_min-cat = 1e−5, h = 0.60) = 2.50e−5 Gpc. Smallest lg* with
every 0.2-dex bin wholly above lg* + log₁₀(2.5) below threshold:
**lg\* = 6.5** (bins ≥ 6.9: 2.2e−5, 3.8e−6 Gpc ✓; lg\* = 6.4 fails on the
6.8–7.0 bin at 9.1e−5). ⇒ **Narrowing to source 10^6.5 is VERIFIED
AVAILABLE** under the strictest reading of the rule.

## Decision: narrowing DECLINED [AUTHOR-REVIEW]

The rule is permissive ("may be narrowed"), not mandatory. Declined because:
1. Adopting would reintroduce a second draw constant below the model band —
   the exact two-constant clamp pattern deleted by `ecb56d6` and forbidden by
   the session directive ("one clear place … no additional clamping").
2. The saving is ~20 % of the cheapest stage (injections measured at
   ~3.5 min/100 events: whole 200k pool ≈ 30 A100-hours).
3. Keeping the band makes p_det ≈ 0 above the collapse a MEASURED statement
   across the full support — selection-function tails and any future
   catalogue revision rest on data, not clamped extrapolation.

The verified-available boundary (source 10^6.5 / detector ≈ 6.9) is recorded
here for the paper's selection-function discussion.

## Campaign consequences

- Bulk released (this decision gate passed): job 6070769 (243×400) +
  follow-up array ≈ 97k + 97k rows → 200k total with the pilot.
- Throughput measured: 400-step tasks ≈ 10–14 min ≪ 30-min cap.
- The catalogue's high-m rate weight (81.4 % above old support) is now
  *covered* by the pool but measured UNDETECTABLE above m ≈ 6.2 — the (g1)
  clamp pathology is replaced by measured ~zero survival, which is the
  scientifically correct object.

Provenance: pilot CSVs local-only (`pilot/injections/`, 60 files);
`pilot_decision.json` (machine-readable verdict); cluster logs in the run dir.
