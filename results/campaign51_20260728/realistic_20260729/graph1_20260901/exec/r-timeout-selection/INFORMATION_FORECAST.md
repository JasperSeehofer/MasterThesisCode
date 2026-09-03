# Stage-1 information forecast — r-timeout-selection

Date 2026-09-04. Author: batch-2 prereg author C (top-tier). Verdict-free: every number is a forecast
INPUT with its source; no registered statistic has been computed. Companion to `REGISTRATION_DRAFT.md`.

## 1. What a perfect analysis of the banked data can say (rule 9)

Everything Q1/Q2 need except two cluster reads is on disk: the 822 timeout parameter dicts and every skip
tally (seed61000 logs, 100/100 tasks), the pool of record (200,100 completed draws with M_z, p0, SNR), the
CRB CSV (per-event information), the two 41-h re-baseline CSVs, the pinned influence vector. What the
banked data CANNOT say: (a) the pool's own timeout rate by M_z (its build log is cluster-only — §8 A of the
draft); (b) the per-bin rate among ALL draws on the simulation side — SNR-failed draws and D1-gate drops
log no parameters, so the sim side is reconstructed through the pool a-stratum under an identical-draw-
measure assumption (`g-closure`); (c) whether a timeout correlates with SNR at fixed (d_L, M_z) — only the
rescue run (§8 C) sees it.

## 2. Forecast inputs

| input | value | source | tag |
|---|---|---|---|
| per-draw SNR-stage timeout rate | 822 / 89,456 = 0.92 % (conditional on {kept ∪ timeout}: 34.0 %) | seed61000 logs; read record | [LOCAL] / [DOC] |
| timeouts by M bin (n) | 206 / 302 / 216 / 81 / 15 | `rate_table_M.csv` | [DOC] |
| pool a-stratum M_z support | 29.8 % below 1.955e5; 4.9 % above 3.334e6 | pool of record | [LOCAL] (design input) |
| D1 gate | 4,071 / 5,921 SNR-passers dropped (68.8 %); kept p0 = [10.0025, 15.987] | logs; CRB CSV; D1 record 69.3 % | [LOCAL] / [DOC] |
| kept vs pool-detected M composition (bin 2) | 80.4 % vs 82.7 % (n = 1590 vs 7,548) | CRB CSV; pool a & SNR ≥ 20 — **pre-read of one S2.3 input, disclosed (draft §10)** | [LOCAL] |
| generation time of kept waveforms | 0.16–0.62 s (vs 90 s budget) | CRB CSV `generation_time` | [LOCAL] |
| 2D offset and width | mean_h 0.6658540600, σ_h 0.018475; offset −0.064 carried by k = 82 events | rows #302/#342 | [DOC] |
| G9 first-order bound (stale era, 30/90 s) | "sub-percent of the detected sample" | `G9_timeout_scan.md` §7 | [DOC], STALE ([A11]) |
| D1 tilt-route materiality | m_S = 0.032, m_R = 0.011 (thresholds 0.25) — bounded null | row #94 | [DOC] |

## 3. Expectation (author's own reading, not a measurement)

**Q1.** The pool and the simulation drop timeouts identically in code and share the 90 s budget, generator
and draw measure; the residual asymmetry is hardware (A100-only pool vs mixed H100/A100 simulate) and load.
If the pool's low-M_z rows survived at a rate comparable to the simulation's (the pool's 29.8 % below 1.955e5
against 508 sim timeouts in those bins scaled by s ≈ 0.86 — i.e. ~25k expected completed low-M draws vs ~500
timeouts), then `P_complete^sim` in the low-M bins is ~0.98, not ~0.03, and the whole "97–100 % loss" is the
conditional-rate artefact of `MECHANISM_NOTE.md` §3(a). Under that reading S1.2 lands SHARED-FILTER and S1.3
is bounded by a ≲ 2 % multiplicative tilt on p_det in bins that hold < 1 % of kept events →
**expected P_DET-MISSPECIFIED-IMMATERIAL** (|Δ| ≲ 1e-3). The alternative — the pool's low-M rows are the
survivors of a pool-side ~97 % loss — is only distinguishable with §8 A; it would put the pool's pre-timeout
low-M share near 90 %, which the M1 rate density (`cosmological_model.py:284-290`, R_emri rising to
2.5e5 M☉ then ∝ M^−0.25) does not obviously support. Either way the p0 axis is D1's, and D1's own
materiality is a ratified bounded null (m_S = 0.032).

**Q2.** The kept M composition already tracks the pool's SNR ≥ 20 composition in the dominant bin (80.4 vs
82.7 %), which is the SNR-threshold reading (c-Q2-snr). Re-weighting to the pool-detected shares over bins
1–3 moves ≲ 20 % of the ensemble weight between bins whose per-event likelihoods differ mainly through
z (M_z–z correlation) — a shift of order (Δ share) × (between-bin mean_h difference); with the influence
vector's 82-event structure NOT known to be M-aligned, the honest prior is **POPULATION-MISMATCH-IMMATERIAL
to INTERMEDIATE**, with the width ratio near 1. An M-STRUCTURED S2.2 (influence concentrating in bin 3,
the high-M, high-z shoulder) is the one outcome that would make Q2 a live lead — it would join the
r-offset-subset covariate C10 (`log10 M` with the timeout-bin edge) rather than open a new thread.

**Expected nulls (secondary):** e0 axis flat (already 1.0σ); `δ^den(h)` ≡ 0 if SHARED; `σ'_h/σ_h ∈ [0.95, 1.05]`.

## 4. Decision value

The arm's value is not the H0 number but the record: it replaces a two-axis "14σ/13.6σ systematic
candidate" with (i) a per-draw rate, (ii) a corrected p0 row, (iii) a sharing test with a named hardware
caveat, and (iv) a cost-banded rescue experiment that decides whether the 90 s constant is a defect. If
both dispositions are IMMATERIAL, G7 row 8 closes as a bounded, disclosed selection with the stale 30/90 s
premise retired; if either is MATERIAL, the rescue run (§8 C) is the next measurement, not a new pool.
