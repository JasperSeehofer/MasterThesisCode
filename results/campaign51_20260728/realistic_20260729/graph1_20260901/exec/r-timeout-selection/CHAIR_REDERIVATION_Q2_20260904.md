# m-timeout-q2 — chair note + booking (2026-09-04 ~12:00 CEST)

Read of record: READ_RECORD_Q2.md + timeout_q2_result_read.json (disjoint reader, real mode once).
Gates: computability rev2 (RED only on F6, closed by the chair erratum) + formula gate GREEN
(31-item own enumeration; T0 import verified; INSTRUMENT-DEFECT contract verified). DISCLOSURE: the
rerun's computability-gate slot returned a placeholder ({"overall":"GREEN","checks":[]}) — that agent
did no review; the effective computability gate of record is DESIGN_GATE_Q2_computability_rev2.md
with the F6 erratum. Non-blocking F1 (S2.1 p0/e0 reported-only bins use qcut, not the pinned edges)
is carried; it touches no disposition.

## Numbers (verbatim from the JSON; chair spot-checks in brackets)
- g-byteid: n_kept [0,9,1276,303,0], n_timeout_snr [206,302,216,81,15] — anchors matched.
- S2.2 (iiib_2d, k = 82): ρ_S(log10 M, d_e) = +0.226, permutation p < 1e-4; top-k bins Holm p 1.7e-9
  → M-STRUCTURED. Replicates ρ = 0.185 (iiib 1D), 0.230 (joint_r1 2D), both p < 1e-4.
- S2.3 (iiib_2d): w_b = {bin 2: 0.870, bin 3: 1.547} over supported bins {2,3}; [pool-detected share
  of bin 3 = 1852/(4387+1852) = 0.297 vs kept 303/1579 = 0.192 → 0.297/0.192 = 1.547 ✓; bin 2
  0.703/0.808 = 0.870 ✓]. Re-weighted mean_h 0.655075 vs anchor 0.665854 → Δmean_h^Q2 = −0.01078
  (T_mat 0.008; null SD 0.00299, T_null 0.00598) → 3.6 null-SDs; σ'_h/σ_h = 1.0055 (inside [0.80,1.25]).
  Replicates: iiib 1D −0.00502 (below T_mat), joint_r1 2D −0.01274.
- Dispositions (script): Q2-S2.2 M-STRUCTURED; Q2-S2.3 POPULATION-MISMATCH-MATERIAL (|Δ| ≥ T_mat).

## Booking (chair-derived; returns as fresh RULE R19)
POPULATION-MISMATCH-MATERIAL on the primary (2D), with the 1D replicate below band (INTERMEDIATE
across channels — disclosed). Facts for the decider: the pool's DETECTED population is richer in the
high-M bin than the kept (timeout-surviving) population; re-weighting toward the pool composition moves
mean_h DOWN by 0.011 — i.e. the timeout truncation currently biases the estimate UP by ~0.011,
partially masking the −0.064 offset. Since the timeouts are a runtime-budget artefact (90 s, absent
from p_det), the "true" production population would sit ~0.011 further from truth. This is a
selection systematic of the injection pipeline, not of the estimator; the fix is a longer timeout /
rescue re-run (NOT-covered [DO] C in the draft §8: 5–140 GPU-h), which returns to the author.
