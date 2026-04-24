---
phase: 40-verification-gate
plan: "06"
status: COMPLETE
date: 2026-04-24
requirements: []
tags: [wave-4, phase-close, verify-gate-index, state-update]
next_phase: "fix-phase (VERIFY-03 SC-3 angle audit + D(h) in --combine)"
---

# Phase 40 Plan 06: Phase Close — Summary

**One-liner:** Assembled D-20 verify-gate index and phase-level 40-VERIFICATION.md from all five VERIFY-NN verdicts; Phase 40 status GAPS_FOUND (SC-3 NOT VERIFIED, MAP=0.860 vs expected 0.73±0.01); Phase 42 triggered; Phase 41 borderline pending user decision.

## Routing Status: GAPS_FOUND

Per VERIFY-03 SC-3: the `extract_baseline` MAP metric reads 0.860 (not 0.73±0.01). Root cause
documented — extract_baseline lacks D(h) denominator correction. This is a verification-metric
limitation, not a physics regression in the `--evaluate` pipeline. Investigation required before
Phase 40 can be declared PASS.

Per VERIFY-05: mean_lb = 0.0409 (borderline band [0.03, 0.05]). Phase 41 trigger requires user
decision.

Per VERIFY-04: stage_2_trigger = true. Phase 42 routing CONFIRMED (independent of other
decisions).

## User Decisions (resolved at checkpoint)

1. **Q1 — SC-3 investigation path (VERIFY-03 MAP=0.86):** Insert fix phase.
   - Independently verify angle mapping after COORD fix (equatorial CRBs vs ecliptic catalog).
   - Check whether D(h) is missing from `--combine`/`_combine_posteriors` code path.
   - Key question: "If angles were always wrong in v2.1 it should have been self-consistent —
     why does MAP=0.86 appear only after the COORD fix?" Must be answered by fix phase.

2. **Q2 — VERIFY-05 Phase 41 routing:** Skip Phase 41. Accept mean_lb=0.041 as known
   limitation; document 19 off-grid events (3.5%). Phase 41 not triggered.

3. **Q3 — Phase 42 routing:** Deferred until Q1 resolved. The anisotropy result (STAGE-2-TRIGGER)
   may be a symptom of the same D(h)/COORD bug rather than genuine sky-dependent P_det.

## Artifacts

- `.planning/debug/verify_gate_20260423T172607Z.md`  (D-20 top-level index)
- `.planning/phases/40-verification-gate/40-VERIFICATION.md` (phase-level — SC-1..SC-5)

## Commits

| Hash | Message |
|------|---------|
| `(this commit)` | docs(40-06): Phase 40 verify-gate index + 40-VERIFICATION.md — GAPS_FOUND; Phase 42 triggered [ts=20260423T172607Z] |

## Next

Plan fix phase: angle audit (verify CRB qS/phiS frame vs v2.2 ecliptic catalog) + D(h) diagnosis
in `--combine`/`_combine_posteriors`. After fix: re-run VERIFY-03. Phase 41/42 on hold until Q1 resolved.
