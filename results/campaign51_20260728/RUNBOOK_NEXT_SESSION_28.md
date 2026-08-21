# Runbook — next session (written 2026-08-21 ~23:20, supersedes RUNBOOK_NEXT_SESSION_27)

**Read first:** ledger rows **#157 → #158**, then the prereg's O6 blocks
(`PREREGISTRATION_SELFGEN_CONTROL.md`: CONFIRMATION RUN O6 — REGISTRATION → REFERENCE-VALUE
REGISTRATION → VERDICT) and the banked review `A20_REVIEW_O6_20260821.md`. Runbook 27's §1
decision queue is FULLY RESOLVED (row #157); its §3 housekeeping is done except where noted
below. Do not redo anything.

## 0. Where the campaign stands (one paragraph, typed)

Row #157 ruled all three open decisions: defect label **RATIFIED** (IMPLEMENTATION-CONVENTION
DEFECT, off-cell S̄_φ omission), fused confirmation seed **APPROVED**, production fork
**DEFERRED** (joint with impostor-leg). O6 then executed the same evening, **A21-clean** (first
run with a fully clean registration–execution identity), and fired **MECHANISM-CONFIRMED
[MEASURED, row #158]**: the real `fused` cell end-to-end on seed 910101 matches the pre-data
harness prediction to +1.94e-6 (band ±1e-4; the residual is fully diagnosed as 7-sf CSV storage),
with all four fail-able gates passing and the off→fused axis moving the seed +0.127368. The A20
review (second application, zero FATAL) BANKED it with scope amendments: O6 proves the
**harness→production transfer** of the mechanism — NOT the fleet-level fused null, which one
seed cannot power (fused per-seed span collapses 7.59 → 1.53 nats). Fused still does not cure
the H₀ rail (F6 full channel mean_h = 0.618, r_low).

## 1. OPEN AUTHOR DECISIONS (fresh per the binding default)

1. **[DO] Multi-seed fused arm** (closes the approved-but-unanswered "matched score ≈ 0" fleet
   claim; A20 amendment 2). Costing note: fused per-seed power is collapsed — more seeds or a
   different statistic needed; evaluate ≈ 9 GB/seed, ~30 min/seed; cluster venue for ≥4 seeds.
2. **[RULE] Production-basis fork off→fused** — DEFERRED by row #157 item 3, to be decided
   **jointly with the impostor-leg question and decision 1** via a reviewable physics-change
   proposal (`bayesian_statistics.py` trigger, full 6-item gate package).
3. Carried: landscape/T1 un-gate (chain at link 3: mechanism ✓ → fused confirmation ✓ → fix
   fork → landscape); systematics row 16 re-grade; workspace `emri` expires **2026-09-23**.

## 2. Standing rules (A17–A21 all in force; new evidence this session)

- A21 +1: first clean registration–execution identity (O6). A20 +1: second application narrowed
  an over-broad verdict sentence pre-banking. A17 fold-in (NEW): identity-band noise floors
  scored off `event_likelihoods.csv` derive from its 7-sf storage (measured 1.94e-6), never from
  internal precision. A18 exercised throughout O6.
- Disclosed O6 blind spot to carry: `precompute_phi_marginal_survival` is common-mode to replica
  and cell — an independent S̄_φ-table implementation is the only instrument that would see an
  error there.

## 3. Housekeeping

- **Overview artifact + campaign readout report refresh** — still owed if not done by end of
  this session (check the transcript/artifact before redoing).
- **Large uncommitted dirs:** `o6_work/d6_work` + `o6_work/f6_work` (386 MB each; regeneration
  deterministic — prune candidates), `o4_pairing_test_work/` (same status, carried from
  runbook 27). Full O6 logs (4.5 MB each) uncommitted; `L6_LOG_EXCERPT.txt` committed instead.
- Chronicler: the O4-session lessons were ALREADY FILED (runbook 27's "unfiled" was stale —
  verified against wiki log). This session's debrief owed at session end.

## 4. Resume recipe (one line)

Put §1 items 1–2 to the author (joint decision package: multi-seed fused arm + production fork +
impostor-leg direction, as a reviewable proposal) → then the landscape chain.
