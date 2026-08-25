# Runbook — next session (written 2026-08-25 ~14:15, supersedes RUNBOOK_NEXT_SESSION_31; written CLEAR-SAFE while [P3-2D] round-2 execution is IN FLIGHT — §0 tells you exactly where it stands)

**Read first:** ledger rows **#174 → #190**; `PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md`
(TWIN-CALIBRATED banked #186, ratified #187); `PROPOSAL_CATALOGUE_TWIN_PRODUCTION_20260825.md`
(**PENDING §7 — the author has NOT yet ruled adoption**); `PREREGISTRATION_P3_2D_20260825.md`
(+PA-2D-1/2) with `A20_REVIEW_P3_2D_DESIGN_20260825.md`; session board
https://claude.ai/code/artifact/ed640faf-33d7-42da-bcf2-4e2c09e59347. Ch 10½ is LIVE on Pages.

## 0. [P3-2D] round-2 execution state (the in-flight arc; workflow wf_5702d3e7-44c)

Sequence: Fix (DONE — exact per-cell erf-moment companion rule validated, banked
`ca_rhs_work2d/p3_2d_exact_mass_integral_validation.json`; fleet driver `p3_2d_fleet.py`
committed-in-tree) → Gates (RUNNING as of 14:09: the FULL companion pass
`p3_2d_companion.py`, log in the session scratchpad; then pilot seed 900101 both arms +
M2-LINK/RHS-F₂/F10(c)/ACC-extended; STOP-gated) → Cluster (fires only on gates PASS: bundle-
sync to HEAD, 24-task b0i2d fleet array + capped RHS₂ score2d array (SE ≤ 6.38e-4, ~40 CPU-h
cap), retrieval + sha256 manifest; queue-waits banked per row #185). **Resume recipe if
orphaned:** check `ca_rhs_work2d/p3_2d_companion.json` (companion banked?) → run
`p3_2d_fleet.py --stage gates` locally → on PASS submit the arrays per the prereg §7 costing
(the sbatch conventions = correspondence_fleet.sbatch + srun-wrap; NEVER resubmit cancelled
tasks) → retrieve → `--stage lhs2d` + score2d aggregation → A20 review (clean-context, xhigh;
the C-A verdict-review prompt is the template) → verdict [ORCH-banked provisional] → author.
C₂\* freezes at companion landing; POWER GATE (PA-2D-1/F14) before any TWIN2 verdict.

## 1. OPEN AUTHOR DECISIONS

1. **[RULE] `PROPOSAL_CATALOGUE_TWIN_PRODUCTION_20260825.md` §7 item 1** — production adoption
   of the 1D catalogue-leg twin (the full ladder is banked; presented-then-STOP).
2. **[RULE when it lands] the [P3-2D] verdict** (returns per §0).
3. **[DO?] the hierarchical/ensemble-coherence thread** — the ONE unexplored impostor-drag
   axis (lit-campaign finding: unproven at our σ_z/z); weeks-scale design question; NOT opened
   — awaiting an explicit author go.
4. Carried: the [P3-HGRID] claim card (falsified single-h invariant — rows #182–#184); MFG-a
   verbatim check before paper quotation; the F0-SEL cheap follow-up (per-seed dropped-event
   stats); landscape/T1; workspace `emri` expires 2026-09-23.

## 2. Standing rules & session-earned ops (delta over runbook 31)

- **Row #185 [STANDING, author]:** cluster-first ≥2 CPU-h; queue-waits banked; chronic-blockade
  reversion test. Row #187/#188 author rulings verbatim in the ledger.
- **PA-2D-2 lesson (reusable):** GH quadrature borrowed from a narrow-σ regime silently biases
  wide-σ marginals over piecewise-linear grids — the exact per-cell erf-moment rule is the fix;
  every S_4D consumer should state its σ regime.
- **Agent-prompt hard opener (10+ parking incidents):** the no-parking rule + no-resubmit rule
  + __main__-guard/CWD rule go verbatim at the TOP of every spawned agent prompt (this worked;
  omission did not).
- CI is GREEN since `26795160` (machine-of-record artifacts get skipif guards; exact-pin tests
  carry rel_tol 1e-12 for cross-arch portability).

## 3. Housekeeping

- Prune candidates: runbook 30 §3 set + `ca_rhs_work/score_chunk*_work` contaminated dirs
  (PA-CA-11 evidence — keep chunks 0–4 quarantined copies until the author has read row #186).
- The bias-state artifact refresh (rows #166–#190) still pending; the session board covers the
  narrative meanwhile.
- Chronicler: file at close (rich: three contaminated-number catches, the erf-rule lesson,
  parking-rule efficacy, the CI-debt lesson).

## 4. Resume recipe (one line)

§0's in-flight arc to its verdict → author rules items 1–2 → the production adoption commit
([PHYSICS], the row-#178 pattern) → then either the hierarchical thread (item 3, if granted)
or the paper-facing consolidation.
