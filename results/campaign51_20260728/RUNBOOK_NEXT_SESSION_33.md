# Runbook — next session (2026-08-25 close, supersedes runbook 32; rows #198 → #202; everything committed AND PUSHED)

**Read first:** ledger rows **#198 → #202** (the one-day [P3-WBHZERO] arc: author's measure-first
ruling → flag `9c948ea0` → prereg + PA-WBZ-1/2/3 → mirror EXCLUSION-MATERIAL (row #200) →
production read 43.3%→0.0% (row #201) → **symmetric window ADOPTED, [PHYSICS] `cf4f8a2a`**
(row #202)). The 6-item package: `docs/derivations/PROPOSAL_MASS_FILTER_SYMMETRIC_20260825.md`.
Mission board: https://claude.ai/code/artifact/1b605b24-078a-48b9-bf92-1e573965f9c3

## 0. State at close

- **[P3-WBHZERO] CLOSED.** Symmetric is production physics; explicit `"asymmetric"` = the
  counterfactual. Suite 1827 green, CI expected green. All measurement artifacts banked under
  `realistic_20260729/wbhzero_work/` (readout.json, prod_readout.json, gates, hostcounts).
- **[P3-2D] un-HOLD is the next action** (row #202 sequencing): A21-amend
  `PREREGISTRATION_P3_2D_20260825.md` for (i) the SYMMETRIC eligibility model (the twin
  calibrates against post-adoption production), (ii) the M2-LINK(iii) re-attribution (zeros =
  filter exclusions — now FIXED by the adoption; the 7/84 pilot zero class should VANISH on
  re-run: a free prediction, register it), then run the fleet/RHS₂ per the registered costing.
- **Companion re-run (PA-2D-3) before any C₂\* freeze:** segment-aware z-quadrature (GL(50)
  under-resolves — row #199), arbiter-grounded spot-check target, AND the
  eligibility-independence check vs the symmetric adoption (Σ̃^4D's draw law binds to the
  candidate set). The held candidates (348079019.37 / 0.061244) are superseded numbers.
- **[HIER]** stays sequenced after [P3-2D] (rows #192/#195/#197).
- No cluster jobs; ~20 GB pruned locally (author ran the rm; classifier blocks bulk rm for
  agents — one-liner via `!` is the pattern).

## 1. OPEN AUTHOR DECISIONS

None pending. Carried non-blocking: [P3-HGRID] claim card; joint_r1 attribution (needs the
cluster-side r1 observed-catalogue artifact); MFG-a verbatim check before paper quotation;
F0-SEL cheap follow-up; AMEND-2 stale log-substring gates (fix on next instrument use);
bias-state artifact refresh (rows #166–#202); workspace `emri` expires 2026-09-23.

## 2. Open physics questions spawned by the adoption (proposal §6 caveats — candidate threads)

1. **Filter-vs-kernel model consistency** (caveat 2): the mass filter remains an unmodeled
   numerator selection even symmetric; the modeled limit is kernel-weighted eligibility with
   no hard window. Un-opened thread; would need its own stage-0.
2. **h-dependence of the adoption Δ** (caveat 1): measured at h=0.73 only; fresh costing line
   if wanted (a multi-h production read ≈ 42 min/h-value local).
3. **The redshift filter's sibling ±1σ convention** (caveat 4): same unratified interval form,
   out of the row-#198 grant; deliberately untouched.
4. The production-vs-mirror Δw̄ order gap (+0.00045 vs +0.0049) — recorded fact, uninterpreted.

## 3. Standing rules & session-earned ops (delta over runbook 32)

- **The agent-prompt hard opener DOES NOT reliably prevent parking:** 2 incidents THIS session
  with the opener verbatim at top (both auto-backgrounded a >120s command, then waited).
  Working correction: the opener now must also say "pass a longer timeout instead of
  backgrounding"; a direct resume message fixes an already-parked agent.
- **Background Bash instability:** two chained background commands were killed instantly by
  the environment (not the author); single-stage background commands ran fine. Prefer
  foreground with explicit long timeouts for chained/critical runs.
- **PROD-A0 lesson (reusable):** a banked production run is only a valid counterfactual arm at
  the physics it was run under — post-banking no-flag adoptions (the 0.665035804 completion
  multiplier removal) make fresh-at-HEAD paired arms the correct design; ingredient-level
  comparison still validates the config (PA-WBZ-3 pattern).
- **Spot-check targets must be falsifiable:** a 1e-6 target with a ~1e-4-noise-floor arbiter
  is unfalsifiable as posed (row #199); derive targets from a demonstrated-convergent arbiter.
- Row #185 cluster-first + queue-wait banking unchanged; this session's fleet ran local
  (~1–2 CPU-h, below threshold, disclosed).

## 4. Resume recipe (one line)

[P3-2D] A21 amendment (symmetric model + M2-LINK re-attribution + the vanishing-zero-class
prediction) → companion re-run w/ segment-aware z-rule → [P3-2D] fleet/RHS₂ to its verdict →
[HIER] (h,θ)-grid prereg → paper-facing consolidation (the WBHZERO arc is a thesis chapter:
defect discovery by a registered gate, measure-first chain, one-day adoption).
