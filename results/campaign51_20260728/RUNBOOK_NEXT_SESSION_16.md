# Runbook — next session (written 2026-08-17, supersedes RUNBOOK_NEXT_SESSION_15)

**Read first:** `docs/derivations/PROPOSAL_2D_SELECTION_FUSION_20260817.md` (the pending
decision), then `results/mechanism_study_20260813/AFULL2D_ARM_READOUT_20260817.md` and the
prereg's §9. Ledger rows #115–#116 carry the rulings.

## 0. State (end of the 2026-08-16/17 session)

- **2D venue thread: CLOSED, M-OWNED, ratified** (row #116, branch 1): the fused `g_sel`
  correct-form estimator passed its registered arm on 25 fresh seeds — DS-G1 −11.8 ± 0.61
  (band [−15.7, −7.8], mirror prediction −11.74), 2D bias +0.0006 ± 0.0013, coverage
  restored (necessary-but-weak read per verifier MAJOR-1), 1D bit-untouched (DS-G4 = 0.0).
  Chain: derivation `09c02c06` → premeasure `fbc60b3a` → verifier `453d1b29` → registering
  commit `d50de222` → arm readout `bcd66529`.
- **Production derivation banked** (`L6_DER3_..._20260816.md`, `e3eec5c0`): production is
  latent-thresholded in (z,M) → the MFG/Gray denominator-only arrangement is
  convention-conditional; fused survival belongs in BOTH completion legs (paired), catalogue
  leg is its own fork (= row #110 Gray-convention paper task, mixture-skew forcing function).
- **The `/physics-change` proposal is AUTHORED and PENDING** (`298c4963`): decision table
  items 1–5 ([P1]+[P2] paired fusion, [P3] defer-unless-material, [P4] measure ruling inside
  item 1, [P5-3] production counterfactual before any campaign re-run, xhigh verifier on the
  proposal). **The slot moves from occupied-paused to occupied-active when the author rules.**
- Open residuals of record: the −11.7-class residual (r = 0.847 with the c2 switch residual;
  origin decomposition open), pool-vs-model prior mismatch, low-dose FULL-F residual, book
  ch14 (owed the entire L4→L6→arm arc), Gray-convention paper task (row #110), N-2 sel_1d
  claim still DRAFT.
- Budget deviation of record (row #116 item 2): AFULL2D arm realized 406.5 CPU-h vs 300
  ceiling; future prereg ceilings use the pessimistic premeasure rate (verifier MINOR-4 was
  right).

## 1. Next tasks

1. **Author adjudication of the proposal's decision table** (items 1–5). [RULE]/[DO] mix as
   tagged in the table; item 1 approval starts the full /physics-change gate (presentation →
   implementation → checks → [PHYSICS] commit → PHYSICS-GATE-LEDGER row).
2. If item 5 approved: xhigh verifier on the proposal BEFORE implementation begins.
3. If item 1 approved (post-verifier): implement [P1]+[P2]+[P4] behind the gate; [P5] items
   1–2 in the same commit.
4. Then item 4: the production counterfactual paired cell (old vs fused, same seeds) — its
   own mini-prereg; campaign re-run scope returns with its result.
5. Carried: book ch14 (the whole arc is unwritten); veto-flagged branch readings rows
   #109-#111; author WIP (3 book files) still uncommitted in the working tree.

## 2. Standing constraints

Append-only; A8-v2 on registration; top-tier cap ≤3 inherit/workflow; branch calls presented,
never self-adjudicated; results scripts run from repo root; A3 (venue magnitudes never
transfer) governs all production expectations.

## 3. Operational notes

- **Workspace expires 2026-09-23 with 0 extensions** (`ws_extend` exhausted this session).
  AFULL2D outputs are already retrieved+committed; anything else needed from the workspace
  must be copied off before then.
- Cluster SSH needs interactive re-auth when the ControlPersist socket lapses (KIT 2FA):
  author runs `! ssh bwunicluster true` in-session.
- Subagent ops note: background watchers on subagents silently lose exited processes — brief
  agents to poll with foreground bounded waits (`timeout 550 tail --pid=... -f log`), never
  wait on notifications.

## 4. Resume recipe

1. `git log --oneline -3` — expect the proposal commit `298c4963` at HEAD or a descendant.
2. Read §0's first document (the proposal). 3. Author rules on the table; execute per §1.
4. Nothing in production changes before the gate; nothing runs on the cluster before item 4's
   own registration.
