# Runbook — next session (written 2026-08-17, supersedes RUNBOOK_NEXT_SESSION_16)

**Read first:** `results/run_20260817_fusion_counterfactual/PREREGISTRATION_FUSION_COUNTERFACTUAL.md`
(the registered next measurement), then `docs/derivations/GATE_PRESENTATION_SELECTION_FUSION_20260817.md`
+ `PROPOSAL_2D_SELECTION_FUSION_VERIFIER_ADDENDUM_20260817.md`. Ledger rows #117–#118 carry
the rulings; PHYSICS-GATE-LEDGER rows at `2b10b8b8`.

## 0. State (end of the 2026-08-17 session)

- **Selection fusion IS LANDED IN PRODUCTION** (`[PHYSICS]` commit `2b10b8b8`): fused
  survival in both `absolute_marginal` completion legs — [P2] S̄_φ in the 1D numerator,
  [P1] `completion_mass_factor_g_sel` (S_4D inside the mass quadrature) in the 2D leg.
  `--selection_in_completion_numerator` default `auto` → `fused` under absolute_marginal;
  cells `off`/`1d`/`2d` are counterfactual decompositions. 1506 tests pass; off/1d cells
  byte-frozen against pre-change pins (recorded at `4ab5da0e`).
- **Chain of the day:** row #117 (all 5 items ratified) → xhigh verifier GO-W-AMENDMENTS,
  MAJOR-1..4 + MINOR-1..6 (`44aa239e`) → author rulings G1 adaptive+guard / G2 ratio+track
  / G3 deferral confirmed (row #118, `4ab5da0e`) → gate presented/implemented/verified
  (`2b10b8b8`) → item-4 mini-prereg REGISTERED (fusion counterfactual).
- **Key facts of record:** production σ_cond (d_L-conditional) p50 = 8.8e-8 → sharp-likelihood
  regime, expected action 1D-dominated, [P1] possibly near-inert (MAJOR-1); [P3] skew
  direction CORRECTED — catalogue leg OVER-weighted under [P2] (MAJOR-3); V2 = G7 systematics
  row 17 (G2); G1 recorded bound 6.65e-16 adaptive-vs-pinned.
- Open residuals unchanged: −11.7-class residual (r=0.847), pool-vs-model prior mismatch,
  low-dose FULL-F residual, #66/#67 calibration caveat (carried in the prereg), book ch14,
  Gray-convention paper task (row #110, now also owed the M-4 skew input), N-2 claim DRAFT.

## 1. Next tasks

1. **Execute the fusion counterfactual** per its prereg: 4 CPU evaluate arrays (off/fused ×
   iiib/joint_r1) at `2b10b8b8`+, canonical 41-h grid. BEFORE submission: author runs
   `! ssh bwunicluster true` (KIT 2FA re-auth), then `/cluster` preflight (require READY),
   then fill the budget ceiling in the prereg from `sacct` of jobs 6152554/6152556
   (pessimistic ×1.15×1.3 per the prereg) and commit that append BEFORE `sbatch`.
2. **Readout** → comprehension-first report (A7); M-4 skew returns to the author as the
   fresh [RULE] for decision-table row 2; M-3 feeds the campaign-re-run decision (returns
   to author — NOT pre-authorized).
3. Carried: book ch14 (whole L4→L6→fusion arc unwritten); veto-flagged branch readings rows
   #109–#111; author WIP (3 book files) still uncommitted in the working tree.

## 2. Standing constraints

Append-only; A8-v2 on registration; top-tier cap ≤3 inherit/workflow; branch calls presented,
never self-adjudicated; results scripts run from repo root; A3 governs (venue magnitudes
never transfer — the prereg registers NO bands, it is a measurement seeded from nothing).

## 3. Operational notes

- **Workspace expires 2026-09-23 with 0 extensions.** Retrieve+commit counterfactual outputs
  immediately after completion.
- Explicit `--selection_in_completion_numerator off` is REQUIRED for the off twin (`auto`
  now resolves to `fused`).
- The old off-runs of record are NOT byte-comparable at `2b10b8b8` (ratified 08-12 φ/Route-1
  divergence classes) — hence the fresh off twin; NULL-2 in the prereg bounds the drift.
- Subagent ops: poll with foreground bounded waits, never background watchers.

## 4. Resume recipe

1. `git log --oneline -3` — expect the prereg commit at HEAD or a descendant.
2. Read the prereg. 3. Author SSH re-auth → preflight → fill budget → submit per §1.
4. Nothing campaign-scoped runs before the author rules on M-3.
