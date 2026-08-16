# Runbook — next session (written 2026-08-16, supersedes RUNBOOK_NEXT_SESSION_14)

**Read first:** `results/mechanism_study_20260813/L6_DER_2D_CHANNEL_20260816.md` **with both
addenda** (addendum 2 supersedes everything above it where they conflict), then
`PRODUCTION_TRANSFER_RECON_20260816.md`. Ledger rows #109–#113 carry the rulings.

## 0. State (end of the 2026-08-15/16 double session)

- **1D venue thread: CLOSED, M-OWNED, ratified** (row #111; A-FULL branch 1: tilt +22.0 ± 29.2,
  bias +0.0010 ± 0.0011, coverage restored).
- **2D excess: derived, switch-confirmed, verifier-amended (GO):** channel B — h-moving
  evaluation of `completion_mass_factor_g`'s z-argument against the φ slope — **owns the 2D−1D
  excess to within ~6%** (ΔT2(S-B) = −139.0, base excess +131.5, residual −7.489 ± 0.065
  nonzero/unattributed; channel A null at f=1). Prediction +139.0 committed pre-run,
  independently reproduced by the verifier (+139.01), structural (±2% under z-shifts).
- **Production scope (verifier A3, supersedes earlier "direct" wording):** g is called ONLY in
  the `absolute_marginal` completion leg (`bayesian_statistics.py:4344`, gate `:3178`) — the
  campaign path, not the default; h-moving windows exist there (`:4368-4384`), so the channel
  EXISTS in production, but venue magnitudes do NOT transfer; the catalogue leg's 2D factor is
  `mz_integral` (no φ-slope). D-ii (ratio-form GW factor) remains present-as-is in production.
- Open items of record: the §3 correct-form M-side derivation (the cancellation-partner
  question — the only step before an A-FULL-2D candidate); the −7.5 residual's origin; the
  unattributed remainder of the 2D−1D bias gap (0.0066 total; width attribution withdrawn);
  pool-vs-model prior mismatch; low-dose FULL-F residual; Gray-convention paper task (in scope,
  row #110); book ch14 (owed the entire arc); veto-flagged branch readings rows #109/#110/#111.

## 1. Next tasks

1. **Author adjudication:** ratify the L6 findings as amended (channel B ~94–106% ownership,
   channel A null, production scope per A3). [RULE]
2. **The correct-form 2D derivation** (orchestrator, top-tier): does the correct joint
   (d_obs, M_z_obs)-density form cancel channel B via an M-side selection pairing (S̄_φ-in-α
   conditioned on the 2D data), or is the φ-prior z-drift spurious for pinned events? Output: an
   A-FULL-2D candidate (or a derived no-repair verdict). Mirror pre-measurement as in stage 4/5;
   xhigh verifier; then a registered arm decision returns to the author.
3. **Production 2D derivation** (folds into 2): the completion-leg geometry
   (`completion_numerator_integrand_with_bh_mass`, `:4334-4363`) under `absolute_marginal` —
   where the campaign's own 2D bias would come from; connects L6 to the original campaign
   problem. The production `/physics-change` proposal reopens once 2+3 give it a subject.
4. Carried: D-ii narrow fix (option C) folds into whatever proposal emerges; option B
   (correspondence mirror) remains available if 2+3 leave the production bias unowned.

## 2. Standing constraints

Append-only; `/physics-change` slot occupied-paused; A8-v2 on registration; top-tier cap ≤3
inherit/workflow (this double session used 3 verifiers + 5 sonnet implementers/recons, one per
gate); branch calls presented, never self-adjudicated; results scripts run from repo root.

## 3. Operational notes

- Cluster idle; workspace expiry ~5 weeks from 2026-08-14 (`ws_extend emri 60` when convenient).
- Venv healthy (repaired 08-15). Author WIP (3 book files) + 08-15 stash: still pending author
  confirmation.
- Mirror toolkit now covers both channels bit-exactly (`l6_c2_switch_decomposition.py` — reuse
  for any 2D switch; `l4_afull_premeasure.py --stage {abc,de,f}` for 1D variants).

## 4. Resume recipe

1. `git log --oneline -3` — expect the L6 addendum-2 commit at HEAD or a descendant.
2. Read §0's two documents. 3. If item 1 is ratified: execute item 2 (derivation first, always).
4. Nothing runs on the cluster until a 2D repair candidate passes its own registration gate.

---

## Addendum (2026-08-16, same session) — row #114 ratified; L6-DER2 derivation BANKED

Item 1 is done ("ratified", row #114). Item 2's derivation is **written**:
`L6_DER2_CORRECT_FORM_2D_20260816.md` — the coded 2D structure's **S̄_φ×g factorization error**
(two ∫dM where the selected joint prior demands one ∫dM φ·p_det·N) is the channel-B owner
candidate, with the fused `g_sel` as the A-FULL-2D code form. Next session starts at L6-DER2 §4:
(1) mirror pre-measurement of the fused candidate (sonnet-class; the l6 c2 mirror is the base;
needs per-node unmarginalized S(M,z,h) from the S̄_φ tables' integrand); (2) xhigh verifier on
derivation+measurement; (3) if confirmed, the A-FULL-2D arm gate + the production completion-leg
counterpart → the reopened /physics-change proposal.
