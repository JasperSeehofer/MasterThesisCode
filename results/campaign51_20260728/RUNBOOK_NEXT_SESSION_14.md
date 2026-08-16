# Runbook — next session (written 2026-08-16, supersedes RUNBOOK_NEXT_SESSION_13)

**Read first:** `results/mechanism_study_20260813/PRODUCTION_TRANSFER_RECON_20260816.md` (the
open scope fork), then `STAGE5_READOUT.md` (+ RATIFIED addendum) and `L6_2D_GI_PLAN_20260816.md`.
Ledger rows #109–#111 carry the rulings.

## 0. State

- **The 1D venue mechanism thread is CLOSED, M-OWNED, ratified (row #111):** A-FULL (FULL-F)
  zeroed tilt (+22.0 ± 29.2), zeroed bias (+0.0010 ± 0.0011), restored 1D coverage
  (0.64/0.76/0.96) — first configuration ever to do so. Job 6327889, 1:17:52, DS-F1 PASS at
  0.16σ from the mirror prediction. Full account: α-pairing (D1+D4 one defect) + GW z-mass
  growth + exponent scale + LOO weight; renorm and Jacobian refuted as correct-form terms.
- **Row #111 item 2 (production physics-change proposal) is RETURNED as unexecutable as
  premised:** the recon shows D-i ABSENT in production (numerator already w_pop-paired), D-iii
  inapplicable (no synthetic ball), D-ii present but venue-measured near-inert alone — and the
  venue coded base does NOT mirror production's default event term. The production bias has two
  live candidate owners: D-ii (production magnitude unknown) and the **production-shared 2D g_i
  defect**. The `/physics-change` slot holds, occupied but paused on the fork.
- **The 2D investigation (row #111 item 3, authorized) is planned and pinned** (`L6_2D_GI_PLAN`):
  g has exactly two h-channels (mu_cond via d_L_frac; node placement via the h-indexed window);
  the +129…+136 excess is variant-independent; protocol = c2 mirror (bit-exact vs stored
  ln_post_2d) + freeze-switches S-A/S-B/S-AB + convolution-frame derivation first.

## 1. Next tasks

1. **Author ruling on the PRODUCTION_TRANSFER_RECON §3 fork:** A (2D-first, recommended) /
   B (production correspondence mirror) / C (narrow D-ii proposal now). [RULE]
2. If A (or by default of item 3's standing grant): execute L6 — derivation (orchestrator),
   c2 mirror + switches (sonnet implementable, spec in the plan), xhigh verifier, present.
   Local CPU-min; no cluster time until a repair arm is registered.
3. Carried: Gray-convention paper integration (in scope per row #110 item 4 — needs a paper-
   thread task); book ch14 (now owed the whole Part-2 → A-FULL → stage-5 arc); the pool-vs-model
   prior mismatch (KS D = 0.085) as a stated population systematic; low-dose FULL-F residual
   (+169 at f_i = 0.25) unprobed.
- **Veto-flagged branch readings on record:** row #109 item 4, row #110 item 4 (Gray in scope),
  row #111 item 3 (investigate vs carry) — one word from the author re-opens any of them.

## 2. Standing constraints

Append-only registered docs; `/physics-change` slot occupied-but-paused (no production code
changes); A8-v2 on any registration; top-tier cap ≤3 inherit/workflow; branch calls presented,
never self-adjudicated; scripts under `results/mechanism_study_20260813/` run from the repo root.

## 3. Operational notes

- Cluster idle; job 6327889 retrieved + scored + committed. Workspace expiry: extend within ~5
  weeks of 2026-08-14 preflight (`ws_extend emri 60`).
- Dev venv repaired 2026-08-15 (`uv sync --extra cpu --extra dev --reinstall`); mypy hook works.
- Author WIP (3 book files) + the 08-15 stash: still awaiting author confirmation.

## 4. Resume recipe

1. `git log --oneline -3` — expect `8b112b9e` (recon + L6 plan) at HEAD or a descendant.
2. Read §0's three documents. 3. The §1 item-1 fork is the ask; if ruled A, execute L6 per the
plan. 4. Do not draft any production physics-change on the old premise — the recon supersedes it.
