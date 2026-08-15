# Runbook — next session (written 2026-08-15, supersedes RUNBOOK_NEXT_SESSION_12)

**Read first:** `results/mechanism_study_20260813/DRAFT_A_FULL_ESTIMATOR_20260815.md`
**including its addendum** (addendum supersedes the body; the candidate is **FULL-F**), then
`L4_DRIFT_EVAL_20260815.md` and ledger row #109. Runbook 12's §0 state stands beneath this one.

## 0. What happened after row #109 ("all approved", 2026-08-15)

- **Ratification recorded** (row #109): Part-2 account + Part-1-ledger supersession ratified;
  A-FULL drafting granted; residuals carried (item 4 branch reading flagged for veto).
- **Drift-term direct evaluation (the A2 hedge recompute) — DONE, hardening:** the parameter-free
  closed-form model (mass + drift + width) reproduces the exact instrument tilt to ≤1.5% at all
  three doses; drift owns the dose decay (`L4_DRIFT_EVAL_20260815.md`).
- **A-FULL draft — DONE, verifier-amended (GO), and the candidate is COMPLETE:** staged
  pre-measurements on the bit-validated mirror:
  coded +2644 → FULL-A (d_obs-density) +2529 → FULL-B (w_pop pairing) −104 → FULL-D (+S̄_φ)
  +183 → **FULL-F (+ leave-one-out impostor weight 1/imp_k): +30.6 ± 42.7 — consistent with
  zero at full dose.** Kernel renorm REFUTED in the correct form (C/E overshoot ~−1100; F1
  vindicated); the Jacobian is not part of the correct form (density form, F3).
- **Two structural rulings-in-waiting produced by the wave:** (i) α is the selection-prior's
  normalization — D1+D4 were one broken pairing; (ii) Part 1 F1's "impostors normalize out" holds
  only for the overall constant — the LOO weight is a real likelihood term (verifier C1,
  −135 ± 4 nats/h).
- Low-dose residual of FULL-F: +168.9 ± 58.8 (f_i = 0.25) — stated residual; candidate owners:
  pool-vs-model prior mismatch (KS D = 0.085 vs 0.043 critical; prior-score ≈ −152) + LOO-model
  error + drift remainder.

## 1. Next tasks

1. **Author adjudication of the A-FULL draft §6 (as amended):** adopt the FULL-F definition
   [RULE]; register + run A-FULL on the cluster [DO] (~25 CPU-h, N = 25, A8-v2, fresh seeds,
   pre-registration xhigh verifier — the M2P/stage-3 precedent is mandatory); band-seeding
   question [RULE]. **Registration must restate the row-#109-item-3 deviation** (no Jacobian, no
   renorm, + S̄_φ + LOO) — flagged in the addendum.
2. If registered and run: the arm's readout decides whether the venue mechanism thread CLOSES
   (P1–P5 falsifiers pre-stated in the draft §4 as re-seeded by the addendum).
3. **Then the production question opens:** the `/physics-change` proposal for
   `bayesian_statistics.py` (the same broken pairing exists in production — that is the H0-bias
   mechanism at the paper level). Full 5-step gate; the venue arm is its evidence base.
4. **Gray-convention paper-scope ruling** — re-presented (deferral lapsed at row #106 item 3):
   published practice carries neither the Jacobian nor the pairing question at σ_z > 0; the
   FULL-B/D/F results now give that finding teeth (a quantified, mechanism-level account of what
   the convention costs). [RULE], timing + scope.
5. Book ch14 (carried): now also owed the Part-2 → drift-eval → A-FULL arc.
6. 2D +129 excess: unpriced; natural home is the registered A-FULL arm's 2D channel readout.

## 2. Standing constraints (unchanged)

Append-only registered docs; `/physics-change` slot EMPTY until item 3 above is granted; bands
never toy-calibrated; A8-v2 on registration; top-tier cap ≤3 inherit agents/workflow (this
session used 2 inherit verifiers + 2 sonnet implementers); branch calls presented, never
self-adjudicated.

## 3. Operational notes

- All recomputes ran locally on committed data (mirror geometry); cluster untouched; workspace
  39 days at 2026-08-14 preflight.
- Scripts in `results/mechanism_study_20260813/` must run from the **repo root** (pin paths are
  cwd-relative). `l4_afull_premeasure.py --stage {abc,de,f}` reproduces each committed output.
- Author WIP (book files ×3) + the 08-15 pre-rebase stash: still awaiting author confirmation.

## 4. Resume recipe

1. `git log --oneline -3` — expect the A-FULL addendum commit at HEAD or a descendant.
2. Read the draft + addendum. 3. If the author has ruled on §6: registration flow (A8-v2, fresh
seeds, xhigh prereg verifier) then `/cluster`. 4. If not: the §6 decision table is the ask.

---

## Addendum (2026-08-15, same session) — REGISTERED AND SUBMITTED

Row #110 granted §1's items ("all approved"): FULL-F definition ratified, registration + run
authorized, bands seeded from the mirror, Gray-convention **in paper scope** (branch reading
flagged for veto in row #110 item 4). Executed:

- **Registered** at `dec62032`: `PREREGISTRATION_A_FULL_STAGE5.md` (DS-F1 band [−131.5, +192.7],
  0.27% false-fail; binomial coverage bands; branch table), `estimator_variant="a_full"` installed
  (bit-exact vs the premeasure mirror: diff 0.0, T = +29.998 vs pin +30.0), `score_stage5.py`
  (AJREN dry-run reproduces all references), seed block +54200…+54224 (disjointness unit-tested),
  `cluster/stage5_afull.sbatch`. Pre-registration xhigh verifier: **GO, zero CRITICAL/MAJOR**
  (floor probe clean at grid extremes; MINOR-1 remedied in §8 of the prereg).
- **Submitted:** job **6327889** (cpu_il, 15 workers, 05:00:00 wall), preflight READY ✓,
  probe-vs-actual logged per EXP-61. Output lands at
  `results/mechanism_study_20260813/AFULL_h0p730_results_seeds0_25.json` on the cluster.
- **Next session (or later this one): retrieve + score + present.** Recipe:
  `ssh bwunicluster 'sacct -j 6327889 --format=JobID,State,Elapsed,ExitCode'`; on COMPLETED 0:0,
  `rsync -avz bwunicluster:darksiren-emri/results/mechanism_study_20260813/AFULL_h0p730_results_seeds0_25.json results/mechanism_study_20260813/`,
  then from repo root `uv run python results/mechanism_study_20260813/score_stage5.py`, commit the
  JSON + scorer output, write `STAGE5_READOUT.md` (PRESENTED, NOT ADJUDICATED, §5 branch table),
  present to the author. Venv note: the dev-box venv was repaired this session
  (`uv sync --extra cpu --extra dev --reinstall`) — mypy hook works again.
