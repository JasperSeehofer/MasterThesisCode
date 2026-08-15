# Runbook — next session (written 2026-08-15, supersedes RUNBOOK_NEXT_SESSION_11)

**Read first:** `results/mechanism_study_20260813/L4_DER_PART2_20260815.md` **including its
verifier addendum** (the addendum supersedes the body where they conflict), then the erratum
addendum at the foot of `L4_T2_AUDIT_20260815.md`. Ledger rows #102–#108 carry the prior rulings;
Part 2's decision table (§4, as amended by A1/A2) is **in front of the author, unadjudicated**.

## 0. State of the physics (post Part 2, commit `9f2e6c1a` + addenda)

- **The runbook-11 open question is ANSWERED (pending ratification):** the measured +2625 up-tilt
  is α (+1400.6 numeric) + **GW z-mass growth (+1059.6)** + exponent-scale (+175.8) + window
  motion (−31.1) + leftover drift+interactions (+39.1 at full dose). The mass-growth term
  G_e = (1/h)(1 − D·D″/D′²) — the h-growth of the ratio-form GW factor's z-space volume in the
  photo-z-starved (kernel-integrated) regime — is the positive source no D1–D6 row priced.
  Closed-form identity: ΣG = N/h − Σx/h = +1345.2 − 285.7 = +1059.5 ≡ minus the M6R "−N/h +
  tracking" J prediction, term for term — A-M2′ worked because the Jacobian makes the GW z-mass
  h-independent (its measured effect is 98.7% mass-kill, per-node reshaping only −13.7).
- **T_res is (hedged) located:** the dose-decaying leftover (+867/+344/+39 at f_i = 0.25/0.5/1.0)
  — drift + interactions; the §2 drift formula's **direct per-event evaluation is the registered
  next targeted recompute** before the identification hardens (verifier A2).
- **Method:** exact single-switch A/B tilts on a mirror validated **bit-exactly** (0.0) against
  the committed MN0X per-seed vectors; mirror totals match measured S31/S32/S33 out-of-sample at
  0.3–1.3σ; predicted T(AM2P) within 1.4σ of the measured arm.
- **Verifier corrections of record:** the T2 audit's D3 was printed in the opposite sign frame to
  its own convention (erratum appended); D3 exact-weighted is −176 vs −342 isolated — same sign,
  ~2× attenuated, NOT a sign flip. M7/window motion −31 nats/h = 1D local tilt at truth only.
  Lead (b)'s cancellation-locus narrative is conjecture (belongs to the A-FULL derivation).
- Unpriced/carried: instrument REN tilt −978 (installed-repair property; derive within A-FULL);
  the 2D-only +129 channel excess; the α 0.5% numeric/analytic tension.

## 1. Next tasks (in order)

1. **Author adjudication** of Part 2 §4 (as amended): decisions 1–4. Do not proceed to A-FULL
   registration on any blanket approval — decision 3 authorizes a DRAFT only.
2. **Drift-term direct evaluation** (targeted recompute, CPU-cheap, orchestrator): evaluate
   Σ_k r_k (z_obs,k − z*)/(σ_k² + σ_gw²) · dz*/dh per event per seed (window clipping included)
   and compare against the +867/+344/+39 leftover — hardens or breaks "T_res ≡ drift".
3. **A-FULL draft** (row #108 item 2 authorizes drafting): the correct d_obs-density estimator
   (density prefactor + Jacobian measure + p_pop numerator + renormalized kernel), predicted
   tilt ≈ 0, with the REN −978 derivation and the 2D +129 excess addressed inside it. Reviewable
   artifact + fresh xhigh verifier; registration is a separate author gate (A8-v2).
4. **Paper lead, re-present (deferral lapsed, runbook 11 §3):** the Gray-convention finding
   (L0 synthesis §1 item 6) — whether it enters the paper's scope now that A-JREN has run. [RULE]
5. **Book ch14** (carried from runbooks 10–11): must now also cover rows #102–#108, the
   commission arc, and the Part-1→Part-2 composition-failure story (a worked example of why
   isolated-term ledgers fail inside log-integrals).

## 2. Standing constraints (unchanged)

Registered docs append-only (Part 2 body unchanged; amendments live in its addendum); no repair
from a partial read; `/physics-change` slot EMPTY; bands never toy-calibrated; branch calls
presented, never self-adjudicated; A8-v2 on any new registration; top-tier cap ≤3 inherit
agents/workflow. Session tiering used: orchestrator-inline derivation + 1 inherit/xhigh verifier.

## 3. Operational notes

- Cluster: nothing running; nothing submitted this session (all Part-2 recomputes ran locally,
  ~45 core-min total). Workspace 39 days at last preflight (2026-08-14).
- Concurrent-commit race gotcha carried: commit with explicit pathspecs in shared-tree flows.
- Author WIP: 3 files still carry real local edits (book/design/BOOK_DESIGN.md,
  BOOK_TECH_DESIGN.md, book/generators/gen_ch00.py); the 08-15 pre-rebase stash snapshot is
  retained — drop only after the author confirms the book files.

## 4. Resume recipe

1. `git log --oneline -3` — expect the Part-2 addenda commit at HEAD or a descendant.
2. Read §0's two documents (Part 2 + addendum; T2 erratum). 3. If the author has ruled: apply the
rulings to the ledger and proceed down §1. 4. If not: the decision table is the ask — do not
start A-FULL drafting past what row #108 item 2 already covers.
