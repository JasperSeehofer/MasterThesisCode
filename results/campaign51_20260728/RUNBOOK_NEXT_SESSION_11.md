# Runbook — next session (written 2026-08-15, supersedes RUNBOOK_NEXT_SESSION_10 + the 08-14 handoff)

**Read first:** `results/mechanism_study_20260813/L4_T2_AUDIT_20260815.md` (`febc709f`) — the
stage-4 audit that Part 2 must answer — then `L4_DER_PART1_20260815.md`, `L4_T1_TILT_COMPOSITION_20260815.md`,
`STAGE3_READOUT.md`, `L0_SYNTHESIS_STAGE3_20260815.md`. Ledger rows #102–#108 carry every ruling.

## 0. State of the physics (post rows #102–#108)

- **Displacement law CONFIRMED, parameter-free:** bias = T/Ā at 1.15 ± 0.13 across 16 distinct
  cells and **0.989 out-of-sample on A-JREN**. Posterior width is correctly calibrated to local
  curvature; the old "8.5× overconfidence" is displacement over correct width — **the single
  unexplained object in the entire defect is the tilt T at truth.**
- **Measured single-term ledger (all instrument, all ratified):** +0.037250 = J −0.01805
  (A-M2′, on-prediction incl. the ln D′ term) + REN −0.0019 (confirmed via additivity in A-JREN)
  + **+0.0178 remaining**. Repairs are additive at 1D (0.6σ); coverage NOT restored;
  bias/post_sd 8.49 → 3.00. A-REN withdrawn (row #107). 2D-only sub-additivity +0.0027 (≈3.8σ);
  the 2D tilt excess (~+130 nats/h) predates REN — base-estimator property.
- **Measured tilts:** T(MN0X) = +2624.9 ± 18.8 · T(AM2P) = +1492.0 ± 30.7 · T(AJREN) =
  **+514.5 ± 16.8** (1D; 2D +643.2). Remaining tilt ≠ α (only ~37% of it). Instrument REN tilt
  −977 ± 35 — **opposite sign, 10× the L0 toy** (third toy-transfer failure; toys are for shape
  intuition only, never magnitudes — rows #102, REN-toy caveat, and now this).
- **Stage-4 Part 1 enumerated the coded-vs-correct diff (D1–D6)** from the generator: (F1)
  impostor factors h-independent given d_obs; (F2) **no α term belongs in the pinned-event venue**;
  (F3) the GW term is a density in d_obs (missing h/D prefactor; O(σ_d) exponent-scale asymmetry
  D3). **The L4-T2 numeric audit REFUTES the ledger as composed:** all predictions negative vs all
  measurements positive (pulls −69σ…−211σ); D2's naive −N/h is ~90% cancelled by z*-tracking
  (→ −134); D3 measured −541 (exact-z) with a σ_d-tail dominance warning (top 1% of events = 55%
  of D3 — quadrature trap documented in the audit); dose kill test fails at all levels; the
  D3-sourced 2D estimate is ~200× too small.

## 1. The open question Part 2 must answer (the next orchestrator task — derivation, top-tier)

**What produces the measured +2625 up-tilt?** The D-enumeration, composed as (coded − correct)
with correct-tilt ≡ 0, misses the dominant positive source. Leads, in order of suspicion:
(a) a **sign/composition error** in how D-terms stack with the installed J/REN modifications
(J's −N/h piece is NOT part of the correct d_obs-density form — Part 1 F3 — so A-M2′'s empirical
success needs re-derivation in the correct frame); (b) the **correct-form zero-tilt assumption**
(is E[score at truth] = 0 actually guaranteed given the pinned d_true were realized at h = 0.730
and the estimator's p_pop/impostor handling? — derive, don't assume); (c) a **missing positive
term outside D1–D6** (the impostor-sum normalization 1/K vs the correct mixture weights P(k=host)
— the coded flat 1/K mean over candidates vs the correct posterior host weights is D6-adjacent and
never got its own tilt number); (d) D3's σ_d-tail — recompute with the audit's adaptive quadrature
at the per-event level and check whether the tail's sign flips under the smeared kernel.
**Protocol:** Part 2 is a derivation + targeted recomputes; xhigh verifier before it returns to
the author; the A-FULL draft waits for a closed ledger (row #108 item 2 authorizes drafting — do
not register).

## 2. Standing constraints (unchanged)

Registered docs append-only; no repair from a partial read; `/physics-change` slot EMPTY
(author-gated; proposal §3 item 4 rules the timing); bands never toy-calibrated; branch calls
presented, never self-adjudicated; A8-v2 discipline (incl. execution-completeness) on any new
registration; top-tier cap ≤3 inherit agents/workflow (CLAUDE.md; tier-lint hook live).

## 3. Operational notes

- Cluster: nothing running; workspace 39 days at last preflight (2026-08-14). Cost anchors:
  0.969 CPU-h/seed reserved-core definition (recompute note in CLOSURES_D113_D117); AM2P realized
  1.34 h wall, AJREN 1.49 h at 15 workers.
- **Concurrent-commit race gotcha (2026-08-15):** `git commit` after `git add <file>` commits ALL
  staged files — a parallel agent's staged files were swept into `5e77e196` (content correct,
  history left alone). In shared-tree multi-agent flows, commit with explicit pathspecs
  (`git commit -- <paths>`) or serialize commits through one agent.
- Author WIP: 3 files carry real local edits (book/design/BOOK_DESIGN.md, BOOK_TECH_DESIGN.md,
  book/generators/gen_ch00.py); a pre-rebase stash snapshot from the 08-15 race is retained —
  drop it only after the author confirms the book files.
- Book ch14 still NOT DONE (carried from runbook 10 §3.1); it must now also cover rows #102–#108
  and the commission arc; sources are ready-made (the readouts + syntheses).
- Paper leads on hold per row #106 item 3 (Gray-convention finding deferred until after… now
  A-JREN has run — the deferral condition has lapsed; re-present to the author).

## 4. Resume recipe

1. `git log --oneline -5` — expect `febc709f` at HEAD or a descendant.
2. Read the §0 documents. 3. Part 2 derivation (§1) with an xhigh verifier. 4. Present the closed
(or honestly re-broken) ledger + the A-FULL/gate decision table to the author.
