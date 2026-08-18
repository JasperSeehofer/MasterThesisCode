# Runbook — next session (written 2026-08-18, supersedes RUNBOOK_NEXT_SESSION_18)

**Read first:** `results/pp_coverage_prodcal_20260817/CAMPAIGN_REPORT_20260818.md` (the A7
readout) + the prereg's VERDICT/VERDICT-2/VERDICT-3 chain in the same directory, then ledger
rows #120–#124.

## 0. State (end of the 2026-08-18 session)

- **The prodcal calibration cycle is CLOSED (rows #120–#124).** The #66/#67 disappointment path
  closed BENIGN: three mutually consistent measurements (production counterfactual, V-flat,
  V-prod paired deltas) certify the landed fusion lever benign in production's regime. The
  asymmetric-[P2] validity boundary is measured and mechanism-owned (first-order tilt, 3%
  agreement): safe at flat S̄, −0.03-class where S̄ has a strong gradient.
- **Products:** [A3] two-channel `pp_coverage` harness (mass channel, production-N, 3-way noise
  toggle, catalogue balls) + 28 tests; the [C-SYM] claim
  (`CLAIM_SYMMETRIC_SELECTION_INSERTION_20260818.md`, admitted row #122 item 5 — both-legs is
  the correct form; production's one-leg form is a measured-safe regime approximation); the
  Q-0 (UNPAIRED)/Q-1 (SAME-OBJECT)/Q-2 (P3 symmetric) audits; literature-register updates;
  7 verifier passes (addendum Parts I–VII); ledger rows #120–#124.
- **Ch15 shipped** ("The Slot Gets Filled", `a73b46eb`). Gray-convention paper proposal
  committed with §6 grounding-measurement list; **G-3 done (A.10 MATCH, arXiv v4)**.
- Cluster used for the main ladder (job 6355028, DEVIATION-1); harness runs fine locally for
  small cells. Operational gotchas learned: plain background shells get reaped (use
  `nohup setsid … & disown` + Monitor); frozen-seed serial streams make cells irreducibly
  serial — future harness designs should spawn per-realization seeds (SeedSequence) for
  parallel grain (aligns with the realistic-venue performance goal).

## 1. Next fronts (the record's order)

1. **[C-SYM]/[P3] correct-form + Gray-paper front** — the next research cycle (/research-cycle
   stage 0 from the [C-SYM] claim + proposal §5 items). Grounding measurements to pre-register
   (each its own [DO]): **G-1** catalogue-leg fusion counterfactual (harness-scale first,
   ~1 CPU-h, decides whether the ~170 CPU-h production protocol is warranted); **G-2**
   spec-z-kernel σ_z→0 cell. Then the author's [P3] presentation pick (a)/(b) with measurements
   in hand → paper integration (figures/text, §5 items 4–5).
2. **Paper [DO]s already granted** (row #121 item 4): TO-MAKE figures + discussion.tex:235
   rewrite — sequenced after G-1/G-2.
3. **Seeds (not opened):** flattened-detection venue-physics (raised-d50 bias class,
   VERDICT-3); 2D-channel noise-coupling (+0.01, off-cells, audit §C); estimator-side-only S̄
   instrument knob (would serve both seeds AND [C-SYM]'s refute-by).
4. Book ch16 candidate: the prodcal cycle arc (this campaign is unusually teachable: nulls
   firing correctly, verifier catching the record-contradicting power claim, the boundary
   discovery).

## 2. Standing constraints

Append-only ledger/prereg discipline; A8-v2 + verifier pre-checks for every amendment (worked:
7 passes caught 12+ defects pre-run); top-tier agent cap; branch calls presented not
adjudicated; venue-scoping rule binds in both directions. Workspace expires 2026-09-23
(0 extensions). Author WIP in book/design/* + gen_ch00.py remains uncommitted — do not touch.

## 3. Resume recipe

1. `git log --oneline -5` — expect the VERDICT-3/row-#124 chain at HEAD.
2. Read the A7 report + ledger #120–#124. 3. Open front 1 via /research-cycle (now
   model-invocable); G-1/G-2 preregs return to the author before any run.
