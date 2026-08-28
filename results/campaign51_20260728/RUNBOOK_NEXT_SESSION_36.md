# Runbook — next session (2026-08-28 close, supersedes runbook 35; [FABLE-ORCH 2026-08-28])

**Read first:** this session (a) adversarially verified the 2026-08-27 Opus session's work
(6-agent workflow — no blockers, 1 MATERIAL σ-chain finding, 2 MINOR citation slips), then
(b) read out BOTH completed cluster measurements: the **[P3-2D] repair fleet** (job 6723958) and
the **production H₀ HEAD readout** (jobs 6724169/70 + 6725283/84). Both readouts are chair
verdicts, **not author-ratified**. Entry points:
`realistic_20260729/P3_2D_REPAIR_READOUT_20260828.md` and
`realistic_20260729/MEASUREMENT_HEAD_READOUT_20260827.md` (post-data appendix).

## 0. State at close

- **[P3-2D] repair run READ OUT.** All gates G1–G6 PASS (G4 arm ratio 0.865491 in-interval;
  G5 mass-window fraction exactly 0.0 → ×1.1944 transfers; G6 exact identity 48/48). P1 inside
  +0.247 σ, P4 inside +0.351 σ. P2/P3 bank **UNDERPOWERED** by the freeze rule (realized SEMs
  16.7%/4.4% above planning) despite central values at +0.10 σ/+0.98 σ; P3 excludes "Defect 2
  spurious" at 5.77 σ (REPORTED-ONLY); the pre-registered 1.0680-vs-1.1019 non-discrimination
  stands. **Per §v2.5 CONFIRMED cannot bank; not REFUTED.** PARKED row #211 untouched.
  Verification found the v2.2 σ-chain was rounded-roundtrip-derived (true chain
  2.3194/2.5449/2.6560%); dispositions identical under both chains (PA-2DR-14).
- **Production H₀ HEAD readout READ OUT — the 2D bias GREW.**
  - 2D: iiib −0.066653 (was −0.052933), joint_r1 −0.066987 (was −0.051213) →
    **MATERIALLY GROWN on both venues** (Δ = −0.0137/−0.0158, both > T_mat 0.008; ΔMAP −0.010/
    −0.015 both band-bearing). Pull ~+3.6 both venues (was ~+2.25). C68=C90=0 (N=1 indicator).
  - 1D: **RAIL LOOSENED both venues** (MAP still 0.600; mean_h 0.605309/0.611683 crossed the
    0.605 rail statistic; Δoffset +0.0043/+0.0097, neither band-bearing at 0.010). §5.3's
    PERSISTS/LOOSENED conditions overlap — flagged as a map wording gap, LOOSENED read as the
    more specific row, needs author ratification.
  - **Registered blindness holds: NO per-change attribution.** 4-term composition (3 code
    changes ⊕ off→fused config); the §8.5 `off` arm was NOT run and is the only registered
    split instrument. Correctness-over-bias-removal: a grown bias is not evidence any adoption
    was wrong.
  - **RECORD GAP:** the registration says NOT SUBMITTED with §10 author decisions pending, yet
    the jobs ran on 2026-08-27 evening. No recorded author ruling; no shell-history trace. All
    post-hoc-checkable gates pass (config stamps, physics lines, zero COUNTERFACTUAL, §8.7
    items 1–7 incl. scorer↔combine agreement). Flagged as author item A-1.
- **Verification of the Opus session (workflow `verify-opus-p32d`, 6× sonnet):** commit
  `3694233d` clean (exactly reject-M≤0 + floor delete; production `--evaluate` untouched);
  gates non-vacuous and passing at HEAD; submission record PA-2DR-12/13 exact (suite reproduces
  1831/15 exactly); row #132 comparand table reproduces to ≤5e-5; residual-ladder arithmetic
  correct. Defects: the MATERIAL σ-chain item (above), `R_LOW_THRESHOLD` cited :359 is at :360,
  §1a "verbatim" quote has an inserted `RHS2/LHS2 =` clause.
- **Ops:** preflight READY ✓ (WARN: unregistered datasets — the three new dirs are now
  registered in `cluster/datasets.yaml` + `DATA_INVENTORY.md`, lines 168–186 / 276–278).
  Workspace `emri` expires **2026-09-23, 0 extensions** (~26 d). Headreadout diagnostics +
  posteriors retrieved locally to `realistic_20260729/headreadout_20260827/` (~8 GB).
  Queue empty. No source file changed this session.

## 1. OPEN AUTHOR DECISIONS (this session's, all fresh)

From `P3_2D_REPAIR_READOUT_20260828.md` §4:

- **[RULE] R-2DR-1** ratify the corrected σ-chain (PA-2DR-14; no disposition changes).
- **[RULE] R-2DR-2** verdict of record for the repair run (chair recommends: UNDERPOWERED per
  the map's letter, central values as reported companions).
- **[DO] D-2DR-1** optional seed-extension arm (24→33 seeds, ~2.2 CPU-h/seed) to recover the
  SEM deficit; bands stay frozen.
- **[RULE] R-2DR-3** confirm row #211 stays PARKED.

From `MEASUREMENT_HEAD_READOUT_20260827.md` §D:

- **[RULE] A-1** rule on the submission-record gap (retroactive ratification vs remediation).
- **[RULE] A-2** ratify MATERIALLY GROWN on both venues.
- **[RULE] A-3** ratify the RAIL LOOSENED reading + the §5.3 wording-gap resolution.
- **[DO] A-4** authorize/decline the §8.5 `off` companion arm (~105–265 CPU-h) — now the only
  way to split code-changes from the off→fused delta, and materially motivated by A-2.
- **[DO] A-5** archival (§10 item 6): observed catalogue (2.5 GB, sole copy) +
  postfix_baseline pair before 2026-09-23.

Carried from runbook 35 §4.3, still pending: the thirteen [WGEO]/[HIER]/[MKER] one-liners
(R-MKER-1..6, D-MKER-2/3, R-WGEO-1..3, D-WGEO-1, the nine [HIER] items) — unchanged.

## 2. Standing rules & session-earned ops (delta over runbook 35)

- **The freeze rule fired for real this time** — two reads with central values well inside
  their bands banked UNDERPOWERED because realized SEMs exceeded planning by 16.7%/4.4%. The
  discipline held (no post-hoc widening); the lesson for the NEXT prereg is to derive planning
  σ_new from a measured scatter anchor rather than "set equal to σ_pred", which was optimistic
  for a filtered sub-statistic (P2's D1only filter throws away rows, raising SEM).
- **σ-provenance rule:** never back-derive relative SEMs from rounded X ± σ_X roundtrips when
  the raw σ/value pair is in the cited table — the v1 verifier caught a 0.2–0.8% inflation
  that happened to be harmless here (PA-2DR-14).
- **The tier-lint 500-char scan gap is real** (runbook 35 warned; it fired here) — hoist long
  agent prompts into named constants above the `agent()` call.
- **Registered-map conditions must partition** — §5.3's PERSISTS/LOOSENED overlap forced a
  chair interpretation. Pre-launch checklist: verify band/verdict conditions are mutually
  exclusive and exhaustive.
- **Submission actions must stamp their authorization** — the A-1 gap would be a non-issue if
  the submitting session had appended one line ("submitted under author ruling of <date>") to
  the registration. Make that a submission-record requirement alongside job id + out-root.

## 3. Resume recipe (one line)

Author rules on §1 (nine new one-liners + thirteen carried) → on A-4 approval, submit the `off`
arm (STEP 1–4 pattern of the registration, out-roots `run_20260827_headreadout_off_*`); on
D-2DR-1 approval, extend the repair fleet seeds → archive-or-migrate before 2026-09-23 (A-5,
MUST-ARCHIVE items in `cluster/WORKSPACE_ARCHIVAL_TRIAGE_20260827.md`) → then the parked
threads per runbook 35 §4.5 ([HIER] venue one-liner, D-WGEO-1 records read, [P3-2D] S̄_φ fix).

## 4. RULING ADDENDUM [2026-08-28, same day]

Author reply to the docket (verbatim): **"all ratified also the thirteen earlier ones"** —
recorded as ledger rows #212–#214 with orchestrator-derived itemization. Executed same day:

- **D-2DR-1**: extension job **6730213** (seeds 900125–900133, PA-2DR-15) — on completion,
  re-run `stage_lhs2d --seeds <all 33>` both arms, evaluate the SAME frozen bands, report
  24-seed and 33-seed values side by side. No further extension without fresh registration.
- **A-4**: off-arm smokes **6730223/6730224**; on STEP-2 gate pass (run_metadata_21 shows
  `selection_in_completion_numerator=off`, `catalogue_global_selection=phi`, correct venue
  catalogue; the log must show the phi line and NO fused line), submit the full
  `--array=0-40` pair + combine, then score with the §1.3 scorer and compute the 2-way split.
- **A-5**: archival rsync to `results/_archive/` running (observed catalogue sha256-verified on
  arrival); the wider ~400 GB triage program is still open and NOT covered.
- **D-MKER-3**: GitHub issue **#57** filed.
- **D-WGEO-1**: records read launched (agent); result appends to CLAIM_WGEO_20260827.md.
- **D-MKER-2**: window-geometry prereg authoring queued AFTER D-WGEO-1 returns (runbook 35
  §4.5 sequencing), at top-tier/xhigh per the tiering table.
- **[HIER]**: S0-A unblocked (venue b0i); θ-hook design may be drafted against option B +
  H_GRID_41 but NO trigger-file edit until item 3 resolves.

**STILL OPEN — six one-word asks the blanket could not resolve** (no-default forks):
R-MKER-5 (reduce / close) · R-MKER-6 (open / don't) · HIER-3 (gate / no gate) ·
HIER-4 (gate / disclose) · HIER-5 (build / fallback) · HIER-9 (hard / affordable).
Plus veto window on two orchestrator-interpreted assignments: HIER-7 read as RE-ANCHOR,
R-MKER-3 read as ratified-in-R2-form.
