# Runbook — next session (2026-08-26 evening close, supersedes runbook 34; [OPUS-ORCH 2026-08-26])

**Read first:** this session ran two threads end-to-end: **[P3-MKER]** two rounds (R1: the
card's headline exhibit mis-attributed ~300x, refuted on all three factual assertions; R2: the
decisive catalogue read, exhibit retired, two new findings opened) — banked append-only in
`realistic_20260729/CLAIM_P3_MKER_20260826.md` (§R1 RESULTS, §R2 RESULTS, ~884 lines). And
**[HIER]**: pre-registration authored + adversarially reviewed —
`realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md` (~1357 lines, 18 append-only
amendments PA-HIER-1..18) — verdict **LAUNCH-BLOCKED**, six blockers, seven-step zero-compute
path back to LAUNCH-READY banked. [P3-2D] is untouched this session, still PARKED per runbook 34.

## 0. State at close

- **[P3-MKER] R1+R2 executed, exhibit RETIRED, thread re-opened on window geometry.**
  - R1: the headline exhibit (seed 900121 event 20, −176.6 nats) is MIS-ATTRIBUTED by ~300x —
    the mass KERNEL carries only −0.5838 (bc) / −0.6113 (bt) nats; the mass ELIGIBILITY WINDOW
    carries 99.67%, by discarding candidate 6791151. Claim §1(a) REFUTED on all three factual
    assertions: the catalogue width IS convolved (`bayesian_statistics.py:6607,:6613`), it
    DOMINATES sigma_cond by 2.1e8–3.7e8, and the 0.24-dex R&V15 intrinsic scatter IS present
    (`handler.py:44`). Fleet-wide over 2,122,481 window-passed candidates the max mass-axis
    pull bound is 6.482 — "k ~ O(10)" occurs nowhere. Refute-by(a) FAILS (route closed): in
    `generator_marginal` the divisor `n_hat_w = W_cat/V_f(h)` (`:5069-5092`) has no mass term
    and is identical in both channels; `D_g` is "diagnostic only" (`:6199-6200`); `D_gen`/`B_num`
    shared (`:5544-5545`) — kernel AND window sit in the numerator with NO mass-side
    renormalization. Part (b) GAINED: window uses the EVENT's (1+z_max)/(1+z_min)
    (`handler.py:664-672`) while the kernel uses the CANDIDATE's own (1+z) (`:6606-6607`), and
    the same `host_M_error` feeds both — a coupling no prior document records. Census IS
    zero-compute available (contra read-i): ln(num_w/num_no) median −2.9352, p10 −16.4676, k_ub
    median 2.423 / p90 5.739 / max 6.482, 23.73% below −4.5 nats.
  - R2 (dataset pin discharged — md5 `c52c13b5cab61f6b3f04bbe202550969`, 1,681,954,844 bytes,
    cluster copy BYTE-IDENTICAL): candidate 6791151 is NOT readmitted under the full 0.55-dex
    budget. Required margin x2.3150943 on `host_M_error` (291,758.995 → 675,449.592 Msun); the
    0.50-dex measurement component supplies only 25.4% of the gap. Its central M_BH is 5.526x
    below the GW-required floor — a genuine mass mismatch. EXHIBIT RETIRED as evidence. Measured
    twice independently (production-loader route + direct chunked re-implementation) plus a
    third adjudicating derivation; all agree to full float precision; cross-checks on
    6791138/6791158 reproduce exactly.
  - TWO NEW FINDINGS from R2: (i) the true injected host 6791134 is OUTSIDE the 1.5-sigma sky
    cone (chord 1.674660e-03 vs radius 1.4956980e-03, x1.1196) — the exhibit's candidate list
    contains no true host; (ii) **THE WINDOW IS LINEAR-SYMMETRIC AGAINST A LOG-NORMAL ERROR
    MODEL** (sigma_ln = 1.3032; lower edge goes NEGATIVE at −213,766 Msun; upper edge reaches
    2.955x vs a log-space 7.06x). In ln-space 6791151 sits at 1.3117 sigma — INSIDE 1.5 sigma.
    **The live question is now the window's GEOMETRY, not its width** (noted: a log-space
    window would readmit an interloper, not the true host).
  - Part (a) is DE-FUSED: the 0.50-dex omission survives only as a documentation/modelling
    question with no demonstrated consequence.
  - Successor question for thread resume: pre-register the window-GEOMETRY measurement
    (linear-symmetric vs log-space) — see D-MKER-2 below.
- **[HIER] pre-registered then LAUNCH-BLOCKED at review.** Six blockers, two structural:
  - PA-HIER-1 (BLOCKER): `host_mode` is unregistered and decides truth-theta. Under the default
    `host_mode="catalogue"` the generator's z_true law is a DELTA at the catalogue z
    (`correspondence_1d.py:1779-1783,:2141`) while the estimator integrates
    N(z; z_g, sigma_z_eff) — so truth-theta on the s axis is s→0, NOT s=1. The design's central
    premise fails.
  - PA-HIER-4 (BLOCKER, FIXED): `score_s` is mis-formed — nodes log-symmetric, denominator
    linear, so the secant estimates the derivative at s=1.0606602 not at truth; spurious |Z_s|
    grows as sqrt(n). Same defect class as PA-2D-8. Replacement registered: `score_lns`.
  - Other blockers: PA-HIER-2 (the registered hook is generator-side, would make theta move the
    DATA); PA-HIER-3 (the S0-R positive control injects no misspecification — a null instrument,
    D7's early exit unarmed); PA-HIER-6 (no theta prior/marginalization measure registered
    though three verdict families depend on it); PA-HIER-7 (the identifiability statistic never
    says fixed-h vs profiled vs marginalized).
  - The reviewer enumerated a seven-step ALL-ZERO-COMPUTE path back to LAUNCH-READY. Estimated
    ~424 CPU-h of unfalsifiable running was prevented.
  - Costing/recon artifacts: `hier_instrument_recon_20260826.md`, `hier_costing_20260826.md`,
    `hier_provenance_stamps_20260826.md`. Recon found SIX sigma_z dispatch sites (not two), two
    of which (`validation/correspondence_1d.py`, `validation/pp_coverage.py`) are INDEPENDENT
    reimplementations unreachable by any hook placed inside `bayesian_statistics.py`.
- **[P3-2D] still PARKED at UNATTRIBUTED-bounded** (row #211, unchanged from runbook 34). Entry
  point remains `realistic_20260729/STUCK_P3_2D_SYMPTOM_CARD_20260826.md` (independence-clean —
  hand a searcher the CARD ONLY). Exoneration record: rows #207–#210 + PA-2D-9/10 (do NOT
  re-open: C₂\*, the completion-mass axis, the replay machinery). First action on thread resume:
  the GRANTED-but-UNRUN class-G S̄_φ de-double-weight fix + fleet re-run (~2–4 CPU-h).
- **Ops:** cluster reachable, no jobs. Workspace `emri` expires **2026-09-23 with ZERO
  extensions available** (28 days as of 2026-08-26) — hard deadline to archive or migrate. An
  ssh keepalive ran through the session. No source file under `darksiren_emri/` was touched this
  session (verified via `git status --porcelain`) — this was a read-only-on-source, docs/results
  session.

## 1. OPEN AUTHOR DECISIONS

- **[RULE] R-MKER-1** ratify AMENDMENT A-MKER-1 (§1(a) refuted as written, stands only in
  amended form).
- **[RULE] R-MKER-2** ratify the SPLIT and the closure of the Refute-by(a) demotion route.
- **[RULE] R-MKER-3** rule on corrected sequencing (card's "kernel first, window second" is
  backwards) — MUST BE RE-STATED first: R2 removed the sigma-decision from its critical path
  and replaced the window epsilon-question's content.
- **[RULE] R-MKER-4** ratify the R2 NO verdict + exhibit retirement.
- **[RULE] R-MKER-5** rule on part (a)'s standing after de-fusion (reduced priority vs close as
  documented design choice).
- **[RULE] R-MKER-6** rule on whether the true-host-outside-cone finding opens a host-recovery
  thread.
- **[DO] D-MKER-2** authorize PRE-REGISTRATION ONLY of the window-GEOMETRY measurement
  (linear-symmetric vs log-space window).
- **[DO] D-MKER-3** file the `get_redshift_outer_bounds` dead-parameter defect as a GitHub issue
  (`sigma_multiplier` is dead code — body hardcodes "3 *"; `physical_relations.py:563,566`).
- **[RULE] R-HIER-1** rule on the six [HIER] blockers, several of which imply NEW CODE near
  physics-trigger files, re-opening the `/physics-change` scope question the prereg's §1.5
  currently closes.

## 2. Standing rules & session-earned ops (delta over runbook 34)

- **The review chain paid for the third time**: PA-HIER-4 is the PA-2D-8 defect class
  (log-symmetric nodes vs linear denominator in a secant/score construction) caught again
  pre-launch. Three occurrences now in this campaign — treat "check node symmetry against the
  denominator's own space" as a standing pre-launch checklist item, not a one-off catch.
- **A claim card's own exhibit can be mis-attributed by orders of magnitude** — R1 found the
  card's headline −176.6 nats number was ~300x misattributed to the wrong mechanism
  (eligibility window, not kernel). Lesson: decompose the headline number into its named
  mechanisms BEFORE building a thread on it, not after.
- **The two-independent-measurers pattern worked again**: R2's decisive readmission-margin
  number was produced by two structurally different routes (production-loader route + direct
  chunked re-implementation) plus a third adjudicating derivation, with no decisive
  disagreement across any of the three. Reusable for any decisive catalogue-read verdict.
- **The dataset-pin discharge converted an author-gated risk into a mechanical check**:
  checksumming the local copy against the cluster copy of record (md5
  `c52c13b5cab61f6b3f04bbe202550969`, byte-identical) BEFORE the multi-GB catalogue read is now
  demonstrated as routine, not exceptional — do this first on every large-file read, per the
  CLAUDE.md dataset-pinning rule.
- **Workflow tier-lint gap**: the tier-lint hook only scans 500 chars past each `agent(` call,
  so a long inline prompt can hide a `model: 'sonnet'` override past the scan window. Hoist
  long prompts into named constants ABOVE the `agent()` call so the override stays inside the
  scanned prefix.
- Non-blocking carried items (unchanged from runbook 34): [P3-HGRID] claim card; joint_r1
  attribution; MFG-a verbatim check; F0-SEL follow-up; AMEND-2 stale log-substring gates;
  bias-state artifact refresh (rows #166–#211); the WBHZERO proposal §6 caveats (h-dependence;
  redshift-filter sibling convention) as candidate threads.

## 3. Resume recipe (one line)

Author rules on the nine pending [P3-MKER]/[HIER] decisions above (§1) → on ratification,
[P3-MKER] window-geometry prereg (D-MKER-2) and/or [HIER]'s seven-step zero-compute repair →
[P3-2D] resume when fresh (S̄_φ fix first, then the symptom card to a Stage-L searcher) →
archive-or-migrate the `emri` cluster workspace before 2026-09-23 (zero extensions) →
paper-facing consolidation (WBHZERO one-day arc + [P3-2D] forensic discipline are both
thesis-chapter material).
