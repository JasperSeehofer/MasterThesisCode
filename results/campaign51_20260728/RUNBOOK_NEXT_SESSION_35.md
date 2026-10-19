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

## 4. OVERNIGHT ADDENDUM 2026-08-27 [OPUS-ORCH]

Two threads ran overnight, autonomously, Opus-orchestrated, local-CPU-only (no cluster, no SSH,
no source edits). Both are chair verdicts on chair-and-verifier evidence — **not** author-ratified.

### 4.1 [WGEO] — window geometry as a bias lead — **LEAD-DEAD, banked as a CLOSED NULL**

New file: `results/campaign51_20260728/realistic_20260729/CLAIM_WGEO_20260827.md` (552 lines).

**Hypothesis.** The mass-eligibility window is linear-symmetric while the catalogue BH-mass error
model is log-normal, so the induced cut is asymmetric in ln M; if that asymmetry varies with z it
would be a z-structured selection bias matching the dark-class high-z base tilt.

**Killed on four independent sufficient grounds:**

1. **Flat where the tilt is structured.** The window-asymmetry statistic (median 1.5·CV −
   ln(1+1.5·CV)) is **flat across the four banked dark-class tilt z-bins** — 0.3990 / 0.3990 /
   0.3987 / 0.3987, spread 0.08% — while the banked score over those same bins runs
   −0.465 / −0.743 / −0.902 / −1.081, a factor 2.3 of growth. No z-structure for the tilt to be
   attributed to.
2. **Wrong-signed where structure exists.** Where CV does show z-dependence it is low-z-only and
   mechanically GLADE stellar-mass-error quantization, not window geometry
   (Spearman(z, CV) = −0.6521 marginal, decaying to −0.1703 over 0.4 ≤ z < 1.0) — the wrong
   direction for H1.
3. **Structurally impossible.** The C-C control is `L_cat_no_bh == 0` at `handler.py:646`, strictly
   **upstream** of `mass_filter_mask` at `handler.py:663-674` — there is no arm on which the
   window's effect can even be isolated from the class the tilt is measured on.
4. **Channel-identical.** The tilt is 1D 0.6001 vs 2D C-C 0.6004 — statistically indistinguishable
   — but the 1D leg never sees the mass window at all.

**THE MOST IMPORTANT FINDING — HB rediscovery.** The lead was **already measured and
self-refuted** on 2026-07-30 as exoneration **HB** (`CLAIM_2D_BIAS_20260730.md:732-734`): "hard
mass window as support truncation (tilt −0.317 nats = 0.063% of the target, sign-inverted, 40-50×
too small)". HB's banked rationale at `HANDOFF_20260730.md:102-109` is the [WGEO] hypothesis
almost verbatim (negative lower edge, one-sidedness, 193 low-side vs 1 high-side rejections,
h-dependence). One of the three stage-0 reads reported "rule-1 PASSED" and "the window's H0
contribution is UNCONSTRAINED" — **both refuted by the adversarial verifier**, whose rule-1 check
found the coupling read had stopped two lines short of HB in the exoneration list it was
supposedly checking against. The card as originally read was written to CORROBORATE HB, not
challenge it — the correct rule-1 outcome, reached only on the second pass.

**Directional correction (secondary but registered):** the original framing — that a log-symmetric
window would be narrower — is **refuted**: the log window admits FEWER candidates
(n_log/n_lin = 0.4437 on a cone-exact fleet reconstruction), because the linear window's
non-positive lower edge makes the too-heavy-side exclusion vacuous for 99.61% of the catalogue.

**Not blocked on rows #198-#202** (the symmetric-window adoption concerns which side's uncertainty
gets the multiplier — orthogonal to linear-vs-log shape; delimitation verified at source).

**Also filed:** citation drift — `BIAS_HISTORY_LEDGER.md:130` points at
`CLAIM_2D_BIAS_20260730.md:191-204` for the exoneration list, which now begins at line 721 (HB at
:732-734). The pointer is stale.

**New pending author items** (add to §1 below): R-WGEO-1, R-WGEO-2, D-WGEO-1, R-WGEO-3.

### 4.2 [HIER] — blocker discharge — amendments PA-HIER-19..26 + a re-stated LAUNCH GATE

Modified file: `results/campaign51_20260728/realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md`
(1357 → 1834 lines, append-only, verified).

- **PA-HIER-1's worst reading REFUTED.** Of five `host_mode` generator laws, exactly two
  (`catalogue_selected` / arm b0i, and `catalogue_selected_2d`) give a genuine truth-θ = (0, 1);
  the default `catalogue` gives s → 0 and both `population*` modes make the axis inapplicable.
  **Fix is a one-line `host_mode` change, not a redesign.**
- 5 of 7 kernel-identity legs certified at source; the S̃_φ quadrature is GL-50 on both sides —
  explicitly **not** a PA-2D-2/3 borrowed-quadrature mismatch.
- PA-HIER-7 resolved: PROFILED is the only variant with a valid χ²₂ correspondence; FIXED-h
  overstates identifiability in the direction that would authorize Stage F's 424.4 CPU-h;
  MARGINALIZED is REPORTED-ONLY.
- **The critical remaining problem — the null arm is NOT rescuable by one line, and the obvious
  control is itself confounded.** No configuration anywhere in the repo injects a z-kernel
  misspecification: at every `sigma_scale` the generating width and the estimator's quoted width
  are the identical number, by design (`observed_realization.py:454-462`, docstring :185-187).
  Worse, the obvious control (rewriting the quoted `z_error` column) is **confounded**: that column
  also feeds candidate selection (`handler.py:250`, `:636-644`), so it would change which galaxies
  are candidates — a perturbation the lever never makes. That is a built-in CONTROL-FAIL of the
  PA-2D-9 class inside the proposed remedy itself.
- **Verdict: LAUNCH-BLOCKED (unchanged), reason changed shape.** The venue problem is now a
  one-line fix (good news, genuine); the instrument (θ hook, C1/C2) does not exist anywhere in the
  codebase (zero occurrences); the control (C3, the s-axis positive control) is a new instrument
  with its own confound and needs its own pre-registration.
- Nine author one-liners now registered (§(ii) of the re-stated LAUNCH GATE, all zero-compute,
  six [RULE] + two [DO] + one scope [RULE]); five NEEDS-CODE items C1-C5, of which **C1, C2, and
  C3(iii)** land in or adjacent to physics-trigger files and **re-open the `/physics-change` scope
  question** the prereg's §1.5 currently closes.

**New pending author items** (add to §1 below): the nine [HIER] one-liners — venue ratification,
θ-hook authorization, physics-change scope ruling, certification-bar gate/disclose, control
build/fallback, θ-prior adoption, b-grid re-anchor, h-support pin, and support-semantics
hard/affordable ruling. (Full text: PREREGISTRATION_HIER_HTHETA_20260826.md §(ii), lines
1756-1769.)

### 4.3 Updated open-decision list (added overnight to the nine already in §1)

- **[RULE] R-WGEO-1** ratify the [WGEO] KILL and closed-null banking.
- **[RULE] R-WGEO-2** ratify the rule-1 finding that HB governs this object; no window-as-bias
  claim may be banked without new evidence engaging HB's −0.317-nat/0.063%/sign-inverted
  measurement directly.
- **[DO] D-WGEO-1** authorize a ≤1h zero-compute records read reconciling the window-removal
  counterfactual quoted at two magnitudes (+0.010 at `CLAIM_2D_BIAS_20260730.md:726-727` vs
  +0.0015 at `HANDOFF_20260730.md:87-88`, factor ≈7) — until resolved, HB's bound is not quotable
  as a point value.
- **[RULE] R-WGEO-3** rule on whether the linear→log switch's eligible-set mean-redshift moment
  (−14.5%, ✓VER-only) raises the priority of pending D-MKER-2.
- **[HIER] items 1-9** (§4.2 above) — venue, θ-hook, physics-change scope, certification bar,
  control build/fallback, θ-prior, b-grid anchor, h-support, support semantics.

### 4.4 Session-earned ops

- **THE BIG ONE.** A stage-0 rule-1 check by a single agent is NOT sufficient. The [WGEO] coupling
  read reported "rule-1 PASSED" and the governing exoneration (HB) was two lines below the entries
  it quoted from the ledger. Rule-1 checks must grep the exoneration list **exhaustively for the
  mechanism**, not just the tag, and should be adversarially re-checked before a lead is allowed to
  proceed past stage 0.
- **Measure-first discipline paid.** [WGEO] died at stage 0 for a few minutes of local CPU instead
  of a pre-registration + fleet run — the flat-asymmetry check and the HB collision were both
  cheap, decisive, and zero-compute.
- **Citation drift in the ledger's own cross-references is now demonstrated, not hypothetical.**
  `BIAS_HISTORY_LEDGER.md:130`'s pointer into `CLAIM_2D_BIAS_20260730.md` is stale by ~530 lines.
  Line-number pointers into growing files go stale; prefer anchor text (e.g. the exoneration's
  short-name, "HB") as the primary key and treat the line number as a convenience that must be
  re-verified, not trusted.

### 4.5 Revised resume recipe (one line)

Author rules on the thirteen pending [WGEO]/[HIER] one-liners in §4.3 (superset of old §1's nine)
→ on ratification, [HIER]'s venue one-liner (item 1) unblocks S0-A while C1-C3 are built/scoped,
and D-WGEO-1's records read repairs HB's quotable bound before D-MKER-2 is decided → [P3-2D]
resume when fresh (S̄_φ fix first, then the symptom card to a Stage-L searcher) →
archive-or-migrate the `emri` cluster workspace before 2026-09-23 (zero extensions) →
paper-facing consolidation.
