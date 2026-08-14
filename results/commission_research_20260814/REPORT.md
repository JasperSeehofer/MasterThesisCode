# Commission (research mode) — mechanism-isolation-h0-bias — 2026-08-14

## Provenance

- Run ID: `wf_6def92de-d96`
- 27 agents, ~2.38M subagent tokens
- Scope: the five handoff claims C-A..C-E of `HANDOFF_20260814_INDEPENDENT_REVIEW.md`
- Manifest: `.commission-research.yaml`
- Investigators were denied the narrative artifacts (handoff, runbook, campaign report, readout MDs, ledger)

## Chair's adjudication (orchestrator, 2026-08-14)

> The commission's report is ADOPTED as the independent review the author commissioned, with two findings chair-verified by hand before adoption: (1) D2-01 — PREREG_PATH constant at venue_transfer.py:207, stamped at :1906; MEI result JSON confirmed to self-report the 2026-08-11 venue-transfer preregistration. (2) The M5-toy production-K re-execution — independently re-driven by the chair with a fresh driver (impostors-only scatter, 41-point grid, 8 seeds, n_ev=120): K=50 gives +0.0317±0.0007, K=1216 gives +0.0339±0.0006, against the instrument's exact 0.000000. The toy's MEI divergence GROWS with K; the K-saturation mitigation recorded in the campaign report §8 is refuted. (Chair's K=50 differs from the registered +0.0247 because the chair's variant applies the smearing kernel to the exact host rather than point-evaluating it; the conclusion is insensitive to this.)
>
> Verdicts on the five handoff claims: C-A PARTIALLY UPHELD — as executed, no estimator term was identifiable (byte-identical estimator, generator-side doses only), but "unsatisfiable-in-meaning from registration" is WRONG: A-M2′ was a registered estimator-side arm, so branch 2 had an eligible referent at registration; the true fault is readout-time adjudication of a count-based branch while a registered arm capable of changing the count was unrun. C-B OVERTURNED IN SUBSTANCE — the letter still does not fire (zero has no sign) and M1/M4 are demonstrably toy-independent (M3's analytic core stands but its note needs a committed artifact and a bound repair), but abort-(d)'s registered rationale is met in substance: the toy is unfaithful at production K. Goes to the author as a [RULE] with this evidence; pending that ruling every toy-dependent M5/W1 sub-closure is NOT ESTABLISHED. The study does not wholesale stop: M1/M4 stand on toy-independent evidence. C-C OVERTURNED — the register is neither complete nor exhausted: the fired DS-M5 consequence ("returns the study to the M2′ arm") is undischarged, and two omitted candidate classes survive the pre-specified kill tests on committed data (the σ_z-blind tilt × dose-controlled curvature composite; the host/impostor ball-window inclusion asymmetry named in the dossier's own parity text). A-M2′ remains the next arm, but "only survivor" is unlicensed until the two new classes are registered and assessed. C-D OVERTURNED AS DRAFTED — A8's Instance-2 predicate is false and both blocking checks would have PASSED this preregistration; revise before adoption (keep two-sidedness; fix Instance 2; add a blocking execution-completeness check: no count-based branch may be adjudicated while a registered arm capable of changing the count is unrun). C-E TRANSFORMED — neither reading (a) fired-meaning-barred nor (b) NO-OWNER is faithful, because the count itself was adjudicated prematurely over an incomplete registered arm set; the faithful record is PREMATURE ADJUDICATION, resolvable only by running A-M2′ or withdrawing it by [RULE]. All branch calls remain the author's; nothing herein is self-adjudicated.

## Headline

SPLIT VERDICT — the arithmetic survived, the interpretive superstructure did not. Every scored statistic of the mechanism-isolation arc (MN0 +0.034667, MEH +0.004000, MEI exactly 0, MN0X +0.037250/A1-PASS at |Δ|=1.3e-5, the full 16-cell surface, S23 +0.023650) was independently reproduced from raw per-seed ln_post vectors by ≥3 disjoint paths, and a from-scratch reproducer with no project code recovered the host-gate × impostor-amplifier structure — the measurements and the thread's restraint (no repair proposed, empty new-formula slot, A8 left unadopted) are fully vindicated. But four interpretive claims failed falsification: (1) the executed design could not in principle identify an estimator term — every arm and cell is generator-side with a byte-identical estimator, MEI's TERM-OWNS rests on a ~2300-nat single-grid-point posterior collapse that any register candidate would produce, so "branch 2 fires" is mechanical fact with barred meaning, correctly recorded but carrying zero term-attribution power; (2) the M5 L0 toy is unfaithful at production K — the commission re-executed the actual toy at K=1216 and it predicts +0.0341 impostor-only where the instrument measures exactly 0, meeting abort-(d)'s registered rationale in substance and impeaching all toy-dependent M5/W1 sub-closures; (3) the register is neither exhausted nor complete — A-M2′ was never run despite the registered "returns the study to the M2′ arm" consequence having fired, and an omitted tilt×curvature composite class survives three pre-specified kill tests on the committed data; (4) a provenance defect: all 20 result JSONs stamp the wrong preregistration (the 2026-08-11 venue-transfer doc) via an unparameterized PREREG_PATH constant, chair-verified. Net: real progress (structure discovered, M5′-as-registered refuted, A1 remedy sound), but "six candidates narrowed to one unowned structure" is not established as register exhaustion, and no term-level conclusion is licensed until A-M2′ is run or withdrawn.

## Per-claim verdicts

| claim_id | verdict | converging_lines | feedback_verdict |
|---|---|---|---|
| D1-01-arm-biases | CONFIRMED | 5 | raise-consideration |
| D1-02-non-additivity | CONFIRMED | 4 | suggest-change |
| D1-03-dsm5-refutes-m5prime | CONFIRMED | 4 | raise-consideration |
| D1-04-vm1-fails-n15 | CONFIRMED | 3 | request-change |
| D1-05-a1-window-derivation | CONFIRMED | 3 | raise-consideration |
| D1-06-mn0x-a1-pass | CONFIRMED | 4 | raise-consideration |
| D1-07-fhost0-row-scan-confounded | INCONCLUSIVE | 3 | request-change |
| D1-08-s23-dsd3-fires-refutes-hint | CONFIRMED | 4 | suggest-change |
| D1-09-no-referent-is-design-fact | CONFIRMED | 5 | raise-consideration |
| D1-10-m2prime-never-run | CONFIRMED | 5 | request-change |
| D1-11-a8-proposed-not-adopted | CONFIRMED | 3 | suggest-change |
| D1-12-abort-d-unresolved | CONFIRMED | 3 | request-change |
| D1-13-vm5-golden | INCONCLUSIVE | 1 | raise-consideration |
| D1-14-intake-dossier | CONFIRMED | 3 | request-change |
| D1-15-claudemd-approval-tags | CONFIRMED | 2 | raise-consideration |
| D1-16-book-ch14-13of13 | INCONCLUSIVE | 0 | suggest-direction |
| D1-17-cluster-cpu-h | INCONCLUSIVE | 1 | raise-consideration |
| D1-18-a1-det-bit-identity | CONFIRMED | 2 | raise-consideration |
| D2-01-wrong-prereg-metadata | CONFIRMED | 2 | request-change |
| D2-02-untracked-pinned-inputs | CONFIRMED | 2 | suggest-change |
| D2-03-n-asymmetry-preregistered | CONFIRMED | 2 | raise-consideration |

### D1-01-arm-biases

MN0 +0.034667±0.001579, MEH +0.004000, MEI exactly 0.0 on 15/15 seeds — reproduced from raw per-seed vectors by delta reviewer, both REPRO reproducers, tournament, and a from-scratch toy that recovers the same corner structure. The numbers are beyond dispute; note only that MEI's exact zero is a grid-statistic (refined MAPs scatter ±~1e-3 around truth).

### D1-02-non-additivity

The 11.5% recovery / +0.0307 (~18σ) non-additivity arithmetic checks out on every recompute, and the interior-only 3×3 refit independently rejects an f_h-only surface (χ²/dof=46, S23 +11σ), so the interaction is not purely a boundary artifact. However, quote it with the point-evaluation confound caveat: f=0 members switch into the σ_k=0 point-evaluation code branch, so the f=0 corners conflate generator dose with estimator kernel exactification — non-additivity is established, but its "requires a de-pinned host" half is partially anchored on confounded cells.

### D1-03-dsm5-refutes-m5prime

DS-M5's decisive half is inverted (measured 0.000 vs required ≥0.030); confirmed by three raw recomputes plus the commission's re-execution of the actual M5 toy at production K (predicts +0.0341 at K=1216 — divergence grows with K), which also disposes of the "different σ=0 convention" counter-hypothesis (the toy point-evaluates too). M5′-as-registered is dead; carry forward that every toy-calibrated band inherits the toy's demonstrated unfaithfulness.

### D1-04-vm1-fails-n15

|0.034667−0.037237|=0.002570 > 0.002 fires V-M1/STUDY-CONFOUNDED mechanically at N=15 — arithmetic reproduced everywhere. Request-change: Amendment A1 §4.5 explicitly excludes "the branch definitions" from amendment and its verdict keeps MN0 FAILED, so branch 1's first disjunct still names arm N-0 and remains TRUE in the registered text; a registered discharge re-pointing that disjunct to MN0X must be put to the author as a [RULE], else a strict reading voids the downstream branch calls.

### D1-05-a1-window-derivation

The ±0.002 window was underived, ~1.253σ wide with 21.0% false-fail at N=15 (2·Φ̄(0.002/0.0015957)=0.21006 recomputed digit-for-digit by two phases); the observed 1.61σ miss is plausible noise. Consideration: the buy-more-seeds-after-a-fail pattern was legitimate here (window unchanged, pre-committed FAIL reading, unfavourable seeds included) but must not become routine without pre-registered sequential rules.

### D1-06-mn0x-a1-pass

MN0X N=100: +0.037250±~0.000494, |Δ|=0.000013 vs reference — recomputed independently by three phases; fresh-85 seeds land +0.0377, 3.6-4.8σ ABOVE the fail boundary, establishing the N=15 miss as noise rather than a narrated-away confound. A1-PASS is decisive on its own terms (pending the D1-04 branch-referent discharge).

### D1-07-fhost0-row-scan-confounded

The measurement is confirmed (all four f_h=0 cells zero at grid precision, 60/60 seeds), but the compound claim "SCAN-CONFOUNDED does not fire" is contested by two independent lines: the adjudicator's own literal branch tree records b_S00=2.2e-16, fires=true under the registered exact-equality rule (contradicting the appended "Branch 1 did not fire"), and refined MAPs are nonzero (~1e-4) in all 60 seeds, so "exact annihilation" is a 0.005-grid artifact. Request-change: reconcile the float-epsilon vs exact-equality reading in a registered record and restate DS-D4 as a bound (|residual| ≲ 1e-4), not annihilation.

### D1-08-s23-dsd3-fires-refutes-hint

b(S23)=+0.023650 fires the one-sided SHAPE-INTERACTION threshold while sitting +10.33σ above H-INT's own registered prediction (recomputed exactly: (0.023650−0.017333)/0.00061154=10.3297) — a genuine structural prereg fault, legible in text committed before the data. Suggest-change: formally adopt the two-sidedness check and bar the SHAPE-INTERACTION label anywhere it appears without the refutes-H-INT caveat; both registered shapes are wrong and only "both shapes wrong" is quotable.

### D1-09-no-referent-is-design-fact

ARMS.md and the prereg, committed before any arm ran, state E1 requires ZERO estimator change with byte-identical estimator code — verified against the actual code diff (dose_scales only ever touches sigma_pairs/z_obs in _draw_seed_realization; _channel_terms_at_h et al. untouched) by four independent phases. The "no referent" disclosure is a pre-existing design contradiction, not post-hoc spin. Note the prereg is also internally inconsistent (§1 says "estimator-side only", §2 says generator-side) — worth a registered erratum.

### D1-10-m2prime-never-run

No A-M2′ artifact, script, or ARMS.md code form exists anywhere; every executed arm/cell varies a generator-side dose. The registered DS-M5 consequence ("returns the study to the M2′ arm") fired and was never executed or withdrawn by any registered amendment. Request-change: run A-M2′ (estimator-side Jacobian restoration, DS-M1 scoring, reserved seed decade) or put a formal withdrawal to the author as a [RULE] — until then, all "register exhausted" / NO-OWNER / "one unowned structure" language is unlicensed. Note the M5 toy's own τ-ablation (−32%) predicts A-M2′ lands TERM-PARTIAL at best, so budget expectations accordingly.

### D1-11-a8-proposed-not-adopted

A8 is verifiably PROPOSED/NOT ADOPTED. Before adoption, revise it: its flagship Instance-2 predicate is false against the prereg's own design matrix (A-M2′ was a registered estimator-side arm, so branch 2 was NOT "unsatisfiable-in-meaning from the moment written"), and its two blocking checks would have PASSED this prereg — the actual fault was readout-time (scoring a count-based branch over an incomplete registered arm set). Add a blocking execution-completeness check: no count-based branch may be adjudicated while a registered arm capable of changing the count is unrun.

### D1-12-abort-d-unresolved

Correctly presented to the author unresolved rather than silently decided. The commission now supplies the discriminating evidence for that ruling: re-executing the actual M5 toy (recovered byte-identical) at K=84/1216 shows the impostor-only prediction GROWS to +0.0341 — the toy's K-saturation account is inverted at production K, meeting abort-(d)'s registered rationale ("the toy is then not faithful") in substance even though the letter (sign disagreement) does not fire on a zero. Request-change: put abort-(d) to the author as a [RULE] with this evidence; pending the ruling, downgrade every toy-dependent M5/W1 sub-closure (rows B/C/D/H/I/P/Q/R, the W1 toy leg, the "not worth running" list) to NOT ESTABLISHED. M1 and M4 are demonstrably independent of this toy and stand; M3 needs the separate repair in D1-14/directions.

### D1-13-vm5-golden

The 1.6135e-14 max-rel-dev pass is artifact-grounded (real script reading real committed references, logic inspected by two phases) but no reviewer re-executed it — inspection of the same JSON is one line, not two. It is cheap to re-run; do so once independently and this closes.

### D1-14-intake-dossier

The dossier exists as described — 16 constraints, OLD formula cited to code lines, NEW slot empty and author-gated, no repair proposed — and its restraint is vindicated. But as a mechanical filter it is defective and must be repaired before use: C1's "EXACT... not a small residual" overclaims grid resolution (refined f_h=0 residuals are nonzero at ~1e-4, so the data only bound, never annihilate), C2 (linear) and C2′ (more-than-first-order) are jointly unsatisfiable so the stated any-row-fails-refutes rule refutes every possible candidate, and the diagonal R_dose (1.214/0.766/0.949, non-constant at 6.1σ) already kills C2 as a functional-form law on the register's own new lever.

### D1-15-claudemd-approval-tags

The [DO]/[RULE]/[STANDING] approval-scope convention is present: confirmed via the commit diff (delta reviewer) and by the chair's direct read of the current CLAUDE.md, which carries the full section including the non-propagation default. Governance change, no scientific content; it directly addresses the "all approved" ambiguity this arc exposed.

### D1-16-book-ch14-13of13

book/** was scope-excluded for every reviewer; the 13/13 gate-check claim was neither verified nor falsified. Direction: have a future non-excluded pass (or a one-shot mechanical checker) verify ch14's headline numbers against the scored JSON twins — and note ch14's "narrowed to one unowned structure" framing must be revised per D1-10 regardless (M2′ unrun means the register is not exhausted).

### D1-17-cluster-cpu-h

0.969 CPU-h/seed was never recomputed from raw SLURM logs. Loose corroboration exists (a stored wall_time_per_seed_s=221s with 15 workers ≈ 0.92 CPU-h/seed), but that is a consistency glance, not a recompute; the "~3.9× faster" anchor comparison and the Route-1 attribution remain unaudited. Cheap to close from results/mechanism_study_20260813/logs/*.out.

### D1-18-a1-det-bit-identity

The regression test file is real and, decisively, REPRO independently reproduced the substance: the 15 MN0 seeds inside MN0X are bit-identical across instrument commits e83ed0b9 vs 3aedbe55 (max deviation exactly 0.0), and the code diff shows dose_scales=(1,1)/'all' cannot reach the estimator. The "three independent ways" framing of the original verification was not itself re-run, but the invariant it certifies is independently established.

### D2-01-wrong-prereg-metadata

All 20 period result JSONs self-report the unrelated 2026-08-11 venue-transfer preregistration — chair-verified this session (every file returns that path; PREREG_PATH is a single hardcoded constant at venue_transfer.py:207 stamped at :1906, never parameterized when MECH_CELL_SPECS/SCAN_CELL_SPECS were added). Request-change: parameterize PREREG_PATH per registry, and ship a correction artifact mapping the 20 JSONs to their true governing documents so no future auditor or tool is misdirected by machine-readable provenance.

### D2-02-untracked-pinned-inputs

The three scientific inputs every arm consumes (CRB CSV, frozen-α JSON, pruned catalogue) plus the injection pool live in git-untracked directories — chair-verified (git ls-files returns nothing; status shows "??") — defended only by self-computed MD5s against hardcoded constants, which currently match (reviewer's live md5sum). No active inconsistency, but a clean checkout cannot reconstruct these runs. Suggest-change: archive the pinned inputs with committed checksums to persistent storage now, especially given bwHPC workspace expiry is a known project constraint.

### D2-03-n-asymmetry-preregistered

The N=15 vs N=100 asymmetry (S23, MN0X only) was attacked as a cherry-picking smell by two independent phases and survives both: registered before any data with a stated statistical rationale (S23 is the only cell whose rival predictions clear a 3σ dead-band at N=15) and verbatim author ratification. Resolved non-issue — recorded here so downstream reviewers do not re-flag it.

## Regressions and standing defects

- No regressions detectable: CLAIM_HISTORY is empty, so there is no past-CONFIRMED baseline to diff against. This is the founding entry for the research-thread ledger.
- NEW standing defect registered for future diffs (not a regression): all 20 mechanism-study result JSONs carry machine-readable provenance pointing at the wrong preregistration (venue_transfer.py PREREG_PATH constant) — if a future period claims "provenance metadata correct" without fixing this, that would be a regression against this report.
- NEW standing risk registered for future diffs: the pinned scientific inputs (CRB CSV, frozen-α JSON, pruned catalogue, injection pool) are git-untracked and defended only by in-code MD5 constants; any future claim built on these inputs silently regresses if the local files are regenerated or lost.
- Watch item: the L0 toys (m5_toy.py, m3_toy.py) that closed register candidates exist nowhere in git history. The M5 toy was recovered from a stale scratchpad this period and shown unfaithful at production K; if the scratchpad is garbage-collected, the M5/W1 sub-closures become permanently unauditable — commit reproducers or the closures regress to unverifiable.

## Directions & considerations

- **request-change**: Run A-M2′ or formally withdraw it via an author-ratified [RULE] before any "register exhausted", NO-OWNER, or "one unowned structure" language is used anywhere (including book ch14). The registered DS-M5 consequence ("returns the study to the M2′ arm") fired and stands undischarged; it is the single largest open obligation of the period. The toy's own τ-ablation (−32%) predicts TERM-PARTIAL, so pre-register bands accordingly.
- **request-change**: Register one reading rule and apply it uniformly to the four letter-vs-substance adjudications, each put to the author as a [RULE]: (i) branch 1's V-M1 disjunct still names arm N-0 (FAILED at N=15) — discharge it to MN0X by registered amendment; (ii) the S00 float-epsilon (2.2e-16) vs exact-equality SCAN-CONFOUNDED reading — reconcile the adjudicator's fires=true against the appended "did not fire"; (iii) abort-(d) letter (no sign flip on a zero) vs substance (toy re-executed at K=1216 predicts +0.0341 vs measured 0.000) — adjudicate with the new evidence; (iv) DS-D4 "exact annihilation" → restate as a bound (refined residuals ~1e-4). Selective letter/substance application is currently the record's biggest credibility exposure.
- **request-change**: Fix venue_transfer.py's PREREG_PATH to be per-registry (parameterize into MECH_CELL_SPECS/SCAN_CELL_SPECS) and commit a correction artifact for the 20 existing JSONs; simultaneously repair the intake dossier before it is used as a mechanical gate (C1 EXACT→bound; resolve the C2/C2′ joint unsatisfiability; add the omitted candidate classes below).
- **suggest-change**: Revise A8 before adoption: correct Instance 2's false predicate (A-M2′ was a registered estimator-side arm, so branch 2 had a referent-bearing eligible arm at registration), keep the two-sidedness check (it would have caught DS-D3), and add the check the actual fault demands — a blocking rule that no count-based branch is adjudicated while a registered arm capable of changing the count is unrun.
- **suggest-change**: Commit the L0 toys or replace them with committed reproducers: m5_toy.py (recovered from scratchpad this period, load-bearing for M5/W1 closures, unfaithful at production K) and m3_toy.py (source of M3's decisive numerics; the §2 "tight bound" claim is false in the admitted z_obs∈(z_hi, z_hi+5σ_k] regime, though an independent adversarial rebuild reproduced the bottom-line refutation numbers). Downgrade M3 to "plausible pending committed artifact" on the record; the physics conclusion likely survives, the documentation does not meet the project's own reproduction bar.
- **suggest-change**: Archive the untracked pinned inputs (CRB CSV, frozen-α JSON, pruned catalogue, injection pool) to persistent storage with checksums committed to git — the MD5 guard is currently self-referential and dies with the local disk.
- **suggest-direction**: Break the point-evaluation confound before any further term attribution: add an estimator-only kernel scale so generator scatter and kernel width can be varied independently (the decisive MEI-rerun with host kernel held at full σ_z), and/or run the cheap f_host=1e-3 probe — a discontinuous jump off the f_h=0 row's zero would establish that the row's evidential value is a branch-switch artifact. Until then, f=0 cells conflate dose with kernel exactification and carry no discriminating power over the register.
- **suggest-direction**: Register the two candidate classes the period's own data support but the register omits: (a) the σ_z-blind aggregate log-posterior tilt × dose-controlled curvature composite (gradient at truth dose-invariant at 2625–2720 nats/h across cells; bias/post_sd² ≈ constant ≈ the registered S_need; α-share 52.7% predicted vs 53.3% measured — survives all three pre-specified kill tests); (b) the host/impostor ball-window inclusion asymmetry named verbatim in the dossier's own parity text but never assigned an M-ID or arm. Both are cheap to formalize from committed data before any new cluster spend.
- **raise-consideration**: Sequential-amendment hazard: A1 (more seeds after a fired fail) was legitimate here — window unchanged, FAIL reading pre-committed, unfavourable seeds included — but the pattern is exploitable in general; adopt pre-registered sequential rules before it recurs.
- **raise-consideration**: Independence-protocol hygiene: two phase agents disclosed that a directory-wide grep leaked lines from excluded readout files (content disclosed as unused). Future commissions should sandbox greps to an allowlist; also one agent's scratchpad contained a sibling agent's files. Neither affected verdicts, both are process leaks worth closing.
- **raise-consideration**: Two period claims remain unadjudicated within scope and should be closed cheaply next period: the book ch14 "13/13 gate-checked" claim (scope-excluded) and the 0.969 CPU-h/seed / ~3.9× performance claim (needs a SLURM-log recompute; loosely consistent with stored wall_time_per_seed_s). The V-M5 golden (1.6e-14) needs one independent re-execution.
- **raise-consideration**: What positively survived and deserves to be quoted as the period's real yield: the host-gate × impostor-amplifier dose structure (reproduced from scratch with no project code), the refutation of M5′-as-registered, both registered shape hypotheses wrong (super-bilinear surface), the A1 remedy as a model of honest band repair, and a reproduction discipline (raw-vector rescoring at max deviation exactly 0.0 across 455 seeds, now confirmed by third and fourth independent paths) that is the strongest part of the whole record.

## Decision log

- **Hypothesis**: MEI's TERM-OWNS / branch-2 SINGLE-OWNER identifies an estimator term.
  **Result**: Refuted as identification: code diff at 3aedbe55 shows all arms/cells share a byte-identical estimator (dose_scales reaches only sigma_pairs/z_obs in the generator step); MEI's classification rests on a ~2300-nat single-grid-point posterior lock (median best-vs-runner-up gap 2298.5 nats, 15/15 seeds) that any register candidate would produce; the curvature ladder (~1.8e8 vs ~3e4 nats/h²) shows a full-bias-sized smooth per-h defect would displace exact-host cells by only ~1e-5, invisible at the 0.005 grid. The mechanical firing is fact; the meaning is correctly barred; term-attribution power of the executed design is zero.

- **Hypothesis**: The M5 L0 toy is faithful at production K (its K=50 split extrapolates).
  **Result**: Refuted by direct re-execution: the recovered byte-identical toy reproduces its registered K=50 value (+0.02468) and, extended unchanged to K=84/1216, predicts +0.0279/+0.0341 impostor-only — divergence from the instrument's exact 0.000 GROWS with K. The "wrong σ=0 convention" counter-explanation also fails (the toy point-evaluates undosed members). Abort-(d)'s substance is met; toy-dependent M5/W1 sub-closures are impeached.

- **Hypothesis**: M1/M3/M4 closures depend on the falsified M5 toy (so all L0 closures reopen).
  **Result**: Refuted: greps and independent rebuilds show zero m5_toy dependence — M1 is analytic + committed pp_coverage artifacts, M4 is an exactness identity + α-deletion on stored campaign posteriors, M3 uses a separate (uncommitted) toy. An adversarial from-scratch M3 rebuild including the adverse z_obs shoulder reproduced the refutation's order of magnitude (+3.2e-2 slope vs S_need 1.94e3), though it also proved the note's §2 "tight bound" claim false as stated. M1/M4 stand; M3 stands physically but fails the documentation bar.

- **Hypothesis**: M2′ is the register's only surviving candidate and the register is complete.
  **Result**: Refuted twice over: the frozen register is 7-wide (M5′ silently dropped from the six-item framing); the dossier's own §3.1 names a never-armed ball-window asymmetry shape; and the tilt×curvature composite class survives all three pre-specified kill tests on committed data (gradient dose-invariance 2625–2720 nats/h; bias/post_sd²≈const over a 3.3× bias range; α-share 52.7% vs measured 53.3%). Register incomplete; M2′ likely TERM-PARTIAL at best per the toy's own τ-ablation (−32%).

- **Hypothesis**: The N=100 upgrades (S23, MN0X) were post-hoc cherry-picks.
  **Result**: Refuted: both registered before any data (scan prereg §B/§4.3 with a 3σ-dead-band rationale; Amendment A1) under verbatim author ratification. Resolved non-issue.

- **Hypothesis**: The V-M1 N=15 failure indicates a real confound.
  **Result**: Refuted: the window was a ±1.253σ acceptance with 21.0% design false-fail rate (recomputed digit-for-digit); MN0X at N=100 with the window unchanged lands |Δ|=1.3e-5 (153.8× inside), and the 85 fresh seeds land 3.6–4.8σ ABOVE the fail boundary. The miss was noise; A1-PASS is decisive — pending only the formal branch-1 referent discharge.

- **Hypothesis**: The host×impostor interaction is a boundary artifact of the confounded f=0 cells.
  **Result**: Refuted for the interior: inverse-variance refit of the 3×3 interior alone rejects an f_h-only surface (χ²/dof=46; S23 residual +11σ) and requires an interaction term — though the surface is super-bilinear (both registered shapes wrong: S23 +10.33σ above H-INT, H-THRESH dead at 18–50σ) and dose-surface curvature is not term attribution.

- **Hypothesis**: The result JSONs' machine-readable provenance is correct.
  **Result**: Refuted, chair-verified: all 20 period JSONs stamp the 2026-08-11 venue-transfer preregistration via the unparameterized PREREG_PATH constant (venue_transfer.py:207→:1906); the two governing 2026-08-13 preregistrations appear in no result file's metadata.

- **Hypothesis**: A8's two blocking checks would have caught the branch-referent fault they are motivated by.
  **Result**: Refuted: applied to the prereg as registered, the branch-referent check PASSES (A-M2′ was a registered estimator-side, referent-bearing arm), and the two-sidedness check does not touch V-M1 (already two-sided, defect was statistical power). The actual fault was readout-time — a count-based branch scored over an incomplete registered arm set — which no proposed A8 check guards against.

- **Hypothesis**: Every scored statistic of the period reproduces from raw data.
  **Result**: Confirmed at maximum strength: three-to-four disjoint paths (two delta recomputes, two REPRO rescores over 455 seeds from raw ln_post vectors, tournament re-scoring, a from-scratch no-project-code toy recovering the corner structure and non-additivity) all match to stated precision, max deviation exactly 0.0; cross-commit bit-identity of the 15 shared MN0/MN0X seeds independently established. The arithmetic layer of this thread is the most trustworthy object in the repo.
