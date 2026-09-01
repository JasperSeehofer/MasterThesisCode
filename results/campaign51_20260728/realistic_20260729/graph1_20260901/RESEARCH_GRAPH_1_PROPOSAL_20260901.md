# Research Graph 1 — Proposal for the Next Batch (first instantiation of the typed research graph)

Date: 2026-09-01. Author of record for all scientific decisions: Jasper Seehofer.
Status: PROPOSAL — a reviewable decision artifact per CLAUDE.md "Proposing decisions". Nothing
here runs until the author ratifies the charter (decision row 0) and grants the branch-head rows.

Schema of record: INFRA_AUTONOMOUS_RESEARCH_PROPOSAL_20260901.md in this directory ("infra").
This graph is that proposal's first instantiation: node types per infra section 2.1, edge kinds per
2.2, convergence manifests per 2.3, bounded re-entry per 2.4, the gate panel per 2.5. Infra 3.4's
approval-scope semantics are asked for separately, as their own STANDING item, in section 3 row 1
below — ratifying row 0 (the charter/topology) does NOT by itself grant them (per the CLAUDE.md
approval-scope convention: a STANDING item is granted only when the author says so explicitly).

State of record: STATE_AND_CANDIDATES_20260901.md in this directory ("state"). Every number
carried below is quoted with its ledger row or source document. Cost caps that have no source are
marked ORCHESTRATOR-DERIVED and are charter proposals, not facts (row #268 lesson).

---

## 0. Charter frame

- Objective (per infra 3.2, to be ratified as part of row 0): the batch score is the number of
  registered questions moved to a SETTLED state — verified, refuted, or bounded-undetermined —
  with all consumed panel stamps green or waived. Refuted pays like verified. Bias reduction is
  not in the objective (author's binding 2026-08-05 value).
- Topology: a directed acyclic graph, 9 execution branches (A-I) plus one closure chain, depth
  3-4 execution nodes per branch before its first convergence node, converging on six fresh-RULE
  decide nodes and three terminal paper decide nodes. Re-entry is by counted revision nodes
  (max_revisions frozen below), never back-edges.
- Scope hash: computed at ratification over the node/edge set of section 1 and the instrument set
  of section 3. Any node added later fails L10 and returns as a fresh RULE by construction.
- Cluster precondition: every cluster launch behind the /cluster preflight gate — VERDICT: READY
  required (CLAUDE.md); the Lustre OST 5 blocker seen 2026-08-31 must be confirmed cleared first.

## 1. The graph

### 1.0 Question and claim layer

Question nodes (kill_criterion mandatory per infra 2.1):

| id | question it settles | kill_criterion |
|---|---|---|
| q-postflip-calibration | is the post-flip pipeline (default auto, commit 5e7fda16 — narrated at row #286, hash quoted at row #288) coverage/F-validated and venue-general? | coverage unusable post-flip in both channels at registered bands after revision 2 of r-b82-s4 -> park bounded-undetermined |
| q-theta-pull | is the S0-B photo-z error-model theta-pull real venue physics, now that the instrument is null-certified on both axes (Z_b -1.808 / +0.773, both abs Z <= 3, row #287)? | production S0-B null at abs Z <= 3 -> theta-pull not venue physics at measurable size; report the bound and stop |
| q-completion-residual | how much of the approx -0.14/event dark-class completion-leg residual (artifact section 09) is illegitimate estimator inconsistency vs floor-consistent noise, given F? | registered arm fails to discriminate at its registered band after revision 2 -> park bounded-undetermined with the measured bound |
| q-cone-loss | how much of the absolute bias floor is the approx 17% of cones structurally unable to contain the true host (artifact section 09)? | measurement confirms the floor within its registered uncertainty band -> settled as irreducible geometry; no fix pursued |
| q-a4-provisional | does the A4 mz_sel/eff structural-consistency ratification survive the class-G falsifier? | the falsifier verdict lands either way -> the question closes at d-a4-final-ratification with numbers |

Claim nodes (status gate/decide-written only):

| id | status now | discriminated / verified by |
|---|---|---|
| c-auto-default-venue-general | conjectured | discriminates edge from m-joint-r1-mass-aware (infra 6.3; state candidate 10) |
| c-a4-structural | supported, PROVISIONAL cap (rows #278(4)/#280/#284(3)) | verifies/refutes edge from v-falsifier-ii-classG |
| c-theta-pull-venue-physics | conjectured | discriminates edge from m-s0b-production (vs instrument-artifact, retired by row #287 certification) |
| c-residual-illegitimate / c-residual-floor-consistent | both conjectured (multiple working hypotheses) | jointly discriminated by m-completion-residual + m-cone-loss + the F value from rd-s3-readout |
| c-rphi-mismatch | superseded-pending-retirement (r_phi approx 0.886 mass-blind signature; flip lands Z=1 by construction, rows #269/#286) | d-rphi-retire, fed by rd-rphi-note |
| c-massinfo-negative | supported (sigma_M sweep, artifact section 08; no-GLADE-rescue memory) | promoted to paper claim at d-paper-massinfo |

### 1.1 Branch A — S4 harness repair -> post-flip S3 coverage re-run (calibration chain)

Depth 4. Settles the usability of B8.2's F/coverage numbers post-flip; the pre-flip pilot aggregate
was mixed-N contaminated (3 ladder-costing seeds pooled with 63 real seeds) and missing the cell-T
aggregate (row #288). Pre-flip pilot values now known unusable as calibration: no_bh F=7.426
(pilot record section 3.1; row #288 states the same value to its own precision as F=7.43),
with_bh F=11.35; cell-S HPD 50/68/90/95 = 0.015/0.015/0.061/0.121 (no_bh) and
0.364/0.470/0.803/0.894 (with_bh), PIT-KS D = 0.8045 / 0.3313, all out of band (row #288).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| d-s3-rerun | decide (RULE) | whether S3 re-runs post-flip at all, given the pre-flip no_bh numbers cannot calibrate a post-flip stop rule | requires-manifest: rows #286/#288 (already-existing evidence; no upstream node dependency, so this RULE is eligible immediately and can be resolved before or alongside row 0); authorized-by d-batch1-charter | — (author ruling; no gate instrument) | a No disposition halts Branch A entirely (no b-s4-harness-repair launch); a Yes gates b-s4-harness-repair's launch via an authorized-by edge | authoring only | author |
| b-s4-harness-repair | build | the three S4 defects of row #288 (a)-(c): seed-population separation, missing cell-T aggregation, wall-limited stop rule | authorized-by d-batch1-charter AND d-s3-rerun (Yes disposition); feeds from row #288 pilot record | g-byte-id on untouched code paths; g-population lint on the repaired aggregator | 0 mismatches at N >= 1e5 pairs (infra 2.5); 0 mixed rows | cheap | sonnet / medium |
| r-b82-s4 | register | the repaired S3 registration: bands re-frozen, stop rule under wall-limited runs, population declared n200-postflip | feeds from b-s4-harness-repair; supersedes the S4 items of B8_2_HARNESS_DESIGN_20260829.md section 8; max_revisions 2 ORCHESTRATOR-DERIVED (provisional default, ratified with the charter: two revision attempts is the smallest budget that lets a fixable design flaw survive one round-trip without paying full SETTLED credit for a first-attempt park; no per-node source constrains it) | design-validity gate only (blind to results) | design gate red -> STOP m-s3 launch; stop-rule content returns as part of the d-s4-review fresh RULE | authoring only | top-tier / xhigh (wave-1 prereg author) |
| m-s3-postflip-coverage | measure | clean post-flip coverage + F, cells S and T, N=200 | registered-by r-b82-s4; feeds from m-head-rebaseline (comparand); authorized-by d-batch1-charter | g-sbc-coverage, g-population, g-censoring, g-znorm | registered bands per r-b82-s4; any MAP at a rail flags the read as a bound | approx 12h + 4h wall per cell pair at N=200, wall-limited not completion-limited (row #288) | sonnet / low (cluster array) |
| rd-s3-readout | read | the coverage/F table, verdict-free; three-valued existence contract on every remote read (row #288 SSH lesson) | feeds from m-s3-postflip-coverage | consumes panel stamps; g-population, g-precision on any F/nats arithmetic | red stamp -> STOP d-calibration consumption (panel law) | cheap | sonnet / high |

### 1.2 Branch B — post-flip HEAD re-baseline

Depth 1. Foundational comparand: the banked 2026-08-27 comparand predates the flip and only ever
validated the 2D-twin (A14: +0.002507 iiib / +0.004114 joint_r1, both <= T_mat 0.008, row #284),
never the 1D change (state candidate 11).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| m-head-rebaseline | measure | the new banked HEAD readout under the post-flip default, both venues, C0-prime-then-blind-HEAD pattern (rows #279/#281/#283) | authorized-by d-batch1-charter; feeds from commit 5e7fda16 (row #286; hash quoted at row #288) | g-c0-baseline first, then blind HEAD; g-znorm on the flipped leg | max_abs = 0 on shared columns, md5 match (infra 2.5); red -> STOP every downstream delta-read | single-digit CPU-h (84 tasks x approx 6.5 min, wave-3 model; state candidate 11) | sonnet / low |

feeds edges out: m-s3-postflip-coverage, v-falsifier-ii-classG, m-joint-r1-mass-aware, m-t5-armS,
m-t5-armR, r-completion-residual (every subsequent delta-read measures against it — state candidate 11).
Banking it as the comparand of record is NOT covered by the launch grant; it returns inside d-calibration.

### 1.3 Branch C — joint_r1 mass-aware transfer (venue generality of the flip)

Depth 3. A18 ran on iiib only (job 6747032); the mass law is venue-dependent (delta law on iiib vs
log-normal on joint_r1, row #270), so the iiib transform S_4D/S-bar_phi = 1.039 (row #282) and the
iiib band do NOT transfer — a fresh derivation and a fresh registered band are required (state candidate 10).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| dv-jr1-transform | derive | joint_r1's T2.2b-equivalent transform under its log-normal realized-forward mass law | authorized-by d-batch1-charter; feeds from row #282 (T2.2b) + row #270 (venue mass laws) | /physics-change if any trigger file is touched (expected analysis-only); g-invariance on the derived transform | derivation must state its own h-stability check as T2.2b did (1.039 h-stable, row #282) | authoring | top-tier / xhigh (wave-1 derivation author) |
| r-jr1-massaware | register | the joint_r1 readout rule: registered band, MAP-AND-mean criterion analogous to A18; grid scope (41-node vs G-EXT extended) | feeds from dv-jr1-transform AND b-hprior-fix (grid-scope election); max_revisions 2 ORCHESTRATOR-DERIVED (provisional default, ratified with the charter; see r-b82-s4's derivation sentence above) | design-validity gate | band content returns as fresh RULE d-jr1-band before launch; neither-band disposition INTERMEDIATE -> fresh RULE | authoring | top-tier / xhigh (batched with dv author) |
| m-joint-r1-mass-aware | measure | whether auto lands in-band on joint_r1 | registered-by r-jr1-massaware; feeds from m-head-rebaseline; discriminates c-auto-default-venue-general | g-znorm, g-score-null, g-censoring, g-precision | registered band per r-jr1-massaware; rail-flagged reads demote to bounds | approx 90-100 CPU-h for an A18-equivalent grid, possibly smaller on the 41-node grid (state candidate 10) | sonnet / low (cluster array) |

### 1.4 Branch D — S0-B build -> production run (photo-z leverage)

Depth 3. Unblocked by the both-axes instrument certification (row #287).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| rd-runner11 | read | what runner-11's 8-cell b-node output actually contains (read-first precondition, runbook 40 section 3 item 1) | authorized-by d-batch1-charter | three-valued existence contract | unreachable is not absent (row #288 lesson) | cheap | sonnet / low |
| b-pahier33-scorer | build | the PA-HIER-33 scorer (convention ratified rows #278/#280 via the Richardson adjudication, row #275; never built) + the driver's missing iiib venue path | feeds from rd-runner11; authorized-by d-batch1-charter | g-byte-id on all non-S0-B default paths | 0 mismatches at N >= 1e5 pairs; red -> STOP m-s0b launch | cheap | sonnet / medium (implementation from a ratified spec) |
| m-s0b-production | measure | whether the theta-pull is real venue physics | registered-by the standing S0-B prereg (ratified rows #278/#280); feeds from b-pahier33-scorer; discriminates c-theta-pull-venue-physics | g-score-null (certified instrument, row #287), g-znorm | abs Z <= 3 null band (rows #225/#251/#287); red score-null -> STOP d-photoz-leverage, reopen the instrument question as a fresh RULE (never auto-recertify) | cheap-to-moderate, comparable to the runner-11 8-cell precursor (state candidate 2) | sonnet / low |

### 1.5 Branch E — falsifier (ii), class-G fleet

Depth 1. The A4 [RULE] is explicitly pending this: returns with numbers, not auto-ratified
(rows #278(4)/#280/#284(3)).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| v-falsifier-ii-classG | verify | the PROVISIONAL attribution cap on B7.1/B7.2 (verifies-or-refutes c-a4-structural) | authorized-by d-batch1-charter; feeds from m-head-rebaseline (comparand for the read); independent of the flip (state candidate 3) | structured artifact either way — counterexample or survival record naming what was tried (infra 2.1) | disposition rule of the A4 RULE; fleet cost cap 60 CPU-h hard | approx 40-60 CPU-h (runbook 40 section 2) | fleet: sonnet / medium (fan-out capped); adjudicating read + re-derivation: top-tier / xhigh (the wave-2 decisive-verifier slot) |

### 1.6 Branch F — T5 k-scan, Arms S and R

Depth 1-2 per arm. Design already ratified: the F-ii RULE (row #278(2), restated row #284(4));
Arm R is RATIFIED-AS-RECOMMENDED, launch when cluster allows (row #284(4a)).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| m-t5-armS | measure | log-symmetric window vs linear k=1.5 on iiib (k in 2.0/2.5/3.5 + optional k=infinity anchor) | registered-by the F-ii-ratified design; feeds from m-head-rebaseline | g-znorm, g-c0-baseline (comparand check) | per the ratified design | approx 15-20 CPU-h (state candidate 6) | sonnet / low |
| m-t5-armR | measure | the same question on joint_r1 at decisive k=3 | registered-by the F-ii-ratified design; feeds from m-head-rebaseline; gated behind its OWN C0-prime-equivalent ingredient check — the wave-3 generic C0-prime (job 6746274) was for the 2D-twin, not this arm (state candidate 6) | g-c0-baseline (fresh evaluation for this configuration), g-znorm | C0-prime-equivalent red -> STOP Arm R launch | approx 11-15 CPU-h (state candidate 6) | sonnet / low |

### 1.7 Branch G — dark-class completion-leg residual program (B8 centerpiece)

Depth 2 inside this graph (register -> measure), deliberately gated late: its register node needs
the F value (from rd-s3-readout) and the re-baseline. Object: the approx -0.14/event dark-class
completion residual, the largest unexplained item on the board (artifact section 09; runbook 40
section 0 names it B8 [CAL]'s next centerpiece, row #286). Context numbers it must reconcile: the
residual 1D rail -0.063 (mean 0.667 vs truth 0.730, row #286), the 2D offset -0.0667 with
sigma_h = 0.0184 (bias approx 3.6x its own width, artifact section 00), the information floor
sigma_h = 0.001747058397810697 (no_bh) / 0.001746970592930231 (with_bh) at N_ref = 1588
(b8_information_floor.json, pilot record section 3.5), and the depth-skew 73.0% +/- 1.4% of
catalogue-leg weight below true redshift, 16 sigma from no-skew (artifact section 05).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| r-completion-residual | register | the first registered arm splitting the -0.14/event into illegitimate vs floor-consistent, with discrimination band and disposition rule | feeds from rd-s3-readout (F) AND m-head-rebaseline; spawned-by q-completion-residual; max_revisions 2 ORCHESTRATOR-DERIVED (provisional default, ratified with the charter; see r-b82-s4's derivation sentence) | design-validity gate; research-cycle stages 2/3 | band + cap ratification returns as fresh RULE d-completion-register before launch | authoring; arm cap <= 80 CPU-h ORCHESTRATOR-DERIVED, unscoped in sources | top-tier / xhigh (wave-2/3 prereg author) |
| m-completion-residual | measure | the registered discrimination | registered-by r-completion-residual; discriminates c-residual-illegitimate vs c-residual-floor-consistent | g-closure (decomposition terms sum to the total), g-population, g-precision | registered residual band; neither-band -> INTERMEDIATE -> fresh RULE | within the ratified cap | sonnet / medium |

### 1.8 Branch H — cone-loss quantification

Depth 2. The approx 17% of localisation cones structurally unable to contain the true host —
leading candidate for the absolute bias floor that no consistency fix will touch (artifact
section 09). Undesigned; likely catalogue-geometry analysis, not cluster-heavy (state candidate 5).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| r-cone-loss | register | the cone-loss measurement design against localisation-cone / catalogue-completeness geometry, with an uncertainty band | spawned-by q-cone-loss; authorized-by d-batch1-charter; max_revisions 2 ORCHESTRATOR-DERIVED (provisional default, ratified with the charter; see r-b82-s4's derivation sentence) | design-validity gate | band ratification returns as fresh RULE d-cone-register before launch | authoring; arm cap <= 20 CPU-h ORCHESTRATOR-DERIVED, unscoped in sources | top-tier / xhigh (batched with r-completion author) |
| m-cone-loss | measure | the quantified floor share | registered-by r-cone-loss; discriminates c-residual-floor-consistent | g-population, g-censoring | registered band | within the ratified cap | sonnet / medium |

### 1.9 Branch I — h-prior upper-bound fix (small enabling node)

Depth 1. 14 G-EXT extension-node tasks failed on the h-prior upper bound; disclosed
verdict-irrelevant at tail 5e-13 for A18 (row #286) — maintenance that becomes load-bearing only
where a registration elects the extended grid.

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| b-hprior-fix | build | trustworthiness of the extended h-grid's outer nodes (above 0.86) | authorized-by d-batch1-charter; feeds into r-jr1-massaware (grid-scope election) | g-byte-id below the old bound | 0 mismatches below the old bound; red -> STOP any registration electing the extended grid | cheap: config fix + rerun of the 14 failed tasks, a fraction of the approx 94 CPU-h original grid (state candidate 8) | sonnet / low |

### 1.10 Closure chain — r_phi retirement

Depth 2. The flip moots r_phi by construction: under the auto/on mass-aware branch the numerator
and divisor are matched-content, Z=1 identically (rows #269/#286; transform 1.039, row #282).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| rd-rphi-note | read | the written confirmation: the FIRST standing g-znorm panel evaluation on the flipped production leg, tied into the A18 closure record | feeds from commit 5e7fda16 (row #286; hash quoted at row #288) + row #282 | g-znorm (this node IS its first panel evaluation) | abs dev <= 1e-6 green, > 1e-3 anomalous (infra 2.5); anomalous -> STOP d-rphi-retire and reopen as fresh RULE | cheap | sonnet / low |
| d-rphi-retire | decide (RULE) | formal retirement of c-rphi-mismatch on the open-branches board | requires rd-rphi-note done with g-znorm green | — | returns to the author WITH the note; never pre-granted | — | author |

### 1.11 Convergence decide nodes (all return as fresh RULEs; eligibility computed from manifests)

| id | tag | question put to the author | requires-manifest |
|---|---|---|---|
| d-s3-rerun | RULE | does S3 re-run post-flip at all, given the pre-flip no_bh numbers cannot calibrate a post-flip stop rule? | rows #286/#288 (already-existing evidence); no upstream node dependency — eligible immediately, resolvable before or alongside row 0 |
| d-calibration | RULE | is the post-flip pipeline calibration-validated: F/coverage usable, flip venue-general, re-baseline banked as comparand of record? | rd-s3-readout done with g-sbc-coverage + g-population green-or-waived; m-head-rebaseline done with g-c0-baseline green; m-joint-r1-mass-aware done with band disposition assigned |
| d-photoz-leverage | RULE | is the theta-pull real venue physics, and what bound goes into the irreducible-venue-physics split? | m-s0b-production done with g-score-null green; rd-runner11 record attached |
| d-a4-final-ratification | RULE | drop the PROVISIONAL flag on A4 mz_sel/eff? | v-falsifier-ii-classG verdict record done; panel g-score-null + g-c0-baseline green-or-waived (infra 6.3, verbatim) |
| d-t5-window | RULE | adopt the log-symmetric window, per venue? | m-t5-armS done; m-t5-armR done; both with band dispositions assigned |
| d-residual-attribution | RULE | the three-way residual split (artifact section 10): illegitimate inconsistency vs floor-consistent noise vs irreducible venue physics | d-calibration ruled; d-photoz-leverage ruled; m-completion-residual done with g-closure green; m-cone-loss done |
| d-s4-review | RULE | ratify r-b82-s4's re-frozen bands + the wall-limited stop rule (the S5 production-N launch stays behind B8_2 design section 8 regardless) | r-b82-s4 design-gate record done |
| d-jr1-band | RULE | ratify the joint_r1 registered band + grid scope | dv-jr1-transform done; r-jr1-massaware draft done |
| d-completion-register / d-cone-register | RULE | ratify each residual-program arm's band and cost cap | the respective register-node design-gate record done |

### 1.12 Terminal paper nodes (convergence decide nodes; the paper is a traversal of this graph)

| id | deliverable | requires-manifest (explicit input edges) |
|---|---|---|
| d-paper-coverage | the F/coverage-validated 2D result | d-calibration ruled; rd-s3-readout artifact (clean F + coverage, both cells); m-head-rebaseline banked; b8_information_floor.json floor numbers (sigma_h 0.001747058397810697 / 0.001746970592930231, N_ref 1588, pilot record section 3.5) |
| d-paper-1d2d-verdict | the 1D-vs-2D structural verdict | d-residual-attribution ruled; d-a4-final-ratification ruled; d-t5-window ruled; existing feeds: the owned 1D rail -0.063 (row #286), the 2D offset -0.0667 / sigma_h 0.0184 (artifact section 00), depth-skew 73.0% +/- 1.4% at 16 sigma (artifact section 05) |
| d-paper-massinfo | the negative mass-channel information result | d-calibration ruled; rd-s3-readout with_bh-vs-no_bh contrast (the pre-flip contaminated contrast F 11.35 vs 7.426, row #288, is quotable only as motivation, never as the result); existing sigma_M sweep (artifact section 08); the information-floor pair above |

Each paper node, once ruled, dispatches its authoring as a reviewable artifact under the
physics-change protocol and approval-scope tagging (state candidate 9); authoring is chair work,
not covered by any branch-head grant.

### 1.13 Bounded cycles

Three register nodes carry max_revisions 2 (r-b82-s4, r-jr1-massaware, r-completion-residual;
r-cone-loss likewise), all four ORCHESTRATOR-DERIVED per their table entries in 1.1/1.3/1.7/1.8. A
neither-band or refuted outcome spawns revision n+1 as a NEW register node with a spawned-by edge;
exceeding max_revisions forces Hold plus author return (infra 2.4). No back-edges exist anywhere in
this graph.

### 1.14 Checkpoint (k-) nodes — fan-out and tiering caps declared before expansion

Per infra 2.1's k- schema (fanout_cap, tier, per-child cost, all declared BEFORE expansion) and
infra 3.3's stated replacement of launch-summary prose. Every fan-out or tiering number quoted
narratively in section 5.2 is sourced from one of these node attributes, not asserted fresh there.

| id | gates | fanout_cap | tier cap (attribute) | per-child / fleet cost cap | declared before |
|---|---|---|---|---|---|
| k-wave1-fanout | wave 1 launch (b-s4-harness-repair, m-head-rebaseline's 84-task array, dv-jr1-transform, rd-runner11, b-pahier33-scorer, b-hprior-fix, rd-rphi-note, m-t5-armS's array) | 6 build/read agents (fixed, no dynamic expansion) + 84-task HEAD array + Arm S array | top-tier = 2 (chair + one derivation/prereg author covering dv-jr1-transform, r-jr1-massaware draft, r-b82-s4); everything else sonnet low/medium | per §5.1 sourced costs | wave 1 launch |
| k-falsifier-ii-fleet | v-falsifier-ii-classG launch | class-G configuration count (runbook 40 section 2; count fixed at launch, not re-expanded mid-run) | fleet: sonnet/medium (fan-out capped); 1 top-tier adjudicating-read slot (the wave-2 decisive verifier) | 60 CPU-h fleet-wide, hard cap | wave 2 launch |
| k-wave2-fanout | wave 2 launch (m-s3-postflip-coverage's 2-cell x N=200 array, m-s0b-production, v-falsifier-ii-classG via k-falsifier-ii-fleet, m-t5-armR, m-joint-r1-mass-aware's grid array, r-completion-residual + r-cone-loss authoring) | S3 2-cell x N=200 array + joint_r1 grid array + the falsifier fleet (capped at k-falsifier-ii-fleet) | top-tier = 3, at the cap (chair; one prereg author for r-completion-residual + r-cone-loss; one decisive verifier for the falsifier-ii adjudicating read); agent-side panels all sonnet | per §5.1 sourced costs | wave 2 launch |
| k-wave3-fanout | wave 3 launch (rd-s3-readout if not closed, m-completion-residual, m-cone-loss, the decide cascade of six RULEs, three terminal paper rulings) | no cluster fan-out; dossier assembly + authoring only | top-tier = 2 (the chair, who assembles decide-node dossiers and drafts paper artifacts; one end-verifier who re-derives every decisive number feeding a d- node); all dossier mechanics sonnet | authoring only | wave 3 launch |

Batch-cumulative top-tier headcount is tracked, not just per-wave counts, in section 5.2.

---

## 2. Gate panel for this graph

Live instruments this batch (from the infra 2.5 catalogue v0), their triggers, and the node a red
STOPs. Panel law: no rd- or d- node consumes a measure artifact without a green or author-waived
stamp; a red never suppresses the number — it is banked and blocked from interpretation.

| instrument | class | runs continuously on | band | a red STOPs |
|---|---|---|---|---|
| g-znorm | identity | EVERY new production likelihood leg: the flipped 1D leg (first evaluation = rd-rphi-note), m-joint-r1-mass-aware, m-s0b-production, both T5 arms, the completion arm | abs dev <= 1e-6 green; > 1e-3 anomalous | the consuming read/decide: rd-rphi-note -> d-rphi-retire; m-jr1 -> d-calibration; etc. |
| g-score-null | machinery | control venues: m-s0b-production (certified basis: row #287), T5 arms | abs Z <= 3 (rows #225/#251/#287) | d-photoz-leverage; reopens the instrument question as a fresh RULE, never auto-recertifies |
| g-sbc-coverage | machinery | m-s3-postflip-coverage (pp_coverage.py promoted to standing upstream gate) | registered bands per r-b82-s4, population-pure input only | d-calibration (and the S5 production-N launch) |
| g-byte-id | baseline | every build touching a default path: b-s4-harness-repair, b-pahier33-scorer, b-hprior-fix — and any future default flip | 0 mismatches, N >= 1e5 pairs (row #229 pattern) | the build's downstream measure launch |
| g-c0-baseline | baseline | m-head-rebaseline; the Arm R C0-prime-equivalent; every delta-read against the new comparand | max_abs = 0 on shared columns; md5 match | every downstream delta-read; m-t5-armR launch |
| g-population | baseline | every aggregate: the repaired S3 aggregator, rd-s3-readout, m-completion-residual, m-cone-loss | 0 mixed rows (row #288 contamination fix) | the aggregate's consuming read |
| g-censoring | model-given-data | every gridded posterior read: m-s3, m-jr1, m-cone-loss | registered per grid; any MAP at a rail flags the read AS A BOUND (rows #267/#280: 3-of-4 was really 4-of-4) | promotion of the read from bound to estimate at d-calibration / d-residual-attribution |
| g-precision | baseline | every nats/log-sum comparison and F arithmetic | full-precision pinning; cancellation sentinel within 2 s.f. (the +157.92 vs +123.11 storage artifact, row #282) | the comparison read |
| g-closure | identity | m-completion-residual (the decomposition must sum to the total it decomposes) | registered residual band | d-residual-attribution |

Panel meta-rules for this batch: instrument definitions frozen at the row-0 charter; any edit is a
physics change plus a fresh RULE; every evaluation appends to gates/panel/ history; every waives
edge is a per-instance author RULE, never pre-granted (the STANDING grant in decision row 12
covers evaluation, never waiver).

---

## 3. Initial decisions table (one row per branch head)

One-word replies suffice: Approved grants a DO, Ratified grants a RULE, Granted grants a
STANDING row. Per the binding default, NOTHING in this table covers a disposition whose inputs do
not yet exist; every such disposition is listed in its NOT-covered cell and returns via section
1.11. Row 1 below is its own explicit STANDING item — per MUST-FIX 1 of the 2026-09-01 critic pass,
it is NOT folded into row 0's charter ratification; a one-word "Ratified" on row 0 grants ONLY row
0's DO/RULE content, never row 1's STANDING grant.

| # | branch head (node) | tag | ask | triggers on grant | explicitly NOT covered (returns as fresh RULE) |
|---|---|---|---|---|---|
| 0 | d-batch1-charter (the whole graph) | RULE | Ratified | freezes topology (sections 1.0-1.13), caps, the section-2 instrument set, and the section-0 objective at a scope hash; unblocks rows 2-12 | any disposition not yet computed: every band-edge or neither-band call, every claim promotion, every node added after the hash (L10); row 1's STANDING grant (separate ask, separate reply) |
| 1 | infra section 3.4's approval-scope semantics, as the meaning of ratification for graph batches in this project | STANDING | Granted | adopts infra 3.4 (section 3.4 of INFRA_AUTONOMOUS_RESEARCH_PROPOSAL_20260901.md) as the standing meaning of "ratification"/"grant" for every row in this table and every future graph-batch charter in this project | scope: graph batches in this project, this campaign; lapses at campaign end (verbatim from INFRA's own ratification ask) — does NOT itself authorize any topology, band, cap, or node; does not survive to a successor campaign without a fresh STANDING ask |
| 2 | d-s3-rerun (row #288 item (d)'s fresh RULE — its inputs all exist: rows #286/#288) | RULE | Ratified | rules that S3 re-runs post-flip at all, given that the pre-flip no_bh numbers cannot calibrate a post-flip stop rule | the re-run's band re-freeze and stop rule (d-s4-review); the coverage disposition; the S5 production-N launch (stays behind B8_2 design section 8) |
| 3 | b-s4-harness-repair (Branch A) | DO | Approved | the row #288 (a)-(c) repairs; r-b82-s4 registration authoring; m-s3 launches only after d-s4-review and a green design gate | everything in row 2's NOT-covered cell, plus the d-calibration ruling |
| 4 | m-head-rebaseline (Branch B) | DO | Approved | C0-prime check then blind HEAD arrays under the post-flip default, both venues (wave-3 pattern, rows #279/#281/#283) | banking it as the comparand of record and any delta interpretation — both return inside d-calibration with numbers |
| 5 | dv-jr1-transform (Branch C) | DO | Approved | the joint_r1 T2.2b-equivalent transform derivation + the r-jr1-massaware draft; the run launches only after d-jr1-band | the registered band itself (d-jr1-band); any in/out-of-band call; promotion of c-auto-default-venue-general |
| 6 | rd-runner11 -> b-pahier33-scorer (Branch D) | DO | Approved | runner-11 read-first; the PA-HIER-33 scorer build (convention already ratified, rows #278/#280) + the driver iiib fix; m-s0b-production behind g-byte-id and g-score-null green | the theta-pull interpretation (d-photoz-leverage); any instrument re-certification if g-score-null reds |
| 7 | v-falsifier-ii-classG (Branch E) | DO | Approved | the class-G fleet at the 40-60 CPU-h envelope (runbook 40 section 2), hard-capped at 60 | dropping A4's PROVISIONAL — d-a4-final-ratification returns with numbers, never auto-ratified (rows #278(4)/#280/#284(3)) |
| 8 | m-t5-armS + m-t5-armR (Branch F) | DO | Approved | Arm S launch (design ratified: rows #278(2)/#284(4)); Arm R launch strictly behind its own C0-prime-equivalent gate (row #284(4a)) | the window-adoption ruling (d-t5-window); any cross-venue generalization of either arm's result |
| 9 | r-completion-residual (Branch G) | DO | Approved | registration AUTHORING only, starting once rd-s3-readout delivers F; proposed arm cap <= 80 CPU-h is ORCHESTRATOR-DERIVED and part of the ask | the arm's band + cap ratification (d-completion-register); the launch; the attribution split (d-residual-attribution) |
| 10 | r-cone-loss (Branch H) | DO | Approved | registration authoring for the approx 17% cone-loss quantification (artifact section 09); proposed arm cap <= 20 CPU-h is ORCHESTRATOR-DERIVED | band + launch ratification (d-cone-register); the floor attribution |
| 11 | b-hprior-fix (Branch I) | DO | Approved | the config fix + rerun of the 14 failed G-EXT tasks (row #286), byte-identity below the old bound | any claim that the extended grid is load-bearing for a given arm — decided at that arm's registration |
| 12 | rd-rphi-note (closure) + the gate panel | DO + STANDING | Approved / Granted | the closure note as the first standing g-znorm evaluation on the flipped leg; STANDING: the section-2 panel evaluates before every science read in this graph — scope: this batch only; lapses at graph close | the board retirement itself (d-rphi-retire, returns with the note); every waives edge (per-instance RULE); any new instrument or band edit (physics-change + fresh RULE) |

Decisions that will RETURN to the author as this graph executes (none pre-granted above):
d-s4-review, d-jr1-band, d-completion-register, d-cone-register, d-calibration,
d-photoz-leverage, d-a4-final-ratification, d-t5-window, d-residual-attribution, d-rphi-retire,
and the three paper rulings d-paper-coverage / d-paper-1d2d-verdict / d-paper-massinfo.

---

## 4. The graph, drawn

Amber = author decide nodes, gold-heavy = terminal paper decides, red hexagons = gate
instruments (dotted = evaluates/STOPs), teal family = agent execution nodes, dashed-gold =
checkpoint (k-) nodes declaring fan-out caps before expansion (dotted = gates entry into the
capped nodes). d-s3-rerun (amber, DSR) is drawn as its own decide node gating Branch A's launch,
per section 3 row 2. Authorization edges from d-batch1-charter to every OTHER branch head are
drawn once (exemplar) to keep the figure legible.

~~~mermaid
graph TD
    CH[d-batch1-charter]:::decide

    subgraph BA [A calibration chain]
      A1[b-s4-repair]:::build --> A2[r-b82-s4]:::reg --> A3[m-s3-coverage]:::meas --> A4[rd-s3-readout]:::read
    end
    subgraph BB [B rebaseline]
      B1[m-head-rebaseline]:::meas
    end
    subgraph BC [C joint-r1 transfer]
      C1[dv-jr1-transform]:::derive --> C2[r-jr1-massaware]:::reg --> C3[m-jr1-massaware]:::meas
    end
    subgraph BD [D S0-B photo-z]
      D1[rd-runner11]:::read --> D2[b-pahier33]:::build --> D3[m-s0b-run]:::meas
    end
    subgraph BE [E falsifier ii]
      E1[v-falsifier-ii-classG]:::verify
    end
    subgraph BF [F T5 k-scan]
      F1[m-t5-armS]:::meas
      F2[m-t5-armR]:::meas
    end
    subgraph BG [G completion residual]
      G1[r-completion]:::reg --> G2[m-completion]:::meas
    end
    subgraph BH [H cone loss]
      H1[r-cone-loss]:::reg --> H2[m-cone-loss]:::meas
    end
    I1[b-hprior-fix]:::build
    J1[rd-rphi-note]:::read
    DSR[d-s3-rerun]:::decide

    K1[k-wave1-fanout]:::checkpoint
    K2[k-wave2-fanout]:::checkpoint
    KF[k-falsifier-ii-fleet]:::checkpoint
    K3[k-wave3-fanout]:::checkpoint

    CH -. authorizes .-> DSR
    DSR --> A1
    CH -. authorizes every other branch head .-> B1

    K1 -.-> A1
    K1 -.-> B1
    K2 -.-> A3
    K2 -.-> C3
    KF -.-> E1
    K2 -.-> KF
    K3 -.-> DCAL

    B1 --> A3
    B1 --> C3
    B1 --> E1
    B1 --> F1
    B1 --> F2
    I1 --> C2
    A4 --> G1
    B1 --> G1

    DCAL[d-calibration]:::decide
    DPZ[d-photoz-leverage]:::decide
    DA4[d-a4-ratification]:::decide
    DT5[d-t5-window]:::decide
    DRA[d-residual-attribution]:::decide
    DRP[d-rphi-retire]:::decide

    A4 --> DCAL
    B1 --> DCAL
    C3 --> DCAL
    D3 --> DPZ
    E1 --> DA4
    F1 --> DT5
    F2 --> DT5
    G2 --> DRA
    H2 --> DRA
    DCAL --> DRA
    DPZ --> DRA
    J1 --> DRP

    P1[d-paper-coverage]:::paper
    P2[d-paper-1d2d-verdict]:::paper
    P3[d-paper-massinfo]:::paper

    DCAL --> P1
    A4 --> P1
    DRA --> P2
    DA4 --> P2
    DT5 --> P2
    DCAL --> P3
    A4 --> P3

    GZ{{g-znorm}}:::gate -.-> C3
    GZ -.-> J1
    GS{{g-score-null}}:::gate -.-> D3
    GC{{g-sbc-coverage}}:::gate -.-> A3
    GP{{g-population}}:::gate -.-> A4
    GP -.-> G2
    GB{{g-byte-id}}:::gate -.-> A1
    GB -.-> D2
    GB -.-> I1
    G0{{g-c0-baseline}}:::gate -.-> B1
    G0 -.-> F2

    classDef decide fill:#8a6d00,stroke:#ffc857,color:#ffffff
    classDef paper fill:#7a4a00,stroke:#ffd166,stroke-width:3px,color:#ffffff
    classDef meas fill:#0e4f4f,stroke:#2dd4bf,color:#e6fffa
    classDef build fill:#123a5c,stroke:#60a5fa,color:#eff6ff
    classDef reg fill:#3b2b57,stroke:#a78bfa,color:#f5f3ff
    classDef read fill:#233042,stroke:#94a3b8,color:#f1f5f9
    classDef derive fill:#1f4d33,stroke:#4ade80,color:#f0fdf4
    classDef verify fill:#5b2333,stroke:#f472b6,color:#fff1f2
    classDef gate fill:#5c1616,stroke:#ff6b6b,color:#ffe4e4
    classDef checkpoint fill:#4a3300,stroke:#ffb020,color:#fff7e6,stroke-dasharray: 4 2
~~~

---

## 5. Cost envelope and 3-wave execution sketch

### 5.1 Cost envelope

| item | cost | source |
|---|---|---|
| m-s3-postflip-coverage | approx 12h + 4h wall (cells S + T at N=200), wall-limited not completion-limited; worker-parallelizable | row #288 / state candidate 1 |
| m-head-rebaseline | single-digit CPU-h (84 tasks x approx 6.5 min model) | state candidate 11 |
| m-s0b-production | cheap-to-moderate, comparable to the runner-11 8-cell precursor; cap <= 20 CPU-h ORCHESTRATOR-DERIVED | state candidate 2 |
| v-falsifier-ii-classG | 40-60 CPU-h, hard cap 60 | runbook 40 section 2 via state candidate 3 |
| m-t5-armS | 15-20 CPU-h | state candidate 6 |
| m-t5-armR | 11-15 CPU-h | state candidate 6 |
| m-joint-r1-mass-aware | 90-100 CPU-h (A18-equivalent grid; possibly less on the 41-node grid) | state candidate 10 |
| b-hprior-fix rerun | 14 tasks, a fraction of the approx 94 CPU-h original grid; bound <= 20 CPU-h ORCHESTRATOR-DERIVED | state candidate 8 |
| m-completion-residual | cap <= 80 CPU-h ORCHESTRATOR-DERIVED (unscoped in sources; the cap is the charter ask) | — |
| m-cone-loss | cap <= 20 CPU-h ORCHESTRATOR-DERIVED (unscoped in sources) | — |
| builds, registrations, reads, notes | agent time, negligible compute | — |

Envelope: sourced cluster items sum to approx 156-195 CPU-h; bounded small items add <= 50; the
two proposed residual-program caps add <= 100. TOTAL <= approx 345 CPU-h, plus the S3 re-run's
approx 16 wall-hours across two cells (its CPU-h is not separately sourced; it parallelizes across
cluster workers). Nothing in this graph approaches the scale of a single historical production
grid (approx 94 CPU-h, state candidate 8) except the joint_r1 arm, which is the price of venue
generality.

### 5.2 Three waves, with per-phase tiering (routing table of record: CLAUDE.md)

Fan-out and tiering caps for each wave are node attributes of k-wave1-fanout, k-wave2-fanout,
k-falsifier-ii-fleet, and k-wave3-fanout (section 1.14) — restated narratively below for reading
convenience only; the k- node table is the source of record, not this prose.

Wave 1 — enablement (no science reads). Nodes: b-s4-harness-repair, r-b82-s4, m-head-rebaseline,
rd-runner11, b-pahier33-scorer, b-hprior-fix, rd-rphi-note, dv-jr1-transform + r-jr1-massaware
draft, and the m-t5-armS launch (already design-ratified, row #284(4)), gated by k-wave1-fanout.
Everything else sonnet low/medium. Returning RULEs at wave end: d-s4-review, d-jr1-band,
d-rphi-retire.

Wave 2 — production measures. Nodes: m-s3-postflip-coverage, m-s0b-production,
v-falsifier-ii-classG (gated by k-falsifier-ii-fleet), m-t5-armR (behind its C0-prime-equivalent),
m-joint-r1-mass-aware (behind d-jr1-band); r-completion-residual and r-cone-loss authoring start
once rd-s3-readout delivers F. Gated by k-wave2-fanout. Agent-side panels all sonnet. Returning
RULEs at wave end: d-completion-register, d-cone-register.

Wave 3 — convergence and synthesis. Nodes: rd-s3-readout (if not closed in wave 2),
m-completion-residual, m-cone-loss, then the decide cascade as manifests fill: d-calibration,
d-photoz-leverage, d-a4-final-ratification, d-t5-window, d-residual-attribution, and the three
terminal paper rulings. Gated by k-wave3-fanout. All dossier assembly and readout mechanics
sonnet. Honest spill risk: if the completion arm needs its revision 2, d-residual-attribution and
d-paper-1d2d-verdict spill to the next graph — bounded by max_revisions, surfaced as Hold, never
silently extended.

Batch-cumulative top-tier headcount (CLAUDE.md's cap is stated per workflow; graph1 is ratified as
one workflow at one scope hash, so the relevant count is cumulative across waves, not the largest
per-wave number in isolation). Read literally, the k- node attributes name up to five distinct
top-tier roles across the batch: the chair (persists across all three waves by design — one
identity); the wave-1 derivation/prereg author (dv-jr1-transform, r-jr1-massaware draft,
r-b82-s4); the wave-2 prereg author (r-completion-residual, r-cone-loss); the wave-2 decisive
verifier (the falsifier-ii adjudicating read); and the wave-3 end-verifier. If each of the four
non-chair roles is staffed by a distinct agent, the batch runs 5 top-tier identities against a cap
of ~3 per workflow — over cap. This proposal's ask is that the wave-1 derivation/prereg author ALSO
serves as the wave-3 end-verifier (re-deriving numbers they did not themselves author in waves 2-3,
preserving the "verifier output is evidence, not authority" independence rule since the specific
falsifier-ii read is re-derived by the separate wave-2 decisive verifier, not by this reused
identity), collapsing the count to 4 (chair + prereg/end-verifier + wave-2 prereg author + wave-2
decisive verifier). This is still one over the stated ~3 cap and is flagged here rather than
asserted compliant; ratifying row 0 does not by itself waive the cap; a one-line author call on
whether 4 is accepted for this batch, or a further identity merge is required, is requested
alongside charter ratification.

Every wave: cluster launches behind /cluster preflight VERDICT: READY; no agent ends a turn
waiting on an untracked process (CLAUDE.md standing rule); every node writes exactly one record
file (write isolation, infra 3.8); crashed runs still write cost records (infra 3.5).

---

## REVISION 1 (2026-09-01, applying GRAPH1_CRITIC_NOTES_20260901.md)

MUST-FIX 1: split the STANDING approval-scope item out of row 0. Section 0's schema-of-record
paragraph no longer folds infra 3.4 into "the meaning of whole-graph ratification"; section 3 now
carries it as its own row 1 (tag STANDING, ask "Granted", scope "graph batches in this project,
this campaign", lapse "campaign end"), and row 0's triggers/NOT-covered cells say explicitly that
ratifying row 0 does not grant row 1. All other decision-table rows renumbered 2-12 (formerly
1-11); cross-references to "row 11" (panel meta-rules) and "row 1"/"row 2" (intro, section-4
legend) updated to match.

MUST-FIX 2: the four register nodes carrying max_revisions 2 (r-b82-s4 section 1.1, r-jr1-massaware
section 1.3, r-completion-residual section 1.7, r-cone-loss section 1.8) now tag the value
ORCHESTRATOR-DERIVED with a one-sentence derivation ("provisional default, ratified with the
charter: two revision attempts is the smallest budget that lets a fixable design flaw survive one
round-trip without paying full SETTLED credit for a first-attempt park"); section 1.13 restates the
tag for all four.

MUST-FIX 3: d-s3-rerun is now a real node — added to section 1.1's Branch A table (type decide/RULE,
requires-manifest rows #286/#288, gating b-s4-harness-repair's authorized-by edge), added to section
1.11's convergence decide-node list, and drawn in the section-4 mermaid as DSR with CH -.-> DSR and
DSR --> A1 edges.

MUST-FIX 4: added section 1.14, instantiating four k- checkpoint nodes (k-wave1-fanout,
k-falsifier-ii-fleet, k-wave2-fanout, k-wave3-fanout) with fanout_cap/tier/cost-cap as node
attributes; section 5.2's prose now points to these nodes as the source of record instead of
asserting the numbers fresh; the mermaid gained K1/K2/K3/KF checkpoint nodes (dashed-gold class)
gating their wave's representative launch nodes.

SHOULD 7: every "commit 5e7fda16" citation (question layer, Branch B, closure chain) now notes the
hash is narrated at row #286 but literally quoted only at row #288.

SHOULD 5 (applied, not just noted): section 5.2 now computes the batch-cumulative top-tier
headcount across all three waves (up to 5 distinct roles read literally; proposes collapsing the
wave-1 derivation/prereg author into the wave-3 end-verifier role to reach 4, one over the ~3 cap,
and asks the author to rule on the remainder explicitly rather than assuming row-0 ratification
covers it).

NOTE 9 (applied): the no_bh F=7.426 first use (section 1.1) now cites pilot record section 3.1 and
notes row #288's own precision (F=7.43) is the same measurement, not a discrepancy.

Left as-is (not one-line fixes, not requested for this revision):
- SHOULD 6 (rd- nodes + feeds edges with sha256 for the three terminal paper decide-nodes) —
  requires real artifact paths and checksums, a structural addition beyond a wording fix.
- SHOULD 8 (a worked acquisition-scoring example against the 11 state candidates) — requires an
  actual computation, not an edit to this document.
- NOTE 10 (v- node max_revisions asymmetry) — applied instead, in
  INFRA_AUTONOMOUS_RESEARCH_PROPOSAL_20260901.md section 2.4 (see that file's own REVISION 1 note).
