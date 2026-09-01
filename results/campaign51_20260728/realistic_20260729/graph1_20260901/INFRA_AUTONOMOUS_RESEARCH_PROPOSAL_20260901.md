# Infrastructure Proposal: From Trees to a Typed Directed Research Graph with a Standing Gate Panel

Date: 2026-09-01. Author of record for all scientific decisions: Jasper Seehofer.
Status: PROPOSAL (a reviewable decision artifact per CLAUDE.md "Proposing decisions"; nothing here is adopted until ratified).
Scope: the orchestration infrastructure for autonomous research batches in this project and its successors. No physics content changes proposed here; where the proposal touches gate formulas it routes through /physics-change as usual.

Inputs of record: (i) garden-vault extract of wiki/analyses/atomic-research-structure-spec.md ("vault"); (ii) practice-mining report over ledger rows #221-#288 ("practice (a)/(b)/(c)"); (iii) external survey 1, autonomous-science systems ("ext1"); (iv) external survey 2, workflow structures, on disk at results/campaign51_20260728/realistic_20260729/graph1_20260901/external_research_2_workflow_structures.md ("ext2"); (v) the reader-node state record STATE_AND_CANDIDATES_20260901.md in this directory ("state").

---

## 1. Diagnosis: what the tree model earned, and where it chafes

### 1.1 What demonstrably worked (keep all of it — these become graph invariants)

Every mechanism below caught a real error in rows #221-#288. The graph design in section 2 hard-codes each one as a structural feature rather than a discipline.

1. Pre-registered bands frozen before data: B7.3's C4 counterfactual landed cleanly inside its pre-bounded IMMATERIAL band (row #248), and B5.2's C3 fell in NEITHER band and was blocked from adoption instead of waved through (row #247). Bands are the single highest-value habit on record.
2. Byte-identity gates: B5.1's new flags proven identical-by-default with 100,000 pairs, 0 mismatches, before any counterfactual was trusted (row #229).
3. Adversarial panels that round-trip must-fixes: the T5.1 window proposal REFUTED at round 2 for a stale premise, re-checked from source by a second refuter (rows #270-272).
4. The approval-scope rule with teeth: a builder DECLINED a literal "implement exactly as presented" dispatch because the cited presentation's own section 13 said STOP (row #234). This rule works even against the orchestrator.
5. REPORTED-ONLY structural caps: a p=0.0358 near-band read stayed a report, not an H0 claim (row #226).
6. Builder/runner/reader independence: the three-agent split caught a driver gap BEFORE wasted compute (row #260).
7. Sharp identities as discriminators: the Z(h)=1 one-density-everywhere criterion ruled out BOTH branches of the row #167 impostor-weight fork as non-derived, and showed the mass-aware 1D leg moots the question — Z=1 by identity (row #269).
8. Score-at-truth nulls: the S0-A mirror null returned Z_b=-3.68 / Z_s=-7.08 and correctly flagged INSTRUMENT-DEFECT, seeding the whole T1 chain (rows #225, #251).
9. Richardson bias-free falsifiers: settled PA-HIER-33 among three conventions with fresh data, refuting two at ~34.5 and ~19.5 sigma (row #275).
10. Bit-identical C0 baselines: a banked comparand certified identical across four intervening commits, enabling zero-compute delta reads (rows #246, #281).
11. Registered falsifiers with a stated disposition rule: the S_4D-homogeneity falsifier discriminated twin (invariant to 1e-16) from coded form (not invariant) and caught the synthetic defect (row #236).

### 1.2 Where the structure — not the people or the models — failed

Each failure mode below maps to a missing structural feature, named in brackets; section 2 supplies it.

1. Itemization deviations from source: row #268's itemization invented a cost figure with no source on disk and silently dropped the second condition of a registered rule (TREE2_DECISIONS section 2). [Missing: machine-checkable feeds edges — every quoted number carries its source artifact and checksum.]
2. Corrections that mis-cite: row #258's citation fix was itself wrong by commit-relative line drift. [Missing: artifact-addressed, slug-stable references instead of line numbers.]
3. Verdict-file write races: 41 of 42 parallel verdicts lost to a shared-file race (row #280). [Missing: per-node write isolation as a schema law, not a post-hoc fix.]
4. Subagent parking on untracked waits: five incidents, three agents, one session (CLAUDE.md standing rule). [Missing: node states tracked by the harness, so waiting is a graph state, not an agent behavior.]
5. Stale comparands declared settled: a provenance JSON's own tree_dirty_file_count=296 contradicted the "tree clean" stamp, and a prior docket had called it resolved with no agent having checked (row #247/D12, F5). [Missing: convergence-node eligibility computed mechanically from input existence + freshness, never from prose.]
6. Storage-precision artifact masquerading as a physics fork: +157.92 vs +123.11 nats, 35 nats apart, ultimately catastrophic cancellation in a 7-significant-figure JSON field (rows #280 sec.3 item 6, #282). [Missing: a serialization-precision gate instrument.]
7. Censoring understated: "rails in 3 of 4 seeds" was actually 4 of 4 (rows #267, #280). [Missing: censoring flags emitted by gates, not narrated by readers.]
8. Mixed-N aggregate contamination: 3 ladder-costing seeds pooled with 63 real seeds into one statistic (row #288). [Missing: population-typed artifacts + an aggregation purity lint.]
9. SSH failure read as "does not exist" (row #288). [Missing: three-valued existence — present / absent / unreachable — in every reader contract.]
10. Symlink-dance leakage across concurrent runs (row #286, runbook 40 sec.5). [Missing: declared shared-mutable-state on nodes; the scheduler forbids co-scheduling nodes that share it.]
11. Dead parallelism and unbanked crashed compute (rows #245, #258, D11). [Missing: the cost ledger as automatic node accounting — a node that ran and crashed still writes its cost record.]

### 1.3 Where the linear tree specifically chafes

- Cross-branch dependencies had to be bolted on informally (practice (c) item 1). The current candidate list is the proof: candidate 4 (completion-leg program) needs candidate 1's F measurement AND candidate 11's re-baseline; candidate 9 (paper) "draws on nearly everything else on this list" (state, sec.2). A tree cannot represent a diamond; the paper node has in-degree ~7.
- Session state is reconstructed by hand every time: the memory index's "Entry point: runbook N" convention is a hand-rolled frontier pointer; each fresh session re-derives graph state from prose. That is a graph being simulated in narrative, at cost.
- Two concurrent prose specs diverged within hours in the vault's own history (the M6/M7 twin-touchpoint-table drift, Gate-4 record) — narrative duplication does not stay consistent even for a day. One canonical machine-readable structure is the fix, not better prose discipline.
- The author's requirement names it directly: research cycles and branches CONVERGING on shared decisions that need inputs from several branches. That is a directed graph with typed convergence nodes, full stop.

---

## 2. The proposed model: a typed directed research graph

Two layers, deliberately distinct (ext2 sec.1: workflow engines conflate task=node because they have no claim layer; we must not):

- The EXECUTION layer: nodes that do work (register, build, measure, derive, verify, read, checkpoint, subgraph) and the decisions that govern them (gate evaluations, decide nodes).
- The CLAIM layer: question and claim nodes whose status is only ever written by gate/decide events.

### 2.1 Node types (12, each with a slug prefix)

| Prefix | Type | What it is | Key fields |
|---|---|---|---|
| q- | question | what is to be settled | kill_criterion (mandatory: a branch without a kill condition is not tracked — vault, adopted) |
| c- | claim | a claim-bearing statement | status: conjectured/supported/verified/refuted; confidence; regime (validity window); both gate-written only |
| r- | register | a pre-registration | bands (frozen), disposition rule, cost estimate, max_revisions, deviation policy. Its own gate checks DESIGN VALIDITY ONLY, blind to results (ext2 sec.2, Registered Reports Stage 1) |
| b- | build | instrument/tool construction | byte-identity proof obligations (row #229 pattern) |
| m- | measure | a run/computation | registered-by edge mandatory to feed any decide; outputs carry gate-panel stamps and a population type |
| dv- | derive | an analytic derivation | routes through /physics-change when it touches trigger files |
| v- | verify | an adversarial refutation attempt | MUST emit a structured artifact either way — counterexample or survival record, never a bare boolean (ext2 sec.4); never executed by the builder (row #260) |
| g- | gate | one EVALUATION of a standing gate instrument against a run | outcome enum: Go / Kill / Hold / Recycle / Conditional-Go (ext2 sec.6) |
| d- | decide | an author ruling; THE convergence node type | tag: DO / RULE / STANDING; requires-manifest; eligibility computed mechanically |
| k- | checkpoint | dynamic expansion point | fanout_cap, tier, per-child cost — all declared BEFORE expansion (CLAUDE.md tiering mandate, formalized per ext2 sec.1 Snakemake checkpoints) |
| s- | subgraph | a whole runbook/campaign as one opaque node | keeps the top-level roadmap small (ext2 sec.1 subworkflows) |
| rd- | read | state collection, verdict-free | three-valued existence contract: present/absent/unreachable (fix for row #288). STATE_AND_CANDIDATES_20260901.md is already an rd- node in all but format |

### 2.2 Edge kinds (semantics, not arrows)

Three families, because "data dependency", "authorization", and "falsification" are different relations that the tree model kept collapsing into sequencing:

DATA edges
- feeds: artifact dependency. Carries the artifact path AND checksum. If the upstream artifact changes, the edge is invalidated even if the downstream node "ran" (ext2 sec.1; this is the dataset-pinning rule generalized, and the structural fix for row #247/D12 stale comparands and row #268 sourceless itemizations).
- registered-by: measure -> its register node. Any divergence is a deviates-from edge with a written justification — deviations become searchable objects, not buried prose (ext2 sec.2, Stage-2 adherence review).

AUTHORIZATION edges
- authorized-by: node -> decide node, carrying the tag (DO/RULE/STANDING) and a scope hash — the hash of the graph as it stood when the grant was given. THE LAW: an authorization never covers a node whose inputs postdate its scope hash (CLAUDE.md binding default; row #234 is the proof it must be mechanical, since it already works when merely social).
- waives: decide -> a red gate evaluation. The ONLY way a red gate stops blocking; always an explicit author RULE.

EPISTEMIC edges
- verifies / refutes: verify -> claim. A refutes edge triggers the revision protocol (2.4).
- discriminates: a verify or measure -> two or more claims it can separate (vault, adopted verbatim: the generalization of multiple working hypotheses; input to acquisition scoring, 2.6).
- spawned-by / supersedes: lineage, always backward in time; slugs are immutable and superseded nodes retire in place (vault, adopted).

### 2.3 Convergence nodes

A d- node carries a requires-manifest: a typed list of inputs, each naming the source node, the artifact expected, and the state required (e.g. "done with green panel", "band disposition assigned"). Eligibility is computed from the manifest — a decide node whose manifest is unsatisfied cannot be put to the author, and one whose manifest IS satisfied surfaces automatically. This kills two observed failures at once: decisions "resolved" that no one checked (row #247/D12), and decisions the author was asked to make before their inputs existed (the binding default exists precisely because this kept happening).

Worked example from the live board: d-a4-final-ratification requires the falsifier (ii) class-G fleet verdict (state, candidate 3: "returns with numbers, not auto-ratified") — in graph form the A4 RULE is simply ineligible until v-falsifier-ii-classG is done, and then surfaces with the numbers attached.

### 2.4 Cycle handling: bounded re-entry without back-edges

Every serious workflow engine is strictly acyclic and handles iteration by re-instantiation (ext2 sec.1). We adopt that, because it is also the audit-honest choice — the existing convention of appending ledger rows rather than mutating them is the same principle, and the graph should state it as law:

1. A refutes edge landing on a claim never deletes anything. The claim keeps its slug, its status moves to refuted (gate-written), lifecycle to retired if superseded.
2. Re-opening = a NEW register node, spawned-by the refuted line, with revision: n+1. The register node's max_revisions (frozen at charter time) bounds the loop; exceeding it forces gate outcome Hold and a fresh author RULE. This is the formal home for the five-way outcome Recycle: "back to pre-registration with a revised plan" is a named, counted move, not prose.
3. Checkpoint (k-) nodes handle the other dynamic case — fan-out counts unknown until an upstream output exists — with the cap declared before expansion.
4. Noted asymmetry: this bounded re-entry law caps repeat attempts only at register (r-) nodes. A verify (v-) node whose survival record is itself disputed has no stated cap on repeat verification rounds; in the current design this fails safe (the downstream decide node simply stays permanently blocked, never causing runaway compute), but it is not yet a symmetric claim and is flagged here rather than left implicit.

### 2.5 THE GATE LAYER: a standing instrument panel, evaluated before any science read

This is the author's sharpest requirement and the largest single change. Gate instruments stop being things we remember to run and become a panel that is always on.

Definition: a gate INSTRUMENT is a registered, frozen, sharp calibration measurement with (a) a definition, (b) a registered band, (c) an evaluation trigger, (d) an append-only history file (the panel), and (e) a stated question-class: MACHINERY (is the inference engine right), MODEL-GIVEN-DATA (does the model fit this data), IDENTITY (does an exact mathematical identity hold), or BASELINE (is the comparand what we think it is). The machinery/model split is the SBC-vs-PPC distinction, which the Bayesian-workflow literature insists are sequential and non-interchangeable (ext2 sec.3): a single "checks passed" conflates broken sampler with wrong model.

THE PANEL LAW: no rd- or d- node may consume a measure artifact that lacks a green (or explicitly waived) panel stamp. Gates run BEFORE the science read, structurally — a red gate does not suppress the number (the number is always banked; see 3.2), it blocks INTERPRETATION until waived or fixed.

The worked example — Z(h)=1, told as the counterfactual it is:
- The instrument: the integral of the likelihood over data space equals 1 at every h-node — one density everywhere, no h-dependent leakage of normalization into the posterior shape. An IDENTITY-class gate: the pass band is a numerical tolerance on an exact statement, not a p-value.
- What actually happened: the criterion was deployed late, inside the A11 derivation, where it ruled out both branches of the impostor-weight fork as non-derived and revealed that the mass-aware 1D leg satisfies Z=1 by identity (row #269).
- The counterfactual: the mass-blind catalogue leg ran for weeks with a mismatched numerator/divisor whose measured signature was r_phi = 0.886 — a number a standing Z(h)=1 instrument would have painted red on day one, at near-zero compute. The flip (commit 5e7fda16, row #286) now moots r_phi by construction (state, candidate 7). The author's own verdict — we should have used it from the beginning — is the design requirement: sharp identities go on the panel at t=0, not into forensics at t=weeks.

Proposed instrument catalogue v0 (each with class, sharp question, band sketch, evidence):

| Instrument | Class | Sharp question | Band sketch | Evidence |
|---|---|---|---|---|
| g-znorm | identity | Z(h)=1 at every h-node of every production likelihood | abs dev <= 1e-6 green; > 1e-3 anomalous | row #269; r_phi counterfactual |
| g-score-null | machinery | mean score at truth-theta = 0 (first Bartlett identity) | abs Z <= 3 | rows #225/#251 caught a defect; row #287 certified both axes with exactly this band |
| g-fisher-pull | machinery | var(score) vs -E[Hessian]; pulls (est-truth)/sigma ~ N(0,1) | registered per venue; convention frozen by the Richardson adjudication | row #275 |
| g-sbc-coverage | machinery | HPD coverage at 50/68/90/95 + PIT-KS on synthetic universes | registered bands per cell; population-pure input only | row #288 fired correctly (and exposed the mixed-N contamination); pp_coverage.py already exists — promote it from side-tool to standing upstream gate (ext2 sec.3) |
| g-ppc | model-given-data | posterior predictive vs the actually observed data; posterior-SBC style conditional check for the photo-z-starved regime | to be registered | ext2 sec.3 (arXiv:2502.03279) |
| g-byte-id | baseline | new flag/refactor is byte-identical on the default path | 0 mismatches, N >= 1e5 pairs | row #229 |
| g-c0-baseline | baseline | the banked comparand is bit-identical to what it claims to be | max_abs = 0 on shared columns; md5 match | rows #246/#281 |
| g-invariance | identity | invariance under rescaling / reparametrization the math promises | dev <= 1e-12 (measured 1e-16 when true) | row #236 |
| g-closure | identity | decomposition terms sum to the total they decompose | registered residual band | the -0.1083 = impostor + tilt + matched decomposition (memory, rows #149-#156) |
| g-dim-limit | identity | dimensional analysis + limiting cases: dist(z=0)=0, PSD > 0, p_det in [0,1], scalar_product(h,h) > 0 | exact | CLAUDE.md test-priority list, promoted to panel |
| g-precision | baseline | no decision-bearing number rides a lossy serialization; cancellation sentinel on log-sums | full-precision pinning; flag any read within 2 s.f. of cancellation | rows #280 sec.3 item 6 / #282 (35-nat storage artifact) |
| g-population | baseline | every aggregate's inputs are one declared population | 0 mixed rows | row #288 |
| g-censoring | model-given-data | rails/censoring fraction at grid edges, reported AS a gate output | registered per grid; any MAP at a rail flags the read as a bound | rows #267/#280 (3-of-4 was really 4-of-4) |

Panel meta-rules:
- Instrument definitions are FROZEN within a batch. Editing one is a physics change (formula) plus a fresh RULE (band) — never something the executing agent can touch (see 3.6).
- Every evaluation appends to the instrument's history file — the panel is a time series, so drift is visible, exactly like PHYSICS-GATE-LEDGER.md makes compliance evidence rather than inference.
- Every gate evaluation states its question-class, so a red never gets answered by fixing the wrong layer.

### 2.6 Acquisition: choosing the next batch is a scored, inspectable node

The DoE/active-learning triad — belief state, acquisition function, experiment (ext2 sec.5) — is already half-implemented here by hand: the state file's candidate list with cost bands, inputs, and shared-decision flags IS an acquisition input. Formalize: an rd- belief-state node summarizes live claims and open questions; an acquisition record scores candidates by discriminating power (count and weight of discriminates edges — the vault's EIG idea, adopted) against declared cost (probe_cost, adopted), and TAGS each advance explore or exploit. The tag makes stop-vs-continue auditable against a stated intent instead of retroactive narrative. The author ratifies the batch charter, not each pick — within the ratified scoring rule.

---

## 3. Anti-derailment by construction

The external record is blunt about how autonomous loops go wrong: Sakana's v1 edited its own timeout and spawned unbounded processes; the AI CUDA Engineer exploited its own evaluation harness and reported fabricated speedups; internal tournaments select for what LOOKS well-argued, which is a different axis from what is TRUE (ext1). Every binding below is placed where one of those arrows — or one of ours — actually landed.

### 3.1 Bands frozen before data, with the neither-band branch wired
Every register node freezes bands and a disposition rule before any run (rows #247/#248). A result in neither band forces disposition INTERMEDIATE: no adoption, automatic return as a fresh RULE. The graph makes the row #247 save the default path instead of a good catch.

### 3.2 Refuted and undetermined are valued outcomes — stated as the objective function
Explicit objective, to be ratified as part of the charter: the system's score for a batch is the number of registered questions moved to a SETTLED state — verified, refuted, or bounded-undetermined — with all consumed panels green or waived. Refuted pays the same as verified. Bias reduction is NOT in the objective; it is a finding, not a goal (author's binding 2026-08-05 value: correctness and novel insight outrank eliminating the H0 bias). And the Registered-Reports IPA mechanism (ext2 sec.2) is adopted as a graph edge property: once a register node's design gate passes, the resulting number is banked and reported regardless of direction — a can't-un-approve-post-hoc guarantee. Boring results cannot be closed by convenience, structurally.

### 3.3 Adversarial verification tiers, with independence as topology
Builder, runner, and reader of any decisive node are distinct agents (row #260 caught a defect before compute); verify nodes re-check from source, not from the proposer's summary (rows #270-272); decisive numbers from any subagent are re-derived before a decide node consumes them (memory: verifier output is evidence, not authority — now an edge rule: a v- artifact feeding a d- node must carry its re-derivation stamp). Tiering per the standing mandate: panels run sonnet, at most ~3 top-tier agents per workflow, fan-outs computed before launch — now enforced by k- node caps rather than launch-summary prose.

### 3.4 Approval-scope semantics on the graph — what whole-graph ratification means
When the author ratifies a batch charter, the grant covers exactly: the frozen topology (nodes + edges at the scope hash), the caps, the registered bands, and the gate-instrument set. It NEVER covers: the disposition of any comparison not yet computed, any band-edge or neither-band call, any promotion of a claim's status, or any node added after the hash. All of those surface as fresh RULE decide-nodes — mechanically, because their authorized-by edge fails the scope-hash check (2.2). This is the existing binding default made executable; row #234 shows the rule already works socially, so encoding it costs nothing and closes the gap where an eager batch would have stretched a grant.

### 3.5 Caps propagate along edges
Cost, fan-out, and model-tier caps declared at k- nodes are inherited by everything they expand. An expansion that would exceed its cap does not run; it parks the branch at Hold and surfaces a decide node. Crashed and partial runs still write cost records (fix for the ~8.6 unbanked CPU-h, row #258/D11). Runaway spawning of the Sakana kind is then not a behavior to detect but a state the scheduler cannot reach.

### 3.6 Harness integrity: the agent never grades its own homework
The two documented external disasters are both harness attacks (ext1: timeout self-edit; evaluation-bypass "reward hacking against benchmark code"). Bindings: (a) gate-instrument code, band registries, the graph schema, and the linter are OUTSIDE the write scope of any node-executing agent — they sit behind the same trigger-file discipline as physics files, and editing them is a physics-change plus fresh RULE; (b) gate evaluations are computed by an agent that holds no stake in the node's outcome (the goal-holder never computes its own green); (c) a v- node's survival record must name what it tried, so an empty refutation attempt is visible as such (tournament-winning prose is not evidence — ext1's critique of internal tournaments, and ours).

### 3.7 Structural caps on statistical promotion
REPORTED-ONLY stays a first-class disposition (row #226). Adopted from the vault as a lint law: a claim whose provenance is AGENT-only may never reach status verified — some human-checked, document-anchored, or literature-anchored input must be on its evidence path (vault L12; identical in spirit to the re-derivation rule in 3.3).

### 3.8 Write isolation as schema law
Every node writes exactly one record file of its own; graph state is the directory, aggregation is read-only (the row #280 race, 41 of 42 verdicts lost, adopted as prevention rather than repair).

---

## 4. Critical assessment: the atomic-structure-of-research idea

Verdict: ADAPT the skeleton; ADOPT five specific disciplines verbatim; REJECT wholesale adoption and REJECT running it as a parallel second store. Reasons, all from the record:

ADOPT verbatim (these independently converge with what our practice already earned):
1. Immutable slugs + retire-in-place + spawned-by lineage ("renames are forbidden") — precisely the append-only ledger convention that made rows #267/#280-style corrections traceable; it is what makes node ids safe as primary keys for the paper, the panel, and any later DB.
2. Gate-written status/confidence (vault L8: changing either without a gate bump is a lint error) — matches "bands frozen before data" and blocks HARKing at the schema level.
3. Provenance vocabulary + the L12 cap ([AGENT]-only never reaches verified) — the vault arrived at our "verifier output is evidence, not authority" memory from the other direction; convergent evolution is good evidence both are right.
4. kill_criterion, probe_cost, regime (the three Gate-4 additions) — kill_criterion is our disposition rule, probe_cost is our fan-out-cost-before-launch mandate, regime is our per-venue validity scoping (the iiib band does not transfer to joint_r1 — state, candidate 10). All three were adopted there for reasons we independently learned here.
5. Felt-need DB triggers: files first, database only at ~500 nodes or when grep fails. Correct for us too; section 6 is files.

ADAPT (right shape, wrong taxonomy for this use):
- The six node types (question/tool/exploration/answer/derivation/verification) describe a KNOWLEDGE graph — claims and their relations. What rows #221-#288 show failing is the EXECUTION layer: authorization scope, convergence eligibility, gate stamps, caps, write isolation. The vault taxonomy has no register, no gate, no decide, no checkpoint — the four types where every observed save or failure actually lived. Section 2's taxonomy embeds their claim layer (question/claim/derive/verify map cleanly) inside an execution layer they lack.
- Its status enum (conjectured/supported/verified/refuted) is a claim property; gate outcomes need the five-way Go/Kill/Hold/Recycle/Conditional-Go enum (ext2 sec.6). Keep both, on different node types — the vault's own recorded self-critique (status semantics on non-claim-bearing types "to be tightened in v0.1") says the same thing.
- The living-book consumer contract (a chapter is a traversal from a root question) adapts directly to candidate 9: the paper as a traversal of this graph in topological order. Worth keeping as a named consumer from day one.

REJECT, with reasons:
- Wholesale adoption: the spec has never been implemented or stress-tested; its own open-questions section (home repo unresolved, lifecycle semantics unresolved, hypothesis-ledger relationship unresolved) is the honest caveat, and no independent critique of it exists on record. We do not bet a production campaign on an untested schema; we implement the merged schema in-repo (section 6) and feed lessons back to the vault spec as its first real stress test.
- A parallel second store: the one concrete near-miss in the vault's own history is the twin-touchpoint-table drift — two concurrent specs, written hours apart, diverged and had to be deduplicated (Gate-4 record). Running an atlas beside runbooks reproduces exactly that failure class at project scale. ONE canonical graph store; runbooks become generated views of it, then retire.
- Caveat on completeness: the vault extract supplied to this proposal truncates mid-sentence at "A parallel, competing recomme..." — a competing recommendation exists in the vault that this assessment has not seen. Flag: retrieve it before ratifying this section. (Stated rather than guessed, per the itemization-deviation lesson of row #268.)

---

## 5. Think-ahead roadmap

### Horizon 1 — this project, next month (graph1 pilot)
1. Stand up the schema of section 6 in this directory: encode the state file's 11 candidates as nodes (they already carry inputs, costs, gates, and shared-decision flags — the translation is mechanical, a sonnet task).
2. Stand up panel v0 with the instruments that already exist as code or habit: g-znorm, g-score-null, g-sbc-coverage (pp_coverage.py), g-byte-id, g-c0-baseline, g-dim-limit, g-population, g-censoring. Wire the PANEL LAW into the batch runner: no science read without stamps.
3. Ratify the first graph charter (topology + caps + bands + instrument set at a scope hash) and run the S3-postflip / S0-B / falsifier-ii / T5 wave as the first graph-native batch, with d-a4-final-ratification and d-s3-rerun (row #288 item (d)'s fresh RULE) as the first mechanically-surfaced convergence nodes.
4. Deliverables: the graph directory, a ~12-rule linter (section 6.4), the panel report, and a one-page charter artifact for ratification. Runbook 41 is written as a generated view of the graph — the experiment that decides whether runbooks can retire.

### Horizon 2 — the reusable engine for future EMRI/LISA projects
Efficient realistic-venue campaigns are a first-class deliverable of this project (author, 2026-08-12). The graph engine is that deliverable's skeleton:
1. Extract venue-agnostic parts into a package: schema + linter, the gate-instrument registry with the Bayesian-inference catalogue (2.5) as a library any inference project instantiates, the charter/scope-hash tooling, and a compiler from k-/m- nodes to cluster job arrays (Snakemake-checkpoint semantics, ext2 sec.1 — our fan-out pattern is hand-rolling what checkpoints automate).
2. Cross-campaign claim registry: immutable slugs let paper N+1 cite paper N's claim nodes directly — the beginning of a lab-internal citation graph in which a refutation in a later campaign mechanically flags every downstream consumer in earlier ones.
3. The acquisition layer becomes the campaign scheduler: belief-state read, scored candidate list, explore/exploit tags, author-ratified batch — the shape FutureHouse assigns to dedicated agents by scientific function (ext1) mapped onto our research-cycle stages.

### Horizon 3 — the speculative far end: self-improving, value-anchored
What a system that improves itself without derailing looks like, given everything above:
- The graph IS the belief state; batches are proposed by acquisition against it; the panel is the immune system that runs before every read. The loop closes on gates and author rulings, never on self-assessment (the Google co-scientist lesson, ext1: tournaments select for looking right; only held-out reality checks select for being right — our held-out reality is the panel plus the author).
- Self-improvement is confined to three channels, each gated: (a) PROPOSING new gate instruments from recurring forensic patterns (a repeated failure class becomes a candidate instrument — e.g. rows #280/#282 would have auto-proposed g-precision), adopted only by author RULE; (b) tuning acquisition weights against retrospective regret, measured against the author's actual rulings — the system learns to predict the author's scientific taste, never to replace it; (c) compiling recurring must-fixes into lint rules, same gate.
- NEVER automated, as a charter constant: the objective function and values (3.2); gate-instrument definitions and bands (3.6); promotion past the L12 provenance cap; spending caps; the physics-change protocol; and the choice of which questions matter. The RULE class never shrinks by automation — only the author may reclassify a RULE pattern to STANDING, explicitly, with a stated lapse condition.
- Where the human sits: at charter ratification (batch boundaries), at every convergence RULE, at every waives edge, and at every amendment to the constants above. The human moves from in-the-loop per decision to on-the-graph per batch — and the graph's own scope-hash law guarantees that anything genuinely new walks back to the human by construction, because its authorization check fails. That is the property that makes MORE autonomy safe rather than riskier: the derail paths are not discouraged; they are unreachable states.

---

## 6. Minimal concrete schema (adoptable now)

Files, not a database (vault D5, adopted). One YAML record per node (write isolation, 3.8). Directory layout under this campaign:

    graph/
      charter.yaml          frozen scope hash, caps, objective, ratification record
      nodes/                one file per node, named by slug
      gates/instruments/    one file per standing instrument
      gates/panel/          append-only per-instrument history (csv)
      ledger/costs.csv      automatic per-node cost records, crashes included

### 6.1 Node record shape

    id: m-s3-postflip-coverage
    type: measure
    title: S3 coverage re-run post-flip, cells S and T, N=200
    status: blocked            # blocked | eligible | running | done | retired
    revision: 1
    max_revisions: 2
    population: n200-postflip  # g-population lints aggregates on this
    shared_state: [repo-root simulations symlink]   # scheduler forbids co-runs (row 286)
    cost: {cpu_h_est: 16, tier: sonnet, fanout_cap: 100}
    kill_criterion: coverage unusable post-flip in both channels at registered bands
    edges:
      - {kind: registered-by, to: r-b82-s4, frozen: 2026-09-01}
      - {kind: feeds, from: m-head-rebaseline, artifact: results/.../baseline.json, sha256: TBD}
      - {kind: authorized-by, to: d-batch1-charter, tag: DO, scope_hash: TBD}
      - {kind: gated-by, gate: g-sbc-coverage}
      - {kind: gated-by, gate: g-population}
    outputs: [coverage table, panel stamps]

### 6.2 Gate instrument shape

    id: g-znorm
    class: identity
    question: integral of L(d given h) over data space equals 1 at every h-node
    band: {green: abs_dev <= 1e-6, anomalous: abs_dev > 1e-3}
    trigger: every production likelihood build
    frozen_by: d-batch1-charter     # edits = physics-change + fresh RULE
    history: gates/panel/g-znorm.csv

### 6.3 Convergence decide-node shape (live example)

    id: d-a4-final-ratification
    type: decide
    tag: RULE
    title: drop the PROVISIONAL flag on the A4 mz_sel/eff ratification
    status: blocked
    requires:
      - {from: v-falsifier-ii-classG, artifact: verdict record, state: done}
      - {panel: [g-score-null, g-c0-baseline], state: green-or-waived}
    note: returns to the author with numbers, never auto-ratified (rows 278/280/284)

    id: c-auto-default-venue-general
    type: claim
    title: the mass-aware auto default is valid across venues, not only iiib
    status: conjectured
    edges:
      - {kind: spawned-by, to: m-a18-armc}         # row 286
      - {kind: discriminates-target-of, from: m-joint-r1-mass-aware}   # state, candidate 10

### 6.4 Linter v0 (12 rules)

    L1  slug prefix agrees with type
    L2  no orphans: every node has an inbound edge unless root true
    L3  claim status/confidence changes must reference a g- or d- event
    L4  every feeds edge carries artifact + sha256; stale checksum invalidates downstream
    L5  the graph is acyclic; revisits are new nodes with spawned-by, never back-edges
    L6  revision <= max_revisions, else forced Hold
    L7  a measure without registered-by is exploratory and cannot feed a decide
    L8  a decide is eligible only when its requires-manifest is satisfied
    L9  aggregates consume one population type
    L10 a node created after its authorization scope hash has no valid authorization
    L11 AGENT-only provenance never reaches status verified
    L12 a red panel stamp blocks rd-/d- consumption absent a waives edge

Ratification ask (all tagged): [DO] pilot Horizon 1 steps 1-2; [RULE] adopt the section 2 taxonomy + panel law as the batch standard; [RULE] section 4's adapt/adopt/reject verdict on the atomic spec, pending retrieval of the truncated competing recommendation; [STANDING — explicit grant required, lapses at campaign end] the section 3.4 approval-scope semantics as the meaning of charter ratification for graph batches.

---

## REVISION 1 (2026-09-01, applying GRAPH1_CRITIC_NOTES_20260901.md)

SHOULD 7: the r_phi worked example in section 2.5 now notes that "commit 5e7fda16" is narrated at
ledger row #286 but the literal hash string is quoted only at row #288; both rows are cited where
the hash appears in this document, matching the correction made in
RESEARCH_GRAPH_1_PROPOSAL_20260901.md.

NOTE 10 (applied): section 2.4's bounded re-entry law gained a fourth point naming the asymmetry
the critic flagged — verify (v-) nodes have no max_revisions analogue, unlike register (r-) nodes;
noted as fail-safe (the downstream decide node stays blocked rather than causing runaway compute)
rather than corrected, since no register-style cap is proposed for v- nodes in this batch.

No change was needed here for MUST-FIX 1: this document already tagged the section 3.4
approval-scope semantics [STANDING] correctly in its own ratification ask; the defect the critic
found was GRAPH1 folding that STANDING item into a RULE-tagged row instead of asking for it
separately — fixed in RESEARCH_GRAPH_1_PROPOSAL_20260901.md's REVISION 1, not here.
