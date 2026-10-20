# External research 2 — structures for organizing research work beyond trees

Scope: scientific workflow DAG engines, registered reports, Bayesian workflow literature (SBC, PPCs as gates), formal-verification-in-the-loop, and design-of-experiments / active-learning loops. Goal: extract adoptable node types, edge semantics, cycle-handling conventions, and STOP/gate conventions for a directed research graph covering statistical-methodology research (the kind this project's runbooks/research-cycle already do informally).

## 1. Scientific workflow DAG engines (Snakemake, Nextflow, Airflow)

- **Node = unit of computation, not unit of decision.** In all three systems a node is a job/task/process; the DAG's job is reproducible re-execution of a pipeline, not representing an argument or claim. Adoptable idea: keep a research graph's "computation nodes" (measure/reproduce/run-script) structurally distinct from its "claim/decision nodes" (verdict, ruling, STOP) — the engines conflate task=node because they have no claim layer; a research graph should not.
- **Edge = data dependency, not just "happens after."** Nextflow edges are explicit data channels; Snakemake edges are input/output file dependencies inferred from rule signatures. Adoptable: label research-graph edges with *what artifact* flows across them (a claim card, a dataset, a verdict), not just an arrow — makes stale-input detection possible (if the upstream artifact changed, the edge is invalidated even if the node "ran").
- **Static vs. dynamic DAG.** Airflow's DAG is fixed before execution; Snakemake/Nextflow DAGs unfold at runtime because the number of jobs depends on data discovered mid-run (e.g. wildcards, glob expansion). Adoptable: a research graph should allow "template" nodes that expand into N sibling nodes once an earlier node's output count is known (e.g. "one verification agent per claim-card row, count TBD until the claim card is read") — matches this project's fan-out pattern, and the tiering rule ("compute each phase's fan-out before launch") is effectively hand-rolling what Snakemake's checkpoint mechanism automates.
- **Snakemake checkpoints — the cycle/expansion answer.** A rule marked as a `checkpoint` triggers re-evaluation of downstream input functions once it completes; this is how Snakemake supports conditional/dynamic graph expansion without a real cycle. Adoptable: model "re-plan downstream nodes after this result" as a checkpoint node type, distinct from a normal computation node — this is a cleaner primitive than a soft convention like "re-open the runbook."
- **No true cycles, ever.** All three engines are strictly acyclic; DAG tools uniformly reject loops (Airflow explicitly: "acyclic — guaranteeing absence of cycles or loops"). Iteration is handled by (a) re-triggering a whole new DAG run (a new "wave"/campaign in this project's own vocabulary), or (b) subworkflows/subDAGs invoked as a single node from the parent. Adoptable: a "revisit" in a research graph should be modeled as a **new node pointing back to the same claim**, not a literal back-edge — preserves the audit trail (you can see attempt 1 was refuted, attempt 2 revised the model) instead of overwriting history. This matches the existing convention of numbered rows in decision ledgers rather than mutating old rows — worth stating as the graph-formal reason that convention is correct.
- **Subworkflows as opaque composite nodes.** Snakemake subworkflows run to completion and hand off files before the parent resumes; this is the DAG analogue of "a whole runbook is one node from the parent tree's point of view." Adoptable: allow a research-graph node to be typed `subgraph-ref` pointing at a whole other graph/runbook, so a top-level roadmap doesn't need to inline every wave's internal structure.

Sources:
- [How Do Users Design Scientific Workflows? The Case of Snakemake and Nextflow](https://dl.acm.org/doi/fullHtml/10.1145/3676288.3676290)
- [How do users design scientific workflows? (arXiv PDF)](https://arxiv.org/pdf/2309.14097)
- [Snakemake vs. Nextflow: strengths and weaknesses (Biostars)](https://www.biostars.org/p/258436/)
- [snakemake.dag module docs](https://snakemake.readthedocs.io/en/v5.6.0/_modules/snakemake/dag.html)
- [DAG including subworkflows · Issue #513 · snakemake/snakemake](https://github.com/snakemake/snakemake/issues/513)
- [How to run an Airflow DAG in a loop](https://mikulskibartosz.name/run-airflow-dag-in-loop)
- [An Empirical Study of Developers' Challenges Implementing Workflows as Code (Airflow)](https://arxiv.org/pdf/2406.00180)

## 2. Registered Reports model

- **Two formal gate types, both binding before the outcome is known.** Stage 1 review evaluates only the *question + design + analysis plan*, with no data yet collected; the gate output is Reject / Revise / **In-Principle Acceptance (IPA)**. IPA is a real commitment device: the venue is bound to publish regardless of result, provided the approved plan was followed and quality bars are met. Stage 2 review checks *adherence* to the Stage-1 plan and whether conclusions follow from data — it does not re-litigate whether the question was worth asking.
- **Adoptable node type: "plan" node, separate from "result" node, with its own gate.** This project's `research-cycle` skill already has a pre-registration stage, but the registered-reports model sharpens the gate semantics: the Stage-1 gate should check *design validity* only (is the analysis plan sound, falsifiable, adequately powered) and must not be contaminated by a peek at results; a separate Stage-2 gate checks only *adherence* (did the executed measurement match the registered plan, and do the stated conclusions follow). Conflating these two checks into one "did the result look right" review is exactly the failure mode registered reports were built to close (outcome bias / HARKing).
- **Adoptable edge semantic: "deviation" edges must be explicit and justified.** A Stage-2 review explicitly interrogates any deviation from the Stage-1 plan. A research graph should have a dedicated edge/annotation type for "this run deviated from its pre-registration, here is why," separate from the normal artifact-dependency edge — makes deviations searchable/auditable rather than buried in prose.
- **IPA as a STOP-preventing convention, not a STOP.** Once IPA is granted, a null/negative result cannot retroactively kill the report — this is the opposite direction from a stage-gate "kill" and worth naming separately: a **"can't-un-approve-post-hoc" edge** protects a pre-registered measurement's write-up from being suppressed just because the number came out boring. Directly reinforces this project's own standing value ("never close by convenience" / correctness over bias-removal) — the registered-reports IPA gate is the formal mechanism other fields use to enforce that same value.

Sources:
- [Registered report — Wikipedia](https://en.wikipedia.org/wiki/Registered_report)
- [Preregistration and Registered Reports | University of Surrey](https://www.surrey.ac.uk/library/open-research/preregistration-and-registered-reports)
- [Pre-registration vs. Registered Reports: What's the difference? (AJE)](https://www.aje.com/arc/pre-registration-vs-registered-reports)
- [Reviewing Registered Reports (Wiley)](https://authors.wiley.com/Reviewers/journal-reviewers/how-to-perform-a-peer-review/reviewing-registered-reports.html)
- [Registered Reports policy (Wiley Author Services)](https://authorservices.wiley.com/author-resources/Journal-Authors/submission-peer-review/registered-reports-policy.html)

## 3. Bayesian workflow literature (Gelman et al. 2020; SBC; PPCs as gates)

- **"Tangled workflow," explicitly non-linear by design.** Gelman et al. (arXiv:2011.01808) name the stages — model construction, inference, model checking, validation, computational troubleshooting, model understanding, model comparison — but stress the graph among them is not a line: "in practice we will be fitting many models for any given problem, even if only a subset of them will ultimately be relevant." This is a peer-reviewed statistics-methodology precedent for exactly the branching/pruning structure this project already uses informally (many candidate mechanisms explored, most refuted, one or two survive to the write-up). Adoptable: name the "tangled" property explicitly in the graph's design doc as a feature, not an artifact of messy process — it legitimizes keeping refuted branches visible in the graph rather than deleting them.
- **SBC as a *necessary-but-not-sufficient* gate on the inference algorithm itself**, distinct from a gate on the model's fit to real data. SBC checks that the sampler/algorithm recovers known simulated truth across the prior — a property of the *machinery*, checkable before real data is ever touched. Adoptable node type: a "machinery-validation" gate that must pass before any node downstream is allowed to consume real-data results — directly matches this project's own `pp_coverage.py` P–P/coverage harness, and suggests that harness's calibration runs should be a formal upstream gate node in the graph feeding every inference run, not a side-tool.
- **PPCs as the complementary sufficient check**, applied *after* SBC, checking the fitted model against the actual observed data (not simulated truth). The two are sequential, not interchangeable: SBC gates the algorithm; PPCs gate the model-given-this-data. Adoptable edge semantic: a gate node should declare *which* of these two questions it answers, since a single "checks passed" label conflates two different failure modes (broken sampler vs. wrong model) that need different fixes.
- **Posterior SBC** (Gelman/Modrák et al., arXiv:2502.03279) extends SBC to be conditional on the observed data rather than the full prior — closer to what a real analysis needs when the observed data lands in an atypical part of parameter space. Adoptable: for this project's per-event diagnostics (the both-channel CSV emitted on every `--evaluate` run), a posterior-SBC-style conditional check would be the more decision-relevant gate than a global prior-averaged SBC, particularly for the photo-z-starved regime already flagged in project memory.

Sources:
- [Bayesian Workflow — arXiv:2011.01808](https://arxiv.org/abs/2011.01808)
- [Bayesian Workflow book (Vehtari et al.)](https://avehtari.github.io/Bayesian-Workflow/)
- [Simulations in Statistical Workflows — arXiv:2503.24011](https://arxiv.org/pdf/2503.24011)
- [Posterior SBC: Simulation-Based Calibration Checking Conditional on Data — arXiv:2502.03279](https://arxiv.org/pdf/2502.03279)
- [Validating Bayesian Inference Algorithms with Simulation-Based Calibration (Talts, Betancourt, Simpson, Vehtari, Gelman)](https://sites.stat.columbia.edu/gelman/research/unpublished/sbc.pdf)
- [Simulation-Based Calibration Checking: The Choice of Test Quantities Shapes Sensitivity](https://www.researchgate.net/publication/375884312_Simulation-Based_Calibration_Checking_for_Bayesian_Computation_The_Choice_of_Test_Quantities_Shapes_Sensitivity)
- [Simulation-based Calibration tutorial (sbi docs)](https://sbi.readthedocs.io/en/stable/advanced_tutorials/11_diagnostics_simulation_based_calibration.html)

## 4. Formal verification-in-the-loop

- **Verification as a typed pipeline stage with its own artifact, not a boolean.** Modern formal-verification pipelines emit machine-checked proof objects / counterexamples, and increasingly automate "replay and filtering after assertion failure" — i.e. a failed verification produces a *diagnostic artifact* (the counterexample), which becomes the input to the next authoring step, not just a red X. Adoptable node type: a verification/refutation node's output should always be a structured artifact (the counterexample, the failing seed, the falsifying statistic) that the next node can consume — this project's claim-card + exhibit convention already does this; the formal-verification literature is independent confirmation this is the right shape, not an accident of this project's habits.
- **Human-in-the-loop review as an explicit downstream node**, not an implicit assumption, in pipelines that combine automated proof search with human judgment — "generation of explanatory assurance cases and human-in-the-loop review" is named as a distinct pipeline stage after automated counterexample diagnosis. Adoptable: keep "author ruling" as its own node type consuming a verification artifact, matching the [DO]/[RULE]/[STANDING] tagging convention already in CLAUDE.md, rather than treating author sign-off as a graph-external formality.
- Caveat: most formal-verification-pipeline material found is about program/hardware correctness (Coq/Lean proofs, loop invariants), not statistical-methodology claims — the transfer is at the level of pipeline *shape* (artifact-producing refutation attempts feeding structured human review), not a directly reusable tool.

Sources:
- [Formal Verification Pipeline — overview](https://www.emergentmind.com/topics/formal-verification-pipeline)
- [Formal Verification Pipelines — overview](https://www.emergentmind.com/topics/formal-verification-pipelines)
- [Runtime verification — Wikipedia](https://en.wikipedia.org/wiki/Runtime_verification)
- [Formal-Method-Guided Vibe Coding: Closing the Verification Loop on AI-Generated Safety-Critical Software — arXiv:2606.22413](https://arxiv.org/pdf/2606.22413)

## 5. Design-of-experiments / active-learning / Bayesian optimization loops

- **The canonical loop has exactly three roles that recur regardless of field: surrogate model → acquisition function → next experiment.** At each iteration a probabilistic surrogate is fit on all data so far; an acquisition function scores candidate next experiments trading off exploration vs. exploitation; the top-scoring candidate is run and its outcome folds back into the surrogate. Adoptable node types for a research graph choosing its next branch to pursue: a **"belief-state" node** (current best model of which hypotheses/branches look promising, analogous to the surrogate), an **"acquisition" node** (an explicit, inspectable scoring rule for which next branch to fund — this project's runbooks currently make this choice by narrative judgment call; naming it as a distinct node type would force the scoring rationale into the open), and an **"experiment" node** (the actual measurement).
- **Sequential design under a budget constraint is treated as a first-class variable of the loop**, not an afterthought — the literature frames the choice explicitly as "given past experiments, choose the best next experiment with consideration of associated costs and limited total budgets." Adoptable: an acquisition node should be required to state the cost estimate (compute/agent-hours, wall-clock) alongside the expected information gain, matching this project's own fan-out-cost-before-launch rule (CLAUDE.md's tiering mandate) — the DoE literature is a formal justification for turning that mandate into an explicit graph node type rather than a launch-summary line.
- **Exploration vs. exploitation is the graph's branching decision, made legible.** In closed-loop neurophysiology and BO-for-manufacturing applications, the acquisition function is explicitly the place where "explore an uncertain region" vs. "exploit a promising one" gets decided numerically rather than by feel. Adoptable: when a research graph has multiple live, un-refuted branches, the choice of which to advance next should be tagged as either an "exploit" move (double down on the branch with the strongest existing signal) or an "explore" move (spend budget on the branch with the highest uncertainty/highest potential payoff if right) — makes the STOP-vs-continue decision at each branch auditable against a stated exploration/exploitation intent instead of retroactive narrative.

Sources:
- [A Bayesian active learning strategy for sequential experimental design in systems biology — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4181721/)
- [Bayesian Optimisation for Sequential Experimental Design — arXiv:2107.12809](https://arxiv.org/pdf/2107.12809)
- [Adaptive Bayesian methods for closed-loop neurophysiology (Pillow)](https://pillowlab.princeton.edu/pubs/Pillow2016_ActiveLearningChap.pdf)
- [Bayesian Optimization with Active Constraint Learning for Advanced Manufacturing Process Design](https://www.tandfonline.com/doi/full/10.1080/24725854.2025.2475505)
- [Multi-step lookahead Bayesian optimization with active learning (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0098135422003210)

## 6. Stage-gate / phase-gate project governance (adjacent, for gate vocabulary)

- **Five-way gate outcome vocabulary, richer than binary pass/fail.** Standard stage-gate practice defines a gate decision as one of: **Go** (proceed), **Kill** (terminate), **Hold** (pause pending conditions), **Recycle** (return to the previous stage for more work), or **Conditional Go** (proceed pending specific fixes). Adoptable: this project's research-cycle / decision-ledger vocabulary currently leans on ad hoc labels (PARKED, HOLD, REFUTED, CONTROL-FAIL); adopting the five-way Go/Kill/Hold/Recycle/Conditional-Go taxonomy as the canonical gate-outcome enum would make every gate node's result machine-comparable across runbooks, and "Recycle" names precisely the "send this branch back to the pre-registration stage with a revised plan" move that currently gets described in prose each time.
- **Criteria are fixed *before* the gate is reached**, not chosen at gate time — "the gate team defines explicit criteria for each decision before a project receives funding," i.e. thresholds are pre-registered, not discovered. This is the same discipline as the registered-reports Stage-1 plan, arrived at independently from industrial process design — cross-field convergence worth citing as corroboration when proposing a stricter pre-registration-of-gate-criteria rule for this project's own runbooks.

Sources:
- [Gate Reviews in Project Management: Process, Stages, Template (monday.com)](https://monday.com/blog/project-management/gate-review/)
- [Stage Gate Process Guide (Cora Systems)](https://corasystems.com/guidebooks/stage-gate-process-modern-innovation-guide)
- [What is Stage-Gate®? (PreScouter)](https://www.prescouter.com/2013/09/what-is-stage-gate/)
- [The Ultimate Guide to the Phase-Gate Process (Viima)](https://www.viima.com/blog/guide-to-phase-gate-process)

## Summary table — adoptable elements for a directed research graph

| Element | Source system | What it adds beyond a plain tree |
|---|---|---|
| Computation node vs. claim/decision node, kept distinct | Workflow engines (implicit gap) + registered reports | Prevents conflating "a script ran" with "a claim is settled" |
| Edge labeled with the artifact crossing it | Nextflow channels / Snakemake I/O | Enables stale-input invalidation when an upstream artifact changes |
| Checkpoint node → dynamic re-expansion of downstream nodes | Snakemake checkpoints | Formalizes this project's fan-out-count-depends-on-earlier-output pattern |
| Revisit-as-new-node, never a literal back-edge | All DAG engines (strictly acyclic) | Preserves full audit history instead of overwriting a row |
| Subgraph-reference node (whole runbook = one node from outside) | Snakemake subworkflows | Lets a top-level roadmap stay small |
| Plan node with its own gate, separate from result node | Registered Reports Stage 1/2 | Blocks outcome bias / HARKing structurally, not by discipline alone |
| Deviation edge (explicit, justified) | Registered Reports Stage-2 adherence check | Makes "did we follow the pre-reg" auditable, not buried in prose |
| Machinery-validation gate (SBC) upstream of any real-data node | Bayesian workflow / SBC | Separates "is the sampler right" from "is the model right" |
| Model-fit gate (PPC) downstream of SBC | Bayesian workflow / PPC | The second, distinct sufficient check, not interchangeable with SBC |
| Verification node emits a structured counterexample artifact | Formal-verification pipelines | Failed checks produce fuel for the next step, not a bare boolean |
| Author-ruling node consuming a verification artifact | Formal-verification human-in-the-loop stage; this project's [DO]/[RULE]/[STANDING] tags | Keeps sign-off graph-internal and typed |
| Belief-state / acquisition / experiment node triad | Bayesian optimization / active learning | Makes "why this branch next" a scored, inspectable decision |
| Cost-and-expected-gain stated at the acquisition node | Sequential experimental design under budget | Ties directly to this project's existing fan-out-cost-before-launch rule |
| Explore vs. exploit tag on a branch-advance decision | Active learning / BO | Makes STOP-vs-continue calls auditable against a stated intent |
| Five-way gate outcome enum: Go / Kill / Hold / Recycle / Conditional-Go | Stage-gate governance | Replaces this project's ad hoc PARKED/HOLD/REFUTED vocabulary with a canonical, comparable set |
| Gate criteria fixed before the gate is reached | Stage-gate governance + Registered Reports (convergent) | Cross-field corroboration for pre-registering thresholds, not just designs |
