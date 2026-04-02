---
phase: quick-2
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .gpd/quick/2-analyze-evolution-of-residual-bias-ac/bias-evolution-analysis.md
interactive: false

conventions:
  units: "SI / dimensionless h = H0 / (100 km/s/Mpc)"
  metric: "n/a (analysis document)"
  coordinates: "n/a"

contract:
  scope:
    question: "Which bias sources in the H0 posterior have been eliminated across milestones v1.2–v1.2.2, and what distinct bias sources remain that could explain the observed 'with BH mass' low-h bias and 'without BH mass' high-h bias?"
  claims:
    - id: "claim-timeline"
      statement: "There is a coherent, milestone-ordered sequence of identified and fixed bias sources; the remaining biases can be attributed to distinct, named mechanisms; and the two pipelines ('with BH mass' low-h, 'without BH mass' high-h) have different root-cause structures."
      deliverables: ["deliv-analysis"]
      acceptance_tests: ["test-completeness", "test-attribution"]
      references: ["ref-state", "ref-summaries"]
  deliverables:
    - id: "deliv-analysis"
      kind: "document"
      path: ".gpd/quick/2-analyze-evolution-of-residual-bias-ac/bias-evolution-analysis.md"
      description: "Structured analysis covering: (1) eliminated bias sources per milestone, (2) remaining bias sources with mechanism descriptions, (3) differential diagnosis of why 'with BH mass' pulls low while 'without BH mass' pulls high, (4) prioritized investigation agenda."
  references:
    - id: "ref-state"
      kind: "artifact"
      locator: ".gpd/STATE.md"
      role: "source"
      why_it_matters: "Canonical record of open questions and intermediate posterior peaks"
      applies_to: ["claim-timeline"]
      must_surface: true
      required_actions: ["read"]
    - id: "ref-summaries"
      kind: "artifact"
      locator: "Pre-researched context in planning_context block"
      role: "primary data"
      why_it_matters: "Orchestrator-assembled bias timeline with quantitative posterior peaks across all milestones"
      applies_to: ["claim-timeline"]
      must_surface: true
      required_actions: ["synthesize"]
  acceptance_tests:
    - id: "test-completeness"
      subject: "claim-timeline"
      kind: "structural"
      procedure: "Verify analysis document lists every named fix from the timeline (Gaussian index bug, /(1+z) Jacobian removal, five-point stencil, confusion noise, KDE->IS) and correctly classifies each as eliminated."
      pass_condition: "All 5 major fixes are represented and marked eliminated; no eliminated fix appears in the remaining-bias list."
      evidence_required: ["deliv-analysis"]
    - id: "test-attribution"
      subject: "claim-timeline"
      kind: "physical"
      procedure: "For each remaining bias source, the analysis provides a mechanistic explanation (formula or data flow) that would produce the observed direction of bias (low-h or high-h)."
      pass_condition: "At least one remaining mechanism plausibly explains 'with BH mass' monotonically decreasing shape; at least one plausibly explains 'without BH mass' h=0.86 overshoot."
      evidence_required: ["deliv-analysis"]
  forbidden_proxies:
    - id: "fp-list"
      subject: "claim-timeline"
      proxy: "A flat chronological list of all events without mechanistic attribution or directional prediction."
      reason: "Listing events does not constitute understanding. The analysis must explain WHY each remaining source pulls in the observed direction."
    - id: "fp-speculation"
      subject: "claim-timeline"
      proxy: "Speculative mechanisms with no grounding in the actual likelihood formula or data-flow."
      reason: "Bias sources must be traceable to specific code paths, formula terms, or data structure asymmetries documented in the timeline context."
  links:
    - id: "link-main"
      source: "claim-timeline"
      target: "deliv-analysis"
      relation: "supports"
      verified_by: ["test-completeness", "test-attribution"]
  uncertainty_markers:
    weakest_anchors:
      - "The 'without BH mass' h=0.86 overshoot in the production run may partially reflect real P_det selection effects rather than a likelihood formula bug — cannot be resolved without a P_det=1 production run."
    disconfirming_observations:
      - "If 'with BH mass' posterior peak moves to h=0.73 after removing p_det(M_detection) mismatch, this would confirm that as the dominant remaining source."
      - "If both posteriors show similar bias direction after switching to P_det=1 in production, the asymmetry is driven by P_det rather than mass-dependent likelihood terms."
---

<objective>
Synthesize the documented bias evolution across milestones v1.2 through v1.2.2 into a structured analysis that clearly separates eliminated bias sources from remaining ones, explains the mechanistic direction of remaining biases, and produces a prioritized investigation agenda.

Purpose: Establish a clear diagnostic baseline before v1.3 planning. The two posteriors pull in opposite directions (h=0.652 low, h=0.86 high) after all documented fixes — understanding why is prerequisite to targeted remediation.

Output: bias-evolution-analysis.md — a structured analysis document usable as context for the next milestone planning session.
</objective>

<execution_context>
@/home/jasper/.claude/get-physics-done/workflows/execute-plan.md
@/home/jasper/.claude/get-physics-done/templates/summary.md
</execution_context>

<context>
@.gpd/STATE.md

# Pre-researched timeline (from orchestrator — primary source)
# All quantitative data points, milestone events, and documented bias sources
# are provided in the <planning_context> block of this task's spawning prompt.
# The executor must treat that block as the canonical input.
</context>

<tasks>

<task type="auto">
  <name>Task 1: Classify bias sources as eliminated or remaining, with mechanistic attribution</name>
  <files>.gpd/quick/2-analyze-evolution-of-residual-bias-ac/bias-evolution-analysis.md</files>
  <action>
    Using the pre-researched timeline from the planning_context block, produce a structured analysis document with the following sections:

    ## 1. Bias Timeline — What Was Fixed and When

    For each milestone (pre-v1.2, v1.2, v1.2.1, v1.2.2), list every bias-relevant fix. For each fix:
    - Name and milestone
    - What the bug was (formula term, wrong index, missing factor, etc.)
    - What posterior behavior it was expected to change
    - Whether the fix was confirmed to change the posterior (with data point if available)
    - Status: ELIMINATED (fix confirmed effective) or INEFFECTIVE (fix applied but bias unchanged)

    Fixes to classify:
    1. Fisher matrix O(ε) -> O(ε⁴) five-point stencil (PHYS-01, v1.2)
    2. Galactic confusion noise added to LISA PSD (PHYS-02, v1.2)
    3. KDE -> IS estimator for P_det (v1.2)
    4. Gaussian index bug fix: [0] (3D) -> [1] (4D) in "with BH mass" numerator (GPD debug, v1.2)
    5. Spurious /(1+z) Jacobian removal from "with BH mass" numerator (Phase 14-15, v1.2.1)

    ## 2. Remaining Bias Sources — Mechanistic Analysis

    For each documented remaining source from the timeline:
    a. p_det(M_detection) mismatch in numerator
    b. Galaxy mass distribution z-asymmetry
    c. Conditional decomposition tilt in M_z_frac coordinates
    d. Redshift-mass correlation in joint p_gal(z) * mass_integral
    e. "Without BH mass" high-h: P_det normalization or selection effect
    f. Zero-likelihood problem (21% of events)
    g. Quadrature (numerator) vs MC (denominator) methodology asymmetry

    For each source:
    - Trace to the specific formula term or data flow step that produces it
    - Predict the direction of bias (toward lower h or higher h) and explain why
    - Assess whether it affects "with BH mass" only, "without BH mass" only, or both
    - Estimate relative severity (high/medium/low) based on available data

    ## 3. Differential Diagnosis

    Explain the asymmetry: why does "with BH mass" pull low while "without BH mass" pulls high?

    The explanation must:
    - Identify which remaining sources are present in "with BH mass" but absent (or reversed) in "without BH mass"
    - Explain why the mass-dependent terms in the "with BH mass" likelihood systematically favor lower h
    - Assess whether the "without BH mass" high-h result is consistent with a P_det selection effect or requires a separate explanation

    Cross-check: "Without BH mass" with P_det=1 (22 det) gives h=0.678 — close to h_true=0.73, within 0.7σ. "Without BH mass" with real P_det (1000+ det) gives h=0.86. This points strongly to P_det as the driver of the high-h overshoot, not the likelihood formula itself.

    ## 4. Prioritized Investigation Agenda

    Rank the remaining bias sources by:
    1. Estimated impact magnitude
    2. Feasibility of isolation (can it be tested with P_det=1 mock or a synthetic catalog?)
    3. Whether it blocks Phase 16 (full posterior validation)

    Produce a prioritized list with recommended next actions for each item. Flag which items require new code/data vs which can be tested with existing infrastructure.

    ## 5. Key Numerical Anchors

    Tabulate the posterior peak estimates across conditions:
    | Condition | Pipeline | h_peak | Dataset | P_det |
    |-----------|----------|--------|---------|-------|
    | Pre-v1.2 baseline | Without BH mass (sum) | 0.730 | 22 det (run_v12_validation) | 1 |
    | Pre-v1.2 baseline | Without BH mass (product) | 0.678 | 22 det | 1 |
    | Pre-v1.2 baseline | With BH mass | ≤0.600 | 22 det | 1 |
    | Post-all-fixes | Without BH mass | 0.678 | 22 det | 1 |
    | Post-all-fixes | With BH mass | ≤0.652 | 22 det | 1 |
    | Production (v1.2) | Without BH mass | 0.86 | 1000+ det | real |
    | Production (v1.2) | With BH mass | 0.72 | 1000+ det | real |

    h_true = 0.73. Note which deviations are within measurement uncertainty vs systematic.
  </action>
  <verify>
    Structural checks:
    - All 5 documented fixes appear in Section 1 with a classification (ELIMINATED or INEFFECTIVE)
    - All 7 remaining sources appear in Section 2 with directional predictions
    - Section 3 differential diagnosis explicitly addresses the "without BH mass" P_det=1 vs real-P_det comparison (h=0.678 vs h=0.86) as the key diagnostic
    - Section 4 produces an ordered list (not a flat list) with feasibility assessment
    - Section 5 numerical table is complete and consistent with the timeline data points

    Physical consistency checks:
    - No eliminated source appears as a remaining source
    - Directional predictions in Section 2 are consistent with the observed posterior directions (low for "with BH mass", high for "without BH mass" with real P_det)
    - The differential diagnosis does not contradict the P_det=1 data point (h=0.678 close to h_true means the "without BH mass" likelihood formula itself is approximately correct)
  </verify>
  <done>
    bias-evolution-analysis.md exists with all 5 sections complete. The document is self-contained: a reader who has not seen the timeline context can understand which fixes were applied, which biases remain, why each remaining bias pulls in its observed direction, and what to investigate next. The differential diagnosis section provides a mechanistically grounded explanation for the observed asymmetry between the two pipelines.
  </done>
</task>

</tasks>

<verification>
The analysis document must not introduce new speculation beyond what the timeline data supports. Every claim about a remaining bias source must reference either a specific formula term in the likelihood (traceable to bayesian_statistics.py) or a documented data-flow asymmetry. The key cross-check is the P_det=1 vs real-P_det comparison for "without BH mass": if h_peak moves from 0.678 to 0.86 solely due to switching P_det, the "without BH mass" likelihood formula is exonerated and P_det normalization/selection is the dominant remaining source for that pipeline.
</verification>

<success_criteria>
1. Every milestone fix is correctly classified as eliminated (confirmed by posterior data) or ineffective.
2. The "with BH mass" low-h bias has at least one remaining source with a mechanistic explanation traceable to the likelihood formula.
3. The "without BH mass" high-h overshoot is correctly attributed primarily to P_det rather than the likelihood formula, with the P_det=1 data point as evidence.
4. The prioritized agenda identifies at least one investigation path feasible with existing infrastructure (P_det=1 mock + synthetic catalog).
5. The numerical anchor table is internally consistent with h_true=0.73.
</success_criteria>

<output>
After completion, create `.gpd/quick/2-analyze-evolution-of-residual-bias-ac/2-SUMMARY.md` summarizing:
- Key conclusions on eliminated vs remaining bias sources
- Differential diagnosis result (1-2 sentences)
- Top 2-3 prioritized next investigation steps
</output>
