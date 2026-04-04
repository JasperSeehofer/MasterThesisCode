---
phase: quick-3
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .gpd/quick/3-literature-research-galaxy-catalog-in/galaxy-catalog-completeness-research.md
interactive: false

conventions:
  units: "SI (c, G explicit)"
  metric: "mostly-plus"
  coordinates: "spherical"

contract:
  scope:
    question: "How should galaxy catalog incompleteness be corrected in the dark siren likelihood, and what specific methods apply to GLADE+ for LISA EMRI cosmology?"
  claims:
    - id: claim-likelihood-formula
      statement: "The standard completeness-corrected dark siren likelihood formula is documented with equation references"
      deliverables: [deliv-research-doc]
      acceptance_tests: [test-formula-complete]
      references: [ref-gray2020, ref-gray2022, ref-finke2021]
    - id: claim-glade-completeness
      statement: "GLADE+ completeness estimation methods are documented with practical implementation guidance"
      deliverables: [deliv-research-doc]
      acceptance_tests: [test-completeness-method]
      references: [ref-dalya2022]
    - id: claim-implementation-path
      statement: "A concrete implementation approach for this codebase is outlined with specific file/function changes"
      deliverables: [deliv-research-doc]
      acceptance_tests: [test-implementation-spec]
      references: [ref-gray2020]
  deliverables:
    - id: deliv-research-doc
      kind: report
      path: ".gpd/quick/3-literature-research-galaxy-catalog-in/galaxy-catalog-completeness-research.md"
      description: "Comprehensive literature research on galaxy catalog completeness correction for dark siren inference"
      must_contain:
        - "Completeness-corrected likelihood equation with full derivation context"
        - "GLADE+ completeness estimation approach"
        - "Implementation specification for this codebase"
  acceptance_tests:
    - id: test-formula-complete
      subject: claim-likelihood-formula
      kind: existence
      procedure: "Verify the research doc contains the full completeness-corrected likelihood formula with each term explained and at least 3 primary references"
      pass_condition: "Formula present with Gray et al. 2020 Eq. reference, all terms defined, and limiting cases discussed"
      evidence_required: [deliv-research-doc]
    - id: test-completeness-method
      subject: claim-glade-completeness
      kind: existence
      procedure: "Verify the doc specifies how to estimate f(z) for GLADE+ with concrete approach"
      pass_condition: "At least one practical method for estimating GLADE+ completeness as a function of redshift is documented"
      evidence_required: [deliv-research-doc]
    - id: test-implementation-spec
      subject: claim-implementation-path
      kind: consistency
      procedure: "Verify the doc maps the formula to specific functions/files in the codebase"
      pass_condition: "Implementation section references bayesian_statistics.py single_host_likelihood and specifies where completeness enters"
      evidence_required: [deliv-research-doc]
  references:
    - id: ref-gray2020
      kind: paper
      locator: "arXiv:1908.06050"
      aliases: ["Gray et al. (2020)"]
      role: definition
      why_it_matters: "Primary reference for the standard dark siren method with catalog completeness correction"
      applies_to: [claim-likelihood-formula, claim-implementation-path]
      must_surface: true
      required_actions: [read, use, cite]
    - id: ref-gray2022
      kind: paper
      locator: "arXiv:2111.04629"
      aliases: ["Gray et al. (2022)"]
      role: method
      why_it_matters: "Updated gwcosmo framework with improved completeness handling"
      required_actions: [read, cite]
    - id: ref-finke2021
      kind: paper
      locator: "arXiv:2101.12660"
      aliases: ["Finke et al. (2021)"]
      role: method
      why_it_matters: "Alternative dark siren approach with pixelized completeness"
      required_actions: [read, compare]
    - id: ref-dalya2022
      kind: paper
      locator: "arXiv:2110.06184"
      aliases: ["Dalya et al. (2022)"]
      role: definition
      why_it_matters: "GLADE+ catalog paper with completeness characterization data"
      required_actions: [read, use, cite]
  forbidden_proxies:
    - id: fp-no-equations
      subject: claim-likelihood-formula
      proxy: "Listing paper titles without extracting the actual equations and methods"
      reason: "The research doc must contain the actual mathematical formula, not just citations"
    - id: fp-no-vague
      subject: claim-implementation-path
      proxy: "Vague statements like 'completeness correction is needed' without specifying the mathematical form"
      reason: "The implementation specification must be concrete enough to serve as input for a Physics Change Protocol"
  uncertainty_markers:
    weakest_anchors:
      - "GLADE+ completeness at z > 0.1 may not be well-characterized in the literature; may need empirical estimation"
    disconfirming_observations:
      - "If Gray et al. formula assumes GW170817-style localization (O(10 deg^2)), it may need adaptation for LISA EMRI localization (O(1 deg^2))"
---

<objective>
Research the standard methods for correcting galaxy catalog incompleteness in dark siren Hubble constant inference, with specific focus on GLADE+ completeness and practical implementation for LISA EMRI cosmology.

Purpose: Our bias investigation (scripts/bias_investigation/FINDINGS.md) identified GLADE catalog incompleteness at z > 0.08 as the root cause of the H0 posterior bias (MAP=0.66 vs true h=0.73). This research task surveys the literature to determine the correct completeness-corrected likelihood formula and how to implement it.

Output: A comprehensive research document covering the mathematical framework, GLADE+ completeness estimation, and a concrete implementation specification for this codebase.
</objective>

<execution_context>
@/home/jasper/.claude/get-physics-done/workflows/execute-plan.md
@/home/jasper/.claude/get-physics-done/templates/summary.md
</execution_context>

<context>
@.gpd/STATE.md
@scripts/bias_investigation/FINDINGS.md
@master_thesis_code/bayesian_inference/bayesian_statistics.py
@master_thesis_code/galaxy_catalogue/handler.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Survey dark siren likelihood with catalog completeness correction</name>
  <files>.gpd/quick/3-literature-research-galaxy-catalog-in/galaxy-catalog-completeness-research.md</files>
  <action>
    Conduct a focused literature survey on galaxy catalog completeness corrections in dark siren cosmology. Use WebSearch and WebFetch to access arXiv papers. Structure the research document with these sections:

    ## 1. The Standard Dark Siren Likelihood Formula

    Start from the uncorrected galaxy catalog method (our current implementation in bayesian_statistics.py:single_host_likelihood), then derive/present the completeness-corrected version:

    - Retrieve and document the full likelihood from Gray et al. (2020), arXiv:1908.06050, specifically their Eq. (3)-(8) covering the galaxy catalog term and the "completion" term for uncataloged galaxies
    - The key modification: the likelihood for event i given H0 is split into a catalog term (sum over known galaxies weighted by completeness) plus a uniform-in-comoving-volume term for the uncataloged fraction
    - Document each term: p(x_GW | z_gal, Omega_gal, H0), p_cat(z, Omega) (catalog galaxy prior), p_uncataloged(z, Omega | H0) (uniform completion term), and f_complete(Omega, z) (completeness fraction)
    - Show how our current formula maps to the catalog-only term with f_complete = 1 (complete catalog assumption)
    - Document the limiting cases: f_complete = 1 everywhere recovers our current formula; f_complete = 0 everywhere recovers the statistical (no catalog) method

    ## 2. GLADE+ Catalog Completeness

    Research how GLADE+ completeness is characterized:

    - From Dalya et al. (2022), arXiv:2110.06184: how is GLADE+ completeness defined and estimated?
    - What is the completeness as a function of redshift? (Look for f(z) curves or luminosity-based completeness estimates)
    - Is completeness sky-direction dependent? (Relevant for LISA's different sky coverage vs LIGO)
    - How do LVK analyses (O3 papers) estimate GLADE completeness in practice?
    - Search for: gwcosmo package documentation on completeness estimation methods

    ## 3. Implementation Approaches in the Literature

    Document how existing codes implement this:

    - gwcosmo (Gray et al.): their specific implementation choices
    - Finke et al. (2021), arXiv:2101.12660: alternative pixelized approach
    - Any LISA-specific dark siren papers (search for "LISA dark siren" completeness)
    - Note differences between LIGO (poor sky localization, many galaxies per pixel) and LISA EMRI (excellent sky localization, fewer galaxies per error box) that affect the approach

    ## 4. Mathematical Specification for Our Codebase

    Map the corrected likelihood to our specific implementation:

    - Current code: `single_host_likelihood()` in bayesian_statistics.py (lines 500-570) computes the numerator as sum over galaxies of integral(p_det * GW_likelihood * galaxy_z_prior dz)
    - Specify exactly where the completeness fraction f(z, Omega) enters
    - Specify the form of the completion term (uniform galaxy density * GW_likelihood integrated over uncataloged sky fraction)
    - Specify what data/function is needed: f_complete(z) at minimum (angle-averaged), or f_complete(z, Omega) ideally
    - List the specific functions/files that need modification
    - Note any approximations appropriate for LISA EMRI (e.g., sky localization good enough that angular completeness variation within the error box is small)

    ## 5. Practical Considerations

    - How sensitive is the corrected posterior to errors in the completeness estimate?
    - Is a simple f(z) (no angular dependence) sufficient for our use case?
    - What is the minimum catalog completeness at which the method still works?
    - How does the completion term interact with our existing P_det grid?

    Use WebSearch for each major paper and concept. Fetch the actual arXiv abstract pages to confirm equation numbers and results. Be precise about equation references.
  </action>
  <verify>
    1. The completeness-corrected likelihood formula is written explicitly with all terms defined
    2. At least 3 primary references are cited with specific equation numbers
    3. GLADE+ completeness characterization is documented with quantitative information (z-dependence)
    4. The implementation specification maps to actual functions in bayesian_statistics.py
    5. Limiting cases are verified: f=1 recovers current code, f=0 recovers statistical method
  </verify>
  <done>Research document complete with: (1) full corrected likelihood formula with term-by-term explanation, (2) GLADE+ completeness data/estimation methods, (3) comparison of implementation approaches, (4) concrete implementation specification for this codebase, (5) practical considerations for LISA EMRI regime</done>
</task>

</tasks>

<verification>
- Research document covers all 5 sections with concrete equations and references
- The corrected likelihood formula is self-contained (reader can implement from the document alone)
- Implementation path is specific enough to serve as a specification for a future Physics Change Protocol
</verification>

<success_criteria>
A single comprehensive research document that: (1) presents the completeness-corrected dark siren likelihood with full mathematical detail and primary references, (2) characterizes GLADE+ completeness with quantitative estimates, (3) specifies exactly how to modify single_host_likelihood() and related functions, and (4) identifies LISA-specific considerations that differ from LVK analyses.
</success_criteria>

<output>
After completion, the research document serves as the specification for a future implementation milestone.
</output>
