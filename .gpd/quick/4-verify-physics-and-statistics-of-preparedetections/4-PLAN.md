---
phase: quick-4
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: []
interactive: false

conventions:
  units: "SI (Gpc, M_sun, radians)"
  metric: "(+,-,-,-)"
  coordinates: "spherical (phiS azimuthal, qS polar)"

contract:
  scope:
    question: "Is the PrepareDetections truncated-normal sampling procedure statistically correct, and does ignoring Fisher matrix off-diagonal terms in the sampling (but not in the likelihood) introduce bias?"
  claims:
    - id: claim-sigma-extraction
      statement: "The 1-sigma errors used for sampling are correctly extracted as sqrt of diagonal elements of the inverse Fisher matrix"
      deliverables: [deliv-audit-report]
      acceptance_tests: [test-sigma-chain]
      references: [ref-vallisneri]
    - id: claim-truncation-bounds
      statement: "Truncation bounds (phi in [0,2pi], theta in [0,pi], d_L in [0,dist(1.5)], M in [1e4,1e6]) are physically appropriate"
      deliverables: [deliv-audit-report]
      acceptance_tests: [test-bounds-physics]
      references: [ref-cutler-flanagan]
    - id: claim-independent-sampling
      statement: "Drawing parameters independently (ignoring off-diagonal Fisher terms) for the 'best guess' mean values does not systematically bias the posterior, because the full covariance IS used in the likelihood evaluation"
      deliverables: [deliv-audit-report]
      acceptance_tests: [test-correlation-impact]
      references: [ref-gray2020]
    - id: claim-pipeline-consistency
      statement: "The prepare step and the evaluate step use consistent conventions (same columns, same meaning of 'best guess' vs 'true' values)"
      deliverables: [deliv-audit-report]
      acceptance_tests: [test-column-consistency]
      references: []
  deliverables:
    - id: deliv-audit-report
      kind: analysis
      path: "(returned as text output, no file created)"
      description: "Physics and statistics audit of PrepareDetections sampling procedure"
      must_contain: ["sigma extraction chain", "truncation bound assessment", "correlation impact analysis", "pipeline consistency check"]
  acceptance_tests:
    - id: test-sigma-chain
      subject: claim-sigma-extraction
      kind: analytical
      procedure: "Trace the sigma values from Fisher matrix inversion (parameter_estimation.py:395) through CRB CSV columns (delta_X_delta_X) through Detection.__init__ (detection.py:95-103) to truncnorm scale parameter"
      pass_condition: "sigma = sqrt(Gamma^{-1}_{ii}) at every stage, no missing factors of 2, no confusion between variance and standard deviation"
      evidence_required: [deliv-audit-report]
    - id: test-bounds-physics
      subject: claim-truncation-bounds
      kind: analytical
      procedure: "Verify each bound against physical constraints: phi is periodic on [0,2pi], theta on [0,pi], d_L >= 0 with dist(1.5) as max detectable redshift, M in EMRI range"
      pass_condition: "All bounds are physically motivated; dist(1.5) is a reasonable upper cutoff for LISA EMRI detection"
      evidence_required: [deliv-audit-report]
    - id: test-correlation-impact
      subject: claim-independent-sampling
      kind: analytical
      procedure: "Analyze whether independent draws shift the mean of the multivariate Gaussian likelihood systematically in one direction. Key: the 'best guess' values become the MEAN of the likelihood Gaussian in bayesian_statistics.py. If off-diagonal terms are large (rho > 0.5), independent draws can produce inconsistent (phi, theta, d_L, M) combinations that would be unlikely under the joint posterior."
      pass_condition: "Either (a) correlations are typically small (rho < 0.3) so independent draws are adequate, or (b) the bias is identified and quantified"
      evidence_required: [deliv-audit-report]
    - id: test-column-consistency
      subject: claim-pipeline-consistency
      kind: analytical
      procedure: "Verify that prepare_detections.py overwrites the same columns that bayesian_statistics.py reads as 'measured' values, and that the covariance columns are NOT overwritten"
      pass_condition: "Column names match, covariances preserved, no double-application of perturbation"
      evidence_required: [deliv-audit-report]
  forbidden_proxies:
    - "Do not claim the procedure is correct because 'it runs without errors' -- statistical correctness requires mathematical justification"
    - "Do not dismiss correlation effects without estimating typical off-diagonal magnitudes from the actual production data"
  uncertainty_markers:
    weakest_anchors:
      - "The impact of ignoring correlations depends on the actual correlation coefficients in production data, which may not be available in this audit"
    disconfirming_observations:
      - "If typical correlation coefficients |rho_{ij}| > 0.5 for parameters that affect galaxy matching (phi, theta, d_L), the independent sampling is statistically inappropriate"
      - "If the truncation bounds cut significant probability mass (e.g., d_L close to dist(1.5) with large uncertainty), the truncated normal is no longer a good approximation to the posterior"
  references:
    - id: ref-vallisneri
      citation: "Vallisneri (2008), arXiv:gr-qc/0703086"
      relevant_for: "Fisher matrix as Gaussian approximation to the posterior"
    - id: ref-cutler-flanagan
      citation: "Cutler & Flanagan (1994), PRD 49, 2658"
      relevant_for: "Fisher matrix formalism for GW parameter estimation"
    - id: ref-gray2020
      citation: "Gray et al. (2020), arXiv:1908.06050"
      relevant_for: "Dark siren likelihood framework consumed by this pipeline"

estimated_execution:
  total_minutes: 40
  breakdown:
    - task: 1
      minutes: 25
      note: "Trace sigma chain + truncation bound analysis + correlation assessment"
    - task: 2
      minutes: 15
      note: "Pipeline consistency check + production data correlation estimate"
---

<objective>
Audit the physics and statistics of the PrepareDetections step that converts true EMRI parameters to simulated "best guess" (measured) parameters via truncated-normal sampling.

Purpose: Determine whether the independent truncated-normal sampling procedure introduces systematic bias into the H0 posterior, given that the downstream Bayesian inference uses the full Fisher matrix covariance.
Output: Text report with findings on each of the 5 key questions. No code changes.
</objective>

<execution_context>
@/home/jasper/.claude/get-physics-done/workflows/execute-plan.md
@/home/jasper/.claude/get-physics-done/templates/summary.md
</execution_context>

<context>
@.gpd/PROJECT.md
@.gpd/STATE.md

@scripts/prepare_detections.py
@master_thesis_code/datamodels/detection.py
@master_thesis_code/parameter_estimation/parameter_estimation.py
@master_thesis_code/bayesian_inference/bayesian_statistics.py
@master_thesis_code/constants.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Trace the sigma extraction chain and assess truncation bounds</name>
  <files>(read-only analysis, no files modified)</files>
  <action>
    This is a READ-ONLY audit. Do not modify any files.

    PART A — Sigma extraction chain:
    Trace the 1-sigma error values from origin to consumption:
    1. Fisher matrix Gamma_{ij} computed in parameter_estimation.py:compute_fisher_information_matrix()
    2. CRB = Gamma^{-1} computed in parameter_estimation.py:395 via np.linalg.inv()
    3. Off-diagonal elements stored as delta_X_delta_Y in the CSV (parameter_estimation.py:412-418)
    4. Detection.__init__ (detection.py:95-103) extracts sigma = sqrt(delta_X_delta_X) for diagonal entries
    5. convert_to_best_guess_parameters (detection.py:121-148) uses these sigma values as the scale parameter of truncnorm

    Verify: At every stage, confirm sigma = sqrt(Gamma^{-1}_{ii}). Check for potential confusion between:
    - Variance vs standard deviation (sqrt missing or doubled)
    - Fisher matrix element vs inverse Fisher matrix element
    - 1-sigma vs 2-sigma

    PART B — Truncation bounds assessment:
    For each parameter, verify the physical motivation of the truncation bounds:
    - phi in [0, 2*pi]: azimuthal angle periodicity. NOTE: truncated normal on a periodic variable is not ideal — assess whether wrapping effects matter for typical phi_error values.
    - theta in [0, pi]: polar angle range. Check: is theta = qS (colatitude) or latitude? Verify consistency with the coordinate convention.
    - d_L in [0, dist(1.5)]: luminosity distance. Check: is z=1.5 a reasonable maximum for LISA EMRIs? What fraction of events have d_L close to dist(1.5)?
    - M in [1e4, 1e6] M_sun: central BH mass. Check: does this match the EMRI parameter space bounds in datamodels/parameter_space.py?

    PART C — Is truncated normal the right distribution?
    The Fisher matrix formalism gives a Gaussian approximation to the posterior: p(theta|d) ~ N(theta_true, Gamma^{-1}). For high-SNR events (SNR > 20), this is valid (Vallisneri 2008). The truncated normal respects physical bounds while preserving the Gaussian shape. Assess whether this is standard practice in the GW literature.
  </action>
  <verify>
    1. Sigma chain: No factors of 2, sqrt, or pi missing at any stage
    2. Truncation bounds: All bounds match physical parameter ranges
    3. Coordinate consistency: theta/qS convention consistent between prepare and evaluate
    4. Distribution choice: Truncated normal is or is not standard; justify
  </verify>
  <done>Complete trace of sigma extraction with pass/fail at each stage; truncation bound assessment with physical justification for each; distribution choice assessment with literature context</done>
</task>

<task type="auto">
  <name>Task 2: Assess correlation impact and pipeline consistency</name>
  <files>(read-only analysis, no files modified)</files>
  <action>
    This is a READ-ONLY audit. Do not modify any files.

    PART A — Independent sampling vs correlated sampling:
    The key architectural observation is:
    - prepare_detections.py draws (phi, theta, d_L, M) INDEPENDENTLY from 4 separate truncated normals
    - bayesian_statistics.py constructs a MULTIVARIATE normal likelihood using the FULL covariance matrix (lines 165-218), centered on these independently-drawn values

    Analyze the statistical consequences:
    1. The "best guess" values simulate what a GW detector would measure. In reality, the measurement errors ARE correlated (the Fisher matrix has off-diagonal terms). Drawing independently produces "measured" values that are statistically inconsistent with the joint measurement distribution.
    2. However, the posterior evaluation uses the full covariance matrix. The question is: does the MEAN of the likelihood Gaussian need to be a correlated draw, or does it only matter that the covariance structure is correct?
    3. Key insight: In the dark siren framework, the "best guess" values determine which galaxies are nearby in (phi, theta, d_L) space. If the independent draws produce a (phi, theta) that is shifted in a correlated way with d_L, the galaxy matching could be affected.
    4. Estimate the typical magnitude of off-diagonal correlation coefficients rho_{ij} = C_{ij} / sqrt(C_{ii} * C_{jj}) from the production CSV data if available at cluster_results/. If not available, assess from the structure of the Fisher matrix (which parameters are expected to be correlated for EMRIs).

    PART B — Pipeline consistency check:
    Verify the data flow:
    1. prepare_detections.py reads CRAMER_RAO_BOUNDS_OUTPUT_PATH (raw CRB CSV with true parameter values)
    2. It overwrites columns: M, luminosity_distance, phiS, qS with perturbed "best guess" values
    3. It writes to PREPARED_CRAMER_RAO_BOUNDS_PATH
    4. bayesian_statistics.py reads PREPARED_CRAMER_RAO_BOUNDS_PATH (line 111) for the likelihood means
    5. bayesian_statistics.py ALSO reads CRAMER_RAO_BOUNDS_OUTPUT_PATH (line 112) as true_cramer_rao_bounds

    Check:
    - Are the covariance columns (delta_X_delta_Y) preserved in the prepared CSV? (They should be — only 4 columns are overwritten)
    - Is there any risk of the perturbation being applied twice?
    - Does the evaluate step ever accidentally use true values where it should use best-guess, or vice versa?

    PART C — Reference check:
    Is there a standard reference for this exact procedure (truncated-normal independent sampling from Fisher diagonal)?
    Check: Cutler & Flanagan (1994), Vallisneri (2008), Babak et al. (2017, LISA science case). In most GW PE papers, the Fisher matrix gives the full covariance, and mock data is generated by drawing from the multivariate Gaussian — not independent marginals.
  </action>
  <verify>
    1. Correlation impact: Quantified or bounded (either from data or from Fisher matrix structure)
    2. Pipeline consistency: Column names match between prepare and evaluate; covariances preserved
    3. No double-perturbation risk identified
    4. Literature comparison: Standard practice identified or deviation documented
  </verify>
  <done>Correlation impact assessed with quantitative estimate or bound; pipeline consistency verified; literature comparison complete with specific references</done>
</task>

</tasks>

<verification>
- Sigma chain traced end-to-end with no missing factors
- All truncation bounds physically justified
- Correlation impact quantified or bounded
- Pipeline data flow verified for consistency
- Literature context provided for the sampling procedure
</verification>

<success_criteria>
All 5 key questions from the task description answered with physics justification:
1. Truncated normal appropriateness assessed with Fisher matrix / high-SNR justification
2. Truncation bounds verified against physical parameter ranges
3. Independent vs correlated sampling impact quantified or bounded
4. Error columns verified as sqrt(diagonal of inverse Fisher)
5. Literature reference provided or absence documented
</success_criteria>

<output>
After completion, create `.gpd/quick/4-verify-physics-and-statistics-of-preparedetections/4-SUMMARY.md`
</output>
