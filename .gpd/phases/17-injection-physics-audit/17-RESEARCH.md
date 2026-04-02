# Phase 17: Injection Physics Audit - Research

**Researched:** 2026-03-31
**Domain:** Gravitational wave injection campaign consistency, EMRI parameter distributions, cosmological model verification
**Confidence:** HIGH

## Summary

Phase 17 is a code audit phase. The goal is to verify that `injection_campaign()` and `data_simulation()` draw from identical parameter distributions, that the cosmological model (`dist()`) is called consistently across injection and evaluation pipelines, that the d_L-to-z round-trip inversion is numerically stable, and that waveform failure patterns are characterized.

This is primarily a code-reading and numerical-testing phase -- no new physics derivations are needed. The audit methodology is straightforward: line-by-line comparison of the two code paths, identification of parameter flow differences, and simple numerical round-trip tests. The main risk is not difficulty but subtlety: the two pipelines share `Model1CrossCheck` and `ParameterSpace` but wire parameters differently, and the differences are physically motivated but must be documented.

**Primary recommendation:** Perform systematic line-by-line comparison of `injection_campaign()` (main.py:396-572) vs `data_simulation()` (main.py:188-383), focusing on five audit axes: (1) M sampling, (2) redshift/d_L computation, (3) extrinsic parameter randomization, (4) parameter bounds from `_apply_model_assumptions()`, and (5) `dist()` argument defaults. Write round-trip test `z -> d_L(z, h) -> dist_to_redshift(d_L, h) -> z_recovered` for all 7 h-values across z in [0, 0.5]. Parse SLURM logs or injection CSVs for waveform failure categorization.

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
| --- | --- | --- | --- | --- |
| Phase 11.1 CONTEXT.md decisions D-01 through D-09 | design decisions | Define the injection campaign's intended behavior -- audit verifies these were implemented correctly | read and cross-check against code | plan / execution / verification |
| `injection_campaign()` in main.py:396-572 | prior artifact | Primary audit target -- the injection code path | line-by-line comparison with `data_simulation()` | plan / execution |
| `data_simulation()` in main.py:188-383 | prior artifact | Reference code path -- the "ground truth" simulation pipeline | line-by-line comparison with `injection_campaign()` | plan / execution |
| `Model1CrossCheck` in cosmological_model.py | method | Shared population model -- must behave identically in both paths | verify same `_apply_model_assumptions()` and `sample_emri_events()` | plan / execution |
| `dist()` in physical_relations.py | method | Cosmological distance function -- h-dependence is the key difference | verify argument passing in both pipelines | plan / execution / verification |
| Project-level SUMMARY.md | prior research | Documents overall methodology, pitfalls, and conventions | build on, do not re-derive | plan |

**Missing or weak anchors:** No injection CSV data is available locally yet (awaiting rsync from bwUniCluster). AUDT-03 (waveform failure characterization) may need to proceed from SLURM log analysis rather than CSV status columns, since the injection CSV only stores successful SNR computations -- failed events are logged but not written to CSV.

## Conventions

| Choice | Convention | Alternatives | Source |
| --- | --- | --- | --- |
| Units: distances | Gpc | Mpc | physical_relations.py, constants.py |
| Units: masses | Solar masses (source-frame) | kg, redshifted mass | ParameterSpace, Model1CrossCheck |
| Hubble parameter | Dimensionless h = H0/(100 km/s/Mpc) | H0 in km/s/Mpc | constants.py: H = 0.73 |
| Cosmological parameters | WMAP-era: Omega_m = 0.25, Omega_DE = 0.75 | Planck 2018 | constants.py (known outdated, internally consistent) |
| Cosmological model | Flat LambdaCDM (w0=-1, wa=0) | wCDM | physical_relations.py (wCDM params accepted but unused -- known bug) |

**CRITICAL: The injection pipeline calls `dist(z, h=h_value)` with a variable h, while `set_host_galaxy_parameters()` in the simulation pipeline calls `dist(z)` which defaults to `h=H=0.73`. This is an intentional design difference (D-04), not a bug. The audit must verify this is the ONLY place where h handling differs.**

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
| --- | --- | --- | --- |
| d_L(z, h) = (c/H0)(1+z) * integral_0^z dz'/E(z') | Luminosity distance | physical_relations.py:27, Hogg (1999) astro-ph/9905116 Eq. 16 | Verify identical formula in both paths |
| E(z) = sqrt(Omega_m(1+z)^3 + Omega_DE) | Hubble function (LambdaCDM) | physical_relations.py:198 | Verify same Omega_m, Omega_DE in both paths |
| lambda_cdm_analytic_distance() using 2F1 | Analytic comoving distance via hypergeometric | physical_relations.py:240 | Verify same function called in both paths |
| dist_to_redshift(d_L, h) via fsolve | Numerical inversion z -> d_L -> z | physical_relations.py:271 | Round-trip accuracy test |
| dN/dz(M, z) * R_emri(M) | EMRI rate distribution | cosmological_model.py:198 | Verify same polynomial coefficients in both paths |

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
| --- | --- | --- | --- |
| Line-by-line code comparison | Identifies parameter flow differences | injection_campaign() vs data_simulation() | Software engineering best practice |
| Round-trip numerical test | Verifies z -> d_L(z,h) -> z_recovered to 1e-4 | dist() and dist_to_redshift() | Numerical analysis |
| SLURM log parsing | Extracts waveform failure patterns | Cluster log files | -- |
| Histogram analysis of injection CSV | Characterizes (z, M) distribution of failures | Injection data (if available) | -- |

### Approximation Schemes

No approximations are introduced in this phase. The phase verifies that existing approximations (analytic hypergeometric distance integral, fsolve root-finding) are applied consistently across pipelines.

## Standard Approaches

### Approach 1: Systematic Code Path Comparison (RECOMMENDED)

**What:** Walk through both `injection_campaign()` and `data_simulation()` line by line, documenting every function call that sets a physical parameter, and comparing the argument lists.

**Why standard:** This is the only reliable way to verify consistency -- "looks similar" is explicitly forbidden by the phase contract.

**Key steps:**

1. Document how `Model1CrossCheck` is instantiated and shared (both paths use the same instance created in `main()`)
2. Compare how `sample_emri_events()` output is consumed: injection uses `ParameterSample.M` and `ParameterSample.redshift` directly; simulation routes through `GalaxyCatalogueHandler.get_hosts_from_parameter_samples()` which resolves to a `HostGalaxy`
3. Compare how M is set: injection sets `parameter_space.M.value = sample.M`; simulation calls `set_host_galaxy_parameters(host_galaxy)` which sets `M.value = host_galaxy.M`
4. Compare how d_L is set: injection calls `dist(sample.redshift, h=h_value)`; simulation calls `dist(host_galaxy.z)` with default h=0.73
5. Compare extrinsic parameter randomization: both call `randomize_parameters(rng=rng)` but injection then overwrites only M and d_L, while simulation then overwrites M, phiS, qS, and d_L via `set_host_galaxy_parameters()`
6. Compare parameter bounds: both use the same `ParameterSpace` (from the same `Model1CrossCheck`), so `_apply_model_assumptions()` runs once
7. Compare error handling: catalog identical exception types caught

**Known difficulties at each step:**

- Step 2: The galaxy catalog lookup introduces host galaxy properties (phiS, qS from GLADE catalog) that the injection campaign randomizes independently. This is an intentional difference but must be documented.
- Step 4: The h-dependence difference is intentional per D-04. The audit must verify that `dist()` defaults (`Omega_m`, `Omega_de`, `w_0`, `w_a`) are the same in both calls.
- Step 5: The injection stores `phiS` and `qS` in the CSV (line 556-557) but these come from `randomize_parameters()`, not from a galaxy catalog. The simulation gets these from the host galaxy. For P_det estimation (marginalizing over sky angles), this is correct -- the injection samples the marginal distribution.

### Approach 2: Automated Diff-Based Comparison (FALLBACK)

**What:** Extract the sequence of `parameter_space.*` assignments from both functions programmatically and diff them.

**When to switch:** If the code is too long for manual comparison or if additional injection-like functions exist.

**Tradeoffs:** Faster but may miss implicit differences (e.g., default argument values, shared mutable state).

### Anti-Patterns to Avoid

- **"Looks similar" without specifics:** The phase contract explicitly forbids this. Every comparison must cite specific lines and values.
- **Assuming shared Model1CrossCheck means identical behavior:** Both paths create a fresh `Model1CrossCheck(rng=rng)` in `main()` and pass it, but they USE it differently (injection skips galaxy catalog).
- **Ignoring the z_cut in injection:** The injection has `z_cut = 0.5` (line 456) which truncates the sampling distribution. The simulation has no such cut. This is intentional but must be documented.
- **Treating waveform failures as random:** Failures correlate with parameter regions (high eccentricity, extreme mass ratios). A simple failure rate without parameter-region breakdown is explicitly forbidden.

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
| --- | --- | --- | --- |
| LambdaCDM distance integral | Analytic 2F1 form in `lambda_cdm_analytic_distance()` | physical_relations.py:240 | Verify same function called in both paths; do not re-derive |
| EMRI rate model dN/dz * R_emri | Polynomial coefficients in `merger_distribution_coefficients` | cosmological_model.py:70-126 | Verify same coefficients used; do not re-fit |
| Model1CrossCheck parameter bounds | M in [10^4.5, 10^6], a=0.98 fixed, mu=10 fixed, e0<0.2, max_z=1.5 | cosmological_model.py:182-196 | Verify identical in both paths |
| D-04 design decision | d_L computed with candidate h in injection, default h in simulation | Phase 11.1 CONTEXT.md | Document and verify; do not re-justify |

**Key insight:** This phase does not derive anything new. It verifies that existing code correctly implements the Phase 11.1 design decisions. Re-deriving the physics would be a waste of context budget.

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
| --- | --- | --- | --- |
| Project SUMMARY.md waveform failure rate 30-50% | Baseline failure rate to compare against | .gpd/research/SUMMARY.md | From prior campaign experience |
| 24/69500 detections at z < 0.18 | Validates z_cut = 0.5 is generous | SUMMARY.md, PITFALLS.md | From initial (non-injection) campaign |
| h-values {0.60, 0.65, 0.70, 0.73, 0.80, 0.85, 0.90} | The 7 h grid points for round-trip test | cluster/submit_injection.sh | As designed in D-05 |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
| --- | --- | --- | --- | --- |
| Extracting distribution parameters with selection biases | Mandel, Farr & Gair | 2019 | Framework for selection effects in hierarchical inference | Eq. for selection integral alpha(h); why injection consistency matters |
| LISA V: EMRIs | Babak et al. | 2017 | EMRI rate model M1 that Model1CrossCheck implements | Verify Model1CrossCheck polynomial fits against Table III / Fig. 3 |
| Accuracy requirements for selection functions | Farr | 2019 | N_eff > 4*N_det criterion | Sets the bar for how many effective injections are needed |

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
| --- | --- | --- | --- |
| Python diff/comparison | Built-in | Line-by-line code comparison | Human-readable audit trail |
| scipy.optimize.fsolve | scipy | dist_to_redshift inversion | Already used in codebase |
| numpy | numpy | Round-trip numerical test | Already in dependency tree |
| grep/log parsing | Shell | SLURM log analysis for failure categorization | Standard sysadmin tool |

### Supporting Tools

| Tool | Purpose | When to Use |
| --- | --- | --- |
| pandas | Load injection CSVs for failure analysis (if available) | AUDT-03 if CSV data is rsynced |
| matplotlib | Visualize (z, M) distribution of failures | Optional diagnostic |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
| --- | --- | --- | --- |
| Round-trip d_L-to-z test (7 h-values x 100 z-points) | < 1 second | None | Trivial |
| Injection CSV parsing | < 10 seconds | I/O | Read only needed columns |
| SLURM log parsing | < 30 seconds | Log file size | grep for known error patterns |

**Installation / Setup:**
No additional packages needed. All tools are already in the project's dependency tree.

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
| --- | --- | --- | --- |
| dist(z, h) round-trip | d_L-to-z inversion accuracy | For each h in {0.60,...,0.90}, for z in linspace(0.001, 0.5, 100): compute d_L = dist(z, h), z_rec = dist_to_redshift(d_L, h), assert abs(z - z_rec)/z < 1e-4 | All pass |
| Model1CrossCheck identity | Same instance in both paths | Trace from `main()`: both paths receive the same `cosmological_model` object | Same object (Python `id()`) |
| Parameter bounds comparison | _apply_model_assumptions() runs once | Print bounds after init, verify M in [10^4.5, 10^6], a=0.98, mu=10, e0_max=0.2, max_z=1.5 | Match |
| dist() default arguments | Same Omega_m, Omega_DE, w_0, w_a | Inspect default values in function signature | Omega_m=0.25, Omega_DE=0.75, w_0=-1, w_a=0 |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
| --- | --- | --- | --- |
| dist(z=0, h) = 0 for all h | z=0 | 0.0 Gpc | Trivial |
| dist(z, h) proportional to 1/h at fixed z | Low z | d_L ~ cz/(H0) for z << 1 | Hogg (1999) |
| dist_to_redshift(0, h) = 0 | d_L=0 | z=0 | Trivial |

### Red Flags During Computation

- If the round-trip test fails for small z (< 0.01), the fsolve initial guess z=1 may be causing convergence to a wrong root. Check that fsolve converges for small d_L values.
- If Model1CrossCheck produces different parameter bounds in the two paths, a second `_apply_model_assumptions()` call may be overwriting bounds.
- If the injection CSV has no failed events recorded, failures are being silently dropped (confirmed by code reading: failures `continue` without recording).

## Common Pitfalls

### Pitfall 1: Confusing Intentional Differences with Bugs

**What goes wrong:** The auditor flags the `dist(z, h=h_value)` vs `dist(z)` difference as a bug, when it is an intentional design decision (D-04).

**Why it happens:** The two code paths are designed to behave differently for h -- injection varies h to build P_det(h), simulation uses fiducial h=0.73 for CRB computation.

**How to avoid:** Read Phase 11.1 CONTEXT.md D-04 before auditing. Categorize each difference as "intentional per D-XX" or "unintentional discrepancy".

**Warning signs:** Flagging a difference that is documented in D-01 through D-09.

**Recovery:** Re-read the design decisions and re-categorize.

### Pitfall 2: Missing the Galaxy Catalog Intermediary

**What goes wrong:** Concluding that the simulation sets M directly from `sample.M`, when in fact it routes through `GalaxyCatalogueHandler.get_hosts_from_parameter_samples()` which resolves to a nearby galaxy from the GLADE catalog. The resolved galaxy's M, phiS, qS, z may differ from the original ParameterSample values.

**Why it happens:** The injection path sets M = sample.M directly, while the simulation path goes: ParameterSample -> GalaxyCatalogueHandler -> HostGalaxy -> set_host_galaxy_parameters().

**How to avoid:** Trace the full parameter flow from `sample_emri_events()` to the point where SNR is computed, in both paths.

**Warning signs:** Claiming "both paths set M from sample.M" without tracing through the galaxy catalog.

**Recovery:** Read `handler.py:get_hosts_from_parameter_samples()` to understand the resolution step.

### Pitfall 3: Incomplete Waveform Failure Characterization

**What goes wrong:** Reporting a single aggregate failure rate (e.g., "35% of events fail") without breaking it down by parameter region, failure type, or (z, M) bin.

**Why it happens:** The injection CSV only stores successful events. Failed events are logged (warnings) but not recorded in structured form.

**How to avoid:** Parse SLURM log files for warning messages, categorize by exception type (TimeoutError, ParameterOutOfBoundsError, RuntimeError, ValueError/EllipticK, ValueError/Brent, ZeroDivisionError), and if injection CSV + log data can be correlated, map failures to (z, M) regions.

**Warning signs:** The contract explicitly forbids "failure rate estimate without parameter-region breakdown."

**Recovery:** If logs are unavailable, document this as a data gap and propose a structured failure-tracking enhancement for future campaigns.

### Pitfall 4: fsolve Convergence Issues at Extreme z or h

**What goes wrong:** `dist_to_redshift()` uses initial guess z=1 for all inputs. For very small d_L (low z) or extreme h values, fsolve may converge slowly or to a wrong root.

**Why it happens:** The initial guess is hardcoded. For d_L near zero, the true z is near zero but the guess is 1.

**How to avoid:** Include edge cases in the round-trip test: z = 0.001, z = 0.01, z = 0.5, for all 7 h values. Check not just relative error but also the number of fsolve iterations.

**Warning signs:** Round-trip error suddenly spikes at low z.

**Recovery:** If the round-trip fails, recommend a better initial guess (e.g., z_guess = d_L * H0 / c for the linear Hubble law regime).

## Level of Rigor

**Required for this phase:** Code audit with numerical verification

**Justification:** This is a consistency-checking phase, not a derivation phase. The standard is: (1) every comparison must cite specific code lines, (2) every numerical claim must have a runnable test, (3) every difference must be categorized as intentional or unintentional.

**What this means concretely:**

- Line numbers must accompany every code comparison claim
- Round-trip tests must report actual numerical errors, not just "it works"
- Waveform failure analysis must produce counts by category, not qualitative descriptions
- The output is a structured report, not a narrative

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
| --- | --- | --- | --- |
| KDE-based P_det from observed events | Simulation-based P_det from injection campaign | Phase 11.1 (2026-03-30) | Injection campaign must be audited for consistency with simulation pipeline |
| P_det assumed = 1 (no selection effects) | P_det(z, M, h) from histogram grid | Phase 11.1 | Incorrect P_det causes biased H0 posterior -- audit prevents this |

**Superseded approaches to avoid:**

- The old KDE-based `DetectionProbability` class has been replaced by `SimulationDetectionProbability`. Do not audit the old class -- it is no longer in the evaluation pipeline.

## Open Questions

1. **Are injection CSV data files available locally?**
   - What we know: The injection campaign was submitted to bwUniCluster on 2026-03-31 (from project memory "H0 Sweep"). Data needs to be rsynced.
   - What's unclear: Whether the data has arrived yet.
   - Impact on this phase: AUDT-01 and AUDT-02 can proceed from code reading alone. AUDT-03 (waveform failure characterization) benefits from CSV data but can also use SLURM logs.
   - Recommendation: Proceed with code audit (AUDT-01, AUDT-02) immediately. Attempt to load injection CSVs; if unavailable, document the gap and proceed with log-based analysis.

2. **Does the injection CSV record failed events?**
   - What we know: Reading the code (main.py:459-549), failed events hit `continue` without being appended to `results`. Only successful SNR computations are stored.
   - What's unclear: Whether SLURM logs capture enough detail (z, M values) for parameter-region breakdown.
   - Impact on this phase: AUDT-03 may be limited to aggregate failure rates by exception type unless logs contain the parameter values.
   - Recommendation: Document this as a design limitation. Propose adding a failure CSV or status column for future campaigns.

3. **Does `ParameterSample.get_distance()` use default h?**
   - What we know: `handler.py:44-45` shows `get_distance()` calls `dist(self.redshift)` with default h=0.73. This method is available on ParameterSample but it's unclear if it's called in the injection path.
   - What's unclear: Whether any code path in the injection uses this method instead of `dist(z, h=h_value)`.
   - Impact: If called, this would use the wrong h.
   - Recommendation: Grep for all call sites of `get_distance()` and `ParameterSample.get_distance()` to verify none are in the injection path.

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
| --- | --- | --- | --- |
| Manual code comparison | Code too complex to trace | Instrumented test run with parameter logging | Add logging to both paths, run 10 events each, compare parameter DataFrames |
| SLURM log parsing | Logs unavailable or insufficient | Re-run small injection campaign (100 events) with verbose logging | ~10 minutes on GPU |
| Round-trip numerical test | fsolve fails for edge cases | Replace with scipy.optimize.brentq on a bounded interval | Small code change to test script |

**Decision criteria:** If manual code comparison finds no discrepancies in 2 hours of auditing, the approach is working. If logs are completely unavailable, escalate to user for rsync.

## Caveats and Alternatives

1. **Assumption: Both paths share the same `Model1CrossCheck` instance.** This is true in `main()` (line 48 creates it, lines 56-83 pass it to both functions). But if someone runs `injection_campaign()` from a different entry point with a separately constructed model, the MCMC burn-in (emcee sampler) would produce different initial samples. The audit should verify that the shared instance assumption holds for all realistic invocation paths.

2. **Dismissed alternative: Automated property-based testing.** Could use hypothesis/property-based testing to fuzz-test parameter consistency. Dismissed because the phase scope is a one-time audit, not ongoing testing. However, a simpler version (run both paths with the same seed, compare parameter logs) would be a valuable addition.

3. **Understated limitation: The injection campaign does NOT record spin (`a`) or eccentricity (`e0`) in the CSV.** The CSV columns are `[z, M, phiS, qS, SNR, h_inj, luminosity_distance]`. Since `a` is fixed at 0.98 and `e0` is randomized but not stored, waveform failure characterization by eccentricity is impossible from CSV data alone. This limits AUDT-03's ability to correlate failures with all relevant parameters.

4. **Simplification: The z_cut = 0.5 in injection is treated as "obviously safe."** While all 24 detections from the initial campaign were at z < 0.18, the injection campaign uses different h values. For h = 0.60, d_L at fixed z is smaller, potentially pushing the detection horizon to higher z. The round-trip test should include z up to 0.5 but the z_cut validity should be flagged for Phase 18 to assess with actual data.

5. **A physicist specializing in EMRI population modeling might note** that the polynomial fit for dN/dz (cosmological_model.py:70-126) is a fit to specific simulation data from Babak et al. (2017) and may not capture the full uncertainty in EMRI rates. This is out of scope for the audit (we verify consistency, not correctness of the rate model) but should be noted for the thesis discussion.

## Sources

### Primary (HIGH confidence)

- [Mandel, Farr & Gair (2019), "Extracting distribution parameters from multiple uncertain observations with selection biases", MNRAS 486, 1086](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.1086M/abstract) - Framework for selection effects in hierarchical inference
- [Farr (2019), "Accuracy Requirements for Empirically-Measured Selection Functions", arXiv:1904.10879](https://arxiv.org/abs/1904.10879) - N_eff criterion for injection sample sufficiency
- [Babak et al. (2017), "Science with LISA. V: EMRIs", Phys. Rev. D 95, 103012](https://arxiv.org/abs/1703.09722) - EMRI rate model M1 implemented in Model1CrossCheck
- [Hogg (1999), "Distance measures in cosmology", astro-ph/9905116](https://arxiv.org/abs/astro-ph/9905116) - Luminosity distance formula used in dist()

### Secondary (MEDIUM confidence)

- Phase 11.1 CONTEXT.md - Design decisions D-01 through D-09 governing injection campaign behavior
- Project-level SUMMARY.md (.gpd/research/SUMMARY.md) - Overall methodology and pitfall catalog

### Tertiary (LOW confidence)

- Waveform failure rate "30-50%" from project experience (documented in SUMMARY.md and CLAUDE.md memory) - not independently verified for the current injection campaign

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH - The audit compares well-understood functions (dist, Model1CrossCheck) that are already implemented and documented
- Standard approaches: HIGH - Line-by-line code comparison is the only reliable audit method; no ambiguity
- Computational tools: HIGH - All tools already in the dependency tree; no new installations needed
- Validation strategies: HIGH - Round-trip tests are straightforward; known limits are trivial

**Research date:** 2026-03-31
**Valid until:** Until the code in main.py or physical_relations.py is significantly modified. The audit findings are tied to specific line numbers and code versions.
