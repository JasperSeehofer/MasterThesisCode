# Phase 43: Posterior Calibration Fix — Research

**Researched:** 2026-04-27
**Domain:** Bayesian dark-siren H₀ inference / posterior normalisation / sky-coordinate frame algebra
**Confidence:** HIGH (all root causes confirmed from code; physics is Gray et al. 2020 well-understood)

---

## Summary

Phase 43 diagnoses and fixes the SC-3 failure from Phase 40 (VERIFY-03), where the extract_baseline
MAP was 0.860 instead of the expected 0.73 ± 0.01.

Two root causes have been confirmed by code trace in the prior session:

**H1 — Missing D(h) normalisation in extract_baseline/combine_posteriors (CONFIRMED):** The
`extract_baseline()` function in `evaluation_report.py` (lines 156-163) computes
`log_posterior = Σ_i log L_i(h)` with no `-N × log D(h)` term. This is mathematically required by
Gray et al. (2020) Eq. A.19. Without it, as h increases, more events find galaxy hosts (because the
detectable volume D(h) grows with h), so the raw likelihood sum increases monotonically → MAP biased
to h_max = 0.860. This is confirmed directly: the per-h nonzero count rises from 27/60 at h=0.60 to
37/60 at h=0.86.

**H2 — BallTree search center in wrong coordinate frame (CONFIRMED):** `Detection.__init__`
(detection.py:101-103) reads `phiS`/`qS` from the CRB CSV directly as `self.phi`/`self.theta`.
These CRBs were generated BEFORE Phase 36 (the equatorial→ecliptic rotation fix). The simulation's
parameter_space.py explicitly labels qS/phiS as "ecliptic coordinates" but the pre-Phase-36
simulation code produced CRBs in the equatorial frame. The v2.2 BallTree in handler.py is built
from ecliptic coordinates (after `_rotate_equatorial_to_ecliptic`). So the search center is passed
in equatorial frame to an ecliptic BallTree.

**The critical unknown** is whether H2 actually causes the --evaluate (v2.2) MAP to deviate from
0.73. The angular mismatch (up to 23.4°, obliquity of the ecliptic) is ~50× larger than the typical
EMRI sky localization error ellipse (median sigma_phi ≈ 0.49°, sigma_theta ≈ 0.81°), suggesting H2
severely impairs host galaxy recovery. However, `BayesianStatistics.evaluate()` DOES include D(h)
via `precompute_completion_denominator`, so `--evaluate` MAP should be unaffected by H1. The
question is whether H2 shifts the --evaluate MAP away from 0.73.

**Primary recommendation:** Run `--evaluate` at a single h-value as a diagnostic first step. If the
existing posteriors' h=0.73 likelihood is higher than h=0.86, H2 is not catastrophic. The minimal
fix plan is: (1) run the diagnostic --evaluate, (2) fix H2 by transforming CRBs at load time or via
a migration script, (3) fix H1 by adding D(h) to extract_baseline, (4) re-run full h-sweep on
cluster.

---

## User Constraints

See `.planning/phases/43-posterior-calibration-fix/.continue-here.md` for locked decisions:

- **FIX-01:** Root cause(s) identified (H1 and H2 already confirmed — this phase's first task
  verifies impact).
- **FIX-02:** Root cause(s) fixed (formula changes with [PHYSICS] prefix per /physics-change
  protocol).
- **FIX-03:** `--evaluate` confirms MAP ≈ 0.73 ± 0.01 (FORBIDDEN PROXY: extract_baseline MAP alone
  is not acceptable — must use the D(h)-corrected log-posterior from `--evaluate` sweep).
- Phase 41 (injection campaign): SKIPPED.
- Phase 42 (sky-dependent analysis): DEFERRED until Phase 43 resolved.
- VERIFY-04 anisotropy (|ΔMAP|=0.020, Q3): Stage-2 trigger for Phase 42 — NOT a Phase 43 blocker.

---

## Active Anchor References

| Anchor / Artifact | Type | Why It Matters Here | Required Action | Where It Must Reappear |
|---|---|---|---|---|
| Gray et al. (2020), arXiv:1908.06050, Eq. A.19 | Physics reference | Defines joint posterior with -N log D(h) | Read and cite correctly | Plan task that adds D(h) to extract_baseline |
| Gray et al. (2020), arXiv:1908.06050, Eq. 9 | Physics reference | Defines per-event likelihood L_i = f_i L_cat + (1-f_i) L_comp | Confirm L_comp already divides by D(h) internally | Plan section on why H1 fix is specifically in the combinator |
| `simulations/prepared_cramer_rao_bounds.csv` | Prior artifact | Contains 60 SNR≥20 CRBs with qS/phiS in EQUATORIAL frame (pre-Phase-36) | Audit frame, apply transform | Plan task: equatorial→ecliptic CRB migration |
| Phase 36 COORD-03 (astropy rotation) | Prior phase fix | The catalog is now in ecliptic; CRBs are still equatorial | Verify CRB frame; write transform script | CRB migration task |
| Phase 40 VERIFY-03 result: MAP=0.860 | Prior artifact | The baseline failure this phase must fix | Confirm it reproduces; confirm fix eliminates it | FIX-03 verification task |
| Phase 40 VERIFY-04 result: Q3 |ΔMAP|=0.020 | Prior artifact | May resolve after H2 fix (wrong centers → artificial sky anisotropy) | Re-check after H2 fix | Post-fix anisotropy assessment |

**Missing or weak anchors:** None. All required anchors are present and well-documented.

---

## Conventions

| Choice | Convention | Alternatives | Source |
|---|---|---|---|
| Coordinate frame (catalog) | BarycentricTrueEcliptic J2000 (ecliptic SSB) | ICRS equatorial | handler.py:_rotate_equatorial_to_ecliptic (Phase 36) |
| Coordinate frame (CRBs) | EQUATORIAL ICRS J2000 (pre-Phase-36) | ecliptic SSB | parameter_space.py generated CRBs before Phase 36 |
| Sky angles | phiS = azimuth (0 to 2π), qS = polar angle (0 to π) | latitude | few/fastemriwaveforms convention |
| Posterior | log-space, unnormalised | probability space | Gray et al. (2020) Eq. A.19 |
| H₀ units | dimensionless (h = H₀ / 100 km/s/Mpc) | km/s/Mpc | project convention |
| Unit system | SI via astropy for angle transforms; natural (c=1) elsewhere | — | project convention |

**CRITICAL:** The CRBs in the CSV use the EQUATORIAL ICRS convention for qS/phiS. The galaxy catalog
after Phase 36 uses the ECLIPTIC SSB convention. These are NOT the same. The ecliptic is tilted 23.4°
relative to the equatorial plane.

---

## Mathematical Framework

### Key Equations and Starting Points

| Equation | Name/Description | Source | Role in This Phase |
|---|---|---|---|
| `log p(h\|data) = Σ_i log L_i(h) - N × log D(h) + const` | Joint posterior (Dark Siren) | Gray et al. (2020) Eq. A.19 | H1 fix: add -N log D(h) to extract_baseline |
| `L_i(h) = f_i(h) × L_cat + (1 - f_i(h)) × L_comp` | Per-event likelihood | Gray et al. (2020) Eq. 9 | Per-event combination (already correctly computed) |
| `L_comp = N_i(h) / D(h)` | Completion term | Gray et al. (2020) Eqs. 31-32 + A.19 | D(h) already appears in L_comp numerator/denominator — see NOTE below |
| `D(h) = ∫ P_det(d_L(z,h)) dVc/dz dz` | Detectable volume (normalisation) | Gray et al. (2020) Eq. A.19 | Already computed by precompute_completion_denominator; used in BayesianStatistics.evaluate() |
| `SkyCoord(ra, dec, frame='icrs').transform_to(BarycentricTrueEcliptic('J2000'))` | Equatorial→Ecliptic rotation | astropy / IAU | H2 fix: transform CRB qS/phiS columns |
| `θ_polar = π/2 - latitude` | Ecliptic latitude → polar angle | standard spherical polar | Used in handler.py _map_angles_to_spherical_coordinates (Phase 36) |

**NOTE on D(h) double-counting:** The L_comp term already contains D(h) in its denominator (it's
N_i(h)/D(h)). The additional `-N × log D(h)` in the joint posterior is a SEPARATE normalisation for
the SELECTION EFFECT — it accounts for the fact that events with higher h are more likely to be
detected. This is NOT the same D(h) that appears in L_comp. See Gray et al. (2020) Eq. A.19 for
the derivation showing why both terms appear.

**Practical meaning:** Without `-N log D(h)`, as h increases:
- D(h) grows (more volume is detectable)
- More events find galaxy hosts (nonzero L_i)
- Sum log L_i grows monotonically with h → MAP at h_max = 0.860

This is confirmed by the data: nonzero-likelihood event count grows from 27/60 at h=0.60 to 37/60
at h=0.86. The log_posterior curve is monotonically increasing (confirmed by examining all 38
h-values: 154.3 at h=0.60 → 199.3 at h=0.86).

### Required Techniques

| Technique | What It Does | Where Applied | Standard Reference |
|---|---|---|---|
| astropy coordinate transform | Equatorial ICRS J2000 → BarycentricTrueEcliptic J2000 | CRB migration: transform qS/phiS columns | handler.py:_rotate_equatorial_to_ecliptic (Phase 36) |
| precompute_completion_denominator | Computes D(h) for each h via fixed_quad integration | H1 fix: call for all 38 h-values in the existing posteriors | bayesian_statistics.py:70-151 |
| log-space MAP estimation | argmax(Σ log L_i - N log D) | H1 fix: add D(h) correction after loading existing posteriors | evaluation_report.py:252-253 |
| BallTree Cartesian sky embedding | (sin θ cos φ, sin θ sin φ, cos θ) | Host galaxy lookup | handler.py:_polar_to_cartesian (Phase 36) |
| COORD-04 eigenvalue search radius | λ_max of J Σ J^T where J = diag(|sin θ|, 1) | BallTree query radius | handler.py:349-357 |

### Approximation Schemes

| Approximation | Small Parameter | Regime of Validity | Error Estimate | Alternatives if Invalid |
|---|---|---|---|---|
| Fisher matrix CRBs as true posteriors | SNR >> 1 (SNR ≥ 20) | Valid for well-localised events | O(1/SNR²) | Full MCMC posterior sampling |
| Gaussian sky error ellipse | sigma_phi, sigma_theta << 1 rad | Valid: median sigma ≈ 0.5° << 1 rad | Negligible | Non-Gaussian sky prior |
| Equatorial→ecliptic frame as static rotation | Precession neglected | Valid for J2000 epoch comparison | < 0.01° | Epoch-specific rotation |

---

## Standard Approaches

### Approach 1: Minimal Code Fix + Re-sweep (RECOMMENDED)

**What:** Fix H2 by transforming CRBs at detection load time (in Detection.__init__ or at CSV load),
fix H1 by adding D(h) correction in extract_baseline/combine_posteriors, then run --evaluate sweep
on cluster to regenerate all 38 h-value posteriors.

**Why standard:** Preserves the existing data flow. The existing h_*.json posteriors were computed
with the wrong search center — they cannot be trusted (the BallTree returned wrong candidate hosts).
New posteriors must be generated with corrected CRBs.

**Key steps:**

1. **Diagnostic:** Run `--evaluate` at h=0.73 with v2.2 code + equatorial CRBs. Check if host
   galaxy lookup fails (zero likelihoods) due to H2. Compare log-likelihood to h=0.86 from
   existing posteriors to establish whether H2 affects --evaluate MAP.
2. **H2 fix:** Transform `qS`/`phiS` columns from equatorial ICRS to ecliptic SSB in the CRBs
   at load time in Detection.__init__ OR write a one-time migration script that updates
   `simulations/prepared_cramer_rao_bounds.csv`.
3. **H1 fix:** Add `-N × log D(h)` to the log-posterior in `extract_baseline` / `combine_posteriors`.
   This requires calling `precompute_completion_denominator` at extract time for each h-value.
   ALTERNATIVELY: define that MAP is always reported from `--evaluate` runs (which DO include D(h)
   via BayesianStatistics.evaluate), making extract_baseline a deprecated diagnostic-only tool.
4. **Re-sweep:** Run full 38-h-value cluster evaluation with corrected CRBs.
5. **Verify:** Check --evaluate MAP ≈ 0.73 ± 0.01 (FIX-03).

**Known difficulties at each step:**

- Step 1: --evaluate is expensive (full multiprocessing pool, ~15 min per h-value on cluster). On
  local machine it may be slow. Alternatively, examine existing diagnostics log for h=0.73 results.
- Step 2: The CRB frame transform must match exactly what Phase 36 did (BarycentricTrueEcliptic
  J2000, then convert lat→polar: θ_polar = π/2 - latitude). Must be identical to handler.py
  `_rotate_equatorial_to_ecliptic` + `_map_angles_to_spherical_coordinates`.
- Step 3: D(h) computation requires SimulationDetectionProbability (needs injection data). This is
  available locally. Estimated cost: ~2 min per h-value for fixed_quad (38 × 2 min = 76 min
  total, parallelisable).
- Step 4: Cluster job array (38 tasks × ~15 min = cluster time). Must be submitted as SLURM array.
- Step 5: FIX-03 uses --evaluate MAP, not extract_baseline MAP.

### Approach 2: Load-time Transform Only, Skip Re-sweep (FALLBACK if cluster unavailable)

**What:** Apply H2 fix at detection load time (transform CRBs in Detection.__init__ without touching
the CSV), apply H1 fix to extract_baseline, but use EXISTING h_*.json posteriors with corrected
combination formula.

**When to switch:** If cluster is unavailable and the diagnostic shows existing posteriors already
have correct D(h) correction (i.e., the issue is purely in extract_baseline, not in the per-event
likelihoods stored in JSONs).

**Tradeoffs:** The existing h_*.json per-event likelihoods were computed with wrong search centers
(equatorial BallTree center on ecliptic tree). They cannot be trusted — different events will have
found the wrong host candidates. This approach only fixes the combination step, not the likelihood
data itself. NOT RECOMMENDED unless there is compelling evidence H2 has no impact.

### Anti-Patterns to Avoid

- **Using extract_baseline MAP as FIX-03 success metric:** extract_baseline currently lacks D(h)
  correction; even after fix, the --evaluate sweep is the ground truth.
  - _Example:_ Prior VERIFY-02 used archived v2.1 posteriors with extract_baseline → produced
    MAP=0.735 which was NOT a true v2.2 result.
- **Modifying the h_*.json posteriors in-place:** The per-event likelihoods in existing JSONs were
  computed with wrong BallTree centers. Do not attempt to patch them mathematically — regenerate.
- **Applying the ecliptic transform twice:** handler.py already rotates the catalog from equatorial
  to ecliptic. The CRB transform must match exactly. Using astropy with the same equinox='J2000'
  argument is required for consistency.
- **Computing D(h) from a re-scaled heuristic:** D(h) must come from `precompute_completion_denominator`
  using the actual `SimulationDetectionProbability` P_det grid, not from any approximation.

---

## Existing Results to Leverage

### Established Results (DO NOT RE-DERIVE)

| Result | Exact Form | Source | How to Use |
|---|---|---|---|
| Gray et al. (2020) joint posterior | log p(h\|data) = Σ log L_i - N log D + const | arXiv:1908.06050, Eq. A.19 | Direct template for H1 fix |
| precompute_completion_denominator | D_h_table = {h: float} via fixed_quad | bayesian_statistics.py:70-151 | Call this function; do not re-implement |
| astropy equatorial→ecliptic rotation | handler.py:_rotate_equatorial_to_ecliptic (lines 574-613) | Phase 36 COORD-03 | Copy exactly for CRB migration |
| _map_angles_to_spherical_coordinates | θ_polar = π/2 - lat, φ = lon | handler.py:614-648 | Copy exactly for CRB migration |
| COORD-04 eigenvalue search radius | lambda_max from J Σ J^T | handler.py:349-357 | Already in production path — no change needed |

**Key insight:** The D(h) computation infrastructure already exists in bayesian_statistics.py. The
H1 fix is a 5-10 line change to extract_baseline to call precompute_completion_denominator and add
the -N log D(h) correction. Do NOT re-derive or re-implement D(h).

### Useful Intermediate Results

| Result | What It Gives You | Source | Conditions |
|---|---|---|---|
| Monotonically increasing log_posterior (no D(h)) | Proof H1 is real | Current posteriors in simulations/ | Confirmed: 154.3 → 199.3 as h goes 0.60 → 0.86 |
| Nonzero-event-count growth with h | Confirms D(h) effect: more detectable events at higher h | Current posteriors: 27/60 → 37/60 | Confirmed numerically |
| median sigma_phi ≈ 0.49°, sigma_theta ≈ 0.81° | Typical sky localization, vs 23.4° mismatch | CRB CSV (60 events, SNR≥20) | Shows H2 mismatch (≈50×) is catastrophic |
| VERIFY-04 Q3 |ΔMAP|=0.020 (5.4σ) | Sky anisotropy exists with wrong BallTree centers | Phase 40 VERIFY-04 audit | May resolve after H2 fix |

### Relevant Prior Work

| Paper/Result | Authors | Year | Relevance | What to Extract |
|---|---|---|---|---|
| Dark siren cosmology with galaxy catalog | Gray et al. | 2020 | Core methodology | Eq. A.19 (joint posterior with D(h)), Eq. 9 (per-event likelihood), Eqs. 24-25 (L_cat), Eqs. 31-32 (L_comp) |
| astropy BarycentricTrueEcliptic | Astropy Collaboration | 2022 | Coordinate transform | Already implemented in Phase 36 — reuse exactly |

---

## Computational Tools

### Core Tools

| Tool | Version/Module | Purpose | Why Standard |
|---|---|---|---|
| astropy.coordinates | SkyCoord, BarycentricTrueEcliptic | CRB equatorial→ecliptic transform | Same library used in Phase 36 handler.py |
| scipy.integrate.fixed_quad | n=100 (default) | D(h) integration | Already in precompute_completion_denominator |
| SimulationDetectionProbability | master_thesis_code.bayesian_inference.simulation_detection_probability | P_det grid for D(h) | Required by precompute_completion_denominator |
| uv run python3 -m master_thesis_code simulations/ --evaluate --h_value H | CLI | Single-h evaluation | Production evaluation path |

### Supporting Tools

| Tool | Purpose | When to Use |
|---|---|---|
| uv run pytest -m "not gpu and not slow" | Regression check before/after fix | After each code change |
| simulations/diagnostics/event_likelihoods.csv | Per-event L_cat/L_comp/f_i diagnostic | Checking whether host galaxies are found |
| .planning/debug/anisotropy_audit_*.md | VERIFY-04 baseline result | Comparing before/after anisotropy |

### Computational Feasibility

| Computation | Estimated Cost | Bottleneck | Mitigation |
|---|---|---|---|
| Single --evaluate run (local) | 15-30 min (60 events, 14 workers) | Multiprocessing pool + BallTree lookup per event | Run at h=0.73 only for diagnostic |
| D(h) for 38 h-values | ~2 min/h × 38 = 76 min serial; ~10 min parallel | fixed_quad integration | Parallelise with multiprocessing |
| Full 38-h cluster sweep | 38 × 15 min SLURM array, ~15 min wall time | CPU array job | Submit as evaluate.sbatch array (already configured) |
| CRB migration script | < 1 min | Pandas + astropy vectorised | Trivial |

**Installation / Setup:**
No new packages needed. astropy is already in the project dependencies.

---

## Validation Strategies

### Internal Consistency Checks

| Check | What It Validates | How to Perform | Expected Result |
|---|---|---|---|
| D(h) increases monotonically with h | Physical: larger h → more detectable volume | Print D_h_table after precompute_completion_denominator | D(0.86) > D(0.73) > D(0.60) |
| After H1 fix: log_posterior has peak near h=0.73 | H1 fix correct | Run extract_baseline with D(h) correction on existing posteriors | MAP shifts from 0.860 toward 0.73 (or confirms H2 effect) |
| After H2 fix: host found for injected event at known sky position | H2 fix correct | Check event with known ecliptic sky position finds a host galaxy | Host galaxy index matches original host_galaxy_index in CRB CSV |
| Coordinate roundtrip: equatorial → ecliptic → equatorial = identity | No numerical errors in transform | Apply forward + inverse transform to qS/phiS columns | Residuals < 0.001° |

### Known Limits and Benchmarks

| Limit | Parameter Regime | Known Result | Source |
|---|---|---|---|
| h = h_true = 0.73 | Correct injection h-value | MAP ≈ 0.73 ± 0.01 | FIX-03 target; Phase 32 result (MAP=0.73, bias=0%) before Phase 36 broke the frame |
| VERIFY-04 anisotropy after H2 fix | If H2 was causing sky-quartile bias | |ΔMAP_q| should drop toward < 1σ | Phase 40 VERIFY-04 triggered Q3 at 5.4σ |

### Numerical Validation

| Test | Method | Tolerance | Reference Value |
|---|---|---|---|
| Coordinate transform accuracy | compare_to_astropy_ground_truth(qS_transformed, phiS_transformed) | < 0.001° | astropy BarycentricTrueEcliptic J2000 |
| D(h=0.73) magnitude | Print and compare to prior Phase 32 log | order of magnitude consistent | Phase 32 used same precompute_completion_denominator |
| MAP after full fix | --evaluate sweep + extract_baseline with D(h) | 0.73 ± 0.01 | Project target; Gray et al. (2020) methodology |

### Red Flags During Computation

- If the D(h)-corrected extract_baseline still gives MAP=0.860: D(h) computation is wrong or
  the N factor is incorrect (check N = number of events with nonzero likelihood).
- If after H2 fix the host galaxy lookup returns 0 hosts for events that should have hosts: the
  coordinate transform is wrong (check astropy equinox argument; must match exactly J2000).
- If the D(h) curve is FLAT across h-values: P_det grid or cosmological model is broken (D(h)
  should grow significantly between h=0.60 and h=0.86 because dl_max scales with h via the
  Hubble distance).
- If extract_baseline MAP with D(h) correction = 0.73 but --evaluate MAP ≠ 0.73: H2 is affecting
  the per-event likelihoods (which confirms the existing h_*.json posteriors are corrupted by H2
  and must be regenerated).

---

## Common Pitfalls

### Pitfall 1: Confusing the Two D(h) Appearances in the Formula

**What goes wrong:** There are TWO places D(h) appears: (a) inside L_comp = N_i(h)/D(h), and (b) in
the joint posterior normalization `-N × log D(h)`. A developer might assume the L_comp division
already handles the normalization.

**Why it happens:** The formula in Gray et al. (2020) Eq. 9 (per-event likelihood) and Eq. A.19
(joint posterior) look similar but serve different purposes. L_comp accounts for the probability of
the event being associated with an uncatalogued host; the -N log D(h) accounts for the selection
effect (only detecting events above the SNR threshold).

**How to avoid:** The H1 fix goes in `extract_baseline` (or `combine_posteriors`), NOT in
`single_host_likelihood`. The per-event likelihoods stored in h_*.json are correctly computed.

**Warning signs:** If MAP remains at 0.860 after adding D(h) to extract_baseline, the sign is wrong
or the N count is wrong.

**Recovery:** Verify sign: the correct formula is `log_post = sum_log_L - N * log(D_h)`. Since D(h)
grows with h, `-N log D(h)` decreases with h, counteracting the upward trend in `sum_log_L`.

### Pitfall 2: Wrong N in -N log D(h)

**What goes wrong:** N is the number of DETECTED events contributing to the posterior, not the total
number of events in the CSV (60 events) or the number with nonzero likelihood.

**Why it happens:** Per Gray et al. (2020) Eq. A.19, N is the number of gravitational wave events
observed above the threshold, which equals the number of events with any likelihood (including zero).

**How to avoid:** N = n_detections (total events processed by BayesianStatistics.evaluate), NOT
only the nonzero-likelihood events. Currently available as `n_detections` in the h_*.json structure.

**Warning signs:** MAP is too far from 0.73 in either direction.

### Pitfall 3: Equatorial→Ecliptic Transform Mismatch

**What goes wrong:** The CRB migration transform must exactly match the handler.py Phase 36
implementation. Using a different astropy frame name, epoch, or conversion convention will result
in a small but non-zero systematic offset.

**Why it happens:** The equatorial→ecliptic transform depends on the equinox. J2000 is used
throughout Phase 36 (handler.py line 596: `BarycentricTrueEcliptic(equinox='J2000')`).

**How to avoid:** Copy the exact astropy call from handler.py:_rotate_equatorial_to_ecliptic and
handler.py:_map_angles_to_spherical_coordinates. The conversion from lat/lon to polar angle is
`theta_polar = pi/2 - lat_rad`, `phi = lon_rad`.

**Warning signs:** Coordinate roundtrip test fails; reconstructed ecliptic positions differ from
GLADE catalog ecliptic positions by > 0.01°.

### Pitfall 4: Re-using Existing h_*.json Posteriors After Only H1 Fix

**What goes wrong:** If H2 is causing significant likelihood corruption (wrong host galaxies found),
then fixing only H1 (adding D(h) to extract_baseline) and re-using the existing posteriors will give
an incorrect MAP even with the D(h) correction.

**Why it happens:** The existing h_*.json files store per-event likelihoods computed with equatorial
BallTree centers on an ecliptic tree. For events where the ecliptic mismatch > sigma_sky, no host
galaxy was found → likelihood = 0 → those events contribute nothing to the posterior.

**How to avoid:** The diagnostic --evaluate run at h=0.73 is the critical gate. If it gives MAP ≈
0.73, existing posteriors may be usable. If --evaluate at h=0.73 gives significantly different
results from what's in the existing h_*.json, a full re-sweep is required.

**Warning signs:** After H2 fix, the h=0.73 likelihood from a fresh --evaluate run is very different
from the h_0_73.json stored value.

### Pitfall 5: VERIFY-04 Anisotropy as Ambiguous Indicator

**What goes wrong:** The Q3 anisotropy (|ΔMAP|=0.020, 5.4σ) was computed with the wrong BallTree
centers. After H2 fix, the anisotropy may decrease significantly, making Phase 42 (sky-dependent
analysis) unnecessary. But the researcher may prematurely conclude Phase 42 is unneeded.

**How to avoid:** Re-run VERIFY-04 after the H2 fix using the new posteriors. If |ΔMAP_q| < 1σ for
all quartiles, Phase 42 is resolved. If it remains > 1σ, Phase 42 proceeds as planned.

**Warning signs:** Re-running VERIFY-04 with the old (wrong BallTree center) posteriors and claiming
it validates Phase 42 trigger.

---

## Level of Rigor

**Required for this phase:** Controlled approximation (not formal proof). The physics is established;
the task is diagnostic + implementation.

**What this means concretely:**

- Coordinate transform must be validated numerically (roundtrip test, < 0.001° residuals).
- D(h) correction must use the existing production function `precompute_completion_denominator`; no
  approximation.
- MAP estimation can use grid argmax (existing approach); no interpolation required for FIX-03.
- The --evaluate run is the mandatory success criterion (not extract_baseline MAP).

---

## Diagnostic --evaluate Run Design

**Command:**
```bash
uv run python -m master_thesis_code simulations/ --evaluate --h_value 0.73 --log_level INFO
```

**What to look for in the log:**
- `"Computing posteriors for h = 0.73..."` — confirms evaluate() is running
- `"possible hosts found X/Y..."` — per-event host count. If X=0 for most events, H2 is severe.
- `"D(h=0.7300) = X.XXXXE+XX"` — D(h) value at h=0.73
- `"posteriors computed for h = 0.73"` — completion
- After run: examine `simulations/posteriors/h_0_73.json` and compare per-event likelihoods to
  existing file (which was generated with wrong BallTree center).

**How to extract MAP from --evaluate results (NOT extract_baseline):**
```python
# After running --evaluate for ALL 38 h-values, MAP is found by:
# 1. Call precompute_completion_denominator for all h-values
# 2. For each h-value JSON: log_post = sum_log_L - N * log(D_h)
# 3. argmax over h-values
```

**Key comparison:** If `simulations/diagnostics/event_likelihoods.csv` exists after the h=0.73
run, compare `L_cat_no_bh` and `L_cat_with_bh` and `f_i` values to the values from an earlier
h=0.73 run. If host finding is severely impaired, L_cat terms will be near-zero.

**What MAP from --evaluate means:**
- The BayesianStatistics.evaluate() code at bayesian_statistics.py:349-358 calls
  `precompute_completion_denominator` and passes `_D_h_table` to worker processes.
- The D(h) table is used in `p_Di` at line 1022 for `comp_denominator = self._D_h_table.get(self.h)`.
- This means the per-event `combined_without_bh_mass` value ALREADY includes D(h) in the L_comp
  term (via N_i/D(h)).
- BUT: the `posterior_data` dict stored in h_*.json still contains just the per-event likelihoods,
  not the joint posterior. The MAP from the joint posterior still requires adding -N log D(h) across
  h-values.
- **Conclusion:** Even with corrected CRBs (H2 fix), the MAP from `extract_baseline` will STILL
  be wrong unless H1 (the -N log D(h) term in combination) is also fixed.

---

## Fix Approaches

### H1 Fix: Add -N log D(h) to extract_baseline

**Location:** `master_thesis_code/bayesian_inference/evaluation_report.py:load_posteriors()` (lines
148-175), specifically the `log_posterior` computation at lines 156-163.

**Change:**
```python
# Current (WRONG — missing -N * log D(h)):
log_posterior = 0.0
for key in detection_keys:
    lk_list = data[key]
    if isinstance(lk_list, list) and len(lk_list) > 0:
        lk = float(lk_list[0])
        if lk > 0:
            log_posterior += np.log(lk)

# Fix option A — add D(h) correction inline (requires D_h_table as parameter):
log_posterior = sum_log_L - n_detections * np.log(D_h)
```

**Alternative H1 fix (simpler, less coupled):** Deprecate `extract_baseline` as the MAP reporter.
Define FIX-03 as: "run `--evaluate` for all 38 h-values, compute MAP directly from h_*.json
log-posteriors with D(h) correction, report that MAP." The extract_baseline function remains for
backward compatibility but is documented as NOT including D(h).

**Physics Change Protocol:** Required — this changes the MAP formula. Must follow `/physics-change`
with old formula, new formula, reference (Gray et al. 2020 Eq. A.19), dimensional analysis, and
limiting case (h=h_true should give MAP near h_true).

### H2 Fix: Transform CRBs at Load Time

**Option A — CSV migration (recommended for reproducibility):**
Write a one-time migration script that reads `simulations/prepared_cramer_rao_bounds.csv`,
transforms `qS` and `phiS` columns from equatorial to ecliptic frame (using astropy), writes
back to the CSV. Tag the CSV with a metadata comment or renamed backup.

```python
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
import astropy.units as u

df = pd.read_csv('simulations/prepared_cramer_rao_bounds.csv')

# qS and phiS are in equatorial ICRS (pre-Phase-36). Convert to ecliptic SSB.
# In equatorial: phiS = RA (rad), qS = polar angle = pi/2 - Dec (rad)
ra_rad = df['phiS'].values
dec_rad = np.pi/2 - df['qS'].values  # polar -> latitude

coord = SkyCoord(ra=np.degrees(ra_rad)*u.deg, dec=np.degrees(dec_rad)*u.deg, frame='icrs')
ecl = coord.transform_to(BarycentricTrueEcliptic(equinox='J2000'))

# Convert ecliptic lon/lat to polar angle convention used by handler.py
lon_rad = np.radians(ecl.lon.deg)
lat_rad = np.radians(ecl.lat.deg)
theta_polar = np.pi/2 - lat_rad  # ecliptic polar angle in [0, pi]
phi_ecl = lon_rad % (2 * np.pi)  # ecliptic azimuth in [0, 2pi)

df['phiS'] = phi_ecl
df['qS'] = theta_polar
df.to_csv('simulations/prepared_cramer_rao_bounds.csv', index=False)
```

**Option B — Load-time transform in Detection.__init__:** Add the transform in detection.py
`__init__`. Disadvantage: runs the transform N×38 times (once per event per h-value sweep).

**Recommendation:** Option A (CSV migration) is preferred. It is transparent, runs once, and the
corrected CSV can be version-tracked. Add a column `_coord_frame = 'ecliptic_BarycentricTrue_J2000'`
as a guard against re-applying the transform.

**Physics Change Protocol:** Required — changes the sky coordinates used for host galaxy lookup.

---

## Post-Fix VERIFY-04 Anisotropy Re-assessment

After the full H2 + H1 fix and cluster re-sweep:

1. Re-run the Phase 40 VERIFY-04 quartile anisotropy audit with the new posteriors.
2. Compute per-quartile MAP_q on |qS − π/2| equal-count bins using corrected posteriors.
3. Compare |MAP_q − MAP_total| to σ = (CI_upper − CI_lower) / 2.
4. If all quartiles pass (|ΔMAP_q| < σ): Phase 42 is resolved; no sky-dependent analysis needed.
5. If Q3 still fails: Phase 42 proceeds.

**Hypothesis:** The VERIFY-04 Q3 trigger was caused by H2 (wrong BallTree centers biased host
galaxy recovery as a function of ecliptic latitude quartile). After H2 fix, the anisotropy should
reduce significantly. However, the ecliptic-equatorial mismatch is not uniformly distributed across
the sky — events near the ecliptic have nearly zero mismatch while polar events have large mismatch
(up to 23.4°). This could explain the Q3 anisotropy (mid-polar region where mismatch is
intermediate).

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| Posterior MAP from extract_baseline (sum log L_i) | Posterior MAP from --evaluate sweep with D(h) correction | This phase | Fixes SC-3 MAP=0.860 → ~0.73 |
| CRBs in equatorial frame | CRBs in ecliptic frame (matching v2.2 catalog) | Phase 36 (catalog), Phase 43 (CRBs) | Correct host galaxy recovery |
| VERIFY-04 anisotropy with wrong BallTree centers | VERIFY-04 with corrected centers | After H2 fix | May resolve Phase 42 trigger |

---

## Open Questions

1. **Does --evaluate MAP with v2.2 code + equatorial CRBs give ≈0.73 or ≠0.73?**
   - What we know: `BayesianStatistics.evaluate()` includes D(h) in L_comp (via N_i/D(h)), but the
     BallTree search center is wrong (equatorial center on ecliptic tree, mismatch up to 23.4°).
   - What's unclear: Whether the wrong center prevents finding any hosts at all, or whether nearby
     galaxies (within sigma_multiplier * lambda_max radius) are still found despite the offset.
   - Impact: If --evaluate MAP ≈ 0.73 already, then existing posteriors are usable (just need H1
     fix in extract_baseline). If not, full re-sweep required.
   - Recommendation: Run `--evaluate` at h=0.73 locally (diagnostic, ~20 min). This is Task 1 of
     the plan.
   - Estimate: Given sigma_sky ≈ 0.5° and mismatch ≈ 23.4°, the mismatch is ~50× the search
     radius. The BallTree search uses `sigma_multiplier=1.5`, so radius ≈ 1.5 × 0.5° ≈ 0.75°.
     For most events, the equatorial center will point ~23.4° away from the true ecliptic position,
     far outside the 0.75° search radius. Host recovery will be severely impaired for most events.
     **Strong prior: --evaluate MAP ≠ 0.73 when H2 is present.**

2. **Does VERIFY-04 Q3 anisotropy resolve after H2 fix?**
   - What we know: The anisotropy is measured by |qS − π/2| quartiles (ecliptic latitude).
     H2 (equatorial center on ecliptic BallTree) biases recovery differently across ecliptic
     latitudes.
   - What's unclear: The exact dependence of the recovery rate on ecliptic latitude; whether the
     bias is structured enough to explain a 5.4σ Q3 anomaly.
   - Impact: If resolved, Phase 42 is unnecessary. If not, Phase 42 proceeds.
   - Recommendation: Re-run VERIFY-04 after H2 fix. Do not pre-judge.

---

## Alternative Approaches if Primary Fails

| If This Fails | Because Of | Switch To | Cost of Switching |
|---|---|---|---|
| CSV migration (H2 fix) | Transform introduces unexpected errors | Load-time transform in Detection.__init__ | Small (~1 hour) |
| Cluster re-sweep | SLURM queue / cluster unavailable | Local re-sweep (38 × 20 min ≈ 12 hours serial, parallelisable) | ~1 day compute time |
| extract_baseline with D(h) | Adds too much complexity to load_posteriors | Define MAP always from --evaluate, not extract_baseline | Architecture change (extract_baseline becomes diagnostic-only) |

**Decision criteria:** If the local diagnostic --evaluate at h=0.73 shows > 80% of events have
zero likelihood, proceed to full H2+H1 fix + re-sweep. If < 20% of events have zero likelihood,
the H2 impact is smaller than expected — investigate further before deciding whether to re-sweep.

---

## Sources

### Primary (HIGH confidence)

- [Gray et al. (2020), arXiv:1908.06050] — Dark siren methodology. Eqs. 9, A.19, 24-25, 31-32.
- [bayesian_statistics.py:70-151] — precompute_completion_denominator (verified against code).
- [evaluation_report.py:148-175] — load_posteriors missing D(h) (verified against code).
- [detection.py:101-103] — Detection.__init__ reads equatorial CRBs (verified against code).
- [handler.py:574-648] — Phase 36 equatorial→ecliptic rotation (verified against code).

### Secondary (MEDIUM confidence)

- [handler.py:324-398] — BallTree search with COORD-04 eigenvalue radius — correct implementation.
- [Phase 40 VERIFY-04 audit] — Q3 |ΔMAP|=0.020, 5.4σ — measured before H2 fix.
- [simulations/posteriors/ inspection] — Monotonically increasing log_posterior confirmed (code run).

### Tertiary (LOW confidence)

- [Estimate: H2 mismatch severity] — Based on sigma_sky ≈ 0.5° vs 23.4° mismatch, ~50× outside
  search radius. Strong prior that H2 severely impairs host recovery. Not confirmed by --evaluate run.

---

## Metadata

**Confidence breakdown:**

- Mathematical framework: HIGH — Gray et al. (2020) formula verified; code path traced fully.
- Standard approaches: HIGH — fix options are standard Pandas + astropy; no novel methods needed.
- Computational tools: HIGH — all tools are in existing project dependencies.
- Validation strategies: HIGH — test protocol mirrors Phase 36 + Phase 40 methodology.

**Research date:** 2026-04-27
**Valid until:** Stable; physics is well-established. astropy coordinate transforms are deterministic.
Code citations valid as of commit e013151.

---

## Derivation State Note

**KNOWN (confirmed from code):**

- H1: `load_posteriors()` in `evaluation_report.py` (lines 156-163) sums log-likelihoods with NO
  `-N × log D(h)` term → MAP monotonically biased to h_max = 0.860. Confirmed by running the
  sum on all 38 h_*.json files: 154.3 → 199.3 monotonically.
- H2: `Detection.__init__` (detection.py:101-103) reads equatorial qS/phiS from CRBs generated
  pre-Phase-36. The BallTree in handler.py is built from ecliptic coordinates. Mismatch up to
  23.4° vs median search radius ≈ 0.75°.
- `BayesianStatistics.evaluate()` DOES include D(h) via `precompute_completion_denominator`
  (bayesian_statistics.py:349-358), so the D(h) correction is available in the production path.
- Nonzero-event count grows 27/60 → 37/60 as h: 0.60 → 0.86, confirming the D(h) monotonic bias.
- CRBs were generated before Phase 36 (git log shows Phase 36 commits fixed handler.py, not the
  simulation pipeline). The prepared_cramer_rao_bounds.csv was last modified 2026-04-22 (but this
  was a cluster rsync/merge of data generated by pre-Phase-36 code).

**UNKNOWN (requires code run):**

- Whether `--evaluate` at h=0.73 with v2.2 code + equatorial CRBs gives MAP ≈ 0.73 or ≠ 0.73.
  - **Strong prior:** H2 mismatch is ~50× sigma_sky → host recovery severely impaired → MAP ≠ 0.73.
  - **Required action:** Run `uv run python -m master_thesis_code simulations/ --evaluate --h_value 0.73`
    and check per-event `L_cat_no_bh` in `simulations/diagnostics/event_likelihoods.csv`.
- Whether VERIFY-04 Q3 anisotropy resolves after H2 fix (requires cluster re-sweep).
- The exact D(h) values for all 38 h-values in the current v2.2 grid (must run
  `precompute_completion_denominator`; not stored anywhere in the current outputs).
