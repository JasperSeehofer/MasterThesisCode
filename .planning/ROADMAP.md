# Roadmap: v2.2 Pipeline Correctness

**Milestone:** v2.2 Pipeline Correctness
**Defined:** 2026-04-21
**Source plan:** `~/.claude/plans/i-want-a-last-elegant-feather.md`
**Granularity:** standard (8 phases)
**Coverage:** 28/28 requirements mapped

> **Scope note:** This ROADMAP.md is v2.2-scoped only. The prior cumulative roadmap is preserved at `.planning/milestones/v2.1-cumulative-ROADMAP.md`. Individual shipped milestone roadmaps live at `.planning/milestones/v{1.0,1.1,1.2,1.3,1.4,2.1-biasres}-ROADMAP.md`. Last shipped v2.1 BiasRes phase was Phase 34. v2.1 Publication Figures (Phases 29, 35 already completed; 36-38 paused) will resume post-v2.2 — the PubFigs Phase 35 number collides with the v2.2 Phase 35 chosen here; since v2.2 is the active milestone and the user instruction was explicit ("Start at Phase 35"), PubFigs-paused phases will need renumbering when resumed.

## Goal

Fix all 10 findings from the 2026-04-21 pre-batch audit — two critical coordinate-frame bugs (no equatorial→ecliptic rotation + singular BallTree embedding at ecliptic equator), plus L_cat drift from Gray Eq. 24-25, P_det extrapolation asymmetry, h-hardcode in Fisher CRBs, uniform derivative_epsilon, and latent correctness hygiene — before investing new cluster compute on the next simulation batch and extended P_det injection run.

**Verification gate:** existing CRBs re-evaluated under fixed frame. Abort new cluster compute if MAP at h=0.73 shifts >5% from v2.1 baseline MAP=0.73.

## Phases

- [x] **Phase 35: Coordinate Bug Characterization** [GSD] — Failing tests + baseline equator-band count before any fix (completed 2026-04-21)
- [x] **Phase 36: Coordinate Frame Fix** [GPD] — Equatorial→ecliptic rotation, correct polar Cartesian embedding, eigenvalue sky ellipse (completed 2026-04-22)
- [x] **Phase 37: Parameter Estimation Correctness** [GSD+GPD] — Thread h into host-distance, per-parameter derivative_epsilon, plus hygiene (Omega_m limits, unified SNR threshold, C/1000, idempotency guard) (completed 2026-04-22)
- [x] **Phase 38: Statistical Correctness** [GSD+GPD] — L_cat form (proof or fix), P_det extrapolation alignment, per-event quadrature-weight diagnostic (completed 2026-04-22)
- [ ] **Phase 39: HPC & Visualization Safe Wins** [GSD+GPD] — CPU-importable parameter_estimation, raise flush interval, drop per-iteration FFT clear, dead-code removal, flip_hx verify, LaTeX figures, HDI bands
- [ ] **Phase 40: Verification Gate** [GSD+GPD] — Full regression + posterior re-evaluation + h-sweep + anisotropy audit + P_det diagnostic; abort gate on >5% MAP shift
- [ ] **Phase 41: Stage 1 Injection Campaign (Conditional)** [GSD] — Densified M×z×d_L grid on gpu_h100 if >5% extrapolation weight
- [ ] **Phase 42: Stage 2 Sky-Dependent Injection (Conditional)** [GSD] — Sky-grid P_det campaign if anisotropy or Stage 1 residual

## Phase Details

### Phase 35: Coordinate Bug Characterization
**Goal**: Encode the two critical coordinate bugs in failing tests before touching the production code, and capture the pre-fix baseline of how many events sit in the danger zone (±5° ecliptic equator).
**Routing**: [GSD] — test infrastructure only, no physics formulas changed
**Depends on**: Nothing (first phase of v2.2)
**Requirements**: COORD-01
**Success Criteria** (what must be TRUE):
  1. `master_thesis_code_test/test_coordinate_roundtrip.py` exists and contains at least three tests: (a) synthetic galaxy at ecliptic Dec=0° is retrieved by BallTree for a query at the same position, (b) synthetic galaxy at Dec=90° (NCP) round-trip, (c) equatorial→ecliptic conversion matches `astropy.coordinates.SkyCoord` ground truth to <0.001°
  2. The three new tests are **RED** against the current code (assert that the current embedding fails the Dec=0° retrieval and that no rotation is applied in ingestion)
  3. A `.planning/audit_coordinate_bug.md` artifact reports the baseline count and percentage of events in the production CRB CSV whose recovered `qS` lands within ±5° of the ecliptic equator (`|qS − π/2| < 5° × π/180`)
  4. Baseline artifact is committed so subsequent phases can reference it
**Plans**: 3 plans
  - [x] 35-01-PLAN.md — Fixtures module: synthetic_catalog_builder, equatorial_to_ecliptic_astropy, build_balltree (Wave 1)
  - [x] 35-02-PLAN.md — RED test file: 6 xfail(strict) tests + 3 ground-truth checks covering D-04/D-05/D-06/D-07 scenarios (Wave 2)
  - [x] 35-03-PLAN.md — Baseline audit CLI + 3 committed artifacts (md, json, png) for ±5°/±10°/±15° ecliptic-equator bands (Wave 2)

### Phase 36: Coordinate Frame Fix
**Goal**: Fix both critical coordinate bugs — apply equatorial→ecliptic rotation on catalog ingestion, correct the BallTree embedding to match polar-angle convention, and replace the axis-aligned search radius with an eigenvalue-based ellipse on the 2×2 sky covariance.
**Routing**: [GPD] — all three requirements change a computed value (catalog positions propagate through Fisher matrix and waveform response); `/physics-change` gate required per CLAUDE.md
**Physics-gate REQ-IDs**: COORD-02, COORD-03, COORD-04
**Depends on**: Phase 35 (failing tests must exist before fix)
**Requirements**: COORD-02, COORD-03, COORD-04
**Success Criteria** (what must be TRUE):
  1. Phase 35 tests go **GREEN**: BallTree returns the known host when a synthetic galaxy is injected at ecliptic equator (`θ=π/2`) with recovery rate ≥99% over randomized offsets
  2. `galaxy_catalogue/handler.py:_map_angles_to_spherical_coordinates` uses `astropy.coordinates.SkyCoord.transform_to(BarycentricTrueEcliptic())` at J2000 and the docstring declares "all stored angles henceforth in ecliptic SSB frame"
  3. BallTree Cartesian embedding is `(sin θ cos φ, sin θ sin φ, cos θ)` with `θ ∈ [0, π]` polar, consistent across both `setup_galaxy_catalog_balltree` and `get_possible_hosts_from_ball_tree`
  4. Candidate-host search radius uses the 2×2 sky-covariance eigendecomposition (including `|sin θ|` Jacobian on the φ-component); a regression pickle shows the new candidate-host set is a **superset** of the old one for one reference event
  5. Sign convention and dimensional analysis reports pass the `/physics-change` post-implementation checks; reference comments (Eq./arXiv) added above each changed line
**Plans**: 5 plans
  - [x] 36-01-PLAN.md — COORD-03 rotation + _polar_to_cartesian helper (Wave 1)
  - [x] 36-02-PLAN.md — COORD-02 3D BallTree embedding (Wave 2)
  - [x] 36-03-PLAN.md — COORD-04 eigenvalue search radius + regression pickle (Wave 3)
  - [x] 36-04-PLAN.md — COORD-02b 4D BallTree spherical embedding (Wave 3)
  - [x] 36-05-PLAN.md — Verification + state update (Wave 4)

### Phase 37: Parameter Estimation Correctness
**Goal**: Remove hidden coupling between `h` and the saved Fisher CRBs, replace uniform `derivative_epsilon` with per-parameter values appropriate to each scale, and close the latent hygiene holes (swapped `Omega_m` limits, unified `SNR_THRESHOLD`, `SPEED_OF_LIGHT_KM_S = C/1000`, idempotency guard on the angle mapping).
**Routing**: [GSD+GPD] — GSD phase tracks overall progress; GPD invoked for physics subtasks
**Physics-gate REQ-IDs**: PE-01 (h-threading changes Fisher derivatives at host position), PE-02 (per-parameter epsilon changes Fisher matrix values)
**Non-physics REQ-IDs**: COORD-05, PE-03, PE-04, PE-05
**Depends on**: Phase 36 (frame fix — `h` threading must land on top of the corrected frame, otherwise double-fixing the same event at different layers)
**Requirements**: COORD-05, PE-01, PE-02, PE-03, PE-04, PE-05
**Success Criteria** (what must be TRUE):
  1. `ParameterSpace.set_host_galaxy_parameters(host, h=0.5)` produces exactly half the `luminosity_distance` of the same call with `h=1.0`; regression test in `test_parameter_space_h.py` pins this ratio to within floating-point tolerance
  2. Running `set_host_galaxy_parameters` without an explicit `h` argument raises `TypeError` (no hardcoded default): `dist(host.z)` no longer appears in `parameter_space.py:148`
  3. Fisher determinant on one representative event (seed-pinned in regression test) differs by less than 1% after switching to per-parameter `derivative_epsilon`; behavior is stable across at least three random-perturbation resamples (demonstrates non-degeneracy)
  4. `LamCDMScenario.Omega_m.lower_limit < upper_limit` with both values in the physically plausible range `[0.04, 0.5]`
  5. Greping for literal `15` and `20` adjacent to the token `snr` finds no direct uses: `Model1CrossCheck`, pre-screen coefficient, evaluation filter, injection quality gate all resolve to `SNR_THRESHOLD` (constant) — enforced by a smoke test
  6. `SPEED_OF_LIGHT_KM_S` is defined as `C / 1000` in `constants.py`; `comoving_volume_element` test matches to 14 decimal places with no `300000.0` hardcode
  7. `_map_angles_to_spherical_coordinates` raises `AssertionError` when called a second time on already-ecliptic angles (idempotency guard); existing callers unaffected
**Plans**:
  - [x] 37-01-PLAN.md — Software hygiene (PE-03, PE-04, PE-05, COORD-05)
  - [x] 37-02-PLAN.md — Physics tasks: PE-01 h-threading ([PHYSICS]) + PE-02 per-parameter epsilon (GPD, [PHYSICS])
  - [x] 37-03-PLAN.md — Verification (SC-1..SC-7) + state update

### Phase 38: Statistical Correctness
**Goal**: Reconcile the catalog likelihood with Gray et al. (2020) Eq. 24-25 (prove equivalence or replace), and make the P_det extrapolation symmetric between the single-host integrand and the completion-term denominator so the posterior doesn't silently lose quadrature weight at the grid edges.
**Routing**: [GSD+GPD] — STAT-01 routes to GSD if equivalence proof succeeds, GPD if a fix is required; STAT-03 routes to GPD (extrapolation policy change affects computed posteriors); STAT-02 and STAT-04 are GSD (tests and diagnostics)
**Physics-gate REQ-IDs**: STAT-03 always; STAT-01 if it becomes a fix
**Non-physics REQ-IDs**: STAT-02, STAT-04, STAT-01 (if equivalence proof)
**Depends on**: Phase 37 (stable CRB inputs; per-parameter epsilon landed)
**Requirements**: STAT-01, STAT-02, STAT-03, STAT-04
**Success Criteria** (what must be TRUE):
  1. `bayesian_statistics.py:922` contains either (a) a docstring derivation proving `(Σ N_g) / (Σ D_g) = (1/N) Σ (N_g/D_g)` under the code's implicit uniform `1/N` galaxy prior, with a reference to Gray Eq. 24-25, OR (b) the canonical `(1/N) Σ (N_g/D_g)` form after `/physics-change` protocol
  2. `test_l_cat_equivalence.py` unit test with 3 synthetic galaxies computes both forms; asserts they agree numerically (if proof) or documents the exact numerical divergence and explains why (if fix)
  3. `single_host_likelihood` numerator and `precompute_completion_denominator` both use the same P_det extrapolation policy (both zero-fill OR both NN-fill), grep-verified in CI; default is zero-fill per the plan
  4. Per-event diagnostic log writes `quadrature_weight_outside_grid_numerator` and `quadrature_weight_outside_grid_denominator` columns; any event where either exceeds 5% triggers a WARNING log line with the event ID
  5. End-to-end smoke run of `--evaluate` on a small subset (e.g., `simulations/h_0_73` at N≤5) completes without NaN/Inf in the posterior and the diagnostic CSV is non-empty
**Plans**: 3 plans
  - [x] 38-01-PLAN.md — STAT-01/STAT-02: L_cat fix + unit test (Wave 1)
  - [x] 38-02-PLAN.md — STAT-03/STAT-04: P_det zero-fill + quadrature diagnostic (Wave 2)
  - [x] 38-03-PLAN.md — Verification (SC-1..SC-5) + state update (Wave 3)

### Phase 39: HPC & Visualization Safe Wins
**Goal**: Make `parameter_estimation.py` CPU-importable via the `_get_xp` shim, remove the Lustre-heavy per-iteration I/O, drop the per-iteration FFT cache clear, delete dead code, verify the `flip_hx` flag against the current `fastlisaresponse`, enable LaTeX in production figures, and add bootstrap HDI bands to the static convergence plot.
**Routing**: [GSD+GPD] — HPC-05 (`flip_hx` verification) routes to GPD only if the flag is removed; HPC-01..HPC-04 and VIZ-01..VIZ-02 are software-only
**Physics-gate REQ-IDs**: HPC-05 conditional (only if `flip_hx` is removed after verification finds it obsolete)
**Non-physics REQ-IDs**: HPC-01, HPC-02, HPC-03, HPC-04, VIZ-01, VIZ-02
**Depends on**: Phase 38 (downstream runs must already be correct before optimizing I/O)
**Requirements**: HPC-01, HPC-02, HPC-03, HPC-04, HPC-05, VIZ-01, VIZ-02
**Success Criteria** (what must be TRUE):
  1. `from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation` succeeds on a machine without `cupy` installed; `pytest -m "not gpu"` runs the module's tests without skipping on CPU-only CI
  2. `_crb_flush_interval = 25` in `parameter_estimation.py:99`; SIGTERM handler still flushes on termination; a single-task dry run writes the CSV in batches of 25 (verified by a small N=50 simulation test)
  3. `memory_management.py` no longer clears `_fft_cache` per Fisher iteration; cache clear happens only on explicit `free_gpu_memory()` call or memory-pressure trigger
  4. `_crop_frequency_domain` no longer appears in `parameter_estimation.py`; grep returns empty
  5. `flip_hx=True` status is documented in `waveform_generator.py` with either a kept-for-reason comment citing current `fastlisaresponse` behavior, or removed via `/physics-change` with waveform regression test showing pre/post identical result
  6. `main.py:generate_figures` calls `apply_style(use_latex=True)` when `shutil.which("latex")` is truthy and falls back to mathtext otherwise; smoke test confirms both branches run without crash
  7. Static `plot_h0_convergence` displays a 16/84 percentile shaded HDI band; visual smoke test confirms the band is present and sits inside the CI rails
**Plans**: TBD

### Phase 40: Verification Gate
**Goal**: Before committing any new cluster compute, re-evaluate the existing CRBs under all v2.2 fixes and confirm the posterior at h=0.73 is stable; run the 27-value h-sweep; audit anisotropy by `|qS − π/2|` quartiles; report the P_det quadrature-weight diagnostic summary. This phase is the **abort gate** for the staged campaign.
**Routing**: [GSD+GPD] — GSD phase tracking; GPD invoked for VERIFY-02 because it runs the physics-changed evaluation code (but the runner itself is software)
**Physics-gate REQ-IDs**: VERIFY-02 (execution of re-evaluation invokes physics-changed code paths; GPD runs the pipeline with physics protocols active)
**Non-physics REQ-IDs**: VERIFY-01, VERIFY-03, VERIFY-04, VERIFY-05
**Depends on**: Phases 35–39 (all fixes must be landed)
**Requirements**: VERIFY-01, VERIFY-02, VERIFY-03, VERIFY-04, VERIFY-05
**Abort gate**: if the re-evaluated MAP at h=0.73 shifts more than 5% from the v2.1 baseline MAP=0.73, **pause and investigate before any CAMP-\* phase begins**. The pause is mandatory — do not advance to Phase 41.
**Success Criteria** (what must be TRUE):
  1. `uv run pytest -m "not gpu"` passes with all v2.2-era tests included (baseline ≥508 tests plus the new coordinate-roundtrip, h-threading, L_cat equivalence, and SNR-threshold unification tests)
  2. `uv run python -m master_thesis_code simulations/h_0_73 --evaluate --h_value 0.73` on the existing CRB set produces a posterior whose MAP is within 1% bias of h=0.73 and is within 5% of the v2.1 baseline MAP (abort gate)
  3. All 27 h-value posteriors regenerated; `docs_src/interactive/m_z_improvement.html` is updated; static convergence figure shows MAP at h=0.73 is 0.73 ± 0.01 with ≥95% CI coverage (sanity assertion, not a pre-registered gate)
  4. H₀ MAP binned by `|qS − π/2|` quartiles shows no systematic trend: per-quartile MAP shifts stay within 1σ; any >1σ shift is logged and treated as a Stage 2 trigger rather than an abort condition
  5. `.planning/debug/verify_gate_{timestamp}.md` reports the P_det quadrature-weight-outside-grid summary: mean and max per-event fraction across all events, plus a histogram; this number decides whether Phase 41 fires
**Plans**: TBD

### Phase 41: Stage 1 Injection Campaign (Conditional)
**Goal**: If VERIFY-05 reports that more than 5% of quadrature weight lands outside the P_det injection grid on average, submit a densified M×z×d_L injection campaign to bwUniCluster `gpu_h100`; re-evaluate posteriors against the Phase 40 baseline.
**Routing**: [GSD] — cluster submission, data management, re-evaluation pipeline; no formula changes
**Non-physics REQ-IDs**: CAMP-01
**Depends on**: Phase 40 (verification passes and Phase 40 reports >5% mean extrapolation weight)
**Conditional**: skip if Phase 40 reports mean extrapolation weight ≤5% per event — in that case, mark the phase `skipped (not triggered)` in MILESTONES.md
**Requirements**: CAMP-01
**Success Criteria** (what must be TRUE):
  1. Injection grid upper bound on `d_L` is extended and all three axes (M, z, d_L) are densified by a factor of 1.5×; the new grid spec is committed and documented
  2. SLURM array job on `gpu_h100` completes with >95% task success rate using the existing `cluster/injection` infrastructure (per Phase 33 memory); failed tasks retried successfully
  3. Re-evaluated posterior using the new P_det table is indistinguishable from Phase 40 result under a Kolmogorov–Smirnov test (p-value > 0.05) at h=0.73, OR shows a quantified shift that is documented
  4. Post-Stage-1 VERIFY-05 diagnostic confirms the mean quadrature-weight-outside-grid has dropped below the 5% threshold
**Plans**: TBD

### Phase 42: Stage 2 Sky-Dependent Injection (Conditional)
**Goal**: If VERIFY-04 or Phase 41 reveals residual sky anisotropy, submit a second injection campaign with `qS` and `phiS` as additional grid axes (6×12 sky grid); otherwise document that isotropic-sky P_det marginalization is verified sufficient.
**Routing**: [GSD] — cluster submission and re-evaluation; no formula changes
**Non-physics REQ-IDs**: CAMP-02
**Depends on**: Phase 41 (outcome of Stage 1) or Phase 40 (VERIFY-04 anisotropy result)
**Conditional**: skip if Phase 40 VERIFY-04 shows no >1σ MAP trend across `|qS − π/2|` quartiles AND Phase 41 shows no residual sky anisotropy — in that case, write a short documentation note in `.planning/debug/` stating that isotropic P_det marginalization is verified sufficient for the paper
**Requirements**: CAMP-02
**Success Criteria** (what must be TRUE):
  1. Pre-registered anisotropy test specification (e.g., event groups binned by `sin²(2·qS)` with `>1σ` per-bin MAP shift threshold) is written and committed **before** the campaign launches; this pins the decision criterion
  2. If the pre-registered test triggers: a 6×12 `(qS, phiS)` sky grid P_det campaign completes on `gpu_h100`; new P_det tables replace the isotropic tables and re-evaluated posteriors show the sky-trend is eliminated (per-bin MAP trend within 1σ)
  3. If the pre-registered test does not trigger: `.planning/debug/isotropic_p_det_verification.md` documents the quartile MAP values, the 1σ thresholds, and a final paragraph explaining that marginalized isotropic P_det is sufficient for the paper
  4. Phase 42 closes v2.2 by updating `.planning/MILESTONES.md` with the shipped status, linking to the campaign artifacts, and flipping the active milestone in STATE.md
**Plans**: TBD

## Progress

| Phase | Routing | Plans Complete | Status | Completed |
|-------|---------|----------------|--------|-----------|
| 35. Coordinate Bug Characterization | GSD | 3/3 | Complete    | 2026-04-21 |
| 36. Coordinate Frame Fix | GPD | 5/5 | Complete    | 2026-04-22 |
| 37. Parameter Estimation Correctness | GSD+GPD | 3/3 | Complete    | 2026-04-22 |
| 38. Statistical Correctness | GSD+GPD | 3/3 | Complete    | 2026-04-22 |
| 39. HPC & Visualization Safe Wins | GSD+GPD | 0/? | Not started | - |
| 40. Verification Gate | GSD+GPD | 0/? | Not started | - |
| 41. Stage 1 Injection Campaign | GSD | 0/? | Not started (conditional on Phase 40) | - |
| 42. Stage 2 Sky-Dependent Injection | GSD | 0/? | Not started (conditional on Phases 40, 41) | - |

## Coverage

| Phase | REQ-IDs | Count |
|-------|---------|-------|
| 35 | COORD-01 | 1 |
| 36 | COORD-02, COORD-03, COORD-04 | 3 |
| 37 | COORD-05, PE-01, PE-02, PE-03, PE-04, PE-05 | 6 |
| 38 | STAT-01, STAT-02, STAT-03, STAT-04 | 4 |
| 39 | HPC-01, HPC-02, HPC-03, HPC-04, HPC-05, VIZ-01, VIZ-02 | 7 |
| 40 | VERIFY-01, VERIFY-02, VERIFY-03, VERIFY-04, VERIFY-05 | 5 |
| 41 | CAMP-01 | 1 |
| 42 | CAMP-02 | 1 |

**Total mapped:** 28/28 ✓ — 100% coverage, no orphans.

## Hard Constraints (from plan)

- **COORD-01 (Phase 35) MUST precede all other COORD-\* requirements** — failing-test baseline captures pre-fix state before Phase 36 fixes.
- **VERIFY-02 (Phase 40) MUST complete BEFORE any CAMP-\* phase begins** — no new cluster compute without the verification gate.
- **CAMP-02 (Phase 42) is conditional on VERIFY-04 or CAMP-01 outcome** — not guaranteed to execute.
- **Abort condition at Phase 40:** if re-evaluated MAP at h=0.73 shifts >5% from v2.1 baseline MAP=0.73, pause and investigate before CAMP-\* phases. Do not advance until diagnosed.

## Links

- **Plan artifact:** `~/.claude/plans/i-want-a-last-elegant-feather.md`
- **Requirements:** `.planning/REQUIREMENTS.md`
- **State:** `.planning/STATE.md`
- **Audit memory:** `.claude/projects/-home-jasper-Repositories-MasterThesisCode/memory/project_coordinate_bugs.md`, `project_audit_2026_04_21.md`
- **Prior cumulative roadmap:** `.planning/milestones/v2.1-cumulative-ROADMAP.md`
- **Prior shipped milestones:** `.planning/milestones/v{1.0,1.1,1.2,1.3,1.4,2.1-biasres}-ROADMAP.md`

---
*Created: 2026-04-21 — v2.2 Pipeline Correctness roadmap from pre-batch audit plan*
