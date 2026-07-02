# Independent Verification Report — LISA EMRI Dark-Siren H₀ Pipeline (2026-07-01)

*Status: COMPLETE. §7 finalized — synthetic P–P/coverage (D2) + a real-data de-rail of the production code
(0.86 ↑rail → 0.60 ↓rail → peaked 0.73 via either principled fix). All sections and the verdict are final and
independently corroborated.*

**Commission:** independent, skeptical re-investigation of whether the pipeline is correct (physically,
statistically, software) and whether a **peaked, non-railing H₀ posterior is achievable** — trying hard to
*falsify* the project's stored "in-catalogue photo-z dark sirens irreducibly rail to the H₀ grid edge" claim.

**Method (independence enforced):** all findings below were produced by investigator agents given ONLY a neutral
problem statement + code/data access, blind to the project's `.planning/`/`docs/`/memory conclusions and to each
other; the injected H₀ was withheld from the from-scratch reproducers (each chose its own). Multi-agent workflows:
WF1 (5 audit lenses + 3 archaeology + adversarial verify), WF2 (3 blind from-scratch estimators + 10-way
hypothesis tournament + adversarial verify), WF3a (5 evidence-locker faithfulness audits), plus an orchestrator
real-data crux run (production inference on real seed600, default vs `--catalog_only`). Every prior project result
was treated as an untested claim and audit-gated. Total ≈ 55 agents, ≈ 4.9M agent tokens.

---
## Headline verdict
The stored "**irreducible** photo-z railing" claim is **WRONG / INCOMPLETE**. The railing is a **curable
artifact of specific normalization + numerator defects**, not irreducible information starvation. Supported
hypotheses: **H-new-bug + H-artifact** (not H-physics, not H-past-bug-good).

The decisive, adversarially-verified fact: **the production estimator's MAP is essentially independent of the
injected H₀** in the railing regime (tournament injection scan: production MAP = 0.86 for *every* injected truth
0.63→0.77, while the local `catalog_only` normalization tracks the truth exactly). An estimator whose mode does
not move with the truth is not measuring H₀ — it is reporting where a mis-normalized selection term points.

**Nuance preserved:** photometric redshifts *do* genuinely degrade H₀ **precision** (posteriors widen sharply,
central value becomes prior-sensitive). But "wide/weak" ≠ "railed to the grid edge": a correctly-specified
estimator peaks in the interior and recovers the injected H₀ to ~1–2% even at σ_z≈0.035.

---
## Why the past results were sensible, and what changed (temporal archaeology)
- A **2026-04-09 baseline (n=417) peaked at map_h=0.735, unbiased** — a raw per-event distance–redshift product
  with **no selection-function machinery**. This is why past results were sensible.
- Railing to 0.86 first appeared **~2026-04-24**, when the Gray selection/completeness machinery (h-dependent
  zero-fill p_det + D(h)) was switched on. The zero-fill bug was later fixed (interior but +high bias); the June
  restructure to the single global-denominator Gray ratio re-expressed the same normalization sensitivity.
- **Bookkeeping ruled out (high confidence, runtime-proven):** injected = target = 0.73; grid [0.60,0.86]
  dead-center; Ω_m matched. `TRUE_HUBBLE_CONSTANT=0.70` has 3 usages, all in `datamodels/galaxy.py`, which has
  **zero production importers** — a fossil confined to the synthetic Pipeline A.

---
## 1. Physically correct?
Mostly yes, with defensible-but-consequential modeling omissions:
- **Distance–redshift** `dist(z,h)∝1/h` is exact (matches numerical integration to 1e-14); d_L→z inversion
  forward-marginalized (no point-inversion σ² rail; sub-%). **REFUTED as a rail cause.**
- **Selection/population inconsistency [CONFIRMED]:** the detection-probability p_det is built from a *different*
  EMRI population (`cosmological_model.Model1CrossCheck`, dN/dz·R_emri) than the analyzed events + inference prior
  (`emri_rate.py`); mass marginals differ by +0.158 dex, exaggerating the selection log-slope dlnD/dh
  (−1.07→−1.54). Biases the H₀-dependence of the normalization.
- **Host-redshift marginalization [CONFIRMED bug]:** the in-catalogue numerator marginalizes the host-z Gaussian
  through the convex d_L(z) with **no dd_L/dz Jacobian and no comoving-volume weight**, while the selection
  denominator *does* carry the volume element. This numerator/denominator **prior inconsistency** produces an
  upward Jensen bias (+16% at z=0.02, ~0 at z>0.3; adding the Jacobian recovers truth at all z) and, combined with
  the selection over-correction, drives railing. Independently reproduced by all three from-scratch estimators.
- **Completion (dark-event) term [CONFIRMED bug]:** `B_num` evaluates the GW sky-Gaussian at its *peak* with **no
  1/(4π) solid-angle normalization**, inflating it ~5000× vs the catalogue term → it dominates → monotone in h.
- Cosmology is WMAP-era (Ω_m=0.25) and fixed (not marginalized); wCDM w0/wa silently ignored; heliocentric z used
  (CMB-frame/peculiar-velocity `z_cmb` neglected). Low-order for the rail but real.

## 2. Statistically correct?
Structure is a proper flat-prior selection-corrected hierarchical posterior (log P(h)=Σ log p_i, p_det in the
denominators, density-correct trapezoid CIs). But:
- **Normalization choice controls & flips the mode [CONFIRMED, critical]:** production global-denominator
  single-ratio vs local `catalog_only` ratio-of-sums give MAPs on **opposite edges** on identical data (0.86 vs
  0.60/0.71). A genuine data constraint cannot flip sign under a normalization choice → the "posterior" is
  normalization-dominated in the photo-z regime.
- **The production estimator is mis-calibrated:** the tournament injection scan shows the production MAP = 0.86
  for *every* injected truth (0.63→0.77) — an estimator whose mode is independent of the truth has ~0% coverage.
  No formal calibration had ever been run on the production estimator (the closure unit tests below use σ_z=0 and
  clamp railing realizations, so they do not test it). A dedicated D2 P–P/coverage test (production-style vs
  consistent-normalization estimator, ≥100 photo-z realizations) is included below [§7].
- The 2-D (with-BH-mass) denominator uses **unseeded Monte-Carlo importance sampling** (non-deterministic, noisy);
  `--seed` never reaches the inference.

## 3. Software correct?
- **No truth leakage found** into the inference (cross-match/frame audit: both sides ecliptic; BallTree query =
  true host; no injected-H₀ leak). Frames consistent. **Cross-match/frame REFUTED as a rail cause.**
- Production data are **noise-realized** (prepared CRB carries correlated MVN scatter, seed 1000399/1000599;
  median rel d_L unc 0.035) — the "Asimov" concern is refuted for the distance observable (M_z is ~exact).
- **Reproducibility gap:** `--seed` is not threaded into the inference; the 2-D denominator is non-deterministic.
- **Performance/code-health:** the per-event likelihood emits an **unthrottled root-logger WARNING ~2600×/event**
  (~9M lines / 877 MB per h-value), dominating inference runtime. Non-uniform-grid `np.sum` normalization in some
  combination paths; `parabolic_refine_map` uses the equal-spacing vertex formula on a non-uniform grid; dead
  angular-distance helper with swapped sin/cos; dead full-CRB read can crash construction.
- **Stale on-disk catalogue (found this session):** the local `reduced_galaxy_catalogue.csv` was the pre-June
  6-column schema (missing `APPARENT_B_MAG` + redshift flag) → the reader mis-aligned columns (redshift read as
  B-magnitude). The cluster copy was likewise stale (7-col). Rebuilt to the current 8-col schema and verified.
  (Any real-data eval run before the rebuild is invalid.)

## 4. Can a peaked, non-railing posterior be achieved? [CONFIRMED — real-data de-rail, §7.2]
**Yes.**
- **Spec-z recovers cleanly:** all three blind from-scratch estimators recover their own injected H₀ to <1% with
  spec-z hosts; the production-faithful reproduction recovers 0.700 at σ_z=0.0017.
- **Photo-z with a *consistent* estimator recovers truth:** restoring the comoving-volume/Jacobian weight to the
  numerator host-z term (so numerator and denominator share the same z-prior) gives −1.4% at σ_z=0.035
  (Repro C); volume-weighting brackets the truth (Repro A: bare −6.6% ↔ volume-weighted +6.3%).
- **THE RAIL IS NORMALIZATION-DRIVEN, NOT DATA-DRIVEN — confirmed on REAL DATA (crux):** the *actual production
  inference* was run on real seed600 events (500-event subsample of the 3375; the full stored 3375-event result
  also rails to 0.86) over the h∈[0.60,0.86] grid, on the freshly-rebuilt 8-column catalogue, in two modes:
  - **Production (default):** MAP = **0.86**, mean 0.860, **edge mass 1.000** — a delta railed to the upper edge.
  - **`--catalog_only`** (drop the completion term / local self-normalized ratio): MAP = **0.73**, mean 0.737,
    **edge mass 0.000** — a **peaked interior posterior recovering the injected H₀=0.73** (mass 0.77 at 0.73,
    0.22 at 0.76; width ~0.03).
  Same real data, same events, same catalogue — **switching only the normalization moves the mode from the 0.86
  grid edge to a clean peak at the injected 0.73.** This is the decisive falsification: a defensible normalization
  produces a peaked, near-truth, reasonably tight posterior on the real photo-z-dominated data. Synthetic +
  production-faithful reproductions independently show the same edge-flip, and the tournament injection scan showed
  the production mode is *independent of the injected truth* (=0.86 for every truth 0.63→0.77).
  *(Caveat: `catalog_only` drops the incompleteness correction, so it is a demonstration that the rail lives in the
  completion/normalization term, not a finished physically-complete estimator; the principled fix keeps a
  completeness correction but with a consistent numerator/denominator prior and a 1/(4π)-normalized completion —
  which the from-scratch reproductions show also recovers truth.)*
- **Mechanism (independently derived):** two σ_z-driven mechanisms — a numerator Jensen bias (UP) from the missing
  dd_L/dz Jacobian/volume weight, and a selection over-correction (DOWN) from the global-denominator +
  missing-1/(4π) completion term — whose imbalance pins the combined posterior to whichever grid edge dominates
  (explains both the production up-rail to 0.86 and the down-rail to 0.60 seen in faithful closures, i.e. the
  sign-flip). Curable by (a) making the numerator/denominator z-priors consistent, (b) 1/(4π)-normalizing the
  completion term, (c) matching the p_det population, and/or (d) the local self-normalized catalogue ratio.

## 5. Verdict on the project's conclusions
- "In-catalogue photo-z dark sirens **irreducibly** rail to the grid edge" → **WRONG / INCOMPLETE.**
- Supported: **H-new-bug + H-artifact.** The railing was *introduced* by the selection/completeness machinery
  (bugs: missing numerator Jacobian/volume weight; missing 1/(4π) in the completion term; normalization-choice
  sign-flip; p_det population mismatch) and is an artifact curable by defensible fixes.
- **Not** H-past-bug-good (past peaks were the correct raw likelihood, not a truth-leak) and **not** pure
  H-physics (photo-z limits precision but does not force an edge rail).
- **Evidence-locker audit:** the project's own "irreducible railing" harnesses are **none faithful/trustworthy** —
  rung_I is an artifact (sign-flips, omits the population prior, rails the wrong direction); the closure unit
  tests use σ_z=0 and clamp railing realizations (do not test photo-z); F5's <1% floor is a grid-node artifact;
  rung_G is faithful term-by-term but hinges on the exact normalization choice in question.

## 6. Ranked bug list (independent of the project's list)
1. **[CRITICAL] Numerator/denominator prior inconsistency** — host-z marginalization lacks dd_L/dz Jacobian +
   comoving-volume weight that the selection denominator carries. `bayesian_statistics.py:1789-1807`. Fix: weight
   the host-z integrand by dVc/dz·(1+z)⁻¹ (or integrate in d_L with the Jacobian) consistently with D(h).
2. **[CRITICAL] Normalization-choice sign-flip / normalization domination** — global-denominator single ratio
   vs local ratio-of-sums put the MAP on opposite edges; production mode independent of injected H₀.
   `bayesian_statistics.py:1540-1673 vs 1517-1539`. Fix: adopt the self-consistent local/partition normalization
   and verify the mode tracks injected H₀.
3. **[HIGH] Completion term missing 1/(4π) sky normalization** — `B_num` at sky-peak density, ~5000× inflated,
   dominates + monotone in h. `bayesian_statistics.py:1607-1645`. Fix: solid-angle-marginalize the completion sky
   factor.
4. **[HIGH] p_det built from a different EMRI population** than events+prior (+0.158 dex mass offset). Fix:
   build p_det from `emri_rate` (or reweight the injection pool).
5. **[MEDIUM] Analysis cuts (rel d_L error <0.10, Fisher conditioning) absent from the modeled selection D(h)**
   (Malmquist). 6. **[MEDIUM] Unseeded 2-D MC denominator + `--seed` not threaded to inference** (non-reproducible).
7. **[MEDIUM] Heliocentric z used; CMB-frame/peculiar-velocity `z_cmb` (GLADE col 29) neglected** (physics change;
   requires re-sim; a `fix/cmb-frame-redshift` branch already exists). 8. **[MEDIUM/perf] Unthrottled hot-loop
   root-logger WARNING** (~9M lines/h). 9. **[LOW] Non-uniform-grid np.sum normalization / parabolic MAP vertex
   formula / dead sin-cos helper / WMAP Ω_m / wCDM ignored / galaxy z-error 0.013(1+z)³ unreferenced.**

---
## 7. Statistical calibration (D2 P–P / coverage) + real-data de-rail — [COMPLETE]
Two decisive, independent additions close the loop: a from-scratch synthetic P–P/coverage test of the estimator
(no repo import) and a real-data de-rail of the *production* `evaluate()` across normalization choices. **They
converge: the production in-catalogue normalization is the defect, and two principled fixes restore a peaked,
near-truth, calibrated posterior.**

### 7.1 Synthetic P–P / coverage (`scratch/d2/`, 120 realizations, σ_z≈0.035)
The production in-catalogue numerator marginalizes the host photo-z against a **bare Gaussian N(z; z_g, σ_z)**
(no dV_c/dz), while D(h), β_Gbar and B_num all carry the comoving-volume prior dV_c/(1+z) — an internal
prior inconsistency. Isolating *only* that numerator z-prior (single host, complete catalogue, no completion;
120 realizations × 250 events; injected truths 0.66/0.72/0.78):

| host-z numerator | cov 50% | cov 68% | cov 90% | MAP bias |
|---|---|---|---|---|
| bare Gaussian (production) | ~0.00 | ~0.02 | ~0.03 | **−0.024 (≈3.3% low)** |
| volume-weighted dV_c/(1+z) (**fix #1**) | ~0.55 | ~0.70 | ~0.88 | ~−0.002 |

Production coverage **collapses to ≈0–3%** at every truth (a rigid −0.024 low bias dwarfs the ~0.008 event
spread); the volume-weighted numerator is **calibrated** (coverage ≈ nominal). The bias grows as **σ_z²**
(−0.0016 / −0.0064 / −0.023 / −0.046 at σ_z = 0.005 / 0.015 / 0.035 / 0.050; → 0 as σ_z→0) — the textbook
signature of an omitted redshift prior (an Eddington/Malmquist-in-z effect), maximal exactly at the catalogue's
σ_z≈0.035. **Honest nuance / correction:** the production MAP still **tracks the truth** (slope ≈ 1) — it is a
*biased-but-responsive* estimator here, NOT "MAP independent of truth"; the latter was seen for the
*global-denominator rail on real data* (a distinct normalization-domination regime, §2/§4). With the full
completion+interloper machinery the production coverage is 0.00/0.00/0.02 (MAP 0.682) and the volume-prior fix
restores 0.40/0.54/0.82 (MAP 0.707); a residual −0.013 / compressed slope come from the completion term being
only weakly H0-informative (secondary). The literal *global* selection denominator is delicate to normalize in
a from-scratch synthetic (it railed there) — flagged for a direct check against the real GLADE β_G sum; the
z-prior verdict does not depend on it.

### 7.2 Real-data de-rail of the production code (`redteam/crux_realdata.md`, `redteam/derail_matrix_results.json`)
Production `evaluate()` on the real seed600 494-event subsample, 7-h grid, injected H0 = 0.73, each fix applied
and re-run on the identical data (all with the 1/(4π) completion fix `cb16142`):

| step | in-catalogue normalization | MAP | mean | posterior | railed |
|---|---|---|---|---|---|
| pre-4π (global-denom, peak-density B_num) | bare Gaussian | 0.86 | 0.860 | 100% @ 0.86 | ↑ rail |
| 4π only (`prod_global`) | bare Gaussian, global-denom ratio | 0.60 | 0.600 | 100% @ 0.60 | ↓ rail |
| **fix #2** (`local_ratio`) | Gray A.9/A.10 local ratio-of-sums | **0.73** | 0.730 | 98% @ 0.73 | peaked |
| **fix #1** (`volume_deconv`) | local ratio + dV_c/(1+z) host-z prior | **0.73** | 0.740 | 68% @ 0.73, 31% @ 0.76 | peaked |
| `catalog_only` baseline | local ratio, no completion | 0.73 | 0.737 | 77% @ 0.73, 22% @ 0.76 | peaked |

- The **1/(4π) completion fix alone flips the rail 0.86 → 0.60** (upper → lower edge): it removes the
  ~1640× completion inflation (confirming that defect) but exposes the still-uncorrected in-catalogue
  normalization — **necessary but not sufficient**, the exact sign-flip mechanism §4 predicts.
- **Either principled fix de-rails to a peaked 0.73**, recovering the injected H0 on the real photo-z-dominated
  data. `volume_deconv`'s mean sits **+0.010 above** `local_ratio` (0.740 vs 0.730) — the same volume-prior
  debiasing the D2 coverage test measured, agreeing in sign and rough magnitude across the two independent lines.

**Verdict (finalized).** The railing is a **curable normalization artifact**, not irreducible information
starvation. The production estimator is (i) **mis-normalized** — the global-denominator single ratio pins the
mode to a grid edge on real data — and (ii) **statistically mis-calibrated** — the bare-Gaussian host-z
numerator gives ≈0% coverage and an σ_z² Eddington bias. Both are cured by defensible changes: the Gray A.9/A.10
local self-normalized ratio-of-sums (#2) and the comoving-volume host-z deconvolution (#1), implemented behind
`normalization_mode` in `bayesian_statistics.py` (with unit tests) on top of the 1/(4π) completion fix
(`cb16142`). Reproduce: `scratch/d2/` (synthetic P–P), `redteam/crux_realdata.md` + `derail_matrix_results.json`
(real-data de-rail).

---
## Methods, independence & confidence
- **Independence mechanism:** every investigator agent received only the neutral §B2 problem statement + code/data
  access and a hard denylist (no `.planning/`/`docs/`/memory, CHANGELOG/TODO, or files named
  *BIAS*/*RESOLUTION*/*HANDOFF*/*VERIFICATION*/*DIGEST*; ignore CLAUDE.md Known-Bugs). Commit messages/comments
  were treated as unverified claims. The injected H₀ was withheld from the from-scratch reproducers (each chose its
  own value — 0.68/0.70/0.70 — and still recovered it). The orchestrator withheld all stored conclusions until this
  synthesis.
- **Consensus:** the core mechanism (normalization/prior-inconsistency, not irreducible starvation) is supported by
  ≥2 independent from-scratch reproductions, a 10-way hypothesis tournament with adversarial verification, a
  faithfulness audit of the project's own harnesses, AND a real-data run of the production code — four independent
  lines converging.
- **Real-data crux fidelity:** ran the actual `BayesianStatistics().evaluate(...)` production path (not a
  reimplementation) on real seed600 CRBs + real injections + the rebuilt real catalogue; only the event count was
  reduced (500 of 3375) for tractability, and the full 3375-event stored result rails identically to 0.86.
- **Key caveats to preserve honesty:** (i) `catalog_only` demonstrates the rail's location (the completion/
  normalization term) but drops incompleteness correction — it is a diagnosis, not a finished estimator; the
  principled fix keeps completeness with a consistent prior + 1/(4π) completion. (ii) Photo-z genuinely limits
  precision; the claim refuted is specifically the *irreducible edge rail*, not that photo-z is as informative as
  spec-z. (iii) The rebuilt catalogue uses heliocentric z (per your decision to defer z_cmb); recommended as a
  separate physics fix.

---
## Decision log (hypotheses tested → discriminating result)
| Hypothesis | Result |
|---|---|
| Photo-z information starvation → irreducible rail | REFUTED (rail is curable; spec-z + consistent-prior photo-z recover) |
| Normalization choice controls/flips the rail | **CONFIRMED** (mode flips edges; production mode ⊥ injected H₀) |
| Numerator missing Jacobian/volume weight (Jensen) | **CONFIRMED** (adding it recovers truth at all z) |
| Completion term missing 1/(4π) | **CONFIRMED** (~5000× inflation; dominates) |
| p_det population mismatch | CONFIRMED (secondary amplifier) |
| Grid/prior artifact (D4) | REFUTED (flat prior; MAP argmax-invariant; interior for well-specified) |
| d_L→z inversion σ² bias (D8) | REFUTED (sub-%) |
| Truth/fiducial bookkeeping 0.70↔0.73 (D6) | REFUTED (runtime-proven fossil) |
| Cross-match / frame error | REFUTED (no active bug) |
| Asimov / no measurement scatter | REFUTED (prepared CRB is noise-realized) |
