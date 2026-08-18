# PRE-REGISTRATION — Production-configuration calibration ladder ([R-1]/[R-2] of row #120)

**Date:** 2026-08-17 · **Status:** DRAFT v3 — verifier amendments 1–8 AND delta amendments
D-1..D-7 applied verbatim (`VERIFIER_ADDENDUM_PRODCAL.md`, final gate GO-WITH-AMENDMENTS, "no
further verifier pass required if applied verbatim"); awaiting author [DO]. **Append-only after
commit: no edits above the VERDICT line once this file is committed. Any later edit to §7 voids
the registration (new prereg required), except the single registered pretuning fill-in (§7),
itself append-only.**

**Provenance gating (what upstream gate makes this necessary):** ledger row #120 (front opened,
D-5 granted); the Q-0 audit verdict **UNPAIRED** (2026-08-17): production carries
const-σ-at-truth with no measurement scatter — the selection-inside factor landed by the fusion
(`2b10b8b8`, rows #117–#118) therefore sits in the *configuration class* that harness rows
#66/#67 measured as breaking calibrated controls, with the mitigating facts that (i) production's
d_L_obs ≡ d_L_true kills const-σ sub-term (a), leaving only sub-term (b) (the dropped 1/σ(z)
variation), whose isolated weight is UNMEASURED; and (ii) the venue differs on every axis of the
standing scoping rule. Q-0's two decisive code facts were **independently re-verified by the
orchestrator** (not the harness builder), per verifier amendment 7 — quotes in §7. Structural
same-object checks Q-1/Q-2 PASSED (G23-c SAME-OBJECT, spot-verified; P3 symmetric, fusion removed
the pre-existing asymmetry) — the calibration question is therefore concentrated entirely on the
noise-model pairing. Intake:
`results/campaign51_20260728/realistic_20260729/CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md`.

**Instrument:** the [A3]-extended `darksiren_emri/validation/pp_coverage.py`, **frozen in the
same commit as this file** (the freezing commit IS the harness hash of record). Structural nulls
pinned by `darksiren_emri_test/validation/test_pp_coverage_mass.py` (28 tests): α_M=0 ⇒ bit-exact
mass-blind reduction; σ_cond→0 ⇒ g_sel → g·S(μ_cond·M_z,det) (the production operating point,
row #118/MAJOR-1); channel-locality (off/2d share the 1D block bit-for-bit and off/1d the 2D
block); byte-identity of all pre-existing modes vs `07bbecc9`; paired RNG stream across the
scatter/no-scatter switch; three-way noise-cell distinctness incl. CLI mapping.

---

## 1. Hypotheses (registered BEFORE any cell is run)

- **H-P (primary).** The **production-analog configuration** — `noise-model=production`
  (no scatter, const-σ at truth) + `selection_cell=fused` — is **calibrated** at a mass-bearing,
  multi-candidate, deep venue: MAP bias and coverage within the §4 PASS bands. This is [C-CAL]
  of the intake restated at harness scale.
- **H-B (the sub-term-(b) weight).** The #66/#67 const-σ floor is **dominated by sub-term (a)**
  (σ at scattered d_L_obs), so the production-analog (b)-only cell shows a floor at least 2×
  smaller than the `noise-model=const` twin. **Registered precondition (verifier amendment 3):**
  the V-deep const+fused floor must be ≥ +0.0015 AND sign-coherent across all three truths (each
  truth's floor > +2·SE). If the precondition fails, the H-B read is
  **UNDETERMINED-BY-DESIGN** (the venue does not reproduce a decomposable floor) and is reported
  as such — not scored PASS/FAIL/MIXED.
- **H-N (N-coherence).** IF a floor exists in the production-analog cell, it is a real asymptotic
  bias: flat in n_events while cov68 collapses (the #67 signature,
  `pp_coverage_noisemodel_20260711/SUMMARY.md:74,80-82`).
- **Expected NULLs (registered):**
  - **N-1 (instrument continuity):** dedicated **replication cells** (§3) — mass channel OFF,
    continuum mode, the 2026-07-11 venue parameters and master seed 20260701 — reproduce the
    07-11 const-σ floor and model+p_det-corrected values within 2·SE (same-seed ⇒ plain SE).
    Outside 4·SE = instrument suspect: STOP, audit harness before interpreting any other cell.
  - **N-2 (consistent-estimator check):** the `model` + `fused` cell at V-deep shows
    |bias| ≤ 0.0010 (the #67 "≤+0.0008 + ~+0.0005 residual" class). |bias| > 0.0020 =
    instrument suspect: STOP as N-1.
  - **N-3 (control venue):** production+fused at V-ctrl stays calibrated **relative to its own
    `off` twin** (paired read, §4). The #66 failure mode was controls *flipping*; its recurrence
    is a first-class read, not noise.
  - **N-4 (2D channel):** 2D-channel coverage in `fused` does not degrade relative to `off`
    beyond 2·SE (the counterfactual's near-inert 2D channel, M-1/M-3 of row #119).
  - **N-5 (engagement — verifier amendment 5; guards the false-PASS direction):** in every
    venue × n, the paired per-realization MAP-delta distributions (fused−off) and
    (const−production) are **non-degenerate** (not identically zero); at V-ctrl,
    const+fused − const+off reproduces a **positive** shift (the #66 signature direction).
    A degenerate delta = silently-inert lever = instrument suspect: STOP as N-1.

## 2. Stage-1 information forecast / power (verifier-corrected, amendment 1)

Anchored on the #67 record's measured `map_std` (not a 1/√n idealization — the record shows
sub-1/√n improvement):

- Per-realization MAP scatter (deep cells, `pp_coverage_noisemodel_20260711/*.json`):
  0.0051–0.0074 at n=250; 0.0027–0.0042 at n=1000; 0.0017–0.0021 at n=4000 (n=4000 sits ~25%
  above the 1/√n extrapolation from n=1000). Interpolated σ(n=1600) ≈ 0.0025.
- SE(bias) at R=120: n=250 ≈ 5.5e-4; n=1600 ≈ **2.3e-4**. Therefore: the +0.002-class floor is a
  **≥8σ** read at n=1600; the ~+0.0005 #67-residual class is a **~2σ read at R=120 — registered
  as a BOUND, not a detection**. The primary [R-2] discriminator for small floors is the
  **n=1600 cov68 read** (SE(cov68) = √(0.68·0.32/120) = 0.043; a #67-type collapse to ≤0.38 is a
  7σ-class effect).
- H-B decomposition: paired read on the shared stream; paired-delta SE ≤ √2·2.3e-4 ≈ 3.3e-4 at
  n=1600 (an upper bound — positive correlation on the shared stream reduces it; the realized
  paired SE is what the scorer reports). Under the H-B precondition, at the precondition edge
  (floor exactly +0.0015 ⇒ delta threshold 0.00075) the worst-case read is **≥2σ**; it is ≥3σ
  for floors ≥ +0.002 (D-6). **H-B's scored read lives on the Block B n=1600 const/production ×
  off/fused cells; the Block A n=250 H-B read (~1σ at the edge) is descriptive only.**
- Coverage read power: 2σ detectability ±0.085 around 0.68 at R=120; the #67 floor produced
  cov68 = 0.38 (n=1000) and 0.12 (n=4000) for a +0.0022-class bias — ≥7σ.
- Fisher-forecast leg: NOT a repo asset (known-gaps register) — this sizing deliberately uses
  measured #67 scatter. No pretend-Fisher.

## 3. Design — cells, venues, seeds, scorer

**Venues (2):**
- **V-deep** (production-analog): catalogue_mode, mass channel ON (α_M = 0.25), n_galaxies =
  2e5; z_support/sky_frac fixed in §7 by the registered pretuning procedure against the
  production anchors (catalogue-bearing event fraction 0.60–0.70, completion share ≈ 0.3–0.4 —
  anchor provenance in §7).
- **V-ctrl** (calibrated control): the shallow/untruncated configuration class of the 07-11
  controls, mass channel ON with the same α_M.

**Block A — configuration grid** at n_events=250, R=120, truths h_true ∈ {0.62, 0.72, 0.84}:
`noise-model ∈ {production, const, model}` × `selection_cell ∈ {off, fused}` × 2 venues
= 36 cells. Decides H-P (first pass), N-2, N-3, N-5; the Block A H-B read is descriptive only
(D-6 — the scored H-B read is Block B's n=1600 cells).

**Block N1 — replication cells (verifier amendment 4a):** mass OFF, continuum mode, 07-11 venue
parameters (z_support ∈ {0.3, 0.5}, σ_z = 0.035, d50/w_pdet per the 07-11 RUNBOOKs), noise
cells {const, model+p_det-inside} per `pp_coverage_noisemodel_20260711/RUNBOOK.md`, R=120,
**master seed 20260701** (the 07-11 seed, for direct comparability). ≈ 0.5 CPU-h. Decides N-1.

**Block B — N-ladder** at V-deep, `noise-model ∈ {production, const}` × `selection_cell ∈
{off, fused}` × n_events ∈ {800, 1600}, R=120, same 3 truths = 24 cells. Decides H-N and the
[R-2] bias/coverage reads at production N.

**Secondary reads (registered):** (S-1) channel decomposition cells `1d`/`2d` at V-deep,
production noise, n=250, truth 0.72. (S-2) [A2] paired per-realization read: every cross-cell
comparison is reported as the paired per-realization delta distribution ALONGSIDE the class
mean — never the aggregate alone. (S-3) 2D-channel coverage in every mass-bearing cell (N-4).

**Grid step (verifier amendments 8c + D-5):** `--h-step 0.004` for Block A, Block N1, and the
n=800 cells of Block B; `--h-step 0.002` for the n=1600 cells (2× the default cost, measured
23.5 s/realization; the #67 fine-grid confirm — bias identical to ±0.0001 between 0.004 and
0.001 — is the fidelity warrant for 0.002).

**Seeds (verifier amendment 2 — pairing-preserving):** One master `--seed` per invocation, per
the instrument's interface. **Seed = 20270818 + 100·venue_index + 10·n_index** (venue_index ∈
{0: V-deep, 1: V-ctrl}, n_index ∈ {0: 250, 1: 800, 2: 1600}); the **SAME seed is used across all
noise-model × selection_cell cells of a given (venue, n)** so every cross-cell read is paired via
the shared generative stream (pinned by the harness's stream-alignment and channel-locality
tests). Truths run in one invocation. Base 20270818 lies outside every seed range consumed in
`results/` (prior coverage-family harnesses used 20260701–20261207). Block N1 uses 20260701 as
registered above — a deliberate reuse for replication, flagged as such.

**Scorer (pre-committed, same commit):** `readout_prodcal.py` in this directory computes, per
cell × truth × channel: MAP bias mean ± SE, cov50/68/90 ± binomial SE, rail fraction; per
registered pair: the [A2] per-realization delta mean ± SE, quartiles, and the **N-5 degeneracy
flag**. **The registered pair list is the `PAIRS` manifest at the top of the scorer (D-3); the
scorer invocation of record is `readout_prodcal.py --registered <cells_dir>`, which scores
exactly that manifest** — `--pair` outside the manifest is exploratory and never verdict-bearing.
The H-B precondition and every §4 band are evaluated ONLY on scorer output. No statistic outside
the scorer enters the verdict.

**Budget (measured dev-machine runtimes at the registered h-steps; ceiling binding):** Block A ≈
2 CPU-h; Block N1 ≈ 0.5 CPU-h; Block B ≈ 11 CPU-h (n=1600 half ≈ 9.4 CPU-h at 23.5 s/realization
× 120 R × 3 truths × 4 cells; n=800 half ≈ 1.7 CPU-h); secondary ≤ 1 CPU-h; total ≈ 14.6 CPU-h.
**Ceiling: 18 CPU-h** (margin against the D-5 mid-campaign-STOP failure mode), single machine,
no cluster. Exceeding the ceiling stops execution and returns to the author.

## 4. Falsifiable bands (registered)

Per truth, per cell; "floor" = MAP bias mean, sign-carrying. SEs from §2; paired reads on the
shared stream per §3.

| read | PASS | FAIL | MIXED (first-class) |
|---|---|---|---|
| H-P, production+fused, V-deep, n=1600 | \|bias\| ≤ 0.0010 AND cov68 ∈ [0.594, 0.766] | bias ≥ +0.0020 (coherent sign across truths) OR cov68 < 0.50 | anything else — incl. a sub-0.002 but ≥3σ floor: quantify vs §5 materiality |
| N-3, production+fused, V-ctrl | **paired** delta vs `off` twin consistent with 0 within 2·paired-SE AND no coherent sign flip. Reference (reported, NOT scored): the recomputed 07-11 mass-free control band −0.0030…−0.0010 ± 2·SEM ≈ [−0.0041, +0.0001] (verifier amendment 8e: the v1 band [−0.0045, +0.0010] did not follow from its recipe and is withdrawn) | the #66 flip: paired delta ≥ +0.0020 with sign change vs `off` twin | else |
| H-B, (const − production) paired floor delta, V-deep | **precondition holds (§1)** AND delta ≥ ½·const-floor | precondition holds AND delta ≤ ¼·const-floor ((b) dominates) | precondition holds, delta between ¼ and ½; **precondition fails ⇒ UNDETERMINED-BY-DESIGN, unscored** |
| H-N, production+fused across n = 250/800/1600 | \|bias\| ≤ 2·SE(n) at every n AND cov68 within its ±2·SE band at every n | bias flat in n AND cov68 monotone-collapsing (the #67 signature) | else |
| N-2, model+fused, V-deep | \|bias\| ≤ 0.0010 | \|bias\| > 0.0020 (instrument suspect: STOP) | else |
| N-1, replication cells vs 07-11 record (same seed) | within 2·SE | outside 4·SE (instrument suspect: STOP) | else |
| N-5, engagement | all registered paired deltas non-degenerate; **V-ctrl #66-direction check (D-4): channel-1d delta_mean(const+fused − const+off) > 0 with delta_mean ≥ 2·delta_se at every truth** | any degenerate delta, OR the V-ctrl shift ≤ 0 at every truth (instrument suspect: STOP) | else (shift positive but < 2·delta_se at some truth) |

**Branch calls (registered, returned to author as [RULE]s with the data):**
- **PASS branch:** H-P + N-3 + N-4 pass (N-1/N-2/N-5 clean) ⇒ the #66/#67 caveat is proposed
  CLOSED for the production configuration at harness fidelity; [R-3] proposed NOT NEEDED;
  stage-4 leg 1 (coverage) satisfied for the mass channel pending the absolute-count leg.
- **FAIL branch:** materiality per §5, then the one-more-measurement decision ([R-3] or a
  production-side measurement) returns as a fresh [RULE]. **No production change is proposed by
  any branch of this prereg.**
- **H-P PASS + N-3 FAIL** (control flips while deep passes — the #66 phenomenology recurring in
  the production configuration; verifier amendment 6): **first-class MIXED**; designated
  separating cell: the V-ctrl **mass-off** production+fused cell (does the flip need the mass
  channel?). Returns as a fresh [RULE].
- **Other MIXED:** the specific pattern is reported with its paired reads; the designated next
  measurement is the smallest cell that separates the live interpretations.

## 5. Materiality yardstick (registered before seeing data)

A detected floor b (in h) is compared against: (i) the campaign posterior widths of record
(row #119 context); (ii) the F5 forecast width at the campaign venue. **Material** = |b| ≥ ⅓ of
the narrower of the two. The comparison is a computation; its consequence is the author's [RULE].

## 6. Carried caveats and validity limits (registered)

1. **Venue transfer.** The harness is an analog, not production code. A PASS is evidence at
   harness fidelity, not a production certificate; the standing venue-scoping rule binds both
   ways. The [R-3] rung exists for a contested transfer.
2. **CC-1..CC-3 of the intake** carried unchanged.
3. **The harness's spec choices** (builder's ambiguity resolutions 1–8 in the build report,
   notably aggregate S̄_φ in Σ_glob; analytic S_4D without a `_G_SEL_S_VAR_TOL` mirror) are part
   of the instrument under test — a FAIL traced to one is an instrument finding; N-1/N-2/N-5
   are the guards.
4. **SBC blind spot.** Coverage alone cannot catch a filter both sides share (D1-class); this
   ladder tests the noise-pairing question where generator and estimator DIFFER per cell. The
   absolute detected-count audit (stage-4 leg 2) remains open and is not this prereg's business.
5. **Q-0 dependence.** H-P is calibrated against Q-0's UNPAIRED reading of production. Q-0's two
   decisive facts are independently re-verified in §7; the residual risk (a mis-read of a
   *third* production fact) is carried openly.

## 7. Execution appendix — filled in the same commit as this file, before any run; any later edit voids the registration

- **Harness of record:** frozen in this file's commit (`darksiren_emri/validation/pp_coverage.py`
  + `darksiren_emri_test/validation/test_pp_coverage_mass.py`; per-realization `maps` output
  included for the [A2] paired reads).
- **Q-0 independent verification (verifier amendment 7)** — orchestrator re-read, distinct from
  both the Q-0 auditor and the harness builder, 2026-08-17:
  (i) event d_L enters as the injected truth: `darksiren_emri/datamodels/detection.py:133-136` —
  `self.d_L = parameters["luminosity_distance"]`; `self.d_L_uncertainty = np.sqrt(parameters
  ["delta_luminosity_distance_delta_luminosity_distance"])` — where `luminosity_distance` is the
  injected value written at CSV-write time and the delta term is the Fisher/CRB diagonal;
  (ii) the scatter routine `convert_to_best_guess_parameters` has **no call site** in
  `darksiren_emri/` (grep, definition only); (iii) the kernel's 1/σ normalization is precomputed
  once, constant in z and h: `bayesian_statistics.py:3613` (`_log_norm_3d[slot] = -0.5*(3*
  log(2π) + logdet_3d)`).
- **V-deep production anchors** (zero-compute read, 2026-08-17):
  `results/run_20260817_fusion_counterfactual/{fused_iiib,fused_joint_r1}/simulations/
  diagnostics/event_likelihoods.csv` — n_events = 1588; catalogue-bearing event fraction
  (L_cat_no_bh > 0 at mid-grid h) = **0.618 (iiib) / 0.690 (joint_r1)**; mean completion share
  g_frac = **0.371** (both venues). **Pretuning procedure (registered, non-verdict-bearing):**
  one R=8, n=250 pretuning invocation per candidate (z_support, sky_frac) pair, tuning ONLY
  until the harness's host-in-ball fraction lands in [0.60, 0.70] and mean completion fraction
  in [0.30, 0.42]; the first pair to land is frozen here before Block A runs; pretuning outputs
  are archived in `pretuning/` and never scored. Chosen values: **z_support = 0.40,
  sky_frac = 1e-4** (filled 2026-08-18 by `uv run python -u pretune.py` under AMENDMENT-1,
  first-to-land at the tenth candidate: host_in_ball_fraction = 0.670, completion_fraction =
  0.330, both in band; disclosed descriptive facts per AMENDMENT-1 item 3: mean_ball_size =
  2.98, impostor_fraction = 0.775, empty_ball_fraction = 0.033 — carried under the §6
  venue-transfer caveat. This is the sole permitted §7 fill-in after commit, itself
  append-only.)
- **V-ctrl parameters (D-1):** the 07-11 shallow-control class, executable form:
  **z_support = 1.5** (= the harness Z-grid ceiling, making the truncation non-binding — the
  operative meaning of "untruncated" under catalogue_mode's z_support requirement),
  σ_z = 0.035, d50/w_pdet per `pp_coverage_noisemodel_20260711/RUNBOOK.md`, mass ON (α_M=0.25),
  **n_galaxies = 200000, sky_frac = 1e-4** (registered here; the control venue is fully
  determined).
- **CLI per mass-bearing cell (template, D-2):** `uv run python -m
  darksiren_emri.validation.pp_coverage --catalogue-mode --mixture-mode absolute
  --z-support {zs} --sky-frac {sf} --kernel volume --mass-channel --mass-horizon-index 0.25
  --n-galaxies 200000 --n-realizations 120 --n-events {N} --truths 0.62 0.72 0.84
  --h-step {hs} --noise-model {nm} --selection-cell {sc} --seed {seed}
  --output {cell_id}.json` where cell_id = `{venue}_{N}_{nm}_{sc}` (the scorer PAIRS manifest
  keys on this naming). `--mixture-mode absolute` is estimator-defining (the production
  `absolute_marginal` analog, intake §4) and is registered here, not left to execution.
- **CLI per Block N1 replication cell (D-2):** continuum mode, no catalogue/mass flags, the
  07-11 command lines per `pp_coverage_noisemodel_20260711/RUNBOOK.md` verbatim (z_support ∈
  {0.3, 0.5}, σ_z 0.035, {const | model+p_det-inside} cells, `--n-realizations 120
  --n-events 250 --h-step 0.004 --seed 20260701`), `--output n1_{zs}_{cell}.json`.
- **Pretuning discipline (D-7):** pretuning seed = **20270999** (fixed, disjoint from every
  campaign and scored seed); candidate sweep is the fixed lexicographic list z_support ∈
  {0.25, 0.30, 0.35} × sky_frac ∈ {1e-4, 2e-4, 4e-4}, first-to-land wins — a procedure, not a
  choice. Disclosure: the pretuning target (harness `host_in_ball_fraction` / completion
  fraction) and the production anchor (fraction of events with L_cat_no_bh > 0 at mid-grid h /
  mean g_frac) are **analog estimands, not the same quantity** — the tuning is to the analog's
  plausible-range image of the anchors, and this is registered as such.
- **Environment:** dev machine, single-core per cell, `uv.lock` of the freezing commit.

---

## VERDICT

**Filled 2026-08-18 from the registered scorer output (`readout_prodcal_output.json`,
`--registered` invocation, 18/18 pairs), under AMENDMENT-1 and DEVIATION-1 below. Execution:
cluster job 6355028, 14.1 of 18 CPU-h, 26/26 invocations. Full comprehension artifact:
`CAMPAIGN_REPORT_20260818.md` (A7). Branch presented, not adjudicated.**

- **H-P: FAIL** (production+fused, V-deep, n=1600: bias −0.032, cov68 0.000 — the cov<0.50 leg).
- **H-N: asymptotic-bias signature FIRES** (flat −0.030 across n=250/800/1600, cov68 collapsing).
- **H-B: UNDETERMINED-BY-DESIGN** (registered precondition unmet — the const+fused floor is
  dominated by the fused shift, not a decomposable const floor).
- **N-1: PASS after execution erratum** — first pass ran `two_branch` (driver omitted
  `mixture_mode="exact"`; quarantined in `cells_unfaithful_n1/`; diagnosis: the +0.0235 matched
  the July two_branch record exactly); the faithful rerun reproduces the July record on all cells.
- **N-2: FAIL → registered STOP fired.** Instrument audit (read-only) resolved it: no code
  defect; the −0.03 fused shift is a first-order tilt (slope·σ² predicts −0.0309 vs measured
  −0.030) driven by S̄_φ's ~5× gradient across V-deep's completion window; the band was anchored
  on a venue where that gradient is absent.
- **N-3: VOID · N-5: STOP at V-ctrl, resolved as structural void** — the D-1 amendment's
  z_support=1.5 exceeds Z_MAX_POP=0.95, emptying the completion window (completion_fraction ≡ 0):
  the control could not test what it was registered to test.
- **N-4: PASS** (2D fused−off delta −0.0006); the 2D channel's +0.01 bias is present in `off`
  identically — venue noise physics, NOT a fusion effect (misattribution corrected in the
  report).
- **N-5 at V-deep: PASS** (all registered deltas non-degenerate).

**Fired branch:** H-P FAIL → §4 FAIL branch (materiality per §5, then author [RULE]). §5
computation: |−0.032| ≥ ⅓ × campaign width 0.0053 — *at the harness venue*; the audit's
regime analysis and production's own zero-MAP counterfactual are the interpretation inputs put
before the author. Decisions table: report §10. No production change is proposed by this verdict.

---

## AMENDMENT-1 — 2026-08-18 — pretuning sweep extension (verifier Part-IV pre-check: GO; author: "approved")

The registered pretuning candidate sweep exhausted without landing: host_in_ball_fraction =
0.279 / 0.415 / 0.534 at z_support = 0.25 / 0.30 / 0.35, monotonically short of the [0.60, 0.70]
target, and insensitive to sky_frac (which was verified live on the archived cells:
mean_ball_size 0.91→7.17 and impostor_fraction 0.693→0.925 across the sky_frac range, as
designed — host-in-ball is a pure z_support truncation effect). **The registered candidate sweep
is extended by z_support ∈ {0.40, 0.45} × sky_frac ∈ {1e-4, 2e-4, 4e-4}, appended to the
lexicographic order after the original nine; same pretuning seed 20270999, same targets, same
first-to-land rule.** sky_frac remains swept, not pinned: landing requires both targets, and the
completion-fraction band has not been shown sky_frac-inert.

Discipline statements (verifier Part-IV conditions, incorporated verbatim):
1. This amendment is committed BEFORE any extension pretuning cell runs.
2. **No MAP/coverage field of the archived pretuning outputs was read** — only the two
   registered tuning-target fields (host_in_ball_fraction, completion_fraction) and the ball
   diagnostics (mean_ball_size, impostor_fraction, empty_ball_fraction) were consulted.
3. Impostor loading is NOT added as a tuning target. The frozen pair's mean_ball_size and
   impostor_fraction are recorded in the §7 fill-in line as **disclosed descriptive facts**,
   carried under the §6 venue-transfer caveat.
4. **Exhaustion clause:** if the extended sweep also exhausts, execution stops and returns to
   the author; any further extension is an AMENDMENT-2 under the same pre-check discipline.
5. The six R=8 extension cells are budget-negligible; the 18 CPU-h ceiling is untouched.

---

## DEVIATION-1 — 2026-08-18 — execution environment migrated to bwUniCluster (author instruction)

The §7 environment ("dev machine, single-core per cell") is superseded on the author's explicit
instruction ("please submit it to the cluster instead" — the dev machine is needed for other
work): execution moves to bwUniCluster 3.0, one `cpu`-partition node, 26 workers (one per
registered invocation), via `run_ladder.sbatch` in this directory. **Nothing verdict-bearing
changes:** same frozen code (the freezing-commit chain must be checked out on the cluster and
its hash recorded in the `.out` log), same registered seeds (the paired-seed scheme lives inside
`run_ladder.py`; no SLURM per-task seeding is layered over it), same scorer. The local run had
completed 0 of 26 invocations when stopped (verified: empty `cells/`), so no partial results
mix environments. Known residual: per-realization runtimes will differ from the §3 dev-machine
anchors (contention profile differs); the CPU-h ceiling continues to bind on measured CPU time.
