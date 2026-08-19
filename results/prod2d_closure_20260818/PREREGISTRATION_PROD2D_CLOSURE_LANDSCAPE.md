# PRE-REGISTRATION — Production-2D closure + catalog-quality landscape

**Date:** 2026-08-18 · **Status:** DRAFT v1 — awaiting verifier pre-check. **Execution
authorization:** author's verbatim directive (2026-08-18): *"I want this measured for sure!
please do that closure. you can also think about if anything else is needed to be run on the
cluster and please go ahead autonomous, I will ensure the ssh connection one more now, so you
should keep it active over night. if would be huge news if given this horrible data (redshift
and mass error) 1d starves while 2d is able to constrain. we could then further extrapolate
other realisations of the errors due to improved measurements ( in the best case motivated by
known missions to come) and have a landscape that clearly tells you: given this good of a
catalog we can expect that constraint."* — grants the closure measurement, cluster use, and
overnight autonomous execution (orchestrator-derived reading, flagged for veto). **Every
verdict branch returns to the author presented, not adjudicated.** **Append-only after
commit; §7 edits void the registration except the registered fill-ins.**

**Provenance gating:** row #123 item 3 + row #124 (bias threads closed by ownership-of-classes
with documented residuals — the specific number-to-number production-2D budget never
computed); rows #111/#116 (venue threads M-OWNED-CLOSED; channel B owns the venue excess to
~6%; fused estimator zero-consistent in venue); row #119 (fusion near-inert in production 2D
⇒ channel B's class cannot own production's offset); the author's closure directive above.
The production offsets of record (readout.json M3, off legs): **iiib 2D mean_h 0.7842
(Δ = +0.054), joint_r1 0.7966 (Δ = +0.067)**, σ_h 0.0177/0.0216, single realization
(seed61000, shared CRB between venues [LOCAL recon 2026-08-18]).

**Exoneration check:** no §2 item is re-opened. The mass-channel entries (§2 items 1, 6) are
venue-scoped verdicts on other questions; this campaign measures scatter and budget, proposes
no mechanism re-litigation and no production change on any branch. **Stage-L note:** R0 for
this thread = the mechanism-study L-series (already run, rows #104–#111); no new sweep —
lightweight rule satisfied by the front's direct inheritance from those threads.

**Load-bearing recon facts ([LOCAL], 2026-08-18, both flagged into the design):**
1. **iiib's 2D pull is single-event-dominated:** event 889 (SNR 1424.7, rank 1/1588,
   in-catalog, σ_dL/dL ≈ 9e-4) carries a per-event h-slope of the 2D combined likelihood
   85× the next largest; joint_r1's pull is broad-based (~10 events).
2. **Production σ_M of record:** R&V15 with ε₀ = 0.24 dex intrinsic scatter (post-`555f018`);
   documented empirical 2D impact of the Eddington-in-M treatment: mean_h −0.020
   (`bayesian_statistics.py:5454`). The harness mass-error analog is FRACTIONAL
   (`sigma_m_gal_frac`); mapping 0.24 dex → σ_frac ≈ ln(10)·0.24 ≈ 0.55 (first-order;
   the mapping is an analog, not an identity — carried as a §6 caveat).
3. Harness 2D channel at V-deep n=1600 (existing prodcal cells): bias +0.008…+0.017
   (present in `off`; the audit's noise-coupling class: photo-z ~+0.006 + mass-obs ~+0.005,
   both →0 as σ→0), realization scatter map_std 0.0032–0.0043.

---

## 0. The three tiers

- **T0 — production-side free reads (local, pre-committed scorer, zero new simulation):**
  bootstrap + jackknife on `event_likelihoods.csv` (both venues, both channels). Measures the
  EVENT-DRAW realization scatter of the production estimator itself and the fragility of each
  venue's offset. Production-native: no venue-transfer caveat at all.
- **T1 — closure factorial (cluster):** harness cells at the calibrated deep venue,
  production-N, toggling (σ_z, σ_m_gal) to attribute the harness 2D bias class and measure
  the full generative realization scatter (catalogue + events) at production-mapped noise.
  Mechanism attribution at CLASS level — magnitudes carry the standing venue-scoping caveat.
- **T2 — catalog-quality landscape (cluster):** the (σ_z × σ_m_gal) grid, both channels,
  production-N: bias, realization scatter, coverage, and the CALIBRATED constraint
  (RMS error) per cell — the "given this good a catalog, expect this constraint" deliverable.

## 1. Hypotheses and registered questions

- **H-T0a (fragility, iiib).** Dropping event 889 collapses iiib's 2D offset: registered
  read Δ_2D(iiib, without 889) with band: FRAGILE if |Δ_without| ≤ ½·|Δ_full| (0.027);
  ROBUST if ≥ ¾·|Δ_full|; MIXED between. Two-sided: a sign flip beyond −0.01 is
  reported as OVERSHOOT (the event was masking a negative ensemble pull — a distinct,
  quotable outcome).
- **H-T0b (scatter).** The event-bootstrap scatter σ_boot(2D mean_h) per venue. No
  PASS/FAIL — this is a measurement; its registered USE: the closure z-score
  z_v = Δ_v / σ_boot,v per venue. Registered interpretation bands: z ≤ 2 ⇒ the offset is
  consistent with event-draw luck alone; 2 < z < 4 ⇒ partial; z ≥ 4 ⇒ event-draw scatter
  alone cannot own the offset (systematic component required). (Bootstrap covers event-draw
  variance only — not catalogue/sky draw; T1's full generative scatter complements; §6.)
  **Grid-truncation diagnostic (P7-5, registered):** joint_r1's full-sample posterior
  retains ≈ 29% of peak at the penultimate node and 1.8% at h = 0.86 (iiib: 4.8%/0.2%) —
  upward resamples truncate, biasing σ_boot low and z_v high. The scorer reports the
  top-2-node relative mass per resample; if the median exceeds 0.05 for a venue, a z_v in
  (2, 4) is flagged UNDETERMINED-BY-GRID-TRUNCATION (B-OWNED-SCATTER firing anyway is
  conservative and stands).
- **H-T1a (class attribution).** At the deep venue, production-N, the harness 2D bias at
  (σ_z = 0.035, σ_m_gal = 0.55): both-small toggle (σ_z = 0.002, σ_m_gal = 0.02) collapses
  it. **Engagement precondition (P7-3):** |bias(anchor)| ≥ max(0.004, 5·SE) at the primary
  truth (else the collapse read is UNDETERMINED-BY-DESIGN, unscored — there is no amplitude
  to attribute). Under the precondition: PASS(class-owned) if |bias(both-small)| ≤
  max(0.002, 2·SE) AND each single toggle reduces |bias| by ≥ 25%; FAIL if
  |bias(both-small)| ≥ ½·|bias(anchor)| with the anchor's sign (a σ-independent 2D bias
  component exists — NEW CLAIM intake); MIXED else. The precondition floor 0.004 keeps the
  PASS cap (0.002) and FAIL edge (≥ ½·anchor ≥ 0.002) disjoint — no both-fire.
- **H-T1b (production-mapped amplitude).** The anchor-cell 2D bias at σ_m_gal = 0.55 vs the
  existing σ_m_gal = 0.30 class (+0.009…+0.012): registered expectation (REPORTED-ONLY
  anchor) ≈ ×2–4 growth if the mass-error channel dominates. No adjudicating band — feeds
  the budget table.
- **H-L1-prod (the headline "1D starves / 2D constrains" read — production-native, T0).**
  Per venue, from the registered T0 scorer: 1D-starves = the production 1D-channel
  posterior is uninformative on the grid (68% HPD width ≥ ½ the grid span OR posterior
  mode on the grid edge); 2D-constrains = the 2D posterior width σ_h with the H-T0b
  closure z-score caveat attached (a 2D "constraint" is quoted as σ_h ⊕ σ_boot and carries
  the B-branch outcome). This is the only arm on which the author's "1D starves while 2D
  constrains" sentence may be quoted.
- **H-L1-harness (the landscape frontier — class-level, venue-scoped).** Per grid rung:
  2D-constrains = |2D bias| ≤ max(0.002, 2·SE) AND 2D cov68 ∈ [0.594, 0.766] AND
  RMS ≤ 0.02, scored on the fused cells (the selection lever is certified +0.001-class);
  1D reference = the **off-basis** 1D read at that σ_z (three off cells, §3-amended below;
  the 1D channel is mass-blind, so three σ_z rungs cover the full grid): 1D-calibrated /
  degraded / starved by the same band family. The fused-1D values are reported alongside
  with the insertion delta (fused − off) explicitly attributed to the venue-scoped
  asymmetric-insertion class (rows #120–#124, G-2 σ_z-collapse) — the fused-1D failure at
  this venue is NEVER quoted as "1D starves". *(P7-1, applied verbatim.)*
- **H-L2 (frontier existence).** There exists a grid rung where the 2D channel calibrates
  (H-L1-harness legs) AND the off-basis 1D read at that rung's σ_z is calibrated (|bias| ≤
  max(0.002, 2·SE), cov68 ∈ [0.594, 0.766], rail ≤ 0.10) and the 2D calibrated RMS ≤ 0.01:
  expected at the good corner per the audit's σ→0 prediction. *(P7-1c.)*
  FAIL (no rung calibrates the 2D channel anywhere) = a σ-independent 2D venue bias —
  same intake as H-T1a FAIL.
- **Expected NULLs:** N-1 continuity — the (0.035, 0.30) grid cell reproduces the existing
  prodcal `vdeep_1600_production_fused` class (bias within 3·combined-SE per truth per
  channel; different seed/grid ⇒ class comparison). N-2 engagement/preflight per §3b.
  N-3 grid-quantization guard: map_std must be ≥ 1.5× h_step for the σ_real read to be
  quoted (else UNDETERMINED-BY-QUANTIZATION at that cell — h_step 0.002 chosen so even
  0.003-class scatter passes). **P7-7 (registered):** at good rungs the true map_std is
  expected below the 0.003 floor (G-2 rung-0.002 classes scale to ~0.0005 at n=1600):
  flagged cells quote σ_real and RMS as upper bounds ("< max(measured, 1.5·h_step)") in the
  landscape table — the deliverable's best-catalog rows are resolution-bounded, stated as
  such. A finer-grid (h_step 0.001) rerun of the ≤ 4 best rungs (~+10 CPU-h) is a
  registered author option at readout, not run by default.

## 2. Stage-1 sizing (measured anchors, no Fisher leg)

- T0: bootstrap B = 10,000 over 1588 events × 41-point grids = trivial (minutes, local).
  SE(σ_boot) ≈ σ_boot/√(2B) — negligible.
- T1/T2 scatter precision: R = 120 ⇒ SE(map_std) ≈ map_std/√238 ≈ 6.5% relative — adequate
  for a landscape. Bias SE at the existing n=1600 2D class (map_std 0.0036) ≈ 3.3e-4 ⇒ the
  +0.01-class harness bias is a ≥ 30σ read; toggle collapses decisive.
- Cost (measured cluster anchors, 26-worker node, prodcal): n=1600 fused ≈ 12,918 s at 131
  h-nodes; this campaign's grid (h ∈ [0.56, 0.92], step 0.002 ⇒ 181 nodes) scales ≈ ×1.38 ⇒
  ≈ 17,850 s ≈ 5.0 CPU-h/cell (upper bound; sub-linear in practice). 15 cluster cells
  (12 T2 grid + 1 anchor off twin + 2 V-prod) ≈ 70–75 CPU-h (off/V-prod cells are cheaper
  than the fused V-deep class); +3 off-basis 1D cells (P7-1) ≈ +3.7 ⇒ ≈ 74–79 CPU-h total —
  **ceiling 160 CPU-h** unchanged, wall at 18 workers still ≈ 5–6 h (fits overnight;
  walltime request 14 h — margin ≥ 2× against the contended anchor per cluster gotcha 6/9).
  *(P7-6, applied verbatim.)*

## 3. Design — cells, venues, seeds

**T0 (local):** scorer `tier0_bootstrap_jackknife.py` (pre-committed) on
`results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/event_likelihoods.csv`:
per venue × channel: full-sample posterior (Σ per-event logL on the 41-grid, uniform prior —
the canonical Σ log L reference form), jackknife-889, full leave-one-out distribution,
drop-top-k curve (k = 1…20 by |slope|), bootstrap B = 10,000 (resample 1588 event columns
with replacement): mean_h/MAP distributions → σ_boot, q05–q95. Seed 20280611 (numpy default_rng).
**Registered T0 conventions (P7-2, applied verbatim):** (a) all grid moments use trapezoid
weights w_i = gradient(h) on the non-uniform 41-grid (the naive equal-weight mean differs by
−0.004/−0.006 — materiality-scale); (b) the per-event 2D likelihood is assembled from the
CSV by the registered formula in the scorer header (pinned at freeze; the canonical
Σ log L reference form of `posterior_combination.py`); (c) the physics-floor zero-handling
of the production combine (strategy of record, n_events_empty = 2) is applied identically.
**N-0 continuity gate (scored):** the T0 full-sample mean_h must reproduce the M3 anchors
0.7842 (iiib) / 0.7966 (joint_r1) to within 5e-4 per venue BEFORE any bootstrap/jackknife
statistic is quoted; failure ⇒ STOP (convention mismatch), no T0 number enters the budget.

**T1 + T2 (cluster, one job):** venue = V-deep (z_support 0.40, sky_frac 1e-4, default d50 —
the venue whose off cells are calibrated in 1D), n_events = 1600 (production-N class,
production is 1588), R = 120, truths {0.62, 0.72, 0.84}, noise = production,
selection_cell = fused, mass ON (α_M = 0.25), sigma_mz_frac = 0.10 (LISA-fixed; not swept),
**h grid [0.56, 0.92], h_step 0.002 (181 nodes)**, n_z_quad 160, **one master seed 20280411
shared by ALL T1/T2 cells** (σ knobs enter the generative stream continuously ⇒ every
cross-cell read is paired):

- **T2 grid (12 cells):** σ_z ∈ {0.035, 0.010, 0.002} × σ_m_gal ∈ {0.55, 0.30, 0.10, 0.02}.
- **T1 reuses four grid corners** — anchor (0.035, 0.55), photo-z toggle (0.002, 0.55),
  mass toggle (0.035, 0.02), both-small (0.002, 0.02) — plus **1 off twin** at the anchor
  (0.035, 0.55) for the paired selection read.
- **1D off-basis cells (3):** selection_cell = off at (σ_z, σ_m_gal) = (0.035, 0.55),
  (0.010, 0.55), (0.002, 0.55), same seed 20280411 — the 1D-leg basis for H-L1-harness
  and H-L2 (≈ +3.7 CPU-h; total cluster cells 18). *(P7-1, applied verbatim.)*
- **V-prod secondary (2 cells, seed 20280511, paired-only per the registered confound):**
  fused + off at (0.035, 0.55), z_support 0.75, d50×8 — the flat-S̄ regime check that class
  conclusions do not flip in the production-analog completion geometry (paired deltas only;
  absolute legs descriptive, VERDICT-3 confound).

**Total cluster cells: 15** (13 V-deep + 2 V-prod). Execution: one sbatch job, 1 `cpu`
node, 15 single-core workers (ProcessPoolExecutor, prodcal `run_ladder` pattern),
`--time=14:00:00`, per-cell JSON written at cell end into the workspace run dir, idempotent
skip-if-exists (a walltime kill loses only in-flight cells; resubmit completes the rest).
Repo on cluster: pull current main (freeze commit) — ONE repo, tag `prod2d-closure-base`.

**Scorers (pre-committed same commit):** `tier0_bootstrap_jackknife.py`,
`readout_prod2d.py --registered` (per cell × truth × channel: bias ± SE, map_std, cov68 ±
binomial SE, rail fraction, RMS error = √(bias² + map_std²); registered pair list: every
grid cell − anchor cell, fused − off at both venues; [A2] paired distributions alongside
class means; rail gate > 0.10 ⇒ UNDETERMINED-BY-RAIL with the A-PF-1 precedence; the
1D floor-rail at bad rungs is an EXPECTED registered outcome (it is the "1D starves"
phenomenon), reported as rail fraction, never silently dropped).

## 3b. Arm-validity preflight (registered, non-scored — the anti-void gate)

Before submission, R = 4 probes of: the anchor cell, both toggles' extreme corners
(0.002, 0.02) and (0.035, 0.02), and one V-prod cell — run LOCALLY at exact registered
configs (probe-flagged, archived `preflight/`): completion/catalogue-bearing fractions in
bounds ([0.05, 0.95] / > 0.3 V-deep, [0.05, 1.0) V-prod), finite MAPs, no 2D probe rail
= 1.0, mass-channel engagement (2D maps non-degenerate vs 1D), and at the good corner the
1D channel must UN-rail (else the landscape's good-corner claim is void — the exact
grid-headroom class again). Any violation ⇒ STOP, amend before submission. Probe-scale
byte/N-A comparisons are VOID (equal-R rule of record); N-1 continuity is scored at full R.

## 4. Verdict framework (registered before any run)

The closure verdict is an ASSEMBLED BUDGET, presented to the author:

| leg | source | transfer status |
|---|---|---|
| Δ_2D per venue (+0.054 / +0.067) | production readout.json (off legs) | production-native |
| fragility: Δ without 889 / top-k curve | T0 jackknife | production-native |
| event-draw scatter σ_boot per venue | T0 bootstrap | production-native |
| σ_M Eddington shift −0.020 | documented (`:5454`, G7row9) | production-native |
| noise-coupling class (photo-z, mass-obs) sign+collapse | T1 toggles | CLASS-level only (venue-scoped) |
| full generative scatter at production-mapped σ | T1 anchor map_std | CLASS-level only |
| selection lever | T1/V-prod fused−off pairs | already certified +0.001-class (row #124) |

**Registered branch calls (each a [RULE] returning with the data):**
- **B-OWNED-SCATTER:** z_v ≤ 2 for a venue ⇒ that venue's offset needs no systematic beyond
  event-draw luck; the honest "2D constraint" for one realization is σ_h ⊕ σ_boot.
- **B-OWNED-BUDGET (P7-4, applied verbatim):** z_v > 2 but the production-native residual
  r_v = Δ_v − s_Edd (s_Edd = the documented −0.020 Eddington-in-M shift where its
  configuration applies to venue v, else 0) satisfies |r_v| ≤ 2·σ_total,v with
  σ_total,v = σ_boot,v ⊕ u(s_Edd) (u(s_Edd) registered at freeze; if no uncertainty is
  documented it enters as a point value with that stated) ⇒ closure by budget; the
  residual is quoted with its uncertainty. **Only production-native magnitudes enter this
  arithmetic.** The T1 class attributions may only be cited as sign/collapse consistency
  for the residual's CLASS (photo-z + mass-obs coupling), never as magnitude legs; the
  fragility read (jackknife-889/top-k) modifies the INTERPRETATION (which Δ_v is quoted:
  full vs without-889), not σ_total.
- **B-UNOWNED:** a residual ≥ 2·σ_total remains after all legs ⇒ NEW CLAIM intake
  (stage 0) with the residual as its size; no mechanism asserted here.
- **Landscape products (T2):** the per-cell (bias, σ_real, cov68, RMS) tables + the H-L1 /
  H-L2 branch outcomes; the mission mapping (σ_z: GLADE 0.035 / LSST-class 0.02–0.03 /
  spec-z 0.002-class; σ_m: R&V-current 0.55-frac / improved-EM ~0.25 / optimistic 0.02) is
  a PRESENTATION overlay added at readout, never band-bearing.
- **Execution-completeness:** no branch adjudicated while any registered cell or T0 read is
  missing (or a STOP fired); T0 may be REPORTED before the cluster job returns (it is
  production-native and self-contained), flagged interim.
- **Venue-correlation disclosure (P7-8, registered):** the two venues share the single
  event realization and CRB set (seed61000 recon): z_iiib and z_joint are strongly
  correlated draws of the same universe — venue agreement is ONE realization's evidence,
  never counted as two independent confirmations in the budget presentation.

## 5. Materiality yardstick

Production posterior widths σ_h 0.0177/0.0216 ⇒ material residual = ≥ ⅓·0.0177 ≈ 0.006 in h.

## 6. Carried caveats and validity limits (registered)

1. **Venue transfer binds both ways** — T1/T2 magnitudes never transfer to production; only
   production-native legs (T0, the documented −0.020) enter the budget quantitatively at
   production scale; harness legs enter as class signs/collapse factors.
2. **Bootstrap covers event-draw variance only** (fixed catalogue/sky realization); the full
   generative scatter is T1's map_std — measured at harness fidelity. The gap is stated in
   the budget, not silently bridged.
3. **The σ_m mapping (0.24 dex ↔ 0.55 fractional) is a first-order analog**; the harness has
   no R&V15 lognormal (a production-matched mass model would be a physics-change-class
   build, explicitly out of scope here).
4. **Harness population n(z) is the fixed comoving form** — the F5 caveat (frontier is
   population-dependent) carries to every landscape statement.
5. **h_step 0.002 quantization** — N-3 guard; σ_real reads below 0.003 are flagged.
6. **1600 vs 1588** — production-N class, not identity.
7. **No production change on any branch.**

## 7. Execution appendix — filled at freeze; cluster fill-in appended post-submission

- Instrument = the freeze commit (harness unchanged from `4dd822ad` class; only new driver/
  scorer scripts in this directory). Scorers/drivers hashes = this commit.
- Invocations: T0 `uv run python tier0_bootstrap_jackknife.py` (local) →
  `tier0_output.json`; cluster `sbatch run_prod2d.sbatch` (JOB_TEMPLATE-derived; module
  load + venv per cluster skill; RUN_DIR in workspace; cells copied back to this directory
  post-run); readout `uv run python readout_prod2d.py --registered cells/`.
- Preflight: local R = 4 probes (§3b) archived BEFORE submission; cluster preflight
  (`cluster/preflight.sh`) must print READY ✓ before sbatch (printed 2026-08-18: READY ✓).
- **Cluster fill-in (append-only):** job id, node, wall, per-cell timings, any resubmits.

**CLUSTER FILL-IN (2026-08-19, appended):** cluster repo fast-forwarded to the freeze commit
`d6fc1ccf` (tag `prod2d-closure-base`); colliding untracked prodcal originals moved to
`~/prodcal_untracked_backup_20260819/` and verified bit-identical (md5) to the tracked
copies before merge; **job 6364803** submitted (cpu partition, 1 node, 18 workers,
walltime 14 h) at 2026-08-19 ~01:20; T0 registered output `tier0_output.json` produced
locally pre-submission (N-0 gate PASS: 2.2e-5 / 1.1e-4 vs 5e-4 tolerance). GitHub push of
the freeze commit was REJECTED by a pre-receive hook (reason to be captured; cluster and
local copies are the working records — the origin sync returns as a session housekeeping
item, not verdict-relevant).

---

## VERDICT

*(append-only below this line after execution)*

**CLUSTER FILL-IN 2 (2026-08-19, appended):** job **6364821** (resubmission of 6364803 with
RUN_DIR export) ran to ~12.7 h of 14 h wall; **5/18 cells completed** — exactly the five off
cells (vdeep_anchor_off 8772 s, vdeep_off_sz{0.002, 0.010, 0.035} 9330/11555/9620 s,
vprod_anchor_off 19302 s); all 13 fused cells still in flight at 12.3 h against the §2
scaled estimate ≈ 5 h/cell (18-worker single-node contention; the estimate's 26-worker
prodcal anchor did not transfer). Author-directed CANCEL in preference to a timeout kill
(verbatim: *"If we can find evidence that these jobs will not finish, we can also stop them
right away and prepare the next round."*).

**AUTHOR [RULE] (2026-08-19, verbatim):** *"but if we find that we need another physics
change due to the 2d residual we need to rerun the landscape job anyway, or not? so we could
gate that behind the final resolution of the bias"* — the 13 fused T1/T2 cells are
**DEFERRED, gated behind the final resolution of the 2D residual** (fix fork a/b ⇒ the
landscape is measured once, with the corrected estimator; fork c ⇒ the current-estimator
landscape runs as registered). This amends §4's execution-completeness clause for the
closure: branches are presented on the production-native legs (T0, s_Edd, the executed
production-native regression `PREREGISTRATION_PROD_REGRESSION.md`) plus the five banked off
cells as interim class evidence, with the deferral stated. The five off cells remain fully
registered reads (H-L1-harness 1D basis; σ_z-ladder class evidence). The mechanism doc's
blind T2 predictions (`MECHANISM_SIGMA_M_SIGMA_Z_DERIVATION.md` §5) remain registered and
apply to the current-estimator landscape whenever it runs.
