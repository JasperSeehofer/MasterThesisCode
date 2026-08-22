# PRE-REGISTRATION — [P3-IMP] "the catalogue-leg twin" (stage 2)

**Date:** 2026-08-22 (overnight, row #160 grant) · **Thread:** `[P3-IMP]`
(claim file `CLAIM_P3_IMPOSTOR_CONVENTION_20260822.md`, stages 0–1 complete).
**Append-only after commit.** A21 governs: any premise correction discovered during
implementation STOPS execution; this file is amended and bands re-derived first.

## 1. The mechanism hypothesis (derivation-grounded, stage-1 refined)

**[INFER from verified code reads + [DOC]]** The production catalogue leg has the SAME broken
numerator/normalizer pairing the fused fix repaired in the completion leg, on its own side of
the mixture:

- Its mixture normalizer **carries the survival factor**: `beta_G_phi(h) = ∫ f̄·S̄_φ·w_pop dz`
  (`bayesian_statistics.py:2065`, verified at source), rescaled into the mixture weight
  `alpha_G_phi = Σ_4D·β_G_φ/Σ_φ` (`:2423-2427`).
- Its per-host numerator **carries none**: `numerator_integrant_without_bh_mass`
  (`:5889-5907`) = GW-Gaussian × volume-deconv host-z kernel, with the `:5896-5901` comment
  codifying the absence as the Gray (2020) Eq. (A.10)/MFG convention.
- The generator is latent-thresholded — the accepted per-event density carries S̄_φ (proven,
  A20/O4 review FATAL-2 table + O6/row #158) — so for THIS venue the generator-matched
  catalogue numerator gains a per-host S̄-type factor, exactly as the completion numerator did
  (L6-DER3 §4: "the catalogue leg is the same fork, per-galaxy"; a per-host S̄-type factor in
  the 1D leg).

**Registered approximation:** the per-host survival is taken as the φ-marginal S̄_φ(z;h)
(population proxy for the host's own p_gal — L6-DER3's "with p_gal in place of φ" refinement is
a registered follow-up, NOT tonight's measurement). The FULL-F LOO weight (1/imp_k) and the
density-form GW factor (V2/D-ii) are NOT included — this registration isolates ONE axis: the
survival-factor pairing. (Ledger §2 item 6 compliance: this is a PAIRED insertion — the
numerator gains the factor its own normalizer already integrates — not the exonerated
"p_det alone" move.)

## 2. The instrument (branch-only; instrumentation-tagged, not a physics change)

Branch **`p3/catalogue-survival-counterfactual`** (never merged tonight; row #160 grant 3).
`BayesianStatistics` gains `catalogue_numerator_survival ∈ {"off","phi"}`:

- `"off"` (default): **byte-identical** to production — the whole-tree test suite plus GATE
  R-P3 prove it.
- `"phi"`: `numerator_integrant_without_bh_mass`'s integrand is multiplied by
  `np.interp(z, *self._phi_survival_table[h])` (endpoint-clamped — the SAME table object the
  normalizer chain integrates; the O6-verified identity `:3812→:3819→:4133` extends to this
  consumer). Applied in **BOTH dispatch paths** — `single_host_likelihood` AND
  `single_host_likelihood_batch` (A13; the batch path is the one production actually uses).
  Log line on activation: `COUNTERFACTUAL: catalogue_numerator_survival='phi' — per-host
  S̄_φ in the catalogue numerator ([P3-IMP] twin cell). Not a production posterior.`
- The **with-BH-mass** catalogue numerator (`mz_integral` leg) is HELD at coded (invariant;
  its S-fork is L6-DER3's 2D analog — follow-up, not tonight).
- Refused unless `normalization_mode="absolute_marginal"` (the S̄_φ table exists only there).

## 3. Arms

| arm | what runs | seeds | cost |
|---|---|---|---|
| **LEV (leverage instrument, pre-pilot)** | zero-`evaluate()`: per banked B-SEL seed, deterministic candidate redraw; compute the share_cat-weighted prediction of the per-event catalogue-term reweight `∂_h ln S̄_φ(z_obs,k)` → predicted per-seed Δscore order of magnitude | 12 | < 15 min local |
| **P (replica gate)** | branch code, flag `"off"`, fresh work root → bit-exact vs banked columns (O6 D6 pattern: ≤1e-12 fallback + wall-time floor > 60 s) | 900101 | ~30 min |
| **PILOT** | flag `"phi"`, 2 seeds (900101, 900102 — first two banked, registry order) → realized paired σ̂ for band freezing + costing line | 2 | ~1 h local |
| **F-φ (primary)** | flag `"phi"`, all 12 banked B-SEL seeds (fresh work roots) | 12 | ≈ 6 CPU-h — local 2-wide ~3 h; cluster fallback (preflight first) within the 50 CPU-h cap |
| **K-flat (kill test, reported-only)** | 1 seed with S̄_φ replaced by its window-average (constant): if the effect is the z-SLOPE mechanism, K-flat ≈ coded | 900101 | ~30 min |

Coded baseline = the banked 12-seed columns and posteriors (no re-run; GATE R-P3 anchors
comparability).

## 4. Gates (scored before any statistic is read; failure ⇒ VOID)

- **GATE R-P3:** arm P reproduces the banked `event_likelihoods.csv` `L_cat_no_bh`, `B_num`,
  `combined_no_bh` columns bit-exact (≤1e-12 multiprocessing fallback), wall > 60 s.
- **GATE E-P3 (A13 engagement + dispatch):** under `"phi"`, ≥10% of catalogue-bearing events'
  `L_cat_no_bh` move ≥1e-6 relative vs banked, AND runtime assertions confirm the flag value
  reached BOTH the scalar and batch host-likelihood paths (counter or log per path).
- **GATE L-P3:** the counterfactual log line present in every `"phi"` run, absent in arm P;
  flag recorded in each run's metadata.
- **GATE N-P3 (A18):** every readout prints, per statistic, the reference it subtracts and the
  band constants, machine-readable.

## 5. Statistics and bands

**Primary:** paired per-seed `Δ_s = mean_h(F-φ, s) − mean_h(coded banked, s)`, fleet mean
`Δ̄(12)` with **paired** SEM (the C-SG BAND R lesson: same seeds, same realizations — the
paired band is the informative one).

**Band FORMULAS (registered now; NUMBERS frozen from the pilot's realized paired σ̂ and
appended below pre-fleet — the O2/pilot precedent):**

- **TWIN-MATERIAL:** `|Δ̄(12)| > max(3·SEM_paired(12), 0.02)` — the catalogue-leg pairing owns
  a material share of the headline bias; the production-facing fix becomes a physics-change
  [RULE] for the author (branch implementation permitted tonight per grant 3; merge never).
- **TWIN-IMMATERIAL:** `|Δ̄(12)| ≤ 0.01` AND `SEM_paired(12) ≤ 0.004` — a precision-qualified
  bound; [P3] returns to the row #110 paper task carrying it.
- **REPORT-BOUND (first-class):** anything else — report `Δ̄ ± SEM` as a bound, no label.

The 0.02/0.01 anchors are ~25%/13% of the impostor drag (−0.079) and ~2σ/1σ of the UNPAIRED
fleet SEM; they are provisional pending the pilot σ̂ re-derivation (A17 realized-scatter
re-check) and may only TIGHTEN, never widen, after data exist.

**Secondary (registered, reported with the verdict):**
1. O2-style impostor decomposition re-run on the F-φ diagnostics (impostor-leg share under the
   twin cell) — the direct answer to the stage-0 claim.
2. Per-event paired Δln L_cat read (rule 10/[A2]) — distribution, not just the mean.
3. Score-at-truth (A12) for the matched and full channels under F-φ.
4. Rail read: r_low count and floor-node mass under F-φ vs banked (expected NULL — the rail is
   photo-z territory; a rail change would be a surprise finding, registered as such).
5. K-flat kill test: |mean_h(K-flat) − mean_h(coded)| ≤ the pilot noise floor if the z-slope
   is the mechanism.

**Axis-leverage statement (A17, pre-data):** GATE-LEV must show the predicted |Δ per seed| is
≥ 5× the band resolution (SEM_paired-scale) BEFORE the pilot runs; if the instrument predicts
sub-resolvable leverage, execution STOPS at LEV and the thread returns with a re-design (the
O4 lesson applied prospectively). The completion-leg twin's measured leverage (+0.125/seed on
the matched channel) is the order-of-magnitude prior; the catalogue leg is active in ~74% of
events and carries −0.079 of the headline.

## 6. A10 — invariants and structural blindness

**Invariants (held fixed in every arm):** completion-leg cell = `off` (the banked baseline of
record; the fused-basis replication is a registered follow-up) · with-BH catalogue numerator =
coded · candidate membership and ball geometry · photo-z kernels/σ · σ_frac columns · the
φ-survival table construction · H grid (41-node + wing) · `PRODUCTION_FLAGS` otherwise
verbatim.

**Structural blindness:** (i) errors COMMON to both conventions — above all an error inside
`precompute_phi_marginal_survival` itself (the O6/O7 disclosed blind spot, now shared by three
instruments); (ii) this design cannot adjudicate data-deterministic vs latent-thresholded
detection for the REAL universe — it measures generator-matched consistency for this venue;
(iii) the p_gal→φ proxy; (iv) the LOO/density-form axes (excluded by design).

## 7. Falsifiers (A19)

- TWIN-MATERIAL is falsified by (i) K-flat NOT nulling (the effect would not be the z-slope
  mechanism claimed); (ii) GATE E-P3 failing on re-audit (engagement vacuity); (iii) a
  zero-compute audit showing the inserted factor is not the same object β_G_φ integrates.
- TWIN-IMMATERIAL is falsified by the paired per-event read (secondary 2) exhibiting opposing
  sub-population shifts that cancel in the mean ([A2] — the read is registered, so the
  cancellation cannot hide).
- The banked verdict, either way, carries its provisional status until the author's stage-5
  ruling (row #160: rulings stay author-gated).

## 8. Costing line (A6/A17)

LEV < 15 min, < 4 GB. P + PILOT + K-flat: 4 × `evaluate()` ≈ 2 h local sequential, ~9 GB each.
F-φ: 12 × ~30 min ≈ 6 CPU-h — local 2-wide (~18 GB peak, 30 GB box) ≈ 3 h wall; cluster
fallback via preflight if the box is contended. Total ≤ ~10 CPU-h of the 50 CPU-h grant.

*(Committed before the branch is created; LEV values and frozen band numbers appended below
pre-fleet; VERDICT appended when the committed scorer reports; A20 review before any banking.)*

## AMENDMENT 1 (2026-08-22, pre-execution; A21 — registration defect found during implementation, NO arm has run)

**Defect:** §4's GATE E-P3 required runtime engagement evidence from BOTH the scalar and batch
host-likelihood paths. Implementation-time verification shows production has **no runtime call
site** of the scalar `single_host_likelihood` (grep: comments/docstrings only; the batch
`_starmap_host_batches → single_host_likelihood_batch` is the sole dispatch in `evaluate()`).
The scalar engagement log therefore CANNOT fire in any real run, and the gate as registered
was unsatisfiable — the inverse of the A13 incident class (a gate that must fail regardless of
correctness), caught before execution.

**Amended GATE E-P3 (replaces §4's, before any data exist):**
(a) ≥10% of catalogue-bearing events' `L_cat_no_bh` move ≥1e-6 relative vs banked under
`"phi"`; (b) the BATCH-path engagement log line (`[P3-IMP] … ENGAGED in the batch host path`)
present in every `"phi"` run's log; (c) the scalar twin carries the same factor by CODE AUDIT
(same expression, same table input — auditable at
`bayesian_statistics.py` `numerator_integrant_without_bh_mass` + the delta-kernel branch),
explicitly in scope for the A20 review; (d) the implementation note that the factor is applied
in BOTH `_starmap_host_batches` calls — the with-BH host batch's `r[0]` is also a no-BH
numerator feeding `L_cat_no_bh` (the caller's `all_results_without_bh` concatenation) — is
part of the audit surface (a one-call engagement would have been the silent-subset A13
failure; found and prevented at implementation).

No other section changes; bands, arms, statistics, invariants stand as registered.

## LEV — AXIS-LEVERAGE VALUES (2026-08-22, appended pre-pilot; no `evaluate()` has run for [P3-IMP])

`p3_leverage_estimate.py` (zero-`evaluate()`, 103 s local — within the costing line), with the
two DISCLOSED substitutions carried in its output (`flagged_substitutions`): effective event-z
in place of per-candidate z_obs (the real ball search is not reimplemented outside
`evaluate()`), and a product-of-means in place of a paired per-event average (the redrawn
200-event set cannot be index-aligned to the banked post-filter 174-event set without
`evaluate()`). Both are order-of-magnitude-safe for a gate whose threshold check has this
margin:

- Per-seed predicted score-scale delta: +0.095 … +0.157 (12/12 positive);
  fleet mean **+0.1224, sd 0.0185**.
- **Gate check: ratio |fleet mean| / threshold (5 × 0.004) = 6.12; min per-seed ratio 4.75,
  max 7.85 ⇒ GATE-LEV PASS** — the registered axis can reach every band; execution proceeds
  to arm P and the pilot.
- Note for the verdict's context (NARRATIVE-HYPOTHESIS until measured): the predicted
  score-scale leverage (+0.122) is the same order as the completion-leg twin's measured
  per-seed shift (+0.125) — consistent with the one-arrangement ([P2]+[P3]) reading; the
  measured primary remains Δmean_h, not this estimate.

## AMENDMENT 2 (2026-08-22 ~03:30; GATE R-P3 FIRED, diagnosed, driver fixed — gate re-runs before anything else)

**The gate fired as designed.** Arm P's first run: B_num bit-exact (0.0) but L_cat_no_bh max
rel 1.0 with 738 rows fresh-zero-where-banked-nonzero (18 events fully zeroed at all h, 110
partially reduced, 56 exact) and a key mismatch (41 vs 46 h rows). Execution STOPPED per §4.

**Diagnosis [MEASURED]:** the driver deviated from the canonical bsel producer in ONE input —
`h_values=H_GRID_41` where the canonical `run_arm_seed` uses `H_GRID_FULL`. The mirror driver
widens the h-prior lower limit to `min(h_values)` (0.6 vs the canonical 0.5), which narrows the
candidate z-window at the low-z end and DROPS low-z catalogue candidates — a strict membership
subset, exactly the observed signature (the completion leg, which has no candidate membership,
stayed bit-exact). **Discriminator [MEASURED]:** the canonical producer re-run on unmodified
main reproduces the banked seed-900101 CSV at max rel **1.3e-14** on L_cat_no_bh (B_num 0.0,
8004/8004 keys) — the banked baseline is sound; no upstream drift; the twin cell is not
implicated by this failure.

**Fix (driver-only, pre-measurement):** `p3_twin_test.py` evaluates over `H_GRID_FULL`
(canonical), scores on `H_GRID_41` (unchanged — the same split the canonical uses). Also added:
`--seeds` fleet subset for a disclosed 2-wide operational split. **GATE R-P3 re-runs from
scratch before the pilot; no band, arm, or statistic changes.** The h-grid → h-prior →
candidate-window coupling joins the A17 checklist as a portability hazard (an "evaluation grid"
input that is secretly a SELECTION input).

## AMENDMENT 3 (2026-08-22 ~05:25; pre-scoring — E-P3(b)'s evidence channel unsatisfiable for a mechanics reason, second instance of the AMENDMENT-1 class)

The batch-path engagement log (`_p3_engagement_log_once("batch")`) is emitted inside FORKSERVER
WORKER processes, which do not inherit the parent's `_capture_root_log` FileHandler — the line
cannot appear in any captured run log regardless of genuine engagement (seed 900101: engagement
real — the counterfactual init line is present in the parent log, and `L_cat_no_bh` moves
fleet-wide, E-P3(a)). E-P3(b) is REPLACED, before any scoring, by: (b′) the parent-process
counterfactual init line (`catalogue_numerator_survival='phi'`) present in every `"phi"` log —
already cell-specific at source; the batch-dispatch reach remains covered by (a)'s data-level
movement plus AMENDMENT 1(c)'s code audit, explicitly in the A20 review's scope. No band, arm,
or statistic changes. (Two gate-evidence mechanics defects in one registration — logging
observability joins the A17 checklist: an engagement gate's evidence channel must be verified
OBSERVABLE in the harness before registration.)

## BAND FREEZE (2026-08-22 ~05:40, appended post-pilot, PRE-FLEET; formulas as registered in §5)

Pilot (seeds 900101/900102 under `"phi"`, GATE R-P3 already green): Δ_s = **+0.019497,
+0.027370**; realized paired σ̂ (n=2 proxy) = **0.005567** ⇒ SEM_paired(12) forecast ≈ 0.00161
⇒ 3·SEM = 0.0048 < 0.02, so the registered max() resolves to the anchor:

- **TWIN-MATERIAL iff |Δ̄(12)| > 0.02** (frozen)
- **TWIN-IMMATERIAL iff |Δ̄(12)| ≤ 0.01 AND SEM_paired(12) ≤ 0.004** (frozen)
- **REPORT-BOUND otherwise** (first-class)

Realized-scatter re-check (A17) at scoring uses the fleet's own 12-seed paired SEM, not this
n=2 proxy. Costing realized: ~30 min/seed wall (46-node grid), ~9 GB — fleet 2-wide ≈ 2.5 h.

## VERDICT (2026-08-22 ~09:00; all gates PASS; band fired: REPORT-BOUND — banked PENDING the A20 review)

**Gates:** R-P3 PASS (post-AMENDMENT-2 re-run: L_cat 1.3e-14, B_num bit-exact, 8004/8004 keys,
1810 s genuine); E-P3 PASS (AMENDMENTS 1+3 form: pooled engagement fraction **1.00**, per-seed
1.00 ×12; parent init line ×12; code audits (c)/(d)); L-P3 PASS (counterfactual line in every φ
log, absent in P).

**Primary [MEASURED]: Δ̄(12) = +0.019257 ± 0.003704 (paired SEM; sd 0.012829; 12/12 positive;
5.2σ from zero).** Frozen bands: 0.019257 < 0.02 ⇒ **REPORT-BOUND** (not TWIN-MATERIAL by
0.0007; not IMMATERIAL — both its conditions fail). Reported bound, no label; the
production-fork question returns to the author with this number, per the registered branch.

**K-flat [MEASURED, reported-only — the kill test fired INFORMATIVELY]:** Δ(phi_flat, 900101) =
**+0.0431** vs Δ(phi, 900101) = +0.0195. The constant-table arm moves the posterior MORE than
the real table: the twin insertion's effect decomposes as **level ⊕ slope** — the LEVEL
component (catalogue-leg suppression rebalancing the mixture toward completion, +0.043)
dominates, and the z-SLOPE component acts OPPOSITE (net ≈ −0.024 on this seed: S̄_φ is larger
at low z, relatively up-weighting the low-z impostors within the leg). The §7 falsifier
consequence: a TWIN-MATERIAL claim "owned by the z-slope mechanism" would have been falsified;
REPORT-BOUND carries no such claim. **Promoted to the author as the α-pairing sub-convention
question:** the registered unnormalized insertion changes the mixture LEVEL alongside the shape
(the double-counting risk the registration's structural-blindness section adjoined); the
shape-only (per-event-normalized) sub-convention is the registered follow-up arm that would
isolate the slope.

**Secondaries [REPORTED]:** (1) impostor decomposition under φ: pure−full ≈ +0.076 per seed —
the impostor drag PERSISTS at ~96%-of-coded magnitude; the twin recovers +0.019 of the
headline, not the drag itself. (2) per-event paired Δln L_cat: mean −0.0123, median ≈ 0, min
−0.250, max 0.000 — pure suppression, heterogeneous, no cancellation hiding ([A2]).
(3) score-at-truth under φ: full ≈ −0.197 — the full-channel violation stands. (4) rail read:
11/12 r_low under φ (banked 12/12) — NULL as expected, one marginal un-rail.

**Costing realized:** 17 × `evaluate()` (P re-run ×2, pilot 2, fleet 10, K-flat 1, + the
diagnosis discriminator ×2) ≈ **9 CPU-h** of the 50 CPU-h grant; all local.

## A20 REVIEW AMENDMENTS 4–7 (2026-08-22 ~09:40; review banked verbatim in `A20_REVIEW_P3_TWIN_20260822.md`, BANK-WITH-AMENDMENTS, zero FATAL; every decisive number orchestrator-re-derived before this block was written)

**AMENDMENT 4 — primary re-referenced (scoring-convention mismatch found post-verdict).** The
primary as scored subtracted the banked JSONs' `mean_h` (produced pre-row-#145 with the
superseded `legacy_gradient` weights) from a φ arm scored with the corrected `trapezoid`
default — a systematic +0.003733 offset. **The primary of record, re-referenced to the
convention the −0.1083 headline is quoted on (banked trapezoid fleet bias −0.108302):
Δ̄(12) = +0.015524 ± 0.003657 (paired SEM; sd 0.012669; 12/12 positive; 4.24σ) — recovering
14.3% of the headline bias.** Cross-check legacy/legacy: +0.015769 ± 0.003854. **REPORT-BOUND
is unchanged under every consistent pairing.** The superseded +0.019257 ± 0.003704 is
WITHDRAWN. Orchestrator re-derivation: exact match on all three pairings.
**New A17 rule:** a paired counterfactual re-derives its baseline statistic from the baseline
CSV through the arm's own scoring path and GATES that recomputation against the banked JSON
before any Δ is formed — a banked summary field is a different object from a re-scored column.

**AMENDMENT 5 — secondaries corrected from single-seed to fleet.** (1) impostor decomposition
under φ: pure−full = **+0.06366 ± 0.0090** fleet mean (sd 0.0312, range 0.0248–0.1128) =
**80.6%** of the coded −0.079 (not "+0.076 / ~96%"; the 4.5× per-seed spread admits no single
number). (3) score-at-truth under φ, full channel fleet mean **−0.21145** (not −0.197).
Directions unchanged: the drag persists; the full-channel violation stands.

**AMENDMENT 6 — materiality framing withdrawn.** "Not TWIN-MATERIAL by 0.0007" is withdrawn;
under the corrected reference the gap is 0.00448 and 3·SEM (0.0110) never approaches the 0.02
anchor. REPORT-BOUND is reported as a bound, no distance-to-anchor commentary.

**AMENDMENT 7 — K-flat qualified; evidence banked.** (a) The K-flat constant is the unweighted
grid-mean (c ≈ 0.270–0.280 across h) vs an operative per-event median 0.353 (range 0.017–0.94):
the level/slope partition is CONDITIONED on that constant (c=1 ⇒ all "slope"), and c(h) is
itself h-sloped (+3.6% across the grid) — the "slope" residual is per-event/per-host
heterogeneity, not the factor's h-tilt. Re-referenced: **level +0.039283, slope −0.023639**
(slope offset-invariant, stands). The α-pairing sub-convention question to the author stands,
strengthened. (b) Discriminator artifacts banked at `p3_work/rp3_discriminator/`; the FIRED
gate's work root was overwritten pre-mandate — loss disclosed in
`FIRED_GATE_EVIDENCE_NOTE.txt`. (c) The R-P3 `fallback_justification` (run-to-run
multiprocessing float order) is factually wrong — two independent runs on different commits
agreed to 17 digits; restated as an unexplained deterministic residual vs the banked CSV,
bounded 1.3e-14, and the cross-run bit-identity recorded as the strongest evidence for the
off-default byte-identity.

## SHAPE-ONLY ARM — REGISTRATION (2026-08-22 morning; author-approved row #163 item 3; pre-data; A21/A22 in force)

**Question:** isolated from the mixture-level component, does the S̄_φ z/h-SHAPE alone move the
headline — i.e. what does the derivation-coherent per-host reweighting do when the catalogue
leg's per-event level is held at its coded value at the truth anchor?

**Construction (registered exactly; ZERO-`evaluate()` — a rescore of banked columns):** per
event e, seed s, node h:

    L_cat_shape(s,e,h) ≡ L_cat_phi(s,e,h) · [ L_cat_off(s,e,h_ref) / L_cat_phi(s,e,h_ref) ],
    h_ref = 0.73 (the truth anchor; registered choice)

with `L_cat_phi` from the banked φ fleet CSVs (`p3_work/phi_*_work/.../event_likelihoods.csv`)
and `L_cat_off` from the banked baseline CSVs. The event's mixture is reassembled via the
verified identity (`decompose_impostor_leg` GATE I):
`combined_shape = combined_off − cat_term_off + cat_term_shape` with
`cat_term_X = (α_G_φ/r_Malm)·L_cat_X/D̃_φ` (h-only columns identical across cells, O-series
T-gates). The per-seed posterior and `mean_h` are computed by writing the reconstructed
`combined_no_bh` column into a patched CSV and calling the COMMITTED `compute_seed_statistics`
(trapezoid — the convention of record per A20 amendment 4), never a reimplementation.
Events with `L_cat_phi(h_ref) = 0` (dark/no-candidate events) take factor ≡ cat_term ≡ 0 —
identical in both cells by construction; registered edge case.

**Arms:** SHAPE (above, 12 seeds) + REPORTED-ONLY sensitivity: h_ref ∈ {0.70, 0.76} re-runs of
the same rescore (zero cost) — the anchor-choice leverage is measured, not assumed (K-flat's
constant-conditioning lesson, A20 amendment 7a).

**Gates (fail ⇒ VOID):**
- **GATE I-S:** the mixture identity reproduces `combined_no_bh` on BOTH input CSV sets to
  ≤ 2e-6 relative before any reconstruction.
- **GATE N-S:** `L_cat_shape(h_ref) = L_cat_off(h_ref)` exactly (≤1e-12; true by construction —
  the assert is the anti-bug gate).
- **GATE B-S (A17(e), first application):** the baseline `mean_h` re-derived from the banked
  CSVs through `compute_seed_statistics` (trapezoid) matches the A20-amendment-4 orchestrator
  re-derivation values per seed (≤1e-9) BEFORE any Δ is formed.
- **A22 stamp:** the instrument records git commit + dirty state at START.

**Primary:** `Δ̄_shape(12) = mean_s[mean_h(shape,s) − mean_h(banked trapezoid,s)]`, paired SEM.
**Decomposition report (registered):** `Δ̄_level ≡ Δ̄_phi(re-referenced, +0.015524) − Δ̄_shape`,
reported alongside — the two sub-conventions' split at the fleet level, superseding K-flat's
constant-conditioned version.

**Bands (frozen NOW — the paired scatter is already measured on this exact fleet):**
SHAPE-MATERIAL iff |Δ̄_shape| > 0.02 · SHAPE-NULL iff |Δ̄_shape| ≤ 0.01 AND SEM_paired ≤ 0.004 ·
REPORT-BOUND otherwise. **Two-sided by construction** — K-flat's constant-conditioned slope
(−0.024) makes a NEGATIVE (bias-worsening) Δ̄_shape a live branch, registered as such
[NARRATIVE-HYPOTHESIS until measured].

**Axis leverage (A17):** the φ arm moved every seed (+0.0155 fleet, per-seed spread
0.008–0.057 as-scored); the shape rescore differs from φ only by the per-event constants, whose
K-flat analog carried +0.039 — the axis trivially resolves both bands. **Costing (A6/A17):**
zero `evaluate()`; < 5 min wall, < 2 GB; local.

**A19 falsifiers:** SHAPE-MATERIAL/NULL are falsified by the h_ref sensitivity arms moving the
verdict across a band edge (anchor-conditioning — the K-flat lesson applied prospectively);
any gate failure voids. **A10:** everything held fixed except the per-event level constant;
structural blindness: (i) inherits every φ-run blindness (S̄_φ table common-mode); (ii) the
shape/level split is h_ref-anchored — a different anchor family (e.g. h-averaged) is a
different registered choice; (iii) zero-evaluate means no new engagement evidence — the φ
CSVs' provenance carries it.

*(Instrument `p3_shape_rescore.py` committed before it runs; VERDICT + A20 review before any
banking.)*
