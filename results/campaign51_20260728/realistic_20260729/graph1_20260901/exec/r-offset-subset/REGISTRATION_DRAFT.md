# r-offset-subset — REGISTRATION DRAFT: what distinguishes the 3–6 % of events that carry the −0.064 2D offset?

Date: 2026-09-03 (night). Node: r-offset-subset — the **Graph 2 seed** (morning docket R1: "route the
catalogue-hosted-class localization (F1) + the 3–6 % subset (F6) into ONE follow-up register node").
**DRAFT — PROPOSED THROUGHOUT, nothing frozen until the author rules.** Author of record for every
scientific decision: Jasper Seehofer. Band + launch ratification returns as fresh RULE
**d-offset-subset-register**. max_revisions 2; cost cap ≤ 2 CPU-h local, zero cluster (both
ORCHESTRATOR-DERIVED). Research-cycle stages 0–2 applied; stage-1 forecast in `INFORMATION_FORECAST.md`.
Append-only after commit. The registration author has NOT computed the registered statistic (§10).

## 0. Claim intake (stage 0) — provenance of the object

| hop | what the record says | tag |
|---|---|---|
| row #302 | re-baseline iiib 2D: map 0.665 / **mean_h 0.665854** / σ_h 0.018475; 1D 0.665 / 0.666987 / 0.017526; joint_r1 2D 0.667127, 1D 0.667032 | [DOC] |
| row #342 (3), `exec/rd-2d-bootstrap-jackknife/READ_RECORD.md` §1–§4 | targets reproduced to 4.8e-7 under the **frozen T0 convention** (gradient-trapezoid weights, physics floor); minimal directional-influence subset **k = 82 (5.164 %)** iiib 2D, 94 (5.919 %) iiib 1D, 72 (4.534 %) / 46 (2.897 %) joint_r1 2D/1D; bootstrap SE/σ_h 0.893/0.995/0.682/1.033; no rail; **NOT diffuse** (>10 % never reached) | [DOC], verdict-free |
| row #342 chair reading | "carried by ~3–6 % of events (≈50–90 of 1588) … the jackknife influence ranking is the natural next lead (which events, which class)" | [DOC], flagged |
| row #335 | S3 harness (N=200, cell S): score-zero fails ONLY in the **catalogue-hosted** class (Z 9.76/7.15) — dark inside (1.26/1.76). Harness venue, TRUTH class label | [DOC] |
| row #337 | the dark-class criterion `L_cat_no_bh == 0` is numerically fragile (boundary crossings 1e-110…2.3e-8); a relative-threshold label proposed | [DOC] |
| row #344 | cone loss immaterial: 10 OUT events, φ 0.4 %, **leave-out of the 10 moves mean_h −0.0049 (toward the rail's far side, i.e. the OUT events pull TOWARD truth)**; IN class heavy-tailed (plain/MAD SD = 8.5; events 889 s_e +52, 474 s_e −24) | [DOC] — **partial pre-read of one covariate, §10** |
| row #347 | completion residual: 74 % of the dark-class matched-channel score is production-only (ρ = 0.257); the "what is it" question routed to d-residual-attribution | [DOC] |
| **R8 build, chair amendment 2026-09-04 00:45** (`exec/b-dark-class-relative/{BUILD_RECORD.md, CHAIR_NOTE_20260904.md, dark_class.py}`) | relative criterion `L_cat_no_bh/combined_no_bh < 1e-6` changes the post-flip iiib split from 606/982 (exact-zero dark/hosted) to **1241/347** — 635 "hosted" events carry a catalogue-leg weight < 1e-6; joint_r1 493/1095 → 967/621; threshold is a margin call (max moved ratio 9.75e-7, no natural gap). Returns as [RULE] R14 ("what does catalogue-hosted MEAN"); this arm must carry BOTH labels and the continuous fraction so R14 gets a measurement | [DOC] |
| pre-flip production (row #137, ledger line 1347; `docs/RETROSPECTIVE_D1_20260820.md`) | base tilt localized to the DARK class, high-z (score −0.635, 37σ; ≈0 below z≈0.4, −1.08 by z≈0.9) | [DOC], STALE under [A11] (pre-flip channel) — forecast input only |

**Claim registered (conjecture):** c-offset-subset-covariate — "the k=82 events carrying the iiib 2D offset are
separable from the bulk by at least one registered covariate, and removing the covariate-defined stratum
moves mean_h by ≥ T_mat toward truth." Counter-claim: c-offset-diffuse-in-covariates.
`Refute by:` (rule 3) the family of separation tests in §4 on the banked CSVs — zero compute; a null on
every covariate at the registered band refutes the claim outright.

**Exoneration check (both layers, MECHANISM-grepped, memory `rule1-exoneration-check-insufficient`):**
`CLAIM_2D_BIAS_20260730.md:721-757` and `BIAS_HISTORY_LEDGER.md:127-171` read entry by entry for the
mechanism "an identifiable event SUBSET carries the offset / a covariate-defined stratum owns the bias".
Nearest entries: (a) ⚠14 **spec-z-subset rescue — refuted (#42)**: asks whether a subset carries the H₀
SHAPE (information), on the pre-flip 40-event venue; this arm asks whether a subset carries the OFFSET on
the post-flip 1588-event re-baseline — a different object; the spec-z flag is deliberately NOT a registered
covariate here. (b) "candidate-window membership (exact removal moves MAP 0.81→0.82, wrong sign)": an
estimator-change exoneration, not a stratification; C7 (candidate count) is a covariate, not a membership
change — no re-litigation. (c) EXP-40 "top-decile carries 25.5 % of the host sum — 2D excess carried
broadly" (deep seed1000 venue, `exp40-mechanism.md:46`): prior EVIDENCE against subset concentration on a
DIFFERENT venue and pre-fusion estimator — venue-scoped by the standing rule; disclosed as a forecast
input, not a bar. (d) ⚠13 information starvation — not this mechanism; no verdict here may use the word.
**Not exonerated.** Per the memory rule this single-agent check goes to the adversarial design-gate verifier
as a decisive claim. R0 sweep: no new literature row needed (Gray 2020 / MFG 2019 rows already registered).

## 1. Populations and data of record (pins — STOP on mismatch)

| object | path (repo-relative) | pin |
|---|---|---|
| production CRB (truth labels, sky, M, d_L) | `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv` | md5 `9a1f2a14384a9281c97ca3be312ddaab`; 1590 rows; scored set = event_idx {0..1589} − {1203, 1356} |
| **g-c0-baseline pin** iiib re-baseline CSV | `graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv` | md5 `8e6a2c18dc5838dd1d52641589243672`; 65,108 rows = 41 h × 1588 |
| **g-c0-baseline pin** joint_r1 replicate CSV | `…/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv` | md5 `745954a0fdee5f10878fb5e622a06144` |
| run commit | `…_iiib/GIT_COMMIT_AT_RUN.txt` | `1ec9514dd1808c48b18c0792dce558e5bba0f116` (row #302 GREEN-AS-CORRECTED) |
| production h=0.73 log (candidate counts, P6) | `…_iiib/darksiren_emri_20260902_000633_h_0_73.log` | 1588 "Progess: detections" lines; 606 "no catalog results found"; 982 "possible hosts found"; P6 line 8622: 1D 66/76 |
| catalogue (host positions for C8) | `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` | md5 `c52c13b5cab61f6b3f04bbe202550969` |
| class-label helper (C2/C3) | `exec/b-dark-class-relative/dark_class.py` (`is_dark_exact`, `is_dark_relative`, `THRESHOLD = 1e-6`) | md5 `841225ac9206ff18bf0145a81cac3a54` — imported, never re-implemented |
| influence-ranking reference | `exec/rd-2d-bootstrap-jackknife/rd_2d_bootstrap_jackknife_output.json` | carries `full_sample`, `minimal_subset.minimal_k_events_removed`, `top10_events_by_abs_influence` — **NOT the full per-event vector** (must be recomputed, §3 phase B) |
| grid / convention | `darksiren_emri/validation/correspondence_1d.py:353` (`H_GRID_41`); `results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py` (`_moments`, `_physics_floor_apply`, `w = np.gradient(h_grid)`) | the frozen T0 convention of rows #302/#342 |

Population facts seen by the registration author (design inputs, not the statistic): in_catalog 76 / dark
1512 (truth); exact-zero 606 / 982 and relative 1241 / 347 (estimator labels, h = 0.73, iiib; joint_r1
493/1095 and 967/621); production M range 1.33e5–1.63e6 M☉; H_TRUE 0.73 (`constants.py:25`).

## 2. Definitions

**Influence.** For channel c ∈ {2D=`combined_with_bh`, 1D=`combined_no_bh`} and venue v ∈ {iiib, joint_r1}:
`infl_e = mean_h(full) − mean_h(full − e)` under the frozen T0 convention. Directional influence
`d_e = sign(0.73 − mean_h(full)) · (−infl_e)` — positive means removing e moves mean_h toward truth.
**High-influence subset S_{c,v}** = the first k events of the ranking by decreasing d_e, with k the
row-#342 minimal k for that (c,v): **k = 82 (iiib 2D, PRIMARY)**, 94 (iiib 1D), 72 (joint_r1 2D),
46 (joint_r1 1D). Bulk B = the remaining events. S is defined by the BANKED k, not re-derived — the
re-derivation is the byte-id anchor (§6 G-2), not a free parameter.

**Covariates (columns of the blind table; every one computable from the pinned inputs; frozen here).**

| id | name | definition (exact) | type | source | family |
|---|---|---|---|---|---|
| C1 | `in_catalog` | CRB `in_catalog` (truth: host in GLADE+; ≡ `host_galaxy_index ≥ 0`) | binary | CRB | yes |
| C2 | `hosted_exact` | NOT `is_dark_exact(L_cat_no_bh)` (`L_cat_no_bh == 0.0`) on the h = 0.73 row — the exact-zero label (a); expected 606 dark / 982 hosted (iiib) | binary | iiib CSV | yes |
| C3 | `hosted_rel` | NOT `is_dark_relative(L_cat_no_bh, combined_no_bh, threshold = 1e-6)` from `exec/b-dark-class-relative/dark_class.py` (md5 pinned §1) at the h = 0.73 row — the R8 relative label (b); expected split 1241 dark / 347 hosted (iiib) | binary | iiib CSV | yes |
| C3c | `log10_f_cat` | `f_cat,e = L_cat_no_bh / combined_no_bh` at h = 0.73, as log10; exact zeros (and combined = 0) set to a **censored floor −320** (below any representable ratio) and flagged; the continuous catalogue-leg fraction (c). Mann–Whitney is rank-based, so the floor value is immaterial as long as it is below every finite ratio (gate) | continuous | iiib CSV | yes |
| C4 | `z_gw` | `dist_to_redshift(luminosity_distance, h = 0.73)` (`physical_relations.py:447`) on the CRB d_L | continuous | CRB | yes |
| C5 | `log10_sky_area` | `log10(π · r_cone²)`, r_cone = `cone_radius(qS, δφφ, δθθ, δφθ, k = 1.5)` from `exec/r-cone-loss/cone_loss_reads.py:123` (the R-MKER-6 anchored function) | continuous | CRB | yes |
| C6 | `mass_window_retention` | n_2D / n_1D from the log line "possible hosts found n_1D/n_2D" (2D-vs-1D candidate retention, the host-mass-channel exposure); NaN for zero-candidate events (excluded from this test; n disclosed) | continuous | log | yes |
| C7 | `log10_n_cand_1d` | log10(1 + n_1D); n_1D = 0 for "no catalog results found" | continuous | log | yes |
| C8 | `cone_outside` | the r-cone-loss OUT flag (`build_census`, chord > radius at k=1.5); defined for in_catalog events only (n = 76; expected 10 OUT) | binary | CRB + catalogue | yes — **direction pre-read, §10** |
| C9 | `class_G` | **ALIAS of C1**: class G is the in-catalogue generative class (`PHYSICS_CHANGE_SBARPHI_20260827.md:223`); Option A′ (`2b657255`) touches `validation/correspondence_1d.py` only — a harness draw law, with no production-event footprint. No separate test; recorded so the mandate's covariate list is answered | — | — | no (alias) |
| C10 | `log10_M` | log10 of CRB `M` (the timeout-selection axis, row #342 L.1) | continuous | CRB | yes |
| C10b | `low_M_timeout_bins12` | M < 169 568.13 M☉ (rd-timeout `M_edges[2]`, `design_gate_bin_edges.json`); **run only if n ≥ 10 in the scored set**, else NOT-TESTED (n reported) | binary | CRB | yes (conditional) |
| C11 | `log10_snr` | log10 CRB `SNR` | continuous | CRB | **reported-only** (not in the mandate's list; outside the Holm family) |

Family size for multiplicity: **m = 11** (C1, C2, C3, C3c, C4–C8, C10, C10b) when C10b is testable, m = 10
otherwise. The three class labels (a) = C2, (b) = C3, (c) = C3c are deliberately all in the family: the
disposition MUST state which of (a)/(b)/(c) separates S (the R14 measurement, §5), even when the arm's
overall disposition is DIFFUSE. C1 (truth) is the fourth class axis; C1 vs C2/C3 disagreement (truth-hosted
events labelled dark and vice versa) is reported as a 2×2 table per label, reported-only.

## 3. Design — blind table FIRST, influence SECOND, join THIRD (three disjoint agents)

- **Phase A (b-offset-subset-table, sonnet/medium):** writes `covariate_table_blind.csv` — one row per scored
  event_idx (1588), columns exactly C1–C11 plus `event_idx`; NO likelihood-derived column other than C2/C3
  at the single h = 0.73 row; NO influence. Writes `sha256(covariate_table_blind.csv)` into `BUILD_RECORD.md`
  and runs the join gates G-3a/b (§6). Never opens the jackknife JSON's top-10 list.
- **Phase B (b-offset-subset-scorer, sonnet/medium, a different agent):** recomputes the full per-event
  influence vector for all four (c,v) under the frozen T0 convention; passes the byte-id anchors G-2; writes
  `influence_vectors.csv` (event_idx, d_e, rank, in_S per family). Never opens the covariate table.
- **Phase C (the reader, sonnet/medium, a third agent; verifier top-tier re-derives every decisive
  number):** verifies the table hash, joins on event_idx, runs §4, writes `offset_subset_result.json` +
  `READ_RECORD.md`. Design-gate reviewers check formula/computability on a synthetic 20-row table ONLY
  (memory `gate-reviewers-must-not-compute-registered-statistic`); they never touch the registered columns.

## 4. Registered statistics (primary family iiib 2D; three replicate families reported alongside)

**4.1 Separation (per covariate, S vs B).** Continuous: AUC of the covariate for S vs B (Mann–Whitney U /
(n_S · n_B); two-sided p from `scipy.stats.mannwhitneyu`). Binary: odds ratio OR = odds(TRUE | S) /
odds(TRUE | B) with Haldane 0.5 correction; two-sided Fisher exact p. C8 is tested inside the in_catalog
stratum only (S∩in_cat vs B∩in_cat). Multiplicity: **Holm step-down at family-wise α = 0.05 over m** (§2).
**Separation band (registered):** a covariate SEPARATES iff Holm-adjusted p < 0.05 AND effect outside the
practical-null band: |AUC − 0.5| ≥ **0.20** (continuous) or OR ∉ [1/3, 3] (binary). Both conditions; a
significant-but-small effect (e.g. AUC 0.58 at n_S = 82) is reported as WEAK, never as SEPARATES.
Secondary (reported-only, expected partly null): Spearman ρ between d_e and each continuous covariate over
all 1588 events; the class composition of S as raw counts (C1/C2/C3 tables).

**4.2 Materiality (leave-out re-marginalisation, frozen T0 convention, as in r-cone-loss §2).** For every
covariate that SEPARATES, the **stratum** is frozen by rule: binary → the level enriched in S; continuous →
the decile tail (10 % of the 1588 by rank, n = 159) on the side enriched in S (AUC > 0.5 ⇒ top decile,
AUC < 0.5 ⇒ bottom decile). `Δ_strat = mean_h(full − stratum) − mean_h(full)`, same channel and venue.
**Null:** 1000 uniformly random subsets of the same size (seed 20260904), same re-marginalisation;
report the two-sided empirical percentile of Δ_strat and the null's central 99 % interval.
**Material iff** Δ_strat ≥ **T_mat = 0.008** (toward truth: mean_h < 0.73 in every family, so positive)
AND Δ_strat lies outside the null's central 99 % interval. Reported alongside: the oracle Δ_S (leave-out
of S itself; ≥ |offset| − σ_h by construction) and the captured fraction Δ_strat/Δ_S; the MAP rail flag
for every re-marginalisation (g-censoring). Reweighting is NOT registered (it would be a fresh choice).

**4.3 Replicate consistency.** A covariate that SEPARATES in the primary family must SEPARATE (same
sign) in ≥ 2 of the 3 replicate families for the primary disposition to be SUBSET-IDENTIFIED; otherwise
INTERMEDIATE. Replicates are consistency reads, never a second verdict.

## 5. Bands (ORCHESTRATOR-DERIVED) and the three-valued disposition — every row returns as a fresh RULE

Power context (design only, no data touched): at n_S = 82 vs n_B = 1506, SE(AUC) ≈ 0.03 under the null,
so AUC = 0.70 is ≈ 6.5σ before Holm; for C1 (76 in_catalog of 1588 = 4.8 %) OR = 3 corresponds to ≈ 11
in_catalog members of S vs 3.9 expected — Fisher p ≈ 2e-3, survives Holm at m = 11. The arm is POWERED for
its bands on the primary family; the joint_r1 1D family (k = 46) is under-powered and is reported only.

| disposition | trigger | claim writeback | action |
|---|---|---|---|
| **SUBSET-IDENTIFIED** | ≥ 1 covariate SEPARATES (4.1) AND its stratum is MATERIAL (4.2) AND replicate-consistent (4.3) | c-offset-subset-covariate SUPPORTED with the named covariate(s), AUC/OR, Δ_strat, captured fraction | Graph 2 mechanism node on that covariate; the S3 revision-2 (docket R2) gets its "what to change"; d-residual-attribution receives the stratum as a candidate for the 74 % production-only part; fresh RULE |
| **DIFFUSE-IN-COVARIATES** | NO covariate SEPARATES in the primary family (kill criterion, verbatim from the mandate: "no single registered covariate separates the influence ranking from the bulk at the registered band") | c-offset-diffuse-in-covariates SUPPORTED at the bound: every registered covariate has \|AUC − 0.5\| < 0.20 / OR ∈ [1/3, 3] or fails Holm | q-offset-subset **SETTLED-BOUNDED**: the 3–6 % structure is a per-event likelihood-shape object not indexed by any registered covariate; R2 stays parked; the honest-bound paper framing inherits the bound; fresh RULE |
| **INTERMEDIATE** | any of: a covariate SEPARATES but no stratum is MATERIAL; MATERIAL but not replicate-consistent; C8 or C10b NOT-TESTED and no other covariate separates; primary 2D and 1D iiib families disagree in disposition | partial: the separating covariate(s) named with their non-material Δ_strat | fresh RULE: bank as-is, or one revision (≤ 2) with a finer stratum on the flagged covariate only — no new covariates may be added post hoc |
| **INSTRUMENT / NO-READ** | any §6 gate red | nothing banked | repair; no revision consumed |

**Mandatory class-label line (R14 measurement, every disposition):** the record states, for (a) C2, (b) C3,
(c) C3c separately: AUC/OR, Holm p, SEPARATES / WEAK / NULL, and (if separating) Δ_strat. Readings: only (c)
separates ⇒ "catalogue-hosted" is a continuous-weight notion, neither binary label indexes S; (b) but not
(a) ⇒ the 635 negligible-weight events are bulk and the materiality label is the right class; (a) but not
(b) ⇒ S sits among the 635 (support-only hosted events) — a NEW lead on its own; none ⇒ class is not the
axis. This line is evidence for R14, not the R14 ruling.

## 6. Gates

- **G-1 pins:** the four md5s and the commit in §1; STOP on mismatch (CLAUDE.md dataset-pinning rule).
- **G-2 byte-id anchors (phase B, b-offset-subset-scorer):** (i) iiib 2D mean_h = **0.6658540600** (row #342
  JSON `full_sample.mean_h`, |Δ| ≤ 1e-9) and iiib 1D 0.6669869414; (ii) directional minimal k = **82 / 94 / 72 /
  46** EXACTLY for the four families; (iii) the JSON `top10_events_by_abs_influence` (event_idx and values)
  reproduced to 1e-12 relative for all four families; (iv) k = 1588 endpoint = 0.73 to 1e-12; (v) 0 events
  physics-floor-excluded. Any miss = INSTRUMENT-DEFECT. (vi) C5 reuses `cone_radius`; the r-cone-loss G-2
  anchor (R-MKER-6 radius 1.4956979545757095e-03 ± 1e-15) is re-run by phase A as its own byte-id.
- **G-3 joins (phase A):** (a) log detection order → event_idx: the k-th "Progess: detections: k/1588" block is
  the k-th scored event_idx in ascending order (gaps {1203, 1356}); DECISIVE check: the 606 "no catalog results
  found" detections must map exactly onto the 606 rows with `L_cat_no_bh == 0` at h = 0.73 (set equality);
  (b) in_catalog count 76 = P6 denominator; OUT count = 10 = 76 − 66 (row #344); (c) f_cat ∈ [0, 1] for
  every row with combined_no_bh > 0. Any failure = INSTRUMENT-DEFECT.
- **G-4 blindness hash:** phase C refuses to run unless `sha256(covariate_table_blind.csv)` equals the value
  committed in `BUILD_RECORD.md` before phase B's first run (timestamps in the record).
- **g-population:** 1588 rows per h-node × 41 nodes, no G-EXT nodes; every table row joined (0 unmatched);
  n per binary level and n_NaN per covariate reported; C10b n ≥ 10 rule applied and disclosed.
- **g-precision:** full-precision columns only (`combined_*`, `L_cat_no_bh`); float64 log-sums; C3c floor
  −320 must lie below min(finite log10 f_cat) (gate); the C2/C3 counts must reproduce the R8 table
  (606/982, 1241/347 iiib) EXACTLY (gate); no 7-s.f. column (`D_tilde_phi`, `alpha_G_phi`) enters any covariate.
- **g-censoring (rail disclosure):** MAP position for the full sample, every stratum leave-out and every
  null draw; any MAP at 0.60/0.86 ⇒ that Δ is a BOUND, rail fraction reported; Δ_strat uses mean_h.

**Invariants ([A10]):** frozen T0 convention (gradient weights, physics floor; audited 2026-09-03 rows #302/
#342) · H_GRID_41 · h_true 0.73 · k = 1.5 sky cone (audited 2026-08-28) · the banked k per family ·
`dark_class.THRESHOLD` = 1e-6 (R8, a margin call — disclosed, NEVER re-tuned inside this arm) · decile =
10 % · seed 20260904 · the md5/commit pins. **Structural blindness:** the arm can only
find structure along the eleven registered axes; a subset defined by an unregistered property (e.g. a
specific impostor geometry, host photo-z error realisation, or a likelihood-shape feature) is invisible by
construction and lands in DIFFUSE-IN-COVARIATES — the bound says "not these axes", never "no structure".
Second blindness: S is a leave-one-out object; interaction effects (events influential only jointly) are
not captured by single-event influence.

## 7. Cost (ORCHESTRATOR-DERIVED, cap ≤ 2 CPU-h, zero cluster)

Full LOO jackknife: 4 families × 1588 × (41 × 1588) log-sums ≈ 4e8 flops. Null draws: ≤ 10 strata × 1000
draws × 4 families × 6.5e4 sums ≈ 2.6e9 flops — seconds. One catalogue load (≈ 55 s) for C8. Log parse
1588 blocks. **≈ 0.1 CPU-h total**; cap headroom 20×.

## 8. Launch block (zero fresh choices)

Build nodes (wait): `b-offset-subset-table` (phase A), `b-offset-subset-scorer` (phase B) — both sonnet/
medium, disjoint agents, each writes its BUILD_RECORD section; launch waits on G-1…G-4 GREEN and on the
author's RULE on §9. Reader (phase C) sonnet/medium; decisive verifier top-tier re-derives AUC/OR/p/Δ_strat.

    # from REPO ROOT — phase A (blind table)
    uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_table.py \
      --production-crb results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
      --production-run results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib \
      --dark-class-module results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/b-dark-class-relative/dark_class.py \
      --h-true 0.73 --sky-cone-k 1.5 --fcat-floor-log10 -320 --low-m-edge 169568.12917853205 \
      --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
      --csv-md5 8e6a2c18dc5838dd1d52641589243672 \
      --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/covariate_table_blind.csv
    # phase B (influence vectors + byte-id anchors)
    uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset/offset_subset_influence.py \
      --production-run  …/run_20260902_graph1_headrebaseline_iiib --replicate-run …/run_20260902_graph1_headrebaseline_joint_r1 \
      --reference-json results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/rd-2d-bootstrap-jackknife/rd_2d_bootstrap_jackknife_output.json \
      --k-iiib-2d 82 --k-iiib-1d 94 --k-jr1-2d 72 --k-jr1-1d 46 --h-true 0.73 \
      --csv-md5-iiib 8e6a2c18dc5838dd1d52641589243672 --csv-md5-jr1 745954a0fdee5f10878fb5e622a06144 \
      --out …/r-offset-subset/influence_vectors.csv
    # phase C (registered read; refuses without the G-4 hash)
    uv run python …/r-offset-subset/offset_subset_reads.py --table …/covariate_table_blind.csv --table-sha256 <from BUILD_RECORD> \
      --influence …/influence_vectors.csv --alpha 0.05 --auc-band 0.20 --or-band 3.0 --t-mat 0.008 \
      --decile 0.10 --null-draws 1000 --null-seed 20260904 --out …/r-offset-subset/offset_subset_result.json [--dry-run]

## 9. Open questions routed to d-offset-subset-register (fresh RULE)

1. Ratify the covariate set C1–C11 (incl. the three class labels (a)/(b)/(c) per the chair amendment) as the
   closed family (adding any later = a revision, never post hoc).
2. Ratify the bands: AUC ±0.20 / OR 3, Holm α 0.05, T_mat 0.008, decile 10 %, null 99 %, replicate 2-of-3.
3. Ratify the primary family (iiib 2D, k = 82) and the leave-out-only materiality convention.
4. Accept that C9 (class-G membership) is an alias of C1 on production (§2) — or name a distinct definition.
5. Accept the C8 pre-read disclosure (§10) — C8 stays in the family as a consistency check, flagged NOT BLIND.
6. Accept the three-agent phase design (§3) and the G-4 hash gate as the blindness mechanism.

## 10. Blindness status and leak inventory (binding)

**Blindness status:** the registered statistics (AUC/OR/Holm p per covariate, Δ_strat, null percentiles)
have NOT been computed by anyone; the full influence vector exists nowhere on disk (only the top-10 |infl|
per family and the minimal k are banked). Band thresholds were frozen in this draft, time-stamped
2026-09-03 night, before any build. **Leaks known to the registration author, disclosed:** (i) row #344:
the 10 OUT events pull TOWARD truth, so C8's direction (OUT anti-enriched in S) is effectively pre-read —
C8 is retained as a consistency check, flagged; (ii) row #344's two-outlier line names in_catalog events
889 (s_e +52, positive influence, hence NOT in S) and 474 (s_e −24, very likely in S) — two of 1588 event
labels partially known; (iii) the JSON top-10 |influence| lists event_idx 576/160/1176 (1D, negative) —
membership hints, no covariate attached; (iv) population counts (76/1512, 606/982) seen as design facts.
None of (i)–(iv) is a registered aggregate; (i) is the only one bearing on a covariate's verdict, and C8
alone cannot produce SUBSET-IDENTIFIED (it is anti-enriched, and its stratum of 10 was already measured
IMMATERIAL at −0.0049, row #344).

## 11. Design-gate self-check

1 Executability: every input on disk and pinned (§1); the log→event_idx join is NEW and gated (G-3a).
2 Stop rule: none needed (no generative run). 3 Population: G-3 + g-population. 4 Byte-pin: G-2 five-part
anchor. 5 Blindness: §3 phases + G-4 hash + §10 inventory. 6 Internal consistency: one primary family, three
replicates as consistency reads; separation and materiality are sequential, not alternative, conditions.

**What makes this un-launchable tonight:** (a) bands and covariate set are orchestrator-derived — RULE
pending (R14 itself is NOT pre-empted: the arm measures all three labels and rules on none); (b) two build scripts do not exist yet (phase A/B), and the phase-A log join has never been
exercised; (c) C8 is not blind (disclosed); (d) C9 is not a distinct covariate on production; (e) C10b may
be near-empty (production M ≥ 1.33e5, edge 1.70e5) — n unknown until phase A runs.
