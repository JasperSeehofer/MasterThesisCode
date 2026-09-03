# r-cone-loss — REGISTRATION DRAFT: the share of the absolute bias floor owned by cones that cannot contain the true host

Date: 2026-09-03 (evening). Node: r-cone-loss (Research Graph 1, Branch H, wave 3) — **DRAFT**.
Author of record for all scientific decisions: Jasper Seehofer.
Status: **PROPOSED THROUGHOUT — nothing here is frozen.** Authorization: charter row #290 decisions-table
row 10 (registration authoring) + docket `DECISION_DOCKET_WAVE3_20260903.md` item 2.1. Band + launch
ratification returns as fresh RULE **d-cone-register**; launch only under docket 2.2 (design gate GREEN;
cap ≤ 20 CPU-h). max_revisions 2 (ORCHESTRATOR-DERIVED, charter §1.8/§1.13). Research-cycle stages 0–2
applied; stage-1 forecast in `../r-completion-residual/INFORMATION_FORECAST.md`. Append-only after commit.

## 0. Provenance of "≈ 17 %" — VERDICT: SOUND, but venue- and definition-scoped

| hop | record | tag |
|---|---|---|
| charter §1.0/§1.8 "approx 17 % of localisation cones structurally unable to contain the true host (artifact section 09)" | board card `a8824799` §09 line 562 ("no registered arm yet") | [DOC] |
| **R-MKER-6 STAGE-0 CENSUS (`CLAIM_P3_MKER_20260826.md:928-947`)** | 2261 events (24-seed `p3_2d_fleet_20260825` bc arm), **380 outside, 0.1681**; per-seed 10.1–24.5 %; anchor seed 900121 event 20 chord 1.6746585172e-03 / radius 1.4956979546e-03 reproduced full-float before counting | [DOC], agent-run, anchor-gated |
| chair closed-form envelope (same entry) | a hard 1.5·√λ_max circle excludes between 13.4 % (1-D limit, 2Φ(−1.5)) and 32.5 % (isotropic Rayleigh tail) of true hosts ⇒ "consistent with the cone working AS DESIGNED" | [LOCAL] chair |
| [CMEM] A1 replicate (`fanout1_20260829/B2_1_CMEM_A1_RECORD.md`) | 380/2336 = 0.1627 on `p3_b0_work` (different fleet; the 380 is a coincidence, explicitly flagged) | [DOC] |
| [CMEM] verdict of record (row #220) | C-STRUCTURAL-ONLY; the outside-cone truth-likelihood deficit R2c NOT-DISTINGUISHED (p = 0.0358 vs α = 0.01; power ≈ 68 %) | [DOC] |
| **production pool (this session, 2026-09-03, stage-0 fact — DISCLOSED, see §4)** | `seed61000/prepared_cramer_rao_bounds.csv`: 76 in-catalogue events, all 76 with chord > 0 (median 1.24e-3, max 5.11e-2 rad; cone radius median 2.70e-3), **10/76 = 13.2 % outside at k = 1.5**; independently, the production log's P6 counter: "1D 66/76 hosts recovered/in-cat events seen (86.84 %)" (`retrieved/run_20260902_graph1_headrebaseline_iiib/darksiren_emri_20260902_000633_h_0_73.log:8622`) | [LOCAL] |

Findings: the 17 % is a genuine, anchor-gated census — but on the **mirror fleet**, and it counts sky
geometry only. On the production pool the equivalent number is **13.2 % (10/76)**, and the estimator's own
P6 counter agrees (10 hosts not recovered). What has NEVER been measured is the object the charter asks
for: the **bias-floor contribution** of those events. That is this arm's registered statistic.

## 1. Definition — what "cannot contain" means, and which part is structural

For an in-catalogue event (`host_galaxy_index ≥ 0`), the 1D candidate list is the intersection of
(handler.py:690-735; primer §1):

| factor | rule | h-dependent? | structural? | this arm's treatment |
|---|---|---|---|---|
| **sky cone** | chord(host, observed sky) > k·√λ_max(J Σ' Jᵀ), k = 1.5 (`_sky_cone_k`, bayesian_statistics.py:3659) | no | **YES** — set by the forward model's sky scatter; the designed-in Gaussian tail (envelope 13.4–32.5 %) | the OUT class |
| z-window | host z_g outside the window derived from d̂ and the h-bounds (`z_window_k` = 1.0) | yes | no — membership varies with h (the exonerated [WINDOW-MEMBERSHIP] axis; NOT re-opened) | counted at h_true only, as a disclosure: n_z = (76 − P6 recovered) − n_sky-OUT |
| mass window (2D channel only) | `mass_filter_k` = 1.5 linear | no | no — retention read row #247: the 10 lost hosts are NOT mass-window losses | 2D channel replicate carries the same OUT class |
| catalogue completeness | host not in GLADE+ at all (`host_galaxy_index = −1`) | — | not cone loss — the completion leg's business (Branch G) | excluded from the definition |

**Registered definition:** OUT ≡ {in-catalogue events whose true host is absent from the 1D candidate
list at h_true}, operationally the P6 counter's complement; the sky-cone census must account for it
(G-4 below). "Structural" = the sky component: h-independent, unfixable by any estimator consistency
change without widening k (which would re-open the exonerated membership/HB family — out of scope).

## 2. The estimator (production geometry + banked scores; zero compute)

Population of record: the production pool seed61000 (CRB md5 `9a1f2a14384a9281c97ca3be312ddaab`; 1590
rows, 1588 scored, event_idx gaps {1203, 1356}) scored under the post-flip re-baseline (commit `1ec9514d`,
g-c0-baseline GREEN-AS-CORRECTED, row #302), venues iiib (primary) and joint_r1 (replicate).
Catalogue: `reduced_galaxy_catalogue.csv` md5 `c52c13b5cab61f6b3f04bbe202550969`, loaded as
`GalaxyCatalogueHandler(1e4, 1e7, 1.5).reduced_galaxy_catalog.reset_index(drop=True)` (the frame
`host_galaxy_index` lives in; cmem_a1.py:103-110 line-for-line).

Per-event full score on the stencil (0.725, 0.735): s_e = Δ ln combined_no_bh / Δh (1D) and the same on
`combined_with_bh` (2D) — `b4_imp_stage1_forecast.py:136-143` convention. Classes: OUT (n_OUT, expected
10), IN (in-catalogue, recovered; expected 66), DARK (1512).

**Primary statistic — the bias-floor contribution (linear response):**

    Δh_cone,c  =  (1/I_c) · Σ_{e ∈ OUT} ( s_e,c − s̄_IN,c ),   c ∈ {1D, 2D}
    I_1D = 1/σ_h,1D² = 1/0.017526² = 3256 ;  I_2D = 1/0.018475² = 2930   (re-baseline iiib, row #302)
    φ_cone,c  =  Δh_cone,c / (mean_h,c − 0.73)   with  mean_h − 0.73 = −0.0630 (1D) / −0.0641 (2D)

s̄_IN is the paired comparand (rule 10, [A2]): the same class with the host inside the cone, so the
contribution is the EXCESS pull of cone loss, not the in-catalogue class's pull as a whole.
Uncertainty: SE(Δh) = SD_IN(s)·√(n_OUT + n_OUT²/n_IN) / I_c (SD from the 66 IN events); Z = Δh/SE.
**Cross-check (registered, must agree within 2·SE):** the exact leave-out counterfactual — the frozen
T0 scorer (`prod2d_closure_20260818/tier0_bootstrap_jackknife.py` convention; gradient-trapezoid weights,
physics floor) on the 1578 non-OUT events ⇒ Δmean_h,leave-out. Disagreement beyond 2·SE flags the linear
response as non-linear and the read is booked on the leave-out number with the flag.

**Harness replicate (zero compute; the CMEM-mechanism check):** the same OUT/IN split on the 67 post-flip
S3 cell-S universes (`b8_cal_harness_work_s4_postflip/seed9010NN_S/simulations/{prepared_cramer_rao_bounds,
diagnostics/event_likelihoods}.csv`; population n200-postflip; 843 catalogue-hosted events, expected
≈ 140 OUT). Statistics: f_OUT,harn with per-universe SE; Δs = s̄_OUT − s̄_IN with between-universe SE.
Reads whether the catalogue-hosted defect signature (+0.587 ± 0.064 per event, 67 universes; rd-s3-readout
Z 9.76) concentrates on OUT events — REPORTED to d-calibration; not verdict-bearing here.

## 3. Bands (ORCHESTRATOR-DERIVED) and mapping to the floor

- **Materiality** T_mat = 0.008 in h — the standing convention (rows #247/#280/#284). Derivation: it is the
  threshold every counterfactual delta on this board has been judged against since wave 2.
- **Share thresholds:** φ ≥ 0.5 ⇒ the cone loss OWNS the floor (majority); φ < 0.2 ⇒ it does not (the
  remainder ≥ 4× the cone part; 0.2 ≈ T_mat/|offset| = 0.13 rounded up to one-in-five).
- **Fraction band (harness replicate only; the production fraction is disclosed-seen, §4):** f_OUT within
  the closed-form envelope [13.4 %, 32.5 %] ⇒ AS-DESIGNED; outside ⇒ INSTRUMENT question (fresh RULE).
- Power: SD_IN(s) ≈ 0.68 (per-event score SD, harness) ⇒ SE(Δh_1D) ≈ 0.68·√(10 + 1.5)/3256 ≈ **0.0007**;
  T_mat is 11 SE away; φ = 0.2 (Δh = −0.0126) is 18 SE. False-fail at |Z| ≤ 3: 0.27 %.
- Forecast (stage 1, disclosed): C5's pre-flip in-catalogue impostor score −1.707/event × 10 ⇒ Σ ≈ −17
  nats/h ⇒ Δh_1D ≈ −0.005, φ ≈ 0.08 — IMMATERIAL predicted; the arm is designed to falsify that.

## 4. Disposition table (every row returns as a fresh RULE)

| disposition | trigger (1D primary; 2D reported alongside) | claim writeback | action |
|---|---|---|---|
| **IMMATERIAL-FLOOR-SHARE** | \|Δh_cone\| < 0.008 AND φ < 0.2 | c-residual-floor-consistent: the cone-loss component is bounded at φ + 3·SE/\|offset\|; "leading candidate for the absolute floor" DEMOTED | q-cone-loss SETTLED (kill criterion: "confirms the floor within its band ⇒ irreducible geometry, no fix") — with the bound |
| **CONE-OWNS-FLOOR** | \|Z\| > 3 AND φ ≥ 0.5 | c-residual-floor-consistent SUPPORTED as geometry | d-residual-attribution: the floor is geometric; no consistency fix pursued (charter) |
| **INTERMEDIATE** | \|Z\| > 3 AND 0.2 ≤ φ < 0.5; or \|Δh\| ≥ 0.008 with φ < 0.2; or 1D/2D disagree in disposition; or linear vs leave-out disagree > 2·SE | partial share quoted | fresh RULE: bank the share; a revision (≤ 2) only if the author wants the z-window component resolved |
| **INSTRUMENT / NO-READ** | G-1…G-4 red; g-population red | nothing banked | repair; no revision consumed |

**Blindness disclosure (binding):** the registration author computed the production geometric census
(10/76 OUT; chord/radius distribution) and read the P6 log line BEFORE writing this draft, as a
stage-0 design fact (it decides whether production is sky-scattered at all — it is). The fraction is
therefore NOT a blind read on production and is banked as context, never as a verdict. The primary
statistic (the OUT events' scores, Δh_cone, φ_cone, the leave-out delta, the harness replicate) has NOT
been read by anyone.

## 5. Gates

- **G-1 pins:** catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`; CRB md5 `9a1f2a14384a9281c97ca3be312ddaab`;
  `GIT_COMMIT_AT_RUN.txt` = `1ec9514d`; STOP on mismatch (CLAUDE.md pinning rule).
- **G-2 anchors (instrument byte-id):** reproduce R-MKER-6's anchor on `p3_2d_fleet_20260825/bc_900121_work`
  event 20 (chord 1.674660e-03 ± 5e-10, radius 1.4956979545757095e-03 ± 1e-15 — `cmem_reads.py:32,107-111`)
  AND CMEM-A1's anchor on `p3_b0_work` bc/900101/event 0 (0.0116656941007181 / 0.0359121946154451,
  `cmem_a1.py:67`). Both fleets are on disk. A miss = INSTRUMENT-DEFECT.
- **G-3 join:** event_idx = CRB row index; scored set = {0..1589} − {1203, 1356}; in-catalogue count = 76 =
  the P6 denominator; the P6 numerator 66 must equal n_IN (the log line quoted verbatim in the record).
- **G-4 scatter law (production is sky-scattered by the forward model, not by a frame artefact):** the
  sky offsets' Mahalanobis² under the row's own J Σ' Jᵀ must be χ²₂-distributed (KS at α = 0.05 on 76
  events) AND f_OUT at k = 1.5 must sit inside [13.4 %, 32.5 %]. A failure means the offsets are not the
  designed Fisher tail (e.g. a frame translation — cf. `resolve_host_recovery_position`, handler.py:853)
  ⇒ INSTRUMENT-DEFECT, STOP, fresh RULE. Open audit item: which of (d_L, qS, phiS) the production pool
  scatters is set in `datamodels/detection.py:161-178` (`convert_to_best_guess_parameters`) — the
  record must quote the resolved draw, because the B8 design's "production is truth-centred" (§2.3
  cell-T convention) is contradicted in the sky by the 76/76 nonzero chords found here.
- **g-population:** harness 0 mixed rows (`--population 200`, seeds 901000–901066); production single pool.
- **g-censoring:** the production 1D MAP is 0.665 (interior); the offset denominators use mean_h;
  the harness replicate carries the S3 rail disclosure (14.9 % / 20.9 % upper-rail) on any h-space quote;
  the score statistics themselves are grid-interior (stencil).
- **g-precision:** scores from full-precision columns (`combined_*`); no 7-s.f. column enters.

Invariants ([A10]): k = 1.5 sky cone (audited 2026-08-28, R-MKER-6) · z_window_k = 1.0, theta_zwindow off
(checkpoint resolved flags) · mass_filter linear k = 1.5 (row #229) · the stencil · both md5 pins · the
frozen T0 scorer. **Structural blindness:** the arm measures the cone-loss pull as the data realise it;
it cannot detect a cone-loss effect that is exactly compensated by the completion-leg partition
([CMEM]'s rerouting hypothesis) — that is what the harness Δs replicate probes, and only as a report.

## 6. Cost against the ≤ 20 CPU-h cap (ORCHESTRATOR-DERIVED)

Zero cluster. Local: one catalogue load (55 s, measured 2026-09-03) + 1 production CSV (65,108 rows) ×
2 venues + 67 harness CSVs (7216 rows) + 67 harness CRBs + two anchor fleets' CRBs ⇒ **≈ 0.1 CPU-h**.
No optional cell: the population is the production pool (there is exactly one) and the harness
replicate is banked. Cap headroom: 200×.

## 7. Launch block (zero fresh choices)

**Build node (waits):** `b-cone-scorer` (sonnet/medium) writes
`graph1_20260901/exec/r-cone-loss/cone_loss_reads.py`, reusing `cmem_a1.py:85-160` (cone_radius,
build_census) and the frozen T0 helper; builder runs ONLY `--dry-run` (G-1…G-4 + census, no scores);
a DIFFERENT agent runs the statistic. Launch waits on G-2 anchors GREEN.

    # from REPO ROOT
    uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_loss_reads.py \
      --production-crb results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
      --production-run results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib \
      --replicate-run results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1 \
      --harness-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip --population 200 \
      --anchor-fleet-mker results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825 \
      --anchor-fleet-cmem results/campaign51_20260728/realistic_20260729/p3_b0_work \
      --sky-cone-k 1.5 --h-lo 0.725 --h-hi 0.735 --h-true 0.73 \
      --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
      --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_loss_result.json \
      [--dry-run]

## 8. Design-gate self-check (six checks)

1 Executability: all inputs on disk (verified paths above); the handler load is the only heavy step.
2 Stop rule: none needed (no generative run). 3 Population: G-3 + g-population. 4 Byte-pin: G-2 double
anchor. 5 Blindness: §4 disclosure — fraction seen, statistic unread. 6 Internal consistency: 1D primary,
2D reported; linear-vs-leave-out agreement is a registered cross-check, not a second verdict.

## 9. Open questions routed to d-cone-register (fresh RULE)

1. Ratify the definition (§1): OUT = the P6 complement, sky component structural, z-window counted only.
2. Ratify the bands (§3): T_mat 0.008; φ thresholds 0.5 / 0.2; envelope [13.4 %, 32.5 %] for the harness.
3. Ratify the disclosure (§4) — the production fraction 13.2 % is context, not a blind read.
4. Ratify the venue statement: the 17 % is a mirror-fleet number; production carries 13.2 % (10/76);
   both inside the as-designed envelope — the charter's "17 %" is re-labelled accordingly.
5. Accept G-4 as a launch gate and the `convert_to_best_guess_parameters` audit item as a record line.
