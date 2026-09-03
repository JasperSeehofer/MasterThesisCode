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
| [CMEM] verdict of record | C-STRUCTURAL-ONLY (row #220); the outside-cone truth-likelihood deficit R2c NOT-DISTINGUISHED, p = 0.0358 vs α = 0.01, pre-registered power ≈ 68 % (**row #226**, `gate_b_20260730/BIAS_HISTORY_LEDGER.md:3026`) | [DOC] |
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
Uncertainty (rev. 1 item 1): **SE(Δh_cone,c) = SD_IN,c · √(n_OUT + n_OUT²/n_IN) / I_c**, where SD_IN,c is
the per-event spread of s_e,c over the PRODUCTION in-catalogue IN class (n_IN = 66) — the population
the OUT events are drawn from — for each channel c separately, NEVER the harness dark-class SD.
**Robust-SD convention, fixed before any number is read:** SD_IN,c = 1.4826 · MAD_IN(s_e,c) (MAD-scaled),
with the plain sample SD reported alongside; the registered Z uses the MAD-scaled value. Rationale: at
n_IN = 66 a 2-event outlier pair can move the sample SD by a large factor; the read must disclose the
sample-SD/MAD-SD ratio and the two largest |s_e − median| IN events (2-outlier sensitivity). Z = Δh/SE.
**Cross-check (registered, must agree within 2·SE):** the exact leave-out counterfactual — the frozen
T0 scorer (`prod2d_closure_20260818/tier0_bootstrap_jackknife.py` convention; gradient-trapezoid weights,
physics floor) on the 1578 non-OUT events ⇒ Δmean_h,leave-out. Disagreement beyond 2·SE flags the linear
response as non-linear and the read is booked on the leave-out number with the flag.

**Harness replicate (zero compute; the CMEM-mechanism check):** the same OUT/IN split on the 67 post-flip
S3 cell-S universes (`b8_cal_harness_work_s4_postflip/seed9010NN_S/simulations/{prepared_cramer_rao_bounds,
diagnostics/event_likelihoods}.csv`; population n200-postflip; 843 catalogue-hosted events, expected
≈ 140 OUT). Statistics: f_OUT,harn with per-universe SE; Δs = s̄_OUT − s̄_IN with between-universe SE.
Reads whether the catalogue-hosted defect signature (+0.587 ± 0.064 per event over 67 universes — raw
checkpoints `b8_cal_harness_work_s4_postflip/universe_seed*_S.json`, `score_at_truth.no_bh.catalogue_hosted`;
also `../r-completion-residual/INFORMATION_FORECAST.md:19`; rd-s3-readout carries only the Z = 9.76 for
this class) concentrates on OUT events — REPORTED to d-calibration; not verdict-bearing here.

## 3. Bands (ORCHESTRATOR-DERIVED) and mapping to the floor

- **Materiality** T_mat = 0.008 in h — the standing convention (rows #247/#280/#284). Derivation: it is the
  threshold every counterfactual delta on this board has been judged against since wave 2.
- **Share thresholds:** φ ≥ 0.5 ⇒ the cone loss OWNS the floor (majority); φ < 0.2 ⇒ it does not (the
  remainder ≥ 4× the cone part; 0.2 ≈ T_mat/|offset| = 0.13 rounded up to one-in-five).
- **Fraction band (harness replicate only; the production fraction is disclosed-seen, §4):** f_OUT within
  the closed-form envelope [13.4 %, 32.5 %] ⇒ AS-DESIGNED; outside ⇒ INSTRUMENT question (fresh RULE).
- Power (rev. 1 item 1): the materiality margin is **M = T_mat / SE(Δh_cone,1D)** in SE units, filled by
  the registered read from the production IN-class SD (formula above). No numeric margin is quoted
  here: the earlier "0.68 ⇒ SE ≈ 0.0007, 11 SE" figure used the harness DARK-class SD as a proxy and is
  withdrawn; the production in-catalogue class carries a much larger per-event spread (C5's impostor
  scores are O(1) nats/h per event), so **M may be of order 1 — the arm may be UNDER-POWERED to
  distinguish CONE-OWNS-FLOOR from IMMATERIAL.** Honest consequence, registered now: if
  SE(Δh_cone,1D) > T_mat/3 (M < 3) the disposition is INTERMEDIATE-UNPOWERED (a bound), §4.
  False-fail at |Z| ≤ 3 under the null: 0.27 % (band-only, independent of SE sourcing).
- Forecast (stage 1, disclosed): C5's pre-flip in-catalogue impostor score −1.707/event × 10 ⇒ Σ ≈ −17
  nats/h ⇒ Δh_1D ≈ −0.005, φ ≈ 0.08 — IMMATERIAL predicted; the arm is designed to falsify that.

## 4. Disposition table (every row returns as a fresh RULE)

| disposition | trigger (1D primary; 2D reported alongside) | claim writeback | action |
|---|---|---|---|
| **IMMATERIAL-FLOOR-SHARE** | \|Δh_cone\| < 0.008 AND φ < 0.2 AND M ≥ 3 | c-residual-floor-consistent: the cone-loss component is bounded at φ + 3·SE/\|offset\|; "leading candidate for the absolute floor" DEMOTED | q-cone-loss SETTLED (kill criterion: "confirms the floor within its band ⇒ irreducible geometry, no fix") — with the bound |
| **CONE-OWNS-FLOOR** | \|Z\| > 3 AND φ ≥ 0.5 AND M ≥ 3 | c-residual-floor-consistent SUPPORTED as geometry (evidence, not a ruling) | contributes evidence toward d-residual-attribution, which stays OPEN pending d-calibration + d-photoz-leverage (charter §1.11, line 189); returns as a fresh RULE deferred to the morning (docket 2.3), exactly as the INTERMEDIATE row |
| **INTERMEDIATE-UNPOWERED** | SE(Δh_cone,1D) > T_mat/3 (M < 3), whatever Δh and φ read | no share claim; the cone-loss contribution is banked as the BOUND \|Δh_cone\| + 3·SE with φ_max = (\|Δh_cone\| + 3·SE)/\|offset\| | fresh RULE: bank the bound; the harness replicate (n_OUT ≈ 140) is the only in-cap route to power and returns as a revision-2 election |
| **INTERMEDIATE** | M ≥ 3 AND (\|Z\| > 3 AND 0.2 ≤ φ < 0.5; or \|Δh\| ≥ 0.008 with φ < 0.2; or 1D/2D disagree in disposition; or linear vs leave-out disagree > 2·SE) | partial share quoted | fresh RULE: bank the share; a revision (≤ 2) only if the author wants the z-window component resolved |
| **INSTRUMENT / NO-READ** | G-1…G-4 red; g-population red | nothing banked | repair; no revision consumed |

**Blindness disclosure (binding):** the registration author computed the production geometric census
(10/76 OUT; chord/radius distribution) and read the P6 log line BEFORE writing this draft, as a
stage-0 design fact (it decides whether production is sky-scattered at all — it is). The fraction is
therefore NOT a blind read on production and is banked as context, never as a verdict. The primary
statistic (the OUT events' scores, Δh_cone, φ_cone, the leave-out delta, the harness replicate) has NOT
been read by anyone.

## 4b. Parent kill criterion (verbatim) and blindness status

Charter `RESEARCH_GRAPH_1_PROPOSAL_20260901.md:46`, q-cone-loss kill_criterion, quoted verbatim:
"measurement confirms the floor within its registered uncertainty band -> settled as irreducible
geometry; no fix pursued". Read with §4: IMMATERIAL-FLOOR-SHARE and CONE-OWNS-FLOOR both "confirm the
floor within its band" in the charter's sense (a bounded share is a confirmed floor component); the
question SETTLES on either; INTERMEDIATE-UNPOWERED does not settle it and consumes a revision only
if the author elects the harness-replicate route (max_revisions 2).

**Blindness status:** primary statistic point estimates exist in a gate record dated 2026-09-03
(unblinded by a design-gate side effect: DESIGN_GATE_stats.md); band thresholds were frozen before
that record; the registered read is executed by an agent that has not opened that record. The
revising author did not open it. (The production OUT FRACTION, 10/76, was additionally seen by the
registration author at stage 0 — §4 disclosure — and is context, never a verdict.)

## 5. Gates

- **G-1 pins:** catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`; CRB md5 `9a1f2a14384a9281c97ca3be312ddaab`;
  `GIT_COMMIT_AT_RUN.txt` = `1ec9514d`; STOP on mismatch (CLAUDE.md pinning rule).
- **G-2 anchors (instrument byte-id):** reproduce R-MKER-6's anchor on `p3_2d_fleet_20260825/bc_900121_work`
  event 20 (chord 1.674660e-03 ± 5e-10, radius 1.4956979545757095e-03 ± 1e-15 — `cmem_reads.py:32,107-111`)
  AND CMEM-A1's anchor on `p3_b0_work` bc/900101/event 0 (0.0116656941007181 / 0.0359121946154451,
  `cmem_a1.py:67`). Both fleets are on disk. A miss = INSTRUMENT-DEFECT.
- **G-3 join:** event_idx = CRB row index; scored set = {0..1589} − {1203, 1356}; in-catalogue count = 76 =
  the P6 denominator; the P6 numerator 66 must equal n_IN (the log line quoted verbatim in the record).
  Mismatch ⇒ INSTRUMENT-DEFECT (rev. 1 item 3). Dry-run status: GREEN (`cone_loss_work/cone_loss_gates.json`).
- **G-4 scatter law (production is sky-scattered by the forward model, not by a frame artefact):** the
  sky offsets' Mahalanobis² under the row's own Σ' must be χ²₂-distributed (KS at α = 0.05 on 76
  events) — the DECISIVE clause — AND the envelope clause, **re-stated in rev. 1 (item 7)**: the
  realised n_OUT must be consistent with SOME expected fraction p in the closed-form envelope
  [13.4 %, 32.5 %] under Binomial(n_in-cat, p), i.e. the exact two-sided binomial test of n_OUT against
  the NEAREST envelope edge must not reject at α = 0.05. The draft's original wording ("f_OUT must sit
  inside the envelope") applied an asymptotic expectation to a 76-event realisation whose 1σ sampling
  width is ≈ 4 %; the dry-run correctly reported it RED (f_OUT = 10/76 = 0.1316 vs edge 0.134,
  `cone_loss_gates.json` g4_scatter_law.envelope_passed = false) while the decisive KS clause passed
  (D = 0.066, p = 0.87). Under the corrected rule 10/76 against p = 0.134 (expected 10.2) is not a
  rejection — the registration's expectation was wrong, not the instrument; **no STOP is declared**.
  The harness replicate (n ≈ 843) uses the same binomial form, where the envelope is genuinely tight.
  A KS failure, or a binomial rejection against BOTH edges, means the offsets are not the designed
  Fisher tail (e.g. a frame translation — cf. `resolve_host_recovery_position`, handler.py:853)
  ⇒ INSTRUMENT-DEFECT, STOP, fresh RULE. Consequence for the build: `cone_loss_reads.py`'s envelope
  clause must be changed to the binomial form and the `--dry-run` re-run (builder, b-cone-scorer)
  before launch; the currently written `cone_loss_result.json` (verdict INSTRUMENT-DEFECT) is
  superseded by that re-run and must not be read. Open audit item: which of (d_L, qS, phiS) the production pool
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

## REVISION 1 (2026-09-03) — changes against MUSTFIX_REVISION1_20260903.md (band thresholds untouched)

| item | change |
|---|---|
| 1 (stats) | §2: SE(Δh_cone,c) re-sourced to the PRODUCTION in-catalogue IN class per channel; MAD-scaled robust-SD convention fixed before any read, sample SD reported alongside, 2-outlier sensitivity disclosed. §3: numeric margin withdrawn (it used the harness dark-class SD); margin M = T_mat/SE stated as a formula; under-power disclosed plainly. §4: **INTERMEDIATE-UNPOWERED** row added (SE > T_mat/3 ⇒ bound only); IMMATERIAL / CONE-OWNS-FLOOR / INTERMEDIATE now require M ≥ 3. |
| 2 (design) | §4 CONE-OWNS-FLOOR action cell rewritten: evidence toward d-residual-attribution (open pending d-calibration + d-photoz-leverage, charter line 189), fresh RULE deferred to the morning (docket 2.3), same phrasing as INTERMEDIATE; the "(charter)" citation dropped. |
| 3 (design) | §5 G-3: "Mismatch ⇒ INSTRUMENT-DEFECT" on its own line; dry-run status GREEN noted. |
| 4 (design) | §4b: charter line 46 kill criterion quoted verbatim in quotation marks, with its reading against the §4 rows. |
| 5 (provenance) | §0: the p = 0.0358 / power ≈ 68 % clause re-cited to row #226 (ledger line 3026); row #220 kept for C-STRUCTURAL-ONLY only. |
| 6 (provenance) | §2: "+0.587 ± 0.064" re-cited to the raw checkpoints' `score_at_truth.no_bh.catalogue_hosted` and INFORMATION_FORECAST.md:19; rd-s3-readout credited only with Z = 9.76. |
| 7 (build dry-run) | `cone_loss_gates.json` read: G-1 (3 pins), G-2 (both anchors), G-3 all GREEN; G-4 KS clause GREEN (D 0.066, p 0.87); the RED is the envelope clause (0.1316 vs 0.134). Diagnosis: registration expectation wrong (asymptotic envelope applied to a 76-event realisation). §5 G-4 re-stated as an exact binomial test against the nearest envelope edge; NO STOP declared; the builder must apply the one-clause change and re-run `--dry-run`; the existing `cone_loss_result.json` is superseded and must not be read. |
| both | §4b: blindness-status line added verbatim (plus the stage-0 fraction disclosure). |

Not changed: bands (T_mat 0.008, φ 0.5 / 0.2, \|Z\| ≤ 3, envelope [13.4 %, 32.5 %]), the statistic definition, the launch CLI (§7 — matches the built `cone_loss_reads.py` argparse), the cost line.
