# Research Graph 1 — ADDENDUM proposal (uncharted branches J–N for the 2026-09-03 overnight slot)

Date: 2026-09-03 (evening). Chair of record: Fable 5.1 orchestration session. Synthesis author: a
top-tier subagent (this document). Author of record for all scientific decisions: Jasper Seehofer.
Status: PROPOSAL — a reviewable decision artifact per CLAUDE.md "Proposing decisions". Per row #334
item (8) ("the candidate scan is authorized. Any Graph 1 addendum branch returns for ratification at
L10"), every branch below is a NEW node set outside the row #290 scope hash and therefore returns as a
fresh RULE by construction. What can run tonight without a ruling is stated in section 6 and is
confined to registration AUTHORING and verdict-free reads under the row #325 grant ("please continue
autonomous. you can make decisions but flag them", quoted verbatim at row #325).

Parent charter: RESEARCH_GRAPH_1_PROPOSAL_20260901.md (node types, edge kinds, panel law, k- schema).
Wave-3 envelope of record: DECISION_DOCKET_WAVE3_20260903.md items 2.1–2.3 (rows #334/#335).
Convention: every number carries its file/row source; every cost the synthesis author derived is
marked ORCHESTRATOR-DERIVED; nothing below was run — this document was produced read-only.

---

## 0. Frame

- Objective, unchanged from the parent charter section 0: questions moved to SETTLED (verified,
  refuted, or bounded-undetermined). Refuted pays like verified. Bias reduction is not the objective
  (author's binding 2026-08-05 value).
- Selection rule applied: rank by (information gained per CPU-h) x (independence from wave 3 so the
  branch can run tonight in parallel). Five branches survive; two candidates are deferred with reasons
  (section 4); nine were refuted at scan (Appendix A).
- Synthesis corrections to the scan (found by opening the sources, not the scan summaries):
  1. `paper-pg-b0-twin-identity` claimed the b0 identity test was "never instantiated". It WAS
     executed and adjudicated UNDISCRIMINATING on 2026-08-24 (row #177); the author granted a
     finite-moment redesign (row #178 item 3); a stage-0 DRAFT exists (CLAIM_B0_FINITE_MOMENT_20260824.md,
     header: "STATUS: DRAFT ... Registers NOTHING"); no mention of "finite-moment" or "identity test"
     appears anywhere in the ledger after row #186. The branch is RESCOPED to the un-registered redesign
     (Branch N).
  2. `paper-pg-n-scaling-bias` proposed sub-sampling N=400/800/1200 and reading the offset's mean. The
     mean offset over random sub-samples of a fixed event set equals the full-set offset in expectation
     regardless of whether the full-set offset is a fluctuation — that read is uninformative by
     construction. The informative quantities are the bootstrap WIDTH at N and the jackknife influence
     structure; the branch is RESCOPED accordingly (Branch K).
  3. `systematics-SB-01` claimed no alternative-truth mock exists. An UNSEALED alternative-truth
     closure seed exists: `closure_seed64000_h0p67/combined_posterior_2d.json` (dated 2026-08-03):
     `map_h 0.67`, `n_events_used 1343`, `variant posteriors_with_bh_mass`, 44 posterior nodes;
     REALISTIC_READOUT.md section 7 records the pool as lossy (GPU tasks hit the 30-min wall,
     "~10 of 40 requested steps per task"). What has never run is (a) the SEALED protocol, (b) a
     p_det injection pool regenerated at h_inj (the only pool is single `h_ref: 0.73`,
     cluster/datasets.yaml lines 36–41), (c) the current post-flip stack (six production [PHYSICS]
     flips since, row #328). The branch is RESCOPED to those three increments (Branch M).
  4. `systematics-SB-04` assumed the instrumented timeout logs are local. The seed61000 production
     pool's simulation logs are NOT on this machine (`results/campaign51_20260728/realistic_20260729/seed61000/simulations/`
     holds only `cramer_rao_bounds.csv` + `injections`; 0 log files); the only local post-instrumentation
     logs are `results/_archive/run_20260707_seed3000/` (99 logs, 1 198 "timed out ... params=" lines,
     commit a545c0eb, timestamp 2026-07-08 per `run_metadata_16.json`). Branch L runs locally on
     seed3000 first; the seed61000 logs are a cluster read that returns for a word (section 6).

---

## 1. The addendum graph

### 1.0 Question and claim layer

Question nodes (kill_criterion mandatory per infra 2.1):

| id | question it settles | kill_criterion |
|---|---|---|
| q-parity-growth | Is the T1.3-zwin GATE PARITY residual (row #273: "max rel diff 3.9%-44.7% across seeds", "consistent in kind with, but numerically larger than, the previously-RATIFIED E19 comparand residual; not re-adjudicated") accounted for EXACTLY by the z-window's own registered added-candidate term at the truth node (PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md lines 401–403: "every event's L_cat_no_bh differs from T1.2's truth node ONLY through the added candidates (the T1.2 candidates' contributions are bit-identical"), with only the E19-level residual (no_bh max_rel 2.03e-5–4.88e-5, hier_s0_registered_run/s0a_score_output.json) left against the banked bc — or does an uncaptured mechanism sit under the row #287-certified [HIER] instrument? | (i) per-event Delta ln L between the zwin truth node and the T1.2 truth node is zero (to the E19 floor) for every event with no added candidate, and equals the added-candidate sum where candidates were added -> EXPLAINED-BY-DESIGN, the row #273 "consistent in kind with E19" wording is CORRECTED (E19 is the wrong mechanism for the 44.7%; the window is), question SETTLED-refuted (no anomaly). (ii) any event with Delta ln L above 10x the E19 floor and no added candidate, or a mismatch between Delta and the added-candidate sum beyond the floor -> live instrument anomaly; caveat attached to d-photoz-leverage; fresh RULE. Revision cap: 1 (this is a decomposition, not a design). |
| q-s0b-parity | Does the S0-B production truth node (retrieved/s0b_run_20260902/s0a_full_output.json: `config iiib`, `h_values [0.73]`, `theta_zwindow off`, `theta_phi_divisor off`, `catalogue_leg_1d_mass_aware off`, 1 588 rows per node) reproduce the production comparand at h=0.73 exactly? The driver could not evaluate this: `gate_parity` status is `NO_BANKED_CSV` (it looked for `p3_b0_work/bc_900101_work/.../event_likelihoods.csv`, a b0-venue path). The correct comparand exists locally: `retrieved/run_20260902_graph1_c0prime_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv` at h=0.73 (1 588 rows; mass-aware off, i.e. the same flag state as S0-B), whose with-BH columns are flip-invariant `ndiff=0/1588, max_abs=0` (row #299). | with-BH columns (`L_cat_with_bh`, `num_log_term_with_bh`, `combined_with_bh`) max_abs = 0 AND no-BH columns max_abs = 0 against c0prime_off -> S0-B truth node is the production evaluation, parity gate discharged for d-photoz-leverage. Any nonzero -> the S0-B run is not on the production comparand; d-photoz-leverage dossier carries a STOP flag (panel law: banked, blocked from interpretation) and returns as a fresh RULE. |
| q-2d-offset-frequentist | Is the 2D offset's stated significance frequentist-robust on the production event set itself: bootstrap-over-events width of mean_h at N=1588 vs the posterior sigma_h (re-baseline iiib 2D: map 0.665 / mean 0.665854 / sigma 0.018475, row #302; 2026-08-27 readout: mean 0.6633, offset -0.0667, sigma 0.0184, pull +3.63, CLAUDE_SCIENCE_BRIEF.md section 3 table) — and is the offset diffuse (per-event systematic) or carried by a small influential subset (jackknife)? | (i) bootstrap SE(mean_h) at N=1588 within a factor 1.5 of sigma_h AND the smallest jackknife-removed subset that moves mean_h to within 1 sigma_h of truth is > 10% of events -> the offset is a diffuse per-event systematic; the "3.6x its own width" framing (artifact section 00 via state table) is corroborated on an axis the mechanism split never tested; SETTLED-verified. (ii) bootstrap SE >= 2x sigma_h (posterior over-confident) OR <= 5% of events carry the offset (heavy-tail) -> the significance/uniformity claim needs re-scoping before d-paper-1d2d-verdict; fresh RULE. Scaling read at N=400/800/1200 is reported for the WIDTH only (never the mean, section 0 correction 2). |
| q-timeout-selection | Is the waveform-timeout exclusion (G7 row 8: "0.6–1.25% of events per stage removed"; status "CAMPAIGN: params now logged at all catch sites `4d1c27a` -> bin by (M, e0, p0); expect higher rates at the deeper post-dt2 population") flat across (M, e0, p0) within Poisson noise, or does it carry a gradient that imprints a differential selection on the detected set? | flat within Poisson (chi-square p > 0.05 across the registered bins, both stages) on seed3000 AND (when fetched) seed61000 -> G7 row 8 downgraded to NON-ISSUE with a citable bound; SETTLED-refuted. A gradient at > 3 sigma along any of M, e0, p0 -> a new quantified selection systematic; its H0 projection returns as a fresh RULE (and as an input to G7). |
| q-anti-tuning | Does the current production stack recover a SEALED h_inj != 0.73 (redteam T-1, PHYSICS_METHODOLOGY_REVIEW.md lines 401–408: "the only test that can rule out 'the estimator was assembled, over 20+ variants, until it pointed at 0.73'"; GitHub #39 OPEN, 0 comments; ledger preamble section 4 item 7 "ordered, never run") — given that the unsealed 0.67 closure (map_h 0.67, 2026-08-03, section 0 correction 3) predates six production flips and used the h_ref=0.73 p_det pool? | staged: (m1) HEAD-stack re-score of the existing 0.67 pool: 2D map AND mean inside the pre-registered band around 0.67 -> the alternative-truth mechanism check survives the flips (partial, unsealed; explicitly NOT the T-1 verdict); outside -> the post-flip stack has an h-dependent defect; HALT paper claims pending a fresh RULE. (m2) sealed pool with p_det regenerated at h_inj: recovered MAP/mean inside the band -> T-1 PASSED, every banked known-truth verdict gains the anti-tuning stamp; outside -> undetected tuning dependency; all iiib/joint_r1 verdicts re-scoped. |
| q-b0-finite-moment | Does a finite-moment identity statistic (the row #178 item-3 redesign; DRAFT targets under the F-0 conditioning: "B-T ≈ 1.59, B-C ≈ 0.52 — not 1 and not <S-bar_phi>-anything", CLAIM_B0_FINITE_MOMENT_20260824.md section 0) DISCRIMINATE the twin from the coded catalogue leg on the catalogued-host venue b0 ("the only venue able to adjudicate catalogue-leg correctness", CLAUDE_SCIENCE_BRIEF.md lines 82–83) — where the registered mean-of-odds statistic was UNDISCRIMINATING (row #177: "B-R (the refuted arrangement) passes the same bands ... k-hat up to 2.7") — now that the twin is production (A14: +0.002507 iiib / +0.004114 joint_r1, both <= T_mat 0.008, row #284) and TWIN-CALIBRATED on C-A only at the self-consistency level (row #186)? | design gate first: the B-R control (refuted arrangement) must FAIL the redesigned band on the banked b0 CSVs while B-T/B-C are read blind -> statistic DISCRIMINATING, proceed to fresh HEAD pairs; B-R passes again -> the redesign is also vacuous, park the b0 identity question bounded-undetermined with the reason (heavy tail is structural to the venue) and record it as the brief's "awaits a catalogued-host identity test" closure. On fresh pairs: twin identity inside band -> CONFIRMED on the adjudicating venue; outside -> twin correctness REFUTED on b0 despite C-A calibration; fresh RULE either way. |

Claim nodes touched (status gate/decide-written only):

| id | status now | discriminated / verified by |
|---|---|---|
| c-hier-instrument-certified | supported (row #287: Z_b -1.808 / +0.773, both abs Z <= 3) | q-parity-growth can attach a caveat, never re-certify or de-certify (that is a fresh RULE, parent section 2 g-score-null rule) |
| c-2d-offset-systematic | supported by mechanism decomposition only (artifact section 10 three-way split; d-residual-attribution pending) | q-2d-offset-frequentist (independent axis) |
| c-timeout-selection-negligible | conjectured (G7 row 8 "bounded sub-% on H0", never binned) | q-timeout-selection |
| c-no-tuning-to-0.73 | supported by inspection only (redteam: "no numerical anchor to 0.73", GitHub #39 body) + one unsealed pre-flip closure | q-anti-tuning |
| c-twin-correct-on-catalogued-host | UNDISCRIMINATING (row #177, ratified row #178 item 1) | q-b0-finite-moment |

### 1.1 Branch J — [PAR] GATE PARITY growth under the z-window (rank 1)

Depth 2 (+1 optional). Zero compute. Object: the parity numbers in
`tree2_20260830/hier_s0_zwin_run/s0a_score_output.json` `gate_parity` (per seed, no_bh / with_bh
`max_rel_diff`, `max_abs_diff` in nats, n events): 900101 0.447 / 0.345 (max_abs 5.233 / 4.747, n=106);
900102 0.0388 / 0.0919 (0.226 / 0.796, n=120); 900103 0.216 / 0.188 (1.355 / 1.542, n=105);
900104 0.0435 / 0.0743 (0.330 / 0.594, n=130); `pass_exact=false` on all. Pre-window baseline on the
same seeds and the same banked bc (both `fanout1_20260829/hier_s0_registered_run/s0a_score_output.json`
and `tree2_20260830/hier_s0_recert_run/s0a_score_output.json`, identical values): no_bh 4.88e-5 /
4.27e-5 / 3.19e-5 / 2.03e-5; with_bh 0.0594 / 0.0892 / 0.12 / 0.0343. So the no_bh residual grew by
~10^4 when `theta_zwindow=on, z_window_k=4.0` was engaged (ORCHESTRATOR-DERIVED ratio 0.447/4.88e-5 ≈ 9.2e3),
while E19 — the ratified disposition (row #255 A2(c): "the forensic's E19 diagnosis of the 5.718e-4
residual (generator grid 401→4001) RATIFIED") — is an injection-level `z_true` delta of max 1.06e-5
(B1_1_S0A_DEFECT_FORENSIC_20260829.md line 94). The gate document registered the truth-node
expectation explicitly (lines 401–403 and R8, lines 472–474: "at k = 4 the intersection candidates'
per-candidate values (T2.2 dump) are bit-identical and only added rows differ"). Sub-question
disclosed: the with_bh 3.4–12% residual pre-dates the window and is not obviously covered by the
5.718e-4 E19 wording; it is read in the same pass, reported separately, not adjudicated.

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| rd-parity-decomp | read | per-event Delta ln L (both channels) zwin-truth vs T1.2-truth (`hier_s0_recert_run/s0a_seed9001xx/node_truth_sites2.2_nosmear_divisor/.../event_likelihoods.csv`) and vs the banked bc (`p3_b0_work/bc_9001xx_work/seed9001xx/simulations/diagnostics/event_likelihoods.csv`, 4 seeds, all local); classification of every event into {no added candidate, added candidates}; the added-candidate set from the k=1 vs k=4 window on the catalogue query (zero evaluate()) — the T2.2 candidate dump (`tree2_20260830/candidate_dump_run/s0a_seed9001xx/node_truth_ft/simulations/diagnostics/*.csv`, columns `event_idx,h,catalog_index,batch,z_g,...,is_true_host`) is used ONLY if its venue tag (`ft`) is verified to match the bc comparand; else reconstructed | spawned-by q-parity-growth; feeds from row #273 + the three CSV families above; independent of every wave-3 arm | g-precision (nats arithmetic pinned at full precision); g-population (0 mixed rows across the 4 seeds) | kill_criterion (i)/(ii) of q-parity-growth, applied per event; the E19 floor is the pre-window no_bh max_rel (<= 4.88e-5) | 0 CPU-h (pandas on 4 x ~110 rows x 2 comparands) | sonnet / high (reader); the decisive classification re-derived by the addendum's decisive verifier (top-tier) |
| rd-s0b-parity-vs-c0prime | read | q-s0b-parity: exact-parity diff of the S0-B truth node (`retrieved/s0b_run_20260902/s0a_seed900101/node_truth_iiib_sites2.2_nosmear/simulations/diagnostics/event_likelihoods.csv`, 1 588 rows) against c0prime_off at h=0.73 (1 588 rows), all 19 columns | spawned-by q-s0b-parity; feeds from rows #299/#302 (c0prime_off is the certified comparand); feeds INTO the d-photoz-leverage dossier (wave-3 item 2) as its missing parity stamp | g-c0-baseline pattern (max_abs = 0 on shared columns; same-machine rule, row #325 item 3 — both files were produced on the cluster and retrieved with md5 MATCH, row #334 item 4) | max_abs = 0 both channels -> stamp GREEN; nonzero -> STOP interpretation of the S0-B read (banked, blocked), fresh RULE | 0 CPU-h | sonnet / medium; the chair re-derives the max_abs pair before attaching it to the dossier |
| m-parity-401grid (optional, revision-free) | measure | only if rd-parity-decomp lands on (ii): re-run the 4 truth nodes with the comparand's own generator grid (`_B0I_ZTRUE_GRID_N` 401, forensic E19 row) to isolate the grid delta from the window delta | registered-by the q-parity-growth kill criterion; authorized only by a fresh RULE (section 3 row A-J2) | g-byte-id on the k=1 identity node first (gate doc R8: "at k = 1.0 ... max_abs 0.0 on every numeric column") | if the 401-grid truth node reproduces the bc to the E19 floor and the window term to the floor -> EXPLAINED; else anomaly stands | approx 11 CPU-h ORCHESTRATOR-DERIVED (32.08 CPU-h / 12 seed-nodes ≈ 2.7 CPU-h per seed-node, row #273, x 4 truth nodes); local 14-core, no cluster | sonnet / low |

### 1.2 Branch K — [NSC] frequentist width and influence structure of the 2D offset (rank 2)

Depth 1. Zero compute. Data: `retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv`
(65 108 data rows = 41 h-nodes x 1 588 events; columns include `combined_no_bh`, `combined_with_bh`)
plus `posteriors/` and `posteriors_with_bh_mass/` `h_*.json` (41 nodes each). Reference implementation
for the combination: `bayesian_inference/posterior_combination.py` (loads `h_*.json`, canonical
Sigma log L). Information floor for context: sigma_h 0.001747058397810697 (no_bh) / 0.001746970592930231
(with_bh) at N_ref 1588 (state table, b8_information_floor.json).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| rd-2d-bootstrap-jackknife | read | (a) bootstrap-over-events (B = 2 000 resamples, seed pinned) SE of mean_h and MAP at N=1588, and at N=400/800/1200 (width only); (b) leave-one-out jackknife influence of every event on mean_h; (c) the minimal-subset statistic: the fraction of events (ranked by influence) whose removal brings mean_h within 1 sigma_h of 0.730; both channels, iiib; joint_r1 as a replica from `retrieved/run_20260902_graph1_headrebaseline_joint_r1` | spawned-by q-2d-offset-frequentist; feeds from m-head-rebaseline (rows #299/#302, GREEN-AS-CORRECTED); consumes NOTHING from wave 3 | g-precision (log-sum at full precision; cancellation sentinel per row #282); g-population (1 588 rows per node, no G-EXT nodes mixed in — the 41-node grid only); g-censoring (any bootstrap MAP at the 0.60/0.86 rail is counted and reported as a rail fraction, rows #267/#280 rule) | kill_criterion of q-2d-offset-frequentist; the ratio bootstrap-SE / sigma_h and the minimal-subset fraction are the two decisive numbers; verdict-free tonight | 0 CPU-h (numpy: 2 000 x 41 x 1 588 log-sums ≈ 1.3e8 flops) | sonnet / high (analyst); both decisive numbers re-derived by the addendum's decisive verifier (top-tier) |

### 1.3 Branch L — [TMO] waveform-timeout selection binned by (M, e0, p0) (rank 3)

Depth 2. Instrumentation confirmed at `darksiren_emri/main.py` lines 763–770 (SNR stage: "Waveform/SNR
computation timed out (>90s). Skipping event... params=%s" with `_parameters_to_dict()`) and lines
1293–1302 (injection stage, "%d total"); G9_timeout_scan.md line 120 records the PRE-instrumentation
state ("No parameter information at any timeout site ... timeouts cannot be binned"). Local data:
`results/_archive/run_20260707_seed3000/` — 99 logs, 1 198 "timed out" records with full parameter
dicts, `simulations/cramer_rao_bounds.csv` 3 325 rows (denominator), commit a545c0eb (2026-07-08).
The production pool's logs (`run_20260729_seed61000`, DATA_INVENTORY.md line 277; `simulation_steps=40`,
cluster/datasets.yaml line 137) are on the cluster only.

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| rd-timeout-bin-seed3000 | read | timeout rate per stage in registered bins of M (log-spaced, 5 bins), e0 (5), p0 (5) and the 2-D (M, p0) cell, with Poisson intervals; the population depth of seed3000 (z_cut / HOST_DRAW_Z_MAX) read from its CRB z distribution and disclosed | spawned-by q-timeout-selection; feeds from the local archive above; independent of wave 3 | g-population (the timeout records and the kept-event denominator must come from the same tasks — match by log file / task id); design-validity gate on the bin edges BEFORE the rates are read (blind) | kill_criterion of q-timeout-selection on seed3000; the seed3000 result is a partial (depth to be disclosed), never the final bound | 0 CPU-h (grep + pandas) | sonnet / medium |
| rd-timeout-bin-seed61000 | read | the same on the production pool (the "deeper post-dt2 population" G7 row 8 names) | requires the `run_20260729_seed61000` `*.log` files fetched from the cluster (a read, not an evacuation; still a cluster operation under /cluster — see section 6) | as above | as above; the two populations reported side by side | 0 CPU-h compute; transfer of the log files (size unknown, not sourced locally) | sonnet / medium |

### 1.4 Branch M — [SEAL] sealed-truth anti-tuning mock (rank 4 by information per CPU-h; highest novelty)

Depth 3, staged. Existing machinery: the cheap d_L-rescaling variant
(`scripts/bias_investigation/test_17_rescale_crb_to_h065.py` + `cluster/evaluate_closure_h065.sbatch`,
whose header registers gates G7a–G7d incl. "G7d MAP ≈ 0.730 → pipeline TUNED to 0.73; HALT paper") —
T-1 itself labels this "NOT a substitute" (PHYSICS_METHODOLOGY_REVIEW.md lines 409–413). The unsealed
0.67 pool exists on the cluster (`run_20260729_seed64000_h0p67`, jobs 6090909–6090912,
REALISTIC_READOUT.md section 7) with a local 2D combination (section 0 correction 3). No sealed-h
machinery exists in the repo (grep of scripts/, darksiren_emri/, cluster/ for sealed / h_inj / blind_mock:
only the rescaling scripts).

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| r-sealed-mock | register | the T-1 protocol: (1) h_inj drawn by a script from a registered prior (uniform on [0.62, 0.84] proposed — ORCHESTRATOR-DERIVED, inside the decoupled h-grid of [PHYSICS] a26959b4, row #313), stored as a salted SHA-256 commitment in the registration and the value in a file the analysis code never opens; (2) a fresh p_det injection pool at h_inj (same catalogue, completeness cache, population model — T-1 lines 402–404); (3) production simulate at h_inj; (4) unchanged evaluate on the 41-node grid; (5) pre-registered band on the recovered PULL (MAP and mean), not on h; (6) unseal ceremony = one ledger row; max_revisions 1 | spawned-by q-anti-tuning; feeds from GitHub #39, redteam T-1, `docs/derivations/realistic_host_observation_model.md` lines 532–533, the 0.67 closure record; feeds INTO d-paper-1d2d-verdict and d-paper-coverage as an anti-tuning stamp (informational edge, not a requires-manifest change — that would fail L10 on the parent) | design-validity gate (sonnet panel, blind to results); research-cycle stages 2/3; /physics-change NOT triggered (no formula changes; new scripts only) | band content, prior on h_inj, and the pool cost return as fresh RULE d-sealed-register before ANY launch | authoring only | top-tier / xhigh (the addendum's single prereg author; batched with r-b0-finite-moment) |
| m-closure067-headstack | measure | stage (m1): the existing 0.67 pool re-scored on the current production default (post-flip, post-A14), 41 nodes, both channels — the mechanism check the flips have never seen | registered-by r-sealed-mock (its stage-1 section); feeds from `run_20260729_seed64000_h0p67` CRB on the cluster + the pool's lossy-task disclosure (REALISTIC_READOUT.md section 7) | g-znorm on the flipped leg; g-censoring; g-population (the pool's event count disclosed vs 1 343 used on 2026-08-03) | pre-registered pull band; outside -> HALT paper rulings, fresh RULE | 9–94 CPU-h ORCHESTRATOR-DERIVED band from two sourced anchors: 84 tasks x approx 6.5 min for both venues (state candidate 11, i.e. approx 4.5 CPU-h per 41-node venue) vs approx 94 CPU-h for the 55-node A18 arm (row #285); cluster CPU, /cluster preflight READY required | sonnet / low (array) |
| m-sealed-pool | measure | stage (m2): the decisive sealed run (fresh p_det pool at h_inj + simulate + evaluate) | registered-by r-sealed-mock; authorized only by d-sealed-register | g-znorm, g-censoring, g-population, plus the unseal ceremony | the registered pull band; the verdict is binary and cannot be revised (max_revisions 1) | GPU: injection pool 500 tasks x <= 0.5 GPU-h (cluster/datasets.yaml lines 39–40: 500 files / 50 000 events; cluster/inject.sbatch line 19: `--time=00:30:00`) -> <= 250 GPU-h upper bound ORCHESTRATOR-DERIVED; simulate N_tasks x <= 0.5 GPU-h (cluster/simulate.sbatch lines 36–38; N_tasks for seed61000 NOT sourced locally — derive from the cluster directory before the ruling); evaluate approx 94 CPU-h (row #285 anchor) — heavy; must land before the 2026-09-23 workspace expiry or after the next workspace (docket item 12b) | sonnet / low (arrays); the unseal read: top-tier decisive verifier |

### 1.5 Branch N — [B0FM] finite-moment b0 identity statistic (rank 5)

Depth 2 (+1). The registered mean-of-odds statistic was UNDISCRIMINATING because "one legitimate
low-responsibility event (seed 900108 idx 2, w ≈ 2.3e-5, not anomalous, pull −0.79σ) inflates the raw
SEMs into vacuity (k̂ up to 2.7; k̂ > 1 pervasive)" (row #177 item 1). The DRAFT redesign found F-0
(the intake filter `distance_relative_error < 0.10` "silently removes 41.8% of the drawn events",
class-asymmetric, outside the b0 blindness list; CLAIM_B0_FINITE_MOMENT_20260824.md section 0) and
derived conditioned targets. Banked b0 data are local and complete
(`p3_b0_work/bc_9001xx_work` + `bt_*`, 24 seeds, commit 3bd6b564, DATA_INVENTORY.md entry
`p3_b0_identity_fleet_20260823`: "FULLY RECOVERABLE") but PRE-DATE Sigma^phi (row #179), WBHZERO
(cf4f8a2a), A18 and A14 — a stale basis usable for a statistic design gate, never for a verdict.

| node | type | settles | inputs (edges) | gate/instrument | band / STOP | cost | tier |
|---|---|---|---|---|---|---|---|
| r-b0-finite-moment | register | the finite-moment identity statistic (trimmed/Winsorised log-odds or a rank statistic — the DRAFT's candidates), its F-0-conditioned targets, the B-R refuted-arrangement control as a MANDATORY design gate, blindness list amended with the intake filter; max_revisions 2 ORCHESTRATOR-DERIVED (parent's provisional default) | spawned-by q-b0-finite-moment; feeds from rows #177/#178/#179/#180, CLAIM_B0_FINITE_MOMENT_20260824.md, PREREGISTRATION_B0_IDENTITY_20260823.md (A21-B0-C binds future bands of this family, row #177) | design-validity gate; the B-R control evaluated on the banked (stale-basis) CSVs BEFORE B-T/B-C are looked at (blind) | B-R passes the redesigned band -> STOP (statistic vacuous; park bounded-undetermined, close the brief's open thread with the reason); B-R fails as designed -> band content returns as fresh RULE d-b0fm-band | authoring + 0 CPU-h (the control read is zero-compute on 24 banked CSVs) | top-tier / xhigh (batched with r-sealed-mock, same identity) |
| m-b0-finite-moment | measure | the twin vs coded identity on FRESH HEAD b0 pairs (post-A14 production default), 12 seed-pairs | registered-by r-b0-finite-moment; authorized only by d-b0fm-band | g-znorm (the twin's S-bar_phi content, the brief's "g-znorm identity check"), g-population, g-censoring | registered band; neither-band -> INTERMEDIATE -> fresh RULE | 12–20 CPU-h (PREREGISTRATION_B0_IDENTITY_20260823.md lines 173–174: "2 arms × 12 seeds = 24 × evaluate() ≈ 12–20 CPU-h (banked b0 anchor: 0.478–0.9 CPU-h/seed)"); local or cluster | sonnet / low |

### 1.6 Convergence decide nodes (all return as fresh RULEs; none pre-granted)

| id | tag | question put to the author | requires-manifest |
|---|---|---|---|
| d-parity-disposition | RULE | accept the rd-parity-decomp classification (EXPLAINED-BY-DESIGN with the row #273 wording corrected, or ANOMALY with a caveat on the [HIER] certification)? | rd-parity-decomp record with g-precision green; the verifier's re-derivation attached |
| d-s0b-parity-stamp | RULE (folds into d-photoz-leverage) | attach the S0-B parity stamp to the d-photoz-leverage dossier as GREEN / STOP? | rd-s0b-parity-vs-c0prime record |
| d-2d-frequentist | RULE (informs d-paper-1d2d-verdict) | is the offset's significance frequentist-robust and diffuse? | rd-2d-bootstrap-jackknife record with g-precision + g-population green |
| d-timeout-bound | RULE (re-grades G7 row 8) | NON-ISSUE with bound, or a new systematic? | rd-timeout-bin-seed3000 (and -seed61000 if fetched) |
| d-sealed-register | RULE | ratify the sealed-mock protocol, prior on h_inj, the pull band, the stage (m1) launch, and the (m2) pool cost | r-sealed-mock design-gate record |
| d-b0fm-band | RULE | ratify the finite-moment band and the fresh-pair launch | r-b0-finite-moment design-gate record incl. the B-R control result |

### 1.7 Bounded cycles

r-sealed-mock max_revisions 1 (a sealed test cannot be re-drawn without destroying the seal);
r-b0-finite-moment max_revisions 2 (parent's provisional default, ORCHESTRATOR-DERIVED);
rd-parity-decomp revision cap 1 (decomposition). No back-edges. Any addendum node not in this
document fails the addendum's own scope hash and returns as a fresh RULE.

### 1.8 Checkpoint (k-) node — fan-out and tiering caps declared BEFORE expansion

| id | gates | fanout_cap | tier cap (attribute) | per-child / fleet cost cap | declared before |
|---|---|---|---|---|---|
| k-addendum-fanout | tonight's addendum launch: rd-parity-decomp, rd-s0b-parity-vs-c0prime, rd-2d-bootstrap-jackknife, rd-timeout-bin-seed3000, r-sealed-mock + r-b0-finite-moment authoring, two design-gate panels | 5 sonnet identities fixed (J reader; K analyst; L analyst; 2 panel members shared across both design gates); no dynamic expansion; no cluster fan-out | top-tier = 3 for the ADDENDUM workflow: the chair (the same identity as wave 3's chair — disclosed as shared across two concurrent workflows, so the session's live top-tier count is 3 + 2 = 5 if wave 3's prereg author and end-verifier are still alive; the author is asked to accept or serialize, section 3 row A-0) + 1 prereg author (r-sealed-mock, r-b0-finite-moment) + 1 decisive verifier (re-derives J's classification and K's two decisive numbers) | 0 CPU-h tonight (all nodes zero-compute); no launch of m-parity-401grid, m-closure067-headstack, m-sealed-pool, m-b0-finite-moment, rd-timeout-bin-seed61000 | addendum launch (this document) |

---

## 2. Gate panel for the addendum

Same instruments as the parent section 2 (frozen at row #290; no new instrument, no band edit).
Instruments evaluated tonight: g-precision (J, K), g-population (J, K, L), g-censoring (K), the
g-c0-baseline PATTERN (J.2, same-machine rule row #325 item 3 satisfied by md5-matched retrievals).
Panel law unchanged: no rd- or d- node consumes a measure without a green or author-waived stamp; a
red never suppresses the number. Every waives edge is a per-instance author RULE.

---

## 3. Decisions table (addendum branch heads)

One-word replies: Approved grants a DO, Ratified grants a RULE, Granted grants a STANDING row. Binding
default: nothing here covers a disposition whose inputs do not yet exist; every such disposition sits
in its NOT-covered cell and returns via section 1.6.

| # | branch head (node) | tag | ask | triggers on grant | explicitly NOT covered (returns as fresh RULE) |
|---|---|---|---|---|---|
| A-0 | d-addendum-charter (branches J–N as one addendum at its own scope hash) | RULE | Ratified | freezes sections 1.0–1.8 as a second workflow beside wave 3; accepts the shared-chair top-tier accounting of k-addendum-fanout (3 addendum + 2 live wave-3 roles = 5 distinct identities, over the ~3-per-workflow cap only if the two workflows are read as one) OR orders the addendum serialized after wave 3's prereg author and end-verifier retire | every band, classification, or verdict below; any node not in this document |
| A-J1 | rd-parity-decomp + rd-s0b-parity-vs-c0prime (Branch J, zero-compute reads) | DO | Approved (RUNNABLE TONIGHT under row #325 as verdict-free reads; the grant is the ask of record, the row #325 flag is the fallback) | the two reads; the S0-B parity stamp attached to the d-photoz-leverage dossier as a FLAGGED chair decision | d-parity-disposition (the classification's ruling); any caveat on the row #287 certification |
| A-J2 | m-parity-401grid | DO (conditional) | Approved | the 4 local truth-node re-runs at the 401 grid ONLY if rd-parity-decomp lands on (ii); cap 15 CPU-h ORCHESTRATOR-DERIVED (11 + margin) | the anomaly disposition |
| A-K1 | rd-2d-bootstrap-jackknife (Branch K) | DO | Approved (RUNNABLE TONIGHT, verdict-free) | the bootstrap/jackknife read on the re-baseline event set, both venues, both channels | d-2d-frequentist; any edit to d-paper-1d2d-verdict's requires-manifest |
| A-L1 | rd-timeout-bin-seed3000 (Branch L) | DO | Approved (RUNNABLE TONIGHT, local archive) | the local binning with a blind bin-edge gate | d-timeout-bound; the G7 row 8 re-grade |
| A-L2 | rd-timeout-bin-seed61000 | DO | Approved | fetching the `run_20260729_seed61000` `*.log` files from the cluster (a read under /cluster; NOT docket 12b evacuation) and the same binning | same as A-L1 |
| A-M1 | r-sealed-mock (Branch M) | DO | Approved (AUTHORING RUNNABLE TONIGHT) | registration authoring incl. the sealed-draw script design; NO draw is performed, NO pool is generated | d-sealed-register: the prior on h_inj, the pull band, stage (m1) launch (9–94 CPU-h band), stage (m2) pool cost (<= 250 GPU-h + simulate + approx 94 CPU-h, all ORCHESTRATOR-DERIVED) |
| A-M2 | m-closure067-headstack | DO | Approved | the HEAD re-score of the existing 0.67 pool after d-sealed-register, behind /cluster preflight READY | the (m1) disposition; any HALT |
| A-M3 | m-sealed-pool | DO | Approved | the decisive sealed campaign after d-sealed-register; scheduling vs the 2026-09-23 expiry is the author's word | the T-1 verdict; the unseal |
| A-N1 | r-b0-finite-moment (Branch N) | DO | Approved (AUTHORING RUNNABLE TONIGHT; the B-R control on banked CSVs is a zero-compute design gate) | registration authoring; the B-R control evaluated on the stale banked basis as a design gate only | d-b0fm-band; any verdict from the stale basis (forbidden by construction) |
| A-N2 | m-b0-finite-moment | DO | Approved | 12 fresh HEAD seed-pairs (12–20 CPU-h sourced) after d-b0fm-band | the identity verdict; the twin's correctness call |
| A-S | STANDING: the addendum's reads may be attached as FLAGGED caveats/stamps to wave-3 dossiers (d-photoz-leverage, d-paper-1d2d-verdict) without re-opening those dossiers' requires-manifests | STANDING (this night only; lapses at the author's next message, mirroring docket item 2.2) | Granted | attachment as annotations only | any ruling on the annotated dossier; any manifest edit (fails L10 on the parent) |

Decisions that WILL RETURN as this addendum executes (none pre-granted): d-parity-disposition,
d-s0b-parity-stamp, d-2d-frequentist, d-timeout-bound, d-sealed-register, d-b0fm-band.

---

## 4. Deferred candidates (survived the scan; not scheduled tonight) — with reasons

| candidate | reason for deferral (sourced) | what would revive it |
|---|---|---|
| `parked-cmem-outside-cone-deficit` (A14 CMEM re-routing falsifier) | (1) The signal it would falsify is weaker than the scan quotes: row #280 correction (g), verbatim: "[CMEM] A1: bc/bt strata correlate at 0.9994 — the 20-strata permutation p (0.029–0.036) overstates evidence; dependence-respecting p = 0.127; A7/A8 inputs MUST use the seed-level null (n≈10)." (2) A14 is not a re-run of `cmem_a1.py`: PREREGISTRATION_CMEM_A1_20260829.md section 7 defines it as "a future registered arm that re-routes the dropped weight (e.g. adding the out-of-cone in-catalogue term to `B_num`)" — a counterfactual in `bayesian_statistics.py` (physics-trigger file) needing /physics-change and an author word, not tonight's grant. (3) Its object — the true-host-outside-cone class — IS Branch H's cone-loss object (approx 17%, artifact section 09), whose registration r-cone-loss is being authored tonight (row #335 item 5); launching a second arm on the same class before d-cone-register collides. A2's cost band is real but conditional (COMPUTE_LEDGER.md line 49: "B2.2 (105–265) is not triggered"). | d-cone-register ruled; then A14 as a counterfactual switch proposal under /physics-change, scored against the seed-level null (n≈10) — a different power class than the scan's "single-digit CPU-h". |
| `systematics-SB-05` (completion-term realism triad COM-01/03/04) | Not independent of wave 3 (the scan's own flag): it changes the completeness weighting inside the same completion machinery that r-completion-residual registers tonight; GitHub #23 itself says "Do NOT change code while campaign runs" and "Proposed (post-campaign)"; every re-run edits `pixel_completeness.py` / `bayesian_statistics.py` (physics-trigger) -> /physics-change + author word. | The measured gaps (issue #23: f_lum/f_num 4–9x at z=0.1, K=0) are handed to the r-completion-residual prereg author tonight as an INFORM edge (no node): the registration must state whether its band is robust to COM-03's weighting choice. Revive as its own branch after d-completion-register. |

---

## 5. Ranking, cost envelope, and tiering

### 5.1 Ranking (information per CPU-h x independence from wave 3)

| rank | branch | information gained (what settles) | CPU-h tonight | independence | why this order |
|---|---|---|---|---|---|
| 1 | J [PAR] | corrects or confirms a disclosed-but-unadjudicated anomaly under the certified [HIER] instrument (row #273) AND supplies the missing S0-B parity stamp (NO_BANKED_CSV) that d-photoz-leverage would otherwise consume blind | 0 | full (reads banked CSVs only); conditions a wave-3 dossier without touching its arms | highest leverage per hour: two decisive stamps at zero compute, all inputs local |
| 2 | K [NSC] | an independent frequentist axis on the paper's central 2D number (offset -0.0641 re-baseline / -0.0667 08-27; "3.6x its own width") that the mechanism split never tests | 0 | full | zero compute, referee-facing, both venues available locally |
| 3 | L [TMO] | closes G7 row 8 with a measured bound or opens a quantified selection systematic | 0 (seed3000) | full | zero compute locally; seed61000 fetch is the only cluster touch |
| 4 | M [SEAL] | the single decisive anti-tuning test (redteam T-1); its stage (m1) is a cheap flip-survival check of the existing alternative-truth pool | 0 tonight (authoring); 9–94 (m1); heavy (m2) | full | highest novelty, lowest info-per-CPU-h once (m2) is counted; authoring is free tonight |
| 5 | N [B0FM] | the one venue that adjudicates catalogue-leg correctness; the brief's open thread; a vacuous-again outcome pays as a bounded park | 0 tonight (authoring + zero-compute control on banked CSVs); 12–20 (fresh pairs) | full | design gate first; the stale basis caps what tonight can claim |

### 5.2 Cost envelope

| item | cost | source |
|---|---|---|
| rd-parity-decomp, rd-s0b-parity-vs-c0prime, rd-2d-bootstrap-jackknife, rd-timeout-bin-seed3000 | 0 CPU-h (local pandas/numpy on banked CSVs) | data paths in sections 1.1–1.3, all verified present locally |
| m-parity-401grid (conditional) | approx 11 CPU-h, cap 15, ORCHESTRATOR-DERIVED | 32.08 CPU-h / 12 seed-nodes (row #273) x 4 |
| rd-timeout-bin-seed61000 | log transfer only; size NOT sourced | DATA_INVENTORY.md line 277 |
| m-closure067-headstack | 9–94 CPU-h ORCHESTRATOR-DERIVED band | state candidate 11 (84 tasks x approx 6.5 min) vs row #285 (approx 94 CPU-h, 55 nodes) |
| m-sealed-pool | <= 250 GPU-h injection pool + N_tasks x <= 0.5 GPU-h simulate + approx 94 CPU-h evaluate, ORCHESTRATOR-DERIVED upper bounds | datasets.yaml lines 39–40; inject.sbatch line 19; simulate.sbatch lines 36–38; row #285 |
| m-b0-finite-moment | 12–20 CPU-h | PREREGISTRATION_B0_IDENTITY_20260823.md lines 173–174 |
| registrations, reads, verifier | agent time, negligible compute | — |

Envelope TONIGHT: 0 CPU-h. After rulings: sourced items 21–114 CPU-h (m1 + b0 pairs); the sealed
pool is the only heavy item and is deliberately gated behind its own RULE and the Sep-23 expiry.

### 5.3 Tiering (routing table of record: CLAUDE.md)

| role | model / effort | nodes | justification |
|---|---|---|---|
| chair | inherit (shared with wave 3) | dossier attachment of J's stamps; re-derives the two S0-B max_abs numbers | orchestration |
| prereg author | inherit / xhigh | r-sealed-mock, r-b0-finite-moment (batched, one identity) | pre-registration authoring is a listed top-tier use |
| decisive verifier | inherit / xhigh | re-derives rd-parity-decomp's classification and rd-2d-bootstrap-jackknife's two decisive numbers | "verifier output is evidence, not authority" — every decisive number re-derived |
| J reader, K analyst, L analyst | sonnet / high (J, K), medium (L) | the four zero-compute reads | mechanical stats on existing CSVs |
| design-gate panel | sonnet / medium x 2 (shared across both registrations) | r-sealed-mock, r-b0-finite-moment | panel redundancy substitutes for tier |
| cluster arrays (after rulings) | sonnet / low | m-closure067-headstack, m-sealed-pool, m-b0-finite-moment, m-parity-401grid | running existing scripts on new inputs |

Top-tier count for the addendum workflow: 3 (chair + prereg author + decisive verifier) — at the cap.
Disclosed: the chair identity is shared with wave 3, whose own cap-3 roster (docket section 1) may
still be live; row A-0 asks the author to accept 5 distinct live identities across the two
workflows or to serialize the addendum's two non-chair roles after wave 3's retire.

---

## 6. EXECUTION ORDER for tonight

Legal basis: row #325 (chair "can make decisions but flag them"); row #334 item (8) ("the candidate
scan is authorized ... Any Graph 1 addendum branch returns for ratification at L10"). Reading of
record: registration AUTHORING and VERDICT-FREE READS of already-banked local data are within the
grant (they create no cluster job, touch no source file, bank no verdict); every launch, every
physics-trigger edit, every band and every disposition is NOT.

### 6.1 Start immediately — no author ruling needed (all flagged in their ledger rows)

| order | node | why it is inside the grant | output tonight |
|---|---|---|---|
| 1 | rd-s0b-parity-vs-c0prime (J.2) | zero-compute diff of two retrieved, md5-matched CSVs; feeds the d-photoz-leverage dossier the chair is already assembling under docket 2.1 | a GREEN/STOP parity stamp, attached to the dossier as a FLAGGED chair annotation (STANDING A-S asked; the row #325 flag is the fallback) |
| 2 | rd-parity-decomp (J.1) | zero-compute read of banked T1.3-zwin / T1.2 / bc CSVs; the classification is banked verdict-free; the decisive verifier re-derives it | a per-event decomposition table + the (i)/(ii) classification as a RECOMMENDATION for d-parity-disposition |
| 3 | rd-2d-bootstrap-jackknife (K) | zero-compute on the re-baseline event set; verdict-free; both decisive numbers re-derived | bootstrap-SE/sigma_h ratio, minimal-subset fraction, width-vs-N table; a RECOMMENDATION for d-2d-frequentist |
| 4 | rd-timeout-bin-seed3000 (L.1) | local archive; bin edges frozen by a blind design gate before rates are read | binned rates + Poisson intervals; population depth disclosed; a RECOMMENDATION for d-timeout-bound (partial) |
| 5 | r-sealed-mock authoring (M.r) | pre-registration authoring; no draw, no pool, no code path executed | the registration draft + its design-gate record; d-sealed-register dossier for the morning |
| 6 | r-b0-finite-moment authoring + the B-R control on banked CSVs (N.r) | authoring + a zero-compute design gate on stale data (explicitly verdict-free by construction) | the registration draft + the B-R control result; d-b0fm-band dossier for the morning |

Ordering rationale: 1 before the d-photoz-leverage dossier is finalized (wave-3 item 2); 2–4 are
independent and run in parallel (three sonnet identities); 5–6 are one top-tier identity, sequential.

### 6.2 Needs ratification FIRST (returns in the morning docket)

| node | blocking ruling | what the ruling needs in hand |
|---|---|---|
| m-parity-401grid (J.3) | A-J2 + d-parity-disposition landing on (ii) | the rd-parity-decomp table |
| rd-timeout-bin-seed61000 (L.2) | A-L2 (a cluster read; the author's row #334 words put cluster data handling on the backlog — this fetch is analysis input, not evacuation, but the word is the author's) | the seed3000 partial result |
| m-closure067-headstack (M.m1) | A-M2 + d-sealed-register + /cluster preflight READY | the registration + its band |
| m-sealed-pool (M.m2) | A-M3 + d-sealed-register + a scheduling word vs the 2026-09-23 expiry | the pool cost derived from the cluster directory |
| m-b0-finite-moment (N.m) | A-N2 + d-b0fm-band (only if the B-R control FAILED as designed) | the control result |
| any caveat on the row #287 certification; any G7 row 8 re-grade; any manifest edit on a paper node | fresh RULEs (section 1.6) | the respective read records |

---

## Appendix A — refuted at scan (skeptic verdicts, condensed; sources as given by the skeptic and spot-checked here)

| candidate | skeptic's reason (condensed) |
|---|---|
| `parked-lognormal-massinfo-stress-test` | Exonerated by mechanism: docs/SIGMA_Z_SIGMA_M_FORECAST.md section 4 caveat 2(c) — "a linear-Gaussian (not log-normal) host-mass kernel — but at realistic σ_M the anchor width σ_M(1+z) ≈ 1.1 ≫ z ≈ 0.15, so the anchor is uninformative regardless of kernel shape"; three-lens verified. A footnote, not a question. |
| `ledger-cmem-r2c-highpower` | Row #280 correction (g): dependence-respecting p = 0.127 (bc/bt strata correlate at 0.9994); the null is seed-level (n≈10), so "higher power by re-running the same instrument" is mechanistically false — a different, much more expensive design. |
| `systematics-SB-02` | Kill criterion executed by the skeptic: 522/524 `run_metadata*.json` under realistic_20260729 post-date fix 49251f38; the 2 pre-fix files are a frozen 2026-07-10 A/B snapshot (same-PSD differential), not the production pool. CLEAN. |
| `systematics-SB-03` | Already measured: G2d derivation + docs/gates/G7row9_N5_postDgfix_SUMMARY.md (Δmean_2d = −0.00218) + PREREGISTRATION_TILT_BATTERY.md R-E instrument (s_Edd,new +0.0012 iiib / +0.0019 joint_r1 vs 0.008 materiality) — immaterial; only the G7 row 9 table cell is stale. |
| `systematics-SB-06` | Premise refuted: results/lcat_h_dependence_20260725/INFORMATION_FLOOR_PREREGISTRATION.md — the 5-ln floor was pre-registered and author-ratified 2026-07-26; issue #44 stays open by design as the revisit tracker; the floor is "NOT applied retroactively", so a retroactive audit needs its own scope decision first. |
| `paper-pg-1d-grid-extension` | Moot: the post-flip production run (row #286) moved the 1D MAP from the 0.600 rail to 0.665 (mean 0.667) on the unchanged 41-node grid; the residual −0.063 is owned as the mass-blind/mass-aware mismatch (STATE line 15); G-EXT touched only the upper bound and was verdict-irrelevant at tail 5e-13. |
| `paper-pg-massrelation-scatter-sensitivity` | Already swept: MASS_RELATION_ASSESSMENT.md section 2 (0.08/0.24/0.50/0.55 dex -> σ_M 0.19/0.60/1.66/1.99) and SIGMA_Z_SIGMA_M_FORECAST.md section 3 (0.3/0.4/0.5 dex); every value is ≥ 10x above the ~1–2% pay-off threshold; section 3.6 kills the rescue independently via confusers. |
| `paper-pg-b0-coverage` | (1) The equivalent instrument (`pp_coverage.py`, `catalogue_mode=True`) already ran in the prodcal ladder (rows #120–#124): H-P FAIL at V-deep with the mechanism owned (first-order tilt·slope·σ², 3% agreement) and production's flat-S̄ regime certified by three measurements. (2) b0 is native to `correspondence_1d.py`, not `pp_coverage.py` — new instrument work the cost band ignores. |
| `paper-pg-literature-convention-audit` | Already done and re-checked post-fusion: docs/gates/G5a_gwcosmo_inspection.md, G5b_chimera_icarogw_inspection.md (P1/P2/P3 tables), LITERATURE_WARNINGS.md "G5b staleness note" (P3 re-check row #120), PROPOSAL_GRAY_CONVENTION_PAPER_INTEGRATION_20260817.md section 7 "G-3 COLLECTED ... Verdict: MATCH". |

## Appendix B — source index for every quoted number (verified by opening the file)

- Ledger (`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`):
  preamble section 4 item 7 (line 218); rows #177, #178, #179, #180, #186, #250, #255, #273, #280 (g),
  #284, #285, #286, #287, #299, #302, #313, #325, #328, #334, #335.
- `tree2_20260830/hier_s0_zwin_run/s0a_score_output.json` (`gate_parity`, 4 seeds, both channels);
  `fanout1_20260829/hier_s0_registered_run/s0a_score_output.json` and `tree2_20260830/hier_s0_recert_run/s0a_score_output.json`
  (pre-window baselines, identical); `tree2_20260830/T1_3_ZWINDOW_P1_READOUT_RECORD.md` section 6;
  `tree2_20260830/PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md` lines 401–403, 472–474;
  `fanout1_20260829/B1_1_S0A_DEFECT_FORENSIC_20260829.md` line 94 (E19).
- `graph1_20260901/retrieved/s0b_run_20260902/s0a_full_output.json` (config, `gate_parity: NO_BANKED_CSV`);
  `graph1_20260901/retrieved/run_20260902_graph1_c0prime_headrebaseline_iiib/...` and
  `.../run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv` (65 108 rows).
- `docs/CLAUDE_SCIENCE_BRIEF.md` (prepared 2026-08-29) lines 82–83, section 3 table, lines 162–163.
- `docs/gates/G7_systematics_budget.md` rows 2/8/9; `docs/gates/G9_timeout_scan.md` lines 106, 120;
  `darksiren_emri/main.py` lines 763–770, 1293–1302; `results/_archive/run_20260707_seed3000/`
  (99 logs; 1 198 timeout records; CRB 3 325 rows; `run_metadata_16.json`).
- `results/redteam_20260726/PHYSICS_METHODOLOGY_REVIEW.md` lines 401–413; GitHub #39 (OPEN, 0 comments,
  milestone "Paper Submission"); `docs/derivations/realistic_host_observation_model.md` lines 532–533;
  `results/campaign51_20260728/realistic_20260729/closure_seed64000_h0p67/combined_posterior_2d.json`;
  `REALISTIC_READOUT.md` section 7; `RUNBOOK_NEXT_SESSION_5.md` line 16; `cluster/datasets.yaml`
  lines 36–41, 137; `cluster/inject.sbatch` line 19; `cluster/simulate.sbatch` lines 36–38;
  `cluster/evaluate_closure_h065.sbatch` header; `DATA_INVENTORY.md` line 277 and the
  `p3_b0_identity_fleet_20260823` entry.
- `CLAIM_B0_FINITE_MOMENT_20260824.md` header + section 0; `PREREGISTRATION_B0_IDENTITY_20260823.md`
  lines 173–174; `p3_b0_work/` (24 banked seeds, local).
- `fanout1_20260829/PREREGISTRATION_CMEM_A1_20260829.md` section 7; `B2_1_CMEM_A1_RECORD.md` line 108;
  `fanout1_20260829/COMPUTE_LEDGER.md` line 49; GitHub #23 body (COM-01/03/04, "Do NOT change code
  while campaign runs"); `RUNBOOK_NEXT_SESSION_38.md` line 57.

## RATIFICATION (2026-09-04)

Per `MORNING_DOCKET_20260904.md` and ledger row #367 (author "all approved, granted, ratified"):

- A-0 — RATIFIED.
- A-J1 reads done: J.1 EXPLAINED-BY-DESIGN; J.2 NOT-EVALUABLE-pending-comparand → superseded by
  R4/R4b.
- K read done: not diffuse; 3–6 % subset; superseded by `r-offset-subset`.
- L.1/L.2 done, with the p0 axis withdrawn (D1 bound) and the M axis ~1 %/draw (rows #342/#355/#358).
- `r-sealed-mock` dossier ratified-by-chair; m1 job 6790859 FAILED at a guard (row #367, under
  investigation).
- `r-b0-finite-moment`: close on C-A (row #342).
