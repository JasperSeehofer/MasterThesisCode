# r-completion-residual — REGISTRATION DRAFT: the first registered arm on the dark-class completion-leg residual

Date: 2026-09-03 (evening). Node: r-completion-residual (Research Graph 1, Branch G, wave 3) — **DRAFT**.
Author of record for all scientific decisions: Jasper Seehofer.
Status: **PROPOSED THROUGHOUT — nothing here is frozen.** Authorization: charter row #290 decisions-table
row 9 (registration AUTHORING only) + docket `DECISION_DOCKET_WAVE3_20260903.md` item 2.1 (approved).
Band + cap ratification returns as fresh RULE **d-completion-register**; launch only under docket 2.2
(design gate GREEN; cap ≤ 80 CPU-h). max_revisions 2 (ORCHESTRATOR-DERIVED, charter §1.7/§1.13).
Research-cycle stages 0–2 applied (`.claude/skills/research-cycle/SKILL.md`); stage-1 forecast in
`INFORMATION_FORECAST.md` (same directory). Append-only after commit.

Inputs of record: F (rd-s3-readout, `exec/m-s3-postflip-coverage/CHAIR_REDERIVATION_20260903.md` §1/§4);
the re-baseline (`exec/m-head-rebaseline/READOUT_RECORD.md`, rows #298/#299/#302); the B4.3 derivation
(`tree2_20260830/B4_3_MIXTURE_WEIGHT_DERIVATION_20260830.md` §4.4, §7); the primer
(`docs/PRIMER_BIAS_CHANNELS_20260822.md` §0–§3); the S3 harness (`tree2_20260830/b8_cal_harness.py`,
row #291 state) and its 67 post-flip cell-S checkpoints.

## 0. Binding premises

1. **F is a context number, not a licence.** F = 11.44 (no_bh, N = 200) was booked DEFECT-SIGNATURE
   (KS OUTSIDE both channels; catalogue-hosted score Z 9.76; 15–24 % upper-rail censored; the harness
   universe centres +0.042 ABOVE truth while production sits −0.063 BELOW) — chair re-derivation §3/§4.
   Nothing in this arm's discrimination consumes coverage validity. Every statistic below is a
   **score at truth** (research-cycle rule 13, [A12]) carrying its own closure identity (§2). The one
   h-space translation that uses F is REPORTED-ONLY and flagged as such (§3.4).
2. **The −0.14/event object is re-measured before it is decomposed** (§1.2): its provenance is weaker
   than the charter card implies.
3. **Zero fresh choices at launch**: the two decisive reads are re-reads of banked artifacts (rule 9,
   [A1]); the only build is the scorer (§7, a b- node). The optional replication cell uses a disjoint
   seed block and the production-identical harness CLI (§6).

## 1. Object and provenance audit

### 1.1 The object (stage-0 claim intake)

**Claim pair (charter §1.0):** c-residual-illegitimate ("the dark-class completion-leg residual is
estimator inconsistency — the estimator disagrees with its own model") vs c-residual-floor-consistent
("it is noise at the information floor, given F"). Both conjectured; jointly discriminated here.

**Registered definition of the residual.** On the stencil (h_lo, h_hi) = (0.725, 0.735), Δh = 0.01
(B4's `per_event_scores`, `fanout1_20260829/b4_imp_stage1_forecast.py:136-143`; the harness's
`_score_at_truth_by_class`, `b8_cal_harness.py:1183-1214`), for a **dark** event e (truth host not in
the catalogue: `host_galaxy_index = −1`, `in_catalog = False`):

    s_M,e  =  [ ln B_num(e, h_hi) − ln B_num(e, h_lo) ] / Δh  −  [ ln β̄_Ḡ^φ(h_hi) − ln β̄_Ḡ^φ(h_lo) ] / Δh

i.e. the **matched-channel score** (primer §2: the dark-conditional posterior `B_num/β̄_Ḡ_φ` — "the
completion leg alone, correctly normalised"). `D_res ≡ S_M = mean_e s_M,e` over the dark class. For a
consistent estimator on data from its own dark law, E[s_M,e] = 0 (it is the score of a normalised
conditional likelihood). **This is the quantity B4.3 §4.4 called "the dark-class completion leg
scoring ≈ −0.15/event below the estimator's own model."**

`Refute by:` (rule 3) the cheapest decisive falsification of c-residual-illegitimate is the same
statistic on the S3 harness universes (data drawn from the estimator's own law): |Z_harn| ≤ 3 refutes it
at the harness's SE. The cheapest falsification of c-residual-floor-consistent is |Z_prod| > 3 on the
production re-baseline. Both are zero-compute (§2.3).

### 1.2 Provenance audit of "≈ −0.14/event" — VERDICT: WEAKER THAN THE CHARTER IMPLIES

| hop | what the record actually says | tag |
|---|---|---|
| charter §1.0/§1.7, docket item 3 | "the approx −0.14/event dark-class completion-leg residual (artifact section 09)" | [DOC] |
| "artifact section 09" (charter wording) | the board card is not a git-tracked source; its only git-tracked antecedents are B4_3 §4.4 (the derivation, next row) and ledger row #261 (`gate_b_20260730/BIAS_HISTORY_LEDGER.md:3144`) — cited here in its place | [DOC] |
| ledger row #261 (`gate_b_20260730/BIAS_HISTORY_LEDGER.md:3144`) | "not to truth — a separate ~−0.14/event completion-leg residual remains, routed to B8 [CAL]" — a parenthetical inside the A18 prediction | [DOC] |
| **B4_3 §4.4 (the only derivation)** | production dark-only pure arm at 0.7134, σ 0.0277 (C5) ⇒ total dark pure score ≈ −22 nats/h (−0.014/event); the model's composition tilt +0.1326/event × 1514 = +201 ⇒ shortfall (−22 − 201)/1514 ≈ **−0.147/event** (the text says "about −0.15 per event") | [INFER] from [DOC] + ARITH |
| C5 itself (`fanout1_20260829/CLAIM_IMPOSTOR_DRAG_20260829.md:202`) | "`[LOCAL; ASSUMPTION-JOIN — secondary until validated]`" — the dark/in-catalogue split was a CSV row-order join | [DOC], flagged secondary |
| configuration of record | `headreadout_20260827` = PRE-FLIP, mass-blind 1D leg; the flip `5e7fda16` (row #286) changed exactly this channel | STALE under [A11] |

Findings: (i) the number was never measured as a per-event score — it is a difference between a
posterior-derived total and a model tilt, on an ASSUMPTION-JOIN class split; (ii) it is pre-flip on the
channel the flip changed — under rule 12 ([A11]) it may not be quoted as a point value; (iii) the mirror
FT fleet's `s_full = −0.1470` (B4_3 §4.2) is a **different quantity** (full score, mirror venue) that
coincides numerically — the two must never be conflated; (iv) the other "0.14" hits in the ledger
(row #130 B_scale +0.12/+0.14; rows #217/#218 WGEOM −0.145) are unrelated objects. **Consequence:** every
band below is set in units of the arm's own standard error, never relative to −0.14; the arm's first
deliverable is the post-flip measured value of D_res with its SE.

### 1.3 Exoneration check (both layers, mechanism-grepped — memory `rule1-exoneration-check-insufficient`)

Swept `CLAIM_2D_BIAS_20260730.md:721-757`, `BIAS_HISTORY_LEDGER.md:127-171`, `EXONERATION_REGISTER_20260827.md`
§1–§2 for the MECHANISM "completion-leg numerator/normaliser pairing; dark-conditional score; B_num as
carrier": (a) **matched-channel violation −0.085 — RESOLVED by the `fused` fix (O6–O8, primer §3)**: this
arm does not re-litigate the fix; it measures whether a residual SURVIVES it post-flip (what B4.3 §4.4
hands to B8). (b) §2 item 10 [LCOMP-BNUM-DEFECT] "B_num as a defective integral — exonerated by
self-consistency MC (#80)": T_harn (§2.3) is itself a self-consistency read on a production-faithful
harness that #80 did not have; a |Z_harn| > 3 outcome is NEW EVIDENCE and re-opens #80 by the standing
rule, not a re-litigation. (c) HA "completion term not mass-marginalised" — upheld and decomposed, not
this object. (d) [INFO-STARVATION], [WINDOW-MEMBERSHIP], HB — different mechanisms. **Not exonerated.**
R0 sweep: Gray et al. 2020 / Mandel-Farr-Gair 2019 rows already in `docs/LITERATURE_WARNINGS.md` via [CMEM];
nothing new.

## 2. The decomposition identity (g-closure) and the reads

### 2.1 Per-event identity (exact on the CSV columns)

`event_likelihoods.csv` writes `ln L_e(h) = num_log_term_no_bh − den_log_term` exactly
(`bayesian_statistics.py:6800-6803`; `den_log_term` is global per h — verified unique per h on the
re-baseline CSV). With `num = β·L_cat + B_num` (primer §0), define on the stencil:

| term | definition | meaning | zero when |
|---|---|---|---|
| s_M,e | Δ ln B_num(e) / Δh − Δ ln β̄_Ḡ^φ / Δh | matched-channel (completion-leg-alone) score | consistent dark law |
| s_T | Δ ln β̄_Ḡ^φ / Δh − Δ den_log_term / Δh | global composition tilt (event-independent) | — |
| s_C,e | Δ [num_log_term_no_bh(e) − ln B_num(e)] / Δh | catalogue-leg increment (the impostor drag) | L_cat = 0 (38.2 % of stencil rows on production) |

**Identity:** s_M,e + s_T + s_C,e = Δ num_log_term / Δh − Δ den_log_term / Δh = s_e (the full score).
**g-closure gate:** max over events of |s_M,e + s_T + s_C,e − s_e| ≤ 1e-9·(|s_e| + 1) — an identity, so a
miss localises a storage-precision defect (g-precision red), never a physics read. Class closure:
S_all = π_G·S_G + π_Ḡ·S_dark with π from the class counts (0 unmatched events).

**Why β is never reconstructed:** a naive `β = alpha_G_phi/r_Malm` rebuild (B4.1's `matrices()`)
reproduces `num_log_term` to 1e-15 on the median row but fails by up to 1.61 relative on the
mass-aware-flipped candidate-bearing rows (checked on the re-baseline CSV, 2026-09-03) — the flip put
`S_4D` into the catalogue term. The identity above uses only the exact columns and is flip-agnostic.

`β̄_Ḡ^φ(h)` operational source: `D_tilde_phi − alpha_G_phi` from the CSV (7 s.f. storage; induced error
on s_T ≤ 2e-5 per unit h, immaterial at the 1e-2 band scale — disclosed). g-precision cross-check where a
full-precision `selection_tables_h_*.json` exists (harness dirs carry `beta_Gbar_phi`, `sigma_phi`): the
two must agree to 1e-3 relative, else disclose and use the column definition (closure is defined on
the columns).

### 2.2 Read A — production, post-flip (zero compute)

Data: `graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/
event_likelihoods.csv` (41 × 1588 rows; commit `1ec9514d`; g-c0-baseline GREEN-AS-CORRECTED, row #302).
Class label: `seed61000/prepared_cramer_rao_bounds.csv` (md5 `9a1f2a14384a9281c97ca3be312ddaab`), column
`host_galaxy_index` (−1 = dark), joined on `event_idx` = CRB row index — **JOIN gate**: the CSV's
`event_idx` set is {0..1589} minus {1203, 1356} (2 unscored rows; `fisher_quality.csv` excluded = 0), and
the in-catalogue count must equal the P6 denominator 76 (`darksiren_emri_20260902_000633_h_0_73.log:8622`:
"1D 66/76 hosts recovered/in-cat events seen"). This closes C5's ASSUMPTION-JOIN caveat by construction.
Outputs: S_M,prod, s_T,prod, S_C,prod (dark), the same three for the catalogue-hosted class, S_all, and
**SE_prod = SD_e(s_M,e | dark, production)/√N_Ḡ** (N_Ḡ = 1512) — the per-event SD is that of the
MATCHED-CHANNEL score on the PRODUCTION dark class, computed by the registered read itself; no
harness-borrowed proxy enters SE_prod (rev. 1 item 1). Venue replicate (REPORTED): joint_r1 CSV, same script.

### 2.3 Read B — the S3 harness, cell S, post-flip (zero compute)

Data: 67 per-universe CSVs `tree2_20260830/b8_cal_harness_work_s4_postflip/seed9010NN_S/simulations/
diagnostics/event_likelihoods.csv` (7216 rows = 176×41 each) + each universe's `prepared_cramer_rao_bounds.csv`
(`host_galaxy_index`). Population `n200-postflip` (rd-s3-readout §1: 0 mixed rows; seeds 901000–901066;
resolved flags 13 tokens, checkpoint `resolved_flags`). Sampling unit = universe (PA-HIER-5): T_harn =
mean of the 67 per-universe dark S_M; SE_harn = SD/√67. **Byte-id gate for the instrument:** the script's
per-universe dark FULL score must reproduce `score_at_truth.no_bh.dark.mean` in all 67 checkpoints
bit-for-bit (same convention, same machine).

Two SEs, kept separate (rev. 1 item 2): (i) **SE_full,harn = 0.0063** — the harness FULL-score
between-universe SE (per-universe dark mean +0.0082, SD 0.0517, 67 universes, 11,525 dark events;
raw checkpoints `score_at_truth.no_bh.dark`; reproduced exactly, BYTEID_RECORD.md) — INFORMATIONAL
only, a design-power proxy; (ii) **SE_harn = SD_U(S_M,harn,U)/√67** — the matched-channel
between-universe SE, the registered statistic's OWN SE, computed by the registered read; it is
expected to be LARGER than (i) (the matched-channel score has the larger per-event spread) and it,
not (i), enters Z_harn. Per-event SD proxies quoted anywhere in this draft (0.68) are harness
FULL-score values and are labelled "harness-borrowed proxy".

### 2.4 Registered statistics

| symbol | definition | null |
|---|---|---|
| T_prod | S_M,prod (dark, iiib) | 0 |
| Z_prod | T_prod / SE_prod, SE_prod = SD_e(s_M,e \| dark, production)/√1512 | N(0,1) |
| T_harn | mean over universes of S_M,harn | 0 |
| Z_harn | T_harn / SE_harn, SE_harn = SD_U(S_M,harn,U)/√67 (matched-channel, per-universe) | N(0,1) |
| ρ | T_harn / T_prod, evaluated only when \|Z_prod\| > 3 | — |
| δh_M (REPORTED-ONLY) | N_Ḡ·T_prod / I_1D, I_1D = 1/σ_h,1D² = 1/0.017526² = 3256 (re-baseline iiib 1D) | linear response, F-free |

## 3. Bands (ORCHESTRATOR-DERIVED) and the false-fail rate

- |Z| ≤ 3 is the panel's standing null band (rows #225/#251/#287; g-score-null). One-sentence derivation:
  it is the same band every score-zero read on this board uses, so the two claims are judged on the
  same scale as the S3 class-resolved scores.
- ρ ≥ 0.5 = ILLEGITIMATE threshold: the estimator's own inconsistency owns the majority of the residual.
  ρ ≤ 0.2 = not-illegitimate threshold: the un-owned remainder is at least 4× the owned part (0.2 ≈ the
  materiality ratio T_mat/|offset| = 0.008/0.063 = 0.13 rounded up to one-in-five).
- False-fail under the null: two |Z| ≤ 3 tests ⇒ ≤ 0.54 % joint (2 × 0.27 %) — this rate depends only
  on the band, not on SE sourcing. Power (rev. 1 item 2): stated as formulas filled by the registered
  read — the smallest illegitimate component detectable at 3σ is 3·SE_harn (matched-channel);
  using the INFORMATIONAL full-score proxy SE_full,harn = 0.0063 that would be ≈ 0.02/event (≈ 14 % of
  −0.14), and a 0.07/event component ≈ 11σ; with the matched-channel SE_harn (larger) both figures
  degrade in proportion SE_harn/0.0063. The read reports 3·SE_harn and 3·SE_prod explicitly as the
  arm's realised detection floors.
- 3.4 The h-space context read (leans on F — REPORTED, never verdict-bearing): "floor-consistent" in
  h-units means |mean_h − 0.73| ≤ 3·F·σ_floor(1588) = 3 × 11.44 × 0.001747058397810697 = **0.0600**
  against the measured −0.0630 (iiib 1D, row #302) — a 5 % excess, i.e. the object sits at the edge of the
  F band. Carried with the g-censoring disclosure (§5): F is upper-rail-censored 15–24 % and DEFECT-context.

## 4. Disposition table (three-valued; every row returns as a fresh RULE — nothing self-ratifies)

| disposition | trigger | claim writeback | stage-5 action (returns to author) |
|---|---|---|---|
| **ILLEGITIMATE** | \|Z_harn\| > 3 AND ρ ≥ 0.5 | c-residual-illegitimate SUPPORTED; c-residual-floor-consistent REFUTED | `/physics-change` intake on the completion-leg normalisation (fresh RULE); re-opens ledger §2 item 10 with this evidence |
| **FLOOR-CONSISTENT** | \|Z_harn\| ≤ 3 AND \|Z_prod\| ≤ 3 | c-residual-floor-consistent SUPPORTED; c-residual-illegitimate REFUTED with bound \|T_harn\| + 3·SE_harn | report the bound; q-completion-residual settles (charter kill criterion NOT the path — this is a clean result) |
| **INTERMEDIATE (a) harness-clean, production-displaced** | \|Z_harn\| ≤ 3 AND \|Z_prod\| > 3 | NEITHER claim supported; c-residual-illegitimate bounded (\|T_harn\| + 3·SE_harn); the residual is a generator–estimator population mismatch | routed to d-residual-attribution as the THIRD bucket ("irreducible venue physics / population misspecification") with T_prod ± SE_prod and δh_M — the expected branch (forecast) |
| **INTERMEDIATE (b) partial** | \|Z_harn\| > 3 AND 0.2 < ρ < 0.5 | both claims partial; split quoted | fresh RULE: replication cell R (§6) within the cap, or park with the split |
| **INTERMEDIATE (c) minor-illegitimate** | \|Z_harn\| > 3 AND ρ ≤ 0.2 | c-residual-illegitimate SUPPORTED-BUT-IMMATERIAL | fresh RULE: physics-change intake deferred; residual attribution proceeds on (a) |
| **NO-READ** | g-closure red, JOIN gate red, byte-id red, g-population red, **g-znorm red** | nothing banked | INSTRUMENT-DEFECT: repair; revision counter not consumed |

Sign convention: T_prod < 0 is the §4.4 direction; a positive T_prod is booked by the same rows (the
bands are two-sided). Neither-band ⇒ INTERMEDIATE ⇒ fresh RULE (charter §1.7 row).

## 5. Gates consumed (graph §2 panel)

- **g-closure** — §2.1 identity ≤ 1e-9; class closure S_all = π_G S_G + π_Ḡ S_dark exact. Red STOPs the read.
- **g-population** — harness: 0 mixed rows (`--population 200`, seeds 901000–901066, resolved-flag token
  per checkpoint); production: 1588 rows at every h-node, event_idx gaps exactly {1203, 1356}, in-cat = 76.
- **g-precision** — full-precision columns only for s_M, s_C (B_num, num_log_term, den_log_term); the
  7-s.f. columns enter only s_T with the bound disclosed; the selection-table cross-check where available.
- **g-censoring** — score reads are grid-interior (stencil 0.725/0.735) and carry no rail exposure.
  **Rail-fraction disclosure rule:** any h-space quote (δh_M, the §3.4 F band) MUST carry the S3 rail
  fraction (S no_bh 10/67 = 14.9 %, with_bh 14/67 = 20.9 %; all at the upper rail 0.86) and the production
  MAP (0.665, interior); a quote without the disclosure is void.
- **g-znorm** — standing on the flipped leg (row #292); registered check: at every h-node,
  max_e |den_log_term(e,h) − den_log_term(0,h)| = 0 EXACTLY (it is one `math.log` per h-node,
  bayesian_statistics.py:6800-6803), in both venues and in every harness universe; any nonzero
  ⇒ **g-znorm red ⇒ NO-READ** (rev. 1 item 3).
- **g-byte-id (instrument)** — §2.3: 67/67 harness dark full-score means reproduced bit-for-bit
  (GREEN, BYTEID_RECORD.md); the T0 re-baseline anchor is re-stated (rev. 1 item 6): the ONLY committed
  source is the 6-dp display `mean_h = 0.666987` (`exec/m-head-rebaseline/READOUT_RECORD.md:40`), so the
  tolerance is **1e-6 on that display value**; the full-precision value `0.6669869414473403`
  (independently computed twice, BUILD_RECORD.md and BYTEID_RECORD.md, from the retrieved CSV) is
  banked here as the full-precision anchor with tolerance 1e-12 for every later re-run. A literal 1e-9
  against the display value was unsatisfiable by construction and is withdrawn.

Invariants ([A10], one line each): stencil (0.725, 0.735) — audited 2026-08-29 (B4.1) · H_GRID_41 and
h_bounds (0.60, 0.86) — audited row #303 · catalogue md5 `c52c13b5cab61f6b3f04bbe202550969` — audited
rd-s3-readout §0 · CRB md5 `9a1f2a14…` — audited LAUNCH_RECORD · production commit `1ec9514d` (5e7fda16
ancestor) — row #294 · harness commit `7e9e1e27` with 1112 dirty paths in the stamp — **NEVER audited
against the production commit** (conditional-on: the harness's resolved 13 flags equal production's
CoR-P CLI; asserted from the checkpoint `resolved_flags` block by the script, else NO-READ).
**Structural blindness:** a defect shared by the harness generator and the estimator (the D1 class)
cancels in T_harn by construction; and INTERMEDIATE (a) cannot distinguish "true venue physics" from
"an estimator dark-population law that is wrong for the real universe" — both are generator–estimator
mismatch, and only S0-B/cone-loss inputs at d-residual-attribution split them further.

## 6. Cost against the ≤ 80 CPU-h cap (ORCHESTRATOR-DERIVED)

| item | compute | derivation |
|---|---|---|
| Reads A + B (§2.2–2.3) | 0 cluster; ≈ 0.2 CPU-h local | 67 × 7216-row + 2 × 65,108-row CSV reads + one 55-s catalogue-free join; nothing evaluated |
| Optional replication cell R (only on INTERMEDIATE (b) / ILLEGITIMATE, chair-decided under docket 2.2) | 30 universes × N = 200, cell S, full 41-node grid, **seed block 903000–903029** (disjoint from 901000–901099 / 902000–902024 / 901100+) | anchor: S cell 67 universes in 87,016 s wall (rd-s3-readout §0) ⇒ 1,299 s wall/universe at `--workers W`; CPU-h ≈ 30 × 1299 × W / 3600 = 10.8·W ⇒ W = 2: **22 CPU-h**; W = 4: 43 CPU-h — inside the cap, local CPU only, zero cluster (design §6 convention) |
| production timing reference (context) | 84 tasks × ~6.5 min = single-digit CPU-h (m-head-rebaseline LAUNCH_RECORD:113-118) | not consumed: no production re-evaluation is needed |

The grid MUST stay 41-node: the candidate z-window is built from `h_bounds = (min, max)` of the h list
(`b8_cal_harness.py:1278,1361`); a 3-node stencil-only run would shrink the window and change the estimator.

## 7. Launch block (zero fresh choices)

**Build node (waits):** `b-completion-scorer` (sonnet/medium) writes
`graph1_20260901/exec/r-completion-residual/completion_residual_reads.py` implementing §2.1–§2.4 with the
gates of §5; builder runs ONLY `--dry-run` (gates + closure + byte-id, no statistic); a DIFFERENT agent
runs it (standing rule 2; memory `agent-verifier-output-is-evidence-not-authority`). Launch waits on the
byte-id gate GREEN (67/67 + T0 mean_h at the §5 tolerances) — stamped GREEN in BYTEID_RECORD.md.

    # from REPO ROOT (runbook 42 §5 gotcha)
    uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_residual_reads.py \
      --production-csv results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv \
      --production-crb results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
      --replicate-csv results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv \
      --harness-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip \
      --population 200 --h-lo 0.725 --h-hi 0.735 --h-true 0.73 \
      --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
      --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_residual_result.json \
      [--dry-run]

Optional cell R (chair decision under 2.2 only; otherwise parked for the author):

    uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
      --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_r_completion \
      --N 200 --cell S --seed-block 903000 --n-universes 30 --max-wall-s 43200 --workers 2
    # then the reads script with --harness-root …/b8_cal_harness_work_r_completion --population 200

## 8. Design-gate self-check (the six r-b82-s4 checks, for the sonnet panel)

1 Executability: every statistic is a column arithmetic on existing CSVs; no missing field. 2 Stop rule:
zero-compute reads have none; cell R inherits the S3 sidecar rule (n_U_min = 20 of 30, one invocation).
3 Population: §5 lint + JOIN gate. 4 Byte-pin: §2.3/§5 instrument byte-id. 5 Blindness: the registration
author has NOT read S_M on either dataset; the only numbers seen are the banked full-score aggregates
(§2.3 power inputs) and the identity/reconstruction checks of §2.1 (which read no class-resolved score).
6 Internal consistency: bands are two-sided; INTERMEDIATE is first-class with three named sub-branches.

## 8b. Parent kill criterion (verbatim) and blindness status

Charter `RESEARCH_GRAPH_1_PROPOSAL_20260901.md:45`, q-completion-residual kill_criterion, quoted verbatim:
"registered arm fails to discriminate at its registered band after revision 2 -> park
bounded-undetermined with the measured bound". This draft is revision 1; a NO-READ or an
INTERMEDIATE (b) that the author elects to re-register consumes revision 2; a third failure parks
the question with the measured bound (|T_harn| + 3·SE_harn, T_prod ± SE_prod).

**Blindness status:** primary statistic point estimates exist in a gate record dated 2026-09-03
(unblinded by a design-gate side effect: DESIGN_GATE_stats.md); band thresholds were frozen before
that record; the registered read is executed by an agent that has not opened that record. The
revising author did not open it.

## 9. Open questions routed to d-completion-register (fresh RULE)

1. Ratify the operational definition of the residual as the matched-channel dark score (§1.1) and the
   provenance downgrade of "−0.14/event" to a stale [INFER] number (§1.2).
2. Ratify the bands: |Z| ≤ 3 twice, ρ thresholds 0.5 / 0.2 (§3).
3. Ratify that INTERMEDIATE (a) is routed to d-residual-attribution as the third bucket with a bound,
   not as a failed arm (it does not consume a revision).
4. Ratify the cost line: reads at ≈ 0.2 CPU-h; cell R ≤ 43 CPU-h (W ≤ 4) inside the 80 CPU-h cap; seed
   block 903000–903029 reserved.
5. Ratify the F-leaning §3.4 line as REPORTED-ONLY with the rail disclosure.
6. Note for the author: the harness commit stamp carries 1112 dirty paths; the resolved-flag equality
   assertion (§5 invariants) is the substitute for a byte-id of harness vs production code paths.

## REVISION 1 (2026-09-03) — changes against MUSTFIX_REVISION1_20260903.md (band thresholds untouched)

| item | change |
|---|---|
| 1 (stats) | §2.2: SE_prod re-sourced to the matched-channel per-event SD on the PRODUCTION dark class, formula given, computed by the read; §2.4 table updated; the 0.68 per-event SD is labelled "harness-borrowed proxy" wherever it appears. |
| 2 (stats) | §2.3: the harness FULL-score SE (0.0063, informational, reproduced exactly) separated from the matched-channel SE_harn (registered, larger, between-universe); §3 power restated as 3·SE_harn / 3·SE_prod formulas, the 0.02/event figure explicitly tied to the informational proxy. |
| 3 (design) | §5 g-znorm: exact-equality tolerance (max_e \|Δ den_log_term\| = 0 per h-node); "g-znorm red" added to the §4 NO-READ trigger list. |
| 4 (design) | §8b: charter line 45 kill criterion quoted verbatim, with the revision-counter consequence. |
| 5 (provenance) | §1.2: the artifact board-card row re-cited to B4_3 §4.4 + ledger row #261 (git-tracked sources). |
| 6 (byte-id) | §5: T0 anchor tolerance 1e-6 on the 6-dp display (READOUT_RECORD.md:40); full-precision 0.6669869414473403 banked as the re-run anchor at 1e-12; the literal 1e-9 withdrawn. §7 launch note updated. |
| both | §8b: blindness-status line added verbatim. |

Not changed: bands (|Z| ≤ 3, ρ 0.5 / 0.2), the identity (§2.1), the launch CLI (§7 — matches the built `completion_residual_reads.py` argparse exactly), the cost line, the disposition rows other than the NO-READ trigger list.
