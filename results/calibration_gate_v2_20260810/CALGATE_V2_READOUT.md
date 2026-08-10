# CALGATE v2 READOUT — mechanical scoring against the registered prereg

**Campaign:** stage-4 calibration-gate v2, `results/calibration_gate_v2_20260810/`
**Prereg of record:** `PREREGISTRATION_CALIBRATION_GATE_V2.md`, registered commit `065e7f58` (instrument + tests + prereg, atomic)
**Run commit:** `dbde71dc` (one-commit child of `065e7f58`; sole diff = `cluster/calibration_gate_v2.sbatch`, +46 lines; import-path diff **empty** — verified `git diff --stat 065e7f58 dbde71dc -- master_thesis_code/ master_thesis_code_test/` returns nothing)
**Venue:** bwUniCluster `cpu_il`/`dev_cpu_il`, SLURM array `6250988`, 10/10 tasks COMPLETED 0:0 (author-directed venue switch — see Disclosures D-4)
**Scorer:** `readout_score_v2.py` (this directory) — reads only the 10 campaign JSONs; every band is a literal transcription of the registered text; output `CALGATE_V2_READOUT.json`. Zero judgment calls: everything the prereg reserves to the author is labelled AUTHOR and left open.
**This readout makes no ruling. The branch call is presented to the author, never self-adjudicated (prereg policy of record).**

---

## 1. VALIDITY FIRST — V1…V5, abort criteria (prereg §10)

| check | registered requirement | measured | status |
|---|---|---|---|
| **V1** plumbing control | MAP = 0.730 **exactly**, both channels, all 50 seeds | 50/50 exact (1D) and 50/50 exact (2D), recomputed from `per_seed`; 0/50 non-finite ln_post | **PASS** |
| **V2** HPD port certification | boolean-exact agreement with `pp_coverage._hpd_contains`, 1000 posteriors (CI) | 30/30 calibration-gate tests pass at registration commit `065e7f58` (prereg §11); CI-owned, not re-executed by this readout | **PASS-AT-REGISTRATION** (no failure observed) |
| **V3** determinism | bit-identical re-runs; smoke spot-check | registration-time smokes reproduced v1 committed smoke `per_seed` records bit-identically (prereg §11); campaign cells are non-smoke (`smoke=false`), no per-cell V3 record embedded | **PASS-AT-REGISTRATION** (no failure observed) |
| **V4** texture certification | median corr(ln σ_dL, ln d_L) ∈ **[0.63, 0.75]** (v2 band, D1) | all 9 dl_binned cells: 0.66160 – 0.66687 (A: .66481/.66465/.66348 · B0: .66687 · B1: .66351 · B2: .66655/.66537/.66374 · V1: .66160); R0 N/A by construction (`independent` texture) | **PASS (9/9)** |
| **V5** R0 reproduction | ≤ 1e-12 relative | `{"pass": true, "mismatches": [], "rtol": 1e-12}` | **PASS** |
| abort (a) runtime | smoke extrap > 12 h ⇒ fallback | Σ task wall ≈ 3 649 s; longest task 10 m 34 s | not triggered |
| abort (b) non-finite | > 1 % non-finite ln_post in any cell ⇒ STOP | 0 non-finite in 3 250 + 200 seeds × 2 channels, independently recomputed per seed | not triggered |
| abort (c) V-failure | any ⇒ STOP | none | not triggered |

**No validity check failed.** (Contrast v1: V4 + DS-7 fired ⇒ GATE-NOT-TRUSTWORTHY.)

## 2. Provenance / clean rule (D5)

- `import_path_clean = true`, `dirt_inventory.import_path = []`, `allow_dirty = false` in **all 9** registered JSONs — the binding clause of the v2 clean rule holds literally in every output.
- `git_dirty = true` (whole tree) with non-empty `dirt_inventory.other` — all entries are sibling tasks' own output JSONs accumulating during the array run; none under the import path. Inventoried dirt never blocks (D5.3).
- `git_commit = dbde71dc` in all 10 JSONs — **not literally** the registered commit `065e7f58`. Mitigation (git-verified): direct one-commit child; only diff is the sbatch file; import-path diff empty. The clean rule binds import-path cleanliness, not SHA-equality to the prereg commit. Flagged as Disclosure D-5 for the author's read.
- CRB CSV md5 = `9a1f2a14384a9281c97ca3be312ddaab` — **matches the prereg pin** (recomputed by the scorer at readout time).
- Seed plan: all 9 blocks match prereg §5 exactly (offsets, counts, contiguity); all 3 250 seeds mutually disjoint; **zero** seeds inside v1's envelope `20260808+[0,9049]` (D6); O1's reserved block `+28000…+28399` untouched (O1 NOT-BUILT, §9 item 6).
- Record completeness: `len(per_seed) == len(seeds) == n_seeds` with ordered seed match, all files; no null fields.
- R0: retro-read of committed `closed_loop_results.json` (200 seeds, zero new compute, no gate weight); `allow_dirty` field null — consistent with a retro-read.

## 3. Trigger set evaluated (v2 §10; DS-7 removed per D2)

| trigger | fired? |
|---|---|
| V1 failure | NO |
| V2 failure | NO |
| V3 failure | NO |
| V4 failure | NO |
| V5 failure | NO |
| abort (b) | NO |
| both decision cells EDGE-CONTAMINATED in the channel read — 2D read (decision cells: A-2D, B2-2D) | NO — A-2D IS contaminated at all truths (0.110/0.155/0.2325 > 0.10) but B2-2D is 0.0 at all truths |
| — 1D read (B2 is the only 1D decision cell; A-1D exempt per v1 §5) | NO — B2-1D edge_loaded_frac = 0.0 |

**No trigger fires ⇒ the gate is TRUSTWORTHY.** GATE-NOT-TRUSTWORTHY does not fire; all measurements below carry their registered weight, and DS-8 verdicts are **not void**.

## 4. Per-cell statistics vs locked bands

DS-1 bands (N=400): 2σ [0.450,0.550]/[0.633,0.727]/[0.870,0.930]; FAIL = any β outside 3σ [0.425,0.575]/[0.610,0.750]/[0.855,0.945]. DS-2: PASS D≤0.0679, FAIL D>0.0814. DS-3: in-band |b|≤0.010, defect |b|≥0.030. Scorer independently recomputed every DS-1/DS-2/DS-3 status from per-β values — **all match the instrument's embedded labels** (zero mismatches).

| cell (h_true) | ch | DS-1 (C50/C68/C90) | DS-2 D | DS-3 bias | DS-4 R_low/R_high | edge frac | DS-1/2 gate weight |
|---|---|---|---|---|---|---|---|
| A (0.690) | 1D | FAIL 0/0/0 | 1.000 FAIL | −0.230 DEFECT | 1.000/0.000 | 1.000 | none (edge + A-1D exemption) |
| A (0.690) | 2D | FAIL .325/.435/.6125 | 0.2819 FAIL | +0.0800 DEFECT | 0.000/0.0025 | **0.110** | none (**EDGE-CONTAMINATED**) |
| A (0.730) | 1D | FAIL 0/0/0 | 1.000 FAIL | −0.270 DEFECT | 1.000/0.000 | 1.000 | none |
| A (0.730) | 2D | FAIL .275/.410/.6125 | 0.3334 FAIL | +0.0826 DEFECT | 0.000/0.015 | **0.155** | none (**EDGE-CONTAMINATED**) |
| A (0.770) | 1D | FAIL 0/0/0 | 1.000 FAIL | −0.310 DEFECT | 1.000/0.000 | 1.000 | none |
| A (0.770) | 2D | FAIL .2725/.4125/.6025 | 0.3201 FAIL | +0.0806 DEFECT | 0.000/0.010 | **0.2325** | none (**EDGE-CONTAMINATED**) |
| B0 (0.730) | 1D | (1/1/1, exempt) | (0.5, exempt) | +1.1e-16 IN-BAND | 0.000/0.000 | 0.000 | none (D3 degenerate-PIT exemption) |
| B0 (0.730) | 2D | (1/1/1, exempt) | (0.5, exempt) | +1.1e-16 IN-BAND | 0.000/0.000 | 0.000 | none (D3) |
| B1 (0.730) | 1D | FAIL 0/0/0 | 0.99997 FAIL | +0.01069 MIXED | 0.000/0.000 | 0.000 | yes (dose cell — not a decision cell) |
| B1 (0.730) | 2D | FAIL 0/0/0 | 0.99999 FAIL | +0.01095 MIXED | 0.000/0.000 | 0.000 | yes (dose cell) |
| **B2 (0.690)** | 1D | **FAIL 0/0/0** | **1.000 FAIL** | +0.03490 DEFECT | 0.000/0.000 | 0.000 | **yes — decision cell** |
| **B2 (0.690)** | 2D | **FAIL 0/0/0** | **1.000 FAIL** | +0.03506 DEFECT | 0.000/0.000 | 0.000 | **yes — decision cell** |
| **B2 (0.730)** | 1D | **FAIL 0/0/0** | **1.000 FAIL** | +0.03526 DEFECT | 0.000/0.000 | 0.000 | **yes — decision cell** |
| **B2 (0.730)** | 2D | **FAIL 0/0/0** | **1.000 FAIL** | +0.03574 DEFECT | 0.000/0.000 | 0.000 | **yes — decision cell** |
| **B2 (0.770)** | 1D | **FAIL 0/0/0** | **1.000 FAIL** | +0.03718 DEFECT | 0.000/0.000 | 0.000 | **yes — decision cell** |
| **B2 (0.770)** | 2D | **FAIL 0/0/0** | **1.000 FAIL** | +0.03738 DEFECT | 0.000/0.000 | 0.000 | **yes — decision cell** |
| V1 (0.730) | 1D+2D | (1/1/1, exempt) | (0.5, exempt) | ±1.1e-16 IN-BAND | 0.000/0.000 | 0.000 | none (D3; control) |
| R0 anchor | 1D | FAIL 0/0/0 | 0.99999996 FAIL | −0.130 | 1.000/0.000 | 1.000 | anchor-only, no gate weight |
| R0 anchor | 2D | FAIL .395/.575/.805 | 0.0597 PASS | +0.01128 MIXED | 0.005/0.035 | 0.880 | anchor-only, no gate weight |

**DS-5** (width vs F5): **NOT-EVALUABLE** as registered (§9 item 3 — no exact σ_z nodes in the committed F5 sweep). Raw context only: post_sd_median B1 ≈ 0.0012 (both ch), B2 ≈ 0.0021–0.0059 — far below the bias scale (the "posteriors far too narrow" feature).

**A-cell canonical-restricted reads (D4):** restricted 1D argmax = 0.600 for 400/400 at every truth (= DS-8 T1 statistic); full-75-grid 1D rail sits at the new low edge 0.460 with R_low = 1.0 (new information, un-banded). A-2D MAPs are interior (mean 0.770/0.813/0.851) — values that were truncated at 0.860 on the v1 grid.

**Edge-guard prediction vs observation (D4):** pre-declared residual A-2D edge-load was ~2–5 % per truth; observed 11.0 % / 15.5 % / 23.25 % — above the prediction and above the 10 % guard at all three truths. Per the registered D4 wording this is "an honest EDGE-CONTAMINATED outcome, not a repair failure": the guard fires and strips A-2D DS-1/DS-2 gate weight. Recorded as Disclosure D-6.

## 5. DS-6 — rail-reproduction contrast (Q2)

- R_low(B2, 1D) = 0.000 / 0.000 / 0.000 at h = 0.690/0.730/0.770 (all ≤ 0.05) — **not** RAIL-REPRODUCED (needs ≥ 0.90).
- RAIL-NOT-REPRODUCED additionally requires B2-1D to PASS DS-1 and DS-2 — B2-1D FAILs both at all truths.
- ⇒ **DS-6 = MIXED** (mechanical). Dose–response R_low(σ_z), 1D: 0.000 (σ_z=0, B0) → 0.000 (0.010, B1) → 0.000 (0.035, B2). R_low(B0) = 0.000 ≤ 0.05 ⇒ the pre-named impostor-ball N-2-analog anomaly did **not** occur.
- Content of the MIXED cell: the production-mirroring low rail does not appear anywhere in the ball venue; instead a uniform +≈σ_z MAP bias with collapsed coverage (0/0/0) and no railing, both channels — reproducing v1's barred pattern (now DS-8 T2, confirmed below).

## 6. DS-7 — generator-closure accounting: REPORT-ONLY, BOTH FORMS (D2; no V-class weight, no branch weight, not in the trigger set)

| cell | p_bar (± mc_se) | raw ratio | pass_raw | corrected ratio | pass_corrected |
|---|---|---|---|---|---|
| A (0.690) | 0.091969 (±0.000219) | 0.90293 | ✗ | 1.00261 | ✓ |
| A (0.730) | 0.095372 | 0.95045 | ✓ | 1.00223 | ✓ |
| A (0.770) | 0.098741 | 0.92662 | ✗ | 1.00012 | ✓ |
| B0 | 0.095372 | 0.94345 | ✗ | 0.99484 | ✓ |
| B1 | 0.095372 | 0.95399 | ✓ | 1.00596 | ✓ |
| B2 (0.690) | 0.091969 | 0.90293 | ✗ | 1.00261 | ✓ |
| B2 (0.730) | 0.095372 | 0.94986 | ✗ (0.0001 under the edge) | 1.00161 | ✓ |
| B2 (0.770) | 0.098741 | 0.92662 | ✗ | 1.00012 | ✓ |
| V1 | 0.095372 | 0.95045 | ✓ | 1.00223 | ✓ |

Raw form: 3/9 inside the (weightless) 0.05 band, straddling the edge — consistent with the v1-adjudicated MC-seed-fragility that motivated D2. Corrected form: 9/9 inside. **The raw-vs-corrected author call remains OPEN (registered).**

## 7. DS-8 — v1 pattern-reproduction targets (Q4), scored per target

Gate is trustworthy (§3) ⇒ DS-8 verdicts are **not void**. DS-8 carries **no branch weight** (D7) — it is the pattern-reproduction meter for the author's stage-5 read.

**T1 — single-host starvation rail (A-1D): CONFIRMED.**
Canonical-restricted argmax = 0.600 for 400/400 seeds at each truth ⇒ fractions 1.0000/1.0000/1.0000, all ≥ 0.98 (band). Un-banded new info: full-75-grid R_low(0.460) = 1.0 at all truths — the starvation rail follows the grid edge outward.

**T2 — ball-venue uniform +σ_z bias with collapsed coverage (B1, B2): CONFIRMED.**
All 8 banded bias components inside their v1±4√2·SE bands; all C90 and R_low/R_high components ≤ 0.02 (all exactly 0.0):

| component | v2 value | band | inside |
|---|---|---|---|
| B1-1D | +0.010688 | [+0.01036, +0.01147] | ✓ |
| B1-2D | +0.010950 | [+0.01059, +0.01181] | ✓ |
| B2(0.690)-1D | +0.034900 | [+0.03434, +0.03551] | ✓ |
| B2(0.690)-2D | +0.035063 | [+0.03408, +0.03627] | ✓ |
| B2(0.730)-1D | +0.035263 | [+0.03476, +0.03606] | ✓ |
| B2(0.730)-2D | +0.035737 | [+0.03456, +0.03696] | ✓ |
| B2(0.770)-1D | +0.037175 | [+0.03584, +0.03841] | ✓ |
| B2(0.770)-2D | +0.037375 | [+0.03673, +0.03957] | ✓ |

Un-banded companion (reported per prereg): post_sd_median ≈ 0.0012–0.0059 ≪ bias — the "too narrow" feature reproduces.

**T3 — B0 exactly on truth: CONFIRMED.**
grid-MAP = 0.730 exactly: 400/400 (1D) and 400/400 (2D), both ≥ 0.98; R_low = R_high = 0.0 both channels (≤ 0.02).

**All three targets CONFIRMED on disjoint seeds.** Per prereg Q4: "Confirmation converts v1's barred patterns into quotable measured properties" — subject to the author's ratification (AUTHOR-RATIFY flags, §2) and the Disclosure list below.

## 8. THE BRANCH THAT FIRES (mechanical application of the registered tree)

1. **GATE-NOT-TRUSTWORTHY**: no §10 trigger fires (§3) ⇒ does not fire.
2. **KEEP-DIGGING** = trustworthy AND ((a) DS-6 = RAIL-NOT-REPRODUCED OR (b) DEFECT-class: DS-1 or DS-2 FAIL in a non-exempt decision cell×channel that is not the registered starvation signature):
   - (a): DS-6 = MIXED ⇒ (a) does not hold.
   - (b): **holds** — six gate-weighted decision cell×channel entries FAIL both DS-1 and DS-2: B2-1D and B2-2D at all three truths (none exempt, none edge-stripped, none the registered A-1D starvation signature). A-2D also FAILs but is stripped by the §8 edge guard and does not contribute.
3. REPORT-BOUND requires A-2D and B2-2D PASS DS-1+DS-2 AND DS-6 = RAIL-REPRODUCED ⇒ does not hold.
4. MIXED = "anything else" ⇒ not reached (a prior branch's condition holds).

### ⇒ Branch: **KEEP-DIGGING, via clause (b) DEFECT-class**

Stage-5 mapping (registered decision table): the DEFECT row — "≥3σ coherent class displacement, or coverage failure ⇒ fix via `/physics-change`", author-gated. Per the registered clause-(b) requirement, **the "one measurement that decides" (stage-5 UNDETERMINED row) must be named — that naming is the author's** (this readout does not rule); the raw pattern it must decide on is recorded in §10 below.

Stage-5 stop rule of record: "stop digging" requires coverage pass ∧ width-on-forecast ∧ no-unmodeled-selection. Measured: coverage **FAIL** (decision cells); width **NOT-EVALUABLE**; unmodeled-selection leg **OPEN** (§9 item 1). The conjunction is not satisfied; no conjunct supports it.

**DS-6 = MIXED is reported alongside** (registered: the dose–response and truth-dependence are part of the record, not folded into the branch).

## 9. Stage-4 gate table (docs/RESEARCH_CYCLE.md — three legs, all required)

| leg | status | numbers |
|---|---|---|
| 1. SBC / P–P coverage of the full 2-channel estimator | **EVALUATED (in-loop venue): FAIL** | C50/C68/C90 = 0/0/0 and KS D ≈ 1.0 in every gate-weighted decision cell×channel (B2-1D, B2-2D × 3 truths); MAP bias +0.0349…+0.0374 (DEFECT-scale). A3 criteria met by the instrument: genuinely 2-channel, production N = 1500, multi-candidate balls (λ=4, K_mean ≈ 5.00). A-2D stripped by edge guard. **Venue transfer to production: NOT-EVALUABLE** (§9 items 2, 5 — z-window Poisson caricature, no GLADE/n(z)/sky/completeness). |
| 2. Generator-closure absolute-count audit | **REPORT-ONLY / NOT-EVALUABLE as a gate leg** (D2, §9 item 1) | DS-7 corrected ratio 9/9 inside 0.05; raw 3/9 (MC-seed-fragile); no branch weight; raw-vs-corrected author call OPEN. Leg 2 remains carried by the standing FIXB result + the open f_k–pool-coupling intake thread. |
| 3. Forecast-consistent width | **NOT-EVALUABLE** (§9 item 3) | No exact σ_z nodes {0, 0.010, 0.035} in the committed F5 sweep; matched-population F5 run remains a registered follow-up. Raw context: B1/B2 post_sd_median 0.0012–0.0059. |

## 10. DISCLOSURE LIST (mandatory)

**D-1 (verifier note 1) — validate-output provenance field gap.** `run_validate()` (`master_thesis_code/validation/calibration_gate.py:1534`) builds its V3/V4/V5 document without embedding `git_commit`/`import_path_clean`/`dirt_inventory`, unlike registered cell runs (which embed full provenance at `calibration_gate.py:1309`). Mitigation: `--validate` is a pre-registration/CI-time check, not a registered cell run — but the gap in `validate_results.json` is real.

**D-2 (verifier note 2) — D4 clearance-sum presentation typo.** Prereg D4 states "≈ 0.04 + 0.21 + 0.12 ≈ 0.31 > 0.26"; the sum is 0.37. The qualitative conclusion (> 0.26 ⇒ extended grid) is unaffected; the stated intermediate is arithmetically wrong as written.

**D-3 (verifier note 3) — stale `pp_readout` docstring.** `pp_readout()` (`calibration_gate.py:402`) docstring still says "41-point … q = ∫_{0.600}"; v2 A cells run it on the 75-point grid with q = ∫_{0.460} (prereg §4). The function is grid-generic and behaves correctly (verified against per-seed PIT values); only the docstring is stale.

**D-4 — venue switch mid-campaign (author-directed).** Prereg §0 registers "**Local CPU only; no cluster jobs**" as a binding constraint. The local run was aborted after only the R0 retro-read (2 s; `driver_logs/summary.log` shows R0 START/END rc=0 then "START A_h0p690" with no END — killed mid-flight before any registered seed's posterior was recorded; **no registered cell ran locally**). The full campaign then ran on bwUniCluster cpu_il (array 6250988, sbatch commit `dbde71dc`) at the author's direction. This is a literal deviation from a registered §0 constraint, disclosed here for the author's ratification. Mitigations, all machine-checked: seeds are prereg-pinned inside the instrument (base + registered offsets selected by `--cell`/`--truth`, never by SLURM task id), so the venue cannot alter which seeds were drawn or their order; the sbatch TID→cell map matches prereg §5 exactly; every output JSON carries its own commit/clean-rule/provenance block, and those per-file checks — not the venue narrative — are the actual check. The venue-switch is **not** a member of the registered §10 trigger set and therefore cannot mechanically void the gate; whether it voids anything is the author's call.

**D-5 — git_commit literal mismatch.** All 10 JSONs record `git_commit = dbde71dc` ≠ registered `065e7f58`. Git-verified: `dbde71dc` is a direct one-commit child whose only diff is the sbatch script; the import-path diff is empty. The D5 clean rule binds import-path cleanliness (holds literally in every file), not SHA-equality to the prereg commit.

**D-6 — edge-guard prediction undershot (A-2D).** Pre-declared residual edge-load ~2–5 % per truth; observed 11.0/15.5/23.25 % — the guard fired at all three truths, stripping A-2D DS-1/DS-2 weight. Registered D4 wording pre-classifies this as an honest EDGE-CONTAMINATED outcome, not a repair failure; the magnitude gap versus the prediction is recorded here as a raw fact.

**D-7 — runtime vs budget.** Prereg budget 3.5–5.0 h at 14 local workers; actual ≈ 60.8 CPU-task-minutes summed across 10 parallel 64-worker tasks (longest task 10 m 34 s). A venue-driven runtime difference only; no seed or statistic is affected.

## 11. Formulation for the author — what this means for the 1D rail account and paper #47 (NO RULING MADE HERE)

**Mechanically:** the v2 gate is **trustworthy** (all validity checks pass, no trigger fires — the v1→v2 repairs did their job), and the registered tree fires **KEEP-DIGGING via clause (b) DEFECT-class**: the ball-venue decision cells show a coverage collapse (0/0/0 with KS D ≈ 1) around a uniform +≈σ_z MAP bias in *both* channels at all three truths — and that pattern is now **reproduced out-of-sample** (DS-8 T2 CONFIRMED on disjoint seeds), as are the single-host starvation rail (T1) and the B0-on-truth null (T3).

**For the 1D rail account** ([[h0-railing-rootcause-photoz]]: photo-z information starvation): DS-6 = MIXED with content — the production low rail does **not** reproduce in the multi-candidate ball venue at the GLADE σ_z of record (R_low = 0 everywhere, any σ_z dose); the starvation rail reproduces only in the single-host configuration (A-1D, 400/400, and it follows the extended grid to 0.460). In-loop, σ_z = 0.035 with impostor confusion manufactures a *uniform positive bias with delta-narrow posteriors*, not a rail; B0 (impostors, perfect z) is exactly on truth, so the defect is σ_z-dosed, not ball-structural (the pre-named N-2-analog did not occur; dose: 0 → +0.011 → +0.035 ≈ +σ_z). Whether this bounds, replaces, or coexists with the starvation account of the *production* rail is exactly what venue transfer (§9 items 2, 5) leaves NOT-EVALUABLE — the formulation the author must rule on is: *does a trustworthy in-loop KEEP-DIGGING(b) DEFECT — a σ_z-proportional coherent bias the estimator does not propagate into its width — become the named owner-candidate for the production 1D behaviour, and what is the one measurement that decides (clause-b requirement, author-named)?*

**For paper #47's hold** (RUNBOOK-8 §1 item 8 / RUNBOOK-7 §4: held on the missing trusted P–P leg): the P–P leg now **exists and is trustworthy for the first time — and it FAILED in the decision cells**. The stage-5 stop rule's conjunction (coverage pass ∧ width-on-forecast ∧ no-unmodeled-selection) is not satisfied: coverage FAIL, width NOT-EVALUABLE, selection leg OPEN. Nothing in this readout mechanically supports lifting the hold; whether the hold's *reason* is now upgraded (from "leg missing" to "leg failed — DEFECT route via `/physics-change`") is the author's ruling, together with: ratification of the §2 AUTHOR-RATIFY register (D1–D8), the D-4 venue deviation, acceptance of DS-8 confirmations as quotable measured properties (Q4), and the OPEN DS-7 form call.

---
*Scored 2026-08-10 by `readout_score_v2.py` (mechanical; zero band adjustments; zero adjudication). Machine-readable twin: `CALGATE_V2_READOUT.json`. Ledger and book untouched by this readout. No commit made.*
