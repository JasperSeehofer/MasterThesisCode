# VENUE-TRANSFER READ — mechanical readout

**Scored against** `results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md`
(registered `e77eecad`; instrument `2ece8801`). **Scorer:**
`results/venue_transfer_20260811/score_venue_transfer.py` → machine-readable twin
`VENUE_TRANSFER_READOUT.json`. Raw upstream extraction: `collect_raw.json`
(independently re-derived here from the 49 chunk JSONs — this readout does not trust it).

**Campaign:** 49 chunk files · 1400 seeds · 6 cells · 982 pinned events/seed · both channels.

> **This readout does not rule.** It reports which branch the registered decision tree fires
> and hands it to the author. Prereg model/effort policy: *the branch call is presented to the
> author, never self-adjudicated.* No ledger, book, claim, or prereg file was touched.

---

## 1. VALIDITY FIRST

### 1.1 V-T1 — T-0 anchor (real events + real K at σ_z = 0)

Registered edges (§10): **|bias| ≤ 0.010 both channels** and **R_low, R_high ≤ 0.05**;
|bias| ∈ (0.010, 0.030) = **ANCHOR-MARGINAL** (reported, disclosed, does not void);
**|bias| ≥ 0.030 or a rail > 0.05 ⇒ VENUE-CONFOUNDED** — and simultaneously a first-class
new raw finding. DS-VT1/DS-VT2 not scored on T-0 (degenerate-PIT exemption, VT-D8).

| T-0 (N=200) | bias (grid-argmax) | bias (refined) | R_low | R_high | zone |
|---|---|---|---|---|---|
| 1D | **+0.000000** (SE 0.000000) | +0.000033 (SE 0.000033) | 0.000 | 0.000 | in-band (≤ 0.010) |
| 2D | **+0.000000** (SE 0.000000) | +0.000033 (SE 0.000033) | 0.000 | 0.000 | in-band (≤ 0.010) |

- **ANCHOR-MARGINAL zone (0.010, 0.030): NOT entered** — the anchor sits ~300× below the
  in-band edge on grid-argmax (all 200 seeds argmax exactly at h = 0.730).
- **Hard VENUE-CONFOUNDED trigger (|bias| ≥ 0.030 or rail > 0.05): DID NOT FIRE.** The
  pre-named "first-class new raw finding" (bare-kernel ball estimator uncalibrated under real
  multiplicity at perfect redshifts) therefore **has no subject in this campaign**.
- Degenerate readouts recorded for completeness (carry no coverage information, VT-D8):
  HPD50/68/90 = 1.000/1.000/1.000; PIT ≡ 0.5 on all 200 seeds ⇒ KS D = 0.5 (degenerate).
- **V-T1 verdict: PASS.** Evidence: 8 T-0 chunk JSONs, per-seed records recomputed by the scorer.

### 1.2 V-T2 — determinism

**PASS.** Evidence: `validate_results_full.json` → `v_t2.pass = true` (bit-identical re-run
spot-check, seed 20303808, `n_events_cap = 40`, dev box, instrument `2ece8801`).
Independent corroboration in-campaign: the `workers` field differs across resubmission waves
(64 on 10 chunks, 25 on 39 chunks) yet **every one of the 1200 `real_k` seeds reproduces
`K_sum = 1,193,703` exactly** — worker grain is result-invariant as the prereg §11 note asserts.

### 1.3 V-T3 — pin integrity

**PASS.** Two independent evidence sources, both checked:
- `validate_results_full.json` → `v_t3.pass = true` (CRB CSV md5 `9a1f2a14…` ✓; frozeng emit md5
  `34c50e91…` ✓; K census 1588/606/74/median 6/p99 11325.26/max 245364/ΣK 1,193,703/nonempty 982 ✓;
  pruned-frame σ_z stats n 20,834,171 / median 0.0393412950539589 / min 0.0005317263419419 /
  n<5e-3 231,098 / n<1e-2 235,731 ✓).
- `pin_integrity.pass = true` in **all 49** chunk JSONs (re-read here).
- Per-seed census pin: **1200/1200 `real_k` seeds at ΣK = 1,193,703 exactly** (T-0, T-b, all three
  T-c truths). **T-a is structurally exempt** — registered `balls = "poisson4"` (§5 / VT-D2), so its
  K_sum is a per-seed Poisson-λ=4 draw (realized 4757–5128, 137 distinct values). Not a mismatch;
  see disclosure **D-VT-8**.

### 1.4 V-T4 — clean rule

**PASS.** Evidence embedded per-chunk (the clean rule is enforced per registered run, not by a
standalone validate field): across all 49 chunks — `import_path_clean = true`,
`allow_dirty = false`, `smoke = false`, and `dirt_inventory.import_path = []` (every dirt line
classified as `other`: untracked results/staging dirs and `logs/`). `git_dirty = true` on all 49
is expected and permitted by V-T4 clause 3.
**Naming disclosure (D-VT-5):** V-T4's registered wording names `master_thesis_code/` +
`master_thesis_code_test/`. The package was renamed to `darksiren_emri` **after** the campaign
(commit `227e7a32`, verified a descendant of both campaign commits), so the flags were evaluated
against the then-current path and are correct for the repo state at run time. The import-path diff
`2ece8801..e93f3068` is **empty under both the old and the new path names** (re-verified).

### 1.5 V-T5 — no-drift anchor (cross-instrument)

**PASS.** `validate_results_full.json` → `v_t5.pass = true`: the vector-σ estimator core in
v2-compat mode bit-reproduces the committed v2 `B2_h0p730_results.json` per-seed records on seeds
20286808–20286810, **3/3 seeds, 41 shared fields each, zero mismatched fields**.
**Sequencing disclosure (D-VT-2):** the earlier launch-phase validate skipped V-T5
(`validate_results_novt5.json`, `v_t5.pass = null`); the full run completed after the first
(partially timed-out) array — the compliance-order deviation logged in prereg §11 addendum 1,
**PENDING AUTHOR RATIFICATION**. The check itself passes.

### 1.6 Registered-commit chain (VT-D0 iii)

10 chunks carry `2ece8801` (the registered instrument commit), 39 carry `e93f3068`.
`2ece8801` **is** an ancestor of `e93f3068` (verified); the diff between them touches exactly 4
markdown files under `results/` and **zero** lines under the import path (old **or** new names) —
the R1-ratified D-4/D-5 pattern holds. One of those 4 files is the prereg itself: the diff is a
**pure append** at line 608 (`@@ -607,0 +608,48 @@`) into §11, so VT-D0(ii) — "leaves every line
above the §11 appendix unmodified" — holds. The on-disk prereg above §11 is byte-identical to the
registered `e77eecad` version (md5 `388edd11254903d216fb16b1dbd476cb`).

### 1.7 Seed plan (VT-D7)

**PASS, exact.** Base 20260808. Per cell, the realized seed set **equals** the registered block
with zero gaps, zero extras, zero within-cell duplicates:

| cell | offsets | block | expected N | realized N | exact |
|---|---|---|---|---|---|
| T-0 | +40000…40199 | 20300808–20301007 | 200 | 200 | ✓ |
| T-a | +41000…41199 | 20301808–20302007 | 200 | 200 | ✓ |
| T-b | +42000…42199 | 20302808–20303007 | 200 | 200 | ✓ |
| T-c(0.690) | +43000…43199 | 20303808–20304007 | 200 | 200 | ✓ |
| **T-c(0.730)** | +44000…44399 | 20304808–20305207 | **400** | **400** | ✓ |
| T-c(0.770) | +45000…45199 | 20305808–20306007 | 200 | 200 | ✓ |

Cross-cell: 1400 seeds, 1400 unique, 0 cross-cell duplicates; 0 collisions with the v1 envelope
`+[0, 9049]`, the v2 envelope `+[20000, 29049]`, or the reserved-but-unbuilt **W1** (+46000…46399)
and **O2** (+47000…47399) blocks. Corroborated by `validate_results_full.json` `seed_plan.pass = true`.
All 1400 seeds ran `n_events = n_events_run = 982`, `n_events_cap = null`, `f_incl = 1.0` (VT-D5).

### 1.8 Abort criteria (a)–(d)

| criterion | registered trigger | measured | triggered? |
|---|---|---|---|
| **(a)** smoke-measured heavy per-seed CPU | > 2 × 4.33 = **8.66 CPU-h/seed** | **3.79 CPU-h/seed** (0.87× the anchor; §11 note 1, array 6252702 task 28: 94.63 CPU-h / 25 seeds, uncontended 64-core node) | **NO** |
| **(b)** non-finite `ln_post` | > 1 % of any cell's seeds | **0.0 in every cell, both channels** (recomputed from all 1400 × 2 × 41-point `ln_post` vectors — zero non-finite entries anywhere) | **NO** |
| **(c)** any V-T failure | any of V-T1…V-T5 FAIL | **V-T1…V-T5 all PASS** | **NO** |
| **(d)** horizon-drop guard | > 5 % of the pinned 982 | **`n_horizon_dropped = 0` on all 1400 seeds** (max fraction 0.0) | **NO** |

Consequence of (a) not tripping: **no N-floor fallback stage was invoked** — T-c wings retained,
T-b at N = 200, decision cell at the full registered **N = 400**.

### 1.9 §8 edge-contamination guard (per cell × channel)

edge-loaded = `edge_mass > 0.01`; a cell × channel with **> 10 %** edge-loaded seeds is
EDGE-CONTAMINATED ⇒ its DS-VT1/DS-VT2 carry no weight (DS-VT4 exempt).

| cell | ch | edge-loaded fraction | max `edge_mass` | EDGE-CONTAMINATED |
|---|---|---|---|---|
| T-0 | 1D / 2D | 0.000 / 0.000 | 0 / 0 | NO / NO |
| T-a | 1D / 2D | 0.000 / 0.000 | 0 / 0 | NO / NO |
| T-b | 1D / 2D | 0.000 / 0.000 | 0 / 0 | NO / NO |
| T-c(0.690) | 1D / 2D | 0.000 / 0.000 | 7.8e-196 / 9.0e-203 | NO / NO |
| **T-c(0.730)** | 1D / 2D | **0.000 / 0.000** | 0 / 0 | **NO / NO** |
| T-c(0.770) | 1D / 2D | 0.000 / 0.000 | 3.1e-11 / 1.2e-09 | NO / NO |

Zero edge-loaded seeds anywhere (0/1400 × 2). The §4.6 grid-clearance prediction held: MAP lands
≈ truth + 0.037…0.041, i.e. ≤ 0.811 at the highest truth, far from the 0.860 grid edge.

### 1.10 VENUE-CONFOUNDED trigger set — member by member (§10)

| # | trigger-set member | fired? | basis |
|---|---|---|---|
| 1 | V-T2 failure | **NO** | `v_t2.pass = true` |
| 2 | V-T3 failure | **NO** | `v_t3.pass = true` + 49/49 chunk `pin_integrity.pass` |
| 3 | V-T4 failure | **NO** | 49/49 `import_path_clean`, `allow_dirty=false`, `smoke=false`, empty import-path dirt |
| 4 | V-T5 failure | **NO** | `v_t5.pass = true`, 3/3 bit-identical |
| 5 | abort (b) non-finite `ln_post` > 1 % | **NO** | max fraction 0.0 |
| 6 | abort (d) horizon drop > 5 % | **NO** | max fraction 0.0 |
| 7 | V-T1 T-0 hard trigger (\|bias\| ≥ 0.030 or rail > 0.05) | **NO** | bias 0.000000, rails 0.000 |
| 8 | decision cell EDGE-CONTAMINATED in the read channel (1D, VT-D6) | **NO** | 0.000 edge-loaded |
| 9 | decision cell EDGE-CONTAMINATED, 2D secondary | **NO** | 0.000 edge-loaded |

**VENUE-CONFOUNDED: does not fire (0 of 9 members).** Branch 1 is not taken.

---

## 2. DS-VT1 … DS-VT4, scored at the registered per-N rows, both channels

Bands: decision cell **T-c(0.730) on the N = 400 rows**; wings T-c(0.690)/T-c(0.770) and T-b/T-a on
the **N = 200 rows**. DS-VT3 primary statistic is the registered **grid-argmax** bias (the committed
v2 anchors +0.035263/+0.035737 are grid-argmax); the refined-argmax companion is reported alongside
and changes no status anywhere (**D-VT-7**).

### 2.1 DS-VT1 — HPD coverage

Every dosed cell × channel returns **0/N at all three levels** (HPD50 = HPD68 = HPD90 = 0.000):
the posterior contains the truth on **zero of the 1200 dosed seeds**, in both channels.

| cell (N row) | level | value | 2σ band | 3σ band | inside 3σ? |
|---|---|---|---|---|---|
| **T-c(0.730)** (400) | HPD50 / 68 / 90 | 0.000 / 0.000 / 0.000 | [0.450,0.550] / [0.633,0.727] / [0.870,0.930] | [0.425,0.575] / [0.610,0.750] / [0.855,0.945] | **NO / NO / NO** |
| T-c(0.690) (200) | HPD50 / 68 / 90 | 0.000 / 0.000 / 0.000 | [0.429,0.571] / [0.614,0.746] / [0.858,0.942] | [0.394,0.606] / [0.581,0.779] / [0.836,0.964] | NO / NO / NO |
| T-c(0.770) (200) | HPD50 / 68 / 90 | 0.000 / 0.000 / 0.000 | same as above | same as above | NO / NO / NO |
| T-b (200) | HPD50 / 68 / 90 | 0.000 / 0.000 / 0.000 | same as above | same as above | NO / NO / NO |
| T-a (200) | HPD50 / 68 / 90 | 0.000 / 0.000 / 0.000 | same as above | same as above | NO / NO / NO |

Identical in 1D and 2D. **DS-VT1 status: FAIL in every dosed cell × channel.**
T-0: EXEMPT (VT-D8); its 1.000/1.000/1.000 is degenerate.

### 2.2 DS-VT2 — P–P / KS

| cell | N row | PASS D ≤ | FAIL D > | D (1D) | D (2D) | status |
|---|---|---|---|---|---|---|
| **T-c(0.730)** | 400 | 0.0679 | 0.0814 | **1.000000** | **1.000000** | **FAIL / FAIL** |
| T-c(0.690) | 200 | 0.0960 | 0.1151 | 1.000000 | 1.000000 | FAIL / FAIL |
| T-c(0.770) | 200 | 0.0960 | 0.1151 | 1.000000 | 1.000000 | FAIL / FAIL |
| T-b | 200 | 0.0960 | 0.1151 | 1.000000 | 1.000000 | FAIL / FAIL |
| T-a | 200 | 0.0960 | 0.1151 | 1.000000 | 1.000000 | FAIL / FAIL |
| T-0 | — | — | — | 0.500000 (degenerate) | 0.500000 (degenerate) | EXEMPT (VT-D8) |

D is **saturated**, not marginal: per-seed PIT values sit at ~1e-20 and below (**D-VT-9**).

### 2.3 DS-VT3 — MAP bias and dose ratio

In-band \|b\| ≤ 0.010 · DEFECT-scale \|b\| ≥ 0.030 · **R_dose = bias / σ̄_pairs ∈ [0.75, 1.25]**.

| cell | ch | σ̄_pairs (realized) | bias (grid-argmax) ± SE | bias (refined) ± SE | status | R_dose (argmax) | R_dose (refined) | in band? |
|---|---|---|---|---|---|---|---|---|
| **T-c(0.730)** N=400 | **1D** | **0.041775** | **+0.037237 ± 0.000230** | +0.037245 ± 0.000223 | **DEFECT-SCALE** | **0.8914** | 0.8916 | **YES** |
| **T-c(0.730)** N=400 | 2D | 0.041775 | +0.039713 ± 0.000246 | +0.039720 ± 0.000230 | DEFECT-SCALE | 0.9506 | 0.9508 | YES |
| T-c(0.690) N=200 | 1D | 0.041452 | +0.036350 ± 0.000310 | +0.036535 ± 0.000283 | DEFECT-SCALE | 0.8769 | 0.8814 | YES |
| T-c(0.690) N=200 | 2D | 0.041452 | +0.038950 ± 0.000316 | +0.038967 ± 0.000293 | DEFECT-SCALE | 0.9396 | 0.9401 | YES |
| T-c(0.770) N=200 | 1D | 0.042082 | +0.038400 ± 0.000320 | +0.038528 ± 0.000286 | DEFECT-SCALE | 0.9125 | 0.9156 | YES |
| T-c(0.770) N=200 | 2D | 0.042082 | +0.040900 ± 0.000349 | +0.041032 ± 0.000297 | DEFECT-SCALE | 0.9719 | 0.9750 | YES |
| T-b N=200 | 1D | 0.035000 (flat, exact) | +0.035875 ± 0.000175 | +0.035953 ± 0.000156 | DEFECT-SCALE | 1.0250 | 1.0272 | YES |
| T-b N=200 | 2D | 0.035000 | +0.037625 ± 0.000194 | +0.037656 ± 0.000156 | DEFECT-SCALE | 1.0750 | 1.0759 | YES |
| T-a N=200 | 1D | 0.035000 (flat, exact) | +0.034900 ± 0.000184 | +0.034824 ± 0.000158 | DEFECT-SCALE | 0.9971 | 0.9950 | YES |
| T-a N=200 | 2D | 0.035000 | +0.036450 ± 0.000193 | +0.036469 ± 0.000158 | DEFECT-SCALE | 1.0414 | 1.0420 | YES |
| T-0 N=200 | 1D / 2D | 0 (σ_z = 0) | +0.000000 / +0.000000 | +0.000033 / +0.000033 | IN-BAND | n/a | n/a | n/a |

Bias is **positive** in every dosed cell × channel; the sign is uniform. Realized dose statistics
in T-c: σ̄_pairs 0.041452 / 0.041775 / 0.042082 and spec-z-like tail fraction
(`frac_pairs_sigma_lt_5e-3`) 0.000796 / 0.000650 / 0.000515 — the GLADE tail **is** present in the
balls (**D-VT-10** on σ̄ vs the predicted [0.039, 0.041] window).

### 2.4 DS-VT4 — rail fractions + RAIL-EMERGENT check

Collapse band ≤ 0.02 (N = 400) / ≤ 0.04 (N = 200).

| cell | N row | band | R_low 1D / 2D | R_high 1D / 2D | in band? | RAIL-EMERGENT (≥ 0.90, decision cells) |
|---|---|---|---|---|---|---|
| **T-c(0.730)** | 400 | ≤ 0.02 | **0.000 / 0.000** | **0.000 / 0.000** | **YES** | **NO** |
| T-c(0.690) | 200 | ≤ 0.04 | 0.000 / 0.000 | 0.000 / 0.000 | YES | NO |
| T-c(0.770) | 200 | ≤ 0.04 | 0.000 / 0.000 | 0.000 / 0.000 | YES | NO |
| T-b | 200 | ≤ 0.04 | 0.000 / 0.000 | 0.000 / 0.000 | YES | (n/a) |
| T-a | 200 | ≤ 0.04 | 0.000 / 0.000 | 0.000 / 0.000 | YES | (n/a) |
| T-0 | 200 | ≤ 0.04 (V-T1 edge 0.05) | 0.000 / 0.000 | 0.000 / 0.000 | YES | (n/a) |

**RAIL-EMERGENT did NOT fire** — zero railed seeds in 1400 × 2 channel-seeds. The venue
manufactures **no** railing.

### 2.5 Delta-narrow companion (REPORTED UN-BANDED, v2 convention)

v2 committed reference range for `post_sd_median`: 0.0012–0.0059.

| cell | ch | `post_sd_median` | bias / post_sd_median |
|---|---|---|---|
| **T-c(0.730)** | 1D / 2D | 0.004376 / 0.004410 | **8.51** / 9.00 |
| T-c(0.690) | 1D / 2D | 0.004249 / 0.004300 | 8.55 / 9.06 |
| T-c(0.770) | 1D / 2D | 0.004534 / 0.004622 | 8.47 / 8.85 |
| T-b | 1D / 2D | 0.003689 / 0.003687 | 9.73 / 10.21 |
| T-a | 1D / 2D | 0.003619 / 0.003615 | 9.64 / 10.08 |
| T-0 | 1D / 2D | 0.000000 / 0.000000 | n/a (degenerate) |

All within the v2 committed range; the posteriors are ~8.5–10 σ away from the truth. **Un-banded —
no PASS/FAIL is asserted** (no committed SE for a median; DS-5 stays NOT-EVALUABLE).

### 2.6 Mechanical per-channel cell classification (prereg §7 rule, applied verbatim)

`COLLAPSE-REPRODUCED` = C90 ≤ band **and** R_low, R_high in band **and** bias ≥ +0.030 **and**
R_dose ∈ [0.75, 1.25]. `CALIBRATED` = DS-VT1 all three inside 3σ **and** DS-VT2 PASS **and**
\|bias\| ≤ 0.010 **and** rails in band. `OTHER` = anything else.

| cell | N | 1D classification | 2D classification |
|---|---|---|---|
| T-0 (anchor) | 200 | *ANCHOR — not classified (DS-VT1/DS-VT2 exempt, VT-D8; scored on DS-3/DS-4 only): bias in-band, no rails* | *same* |
| T-a | 200 | **COLLAPSE-REPRODUCED** | **COLLAPSE-REPRODUCED** |
| T-b | 200 | **COLLAPSE-REPRODUCED** | **COLLAPSE-REPRODUCED** |
| T-c(0.690) | 200 | **COLLAPSE-REPRODUCED** | **COLLAPSE-REPRODUCED** |
| **T-c(0.730) — THE decision read** | **400** | **COLLAPSE-REPRODUCED** | **COLLAPSE-REPRODUCED** |
| T-c(0.770) | 200 | **COLLAPSE-REPRODUCED** | **COLLAPSE-REPRODUCED** |

No cell in either channel is `CALIBRATED`; no cell in either channel is `OTHER`.

---

## 3. DS-VT5 — per-axis ablation ladder (report-graded, no branch weight)

Ladder order as registered: **v2 B2(0.730) committed baseline → T-a (+ real events) → T-b
(+ real multiplicity) → T-c (+ real σ_z)**. Each arm classified at its own N.

| rung | arm | N | axis added | 1D bias (argmax) | 1D R_dose | 1D HPD90 | 1D rails | **1D class** | 2D class |
|---|---|---|---|---|---|---|---|---|---|
| 0 | v2 B2(0.730) *(committed baseline, quotable per R2)* | 400 | gate caricature: synthetic universe, Poisson λ=4 balls, flat σ_z = 0.035 | +0.035263 | 1.0075 | 0.000 | 0.000/0.000 | **COLLAPSE-REPRODUCED** *(committed v2 record)* | COLLAPSE-REPRODUCED |
| 1 | **T-a** | 200 | **+ real detected event population** (axis a) | +0.034900 | 0.9971 | 0.000 | 0.000/0.000 | **COLLAPSE-REPRODUCED** | COLLAPSE-REPRODUCED |
| 2 | **T-b** | 200 | **+ real ball multiplicity, real K_i** (axis b-multiplicity; ΣK = 1,193,703 vs ~7.5 k) | +0.035875 | 1.0250 | 0.000 | 0.000/0.000 | **COLLAPSE-REPRODUCED** | COLLAPSE-REPRODUCED |
| 3 | **T-c(0.730)** | 400 | **+ real heterogeneous GLADE σ_z, spec-z tail included** (axis c) | +0.037237 | 0.8914 | 0.000 | 0.000/0.000 | **COLLAPSE-REPRODUCED** | COLLAPSE-REPRODUCED |

**KILLING AXIS: NONE.** No arm leaves `COLLAPSE-REPRODUCED`; the ladder never breaks, so the
registered "first arm in ladder order whose classification leaves COLLAPSE-REPRODUCED" has no
member. Each production-matching axis is added without attenuating the pattern (bias drifts
+0.0353 → +0.0349 → +0.0359 → +0.0372 while σ̄ rises 0.035 → 0.0418, i.e. R_dose stays inside
[0.75, 1.25] throughout).

**T-a vs the committed v2 B2(0.730) values — raw context only, no band carries (pre-stated):**

| | v2 B2(0.730) committed | T-a realized | Δ |
|---|---|---|---|
| 1D bias | +0.035263 | +0.034900 | **−0.000363** |
| 2D bias | +0.035737 | +0.036450 | **+0.000713** |

Swapping the synthetic universe for the pinned real detected event population moves the bias by
< 0.0008 in either channel.

---

## 4. THE BRANCH THAT FIRES — checked in the registered order, zero judgment calls

| order | branch | registered condition | fires? |
|---|---|---|---|
| 1 | **VENUE-CONFOUNDED** | any trigger-set member fires | **NO** — 0 of 9 members fired (§1.10) |
| 2 | **TRANSFER-CONFIRMED** | T-c **1D** is COLLAPSE-REPRODUCED at **all three truths** (0.730 @ N=400 rows; wings @ N=200 rows) | **YES** — 0.690 ✓, 0.730 ✓, 0.770 ✓ |
| 3 | **TRANSFER-REFUTED** | T-c(0.730) 1D is CALIBRATED | not reached (and would be NO: it is COLLAPSE-REPRODUCED) |
| 4 | **MIXED** | anything else (attenuated bias, partial coverage failure, RAIL-EMERGENT, 1D/2D split, wings disagreeing with centre) | not reached (and would be NO: no attenuation, no partial failure, RAIL-EMERGENT did not fire, no 1D/2D split, wings agree with centre) |

> ### BRANCH FIRED BY THE TREE: **TRANSFER-CONFIRMED**
>
> - **Headline verdict channel (VT-D6): 1D.** T-c(0.730) N = 400 1D = **COLLAPSE-REPRODUCED**.
> - **Secondary channel reported alongside: 2D.** T-c(0.730) N = 400 2D = **COLLAPSE-REPRODUCED**.
> - **1D/2D split: NONE** (all six decision cell × channel entries agree) ⇒ no MIXED routing.
> - **Truth-uniformity leg:** wings 0.690 and 0.770 both COLLAPSE-REPRODUCED in both channels ⇒
>   the pattern is uniform across truths, not a single-truth artifact.
> - **Verdict-line disclosures required by the prereg:** ANCHOR-MARGINAL — *not* triggered;
>   RAIL-EMERGENT — *not* triggered; EDGE-CONTAMINATED — *nowhere*.
>
> **NOT ADJUDICATED HERE.** Per the registered model/effort policy the branch call is presented to
> the author. The author's ruling is what makes this a verdict of record; only the Ship agent may
> append a verdict block below the prereg's verdict line, and only if the author hands one down.

Pre-stated meaning attached to this branch by the prereg (quoted, not endorsed): *the σ_z-dosed
coverage DEFECT survives production-matched population, multiplicity, and GLADE σ_z heterogeneity
(spec-z tail included) ⇒ it is the production mechanism **candidate** for what the estimator does
under GLADE photo-z, alongside (not replacing) the starvation account's railing shape (R3).*

---

## 5. §9 NOT-EVALUABLE registry — rows carried forward

| # | item | status after this read | note |
|---|---|---|---|
| 1 | Estimator code-path identity (axis d) | **NOT-EVALUABLE — carried** | the gate mirror, not `BayesianStatistics`. Certification chain V-T5 + T-0/T-a anchors: **all PASS**. Any estimator fix routes `/physics-change` (R6). |
| 2 | `volume_deconv` kernel form | **NOT-EVALUABLE — carried** | **O2 arm reserved (+47000…47399), NOT BUILT** — zero seeds realized in that block (verified). |
| 3 | Per-galaxy rate weights `R_eff(M_g)/(1+z_g)` | **NOT-EVALUABLE — carried** | **W1 arm reserved (+46000…46399), NOT BUILT** — zero seeds realized (verified). VT-D2 bracketing argument stands; author may order it post-read. |
| 4 | `f_incl < 1` / empty-ball events / completeness | **NOT-EVALUABLE — carried** | the 606 zero-ball events excluded (VT-D5); read is conditional on host-in-ball (`f_incl = 1.0` on all 1400 seeds). |
| 5 | Window-interior n(z) shape (GLADE clustering, completeness roll-off inside W_i) | **NOT-EVALUABLE — carried** | impostors stay `w_pop\|W`; concentration bracket VT-D2. |
| 6 | Sky-cone geometry / per-event sky selection | **NOT-EVALUABLE — carried** | no sky in the mirror (v2 §9 item 5 residue). |
| 7 | With-BH-subset 2D ball realism | **NOT-EVALUABLE — carried** | VT-D6: 2D applies `g_i` over the **same 1D ball**; production's 2D ball is the with-BH subset (1294/1588 empty). The 2D verdict is secondary. |
| 8 | **DS-5 width-vs-F5 fine read** | **NOT-EVALUABLE — carried** | matched-population F5 run remains the registered follow-up (v2 §9 item 3). The delta-narrow companion (§2.5) is reported **un-banded** only — no width claim is made. |
| — | DS-7 generator closure | **N/A in this venue** (VT-D8) | no accept/reject generator (VT-D1); the R5 OPEN form call is untouched by this read. |

**Also carried:** the transfer leg itself moves from NOT-EVALUABLE (v2 §9 items 2/5) to
**EVALUATED** by this campaign — pending the author's ruling on the fired branch.

---

## 6. DISCLOSURE LIST

**Pending author ratification (prereg §11 — all three, verbatim status):**

1. **D-VT-1 — §11 note 1 (2026-08-11): array 6252702 runtime blowout + resubmission.**
   49 tasks at `--time=04:00:00` → **10 COMPLETED / 39 TIMEOUT**, no partial output (the instrument
   writes its chunk JSON only at completion). Root cause operational: `mp.Pool` parallelizes over
   **seeds**, so a 25-seed chunk cannot finish faster than one seed's single-process wall
   (≈ 3.8–3.95 h). Seeds, seed→cell map, chunking, bands, statistics, thresholds and the instrument
   commit untouched; only `--time`/`--cpus-per-task` changed. **NON-STATISTICAL.** Abort (a) does
   not trip (3.79 vs 8.66 CPU-h/seed) ⇒ no N-floor fallback. **PENDING AUTHOR RATIFICATION.**
2. **D-VT-2 — §11 addendum 1 (2026-08-12): V-T5 compliance-order / sequencing deviation.**
   The prereg required the full §11 validity evidence *before* the campaign; the launch-phase
   validate skipped V-T5 (`validate_results_novt5.json`, `v_t5.pass = null`) and the full run
   (`validate_results_full.json`, **V-T5 PASS**) completed only after the first, partially
   timed-out array. No statistical content; the check itself passes. **PENDING AUTHOR RATIFICATION.**
3. **D-VT-3 — §11 addendum 2 (2026-08-12): second straggler resubmission (contention).**
   Array 6253922 (39 tasks, 9 h, 25 cores) → 17 COMPLETED / 22 TIMEOUT; packed 25-core tasks run
   ~1.6–1.9× slower than the uncontended 64-core reference (memory-bandwidth contention; completed
   walls 6:08–7:38 vs the 3:56 reference). Remaining 22 resubmitted as array **6259842**
   (`--time=24:00:00`, same grain, same registered chunks) → **all 22 COMPLETED**, sacct-verified
   2026-08-13, zero FAILED/TIMEOUT. **NON-STATISTICAL. PENDING AUTHOR RATIFICATION.**

**Further disclosures materially affecting how the numbers should be read:**

4. **D-VT-4 — worker grain 64 → 25.** 10 first-wave chunks embed `workers = 64`, 39 resubmitted
   chunks embed `workers = 25`. Result-invariance is certified by V-T2 and corroborated here:
   every `real_k` seed at **both** worker counts reproduces `K_sum = 1,193,703` exactly, and the
   seed→cell map is unchanged. Nothing statistical moved with the grain.
5. **D-VT-5 — post-campaign package/repo rename vs the V-T4 import-path wording.**
   V-T4 names `master_thesis_code/` + `master_thesis_code_test/`; after the campaign the package
   was renamed to `darksiren_emri` (commit `227e7a32`, a descendant of both campaign commits) and
   the repo dir `MasterThesisCode` → `darksiren-emri`. Every chunk's `import_path_clean` flag was
   evaluated against the then-current name at run time and is a correct clean-rule evaluation of
   the repo state at run time. The import-path diff `2ece8801..e93f3068` is empty under **both**
   naming conventions (re-verified). **Naming shift, not a defect.**
6. **D-VT-6 — two run commits + the prereg's own §11 append.**
   `2ece8801` (10 chunks) is the registered instrument commit; `e93f3068` (39 chunks) is a
   descendant with an **empty import-path diff** (R1-ratified D-4/D-5 pattern). The 4 files that
   differ are markdown under `results/`; one is the prereg itself, and its diff is a **pure append**
   at line 608 into §11 ⇒ VT-D0(ii) holds. On-disk prereg above §11 is byte-identical to registered
   `e77eecad`.
7. **D-VT-7 — MAP-bias statistic definition.** DS-VT3 registers **grid-argmax** bias (and the
   committed v2 anchors are grid-argmax). This readout uses grid-argmax as primary and reports the
   refined companion; **every status and classification is identical under both** (max difference
   across all 12 cell × channel entries < 0.0002). The upstream `collect_raw.json` reported the
   refined variant.
8. **D-VT-8 — T-a is exempt from the ΣK census pin by registered design.** T-a runs
   `balls = "poisson4"` (§5 / VT-D2), so `K_sum` is a per-seed Poisson-λ=4 draw (4757–5128, 137
   distinct values), **not** a pin mismatch. The five `real_k` cells match 1,193,703 exactly on all
   1200 of their seeds. An earlier draft of the upstream extraction applied the pin to T-a and
   produced a spurious "mismatch"; corrected before finalizing (never reached `collect_raw.json`).
9. **D-VT-9 — the PIT is saturated, not marginal.** Per-seed PIT sits at ~1e-20 and below in every
   dosed cell ⇒ KS D ≈ 1.0. Coverage is not degraded, it is **absent**: 0/1200 dosed seeds contain
   the truth at any HPD level, both channels. T-0's PIT is identically 0.5 on all 200 seeds
   (degenerate) — the reason for the VT-D8 exemption; its 1.000 HPD carries no information.
10. **D-VT-10 — realized dose slightly above the pre-registered prediction.** VT-D3 predicted
    σ̄_pairs ≈ [0.039, 0.041]; realized T-c values are 0.041452 / 0.041775 / 0.042082. R_dose is
    computed against the **realized** σ̄ as registered, so the band comparison is unaffected — and a
    larger σ̄ mechanically **lowers** R_dose, i.e. this works against, not for, the CONFIRMED call.
11. **D-VT-11 — compute context.** Summed wall across the 49 retained chunks = **303.1 h**. Median
    wall/seed: T-0 8 s, T-a 1 s, T-b 1172 s, T-c 1007–1128 s. At 25-seed chunks on 25 cores the
    wall/seed is a *contended* per-seed CPU proxy (~7–8 CPU-h); the registered abort-(a) evidence is
    the **uncontended 3.79 CPU-h/seed** figure in §11. Lesson already filed to the perf roadmap:
    size margins against contended timing, or one task per node.
12. **D-VT-12 — scope.** No production posterior exists in this read (prereg §0). Every posterior is
    a synthetic-universe diagnostic quotable only against its own truth. The coverage notion is
    **conditional (fixed-design) frequentist coverage** over noise + ball + σ_z randomness at fixed
    truth (VT-D1), on the 982 nonempty-ball events (VT-D5), with `f_incl = 1`.

---

## 7. FORMULATION AWAITING THE AUTHOR'S RULING

*Nothing in this section is adjudicated. It states what the fired branch would mean, and what the
author is being asked to decide.*

### 7.1 The 1D rail account: starvation vs the σ_z-dosed co-candidate

- **What this read settles (if the author accepts the fired branch).** The σ_z-dosed coverage
  DEFECT is no longer confined to the v2 caricature. It survives, simultaneously: the real detected
  event population (T-a), the real per-event candidate multiplicities (T-b: ΣK = 1,193,703 vs the
  caricature's ≈ 7.5 k — a ~160× increase in candidates that maximally favours dilution under the
  registered equal-weight convention), and the real heterogeneous GLADE per-galaxy σ_z with the
  spec-z tail natively present (T-c). It survives at **all three truths** and in **both channels**,
  with `R_dose` ∈ [0.877, 0.975] — i.e. the estimator's MAP is displaced by **≈ +1 × the realized
  photo-z σ**, with posteriors ~8.5–10 σ narrower than that displacement.
- **What this read does NOT settle.** This venue produced **zero rails** (0/1400 seeds, both
  channels, every cell). The production **1D railing shape** — the A-1D starvation rail 400/400
  (quotable per R2) — is *not* reproduced here and is *not* explained by this mechanism.
  The two accounts therefore remain **compatible, not competing**, exactly as R3 registered them:
  **starvation owns the railing shape; the dosed coverage collapse is the candidate for what the
  estimator does underneath it** (a uniform +≈σ_z displacement with delta-narrow posteriors).
  The pre-named RAIL-EMERGENT pattern — which would have made this a *shape* transfer — did not fire.
- **The author's call:** whether the co-candidate is promoted from "named owner-candidate thread"
  (R3) to a load-bearing production-mechanism candidate, and in what wording it may be quoted.

### 7.2 Paper #47's hold reason

- Current hold reason of record (R6): *"P–P leg FAILED — coverage DEFECT; fix routes through
  `/physics-change`."*
- Mechanical consequence of the fired branch, as pre-stated by the prereg: the hold **stands as
  upgraded by R6**, with the transfer leg now **EVALUATED-CONFIRMED** (it was a NOT-EVALUABLE row
  before this campaign). **Nothing here lifts or weakens the hold**, and nothing here mechanically
  changes the stage-5 conjunction.
- **The author's call:** whether the hold wording is amended to cite the venue-transfer evidence.

### 7.3 `/physics-change` intake — PREPARED, **NOT OPENED** (AUTHOR-GATED)

Prereg §0 and branch 2: a TRANSFER-CONFIRMED escalation routes through `/physics-change` intake on
the **estimator's photo-z handling**. This readout **prepares** the intake and **does not open it**;
no production physics file was touched.

- **Target of record:** the estimator's photo-z handling — the per-candidate redshift-kernel
  treatment in the H₀ likelihood (production analogue of the mirror's bare
  `N(z; z_obs,k, σ_z,k)` × distance-likelihood form, prereg §4 step 5).
- **Measured symptom to be explained by any proposed fix:** a uniform **positive** MAP displacement
  of magnitude ≈ +1 × σ̄ (`R_dose` 0.877–0.975 across truths and channels), delta-narrow posteriors
  (bias / post_sd ≈ 8.5–10), and 0/N HPD coverage at all three levels — the estimator is
  *confidently wrong by about one photo-z σ*.
- **Axes the prereg names but this campaign did NOT measure** (each must be stated as open in the
  intake): kernel **form** `volume_deconv` (O2 reserved, NOT BUILT, §9 row 2); per-galaxy **rate
  weights** (W1 reserved, NOT BUILT, §9 row 3); the **completeness / out-of-catalogue** term for the
  606 empty-ball events (§9 row 4); production **code-path identity** (§9 row 1 — this read used the
  certified mirror, V-T5 PASS).
- **Gate:** the full `/physics-change` protocol (derivation, dimensional analysis, limiting case,
  literature reference, regression test, `PHYSICS-GATE-LEDGER` row) applies before any edit.
  **State: PREPARED, NOT OPENED — awaiting the author's explicit order.**

### 7.4 Decisions this readout asks the author for

1. Ratify or reject the three §11 deviation notes (**D-VT-1 / D-VT-2 / D-VT-3**).
2. Rule on the branch the tree fired (**TRANSFER-CONFIRMED**) — this readout does not adjudicate.
3. Decide whether to order the reserved **W1** (per-galaxy rate weights) and/or **O2**
   (`volume_deconv` kernel form) arms.
4. Decide whether to open the `/physics-change` intake on the estimator's photo-z handling.

---

*Generated by `results/venue_transfer_20260811/score_venue_transfer.py` (read-only on all campaign
inputs and on the prereg). Machine-readable twin: `VENUE_TRANSFER_READOUT.json`.*
