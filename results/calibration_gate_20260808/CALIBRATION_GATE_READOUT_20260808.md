# Calibration-gate readout — campaign 2026-08-08 (scored 2026-08-10)

**Prereg of record:** `PREREGISTRATION_CALIBRATION_GATE.md`, commit
`b50ccc65a544648fb5f07e4cf2ec273a32be4170` (bands locked blind, single-file
commit, parent `2c6fdbc7`). **Scoring:** mechanical — every statistic
re-derived from the raw per-seed/aggregate values in the 9 registered
`*_results.json` files and scored against the §7/§8/§10 bands quoted
verbatim; the instrument's own status labels were cross-checked and agree
in **all** cell×channel×statistic combinations (zero mismatches). Scorer:
`readout_score.py` → `CALIBRATION_GATE_READOUT_20260808.json` (this file
quotes that JSON). No band was adjusted; no judgment call enters the branch
determination (the two author-call flags below do not change the branch).

**Campaign integrity (scored, all PASS):** all 9 registered cell×truth
JSONs present, rc=0; seed plan exact (3250/3250 seeds, every cell's block
contiguous and complete, zero cross-cell collisions); configs match the §5
table verbatim (independently re-verified + campaign pytest); zero
non-finite `ln_post` anywhere (abort (b): max fraction 0.0 vs 1% threshold);
per-seed schema = §6 fields + 2 additive extras; wall time 3h28m vs
budgeted 3.9–5.0h; O1 not run (correctly registered NOT-EVALUABLE);
V2 suite re-run at readout: 21/21 PASS.

---

## 1. Decision statistics vs locked bands (per cell × channel)

DS-1: PASS = all of C50/68/90 in 2σ bands [0.450,0.550]/[0.633,0.727]/[0.870,0.930];
FAIL = any outside 3σ [0.425,0.575]/[0.610,0.750]/[0.855,0.945].
DS-2: PASS D≤0.0679, FAIL D>0.0814 (N=400). DS-3: IN-BAND |b|≤0.010,
DEFECT-SCALE |b|≥0.030. Edge guard §8: cell×channel with >10% edge-loaded
seeds ⇒ DS-1/DS-2 carry **no gate weight**.

| cell (truth) | ch | DS-1 C50/68/90 | DS-1 | DS-2 D | DS-2 | DS-3 bias | DS-3 | DS-4 R_low/R_high | edge-loaded | gate weight (§8/§5) |
|---|---|---|---|---|---|---|---|---|---|---|
| A (0.690) | 1D | 0/0/0 | FAIL | 1.000 | FAIL | −0.0900 | DEFECT | 1.000/0.000 | 100% | exempt (§5: 1D starved anchor) + contaminated |
| A (0.690) | 2D | .248/.403/.633 | FAIL | 0.2945 | FAIL | +0.0452±.0036 | DEFECT | 0.030/0.038 | 92.8% | **EDGE-CONTAMINATED — none** |
| A (0.730) | 1D | 0/0/0 | FAIL | 1.000 | FAIL | −0.1300 | DEFECT | 1.000/0.000 | 100% | exempt + contaminated |
| A (0.730) | 2D | .308/.413/.623 | FAIL | 0.2208 | FAIL | +0.0372±.0034 | DEFECT | 0.020/0.085 | 91.2% | **EDGE-CONTAMINATED — none** |
| A (0.770) | 1D | 0/0/0 | FAIL | 1.000 | FAIL | −0.1700 | DEFECT | 1.000/0.000 | 100% | exempt + contaminated |
| A (0.770) | 2D | .328/.468/.708 | FAIL | 0.0914 | FAIL | +0.0204±.0030 | MIXED | 0.003/0.190 | 91.2% | **EDGE-CONTAMINATED — none** |
| B0 (0.730) σ_z=0 | 1D | 1/1/1 | FAIL | 0.500 | FAIL | 0.0000 | IN-BAND | 0.000/0.000 | 0% | dose/control cell |
| B0 (0.730) σ_z=0 | 2D | 1/1/1 | FAIL | 0.500 | FAIL | 0.0000 | IN-BAND | 0.000/0.000 | 0% | dose/control cell |
| B1 (0.730) σ_z=.010 | 1D | 0/0/0 | FAIL | 1.000 | FAIL | +0.0109±.0001 | MIXED | 0.000/0.000 | 0% | dose cell |
| B1 (0.730) σ_z=.010 | 2D | 0/0/0 | FAIL | 1.000 | FAIL | +0.0112±.0001 | MIXED | 0.000/0.000 | 0% | dose cell |
| **B2 (0.690)** σ_z=.035 | 1D | 0/0/0 | **FAIL** | 1.000 | **FAIL** | +0.0349±.0001 | DEFECT | 0.000/0.000 | 0% | decision cell, weight-bearing¹ |
| **B2 (0.690)** | 2D | 0/0/0 | **FAIL** | 1.000 | **FAIL** | +0.0352±.0002 | DEFECT | 0.000/0.000 | 0% | decision cell, weight-bearing¹ |
| **B2 (0.730)** | 1D | 0/0/0 | **FAIL** | 1.000 | **FAIL** | +0.0354±.0001 | DEFECT | 0.000/0.000 | 0% | decision cell¹ |
| **B2 (0.730)** | 2D | 0/0/0 | **FAIL** | 1.000 | **FAIL** | +0.0358±.0002 | DEFECT | 0.000/0.000 | 0% | decision cell¹ |
| **B2 (0.770)** | 1D | 0/0/0 | **FAIL** | 1.000 | **FAIL** | +0.0371±.0002 | DEFECT | 0.000/0.000 | 0% | decision cell¹ |
| **B2 (0.770)** | 2D | 0/0/0 | **FAIL** | 1.000 | **FAIL** | +0.0382±.0003 | DEFECT | 0.000/0.000 | 0% | decision cell¹ |
| V1 (0.730) control | 1D | 1/1/1 | no band at N=50 | 0.500 | no band | 0.0000 | IN-BAND | 0.000/0.000 | 0% | §10 control (see V1 below) |
| V1 (0.730) control | 2D | 1/1/1 | no band at N=50 | 0.500 | no band | 0.0000 | IN-BAND | 0.000/0.000 | 0% | §10 control |

¹ …but the branch below voids all measurement cells before any DS-1/DS-2
value can be read as a calibration verdict (V4 failure ⇒ texture cells void;
every run cell is `dl_binned`).

**Structural-degeneracy flag (B0, V1, raw pattern, not scored):** at
σ_z = 0 the posteriors are near-delta (post_sd ∼ 1e−7, MAP = truth exactly
for all seeds) ⇒ PIT ≡ 0.5, C_β ≡ 1, KS D ≡ 0.5 **by construction**. The
FAIL labels at B0 are the degenerate-PIT artefact of a point posterior at
truth, not ordinary miscoverage; the prereg registered no exemption for it,
so the mechanical label stands as printed with this flag attached.

## 2. DS-6 — rail-reproduction contrast (the Q2 statistic): **MIXED**

Locked thresholds 0.90/0.05, 1D channel:

- R_low(B2-1D) = **0.000 / 0.000 / 0.000** at truths 0.690/0.730/0.770 — the
  RAIL-REPRODUCED condition (≥0.90 at all three) **fails**.
- R_low(B2-1D) ≤ 0.05 holds, **but** RAIL-NOT-REPRODUCED additionally
  requires B2-1D to PASS DS-1 and DS-2 — B2-1D **FAILS both** at all three
  truths ⇒ condition **fails**.
- ⇒ **MIXED** (the prereg's "otherwise" clause, first-class and non-forcing).
- Registered dose–response R_low(1D | σ_z): 0.000 @ σ_z=0 → 0.000 @ 0.010 →
  0.000 @ 0.035. No truth-dependence (0.000 at all three B2 truths).
- R_low(B0-1D) = 0.000 ≤ 0.05: the pre-named "impostor-ball analog of the
  N-2 finding" (railing under confusion at perfect z) did **not** occur.
- Single-host anchor reproduced: A-1D R_low = 1.000 at all three truths
  (the committed 200/200 starvation signature, now 400/400 ×3).

Raw pattern accompanying MIXED (reported, not adjudicated): in the ball
venue the 1D channel does not rail — it acquires a uniform **positive** MAP
bias ≈ +σ_z (+0.0109 at σ_z = 0.010; +0.0349…+0.0371 at σ_z = 0.035, weakly
increasing with truth) with posteriors far too narrow for it (sd_med
≈ 0.003 ≪ bias) ⇒ 0/0/0 coverage without any railing.

## 3. DS-7 — in-loop generator-closure identity (leg-2, in-loop form)

Registered §7 statistic: |N_det/(⟨n_drawn⟩·p̄) − 1| ≤ 0.05, raw.

| cell | raw ratio | raw pass | corrected ratio | corrected pass |
|---|---|---|---|---|
| A 0.690/0.730/0.770 | 0.9081 / 0.9458 / 0.9266 | FAIL / FAIL / FAIL | 1.0083 / 0.9973 / 1.0001 | PASS ×3 |
| B0 / B1 | 0.9493 / 0.9505 | FAIL / PASS | 1.0010 / 1.0022 | PASS ×2 |
| B2 0.690/0.730/0.770 | 0.9050 / 0.9510 / 0.9272 | FAIL / PASS / FAIL | 1.0049 / 1.0028 / 1.0008 | PASS ×3 |
| V1 | 0.9552 | PASS | 1.0072 | PASS |

**Registered raw form: VIOLATED in 6/9 cells** (−5.4% to −9.5%). The
instrument's granularity-corrected companion (documented at build time as
compensating a known 4096-batch proposal-counting undercount in the parent's
`draw_universe`) passes 9/9 (0.997–1.008). **Author call (module
divergence-log item 9, flagged, not exercised here):** which form carries
V-class weight. Scored mechanically, the registered raw form is what §10's
trigger set names, so the violation stands for branch purposes.

## 4. Validity controls (§10) — scored

| control | result | evidence |
|---|---|---|
| V1 plumbing control | **PASS** | MAP = 0.730 exactly, both channels, 50/50 seeds (unique-value set {0.73}); the build-time-predicted ~9% tail-outlier risk did not manifest |
| V2 HPD-port certification | **PASS** | 21/21 pytest re-run at readout, incl. boolean-exact agreement with `pp_coverage._hpd_contains` |
| V3 determinism | **PASS** | `validate_results.json` v3.pass=true + adjudication P1 smoke re-runs bit-identical (per_seed_identical=true, V1 and B2) |
| **V4 texture certification** | **FAIL** | corr(ln σ_dL, ln d_L) median 0.6637–0.6662 across all 9 cells vs locked band **[0.72, 0.92]**; recomputed independently from per-seed records, agrees with instrument |
| V5 R0 reproduction | **PASS** | rtol 1e−12, zero mismatches |
| Config provenance (dirty tree) | **VIOLATED** | §10: "Runs that would execute on a dirty tree STOP instead" — all 9 registered cells ran `git_dirty=true, allow_dirty=true`; the instrument module `master_thesis_code/validation/calibration_gate.py` is **still untracked/uncommitted** (no git code identity); prereg §11 appendix ("code commit appended when it exists") is **empty** |
| Abort (b) non-finite | not triggered | 0 non-finite ln_post in 3250 seeds (threshold 1%) |

**V4 consequence, § 10 verbatim:** "Failure ⇒ the texture cells are void;
`independent`-texture cells are unaffected." Every one of the 9 run cells
(including V1) has `sigma_texture="dl_binned"` — **there is no
independent-texture cell in the campaign**, so V4 failure voids the entire
measurement layer. Context (does not alter the score): the module's own
build-time docstring (divergence-log item 7) **pre-declared** this failure
(decile rank-matching attenuates the CSV's 0.816 to ≈0.69±0.02, "if it
fails, the texture cells are void per §10"); that prediction was never
appended to §11 as an amendment, and §7's anti-tuning clause forbids
adjusting the ±0.10 band after readout — the band stands, the FAIL stands.
Measured 0.664–0.666 is below even the pre-declared ≈0.69.

## 5. Branch determination (prereg "Branches", applied with zero judgment)

**GATE-NOT-TRUSTWORTHY trigger set** = {V1…V5 failure, DS-7 violation,
abort (b)} ∪ {both decision cells EDGE-CONTAMINATED in the channel read}.

| trigger | fired? |
|---|---|
| V1 / V2 / V3 / V5 failure | no |
| **V4 failure** | **YES** |
| **DS-7 violation (registered raw form)** | **YES (6/9 cells)** — author call open on the corrected form (9/9 pass) |
| abort (b) | no |
| both decision cells edge-contaminated, 2D read | no (A-2D yes at all truths ×91–93%; B2-2D no, 0%) |
| both decision cells edge-contaminated, 1D read | no (B2-1D 0%; A-1D exempt from 1D gate reads per §5) |

# ⇒ **BRANCH FIRED: GATE-NOT-TRUSTWORTHY**

Prereg, verbatim: *"The instrument's own verdict is void; report which
control failed and why; no stage-4 leg-1 claim of any kind may be made."*
The failed controls: **V4** (texture certification, voiding all-`dl_binned`
measurement cells) and **DS-7 in registered raw form** (6/9 cells).
KEEP-DIGGING, REPORT-BOUND and MIXED are all unreachable: each requires
"gate trustworthy" as its first conjunct. Note the branch does **not**
hinge on the author's DS-7 raw-vs-corrected call: V4 alone fires it.
The dirty-tree/§11 provenance violations are outside the enumerated trigger
set and are reported as separate findings, not branch inputs.

## 6. Stage-4 gate table (docs/RESEARCH_CYCLE.md), each leg evaluated or NOT-EVALUABLE

| leg | doc requirement | status from this campaign |
|---|---|---|
| **1 — SBC/P–P coverage** of the full 2-channel estimator, A3 (i)(ii)(iii) | coverage verdict counts only if harness accepted and instrument trustworthy | **NO CLAIM PERMITTED** — the branch voids leg 1 entirely. A3 acceptance itself: (i) g recomputed per h — built and inherited from `77b524af`; (ii) N=1500 — met; (iii) multi-candidate balls — built (K_mean≈5.0, ~2.4M impostors/cell). The harness exists and is A3-conformant, but its **coverage verdict is void** (V4 + DS-7-raw). |
| **2 — generator-closure absolute-count audit** | in-loop form = DS-7; production result stands separately | In-loop: **registered raw form VIOLATED 6/9**, corrected form passes 9/9 — author call. Production leg-2 result (D̃/D=0.926, p0-window CONFIRMED) **not re-run here** by design; its venue transfer remains **PENDING-AUTHOR-CONFIRMATION** (§9 item 2). |
| **3 — forecast-consistent width** | measured width on the F5 stage-1 forecast | Coarse screen (DS-5): **SCREEN-NOT-EVALUABLE at exact venue points** — the committed F5 sweep has no σ_z node at {0, 0.010, 0.035} and no matched-population run was executed (§9 item 3, registered follow-up). Bracket reads at the nearest committed nodes, raw context only: no bracketing reading places any ball-cell 1D W = σ_med/σ_F5 inside [0.5, 2.0] (all W ≪ 0.5; e.g. B2-1D σ_med ≈ 0.0030 vs bracketed σ_F5 ≈ 0.011–0.112). 2D channel NOT-EVALUABLE (F5's 2D axis is the with-BH-mass channel; the gate's 2D is the completion-leg g — §9 item 4). A cells NOT-EVALUABLE (no host-z axis). Fine read NOT-EVALUABLE as registered. |

§9 NOT-EVALUABLE registry (unchanged, all still open): production-side
no-unmodeled-selection (carried by leg 2 + the open f_k–pool-coupling
thread); leg-2 venue transfer; leg-3 fine read; R&V15 in-catalogue host-mass
kernel; GLADE n(z)/completeness/sky/f_incl<1; `volume_deconv` kernel form
(O1 not built).

Stage-5 decision table: **not reached.** The stop rule requires coverage
pass ∧ width-on-forecast ∧ no-unmodeled-selection; none of the three
conjuncts is establishable from a voided instrument.

## 7. What this means for the 1D rail and paper #47 — formulation for the author (no ruling made here)

**The mechanical outcome is GATE-NOT-TRUSTWORTHY: this campaign produces no
admissible stage-4 leg-1 evidence in either direction about the 1D rail.**
The standing account ([[h0-railing-rootcause-photoz]]: photo-z information
starvation) is neither confirmed nor weakened by an instrument whose own
registered validity controls failed; paper #47's hold (RUNBOOK-7 §4: the
P–P leg) is **not lifted** — the P–P leg still lacks a trustworthy
instrument, though an A3-conformant harness now exists and its defects are
enumerated, pre-declared in part, and cheaply repairable.

Raw patterns preserved for the eventual trustworthy re-run (explicitly NOT
claims — barred by the fired branch): (a) the single-host starvation rail
reproduced exactly (A-1D 400/400 railed low ×3 truths, matching the 200/200
anchor); (b) in the multi-candidate ball venue the rail did **not**
reproduce at any σ_z — instead the 1D channel shows a uniform positive
bias ≈ +σ_z with drastically over-narrow posteriors (0/0/0 coverage,
no railing), and the 2D channel tracks it; had the gate been trustworthy,
that pattern feeds DS-6=MIXED and DS-1/DS-2 FAIL in decision cells — the
KEEP-DIGGING(b)/DEFECT-class direction, **not** REPORT-BOUND; (c) B0
(σ_z=0, impostors present) is bias-free and exactly on truth — the
pre-named N-2-analog anomaly did not occur; (d) A-2D is 91–93%
edge-contaminated, so the chosen truths are not distal enough from the grid
edge for the single-host 2D coverage read at this venue.

**Decisions the formulation leaves to the author (each named by a
registered document, none exercised here):**
1. **DS-7 weight** — registered raw form (VIOLATED 6/9) vs build-documented
   granularity-corrected form (9/9 PASS): module divergence-log item 9
   explicitly reserves this to the author. (Branch-insensitive: V4 fires
   GATE-NOT-TRUSTWORTHY regardless.)
2. **V4 band** — the ±0.10 band cannot be adjusted post-readout
   (anti-tuning clause); if the author accepts the build-time attenuation
   analysis (decile rank-matching ⇒ ≈0.69, measured 0.664–0.666), the path
   is an appended §11 amendment + **re-registration and re-run**, not a
   re-score of this campaign.
3. **Provenance repair** — commit the instrument
   (`master_thesis_code/validation/calibration_gate.py` +
   `master_thesis_code_test/validation/test_calibration_gate.py`), append
   the code commit to §11, and re-run on a clean tree so the §10 dirty-tree
   STOP clause is honored; only then can any future readout carry weight.
4. Whether the trivially-degenerate σ_z=0 PIT/coverage labels (B0/V1)
   warrant a registered exemption or a widened-grid design in the re-run.

*Readout complete. Nothing above rules; the ledger, the claim files, the
book, and prereg §11 are untouched by this pass.*


---

*Note (2026-08-11): the four open author decisions listed above were resolved by the v2 cycle — see
`../calibration_gate_v2_20260810/PREREGISTRATION_CALIBRATION_GATE_V2.md` (deviation register D1–D8),
`CALGATE_V2_READOUT.md` (gate TRUSTWORTHY, KEEP-DIGGING(b)), and ledger row #98 + §5 (2026-08-11 continuation,
itemized confirmation queued 2026-08-12).*
