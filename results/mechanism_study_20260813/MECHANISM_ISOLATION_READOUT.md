# MECHANISM ISOLATION READ — the parent study, split-dose arms N-0 / E1-host / E1-imp

**Scored against** `results/mechanism_study_20260813/PREREGISTRATION_MECHANISM_ISOLATION.md`
(registered 2026-08-13, author-ratified "all approved"), with its companion `ARMS.md` and its
append-only **Amendment A1** (`AMENDMENT_A1_VM1_NULL_AT_N100.md`, registered `73141160`,
author-ratified 2026-08-13, verdict **A1-PASS** appended 2026-08-14).
**Arms:** `MN0` (N-0), `MEH` (E1-host), `MEI` (E1-imp) at instrument commit `e83ed0b9`;
`MN0X` (Amendment-A1 null, N = 100) at `3aedbe55`. **Data commit:** `5b0bd17a`.
**Scorer:** `results/mechanism_study_20260813/score_mechanism_isolation.py` → machine-readable twin
`MECHANISM_ISOLATION_READOUT.json`.
**Companions already filed, built upon and not re-adjudicated here:** `A1_READOUT.md`
(V-M1 at N = 100), `SCAN_READOUT.md` (the 2-D dose scan), `VM5_GOLDEN_20260814.md` (V-M5).

> **Every number below is recomputed from the raw per-seed `ln_post_1d` / `ln_post_2d` vectors.**
> No `aggregate` block, no orchestrator extraction and no figure was used as an input. The stored
> per-seed scalars were *compared against* the recomputation (§1.10): agreement is exact, 0.0
> relative deviation, on every field of every seed of all four arms.

> **This readout does not rule.** Parent §4: *branches are presented to the author, never
> self-adjudicated.* **No repair is proposed.** No registered document was edited: the parent,
> `ARMS.md`, `AMENDMENT_A1_*`, `PREREGISTRATION_2D_DOSE_SCAN.md`, the M1/M3/M4/M5 notes and
> `BIAS_HISTORY_LEDGER.md` are byte-unmodified.

---

## 1. VALIDITY FIRST

### 1.1 Seed-block integrity (parent §1, VT-D7 discipline; `ARMS.md` arm table)

**PASS, exact, all four arms.**

| arm | registered block | realized | n / unique | exact set match |
|---|---|---|---|---|
| MN0 (N-0) | base+50000…50014 = 20310808–20310822 | 20310808–20310822 | 15 / 15 | ✓ |
| MEH (E1-host) | base+50100…50114 = 20310908–20310922 | 20310908–20310922 | 15 / 15 | ✓ |
| MEI (E1-imp) | base+50200…50214 = 20311008–20311022 | 20311008–20311022 | 15 / 15 | ✓ |
| MN0X (A1) | base+50000…50099 = 20310808–20310907 | 20310808–20310907 | 100 / 100 | ✓ |

Pairwise intersections: MN0∩MEH = ∅, MN0∩MEI = ∅, MEH∩MEI = ∅, MN0X∩MEH = ∅, MN0X∩MEI = ∅;
MN0 ⊂ MN0X (the A1 §4.3 superset requirement). Every seed lies inside the parent's registered
decade +50000…+50999. **No collision anywhere.**

### 1.2 V-M3 — pin integrity

**PASS in all three arms** (and in MN0X, already scored in `A1_READOUT.md` §1.2).
`pin_integrity.pass = true` in every arm JSON, and each pinned field re-read here:

- CRB CSV md5 **`9a1f2a14384a9281c97ca3be312ddaab`** ✓ · frozeng emit md5
  **`34c50e91028b6a6458a2b145db545705`** ✓ (identical in MN0, MEH, MEI, MN0X)
- K census / ΣK = **1,193,703** ✓ · pruned-frame σ_z stats n = 20,834,171, median
  0.0393412950539589, min 0.0005317263419419, n<5e-3 = 231,098 ✓

### 1.3 Per-seed pins, verified on every seed of every arm

| pin | registered | MN0 (15) | MEH (15) | MEI (15) |
|---|---|---|---|---|
| `K_sum` | 1,193,703 | {1193703} | {1193703} | {1193703} |
| `n_events` / `n_events_run` | 982 / 982 | {982}/{982} | {982}/{982} | {982}/{982} |
| `f_incl` | 1.0 | {1.0} | {1.0} | {1.0} |
| `n_horizon_dropped` | 0 | max 0 | max 0 | max 0 |
| rails, 1D and 2D | zero | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 |
| non-finite `ln_post` | zero | 0 seeds | 0 seeds | 0 seeds |
| edge-loaded fraction | — | 0.000 | 0.000 | 0.000 |
| `sigma_z_mean_pairs` | dose-specific | **0.041813** | **0.000035** | **0.041786** |

The dosing check is the load-bearing one and it reproduces the parent's operational record exactly:
MEH's pair-mean σ̄ = 0.0000352 = 0.041813 × 982/1,193,703 = 0.0000344 (within the sampler noise the
2-D scan's §5.3 registered at 10 % for this column) — **exactly and only the 982 hosts are dosed**;
MEI's 0.041786 is the full GLADE dose on the 1,192,721 impostors with the hosts exact. **Neither
split arm is an un-dosed arm.**

### 1.4 V-M4 — clean rule

**PASS in all three arms.** `import_path_clean = true`, `allow_dirty = false`, `smoke = false`,
`dirt_inventory.import_path = []` in MN0, MEH, MEI (and MN0X). `git_dirty = true` throughout, with
every entry under `results/` — permitted, and consistent with every chunk in this thread.

### 1.5 V-M2 — generator invariance (ARMS.md AR-1 / AR-2 / AR-3)

**PASS on the evidence available, with one structural limitation disclosed (D-M-6).**

- The 11 registered unit tests `darksiren_emri_test/validation/test_venue_transfer_arms.py`
  (AR-1/AR-2/AR-3) **pass at HEAD**, re-run for this readout: `11 passed`.
- **AR-3 cannot be checked directly in the arm data**: the three arms were registered on
  *disjoint* seed blocks (`ARMS.md` arm table), so no fixed seed is shared between MN0, MEH and
  MEI and no cross-arm pre-dose comparison exists in these JSONs. AR-3 is a unit-test obligation,
  discharged there. Its observable consequences in the data all hold: `K_sum` pinned at the same
  1,193,703 in all three arms, `n_events = n_events_run = 982` everywhere, `f_incl = 1.0`
  everywhere, and the §1.3 dosing table showing that only the mask differs.
- **AR-1 is confirmed empirically and cross-commit** by A1-DET (`A1_READOUT.md` §1.6): the 15
  shared seeds compared across `e83ed0b9 → 3aedbe55` on 44 fields including both full 41-point
  `ln_post` vectors at **max relative deviation exactly 0.0**, MAPs exactly equal.

### 1.6 V-M5 — no-drift anchor (values golden, rtol ≤ 1e-12), and disclosure D-A1-2

**PASS. D-A1-2 is CLOSED as of 2026-08-14** — and the closure is *not* this readout's work.

`A1_READOUT.md` carried D-A1-2 forward as an **open** gap: V-M5 had been re-registered as a values
golden in parent §5 but had never been re-executed as that artifact at the current instrument
commit; only A1-DET bit-identity evidence existed, which certifies a different, still-registered
check. Carrying it forward was this session's task. **The artifact now exists**:
`results/mechanism_study_20260813/VM5_GOLDEN_20260814.md` (+ `.json`, producer
`verify_vm5_golden.py`), filed by a parallel session and committed at `38465df8`. Its result, read
here and adopted as an input:

| seed | V-M5 | MAPs exactly equal | max rel dev | worst field |
|---|---|---|---|---|
| 20286808 | PASS | YES (1d/2d + refined) | 1.6135e-14 | `pit_2d` |
| 20286809 | PASS | YES | 1.0406e-15 | `M_source_median` |
| 20286810 | PASS | YES | 1.0302e-15 | `M_source_median` |

Overall max relative deviation **1.6135e-14** against the registered ceiling **1e-12** — two orders
of magnitude inside — with the 1D channel bit-identical in every seed and all four MAP fields
exactly equal. **The registered V-M5 condition is met and no STOP fired.**

**Honest status statement, since this readout consumes an artifact it did not produce:** the V-M5
artifact is one day old, was produced against the registered committed reference
(`results/calibration_gate_v2_20260810/B2_h0p730_results.json`) on the registered v2 seeds using
the same instrument path, and **the author has not ruled on it** — as with every readout in this
thread, it is presented, not adjudicated. If the author declines it, V-M5 reverts to *not
re-executed at the current commit*, which is an **open** check, not a **failed** one (see §1.9,
where that distinction is what branch 1 turns on). D-A1-3 (MEH/MEI sitting at the pre-refactor
commit `e83ed0b9`) is separately carried forward here as **D-M-5**.

### 1.7 V-M1 — null-arm reproduction, settled by Amendment A1

Registered rule (parent §5): *arm N-0 must reproduce the campaign's decision-cell bias within
±0.002.* Reference: campaign T-c(0.730), N = 400, 1D, **+0.037237 ± 0.000230**.

| measurement | bias | SE | \|Δ\| vs 0.037237 | ±0.002 window |
|---|---|---|---|---|
| MN0, N = 15 (the registered arm) | +0.034667 | 0.001579 | **0.002570** | **MISSED** (1.63σ of the arm SE) |
| **MN0X, N = 100 (Amendment A1)** | **+0.037250** | 0.000494 | **0.000013** | **INSIDE — 153.8× within** |

Recomputed here from raw vectors, independently of `score_a1.py`: MN0X 1D MAP histogram
{0.755:5, 0.760:11, 0.765:28, 0.770:46, 0.775:10}; Σ bias = 5(0.025) + 11(0.030) + 28(0.035) +
46(0.040) + 10(0.045) = 0.125 + 0.330 + 0.980 + 1.840 + 0.450 = **3.725**, /100 = **+0.037250**.
Δ = 0.037250 − 0.037237 = **+0.000013** = **0.024σ** of SE_diff = √(0.000494² + 0.000230²) =
0.000545. A1's registered rule returns **A1-PASS**; A1's §4.4 **A1-FAIL** clause does not fire.

**What that does to branch 1, stated exactly and without extension.** A1 §4.1: *"The V-M1 leg of
STUDY-CONFOUNDED is not satisfied by the null arm as measured at N = 100."* A1 §4.2, equally
binding: *"It does NOT retroactively pass the N = 15 MN0 result … MN0's V-M1 status at N = 15
remains FAILED."* Both are registered pre-data readings. **This readout applies them as written and
adds nothing**: the leg is not satisfied at the precision at which the window is a meaningful test,
and the N = 15 failure stands on the record as a failure of an under-powered check.

> **This is the one place where the parent's branch call is load-bearing on an unruled verdict.**
> A1-PASS is itself *presented, not adjudicated* (`A1_READOUT.md` §4.2 item 5). **The parent branch
> below is therefore conditional on the author ratifying A1-PASS.** If the author declines it, the
> V-M1 leg fires on the N = 15 arm as registered and branch 1 (STUDY-CONFOUNDED) fires ahead of
> everything in §5.

### 1.8 Abort criteria (parent §5 a–d)

| criterion | trigger | measured across MN0 / MEH / MEI | fires? |
|---|---|---|---|
| (a) non-finite `ln_post` | > 1 % of any arm's seeds | **0.0 %** (0 of 15 in each arm, both channels; 90 vectors scanned) | **NO** |
| (b) horizon drop | > 5 % | `n_horizon_dropped` max = **0** on every seed of every arm | **NO** |
| (c) any V-M failure | any | V-M1 §1.7 · V-M2 §1.5 · V-M3 §1.2 · V-M4 §1.4 · V-M5 §1.6 — none failed | **NO** |
| (d) L0 toy vs L1 instrument **disagree in sign** | any arm | see below — **literal reading: NO**; a magnitude disagreement of 100 % on MEI | **NO (literal)**, disclosed as **D-M-7** |

**Criterion (d), spelled out because it is the closest call in this readout.** The L0 toy's
registered split-dose values (parent §7, K = 50) against the instrument at production K̄ ≈ 1,216:

| arm | L0 toy | L1 instrument | signs |
|---|---|---|---|
| MN0 (all dosed) | +0.0334 | **+0.034667** | both positive — agree, and within 4 % |
| MEH (host only) | +0.0062 | **+0.004000** | both positive — agree |
| MEI (impostors only) | +0.0247 | **+0.000000** | toy positive; instrument **exactly zero** |

Criterion (d) as registered requires the two to *disagree in sign*. **0.000000 has no sign**, so on
the literal wording (d) does not fire. What MEI shows is not a sign flip but a **complete absence**:
100 % of the toy's predicted arm effect. That is a first-class finding about the toy's faithfulness
in the K regime, and it is handed to the author in §7 and §9 rather than resolved here — because
firing (d) would STOP the study and force *every* L0 closure (M3, M4, M5→M5′, W1) to be revisited,
and that consequence may not be reached on a reading this readout supplies for itself.

### 1.9 STUDY-CONFOUNDED trigger set — member by member (parent §4 branch 1)

| # | leg | fires? |
|---|---|---|
| 1 | arm N-0 fails to reproduce the campaign bias within ±0.002 | **NO**, per Amendment A1 (MN0X: \|Δ\| = 0.000013). Conditional on the author ratifying A1-PASS (§1.7). |
| 2 | **any validity check in §5 fails** — V-M1 | **NO** (same as leg 1) |
| 2 | V-M2 generator invariance / AR-1, AR-2, AR-3 | **NO** — 11/11 unit tests pass at HEAD; AR-1 confirmed cross-commit at 0.0 deviation; AR-3 not checkable in-data by design (D-M-6) |
| 2 | V-M3 pin integrity | **NO** — all pins re-verified, all three arms |
| 2 | V-M4 clean rule | **NO** — import path clean, `allow_dirty=false` in all three arms |
| 2 | V-M5 values golden | **NO** — PASS at 1.6e-14 vs rtol 1e-12 (`VM5_GOLDEN_20260814.md`, §1.6) |
| 2 | abort (a) / (b) / (c) / (d) | **NO / NO / NO / NO (literal)** (§1.8) |

> ### **STUDY-CONFOUNDED DOES NOT FIRE — 0 of its legs are satisfied on the evidence in hand.**
> Conditional on (i) the author ratifying **A1-PASS**, and (ii) the author accepting the
> **V-M5 golden filed 2026-08-14**. Both are presented verdicts, not rulings. **No validity check
> is failing; the one that was open (D-A1-2) has been closed by artifact, not by argument.**

### 1.10 Recomputation integrity — the stored scalars are confirmed, not adopted

Every per-seed scalar this readout uses was rebuilt from the raw 41-point `ln_post` vectors with
verbatim ports of `closed_loop_gfrac.posterior_readout` (grid argmax, parabolic-refined argmax,
posterior mean, rail flags) and `calibration_gate.pp_readout` / `hpd_contains` (trapezoid
normalisation on the non-uniform canonical grid, PIT, HPD 50/68/90, posterior sd, edge mass), then
compared field-by-field against the stored values.

| arm | fields compared | **max relative deviation, 1D** | **max relative deviation, 2D** |
|---|---|---|---|
| MN0 | 11 derived fields × 15 seeds × 2 channels | **0.0** | **0.0** |
| MEH | same | **0.0** | **0.0** |
| MEI | same | **0.0** | **0.0** |
| MN0X | 11 × 100 × 2 | **0.0** | **0.0** |

**Exact agreement everywhere.** The stored per-seed records are confirmed as faithful functions of
the committed `ln_post` vectors; nothing below rests on an unverified upstream extraction.

---

## 2. THE ARMS — recomputed from raw `ln_post`

### 2.1 1D (the registered headline channel)

| arm | dose | N | **bias** | sd (ddof=1) | **SE** | post_sd median | bias / post_sd | MAP values | rails | non-finite |
|---|---|---|---|---|---|---|---|---|---|---|
| **MN0** | all | 15 | **+0.034667** | 0.006114 | 0.001579 | 0.004265 | **8.13** | {0.755, 0.760, 0.765, 0.770, 0.775} | 0.000 | 0 |
| **MEH** | host only | 15 | **+0.004000** | 0.002070 | 0.000535 | 0.000187 | **21.4** | {0.730, 0.735} | 0.000 | 0 |
| **MEI** | impostors only | 15 | **+0.000000** | 0.000000 | 0.000000 | 0.000000 | — (0/0) | **{0.730}** | 0.000 | 0 |
| **MN0X** | all | 100 | **+0.037250** | 0.004943 | 0.000494 | 0.004386 | **8.49** | {0.755, 0.760, 0.765, 0.770, 0.775} | 0.000 | 0 |

**Arithmetic, shown per the rails.** A per-seed MAP bias is an integer multiple of the 0.005 grid
step in the fine region (all MAPs land there), so each arm mean is Σticks × 0.005 / N:

```
MN0 :  {0.755:3, 0.760:1, 0.765:6, 0.770:4, 0.775:1}
       3(0.025)+1(0.030)+6(0.035)+4(0.040)+1(0.045) = 0.520000 / 15 = +0.03466667   (104 ticks)
MEH :  {0.730:3, 0.735:12}      12(0.005) + 3(0.000) = 0.060000 / 15 = +0.00400000   ( 12 ticks)
MEI :  {0.730:15}                            15(0.000) = 0.000000 / 15 = +0.00000000   (  0 ticks)
MN0X:  {0.755:5, 0.760:11, 0.765:28, 0.770:46, 0.775:10} = 3.725000 /100 = +0.03725000 (745 ticks)
```

**All four values reproduce the known values quoted in the parent's operational record and in
`A1_READOUT.md` exactly** (+0.034667 ± 0.001579 / +0.004000 ± 0.000535 / +0.000000 ± 0.000000 /
+0.037250 ± 0.000494). Nothing is contradicted.

### 2.2 2D (reported alongside, parent §6 convention)

| arm | 2D bias | SE | post_sd median | MAP values | 1D − 2D |
|---|---|---|---|---|---|
| MN0 | +0.037000 | 0.001604 | 0.004315 | {0.755…0.775} | −0.002333 |
| MEH | +0.004333 | 0.000454 | 0.000262 | {0.730, 0.735} | −0.000333 |
| MEI | +0.000000 | 0.000000 | 0.000000 | {0.730} | 0.000000 |
| MN0X | +0.039750 | 0.000519 | 0.004407 | {0.755…0.775} | −0.002500 |

**No 1D/2D split anywhere**: every arm's 2D value tracks its 1D value in sign and magnitude, and —
checked explicitly in §3.1 — **every registered classification is identical in both channels**. The
parent §6 clause *"a 1D/2D split in any arm is itself a finding and forces the MULTI-TERM branch"*
therefore has no subject. 2D runs +0.0003…+0.0025 above 1D, all below one h-grid step, the same
ordering the campaign and the 2-D scan showed.

### 2.3 The structural fact behind the numbers — how the three posteriors are shaped

This is where the arms stop being three numbers and become three different objects. Recomputed
from the raw vectors:

| arm | grid points holding ≥1e-6 of the mass (median) | max single-point mass (median) | ln_post margin, best over 2nd-best (median) | ln_post dynamic range (median) | PIT (1D) |
|---|---|---|---|---|---|
| **MN0** | **9** | 0.447 | **0.365 nats** | 970 | 1.5e-24 … 3.5e-13 (median 6.2e-19) |
| **MEH** | **2** | 0.999 | 6.56 nats | 4,449 | 1.0e-12 … 0.49998 (median 6.7e-05) |
| **MEI** | **1** | **1.000** | **2,298.5 nats** | 48,201 | **exactly 0.5 in 15/15 seeds** |

- **MN0** is the campaign object: a ~9-point-wide posterior sitting **12.2 nats below its own peak
  at the true h**, i.e. displaced by 8.1× its own claimed width, with PIT saturated at ~1e-19.
- **MEI is not a "small bias" — it is a numerical delta at truth.** The true grid point beats its
  best competitor by a **median 2,299 nats** (min 1,562, max 3,031). The 15/15 landing on 0.730 is
  not a lucky draw at N = 15: a displacement of one grid step would require overturning e^−1562 at
  worst. **The arm's SE of exactly 0.000000 is a genuine degeneracy, not an under-powered
  measurement**, and no plausible N would move it.
- **MEH** is the interesting middle: an in-band bias (+0.004000) carried by a posterior so narrow
  (post_sd 0.000187) that the displacement is **21.4× its own width** — DS-M1's in-band edge reads
  the bias and does not read the width (D-M-9).

---

## 3. SCORECARD — the registered decision statistics

### 3.1 DS-M1 — per-arm bias classification (parent §3, edges applied verbatim)

Registered rule, unchanged: **TERM-OWNS** = |b| ≤ 0.010 **and** HPD90 ≥ 0.60 · **TERM-PARTIAL** =
0.010 < |b| < 0.030 · **TERM-INNOCENT** = |b| ≥ 0.030 **and** |b − b_N0| ≤ 0.004 · **OTHER** =
anything else, reported raw with direction stated.

| arm | ch | b | \|b\| | \|b − b_N0\| | HPD90 | **class** |
|---|---|---|---|---|---|---|
| MN0 | 1D | +0.034667 | 0.034667 | 0.000000 | 0.000 | **TERM-INNOCENT** |
| MN0 | 2D | +0.037000 | 0.037000 | 0.000000 | 0.000 | **TERM-INNOCENT** |
| **MEH** | 1D | +0.004000 | 0.004000 | 0.030667 | **0.333** | **OTHER** |
| **MEH** | 2D | +0.004333 | 0.004333 | 0.032667 | 0.267 | **OTHER** |
| **MEI** | 1D | +0.000000 | 0.000000 | 0.034667 | **1.000** | **TERM-OWNS** |
| **MEI** | 2D | +0.000000 | 0.000000 | 0.037000 | 1.000 | **TERM-OWNS** |
| MN0X | 1D | +0.037250 | 0.037250 | 0.002583 | 0.000 | TERM-INNOCENT |
| MN0X | 2D | +0.039750 | 0.039750 | 0.002750 | 0.000 | TERM-INNOCENT |

**Mechanics, arm by arm, with no interpretation added:**

- **MN0 → TERM-INNOCENT, trivially and by construction.** It alters nothing, so |b − b_N0| = 0 and
  |b| = 0.034667 ≥ 0.030. The null arm classifying TERM-INNOCENT is the design working, not a
  finding. MN0X classifies the same way against MN0 (|Δ| = 0.002583 ≤ 0.004), so the classification
  is robust to which null is used as the comparator.
- **MEH → OTHER.** |b| = 0.004000 ≤ 0.010 satisfies the first TERM-OWNS conjunct, but HPD90 =
  0.333 < 0.60 fails the second; |b| is below the TERM-PARTIAL floor (0.010) and below the
  TERM-INNOCENT floor (0.030). The registered residual class is **OTHER — reported raw, direction
  stated: positive, +0.004000 ± 0.000535, i.e. 7.5σ from zero in its own SE, and 21.4× its own
  posterior width.** No branch is forced by it.
- **MEI → TERM-OWNS.** |b| = 0.000000 ≤ 0.010 ✓ **and** HPD90 = 1.000 ≥ 0.60 ✓. Both conjuncts of
  the registered rule are satisfied, in **both channels**. **Disclosed alongside (D-M-2): the HPD90
  = 1.000 arises from the degenerate delta posterior of §2.3** — containment of h_true is trivially
  true when all the mass sits on the h_true grid point — so the coverage conjunct is satisfied
  without testing calibration in the sense DS-M2 was written for. **The rule as registered reads
  HPD90 ≥ 0.60 and nothing else; it is satisfied as written and this readout does not re-write it.**

### 3.2 DS-M2 — HPD coverage against the registered 2σ bands

Registered bands (binomial, the parent's N = 25 rows, applied unchanged): 0.500 ± 0.200 /
0.680 ± 0.187 / 0.900 ± 0.120. Arms ran at N = 15 (see D-M-1).

| arm | ch | HPD50 | HPD68 | HPD90 | inside bands (50/68/90) |
|---|---|---|---|---|---|
| MN0 | 1D | 0.000 | 0.000 | 0.000 | **no / no / no** (collapse, campaign-identical) |
| MN0 | 2D | 0.000 | 0.000 | 0.000 | no / no / no |
| MEH | 1D | 0.200 | 0.200 | 0.333 | **no / no / no** (under-coverage) |
| MEH | 2D | 0.133 | 0.200 | 0.267 | no / no / no |
| MEI | 1D | **1.000** | **1.000** | **1.000** | **no / no / YES** (**over**-coverage at 50 and 68) |
| MEI | 2D | 1.000 | 1.000 | 1.000 | no / no / YES |
| MN0X | 1D | 0.000 | 0.000 | 0.000 | no / no / no |
| MN0X | 2D | 0.000 | 0.000 | 0.000 | no / no / no |

**Reading, mechanically:** the null arms reproduce the campaign's total coverage collapse
(0.000/0.000/0.000, PIT–KS D = 1.000, PIT ~1e-19); MEH under-covers at every level; **MEI
over-covers at 50 and 68 — 100 % containment where 50 % and 68 % are expected — which is the
signature of a degenerate posterior, not of a calibrated one.** DS-M2 carries branch weight *only*
through the TERM-OWNS conjunction (parent §3), which reads HPD90 alone; §3.1 is therefore unaffected
by the 50/68 excursions, and they are reported here because omitting them would flatter the
TERM-OWNS call.

### 3.3 DS-M3 — dose-scaling of the residual: **NOT EVALUABLE**

Registered: *each surviving arm re-run at the two flat doses 0.011 and 0.035 (5 seeds each);
a term that OWNS the defect must remove the linearity — residual R_dose below 0.25 at both doses.*

**No flat-dose arm exists in this study.** All four arms are `sigma_mode = "glade"`; arm **E3**
(the extended dose ladder at the crossover) was never run, and no `flat011`/`flat035` cell was
produced under this prereg. **DS-M3 is NOT EVALUABLE and carries no weight in §5.** The 2-D dose
scan measures a dose response under its *own* pre-registration and has its own filed verdict; it is
not imported here as a substitute for DS-M3 (§6 states only what the two documents jointly
establish).

**Companion R_dose, reported UNBANDED** (parent's convention; the denominator is ill-defined for the
split arms because `sigma_z_mean_pairs` averages over dosed and undosed candidates):

| arm | bias | σ̄ (pairs) | R_dose = b/σ̄ | status |
|---|---|---|---|---|
| MN0 | +0.034667 | 0.041813 | **0.8291** | reported (matches the scan's quoted MN0 anchor) |
| MN0X | +0.037250 | 0.041796 | **0.8912** | reported (on the campaign's 0.8914) |
| MEI | +0.000000 | 0.041786 | **0.0000** | reported |
| MEH | +0.004000 | 0.000035 | 113.8 | **meaningless** — σ̄ here is the host dose diluted by 982/1,193,703; not a dose ratio |

### 3.4 DS-M4 — the W1 question: **NOT EVALUABLE on the instrument**

Registered: *arm A-M5b is classified WEIGHTS-MATTER (|b| changes by > 0.004) or WEIGHTS-INERT
(≤ 0.004).* **Arm A-M5b was withdrawn at registration** (parent §2: *"withdrawn — closed as a null
and shown to double-count"*), its seeds +46000…+46399 remain **reserved and unconsumed**, and no
weighting variant was run in this study. **DS-M4 is NOT EVALUABLE as registered.**

The question it was to answer was closed at L0 before registration (parent §7, W1). Scoring those
committed toy numbers against DS-M4's own ±0.004 edge, for the record only and with **no branch
weight** (parent §3: *"neither carries branch weight for the mechanism verdict"*):

| L0 variant (toy total +0.0334) | Δ|b| | vs the 0.004 edge | class as the edge reads |
|---|---|---|---|
| rate-shaped weights (+2 %) | +0.00067 | below | **WEIGHTS-INERT** |
| oracle weights at true z (+1 %) | +0.00033 | below | **WEIGHTS-INERT** |
| w_pop *inside the integral* (+28 %) | +0.00935 | above | exceeds — **but this is not a weight change**; it moves a term inside the z-integral |
| window renormalisation (+22 %) | +0.00735 | above | exceeds — likewise a change of the integrand's normalisation, not of the candidate weights |

**On the registered statistic — a change to the h-independent candidate weights — the answer is
WEIGHTS-INERT**, which is exactly the parent §6 pre-registered expectation ("the adjudicator's
common-mode reading (finding D11) predicts WEIGHTS-INERT"). The §6 clause that a WEIGHTS-MATTER
result would have made the 2026-08-13 drop-the-W1-arm decision wrong is therefore **resolved in
favour of that decision**, on the toy, as §7 already recorded. The two structural variants that
exceed the edge are named here so the row is not read as broader than it is.

---

## 4. THE CENTRAL REGISTERED CONFRONTATION — DS-M5

### 4.1 The registered prediction, and the measurement

Parent §2, verbatim: **"DS-M5 — the split-dose read (primary). M5′-CONFIRMED requires both:
E1-imp bias ≥ 0.030 and E1-host bias ≤ 0.012, at N = 15 (SE ≈ 0.0013). A split in the opposite
direction, or both cells large, refutes M5′ and returns the study to the M2′ arm."**

| conjunct | required | **measured** | satisfied? | distance |
|---|---|---|---|---|
| E1-imp (MEI) | **≥ 0.030** | **+0.000000** | **NO** | shortfall **0.030000**, i.e. the *entire* requirement |
| E1-host (MEH) | **≤ 0.012** | **+0.004000** | **YES** | margin 0.008000 = **14.97** MEH-SE |

> ### **DS-M5 → M5′ NOT CONFIRMED. The conjunction fails on its decisive half, and it fails
> completely: the impostor-only arm carries not "less than predicted" but exactly nothing.**

**The shortfall in σ.** MEI's own SE is exactly 0.000000 (zero spread over 15 seeds), so the
deficit has no finite expression in the arm's own units — which is itself the point: this is not a
noisy miss. Expressed against the comparators that do have spread:

```
against MN0  (SE 0.001579):  0.030000 / 0.001579 = 19.0 sigma
against MN0X (SE 0.000494):  0.030000 / 0.000494 = 60.7 sigma
```

And structurally (§2.3): reaching b = 0.030 would require MEI's MAP to move **six grid steps**
against a median **2,299-nat** margin at the true grid point. **The registered threshold is not
merely missed, it is unreachable by this arm.**

### 4.2 The direction is inverted

The prediction ordered the two cells **imp ≫ host** (≥0.030 vs ≤0.012), on the L0 toy's K = 50
split +0.0247 / +0.0062. The measurement orders them **host > imp**:

```
b(host) - b(imp)  =  +0.004000 - 0.000000  =  +0.004000   (+7.5 sigma in MEH's own SE)
toy ratio  imp/host  =  0.0247 / 0.0062 =  3.98
instrument ratio     =  0.000000 / 0.004000 =  0.00
```

The registered refutation clause — *"a split in the opposite direction … refutes M5′"* — is
satisfied on its face: the entire (small) split-dose signal sits in the cell the prediction assigned
the minor role, and none of it in the cell the prediction assigned the major role.

### 4.3 The split is strongly non-additive — quantified

```
MEH + MEI       =  +0.004000 + 0.000000     =  +0.004000
SE(sum)         =  sqrt(0.000535^2 + 0.000000^2)  =  0.000535

vs the registered null MN0 (N = 15, same seed decade, same instrument commit):
residual        =  0.034667 - 0.004000      =  +0.030667
SE(residual)    =  sqrt(0.001579^2 + 0.000535^2)  =  sqrt(2.4921e-6 + 2.8571e-7)
                =  0.00166667
                ->  +0.030667 / 0.00166667  =  +18.40 sigma

vs the Amendment-A1 null MN0X (N = 100, disjoint seeds from MEH/MEI):
residual        =  0.037250 - 0.004000      =  +0.033250
SE(residual)    =  sqrt(0.000494^2 + 0.000535^2)  =  0.000728
                ->  +0.033250 / 0.000728    =  +45.67 sigma
```

**The two doses together produce 8.7× what they produce apart.** The split arms recover
0.004000 / 0.034667 = **11.5 %** of the null (10.7 % against MN0X); **88.5 % of the effect exists
only when both ingredients are dosed simultaneously.** The 2-D scan measured the same residual
independently, on fresh seeds in a different decade, as D(1,1) = **+0.033667 at 23.4σ**
(`SCAN_READOUT.md` §3.1) — quoted as corroboration, not re-adjudicated.

### 4.4 What this does to M5′ **as registered**

Stated as consequences of the registered text, with nothing added:

1. **M5′-CONFIRMED does not fire.** Its registered conjunction fails, and fails on the conjunct the
   prereg itself labelled decisive (parent §2: *"E1 is the decisive arm"*).
2. **The registered refutation clause fires**: the split is in the opposite direction, so M5′ is
   **REFUTED AS STATED**, and the registered handling *"returns the study to the M2′ arm"* — an arm
   which **was never run** (parent §2 lists A-M2′; no A-M2′ JSON exists in the run dir).
3. **The specific sub-claim that fails is M5′'s attribution of the carrier**, not its algebra. M5′
   located the displacement in *the estimator's over-broad effective measure over the smeared
   impostor candidates*, with the exact host merely pinning at weight 1/K — and derived from that
   the K-saturation account (parent §7: *"the host's exact redshift is the only thing pinning the
   estimate at σ_z = 0, and its pinning power is just 1/K … which is why the effect saturates in K
   (K = 2/5/20/100 → 0.0138/0.0252/0.0314/0.0341)"*). **At production K̄ ≈ 1,216 the instrument says
   the opposite: one exact host at weight ≈1/1216 annihilates the bias outright against ~1,192,721
   fully smeared impostors** — exactly zero, 15/15 seeds, degenerate posterior, 2,299-nat margin.
   The saturation-in-K extrapolation is falsified at the production K, in the direction of *more*
   host pinning power at large K, not less.
4. **What survives untouched:** M5′'s *validated* properties — the toy reproduces T-0 at σ_z = 0
   and reproduces the defect's dose ratio (R_dose 0.72–0.95 against the instrument's 0.83–0.89,
   recomputed here as MN0 0.8291 / MN0X 0.8912) — are not contradicted by anything in these arms.
   The toy fails **on the split**, in the K regime where its own account predicted it might.
5. **What this does NOT establish.** It does not name a replacement mechanism, it does not identify
   an estimator term, and it does not license M2′ (whose arm was not run). Parent §4's NO-OWNER
   handling — *"the register is exhausted, not the question"* — is quoted here only because it is
   the registered posture toward an unnamed mechanism; **which branch actually fires is §5's
   business and is not decided by this section.**

---

## 5. THE BRANCH THAT FIRES — checked in the registered order, zero judgement calls

Parent §4 branches, in the registered order:

| order | branch | registered condition | fires? |
|---|---|---|---|
| 1 | **STUDY-CONFOUNDED** | arm N-0 fails to reproduce within ±0.002, **or** any §5 validity check fails | **NO** — both legs clear (§1.9): A1-PASS at \|Δ\| = 0.000013 (leg 1); V-M1…V-M5 all passing, abort (a)–(d) none fired (leg 2). Conditional on the author ratifying A1-PASS and the 2026-08-14 V-M5 golden. |
| 2 | **SINGLE-OWNER** | **exactly one arm is TERM-OWNS** | **YES** — **MEI (E1-imp) is TERM-OWNS in both channels; MN0 is TERM-INNOCENT; MEH is OTHER. Count = 1.** |
| 3 | MULTI-TERM | two or more arms TERM-OWNS, or A-ALL owns while no single arm does | not reached (and would be NO: count = 1; no A-ALL arm was run) |
| 4 | NO-OWNER | every arm is TERM-INNOCENT or TERM-PARTIAL | not reached (and would be NO: MEI is TERM-OWNS, MEH is OTHER — neither class) |

> ### BRANCH FIRED BY THE TREE: **branch 2 — SINGLE-OWNER**
>
> - **Sole TERM-OWNS arm: MEI (E1-imp)** — |b| = 0.000000 ≤ 0.010 **and** HPD90 = 1.000 ≥ 0.60,
>   **identically in 1D and 2D**. No other arm meets either conjunction.
> - **The 2D channel returns the identical classification in every arm**, so parent §6's
>   "a 1D/2D split forces MULTI-TERM" clause has no subject.
> - **Disclosures required alongside the verdict line:** STUDY-CONFOUNDED — *not* triggered, but
>   **conditional on two presented-not-ruled verdicts** (A1-PASS, V-M5 golden); DS-M5 —
>   **M5′ NOT CONFIRMED and refuted as stated**; abort (d) — *not* triggered on the literal wording,
>   with a 100 % magnitude disagreement between the L0 toy and the MEI arm (D-M-7); DS-M3 and DS-M4
>   — **NOT EVALUABLE**, no arm was run for either.

> ### **THE BRANCH FIRES ON AN ARM THAT NAMES NO ESTIMATOR TERM. THAT IS THE FINDING.**
>
> Branch 2's **condition** is a count over TERM-OWNS arms, and the data satisfies it exactly once.
> Branch 2's pre-stated **meaning** is: *"That term is the identified mechanism; the
> `/physics-change` package is written against it, with this study's arm as its regression test."*
>
> **The arm that fires it alters no estimator term.** Parent §2 is explicit: *"**E1 is the decisive
> arm and it requires ZERO estimator change.** The per-candidate σ vector already exists
> (`venue_transfer.py:1139`); only the *generator-side* assignment at `venue_transfer.py:1393, 1396`
> varies between the two cells … the estimator is byte-identical across N-0, E1-host and E1-imp."*
> `ARMS.md` says the same: *"No estimator code. `_channel_terms_at_h`,
> `log_channel_posteriors_ball_sigma_vector` and `_g_ball_capped` are byte-identical across all
> three arms."*
>
> So what MEI removes, relative to N-0, is **the host's redshift uncertainty — an input condition of
> the mock universe, not a term of the estimator.** Read literally, branch 2 identifies "the
> mechanism" as *host redshift uncertainty*, which is the venue's premise rather than a defect: the
> whole point of the dark-siren venue is that host redshifts are uncertain. **There is no term to
> write a `/physics-change` package against, and no formula in the estimator that MEI ablates.**
>
> The mechanical facts, stated so the author can rule:
>
> | | |
> |---|---|
> | Branch 2 condition as written (exactly one TERM-OWNS arm) | **SATISFIED** — MEI, both channels |
> | Branch 2 meaning clause (*"that term is the identified mechanism"*) | **HAS NO REFERENT** — the arm alters a generator-side dose, by registered design (§2, `ARMS.md`) |
> | Branch 4 NO-OWNER's premise (*"the mechanism is not in the register"*) | **factually consistent** with §4: M5′ refuted, M1/M2/M3/M4/M5 already closed in §7, A-M2′ never run — but branch 4's **condition** (every arm TERM-INNOCENT or TERM-PARTIAL) is **NOT satisfied**, and branch 4 is checked last |
> | DS-M5's own instruction on refutation | *"returns the study to the M2′ arm"* — **an arm that was never run** |
>
> **This readout does not choose between them.** It reports that the registered ordering fires
> branch 2 on a count, that branch 2's consequence clause cannot be executed as written because no
> estimator term was ablated, and that the study's registered fallback instruction (DS-M5 → M2′) and
> its registered non-forcing branch (NO-OWNER) both point at an unrun arm and an unexhausted
> question. **Pending the author's ruling this readout behaves as if the most restrictive reading
> governs: it names no mechanism, adopts no candidate, and proposes no repair.**
>
> **NOT ADJUDICATED HERE.** Parent §4: branches are *presented to the author, never
> self-adjudicated*. Only a session the author authorises may append a verdict block below the
> parent's verdict line.

---

## 6. INTEGRATION WITH THE 2-D DOSE SCAN

`SCAN_READOUT.md` has its own filed verdict (branch 2, INTERACTION-BILINEAR, with its meaning clause
contradicted and the surface characterised as **gate × amplifier**). **It is not re-adjudicated
here.** What follows is only what the split-dose arms and the scan establish *jointly*, and what
they *jointly rule out*.

### 6.1 What they jointly ESTABLISH

1. **Host exactness is a hard gate, replicated across two seed decades and two preregistrations.**
   The arms: MEI = +0.000000, 15/15 seeds on the true grid point, 2,299-nat margin, at full impostor
   dose. The scan: the entire f_host = 0 row (S00–S03, 60/60 seeds, fresh seeds in the +51000 block)
   exactly +0.000000 at **every** impostor dose from 0 to 1× GLADE. **The zero is not a coincidence
   of one impostor dose, and not a low-N artefact.** The scan's cross-check S03 ↔ MEI is an exact
   replication (`SCAN_READOUT.md` §1.8).
2. **The impostor sea alone is neither necessary nor sufficient.** MEH (+0.004000) and the scan's
   f_imp = 0 column (+0.004667 / +0.005333 / +0.006000; S30 ↔ MEH replicates at 2.64σ, inside
   tolerance) agree that removing the sea leaves **~11–15 % of the null**, small and positive.
3. **The two doses are strongly non-additive**, at +18.4σ against MN0 and +45.7σ against MN0X here
   (§4.3), and at +23.4σ on the scan's own independent seeds. Additivity is dead three times over.
4. **The full-dose null is the campaign object.** MN0X reproduces T-c(0.730) in both channels on the
   mean *and* on the whole signature — coverage 0.000/0.000/0.000, PIT–KS 1.000, bias/post_sd 8.49
   vs the campaign's 8.51, rails 0.000 (recomputed here in §2, §3.2).
5. **No 1D/2D split anywhere** — not in any arm, not in any of the scan's 16 cells.

### 6.2 What they jointly RULE OUT

1. **M5′ as registered** (§4). Its decisive split prediction is inverted; its K-saturation account
   is falsified at production K̄ ≈ 1,216 by the arms and, along the entire impostor axis, by the
   scan.
2. **Every mechanism that lives in the impostor/candidate measure alone.** Any term acting only on
   impostor kernels — however over-broad, however mis-normalised — must produce *some* displacement
   when impostors are dosed and hosts are exact. The measurement is **exactly zero**, with a
   degenerate posterior, at every impostor dose tested. **This class is closed, not attenuated.**
3. **Additive (separable) accounts** — "a host term plus an impostor term". Refuted at ≥18σ here and
   ≥23σ in the scan.
4. **Reweighting-class repairs.** The L0 closure (parent §7, W1: rate-shaped +2 %, oracle +1 %,
   both inside DS-M4's ±0.004 edge — §3.4) and the structural argument now have a decisive
   companion fact: **a change of h-independent candidate weights cannot turn a nonzero bias into an
   exact zero with a degenerate posterior**, which is what host exactness does.
5. **Symmetric-smoothing / "we convolved wrong" accounts** were already excluded by the parent §7
   parity argument (Gaussian convolution is even in σ ⇒ O(σ²) ⇒ R_dose ∝ σ_z, predicted 3.5×
   across the B1→B2 lever against a measured 0.92). Nothing in the arms rehabilitates them; the
   scan's independent dose-response statistic points the same way under its own verdict.
6. **Any account in which the effect is a property of the estimator's treatment of *many*
   candidates** — the effect requires the *host's own* smearing, and the sea only amplifies it.

### 6.3 What they jointly do **NOT** establish — stated so it is not read in

- **They name no estimator term.** Both E1 arms and all 16 scan cells vary the **generator-side
  dose**; not one of them ablates an estimator formula. The parent's own title question — *which
  term of the estimator produces the +1×σ_z displacement?* — **is not answered by a term name.**
  The answer these two documents return is an *input condition* (host exactness) and a *shape*
  (gate × amplifier).
- **M2′ is untested.** The one open candidate on the register after §7 (missing measure/Jacobian
  inside the z-integral, `venue_transfer.py:1138-1141`) has no arm in either document.
- **The parent §7 α-deletion constraint is unchecked here** — any candidate that OWNS the defect
  must still produce ≈ +0.0165 at σ_z = 0.035 with α removed. No arm in this study tests it.
- **DS-M3's linearity read is absent** (§3.3): no flat-dose arm was run under this prereg.
- **The pp_coverage sign flip** remains unexplained by anything in this study (it carries no branch
  weight; the scan's §5.4 sub-prediction is its own).
- **Nothing about K-scaling, f_incl < 1, completeness, sky-cone geometry, or transfer to production
  `BayesianStatistics`.** K is pinned at `real_k` (ΣK = 1,193,703) in every seed of every arm.

---

## 7. DISCLOSURE LIST

| # | disclosure |
|---|---|
| **D-M-1** | **Registered N mismatch inside the parent.** §3 states DS-M1's SE and DS-M2's binomial bands "at N = 25"; §2 registers E1 as *"2 cells × 15 seeds"* and `ARMS.md` fixes N = 15 for all three arms. The **edges themselves** (0.010 / 0.030 / 0.004 / 0.60 and the three coverage bands) are absolute numbers and were applied **unchanged**; only the SE/band-width commentary attached to them assumed N = 25. No classification in §3 changes under either N, because every arm is far from every edge except where noted. |
| **D-M-2** | **MEI's TERM-OWNS turns on a degenerate coverage number.** HPD90 = 1.000 because the posterior is a numerical delta on the true grid point (max single-point mass 1.000, 2,299-nat margin, PIT exactly 0.5 in 15/15). Containment is trivially satisfied; it is not evidence of calibration. DS-M2 at the 50 % and 68 % levels reads 1.000 against bands topping out at 0.700 and 0.867 — **over**-coverage — which the TERM-OWNS conjunction does not test (it reads HPD90 only). The rule as registered is satisfied as written; this readout neither relaxes nor tightens it. |
| **D-M-3** | **The branch-2 arm alters no estimator term** (§5). Registered by design: parent §2 and `ARMS.md` both state the estimator is byte-identical across N-0, E1-host and E1-imp, and that only the generator-side dose assignment varies. Branch 2's meaning clause presumes an identified *term*; there is none to write a `/physics-change` package against. Surfaced as a defect of the registered tree's applicability to a generator-side arm, not of the data. |
| **D-M-4** | **D-A1-2 is CLOSED, by an artifact this readout did not produce.** `VM5_GOLDEN_20260814.md` / `.json` (producer `verify_vm5_golden.py`, committed `38465df8`) re-executes the registered V-M5 values-golden condition at the current commit: **PASS**, max relative deviation **1.6135e-14** vs rtol 1e-12, all four MAP fields exactly equal, 1D channel bit-identical. This readout consumes it as an input and reports it as a **presented, unruled** verdict like every other in this thread (§1.6). Had it not existed, V-M5 would be recorded here as **OPEN — not re-executed at `3aedbe55`**, which is *not* a failure and would not by itself fire branch 1. |
| **D-M-5** | **D-A1-3 carried forward: the arms sit at two instrument commits.** MN0, MEH, MEI at `e83ed0b9`; MN0X at `3aedbe55` (which refactored the dose application from a boolean mask to continuous `dose_scales`). **Every DS-M1 and DS-M5 comparison in §3–§4 is within-commit** (MN0 is the registered comparator and shares `e83ed0b9` with MEH and MEI). The cross-commit comparisons — MN0X used as an alternative null in §4.3 and §3.1 — are labelled as such. A1-DET showed the refactor bit-inert on the `"all"` path across the two commits (0.0 deviation, 15 seeds × 44 fields); the `"host"` and `"impostors"` paths are covered by the AR unit tests at HEAD but have **no stored-record cross-commit check**. |
| **D-M-6** | **V-M2 / AR-3 is not checkable in the arm data.** The three arms were registered on disjoint seed blocks, so there is no fixed seed on which to compare pre-dose `z_obs`, the σ vector and the scatter vector across arms. AR-3 is discharged by unit test (11/11 pass at HEAD, re-run 2026-08-14) and by its observable consequences (identical `K_sum`, `n_events`, `f_incl`, and the §1.3 dosing table). Recorded rather than asserted as an in-data check. |
| **D-M-7** | **Abort criterion (d) — the closest call in this readout.** The L0 toy predicted +0.0247 for the impostor-only arm; the instrument returned **exactly 0.000000** — a 100 % magnitude disagreement. Criterion (d) as registered fires on a **sign** disagreement, and zero has no sign, so on the literal wording it does not fire. **This readout applies the literal wording and flags the alternative reading for the author**, because firing (d) would STOP the study and force every L0 closure (M3, M4, M5→M5′, W1) to be revisited. Relevant context, offered without resolving it: the toy ran at K = 50 and *itself* predicted K-saturation; the instrument runs at K̄ ≈ 1,216. |
| **D-M-8** | **The MN0 2D channel also misses the ±0.002 window against its own reference.** The parent's operational record notes *"The 2D channel lands at +0.037000, on the campaign value"* — comparing MN0's **2D** value against the **1D** reference 0.037237. The campaign's 2D reference is **+0.039713 ± 0.000246** (`VENUE_TRANSFER_READOUT.md` line 203), so \|Δ\|₂D = 0.002713, **also outside ±0.002**. **No verdict depends on this**: V-M1 and A1 are registered on the 1D channel, and MN0X's 2D lands at +0.039750, \|Δ\| = 0.000037, inside. Reported because the "2D was fine" framing would otherwise carry into the parent's branch discussion unchallenged. |
| **D-M-9** | **MEH is "in band" only on the statistic DS-M1 reads.** Its +0.004000 is 21.4× its own `post_sd` median (0.000187) and its coverage is 0.200/0.200/0.333, below all three registered bands. DS-M1's in-band edge reads \|b\| and not the posterior width, so the arm classifies OTHER on the coverage conjunct, not on the bias. Stated so that "E1-host is nearly unbiased" is not read out of §3.1. |
| **D-M-10** | **DS-M3 and DS-M4 have no data** (§3.3, §3.4). Arms E3, A-M2′ and A-M5b were never run; A-M5b's seeds +46000…+46399 and O2's +47000…+47400 remain reserved and unconsumed. The registered budget (§1: L1 ≤ 5 arms, L2 ≤ 1) was **not exceeded**: three L1 arms were run, and MN0X is a re-run of the null at higher N under Amendment A1 §4.5, consuming no arm slot. No L2 arm was run and none is requested. |
| **D-M-11** | **Recomputation is exact and the aggregates were never used** (§1.10): 0.0 relative deviation on every derived field of every seed of all four arms, both channels, against verbatim ports of the instrument's own `posterior_readout` / `pp_readout` / `hpd_contains`. Where this readout quotes a companion document's number (A1's, the scan's, V-M5's), it is labelled as quoted. |
| **D-M-12** | **This readout created two files and touched nothing else**: `MECHANISM_ISOLATION_READOUT.md` and `score_mechanism_isolation.py` (+ its `MECHANISM_ISOLATION_READOUT.json` twin). Every registered `.md` under `results/mechanism_study_20260813/` is byte-unmodified in `git status`. No ledger row, no prereg verdict block, no book page, no paper file was touched. |

---

## 8. NOT-EVALUABLE REGISTRY — rows carried forward

| # | item | status after this read |
|---|---|---|
| 1 | **DS-M3 dose-scaling** (flat 0.011 / 0.035, residual R_dose < 0.25) | **NOT EVALUABLE — carried.** No flat-dose arm; E3 never run. |
| 2 | **DS-M4 / arm A-M5b** | **NOT EVALUABLE — carried.** Arm withdrawn at registration; seeds unconsumed. The L0 answer (WEIGHTS-INERT) stands as a toy result with no branch weight. |
| 3 | **Arm A-M2′** — the one open candidate on the register | **NOT EVALUABLE — carried, and now the register's only unrun item.** DS-M5's refutation clause routes to it; no arm exists. |
| 4 | **Parent §7 α-deletion constraint** (≈ +0.0165 at σ_z = 0.035 with α deleted) | **NOT EVALUABLE — carried.** No arm in this study tests it. |
| 5 | **K-dependence** | **NOT EVALUABLE — carried.** K pinned at `real_k`, ΣK = 1,193,703 on every seed of every arm. The K = 50 → K̄ = 1,216 contrast in §4.4 is toy-vs-instrument, not a K ladder. |
| 6 | **Transfer to production `BayesianStatistics`** | **NOT EVALUABLE — carried.** `venue_transfer.py` is a certified mirror; any production change routes `/physics-change`. |
| 7 | **The pp_coverage sign flip** | **NOT EVALUABLE — carried.** No branch weight; nothing in these arms addresses it. |
| 8 | **Any repair** | **NOT EVALUABLE — carried. No repair is proposed anywhere in this readout.** |

---

## 9. FORMULATION AWAITING THE AUTHOR'S RULING

**Nothing below is adopted. These are the decisions this readout hands up.**

1. **The branch call.** Checked in the registered order, the tree fires **branch 2 — SINGLE-OWNER**,
   with **MEI (E1-imp) the sole TERM-OWNS arm in both channels**. Its consequence clause — *"that
   term is the identified mechanism; the `/physics-change` package is written against it"* —
   **cannot be executed as written, because the arm ablates no estimator term** (parent §2 and
   `ARMS.md` both register E1 as a zero-estimator-change, generator-side arm). The author's ruling
   is required on whether branch 2 is read as having fired with an inexecutable consequence, or
   whether a generator-side arm was ever eligible to be scored by DS-M1 at all. Pending that ruling
   this readout names no mechanism and proposes no repair.

2. **Whether the branch call may stand on two unruled verdicts.** Branch 1's non-firing rests on
   (i) **A1-PASS** and (ii) the **2026-08-14 V-M5 golden**, both of which are *presented, not
   adjudicated*. If the author declines A1-PASS, branch 1 fires on the registered N = 15 arm and
   everything in §3–§6 is void. This is stated first because it is the only dependency that can
   overturn the whole readout.

3. **DS-M5 and where the study goes.** DS-M5 returns **M5′ NOT CONFIRMED / REFUTED AS STATED** —
   the impostor conjunct fails by its entire 0.030, the split is inverted, and the residual
   non-additivity is +18.4σ (against MN0) / +45.7σ (against MN0X). The registered handling is
   *"returns the study to the M2′ arm"*, **which was never run and is now the register's only open
   candidate**. Whether to run A-M2′, or to treat the register as exhausted and open a fresh
   stage-0 intake with the mandatory Stage-L literature sweep (parent §4 branch 4's pre-stated
   handling), is the author's call. **This readout requests neither.**

4. **Abort criterion (d) and the standing of the L0 closures.** The literal wording does not fire on
   MEI (D-M-7). If the author reads a 100 %-magnitude disagreement as within (d)'s intent, the
   registered consequence is STOP-and-report with **every** L0 closure revisited — M3, M4, M5→M5′
   and W1 all rest on the same toy. The relevant fact for that decision: the toy validated against
   T-0 and reproduced the dose ratio, and failed only on the split, at a K 24× below production.

5. **The 2-D scan's parallel item.** The scan (own verdict, not re-adjudicated) fired **branch 2 —
   INTERACTION-BILINEAR** whose meaning clause its own statistics contradict. **Two registered trees
   in this thread have now fired a branch whose condition is met and whose meaning clause does not
   describe the data.** Whether that is a coincidence of two independently drafted trees or a
   systematic feature of how the branches were written is a question about the thread's method, and
   it is the author's to answer.

6. **What this readout does NOT ask for.** No repair. No `/physics-change` intake. No new arm, no
   L2 confirmation arm, no higher N on MEH or MEI (the MEI result is degenerate, not
   noise-limited — §2.3), no re-scoring of any band, no ledger row. The registered budget was not
   exceeded and no band was moved.

---

*This readout is a mechanical scoring of the parent pre-registration plus the interpretation §4
assigns to the readout session. It appends nothing to any pre-registration, adjudicates nothing,
proposes no repair, and edits no registered document.*

---

## Addendum (2026-08-14) — rulings of record after the independent review

Ledger row #102 records the author's 2026-08-14 ruling following the adopted commission review (`results/commission_research_20260814/REPORT.md`). Binding on this readout: **(i)** branch 2 is recorded as **PREMATURE ADJUDICATION** (count-based branch adjudicated while registered arm A-M2′ was unrun) — no term is named and both previously tabled readings are superseded; **(ii)** abort (d) is deemed **met in substance** (toy unfaithful at production K: re-executed prediction +0.0341 impostor-only at K=1216 vs measured exactly 0.000000) — toy-dependent M5/W1 sub-closures are NOT ESTABLISHED, M1/M4 stand toy-independent, M3 stands on its analytic core; **(iii)** the V-M1 branch-1 disjunct is discharged to MN0X by registered amendment, MN0's N=15 FAILED status unchanged; **(iv)** MEI's TERM-OWNS classification carries **zero term-attribution power** (degenerate single-grid-point posterior; any register candidate would produce it). This addendum edits nothing above; it is the registered landing of ledger row #102.
