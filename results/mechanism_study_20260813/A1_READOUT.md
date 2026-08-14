# A1 READ — arm MN0X, V-M1 null at N = 100 — mechanical readout

**Scored against** `results/mechanism_study_20260813/AMENDMENT_A1_VM1_NULL_AT_N100.md`
(registered `73141160`, author-ratified 2026-08-13), which amends
`PREREGISTRATION_MECHANISM_ISOLATION.md`. **Instrument commit:** `3aedbe55`
(`darksiren_emri/validation/venue_transfer.py`). **Data commit:** `5b0bd17a`.
**Scorer:** `results/mechanism_study_20260813/score_a1.py` → machine-readable twin
`A1_READOUT.json`. Every number below is recomputed from the 100 raw per-seed records in
`MN0X_h0p730_results_seeds0_100.json`; **the `aggregate` block of that file is not trusted and was
not used** (it is compared against, in §5.7).

**Arm:** MN0X · `sigma_mode=glade` · `dose_target="all"` · `h_true = 0.730` · N = 100 ·
seeds 20310808–20310907 (base 20260808 + 50000…50099) · 982 pinned events/seed · both channels.

> **This readout does not rule.** It reports what the registered A1 decision rule yields and hands
> it to the author. No registered file was edited: `AMENDMENT_A1_VM1_NULL_AT_N100.md`,
> `PREREGISTRATION_MECHANISM_ISOLATION.md`, `PREREGISTRATION_2D_DOSE_SCAN.md` and `ARMS.md` are
> byte-unmodified in `git status`. No ledger, book or prereg verdict block was touched. Nothing
> here is committed.

---

## 1. VALIDITY FIRST

### 1.1 Seed-block integrity (amendment §4.2, parent VT-D7 discipline)

**PASS, exact.** The realized seed set **equals** the registered block, with zero gaps, zero
extras, zero duplicates.

| check | registered | realized | ✓ |
|---|---|---|---|
| N | 100 | **100** per-seed records, **100** unique seeds | ✓ |
| block | base + 50000…50099 = 20310808–20310907 | min 20310808, max 20310907, set equality **exact** | ✓ |
| top-level `seeds` field | same block | identical to the per-seed set | ✓ |
| MN0 ⊂ MN0X | superset required (§4.3) | MN0's 15 seeds are all present | ✓ |
| disjoint from MEH | MEH starts at +50100 | intersection **∅**; MEH's first seed is **20310908**, exactly one above MN0X's last | ✓ |
| disjoint from MEI | MEI at +50200…50214 | intersection **∅** | ✓ |

The tight abutment the amendment asked to have *checked rather than assumed* holds: 20310907 / 20310908.

### 1.2 V-M3 — pin integrity

**PASS.** `pin_integrity.pass = true` in the arm JSON, and every pinned field re-read here:

- CRB CSV md5 **`9a1f2a14384a9281c97ca3be312ddaab`** ✓ · frozeng emit md5
  **`34c50e91028b6a6458a2b145db545705`** ✓
- K census 1588 / 606 zeros / 74 ones / median 6 / max 245,364 / **ΣK = 1,193,703** / nonempty 982 ✓
- pruned-frame σ_z stats n = 20,834,171 · median 0.0393412950539589 · min 0.0005317263419419 ·
  n<5e-3 = 231,098 ✓

### 1.3 Per-seed pins registered in §6, verified on **all 100** seeds

| pin | registered | realized |
|---|---|---|
| `K_sum` | 1,193,703 | **{1193703}** — one distinct value across 100 seeds |
| `n_events` / `n_events_run` | 982 / 982 | **{982} / {982}** |
| `f_incl` | 1.0 | **{1.0}** |
| rails, 1D and 2D | zero | R_low = R_high = **0.000** in both channels |
| non-finite `ln_post` | zero | **0 seeds** with any non-finite entry in either 41-point vector (200 vectors scanned) |
| `sigma_z_mean_pairs` | ≈ 0.0418 (MN0 measured 0.041813) | mean **0.041796**, range [0.041459, 0.042005] — full dosing |
| horizon drop | ≤ 5 % | `n_horizon_dropped` max = **0** |
| edge guard (parent §8) | — | edge-loaded fraction **0.000** in both channels |

### 1.4 Abort criteria (parent §5 a–d)

| criterion | trigger | measured | fired? |
|---|---|---|---|
| (a) non-finite `ln_post` | > 1 % of seeds | **0.0 %** | **NO** |
| (b) horizon drop | > 5 % | **0** dropped on every seed | **NO** |
| (c) any V-M failure | any | none observed (§1.2, §1.5, §1.6) | **NO** |

### 1.5 V-M4 — clean rule

**PASS.** `import_path_clean = true`, `allow_dirty = false`, `smoke = false`,
`dirt_inventory.import_path = []`. `git_dirty = true` is expected and permitted (untracked
`results/` staging dirs and `logs/`).

### 1.6 A1-DET — the registered determinism check (§4.3)

Registered form: *re-run **two** of the 15 stored seeds; require per-seed records to match to
rtol ≤ 1e-12 with MAP values exactly equal. Failure ⇒ do not reuse; run all 100 fresh.*

**Executed on all 15 shared seeds, not two.** This is possible because the arm did **not** reuse
the stored records: job **6304141** ran `cell=MN0X … seeds=0:100`, i.e. all 100 seeds fresh
(log `logs/mech-null-n100_6304141.out`, wall 03:24:56, 15 workers, host uc2n872). The 15 shared
seeds are therefore genuine re-runs and A1-DET is a real determinism test.

| A1-DET evidence | value |
|---|---|
| shared seeds compared | **15 / 15** (20310808–20310822) — registered minimum was 2 |
| value fields compared per seed | **44**, including the full 41-point `ln_post_1d` and `ln_post_2d` vectors, `map_*`, `map_*_refined`, `mean_*`, `post_sd_*`, `pit_*`, `hpd50/68/90_*`, `railed_low/high_*`, `edge_mass_*`, `K_sum`, `sigma_z_*`, `M_source_median`, `texture_corr` |
| **max relative deviation, over all 15 seeds × 44 fields** | **0.0 — bit-identical**, against the registered tolerance 1e-12 |
| **MAPs exactly equal** | **YES**, both channels, 15/15 seeds |
| fields excluded | **`cell` only** — the arm *label* ("MN0X" vs "MN0"), an identity string, not a value. Reported here rather than silently dropped. |

> ### **A1-DET: PASS.** (Not a marginal pass: the deviation is exactly zero, twelve orders of
> magnitude inside the registered tolerance, on 7.5× the registered number of seeds.)

**Why this is stronger than the registered check.** MN0 ran at commit `e83ed0b9`; MN0X ran at
`3aedbe55` (a descendant), and `3aedbe55` **does** modify `venue_transfer.py` (+128 lines): the
dose application was refactored from a boolean mask to continuous `dose_scales`
(`scale = np.where(host_mask, s_host, s_imp)`), for the 2D dose-scan companion. On the
`dose_target="all"` path this reduces to multiplying σ by 1.0, and A1-DET now shows **empirically,
on 15 seeds and 44 fields, that the refactor is bit-inert on this arm's path** — which is the
`ARMS.md` AR-1 claim, cross-commit. The 11 tests in
`darksiren_emri_test/validation/test_venue_transfer_arms.py` (AR-1/AR-2/AR-3) also pass at HEAD.
The import-path diff `e83ed0b9..3aedbe55` touches only `venue_transfer.py` and its test file.

**Consequence:** the reuse clause of §4.3 is satisfied and, in fact, moot — no record was reused;
the 15 already-run seeds were *re-computed* and are in the N = 100 mean as themselves. The
selection hazard §4.3 was written to prevent (dropping the inconvenient 15) did not arise: they
are included, and they are §3.2's largest single drag on the mean.

---

## 2. THE SCORED RULE — A1-PASS / A1-FAIL (§4.4)

**Registered rule, quoted:** `| mean_bias_1D(MN0X, N=100) − 0.037237 | <= 0.002` — the original
window, unchanged, on the original 1D channel, against the same reference.

**Recomputed measurement, 1D channel, grid-argmax MAP bias (the registered statistic):**

```
mean_bias_1D  =  mean(map_1d) − 0.730  =  0.76725 − 0.730  =  +0.037250
sd (ddof=1)   =  0.00494286      SE = sd/sqrt(100) = 0.000494
sd (ddof=0)   =  0.00491808      SE = 0.000492
```

```
| 0.037250 − 0.037237 |  =  0.000013
margin to the window     =  0.002000 − 0.000013  =  0.001987
```

| quantity | value |
|---|---|
| MN0X 1D mean bias (N = 100) | **+0.037250 ± 0.000494** |
| campaign reference, T-c(0.730) N = 400 1D | +0.037237 ± 0.000230 |
| **\|Δ\|** | **0.000013** |
| registered window | ±0.002 (unchanged) |
| **margin** | **0.001987** |
| **how far inside** | **153.8× inside the window** (0.002 / 0.000013) |
| Δ in σ of the difference (SE_diff = √(0.000494² + 0.000230²) = 0.000545) | **0.024 σ** |
| the window, in σ, as bought | **4.07 σ** (arm SE) / **3.68 σ** (difference SE) — the amendment §5 anticipated 3.27 / 3.06; the realized per-seed sd 0.00494 came in below the 0.00612 assumed from N = 15 |

> ### **VERDICT YIELDED BY THE REGISTERED RULE: A1-PASS.**
> **V-M1 is satisfied on the 1D channel at N = 100 against the original, unwidened ±0.002 window.**
> The §4.4 **A1-FAIL** clause — the pre-committed reading that "the null arm genuinely does not
> reproduce the campaign decision cell" — **does not fire.**

**The measurement is as close to the reference as the statistic can be.** §3(iii)'s quantisation
argument, re-applied at N = 100: a per-seed MAP bias is an integer multiple of the 0.005 grid
spacing, so an N = 100 mean is an integer multiple of 5e-5. The reference +0.037237 sits at
**744.74 ticks** and is not attainable; the measurement is **745 ticks exactly** — the nearest
attainable value. |Δ| = 0.000013 is **below the 2.5e-5 maximum quantisation offset**, i.e. the arm
lands on the closest grid-mean to the reference that exists. No smaller |Δ| is possible at N = 100.

**2D channel, reported alongside under the same rule (parent §6 convention):**

| channel | mean bias (N = 100) | SE | vs. its own campaign reference | \|Δ\| | inside ±0.002? |
|---|---|---|---|---|---|
| **1D (registered)** | **+0.037250** | 0.000494 | +0.037237 (N = 400) | **0.000013** | **YES** |
| 2D (secondary) | +0.039750 | 0.000519 | +0.039713 (N = 400) | **0.000037** | **YES** |

**No 1D/2D split.** Both channels reproduce their own campaign value to inside 4e-5, so the parent
§6 "a 1D/2D split is itself a finding" clause has no subject here. (The N = 15 arm's apparent
1D/2D split — 1D outside, 2D inside — is resolved: it was the 1D channel's noise, not a channel
asymmetry.)

**Companion statistics, reported un-banded (no A1 band attaches to any of these):**

| statistic | MN0X 1D / 2D | campaign T-c(0.730) 1D / 2D |
|---|---|---|
| `post_sd_median` | **0.004386 / 0.004407** | 0.004376 / 0.004410 |
| bias / `post_sd_median` | **8.49 / 9.02** | 8.51 / 9.00 |
| HPD 50/68/90 coverage | 0.000 / 0.000 / 0.000 (both channels) | 0.000 / 0.000 / 0.000 |
| PIT–KS D (recomputed from the 100 PIT values) | 1.0000 / 1.0000 (max PIT 1.9e-11 / 6.0e-13) | 1.000 / 1.000 |
| rails R_low / R_high | 0.000 / 0.000 (both channels) | 0.000 / 0.000 |

The null arm reproduces the campaign's **entire** decision-cell signature, not merely its mean: the
coverage collapse, the saturated PIT, the ~8.5–9σ delta-narrowness and the zero railing all land on
the campaign values. This is context, not a scored band.

---

## 3. THE PRE-REGISTERED PREDICTION (§5) — scored

The amendment registered a point prediction **before** the arm ran, derived from the 15 known seeds
plus 85 seeds at the reference:

> **REGISTERED PREDICTION: MN0X 1D mean bias = +0.03685 ± 0.00056 (1σ), 68 % interval
> [0.03629, 0.03741].**

| | value |
|---|---|
| predicted | +0.036850 ± 0.000564 |
| **measured** | **+0.037250** |
| residual | **+0.000400** |
| **in σ of the registered prediction** | **+0.71 σ** |
| inside the registered 68 % interval [0.03629, 0.03741]? | **YES** |

**The pre-registered prediction is met, at 0.71σ and inside its own 68 % interval.** This was a
falsifiable forecast written before the data existed and it is not a tautology: the same §5
registered the exact excursion that would have broken it (below).

### 3.1 The registered falsification arithmetic, checked against what happened

§5 registered that A1-FAIL required the **85 fresh seeds** to mean ≤ 0.0353376, a −2.86σ excursion.

| | value |
|---|---|
| fresh-85 mean bias, 1D (recomputed; the 85 seeds not in MN0) | **+0.037706 ± 0.000499** |
| vs. the reference +0.037237 | **+0.000469** = **+0.71 σ** (registered fresh SE 0.000663) |
| vs. the registered fail threshold 0.0353376 | **+0.002368 above it = +3.57 σ** |

The fresh seeds, which are independent of everything that produced the N = 15 reading, land **above**
the reference by 0.71σ. The downward offset that A1-FAIL would have required is not merely absent —
it is contradicted at 3.6σ by the new data alone.

### 3.2 What the N = 15 shortfall was, measured

| | value |
|---|---|
| included 15 seeds, 1D mean | +0.034667 (SE 0.001525, ddof = 0) |
| fresh 85 seeds, 1D mean | +0.037706 (SE 0.000496) |
| difference | **−0.003039 ± 0.001604** = **−1.90 σ** |

The 15 seeds are a **−1.9σ downward fluctuation of their own block.** That is the shape §3 said the
N = 15 data could not distinguish from a real offset — and the extra 85 seeds distinguish it. The
"about eight seeds each landing one grid point low" account in §3 is the one that survives.

---

## 4. CONSEQUENCE FOR THE PARENT STUDY — stated exactly

V-M1 is parent §4 branch 1's first leg: *"STUDY-CONFOUNDED — arm N-0 fails to reproduce the
campaign bias within ±0.002, **or any validity check in §5 fails**."* At N = 15 that leg fired
mechanically (|Δ| = 0.002570). At N = 100 it does not.

### 4.1 What an A1-PASS **does** license

1. **The V-M1 leg of STUDY-CONFOUNDED is not satisfied by the null arm as measured at N = 100.**
   The study's own anchor reproduces the campaign decision cell — in both channels, on the mean and
   on the full signature (§2).
2. **The parent's other registered reads become readable** — the MN0 / MEH / MEI split-dose arms,
   the DS-M1 per-arm classification, the DS-M5 conjunction (MEI ≥ 0.030 **and** MEH ≤ 0.012), and
   the parent's *"two facts the readout session must confront"*, including the strongly
   non-additive split (MEH + MEI = +0.004 against MN0's +0.0347). Those reads were blocked by a
   failing anchor; the anchor is no longer failing.
3. **Subject to the second leg.** Branch 1 also fires on *any* §5 validity failure. §1 of this
   readout scores V-M1, V-M3, V-M4, A1-DET and the abort criteria on MN0X and finds them all
   passing. **V-M2 and V-M5 are not scored here** (see disclosures D-A1-2 and D-A1-3). The parent's
   branch call therefore remains **pending and belongs to the parent readout session** — this
   document removes one blocker, it does not make that call.

### 4.2 What an A1-PASS **does NOT** license

1. **It does NOT retroactively pass the N = 15 MN0 result.** A1 §3 is explicit and binding:
   *"THIS DERIVATION IS BEING WRITTEN AFTER THE READOUT … barred from use as grounds to pass the
   N = 15 result. Nothing below reclassifies MN0. MN0's V-M1 status is exactly what the registered
   rule says it is at N = 15: **FAILED**."* That stands. **The registered reading is not "MN0
   passed after all"; it is "the null is now measured at N = 100, at a precision (4.07σ arm SE /
   3.68σ difference SE) at which the ±0.002 window is a meaningful test, and at that precision the
   null reproduces the campaign."** The N = 15 arm's failure remains on the record as a failure of
   an under-powered check — a design fault in precision purchased, per §3 — and the N = 100 arm is
   a *new measurement*, not a re-scoring of the old one.
2. **It does not widen, narrow or otherwise touch any band.** ±0.002 was scored exactly as
   registered; §4.5's "what is NOT amended" list (0.010 / 0.030 / 0.004 / 0.60 / 0.25 / 1e-3, the
   DS-M5 conjunction 0.030 / 0.012, the branch definitions) is untouched by this readout.
3. **It does not adjudicate anything else in the parent** — not DS-M5, not the non-additive split,
   not the M5′ carrier hypothesis, not the branch call, not the §9 companion 2D dose scan. §1 of
   the amendment: *"This amendment concerns V-M1 and nothing else."*
4. **It does not settle the split-dose arms' own precision.** MEH and MEI remain at N = 15. Their
   registered edges (0.030 / 0.012) are ~10–20× wider than ±0.002 relative to their SEs, so the
   §3 diagnosis does not automatically transfer — but their N = 15 SEs are a matter for the parent
   readout, not for this document.
5. **It is not an author ruling.** Per the amendment and the house convention, the branch call is
   presented, never self-adjudicated.

---

## 5. DISCLOSURE LIST

1. **D-A1-1 — the 15 shared seeds were re-run, not reused.** §4.3 registered reuse of the stored
   records with a 2-seed determinism spot-check. The arm instead re-computed all 100 seeds in one
   job (6304141, `seeds=0:100`). This is *more* conservative than registered (no stored record
   enters the mean) and it converted A1-DET from a 2-seed spot-check into a 15-seed, 44-field,
   cross-commit determinism test, which passed bit-identically. The registered cost anchor is
   correspondingly overrun in the accounting sense — 100 seeds were paid for, not 85 — but the
   realized cost was **≈51 CPU-h** (instrument wall 12,221 s × 15 workers; sacct CPU-utilized
   2-00:29:10 = 48.5 CPU-h), i.e. **≈0.49–0.51 CPU-h/seed**, *below* the
   registered 0.97 anchor and roughly half the §7 "≈97 CPU-h if everything is re-run" ceiling.
   No band, seed, or statistic is affected.
2. **D-A1-2 — V-M5 (values golden at rtol 1e-12) was not re-executed at instrument commit
   `3aedbe55` by this readout.** V-M5 was verified in the parent's registration session against the
   then-current main. `3aedbe55` modifies `venue_transfer.py` (+128 lines, dose-scale refactor +
   2D scan cell specs) relative to MN0's `e83ed0b9`. The in-arm evidence that the change is inert
   on this path is A1-DET's exact bit-identity across the two commits (§1.6) plus the 11 passing
   AR-1/AR-2/AR-3 unit tests, which is strong but is **not** the registered V-M5 artifact. Flagged
   for the parent readout session, which carries V-M5.
3. **D-A1-3 — V-M2/AR-3 cross-arm generator invariance is not scored here.** A1-DET certifies the
   `dose_target="all"` path across `e83ed0b9 → 3aedbe55`. The `"host"` and `"impostors"` paths were
   also refactored by that commit (boolean mask → `s_host/s_imp ∈ {0,1}` scale), and MEH/MEI were
   run at `e83ed0b9`, **before** the refactor. Their comparability across commits is not an A1
   question, but any parent readout that compares MN0X against MEH/MEI should note that MN0X sits
   at a different instrument commit. The unit tests covering AR-1/AR-3 pass at HEAD.
4. **D-A1-4 — SE convention.** The arm JSON's `mc_error` uses the sample sd (ddof = 1) over
   √N: 0.000494 (1D), 0.000519 (2D). The population convention (ddof = 0) gives 0.000492 / 0.000517.
   **Every A1 verdict is identical under both** — the difference is ≪ the 0.001987 margin. The
   registered reference SE 0.000230 is carried verbatim from the venue-transfer readout.
5. **D-A1-5 — statistic definition.** The registered V-M1 statistic is the **grid-argmax** MAP bias
   (as is the +0.037237 reference). The refined-argmax companion is +0.037148 (1D mean_map_refined
   0.7671480 − 0.730) and +0.039623 (2D) — also comfortably inside the window; **no verdict changes
   under either variant.** Grid-argmax is used as primary throughout, per registration.
6. **D-A1-6 — `git_dirty = true` on the arm.** Permitted by V-M4 clause 3;
   `dirt_inventory.import_path` is empty and `import_path_clean = true`, so no dirt touches the
   import path. Consistent with every campaign chunk in this thread.
7. **D-A1-7 — the `cell` field is the sole A1-DET exclusion.** It differs by construction
   ("MN0X" vs "MN0") and is an arm identity label, not a computed value. Disclosed rather than
   dropped silently; every one of the other 44 shared fields matched exactly.
8. **D-A1-8 — the companion 2D dose scan is out of scope.** `PREREGISTRATION_2D_DOSE_SCAN.md` and
   its S00–S33 arms (including S23 at N = 100) were retrieved in the same commit `5b0bd17a` and are
   **not** read here. A1 §9's mutual hold was released by the author's ratification; the scan's
   readout is a separate document.
9. **D-A1-9 — nothing is committed and nothing registered is edited.** `git status` shows the four
   registered `.md` files unmodified; this readout, its JSON twin and `score_a1.py` are the only new
   files.

---

## 6. SUMMARY LINE

| item | result |
|---|---|
| seed block | exact, 100/100, disjoint from MEH/MEI, MN0 ⊂ MN0X |
| pins (V-M3, per-seed) | all pass on all 100 seeds |
| rails / non-finite `ln_post` / edge-loaded | 0.000 / 0 / 0.000, both channels |
| **A1-DET** | **PASS** — 15/15 shared seeds, 44 fields, max rel dev **0.0**, MAPs exactly equal |
| **A1 decision rule** | **A1-PASS** — \|+0.037250 − 0.037237\| = **0.000013** ≤ 0.002, margin **0.001987**, **153.8× inside** |
| 2D channel (secondary) | +0.039750 vs +0.039713, \|Δ\| = 0.000037 — also inside; **no 1D/2D split** |
| registered prediction +0.03685 ± 0.00056 | **met at +0.71 σ**, inside its 68 % interval |
| A1-FAIL clause (§4.4) | **does not fire** |
| parent branch call | **pending — not made here** |

> **NOT ADJUDICATED HERE.** The author's ratification is what makes this a verdict of record.
> Only the session the author authorises may append a verdict block below the amendment's
> verdict line.
