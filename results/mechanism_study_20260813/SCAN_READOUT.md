# 2D DOSE SCAN READ — mechanical readout

**Scored against** `results/mechanism_study_20260813/PREREGISTRATION_2D_DOSE_SCAN.md`
(registered 2026-08-13; instrument commit `3aedbe55`; data committed at `5b0bd17a`).
**Parent:** `PREREGISTRATION_MECHANISM_ISOLATION.md`. **Companion:** `AMENDMENT_A1_VM1_NULL_AT_N100.md`.
**Scorer:** `results/mechanism_study_20260813/score_2d_scan.py` → machine-readable twin
`score_2d_scan_output.json`.

**Data:** 16 cell JSONs (`S00`…`S33`), 325 seeds, 982 pinned events/seed, both channels, plus the
parent arms `MN0`/`MEH`/`MEI` and the Amendment-A1 arm `MN0X`.

> **Every number below was recomputed from the raw `per_seed` records.** No `aggregate` block in any
> JSON, and no upstream extraction, was trusted. The orchestrator's 1D table was reproduced cell for
> cell (all 16 agree to the printed digit); it is confirmed, not adopted.

> **This readout does not rule.** It reports which branch the registered decision tree fires and
> hands it to the author. Prereg §10: *the branch call is presented to the author, never
> self-adjudicated.* **No repair is proposed.** No ledger, book, claim, or prereg file was touched.

---

## 1. VALIDITY FIRST

### 1.1 V-D1 — pin integrity (before any measurement is quoted)

`pin_integrity.pass == true` in **all 16 cells**. CRB CSV md5 `9a1f2a14384a9281c97ca3be312ddaab` ✓;
frozeng emit md5 `34c50e91028b6a6458a2b145db545705` ✓; K census 1588 / 606 / 982 / ΣK 1,193,703 /
max 245,364 ✓; pruned-frame σ_z stats n = 20,834,171, median 0.0393412950539589,
min 0.0005317263419419, n<5e-3 231,098 ✓. **PASS.**

### 1.2 V-D3 — K_sum and event-count pin, every seed of every cell

`K_sum == 1,193,703` and `n_events == n_events_run == 982` on **all 325 seeds across all 16 cells**.
`n_horizon_dropped == 0` on every seed. **PASS.**

### 1.3 V-D4 — clean rule

`import_path_clean == true` in all 16 cells; `dirt_inventory.import_path == []`. `git_dirty` is
`true`, but every entry is under `results/` (staged campaign artifacts and the scan's own output
JSONs) — **nothing inside `darksiren_emri/` or `darksiren_emri_test/`**. `allow_dirty == false`, so
the cells started under the registered rule. Instrument commit `3aedbe55` identical in all 16.
**PASS.**

### 1.4 V-D5 — values golden

The 16 scan cells run on **fresh, disjoint seeds by design** (§3.2), so no committed per-seed record
exists to compare them against; V-D5 is evaluable only through the MN0 → MN0X reuse chain that
Amendment A1 §4.3 (A1-DET) registers. Checked here: the 15 stored `MN0` seeds against the same 15
seeds inside `MN0X`, **max relative deviation on every shared scalar field = 0.000e+00** (gate
rtol ≤ 1e-12), **MAPs exactly equal in both channels**. **PASS**, with the scope limitation
disclosed (§6, D-5).

### 1.5 V-D6 / §5.3 — dosing verification: the dose that was applied IS the dose registered

Two independent checks per cell.

**(a) Configured dose.** `config.dose_scales` read back from each JSON equals the cell's registered
`(f_host, f_imp)` in **all 16 cells** — `[0.0,0.0]`, `[0.0,0.25]`, … `[1.0,1.0]`, in the exact
§2 grid order. **PASS.**

**(b) Realized `sigma_z_mean_pairs` against §5.3's prediction**
`σ̄(f_h,f_i) = 0.041813·[f_h·8.2265e-4 + f_i·(1−8.2265e-4)]`:

| cell | f_h | f_i | predicted σ̄ | measured σ̄ | rel. err | tolerance | verdict |
|---|---|---|---|---|---|---|---|
| S00 | 0.00 | 0.00 | 0.000000 | 0.000000 | exact | — | PASS |
| S01 | 0.00 | 0.25 | 0.010445 | 0.010424 | 0.199 % | 2 % | PASS |
| S02 | 0.00 | 0.50 | 0.020889 | 0.020881 | 0.038 % | 2 % | PASS |
| S03 | 0.00 | 1.00 | 0.041779 | 0.041732 | 0.112 % | 2 % | PASS |
| S10 | 0.25 | 0.00 | 0.0000086 | 0.0000088 | 2.306 % | 10 % | PASS |
| S11 | 0.25 | 0.25 | 0.010453 | 0.010437 | 0.156 % | 2 % | PASS |
| S12 | 0.25 | 0.50 | 0.020898 | 0.020875 | 0.109 % | 2 % | PASS |
| S13 | 0.25 | 1.00 | 0.041787 | 0.041745 | 0.102 % | 2 % | PASS |
| S20 | 0.50 | 0.00 | 0.0000172 | 0.0000176 | 2.248 % | 10 % | PASS |
| S21 | 0.50 | 0.25 | 0.010462 | 0.010463 | 0.014 % | 2 % | PASS |
| S22 | 0.50 | 0.50 | 0.020907 | 0.020899 | 0.036 % | 2 % | PASS |
| S23 | 0.50 | 1.00 | 0.041796 | 0.041760 | 0.085 % | 2 % | PASS |
| S30 | 1.00 | 0.00 | 0.0000344 | 0.0000352 | 2.159 % | 10 % | PASS |
| S31 | 1.00 | 0.25 | 0.010479 | 0.010456 | 0.216 % | 2 % | PASS |
| S32 | 1.00 | 0.50 | 0.020924 | 0.020915 | 0.043 % | 2 % | PASS |
| S33 | 1.00 | 1.00 | 0.041813 | 0.041795 | 0.043 % | 2 % | PASS |

**16/16 in tolerance.** The `f_i = 0` column sits at 2.2 %, inside its registered 10 % band and
exactly where §5.3 anticipated ("~1e-5 and sampler noise dominates"). The `f_i > 0` cells are all
≤ 0.22 % against a 2 % band. **The dose that was applied is the dose that was registered, in every
cell.**

### 1.6 Seed plan — verified against §3.2 cell by cell

`seed(S{h}{i}, j) = 20260808 + 51000 + 100·(4h+i) + j` reproduced exactly in all 16 cells:
first/last seeds match the §3.2 table, `j = 0…14` in fifteen cells and `j = 0…99` at S23
(20312908–20313007, terminating one seed below S30's 20313008 exactly as §3.2 pre-computed).
**No collision anywhere in the grid.** N = 15 × 15 + 100 = **325 seeds**, as registered. **PASS.**

### 1.7 DS-D4 — the f_host = 0 row, and the SCAN-CONFOUNDED trigger

| cell | f_i | bias | per-seed sd | distinct MAP values | post_sd median |
|---|---|---|---|---|---|
| S00 | 0.00 | **+0.000000000** | 0.000000000 | **1** | 0.000000000 |
| S01 | 0.25 | **+0.000000000** | 0.000000000 | **1** | 0.000000000 |
| S02 | 0.50 | **+0.000000000** | 0.000000000 | **1** | 0.000000000 |
| S03 | 1.00 | **+0.000000000** | 0.000000000 | **1** | 0.000000000 |

All four cells return bias **exactly** +0.000000 (|bias| = 0.0, not "small"), per-seed sd exactly
0.000000, every one of the 60 posteriors on a **single grid point**, and `post_sd` identically zero.

- **DS-D4 → PIN-BINARY.** The registered prediction is confirmed in full: the pin is a property of
  **host exactness alone**, and is completely insensitive to the impostor dose across the entire
  registered range 0 → 1×σ_z. **PIN-GRADED does not fire.**
- **`b(S00) = +0.000000` exactly ⇒ the SCAN-CONFOUNDED trigger does NOT fire.** The σ = 0 apparatus
  anchor reproduces the campaign's T-0 behaviour (all seeds exactly on truth).

### 1.8 §5.2 corner cross-checks — independent replications at different seeds

| cell | replicates | this scan | parent value | Δ | registered tolerance (realized) | verdict |
|---|---|---|---|---|---|---|
| **S33** | MN0 | +0.039667 (SE 0.001333) | +0.034667 ± 0.001579 | **+0.005000** | 3·√(0.001333²+0.001579²) = **0.0061999** | **PASS** (0.81× tol, 2.42σ) |
| **S30** | MEH | +0.006000 (SE 0.000535) | +0.004000 ± 0.000535 | **+0.002000** | 3·√(0.000535²+0.000535²) = **0.0022688** | **PASS** (0.88× tol, 2.64σ) |
| **S03** | MEI | +0.000000, sd 0.000000 | +0.000000, zero spread | **0** | exact equality | **PASS (exact)** |
| **S00** | σ = 0 anchor | +0.000000 | +0.000000 | **0** | exact | **PASS (exact)** |

**Zero CROSS-CHECK-FAILED, zero CROSS-CHECK-DISCREPANT.** Both live corners land high but inside —
S33 at 2.42σ and S30 at 2.64σ, **both in the same (positive) direction**, which is disclosed in §6
(D-2) as a possible common offset between the two seed blocks rather than scored, because the
registered rule is a per-cell tolerance and both cells pass it.

### 1.9 Abort criteria §8 (a)–(e)

| criterion | measured | verdict |
|---|---|---|
| (a) non-finite `ln_post` > 1 % of any cell's seeds | **0 non-finite values** in either channel, all 325 seeds | PASS |
| (b) horizon-drop > 5 % | `n_horizon_dropped == 0` on all 325 seeds | PASS |
| (c) any V-D failure | none (§1.1–1.6) | PASS |
| (d) **any** rail in any cell or channel | `railed_low` = `railed_high` = **0.000** in all 16 cells × 2 channels | PASS |
| (e) per-seed cost > 2 × 0.969 CPU-h | max **1.0984** CPU-h/seed (S33; S32 1.0976); 2× anchor = 1.938 | PASS |

Total realized cost **177.8 CPU-h** across the 16 cells against the ≈259 CPU-h budgeted (§3.1) —
under budget, because most `f_i > 0` cells came in at 0.44–0.71 rather than 0.969 CPU-h/seed.

### 1.10 Branch-1 leg: Amendment A1

Branch 1 also fires if **Amendment A1 returns A1-FAIL**. Recomputed from `MN0X` raw per-seed records:

```
MN0X 1D mean bias (N = 100)  =  +0.037250      (realized SE 0.000494)
| +0.037250 - 0.037237 |     =   0.000013      vs the registered +-0.002 window
```

**A1-PASS**, and by a factor ~150 in margin. It also lands **+0.71σ** from A1 §5's registered point
prediction +0.03685 ± 0.00056. 2D reported alongside: +0.039750. **This leg of branch 1 does not
fire.** (A1's own verdict belongs to A1's readout; quoted here only as the branch-1 input the 2D
prereg registers.)

### 1.11 SCAN-CONFOUNDED trigger set — member by member

| # | §7 branch-1 member | fires? |
|---|---|---|
| 1 | `b(S00) ≠ 0.000000` | **NO** — exactly zero (§1.7) |
| 2 | any §8 validity check fails | **NO** — V-D1/D3/D4/D5/D6/D7 all pass (§1.1–1.8) |
| 3 | any cell's §5.3 dosing verification out of tolerance | **NO** — 16/16 in tolerance (§1.5) |
| 4 | Amendment A1 returns A1-FAIL | **NO** — A1-PASS at 0.000013 vs 0.002 (§1.10) |

> ### **SCAN-CONFOUNDED DOES NOT FIRE — 0 of 4 members.** The measurements below stand.

---

## 2. DS-D1 — the surface, both channels

### 2.1 1D (headline channel, carried from the parent)

| cell | f_h | f_i | N | bias | SE | per-seed sd | post_sd median | rails | non-finite | distinct MAPs |
|---|---|---|---|---|---|---|---|---|---|---|
| S00 | 0.00 | 0.00 | 15 | **+0.000000** | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0 | 1 |
| S01 | 0.00 | 0.25 | 15 | **+0.000000** | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0 | 1 |
| S02 | 0.00 | 0.50 | 15 | **+0.000000** | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0 | 1 |
| S03 | 0.00 | 1.00 | 15 | **+0.000000** | 0.000000 | 0.000000 | 0.000000 | 0.000 | 0 | 1 |
| S10 | 0.25 | 0.00 | 15 | **+0.004667** | 0.000333 | 0.001291 | 0.000070 | 0.000 | 0 | 2 |
| S11 | 0.25 | 0.25 | 15 | **+0.012667** | 0.000667 | 0.002582 | 0.002331 | 0.000 | 0 | 2 |
| S12 | 0.25 | 0.50 | 15 | **+0.012000** | 0.000655 | 0.002535 | 0.002487 | 0.000 | 0 | 2 |
| S13 | 0.25 | 1.00 | 15 | **+0.014000** | 0.000724 | 0.002803 | 0.002615 | 0.000 | 0 | 3 |
| S20 | 0.50 | 0.00 | 15 | **+0.005333** | 0.000333 | 0.001291 | 0.000228 | 0.000 | 0 | 2 |
| S21 | 0.50 | 0.25 | 15 | **+0.019000** | 0.000724 | 0.002803 | 0.002672 | 0.000 | 0 | 3 |
| S22 | 0.50 | 0.50 | 15 | **+0.016000** | 0.000724 | 0.002803 | 0.002764 | 0.000 | 0 | 3 |
| **S23** | 0.50 | 1.00 | **100** | **+0.023650** | **0.000431** | 0.004314 | 0.003352 | 0.000 | 0 | 4 |
| S30 | 1.00 | 0.00 | 15 | **+0.006000** | 0.000535 | 0.002070 | 0.000731 | 0.000 | 0 | 2 |
| S31 | 1.00 | 0.25 | 15 | **+0.022000** | 0.000951 | 0.003684 | 0.002870 | 0.000 | 0 | 4 |
| S32 | 1.00 | 0.50 | 15 | **+0.023333** | 0.000630 | 0.002440 | 0.003198 | 0.000 | 0 | 2 |
| S33 | 1.00 | 1.00 | 15 | **+0.039667** | 0.001333 | 0.005164 | 0.004384 | 0.000 | 0 | 4 |

**The surface, 1D:**

| f_h \ f_i | 0.0 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| **0.0** | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| **0.25** | +0.004667 | +0.012667 | +0.012000 | +0.014000 |
| **0.5** | +0.005333 | +0.019000 | +0.016000 | **+0.023650** |
| **1.0** | +0.006000 | +0.022000 | +0.023333 | +0.039667 |

**The displacement is displaced far beyond its own claimed width in every dosed cell**, exactly as in
the campaign: e.g. S33 bias +0.039667 against a `post_sd` median of 0.004384 (9.0× its own width);
S23 +0.023650 against 0.003352 (7.1×).

### 2.2 2D (reported alongside in every cell, parent §6 convention)

| cell | 2D bias | 2D SE | 2D post_sd median | 1D − 2D |
|---|---|---|---|---|
| S00 | +0.000000 | 0.000000 | 0.000000 | +0.000000 |
| S01 | +0.000000 | 0.000000 | 0.000000 | +0.000000 |
| S02 | +0.000000 | 0.000000 | 0.000000 | +0.000000 |
| S03 | +0.000000 | 0.000000 | 0.000000 | +0.000000 |
| S10 | +0.005000 | 0.000000 | 0.000077 | −0.000333 |
| S11 | +0.013000 | 0.000655 | 0.002271 | −0.000333 |
| S12 | +0.013000 | 0.000655 | 0.002535 | −0.001000 |
| S13 | +0.015333 | 0.000909 | 0.002650 | −0.001333 |
| S20 | +0.005333 | 0.000333 | 0.000283 | +0.000000 |
| S21 | +0.019333 | 0.000826 | 0.002607 | −0.000333 |
| S22 | +0.017000 | 0.000816 | 0.002785 | −0.001000 |
| S23 | +0.024650 | 0.000416 | 0.003387 | −0.001000 |
| S30 | +0.006000 | 0.000535 | 0.000793 | +0.000000 |
| S31 | +0.023000 | 0.000951 | 0.002891 | −0.001000 |
| S32 | +0.025000 | 0.000488 | 0.003329 | −0.001667 |
| S33 | +0.042000 | 0.001528 | 0.004428 | −0.002333 |

**No 1D/2D split.** 2D tracks 1D in every cell, same sign, and — checked explicitly — **produces the
identical classification at every registered decision point**: the f_h = 0 row is exactly zero in
both channels; DS-D2 is NON-ADDITIVE in both; DS-D3 at S23 gives SHAPE-INTERACTION in both
(2D b = +0.024650 ≥ 0.01150132). 2D runs **+0.000333 to +0.002333 above** 1D, the offset growing with
total dose — the same ordering the parent arms showed (MN0 1D +0.034667 / 2D +0.037000). Every 1D−2D
gap is below one h-grid step (0.005) in the mean. Reported, not scored: the prereg registers no
numerical split threshold (§6, D-3).

---

## 3. SCORECARD — DS-D1 … DS-D6 against the registered bands

### 3.1 DS-D2 — additivity, the primary test

`D(f_h,f_i) = b(f_h,f_i) − b(f_h,0) − b(0,f_i) + b(0,0)`, with
`SE_D = √(s²+s²+s²+s²)` on the four **realized** cell SEs, as registered.
Because the entire f_h = 0 row is exactly zero with zero spread, every `D` reduces to
`b(f_h,f_i) − b(f_h,0)` and `SE_D` to `√(s(f_h,f_i)² + s(f_h,0)²)`.

| cell | arithmetic | D | SE_D | \|D\|/SE_D | class |
|---|---|---|---|---|---|
| S11 | +0.012667 − 0.004667 − 0 + 0 | +0.008000 | 0.000745 | **10.73** | **NON-ADDITIVE** |
| S12 | +0.012000 − 0.004667 − 0 + 0 | +0.007333 | 0.000735 | **9.98** | **NON-ADDITIVE** |
| S13 | +0.014000 − 0.004667 − 0 + 0 | +0.009333 | 0.000797 | **11.71** | **NON-ADDITIVE** |
| S21 | +0.019000 − 0.005333 − 0 + 0 | +0.013667 | 0.000797 | **17.15** | **NON-ADDITIVE** |
| S22 | +0.016000 − 0.005333 − 0 + 0 | +0.010667 | 0.000797 | **13.39** | **NON-ADDITIVE** |
| S23 | +0.023650 − 0.005333 − 0 + 0 | +0.018317 | 0.000545 | **33.60** | **NON-ADDITIVE** |
| S31 | +0.022000 − 0.006000 − 0 + 0 | +0.016000 | 0.001091 | **14.66** | **NON-ADDITIVE** |
| S32 | +0.023333 − 0.006000 − 0 + 0 | +0.017333 | 0.000826 | **20.98** | **NON-ADDITIVE** |
| S33 | +0.039667 − 0.006000 − 0 + 0 | **+0.033667** | 0.001436 | **23.44** | **NON-ADDITIVE** |

> **DS-D2 → NON-ADDITIVE at S33 at 23.4σ, and NON-ADDITIVE at all nine interior cells.**
> **H-ADD is refuted.** §4.2's pre-computed prediction from the parent's corners was
> `D(1,1) = +0.030667` at 9.7σ / 18.4σ against the two pre-stated SE_D brackets; this scan's own
> fresh seeds return **+0.033667** — the prediction is confirmed at **1.41σ** of the difference
> (Δ = +0.003000 against SE = √(0.001436² + 0.001579²) = 0.002133 → 1.41σ). §9's "H-ADD is expected
> to be refuted" is confirmed; the branch-4 ADDITIVE route is dead.

**Note on the realized SE_D.** It came in at 0.00055–0.00144, i.e. **2.2–3.0× tighter** than the
§4.2 "expected" bracket 0.0016672 and 2.2–5.8× tighter than the "conservative" 0.0031580, because
(i) the f_h = 0 row is exactly degenerate and (ii) the interior cells realized SEs of 0.00043–0.00133
against the 0.001579 worst case. **The registered rule was applied with the realized SEs exactly as
§4.2 directs; no band was moved.**

### 3.2 DS-D3 — shape discrimination at S23 (N = 100), the registered rule applied verbatim

```
measured  b(S23)  =  +0.02365000        (N = 100, realized SE 0.00043144)

registered rule (§4.3, fixed at registration, applied unchanged):
   SHAPE-INTERACTION  iff  b(S23) >= 0.01150132
   SHAPE-THRESHOLD    iff  b(S23) <= 0.00783208
   SHAPE-UNDECIDED    otherwise

0.02365000 >= 0.01150132   ->  TRUE
```

> ### **DS-D3 at S23 → SHAPE-INTERACTION.**
> Distance above the SHAPE-INTERACTION boundary: **+0.01214868 = +28.2 realized SE**
> (+19.9 registered SE). The call is not close to the dead-band in either direction.

**Secondary at S13 (N = 15, unchanged, corroborative only):** measured **+0.014000** (SE 0.000724)
against `SHAPE-INTERACTION iff b ≥ 0.0095703` → **SHAPE-INTERACTION**, at +6.1 cell SE above the
boundary. **S13 and S23 agree**, so the §4.3 disagreement-routes-to-branch-5 clause does not fire.

**But the same cell refutes the hypothesis whose label it just returned — stated here, in the
scorecard, because it is a registered quantity and not an interpretation.** H-INT's own point
prediction at S23 is **0.017333**; H-THRESH's is **0.002000**:

```
b(S23) - 0.017333 (H-INT)     =  +0.00631700  =  +14.6 realized SE
b(S23) - 0.002000 (H-THRESH)  =  +0.02165000  =  +50.2 realized SE
```

**The measurement lies above BOTH registered predictions**, 14.6σ above the nearer one. The
registered rule is a one-sided threshold with no upper edge, so **SHAPE-INTERACTION is what the rule
returns; it is not evidence that H-INT is correct.** §7 is analysed in full below.

### 3.3 DS-D4 — the pin test

> **PIN-BINARY** (§1.7). All four f_h = 0 cells return exactly +0.000000 with exactly zero spread and
> a single grid point. **PIN-GRADED does not fire.**

### 3.4 DS-D5 — linearity in the impostor dose along f_host = 1

Registered line through (0, 0.004000)–(1, 0.034667); departure edge ±0.004737 = 3·σ_cell.

| cell | f_i | measured | registered-line prediction | Δ | Δ / realized cell SE | class |
|---|---|---|---|---|---|---|
| S31 | 0.25 | +0.022000 | +0.011667 | **+0.010333** | **+10.9** | **SUPER-LINEAR** (Δ ≥ 0.004737) |
| S32 | 0.50 | +0.023333 | +0.019333 | +0.004000 | +6.4 | **LINEAR-CONSISTENT** (Δ < 0.004737) |

> **DS-D5 → SUPER-LINEAR at S31; LINEAR-CONSISTENT at S32. The row is NOT a straight line.**

**Robustness to the anchor choice** (reported, not a re-scoring): re-drawing the line through this
scan's *own* row endpoints S30 = +0.006000 and S33 = +0.039667 gives predictions +0.014417 and
+0.022833, so S31 is **+0.007583 (+8.0 SE)** high and S32 is **+0.000500 (+0.8 SE)** on the line.
**S31's super-linearity survives the anchoring choice at ≥8σ.** §9's registered expectation
("DS-D5 is expected LINEAR-CONSISTENT along f_host = 1, from M5′'s Δζ ∝ σ") is **refuted**, and
refuted in the *super*-linear direction, which §9 did not name in either direction.

### 3.5 DS-D6 — R_dose per cell

`R_dose = bias / (f_i · 0.041813)`, f_i > 0 only.

| cell | R_dose | band |
|---|---|---|
| S01 / S02 / S03 | 0.0000 / 0.0000 / 0.0000 | UNBANDED |
| S11 / S12 / S13 | 1.2117 / 0.5740 / 0.3348 | UNBANDED |
| S21 / S22 / S23 | 1.8176 / 0.7653 / 0.5656 | UNBANDED |
| S31 / S32 | 2.1046 / 1.1161 | UNBANDED |
| **S33** | **0.9487** | **[0.75, 1.25] → IN BAND** |

> **DS-D6 → S33 IN BAND** (the only banded cell; MN0's own anchor is 0.8291). Every other cell is
> **REPORTED UNBANDED** as registered. The unbanded values are printed for the record only; §4.6
> declines to assert a band for them and this readout does not supply one.

### 3.6 Scorecard summary

| statistic | registered classes | **result** |
|---|---|---|
| **DS-D1** | report the surface, both channels | reported (§2); no 1D/2D split |
| **DS-D2** | NON-ADDITIVE / AMBIGUOUS / ADDITIVE-CONSISTENT | **NON-ADDITIVE at S33 (23.4σ) and at all 9 interior cells** |
| **DS-D3** | SHAPE-INTERACTION / SHAPE-THRESHOLD / SHAPE-UNDECIDED | **SHAPE-INTERACTION at S23** (+28.2 SE above the boundary); S13 agrees |
| **DS-D4** | PIN-BINARY / PIN-GRADED | **PIN-BINARY** |
| **DS-D5** | LINEAR-CONSISTENT / SUB- / SUPER-LINEAR | **SUPER-LINEAR at S31**; LINEAR-CONSISTENT at S32 |
| **DS-D6** | [0.75, 1.25] at S33 only | **IN BAND (0.9487)**; all others unbanded |

---

## 4. THE SHAPE — what the surface does, and what it does not support

**This is the hard part of the read and it is reported as a difficulty, not resolved into a box.**
Every departure below is quoted in units of the **realized** SEs of the cells involved, and each is
labelled as either **>3σ** (established at the registered resolution classes) or **consistent with
noise at the registered resolution floor**.

### 4.1 The two registered hypotheses, tested where they make predictions

**H-THRESH is refuted outright, independently of the S23 rule.** Its registered signature is
"a **step in f_host between rows h = 2 and h = 3**, with **no ramp through row h = 1**", i.e.
`b(f_h, f_i) ≈ a·f_h` (edge only) for f_h < f* = 0.5262. Measured at f_i = 1.0:

```
b(0.25, 1.0)  =  +0.014000 +- 0.000724      H-THRESH predicts +0.001000 (edge only)
excess = +0.013000  =  +18.0 SE                                  >>> 3 sigma
b(0.50, 1.0)  =  +0.023650 +- 0.000431      H-THRESH predicts +0.002000
excess = +0.021650  =  +50.2 SE                                  >>> 3 sigma
```

**Row h = 1 ramps, hard, at a host dose (0.25) less than half the registered threshold f* = 0.5262.**
There is no step and no dead zone below f*. H-THRESH is wrong by 18σ at the first cell that tests it.

**H-INT is refuted too — not by its label, but by its arithmetic.** Its registered signature is
`D(f_h,f_i) = I·f_h·f_i`, **strictly bilinear**. With the registered anchor `I = 0.030667`:

| cell | f_h·f_i | D measured | bilinear prediction | residual | residual / SE_D | status |
|---|---|---|---|---|---|---|
| S11 | 0.0625 | +0.008000 | +0.001917 | +0.006083 | **+8.16** | >3σ — *§6 item 1 NOT-EVALUABLE* |
| S12 | 0.1250 | +0.007333 | +0.003833 | +0.003500 | **+4.76** | >3σ — *§6 item 1 NOT-EVALUABLE* |
| S21 | 0.1250 | +0.013667 | +0.003833 | +0.009833 | **+12.34** | >3σ — *§6 item 1 NOT-EVALUABLE* |
| S13 | 0.2500 | +0.009333 | +0.007667 | +0.001667 | +2.09 | below 3σ |
| S22 | 0.2500 | +0.010667 | +0.007667 | +0.003000 | **+3.76** | **>3σ** |
| S31 | 0.2500 | +0.016000 | +0.007667 | +0.008333 | **+7.64** | **>3σ** |
| S23 | 0.5000 | +0.018317 | +0.015333 | +0.002983 | **+5.47** | **>3σ (N = 100)** |
| S32 | 0.5000 | +0.017333 | +0.015333 | +0.002000 | +2.42 | below 3σ |
| S33 | 1.0000 | +0.033667 | +0.030667 | +0.003000 | +2.09 | below 3σ |

**Every residual is positive**, and four of them clear 3σ **outside** the NOT-EVALUABLE low corner:
S22 (+3.8σ), S31 (+7.6σ), S23 (+5.5σ at N = 100), plus the three low-corner cells which are
registered as unable to test bilinearity (§6 item 1) and are therefore **excluded from the
conclusion**, shown only for completeness. Re-anchoring `I` on this scan's own `D(1,1) = 0.033667`
(which forces the S33 residual to zero by construction) leaves **S31 at +7.0σ** and the low corner at
+7.9σ/+4.3σ/+11.9σ; S23 falls to +2.7σ and S22 to +2.8σ. **Under either anchoring, S31's departure
from bilinearity is established at ≥7σ.**

> **Reading, stated flatly: the DS-D3 rule at S23 returns SHAPE-INTERACTION because the rule is a
> one-sided threshold with no upper edge, and the measurement is far above it. But the measurement is
> also 14.6σ above H-INT's own point prediction, and the bilinear signature `D = I·f_h·f_i` is
> rejected at ≥5σ at S23 and ≥7σ at S31. The surface is non-additive and it is not a threshold — and
> it is not the registered product form either.**

### 4.2 The f_host = 1 row: steep → flat → steep

This row is where **H-INT and H-THRESH are degenerate** (§4.3 of the prereg: "both … degenerate — no
discrimination"): both predict the **same straight line** `b = a + I·f_i`. What is measured:

```
f_i:        0.0        0.25       0.5        1.0
b:       +0.006000  +0.022000  +0.023333  +0.039667
```

Successive steps, with realized SE_diff and the §4.0 resolution classes:

| step | Δ | SE_diff | σ | class |
|---|---|---|---|---|
| S31 − S30 (f_i 0 → 0.25) | **+0.016000** | 0.001091 | **14.66** | **RESOLVED** |
| S32 − S31 (0.25 → 0.5) | +0.001333 | 0.001141 | **1.17** | **UNRESOLVED** |
| S33 − S32 (0.5 → 1.0) | **+0.016333** | 0.001475 | **11.08** | **RESOLVED** |

**What is established (>3σ) and what is not — the distinction matters and is easy to get wrong:**

- **NOT established:** that the middle interval is *flat*. The step S32 − S31 = +0.001333 sits at
  **1.17σ** — **UNRESOLVED**, i.e. entirely consistent with noise at the registered floor. **No claim
  that the surface plateaus between f_i = 0.25 and 0.5 is supported by this scan.**
- **ESTABLISHED at >3σ:** that the row is **not a straight line**. The comparison that carries this
  is the *change in slope*, not the middle step in isolation:

```
second difference (step2 - step1)  =  +0.001333 - 0.016000  =  -0.014667
SE = sqrt(se30^2 + 4*se31^2 + se32^2)
   = sqrt(0.000535^2 + 4*0.000951^2 + 0.000630^2)  =  0.002074
   ->  -7.07 sigma                                              >>> 3 sigma

per-unit slopes  m1 = 0.064000 +- 0.004364   (f_i 0 -> 0.25)
                 m2 = 0.005333 +- 0.004563   (0.25 -> 0.5)
                 m3 = 0.032667 +- 0.002949   (0.5 -> 1.0)
m2 - m1 = -0.058667 +- 0.006315  =  -9.29 sigma                 >>> 3 sigma
m3 - m2 = +0.027333 +- 0.005434  =  +5.03 sigma                 >>> 3 sigma
```

> **The f_host = 1 row is non-linear at 9.3σ (slope drop after f_i = 0.25) and 5.0σ (slope recovery
> after f_i = 0.5). Both registered hypotheses predict a straight line on this row, and both are
> refuted there — on the one row where they agree with each other.** This is the same fact DS-D5
> scores as SUPER-LINEAR at S31 (+10.9 SE), reached by a second, independent registered statistic.

**What this does NOT license.** A three-interval slope pattern is exactly the kind of functional
claim §4.0 and §6 item 8 bar: the registered floor supports **≈5.2 distinguishable levels** across
the full dynamic range, and this row has four points. **The established statement is "not a straight
line, with a resolved slope drop and a resolved slope recovery." "Sigmoid", "two-component",
"saturating-then-reactivating", or any parametric fit to four points is BARRED and is not offered.**
I record explicitly that the S-shape *is* tempting to fit and that the grid cannot resolve it.

### 4.3 The f_host = 0.5 dip — NOT established

```
S22 - S21  =  +0.016000 - 0.019000  =  -0.003000 +- 0.001024  =  -2.93 sigma  -> MARGINAL
```

Below the 3σ RESOLVED edge. Its counterpart one row down is weaker still:

```
S12 - S11  =  +0.012000 - 0.012667  =  -0.000667 +- 0.000934  =  -0.71 sigma  -> UNRESOLVED
```

Pooling the two (an **unregistered, post-hoc** combination, shown only to close the question honestly
rather than leave it open by omission) gives **−0.003667 ± 0.001386 = −2.65σ**, still below 3σ.

> **The dip at f_imp = 0.5 is NOT established. It is consistent with noise at the registered
> resolution floor in both rows that show it, individually and pooled. No account of the surface may
> lean on it, and this readout does not.** What *is* established on the f_h = 0.5 row is that it is
> not monotone-linear either: S21 − S20 = +0.013667 (**17.2σ**, RESOLVED) then S23 − S22 = +0.007650
> (**9.1σ**, RESOLVED) with a non-resolved excursion in between — the same steep-then-slow-then-steep
> shape as the f_h = 1 row, but at this row's precision the middle interval cannot be separated from
> noise.

### 4.4 The f_host = 0 row: an absolute gate

Not a small number — **exactly zero**, in 60/60 seeds, at impostor doses spanning the entire
registered range (σ̄ = 0 → 0.0417, i.e. the full production GLADE dose). Every posterior on a single
grid point; per-seed sd 0.000000; `post_sd` 0.000000. There is no gradient to measure: the pin is
**binary in the host dose and totally insensitive to the impostor dose.**

### 4.5 The f_imp = 0 column: also a fast switch, then flat

```
f_h:        0.0        0.25       0.5        1.0
b:       +0.000000  +0.004667  +0.005333  +0.006000
```

| step | Δ | SE_diff | σ | class |
|---|---|---|---|---|
| S10 − S00 | **+0.004667** | 0.000333 | **14.00** | **RESOLVED** |
| S20 − S10 | +0.000667 | 0.000471 | 1.41 | UNRESOLVED |
| S30 − S20 | +0.000667 | 0.000630 | 1.06 | UNRESOLVED |

**§5.4's pre-stated sub-prediction is CONFIRMED**: the column is *small and positive across all four
host doses* and **does not reproduce pp_coverage's negative sign** anywhere. **No cell of the column
turns negative** (the registered first-class-finding alternative does not fire). The column also
shows the same switch-then-saturate shape as the rows: the entire column value is delivered by the
first quarter-dose, and nothing resolved is added by the remaining 0.75.

### 4.6 The impostor direction: on fast, then saturating — and the saturation depends on f_host

Interaction residual normalised on its own row, `D(f_h,f_i)/D(f_h,1)`:

| f_h | f_i = 0.25 | f_i = 0.5 | f_i = 1.0 |
|---|---|---|---|
| 0.25 | 0.857 | 0.786 | 1.000 |
| 0.50 | 0.746 | 0.582 | 1.000 |
| 1.00 | 0.475 | 0.515 | 1.000 |

At f_h = 0.25, **86 %** of the full-impostor-dose interaction is already delivered by a quarter dose;
at f_h = 1.0, only **48 %**. The interaction therefore does **not** factor as `f(f_h)·g(f_i)` — the
*shape* in f_i changes with f_h, which is a stronger statement than non-additivity and is precisely
what makes the bilinear form fail. **But the row-to-row differences in these ratios are a
second-order comparison on cells whose individual steps are partly UNRESOLVED, and §6 item 8 bars
turning them into a functional form. They are reported as the descriptive numbers they are.**

### 4.7 What the surface supports, and what it does not — plainly

**Supported (each at >3σ against the registered resolution classes):**

1. **The host dose is an absolute gate.** At f_h = 0 the bias is exactly zero at every impostor dose,
   with zero spread and a degenerate posterior. Removing the host's uncertainty removes the defect
   completely — not by 90 %, completely.
2. **The bias is strongly non-additive.** D ≥ 10σ at every interior cell; +33.7σ at S33. H-ADD dead.
3. **Both dose directions switch on fast and then grow slowly.** In every dosed row the first
   quarter-dose delivers 48–86 % of the full-dose interaction; the f_imp = 0 column delivers 78 % of
   its total in its first quarter-step and adds nothing resolved thereafter.
4. **Neither direction is a threshold.** Row h = 1 (f_h = 0.25, less than half the registered
   f* = 0.5262) already carries +0.014000 at f_i = 1 — 18σ above H-THRESH's edge-only prediction.
5. **Neither direction is bilinear.** The bilinear residual is positive everywhere and >3σ at S22,
   S31 and S23 (and at ≥7σ at S31 under both anchorings).
6. **The f_host = 1 row is not a straight line** (slope change −9.3σ then +5.0σ), on the one row where
   the two registered hypotheses are degenerate and both demand a straight line.
7. **The bias remains positive everywhere**, and the f_imp = 0 column stays small and positive.
8. **No 1D/2D split**; every classification is identical in both channels.

**NOT supported (do not appear in any conclusion below):**

1. **That the surface plateaus between f_i = 0.25 and 0.5 at f_h = 1** — the step is 1.17σ,
   UNRESOLVED.
2. **That the f_host = 0.5 row dips at f_imp = 0.5** — 2.93σ (MARGINAL), and 0.71σ in the row below;
   pooled 2.65σ. Consistent with noise.
3. **Any functional form** — sigmoid, saturating exponential, two-component, power law, or a fitted
   `f(f_h)·g(f_i)` — for either direction or for the surface. §4.0's registered floor gives
   **≈5.2 distinguishable levels** across the range; §6 item 8 bars the claim; a 4×4 grid with four
   levels per axis cannot separate these. **I record that fitting one was tempting and that it is
   barred, rather than fitting one and caveating it.**
4. **Anything about how the interaction scales with K** (§6 item 3, pinned at `real_k`), about
   transfer to production `BayesianStatistics` (§6 item 4), or about reweighting (§5.1 — pre-emptively
   barred, and no weighting variant was run).

---

## 5. THE BRANCH THAT FIRES — checked in the registered order, zero judgement calls

| order | branch | registered condition | fires? |
|---|---|---|---|
| 1 | **SCAN-CONFOUNDED** | b(S00) ≠ 0; or any §8 validity check fails; or any cell's §5.3 dosing out of tolerance; or Amendment A1 returns A1-FAIL | **NO** — 0 of 4 members (§1.11) |
| 2 | **INTERACTION-BILINEAR** | DS-D2 **NON-ADDITIVE at S33** *and* DS-D3 **SHAPE-INTERACTION at S23** | **YES** — NON-ADDITIVE at 23.4σ (§3.1) **and** SHAPE-INTERACTION at S23, +28.2 SE above the boundary (§3.2) |
| 3 | **INTERACTION-THRESHOLD** | DS-D2 NON-ADDITIVE at S33 *and* DS-D3 **SHAPE-THRESHOLD** at S23 | not reached (and would be NO: SHAPE-THRESHOLD requires b(S23) ≤ 0.00783208; measured +0.023650) |
| 4 | **ADDITIVE** | \|D(1,1)\| below the DS-D2 non-additivity edge | not reached (and would be NO: 23.4σ above it) |
| 5 | **UNDECIDED (first-class, non-forcing)** | *anything else* — SHAPE-UNDECIDED, a 1D/2D split, PIN-GRADED, **a resolved but non-bilinear and non-threshold surface**, or an f_imp = 0 column that turns negative | **not reached under the registered ordering.** Of its listed conditions: SHAPE-UNDECIDED **no**; 1D/2D split **no**; PIN-GRADED **no**; f_imp = 0 column negative **no**; **"a resolved but non-bilinear and non-threshold surface" — factually YES (§4.1, §4.2)** |

> ### BRANCH FIRED BY THE TREE: **branch 2 — INTERACTION-BILINEAR**
>
> - **Headline channel (1D):** DS-D2 NON-ADDITIVE at S33 (D = +0.033667, 23.4σ) ✓; DS-D3
>   SHAPE-INTERACTION at S23 (b = +0.023650 ≥ 0.01150132) ✓. **Both conditions of branch 2 are met.**
> - **2D reported alongside:** identical — NON-ADDITIVE at S33 (D = +0.036000) and SHAPE-INTERACTION
>   at S23 (b = +0.024650). **No 1D/2D split.**
> - **Corroborating secondary:** S13 also returns SHAPE-INTERACTION, so §4.3's disagreement clause
>   does not fire.
> - **Disclosures required alongside the verdict line:** SCAN-CONFOUNDED — *not* triggered;
>   CROSS-CHECK-FAILED — *not* triggered (S33 2.42σ, S30 2.64σ, both inside tolerance);
>   CROSS-CHECK-DISCREPANT — *not* triggered; PIN-GRADED — *not* triggered; rails — *nowhere*;
>   **DS-D5 SUPER-LINEAR at S31 — triggered.**

> ### **THE TREE AND THE SURFACE DISAGREE, AND THAT IS THE FINDING.**
>
> Branch 2's **condition** is a conjunction of two coarse tests, and the data satisfies both. Branch
> 2's pre-stated **meaning** is *"the bias is a genuine product-form interaction"*, and H-INT's
> signature — `D(f_h,f_i) = I·f_h·f_i`, **strictly bilinear** (§4.3) — is **rejected by this scan's
> own registered statistics**: bilinear residuals >3σ at S22 (+3.8σ), S31 (+7.6σ) and S23 (+5.5σ at
> N = 100), all positive; DS-D5 SUPER-LINEAR at S31 (+10.9 SE, ≥8σ under self-anchoring); and the
> f_h = 1 row — the row on which H-INT and H-THRESH are *degenerate and both demand a straight line* —
> non-linear at 9.3σ and 5.0σ. The measured b(S23) is 14.6σ **above** H-INT's own point prediction.
>
> **Branch 5's listed condition "a resolved but non-bilinear and non-threshold surface" is factually
> satisfied, but branch 5 is checked last and is not reached, because branch 2's condition fires
> first.** The registered ordering therefore routes a surface that H-INT does not describe into the
> branch whose meaning clause asserts H-INT's form. **This readout does not choose between them.**
>
> The mechanical facts, stated so the author can rule on either reading:
>
> | | |
> |---|---|
> | Branch 2 condition as written | **SATISFIED** (23.4σ and +28.2 SE) |
> | Branch 2 meaning clause (product form) | **CONTRADICTED** (≥5σ at S23, ≥7σ at S31) |
> | Branch 5 condition "resolved but non-bilinear and non-threshold" | **SATISFIED**, unreachable in the registered order |
> | Branch 3 (threshold) | **REFUTED** on its own terms, 18σ at S13 and 50σ at S23 |
>
> If the author reads the tree **by its ordering**, branch 2 fires and its meaning clause must be
> read as *falsified in the same breath as the branch is entered* — in which case §4.3's registered
> reading of an UNDECIDED-type outcome applies in substance: *"the measured surface sits in a region
> neither registered hypothesis predicts — i.e. both H-INT and H-THRESH are quantitatively wrong,
> not merely unresolved. That is a substantive negative result about the two accounts on the
> register."* If the author reads the tree **by its conditions taken as a set**, branch 5 fires, and
> with it the bars that branch 5 carries: **neither H-INT nor H-THRESH may be quoted, and no repair
> may be proposed.**
>
> **This readout takes the more restrictive of the two throughout: it quotes neither H-INT nor
> H-THRESH as an account, and it proposes no repair.**
>
> **NOT ADJUDICATED HERE.** Per §10 the branch call is presented to the author. The author's ruling
> is what makes this a verdict of record; only the Ship agent may append a verdict block below the
> prereg's verdict line, and only if the author hands one down.

---

## 6. DISCLOSURE LIST

| # | disclosure |
|---|---|
| **D-1** | **The branch-2 / branch-5 ambiguity above is the single most important item in this readout** and is surfaced as a defect of the registered tree, not of the data. The tree's ordering and one of its meaning clauses give different answers on this surface. Registered pre-data; not adjusted (§4.7 anti-tuning). |
| **D-2** | **Both live corner cross-checks land high, in the same direction.** S33 is +0.005000 (2.42σ) above MN0 and S30 is +0.002000 (2.64σ) above MEH. Each passes its own registered 3σ tolerance, and the registered rule is per-cell, so **CROSS-CHECK-FAILED does not fire**. But the coincidence of sign is noted: if the +51000 block carries a small positive offset relative to the +50000 block, the whole surface is shifted up by ~0.002–0.005 and the *levels* would move while the *shape* statistics (all of which are differences within the block) would not. All conclusions in §4 are difference-based and are insensitive to a common offset; the DS-D6 R_dose values and the DS-D5 registered-line comparison are **not**, and are the two places this matters. Reported, not scored. |
| **D-3** | **No numerical 1D/2D split threshold is registered.** 2D exceeds 1D by +0.000333…+0.002333, the gap growing with total dose, all below one h-grid step and all with identical classifications. Called "no split" on that basis; the author may read the growth-with-dose as a finding. |
| **D-4** | **Realized precision beat the registered floor.** Realized per-cell SEs are 0.00033–0.00133 against the registered worst case 0.001579, and realized SE_D 0.00055–0.00144 against the expected bracket 0.0016672 — 2.2–3.0× tighter, mainly because the f_h = 0 row is exactly degenerate. The registered classes were applied with realized SEs exactly as §4.0/§4.2 direct. **The registered ~5.2-level resolution bar (§4.0, §6 item 8) is NOT relaxed by this**, even though the realized worst-case floor would give ~7 levels; the bar is a registered number and no functional-form claim is made under either. |
| **D-5** | **V-D5 is scope-limited.** The 16 cells are on fresh disjoint seeds and have no committed golden. V-D5 was verified through the only chain that carries one — MN0's 15 stored seeds vs the same seeds inside MN0X — at max relative deviation **0.000e+00** with MAPs exactly equal. AD-1/AD-2/AD-3 (§2.2) are unit-test obligations discharged before the cells ran; this readout verifies their observable consequences (host-mask count via the 982 pin, and the §5.3 dosing table) but does not re-execute the unit tests. |
| **D-6** | **S23's realized SE (0.000431) is 29 % below its registered SE (0.00061154)**, because the per-seed spread at (0.5, 1.0) is 0.004314 rather than MN0's 0.0061154. The **registered decision boundaries 0.01150132 / 0.00783208 were applied unchanged** (§4.7). The realized SE is used only for the *distance-in-σ* reporting, and using the registered SE instead moves the S23 call from +28.2 to +19.9 SE above the boundary — same verdict either way. |
| **D-7** | **The three low-corner cells S11/S12/S21 show the largest bilinearity residuals (+8.2σ, +4.8σ, +12.3σ)**, and they are **registered NOT-EVALUABLE for exactly that test** (§6 item 1). They are excluded from every conclusion in §4 and §5 and shown for completeness only. Note for the author: §6 item 1 justified the exclusion with an *expected* SE_D of 0.0016672, and the realized SE_D there is 0.00074–0.00080; the registered escape (N = 100 on those cells) remains **author-order only** and is not requested here. |
| **D-8** | **Cost.** Realized 177.8 CPU-h against ≈259 budgeted; worst per-seed 1.0984 CPU-h (S33), inside abort criterion (e)'s 2× anchor. `f_i = 0` cells ran at 0.011–0.019 CPU-h/seed, confirming §3.1's point-evaluation-branch account. |
| **D-9** | `git_dirty == true` in all 16 cells with `import_path` dirt **empty** — the dirt is entirely under `results/`. V-D4 as written gates on the import path and passes. Instrument commit `3aedbe55`; the JSONs were committed at `5b0bd17a`. |
| **D-10** | **This readout created two files and touched nothing else**: `SCAN_READOUT.md` and `score_2d_scan.py` (+ its `score_2d_scan_output.json` twin). No `.md` under `results/mechanism_study_20260813/` was modified — the registered prereg, the parent, `ARMS.md`, Amendment A1 and the M1/M3/M4/M5 notes are byte-unchanged. Nothing was committed. |

---

## 7. §6 NOT-EVALUABLE REGISTRY — rows carried forward

| # | item | status after this read |
|---|---|---|
| 1 | Bilinear interaction below f_h·f_i ≈ 0.16 (S11, S12, S21) | **NOT-EVALUABLE — carried.** Excluded from all conclusions (D-7). Escape (N = 100 on those cells) **not requested**; author order only. |
| 2 | Paired variance reduction forfeited | **NOT-EVALUABLE — carried.** All cell-to-cell comparisons in §4 carry the full √2 inflation, as registered; the corner independence it bought is what makes §1.8 a genuine replication. No escape registered. |
| 3 | K-dependence | **NOT-EVALUABLE — carried.** K pinned at `real_k` in all 16 cells (ΣK = 1,193,703 on all 325 seeds). The scan says nothing about how the interaction scales with multiplicity. |
| 4 | Transfer to production `BayesianStatistics` | **NOT-EVALUABLE — carried.** Certified mirror, not the production path. Any estimator fix routes `/physics-change`. |
| 5 | f_incl < 1 / empty-ball events / completeness / window-interior n(z) / sky-cone geometry | **NOT-EVALUABLE — carried.** Read is conditional on host-in-ball over the 982 nonempty-ball events. |
| 6 | The pp_coverage sign flip | **NOT-EVALUABLE — carried.** §5.4's sub-prediction confirmed directionally (§4.5, §8.3) and carries **no branch weight**; K and α both differ, so the f_imp = 0 column is an analogue, not a replication. |
| 7 | Any repair | **NOT-EVALUABLE — carried. No repair is proposed anywhere in this readout.** |
| 8 | Functional forms finer than ~5 levels of contrast | **NOT-EVALUABLE — carried, and actively enforced** (§4.2, §4.6, §4.7). |

---

## 8. WHAT THIS MEANS FOR THE MECHANISM ACCOUNT

Stated as constraints the surface places on any future candidate, and related to the parent §7
closures. **No repair is proposed, and none may be read out of this section.**

### 8.1 The host is an absolute gate; the impostor sea is a graded amplifier

The two ingredients are **not symmetric**, and the asymmetry is the scan's cleanest structural
result:

- **Host dose = 0 ⇒ bias = 0, exactly, at every impostor dose.** Not attenuated — annihilated, with a
  degenerate single-grid-point posterior in 60/60 seeds. One exact redshift at weight 1/K with
  K̄ ≈ 1216 defeats ~1216 fully smeared impostors outright.
- **Impostor dose = 0 ⇒ bias = +0.0047…+0.0060**, small, positive, and essentially flat in the host
  dose (only the first quarter-step is RESOLVED). Removing the sea leaves ~15 % of the full effect.

So the interaction is **gated by the host and driven by the sea** — the host is a switch, the sea is
the gain. This inverts the M5′ split-dose evidence at K = 50 (`M5_smeared_candidate_prior.md` §4.3:
impostors-only +0.0247, host-only +0.0062) on its decisive half, and the parent §0(ii) already
recorded that inversion. **This scan extends it: the impostors-only null is not a coincidence of
f_i = 1, it holds identically across the entire impostor-dose range.** The toy's K-saturation account
("at K = 50 the sharp host cannot pin") is falsified at production K along the whole axis, not just at
one point.

### 8.2 Where a repair must act — three constraints, stated without naming one

Combining this surface with the parent §7 closures:

1. **The parity argument (parent §7) is confirmed and sharpened.** Gaussian convolution is
   `exp(σ²∂²/2)`, even in σ, so any symmetric-smoothing story is O(σ²) and cannot produce a
   first-order dose response. This scan's response along f_i at fixed f_h is not merely first-order —
   it is **steeper than linear at small dose and shallower at large dose** (§4.2, established at 9.3σ
   in the slope change), i.e. **even further from an O(σ²) signature than the campaign's dose ladder
   was.** A symmetric-smoothing term is excluded a second time, by an independent statistic.
2. **The reweighting closure (M5 §4.2, §5.1) is untouched and remains binding.** Every reweighting
   variant made the toy's bias worse or left it unmoved, for the exact algebraic reason that a
   displacement shared by all candidate kernels commutes with any h-independent convex combination.
   **A term that acts through the candidate weights cannot produce this surface**, and in particular
   cannot produce the host gate: a weight change cannot turn a nonzero bias into an *exact* zero with
   a degenerate posterior. §5.1 pre-emptively bars reading any cell as reweighting-repairable, and
   nothing here does.
3. **The pp_coverage sign flip (parent §7, prereg §5.4) is NOT explained by the interaction alone.**
   §5.4's sub-prediction is confirmed exactly: the f_imp = 0 column — the instrument's nearest
   analogue to a single-host estimator with no impostor sea — is **small and POSITIVE at all four
   host doses (+0.0047, +0.0053, +0.0060)** and **never turns negative**. Removing the sea removes
   ~85 % of the positive bias but does **not** reveal pp_coverage's −0.02…−0.046. The pre-named
   carrier of the sign flip therefore remains the parent §7 **M1 negative term** (missing `w_pop`
   volume prior, `Δz = σ_z²·λ`, λ ≈ 2.3–2.9, fitted a ≈ +1.15 / b ≈ −5.29). **Carries no branch
   weight**, and §6 item 6 stands: K (1 vs K̄ ≈ 1216) and the α(h) normalisation both differ, so this
   is an analogue and not a replication.

**Where a repair must act, as a constraint set only:** any surviving candidate term must
**(i)** vanish identically when the host is exact, for *any* impostor dose — an exact zero, not a
small residual; **(ii)** be first-order in the impostor dose but **not** proportional to it, since the
f_i response is resolved-nonlinear on the f_h = 1 row; **(iii)** be non-separable in the two doses,
since the *shape* in f_i changes with f_h (§4.6); **(iv)** survive the parent's α-deletion test
(≈ +0.0165 at σ_z = 0.035 with α removed, parent §7 M4); and **(v)** not be reachable by any
candidate reweighting. **This readout names no term satisfying these, and does not propose one.**
Naming the mechanism is the next stage's job (§6 item 7).

### 8.3 Two registered expectations that failed, recorded as findings

- **§9: "DS-D5 is expected LINEAR-CONSISTENT along f_host = 1, from M5′'s Δζ ∝ σ."** **Refuted** —
  SUPER-LINEAR at S31 by +10.9 SE (≥8σ self-anchored). §9 named sub-linearity as the interesting
  failure ("would reopen the saturation question the MEI collapse closed"); the failure came in the
  **opposite** direction, which §9 did not anticipate at all. M5′'s `Δζ ∝ σ` flank scaling does not
  hold along the host-fully-dosed row.
- **§4.3's structural claim that "H-INT and H-THRESH are degenerate on the f_h = 1 row"** is correct
  as stated — and this scan finds that **the row on which they agree is the row that refutes them
  both**, because the shared prediction is a straight line and the row is not one.

---

## 9. FORMULATION AWAITING THE AUTHOR'S RULING

**Nothing below is adopted. These are the decisions this readout hands up.**

1. **The branch call.** The tree, checked in the registered order, fires **branch 2 —
   INTERACTION-BILINEAR**, on two conditions both satisfied with wide margins. Its **meaning clause
   is contradicted by the scan's own registered statistics at ≥5σ**, and **branch 5's listed
   condition "a resolved but non-bilinear and non-threshold surface" is factually satisfied but
   unreachable in the registered ordering** (§5). **The author's ruling is required on which reading
   governs.** Pending that ruling this readout has behaved as if branch 5 governs — quoting neither
   H-INT nor H-THRESH as an account, and proposing no repair.

2. **Whether the branch-2 meaning clause survives its own condition.** If branch 2 is ruled to have
   fired, the author must also rule whether its pre-stated consequence — *"any candidate term proposed
   downstream must be one that vanishes when either ingredient is removed"* — is retained. Note that
   the scan supports **half** of it very strongly (vanishes exactly when the host is exact) and
   **refutes the other half** (removing the impostor sea leaves +0.0047…+0.0060, not zero). The
   surface is a **gate × amplifier**, not a symmetric product.

3. **§6 item 1's escape (S11, S12, S21 at N = 100).** Registered as author-order only and **not
   requested here.** The relevant fact for the decision is D-7: the realized SE_D in the low corner
   came in at 0.00074–0.00080 against the 0.0016672 the exclusion was written against, and the three
   cells show the largest bilinearity residuals on the grid. Whether that makes the escape worth
   ordering is the author's call, not this readout's.

4. **Whether D-2 (both corners high, same sign) warrants a block-offset check.** Both corner
   cross-checks pass their registered per-cell tolerances; the shared sign is a coincidence at
   2.4σ/2.6σ. It affects the DS-D6 R_dose values and the DS-D5 registered-line comparison, and affects
   **none** of the shape conclusions, which are all differences inside the +51000 block.

5. **What this scan does NOT ask for.** No repair. No `/physics-change` intake. No new arm, no higher
   N at S23 (§4.3: no further-N escape is registered and none is sought), no weighting variant
   (§5.1), no K-ladder (§6 item 3). The registered budget was not exceeded.

---

*This readout is a mechanical scoring plus the interpretation §10 assigns to the readout session. It
appends nothing to any pre-registration, adjudicates nothing, and commits nothing.*
