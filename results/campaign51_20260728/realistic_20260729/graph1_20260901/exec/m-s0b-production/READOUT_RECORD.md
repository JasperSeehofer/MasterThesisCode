# S0-B production READOUT — mechanical dispositions (g-score-null, score_b_re, B0-B)

**Date:** 2026-09-03. **Role:** S0-B production READER (fresh session, read-only except this
file). Mechanical registered dispositions only, per §2.1/§4.1/PA-HIER-29/PA-HIER-31/PA-HIER-33
of `PREREGISTRATION_HIER_HTHETA_20260826.md`. No interpretation offered; `d-photoz-leverage`
(is the theta-pull real venue physics) belongs to the author.

**Sources read (verbatim citations inline below):**
- `results/campaign51_20260728/realistic_20260729/PREREGISTRATION_HIER_HTHETA_20260826.md` —
  §2.1 (Stage-0 arms), §4.1 (bands), `PA-HIER-29`, `PA-HIER-31` (a)-(j) + Revision Notes 1/2,
  `PA-HIER-33` (proposal + ratification-adjacent text).
- `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md` —
  rows #278, #280, #287, #320, #323, #324, #332 (and #274/#275/#276 for PA-HIER-33 provenance).
- `graph1_20260901/exec/m-s0b-production/{LAUNCH_RECORD.md, DRIVER_BNODE_BUILD_RECORD.md,
  RETRIEVAL_RECORD.md}`; `graph1_20260901/exec/b-pahier33-scorer/RECORD.md` (existence/role only,
  not re-read line-by-line here — its build is exercised directly via the driver run below).
- `graph1_20260901/retrieved/s0b_run_20260902/` (5 node dirs under `s0a_seed900101/`, 5
  `provenance_*.json`, `s0a_full_output.json`, `logs/`).
- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py` at repo
  HEAD (`cc176225898c10cf7060e67d77dab0df1803cf20`; the driver file itself is pinned at
  `081b1f28f9d6c36c950954c64f5920f7ea15034d` — see Provenance below) — used via its own
  `--score-only` registered entry point (`gather_node_results_from_disk` /
  `score_only_payload` / `compute_scores`), **plus** an independent hand-recomputation from the
  raw `event_likelihoods.csv` files (g-precision cross-check, §6 below) that reproduces the
  driver's numbers to the digit.

---

## 1. Provenance check

All 5 SLURM array tasks of job `6779532` (`provenance_6779532_4.json`, `_6779535_0.json`,
`_6779536_1.json`, `_6779537_2.json`, `_6779538_3.json`) carry:

```
git_commit:  081b1f28f9d6c36c950954c64f5920f7ea15034d   (all 5, identical)
git_branch:  fix/p32d-classg-venue-repair                (all 5, identical)
tree_dirty_file_count: 606 (tasks 0,1,2,3) / 607 (task 4, b_minus_re)
```

`081b1f28` = "fix(cluster): s0b sbatch HEAD pin — ancestor check instead of strict equality"
(repo log). The 606-vs-607 dirty-file-count difference (task 4 only) is reported as a bare
fact; not diagnosed here.

**No `run_metadata.json` exists in this retrieved run** (three-valued: ABSENT). The per-task
`provenance_*.json` files are the run's own provenance record instead (existence-contract §7).

**Config verbatim** (from `s0a_score_output.json`'s echoed CLI state, reproduced identically by
the `--score-only` invocation below and by all 5 node directory names
`_iiib_sites2.2_nosmear`):

| field | S0-B run (this record) | row #287 certified instrument config | match? |
|---|---|---|---|
| `config` (venue) | `iiib` | `b0i` | **DEVIATION** (expected: iiib = production/CoR-P, b0i = mirror-cert venue; disclosed PA-HIER-31(g)) |
| `theta_sites` | `2.2` | `2.2` | match |
| `smear` / `smear_global_selection` | `off` / `False` | `False` | match |
| `theta_phi_divisor` | `off` | `on` | **DEVIATION** (row #332 item 2: "the mirror-cert-only divisor/zwin flags correctly left at their defaults (off)") |
| `theta_zwindow` / `z_window_k` | `off` / `1.0` | `on` / `zk4` (`z_window_k=4.0`) | **DEVIATION** (same row #332 disclosure) |
| `sky_cone_k` | `1.5` | `1.5` | match |
| `catalogue_leg_1d_mass_aware` | `off` | `off` | match |
| `b_half_width` | `0.033` (PA-HIER-31(a)/(d), re-derived) | n/a (not part of row #287's cert) | — |

Listed, not judged: three fields deviate from the row #287 (mirror b0i-configuration)
certified instrument. Row #332 (item 2, chair-derived) already discloses these as expected —
divisor/zwin are T1.2/T1.3-zwin driver flags with no counterpart in PA-HIER-31(g)'s own
registered CoR-P CLI list (`absolute_marginal` / `volume_deconv` / `fused` / `phi` /
`smear_global_selection=False` / `pdet_wbh_z_resolved=False` / `eddington_m=on` /
`sigma4d_mass_kernel=point` / `catalogue_numerator_survival_2d=off` / `--mass_filter_geometry
linear --mass_filter_k 1.5`) — not re-verified against that full CLI list here beyond the
fields the driver's own JSON echoes.

**Selection-table legitimacy** (row #332 item 2, reproduced here from the retrieved files
directly, not re-derived): `node_truth_iiib_sites2.2_nosmear/selection_tables_h_0_73.json`:
`sigma_phi = 980867125.674`, `sigma_4d = 375452610.321`, `r_Malm = 0.3827762`. These are
production-scale magnitudes (not stub/mirror-scale), consistent with row #332's claim.

---

## 2. g-score-null gate

Registered band (Research Graph 1 machinery table, `RESEARCH_GRAPH_1_PROPOSAL_20260901.md`
line 240 / `INFRA_AUTONOMOUS_RESEARCH_PROPOSAL_20260901.md` line 128, quoting rows #225/#251/
#287): *"g-score-null | mean score at truth-theta = 0 (first Bartlett identity) | abs Z <= 3"*.
At the production venue (S0-B), this coincides with the same secant-Z statistic that also
carries the B0-B scientific read (there is no separate mirror-style control at CoR-P — S0-B
*is* the decisive/only production measurement, PA-HIER-31(h)).

Evaluated on the no-BH primary channel (`ln_L_no_bh`, pooled N=1588):

| statistic | Z | \|Z\| ≤ 3 ? |
|---|---|---|
| `score_b_re` (b-axis secant) | **−5.274** | **OUTSIDE** |
| `score_lns` (PA-HIER-4 raw s-axis secant) | **−7.188** | **OUTSIDE** |
| `score_pahier33` (PA-HIER-33-ratified, Bartlett-corrected s-axis null; rows #278/#280) | **−7.101** | **OUTSIDE** |

**g-score-null: RED on both axes** (all three Z values exceed the |Z|≤3 band by a wide margin).
Per the machinery table's own consequence clause: *"red score-null -> STOP d-photoz-leverage,
reopen the instrument question as a fresh RULE (never auto-recertify)"* — quoted verbatim,
reported as the registered consequence text, not applied/adjudicated by this reader.

---

## 3. `score_b_re` secant (PA-HIER-31(d))

Registered form, quoted: *"score_b,i = [ lnL_i(+0.033,1) − lnL_i(−0.033,1) ] / 0.066"* (row
#324, `hier_s0_driver.py:1657`, `_B_RE_DENOM = 2.0*0.033 = 0.066`).

| channel | mean | SEM | Z | n_pooled |
|---|---|---|---|---|
| `ln_L_no_bh` (primary) | −0.6822 | 0.1293 | **−5.274** | 1588 |
| `ln_L_with_bh` (secondary) | −0.7412 | 0.1195 | **−6.204** | 1588 |

Source: `--score-only` output (`s0a_score_output.json`, `scores.ln_L_no_bh.score_b_re` /
`ln_L_with_bh.score_b_re`), reproduced bit-for-bit by an independent hand-recomputation from
the raw `event_likelihoods.csv` files (§6).

GATE ENG (prereg §3.4, ≥10% of events move ≥1e-6 relative vs truth) — the driver's own
`gate_eng` function iterates only the as-built `b_plus`/`b_minus` node names (not the `_re`
pair; those nodes were never requested), so it reports `pass=False`/`eng_available=True`/
`per_seed_fraction_moved=[]` for `b_plus`/`b_minus` — **absence of the as-built pair, not an
ENG failure of the registered `_re` pair.** Hand-computed GATE ENG on `b_plus_re`/`b_minus_re`
directly (same formula, `ENG_REL_THRESHOLD=1e-6`, `ENG_EVENT_FRACTION=0.10`):

| node | n common w/ truth | frac moved ≥1e-6 rel | pass (≥0.10)? |
|---|---|---|---|
| `b_plus_re` | 1588 | 0.5447 | **True** |
| `b_minus_re` | 1588 | 0.4861 | **True** |

---

## 4. `score_s` per the standing convention

Two objects exist under the name "score_s" in the record; both reported (PA-HIER-31(c)
relabelled the linear secant `score_s`→driver's `score_lns`; PA-HIER-33 is the ratified
correction of the *null offset*, not of `score_lns` itself):

**(a) `score_lns` — PA-HIER-4's raw ln-s-centred secant** (`[lnL_i(0,√2) − lnL_i(0,1/√2)]/ln2`):

| channel | mean | SEM | Z | n_pooled |
|---|---|---|---|---|
| `ln_L_no_bh` | −0.032661 | 0.004544 | **−7.188** | 1588 |
| `ln_L_with_bh` | −0.036824 | 0.005022 | **−7.333** | 1588 |

**(b) `score_pahier33` — the ratified statistic (rows #278/#280 RATIFIED; band structure
unchanged, §4.1/§4.5, restated in `score_lns` per PA-HIER-31(c))**: `score_lns −
Es_null^{(arm)}`, `Es_null^{(arm)} = (Δ²/6)·[−3⟨l'ᵢl''ᵢ⟩ − ⟨l'ᵢ³⟩]` computed from S0-B's own
three s-nodes (truth, s_plus, s_minus) at zero extra compute:

| channel | Es_null^(arm) (± bootstrap SD) | mean (corrected) | SEM | Z | n_pooled |
|---|---|---|---|---|---|
| `ln_L_no_bh` | −0.0003762 ± 0.0001637 | −0.032285 | 0.004547 | **−7.101** | 1588 |
| `ln_L_with_bh` | −0.0001236 ± 0.0001282 | −0.036701 | 0.005024 | **−7.306** | 1588 |

`score_s_available` for the OLDER PA-HIER-32(d) per-host closed-form correction (a distinct,
un-ratified-for-this-purpose object, restricted to catalogue-matched hosts) is also present in
the driver output: `n_pooled=76` (no-BH), Z=−12.397 — **not** the object this record's "§4.1
`score_s`/`score_lns`" band references (PA-HIER-33 explicitly leaves the per-host form
superseded for the band read); reported here only because the driver emits it under the key
`score_s`.

`σ_ln s` and `ln ŝ` (curvature leg, §5 below) come from the SAME raw data as (a)/(b); the
Es_null correction shifts the pooled mean by ~1/80th of its magnitude at production N and does
not change any 3σ disposition below.

---

## 5. B0-B disposition (§2.1(e) / PA-HIER-31(e))

Quoted: *"B0-B ≡ \|Z_b\| ≤ 3 and \|Z_lns\| ≤ 3 pooled (two-sided) ⇒ LEVER-DEAD-AT-N
(production); either > 3 ⇒ LEVER-LIVE — then B0-M (materiality): MIXED if \|b̂\| < 0.0165 (half
the 0.033 step) or \|ln ŝ\| < 0.5·ln√2 = 0.173; B0-P (power): σ_b < 0.0661 and σ_ln s < ln 2,
else UNPOWERED (no DEAD claim). Curvature leg: quadratic fit through the three b-nodes (truth,
b_plus_re, b_minus_re) → b̂ = −S′/S″, σ_b = 1/√(−S″); likewise in ln s."*

Curvature-leg point estimates (hand-derived, no-BH primary channel; §6 shows the with-BH
companion):

| axis | S′ (Σ score) | S″ (Σ curvature) | θ̂ | σ_θ |
|---|---|---|---|---|
| b | −1083.326 | −95317.073 | **b̂ = −0.011365** | **σ_b = 0.003239** |
| ln s | −51.866 | −44.513 | **ln ŝ = −1.165182** | **σ_ln s = 0.149885** |

### B0-B (LEVER-DEAD-AT-N vs LEVER-LIVE)

| condition | value | \|·\| ≤ 3 ? |
|---|---|---|
| \|Z_b\| ≤ 3 (Z_b = score_b_re, no-BH) | 5.274 | **OUTSIDE** |
| \|Z_lns\| ≤ 3 (score_pahier33, ratified) | 7.101 | **OUTSIDE** |

Both conditions fail ⇒ **B0-B = LEVER-LIVE.**

### B0-M (materiality; evaluated because B0-B = LEVER-LIVE)

| condition | value | threshold | INSIDE/OUTSIDE |
|---|---|---|---|
| \|b̂\| < 0.0165 | 0.011365 | 0.0165 | **INSIDE** (small on the b-axis) |
| \|ln ŝ\| < 0.173 | 1.165182 | 0.173 | **OUTSIDE** (large on the s-axis) |

Per the OR structure of the registered rule, one side small + one side large is the **MIXED**
disposition (materiality driven by the s-axis; the b-axis is small).

### B0-P (power)

| condition | value | threshold | INSIDE/OUTSIDE |
|---|---|---|---|
| σ_b < 0.0661 | 0.003239 | 0.0661 | **INSIDE** (powered) |
| σ_ln s < ln 2 (0.6931) | 0.149885 | 0.6931 | **INSIDE** (powered) |

Neither axis is UNPOWERED.

### Composite (mechanical, per §2.1(e)/PA-HIER-31(e); no other disposition named in the
registration for this state)

**LEVER-LIVE, MIXED (materiality: small on b, material on ln s), POWERED on both axes.** Both
axes are individually well-powered to detect the registered materiality thresholds, and the
observed |Z| exceeds 3 by 1.7×–2.4× on the two respective secants — this is a measurement
outcome inside the registered decision table, not an under-powered or ambiguous read by the
prereg's own POWER band. **REPORTED-ONLY cap applies unconditionally** (PA-HIER-28 item 9,
PA-HIER-31(e): *"All verdicts carry the REPORTED-ONLY cap ... Upgrade to CALIBRATED requires a
registered justification AND a positive control"*) — this is a mechanical disposition, not a
CALIBRATED claim.

### C-C identity check (item (b)/(d), instrument pass/fail, not a band)

For every C-C event (`L_cat_no_bh == 0` at truth, this arm's single node): `combined_no_bh`
must be bit-identical across all five θ-nodes. Hand-verified: **n=449 C-C events, max\|Δ\| =
0.000e+00 across truth/b_plus_re/b_minus_re/s_plus/s_minus. PASS.**

**Class-count discrepancy, disclosed not resolved.** PA-HIER-31(d) registers C-A∪C-B = 982,
C-C = 606 (`b3_pop_prediction.json:venues.iiib.n_matched`, "class is defined at h=0.73, this
arm's single node"). This reader's own single-node split on `L_cat_no_bh>0` at truth (this
run's own data) gives **C-A∪C-B = 1139, C-C = 449** (sum 1588 either way). The two counts do
not match at the same nominal definition; not reconciled here (per-class scores below are
REPORTED-ONLY / non-gating regardless, so the discrepancy does not change §5's gating
disposition, but is flagged as an open existence-contract fact for the author/decider).

**Per-class read (REPORTED, not gating; this reader's class split)** — no-BH channel:

| class | n | score_b_re mean | SEM | Z |
|---|---|---|---|---|
| C-A∪C-B (own split, n=1139) | 1139 | −0.951120 | 0.179726 | **−5.292** |
| C-C (n=449) | 449 | 0.000000 | 0.000000 | n/a (identically zero by the C-C identity, expected) |

| class | n | score_lns mean | SEM | Z |
|---|---|---|---|---|
| C-A∪C-B (own split, n=1139) | 1139 | −0.045536 | 0.006295 | **−7.234** |
| C-C (n=449) | 449 | 0.000000 | 0.000000 | n/a |

Per-z-bin read: **NOT COMPUTED** by this reader (would require reconstructing
`z_true = dist_to_redshift(d_L, 0.73)` from the CRB CSVs per event and the registered
`{0.075, 0.392, 0.559, 0.659, 0.753, 1.018}` edges — out of scope for this record's time
budget; flagged as an absent read, not attempted, not a negative result).

---

## 6. g-precision — hand recomputation (4 s.f.)

Independently re-derived from the raw `event_likelihoods.csv` files (h=0.73 rows only,
`combined_no_bh`/`combined_with_bh` → `ln`, `NaN` where ≤0), NOT the driver's cached JSON,
mirroring `read_event_ln_l`/`compute_scores`/`compute_es_null_arm` verbatim:

| quantity | numerator | denominator | Z (this reader) | Z (driver `--score-only`) | match |
|---|---|---|---|---|---|
| Z_b_re, no-BH | −0.6822 | 0.1293 | −5.274 | −5.274391509683036 | ✓ |
| Z_b_re, with-BH | −0.7412 | 0.1195 | −6.204 | −6.203632263701302 | ✓ |
| Z_lns (raw), no-BH | −0.03266 | 0.004544 | −7.188 | −7.188047761090759 | ✓ |
| Z_lns (raw), with-BH | −0.03682 | 0.005022 | −7.333 | −7.3325599596891795 | ✓ |
| Z_pahier33, no-BH | −0.03228 | 0.004547 | −7.101 | −7.100646322144426 | ✓ |
| Z_pahier33, with-BH | −0.03670 | 0.005024 | −7.306 | −7.305572701291795 | ✓ |

All six reproduce to the digit (the driver's `Z_pahier33` in the raw JSON was recomputed here
from `mean_p33/sem_total_p33` independently and matches to the printed precision). No
recomputation reveals a discrepancy with the driver's `--score-only` output.

---

## 7. Existence contract (three-valued)

| file/object | status |
|---|---|
| `PREREGISTRATION_HIER_HTHETA_20260826.md` | EXISTS |
| `BIAS_HISTORY_LEDGER.md` rows #278, #280, #287, #320, #323, #324, #332 | EXIST |
| `exec/m-s0b-production/LAUNCH_RECORD.md` | EXISTS |
| `exec/m-s0b-production/DRIVER_BNODE_BUILD_RECORD.md` | EXISTS |
| `exec/m-s0b-production/RETRIEVAL_RECORD.md` | EXISTS |
| `exec/b-pahier33-scorer/RECORD.md` | EXISTS (not re-read line-by-line; its scorer build is exercised live via `--score-only` above) |
| `retrieved/s0b_run_20260902/` (57 files, MD5-matched per RETRIEVAL_RECORD) | EXISTS |
| `retrieved/s0b_run_20260902/s0a_full_output.json` | EXISTS, but STALE — records only the `truth` node's `per_seed_summary` (pre-full-retrieval state); superseded by this record's own `--score-only` re-derivation over all 5 nodes |
| `retrieved/s0b_run_20260902/run_metadata.json` (any run_metadata file) | **ABSENT** — no such file in the retrieved tree; `provenance_*.json` × 5 are the run's provenance record instead |
| `fanout1_20260829/hier_s0_driver.py` (HEAD) | EXISTS, at commit `081b1f28f9d6c36c950954c64f5920f7ea15034d` per the S0-B job's own provenance stamps |
| `gates_bt.json` / `gates_bc.json` / `gates_33seed.DONE` (unrelated GATE-ACC addendum, RETRIEVAL_RECORD task 2) | **ABSENT** (not gating S0-B; reported for completeness since RETRIEVAL_RECORD checked them in the same pass) |
| A g-znorm normalisation-deviation field in this run's outputs | **ABSENT** — `selection_tables_h_0_73.json` carries `h, beta_G_phi, beta_Gbar_phi, sigma_phi, sigma_4d, r_Malm` but no explicit `dev`/normalisation-residual field, and no g-znorm identity formula for this quantity is registered in the sources read; **NOT COMPUTED** (not guessed) |

---

## 8. Cost (sacct elapsed, from RETRIEVAL_RECORD)

`sacct -j 6779532 --format=JobID,State,Elapsed,MaxRSS,TotalCPU -X` (verbatim from
RETRIEVAL_RECORD.md; `TotalCPU`/`MaxRSS` blank as reported — not re-queried by this reader):

| task | node | Elapsed |
|---|---|---|
| 6779532_0 | truth | 00:07:36 |
| 6779532_1 | b_plus_re | 00:07:31 |
| 6779532_2 | b_minus_re | 00:07:31 |
| 6779532_3 | s_plus | 00:07:27 |
| 6779532_4 | s_minus | 00:07:21 |

Sum of elapsed = 37.43 min (2246 s). `sbatch` requested `--cpus-per-task=16`
(`LAUNCH_RECORD.md:229`). Allocated-core-hours = Σelapsed(h) × 16 = 0.6239 h × 16 =
**9.98 CPU-h**. Row #332 (chair-derived, this reader's re-derivation not requested to redo it)
separately states *"actual total cost ≈ 2 CPU-h"* using the JSON's internally-measured
`wall_s` (338.09 s per node × 5 ≈ 0.47 h) as its basis, not the sbatch-allocated
core-hours. **The two figures (9.98 CPU-h allocated vs ≈2 CPU-h row #332 claims) are not
reconciled here** — reported as two distinct, sourced numbers for the decider, not merged.

---

## 9. Facts a decider needs

- **g-score-null is RED** on both axes at production (Z_b_re = −5.274, Z_lns/score_pahier33 =
  −7.101, both |Z|>3), triggering the machinery table's own stated consequence text ("STOP
  d-photoz-leverage, reopen the instrument question as a fresh RULE").
- **B0-B = LEVER-LIVE** (both |Z_b| and |Z_lns| exceed 3); **B0-M = MIXED** (b-axis small,
  |b̂|=0.0114<0.0165; s-axis material, |ln ŝ|=1.165≫0.173); **B0-P = POWERED on both axes**
  (σ_b=0.0032≪0.0661; σ_ln s=0.150≪0.693) — this is not an underpowered or ambiguous read by
  the registered power band.
- **The C-C instrument identity check PASSES exactly** (449 events, 0.000e+00 deviation across
  all 5 nodes) — the θ-inert dark-class events behave exactly as the instrument design requires.
- **GATE ENG passes** for both `b_plus_re`/`b_minus_re` (54.5%/48.6% moved, threshold 10%) and
  for `s_plus`/`s_minus` (50.8%/47.6%, per the driver's own output).
- **A class-count discrepancy exists and is unresolved**: this reader's C-A∪C-B/C-C split
  (1139/449) does not match PA-HIER-31(d)'s registered anchor (982/606) at the same nominal
  single-node definition. Does not affect the gating disposition (per-class reads are
  REPORTED-ONLY) but is an open existence-contract fact.
- **Per-z-bin reads were not attempted** by this reader (time-budget scope decision, disclosed
  above), so the B4/KW-Q1 localisation prediction (z-bin 1 share ≥ 0.50) is **not evaluated
  here** and remains open.
- **g-znorm could not be evaluated** — no normalisation-residual field or registered formula
  for it was found in the sources read for this run.
- **Cost is reported as two unreconciled figures**: 9.98 CPU-h (sbatch-allocated, this reader's
  computation from sacct Elapsed × 16 cores) vs ≈2 CPU-h (row #332's own chair-derived figure
  from the JSON's internal `wall_s`).
- **Provenance is clean and single-commit** across all 5 tasks (`081b1f28`, `fix/p32d-classg-
  venue-repair`), with 3 disclosed, already-flagged (row #332) config deviations from the row
  #287 mirror-cert config (venue, divisor, zwindow) — none of which the sources read describe
  as applicable to CoR-P in the first place.
- **`d-photoz-leverage`** (whether the theta-pull is real venue physics) is the author's
  ruling to make from the above; this record supplies the mechanical numbers only.
