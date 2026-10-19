# r-cone-loss — READ RECORD (revision 2, REAL mode) — DISJOINT READER

Node: r-cone-loss (Research Graph 1, Branch H, wave 3). Docket: launched under docket 2.2 after
design-gate GREEN (`DESIGN_GATE_rev2_computability.md`, verdict **GREEN**, 4 non-blocking
findings A–D, read in full before launch). Launch source: `REGISTRATION_DRAFT.md` §7, executed
**exactly** as written except `--out` redirected to this read's own output path (per task
instruction) and `--dry-run` omitted (REAL mode). Script not modified. Run once.

This record is **VERDICT-FREE**: it reports gate results, the registered intermediates, and the
raw TRUE/FALSE/N/A outcome of each disposition-table clause as computed by the frozen script. No
ruling, promotion, or recommendation is made here; disposition is reserved for the author /
fresh RULE d-cone-register per the draft.

## 1. Exact command run

From repo root (`/home/jasper/Repositories/darksiren-emri`):

```
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_loss_reads.py \
  --production-crb results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
  --production-run results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib \
  --replicate-run results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1 \
  --harness-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip --population 200 \
  --anchor-fleet-mker results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825 \
  --anchor-fleet-cmem results/campaign51_20260728/realistic_20260729/p3_b0_work \
  --sky-cone-k 1.5 --h-lo 0.725 --h-hi 0.735 --h-true 0.73 \
  --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
  --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_result_rev2_read.json
```

(`--dry-run` intentionally omitted — this is the REAL-mode statistic read. Every other flag/value
is byte-identical to REGISTRATION_DRAFT.md §7's launch block.)

**Exit code: 0.**

**Existence contract for inputs (checked before launch, all present on disk):** script
`cone_loss_reads.py`; production CRB `seed61000/prepared_cramer_rao_bounds.csv`; production run
dir `run_20260902_graph1_headrebaseline_iiib/` (with `GIT_COMMIT_AT_RUN.txt` and the log file);
replicate run dir `run_20260902_graph1_headrebaseline_joint_r1/`; harness root
`tree2_20260830/b8_cal_harness_work_s4_postflip/`; anchor fleets `p3_2d_fleet_20260825/` and
`p3_b0_work/`; output directory writable. No missing-input failure occurred.

**stderr:** 22 `RuntimeWarning: Mean of empty slice` lines from `np.nanmean(s_out)` /
`np.nanmean(s_out) - np.nanmean(s_in)` inside the harness-replicate per-universe loop (11
universes × 2 warning lines each — see §6 below for which universes). Non-fatal; script
proceeded and exited 0. This is the harness-replicate manifestation of design-gate rev2 Finding
A (stencil `s_e` can be NaN when a stencil endpoint likelihood is ≤0; here it additionally
produces an **all-NaN OUT class** for some harness universes, which differs from Finding A's
production-scope framing — disclosed, not adjudicated).

## 2. Gate results (verbatim from the script's own JSON, `gates` block)

| gate | result | key figures |
|---|---|---|
| G-1 catalogue md5 pin | **passed: true** | found `c52c13b5cab61f6b3f04bbe202550969` = expected |
| G-1 CRB md5 pin | **passed: true** | found `9a1f2a14384a9281c97ca3be312ddaab` = expected |
| G-1 git commit pin (production) | **passed: true** | `1ec9514dd1808c48b18c0792dce558e5bba0f116`, prefix `1ec9514d` matches |
| G-1 git commit pin (replicate) | **passed: true** | same commit |
| G-2 anchor MKER-6 (bc_900121/event 20) | **passed: true** | chord found `0.001674659860716462` vs expected `0.00167466` (tol 5e-10); radius found/expected `0.0014956979545757095` exact |
| G-2 anchor CMEM-A1 (bc_900101/event 0) | **passed: true** | chord found `0.01166569410071811` vs expected `0.0116656941007181`; radius found `0.035912194615445196` vs expected `0.0359121946154451` |
| G-2 aggregate | **g2_passed: true** | |
| G-3 join | **passed: true** | 1590 CRB rows, 1588 scored (gaps {1203,1356} implied), n_in_catalogue=76, n_out=10, n_in=66; P6 log line matched verbatim (`1D 66/76 hosts recovered/in-cat events seen (86.84211%)`), P6 numerator (66) = n_in (66) |
| G-4 KS clause (decisive) | **ks_passed: true** | D=0.06614822414302035, p=0.8715984091477792 (α=0.05) |
| G-4 envelope clause (binomial, rev.1 form) | **envelope_passed: true** | f_outside=10/76=0.13157894736842105; nearest envelope edge 0.134; two-sided binomial p=1.0 (α=0.05) |
| G-4 aggregate | **passed: true** | |
| g-population disclosure | **no `passed` field (gate has no implementable criterion — design-gate Finding C, confirmed unreachable in this run's JSON too)** | harness root: n_seed_S=67, n_seed_T=25, `--population` echoed 200 |
| **`gates.passed` (aggregate gating `main()`'s hard stop)** | **true** | run proceeded to REAL-mode statistic computation (not the `INSTRUMENT-DEFECT` early-exit branch) |

## 3. Census (production pool, at h_true = 0.73)

n_in_catalogue = 76, n_OUT = 10, n_IN = 66, f_OUT = 0.1316 (10/76).

## 4. Registered intermediates — primary statistic (§2 of the draft)

### 4a. 1D channel

| quantity | value |
|---|---|
| n_out | 10 |
| n_in | 66 |
| s̄_IN (s_bar_in) | 1.219636810462713 |
| SD_IN, MAD-scaled (robust SD, registered) | 0.8401264244982737 |
| SD_IN, plain sample SD (ddof=1, disclosed alongside) | 7.169716251090568 |
| SD ratio ρ = plain/MAD | 8.534092062837265 |
| 2-outlier sensitivity — top event | event_idx 889, s_e=52.23271813199195, \|dev\|=51.229516231811516 |
| 2-outlier sensitivity — 2nd event | event_idx 474, s_e=-24.43570199832752, \|dev\|=25.43890389850796 |
| I_1D (Fisher info) | 3256.0 |
| Δh_cone,1D | -0.00027312725025803014 |
| SE(Δh_cone,1D) | 0.000875578476007602 |
| Z_1D | -0.3119392010450229 |
| offset (mean_h − 0.73), 1D | -0.063 |
| φ_cone,1D | 0.004335353178698891 |
| T_mat | 0.008 (registered constant) |
| M_1D (= T_mat/SE) | 9.136816652320885 |

### 4b. 2D channel

| quantity | value |
|---|---|
| n_out | 10 |
| n_in | 66 |
| s̄_IN | 1.0024566177684917 |
| SD_IN, MAD-scaled | 0.7821358448052469 |
| SD_IN, plain sample SD | 5.89259188124013 |
| SD ratio ρ = plain/MAD | 7.5339749742672835 |
| 2-outlier sensitivity — top event | event_idx 889, s_e=46.4828469891915, \|dev\|=45.62154521811586 |
| 2-outlier sensitivity — 2nd event | event_idx 474, s_e=-8.503582690506832, \|dev\|=9.364884461582472 |
| I_2D | 2930.0 |
| Δh_cone,2D | -0.0003090817990897829 |
| SE(Δh_cone,2D) | 0.0009058356271907957 |
| Z_2D | -0.34121179363226917 |
| offset (mean_h − 0.73), 2D | -0.0641 |
| φ_cone,2D | 0.0048218689405582355 |
| T_mat | 0.008 |
| M_2D | 8.831624369655053 |

Note (raw fact, not interpretation): the two largest \|dev\| IN-class events (889, 474) are the
same event indices in both channels.

## 5. Leave-out cross-check (registered, §2)

| quantity | value |
|---|---|
| full sample: mean_h | 0.6669869414473403 |
| full sample: n_events_used | 1588 |
| full sample: n_events_floor_excluded | 0 |
| full sample: h_grid_n | 41 |
| leave-out-OUT: mean_h | 0.6620831623777804 |
| leave-out-OUT: n_events_used | 1578 |
| leave-out-OUT: n_events_floor_excluded | 0 |
| leave-out-OUT: h_grid_n | 41 |
| n_OUT_excluded | 10 |
| Δmean_h,leave-out | -0.004903779069559855 |
| `agrees_within_2SE_of_linear` (script's own flag) | **false** |

**Closure residual (leave-out minus linear, 1D channel, computed from the two registered
numbers above — not a separate script field):**

    residual = Δmean_h,leave-out − Δh_cone,1D
             = -0.004903779069559855 − (-0.00027312725025803014)
             = -0.004630651819301825

    2·SE(Δh_cone,1D) = 2 × 0.000875578476007602 = 0.001751156952015204

    |residual| = 0.004630651819301825  >  2·SE = 0.001751156952015204  →  TRUE

(This is exactly the comparison the script's `agrees_within_2SE_of_linear: false` /
`leave_out_disagrees_gt_2SE: true` flags already encode; shown here with the explicit arithmetic
per the task's request for the closure residual as a registered intermediate.)

## 6. Harness replicate (§2, "zero compute", reported only — not verdict-bearing per the draft)

67 post-flip S3 cell-S universes (`seed901000_S` … `seed901066_S`).

| aggregate quantity | value |
|---|---|
| n_universes | 67 |
| f_out_harn_mean | 0.14612213487727557 |
| f_out_harn_SE | 0.01309229320536408 |
| Δs_mean (s̄_OUT − s̄_IN, between-universe) | 0.3232258836302527 |
| Δs_SE | 0.20317562620660612 |
| Δs paired t | 1.5908693855904217 |
| Δs paired p | 0.11834358508112419 |

Raw data-quality facts (disclosed, not adjudicated): of 67 universes, **8** have n_out=0 (no OUT
event that universe, `f_out=0.0`, `s_bar_out`/`delta_s` reported `NaN` by construction); a
further **11** have n_out≥1 but still report `s_bar_out`/`delta_s` as `NaN` (all of that
universe's OUT-class stencil scores hit the `s_e` NaN guard — see §1 stderr note / design-gate
Finding A), for a total of **19/67 universes with `delta_s = NaN`**. The reported
`delta_s_mean`/`delta_s_SE`/paired-t/p above are computed over the remaining **48** universes
with a finite `delta_s` (the script's own aggregation; not re-derived independently here). Full
per-universe table is in `cone_result_rev2_read.json` → `harness_replicate.per_universe` (67
rows, not reproduced in full here for length; available verbatim at that path).

## 7. Disposition table — three-valued outcome of EACH clause, exactly as the draft's §4 rows
define them (VERDICT-FREE — raw script output only, no ruling/promotion/recommendation)

| draft §4 row | trigger (verbatim, 1D primary) | outcome |
|---|---|---|
| **IMMATERIAL-FLOOR-SHARE** | \|Δh_cone,1D\| < T_mat AND φ_1D < 0.2 AND M ≥ 3 | **TRUE** |
| **CONE-OWNS-FLOOR** | \|Z_1D\| > 3 AND φ_1D ≥ 0.5 AND M ≥ 3 | **FALSE** |
| **INTERMEDIATE-UNPOWERED** | SE(Δh_cone,1D) > T_mat/3 (M < 3) | **FALSE** |
| **INTERMEDIATE** | M ≥ 3 AND (\|Z\|>3 & 0.2≤φ<0.5; or \|Δh\|≥T_mat & φ<0.2; or 1D/2D disagree in disposition; or linear-vs-leave-out disagree > 2·SE) | **TRUE** |
| **INSTRUMENT / NO-READ** | G-1…G-4 red; g-population red | **N/A** — not evaluated as a named row in the script's `dispositions.rows` output; inferred **FALSE** from the run's own control flow: `gates["passed"] = true` (§2 above) means `main()` did not take the pre-statistic `SystemExit`/`INSTRUMENT-DEFECT` branch, so this row's trigger condition did not fire. (g-population itself carries no `passed` field at all — design-gate Finding C, confirmed unreachable — so this row's second clause has no code path in this run either.) |

Supporting sub-flags the script reports alongside the table (also raw, verdict-free):

- `disagree_1D_2D`: **false** (the code's `disagree_1d_2d` = materiality-flag/φ-band mismatch
  between channels — design-gate Finding D notes this is an operationalization, not a
  verbatim-registered formula)
- `leave_out_disagrees_gt_2SE`: **true** (per §5 above)

Both IMMATERIAL-FLOOR-SHARE and INTERMEDIATE evaluate TRUE simultaneously on this run's numbers,
as literally computed by the frozen script from the registered triggers — reported as-is; which
row governs (if the draft intends the rows to be mutually exclusive, evaluated in a stated
order, or resolved some other way) is not specified in the script's output and is not decided
here.

## 8. Output artifact

Full JSON (gates + both channel statistics + leave-out cross-check + full 67-row harness
replicate + dispositions) written to:

`results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_result_rev2_read.json`

(20651 bytes, `verdict: "SCORED"`, `dry_run: false`.) Script `cone_loss_reads.py` was not
modified. Run once, as instructed.
