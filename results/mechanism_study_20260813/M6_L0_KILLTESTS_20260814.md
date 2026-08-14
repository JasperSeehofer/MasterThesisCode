# M6-L0 registered kill tests — result

**Status: PRESENTED, NOT ADJUDICATED.** These are recomputed measurements against the
three tests registered in `PREREGISTRATION_M2PRIME_ABLATION.md` §2 "M6/M7 L0 obligations".
No branch call, repair, or register status change is made here; that is the author's.

**Data:** the 20 committed `*_h0p730_results_seeds*.json` files in this directory. The kill
tests use the 16-cell `S00`–`S33` dose grid (`dose_scales = [f_h, f_i]` with
`f ∈ {0, 0.25, 0.5, 1.0}`); `MEH`/`MEI`/`MN0`/`MN0X` are not part of the dose grid and are
not used. All quantities below are recomputed from raw per-seed `ln_post_1d` /
`ln_post_2d` vectors (41-point `h_grid`) and per-seed `map_1d`/`map_2d`/`post_sd_1d`/
`post_sd_2d` fields — the per-file `aggregate` block is never read, per the task's binding
instruction (`aggregate` includes `sum_dlog_gfrac_dh`, a different quantity: the
Σ ln(L2/L1) gfrac slope, not either channel's own log-posterior tilt).

Script: `m6_l0_killtests.py` (this directory). Output: `M6_L0_killtests_output.json`.

## Method

- **Slope at truth.** For each cell, each seed, each channel: central difference of
  `ln_post` on the two `h_grid` neighbours of `h_true = 0.730` (`h=0.725` and `h=0.735`,
  Δh = 0.01) — this is `venue_transfer._slope_at_truth` verbatim, applied directly to the
  channel log-posterior rather than to the precomputed gfrac slope. Averaged over seeds per
  cell.
- **Registered set for (i):** "all `f_h > 0` cells" is read as the full 12-cell set
  `{S10..S13, S20..S23, S30..S33}` (`dose_scales[0] > 0`, no restriction on `dose_scales[1]`)
  — the literal, most inclusive reading of the registered wording, and the reading most
  likely to expose non-invariance. This is a **falsification-first choice**: the task
  instructs that an ambiguous operationalization be read in the direction most favorable to
  killing M6. A narrower reading (fixed `f_i = 1.0`, varying `f_h` only — the 3-cell column
  `{S13, S23, S33}`) is reported separately below because it reproduces the commission's
  reference band closely; it is **not** used for the registered verdict.
- **Interior cells for (ii):** the 9 cells with `f_h > 0` and `f_i > 0`:
  `{S11, S12, S13, S21, S22, S23, S31, S32, S33}`.
- **Bias:** `mean over seeds of (map_{1d,2d} − h_true)` per cell. **σ²_post:** `median over
  seeds of post_sd_{1d,2d}`, squared.
- **α tilt:** `+1.036 × N/h` with `N = 982`, `h_true = 0.730` ⇒ `+1393.6 nats/h` (registered
  §2 constant, not recomputed — it is closed-form arithmetic, not data-dependent).
- Both channels (1D, 2D) are computed; the registered verdict is the **AND of both** (a test
  survives only if it survives in both channels) — a channel-favorable pick would not be the
  falsification-first choice.

## Results table

### (i) Tilt dose-invariance — registered set, 12 cells `{S1*, S2*, S3*}`

| cell | dose (f_h, f_i) | slope 1D (nats/h) | dev. from mean | slope 2D (nats/h) | dev. from mean |
|---|---|---:|---:|---:|---:|
| S10 | (0.25, 0.00) | 4484.8 | +37.9% | 4609.6 | +36.4% |
| S11 | (0.25, 0.25) | 3370.7 | +3.7% | 3492.3 | +3.4% |
| S12 | (0.25, 0.50) | 2956.2 | −9.1% | 3076.1 | −9.0% |
| S13 | (0.25, 1.00) | 2719.5 | −16.4% | 2840.4 | −16.0% |
| S20 | (0.50, 0.00) | 3840.7 | +18.1% | 3970.7 | +17.5% |
| S21 | (0.50, 0.25) | 3544.3 | +9.0% | 3672.0 | +8.7% |
| S22 | (0.50, 0.50) | 2745.0 | −15.6% | 2870.2 | −15.0% |
| S23 | (0.50, 1.00) | 2642.9 | −18.7% | 2769.6 | −18.0% |
| S30 | (1.00, 0.00) | 3682.9 | +13.3% | 3818.1 | +13.0% |
| S31 | (1.00, 0.25) | 3399.6 | +4.6% | 3531.1 | +4.5% |
| S32 | (1.00, 0.50) | 2962.8 | −8.9% | 3092.6 | −8.5% |
| S33 | (1.00, 1.00) | 2667.1 | −18.0% | 2798.7 | −17.2% |
| **grand mean** | — | **3251.4** | — | **3378.4** | — |
| **max \|dev\|** | — | **37.9%** | — | **36.4%** | — |

Tolerance: ±10%. Measured max deviation 37.9% (1D) / 36.4% (2D), both **> 4× the ±10%
tolerance band**. The spread is systematic, not noise: it tracks `f_i` monotonically within
each `f_h` row (highest at `f_i = 0`, lowest at `f_i = 1.0`) and only partially cancels
across `f_h`.

**Verdict (i): KILL** (both channels).

*Comparison-only, not the registered set:* the fixed-`f_i = 1.0` column
`{S13, S23, S33}` gives 1D range **2642.9–2719.5 nats/h** (spread 2.9%) and 2D range
**2769.6–2840.4 nats/h** (spread 2.5%) — this reproduces the commission's reported
2625–2720 nats/h "dose-invariant" band almost exactly. The commission's claim of
dose-invariance holds only under this narrower operationalization (varying host dose at
fixed, saturating impostor dose); it does not hold under the registered wording's more
natural full-grid reading, where the impostor-dose axis itself produces the dominant
~20–40% swing. The interior-9 set `{S1[1-3],S2[1-3],S3[1-3]}` (used for (ii)/(iii) context)
falls in between: 1D range 2642.9–3544.3 (spread 34%), 2D range 2769.6–3672.0 (spread 33%).

### (ii) Bias/σ²_post constancy — 9 interior cells

| cell | mean bias (1D) | median σ_post (1D) | ratio bias/σ²_post (1D) | ratio (2D) |
|---|---:|---:|---:|---:|
| S11 | 0.01267 | 0.002331 | 2330.6 | 2521.0 |
| S12 | 0.01200 | 0.002487 | 1940.3 | 2022.7 |
| S13 | 0.01400 | 0.002615 | 2047.1 | 2183.1 |
| S21 | 0.01900 | 0.002672 | 2660.8 | 2844.5 |
| S22 | 0.01600 | 0.002764 | 2093.7 | 2191.3 |
| S23 | 0.02365 | 0.003352 | 2104.5 | 2148.3 |
| S31 | 0.02200 | 0.002870 | 2670.4 | 2752.5 |
| S32 | 0.02333 | 0.003198 | 2281.4 | 2255.6 |
| S33 | 0.03967 | 0.004384 | 2064.1 | 2142.4 |
| **max/min** | — | — | **1.38** | **1.41** |

All 9 ratios are same-signed (positive) in both channels; max/min = 1.38 (1D) / 1.41 (2D),
both **within the factor-2 tolerance**, despite the underlying bias ranging over 3.1× (1D:
0.0120–0.0397) and σ_post over 1.9× across the 9 cells — the ratio's near-constancy is a
real, non-trivial pattern in the committed data.

**Verdict (ii): SURVIVE** (both channels).

### (iii) α-share of the measured total tilt

| quantity | value |
|---|---:|
| α tilt (closed-form, registered) | +1393.6 nats/h |
| measured total tilt, test (i) registered set, 1D | 3251.4 nats/h |
| measured total tilt, test (i) registered set, 2D | 3378.4 nats/h |
| **α-share, 1D** | **42.9%** |
| **α-share, 2D** | **41.3%** |
| target window (52.7% ± 5pp) | [47.7%, 57.7%] |

Both channel shares fall **below** the window (42.9% and 41.3% vs the 47.7% floor) using
the registered (i)-set total.

**Verdict (iii): KILL** (both channels).

*Comparison-only:* using the fixed-`f_i = 1.0`-column total from the alternative
operationalization above, α-share is 52.1% (1D) / 49.7% (2D) — inside or just below the
window, close to the commission's 53.3% measurement. This confirms the commission's
alpha-share number is tied to the same narrower cell selection that produces its
dose-invariance claim in (i); it is not reproduced under the registered set's own total.

## Commission and §2 reference values (comparison, not part of the verdicts)

| source | gradient at truth | claimed dose-invariance | α-share |
|---|---|---|---|
| commission (`results/commission_research_20260814/REPORT.md`) | 2625–2720 nats/h | yes, "across cells" | 52.7% predicted vs 53.3% measured |
| §2 prediction (`PREREGISTRATION_M2PRIME_ABLATION.md`) | 2738.8 nats/h total (α 1393.6 + missing-J 1345.4) | assumed | 52.7% predicted (missing-J 49.1%) |
| **this M6-L0 recomputation, registered (i)-set** | **3251.4 (1D) / 3378.4 (2D) nats/h grand mean, 37.9%/36.4% max deviation** | **no** | **42.9% (1D) / 41.3% (2D)** |
| this recomputation, fi=1.0-column only (comparison) | 2676.5 (1D) / 2802.9 (2D) nats/h, 2.9%/2.5% spread | yes (narrow set) | 52.1% (1D) / 49.7% (2D) |

The §2 prediction (2738.8 nats/h) sits inside the fi=1.0-column comparison band but well
below the registered-set grand mean (3251.4/3378.4); the commission's own reported band
(2625–2720) is reproduced almost exactly by the fi=1.0-column reading, not by the full
`f_h > 0` set the registered wording literally specifies.

## Overall

| test | verdict |
|---|---|
| (i) tilt dose-invariance | **KILL** |
| (ii) bias/σ²_post constancy | **SURVIVE** |
| (iii) α-share | **KILL** |

Per §2's own binding language ("Any failure kills the composite as stated"), two of the
three registered L0 kill tests fail under the registered operationalization; the third
(bias/σ²_post constancy) survives comfortably (max/min 1.38–1.41 ≪ 2). This is a mixed,
presented-not-adjudicated result: the composite's curvature-controlled bias account (ii)
holds up on committed data, but the tilt account's headline dose-invariance and α-share
claims — both of which the commission and §2 state in the form measured only on the
narrower fixed-high-impostor-dose column — do not hold once the impostor-dose axis is
allowed to vary across its full registered range. Which reading of "across all f_h > 0
cells" governs the composite's status is an author call, not resolved here.
