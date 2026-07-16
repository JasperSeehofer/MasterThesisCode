# pp_coverage shallow-venue N-4 probe — VERDICT (2026-07-11)

**Provenance:** quick task `260711-iic-shallow-venue-n4` (handoff item **N-4**, the
separate shallow-venue 1D residual; `.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`).
Code at `baeaa1c` on `physics/zero-host-completion-fallback` (adds tunable detection
horizon `d50_gpc`/`w_pdet_gpc` + `--d50-gpc`/`--w-pdet-gpc` to
`master_thesis_code/validation/pp_coverage.py`). RUNBOOK.md in this directory (grid,
commands, pre-registered predictions — written BEFORE the depth-sweep runs). Harness-only,
no `/physics-change`. **This is a DIFFERENT regime from the deep-incompleteness floor**
(260711-hx1): seed600 is comp_frac ≈ 0.4% (deep mechanisms ~absent), z_median 0.046.

## VERDICT: P-B CONFIRMED, localized by Set B — the shallow-venue 1D high bias is ESTIMATOR-INTRINSIC and specifically a large-σ_z/z-at-low-z effect. The calibrated volume-kernel estimator (which is unbiased at the commission depth z_med 0.28) develops a strong POSITIVE bias as the venue shallows — +0.011 at z_med 0.056, +0.030 at z_med 0.044 (seed600 depth) — but ONLY at σ_z=0.035; at σ_z ≤ 0.015 it stays calibrated. Mechanism: at z_med 0.044 with σ_z=0.035, σ_z/z ≈ 0.8, so the host-z kernel N(z; z_gal, σ_z) truncates against the physical z ≥ 0 boundary and the volume/Eddington-in-z correction (derived for an UN-truncated kernel) no longer cancels.

Pre-registered **P-A (calibrated-stays)** is REFUTED; **P-B (shallow-bias)** holds and
Set B pins it to the σ_z/z ratio (not depth alone).

## (a) Depth ladder — calibrated volume kernel, NO truncation (comp_frac 0), σ_z=0.035

| d50 [Gpc] | z_median | σ_z/z_med | h=0.62 bias[cov68] | h=0.73 bias[cov68] (2·SEM) | h=0.84 bias[cov68] |
|---|---|---|---|---|---|
| 1.85 (commission) | 0.280 | 0.12 | −0.0030[0.76] | **−0.0019[0.68]** (0.0012) | −0.0024[0.68] |
| 1.0 | 0.168 | 0.21 | −0.0033[0.68] | −0.0022[0.69] (0.0019) | −0.0036[0.68] |
| 0.6 | 0.107 | 0.33 | −0.0017[0.62] | −0.0013[0.70] (0.0028) | −0.0041[0.62] |
| 0.4 | 0.074 | 0.47 | +0.0017[0.74] | +0.0022[0.67] (0.0041) | −0.0039[0.67] |
| 0.30 | 0.056 | 0.62 | +0.0113[0.70] | +0.0108[0.66] (0.0051) | −0.0005[0.76] |
| **0.23 (seed600)** | **0.044** | **0.80** | +0.0348[0.50] | **+0.0303[0.57]** (0.0064) | +0.0035[0.79] |

The estimator crosses from the calibrated deep reference (−0.002) through zero near
z_med 0.074 to a large positive bias at seed600 depth. At the seed600-matched rung
(z_med 0.044) the harness bias is +0.030 in h — the SAME SIGN as and ~2× seed600's
raw +0.0132 (larger because the harness's flat σ_z=0.035 likely exceeds seed600's
effective low-z scatter, and seed600's informative events counterweight — see (b)).
Note the h_true dependence: the bias is largest at low h_true (h=0.62: +0.035) and
smallest at high h_true (h=0.84: +0.004), because higher h_true maps the same z to a
smaller d_L and pushes the population slightly deeper.

## (b) σ_z at the shallow rung (d50=0.23, z_med 0.044) — the localizer

| σ_z | σ_z/z_med | h=0.62 | h=0.73 (2·SEM) | h=0.84 |
|---|---|---|---|---|
| 0.005 | 0.11 | −0.0040[0.62] | **−0.0035[0.53]** (0.0010) | −0.0042[0.66] |
| 0.015 | 0.34 | −0.0024[0.62] | −0.0020[0.69] (0.0028) | −0.0051[0.63] |
| 0.035 | 0.80 | +0.0348[0.50] | **+0.0303[0.57]** (0.0064) | +0.0035[0.79] |

**The shallow high bias VANISHES at small σ_z** (σ_z=0.005 → −0.0035, calibrated like
the deep venue) and appears only at σ_z=0.035 → the bias is driven by the σ_z/z ratio,
not by depth per se. This is the σ_z/z-at-low-z truncated-kernel Eddington effect: the
volume-weighted host-z kernel calibration (which fixes the bare-Gaussian Eddington-in-z
bias at deep z — the commission finding) is itself derived assuming the kernel integrates
over an un-truncated z line; when σ_z/z ~ 1 the kernel hits the z ≥ 0 boundary, the
asymmetric truncation interacts with the rising w_pop(z) ∝ dV_c/dz weight, and a residual
high bias survives the volume correction.

## (b) Jackknife/influence on the seed600 run_live per-event JSONs (on disk, no re-eval)

`results/pv_correction_test_20260703/run_live/simulations/posteriors/` (3355 events,
13 all-zero excluded → 3342; 17-pt grid). Faithful reconstruction via the production
`apply_strategy(PHYSICS_FLOOR)` + `combine_log_space` (Σ log L_i, D_h ignored):
reproduces the committed grid-MAP **0.745** and posterior **mean 0.74320 → residual
+0.01320** (= ledger raw). Parabolic-peak residual +0.01331.

- **Leave-one-out influence on the posterior mean:** Gini(|influence|)=0.65; signed
  influences near-cancel (Σ+ = +0.087, Σ− = −0.088); 50% of Σ|influence| from the top
  251 events (7.5% of the sample), 90% from the top 1728 (51.7%). No dominating subset.
- **Per-event tilt d(logL_i)/dh at truth h=0.73:** Σ = +523.8 (net rightward pull, ⇒
  MAP > 0.73); 61.6% of events tilt high; median per-event tilt +0.34. The net +524 is a
  small imbalance of large opposing sums (Σ+ = +3630, Σ− = −3107).
- **Trimming is decisive:** removing the highest-|tilt| events does NOT shrink the
  residual — it GROWS it (drop top-10 → +0.0157, top-100 → +0.0430, central-90% → +0.0737).
  The high-|tilt| (most informative) events are a net-NEGATIVE counterweight; the
  systematic high-drift lives in the shallow BULK.

**(b) verdict:** the seed600 +0.0132 is **broad / systematic, NOT a heavy-tailed outlier
subset** — every event carries a small positive drift, partially offset by the informative
events. This is exactly the footprint the (a) mechanism predicts (a per-event, depth-driven
Eddington bias spread across the shallow population).

## Synthesis / decision-tree mapping

1. **N-4 answered:** the separate shallow-venue 1D residual is (a) **reproducible as an
   estimator-intrinsic bias** in a venue-matched harness — a large-σ_z/z, low-z
   truncated-volume-kernel Eddington effect — and (b) **broad/systematic** in the real
   seed600 data (per-event, not outlier-driven), consistent with that mechanism.
2. **Load-bearing caveat (what closes it):** whether this FULLY explains seed600's
   +0.0132 hinges on **seed600's effective redshift-uncertainty at z ≈ 0.046**. The harness
   effect needs σ_z/z ~ O(1); if seed600's z-errors are small spec-z (σ_z/z << 1) the
   mechanism does NOT apply and the residual is something else. **Next input:** the
   seed600 catalogue/CRB redshift-error model at low z (checkable; not this task). If it is
   large-fractional (photo-z-like), the shallow bias is (partly) this Eddington effect.
3. **Production correction (user-gated /physics-change, NOT this task):** if confirmed, the
   fix is a low-z-safe host-z kernel — a photo-z-marginalized / truncation-aware volume
   weight (the same soft-membership family flagged by the deep probe 260711-117), so a
   single production change (a properly z≥0-truncation-normalized volume kernel) addresses
   BOTH the deep membership-support leak and the shallow σ_z/z Eddington effect. Literature:
   the volume/Eddington correction (commission d2); Gray 2020; Chen–Fishbach–Holz 2018.
4. **Systematic-vs-scatter (cross-seed):** genuinely needs the multi-seed campaign — do NOT
   force it locally (handoff constraint). (a)+(b) establish the mechanism and that it is
   broad within seed600; the campaign establishes whether the OFFSET reproduces across seeds.

## ADDENDUM 2026-07-12 — load-bearing caveat CLOSED: seed600's low-z σ_z IS photo-z (σ_z/z ~ O(1))

The item-2 load-bearing input ("seed600's effective redshift-uncertainty at z ≈ 0.046") is now
measured directly on the reduced GLADE+ catalogue seed600 evaluated (no re-eval; inline column
inspection), and cross-checked against the likelihood code. **The N-4 mechanism applies to seed600.**

**Measurement** (reduced catalogue, z-shell 0.03–0.06 around z_med 0.046, n = 767 552 galaxies):

| population | fraction | σ_z median | σ_z/z median |
|---|---|---|---|
| all in shell | — | 0.0344 | **0.65** |
| flag=1 photometric | **0.897** | 0.0345 | 0.669 |
| flag=3 spectroscopic | 0.103 | 0.0014 | 0.033 |

σ_z ≈ 0.0344 is an almost exact match to the harness's flat σ_z = 0.035 rung (Set B) that produced
+0.030. σ_z/z ≈ 0.65 is squarely in the O(1) regime the mechanism requires; the spec-z minority sits
at σ_z/z ≈ 0.033 (the "vanishes" regime) and is the calibrated counterweight the jackknife saw.

**Code trace (airtight):** the likelihood host-z kernel width IS the catalogue σ_z —
`bayesian_statistics.py:2243` `norm(loc=host_z, scale=host_z_error_eff)`,
`host_z_error_eff = sqrt(catalogue_σ_z² + σ_z_pv²)` — and `:2234-2239` applies the `[PHYSICS]`
z ≥ 0 clamp explicitly "for low-z photo-z hosts (z_g < 4·σ_z)". At z_g = 0.046, 4·σ_z = 0.14 > z_g,
so the clamp is ACTIVE for these hosts: the host-z kernel truncates against z ≥ 0 and the
un-truncated-derived volume/Eddington correction stops cancelling — exactly the (a)/(b) mechanism.

**Verdict:** the shallow +0.0132 IS (substantially) the σ_z/z-at-low-z truncated-volume-kernel
Eddington effect. N-4 is now "attributed to seed600," not just "reproduced." Remaining open item is
cross-seed systematic-vs-scatter, which needs the multi-seed campaign (do NOT force locally).

## Carried caveats

1. **1D-channel only** (the 2D +0.025 is N-5, not covered here).
2. **Harness venue-match is approximate:** d50=0.23/w=0.037 matches seed600's z_median
   (0.044) but is a smooth Malmquist, not seed600's exact selection/σ_z distribution; the
   +0.030 harness magnitude is not a seed600 prediction, only a same-sign, same-scale
   demonstration of the mechanism.
3. **σ_z is flat** in the harness (vs a per-galaxy z-error model in production).
