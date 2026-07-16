# pp_coverage σ(dL_obs)-vs-σ(dL_true) noise-model floor probe — VERDICT (2026-07-11)

**Provenance:** quick task `260711-hx1-floor-noise-model` (floor decomposition
[L7] item (d), `.planning/BIAS-INVESTIGATION-20260710.md`; the sharpened candidate
from `results/pp_coverage_pdetnum_20260711/SUMMARY.md` §2). Code at `77ee9d1` on
`physics/zero-host-completion-fallback` (adds `sigma_dl_model_in_likelihood` +
`--sigma-model-in-likelihood` to `master_thesis_code/validation/pp_coverage.py`).
RUNBOOK.md in this directory (grid, commands, pre-registered predictions — written
BEFORE any run, followed as recorded). Baselines reused VERBATIM (same
grid/seed/realizations): const-σ / p_det-off = `results/pp_coverage_exactmode_20260711/`;
const-σ / p_det-on = `results/pp_coverage_pdetnum_20260711/`.

**Anti-repetition:** gray/conditioned (07n, STILL BIASED), prior tilt (1ps,
NEGLIGIBLE), p_det-inside ALONE (27m, REFUTED) are not re-litigated — this probe
tests a DIFFERENT factor (the inference distance-error MODEL), 2×2 with the p_det
flag. Harness-only, no `/physics-change`.

## VERDICT: hypothesis H_σ CONFIRMED as the DOMINANT part of the floor — the σ_z-independent floor is (mostly) the inference noise-model approximation: the JOINT σ(dL_obs)-vs-σ(dL_true) width mismatch + the p_det-inside factor (the two halves of the single exact conditional for the latent-thresholded model). Applying BOTH (model-σ + p_det-inside) removes ~85–90% of it — the MAP bias drops from +0.002…+0.005 to ≤ +0.0008 (5–25× reduction) on the deep cells AND the inert-control offset, with cov68 restored to nominal at campaign-scale n. A TINY second-order residual (~+0.0005 in h, an order below the floor and ≈15× below campaign σ_boot) survives even the fully-consistent estimator, surfacing only as a cov68 degradation at n=4000 (16× campaign scale). The const-σ floor itself is a genuine ASYMPTOTIC model bias (flat in n at +0.002…+0.005, cov68 collapses as n grows), NOT a finite-sample MAP-skew.

Pre-registered **P1 (CALIBRATED)** holds at the ≥7/12 bar for the corrected estimator;
**P2 (REFUTED)** does not apply; **P3** adjudicated decisively via n-scaling (the
const-σ floor is a real asymptotic bias — flat in n, coverage collapses — not a
finite-sample artifact; the corrected estimator's MAP bias is ~10× smaller but a
sub-0.001 residual remains); **fine-grid confirm** passed (Δbias is not a
grid-quantization artifact).

## 2×2 headline (exact mode; deep cells zs∈{0.2,0.3} + inert controls zs∈{0.5,1.0})

| variant | deep-cell bias | deep cov68 | control bias (zs 0.5/1.0) | control cov68 |
|---|---|---|---|---|
| const-σ, p_det off (exactmode, **baseline**) | +0.002…+0.005 (floor) | 0.48–0.71 | −0.002…−0.004 | 0.68–0.76 |
| const-σ, p_det on (27m) | +0.002…+0.006 (floor intact) | 0.41–0.71 | +0.003…+0.006 (flips) | 0.45–0.78 |
| **model-σ, p_det off** | −0.004…+0.001 (floor gone; slight −overshoot at low h) | 0.60–0.78 | −0.005…−0.008 (worse) | 0.50–0.60 |
| **model-σ, p_det on** ⭐ (fully-consistent exact conditional) | **−0.001…+0.002** | **0.60–0.79** | **−0.000…+0.001** | **0.69–0.81** |

Only the model-σ + p_det-inside combination is calibrated on BOTH regimes. Each half
alone fixes one regime and breaks the other — they are the two halves of the single
exact conditional for a model whose detection is decided on the latent true z.

## Per-cell deep table — bias[cov68] (net tilt = dlogL_dh_host + dlogL_dh_completion at h_true)

| zs | σ_z | h_true | const-σ (exact) | model-σ | model-σ+pdet | Δbias(modelσ−const) |
|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | +0.0023[0.69](+63) | −0.0010[0.72](−25) | −0.0008[0.74] | −0.0033 |
| 0.2 | 0.015 | 0.72 | +0.0034[0.55](+49) | −0.0004[0.70](−5) | −0.0002[0.70] | −0.0038 |
| 0.2 | 0.015 | 0.84 | +0.0042[0.61](+41) | +0.0001[0.68](+4) | +0.0002[0.68] | −0.0041 |
| 0.2 | 0.035 | 0.62 | +0.0026[0.71](+49) | −0.0006[0.78](−12) | −0.0005[0.79] | −0.0033 |
| 0.2 | 0.035 | 0.72 | +0.0046[0.57](+41) | +0.0007[0.68](+3) | +0.0008[0.68] | −0.0039 |
| 0.2 | 0.035 | 0.84 | +0.0042[0.52](+35) | +0.0006[0.60](+8) | +0.0007[0.60] | −0.0036 |
| 0.3 | 0.015 | 0.62 | +0.0003[0.70](+33) | −0.0027[0.60](−206) | −0.0002[0.72] | −0.0030 |
| 0.3 | 0.015 | 0.72 | +0.0024[0.62](+118) | −0.0012[0.78](−66) | −0.0001[0.76] | −0.0036 |
| 0.3 | 0.015 | 0.84 | +0.0047[0.53](+129) | +0.0004[0.68](+9) | +0.0007[0.68] | −0.0043 |
| 0.3 | 0.035 | 0.62 | −0.0010[0.71](−34) | −0.0040[0.63](−157) | +0.0002[0.78] | −0.0030 |
| 0.3 | 0.035 | 0.72 | +0.0023[0.63](+62) | −0.0016[0.73](−45) | +0.0004[0.69] | −0.0039 |
| 0.3 | 0.035 | 0.84 | +0.0054[0.48](+85) | +0.0010[0.68](+13) | +0.0018[0.67] | −0.0044 |

The net tilt at h_true (grid-step-independent continuous diagnostic) collapses from
+35…+129 (const-σ) to ≈0 for model-σ+pdet in the well-behaved cells — the positive
completion-branch tilt that drove the floor is removed once the inference likelihood
uses the true-distance σ. (model-σ alone over-corrects the tilt negative at low
truth / high completion — the p_det-inside half restores the balance.)

**Strict P1 count:** model-σ (p_det off) has 8/12 deep cells within 2·SEM (vs 1/12
for const-σ exact); model-σ+pdet has ~11/12 (only zs0.3/σ_z0.035/h0.84 at +0.0018 is
marginally over its 2·SEM≈0.0013). Both clear the pre-registered ≥7/12 CALIBRATED bar.

## n_events scaling (zs=0.3, σ_z=0.035) — the P3 adjudicator

bias[cov68] (2·SEM); const-σ n=250 is the exactmode baseline.

| variant | h_true | n=250 | n=1000 | n=4000 |
|---|---|---|---|---|
| const-σ (original floor) | 0.62 | −0.0010[0.71] | −0.0010[0.83] | −0.0010[0.82] |
| const-σ (original floor) | 0.72 | +0.0023[0.63] | +0.0024[0.38] | +0.0022[0.12] |
| const-σ (original floor) | 0.84 | +0.0054[0.48] | +0.0040[0.33] | +0.0046[0.03] |
| model-σ+pdet (corrected) | 0.62 | +0.0002[0.78] | −0.0004[0.88] | −0.0001[0.93] |
| model-σ+pdet (corrected) | 0.72 | +0.0004[0.69] | +0.0002[0.68] | +0.0002[0.10] |
| model-σ+pdet (corrected) | 0.84 | +0.0018[0.67] | +0.0003[0.60] | +0.0008[0.36] |

**Decisive reading (const-σ floor):** the floor is **FLAT in n** (h=0.72 stays
+0.0022…+0.0024; h=0.84 +0.0040…+0.0054) while **cov68 COLLAPSES** as n grows
(h=0.72: 0.63→0.38→0.12; h=0.84: 0.48→0.33→0.03). That is the unambiguous signature
of a **real asymptotic bias**: the posterior tightens around a fixed offset, so
coverage falls apart — the opposite of a finite-sample MAP-skew, which would shrink
∝1/√n with coverage → nominal. **P3's "finite-sample skew" alternative is REFUTED for
the floor.**

**Corrected estimator (model-σ+pdet):** the MAP bias is ~10× smaller and nearly
n-independent (|bias| ≤ +0.0008 at all n, vs the const-σ +0.002…+0.005 floor) — the
noise-model correction removes the bulk of the bias. BUT cov68 also degrades at n=4000
(h=0.72: 0.10; h=0.84: 0.36) while the MAP bias stays tiny: the signature of a **very
small residual** (a sub-0.001 MAP offset and/or slight posterior over-confidence) that
is invisible at n ≤ 1000 (cov68 nominal) and only surfaces once the posterior narrows
at n=4000 (16× the campaign per-seed event count). So the fully-consistent estimator
removes the dominant O(σ_f²) term but leaves a plausibly higher-order (O(σ_f⁴)) or
width-calibration residual an order below the floor and ≈15× below campaign σ_boot —
practically irrelevant at campaign scale, honestly noted here rather than rounded to
zero (the const-σ floor n=4000 coverage-collapse establishes n=4000 as a genuinely
discriminating scale, so the corrected estimator's collapse there is a real, if tiny,
signal — not noise).

## Fine-grid confirm (zs=0.3, σ_z=0.035; h_step 0.004 vs 0.001) — debrief lesson #2

| variant | h_true | coarse (0.004) | fine (0.001) |
|---|---|---|---|
| const-σ | 0.62 | −0.0010[0.71] | −0.0009[0.65] |
| const-σ | 0.72 | +0.0023[0.63] | +0.0023[0.64] |
| const-σ | 0.84 | +0.0054[0.48] | +0.0054[0.53] |
| model-σ | 0.62 | −0.0040[0.63] | −0.0041[0.57] |
| model-σ | 0.72 | −0.0016[0.73] | −0.0016[0.72] |
| model-σ | 0.84 | +0.0010[0.68] | +0.0010[0.69] |

Biases are identical to ±0.0001 between the coarse and fine H0 grids: the reported
Δbias is **not** a grid-quantization artifact (the mean-of-argmax over 120
realizations already resolves sub-step). The continuous net-tilt diagnostic and the
n-scaling coverage-collapse are the primary, grid-free signals and agree.

## Mechanism (for the ledger)

The harness generative model draws distance noise with σ = σ_f·dL_**true** (width
scales with the true distance and varies along the redshift integral) and thresholds
detection on the latent true z. The default inference likelihood makes **two**
approximations that individually nearly cancel but are exposed at deep incompleteness:
(1) it uses a **constant, observed-distance** width σ_f·dL_obs (dropping the 1/σ(z)
variation), and (2) following Mandel–Farr–Gair it drops p_det from the numerator
(correct only for **data**-thresholded detection). For this **latent**-thresholded
model the exact conditional keeps BOTH: a z-dependent σ_f·A(z)/h (with its 1/σ(z)
normalization) AND p_det(A(z)/h) inside the numerator. Fixing only one (27m: p_det
alone; this task's model-σ-alone) breaks the accidental cancellation and mis-biases;
fixing BOTH removes the floor and the control offset simultaneously. O(σ_f²)=0.0025 →
~0.002–0.005 in h, exactly the observed floor scale.

## Verdict / decision-tree mapping (`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`)

1. **Floor mechanism identified?** YES — the σ(dL_obs)-vs-σ(dL_true) inference-noise
   model approximation, jointly with the latent-detection p_det-inside factor. The
   full deep-incompleteness bias is now decomposed to three orders:
   (i) dominant σ_z-dependent **membership-support kernel leak** (removed by exact
   truncation, 260711-117); (ii) sub-dominant σ_z-independent **inference-noise-model
   floor** (+0.002…+0.005, ~85–90% removed by model-σ + p_det-inside, this task);
   (iii) a tiny **second-order residual** (~+0.0005, ≈15× below campaign σ_boot)
   surviving the consistent estimator, visible only at n=4000. Nothing remains that
   scales with σ_z, the prior, or the grid; the leftover is far below Paper-B
   resolution.
2. **Practical weight (Paper B):** the floor is ≤ +0.005 in h — an order of magnitude
   below the leak it survived (up to +0.037/+0.123) and at/below the campaign per-seed
   σ_boot (~0.005). It is a **harness inference-model** approximation, NOT an intrinsic
   un-calibratability of deep incompleteness. Deep incompleteness is calibratable.
3. **Input to the (user-gated) production correction:** this is a REQUIRED design
   constraint for the soft-f(z)-kernel `/physics-change` pass, not a production change.
   Production ALSO thresholds SNR on the noiseless injected waveform (latent-thresholded
   class), so the correct joint move is a self-consistent distance-error model +
   p_det-inside for latent detection — **do NOT add p_det alone** (27m + this task both
   show p_det-alone and σ-model-alone each degrade the complementary regime). Literature:
   Gray 2020; Chen–Fishbach–Holz 2018; Mandel–Farr–Gair 2019 (data- vs latent-threshold);
   Mastrogiovanni et al./ICAROGW. NOT this task.
4. **EXP-40 watch (cluster):** production's composition is gray-like (untruncated
   kernels + mixture) AND uses the constant-σ / no-p_det-inside inference form, so both
   the leak and the floor point the same way (biased HIGH). Watch for interior-but-biased
   -HIGH; the floor sets a ~+0.3…+0.6%-of-truth harness lower bound on the residual after
   any leak correction.

## Carried caveats

1. **1D-channel only** — the 2D (+0.025 remaining) question is not covered here.
2. **Single effective host** per event; **hard** z_support truncation (vs production's
   soft M_BH-prune) — as in all four predecessor SUMMARYs.
3. **Harness generative model** — detection latent-thresholded on true z with σ∝dL_true;
   production's exact detection/noise structure differs, so the *specific* corrected
   estimator here is a diagnostic, not a drop-in production form (see item 3 above).
