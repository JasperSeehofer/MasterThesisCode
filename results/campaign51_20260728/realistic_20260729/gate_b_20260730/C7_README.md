# C7 — Gate B measurement of the host-z `volume_deconv` kernel (2026-07-30)

Target: **claim C7** of `../CLAIM_2D_BIAS_20260730.md` (RUNBOOK_NEXT_SESSION_6 §4
item 3). Everything here is **[LOCAL, VERIFIED]** — no cluster access, no jobs,
read-only w.r.t. `master_thesis_code/`.

## Verdict

**C7 is CONFIRMED as the mechanism for C5 (the in-catalogue rail), and its
quoted magnitude is UNDERSTATED by ~1.5x.**
**C7 is NOT a candidate for the 2D-minus-1D difference (C3/C4): the host-z
kernel is channel-common** (`prior_num` multiplies both `numerator_without_bh_mass`
:4240-4245 and `numerator_with_bh_mass` :4379-4384) — consistent with "the z leg
(channel-common)" already on the exoneration list. Nothing exonerated was
re-opened.

## Files

| file | what it is |
|---|---|
| `c7_orient.py` / `c7_orient_results.json` | Orientation: which mixture leg carries the observed in-cat rail. Validates the diagnostics CSV against `posteriors/h_0_73.json` (max rel diff 2.9e-13) and the mixture identity `p = w_G L_cat + (1-w_G) L_comp` (3.9e-13). |
| `c7_kernel_measure.py` / `c7_kernel_measure_results.json` | **The measurement.** Drives the code's own numerator kernel for the 76 in-cat hosts, `volume_deconv` vs `point`, over a fine h grid extended to 2.4. Legs A (pure kernel), A' (indicative catalogue sigma_z), B (with realization scatter), S (sigma_z -> 0 scaling gate). |
| `c7_vs_production.py` / `c7_vs_production_results.json` | Confronts the prediction with the delivered `real_r1` per-event catalogue leg, and inverts every event's delivered peak into an effective host redshift (dark-class direction test). |
| `c7_checks.py` | Correctness/robustness: fixed_quad parity, h-invariance of the kernel, quadrature-order robustness, analytic point-kernel limit. |

Run with `PYTHONPATH=<repo>:<this dir> .venv/bin/python <script>` from the repo root.

## What was measured

For each in-cat host the code's numerator was rebuilt exactly
(`bayesian_statistics.py:4099-4245`):

```
volume_deconv: N_g(h) = INT_{z(dL-4s)}^{z(dL+4s)} p_GW(dL(z;h)/dL_hat)
                          * N(z; z_obs, sigma_z) * w_pop(z;h) / Z_g(h) dz
point:         N_g(h) = p_GW(dL(z_obs;h)/dL_hat)
```

In `absolute_marginal` the per-event catalogue leg is `(SUM_ball w_g N_g)/Sigma_glob(h)`
with `Sigma_glob` event-independent **and identical for both kernels**
(`generator_marginal` joins the volume_deconv set for the denominator/Z_g machinery
only, :4125-4135), so `argmax_h N_g` *is* the catalogue-leg argmax and the kernel
difference is the whole kernel-induced shift.

## Results

### 0. Orientation — the rail lives in the catalogue leg

| leg | class | median argmax | frac at 0.86 |
|---|---|---|---|
| combined `p_i` | IN-CAT | 0.860 | 0.579 (= claim's 44/76) |
| **`L_cat` alone** | IN-CAT | **0.860** | **0.892 (66/74)** |
| `L_comp` alone | IN-CAT | 0.860 | 1.000 |
| `L_cat` alone | DARK | 0.600 | 0.016 |

The catalogue leg carries a median 96.3% of the in-cat mixture at h=0.73.
(The completion leg also rises for these nearby events — a completeness effect,
not a kernel effect — but it is the minority leg.)

### 1. The kernel shift, measured (Leg A, z_obs = z_true)

Point kernel peaks at **exactly** h_true = 0.730000 for all 76 hosts (analytic:
`h = dist(z_obs,h=1)/dL_hat`). The `volume_deconv` peak:

| sigma_z/z | measured peak h | frac shift | claim's mode formula `[1+sqrt(1+8e^2)]/2` | claim's `2e^2` |
|---|---|---|---|---|
| 0.10 | 0.7511 | +0.0289 | +0.0196 | +0.020 |
| 0.15 | 0.7761 | +0.0631 | +0.0431 | +0.045 |
| 0.25 | 0.8476 | **+0.1611** | +0.1124 | +0.125 |
| 0.35 | 0.9390 | **+0.2863** | +0.2036 | +0.245 |
| 0.49 | 1.0864 | **+0.4882** | +0.3545 | +0.480 |
| 0.80 | 1.4528 | +0.9901 | +0.7369 | +1.280 |

**The correct closed form is the mode formula with 12, not 8:**

```
h_eff / h_true  =  [1 + sqrt(1 + 12 (sigma_z/z)^2)] / 2      ->   3 (sigma_z/z)^2  for small sigma_z
```

Reproduces the measurement to <1% up to sigma_z/z = 0.5 (16.14 vs 16.11%,
28.58 vs 28.63%, 48.50 vs 48.82%). Derivation of the 3 (verified by Leg S):
`d ln N_g/dz = -(z-z_g)/sigma^2 + 2/z (volume weight w_pop ∝ z^2) + 1/z (the
numerator window's z-width ∝ z, because the GW kernel has fixed FRACTIONAL
distance width) - O(1)`. The point kernel has no window and therefore no `+1/z`.

**Rail threshold: sigma_z/z > 0.256 puts a host's peak above the 0.86 prior edge.**

### 2. sigma_z -> 0 gate (Leg S) — PASSES

At high quadrature order (200-2800 nodes; GL-50 cannot resolve a prior narrower
than the GW window), over sigma_z/z = 0.30 down to 0.012:
shift -> 0, log-log slope **1.89** over the whole range and **1.99** over the
last decade, with `shift / (2 e^2)` converging monotonically to **1.50**
(= 3 e^2). So the §7 fix gate ("must vanish ∝ (sigma_z/z)^2, cannot disturb #51")
is satisfied — with the caveat that the coefficient is 3, not 2.

### 3. Does it account for the observed rail? — YES, quantitatively

Local photometric `z_error/z` at the in-cat hosts' redshifts (z_true median
0.0706): quartiles **0.379 / 0.519 / 0.644** — *indicative only*, the local
`reduced_galaxy_catalogue.csv` is not the realization parent (differs in exactly
the `z_error` column, #40b PV width).

* Leg A' at those widths: median peak **h = 1.119**, 98.7% of hosts peak above 0.86.
* Leg B (adds the realization's own `z_obs` scatter): frac(peak > 0.86) =
  0.48 / 0.65 / 0.81 / 1.00 at sigma_z/z = 0.25 / 0.35 / 0.49 / 0.80.
  **Observed: 0.892.**
* Direct tilt test against production. Delivered ball numerator
  `S_i(h) = L_cat_i(h) * Sigma_glob(h)` (`Sigma_glob` = `sum_w_Dg(no_bh)` from the
  per-h log, `Delta ln Sigma_glob(0.73->0.86) = +0.0276`):

  | | median `Delta ln` 0.73->0.86 | IQR | frac > 0 |
  |---|---|---|---|
  | **OBSERVED**, 74 in-cat events | **+0.308** | [+0.208, +0.413] | **0.932** |
  | predicted, sigma_z/z = 0.35 | +0.329 | [+0.001, +0.643] | 0.751 |
  | predicted, sigma_z/z = 0.49 | +0.389 | [+0.162, +0.622] | 0.860 |
  | predicted, sigma_z/z = 0.65 | +0.385 | [+0.217, +0.542] | 0.978 |
  | **POINT kernel (analytic)** | **-408** | [-4064, -10] (p5/p95) | 0 |

  The production data therefore *independently* implies sigma_z/z ~ 0.35-0.6 for
  these hosts, which agrees with the (stale-column) catalogue value — so the
  conclusion does **not** rest on the stale column. The observed spread is
  narrower than predicted because the observation is a ball SUM over many
  galaxies while the prediction is a single host.

**The kernel swap moves the in-cat catalogue-leg tilt over 0.73 -> 0.86 from
-408 nats to +0.31 nats per event.** That is the C5 rail.

**Bonus, answering Gate B item 2's decisive sub-test without the cluster:** the
single-host `volume_deconv` peaks are *interior* on a grid extended to h = 2.4
(median 1.12 at the indicative widths). The 0.86 concentration is a **clipped
real runaway, not an edge artifact**.

### 4. Dark class — the SAME kernel pushes the WRONG WAY there

Inverting each delivered catalogue-leg peak into an effective host redshift
(`h_peak/h_true = f(z_eff)/f(z_hat)`, `f(z) = h*d_L` is h-independent):

| class | n | z_hat median | h_peak median | z_eff/z_hat median | at 0.60 edge | at 0.86 edge |
|---|---|---|---|---|---|---|
| IN-CAT | 74 | 0.071 | 0.860 | 1.168 (censored) | 0 | 0.892 |
| DARK | 1021 | 0.487 | 0.600 | 0.850 (censored) | 0.808 | 0.021 |

Dark events sit at z ~ 0.49, where `sigma_z/z` is ~0.10 and the kernel factor is
only **K = 1.03** — and K is **always > 1**. The dark peak at 0.60-0.64 requires
bare impostor hosts at `z_g/z_hat <= 0.83`, i.e. **foreground contamination**.
The C7 kernel acts *against* the dark rail, it cannot cause it. This also
explains why C7 is an in-cat-only effect: the shift goes as `(sigma_z/z)^2` and
`sigma_z/z` falls steeply with z (local catalogue medians: 0.42 at z~0.07,
0.25 at z~0.15, 0.16 at z~0.3).

### 5. Arithmetic audit of the claim's quoted magnitudes

* `2(sigma_z/z)^2` **is** the small-e expansion of `[z+sqrt(z^2+8s^2)]/2` — the
  claim's two statements are consistent to O(e^2), diverging at large e
  (e=0.49: 48% vs 35%).
* "+11% to +36% at sigma_z/z = 0.25-0.49" is arithmetically **correct for that
  formula** (11.24%, 35.45%).
* But the formula is the **wrong one for this code**: measured **+16.1% to
  +48.8%**. Corrected sentence: *at sigma_z/z = 0.25-0.49 the kernel inflates the
  effective h by +16% to +49%, h_eff 0.85-1.11* (claim said 0.81-0.99).
* **z-shift -> h-shift mapping for a d_L-anchored event** (the claim never states
  it): the numerator peaks where `d_L(z;h) = d_L_hat`, and `d_L(z;h) = f(z)/h`
  with `f(z) = (1+z) INT_0^z dz'/E(z')` h-independent, so exactly
  ```
  h_eff / h_true = f(z_eff) / f(z_true)
  ```
  At this venue (z ~ 0.07) f is near-linear, so h-shift = z-shift to within a few
  percent (+16.14% in z -> +17.09% in h at e=0.25). It is **not** a 3x lever, and
  it is **not** the `Delta h = Delta nats * sigma_h^2 / Delta h_window` conversion
  (claim's error #1) — that conversion is for nats budgets, not for peak locations,
  and is not used anywhere here.

## Correctness checks (`c7_checks.py`)

1. Driver vs `scipy.integrate.fixed_quad` on the same integrand: rel diff **0.0e0**.
2. `w_pop(z;h)/Z_g(h)` h-invariance over h in [0.6, 1.2]: **9.1e-16** (machine
   precision) — the kernel is exactly h-invariant, so the shift is a statement
   about the prior's SHAPE, not about an h-dependent prior.
3. n=50 vs n=400 quadrature: **<5e-12** at sigma_z/z = 0.25/0.49/0.80. No aliasing
   (unlike `volume_trunc`, which widened the numerator window to the host window).
4. Point-kernel peak: exactly 0.730000 for all 76 hosts.

## Caveats

* The **realized** observed catalogue is not local, so Leg B uses synthetic
  `z_obs = z_true + sigma_z N(0,1)` draws rather than the actual realization.
* `sigma_z` calibration from the local parent is **indicative** (stale `z_error`
  column, #40b). Mitigated by §3's independent production-side implication.
* Host sky is taken equal to event sky (true for the real host; the realization
  copies sky positions unscattered, `observed_realization.py`), which makes the
  3x3 GW MVN collapse exactly to the conditional 1D d_L-fraction Gaussian.
* The comparison in §3 is single-true-host prediction vs ball-sum observation.
* This says nothing about whether the completion leg's own in-cat rise is
  correct, nor about C3/C4/C8.
