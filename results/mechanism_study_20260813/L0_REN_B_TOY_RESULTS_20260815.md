# L0-REN-B toy results — A/B renormalization harness, retained-kernel-mass tilt measured

**Date:** 2026-08-15 · **Authorized:** ledger row #105 item 1 ([DO]) · **Status: PRESENTED, NOT
ADJUDICATED.** Numbers only; the four reads below apply the rule table registered in
`L0_REN_A_DERIVATION_20260815.md` §4 mechanically. The author adjudicates.

**Script:** `toys/lren_ab_toy.py` · **Output:** `L0_REN_B_toy_output.json` · `ruff check` /
`ruff format --check` / `mypy` clean.

## 1. Setup as run

- N_EV = 500, K = 400 candidates/event (1 host + 399 impostors), GL-50 quadrature, 8 seeds
  (101–108), h_true = 0.73, dh = 0.005 (central difference for T_REN; also used for arm-A's
  second-difference curvature) — same harness geometry as `m7_ab_toy.py`.
- sigma_d: bootstrapped fractional σ_dL/d_L from the production CRB CSV
  (`closed_loop_gfrac.load_sigma_triples`).
- sigma_z: the GLADE-empirical mix (`venue_transfer.load_pruned_z_sigma` +
  `venue_transfer.build_sigma_sampler`, the production VT-D3 z-decile sampler, reused verbatim).
- Population: per event, K candidates drawn on **true** z, uniform-in-comoving-volume inside the
  ball window built at **h_true** from d_obs (membership fixed, matching the production fixed-K
  ball). z_obs = z_true + σ_z·ε for every candidate.
- **Dose convention (differs from M7):** f_i ∈ {0.25, 0.5, 1.0} scales σ_z for **every**
  candidate, host included — the full-dose venue configuration §4 calls for, not the
  impostor-only dosing M7 used.
- **Arm A:** the estimator as coded — c₁ₖ = ∫ₐᵇ N(z; z_obsₖ, σₖ)·p_gw dz, a = max(z_lo(h),
  z_obs−5σ), b = min(z_hi(h), z_obs+5σ), h-moving edges, never divided by retained kernel mass.
- **Arm B:** identical a, b at the same h, same draws — each candidate's integral additionally
  divided by W_k(h) = Φ((b−zo)/σₖ) − Φ((a−zo)/σₖ), computed with the **same** a, b used for the
  integral at that h (recomputed at every h evaluation, so both arms move together as h moves).
  Only the renormalization differs; nothing else.
- T_REN = d/dh[Σᵢ(lnLᵢᴬ − lnLᵢᴮ)] at h_true, stacked over 500 events, scaled to 982 (the
  production event count), central difference over the same three h-grid points as the arm-A
  curvature.
- Implied MAP shift, **both conversions**: (i) the toy's own arm-A joint-posterior curvature
  (Laplace, second difference of Σ lnLᴬ at h_true); (ii) the fixed production σ_post = 0.004386
  (M3-note / M7-addendum convention) — shift = T_REN(982) · σ_post².
- Population fractions at h_true: double-clipped (both a and b bound by the box, not the kernel),
  single-clipped (exactly one side bound by the box), unclipped (kernel window entirely inside the
  box) — classified candidate-by-candidate from z_lo(h_true), z_hi(h_true) vs. z_obs ± 5σₖ.

Wall time: 2 min 54 s for the full sweep (3 doses × 8 seeds, 3 h-points each) — no event-count
reduction was needed.

## 2. T_REN by dose (mean ± seed-scatter std, nats/h, n=8 seeds)

| dose f_i | T_REN(982) | seed std | SE (std/√8) |
|---|---|---|---|
| 0.25 | −13.90 | 40.39 | 14.28 |
| 0.50 | +7.35 | 39.51 | 13.96 |
| 1.00 | **+99.52** | 52.13 | 18.43 |

Per-seed values span both signs at every dose (e.g. f=1.00: seeds 101–105 give +40…+171, seeds
106–108 give +40…+43 — all eight are positive at full dose, unlike the mixed-sign scatter seen at
the lower doses). The mean is dose-monotone (−13.9 → +7.3 → +99.5) and only the full-dose point
clears its own SE by more than one sigma.

**Dose-shape steps (paired per-seed, tighter than the naive std combination since the same seed's
draw underlies both doses being differenced):**

| step | value | seed-paired std | SE (std/√8) |
|---|---|---|---|
| T_REN(0.25→0.5) | +21.24 | 22.68 | 8.02 |
| T_REN(0.5→1.0) | +92.18 | 25.34 | 8.96 |

## 3. Implied MAP shift — both conversions (mean ± seed-scatter std)

| dose f_i | σ_post (toy, mean) | implied shift, toy curvature | implied shift, production σ_post=0.004386 |
|---|---|---|---|
| 0.25 | 1.927e-3 | −7.10e-5 ± 1.60e-4 | −2.67e-4 ± 7.77e-4 |
| 0.50 | 2.218e-3 | +3.14e-5 ± 1.91e-4 | +1.41e-4 ± 7.60e-4 |
| 1.00 (**registered dose**) | 2.478e-3 | +6.18e-4 ± 3.44e-4 | **+1.9145e-3 ± 1.0028e-3** |

The production conversion sits above the toy's own-curvature conversion at every dose (the same
direction M7's addendum found, ~3× larger at full dose here), because σ_post=0.004386 exceeds the
toy's own curvature-implied σ_post at this event count.

## 4. Population fractions at h_true (mean ± seed-scatter std, n=8 seeds)

| dose f_i | double-clipped | single-clipped | unclipped |
|---|---|---|---|
| 0.25 | 5.60% ± 1.11% | 36.13% ± 1.10% | 58.27% ± 1.21% |
| 0.50 | 15.59% ± 1.57% | 51.81% ± 1.11% | 32.60% ± 1.00% |
| 1.00 | 41.91% ± 1.90% | 49.59% ± 2.05% | 8.50% ± 0.52% |

The double-clipped fraction grows fastest with dose (5.6% → 15.6% → 41.9%), consistent with §2 of
the derivation note's picture that the double-clip (box-spans-window) regime dominates at full
dose; the unclipped fraction correspondingly collapses from 58% to 8.5%.

## 5. Mechanical reads (applying §4's rule table as registered)

| read | rule | this result | outcome |
|---|---|---|---|
| **R1 magnitude** | CLOSED iff full-dose implied shift (production conversion) ∈ [−1e-3, +1e-3] | +1.9145e-3 | **LIVE** — outside the band, ~1.9× above the +1e-3 edge |
| **R2 dose shape** | OWNS-SHAPE iff both steps within ±150 of (−550, −212); WRONG-SHAPE if either differs by >±300; PARTIAL-SHAPE between | step1 = +21.24 (target −550, diff 571.24); step2 = +92.18 (target −212, diff 304.18) | **WRONG-SHAPE** — both steps miss by more than the ±300 threshold; both also have the **opposite sign** from the target steps |
| **R3 budget** | CONSISTENT iff T_REN(1.0) ∈ −62 ± 150 | +99.52 (target −62, |diff| = 161.52) | **BUDGET-TENSION** — just outside the ±150 band (161.52 vs 150), and opposite sign from the target |
| **R-sign** | reported, not read | T_REN(1.0) = +99.52, positive | **positive** — matches §2's pre-registered structural prediction for the width term (h pushed up) |

**R3 note on possibility (ii):** per §3 of the derivation note, BUDGET-TENSION triggers "possibility
(ii) — genuine non-additivity of ablations... the joint arm (A-JREN) is mandatory before any
conclusion about the repair," and rules out any single-term ownership claim from this toy alone.

**R1/R3 relationship:** R1 is LIVE and R3 is BUDGET-TENSION together — the toy measures a
positive, non-trivial T_REN at full dose that both exceeds the L0 closure threshold and misses the
measured T_res(1.0) budget residual, in the sign direction §2's width-term analysis predicted
(+1055 nats/h at saturation) rather than the direction needed to explain T_res's negative
full-dose value.

## 6. No repair, no adjudication

No repair is proposed. Per the note's §3 and the R3 read above, the A-JREN joint arm becomes the
next indicated step if the author wants to pursue the non-additivity possibility — that decision is
the author's, not this toy's.

---

## Addendum (2026-08-15) — adversarial-verification notes

Verifier re-ran all 8 full-dose seeds bit-exact (+99.5227) with independent quadrature spot-checks (1e-6 agreement) and verified all four reads' arithmetic. The 1055-vs-99.5 gap is UNDERSTOOD, not a bug — exact per-seed decomposition (seed 101): double-clip +373.4, single-clip +101.8, weight-shift residual −324.5, net +150.8; suppression = toy-population ⟨D″D/D′²⟩ ≈ 0.73 vs production 0.216 (toy's own ceiling ~+367), partial saturation (median W = 0.80 among double-clipped), and offset/weight-shift cancellation — the derivation's pre-stated possibility (i), with numbers. Two caveats of record: (1) the toy population's host z_median is 1.66 vs the production venue's 0.44 (inherited from the committed harness), so the LIVE magnitude carries population-transfer uncertainty well beyond the seed scatter — the supported statement is "LIVE, order 1e-3, sign +", not a precise +1.9e-3; (2) R3's mean sits 0.6 SE outside its band (fragile as a band read) — the BUDGET-TENSION trigger stands primarily on the sign mismatch and R2's decisive WRONG-SHAPE. Prose slip corrected for the record: seeds 101–105 span +102…+171; +40…+43 are seeds 106–108 (JSON correct throughout).
