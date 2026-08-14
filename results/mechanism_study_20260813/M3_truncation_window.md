# M3 — h-dependent truncation of the unrenormalised z-kernel

**Date:** 2026-08-13 · **Scope:** analysis only (no production code touched, no
`/physics-change` gate opened, no commit) · **Target defect:** `bias = +1 × σ_z` (h units),
0/N coverage, posterior ~8.5× narrower than the displacement
(`results/venue_transfer_20260811/VENUE_TRANSFER_READOUT.md`).

---

## VERDICT

> **M3 REFUTED.** The mechanism is REAL (the code does truncate an unrenormalised Gaussian
> kernel on an h-dependent domain — confirmed at `venue_transfer.py:1110-1147`), but it is
> **~5 orders of magnitude too small** and has the **wrong σ_z scaling**.
> Numerically isolated: MAP displacement from M3 alone ≈ **+6 × 10⁻⁷ in h**, against the
> **+3.72 × 10⁻²** to be explained — ratio **1.6 × 10⁻⁵**. Sign is **positive (h up)**, matching
> the defect's direction, but the amplitude kills it. Scaling is **decreasing** in σ_z
> (2.3e-6 → 8.1e-7 → 6.5e-7 at σ_z = 0.011 / 0.035 / 0.042), not linear-increasing, so it also
> fails constraint (b) independently of amplitude.

---

## 1. What the code actually does (confirmed, with line numbers)

`darksiren_emri/validation/venue_transfer.py::_channel_terms_at_h` (lines 1061-1180).

**Quadrature nodes.** `x = gctx.cl_ctx.gl_nodes`, `w_gl = gctx.cl_ctx.gl_weights`
(lines 1096-1097). These are built once in
`closed_loop_gfrac.py:388` — `roots_legendre(config.n_quad)` with
`n_quad = _HOST_QUAD_N` (`closed_loop_gfrac.py:190`), and `_HOST_QUAD_N = 50`
(`bayesian_inference/bayesian_statistics.py:392`). Standard Gauss–Legendre on `[-1, 1]`,
affine-mapped per row.

**Constants.**
- `cl._SIGMA_WINDOW = 4.0` — `closed_loop_gfrac.py:155` ("the estimator's ±4 sigma z window (production)").
- `cg._IMPOSTOR_KERNEL_WINDOW = 5.0` — `calibration_gate.py:215` ("per-candidate ±5 sigma_z kernel window (prereg §4.2)").

**The h-dependent outer window** (lines 1109-1115):
```
d_L_nodes, z_tab = gctx.cl_ctx.z_of_dl_tables[k]        # the per-h d_L↔z ladder
z_hi_e = interp(d_obs_e * (1 + 4·σ_dL_e), d_L_nodes, z_tab)
z_lo_e = interp(d_obs_e * (1 − 4·σ_dL_e), d_L_nodes, z_tab)
z_lo_e = max(z_lo_e, 1e-6);  z_hi_e = min(z_hi_e, z_tab[-1])
```
`z_of_dl_tables[k]` is the inverse-distance table **at h = h_grid[k]**, so `z_lo(h), z_hi(h)`
move with h: since `d_L(z,h) ∝ 1/h`, raising h pushes both edges **up** in z.

**The per-candidate domain** (lines 1127-1131):
```
a = max(z_lo_p, z_obs − 5·σ_k)
b = min(z_hi_p, z_obs + 5·σ_k)
valid = b > a;  half = (b−a)/2;  mid = (b+a)/2
z_nodes = mid + half·x
```

**No renormalisation — CONFIRMED.** Line 1139 is
`kern = norm.pdf(z_nodes, loc=zo, scale=so)` and line 1141 is
`c1q = half * (integ @ w_gl)`. There is **no** division by
`Φ((b−z_obs)/σ_k) − Φ((a−z_obs)/σ_k)` anywhere between lines 1127 and 1147, and no other
per-row normaliser. The retained kernel mass is therefore a function of h, exactly as M3
posits. (Identical structure in the gate twin, `calibration_gate.py:833-842`.)

**σ_z = 0 branch** (lines 1148-1165): `q = sig_c > 0.0` splits the rows; zero-σ rows take a
point evaluation `p_gw(z_obs)` gated only by `valid_p = (zo ≥ z_lo_p) & (zo ≤ z_hi_p)`. No
kernel, no integration domain, hence no truncation effect — consistent with M3's own claim and
with constraint (a). **So M3 passes (a) trivially.**

---

## 2. Analytic size of the effect — the decisive step

Write, for candidate k of event i,

  c₁ₖ(h) = ∫ₐᵇ N(z; z_obsₖ, σₖ) · p_gw(z; h) dz,  p_gw = N(d_L(z,h)/d_obsᵢ ; 1, σ_d,ᵢ).

The *only* thing truncation removes is the region outside `[z_lo(h), z_hi(h)]` (the `±5σ_k`
clip is symmetric in the kernel's own variable and h-independent, so it cannot tilt in h; only
the GW-window clip can).

**Key observation: the outer window is placed at ±4σ in the GW variable, not in the kernel
variable.** By construction `d_L(z_lo/hi, h)/d_obs = 1 ∓ 4σ_d` for every h. Hence everywhere
outside the window,

  p_gw ≤ e^(−½·4²) · p_gw^peak = 3.35 × 10⁻⁴ · p_gw^peak,

and integrating the tail,

  ∫_outside p_gw dz = 2·Φ̄(4) · ∫_all p_gw dz = 6.3 × 10⁻⁵ · ∫_all p_gw dz.

So the *fractional* mass the truncation can remove from the product `kern · p_gw` is bounded by

  |Δc₁ₖ| / c₁ₖ ≲ 6.3 × 10⁻⁵ · [ kern(edge) / kern(z*(h)) ],

where z*(h) is the p_gw peak (always interior). This bound holds **regardless of how much
kernel mass is clipped** — and in this venue a lot is clipped: median σ_dL/d_L = 0.0373
(from `prepared_cramer_rao_bounds.csv`, n = 1590), so the window half-width in z is ≈ 0.06-0.10,
while 5σ_z = 0.21 at the GLADE mix. Most candidates ARE clipped, but always in the region where
the integrand is already below e⁻⁸ of its peak. The clipping is *cosmetically* severe and
*numerically* irrelevant.

**Sign.** Raising h moves `z_lo, z_hi` up together (d_L ∝ 1/h). For a candidate below the
window centre the upper edge recedes and the lower edge advances; averaged over a candidate
population filling the window the two do not cancel exactly, and the residual is positive
(more mass retained at higher h in this geometry) — the toy below measures a **positive**
stacked slope, i.e. M3 pushes h **up**, the same direction as the observed bias. Direction is
right; magnitude is not.

**Scaling in σ_z.** Two competing dependences: the *number* of candidates whose ±5σ_k box
crosses an edge grows ∝ σ_z while σ_z ≲ window half-width, but the per-candidate clipped
integrand mass is set by the p_gw tail and is σ_z-independent, and once 5σ_z exceeds the window
half-width (already true at σ_z ≳ 0.02 here) the edge-crossing fraction saturates at 1 while the
kernel density at the edge falls ∝ 1/σ_z. Net: **flat-to-decreasing** in σ_z in this regime, not
∝ σ_z. Measured below: 2.3e-6 → 8.1e-7 → 6.5e-7 for σ_z = 0.011 → 0.035 → 0.042. That is the
opposite trend to R_dose ≈ const.

---

## 3. Which candidates dominate — and is the edge population big enough?

With K ≈ 1216 candidates/event and 982 events, the edge population is essentially **all** of
them (5σ_z ≈ 0.21 ≫ half-window ≈ 0.08). So M3 gets the most favourable possible population
factor — it is not a rare-edge-case argument that fails on counting. It still loses, because
the per-candidate ceiling is 6.3 × 10⁻⁵ *fractional*, and the ceiling does not improve with
candidate count: `L_i` is a **mean** over k (line 1167, `/ K`), so a uniformly-applied
fractional perturbation stays a fractional perturbation of `ln L_i` no matter how large K is.
Counting cannot rescue a 10⁻⁵ per-event lever.

**What is needed.** Joint posterior sd = 0.004376 (readout §, T-c(0.730) 1D). To displace the
MAP of a locally-Gaussian joint by Δ = +0.037237 requires an extra linear term of slope

  S_need = Δ / σ_post² = 0.037237 / 0.004376² = **1.94 × 10³ per unit h**,

i.e. ≈ 1.98 per event per unit h, i.e. each event's `L_i` must swing by ~8% across the ±0.04
h range that matters. M3's ceiling is ~10⁻⁴.

---

## 4. Numerical isolation (the cheap decisive experiment, already run)

Toy: `scratchpad/m3_toy.py` (not committed; reproduced below in method).
982-event-equivalent stack, K = 400 candidates/event, 50-node GL, real
`σ_dL/d_L` bootstrapped from `prepared_cramer_rao_bounds.csv`, candidates uniform in the
4σ window, `z_obs = z_cand + σ_z·ε` (zero-mean, generator/estimator matched).

- **Arm A** = the code as written (outer window at `_SIGMA_WINDOW = 4`).
- **Arm B** = identical in every respect except the outer window built at 12σ_d, which makes
  the GW-window truncation negligible while leaving the ±5σ_k kernel clip, the candidate set,
  the GL rule and everything else bit-for-bit the same. `A − B` is therefore **exactly the M3
  term**, with α(h), the population, and every other estimator ingredient cancelling.

| quantity | value |
|---|---|
| mean per-event \|ln L_A − ln L_B\| | 3.8 × 10⁻⁵ |
| max per-event \|ln L_A − ln L_B\| | 6.3 × 10⁻⁵ (= the analytic 2Φ̄(4) ceiling — the bound in §2 is tight) |
| M3 stacked slope d/dh Σᵢ(ln L_A − ln L_B), 982 events | **+3.1 × 10⁻²** per unit h |
| slope needed for +0.0372 | 1.94 × 10³ per unit h |
| **ratio M3 / needed** | **1.6 × 10⁻⁵** |
| **implied MAP shift from M3 alone** | **+6.0 × 10⁻⁷ in h** |

Dose sweep (same toy, 120 events):

| σ_z | implied MAP shift from M3 | required (R_dose ≈ 1) |
|---|---|---|
| 0.011 | +2.3 × 10⁻⁶ | +1.2 × 10⁻² |
| 0.035 | +8.1 × 10⁻⁷ | +3.6 × 10⁻² |
| 0.042 | +6.5 × 10⁻⁷ | +3.7 × 10⁻² |

M3 *falls* as the dose rises. R_dose from M3 spans 2.1e-4 → 1.6e-5 — a factor 13 the wrong way
across a factor 4 in dose.

---

## 5. Self-falsification against the three constraints

| constraint | M3 | note |
|---|---|---|
| **(a) vanishes identically at σ_z = 0** | **PASSES** | lines 1148-1165 take the point branch; no kernel, no domain-truncation term. |
| **(b) linear in σ_z (R_dose ≈ const, 1.07 → 1.01 → 0.89)** | **FAILS** | M3's contribution is flat-to-decreasing in σ_z (§2, §4); it produces R_dose ∝ roughly 1/σ_z in this regime, a factor ~13 change where the data show ~1.2. |
| **(c) not misspecification** | **PASSES (irrelevant)** | M3 is a genuine estimator-internal defect, not a generator/estimator mismatch; the toy shares σ_z between arms by construction. |
| **amplitude** | **FAILS by 6.2 × 10⁴** | +6 × 10⁻⁷ vs +3.7 × 10⁻². |

M3 fails (b) and fails on amplitude. Passing (a) and (c) is not enough.

**Robustness of the refutation.** The refutation does not depend on toy details (candidate
z-distribution, K, event z-distribution). It rests on one structural fact: the truncation edge
is pinned at **4σ in the GW-likelihood variable**, so everything discarded lives under
e^(−8) of the integrand peak, capping the per-event log-perturbation at 2Φ̄(4) = 6.3 × 10⁻⁵
whatever the kernel does. To make M3 competitive you would need `_SIGMA_WINDOW ≈ 1`, not 4.
(Corollary worth recording: this same fact means **widening `_SIGMA_WINDOW` cannot fix the
defect either** — it is a genuinely inert knob at 4σ.)

---

## 6. If it had survived — and what to do now

The minimal test *would* have been to divide `kern` at line 1139 by
`Φ((b−z_obs)/σ_k) − Φ((a−z_obs)/σ_k)` (renormalise over the retained domain), or to widen
`_SIGMA_WINDOW`. **Neither is worth doing**: the A/B toy above already isolates the entire term
with the window-widening arm and returns 10⁻⁷. Running N = 10-25 seeds of the full instrument
to chase a 6 × 10⁻⁷ h shift, against a per-seed noise of ~5 × 10⁻³, would be ~10⁴× under the
detection floor. **Recommend: close M3 without an instrument run and without a code change.**

**Redirection for the mechanism hunt.** The 4σ-pinning argument generalises: any candidate
mechanism whose effect is confined to the wings of `p_gw` is capped at ~10⁻⁴ per event and
cannot produce a 2 × 10³ slope. A surviving mechanism must act **at the peak of the integrand**
— i.e. on the *shape or normalisation of the kernel where p_gw is O(1)*, or on α(h). Two
structural leads consistent with that and with (a)+(b):
1. **Missing Jacobian / measure on the kernel.** `kern` is a density in `z`, multiplied by a
   density in `d_L/d_obs`, and integrated `dz` (lines 1138-1141) — a `|d(d_L)/dz|` factor is
   absent. Any such measure mismatch acts at the peak and its leading effect is O(σ_z) once
   convolved with a σ_z-wide kernel (the σ_z = 0 point branch has no such factor to get wrong).
2. **Kernel-vs-population asymmetry at the peak.** The candidate population inside the window
   is not flat in z (comoving-volume rise), so smearing a rising population with a symmetric
   σ_z kernel displaces the effective z of every candidate by ≈ +σ_z² dlnn/dz — but note that
   is σ_z², so it fails (b) unless the compounding is with the GW window's own gradient, which
   would be O(σ_z). Worth a targeted derivation; the σ_z¹-vs-σ_z² accounting is the whole game.

Both are testable with the same A/B toy harness used here (swap one term, re-measure the
stacked slope against S_need = 1.94 × 10³), at a few CPU-minutes each — far cheaper than seeds.

---

**Method reproducibility.** Toy script lived at
`$SCRATCH/m3_toy.py` (session scratchpad; not part of the repo). It imports only
`darksiren_emri.physical_relations.dist_vectorized` and
`darksiren_emri.validation.closed_loop_gfrac.load_sigma_triples`, and mirrors lines 1109-1147
verbatim in structure. No production module was modified.

---

## Addendum (2026-08-14) — status downgraded to *plausible pending committed artifact* (ledger row #102)

The refutation's analytic core stands and is toy-independent: the 2Φ̄(4) = 6.3e-5 per-event ceiling against S_need = 1.94e3 closes M3 by ≥3 orders of magnitude with no toy input, and an independent adversarial rebuild reproduced the bottom-line numbers. Two record defects, per the commission: `m3_toy.py` was never committed (this note's decisive numerics are not reproducible from the repo), and §2's "the bound is tight" claim is false in the admitted z_obs ∈ (z_hi, z_hi + 5σ_k] regime. The closure verdict is unchanged; the note does not meet the project's own reproduction bar until a reproducer is committed.
