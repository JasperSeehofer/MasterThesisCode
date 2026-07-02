# Photo-z in-catalogue normalisation: candidate synthesis and bridge verdict

Status: **NEGATIVE RESULT.** Every proposed normalisation that keeps the GLOBAL Option-A
selection denominator and changes only the numerator kernel **passes the sigma_z -> 0 gate but
FAILS to de-rail** — it rails UP to the upper grid edge (h = 0.8700) at sigma_z = 0.035, i.e. it
sign-flips the standard bias (-0.13 -> +0.14) instead of curing it. The local same-kernel
denominator rails at BOTH sigma_z (gate fail). No candidate produces an interior peak at the truth.

All numbers below are from the self-consistent rung_I closure (no sky, f = 1, truth h = 0.73),
reduced settings n_gal = 12000, n_events = 250, seed = 1, which reproduce the documented baseline
exactly. Prototype: `scripts/bridge_closure/_rungI_verify_B.py` (all four candidates side by side).

---

## 1. The empirical landscape (measured, not predicted)

| Candidate | numerator effective prior | denominator | sigma_z=0.002 | sigma_z=0.035 | verdict |
|---|---|---|---|---|---|
| STANDARD (current code) | `n_smooth = Σ w_g N(z;z_g,σ)` (doubly-smeared dVc) | GLOBAL Option-A `Σ_global w_g p_det(z_g)` | **0.7438** peaked | **0.6000** rail DOWN | baseline |
| Angle A/C — per-galaxy posterior `N·p_bg/Z_g` | -> `p_bg` (clean dVc) | GLOBAL (unchanged) | 0.7478 peaked | **0.8700** rail UP | disqualified |
| Angle B — global de-count `g=p_bg/(S p_bg)` | -> `p_bg` (clean dVc) | GLOBAL (unchanged) | 0.7439 peaked | **0.8700** rail UP | disqualified |
| Local same-kernel (consistent-denom, any kernel) | `n_smooth` or `p_bg` | LOCAL `Σ w_g ∫ p_det N dz` | **0.8700** rail | 0.8700 rail | gate FAIL |

The truth (0.73) sits **strictly between** the two rails the search produces: the global denom
rails DOWN to 0.600, every flat-numerator / local-denom variant rails UP to 0.870. **No pure
redshift-prior normalisation in the searched space lands on the truth.**

Angle A, Angle B and Angle C are algebraically distinct constructions but, in the f = 1
continuous limit, all three drive the numerator's effective prior to the same clean `∫ p_GW p_bg`.
Against the frozen global denominator this yields the **identical** upper rail (0.8700, bit-for-bit),
which is also the exact rail of the already-disqualified consistent-denom candidate. The
"de-rail" prediction in GT0, GT1, and all three derivation angles is **empirically falsified**.

---

## 2. The candidates DO pass the sigma_z -> 0 gate (proofs hold; gate is necessary, not sufficient)

The reductions are correct — the failure is at the de-rail step, not the gate.

**Angle A/C (per-galaxy posterior).** `K_reg(z;z_g) = N(z;z_g,σ)·p_bg(z)/Z_g`,
`Z_g = ∫ N(z';z_g,σ) p_bg(z') dz'`. As σ -> 0, `N -> δ(z-z_g)`, so `Z_g -> p_bg(z_g)` and
`K_reg -> δ(z-z_g)·p_bg(z)/p_bg(z_g) = δ(z-z_g)`. The `p_bg` factor cancels identically;
`N_g -> p_GW(d_L(z_g,h))`, recovering the bare Option-A catalogue sum. Denominator untouched.
Measured: 0.7478 vs standard 0.7438 — gate PASS.

**Angle B (global de-counting).** `g(z) = p_bg(z)/(S p_bg)(z)`, `(S p_bg)(z) = ∫ p_bg(z')N(z;z',σ)dz'`.
As σ -> 0, `(S p_bg) -> p_bg`, so `g -> 1` pointwise, and `n_smooth -> Σ w_g δ(z-z_g)`. The
corrected integrand `n_smooth·g -> Σ w_g δ(z-z_g)`, the bare Option-A sum. `g` is exactly
h-independent (dVc/dz ∝ h^-3 × pure-z-shape cancels in the ratio). Measured: 0.7439 — gate PASS.

Both preserve the global scale-free Option-A structure (n_gal cancellation intact); the gate is
satisfied by construction. **It is the wrong test to gate on.**

---

## 3. Why every numerator-only fix rails — the root mechanism

The sharp GW factor collapses the numerator to `A(h)·p_eff(z*(h))`, where `z*(h)` solves
`d_L(z*,h) = d_meas` and is **increasing in h**, and `p_eff` is whatever effective redshift
prior the numerator carries. The single global selection denominator `D(h)` (a scalar per h) does
**not** track the LOCAL gradient of `p_eff` at `z*(h)`. So:

- If `p_eff = n_smooth` (steep, doubly-smeared dVc): vs the global denom that grows with h, the net
  is over-suppressed -> rails DOWN (0.600).
- If `p_eff = p_bg` (cleaned, but still `∝ dVc/(1+z)`, a **rising** function): the climb relative to
  the same growing global denom now wins -> rails UP (0.870).

Cleaning the numerator from `n_smooth` to `p_bg` does not remove the gradient — **`p_bg` is itself
rising**. It only changes which way the imbalance tips. The truth lies between because neither prior
shape is the one the fixed global denominator happens to normalise.

**The deeper cause: `p_det ≈ 1 across the entire in-catalogue redshift range** (hosts at z ~ 0.046
<< the GW horizon). There is **no selection gradient** locally. Consequently:

1. The global Option-A denominator varies with h only through the distant horizon edge (far from
   the catalogue galaxies) — it cannot track the local catalogue-density gradient at `z*(h)`.
2. The only object that DOES track that local gradient is the local catalogue count, i.e. the
   `∫ p_det·p_cat` over the event ball — but with `p_det ≈ 1` this is just `Σ w_g` (the local
   number density = n_gal), which the local same-kernel candidate reintroduces -> rails UP at ALL
   sigma_z (0.870/0.870), breaking the gate.

A selection-based normalisation can only cancel a local density gradient if the selection function
**varies** across that region. For nearby photometric in-catalogue EMRIs it does not. This is the
structural obstruction.

---

## 4. Is the bias irreducible?

**Within the imposed constraint set — change only the numerator kernel, freeze the global
Option-A point-catalogue denominator — YES, the bias is irreducible.** The bridge falsifies all
three angles and the shared GT0/GT1 premise that "the bias is a numerator-kernel defect curable
without touching the denominator." It is not. The numerator and the denominator are coupled
through `z*(h)`, and no kernel choice closes that coupling while the denominator stays a global
scalar that is blind to the local gradient.

**Fundamentally (no constraints) — NO.** The literature (Echoes 2509.18243; 2502.17747) demonstrates
that consistently-normalised photometric catalogues give unbiased, variance-only posteriors, so a
correct form exists. But it lies **outside** the searched space:

- The genuine Gray/Hitchhiker same-kernel ratio puts the **identical** `p_cat(z)` in BOTH the
  numerator and the denominator over the **full** population. The bridge's "consistent-denom" is a
  LOCAL, n_gal-reintroducing approximation of it (and rails); the true global same-kernel ratio is a
  **denominator change**, contradicting the "freeze the global denom" rule. In the `p_det ≈ 1`
  regime even this likely degenerates per-event (`∫ p_det p_cat ≈ ∫ p_cat = W`), so the
  cancellation must come from **ensemble coherence** across many events whose `z*_i(h)` shift
  together — a property the per-event normalisation cannot supply on its own.
- The residual half-inconsistency in the closure (events injected at the host's TRUE z, while the
  inference convolves the REPORTED z_g) means the per-event normalisation and the injection prior
  are not matched object-for-object; this must be reconciled before any normalisation can be
  unbiased at fixed catalogue.

**Conclusion: chasing the numerator kernel is the wrong lever** (now proven, not argued). The fix
requires the numerator AND the selection denominator to share one consistent population density over
the full redshift range, with the n_gal cancellation re-derived under that shared density — i.e. a
joint num+denom reconstruction, not a frozen-denominator numerator patch.

---

## 5. Bridge prototype and the (failed) acceptance criterion

Prototype: `scripts/bridge_closure/_rungI_verify_B.py`, function `run_closure_photoz(..., flag)`.
Flags: `regularised_kernel` (Angle A/C), `global_voldecount` (Angle B), `consistent_denom` (local).

Hard acceptance criterion (REQUIRED, NONE MET):
- gate: sigma_z = 0.002 -> ~0.73 PEAKED, matching standard 0.744. **MET by A/B/C** (0.748/0.744/0.744).
- de-rail: sigma_z = 0.035 -> ~0.73 interior peak, beating standard's 0.600. **MET BY NONE**
  (A/C and B both -> 0.8700 EDGE, railed=True; local same-kernel -> 0.8700 at both sigma_z).

First bridge test command:
```
uv run python scripts/bridge_closure/_rungI_verify_B.py
```
Measured output: STANDARD 0.7438/0.6000; REG-KERNEL (A/C) 0.7478/0.8700; ANGLE B 0.7439/0.8700;
local same-kernel 0.8700/0.8700.

**Recommended next rung (outside this synthesis's scope):** test the genuine GLOBAL same-kernel
ratio `∫ p_GW p_cat / ∫ p_det p_cat` with one shared `p_cat` to the horizon AND a photo-z-consistent
re-injection (events at z = z_g + N(0,σ_z)), and check ensemble-coherent de-railing. Do NOT promote
any Section-2 candidate to `bayesian_statistics.py` — they are confirmed to rail.
```
