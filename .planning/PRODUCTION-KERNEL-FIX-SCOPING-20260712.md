# Production host-z kernel correction — SCOPING (physics-change presentation gate)

**Date:** 2026-07-12 · **Branch:** `physics/zero-host-completion-fallback` · **Status:**
**SCOPING ONLY — user-gated, NO production code written or proposed for merge.** This is the
"before writing any code" presentation the `/physics-change` hard gate requires (old formula,
new formula candidates, references, dimensional analysis, limiting cases), assembled so the
user can decide **whether** and **how** to proceed. The actual derivation + implementation is a
separate `/gpd:derive-equation` → `/physics-change` → approval → implement → re-verify pass.

---

## 0. Why this is on the table (one paragraph)

The bias investigation converged: the deep-incompleteness bias ([L7]) and the shallow-venue
residual ([L8], seed600 +0.0132) are **two faces of the same estimator limitation** — the
`volume_deconv` host-redshift kernel is derived/calibrated for the regime σ_z/z ≪ 1 (the deep
commission venue, z_med ≈ 0.28, σ_z/z ≈ 0.12, where it is unbiased) and it **breaks when
σ_z/z ~ O(1)** — precisely GLADE's low-z **photometric** hosts. Measured 2026-07-12: at
seed600's z_med ≈ 0.046 the candidate population is **89.7% photometric, σ_z ≈ 0.0344,
σ_z/z ≈ 0.65** (`[L8]`). Both harness probes point to **one** production change: a
**z≥0-truncation-aware / photo-z-marginalized volume host-z kernel**. Both are *estimator*
limitations, not intrinsic un-calibratability — deep incompleteness IS calibratable at the
estimator level ([L7]).

---

## 1. OLD formula (what production computes today)

**Per-galaxy in-catalogue host-redshift prior**, `single_host_likelihood`,
`master_thesis_code/bayesian_inference/bayesian_statistics.py:2243–2281` (scalar) and the
batched twin `:2516–2600`:

```
p_g(z) = N(z; z_g, σ_z_eff) · w_pop(z) / Z_g          (volume_deconv / volume_global modes)
w_pop(z) = (dV_c/dz) / (1 + z)                         [:2261, :2279]
σ_z_eff  = sqrt(σ_z_catalogue² + σ_z_pv²)              [:2222–2223], σ_z_pv=(1+z)σ_v/c
Z_g      = ∫ N·w_pop dz  over  [max(z_g−4σ_z_eff, 1e-6),  z_g+4σ_z_eff]   (fixed_quad n=50) [:2265–2272]
```

Used in the single-host likelihood ratio (Gray 2020 A.10/A.19):

```
L_i(h) = N_g / D_g
N_g = ∫ p(x_GW | d_L(z,h), Ω_g) · p_g(z) dz      over GW window [z(d_L−4σ_dL), z(d_L+4σ_dL)]   [:2286–2304]
D_g = ∫ p_det(d_L(z,h), Ω_g) · p_g(z) dz          over [max(z_g−4σ_z_eff,1e-6), z_g+4σ_z_eff]  [:2306–2317]
```

The **z ≥ 0 clamp** (`max(z_g − 4σ_z_eff, 1e-6)`, `:2234–2240`) is the current, minimal handling
of the boundary: it truncates the lower integration limit but the kernel is otherwise the
un-truncated construction.

**Derivation of record:** `docs/derivations/G2b_host_z_volume_prior.md` (VERDICT: CONFIRMED
Bayes-correct **given** w_pop as the population prior; the "dV_c counted once" symmetry holds;
h-independent; reduces to spec-z as σ_z→0). **Empirical calibration of record:**
`results/commission_20260701/scratch/d2/NOTE_calibration_findings.md` — at z_med ≈ 0.3, σ_z=0.035,
the volume kernel restores coverage from ≈0 to nominal and drops the MAP bias from **−0.024
(bare Gaussian)** to **−0.002 (volume)**.

---

## 2. The identified limitation (empirical, two independent probes)

The G2b derivation itself flags the danger zone (§2.3 "the z~0.05 host"): at z_g=0.05 the
expansion parameter σ_z·s = 0.19 / 0.57 / 1.33 / 1.91 at σ_z = 0.005/0.015/0.035/0.050 — the
Eddington-in-z correction is **non-perturbative** for σ_z ≳ 0.015, and the exact per-host
redshift shift is "comparable to z_g itself (≳60% fractional)." The claim there is that the
*exact* deconvolution handles it. The two 2026-07-11 harness probes show it **over-corrects**:

| regime | venue | σ_z/z | volume-kernel bias | source |
|---|---|---|---|---|
| deep, calibrated | commission z_med 0.28 | ≈0.12 | −0.002 (nominal cov) | commission-d2 |
| **shallow, low-z photo-z** | seed600 z_med 0.044 | ≈0.65–0.80 | **+0.030** (cov68 collapses) | [L8] 260711-iic |
| **deep incompleteness** | comp_frac 0.2–0.85 | σ_z-dependent | membership-support **kernel leak** (removed by exact truncated mode) | [L7] 260711-117 |

**Mechanism (both):** the volume/Eddington-in-z correction is derived assuming the kernel
integrates over an **un-truncated** z line. When σ_z/z ~ 1 the Gaussian hits the physical z ≥ 0
boundary; the asymmetric truncation interacts with the steeply rising w_pop(z) ∝ dV_c/dz (∝ z²
at low z), and the correction stops exactly cancelling → residual HIGH bias. The deep-venue
analog is a **membership-support kernel leak**: a hard z-window over a common D fails to keep the
host kernel truncated consistently. N-2d specifically found the **hard** clamp is misspecified
under *observed-z* membership → the production candidate must use **soft (photo-z-marginalized)
membership**, not a harder truncation.

**Coupling constraint (do NOT fix the z-kernel in isolation — [L7] 260711-hx1):** production
also thresholds SNR on the noiseless injected waveform (a *latent*-threshold model). The exact
conditional for that class keeps BOTH a z-dependent inference σ (σ_f·A(z)/h, not const·d_L,obs)
AND p_det inside the numerator; fixing only one breaks the accidental cancellation. Any
kernel change should be co-designed with the distance-error model, or it can re-open the
+0.002…+0.005 noise-model floor. (The floor is ≤ campaign σ_boot — subdominant — but the
interaction is real.)

---

## 3. Candidate NEW formulas (directions — NOT a decided design)

Both need a full derivation pass; listed with their trade-offs. The regression gates in §6 are
binding on whichever is chosen.

**Candidate A — truncated-normal-consistent volume kernel (the L7 "exact" mode, hardened).**
Replace the un-truncated Gaussian by a proper **truncated normal** on the physical support and
normalize numerator and denominator over the *identical* truncated support:

```
p_g(z) = TN(z; z_g, σ_z; [z_lo, z_hi]) · w_pop(z) / Z_g ,   z_lo = 0 (or 1e-6),  z_hi = z_max
Z_g    = ∫_{z_lo}^{z_hi} TN·w_pop dz ,   with N_g and D_g sharing [z_lo, z_hi] (no separate GW window offset)
```
Pros: minimal conceptual change; L7 harness showed the exact truncated mode removes the entire
σ_z-dependent leak. Cons: N-2d warns a *hard* clamp is misspecified under observed-z membership —
A may under-perform the soft form; must reconcile the GW (numerator) window with the truncated
support so the prior is not evaluated outside its normalization domain (G2b §3.3 flag #2).

**Candidate B — photo-z-marginalized / full-PDF soft-membership kernel (the modern standard).**
Instead of a Gaussian σ_z with a hard z-window membership, carry each galaxy's **full photo-z
posterior** p_g(z) (or a truncated, volume-prior-consistent surrogate) and let membership in the
event's z-region be a **soft, photo-z-weighted** contribution:

```
p_g(z) ∝ p_photoz,g(z) · w_pop(z) / Z_g ,   membership weight = ∫_window p_g(z) dz  (soft, not 0/1)
```
This is what the LVK-era statistical dark-siren pipelines do (Alfradique/Bom et al. 2023–2026 use
full DL-derived photo-z PDFs; ICAROGW/GWCosmo marginalize the per-galaxy redshift likelihood with
the volume prior). Pros: addresses the N-2d "observed-z membership" defect at the root; general
(handles non-Gaussian GLADE photo-z). Cons: larger change; needs a truncation-consistent
normalization at low z regardless (B without a z≥0-consistent volume prior still has the boundary
issue); GLADE+ gives σ_z, not full PDFs, so B reduces in practice to "truncated-normal × volume
prior with soft membership" — i.e. **A + soft membership**.

**Working hypothesis for the user:** the fix is most likely **A's truncation-consistent
normalization + B's soft photo-z membership**, co-designed with the §2 distance-error coupling.
Not decided here.

---

## 4. References to ground the derivation (literature pass — consult before coding)

- **Gray et al. (2020)**, arXiv:1908.06050 — Eqs. A.10/A.19, 31–33: the in-catalogue numerator/
  denominator and the volume completion prior (the estimator's backbone; G2b/G2c map to it).
- **Mandel, Farr & Gair (2019)**, arXiv:1809.02063 — data- vs latent-threshold selection; the
  p_det-in-numerator rule that couples to §2.
- **Chen, Fishbach & Holz (2018)**, Nature 562:545 / arXiv:1712.06531 — statistical dark-siren
  host marginalization foundations.
- **Mastrogiovanni et al. (2023)**, arXiv:2305.10488 (ICAROGW) — Sec. IV: per-galaxy redshift
  likelihood marginalization with the comoving-volume prior and selection; the reference
  implementation of the "soft photo-z membership" family (Candidate B).
- **Alfradique, Bom et al. (2023–2026)**, arXiv:2310.13695, 2404.16092, **2603.20195** — LVK O3/O4a
  statistical dark sirens using **full per-galaxy photo-z PDFs** (DL-derived) + magnitude-limited
  selection: current best practice for exactly the photo-z-marginalized kernel (Candidate B).
- **Wang & Chen (2408.10382)** — Fisher tolerance study: galaxy redshift uncertainty + population
  model error are first-order H0 systematics (galaxy mass-function z-evolution must be known to
  O(1%) for a 1% H0). Motivates getting the kernel right and sets a tolerance yardstick.
- Project-internal: `docs/derivations/G2b_host_z_volume_prior.md`,
  `docs/derivations/G2c_gray_a9_a10_mapping.md`,
  `results/commission_20260701/scratch/d2/NOTE_calibration_findings.md`, ledger [L7]/[L8].

---

## 5. Dimensional analysis (unchanged; a fix must preserve it)

`[N] = z⁻¹`; `[w_pop] = Mpc³ sr⁻¹` (per unit z); `[Z_g] = Mpc³ sr⁻¹`; hence `[p_g] = z⁻¹`, a
proper density in z integrating to 1 over its (now truncated) support. The overall 4π and the
h⁻³ prefactor of w_pop cancel between numerator and Z_g (G2b §1: exact h-independence of the
prior shape). Any candidate MUST keep: (i) p_g a normalized density on its support, (ii) the
"dV_c counted once" symmetry between N_g, D_g, B_num, D(h), β_Ḡ, (iii) exact h-independence of
the prior shape.

---

## 6. Limiting cases / regression gates (binding on ANY chosen fix)

1. **σ_z → 0** ⇒ p_g → δ(z − z_g): must reduce continuously to the spectroscopic (bare) kernel.
2. **σ_z/z ≪ 1 (deep venue)** ⇒ must REPRODUCE the commission-d2 calibration: MAP bias −0.002,
   nominal coverage at z_med ≈ 0.3, σ_z = 0.035. **A fix that improves the shallow venue but
   regresses the deep venue is rejected.**
3. **σ_z/z ~ O(1) (shallow venue)** ⇒ must REMOVE the +0.030 harness bias / seed600 +0.0132
   (verified in the venue-matched pp_coverage harness before any production run).
4. **Deep incompleteness (comp_frac 0.2–0.85)** ⇒ must not reintroduce the L7 membership leak;
   check against the exact-mode harness result.
5. **Noise-model coupling** ⇒ must not re-open the §2 +0.002…+0.005 floor (co-verify with the
   model-σ + p_det-inside estimator, [L7] hx1).
6. **h-independence of the prior shape** ⇒ unit test (G2b §1.5): p_g(z) identical across trial h.

---

## 7. Open decisions for the user (do NOT assume)

- **D1 (whether to fix now vs Paper-B robustness bound):** the estimator limitation is bounded
  and understood; truncation stays a valid robustness bound. Fix in production, or quote as a
  systematic and defer? Evidence: deep incompleteness IS calibratable ([L7]); shallow is
  estimator-intrinsic but ≤ campaign σ_boot after de-rail.
- **Candidate choice (A / B / A+soft):** §3.
- **Coupling scope:** fix the z-kernel alone, or co-design with the distance-error model (§2)?
  ([L7] says do NOT add p_det-inside alone; the pieces cancel pairwise.)
- **Validation venue:** the multi-seed campaign is the real cross-seed adjudicator; local
  harness + commission-d2 + seed600 A/B are the pre-registration gates.

---

## 8. Process from here (NOT this session)

`/gpd:derive-equation` (truncated-normal × volume prior, soft membership, distance-error coupling)
→ dimensional + limiting-case verification (§5/§6) → `/physics-change` presentation of the single
chosen formula with old/new/reference → **user approval** → implement behind a new
`normalization_mode` (keep `volume_deconv` as the golden baseline; bit-identical default) →
re-verify all six §6 gates → campaign. Physics-trigger file:
`bayesian_inference/bayesian_statistics.py` — hard gate applies.
