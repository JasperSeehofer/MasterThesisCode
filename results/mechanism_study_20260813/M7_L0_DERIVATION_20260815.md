# M7 — host/impostor ball-window inclusion asymmetry — L0 derivation

**Date:** 2026-08-15 · **Authorized:** ledger row #103 item 3 ([DO]) · **Status: PRESENTED, NOT
ADJUDICATED** · **Candidate origin:** intake dossier Erratum E1 (M-ID assigned 2026-08-14; the
structure is named in the dossier's own parity text but was never derived or armed).

## 1. The candidate, stated precisely

Ball membership is decided on **true** z (h-independent, K pinned; `calibration_gate.py:677-702`,
`venue_transfer.py:845-868`); the estimator then reads each candidate at its **scattered** z_obs and
integrates the kernel over the **h-dependent** window
`[max(z_lo(h), z_obs − 5σ_k), min(z_hi(h), z_obs + 5σ_k)]`, edges at d_obs(1 ∓ 4σ_d) mapped through
d_L(z,h) = D(z)/h. M7 posits that the asymmetry between the fixed membership rule and the moving
read window produces an O(σ_z) net tilt: scattering pushes members' kernels across the edges
(5σ_z ≈ 0.21 ≫ window half-width ≈ 0.06–0.10 in z at this venue, so the boundary population is
saturated), the two edges are asymmetrically populated (w_pop rises across the window; the
d-symmetric window is z-asymmetric), and the edges move with h while membership does not. The host
is structurally exempt (its p_gw peak is central by construction), so the effect is
impostor-dose-keyed — qualitatively matching the measured dose-dependent tilt residual.

## 2. The derivation — both sub-channels share the p_gw-edge suppression

Every h-derivative of a candidate's contribution that flows through **domain motion** is a boundary
flux: dc₁ₖ/dh ⊇ kern(z_edge)·p_gw(z_edge)·(dz_edge/dh). By the window's construction the integrand
at either edge carries **p_gw(edge) = e^(−8)·p_gw^peak** — the same 4σ-pinning that killed M3. This
holds for BOTH sub-channels:

- **Interior-centred kernels** (M3's accounting): flux multiplier kern(edge)/kern(z*) ≤ 1. Bounded
  at 2Φ̄(4) = 6.3e-5 fractional; measured +6.0e-7 in h. CLOSED (M3 note, analytic core toy-free).
- **Boundary-layer / straddler kernels** (the new piece M7 adds): kernels centred within O(σ_z) of
  an edge, or outside with tails reaching in. Here kern(edge)/kern(z*) **> 1** — for a kernel
  centred at the upper edge, kern(edge) ≈ kern_max while kern(z*) ≈ e^(−δ²/2σ²)·kern_max with
  δ ≈ the edge-to-peak distance (~2σ_z at this venue) — an enhancement of order **e^(+2) ≈ 7** per
  candidate, times a **saturated boundary-layer population fraction** of order min(1, σ_z/width)
  ~ O(0.2–0.5) rather than M3's implicit O(σ_z) rare-edge factor.

**The honest bound therefore lifts M3's +6e-7 by a factor of order 10²–10³** (kernel enhancement ×
population enhancement × the loss of M3's interior-normalisation slack), i.e. an implied MAP
displacement of order **6e-5 … 6e-4 in h**. That is still 30–600× short of the residual +0.0192,
**but it brackets the registered L0 closure threshold (1e-3 in h) from below without clearing it
analytically** — unlike M3, the analytic argument alone does NOT close M7 with a comfortable
margin. The parity constraint does not kill it either: a truncation at a moving support edge is
genuine O(σ_z¹) structure (the boundary layer is O(σ_z) wide and the distortion inside it is O(1)),
which is exactly the loophole class the parity argument names.

**Sign (pre-stated):** w_pop rises across the window and the z-space window is wider above its
centre, so the upper boundary layer outweighs the lower; raising h moves both edges up, un-clipping
upper-layer kernels (dc₁ₖ/dh > 0) and further clipping lower-layer ones (dc₁ₖ/dh < 0); the net is
**positive — h is pushed UP**, matching the defect's direction. A measured negative slope refutes
M7 outright regardless of magnitude.

## 3. The decisive L0 experiment (registered here, before it is run)

Adapt the committed `toys/m3_toy.py` A/B harness (same estimator mirror, same GL-50 rule, same
bootstrapped σ_dL): **arm A** = as coded (edges move with h); **arm B** = identical except
`z_lo, z_hi` frozen at their h_true values (kills ALL edge-motion channels — M3's interior clip and
M7's boundary layer together; the M3 share is known ≤ 6e-7 and is subtracted as background).
Population faithful to the venue on the one axis that matters: members selected on true z inside
the window, THEN scattered (σ_z at the GLADE mix), so straddlers arise with their natural weights;
report the realized boundary-layer fraction alongside.

| registered read | rule (two-sided, fixed before running) |
|---|---|
| **M7-CLOSED** | implied MAP shift from A−B, at full dose, in [−1e-3, +1e-3] in h — the parent prereg's registered L0 closure band |
| **M7-LIVE** | shift > +1e-3 (direction must be positive per §2; a live M7 becomes the leading candidate for the dose-dependent tilt residual and requires a freshly registered instrument arm — both stage-2 L1 slots are spent) |
| **M7-REFUTED-ON-SIGN** | shift < −1e-3 |
| dose scaling reported | f_i ∈ {0.25, 0.5, 1.0}, trend vs the M6R residual's measured f_i-dependence — consistency check only, no closure weight |

False-read note: the toy's K-regime caveat (row #102) applies to *faithfulness of magnitude*, not
to the A−B isolation (the term cancels exactly between arms); the closure band at 1e-3 carries a
≥30× margin over the §2 lower bracket, so a CLOSED read is robust to O(1) toy-scale error; a LIVE
read at this L0 is a *promotion to candidate*, never a confirmation.

## 4. Relation to the other open accounts

If M7 reads LIVE with an f_i-dependence matching the M6R residual, the natural hypothesis is
**M6-residual ≡ M7** (the dose-dependent tilt component *is* the boundary-layer flux). That identity
is NOT asserted here and would need its own registered discriminator. If M7 reads CLOSED, the
residual account falls back to the σ_z²-class compounding terms (M1's retained negative quadratic
among them) and any remaining structure the M6R decomposition surfaces.

*This note contains the derivation and the pre-stated reads; the toy numbers land in
`M7_L0_TOY_RESULTS_20260815.md` + json. No repair is proposed; no instrument arm is requested.*
