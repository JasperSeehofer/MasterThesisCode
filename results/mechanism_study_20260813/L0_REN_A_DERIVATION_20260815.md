# L0-REN-A — the unrenormalized truncated kernel: derivation and pre-stated toy reads

**Date:** 2026-08-15 · **Authorized:** ledger row #105 item 1 · **Status: PRESENTED, NOT
ADJUDICATED** · Frozen before L0-REN-B runs (same discipline as M7-L0).

## 1. The defect term

The code computes c₁ₖ = ∫ₐᵇ N(z; z_obsₖ, σₖ)·p_gw dz with a = max(z_lo(h), zo−5σ),
b = min(z_hi(h), zo+5σ) and **never divides by the retained kernel mass**
W_k(h) = Φ((b−zo)/σ) − Φ((a−zo)/σ). The renormalized estimator would use c₁ₖ/W_k. The defect in
each event's log-likelihood is Δᵢ(h) = ln Σₖc₁ₖ − ln Σₖ(c₁ₖ/W_k); the tilt defect is
T_REN = Σᵢ dΔᵢ/dh at truth. At σ = 0 the point branch has no kernel and W ≡ 1: the defect vanishes
identically — constraint (a) holds.

## 2. Regime structure (venue numbers: 5σ̄_k ≈ 0.21 at full dose; window full width in z ≈ 0.12–0.20)

- **Double-clipped regime** (box spans the window; mid-window candidates at full dose):
  W = Φ((z_hi−zo)/σ) − Φ((z_lo−zo)/σ). Large-σ expansion: ln W ≈ ln width_z(h) − (z_mid−zo)²/2σ²
  − ln(√2π σ), giving per candidate
  **d ln W/dh ≈ d ln width_z/dh − (z_mid−zo)·z_mid′/σ²** — a **width term** and an **offset term**.
- **Single-clipped boundary layer** (small σ, or edge-adjacent candidates): d ln W/dh =
  (φ/Φ)·z_edge′/σ — inverse-Mills flux, boundary population ∝ σ at small dose.
- Interior, unclipped: W = 1, no contribution.

**The width term, evaluated.** From D(z_edge) = h·d_obs(1 ± 4σ_d): width_z ≈ 8σ_d·d_obs·h/D′(z_c),
so d ln width_z/dh = (1/h)·[1 − D″D/D′²]. On the pinned 982-event population ⟨D″D/D′²⟩ ≈ 0.216
(from the verifier's +291 nats/h D′-tracking computation), giving a saturated-regime aggregate of
**≈ N/h·(1 − 0.216) ≈ +1055 nats/h — positive (pushes h up), σ-blind at saturation.** The offset
term competes with the opposite sign where the w_pop-weighted mean candidate sits above mid-window,
scaling ∝ 1/σ² inside the double-clip regime; the single-clip layer contributes with mixed sign.
**The net sign at full dose is therefore NOT pre-stated; the width term's sign and scale are.**

## 3. The budget tension — stated before the toy runs

The ratified accounting at full dose leaves a residual of only **−62 ± 36 nats/h** after α and the
measured J-tilt. A width-term-dominated T_REN ≈ +10³ does not fit that budget additively. Note also
the structural entanglement: d ln width_z/dh = 1/h − [D′-tracking]/N — the same D″D/D′² object that
closed the J-gap. Two live possibilities, both pre-stated:
- **(i) cancellation within T_REN** (offset + boundary terms ≈ −width term at this venue's σ/width
  ratio), leaving a small net compatible with the budget; then T_REN's *dose shape* is the test —
  it must reproduce T_res's measured decelerating decay to claim ownership.
- **(ii) genuine non-additivity of ablations** — the J and renormalization defects do not commute
  (both reshape the same per-candidate weights), so single-ablation tilts do not sum to the joint
  defect. Then the additive decomposition picture itself is wrong, and **the joint arm (A-JREN) is
  mandatory before any conclusion** about the repair.

## 4. Pre-stated reads for L0-REN-B (two-sided, frozen here)

Toy: committed harness pattern; arm A = as coded, arm B = identical + per-candidate division by
W_k; full production-mix σ; f ∈ {0.25, 0.5, 1.0}; ≥ 8 seeds; report stacked tilt T_REN(f), implied
MAP shift (BOTH conversions: toy-own curvature AND production σ_post = 0.004386, per the M7
correction), boundary/double-clip population fractions.

| read | rule |
|---|---|
| **R1 magnitude** | CLOSED iff full-dose implied MAP shift (production conversion) ∈ [−1e-3, +1e-3]; LIVE outside |
| **R2 dose shape** | OWNS-SHAPE iff T_REN(0.25→0.5) and T_REN(0.5→1.0) match the measured T_res steps (−550, −212 nats/h) within ±150 each; WRONG-SHAPE if either differs by >±300; PARTIAL-SHAPE between |
| **R3 budget** | CONSISTENT iff T_REN(1.0) ∈ −62 ± 150; **BUDGET-TENSION otherwise → possibility (ii), A-JREN mandatory, and no single-term ownership claim may be made** |
| **R-sign** | reported, not read (net sign not pre-stated, §2) |

Tolerances: ±150 = 3× the pooled level SEs (49/46/36) rounded up for toy-transfer error; ±300 =
2× that. The 1e-3 band is the parent prereg's registered L0 closure threshold, unchanged.

*No repair proposed. A-REN registration is drafted in parallel (row #105 item 2) but registers
nothing until the author sees these reads filled.*
