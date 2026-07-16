# P–P coverage at order-unity σ_z/z — referee blocker REF-P001/S006 (2026-07-03)

**Question (Paper A referee):** does the volume-kernel calibration claim survive when
σ_z/z is order unity, i.e. beyond the committed anchor run (σ_z = 0.035, σ_z/z ≈ 0.1–0.2)?

**Setup:** `master_thesis_code.validation.pp_coverage` (G4b harness, unchanged code),
250 realizations × 250 events, truths {0.62, 0.72, 0.84} (incl. near-grid-edge),
paired master seed 20260701 across kernels, σ_z ∈ {0.10, 0.15, 0.25}.
Typical event z in the harness ≈ 0.18–0.35, so σ_z/z ≈ 0.3–0.6 / 0.5–0.8 / 0.7–1.4.
Binomial 1σ on cov68 at n=250 ≈ ±0.03.

## Results

| σ_z | kernel | cov68 (h=.62/.72/.84) | cov90 | rail | MAP bias |
|---|---|---|---|---|---|
| 0.10 | bare | 0.00 / 0.00 / 0.00 | 0 / 0 / 0 | 1.00 / 1.00 / 0.01 | −0.020 / −0.120 / −0.182 |
| 0.10 | volume | 0.66 / 0.68 / 0.62 | .91/.93/.86 | .14/.00/.14 | +0.002 / −0.002 / −0.003 |
| 0.15 | bare | 0.00 / 0.00 / 0.00 | 0 / 0 / 0 | 1.00 all | −0.020 / −0.120 / −0.240 |
| 0.15 | volume | 0.70 / 0.65 / 0.70 | .90/.93/.93 | .12/.00/.28 | +0.011 / +0.004 / −0.002 |
| 0.25 | bare | 0.00 / 0.00 / 0.00 | 0 / 0 / 0 | 1.00 all | −0.020 / −0.119 / −0.240 |
| 0.25 | volume | 0.33 / 0.44 / 0.75 | .60/.62/.96 | .03/.02/.60 | **+0.055 / +0.040** / +0.005 |

## Verdict

1. **Bare kernel is unusable at order-unity σ_z/z**: rails to the lower grid edge in
   ~100% of realizations at every σ_z ≥ 0.10, zero coverage, bias up to −0.24 —
   the Eddington-in-z term −σ_z²·d ln(dV_c/dz)/dz grows without bound.
2. **Volume kernel stays near-nominal up to σ_z/z ≈ 0.5–0.8** (σ_z ≤ 0.15):
   cov68 = 0.62–0.70 (nominal 0.68 within binomial error), cov90 = 0.86–0.93,
   |MAP bias| ≤ 0.011.
3. **Volume kernel degrades at σ_z/z ≳ 1** (σ_z = 0.25): 68% coverage drops to
   0.33–0.44 at h_true = 0.62/0.72 with systematic +0.04–0.06 MAP bias; the
   h_true = 0.84 row is dominated by the upper grid edge (rail 0.60, h_max = 0.86).

**Paper A framing:** the calibration claim gets a *measured* validity boundary —
volume-kernel coverage is demonstrated for σ_z/z ≲ 0.8 and demonstrably fails by
σ_z/z ≳ 1 — replacing the asserted decisiveness the referee dinged (REF-P001/S006).

Provenance: JSONs + logs in this directory; harness commit = repo state on branch
`physics/campaign-depth-pv` @ post-PR#21 main (`1caf750`), pp_coverage.py unchanged
since G4b promotion (`d1cff04`).
