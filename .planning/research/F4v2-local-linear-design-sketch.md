# F4-v2 design sketch — local-linear p_det estimator (boundary-bias fix)

Status: DESIGN SKETCH (pre-`/physics-change`). Implementation requires the
physics-change gate (trigger file `simulation_detection_probability.py`).

## Problem (confirmed)
F4 = Nadaraya-Watson (local-constant) kernel regression of binary detection
label Y∈{0,1} on x=(d_L, log10 M_z). At the d_L→0 boundary the one-sided
kernel averages in far-field non-detections → p̂ collapses to ~0.5 where
ground truth = 1.0. O(h) boundary bias. Propagates into D(h)=∫p_det dV_c/dz,
biases H0 MAP high.

## Fix (literature-backed: Fan & Gijbels 1996; Fan-Heckman-Wand 1995)
Replace local-constant with **local-linear** regression. Local-linear is
design-adaptive: the linear term absorbs the local slope, so a one-sided
boundary neighbourhood no longer drags the estimate. Boundary bias O(h)→O(h²),
no variance penalty.

## Where it changes (surgical)
Only the per-grid-cell estimator inside `_build_grid_2d` (≈L581-590) and the
matching block in `_build_grid_1d`. Everything else is UNCHANGED:
- grid support / centers / edges (L529-542)
- Scott bandwidths (`_compute_bandwidths`)
- 3σ searchsorted truncation (L546-570)
- `RegularGridInterpolator` output (L618-624)
- the bridge/extrapolation methods (`..._zero_fill`, the 2D interpolators)
- public interface (`get_dl_max`, `detection_probability_*`) — untouched

Current (local-constant):
```
sum_w     = Σ_k w_k                       # w_k = K(d_L)·K(logM)·w_IS
sum_w_det = Σ_k w_k y_k
p̂_ij      = sum_w_det / sum_w             # <-- boundary-biased ratio
```

## Estimator math — two tiers

### Tier 2 (recommended first: closed-form local-linear on probability scale)
At query center x_q, with local residual u_k = x_k − x_q and weights w_k:
```
S0 = Σ w_k ;  S1 = Σ w_k u_k ;  S2 = Σ w_k u_k u_kᵀ
T0 = Σ w_k y_k ;  T1 = Σ w_k u_k y_k
[a, b]ᵀ = [[S0, S1ᵀ],[S1, S2]]⁻¹ [T0, T1]ᵀ
p̂(x_q) = a                                 # intercept = local-linear estimate
```
- 1D-in-d_L (minimal): 2×2 solve per cell. Recommended — the boundary that
  matters is d_L; M_z has no known-limit boundary.
- Full 2D: 3×3 solve per cell (local-linear in both d_L and logM).
- Vectorizable over M-centers exactly like the current code.
- Clip to [0,1] (local-linear can overshoot slightly near sharp transitions).

### Tier 1 (more robust: local logistic = local-linear on logit)
Local likelihood for Bernoulli Y (Fan-Heckman-Wand 1995): per cell, IRLS fit
of η(x)=logit p(x) ≈ a + b·u. p̂ = expit(a). Inherently bounded in (0,1),
inherits local-linear boundary correction, natural slot for the known limit
(η→+∞ as d_L→0). Cost: small iterative solve per cell (~3-5 IRLS steps).

## Known-boundary anchor (p→1 as d_L→0)
The physical limit is exact. Encode it (any/all of):
1. Pseudo-observations: a few (d_L=0, y=1) points with high weight, so the
   local fit near the edge is pinned to 1. (Tier 1: η=+large.)
2. Monotone projection: enforce p̂ non-increasing in d_L via PAVA (pool-
   adjacent-violators) on each M-column after the fit. Cheap, removes residual
   wiggles, consistent with physics (closer ⇒ more detectable).
3. The existing (0,1) bridge below the first grid center STAYS — but now its
   upper anchor p̂(c0) comes from local-linear (~0.9) not local-constant (~0.5),
   so the bridge descends correctly.

## Constructor / config
Add `estimator: Literal["local_linear","local_logistic","nadaraya_watson"]`
(default `"local_linear"`). Keep `nadaraya_watson` selectable for regression
tests / reproducing the old behaviour. `bandwidth_scale` retained.

## Tests (new + regression)
- **Boundary limit** (hard gate): `p̂(d_L→0) ≥ 0.95`; monotone non-increasing in d_L.
- **Near-field ground truth**: on the injection set, p̂(d_L<0.1 Gpc) within
  ~0.1 of the empirical detection fraction (≈1.0). (Reuses the ground-truth
  harness already written in this investigation.)
- **Smoothness preserved**: re-run `test_30_f4_estimator_smoothness` — Σ(Δp)²
  must stay low (this was F4's win; must not regress).
- **[0,1] bounds**: all grid + interpolated values in [0,1].
- **NW-equivalence escape hatch**: `estimator="nadaraya_watson"` reproduces
  current values (frozen regression).

## Validation ladder (cheap → expensive)
1. Local D(h) proxy (already scripted): old-histogram, F4-NW, F4v2-local-linear —
   F4v2 should match ground-truth near-field and have a D(h) slope close to the
   histogram's (which was ~correct), not the steepened NW slope.
2. Re-run the **closure tests** that H3 passed (h_true=0.65, 0.73) — must still PASS.
3. Re-eval the **seed400 phase50 CRB** → expect MAP to drop from 0.76 toward ~0.73.

## Physics-change protocol checklist (for the gate)
- Old formula: p̂ = Σw_k y_k / Σw_k (Nadaraya-Watson, local-constant).
- New formula: local-linear (intercept of weighted LS) / local-logistic.
- Reference: Fan & Gijbels (1996) *Local Polynomial Modelling*; Fan, Heckman &
  Wand (1995) JASA 90:141 (local likelihood, binary response).
- Dimensional analysis: p_det dimensionless ∈[0,1]; inputs d_L [Gpc], log10 M_z
  [dex] — unchanged.
- Limiting case: p_det(d_L→0)→1 (now satisfied by construction; was violated).
- Commit prefix: `[PHYSICS]`.

## Locked decisions (2026-05-26)
- **D1 = Tier 2**: closed-form local-linear on the probability scale, clipped to
  [0,1]. Escalate to Tier 1 (local-logistic) only if residual issues surface in
  the validation ladder.
- **D2 = d_L only**: local-linear in d_L (2×2 per-cell solve); keep the logM
  direction local-constant (kernel-weighted). The known boundary lives in d_L.
- **D3 = deferred/optional**: Farr-2019 importance-sampled found-injection
  estimator for D(h) is a *cross-check only*, not part of the v2 estimator. Add
  later if an independent D(h) validation is wanted.

## Gating: implementation BLOCKED until cluster job 4944382 confirms net MAP
If the 6-point eval of the old seed200⊕300 CRB through F4 lands ≈0.76 → F4 is
the net mover, premise confirmed, proceed to `/physics-change` → Tier 2.
If it lands ≈0.73 → numerator compensates, premise breaks, re-investigate
before touching the estimator.
