# Phase 49 — F4: Smooth Kernel p_det Estimator (Farr 2019)

## Status

`READY-TO-PLAN` — Mechanism diagnostic complete; F4 justified by test_29 results.

## Decision gate outcome (test_29, commit `995b8ce`)

| Mechanism | Spike count | Σ(Δp)² share |
|---|---|---|
| A — d_L motion across fixed bin edges | 19 of 24 | **96.4%** |
| B — SNR threshold crossings | 5 of 24 | 3.6% |

The handoff hypothesised B as dominant; diagnostic refuted that. The dominant
residual after F1 is **mechanism A** — a d_L-motion sub-case not addressed by
the h-stable-edges fix. Injections whose `d_L_i(h) = d_L(z_i, h)` crosses a
fixed bin edge as h shifts cause integer-count jumps in n_total and n_det,
producing Δp_det ≈ 1/n_bin ≈ 0.01–0.05 per step.

F4 closes both A and B by eliminating fixed bin edges entirely.

## Motivation: why histogram is structurally limited

F1 stabilised the edges (closed the drift sub-case) but cannot close the motion
sub-case without infinite bins. The histogram estimator has:

```
p_det[i,j] = n_det[i,j] / n_total[i,j]    # step function in (d_L, M_z, h)
```

Any h-step that moves one injection across a fixed edge produces
|Δp_det| ≈ 1/n_bin_total at that bin, which feeds directly into ΣlogL as
coherent noise. F4 replaces this with the **Nadaraya-Watson kernel estimator**:

```
p_det(d_L_q, M_q, h) = Σ_k K(d_L_k(h), M_k; d_L_q, M_q) · det_k(h)
                       / Σ_k K(d_L_k(h), M_k; d_L_q, M_q)
```

where K is a smooth kernel. Because K is continuous in d_L_k(h), each
injection's contribution varies smoothly as h shifts — no integer-count jumps.
This is the "per-injection" form in Farr (2019) arXiv:1904.10879 Sec. III,
adapted from selection-function normalisation to conditional p_det.

## Physics change — formal presentation

**Old estimator (histogram):**
```
p_det[i,j](h) = N_det[i,j](h) / N_total[i,j](h)
```
where indices i, j are the (d_L, M_z) histogram bins, and N are integer counts.

**New estimator (Nadaraya-Watson, smooth kernel):**
```
p̂_det(d_L_q, M_q, h) = Σ_k w_k(d_L_q, M_q, h) · 1[SNR_k(h) ≥ 20]
                        / Σ_k w_k(d_L_q, M_q, h)
```
with kernel weights
```
w_k = exp[ -½ · ((d_L_k(h) - d_L_q) / σ_dl)²
           -½ · ((log M_k - log M_q) / σ_M)² ]
```

- `d_L_k(h) = d_L(z_k, h)` — shifts smoothly with h (continuous)
- `M_k = M_k_source · (1 + z_k)` — observer-frame, h-independent
- `σ_dl`, `σ_M` — bandwidths, chosen by Scott's rule on the injection sample

**References:**
- Farr (2019) arXiv:1904.10879 Sec. III — per-injection selection-function form
- Mandel, Farr & Gair (2019) arXiv:1809.02063 Eq. 18 — importance-sampling weight
- Nadaraya (1964); Watson (1964) — original kernel regression estimator

**Dimensional analysis:**
- All arguments (d_L_q, d_L_k, σ_dl) in Gpc — consistent
- M_q, M_k, σ_M arguments in log-M_sun — consistent
- Output: dimensionless probability ∈ [0, 1] — consistent

**Limiting case:**
- σ_dl → 0, σ_M → 0 (infinitely sharp): recovers nearest-neighbour (single
  injection) estimate; noisy but unbiased.
- σ_dl → ∞, σ_M → ∞: global fraction of detections; equal to 1D marginal.
- At true h, p̂_det → p_det(d_L_q, M_q) as N_inj → ∞ for fixed bandwidth.

## Implementation plan

### Files changed

1. **`master_thesis_code/bayesian_inference/simulation_detection_probability.py`**
   - `_build_grid_2d`: replace `np.histogram2d` blocks with kernel-weighted sum
   - `_build_grid_1d`: replace `np.histogram` block with 1D kernel-weighted sum
   - `__init__`: add `bandwidth_scale: float = 1.0` parameter (Scott's rule multiplier)
   - `_compute_bandwidths()`: new private method, returns (σ_dl, σ_M) per Scott's rule
   - Quality flags: replace `n_total`, `n_det` integer counts with `n_eff` (Kish formula)

2. **`master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py`**
   - Update the `test_dl_edges_identical_across_two_trial_h` test — bin edges no
     longer exist; replace with monotonicity test on p̂_det(h) over the smooth region
   - Add `test_p_det_continuity`: assert |Δp_det(h+Δh) - Δp_det(h)| < ε for
     Δh=0.001, ε=0.005 at a fixed (d_L_q, M_q) probe point.
   - Add `test_p_det_smoothness_at_threshold`: probe across a known SNR-crossing h*
     and assert no step exceeds 0.002 (the B-mechanism bound from test_29).
   - Retain all other existing tests.

### Key implementation note — performance

Naive O(N_inj × N_grid) kernel evaluation: 105k × 2400 ≈ 252M ops per h value.
For 50 cached h values that is 12.6B ops — too slow.

Use **truncated kernel**: set w_k = 0 for |d_L_k(h) - d_L_q| > 3 σ_dl. With
σ_dl ≈ dl_global_max / (2 · dl_bins) ≈ 0.031 Gpc and a 3σ cutoff, each grid
center sees ~1/20 of injections on average → 252M / 20 ≈ 12M ops per h, 600M
total. Acceptable on CPU in ~1–2 min.

Implementation: for each grid row (fixed d_L_center), pre-sort injections by
d_L_k(h) and use `np.searchsorted` to find the contributing window. Inner loop
over M_z is vectorised with `np.exp`.

### Scott's rule bandwidth

For d_L (linear scale, n = 105500):
```
σ_dl = n^(-1/6) · std(d_L_k(h))     # 1D Scott's rule, adapted to 2D as n^(-1/(d+4))
```

For M_z (log scale):
```
σ_M = n^(-1/6) · std(log10(M_k))
```

Scott's rule is a starting point; `bandwidth_scale` parameter allows tuning if
the posterior still shows residual spikes. Default = 1.0 (untuned). Run
diagnostic analogous to test_29 post-F4 to confirm Σ(Δp)² < 0.001 total.

### Backward compatibility

- `SimulationDetectionProbability` public interface unchanged (same `__init__`
  args except the new optional `bandwidth_scale`).
- `detection_probability_interpolated` and `detection_probability_without_bh_mass_interpolated_zero_fill` unchanged.
- The quality flags dict keys change: `n_total`/`n_det` removed; `n_eff`
  remains. No external code reads `_quality_flags` directly (it is a private
  diagnostic attribute).

## Verification steps

1. **Unit tests pass**: `uv run pytest -m "not gpu and not slow"` — all 523+
   existing tests green; new continuity tests green.
2. **mypy clean**: `uv run mypy master_thesis_code/ master_thesis_code_test/`.
3. **Continuity check local**: run `test_29` analogue (test_30_f4_smoothness.py)
   post-F4 — target Σ(Δp)² < 0.01 (10× reduction from current 1.54) for the
   same 48 query points.
4. **Cluster validation**: re-run production inference with the F4 estimator.
   Compare `combined_posterior.json` MAP and σ_boot against F1-PARTIAL results.
   Target: MAP within 0.5σ of truth (h=0.73); σ_boot > 0.002 (no more pinned
   bootstrap); posterior visually unimodal and smooth.

## Physics Change Protocol gate

This plan requires the `/physics-change` protocol before any code is written.
The five-item checklist (old formula, new formula, reference, dimensional
analysis, limiting case) is answered above. Present this plan to the user for
approval, then proceed to implementation.

## Sequencing

```
F4a (closes A, primary):
  → Implement kernel estimator in _build_grid_2d + _build_grid_1d
  → Unit tests + mypy
  → test_30_f4_smoothness.py (local continuity diagnostic)
  → /check gate
  → [PHYSICS] commit
  → Cluster job

F4b (closes B, minor, optional):
  → Replace hard det_k with smooth Φ((SNR_k - 20)/σ_SNR)
  → Requires σ_SNR per injection (from Fisher matrix or fixed 1-σ SNR uncertainty)
  → Defer unless post-F4a cluster validation still shows B spikes
```

## Commit convention

```
[PHYSICS] F4: Nadaraya-Watson kernel p_det estimator (closes A + B mechanisms)
```
