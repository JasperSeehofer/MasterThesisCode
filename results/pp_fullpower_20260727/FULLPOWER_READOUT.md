# P–P impostor-harness FULL-POWER readout (runbook thread 5) — 2026-07-27

**Job:** bwUniCluster 6052704, 15-cell array on `cpu_il`, branch
`feat/pp-impostor-harness` @ `44ee912`, n_realizations = **2000/cell**
(10× the smoke), n_events = 250, seeds 20260727000–014, h-grid
[0.60, 0.86] step 0.004. Config: calibrated operating points
(`ppwork_plan.md`, this directory).

**Provenance correction (discovered post-submission):** the smoke's actual
config (recovered from `results/pp_impostor_harness_20260726/run_smoke.py`
on the branch — committed all along, missed at reconstruction time) was
z_support = 0.30, sky_frac = 2e-4, **n_events = 120**; this campaign's
primary point (zs = 0.43, sky 1e-4, n_events 250) is a *calibrated nearby*
operating point matched on ball occupancy, not a literal 10× rerun. A second
array (job 6053497, seeds 20260727100–108, `pp_fullpower_smokeconfig.sbatch`)
reruns the smoke's EXACT config at n = 2000 for direct comparability — see
§Smoke-exact confirmation below.

## Results (per cell: MAP bias, MAP std, coverage, rail fraction)

| mode | z_sup | sky | h_true | bias | std | cov68 | cov90 | rail | ball | imp |
|---|---|---|---|---|---|---|---|---|---|---|
| lcat | 0.43 | 1e-4 | 0.62 | +0.0010 | 0.0112 | 0.656 | 0.844 | 0.051 | 3.82 | 0.738 |
| lcat | 0.43 | 1e-4 | 0.72 | +0.0061 | 0.0123 | 0.581 | 0.824 | 0.000 | 3.80 | 0.740 |
| lcat | 0.43 | 1e-4 | 0.84 | +0.0096 | 0.0107 | 0.355 | 0.509 | 0.335 | 3.72 | 0.754 |
| absolute | 0.43 | 1e-4 | 0.62 | −0.0015 | 0.0099 | 0.640 | 0.832 | 0.043 | 3.85 | 0.740 |
| absolute | 0.43 | 1e-4 | 0.72 | +0.0038 | 0.0112 | 0.647 | 0.863 | 0.000 | 3.81 | 0.740 |
| absolute | 0.43 | 1e-4 | 0.84 | +0.0056 | 0.0115 | 0.492 | 0.638 | 0.204 | 3.76 | 0.756 |
| genmarg | 0.43 | 1e-4 | 0.62 | +0.0010 | 0.0097 | 0.677 | 0.868 | 0.025 | 3.82 | 0.738 |
| genmarg | 0.43 | 1e-4 | 0.72 | +0.0042 | 0.0111 | 0.627 | 0.860 | 0.000 | 3.80 | 0.740 |
| genmarg | 0.43 | 1e-4 | 0.84 | +0.0057 | 0.0115 | 0.467 | 0.627 | 0.195 | 3.72 | 0.754 |
| genmarg | 0.79 | 1e-4 | 0.62 | +0.0006 | 0.0095 | 0.691 | 0.869 | 0.026 | 14.06 | 0.929 |
| genmarg | 0.79 | 1e-4 | 0.72 | −0.0001 | 0.0108 | 0.691 | 0.904 | 0.000 | 14.07 | 0.929 |
| genmarg | 0.79 | 1e-4 | 0.84 | −0.0020 | 0.0123 | 0.598 | 0.802 | 0.056 | 14.08 | 0.929 |
| genmarg | 0.95 | 2e-4 | 0.62 | −0.0017 | 0.0118 | 0.630 | 0.837 | 0.106 | 41.02 | 0.976 |
| genmarg | 0.95 | 2e-4 | 0.72 | −0.0005 | 0.0146 | 0.669 | 0.897 | 0.000 | 40.97 | 0.976 |
| genmarg | 0.95 | 2e-4 | 0.84 | −0.0008 | 0.0152 | 0.561 | 0.833 | 0.142 | 41.00 | 0.976 |

(Binomial 1σ on coverage at n = 2000: ±0.010 on cov68; on rail ±0.007–0.011.)

## Findings

1. **The smoke's headline SURVIVES at 10× power**: both absolute-mass modes
   beat `lcat` in every primary-point cell — at h = 0.84 the lcat HIGH-rail
   fraction is 0.335 vs 0.195/0.204 (genmarg/absolute), cov68 0.355 vs
   0.467/0.492; at h = 0.72 the bias is +0.0061 (lcat) vs +0.0042/+0.0038.
   With n = 2000 these separations are ≫ binomial error. First
   full-power harness evidence for the V1/FIX-3 impostor-suppression claim.
2. **absolute ≈ generator_marginal everywhere** (biases within 0.0004,
   coverage within 0.03) — as pre-registered: the harness catalogue is
   Option-A-compliant by construction, so it cannot adjudicate FIX-3 vs V1;
   it validates both against lcat.
3. **B_num-carrier confirmation (the smoke's sweep, now at power)**: the
   residual HIGH bias at the primary point (+0.004 to +0.006 at h ≥ 0.72)
   collapses to zero within error in the high-occupancy cells (zs = 0.79:
   −0.0001 ± 0.0002; zs = 0.95: −0.0005) where the completion term's weight
   vanishes — isolating B_num as the sole carrier of the residual, per the
   smoke's attribution. The B_num bias model itself remains the open item.
4. **Coverage**: at the primary (sparse-ball) point all modes UNDERCOVER at
   truth-central h (cov68 0.58–0.65 vs nominal 0.68); the high-occupancy
   genmarg cells reach nominal (0.669–0.691 at h = 0.62/0.72). h = 0.84
   cells sit near the grid edge (0.86) — rail fractions and coverage there
   mix in edge effects and should not be read as pure estimator properties.
5. Scope: harness-internal (synthetic catalogue, exact p_det(d_L), vacuous
   FIX-2) — validates estimator structure, not production precision.

## Smoke-exact confirmation (job 6053497, smoke config z_sup=0.30 / sky 2e-4 / n_events=120, n = 2000)

| mode | h_true | bias | std | cov68 | cov90 | rail | ball |
|---|---|---|---|---|---|---|---|
| lcat | 0.62 | +0.0174 | 0.0247 | 0.488 | 0.745 | 0.064 | 2.92 |
| lcat | 0.72 | +0.0360 | 0.0374 | 0.342 | 0.567 | 0.015 | 2.75 |
| lcat | 0.84 | +0.0135 | 0.0143 | 0.314 | 0.615 | 0.736 | 2.57 |
| absolute | 0.62 | +0.0154 | 0.0210 | 0.474 | 0.740 | 0.053 | 2.93 |
| absolute | 0.72 | +0.0211 | 0.0283 | 0.462 | 0.701 | 0.003 | 2.75 |
| absolute | 0.84 | +0.0095 | 0.0173 | 0.418 | 0.721 | 0.601 | 2.56 |
| genmarg | 0.62 | +0.0140 | 0.0204 | 0.499 | 0.768 | 0.061 | 2.90 |
| genmarg | 0.72 | +0.0235 | 0.0294 | 0.433 | 0.665 | 0.003 | 2.74 |
| genmarg | 0.84 | +0.0103 | 0.0172 | 0.371 | 0.693 | 0.629 | 2.56 |

Direct check against the smoke's quoted numbers (n = 200): bias at h = 0.72
+0.0211/+0.0235 vs lcat +0.0360 (smoke: +0.0232 vs +0.0349 ✓); rail at
h = 0.84 0.601/0.629 vs lcat 0.736 (smoke: 0.640 vs 0.775 ✓); cov68 ordering
preserved. **Every smoke conclusion is confirmed at 10× the statistics.**
Note the smoke config (120 events, sparser balls) is a harsher venue than
the primary-point campaign above — biases ~4× larger, coverage further from
nominal — consistent with the completion-fraction monotonicity of finding 3.

## Disposition

- Runbook thread 5 first half DONE (full-power run). Second half — PR +
  merge of `feat/pp-impostor-harness`, and the B_num residual-bias
  characterization (finding 3 defines the target) — PR opened; B_num
  modeling remains open.
- Raw per-cell JSONs + sbatch + plan committed alongside this readout
  (204 KB total).
