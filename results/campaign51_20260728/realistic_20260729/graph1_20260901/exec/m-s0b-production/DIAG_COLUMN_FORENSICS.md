# `L_cat_no_bh` exact-zero → tiny-positive: diagnostic-column forensics

**Verdict: DIAGNOSTIC-ONLY (physics identical).** No formula reaching the S0-B truth-node's
resolved config (`normalization_mode=absolute_marginal`, `catalogue_leg_1d_mass_aware=off`,
`theta_b=0.0`/`theta_s=1.0`) changed unconditionally between `d04d9dc9` and `081b1f28`. The
likelihood quantity actually used downstream, `combined_no_bh`, is unchanged to floating-point
noise (max 3.7e-9 abs / 1.6e-7 rel over the 157 moved events) — only the raw sub-underflow
diagnostic crossed the hard-zero boundary.

## 1. Assignment sites (both commits, direct assignment, no floor/clip)

- `d04d9dc9` (old): `bayesian_statistics.py:5645` — `"L_cat_no_bh": L_cat_without_bh_mass,`
- `081b1f28` (new): `bayesian_statistics.py:6791` — `"L_cat_no_bh": L_cat_without_bh_mass,`

Both are a bare dict assignment of `L_cat_without_bh_mass`, the same local computed a few lines
above via (resolved mode `absolute_marginal` ≡ branch `"global"/"volume_global"/"absolute_marginal"`):

```python
cat_num_sum_no_bh = weighted_sum([r[0] for r in all_results_without_bh], weights_without_bh)
L_cat_without_bh_mass = cat_num_sum_no_bh / global_denom_no_bh if global_denom_no_bh > 0 else 0.0
```

(old `:5086-5089` = new `:6216-6219`, **byte-identical**, confirmed by direct excerpt diff). No
`np.maximum`/`clip`/floor wraps this expression at either commit, and no code sits between this
line and the diagnostics-dict write at either commit that could floor/clip it. Ruling out the
"the diagnostic used to report a floored/post-threshold quantity" branch of the hypothesis.

## 2. `combined_no_bh` construction, resolved config

Old `:5598` uses `beta_G_phi`; new `:6742` renames the same slot to `_cat_num_weight_no_bh` —
cosmetic rename only, values traced identical (`D_tilde_phi`/`alpha_G_phi` bit-identical for all
157 moved events per §4). No structural change to the combine formula.

## 3. `single_host_likelihood_batch` (produces `r[0]`, the per-host catalogue numerator feeding
`weighted_sum` above) — full diff, `d04d9dc9:6753-7405` vs `081b1f28:8085-8811`

Complete diff is 88 lines, **100% gated**:
- A `theta_b != 0.0 or theta_s != 1.0` block (new `θ`-hook site 2.2) — truth node runs
  `theta=(0.0, 1.0)`, branch not entered.
- A `catalogue_leg_1d_mass_aware == "on"` branch (T2.3 mass-aware 1D leg) — confirmed `off` in
  the S0-B config (CLASS_COUNT_FORENSICS.md §4(a)); falls through to the original
  `np.interp(host_z, _z_s, _s_phi)` / `np.interp(y_num_nodes, _z_s, _s_phi)` lines, unchanged.

`gw_3d = _mvn_pdf(...)` (the Gaussian-tail kernel that actually underflows) is byte-identical at
old `:304-305` / new `:331-332`. No unconditional change touches it.

## 4. Data verification (both CSVs, 157 moved events, h=0.73)

`headreadout_20260827/iiib/event_likelihoods.csv` (d04d9dc9) vs
`graph1_20260901/retrieved/s0b_run_20260902/.../node_truth_iiib_sites2.2_nosmear/.../event_likelihoods.csv`
(081b1f28, truth node):

| column | max abs diff | max rel diff | bit-identical count |
|---|---|---|---|
| `combined_no_bh` | 3.73e-09 | 1.58e-07 | 17/157 |
| `B_num` | 3.35e-08 | 1.23e-15 | 151/157 |
| `D_tilde_phi` | 0.0 | 0.0 | 157/157 |
| `alpha_G_phi` | 0.0 | 0.0 | 157/157 |
| `L_cat_no_bh` | 2.31e-08 | — (0 → tiny, ratio undefined) | 0/157 |

`combined_no_bh`'s ~1e-7 relative spread and `B_num`'s ~1e-15 spread are exactly
double-precision-summation-order noise, not a formula change (a real formula change would move
`combined_no_bh` by orders of magnitude, not parts-per-10-million). `D_tilde_phi`/`alpha_G_phi`
— the window/weight quantities — are exactly unchanged.

## 5. Which commit

None: `git log --oneline d04d9dc9..081b1f28 -- .../bayesian_statistics.py` lists 9 commits (see
CLASS_COUNT_FORENSICS.md §4); every one that touches the no-BH 1D catalogue leg does so behind a
flag confirmed `off`/identity in the S0-B run, independently reconfirmed here for
`single_host_likelihood_batch` (§3) and the `absolute_marginal` combine block (§1-2). The
exact-zero→subnormal flip is consistent with a run-to-run floating-point summation-order
perturbation (unrelated code insertions elsewhere in the ~9-commit diff shift array
construction/dispatch order without changing any formula on the resolved path) tipping a
deep Gaussian-tail term across the IEEE-754 underflow boundary — invisible in `combined_no_bh`.
The classifier (`b3_1_pop_measure.py`, hard `L_cat_no_bh == 0` test, CLASS_COUNT_FORENSICS.md
§3) is exquisitely sensitive to exactly this noise; the dirty-tree gap flagged in
CLASS_COUNT_FORENSICS.md §4 is now lower-priority since the physics output did not move.
