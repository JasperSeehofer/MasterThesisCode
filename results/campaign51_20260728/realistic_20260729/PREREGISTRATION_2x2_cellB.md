# Pre-registration — 2×2 cell B: unscattered #51 catalogue through the #53 estimator

Registered 2026-07-30, BEFORE submission. Per `RUNBOOK_NEXT_SESSION_6.md` §6.
Gate A1 confirmed C6 (the σ→0 control ran `generator_marginal`, metadata
`run_20260729_seed61000/sig0_control/run_metadata_0.json`, job 6092537), so no
estimator control exists and this test is required.

## The run

One CPU evaluate array (41 h-points, canonical grid) + combine on the cluster:

- RUN_DIR: `$WS/run_20260729_seed61000/estimatorB_2x2/`
- Inputs: seed61000's existing `prepared_cramer_rao_bounds.csv` (symlink) and
  `injections/` pool (symlink) — identical to cells A and C.
- Catalogue: the **unscattered parent** (`reduced_galaxy_catalogue.csv` on the
  cluster, sha256 `7af3f4f4a2…`) — no `OBSERVED_CATALOGUE` export, so the
  scatter guards no-op (`bayesian_statistics.py:310`). **No code change.**
- Estimator: `NORMALIZATION_MODE=absolute_marginal`,
  `HOST_Z_KERNEL=volume_deconv`, `HOST_MASS_KERNEL=auto` (the #53 pairing).
- Code: cluster repo at `7fd60bb` (main, clean) — the same commit the #53
  evaluates ran at.

## The 2×2

|             | point / generator_marginal    | volume_deconv / absolute_marginal |
|-------------|-------------------------------|-----------------------------------|
| unscattered | A = #51: 1D 0.7299, 2D 0.7300 | **B = this run**                  |
| scattered   | forbidden by guard            | C = #53 r1–r5: 1D 0.732, 2D 0.813 |

**B − A = estimator effect. C − B = scatter effect.**
Read `map_h`, `map_h_2d`, and the per-class summed profiles.

## Pre-registered readings (verbatim from runbook §6)

- **Estimator owns it**: B's 2D MAP ≈ 0.78–0.82 and B's in-cat class argmax
  ≈ 0.86 **even with exact host redshifts** ⇒ the realistic host-observation
  model is largely exonerated and the target is the host-z kernel's population
  weight (C7).
- **Scatter owns it**: B ≈ A (2D 0.730, in-cat argmax 0.730) ⇒ the estimator
  switch is inert and the mass-window/completion imbalance under scatter is the
  whole story.
- **Mixed**: read the split directly off B, in nats per class.

Secondary pre-registered reads (added at registration time, before results):

- The per-class C1 analog (Σ Δ ln p_i, 0.73→0.81, both channels) computed with
  `attack_c1_c5.py` conventions on B's per-event posteriors.
- B's in-cat per-event argmax distribution over [0.60, 0.86] (C5 analog) — the
  0.86-edge fraction with **exact** host redshifts. C7 predicts this collapses
  toward the idealized ~5% (a δ-width kernel input... but note volume_deconv
  still integrates against the parent's z_error column even for exact z, so a
  nonzero shift is possible; direction must be UP if C7's mechanism is real).
- w_G(h): expected bit-identical to the #53 runs (pure quadrature, no
  catalogue input — handoff §6). If it differs, that itself is a finding.
