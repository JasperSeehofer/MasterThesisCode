# Project State

A short, human-readable snapshot of where the project is, for continuity across sessions and
machines. Detailed, ephemeral working notes are kept out of the repository by design (see the
`.gitignore` note on internal planning state); this file is the curated, durable surface.

**Last updated:** 2026-07-26

## What works today

- End-to-end EMRI simulation → SNR + Cramér–Rao bounds (GPU / cluster), and the CPU Bayesian H₀
  inference pipeline over the GLADE+ catalogue with completeness correction.
- Full CPU test suite green; `ruff` + `mypy` clean; docs and interactive figures deploy to GitHub Pages.
- `main` reflects the current, soundness-verified pipeline, including: zero-host pure-completion
  fallback, deep (z ≤ 1.5) population support, peculiar-velocity marginalization in the host-z
  kernel, an exact semi-analytic 2D denominator, and a value-preserving batched/fused likelihood
  evaluation (~3.8× faster).

## Current focus

- **Campaign NO-GO LIFTED (2026-07-26), on the valid-4 basis.** The five-seed production-stack
  campaign (jobs 6044799–6044808, code @ `6dae9d3`, `generator_marginal` + `--pdet_z_resolved`,
  41-pt grid) passed all pre-registered criteria on seeds 1000/2000/3000/90000: bias
  −0.0003 ± 0.0004 (base channel), width χ²=8.0/3.7 both VALID, MAPs interior, both channels
  clean. seed900 was **dropped from the registered set (author-ratified)** for a diagnosed
  input-provenance defect — see below — not an estimator failure. See
  `results/lcat_h_dependence_20260725/MULTISEED_READOUT_20260726.md` (incl. "Author
  ratification" section).
- **Production defaults flipped** (`[PHYSICS]` `ce6338e`): `--normalization_mode` default is
  now `generator_marginal`, `--pdet_z_resolved` defaults to `True`
  (`--no-pdet_z_resolved` for legacy pooled behaviour). Library defaults match. 1017 tests green.
- **seed900 root cause:** `$WS/run_20260703_seed900/simulations/injections` was mis-populated
  with a bespoke 204-injection pool (`injection_20260703-112746_seed46910`) instead of the
  canonical `injection_pool_depth15_50k` (50k events); the prodstack run inherited the broken
  symlink, undersampling the z-resolved survival estimator (ESS min/median 6/55, 57.6% of
  sky-band cells below the ESS floor) and railing the venue to 0.86. Fix submitted 2026-07-26:
  `run_20260726_seed900_fixpool`, eval job 6051189, combine 6051190, canonical pool relinked
  and provenance-verified (500 CSVs, era-consistent).
- **Merge to main is additionally gated** on an independent adversarial review (math + physics +
  anti-tuning audit) of the estimator chain, ordered by the author 2026-07-26 — in progress,
  `results/redteam_20260726/` pending. Issue #30 is **closed**; the gate remaining before
  merge/manuscript is the redteam verdict (+ the seed900 fixpool restoring the registered n=5
  test, non-blocking robustness check).
- **P–P impostor-capable harness extension** in progress on branch `feat/pp-impostor-harness`
  (verification hardening, non-blocking).
- **2026-07-31 (quick task 260731-14d, on `physics/kernel-soft-membership`): evaluate-path
  instrumentation landed** (`9522467`/`b287670`/`234890f`): per-class per-h Σ ln p_i logging both
  channels, w_G at 7 s.f., P6 host-recovery counter with pruned-frame index translation +
  regression test, derivation-doc P6 claim corrected. Output-invariant (verified); 1170 tests
  green. Ordered by the Gate B/C adjudication (`results/campaign51_20260728/realistic_20260729/
  gate_b_20260730/ADJUDICATION_20260730.md` §5.2) after the 2026-07-30 gates executed on the #53
  2D-bias claim set (C3/C5/C7/C8 → FINDING; new C9 w_G ×2.5 mis-calibration z=−11.86; cell B 2×2
  jobs 6101146/6101147 pending). Caveat (executor-flagged): the P6 counter's scattered path
  assumes injection and evaluation share M_min/M_max/z_max — true for the standard main.py
  workflow, documented inline.

### Resolved (2026-07-25 investigation, superseded by the 2026-07-26 milestone below)

- **Deep-venue rail (issue #30)** — the EXP-40 re-evaluation (`run_20260719_seed1000_exp40`,
  main @ ba2b381, #29 fallback active) confirmed the seed1000 posterior still rails at the lower
  grid edge (MAP h=0.60, both channels). Post-fix diagnostics re-attribute the rail: ~82% of the
  tilt is the host-found `L_cat` term (the 57.7% completion-fallback events are nearly h-inert),
  and z ≤ 0.3 subsets still rail — weakening pure depth truncation, strengthening the
  L_cat/Gray-mixture estimator path. See
  `results/campaign_phase2_runs/run_20260719_seed1000_exp40/FINDINGS_EXP40_20260725.md`.
- **z_cut truncation scan (2026-07-25): ALL RAIL.** Consistently truncated re-evals
  (`--max_redshift` with B_num domain-matched to D(h) — `[PHYSICS]` 7d3573d + 276c8c7 on
  `feat/max-redshift-cli`) at z_cut ∈ {0.2, 0.3, 0.5} on the same seed1000 CRB all rail at
  h = 0.60 in both channels — **depth truncation (issue #30 option b) is empirically dead**.
  The untruncated z ≤ 0.2 *subset* closes at 0.729 while the truncated z_cut = 0.2 re-eval
  rails, isolating the rail in the h-dependence of the truncated selection/normalization
  structure (w_G = β_G/D) interacting with L_cat. Both findings were resolved by the
  `generator_marginal` + `--pdet_z_resolved` estimator redesign below.

## Known open questions (tracked honestly)

- Whether the H₀ posterior peak is fully information-driven versus partly a normalization effect in
  the deepest-incompleteness regime — under investigation (`docs/H0_BIAS_RESOLUTION.md`).
- Two alternative host-kernel normalization modes (`volume_trunc`, `mass_trunc`) were implemented
  and empirically rejected; they are retained as documented, non-default experimental modes.
- Standing modelling choices are documented in `docs/source/limitations.rst` (flat-ΛCDM distance
  integrals, cosmology-constant vintage matched to the mock universe, redshift-uncertainty scaling).

## Milestone 2026-07-26: the deep venue closes at truth

- **`generator_marginal` + `--pdet_z_resolved`** (branch `physics/absolute-mass-marginal`,
  `[PHYSICS]` commits `8fbb21e` + `a608c4f`): seed1000 MAP = **0.7300 = truth in both
  channels**, sharp broad-based peak, 1017 tests green. Fully derivation-backed chain
  (generator-consistent normalization n̂_w = W_cat/V_f + D_gen; z-resolved survival
  S(d_L|z) in u = ln(1+z); point/point pairing verified against the generator — the mock
  draws catalogue z verbatim). Two-sided mechanism validation: FIX-2-alone measured
  −68.75 ln vs −69 predicted. No truth-referencing constants anywhere. For REAL data the
  photo-z kernel must return (point/point is generator-exact for the mock only).
- Remaining gates at the time of writing: dense-core peak width/σ(h) (done, χ² VALID above),
  re-registered seed600 shallow gate (PASS, `SEED600_GATE_REGISTRATION.md`), 41-pt full grid
  (done), multi-seed (900/1000/2000/3000/90000) residual-bias measurement (done — valid-4 PASS,
  seed900 dropped for provenance defect, see "Current focus" above). Outstanding before merge:
  the independent adversarial (redteam) review.

## Next (2026-07-26)

- **Redteam adversarial review** (math + physics + anti-tuning audit of the estimator chain) —
  in progress, `results/redteam_20260726/` pending. Merge-to-main is gated on its verdict.
- **seed900 fixpool re-run** — `run_20260726_seed900_fixpool`, eval job 6051189, combine 6051190
  submitted 2026-07-26; restores the registered n=5 multi-seed test (non-blocking robustness
  check, campaign verdict already stands on the valid-4 basis).
- **P–P impostor-capable harness extension** on branch `feat/pp-impostor-harness` — in progress.
- Workspace note (below) still applies: copy finals off before 2026-09-23.

## Prior scoping (2026-07-25, kept for the record — resolved by the milestone above)

- **Rail mechanism identified (2026-07-25): host misassociation.** Two independent
  investigations (empirical per-event decomposition, validated to ≤4.5e-13 against cluster
  diagnostics; structural audit vs Gray A9/Gair 2023/gwcosmo v2) show 91–100% of each rail
  event's tilt is the numerator GW-likelihood × host-z overlap: candidate balls contain only
  foreground galaxies (preferred h* ≈ 0.42–0.48, below the grid). volume_deconv is exonerated
  (exactly h-invariant); the ball-local selection denominator is a real-but-secondary
  discrepancy vs the references (1–14%). See `results/lcat_h_dependence_20260725/SYNTHESIS.md`.
- Decide and implement the **estimator redesign** (author physics decision, /physics-change):
  the fix domain is per-event catalogue-vs-dark weighting (Gray membership mixture /
  non-self-normalized catalogue mass) so impostor-only balls defer to the completion term.
  Campaign relaunch stays NO-GO until a redesigned estimator closes on the deep venue.
- Workspace note: `ws_extend` used the **last** available extension (expires 2026-09-23) — copy
  finals off before then.
