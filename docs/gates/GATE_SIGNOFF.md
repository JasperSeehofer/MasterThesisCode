# Phase-1 Scientific-Soundness Gate — SIGN-OFF (2026-07-02/03)

**Verdict: GATE PASSED. The Phase-2 multi-seed production campaign is unlocked.**
All eleven gate items (G1–G11) are complete with committed artifacts; every
estimator-calibration defect found was fixed under the physics-change protocol with
regression tests; the remaining error budget consists of quoted model-scope choices.
Branch: `physics/derail-completion-4pi` (PR #18), suite 718 passed / mypy / ruff clean.

## Item-by-item

| Item | Outcome | Artifact / commit |
|---|---|---|
| G1 β_G discrete-sum check (P0) | FAIL for 'global' mode: real −17% residual h-tilt after the expected n_gal∝h³ factor; local modes immune | `G1_beta_g_check.md` · `8c04544` |
| G2a 4π sky marginal | Derivation confirmed + sin θ_det Jacobian deviation found and FIXED | `docs/derivations/G2a…` · `4a259b7` |
| G2b volume_deconv prior | CONFIRMED; σ_z² Eddington law derived, matches measured biases; z≥0 clamp FIXED | `docs/derivations/G2b…` · `4a259b7` |
| G2c Gray mapping | CONFIRMED; 'local_ratio' = deliberate deviation from A.10 (nearest analogue A.20) — paper wording fixed | `docs/derivations/G2c…` |
| G2d Eddington-in-M | IMPLEMENTED (user decision): exact moment-matched rate prior; tests caught the sign flip at the kappa_cap roll-off; 2D mean −0.020 → toward truth | `docs/derivations/G2d…` · `4d780f0` |
| G3 ablation cube | Two-factor decomposition: volume kernel de-rails (0.60→0.76), local denominator removes the +0.03 rest (0.73); `volume_global` diagnostic added | `G3_ablation_cube.json` · `0e67b88`/`45c9ff2` |
| G4 P–P harness + seeding | d2 harness promoted to `master_thesis_code.validation.pp_coverage` (verifier: PASS, zero fidelity issues); 2D MC denominator seeded, `--seed` reaches inference | `d1cff04` · `2f094d3` |
| G5 external codes | gwcosmo: rail bugs absent, bare-z-pdf under untested assumption, NO known-truth validation at σ_z/z~0.7 anywhere; CHIMERA/icarogw/DarkSirensStat volume-weight (corroborates the fix) → two-paper strategy STANDS | `G5a/G5b_…md` |
| G6 starvation post-mortem | CLOSED: every 'starved' config was prior-inconsistent; numerator-only tilt still rails (negative control); consistent estimator calibrated | `G6_starvation_postmortem.md` · `2a6dea7` |
| G7 systematics budget | 16-row table; calibration rows all FIXED; quoted rows are model-scope (Ω_m fiducial, M1 shape) | `G7_systematics_budget.md` · `e82711c` |
| G8 inner product | **dt² missing — fixed**: SNR was physical/10, CRB σ ×10; five evidence lines incl. FFT-free Parseval | `docs/derivations/G8…` · `fcc49c4` |
| G9 timeouts | 0.6–1.25%/stage measured; parameter logging + message fix + separatrix guard landed | `G9_timeout_scan.md` · `4d1c27a` |
| G10 Fisher κ gate | κ ≤ 1e14 enforced (was log-only) | `d17230d` |
| G11 Ω_m fiducial | 0.2726, population-matched (Barausse 2012); C_NORM re-pinned 2.750; Planck case quoted (+1.5–3%) | `bdf5339` |
| z_cmb (G7 row 7) | Fix cherry-picked (`18e9608`, was never on main) + catalogue REBUILT from raw GLADE+ (99.9% rows shifted, median |Δz| 6e-4; z_helio kept as `.zhelio_20260702`) | rebuild verified 2026-07-02 |

## Campaign preconditions (all met) and campaign-time actions

Met: physical SNR semantics (dt²); calibrated default estimator (volume_deconv, both channels
prior-consistent); deterministic inference (seeded); population-consistent cosmology; z_cmb
catalogue; timeout instrumentation; validation harness in-package.
At campaign time: timeout histograms by (M, e₀, p₀); finer-grid pass on the final posterior;
count κ-gate exclusions; per-seed `pp_coverage` runs; PRE_SCREEN_SNR_FACTOR re-check at the new
SNR scale (population deepens to z ≲ 1.5 → longer waveforms, more events).

## Open, non-blocking

- Cluster confirmation 5698617/5698618 still queued (Paper-A polish at the pre-dt² SNR scale —
  scientifically valid for the de-rail claim; escalate 2026-07-05).
- Full 4-estimator (completion+interlopers) pp_coverage variant — follow-up extension.
- CHANGELOG line for G4b; committed de-rail/cube baselines refer to the z_helio catalogue.
- PR #18 merge to main after CI re-run on the accumulated gate commits.
