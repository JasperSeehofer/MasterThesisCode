# Per-mode combined posteriors — de-rail matrix evidence (seed600, 494-event subsample)

Rescued 2026-07-02 from ephemeral `/tmp/seed600_local/crux_ws/simulations/posteriors_*`
(the working directory of the 2026-07-01 de-rail demonstration; full copy at
`~/data-backups/seed600_local_derail_20260702/`). These are the full per-h posteriors behind
`../derail_matrix_results.json` — the evidence for the H₀ de-rail resolution.

| dir | estimator | MAP | verdict |
|---|---|---|---|
| `prod/` | production pre-4π (peak-density completion, global denominator, bare-Gaussian numerator) | 0.86 | railed ↑ |
| `prod_global/` | + 1/(4π) completion sky-marginal only (cb16142) | 0.60 | railed ↓ (sign flip) |
| `local_ratio/` | + Gray A.9/A.10 local ratio-of-sums (fix #2, 6d4c4e1) | 0.73 | peaked |
| `volume_deconv/` | + dV_c/(1+z) host-z prior deconvolution (fix #1, 6d4c4e1) | 0.73 | peaked |
| `catonly/` | diagnostic: completion term dropped, local self-normalized ratio | 0.73 | peaked |

Data: real seed600 CRB subsample (494 events, `prepared_crb_sub400.csv`), fixed 8-column GLADE+
reduced catalogue, injected truth h = 0.73. Code: branch `physics/derail-completion-4pi`
(cb16142 + 6d4c4e1), production `BayesianStatistics().evaluate()` code path.

Each dir: `combined_posterior.json` + per-h `h_*.json` (+ `comparison_table.md`,
`diagnostic_report.md` where produced). The large per-event with-BH-mass posteriors (474 MB)
are only in the home backup, not committed.
