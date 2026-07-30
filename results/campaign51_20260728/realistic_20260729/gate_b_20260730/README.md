# Gate C items 1 + 4 — `w_G` calibration and mixture-leg consistency (2026-07-30)

Target: `GateC-1+4 (w_G / mixture-leg consistency)` per `RUNBOOK_NEXT_SESSION_6.md` §5.
LOCAL only, read-only w.r.t. `master_thesis_code/`. Nothing was submitted anywhere.

## Verdict in one line

`w_G = beta_G/D` is **mis-calibrated by a factor 2.35 at truth**: it uses the
*population-marginal* `p_det` for an in-catalogue population that is Malmquist-selected
to be BH-mass-atypical. The realized in-catalogue host counts (164/3135, both seeds)
reject the delivered `w_G` at **z = -11.9** and match the mass-aware value at **z = +0.21**.

## Scripts (run from the repo root with `.venv/bin/python`)

| file | what it does |
|---|---|
| `g1_basics.py` | re-derives `w_G(h)` at 7 s.f. from `D - beta_Gbar`; empirical in-cat rate, both seeds |
| `g2_catalogue_vs_f.py` | catalogue rate-weighted z-density vs the model's `f_bar(z) p_pop(z)`; `W_cat`, `V_f`, `F`, `n_hat_w` |
| `g3_mass_shape.py` | catalogue rate-weighted BH-mass distribution vs the population mass marginal |
| `g4_wg_tension.py` | identity checks on the diagnostics CSV; tension z-scores; **counterfactual posteriors** with `beta_G -> r(h) beta_G` |
| `g5_leg_consistency.py` | Gate C item 1 level/slope table; completion-only event counts; per-class nats table |
| `g6_fz_pointwise.py` | p_det-free pointwise test: realized `P(in-cat | detected, z)` vs `f_bar(z)` |
| `g7_mass_vs_z.py` | catalogue BH mass vs z (the Malmquist mechanism) |
| `g8_ghost_and_zscores.py` | the "0.0697 ghost"; corrected tension z-scores |

Outputs: `g2_catalogue_summary.json`, `g2_zshape.npz`, `g4_results.json`,
`g4_posterior_curves.json`, `g6_results.json`, `legs.npy`.

## Provenance / staleness

Load-bearing numbers are **staleness-free**: `w_G`, `D`, `beta_Gbar`,
`sum_w_Dg(no_bh/with_bh)` come from the run's own per-h logs
(`seed61000/mixture_leg_log_extract.txt`); host counts come from the cluster
`prepared_cramer_rao_bounds.csv`; `F` comes from the git-tracked, unmodified frozen
`m_th_map_nside32.npy` (sha256 `73a4bbea...`) plus analytic `p_pop` — no catalogue rows.

INDICATIVE only (they read the local `reduced_galaxy_catalogue.csv`, which differs from
the realization parent in `z_error` and therefore in the prune): `W_cat`, `n_hat_w`, and
every catalogue z-/mass-shape table in `g2`, `g3`, `g7`.

## Not a re-opening of anything exonerated

- The **Option-A drift** `beta_G/Sigma_glob` exoneration concerns the *h-slope* of the
  MASS-BLIND ratio. Measured here and CONSISTENT with it: level 1.083, slope -6.3% over
  0.60->0.86. The new content is the MASS-AWARE comparison (`Sigma_glob(with_bh)`),
  which no prior test made.
- **HA** mass-marginalises `D`/`beta_G`/`beta_Gbar` (makes the completion leg 4D) and moves
  the MAP the WRONG way (+0.053). The intervention here is the opposite one: keep the
  completion leg 3D (its population IS mass-typical by construction) and shrink `beta_G`
  to the catalogue's own mass-aware level. It moves the 2D MAP the RIGHT way (-0.069).

---

`ADJUDICATION_20260730.md` — the Gate B/C adjudication of record (fable/xhigh, 2026-07-30): per-claim verdicts C1-C11, Gate C ranking, cell-B mechanical readout, fix routing, and the claim-file edit list.
