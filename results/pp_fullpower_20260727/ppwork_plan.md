# P–P impostor-harness full-power campaign plan

Branch: `feat/pp-impostor-harness` (commit 44ee912). Harness:
`master_thesis_code/validation/pp_coverage.py`, CLI
`python -m master_thesis_code.validation.pp_coverage`. Fully self-contained
synthetic universe: **no GPU, no galaxy-catalog file, no simulations dir**
(verified: tiny runs executed from a bare worktree with only the venv).

## Reconstructed smoke operating points (calibration, 2026-07-26)

The smoke artifacts were never committed; operating points were reconstructed
from the code's own occupancy math —
`E[ball] = n_galaxies * sky_frac * P(z_true < z_support) + P(host catalogued)`,
with galaxy density `n_gal(z) ∝ (1+z) w_pop(z)` on `[Z_MIN=1e-3, Z_MAX_POP=0.95]`
and effective host-draw density `∝ w_pop(z) p_det(A(z)/h)` — then confirmed
with tiny runs (`--n-realizations 2 --n-events 50 --truths 0.72`, defaults
n_galaxies=200000, sky_frac=1e-4, d50=1.85, sigma_z=0.035, sigma_dl_frac=0.05).

| Point | z_support | sky_frac | measured mean_ball_size | measured impostor_fraction | smoke target |
|---|---|---|---|---|---|
| P1 (primary)        | **0.43** | 1e-4 (default) | 3.75 | 0.733 | "2.5–2.9 cand/ball @ ~78%" |
| P2 (high-impostor)  | **0.79** | 1e-4 (default) | 13.89 | 0.9280 | "14 @ 93%" |
| P3 (extreme)        | **0.95** | **2e-4** | 40.89 | 0.9755 | "41 @ 97.6%" |

Analytic (continuum) predictions at these settings: P1 ball 3.8 @ 0.74,
P2 ball 14.0 @ 0.929, P3 ball 41.0 @ 0.9756.

### Deviations / caveats from the smoke reconstruction

1. **P1 is not exactly reproducible as stated.** The commit message's
   "2.5–2.9 candidates/ball at ~78% impostors" is internally inconsistent
   under the code's occupancy model: an *expected occupancy* (the background
   term `n_galaxies*sky_frac*catalogued_fraction`, i.e. the formula quoted in
   the code and presumably in the smoke note) of 2.5–2.9 corresponds to
   z_support ≈ 0.42–0.44 with impostor fraction 0.73–0.75, while 78%
   impostors would require z_support ≈ 0.47 (total ball ≈ 4.5). I chose
   z_support=0.43: background occupancy 2.8 (inside 2.5–2.9) and measured
   impostor fraction 0.733 ("~78%" read as loose rounding). If the smoke's
   "2.5–2.9" was instead the *total* mean_ball_size, use z_support≈0.36
   (ball 2.67 @ imp 0.69 measured) — even further from 78%.
2. **P3 cannot be a pure z_support variation.** At the defaults the ball
   ceiling is 200000*1e-4 + 1 = 21 candidates @ 95.2% impostors
   (z_support=0.95 = full support). 41 @ 97.6% matches *exactly*
   (E[ball]=41.0, imp=40/41=0.97561) a doubled cap surface density:
   z_support=0.95 with **sky_frac=2e-4** (equivalently n_galaxies=400000).
   The smoke sweep must have varied sky_frac (or n_galaxies) for this point;
   adopted sky_frac=2e-4. Unverifiable which of the two knobs was used —
   they are statistically equivalent for ball occupancy, but n_hat_w/W_cat
   normalizers differ trivially only through catalogue realization noise.
3. **P2 matches exactly** (14.0 @ 92.9% predicted): no deviation.
4. Assumed the smoke used all remaining defaults (n_events=250,
   sigma_z=0.035, sigma_dl_frac=0.05, d50=1.85, w_pdet=0.30, h_step=0.004,
   n_z_quad=160, frozen shared catalogue, membership on true z). Cannot be
   verified against artifacts; these are the code defaults at 44ee912.

## Cost measurement (Step 2)

Dev machine, worktree venv, one cell = one mode × one truth × n_events=250:

- `lcat` @ P1, `--n-realizations 10`: **3.0 s wall** (4.95 s user)
- `generator_marginal` @ P3 (41-cand balls), `--n-realizations 10`: 2.8 s wall

→ ~0.30 s/realization → **~10 min per cell at n_realizations=2000**.
×1.5 margin ≈ 15 min; allowing ~3× for slower cluster cores/BLAS config the
sbatch requests `--time=01:00:00` (well under the 72 h cap). Whole 15-cell
array ≈ 2.5 CPU-core-hours total.

## Cell matrix (15 cells, n_realizations=2000 each)

Common flags: `--catalogue-mode --n-realizations 2000 --n-events 250`
(defaults elsewhere). Seed = 20260727000 + cell index. One truth per cell.

| idx | mode | z_support | sky_frac | truth | seed |
|---|---|---|---|---|---|
| 0 | lcat | 0.43 | 1e-4 | 0.62 | 20260727000 |
| 1 | lcat | 0.43 | 1e-4 | 0.72 | 20260727001 |
| 2 | lcat | 0.43 | 1e-4 | 0.84 | 20260727002 |
| 3 | absolute | 0.43 | 1e-4 | 0.62 | 20260727003 |
| 4 | absolute | 0.43 | 1e-4 | 0.72 | 20260727004 |
| 5 | absolute | 0.43 | 1e-4 | 0.84 | 20260727005 |
| 6 | generator_marginal | 0.43 | 1e-4 | 0.62 | 20260727006 |
| 7 | generator_marginal | 0.43 | 1e-4 | 0.72 | 20260727007 |
| 8 | generator_marginal | 0.43 | 1e-4 | 0.84 | 20260727008 |
| 9 | generator_marginal | 0.79 | 1e-4 | 0.62 | 20260727009 |
| 10 | generator_marginal | 0.79 | 1e-4 | 0.72 | 20260727010 |
| 11 | generator_marginal | 0.79 | 1e-4 | 0.84 | 20260727011 |
| 12 | generator_marginal | 0.95 | 2e-4 | 0.62 | 20260727012 |
| 13 | generator_marginal | 0.95 | 2e-4 | 0.72 | 20260727013 |
| 14 | generator_marginal | 0.95 | 2e-4 | 0.84 | 20260727014 |

Cells 9–14 are the B_num-characterization axis (generator_marginal at the
two high-impostor operating points).

### Exact CLI lines

```bash
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode lcat               --z-support 0.43 --sky-frac 1e-4 --truths 0.62 --seed 20260727000 --output "$RUN_DIR/pp_cat_lcat_zs0.43_sky1e-4_h0.62.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode lcat               --z-support 0.43 --sky-frac 1e-4 --truths 0.72 --seed 20260727001 --output "$RUN_DIR/pp_cat_lcat_zs0.43_sky1e-4_h0.72.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode lcat               --z-support 0.43 --sky-frac 1e-4 --truths 0.84 --seed 20260727002 --output "$RUN_DIR/pp_cat_lcat_zs0.43_sky1e-4_h0.84.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode absolute           --z-support 0.43 --sky-frac 1e-4 --truths 0.62 --seed 20260727003 --output "$RUN_DIR/pp_cat_absolute_zs0.43_sky1e-4_h0.62.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode absolute           --z-support 0.43 --sky-frac 1e-4 --truths 0.72 --seed 20260727004 --output "$RUN_DIR/pp_cat_absolute_zs0.43_sky1e-4_h0.72.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode absolute           --z-support 0.43 --sky-frac 1e-4 --truths 0.84 --seed 20260727005 --output "$RUN_DIR/pp_cat_absolute_zs0.43_sky1e-4_h0.84.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode generator_marginal --z-support 0.43 --sky-frac 1e-4 --truths 0.62 --seed 20260727006 --output "$RUN_DIR/pp_cat_genmarg_zs0.43_sky1e-4_h0.62.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode generator_marginal --z-support 0.43 --sky-frac 1e-4 --truths 0.72 --seed 20260727007 --output "$RUN_DIR/pp_cat_genmarg_zs0.43_sky1e-4_h0.72.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode generator_marginal --z-support 0.43 --sky-frac 1e-4 --truths 0.84 --seed 20260727008 --output "$RUN_DIR/pp_cat_genmarg_zs0.43_sky1e-4_h0.84.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode generator_marginal --z-support 0.79 --sky-frac 1e-4 --truths 0.62 --seed 20260727009 --output "$RUN_DIR/pp_cat_genmarg_zs0.79_sky1e-4_h0.62.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode generator_marginal --z-support 0.79 --sky-frac 1e-4 --truths 0.72 --seed 20260727010 --output "$RUN_DIR/pp_cat_genmarg_zs0.79_sky1e-4_h0.72.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode generator_marginal --z-support 0.79 --sky-frac 1e-4 --truths 0.84 --seed 20260727011 --output "$RUN_DIR/pp_cat_genmarg_zs0.79_sky1e-4_h0.84.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode generator_marginal --z-support 0.95 --sky-frac 2e-4 --truths 0.62 --seed 20260727012 --output "$RUN_DIR/pp_cat_genmarg_zs0.95_sky2e-4_h0.62.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode generator_marginal --z-support 0.95 --sky-frac 2e-4 --truths 0.72 --seed 20260727013 --output "$RUN_DIR/pp_cat_genmarg_zs0.95_sky2e-4_h0.72.json"
python -m master_thesis_code.validation.pp_coverage --catalogue-mode --n-realizations 2000 --n-events 250 --mixture-mode generator_marginal --z-support 0.95 --sky-frac 2e-4 --truths 0.84 --seed 20260727014 --output "$RUN_DIR/pp_cat_genmarg_zs0.95_sky2e-4_h0.84.json"
```

Note: each cell uses a distinct seed, so the frozen catalogue differs per
cell even at the same operating point. If the analysis wants the same frozen
catalogue shared across modes/truths at one operating point (closer to
"one GLADE+ table"), give the three truth-cells of one (mode, z_support)
the SAME seed instead — coverage independence across cells is then lost.
Left as-is (distinct seeds) to keep realizations independent.

## sbatch draft

`pp_fullpower.sbatch` in this directory: array 0–14 on `cpu_il`,
4 CPUs/task, `--time=01:00:00`, venv via `source cluster/modules.sh`.
**Prerequisite:** the cluster checkout must have branch
`feat/pp-impostor-harness` (commit 44ee912) checked out — the harness's
catalogue mode does not exist on main. No GPU, no catalog file, no
simulations symlink needed. Submit with a wrapper exporting `RUN_DIR`
(and optionally `PROJECT_ROOT`), per cluster conventions — NOT submitted
by this task (do-not-touch-cluster constraint).
