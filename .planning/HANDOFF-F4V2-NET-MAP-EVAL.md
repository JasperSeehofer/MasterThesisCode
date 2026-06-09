# HANDOFF — F4-v2 net-MAP verification (local run on the 64 GB device)

**Created:** 2026-06-09
**For:** the 64 GB / 32-core device (project already installed & set up)
**Goal:** measure the **net H0 MAP** of the OLD seed200⊕300 production CRB
evaluated through **both** p_det estimators — `nadaraya_watson` (pre-F4-v2,
local-constant) and `local_linear` (F4-v2) — to settle whether the F4 near-field
boundary bias actually drives the H0 bias, and whether F4-v2 fixes it.

This is the verification the cluster could not run (days stuck on fairshare).

---

## Why this run is the arbiter

Confirmed so far (see commit `bd9081e` + `.planning/research/F4v2-local-linear-design-sketch.md`):
- F4's Nadaraya-Watson p_det has a real **d_L→0 boundary bias** (gives ~0.5 where
  ground truth = 1.0). F4-v2 (local-linear) fixes the *accuracy* — verified.
- **But** the cheap D(h)-denominator proxy *steepens* under v2 (0.73→0.76 decline
  −2.68% NW → −3.95% LL), which would push the MAP *up*, not toward truth.
- The proxy is denominator-only. The full joint is `Σ log num_i(h) − N log D(h)`;
  the near-field fix hits the numerator too. **Only a full eval resolves the net.**

Three possible outcomes (truth h = 0.73):

| NW MAP | LL-v2 MAP | Conclusion |
|---|---|---|
| ~0.76 | ~0.73 | F4 boundary bias **was** the H0-bias driver; **v2 fixes it.** ✅ |
| ~0.76 | ~0.76+ | v2 fixes accuracy but **not** the H0 bias; cause is elsewhere. |
| ~0.73 | ~0.73 | old CRB never showed the bias → premise breaks; re-investigate. |

(For reference: this same CRB through the **pre-F4 histogram** estimator gave raw
ΣlogL MAP **0.738** locally; the cluster phase50 seed400 run gave **0.760**.)

---

## Step 0 — sync the repo

```bash
cd <project_root>
git pull            # must include commit bd9081e "[PHYSICS] F4-v2 ..."
uv sync --extra cpu --extra dev     # if deps changed (they didn't, but safe)
```

## Step 1 — make sure the three data inputs are present

1. **GLADE catalog** (you have this): `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv` (~1.44 GB).
2. **Old seed200⊕300 CRB** — needs both files. Pull from the cluster:
   ```bash
   mkdir -p ~/data/run_production_h0p73_20260506/simulations
   rsync -avz bwunicluster:/pfs/work9/workspace/scratch/st_ac147838-emri/run_production_h0p73_20260506/simulations/cramer_rao_bounds.csv \
             bwunicluster:/pfs/work9/workspace/scratch/st_ac147838-emri/run_production_h0p73_20260506/simulations/prepared_cramer_rao_bounds.csv \
             ~/data/run_production_h0p73_20260506/simulations/
   ```
   (1549-row CRB; ~8 MB total.)
3. **Injection set** — the 105 500-event F4 reference set is simplest (it is what
   all the diagnostics + the v2 verification used). Either copy it from the main
   dev machine's `simulations/injections/`, or pull the canonical pool from the
   cluster:
   ```bash
   mkdir -p ~/data/injections
   rsync -avz bwunicluster:/pfs/work9/workspace/scratch/st_ac147838-emri/run_closure_h0p73_h3_20260505/simulations/injections/injection_h_*.csv \
             ~/data/injections/
   ```
   (NW-vs-LL is internally valid with either set since both estimators use the
   same injections; the 105 500 set matches the local diagnostics.)

## Step 2 — run the driver (both estimators × 6 h-values)

```bash
scripts/f4_net_map_eval.sh ~/data/run_production_h0p73_20260506 ~/data/injections ~/data/f4v2_verify
```

- h-grid: `0.72 0.73 0.74 0.75 0.76 0.78` (decisive bracket around 0.73 vs 0.76).
- Runs ~12 evals total (~12 min each on 32 cores → **~2.5 h wall**; or trim the
  grid to `0.73 0.745 0.76` in the script for a faster first read).
- Writes posteriors to `~/data/f4v2_verify/{nadaraya_watson,local_linear}/simulations/posteriors{,_with_bh_mass}/`.
- Auto-prints the net-MAP comparison at the end.

## Step 3 — read the result

The driver finishes by running:
```bash
uv run python scripts/f4_net_map_compare.py \
    ~/data/f4v2_verify/nadaraya_watson ~/data/f4v2_verify/local_linear
```
Match the printed NW / LL MAPs against the outcome table above.

---

## After the result — report back

Paste the net-MAP table into the main session. Then:
- **Outcome 1** (LL→0.73): F4-v2 is confirmed as the H0-bias fix → run the
  validation ladder (closure tests h=0.65/0.73; re-eval seed400 phase50) and
  update `docs/H0_BIAS_RESOLUTION.md` with a new section.
- **Outcome 2** (LL still high): keep F4-v2 as a verified *accuracy* fix (do not
  claim H0-bias resolution) and re-open the bias investigation — the D(h)
  steepening suggests the dominant driver may be the dV_c/dz × p_det interplay,
  not the estimator alone.
- **Outcome 3** (NW already 0.73): the seed400 0.76 is then seed/N-specific, not
  estimator-driven → pivot to the multi-seed campaign.

## Notes / gotchas
- `--pdet_estimator` is the new flag (commit `bd9081e`); default is `local_linear`.
- The eval reads BOTH `cramer_rao_bounds.csv` (at construction) and
  `prepared_cramer_rao_bounds.csv` — both must be present (this bit us on the
  cluster).
- RAM: the 22.6 M-row catalog load peaks ~6–10 GB; fine on 64 GB, **not** on a
  16 GB machine.
- Companion artifacts: `scripts/dh_estimator_proxy.py`,
  `scripts/dh_steepness_investigation.py` (the diagnostics),
  `.planning/research/pdet-boundary-estimation-literature.md` (the methods review).
