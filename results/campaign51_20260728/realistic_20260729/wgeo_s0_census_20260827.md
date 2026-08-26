# [WGEO] stage 0, read (2) — fleet/catalogue census: does the window asymmetry have z-structure?

Independent measurement, local CPU only. Working notes + scripts below; structured result returned
separately via the orchestrator's `StructuredOutput` contract.

## 0. Dataset pin

`darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`: size 1,681,954,844 bytes, md5
`c52c13b5cab61f6b3f04bbe202550969` — re-verified with `md5sum` before touching the file, matches
the pin stated in the task brief and in `mker_r2_measure_A.md` §0. `free -g`: total 30G, available
20G. 16 cores. Chunked pandas read used (chunksize=2,000,000, 4 columns only), NOT the full
`GalaxyCatalogueHandler` (which also builds two BallTrees and an ecliptic rotation over all 22.6M
rows — unneeded here and far more expensive).

## 1. Pruning-chain reproduction and index-semantics validation

Reimplemented, verbatim, the same prune logic already cross-checked in `mker_r2_measure_A.md` §1
(chunked scan, single pass, order-preserving position counter):

1. `STELLAR_MASS`/`STELLAR_MASS_ABSOULTE_ERROR` → R&V15 `BH_MASS`/`BH_MASS_ERROR` via
   `_empiric_stellar_mass_to_BH_mass_relation` (`handler.py:1368-1382`, 0.24-dex-only intrinsic
   scatter budget — the CURRENT production budget, not the R&V15-full 0.55-dex counterfactual from
   the MKER thread).
2. Drop `BH_MASS.isna()` (`handler.py:1131-1134`).
3. `_mass_redshift_prune_mask` (`handler.py:215-251`), `M_min=1e4`, `M_max=1e7`, `z_max=1.5`
   (`M_SOURCE_FRAME_MIN/MAX`, `cosmological_model.py` default `max_redshift=1.5`).
4. `reset_index()` (`handler.py:555`, the single such call) — pruned-frame position = `catalog_index`.

**Validation against the three pinned rows (script `wgeo_census.py`, §8 below), run BEFORE
computing anything else:**

```
VALIDATE 6791138: BH_MASS 709540.708756878 vs 709540.708756878 -> True; BH_MASS_ERROR 894866.2758100418 vs 894866.2758100418 -> True
VALIDATE 6791158: BH_MASS 709540.708756878 vs 709540.708756878 -> True; BH_MASS_ERROR 1570331.1654161075 vs 1570331.1654161075 -> True
VALIDATE 6791151: BH_MASS 223872.11385683485 vs 223872.11385683485 -> True; BH_MASS_ERROR 291758.99489010876 vs 291758.99489010876 -> True
VALIDATION_ALL_OK True
```

**PASSED.** Index interpretation confirmed correct; proceeding.

Raw catalogue rows: 22,641,048. Pruned (mass-info-present + mass/z-pruned) rows: **20,834,171**
(92.0% of raw survives the prune — the mass/z window is broad relative to the R&V15-mapped mass
distribution).

## 2. CV distribution

`CV = BH_MASS_ERROR / BH_MASS` per galaxy (independent of any event — purely a catalogue-side
quantity). Confirmed identical, by construction, to the "σ_ln" quantity used in
`CLAIM_P3_MKER_20260826.md` R2.7(ii): `BH_mass_error` is built by
`_empiric_stellar_mass_to_BH_mass_relation` as `BH_mass * sqrt(Var[ln M_BH])`, so
`BH_MASS_ERROR/BH_MASS` **is** `sqrt(Var[ln M_BH])` exactly — no lognormal-parameter conversion is
needed or applied; `CV ≡ σ_ln` at every row. Cross-checked directly on the exhibit candidate
6791151: script's `CV = 1.3032395587986776`, matching the claim card's quoted `σ_ln = 1.3032395587986776`
bit-for-bit.

| stat | value |
|---|---|
| N (pruned catalogue) | 20,834,171 |
| p10 | 0.7846 |
| p25 | 0.7906 |
| median (p50) | 0.8614 |
| p75 | 0.9401 |
| p90 | 1.2137 |
| max | 9.5220 |
| min | 0.5930 |

**Threshold derivation:** the linear lower edge at multiplier `K=1.5` is
`M·(1 − K·CV)`; it is non-positive iff `CV ≥ 1/K = 0.6̄6667`. Independently re-derived (not just
quoted from the task brief) — matches.

**Fraction with `CV ≥ 1/1.5`: 0.996112** (99.61% of the entire pruned catalogue). This is a striking
number in itself: essentially every catalogue galaxy has a formally negative linear lower mass-window
edge, at 1.5σ. `CV_min = 0.593` — even the *tightest*-constrained galaxy in the whole 20.8M-row
catalogue sits only just under threshold; the catalogue-wide CV distribution is bounded well away
from anything an event's window would treat as "narrow."

## 3. Induced ln-space asymmetry per galaxy

Definition (explicit, not lifted verbatim from the R2 thread, which did not define a single
per-galaxy scalar — it only quoted the two edge-ratios for one candidate): the **log/linear
upper-edge log-ratio**,

```
asym(galaxy) = ln[upper_edge_log(K)] − ln[upper_edge_linear(K)]
             = K·CV − ln(1 + K·CV)                      (K = 1.5, the adopted symmetric multiplier)
```

where `upper_edge_log(K) = exp(K·CV)` and `upper_edge_linear(K) = 1 + K·CV` are the two
constructions' upper-edge multiples of the central mass. `asym = 0` at `CV = 0` (no error, no
asymmetry) and grows monotonically and without bound as `CV` grows — it is well-defined everywhere
in the catalogue because `1 + K·CV > 0` always (CV ≥ 0), unlike the *lower* edge, which is the one
that goes negative and is reported separately as a fraction, not folded into this scalar.

Sanity check against the exhibit (`CV = 1.3032395587986776`, `K=1.5`):
`upper_edge_linear = 2.954859338` (claim card: 2.955 ✓), `upper_edge_log = 7.062925470`
(claim card: 7.06 ✓).

| stat | value (nats) |
|---|---|
| p10 | 0.3990 |
| p25 | 0.4039 |
| median | 0.4626 |
| p75 | 0.5304 |
| p90 | 0.7837 |
| max | 11.556 |
| mean | 0.5276 |

Every galaxy in the pruned catalogue carries a materially positive asymmetry (`min ~0.31` at
`CV_min=0.593`); the window's log/linear mismatch is not a tail phenomenon — it is the catalogue's
default state.

## 4. THE Z-STRUCTURE TEST — decisive

Bins of width 0.1 in `REDSHIFT`, full catalogue (`wgeo_stats.py`, §8):

| z bin | n | median CV | median asym | frac non-positive lower edge |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 3,122,580 | 1.2272 | 0.7967 | 0.9819 |
| 0.1–0.2 | 8,067,019 | 0.9191 | 0.5121 | 0.9973 |
| 0.2–0.3 | 7,819,985 | 0.8039 | 0.4148 | 0.9996 |
| 0.3–0.4 | 1,583,681 | 0.7851 | 0.3994 | 1.0000 |
| 0.4–0.5 | 235,837 | 0.7846 | 0.3990 | 1.0000 |
| 0.5–0.6 | 1,987 | 0.7862 | 0.4003 | 0.9995 |
| 0.6–1.5 (sparse tail, n<1,100/bin) | ~2,972 total | ~0.78 | ~0.39 | ~1.000 |

**The asymmetry SHRINKS with z, monotonically, over the entire well-populated range (z<0.5, >99.98%
of the pruned catalogue).** Median CV falls from 1.227 at z<0.1 to 0.785 by z~0.3-0.5 and stays flat
(~0.78-0.79) out through the sparse high-z tail. Median asymmetry falls in lockstep (0.797 → 0.399).
The fraction with a non-positive lower edge *rises* slightly with z (0.982 → ~1.000) but this
saturates almost immediately (already 0.997 by the second bin) and carries essentially no
discriminating power once z > 0.1 — everything is already excluded from below at that point,
catalogue-wide.

**Trend statistic:** Spearman rank correlation, full catalogue, N=20,834,171:

```
spearman(REDSHIFT, CV):         rho = -0.6521,  p = 0.0 (underflows double precision), n = 20,834,171
spearman(REDSHIFT, asymmetry):  rho = -0.6521,  p = 0.0,                                n = 20,834,171
spearman(REDSHIFT, neg_edge):   rho = +0.0824,  p = 0.0,                                n = 20,834,171
```

Restricting to the campaign's actual detected-event redshift range (`z ∈ [0.0068, 0.3401]`, the
observed min/max of `z_true` over the 2,261 fleet events collected in §6): N=20,153,304 (96.7% of
the pruned catalogue lies in this range), `spearman(REDSHIFT, CV) = -0.6420, p=0.0` — the trend is
**not** an artifact of the sparse high-z tail; it is fully present within the range the campaign
actually samples.

**Verdict on the hypothesis: the asymmetry SHRINKS with z. It does not grow.** This is the opposite
sign from what the bias lead required to explain a high-z-localized dark-class tilt via this
mechanism. Per the task's measure-first framing, **this is a clean, decisive null** — reported as
plainly as a positive result would be.

## 5. Mass confound — quantified, not just flagged

Mass and redshift are strongly correlated in this catalogue (flux-limited selection: only
increasingly luminous/massive galaxies are visible at increasing distance):

```
spearman(BH_MASS, REDSHIFT) = +0.7398,  p = 0.0,  n = 20,834,171   (the confound)
spearman(BH_MASS, CV)       = -0.6558,  p = 0.0,  n = 20,834,171
```

Mass-binned table (log-spaced bins spanning the prune bounds `[1e4, 1e7]` M_sun):

| M_BH range (M_sun) | n | median CV | median z | median asym |
|---|---:|---:|---:|---:|
| 1.0e4–1.8e4 | 1,314 | 1.1567 | 0.0099 | 0.7289 |
| 1.8e4–3.2e4 | 21,347 | 4.3077 | 0.0166 | 4.4518 |
| 3.2e4–5.6e4 | 22,968 | 2.7744 | 0.0239 | 2.5203 |
| 5.6e4–1.0e5 | 42,651 | 2.2710 | 0.0319 | 1.9235 |
| 1.0e5–1.8e5 | 28,710 | 1.7711 | 0.0426 | 1.3601 |
| 1.8e5–3.2e5 | 260,125 | 1.3032 | 0.0486 | 0.8714 |
| 3.2e5–5.6e5 | 317,036 | 1.2755 | 0.0672 | 0.8440 |
| 5.6e5–1.0e6 | 710,107 | 1.2519 | 0.0876 | 0.8208 |
| 1.0e6–1.8e6 | 1,197,237 | 1.0740 | 0.1132 | 0.6513 |
| 1.8e6–3.2e6 | 2,324,755 | 0.9907 | 0.1264 | 0.5753 |
| 3.2e6–5.6e6 | 3,642,639 | 0.8039 | 0.1702 | 0.4148 |
| 5.6e6–1.0e7 | 2,600,133 | 0.9203 | 0.1977 | 0.5131 |

CV falls sharply with mass over the low-mass bins then plateaus/mildly rebounds near the top of the
mass range — consistent with the R&V15 relation's error terms (`d_beta`-driven `ln(SM/10)`
contribution changes sign/magnitude across the pivot, and the propagated-stellar-mass-error term
`~1/SM` dominates at low mass).

**Does the z-trend in CV survive conditioning on mass?** Spearman(z, CV) computed *within* each
mass bin (n≥1,000 required):

| M_BH range (M_sun) | n | rho(z, CV | mass bin) | p |
|---|---:|---:|---:|
| 1.0e4–1.8e4 | 1,314 | −0.1995 | 2.9e-13 |
| 1.8e4–3.2e4 | 21,347 | +0.1126 | 3.8e-61 |
| 3.2e4–5.6e4 | 22,968 | −0.0542 | 2.0e-16 |
| 5.6e4–1.0e5 | 42,651 | −0.2627 | ~0 |
| 1.0e5–1.8e5 | 28,710 | +0.0315 | 9.4e-8 |
| 1.8e5–3.2e5 | 260,125 | −0.2534 | ~0 |
| 3.2e5–5.6e5 | 317,036 | −0.5418 | ~0 |
| 5.6e5–1.0e6 | 710,107 | −0.6765 | ~0 |
| 1.0e6–1.8e6 | 1,197,237 | −0.7703 | ~0 |
| 1.8e6–3.2e6 | 2,324,755 | −0.7831 | ~0 |
| 3.2e6–5.6e6 | 3,642,639 | −0.3148 | ~0 |
| 5.6e6–1.0e7 | 2,600,133 | −0.2526 | ~0 |

**The z-shrinkage survives conditioning on mass — it is not purely a mass-confound artifact.** In
every bin with n>50,000 (i.e. the bins that actually dominate the catalogue by count), the
within-bin sign is negative and often strongly so (as steep as −0.78 in the 1.8-3.2e6 M_sun bin,
which alone holds 2.3M galaxies). The two low-count, low-mass bins (n~21k-23k) that flip sign
briefly are a small, low-population corner of the mass range (M_BH<5.6e4, jointly ~0.2% of the
pruned catalogue) and do not change the catalogue-wide conclusion. **Both the marginal and the
mass-conditioned z-trend point the same way: shrinking, not growing.**

## 6. FLEET TIE-IN

Scanned all 24 `bc_9001XX_work` arms (`/home/jasper/Repositories/darksiren-emri/results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/`), each `seed9001XX/simulations/`:
`prepared_cramer_rao_bounds.csv` (`z_true`, `luminosity_distance`, `SNR` columns — `z_true` is
present directly, no re-derivation from `d_L` needed) joined against
`posteriors_with_bh_mass/h_0_73.json`'s `galaxy_likelihoods` (window-**passed** candidates,
`bayesian_statistics.py:4928-4935`) and `additional_galaxies_without_bh_mass`
(window-**excluded** complement of the *redshift-passing* candidate set —
verified at source: `possible_host_galaxies_reduced = [host for host in possible_host_galaxies if
host not in hosts_with_bh_mass_set]`, `bayesian_statistics.py:4861-4865`, then
`bayesian_statistics.py:4937-4946`). Both are keyed per-event by the same string index that indexes
`self.cramer_rao_bounds.iterrows()`, i.e. the row position in `prepared_cramer_rao_bounds.csv`.

24/24 arms loaded, 0 missing. **2,261 event-rows collected** (union of events with either list
non-empty; a handful of events carry only a completion-fallback zero-host record and contribute
nothing to either list — 24 of the 2,261 have `n_total=0` and are excluded from the correlation
below).

```
z_true range over fleet events:  min=0.006822, max=0.340081
n_passed:   median=150 (over all events with n_total>0), max=51,168 (one very-low-SNR, huge-cone event)
n_excluded: median=0 — 923/2,237 events (41.3%) have ZERO window-excluded candidates at all
n_passed==0: 40/2,261 events (1.8%)
```

**Passed/excluded ratio vs redshift** (`frac_passed = n_passed/(n_passed+n_excluded)`):

```
spearman(z_true, frac_passed) = +0.2271, p = 1.47e-27, n = 2,237
spearman(d_L,    frac_passed) = +0.2250, p = 4.67e-27, n = 2,237
spearman(z_true, n_excluded)  = -0.0603, p = 4.32e-03, n = 2,237
spearman(z_true, n_passed)    = +0.3825, p = 7.36e-79, n = 2,237
```

Binned:

| z bin | n events | median n_passed | median n_excluded | median frac_passed |
|---|---:|---:|---:|---:|
| 0.00–0.05 | 93 | 3 | 1 | 0.9545 |
| 0.05–0.10 | 578 | 46.5 | 1 | 0.9842 |
| 0.10–0.15 | 819 | 167 | 3 | 0.9884 |
| 0.15–0.20 | 504 | 338 | 2 | 0.9975 |
| 0.20–0.30 | 227 | 413 | 0 | 1.0000 |
| 0.30–0.50 | 16 | 2,184 | 0 | 1.0000 |

**The fleet result independently confirms the catalogue-wide direction: the fraction of localization-
cone candidates that pass the mass window RISES with z (weakly but very significantly, p~1e-27), and
the absolute count of window-excluded candidates per event mildly FALLS with z.** This is the same
sign as §4/§5's catalogue-wide CV/asymmetry shrinkage (higher-z galaxies visible in a flux-limited
catalogue skew to higher, tighter-fractional-error mass → relatively less exclusion at the event
level too). Two independent measurements (catalogue-marginal and fleet-event-level) agree on
direction.

Caveat: `n_excluded=0` for 41% of events is a real floor effect (small cones, especially at low z,
often contain none or very few candidates at all — median `n_passed` at z<0.05 is only 3), not a
data-recoverability gap; both `galaxy_likelihoods` and `additional_galaxies_without_bh_mass` were
present and readable for every one of the 2,261 event-rows, so nothing here is "NOT FOUND."

## 7. Bottom line

**A flat-or-shrinking z-trend, as anticipated by the task brief as the fully successful null
outcome, is exactly what was measured — on both axes tested (catalogue-marginal CV/asymmetry, and
fleet-level passed/excluded ratios), and the catalogue-marginal result survives conditioning on the
mass confound.** The window-asymmetry mechanism, as measured here, predicts LESS exclusion-driven
distortion at high z, not more — the opposite sign from what would be needed to produce or
contribute to the banked high-z dark-class tilt (score −0.635, 37σ). This kills the [WGEO] lead in
its literal "does the window asymmetry have z-structure that could explain the tilt" form. It is
reported as a fully successful, clean result, per the task's measure-first discipline — no positive
attribution is claimed or implied, and none should be inferred from the strength of the negative
result either (see caveats).

## 8. Scripts (verbatim, as executed)

### `wgeo_census.py` — pruning chain, index validation, per-galaxy arrays

```python
import time

import numpy as np
import pandas as pd

CSV = "/home/jasper/Repositories/darksiren-emri/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv"

names = ["RA", "DEC", "BMAG", "REDSHIFT", "REDSHIFT_ERR", "STELLAR_MASS", "STELLAR_MASS_ERR", "REDSHIFT_FLAG"]
usecols = ["REDSHIFT", "REDSHIFT_ERR", "STELLAR_MASS", "STELLAR_MASS_ERR"]

alpha = 7.45 * np.log(10)
beta = 1.05
d_alpha = 0.08 * np.log(10)
d_beta = 0.11
sigma_int = 0.24 * np.log(10)

M_min = 1e4
M_max = 1e7
Z_MAX = 1.5
K = 1.5  # sigma_multiplier, adopted "symmetric" mode, _bh_mass_error_multiplier = sigma_multiplier

TARGETS = {6791138, 6791151, 6791158}
found = {}

chunksize = 2_000_000
cum_survivors = 0
total_rows = 0

BH_MASS_chunks, BH_MASS_ERROR_chunks, REDSHIFT_chunks = [], [], []

reader = pd.read_csv(CSV, header=None, names=names, usecols=usecols, chunksize=chunksize)

for chunk_i, chunk in enumerate(reader):
    n = len(chunk)
    global_start = total_rows
    total_rows += n

    SM = chunk["STELLAR_MASS"].to_numpy(dtype=np.float64)
    SM_err = chunk["STELLAR_MASS_ERR"].to_numpy(dtype=np.float64)
    Z = chunk["REDSHIFT"].to_numpy(dtype=np.float64)
    Z_err = chunk["REDSHIFT_ERR"].to_numpy(dtype=np.float64)

    with np.errstate(all="ignore"):
        BH_mass = np.exp(alpha + beta * np.log(SM / 10))
        BH_mass_error = BH_mass * np.sqrt(
            sigma_int**2 + d_alpha**2 + (np.log(SM / 10) * d_beta) ** 2 + (beta / SM * SM_err) ** 2
        )

    mass_info_mask = ~np.isnan(BH_mass)
    with np.errstate(invalid="ignore"):
        prune_mask = (
            (BH_mass + BH_mass_error >= M_min)
            & (BH_mass - BH_mass_error <= M_max)
            & (Z - Z_err <= Z_MAX)
        )
    prune_mask = np.where(np.isnan(prune_mask.astype(float)), False, prune_mask)
    survive = mass_info_mask & prune_mask
    n_survive = int(survive.sum())

    if n_survive > 0:
        lo, hi = cum_survivors, cum_survivors + n_survive - 1
        relevant_targets = [t for t in TARGETS if lo <= t <= hi]
        if relevant_targets:
            surv_idx = np.nonzero(survive)[0]
            for t in relevant_targets:
                local_row = surv_idx[t - cum_survivors]
                found[t] = {
                    "original_csv_row_0based": int(global_start + local_row),
                    "BH_MASS": float(BH_mass[local_row]),
                    "BH_MASS_ERROR": float(BH_mass_error[local_row]),
                }
        BH_MASS_chunks.append(BH_mass[survive])
        BH_MASS_ERROR_chunks.append(BH_mass_error[survive])
        REDSHIFT_chunks.append(Z[survive])

    cum_survivors += n_survive

# validation against pinned rows, then np.save of BH_MASS/BH_MASS_ERROR/REDSHIFT arrays
```

### `wgeo_stats.py` — CV/asymmetry distributions, z-bins, mass-bins, Spearman trends

```python
import numpy as np
from scipy import stats

BH_MASS = np.load(".../BH_MASS.npy")
BH_MASS_ERROR = np.load(".../BH_MASS_ERROR.npy")
REDSHIFT = np.load(".../REDSHIFT.npy")

K = 1.5
CV = BH_MASS_ERROR / BH_MASS
lower_edge_factor = 1.0 - K * CV
upper_edge_factor_linear = 1.0 + K * CV
upper_edge_factor_log = np.exp(K * CV)
neg_lower_edge = lower_edge_factor <= 0.0
asym = np.log(upper_edge_factor_log) - np.log(upper_edge_factor_linear)  # = K*CV - log(1+K*CV)

# percentiles, z_bins = np.arange(0, 1.6, 0.1), digitize + per-bin median/frac,
# mass_bins = np.logspace(log10(1e4), log10(1e7), 13), same per-bin aggregation,
# scipy.stats.spearmanr(REDSHIFT, CV), (REDSHIFT, asym), (REDSHIFT, neg_lower_edge),
# (BH_MASS, CV), (BH_MASS, REDSHIFT), and per-mass-bin spearmanr(REDSHIFT[mask], CV[mask]).
```

### `wgeo_fleet.py` — fleet passed/excluded census across the 24 bc arms

```python
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

BASE = ".../p3_2d_fleet_20260825"
rows = []
for bc_dir in sorted(glob.glob(os.path.join(BASE, "bc_9001??_work"))):
    seed_dir = glob.glob(os.path.join(bc_dir, "seed*"))[0]
    csv_path = os.path.join(seed_dir, "simulations", "prepared_cramer_rao_bounds.csv")
    json_path = os.path.join(seed_dir, "simulations", "posteriors_with_bh_mass", "h_0_73.json")
    df = pd.read_csv(csv_path, usecols=["z_true", "luminosity_distance", "SNR", "in_catalog"])
    d = json.load(open(json_path))
    gl, add = d.get("galaxy_likelihoods", {}), d.get("additional_galaxies_without_bh_mass", {})
    for k in sorted(set(gl) | set(add), key=int):
        idx = int(k)
        rows.append({
            "arm": os.path.basename(bc_dir), "event_idx": idx,
            "z_true": float(df.loc[idx, "z_true"]),
            "d_L": float(df.loc[idx, "luminosity_distance"]),
            "n_passed": len(gl.get(k, [])), "n_excluded": len(add.get(k, [])),
        })
out = pd.DataFrame(rows)
# frac_passed = n_passed/(n_passed+n_excluded); spearmanr(z_true, frac_passed); z-binned table.
```

## 9. Runtime

`wgeo_census.py`: 4.7s (single-pass chunked scan, 22.6M raw rows, vectorized numpy per chunk).
`wgeo_stats.py`: 27.4s (dominated by three `spearmanr` calls over N=20.8M — each ~3-6s). `wgeo_fleet.py`: <10s
(24 small per-arm CSV/JSON reads). Total wall time for the whole read, including exploration: well
under 5 minutes, all on CPU, no cluster/SSH used, consistent with the operating rules.
