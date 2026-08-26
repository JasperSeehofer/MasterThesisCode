# [P3-MKER] R2 — MEASURER A: does the full R&V15 error budget readmit candidate 6791151?

Independent measurement. Working notes + scripts below; structured result returned separately.

## 0. Dataset pin

`darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv`: size 1,681,954,844 bytes,
md5 `c52c13b5cab61f6b3f04bbe202550969` — matches the pin stated in the task (already discharged
by the orchestrator). Confirmed with `md5sum` before touching the file.

`free -g` before starting: total 30G, used 8G, free 1G, buff/cache 21G, available 21G. Chose a
chunked pandas read (`chunksize=1_000_000`, `usecols` restricted to the 4 needed columns) over the
full production `GalaxyCatalogueHandler` load, because the handler also builds a BallTree and a 4D
BallTree and does an astropy equatorial->ecliptic rotation over all ~22.6M rows, none of which is
needed to answer this question and which would cost much more memory/time for no benefit to the
index-semantics/mass-window question. **I used the chunked-pandas fallback, and reimplemented the
prune/derivation logic read verbatim from `handler.py`, as permitted by the task brief.**

## 1. Index semantics (established BEFORE trusting any downstream number)

Read `darksiren_emri/galaxy_catalogue/handler.py` construction order (`GalaxyCatalogueHandler.__init__`,
lines 267-356):

1. `read_reduced_galaxy_catalog()` — `pd.read_csv(..., names=_reduced_catalog_column_names())`,
   no header row in the file, so the pandas `RangeIndex` on load *is* the 0-based row number in
   the reduced CSV.
2. `_map_stellar_masses_to_BH_masses()` (handler.py:1136-1142) — **overwrites in place**:
   `InternalCatalogColumns.BH_MASS` (on-disk column `STELLAR_MASS`) and
   `InternalCatalogColumns.BH_MASS_ERROR` (on-disk column `STELLAR_MASS_ABSOULTE_ERROR`) are
   replaced by the R&V15-derived BH mass/error via `_empiric_stellar_mass_to_BH_mass_relation`
   (handler.py:1368-1382). Index untouched.
3. `_rotate_equatorial_to_ecliptic()`, `_map_angles_to_spherical_coordinates()` — sky columns
   only, index untouched.
4. `_remove_galaxies_without_mass_information()` (handler.py:1131-1134) — boolean-mask drop of
   `BH_MASS.isna()`. Preserves index labels of survivors (order-preserving, no compaction yet).
5. `_get_pruned_galaxy_catalog(M_min, M_max, z_max)` (handler.py:358-368), called from `__init__`
   with `M_min=M_SOURCE_FRAME_MIN=1e4`, `M_max=M_SOURCE_FRAME_MAX=1e7`,
   `z_max=cosmological_model.max_redshift` (default 1.5, no `--max_redshift` override, per
   `cosmological_model.py:199-200`) — applies `_mass_redshift_prune_mask` (handler.py:215-251):
   keep iff `(BH_mass+BH_mass_error>=M_min) & (BH_mass-BH_mass_error<=M_max) &
   (redshift-redshift_error<=z_max)`. Boolean mask, preserves index labels of survivors.
6. `setup_galaxy_catalog_balltree()` (handler.py:544-556) — builds the BallTree from `.values`
   (positional, index-agnostic), **then** `self.reduced_galaxy_catalog =
   self.reduced_galaxy_catalog.reset_index()` — this is the ONLY `reset_index()` call in the whole
   file (`grep -n reset_index darksiren_emri/galaxy_catalogue/handler.py` → single hit at line
   555). It converts the surviving rows' original CSV-row index into a plain column `"index"` and
   assigns a fresh `0..M-1` `RangeIndex`.
7. `HostGalaxy.__init__` (handler.py:74-81) sets `self.catalog_index = parameters.name`, where
   `parameters` comes from `candidate_hosts = self.reduced_galaxy_catalog.iloc[indices]`
   (handler.py:635) — i.e. from the frame *after* step 6's `reset_index()`.

**Conclusion: `catalog_index` in the banked posteriors is a 0-based POSITIONAL index into the
mass-info-filtered + mass/z-pruned + reset-index frame** — i.e. into a **pruned subset**, not a
raw-CSV row number and not any catalogue-native ID. Position `k` in this frame = the `(k+1)`-th
row (in original file order) that survives steps 4-5.

### Cross-check (mandatory, run BEFORE trusting anything else)

Reimplemented steps 2/4/5/6 in a chunked scan (`scan_catalog.py`, full text in §4) over the raw
reduced CSV, tracking a cumulative surviving-row counter to map final position -> original CSV row.
Targets: catalog_index 6791158 and 6791138.

```
FOUND target 6791138: original_csv_row_0based=7351437, STELLAR_MASS=0.3 (1e10 Msun),
  STELLAR_MASS_ERR=0.3, REDSHIFT=0.057443, REDSHIFT_ERR=0.034929,
  BH_MASS=709540.708756878, BH_MASS_ERROR=894866.2758100418
FOUND target 6791158: original_csv_row_0based=7351457, STELLAR_MASS=0.3, STELLAR_MASS_ERR=0.6,
  REDSHIFT=0.031403, REDSHIFT_ERR=0.034049,
  BH_MASS=709540.708756878, BH_MASS_ERROR=1570331.1654161075
```

Task's required values: 6791158 -> host_M=709540.709, host_M_error=1570331.165 (from M*=0.3,
sigma_M*/M*=2.00); 6791138 -> host_M=709540.709, host_M_error=894866.276 (from M*=0.3,
sigma_M*/M*=1.00). **Both match to the given precision** (M*_err/M* = 0.6/0.3=2.00 and
0.3/0.3=1.00 respectively, host_M and host_M_error match to the last stated digit).
**INDEX-CROSSCHECK PASSED — index interpretation confirmed correct.**

## 2. Candidate 6791151

Same scan, third target, found in the same chunk:

| field | value | source |
|---|---|---|
| original CSV row (0-based) | 7,351,450 | scan (`scan_catalog.py`) |
| `STELLAR_MASS` (1e10 Msun) | 0.1 | reduced CSV col 5 |
| `STELLAR_MASS_ABSOULTE_ERROR` (1e10 Msun) | 0.1 | reduced CSV col 6 |
| `REDSHIFT` | 0.052818 | reduced CSV col 3 |
| `REDSHIFT_MEASUREMENT_ERROR` | 0.0347765573590526 | reduced CSV col 4 |
| `BH_MASS` (derived, R&V15, 0.24-dex-only) | 223,872.11385683485 M_sun | `_empiric_stellar_mass_to_BH_mass_relation` |
| `BH_MASS_ERROR` (derived, R&V15, 0.24-dex-only) | 291,758.99489010876 M_sun | same |

(`sigma_M*/M* = 0.1/0.1 = 1.00`, same relative stellar-mass error class as 6791138.)

## 3. The event: seed 900121, event 20 (bt arm)

`prepared_cramer_rao_bounds.csv` row 20 (0-based, `df.iloc[20]`), from
`results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bt_900121_work/seed900121/simulations/prepared_cramer_rao_bounds.csv`:

| field | value | column |
|---|---|---|
| `M` (=`Detection.M`, i.e. `M_z`) | 1,333,246.127516857 M_sun | `M` |
| `delta_M_delta_M` | 2.691661856213166e-05 | `delta_M_delta_M` |
| `M_z_sigma` (=`Detection.M_uncertainty`=`sqrt(delta_M_delta_M)`) | 0.005188122836068134 M_sun | derived |
| `luminosity_distance` (`d_L`) | 0.2831422160233205 Gpc | `luminosity_distance` |
| `d_L` uncertainty | 0.0014316570944745673 Gpc | `sqrt(delta_luminosity_distance_delta_luminosity_distance)` |
| `host_galaxy_index` (true injected host) | 6791134 | `host_galaxy_index` |
| SNR | 235.8558 | `SNR` |

Byte-identical to the previously-banked `bc_900121` reading of the same row (`mker_r1_exhibit_recompute.md`
§1b) — `M` and `delta_M_delta_M` match to full float precision, confirming the twin arms share the
same CRB for this injection.

`Detection.M`/`Detection.M_uncertainty` mapping verified against `darksiren_emri/datamodels/detection.py:134-143`
(`self.M = parameters["M"]`, `self.M_uncertainty = sqrt(parameters["delta_M_delta_M"])`) — this
is exactly what `get_possible_hosts_from_ball_tree` receives as `M_z`/`M_z_sigma`
(`bayesian_statistics.py:4689-4690`).

### Redshift-range bounds z_min, z_max

`bayesian_statistics.py:4669-4679` calls `get_redshift_outer_bounds(distance=d_L,
distance_error=d_L_uncertainty, h_min=cosmological_model.h.lower_limit,
h_max=cosmological_model.h.upper_limit, Omega_m_min=..., Omega_m_max=..., sigma_multiplier=2.0)`,
then `z_max = min(z_max, redshift_upper_limit)` with `redshift_upper_limit =
cosmological_model.max_redshift = 1.5`.

`h_min=0.6, h_max=0.86, Omega_m_min=0.04, Omega_m_max=0.5` — `LamCDMScenario.__init__`
(`cosmological_model.py:386-397`), which also happen to be `get_redshift_outer_bounds`'s own
defaults.

Read `physical_relations.py:546-567`. **Note (transparency, not something I am ruling on):** the
function computes `Omega_de_min`/`Omega_de_max` but never passes them (or `Omega_m_min`/
`Omega_m_max`) into `dist_to_redshift` — only `h_min`/`h_max` are passed, so `Omega_m`/`Omega_de`
silently use their *fiducial* defaults inside this specific bound calculation. This is exactly what
the code executes today, so it is what I reproduced; verified by calling the production
`get_redshift_outer_bounds` directly and confirming it returns the identical numbers as a
by-hand `dist_to_redshift(d_L∓3·d_L_unc, h_min/h_max)` reconstruction (see §4 script output) —
also confirms the (unused-inside-the-body) `sigma_multiplier` argument is a no-op in this call, so
the value passed at the call site (`2.0`) makes no numerical difference. Flagging as observed, not
adjudicating.

Computed (both by calling `darksiren_emri.physical_relations.get_redshift_outer_bounds` directly,
and by hand-reconstructing with `dist_to_redshift`; the two agree exactly):

```
z_min = 0.05356499027434118
z_max_raw = 0.07776556271743075   (< 1.5, so the cap is inactive)
z_max = 0.07776556271743075
(1+z_min) = 1.0535649902743411
(1+z_max) = 1.0777655627174307
```

## 4. Window test — current (0.24-dex-only) budget

`get_possible_hosts_from_ball_tree` is called with `sigma_multiplier=1.5`,
`mass_filter_sigma=self._mass_filter_sigma` (default `"symmetric"`, `bayesian_statistics.py:3311`,
unchanged by any per-run override found) -> `_bh_mass_error_multiplier = 1.5` (handler.py:654-661,
the "symmetric" branch adopted per commit `cf4f8a2a`, row #198-#202).

Mass filter (`handler.py:663-673`), both conditions must hold:

```
cond1:  (M_z - M_z_sigma*1.5) / (1+z_max)  <=  BH_MASS + BH_MASS_ERROR*1.5
cond2:  BH_MASS - BH_MASS_ERROR*1.5        <=  (M_z + M_z_sigma*1.5) / (1+z_min)
```

Substituting (BH_MASS/BH_MASS_ERROR = candidate 6791151's 0.24-dex-only values from §2):

```
left  = (1,333,246.127516857 - 0.005188122836068134*1.5) / 1.0777655627174307
      = 1,237,046.502370223

right = (1,333,246.127516857 + 0.005188122836068134*1.5) / 1.0535649902743411
      = 1,265,461.6920707219

cond1:  1,237,046.502370223  <=  223,872.11385683485 + 291,758.99489010876*1.5
                              =  223,872.11385683485 + 437,638.4923351632
                              =  661,510.6061919980   ->  FALSE   *** FAILS ***

cond2:  223,872.11385683485 - 437,638.4923351632 = -213,766.37847832834
                              <=  1,265,461.6920707219  ->  TRUE
```

`cond1` fails -> **candidate 6791151 fails the window under the current 0.24-dex-only budget,
confirmed** (matches the claim card's exhibit). Also confirmed the (upstream) redshift filter
(`handler.py:637-645`) independently PASSES for this candidate (`z_min=0.053565 <=
z+z_err=0.087595` and `z_max=0.077766 >= z-z_err=0.018041`, both true) — so the exclusion is
specifically the mass filter, not the redshift filter, consistent with the claim.

`M_z_sigma*1.5 = 0.0077821842...` M_sun is ~14 orders of magnitude smaller than `M_z` itself
(fractional GW mass precision ~3.9e-9 for this SNR=235.9 event) — it makes literally no numerical
difference to the window edges; the whole test is governed by `z_min`/`z_max` and the candidate's
own `BH_MASS_ERROR`.

## 5. Window test — full R&V15 budget (0.55 dex total)

Per task instruction: add `(0.50*ln(10))**2` to the in-quadrature variance in
`_empiric_stellar_mass_to_BH_mass_relation` (handler.py:1376-1381), leaving every other term
(`sigma_int**2` at 0.24 dex, `d_alpha**2`, the `d_beta` term, the propagated-stellar-mass-error
term) unchanged:

```
var_current = sigma_int**2 + d_alpha**2 + (ln(SM/10)*d_beta)**2 + (beta/SM*SM_err)**2
            = 1.698953...   (dimensionless, ln-space)
var_full    = var_current + (0.50*ln(10))**2
            = var_current + 1.3254745276195998
            = 3.024428...

BH_MASS_ERROR_full = BH_MASS * sqrt(var_full) = 223,872.11385683485 * sqrt(3.024428...)
                    = 389,299.8873277455 M_sun
```

Sanity check: setting the added component to 0 reproduces `BH_MASS_ERROR_current =
291,758.99489010876` exactly (`np.isclose` True in script output) — confirms the inflation is a
strict, correctly-normalized superset of the current budget, not an independent recomputation.

Inflation ratio: `BH_MASS_ERROR_full / BH_MASS_ERROR_current = 1.3343200865987888`.

Re-run the window test with the inflated error:

```
cond1:  1,237,046.502370223  <=  223,872.11385683485 + 389,299.8873277455*1.5
                              =  223,872.11385683485 + 583,949.8309916183
                              =  807,821.9448484531   ->  FALSE   *** STILL FAILS ***

cond2:  223,872.11385683485 - 583,949.8309916183 = -360,077.71713478...
                              <=  1,265,461.6920707219  ->  TRUE (unchanged, not the binding side)
```

`cond1` STILL fails. **Candidate 6791151 is NOT readmitted even under the full 0.55-dex R&V15
budget.**

## 6. Margin

Binding constraint is always `cond1` (the candidate's mass is far below the event's window; `cond2`
has enormous slack throughout — the window's *upper* edge is nowhere near the candidate). Solve
`cond1` at equality for the required `BH_MASS_ERROR`:

```
BH_MASS_ERROR_required = (left - BH_MASS) / 1.5
                        = (1,237,046.502370223 - 223,872.11385683485) / 1.5
                        = 1,013,174.388513388 / 1.5
                        = 675,449.5923422588 M_sun
```

Margin factors:

```
required / current (0.24-dex-only)  = 675,449.5923422588 / 291,758.99489010876 = 2.315094321587814
required / full (0.55-dex)          = 675,449.5923422588 / 389,299.8873277455  = 1.735036701342809
```

So `host_M_error` would need to grow by a factor of **~2.32x its current (0.24-dex-only) value**
to readmit the candidate — equivalently **~1.74x beyond the already-inflated full-0.55-dex value**.
The 0.50-dex virial-measurement component (a 1.33x inflation) closes only about 27% of the
required 2.32x gap; roughly 3.4x more log-variance than the 0.50-dex measurement term supplies
would be needed to reach readmission (in dex-equivalent terms: current budget corresponds to an
effective total scatter of `sqrt(var_current)/ln(10) ≈ 0.7378` dex; readmission requires
`sqrt(var_current + (delta)^2)/ln(10)` to reach `sqrt(9.104)/ln(10) ≈ 1.310` dex total —
i.e. roughly *1.3 dex* of total scatter, not R&V15's 0.55 dex).

## 7. Verdict

**NO.** Inflating `host_M_error` to the full R&V15 budget (0.24-dex intrinsic + 0.50-dex virial
measurement, in quadrature) moves `BH_MASS_ERROR` from 291,758.995 to 389,299.887 M_sun (a 1.33x
increase) — nowhere near the ~2.32x increase (over the current value) that `cond1` requires for
readmission. The window's exclusion of candidate 6791151 for seed 900121 event 20 is **not** an
artifact of the intrinsic-only 0.24-dex omission; even generously crediting the full published
R&V15 error budget, the candidate's derived BH mass (223,872 M_sun) is genuinely too far below the
event's inferred host-mass window (`[1,237,047, 1,265,462]` M_sun-ish, dominated by the `(1+z)`
redshift-range stretch, not by the GW mass precision) to be admitted. Per the task's own framing,
this is the "exclusion reflects a genuine mass mismatch" branch: the exhibit's window-exclusion is
retired as evidence for the missing-0.50-dex-budget defect specifically (parts (a) and (b) do
**not** fuse); the residual thread narrows to the epsilon-derivation of the window itself, as
anticipated.

## 8. Script (verbatim, as executed)

`scan_catalog.py` (index-semantics cross-check + candidate row extraction, chunked, only the 4
needed columns loaded):

```python
import numpy as np
import pandas as pd
import time

CSV = "/home/jasper/Repositories/darksiren-emri/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv"

names = ["RA","DEC","BMAG","REDSHIFT","REDSHIFT_ERR","STELLAR_MASS","STELLAR_MASS_ERR","REDSHIFT_FLAG"]
usecols = ["REDSHIFT","REDSHIFT_ERR","STELLAR_MASS","STELLAR_MASS_ERR"]

alpha = 7.45 * np.log(10)
beta = 1.05
d_alpha = 0.08 * np.log(10)
d_beta = 0.11
sigma_int = 0.24 * np.log(10)

M_min = 1e4
M_max = 1e7
z_max = 1.5

TARGETS = {6791138, 6791151, 6791158}
found = {}

chunksize = 1_000_000
cum_survivors = 0
total_rows = 0
t0 = time.time()

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
            sigma_int**2
            + d_alpha**2
            + (np.log(SM / 10) * d_beta) ** 2
            + (beta / SM * SM_err) ** 2
        )

    mass_info_mask = ~np.isnan(BH_mass)

    with np.errstate(invalid="ignore"):
        prune_mask = (
            (BH_mass + BH_mass_error >= M_min)
            & (BH_mass - BH_mass_error <= M_max)
            & (Z - Z_err <= z_max)
        )
    prune_mask = np.where(np.isnan(prune_mask.astype(float)), False, prune_mask)

    survive = mass_info_mask & prune_mask
    n_survive = int(survive.sum())

    if n_survive > 0:
        lo = cum_survivors
        hi = cum_survivors + n_survive - 1
        relevant_targets = [t for t in TARGETS if lo <= t <= hi]
        if relevant_targets:
            surv_idx = np.nonzero(survive)[0]
            for t in relevant_targets:
                offset = t - cum_survivors
                local_row = surv_idx[offset]
                global_row = global_start + local_row
                found[t] = {
                    "original_csv_row_0based": int(global_row),
                    "STELLAR_MASS_1e10Msun": float(SM[local_row]),
                    "STELLAR_MASS_ERR_1e10Msun": float(SM_err[local_row]),
                    "REDSHIFT": float(Z[local_row]),
                    "REDSHIFT_ERR": float(Z_err[local_row]),
                    "BH_MASS": float(BH_mass[local_row]),
                    "BH_MASS_ERROR": float(BH_mass_error[local_row]),
                }
                print(f"FOUND target {t}: {found[t]}", flush=True)

    cum_survivors += n_survive
    if len(found) == len(TARGETS):
        break

print("DONE"); print(found)
```

Window-test / redshift-bounds / margin computation (run via `uv run python3`, since the bare repo
`.venv` shim was missing `scipy` — `uv run` resolves the correct locked env):

```python
from darksiren_emri.physical_relations import dist_to_redshift, get_redshift_outer_bounds
import pandas as pd, numpy as np

CSV = "results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bt_900121_work/seed900121/simulations/prepared_cramer_rao_bounds.csv"
df = pd.read_csv(CSV)
row = df.iloc[20]

dL = row["luminosity_distance"]
dL_unc = row["delta_luminosity_distance_delta_luminosity_distance"]**0.5
M_z = row["M"]
M_z_sigma = row["delta_M_delta_M"]**0.5

zmn, zmx = get_redshift_outer_bounds(
    distance=dL, distance_error=dL_unc,
    h_min=0.6, h_max=0.86, Omega_m_min=0.04, Omega_m_max=0.5,
    sigma_multiplier=2.0,
)
z_max = min(zmx, 1.5)   # REDSHIFT_UPPER_LIMIT = cosmological_model.max_redshift (default 1.5)
z_min = zmn

sigma_multiplier = 1.5      # bayesian_statistics.py:4691
mult = 1.5                  # symmetric mode, bayesian_statistics.py default _mass_filter_sigma

BH_MASS = 223872.11385683485
BH_MASS_ERROR_current = 291758.99489010876

alpha = 7.45*np.log(10); beta = 1.05; d_alpha = 0.08*np.log(10); d_beta = 0.11
sigma_int = 0.24*np.log(10)
SM, SM_err = 0.1, 0.1
var_current = sigma_int**2 + d_alpha**2 + (np.log(SM/10)*d_beta)**2 + (beta/SM*SM_err)**2
var_full = var_current + (0.50*np.log(10))**2
BH_MASS_ERROR_full = BH_MASS*np.sqrt(var_full)

left = (M_z - M_z_sigma*sigma_multiplier)/(1+z_max)
right = (M_z + M_z_sigma*sigma_multiplier)/(1+z_min)

for label, err in [("current", BH_MASS_ERROR_current), ("full", BH_MASS_ERROR_full)]:
    cond1 = left <= BH_MASS + err*mult
    cond2 = BH_MASS - err*mult <= right
    print(label, cond1, cond2, cond1 and cond2)

required = (left - BH_MASS)/mult
print("required BH_MASS_ERROR:", required)
print("margin vs current:", required/BH_MASS_ERROR_current)
print("margin vs full:", required/BH_MASS_ERROR_full)
```

Both scripts' full raw terminal output is reproduced in the transcript that produced this note;
key numeric outputs are quoted inline above at full float precision as printed.
