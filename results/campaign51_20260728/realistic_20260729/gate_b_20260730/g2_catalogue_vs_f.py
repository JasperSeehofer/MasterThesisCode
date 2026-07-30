"""Gate C item 1+4, step 2.

Does the mass-pruned catalogue's rate-weighted (z, M) distribution reproduce the
completeness model's f_bar(z) * p_pop(z) that beta_G integrates?  This is the
Option-A modeling assumption (constant comoving rate-weight density n_w).

Everything here is LOCAL.  CAVEAT: the local reduced_galaxy_catalogue.csv differs
from the cluster realization parent in exactly the z_error column (#40b PV width),
which enters the prune via (z - z_err <= z_max) and (M +- sigma_M in band).  Shape
conclusions are INDICATIVE at the few-% level, not exact.
"""

import json

import numpy as np
import pandas as pd

from master_thesis_code.constants import HOST_DRAW_Z_MAX, M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from master_thesis_code.dark_siren_injection import _redshift_population_weight
from master_thesis_code.emri_rate import R_eff_per_mbh
from master_thesis_code.galaxy_catalogue.handler import (
    REDUCED_CATALOGUE_FILE_PATH,
    InternalCatalogColumns,
    _empiric_stellar_mass_to_BH_mass_relation,
    _reduced_catalog_column_names,
)
from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build

OUT = "results/campaign51_20260728/realistic_20260729/gate_b_20260730"
Z_MAX_PRUNE = 1.5

cat = pd.read_csv(REDUCED_CATALOGUE_FILE_PATH, names=_reduced_catalog_column_names())
print("raw rows:", len(cat))
M, dM = _empiric_stellar_mass_to_BH_mass_relation(
    cat[InternalCatalogColumns.BH_MASS], cat[InternalCatalogColumns.BH_MASS_ERROR]
)
cat[InternalCatalogColumns.BH_MASS] = M
cat[InternalCatalogColumns.BH_MASS_ERROR] = dM
cat = cat[~cat[InternalCatalogColumns.BH_MASS].isna()]
print("with mass info:", len(cat))
z = cat[InternalCatalogColumns.REDSHIFT].to_numpy(float)
ze = cat[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(float)
Mb = cat[InternalCatalogColumns.BH_MASS].to_numpy(float)
Me = cat[InternalCatalogColumns.BH_MASS_ERROR].to_numpy(float)
mask = (Mb + Me >= M_SOURCE_FRAME_MIN) & (Mb - Me <= M_SOURCE_FRAME_MAX) & (z - ze <= Z_MAX_PRUNE)
z, Mb, ze = z[mask], Mb[mask], ze[mask]
print("pruned rows:", len(z))

# draw-eligible set of draw_rate_weighted_hosts / W_cat: z_g < HOST_DRAW_Z_MAX
elig = z < HOST_DRAW_Z_MAX
zg, Mg = z[elig], Mb[elig]
w_g = R_eff_per_mbh(Mg) / (1.0 + zg)
W_cat = float(np.sum(w_g))
print(f"draw-eligible galaxies (z<{HOST_DRAW_Z_MAX}): {zg.size},  W_cat = {W_cat:.6e}")

comp = from_cache_or_build()
h = 0.73
zgrid = np.linspace(1e-6, HOST_DRAW_Z_MAX, 4096)
fbar = np.clip(comp.f_bar(zgrid, h), 0, 1)
ppop = _redshift_population_weight(zgrid, h)

F = float(np.trapezoid(fbar * ppop, zgrid) / np.trapezoid(ppop, zgrid))
V_f = float(np.trapezoid(fbar * ppop, zgrid))  # = precompute_completeness_population_volume
V_tot = float(np.trapezoid(ppop, zgrid))
print(f"F (global in-catalogue fraction, h=0.73) = {F:.6f}")
print(
    f"V_f(0.73) = {V_f:.6e} Mpc^3/sr,  V_tot = {V_tot:.6e},  n_hat_w = W_cat/V_f = {W_cat / V_f:.6e}"
)

# ---- z-shape comparison: catalogue rate-weight density vs f_bar*p_pop --------
edges = np.concatenate([np.linspace(0, 0.3, 31), np.linspace(0.35, 1.5, 24)])
hist_w, _ = np.histogram(zg, bins=edges, weights=w_g)
cen = 0.5 * (edges[1:] + edges[:-1])
wid = np.diff(edges)
dWdz = hist_w / wid
model_dens = np.interp(cen, zgrid, fbar * ppop)  # in-cat model density, unnormalised
n_w_of_z = np.where(model_dens > 0, dWdz / np.maximum(model_dens, 1e-300), np.nan)
print(
    "\n  z_lo   z_hi     dW/dz(cat)     f*p_pop(model)   n_w(z)=ratio   f_bar   cum_cat  cum_model"
)
cum_c = np.cumsum(hist_w) / W_cat
mm = model_dens * wid
cum_m = np.cumsum(mm) / np.sum(mm)
for i in range(len(cen)):
    if i % 3 == 0 or cen[i] > 0.3:
        print(
            f"{edges[i]:6.3f} {edges[i + 1]:6.3f}  {dWdz[i]:13.5e}  {model_dens[i]:13.5e}  {n_w_of_z[i]:12.5e}"
            f"  {np.interp(cen[i], zgrid, fbar):7.4f}  {cum_c[i]:7.4f}  {cum_m[i]:7.4f}"
        )

json.dump(
    dict(
        W_cat=W_cat,
        V_f=V_f,
        V_tot=V_tot,
        F=F,
        n_hat_w=W_cat / V_f,
        n_eligible=int(zg.size),
        n_pruned=int(z.size),
    ),
    open(f"{OUT}/g2_catalogue_summary.json", "w"),
    indent=1,
)
np.savez(
    f"{OUT}/g2_zshape.npz",
    edges=edges,
    hist_w=hist_w,
    model_dens=model_dens,
    zgrid=zgrid,
    fbar=fbar,
    ppop=ppop,
    zg=zg,
    Mg=Mg,
    w_g=w_g,
)
