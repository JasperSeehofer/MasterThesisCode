"""Gate C item 4, sharpest form: at FIXED redshift the model predicts
P(in-catalogue | detected, z) = f_bar(z) exactly -- p_det and dVc/(1+z) are common
to both mixture legs and cancel.  Any deviation is a difference between the two
legs' *mass* distributions (or a failure of f itself).  Fully local, p_det-free."""

import json

import numpy as np
import pandas as pd

from master_thesis_code.dark_siren_injection import compute_global_catalog_fraction
from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build
from master_thesis_code.physical_relations import dist_to_redshift

BASE = "results/campaign51_20260728/realistic_20260729"
OUT = f"{BASE}/gate_b_20260730"
comp = from_cache_or_build()
for h in (0.60, 0.73, 0.86):
    print(
        f"compute_global_catalog_fraction(h={h}) = {compute_global_catalog_fraction(comp, h=h):.6f}"
    )
F = compute_global_catalog_fraction(comp, h=0.73)

rows = []
for seed in (61000, 62000):
    df = pd.read_csv(f"{BASE}/seed{seed}/prepared_cramer_rao_bounds.csv")
    z = np.array([dist_to_redshift(dl, h=0.73) for dl in df.luminosity_distance])
    rows.append(pd.DataFrame(dict(z=z, inc=(df.host_galaxy_index >= 0).to_numpy())))
d = pd.concat(rows)
print(
    f"\npooled detected events {len(d)}, in-cat {d.inc.sum()}; z range {d.z.min():.4f}..{d.z.max():.3f}"
)
edges = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.9, 1.6])
print("\n  z bin        N_det   n_incat   realized P(in|det,z)   f_bar(z) [model]   ratio")
for i in range(len(edges) - 1):
    m = (d.z >= edges[i]) & (d.z < edges[i + 1])
    n = int(m.sum())
    k = int(d.inc[m].sum())
    if n == 0:
        continue
    zc = float(np.median(d.z[m]))
    fb = float(np.clip(comp.f_bar(np.array([zc]), 0.73), 0, 1)[0])
    print(
        f" {edges[i]:5.2f}-{edges[i + 1]:4.2f} {n:8d} {k:8d}      {k / n:10.4f}          {fb:10.4f}    {k / n / max(fb, 1e-9):7.3f}"
    )
tot_f = np.mean([np.clip(comp.f_bar(np.array([zz]), 0.73), 0, 1)[0] for zz in d.z])
print(
    f"\n  detection-weighted mean f_bar(z_i) over the detected sample = {tot_f:.5f}"
    f"   (this IS the model's P(in-cat|detected) marginalised over the realized z's)"
)
print(
    f"  realized in-cat fraction                                    = {d.inc.mean():.5f}"
    f"   -> ratio {d.inc.mean() / tot_f:.4f}"
)
json.dump(
    dict(F=F, mean_fbar_over_detected=float(tot_f), realized=float(d.inc.mean())),
    open(f"{OUT}/g6_results.json", "w"),
    indent=1,
)
