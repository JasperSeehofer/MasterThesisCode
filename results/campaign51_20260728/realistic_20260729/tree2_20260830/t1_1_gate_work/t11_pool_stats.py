"""T1.1 gate-work (b): pool z / z_error extremes, eligibility at h=0.73, and the spec-z zero-error question
(does any catalogue row carry z_error == 0, which would make the transformed window degenerate at every theta?).
Zero evaluate() calls. Launched under row #255 - tree 2 node T1.1."""
import json, sys, time
import numpy as np
sys.path.insert(0, '/home/jasper/Repositories/darksiren-emri')
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.physical_relations import dist_to_redshift
t0 = time.time()
handler = c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH); pool = c1d._host_pool_from_handler(handler)
z = pool.z; ze = pool.z_error
out = {'pool_n': int(pool.n), 'z_min': float(z.min()), 'z_max': float(z.max()), 'z_error_min': float(ze.min()),
       'z_error_max': float(ze.max()), 'n_z_error_le_0': int((ze <= 0).sum()), 'n_z_error_lt_1e-6': int((ze < 1e-6).sum()),
       'n_z_error_lt_1e-4': int((ze < 1e-4).sum()), 'median_z_error_over_1pz': float(np.median(ze / (1 + z))),
       'n_rows_z_ge_1.5': int((z >= 1.5).sum()), 'n_M_nonfinite_or_le0': int((~np.isfinite(pool.M) | (pool.M <= 0)).sum())}
# degenerate definition check at (0,1): den_hi = z + 4*ze must exceed den_lo = max(z - 4 ze, 1e-6)
den_lo = np.maximum(z - 4 * ze, 1e-6); den_hi = z + 4 * ze
out['n_degenerate_at_truth'] = int((den_hi <= den_lo).sum())
out['load_s'] = time.time() - t0
json.dump(out, open('results/campaign51_20260728/realistic_20260729/tree2_20260830/t1_1_gate_work/t11_pool_stats_out.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
