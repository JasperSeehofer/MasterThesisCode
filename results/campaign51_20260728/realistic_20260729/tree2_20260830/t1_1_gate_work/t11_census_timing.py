"""T1.1 gate-work: (a) md5 pin of the reduced catalogue; (b) E20 edge-case census at b=-0.02 (reproduce
63,036 / 15,618) and at the S0-B node b=-0.033; w_g weight-share upper bound of the degenerate rows;
(c) wall-time of the C7-kernel GL-50 smear (the harness kernel_smeared_survival, the same integral the
registered divisor evaluates) on a 200k-row pool subsample at two theta nodes, single process, to
cost the divisor pass per (theta-node, h). Zero evaluate() calls. Launched under row #255 - tree 2 node T1.1."""
import hashlib, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/jasper/Repositories/darksiren-emri')
t0 = time.time()
import darksiren_emri.validation.correspondence_1d as c1d
from darksiren_emri.emri_rate import R_eff_per_mbh
out = {}
p = c1d.REDUCED_CATALOGUE_PATH
h = hashlib.md5()
with open(p, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 24), b''):
        h.update(chunk)
out['catalogue_md5'] = h.hexdigest(); out['md5_s'] = time.time() - t0
t1 = time.time()
handler = c1d._load_galaxy_catalog_handler(p); pool = c1d._host_pool_from_handler(handler)
out['pool_load_s'] = time.time() - t1; out['pool_n'] = int(pool.n)
z = pool.z; ze = pool.z_error; w = np.asarray(R_eff_per_mbh(pool.M), float) / (1.0 + z)
wsum = float(w.sum())
for b in (-0.02, -0.033):
    zc = z + b * (1.0 + z)
    neg = zc < 0.0
    inv = (zc + 4.0 * ze) <= 1e-6
    out[f'census_b={b}'] = {'neg_centre_rows': int(neg.sum()), 'neg_centre_frac': float(neg.mean()),
        'inverted_window_rows': int(inv.sum()), 'inverted_window_frac': float(inv.mean()),
        'neg_centre_w_share_upper': float(w[neg].sum() / wsum), 'inverted_w_share_upper': float(w[inv].sum() / wsum),
        'z_g_threshold_neg_centre': float(-b / (1.0 + b))}
    for s in (1.0, 1 / math.sqrt(2)):
        inv_s = (zc + 4.0 * s * ze) <= 1e-6
        out[f'census_b={b}'][f'inverted_window_rows_s={s:.4f}'] = int(inv_s.sum())
out['z_lt_0.03_frac'] = float((z < 0.03).mean())
# (c) timing of the C7-kernel smear on 200k rows with the completeness cache and a DUMMY unit survival
# table (S_bar_phi == 1 everywhere -> S_tilde == 1 exactly, a self-check), so no detection-probability grid build.
t2 = time.time()
from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build
completeness = from_cache_or_build(); out['completeness_load_s'] = time.time() - t2
H = c1d.H_TRUE
zg = np.linspace(1e-6, 1.5, 1500); table = {H: (zg, np.ones_like(zg))}
rng = np.random.default_rng(7); sub = rng.choice(pool.n, size=200_000, replace=False)
zs = z[sub]; zes = ze[sub]; ph = pool.phiS[sub]; q = pool.qS[sub]
for name, (bb, ss) in {'truth': (0.0, 1.0), 'b_minus': (-0.02, 1.0)}.items():
    t3 = time.time()
    S = c1d.kernel_smeared_survival(zs + bb * (1 + zs), ss * zes, table, completeness, ph, q, h=H)
    dt = time.time() - t3
    out[f'smear200k_{name}'] = {'wall_s': dt, 'per_row_us': 1e6 * dt / 2e5, 'scaled_to_pool_s': dt * pool.n / 2e5,
        'S_min': float(np.nanmin(S)), 'S_max': float(np.nanmax(S)), 'n_nonfinite': int((~np.isfinite(S)).sum()),
        'n_neg': int((S < 0).sum()), 'n_gt_1p1': int((S > 1.1).sum())}
out['total_s'] = time.time() - t0
json.dump(out, open('results/campaign51_20260728/realistic_20260729/tree2_20260830/t1_1_gate_work/t11_census_timing_out.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
