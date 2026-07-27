"""S1 — ESS-floor pool-sizing scan for the campaign redesign (issue #51).

Question: what (sampling measure, N) does the re-injection campaign need so
that the FIX-3 §7.1 joint (u = ln(1+z), m = log10 M_z) with-BH survival grid
has enough per-node ESS that the ratified (K5) shrinkage
w = ESS/(ESS + n0), n0 = 10, is measured inert on the catalogue's query
support (catalogue-weighted w_bar >= 0.99)?

New support (Amendment 2): source-frame M in [1e4, 1e7], z in (0, 1.5]
=> detector-frame m in [4, 7 + log10(2.5) = 7.39794].

Kernel/ESS conventions replicated from the ratified Z2/Z4 gates
(docs/derivations/fix3_zmz_catalog_selection.md §3.2-§3.4) and the repo
implementation (sdp.py:693-698 ESS; _abramson_lambda_u; z2_zres_slopes.py
build_surv_ulm):
  - product Gaussian kernel in (u, m)
  - Scott d=2 bandwidth  sigma_j = N^(-1/6) * std_j  on BOTH axes  [RATIFY-Z2]
  - Abramson sqrt-law adaptivity on u ONLY, pilot = histogram KDE at the
    kernel sigma_u (sdp.py convention: pilot bandwidth = kernel bandwidth)
  - ESS = (sum w)^2 / sum w^2 per node (Kish), d_L-independent

ESS needs only kernel weights (no waveforms, no d_L), so the whole
(measure x N) scan runs on synthetic (M, z) draws from density grids.

Sampling measures (densities over (lg = log10 M_source, z)):
  a        status-quo Babak M1 rate density emri_distribution(M, z)
           (the EXACT density the emcee sampler targets in these coordinates,
           Model1CrossCheck._log_probability), widened to the new box
  b_rate   log-uniform in M  x  z-marginal of measure (a)
  b_vol    log-uniform in M  x  dVc/dz / (1+z)  (uniform in comoving
           detector-frame event rate volume)
  cat      catalogue-coverage measure: proportional to the GLADE+ R_eff/(1+z)
           rate-weight profile W_z_lm (z0 packet), restricted to the
           physically reachable set m <= 7 + log10(1+z)  (source M <= 1e7)
  mix_aXX  alpha * a + (1 - alpha) * cat,  alpha in {0.75, 0.5, 0.25}
  flat_um  uniform in (u, m) on the reachable region (variance-uniformizing
           design measure for a KDE on a box)

Outputs sizing_results.json.  Read-only w.r.t. master_thesis_code/.
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/home/jasper/Repositories/MasterThesisCode")

from master_thesis_code.cosmological_model import Model1CrossCheck  # noqa: E402
from master_thesis_code.physical_relations import comoving_volume_element  # noqa: E402

RNG = np.random.default_rng(20260728)
BASE = "/home/jasper/Repositories/MasterThesisCode/results/lcat_h_dependence_20260725"
OUT = f"{BASE}/campaign_sizing_20260728"

# ---------------- new-support constants (Amendment 2) ----------------
LG_MIN, LG_MAX = 4.0, 7.0  # source-frame log10 M
Z_MIN, Z_MAX = 1e-4, 1.5
U_MAX = float(np.log1p(Z_MAX))
M_LO, M_HI = 4.0, 7.0 + float(np.log10(2.5))  # detector-frame m support
N0 = 10.0  # (K5) pseudo-count, _MIN_BAND_INJECTIONS
N_SCAN = [50_000, 100_000, 200_000, 500_000, 1_000_000]
N_DRAW = max(N_SCAN)

# grid noding: probe-parity SPACING. Probe: 61 u on [0, ln 2.5] (0.01527/step),
# 31 m over the truncated pool range [4.565, 6.000] (0.0478 dex/step).
# u support unchanged -> keep 61 u-nodes. m: 3.398 dex / 0.05 = 68 steps ->
# 69 m-nodes at exactly 0.05 dex (probe-comparable 0.0478 -> 0.05).
U_NODES = np.linspace(0.0, U_MAX, 61)
M_NODES = np.linspace(M_LO, M_HI, 69)
M_NODES_COARSE = np.linspace(M_LO, M_HI, 31)  # probe COUNT parity (sensitivity)


def ridge_m(z: np.ndarray) -> np.ndarray:
    """Max reachable detector-frame m at redshift z (source cap 1e7)."""
    return 7.0 + np.log10(1.0 + z)


# ---------------- density grids ----------------
NLG, NZ = 601, 601
lg_grid = np.linspace(LG_MIN, LG_MAX, NLG)
z_grid = np.linspace(Z_MIN, Z_MAX, NZ)

# measure (a): emri_distribution on (lg, z) — the emcee target density in
# exactly these coordinates (x = [log10 M, z], log-prob = ln emri_distribution)
rho_a = np.empty((NLG, NZ))
for i, lg in enumerate(lg_grid):
    M = 10.0**lg
    rate = Model1CrossCheck.R_emri(M)
    dndz = np.array(
        [Model1CrossCheck.dN_dz_of_mass(M, float(zz)) for zz in z_grid]
    )
    rho_a[i] = dndz * rate
neg_frac_a = float((rho_a < 0).mean())
rho_a = np.clip(rho_a, 0.0, None)
rho_a /= rho_a.sum()

# measure (b_rate): log-uniform lg x z-marginal of (a)
pz_rate = rho_a.sum(axis=0)
rho_b_rate = np.tile(pz_rate, (NLG, 1))
rho_b_rate /= rho_b_rate.sum()

# measure (b_vol): log-uniform lg x dVc/dz/(1+z)
dvc = np.asarray(comoving_volume_element(z_grid, h=0.73), dtype=np.float64)
pz_vol = dvc / (1.0 + z_grid)
rho_b_vol = np.tile(pz_vol / pz_vol.sum(), (NLG, 1))
rho_b_vol /= rho_b_vol.sum()

# measure (cat): catalogue profile W_z_lm -> density on (lg, z).
# m = lg + log10(1+z) is a shear with unit Jacobian, so the (z, m) cell
# density maps directly to (lg, z). Restrict to reachable cells
# (m <= ridge <=> source M <= 1e7) and to m >= 4 (source M >= 1e4 within z).
prof = json.load(open(f"{BASE}/zres_survival/catalog_zw_profile.json"))
z_edges_c = np.array(prof["z_edges"])
lm_edges_c = np.array(prof["lm_edges"])
W_zlm = np.array(prof["W_z_lm"])  # (300 z, 60 lm)
zc_c = 0.5 * (z_edges_c[:-1] + z_edges_c[1:])
lmc_c = 0.5 * (lm_edges_c[:-1] + lm_edges_c[1:])
cell_area = np.outer(np.diff(z_edges_c), np.diff(lm_edges_c))  # (300, 60)
dens_cat_zm = W_zlm / cell_area  # density in (z, m)

ZZ = z_grid[None, :]  # broadcast over (lg, z)
LG = lg_grid[:, None]
MM = LG + np.log10(1.0 + ZZ)  # detector-frame m at each (lg, z) grid point
# bilinear-free lookup: nearest catalogue cell (piecewise-constant density)
iz = np.clip(np.searchsorted(z_edges_c, ZZ, side="right") - 1, 0, len(zc_c) - 1)
im = np.clip(np.searchsorted(lm_edges_c, MM, side="right") - 1, 0, len(lmc_c) - 1)
rho_cat = dens_cat_zm[iz, im]
rho_cat = np.where((MM >= lm_edges_c[0]) & (MM < lm_edges_c[-1]), rho_cat, 0.0)
# reachability is automatic: the (lg, z) box IS the reachable set.
cat_unreachable_wfrac = float(
    W_zlm[np.add.outer(np.zeros(len(zc_c)), lmc_c) > ridge_m(zc_c)[:, None]].sum()
    / W_zlm.sum()
)
rho_cat = rho_cat * 1.0  # (already zero outside profile support)
rho_cat_sum = rho_cat.sum()
rho_cat /= rho_cat_sum

# measure (flat_um): uniform in (u, m) on the box == density prop. to
# Jacobian |d(u,m)/d(lg,z)| = 1/(1+z) * ln10 ... in (lg, z) coords:
# u = ln(1+z), m = lg + log10(1+z): du dm = dz/(1+z) * dlg  => density in
# (lg, z) prop. to 1/(1+z).
rho_flat = np.tile((1.0 / (1.0 + z_grid))[None, :], (NLG, 1))
rho_flat /= rho_flat.sum()

MEASURES: dict[str, np.ndarray] = {
    "a": rho_a,
    "b_rate": rho_b_rate,
    "b_vol": rho_b_vol,
    "cat": rho_cat,
    "mix_a75": 0.75 * rho_a + 0.25 * rho_cat,
    "mix_a50": 0.50 * rho_a + 0.50 * rho_cat,
    "mix_a25": 0.25 * rho_a + 0.75 * rho_cat,
    "flat_um": rho_flat,
    "mix_f50": 0.50 * rho_a + 0.50 * rho_flat,
}

dlg = lg_grid[1] - lg_grid[0]
dz = z_grid[1] - z_grid[0]


def draw(rho: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-CDF cell draw + uniform in-cell jitter -> (lg, z) samples."""
    p = rho.ravel()
    idx = RNG.choice(p.size, size=n, p=p / p.sum())
    ilg, izz = np.unravel_index(idx, rho.shape)
    lg = lg_grid[ilg] + (RNG.random(n) - 0.5) * dlg
    zz = z_grid[izz] + (RNG.random(n) - 0.5) * dz
    return np.clip(lg, LG_MIN, LG_MAX), np.clip(zz, Z_MIN, Z_MAX)


def abramson_lambda_u(u: np.ndarray, sigma_u: float) -> np.ndarray:
    """Replicates sdp.py _abramson_lambda_u (400 pilot bins, taps cap)."""
    n_bins = 400
    u_max = float(np.max(u))
    edges = np.linspace(0.0, max(u_max, sigma_u), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    du = float(edges[1] - edges[0])
    hist, _ = np.histogram(u, bins=edges, density=True)
    kh = int(min(np.ceil(4.0 * sigma_u / du), (n_bins - 1) // 2))
    taps = np.exp(-0.5 * (np.arange(-kh, kh + 1) * du / sigma_u) ** 2)
    pilot = np.convolve(hist, taps / taps.sum(), mode="same")
    pilot = np.clip(pilot, 1e-12, None)
    f_at = np.interp(u, centers, pilot)
    g_mean = float(np.exp(np.mean(np.log(f_at))))
    return np.sqrt(g_mean / f_at)


def ess_grid(
    u: np.ndarray, m: np.ndarray, u_nodes: np.ndarray, m_nodes: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Per-node ESS of the product kernel via factorized chunked matmuls.

    w_k(a,b) = wu_k(a) * wm_k(b)  =>  S1 = wu^T wm,  S2 = (wu^2)^T (wm^2).
    """
    n = len(u)
    sigma_u = float(n ** (-1.0 / 6.0) * u.std())
    sigma_m = float(n ** (-1.0 / 6.0) * m.std())
    lam = abramson_lambda_u(u, sigma_u)
    sig_i = sigma_u * lam
    S1 = np.zeros((len(u_nodes), len(m_nodes)))
    S2 = np.zeros_like(S1)
    for lo in range(0, n, 200_000):
        hi = min(lo + 200_000, n)
        wu = np.exp(-0.5 * ((u[lo:hi, None] - u_nodes[None, :]) / sig_i[lo:hi, None]) ** 2)
        wm = np.exp(-0.5 * ((m[lo:hi, None] - m_nodes[None, :]) / sigma_m) ** 2)
        S1 += wu.T @ wm
        S2 += (wu * wu).T @ (wm * wm)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(S2 > 0, S1 * S1 / S2, 0.0)
    return ess, sigma_u, sigma_m


# ---------------- catalogue query nodes (projection of W_z_lm) ----------------
CZ, CM = np.meshgrid(zc_c, lmc_c, indexing="ij")
CW = W_zlm.copy()
sel = CW > 0
q_z = CZ[sel]
q_m_raw = CM[sel]
q_w = CW[sel]
q_u = np.log1p(q_z)
q_m = np.clip(q_m_raw, M_LO, M_HI)  # grid-box clamp (production A2-EXTRAP analog)
q_reach = q_m_raw <= ridge_m(q_z)  # physically coverable queries
W_TOT = float(q_w.sum())


def bilinear(ess: np.ndarray, u_nodes: np.ndarray, m_nodes: np.ndarray,
             uq: np.ndarray, mq: np.ndarray) -> np.ndarray:
    a = np.interp(uq, u_nodes, np.arange(len(u_nodes)))
    b = np.interp(mq, m_nodes, np.arange(len(m_nodes)))
    a0 = np.clip(np.floor(a).astype(int), 0, len(u_nodes) - 2)
    b0 = np.clip(np.floor(b).astype(int), 0, len(m_nodes) - 2)
    fa, fb = a - a0, b - b0
    return ((1 - fa) * (1 - fb) * ess[a0, b0] + fa * (1 - fb) * ess[a0 + 1, b0]
            + (1 - fa) * fb * ess[a0, b0 + 1] + fa * fb * ess[a0 + 1, b0 + 1])


def wmedian(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def cat_metrics(ess: np.ndarray, u_nodes: np.ndarray, m_nodes: np.ndarray) -> dict:
    e_q = bilinear(ess, u_nodes, m_nodes, q_u, q_m)
    w_shrink = e_q / (e_q + N0)
    out = {}
    for tag, mask in (("all_clamped", np.ones_like(q_reach)), ("reachable", q_reach)):
        w = q_w[mask]
        e = e_q[mask]
        s = w_shrink[mask]
        out[tag] = {
            "wbar": float(np.sum(w * s) / np.sum(w)),
            "median_ESS": wmedian(e, w),
            "wfrac_ESS_lt_10": float(np.sum(w[e < 10]) / np.sum(w)),
            "wfrac_ESS_lt_100": float(np.sum(w[e < 100]) / np.sum(w)),
            "wfrac_ESS_lt_500": float(np.sum(w[e < 500]) / np.sum(w)),
        }
    return out


# reachable grid nodes (for grid-floor metrics)
node_reach = M_NODES[None, :] <= (7.0 + U_NODES[:, None] / np.log(10.0))

# ---------------- importance-weight diagnostics for pool-marginal legs -------
# If the campaign samples from rho_s != rho_a, every POOL-MARGINAL estimator
# (pooled survival, FIX-2 z-only S(d_L|z) tower identity, m-marginal shrink
# target, and any leg averaging over the pool) must carry per-injection
# importance weights v = rho_a / rho_s. Kish ESS of v measures the cost.
def iw_kish_frac(name: str, lg: np.ndarray, zz: np.ndarray) -> float:
    rho_s = MEASURES[name]
    ilg = np.clip(np.searchsorted(lg_grid, lg) - 0, 0, NLG - 1)
    ilg = np.clip(np.rint((lg - LG_MIN) / dlg).astype(int), 0, NLG - 1)
    izz = np.clip(np.rint((zz - Z_MIN) / dz).astype(int), 0, NZ - 1)
    num = rho_a[ilg, izz]
    den = rho_s[ilg, izz]
    v = np.where(den > 0, num / np.clip(den, 1e-300, None), 0.0)
    s1, s2 = v.sum(), (v * v).sum()
    return float(s1 * s1 / (s2 * len(v))) if s2 > 0 else 0.0


# ---------------- the scan ----------------
def _main() -> None:
    results: dict = {
        "meta": {
            "support": {"lg": [LG_MIN, LG_MAX], "z": [Z_MIN, Z_MAX],
                        "m": [M_LO, M_HI], "u": [0.0, U_MAX]},
            "grid": {"u_nodes": len(U_NODES), "m_nodes": len(M_NODES),
                     "m_nodes_coarse": len(M_NODES_COARSE),
                     "u_spacing": float(U_NODES[1] - U_NODES[0]),
                     "m_spacing": float(M_NODES[1] - M_NODES[0])},
            "kernel": "product Gaussian (u,m); Scott d=2 N^(-1/6)*std both axes; "
                      "Abramson sqrt-law on u (sdp.py pilot, 400 bins)",
            "n0": N0,
            "neg_density_frac_measure_a_clipped": neg_frac_a,
            "catalogue": {
                "W_total": W_TOT,
                "wfrac_m_gt_box": float(q_w[q_m_raw > M_HI].sum() / W_TOT),
                "wfrac_unreachable_ridge": cat_unreachable_wfrac,
                "wfrac_z_lt_0p3": float(q_w[q_z < 0.3].sum() / W_TOT),
                "wfrac_m_6_to_7p4": float(
                    q_w[(q_m_raw > 6.0) & (q_m_raw <= M_HI)].sum() / W_TOT),
            },
            "seed": 20260728,
        },
        "measures": {},
    }

    for name, rho in MEASURES.items():
        print(f"=== measure {name} ===", flush=True)
        lg_s, z_s = draw(rho, N_DRAW)
        u_s = np.log1p(z_s)
        m_s = lg_s + np.log10(1.0 + z_s)
        mrec: dict = {
            "frac_m_gt_6": float((m_s > 6.0).mean()),
            "frac_m_gt_6_and_z_lt_0p3": float(((m_s > 6.0) & (z_s < 0.3)).mean()),
            "frac_z_lt_0p3": float((z_s < 0.3).mean()),
            "iw_kish_frac_marginal_legs": iw_kish_frac(name, lg_s, z_s),
            "by_N": {},
        }
        for n in N_SCAN:
            u, m = u_s[:n], m_s[:n]
            ess, su, sm = ess_grid(u, m, U_NODES, M_NODES)
            rec = {
                "sigma_u": su, "sigma_m": sm,
                "std_u": float(u.std()), "std_m": float(m.std()),
                "grid_min_ESS_reachable": float(ess[node_reach].min()),
                "grid_frac_reachable_ESS_lt_100": float((ess[node_reach] < 100).mean()),
                "grid_frac_reachable_ESS_lt_500": float((ess[node_reach] < 500).mean()),
                "catalogue": cat_metrics(ess, U_NODES, M_NODES),
            }
            if n == 200_000:  # noding sensitivity at one N
                ess_c, _, _ = ess_grid(u, m, U_NODES, M_NODES_COARSE)
                rec["catalogue_coarse_31m"] = cat_metrics(ess_c, U_NODES, M_NODES_COARSE)
            mrec["by_N"][str(n)] = rec
            print(f"  N={n}: wbar_reach={rec['catalogue']['reachable']['wbar']:.4f} "
                  f"medESS={rec['catalogue']['reachable']['median_ESS']:.0f} "
                  f"minESS_grid={rec['grid_min_ESS_reachable']:.2f}", flush=True)
        # frontier: log-N interpolation of reachable wbar to 0.99
        ns = np.array(N_SCAN, dtype=float)
        wb = np.array([mrec["by_N"][str(n)]["catalogue"]["reachable"]["wbar"] for n in N_SCAN])
        if wb[-1] >= 0.99:
            mrec["N_at_wbar_0p99_reachable"] = (
                float(np.exp(np.interp(0.99, wb, np.log(ns)))) if wb[0] < 0.99 else float(ns[0])
            )
        else:
            mrec["N_at_wbar_0p99_reachable"] = None
        results["measures"][name] = mrec

    with open(f"{OUT}/sizing_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("done -> sizing_results.json")


if __name__ == "__main__":
    _main()
