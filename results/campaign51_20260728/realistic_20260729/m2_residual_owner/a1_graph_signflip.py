"""A1: the pre-registered H-e (chance/multiplicity) decisive test.

Verbatim source of the test (CLAIM_M2_RESIDUAL_OWNER_20260807.md, H-e "Refute by (FREE)"):

    (i) sign-flip permutation flipping connected components of the C-4 overlap graph as units
    (components computable from prepared_cramer_rao_bounds.csv alone) -- if the 2D residual
    survives component-level flips at both venues, the intra-stratum-correlation chance route
    is refuted; (ii) jackknife-over-components and re-matching under different RNG seeds for
    stability; (iii) structural evidence -- D-1 localizing the residual coherently to one
    component at both venues is itself strong evidence against chance.

And the table row (claim file Sec.5): "H-e predicts p >= 0.0455 at one or both venues; every
mechanism hypothesis predicts survival at both. Also: jackknife-over-components shows the mean
driven by <= a few components under H-e."

And STAGE1_READOUT Sec.5(A)(1): "component-level sign-flip over the C-4 overlap-graph connected
components at both venues, plus jackknife-over-components and re-matching seed stability.
Survival at both venues closes H-e per the prereg; failure at either venue reopens the chance
account and the chronicle wording changes materially."

PASS/FAIL branch (as pre-stated, executed exactly, not improvised):
  - decisive statistic: component-level sign-flip p on the matched 2D combined-channel residual
    (identical 385 pairs / identical matching as M-2 / D-1), clusters = connected components of
    the C-4 overlap graph (edges = the 279 sky+2sigma-d_L pairs that define "overlap" in the C-4
    census; graph built once over all 1590 CRB rows, restricted per venue to the used pairs).
  - alpha = 0.0455 (the M-2 pre-stated 2-sigma two-sided criterion; this thread inherits it,
    since no different alpha is stated anywhere for A1).
  - PASS (H-e refuted) iff component-signflip p < alpha AT BOTH VENUES ("survival").
  - FAIL (H-e reopens) iff component-signflip p >= alpha AT EITHER VENUE.

Ambiguity resolved conservatively (disclosed): "re-matching under different RNG seeds for
stability" is read as re-running the component-level sign-flip PERMUTATION TEST (the only RNG
consumer in this pipeline) under several seeds and checking the pass/fail side of alpha does not
change -- not as re-deriving the 1-NN match, which is a deterministic argmin with no RNG
dependence (verified: no ties). This is the conservative reading because it adds a robustness
check beyond the single-seed decisive number rather than substituting for it; the primary,
decisive p-value quoted below uses SEED_SIGNFLIP = 20260805 (M-2's own signflip seed), matching
the machinery precedent already used for the M-2/D-1 primary test.

Reuses the EXACT M-2/D-1 machinery: same C-4 census asserts (1620/279/385), same 1-NN-with-
replacement matching on standardized (log10 ball-radius chord, SNR), same 385 pairs, same
combined_with_bh (2D) chord definition ln L(h=0.60) - ln L(h=0.73). M-2 headline totals are
asserted to reproduce exactly (bitwise sanity check against m2_results.json / d1_results.json).

FREE READ: existing CSVs only (event_likelihoods.csv, prepared_cramer_rao_bounds.csv);
no production runs, no cluster. Output: a1_results.json.
"""

import json

import numpy as np
import pandas as pd

N_PERM = 20000
ALPHA = 0.0455
SEED_SIGNFLIP = 20260805  # primary, decisive -- matches M-2's own signflip seed
STABILITY_SEEDS = [20260805, 99, 424242]  # primary + M-2 cluster-robust seed + verifier's seed
ROOT = "/home/jasper/Repositories/MasterThesisCode"
OUTDIR = f"{ROOT}/results/campaign51_20260728/realistic_20260729/m2_residual_owner"
OUT = f"{OUTDIR}/a1_results.json"
CRB = f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
VENUES = {
    "iiib": f"{ROOT}/results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv",
    "joint_r1": f"{ROOT}/results/run_20260804_postfix/joint_r1/diagnostics/event_likelihoods.csv",
}
# M-2 committed matched 2D totals (m2_results.json / d1_results.json) -- asserted below.
M2_TOTALS_2D = {
    "iiib": 0.022252643015992925,
    "joint_r1": 0.020697491999731973,
}

# ---------------------------------------------------------------- C-4 census (verbatim M-2/D-1)
df = pd.read_csv(CRB)
n = len(df)
assert n == 1590, n
theta = df["qS"].to_numpy()
phi = df["phiS"].to_numpy()
s_phi2 = df["delta_phiS_delta_phiS"].to_numpy()
s_theta2 = df["delta_qS_delta_qS"].to_numpy()
cov = df["delta_phiS_delta_qS"].to_numpy()
dl = df["luminosity_distance"].to_numpy()
s_dl = np.sqrt(df["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
snr = df["SNR"].to_numpy()

r = np.empty(n)
for i in range(n):
    sig = np.array([[s_phi2[i], cov[i]], [cov[i], s_theta2[i]]])
    jac = np.diag([abs(np.sin(theta[i])), 1.0])
    lam = float(np.linalg.eigvalsh(jac @ sig @ jac.T).max())
    r[i] = 2.0 * np.sqrt(max(lam, 0.0))

st = np.sin(theta)
xyz = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
iu = np.triu_indices(n, k=1)
sky = d[iu] <= (r[:, None] + r[None, :])[iu]
ii, jj = iu[0][sky], iu[1][sky]
lo, hi = dl - 2 * s_dl, dl + 2 * s_dl
win = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
overlap = np.zeros(n, dtype=bool)
overlap[ii[win]] = True
overlap[jj[win]] = True
assert int(win.sum()) == 279
assert int(overlap.sum()) == 385
assert int(sky.sum()) == 1620

# ---------------------------------------------------------------- C-4 overlap graph: union-find
# Graph: nodes = all 1590 CRB rows; edges = the 279 sky+2sigma-d_L pairs that DEFINE C-4 overlap
# (i.e. the same predicate that sets `overlap[i]=True`). Computable from prepared_cramer_rao_bounds.csv
# alone, per the claim file's own parenthetical.
edge_i, edge_j = ii[win], jj[win]

_parent = np.arange(n)


def _find(x: int) -> int:
    while _parent[x] != x:
        _parent[x] = _parent[_parent[x]]
        x = _parent[x]
    return x


def _union(a: int, b: int) -> None:
    ra, rb = _find(a), _find(b)
    if ra != rb:
        _parent[ra] = rb


for a_i, b_i in zip(edge_i.tolist(), edge_j.tolist(), strict=True):
    _union(int(a_i), int(b_i))

component_of = np.array([_find(i) for i in range(n)])
# sanity: isolated (non-overlap) nodes are singleton components of themselves; not used below.
touched_components = np.unique(component_of[overlap])
comp_sizes_all = {
    int(c): int((component_of[overlap] == c).sum()) for c in touched_components
}
assert sum(comp_sizes_all.values()) == 385

# ---------------------------------------------------------------- helpers


def smd(a: np.ndarray, b: np.ndarray) -> float:
    sp = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def component_signflip_p(diffs: np.ndarray, comp_ids: np.ndarray, seed: int) -> float:
    """Two-sided sign-flip p flipping all pairs in the same C-4 overlap-graph component together."""
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    uniq, inv = np.unique(comp_ids, return_inverse=True)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(uniq)))
    stats = np.abs((signs[:, inv] * diffs[None, :]).mean(axis=1))
    return float((int((stats >= obs).sum()) + 1) / (N_PERM + 1))


def signflip_p_iid(diffs: np.ndarray, seed: int) -> float:
    """Baseline (non-clustered) two-sided sign-flip p, for comparison only."""
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    m = len(diffs)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, m))
    stats = np.abs((signs * diffs[None, :]).mean(axis=1))
    return float((int((stats >= obs).sum()) + 1) / (N_PERM + 1))


# ---------------------------------------------------------------- per-venue read
results: dict = {
    "read": "A1 -- H-e decisive test: component-level sign-flip over C-4 overlap-graph connected "
    "components, both venues, on the matched 2D combined-channel residual",
    "criterion_source": {
        "intake": "CLAIM_M2_RESIDUAL_OWNER_20260807.md H-e 'Refute by (FREE)' (i)-(iii)",
        "readout": "STAGE1_READOUT_20260807.md Sec.5(A)(1) + Sec.2 H-e row + Sec.4 item 3",
    },
    "alpha": ALPHA,
    "pass_rule": "PASS (H-e refuted) iff component-signflip p < alpha at BOTH venues; "
    "FAIL (H-e reopens) iff p >= alpha at EITHER venue",
    "ambiguity_resolution": (
        "'re-matching under different RNG seeds for stability' read as re-running the "
        "component-level sign-flip PERMUTATION under multiple seeds (STABILITY_SEEDS), "
        "since the 1-NN matched-pair assignment is a deterministic argmin with no RNG "
        "dependence (no ties observed). Primary/decisive p uses SEED_SIGNFLIP=20260805, "
        "matching the M-2/D-1 primary-test seed convention."
    ),
    "graph": {
        "nodes": n,
        "edges_used": int(len(edge_i)),
        "edge_definition": "the 279 sky+2sigma-d_L pairs that define C-4 'overlap' "
        "(win mask on recon_c4_census.py predicate)",
        "n_components_touching_overlap_events_full_census": int(len(touched_components)),
        "component_size_distribution_full_census_385": sorted(
            comp_sizes_all.values(), reverse=True
        ),
    },
    "stability_seeds": STABILITY_SEEDS,
    "venues": {},
}

log10_r = np.log10(r)
cov_mat = np.stack([log10_r, snr], axis=1)

overall_pass = True

for venue, path in VENUES.items():
    el = pd.read_csv(path)
    assert len(el) == 65108, (venue, len(el))
    a = el[np.isclose(el.h, 0.60)].set_index("event_idx").sort_index()
    b = el[np.isclose(el.h, 0.73)].set_index("event_idx").sort_index()
    ev = np.array(sorted(set(a.index) & set(b.index)))
    assert len(ev) == 1588, (venue, len(ev))
    a = a.loc[ev]
    b = b.loc[ev]

    ov_mask = overlap[ev]
    ct_mask = ~ov_mask
    ov_ev, ct_ev = ev[ov_mask], ev[ct_mask]
    assert len(ov_ev) == 385 and len(ct_ev) == 1203, venue

    z = (cov_mat[ev] - cov_mat[ev].mean(axis=0)) / cov_mat[ev].std(axis=0, ddof=1)
    z_ov, z_ct = z[ov_mask], z[ct_mask]
    dist2 = ((z_ov[:, None, :] - z_ct[None, :, :]) ** 2).sum(axis=2)
    nn = dist2.argmin(axis=1)
    matched_ct_ev = ct_ev[nn]

    # -- combined 2D chord (identical statistic to M-2 headline / D-1 total_chord)
    col = "combined_with_bh"
    la, lb = a[col], b[col]
    assert (la.loc[ev] > 0).all() and (lb.loc[ev] > 0).all(), venue
    chord = pd.Series(np.log(la.loc[ev].to_numpy()) - np.log(lb.loc[ev].to_numpy()), index=ev)
    x_ov = chord.loc[ov_ev].to_numpy()
    x_mct = chord.loc[matched_ct_ev].to_numpy()
    diffs = x_ov - x_mct
    m_diff = float(diffs.mean())

    # bitwise sanity check against the committed M-2/D-1 headline
    assert abs(m_diff - M2_TOTALS_2D[venue]) < 1e-9, (venue, m_diff, M2_TOTALS_2D[venue])

    # -- component ids for THIS venue's 385 overlap events
    comp_ids = component_of[ov_ev]
    uniq_comp, comp_counts = np.unique(comp_ids, return_counts=True)
    n_components_used = int(len(uniq_comp))
    giant_size = int(comp_counts.max())

    # -- decisive test: component-level sign-flip, primary seed
    p_component_primary = component_signflip_p(diffs, comp_ids, SEED_SIGNFLIP)
    # -- comparison: naive (unclustered) sign-flip, same seed/diffs (== M-2's own test up to seed)
    p_iid_primary = signflip_p_iid(diffs, SEED_SIGNFLIP)

    # -- stability across seeds
    p_component_by_seed = {
        str(s): component_signflip_p(diffs, comp_ids, s) for s in STABILITY_SEEDS
    }
    stable_side_of_alpha = len({p < ALPHA for p in p_component_by_seed.values()}) == 1

    venue_pass = p_component_primary < ALPHA
    overall_pass = overall_pass and venue_pass

    # -- jackknife-over-components: leave-one-component-out matched mean
    loco = []
    for c in uniq_comp:
        keep = comp_ids != c
        if keep.sum() == 0:
            continue
        loco_mean = float(diffs[keep].mean())
        loco.append(
            {
                "component_id": int(c),
                "component_size": int((comp_ids == c).sum()),
                "loco_mean_paired_diff": loco_mean,
                "delta_from_full_mean": loco_mean - m_diff,
                "sign_flip_on_removal": bool(np.sign(loco_mean) != np.sign(m_diff)),
            }
        )
    loco.sort(key=lambda r: abs(r["delta_from_full_mean"]), reverse=True)
    max_loco_delta = loco[0]["delta_from_full_mean"] if loco else 0.0
    max_loco_delta_frac_of_mean = (
        abs(max_loco_delta / m_diff) if m_diff != 0 else float("nan")
    )
    any_loco_sign_flip = any(r["sign_flip_on_removal"] for r in loco)
    # "driven by <= a few components" proxy: fraction of |sum of diffs| contributed by the
    # single largest-|contribution| component (component total diff / total sum of diffs).
    comp_totals = {
        int(c): float(diffs[comp_ids == c].sum()) for c in uniq_comp
    }
    total_sum = float(diffs.sum())
    top_component_share = (
        max(comp_totals.values(), key=abs) / total_sum if total_sum != 0 else float("nan")
    )

    venue_out = {
        "n_overlap_pairs": 385,
        "n_components_used": n_components_used,
        "giant_component_size": giant_size,
        "giant_component_fraction_of_385": giant_size / 385.0,
        "component_size_distribution": sorted(comp_counts.tolist(), reverse=True),
        "matched_2d_residual": {
            "mean_paired_diff": m_diff,
            "n_pairs": 385,
        },
        "tests": {
            "component_signflip_p_primary_seed": p_component_primary,
            "iid_signflip_p_same_seed_for_comparison": p_iid_primary,
            "component_signflip_p_by_seed": p_component_by_seed,
            "stable_same_side_of_alpha_across_seeds": stable_side_of_alpha,
        },
        "jackknife_over_components": {
            "n_components_tested": len(loco),
            "max_abs_delta_from_full_mean": max_loco_delta,
            "max_abs_delta_as_fraction_of_full_mean": max_loco_delta_frac_of_mean,
            "any_single_component_removal_flips_sign": any_loco_sign_flip,
            "top_component_share_of_total_signed_sum": top_component_share,
            "top5_by_abs_delta": loco[:5],
        },
        "venue_call": {
            "alpha": ALPHA,
            "p_used_for_call": p_component_primary,
            "verdict": "PASS (survives -- H-e refuted at this venue)"
            if venue_pass
            else "FAIL (does not survive -- H-e reopens at this venue)",
        },
    }
    results["venues"][venue] = venue_out

results["overall_verdict"] = {
    "rule": "PASS iff component-signflip p < alpha at BOTH venues",
    "iiib_pass": bool(results["venues"]["iiib"]["venue_call"]["verdict"].startswith("PASS")),
    "joint_r1_pass": bool(
        results["venues"]["joint_r1"]["venue_call"]["verdict"].startswith("PASS")
    ),
    "overall_pass": bool(overall_pass),
    "branch": "H-E-STAYS-CLOSED (residual survives component-graph correlation structure; "
    "H-e disfavored/refuted per prereg; thread stands on the stage-1 confounding "
    "verdict)"
    if overall_pass
    else "H-E-REOPENED (residual does not survive at >=1 venue under overlap-graph "
    "component-clustered inference; the chance/effective-N account becomes the "
    "surviving account per the pre-stated branch; chronicle wording must change)",
}

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results["overall_verdict"], indent=2))
for v in VENUES:
    print(v, results["venues"][v]["venue_call"])
