# ruff: noqa: B023, F821
"""ADVERSARIAL ADJUDICATION of M-4 (independent verifier session, 2026-08-05).

Checks, all with code written independently of m4_shared_galaxy_census.py:
  A. Reproduce the C-4 pair criteria (production ball-radius formula, 2-sigma
     chord-sum sky overlap + 2-sigma d_L window) from
     results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv
     with a closed-form 2x2 eigenvalue (no np.linalg loop) and confirm 279 pairs /
     385 events; confirm the M-4 pair_records pair set equals this set; recount
     sharing pairs (1D/2D) from the frozeng ball emits with independent parsing.
  B. Spot-check pairs (>=5): rebuild each involved event's candidate ball FROM
     SCRATCH — full catalogue read incl. sky columns, production mass mapping,
     astropy-derived ICRS->BarycentricTrueEcliptic rotation (matrix validated
     against direct astropy on a sample), brute-force chord scan at the production
     1.5-sigma radius, production z-window (get_redshift_outer_bounds, z cap 1.5)
     and 1.5-sigma mass band — and compare ball sets + shared sets + w_pop shares
     against the frozeng JSON emits and m4_results.json pair_records.
  C. Sanity: every stored share in [0,1]; no 2D-sharing pair touches an event with
     L_cat_with_bh == 0 at h=0.73 (postfix diagnostics).
  D. Recompute every quoted summary statistic from the raw per-pair records.
  E. Independent global inverted-index census (all C(1588,2) pairs) + outside-C4
     predicate diagnosis.

Read-only w.r.t. production data. Output: m4_adjudication_results.json (facts only).
Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python results/campaign51_20260728/realistic_20260729/crossterm_instrument/m4_adjudicate.py
"""

import json
import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

T0 = time.time()
REPO = "/home/jasper/Repositories/MasterThesisCode"
sys.path.insert(0, REPO)
os.chdir(REPO)

from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
)
from master_thesis_code.physical_relations import get_redshift_outer_bounds  # noqa: E402

DIR = os.path.join(REPO, "results/campaign51_20260728/realistic_20260729/crossterm_instrument")
CRB_CSV = os.path.join(
    REPO, "results/run_20260804_postfix/joint_r1/diagnostics/prepared_cramer_rao_bounds.csv"
)
STAGED = os.path.join(REPO, "results/campaign51_20260728/realistic_20260729/realizations_staged")
FROZENG = os.path.join(REPO, "results/run_20260804_frozeng")
POSTFIX = os.path.join(REPO, "results/run_20260804_postfix")
M4_JSON = os.path.join(DIR, "m4_results.json")
OUT_JSON = os.path.join(DIR, "m4_adjudication_results.json")

VENUES = {
    "joint_r1": {
        "catalogue": os.path.join(STAGED, "observed_catalogue_seed900001.csv"),
        "json73": os.path.join(FROZENG, "joint_r1/posteriors_with_bh_mass/h_0_73.json"),
        "diag": os.path.join(POSTFIX, "joint_r1/diagnostics/event_likelihoods.csv"),
    },
    "iiib": {
        "catalogue": os.path.join(STAGED, "cluster_parent_reduced_galaxy_catalogue.csv"),
        "json73": os.path.join(FROZENG, "iiib/posteriors_with_bh_mass/h_0_73.json"),
        "diag": os.path.join(POSTFIX, "iiib/diagnostics/event_likelihoods.csv"),
    },
}

report = {"script": __file__, "checks": {}}
FAILURES = []


def check(name, ok, detail):
    report["checks"][name] = {"pass": bool(ok), "detail": detail}
    flag = "PASS" if ok else "FAIL"
    print(f"[{time.time() - T0:7.1f}s] {flag} {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def qstats(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


# ============================================================ A. C-4 pair set
crb = pd.read_csv(CRB_CSV)
N_EV = len(crb)
theta_e = crb["qS"].to_numpy()
phi_e = crb["phiS"].to_numpy()
sp2 = crb["delta_phiS_delta_phiS"].to_numpy()
st2 = crb["delta_qS_delta_qS"].to_numpy()
ctp = crb["delta_phiS_delta_qS"].to_numpy()
dl = crb["luminosity_distance"].to_numpy()
sdl = np.sqrt(crb["delta_luminosity_distance_delta_luminosity_distance"].to_numpy())
mz = crb["M"].to_numpy()
smz = np.sqrt(crb["delta_M_delta_M"].to_numpy())

# closed-form lambda_max of J Sigma J^T, J = diag(|sin th|,1)  (independent of eigvalsh)
s2 = np.sin(theta_e) ** 2
a11 = s2 * sp2
a12 = np.abs(np.sin(theta_e)) * ctp
a22 = st2
tr = a11 + a22
disc = np.sqrt(((a11 - a22) / 2.0) ** 2 + a12**2)
lam_max = tr / 2.0 + disc
lam_max = np.maximum(lam_max, 0.0)


def radius(sig_mult):
    return sig_mult * np.sqrt(lam_max)


st_e = np.sin(theta_e)
uv = np.stack([st_e * np.cos(phi_e), st_e * np.sin(phi_e), np.cos(theta_e)], axis=1)
chord = np.linalg.norm(uv[:, None, :] - uv[None, :, :], axis=2)

r2 = radius(2.0)
iu = np.triu_indices(N_EV, k=1)
sky_ok = chord[iu] <= (r2[:, None] + r2[None, :])[iu]
ii, jj = iu[0][sky_ok], iu[1][sky_ok]
lo, hi = dl - 2.0 * sdl, dl + 2.0 * sdl
dl_ok = (lo[ii] <= hi[jj]) & (lo[jj] <= hi[ii])
C4_PAIRS = set(zip(ii[dl_ok].tolist(), jj[dl_ok].tolist()))
c4_touched = set(ii[dl_ok].tolist()) | set(jj[dl_ok].tolist())

check(
    "A1_c4_pair_count",
    len(C4_PAIRS) == 279 and len(c4_touched) == 385 and int(sky_ok.sum()) == 1620,
    f"CRB rows={N_EV}; sky pairs={int(sky_ok.sum())} (expect 1620); "
    f"sky+dL pairs={len(C4_PAIRS)} (expect 279); touched events={len(c4_touched)} (expect 385)",
)
check(
    "A2_dropped_rows_touch_no_c4_pair",
    all(1203 not in p and 1356 not in p for p in C4_PAIRS),
    "CRB indices 1203/1356 absent from all 279 C-4 pairs",
)

m4 = json.load(open(M4_JSON))


def load_balls(path):
    d = json.load(open(path))
    gl, add = d["galaxy_likelihoods"], d["additional_galaxies_without_bh_mass"]
    wbh, ball, n1, n2 = {}, {}, {}, {}
    for k in gl:
        ev = int(k)
        w = frozenset(r[0] for r in gl[k])
        a = frozenset(r[0] for r in add[k])
        wbh[ev] = w
        ball[ev] = w | a
        m = {r[0]: r[1][0] for r in add[k]}
        m.update({r[0]: r[1][0] for r in gl[k]})
        n1[ev] = m
        n2[ev] = {r[0]: r[1][2] for r in gl[k]}
    return wbh, ball, n1, n2


BALLS = {}
for venue, cfg in VENUES.items():
    wbh, ball, n1, n2 = load_balls(cfg["json73"])
    BALLS[venue] = {"wbh": wbh, "ball": ball, "n1": n1, "n2": n2}
    prs = m4[venue]["pair_records"]
    m4_pairset = {(r["i"], r["j"]) for r in prs}
    n_share_1d = sum(1 for (i, j) in C4_PAIRS if ball[i] & ball[j])
    n_share_2d = sum(1 for (i, j) in C4_PAIRS if wbh[i] & wbh[j])
    exp = m4[venue]["pair_census"]
    check(
        f"A3_{venue}_pairset_and_sharing_counts",
        m4_pairset == C4_PAIRS
        and n_share_1d == exp["n_pairs_shared_ge1_1d"]
        and n_share_2d == exp["n_pairs_shared_ge1_2d"]
        and set(ball) == set(range(N_EV)) - {1203, 1356},
        f"m4 pair set == my C-4 set ({len(m4_pairset)}); my sharing counts 1D={n_share_1d} "
        f"2D={n_share_2d} vs m4 {exp['n_pairs_shared_ge1_1d']}/{exp['n_pairs_shared_ge1_2d']}; "
        f"evaluated events={len(ball)}",
    )
    # my own shared-count histogram vs stored
    my_counts = sorted(len(ball[i] & ball[j]) for (i, j) in C4_PAIRS)
    stored = sorted(r["n_shared_1d"] for r in prs)
    check(
        f"A4_{venue}_shared_counts_match_records",
        my_counts == stored,
        f"279 per-pair 1D shared counts identical; max={my_counts[-1]}",
    )

# ==================================================== E. global census (cheap, do early)
GLOBAL = {}
for venue in VENUES:
    ball, wbh = BALLS[venue]["ball"], BALLS[venue]["wbh"]
    for chan, sets in (("1d", ball), ("2d", wbh)):
        inv = defaultdict(set)
        for ev, s in sets.items():
            for g in s:
                inv[g].add(ev)
        pairs = set()
        maxdeg = 0
        for g, evs in inv.items():
            if len(evs) > 1:
                maxdeg = max(maxdeg, len(evs))
                pairs.update(combinations(sorted(evs), 2))
        GLOBAL[(venue, chan)] = {"pairs": pairs, "maxdeg": maxdeg}
gj = m4["joint_r1"]["global_sharing_check"]
gi = m4["iiib"]["global_sharing_check"]
check(
    "E1_global_census",
    len(GLOBAL[("joint_r1", "1d")]["pairs"]) == 349 == gj["n_sharing_pairs_global_1d"]
    and len(GLOBAL[("joint_r1", "2d")]["pairs"]) == 104 == gj["n_sharing_pairs_global_2d"]
    and len(GLOBAL[("iiib", "1d")]["pairs"]) == 280 == gi["n_sharing_pairs_global_1d"]
    and len(GLOBAL[("iiib", "2d")]["pairs"]) == 21 == gi["n_sharing_pairs_global_2d"]
    and GLOBAL[("joint_r1", "1d")]["maxdeg"] == 4 == gj["max_events_per_galaxy_1d"]
    and GLOBAL[("iiib", "1d")]["maxdeg"] == 4 == gi["max_events_per_galaxy_1d"],
    f"joint_r1 1d/2d = {len(GLOBAL[('joint_r1', '1d')]['pairs'])}/"
    f"{len(GLOBAL[('joint_r1', '2d')]['pairs'])} (claim 349/104); "
    f"iiib = {len(GLOBAL[('iiib', '1d')]['pairs'])}/{len(GLOBAL[('iiib', '2d')]['pairs'])} "
    f"(claim 280/21); max events/galaxy = {GLOBAL[('joint_r1', '1d')]['maxdeg']}, "
    f"{GLOBAL[('iiib', '1d')]['maxdeg']} (claim 4)",
)
# outside-C4 counts + predicate diagnosis, my own predicate evaluation
for venue, n_out_exp, n_in_exp in (("joint_r1", 269, 80), ("iiib", 217, 63)):
    pairs = GLOBAL[(venue, "1d")]["pairs"]
    inside = pairs & C4_PAIRS
    outside = pairs - C4_PAIRS
    sky_pass = sum(1 for (i, j) in outside if chord[i, j] <= r2[i] + r2[j])
    dl_pass = sum(1 for (i, j) in outside if abs(dl[i] - dl[j]) <= 2.0 * (sdl[i] + sdl[j]))
    check(
        f"E2_{venue}_outside_c4",
        len(inside) == n_in_exp
        and len(outside) == n_out_exp
        and sky_pass == len(outside)
        and dl_pass == 0,
        f"in-C4={len(inside)} (claim {n_in_exp}); outside={len(outside)} (claim {n_out_exp}); "
        f"outside sky-predicate pass={sky_pass}/{len(outside)} (claim all); "
        f"outside dL-predicate pass={dl_pass} (claim 0)",
    )

# ============================================ C. sanity: shares in [0,1]; 2D zeroing
for venue, cfg in VENUES.items():
    prs = m4[venue]["pair_records"]
    all_shares = []
    for r in prs:
        for k, v in r.items():
            if ("share" in k and "Npos" not in k) and isinstance(v, float):
                all_shares.append(v)
    out_recs = m4[venue]["global_pairs_shares"]["outside_c4_records"]
    for r in out_recs:
        for k in ("w_share_1d_i", "w_share_1d_j", "wN_share_1d_i", "wN_share_1d_j"):
            v = r.get(k)
            if isinstance(v, float):
                all_shares.append(v)
    for r in m4[venue]["per_event_union"]["records"]:
        for k in ("w_share_union_1d", "wN_share_union_1d"):
            v = r.get(k)
            if isinstance(v, float):
                all_shares.append(v)
    arr = np.array(all_shares)
    check(
        f"C1_{venue}_shares_in_unit_interval",
        bool((arr >= 0.0).all() and (arr <= 1.0 + 1e-12).all()),
        f"{arr.size} stored share values, min={arr.min():.3e}, max={arr.max():.10f}",
    )
    diag = pd.read_csv(cfg["diag"])
    d73 = diag[np.isclose(diag["h"], 0.73)]
    zero2d = set(d73.loc[d73["L_cat_with_bh"] == 0, "event_idx"].astype(int))
    wbh = BALLS[venue]["wbh"]
    emptywbh = {ev for ev, s in wbh.items() if len(s) == 0}
    bad_pairs = [
        (r["i"], r["j"])
        for r in prs
        if r["n_shared_2d"] > 0 and (r["i"] in zero2d or r["j"] in zero2d)
    ]
    check(
        f"C2_{venue}_mass_conditioning",
        emptywbh == zero2d and not bad_pairs,
        f"empty 2D balls == L_cat_with_bh==0 set ({len(emptywbh)} events, claim "
        f"{981 if venue == 'joint_r1' else 1294}); 2D-sharing pairs touching a zero-2D "
        f"event: {len(bad_pairs)} (must be 0)",
    )

# ================================= D. summary stats recomputed from raw records
prs = m4["joint_r1"]["pair_records"]
w_sides = [
    r[k] for r in prs for k in ("w_share_1d_i", "w_share_1d_j") if isinstance(r.get(k), float)
]
wn_sides = [
    r[k] for r in prs for k in ("wN_share_1d_i", "wN_share_1d_j") if isinstance(r.get(k), float)
]
s = qstats(w_sides)
sn = qstats(wn_sides)
check(
    "D1_joint_r1_1d_share_summary",
    s["n"] == 160
    and abs(s["min"] - 2.9e-3) < 2e-4
    and abs(s["median"] - 0.155) < 5e-3
    and abs(s["mean"] - 0.284) < 5e-3
    and abs(s["p90"] - 0.783) < 5e-3
    and abs(s["max"] - 1.0) < 1e-9
    and abs(sn["median"] - 0.192) < 5e-3
    and abs(sn["p90"] - 0.990) < 5e-3
    and abs(sn["max"] - 1.0) < 1e-9,
    f"160 sides: w min/med/mean/p90/max = {s['min']:.2e}/{s['median']:.3f}/"
    f"{s['mean']:.3f}/{s['p90']:.3f}/{s['max']:.3f} (claim 2.9e-3/0.155/0.284/0.783/1.0); "
    f"wN med/p90/max = {sn['median']:.3f}/{sn['p90']:.3f}/{sn['max']:.3f} "
    f"(claim 0.192/0.990/1.0)",
)
hist = {
    "0": sum(1 for r in prs if r["n_shared_1d"] == 0),
    "1": sum(1 for r in prs if r["n_shared_1d"] == 1),
    "2-10": sum(1 for r in prs if 2 <= r["n_shared_1d"] <= 10),
    "11-100": sum(1 for r in prs if 11 <= r["n_shared_1d"] <= 100),
    "101-1000": sum(1 for r in prs if 101 <= r["n_shared_1d"] <= 1000),
    ">1000": sum(1 for r in prs if r["n_shared_1d"] > 1000),
}
check(
    "D2_joint_r1_hist",
    hist == {"0": 199, "1": 7, "2-10": 21, "11-100": 33, "101-1000": 16, ">1000": 3}
    and max(r["n_shared_1d"] for r in prs) == 8861,
    f"hist={hist}, max={max(r['n_shared_1d'] for r in prs)} (claim {{0:199,1:7,2-10:21,"
    f"11-100:33,101-1000:16,>1000:3}}, 8861)",
)
w2 = [r[k] for r in prs for k in ("w_share_2d_i", "w_share_2d_j") if isinstance(r.get(k), float)]
s2d = qstats(w2)
check(
    "D3_joint_r1_2d_shares",
    s2d["n"] == 54
    and abs(s2d["median"] - 0.098) < 5e-3
    and abs(s2d["p90"] - 0.620) < 5e-3
    and abs(s2d["max"] - 1.0) < 1e-9,
    f"2D sides n={s2d['n']} (claim 54), med={s2d['median']:.3f} (0.098), "
    f"p90={s2d['p90']:.3f} (0.620), max={s2d['max']:.3f} (1.0)",
)
mins_w, geo_w, mins_wn, cnt_frac = [], [], [], []
for r in prs:
    a, b = r.get("w_share_1d_i"), r.get("w_share_1d_j")
    if isinstance(a, float) and isinstance(b, float):
        mins_w.append(min(a, b))
        geo_w.append(float(np.sqrt(a * b)))
    a, b = r.get("wN_share_1d_i"), r.get("wN_share_1d_j")
    if isinstance(a, float) and isinstance(b, float):
        mins_wn.append(min(a, b))
    if r["n_shared_1d"] > 0:
        cnt_frac.append(r["n_shared_1d"] / min(r["n_ball_1d_i"], r["n_ball_1d_j"]))
sm, sg, smn, sc = qstats(mins_w), qstats(geo_w), qstats(mins_wn), qstats(cnt_frac)
check(
    "D4_joint_r1_suppression",
    abs(sm["median"] - 0.087) < 5e-3
    and abs(sm["p90"] - 0.296) < 5e-3
    and abs(sm["max"] - 1.0) < 1e-9
    and abs(sg["median"] - 0.179) < 5e-3
    and abs(sg["p90"] - 0.464) < 5e-3
    and abs(smn["median"] - 0.031) < 5e-3
    and abs(smn["p90"] - 0.518) < 5e-3
    and abs(sc["median"] - 0.40) < 5e-3
    and abs(sc["p90"] - 1.0) < 1e-9,
    f"min-w med/p90/max={sm['median']:.3f}/{sm['p90']:.3f}/{sm['max']:.2f} "
    f"(claim 0.087/0.296/1.0); geo med/p90={sg['median']:.3f}/{sg['p90']:.3f} "
    f"(0.179/0.464); min-wN med/p90={smn['median']:.3f}/{smn['p90']:.3f} (0.031/0.518); "
    f"count-frac med/p90={sc['median']:.3f}/{sc['p90']:.3f} (0.40/1.0)",
)
ev_u = m4["joint_r1"]["per_event_union"]["records"]
u = qstats([r["w_share_union_1d"] for r in ev_u if isinstance(r["w_share_union_1d"], float)])
check(
    "D5_joint_r1_event_union",
    len(ev_u) == 143
    and abs(u["median"] - 0.206) < 5e-3
    and abs(u["p90"] - 0.806) < 5e-3
    and abs(u["max"] - 1.0) < 1e-9,
    f"events={len(ev_u)} (claim 143); union w share med/p90/max = "
    f"{u['median']:.3f}/{u['p90']:.3f}/{u['max']:.2f} (claim 0.206/0.806/1.0)",
)
# iiib summaries
prs_i = m4["iiib"]["pair_records"]
wi = [r[k] for r in prs_i for k in ("w_share_1d_i", "w_share_1d_j") if isinstance(r.get(k), float)]
si = qstats(wi)
hist_i = {
    "0": sum(1 for r in prs_i if r["n_shared_1d"] == 0),
    "1": sum(1 for r in prs_i if r["n_shared_1d"] == 1),
    "2-10": sum(1 for r in prs_i if 2 <= r["n_shared_1d"] <= 10),
    "11-100": sum(1 for r in prs_i if 11 <= r["n_shared_1d"] <= 100),
    "101-1000": sum(1 for r in prs_i if 101 <= r["n_shared_1d"] <= 1000),
    ">1000": sum(1 for r in prs_i if r["n_shared_1d"] > 1000),
}
ev_ui = m4["iiib"]["per_event_union"]["records"]
ui = qstats([r["w_share_union_1d"] for r in ev_ui if isinstance(r["w_share_union_1d"], float)])
cnt_frac_i = [
    r["n_shared_1d"] / min(r["n_ball_1d_i"], r["n_ball_1d_j"])
    for r in prs_i
    if r["n_shared_1d"] > 0
]
sci = qstats(cnt_frac_i)
check(
    "D6_iiib_summaries",
    si["n"] == 126
    and abs(si["median"] - 0.124) < 5e-3
    and abs(si["p90"] - 0.881) < 5e-3
    and abs(si["max"] - 1.0) < 1e-9
    and hist_i == {"0": 216, "1": 8, "2-10": 9, "11-100": 28, "101-1000": 14, ">1000": 4}
    and max(r["n_shared_1d"] for r in prs_i) == 9898
    and len(ev_ui) == 115
    and abs(ui["median"] - 0.150) < 5e-3
    and abs(ui["p90"] - 0.924) < 5e-3
    and abs(sci["median"] - 0.40) < 1e-1,
    f"126 sides med/p90/max={si['median']:.3f}/{si['p90']:.3f}/{si['max']:.2f} "
    f"(claim 0.124/0.881/1.0); hist={hist_i}; max={max(r['n_shared_1d'] for r in prs_i)} "
    f"(claim 9898); events={len(ev_ui)} (claim 115) union med/p90="
    f"{ui['median']:.3f}/{ui['p90']:.3f} (0.150/0.924); count-frac med={sci['median']:.3f}",
)
# outside vs inside min-side share medians (joint_r1)
outr = m4["joint_r1"]["global_pairs_shares"]["outside_c4_records"]
mins_out = [
    min(r["w_share_1d_i"], r["w_share_1d_j"])
    for r in outr
    if isinstance(r.get("w_share_1d_i"), float) and isinstance(r.get("w_share_1d_j"), float)
]
so = qstats(mins_out)
check(
    "D7_joint_r1_outside_weaker",
    abs(so["median"] - 0.011) < 3e-3 and abs(sm["median"] - 0.087) < 5e-3,
    f"outside-pair min-side w share median={so['median']:.4f} (claim 0.011) vs "
    f"in-C4 {sm['median']:.3f} (claim 0.087), n_outside_with_both_sides={so['n']}",
)

# =============================================== B. from-scratch ball spot-checks
from astropy import units as u_ap  # noqa: E402
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord  # noqa: E402


def icrs_to_ecl_matrix():
    basis = np.eye(3)
    ra = np.degrees(np.arctan2(basis[:, 1], basis[:, 0]))
    dec = np.degrees(np.arcsin(basis[:, 2]))
    c = SkyCoord(ra=ra * u_ap.deg, dec=dec * u_ap.deg, frame="icrs")
    e = c.transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    lon, lat = np.radians(e.lon.deg), np.radians(e.lat.deg)
    out = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    return out.T  # columns are images of basis vectors


ROT = icrs_to_ecl_matrix()


def select_spot_pairs(venue):
    prs_v = m4[venue]["pair_records"]
    sharing = [r for r in prs_v if r["n_shared_1d"] > 0]
    picks = {}
    picks["max_1d"] = max(sharing, key=lambda r: r["n_shared_1d"])
    picks["min_1d"] = min(sharing, key=lambda r: r["n_shared_1d"])
    sharing_sorted = sorted(sharing, key=lambda r: r["n_shared_1d"])
    picks["median_1d"] = sharing_sorted[len(sharing_sorted) // 2]
    with2d = [r for r in prs_v if r["n_shared_2d"] > 0]
    if with2d:
        picks["max_2d"] = max(with2d, key=lambda r: r["n_shared_2d"])
    withsub = [
        r
        for r in sharing
        if isinstance(r.get("w_share_1d_i"), float)
        and isinstance(r.get("w_share_1d_j"), float)
        and max(r["w_share_1d_i"], r["w_share_1d_j"]) > 0.999
    ]
    if withsub:
        picks["subset_like"] = withsub[0]
    return picks


SPOT = {"joint_r1": select_spot_pairs("joint_r1"), "iiib": select_spot_pairs("iiib")}
# one outside-C4 pair per venue (largest shared count)
for venue in VENUES:
    outr_v = m4[venue]["global_pairs_shares"]["outside_c4_records"]
    SPOT[venue]["outside_c4"] = max(outr_v, key=lambda r: r["n_shared_1d"])

spot_report = {}
for venue, cfg in VENUES.items():
    tv = time.time()
    # file order (headerless): RA, DEC, B_MAG, Z, SZ, MS, MSE, ZFLAG
    cat = pd.read_csv(
        cfg["catalogue"],
        header=None,
        names=["RA", "DEC", "BMAG", "Z", "SZ", "MS", "MSE", "ZFLAG"],
        usecols=["RA", "DEC", "Z", "SZ", "MS", "MSE"],
    )
    ra = np.radians(cat["RA"].to_numpy(np.float64))
    dec = np.radians(cat["DEC"].to_numpy(np.float64))
    z = cat["Z"].to_numpy(np.float64)
    sz = cat["SZ"].to_numpy(np.float64)
    ms = cat["MS"].to_numpy(np.float64)
    mse = cat["MSE"].to_numpy(np.float64)
    del cat
    mbh, mbh_err = _empiric_stellar_mass_to_BH_mass_relation(ms, mse)
    del ms, mse
    keep = ~np.isnan(mbh)
    ra, dec, z, sz, mbh, mbh_err = (
        ra[keep],
        dec[keep],
        z[keep],
        sz[keep],
        mbh[keep],
        mbh_err[keep],
    )
    pm = _mass_redshift_prune_mask(
        pd.Series(mbh), pd.Series(mbh_err), pd.Series(z), pd.Series(sz), 1e4, 1e7, 1.5
    ).to_numpy()
    ra, dec, z, sz, mbh, mbh_err = ra[pm], dec[pm], z[pm], sz[pm], mbh[pm], mbh_err[pm]
    npr = len(z)
    # ICRS unit vectors -> ecliptic via rotation matrix
    g_icrs = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)
    g_ecl = g_icrs @ ROT.T
    del g_icrs
    # validate rotation on a sample against direct astropy (production path)
    samp = np.random.default_rng(7).choice(npr, size=20000, replace=False)
    c = SkyCoord(
        ra=np.degrees(ra[samp]) * u_ap.deg, dec=np.degrees(dec[samp]) * u_ap.deg, frame="icrs"
    )
    e = c.transform_to(BarycentricTrueEcliptic(equinox="J2000"))
    lon = np.radians(e.lon.deg % 360.0)
    lat = np.radians(e.lat.deg)
    v_ast = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)
    rot_dev = float(np.abs(v_ast - g_ecl[samp]).max())
    check(
        f"B0_{venue}_rotation_matrix_valid",
        rot_dev < 1e-9,
        f"max |astropy - matrix| component dev over 20k sample = {rot_dev:.2e}; "
        f"pruned rows = {npr} (m4 claims "
        f"{m4[venue]['n_catalogue_pruned_rows']})",
    )

    def w_of(idx):
        idx = np.asarray(sorted(idx), dtype=np.int64)
        return idx, R_eff_per_mbh(mbh[idx]) / (1.0 + z[idx])

    def rebuild_ball(ev):
        r15 = 1.5 * np.sqrt(lam_max[ev])
        d2 = np.einsum("ij,j->i", g_ecl, uv[ev])
        chord2 = np.sqrt(np.maximum(2.0 - 2.0 * d2, 0.0))
        sky = chord2 <= r15
        z_min, z_max = get_redshift_outer_bounds(distance=dl[ev], distance_error=sdl[ev])
        z_max = min(z_max, 1.5)
        zok = (z_min <= z + sz) & (z_max >= z - sz)
        ball1 = sky & zok
        lo_b = (mz[ev] - 1.5 * smz[ev]) / (1.0 + z_max)
        hi_b = (mz[ev] + 1.5 * smz[ev]) / (1.0 + z_min)
        mok = (lo_b <= mbh + mbh_err) & (mbh - mbh_err <= hi_b)
        ball2 = ball1 & mok
        return set(np.flatnonzero(ball1).tolist()), set(np.flatnonzero(ball2).tolist())

    ball_j, wbh_j = BALLS[venue]["ball"], BALLS[venue]["wbh"]
    vrep = {}
    events = sorted({r[k] for r in SPOT[venue].values() for k in ("i", "j")})
    rebuilt = {}
    for ev in events:
        b1, b2 = rebuild_ball(ev)
        rebuilt[ev] = (b1, b2)
        vrep[f"event_{ev}"] = {
            "rebuilt_1d": len(b1),
            "json_1d": len(ball_j[ev]),
            "match_1d": b1 == set(ball_j[ev]),
            "rebuilt_2d": len(b2),
            "json_2d": len(wbh_j[ev]),
            "match_2d": b2 == set(wbh_j[ev]),
        }
    all_match = all(v["match_1d"] and v["match_2d"] for v in vrep.values())
    check(
        f"B1_{venue}_ball_rebuild",
        all_match,
        f"{len(events)} events rebuilt from scratch; all 1D+2D ball sets bit-identical "
        f"to frozeng JSON: {all_match}; detail: "
        + "; ".join(
            f"ev{ev}: 1D {vrep[f'event_{ev}']['rebuilt_1d']}=="
            f"{vrep[f'event_{ev}']['json_1d']} 2D {vrep[f'event_{ev}']['rebuilt_2d']}=="
            f"{vrep[f'event_{ev}']['json_2d']}"
            for ev in events
        ),
    )
    pair_ok = True
    pair_detail = []
    for label, r in SPOT[venue].items():
        i, j = r["i"], r["j"]
        s1 = rebuilt[i][0] & rebuilt[j][0]
        s2 = rebuilt[i][1] & rebuilt[j][1]
        exp1 = r["n_shared_1d"]
        exp2 = r.get("n_shared_2d", 0)
        ok = len(s1) == exp1 and len(s2) == exp2
        # also verify shared sets equal JSON-derived intersection
        ok = ok and s1 == set(ball_j[i]) & set(ball_j[j]) and s2 == set(wbh_j[i]) & set(wbh_j[j])
        # recompute w share for side i and j
        sh = {}
        if s1:
            idx_s, w_s = w_of(s1)
            for side, ev in (("i", i), ("j", j)):
                idx_b, w_b = w_of(rebuilt[ev][0])
                sh[side] = float(w_s.sum() / w_b.sum())
                stored_v = r.get(f"w_share_1d_{side}")
                if isinstance(stored_v, float):
                    ok = ok and abs(sh[side] - stored_v) < 1e-9
        pair_ok = pair_ok and ok
        pair_detail.append(
            f"{label} ({i},{j}): shared1d={len(s1)} (exp {exp1}), shared2d={len(s2)} "
            f"(exp {exp2}), w_shares={ {k: round(v, 6) for k, v in sh.items()} }, ok={ok}"
        )
    check(f"B2_{venue}_pair_spotchecks", pair_ok, " | ".join(pair_detail))
    spot_report[venue] = {
        "events": vrep,
        "pairs": pair_detail,
        "runtime_s": round(time.time() - tv, 1),
    }
    del g_ecl, ra, dec, z, sz, mbh, mbh_err

report["spot_checks"] = spot_report
report["n_failures"] = len(FAILURES)
report["failures"] = FAILURES
report["runtime_s"] = round(time.time() - T0, 1)
with open(OUT_JSON, "w") as f:
    json.dump(report, f, indent=1, default=str)
print(f"[{time.time() - T0:7.1f}s] wrote {OUT_JSON}; failures: {FAILURES if FAILURES else 'NONE'}")
