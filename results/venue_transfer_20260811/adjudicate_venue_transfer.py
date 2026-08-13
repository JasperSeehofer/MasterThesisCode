"""ADVERSARIAL ADJUDICATION of the venue-transfer readout.

Fully independent re-derivation. Imports nothing from the campaign scorer,
the collector, or the instrument package. Every statistic is recomputed from
the rawest per-seed data in the 49 chunk JSONs (the 41-point ln_post_* vectors
wherever the statistic derives from a posterior), with hand-written trapezoid
normalisation, PIT, HPD-interval, grid-argmax / parabolic-refined argmax, KS,
and binomial / Jeffreys band arithmetic.

Read-only on every input. Writes only its own JSON next to itself.
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

BASE_SEED = 20260808

# prereg §5 cell matrix, transcribed by hand from the registered file
REG_CELLS = {
    "T-0": {"h": 0.730, "off0": 40000, "n": 200, "balls": "real_k", "sigma": "zero"},
    "T-a": {"h": 0.730, "off0": 41000, "n": 200, "balls": "poisson4", "sigma": "flat"},
    "T-b": {"h": 0.730, "off0": 42000, "n": 200, "balls": "real_k", "sigma": "flat"},
    "T-c(0.690)": {"h": 0.690, "off0": 43000, "n": 200, "balls": "real_k", "sigma": "glade"},
    "T-c(0.730)": {"h": 0.730, "off0": 44000, "n": 400, "balls": "real_k", "sigma": "glade"},
    "T-c(0.770)": {"h": 0.770, "off0": 45000, "n": 200, "balls": "real_k", "sigma": "glade"},
}
RESERVED = {"W1": (46000, 46399), "O2": (47000, 47399)}
V1_ENV = (0, 9049)
V2_ENV = (20000, 29049)

# prereg §7 locked band literals (transcribed by hand from the registered file)
PREREG_DSVT1 = {
    400: {
        0.50: {"2s": (0.450, 0.550), "3s": (0.425, 0.575)},
        0.68: {"2s": (0.633, 0.727), "3s": (0.610, 0.750)},
        0.90: {"2s": (0.870, 0.930), "3s": (0.855, 0.945)},
    },
    200: {
        0.50: {"2s": (0.429, 0.571), "3s": (0.394, 0.606)},
        0.68: {"2s": (0.614, 0.746), "3s": (0.581, 0.779)},
        0.90: {"2s": (0.858, 0.942), "3s": (0.836, 0.964)},
    },
    100: {
        0.50: {"2s": (0.400, 0.600), "3s": (0.350, 0.650)},
        0.68: {"2s": (0.587, 0.773), "3s": (0.540, 0.820)},
        0.90: {"2s": (0.840, 0.960), "3s": (0.810, 0.990)},
    },
}
PREREG_DSVT2 = {400: (0.0679, 0.0814), 200: (0.0960, 0.1151), 100: (0.1358, 0.1628)}
PREREG_COLLAPSE_BAND = {400: 0.02, 200: 0.04, 100: 0.08}
DS3_IN_BAND = 0.010
DS3_DEFECT = 0.030
R_DOSE_BAND = (0.75, 1.25)
EDGE_THRESH = 0.01
EDGE_CONTAM_FRAC = 0.10
RAIL_EMERGENT = 0.90
KS_C95 = 1.358
KS_C99 = 1.628

FILE_CELL = {
    "T0_h0p730": "T-0",
    "Ta_h0p730": "T-a",
    "Tb_h0p730": "T-b",
    "Tc_h0p690": "T-c(0.690)",
    "Tc_h0p730": "T-c(0.730)",
    "Tc_h0p770": "T-c(0.770)",
}


# ── my own posterior primitives ──────────────────────────────────────────────


def trapz_norm(h, lnp):
    """Normalise exp(lnp) on grid h by my own trapezoid rule. Returns density."""
    lnp = np.asarray(lnp, dtype=np.float64)
    p = np.exp(lnp - lnp.max())
    dh = np.diff(h)
    z = float(np.sum(0.5 * (p[1:] + p[:-1]) * dh))
    return p / z


def my_cdf(h, post):
    """Cumulative trapezoid mass, my own implementation."""
    dh = np.diff(h)
    seg = 0.5 * (post[1:] + post[:-1]) * dh
    return np.concatenate([[0.0], np.cumsum(seg)])


def my_pit(h, post, h_true):
    cum = my_cdf(h, post)
    return float(np.interp(h_true, h, cum))


def my_edge_mass(h, post):
    cum = my_cdf(h, post)
    return float(cum[1] + (cum[-1] - cum[-2]))


def my_post_sd(h, post):
    dh = np.diff(h)
    m1 = float(np.sum(0.5 * (post[1:] * h[1:] + post[:-1] * h[:-1]) * dh))
    m2 = float(np.sum(0.5 * (post[1:] * h[1:] ** 2 + post[:-1] * h[:-1] ** 2) * dh))
    return math.sqrt(max(m2 - m1 * m1, 0.0))


def my_hpd_contains(h, post, h_true, level):
    """HPD credible-region containment, written from scratch.

    Density-threshold construction: sort densities descending, accumulate the
    trapezoid-consistent per-node mass (np.gradient weights, the registered
    convention), cut at the first node whose cumulative mass reaches `level`,
    and ask whether the interpolated density at h_true clears that threshold.
    """
    post = np.asarray(post, dtype=np.float64)
    w = np.gradient(h)
    mass = post * w
    order = np.argsort(post)[::-1]
    csum = np.cumsum(mass[order])
    k = int(np.searchsorted(csum, level))
    k = min(k, order.size - 1)
    thresh = float(post[order[k]])
    p_true = float(np.interp(h_true, h, post))
    return 1.0 if p_true >= thresh else 0.0


def my_argmax(h, lnp):
    lnp = np.asarray(lnp, dtype=np.float64)
    i = int(np.argmax(lnp))
    return i, float(h[i])


def my_refined_argmax(h, lnp):
    lnp = np.asarray(lnp, dtype=np.float64)
    i = int(np.argmax(lnp))
    out = float(h[i])
    if 0 < i < len(h) - 1:
        x0, x1, x2 = h[i - 1], h[i], h[i + 1]
        y0, y1, y2 = lnp[i - 1], lnp[i], lnp[i + 1]
        d1 = (y1 - y0) / (x1 - x0)
        d2 = (y2 - y1) / (x2 - x1)
        curv = (d2 - d1) / (0.5 * (x2 - x0))
        if curv < 0.0:
            out = float(min(max(0.5 * (x0 + x1) - d1 / curv, h[0]), h[-1]))
    return out


def my_ks_uniform(vals):
    """One-sample KS distance against Uniform(0,1), from scratch."""
    x = np.sort(np.asarray(vals, dtype=np.float64))
    n = x.size
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = float(np.max(i / n - x))
    d_minus = float(np.max(x - (i - 1) / n))
    return max(d_plus, d_minus)


def binom_bands(p, n):
    s = math.sqrt(p * (1.0 - p) / n)
    return (p - 2 * s, p + 2 * s), (p - 3 * s, p + 3 * s)


# ── load ─────────────────────────────────────────────────────────────────────


def load():
    files = sorted(glob.glob(os.path.join(HERE, "T*_results_seeds*.json")))
    chunks = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        key = os.path.basename(f).split("_results_")[0]
        d["_file"] = os.path.basename(f)
        d["_cell"] = FILE_CELL[key]
        chunks.append(d)
    return chunks


def main():  # noqa: C901
    chunks = load()
    out = {"n_chunk_files": len(chunks)}

    # ── provenance ───────────────────────────────────────────────────────────
    prov = defaultdict(list)
    flags = defaultdict(Counter)
    for c in chunks:
        prov[c["git_commit"][:8]].append(c["_file"])
        flags["import_path_clean"][c["import_path_clean"]] += 1
        flags["allow_dirty"][c["allow_dirty"]] += 1
        flags["smoke"][c["smoke"]] += 1
        flags["git_dirty"][c["git_dirty"]] += 1
        flags["pin_pass"][c["pin_integrity"]["pass"]] += 1
        flags["import_dirt_empty"][len(c["dirt_inventory"]["import_path"]) == 0] += 1
        flags["workers"][c["workers"]] += 1
        flags["n_events_cap"][str(c["config"]["n_events_cap"])] += 1
        flags["chunk_pairs"][c["config"]["chunk_pairs"]] += 1
        # pin sub-flags
        for k in ("crb_csv_md5", "frozeng_emit_md5"):
            flags[k][c["pin_integrity"][k]["match"]] += 1
        flags["k_census_all"][all(c["pin_integrity"]["k_census"]["match"].values())] += 1
        flags["sigma_stats_all"][all(c["pin_integrity"]["sigma_stats"]["match"].values())] += 1
    out["provenance"] = {
        "commits": {k: len(v) for k, v in prov.items()},
        "flags": {k: {str(kk): vv for kk, vv in v.items()} for k, v in flags.items()},
    }

    # h grid identity across chunks
    grids = {tuple(c["config"]["h_grid"]) for c in chunks}
    out["h_grid_unique"] = len(grids)
    h = np.asarray(sorted(grids)[0], dtype=np.float64)
    out["h_grid"] = {
        "n": int(h.size),
        "min": float(h[0]),
        "max": float(h[-1]),
        "uniform_spacing": bool(np.allclose(np.diff(h), np.diff(h)[0])),
        "dh": float(np.diff(h)[0]),
        "contains_0.690": bool(np.any(np.isclose(h, 0.690))),
        "contains_0.730": bool(np.any(np.isclose(h, 0.730))),
        "contains_0.770": bool(np.any(np.isclose(h, 0.770))),
    }

    # ── seed plan ────────────────────────────────────────────────────────────
    cell_seeds = defaultdict(list)
    for c in chunks:
        cell_seeds[c["_cell"]].extend(c["seeds"])
        # cross-check chunk seed list == per_seed seeds
        assert c["seeds"] == [r["seed"] for r in c["per_seed"]], c["_file"]

    seedplan = {}
    allseeds = []
    for cell, spec in REG_CELLS.items():
        got = sorted(cell_seeds[cell])
        exp = [BASE_SEED + spec["off0"] + i for i in range(spec["n"])]
        allseeds.extend(got)
        seedplan[cell] = {
            "expected_n": spec["n"],
            "realized_n": len(got),
            "unique_n": len(set(got)),
            "equals_registered_block": got == exp,
            "missing": sorted(set(exp) - set(got))[:10],
            "extra": sorted(set(got) - set(exp))[:10],
            "seed_min": got[0] if got else None,
            "seed_max": got[-1] if got else None,
        }
    dupes = [s for s, n in Counter(allseeds).items() if n > 1]
    offs = [s - BASE_SEED for s in allseeds]
    seedplan["_global"] = {
        "total": len(allseeds),
        "unique": len(set(allseeds)),
        "cross_cell_duplicates": len(dupes),
        "collide_v1_envelope": sum(V1_ENV[0] <= o <= V1_ENV[1] for o in offs),
        "collide_v2_envelope": sum(V2_ENV[0] <= o <= V2_ENV[1] for o in offs),
        "collide_W1_reserved": sum(RESERVED["W1"][0] <= o <= RESERVED["W1"][1] for o in offs),
        "collide_O2_reserved": sum(RESERVED["O2"][0] <= o <= RESERVED["O2"][1] for o in offs),
        "offset_min": min(offs),
        "offset_max": max(offs),
    }
    out["seed_plan"] = seedplan

    # ── per-seed independent recomputation ───────────────────────────────────
    per_cell = defaultdict(lambda: defaultdict(list))
    recompute_dev = defaultdict(float)
    bad_flags = Counter()
    nonfinite = Counter()
    ksum_by_cell = defaultdict(Counter)
    cfg_by_cell = defaultdict(Counter)
    walls = []

    for c in chunks:
        cell = c["_cell"]
        h_true = float(c["config"]["h_true"])
        cfg_by_cell[cell][
            (c["config"]["balls"], c["config"]["sigma_mode"], c["config"]["h_true"])
        ] += 1
        walls.append((cell, c["wall_time_s"], len(c["seeds"]), c["workers"]))
        for r in c["per_seed"]:
            if r["h_true"] != h_true:
                bad_flags["per_seed_h_true_mismatch"] += 1
            if r["n_events"] != 982 or r["n_events_run"] != 982:
                bad_flags["n_events_not_982"] += 1
            if r["n_horizon_dropped"] != 0:
                bad_flags["horizon_dropped_nonzero"] += 1
            if r["f_incl"] != 1.0:
                bad_flags["f_incl_not_1"] += 1
            ksum_by_cell[cell][r["K_sum"]] += 1
            for ch in ("1d", "2d"):
                lnp = np.asarray(r[f"ln_post_{ch}"], dtype=np.float64)
                if lnp.size != 41:
                    bad_flags["ln_post_len"] += 1
                nf = int(np.sum(~np.isfinite(lnp)))
                if nf:
                    nonfinite[(cell, ch)] += 1
                    per_cell[(cell, ch)]["nonfinite_seed"].append(r["seed"])
                    continue
                post = trapz_norm(h, lnp)
                i_arg, m = my_argmax(h, lnp)
                mref = my_refined_argmax(h, lnp)
                pit = my_pit(h, post, h_true)
                sd = my_post_sd(h, post)
                em = my_edge_mass(h, post)
                hpd = {lv: my_hpd_contains(h, post, h_true, lv) for lv in (0.50, 0.68, 0.90)}
                d = per_cell[(cell, ch)]
                d["map"].append(m)
                d["map_ref"].append(mref)
                d["pit"].append(pit)
                d["sd"].append(sd)
                d["edge"].append(em)
                d["hpd50"].append(hpd[0.50])
                d["hpd68"].append(hpd[0.68])
                d["hpd90"].append(hpd[0.90])
                d["rail_low"].append(1.0 if i_arg == 0 else 0.0)
                d["rail_high"].append(1.0 if i_arg == len(h) - 1 else 0.0)
                d["sigma_mean_pairs"].append(r["sigma_z_mean_pairs"])
                d["sigma_med_pairs"].append(r["sigma_z_median_pairs"])
                d["frac_lt5e3"].append(r["frac_pairs_sigma_lt_5e-3"])
                # deviation of my recomputation from the instrument's stored field
                for key, mine in (
                    (f"map_{ch}", m),
                    (f"map_{ch}_refined", mref),
                    (f"pit_{ch}", pit),
                    (f"post_sd_{ch}", sd),
                    (f"edge_mass_{ch}", em),
                    (f"hpd50_{ch}", hpd[0.50]),
                    (f"hpd68_{ch}", hpd[0.68]),
                    (f"hpd90_{ch}", hpd[0.90]),
                    (f"railed_low_{ch}", 1.0 if i_arg == 0 else 0.0),
                    (f"railed_high_{ch}", 1.0 if i_arg == len(h) - 1 else 0.0),
                ):
                    stored = float(r[key])
                    dev = abs(stored - mine)
                    kk = key.replace("_1d", "").replace("_2d", "")
                    recompute_dev[kk] = max(recompute_dev[kk], dev)

    out["raw_integrity"] = {
        "flag_violations": dict(bad_flags),
        "nonfinite_ln_post_seeds": {f"{k[0]}|{k[1]}": v for k, v in nonfinite.items()},
        "max_abs_dev_my_recompute_vs_stored_perseed": {
            k: float(v) for k, v in sorted(recompute_dev.items())
        },
        "K_sum_by_cell": {
            k: {"distinct": len(v), "values_or_range": (sorted(v) if len(v) <= 3 else [min(v), max(v)])}
            for k, v in ksum_by_cell.items()
        },
        "config_by_cell": {k: {str(kk): vv for kk, vv in v.items()} for k, v in cfg_by_cell.items()},
    }

    # ── cell aggregates + bands ──────────────────────────────────────────────
    cells_out = {}
    for cell, spec in REG_CELLS.items():
        n_row = spec["n"]  # registered N row for this cell
        for ch in ("1d", "2d"):
            d = per_cell[(cell, ch)]
            maps = np.asarray(d["map"])
            mref = np.asarray(d["map_ref"])
            n = maps.size
            h_true = spec["h"]
            bias = float(maps.mean() - h_true)
            se = float(maps.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            bias_r = float(mref.mean() - h_true)
            se_r = float(mref.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            cov = {lv: float(np.mean(d[f"hpd{lv}"])) for lv in (50, 68, 90)}
            pits = np.asarray(d["pit"])
            ks = my_ks_uniform(pits)
            r_low = float(np.mean(d["rail_low"]))
            r_high = float(np.mean(d["rail_high"]))
            sd_med = float(np.median(d["sd"]))
            edges = np.asarray(d["edge"])
            edge_frac = float(np.mean(edges > EDGE_THRESH))
            sigbar = float(np.mean(d["sigma_mean_pairs"]))
            r_dose = bias / sigbar if sigbar > 0 else float("nan")
            r_dose_r = bias_r / sigbar if sigbar > 0 else float("nan")

            # my own re-derived bands at this cell's registered N
            my_b = {}
            for lv in (0.50, 0.68, 0.90):
                b2, b3 = binom_bands(lv, n_row)
                my_b[str(lv)] = {"2s": [round(b2[0], 3), round(b2[1], 3)],
                                 "3s": [round(b3[0], 3), round(b3[1], 3)]}
            my_ks_pass = KS_C95 / math.sqrt(n_row)
            my_ks_fail = KS_C99 / math.sqrt(n_row)

            band_match = all(
                tuple(my_b[str(lv)]["2s"]) == PREREG_DSVT1[n_row][lv]["2s"]
                and tuple(my_b[str(lv)]["3s"]) == PREREG_DSVT1[n_row][lv]["3s"]
                for lv in (0.50, 0.68, 0.90)
            )
            ks_band_match = (
                round(my_ks_pass, 4) == PREREG_DSVT2[n_row][0]
                and round(my_ks_fail, 4) == PREREG_DSVT2[n_row][1]
            )

            ds1_inside3 = all(
                PREREG_DSVT1[n_row][lv]["3s"][0] <= cov[int(lv * 100)] <= PREREG_DSVT1[n_row][lv]["3s"][1]
                for lv in (0.50, 0.68, 0.90)
            )
            ds2_pass = ks <= PREREG_DSVT2[n_row][0]
            band = PREREG_COLLAPSE_BAND[n_row]
            rails_in = (r_low <= band) and (r_high <= band)
            c90_le = cov[90] <= band
            r_dose_in = R_DOSE_BAND[0] <= r_dose <= R_DOSE_BAND[1] if sigbar > 0 else False

            if c90_le and rails_in and bias >= DS3_DEFECT and r_dose_in:
                label = "COLLAPSE-REPRODUCED"
            elif ds1_inside3 and ds2_pass and abs(bias) <= DS3_IN_BAND and rails_in:
                label = "CALIBRATED"
            else:
                label = "OTHER"
            if spec["sigma"] == "zero":
                label = "ANCHOR (DS-VT1/DS-VT2 exempt, VT-D8)"

            cells_out[f"{cell}|{ch}"] = {
                "n_seeds_realized": int(n),
                "registered_N_row": n_row,
                "h_true": h_true,
                "bias_gridargmax": bias,
                "bias_SE": se,
                "bias_refined": bias_r,
                "bias_refined_SE": se_r,
                "sigma_bar_pairs": sigbar,
                "R_dose": r_dose,
                "R_dose_refined": r_dose_r,
                "R_dose_in_band": bool(r_dose_in),
                "hpd50": cov[50],
                "hpd68": cov[68],
                "hpd90": cov[90],
                "ds1_all_inside_3sigma": bool(ds1_inside3),
                "ks_D": ks,
                "ds2_pass": bool(ds2_pass),
                "rail_low": r_low,
                "rail_high": r_high,
                "rails_in_collapse_band": bool(rails_in),
                "rail_emergent": bool(max(r_low, r_high) >= RAIL_EMERGENT),
                "post_sd_median": sd_med,
                "bias_over_post_sd_median": bias / sd_med if sd_med > 0 else float("nan"),
                "edge_loaded_frac": edge_frac,
                "edge_mass_max": float(edges.max()),
                "edge_contaminated": bool(edge_frac > EDGE_CONTAM_FRAC),
                "frac_pairs_sigma_lt_5e-3_mean": float(np.mean(d["frac_lt5e3"])),
                "classification": label,
                "my_bands_match_prereg_literals": bool(band_match),
                "my_ks_bands_match_prereg_literals": bool(ks_band_match),
                "my_ks_pass_edge": my_ks_pass,
                "my_ks_fail_edge": my_ks_fail,
            }
    out["cells"] = cells_out

    # ── branch tree, evaluated in the registered order ───────────────────────
    trig = []
    vf = json.load(open(os.path.join(HERE, "validate_results_full.json")))
    trig.append(("1 V-T2 failure", not bool(vf["v_t2"]["pass"])))
    trig.append(
        ("2 V-T3 failure",
         not bool(vf["v_t3"]["pass"]) or not all(c["pin_integrity"]["pass"] for c in chunks))
    )
    trig.append(
        ("3 V-T4 failure",
         not all(c["import_path_clean"] and not c["allow_dirty"] and not c["smoke"] for c in chunks))
    )
    trig.append(("4 V-T5 failure", not bool(vf["v_t5"]["pass"])))
    max_nf = max(
        [
            len(per_cell[(cell, ch)]["nonfinite_seed"]) / max(1, REG_CELLS[cell]["n"])
            for cell in REG_CELLS
            for ch in ("1d", "2d")
        ]
    )
    trig.append(("5 abort(b) nonfinite>1%", max_nf > 0.01))
    trig.append(("6 abort(d) horizon>5%", bad_flags.get("horizon_dropped_nonzero", 0) > 0))
    t0_1d = cells_out["T-0|1d"]
    t0_2d = cells_out["T-0|2d"]
    t0_hard = any(
        abs(x["bias_gridargmax"]) >= 0.030 or x["rail_low"] > 0.05 or x["rail_high"] > 0.05
        for x in (t0_1d, t0_2d)
    )
    trig.append(("7 V-T1 T-0 hard trigger", bool(t0_hard)))
    trig.append(("8 decision cell EDGE-CONTAMINATED 1D", cells_out["T-c(0.730)|1d"]["edge_contaminated"]))
    trig.append(("9 decision cell EDGE-CONTAMINATED 2D", cells_out["T-c(0.730)|2d"]["edge_contaminated"]))
    venue_confounded = any(v for _, v in trig)

    tc_1d = [cells_out[f"T-c({t})|1d"]["classification"] for t in ("0.690", "0.730", "0.770")]
    transfer_confirmed = all(x == "COLLAPSE-REPRODUCED" for x in tc_1d)
    transfer_refuted = cells_out["T-c(0.730)|1d"]["classification"] == "CALIBRATED"

    if venue_confounded:
        branch = "VENUE-CONFOUNDED"
    elif transfer_confirmed:
        branch = "TRANSFER-CONFIRMED"
    elif transfer_refuted:
        branch = "TRANSFER-REFUTED"
    else:
        branch = "MIXED"

    out["branch"] = {
        "trigger_set_in_registered_order": [{"member": k, "fired": bool(v)} for k, v in trig],
        "venue_confounded": bool(venue_confounded),
        "T-c 1D classifications": tc_1d,
        "transfer_confirmed_condition": bool(transfer_confirmed),
        "transfer_refuted_condition": bool(transfer_refuted),
        "branch_fired": branch,
        "anchor_marginal": bool(0.010 < abs(t0_1d["bias_gridargmax"]) < 0.030
                               or 0.010 < abs(t0_2d["bias_gridargmax"]) < 0.030),
        "rail_emergent_anywhere": bool(any(v["rail_emergent"] for v in cells_out.values())),
        "edge_contaminated_anywhere": bool(any(v["edge_contaminated"] for v in cells_out.values())),
        "1d_2d_split_decision_cell": cells_out["T-c(0.730)|1d"]["classification"]
        != cells_out["T-c(0.730)|2d"]["classification"],
    }

    # ── ladder (DS-VT5), registered order ────────────────────────────────────
    v2b2 = json.load(
        open(os.path.join(REPO, "results/calibration_gate_v2_20260810/B2_h0p730_results.json"))
    )
    rung0 = {}
    for ch in ("1d", "2d"):
        agg = v2b2["aggregate"][f"channel_{ch}"]
        rung0[ch] = {
            "n_seeds": agg["n_seeds"],
            "bias_gridargmax_from_mean_map": agg["ds3_map_bias"]["mean_map"] - 0.730,
            "bias_field": agg["ds3_map_bias"]["bias"],
            "mean_map_refined_bias": agg["ds3_map_bias"]["mean_map_refined"] - 0.730,
            "hpd90": agg["ds1_coverage"]["hpd90"]["value"],
            "rails": [agg["ds4_rails"]["railed_low_frac"], agg["ds4_rails"]["railed_high_frac"]],
            "post_sd_median": agg["ds5_width"]["post_sd_median"],
            "R_dose_vs_0.035": (agg["ds3_map_bias"]["mean_map"] - 0.730) / 0.035,
        }
    out["ladder"] = {
        "rung0_v2_B2_0p730": rung0,
        "rung1_T-a": {ch: cells_out[f"T-a|{ch}"]["classification"] for ch in ("1d", "2d")},
        "rung2_T-b": {ch: cells_out[f"T-b|{ch}"]["classification"] for ch in ("1d", "2d")},
        "rung3_T-c(0.730)": {
            ch: cells_out[f"T-c(0.730)|{ch}"]["classification"] for ch in ("1d", "2d")
        },
        "killing_axis": next(
            (
                name
                for name, lab in (
                    ("T-a", cells_out["T-a|1d"]["classification"]),
                    ("T-b", cells_out["T-b|1d"]["classification"]),
                    ("T-c(0.730)", cells_out["T-c(0.730)|1d"]["classification"]),
                )
                if lab != "COLLAPSE-REPRODUCED"
            ),
            None,
        ),
        "T-a_vs_v2_delta_1d": cells_out["T-a|1d"]["bias_gridargmax"]
        - rung0["1d"]["bias_gridargmax_from_mean_map"],
        "T-a_vs_v2_delta_2d": cells_out["T-a|2d"]["bias_gridargmax"]
        - rung0["2d"]["bias_gridargmax_from_mean_map"],
    }

    # ── compute context ──────────────────────────────────────────────────────
    wall_by_cell = defaultdict(list)
    for cell, w, ns, wk in walls:
        wall_by_cell[cell].append((w, ns, wk))
    out["compute"] = {
        "total_wall_h_sum_over_chunks": sum(w for _, w, _, _ in walls) / 3600.0,
        "per_cell": {
            k: {
                "chunks": len(v),
                "median_wall_per_seed_s": float(np.median([w / ns for w, ns, _ in v])),
                "max_wall_per_seed_s": float(np.max([w / ns for w, ns, _ in v])),
                "workers": sorted({wk for _, _, wk in v}),
                # as-run per-seed CPU: one process per seed, min(workers, n_seeds)
                # seeds resident concurrently, so CPU-h/seed = wall_h * min(w,ns)/ns
                "as_run_cpu_h_per_seed_median": float(
                    np.median([w * min(wk, ns) / ns / 3600.0 for w, ns, wk in v])
                ),
                "as_run_cpu_h_per_seed_max": float(
                    np.max([w * min(wk, ns) / ns / 3600.0 for w, ns, wk in v])
                ),
            }
            for k, v in wall_by_cell.items()
        },
        "abort_a_trip_point_cpu_h_per_seed": 8.66,
        "heavy_chunks_over_trip_point_as_run": sum(
            1
            for cell, w, ns, wk in walls
            if cell in ("T-b", "T-c(0.690)", "T-c(0.730)", "T-c(0.770)")
            and w * min(wk, ns) / ns / 3600.0 > 8.66
        ),
        "heavy_chunks_total": sum(
            1 for cell, _, _, _ in walls if cell in ("T-b", "T-c(0.690)", "T-c(0.730)", "T-c(0.770)")
        ),
    }

    # ── extra adversarial probes ─────────────────────────────────────────────
    pit_stats = {}
    for cell in REG_CELLS:
        p = np.asarray(per_cell[(cell, "1d")]["pit"])
        pit_stats[cell] = {"max": float(p.max()), "median": float(np.median(p))}
    # worker-grain two-sample check inside T-c(0.730)
    m64, m25 = [], []
    for c in chunks:
        if c["_cell"] != "T-c(0.730)":
            continue
        tgt = m64 if c["workers"] == 64 else m25
        tgt.extend(r["map_1d"] for r in c["per_seed"])
    a, b = np.asarray(m64), np.asarray(m25)
    se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    out["extra_probes"] = {
        "pit_1d_by_cell": pit_stats,
        "worker_grain_two_sample_Tc0p730_map1d": {
            "n_workers64": int(a.size),
            "n_workers25": int(b.size),
            "mean64": float(a.mean()),
            "mean25": float(b.mean()),
            "welch_z": float((a.mean() - b.mean()) / se),
        },
        "n_degenerate_windows_nonzero_seeds": sum(
            1 for c in chunks for r in c["per_seed"] if r["n_degenerate_windows"] != 0
        ),
        "distinct_texture_corr": len(
            {round(r["texture_corr"], 9) for c in chunks for r in c["per_seed"]}
        ),
    }

    # ── reserved blocks unbuilt ──────────────────────────────────────────────
    out["reserved_blocks_empty"] = {
        k: sum(1 for s in allseeds if v[0] <= s - BASE_SEED <= v[1]) == 0
        for k, v in RESERVED.items()
    }

    # ── git provenance ───────────────────────────────────────────────────────
    def git(*a):
        return subprocess.run(
            ["git", "-C", REPO] + list(a), capture_output=True, text=True
        ).stdout.strip()

    def gitok(*a):
        return (
            subprocess.run(["git", "-C", REPO] + list(a), capture_output=True).returncode == 0
        )

    out["git"] = {
        "e77eecad_ancestor_of_2ece8801": gitok("merge-base", "--is-ancestor", "e77eecad", "2ece8801"),
        "2ece8801_ancestor_of_e93f3068": gitok("merge-base", "--is-ancestor", "2ece8801", "e93f3068"),
        "diff_files": git("diff", "--name-only", "2ece8801", "e93f3068").split("\n"),
        "import_path_diff_oldname_lines": len(
            git("diff", "2ece8801", "e93f3068", "--", "master_thesis_code", "master_thesis_code_test")
        ),
        "import_path_diff_newname_lines": len(
            git("diff", "2ece8801", "e93f3068", "--", "darksiren_emri", "darksiren_emri_test")
        ),
        "prereg_dirty": git("status", "--porcelain", "results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md"),
    }

    with open(os.path.join(HERE, "adjudicate_venue_transfer_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:200])
    return out


if __name__ == "__main__":
    main()
