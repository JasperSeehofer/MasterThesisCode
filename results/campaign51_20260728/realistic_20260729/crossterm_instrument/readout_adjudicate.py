#!/usr/bin/env python3
"""ADVERSARIAL adjudication of CROSSTERM_READOUT_20260806.

Independent recomputation of the decision statistic T, the [A2] stratified
sums, the anti-dilution clause, and the mechanical prereg decision tree,
directly from outputs/run_*.json.  The mixture composition is RE-DERIVED here
from the production likelihood structure, not copied from any readout code:

    Production per-event combined likelihood (mixture of catalogue leg and
    completion leg):  combined_e = w_G,e * L_cat,e + (completion leg).
    The factorized pair joint is combined_i * combined_j.  The Eq. (31)
    correction multiplies ONLY the catalogue x catalogue cross term by
    e^{Delta}:  L_cat,i*L_cat,j -> L_cat,i*L_cat,j*e^{Delta}.  Hence

    corrected = combined_i*combined_j + w_G,i*w_G,j*L_cat,i*L_cat,j*(e^Delta - 1)
    Delta_tilde = ln(corrected / factorized)
                = log1p( w_G,i*w_G,j*L_cat,i*L_cat,j*expm1(Delta)
                         / (combined_i*combined_j) )

Band of record (prereg, LOCKED at ratification 2026-08-06):
    X = 2.78, Y = 7.96 class-summed chord nats, mixture-composed statistic.
Per-unit denominators W (prereg 7.4 + certification-record completion table).

Writes readout_adjudication_20260807.json next to this script.  Read-only on
all inputs.
"""

import hashlib
import json
import math
import os
from collections import Counter

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "readout_adjudication_20260807.json")

X_BAND = 2.78
Y_BAND = 7.96
H_GRID = [0.6, 0.73, 0.81, 0.86]
RATIFIED_SHA = "340b66d2f970e48cf5152676e8b6bed6b171f9538efa33bab6e5ef04abd87692"
W_DENOM = {  # prereg 7.4 (1D) + certification-record 7.4-completion (2D)
    ("joint_r1", "1d"): 19.803870,
    ("iiib", "1d"): 14.728652,
    ("joint_r1", "2d"): 4.094919851,
    ("iiib", "2d"): 0.664499854,
}
EXPECTED_N = {
    ("joint_r1", "1d"): 349,
    ("joint_r1", "2d"): 104,
    ("iiib", "1d"): 280,
    ("iiib", "2d"): 21,
}
EXPECTED_INC4 = {
    ("joint_r1", "1d"): 80,
    ("joint_r1", "2d"): 27,
    ("iiib", "1d"): 63,
    ("iiib", "2d"): 5,
}

NS_BANDS = [
    ("1", 1, 1),
    ("2-10", 2, 10),
    ("11-100", 11, 100),
    ("101-1000", 101, 1000),
    (">1000", 1001, 10**12),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def compose(row):
    """Mixture-composed correction, re-derived (module docstring)."""
    d = row["delta_joint_lnL_nats"]
    fac = (
        row["w_G_i"]
        * row["w_G_j"]
        * row["L_cat_i"]
        * row["L_cat_j"]
        / (row["combined_i"] * row["combined_j"])
    )
    return math.log1p(fac * math.expm1(d)), fac


def one_signed(vals):
    pos = sum(v for v in vals if v > 0)
    neg = sum(v for v in vals if v < 0)
    return pos, neg


def adjudicate(venue, channel):
    run_path = os.path.join(BASE, "outputs", f"run_{venue}_{channel}.json")
    with open(run_path) as f:
        run = json.load(f)
    with open(os.path.join(BASE, f"target_pairs_{venue}.json")) as f:
        tp = json.load(f)
    rows = run["rows"]
    meta = run["meta"]
    res = {
        "file": os.path.relpath(run_path, BASE),
        "meta_venue": meta["venue"],
        "meta_channel": meta["channel"],
        "git_commit": meta["git_commit"],
    }

    # ---- coverage: exact multiset equality with ratified target x h-grid ----
    target_pairs = [(p["i"], p["j"]) for p in tp["pairs"][channel]]
    expected = Counter((i, j, h) for (i, j) in target_pairs for h in H_GRID)
    got = Counter((r["event_i"], r["event_j"], r["h_requested"]) for r in rows)
    res["n_rows"] = len(rows)
    res["n_target_pairs"] = len(target_pairs)
    res["coverage_exact"] = expected == got
    res["n_pairs_matches_prereg"] = len(target_pairs) == EXPECTED_N[(venue, channel)]
    res["h_grid_matched_ok"] = all(r["h_grid_matched"] == r["h_requested"] for r in rows)

    # in-C-4 split (from run rows; h-invariant per pair)
    inc4 = {}
    for r in rows:
        key = (r["event_i"], r["event_j"])
        if key in inc4:
            assert inc4[key] == r["in_c4"], "in_c4 not h-invariant"
        inc4[key] = r["in_c4"]
    res["n_in_c4"] = sum(1 for v in inc4.values() if v)
    res["in_c4_split_matches_prereg"] = res["n_in_c4"] == EXPECTED_INC4[(venue, channel)]

    # ---- guard census ----
    n_nan = sum(
        1
        for r in rows
        if not math.isfinite(r["delta_joint_lnL_nats"]) and math.isnan(r["delta_joint_lnL_nats"])
    )
    n_neginf = sum(1 for r in rows if r["delta_joint_lnL_nats"] == -math.inf)
    n_s_nonpos = sum(1 for r in rows if r["S_i_raw"] <= 0 or r["S_j_raw"] <= 0)
    res["guard"] = {
        "n_nan": n_nan,
        "n_neg_inf": n_neginf,
        "n_S_nonpositive": n_s_nonpos,
        "auto_escalate_triggered": n_neginf > 0,
        "voided_pairs": n_nan,
    }

    # ---- escalation audit ----
    esc = [
        (
            r["event_i"],
            r["event_j"],
            r["h_requested"],
            r["quad_n_max_shared"],
            r["n_escalated_shared"],
        )
        for r in rows
        if r["n_escalated_shared"] > 0 or r["quad_n_max_shared"] != 50
    ]
    res["escalated_rows"] = esc

    # ---- S raw vs frozeng ----
    devs = []
    for r in rows:
        for a, b in (("S_i_raw", "S_i_frozeng"), ("S_j_raw", "S_j_frozeng")):
            if r[b] is not None and r[b] != 0:
                devs.append(abs(r[a] - r[b]) / abs(r[b]))
    res["max_rel_dev_S_raw_vs_frozeng"] = max(devs) if devs else None

    # ---- w_G degeneracy ----
    wg_by_h = {}
    for r in rows:
        wg_by_h.setdefault(r["h_requested"], set()).update(
            (round(r["w_G_i"], 15), round(r["w_G_j"], 15))
        )
    res["w_G_unique_per_h"] = {str(h): sorted(s) for h, s in sorted(wg_by_h.items())}
    res["w_G_degenerate"] = all(len(s) == 1 for s in wg_by_h.values())

    # ---- per-(pair,h) composed values; class sums ----
    dt = {}  # (i,j) -> {h: Delta_tilde}
    draw = {}  # (i,j) -> {h: Delta_raw}
    fac73 = {}  # (i,j) -> composition factor at h=0.73
    pair_attr = {}
    wshare_spread = 0.0
    for r in rows:
        key = (r["event_i"], r["event_j"])
        dtv, fac = compose(r)
        dt.setdefault(key, {})[r["h_requested"]] = dtv
        draw.setdefault(key, {})[r["h_requested"]] = r["delta_joint_lnL_nats"]
        if r["h_requested"] == 0.73:
            fac73[key] = fac
        a = pair_attr.setdefault(
            key,
            {
                "n_shared": r["n_shared"],
                "ov_max": max(r["overlap_degree_i"], r["overlap_degree_j"]),
                "w_share_min": min(r["w_share_ball_i"], r["w_share_ball_j"]),
                "in_c4": r["in_c4"],
            },
        )
        wshare_spread = max(
            wshare_spread, abs(min(r["w_share_ball_i"], r["w_share_ball_j"]) - a["w_share_min"])
        )
        assert a["n_shared"] == r["n_shared"], "n_shared not h-invariant"
    res["w_share_min_side_max_h_spread"] = wshare_spread

    D = {h: sum(dt[k][h] for k in dt) for h in H_GRID}
    D_raw = {h: sum(draw[k][h] for k in draw) for h in H_GRID}
    res["class_sum_mixture_D"] = {str(h): D[h] for h in H_GRID}
    res["class_sum_raw_D"] = {str(h): D_raw[h] for h in H_GRID}

    # T = full-grid range of D(h); argmax endpoints
    hs = sorted(H_GRID)
    best = max((abs(D[a] - D[b]), a, b) for a in hs for b in hs if a < b)
    T = best[0]
    res["T"] = T
    res["T_argmax_endpoints"] = [best[1], best[2]]
    res["T_raw"] = max(abs(D_raw[a] - D_raw[b]) for a in hs for b in hs if a < b)
    res["level_D_073"] = D[0.73]
    res["chord_D060_minus_D073"] = D[0.6] - D[0.73]
    res["chord_raw_D060_minus_D073"] = D_raw[0.6] - D_raw[0.73]

    # ---- [A2] strata on the composed chord at class argmax endpoints ----
    h_lo, h_hi = 0.6, 0.86
    chord = {k: dt[k][h_lo] - dt[k][h_hi] for k in dt}
    strat = {}

    def add_stratum(name, keys):
        vals = [chord[k] for k in keys]
        pos, neg = one_signed(vals)
        strat[name] = {
            "n": len(keys),
            "net": sum(vals),
            "pos": pos,
            "neg": neg,
            "max_one_signed": max(pos, -neg),
        }

    for label, lo, hi in NS_BANDS:
        add_stratum(f"n_shared:{label}", [k for k in chord if lo <= pair_attr[k]["n_shared"] <= hi])
    for deg in sorted({pair_attr[k]["ov_max"] for k in chord}):
        add_stratum(f"overlap_deg_max:{deg}", [k for k in chord if pair_attr[k]["ov_max"] == deg])
    add_stratum("w_G:all(degenerate)", list(chord))
    m = np.array([pair_attr[k]["w_share_min"] for k in chord])
    cuts = np.quantile(m, [0.25, 0.5, 0.75])
    res["w_share_quartile_cuts"] = cuts.tolist()
    keys = list(chord)
    qidx = np.searchsorted(cuts, m, side="left")  # 0..3
    for q in range(4):
        add_stratum(f"w_share:Q{q + 1}", [k for k, qi in zip(keys, qidx) if qi == q])
    add_stratum("in_c4", [k for k in chord if pair_attr[k]["in_c4"]])
    add_stratum("outside_c4", [k for k in chord if not pair_attr[k]["in_c4"]])
    npos = sum(1 for v in chord.values() if v > 0)
    nneg = sum(1 for v in chord.values() if v < 0)
    nzero = sum(1 for v in chord.values() if v == 0)
    posall, negall = one_signed(list(chord.values()))
    strat["sign_decomposition"] = {
        "n_pos": npos,
        "n_neg": nneg,
        "n_zero": nzero,
        "pos": posall,
        "neg": negall,
        "max_one_signed": max(posall, -negall),
    }
    res["a2_strata"] = strat
    max_os = max(s["max_one_signed"] for s in strat.values())
    res["anti_dilution"] = {
        "max_one_signed_any_stratum": max_os,
        "ge_Y": max_os >= Y_BAND,
        "triggered": max_os >= Y_BAND,
    }

    # robustness: one-signed sub-sums for ALL 6 grid chords, all strata
    rob_max = 0.0
    for a in hs:
        for b in hs:
            if a >= b:
                continue
            ch_ab = {k: dt[k][a] - dt[k][b] for k in dt}
            groups = []
            for label, lo, hi in NS_BANDS:
                groups.append([k for k in ch_ab if lo <= pair_attr[k]["n_shared"] <= hi])
            for deg in sorted({pair_attr[k]["ov_max"] for k in ch_ab}):
                groups.append([k for k in ch_ab if pair_attr[k]["ov_max"] == deg])
            for q in range(4):
                groups.append([k for k, qi in zip(keys, qidx) if qi == q])
            groups.append([k for k in ch_ab if pair_attr[k]["in_c4"]])
            groups.append([k for k in ch_ab if not pair_attr[k]["in_c4"]])
            groups.append(list(ch_ab))
            for g in groups:
                p, n = one_signed([ch_ab[k] for k in g])
                rob_max = max(rob_max, p, -n)
    res["anti_dilution_robustness_all_6_chords_max_one_signed"] = rob_max

    # ---- diagnostics the readout quotes ----
    res["composition_factor_h073"] = {
        "median": float(np.median(list(fac73.values()))),
        "max": max(fac73.values()),
    }
    top = sorted(chord.items(), key=lambda kv: -abs(kv[1]))[:3]
    res["top_pairs_by_abs_chord"] = [
        {"pair": list(k), "chord": v, "share_of_T": abs(v) / T} for k, v in top
    ]
    raw_max = max(((r["delta_joint_lnL_nats"], r) for r in rows), key=lambda t: t[0])
    rmr = raw_max[1]
    res["largest_raw_delta"] = {
        "pair": [rmr["event_i"], rmr["event_j"]],
        "h": rmr["h_requested"],
        "delta_raw": rmr["delta_joint_lnL_nats"],
        "n_shared": rmr["n_shared"],
        "delta_tilde": compose(rmr)[0],
        "factor": compose(rmr)[1],
    }

    # ---- mechanical decision tree (prereg s9) ----
    W = W_DENOM[(venue, channel)]
    res["band"] = {
        "X": X_BAND,
        "Y": Y_BAND,
        "W": W,
        "x": X_BAND / W,
        "y": Y_BAND / W,
        "T_over_W": T / W,
    }
    if n_neginf > 0:
        verdict = "AUTO-ESCALATE (neg-inf row)"
    elif T >= Y_BAND:
        verdict = "REGARD"
    elif T < X_BAND:
        verdict = (
            "NEGLECT-WITH-NUMBER"
            if not res["anti_dilution"]["triggered"]
            else "NEGLECT-BLOCKED (anti-dilution)"
        )
    else:
        verdict = "GAP"
    res["verdict"] = verdict
    return res


def main():
    out = {
        "instrument_sha256": sha256(os.path.join(BASE, "crossterm_instrument.py")),
        "ratified_pin": RATIFIED_SHA,
    }
    out["sha256_matches_pin"] = out["instrument_sha256"] == RATIFIED_SHA
    out["venues"] = {}
    for venue in ("joint_r1", "iiib"):
        for channel in ("1d", "2d"):
            out["venues"][f"{venue}/{channel}"] = adjudicate(venue, channel)
    verdicts = {k: v["verdict"] for k, v in out["venues"].items()}
    out["venue_split"] = len(set(verdicts.values())) > 1
    out["verdicts"] = verdicts

    # ---- compare against the readout's machine-readable companion ----
    with open(os.path.join(BASE, "readout_20260806.json")) as f:
        ro = json.load(f)
    cmp_rows = []
    for venue in ("joint_r1", "iiib"):
        for channel in ("1d", "2d"):
            mine = out["venues"][f"{venue}/{channel}"]
            theirs = ro["venues"][venue][channel]
            T_ro = theirs["decision_statistic"]["T_full_grid_range"]
            rel = abs(mine["T"] - T_ro) / abs(T_ro)
            ad_ro = theirs["anti_dilution"]["max_one_signed_subsum_any_stratum"]
            ad_rel = abs(mine["anti_dilution"]["max_one_signed_any_stratum"] - ad_ro) / abs(ad_ro)
            lvl_ro = theirs["decision_statistic"]["level_D_0.73_reported_not_scored"]
            cmp_rows.append(
                {
                    "venue_channel": f"{venue}/{channel}",
                    "T_mine": mine["T"],
                    "T_readout": T_ro,
                    "T_rel_diff": rel,
                    "level_rel_diff": abs(mine["level_D_073"] - lvl_ro) / abs(lvl_ro),
                    "anti_dilution_max_rel_diff": ad_rel,
                    "verdict_readout": theirs["verdict"],
                    "verdict_mine": mine["verdict"],
                    "verdict_match": theirs["verdict"].startswith(mine["verdict"])
                    or mine["verdict"] == theirs["verdict"],
                }
            )
    out["comparison_vs_readout_json"] = cmp_rows

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(cmp_rows, indent=1, default=float))
    print("sha matches pin:", out["sha256_matches_pin"])
    print("verdicts:", verdicts, "venue_split:", out["venue_split"])
    for k, v in out["venues"].items():
        print(
            k,
            "coverage_exact:",
            v["coverage_exact"],
            "h_match:",
            v["h_grid_matched_ok"],
            "guards:",
            v["guard"],
            "T:",
            f"{v['T']:.6e}",
            "endpoints:",
            v["T_argmax_endpoints"],
            "T_raw:",
            f"{v['T_raw']:.6f}",
            "robust_max_one_signed:",
            f"{v['anti_dilution_robustness_all_6_chords_max_one_signed']:.3e}",
        )


if __name__ == "__main__":
    main()
