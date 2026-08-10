"""ADVERSARIAL ADJUDICATION of the v2 calibration-gate readout.

Independent verification layer: recomputes every scored statistic from the
rawest per-seed data (the ``ln_post_1d``/``ln_post_2d`` vectors) in the 9
registered campaign JSONs + R0, using OWN implementations (own trapezoid /
PIT / HPD / argmax / KS code — nothing imported from the instrument or the
scorer), re-derives the DS-8 analytic bands from v1's committed readout JSON
and from Jeffreys-interval arithmetic, re-applies the registered branch tree
mechanically, and cross-checks every number the readout quotes.

Read-only on all campaign JSONs. Output: adjudicate_v2_readout_results.json.

Usage: cd <repo root> && uv run python results/calibration_gate_v2_20260810/adjudicate_v2_readout.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

DIR = Path(__file__).resolve().parent
REPO = DIR.parent.parent

REGISTERED_COMMIT = "065e7f58cb94c2e33d7ae1db385bcc85c93168dc"
RUN_COMMIT = "dbde71dc65df11f7e237ece2fd1962488ecf880d"
BASE = 20260808
CRB_PATH = REPO / "results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv"
CRB_MD5_EXPECTED = "9a1f2a14384a9281c97ca3be312ddaab"

FILES = {
    ("A", 0.690): "A_h0p690_results.json",
    ("A", 0.730): "A_h0p730_results.json",
    ("A", 0.770): "A_h0p770_results.json",
    ("B0", 0.730): "B0_h0p730_results.json",
    ("B1", 0.730): "B1_h0p730_results.json",
    ("B2", 0.690): "B2_h0p690_results.json",
    ("B2", 0.730): "B2_h0p730_results.json",
    ("B2", 0.770): "B2_h0p770_results.json",
    ("V1", 0.730): "V1_h0p730_results.json",
}
SEED_PLAN = {
    ("A", 0.690): (20000, 400), ("A", 0.730): (21000, 400), ("A", 0.770): (22000, 400),
    ("B0", 0.730): (23000, 400), ("B1", 0.730): (24000, 400),
    ("B2", 0.690): (25000, 400), ("B2", 0.730): (26000, 400), ("B2", 0.770): (27000, 400),
    ("V1", 0.730): (29000, 50),
}
V1_ENVELOPE = (BASE + 0, BASE + 9049)
O1_BLOCK = (BASE + 28000, BASE + 28399)

TRUTHS = (0.690, 0.730, 0.770)
HPD_LEVELS = (0.50, 0.68, 0.90)
EDGE_MASS_THRESHOLD = 0.01
EDGE_CONTAM_FRAC = 0.10
DS3_INBAND, DS3_DEFECT = 0.010, 0.030
DS6_HI, DS6_LO = 0.90, 0.05
V4_BAND = (0.63, 0.75)
KS_C95, KS_C99 = 1.358, 1.628

# prereg-printed DS-8 T2 bias bands (transcription targets)
PREREG_T2_BANDS = {
    ("B1", 0.730, "1d"): (+0.01036, +0.01147),
    ("B1", 0.730, "2d"): (+0.01059, +0.01181),
    ("B2", 0.690, "1d"): (+0.03434, +0.03551),
    ("B2", 0.690, "2d"): (+0.03408, +0.03627),
    ("B2", 0.730, "1d"): (+0.03476, +0.03606),
    ("B2", 0.730, "2d"): (+0.03456, +0.03696),
    ("B2", 0.770, "1d"): (+0.03584, +0.03841),
    ("B2", 0.770, "2d"): (+0.03673, +0.03957),
}
V1_READOUT_JSON = REPO / "results/calibration_gate_20260808/CALIBRATION_GATE_READOUT_20260808.json"
R0_REFERENCE_JSON = REPO / "results/closed_loop_gfrac_20260805/closed_loop_results.json"


# ── own readout implementations (independent of the instrument) ──────────────


def own_local_spacing(h: np.ndarray) -> np.ndarray:
    """Per-node spacing weights: one-sided at ends, centred interior.

    Reproduces the registered definition's np.gradient(h_grid) semantics with
    an explicit own formula.
    """
    d = np.empty_like(h)
    d[0] = h[1] - h[0]
    d[-1] = h[-1] - h[-2]
    d[1:-1] = (h[2:] - h[:-2]) / 2.0
    return d


def own_pp(h: np.ndarray, ln_post: np.ndarray, h_true: float) -> dict:
    """PIT, HPD50/68/90 containment, post sd, edge mass, MAP, rails — own code."""
    if not np.all(np.isfinite(ln_post)):
        return {"finite": False}
    i_map = int(np.argmax(ln_post))
    p = np.exp(ln_post - ln_post[i_map])
    seg = 0.5 * (p[1:] + p[:-1]) * np.diff(h)
    norm = float(np.sum(seg))
    post = p / norm
    cum = np.concatenate([[0.0], np.cumsum(seg / norm)])
    pit = float(np.interp(h_true, h, cum))
    mean = float(np.sum(0.5 * (post[1:] * h[1:] + post[:-1] * h[:-1]) * np.diff(h)))
    ex2 = float(np.sum(0.5 * (post[1:] * h[1:] ** 2 + post[:-1] * h[:-1] ** 2) * np.diff(h)))
    var = ex2 - mean * mean
    edge = float(cum[1] + (cum[-1] - cum[-2]))
    out = {
        "finite": True,
        "pit": pit,
        "post_sd": math.sqrt(max(var, 0.0)),
        "edge_mass": edge,
        "map": float(h[i_map]),
        "railed_low": float(i_map == 0),
        "railed_high": float(i_map == len(h) - 1),
    }
    dh = own_local_spacing(h)
    mass = post * dh
    order = np.argsort(post, kind="mergesort")[::-1]
    csum = np.cumsum(mass[order])
    p_true = float(np.interp(h_true, h, post))
    for lv in HPD_LEVELS:
        k = int(np.searchsorted(csum, lv, side="left"))
        k = min(k, order.size - 1)
        thresh = float(post[order[k]])
        out[f"hpd{int(round(lv * 100))}"] = float(p_true >= thresh)
    return out


def own_ks(pits: np.ndarray) -> float:
    q = np.sort(pits)
    n = q.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(np.max(np.maximum(i / n - q, q - (i - 1.0) / n)))


def ds1_bands(n: int) -> dict:
    out = {}
    for lv in HPD_LEVELS:
        sig = math.sqrt(lv * (1.0 - lv) / n)
        out[lv] = {"2s": (lv - 2 * sig, lv + 2 * sig), "3s": (lv - 3 * sig, lv + 3 * sig)}
    return out


def ds1_status(cvals: dict, n: int) -> str:
    b = ds1_bands(n)
    status = "PASS"
    for lv in HPD_LEVELS:
        v = cvals[lv]
        lo3, hi3 = b[lv]["3s"]
        lo2, hi2 = b[lv]["2s"]
        if not (lo3 <= v <= hi3):
            return "FAIL"
        if not (lo2 <= v <= hi2):
            status = "MARGINAL"
    return status


def ds2_status(d: float, n: int) -> str:
    d95, d99 = KS_C95 / math.sqrt(n), KS_C99 / math.sqrt(n)
    return "PASS" if d <= d95 else ("FAIL" if d > d99 else "MARGINAL")


def ds3_status(bias: float) -> str:
    a = abs(bias)
    return "IN-BAND" if a <= DS3_INBAND else ("DEFECT-SCALE" if a >= DS3_DEFECT else "MIXED-SCALE")


def load(name: str) -> dict:
    with open(DIR / name) as f:
        return json.load(f)


def main() -> None:
    discrepancies: list[str] = []
    notes: list[str] = []
    docs = {k: load(v) for k, v in FILES.items()}
    r0 = load("R0_results.json")
    readout = load("CALGATE_V2_READOUT.json")

    # ── provenance ───────────────────────────────────────────────────────────
    crb_md5 = hashlib.md5(CRB_PATH.read_bytes()).hexdigest()
    if crb_md5 != CRB_MD5_EXPECTED:
        discrepancies.append(f"CRB md5 mismatch: {crb_md5} != {CRB_MD5_EXPECTED}")

    all_seeds: list[int] = []
    prov = {}
    for k, d in docs.items():
        name = "%s_h%.3f" % k
        start, n_exp = SEED_PLAN[k]
        seeds = d["seeds"]
        plan_ok = (
            len(seeds) == n_exp
            and seeds[0] == BASE + start
            and seeds == list(range(seeds[0], seeds[0] + n_exp))
        )
        if not plan_ok:
            discrepancies.append(f"{name}: seed plan violates prereg §5")
        rec_ok = (
            len(d["per_seed"]) == len(seeds) == d["aggregate"]["n_seeds"]
            and all(ps["seed"] == s for ps, s in zip(d["per_seed"], seeds))
        )
        if not rec_ok:
            discrepancies.append(f"{name}: per_seed/seeds record incomplete or misordered")
        all_seeds += seeds
        prov[name] = {
            "git_commit_run": d["git_commit"] == RUN_COMMIT,
            "import_path_clean": bool(d["import_path_clean"]),
            "dirt_import_empty": d["dirt_inventory"]["import_path"] == [],
            "allow_dirty": d["allow_dirty"],
            "seed_plan_ok": plan_ok,
            "record_complete": rec_ok,
            "workers": d["workers"],
            "wall_time_s": d["wall_time_s"],
        }
        if not (d["git_commit"] == RUN_COMMIT and d["import_path_clean"]
                and d["dirt_inventory"]["import_path"] == [] and d["allow_dirty"] is False):
            discrepancies.append(f"{name}: provenance fields not as reported")
    disjoint = len(set(all_seeds)) == len(all_seeds)
    outside_v1 = all(not (V1_ENVELOPE[0] <= s <= V1_ENVELOPE[1]) for s in all_seeds)
    o1_untouched = all(not (O1_BLOCK[0] <= s <= O1_BLOCK[1]) for s in all_seeds)
    for flag, msg in (
        (disjoint, "seeds not mutually disjoint"),
        (outside_v1, "some seed inside v1 envelope"),
        (o1_untouched, "O1 reserved block touched"),
    ):
        if not flag:
            discrepancies.append(msg)
    if not (r0["git_commit"] == RUN_COMMIT and r0["import_path_clean"]
            and r0["dirt_inventory"]["import_path"] == []):
        discrepancies.append("R0: provenance fields not as reported")
    wall_sum = sum(d["wall_time_s"] for d in docs.values())

    # grid identity
    g41 = np.asarray(docs[("B2", 0.730)]["config"]["h_grid"])
    for k in (("B0", 0.730), ("B1", 0.730), ("B2", 0.690), ("B2", 0.770), ("V1", 0.730)):
        if docs[k]["config"]["h_grid"] != list(g41):
            discrepancies.append(f"{k}: canonical grid differs across ball cells")
    if not (len(g41) == 41 and g41[0] == 0.600 and g41[-1] == 0.860 and 0.730 in g41):
        discrepancies.append("canonical grid is not the registered 41-point 0.600–0.860 grid")
    g75 = np.asarray(docs[("A", 0.730)]["config"]["h_grid"])
    canon_in_75 = [int(np.argmin(np.abs(g75 - v))) for v in g41]
    canon_subset = all(abs(g75[i] - v) < 1e-12 for i, v in zip(canon_in_75, g41))
    if not (len(g75) == 75 and g75[0] == 0.460 and g75[-1] == 1.060 and canon_subset):
        discrepancies.append("A grid is not the registered 75-point 0.460–1.060 superset")
    for t in TRUTHS:
        if docs[("A", t)]["config"]["h_grid"] != list(g75):
            discrepancies.append(f"A({t}): extended grid differs across A cells")

    # config sanity vs prereg §5
    cfg_expect = {
        ("A", "ball"): False, ("B0", "sigma_z"): 0.0, ("B1", "sigma_z"): 0.010,
        ("B2", "sigma_z"): 0.035, ("V1", "lambda_ball"): 0.0,
    }
    for k, d in docs.items():
        c = d["config"]
        if c["n_events"] != 1500 or c["f_incl"] != 1.0 or c["sigma_texture"] != "dl_binned":
            discrepancies.append(f"{k}: config deviates from prereg §5")
        if c["h_true"] != k[1] or c["cell"] != k[0]:
            discrepancies.append(f"{k}: cell/truth mismatch")
    for (cell, key), val in cfg_expect.items():
        for k, d in docs.items():
            if k[0] == cell and d["config"][key] != val:
                discrepancies.append(f"{cell}: config {key} != {val}")

    # ── full per-seed recompute ──────────────────────────────────────────────
    cells: dict[str, dict] = {}
    nonfinite_total = 0
    per_seed_max_dev = 0.0
    hpd_bool_mismatches = 0
    for k, d in docs.items():
        name = "%s_h%.3f" % k
        h = np.asarray(d["config"]["h_grid"])
        h_true = k[1]
        n = len(d["per_seed"])
        ch_out = {}
        for ch in ("1d", "2d"):
            rows = []
            for ps in d["per_seed"]:
                ln = np.asarray(ps[f"ln_post_{ch}"], dtype=np.float64)
                if not np.all(np.isfinite(ln)):
                    nonfinite_total += 1
                    continue
                r = own_pp(h, ln, h_true)
                rows.append(r)
                # cross-check stored per-seed scalars
                for fld, mine in (
                    (f"pit_{ch}", r["pit"]), (f"post_sd_{ch}", r["post_sd"]),
                    (f"edge_mass_{ch}", r["edge_mass"]), (f"map_{ch}", r["map"]),
                    (f"railed_low_{ch}", r["railed_low"]), (f"railed_high_{ch}", r["railed_high"]),
                ):
                    dev = abs(ps[fld] - mine)
                    per_seed_max_dev = max(per_seed_max_dev, dev)
                    if dev > 1e-9:
                        discrepancies.append(
                            f"{name} seed {ps['seed']} {fld}: stored {ps[fld]} vs recomputed {mine}"
                        )
                for lv in HPD_LEVELS:
                    fld = f"hpd{int(round(lv * 100))}_{ch}"
                    if ps[fld] != r[f"hpd{int(round(lv * 100))}"]:
                        hpd_bool_mismatches += 1
                        discrepancies.append(f"{name} seed {ps['seed']} {fld} boolean mismatch")
            pits = np.asarray([r["pit"] for r in rows])
            cov = {lv: float(np.mean([r[f"hpd{int(round(lv * 100))}"] for r in rows])) for lv in HPD_LEVELS}
            maps = np.asarray([r["map"] for r in rows])
            bias = float(np.mean(maps)) - h_true
            se = float(np.std(maps, ddof=1) / math.sqrt(len(maps)))
            edges = np.asarray([r["edge_mass"] for r in rows])
            elf = float(np.mean(edges > EDGE_MASS_THRESHOLD))
            ch_out[ch] = {
                "C": cov,
                "ds1_status": ds1_status(cov, n),
                "ks_D": own_ks(pits),
                "ds2_status": ds2_status(own_ks(pits), n),
                "bias": bias,
                "bias_se": se,
                "ds3_status": ds3_status(bias),
                "R_low": float(np.mean([r["railed_low"] for r in rows])),
                "R_high": float(np.mean([r["railed_high"] for r in rows])),
                "edge_loaded_frac": elf,
                "edge_contaminated": elf > EDGE_CONTAM_FRAC,
                "post_sd_median": float(np.median([r["post_sd"] for r in rows])),
                "n": len(rows),
            }
        cells[name] = {
            "channels": ch_out,
            "exempt_flag": bool(d["aggregate"]["ds1_ds2_degenerate_pit_exempt"]),
            "texture_corr_median_recomputed": float(np.median([ps["texture_corr"] for ps in d["per_seed"]])),
            "K_mean_recomputed": float(np.mean([ps["K_mean"] for ps in d["per_seed"]])),
        }

    # D3 exemption flags as registered
    for name, want in (("B0_h0.730", True), ("V1_h0.730", True)):
        if cells[name]["exempt_flag"] is not want:
            discrepancies.append(f"{name}: degenerate-PIT exemption flag != {want}")
    for name in ("A_h0.690", "A_h0.730", "A_h0.770", "B1_h0.730",
                 "B2_h0.690", "B2_h0.730", "B2_h0.770"):
        if cells[name]["exempt_flag"]:
            discrepancies.append(f"{name}: unexpected degenerate-PIT exemption flag")

    # cross-check my aggregates vs the readout JSON's quoted cells
    for name, rec in cells.items():
        rd = readout["cells"][name]
        for ch in ("1d", "2d"):
            mine, theirs = rec["channels"][ch], rd["channels"][ch]
            for lv, key in ((0.50, "hpd50"), (0.68, "hpd68"), (0.90, "hpd90")):
                if abs(mine["C"][lv] - theirs["ds1"][key]["value"]) > 1e-12:
                    discrepancies.append(f"{name}-{ch} C{int(lv*100)}: readout {theirs['ds1'][key]['value']} vs mine {mine['C'][lv]}")
            if abs(mine["ks_D"] - theirs["ds2"]["D"]) > 1e-9:
                discrepancies.append(f"{name}-{ch} KS D: readout {theirs['ds2']['D']} vs mine {mine['ks_D']}")
            if abs(mine["bias"] - theirs["ds3"]["bias"]) > 1e-9:
                discrepancies.append(f"{name}-{ch} bias: readout {theirs['ds3']['bias']} vs mine {mine['bias']}")
            if abs(mine["R_low"] - theirs["ds4"]["R_low"]) > 1e-12 or abs(mine["R_high"] - theirs["ds4"]["R_high"]) > 1e-12:
                discrepancies.append(f"{name}-{ch} rails mismatch")
            if abs(mine["edge_loaded_frac"] - theirs["edge_guard"]["edge_loaded_frac"]) > 1e-12:
                discrepancies.append(f"{name}-{ch} edge_loaded_frac mismatch")
            if not cells[name]["exempt_flag"]:
                if mine["ds1_status"] != theirs["ds1"]["status"] or mine["ds1_status"] != theirs["ds1_status_instrument"]:
                    discrepancies.append(f"{name}-{ch} DS-1 status mismatch: mine {mine['ds1_status']}")
                if mine["ds2_status"] != theirs["ds2"]["status"] or mine["ds2_status"] != theirs["ds2_status_instrument"]:
                    discrepancies.append(f"{name}-{ch} DS-2 status mismatch: mine {mine['ds2_status']}")
            if mine["ds3_status"] != theirs["ds3"]["status"]:
                discrepancies.append(f"{name}-{ch} DS-3 status mismatch: mine {mine['ds3_status']}")

    # ── validity ─────────────────────────────────────────────────────────────
    v1c = cells["V1_h0.730"]["channels"]
    v1doc = docs[("V1", 0.730)]
    node073 = float(g41[int(np.argmin(np.abs(g41 - 0.73)))])
    v1_1d = sum(1 for ps in v1doc["per_seed"]
                if abs(float(g41[int(np.argmax(ps["ln_post_1d"]))]) - 0.730) < 1e-12)
    v1_2d = sum(1 for ps in v1doc["per_seed"]
                if abs(float(g41[int(np.argmax(ps["ln_post_2d"]))]) - 0.730) < 1e-12)
    v1_pass = v1_1d == 50 and v1_2d == 50 and node073 == 0.730
    if not v1_pass:
        discrepancies.append(f"V1 control: {v1_1d}/50 (1d), {v1_2d}/50 (2d) exact-on-truth")

    v4_meds = {name: rec["texture_corr_median_recomputed"] for name, rec in cells.items()}
    v4_all = all(V4_BAND[0] <= m <= V4_BAND[1] for m in v4_meds.values())
    if not v4_all:
        discrepancies.append(f"V4 medians outside band: {v4_meds}")
    for name, rec in cells.items():
        rep = docs[{v: k for k, v in [(kk, "%s_h%.3f" % kk) for kk in FILES]}[name]][
            "aggregate"]["texture"]["corr_ln_sigma_dl_ln_dl_median"]
        if abs(rec["texture_corr_median_recomputed"] - rep) > 1e-12:
            discrepancies.append(f"{name}: texture median recompute differs from aggregate")

    # V5 — independent R0-vs-committed-reference comparison
    ref = json.load(open(R0_REFERENCE_JSON))
    ref_by_seed = {ps["seed"]: ps for ps in ref["per_seed"]}
    v5_max_rel = 0.0
    v5_n = 0
    v5_scalar_max = 0.0
    for ps in r0["per_seed"]:
        rp = ref_by_seed.get(ps["seed"])
        if rp is None:
            discrepancies.append(f"V5: R0 seed {ps['seed']} absent from committed reference")
            continue
        for ch in ("1d", "2d"):
            a = np.asarray(ps[f"ln_post_{ch}"])
            b = np.asarray(rp[f"ln_post_{ch}"])
            rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
            v5_max_rel = max(v5_max_rel, rel)
            for fld in (f"map_{ch}", f"map_{ch}_refined", f"mean_{ch}",
                        f"railed_low_{ch}", f"railed_high_{ch}"):
                v5_scalar_max = max(v5_scalar_max, abs(ps[fld] - rp[fld]))
        v5_n += 1
    v5_pass = v5_n == 200 and v5_max_rel <= 1e-12 and v5_scalar_max <= 1e-12
    if not v5_pass:
        discrepancies.append(
            f"V5: independent R0 reproduction fails (n={v5_n}, max_rel={v5_max_rel:.3e}, scal={v5_scalar_max:.3e})")

    # abort (b)
    r0_nonfinite = sum(
        1 for ps in r0["per_seed"]
        if not (np.all(np.isfinite(ps["ln_post_1d"])) and np.all(np.isfinite(ps["ln_post_2d"])))
    )
    abort_b = nonfinite_total > 0
    if abort_b:
        discrepancies.append(f"abort(b): {nonfinite_total} non-finite ln_post vectors found")

    # ── trigger set ──────────────────────────────────────────────────────────
    a2d_contam = {t: cells["A_h%.3f" % t]["channels"]["2d"]["edge_contaminated"] for t in TRUTHS}
    b2_2d_contam = {t: cells["B2_h%.3f" % t]["channels"]["2d"]["edge_contaminated"] for t in TRUTHS}
    b2_1d_contam = {t: cells["B2_h%.3f" % t]["channels"]["1d"]["edge_contaminated"] for t in TRUTHS}
    a1d_contam = {t: cells["A_h%.3f" % t]["channels"]["1d"]["edge_contaminated"] for t in TRUTHS}
    both_2d_all = all(a2d_contam.values()) and all(b2_2d_contam.values())
    both_2d_any = any(a2d_contam[t] and b2_2d_contam[t] for t in TRUTHS)
    both_1d_all = all(a1d_contam.values()) and all(b2_1d_contam.values())
    both_1d_any = any(a1d_contam[t] and b2_1d_contam[t] for t in TRUTHS)
    triggers = {
        "V1_failure": not v1_pass,
        "V2_failure_observed": False,
        "V3_failure_observed": False,
        "V4_failure": not v4_all,
        "V5_failure": not v5_pass,
        "abort_b": abort_b,
        "both_decision_cells_edge_2d(all-truth reading)": both_2d_all,
        "both_decision_cells_edge_2d(any-truth reading)": both_2d_any,
        "both_decision_cells_edge_1d(all-truth reading)": both_1d_all,
        "both_decision_cells_edge_1d(any-truth reading)": both_1d_any,
    }
    any_trigger = any(v for v in triggers.values() if isinstance(v, bool))
    gate_trustworthy = not any_trigger

    # ── DS-6 ─────────────────────────────────────────────────────────────────
    b2_rlow = {t: cells["B2_h%.3f" % t]["channels"]["1d"]["R_low"] for t in TRUTHS}
    b0_rlow = cells["B0_h0.730"]["channels"]["1d"]["R_low"]
    b2_1d_pass = all(
        cells["B2_h%.3f" % t]["channels"]["1d"]["ds1_status"] == "PASS"
        and cells["B2_h%.3f" % t]["channels"]["1d"]["ds2_status"] == "PASS"
        for t in TRUTHS
    )
    if all(v >= DS6_HI for v in b2_rlow.values()) and b0_rlow <= DS6_LO:
        ds6 = "RAIL-REPRODUCED"
    elif all(v <= DS6_LO for v in b2_rlow.values()) and b2_1d_pass:
        ds6 = "RAIL-NOT-REPRODUCED"
    else:
        ds6 = "MIXED"
    n2_analog = b0_rlow > DS6_LO

    # ── branch tree ──────────────────────────────────────────────────────────
    defect_hits = []
    for t in TRUTHS:
        for cname, ch in (("A", "2d"), ("B2", "1d"), ("B2", "2d")):
            rec = cells["%s_h%.3f" % (cname, t)]["channels"][ch]
            exempt = cells["%s_h%.3f" % (cname, t)]["exempt_flag"]
            a1d_ex = cname == "A" and ch == "1d"
            if exempt or rec["edge_contaminated"] or a1d_ex:
                continue
            if rec["ds1_status"] == "FAIL" or rec["ds2_status"] == "FAIL":
                defect_hits.append(f"{cname}({t})-{ch}")
    kd_a = ds6 == "RAIL-NOT-REPRODUCED"
    kd_b = len(defect_hits) > 0
    rb = (
        gate_trustworthy
        and all(
            cells["A_h%.3f" % t]["channels"]["2d"]["ds1_status"] == "PASS"
            and cells["A_h%.3f" % t]["channels"]["2d"]["ds2_status"] == "PASS"
            and cells["B2_h%.3f" % t]["channels"]["2d"]["ds1_status"] == "PASS"
            and cells["B2_h%.3f" % t]["channels"]["2d"]["ds2_status"] == "PASS"
            for t in TRUTHS
        )
        and ds6 == "RAIL-REPRODUCED"
    )
    if not gate_trustworthy:
        branch = "GATE-NOT-TRUSTWORTHY"
    elif kd_a or kd_b:
        branch = "KEEP-DIGGING"
    elif rb:
        branch = "REPORT-BOUND"
    else:
        branch = "MIXED"
    branch_via = ("clause (b) DEFECT-class" if branch == "KEEP-DIGGING" and kd_b and not kd_a
                  else ("clause (a)" if branch == "KEEP-DIGGING" else None))

    if branch != readout["branch"]["branch"] or branch_via != readout["branch"]["fired_via"]:
        discrepancies.append(
            f"branch mismatch: mine {branch} via {branch_via} vs readout "
            f"{readout['branch']['branch']} via {readout['branch']['fired_via']}")
    if ds6 != readout["ds6"]["verdict"]:
        discrepancies.append(f"DS-6 mismatch: mine {ds6} vs readout {readout['ds6']['verdict']}")
    if len(defect_hits) != len(readout["branch"]["keep_digging_b_defect_class_hits"]):
        discrepancies.append(
            f"defect-hit count mismatch: mine {len(defect_hits)} vs readout "
            f"{len(readout['branch']['keep_digging_b_defect_class_hits'])}")

    # ── DS-8 band arithmetic re-derivation ───────────────────────────────────
    # Jeffreys thresholds
    p0_hi = float(beta_dist.ppf(0.025, 400.5, 0.5))
    thr_hi = p0_hi - 3 * math.sqrt(p0_hi * (1 - p0_hi) / 400)
    p0_lo = float(beta_dist.ppf(0.975, 0.5, 400.5))
    thr_lo = p0_lo + 3 * math.sqrt(p0_lo * (1 - p0_lo) / 400)
    jeffreys = {
        "p0_hi": p0_hi, "thr_hi_exact": thr_hi, "band_hi_registered": 0.98,
        "p0_lo": p0_lo, "thr_lo_exact": thr_lo, "band_lo_registered": 0.02,
        "hi_rounding_looser": 0.98 <= thr_hi, "lo_rounding_looser": 0.02 >= thr_lo,
    }
    if not (abs(p0_hi - 0.9937) < 5e-4 and abs(p0_lo - 0.0063) < 5e-4
            and jeffreys["hi_rounding_looser"] and jeffreys["lo_rounding_looser"]):
        discrepancies.append(f"DS-8 Jeffreys arithmetic does not reproduce: {jeffreys}")

    # T2 bias bands re-derived from v1 committed readout JSON
    v1r = json.load(open(V1_READOUT_JSON))
    key_map = {
        ("B1", 0.730): "B1_h0p730", ("B2", 0.690): "B2_h0p690",
        ("B2", 0.730): "B2_h0p730", ("B2", 0.770): "B2_h0p770",
    }
    t2_bands_check = {}
    t2_ok = True
    for (cname, truth, ch), (plo, phi) in PREREG_T2_BANDS.items():
        v1cell = v1r["cells"][key_map[(cname, truth)]]["channels"][f"channel_{ch}"]["ds3"]
        v1_bias, v1_se = float(v1cell["bias"]), float(v1cell["mc_error"])
        half = 4.0 * math.sqrt(2.0) * v1_se
        lo_x, hi_x = v1_bias - half, v1_bias + half
        # prereg printed band must equal the exact band to 1e-5 rounding
        transcription_ok = abs(lo_x - plo) < 1.1e-5 and abs(hi_x - phi) < 1.1e-5
        mine_bias = cells["%s_h%.3f" % (cname, truth)]["channels"][ch]["bias"]
        inside_exact = lo_x <= mine_bias <= hi_x
        inside_printed = plo <= mine_bias <= phi
        t2_bands_check[f"{cname}({truth})-{ch}"] = {
            "v1_bias": v1_bias, "v1_se": v1_se,
            "band_exact": [lo_x, hi_x], "band_printed": [plo, phi],
            "transcription_ok": transcription_ok,
            "v2_bias_recomputed": mine_bias,
            "inside_exact": inside_exact, "inside_printed": inside_printed,
        }
        if not transcription_ok:
            discrepancies.append(f"DS-8 T2 band transcription off for {cname}({truth})-{ch}: "
                                 f"exact [{lo_x:.6f},{hi_x:.6f}] vs printed [{plo},{phi}]")
        if inside_exact != inside_printed:
            discrepancies.append(f"DS-8 T2 {cname}({truth})-{ch}: rounding flips the verdict")
        t2_ok &= inside_exact and inside_printed
    # C90/rails components
    for cname, truths in (("B1", (0.730,)), ("B2", TRUTHS)):
        for truth in truths:
            for ch in ("1d", "2d"):
                rec = cells["%s_h%.3f" % (cname, truth)]["channels"][ch]
                ok = rec["C"][0.90] <= 0.02 and rec["R_low"] <= 0.02 and rec["R_high"] <= 0.02
                t2_ok &= ok
                if not ok:
                    discrepancies.append(f"DS-8 T2 C90/rails component out of band: {cname}({truth})-{ch}")
    # v1 committed C90/rails really были 0 (band premise)
    for (cname, truth), key in key_map.items():
        for ch in ("1d", "2d"):
            v1ch = v1r["cells"][key]["channels"][f"channel_{ch}"]
            c90v1 = v1ch["ds1"]["values"]["hpd90"]
            rails_v1 = v1ch["ds4"]
            if c90v1 != 0.0:
                discrepancies.append(f"v1 committed C90 not 0 for {cname}({truth})-{ch}: {c90v1}")
            if rails_v1["R_low"] != 0.0 or rails_v1["R_high"] != 0.0:
                discrepancies.append(f"v1 committed rails not 0 for {cname}({truth})-{ch}")

    # T1
    t1 = {}
    t1_ok = True
    for t in TRUTHS:
        d = docs[("A", t)]
        cnt = 0
        rl460 = 0
        for ps in d["per_seed"]:
            ln = np.asarray(ps["ln_post_1d"])
            sub = ln[canon_in_75]
            am = int(np.argmax(sub))
            if abs(float(g41[am]) - 0.600) < 1e-12:
                cnt += 1
            if int(np.argmax(ln)) == 0:
                rl460 += 1
        frac = cnt / len(d["per_seed"])
        t1["h%.3f" % t] = {"restricted_argmax_0.600_frac": frac,
                           "full_grid_R_low_0.460": rl460 / len(d["per_seed"])}
        t1_ok &= frac >= 0.98
    # T3
    b0doc = docs[("B0", 0.730)]
    f1 = np.mean([abs(float(g41[int(np.argmax(ps["ln_post_1d"]))]) - 0.730) < 1e-12
                  for ps in b0doc["per_seed"]])
    f2 = np.mean([abs(float(g41[int(np.argmax(ps["ln_post_2d"]))]) - 0.730) < 1e-12
                  for ps in b0doc["per_seed"]])
    t3_ok = (f1 >= 0.98 and f2 >= 0.98
             and all(cells["B0_h0.730"]["channels"][ch]["R_low"] <= 0.02
                     and cells["B0_h0.730"]["channels"][ch]["R_high"] <= 0.02 for ch in ("1d", "2d")))
    ds8 = {
        "T1": {"verdict": "CONFIRMED" if t1_ok else "REFUTED", "per_truth": t1},
        "T2": {"verdict": "CONFIRMED" if t2_ok else "REFUTED", "bias_components": t2_bands_check},
        "T3": {"verdict": "CONFIRMED" if t3_ok else "REFUTED",
               "map_exact_frac": {"1d": float(f1), "2d": float(f2)}},
        "jeffreys_rederived": jeffreys,
        "void": not gate_trustworthy,
    }
    for tname, mine_v in (("T1_single_host_starvation_rail", ds8["T1"]["verdict"]),
                          ("T2_ball_venue_sigma_z_bias", ds8["T2"]["verdict"]),
                          ("T3_B0_on_truth", ds8["T3"]["verdict"])):
        if readout["ds8"][tname]["verdict"] != mine_v:
            discrepancies.append(f"DS-8 {tname}: mine {mine_v} vs readout {readout['ds8'][tname]['verdict']}")

    # ── DS-7 arithmetic (REPORT-ONLY; zero weight) ──────────────────────────
    ds7_check = {}
    raw_pass_n = 0
    corr_pass_n = 0
    rng = np.random.default_rng(424242)
    for k, d in docs.items():
        name = "%s_h%.3f" % k
        a7 = d["aggregate"]["ds7"]
        mean_np = float(np.mean([ps["n_proposed"] for ps in d["per_seed"]]))
        ratio_mine = 1500.0 / (mean_np * a7["p_bar"])
        corr_mine = ratio_mine * a7["expected_batch_overcount"]
        raw_ok = abs(ratio_mine - 1.0) <= 0.05
        corr_ok = abs(corr_mine - 1.0) <= 0.05
        raw_pass_n += raw_ok
        corr_pass_n += corr_ok
        arith_ok = (abs(mean_np - a7["mean_n_proposed"]) < 1e-6
                    and abs(ratio_mine - a7["ratio"]) < 1e-9
                    and abs(corr_mine - a7["ratio_corrected"]) < 1e-9
                    and raw_ok == a7["pass_raw"] and corr_ok == a7["pass_corrected"]
                    and a7["status"] == "REPORT-ONLY")
        # own MC of the 4096-batch stopping overcount
        nb = 2000
        acc = np.zeros(nb, dtype=np.int64)
        drawn = np.zeros(nb, dtype=np.int64)
        active = np.ones(nb, dtype=bool)
        while active.any():
            add = rng.binomial(4096, a7["p_bar"], size=int(active.sum()))
            acc[active] += add
            drawn[active] += 4096
            active[active] = acc[active] < 1500
        overcount_mc = float(np.mean(drawn) * a7["p_bar"] / 1500.0)
        mc_close = abs(overcount_mc - a7["expected_batch_overcount"]) < 0.01
        ds7_check[name] = {
            "ratio_recomputed": ratio_mine, "corrected_recomputed": corr_mine,
            "pass_raw": raw_ok, "pass_corrected": corr_ok, "arith_consistent": arith_ok,
            "own_mc_overcount": overcount_mc, "instrument_overcount": a7["expected_batch_overcount"],
            "own_mc_within_0.01": mc_close,
        }
        if not arith_ok:
            discrepancies.append(f"{name}: DS-7 arithmetic inconsistent with embedded fields")
        if not mc_close:
            notes.append(f"{name}: own batch-overcount MC {overcount_mc:.4f} vs instrument "
                         f"{a7['expected_batch_overcount']:.4f} (>0.01 apart; MC noise or model diff)")
    if raw_pass_n != 3 or corr_pass_n != 9:
        discrepancies.append(f"DS-7 form counts: raw {raw_pass_n}/9, corrected {corr_pass_n}/9 (readout said 3/9, 9/9)")

    # ── R0 anchor recompute ──────────────────────────────────────────────────
    h41 = g41
    r0rows = {"1d": [], "2d": []}
    for ps in r0["per_seed"]:
        for ch in ("1d", "2d"):
            r0rows[ch].append(own_pp(h41, np.asarray(ps[f"ln_post_{ch}"]), 0.73))
    r0_out = {}
    for ch in ("1d", "2d"):
        rows = r0rows[ch]
        pits = np.asarray([r["pit"] for r in rows])
        maps = np.asarray([r["map"] for r in rows])
        r0_out[ch] = {
            "C": {lv: float(np.mean([r[f"hpd{int(round(lv*100))}"] for r in rows])) for lv in HPD_LEVELS},
            "ks_D": own_ks(pits),
            "bias": float(np.mean(maps)) - 0.73,
            "R_low": float(np.mean([r["railed_low"] for r in rows])),
            "R_high": float(np.mean([r["railed_high"] for r in rows])),
            "edge_loaded_frac": float(np.mean(np.asarray([r["edge_mass"] for r in rows]) > EDGE_MASS_THRESHOLD)),
        }

    # ── NOT-EVALUABLE integrity ──────────────────────────────────────────────
    ne = {
        "ds5_labelled_not_evaluable_all_cells": all(
            "NOT-EVALUABLE" in readout["cells"][n]["channels"][ch]["ds5"]["status"]
            for n in cells for ch in ("1d", "2d")),
        "no_W_statistic_anywhere": "W" not in json.dumps(readout["cells"]),
        "O1_file_absent": not (DIR / "O1_h0p730_results.json").exists()
                          and not any("O1" in p.name for p in DIR.glob("*_results.json")),
        "O1_seed_block_untouched": o1_untouched,
        "leg3_status": readout["stage4_gate_table"]["leg3_forecast_consistent_width"]["status"],
        "leg2_status": readout["stage4_gate_table"]["leg2_generator_closure_count_audit"]["status"],
        "V4_marginal_sigma_quantile_clause": "NOT EVALUATED anywhere (v1 or v2) — registered in v1 §10 "
            "('marginal σ quantiles matching the CSV's within bootstrap noise'), carried 'unchanged' by v2 §10 V4, "
            "but no quantile battery exists in any aggregate and no readout scores it; per-seed records do not "
            "store the drawn σ triples so it is not recomputable post hoc from the JSONs.",
    }
    if not ne["ds5_labelled_not_evaluable_all_cells"]:
        discrepancies.append("DS-5 not labelled NOT-EVALUABLE in some cell")
    if not ne["O1_file_absent"] or not ne["O1_seed_block_untouched"]:
        discrepancies.append("O1 integrity violated")

    # ── output ───────────────────────────────────────────────────────────────
    out = {
        "adjudication": "adjudicate_v2_readout",
        "recomputed_from": "per-seed ln_post vectors, own implementations",
        "per_seed_scalar_max_abs_dev": per_seed_max_dev,
        "hpd_boolean_mismatches": hpd_bool_mismatches,
        "nonfinite_recount": {"registered_cells": nonfinite_total, "R0": r0_nonfinite},
        "provenance": {
            "crb_md5": crb_md5, "crb_md5_ok": crb_md5 == CRB_MD5_EXPECTED,
            "seeds_disjoint": disjoint, "seeds_outside_v1_envelope": outside_v1,
            "o1_block_untouched": o1_untouched, "wall_time_sum_s": wall_sum,
            "per_cell": prov,
        },
        "validity": {
            "V1": {"pass": v1_pass, "exact_1d": v1_1d, "exact_2d": v1_2d},
            "V4": {"pass": v4_all, "medians": v4_meds, "band": V4_BAND},
            "V5_independent": {"pass": v5_pass, "n_seeds_matched": v5_n,
                               "max_rel_ln_post_dev": v5_max_rel,
                               "max_scalar_dev": v5_scalar_max},
            "abort_b": abort_b,
        },
        "cells_recomputed": {
            name: {ch: {kk: (vv if not isinstance(vv, dict) else {str(a): b for a, b in vv.items()})
                        for kk, vv in rec["channels"][ch].items()}
                   for ch in ("1d", "2d")}
            for name, rec in cells.items()
        },
        "R0_recomputed": r0_out,
        "triggers": triggers,
        "gate_trustworthy": gate_trustworthy,
        "ds6": {"verdict": ds6, "R_low_B2_1d": {str(t): b2_rlow[t] for t in TRUTHS},
                "R_low_B0_1d": b0_rlow, "B2_1d_passes_DS1_DS2": b2_1d_pass,
                "impostor_ball_N2_analog": n2_analog,
                "R_low_B1_1d": cells["B1_h0.730"]["channels"]["1d"]["R_low"]},
        "branch": {"branch": branch, "via": branch_via, "defect_hits": defect_hits,
                   "report_bound": rb},
        "ds8": ds8,
        "ds7": {"raw_pass_n": raw_pass_n, "corrected_pass_n": corr_pass_n, "per_cell": ds7_check},
        "not_evaluable_integrity": ne,
        "notes": notes,
        "discrepancies": discrepancies,
    }
    with open(DIR / "adjudicate_v2_readout_results.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("gate_trustworthy:", gate_trustworthy, "| branch:", branch, "via", branch_via)
    print("DS-6:", ds6, "| DS-8: T1", ds8["T1"]["verdict"], "T2", ds8["T2"]["verdict"],
          "T3", ds8["T3"]["verdict"])
    print("defect hits:", len(defect_hits), defect_hits)
    print("V1", v1_pass, "V4", v4_all, "V5(indep)", v5_pass, "| abort_b", abort_b)
    print("per-seed max scalar dev:", per_seed_max_dev, "| hpd bool mismatches:", hpd_bool_mismatches)
    print("DS-7 raw/corr pass:", raw_pass_n, corr_pass_n)
    print("wall sum:", round(wall_sum, 1), "s | crb md5 ok:", crb_md5 == CRB_MD5_EXPECTED)
    print("discrepancies (%d):" % len(discrepancies))
    for dd in discrepancies:
        print("  -", dd)
    print("notes (%d):" % len(notes))
    for nn in notes:
        print("  -", nn)
    print("wrote", DIR / "adjudicate_v2_readout_results.json")


if __name__ == "__main__":
    main()
