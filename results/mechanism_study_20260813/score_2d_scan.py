"""Independent scoring of the 2D dose scan (PREREGISTRATION_2D_DOSE_SCAN.md).

Recomputes every statistic from the raw per-seed records; does not trust any
`aggregate` block in the JSONs. Read-only on all registered .md files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
H_TRUE = 0.730
BASE = 20260808
FRACS = [0.0, 0.25, 0.5, 1.0]

# ---- registered pins (PREREGISTRATION_2D_DOSE_SCAN.md) --------------------
SIGMA_CELL = 0.001579          # §4.0 worst-case per-cell SE at N=15
PER_SEED_SD = 0.0061154        # §4.0 per-seed spread
SE_S23_REG = 0.00061154        # §4.3
DEADBAND_15 = 0.004737         # §4.3 single-cell dead-band, every cell but S23
DEADBAND_S23 = 0.00183462      # §4.3
MID_S23 = 0.0096667
S23_INT = 0.01150132
S23_THR = 0.00783208
S13_INT = 0.0095703
S13_THR = 0.0000963
SIGMA_BAR_MN0 = 0.041813       # §4.6, §5.3
HOST_SHARE = 8.2265e-4         # 982/1193703
K_SUM_PIN = 1193703
N_EVENTS_PIN = 982


def load(name: str) -> dict:
    return json.loads((HERE / f"{name}.json").read_text())


def stats(recs: list[dict], ch: str) -> dict:
    maps = [r[f"map_{ch}"] for r in recs]
    n = len(maps)
    mean = sum(maps) / n
    bias = mean - H_TRUE
    if n > 1:
        var = sum((m - mean) ** 2 for m in maps) / (n - 1)
    else:
        var = 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    psd = sorted(r[f"post_sd_{ch}"] for r in recs)
    med = psd[n // 2] if n % 2 else 0.5 * (psd[n // 2 - 1] + psd[n // 2])
    return {
        "n": n,
        "bias": bias,
        "sd": sd,
        "se": se,
        "post_sd_median": med,
        "rail_low": sum(r[f"railed_low_{ch}"] for r in recs) / n,
        "rail_high": sum(r[f"railed_high_{ch}"] for r in recs) / n,
        "n_distinct_map": len(set(maps)),
    }


def nonfinite_count(recs: list[dict]) -> int:
    bad = 0
    for r in recs:
        for key in ("ln_post_1d", "ln_post_2d"):
            if any(not math.isfinite(v) for v in r[key]):
                bad += 1
                break
    return bad


def main() -> None:
    cells: dict[str, dict] = {}
    for h in range(4):
        for i in range(4):
            name = f"S{h}{i}"
            n = 100 if name == "S23" else 15
            d = load(f"{name}_h0p730_results_seeds0_{n}")
            recs = d["per_seed"]
            exp_seeds = [BASE + 51000 + 100 * (4 * h + i) + j for j in range(n)]
            got_seeds = [r["seed"] for r in recs]
            sbar_pred = SIGMA_BAR_MN0 * (FRACS[h] * HOST_SHARE + FRACS[i] * (1 - HOST_SHARE))
            sbar_meas = sum(r["sigma_z_mean_pairs"] for r in recs) / len(recs)
            cells[name] = {
                "f_h": FRACS[h],
                "f_i": FRACS[i],
                "N": len(recs),
                "N_registered": n,
                "seeds_ok": got_seeds == exp_seeds,
                "seed_first": got_seeds[0],
                "seed_last": got_seeds[-1],
                "dose_scales": d["config"]["dose_scales"],
                "dose_ok": list(d["config"]["dose_scales"]) == [FRACS[h], FRACS[i]],
                "pin_pass": d["pin_integrity"]["pass"],
                "k_sum_ok": all(r["K_sum"] == K_SUM_PIN for r in recs),
                "n_events_ok": all(
                    r["n_events"] == N_EVENTS_PIN and r["n_events_run"] == N_EVENTS_PIN
                    for r in recs
                ),
                "horizon_drop_max": max(r["n_horizon_dropped"] for r in recs),
                "nonfinite": nonfinite_count(recs),
                "sbar_pred": sbar_pred,
                "sbar_meas": sbar_meas,
                "sbar_relerr": abs(sbar_meas - sbar_pred) / sbar_pred if sbar_pred else float("nan"),
                "git_dirty": d["git_dirty"],
                "import_path_clean": d["import_path_clean"],
                "git_commit": d["git_commit"],
                "1d": stats(recs, "1d"),
                "2d": stats(recs, "2d"),
                "wall_s": d["wall_time_s"],
                "workers": d["workers"],
            }
    # corner cross-checks (parent arms, different seeds)
    corners = {}
    for nm, fn in (("MN0", "MN0_h0p730_results_seeds0_15"),
                   ("MEH", "MEH_h0p730_results_seeds0_15"),
                   ("MEI", "MEI_h0p730_results_seeds0_15"),
                   ("MN0X", "MN0X_h0p730_results_seeds0_100")):
        d = load(fn)
        corners[nm] = {
            "N": len(d["per_seed"]),
            "1d": stats(d["per_seed"], "1d"),
            "2d": stats(d["per_seed"], "2d"),
            "sbar": sum(r["sigma_z_mean_pairs"] for r in d["per_seed"]) / len(d["per_seed"]),
            "seeds": (d["seeds"][0], d["seeds"][-1]),
        }

    out = {"cells": cells, "corners": corners}
    (HERE / "score_2d_scan_output.json").write_text(json.dumps(out, indent=1))

    # ---------------- report ----------------
    P = print
    P("=== DS-D1 SURFACE (1D headline) ===")
    P(f"{'cell':5} {'f_h':>5} {'f_i':>5} {'N':>4} {'bias':>11} {'SE':>10} {'sd':>10} "
      f"{'post_sd_med':>12} {'rails':>6} {'nonfin':>6} {'sbar_meas':>11} {'sbar_pred':>11} {'rel%':>7} {'seeds':>6} {'#MAP':>5}")
    for name, c in cells.items():
        s = c["1d"]
        P(f"{name:5} {c['f_h']:5.2f} {c['f_i']:5.2f} {c['N']:4d} {s['bias']:+11.6f} {s['se']:10.6f} "
          f"{s['sd']:10.6f} {s['post_sd_median']:12.6f} "
          f"{s['rail_low']+s['rail_high']:6.3f} {c['nonfinite']:6d} "
          f"{c['sbar_meas']:11.6f} {c['sbar_pred']:11.6f} {100*c['sbar_relerr']:7.3f} "
          f"{'OK' if c['seeds_ok'] and c['dose_ok'] else 'BAD':>6} {s['n_distinct_map']:5d}")
    P()
    P("=== DS-D1 SURFACE (2D) ===")
    for name, c in cells.items():
        s = c["2d"]
        P(f"{name:5} bias {s['bias']:+.6f} SE {s['se']:.6f} post_sd_med {s['post_sd_median']:.6f} "
          f"rails {s['rail_low']+s['rail_high']:.3f}  (1D-2D = {c['1d']['bias']-s['bias']:+.6f})")
    P()
    P("=== VALIDITY ===")
    for name, c in cells.items():
        P(f"{name}: pin {c['pin_pass']} K_sum {c['k_sum_ok']} n_events {c['n_events_ok']} "
          f"horizon_drop_max {c['horizon_drop_max']} nonfinite {c['nonfinite']} "
          f"seeds {c['seed_first']}-{c['seed_last']} ok={c['seeds_ok']} "
          f"dose {c['dose_scales']} ok={c['dose_ok']} import_clean={c['import_path_clean']} commit={c['git_commit'][:8]}")
    P()
    P("=== CORNERS (parent arms, different seeds) ===")
    for nm, c in corners.items():
        P(f"{nm}: N={c['N']} 1D bias {c['1d']['bias']:+.6f} SE {c['1d']['se']:.6f} | "
          f"2D bias {c['2d']['bias']:+.6f} SE {c['2d']['se']:.6f} | sbar {c['sbar']:.6f} seeds {c['seeds']}")
    P()

    b = {k: v["1d"]["bias"] for k, v in cells.items()}
    se = {k: v["1d"]["se"] for k, v in cells.items()}

    P("=== DS-D4: f_host=0 row ===")
    for name in ("S00", "S01", "S02", "S03"):
        s = cells[name]["1d"]
        P(f"  {name}: bias {s['bias']:+.9e} sd {s['sd']:.9e} distinct MAPs {s['n_distinct_map']} "
          f"post_sd_med {s['post_sd_median']:.9e}")
    P(f"  S00 exactly zero (|bias| < 1e-12): {abs(b['S00']) < 1e-12}  -> SCAN-CONFOUNDED trigger: "
      f"{'FIRES' if abs(b['S00']) >= 1e-12 else 'does NOT fire'}")
    row0_zero = all(abs(b[n]) < 1e-12 and cells[n]['1d']['sd'] < 1e-12 for n in ('S00','S01','S02','S03'))
    P(f"  PIN-BINARY (all four exactly zero, zero spread): {row0_zero}")
    P()

    P("=== §5.2 CORNER CROSS-CHECKS ===")
    def cc(cell, ref_name, ref_val, ref_se):
        s = se[cell]
        tol = 3 * math.sqrt(s ** 2 + ref_se ** 2)
        d = b[cell] - ref_val
        P(f"  {cell} vs {ref_name}: {b[cell]:+.6f} - {ref_val:+.6f} = {d:+.6f}; "
          f"tol 3*sqrt({s:.6f}^2+{ref_se:.6f}^2) = {tol:.7f}; "
          f"{'PASS' if abs(d) <= tol else 'FAILED'} ({abs(d)/tol if tol else float('nan'):.2f}x tol)")
    cc("S33", "MN0", 0.034667, 0.001579)
    cc("S30", "MEH", 0.004000, 0.000535)
    P(f"  S03 vs MEI: exact equality required. S03 bias {b['S03']:+.9e}, sd {cells['S03']['1d']['sd']:.3e} -> "
      f"{'PASS (exact)' if abs(b['S03'])<1e-12 else 'CROSS-CHECK-DISCREPANT'}")
    P(f"  S00 sigma=0 anchor: {b['S00']:+.9e} -> {'PASS (exact)' if abs(b['S00'])<1e-12 else 'SCAN-CONFOUNDED'}")
    P()

    P("=== DS-D2: additivity residual D = b(h,i) - b(h,0) - b(0,i) + b(0,0) ===")
    for h in range(1, 4):
        for i in range(1, 4):
            n = f"S{h}{i}"; a = f"S{h}0"; c_ = f"S0{i}"
            D = b[n] - b[a] - b[c_] + b["S00"]
            SE_D = math.sqrt(se[n]**2 + se[a]**2 + se[c_]**2 + se["S00"]**2)
            lab = "NON-ADDITIVE" if abs(D) >= 3*SE_D else ("ADDITIVE-CONSISTENT" if abs(D) < 2*SE_D else "AMBIGUOUS")
            P(f"  {n}: D = {b[n]:+.6f} - {b[a]:+.6f} - {b[c_]:+.6f} + {b['S00']:+.6f} = {D:+.6f}; "
              f"SE_D = {SE_D:.6f}; |D|/SE_D = {abs(D)/SE_D if SE_D else float('nan'):6.2f}  {lab}")
    P()

    P("=== DS-D3: shape discrimination ===")
    P(f"  S23 (N={cells['S23']['N']}): b = {b['S23']:+.8f}, realized SE = {se['S23']:.8f} "
      f"(registered SE {SE_S23_REG:.8f})")
    P(f"    SHAPE-INTERACTION iff b >= {S23_INT}; SHAPE-THRESHOLD iff b <= {S23_THR}")
    v = b["S23"]
    shape = "SHAPE-INTERACTION" if v >= S23_INT else ("SHAPE-THRESHOLD" if v <= S23_THR else "SHAPE-UNDECIDED")
    P(f"    -> {shape}")
    P(f"    distance to upper boundary: {v - S23_INT:+.8f} = {(v-S23_INT)/se['S23']:+.2f} realized SE")
    P(f"    distance to lower boundary: {v - S23_THR:+.8f} = {(v-S23_THR)/se['S23']:+.2f} realized SE")
    P(f"    vs H-INT prediction 0.017333: {v-0.017333:+.8f} = {(v-0.017333)/se['S23']:+.2f} SE")
    P(f"    vs H-THRESH prediction 0.002000: {v-0.002000:+.8f} = {(v-0.002000)/se['S23']:+.2f} SE")
    P(f"  S13 (secondary, N=15): b = {b['S13']:+.6f}, SE {se['S13']:.6f}")
    P(f"    SHAPE-INTERACTION iff b >= {S13_INT}; SHAPE-THRESHOLD iff b <= {S13_THR}")
    v13 = b["S13"]
    sh13 = "SHAPE-INTERACTION" if v13 >= S13_INT else ("SHAPE-THRESHOLD" if v13 <= S13_THR else "SHAPE-UNDECIDED")
    P(f"    -> {sh13}")
    P()

    P("=== DS-D5: linearity along f_host = 1 ===")
    P(f"  registered line: (0, 0.004000) -> (1, 0.034667); departure edge +-{DEADBAND_15}")
    for i, name in ((0.25, "S31"), (0.5, "S32")):
        pred = 0.004000 + (0.034667 - 0.004000) * i
        d = b[name] - pred
        lab = "SUPER-LINEAR" if d >= DEADBAND_15 else ("SUB-LINEAR" if d <= -DEADBAND_15 else "LINEAR-CONSISTENT")
        P(f"  {name} (f_i={i}): measured {b[name]:+.6f}, predicted {pred:+.6f}, "
          f"delta {d:+.6f} = {d/se[name]:+.2f} realized SE -> {lab}")
    P(f"  endpoints as measured on THIS scan: S30 {b['S30']:+.6f}, S33 {b['S33']:+.6f}")
    for i, name in ((0.25, "S31"), (0.5, "S32")):
        pred = b["S30"] + (b["S33"] - b["S30"]) * i
        d = b[name] - pred
        P(f"    self-anchored line: {name} pred {pred:+.6f}, delta {d:+.6f} "
          f"= {d/se[name]:+.2f} cell SE")
    P()

    P("=== DS-D6: R_dose = bias / (f_i * 0.041813) ===")
    for name, c in cells.items():
        if c["f_i"] > 0:
            den = c["f_i"] * SIGMA_BAR_MN0
            band = "  [banded 0.75-1.25, S33 only]" if name == "S33" else "  (UNBANDED)"
            P(f"  {name}: {b[name]:+.6f} / ({c['f_i']} * {SIGMA_BAR_MN0}) = {b[name]/den:+.4f}{band}")
    P()

    P("=== SHAPE DEPARTURES (§4.0 resolution classes) ===")
    def diff(a, c_):
        d = b[a] - b[c_]
        s = math.sqrt(se[a]**2 + se[c_]**2)
        cls = "RESOLVED" if abs(d) >= 3*s else ("MARGINAL" if abs(d) >= 2*s else "UNRESOLVED")
        P(f"  {a} - {c_} = {b[a]:+.6f} - {b[c_]:+.6f} = {d:+.6f}; SE_diff {s:.6f}; "
          f"{abs(d)/s if s else float('nan'):5.2f} sigma -> {cls}")
    P(" row f_host=1.0 successive steps:")
    diff("S31", "S30"); diff("S32", "S31"); diff("S33", "S32")
    P(" row f_host=0.5 successive steps:")
    diff("S21", "S20"); diff("S22", "S21"); diff("S23", "S22")
    P(" row f_host=0.25 successive steps:")
    diff("S11", "S10"); diff("S12", "S11"); diff("S13", "S12")
    P(" column f_imp=1.0 successive steps:")
    diff("S13", "S03"); diff("S23", "S13"); diff("S33", "S23")
    P(" column f_imp=0 (pp_coverage analogue):")
    diff("S10", "S00"); diff("S20", "S10"); diff("S30", "S20")
    P()
    P(" curvature/second differences along f_host=1 (equal spacing 0.25 for first two, then 0.5):")
    P("  slope 0->0.25 per unit f_i: %.6f" % ((b['S31']-b['S30'])/0.25))
    P("  slope 0.25->0.5:            %.6f" % ((b['S32']-b['S31'])/0.25))
    P("  slope 0.5->1.0:             %.6f" % ((b['S33']-b['S32'])/0.5))
    P(" row f_host=0.5:")
    P("  slope 0->0.25: %.6f" % ((b['S21']-b['S20'])/0.25))
    P("  slope 0.25->0.5: %.6f" % ((b['S22']-b['S21'])/0.25))
    P("  slope 0.5->1.0: %.6f" % ((b['S23']-b['S22'])/0.5))
    P()
    P("=== COST (abort criterion e: >2x 0.969 CPU-h/seed) ===")
    for name, c in cells.items():
        cpuh = c["wall_s"] * c["workers"] / 3600.0 / c["N"]
        P(f"  {name}: wall {c['wall_s']:8.1f}s x {c['workers']} workers / {c['N']} seeds = {cpuh:.4f} CPU-h/seed"
          + ("  >2x ANCHOR" if cpuh > 2*0.969 else ""))




def analysis() -> None:
    """Second pass: bilinearity residuals, slope tests, A1 coupling, determinism."""
    out = json.loads((HERE / "score_2d_scan_output.json").read_text())
    cells = out["cells"]
    b = {k: v["1d"]["bias"] for k, v in cells.items()}
    se = {k: v["1d"]["se"] for k, v in cells.items()}
    P = print

    P("\n\n=== A1 COUPLING (branch 1 leg: Amendment A1 A1-FAIL) ===")
    x = out["corners"]["MN0X"]
    P(f"  MN0X 1D mean bias = {x['1d']['bias']:+.6f} (N={x['N']}, realized SE {x['1d']['se']:.6f})")
    P(f"  |{x['1d']['bias']:.6f} - 0.037237| = {abs(x['1d']['bias']-0.037237):.6f}  vs window 0.002")
    P(f"  -> {'A1-PASS' if abs(x['1d']['bias']-0.037237) <= 0.002 else 'A1-FAIL'}")
    P(f"  registered point prediction +0.03685 +- 0.00056: measured is "
      f"{(x['1d']['bias']-0.0368515)/0.000564:+.2f} sigma from it")
    P(f"  2D reported alongside: {x['2d']['bias']:+.6f}")

    P("\n=== A1-DET / determinism: MN0 15 seeds vs MN0X first 15 ===")
    mn0 = load("MN0_h0p730_results_seeds0_15")["per_seed"]
    mn0x = load("MN0X_h0p730_results_seeds0_100")["per_seed"]
    idx = {r["seed"]: r for r in mn0x}
    worst = 0.0
    map_eq = True
    for r in mn0:
        q = idx.get(r["seed"])
        if q is None:
            P(f"  seed {r['seed']} MISSING from MN0X"); continue
        for k, v in r.items():
            if isinstance(v, float) and k in q and isinstance(q[k], float) and v:
                worst = max(worst, abs(v - q[k]) / abs(v))
        map_eq &= (r["map_1d"] == q["map_1d"] and r["map_2d"] == q["map_2d"])
    P(f"  max relative deviation on shared scalar fields = {worst:.3e} (rtol gate 1e-12); "
      f"MAPs exactly equal: {map_eq}")

    P("\n=== H-INT BILINEARITY RESIDUAL: D_meas - I*f_h*f_i ===")
    for I, tag in ((0.030667, "registered anchor I = D(1,1)_parent = 0.030667"),
                   (b["S33"] - b["S30"] - b["S03"] + b["S00"], "self-anchored I = D(1,1)_this scan")):
        P(f"  [{tag} = {I:.6f}]")
        for h in range(1, 4):
            for i in range(1, 4):
                n = f"S{h}{i}"; a = f"S{h}0"; c_ = f"S0{i}"
                D = b[n] - b[a] - b[c_] + b["S00"]
                SE_D = math.sqrt(se[n]**2 + se[a]**2 + se[c_]**2 + se["S00"]**2)
                prod = FRACS[h] * FRACS[i]
                pred = I * prod
                r = D - pred
                flag = " >3sigma" if abs(r) >= 3 * SE_D else ""
                ne = "  [NOT-EVALUABLE §6 item 1]" if n in ("S11", "S12", "S21") else ""
                P(f"    {n} (f_h*f_i={prod:6.4f}): D={D:+.6f} pred={pred:+.6f} "
                  f"resid={r:+.6f} = {r/SE_D:+6.2f} SE_D{flag}{ne}")

    P("\n=== SLOPE / CURVATURE TESTS ALONG f_host = 1 (both hypotheses predict a straight line) ===")
    def step(a, c_):
        return b[a] - b[c_], math.sqrt(se[a]**2 + se[c_]**2)
    d1, s1 = step("S31", "S30")   # over df_i = 0.25
    d2, s2 = step("S32", "S31")   # over 0.25
    d3, s3 = step("S33", "S32")   # over 0.50
    P(f"  step1 (f_i 0->0.25)   = {d1:+.6f} +- {s1:.6f}")
    P(f"  step2 (f_i 0.25->0.5) = {d2:+.6f} +- {s2:.6f}")
    P(f"  step3 (f_i 0.5->1.0)  = {d3:+.6f} +- {s3:.6f}")
    se_sd = math.sqrt(se["S30"]**2 + 4*se["S31"]**2 + se["S32"]**2)
    P(f"  second difference (step2 - step1) = {d2-d1:+.6f}; SE = sqrt(se30^2+4*se31^2+se32^2) "
      f"= {se_sd:.6f}; {(d2-d1)/se_sd:+.2f} sigma")
    m1, sm1 = d1/0.25, s1/0.25
    m2, sm2 = d2/0.25, s2/0.25
    m3, sm3 = d3/0.50, s3/0.50
    P(f"  per-unit slopes: m1={m1:.6f}+-{sm1:.6f}  m2={m2:.6f}+-{sm2:.6f}  m3={m3:.6f}+-{sm3:.6f}")
    P(f"  m2-m1 = {m2-m1:+.6f} +- {math.sqrt(sm1**2+sm2**2):.6f} = "
      f"{(m2-m1)/math.sqrt(sm1**2+sm2**2):+.2f} sigma")
    P(f"  m3-m2 = {m3-m2:+.6f} +- {math.sqrt(sm2**2+sm3**2):.6f} = "
      f"{(m3-m2)/math.sqrt(sm2**2+sm3**2):+.2f} sigma")

    P("\n=== THE f_host = 0.5 DIP (and its f_host = 0.25 counterpart) ===")
    for a, c_ in (("S22", "S21"), ("S12", "S11")):
        d, s = step(a, c_)
        cls = "RESOLVED" if abs(d) >= 3*s else ("MARGINAL" if abs(d) >= 2*s else "UNRESOLVED")
        P(f"  {a} - {c_} = {d:+.6f} +- {s:.6f} = {d/s:+.2f} sigma -> {cls}")
    dsum = (b["S22"]-b["S21"]) + (b["S12"]-b["S11"])
    ssum = math.sqrt(se["S22"]**2+se["S21"]**2+se["S12"]**2+se["S11"]**2)
    P(f"  POST-HOC (unregistered) pooled dip = {dsum:+.6f} +- {ssum:.6f} = {dsum/ssum:+.2f} sigma "
      f"-> {'>=3 sigma' if abs(dsum)>=3*ssum else 'BELOW 3 sigma'}")

    P("\n=== IMPOSTOR-DIRECTION SATURATION (interaction residual normalised on its own row) ===")
    for h in range(1, 4):
        Dfull = b[f"S{h}3"] - b[f"S{h}0"]
        for i in (1, 2, 3):
            D = b[f"S{h}{i}"] - b[f"S{h}0"]
            P(f"  f_h={FRACS[h]}: [b(f_i={FRACS[i]}) - b(f_i=0)] / [b(f_i=1) - b(f_i=0)] = "
              f"{D:+.6f}/{Dfull:+.6f} = {D/Dfull:6.3f}")

    P("\n=== HOST-DIRECTION at f_imp = 1 (column) ===")
    for h in range(4):
        P(f"  f_h={FRACS[h]}: b = {b[f'S{h}3']:+.6f} +- {se[f'S{h}3']:.6f}")

    P("\n=== §4.0 RESOLUTION FLOOR: registered vs realized ===")
    ses = sorted(se.values())
    P(f"  registered worst-case SE_diff 0.0022330 -> 3sigma floor 0.0066990 -> "
      f"0.034667/0.0066990 = {0.034667/0.0066990:.1f} distinguishable levels (REGISTERED, §4.0/§6 item 8)")
    worst_se = max(se.values())
    real_floor = 3*math.sqrt(2)*worst_se
    P(f"  realized worst per-cell SE = {worst_se:.6f} -> worst realized 3sigma floor {real_floor:.7f} "
      f"-> full range {max(b.values()):.6f}/{real_floor:.7f} = {max(b.values())/real_floor:.1f} levels "
      f"(DISCLOSURE ONLY; the registered bar is not relaxed)")


if __name__ == "__main__":
    main()
    analysis()
