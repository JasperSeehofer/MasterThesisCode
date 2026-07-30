"""Gate B attack on C5 (RUNBOOK_NEXT_SESSION_6.md §4 item 2).

C5: "54-67% of identified in-catalogue hosts rail at the h=0.86 prior edge, so the
1D headline is a crossing of two railed, opposing runaways, not a centred
measurement."  Replication is established (Gate A3).  This script attacks the two
things that are NOT established.

(a) EDGE-ARTIFACT TEST.  The grid cannot be widened locally, so use the terminal
    behaviour of each railed event's ln p as a surrogate.  For window sizes
    K = 3,5,7,9 the top-K points are fit with a parabola and the vertex located:
        h_peak = h_K* ,  finite iff the fitted curvature is negative.
    Distribution of implied peaks -> finite (where?) vs unbounded runaway.
    C7 predicts finite h_eff in 0.81-0.99.

(b) "NOT A CENTRED MEASUREMENT" TEST.  Three discriminators:
    (i)   jackknife / leave-k-out on the in-cat CLASS-SUMMED profile argmax:
          is the 0.73 -> 0.86 displacement carried by a few events or broad?
    (ii)  per-event Delta ln p between each in-cat event's peak and its value at
          h = 0.73.  If O(0.01 nat) the per-event "rail" is cosmetic and only the
          SUMMED slope is damaging.  Null model: a flat surface + noise gives a
          ~uniform argmax, i.e. 1/41 = 2.4% at the edge, not 58%.
    (iii) class-composition sensitivity: reweight the in-cat and dark class-summed
          profiles by Poisson factors 1 +- 1/sqrt(N_class) and read the combined
          MAP shift (sub-grid, 3-point parabola).  Compare against the measured
          run-to-run MAP sd ~0.006-0.008.

(c) BONUS DISCRIMINATOR (from the C4 decomposition): split each class-summed
    profile into its catalogue leg and its channel-common completion leg
        ln p_i = ln[(1-w_G)L_comp,i] + ln(1 + R_i)
    and locate each leg's argmax.  If the in-cat rail is inherited from the
    completion leg it is NOT a statement about the identified hosts at all.
    (r1/seed61000 only -- needs the diagnostics CSV.)

Read-only w.r.t. master_thesis_code/.  Run from the repo root with .venv/bin/python.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REAL = HERE.parent
CAMPAIGN = REAL.parent
SEEDS = (61000, 62000)
REALIZATIONS = (1, 2, 3, 4, 5)
IDEALIZED = {
    61000: CAMPAIGN / "run_seed61000" / "posteriors_fixed",
    62000: CAMPAIGN / "run_seed62000" / "posteriors",
}
RNG = np.random.default_rng(20260730)


def h_of_tag(tag: str) -> float:
    return float("0." + tag)


def load_grid(post_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    tags = sorted((p.stem.removeprefix("h_0_") for p in post_dir.glob("h_0_*.json")), key=h_of_tag)
    h = np.array([h_of_tag(t) for t in tags])
    per_event: dict[int, list[float]] = {}
    for t in tags:
        with open(post_dir / f"h_0_{t}.json") as f:
            j = json.load(f)
        for k, v in j.items():
            if k.isdigit():
                per_event.setdefault(int(k), []).append(v[0] if isinstance(v, list) else v)
    n = len(h)
    return h, {k: np.asarray(v) for k, v in per_event.items() if len(v) == n}


def incat_index(seed: int) -> set[int]:
    df = pd.read_csv(REAL / f"seed{seed}" / "prepared_cramer_rao_bounds.csv")
    return set(df.index[df.host_galaxy_index >= 0])


def parabola_vertex(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares parabola vertex. Returns (h_peak, curvature a2). a2<0 => max."""
    c = np.polyfit(x, y, 2)
    a2 = float(c[0])
    if a2 == 0.0:
        return float("inf"), 0.0
    return float(-c[1] / (2 * a2)), a2


def analyse_run(name: str, h: np.ndarray, pe: dict[int, np.ndarray], incat: set[int]) -> dict:
    i73 = int(np.argmin(np.abs(h - 0.73)))
    iedge = len(h) - 1
    res: dict[str, object] = {"name": name}

    lnp = {k: np.log(v) for k, v in pe.items() if np.all(v > 0)}
    inc = sorted(k for k in lnp if k in incat)
    drk = sorted(k for k in lnp if k not in incat)

    # ---------- (a) terminal-window parabola extrapolation, railed in-cat events ----
    railed = [k for k in inc if int(np.argmax(lnp[k])) == iedge]
    ext: dict[str, dict] = {}
    for K in (3, 5, 7, 9):
        peaks, curv = [], []
        for k in railed:
            y = lnp[k][-K:]
            x = h[-K:]
            hp, a2 = parabola_vertex(x, y)
            curv.append(a2)
            peaks.append(hp if a2 < 0 and hp > h[iedge] else (np.inf if a2 >= 0 else np.nan))
        p = np.asarray(peaks, dtype=float)
        finite = np.isfinite(p) & ~np.isnan(p)
        ext[f"K{K}"] = {
            "n_railed": len(railed),
            "n_finite_peak": int(finite.sum()),
            "n_runaway_convex": int(np.isinf(p).sum()),
            "n_vertex_below_edge": int(np.isnan(p).sum()),
            "median_peak": float(np.median(p[finite])) if finite.any() else float("nan"),
            "q25_peak": float(np.percentile(p[finite], 25)) if finite.any() else float("nan"),
            "q75_peak": float(np.percentile(p[finite], 75)) if finite.any() else float("nan"),
            "frac_peak_in_081_099": float(np.mean((p[finite] >= 0.81) & (p[finite] <= 0.99)))
            if finite.any()
            else float("nan"),
            "frac_peak_gt_099": float(np.mean(p[finite] > 0.99)) if finite.any() else float("nan"),
            "median_curvature": float(np.median(curv)) if curv else float("nan"),
        }
    res["edge_extrapolation"] = ext

    # ---------- (b)(ii) per-event peak height above h=0.73 --------------------------
    d_peak_73_in = np.array([lnp[k].max() - lnp[k][i73] for k in inc])
    d_peak_73_dk = np.array([lnp[k].max() - lnp[k][i73] for k in drk])
    d_edge_73_in = np.array([lnp[k][iedge] - lnp[k][i73] for k in inc])
    res["per_event_peak_height"] = {
        "incat_median_dlnp_peak_minus_073": float(np.median(d_peak_73_in)),
        "incat_q90_dlnp": float(np.percentile(d_peak_73_in, 90)),
        "incat_max_dlnp": float(d_peak_73_in.max()),
        "incat_frac_dlnp_lt_0p01": float(np.mean(d_peak_73_in < 0.01)),
        "incat_frac_dlnp_lt_0p10": float(np.mean(d_peak_73_in < 0.10)),
        "incat_median_dlnp_edge_minus_073": float(np.median(d_edge_73_in)),
        "dark_median_dlnp_peak_minus_073": float(np.median(d_peak_73_dk)),
        "n_incat": len(inc),
        "n_dark": len(drk),
        "uniform_null_frac_at_edge": 1.0 / len(h),
    }

    # ---------- class-summed profiles ----------------------------------------------
    S_in = np.sum([lnp[k] for k in inc], axis=0)
    S_dk = np.sum([lnp[k] for k in drk], axis=0)
    S_all = S_in + S_dk
    res["sum_in_argmax"] = float(h[int(np.argmax(S_in))])
    res["sum_dark_argmax"] = float(h[int(np.argmax(S_dk))])
    res["sum_all_argmax"] = float(h[int(np.argmax(S_all))])
    res["sum_in_rise_073_to_086"] = float(S_in[iedge] - S_in[i73])
    res["sum_in_rise_per_event"] = float((S_in[iedge] - S_in[i73]) / len(inc))

    # ---------- (b)(i) jackknife on the in-cat summed argmax ------------------------
    loo = []
    for k in inc:
        loo.append(h[int(np.argmax(S_in - lnp[k]))])
    loo = np.asarray(loo)
    res["jackknife_incat"] = {
        "n": len(inc),
        "loo_argmax_unique": sorted(set(map(float, loo))),
        "loo_frac_still_086": float(np.mean(loo == h[-1])),
        # leave-k-out: drop the k events with the largest individual 0.73->0.86 rise
        "drop_top_k": {},
    }
    order = np.argsort(-d_edge_73_in)  # most edge-favouring first
    for kdrop in (1, 2, 3, 5, 8, 12, 20, 30, 40):
        if kdrop >= len(inc):
            continue
        keep = [inc[i] for i in order[kdrop:]]
        Sk = np.sum([lnp[k] for k in keep], axis=0)
        res["jackknife_incat"]["drop_top_k"][str(kdrop)] = {
            "argmax": float(h[int(np.argmax(Sk))]),
            "rise_073_086": float(Sk[iedge] - Sk[i73]),
        }
    # random leave-half-out: how stable is the summed argmax?
    half_arg = []
    for _ in range(200):
        sel = RNG.choice(len(inc), size=len(inc) // 2, replace=False)
        Sh = np.sum([lnp[inc[i]] for i in sel], axis=0)
        half_arg.append(h[int(np.argmax(Sh))])
    half_arg = np.asarray(half_arg)
    res["jackknife_incat"]["random_half_frac_086"] = float(np.mean(half_arg == h[-1]))
    res["jackknife_incat"]["random_half_median_argmax"] = float(np.median(half_arg))

    # ---------- (b)(iii) class-composition (Poisson) reweighting --------------------
    def map_subgrid(prof: np.ndarray) -> float:
        j = int(np.argmax(prof))
        if j == 0 or j == len(prof) - 1:
            return float(h[j])
        y0, y1, y2 = prof[j - 1], prof[j], prof[j + 1]
        den = y0 - 2 * y1 + y2
        if den == 0:
            return float(h[j])
        return float(h[j] - 0.5 * (h[j + 1] - h[j]) * (y2 - y0) / den)

    base = map_subgrid(S_all)
    rw = {}
    a_in = 1.0 / np.sqrt(len(inc))
    a_dk = 1.0 / np.sqrt(len(drk))
    for lab, (fin, fdk) in {
        "incat+1sig": (1 + a_in, 1.0),
        "incat-1sig": (1 - a_in, 1.0),
        "dark+1sig": (1.0, 1 + a_dk),
        "dark-1sig": (1.0, 1 - a_dk),
        "incat+ dark-": (1 + a_in, 1 - a_dk),
        "incat- dark+": (1 - a_in, 1 + a_dk),
    }.items():
        rw[lab] = map_subgrid(fin * S_in + fdk * S_dk)
    res["poisson_reweight"] = {
        "base_map": base,
        "grid_argmax": float(h[int(np.argmax(S_all))]),
        "sigma_incat_frac": float(a_in),
        "sigma_dark_frac": float(a_dk),
        "maps": rw,
        "max_abs_shift": float(max(abs(v - base) for v in rw.values())),
    }

    res["_S_in"] = S_in.tolist()
    res["_S_dk"] = S_dk.tolist()
    res["_h"] = h.tolist()
    return res


def leg_split_r1(incat: set[int]) -> dict:
    """(c) split the in-cat / dark summed 1D profile into catalogue and completion legs."""
    ev = pd.read_csv(REAL / "seed61000" / "real_r1" / "diagnostics" / "event_likelihoods.csv")
    ev["is_incat"] = ev.event_idx.isin(incat)
    ev["C"] = (1 - ev.w_G) * ev.L_comp
    ev["R1"] = ev.w_G * ev.L_cat_no_bh / ev.C
    ev["R2"] = ev.w_G * ev.L_cat_with_bh / ev.C
    hs = np.sort(ev.h.unique())
    out: dict[str, object] = {"h": hs.tolist()}
    for lab, sel in (("INCAT", True), ("DARK", False)):
        sub = ev[ev.is_incat == sel]
        lnC = sub.groupby("h").C.apply(lambda c: float(np.log(c).sum())).reindex(hs).to_numpy()
        t1 = sub.groupby("h").R1.apply(lambda r: float(np.log1p(r).sum())).reindex(hs).to_numpy()
        t2 = sub.groupby("h").R2.apply(lambda r: float(np.log1p(r).sum())).reindex(hs).to_numpy()
        out[lab] = {
            "completion_leg_argmax": float(hs[int(np.argmax(lnC))]),
            "cat_leg_1D_argmax": float(hs[int(np.argmax(t1))]),
            "cat_leg_2D_argmax": float(hs[int(np.argmax(t2))]),
            "mixture_1D_argmax": float(hs[int(np.argmax(lnC + t1))]),
            "mixture_2D_argmax": float(hs[int(np.argmax(lnC + t2))]),
            "completion_rise_073_086": float(
                lnC[np.argmin(abs(hs - 0.86))] - lnC[np.argmin(abs(hs - 0.73))]
            ),
            "cat1D_rise_073_086": float(
                t1[np.argmin(abs(hs - 0.86))] - t1[np.argmin(abs(hs - 0.73))]
            ),
            "cat2D_rise_073_086": float(
                t2[np.argmin(abs(hs - 0.86))] - t2[np.argmin(abs(hs - 0.73))]
            ),
            "_lnC": lnC.tolist(),
            "_t1": t1.tolist(),
            "_t2": t2.tolist(),
        }
    # per-event: how much of each in-cat event's 0.73->0.86 rise is completion?
    lo = ev[np.isclose(ev.h, 0.73)].set_index("event_idx").sort_index()
    hi = ev[np.isclose(ev.h, 0.86)].set_index("event_idx").sort_index()
    dC = np.log(hi.C.to_numpy() / lo.C.to_numpy())
    dt1 = np.log1p(hi.R1.to_numpy()) - np.log1p(lo.R1.to_numpy())
    m = lo.is_incat.to_numpy()
    out["incat_per_event_rise"] = {
        "median_completion": float(np.median(dC[m])),
        "median_catalogue": float(np.median(dt1[m])),
        "sum_completion": float(dC[m].sum()),
        "sum_catalogue": float(dt1[m].sum()),
        "frac_events_completion_dominates": float(np.mean(np.abs(dC[m]) > np.abs(dt1[m]))),
    }
    return out


def main() -> None:
    rows = []
    for seed in SEEDS:
        incat = incat_index(seed)
        h, pe = load_grid(IDEALIZED[seed])
        rows.append(analyse_run(f"seed{seed} IDEAL", h, pe, incat))
        for r in REALIZATIONS:
            h, pe = load_grid(REAL / f"seed{seed}" / f"real_r{r}" / "posteriors")
            rows.append(analyse_run(f"seed{seed} real_r{r}", h, pe, incat))

    pd.set_option("display.width", 250, "display.max_columns", 60)

    print("=== (a) EDGE-ARTIFACT TEST: parabola vertex from the top-K grid points ===")
    print("    railed in-cat events only; 'runaway' = fitted curvature >= 0\n")
    for K in (3, 5, 7, 9):
        print(f"  --- window K = {K} ---")
        t = pd.DataFrame(
            [
                {"name": r["name"], **{k: v for k, v in r["edge_extrapolation"][f"K{K}"].items()}}
                for r in rows
            ]
        )
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print()

    print("=== (b)(ii) per-event peak height: is the rail cosmetic? ===")
    t = pd.DataFrame([{"name": r["name"], **r["per_event_peak_height"]} for r in rows])
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n=== (b)(i) jackknife on the in-cat class-summed argmax ===")
    t = pd.DataFrame(
        [
            {
                "name": r["name"],
                "sum_in_argmax": r["sum_in_argmax"],
                "rise_073_086": r["sum_in_rise_073_to_086"],
                "rise_per_event": r["sum_in_rise_per_event"],
                "loo_frac_086": r["jackknife_incat"]["loo_frac_still_086"],
                "loo_argmax_set": r["jackknife_incat"]["loo_argmax_unique"],
                "half_frac_086": r["jackknife_incat"]["random_half_frac_086"],
                "half_median": r["jackknife_incat"]["random_half_median_argmax"],
            }
            for r in rows
        ]
    )
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n  leave-top-k-out (drop the k most edge-favouring in-cat events) -> argmax")
    t = pd.DataFrame(
        [
            {
                "name": r["name"],
                **{f"drop{k}": v["argmax"] for k, v in r["jackknife_incat"]["drop_top_k"].items()},
            }
            for r in rows
        ]
    )
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== (b)(iii) Poisson class-composition reweighting -> combined MAP ===")
    t = pd.DataFrame(
        [
            {
                "name": r["name"],
                "base_map": r["poisson_reweight"]["base_map"],
                "grid_argmax": r["poisson_reweight"]["grid_argmax"],
                **r["poisson_reweight"]["maps"],
                "max_shift": r["poisson_reweight"]["max_abs_shift"],
            }
            for r in rows
        ]
    )
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== (c) leg split of the summed profiles (seed61000 real_r1) ===")
    ls = leg_split_r1(incat_index(61000))
    for lab in ("INCAT", "DARK"):
        d = ls[lab]
        print(f"  {lab}:")
        for k in (
            "completion_leg_argmax",
            "cat_leg_1D_argmax",
            "cat_leg_2D_argmax",
            "mixture_1D_argmax",
            "mixture_2D_argmax",
            "completion_rise_073_086",
            "cat1D_rise_073_086",
            "cat2D_rise_073_086",
        ):
            print(f"    {k:28s} {d[k]:+.4f}" if "rise" in k else f"    {k:28s} {d[k]:.3f}")
    print(f"  in-cat per-event 0.73->0.86 rise: {ls['incat_per_event_rise']}")

    with open(HERE / "c5_rail_results.json", "w") as f:
        json.dump({"runs": rows, "leg_split_r1": ls}, f, indent=1)
    print(f"\nwrote {HERE / 'c5_rail_results.json'}")


if __name__ == "__main__":
    main()
