"""Gate B, follow-up to attack_c5_rail.py — the two questions that decide whether
"not a centred measurement" is a fair characterisation of the 1D headline.

1. PURE-LEG PROFILES (seed61000 real_r1, diagnostics CSV).
   attack_c5_rail.py used the exact but ratio-flavoured split
       ln p_i = ln[(1-w_G)L_comp,i] + ln(1 + R_i),  R_i = w_G L_cat,i / [(1-w_G)L_comp,i]
   and found the in-cat "rail" is carried by the FIRST term (+33.1 nats over
   0.73->0.86) against the second (-29.2).  Here the two legs are also profiled
   *standalone*:
       Sum_i ln[w_G(h) L_cat,i(h)]        (catalogue leg as its own posterior)
       Sum_i ln[(1-w_G(h)) L_comp,i(h)]   (completion leg as its own posterior)
   plus the actual mixture weight f_i = w_G L_cat / p, so we can say how much
   weight the identified hosts' catalogue information actually carries.
   Also splits the completion leg into its (1-w_G)^N prefactor and Sum ln L_comp.

2. CROSSING LEVERAGE.  The combined MAP h* solves S_in'(h*) + S_dk'(h*) = 0.
   Under a fractional reweight of the in-cat class, S_in -> (1+eps) S_in,
       dh*/deps = -S_in'(h*) / [S_in''(h*) + S_dk''(h*)] .
   A large |S_in'(h*)| means the headline is a cancellation of two large opposing
   slopes -- i.e. a crossing -- and is fragile to class composition.  Reported
   alongside the realisation-to-realisation MAP sd (0.006-0.008) and the
   truth-seed difference (0.023) so the numbers can be compared to something.
   The idealized #51 runs are the control: there the in-cat class is INFORMATIVE
   and the crossing language should NOT apply.

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


def h_of_tag(t: str) -> float:
    return float("0." + t)


def load_grid(post_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    tags = sorted((p.stem.removeprefix("h_0_") for p in post_dir.glob("h_0_*.json")), key=h_of_tag)
    h = np.array([h_of_tag(t) for t in tags])
    pe: dict[int, list[float]] = {}
    for t in tags:
        with open(post_dir / f"h_0_{t}.json") as f:
            j = json.load(f)
        for k, v in j.items():
            if k.isdigit():
                pe.setdefault(int(k), []).append(v[0] if isinstance(v, list) else v)
    return h, {k: np.asarray(v) for k, v in pe.items() if len(v) == len(h)}


def incat_index(seed: int) -> set[int]:
    df = pd.read_csv(REAL / f"seed{seed}" / "prepared_cramer_rao_bounds.csv")
    return set(df.index[df.host_galaxy_index >= 0])


def part1_pure_legs() -> dict:
    incat = incat_index(61000)
    ev = pd.read_csv(REAL / "seed61000" / "real_r1" / "diagnostics" / "event_likelihoods.csv")
    ev["is_incat"] = ev.event_idx.isin(incat)
    hs = np.sort(ev.h.unique())
    n_h = len(hs)
    wg = ev.groupby("h").w_G.first().reindex(hs).to_numpy()

    out: dict[str, object] = {"h": hs.tolist(), "w_G": wg.tolist()}
    print("=== 1. PURE-LEG PROFILES (seed61000 real_r1) ===")
    print(
        f"  w_G: 0.73 -> {wg[np.argmin(abs(hs - 0.73))]:.7f}, "
        f"0.86 -> {wg[np.argmin(abs(hs - 0.86))]:.7f}"
    )
    i73, i86 = int(np.argmin(abs(hs - 0.73))), int(np.argmin(abs(hs - 0.86)))

    for lab, sel in (("IN-CAT", True), ("DARK", False)):
        sub = ev[ev.is_incat == sel].copy()
        n_ev = sub.event_idx.nunique()
        # catalogue leg standalone: only events with L_cat > 0 at ALL h (else -inf)
        alive = sub.groupby("event_idx").L_cat_no_bh.apply(lambda s: bool((s > 0).all()))
        alive_ids = set(alive.index[alive])
        cat = sub[sub.event_idx.isin(alive_ids)]
        prof_cat = (
            cat.assign(x=np.log(cat.w_G * cat.L_cat_no_bh))
            .groupby("h")
            .x.sum()
            .reindex(hs)
            .to_numpy()
        )
        prof_comp = (
            sub.assign(x=np.log((1 - sub.w_G) * sub.L_comp))
            .groupby("h")
            .x.sum()
            .reindex(hs)
            .to_numpy()
        )
        prof_lcomp = sub.assign(x=np.log(sub.L_comp)).groupby("h").x.sum().reindex(hs).to_numpy()
        prof_pref = n_ev * np.log(1 - wg)
        prof_mix = (
            sub.assign(x=np.log(sub.combined_no_bh)).groupby("h").x.sum().reindex(hs).to_numpy()
        )
        # mixture weight of the catalogue leg
        f = sub.w_G * sub.L_cat_no_bh / sub.combined_no_bh
        f73 = f[np.isclose(sub.h, 0.73)]
        f86 = f[np.isclose(sub.h, 0.86)]

        print(f"\n  --- {lab} (n={n_ev}, {len(alive_ids)} with L_cat>0 at all h) ---")
        print(
            f"    argmax  catalogue-leg-alone : {hs[int(np.argmax(prof_cat))]:.3f}"
            f"   (rise 0.73->0.86 = {prof_cat[i86] - prof_cat[i73]:+.2f} nats)"
        )
        print(
            f"    argmax  completion-leg-alone: {hs[int(np.argmax(prof_comp))]:.3f}"
            f"   (rise 0.73->0.86 = {prof_comp[i86] - prof_comp[i73]:+.2f} nats)"
        )
        print(
            f"       of which (1-w_G)^N prefactor: {prof_pref[i86] - prof_pref[i73]:+.2f}, "
            f"Sum ln L_comp: {prof_lcomp[i86] - prof_lcomp[i73]:+.2f}"
        )
        print(
            f"    argmax  MIXTURE (delivered)  : {hs[int(np.argmax(prof_mix))]:.3f}"
            f"   (rise 0.73->0.86 = {prof_mix[i86] - prof_mix[i73]:+.2f} nats)"
        )
        print(
            f"    catalogue mixture weight f = w_G L_cat / p : "
            f"median {f73.median():.4f} @0.73, {f86.median():.4f} @0.86; "
            f"mean {f73.mean():.4f} / {f86.mean():.4f}"
        )
        out[lab] = {
            "n_events": int(n_ev),
            "n_cat_alive": len(alive_ids),
            "cat_leg_argmax": float(hs[int(np.argmax(prof_cat))]),
            "cat_leg_rise": float(prof_cat[i86] - prof_cat[i73]),
            "comp_leg_argmax": float(hs[int(np.argmax(prof_comp))]),
            "comp_leg_rise": float(prof_comp[i86] - prof_comp[i73]),
            "prefactor_rise": float(prof_pref[i86] - prof_pref[i73]),
            "lncomp_rise": float(prof_lcomp[i86] - prof_lcomp[i73]),
            "mixture_argmax": float(hs[int(np.argmax(prof_mix))]),
            "mixture_rise": float(prof_mix[i86] - prof_mix[i73]),
            "f_median_073": float(f73.median()),
            "f_median_086": float(f86.median()),
            "f_mean_073": float(f73.mean()),
            "f_mean_086": float(f86.mean()),
            "_prof_cat": prof_cat.tolist(),
            "_prof_comp": prof_comp.tolist(),
            "_prof_mix": prof_mix.tolist(),
        }
    assert n_h == 41
    return out


def part2_leverage() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        incat = incat_index(seed)
        srcs = [("IDEAL", IDEALIZED[seed])] + [
            (f"real_r{r}", REAL / f"seed{seed}" / f"real_r{r}" / "posteriors") for r in REALIZATIONS
        ]
        for tag, d in srcs:
            h, pe = load_grid(d)
            lnp = {k: np.log(v) for k, v in pe.items() if np.all(v > 0)}
            S_in = np.sum([v for k, v in lnp.items() if k in incat], axis=0)
            S_dk = np.sum([v for k, v in lnp.items() if k not in incat], axis=0)
            S = S_in + S_dk
            j = int(np.argmax(S))
            # sub-grid MAP + local derivatives from a 5-point parabola around the peak
            lo, hi = max(0, j - 2), min(len(h), j + 3)
            c = np.polyfit(h[lo:hi], S[lo:hi], 2)
            hstar = float(-c[1] / (2 * c[0]))
            S2 = float(2 * c[0])  # S''(h*)

            # class slopes at h*, from local 5-point parabolas of each class
            def slope_at(P: np.ndarray, h=h, lo=lo, hi=hi, hstar=hstar) -> float:  # bind loop vars
                cc = np.polyfit(h[lo:hi], P[lo:hi], 2)
                return float(2 * cc[0] * hstar + cc[1])

            s_in, s_dk = slope_at(S_in), slope_at(S_dk)
            sigma = float(np.sqrt(-1.0 / S2)) if S2 < 0 else float("nan")
            rows.append(
                {
                    "name": f"seed{seed} {tag}",
                    "map": hstar,
                    "sigma_h": sigma,
                    "S_in_slope": s_in,
                    "S_dk_slope": s_dk,
                    "net_slope": s_in + s_dk,
                    "S_tot_curv": S2,
                    "dh_deps_incat": -s_in / S2,
                    "dh_1sig_poisson": -s_in / S2 / np.sqrt(sum(1 for k in lnp if k in incat)),
                    "n_incat": sum(1 for k in lnp if k in incat),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    p1 = part1_pure_legs()
    print("\n\n=== 2. CROSSING LEVERAGE ===")
    print("  S_in_slope / S_dk_slope = class log-likelihood slopes at the combined MAP")
    print("  dh_deps_incat = MAP shift per unit FRACTIONAL reweight of the in-cat class")
    print("  dh_1sig_poisson = that shift for a +1 Poisson sigma (1/sqrt(N_incat))\n")
    df = part2_leverage()
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n  reference scales: realisation-to-realisation MAP sd 0.006-0.008;")
    print("                    truth-seed difference 0.023; per-run sigma_h ~0.020")

    with open(HERE / "c5_leverage_results.json", "w") as f:
        json.dump({"pure_legs_r1": p1, "leverage": df.to_dict("records")}, f, indent=1)
    print(f"\nwrote {HERE / 'c5_leverage_results.json'}")


if __name__ == "__main__":
    main()
