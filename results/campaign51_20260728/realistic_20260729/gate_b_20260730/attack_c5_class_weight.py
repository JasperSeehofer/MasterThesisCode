"""Gate B, final C5 discriminator: how much does the headline MAP depend on the
in-catalogue class at all?

attack_c5_leverage.py gave the LINEARISED sensitivity dh*/deps.  This does the
exact nonlinear version: rescale the in-cat class-summed log-likelihood by
lambda in {0, 0.5, 1, 1.5, 2} and read the combined sub-grid MAP.  lambda = 0 is
"throw away every identified host"; if the MAP barely moves, the 76-88 identified
hosts are not carrying the headline and the "crossing of two railed runaways"
framing overstates their role.  The idealized #51 runs are the control.

Also: the 2D-channel pure-leg split for seed61000 real_r1 (the 1D version is in
attack_c5_leverage.py), so the in-cat rail can be compared across channels.

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
LAMBDAS = (0.0, 0.5, 1.0, 1.5, 2.0)


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


def submap(h: np.ndarray, S: np.ndarray) -> float:
    j = int(np.argmax(S))
    if j == 0 or j == len(S) - 1:
        return float(h[j])
    lo, hi = max(0, j - 2), min(len(h), j + 3)
    c = np.polyfit(h[lo:hi], S[lo:hi], 2)
    hv = -c[1] / (2 * c[0])
    return float(hv) if h[lo] <= hv <= h[hi - 1] else float(h[j])


def main() -> None:
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
            r = {"name": f"seed{seed} {tag}"}
            for lam in LAMBDAS:
                r[f"lam{lam}"] = submap(h, lam * S_in + S_dk)
            r["shift_drop_incat"] = r["lam0.0"] - r["lam1.0"]
            r["shift_double_incat"] = r["lam2.0"] - r["lam1.0"]
            r["dark_only_argmax"] = float(h[int(np.argmax(S_dk))])
            r["incat_only_argmax"] = float(h[int(np.argmax(S_in))])
            rows.append(r)

    pd.set_option("display.width", 260, "display.max_columns", 30)
    df = pd.DataFrame(rows)
    print("=== combined 1D MAP with the in-cat class rescaled by lambda ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- 2D pure-leg split for real_r1 -----------------------------------------
    print("\n=== 2D pure-leg split (seed61000 real_r1, diagnostics CSV) ===")
    incat = incat_index(61000)
    ev = pd.read_csv(REAL / "seed61000" / "real_r1" / "diagnostics" / "event_likelihoods.csv")
    ev["is_incat"] = ev.event_idx.isin(incat)
    hs = np.sort(ev.h.unique())
    i73, i86 = int(np.argmin(abs(hs - 0.73))), int(np.argmin(abs(hs - 0.86)))
    legs: dict[str, object] = {"h": hs.tolist()}
    for lab, sel in (("IN-CAT", True), ("DARK", False)):
        sub = ev[ev.is_incat == sel].copy()
        out = {}
        for chan, lcat, lcomb in (
            ("1D", "L_cat_no_bh", "combined_no_bh"),
            ("2D", "L_cat_with_bh", "combined_with_bh"),
        ):
            alive = sub.groupby("event_idx")[lcat].apply(lambda s: bool((s > 0).all()))
            ids = set(alive.index[alive])
            cat = sub[sub.event_idx.isin(ids)]
            pc = (
                cat.assign(x=np.log(cat.w_G * cat[lcat]))
                .groupby("h")
                .x.sum()
                .reindex(hs)
                .to_numpy()
                if ids
                else np.full(len(hs), np.nan)
            )
            pm = sub.assign(x=np.log(sub[lcomb])).groupby("h").x.sum().reindex(hs).to_numpy()
            f = sub.w_G * sub[lcat] / sub[lcomb]
            out[chan] = {
                "n_cat_alive": len(ids),
                "cat_leg_argmax": float(hs[int(np.argmax(pc))]) if ids else float("nan"),
                "cat_leg_rise_073_086": float(pc[i86] - pc[i73]) if ids else float("nan"),
                "mixture_argmax": float(hs[int(np.argmax(pm))]),
                "mixture_rise_073_086": float(pm[i86] - pm[i73]),
                "f_mean_073": float(f[np.isclose(sub.h, 0.73)].mean()),
                "f_mean_086": float(f[np.isclose(sub.h, 0.86)].mean()),
            }
            print(
                f"  {lab:7s} {chan}: cat-leg argmax {out[chan]['cat_leg_argmax']:.3f} "
                f"(rise {out[chan]['cat_leg_rise_073_086']:+9.2f}, n_alive {len(ids):4d}) | "
                f"mixture argmax {out[chan]['mixture_argmax']:.3f} "
                f"(rise {out[chan]['mixture_rise_073_086']:+8.2f}) | "
                f"cat weight f {out[chan]['f_mean_073']:.4f} -> {out[chan]['f_mean_086']:.4f}"
            )
        legs[lab] = out
    # completion leg (channel-common)
    for lab, sel in (("IN-CAT", True), ("DARK", False)):
        sub = ev[ev.is_incat == sel]
        pcomp = (
            sub.assign(x=np.log((1 - sub.w_G) * sub.L_comp))
            .groupby("h")
            .x.sum()
            .reindex(hs)
            .to_numpy()
        )
        print(
            f"  {lab:7s} COMPLETION leg (channel-common): argmax "
            f"{hs[int(np.argmax(pcomp))]:.3f} (rise {pcomp[i86] - pcomp[i73]:+.2f})"
        )

    with open(HERE / "c5_class_weight_results.json", "w") as f:
        json.dump({"lambda_scan": rows, "leg_split_2d_r1": legs}, f, indent=1)
    print(f"\nwrote {HERE / 'c5_class_weight_results.json'}")


if __name__ == "__main__":
    main()
