"""C7 orientation — what does the DELIVERED catalogue leg actually do per event?

Before re-implementing the host-z kernel (c7_kernel_measure.py) we need to know
which leg carries the observed in-catalogue rail (C5, argmax 0.86 for 44/76
hosts).  The diagnostics CSV gives, per event and per h:
    w_G, L_cat_no_bh, B_num, L_comp, combined_no_bh
and combined_no_bh = w_G*L_cat + (1-w_G)*L_comp.

Outputs
  * consistency check diagnostics CSV vs posteriors/h_0_73.json
  * per-class argmax of combined_no_bh, of L_cat_no_bh alone, and of L_comp alone
  * for the 76 in-cat events: how many have their combined argmax at 0.86, and
    what the *catalogue-leg-only* argmax is for those same events.

Read-only.  Run from the repo root with .venv/bin/python.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RUN = HERE.parent / "seed61000" / "real_r1"
CRB = HERE.parent / "seed61000" / "prepared_cramer_rao_bounds.csv"


def main() -> None:
    d = pd.read_csv(RUN / "diagnostics" / "event_likelihoods.csv")
    crb = pd.read_csv(CRB)
    incat = set(crb.index[crb.host_galaxy_index >= 0])

    hs = np.sort(d.h.unique())
    n_h = len(hs)
    piv = {
        col: d.pivot(index="event_idx", columns="h", values=col)[hs]
        for col in ("w_G", "L_cat_no_bh", "L_comp", "combined_no_bh")
    }
    ev = piv["combined_no_bh"].index.to_numpy()

    # --- consistency vs the delivered posterior JSON ---
    with open(RUN / "posteriors" / "h_0_73.json") as f:
        j = json.load(f)
    j73 = np.array([j[str(k)][0] if isinstance(j[str(k)], list) else j[str(k)] for k in ev])
    c73 = piv["combined_no_bh"][0.73].to_numpy()
    rel = np.abs(j73 - c73) / np.maximum(np.abs(j73), 1e-300)
    print(f"diagnostics vs posteriors h=0.73: max rel diff = {rel.max():.3e} over {len(ev)} events")

    # --- mixture identity check ---
    lhs = piv["combined_no_bh"].to_numpy()
    rhs = (
        piv["w_G"].to_numpy() * piv["L_cat_no_bh"].to_numpy()
        + (1 - piv["w_G"].to_numpy()) * piv["L_comp"].to_numpy()
    )
    print(
        f"mixture identity max rel resid = {np.nanmax(np.abs(lhs - rhs) / np.maximum(lhs, 1e-300)):.3e}"
    )

    mask_in = np.isin(ev, sorted(incat))
    print(f"\nevents={len(ev)}  in-cat={mask_in.sum()}  dark={(~mask_in).sum()}  n_h={n_h}")

    def argmax_h(mat: np.ndarray) -> np.ndarray:
        out = np.full(mat.shape[0], np.nan)
        ok = np.all(np.isfinite(mat), axis=1) & (mat.max(axis=1) > 0)
        out[ok] = hs[np.argmax(mat[ok], axis=1)]
        return out

    rows = []
    for label, key in (
        ("combined", "combined_no_bh"),
        ("L_cat only", "L_cat_no_bh"),
        ("L_comp only", "L_comp"),
        ("w_G*L_cat", None),
    ):
        if key is None:
            mat = piv["w_G"].to_numpy() * piv["L_cat_no_bh"].to_numpy()
        else:
            mat = piv[key].to_numpy()
        for cls, m in (("IN-CAT", mask_in), ("DARK", ~mask_in)):
            a = argmax_h(mat[m])
            a = a[np.isfinite(a)]
            if len(a) == 0:
                continue
            rows.append(
                dict(
                    leg=label,
                    cls=cls,
                    n=len(a),
                    median=float(np.median(a)),
                    frac_at_086=float(np.mean(a == hs[-1])),
                    frac_at_060=float(np.mean(a == hs[0])),
                )
            )
    print("\n=== per-event argmax by leg and class (nonzero events only) ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # --- for the in-cat events: pair combined-argmax with L_cat-argmax ---
    a_comb = argmax_h(piv["combined_no_bh"].to_numpy()[mask_in])
    a_cat = argmax_h(piv["L_cat_no_bh"].to_numpy()[mask_in])
    both = np.isfinite(a_comb) & np.isfinite(a_cat)
    print("\n=== in-cat: combined argmax vs catalogue-leg argmax ===")
    print(
        pd.crosstab(
            pd.cut(a_comb[both], [0.59, 0.70, 0.75, 0.80, 0.855, 0.87]),
            pd.cut(a_cat[both], [0.59, 0.70, 0.75, 0.80, 0.855, 0.87]),
            rownames=["combined"],
            colnames=["L_cat"],
        )
    )
    nz = np.isfinite(a_cat)
    print(f"in-cat events with an all-zero L_cat over the whole grid: {mask_in.sum() - nz.sum()}")

    # how much does the catalogue leg dominate the mixture for in-cat events?
    wl = (piv["w_G"].to_numpy() * piv["L_cat_no_bh"].to_numpy())[mask_in]
    cl = ((1 - piv["w_G"].to_numpy()) * piv["L_comp"].to_numpy())[mask_in]
    frac = wl / np.maximum(wl + cl, 1e-300)
    i73 = int(np.argmin(np.abs(hs - 0.73)))
    i86 = n_h - 1
    print(
        "\nin-cat catalogue-leg share of the mixture: "
        f"median @0.73 = {np.median(frac[:, i73]):.3f}, @0.86 = {np.median(frac[:, i86]):.3f}"
    )

    out = {
        "h_grid": hs.tolist(),
        "incat_event_ids": ev[mask_in].tolist(),
        "incat_argmax_combined": a_comb.tolist(),
        "incat_argmax_Lcat": a_cat.tolist(),
    }
    with open(HERE / "c7_orient_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {HERE / 'c7_orient_results.json'}")


if __name__ == "__main__":
    main()
