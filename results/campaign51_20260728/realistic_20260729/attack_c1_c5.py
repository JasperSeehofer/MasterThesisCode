"""Gate A3 of RUNBOOK_NEXT_SESSION_6.md — provenance repair for claims C1/C5.

Extends the [LOCAL] measurements of `CLAIM_2D_BIAS_20260730.md` from seed61000
real_r1 to ALL 10 realistic runs plus both idealized #51 baselines:

  C1 — per-class nats budget: sum over class of Delta ln p_i from h=0.73 -> 0.81
       (in-cat = `host_galaxy_index >= 0` in the prepared CRB table).
  C5 — per-event 1D argmax over the full prior grid [0.60, 0.86] for the
       in-catalogue hosts: median peak, fraction railed at the 0.86 edge.

Additional edge diagnostics feeding the Gate B refutation of C5 (the grid
cannot be widened above 0.86 while the cluster is down, so the local surrogate
is the terminal behaviour of each railed event's ln p at the top of the grid):

  - terminal slope  s_i = [ln p_i(0.86) - ln p_i(0.85)] / 0.01  (nats/unit-h)
  - terminal 2nd difference over (0.84, 0.85, 0.86): flattening vs still-convex
  - class-summed profiles Sum_class ln p_i(h) over the full grid and their
    argmax — the claim says the 1D headline is a crossing of two railed,
    opposing runaways (in-cat toward 0.86, dark toward the lower edge).

Validation: the r1 rows must reproduce the claim file's numbers exactly
(C1: +2.48 / -11.77 realistic, -338.10 / -23.52 idealized; C5: median 0.860,
44/76 at edge realistic, 0.730, 4/76 idealized) before the extension is
trusted.

Conventions reused verbatim from `score_realistic.py` (same directory):
per-event JSONs `h_0_<tag>.json` keyed by event index, in-cat flag from the
seed-level `prepared_cramer_rao_bounds.csv`. Read-only w.r.t.
master_thesis_code/. Run from the repo root with .venv/bin/python.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CAMPAIGN = HERE.parent
SEEDS = (61000, 62000)
REALIZATIONS = (1, 2, 3, 4, 5)

IDEALIZED = {
    61000: CAMPAIGN / "run_seed61000" / "posteriors_fixed",
    62000: CAMPAIGN / "run_seed62000" / "posteriors",
}


def h_of_tag(tag: str) -> float:
    """Map a file tag like '6', '73', '655' to its h value (0.60, 0.73, 0.655)."""
    return float("0." + tag)


def load_grid(post_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Full per-event likelihood grid: sorted h values, per-event arrays."""
    tags = sorted(
        (p.stem.removeprefix("h_0_") for p in post_dir.glob("h_0_*.json")),
        key=h_of_tag,
    )
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
    df = pd.read_csv(HERE / f"seed{seed}" / "prepared_cramer_rao_bounds.csv")
    return set(df.index[df.host_galaxy_index >= 0])


def analyze(name: str, h: np.ndarray, pe: dict[int, np.ndarray], incat: set[int]) -> dict:
    i73 = int(np.argmin(np.abs(h - 0.73)))
    i81 = int(np.argmin(np.abs(h - 0.81)))
    i_edge = len(h) - 1  # 0.86
    assert abs(h[i73] - 0.73) < 1e-12 and abs(h[i81] - 0.81) < 1e-12
    assert abs(h[i_edge] - 0.86) < 1e-12 and abs(h[i_edge - 1] - 0.85) < 1e-12

    # --- C1: per-class nats budget 0.73 -> 0.81 ---
    d_in = d_dark = 0.0
    n_dropped = 0
    for k, p in pe.items():
        if p[i73] > 0 and p[i81] > 0:
            d = float(np.log(p[i81] / p[i73]))
            if k in incat:
                d_in += d
            else:
                d_dark += d
        else:
            n_dropped += 1

    # --- C5: per-event argmax over the full grid, in-cat hosts ---
    peaks, edge_slopes, edge_2nd = [], [], []
    n_edge_hi = n_edge_lo = 0
    for k in sorted(incat):
        p = pe.get(k)
        if p is None or not np.all(p > 0):
            continue
        lnp = np.log(p)
        j = int(np.argmax(lnp))
        peaks.append(h[j])
        if j == i_edge:
            n_edge_hi += 1
            edge_slopes.append((lnp[i_edge] - lnp[i_edge - 1]) / (h[i_edge] - h[i_edge - 1]))
            edge_2nd.append(lnp[i_edge] - 2 * lnp[i_edge - 1] + lnp[i_edge - 2])
        elif j == 0:
            n_edge_lo += 1

    # --- dark-class argmax distribution (the claimed 0.64-ward rail) ---
    dark_peaks = []
    for k, p in pe.items():
        if k not in incat and np.all(p > 0):
            dark_peaks.append(h[int(np.argmax(np.log(p)))])

    # --- class-summed ln-likelihood profiles over the full grid ---
    sum_in = np.zeros_like(h)
    sum_dark = np.zeros_like(h)
    ok_in = ok_dark = 0
    for k, p in pe.items():
        if not np.all(p > 0):
            continue
        lnp = np.log(p)
        if k in incat:
            sum_in += lnp
            ok_in += 1
        else:
            sum_dark += lnp
            ok_dark += 1

    return {
        "name": name,
        "n_events": len(pe),
        "n_incat": len(peaks),
        "n_dropped_c1": n_dropped,
        "c1_incat": d_in,
        "c1_dark": d_dark,
        "c1_total": d_in + d_dark,
        "c5_median_peak": float(np.median(peaks)) if peaks else float("nan"),
        "c5_n_edge_hi": n_edge_hi,
        "c5_frac_edge_hi": n_edge_hi / len(peaks) if peaks else float("nan"),
        "c5_n_edge_lo": n_edge_lo,
        "edge_slope_median": float(np.median(edge_slopes)) if edge_slopes else float("nan"),
        "edge_slope_min": float(np.min(edge_slopes)) if edge_slopes else float("nan"),
        "edge_2nd_median": float(np.median(edge_2nd)) if edge_2nd else float("nan"),
        "n_edge_still_convex": int(np.sum(np.asarray(edge_2nd) > 0)) if edge_2nd else 0,
        "dark_median_peak": float(np.median(dark_peaks)) if dark_peaks else float("nan"),
        "dark_frac_at_lo": float(np.mean(np.asarray(dark_peaks) == h[0]))
        if dark_peaks
        else float("nan"),
        "dark_frac_le_066": float(np.mean(np.asarray(dark_peaks) <= 0.66))
        if dark_peaks
        else float("nan"),
        "sum_in_argmax": float(h[int(np.argmax(sum_in))]),
        "sum_dark_argmax": float(h[int(np.argmax(sum_dark))]),
        "sum_in_profile": sum_in.tolist(),
        "sum_dark_profile": sum_dark.tolist(),
        "h_grid": h.tolist(),
    }


def main() -> None:
    rows = []
    for seed in SEEDS:
        incat = incat_index(seed)
        h, pe = load_grid(IDEALIZED[seed])
        rows.append(analyze(f"seed{seed} IDEALIZED", h, pe, incat))
        for r in REALIZATIONS:
            h, pe = load_grid(HERE / f"seed{seed}" / f"real_r{r}" / "posteriors")
            rows.append(analyze(f"seed{seed} real_r{r}", h, pe, incat))

    df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows])
    pd.set_option("display.width", 250, "display.max_columns", 40)

    print("=== C1: per-class nats budget, h = 0.73 -> 0.81 ===")
    print(
        df[["name", "n_events", "n_dropped_c1", "c1_incat", "c1_dark", "c1_total"]].to_string(
            index=False, float_format=lambda v: f"{v:+.2f}"
        )
    )

    print("\n=== C5: in-cat per-event argmax over [0.60, 0.86] ===")
    print(
        df[
            [
                "name",
                "n_incat",
                "c5_median_peak",
                "c5_n_edge_hi",
                "c5_frac_edge_hi",
                "c5_n_edge_lo",
                "edge_slope_median",
                "edge_slope_min",
                "edge_2nd_median",
                "n_edge_still_convex",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4g}")
    )

    print("\n=== dark-class argmax + class-summed profile peaks ===")
    print(
        df[
            [
                "name",
                "dark_median_peak",
                "dark_frac_at_lo",
                "dark_frac_le_066",
                "sum_in_argmax",
                "sum_dark_argmax",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4g}")
    )

    out = HERE / "attack_c1_c5_results.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {out} (incl. full class-summed profiles)")


if __name__ == "__main__":
    main()
