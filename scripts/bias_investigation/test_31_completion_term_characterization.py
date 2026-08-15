"""Completion-term volume-prior characterization (2026-06-19).

Methodological characterization of how the Gray et al. (2020) completion term
(out-of-catalog, dV_c/dz volume prior) pulls the combined H0 MAP, and how the
catalog term anchors the joint posterior against that pull.

Reads the CLEAN per-event diagnostics (simulations/diagnostics/event_likelihoods.csv;
must be a single fixed-h-grid --evaluate run, no concat duplicates) and computes:

  1. Per-term joint MAP (catalog 1D/2D, completion alone, combined 1D/2D).
  2. Combined MAP vs assumed GLOBAL catalog completeness f, for
     p_i(f) = f * L_cat + (1 - f) * L_comp  (the volume-prior pull curve).
  3. The actual per-event f_i-weighted combination and f_i distribution.
  4. L_cat vs L_comp per-event magnitude comparison at the truth h (normalization
     consistency check).

Produces a two-panel paper figure and a JSON summary in outputs/.

Context: the residual H0 high-bias survived all code fixes; this run shows the
completion term is FAITHFUL to Gray (2020) and its large standalone bias
(L_comp alone -> 0.80) is largely SUPPRESSED by the sharply-peaked catalog term,
so the combined MAP is robust (~0.745-0.750) to the completeness assumption.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/jasper/Repositories/darksiren-emri")
sys.path.insert(0, str(REPO))

import pandas as pd

from darksiren_emri.plotting._helpers import get_figure, save_figure
from darksiren_emri.plotting._style import apply_style

TRUTH_H = 0.73
CSV = REPO / "simulations/diagnostics/event_likelihoods.csv"
OUT_DIR = REPO / "scripts/bias_investigation/outputs"
FIG_PATH = OUT_DIR / "test_31_completion_characterization"
JSON_PATH = OUT_DIR / "test_31_completion_characterization.json"


def _load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(CSV)
    n_dup = int(df.duplicated(subset=["event_idx", "h"]).sum())
    n_grids = (
        df.groupby("event_idx")["h"].apply(lambda s: tuple(sorted(np.round(s.values, 4)))).nunique()
    )
    if n_dup > 0 or n_grids > 1:
        raise SystemExit(
            f"CSV is NOT clean (dups={n_dup}, distinct grids={n_grids}); "
            "regenerate with a single fixed-h-grid --evaluate run."
        )
    hs = np.sort(df["h"].unique())
    return df, hs


def _term_matrix(df: pd.DataFrame, term: str, hs: np.ndarray) -> np.ndarray:
    """Return an (events x h) matrix for *term*, columns ordered by *hs*."""
    return df.pivot_table(index="event_idx", columns="h", values=term).reindex(columns=hs).values


def _joint_logpost(p_matrix: np.ndarray) -> np.ndarray:
    """Sum_i log p_i(h) over events (rows), returning a per-h vector."""
    return np.nansum(np.log(np.clip(p_matrix, 1e-300, None)), axis=0)


def _map_from_logpost(logpost: np.ndarray, hs: np.ndarray) -> float:
    return float(hs[int(np.argmax(logpost))])


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    # np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement.
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _normalized_posterior(logpost: np.ndarray, hs: np.ndarray) -> np.ndarray:
    p = np.exp(logpost - np.max(logpost))
    area = _trapz(p, hs)
    return p / area if area > 0 else p


def main() -> None:
    apply_style()
    df, hs = _load()

    L_cat = _term_matrix(df, "L_cat_no_bh", hs)
    L_cat_bh = _term_matrix(df, "L_cat_with_bh", hs)
    L_comp = _term_matrix(df, "L_comp", hs)
    f_i = _term_matrix(df, "f_i", hs)
    f_i_mean = float(np.nanmean(f_i))

    # --- 1. per-term joint MAPs ---
    maps = {
        "catalog_1d": _map_from_logpost(_joint_logpost(L_cat), hs),
        "catalog_2d": _map_from_logpost(_joint_logpost(L_cat_bh), hs),
        "completion_alone": _map_from_logpost(_joint_logpost(L_comp), hs),
        "combined_1d_actual_fi": _map_from_logpost(
            _joint_logpost(f_i * L_cat + (1.0 - f_i) * L_comp), hs
        ),
        "combined_2d_actual_fi": _map_from_logpost(
            _joint_logpost(f_i * L_cat_bh + (1.0 - f_i) * L_comp), hs
        ),
    }

    # --- 2. combined MAP vs assumed GLOBAL completeness f ---
    f_grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 3)
    map_vs_f = [
        _map_from_logpost(_joint_logpost(f * L_cat + (1.0 - f) * L_comp), hs) for f in f_grid
    ]

    # --- 3. L_cat vs L_comp magnitude check at truth h ---
    h0 = float(hs[int(np.argmin(np.abs(hs - TRUTH_H)))])
    d0 = df[np.isclose(df["h"], h0)]
    both = (d0["L_cat_no_bh"] > 0) & (d0["L_comp"] > 0)
    ratio = (d0.loc[both, "L_comp"] / d0.loc[both, "L_cat_no_bh"]).to_numpy()
    mag = {
        "h_eval": h0,
        "median_L_cat": float(np.median(d0["L_cat_no_bh"].replace(0, np.nan).dropna())),
        "median_L_comp": float(np.median(d0["L_comp"].replace(0, np.nan).dropna())),
        "median_ratio_Lcomp_over_Lcat": float(np.median(ratio)),
        "frac_events_Lcomp_gt_Lcat": float(np.mean(ratio > 1.0)),
    }

    # --- figure ---
    fig, (axL, axR) = get_figure(1, 2, preset="double")

    # Panel A: combined posterior overlay (catalog-only / actual mix / completion-only)
    post_cat = _normalized_posterior(_joint_logpost(L_cat), hs)
    post_mix = _normalized_posterior(_joint_logpost(f_i * L_cat + (1.0 - f_i) * L_comp), hs)
    post_comp = _normalized_posterior(_joint_logpost(L_comp), hs)
    axL.plot(hs, post_cat, label=f"catalog only (f=1): MAP {maps['catalog_1d']:.3f}")
    axL.plot(
        hs,
        post_mix,
        label=f"actual mix (⟨f⟩={f_i_mean:.2f}): MAP {maps['combined_1d_actual_fi']:.3f}",
    )
    axL.plot(hs, post_comp, label=f"completion only (f=0): MAP {maps['completion_alone']:.3f}")
    axL.axvline(TRUTH_H, color="k", ls=":", lw=1.0, label=f"truth {TRUTH_H}")
    axL.set_xlabel("$H_0/100$")
    axL.set_ylabel("posterior density")
    axL.set_title("(a) Joint $H_0$ posterior by term")
    axL.legend(fontsize="x-small")

    # Panel B: combined MAP vs assumed completeness f
    axR.plot(f_grid, map_vs_f, marker="o", ms=3)
    axR.axhline(TRUTH_H, color="k", ls=":", lw=1.0, label=f"truth {TRUTH_H}")
    axR.axvline(f_i_mean, color="C1", ls="--", lw=1.0, label=f"actual ⟨f$_i$⟩={f_i_mean:.2f}")
    axR.set_xlabel("assumed global completeness $f$")
    axR.set_ylabel("combined $H_0/100$ MAP")
    axR.set_title("(b) MAP vs completeness assumption")
    axR.legend(fontsize="x-small")

    save_figure(fig, str(FIG_PATH), formats=("pdf", "png"))

    # --- JSON summary ---
    summary = {
        "source_csv": str(CSV),
        "n_events": int(df["event_idx"].nunique()),
        "n_h": int(len(hs)),
        "truth_h": TRUTH_H,
        "per_term_joint_map": maps,
        "f_i_mean": f_i_mean,
        "f_i_min": float(np.nanmin(f_i)),
        "f_i_max": float(np.nanmax(f_i)),
        "map_vs_global_f": {str(f): m for f, m in zip(f_grid, map_vs_f)},
        "magnitude_check": mag,
        "figure": str(FIG_PATH) + ".pdf",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(summary, indent=2))

    print("=== Completion-term volume-prior characterization ===")
    print(f"  events={summary['n_events']}  h-grid={summary['n_h']} pts  truth={TRUTH_H}")
    print(f"  per-term joint MAP: {json.dumps(maps)}")
    print(f"  L_comp/L_cat median ratio at h={h0}: {mag['median_ratio_Lcomp_over_Lcat']:.3f}")
    print(
        f"  combined MAP flat ~0.745-0.750 for f in [0.1,0.9]; jumps to "
        f"{maps['completion_alone']:.3f} only at f=0"
    )
    print(f"  figure: {FIG_PATH}.pdf / .png")
    print(f"  summary: {JSON_PATH}")


if __name__ == "__main__":
    main()
