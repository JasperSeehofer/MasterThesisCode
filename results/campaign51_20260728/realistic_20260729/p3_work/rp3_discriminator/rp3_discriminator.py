"""[P3-IMP] GATE R-P3 diagnosis: canonical-producer reproduction of banked bsel_seed900101."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from darksiren_emri.validation import correspondence_1d as c1d

BANKED = (
    "results/prod2d_closure_20260818/arm_event_likelihoods/bsel_seed900101/"
    "seed900101/simulations/diagnostics/event_likelihoods.csv"
)


def main() -> int:
    work = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else "CANONICAL"
    out = c1d.run_arm_seed(work / "wr", "bsel", 900101, work / "out")
    print("record:", out)
    csvs = list((work / "wr").rglob("event_likelihoods.csv"))
    fresh = pd.read_csv(csvs[0])
    banked = pd.read_csv(BANKED)
    m = fresh.merge(banked, on=["event_idx", "h"], suffixes=("_f", "_b"))
    print("merged rows", len(m), "fresh rows", len(fresh), "banked rows", len(banked))
    summary = {}
    for col in ["L_cat_no_bh", "B_num", "combined_no_bh"]:
        a = m[col + "_f"].to_numpy()
        b = m[col + "_b"].to_numpy()
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
        n_diff = int((rel > 1e-12).sum())
        print(label, col, "differing rows:", n_diff, "/", len(m), "max rel", float(rel.max()))
        summary[col] = {"n_diff": n_diff, "n": len(m), "max_rel": float(rel.max())}
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
