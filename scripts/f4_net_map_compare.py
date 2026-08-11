"""Compare the net H0 MAP between two posterior directories (NW vs LL-v2).

Reads per-h ``h_*.json`` posteriors from two run directories (one evaluated
with ``--pdet_estimator nadaraya_watson``, one with ``local_linear``) and prints
the raw Sigma-log-L MAP for each, for both the 1D (without BH mass) and 2D
(with BH mass) channels.  Raw Sigma-log-L is the canonical paper-grade
combination (used by the bias-investigation suite), independent of the
``physics-floor`` zero-handling strategy.

Usage:
    uv run python scripts/f4_net_map_compare.py <nw_run_dir> <ll_run_dir>

Each <run_dir> is the directory passed to ``--evaluate`` (it contains
``simulations/posteriors`` and ``simulations/posteriors_with_bh_mass``).
"""

import sys
from pathlib import Path

import numpy as np

from darksiren_emri.bayesian_inference.posterior_combination import (
    load_per_h_likelihoods,
)


def _map_for(run_dir: Path, variant: str) -> tuple[float, int, int]:
    """Return (raw Sigma-log-L MAP, n_events, n_h) for one variant dir."""
    d = run_dir / "simulations" / variant
    hs, log_l = load_per_h_likelihoods(d)
    if not hs:
        return float("nan"), 0, 0
    hs_arr = np.asarray(hs, dtype=np.float64)
    sld = np.nansum(log_l, axis=0)
    return float(hs_arr[int(np.argmax(sld))]), log_l.shape[0], len(hs)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    nw_dir, ll_dir = Path(sys.argv[1]), Path(sys.argv[2])

    print("Net H0 MAP (raw Sigma-log-L) -- old seed200+300 CRB through each estimator\n")
    header = f"{'variant':24s} {'NW (local-const)':>18s} {'LL-v2 (local-lin)':>18s}"
    print(header)
    print("-" * len(header))
    for variant, label in [
        ("posteriors", "1D (no BH mass)"),
        ("posteriors_with_bh_mass", "2D (with BH mass)"),
    ]:
        nw_map, nw_n, nw_h = _map_for(nw_dir, variant)
        ll_map, ll_n, ll_h = _map_for(ll_dir, variant)
        print(
            f"{label:24s} {nw_map:>18.4f} {ll_map:>18.4f}   (n_ev~{nw_n}/{ll_n}, n_h={nw_h}/{ll_h})"
        )

    print("\nReading (truth h=0.73):")
    print(
        "  If NW MAP ~0.76 and LL MAP ~0.73  -> F4 boundary bias WAS the H0-bias driver; v2 fixes it."
    )
    print(
        "  If NW MAP ~0.76 and LL MAP ~0.76+ -> v2 fixes accuracy but NOT the H0 bias; cause is elsewhere."
    )
    print(
        "  If NW MAP ~0.73 already           -> the old CRB never showed the bias; premise breaks (re-investigate)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
