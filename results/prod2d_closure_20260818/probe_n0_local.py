#!/usr/bin/env python3
"""N-0/N-1/N-2 local probe for the prod2d closure counterfactual instrument.

Pre-registration: results/prod2d_closure_20260818/
PREREGISTRATION_PROD_COUNTERFACTUAL.md v2, section 1.

Runs the full production ``--evaluate`` machinery for venue iiib at h=0.72
with ``--catalogue_mass_overlap production`` and compares the per-event
``combined_no_bh``/``combined_with_bh`` columns against the banked rows in
``results/run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv``
(max relative diff, reported against the 1e-10 N-0 gate). It then repeats
the run with ``--catalogue_mass_overlap neutralized`` to check:

  * N-1: ``combined_no_bh`` bit-identical to the production run (the mode
    only touches the with-BH-mass mz_integral).
  * N-2: at least 10% of catalogue-supported events (``L_cat_with_bh > 0``
    in the banked CSV) change ``combined_with_bh`` by >= 1e-6 relative.

CLI mirror (P3 byte-diff referent): results/run_20260817_fusion_counterfactual/
off_iiib/run_metadata_19.json (h=0.72, seed=777019) — the SAME seed61000
prepared_cramer_rao_bounds.csv, --normalization_mode absolute_marginal,
--host_z_kernel volume_deconv, --selection_in_completion_numerator off.

KNOWN ENVIRONMENT BLOCKER (flagged prominently, see the implementation
report): the seed61000 injection pool (``simulations/injections/
injection_h_*_task_*.csv``) that ``SimulationDetectionProbability`` requires
lives only in the cluster's /pfs workspace (results/run_20260817_fusion_
counterfactual/off_iiib/simulations/injections is a broken symlink to that
path on this dev machine) and is not checked into git or otherwise available
locally. This script FAILS FAST with a clear message if it cannot find a
usable injection pool -- it does not silently substitute a different pool
and report a fabricated N-0 (a different pool changes the P_det grid, which
would make the N-0 comparison meaningless).

Set ``PROD2D_INJECTION_DIR`` to override the injection pool directory (e.g.
on the cluster, or once the seed61000 pool has been synced locally).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROBE_DIR = Path(__file__).resolve().parent
_WORKDIR = _PROBE_DIR / "probe_n0_workdir"

_BANKED_CSV = (
    _REPO_ROOT
    / "results"
    / "run_20260804_postfix"
    / "iiib"
    / "diagnostics"
    / "event_likelihoods.csv"
)
_CRB_SRC = (
    _REPO_ROOT
    / "results"
    / "run_20260804_postfix"
    / "iiib"
    / "diagnostics"
    / "prepared_cramer_rao_bounds.csv"
)
_PACKAGE_SRC = _REPO_ROOT / "darksiren_emri"

_H_VALUE = "0.72"
_SEED = "777019"  # EVAL_SEED(777000) + SLURM_ARRAY_TASK_ID(19) for h=0.72
_N0_GATE_RTOL = 1e-10
_N2_MIN_RTOL = 1e-6
_N2_MIN_ENGAGEMENT_FRAC = 0.10

# Canonical cluster-side location (cluster/evaluate.sbatch pattern);
# override via PROD2D_INJECTION_DIR when running with real data.
_DEFAULT_INJECTION_DIR_HINT = (
    "the seed61000 injection pool used by run_20260804_postfix / "
    "run_20260817_fusion_counterfactual (see cluster/datasets.yaml + "
    "cluster/evaluate.sbatch INJECTION pass-through); on the cluster this "
    "is $RUN_DIR/simulations/injections for the banked seed61000 run."
)


class ProbeBlockedError(RuntimeError):
    """Raised when the probe cannot execute (missing local data), not a test failure."""


def _resolve_injection_dir() -> Path:
    override = os.environ.get("PROD2D_INJECTION_DIR")
    if override:
        p = Path(override)
        if not p.is_dir():
            raise ProbeBlockedError(f"PROD2D_INJECTION_DIR={p} does not exist.")
        return p
    raise ProbeBlockedError(
        "No injection pool available: SimulationDetectionProbability requires "
        "simulations/injections/injection_h_*_task_*.csv, and the seed61000 "
        "pool is not present on this machine (only broken /pfs symlinks under "
        "results/run_20260817_fusion_counterfactual/*/simulations/injections "
        "exist locally). Set PROD2D_INJECTION_DIR to " + _DEFAULT_INJECTION_DIR_HINT
    )


def _setup_cwd(mode: str, injection_dir: Path) -> Path:
    cwd = _WORKDIR / f"cwd_{mode}"
    sims = cwd / "simulations"
    sims.mkdir(parents=True, exist_ok=True)
    _symlink(sims / "prepared_cramer_rao_bounds.csv", _CRB_SRC)
    # true_cramer_rao_bounds is read in BayesianStatistics.__init__ but never
    # consumed downstream of evaluate() -- the prepared CSV is a harmless
    # stand-in just to satisfy the read.
    _symlink(sims / "cramer_rao_bounds.csv", _CRB_SRC)
    _symlink(cwd / "darksiren_emri", _PACKAGE_SRC)
    _symlink(sims / "injections", injection_dir)
    return cwd


def _symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    link.symlink_to(target)


def _run_evaluate(mode: str, cwd: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "darksiren_emri",
        str(out_dir),
        "--evaluate",
        "--h_value",
        _H_VALUE,
        "--seed",
        _SEED,
        "--pdet_dl_bins",
        "60",
        "--pdet_mass_bins",
        "40",
        "--log_level",
        "INFO",
        "--normalization_mode",
        "absolute_marginal",
        "--host_z_kernel",
        "volume_deconv",
        "--selection_in_completion_numerator",
        "off",
        "--catalogue_mass_overlap",
        mode,
    ]
    print(f"[probe] running mode={mode!r}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-8000:])
        sys.stderr.write(result.stderr[-8000:])
        raise RuntimeError(f"evaluate() subprocess failed for mode={mode!r} (see output above)")
    print(f"[probe] mode={mode!r} evaluate() completed.", flush=True)


def _load_diagnostics(cwd: Path, h_value: float) -> pd.DataFrame:
    csv_path = cwd / "simulations" / "diagnostics" / "event_likelihoods.csv"
    if not csv_path.is_file():
        raise RuntimeError(f"expected diagnostics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df[df["h"] == h_value].sort_values("event_idx").reset_index(drop=True)


def _max_rel_diff(a: pd.Series, b: pd.Series) -> float:
    a_arr = a.to_numpy(dtype=float)
    b_arr = b.to_numpy(dtype=float)
    denom = a_arr.copy()
    denom[denom == 0.0] = 1.0
    rel = abs(a_arr - b_arr) / abs(denom)
    # Rows where both sides are exactly 0.0 contribute 0.0 rel diff.
    both_zero = (a_arr == 0.0) & (b_arr == 0.0)
    rel[both_zero] = 0.0
    return float(rel.max()) if rel.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip the evaluate() subprocess calls and only read/compare diagnostics "
        "CSVs already present under probe_n0_workdir/out_{production,neutralized}.",
    )
    args = parser.parse_args()

    banked = pd.read_csv(_BANKED_CSV)
    banked_h = (
        banked[banked["h"] == float(_H_VALUE)].sort_values("event_idx").reset_index(drop=True)
    )
    if banked_h.empty:
        print(f"ERROR: no banked rows at h={_H_VALUE} in {_BANKED_CSV}", file=sys.stderr)
        return 2

    out_production = _WORKDIR / "out_production"
    out_neutralized = _WORKDIR / "out_neutralized"

    if not args.skip_run:
        try:
            injection_dir = _resolve_injection_dir()
        except ProbeBlockedError as exc:
            print("PROBE BLOCKED (cannot run; not a test failure):", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 3
        cwd_production = _setup_cwd("production", injection_dir)
        cwd_neutralized = _setup_cwd("neutralized", injection_dir)
        _run_evaluate("production", cwd_production, out_production)
        _run_evaluate("neutralized", cwd_neutralized, out_neutralized)
        prod_cwd_for_read = cwd_production
        neut_cwd_for_read = cwd_neutralized
    else:
        prod_cwd_for_read = _WORKDIR / "cwd_production"
        neut_cwd_for_read = _WORKDIR / "cwd_neutralized"

    production = _load_diagnostics(prod_cwd_for_read, float(_H_VALUE))
    neutralized = _load_diagnostics(neut_cwd_for_read, float(_H_VALUE))

    merged_bank_prod = banked_h.merge(
        production, on="event_idx", suffixes=("_banked", "_prod"), how="inner"
    )
    if len(merged_bank_prod) != len(banked_h):
        print(
            f"WARNING: event_idx mismatch banked({len(banked_h)}) vs "
            f"production({len(production)}) -> merged {len(merged_bank_prod)}",
            file=sys.stderr,
        )

    n0_no_bh = _max_rel_diff(
        merged_bank_prod["combined_no_bh_banked"], merged_bank_prod["combined_no_bh_prod"]
    )
    n0_with_bh = _max_rel_diff(
        merged_bank_prod["combined_with_bh_banked"], merged_bank_prod["combined_with_bh_prod"]
    )
    n0_max = max(n0_no_bh, n0_with_bh)

    merged_prod_neut = production.merge(
        neutralized, on="event_idx", suffixes=("_prod", "_neut"), how="inner"
    )
    n1_no_bh_identical = bool(
        (merged_prod_neut["combined_no_bh_prod"] == merged_prod_neut["combined_no_bh_neut"]).all()
    )

    cat_supported = banked_h.merge(
        production, on="event_idx", suffixes=("_banked", ""), how="inner"
    )
    cat_supported_mask = cat_supported["L_cat_with_bh"] > 0.0
    n_cat_supported = int(cat_supported_mask.sum())

    engaged = 0
    if n_cat_supported:
        idx = cat_supported.loc[cat_supported_mask, "event_idx"]
        sub = merged_prod_neut[merged_prod_neut["event_idx"].isin(idx)]
        denom = sub["combined_with_bh_prod"].to_numpy(dtype=float)
        denom_safe = denom.copy()
        denom_safe[denom_safe == 0.0] = 1.0
        rel = abs(
            sub["combined_with_bh_prod"].to_numpy(dtype=float)
            - sub["combined_with_bh_neut"].to_numpy(dtype=float)
        ) / abs(denom_safe)
        engaged = int((rel >= _N2_MIN_RTOL).sum())
    n2_engagement_frac = engaged / n_cat_supported if n_cat_supported else float("nan")

    print("")
    print("=" * 70)
    print("N-0 (production vs banked run_20260804_postfix/iiib, h=0.72)")
    print(f"  max rel diff combined_no_bh:   {n0_no_bh:.3e}")
    print(f"  max rel diff combined_with_bh: {n0_with_bh:.3e}")
    print(f"  max rel diff (overall):        {n0_max:.3e}  (gate: <= {_N0_GATE_RTOL:.0e})")
    print(f"  N-0 VERDICT: {'PASS' if n0_max <= _N0_GATE_RTOL else 'FAIL'}")
    print("")
    print("N-1 (combined_no_bh bit-identical, production vs neutralized)")
    print(f"  N-1 VERDICT: {'PASS' if n1_no_bh_identical else 'FAIL'}")
    print("")
    print(
        "N-2 (engagement: >=10% of catalogue-supported events move combined_with_bh by >=1e-6 rel)"
    )
    print(f"  catalogue-supported events (L_cat_with_bh > 0): {n_cat_supported}")
    print(f"  engaged (>= {_N2_MIN_RTOL:.0e} rel):             {engaged}")
    print(f"  engagement fraction:                            {n2_engagement_frac:.3%}")
    print(f"  N-2 VERDICT: {'PASS' if n2_engagement_frac >= _N2_MIN_ENGAGEMENT_FRAC else 'FAIL'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
