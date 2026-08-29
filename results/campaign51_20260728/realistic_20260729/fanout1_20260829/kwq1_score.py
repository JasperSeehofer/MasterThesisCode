#!/usr/bin/env python3
"""B4.2 "KW-Q1" post-processor -- kernel-width discriminator on the low-z quartile.

Registration: ``CLAIM_IMPOSTOR_DRAG_20260829.md`` sec 1.3 (the named 4.2 read),
sequenced behind B1.1's theta-driver (``hier_s0_driver.py``) per
``SYNTHESIS_DOCKET_1_20260829.md`` sec 2 B4. **BUILD ONLY -- charter node
B4.2, launched under rows #222/#223.** Rule 2 (verifier independence): this
script's author (also ``hier_s0_driver.py``'s ``--config ft`` author) may
only SMOKE-TEST it; the registered KW-Q1 measurement (4 seeds x 3 theta
nodes x 2 h at the FT config, prereg cost 8.4-13.7 CPU-h) must be RUN by a
different agent, and per the claim card's own Rule-2 clause "B1.1's driver
author may not be the KW-Q1 runner either" -- three distinct agents across
build/run/score.

**This script does NO evaluation** -- it reads ``event_likelihoods.csv``
files that ``hier_s0_driver.py --config ft`` already wrote to disk (one CSV
per (seed, theta node), each carrying BOTH h=0.725 and h=0.735 rows from a
single fused ``evaluate()`` call), and computes the KW-Q1 statistic from
them. **Prints the band read (OWNS/INERT/MIXED) -- does NOT write a
"verdict"/"band" key to its JSON output.** Per the launch instruction:
"printing only; the runner writes the verdict."

Prerequisite driver invocations (one per theta node, FT config, the two
KW-Q1 h-nodes, seeds 900101-900104 -- run_theta_node's own GATE gets checked
per-node against evaluate()'s guard; none of these 3 nodes is theta-engaged
at site 2.3 alone since theta_sites defaults to "all" and s != 1 engages it,
so --smear is left at "auto" = smeared, the registered/paired form the claim
card names as "primary"):

    uv run python hier_s0_driver.py --arm S0-A --config ft \\
        --seeds 900101,900102,900103,900104 --nodes s_minus,truth,s_plus \\
        --h-nodes 0.725,0.735 --score-h 0.725 \\
        --out-root <out-root>

(a single invocation covers all 3 nodes x 4 seeds x 2 h in one process,
since run_arm's own seed-level parallelism (--jobs) already fans out across
seeds; --score-h is required because H_GEN=0.73 is not among --h-nodes, see
hier_s0_driver.py's ``_resolve_score_h``.) This writes:

    <out-root>/s0a_seed<seed>/node_<name>_ft/simulations/diagnostics/event_likelihoods.csv

for name in {s_minus, truth, s_plus} (the FT-config node-dir suffix is
"_ft", from hier_s0_driver.py's ``_node_dir_suffix("all", "auto", "ft")`` --
imported here, not re-derived, so this script never falls out of sync with
the driver's own naming).

**Statistic** (claim card sec 1.3, C2's own subtraction reused verbatim from
``b4_imp_stage1_forecast.py``'s ``matrices()``/``per_event_scores()``):
for each theta node (s-value) and event i, at h in {0.725, 0.735}:

    cat_term_i(h)  = (alpha_G_phi_i(h) / r_Malm_i(h)) * L_cat_no_bh_i(h) / D_tilde_phi_i(h)
    comp_term_i(h) = B_num_i(h) / D_tilde_phi_i(h)
    full_i(h)      = combined_no_bh_i(h)               (GATE I: cat_term+comp_term == full)
    pure_i(h)      = clip(full_i(h) - cat_term_i(h), 0, None)
    s_full_i  = [ln full_i(0.735)  - ln full_i(0.725)] / (0.735 - 0.725)   (both > 0, else NaN)
    s_pure_i  = [ln pure_i(0.735)  - ln pure_i(0.725)] / (0.735 - 0.725)   (both > 0, else NaN)
    s_imp,i(s) = s_full_i - s_pure_i           (identically 0 for L_cat_no_bh == 0 events)

**Frozen q1 membership** -- loaded, NOT recomputed, from two already-banked
artifacts: ``b4_imp_stage1_events.csv`` (arm="ft" rows; z_true/active_073
are identical across arms, verified 2026-08-29: same (seed, event_idx) ->
same z_true/active_073 regardless of arm) and ``b4_imp_stage1_forecast.json``
``covariates.ft.z_true.edges`` (the ALL-EVENTS, not active-only, z_true
quartile edges -- ``[0.3575..., 0.4593..., 0.5840...]``; edges[0]=0.35750
is the claim card's stated "0.358" cutoff, to more decimal places than the
card quotes. NOTE: this is a DIFFERENT quartiling than the card's own
Sec. C2 z_true_active_only table (edges [0.3381,...], active events only) --
this script uses the ALL-EVENTS edges because that is what the claim card's
sec 1.3 explicitly cites ("z_true < 0.358") and what its Sec 1.2 table row
"z_true (edges 0.358/0.459/0.584)" reports; disclosed, not silently
resolved).

**S(s)** = mean of s_imp,i(s) over the pooled (seed, event_idx) q1 set
present in that node's CSV (fleet-pooled across all requested seeds, not
averaged-then-pooled -- same convention as ``hier_s0_driver.compute_scores``).
**R** = [S(sqrt2) - S(1/sqrt2)] / |S(1)|.
**Bands** (two-sided in |R|, printed only): OWNS |R|>=0.5, INERT |R|<=0.2,
MIXED otherwise.

**GATE ENG** (catalogue leg): fraction of "active" rows (L_cat_no_bh > 0 at
h=0.725 on the truth/s=1 node) where L_cat_no_bh differs across s-nodes;
must be >= 0.99 or the read is NULL-BY-CONSTRUCTION (A15 corollary, claim
card sec 1.3).

**Falsifier** (A14): q1's share of Sigma s_imp at s=1 (pooled across ALL 4
quartiles, all requested seeds) -- if it falls below 50%, C2's low-z
localisation attribution is WITHDRAWN regardless of R (printed, not banked).

**GATE PARITY** (T-ID, inherited from B1): NOT computed by this script --
the s=1 (truth) node's CSV under this invocation IS the theta=(0,1) node,
identical in construction to a "fresh same-commit FT re-evaluation"; the
runner should independently re-run
``hier_s0_driver.py --arm S0-A --config ft --nodes truth --h-nodes 0.725,0.735
--seeds <one of 900101-900104> --out-root <a SEPARATE out-root>`` and diff
its ``combined_no_bh``/``L_cat_no_bh`` columns against this run's truth-node
CSV (both h rows) -- bit-identical is the target, matching GATE PARITY's
convention elsewhere in this driver family. (The 2026-08-23 banked FT CSVs
under ``p3_work/ft_*_work/`` predate Sigma^phi and are explicitly NOT the
comparand, per the claim card.)

Outputs: ``kwq1_score_output.json`` (all computed numbers; NO "verdict"/
"band" key -- printing only, per the launch instruction) under ``--out-root``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import hier_s0_driver as hsd  # noqa: E402 -- reuse _node_dir_suffix/THETA_NODES/H_GEN, not re-derive

REPO_ROOT = Path(__file__).resolve().parents[4]
EVENTS_CSV = THIS_DIR / "b4_imp_stage1_events.csv"
FORECAST_JSON = THIS_DIR / "b4_imp_stage1_forecast.json"
CLAIM_CARD = THIS_DIR / "CLAIM_IMPOSTOR_DRAG_20260829.md"

H_LO_DEFAULT: float = 0.725
H_HI_DEFAULT: float = 0.735
DEFAULT_SEEDS: tuple[int, ...] = (900101, 900102, 900103, 900104)
KWQ1_NODES: tuple[str, ...] = ("s_minus", "truth", "s_plus")  # s = 1/sqrt2, 1, sqrt2 at b=0

GATE_I_TOL = 2.0e-6  # b4_imp_stage1_forecast.py's own tolerance, reused verbatim
GATE_ENG_MIN_FRACTION = 0.99  # claim card sec 1.3
FALSIFIER_MIN_Q1_SHARE = 0.50  # claim card sec 1.3 / A14

BAND_OWNS = 0.5
BAND_INERT = 0.2


def _node_csv_path(out_root: Path, arm_prefix: str, seed: int, node: str, suffix: str) -> Path:
    return (
        out_root
        / f"{arm_prefix}{seed}"
        / f"node_{node}{suffix}"
        / "simulations"
        / "diagnostics"
        / "event_likelihoods.csv"
    )


def per_event_scores_two_h(vals_lo: npt.NDArray[np.float64], vals_hi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Central-difference secant over exactly two h values -- identical formula
    to ``b4_imp_stage1_forecast.py``'s ``per_event_scores`` restricted to a
    2-column grid (this script's CSVs only ever carry H_LO/H_HI, not the
    full H_GRID_41)."""
    out = np.full(vals_lo.shape[0], np.nan)
    ok = (vals_lo > 0.0) & (vals_hi > 0.0)
    out[ok] = (np.log(vals_hi[ok]) - np.log(vals_lo[ok])) / (H_HI_DEFAULT - H_LO_DEFAULT)
    return out


def load_node_frame(diag_csv: Path, h_lo: float, h_hi: float) -> pd.DataFrame:
    """Read one node's diagnostics CSV and return a per-event_idx frame with
    s_full, s_pure, s_imp, gate_i_max_rel, and L_cat_no_bh at h_lo (for GATE ENG).
    """
    df = pd.read_csv(diag_csv)
    at_lo = df[np.isclose(df["h"].to_numpy(dtype=np.float64), h_lo)].sort_values("event_idx").reset_index(drop=True)
    at_hi = df[np.isclose(df["h"].to_numpy(dtype=np.float64), h_hi)].sort_values("event_idx").reset_index(drop=True)
    if at_lo.empty or at_hi.empty:
        raise RuntimeError(
            f"{diag_csv}: missing h={h_lo!r} or h={h_hi!r} rows (h values present: {sorted(set(df['h']))})"
        )
    merged = at_lo.merge(at_hi, on="event_idx", suffixes=("_lo", "_hi"))

    def _cat_term(suf: str) -> npt.NDArray[np.float64]:
        alpha = merged[f"alpha_G_phi{suf}"].to_numpy(np.float64)
        r_malm = merged[f"r_Malm{suf}"].to_numpy(np.float64)
        lcat = merged[f"L_cat_no_bh{suf}"].to_numpy(np.float64)
        dtilde = merged[f"D_tilde_phi{suf}"].to_numpy(np.float64)
        return np.asarray((alpha / r_malm) * lcat / dtilde, dtype=np.float64)

    def _comp_term(suf: str) -> npt.NDArray[np.float64]:
        bnum = merged[f"B_num{suf}"].to_numpy(np.float64)
        dtilde = merged[f"D_tilde_phi{suf}"].to_numpy(np.float64)
        return np.asarray(bnum / dtilde, dtype=np.float64)

    full_lo = merged["combined_no_bh_lo"].to_numpy(np.float64)
    full_hi = merged["combined_no_bh_hi"].to_numpy(np.float64)
    cat_lo, cat_hi = _cat_term("_lo"), _cat_term("_hi")
    comp_lo, comp_hi = _comp_term("_lo"), _comp_term("_hi")

    scale_lo = np.maximum(np.abs(full_lo), np.finfo(float).tiny)
    scale_hi = np.maximum(np.abs(full_hi), np.finfo(float).tiny)
    gate_i_max_rel = float(
        np.nanmax(
            np.concatenate(
                [
                    np.abs(cat_lo + comp_lo - full_lo) / scale_lo,
                    np.abs(cat_hi + comp_hi - full_hi) / scale_hi,
                ]
            )
        )
    )

    pure_lo = np.clip(full_lo - cat_lo, 0.0, None)
    pure_hi = np.clip(full_hi - cat_hi, 0.0, None)
    s_full = per_event_scores_two_h(full_lo, full_hi)
    s_pure = per_event_scores_two_h(pure_lo, pure_hi)
    s_imp = s_full - s_pure

    out = pd.DataFrame(
        {
            "event_idx": merged["event_idx"],
            "s_full": s_full,
            "s_pure": s_pure,
            "s_imp": s_imp,
            "L_cat_no_bh_lo": merged["L_cat_no_bh_lo"].to_numpy(np.float64),
        }
    )
    out.attrs["gate_i_max_rel"] = gate_i_max_rel
    return out


def load_q1_membership(events_csv: Path, forecast_json: Path, seeds: tuple[int, ...]) -> tuple[set[tuple[int, int]], dict[int, set[tuple[int, int]]], float]:
    """Load the FROZEN quartile membership from already-banked artifacts (do
    NOT recompute): ``b4_imp_stage1_events.csv`` (arm="ft" rows) quartiled by
    ``b4_imp_stage1_forecast.json``'s ``covariates.ft.z_true.edges`` (the
    ALL-EVENTS z_true quartile -- see module docstring for why this edge set,
    not the z_true_active_only one, is the "0.358" the claim card names).

    Returns ``(q1_pairs, all_quartile_pairs, edge_q1)`` where ``q1_pairs`` is
    the frozen (seed, event_idx) set restricted to the requested seeds,
    ``all_quartile_pairs`` maps quartile number (1-4) to its (seed,
    event_idx) set (for the falsifier share), and ``edge_q1`` is the loaded
    z_true < edge_q1 cutoff (for disclosure in the output JSON).
    """
    forecast = json.loads(forecast_json.read_text())
    edges = forecast["covariates"]["ft"]["z_true"]["edges"]
    e0, e1, e2 = edges
    df = pd.read_csv(events_csv)
    df = df[(df["arm"] == "ft") & (df["seed"].isin(seeds))]
    z = df["z_true"].to_numpy(np.float64)
    q = np.where(z < e0, 1, np.where(z < e1, 2, np.where(z < e2, 3, 4)))
    pairs_by_q: dict[int, set[tuple[int, int]]] = {k: set() for k in (1, 2, 3, 4)}
    for seed, event_idx, qk in zip(df["seed"].to_numpy(int), df["event_idx"].to_numpy(int), q):
        pairs_by_q[int(qk)].add((int(seed), int(event_idx)))
    return pairs_by_q[1], pairs_by_q, float(e0)


def pooled_mean(frame_by_seed: dict[int, pd.DataFrame], column: str, pairs: set[tuple[int, int]]) -> tuple[float, float, int]:
    vals: list[float] = []
    for seed, frame in frame_by_seed.items():
        mask = frame["event_idx"].apply(lambda ei, s=seed: (s, ei) in pairs)
        sub = frame.loc[mask, column].to_numpy(np.float64)
        vals.extend(sub[np.isfinite(sub)].tolist())
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    return mean, sem, int(arr.size)


def gate_eng_catalogue_leg(frames: dict[str, dict[int, pd.DataFrame]], seeds: tuple[int, ...]) -> dict[str, Any]:
    """GATE ENG on the catalogue leg (claim card sec 1.3): fraction of active
    rows (L_cat_no_bh > 0 at h_lo on the truth/s=1 node) where L_cat_no_bh
    differs between the two extreme s-nodes (s_minus vs s_plus -- the largest
    expected separation)."""
    diffs: list[bool] = []
    active_flags: list[bool] = []
    for seed in seeds:
        truth = frames["truth"].get(seed)
        sm = frames["s_minus"].get(seed)
        sp = frames["s_plus"].get(seed)
        if truth is None or sm is None or sp is None:
            continue
        merged = truth[["event_idx", "L_cat_no_bh_lo"]].merge(
            sm[["event_idx", "L_cat_no_bh_lo"]], on="event_idx", suffixes=("_truth", "_sm")
        ).merge(sp[["event_idx", "L_cat_no_bh_lo"]].rename(columns={"L_cat_no_bh_lo": "L_cat_no_bh_lo_sp"}), on="event_idx")
        active = merged["L_cat_no_bh_lo_truth"].to_numpy(np.float64) > 0.0
        differ = merged["L_cat_no_bh_lo_sm"].to_numpy(np.float64) != merged["L_cat_no_bh_lo_sp"].to_numpy(np.float64)
        active_flags.extend(active.tolist())
        diffs.extend((differ[active]).tolist())
    n_active = int(sum(active_flags))
    frac_differ = float(np.mean(diffs)) if diffs else float("nan")
    return {
        "n_active_rows_pooled": n_active,
        "fraction_L_cat_differs_across_s": frac_differ,
        "pass": bool(np.isfinite(frac_differ) and frac_differ >= GATE_ENG_MIN_FRACTION),
        "threshold": GATE_ENG_MIN_FRACTION,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=str, required=True, help="hier_s0_driver.py --out-root used for the --config ft run(s).")
    ap.add_argument("--seeds", type=str, default=None, help=f"Comma-separated seeds; default {DEFAULT_SEEDS}.")
    ap.add_argument("--arm-prefix", type=str, default="s0a_seed", help="Per-seed dir prefix hier_s0_driver.py used (S0-A default 's0a_seed').")
    ap.add_argument("--theta-sites", type=str, default="all", choices=hsd.THETA_SITES_CHOICES)
    ap.add_argument("--smear", type=str, default="auto", choices=hsd.SMEAR_CHOICES)
    ap.add_argument("--config", type=str, default="ft", choices=hsd.CONFIG_CHOICES)
    ap.add_argument("--h-lo", type=float, default=H_LO_DEFAULT)
    ap.add_argument("--h-hi", type=float, default=H_HI_DEFAULT)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    seeds = tuple(int(x) for x in args.seeds.split(",")) if args.seeds else DEFAULT_SEEDS
    suffix = hsd._node_dir_suffix(args.theta_sites, args.smear, args.config)

    frames: dict[str, dict[int, pd.DataFrame]] = {node: {} for node in KWQ1_NODES}
    missing: list[str] = []
    gate_i_max_rel = 0.0
    for node in KWQ1_NODES:
        for seed in seeds:
            diag_csv = _node_csv_path(out_root, args.arm_prefix, seed, node, suffix)
            if not diag_csv.is_file():
                missing.append(str(diag_csv))
                continue
            frame = load_node_frame(diag_csv, args.h_lo, args.h_hi)
            gate_i_max_rel = max(gate_i_max_rel, frame.attrs["gate_i_max_rel"])
            frames[node][seed] = frame

    if missing:
        print(f"[kwq1_score] {len(missing)} node CSV(s) missing on disk: {missing}", flush=True)

    q1_pairs, quartile_pairs, edge_q1 = load_q1_membership(EVENTS_CSV, FORECAST_JSON, seeds)

    s_of_node: dict[str, dict[str, Any]] = {}
    for node in KWQ1_NODES:
        mean, sem, n = pooled_mean(frames[node], "s_imp", q1_pairs)
        s_of_node[node] = {"S": mean, "sem": sem, "n_pooled": n}

    s_minus_S = s_of_node["s_minus"]["S"]
    truth_S = s_of_node["truth"]["S"]
    s_plus_S = s_of_node["s_plus"]["S"]
    R = (s_plus_S - s_minus_S) / abs(truth_S) if np.isfinite(truth_S) and truth_S != 0 else float("nan")

    # Falsifier (A14): q1's share of Sigma s_imp at s=1 (truth), over ALL
    # quartiles pooled, same seeds.
    quartile_sums: dict[int, float] = {}
    for qk, pairs in quartile_pairs.items():
        mean_q, _sem_q, n_q = pooled_mean(frames["truth"], "s_imp", pairs)
        quartile_sums[qk] = mean_q * n_q if np.isfinite(mean_q) else 0.0
    total_s_imp = sum(quartile_sums.values())
    q1_share = quartile_sums[1] / total_s_imp if total_s_imp != 0 else float("nan")

    gate_eng = gate_eng_catalogue_leg(frames, seeds)

    if not np.isfinite(R):
        band = "UNDETERMINED (non-finite R -- insufficient data)"
    elif abs(R) >= BAND_OWNS:
        band = "KERNEL-WIDTH-OWNS"
    elif abs(R) <= BAND_INERT:
        band = "KERNEL-WIDTH-INERT"
    else:
        band = "MIXED"

    output: dict[str, Any] = {
        "instrument": "KW-Q1 (B4.2), CLAIM_IMPOSTOR_DRAG_20260829.md sec 1.3",
        "out_root": str(out_root),
        "seeds": list(seeds),
        "config": args.config,
        "theta_sites": args.theta_sites,
        "smear": args.smear,
        "h_lo": args.h_lo,
        "h_hi": args.h_hi,
        "node_dir_suffix": suffix,
        "missing_csv_paths": missing,
        "q1_edge_z_true": edge_q1,
        "q1_n_pairs_requested_seeds": len(q1_pairs),
        "S_by_node": s_of_node,
        "R": R,
        "gate_i_max_rel": gate_i_max_rel,
        "gate_i_tol": GATE_I_TOL,
        "gate_i_pass": bool(gate_i_max_rel <= GATE_I_TOL) if frames["truth"] else None,
        "gate_eng": gate_eng,
        "falsifier_q1_share_of_total_s_imp_at_truth": q1_share,
        "falsifier_min_q1_share": FALSIFIER_MIN_Q1_SHARE,
        "falsifier_withdrawn": (bool(np.isfinite(q1_share) and q1_share < FALSIFIER_MIN_Q1_SHARE)),
        # NO "verdict"/"band" key persisted -- printing only, per the launch
        # instruction ("printing only; the runner writes the verdict").
    }

    out_json = out_root / "kwq1_score_output.json"
    out_json.write_text(json.dumps(output, indent=2, default=str))
    print(f"wrote {out_json}")
    print(f"S(1/sqrt2)={s_minus_S!r}  S(1)={truth_S!r}  S(sqrt2)={s_plus_S!r}")
    print(f"R = [S(sqrt2) - S(1/sqrt2)] / |S(1)| = {R!r}")
    print(f"GATE I max rel (cat+comp vs full identity): {gate_i_max_rel!r} (tol {GATE_I_TOL!r})")
    print(f"GATE ENG (catalogue leg, s_minus vs s_plus, active rows): {gate_eng}")
    print(f"Falsifier: q1 share of total s_imp at s=1 = {q1_share!r} (threshold {FALSIFIER_MIN_Q1_SHARE!r}; "
          f"WITHDRAWN={output['falsifier_withdrawn']})")
    print(f"BAND READ (printed only, not banked): {band}")
    print(
        "REPORTED-ONLY -- this script prints the band; the runner adjudicates and records the "
        "verdict per rule 2 (this script's author may not run/verdict the registered measurement)."
    )
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
