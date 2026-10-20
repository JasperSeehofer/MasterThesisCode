"""Synthetic (<=10-row) exercise of `timeout_q2_reads.py`'s registered logic.

NOT run on the registered population — every table here has <=10 rows and is
built in this script, never touching a real §1-pinned input except through a
DELIBERATE md5 mismatch (to exercise the INSTRUMENT-DEFECT path).

`timeout_q2_reads.py`'s population-size anchors (N_SCORED=1588, pool=200,100,
Sigma Y=89,456, g-byteid n_kept/n_timeout, ...) are hard-coded to the real
registered population, so a small synthetic table can never reach the
disposition rows THROUGH `main()`/the CLI — any synthetic run trips one of
those anchor checks first, which is exactly the INSTRUMENT-DEFECT surface
this harness exercises (Part 1). The disposition and weighting FUNCTIONS
themselves (`disposition_s2_2`, `disposition_s2_3`, `s2_3_weights`,
`_weighted_moments`, `gbyteid_gate`) take plain arrays/dataframes with no
hard-coded size, so Part 2 calls them directly, off the CLI, with fabricated
<=10-row inputs to hit every disposition tag.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeout_q2_reads as q2  # noqa: E402

NODE = Path(__file__).resolve().parent
SCRATCH = Path(
    "/tmp/claude-1000/-home-jasper-Repositories-darksiren-emri/"
    "b548869c-f9b8-4d04-9d35-636c6aa4e4c6/scratchpad/synth_q2"
)
SCRATCH.mkdir(parents=True, exist_ok=True)

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


# ===========================================================================
# Part 1 — INSTRUMENT-DEFECT, via the real CLI, tiny fixtures
# ===========================================================================


def make_tiny_crb_csv(path: Path) -> None:
    """3-row CRB CSV, real column names, minimal viable content."""
    cols = {
        "M": [1e5, 5e5, 2e6],
        "mu": [10, 10, 10],
        "a": [0.98, 0.98, 0.98],
        "p0": [12.0, 13.0, 11.0],
        "e0": [0.1, 0.1, 0.1],
        "x0": [0.0, 0.0, 0.0],
        "luminosity_distance": [3.0, 4.0, 5.0],
        "qS": [1.0, 1.2, 1.4],
        "phiS": [1.0, 1.2, 1.4],
        "delta_luminosity_distance_delta_luminosity_distance": [0.01, 0.02, 0.03],
        "delta_qS_delta_qS": [0.001, 0.001, 0.001],
        "delta_phiS_delta_qS": [0.0001, 0.0001, 0.0001],
        "delta_phiS_delta_phiS": [0.001, 0.001, 0.001],
        "SNR": [30, 40, 50],
        "generation_time": [0.2, 0.3, 0.4],
    }
    pd.DataFrame(cols).to_csv(path, index=False)


def part1_instrument_defect() -> None:
    print("\n=== Part 1: INSTRUMENT-DEFECT (real CLI, tiny fixtures) ===")
    crb_path = SCRATCH / "tiny_crb.csv"
    make_tiny_crb_csv(crb_path)
    actual_md5 = q2._md5(crb_path)

    edges_path = SCRATCH / "tiny_edges.json"
    edges_path.write_text(
        '{"seed61000_M_edges": [10000, 50000, 200000, 800000, 3000000, 14000000]}'
    )
    edges_md5 = q2._md5(edges_path)

    rate_path = SCRATCH / "tiny_rate.csv"
    rate_path.write_text("bin,n_timeout\n0,0\n")
    rate_md5 = q2._md5(rate_path)

    common_args = [
        "--crb-csv",
        str(crb_path),
        "--bin-edges-json",
        str(edges_path),
        "--bin-edges-md5",
        edges_md5,
        "--rate-table-m-csv",
        str(rate_path),
        "--rate-table-m-md5",
        rate_md5,
        "--pool-dir",
        str(SCRATCH / "pool"),
        "--pool-manifest",
        str(SCRATCH / "pool_manifest.md5"),
        "--pool-manifest-md5",
        "deadbeefdeadbeefdeadbeefdeadbeef",
        "--log-dir",
        str(SCRATCH / "logs_root"),
        "--log-manifest",
        str(SCRATCH / "log_manifest.md5"),
        "--log-manifest-md5",
        "deadbeefdeadbeefdeadbeefdeadbeef",
        "--event-likelihoods-iiib",
        str(SCRATCH / "nonexistent_iiib.csv"),
        "--event-likelihoods-iiib-md5",
        "deadbeefdeadbeefdeadbeefdeadbeef",
        "--event-likelihoods-jr1",
        str(SCRATCH / "nonexistent_jr1.csv"),
        "--event-likelihoods-jr1-md5",
        "deadbeefdeadbeefdeadbeefdeadbeef",
        "--influence-iiib",
        str(SCRATCH / "nonexistent_infl_iiib.csv"),
        "--influence-iiib-md5",
        "deadbeefdeadbeefdeadbeefdeadbeef",
        "--influence-jr1",
        str(SCRATCH / "nonexistent_infl_jr1.csv"),
        "--influence-jr1-md5",
        "deadbeefdeadbeefdeadbeefdeadbeef",
    ]

    # (a) CRB CSV md5 mismatch -- the very first pin check.
    out_bad_md5 = SCRATCH / "out_bad_crb_md5.json"
    r = subprocess.run(
        [
            sys.executable,
            str(NODE / "timeout_q2_reads.py"),
            "--crb-csv-md5",
            "0" * 32,
            *common_args,
            "--out",
            str(out_bad_md5),
        ],
        capture_output=True,
        text=True,
    )
    check("(a) wrong CRB md5 -> exit 1", r.returncode == 1, r.stdout + r.stderr)
    check("(a) INSTRUMENT-DEFECT printed", "INSTRUMENT-DEFECT" in r.stdout, r.stdout)
    check("(a) 'CRB CSV' pin named in message", "CRB CSV" in r.stdout, r.stdout)
    if out_bad_md5.exists():
        import json

        rep = json.loads(out_bad_md5.read_text())
        check(
            "(a) JSON disposition == INSTRUMENT-DEFECT",
            rep["disposition"]["value"] == "INSTRUMENT-DEFECT",
        )

    # (b) correct CRB md5, but the scored-row-count anchor (1588) cannot match
    # a 3-row synthetic table -> INSTRUMENT-DEFECT downstream of the pin check.
    out_bad_pop = SCRATCH / "out_bad_population.json"
    r2 = subprocess.run(
        [
            sys.executable,
            str(NODE / "timeout_q2_reads.py"),
            "--crb-csv-md5",
            actual_md5,
            *common_args,
            "--out",
            str(out_bad_pop),
        ],
        capture_output=True,
        text=True,
    )
    check(
        "(b) correct CRB md5, wrong population size -> exit 1",
        r2.returncode == 1,
        r2.stdout + r2.stderr,
    )
    check("(b) INSTRUMENT-DEFECT printed", "INSTRUMENT-DEFECT" in r2.stdout, r2.stdout)
    check(
        "(b) message names the scored-subset row-count mismatch",
        "scored CRB subset has" in r2.stdout or "pool" in r2.stdout.lower(),
        r2.stdout,
    )

    # (c) --dry-run with the same broken fixture: prints, never writes --out.
    out_dry = SCRATCH / "out_dry_should_not_exist.json"
    if out_dry.exists():
        out_dry.unlink()
    r3 = subprocess.run(
        [
            sys.executable,
            str(NODE / "timeout_q2_reads.py"),
            "--crb-csv-md5",
            actual_md5,
            *common_args,
            "--out",
            str(out_dry),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    check("(c) --dry-run on broken fixture -> exit 1", r3.returncode == 1, r3.stdout + r3.stderr)
    check("(c) --dry-run never writes --out on INSTRUMENT-DEFECT", not out_dry.exists())


# ===========================================================================
# Part 2 — disposition rows + g-byteid + weights + moments, direct function calls
# ===========================================================================


def part2_gbyteid() -> None:
    print("\n=== Part 2a: gbyteid_gate ===")
    n_kept_ok = np.array(list(q2.N_KEPT_ANCHOR))
    n_to_ok = np.array(list(q2.N_TIMEOUT_SNR_STAGE_ANCHOR))
    try:
        q2.gbyteid_gate(n_kept_ok, n_to_ok)
        check("gbyteid_gate: matching anchors -> no raise", True)
    except q2.InstrumentDefectError:
        check("gbyteid_gate: matching anchors -> no raise", False)

    n_kept_bad = n_kept_ok.copy()
    n_kept_bad[2] += 1
    try:
        q2.gbyteid_gate(n_kept_bad, n_to_ok)
        check("gbyteid_gate: n_kept mismatch -> raises InstrumentDefectError", False)
    except q2.InstrumentDefectError as e:
        check("gbyteid_gate: n_kept mismatch -> raises InstrumentDefectError", True)
        check("  message names n_kept", "n_kept" in e.message)

    n_to_bad = n_to_ok.copy()
    n_to_bad[0] += 5
    try:
        q2.gbyteid_gate(n_kept_ok, n_to_bad)
        check("gbyteid_gate: n_timeout mismatch -> raises InstrumentDefectError", False)
    except q2.InstrumentDefectError as e:
        check("gbyteid_gate: n_timeout mismatch -> raises InstrumentDefectError", True)
        check("  message names n_timeout", "n_timeout" in e.message)


def part2_s2_3_weights() -> None:
    print("\n=== Part 2b: s2_3_weights (support bins {2,3}, unit weight elsewhere) ===")
    # 6 kept "events": 1 in bin1 (unsupported -> w_e=1), 3 in bin2, 2 in bin3.
    crb_scored = pd.DataFrame(
        {
            "event_idx": [0, 1, 2, 3, 4, 5],
            "M": [
                60000.0,  # bin 1
                300000.0,  # bin 2
                350000.0,  # bin 2
                400000.0,  # bin 2
                1000000.0,  # bin 3
                1200000.0,  # bin 3
            ],
        }
    )
    edges = np.array([11467.7, 47353.3, 195533.8, 807408.8, 3333997.1, 13766938.7])
    pool = pd.DataFrame(
        {
            "M": [300000.0, 300000.0, 1000000.0, 1000000.0, 1000000.0, 1000000.0],
            "SNR": [25, 25, 25, 25, 25, 25],
            "stratum": ["a"] * 6,
        }
    )
    out = q2.s2_3_weights(crb_scored, pool, edges)
    check("s2_3_weights: supported bins are {2,3}", out["supported_bins"] == [2, 3])
    check(
        "s2_3_weights: n_events_unit_weight == 1 (the bin-1 event)",
        out["n_events_unit_weight"] == 1,
    )
    check("s2_3_weights: n_events_reweighted == 5 (bins 2+3)", out["n_events_reweighted"] == 5)
    events = out["events"]
    total = float(events.sum())
    check(
        "s2_3_weights: Sigma w_e == n_scored (6) after renormalisation",
        abs(total - 6.0) < 1e-9,
        f"got {total}",
    )
    check(
        "s2_3_weights: w_b['2'] is a positive finite ratio",
        float(out["w_b"].get("2", -1.0)) > 0,
    )


def part2_weighted_moments_and_null() -> None:
    print("\n=== Part 2c: _weighted_moments (frozen T0 convention, tiny grid) ===")
    h_grid = np.array([0.60, 0.65, 0.70, 0.73, 0.75, 0.80])
    n_events = 4
    rng = np.random.default_rng(0)
    logL = rng.normal(loc=-1.0, scale=0.3, size=(n_events, h_grid.size))
    w_uniform = np.ones(n_events)
    mean_h, sigma_h = q2._weighted_moments(logL, h_grid, w_uniform)
    check("weighted_moments: mean_h within grid range", h_grid.min() <= mean_h <= h_grid.max())
    check("weighted_moments: sigma_h >= 0", sigma_h >= 0)

    # All-zero weights -> flat weighted posterior over H_GRID_41 -- a grid-
    # symmetry check independent of the logL data (same cross-check
    # `build_influence_vector.py` reports as `mean_h_all_removed`).
    mean_h0, _sigma_h0 = q2._weighted_moments(logL, h_grid, np.zeros(n_events))
    grid_w = np.gradient(h_grid)
    expected_flat_mean = float((h_grid * grid_w).sum() / grid_w.sum())
    check(
        "weighted_moments: all-zero weights -> flat-posterior grid mean",
        abs(mean_h0 - expected_flat_mean) < 1e-9,
        f"got {mean_h0}, expected {expected_flat_mean}",
    )

    # Sharpening (larger uniform weight on every event) concentrates the
    # posterior around its mode -> sigma_h should not increase.
    _mean_h5, sigma_h5 = q2._weighted_moments(logL, h_grid, w_uniform * 5.0)
    check(
        "weighted_moments: larger uniform weight sharpens (sigma_h non-increasing)",
        sigma_h5 <= sigma_h + 1e-12,
        f"sigma_h={sigma_h}, sigma_h(5x)={sigma_h5}",
    )


def part2_dispositions() -> None:
    print("\n=== Part 2d: disposition_s2_2 (M-STRUCTURED / M-FLAT / INTERMEDIATE) ===")
    material = {"p_perm_d_e": 0.005, "any_bin_holm_p_lt_0p05": True}
    check(
        "disposition_s2_2: MATERIAL case -> M-STRUCTURED",
        q2.disposition_s2_2(material)["value"] == "M-STRUCTURED",
    )

    immaterial = {"p_perm_d_e": 0.5, "any_bin_holm_p_lt_0p05": False}
    check(
        "disposition_s2_2: IMMATERIAL case -> M-FLAT",
        q2.disposition_s2_2(immaterial)["value"] == "M-FLAT",
    )

    intermediate = {"p_perm_d_e": 0.05, "any_bin_holm_p_lt_0p05": False}
    check(
        "disposition_s2_2: borderline case -> INTERMEDIATE",
        q2.disposition_s2_2(intermediate)["value"] == "INTERMEDIATE",
    )

    # p_perm < 0.01 but Fisher/Holm did not clear -> not MATERIAL (both legs required).
    borderline2 = {"p_perm_d_e": 0.005, "any_bin_holm_p_lt_0p05": False}
    check(
        "disposition_s2_2: p_perm<0.01 alone (no Holm) -> not M-STRUCTURED",
        q2.disposition_s2_2(borderline2)["value"] != "M-STRUCTURED",
    )
    for d in (material, immaterial, intermediate):
        check(
            "disposition_s2_2: mandatory p0-scope line present",
            q2.disposition_s2_2(d)["mandatory_note"] == q2.MANDATORY_P0_LINE,
        )

    print("\n=== Part 2e: disposition_s2_3 (MATERIAL / IMMATERIAL / INTERMEDIATE) ===")
    mat_by_delta = {"delta_mean_h": 0.02, "sigma_ratio": 1.0, "t_null": 0.002}
    check(
        "disposition_s2_3: |Delta|>=T_mat -> MATERIAL",
        q2.disposition_s2_3(mat_by_delta)["value"] == "POPULATION-MISMATCH-MATERIAL",
    )
    mat_by_ratio = {"delta_mean_h": 0.0, "sigma_ratio": 1.5, "t_null": 0.002}
    check(
        "disposition_s2_3: ratio outside [0.80,1.25] -> MATERIAL (even with Delta=0)",
        q2.disposition_s2_3(mat_by_ratio)["value"] == "POPULATION-MISMATCH-MATERIAL",
    )
    immaterial3 = {"delta_mean_h": 0.0005, "sigma_ratio": 1.0, "t_null": 0.002}
    check(
        "disposition_s2_3: small Delta + ratio in [0.95,1.05] -> IMMATERIAL",
        q2.disposition_s2_3(immaterial3)["value"] == "POPULATION-MISMATCH-IMMATERIAL",
    )
    intermediate3 = {"delta_mean_h": 0.004, "sigma_ratio": 1.1, "t_null": 0.002}
    check(
        "disposition_s2_3: mid-band -> INTERMEDIATE",
        q2.disposition_s2_3(intermediate3)["value"] == "POPULATION-MISMATCH-INTERMEDIATE",
    )
    for d in (mat_by_delta, mat_by_ratio, immaterial3, intermediate3):
        check(
            "disposition_s2_3: mandatory p0-scope line present",
            q2.disposition_s2_3(d)["mandatory_note"] == q2.MANDATORY_P0_LINE,
        )


def part2_holm() -> None:
    print("\n=== Part 2f: _holm (monotone, dominates raw p, alpha=0.05 boundary) ===")
    pvals = [0.001, 0.02, 0.03, 0.2, 0.9]
    adj = q2._holm(pvals)
    check("_holm: length preserved", len(adj) == len(pvals))
    check(
        "_holm: every adjusted p >= raw p",
        all(a >= p - 1e-15 for a, p in zip(adj, pvals, strict=True)),
    )
    check(
        "_holm: smallest raw p (0.001) clears alpha=0.05 after correction", adj[0] < 0.05, str(adj)
    )
    check("_holm: largest raw p (0.9) stays >= alpha=0.05", adj[-1] >= 0.05, str(adj))


def main() -> int:
    part1_instrument_defect()
    part2_gbyteid()
    part2_s2_3_weights()
    part2_weighted_moments_and_null()
    part2_dispositions()
    part2_holm()

    print(f"\n{'=' * 60}")
    if FAILURES:
        print(f"SYNTH CHECK: {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("SYNTH CHECK: all checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
