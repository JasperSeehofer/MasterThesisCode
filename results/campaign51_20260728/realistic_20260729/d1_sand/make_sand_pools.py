#!/usr/bin/env python
"""D1 stage-3 instrumentation: materialise the two derived injection pools.

Registered by `PREREGISTRATION_D1_SAND_REWEIGHT.md` ("The run" -> "Minimal
instrumentation (named)"), Research Cycle stage 3. Parent claim:
`CLAIM_D1_P0WINDOW_20260805.md` (commit 751d7d98).

Tagged **instrumentation**, NOT **formula**: this script touches no file on the
`/physics-change` trigger list, imports nothing from `master_thesis_code`, and
changes no production code path. It only writes CSV copies of an existing
injection pool.

WHAT IT BUILDS
--------------
From the staged pool of record

    results/campaign51_20260728/realistic_20260729/gate_b_20260730/
        injection_pool_mix200k_20260728/          (707 files, 200 100 data rows)

two derived pools, row-for-row aligned:

    pool_p0kept    (arm A1)  the 647 files that carry a `p0` column, rows
                             unmodified -- the POOL-COMPOSITION CONTROL.
    pool_p0window  (arm A2)  the same 647 files with `SNR := 0.0` wherever
                             `p0 not in [10.002, 15.998]` -- the S_and-consistent
                             selection.

WHY `SNR := 0` IS THE RIGHT SUBSTITUTION
----------------------------------------
`SimulationDetectionProbability` builds `p_det` as the survival function of the
h-invariant horizon `d_hor = SNR * d_L / SNR_thr`. Zeroing the SNR of the
p0-rejected injections therefore yields EXACTLY the joint survival

    S_and = P(d_hor >= d_L  AND  p0 in W | M_z)

on the same 60x40 grid, with the same estimator and the same flags -- no
estimator edit required. Verbatim from
`.planning/derivation-2dbias-fix-20260803/fixb_x15_attribution/
cand_b_joint_selection.py:1-21` (module docstring) and `:90-94` (the mask + the
`df2.loc[bad, "SNR"] = 0.0` assignment).

THE 60 p0-LESS FILES
--------------------
60 of the 707 pool files come from the PRE-plunge-window `code_rev a9f29e82` and
carry no `p0` column at all (6 000 rows, 3.0 % of the pool, 1 426 of them
SNR >= 20). They are dropped from **BOTH** derived pools so numerator and
denominator see the identical injection population -- lifted verbatim from
`cand_b_joint_selection.py:79-88`. Any re-implementation that drops them from
only one side manufactures a spurious retention ratio (parent claim, "Errors to
avoid in this thread" item 6). Arm A1 exists precisely to control for this drop.

REGISTERED EXPECTATIONS (pre-registration, "The run" table)
-----------------------------------------------------------
    647 files, 194 100 data rows in each derived pool
    exactly 149 092 rows carry SNR = 0.0 in pool_p0window
    (= `d1_b2_sand_hslope.json:pool_fingerprint`)
A mismatch is a hard failure: the script exits non-zero.

OUTPUT
------
    <outdir>/pool_p0kept/*.csv
    <outdir>/pool_p0window/*.csv
    <outdir>/manifest_pool_p0kept.json     sha256 per file + pool-level digest
    <outdir>/manifest_pool_p0window.json   ditto

The manifests are committed; the pool CSVs are not (they are bulk data,
regenerable from this script plus the staged pool, and the manifests are the
provenance record).

USAGE
-----
    uv run python results/campaign51_20260728/realistic_20260729/d1_sand/\
make_sand_pools.py [--outdir DIR] [--force]

Default `--outdir` is this script's own directory.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
POOL = os.path.join(
    ROOT,
    "results/campaign51_20260728/realistic_20260729/gate_b_20260730/"
    "injection_pool_mix200k_20260728",
)

# The effective survival window: ParameterSpace.p0 = [10.0, 16.0] with
# derivative_epsilon = 1e-3, and five_point_stencil_derivative rejects whenever
# value +- 2*eps leaves the declared bounds
# (parameter_space.py:95-113, parameter_estimation.py:268-276).
# Identical literals to cand_b_joint_selection.py:63.
P0_LO, P0_HI = 10.0 + 2e-3, 16.0 - 2e-3

# Registered expectations (PREREGISTRATION_D1_SAND_REWEIGHT.md, "The run").
EXPECT_FILES_TOTAL = 707
EXPECT_FILES_KEPT = 647
EXPECT_ROWS_KEPT = 194_100
EXPECT_ROWS_P0_REJECT = 149_092


def sha256_file(path: str) -> str:
    """Streaming sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(pool_dir: str, out_json: str, extra: dict) -> dict:
    """sha256 every CSV in `pool_dir`, plus a pool-level digest of the sorted
    (basename, sha256) pairs -- one number that fingerprints the whole pool."""
    files = sorted(glob.glob(os.path.join(pool_dir, "*.csv")))
    entries = []
    agg = hashlib.sha256()
    for f in files:
        d = sha256_file(f)
        entries.append(
            {"file": os.path.basename(f), "sha256": d, "bytes": os.path.getsize(f)}
        )
        agg.update(os.path.basename(f).encode())
        agg.update(d.encode())
    manifest = {
        "pool_dir": os.path.relpath(pool_dir, ROOT),
        "source_pool": os.path.relpath(POOL, ROOT),
        "p0_window": [P0_LO, P0_HI],
        "n_files": len(files),
        "total_bytes": sum(e["bytes"] for e in entries),
        "pool_sha256": agg.hexdigest(),
        **extra,
        "files": entries,
    }
    with open(out_json, "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if the pool directories already exist",
    )
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    dir_S = os.path.join(outdir, "pool_p0kept")
    dir_A = os.path.join(outdir, "pool_p0window")

    src = sorted(glob.glob(os.path.join(POOL, "*.csv")))
    if len(src) != EXPECT_FILES_TOTAL:
        print(
            f"FAIL: source pool has {len(src)} csv files, expected "
            f"{EXPECT_FILES_TOTAL}: {POOL}",
            file=sys.stderr,
        )
        return 2

    if (os.path.isdir(dir_S) or os.path.isdir(dir_A)) and not args.force:
        print(f"derived pools already exist under {outdir}; pass --force to rebuild")
    else:
        for d in (dir_S, dir_A):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        t0 = time.time()
        # --- lifted verbatim from cand_b_joint_selection.py:76-95 -------------
        n_files_kept = 0
        n_in = n_out = 0
        n_files_skipped = 0
        n_rows_skipped = 0
        for f in src:
            df = pd.read_csv(f)
            # 60 of the 707 pool files come from the PRE-plunge-window code_rev
            # a9f29e82 and carry no p0 column at all (6000 rows, 3.0 % of the
            # pool, 1426 of them SNR >= 20): they are dropped from BOTH derived
            # pools so the numerator and denominator see the identical
            # injection population.  [cand_b_joint_selection.py:79-88]
            if "p0" not in df.columns:
                n_files_skipped += 1
                n_rows_skipped += len(df)
                continue
            df = df[df.p0.notna()]
            if len(df) == 0:
                n_files_skipped += 1
                continue
            n_files_kept += 1
            n_in += len(df)
            df.to_csv(os.path.join(dir_S, os.path.basename(f)), index=False)
            bad = (df.p0 < P0_LO) | (df.p0 > P0_HI)
            n_out += int(bad.sum())
            df2 = df.copy()
            df2.loc[bad, "SNR"] = 0.0
            df2.to_csv(os.path.join(dir_A, os.path.basename(f)), index=False)
        # ---------------------------------------------------------------------
        print(
            f"[pools] files kept {n_files_kept} (skipped {n_files_skipped}, "
            f"{n_rows_skipped} rows); rows kept {n_in}, p0-rejected {n_out} "
            f"({n_out / n_in:.4%})  [{time.time() - t0:.1f}s]"
        )
        ok = True
        for label, got, want in (
            ("files kept", n_files_kept, EXPECT_FILES_KEPT),
            ("rows kept", n_in, EXPECT_ROWS_KEPT),
            ("p0-rejected rows", n_out, EXPECT_ROWS_P0_REJECT),
        ):
            mark = "OK " if got == want else "FAIL"
            if got != want:
                ok = False
            print(f"  [{mark}] {label}: {got} (registered {want})")
        if not ok:
            print(
                "FAIL: derived pools do not match the registered expectations "
                "in PREREGISTRATION_D1_SAND_REWEIGHT.md",
                file=sys.stderr,
            )
            return 3

    # --- verification pass over what is on disk -----------------------------
    rows_S = rows_A = zeros_A = 0
    aligned = True
    for f in sorted(glob.glob(os.path.join(dir_S, "*.csv"))):
        b = os.path.basename(f)
        a = os.path.join(dir_A, b)
        dS = pd.read_csv(f)
        dA = pd.read_csv(a)
        rows_S += len(dS)
        rows_A += len(dA)
        zeros_A += int((dA["SNR"] == 0.0).sum())
        if len(dS) != len(dA) or not dS["p0"].equals(dA["p0"]):
            aligned = False
            print(f"  [FAIL] row misalignment between arms in {b}", file=sys.stderr)

    verify = {
        "n_files_p0kept": len(glob.glob(os.path.join(dir_S, "*.csv"))),
        "n_files_p0window": len(glob.glob(os.path.join(dir_A, "*.csv"))),
        "n_rows_p0kept": rows_S,
        "n_rows_p0window": rows_A,
        "n_rows_snr_zeroed_p0window": zeros_A,
        "arms_row_aligned": aligned,
        "registered_n_files": EXPECT_FILES_KEPT,
        "registered_n_rows": EXPECT_ROWS_KEPT,
        "registered_n_snr_zeroed": EXPECT_ROWS_P0_REJECT,
    }
    checks = [
        verify["n_files_p0kept"] == EXPECT_FILES_KEPT,
        verify["n_files_p0window"] == EXPECT_FILES_KEPT,
        rows_S == EXPECT_ROWS_KEPT,
        rows_A == EXPECT_ROWS_KEPT,
        zeros_A == EXPECT_ROWS_P0_REJECT,
        aligned,
    ]
    verify["all_checks_pass"] = all(checks)

    mS = write_manifest(
        dir_S,
        os.path.join(outdir, "manifest_pool_p0kept.json"),
        {"arm": "A1", "role": "pool-composition control", "verification": verify},
    )
    mA = write_manifest(
        dir_A,
        os.path.join(outdir, "manifest_pool_p0window.json"),
        {"arm": "A2", "role": "S_and-consistent selection", "verification": verify},
    )

    print(json.dumps(verify, indent=1))
    print(f"[manifest] pool_p0kept   sha256 {mS['pool_sha256']}")
    print(f"[manifest] pool_p0window sha256 {mA['pool_sha256']}")
    if not verify["all_checks_pass"]:
        print("FAIL: on-disk verification did not pass", file=sys.stderr)
        return 4
    print("VERDICT: derived pools match the registered expectations OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
