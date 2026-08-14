#!/usr/bin/env python
"""V-M5 no-drift anchor — REGISTERED VALUES-GOLDEN comparison (rtol <= 1e-12).

Closes disclosure D-A1-2 (`results/mechanism_study_20260813/A1_READOUT.md`):
V-M5 was re-registered in `PREREGISTRATION_MECHANISM_ISOLATION.md` §5 as a
values golden — *not* a bit-identity test — after the author-ratified Route 1
adaptive Gauss-Hermite change to `bayesian_statistics.py` made bit-identity
unsatisfiable by construction. The registered pass condition (quoted
verbatim from §5):

    "V-M5 passes when every shared field agrees with the committed record to
    rtol <= 1e-12 *and* both channels' MAP values are exactly equal."

`darksiren_emri/validation/venue_transfer.py::run_v_t5_compat_check` performs
the OLD bit-identity (`!=`) comparison and is NOT modified here (it is a
separate, still-registered check — A1-DET, not V-M5). This script is a
read-only, additive artifact that reuses the exact same instrument path
(`log_channel_posteriors_ball_sigma_vector` in v2-compat mode, the same
gate-style record assembly, the same committed reference file and seeds) but
scores it against the REGISTERED V-M5 rtol condition instead, and reports a
per-field breakdown so any failure is diagnosable.

No production module is modified. No registered document is modified.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from darksiren_emri.validation import calibration_gate as cg  # noqa: E402
from darksiren_emri.validation import venue_transfer as vt  # noqa: E402

RTOL = 1.0e-12
MAP_FIELDS = ("map_1d", "map_2d", "map_1d_refined", "map_2d_refined")


def _rel_dev(mine: float, ref: float) -> float:
    """Relative deviation |mine-ref| / max(|ref|, tiny), 0.0 if exactly equal."""
    if mine == ref:
        return 0.0
    denom = max(abs(ref), 1.0e-300)
    return abs(mine - ref) / denom


def _flatten(key: str, value: Any) -> list[tuple[str, Any]]:
    """Flatten list-valued fields (e.g. the 41-point ln_post vectors) into
    per-index scalar entries so a single-element deviation is diagnosable."""
    if isinstance(value, list):
        return [(f"{key}[{i}]", v) for i, v in enumerate(value)]
    return [(key, value)]


def compute_records() -> dict[int, dict[str, Any]]:
    """Recompute the gate-style record for each V-T5 seed at current HEAD,
    using exactly the instrument path `run_v_t5_compat_check` uses (no
    changes to that function or to any production module)."""
    with open(REPO_ROOT / vt.B2_H0P730_RESULTS_JSON) as fh:
        committed = json.load(fh)
    cfg_dict = dict(committed["config"])
    cfg_dict["h_grid"] = tuple(cfg_dict["h_grid"])
    gcfg = cg.GateConfig(**cfg_dict)
    gctx = cg.build_gate_context(gcfg)

    mine_by_seed: dict[int, dict[str, Any]] = {}
    for seed in vt.V_T5_SEEDS:
        rng = np.random.default_rng(seed)
        universe = cg.draw_universe_gate(gctx, rng)
        ball = cg.draw_ball(gctx, universe, rng)
        sigma_pairs = np.full(ball.z_obs.size, gcfg.sigma_z, dtype=np.float64)
        ln1, ln2, slope = vt.log_channel_posteriors_ball_sigma_vector(
            gctx, universe, ball, sigma_pairs, chunk_pairs=0
        )
        mine_by_seed[seed] = vt._gate_style_record(
            gctx, seed, universe, ball, ln1, ln2, slope
        )
    return mine_by_seed


def load_committed() -> dict[int, dict[str, Any]]:
    with open(REPO_ROOT / vt.B2_H0P730_RESULTS_JSON) as fh:
        committed = json.load(fh)
    return {r["seed"]: r for r in committed["per_seed"]}


def score_seed(mine: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    shared_keys = sorted(set(mine) & set(ref))
    field_reports: list[dict[str, Any]] = []
    max_rel_dev = 0.0
    max_rel_dev_field = None
    map_mismatches: list[str] = []

    for key in shared_keys:
        mv, rv = mine[key], ref[key]
        flat_mine = _flatten(key, mv)
        flat_ref = _flatten(key, rv)
        if len(flat_mine) != len(flat_ref):
            field_reports.append(
                {
                    "field": key,
                    "error": "length mismatch",
                    "n_mine": len(flat_mine),
                    "n_ref": len(flat_ref),
                }
            )
            max_rel_dev = math.inf
            max_rel_dev_field = key
            continue
        field_max_dev = 0.0
        for (fk, fmv), (_, frv) in zip(flat_mine, flat_ref):
            if isinstance(fmv, bool) or isinstance(frv, bool):
                dev = 0.0 if fmv == frv else math.inf
            elif isinstance(fmv, (int, float)) and isinstance(frv, (int, float)):
                dev = _rel_dev(float(fmv), float(frv))
            elif isinstance(fmv, str) and isinstance(frv, str):
                dev = 0.0 if fmv == frv else math.inf
            else:
                dev = 0.0 if fmv == frv else math.inf
            if dev > field_max_dev:
                field_max_dev = dev
            if key in MAP_FIELDS and fmv != frv:
                map_mismatches.append(fk)
        if field_max_dev > max_rel_dev:
            max_rel_dev = field_max_dev
            max_rel_dev_field = key
        field_reports.append(
            {
                "field": key,
                "max_rel_dev": field_max_dev,
                "within_rtol": field_max_dev <= RTOL,
            }
        )

    golden_pass = all(fr.get("within_rtol", False) for fr in field_reports)
    maps_exact = not map_mismatches

    return {
        "n_shared_fields": len(shared_keys),
        "fields": field_reports,
        "max_rel_dev": max_rel_dev,
        "max_rel_dev_field": max_rel_dev_field,
        "map_fields_checked": list(MAP_FIELDS),
        "map_mismatches": map_mismatches,
        "maps_exact": maps_exact,
        "golden_pass_rtol": golden_pass,
        "vm5_pass": bool(golden_pass and maps_exact),
    }


def env_report() -> dict[str, Any]:
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    show_config_text = buf.getvalue()

    import os

    thread_env = {
        k: os.environ.get(k)
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "numpy_show_config": show_config_text,
        "thread_env_vars": thread_env,
    }


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> int:
    committed_by_seed = load_committed()
    mine_by_seed = compute_records()

    per_seed_results: dict[str, Any] = {}
    overall_pass = True
    overall_max_dev = 0.0
    overall_max_dev_field = None

    for seed in vt.V_T5_SEEDS:
        ref = committed_by_seed[seed]
        mine = mine_by_seed[seed]
        result = score_seed(mine, ref)
        per_seed_results[str(seed)] = result
        overall_pass = overall_pass and result["vm5_pass"]
        if result["max_rel_dev"] > overall_max_dev:
            overall_max_dev = result["max_rel_dev"]
            overall_max_dev_field = (seed, result["max_rel_dev_field"])

    report = {
        "check": "V-M5 (registered values golden, rtol<=1e-12)",
        "registered_condition": (
            "V-M5 passes when every shared field agrees with the committed "
            "record to rtol <= 1e-12 and both channels' MAP values are "
            "exactly equal."
        ),
        "rtol": RTOL,
        "committed_json": vt.B2_H0P730_RESULTS_JSON,
        "seeds": list(vt.V_T5_SEEDS),
        "git_head": git_head(),
        "overall_max_rel_dev": overall_max_dev,
        "overall_max_rel_dev_location": overall_max_dev_field,
        "per_seed": per_seed_results,
        "vm5_pass": overall_pass,
        "environment": env_report(),
    }

    out_path = Path(__file__).resolve().parent / "VM5_GOLDEN_20260814.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"git HEAD: {report['git_head']}")
    print(f"numpy: {report['environment']['numpy_version']}")
    print(f"overall max rel dev: {overall_max_dev}  at {overall_max_dev_field}")
    print()
    for seed in vt.V_T5_SEEDS:
        r = per_seed_results[str(seed)]
        print(f"seed {seed}: vm5_pass={r['vm5_pass']}  maps_exact={r['maps_exact']}  "
              f"max_rel_dev={r['max_rel_dev']}  at field={r['max_rel_dev_field']}")
        nontrivial = [f for f in r["fields"] if f.get("max_rel_dev", 0.0) > 0.0]
        for f in sorted(nontrivial, key=lambda x: -x["max_rel_dev"])[:10]:
            print(f"    {f['field']}: max_rel_dev={f['max_rel_dev']:.3e}  within_rtol={f['within_rtol']}")
        if r["map_mismatches"]:
            print(f"    MAP MISMATCHES: {r['map_mismatches']}")
    print()
    print(f"V-M5 VERDICT: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Full report written to {out_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
