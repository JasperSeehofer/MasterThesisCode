"""Merge the sharded O4 runs into the registered fleet statistic.

Sharded execution (2026-08-21, author-agreed operating lesson: instrument runs
get costed/parallelized up front): the pre-committed scorer
``o4_pairing_test.py`` (commit bfe4d09c) was run once per F seed in parallel
(``--seeds <s> --out shard_<s>.json``). This merger concatenates the per-seed
entries and applies the SAME committed reduction: the fleet statistic is the
mean over per-seed ``mean_score`` values and the bands come from the scorer's
own ``apply_bands`` — no formula is reimplemented here.

Usage:
    uv run python results/prod2d_closure_20260818/o4_merge_shards.py
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from o4_pairing_test import ARM, F_SEEDS, H_GEN, apply_bands  # noqa: E402

SHARD_DIR = HERE / "o4_shards"
OUT = HERE / "o4_pairing_test_output.json"
ARMS = ("A", "A1_window_to_full", "A2_gl50_to_trapz1500", "A3_clamp_to_zeroext")


def main() -> int:
    gate_r4: list[dict[str, Any]] = []
    gate_t4: list[dict[str, Any]] = []
    per_seed: dict[str, list[dict[str, Any]]] = {a: [] for a in ARMS}
    base: dict[str, Any] | None = None
    for seed in F_SEEDS:
        p = SHARD_DIR / f"shard_{seed}.json"
        if not p.is_file():
            print(f"MISSING shard for seed {seed} -- merge refused", file=sys.stderr)
            return 1
        d = json.loads(p.read_text())
        if d.get("status") == "GATES_ONLY_PASS" or "fleet" not in d:
            print(f"shard {seed} carries no fleet section -- merge refused", file=sys.stderr)
            return 1
        base = d
        gate_r4.extend(d["gate_r4"])
        gate_t4.extend(d["gate_t4"])
        for a in ARMS:
            per_seed[a].extend(d["fleet"][a]["per_seed"])

    r4_fail = [r for r in gate_r4 if not r.get("pass")]
    t4_fail = [r for r in gate_t4 if not r.get("pass")]
    if r4_fail or t4_fail:
        print(
            f"GATE FAILURES across shards: R4 {len(r4_fail)}, T4 {len(t4_fail)} -- "
            "the O4 statistic may not be read",
            file=sys.stderr,
        )

    fleet: dict[str, Any] = {}
    for a in ARMS:
        means = np.array(
            [s["mean_score"] for s in per_seed[a] if s["mean_score"] is not None],
            dtype=np.float64,
        )
        n = int(means.size)
        s_bar = float(means.mean()) if n else None
        sem = float(means.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        band, owned = apply_bands(s_bar) if (a == "A" and s_bar is not None) else (None, None)
        fleet[a] = {
            "n_seeds": n,
            "S_bar": s_bar,
            "sem_seeds": sem,
            "per_seed": per_seed[a],
            "band_fired": band,
            "owned_fraction": owned,
            "reference": (
                "PREREGISTRATION_SELFGEN_CONTROL.md PRE-CHECK O4, Statistic + Bands "
                "(bands applied only to arm A); merged from 15 single-seed shards "
                "of the committed scorer (bfe4d09c), reduction via its own apply_bands"
            ),
        }

    assert base is not None
    output = {
        "registered_in": base["registered_in"],
        "arm": ARM,
        "h_gen": H_GEN,
        "sharded_execution_note": (
            "15 parallel single-seed invocations of o4_pairing_test.py (identical "
            "committed code) merged by o4_merge_shards.py; per-seed numbers are "
            "bit-identical to a serial run (seeds are independent in the scorer)."
        ),
        "n_seeds_requested": len(F_SEEDS),
        "redshift_upper_limit_used": base["redshift_upper_limit_used"],
        "redshift_upper_limit_note": base["redshift_upper_limit_note"],
        "gate_r4": gate_r4,
        "gate_t4": gate_t4,
        "gates_pass": not (r4_fail or t4_fail),
        "fleet": fleet,
    }
    OUT.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                "status": "OK" if not (r4_fail or t4_fail) else "GATES-FAILED",
                "out": str(OUT),
                **{a: fleet[a]["S_bar"] for a in ARMS},
                "band_fired_A": fleet["A"]["band_fired"],
                "owned_fraction_A": fleet["A"]["owned_fraction"],
            },
            indent=2,
        )
    )
    return 0 if not (r4_fail or t4_fail) else 1


if __name__ == "__main__":
    raise SystemExit(main())
