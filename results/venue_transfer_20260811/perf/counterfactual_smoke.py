"""Counterfactual equivalence smoke: affine default vs. exact pre-swap chain.

PERF WORK ONLY — not part of the certified instrument, not imported by it,
never touches a registered output path. Builds the venue-transfer smoke
context ONCE (cell Tc, h_true=0.730, balls='real_k', sigma_mode='glade',
n_events_cap=30 — identical parameters to ``profile_smoke.py``) and runs the
same seed TWICE:

  run A: code as committed (uncommitted swap active) — the cached
         two-segment affine evaluation of ``dark_mass_density_per_mass``.
  run B: the module-global ``dark_mass_density_per_mass`` name (the one
         ``completion_mass_factor_g`` resolves at call time, since it is a
         bare-name call inside the SAME module) rebound to
         ``functools.partial(original, exact=True)`` for the duration of the
         run, then restored — the verbatim pre-swap chain.

Both records are serialised through the identical deterministic JSON
encoder (sort_keys=True, floats via repr through a custom default) and
compared byte-for-byte; if not identical, per-leaf max abs/rel diffs are
computed on every float leaf.

Usage (from repo root):
    uv run python results/venue_transfer_20260811/perf/counterfactual_smoke.py [n_events_cap]

Does not write anything under results/venue_transfer_20260811/ except this
perf/ subdirectory.
"""

from __future__ import annotations

import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

from master_thesis_code.bayesian_inference import bayesian_statistics as bs
from master_thesis_code.validation import venue_transfer as vt

PERF_DIR = Path(__file__).resolve().parent


def _json_default(o: Any) -> Any:
    """Deterministic float/ndarray-safe fallback for json.dumps.

    Numpy scalar/array leaves are cast to native Python types so the
    serialisation is identical regardless of which code path produced them
    (numpy float64 vs. python float would otherwise repr identically via
    json anyway, but this also covers numpy integer/bool leaves and
    ndarrays that may appear in the record).
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dependency here
        np = None  # type: ignore[assignment]
    if np is not None:
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serialisable: {o!r}")


def to_deterministic_json(rec: dict[str, Any]) -> str:
    """Serialise ``rec`` deterministically: sorted keys, stable float repr."""
    return json.dumps(rec, sort_keys=True, default=_json_default, allow_nan=True)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a JSON-compatible nested structure to {dotted.path: leaf}."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            out.update(_flatten(obj[k], f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def diff_leaves(rec_a: dict[str, Any], rec_b: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-leaf max abs/rel differences between two JSON-compatible records."""
    flat_a = _flatten(rec_a)
    flat_b = _flatten(rec_b)
    diffs: list[dict[str, Any]] = []
    keys = sorted(set(flat_a) | set(flat_b))
    for k in keys:
        va = flat_a.get(k, None)
        vb = flat_b.get(k, None)
        if va == vb:
            continue
        is_num = isinstance(va, (int, float)) and isinstance(vb, (int, float))
        if is_num:
            abs_diff = abs(float(va) - float(vb))
            denom = max(abs(float(va)), abs(float(vb)), 1e-300)
            rel_diff = abs_diff / denom
            diffs.append(
                {
                    "leaf": k,
                    "a": va,
                    "b": vb,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                }
            )
        else:
            diffs.append({"leaf": k, "a": va, "b": vb, "abs_diff": None, "rel_diff": None})
    return diffs


def main() -> None:
    n_cap = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    vcfg = vt.VenueConfig(
        cell="Tc", h_true=0.730, balls="real_k", sigma_mode="glade", n_events_cap=n_cap
    )

    t0 = time.time()
    vctx = vt.build_venue_context(vcfg)
    t1 = time.time()
    print(
        f"[context] build_venue_context: {t1 - t0:.2f}s "
        f"n_events={vctx.event_rows.size} sum_K={int(vctx.K.sum())} max_K={int(vctx.K.max())}",
        flush=True,
    )

    seed = vt.VT_BASE_SEED + 44000  # inside the registered Tc(0.730) block, matches profile_smoke.py

    # --- run A: as-committed (affine default) ---
    ta0 = time.perf_counter()
    rec_a = vt.run_seed_venue(seed, vctx)
    ta1 = time.perf_counter()
    wall_a = ta1 - ta0
    print(f"[run A] affine default: {wall_a:.3f}s", flush=True)

    # --- run B: exact pre-swap chain, forced via module-global rebind ---
    original_fn = bs.dark_mass_density_per_mass
    bs.dark_mass_density_per_mass = functools.partial(original_fn, exact=True)  # type: ignore[assignment]
    try:
        tb0 = time.perf_counter()
        rec_b = vt.run_seed_venue(seed, vctx)
        tb1 = time.perf_counter()
    finally:
        bs.dark_mass_density_per_mass = original_fn
    wall_b = tb1 - tb0
    print(f"[run B] exact (pre-swap): {wall_b:.3f}s", flush=True)

    # sanity: restoration verified
    assert bs.dark_mass_density_per_mass is original_fn, "restore of module global failed"

    json_a = to_deterministic_json(rec_a)
    json_b = to_deterministic_json(rec_b)
    byte_identical = json_a == json_b

    diffs = [] if byte_identical else diff_leaves(rec_a, rec_b)
    max_abs = max((d["abs_diff"] for d in diffs if d["abs_diff"] is not None), default=0.0)
    max_rel = max((d["rel_diff"] for d in diffs if d["rel_diff"] is not None), default=0.0)

    result = {
        "n_events_cap": n_cap,
        "n_events": int(vctx.event_rows.size),
        "sum_K": int(vctx.K.sum()),
        "max_K": int(vctx.K.max()),
        "seed": seed,
        "context_build_s": t1 - t0,
        "wall_a_affine_s": wall_a,
        "wall_b_exact_s": wall_b,
        "byte_identical": byte_identical,
        "n_diffs": len(diffs),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "diffs": diffs,
        "rec_a": rec_a,
        "rec_b": rec_b,
    }

    out_path = PERF_DIR / "counterfactual_smoke.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default))

    print(f"[compare] byte_identical={byte_identical} n_diffs={len(diffs)}", flush=True)
    if not byte_identical:
        print(f"[compare] max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}", flush=True)
        for d in diffs[:20]:
            print(f"    {d}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
