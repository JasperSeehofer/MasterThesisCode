"""Route-1 harvest: log every ``completion_mass_factor_g`` call on the
realistic venue query distribution (Tc, h_true=0.730, real_k, glade,
n_events_cap=30, seed VT_BASE_SEED+44000).

PERF/MEASUREMENT ONLY — does not touch ``master_thesis_code/``. Follows the
module-global rebind pattern of ``results/venue_transfer_20260811/perf/
counterfactual_smoke.py``: ``bs.completion_mass_factor_g`` is rebound to a
logging wrapper for the duration of the ONE seed run, then restored. The
wrapper records, per call, the four scalars (det_M_z, proj_d_L_to_M,
sigma_cond_M, n_hermite) and the per-node ``(z_nodes, d_L_fraction)`` arrays
that were passed in — everything needed to recompute ``mu_cond(z)`` and
``scale(z) = det_M_z / (1+z)`` offline, without re-running the instrument.

Usage (from repo root):
    uv run python results/venue_transfer_20260811/perf/route1_study/harvest_route1.py [n_events_cap]

Writes route1_harvest.npz under this directory.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from master_thesis_code.bayesian_inference import bayesian_statistics as bs
from master_thesis_code.validation import venue_transfer as vt

STUDY_DIR = Path(__file__).resolve().parent
MAX_TOTAL_ROWS = 10_000_000  # subsample threshold for per-node arrays


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

    seed = vt.VT_BASE_SEED + 44000  # matches counterfactual_smoke.py / profile_smoke.py

    # --- instrumentation: rebind the module-global name bayesian_statistics
    # resolves at call time (venue_transfer imports the SYMBOL, not the
    # module, so we must patch it in BOTH places it is bound). ---
    calls: list[dict[str, Any]] = []
    original_fn = bs.completion_mass_factor_g

    def _logging_wrapper(
        z_nodes: np.ndarray,
        d_L_fraction: np.ndarray,
        det_M_z: float,
        proj_d_L_to_M: float,
        sigma_cond_M: float,
        *,
        n_hermite: int = bs._G_I_HERMITE_NODES,
    ) -> np.ndarray:
        calls.append(
            {
                "det_M_z": float(det_M_z),
                "proj_d_L_to_M": float(proj_d_L_to_M),
                "sigma_cond_M": float(sigma_cond_M),
                "n_hermite": int(n_hermite),
                "z_nodes": np.asarray(z_nodes, dtype=np.float64).copy(),
                "d_L_fraction": np.asarray(d_L_fraction, dtype=np.float64).copy(),
            }
        )
        return original_fn(
            z_nodes,
            d_L_fraction,
            det_M_z,
            proj_d_L_to_M,
            sigma_cond_M,
            n_hermite=n_hermite,
        )

    bs.completion_mass_factor_g = _logging_wrapper  # type: ignore[assignment]
    # venue_transfer does `from ...bayesian_statistics import completion_mass_factor_g`
    # (bare-name call site in _g_i_completion_kernel-style helpers) so the vt
    # module's own binding must be patched too, matching the call-site lookup.
    vt_original_fn = getattr(vt, "completion_mass_factor_g", None)
    if vt_original_fn is not None:
        vt.completion_mass_factor_g = _logging_wrapper  # type: ignore[assignment]

    try:
        tw0 = time.perf_counter()
        rec = vt.run_seed_venue(seed, vctx)
        tw1 = time.perf_counter()
    finally:
        bs.completion_mass_factor_g = original_fn
        if vt_original_fn is not None:
            vt.completion_mass_factor_g = vt_original_fn
        assert bs.completion_mass_factor_g is original_fn, "restore of module global failed"

    wall = tw1 - tw0
    n_calls = len(calls)
    total_rows = sum(c["z_nodes"].size for c in calls)
    print(
        f"[harvest] seed run: {wall:.3f}s, n_calls={n_calls}, "
        f"total_z_nodes={total_rows}, n_events_result={rec.get('n_events', 'n/a')}",
        flush=True,
    )

    # Flatten per-call scalars broadcast over their node arrays into one
    # long per-node table (call_id, det_M_z, proj_d_L_to_M, sigma_cond_M,
    # n_hermite, z_node, d_L_fraction).
    call_id_parts = []
    det_M_z_parts = []
    proj_parts = []
    sigma_parts = []
    n_hermite_parts = []
    z_parts = []
    dlfrac_parts = []
    for cid, c in enumerate(calls):
        n = c["z_nodes"].size
        call_id_parts.append(np.full(n, cid, dtype=np.int64))
        det_M_z_parts.append(np.full(n, c["det_M_z"], dtype=np.float64))
        proj_parts.append(np.full(n, c["proj_d_L_to_M"], dtype=np.float64))
        sigma_parts.append(np.full(n, c["sigma_cond_M"], dtype=np.float64))
        n_hermite_parts.append(np.full(n, c["n_hermite"], dtype=np.int64))
        z_parts.append(c["z_nodes"].reshape(-1))
        dlfrac_parts.append(c["d_L_fraction"].reshape(-1))

    call_id = np.concatenate(call_id_parts) if call_id_parts else np.zeros(0, dtype=np.int64)
    det_M_z_arr = np.concatenate(det_M_z_parts) if det_M_z_parts else np.zeros(0)
    proj_arr = np.concatenate(proj_parts) if proj_parts else np.zeros(0)
    sigma_arr = np.concatenate(sigma_parts) if sigma_parts else np.zeros(0)
    n_hermite_arr = np.concatenate(n_hermite_parts) if n_hermite_parts else np.zeros(0, dtype=np.int64)
    z_arr = np.concatenate(z_parts) if z_parts else np.zeros(0)
    dlfrac_arr = np.concatenate(dlfrac_parts) if dlfrac_parts else np.zeros(0)

    n_total = z_arr.size
    subsampled = False
    rng = np.random.default_rng(20260808)
    if n_total > MAX_TOTAL_ROWS:
        subsampled = True
        # Stratified by call_id: sample a fixed fraction of rows within each
        # call so every event/chunk stays represented.
        frac = MAX_TOTAL_ROWS / n_total
        keep_mask = np.zeros(n_total, dtype=bool)
        for cid in np.unique(call_id):
            idx = np.flatnonzero(call_id == cid)
            k = max(1, int(round(idx.size * frac)))
            k = min(k, idx.size)
            sel = rng.choice(idx, size=k, replace=False)
            keep_mask[sel] = True
        call_id = call_id[keep_mask]
        det_M_z_arr = det_M_z_arr[keep_mask]
        proj_arr = proj_arr[keep_mask]
        sigma_arr = sigma_arr[keep_mask]
        n_hermite_arr = n_hermite_arr[keep_mask]
        z_arr = z_arr[keep_mask]
        dlfrac_arr = dlfrac_arr[keep_mask]
        print(
            f"[harvest] subsampled per-node rows: {n_total} -> {z_arr.size} "
            f"(stratified by call_id, target {MAX_TOTAL_ROWS})",
            flush=True,
        )

    out_path = STUDY_DIR / "route1_harvest.npz"
    np.savez_compressed(
        out_path,
        call_id=call_id,
        det_M_z=det_M_z_arr,
        proj_d_L_to_M=proj_arr,
        sigma_cond_M=sigma_arr,
        n_hermite=n_hermite_arr,
        z_node=z_arr,
        d_L_fraction=dlfrac_arr,
        n_calls=np.array([n_calls]),
        n_total_rows_before_subsample=np.array([n_total]),
        subsampled=np.array([subsampled]),
        wall_seed_s=np.array([wall]),
        n_events_cap=np.array([n_cap]),
        seed=np.array([seed]),
    )
    print(f"wrote {out_path} ({z_arr.size} per-node rows, {n_calls} calls)", flush=True)


if __name__ == "__main__":
    main()
