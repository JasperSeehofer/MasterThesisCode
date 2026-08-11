"""F4 (money figure) — spec-z vs photo-z decomposition, PHYSICALLY-CORRECT channel.

The literal channel (``f4_specz_decomposition.py``: 1-D ``event_log_likelihood`` +
"any spec-z in the +/-5 sigma box") CANNOT carry F4, for two structural reasons that
run confirmed on all 3361 real events:

  (1) DEGENERATE classifier: the H0-independent +/-5 sigma 1-D d_L candidate box
      contains 1.5e4 - 2.7e6 GLADE galaxies and therefore ALWAYS contains thousands
      of spectroscopic hosts -> essentially every event is tagged "spec-z hosted",
      so there is no split to decompose.
  (2) sigma_z-BLIND likelihood: ``event_log_likelihood`` uses each catalogue host
      redshift as EXACT (a delta, ``norm.pdf(dist(z,h), d_meas, sigma_dL)``), so the
      photo-z vs spec-z distinction NEVER enters the likelihood; the single-event
      posterior shape is independent of host-z provenance.

F4's mechanism -- photo-z information starvation -- lives in the HOST-Z CONVOLUTION
the real ``single_host_likelihood`` performs: each candidate host is convolved by
its redshift PDF ``norm(z; z_g, sigma_z)``. Photometric hosts (sigma_z ~ 0.035,
~14x the GW distance precision) wash the sharp GW distance information out; spectro-
scopic hosts (sigma_z ~ 0.0017) keep it sharp. This is reproduced faithfully by
``_bridge_sky.event_loglik_sky(mode="conv")``, whose SKY localisation additionally
prunes candidates to the LISA Fisher cone -- which (a) makes the convolution
tractable and (b) makes spec-z presence NON-degenerate (a cone holds few hosts).

This script therefore builds F4 on the sigma_z-aware sky channel:
  * per event: single-event H0 posterior via ``event_loglik_sky(mode="conv")`` (real
    p_det, real pixelated completeness, host-z convolution, B_num) -- the faithful
    pipeline likelihood, and a delta-z reference via ``mode="1d"`` on the SAME sky
    candidates (isolates the sigma_z effect);
  * classification: each event is ``specz_dominated`` if SPECTROSCOPIC hosts
    (GLADE flag==3) carry >= 50% of the sigma_z-broadened, rate-weighted candidate
    contribution near the GW distance, else ``photoz_dominated``; spec-z PRESENCE is
    also recorded;
  * stacked posterior = sum of single-event log-posteriors, split by class.

Nothing in the H0 computation is modified -- this is an ADDITIVE (analysis) change.

Run:  uv run python scripts/bridge_closure/f4_specz_decomposition_conv.py [max_events]
Out:  scripts/bridge_closure/outputs/f4_specz_decomposition_conv.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logging.disable(logging.WARNING)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _bridge_lib as B  # noqa: E402,N812
import _bridge_sky as S  # noqa: E402,N812

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    precompute_completion_denominator,
    precompute_global_catalog_selection,
    precompute_missing_completion_denominator,
)
from darksiren_emri.galaxy_catalogue.handler import (  # noqa: E402
    InternalCatalogColumns as IC,  # noqa: N814
)
from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402

H_GRID: list[float] = [float(h) for h in np.round(np.arange(0.60, 0.8701, 0.01), 4)]
_OMEGA_M = B._OMEGA_M
_OMEGA_DE = B._OMEGA_DE
SPECZ_FLAG = 3
PHOTOZ_FLAG = 1
SIGMA_MULT = 1.5  # pipeline candidate cone (rung_G / single_host_likelihood)
OUT_JSON = B.OUTPUTS / "f4_specz_decomposition_conv.json"


def _build_flag_sky_catalog() -> S.SkyCatalog:
    """SkyCatalog with the retained GLADE ``REDSHIFT_FLAG`` attached (aligned).

    ``SkyCatalog`` (shuffle_sky=False, max_zerr=None) applies the mask
    ``good = finite(z,M,phi,theta,zerr) & z>0`` and preserves row order, so the
    same mask on the handler's flag column yields a flag array aligned to
    ``cat.z``.
    """
    cat = S.SkyCatalog(shuffle_sky=False)
    rc = cat.handler.reduced_galaxy_catalog
    z = rc[IC.REDSHIFT].to_numpy(dtype=np.float64)
    M = rc[IC.BH_MASS].to_numpy(dtype=np.float64)
    phi = rc[IC.PHI_S].to_numpy(dtype=np.float64)
    theta = rc[IC.THETA_S].to_numpy(dtype=np.float64)
    zerr = rc[IC.REDSHIFT_ERROR].to_numpy(dtype=np.float64)
    flag = rc[IC.REDSHIFT_FLAG].to_numpy()
    good = (
        np.isfinite(z)
        & np.isfinite(M)
        & np.isfinite(phi)
        & np.isfinite(theta)
        & np.isfinite(zerr)
        & (z > 0)
    )
    flag_aligned = flag[good].astype(np.int64)
    assert flag_aligned.shape[0] == cat.z.shape[0], "flag/catalog length mismatch"
    assert np.array_equal(z[good], cat.z), "flag misaligned with SkyCatalog rows"
    cat.flag = flag_aligned  # type: ignore[attr-defined]
    return cat


def _classify(ev: dict[str, Any], cat: S.SkyCatalog, h_true: float) -> dict[str, Any]:
    """Spec-z dominance / presence in the event's sky-cone candidate set.

    The dominance weight is the sigma_z-BROADENED, rate-weighted GW-distance
    likelihood at ``h_true``: each candidate host contributes
    ``w_g * N(d_L(z_g); d_meas, sqrt(sigma_dL^2 + (|dd_L/dz| sigma_z_g)^2))`` -- the
    same physics ``mode="conv"`` integrates, collapsed to the host peak. Photo-z
    hosts (large sigma_z) get a broad, low peak; spec-z hosts stay sharp/tall. The
    spec-z share of this weighted sum is the F4 discriminator.
    """
    # sky cone (same selection event_loglik_sky uses)
    ssky = np.sqrt(max(ev["phi2"] * np.sin(ev["theta"]) ** 2, ev["the2"]))
    radius = float(SIGMA_MULT * max(ssky, 1e-3))
    cand = cat.candidates(ev["phi"], ev["theta"], radius)
    n_cone = int(cand.size)
    if n_cone == 0:
        return {
            "n_cone": 0,
            "n_specz_cone": 0,
            "n_photoz_cone": 0,
            "specz_weight_frac": 0.0,
            "class": "photoz_dominated",
            "specz_present": False,
        }
    d_meas = float(ev["d_meas"])
    sdL = float(ev["sigma_dL"])
    zg = cat.z[cand]
    szg = np.maximum(cat.zerr[cand], 1e-5)
    wg = cat.w[cand]
    flg = cat.flag[cand]  # type: ignore[attr-defined]
    # |dd_L/dz| at each host (finite difference), map sigma_z -> sigma_dL_host
    dz = 1e-3
    dl = np.asarray(dist_vectorized(zg, h=h_true), dtype=np.float64)
    dl_p = np.asarray(dist_vectorized(np.clip(zg + dz, 1e-5, None), h=h_true), dtype=np.float64)
    ddl_dz = np.abs(dl_p - dl) / dz
    sig_tot = np.sqrt(sdL**2 + (ddl_dz * szg) ** 2)
    # rate-weighted, sigma_z-broadened GW-distance peak weight
    peak = np.exp(-0.5 * ((dl - d_meas) / sig_tot) ** 2) / (np.sqrt(2 * np.pi) * sig_tot)
    contrib = wg * peak
    total = float(np.sum(contrib))
    is_spec = flg == SPECZ_FLAG
    spec_w = float(np.sum(contrib[is_spec]))
    frac = spec_w / total if total > 0 else 0.0
    return {
        "n_cone": n_cone,
        "n_specz_cone": int(np.count_nonzero(is_spec)),
        "n_photoz_cone": int(np.count_nonzero(flg == PHOTOZ_FLAG)),
        "specz_weight_frac": frac,
        "class": "specz_dominated" if frac >= 0.5 else "photoz_dominated",
        "specz_present": bool(np.any(is_spec)),
    }


def _posterior(
    ev: dict[str, Any],
    cat: S.SkyCatalog,
    mode: str,
    D_tab: dict[float, float],
    bGbar_tab: dict[float, float],
    gdenom_tab: dict[float, float],
    completeness: Any,
) -> npt.NDArray[np.float64]:
    out = np.empty(len(H_GRID), dtype=np.float64)
    for i, h in enumerate(H_GRID):
        out[i] = S.event_loglik_sky(
            ev,
            cat,
            h,
            D_tab[h],
            D_tab[h] - bGbar_tab[h],
            gdenom_tab[h],
            mode=mode,
            sigma_mult=SIGMA_MULT,
            completeness=completeness,
            include_bnum=True,
        )
    return out


def main(max_events: int | None = None) -> dict[str, Any]:
    t0 = time.time()
    events = S.load_real_events_with_sky(apply_cuts=True)
    if max_events is not None:
        events = events[:max_events]
    n_ev = len(events)
    print(f"[f4-conv] events (after cuts): {n_ev}", flush=True)

    cat = _build_flag_sky_catalog()
    n_specz_cat = int(np.count_nonzero(cat.flag == SPECZ_FLAG))  # type: ignore[attr-defined]
    n_photoz_cat = int(np.count_nonzero(cat.flag == PHOTOZ_FLAG))  # type: ignore[attr-defined]
    print(
        f"[f4-conv] catalogue: {len(cat.z)} (flag1 photo={n_photoz_cat}, flag3 spec={n_specz_cat}); "
        f"median sigma_z={np.median(cat.zerr):.4f}",
        flush=True,
    )

    real_pdet = S.make_real_pdet()
    real_comp = S.make_real_completeness()
    tprec = time.time()
    D_tab = precompute_completion_denominator(
        H_GRID, real_pdet, Omega_m=_OMEGA_M, Omega_DE=_OMEGA_DE
    )
    bGbar_tab = precompute_missing_completion_denominator(H_GRID, real_pdet, completeness=real_comp)
    gdenom_tab = precompute_global_catalog_selection(
        H_GRID, cat.handler, real_pdet, with_bh_mass=False
    )
    print(f"[f4-conv] precomputes done ({time.time() - tprec:.1f}s)", flush=True)

    out_events: list[dict[str, Any]] = []
    tev = time.time()
    for j, ev in enumerate(events):
        cls = _classify(ev, cat, B.TRUE_H)
        lp_conv = _posterior(ev, cat, "conv", D_tab, bGbar_tab, gdenom_tab, real_comp)
        lp_1d = _posterior(ev, cat, "1d", D_tab, bGbar_tab, gdenom_tab, real_comp)
        info_conv = B.extract_map(H_GRID, lp_conv, B.TRUE_H)
        info_1d = B.extract_map(H_GRID, lp_1d, B.TRUE_H)
        out_events.append(
            {
                "index": j,
                "d_meas": float(ev["d_meas"]),
                "sigma_dL": float(ev["sigma_dL"]),
                "rel_dist_err": float(ev["sigma_dL"] / ev["d_meas"]),
                "in_catalog": bool(ev["in_catalog"]),
                **cls,
                "logpost_conv": [float(x) for x in lp_conv],
                "logpost_1d": [float(x) for x in lp_1d],
                "h_refined_conv": info_conv["h_refined"],
                "railed_conv": bool(info_conv["railed"]),
                "peaked_conv": bool(not info_conv["railed"]),
                "h_refined_1d": info_1d["h_refined"],
                "railed_1d": bool(info_1d["railed"]),
                "peaked_1d": bool(not info_1d["railed"]),
            }
        )
        if (j + 1) % 200 == 0:
            rate = (time.time() - tev) / (j + 1)
            print(
                f"[f4-conv] {j + 1}/{n_ev} ({rate * 1000:.0f} ms/ev, "
                f"~{rate * (n_ev - j - 1) / 60:.1f} min left)",
                flush=True,
            )

    # stacks -----------------------------------------------------------------
    def _stack(indices: list[int], key: str) -> dict[str, Any] | None:
        if not indices:
            return None
        total = np.sum(np.array([out_events[i][key] for i in indices], dtype=np.float64), axis=0)
        return B.extract_map(H_GRID, total, B.TRUE_H)

    idx_all = list(range(n_ev))
    idx_spec = [i for i in idx_all if out_events[i]["class"] == "specz_dominated"]
    idx_phot = [i for i in idx_all if out_events[i]["class"] == "photoz_dominated"]

    def _pk(indices: list[int], key: str) -> float:
        if not indices:
            return float("nan")
        return float(np.mean([out_events[i][key] for i in indices]))

    stacked = {
        "conv": {
            "all": _stack(idx_all, "logpost_conv"),
            "specz_dominated": _stack(idx_spec, "logpost_conv"),
            "photoz_dominated": _stack(idx_phot, "logpost_conv"),
        },
        "delta_z_1d": {
            "all": _stack(idx_all, "logpost_1d"),
            "specz_dominated": _stack(idx_spec, "logpost_1d"),
            "photoz_dominated": _stack(idx_phot, "logpost_1d"),
        },
    }

    summary = {
        "n_events": n_ev,
        "n_specz_dominated": len(idx_spec),
        "n_photoz_dominated": len(idx_phot),
        "n_specz_present": int(sum(e["specz_present"] for e in out_events)),
        "peaked_frac_conv_all": _pk(idx_all, "peaked_conv"),
        "peaked_frac_conv_specz": _pk(idx_spec, "peaked_conv"),
        "peaked_frac_conv_photoz": _pk(idx_phot, "peaked_conv"),
        "peaked_frac_1d_all": _pk(idx_all, "peaked_1d"),
        "peaked_frac_1d_specz": _pk(idx_spec, "peaked_1d"),
        "peaked_frac_1d_photoz": _pk(idx_phot, "peaked_1d"),
    }

    def _map(s: dict[str, Any] | None) -> float | None:
        return None if s is None else float(s["h_refined"])

    summary["map_conv_all"] = _map(stacked["conv"]["all"])
    summary["map_conv_specz"] = _map(stacked["conv"]["specz_dominated"])
    summary["map_conv_photoz"] = _map(stacked["conv"]["photoz_dominated"])
    summary["map_1d_all"] = _map(stacked["delta_z_1d"]["all"])
    summary["map_1d_specz"] = _map(stacked["delta_z_1d"]["specz_dominated"])
    summary["map_1d_photoz"] = _map(stacked["delta_z_1d"]["photoz_dominated"])

    payload = {
        "meta": {
            "description": "F4 spec-z vs photo-z decomposition on the sigma_z-aware sky (conv) channel",
            "h_grid": H_GRID,
            "h_true": B.TRUE_H,
            "sigma_mult": SIGMA_MULT,
            "specz_flag": SPECZ_FLAG,
            "photoz_flag": PHOTOZ_FLAG,
            "completeness": "pixel_completeness (real, f_k per pixel)",
            "pdet": "real SimulationDetectionProbability",
            "classifier": "spec-z >= 50% of sigma_z-broadened rate-weighted cone contribution at h_true",
            "n_catalog_galaxies": int(len(cat.z)),
            "n_catalog_specz": n_specz_cat,
            "n_catalog_photoz": n_photoz_cat,
            "runtime_s": round(time.time() - t0, 1),
        },
        "summary": summary,
        "stacked": stacked,
        "events": out_events,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    print("\n===== F4 spec-z decomposition (sigma_z-aware sky channel) =====", flush=True)
    print(
        f"events: {n_ev}   spec-z DOMINATED: {len(idx_spec)}   "
        f"photo-z dominated: {len(idx_phot)}   (spec-z present in cone: {summary['n_specz_present']})"
    )
    print(
        f"peaked frac [conv]  all={summary['peaked_frac_conv_all']:.3f}  "
        f"specz={summary['peaked_frac_conv_specz']:.3f}  photoz={summary['peaked_frac_conv_photoz']:.3f}"
    )
    print(
        f"peaked frac [1d/delta-z] all={summary['peaked_frac_1d_all']:.3f}  "
        f"specz={summary['peaked_frac_1d_specz']:.3f}  photoz={summary['peaked_frac_1d_photoz']:.3f}"
    )

    def _fmt(s: dict[str, Any] | None) -> str:
        return (
            "n/a"
            if s is None
            else f"MAP={s['h_refined']:.4f} (grid {s['h_map']:.2f}, railed={s['railed']})"
        )

    print("--- stacked posteriors [conv / sigma_z-aware] ---")
    print(f"  ALL     : {_fmt(stacked['conv']['all'])}")
    print(f"  SPEC-Z  : {_fmt(stacked['conv']['specz_dominated'])}")
    print(f"  PHOTO-Z : {_fmt(stacked['conv']['photoz_dominated'])}")
    print("--- stacked posteriors [1d / delta-z reference] ---")
    print(f"  ALL     : {_fmt(stacked['delta_z_1d']['all'])}")
    print(f"  SPEC-Z  : {_fmt(stacked['delta_z_1d']['specz_dominated'])}")
    print(f"  PHOTO-Z : {_fmt(stacked['delta_z_1d']['photoz_dominated'])}")
    print(f"truth h={B.TRUE_H}   saved -> {OUT_JSON}")
    return payload


if __name__ == "__main__":
    _max = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(_max)
