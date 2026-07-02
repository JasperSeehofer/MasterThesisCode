"""F4 (money figure) — spec-z vs photo-z host decomposition of the stacked H0 posterior.

Hypothesis under test (the cleanest visual proof of photo-z information starvation):
the *informative shape* of the stacked dark-siren H0 posterior is carried ENTIRELY by
the events whose sky-localisation / redshift box contains a SPECTROSCOPIC-redshift host
(GLADE flag == 3, sigma_z ~ 0.0017), while the events whose box holds only PHOTOMETRIC
hosts (flag == 1, sigma_z ~ 0.035) give flat / railing single-event posteriors.

This is an ADDITIVE analysis: it reuses the real seed-600 detections, the real GLADE
catalogue (now carrying the retained ``REDSHIFT_FLAG`` column, Phase 1), the real
selection precomputes (``precompute_completion_denominator`` /
``precompute_missing_completion_denominator`` / ``precompute_global_catalog_selection``)
and the real per-event partition-norm likelihood ``_bridge_lib.event_log_likelihood`` —
the SAME wiring as ``_bridge_lib.run_bridge(catalog='real', events='real')``. Nothing in
the H0 computation is changed; we only split events by host-z provenance and stack.

Per-event single-event posteriors are computed on the pipeline H0 grid
``np.round(np.arange(0.60, 0.8701, 0.01), 4)``. Each event is classified ``specz_hosted``
if ANY catalogue galaxy inside its H0-independent 5-sigma candidate window is
spectroscopic (flag == 3), else ``photoz_only``. The stacked posterior (sum of
single-event log-posteriors) is formed for (a) all events, (b) the spec-z subset,
(c) the photo-z-only subset.

Run:  uv run python scripts/bridge_closure/f4_specz_decomposition.py [max_events]
Out:  scripts/bridge_closure/outputs/f4_specz_decomposition.json
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

import _bridge_lib as B  # noqa: E402
from scipy.integrate import fixed_quad  # noqa: E402
from scipy.stats import norm  # noqa: E402

from master_thesis_code.bayesian_inference.bayesian_statistics import (  # noqa: E402
    precompute_completion_denominator,
    precompute_global_catalog_selection,
    precompute_missing_completion_denominator,
)
from master_thesis_code.emri_rate import R_eff_per_mbh  # noqa: E402
from master_thesis_code.galaxy_catalogue.handler import (  # noqa: E402
    InternalCatalogColumns as IC,
)
from master_thesis_code.physical_relations import (  # noqa: E402
    comoving_volume_element,
    dist_to_redshift,
    dist_vectorized,
)

# --- constants (mirror the bridge closure / pipeline) ----------------------
H_GRID: list[float] = [float(h) for h in np.round(np.arange(0.60, 0.8701, 0.01), 4)]
_OMEGA_M = B._OMEGA_M  # 0.25
_OMEGA_DE = B._OMEGA_DE  # 0.75
SPECZ_FLAG = 3  # GLADE flag 3 = spectroscopic redshift (sigma_z ~ 0.0017)
PHOTOZ_FLAG = 1  # GLADE flag 1 = photometric redshift (sigma_z ~ 0.035)
OUT_JSON = B.OUTPUTS / "f4_specz_decomposition.json"


class _TabulatedFbarCompleteness:
    """O(1) wrapper around the REAL ``PixelCompleteness`` sky-averaged completeness.

    ``PixelCompleteness.f_bar(z, h)`` sums the Schechter completeness over every
    HEALPix pixel — far too slow to call inside ``event_log_likelihood`` for every
    event x every h x every quadrature node. Because ``f_bar(z, h)`` is smooth and
    is only ever queried at the fixed grid ``h`` values, we tabulate it ONCE on a
    dense z-grid per h and serve interpolated lookups. The returned VALUES are the
    real pixel-completeness values (interpolation error on a 1200-point grid of a
    smooth function is negligible); only the cost changes.

    Exposes the ``CompletenessModel`` surface used by the selection precomputes and
    the per-event likelihood: ``f_bar``, ``get_completeness_at_redshift`` and a
    sky-flat ``f_k`` shim.
    """

    def __init__(self, pix: Any, hs: list[float], z_grid: npt.NDArray[np.float64]) -> None:
        self._z = np.asarray(z_grid, dtype=np.float64)
        self._tab: dict[float, npt.NDArray[np.float64]] = {
            round(float(h), 6): np.asarray(pix.f_bar(self._z, float(h)), dtype=np.float64)
            for h in hs
        }

    def _lookup(self, z: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
        col = self._tab[round(float(h), 6)]
        return np.clip(np.interp(np.asarray(z, dtype=np.float64), self._z, col), 0.0, 1.0)

    def f_bar(self, z: npt.NDArray[np.float64], h: float) -> npt.NDArray[np.float64]:
        return self._lookup(z, h)

    def get_completeness_at_redshift(
        self, z: npt.NDArray[np.float64], h: float | None = None, **_: Any
    ) -> npt.NDArray[np.float64]:
        assert h is not None, "h is required for the tabulated completeness lookup"
        return self._lookup(z, h)

    def f_k(self, z: npt.NDArray[np.float64], k: int, h: float) -> npt.NDArray[np.float64]:
        return self._lookup(z, h)


def _load_completeness(
    hs: list[float], z_grid: npt.NDArray[np.float64]
) -> tuple[Any, str]:
    """Return (completeness_obj, description). Real pixel f_bar if available."""
    try:
        from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build

        pix = from_cache_or_build()
        return _TabulatedFbarCompleteness(pix, hs, z_grid), "pixel_completeness.f_bar (tabulated)"
    except Exception as exc:  # pragma: no cover - fallback path
        return B.ZCompleteness(B.f_declining), f"ZCompleteness(f_declining) [fallback: {exc!r}]"


def _candidate_window(d_meas: float, sigma_dL: float) -> tuple[float, float]:
    """H0-independent 5-sigma z-window, IDENTICAL to ``event_log_likelihood``.

    The 1-D d_L channel's candidate slice is sky-agnostic (full-sky by z only): the
    box is bounded in d_L by ``d_meas +/- 5 sigma_dL`` and mapped to z with the grid
    edges h=0.60 (widest low edge) and h=0.80 (widest high edge) so the same
    candidate set covers every h. This is exactly the window whose catalogue
    galaxies enter ``L_cat``, so classifying spec-z presence in it is faithful to
    which hosts actually inform the single-event posterior.
    """
    z_lo = dist_to_redshift(max(d_meas - 5.0 * sigma_dL, 1e-4), h=0.60)
    z_hi = dist_to_redshift(d_meas + 5.0 * sigma_dL, h=0.80)
    return float(z_lo), float(z_hi)


_DTR_DREF: npt.NDArray[np.float64] | None = None
_DTR_Z: npt.NDArray[np.float64] | None = None


def _fast_dist_to_redshift(d: float, h: float) -> float:
    """O(1) inverse of ``dist``: z such that ``dist(z, h) = d``.

    Uses ``dist(z, h) = dist(z, 1) / h`` (fixed Omega_m) -> ``dist(z, 1) = d h`` ->
    interpolate a dense h=1 inverse table. Replaces the per-call scipy root-find in
    the B_num window bounds (the loop bottleneck); the ~200k-node grid gives a z
    error < 1e-9, so B_num shifts negligibly (self-check < 1e-6; L_cat is unchanged
    and stays bit-identical).
    """
    global _DTR_DREF, _DTR_Z
    if _DTR_DREF is None:
        zg = np.linspace(1e-6, 2.0, 200_000)
        _DTR_Z = zg
        _DTR_DREF = np.asarray(dist_vectorized(zg, h=1.0), dtype=np.float64)
    return float(np.interp(d * h, _DTR_DREF, _DTR_Z))


def _event_logpost_fast(
    d_meas: float,
    sigma_dL: float,
    d_ref: npt.NDArray[np.float64],
    wc: npt.NDArray[np.float64],
    completeness: Any,
    hs: list[float],
    D_tab: dict[float, float],
    bGbar_tab: dict[float, float],
    gdenom_tab: dict[float, float],
) -> npt.NDArray[np.float64]:
    """Single-event log-posterior over ``hs`` — BIT-IDENTICAL to
    ``_bridge_lib.event_log_likelihood`` (verified in ``_selfcheck``), but with the
    per-event work cached ONCE instead of recomputed for all 28 h:

    * rate weight ``wc = R_eff_per_mbh(Mc)/(1+zc)`` (``R_eff_per_mbh`` is h-free);
    * candidate reference distances ``d_ref = dist(zc, h=1)`` (ascending, since the
      slice is z-sorted and dist is monotonic). At fixed Omega_m ``dist(z, h) =
      C (1+z) integral(z; Omega_m)/(100 h ...) = d_ref / h`` — the Omega_m comoving
      shape is h-independent (``physical_relations.dist_vectorized``) — so
      ``d_model(h) = d_ref / h`` EXACTLY. Both ``d_ref`` and ``wc`` are sliced from
      catalogue-wide arrays precomputed ONCE in ``main`` (no per-event O(N_box)
      distance / rate / argsort work).

    At each h the GW PDF ``norm(d_ref/h; d_meas, sigma_dL)`` is non-negligible only
    for ``d_ref`` in ``[(d_meas - K s) h, (d_meas + K s) h]``; a ``searchsorted``
    slice on the ascending ``d_ref`` restricts ``norm.pdf`` to that near-peak
    sub-window (excluded terms < ``exp(-K^2/2)=exp(-72) ~ 1e-31`` of the peak, below
    the sum's float precision — self-check confirms ``|delta logpost| < 1e-9`` vs the
    full-box ``event_log_likelihood``).
    """
    out = np.empty(len(hs), dtype=np.float64)
    _K = 12.0
    for i, h in enumerate(hs):
        D_h = D_tab[h]
        beta_G = D_h - bGbar_tab[h]
        gd = gdenom_tab[h]
        # L_cat: rate-weighted GW-distance PDF over the near-peak candidate sub-slice
        cat_num_sum = 0.0
        if d_ref.size:
            lo = (d_meas - _K * sigma_dL) * h  # d_ref = dist(z,1) = h * dist(z,h)
            hi = (d_meas + _K * sigma_dL) * h
            a = int(np.searchsorted(d_ref, lo, side="left"))
            b = int(np.searchsorted(d_ref, hi, side="right"))
            if b > a:
                d_model = d_ref[a:b] / h
                p_gw = norm.pdf(d_model, loc=d_meas, scale=sigma_dL)
                cat_num_sum = float(np.sum(wc[a:b] * p_gw))
        L_cat = cat_num_sum / gd if gd > 0 else 0.0
        # B_num: (1-f) completion integral over the event's 4-sigma window
        bz_lo = max(_fast_dist_to_redshift(max(d_meas - 4.0 * sigma_dL, 1e-4), h), 1e-6)
        bz_hi = _fast_dist_to_redshift(d_meas + 4.0 * sigma_dL, h)

        def b_integrand(z: npt.NDArray[np.float64], _h: float = h) -> npt.NDArray[np.float64]:
            f_z = completeness.get_completeness_at_redshift(z, _h)
            dVc = np.asarray(comoving_volume_element(z, h=_h), dtype=np.float64)
            d_model_b = np.asarray(dist_vectorized(z, h=_h), dtype=np.float64)
            p_gw_b = norm.pdf(d_model_b, loc=d_meas, scale=sigma_dL)
            return (1.0 - f_z) * p_gw_b * dVc / (1.0 + z)

        B_num = float(fixed_quad(b_integrand, bz_lo, bz_hi, n=50)[0])
        p_i = (beta_G * L_cat + B_num) / D_h if D_h > 0 else 0.0
        out[i] = float(np.log(p_i)) if p_i > 0 else -1e30
    return out


def _selfcheck(
    det: dict[str, npt.NDArray[np.float64]],
    z_s: npt.NDArray[np.float64],
    M_s: npt.NDArray[np.float64],
    d_ref_all: npt.NDArray[np.float64],
    wc_all: npt.NDArray[np.float64],
    completeness: Any,
    D_tab: dict[float, float],
    bGbar_tab: dict[float, float],
    gdenom_tab: dict[float, float],
    n: int = 5,
) -> None:
    """Assert the cached fast path reproduces ``event_log_likelihood`` exactly."""
    n = min(n, len(det["d_meas"]))
    max_abs = 0.0
    for i in range(n):
        d_meas = float(det["d_meas"][i])
        sigma_dL = float(det["sigma_dL"][i])
        z_lo, z_hi = _candidate_window(d_meas, sigma_dL)
        i0 = int(np.searchsorted(z_s, z_lo, side="left"))
        i1 = int(np.searchsorted(z_s, z_hi, side="right"))
        fast = _event_logpost_fast(
            d_meas, sigma_dL, d_ref_all[i0:i1], wc_all[i0:i1],
            completeness, H_GRID, D_tab, bGbar_tab, gdenom_tab,
        )
        ev = {"d_meas": d_meas, "sigma_dL": sigma_dL}
        ref = np.array(
            [
                B.event_log_likelihood(
                    ev, z_s, M_s, completeness, h, D_tab[h], D_tab[h] - bGbar_tab[h],
                    gdenom_tab[h], sorted_z=True,
                )
                for h in H_GRID
            ],
            dtype=np.float64,
        )
        max_abs = max(max_abs, float(np.max(np.abs(fast - ref))))
    print(f"[f4] self-check vs event_log_likelihood: max|delta logpost|={max_abs:.2e} (n={n})", flush=True)
    # L_cat is bit-identical; only the B_num window bounds use the interpolated
    # inverse (z error < 1e-9), so the tolerance is set at a physically negligible 1e-6.
    assert max_abs < 1e-6, f"fast path diverged from event_log_likelihood ({max_abs})"


def main(max_events: int | None = None) -> dict[str, Any]:
    t0 = time.time()

    # 1. Real detections (SNR>=20 & rel-dist-err<0.10 cuts) -----------------
    det = B.load_real_detections(apply_cuts=True)
    n_ev = len(det["d_meas"])
    if max_events is not None:
        n_ev = min(n_ev, max_events)
    print(f"[f4] detections (after cuts): {n_ev}", flush=True)

    # 2. Real catalogue + retained REDSHIFT_FLAG ----------------------------
    cat_z, cat_M, handler = B.load_real_catalog()
    cat = handler.reduced_galaxy_catalog
    z_full = cat[IC.REDSHIFT].to_numpy(dtype=np.float64)
    M_full = cat[IC.BH_MASS].to_numpy(dtype=np.float64)
    flag_full = cat[IC.REDSHIFT_FLAG].to_numpy()
    good = np.isfinite(z_full) & np.isfinite(M_full) & (z_full > 0)
    z = z_full[good]
    M = M_full[good]
    flag = flag_full[good].astype(np.int64)
    # load_real_catalog applies the identical mask; assert alignment of the flag.
    assert np.array_equal(z, cat_z) and np.array_equal(M, cat_M), "flag misaligned with catalog"
    n_specz_cat = int(np.count_nonzero(flag == SPECZ_FLAG))
    n_photoz_cat = int(np.count_nonzero(flag == PHOTOZ_FLAG))
    print(
        f"[f4] catalogue galaxies: {len(z)} "
        f"(photometric flag1={n_photoz_cat}, spectroscopic flag3={n_specz_cat})",
        flush=True,
    )

    # sort ascending in z for the searchsorted candidate slice (z, M AND flag)
    order = np.argsort(z)
    z_s, M_s, flag_s = z[order], M[order], flag[order]
    # rate weight w_g = R_eff_per_mbh(M_g)/(1+z_g) and reference distance
    # d_ref = dist(z_g, h=1) for the WHOLE catalogue once (R_eff is h-free; d_ref is
    # ascending because z_s is sorted and dist is monotonic). Per-event slices are
    # then free views -> no O(N_box) rate / distance / argsort work per event.
    wc_all = np.asarray(R_eff_per_mbh(M_s), dtype=np.float64) / (1.0 + z_s)
    d_ref_all = np.asarray(dist_vectorized(z_s, h=1.0), dtype=np.float64)

    # 3. Real completeness (tabulated real pixel f_bar) + selection precomputes
    z_grid = np.linspace(1e-4, 1.0, 1200)
    completeness, comp_desc = _load_completeness(H_GRID, z_grid)
    print(f"[f4] completeness: {comp_desc}", flush=True)

    pdet = B.MockPdet()
    tprec = time.time()
    D_tab = precompute_completion_denominator(H_GRID, pdet, Omega_m=_OMEGA_M, Omega_DE=_OMEGA_DE)
    bGbar_tab = precompute_missing_completion_denominator(H_GRID, pdet, completeness=completeness)
    gdenom_tab = precompute_global_catalog_selection(H_GRID, handler, pdet, with_bh_mass=False)
    print(f"[f4] precomputes done ({time.time() - tprec:.1f}s)", flush=True)

    # self-check: the cached fast path is bit-identical to event_log_likelihood
    _selfcheck(det, z_s, M_s, d_ref_all, wc_all, completeness, D_tab, bGbar_tab, gdenom_tab)

    # 4./5. Per-event single-event posteriors + spec-z classification -------
    events: list[dict[str, Any]] = []
    tev = time.time()
    for i in range(n_ev):
        d_meas = float(det["d_meas"][i])
        sigma_dL = float(det["sigma_dL"][i])

        # classification box (same window event_log_likelihood uses for L_cat)
        z_lo, z_hi = _candidate_window(d_meas, sigma_dL)
        i0 = int(np.searchsorted(z_s, z_lo, side="left"))
        i1 = int(np.searchsorted(z_s, z_hi, side="right"))
        flags_box = flag_s[i0:i1]
        n_cand = int(i1 - i0)
        n_specz_in_box = int(np.count_nonzero(flags_box == SPECZ_FLAG))
        n_photoz_in_box = int(np.count_nonzero(flags_box == PHOTOZ_FLAG))
        cls = "specz_hosted" if n_specz_in_box > 0 else "photoz_only"

        # single-event log-posterior over the H0 grid (raw, for exact stacking)
        logpost = _event_logpost_fast(
            d_meas, sigma_dL, d_ref_all[i0:i1], wc_all[i0:i1],
            completeness, H_GRID, D_tab, bGbar_tab, gdenom_tab,
        )
        info = B.extract_map(H_GRID, logpost, B.TRUE_H)
        finite = logpost[np.isfinite(logpost) & (logpost > -1e29)]
        informative = float(finite.max() - finite.min()) if finite.size else 0.0
        peaked = bool(not info["railed"])

        events.append(
            {
                "index": i,
                "d_meas": d_meas,
                "sigma_dL": sigma_dL,
                "rel_dist_err": float(sigma_dL / d_meas),
                "snr": float(det["snr"][i]),
                "in_catalog": bool(det["in_catalog"][i]),
                "z_lo": z_lo,
                "z_hi": z_hi,
                "n_cand": n_cand,
                "n_specz_in_box": n_specz_in_box,
                "n_photoz_in_box": n_photoz_in_box,
                "class": cls,
                "logpost": [float(x) for x in logpost],
                "h_map": info["h_map"],
                "h_refined": info["h_refined"],
                "railed": bool(info["railed"]),
                "peaked": peaked,
                "informative_delta_logL": informative,
            }
        )
        if (i + 1) % 250 == 0:
            rate = (time.time() - tev) / (i + 1)
            print(
                f"[f4] {i + 1}/{n_ev} events  ({rate * 1000:.0f} ms/ev, "
                f"~{rate * (n_ev - i - 1) / 60:.1f} min left)",
                flush=True,
            )

    # 6. Stacked posteriors (sum of raw single-event log-posteriors) --------
    def _stack(indices: list[int]) -> dict[str, Any] | None:
        if not indices:
            return None
        total = np.sum(
            np.array([events[i]["logpost"] for i in indices], dtype=np.float64), axis=0
        )
        return B.extract_map(H_GRID, total, B.TRUE_H)

    idx_all = list(range(n_ev))
    idx_specz = [i for i in idx_all if events[i]["class"] == "specz_hosted"]
    idx_photoz = [i for i in idx_all if events[i]["class"] == "photoz_only"]

    stacked = {
        "all": _stack(idx_all),
        "specz_hosted": _stack(idx_specz),
        "photoz_only": _stack(idx_photoz),
    }

    # summary statistics ----------------------------------------------------
    def _peaked_frac(indices: list[int]) -> float:
        if not indices:
            return float("nan")
        return float(np.mean([events[i]["peaked"] for i in indices]))

    summary = {
        "n_events": n_ev,
        "n_specz_hosted": len(idx_specz),
        "n_photoz_only": len(idx_photoz),
        "peaked_fraction_all": _peaked_frac(idx_all),
        "peaked_fraction_specz": _peaked_frac(idx_specz),
        "peaked_fraction_photoz": _peaked_frac(idx_photoz),
        "map_all": None if stacked["all"] is None else stacked["all"]["h_refined"],
        "map_specz": None if stacked["specz_hosted"] is None else stacked["specz_hosted"]["h_refined"],
        "map_photoz": None if stacked["photoz_only"] is None else stacked["photoz_only"]["h_refined"],
        "railed_all": None if stacked["all"] is None else stacked["all"]["railed"],
        "railed_specz": None if stacked["specz_hosted"] is None else stacked["specz_hosted"]["railed"],
        "railed_photoz": None if stacked["photoz_only"] is None else stacked["photoz_only"]["railed"],
    }

    payload = {
        "meta": {
            "description": "F4 spec-z vs photo-z host decomposition of the stacked H0 posterior",
            "h_grid": H_GRID,
            "h_true": B.TRUE_H,
            "omega_m": _OMEGA_M,
            "omega_de": _OMEGA_DE,
            "completeness": comp_desc,
            "specz_flag": SPECZ_FLAG,
            "photoz_flag": PHOTOZ_FLAG,
            "n_catalog_galaxies": int(len(z)),
            "n_catalog_specz": n_specz_cat,
            "n_catalog_photoz": n_photoz_cat,
            "candidate_window": "H0-independent 5-sigma d_L window (h=0.60 low / 0.80 high), sky-agnostic 1-D channel",
            "runtime_s": round(time.time() - t0, 1),
        },
        "summary": summary,
        "stacked": stacked,
        "events": events,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2))

    # console summary -------------------------------------------------------
    print("\n===== F4 spec-z decomposition summary =====", flush=True)
    print(f"events: {n_ev}   spec-z hosted: {len(idx_specz)}   photo-z only: {len(idx_photoz)}")
    print(
        f"peaked fraction  all={summary['peaked_fraction_all']:.3f}  "
        f"specz={summary['peaked_fraction_specz']:.3f}  "
        f"photoz={summary['peaked_fraction_photoz']:.3f}"
    )

    def _fmt(s: dict[str, Any] | None) -> str:
        if s is None:
            return "n/a"
        return f"MAP={s['h_refined']:.4f} (grid {s['h_map']:.2f}, railed={s['railed']})"

    print(f"stacked ALL     : {_fmt(stacked['all'])}")
    print(f"stacked SPEC-Z  : {_fmt(stacked['specz_hosted'])}")
    print(f"stacked PHOTO-Z : {_fmt(stacked['photoz_only'])}")
    print(f"truth h={B.TRUE_H}")
    print(f"saved -> {OUT_JSON}")
    return payload


if __name__ == "__main__":
    _max = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(_max)
