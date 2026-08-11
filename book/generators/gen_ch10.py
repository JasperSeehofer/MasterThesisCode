"""Generator for Chapter 10 — "Is It Calibrated?".

Produces the two data files behind the chapter's interactives.

``book/site/data/ch10_pp.json``  (I10.1 "The P-P Slot Machine")
    The independent P-P / coverage harness (``darksiren_emri/validation/
    pp_coverage.py`` — pure numpy/scipy, deliberately importing NONE of the
    production inference code) re-run over the archived
    ``results/pp_coverage_*`` configurations, at full fidelity:

      * every cell is re-run from its OWN archived ``config`` block, so the
        seeded RNG stream is identical and the recomputation must reproduce
        the archived ``coverage`` / ``map_bias`` / ``rail_fraction`` /
        ``completion_fraction`` **exactly**.  That equality is a hard gate:
        a mismatch raises rather than being written out.
      * the re-run additionally records, per realization, the *continuous*
        credible level at which truth enters the HPD region,

            L_i  =  integral over { h : p(h) >= p(h_true) } of p dh,

        so that ``truth in HPD_q  <=>  L_i <= q``.  The empirical CDF of
        ``L_i`` over realizations IS the P-P curve.  The archives store
        coverage at three levels only (50/68/90); the curve is the same
        object sampled on a 41-point level grid.

    Cells: kernel ``volume``, ``mixture_mode two_branch``, sigma_z in
    {0.015, 0.035} x z_support in {1.0, 0.5, 0.43, 0.41, 0.39, 0.38, 0.3,
    0.2} x injected truth in {0.62, 0.72, 0.84} — a completion-fraction
    ladder from 0.0 (the untruncated control) to 0.85.  That ladder is the
    measurement behind claim C11.

    Also carried, read straight from the archives (NOT re-run — no P-P curve
    is needed for them and the n=4000 cells are expensive):

      * the n_events scaling ladder (250 / 1000 / 4000) that separates a real
        asymptotic bias from finite-sample scatter;
      * the 2026-07-03 sigma_z scan, bare vs volume host-z kernel, which is
        the defect the harness was built to expose.

``book/site/data/ch10_closure.json``  (I10.2 "What Did the Closure Test
Actually Test?")
    The idealized campaign-#51 baseline for seed 61000, from the CANONICAL
    posterior directory ``run_seed61000/posteriors_fixed`` (plain
    ``posteriors/`` is the stale pre-``ec09ed0`` backup — BOOK_DESIGN §4.2
    rule 1).  Per-event log-likelihood rows for the 60 highest-curvature
    events plus the aggregated remainder, so the browser can delete the
    top-K events and re-combine in log space exactly.

    Event ranking and the class decomposition use the 3-point curvature at
    h in {0.725, 0.730, 0.735} — the interior of the 0.005-uniform stretch of
    the h-grid, so no second difference crosses the 0.65 / 0.80 spacing seams
    (BOOK_DESIGN §4.2 rule 3).  This is the same statistic, on the same three
    grid points, as ``idealization_audit/audit_information_decomposition.py``,
    and the generator gates against that script's published output.

Determinism
-----------
No unseeded RNG anywhere: the harness cells are re-run from their archived
``seed`` field, and everything else is read from committed artifacts.
Read-only outside ``book/``.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch10.py

Runtime ~70 s (the 16 re-run harness cells dominate).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from darksiren_emri.constants import H as H_TRUE  # noqa: E402
from darksiren_emri.validation import pp_coverage as pc  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
OUT_PP = OUT_DIR / "ch10_pp.json"
OUT_CLOSURE = OUT_DIR / "ch10_closure.json"

# --- repo-relative artifact paths (§4.2 rule 7; never absolute) ------------
CAMPAIGN_REL = Path("results/campaign51_20260728")
IDEAL_POST_REL = CAMPAIGN_REL / "run_seed61000" / "posteriors_fixed"
IDEAL_COMBINED_REL = IDEAL_POST_REL / "combined_posterior.json"
# The realistic run reuses seed 61000's CRB table verbatim (RATIFY-R7: "every
# waveform, SNR, Fisher matrix and CRB CSV ... reused as-is"), so it is the
# index-aligned class/SNR table for the idealized posteriors as well.  The
# idealized run's own prepared_fixed.csv is gitignored bulk.
CRB_REL = CAMPAIGN_REL / "realistic_20260729" / "seed61000" / "prepared_cramer_rao_bounds.csv"
AUDIT_REL = CAMPAIGN_REL / "idealization_audit"

PP_DEEP_2010 = Path("results/pp_coverage_deepvenue_20260710")
PP_DEEP_2030 = Path("results/pp_coverage_deepvenue_20260730")
PP_NOISE = Path("results/pp_coverage_noisemodel_20260711")
PP_EXACT = Path("results/pp_coverage_exactmode_20260711")
PP_SZSCAN = Path("results/pp_coverage_sigmaz_scan_20260703")

EVENT_889 = 889  # the book's running example (pedagogy B4)

# The 41-point production h-grid is non-uniform: 0.01 on [0.60,0.65] and
# [0.80,0.86], 0.005 on [0.65,0.80].  These three sit inside the uniform
# stretch, so the 3-point second difference is legal.
CURV_TAGS = ("725", "73", "735")
CURV_STEP = 0.005

# Published numbers this generator must reproduce, or stop.
# IDEALIZED_BASELINE_READOUT.md:25-30 + idealization_audit/IDEALIZATION_LEDGER.md §1
GATE_N_EVENTS = 1588
GATE_MAP_H = 0.73
GATE_CURV_INCAT = 241.3
GATE_CURV_DARK = -3.0
GATE_SIGMA_H0 = 0.032  # km/s/Mpc, quoted to 2 s.f.
GATE_N_INCAT = 76

# P-P level grid (the curve's x axis).
PP_LEVELS = np.round(np.linspace(0.025, 1.0, 40), 6)

# The cells re-run for I10.1: (archive file, label).
PP_CELLS: list[tuple[Path, str]] = [
    (PP_DEEP_2010 / "pp_zs1.0_sz0.015_volume.json", "control"),
    (PP_DEEP_2010 / "pp_zs1.0_sz0.035_volume.json", "control"),
    (PP_DEEP_2010 / "pp_zs0.5_sz0.015_volume.json", "inert"),
    (PP_DEEP_2010 / "pp_zs0.5_sz0.035_volume.json", "inert"),
    (PP_DEEP_2030 / "pp_zs0.43_sz0.015_volume.json", "c11"),
    (PP_DEEP_2030 / "pp_zs0.43_sz0.035_volume.json", "c11"),
    (PP_DEEP_2030 / "pp_zs0.41_sz0.015_volume.json", "c11"),
    (PP_DEEP_2030 / "pp_zs0.41_sz0.035_volume.json", "c11"),
    (PP_DEEP_2030 / "pp_zs0.39_sz0.015_volume.json", "c11"),
    (PP_DEEP_2030 / "pp_zs0.39_sz0.035_volume.json", "c11"),
    (PP_DEEP_2030 / "pp_zs0.38_sz0.015_volume.json", "c11"),
    (PP_DEEP_2030 / "pp_zs0.38_sz0.035_volume.json", "c11"),
    (PP_DEEP_2010 / "pp_zs0.3_sz0.015_volume.json", "deep"),
    (PP_DEEP_2010 / "pp_zs0.3_sz0.035_volume.json", "deep"),
    (PP_DEEP_2010 / "pp_zs0.2_sz0.015_volume.json", "deep"),
    (PP_DEEP_2010 / "pp_zs0.2_sz0.035_volume.json", "deep"),
]

# Read-only archive rows (no re-run needed): the n_events ladder.
N_SCALING: list[tuple[str, int, Path]] = [
    ("const-sigma", 250, PP_EXACT / "pp_exact_zs0.3_sz0.035.json"),
    ("const-sigma", 1000, PP_NOISE / "pp_nscale_constsig_n1000.json"),
    ("const-sigma", 4000, PP_NOISE / "pp_nscale_constsig_n4000.json"),
    ("model-sigma+pdet", 1000, PP_NOISE / "pp_nscale_modelsigpdet_n1000.json"),
    ("model-sigma+pdet", 4000, PP_NOISE / "pp_nscale_modelsigpdet_n4000.json"),
]

# Read-only archive rows: the bare-vs-volume host-z kernel scan.
KERNEL_SCAN: list[tuple[str, float, Path]] = [
    ("bare", 0.10, PP_SZSCAN / "pp_sigmaz0.10_bare.json"),
    ("volume", 0.10, PP_SZSCAN / "pp_sigmaz0.10_volume.json"),
    ("bare", 0.15, PP_SZSCAN / "pp_sigmaz0.15_bare.json"),
    ("volume", 0.15, PP_SZSCAN / "pp_sigmaz0.15_volume.json"),
    ("bare", 0.25, PP_SZSCAN / "pp_sigmaz0.25_bare.json"),
    ("volume", 0.25, PP_SZSCAN / "pp_sigmaz0.25_volume.json"),
]


def _resolve(rel: Path) -> Path:
    """Locate a read-only artifact without hardcoding a machine path.

    Most of what this chapter reads is git-tracked and present in any checkout
    of this branch.  ``pp_coverage_deepvenue_20260730`` (the C11 window cells)
    is untracked bulk that lives only in the main checkout's working tree, so
    resolution falls back to a sibling ``MasterThesisCode`` directory — the
    same convention ``gen_ch04.py`` uses for the injection pool.
    """
    for root in (REPO_ROOT, REPO_ROOT.parent / "MasterThesisCode"):
        candidate = root / rel
        if candidate.exists():
            return candidate
    msg = (
        f"Required read-only artifact not found: {rel}\n"
        f"  looked in {REPO_ROOT} and {REPO_ROOT.parent / 'MasterThesisCode'}"
    )
    raise FileNotFoundError(msg)


def _r(x: Any, sig: int = 8) -> float:
    """Round to `sig` significant digits (JSON size hygiene)."""
    v = float(x)
    if v == 0.0 or not np.isfinite(v):
        return v
    return float(f"%.{sig}g" % v)


def _rl(a: Any, sig: int = 8) -> list[float]:
    return [_r(v, sig) for v in np.asarray(a, dtype=np.float64).ravel()]


# ---------------------------------------------------------------------------
# I10.1 — re-run the harness, keeping the per-realization credible level
# ---------------------------------------------------------------------------
def _config_from_archive(cfg: dict[str, Any]) -> pc.PPCoverageConfig:
    """Rebuild the exact archived config (fields the current dataclass knows)."""
    names = {f.name for f in fields(pc.PPCoverageConfig)}
    unknown = sorted(k for k in cfg if k not in names)
    if unknown:
        msg = (
            f"Archived config carries fields this build of PPCoverageConfig does not "
            f"know: {unknown} — the harness has changed shape; STOP rather than "
            f"silently re-running a different estimator."
        )
        raise RuntimeError(msg)
    return pc.PPCoverageConfig(**{k: v for k, v in cfg.items() if k in names})


def _credible_level_of_truth(h_grid: np.ndarray, post: np.ndarray, h_true: float) -> float:
    """Smallest HPD credible level whose region contains ``h_true``.

    ``L = int_{p >= p(h_true)} p dh``; truth is inside HPD_q iff ``L <= q``.
    Uses the same trapezoid-free ``p * gradient(h)`` mass convention as the
    harness's own ``_hpd_contains`` so the two agree by construction.
    """
    dh = np.gradient(h_grid)
    mass = post * dh
    p_true = float(np.interp(h_true, h_grid, post))
    return float(mass[post >= p_true].sum())


def rerun_cell(archive: Path) -> dict[str, Any]:
    """Re-run one archived harness configuration, gating on bit-equality."""
    stored = json.loads(_resolve(archive).read_text())
    config = _config_from_archive(stored["config"])
    h_grid = config.h_grid()

    # --- the harness's own shared precomputation (mirrors run_coverage) ----
    zint = np.linspace(pc.Z_MIN, pc.Z_MAX_POP, 3000)
    wpop = pc._inference_population_weight(zint, config.inference_wpop_tilt)  # noqa: SLF001
    d_h = np.trapezoid(
        pc.detection_probability(
            pc.comoving_amplitude_of_z(zint)[:, None] / h_grid[None, :],
            config.d50_gpc,
            config.w_pdet_gpc,
        )
        * wpop[:, None],
        zint,
        axis=0,
    )
    log_dh = np.log(d_h)
    if config.mixture_mode != "two_branch" or config.catalogue_mode:
        msg = f"{archive}: only the two_branch continuum cells are re-run here."
        raise RuntimeError(msg)

    master = np.random.default_rng(config.seed)
    out_truths: dict[str, Any] = {}
    levels = {"50": 0.50, "68": 0.68, "90": 0.90}

    for h_true in config.injected_truths:
        cov = {name: 0 for name in levels}
        rail = 0
        maps: list[float] = []
        comp_fracs: list[float] = []
        cred: list[float] = []
        for _ in range(config.n_realizations):
            rng = np.random.default_rng(int(master.integers(1 << 62)))
            log_l, n_zero_host, _lh, _lc, _nh, _nc = pc._run_realization(  # noqa: SLF001
                h_true, h_grid, log_dh, config, rng
            )
            comp_fracs.append(n_zero_host / config.n_events)
            post = np.exp(log_l - log_l.max())
            post /= np.trapezoid(post, h_grid)
            mi = int(np.argmax(post))
            maps.append(float(h_grid[mi]))
            if mi == 0 or mi == h_grid.size - 1:
                rail += 1
            for name, lv in levels.items():
                if pc._hpd_contains(h_grid, post, h_true, lv):  # noqa: SLF001
                    cov[name] += 1
            cred.append(_credible_level_of_truth(h_grid, post, h_true))

        n = config.n_realizations
        maps_a = np.asarray(maps)
        cred_a = np.asarray(cred)
        recomputed = {
            "coverage": {name: cov[name] / n for name in levels},
            "rail_fraction": rail / n,
            "map_mean": float(maps_a.mean()),
            "map_std": float(maps_a.std()),
            "map_bias": float(maps_a.mean()) - h_true,
            "completion_fraction": float(np.mean(comp_fracs)),
        }

        # ---- HARD GATE: bit-equality against the archived measurement -----
        arch = stored["results"][f"{h_true:.4f}"]
        for key in ("coverage", "rail_fraction", "map_mean", "map_bias", "completion_fraction"):
            if arch.get(key) != recomputed[key]:
                msg = (
                    f"{archive} h_true={h_true}: re-run disagrees with the archive on "
                    f"'{key}': archived {arch.get(key)!r} vs recomputed "
                    f"{recomputed[key]!r} — STOP and flag; do not reconcile silently."
                )
                raise RuntimeError(msg)

        pp_curve = [float((cred_a <= q).mean()) for q in PP_LEVELS]
        out_truths[f"{h_true:.4f}"] = {
            "h_true": h_true,
            "coverage": {k: _r(v, 6) for k, v in recomputed["coverage"].items()},
            "rail_fraction": _r(recomputed["rail_fraction"], 4),
            "map_mean": _r(recomputed["map_mean"], 7),
            "map_std": _r(recomputed["map_std"], 4),
            "map_bias": _r(recomputed["map_bias"], 4),
            "completion_fraction": _r(recomputed["completion_fraction"], 5),
            "pp_curve": [_r(v, 4) for v in pp_curve],
            "maps": _rl(maps_a, 5),
            "archive_match": True,
        }

    return {
        "sigma_z": config.sigma_z,
        "z_support": config.z_support if config.z_support is not None else 1.0,
        "n_realizations": config.n_realizations,
        "n_events": config.n_events,
        "kernel": config.kernel,
        "mixture_mode": config.mixture_mode,
        "h_min": config.h_min,
        "h_max": config.h_max,
        "h_step": config.h_step,
        "seed": config.seed,
        "archive": str(archive),
        "truths": out_truths,
    }


def _archive_rows(spec: list[tuple[Any, Any, Path]], keys: tuple[str, str]) -> list[dict[str, Any]]:
    """Read (not re-run) coverage/bias rows out of archived harness JSONs."""
    rows: list[dict[str, Any]] = []
    for a, b, path in spec:
        stored = json.loads(_resolve(path).read_text())
        for h_key, res in sorted(stored["results"].items()):
            rows.append(
                {
                    keys[0]: a,
                    keys[1]: b,
                    "h_true": _r(res["h_true"], 4),
                    "coverage": {k: _r(v, 5) for k, v in res["coverage"].items()},
                    "map_bias": _r(res["map_bias"], 4),
                    "map_std": _r(res["map_std"], 4),
                    "rail_fraction": _r(res["rail_fraction"], 4),
                    "sigma_z": _r(stored["config"]["sigma_z"], 4),
                    "kernel": stored["config"]["kernel"],
                    "n_events": stored["config"]["n_events"],
                    "n_realizations": stored["config"]["n_realizations"],
                    "archive": str(path),
                }
            )
    return rows


def _monotone_report(ladder: list[dict[str, Any]], sigma_z: float) -> dict[str, Any]:
    """C11 asserts monotonicity in completion fraction; measure it, don't assume it.

    Reported as Spearman-free order statistics: the number of adjacent pairs
    (sorted by completion fraction, at fixed sigma_z) whose bias decreases, and
    whether any sign flip occurs above the control level.
    """
    sel = sorted(
        (r for r in ladder if abs(r["sigma_z"] - sigma_z) < 1e-12),
        key=lambda r: r["completion_fraction"],
    )
    bs = [r["map_bias"] for r in sel]
    descents = sum(1 for a, b in zip(bs[:-1], bs[1:], strict=True) if b < a)
    return {
        "n_points": len(bs),
        "n_adjacent_descents": descents,
        "bias_first": _r(bs[0], 4),
        "bias_last": _r(bs[-1], 4),
        "sign_flips_above_comp_frac_0.05": sum(
            1
            for a, b in zip(sel[:-1], sel[1:], strict=True)
            if a["completion_fraction"] > 0.05 and (a["map_bias"] > 0) != (b["map_bias"] > 0)
        ),
        "note": (
            "Pooled over the three injected truths, so adjacent points can differ in "
            "h_true as well as in completion fraction; a few local descents are "
            "expected and are NOT a refutation of C11's monotonicity statement."
        ),
    }


def build_pp_payload() -> dict[str, Any]:
    cells = []
    for archive, family in PP_CELLS:
        t0 = time.time()
        cell = rerun_cell(archive)
        cell["family"] = family
        cells.append(cell)
        print(
            f"  re-ran {archive.name}: sigma_z={cell['sigma_z']} "
            f"z_support={cell['z_support']} — archive gate PASS ({time.time() - t0:.1f} s)"
        )

    # C11's own ladder: every (cell, truth) as (completion_fraction, bias).
    ladder = sorted(
        (
            {
                "completion_fraction": t["completion_fraction"],
                "map_bias": t["map_bias"],
                "coverage68": t["coverage"]["68"],
                "sigma_z": c["sigma_z"],
                "h_true": t["h_true"],
                "z_support": c["z_support"],
            }
            for c in cells
            for t in c["truths"].values()
        ),
        key=lambda r: (r["completion_fraction"], r["sigma_z"], r["h_true"]),
    )
    in_window = [r for r in ladder if 0.008 <= r["completion_fraction"] <= 0.234]
    biases = [r["map_bias"] for r in in_window]

    # ---- C11's two quoted sub-bands, recomputed ---------------------------
    # ⚠ The LOWER endpoints reproduce exactly; the UPPER endpoints do not.
    # Recorded, not reconciled: book/design/flags/ch10_FLAGS.md.
    def _band(lo: float, hi: float) -> dict[str, Any]:
        sel = [r for r in ladder if lo <= r["completion_fraction"] <= hi]
        bs = [r["map_bias"] for r in sel]
        return {
            "comp_frac_range": [lo, hi],
            "recomputed_min": _r(min(bs), 4),
            "recomputed_max": _r(max(bs), 4),
            "n_cells": len(sel),
        }

    band_a = _band(0.06, 0.09)
    band_b = _band(0.13, 0.24)
    band_a["claim_min"], band_a["claim_max"] = 0.0008, 0.0097
    band_b["claim_min"], band_b["claim_max"] = 0.0034, 0.0181
    band_disagreement = {
        "status": "OPEN — recorded, not reconciled",
        "what": (
            "C11 quotes bias +0.0008..+0.0097 (comp_frac 0.06-0.09) and "
            "+0.0034..+0.0181 (0.13-0.24). Re-running the archived "
            "pp_coverage_deepvenue_20260730 / _20260710 two_branch cells that span "
            "exactly that comp_frac window reproduces the LOWER endpoints exactly "
            "(+0.0008, +0.0034) but gives smaller UPPER endpoints."
        ),
        "claim_source": "CLAIM_2D_BIAS_20260730.md C11 / ADJUDICATION_20260730.md §1",
        "recompute_source": (
            "archive-gated re-run of pp_coverage_deepvenue_20260730 + _20260710 "
            "(two_branch, volume kernel); the archives' own .log files carry the "
            "same numbers this generator recomputes"
        ),
        "note": (
            "+0.00963 does appear in results/pp_fullpower_20260727 "
            "(pp_cat_lcat_zs0.43_sky1e-4_h0.84, comp_frac 0.0847) — a DIFFERENT "
            "harness family (catalogue_mode impostor ball, mixture_mode 'lcat'), so "
            "the claim's band may pool two harness families. No archived cell "
            "anywhere in results/pp_* reproduces +0.0181 inside comp_frac 0.13-0.24. "
            "The qualitative verdict is unaffected in either direction: every "
            "candidate maximum (0.0157 or 0.0181) is far below +0.077."
        ),
        "flag_file": "book/design/flags/ch10_FLAGS.md",
    }

    return {
        "chapter": "ch10",
        "h_true_project": float(H_TRUE),
        "pp_levels": _rl(PP_LEVELS, 4),
        "cells": cells,
        "ladder": [
            {
                "completion_fraction": r["completion_fraction"],
                "map_bias": r["map_bias"],
                "coverage68": r["coverage68"],
                "sigma_z": r["sigma_z"],
                "h_true": r["h_true"],
                "z_support": r["z_support"],
            }
            for r in ladder
        ],
        "c11_window": {
            "comp_frac_min": _r(min(r["completion_fraction"] for r in in_window), 4),
            "comp_frac_max": _r(max(r["completion_fraction"] for r in in_window), 4),
            "bias_min": _r(min(biases), 4),
            "bias_max": _r(max(biases), 4),
            "n_cells": len(in_window),
            "target_2d_bias": 0.077,
            "ratio_at_recomputed_max": _r(0.077 / max(biases), 4),
            "bands": {"A": band_a, "B": band_b},
            "band_disagreement": band_disagreement,
            "claim": (
                "CLAIM_2D_BIAS_20260730.md C11: bias +0.0008..+0.0097 at comp_frac "
                "0.06-0.09 and +0.0034..+0.0181 at 0.13-0.24; monotone across "
                "0.008-0.85, no sign flip, control-consistent at zero; 6-16x below "
                "+0.077 => REFUTED as the 2D owner. Harness is 1D-only by construction."
            ),
            "monotone_check": {
                "sigma_z_0.015": _monotone_report(ladder, 0.015),
                "sigma_z_0.035": _monotone_report(ladder, 0.035),
            },
        },
        "n_scaling": _archive_rows(N_SCALING, ("variant", "n_events_cell")),
        "kernel_scan": _archive_rows(KERNEL_SCAN, ("kernel_label", "sigma_z_cell")),
        "source": {
            "harness": "darksiren_emri/validation/pp_coverage.py",
            "independence": (
                "pure numpy/scipy; imports NO production inference code "
                "(module docstring, 'Scientific independence')"
            ),
            "method": (
                "Every cell is re-run from its archived config block (same seed, same "
                "RNG stream) and gated to bit-equality on coverage/map_bias/"
                "rail_fraction/completion_fraction. The P-P curve is the empirical CDF "
                "of L_i = int_{p >= p(h_true)} p dh over realizations; truth is inside "
                "HPD_q iff L_i <= q."
            ),
        },
    }


# ---------------------------------------------------------------------------
# I10.2 — the idealized closure, event by event
# ---------------------------------------------------------------------------
def _load_ideal_posteriors() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load the canonical idealized per-event likelihood rows on the h-grid."""
    combined = json.loads(_resolve(IDEAL_COMBINED_REL).read_text())
    h_grid = np.asarray(combined["h_values"], dtype=np.float64)

    per_event: dict[int, list[float]] = {}
    for i, h in enumerate(h_grid):
        tag = f"{h:.10g}".replace("0.", "0_").replace(".", "_")
        path = _resolve(IDEAL_POST_REL / f"h_{tag}.json")
        raw = json.loads(path.read_text())
        for k, v in raw.items():
            if not k.isdigit():
                continue
            val = float(v[0] if isinstance(v, list) else v)
            per_event.setdefault(int(k), [np.nan] * len(h_grid))[i] = val

    rows = {k: np.asarray(v, dtype=np.float64) for k, v in per_event.items()}
    return h_grid, rows


def build_closure_payload() -> dict[str, Any]:
    combined = json.loads(_resolve(IDEAL_COMBINED_REL).read_text())
    h_grid, rows = _load_ideal_posteriors()

    keep = {k: v for k, v in rows.items() if np.all(np.isfinite(v)) and np.all(v > 0.0)}
    n_ev = len(keep)
    if n_ev != combined["n_events_used"] or n_ev != GATE_N_EVENTS:
        msg = (
            f"idealized event count {n_ev} != combined_posterior.json "
            f"n_events_used {combined['n_events_used']} / readout {GATE_N_EVENTS} "
            f"— STOP and flag."
        )
        raise RuntimeError(msg)

    idx = sorted(keep)
    log_rows = np.stack([np.log(keep[k]) for k in idx])  # (n_ev, n_h)
    i73 = int(np.argmin(np.abs(h_grid - GATE_MAP_H)))
    i725, i735 = i73 - 1, i73 + 1
    for i, tag in ((i725, CURV_TAGS[0]), (i73, CURV_TAGS[1]), (i735, CURV_TAGS[2])):
        want = float(f"0.{tag}")
        if abs(h_grid[i] - want) > 1e-12:
            msg = f"curvature stencil node mismatch: grid {h_grid[i]} vs expected {want}"
            raise RuntimeError(msg)
    for a, b in ((i725, i73), (i73, i735)):
        if abs((h_grid[b] - h_grid[a]) - CURV_STEP) > 1e-12:
            msg = "the 3-point curvature stencil straddles an h-grid spacing seam — STOP."
            raise RuntimeError(msg)

    # Signed 3-point curvature, in (dh = 0.005) units — the audit script's statistic.
    curv = 2.0 * log_rows[:, i73] - log_rows[:, i725] - log_rows[:, i735]

    crb = pd.read_csv(_resolve(CRB_REL))
    in_cat = np.array([bool(crb["host_galaxy_index"].iloc[k] >= 0) for k in idx])
    snr = np.array([float(crb["SNR"].iloc[k]) for k in idx])

    curv_incat = float(curv[in_cat].sum())
    curv_dark = float(curv[~in_cat].sum())
    curv_total = curv_incat + curv_dark
    sigma_h = CURV_STEP / np.sqrt(curv_total)
    sigma_h0 = 100.0 * sigma_h

    # ---- HARD GATES against IDEALIZATION_LEDGER.md §1 --------------------
    if int(in_cat.sum()) != GATE_N_INCAT:
        msg = f"in-catalogue count {int(in_cat.sum())} != ledger {GATE_N_INCAT} — STOP."
        raise RuntimeError(msg)
    if abs(curv_incat - GATE_CURV_INCAT) > 0.05 or abs(curv_dark - GATE_CURV_DARK) > 0.05:
        msg = (
            f"curvature decomposition {curv_incat:.1f}/{curv_dark:.1f} disagrees with "
            f"IDEALIZATION_LEDGER.md §1 ({GATE_CURV_INCAT}/{GATE_CURV_DARK}) — STOP."
        )
        raise RuntimeError(msg)
    if abs(sigma_h0 - GATE_SIGMA_H0) > 0.0005:
        msg = f"sigma_H0 {sigma_h0:.4f} disagrees with the ledger's {GATE_SIGMA_H0} — STOP."
        raise RuntimeError(msg)

    order = np.argsort(-curv)
    n_top = 60
    top = order[:n_top]
    rest = order[n_top:]

    # Per-event rows are shipped as differences from their own h=0.73 value:
    # an additive per-event constant cancels from the combined SHAPE, from the
    # MAP and from the curvature, and it keeps the JSON to O(1) magnitudes.
    top_rows = log_rows[top] - log_rows[top][:, [i73]]
    rest_sum = (log_rows[rest] - log_rows[rest][:, [i73]]).sum(axis=0)
    all_sum = top_rows.sum(axis=0) + rest_sum

    # Gate: the K=0 combination must reproduce the published MAP.
    post_all = np.exp(all_sum - all_sum.max())
    map_all = float(h_grid[int(np.argmax(post_all))])
    if abs(map_all - float(combined["map_h"])) > 1e-12:
        msg = (
            f"reconstructed K=0 MAP {map_all} != combined_posterior.json "
            f"{combined['map_h']} — STOP and flag."
        )
        raise RuntimeError(msg)

    # The 3 loudest in-catalogue events (the readout's "golden three").
    incat_idx_sorted = sorted(
        (k for k, ic in zip(idx, in_cat, strict=True) if ic),
        key=lambda k: -float(crb["SNR"].iloc[k]),
    )
    golden3 = incat_idx_sorted[:3]
    pos = {k: i for i, k in enumerate(idx)}
    curv3 = float(sum(curv[pos[k]] for k in golden3))

    # The CRB's sqrt(Sigma_dLdL) is the ABSOLUTE sigma_dL, in the same units as
    # the luminosity_distance column (Gpc) -- NOT a fraction.  Carrying it under
    # a fractional key is exactly the book-wide units slip resolved by
    # REVISION_WORKLIST.md §A-D1 (spec value: sigma_dL/d_L = 8.98e-4 for 889,
    # absolute sigma_dL = 7.98e-5 Gpc).  Both are emitted, each under its own
    # name, so no consumer can pick up the absolute value as a fraction.
    top_meta = []
    for rank, j in enumerate(top):
        k = idx[j]
        sigma_dl_gpc = float(
            np.sqrt(float(crb["delta_luminosity_distance_delta_luminosity_distance"].iloc[k]))
        )
        d_l_gpc = float(crb["luminosity_distance"].iloc[k])
        top_meta.append(
            {
                "rank": rank,
                "event": int(k),
                "curvature": _r(float(curv[j]), 6),
                "SNR": _r(float(snr[j]), 5),
                "in_catalog": bool(in_cat[j]),
                "sigma_dL_Gpc": _r(sigma_dl_gpc, 3),
                "sigma_dL_over_dL": _r(sigma_dl_gpc / d_l_gpc, 3),
            }
        )

    row889 = crb.iloc[EVENT_889]
    sigma_dl_gpc_889 = float(
        np.sqrt(float(row889["delta_luminosity_distance_delta_luminosity_distance"]))
    )
    ev889 = {
        "index": EVENT_889,
        "rank_by_curvature": int(np.where(np.array([idx[j] for j in order]) == EVENT_889)[0][0]),
        "curvature": _r(float(curv[pos[EVENT_889]]), 6),
        "SNR": _r(float(row889["SNR"]), 6),
        "d_L_Gpc": _r(float(row889["luminosity_distance"]), 6),
        "in_catalog": bool(row889["host_galaxy_index"] >= 0),
        "host_galaxy_index": int(row889["host_galaxy_index"]),
        "sigma_dL_Gpc": _r(sigma_dl_gpc_889, 3),
        "sigma_dL_over_dL": _r(sigma_dl_gpc_889 / float(row889["luminosity_distance"]), 3),
        "share_of_total_curvature": _r(float(curv[pos[EVENT_889]] / curv_total), 4),
    }

    return {
        "chapter": "ch10",
        "h_grid": _rl(h_grid, 6),
        "h_true": float(H_TRUE),
        "i_curv": [i725, i73, i735],
        "curv_step": CURV_STEP,
        "n_events": n_ev,
        "n_in_catalog": int(in_cat.sum()),
        "n_dark": int((~in_cat).sum()),
        "n_top": n_top,
        # log-space native; per-event rows offset to 0 at h = 0.73
        "log_rows_top": [_rl(r, 8) for r in top_rows],
        "log_sum_rest": _rl(rest_sum, 10),
        "log_sum_all": _rl(all_sum, 10),
        "top_meta": top_meta,
        "curvature": {
            "total": _r(curv_total, 6),
            "in_catalog": _r(curv_incat, 6),
            "dark": _r(curv_dark, 6),
            "in_catalog_share": _r(curv_incat / curv_total, 5),
            "dark_share": _r(curv_dark / curv_total, 5),
            "sigma_h": _r(float(sigma_h), 5),
            "sigma_H0_km_s_Mpc": _r(float(sigma_h0), 5),
            "units": "signed 3-point second difference of sum_i ln L_i, in (dh=0.005) units",
        },
        "golden3": {
            "events": [int(k) for k in golden3],
            "SNR": [_r(float(crb["SNR"].iloc[k]), 5) for k in golden3],
            "curvature_sum": _r(curv3, 6),
            "share_of_total": _r(curv3 / curv_total, 5),
            "share_of_in_catalog": _r(curv3 / curv_incat, 5),
            "ledger_quotes": 0.46,
            "note": (
                "IDEALIZATION_LEDGER.md §1 quotes '46%' without naming the denominator. "
                "Recomputed: 46.41% of the in-catalogue curvature, 47.00% of the signed "
                "total. Both are carried; neither is silently substituted for the other. "
                "See book/design/flags/ch10_FLAGS.md."
            ),
        },
        "event889": ev889,
        "published": {
            "map_h": float(combined["map_h"]),
            "n_events_used": int(combined["n_events_used"]),
            "n_events_excluded": int(combined["n_events_excluded"]),
            "strategy": combined["strategy"],
            "variant": combined["variant"],
            "zoom_map": 0.72990,
            "zoom_mean": 0.72993,
            "zoom_sigma_h": 0.00030,
            "zoom_pull_sigma": -0.24,
            "seed62000_pull_sigma": -0.36,
            "source": "IDEALIZED_BASELINE_READOUT.md:25-30",
        },
        "source": {
            "posteriors": str(IDEAL_POST_REL),
            "posteriors_note": (
                "CANONICAL directory for seed 61000 is posteriors_fixed; plain "
                "posteriors/ is the stale pre-ec09ed0 backup (BOOK_DESIGN §4.2 rule 1)."
            ),
            "crb": str(CRB_REL),
            "crb_note": (
                "seed 61000's CRB table, reused verbatim by campaign #53 "
                "(realistic_host_observation_model.md §7.1, RATIFY-R7) and therefore "
                "index-aligned with the idealized posteriors."
            ),
            "audit_script": str(AUDIT_REL / "audit_information_decomposition.py"),
            "channel": "1D (posteriors_fixed/, without host BH mass)",
        },
    }


def main() -> None:
    for rel in (IDEAL_COMBINED_REL, CRB_REL):
        _resolve(rel)  # raises with both search roots named if absent

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("I10.2 — idealized closure decomposition")
    closure = build_closure_payload()
    with OUT_CLOSURE.open("w") as f:
        json.dump(closure, f, separators=(",", ":"))
    print(f"Wrote {OUT_CLOSURE} ({OUT_CLOSURE.stat().st_size / 1024:.1f} KB)")
    c = closure["curvature"]
    print(
        f"  {closure['n_events']} events ({closure['n_in_catalog']} in-cat) · "
        f"curvature total {c['total']} = in-cat {c['in_catalog']} + dark {c['dark']} · "
        f"sigma_H0 {c['sigma_H0_km_s_Mpc']} km/s/Mpc · ledger gates PASS"
    )
    print(
        f"  golden 3 {closure['golden3']['events']}: "
        f"{closure['golden3']['share_of_in_catalog']:.4f} of in-cat / "
        f"{closure['golden3']['share_of_total']:.4f} of total (ledger quotes 0.46 — FLAGGED)"
    )

    print("I10.1 — re-running the P-P coverage harness (16 archived cells)")
    pp = build_pp_payload()
    with OUT_PP.open("w") as f:
        json.dump(pp, f, separators=(",", ":"))
    print(f"Wrote {OUT_PP} ({OUT_PP.stat().st_size / 1024:.1f} KB)")
    w = pp["c11_window"]
    print(
        f"  {len(pp['cells'])} cells x 3 truths, all archive-gated · "
        f"C11 window comp_frac {w['comp_frac_min']}-{w['comp_frac_max']}: "
        f"bias {w['bias_min']}..{w['bias_max']} over {w['n_cells']} cells"
    )
    for name, band in w["bands"].items():
        print(
            f"  band {name} comp_frac {band['comp_frac_range']}: recomputed "
            f"{band['recomputed_min']:+.4f}..{band['recomputed_max']:+.4f} vs claim "
            f"{band['claim_min']:+.4f}..{band['claim_max']:+.4f}"
            f"{'' if abs(band['recomputed_max'] - band['claim_max']) < 5e-5 else '  <-- FLAGGED'}"
        )
    print(f"Total runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
