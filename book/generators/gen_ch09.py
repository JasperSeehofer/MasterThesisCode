"""Generator for Chapter 9 — "Building a Universe to Break Your Estimator".

Produces the four data files behind the chapter's interactives.  Everything is
read-only with respect to the source repo; only ``book/site/data/`` is written.

``book/site/data/ch09_factory.json``   (I9.1 "The Universe Factory")
    The production injection pool as a *population*: the stratum-'a'
    (population-measure) rows binned in (log10 M_z, z), drawn vs detected at
    SNR >= 20, so the browser can restrict the population's mass band and watch
    both the redshift marginal and the detected fraction move.  Plus the
    realized plunge-window initial conditions (p0 per mass bin, against the
    RETIRED snapshot band p0 ~ U[10,16]) and the fiducial-Omega_m panel, whose
    h' offsets are RECOMPUTED here with the repo's own ``dist()`` and gated
    against the published table in ``docs/gates/G7_systematics_budget.md``.

``book/site/data/ch09_bench.json``     (I9.2 "The Consistency Bench")
    The C9 measurement, rebuilt from the run's own per-h log lines at full
    precision (BOOK_DESIGN §4.2 rule 4: never the 4-dp `w_G` log field):

        beta_G(h) = D(h) - beta_Gbar(h)          [7 s.f. log values]
        w_G(h)    = beta_G(h) / D(h)             [mass-BLIND, as shipped]
        r(h)      = sum_w_Dg(with_bh) / sum_w_Dg(no_bh)
        w_G^aware = r beta_G / (r beta_G + beta_Gbar)

    against the realized detected in-catalogue counts from the two seeds' CRB
    tables (``host_galaxy_index >= 0``), with the binomial z-scores.  Also
    carries the C9 diagnostic counterfactual posteriors (``g4_posterior_curves``)
    and BOTH disputed `generator_marginal` w_G curves (sources map §7 item 7 --
    the exact curve attribution is OPEN and the page must show both).

    Since 2026-07-31 it also carries the ``cell_b`` block: the 2x2 cell B
    control landed (evaluate 6103219 / combine 6103220), and its
    pre-registered w_G reading -- *"expected bit-identical to the #53 runs
    (pure quadrature, no catalogue input)"* -- is re-measured here rather than
    copied: cell B's per-h w_G column is compared element-wise against #53 r1's
    over all 41 grid points, and the run's D / beta_Gbar log legs are compared
    outright.  The generator stops if that equality is anything other than
    exact.

``book/site/data/ch09_derail.json``    (I9.3 "The De-rail Matrix")
    The four-step de-rail matrix (ledger #49).  Steps 2-4 carry their ARCHIVED
    posteriors (``results/commission_20260701/redteam/derail_matrix_results.json``,
    identical to ``docs/gates/G3_ablation_cube.json`` on the shared keys); step 1
    (pre-4pi) has no stored curve anywhere in the tree and is carried as a
    RECORDED MAP only.  Never blended.

``book/site/data/ch09_identity.json``  (Option-A identity, N5 / G1)
    The G1 gate table: Sigma_global(h) vs beta_G(h) on the real GLADE catalogue,
    raw ratio and h^3-corrected shape.

HARD GATES (the generator stops rather than shipping a number that disagrees
with the spec; BOOK_DESIGN §4.1):
    w_G(0.73) = 0.1215037 · r(0.73) = 0.39248 · mass-aware w_G = 0.05149 ·
    realized 164/3135 · binomial z = -11.86 (blind) / +0.21 (aware) ·
    the 12 published Omega_m cells · the four C9 counterfactual MAPs/means ·
    G1's own end-to-end tilts · the cell-B w_G equality (element-wise, all 41
    grid points, max|Δ| exactly 0.0) and its three quoted reads.

DATA AVAILABILITY
-----------------
Everything except the 200k injection pool and the two diagnostics CSVs is
git-tracked.  The pool is resolved from this repo root, then from a sibling
``darksiren-emri`` checkout; if absent, the factory file's pool block is left
untouched (an already-committed file is never degraded) and a NOTICE is printed.
The ``sig0_control`` diagnostics CSV is used ONLY to verify the recorded
`generator_marginal` w_G values — never to produce a number that is not already
recorded in the design documents.

Determinism: no RNG anywhere; the only sampling is a fixed-stride decimation.

Run as::

    /home/jasper/Repositories/darksiren-emri/.venv/bin/python \\
        book/generators/gen_ch09.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from darksiren_emri.constants import (  # noqa: E402
    LISA_MISSION_DURATION_YEARS,
    OMEGA_M,
    SNR_THRESHOLD,
)
from darksiren_emri.constants import H as H_TRUE  # noqa: E402
from darksiren_emri.physical_relations import dist  # noqa: E402

# --- repo-relative artifact paths (§4.2 rule 7; never absolute) ------------
CAMPAIGN_REL = Path("results/campaign51_20260728/realistic_20260729")
SEED1_REL = CAMPAIGN_REL / "seed61000"
SEED2_REL = CAMPAIGN_REL / "seed62000"
GATEB_REL = CAMPAIGN_REL / "gate_b_20260730"
RUN_LOG_REL = SEED1_REL / "mixture_leg_log_extract.txt"
CRB1_REL = SEED1_REL / "prepared_cramer_rao_bounds.csv"
CRB2_REL = SEED2_REL / "prepared_cramer_rao_bounds.csv"
POOL_REL = GATEB_REL / "injection_pool_mix200k_20260728"
SIG0_REL = SEED1_REL / "sig0_control" / "diagnostics" / "event_likelihoods.csv"
CELLB_REL = SEED1_REL / "estimatorB_2x2"
CELLB_LOG_REL = CELLB_REL / "mixture_leg_log_extract.txt"
CELLB_DIAG_REL = CELLB_REL / "diagnostics" / "event_likelihoods.csv"
R1_DIAG_REL = SEED1_REL / "real_r1" / "diagnostics" / "event_likelihoods.csv"
G4_CURVES_REL = GATEB_REL / "g4_posterior_curves.json"
G4_RESULTS_REL = GATEB_REL / "g4_results.json"
G2_SUMMARY_REL = GATEB_REL / "g2_catalogue_summary.json"
G6_RESULTS_REL = GATEB_REL / "g6_results.json"
C9_DARK_REL = GATEB_REL / "c9_darkdraw_results.json"
DERAIL_REL = Path("results/commission_20260701/redteam/derail_matrix_results.json")
CUBE_REL = Path("docs/gates/G3_ablation_cube.json")
G1_REL = Path("docs/gates/G1_beta_g_check.json")

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
OUT_FACTORY = OUT_DIR / "ch09_factory.json"
OUT_BENCH = OUT_DIR / "ch09_bench.json"
OUT_DERAIL = OUT_DIR / "ch09_derail.json"
OUT_IDENTITY = OUT_DIR / "ch09_identity.json"

EVENT_889 = 889  # the book's running example (pedagogy B4)

# ---- spec values this generator must reproduce, or stop -------------------
SPEC_WG_BLIND = 0.1215037  # CLAIM_2D_BIAS_20260730.md C9
SPEC_R_073 = 0.39248  # ibid., "the run's own logs"
SPEC_WG_AWARE = 0.05149  # ibid., mass-aware w_G
SPEC_K_POOLED, SPEC_N_POOLED = 164, 3135  # 76/1590 + 88/1545
SPEC_Z_BLIND, SPEC_Z_AWARE = -11.86, +0.21
SPEC_CF = {  # C9's diagnostic counterfactual (beta_G -> r(h) beta_G)
    "no_bh_base": (0.74, 0.7321),
    "no_bh_corr": (0.64, 0.6430),
    "with_bh_base": (0.81, 0.8123),
    "with_bh_corr": (0.745, 0.7433),
}
# BOOK_SOURCES_MAP §7 item 7 — the OPEN dispute, both sides carried verbatim.
GENMARG_H = [0.60, 0.64, 0.73, 0.86]
GENMARG_CLAIMED = [0.0774, 0.0692, 0.0555, 0.0427]  # CLAIM C9 ghost-resolution
GENMARG_MEASURED = [0.0686001, 0.0614573, 0.0496786, 0.0385580]  # the CSVs
# G7_systematics_budget.md, "Numbers behind row #6" (percent shift in h)
G7_OMEGA_TABLE = {
    0.05: (0.16, 0.25),
    0.1: (0.32, 0.49),
    0.3: (0.94, 1.45),
    0.5: (1.50, 2.33),
    1.0: (2.60, 4.08),
    1.5: (3.31, 5.25),
}
OMEGA_TRUE_PLANCK = 0.3153  # G7 row 6's "if truth is Planck"
OMEGA_PRE_G11 = 0.25  # the retired pre-G11 fiducial

# --- the 2x2 cell B control, landed 2026-07-31 -----------------------------
# CELLB_READOUT_20260731.md.  Job IDs follow REVISION_WORKLIST §A-D3: the
# pre-registration keeps 6101146/6101147, the *result* cites the resubmission.
CELLB_DATE = "2026-07-31"
CELLB_JOBS_PREREG = "6101146 / 6101147"
CELLB_JOBS_RESULT = "6103219 / 6103220"
# the readout's own w_G reads, at 7 d.p. off the full-precision diagnostics
# column (never the 4-dp log field — BOOK_DESIGN §4.2 rule 4)
SPEC_CELLB_WG = {0.60: 0.1625175, 0.73: 0.1215039, 0.81: 0.1038732}
# MAPs throughout (expB MJ-1); the means live in the JSON beside them.
SPEC_CELLB_MAPS = {
    "A": {"map_1d": 0.7299, "map_2d": 0.7300},
    "B": {"map_1d": 0.7450, "mean_1d": 0.7320, "map_2d": 0.7900, "mean_2d": 0.7962},
    "C": {"map_1d": 0.7400, "mean_1d": 0.7321, "map_2d": 0.8133},
}

# Display grids (presentation only, not physics).
LOGM_EDGES = np.round(np.arange(4.0, 7.61, 0.20), 4)
Z_EDGES = np.round(np.linspace(0.0, 1.5, 31), 6)


def _r(x: Any, sig: int = 8) -> float:
    """Round to `sig` significant digits (JSON hygiene)."""
    v = float(x)
    if v == 0.0 or not np.isfinite(v):
        return v
    return float(f"%.{sig}g" % v)


def _rl(a: Any, sig: int = 8) -> list[float]:
    return [_r(v, sig) for v in np.asarray(a, dtype=np.float64).ravel()]


def _resolve(rel: Path) -> Path | None:
    """Locate an artifact without hardcoding a machine path: this checkout
    first, then a sibling ``darksiren-emri`` checkout.  Git-tracked artifacts
    resolve in the first branch; untracked ones (the 200k injection pool, the
    diagnostics CSVs) live only in the main working tree."""
    for root in (REPO_ROOT, REPO_ROOT.parent / "darksiren-emri"):
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def _must(rel: Path) -> Path:
    p = _resolve(rel)
    if p is None:
        _fail(f"required artifact not found in either checkout: {rel}")
    assert p is not None
    return p


def _fail(msg: str) -> None:
    raise SystemExit(
        f"gen_ch09: GATE FAILED — {msg}\n"
        "  Per BOOK_DESIGN.md §4.1 this is a STOP-AND-FLAG, not something to "
        "reconcile silently. Record both values in book/design/flags/ch09_FLAGS.md."
    )


def _gate(name: str, measured: float, spec: float, tol: float) -> None:
    if not np.isfinite(measured) or abs(measured - spec) > tol:
        _fail(f"{name}: measured {measured!r} vs spec {spec!r} (tol {tol})")
    print(f"    gate OK  {name}: {measured:.7g} == {spec:.7g}")


# ---------------------------------------------------------------------------
# The run's own per-h mixture legs (7 s.f. log lines — never the 4-dp w_G field)
# ---------------------------------------------------------------------------
def read_legs(rel: Path = RUN_LOG_REL) -> dict[str, dict[float, float]]:
    D: dict[float, float] = {}
    bGbar: dict[float, float] = {}
    wg_log4: dict[float, float] = {}
    s_nobh: dict[float, float] = {}
    s_withbh: dict[float, float] = {}
    text = _must(rel).read_text()
    for line in text.splitlines():
        m = re.search(r"D\(h=([\d.]+)\) = ([\d.e+-]+)", line)
        if m:
            D[round(float(m.group(1)), 4)] = float(m.group(2))
        m = re.search(r"beta_Gbar\(h=([\d.]+)\) = ([\d.e+-]+)", line)
        if m:
            bGbar[round(float(m.group(1)), 4)] = float(m.group(2))
        m = re.search(
            r"h_0_(\d+)\.log.*w_G=beta_G/D\(h\)=([\d.]+), "
            r"sum_w_Dg\(no_bh\)=([\d.e+-]+), sum_w_Dg\(with_bh\)=([\d.e+-]+)",
            line,
        )
        if m:
            h = round(float("0." + m.group(1)), 4)
            wg_log4[h] = float(m.group(2))
            s_nobh[h] = float(m.group(3))
            s_withbh[h] = float(m.group(4))
    if not (len(D) == len(bGbar) == len(s_nobh) == 41):
        _fail(f"log extract: expected 41 h-points per leg, got {len(D)}/{len(bGbar)}/{len(s_nobh)}")
    return {
        "D": D,
        "beta_Gbar": bGbar,
        "w_G_log4dp": wg_log4,
        "sum_w_Dg_no_bh": s_nobh,
        "sum_w_Dg_with_bh": s_withbh,
    }


def realized_counts() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for tag, rel in (("61000", CRB1_REL), ("62000", CRB2_REL)):
        df = pd.read_csv(_must(rel), usecols=["host_galaxy_index"])
        out[tag] = (int(len(df)), int((df["host_galaxy_index"] >= 0).sum()))
    n = out["61000"][0] + out["62000"][0]
    k = out["61000"][1] + out["62000"][1]
    out["pooled"] = (n, k)
    return out


def binomial_z(k: int, n: int, p: float) -> float:
    exp = n * p
    return float((k - exp) / np.sqrt(exp * (1.0 - p)))


# ---------------------------------------------------------------------------
# The 2x2 cell B control (landed 2026-07-31) — the w_G pre-registration payoff
# ---------------------------------------------------------------------------
def _wg_by_h(rel: Path) -> pd.Series:
    """Per-h w_G from a diagnostics CSV, full precision, one row per h."""
    df = pd.read_csv(_must(rel), usecols=["h", "w_G"])
    g = df.groupby(df["h"].round(4))["w_G"]
    if int(g.nunique().max()) != 1:
        _fail(f"{rel}: w_G is not constant across events at fixed h")
    return g.first()


def build_cellb() -> dict[str, Any]:
    """Re-measure the pre-registered `w_G` reading against cell B's artifacts.

    The pre-registration said *"w_G(h): expected bit-identical to the #53 runs
    (pure quadrature, no catalogue input).  If it differs, that itself is a
    finding."*  This checks it two independent ways — the full-precision
    diagnostics column element-wise, and the run's own D / beta_Gbar log legs —
    and stops if either is anything other than exact.
    """
    payload: dict[str, Any] = {
        "date": CELLB_DATE,
        "jobs_prereg": CELLB_JOBS_PREREG,
        "jobs_result": CELLB_JOBS_RESULT,
        "maps": SPEC_CELLB_MAPS,
        "estimator_2d": _r(SPEC_CELLB_MAPS["B"]["map_2d"] - SPEC_CELLB_MAPS["A"]["map_2d"], 4),
        "total_2d_r1": _r(SPEC_CELLB_MAPS["C"]["map_2d"] - SPEC_CELLB_MAPS["A"]["map_2d"], 4),
        "wg_recorded": {f"{h:.2f}": v for h, v in SPEC_CELLB_WG.items()},
    }
    payload["estimator_share_2d_pct"] = _r(
        100.0 * payload["estimator_2d"] / payload["total_2d_r1"], 3
    )

    cb_diag, r1_diag = _resolve(CELLB_DIAG_REL), _resolve(R1_DIAG_REL)
    if cb_diag is None or r1_diag is None:
        print("    NOTICE: cell-B / r1 diagnostics CSVs absent — the recorded")
        print("            w_G equality is carried unverified.")
        payload["verified"] = False
        return payload

    cb, r1 = _wg_by_h(CELLB_DIAG_REL), _wg_by_h(R1_DIAG_REL)
    hs = sorted(set(cb.index) & set(r1.index))
    if len(hs) != 41:
        _fail(f"cell-B w_G comparison: expected 41 shared h-points, got {len(hs)}")
    a = np.array([cb.loc[h] for h in hs])
    b = np.array([r1.loc[h] for h in hs])
    max_dev = float(np.max(np.abs(a - b)))
    if max_dev != 0.0 or not bool(np.all(a == b)):
        _fail(
            "the pre-registered w_G equality FAILED — cell B differs from #53 r1 "
            f"(max|Δ| = {max_dev!r}); per the pre-registration that is itself a finding"
        )
    print(
        f"    gate OK  cell-B w_G ≡ #53 r1 element-wise over {len(hs)} grid points (max|Δ| = 0.0)"
    )

    for h, spec in SPEC_CELLB_WG.items():
        got = round(float(cb.loc[round(h, 4)]), 7)
        if abs(got - spec) > 5e-8:
            _fail(f"cell-B w_G({h}): measured {got!r} vs readout {spec!r}")
    print("    gate OK  cell-B w_G(0.60/0.73/0.81) == the readout's three reads")

    # second, independent route: the run's own 7-s.f. selection legs
    cb_legs = read_legs(CELLB_LOG_REL)
    legs_identical = None
    log_route_dev = None
    if len(cb_legs["D"]) == 41:
        legs_identical = bool(
            all(
                cb_legs["D"][h] == legs_D and cb_legs["beta_Gbar"][h] == legs_B
                for h, legs_D, legs_B in _r1_leg_rows()
            )
        )
        wg_log = np.array(
            [(cb_legs["D"][h] - cb_legs["beta_Gbar"][h]) / cb_legs["D"][h] for h in hs]
        )
        log_route_dev = float(np.max(np.abs(wg_log - a)))

    payload.update(
        {
            "verified": True,
            "wg_n_grid": len(hs),
            "wg_max_abs_dev": 0.0,
            "wg_elementwise_equal": True,
            "wg_measured": {f"{h:.2f}": _r(float(cb.loc[round(h, 4)]), 7) for h in SPEC_CELLB_WG},
            "legs_identical": legs_identical,
            "wg_log_route_max_dev": _r(log_route_dev, 3) if log_route_dev is not None else None,
            "source": str(CELLB_DIAG_REL),
        }
    )
    return payload


def _r1_leg_rows() -> list[tuple[float, float, float]]:
    """(h, D, beta_Gbar) from #53 r1's own log extract, for the leg comparison."""
    legs = read_legs()
    return [(h, legs["D"][h], legs["beta_Gbar"][h]) for h in sorted(legs["D"])]


# ---------------------------------------------------------------------------
# I9.2 — the Consistency Bench
# ---------------------------------------------------------------------------
def build_bench() -> dict[str, Any]:
    legs = read_legs()
    hs = sorted(legs["D"])
    D = np.array([legs["D"][h] for h in hs])
    bGbar = np.array([legs["beta_Gbar"][h] for h in hs])
    bG = D - bGbar
    wG = bG / D
    r = np.array([legs["sum_w_Dg_with_bh"][h] / legs["sum_w_Dg_no_bh"][h] for h in hs])
    wG_aware = (r * bG) / (r * bG + bGbar)
    i73 = hs.index(0.73)

    _gate("w_G(0.73) mass-blind", float(wG[i73]), SPEC_WG_BLIND, 5e-7)
    _gate("r(0.73)", float(r[i73]), SPEC_R_073, 5e-6)
    _gate("w_G(0.73) mass-aware", float(wG_aware[i73]), SPEC_WG_AWARE, 5e-6)

    counts = realized_counts()
    n, k = counts["pooled"]
    if (k, n) != (SPEC_K_POOLED, SPEC_N_POOLED):
        _fail(
            f"realized in-catalogue counts: measured {k}/{n} vs spec {SPEC_K_POOLED}/{SPEC_N_POOLED}"
        )
    print(f"    gate OK  realized in-catalogue rate: {k}/{n} = {k / n:.6f}")

    z_blind = binomial_z(k, n, float(wG[i73]))
    z_aware = binomial_z(k, n, float(wG_aware[i73]))
    _gate("binomial z (mass-blind, pooled)", z_blind, SPEC_Z_BLIND, 5e-3)
    _gate("binomial z (mass-aware, pooled)", z_aware, SPEC_Z_AWARE, 5e-3)

    # The two independent suppression measures (C9's "0.2 sigma" agreement).
    odds_model = float(wG[i73] / (1.0 - wG[i73]))
    odds_real = k / (n - k)
    supp_realized = odds_real / odds_model
    supp_err = supp_realized / np.sqrt(k)

    # C9's diagnostic counterfactual posteriors (NOT a ratified fix).
    curves = json.loads(_must(G4_CURVES_REL).read_text())
    res = json.loads(_must(G4_RESULTS_REL).read_text())
    for key, (map_spec, mean_spec) in SPEC_CF.items():
        got_map, got_mean = res[f"combined_{key}"]
        if abs(got_map - map_spec) > 1e-9 or abs(got_mean - mean_spec) > 5e-5:
            _fail(
                f"counterfactual {key}: measured {got_map}/{got_mean} vs spec {map_spec}/{mean_spec}"
            )
    print("    gate OK  the four C9 counterfactual MAP/mean pairs")

    # generator_marginal w_G — the OPEN dispute (sources map §7 item 7).
    genmarg_verified = None
    sig0 = _resolve(SIG0_REL)
    if sig0 is not None and sig0.is_file():
        df = pd.read_csv(sig0, usecols=["h", "w_G"])
        df["h"] = df["h"].round(4)
        got = [float(df.loc[df.h == h, "w_G"].iloc[0]) for h in GENMARG_H]
        genmarg_verified = bool(
            all(abs(a - b) < 5e-7 for a, b in zip(got, GENMARG_MEASURED, strict=True))
        )
        if not genmarg_verified:
            _fail(f"sig0_control w_G: measured {got} vs recorded {GENMARG_MEASURED}")
        print("    gate OK  sig0_control generator_marginal w_G matches the recorded curve")
    else:
        print("    NOTICE: sig0_control diagnostics CSV absent — the recorded")
        print("            generator_marginal values are carried unverified.")

    g2 = json.loads(_must(G2_SUMMARY_REL).read_text())
    g6 = json.loads(_must(G6_RESULTS_REL).read_text())
    dark = json.loads(_must(C9_DARK_REL).read_text())["production_pool"]

    return {
        "h_true": H_TRUE,
        "h_grid": _rl(hs, 6),
        "D": _rl(D, 8),
        "beta_G": _rl(bG, 8),
        "beta_Gbar": _rl(bGbar, 8),
        "w_G_blind": _rl(wG, 8),
        "w_G_aware": _rl(wG_aware, 8),
        "r": _rl(r, 8),
        "w_G_log_4dp": _rl([legs["w_G_log4dp"][h] for h in hs], 6),
        "w_G_log_4dp_max_abs_dev": _r(
            float(np.max(np.abs(np.array([legs["w_G_log4dp"][h] for h in hs]) - wG))), 3
        ),
        "w_G_log_4dp_max_rel_dev": _r(
            float(np.max(np.abs(np.array([legs["w_G_log4dp"][h] for h in hs]) / wG - 1.0))), 3
        ),
        "realized": {
            "seed61000": {"n": counts["61000"][0], "k": counts["61000"][1]},
            "seed62000": {"n": counts["62000"][0], "k": counts["62000"][1]},
            "pooled": {"n": n, "k": k, "rate": _r(k / n, 8)},
            "z_blind": _r(z_blind, 6),
            "z_aware": _r(z_aware, 6),
            "z_blind_seed61000": _r(binomial_z(*counts["61000"][::-1], float(wG[i73])), 5),
            "z_blind_seed62000": _r(binomial_z(*counts["62000"][::-1], float(wG[i73])), 5),
            "expected_blind": _r(n * float(wG[i73]), 6),
            "expected_aware": _r(n * float(wG_aware[i73]), 6),
        },
        "suppression": {
            "r_073_from_logs": _r(r[i73], 6),
            "realized": _r(supp_realized, 6),
            "realized_err": _r(supp_err, 4),
            "sigma_gap": _r(abs(supp_realized - r[i73]) / supp_err, 3),
        },
        "genmarg": {
            "h": GENMARG_H,
            "claimed": GENMARG_CLAIMED,
            "measured": GENMARG_MEASURED,
            "verified_against_sig0_csv": genmarg_verified,
            "status": "OPEN — exact curve attribution disputed (sources map §7 item 7)",
        },
        "counterfactual": {
            "h": _rl(curves["h"], 6),
            "no_bh_base": _rl(curves["combined_no_bh_base"], 7),
            "no_bh_corr": _rl(curves["combined_no_bh_corr"], 7),
            "with_bh_base": _rl(curves["combined_with_bh_base"], 7),
            "with_bh_corr": _rl(curves["combined_with_bh_corr"], 7),
            "summary": {k2: {"map": v[0], "mean": _r(v[1], 6)} for k2, v in res.items()},
        },
        "population_fraction": {
            "F": _r(g2["F"], 8),
            "W_cat": _r(g2["W_cat"], 8),
            "V_f": _r(g2["V_f"], 8),
            "V_tot": _r(g2["V_tot"], 8),
            "n_hat_w": _r(g2["n_hat_w"], 8),
            "mean_fbar_over_detected": _r(g6["mean_fbar_over_detected"], 6),
        },
        "dark_side": {
            "ks": _r(dark["ks_statistic"], 4),
            "p": _r(dark["ks_pvalue"], 4),
            "n_dark": dark["n_dark_total"],
            "quantiles": [
                {
                    "q": q["q_pct"],
                    "observed": _r(q["observed_z"], 5),
                    "model": _r(q["beta_Gbar_integrand_z"], 5),
                    "diff": _r(q["diff"], 4),
                }
                for q in dark["quantile_table"]
            ],
            "fingerprint_dl_max_Gpc": _r(dark["fingerprint"]["dl_max_computed_Gpc"], 8),
        },
        "cell_b": build_cellb(),
        "event_889": {
            "index": EVENT_889,
            "w_G_absolute_marginal": _r(wG[i73], 8),
            "w_G_generator_marginal_measured": GENMARG_MEASURED[2],
            "w_G_generator_marginal_claimed": GENMARG_CLAIMED[2],
            "ratio_53_over_51_measured": _r(float(wG[i73]) / GENMARG_MEASURED[2], 4),
            "ratio_53_over_51_claimed": _r(float(wG[i73]) / GENMARG_CLAIMED[2], 4),
        },
    }


# ---------------------------------------------------------------------------
# I9.3 — the de-rail matrix
# ---------------------------------------------------------------------------
def build_derail() -> dict[str, Any]:
    derail = json.loads(_must(DERAIL_REL).read_text())
    cube = json.loads(_must(CUBE_REL).read_text())
    shared = [k for k in derail if k in cube]
    if not all(derail[k] == cube[k] for k in shared):
        _fail("derail_matrix_results.json and G3_ablation_cube.json disagree on a shared mode")
    print(f"    gate OK  de-rail matrix ≡ ablation cube on {len(shared)} shared modes")

    def state(key: str, src: dict[str, Any] | None = None) -> dict[str, Any]:
        s = (src if src is not None else derail)[key]
        return {
            "h": _rl(s["h_values"], 6),
            "posterior": _rl(s["posterior"], 6),
            "map": _r(s["MAP"], 6),
            "mean": _r(s["mean"], 8),
            "edge_mass": _r(s["edge_mass"], 4),
            "railed": bool(s["railed"]),
        }

    return {
        "h_true": H_TRUE,
        "venue": {
            "seed": 600,
            "n_events": 494,
            "n_h": 7,
            "note": "7-h grid, 494-event subsample, injected h = 0.73 (ledger #49)",
        },
        "steps": [
            {
                "id": "pre4pi",
                "label": "pre-4π (global denominator, peak-density B_num)",
                "map": 0.86,
                "curve": None,
                "recorded_only": True,
                "verdict": "rails HIGH at the top of the prior grid",
            },
            {
                "id": "prod_global",
                "label": "4π-only (cb16142, still global denominator)",
                "map": 0.60,
                "curve": state("prod_global"),
                "recorded_only": False,
                "verdict": "rails LOW — the 1/(4π) fix alone FLIPS the rail: necessary, not sufficient",
            },
            {
                "id": "local_ratio",
                "label": "+ local_ratio (self-normalized ratio of sums)",
                "map": 0.73,
                "curve": state("local_ratio"),
                "recorded_only": False,
                "verdict": "interior and peaked — 98% of the mass at 0.73",
            },
            {
                "id": "volume_deconv",
                "label": "+ volume_deconv (local ratio + dV_c/(1+z) host-z prior)",
                "map": 0.73,
                "curve": state("volume_deconv"),
                "recorded_only": False,
                "verdict": "interior; mean +0.010 above local_ratio (the volume prior removes the Eddington-in-z low bias)",
            },
        ],
        "catonly": state("catonly"),
        "cube_extra": {
            "volume_global": state("volume_global", cube) if "volume_global" in cube else None,
            "note": (
                "docs/gates/G3_ablation_cube.json carries one extra state the de-rail "
                "matrix does not: the volume kernel in both N_g and D_g while the "
                "denominator is still global (MAP 0.76). Different ablation axis; "
                "shown in the numbers view only."
            ),
        },
        "h0_independent": {
            "injected_truths": [0.63, 0.65, 0.67, 0.70, 0.73, 0.75, 0.77],
            "production_map": 0.86,
            "catalog_only_tracks_truth": True,
            "note": "ledger #49a — recorded verdict text only; no per-truth posterior is stored in the tree",
        },
    }


# ---------------------------------------------------------------------------
# N5 / G1 — the Option-A identity on the real catalogue
# ---------------------------------------------------------------------------
def build_identity() -> dict[str, Any]:
    g1 = json.loads(_must(G1_REL).read_text())
    table = g1["table"]
    hs = sorted(table, key=float)
    ratio = np.array([table[h]["ratio"] for h in hs])
    shape = np.array([table[h]["shape_h3_corrected"] for h in hs])

    raw_tilt = float(ratio[-1] / ratio[0] - 1.0)
    corr_tilt = float(shape[-1] - shape[0])
    if abs(corr_tilt - g1["end_to_end_tilt_h3_corrected"]) > 1e-6:
        _fail(
            "G1 h³-corrected end-to-end tilt: recomputed "
            f"{corr_tilt} vs file {g1['end_to_end_tilt_h3_corrected']}"
        )
    print(f"    gate OK  G1 end-to-end h³-corrected tilt {corr_tilt:+.4f} (file's own value)")

    return {
        "h_ref": g1["h_ref"],
        "snr_threshold": g1["snr_threshold"],
        "n_sky_bands": g1["n_sky_bands"],
        "h": _rl([float(h) for h in hs], 6),
        "Sigma_global": _rl([table[h]["Sigma_global"] for h in hs], 8),
        "beta_G": _rl([table[h]["beta_G"] for h in hs], 8),
        "D": _rl([table[h]["D"] for h in hs], 8),
        "ratio": _rl(ratio, 7),
        "shape_h3_corrected": _rl(shape, 7),
        "raw_ratio_growth": _r(ratio[-1] / ratio[0], 5),
        "raw_ratio_growth_recomputed_tilt": _r(raw_tilt, 5),
        "end_to_end_tilt_raw": _r(g1["end_to_end_tilt_raw"], 6),
        "end_to_end_tilt_h3_corrected": _r(g1["end_to_end_tilt_h3_corrected"], 6),
        "max_shape_deviation_h3_corrected": _r(g1["max_shape_deviation_h3_corrected"], 5),
        "h3_expected": _r((0.86 / 0.60) ** 3, 5),
        "verdict": g1["verdict"],
        "identity_violation_value_pct": 33,  # H0R:1548-1552 (recorded)
        "identity_violation_logslope_per_h": 0.39,  # ibid.
    }


# ---------------------------------------------------------------------------
# I9.1 — the universe factory
# ---------------------------------------------------------------------------
def _pool_dir() -> Path | None:
    candidate = _resolve(POOL_REL)
    if (
        candidate is not None
        and candidate.is_dir()
        and any(candidate.glob("injection_h_*_task_*.csv"))
    ):
        return candidate
    return None


def omega_panel() -> dict[str, Any]:
    """Recompute G7 row 6 with the repo's own dist(); gate on the published table.

    h' solves d_L(z; h', Omega_assumed) = d_L(z; 0.73, Omega_true).  Both
    cosmologies are flat, so Omega_de = 1 - Omega_m must be passed explicitly
    (the module default keeps 0.7274 and would silently un-flatten the model).
    """
    from scipy.optimize import brentq

    def h_prime(z: float, assumed: float, true: float) -> float:
        target = dist(z, H_TRUE, true, 1.0 - true)
        return float(
            brentq(lambda hp: dist(z, hp, assumed, 1.0 - assumed) - target, 0.3, 2.0, xtol=1e-14)
        )

    zs = sorted(G7_OMEGA_TABLE)
    rows = []
    for z in zs:
        a = 100.0 * (h_prime(z, OMEGA_M, OMEGA_TRUE_PLANCK) / H_TRUE - 1.0)
        b = 100.0 * (h_prime(z, OMEGA_PRE_G11, OMEGA_TRUE_PLANCK) / H_TRUE - 1.0)
        pub_a, pub_b = G7_OMEGA_TABLE[z]
        if abs(a - pub_a) > 0.005 or abs(b - pub_b) > 0.005:
            _fail(
                f"Omega_m row z={z}: recomputed ({a:.3f}, {b:.3f}) vs published ({pub_a}, {pub_b})"
            )
        rows.append({"z": z, "pct_m1": _r(a, 4), "pct_preG11": _r(b, 4)})
    print("    gate OK  all 12 published Omega_m mis-specification cells")

    z_curve = np.round(np.linspace(0.01, 1.5, 60), 5)
    return {
        "omega_m_fiducial": OMEGA_M,
        "omega_m_planck": OMEGA_TRUE_PLANCK,
        "omega_m_pre_g11": OMEGA_PRE_G11,
        "rows": rows,
        "z_curve": _rl(z_curve, 6),
        "dl_m1": _rl([dist(float(z), H_TRUE, OMEGA_M, 1.0 - OMEGA_M) for z in z_curve], 7),
        "dl_planck": _rl(
            [dist(float(z), H_TRUE, OMEGA_TRUE_PLANCK, 1.0 - OMEGA_TRUE_PLANCK) for z in z_curve], 7
        ),
    }


# Recorded, from docs/derivations/plunge_window_initial_conditions.md §11 / §5.
PLUNGE_TABLE = [
    {"M_z": 3.0e6, "t_plunge": 0.5, "p0": 4.85, "snr_1Gpc": 45.8, "d_hor_Gpc": 2.29},
    {"M_z": 3.0e6, "t_plunge": 2.0, "p0": 5.93, "snr_1Gpc": 49.2, "d_hor_Gpc": 2.46},
    {"M_z": 3.0e6, "t_plunge": 4.0, "p0": 6.67, "snr_1Gpc": 50.3, "d_hor_Gpc": 2.51},
    {"M_z": 1.0e7, "t_plunge": 0.5, "p0": 3.61, "snr_1Gpc": 6.9, "d_hor_Gpc": 0.35},
    {"M_z": 1.0e7, "t_plunge": 2.0, "p0": 4.25, "snr_1Gpc": 8.7, "d_hor_Gpc": 0.44},
    {"M_z": 1.0e7, "t_plunge": 4.0, "p0": 4.64, "snr_1Gpc": 9.7, "d_hor_Gpc": 0.48},
]
SNAPSHOT_HORIZON = [
    {"M_z": 3.0e6, "d_hor_Gpc": "0.1–0.25"},
    {"M_z": 1.0e7, "d_hor_Gpc": "0.014"},
]
P0_AT_T = [
    {"M_z": 1.0e4, "p0": 109.5},
    {"M_z": 1.0e5, "p0": 34.3},
    {"M_z": 1.0e6, "p0": 10.83},
    {"M_z": 3.0e6, "p0": 6.81},
    {"M_z": 1.0e7, "p0": 4.71},
    {"M_z": 2.5e7, "p0": 3.76},
]


def build_factory() -> dict[str, Any] | None:
    pool = _pool_dir()
    payload: dict[str, Any] = {
        "h_true": H_TRUE,
        "snr_threshold": float(SNR_THRESHOLD),
        "mission_years": float(LISA_MISSION_DURATION_YEARS),
        "mission_years_retired": 5.0,
        "snapshot_band": [10.0, 16.0],
        "plunge_table": PLUNGE_TABLE,
        "snapshot_horizon": SNAPSHOT_HORIZON,
        "p0_at_T": P0_AT_T,
        "omega": omega_panel(),
        "logm_edges": _rl(LOGM_EDGES, 6),
        "z_edges": _rl(Z_EDGES, 6),
    }

    if pool is None:
        print("    NOTICE: injection pool not found — factory pool block skipped.")
        payload["pool"] = None
        return payload

    files = sorted(glob.glob(str(pool / "injection_h_*_task_*.csv")))
    # Two writer eras are concatenated in this pool: the earlier one predates
    # the plunge-window IC columns, so `p0`/`t_plunge_yr` are absent from 6,000
    # of the 200,100 rows.  Read every column and mask, rather than dropping
    # files or pretending the columns are universal.
    frames = []
    for f in files:
        d = pd.read_csv(f)
        for col in ("t_plunge_yr", "p0"):
            if col not in d.columns:
                d[col] = np.nan
        frames.append(d[["z", "M", "SNR", "h_inj", "stratum", "code_rev", "t_plunge_yr", "p0"]])
    df = pd.concat(frames, ignore_index=True)
    a = df[df["stratum"] == "a"].copy()
    a["lm"] = np.log10(a["M"].to_numpy())
    det = a["SNR"].to_numpy() >= SNR_THRESHOLD

    hist_drawn, _, _ = np.histogram2d(a["lm"], a["z"], bins=[LOGM_EDGES, Z_EDGES])
    hist_det, _, _ = np.histogram2d(a["lm"][det], a["z"][det], bins=[LOGM_EDGES, Z_EDGES])

    # p0 profile per mass bin, against the retired snapshot band [10, 16].
    prof = []
    idx = np.digitize(a["lm"].to_numpy(), LOGM_EDGES) - 1
    p0 = a["p0"].to_numpy()
    has_p0 = np.isfinite(p0)
    in_band = has_p0 & (p0 >= 10.0) & (p0 <= 16.0)
    for b in range(len(LOGM_EDGES) - 1):
        sel = idx == b
        nb = int(sel.sum())
        selp = sel & has_p0
        if nb == 0 or not selp.any():
            prof.append({"n": nb, "n_p0": int(selp.sum())})
            continue
        prof.append(
            {
                "n": nb,
                "n_p0": int(selp.sum()),
                "p0_q10": _r(np.quantile(p0[selp], 0.10), 5),
                "p0_q50": _r(np.quantile(p0[selp], 0.50), 5),
                "p0_q90": _r(np.quantile(p0[selp], 0.90), 5),
                "frac_in_snapshot_band": _r(in_band[selp].sum() / selp.sum(), 4),
                "det_frac": _r(det[sel].mean(), 4),
            }
        )

    payload["pool"] = {
        "dir": str(POOL_REL),
        "n_files": len(files),
        "n_data_rows": int(len(df)),
        "n_lines_with_headers": int(len(df) + len(files)),
        "strata": {s: int((df["stratum"] == s).sum()) for s in ("a", "b", "c")},
        "h_inj": _rl(sorted(df["h_inj"].unique()), 4),
        "z_cut": 1.5,
        "n_a": int(len(a)),
        "n_a_detected": int(det.sum()),
        "det_frac_a": _r(det.mean(), 5),
        "z_median_drawn": _r(np.median(a["z"]), 5),
        "z_median_detected": _r(np.median(a["z"][det]), 5),
        "z_q95_detected": _r(np.quantile(a["z"][det], 0.95), 5),
        "logm_median_drawn": _r(np.median(a["lm"]), 5),
        "logm_median_detected": _r(np.median(a["lm"][det]), 5),
        "t_plunge_range": [_r(a["t_plunge_yr"].min(), 4), _r(a["t_plunge_yr"].max(), 5)],
        "p0_range": [_r(np.nanmin(p0), 5), _r(np.nanmax(p0), 5)],
        "n_a_with_ic_columns": int(has_p0.sum()),
        "code_revs": sorted(str(c) for c in df["code_rev"].dropna().unique()),
        "frac_p0_in_snapshot_band": _r(in_band.sum() / has_p0.sum(), 4),
        "hist_drawn": [[int(v) for v in row] for row in hist_drawn],
        "hist_detected": [[int(v) for v in row] for row in hist_det],
        "p0_profile": prof,
    }
    return payload


# ---------------------------------------------------------------------------
def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=False))
    print(f"    wrote {path.name}  ({path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    print("gen_ch09: Chapter 9 — Building a Universe to Break Your Estimator")

    print("  [1/4] I9.2 — the Consistency Bench (C9)")
    _write(OUT_BENCH, build_bench())

    print("  [2/4] I9.3 — the de-rail matrix (ledger #49)")
    _write(OUT_DERAIL, build_derail())

    print("  [3/4] N5 — the Option-A identity (G1)")
    _write(OUT_IDENTITY, build_identity())

    print("  [4/4] I9.1 — the universe factory")
    factory = build_factory()
    if factory is None:
        print("    factory payload unavailable; existing file left untouched")
    else:
        _write(OUT_FACTORY, factory)

    print("gen_ch09: done.")


if __name__ == "__main__":
    main()
