"""r-cone-loss — the bias-floor contribution of cone-loss events (build node b-cone-scorer).

Registration: ``REGISTRATION_DRAFT.md`` (this directory), Research Graph 1, Branch H,
wave 3. Everything here is **DRAFT** per that file's header (nothing frozen); this
script implements the draft's §2 build node / §7 launch block / §2 statistic verbatim.

VERIFIER-INDEPENDENCE CONTRACT (standing rule 2, matching the ``cmem_a1.py`` /
``cmem_reads.py`` precedent in this tree): this file is authored by the BUILDER agent
(``b-cone-scorer``), who runs it ONLY with ``--dry-run`` — gates G-1..G-4 plus the
production/anchor/harness census, no per-event scores, no Δh_cone/φ_cone/SE/Z (draft
§7: "builder runs ONLY --dry-run (G-1…G-4 + census, no scores); a DIFFERENT agent runs
the statistic"). Real mode (``--dry-run`` omitted) computes the registered statistic
(draft §2) and must be run by a different agent.

FIX ROUND 2 (task instruction, this revision): the builder now IMPLEMENTS the real-mode
statistic (below) so a runner agent can execute it — the verifier-independence contract
above is about WHO EXECUTES real mode on production data, not who writes the code; the
builder still never runs real mode on production/anchor/harness data itself (task
instruction: synthetic 10-row table only, recorded in BUILD_RECORD.md "FIX 2"). All
gate code (G-1..G-4, ``run_gates``, thresholds) is byte-identical to the rev.1 file —
only the post-gate real-mode branch of ``main()`` and new statistic functions below it
are new.

Cone geometry (chord / radius) is the handler's own construction, replicated
line-for-line on the CRB columns per the ``cmem_a1.py``/``cmem_reads.py`` precedent —
NOT re-derived: chord = great-circle chord between the event's drawn sky position
(``qS``, ``phiS``) and the assigned host's catalogue sky position
(``THETA_S``, ``PHI_S``), both embedded via
``darksiren_emri.galaxy_catalogue.handler._polar_to_cartesian``; radius =
``sky_cone_k * sqrt(lambda_max(J Sigma' J^T))`` where ``Sigma' = [[phi_var, cov],
[cov, theta_var]]`` from the CRB's sky Fisher sub-block and ``J = diag([sin(theta),
1])`` (``get_possible_hosts_from_ball_tree``, ``handler.py``, ``_sky_cone_k``,
``bayesian_statistics.py:3659,5751``).

Sky Mahalanobis^2 (gate G-4) uses the SAME Sigma' (not the J-scaled version): for a
correctly jac-scaled offset v_scaled = J @ v, ``v_scaled^T (J Sigma' J^T)^-1
v_scaled == v^T Sigma'^-1 v`` (J is diagonal and invertible, so it cancels
algebraically) — i.e. "the sky offsets' Mahalanobis^2 under the row's own J Sigma'
J^T" (draft G-4) is exactly the raw angular-offset Mahalanobis^2 under Sigma' alone.
Implemented in the closed-form 2x2-inverse style of
``results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py:_mahalanobis_check``.

Usage:
    uv run python cone_loss_reads.py --dry-run [other flags]   # builder smoke-test
    uv run python cone_loss_reads.py [other flags]              # runner only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

import darksiren_emri
from darksiren_emri.galaxy_catalogue.handler import GalaxyCatalogueHandler, _polar_to_cartesian

WORK_DIR = Path(__file__).resolve().parent / "cone_loss_work"

# CRB columns needed to build the sky-cone census (production pool + both anchor
# fleets share this schema).
CRB_COLS = [
    "qS",
    "phiS",
    "delta_qS_delta_qS",
    "delta_phiS_delta_phiS",
    "delta_phiS_delta_qS",
    "host_galaxy_index",
    "in_catalog",
]

# G-2 double anchor (draft §5): full-float chord/radius, both fleets on disk.
ANCHOR_MKER = {
    "fleet_arm_seed": "bc_900121_work",
    "seed": 900121,
    "event_idx": 20,
    "chord": 1.674660e-03,
    "chord_tol": 5e-10,
    "radius": 1.4956979545757095e-03,
    "radius_tol": 1e-15,
}
ANCHOR_CMEM = {
    "fleet_arm_seed": "bc_900101_work",
    "seed": 900101,
    "event_idx": 0,
    "chord": 0.0116656941007181,
    "chord_tol": 5e-10,
    "radius": 0.0359121946154451,
    "radius_tol": 1e-15,
}

# G-4 sky-scatter envelope (draft §1/§3): closed-form 1.5*sqrt(lambda_max) circle,
# 13.4% (1-D limit) to 32.5% (isotropic Rayleigh tail). Rev. 1 item 7: the envelope
# clause is an exact two-sided binomial test of n_out against the NEAREST edge, not
# an asymptotic f_out-in-band comparison (see g4_scatter_law below).
SCATTER_ENVELOPE = (0.134, 0.325)

# G-3 join: production CRB row-index gaps (event_idx not scored in the diagnostics).
SCORED_SET_GAPS = (1203, 1356)

P6_LINE_RE = re.compile(r"P6 host-recovery \(h=([0-9.]+)\): 1D (\d+)/(\d+) hosts recovered")


def md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex MD5 digest of a file, streamed in ``chunk``-byte blocks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cone_radius(theta: float, phi_var: float, theta_var: float, cov: float, k: float) -> float:
    """Sky-cone radius = ``k * sqrt(lambda_max(J Sigma' J^T))``.

    Replicated line-for-line from ``cmem_a1.py:cone_radius`` /
    ``get_possible_hosts_from_ball_tree`` (``handler.py``), generalized to a caller-
    supplied ``k`` (draft's ``--sky-cone-k``, registered value 1.5).

    Args:
        theta: Event polar sky angle (``qS``), radians.
        phi_var: Fisher variance ``delta_phiS_delta_phiS``.
        theta_var: Fisher variance ``delta_qS_delta_qS``.
        cov: Fisher covariance ``delta_phiS_delta_qS``.
        k: Sky-cone multiplier (registered ``_sky_cone_k``).

    Returns:
        The cone radius in chord units.
    """
    sigma = np.array([[phi_var, cov], [cov, theta_var]])
    jac = np.diag([abs(np.sin(theta)), 1.0])
    lam = float(np.linalg.eigvalsh(jac @ sigma @ jac.T).max())
    return float(k * np.sqrt(max(lam, 0.0)))


def sky_mahalanobis2(
    delta_phi: npt.NDArray[np.float64],
    delta_theta: npt.NDArray[np.float64],
    phi_var: npt.NDArray[np.float64],
    theta_var: npt.NDArray[np.float64],
    cov: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Closed-form Mahalanobis^2 of the raw (phi, theta) sky offset under Sigma'.

    Equal to the J-scaled offset's Mahalanobis^2 under ``J Sigma' J^T`` (see module
    docstring) — the quantity draft gate G-4 requires to be chi-squared(2)-distributed
    under the forward model's designed Gaussian sky scatter. 2x2 closed-form inverse,
    in the style of
    ``results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py:_mahalanobis_check``.

    Returns:
        Per-row Mahalanobis^2; ``NaN`` where the 2x2 covariance is singular.
    """
    det = phi_var * theta_var - cov * cov
    ok = det > 0.0
    m2 = np.full(delta_phi.shape, np.nan, dtype=np.float64)
    inv00 = theta_var[ok] / det[ok]
    inv11 = phi_var[ok] / det[ok]
    inv01 = -cov[ok] / det[ok]
    m2[ok] = (
        delta_phi[ok] ** 2 * inv00
        + 2.0 * delta_phi[ok] * delta_theta[ok] * inv01
        + delta_theta[ok] ** 2 * inv11
    )
    return m2


def wrap_angle(delta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Wrap an azimuthal-angle difference to the shortest signed arc in ``(-pi, pi]``."""
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def load_catalogue(k: float) -> tuple[pd.DataFrame, npt.NDArray[np.float64], Path]:
    """Load the reduced GLADE+ catalogue via the handler and embed host sky positions.

    Returns:
        (catalogue frame, host xyz array (N,3), catalogue file path).
    """
    handler = GalaxyCatalogueHandler(1e4, 1e7, k)
    cat = handler.reduced_galaxy_catalog.reset_index(drop=True)
    host_xyz = _polar_to_cartesian(
        cat["THETA_S"].to_numpy(dtype=np.float64), cat["PHI_S"].to_numpy(dtype=np.float64)
    )
    catalogue_path = Path(darksiren_emri.__file__).resolve().parent / (
        "galaxy_catalogue/reduced_galaxy_catalogue.csv"
    )
    return cat, host_xyz, catalogue_path


def build_census(
    crb_path: Path, cat: pd.DataFrame, host_xyz: npt.NDArray[np.float64], k: float
) -> pd.DataFrame:
    """Build the sky-cone census (chord, radius, outside flag, Mahalanobis^2) for one CRB file.

    ``event_idx`` is the CRB's own row position (0-based) — confirmed to be the join
    key used by the diagnostics CSVs (draft G-3). Only in-catalogue rows
    (``in_catalog`` true, ``host_galaxy_index >= 0``) are census members.
    """
    crb = pd.read_csv(crb_path, usecols=CRB_COLS)
    rows = []
    for i, r in crb.iterrows():
        if not bool(r["in_catalog"]) or int(r["host_galaxy_index"]) < 0:
            continue
        hidx = int(r["host_galaxy_index"])
        theta_e, phi_e = float(r["qS"]), float(r["phiS"])
        ev_xyz = _polar_to_cartesian(np.array([theta_e]), np.array([phi_e]))[0]
        chord = float(np.linalg.norm(host_xyz[hidx] - ev_xyz))
        radius = cone_radius(
            theta_e,
            float(r["delta_phiS_delta_phiS"]),
            float(r["delta_qS_delta_qS"]),
            float(r["delta_phiS_delta_qS"]),
            k,
        )
        delta_theta = float(cat.loc[hidx, "THETA_S"]) - theta_e
        delta_phi = float(wrap_angle(np.array([float(cat.loc[hidx, "PHI_S"]) - phi_e]))[0])
        rows.append(
            {
                "event_idx": i,
                "host_galaxy_index": hidx,
                "chord": chord,
                "radius": radius,
                "outside": chord > radius,
                "delta_theta": delta_theta,
                "delta_phi": delta_phi,
                "phi_var": float(r["delta_phiS_delta_phiS"]),
                "theta_var": float(r["delta_qS_delta_qS"]),
                "cov": float(r["delta_phiS_delta_qS"]),
            }
        )
    return pd.DataFrame(rows)


def build_anchor(fleet_root: Path, anchor: dict[str, Any], k: float) -> dict[str, Any]:
    """Reproduce one G-2 anchor row's chord/radius from its fleet's CRB file."""
    crb_path = (
        fleet_root
        / str(anchor["fleet_arm_seed"])
        / f"seed{anchor['seed']}"
        / "simulations"
        / "prepared_cramer_rao_bounds.csv"
    )
    result: dict[str, Any] = {"path": str(crb_path), "expected": anchor}
    if not crb_path.exists():
        result["passed"] = False
        result["error"] = "anchor CRB file not found"
        return result
    crb = pd.read_csv(crb_path, usecols=CRB_COLS)
    idx = anchor["event_idx"]
    if idx >= len(crb):
        result["passed"] = False
        result["error"] = f"event_idx {idx} out of range (n={len(crb)})"
        return result
    r = crb.iloc[idx]
    if not bool(r["in_catalog"]) or int(r["host_galaxy_index"]) < 0:
        result["passed"] = False
        result["error"] = "anchor row is not in-catalogue"
        return result
    return result  # chord/radius filled in by the caller once the catalogue is loaded


def score_anchor(row_chord_radius: tuple[float, float], anchor: dict[str, Any]) -> dict[str, Any]:
    """Compare a computed (chord, radius) pair against a registered anchor within tolerance."""
    chord, radius = row_chord_radius
    chord_ok = abs(chord - anchor["chord"]) < anchor["chord_tol"]
    radius_ok = abs(radius - anchor["radius"]) < anchor["radius_tol"]
    return {
        "found_chord": chord,
        "found_radius": radius,
        "expected_chord": anchor["chord"],
        "expected_radius": anchor["radius"],
        "chord_ok": chord_ok,
        "radius_ok": radius_ok,
        "passed": bool(chord_ok and radius_ok),
    }


def parse_p6_line(run_dir: Path, h_true: float) -> dict[str, Any]:
    """Find and parse the P6 host-recovery log line for ``h_true`` under ``run_dir``.

    Returns the first match across every ``darksiren_emri_*.log`` file in ``run_dir``
    (there is one log per h-grid point; only the ``h_true`` run carries the census's
    own n_IN comparand).
    """
    h_tag = f"{h_true:.4f}"
    for log_path in sorted(run_dir.glob("darksiren_emri_*.log")):
        text = log_path.read_text(errors="replace")
        for line in text.splitlines():
            if "P6 host-recovery" not in line or h_tag not in line:
                continue
            m = P6_LINE_RE.search(line)
            if m:
                return {
                    "log_path": str(log_path),
                    "line": line.strip(),
                    "h": float(m.group(1)),
                    "n_recovered_1d": int(m.group(2)),
                    "n_in_catalogue": int(m.group(3)),
                    "found": True,
                }
    return {"found": False, "run_dir": str(run_dir), "h_true": h_true}


def count_harness_seeds(harness_root: Path) -> dict[str, Any]:
    """Count harness S/T seed directories under ``harness_root`` (g-population disclosure)."""
    s_dirs = sorted(p.name for p in harness_root.glob("seed*_S") if p.is_dir())
    t_dirs = sorted(p.name for p in harness_root.glob("seed*_T") if p.is_dir())
    return {"root": str(harness_root), "n_seed_S": len(s_dirs), "n_seed_T": len(t_dirs)}


def run_gates(args: argparse.Namespace) -> dict[str, Any]:
    """G-1..G-4 + census — everything the ``--dry-run`` mode computes."""
    gates: dict[str, Any] = {}

    # --- load catalogue + G-1 catalogue pin -------------------------------
    cat, host_xyz, catalogue_path = load_catalogue(args.sky_cone_k)
    if catalogue_path.exists():
        cat_md5 = md5_of_file(catalogue_path)
        gates["g1_catalogue_pin"] = {
            "path": str(catalogue_path),
            "md5": cat_md5,
            "expected": args.catalogue_md5,
            "passed": cat_md5 == args.catalogue_md5,
        }
    else:
        gates["g1_catalogue_pin"] = {"passed": False, "error": "catalogue file not found"}

    # --- G-1 production CRB pin -------------------------------------------
    production_crb_path = Path(args.production_crb)
    if production_crb_path.exists():
        crb_md5 = md5_of_file(production_crb_path)
        gates["g1_crb_pin"] = {
            "path": str(production_crb_path),
            "md5": crb_md5,
            "expected": args.crb_md5,
            "passed": crb_md5 == args.crb_md5,
        }
    else:
        gates["g1_crb_pin"] = {"passed": False, "error": "production CRB file not found"}

    # --- G-1 GIT_COMMIT_AT_RUN.txt pin (both venues) ------------------------
    commit_checks = {}
    for tag, run_dir in (("production", args.production_run), ("replicate", args.replicate_run)):
        commit_path = Path(run_dir) / "GIT_COMMIT_AT_RUN.txt"
        if commit_path.exists():
            commit = commit_path.read_text().strip()
            commit_checks[tag] = {
                "path": str(commit_path),
                "commit": commit,
                "expected_prefix": args.git_commit,
                "passed": commit.startswith(args.git_commit),
            }
        else:
            commit_checks[tag] = {"passed": False, "error": "GIT_COMMIT_AT_RUN.txt not found"}
    gates["g1_git_commit_pin"] = {
        "checks": commit_checks,
        "passed": all(c.get("passed", False) for c in commit_checks.values()),
    }

    # --- production census (also feeds G-3, G-4) ---------------------------
    census: pd.DataFrame | None = None
    if production_crb_path.exists() and catalogue_path.exists():
        census = build_census(production_crb_path, cat, host_xyz, args.sky_cone_k)

    # --- G-2 double anchor ---------------------------------------------------
    mker_gate = build_anchor(Path(args.anchor_fleet_mker), ANCHOR_MKER, args.sky_cone_k)
    if "error" not in mker_gate:
        mker_census = build_census(
            Path(args.anchor_fleet_mker)
            / str(ANCHOR_MKER["fleet_arm_seed"])
            / f"seed{ANCHOR_MKER['seed']}"
            / "simulations"
            / "prepared_cramer_rao_bounds.csv",
            cat,
            host_xyz,
            args.sky_cone_k,
        )
        row = mker_census[mker_census["event_idx"] == ANCHOR_MKER["event_idx"]]
        if len(row) == 1:
            mker_gate.update(
                score_anchor(
                    (float(row["chord"].iloc[0]), float(row["radius"].iloc[0])), ANCHOR_MKER
                )
            )
        else:
            mker_gate["passed"] = False
            mker_gate["error"] = f"anchor row not found in census ({len(row)})"

    cmem_gate = build_anchor(Path(args.anchor_fleet_cmem), ANCHOR_CMEM, args.sky_cone_k)
    if "error" not in cmem_gate:
        cmem_census = build_census(
            Path(args.anchor_fleet_cmem)
            / str(ANCHOR_CMEM["fleet_arm_seed"])
            / f"seed{ANCHOR_CMEM['seed']}"
            / "simulations"
            / "prepared_cramer_rao_bounds.csv",
            cat,
            host_xyz,
            args.sky_cone_k,
        )
        row = cmem_census[cmem_census["event_idx"] == ANCHOR_CMEM["event_idx"]]
        if len(row) == 1:
            cmem_gate.update(
                score_anchor(
                    (float(row["chord"].iloc[0]), float(row["radius"].iloc[0])), ANCHOR_CMEM
                )
            )
        else:
            cmem_gate["passed"] = False
            cmem_gate["error"] = f"anchor row not found in census ({len(row)})"

    gates["g2_anchor_mker6"] = mker_gate
    gates["g2_anchor_cmem_a1"] = cmem_gate
    gates["g2_passed"] = bool(mker_gate.get("passed", False) and cmem_gate.get("passed", False))

    # --- G-3 join: scored set, P6 cross-check -------------------------------
    if census is not None:
        n_total_crb = int(pd.read_csv(production_crb_path, usecols=["in_catalog"]).shape[0])
        n_in_catalogue = int(len(census))
        scored_set_size = n_total_crb - len(SCORED_SET_GAPS)
        n_out = int(census["outside"].sum())
        n_in = int((~census["outside"]).sum())
        p6 = parse_p6_line(Path(args.production_run), args.h_true)
        p6_matches = p6.get("found", False) and p6.get("n_recovered_1d") == n_in
        gates["g3_join"] = {
            "n_total_crb_rows": n_total_crb,
            "scored_set_size": scored_set_size,
            "n_in_catalogue": n_in_catalogue,
            "n_out": n_out,
            "n_in": n_in,
            "p6_log": p6,
            "p6_numerator_matches_n_in": bool(p6_matches),
            "passed": bool(n_in_catalogue > 0 and (p6_matches or not p6.get("found", False))),
        }
    else:
        gates["g3_join"] = {"passed": False, "error": "production census not built"}

    # --- G-4 scatter law: chi2_2 KS test + f_OUT envelope -------------------
    if census is not None and len(census) > 0:
        m2 = sky_mahalanobis2(
            census["delta_phi"].to_numpy(dtype=np.float64),
            census["delta_theta"].to_numpy(dtype=np.float64),
            census["phi_var"].to_numpy(dtype=np.float64),
            census["theta_var"].to_numpy(dtype=np.float64),
            census["cov"].to_numpy(dtype=np.float64),
        )
        m2_finite = m2[np.isfinite(m2)]
        ks = stats.kstest(m2_finite, "chi2", args=(2,))
        n_total_g4 = int(len(census))
        n_out_g4 = int(census["outside"].sum())
        f_out = float(census["outside"].mean())
        # G-4 envelope clause (draft §5, rev. 1 item 7): NOT an asymptotic comparison
        # of f_out against the envelope band. The exact two-sided binomial test of
        # n_out against the NEAREST envelope edge p — the realised count must not
        # reject Binomial(n_total, p) at alpha=0.05.
        nearest_edge = min(SCATTER_ENVELOPE, key=lambda edge: abs(f_out - edge))
        binom_result = stats.binomtest(
            n_out_g4, n_total_g4, p=nearest_edge, alternative="two-sided"
        )
        envelope_ok = bool(binom_result.pvalue >= 0.05)
        ks_ok = bool(ks.pvalue >= 0.05)
        gates["g4_scatter_law"] = {
            "n_finite_mahalanobis2": int(len(m2_finite)),
            "n_singular_covariance": int(len(m2) - len(m2_finite)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "ks_alpha": 0.05,
            "ks_passed": ks_ok,
            "f_outside": f_out,
            "n_out": n_out_g4,
            "n_total": n_total_g4,
            "envelope": list(SCATTER_ENVELOPE),
            "envelope_nearest_edge": nearest_edge,
            "envelope_binom_pvalue": float(binom_result.pvalue),
            "envelope_alpha": 0.05,
            "envelope_passed": envelope_ok,
            "passed": bool(ks_ok and envelope_ok),
        }
    else:
        gates["g4_scatter_law"] = {"passed": False, "error": "production census empty or not built"}

    # --- g-population disclosure --------------------------------------------
    gates["g_population_disclosure"] = count_harness_seeds(Path(args.harness_root)) | {
        "population": args.population,
        "note": "harness 0 mixed rows disclosure; production is a single pool (draft G-invariants).",
    }

    gates["passed"] = bool(
        gates["g1_catalogue_pin"]["passed"]
        and gates["g1_crb_pin"]["passed"]
        and gates["g1_git_commit_pin"]["passed"]
        and gates["g2_passed"]
        and gates["g3_join"]["passed"]
        and gates["g4_scatter_law"]["passed"]
    )
    gates["_census_frame"] = census
    return gates


# ---------------------------------------------------------------------------
# FIX ROUND 2: registered §2 statistic (real-mode only; not gate code).
# ---------------------------------------------------------------------------

T_MAT = 0.008  # materiality threshold in h (draft §3, rows #247/#280/#284)
# Registered mean_h offsets and Fisher information, re-baseline iiib (draft §2, row #302).
OFFSET_MEAN_H = {"1D": -0.0630, "2D": -0.0641}
INFO_I = {"1D": 3256.0, "2D": 2930.0}
CHANNEL_COL = {"1D": "combined_no_bh", "2D": "combined_with_bh"}


def robust_sd_mad(x: npt.NDArray[np.float64]) -> float:
    """MAD-scaled robust SD: ``1.4826 * median(|x - median(x)|)``.

    The registered convention (draft REVISION 1 item 1), fixed before any read to
    bound the influence of a 2-event outlier pair at ``n_IN = 66``.
    """
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad)


def sample_sd(x: npt.NDArray[np.float64]) -> float:
    """Plain sample SD (``ddof=1``), reported alongside the robust SD (draft rev.1 item 1)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x, ddof=1)) if x.size > 1 else float("nan")


def two_outlier_sensitivity(
    event_idx: npt.NDArray[np.int64], x: npt.NDArray[np.float64]
) -> dict[str, Any]:
    """The two IN-class events with the largest ``|s_e - median|`` (draft rev.1 item 1)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"median": float("nan"), "events": []}
    med = float(np.median(x))
    dev = np.abs(x - med)
    order = np.argsort(dev)[::-1][:2]
    return {
        "median": med,
        "events": [
            {"event_idx": int(event_idx[i]), "s_e": float(x[i]), "abs_dev": float(dev[i])}
            for i in order
        ],
    }


def stencil_scores(
    diag_csv: Path, event_ids: set[int] | None, h_lo: float, h_hi: float
) -> pd.DataFrame:
    """Per-event central-difference score ``s_e = (ln L(h_hi) - ln L(h_lo)) / (h_hi - h_lo)``
    on the stencil, both channels (draft §2, ``b4_imp_stage1_forecast.py:136-143``
    convention). Reads only the two stencil ``h`` rows of the diagnostics CSV.

    Returns:
        A frame with columns ``event_idx``, ``s_1D``, ``s_2D``.
    """
    usecols = ["event_idx", "h", "combined_no_bh", "combined_with_bh"]
    df = pd.read_csv(diag_csv, usecols=usecols)
    df = df[np.isclose(df["h"], h_lo) | np.isclose(df["h"], h_hi)]
    if event_ids is not None:
        df = df[df["event_idx"].isin(event_ids)]
    out: dict[str, pd.Series] = {}
    for chan, col in CHANNEL_COL.items():
        piv = df.pivot_table(index="event_idx", columns="h", values=col, aggfunc="first")
        cols = sorted(piv.columns)
        if not cols:
            out[f"s_{chan}"] = pd.Series(dtype=np.float64)
            continue
        lo_col = min(cols, key=lambda c: abs(c - h_lo))
        hi_col = min(cols, key=lambda c: abs(c - h_hi))
        lo = piv[lo_col].to_numpy(np.float64)
        hi = piv[hi_col].to_numpy(np.float64)
        ok = (lo > 0.0) & (hi > 0.0)
        s = np.full(lo.shape, np.nan)
        s[ok] = (np.log(hi[ok]) - np.log(lo[ok])) / (h_hi - h_lo)
        out[f"s_{chan}"] = pd.Series(s, index=piv.index)
    result = pd.DataFrame(out)
    result.index.name = "event_idx"
    return result.reset_index()


def cone_bias_floor_statistic(
    census: pd.DataFrame, scores: pd.DataFrame, channel: str, t_mat: float = T_MAT
) -> dict[str, Any]:
    """The draft §2 primary statistic for one channel: Delta h_cone, phi_cone, SE, Z, M.

    ``census`` carries ``event_idx``/``outside`` (build_census); ``scores`` carries
    ``event_idx``/``s_{channel}`` (stencil_scores). SE uses the rev.1 formula:
    ``SD_IN * sqrt(n_OUT + n_OUT**2/n_IN) / I_c``, SD_IN the MAD-scaled robust SD of
    the production in-catalogue IN class for this channel.
    """
    col = f"s_{channel}"
    merged = census.merge(scores[["event_idx", col]], on="event_idx", how="left")
    out_rows = merged[merged["outside"]]
    in_rows = merged[~merged["outside"]]
    s_out = out_rows[col].to_numpy(np.float64)
    s_in = in_rows[col].to_numpy(np.float64)
    n_out, n_in = int(len(s_out)), int(len(s_in))
    s_bar_in = float(np.nanmean(s_in)) if n_in else float("nan")
    sd_in_robust = robust_sd_mad(s_in) if n_in else float("nan")
    sd_in_plain = sample_sd(s_in)
    two_out = (
        two_outlier_sensitivity(in_rows["event_idx"].to_numpy(np.int64), s_in)
        if n_in
        else {"median": float("nan"), "events": []}
    )
    i_c = INFO_I[channel]
    delta_h = float(np.nansum(s_out - s_bar_in) / i_c) if n_out and n_in else float("nan")
    se = (
        float(sd_in_robust * np.sqrt(n_out + (n_out**2) / n_in) / i_c)
        if n_in and n_out
        else float("nan")
    )
    z = delta_h / se if se == se and se != 0 and delta_h == delta_h else float("nan")
    offset = OFFSET_MEAN_H[channel]
    phi = delta_h / offset if delta_h == delta_h else float("nan")
    m = t_mat / se if se == se and se != 0 else float("nan")
    return {
        "channel": channel,
        "n_out": n_out,
        "n_in": n_in,
        "s_bar_in": s_bar_in,
        "sd_in_mad_scaled": sd_in_robust,
        "sd_in_plain": sd_in_plain,
        "sd_ratio_plain_over_mad": (
            sd_in_plain / sd_in_robust
            if sd_in_robust == sd_in_robust and sd_in_robust != 0
            else float("nan")
        ),
        "two_outlier_sensitivity": two_out,
        "I_c": i_c,
        "delta_h_cone": delta_h,
        "SE": se,
        "Z": z,
        "offset_mean_h_minus_h_true": offset,
        "phi_cone": phi,
        "T_mat": t_mat,
        "M": m,
    }


def physics_floor_apply(
    mat: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Per-row physics floor (``tier0_bootstrap_jackknife.py`` P7-2c convention,
    replicated per draft §2's "frozen T0 scorer" citation): zero entries in a row are
    replaced by that row's own minimum nonzero value; a row that is entirely zero is
    excluded (flagged ``True`` in the returned exclusion mask).
    """
    out = mat.copy()
    exclude = np.zeros(mat.shape[0], dtype=bool)
    for i in range(mat.shape[0]):
        row = mat[i]
        nz = row[row > 0.0]
        if nz.size == 0:
            exclude[i] = True
            continue
        if np.any(row <= 0.0):
            out[i] = np.where(row <= 0.0, nz.min(), row)
    return out, exclude


def t0_mean_h(
    diag_csv: Path, channel_col: str, exclude_event_idx: set[int] | None = None
) -> dict[str, Any]:
    """Full-grid mean_h posterior, frozen T0-scorer convention: uniform prior over the
    diagnostics' own h grid, per-row physics floor, gradient-trapezoid weights
    (``tier0_bootstrap_jackknife.py`` P7-2a/P7-2c, cited verbatim in draft §2).
    """
    df = pd.read_csv(diag_csv, usecols=["event_idx", "h", channel_col])
    if exclude_event_idx:
        df = df[~df["event_idx"].isin(exclude_event_idx)]
    piv = df.pivot_table(index="event_idx", columns="h", values=channel_col, aggfunc="first")
    h_grid = np.array(sorted(piv.columns), dtype=np.float64)
    piv = piv.reindex(columns=h_grid)
    mat = piv.to_numpy(dtype=np.float64)
    floored, excl = physics_floor_apply(mat)
    kept = floored[~excl]
    logl = np.log(kept)
    logpost = np.sum(logl, axis=0)
    w = np.gradient(h_grid)
    lp = logpost - logpost.max()
    post = np.exp(lp)
    norm = float((post * w).sum())
    post_n = post / norm
    mean_h = float((post_n * h_grid * w).sum())
    return {
        "mean_h": mean_h,
        "n_events_used": int(kept.shape[0]),
        "n_events_floor_excluded": int(excl.sum()),
        "h_grid_n": int(h_grid.size),
    }


def leave_out_cross_check(
    diag_csv: Path, census: pd.DataFrame, delta_h_cone_1d: float, se_1d: float
) -> dict[str, Any]:
    """Exact leave-out counterfactual (draft §2 cross-check, registered to agree with
    the linear response within 2*SE): the frozen T0 scorer's full-sample mean_h
    (1D channel) vs the same statistic on the 1578 non-OUT events.
    """
    out_idx = {int(i) for i in census.loc[census["outside"], "event_idx"]}
    full = t0_mean_h(diag_csv, "combined_no_bh")
    excl = t0_mean_h(diag_csv, "combined_no_bh", exclude_event_idx=out_idx)
    delta_mean_h_leave_out = excl["mean_h"] - full["mean_h"]
    agree: bool | None
    if se_1d == se_1d and se_1d > 0 and delta_h_cone_1d == delta_h_cone_1d:
        agree = bool(abs(delta_mean_h_leave_out - delta_h_cone_1d) <= 2.0 * se_1d)
    else:
        agree = None
    return {
        "full_sample": full,
        "leave_out_OUT": excl,
        "n_OUT_excluded": len(out_idx),
        "delta_mean_h_leave_out": delta_mean_h_leave_out,
        "agrees_within_2SE_of_linear": agree,
    }


def harness_replicate(
    harness_root: Path,
    cat: pd.DataFrame,
    host_xyz: npt.NDArray[np.float64],
    k: float,
    h_lo: float,
    h_hi: float,
) -> dict[str, Any]:
    """The zero-compute harness replicate (draft §2): the same OUT/IN split on the
    post-flip S3 cell-S universes; ``f_OUT,harn`` (per-universe SE across universes)
    and ``Delta s = s_bar_OUT - s_bar_IN`` (between-universe SE, one-sample paired
    test of the per-universe Delta s values against 0), 1D channel.
    """
    per_universe: list[dict[str, Any]] = []
    for seed_dir in sorted(harness_root.glob("seed*_S")):
        crb_path = seed_dir / "simulations" / "prepared_cramer_rao_bounds.csv"
        diag_p = seed_dir / "simulations" / "diagnostics" / "event_likelihoods.csv"
        if not crb_path.exists() or not diag_p.exists():
            continue
        c = build_census(crb_path, cat, host_xyz, k)
        if len(c) == 0:
            continue
        s = stencil_scores(diag_p, set(c["event_idx"]), h_lo, h_hi)
        m = c.merge(s[["event_idx", "s_1D"]], on="event_idx", how="left")
        s_out = m.loc[m["outside"], "s_1D"].to_numpy(np.float64)
        s_in = m.loc[~m["outside"], "s_1D"].to_numpy(np.float64)
        n_total = len(m)
        n_out = int(m["outside"].sum())
        per_universe.append(
            {
                "seed_dir": seed_dir.name,
                "n_total": n_total,
                "n_out": n_out,
                "f_out": n_out / n_total if n_total else float("nan"),
                "s_bar_out": float(np.nanmean(s_out)) if len(s_out) else float("nan"),
                "s_bar_in": float(np.nanmean(s_in)) if len(s_in) else float("nan"),
                "delta_s": (
                    float(np.nanmean(s_out) - np.nanmean(s_in))
                    if len(s_out) and len(s_in)
                    else float("nan")
                ),
            }
        )
    if not per_universe:
        return {"per_universe": [], "n_universes": 0}
    f_out_vals = np.array(
        [u["f_out"] for u in per_universe if u["f_out"] == u["f_out"]], dtype=np.float64
    )
    delta_s_vals = np.array(
        [u["delta_s"] for u in per_universe if u["delta_s"] == u["delta_s"]], dtype=np.float64
    )
    f_out_se = (
        float(f_out_vals.std(ddof=1) / np.sqrt(f_out_vals.size))
        if f_out_vals.size > 1
        else float("nan")
    )
    if delta_s_vals.size > 1:
        t_res = stats.ttest_1samp(delta_s_vals, 0.0)
        delta_s_se = float(delta_s_vals.std(ddof=1) / np.sqrt(delta_s_vals.size))
        t_stat, p_val = float(t_res.statistic), float(t_res.pvalue)
    else:
        t_stat, p_val, delta_s_se = float("nan"), float("nan"), float("nan")
    return {
        "per_universe": per_universe,
        "n_universes": len(per_universe),
        "f_out_harn_mean": float(f_out_vals.mean()) if f_out_vals.size else float("nan"),
        "f_out_harn_SE": f_out_se,
        "delta_s_mean": float(delta_s_vals.mean()) if delta_s_vals.size else float("nan"),
        "delta_s_SE": delta_s_se,
        "delta_s_paired_t": t_stat,
        "delta_s_paired_p": p_val,
    }


def evaluate_dispositions(
    stat_1d: dict[str, Any],
    stat_2d: dict[str, Any],
    leave_out: dict[str, Any],
    t_mat: float = T_MAT,
) -> dict[str, Any]:
    """Three-valued outcome of every draft §4 disposition-table row.

    Each row's outcome is ``"TRUE"``/``"FALSE"`` when every quantity its trigger
    needs is finite, else ``"INPUTS-DO-NOT-EXIST"`` (matching the READ_RECORD.md §7
    existence-contract convention this fix succeeds).
    """

    def finite(*vals: float) -> bool:
        return all(v == v for v in vals)

    d1, phi1, se1, z1, m1 = (
        stat_1d["delta_h_cone"],
        stat_1d["phi_cone"],
        stat_1d["SE"],
        stat_1d["Z"],
        stat_1d["M"],
    )
    d2, phi2 = stat_2d["delta_h_cone"], stat_2d["phi_cone"]

    disagree_1d_2d = None
    if finite(d1, d2, phi1, phi2):
        disagree_1d_2d = bool(
            ((abs(d1) >= t_mat) != (abs(d2) >= t_mat)) or ((phi1 >= 0.2) != (phi2 >= 0.2))
        )

    leave_out_agree = leave_out.get("agrees_within_2SE_of_linear")
    leave_out_disagree = (not leave_out_agree) if leave_out_agree is not None else None

    immaterial = (
        "TRUE"
        if finite(d1, phi1, m1) and abs(d1) < t_mat and phi1 < 0.2 and m1 >= 3
        else ("FALSE" if finite(d1, phi1, m1) else "INPUTS-DO-NOT-EXIST")
    )
    cone_owns = (
        "TRUE"
        if finite(z1, phi1, m1) and abs(z1) > 3 and phi1 >= 0.5 and m1 >= 3
        else ("FALSE" if finite(z1, phi1, m1) else "INPUTS-DO-NOT-EXIST")
    )
    unpowered = (
        "TRUE"
        if finite(se1) and se1 > t_mat / 3
        else ("FALSE" if finite(se1) else "INPUTS-DO-NOT-EXIST")
    )
    intermediate_condition = None
    if finite(m1, z1, phi1, d1) and m1 >= 3:
        intermediate_condition = bool(
            (abs(z1) > 3 and 0.2 <= phi1 < 0.5)
            or (abs(d1) >= t_mat and phi1 < 0.2)
            or bool(disagree_1d_2d)
            or bool(leave_out_disagree)
        )
    intermediate = (
        "TRUE"
        if intermediate_condition is True
        else ("FALSE" if intermediate_condition is False else "INPUTS-DO-NOT-EXIST")
    )

    return {
        "rows": {
            "IMMATERIAL-FLOOR-SHARE": {
                "trigger": "|Delta_h_cone,1D|<T_mat AND phi_1D<0.2 AND M>=3",
                "outcome": immaterial,
            },
            "CONE-OWNS-FLOOR": {
                "trigger": "|Z_1D|>3 AND phi_1D>=0.5 AND M>=3",
                "outcome": cone_owns,
            },
            "INTERMEDIATE-UNPOWERED": {
                "trigger": "SE(Delta_h_cone,1D) > T_mat/3 (M<3)",
                "outcome": unpowered,
            },
            "INTERMEDIATE": {
                "trigger": (
                    "M>=3 AND (|Z|>3 & 0.2<=phi<0.5; or |Dh|>=T_mat & phi<0.2; "
                    "or 1D/2D disagree; or linear-vs-leave-out disagree>2SE)"
                ),
                "outcome": intermediate,
            },
        },
        "disagree_1D_2D": disagree_1d_2d,
        "leave_out_disagrees_gt_2SE": leave_out_disagree,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--production-crb", required=True)
    ap.add_argument("--production-run", required=True)
    ap.add_argument("--replicate-run", required=True)
    ap.add_argument("--harness-root", required=True)
    ap.add_argument("--population", type=int, default=200)
    ap.add_argument("--anchor-fleet-mker", required=True)
    ap.add_argument("--anchor-fleet-cmem", required=True)
    ap.add_argument("--sky-cone-k", type=float, default=1.5)
    ap.add_argument("--h-lo", type=float, default=0.725)
    ap.add_argument("--h-hi", type=float, default=0.735)
    ap.add_argument("--h-true", type=float, default=0.73)
    ap.add_argument("--crb-md5", required=True)
    ap.add_argument("--catalogue-md5", required=True)
    ap.add_argument(
        "--git-commit",
        default="1ec9514d",
        help="Expected GIT_COMMIT_AT_RUN.txt prefix (draft §5 G-1; row #302 head re-baseline).",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Gates G-1..G-4 + census only. Does NOT compute the registered statistic "
        "(Delta h_cone / phi_cone / SE / Z) — verifier independence (draft §7).",
    )
    args = ap.parse_args()

    print("Building sky-cone census + running gates G-1..G-4...")
    gates = run_gates(args)
    census = gates.pop("_census_frame")

    WORK_DIR.mkdir(exist_ok=True)
    with open(WORK_DIR / "cone_loss_gates.json", "w") as f:
        json.dump(gates, f, indent=1, default=str)
    print("GATES:", json.dumps(gates, indent=1, default=str))

    if census is not None:
        n_out = int(census["outside"].sum())
        n_in = int(len(census) - n_out)
        print(
            f"CENSUS: n_in_catalogue={len(census)} n_OUT={n_out} n_IN={n_in} "
            f"f_OUT={n_out / len(census) if len(census) else float('nan'):.4f}"
        )
        census.to_csv(WORK_DIR / "cone_loss_census.csv", index=False)

    if args.dry_run:
        # Dry-run always exits 0 (builder smoke-test contract): it reports gate
        # status without gating its own exit code on that status — a gate STOP is
        # information for the record (and for the launch decision), not a crash.
        # Real mode below still hard-stops on gate failure, before any statistic.
        verdict = "GATES-GREEN" if gates["passed"] else "INSTRUMENT-DEFECT"
        with open(args.out, "w") as f:
            json.dump(
                {"verdict": verdict, "gates": gates, "dry_run": True}, f, indent=1, default=str
            )
        print(
            f"--dry-run: G-1..G-4 + census only ({verdict}). Registered statistic "
            "(Delta h_cone/phi_cone/SE/Z) NOT computed (verifier independence, draft §7)."
        )
        return

    if not gates["passed"]:
        with open(args.out, "w") as f:
            json.dump({"verdict": "INSTRUMENT-DEFECT", "gates": gates}, f, indent=1, default=str)
        raise SystemExit("GATE STOP: one or more G-1..G-4 gates failed (see cone_loss_gates.json)")

    # --- FIX ROUND 2: registered §2 statistic, real mode ---------------------
    # (implemented by the builder per this fix's task instruction; only a RUNNER
    # agent may execute this branch against production data — see module docstring.)
    print("REAL MODE: computing the registered §2 statistic (1D + 2D)...")
    production_diag = (
        Path(args.production_run) / "simulations" / "diagnostics" / "event_likelihoods.csv"
    )
    scores = stencil_scores(production_diag, None, args.h_lo, args.h_hi)
    stat_1d = cone_bias_floor_statistic(census, scores, "1D")
    stat_2d = cone_bias_floor_statistic(census, scores, "2D")
    leave_out = leave_out_cross_check(
        production_diag, census, stat_1d["delta_h_cone"], stat_1d["SE"]
    )
    harness = harness_replicate(
        Path(args.harness_root),
        *load_catalogue(args.sky_cone_k)[:2],
        args.sky_cone_k,
        args.h_lo,
        args.h_hi,
    )
    dispositions = evaluate_dispositions(stat_1d, stat_2d, leave_out)

    result = {
        "verdict": "SCORED",
        "gates": gates,
        "dry_run": False,
        "statistic_1D": stat_1d,
        "statistic_2D": stat_2d,
        "leave_out_cross_check": leave_out,
        "harness_replicate": harness,
        "dispositions": dispositions,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print("STATISTIC 1D:", json.dumps(stat_1d, indent=1, default=str))
    print("STATISTIC 2D:", json.dumps(stat_2d, indent=1, default=str))
    print("LEAVE-OUT:", json.dumps(leave_out, indent=1, default=str))
    print("DISPOSITIONS:", json.dumps(dispositions, indent=1, default=str))


if __name__ == "__main__":
    main()
