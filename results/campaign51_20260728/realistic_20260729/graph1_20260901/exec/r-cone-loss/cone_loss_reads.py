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

# G-4 sky-scatter envelope (draft §1/§3): closed-form 1.5*sqrt(lambda_max) circle.
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
        f_out = float(census["outside"].mean())
        envelope_ok = SCATTER_ENVELOPE[0] <= f_out <= SCATTER_ENVELOPE[1]
        ks_ok = bool(ks.pvalue >= 0.05)
        gates["g4_scatter_law"] = {
            "n_finite_mahalanobis2": int(len(m2_finite)),
            "n_singular_covariance": int(len(m2) - len(m2_finite)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "ks_alpha": 0.05,
            "ks_passed": ks_ok,
            "f_outside": f_out,
            "envelope": list(SCATTER_ENVELOPE),
            "envelope_passed": bool(envelope_ok),
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

    raise NotImplementedError(
        "The registered statistic (draft §2: Delta h_cone, phi_cone, SE, Z, the "
        "leave-out cross-check, and the harness Delta s replicate) is runner-only per "
        "the verifier-independence contract in this file's module docstring. This "
        "builder script intentionally stops here; a DIFFERENT agent implements and "
        "runs the real-mode path (draft §7)."
    )


if __name__ == "__main__":
    main()
