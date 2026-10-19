"""r-offset-subset — Phase A (BUILDER B1): the BLIND covariate table.

Registration: ``REGISTRATION_DRAFT.md`` (this directory), Research Graph 2 seed node
r-offset-subset. Implements draft §2's covariate family C1–C11 (§2 table) and §3
Phase A (blind table builder) and §6 gates G-1/G-2/G-3/g-population/g-precision, for
BOTH the ``iiib`` and ``joint_r1`` re-baseline event sets, event_idx keyed.

BLINDNESS CONTRACT (draft §3, binding on this file): this builder never opens
``exec/rd-2d-bootstrap-jackknife/`` (the influence data) or ``influence_*.csv`` in this
directory (Phase B's output) at any point — not to read, not to import, not to glob.
Only per-event covariates are computed and written here; no registered aggregate (AUC,
OR, p-value, Δ_strat) is computed by this script.

Covariate family (draft §2, columns of the output table plus ``event_idx``):
    C1  in_catalog            CRB truth column
    C2  hosted_exact          NOT is_dark_exact(L_cat_no_bh) at h=h_true
    C3  hosted_rel            NOT is_dark_relative(L_cat_no_bh, combined_no_bh) at h=h_true
    C3c log10_f_cat           log10(L_cat_no_bh / combined_no_bh), censored floor
    C4  z_gw                  dist_to_redshift(luminosity_distance, h=h_true)
    C5  log10_sky_area        log10(pi * cone_radius(...)**2), k = sky_cone_k
    C6  mass_window_retention n_2D / n_1D from the log ("possible hosts found" line)
    C7  log10_n_cand_1d       log10(1 + n_1D)
    C8  cone_outside          r-cone-loss OUT flag (in_catalog events only)
    C9  class_G               NOT written as a column — alias of C1 (draft §2, no
                               separate production footprint); documented in the build
                               record only.
    C10 log10_M               log10(CRB M)
    C10b low_M_timeout_bins12 M < low_m_edge
    C11 log10_snr             log10(CRB SNR) — reported-only, not in the Holm family

Usage:
    uv run python build_covariate_table.py [--dry-run]

All paths default to the values in REGISTRATION_DRAFT.md §1/§8; override only for
testing. See BUILD_RECORD_B1.md for the executed gate results and column definitions.
"""

import argparse
import hashlib
import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[5]

sys.path.insert(0, str(REPO_ROOT))
from darksiren_emri.physical_relations import dist_to_redshift  # noqa: E402

DEFAULT_PRODUCTION_CRB = (
    "results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv"
)
DEFAULT_IIIB_RUN = (
    "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/"
    "run_20260902_graph1_headrebaseline_iiib"
)
DEFAULT_JR1_RUN = (
    "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/"
    "run_20260902_graph1_headrebaseline_joint_r1"
)
DEFAULT_DARK_CLASS_MODULE = (
    "results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/"
    "b-dark-class-relative/dark_class.py"
)
DEFAULT_CONE_LOSS_MODULE = (
    "results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/"
    "r-cone-loss/cone_loss_reads.py"
)
DEFAULT_MKER_ANCHOR_FLEET = "results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825"

CRB_MD5 = "9a1f2a14384a9281c97ca3be312ddaab"
CATALOGUE_MD5 = "c52c13b5cab61f6b3f04bbe202550969"
CSV_MD5_IIIB = "8e6a2c18dc5838dd1d52641589243672"
CSV_MD5_JR1 = "745954a0fdee5f10878fb5e622a06144"
DARK_CLASS_MD5 = "841225ac9206ff18bf0145a81cac3a54"
GIT_COMMIT_PREFIX = "1ec9514dd1808c48b18c0792dce558e5bba0f116"

SCORED_SET_GAPS = (1203, 1356)
H_TRUE = 0.73
SKY_CONE_K = 1.5
FCAT_FLOOR_LOG10 = -320.0
LOW_M_EDGE = 169568.12917853205

ANCHOR_MKER_FLEET_ARM_SEED = "bc_900121_work"
ANCHOR_MKER_SEED = 900121
ANCHOR_MKER_EVENT_IDX = 20
ANCHOR_MKER_RADIUS = 1.4956979545757095e-03
ANCHOR_MKER_RADIUS_TOL = 1e-15

PROGRESS_RE = re.compile(r"Progess: detections: (\d+)/1588")
NO_CATALOG_RE = re.compile(r"no catalog results found")
POSSIBLE_HOSTS_RE = re.compile(r"possible hosts found (\d+)/(\d+)\.\.\.")


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


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, streamed in ``chunk``-byte blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_module_from_path(path: Path, name: str) -> types.ModuleType:
    """Import a standalone ``.py`` file (outside any package) by filesystem path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scored_event_idx(n_total_crb_rows: int, gaps: tuple[int, ...]) -> npt.NDArray[np.int64]:
    """The ascending event_idx values scored by the diagnostics pipeline (CRB rows minus gaps)."""
    all_idx = np.arange(n_total_crb_rows, dtype=np.int64)
    return all_idx[~np.isin(all_idx, np.array(gaps, dtype=np.int64))]


def parse_log_candidate_counts(run_dir: Path, h_true: float) -> dict[str, Any]:
    """Parse one venue's h_true log for per-detection-block candidate counts (C6/C7 source).

    Returns a dict with:
        n_blocks: number of "Progess: detections: k/1588" blocks found.
        n_1d: array of length n_blocks, n_1D candidates per block (0 for "no catalog results").
        n_2d: array of length n_blocks, n_2D candidates per block.
        n_no_catalog: count of blocks labelled "no catalog results found".
        log_path: the resolved log file path.
    """
    h_tag = f"{h_true:.4f}"
    log_path: Path | None = None
    for candidate in sorted(run_dir.glob("darksiren_emri_*.log")):
        text = candidate.read_text(errors="replace")
        if f"P6 host-recovery (h={h_tag}" in text:
            log_path = candidate
            break
    if log_path is None:
        raise FileNotFoundError(f"no log with P6 host-recovery (h={h_tag}...) under {run_dir}")

    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    n_1d: dict[int, int] = {}
    n_2d: dict[int, int] = {}
    labels: dict[int, str] = {}
    current_block: int | None = None
    for line in lines:
        m_progress = PROGRESS_RE.search(line)
        if m_progress:
            current_block = int(m_progress.group(1))
            continue
        if current_block is None or current_block in labels:
            continue
        if NO_CATALOG_RE.search(line):
            n_1d[current_block] = 0
            n_2d[current_block] = 0
            labels[current_block] = "no_catalog"
            continue
        m_hosts = POSSIBLE_HOSTS_RE.search(line)
        if m_hosts:
            n_1d[current_block] = int(m_hosts.group(1))
            n_2d[current_block] = int(m_hosts.group(2))
            labels[current_block] = "hosted"

    n_blocks = len(labels)
    if n_blocks == 0:
        raise ValueError(f"no detection blocks parsed from {log_path}")
    max_block = max(labels)
    if sorted(labels) != list(range(max_block + 1)):
        raise ValueError(f"detection blocks not contiguous 0..{max_block} in {log_path}")

    n_1d_arr = np.array([n_1d[k] for k in range(n_blocks)], dtype=np.int64)
    n_2d_arr = np.array([n_2d[k] for k in range(n_blocks)], dtype=np.int64)
    is_no_catalog = np.array([labels[k] == "no_catalog" for k in range(n_blocks)], dtype=np.bool_)
    n_no_catalog = int(is_no_catalog.sum())

    return {
        "n_blocks": n_blocks,
        "n_1d": n_1d_arr,
        "n_2d": n_2d_arr,
        "is_no_catalog": is_no_catalog,
        "n_no_catalog": n_no_catalog,
        "log_path": log_path,
    }


def build_crb_covariates(
    crb_path: Path,
    cone_loss_module: types.ModuleType,
    sky_cone_k: float,
    low_m_edge: float,
    h_true: float,
    scored_idx: npt.NDArray[np.int64],
) -> pd.DataFrame:
    """C1, C4, C5, C8, C10, C10b, C11 — covariates computable from the production CRB alone.

    Identical across venues (the CRB does not depend on which re-baseline run scored it),
    so this is computed once and reused for both output tables.
    """
    cols = [
        "M",
        "SNR",
        "luminosity_distance",
        "qS",
        "phiS",
        "delta_qS_delta_qS",
        "delta_phiS_delta_phiS",
        "delta_phiS_delta_qS",
        "host_galaxy_index",
        "in_catalog",
    ]
    crb = pd.read_csv(crb_path, usecols=cols)
    crb = crb.iloc[scored_idx].reset_index(drop=True)
    crb.insert(0, "event_idx", scored_idx)

    z_gw = np.array(
        [dist_to_redshift(float(d), h=h_true) for d in crb["luminosity_distance"]],
        dtype=np.float64,
    )

    cat, host_xyz, _catalogue_path = cone_loss_module.load_catalogue(sky_cone_k)
    log10_sky_area = np.full(len(crb), np.nan, dtype=np.float64)
    cone_outside = np.full(len(crb), np.nan, dtype=np.float64)  # NaN = not in_catalog
    for pos, row in crb.iterrows():
        theta_e, phi_e = float(row["qS"]), float(row["phiS"])
        radius = cone_loss_module.cone_radius(
            theta_e,
            float(row["delta_phiS_delta_phiS"]),
            float(row["delta_qS_delta_qS"]),
            float(row["delta_phiS_delta_qS"]),
            sky_cone_k,
        )
        log10_sky_area[pos] = float(np.log10(np.pi * radius * radius))
        if bool(row["in_catalog"]) and int(row["host_galaxy_index"]) >= 0:
            hidx = int(row["host_galaxy_index"])
            ev_xyz = cone_loss_module._polar_to_cartesian(  # noqa: SLF001 (reuse, not re-derive)
                np.array([theta_e]), np.array([phi_e])
            )[0]
            chord = float(np.linalg.norm(host_xyz[hidx] - ev_xyz))
            cone_outside[pos] = float(chord > radius)

    out = pd.DataFrame(
        {
            "event_idx": crb["event_idx"].to_numpy(dtype=np.int64),
            "C1_in_catalog": crb["in_catalog"].astype(bool).to_numpy(),
            "C4_z_gw": z_gw,
            "C5_log10_sky_area": log10_sky_area,
            "C8_cone_outside": cone_outside,
            "C10_log10_M": np.log10(crb["M"].to_numpy(dtype=np.float64)),
            "C10b_low_M_timeout_bins12": (crb["M"].to_numpy(dtype=np.float64) < low_m_edge),
            "C11_log10_snr": np.log10(crb["SNR"].to_numpy(dtype=np.float64)),
        }
    )
    return out


def build_venue_table(
    venue: str,
    run_dir: Path,
    event_likelihoods_csv: Path,
    dark_class_module: types.ModuleType,
    crb_covariates: pd.DataFrame,
    scored_idx: npt.NDArray[np.int64],
    h_true: float,
    fcat_floor_log10: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """C2, C3, C3c, C6, C7 for one venue, joined onto the shared CRB covariates.

    Returns (table, gate_report) where gate_report carries the G-3a decisive
    set-equality check and other venue-specific gate facts (no aggregate stats).
    """
    counts = parse_log_candidate_counts(run_dir, h_true)
    if counts["n_blocks"] != len(scored_idx):
        raise ValueError(
            f"{venue}: log block count {counts['n_blocks']} != scored set size {len(scored_idx)}"
        )
    log_event_idx = scored_idx  # block k -> the k-th scored event_idx in ascending order (G-3a)
    no_catalog_event_idx = set(int(i) for i in log_event_idx[counts["is_no_catalog"]])

    ev = pd.read_csv(
        event_likelihoods_csv, usecols=["event_idx", "h", "L_cat_no_bh", "combined_no_bh"]
    )
    ev = ev[np.isclose(ev["h"].to_numpy(dtype=np.float64), h_true, atol=1e-9)].copy()
    ev = ev.set_index("event_idx").reindex(scored_idx)
    if ev.isna().any().any():
        missing = ev.index[ev["L_cat_no_bh"].isna()].tolist()
        raise ValueError(f"{venue}: event_likelihoods.csv missing scored event_idx: {missing[:10]}")

    l_cat = ev["L_cat_no_bh"].to_numpy(dtype=np.float64)
    combined = ev["combined_no_bh"].to_numpy(dtype=np.float64)

    exact_zero_event_idx = set(int(i) for i, v in zip(scored_idx, l_cat) if v == 0.0)
    g3a_passed = no_catalog_event_idx == exact_zero_event_idx

    hosted_exact = ~dark_class_module.is_dark_exact(l_cat)
    hosted_rel = ~dark_class_module.is_dark_relative(l_cat, combined)

    censored = (l_cat == 0.0) | (combined == 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_cat = np.where(combined > 0.0, l_cat / combined, 0.0)
        log10_f_cat_finite = np.log10(np.where(f_cat > 0.0, f_cat, 1.0))
    log10_f_cat = np.where(censored, fcat_floor_log10, log10_f_cat_finite)

    n_1d = counts["n_1d"].astype(np.float64)
    n_2d = counts["n_2d"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mass_window_retention = np.where(n_1d > 0, n_2d / n_1d, np.nan)
    log10_n_cand_1d = np.log10(1.0 + n_1d)

    venue_cols = pd.DataFrame(
        {
            "event_idx": scored_idx,
            "C2_hosted_exact": hosted_exact,
            "C3_hosted_rel": hosted_rel,
            "C3c_log10_f_cat": log10_f_cat,
            "C3c_censored": censored,
            "C6_mass_window_retention": mass_window_retention,
            "C7_log10_n_cand_1d": log10_n_cand_1d,
        }
    )

    table = crb_covariates.merge(venue_cols, on="event_idx", how="inner", validate="one_to_one")
    if len(table) != len(scored_idx):
        raise ValueError(f"{venue}: join dropped rows ({len(table)} != {len(scored_idx)})")

    gate_report = {
        "venue": venue,
        "log_path": str(counts["log_path"]),
        "n_blocks": counts["n_blocks"],
        "n_no_catalog": counts["n_no_catalog"],
        "n_exact_zero": len(exact_zero_event_idx),
        "g3a_no_catalog_eq_exact_zero": bool(g3a_passed),
        "n_hosted_exact": int(hosted_exact.sum()),
        "n_hosted_rel": int(hosted_rel.sum()),
        "n_dark_exact": int((~hosted_exact).sum()),
        "n_dark_rel": int((~hosted_rel).sum()),
        "n_c3c_censored": int(censored.sum()),
        "n_c6_nan": int(np.isnan(mass_window_retention).sum()),
    }
    return table, gate_report


def run_gates(args: argparse.Namespace) -> dict[str, Any]:
    """G-1 pins + G-2 cone-radius anchor byte-id (draft §6 (i)/(vi))."""
    gates: dict[str, Any] = {}

    crb_path = REPO_ROOT / args.production_crb
    catalogue_path = (
        REPO_ROOT / "darksiren_emri" / "galaxy_catalogue" / "reduced_galaxy_catalogue.csv"
    )
    csv_iiib = REPO_ROOT / args.production_run / "simulations/diagnostics/event_likelihoods.csv"
    csv_jr1 = REPO_ROOT / args.replicate_run / "simulations/diagnostics/event_likelihoods.csv"
    dark_class_path = REPO_ROOT / args.dark_class_module

    gates["g1_crb_pin"] = _pin_check(crb_path, args.crb_md5)
    gates["g1_catalogue_pin"] = _pin_check(catalogue_path, args.catalogue_md5)
    gates["g1_csv_pin_iiib"] = _pin_check(csv_iiib, args.csv_md5_iiib)
    gates["g1_csv_pin_jr1"] = _pin_check(csv_jr1, args.csv_md5_jr1)
    gates["g1_dark_class_pin"] = _pin_check(dark_class_path, args.dark_class_md5)

    commit_checks = {}
    for tag, run_dir in (("iiib", args.production_run), ("joint_r1", args.replicate_run)):
        commit_path = REPO_ROOT / run_dir / "GIT_COMMIT_AT_RUN.txt"
        if commit_path.exists():
            commit = commit_path.read_text().strip()
            commit_checks[tag] = {
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

    gates["g1_passed"] = all(
        gates[k].get("passed", False)
        for k in (
            "g1_crb_pin",
            "g1_catalogue_pin",
            "g1_csv_pin_iiib",
            "g1_csv_pin_jr1",
            "g1_dark_class_pin",
            "g1_git_commit_pin",
        )
    )
    return gates


def _pin_check(path: Path, expected_md5: str) -> dict[str, Any]:
    if not path.exists():
        return {"passed": False, "error": f"file not found: {path}"}
    md5 = md5_of_file(path)
    return {"path": str(path), "md5": md5, "expected": expected_md5, "passed": md5 == expected_md5}


def run_g2_anchor(
    cone_loss_module: types.ModuleType, sky_cone_k: float, anchor_fleet: Path
) -> dict[str, Any]:
    """G-2 (vi): re-run the r-cone-loss R-MKER-6 radius anchor as this file's own byte-id."""
    crb_path = (
        anchor_fleet
        / ANCHOR_MKER_FLEET_ARM_SEED
        / f"seed{ANCHOR_MKER_SEED}"
        / "simulations"
        / "prepared_cramer_rao_bounds.csv"
    )
    if not crb_path.exists():
        return {"passed": False, "error": f"anchor CRB not found: {crb_path}"}
    cat, host_xyz, _ = cone_loss_module.load_catalogue(sky_cone_k)
    census = cone_loss_module.build_census(crb_path, cat, host_xyz, sky_cone_k)
    row = census[census["event_idx"] == ANCHOR_MKER_EVENT_IDX]
    if len(row) != 1:
        return {"passed": False, "error": f"anchor row not found (n={len(row)})"}
    found_radius = float(row["radius"].iloc[0])
    diff = abs(found_radius - ANCHOR_MKER_RADIUS)
    return {
        "found_radius": found_radius,
        "expected_radius": ANCHOR_MKER_RADIUS,
        "diff": diff,
        "tol": ANCHOR_MKER_RADIUS_TOL,
        "passed": bool(diff < ANCHOR_MKER_RADIUS_TOL),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production-crb", default=DEFAULT_PRODUCTION_CRB)
    p.add_argument("--production-run", default=DEFAULT_IIIB_RUN)
    p.add_argument("--replicate-run", default=DEFAULT_JR1_RUN)
    p.add_argument("--dark-class-module", default=DEFAULT_DARK_CLASS_MODULE)
    p.add_argument("--cone-loss-module", default=DEFAULT_CONE_LOSS_MODULE)
    p.add_argument("--anchor-fleet-mker", default=DEFAULT_MKER_ANCHOR_FLEET)
    p.add_argument("--h-true", type=float, default=H_TRUE)
    p.add_argument("--sky-cone-k", type=float, default=SKY_CONE_K)
    p.add_argument("--fcat-floor-log10", type=float, default=FCAT_FLOOR_LOG10)
    p.add_argument("--low-m-edge", type=float, default=LOW_M_EDGE)
    p.add_argument("--crb-md5", default=CRB_MD5)
    p.add_argument("--catalogue-md5", default=CATALOGUE_MD5)
    p.add_argument("--csv-md5-iiib", default=CSV_MD5_IIIB)
    p.add_argument("--csv-md5-jr1", default=CSV_MD5_JR1)
    p.add_argument("--dark-class-md5", default=DARK_CLASS_MD5)
    p.add_argument("--git-commit", default=GIT_COMMIT_PREFIX)
    p.add_argument(
        "--out-iiib",
        default=str(HERE / "covariate_table_iiib.csv"),
    )
    p.add_argument(
        "--out-jr1",
        default=str(HERE / "covariate_table_joint_r1.csv"),
    )
    p.add_argument("--sha256-out", default=str(HERE / "covariate_table.sha256"))
    p.add_argument("--record-out", default=str(HERE / "BUILD_RECORD_B1.md"))
    p.add_argument("--dry-run", action="store_true", help="run gates only, write no tables")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    gates = run_gates(args)
    if not gates["g1_passed"]:
        print("G-1 FAILED:", gates, file=sys.stderr)
        return 1

    dark_class_module = load_module_from_path(REPO_ROOT / args.dark_class_module, "dark_class")
    cone_loss_module = load_module_from_path(REPO_ROOT / args.cone_loss_module, "cone_loss_reads")

    g2 = run_g2_anchor(cone_loss_module, args.sky_cone_k, REPO_ROOT / args.anchor_fleet_mker)
    if not g2.get("passed", False):
        print("G-2 (vi) cone-radius anchor FAILED:", g2, file=sys.stderr)
        return 1

    if args.dry_run:
        print("G-1 PASSED, G-2 (vi) PASSED, dry-run: no tables written.")
        return 0

    n_total_crb_rows = int(
        pd.read_csv(REPO_ROOT / args.production_crb, usecols=["in_catalog"]).shape[0]
    )
    idx = scored_event_idx(n_total_crb_rows, SCORED_SET_GAPS)

    crb_covariates = build_crb_covariates(
        REPO_ROOT / args.production_crb,
        cone_loss_module,
        args.sky_cone_k,
        args.low_m_edge,
        args.h_true,
        idx,
    )

    table_iiib, gate_iiib = build_venue_table(
        "iiib",
        REPO_ROOT / args.production_run,
        REPO_ROOT / args.production_run / "simulations/diagnostics/event_likelihoods.csv",
        dark_class_module,
        crb_covariates,
        idx,
        args.h_true,
        args.fcat_floor_log10,
    )
    table_jr1, gate_jr1 = build_venue_table(
        "joint_r1",
        REPO_ROOT / args.replicate_run,
        REPO_ROOT / args.replicate_run / "simulations/diagnostics/event_likelihoods.csv",
        dark_class_module,
        crb_covariates,
        idx,
        args.h_true,
        args.fcat_floor_log10,
    )

    out_iiib = Path(args.out_iiib)
    out_jr1 = Path(args.out_jr1)
    table_iiib.to_csv(out_iiib, index=False)
    table_jr1.to_csv(out_jr1, index=False)

    sha_iiib = sha256_of_file(out_iiib)
    sha_jr1 = sha256_of_file(out_jr1)
    Path(args.sha256_out).write_text(f"{sha_iiib}  {out_iiib.name}\n{sha_jr1}  {out_jr1.name}\n")

    _write_build_record(
        Path(args.record_out),
        args,
        gates,
        g2,
        gate_iiib,
        gate_jr1,
        table_iiib,
        table_jr1,
        sha_iiib,
        sha_jr1,
        n_total_crb_rows,
    )

    print(
        "G-1 PASSED, G-2 (vi) PASSED, G-3a iiib:",
        gate_iiib["g3a_no_catalog_eq_exact_zero"],
        "joint_r1:",
        gate_jr1["g3a_no_catalog_eq_exact_zero"],
    )
    print(f"wrote {out_iiib} (sha256 {sha_iiib})")
    print(f"wrote {out_jr1} (sha256 {sha_jr1})")
    return 0


def _missing_counts(table: pd.DataFrame) -> dict[str, int]:
    return {col: int(table[col].isna().sum()) for col in table.columns}


def _write_build_record(
    record_path: Path,
    args: argparse.Namespace,
    gates: dict[str, Any],
    g2: dict[str, Any],
    gate_iiib: dict[str, Any],
    gate_jr1: dict[str, Any],
    table_iiib: pd.DataFrame,
    table_jr1: pd.DataFrame,
    sha_iiib: str,
    sha_jr1: str,
    n_total_crb_rows: int,
) -> None:
    lines: list[str] = []
    lines.append("# BUILD_RECORD_B1 — r-offset-subset blind covariate table (Builder B1)\n")
    lines.append(
        "Implements REGISTRATION_DRAFT.md §2 (C1–C11) + §3 Phase A + §6 gates "
        "G-1/G-2(vi)/G-3a/g-population/g-precision. This builder never opened "
        "`exec/rd-2d-bootstrap-jackknife/` or any `influence_*.csv` in this directory.\n"
    )

    lines.append("## G-1 pins\n")
    for key in (
        "g1_crb_pin",
        "g1_catalogue_pin",
        "g1_csv_pin_iiib",
        "g1_csv_pin_jr1",
        "g1_dark_class_pin",
        "g1_git_commit_pin",
    ):
        lines.append(f"- `{key}`: passed={gates[key].get('passed')}")
    lines.append(f"- **G-1 overall: {'GREEN' if gates['g1_passed'] else 'RED'}**\n")

    lines.append("## G-2 (vi) cone-radius anchor (R-MKER-6, re-run as this file's own byte-id)\n")
    lines.append(f"- found_radius = {g2.get('found_radius')}")
    lines.append(f"- expected_radius = {g2.get('expected_radius')}")
    lines.append(f"- |diff| = {g2.get('diff')} < tol {g2.get('tol')}")
    lines.append(f"- **passed: {g2.get('passed')}**\n")

    lines.append('## G-3a decisive gate: 606-line "no catalog results" = exact-zero L_cat_no_bh\n')
    for gate in (gate_iiib, gate_jr1):
        lines.append(
            f"- **{gate['venue']}**: n_no_catalog (log) = {gate['n_no_catalog']}, "
            f"n_exact_zero (L_cat_no_bh==0 at h={args.h_true}) = {gate['n_exact_zero']}, "
            f"set-equality passed = **{gate['g3a_no_catalog_eq_exact_zero']}**"
        )
    lines.append("")

    lines.append("## Population / row counts\n")
    lines.append(f"- production CRB total rows: {n_total_crb_rows}")
    lines.append(
        f"- scored set size (gaps {SCORED_SET_GAPS}): {len(table_iiib)} (iiib), {len(table_jr1)} (joint_r1)"
    )
    lines.append("")

    lines.append("## Class-label counts (R8 table cross-check, g-precision)\n")
    for gate in (gate_iiib, gate_jr1):
        lines.append(
            f"- **{gate['venue']}**: exact dark/hosted = {gate['n_dark_exact']}/{gate['n_hosted_exact']}; "
            f"relative dark/hosted = {gate['n_dark_rel']}/{gate['n_hosted_rel']}; "
            f"C3c censored (floor applied) = {gate['n_c3c_censored']}; "
            f"C6 NaN (n_1D==0) = {gate['n_c6_nan']}"
        )
    lines.append("")

    lines.append("## Column definitions (exact, as implemented)\n")
    lines.append("| id | column | definition | source |")
    lines.append("|---|---|---|---|")
    lines.append("| C1 | `C1_in_catalog` | CRB `in_catalog` | CRB |")
    lines.append(
        "| C2 | `C2_hosted_exact` | NOT `is_dark_exact(L_cat_no_bh)` at h=h_true (`dark_class.py`) | event_likelihoods.csv |"
    )
    lines.append(
        "| C3 | `C3_hosted_rel` | NOT `is_dark_relative(L_cat_no_bh, combined_no_bh, 1e-6)` at h=h_true | event_likelihoods.csv |"
    )
    lines.append(
        "| C3c | `C3c_log10_f_cat` | log10(L_cat_no_bh/combined_no_bh); censored floor "
        f"{args.fcat_floor_log10} where L_cat_no_bh==0 or combined_no_bh==0 (`C3c_censored` flag) | event_likelihoods.csv |"
    )
    lines.append("| C4 | `C4_z_gw` | `dist_to_redshift(luminosity_distance, h=h_true)` | CRB |")
    lines.append(
        "| C5 | `C5_log10_sky_area` | log10(pi * cone_radius(qS, phi_var, theta_var, cov, k)^2), "
        f"k={args.sky_cone_k}, `cone_radius` reused from `cone_loss_reads.py` | CRB |"
    )
    lines.append(
        '| C6 | `C6_mass_window_retention` | n_2D/n_1D from "possible hosts found n_1D/n_2D"; NaN if n_1D==0 | log |'
    )
    lines.append("| C7 | `C7_log10_n_cand_1d` | log10(1 + n_1D) | log |")
    lines.append(
        "| C8 | `C8_cone_outside` | chord > radius (r-cone-loss OUT flag); NaN for non-in_catalog rows | CRB + catalogue |"
    )
    lines.append(
        "| C9 | (alias of C1) | class G == in_catalog on production; no separate column written | — |"
    )
    lines.append("| C10 | `C10_log10_M` | log10(CRB `M`) | CRB |")
    lines.append(f"| C10b | `C10b_low_M_timeout_bins12` | CRB `M` < {args.low_m_edge} | CRB |")
    lines.append("| C11 | `C11_log10_snr` | log10(CRB `SNR`) | CRB |")
    lines.append("")

    lines.append("## Missing-value counts per column\n")
    for venue, table in (("iiib", table_iiib), ("joint_r1", table_jr1)):
        lines.append(f"**{venue}**:")
        for col, n_missing in _missing_counts(table).items():
            if n_missing:
                lines.append(f"- `{col}`: {n_missing}")
        if not any(_missing_counts(table).values()):
            lines.append("- (none)")
    lines.append("")

    n_c10b_iiib = int(table_iiib["C10b_low_M_timeout_bins12"].sum())
    n_c10b_jr1 = int(table_jr1["C10b_low_M_timeout_bins12"].sum())
    lines.append("## C10b testability (n >= 10 rule, disclosed for Phase C)\n")
    lines.append(
        f"- iiib: n C10b=True = {n_c10b_iiib} ({'testable' if n_c10b_iiib >= 10 else 'NOT-TESTED, n<10'})"
    )
    lines.append(
        f"- joint_r1: n C10b=True = {n_c10b_jr1} ({'testable' if n_c10b_jr1 >= 10 else 'NOT-TESTED, n<10'})"
    )
    lines.append("")

    lines.append("## Output files\n")
    lines.append(f"- `{Path(args.out_iiib).name}`: sha256 `{sha_iiib}`")
    lines.append(f"- `{Path(args.out_jr1).name}`: sha256 `{sha_jr1}`")
    lines.append("- both hashes also recorded in `covariate_table.sha256`")
    lines.append("")

    record_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
