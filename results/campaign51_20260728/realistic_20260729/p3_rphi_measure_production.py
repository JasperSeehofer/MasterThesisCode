"""[P3-RPHI] production re-measurement: r_phi(h) = Sigma^phi / Sigma^3D on the PRODUCTION
injection pool + PRODUCTION reduced catalogue, production pdet settings
(allow_shallow_pool=False, dl_bins=60, mass_bins=40, estimator=local_linear,
pdet_z_resolved=True). Run ON the cluster against cluster-native paths — this
is deliberately NOT the local dev-box harness copy (p3_rphi_measure.py), which
reads the B-SEL venue's mix200k pool via a repo-relative path. Zero-evaluate:
selection-leaf sums only (precompute_phi_marginal_survival +
precompute_global_catalog_selection), no --evaluate run.

Per docs/derivations/PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md Sec 6 item (i).

Provenance note (discrepancy flagged, not silently resolved): the task brief
pointed at DATA_INVENTORY.md row 78 / cluster/datasets.yaml `depth15_campaign`
($WS/injection_pool_depth15_50k, 500 files, "CURRENT (campaign canonical)").
Symlink-chasing the actual injections/ directory of the most recent CANONICAL
prodstack run (run_20260729_seed61000, generator_marginal + pdet_z_resolved,
the config declared canonical by [PHYSICS] ce6338e) shows its injections/
files are symlinks into $WS/injection_pool_mix200k_20260728 (705 csv files),
NOT injection_pool_depth15_50k (500 csv files) -- matching the pool
correspondence_1d.py's INJECTION_POOL_DIR already points at locally. The
datasets.yaml/DATA_INVENTORY.md "canonical" tag for depth15_50k is therefore
STALE documentation, superseded in practice by mix200k without a doc update.
This script measures against mix200k (the pool actually wired into the
canonical production evaluate path) and records both pool paths + file counts
in the output JSON so the author can adjudicate.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from darksiren_emri.bayesian_inference.bayesian_statistics import (
    precompute_global_catalog_selection,
    precompute_phi_marginal_survival,
)
from darksiren_emri.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)
from darksiren_emri.constants import SNR_THRESHOLD
from darksiren_emri.validation.correspondence_1d import (
    HOST_DRAW_Z_MAX,
    _load_galaxy_catalog_handler,
)

H_PROBE = [0.6, 0.665, 0.73, 0.795, 0.86]

# Production paths on the cluster (repo root = CWD when this script is run
# from ~/darksiren-emri per the cluster skill's PROJECT_ROOT convention).
WS_ROOT = Path("/pfs/work9/workspace/scratch/st_ac147838-emri")
PRODUCTION_INJECTION_POOL_DIR = str(WS_ROOT / "injection_pool_mix200k_20260728")
# Row-78 / datasets.yaml-tagged pool, retained for the record (NOT used as the
# measurement pool -- see module docstring). Not read by this script.
ROW78_TAGGED_INJECTION_POOL_DIR = str(WS_ROOT / "injection_pool_depth15_50k")
PRODUCTION_REDUCED_CATALOGUE_PATH = str(
    Path.cwd() / "darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv"
)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNKNOWN ({exc})"


def _md5(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _pool_file_count(pool_dir: str) -> int:
    return len(list(Path(pool_dir).glob("*.csv")))


def main() -> int:
    if not Path(PRODUCTION_INJECTION_POOL_DIR).is_dir():
        print(f"FATAL: injection pool dir not found: {PRODUCTION_INJECTION_POOL_DIR}")
        return 1
    if not Path(PRODUCTION_REDUCED_CATALOGUE_PATH).is_file():
        print(f"FATAL: reduced catalogue not found: {PRODUCTION_REDUCED_CATALOGUE_PATH}")
        return 1

    pool_file_count = _pool_file_count(PRODUCTION_INJECTION_POOL_DIR)
    row78_file_count = (
        _pool_file_count(ROW78_TAGGED_INJECTION_POOL_DIR)
        if Path(ROW78_TAGGED_INJECTION_POOL_DIR).is_dir()
        else None
    )
    print(f"Production injection pool: {PRODUCTION_INJECTION_POOL_DIR} ({pool_file_count} csv)")
    print(
        f"Row-78-tagged pool (NOT used): {ROW78_TAGGED_INJECTION_POOL_DIR} "
        f"({row78_file_count} csv)"
    )
    print(f"Reduced catalogue: {PRODUCTION_REDUCED_CATALOGUE_PATH}")

    catalogue_md5 = _md5(PRODUCTION_REDUCED_CATALOGUE_PATH)
    print(f"Reduced catalogue md5: {catalogue_md5}")

    handler = _load_galaxy_catalog_handler(PRODUCTION_REDUCED_CATALOGUE_PATH)
    det = SimulationDetectionProbability(
        injection_data_dir=PRODUCTION_INJECTION_POOL_DIR,
        snr_threshold=SNR_THRESHOLD,
        dl_bins=60,
        mass_bins=40,
        estimator="local_linear",
        expected_z_max=HOST_DRAW_Z_MAX,
        allow_shallow_pool=False,
        pdet_z_resolved=True,
    )
    phi_table = precompute_phi_marginal_survival(
        h_values=H_PROBE, detection_probability_obj=det, z_max_cap=HOST_DRAW_Z_MAX
    )
    sigma_3d = precompute_global_catalog_selection(
        h_values=H_PROBE,
        galaxy_catalog=handler,
        detection_probability_obj=det,
        with_bh_mass=False,
        z_max_cap=HOST_DRAW_Z_MAX,
        smear_sigma_z=False,
    )
    sigma_phi = precompute_global_catalog_selection(
        h_values=H_PROBE,
        galaxy_catalog=handler,
        detection_probability_obj=det,
        with_bh_mass=False,
        z_max_cap=HOST_DRAW_Z_MAX,
        smear_sigma_z=False,
        phi_survival_table=phi_table,
    )

    print("h      Sigma^phi        Sigma^3D         r_phi")
    r_phi = {}
    for h in H_PROBE:
        r_phi[h] = float(sigma_phi[h] / sigma_3d[h])
        print(f"{h:.3f}  {sigma_phi[h]:.6e}  {sigma_3d[h]:.6e}  {r_phi[h]:.6f}")

    lo, hi = min(H_PROBE), max(H_PROBE)
    chord_slope = float((np.log(r_phi[hi]) - np.log(r_phi[lo])) / (hi - lo))
    print(f"r_phi(0.73) = {r_phi[0.73]:.6f}; d ln r_phi/dh (chord) = {chord_slope:+.4f}")

    result = {
        "instrument": "p3_rphi_measure_production.py",
        "h_probe": H_PROBE,
        "sigma_phi": {str(h): float(sigma_phi[h]) for h in H_PROBE},
        "sigma_3d": {str(h): float(sigma_3d[h]) for h in H_PROBE},
        "r_phi": {str(h): r_phi[h] for h in H_PROBE},
        "r_phi_0p73": r_phi[0.73],
        "chord_slope_dlnr_phi_dh": chord_slope,
        "cluster_git_commit": _git_commit(),
        "injection_pool_dir_used": PRODUCTION_INJECTION_POOL_DIR,
        "injection_pool_file_count": pool_file_count,
        "row78_tagged_pool_dir_NOT_used": ROW78_TAGGED_INJECTION_POOL_DIR,
        "row78_tagged_pool_file_count": row78_file_count,
        "reduced_catalogue_path": PRODUCTION_REDUCED_CATALOGUE_PATH,
        "reduced_catalogue_md5": catalogue_md5,
        "pdet_settings": {
            "dl_bins": 60,
            "mass_bins": 40,
            "estimator": "local_linear",
            "allow_shallow_pool": False,
            "pdet_z_resolved": True,
            "expected_z_max": HOST_DRAW_Z_MAX,
        },
    }
    out_path = Path("p3_rphi_production_result.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
