"""Why is F4's D(h) ~2x steeper than the old histogram? Decompose the source.

Checks three candidate mechanisms at h=0.73 vs h=0.76:
  (a) grid dl_max(h) differs  -> integration range differs
  (b) p_det(d_L) shape differs (global kernel smoothing)
  (c) the noisy low-d_L anchor (27-injection first bin) dominates the gap

Usage: uv run python scripts/dh_steepness_investigation.py
"""

import importlib.util
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability as F4,
)
from master_thesis_code.constants import OMEGA_DE, OMEGA_M, SNR_THRESHOLD  # noqa: F401
from master_thesis_code.physical_relations import (
    comoving_volume_element,
    dist_to_redshift,
    dist_vectorized,
)

INJ = "simulations/injections"


def load_old() -> type:
    src = subprocess.run(
        ["git", "show", "d1087f1^:master_thesis_code/bayesian_inference/"
         "simulation_detection_probability.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    p = Path(tempfile.mkdtemp()) / "old.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("old_sdp", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m.SimulationDetectionProbability


def pdet(est: object, dl: np.ndarray, h: float) -> np.ndarray:
    z0 = np.zeros_like(dl)
    return np.asarray(
        est.detection_probability_without_bh_mass_interpolated_zero_fill(dl, z0, z0, h=h),
        dtype=np.float64,
    )


def integrand(est: object, z: np.ndarray, h: float) -> np.ndarray:
    dl = np.asarray(dist_vectorized(z, h=h), dtype=np.float64)
    dvc = np.atleast_1d(np.asarray(comoving_volume_element(z, h=h), dtype=np.float64))
    return pdet(est, dl, h) * dvc


def main() -> None:
    Old = load_old()
    old = Old(INJ, SNR_THRESHOLD, dl_bins=60, mass_bins=40)
    f4 = F4(INJ, SNR_THRESHOLD, dl_bins=60, mass_bins=40)

    for h in (0.73, 0.76):
        print(f"\n================= h = {h} =================")
        dlm_o, dlm_f = old.get_dl_max(h), f4.get_dl_max(h)
        zmo, zmf = dist_to_redshift(dlm_o, h=h), dist_to_redshift(dlm_f, h=h)
        print(f"(a) dl_max:  old={dlm_o:.4f} Gpc (z_max={zmo:.4f})   "
              f"F4={dlm_f:.4f} Gpc (z_max={zmf:.4f})")

        # (b) p_det(d_L) shape on a common grid up to the smaller dl_max
        dlm = min(dlm_o, dlm_f)
        grid = np.linspace(1e-4, dlm, 12)
        po, pf = pdet(old, grid, h), pdet(f4, grid, h)
        print("(b) p_det(d_L):  d_L | old | F4 | F4-old")
        for d, a, b in zip(grid, po, pf):
            print(f"      {d:6.4f}  {a:6.4f}  {b:6.4f}  {b - a:+.4f}")

        # (c) D(h) decomposed: low-d_L band [0, 0.10 Gpc] vs rest, common z range
        zmax = min(zmo, zmf)
        # split at the d_L = 0.10 Gpc boundary (covers the noisy first bin ~0.062)
        z_split = dist_to_redshift(0.10, h=h)
        zlo = np.linspace(1e-6, z_split, 400)
        zhi = np.linspace(z_split, zmax, 400)
        d_lo_o = np.trapezoid(integrand(old, zlo, h), zlo)
        d_lo_f = np.trapezoid(integrand(f4, zlo, h), zlo)
        d_hi_o = np.trapezoid(integrand(old, zhi, h), zhi)
        d_hi_f = np.trapezoid(integrand(f4, zhi, h), zhi)
        print(f"(c) D(h) by band (trapz):")
        print(f"      d_L<0.10 Gpc:  old={d_lo_o:.4e}  F4={d_lo_f:.4e}  ΔF4-old={d_lo_f - d_lo_o:+.4e}")
        print(f"      d_L>0.10 Gpc:  old={d_hi_o:.4e}  F4={d_hi_f:.4e}  ΔF4-old={d_hi_f - d_hi_o:+.4e}")

    print("\nReading: compare ΔF4-old in the low band vs high band at each h, and")
    print("how those Δ's change from h=0.73 to h=0.76 — that reveals which d_L region")
    print("drives F4's steeper D(h) decline.")


if __name__ == "__main__":
    main()
