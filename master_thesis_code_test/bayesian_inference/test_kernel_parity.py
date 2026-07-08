"""Golden characterization pins for ``single_host_likelihood`` across regimes.

This is the *parity gate* for the Pipeline-B performance refactor
(``perf/eval-vectorization``): the per-host kernel is being rewritten to
vectorize over the host dimension, replace the ``hyp2f1`` luminosity distance
with a spline table, and drop per-host ``scipy`` frozen-distribution objects.
None of that may move a single per-host numeric — the H0 posterior MAP must be
bit-for-bit stable.

The tightest, fastest gate is to pin the exact output of the production kernel
at a curated grid of synthetic inputs spanning the regimes it sees in a real
run (near/far events, spec-z/photo-z hosts, matched/off-window hosts, the bare
vs volume-deconvolved z-prior, the 3D and 4D-with-BH-mass paths incl. the
seeded MC denominator). The kernel reads its detection state from module-level
globals (installed by ``child_process_init`` in production); here we install a
minimal multi-detection state directly, with a smooth analytic stub ``p_det``
whose grid is wide enough that the outside-grid quadrature weights stay zero.

Goldens live in ``golden/kernel_parity_pins.json`` and are committed. Because
every pinned value appears in git, a deliberate physics change shows up as a
golden-file diff in the same commit (the ``/physics-change`` visibility rule) —
a plain refactor must leave the golden untouched. Regenerate with::

    REGEN_KERNEL_GOLDEN=1 uv run pytest \
        master_thesis_code_test/bayesian_inference/test_kernel_parity.py

Regeneration is only legitimate on unmodified production or as the explicit,
reviewed value-update step of an approved physics change.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import master_thesis_code.bayesian_inference.bayesian_statistics as bs
from master_thesis_code_test.bayesian_inference.test_bh_denominator_semianalytic import (
    make_grid2d_pdet,
)

_GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "kernel_parity_pins.json"

# Deterministic MC stream seed for the 4D denominator (see single_host_likelihood).
_BASE_SEED = 20260708

# Tight parity tolerance. The 4D path is seeded-deterministic, so the same rel
# tolerance applies to it as to the analytic 3D path.
_REL_TOL = 1e-9


class _StubDetectionProbability:
    """Deterministic, smooth p_det with a grid spanning every test window.

    Matches the stub used in ``test_bayesian_statistics_host_z_kernel`` so the
    outside-grid quadrature weights are identically zero for in-window hosts.
    """

    def __init__(self) -> None:
        self._dl_centers: npt.NDArray[np.float64] = np.linspace(0.01, 120.0, 400)
        # Real 2-D RegularGridInterpolator p_det for the with-BH-mass path: the
        # semi-analytic denominator reads its M_z knots (interp_2d.grid[1]) and
        # relies on the interpolant being piecewise-linear in M_z (a smooth
        # analytic stub has no grid and would break the erf-sum's premise).
        self._grid2d = make_grid2d_pdet("peaked")

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self,
        d_L: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        return np.exp(-np.asarray(d_L, dtype=np.float64) / 5.0)

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: npt.NDArray[np.float64],
        M_z: npt.NDArray[np.float64],
        phi: npt.NDArray[np.float64],
        theta: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        return self._grid2d.detection_probability_with_bh_mass_interpolated(d_L, M_z, phi, theta, h)

    def _get_or_build_grid(self, h: float) -> tuple[Any, Any]:
        centers = self._dl_centers

        class _Interp:
            grid = (centers,)

        # (2-D interp for the erf-sum, 1-D-grid stub for the STAT-04 diagnostic).
        return self._grid2d._interp, _Interp()


# ── Synthetic detections (module-global slot state) ─────────────────────────
# Each detection contributes one slot to the stacked worker-state arrays. The
# 3D/4D Gaussians are diagonal in (phi, theta, d_L_frac[, M_z_frac]); with a
# diagonal 4D covariance the conditional variance of the M_z fraction is just
# its marginal variance and the projection vector vanishes (matches the
# existing host-z-kernel pin fixture).
_DETECTIONS: list[dict[str, float]] = [
    # index 0: near event, z ~ 0.1, 5% distance error.
    {
        "d_L": 0.47,
        "d_L_unc": 0.0235,
        "M": 3.3e5,
        "phi": 1.2,
        "theta": 1.0,
        "sig_phi": 0.02,
        "sig_theta": 0.02,
        "sig_dl_frac": 0.05,
        "sig_mz_frac": 0.10,
    },
    # index 1: far event, z ~ 0.5, wider localization + mass error.
    {
        "d_L": 2.90,
        "d_L_unc": 0.145,
        "M": 1.0e6,
        "phi": 2.0,
        "theta": 1.5,
        "sig_phi": 0.03,
        "sig_theta": 0.03,
        "sig_dl_frac": 0.06,
        "sig_mz_frac": 0.12,
    },
]


def _install_worker_globals() -> None:
    """Install stacked multi-detection worker state onto the bs module."""
    n = len(_DETECTIONS)
    bs.det_index_to_slot = {i: i for i in range(n)}
    bs.det_d_L_arr = np.array([d["d_L"] for d in _DETECTIONS])
    bs.det_d_L_unc_arr = np.array([d["d_L_unc"] for d in _DETECTIONS])
    bs.det_M_arr = np.array([d["M"] for d in _DETECTIONS])
    bs.det_phi_arr = np.array([d["phi"] for d in _DETECTIONS])
    bs.det_theta_arr = np.array([d["theta"] for d in _DETECTIONS])

    means_3d: list[npt.NDArray[np.float64]] = []
    cov_inv_3d: list[npt.NDArray[np.float64]] = []
    log_norm_3d: list[float] = []
    means_4d: list[npt.NDArray[np.float64]] = []
    cov_inv_4d: list[npt.NDArray[np.float64]] = []
    log_norm_4d: list[float] = []
    sigma2_cond: list[float] = []
    proj: list[npt.NDArray[np.float64]] = []

    for d in _DETECTIONS:
        cov3 = np.diag([d["sig_phi"] ** 2, d["sig_theta"] ** 2, d["sig_dl_frac"] ** 2])
        cov4 = np.diag(
            [d["sig_phi"] ** 2, d["sig_theta"] ** 2, d["sig_dl_frac"] ** 2, d["sig_mz_frac"] ** 2]
        )
        means_3d.append(np.array([d["phi"], d["theta"], 1.0]))
        cov_inv_3d.append(np.linalg.inv(cov3))
        log_norm_3d.append(-0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(cov3)[1]))
        means_4d.append(np.array([d["phi"], d["theta"], 1.0, 1.0]))
        cov_inv_4d.append(np.linalg.inv(cov4))
        log_norm_4d.append(-0.5 * (4 * np.log(2 * np.pi) + np.linalg.slogdet(cov4)[1]))
        sigma2_cond.append(d["sig_mz_frac"] ** 2)
        proj.append(np.zeros(3))

    bs.means_3d = np.array(means_3d)
    bs.cov_inv_3d = np.array(cov_inv_3d)
    bs.log_norm_3d = np.array(log_norm_3d)
    bs.means_4d = np.array(means_4d)
    bs.cov_inv_4d = np.array(cov_inv_4d)
    bs.log_norm_4d = np.array(log_norm_4d)
    bs.sigma2_cond_arr = np.array(sigma2_cond)
    bs.proj_arr = np.array(proj)
    bs.detection_probability = _StubDetectionProbability()


# ── Regime grid ─────────────────────────────────────────────────────────────
# Each case: id -> kwargs for single_host_likelihood. Only two z-prior modes are
# distinct at the kernel level: "local_ratio"/"global" (bare photo-z Gaussian)
# and "volume_deconv"/"volume_global" (volume-deconvolved). We pin one of each.
def _case_grid() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}

    def add(cid: str, **kw: Any) -> None:
        base: dict[str, Any] = {
            "host_phiS": _DETECTIONS[kw["detection_index"]]["phi"],
            "host_qS": _DETECTIONS[kw["detection_index"]]["theta"],
            "host_M": 3.0e5,
            "host_M_error": 3.0e4,
            "h": 0.73,
            "base_seed": _BASE_SEED,
        }
        base.update(kw)
        cases[cid] = base

    # -- Near event (index 0), matched host z ~ 0.1 --------------------------
    for mode in ("volume_deconv", "local_ratio"):
        for wbh in (False, True):
            tag = f"{'vd' if mode == 'volume_deconv' else 'lr'}_{'4d' if wbh else '3d'}"
            # spec-z host, matched
            add(
                f"near_specz_match_{tag}",
                detection_index=0,
                host_z=0.10,
                host_z_error=0.0015,
                normalization_mode=mode,
                evaluate_with_bh_mass=wbh,
            )
            # photo-z host, matched
            add(
                f"near_photoz_match_{tag}",
                detection_index=0,
                host_z=0.10,
                host_z_error=0.03,
                normalization_mode=mode,
                evaluate_with_bh_mass=wbh,
            )
            # offset host (partial window overlap)
            add(
                f"near_offset_{tag}",
                detection_index=0,
                host_z=0.085,
                host_z_error=0.01,
                normalization_mode=mode,
                evaluate_with_bh_mass=wbh,
            )

    # -- Near event, large mass error + wide photo-z -------------------------
    add(
        "near_bigMerr_vd_4d",
        detection_index=0,
        host_z=0.11,
        host_z_error=0.05,
        host_M=2.0e5,
        host_M_error=1.5e5,
        normalization_mode="volume_deconv",
        evaluate_with_bh_mass=True,
    )

    # -- Low-z window clamp (z_g < 4 sigma_z; numerator has no overlap) -------
    add(
        "lowz_clamp_vd_3d",
        detection_index=0,
        host_z=0.004,
        host_z_error=0.0015,
        normalization_mode="volume_deconv",
        evaluate_with_bh_mass=False,
    )

    # -- Far event (index 1), host z ~ 0.5 -----------------------------------
    for mode in ("volume_deconv", "local_ratio"):
        for wbh in (False, True):
            tag = f"{'vd' if mode == 'volume_deconv' else 'lr'}_{'4d' if wbh else '3d'}"
            add(
                f"far_specz_match_{tag}",
                detection_index=1,
                host_z=0.50,
                host_z_error=0.005,
                host_M=1.0e6,
                host_M_error=1.0e5,
                normalization_mode=mode,
                evaluate_with_bh_mass=wbh,
            )
            add(
                f"far_photoz_offset_{tag}",
                detection_index=1,
                host_z=0.46,
                host_z_error=0.03,
                host_M=8.0e5,
                host_M_error=2.0e5,
                normalization_mode=mode,
                evaluate_with_bh_mass=wbh,
            )

    return cases


def _run_case(kw: dict[str, Any]) -> list[float]:
    _install_worker_globals()
    return bs.single_host_likelihood(**kw)


def _compute_all() -> dict[str, list[float]]:
    return {cid: _run_case(kw) for cid, kw in _case_grid().items()}


def _load_golden() -> dict[str, list[float]]:
    with open(_GOLDEN_PATH) as f:
        data: dict[str, list[float]] = json.load(f)
    return data


@pytest.fixture(scope="module")
def golden() -> dict[str, list[float]]:
    if os.environ.get("REGEN_KERNEL_GOLDEN"):
        results = _compute_all()
        _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_GOLDEN_PATH, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        return results
    if not _GOLDEN_PATH.exists():
        pytest.skip(
            "kernel parity golden not generated; run with REGEN_KERNEL_GOLDEN=1 "
            "on unmodified production first"
        )
    return _load_golden()


@pytest.mark.parametrize("case_id", sorted(_case_grid().keys()))
def test_kernel_parity(case_id: str, golden: dict[str, list[float]]) -> None:
    """Every curated case reproduces its committed golden to rel=1e-9."""
    kw = _case_grid()[case_id]
    got = _run_case(kw)
    expected = golden[case_id]
    assert len(got) == len(expected), f"{case_id}: arity {len(got)} != {len(expected)}"
    for i, (g, e) in enumerate(zip(got, expected, strict=True)):
        if e == 0.0:
            assert g == 0.0, f"{case_id}[{i}]: expected exact 0.0, got {g}"
        else:
            assert g == pytest.approx(e, rel=_REL_TOL), (
                f"{case_id}[{i}]: {g} != {e} (rel {_REL_TOL})"
            )


def test_golden_covers_all_cases(golden: dict[str, list[float]]) -> None:
    """Guard against a stale golden missing newly added regime cases."""
    missing = sorted(set(_case_grid()) - set(golden))
    assert not missing, f"golden missing cases (regenerate): {missing}"
