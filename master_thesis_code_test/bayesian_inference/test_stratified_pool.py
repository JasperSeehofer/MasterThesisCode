"""Stratified-pool (issue #51) estimator tests for SimulationDetectionProbability.

Verifies the measure-match rule of SIZING_ANALYSIS.md §4 (results/
lcat_h_dependence_20260725/campaign_sizing_20260728/) as wired into the
estimator:

* Golden/no-op: a pool WITHOUT a ``stratum`` column behaves BIT-IDENTICALLY to
  the same pool with an all-'a' stratum column (hard backward-compatibility
  requirement).
* Mixture semantics: pool-marginal legs (pooled/1D survival, sky bands, FIX-2
  S(d_L|z), the wbh (K5) m-marginal shrinkage target, d_L grid support) use
  stratum-'a' rows only; the FIX-3 joint S(d_L | u, m) grid uses ALL rows.
* Stratum-column hygiene: NaN rows (legacy CSV concat) count as 'a'; unknown
  labels raise; a pool with zero 'a' rows raises.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from master_thesis_code.bayesian_inference.simulation_detection_probability import (
    SimulationDetectionProbability,
)


def _write_pool_csv(
    directory: str,
    z: npt.NDArray[np.float64],
    M: npt.NDArray[np.float64],  # noqa: N803
    snr: npt.NDArray[np.float64],
    d_L: npt.NDArray[np.float64],  # noqa: N803
    stratum: list[str] | None,
    qS: npt.NDArray[np.float64] | None = None,  # noqa: N803
) -> None:
    """Write one controlled injection CSV, optionally with a stratum column."""
    n = len(z)
    if qS is None:
        qS = np.linspace(0.1, 3.0, n)  # noqa: N806
    df = pd.DataFrame(
        {
            "z": z,
            "M": M,
            "phiS": np.zeros(n),
            "qS": qS,
            "SNR": snr,
            "h_inj": 0.73,
            "luminosity_distance": d_L,
        }
    )
    if stratum is not None:
        df["stratum"] = stratum
    df.to_csv(f"{directory}/injection_h_0p73_task_001.csv", index=False)


def _base_pool(
    seed: int = 42, n: int = 300
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Deterministic (z, M, SNR, d_L) arrays for a synthetic population pool."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.01, 1.0, n)
    M = rng.uniform(1e5, 5e6, n)  # noqa: N806
    d_L = 4.0 * z  # noqa: N806
    snr = rng.uniform(5.0, 60.0, n) / np.maximum(d_L, 1e-3)
    return z, M, snr, d_L


def _bc_rows(
    n: int = 150, seed: int = 7, snr_scale: float = 700.0
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Off-population 'b'/'c' rows: low-z / high-m cluster with strong SNR."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.02, 0.1, n)
    M = rng.uniform(4e6, 4.9e6, n)  # noqa: N806
    d_L = 4.0 * z  # noqa: N806
    snr = rng.uniform(0.8 * snr_scale, snr_scale, n) / np.maximum(d_L, 1e-3)
    return z, M, snr, d_L


_PROBE_DL = np.linspace(0.0, 2.5, 61)
_PROBE_Z = np.linspace(0.01, 1.0, 61)
_PROBE_M = np.linspace(1.2e5, 4.8e6, 61)


class TestStratumlessEqualsAllA:
    """Golden/no-op: stratum-less pool == all-'a' pool, bit-identical."""

    @pytest.mark.parametrize("flags_on", [False, True])
    def test_all_accessors_bit_identical(self, tmp_path: Path, flags_on: bool) -> None:
        d1 = tmp_path / "no_col"
        d2 = tmp_path / "all_a"
        d1.mkdir()
        d2.mkdir()
        z, M, snr, d_L = _base_pool()  # noqa: N806
        _write_pool_csv(str(d1), z, M, snr, d_L, stratum=None)
        _write_pool_csv(str(d2), z, M, snr, d_L, stratum=["a"] * len(z))

        s1 = SimulationDetectionProbability(
            str(d1),
            snr_threshold=20.0,
            pdet_z_resolved=flags_on,
            pdet_wbh_z_resolved=flags_on,
        )
        s2 = SimulationDetectionProbability(
            str(d2),
            snr_threshold=20.0,
            pdet_z_resolved=flags_on,
            pdet_wbh_z_resolved=flags_on,
        )

        z_kw = _PROBE_Z if flags_on else None
        r1 = s1.detection_probability_with_bh_mass_interpolated(
            _PROBE_DL, _PROBE_M, 0.0, 1.0, h=0.70, z=z_kw
        )
        r2 = s2.detection_probability_with_bh_mass_interpolated(
            _PROBE_DL, _PROBE_M, 0.0, 1.0, h=0.70, z=z_kw
        )
        assert np.array_equal(np.asarray(r1), np.asarray(r2))

        r1 = s1.detection_probability_without_bh_mass_interpolated(
            _PROBE_DL, 0.0, 1.0, h=0.70, z=z_kw
        )
        r2 = s2.detection_probability_without_bh_mass_interpolated(
            _PROBE_DL, 0.0, 1.0, h=0.70, z=z_kw
        )
        assert np.array_equal(np.asarray(r1), np.asarray(r2))

        assert np.array_equal(
            s1.survival_per_band(_PROBE_DL, z_kw), s2.survival_per_band(_PROBE_DL, z_kw)
        )
        assert s1.get_dl_max(0.70) == s2.get_dl_max(0.70)

        q1 = s1.quality_flags(0.70)
        q2 = s2.quality_flags(0.70)
        for key in ("n_total", "n_detected", "dl_edges", "M_edges"):
            assert np.array_equal(np.asarray(q1[key]), np.asarray(q2[key])), key


class TestMixturePoolLegMasks:
    """b/c rows must be invisible to pool-marginal legs and visible to the joint grid."""

    def _build_pair(
        self, tmp_path: Path, *, bc_snr_scale: float = 700.0
    ) -> tuple[SimulationDetectionProbability, SimulationDetectionProbability]:
        d_mix = tmp_path / "mix"
        d_a = tmp_path / "a_only"
        d_mix.mkdir()
        d_a.mkdir()
        z_a, M_a, snr_a, dl_a = _base_pool()  # noqa: N806
        z_b, M_b, snr_b, dl_b = _bc_rows(snr_scale=bc_snr_scale)  # noqa: N806
        n_bc = len(z_b)
        strata_bc = ["b", "c"] * (n_bc // 2)
        # Per-row qS fixed BEFORE concatenation so the a-rows carry identical
        # sky angles in both pools (sky-band comparability).
        qS_a = np.linspace(0.1, 3.0, len(z_a))  # noqa: N806
        qS_b = np.linspace(0.2, 2.9, n_bc)  # noqa: N806
        _write_pool_csv(
            str(d_mix),
            np.concatenate([z_a, z_b]),
            np.concatenate([M_a, M_b]),
            np.concatenate([snr_a, snr_b]),
            np.concatenate([dl_a, dl_b]),
            stratum=["a"] * len(z_a) + strata_bc,
            qS=np.concatenate([qS_a, qS_b]),
        )
        _write_pool_csv(str(d_a), z_a, M_a, snr_a, dl_a, stratum=["a"] * len(z_a), qS=qS_a)
        kw: dict[str, object] = {
            "snr_threshold": 20.0,
            "pdet_z_resolved": True,
            "pdet_wbh_z_resolved": True,
        }
        s_mix = SimulationDetectionProbability(str(d_mix), **kw)  # type: ignore[arg-type]
        s_a = SimulationDetectionProbability(str(d_a), **kw)  # type: ignore[arg-type]
        return s_mix, s_a

    def test_pooled_and_zres_survival_use_a_only(self, tmp_path: Path) -> None:
        # Leg: FIX-2 z-conditional S(d_L | z) (and the coinciding pooled 1D
        # accessor) — pool-marginal per SIZING_ANALYSIS.md §4 -> a-rows only.
        s_mix, s_a = self._build_pair(tmp_path)
        r_mix = s_mix.detection_probability_without_bh_mass_interpolated(
            _PROBE_DL, 0.0, 1.0, h=0.70, z=_PROBE_Z
        )
        r_a = s_a.detection_probability_without_bh_mass_interpolated(
            _PROBE_DL, 0.0, 1.0, h=0.70, z=_PROBE_Z
        )
        assert np.array_equal(np.asarray(r_mix), np.asarray(r_a))

    def test_sky_bands_use_a_only(self, tmp_path: Path) -> None:
        # Leg: per-band survival (pool-marginal) -> a-rows only.
        s_mix, s_a = self._build_pair(tmp_path)
        assert np.array_equal(
            s_mix.survival_per_band(_PROBE_DL, _PROBE_Z),
            s_a.survival_per_band(_PROBE_DL, _PROBE_Z),
        )

    def test_dl_support_uses_a_only(self, tmp_path: Path) -> None:
        # Leg: d_L grid support / z_max(h) for the full-volume denominator —
        # a-stratum object (marginal legs behave as if only the a-pool existed).
        s_mix, s_a = self._build_pair(tmp_path)
        assert s_mix.get_dl_max(0.70) == s_a.get_dl_max(0.70)

    def test_joint_grid_sees_all_rows(self, tmp_path: Path) -> None:
        # Leg: FIX-3 joint S(d_L | u, m) — measure-free given (u, m) -> ALL
        # rows. The b/c cluster (low z, high m, huge horizons) must lift the
        # joint survival there far above the a-only build.
        s_mix, s_a = self._build_pair(tmp_path)
        val_mix = s_mix.detection_probability_with_bh_mass_interpolated(
            3.0, 4.5e6, 0.0, 1.0, h=0.70, z=0.05
        )
        val_a = s_a.detection_probability_with_bh_mass_interpolated(
            3.0, 4.5e6, 0.0, 1.0, h=0.70, z=0.05
        )
        assert float(np.asarray(val_mix)) > float(np.asarray(val_a)) + 0.2

    def test_wbh_sm_shrinkage_target_uses_a_only(self, tmp_path: Path) -> None:
        # Leg: the (K5) m-marginal shrinkage target S_m(d_L | m) is
        # pool-marginal in z given m -> built from a-rows ONLY, while keeping
        # the JOINT build's conventions (all-row sigma_m / m_nodes / DLQ,
        # fix3 §3.4 "same machinery"). Verified by independent
        # reimplementation: recompute S_m from the a-stratum rows with the
        # mix build's own sigma_m/m_nodes/DLQ and require bitwise equality;
        # the same recomputation over ALL rows must NOT match (proves the
        # a-mask is load-bearing).
        s_mix, _ = self._build_pair(tmp_path, bc_snr_scale=30.0)
        sm_mix = s_mix._wbh_sm
        m_nodes = s_mix._wbh_m_nodes
        dlq = s_mix._wbh_dlq
        assert sm_mix is not None and m_nodes is not None and dlq is not None
        # Joint-convention sigma_m: Scott d=2 over ALL rows (the value used
        # inside _build_wbh_zres_survival).
        _, sigma_m = s_mix._compute_bandwidths(s_mix._dl_raw_all, s_mix._log_M_z_all)

        def _sm_from(
            m_rows: npt.NDArray[np.float64], d_hor_rows: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            order = np.argsort(d_hor_rows, kind="mergesort")
            d_sorted = d_hor_rows[order]
            m_sorted = m_rows[order]
            idx = np.searchsorted(d_sorted, dlq, side="left")
            inside = idx < len(d_sorted)
            idx_c = np.minimum(idx, len(d_sorted) - 1)
            diff = (m_sorted[:, None] - m_nodes[None, :]) / sigma_m
            km = np.exp(-0.5 * diff * diff)
            tot = km.sum(axis=0)
            suffix = np.cumsum(km[::-1, :], axis=0)[::-1, :]
            sm = np.where(inside[:, None], suffix[idx_c, :], 0.0) / tot[None, :]
            return np.asarray(np.clip(sm, 0.0, 1.0), dtype=np.float64)

        sm_a_rows = _sm_from(s_mix._log_M_z, s_mix._d_hor)
        sm_all_rows = _sm_from(s_mix._log_M_z_all, s_mix._d_hor_all)
        assert np.array_equal(sm_mix, sm_a_rows)
        assert not np.array_equal(sm_mix, sm_all_rows)


class TestStratumColumnHygiene:
    """NaN -> 'a' (legacy concat); unknown labels raise; zero 'a' rows raise."""

    def test_nan_rows_count_as_a(self, tmp_path: Path) -> None:
        d1 = tmp_path / "nan_col"
        d2 = tmp_path / "no_col"
        d1.mkdir()
        d2.mkdir()
        z, M, snr, d_L = _base_pool(seed=3, n=120)  # noqa: N806
        _write_pool_csv(str(d2), z, M, snr, d_L, stratum=None)
        # Legacy file (no column) + stratified all-'a' file in one directory:
        # concat leaves NaN for the legacy rows, which must count as 'a'.
        half = len(z) // 2
        df1 = pd.DataFrame(
            {
                "z": z[:half],
                "M": M[:half],
                "phiS": np.zeros(half),
                "qS": np.linspace(0.1, 3.0, half),
                "SNR": snr[:half],
                "h_inj": 0.73,
                "luminosity_distance": d_L[:half],
            }
        )
        df2 = df1.copy()
        df2 = pd.DataFrame(
            {
                "z": z[half:],
                "M": M[half:],
                "phiS": np.zeros(len(z) - half),
                "qS": np.linspace(0.1, 3.0, len(z) - half),
                "SNR": snr[half:],
                "h_inj": 0.73,
                "luminosity_distance": d_L[half:],
                "stratum": "a",
            }
        )
        df1.to_csv(f"{d1}/injection_h_0p73_task_001.csv", index=False)
        df2.to_csv(f"{d1}/injection_h_0p73_task_002.csv", index=False)

        s = SimulationDetectionProbability(str(d1), snr_threshold=20.0)
        # All rows must be treated as 'a': pooled survival over the full pool.
        assert int(np.count_nonzero(s._a_mask)) == len(z)

    def test_unknown_stratum_raises(self, tmp_path: Path) -> None:
        z, M, snr, d_L = _base_pool(seed=4, n=50)  # noqa: N806
        _write_pool_csv(str(tmp_path), z, M, snr, d_L, stratum=["a"] * 49 + ["x"])
        with pytest.raises(ValueError, match="unknown stratum"):
            SimulationDetectionProbability(str(tmp_path), snr_threshold=20.0)

    def test_no_a_rows_raises(self, tmp_path: Path) -> None:
        z, M, snr, d_L = _base_pool(seed=5, n=50)  # noqa: N806
        _write_pool_csv(str(tmp_path), z, M, snr, d_L, stratum=["b", "c"] * 25)
        with pytest.raises(ValueError, match="no stratum-'a'"):
            SimulationDetectionProbability(str(tmp_path), snr_threshold=20.0)
