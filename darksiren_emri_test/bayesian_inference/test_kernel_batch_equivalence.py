"""Differential gate: batched host kernel == scalar kernel, bit-for-bit.

``single_host_likelihood_batch`` is the vectorized twin of
``single_host_likelihood`` (perf/eval-vectorization): the host loop moves from
one starmap task per host into the array axis, per-host ``scipy.stats.norm``
frozen-distribution construction is replaced by an operation-order-identical
explicit Gaussian, and the quadratures replicate ``fixed_quad``'s exact affine
node map and reduction. Nothing numerical may change: this test asserts **bit
equality** (``==``, not approx) between the batch rows and per-host scalar
calls over the full kernel-parity regime grid, both as single-row batches and
as multi-host batches with perturbed neighbours (guarding against cross-host
contamination, broadcasting bugs, and reduction-order drift).

If this test ever fails after an intentional change to the *scalar* kernel,
the batch kernel must be updated in the same commit — the two are one
implementation with two entry points.

Scope note (measured 2026-07-10 on real seed400 data, 986 events, h=0.73):
bit equality holds exactly at these small batch sizes, but at production chunk
sizes (hundreds-to-thousands of hosts) float-path reassociation (BLAS/SIMD
stride effects) perturbs a sparse subset of per-host values — 560 of ~3.0M
values at ≤9.8e-15, moving 4 of 986 per-event likelihoods by ≤1.4e-16, with
the 1D channel byte-identical. That end-to-end drift is 5+ orders below the
rel=1e-9 pipeline-parity contract, which is the governing gate at scale.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import darksiren_emri.bayesian_inference.bayesian_statistics as bs
from darksiren_emri_test.bayesian_inference.test_kernel_parity import (
    _case_grid,
    _install_worker_globals,
)

_HOST_KEYS = ["host_phiS", "host_qS", "host_z", "host_z_error", "host_M", "host_M_error"]

# Deterministic per-host perturbations building a heterogeneous batch around
# each curated case (key, factor). Chosen to vary every host argument that
# enters a data-dependent branch (windows, clamps, Eddington shift).
_PERTURBATIONS: list[tuple[str, float]] = [
    ("host_z", 1.05),
    ("host_z", 0.95),
    ("host_z_error", 1.5),
    ("host_M", 1.3),
    ("host_M_error", 0.7),
    ("host_phiS", 1.01),
]


def _perturbed_hosts(kw: dict[str, Any]) -> list[dict[str, float]]:
    """The case host plus deterministic neighbours (7-row batch)."""
    base = {k: float(kw[k]) for k in _HOST_KEYS}
    variants = [dict(base)]
    for key, fac in _PERTURBATIONS:
        v = dict(base)
        v[key] = base[key] * fac
        variants.append(v)
    return variants


def _scalar_rows(kw: dict[str, Any], hosts: list[dict[str, float]]) -> npt.NDArray[np.float64]:
    _install_worker_globals()
    rows = []
    for hv in hosts:
        skw = dict(kw)
        skw.update(hv)
        rows.append(bs.single_host_likelihood(**skw))
    return np.array(rows, dtype=np.float64)


def _batch_rows(kw: dict[str, Any], hosts: list[dict[str, float]]) -> npt.NDArray[np.float64]:
    _install_worker_globals()
    arrays = {k: np.array([hv[k] for hv in hosts], dtype=np.float64) for k in _HOST_KEYS}
    return bs.single_host_likelihood_batch(
        arrays["host_phiS"],
        arrays["host_qS"],
        arrays["host_z"],
        arrays["host_z_error"],
        arrays["host_M"],
        arrays["host_M_error"],
        detection_index=kw["detection_index"],
        h=kw["h"],
        evaluate_with_bh_mass=kw["evaluate_with_bh_mass"],
        normalization_mode=kw["normalization_mode"],
    )


@pytest.mark.parametrize("case_id", sorted(_case_grid().keys()))
def test_batch_equals_scalar_multi_host(case_id: str) -> None:
    """7-host heterogeneous batch reproduces per-host scalar calls bit-for-bit."""
    kw = _case_grid()[case_id]
    hosts = _perturbed_hosts(kw)
    scalar = _scalar_rows(kw, hosts)
    batch = _batch_rows(kw, hosts)
    assert batch.shape == scalar.shape
    mism = np.nonzero(batch != scalar)
    assert mism[0].size == 0, (
        f"{case_id}: {mism[0].size} non-bit-identical values; first at "
        f"row {mism[0][0]}, col {mism[1][0]}: "
        f"scalar={scalar[mism[0][0], mism[1][0]]!r} batch={batch[mism[0][0], mism[1][0]]!r}"
    )


@pytest.mark.parametrize("case_id", sorted(_case_grid().keys()))
def test_batch_equals_scalar_single_host(case_id: str) -> None:
    """n=1 batch reproduces the scalar call bit-for-bit (row-shape contract)."""
    kw = _case_grid()[case_id]
    hosts = _perturbed_hosts(kw)[:1]
    scalar = _scalar_rows(kw, hosts)
    batch = _batch_rows(kw, hosts)
    assert batch.shape == scalar.shape
    assert (batch == scalar).all()


def test_batch_empty_hosts() -> None:
    """n=0 returns an empty, correctly-shaped array for both channels."""
    _install_worker_globals()
    empty = np.empty(0, dtype=np.float64)
    for wbh, ncols in ((False, 4), (True, 6)):
        out = bs.single_host_likelihood_batch(
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            detection_index=0,
            h=0.73,
            evaluate_with_bh_mass=wbh,
            normalization_mode="volume_deconv",
        )
        assert out.shape == (0, ncols)


class _SerialPool:
    """Minimal pool stand-in: serial starmap, fixed process count."""

    def __init__(self, processes: int) -> None:
        self._processes = processes

    def starmap(self, func: Any, jobs: list[tuple[Any, ...]]) -> list[Any]:
        return [func(*job) for job in jobs]


def test_starmap_host_batches_ordering_and_chunking() -> None:
    """Dispatcher preserves per-host order/values across chunk boundaries."""
    kw = _case_grid()["near_photoz_match_vd_4d"]
    hosts_kw = _perturbed_hosts(kw)
    _install_worker_globals()

    host_objs = []
    for hv in hosts_kw:
        host = type("_H", (), {})()
        host.phiS = hv["host_phiS"]
        host.qS = hv["host_qS"]
        host.z = hv["host_z"]
        host.z_error = hv["host_z_error"]
        host.M = hv["host_M"]
        host.M_error = hv["host_M_error"]
        host_objs.append(host)

    scalar = _scalar_rows(kw, hosts_kw)

    for processes in (1, 3, 16):
        _install_worker_globals()
        rows = bs._starmap_host_batches(
            _SerialPool(processes),  # type: ignore[arg-type]
            host_objs,
            detection_index=kw["detection_index"],
            h=kw["h"],
            evaluate_with_bh_mass=kw["evaluate_with_bh_mass"],
            normalization_mode=kw["normalization_mode"],
        )
        assert len(rows) == len(hosts_kw)
        got = np.array(rows, dtype=np.float64)
        assert (got == scalar).all(), f"processes={processes}: chunked dispatch changed values"

    # Memory-cap branch: a tiny _MAX_BATCH_CHUNK forces more chunks than
    # workers; values and order must be unaffected.
    original_cap = bs._MAX_BATCH_CHUNK
    try:
        bs._MAX_BATCH_CHUNK = 2
        _install_worker_globals()
        rows = bs._starmap_host_batches(
            _SerialPool(1),  # type: ignore[arg-type]
            host_objs,
            detection_index=kw["detection_index"],
            h=kw["h"],
            evaluate_with_bh_mass=kw["evaluate_with_bh_mass"],
            normalization_mode=kw["normalization_mode"],
        )
        got = np.array(rows, dtype=np.float64)
        assert (got == scalar).all(), "capped chunking changed values"
    finally:
        bs._MAX_BATCH_CHUNK = original_cap

    assert (
        bs._starmap_host_batches(
            _SerialPool(4),  # type: ignore[arg-type]
            [],
            detection_index=0,
            h=0.73,
            evaluate_with_bh_mass=True,
            normalization_mode="volume_deconv",
        )
        == []
    )
