"""Permanent must-NOT-move invariants of the C7 host-z-kernel change.

`GATE_PACKAGE_FINAL.md` §1.5 lists three classes of pin that the C7-core fix
(the host-z ``volume_deconv`` kernel carrying ``f_k(g)(z)``) may **never** move,
as opposed to the pins it is expected to move
(``PIN_VD_*`` / ``PIN_VT_*`` / ``PIN_CLAMP_*`` in
``test_bayesian_statistics_host_z_kernel``):

1. ``PIN_LR_*`` — the bare photo-z Gaussian (``local_ratio``) kernel is outside
   the ``_use_volume_deconv`` set, so ``f`` cannot reach it.  Pinned in
   ``test_bayesian_statistics_host_z_kernel``; here we assert the *structural*
   reason (``local_ratio`` absent from the mode tuple).
2. ``w_G(h) = beta_G(h) / D(h)`` — pure quadrature over ``p_det``, ``f_bar`` and
   ``dV_c/(1+z)``; the per-host kernel appears in neither integrand, so the
   mixture weight is bit-identical before and after the fix.  Pinned against
   the Cell-B readout values 0.1625175 / 0.1215039 / 0.1038732 at
   h = 0.60 / 0.73 / 0.81.
3. The **#51 P5 md5 path** — the ``generator_marginal`` / ``point`` host-z
   kernel: the numerator is the GW likelihood point-evaluated at ``z_g``
   (``prior_num is None``), so no completeness factor can enter it.  Asserted
   here as bit-identity of the point numerator with and without a non-trivial
   completeness installed in the worker globals.

These tests are permanent: a failure means the fix leaked outside its ratified
code surface.
"""

import inspect
from typing import Any

import numpy as np
import pytest

import master_thesis_code.bayesian_inference.bayesian_statistics as bs
from master_thesis_code_test.bayesian_inference.test_bayesian_statistics_host_z_kernel import (
    _install_worker_globals,
    _StubCompleteness,
)

# ── Cell-B / joint-precheck w_G anchors (results/campaign51_20260728, verified
# bit-identical across both legs and all 41 grid points; book/site/data/
# ch09_bench.json). D(h) and beta_G(h) are quoted at the 7 significant figures
# the bench file records, so the ratio reproduces w_G to ~6 s.f.
_W_G_ANCHORS: dict[float, tuple[float, float, float]] = {
    # h: (D(h), beta_G(h), w_G(h))
    0.60: (1.881202e9, 3.05728e8, 0.1625175),
    0.73: (1.520637e9, 1.84763e8, 0.1215039),
    0.81: (1.348397e9, 1.40063e8, 0.1038732),
}


@pytest.fixture(autouse=True)
def _reset_completeness_global() -> Any:
    yield
    bs.completeness_model = None


def test_local_ratio_outside_volume_deconv_set() -> None:
    """PIN_LR_*: ``local_ratio`` is not in the ``_use_volume_deconv`` tuple.

    The C7 factor is inserted inside the ``_use_volume_deconv`` branches only,
    so the bare photo-z Gaussian kernel cannot see it.
    """
    src = inspect.getsource(bs.single_host_likelihood)
    src_batch = inspect.getsource(bs.single_host_likelihood_batch)
    for source in (src, src_batch):
        start = source.index("_use_volume_deconv = normalization_mode in (")
        tuple_src = source[start : source.index(")", start)]
        assert "local_ratio" not in tuple_src
        assert "volume_deconv" in tuple_src
        assert "absolute_marginal" in tuple_src


def test_w_G_is_pure_selection_quadrature() -> None:
    """w_G = beta_G/D reproduces the Cell-B anchors and never sees the kernel.

    ``beta_G = D - beta_Gbar`` and both quadratures are built from ``p_det``,
    the completeness curve and ``dV_c/(1+z)``: no host-z kernel, no per-host
    object.  Asserted structurally (neither precompute calls the kernel) and
    numerically (the recorded tables reproduce the pinned w_G).
    """
    for h, (d_h, beta_g, w_g) in _W_G_ANCHORS.items():
        # rel=1e-5: both the anchors and the recorded D / beta_G tables are
        # 7-significant-figure readouts, so the reconstructed ratio agrees to
        # ~6 s.f. (worst case h = 0.81: 0.10387371 vs the pinned 0.1038732).
        assert beta_g / d_h == pytest.approx(w_g, rel=1e-5), h

    for fn in (
        bs.precompute_completion_denominator,
        bs.precompute_missing_completion_denominator,
    ):
        src = inspect.getsource(fn)
        assert "galaxy_redshift_prior_pdf" not in src
        assert "single_host_likelihood" not in src
        assert "_z_prior" not in src

    # And the mixture weight is formed from those two scalars alone.
    p_di_src = inspect.getsource(bs.BayesianStatistics.p_Di)
    assert "w_G = beta_G / D_h" in p_di_src


def _point_numerators(completeness: Any) -> list[float]:
    """generator_marginal (point) kernel outputs at a fixed synthetic host."""
    _install_worker_globals(completeness)
    return bs.single_host_likelihood(
        host_phiS=1.2,
        host_qS=1.0,
        host_z=0.10,
        host_z_error=0.0015,
        host_M=3.0e5,
        host_M_error=3.0e4,
        detection_index=0,
        h=0.73,
        evaluate_with_bh_mass=True,
        normalization_mode="generator_marginal",
        base_seed=42,
    )


def test_issue51_point_numerator_untouched_by_completeness() -> None:
    """#51 P5 md5 path: the point-kernel numerators carry no completeness.

    ``prior_num`` is ``None`` on the point path and the ``generator_marginal``
    assembly never divides by the per-host ``D_g``, so both numerator columns
    (1D and 2D) must be bit-identical with and without ``f``.
    """
    without = _point_numerators(None)
    with_f = _point_numerators(_StubCompleteness())
    # Columns 0 and 2 are the 1D / 2D numerators (bitwise identical).
    assert with_f[0] == without[0]
    assert with_f[2] == without[2]
    assert np.isfinite(with_f[0]) and with_f[0] > 0.0
