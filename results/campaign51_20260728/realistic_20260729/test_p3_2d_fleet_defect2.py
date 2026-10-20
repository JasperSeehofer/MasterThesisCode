r"""DEFECT 2 (dead-row convention) regression tests for ``p3_2d_fleet.py``'s
``_identity_inputs_2d``, per the R-2D-2 repair task authorized in
p32d_residual_accounting_20260827.md.

Not collected by the repo's default pytest run (``testpaths = ["darksiren_emri_test"]`` in
pyproject.toml) -- this script lives next to ``p3_2d_fleet.py`` in the campaign results
directory, the same convention every other script in this directory follows. Run explicitly:

    uv run pytest results/campaign51_20260728/realistic_20260729/test_p3_2d_fleet_defect2.py -v

PA-2D-1 F16's registered dead-row convention: ``A2 = 0 => w2 = 0 => summand (1 - w2) = 1``.
The RHS instrument (``ca_rhs_scorer.py``'s ``_w2_from_csv_columns``, :1315-1324) already applies
this to EVERY F-0-accepted row via ``np.divide(a2, denom, out=np.zeros_like(a2), where=denom >
0.0)`` with no row excluded. The pre-repair ``_identity_inputs_2d`` instead left ``w2`` as
``NaN`` for any row with ``L_cat_with_bh <= 0`` (its own ``live`` mask) and
``stage_lhs2d`` indexed the accumulated sum by that SAME ``live`` mask
(``sum_acc = np.sum(1.0 - w2[live])``) -- silently EXCLUDING those rows instead of counting them
at summand 1, biasing the LHS side downward relative to the RHS side, which has no such
exclusion.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p3_2d_fleet  # noqa: E402


def test_identity_inputs_2d_dead_row_contributes_summand_one() -> None:
    """DEFECT 2 (R-2D-2) regression: an F-0-accepted row with ``A2 == 0`` -- whether because
    ``L_cat_with_bh <= 0`` (the pre-repair ``live`` mask's own "dead" case) or because
    ``alpha_G_phi == 0`` while ``L_cat_with_bh > 0`` -- must get ``w2 == 0`` and therefore
    contribute summand ``(1 - w2) == 1``, matching the RHS instrument's convention and PA-2D-1
    F16 exactly. This FAILS against the pre-repair code (row 1's ``w2`` is ``NaN``, not ``0.0``,
    because ``L_cat_with_bh <= 0`` makes it "dead" and the pre-repair code never assigns ``w2``
    for dead rows) and PASSES against the repair.
    """
    at = pd.DataFrame(
        {
            # row 0: an ordinary live row, A2 > 0 -- unaffected by the repair, sanity check.
            # row 1: dead per the OLD `live = L_cat_with_bh > 0` mask (L_cat_with_bh == 0, so
            #        A2 == 0 too) but denom > 0 (B_num_wbh > 0) -- F16's core case.
            # row 2: the "3 A2=B2=0 pathologies" case (denom == 0 too) -- the STANDARD variant
            #        (not the strict one) still gives w2 == 0, summand 1, matching the RHS's
            #        `out=np.zeros_like(a2)` default.
            "alpha_G_phi": [2.0, 5.0, 3.0],
            "L_cat_with_bh": [1.5, 0.0, 0.0],
            "B_num_wbh": [1.0, 4.0, 0.0],
        }
    )

    result = p3_2d_fleet._identity_inputs_2d(at)
    w2 = result["w2"]

    assert np.isfinite(w2).all(), f"w2 must be finite for every accepted row, got {w2}"
    np.testing.assert_allclose(w2[0], 0.75)  # a2=3.0, denom=4.0
    np.testing.assert_allclose(w2[1], 0.0)  # F16: A2=0, denom=4.0>0 -> w2=0 -> summand 1
    np.testing.assert_allclose(w2[2], 0.0)  # standard variant: denom=0 -> default w2=0

    # live/dead are retained as a diagnostic split only -- they must NOT gate the accumulated
    # sum any more (that was the defect).
    assert result["live"].tolist() == [True, False, False]
    assert result["dead"].tolist() == [False, True, True]

    sum_acc = float(np.sum(1.0 - w2))
    expected = (1.0 - 0.75) + 1.0 + 1.0
    assert np.isclose(sum_acc, expected), (
        f"sum_acc={sum_acc} must count EVERY accepted row (including dead ones at summand 1), "
        f"expected {expected}"
    )


def test_identity_inputs_2d_byte_identical_when_no_dead_no_pathological_rows() -> None:
    """Byte-identity pin (task item 5): when every row is live (``L_cat_with_bh > 0``) and no
    row is the ``A2 = B2 = 0`` pathology (``denom > 0`` everywhere), the pre-repair and repaired
    formulas are mathematically identical (the pre-repair ``w2[live] = ...`` covers the WHOLE
    array when ``live`` is all-``True``, and ``sum(1 - w2[live]) == sum(1 - w2)`` in that case) --
    the repair must reproduce the OLD per-row values and the OLD accumulated sum exactly.
    """
    rng = np.random.default_rng(42)
    n = 12
    alpha_g_phi = rng.uniform(0.5, 3.0, n)
    l_cat_wbh = rng.uniform(0.1, 5.0, n)  # all > 0 -> no dead rows
    b_num_wbh = rng.uniform(0.1, 5.0, n)  # all > 0 -> denom = a2 + b_num_wbh > 0 always
    at = pd.DataFrame(
        {
            "alpha_G_phi": alpha_g_phi,
            "L_cat_with_bh": l_cat_wbh,
            "B_num_wbh": b_num_wbh,
        }
    )

    result = p3_2d_fleet._identity_inputs_2d(at)

    a2 = alpha_g_phi * l_cat_wbh
    old_w2_no_dead_rows = a2 / (a2 + b_num_wbh)  # the OLD formula, valid when `live` is all-True

    np.testing.assert_array_equal(result["live"], np.full(n, True))
    np.testing.assert_allclose(result["w2"], old_w2_no_dead_rows)

    sum_acc = float(np.sum(1.0 - result["w2"]))
    old_sum_acc = float(np.sum(1.0 - old_w2_no_dead_rows))  # OLD stage_lhs2d: sum(1-w2[live])
    assert np.isclose(sum_acc, old_sum_acc)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
