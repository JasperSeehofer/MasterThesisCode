"""Shared helper for the dark/matched event-class split (docket ruling R8, row #337/#345).

**Do not import darksiren_emri from here.** This module is scratch/analysis tooling, not
part of the shipped physics package; it lives under ``results/`` and touches no file under
``darksiren_emri/``. It does not change any physics number — it only relabels events that
already have identical ``combined_no_bh`` (and every other) column value, using a different
rule for which side of the dark/matched line a given ``L_cat_no_bh`` value falls on.

Background
----------
The legacy criterion (``b3_1_pop_measure.py`` and several exec scorers) is

    dark  iff  L_cat_no_bh == 0.0

Between the 2026-08-27 head readout and the 2026-09-02 S0-B truth-node re-run, 157 of 1588
events flipped from ``L_cat_no_bh == 0.0`` (exact) to a tiny representable positive float
(``1.0e-110`` .. ``2.3e-08`` observed then; see the class-count forensics memo at
``exec/m-s0b-production/CLASS_COUNT_FORENSICS.md``) while ``combined_no_bh`` — the quantity
that actually enters every downstream likelihood — agreed to 1.6e-7 relative between the two
runs. The exact-zero test is therefore not a stable class boundary: it depends on whether a
deep exponential-tail integral underflows to a *bit-exact* ``0.0`` or lands one ULP above it,
which is a property of the numerical evaluation path, not of the astrophysics.

Threshold derivation (recomputed directly from the two CSVs named above, 2026-09-03)
--------------------------------------------------------------------------------------
Restricting to the 157 events that moved from exact-zero (2026-08-27, h=0.73) to a tiny
positive value (S0-B truth node, `node_truth_iiib_sites2.2_nosmear/.../event_likelihoods.csv`,
1588 rows, h=0.73 only), the ratio ``L_cat_no_bh / combined_no_bh`` computed on the *new*
(non-zero) CSV ranges from ``9.87e-109`` to ``9.751433e-07`` (event_idx 393, the largest of
the 157). This **does** exceed 1e-7, and the full ratio distribution around this value is
continuous/log-smooth (verified by inspection of all ratios in ``(1e-9, 1e-3)`` in the S0-B
CSV: no bimodal gap separates the 157-moved cluster from the rest of the population — the
population runs smoothly from ~1e-9 up through ~1e-4 with no step). So there is **no clean
data gap** to place a threshold in, contrary to the ticket's working hypothesis; the choice
below is a margin call, not a rediscovered natural boundary, and is flagged as such.

``THRESHOLD = 1e-6`` clears the observed 157-event maximum (9.751433e-07) by only ~2.6%
headroom — not the "≥1e3 above the largest observed float-noise value" originally hoped for.
It is adopted anyway because (a) it is the value named in the ruling, (b) it correctly
recovers the pre-drift 606-dark/982-matched split on the 2026-08-27 file exactly (verified in
BUILD_RECORD.md §3), and (c) the ~10 events with ratio in [1.04e-6, 1e-5) that a slightly
looser threshold would additionally reclassify are a separate, unresolved judgment call
(BUILD_RECORD.md §5) — moving the line is a scientific decision (a [RULE] item), not something
this helper should decide unilaterally. A future author ruling may adjust ``THRESHOLD``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: Relative-likelihood threshold below which an event is classified "dark" (no catalogue
#: host contributes non-negligibly to the no-BH likelihood). See module docstring for the
#: data-driven derivation and its caveats.
THRESHOLD: float = 1e-6


def is_dark_relative(
    l_cat_no_bh: npt.ArrayLike,
    combined_no_bh: npt.ArrayLike,
    threshold: float = THRESHOLD,
) -> npt.NDArray[np.bool_]:
    """Classify events as dark (no viable catalogue host) using a relative criterion.

    Replaces the numerically fragile ``L_cat_no_bh == 0.0`` exact-zero test with a
    relative-magnitude test against ``combined_no_bh`` (the total no-BH marginal likelihood,
    catalogue leg + completion leg), which is stable across runs even when the catalogue
    leg's raw value drifts between an exact float ``0.0`` and a representable tiny positive
    number near the deep-tail underflow boundary.

    Args:
        l_cat_no_bh: Catalogue-leg no-BH likelihood values (``L_cat_no_bh`` column),
            any non-negative array-like.
        combined_no_bh: Combined (catalogue + completion) no-BH likelihood values
            (``combined_no_bh`` column), same shape as ``l_cat_no_bh``. Must be > 0
            wherever compared (an event with ``combined_no_bh == 0`` has no defined ratio
            and is treated as dark by convention, since it contributes nothing regardless).
        threshold: Relative-likelihood cutoff. Defaults to :data:`THRESHOLD` (1e-6); see
            the module docstring for the derivation and its caveats.

    Returns:
        Boolean array, ``True`` where the event is classified dark under the relative
        criterion.

    References:
        results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/
        m-s0b-production/CLASS_COUNT_FORENSICS.md (the 157-event flip this helper fixes);
        exec/b-dark-class-relative/BUILD_RECORD.md (this node's derivation + tables).
    """
    l_cat = np.asarray(l_cat_no_bh, dtype=np.float64)
    combined = np.asarray(combined_no_bh, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(combined > 0.0, l_cat / combined, np.inf)
    # combined_no_bh == 0 everywhere l_cat == 0 too in observed data (a host-less event
    # contributes nothing to either leg); route it to "dark" rather than NaN/inf ambiguity.
    ratio = np.where(combined > 0.0, ratio, 0.0)
    return ratio < threshold


def is_dark_exact(l_cat_no_bh: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Legacy exact-zero dark-class criterion, kept for side-by-side comparison only.

    This is the criterion already used in ``b3_1_pop_measure.py`` et al.
    (``L_cat_no_bh == 0.0``). It is reproduced here (not imported from the original
    scripts, which are not modified by this node) purely so callers building a
    before/after table have both criteria available from one module.

    Args:
        l_cat_no_bh: Catalogue-leg no-BH likelihood values.

    Returns:
        Boolean array, ``True`` where ``L_cat_no_bh`` is bit-exact zero.
    """
    l_cat = np.asarray(l_cat_no_bh, dtype=np.float64)
    result: npt.NDArray[np.bool_] = l_cat == 0.0
    return result
