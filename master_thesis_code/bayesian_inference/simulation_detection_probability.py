"""Simulation-based detection probability from injection campaign data.

Replaces :class:`DetectionProbability` (KDE-based) with a histogram-binned
approach that loads raw injection CSVs, applies an SNR threshold at evaluation
time, and builds P_det grids via SNR rescaling.

All injection data is pooled regardless of the Hubble parameter value used
during the injection campaign.  When querying P_det at a target h value, each
event's SNR is rescaled using the exact relation:

    SNR(h_target) = SNR_raw * d_L(z, h_inj) / d_L(z, h_target)

This is exact because the GW strain amplitude scales as 1/d_L, while the
waveform frequency content depends only on source-frame parameters (which are
h-independent).  See Gray et al. (2020) arXiv:1908.06050 Section III.B-C and
Laghi et al. (2021) arXiv:2102.01708 Section III.A.

Grids are built in (d_L, M) space so that query-time lookups avoid the
expensive ``dist_to_redshift`` numerical inversion (fsolve).
"""

# ASSERT_CONVENTION: natural_units=SI, distance=Gpc, mass=solar_masses,
#   h=dimensionless_H0_over_100, SNR=dimensionless

import glob
import logging
import re
import warnings
from collections import OrderedDict
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from master_thesis_code.physical_relations import dist_vectorized

logger = logging.getLogger(__name__)

# Default number of bins for the P_det grids
_DEFAULT_DL_BINS: int = 60
_DEFAULT_M_BINS: int = 40

# Maximum number of cached grids (LRU eviction)
_MAX_CACHE_SIZE: int = 20

# Default prior support of the trial Hubble constant h (LamCDMScenario, see
# cosmological_model.py:304-305).  Used to compute an h-stable luminosity-
# distance histogram support so that bin edges do not drift with the trial
# cosmology.  Pre-fix bug: per-h `dl_max = max(dl_vals)*1.1` made
# individual injections cross bin boundaries as h shifted by 0.001,
# producing coherent 5-25% jumps in p_det that summed to visible spikes
# in Sigma log L_i.  See .planning/debug/posterior-noisy-peak.md and
# .planning/debug/F1_literature_audit.md.
_DEFAULT_H_PRIOR_MIN: float = 0.60
_DEFAULT_H_PRIOR_MAX: float = 0.86

# 10% headroom above the max d_L over the prior support, preserved from
# the original per-h `* 1.1` factor for empty-bin coverage in the
# principled-bridge extrapolation regime.
_DL_PADDING_FACTOR: float = 1.1

# F4 (Phase 49): Nadaraya-Watson kernel p_det estimator.  The histogram
# form p_det = N_det/N_total is a step function in (d_L, M_z, h); injection
# d_L_k(h) crossing fixed bin edges as h shifts produces integer-count
# discontinuities (96% of post-F1 Σ(Δp)², per
# ``scripts/bias_investigation/test_29_snr_threshold_crossings.py``).
# Replacement: kernel-weighted estimator
#     p̂(d_L_q, M_q, h) = Σ_k K_k(h) · 1[SNR_k(h)≥thr] / Σ_k K_k(h)
# with truncated Gaussian kernel K_k centered at (d_L_q, log M_q).  Bandwidths
# from Scott's rule (N^(-1/(d+4)) with d=2 → N^(-1/6)) on the injection sample.
# Refs: Nadaraya (1964); Watson (1964); Scott (1992) Ch. 6;
# Farr (2019) arXiv:1904.10879 Sec III; Mandel-Farr-Gair (2019) Eq. 18.
_KERNEL_TRUNCATION_SIGMA: float = 3.0  # cut kernel beyond 3σ in d_L (sparse search)
_SCOTT_EXPONENT_2D: float = -1.0 / 6.0  # Scott's rule for d=2 dimensions

# F4-v2 (2026-06): local-LINEAR regression in d_L replaces local-constant
# (Nadaraya-Watson).  NW has O(h) boundary bias: at the d_L→0 edge the
# one-sided kernel averages in far-field non-detections, collapsing p̂ to
# ~0.5 where ground truth = 1.0 (every nearby source is detected).  This
# biases D(h)=∫p_det dV_c/dz and pushes the H0 MAP high.  Local-linear is
# design-adaptive: the slope term absorbs the local trend so a one-sided
# boundary neighbourhood no longer drags the estimate (boundary bias
# O(h)→O(h²), no variance penalty).  In a symmetric interior neighbourhood
# the local-linear intercept reduces to the NW ratio, so the change is
# confined to the boundary.  Refs: Fan (1992) JASA 87:998 (design-adaptive,
# automatic boundary correction); Fan & Gijbels (1996) Local Polynomial
# Modelling Ch. 3 Eq. (3.5-3.6); Wand & Jones (1995) Kernel Smoothing §5.3.
_Estimator = Literal["local_linear", "nadaraya_watson"]
_DEFAULT_ESTIMATOR: _Estimator = "local_linear"


def _local_linear_p_det(
    u: npt.NDArray[np.float64],
    w: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Local-linear-in-d_L detection probability at one query center.

    Fits the weighted line ``p ≈ a + b·u`` (``u = d_L_k − d_L_q``) by weighted
    least squares and returns the intercept ``a`` (the estimate at ``u=0``),
    clipped to ``[0, 1]``.  Vectorized over an arbitrary trailing axis of
    ``w`` and ``y`` (used for the per-M-center 2D build); ``u`` is shared
    across that axis.

    Eq. (3.5-3.6) in Fan & Gijbels (1996), *Local Polynomial Modelling*.

    Args:
        u: Local d_L residuals ``d_L_k − d_L_q`` [Gpc], shape ``(N_loc,)``.
        w: Kernel weights, shape ``(N_loc,)`` or ``(N_loc, M)``.
        y: Detection indicators in {0,1}, shape matching ``w``.

    Returns:
        Local-linear intercept(s) clipped to ``[0, 1]``; scalar-axis shape of
        ``w`` minus the leading ``N_loc`` axis.
    """
    # Sums for the 2x2 weighted normal equations.  u broadcasts over the
    # trailing (M-center) axis when w/y are 2D.
    if w.ndim == 2:
        u_col = u[:, None]
    else:
        u_col = u
    s0 = w.sum(axis=0)
    s1 = (w * u_col).sum(axis=0)
    s2 = (w * u_col * u_col).sum(axis=0)
    t0 = (w * y).sum(axis=0)
    t1 = (w * u_col * y).sum(axis=0)

    det = s0 * s2 - s1 * s1  # determinant of the 2x2 design [Gpc^2]
    # Local-linear intercept a = (S2·T0 − S1·T1) / (S0·S2 − S1²).
    # Singular guard: where det≈0 (degenerate one-point neighbourhood) fall
    # back to the local-constant ratio T0/S0 (Nadaraya-Watson).
    eps = np.finfo(np.float64).tiny
    a_ll = np.divide(s2 * t0 - s1 * t1, det, out=np.zeros_like(s0), where=np.abs(det) > eps)
    a_lc = np.divide(t0, s0, out=np.zeros_like(s0), where=s0 > 0.0)
    a = np.where(np.abs(det) > eps, a_ll, a_lc)
    return np.asarray(np.clip(a, 0.0, 1.0), dtype=np.float64)


# ----------------------------------------------------------------------
# Out-of-grid detection-probability behavior (1D and 2D channels).
#
# As of 2026-05-05, both channels use a principled monotonic-asymptotic
# extrapolation scheme implemented inside the corresponding interpolated
# probability functions:
#
# * Saturating face (d_L < d_L_min): linear bridge from (d_L_min, p_edge)
#   to (0, 1).  The d_L=0 limit is the unique natural physical scale
#   where the asymptote p_det=1 is exact (no source closer than the
#   observer).
#
# * Suppressing faces (d_L > d_L_max; M_z out of bounds in the 2D case):
#   slope-matched linear extrapolation from the boundary, clamped to
#   [0, p_edge] (Option A directional clamp).
#
# * Corner cells (2D, both axes outside): min of the two face
#   extrapolations.
#
# The scheme replaces the Phase 45 Plan 45-02/04 anchor approach (Wilson
# 95% LB at d_L=0 plus an empirical point estimate at d_L=0.05), which
# was fitted to production-posterior behavior and incompatible with the
# project's principled-physics preference.  See
# `.planning/2D-CHANNEL-AUDIT-20260505.md` for the full audit and
# rationale.
# ----------------------------------------------------------------------


class SimulationDetectionProbability:
    """Simulation-based detection probability from injection campaign data.

    Loads raw injection CSVs (z, M, phiS, qS, SNR, h_inj, luminosity_distance),
    pools ALL events regardless of h_inj, and builds P_det grids on-the-fly
    via SNR rescaling at query time.

    For a source at redshift z injected at h_inj with measured SNR_raw, the
    rescaled SNR at target h is:

        SNR(h) = SNR_raw * d_L(z, h_inj) / d_L(z, h)

    This is exact because h(t) ~ 1/d_L for gravitational wave strain, while
    the waveform shape depends only on source-frame parameters.

    Grids are cached with LRU eviction (max 20 entries) for performance.

    Args:
        injection_data_dir: Directory containing injection CSV files matching
            ``injection_h_*_task_*.csv`` or ``injection_h_*.csv``.
        snr_threshold: SNR threshold for detection. Events with
            SNR >= snr_threshold are considered detected.
        h_grid: **Deprecated.** Previously used to specify h grid points for
            pre-computed grids.  Now ignored (grids are built on-the-fly via
            SNR rescaling).  Passing this parameter emits a deprecation
            warning.
        _force_unit_weights: Internal flag for testing. When True, passes
            explicit ``weights=np.ones(N)`` to ``_build_grid_2d`` to verify
            IS estimator backward compatibility.

    References:
        Gray et al. (2020), arXiv:1908.06050, Section III.B-C.
        Laghi et al. (2021), arXiv:2102.01708, Section III.A.
        SNR ~ 1/d_L: Hogg (1999), arXiv:astro-ph/9905116, Eq. (16).
    """

    def __init__(
        self,
        injection_data_dir: str,
        snr_threshold: float,
        h_grid: list[float] | None = None,
        *,
        dl_bins: int = _DEFAULT_DL_BINS,
        mass_bins: int = _DEFAULT_M_BINS,
        h_prior_range: tuple[float, float] = (_DEFAULT_H_PRIOR_MIN, _DEFAULT_H_PRIOR_MAX),
        bandwidth_scale: float = 1.0,
        estimator: _Estimator = _DEFAULT_ESTIMATOR,
        _force_unit_weights: bool = False,
    ) -> None:
        self._dl_bins = dl_bins
        self._mass_bins = mass_bins
        self._snr_threshold = snr_threshold
        self._force_unit_weights = _force_unit_weights
        # F4-v2: "local_linear" (default) corrects the d_L→0 boundary bias of
        # the local-constant "nadaraya_watson" form.  The NW form is retained
        # as an opt-in for regression testing / reproducing pre-F4-v2 values.
        if estimator not in ("local_linear", "nadaraya_watson"):
            msg = f"estimator must be 'local_linear' or 'nadaraya_watson', got {estimator!r}"
            raise ValueError(msg)
        self._estimator: _Estimator = estimator
        # F4 (Phase 49): kernel-bandwidth tuning multiplier applied to Scott's
        # rule.  1.0 = standard Scott's rule (theoretically optimal under
        # Gaussian assumption).  Larger values oversmooth (reduce variance,
        # increase bias); smaller values undersmooth (increase variance,
        # restore step-like behavior).  See ``_compute_bandwidths``.
        if bandwidth_scale <= 0.0:
            msg = f"bandwidth_scale must be positive, got {bandwidth_scale}"
            raise ValueError(msg)
        self._bandwidth_scale: float = float(bandwidth_scale)
        if h_prior_range[0] >= h_prior_range[1]:
            msg = f"h_prior_range must satisfy lower < upper, got {h_prior_range}"
            raise ValueError(msg)
        self._h_prior_min: float = float(h_prior_range[0])
        self._h_prior_max: float = float(h_prior_range[1])

        if h_grid is not None:
            warnings.warn(
                "The 'h_grid' parameter is deprecated and ignored. "
                "SimulationDetectionProbability now builds P_det grids on-the-fly "
                "via SNR rescaling from pooled injection data.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Glob CSV files matching expected patterns
        patterns = [
            f"{injection_data_dir}/injection_h_*_task_*.csv",
            f"{injection_data_dir}/injection_h_*.csv",
        ]
        csv_files: list[str] = []
        for pattern in patterns:
            csv_files.extend(glob.glob(pattern))

        # Remove duplicates (a file may match both patterns)
        csv_files = sorted(set(csv_files))

        if not csv_files:
            msg = (
                f"No injection CSV files found in '{injection_data_dir}'. "
                "Expected files matching 'injection_h_*_task_*.csv' or 'injection_h_*.csv'."
            )
            raise FileNotFoundError(msg)

        # Extract h values from filenames for reference
        h_pattern = re.compile(r"injection_h_(\d+p\d+)")
        h_values_found: set[float] = set()

        # Load ALL CSVs and pool into a single DataFrame
        dfs: list[pd.DataFrame] = []
        for f in csv_files:
            match = h_pattern.search(f)
            if match:
                h_label = match.group(1)
                h_val = float(h_label.replace("p", "."))
                h_values_found.add(h_val)
            dfs.append(pd.read_csv(f))

        if not dfs:
            msg = (
                f"Could not parse any injection CSV files in '{injection_data_dir}'. "
                "Expected format: 'injection_h_0p70_task_001.csv'."
            )
            raise FileNotFoundError(msg)

        self._pooled_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        self._h_values_found: list[float] = sorted(h_values_found)

        logger.info(
            "Pooled %d injection events from %d files (h values: %s).",
            len(self._pooled_df),
            len(csv_files),
            ", ".join(f"{h:.2f}" for h in self._h_values_found),
        )

        # Validate required columns
        required_cols = {"z", "M", "SNR", "h_inj", "luminosity_distance"}
        missing = required_cols - set(self._pooled_df.columns)
        if missing:
            msg = f"Injection CSV missing required columns: {missing}"
            raise ValueError(msg)

        # Pre-extract arrays for efficient rescaling
        self._z_arr: npt.NDArray[np.float64] = self._pooled_df["z"].values.astype(np.float64)
        self._M_arr: npt.NDArray[np.float64] = self._pooled_df["M"].values.astype(np.float64)
        self._snr_raw: npt.NDArray[np.float64] = self._pooled_df["SNR"].values.astype(np.float64)
        self._h_inj_arr: npt.NDArray[np.float64] = self._pooled_df["h_inj"].values.astype(
            np.float64
        )
        self._dl_raw: npt.NDArray[np.float64] = self._pooled_df[
            "luminosity_distance"
        ].values.astype(np.float64)

        # LRU cache for built grids: h_value -> (2D interpolator, 1D interpolator)
        self._grid_cache: OrderedDict[
            float,
            tuple[RegularGridInterpolator, RegularGridInterpolator],
        ] = OrderedDict()

        # Quality flags cache
        self._quality_flags: dict[
            float, dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]
        ] = {}

        # h-stable bin support for p_det histogram estimators, computed
        # once over the prior support [h_min, h_max].  See
        # ``_compute_dl_global_max`` for the rationale and references.
        self._dl_global_max: float = self._compute_dl_global_max()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude heavy data from pickle that workers don't need.

        Workers only call detection_probability_*_interpolated() which uses the
        pre-built RegularGridInterpolator from _grid_cache.  The raw injection
        arrays (_z_arr, etc.) are only needed to build new grids via _rescale_snr,
        which never happens in workers because the cache is pre-warmed for the
        target h before pool spawn.
        """
        state = self.__dict__.copy()
        state["_pooled_df"] = None
        # Raw injection arrays (~18.5 MB) — not needed when grid is pre-warmed
        state["_z_arr"] = None
        state["_M_arr"] = None
        state["_snr_raw"] = None
        state["_h_inj_arr"] = None
        state["_dl_raw"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _rescale_snr(
        self, h_target: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Rescale SNR values from injection h to target h.

        For each event at redshift z injected at h_inj with SNR_raw:
            d_L_inj = dist(z, h_inj)   [from injection campaign]
            d_L_target = dist(z, h_target)
            SNR_target = SNR_raw * d_L_inj / d_L_target

        The d_L_inj values are recomputed from (z, h_inj) rather than using
        the stored luminosity_distance column, ensuring consistency with the
        cosmological model in physical_relations.py.

        Args:
            h_target: Target Hubble parameter value.

        Returns:
            Tuple of (d_L_target, SNR_rescaled) arrays, each shape (N,).

        References:
            SNR ~ 1/d_L: gravitational wave amplitude h(t) ~ 1/d_L.
            Gray et al. (2020), arXiv:1908.06050, Section III.B-C.
        """
        # Compute d_L at injection h for each event
        # Group by unique h_inj values for efficiency
        unique_h_inj = np.unique(self._h_inj_arr)
        d_L_inj = np.empty_like(self._z_arr)
        for h_inj in unique_h_inj:
            mask = self._h_inj_arr == h_inj
            d_L_inj[mask] = dist_vectorized(self._z_arr[mask], h=float(h_inj))

        # Compute d_L at target h for all events
        d_L_target = dist_vectorized(self._z_arr, h=h_target)

        # Rescale SNR: SNR(h_target) = SNR_raw * d_L(z, h_inj) / d_L(z, h_target)
        # Guard against d_L_target = 0 (z = 0 edge case)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_rescaled = np.where(
                d_L_target > 0,
                self._snr_raw * d_L_inj / d_L_target,
                0.0,
            )

        return (
            np.asarray(d_L_target, dtype=np.float64),
            np.asarray(snr_rescaled, dtype=np.float64),
        )

    def _compute_dl_global_max(self) -> float:
        """Maximum luminosity distance over the prior support of h-trials.

        Returns a single ``d_L`` scalar that bounds the histogram support
        used by ``_build_grid_2d`` and ``_build_grid_1d``.  The same
        ``dl_edges`` array is then reused at every h-trial so the
        histogram support does not drift as h changes — a necessary
        condition for converged hierarchical-Bayes inference per
        Farr (2019) and the fixed-injection / per-Lambda-reweighting
        paradigm of Mandel-Farr-Gair (2019).

        d_L is monotone-decreasing in h at fixed z (FLRW cosmology in
        the LamCDM scenario), so the max d_L over h is attained at the
        lower prior bound h = h_min.  A single evaluation at h_min on
        the catalog redshifts ``self._z_arr`` is therefore sufficient.

        Returns:
            float: ``max_i (d_L(z_i, h_min)) * _DL_PADDING_FACTOR``,
            with units of Gpc and a 10% headroom factor preserved from
            the original per-h construction.

        References:
            Farr (2019), arXiv:1904.10879, Sec III (accuracy
            requirements for empirically-measured selection functions).
            Mandel, Farr & Gair (2019), arXiv:1809.02063, Eq. 18.
            Gray et al. (2020), arXiv:1908.06050, gwcosmo MDC.
            See ``.planning/debug/F1_literature_audit.md`` for the
            project's literature audit and
            ``.planning/debug/posterior-noisy-peak.md`` for the
            pre-fix mechanism.
        """
        d_L_at_h_min = dist_vectorized(self._z_arr, h=self._h_prior_min)
        return float(np.max(d_L_at_h_min)) * _DL_PADDING_FACTOR

    def _compute_bandwidths(
        self,
        dl_vals: npt.NDArray[np.float64],
        log_M_vals: npt.NDArray[np.float64],
    ) -> tuple[float, float]:
        """Kernel bandwidths for the Nadaraya-Watson p_det estimator.

        Scott's rule for d=2: σ = bandwidth_scale · n^(-1/6) · std(x).
        d_L axis is linear (Gpc); M axis is log10(M_z).

        Args:
            dl_vals: per-injection d_L_target(h) at the current trial h, in Gpc.
            log_M_vals: per-injection log10(M_z) (observer-frame), dimensionless.

        Returns:
            (σ_dl in Gpc, σ_logM in dex).  Both ≥ a small numerical floor.

        References:
            Scott (1992), Multivariate Density Estimation, Ch. 6.
        """
        n = float(len(dl_vals))
        scale = self._bandwidth_scale * n**_SCOTT_EXPONENT_2D
        sigma_dl = max(scale * float(np.std(dl_vals, ddof=0)), 1e-12)
        sigma_log_M = max(scale * float(np.std(log_M_vals, ddof=0)), 1e-12)
        return sigma_dl, sigma_log_M

    def _get_or_build_grid(
        self, h: float
    ) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
        """Get cached grid or build a new one for the given h value.

        Uses LRU eviction when cache exceeds _MAX_CACHE_SIZE entries.

        Args:
            h: Hubble parameter value.

        Returns:
            Tuple of (2D interpolator, 1D interpolator).
        """
        if h in self._grid_cache:
            # Move to end (most recently used)
            self._grid_cache.move_to_end(h)
            return self._grid_cache[h]

        # Cache miss: build grids via SNR rescaling
        d_L_target, snr_rescaled = self._rescale_snr(h)

        # Build a temporary DataFrame for the grid builders.
        # The 2D grid M-axis is observer-frame M_z = M_source · (1 + z_inj),
        # the natural SNR-determining mass coordinate.  Production queries
        # (numerator: ``host_M·(1+z)``; denominator: ``M·(1+z)``) all pass
        # observer-frame M_z, so the grid axis matches the query coordinate.
        # See ``docs/H0_BIAS_RESOLUTION.md`` §3.15 (H3 fix).
        # Ref: Maggiore (2008) Vol 1 §4.1.4 (M_z = M_source · (1+z));
        # Mandel, Farr & Gair (2019) arXiv:1809.02063 §2 (selection
        # function evaluated at hypothesis observer-frame parameters).
        df_rescaled = pd.DataFrame(
            {
                "luminosity_distance": d_L_target,
                "M": self._M_arr * (1.0 + self._z_arr),
                "SNR": snr_rescaled,
            }
        )

        weights = np.ones(len(df_rescaled)) if self._force_unit_weights else None

        interp_2d = self._build_grid_2d(
            df_rescaled,
            self._snr_threshold,
            h_val=h,
            weights=weights,
        )
        interp_1d = self._build_grid_1d(df_rescaled, self._snr_threshold, h_val=h)

        # LRU eviction
        if len(self._grid_cache) >= _MAX_CACHE_SIZE:
            self._grid_cache.popitem(last=False)  # Remove oldest

        self._grid_cache[h] = (interp_2d, interp_1d)
        return interp_2d, interp_1d

    def _build_grid_2d(
        self,
        df: pd.DataFrame,
        snr_threshold: float,
        h_val: float | None = None,
        *,
        weights: npt.NDArray[np.float64] | None = None,
    ) -> RegularGridInterpolator:
        """Build a 2D P_det(d_L, M_z) grid via the Nadaraya-Watson kernel estimator.

        Replaces the prior histogram form ``p_det = N_det / N_total`` (which is
        a step function in (d_L, M_z, h) and produces coherent integer-count
        jumps as injection d_L_k(h) crosses fixed bin edges) with the smooth
        kernel-weighted form

            p̂(d_L_q, M_q, h) = Σ_k w_k · K_dl(d_L_k(h), d_L_q) · K_M(M_k, M_q)
                                · 1[SNR_k(h)≥thr]
                              / Σ_k w_k · K_dl(d_L_k(h), d_L_q) · K_M(M_k, M_q)

        evaluated at every fixed grid center.  Kernels are Gaussian:
        K_dl(x; x_q) = exp[-½((x − x_q)/σ_dl)²]; K_M uses log10(M).  Bandwidths
        from Scott's rule (``_compute_bandwidths``).  Kernel is truncated at
        ``_KERNEL_TRUNCATION_SIGMA`` along the d_L axis (sparse search via
        sorted-injection ``np.searchsorted``) to keep build cost O(N_inj).

        The M axis of the grid is **observer-frame** M_z = M_source · (1 + z_inj),
        the natural SNR-determining mass coordinate.  ``_get_or_build_grid``
        applies the (1+z_inj) factor when constructing ``df`` so that the M
        column passed here is already observer-frame.  Production queries pass
        observer-frame M_z directly; grid axis and queries share one convention.

        When ``weights`` is provided, kernel weights compose multiplicatively
        with the importance-sampling weights (self-normalized estimator):
        ``p̂ = Σ_k w_IS K_k det_k / Σ_k w_IS K_k``.  When ``weights`` is None
        (default), ``w_IS = 1`` for all k.

        If ``h_val`` is provided, per-cell quality metadata is stored in
        ``self._quality_flags`` for diagnostic use.  Under the kernel form the
        previously-integer counts become continuous "effective" kernel mass;
        the dict keys are preserved (``n_total``, ``n_detected``, ``reliable``,
        ``n_eff``, ``dl_edges``, ``M_edges``) for backward compatibility.

        Args:
            df: DataFrame with columns luminosity_distance, M, SNR.  ``M`` is
                observer-frame M_z (per the convention set in
                ``_get_or_build_grid``).
            snr_threshold: SNR detection threshold.
            h_val: Hubble parameter value for quality flag storage (optional).
            weights: Per-injection importance weights, shape (N,).  If None,
                all weights are implicitly 1.

        Returns:
            RegularGridInterpolator for P_det(d_L, M_z) on the fixed centers.

        References:
            Nadaraya (1964), Watson (1964) — kernel regression estimator.
            Scott (1992), Multivariate Density Estimation, Ch. 6 — bandwidth.
            Farr (2019), arXiv:1904.10879 Sec III — per-injection p_det form.
            Mandel-Farr-Gair (2019), arXiv:1809.02063 Eq. 18 — IS weight form.
            Tiwari (2018), arXiv:1712.00482, Eq. 5-8 — self-normalized IS.
            Kish (1965), Survey Sampling — effective sample size formula.
        """
        # Eq. (A.19) in Gray et al. (2020), arXiv:1908.06050, replaced with
        # the kernel-weighted Nadaraya-Watson form (see method docstring).
        dl_vals = np.asarray(df["luminosity_distance"].values, dtype=np.float64)
        M_vals = np.asarray(df["M"].values, dtype=np.float64)  # noqa: N806
        snr_vals = np.asarray(df["SNR"].values, dtype=np.float64)
        n_events = len(df)

        if weights is not None:
            w_is = np.asarray(weights, dtype=np.float64)
            if len(w_is) != n_events:
                msg = f"weights length {len(w_is)} != DataFrame length {n_events}"
                raise ValueError(msg)
        else:
            w_is = np.ones(n_events, dtype=np.float64)

        # Fixed grid support (h-stable d_L; h-independent observer-frame M_z).
        # The grid CENTERS define where p_det is evaluated; the edges are kept
        # only for backward-compatible quality_flags reporting.
        dl_max = self._dl_global_max
        dl_edges = np.linspace(0.0, dl_max, self._dl_bins + 1)
        M_min = float(np.min(M_vals)) * 0.9  # noqa: N806
        M_max = float(np.max(M_vals)) * 1.1  # noqa: N806
        M_edges = np.geomspace(M_min, M_max, self._mass_bins + 1)  # noqa: N806

        dl_centers = 0.5 * (dl_edges[:-1] + dl_edges[1:])
        M_centers = np.sqrt(M_edges[:-1] * M_edges[1:])  # noqa: N806

        log_M_vals = np.log10(M_vals)  # noqa: N806
        log_M_centers = np.log10(M_centers)  # noqa: N806

        sigma_dl, sigma_log_M = self._compute_bandwidths(dl_vals, log_M_vals)

        # Sort by d_L for searchsorted-based 3σ truncation.
        sort_idx = np.argsort(dl_vals, kind="mergesort")
        dl_sorted = dl_vals[sort_idx]
        log_M_sorted = log_M_vals[sort_idx]  # noqa: N806
        w_sorted = w_is[sort_idx]
        det_sorted = snr_vals[sort_idx] >= snr_threshold

        p_det_grid = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)
        n_total_grid = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)
        n_det_grid = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)
        n_eff_grid = np.zeros((self._dl_bins, self._mass_bins), dtype=np.float64)

        half_window = _KERNEL_TRUNCATION_SIGMA * sigma_dl
        inv_two_sigma_dl_sq = 0.5 / (sigma_dl * sigma_dl)
        inv_two_sigma_log_M_sq = 0.5 / (sigma_log_M * sigma_log_M)

        for i, dl_q in enumerate(dl_centers):
            lo = int(np.searchsorted(dl_sorted, dl_q - half_window, side="left"))
            hi = int(np.searchsorted(dl_sorted, dl_q + half_window, side="right"))
            if lo == hi:
                continue
            dl_loc = dl_sorted[lo:hi]
            log_M_loc = log_M_sorted[lo:hi]  # noqa: N806
            w_loc = w_sorted[lo:hi]
            det_loc = det_sorted[lo:hi]

            # d_L kernel weight (independent of M_q); fold in IS weight.
            w_dl = np.exp(-inv_two_sigma_dl_sq * (dl_loc - dl_q) ** 2) * w_loc

            # log-M kernel for all M centers at once: shape (N_loc, M_bins)
            diff_log_M = log_M_loc[:, None] - log_M_centers[None, :]  # noqa: N806
            w_log_M = np.exp(-inv_two_sigma_log_M_sq * diff_log_M**2)  # noqa: N806

            kernel_w = w_dl[:, None] * w_log_M  # shape (N_loc, M_bins)

            sum_w = kernel_w.sum(axis=0)
            sum_w_det = (kernel_w * det_loc[:, None]).sum(axis=0)
            sum_w_sq = (kernel_w**2).sum(axis=0)

            if self._estimator == "local_linear":
                # Local-linear in d_L (per M-center), local-constant in log M.
                # Eq. (3.5-3.6) in Fan & Gijbels (1996), Local Polynomial Modelling.
                u_dl = dl_loc - dl_q  # [Gpc]
                det_col = np.broadcast_to(det_loc[:, None], kernel_w.shape).astype(np.float64)
                p_det_grid[i, :] = _local_linear_p_det(u_dl, kernel_w, det_col)
            else:
                # Local-constant (Nadaraya-Watson): p̂ = Σ wₖ yₖ / Σ wₖ.
                p_det_grid[i, :] = np.divide(
                    sum_w_det,
                    sum_w,
                    out=np.zeros(self._mass_bins, dtype=np.float64),
                    where=sum_w > 0.0,
                )
            n_total_grid[i, :] = sum_w
            n_det_grid[i, :] = sum_w_det
            # Kish (1965) effective sample size: (Σw)² / Σw²
            n_eff_grid[i, :] = np.divide(
                sum_w**2,
                sum_w_sq,
                out=np.zeros(self._mass_bins, dtype=np.float64),
                where=sum_w_sq > 0.0,
            )

        # Store quality flags (metadata only -- does not affect interpolation).
        # Under the kernel form n_total / n_detected are continuous kernel-mass
        # sums (not integer counts); the keys are preserved for backward
        # compatibility with diagnostic scripts (test_14_channel_audit.py etc.).
        if h_val is not None:
            self._quality_flags[h_val] = {
                "n_total": n_total_grid.copy(),
                "n_detected": n_det_grid.copy(),
                "reliable": (n_eff_grid >= 10.0),
                "dl_edges": dl_edges.copy(),
                "M_edges": M_edges.copy(),
                "n_eff": n_eff_grid.copy(),
            }

        # fill_value=None → nearest-neighbor extrapolation outside grid.
        # fill_value=0.0 caused 44% of events to lose completeness correction
        # because high-SNR events' 4σ integration bounds exceed the injection grid.
        return RegularGridInterpolator(
            (dl_centers, M_centers),
            p_det_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _build_grid_1d(
        self,
        df: pd.DataFrame,
        snr_threshold: float,
        *,
        h_val: float | None = None,
    ) -> RegularGridInterpolator:
        """Build a 1D P_det(d_L) grid via the 1D Nadaraya-Watson kernel estimator.

        Replaces the prior 1D histogram form ``p_det = N_det / N_total`` with
        the smooth kernel-weighted estimator (see ``_build_grid_2d`` for the
        full motivation and references).  Uses the same d_L bandwidth as the
        2D builder (Scott's rule on the injection sample) to keep the two
        channels consistent.

        Off-grid extrapolation is the responsibility of
        :meth:`detection_probability_without_bh_mass_interpolated_zero_fill`
        (principled linear bridge to p=1 at d_L=0, slope-matched extrapolation
        above d_L_max), unchanged.

        Args:
            df: DataFrame with columns luminosity_distance, M, SNR.  ``M`` is
                used only to compute the (h-independent) M-axis bandwidth that
                the 2D builder shares; the 1D estimator marginalizes M.
            snr_threshold: SNR detection threshold.
            h_val: Hubble parameter value (used for diagnostic logging only).

        Returns:
            RegularGridInterpolator for P_det(d_L) on the fixed grid centers
            (length ``dl_bins``).
        """
        dl_vals = np.asarray(df["luminosity_distance"].values, dtype=np.float64)
        M_vals = np.asarray(df["M"].values, dtype=np.float64)  # noqa: N806
        snr_vals = np.asarray(df["SNR"].values, dtype=np.float64)

        # Fixed grid support — h-stable; see _build_grid_2d docstring for the
        # rationale (Farr 2019; Mandel-Farr-Gair 2019; Gray et al. 2020).
        dl_max = self._dl_global_max
        dl_edges = np.linspace(0.0, dl_max, self._dl_bins + 1)
        dl_centers = 0.5 * (dl_edges[:-1] + dl_edges[1:])

        # Reuse the 2D d_L bandwidth (Scott's rule on the full injection
        # sample).  The M argument is required by ``_compute_bandwidths`` but
        # is irrelevant for the 1D estimator's d_L kernel.
        sigma_dl, _ = self._compute_bandwidths(dl_vals, np.log10(M_vals))

        sort_idx = np.argsort(dl_vals, kind="mergesort")
        dl_sorted = dl_vals[sort_idx]
        det_sorted = snr_vals[sort_idx] >= snr_threshold

        half_window = _KERNEL_TRUNCATION_SIGMA * sigma_dl
        inv_two_sigma_dl_sq = 0.5 / (sigma_dl * sigma_dl)

        p_det_1d = np.zeros(self._dl_bins, dtype=np.float64)
        n_eff_first_bin = 0.0
        for i, dl_q in enumerate(dl_centers):
            lo = int(np.searchsorted(dl_sorted, dl_q - half_window, side="left"))
            hi = int(np.searchsorted(dl_sorted, dl_q + half_window, side="right"))
            if lo == hi:
                continue
            dl_loc = dl_sorted[lo:hi]
            det_loc = det_sorted[lo:hi]
            w_k = np.exp(-inv_two_sigma_dl_sq * (dl_loc - dl_q) ** 2)
            sum_w = float(w_k.sum())
            if sum_w > 0.0:
                if self._estimator == "local_linear":
                    # Local-linear in d_L — Eq. (3.5-3.6) in Fan & Gijbels (1996).
                    # Corrects the d_L→0 boundary bias of the NW form below.
                    u_dl = dl_loc - dl_q  # [Gpc]
                    p_det_1d[i] = float(_local_linear_p_det(u_dl, w_k, det_loc.astype(np.float64)))
                else:
                    # Local-constant (Nadaraya-Watson): p̂ = Σ wₖ yₖ / Σ wₖ.
                    p_det_1d[i] = float((w_k * det_loc).sum()) / sum_w
                if i == 0:
                    sum_w_sq = float((w_k**2).sum())
                    n_eff_first_bin = (sum_w * sum_w) / sum_w_sq if sum_w_sq > 0 else 0.0

        # Reliability check: replaces the pre-F4 ``total_counts[0] < 100`` check
        # with the Kish effective sample size at the first grid center.  At
        # n_eff ≈ 100 the Wilson 95% CI half-width on p̂(c_0) is ≈ 1/sqrt(n_eff)
        # ≈ 0.10, so warn below 100 effective injections (matching the prior
        # threshold's spirit).
        if n_eff_first_bin < 100.0:
            logger.warning(
                "P_det 1D grid first center d_L=%.4f Gpc has Kish N_eff=%.1f "
                "(h=%s).  p̂(c_0) may be noisy.  Consider denser low-d_L "
                "injections.",
                float(dl_centers[0]),
                n_eff_first_bin,
                f"{h_val:.4f}" if h_val is not None else "?",
            )

        # Eq. (A.19) in Gray et al. (2020), arXiv:1908.06050.
        # No anchors are prepended: out-of-grid extrapolation is the
        # responsibility of
        # :meth:`detection_probability_without_bh_mass_interpolated_zero_fill`,
        # which applies a principled scheme (linear bridge to the saturated
        # asymptote at d_L=0; slope-matched linear extrapolation toward 0
        # above d_L_max).  Replaced the Plan 45-02/04 anchor scheme on
        # 2026-05-05 — see ``.planning/2D-CHANNEL-AUDIT-20260505.md``.
        return RegularGridInterpolator(
            (dl_centers,),
            p_det_1d,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def quality_flags(self, h: float) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]:
        """Return per-bin quality metadata for the given h value.

        Quality flags are diagnostic metadata stored during grid construction.
        They do **not** affect the P_det interpolation result.

        If the grid for this h has not been built yet, it will be built
        (triggering SNR rescaling and caching).

        The returned dict contains (F4 kernel-form semantics):

        - ``n_total``: float array (dl_bins, M_bins) -- kernel-weighted mass
          (Σ_k K_k · w_IS_k) per cell.  Continuous "effective total"; replaces
          the pre-F4 integer histogram counts.
        - ``n_detected``: float array (dl_bins, M_bins) -- kernel-weighted
          detected mass (Σ_k K_k · w_IS_k · 1[SNR_k≥thr]) per cell.
        - ``reliable``: bool array (dl_bins, M_bins) -- True where n_eff >= 10
          (Kish effective sample size threshold).
        - ``dl_edges``: float array (dl_bins+1,) -- d_L grid support endpoints
          in Gpc (kernel evaluated at centers; edges retained for back-compat).
        - ``M_edges``: float array (M_bins+1,) -- M grid support endpoints.
        - ``n_eff``: float array (dl_bins, M_bins) -- Kish effective sample
          size per cell, (Σ w_k_eff)² / Σ (w_k_eff)² with w_k_eff = K_k · w_IS_k.

        Args:
            h: Hubble parameter value.

        Returns:
            Dict of quality flag arrays.

        Raises:
            ValueError: If no quality flags are available (empty grid).
        """
        # Ensure grid is built (which populates quality flags)
        if h not in self._quality_flags:
            self._get_or_build_grid(h)

        if h not in self._quality_flags:
            msg = f"No quality flags for h={h:.4f} after grid construction."
            raise ValueError(msg)
        return self._quality_flags[h]

    def detection_probability_with_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        M_z: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
    ) -> float | npt.NDArray[np.float64]:
        """Detection probability including BH mass dependence.

        Sky angles (phi, theta) are accepted for API compatibility but are
        marginalized over internally (D-02).

        The grid is in (d_L, M_z) space — observer-frame M_z on both the
        grid axis and the query argument.  See ``_build_grid_2d`` for the
        coordinate convention; ``docs/H0_BIAS_RESOLUTION.md`` §3.15 for the
        history of the convention fix.

        **Out-of-grid policy: principled monotonic-asymptotic extrapolation.**

        For queries inside the (d_L, M) grid, returns scipy's bilinear
        interpolation (clipped to [0, 1]).  For queries outside the grid,
        applies a slope-matched linear extrapolation from the nearest face,
        clamped per the physical asymptote table:

        +------------------+--------------+--------------------------------+
        | Direction        | Asymptote    | Reason                         |
        +==================+==============+================================+
        | d_L > d_L_max    | 0            | SNR-suppressed (sources too    |
        |                  |              | distant for SNR ≥ threshold)   |
        +------------------+--------------+--------------------------------+
        | d_L < d_L_min    | 1            | SNR-saturated (very nearby     |
        |                  |              | sources detected with prob 1)  |
        +------------------+--------------+--------------------------------+
        | M_z > M_z_max    | 0            | EMRI rate / waveform model     |
        |                  |              | breakdown above the EMRI mass  |
        +------------------+--------------+--------------------------------+
        | M_z < M_z_min    | 0            | SNR-suppressed and/or rate     |
        |                  |              | cutoff at low M                |
        +------------------+--------------+--------------------------------+
        | corner (both     | min of the   | "Either axis suppresses" — the |
        | axes outside)    | two faces    | only saturating face is        |
        |                  |              | (d_L<min, M in grid), which is |
        |                  |              | NOT a corner                   |
        +------------------+--------------+--------------------------------+

        **Construction.**  Two cases.

        *Saturating face (d_L<d_L_min):* there is a unique natural scale
        d_L=0 where the asymptote p_det=1 is exact (no source closer than
        the observer).  Linearly interpolate from (dl_min, p_edge) to
        (0, 1) — C0 continuous at dl_min, reaches the asymptote at the
        natural physical boundary, and uses no fitted parameters or noisy
        boundary slope estimates.  Explicitly:
        ``p(dl) = 1 - (1 - p_edge) * (dl / dl_min)`` for ``dl ∈ [0, dl_min]``.

        *Suppressing faces (d_L>d_L_max, M_z>M_max, M_z<M_min):* no analogous
        finite asymptote scale exists (suppression is at infinity).  Use
        slope-matched linear extrapolation from the boundary, computed
        from the last two grid centers along the relevant axis with the
        slope evaluated at the projected position on the other axis.
        ``p_extrap = p_edge + slope · (query - edge)``, clamped to
        ``[0, p_edge]`` (Option A: never overshoots boundary value, floor
        at the asymptote 0).

        Corner cells (both axes outside) take the ``min`` of the two face
        extrapolations.  The only saturating face is (d_L<d_L_min,
        M in grid), which is not a corner; all true corners involve at
        least one suppressing axis, so ``min`` is monotone-correct.

        **Properties.**

        * **C0 continuous at the boundary** by construction (extrapolation
          formula evaluates to p_edge at the boundary, matching the in-grid
          interpolation).  Removes the discontinuity that previously
          drove spurious h_trial-dependence in the joint posterior as
          hosts crossed the grid boundary at small Δh.
        * **Bounded in [0, 1]** by construction (clamp + asymptote table).
          The previous policy (raw scipy linear extrapolation + clip) could
          drift to negative values at the d_L→0 boundary due to KDE noise
          in the low-d_L bins (only ~7 injections per bin in production),
          systematically returning ≈0 instead of ≈1 for 6–12% of events.
          See ``.planning/2D-CHANNEL-AUDIT-20260505.md`` Step 1b for the
          quantitative diagnostic.
        * **Slope from the simulated KDE** (no fitted anchor): the
          boundary slope is the same one scipy would use for its own
          linear extrapolation; the difference from the prior policy is
          the directional clamp (Option A) that prevents wrong-direction
          KDE noise from running into the wrong asymptote.

        Args:
            d_L: Luminosity distance in Gpc.
            M_z: Observer-frame (redshifted) BH mass in solar masses.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter.

        Returns:
            Detection probability in [0, 1].

        References:
            Gray et al. (2020), arXiv:1908.06050, Eq. (8).
            Laghi et al. (2021), arXiv:2102.01708, Section III.A.
            Maggiore (2008), Gravitational Waves Vol 1 §7.7 (SNR scaling
            for inspirals → monotonicity argument).
            Babak et al. (2017), arXiv:1703.09722 §III (EMRI rate density
            with high/low M cutoffs → asymptote table).
        """
        interp_2d, _ = self._get_or_build_grid(h)
        dl_centers = np.asarray(interp_2d.grid[0])
        M_centers = np.asarray(interp_2d.grid[1])  # noqa: N806
        p_grid = np.asarray(interp_2d.values, dtype=np.float64)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        M_arr = np.atleast_1d(np.asarray(M_z, dtype=np.float64))  # noqa: N806

        dl_min = float(dl_centers[0])
        dl_max = float(dl_centers[-1])
        M_min = float(M_centers[0])  # noqa: N806
        M_max = float(M_centers[-1])  # noqa: N806

        # Project queries onto the in-grid box, then evaluate the in-grid
        # interpolator at the projected point — this is p_edge for face/corner
        # cells, and the in-grid value for in-grid cells.
        dl_clamp = np.clip(dl_arr, dl_min, dl_max)
        M_clamp = np.clip(M_arr, M_min, M_max)  # noqa: N806
        p_edge = np.clip(interp_2d(np.column_stack([dl_clamp, M_clamp])), 0.0, 1.0)

        # Start from p_edge; overlay face extrapolations where applicable.
        # In-grid cells keep p_edge as their final value (no face triggers).
        result = p_edge.copy()

        # ---- d_L < d_L_min face (saturating direction; asymptote 1) ----
        # The d_L=0 limit is the unique natural physical scale where the
        # asymptote p_det=1 is exact (closest possible source).  Use a
        # linear bridge from (dl_min, p_edge) to (0, 1) — guaranteed to
        # be C0 continuous at dl_min (matches p_edge by construction) and
        # to reach the asymptote at dl=0.  This deliberately ignores the
        # boundary KDE slope, which is unreliable in the first d_L bin
        # (~7 injections in production; warned by ``_build_grid_1d``).
        # Using the KDE slope here would let counting noise drive the
        # extrapolation; the bridge is the same scheme as the boundary,
        # extended monotonically to the natural asymptote location.
        out_dl_low = dl_arr < dl_min
        if out_dl_low.any():
            idx = np.where(out_dl_low)[0]
            # Bridge: p(dl) = p_edge + (1 - p_edge) * (dl_min - dl) / dl_min
            #              = 1 - (1 - p_edge) * dl / dl_min
            # At dl=dl_min: p = p_edge (C0).  At dl=0: p = 1 (asymptote).
            p_bridge = 1.0 - (1.0 - p_edge[idx]) * (dl_arr[idx] / dl_min)
            # Clamp into [p_edge, 1] for queries with dl < 0 (defensive).
            result[idx] = np.clip(p_bridge, p_edge[idx], 1.0)

        # ---- d_L > d_L_max face (suppressing direction; asymptote 0) ----
        out_dl_high = dl_arr > dl_max
        if out_dl_high.any():
            idx = np.where(out_dl_high)[0]
            slope_row = (p_grid[-1, :] - p_grid[-2, :]) / (dl_centers[-1] - dl_centers[-2])
            slope_at_query = np.interp(M_clamp[idx], M_centers, slope_row)
            delta = dl_arr[idx] - dl_max  # positive
            p_extrap = p_edge[idx] + slope_at_query * delta
            # Option A: floor at 0, ceiling at p_edge (asymptote 0).
            result[idx] = np.clip(p_extrap, 0.0, p_edge[idx])

        # ---- M < M_min face (suppressing; asymptote 0) ----
        out_M_low = M_arr < M_min  # noqa: N806
        if out_M_low.any():
            idx = np.where(out_M_low)[0]
            slope_col = (p_grid[:, 0] - p_grid[:, 1]) / (M_centers[0] - M_centers[1])
            slope_at_query = np.interp(dl_clamp[idx], dl_centers, slope_col)
            delta = M_arr[idx] - M_min  # negative
            p_extrap = p_edge[idx] + slope_at_query * delta
            face_M_low = np.clip(p_extrap, 0.0, p_edge[idx])
            # Corner rule: min of the per-face values for cells outside on
            # both axes.
            result[idx] = np.minimum(result[idx], face_M_low)

        # ---- M > M_max face (suppressing; asymptote 0) ----
        out_M_high = M_arr > M_max  # noqa: N806
        if out_M_high.any():
            idx = np.where(out_M_high)[0]
            slope_col = (p_grid[:, -1] - p_grid[:, -2]) / (M_centers[-1] - M_centers[-2])
            slope_at_query = np.interp(dl_clamp[idx], dl_centers, slope_col)
            delta = M_arr[idx] - M_max  # positive
            p_extrap = p_edge[idx] + slope_at_query * delta
            face_M_high = np.clip(p_extrap, 0.0, p_edge[idx])
            result[idx] = np.minimum(result[idx], face_M_high)

        # Final safety clip (already enforced face-by-face but be defensive).
        result = np.clip(result, 0.0, 1.0)

        if np.ndim(d_L) == 0 and np.ndim(M_z) == 0:
            return float(result[0])
        return result  # type: ignore[no-any-return]

    def get_dl_max(self, h: float) -> float:
        """Return the maximum d_L of the 1D P_det grid for the given h value.

        This is ``max(injection_d_L) * 1.1``, i.e. the upper edge of the
        1D histogram used by :meth:`_build_grid_1d`.  Needed to compute
        ``z_max(h)`` for the full-volume denominator integral.

        Args:
            h: Dimensionless Hubble parameter.

        Returns:
            Maximum d_L in Gpc.
        """
        # Ensure grid is built (populates cache)
        self._get_or_build_grid(h)
        # Reconstruct dl_max from the 1D interpolator's grid points
        _, interp_1d = self._grid_cache[h]
        dl_centers = interp_1d.grid[0]
        # The centers are midpoints of linspace(0, dl_max, N+1).
        # spacing = dl_centers[1] - dl_centers[0], last center = dl_max - spacing/2
        spacing = float(dl_centers[1] - dl_centers[0])
        return float(dl_centers[-1] + spacing / 2)

    def validate_coverage(
        self,
        h: float,
        crb_df: pd.DataFrame,
    ) -> float:
        """Compute fraction of events whose 4-sigma d_L bounds fall within the P_det grid.

        For each event, compute d_L +/- 4*sigma_dL from the Cramer-Rao bounds.
        Check if both bounds fall within the grid's d_L range.

        Args:
            h: Hubble parameter value (to build/retrieve grid).
            crb_df: DataFrame with columns ``luminosity_distance`` and
                ``delta_luminosity_distance_delta_luminosity_distance`` (variance).

        Returns:
            Coverage fraction in [0, 1].
        """
        # Build/retrieve the grid to get d_L edge range
        self._get_or_build_grid(h)
        _, interp_1d = self._grid_cache[h]
        dl_centers = interp_1d.grid[0]
        spacing = float(dl_centers[1] - dl_centers[0])
        dl_grid_min = float(dl_centers[0] - spacing / 2)
        dl_grid_max = float(dl_centers[-1] + spacing / 2)

        # Extract per-event d_L and sigma_dL from CRB DataFrame
        d_L_vals = crb_df["luminosity_distance"].values.astype(np.float64)
        sigma_dL = np.sqrt(
            crb_df["delta_luminosity_distance_delta_luminosity_distance"].values.astype(np.float64)
        )

        # Compute 4-sigma bounds
        lower_bounds = d_L_vals - 4.0 * sigma_dL
        upper_bounds = d_L_vals + 4.0 * sigma_dL

        # Event is covered if both bounds fall within the grid range
        covered = (lower_bounds >= dl_grid_min) & (upper_bounds <= dl_grid_max)
        n_covered = int(np.sum(covered))
        n_total = len(d_L_vals)

        coverage_fraction = n_covered / n_total if n_total > 0 else 1.0

        logger.info(
            "P_det grid coverage: %.1f%% of events have 4-sigma d_L bounds within grid (%d/%d)",
            coverage_fraction * 100,
            n_covered,
            n_total,
        )
        if coverage_fraction < 0.95:
            logger.warning(
                "P_det grid coverage %.1f%% is below 95%% threshold. "
                "Consider increasing --pdet_dl_bins.",
                coverage_fraction * 100,
            )

        return coverage_fraction

    def detection_probability_without_bh_mass_interpolated_zero_fill(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
    ) -> float | npt.NDArray[np.float64]:
        """Detection probability with principled out-of-grid extrapolation.

        Mirror of :meth:`detection_probability_with_bh_mass_interpolated`
        (2D channel) specialized to the 1D (d_L only) grid.  Call sites in
        :mod:`bayesian_statistics`:

        * ``precompute_completion_denominator`` (D(h) full-volume denominator)
        * ``p_Di.completion_numerator_integrand`` (L_comp 4σ window)
        * ``single_host_likelihood.numerator_integrant_without_bh_mass`` (L_cat numerator)
        * ``single_host_likelihood.denominator_integrant_without_bh_mass`` (L_cat denominator)
        * ``single_host_likelihood_integration_testing`` (legacy, two integrands)

        **Out-of-grid policy: principled monotonic-asymptotic extrapolation.**

        +------------------+--------------+--------------------------------+
        | Direction        | Asymptote    | Reason                         |
        +==================+==============+================================+
        | d_L > d_L_max    | 0            | SNR-suppressed (sources too    |
        |                  |              | distant for SNR ≥ threshold)   |
        +------------------+--------------+--------------------------------+
        | d_L < d_L_min    | 1            | SNR-saturated (very nearby     |
        |                  |              | sources detected with prob 1)  |
        +------------------+--------------+--------------------------------+

        **Construction.**

        * **Saturating face (d_L < d_L_min):** linear bridge from
          ``(d_L_min, p_edge)`` to ``(0, 1)``.  The d_L=0 limit is the
          natural physical scale where the asymptote p_det=1 is exact (no
          source closer than the observer).  Explicitly:
          ``p(dl) = 1 - (1 - p_edge) * (dl / d_L_min)`` for
          ``dl ∈ [0, d_L_min]``.  C0 continuous at d_L_min by construction;
          reaches the asymptote 1 at d_L=0 by construction.  Uses no
          fitted constants and no boundary KDE slope (the first bin has
          ~7 injections in production; the bridge construction is
          insensitive to that noise).

        * **Suppressing face (d_L > d_L_max):** slope-matched linear
          extrapolation from the boundary, computed from the last two grid
          centers along d_L.  ``p_extrap = p_edge + slope · (query - edge)``,
          clamped to ``[0, p_edge]`` (Option A: never exceeds boundary,
          asymptotic floor at 0).

        **Why this replaces the Phase 45 anchor scheme (2026-05-05):**
        the previous scheme prepended two empirical anchors at d_L=0
        (Wilson 95% LB = 0.7931) and d_L=0.05 (empirical point estimate =
        1.0).  The Wilson LB was deliberately chosen to "not overshoot
        truth on production posteriors" — fitted to truth, against the
        project's principled-modeling preference.  The augmented Phase 46
        injection campaign now gives p̂(c_0) ≈ 1.0 at the first bin, so
        the Wilson anchor is actively *suppressing* the empirical 1.0 down
        to 0.7931 — the opposite of its original lift purpose.  The bridge
        construction here is the same scheme as the boundary
        (linear, C0) extended to the natural asymptote location; no fit,
        no anchor.  See ``.planning/2D-CHANNEL-AUDIT-20260505.md`` for the
        full rationale and the 2D-channel sibling that motivated this
        alignment.

        Args:
            d_L: Luminosity distance in Gpc.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter.

        Returns:
            Detection probability in [0, 1].

        References:
            Gray et al. (2020), arXiv:1908.06050, Eq. (A.19).
            Maggiore (2008), Gravitational Waves Vol 1 §7.7 (SNR scaling
            for inspirals → monotonicity argument).
        """
        _, interp_1d = self._get_or_build_grid(h)
        dl_centers = np.asarray(interp_1d.grid[0])
        p_grid = np.asarray(interp_1d.values, dtype=np.float64)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        dl_min = float(dl_centers[0])
        dl_max = float(dl_centers[-1])

        # Project to in-grid; evaluate at projected point → p_edge.
        dl_clamp = np.clip(dl_arr, dl_min, dl_max)
        p_edge = np.clip(interp_1d(dl_clamp.reshape(-1, 1)), 0.0, 1.0)
        result = p_edge.copy()

        # ---- d_L < d_L_min face (saturating; asymptote 1) ----
        # Linear bridge from (dl_min, p_edge) to (0, 1).  At dl=dl_min:
        # p = p_edge (C0); at dl=0: p = 1 (asymptote).
        out_low = dl_arr < dl_min
        if out_low.any():
            idx = np.where(out_low)[0]
            p_bridge = 1.0 - (1.0 - p_edge[idx]) * (dl_arr[idx] / dl_min)
            result[idx] = np.clip(p_bridge, p_edge[idx], 1.0)

        # ---- d_L > d_L_max face (suppressing; asymptote 0) ----
        # Slope-matched linear extrapolation, clamped to [0, p_edge].
        out_high = dl_arr > dl_max
        if out_high.any():
            idx = np.where(out_high)[0]
            slope = (p_grid[-1] - p_grid[-2]) / (dl_centers[-1] - dl_centers[-2])
            delta = dl_arr[idx] - dl_max  # positive
            p_extrap = p_edge[idx] + slope * delta
            result[idx] = np.clip(p_extrap, 0.0, p_edge[idx])

        # Final safety clip.
        result = np.clip(result, 0.0, 1.0)

        if np.ndim(d_L) == 0:
            return float(result[0])
        return result  # type: ignore[no-any-return]

    def detection_probability_without_bh_mass_interpolated(
        self,
        d_L: float | npt.NDArray[np.float64],
        phi: float | npt.NDArray[np.float64],
        theta: float | npt.NDArray[np.float64],
        *,
        h: float,
    ) -> float | npt.NDArray[np.float64]:
        """Detection probability marginalized over BH mass.

        Drop-in replacement for
        ``DetectionProbability.detection_probability_without_bh_mass_interpolated``
        with an additional ``h`` keyword for h-dependent P_det.

        Sky angles (phi, theta) are accepted for API compatibility but are
        marginalized over internally (D-02).

        Args:
            d_L: Luminosity distance in Gpc.
            phi: Sky angle phi (unused, marginalized over).
            theta: Sky angle theta (unused, marginalized over).
            h: Dimensionless Hubble parameter.

        Returns:
            Detection probability in [0, 1].

        References:
            Gray et al. (2020), arXiv:1908.06050, Eq. (8).
            Laghi et al. (2021), arXiv:2102.01708, Section III.A.
        """
        _, interp_1d = self._get_or_build_grid(h)

        dl_arr = np.atleast_1d(np.asarray(d_L, dtype=np.float64))
        points = dl_arr.reshape(-1, 1)

        result = np.clip(interp_1d(points), 0.0, 1.0)

        if np.ndim(d_L) == 0:
            return float(result[0])
        return result  # type: ignore[no-any-return]
