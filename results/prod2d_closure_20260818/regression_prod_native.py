"""Production-native per-event slope regression (runbook 21 Sec 3 item 1).

PREREGISTRATION_PROD_REGRESSION.md v2 -- implements Sec 2, 2b, 3, 5
VERBATIM. Zero new simulation; reads the banked
``results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/`` CSVs only.

Conventions inherited from ``tier0_bootstrap_jackknife.py`` (P7-2 pinned
formula): the CSV columns ``combined_no_bh``/``combined_with_bh`` are
per-event PLAIN LIKELIHOOD values; the production physics-floor
zero-handling is replicated per-row (over the full 41-node grid) before
taking logs, even though it is a verified no-op for this CSV.

Per-event two-point SECANT slopes of ln(floored L) are taken on the
registered bracketing node pairs (P4): iiib (0.780, 0.785), joint_r1
(0.790, 0.800) -- the pairs bracketing each venue's T0 full-sample 2D
posterior mean h*. s_e^2D from ``combined_with_bh``, s_e^1D from
``combined_no_bh``; response Delta s_e = s_e^2D - s_e^1D.

Covariates (Sec 2 P1 replacements) are recomputed at the SAME bracketing
nodes: cat_e, g_e, c_e' (Sec 2b exact-residual identity), in_catalog,
log10(SNR), sigma_dL/dL, z_e (dist_to_redshift at the run's fiducial H,
main.py:1480 pattern -- a covariate label, not an inference input, P9),
m_e = |ln(M/median(M))| plus the signed form (P10, reported only).

Registered statistics (Sec 3): S1 Spearman(Delta s, c_e'), S1a
point-biserial(Delta s, cat_e), S1b Spearman(Delta s, g_e), S2
Mann-Whitney U + rank-biserial r_rb (orientation fixed so POSITIVE r_rb
means Delta s stochastically larger in the NON-catalogue-supported group
-- the M-B direction), S3 Spearman(Delta s, m_e), S4 controls (reported,
non-adjudicating) + an HC3 OLS if statsmodels is available.

Gates G-a/G-b/G-c (Sec 3, P2): scored, STOP on G-a/G-b failure (no
statistics computed); G-c is reported (implied, not gating).

Bootstrap B = 10,000, seed 20280612, numpy default_rng, resample events
with replacement; h*/node pairs are FIXED inside every resample (P6).

Sensitivity (Sec 3, reported, non-band-bearing): point estimates of S1,
S2's r_rb, S3 recomputed on the one-step-left / one-step-right node
pairs found on the actual grid (iiib: (0.775,0.780)/(0.785,0.790);
joint_r1: (0.785,0.790)/(0.800,0.810)).

Usage:
    uv run python regression_prod_native.py [--output regression_prod_native_output.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import mannwhitneyu, pointbiserialr, spearmanr

from darksiren_emri.constants import H as TRUTH_H
from darksiren_emri.physical_relations import dist_to_redshift

try:
    import statsmodels.api as sm

    _HAVE_STATSMODELS = True
except ImportError:
    _HAVE_STATSMODELS = False

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]

VENUES = ("iiib", "joint_r1")

# Registered bracketing node pairs (P4), pinned verbatim.
NODE_PAIRS: dict[str, tuple[float, float]] = {
    "iiib": (0.780, 0.785),
    "joint_r1": (0.790, 0.800),
}

# Registered sensitivity node pairs (Sec 3, reported), the actual
# neighboring grid nodes verified against the 41-node grid (see the
# printed grid at runtime).
SENSITIVITY_PAIRS: dict[str, dict[str, tuple[float, float]]] = {
    "iiib": {"left": (0.775, 0.780), "right": (0.785, 0.790)},
    "joint_r1": {"left": (0.785, 0.790), "right": (0.800, 0.810)},
}

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20280612

# Sec 2b registered validity gate on c_e'.
CPRIME_RANGE = (-0.01, 1.01)
CPRIME_STOP_FRAC = 0.01

# Sec 3 aggregate construction gates.
GATE_A_TOL = 0.05


# ---------------------------------------------------------------------------
# Data loading (adapted from tier0_bootstrap_jackknife.py's conventions)
# ---------------------------------------------------------------------------


def _physics_floor_apply(
    likelihoods: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Replicate ``posterior_combination._physics_floor`` per-row, verbatim from T0.

    Per event row: zeros -> the row's own minimum nonzero value; an
    all-zero row has no nonzero value to floor from and is marked for
    exclusion instead. Returns ``(floored, exclude_mask)``.
    """
    result = likelihoods.copy()
    n_events = result.shape[0]
    exclude_mask = np.zeros(n_events, dtype=bool)
    for i in range(n_events):
        row = result[i]
        zero_mask = row == 0.0
        if not zero_mask.any():
            continue
        nonzero = row[~zero_mask]
        if nonzero.size == 0:
            exclude_mask[i] = True
        else:
            result[i, zero_mask] = float(nonzero.min())
    return result, exclude_mask


def _event_likelihoods_path(venue: str) -> Path:
    return REPO_ROOT / "results" / "run_20260804_postfix" / venue / "diagnostics" / "event_likelihoods.csv"


def _crb_path(venue: str) -> Path:
    return REPO_ROOT / "results" / "run_20260804_postfix" / venue / "diagnostics" / "prepared_cramer_rao_bounds.csv"


def _load_venue_frame(venue: str) -> pd.DataFrame:
    """Load and cache the full event_likelihoods CSV for a venue."""
    return pd.read_csv(_event_likelihoods_path(venue))


def _floored_logL_at_nodes(
    df: pd.DataFrame, channel: str, h1: float, h2: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64], int]:
    """Full-grid physics-floor (P7-2c) then extract ln(floored L) at two nodes.

    Returns (event_idx, logL_h1, logL_h2, n_events_excluded).
    """
    h_grid = np.sort(df["h"].unique())
    piv = df.pivot(index="event_idx", columns="h", values=channel).reindex(columns=h_grid)
    if piv.isna().any().any():
        raise ValueError(f"{channel}: pivot has missing (event, h) cells -- ragged CSV")
    event_idx = piv.index.to_numpy()
    L = piv.to_numpy(dtype=np.float64)
    L_floored, exclude_mask = _physics_floor_apply(L)
    n_excluded = int(exclude_mask.sum())
    if n_excluded:
        L_floored = L_floored[~exclude_mask]
        event_idx = event_idx[~exclude_mask]
    logL = np.log(L_floored)
    i1 = int(np.argmin(np.abs(h_grid - h1)))
    i2 = int(np.argmin(np.abs(h_grid - h2)))
    if h_grid[i1] != h1 or h_grid[i2] != h2:
        raise ValueError(f"node pair ({h1}, {h2}) not exact grid nodes (nearest {h_grid[i1]}, {h_grid[i2]})")
    return event_idx, logL[:, i1], logL[:, i2], n_excluded


def _raw_column_at_nodes(
    df: pd.DataFrame, column: str, h1: float, h2: float
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pivot a diagnostic column (no floor) at two h nodes; return (event_idx, v1, v2)."""
    sub = df[df["h"].isin([h1, h2])]
    piv = sub.pivot(index="event_idx", columns="h", values=column)
    if piv.isna().any().any():
        raise ValueError(f"{column}: pivot has missing (event, h) cells at nodes ({h1}, {h2})")
    event_idx = piv.index.to_numpy()
    return event_idx, piv[h1].to_numpy(dtype=np.float64), piv[h2].to_numpy(dtype=np.float64)


def _secant_slope(v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64], h1: float, h2: float) -> npt.NDArray[np.float64]:
    return (v2 - v1) / (h2 - h1)


# ---------------------------------------------------------------------------
# c_e' identity (Sec 2b, exact residual of the pinned Path-A mixture)
# ---------------------------------------------------------------------------


def _c_prime_at_node(
    alpha_G_phi: npt.NDArray[np.float64],
    L_cat_with_bh: npt.NDArray[np.float64],
    combined_with_bh: npt.NDArray[np.float64],
    D_tilde_phi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """c_e'(h) = 1 - alpha_G_phi * L_cat_with_bh / (combined_with_bh * D_tilde_phi)."""
    denom = combined_with_bh * D_tilde_phi
    with np.errstate(divide="ignore", invalid="ignore"):
        c_prime = 1.0 - alpha_G_phi * L_cat_with_bh / denom
    return c_prime


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _rank_biserial(ds_cat: npt.NDArray[np.float64], ds_noncat: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Mann-Whitney U + rank-biserial r_rb = 1 - 2U/(n1*n2).

    U is computed with x = the catalogue-supported group (cat_e = True),
    y = the non-catalogue-supported group (cat_e = False); n1 = len(x),
    n2 = len(y). With this convention POSITIVE r_rb means Delta s is
    stochastically LARGER in the non-catalogue-supported group -- the
    registered M-B direction (Sec 2b). Verified on toy extremes: cat
    group uniformly larger -> r_rb = -1; non-cat group uniformly larger
    -> r_rb = +1.
    """
    n1, n2 = len(ds_cat), len(ds_noncat)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    result = mannwhitneyu(ds_cat, ds_noncat, alternative="two-sided")
    U = float(result.statistic)
    r_rb = 1.0 - 2.0 * U / (n1 * n2)
    return U, r_rb


def _point_estimates(
    ds: npt.NDArray[np.float64],
    c_prime: npt.NDArray[np.float64],
    cat_e: npt.NDArray[np.bool_],
    g_e: npt.NDArray[np.float64],
    m_e: npt.NDArray[np.float64],
) -> dict[str, float]:
    """S1, S1a, S1b, S2 (r_rb), S3 point estimates. c_prime may contain NaNs (excluded)."""
    valid_c = np.isfinite(c_prime)
    s1 = float(spearmanr(ds[valid_c], c_prime[valid_c]).statistic)
    s1a = float(pointbiserialr(cat_e.astype(np.float64), ds).correlation)
    s1b = float(spearmanr(ds, g_e).statistic)
    _u, r_rb = _rank_biserial(ds[cat_e], ds[~cat_e])
    s3 = float(spearmanr(ds, m_e).statistic)
    return {"S1_spearman_c_prime": s1, "S1a_pointbiserial_cat_e": s1a, "S1b_spearman_g_e": s1b, "S2_rank_biserial": r_rb, "S3_spearman_m_e": s3}


def _bootstrap_ci(
    values_getter: Any,
    n_events: int,
    rng: np.random.Generator,
    b: int,
) -> dict[str, tuple[float, float]]:
    """Generic B-resample bootstrap; values_getter(idx) -> dict[str, float]."""
    samples: dict[str, list[float]] = {}
    for _ in range(b):
        idx = rng.integers(0, n_events, size=n_events)
        out = values_getter(idx)
        for k, v in out.items():
            samples.setdefault(k, []).append(v)
    cis: dict[str, tuple[float, float]] = {}
    for k, v in samples.items():
        arr = np.array(v, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            cis[k] = (float("nan"), float("nan"))
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            cis[k] = (float(lo), float(hi))
    return cis


# ---------------------------------------------------------------------------
# Per-venue pipeline
# ---------------------------------------------------------------------------


def _process_venue(venue: str, rng: np.random.Generator) -> dict[str, Any]:
    h1, h2 = NODE_PAIRS[venue]
    df = _load_venue_frame(venue)
    h_grid = np.sort(df["h"].unique())

    # --- slopes (P7-2b/2c pinned formula) -----------------------------------
    idx_1d, logL1_1d, logL2_1d, n_excl_1d = _floored_logL_at_nodes(df, "combined_no_bh", h1, h2)
    idx_2d, logL1_2d, logL2_2d, n_excl_2d = _floored_logL_at_nodes(df, "combined_with_bh", h1, h2)
    if not np.array_equal(idx_1d, idx_2d):
        raise ValueError(f"{venue}: 1D/2D event_idx mismatch after physics-floor exclusion")
    event_idx = idx_1d
    n_events = event_idx.size

    s1d = _secant_slope(logL1_1d, logL2_1d, h1, h2)
    s2d = _secant_slope(logL1_2d, logL2_2d, h1, h2)
    ds = s2d - s1d

    # --- covariates -----------------------------------------------------------
    _, Lcat1, Lcat2 = _raw_column_at_nodes(df, "L_cat_with_bh", h1, h2)
    _, g1, g2 = _raw_column_at_nodes(df, "g_frac", h1, h2)
    _, alpha1, alpha2 = _raw_column_at_nodes(df, "alpha_G_phi", h1, h2)
    _, Dtilde1, Dtilde2 = _raw_column_at_nodes(df, "D_tilde_phi", h1, h2)
    _, combwbh1, combwbh2 = _raw_column_at_nodes(df, "combined_with_bh", h1, h2)

    cat_e = (Lcat1 > 0.0) & (Lcat2 > 0.0)
    g_e = 0.5 * (g1 + g2)

    c_prime_1 = _c_prime_at_node(alpha1, Lcat1, combwbh1, Dtilde1)
    c_prime_2 = _c_prime_at_node(alpha2, Lcat2, combwbh2, Dtilde2)

    lo, hi = CPRIME_RANGE
    valid_1 = np.isfinite(c_prime_1) & (c_prime_1 >= lo) & (c_prime_1 <= hi)
    valid_2 = np.isfinite(c_prime_2) & (c_prime_2 >= lo) & (c_prime_2 <= hi)
    n_pairs = 2 * n_events
    n_violations = int((~valid_1).sum() + (~valid_2).sum())
    violation_frac = n_violations / n_pairs

    violator_event_idx = event_idx[(~valid_1) | (~valid_2)].tolist()
    gate_stop = violation_frac > CPRIME_STOP_FRAC

    c_prime = np.where(valid_1 & valid_2, 0.5 * (c_prime_1 + c_prime_2), np.nan)
    n_c_prime_excluded_events = int(np.isnan(c_prime).sum())

    tie_c_prime_one_exact = float((c_prime == 1.0).sum()) / n_events
    tie_c_prime_one_near = float(np.isclose(c_prime, 1.0, atol=1e-9, equal_nan=False).sum()) / n_events

    # --- crb join (event_idx = pre-filter RangeIndex, crb.iloc[event_idx]) ---
    crb = pd.read_csv(_crb_path(venue))
    crb_rows = crb.iloc[event_idx]
    in_catalog = crb_rows["in_catalog"].to_numpy(dtype=bool)
    snr = crb_rows["SNR"].to_numpy(dtype=np.float64)
    log10_snr = np.log10(snr)
    var_dL = crb_rows["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    dL = crb_rows["luminosity_distance"].to_numpy(dtype=np.float64)
    sigma_dL_over_dL = np.sqrt(var_dL) / dL
    z_e = np.array([dist_to_redshift(float(d), h=TRUTH_H) for d in dL], dtype=np.float64)
    M = crb_rows["M"].to_numpy(dtype=np.float64)
    median_M = float(np.median(M))
    ln_M_ratio_signed = np.log(M / median_M)
    m_e = np.abs(ln_M_ratio_signed)

    # --- gates (Sec 3, P2) ------------------------------------------------
    sum_s2d = float(s2d.sum())
    sum_abs_s2d = float(np.abs(s2d).sum())
    gate_a_pass = bool(abs(sum_s2d) <= GATE_A_TOL * sum_abs_s2d)
    sum_s1d = float(s1d.sum())
    gate_b_pass = bool(sum_s1d < 0.0)
    sum_ds = float(ds.sum())
    gate_c_positive = bool(sum_ds > 0.0)

    gates = {
        "G_a_2d_slopes_sum_near_zero": {
            "sum_s2d": sum_s2d,
            "sum_abs_s2d": sum_abs_s2d,
            "tolerance": GATE_A_TOL,
            "pass": gate_a_pass,
        },
        "G_b_1d_slopes_sum_negative": {"sum_s1d": sum_s1d, "pass": gate_b_pass},
        "G_c_ds_sum_positive_reported": {"sum_ds": sum_ds, "positive": gate_c_positive},
        "c_prime_validity_gate": {
            "range": list(CPRIME_RANGE),
            "n_pairs_checked": n_pairs,
            "n_violations": n_violations,
            "violation_frac": violation_frac,
            "stop_threshold": CPRIME_STOP_FRAC,
            "stop": gate_stop,
            "violator_event_idx": violator_event_idx,
            "n_violator_events_excluded_from_c_prime": n_c_prime_excluded_events,
        },
    }

    tie_zero_fractions = {
        "frac_c_prime_eq_1_exact_bitwise": tie_c_prime_one_exact,
        "frac_c_prime_eq_1_within_1e-9": tie_c_prime_one_near,
        "frac_L_cat_with_bh_zero_both_nodes": float(((Lcat1 == 0.0) & (Lcat2 == 0.0)).mean()),
        "frac_cat_e_true": float(cat_e.mean()),
        "frac_cat_e_false": float((~cat_e).mean()),
        "frac_L_cat_with_bh_zero_either_node": float(1.0 - cat_e.mean()),
        "frac_ds_exactly_zero": float(np.isclose(ds, 0.0).mean()),
    }

    result: dict[str, Any] = {
        "venue": venue,
        "node_pair": [h1, h2],
        "grid": h_grid.tolist(),
        "n_events": n_events,
        "n_events_excluded_physics_floor": {"combined_no_bh": n_excl_1d, "combined_with_bh": n_excl_2d},
        "gates": gates,
        "tie_zero_fractions": tie_zero_fractions,
    }

    if not (gate_a_pass and gate_b_pass) or gate_stop:
        result["STOP"] = True
        result["stop_reason"] = (
            "Gate failure: "
            + ("G-a " if not gate_a_pass else "")
            + ("G-b " if not gate_b_pass else "")
            + ("c_prime-validity " if gate_stop else "")
        ).strip()
        return result

    result["STOP"] = False

    # --- registered statistics (Sec 3) --------------------------------------
    point = _point_estimates(ds, c_prime, cat_e, g_e, m_e)

    # S4 controls (reported, non-adjudicating)
    s4 = {
        "spearman_log10_SNR": float(spearmanr(ds, log10_snr).statistic),
        "spearman_sigma_dL_over_dL": float(spearmanr(ds, sigma_dL_over_dL).statistic),
        "spearman_z_e": float(spearmanr(ds, z_e).statistic),
        "pointbiserial_in_catalog": float(pointbiserialr(in_catalog.astype(np.float64), ds).correlation),
    }
    # P10 reported-only signed-mass leg
    p10 = {
        "spearman_signed_ln_M_ratio": float(spearmanr(ds, ln_M_ratio_signed).statistic),
        "spearman_m_e_positive_leg": float(spearmanr(ds[ln_M_ratio_signed >= 0], m_e[ln_M_ratio_signed >= 0]).statistic)
        if (ln_M_ratio_signed >= 0).sum() >= 3
        else float("nan"),
        "spearman_m_e_negative_leg": float(spearmanr(ds[ln_M_ratio_signed < 0], m_e[ln_M_ratio_signed < 0]).statistic)
        if (ln_M_ratio_signed < 0).sum() >= 3
        else float("nan"),
    }

    # --- OLS (Sec 3 S4, HC3) --------------------------------------------------
    ols_result: dict[str, Any] | None = None
    if _HAVE_STATSMODELS:
        valid_ols = np.isfinite(c_prime)
        X = np.column_stack(
            [
                (c_prime[valid_ols] - np.nanmean(c_prime)) / np.nanstd(c_prime),
                (m_e[valid_ols] - m_e[valid_ols].mean()) / m_e[valid_ols].std(ddof=0),
                (log10_snr[valid_ols] - log10_snr[valid_ols].mean()) / log10_snr[valid_ols].std(ddof=0),
                (z_e[valid_ols] - z_e[valid_ols].mean()) / z_e[valid_ols].std(ddof=0),
            ]
        )
        Xc = sm.add_constant(X)
        model = sm.OLS(ds[valid_ols], Xc).fit(cov_type="HC3")
        ols_result = {
            "n_obs": int(valid_ols.sum()),
            "params": {
                name: float(v)
                for name, v in zip(
                    ["const", "c_prime_std", "m_e_std", "log10_SNR_std", "z_e_std"], model.params, strict=True
                )
            },
            "hc3_se": {
                name: float(v)
                for name, v in zip(
                    ["const", "c_prime_std", "m_e_std", "log10_SNR_std", "z_e_std"], model.bse, strict=True
                )
            },
            "pvalues": {
                name: float(v)
                for name, v in zip(
                    ["const", "c_prime_std", "m_e_std", "log10_SNR_std", "z_e_std"], model.pvalues, strict=True
                )
            },
            "r_squared": float(model.rsquared),
        }
    else:
        ols_result = {"skipped": True, "note": "statsmodels not installed in this venv -- OLS skipped per Sec 3 fallback."}

    # --- bootstrap (B=10,000, seed 20280612, events resampled w/ replacement) -
    def _resample_stats(idx: npt.NDArray[np.intp]) -> dict[str, float]:
        ds_b = ds[idx]
        c_prime_b = c_prime[idx]
        cat_e_b = cat_e[idx]
        g_e_b = g_e[idx]
        m_e_b = m_e[idx]
        log10_snr_b = log10_snr[idx]
        sigma_dL_b = sigma_dL_over_dL[idx]
        z_e_b = z_e[idx]
        in_catalog_b = in_catalog[idx]

        valid_c_b = np.isfinite(c_prime_b)
        out: dict[str, float] = {}
        out["S1"] = float(spearmanr(ds_b[valid_c_b], c_prime_b[valid_c_b]).statistic) if valid_c_b.sum() >= 3 else float("nan")
        out["S1a"] = float(pointbiserialr(cat_e_b.astype(np.float64), ds_b).correlation)
        out["S1b"] = float(spearmanr(ds_b, g_e_b).statistic)
        n1_b, n2_b = int(cat_e_b.sum()), int((~cat_e_b).sum())
        if n1_b >= 1 and n2_b >= 1:
            _u_b, r_rb_b = _rank_biserial(ds_b[cat_e_b], ds_b[~cat_e_b])
        else:
            r_rb_b = float("nan")
        out["S2"] = r_rb_b
        out["S3"] = float(spearmanr(ds_b, m_e_b).statistic)
        out["S4_log10_SNR"] = float(spearmanr(ds_b, log10_snr_b).statistic)
        out["S4_sigma_dL_over_dL"] = float(spearmanr(ds_b, sigma_dL_b).statistic)
        out["S4_z_e"] = float(spearmanr(ds_b, z_e_b).statistic)
        out["S4_in_catalog"] = float(pointbiserialr(in_catalog_b.astype(np.float64), ds_b).correlation)
        return out

    cis = _bootstrap_ci(_resample_stats, n_events, rng, BOOTSTRAP_B)

    def _excludes_zero(ci: tuple[float, float]) -> bool:
        lo, hi = ci
        return not (lo <= 0.0 <= hi)

    def _excludes_zero_positive(ci: tuple[float, float]) -> bool:
        lo, hi = ci
        return _excludes_zero(ci) and lo > 0.0

    statistics = {
        "S1_spearman_c_prime": {"point": point["S1_spearman_c_prime"], "ci95": cis["S1"], "excludes_zero": _excludes_zero(cis["S1"])},
        "S1a_pointbiserial_cat_e": {
            "point": point["S1a_pointbiserial_cat_e"],
            "ci95": cis["S1a"],
            "excludes_zero": _excludes_zero(cis["S1a"]),
        },
        "S1b_spearman_g_e": {"point": point["S1b_spearman_g_e"], "ci95": cis["S1b"], "excludes_zero": _excludes_zero(cis["S1b"])},
        "S2_rank_biserial": {
            "point": point["S2_rank_biserial"],
            "ci95": cis["S2"],
            "excludes_zero": _excludes_zero(cis["S2"]),
            "orientation": (
                "POSITIVE = Delta s stochastically larger in the NON-catalogue-supported group "
                "(cat_e = False) -- the registered M-B direction. r_rb = 1 - 2U/(n1*n2) with "
                "U = mannwhitneyu(ds[cat_e], ds[~cat_e]).statistic, n1 = n_cat_true, n2 = n_cat_false."
            ),
        },
        "S3_spearman_m_e": {"point": point["S3_spearman_m_e"], "ci95": cis["S3"], "excludes_zero": _excludes_zero(cis["S3"])},
        "S4_controls": {
            "log10_SNR": {"point": s4["spearman_log10_SNR"], "ci95": cis["S4_log10_SNR"]},
            "sigma_dL_over_dL": {"point": s4["spearman_sigma_dL_over_dL"], "ci95": cis["S4_sigma_dL_over_dL"]},
            "z_e": {"point": s4["spearman_z_e"], "ci95": cis["S4_z_e"]},
            "in_catalog": {"point": s4["pointbiserial_in_catalog"], "ci95": cis["S4_in_catalog"]},
        },
        "P10_mass_extremity_reported": p10,
        "S4_ols_hc3": ols_result,
    }

    leg_b = _excludes_zero_positive(cis["S1"]) and _excludes_zero_positive(cis["S2"])
    leg_c = _excludes_zero(cis["S3"])
    result["legs"] = {"L_B": leg_b, "L_C": leg_c}

    # --- sensitivity (reported, point estimates only, no bootstrap) ---------
    sensitivity: dict[str, Any] = {}
    for side, (sh1, sh2) in SENSITIVITY_PAIRS[venue].items():
        s_idx_1d, s_l1_1d, s_l2_1d, _ne1 = _floored_logL_at_nodes(df, "combined_no_bh", sh1, sh2)
        s_idx_2d, s_l1_2d, s_l2_2d, _ne2 = _floored_logL_at_nodes(df, "combined_with_bh", sh1, sh2)
        if not np.array_equal(s_idx_1d, s_idx_2d) or not np.array_equal(s_idx_1d, event_idx):
            raise ValueError(f"{venue}/{side}: event_idx mismatch across channels/registered pair")
        s_s1d = _secant_slope(s_l1_1d, s_l2_1d, sh1, sh2)
        s_s2d = _secant_slope(s_l1_2d, s_l2_2d, sh1, sh2)
        s_ds = s_s2d - s_s1d

        _, s_Lcat1, s_Lcat2 = _raw_column_at_nodes(df, "L_cat_with_bh", sh1, sh2)
        _, s_alpha1, s_alpha2 = _raw_column_at_nodes(df, "alpha_G_phi", sh1, sh2)
        _, s_Dtilde1, s_Dtilde2 = _raw_column_at_nodes(df, "D_tilde_phi", sh1, sh2)
        _, s_comb1, s_comb2 = _raw_column_at_nodes(df, "combined_with_bh", sh1, sh2)
        s_cat_e = (s_Lcat1 > 0.0) & (s_Lcat2 > 0.0)
        s_cp1 = _c_prime_at_node(s_alpha1, s_Lcat1, s_comb1, s_Dtilde1)
        s_cp2 = _c_prime_at_node(s_alpha2, s_Lcat2, s_comb2, s_Dtilde2)
        s_v1 = np.isfinite(s_cp1) & (s_cp1 >= lo) & (s_cp1 <= hi)
        s_v2 = np.isfinite(s_cp2) & (s_cp2 >= lo) & (s_cp2 <= hi)
        s_c_prime = np.where(s_v1 & s_v2, 0.5 * (s_cp1 + s_cp2), np.nan)

        s_point = _point_estimates(s_ds, s_c_prime, s_cat_e, g_e, m_e)
        sensitivity[side] = {"node_pair": [sh1, sh2], **s_point}

    result["sensitivity"] = sensitivity
    result["statistics"] = statistics

    return result, {
        "event_idx": event_idx,
        "s1d": s1d,
        "s2d": s2d,
        "ds": ds,
        "cat_e": cat_e,
        "g_e": g_e,
        "c_prime": c_prime,
        "in_catalog": in_catalog,
        "snr": snr,
        "sigma_dL_over_dL": sigma_dL_over_dL,
        "z_e": z_e,
        "m_e": m_e,
        "ln_M_ratio_signed": ln_M_ratio_signed,
    }


def _write_covariate_csv(venue: str, audit: dict[str, Any], out_dir: Path) -> None:
    df = pd.DataFrame(
        {
            "event_idx": audit["event_idx"],
            "s1d": audit["s1d"],
            "s2d": audit["s2d"],
            "ds": audit["ds"],
            "cat_e": audit["cat_e"],
            "g_e": audit["g_e"],
            "c_prime": audit["c_prime"],
            "in_catalog": audit["in_catalog"],
            "SNR": audit["snr"],
            "log10_SNR": np.log10(audit["snr"]),
            "sigma_dL_over_dL": audit["sigma_dL_over_dL"],
            "z_e": audit["z_e"],
            "m_e": audit["m_e"],
            "ln_M_ratio_signed": audit["ln_M_ratio_signed"],
        }
    )
    df.to_csv(out_dir / f"regression_prod_native_covariates_{venue}.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "regression_prod_native_output.json")
    args = parser.parse_args(argv)

    print("Registered node pairs:", NODE_PAIRS)
    print("Registered sensitivity pairs:", SENSITIVITY_PAIRS)

    out: dict[str, Any] = {
        "node_pairs": NODE_PAIRS,
        "sensitivity_pairs": SENSITIVITY_PAIRS,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "statsmodels_available": _HAVE_STATSMODELS,
        "venues": {},
    }

    any_stop = False
    for venue in VENUES:
        print(f"\n=== {venue} ===", flush=True)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        processed = _process_venue(venue, rng)
        if isinstance(processed, tuple):
            venue_result, audit = processed
            _write_covariate_csv(venue, audit, HERE)
        else:
            venue_result = processed
            any_stop = True
            print(f"  STOP: {venue_result.get('stop_reason')}")
        out["venues"][venue] = venue_result

    out["any_stop"] = any_stop
    args.output.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
