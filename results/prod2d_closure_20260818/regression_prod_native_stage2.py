"""Stage-2: per-event R&V15-propagated sigma_M covariate (runbook 21 Sec 3 item 1).

PREREGISTRATION_PROD_REGRESSION.md v2 Sec 2 ("Per-event sigma_M (R&V15-propagated)")
+ Sec 3 decision-table rule (3). Triggered because stage-1
(``regression_prod_native.py``) found: S1 CI excludes 0 in both venues but in the
WRONG (non-M-B) direction, so S2's leg (excludes-zero-positive) fails in both venues
-- exactly decision-table rule (3)'s second clause ("S1 CI excludes 0 in both venues
but S2 fails in >= 1 venue").

Scope (P5): the N_cat ~= 76 in-catalog events (4.78% of 1588), the SAME set in both
venues (verified below) -- disclosure, not independent confirmation.

Production sigma_M formula reproduced EXACTLY (file:line):
    darksiren_emri/galaxy_catalogue/handler.py:1337-1351
    ``_empiric_stellar_mass_to_BH_mass_relation(stellar_mass, stellar_mass_error)``
        BH_mass = exp(alpha + beta * ln(stellar_mass / 10))
        BH_mass_error = BH_mass * sqrt(
            sigma_int**2 + d_alpha**2 + (ln(stellar_mass/10)*d_beta)**2
            + (beta/stellar_mass * stellar_mass_error)**2
        )
    with alpha = 7.45*ln(10), beta = 1.05, d_alpha = 0.08*ln(10), d_beta = 0.11,
    sigma_int = 0.24*ln(10) (Reines & Volonteri 2015, arXiv:1508.06274, Eq. 5 + Sec 4.1).
    Applied catalog-wide at load time by
    ``GalaxyCatalogueHandler._map_stellar_masses_to_BH_masses``
    (handler.py:1105-1111), so ``HostGalaxy.M_error`` (handler.py:79, set from
    ``InternalCatalogColumns.BH_MASS_ERROR`` at HostGalaxy construction, handler.py:77)
    IS sigma_M as consumed downstream unchanged, e.g.
    ``bayesian_statistics.py:6336``: ``sigma_gal_frac = possible_host.M_error * (1+z)
    / detection.M`` and ``bayesian_statistics.py:6247``:
    ``norm(loc=possible_host.M, scale=possible_host.M_error)``.

    Because BH_mass_error = BH_mass * sigma_lnM with
    sigma_lnM = sqrt(sigma_int**2 + d_alpha**2 + (ln(M*/10)*d_beta)**2
                      + (beta/M* * dM*)**2)
    the FRACTIONAL sigma_M = M_error / M = sigma_lnM EXACTLY (no small-error
    approximation needed); the dex width is sigma_lnM / ln(10).

Row alignment (verified against the live handler, see stage-2 execution log):
    GalaxyCatalogueHandler.__init__ prunes the raw reduced catalogue
    (mass-info removal + ``_mass_redshift_prune_mask``) and then calls
    ``setup_galaxy_catalog_balltree``, which does
    ``self.reduced_galaxy_catalog = self.reduced_galaxy_catalog.reset_index()``
    (handler.py:555) -- i.e. by the time hosts are drawn during simulation,
    ``HostGalaxy.catalog_index`` (-> CRB ``host_galaxy_index``) is a POSITIONAL
    index (0..N_pruned-1) into the pruned+reset catalogue, NOT the raw GLADE+
    row number. This script replicates the SAME prune+reset pipeline manually
    (skipping only the ecliptic-rotation/angle steps, which do not reorder or
    remove rows and do not touch mass columns) to recover the pre-transform
    stellar mass at the SAME row position, and verifies the replicate's
    BH_mass/BH_mass_error/redshift against ``GalaxyCatalogueHandler.
    get_host_galaxy_by_index`` for every joined event as an exact-match gate
    (STOP on any mismatch).

Statistic (registered, Sec 3 P5): Spearman rho(Delta s_e, sigma_M,e) on the
in-catalog subset, per venue; bootstrap B = 10,000, seed 20280612, numpy
default_rng, resample events with replacement, 95% percentile CI -- the SAME
bootstrap machinery as stage-1 (``regression_prod_native._bootstrap_ci``).
Registered leg direction: POSITIVE.

Delta s_e, in_catalog, and event_idx are taken from stage-1's own per-venue
processing (``regression_prod_native._process_venue``), imported directly (not
re-derived) so stage-2 cannot silently diverge from the stage-1 pinned slope
formula/node pairs.

Usage:
    uv run python regression_prod_native_stage2.py [--output regression_prod_native_stage2_output.json]
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import regression_prod_native as stage1
from scipy.stats import spearmanr

from darksiren_emri.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
    _empiric_stellar_mass_to_BH_mass_relation,
    _mass_redshift_prune_mask,
)

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]

VENUES = ("iiib", "joint_r1")

BOOTSTRAP_B = 10_000
BOOTSTRAP_SEED = 20280612

# Production catalogue-load band (darksiren_emri/main.py:154-161): the SOURCE-frame
# mass band + default max_redshift=1.5 (Model1CrossCheck, cosmological_model.py:200,
# no --max_redshift override for this campaign). No --observed_catalogue for this
# run (baseline reduced catalogue, byte-identical load path).
GALAXY_Z_MAX = 1.5
LN10 = float(np.log(10.0))


# ---------------------------------------------------------------------------
# Pre-transform stellar-mass replicate of GalaxyCatalogueHandler's pipeline
# ---------------------------------------------------------------------------


def _build_pruned_catalog_with_stellar_mass(handler: GalaxyCatalogueHandler) -> pd.DataFrame:
    """Replicate the handler's prune+reset pipeline, retaining pre-transform M*.

    Mirrors ``GalaxyCatalogueHandler.__init__`` exactly for the steps that
    affect row selection/order/mass values: ``_map_stellar_masses_to_BH_masses``
    (handler.py:1105-1111, i.e. ``_empiric_stellar_mass_to_BH_mass_relation``),
    ``_remove_galaxies_without_mass_information`` (BH_MASS.isna() drop),
    ``_get_pruned_galaxy_catalog`` (``_mass_redshift_prune_mask``), and the
    ``reset_index()`` done inside ``setup_galaxy_catalog_balltree``
    (handler.py:555). The ecliptic-rotation/angle-mapping steps are SKIPPED --
    they touch only PHI_S/THETA_S, never reorder or drop rows, and are
    irrelevant to mass/redshift. Row position ``i`` in the returned frame is
    therefore identical to row position ``i`` in
    ``handler.reduced_galaxy_catalog`` (verified below), i.e. exactly the
    ``host_galaxy_index`` used by the production CRB join.
    """
    raw = handler.read_reduced_galaxy_catalog()
    stellar_mass_raw = raw[InternalCatalogColumns.BH_MASS].copy()
    stellar_mass_error_raw = raw[InternalCatalogColumns.BH_MASS_ERROR].copy()
    BH_mass, BH_mass_error = _empiric_stellar_mass_to_BH_mass_relation(
        raw[InternalCatalogColumns.BH_MASS], raw[InternalCatalogColumns.BH_MASS_ERROR]
    )
    raw[InternalCatalogColumns.BH_MASS] = BH_mass
    raw[InternalCatalogColumns.BH_MASS_ERROR] = BH_mass_error
    raw["STELLAR_MASS_1E10_MSUN"] = stellar_mass_raw
    raw["STELLAR_MASS_ERROR_1E10_MSUN"] = stellar_mass_error_raw

    raw = raw[~raw[InternalCatalogColumns.BH_MASS].isna()]
    keep_mask = _mass_redshift_prune_mask(
        raw[InternalCatalogColumns.BH_MASS],
        raw[InternalCatalogColumns.BH_MASS_ERROR],
        raw[InternalCatalogColumns.REDSHIFT],
        raw[InternalCatalogColumns.REDSHIFT_ERROR],
        M_SOURCE_FRAME_MIN,
        M_SOURCE_FRAME_MAX,
        GALAXY_Z_MAX,
    )
    return raw[keep_mask].reset_index(drop=True)


def _sigma_M_for_events(
    handler: GalaxyCatalogueHandler,
    pruned: pd.DataFrame,
    host_galaxy_index: npt.NDArray[np.int64],
) -> pd.DataFrame:
    """Look up (stellar mass, BH mass, sigma_M) for each host_galaxy_index.

    Cross-checks every row against ``handler.get_host_galaxy_by_index`` (the
    LIVE production accessor) with an exact-match assertion -- STOP on any
    mismatch (row-alignment safety gate).
    """
    rows = pruned.iloc[host_galaxy_index]
    stellar_mass = rows["STELLAR_MASS_1E10_MSUN"].to_numpy(dtype=np.float64)
    stellar_mass_error = rows["STELLAR_MASS_ERROR_1E10_MSUN"].to_numpy(dtype=np.float64)
    bh_mass = rows[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64)
    bh_mass_error = rows[InternalCatalogColumns.BH_MASS_ERROR].to_numpy(dtype=np.float64)
    host_z = rows[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64)

    mismatches = []
    for i, idx in enumerate(host_galaxy_index):
        g = handler.get_host_galaxy_by_index(int(idx))
        if not (
            np.isclose(g.M, bh_mass[i], rtol=1e-10)
            and np.isclose(g.M_error, bh_mass_error[i], rtol=1e-10)
            and np.isclose(g.z, host_z[i], rtol=1e-10)
        ):
            mismatches.append(int(idx))
    if mismatches:
        raise ValueError(
            f"Row-alignment verification FAILED for host_galaxy_index {mismatches[:10]} "
            f"(+{max(0, len(mismatches) - 10)} more) -- pruned-catalogue replicate does not "
            "match the live GalaxyCatalogueHandler.get_host_galaxy_by_index. STOP per P5/row-"
            "alignment safety gate."
        )

    sigma_M_frac = bh_mass_error / bh_mass  # exact lognormal width sigma_lnM
    sigma_M_dex = sigma_M_frac / LN10

    return pd.DataFrame(
        {
            "host_galaxy_index": host_galaxy_index,
            "stellar_mass_1e10_Msun": stellar_mass,
            "stellar_mass_error_1e10_Msun": stellar_mass_error,
            "host_redshift": host_z,
            "BH_mass_Msun": bh_mass,
            "sigma_M_Msun": bh_mass_error,
            "sigma_M_fractional": sigma_M_frac,
            "sigma_M_dex": sigma_M_dex,
        }
    )


# ---------------------------------------------------------------------------
# Statistic + bootstrap (same conventions as stage-1)
# ---------------------------------------------------------------------------


def _bootstrap_spearman_ci(
    ds: npt.NDArray[np.float64],
    sigma_M: npt.NDArray[np.float64],
    seed: int,
    b: int,
) -> tuple[float, tuple[float, float]]:
    point = float(spearmanr(ds, sigma_M).statistic)
    rng = np.random.default_rng(seed)
    n = ds.size
    samples = np.empty(b, dtype=np.float64)
    for k in range(b):
        idx = rng.integers(0, n, size=n)
        samples[k] = spearmanr(ds[idx], sigma_M[idx]).statistic
    samples = samples[np.isfinite(samples)]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return point, (float(lo), float(hi))


def _process_venue(
    venue: str, handler: GalaxyCatalogueHandler, pruned: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    stage1_rng = np.random.default_rng(stage1.BOOTSTRAP_SEED)
    processed = stage1._process_venue(venue, stage1_rng)
    if isinstance(processed, tuple):
        _venue_result, audit = processed
    else:
        raise ValueError(
            f"{venue}: stage-1 processing STOPPED ({processed.get('stop_reason')}); cannot run stage-2"
        )

    event_idx = audit["event_idx"]
    ds = audit["ds"]
    in_catalog = audit["in_catalog"]

    cat_mask = in_catalog
    n_cat = int(cat_mask.sum())

    crb = pd.read_csv(stage1._crb_path(venue))
    crb_rows = crb.iloc[event_idx[cat_mask]]
    host_galaxy_index = crb_rows["host_galaxy_index"].to_numpy(dtype=np.int64)

    ds_cat = ds[cat_mask]
    event_idx_cat = event_idx[cat_mask]

    sigma_df = _sigma_M_for_events(handler, pruned, host_galaxy_index)

    point, ci = _bootstrap_spearman_ci(
        ds_cat, sigma_df["sigma_M_dex"].to_numpy(), BOOTSTRAP_SEED, BOOTSTRAP_B
    )

    ci_lo, ci_hi = ci
    excludes_zero = not (ci_lo <= 0.0 <= ci_hi)
    excludes_zero_positive = excludes_zero and ci_lo > 0.0

    sigma_M_dex_arr = sigma_df["sigma_M_dex"].to_numpy()
    sigma_M_frac_arr = sigma_df["sigma_M_fractional"].to_numpy()

    result = {
        "venue": venue,
        "n_events": n_cat,
        "n_events_total_venue": int(ds.size),
        "frac_in_catalog": n_cat / ds.size,
        "sigma_M_dex_distribution": {
            "min": float(sigma_M_dex_arr.min()),
            "median": float(np.median(sigma_M_dex_arr)),
            "max": float(sigma_M_dex_arr.max()),
        },
        "sigma_M_fractional_distribution": {
            "min": float(sigma_M_frac_arr.min()),
            "median": float(np.median(sigma_M_frac_arr)),
            "max": float(sigma_M_frac_arr.max()),
        },
        "spearman_ds_sigma_M_dex": {
            "point": point,
            "ci95": [ci_lo, ci_hi],
            "ci_width": ci_hi - ci_lo,
            "excludes_zero": excludes_zero,
            "excludes_zero_positive_M_A_direction": excludes_zero_positive,
        },
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    audit_df = pd.DataFrame(
        {
            "event_idx": event_idx_cat,
            "host_galaxy_index": sigma_df["host_galaxy_index"],
            "stellar_mass_1e10_Msun": sigma_df["stellar_mass_1e10_Msun"],
            "host_redshift": sigma_df["host_redshift"],
            "BH_mass_Msun": sigma_df["BH_mass_Msun"],
            "sigma_M_Msun": sigma_df["sigma_M_Msun"],
            "sigma_M_fractional": sigma_df["sigma_M_fractional"],
            "sigma_M_dex": sigma_df["sigma_M_dex"],
            "ds": ds_cat,
        }
    )

    return result, audit_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE / "regression_prod_native_stage2_output.json"
    )
    args = parser.parse_args(argv)

    print(
        "Building GalaxyCatalogueHandler (production band: "
        f"M in [{M_SOURCE_FRAME_MIN}, {M_SOURCE_FRAME_MAX}], z_max={GALAXY_Z_MAX})...",
        flush=True,
    )
    handler = GalaxyCatalogueHandler(
        M_min=M_SOURCE_FRAME_MIN, M_max=M_SOURCE_FRAME_MAX, z_max=GALAXY_Z_MAX
    )
    print(f"  handler catalogue rows: {len(handler.reduced_galaxy_catalog)}", flush=True)

    print(
        "Replicating prune+reset pipeline with retained pre-transform stellar mass...", flush=True
    )
    pruned = _build_pruned_catalog_with_stellar_mass(handler)
    print(f"  replicate rows: {len(pruned)}", flush=True)
    if len(pruned) != len(handler.reduced_galaxy_catalog):
        raise ValueError(
            f"Row-count mismatch: replicate={len(pruned)} vs handler={len(handler.reduced_galaxy_catalog)} "
            "-- STOP, pipeline replicate diverges from production."
        )

    out: dict[str, Any] = {
        "production_formula": (
            "darksiren_emri/galaxy_catalogue/handler.py:1337-1351 "
            "_empiric_stellar_mass_to_BH_mass_relation (Reines & Volonteri 2015, "
            "arXiv:1508.06274, Eq. 5 + Sec 4.1 0.24 dex intrinsic scatter); applied "
            "catalog-wide at load by _map_stellar_masses_to_BH_masses (handler.py:1105-1111)."
        ),
        "galaxy_catalog_band": {
            "M_min": M_SOURCE_FRAME_MIN,
            "M_max": M_SOURCE_FRAME_MAX,
            "z_max": GALAXY_Z_MAX,
        },
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "venues": {},
    }

    host_sets: dict[str, set] = {}
    for venue in VENUES:
        print(f"\n=== {venue} ===", flush=True)
        venue_result, audit_df = _process_venue(venue, handler, pruned)
        audit_df.to_csv(HERE / f"regression_prod_native_stage2_covariates_{venue}.csv", index=False)
        out["venues"][venue] = venue_result
        host_sets[venue] = set(audit_df["host_galaxy_index"].tolist())
        print(
            f"  n={venue_result['n_events']}  "
            f"rho(ds, sigma_M_dex) point={venue_result['spearman_ds_sigma_M_dex']['point']:.4f}  "
            f"ci95={venue_result['spearman_ds_sigma_M_dex']['ci95']}",
            flush=True,
        )

    out["P5_same_subset_both_venues"] = host_sets[VENUES[0]] == host_sets[VENUES[1]]
    out["n_common_host_galaxy_index"] = len(host_sets[VENUES[0]] & host_sets[VENUES[1]])

    both_venues_positive = all(
        out["venues"][v]["spearman_ds_sigma_M_dex"]["excludes_zero_positive_M_A_direction"]
        for v in VENUES
    )
    out["sigma_M_leg_R_CLASS_OWNED_M_A_form"] = both_venues_positive
    if not both_venues_positive:
        out["disclosure"] = (
            "UNDERPOWERED-NULL: sigma_M leg does not fire (CI does not exclude 0 in the "
            "positive M-A direction in both venues) on the n~=76 in-catalog subset. Per "
            "PREREGISTRATION_PROD_REGRESSION.md P5, this is reported as underpowered-null, "
            "never as refutation of M-A. See per-venue ci_width above."
        )

    args.output.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {args.output}")
    print(f"P5 same-subset-both-venues: {out['P5_same_subset_both_venues']}")
    print(f"sigma_M leg R-CLASS-OWNED (M-A form): {both_venues_positive}")


if __name__ == "__main__":
    main()
