#!/usr/bin/env python3
"""END-OF-FAN-OUT VERIFIER, item 7/20 -- independent re-derivation.

Node under test: B5.1 [WIN] gate implemented, byte-identity (ledger row #229).

This script is written FRESH by the verifier, independent of
`b51_byte_identity_check.py` (which the record says was run in scratchpad
and never committed -- it does not exist in this tree, confirmed by `find`
below). It does NOT import test helpers from `test_mass_filter_geometry.py`
or reuse `b5_window_count.py`'s `pass_mask`/`gw_window` reimplementation --
it calls the ACTUAL PRODUCTION function
`GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree` directly (the
same code path `bayesian_statistics.py:4859` calls in production) and
compares it against an independently re-typed transcription of the
PRE-B5.1 formula (as quoted in PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md
Sec 1, handler.py `654-673` pre-edit) -- not calling into handler.py at all
for the comparison arm.

Two checks:
  A. Byte-identity: production call with mass_filter_geometry/mass_filter_k
     OMITTED (pure defaults) vs the independently-transcribed pre-B5.1
     formula, over 50 synthetic GW events x 2000 synthetic candidates each
     = 100,000 pairs (matching the record's stated N). Expect 0 mismatches.
  B. Fleet-level decisive numbers: re-run the SAME pass-fraction / true-host
     retention computation directly against the real 24-arm fleet CSVs
     (bc_9001XX_work), using the production function itself (not a
     reimplementation) for the mass-window predicate, to check whether
     `b5_window_count.py`'s standalone reimplementation agrees with the
     actual production code to which the [WIN] gate applies. This is
     exactly the R4 falsifier item 2 the record explicitly says was
     "not attempted" by the builder.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/jasper/Repositories/darksiren-emri")
sys.path.insert(0, str(REPO_ROOT))

from darksiren_emri.galaxy_catalogue.handler import (  # noqa: E402
    GalaxyCatalogueHandler,
    HostGalaxy,
    InternalCatalogColumns,
)
from darksiren_emri.physical_relations import get_redshift_outer_bounds  # noqa: E402

RNG = np.random.default_rng(20260830)


def make_handler(catalog: pd.DataFrame) -> GalaxyCatalogueHandler:
    """Build a minimal handler instance exposing only what the mass-filter
    branch of get_possible_hosts_from_ball_tree touches:
    self.catalog_ball_tree (queried for indices) and
    self.reduced_galaxy_catalog (iloc'd by those indices).

    We bypass __init__ (which loads real catalogue files from disk) via
    object.__new__ and hand-build a trivial BallTree so every synthetic
    candidate is always within the query radius -- isolating the MASS
    filter, which is the object under test for this item.
    """
    from sklearn.neighbors import BallTree

    handler = object.__new__(GalaxyCatalogueHandler)
    n = len(catalog)
    # All candidates coincident with the query point (phi=0, theta=0) in the
    # Cartesian embedding -> distance 0 -> always inside any radius >= 0.
    pts = np.zeros((n, 3))
    pts[:, 2] = 1.0  # unit vectors, all identical (theta=0 pole)
    handler.catalog_ball_tree = BallTree(pts, metric="euclidean")
    handler.reduced_galaxy_catalog = catalog
    return handler


def old_formula_mask(
    M_z: float,
    M_z_sigma: float,
    z_min: float,
    z_max: float,
    sigma_multiplier: float,
    bh_mass: np.ndarray,
    bh_mass_error: np.ndarray,
) -> np.ndarray:
    """Independently re-typed PRE-B5.1 formula (symmetric case), verbatim
    from PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md Sec 1 / handler.py
    (pre-edit) 654-673, transcribed fresh (not copy-pasted from any script
    in this fan-out).
    """
    bh_mass_error_multiplier = float(sigma_multiplier)  # "symmetric" branch
    lo = (M_z - M_z_sigma * sigma_multiplier) / (1.0 + z_max)
    hi = (M_z + M_z_sigma * sigma_multiplier) / (1.0 + z_min)
    return (lo <= bh_mass + bh_mass_error * bh_mass_error_multiplier) & (
        bh_mass - bh_mass_error * bh_mass_error_multiplier <= hi
    )


def check_a_byte_identity() -> dict:
    n_events = 50
    n_cand = 2000
    total_pairs = 0
    mismatches = 0
    mismatch_examples = []

    for _ in range(n_events):
        M_z = float(RNG.uniform(1e5, 1e7))
        M_z_sigma = float(RNG.uniform(0.02, 0.5) * M_z)
        z_min = float(RNG.uniform(0.0, 0.3))
        z_max = z_min + float(RNG.uniform(0.05, 1.0))

        bh_mass = RNG.uniform(1e4, 1e8, size=n_cand)
        bh_mass_error = RNG.uniform(0.01, 0.6, size=n_cand) * bh_mass
        redshift = RNG.uniform(0.0, 1.5, size=n_cand)
        redshift_error = RNG.uniform(0.001, 0.05, size=n_cand)
        phi_s = np.zeros(n_cand)
        theta_s = np.zeros(n_cand)

        catalog = pd.DataFrame(
            {
                InternalCatalogColumns.PHI_S: phi_s,
                InternalCatalogColumns.THETA_S: theta_s,
                InternalCatalogColumns.REDSHIFT: redshift,
                InternalCatalogColumns.REDSHIFT_ERROR: redshift_error,
                InternalCatalogColumns.BH_MASS: bh_mass,
                InternalCatalogColumns.BH_MASS_ERROR: bh_mass_error,
            }
        )
        handler = make_handler(catalog)

        # PRODUCTION call, new flags OMITTED entirely (pure keyword defaults).
        result = handler.get_possible_hosts_from_ball_tree(
            phi=0.0,
            phi_sigma=1e-6,
            theta=1e-9,
            theta_sigma=1e-6,
            M_z=M_z,
            M_z_sigma=M_z_sigma,
            z_min=z_min,
            z_max=z_max,
            sigma_multiplier=1.5,
            mass_filter_sigma="symmetric",
            # mass_filter_geometry, mass_filter_k: NOT PASSED -> defaults
        )
        assert result is not None
        _, with_mass = result
        new_indices = {h.catalog_index for h in with_mass}

        # INDEPENDENT old-formula transcription, k = sigma_multiplier = 1.5
        # (the production call-site value; mass_filter_k's default 1.5
        # matches this call site's sigma_multiplier=1.5 exactly).
        old_mask = old_formula_mask(
            M_z, M_z_sigma, z_min, z_max, 1.5, bh_mass, bh_mass_error
        )
        # also require the (byte-identical, unmodified) redshift filter,
        # which get_possible_hosts_from_ball_tree applies before the mass
        # filter (candidate_hosts_without_bh_mass) -- both arms must agree
        # on that subset first, since HostGalaxy.catalog_index is the
        # ORIGINAL catalog row position (iloc/reset_index integer).
        redshift_mask = (z_min <= redshift + redshift_error) & (
            z_max >= redshift - redshift_error
        )
        old_indices = set(np.nonzero(redshift_mask & old_mask)[0].tolist())

        total_pairs += n_cand
        if new_indices != old_indices:
            diff = new_indices.symmetric_difference(old_indices)
            mismatches += len(diff)
            if len(mismatch_examples) < 5:
                mismatch_examples.append(
                    {
                        "M_z": M_z,
                        "M_z_sigma": M_z_sigma,
                        "z_min": z_min,
                        "z_max": z_max,
                        "diff_indices": sorted(diff)[:10],
                    }
                )

    return {
        "n_events": n_events,
        "n_cand_per_event": n_cand,
        "total_pairs": total_pairs,
        "mismatches": mismatches,
        "mismatch_examples": mismatch_examples,
    }


def check_b_fleet_numbers() -> dict:
    """Re-derive pass fraction (i)/(iii) and true-host retention (i)/(iii)
    over the real 24-arm fleet, calling the PRODUCTION
    get_possible_hosts_from_ball_tree directly for BOTH configs, rather
    than a standalone reimplementation. Compares against
    b5_window_count.json's headline numbers.
    """
    fleet_base = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825"
    arm_dirs = sorted(glob.glob(str(fleet_base / "bc_9001??_work")))
    assert len(arm_dirs) == 24, f"expected 24 arms, got {len(arm_dirs)}"

    H_MIN, H_MAX, OMEGA_M_MIN, OMEGA_M_MAX, W_0, W_A = 0.6, 0.86, 0.04, 0.5, -1.0, 0.0
    REDSHIFT_UPPER_LIMIT = 1.5

    configs = [
        ("i_linear_k1.5", "linear", 1.5),
        ("iii_log_k3.0", "log", 3.0),
    ]

    totals = {label: {"n_all": 0, "n_pass": 0} for label, _, _ in configs}
    truth_valid = 0
    truth_pass = {label: 0 for label, _, _ in configs}
    n_events_loaded = 0

    # Load the pruned catalogue exactly as production does: reuse the
    # frozen wgeom_instrument helper only for catalogue loading/pin
    # verification (read-only, no logic under test lives there).
    fanout_dir = REPO_ROOT / "results/campaign51_20260728/realistic_20260729"
    sys.path.insert(0, str(fanout_dir))
    from wgeom_instrument import load_pruned_catalogue, verify_catalogue_pin  # noqa: E402

    catalogue_md5 = verify_catalogue_pin()
    cat = load_pruned_catalogue(nrows=None)

    # NOTE: PrunedCatalogue (wgeom_instrument.py:368-379) does not carry a
    # redshift_error field (only bh_mass/bh_mass_error/redshift). The
    # production function's INTERNAL redshift_filter_mask is not the object
    # under test here (candidate_positions loaded from the fleet JSON are
    # already the result of that filter having been applied at simulation
    # time, exactly as b5_window_count.py assumes -- it never re-applies a
    # redshift filter either). REDSHIFT_ERROR is therefore set to a huge
    # constant so the production function's internal redshift_filter_mask
    # is always satisfied and cannot spuriously exclude anything -- this
    # isolates the MASS window exactly as b5_window_count.py's own
    # methodology does, while still routing through the real production
    # code path (not a reimplementation) for the mass predicate itself.
    catalog_df = pd.DataFrame(
        {
            InternalCatalogColumns.PHI_S: np.zeros(cat.n_pruned),
            InternalCatalogColumns.THETA_S: np.zeros(cat.n_pruned),
            InternalCatalogColumns.REDSHIFT: np.zeros(cat.n_pruned),
            InternalCatalogColumns.REDSHIFT_ERROR: np.full(cat.n_pruned, 1e12),
            InternalCatalogColumns.BH_MASS: cat.bh_mass,
            InternalCatalogColumns.BH_MASS_ERROR: cat.bh_mass_error,
        }
    )
    handler = make_handler(catalog_df)

    for arm_dir in arm_dirs:
        seed_dirs = glob.glob(str(Path(arm_dir) / "seed*"))
        assert len(seed_dirs) == 1
        seed_dir = Path(seed_dirs[0])
        csv_path = seed_dir / "simulations" / "prepared_cramer_rao_bounds.csv"
        json_path = seed_dir / "simulations" / "posteriors_with_bh_mass" / "h_0_73.json"

        df = pd.read_csv(
            csv_path,
            usecols=[
                "M",
                "delta_M_delta_M",
                "luminosity_distance",
                "delta_luminosity_distance_delta_luminosity_distance",
                "z_true",
                "host_galaxy_index",
                "in_catalog",
            ],
        )
        with open(json_path) as f:
            posteriors = json.load(f)
        gl = posteriors.get("galaxy_likelihoods", {})
        add = posteriors.get("additional_galaxies_without_bh_mass", {})
        keys = sorted(set(gl.keys()) | set(add.keys()), key=int)

        for k in keys:
            idx = int(k)
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            M_z = float(row["M"])
            M_z_sigma = float(np.sqrt(row["delta_M_delta_M"]))
            d_L = float(row["luminosity_distance"])
            d_L_sigma = float(
                np.sqrt(row["delta_luminosity_distance_delta_luminosity_distance"])
            )
            z_min, z_max = get_redshift_outer_bounds(
                distance=d_L,
                distance_error=d_L_sigma,
                h_min=H_MIN,
                h_max=H_MAX,
                Omega_m_min=OMEGA_M_MIN,
                Omega_m_max=OMEGA_M_MAX,
                w_0=W_0,
                w_a=W_A,
                sigma_multiplier=2.0,
            )
            z_max = min(z_max, REDSHIFT_UPPER_LIMIT)

            gl_entries = gl.get(k, [])
            add_entries = add.get(k, [])
            lin_pass_positions = {int(e[0]) for e in gl_entries}
            all_positions = sorted(lin_pass_positions | {int(e[0]) for e in add_entries})
            n_events_loaded += 1

            from sklearn.neighbors import BallTree as _BT

            # BUG FOUND BY THIS VERIFIER (self-correction, disclosed): an
            # earlier version of this script had `if not all_positions:
            # continue` here, which ALSO skipped the true-host-retention
            # check below for the 24 "zero_under_both" events (per
            # b5_window_count.json's growth_factor.n_events_zero_under_both
            # = 24). b5_window_count.py's own retention check is
            # UNCONDITIONAL on pos.size (host-mass-only, independent of
            # whether any OTHER candidate passed the cone) -- so the skip
            # was wrong and produced truth_valid_n=2237 instead of 2261.
            # Fixed: only the TOTALS accumulation (candidate-count pass
            # fraction) is skipped for zero-candidate events; retention is
            # always evaluated.
            if all_positions:
                sub_catalog = catalog_df.iloc[all_positions]
                sub_handler = object.__new__(GalaxyCatalogueHandler)
                n_sub = len(sub_catalog)
                pts = np.zeros((n_sub, 3))
                pts[:, 2] = 1.0
                sub_handler.catalog_ball_tree = _BT(pts, metric="euclidean")
                sub_handler.reduced_galaxy_catalog = sub_catalog.reset_index(drop=True)

                for label, geometry, k_val in configs:
                    result = sub_handler.get_possible_hosts_from_ball_tree(
                        phi=0.0,
                        phi_sigma=1e-6,
                        theta=1e-9,
                        theta_sigma=1e-6,
                        M_z=M_z,
                        M_z_sigma=M_z_sigma,
                        z_min=z_min,
                        z_max=z_max,
                        sigma_multiplier=1.5,
                        mass_filter_sigma="symmetric",
                        mass_filter_geometry=geometry,
                        mass_filter_k=k_val,
                    )
                    assert result is not None
                    _, with_mass = result
                    totals[label]["n_all"] += n_sub
                    totals[label]["n_pass"] += len(with_mass)

            # True-host retention: mass-window-only test on the host's own
            # row, independent of sky+redshift cone (matches
            # b5_window_count.py's isolation approach) -- call production
            # function directly with a single-row catalogue = the host.
            if bool(row["in_catalog"]) and int(row["host_galaxy_index"]) >= 0:
                hp = int(row["host_galaxy_index"])
                if hp < cat.n_pruned:
                    truth_valid += 1
                    host_row_df = catalog_df.iloc[[hp]]
                    host_handler = object.__new__(GalaxyCatalogueHandler)
                    hp_pts = np.zeros((1, 3))
                    hp_pts[:, 2] = 1.0
                    host_handler.catalog_ball_tree = _BT(hp_pts, metric="euclidean")
                    host_handler.reduced_galaxy_catalog = host_row_df.reset_index(drop=True)
                    for label, geometry, k_val in configs:
                        # SELF-CORRECTION (this verifier, second bug found):
                        # an earlier version passed artificially wide z
                        # bounds here (z_min=-0.999, z_max=1e6) intending to
                        # bypass "the redshift cone cut" per
                        # b5_window_count.py's own docstring
                        # ("independent of the sky+redshift cone cut").
                        # That conflated TWO different things: (1) the
                        # catalogue redshift_filter_mask (host REDSHIFT +-
                        # REDSHIFT_ERROR vs z_min/z_max) -- correctly
                        # bypassed via the huge REDSHIFT_ERROR already baked
                        # into catalog_df/host_row_df -- and (2) the
                        # (1+z_max)/(1+z_min) terms INSIDE the GW-side mass
                        # window formula itself, which is core physics (the
                        # M_z -> source-frame-mass conversion), not part of
                        # "the cone cut". Using wide z bounds drove gw_lo->0
                        # and gw_hi->huge, making the mass window pass
                        # almost everything and inflating retention (0.9996
                        # / 0.8438 instead of the record's 0.9567 / 0.7890).
                        # Fixed: use the EVENT's real z_min/z_max (computed
                        # above from the GW distance posterior, same as the
                        # totals loop) -- REDSHIFT_ERROR alone (already
                        # 1e12 in catalog_df) suffices to bypass the
                        # catalogue-side cone/redshift filter.
                        res = host_handler.get_possible_hosts_from_ball_tree(
                            phi=0.0,
                            phi_sigma=1e-6,
                            theta=1e-9,
                            theta_sigma=1e-6,
                            M_z=M_z,
                            M_z_sigma=M_z_sigma,
                            z_min=z_min,
                            z_max=z_max,
                            sigma_multiplier=1.5,
                            mass_filter_sigma="symmetric",
                            mass_filter_geometry=geometry,
                            mass_filter_k=k_val,
                        )
                        assert res is not None
                        _, with_mass_host = res
                        if len(with_mass_host) == 1:
                            truth_pass[label] += 1

    pass_fraction = {
        label: (totals[label]["n_pass"] / totals[label]["n_all"] if totals[label]["n_all"] else None)
        for label, _, _ in configs
    }
    retention = {
        label: (truth_pass[label] / truth_valid if truth_valid else None) for label, _, _ in configs
    }
    return {
        "catalogue_md5": catalogue_md5,
        "n_events_loaded": n_events_loaded,
        "totals": totals,
        "pass_fraction": pass_fraction,
        "truth_valid_n": truth_valid,
        "retention": retention,
    }


def main() -> None:
    print("=== Check A: byte-identity (production code vs independent old-formula transcription) ===")
    a = check_a_byte_identity()
    print(json.dumps(a, indent=2, default=str))

    print("\n=== Check B: fleet-level pass fraction / true-host retention (production code) ===")
    b = check_b_fleet_numbers()
    print(json.dumps(b, indent=2, default=str))

    out = {"check_a_byte_identity": a, "check_b_fleet_numbers": b}
    out_path = Path(__file__).resolve().parent / "item7_rederive_output.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
