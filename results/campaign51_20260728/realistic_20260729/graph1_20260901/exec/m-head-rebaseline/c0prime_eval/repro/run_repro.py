"""Minimal local repro: catalogue_leg_1d_mass_aware off vs on, on a handful of
real iiib candidate-bearing (with-BH-mass) events, to test whether the
_get_or_build_grid shared cache (simulation_detection_probability.py:1689)
leaks state across the no-BH/with-BH legs.

Scratch script only. Never edits darksiren_emri/. Reads locally-pinned CRB +
catalogue (md5-verified against the C0-prime gate's own pins). Matches the
C0-prime sbatch's CLI flags exactly (iiib venue, h=0.730, seed 777021),
except catalogue_leg_1d_mass_aware is set explicitly (off, then on) instead
of left at "auto", and the event set is truncated to a handful of rows for
speed.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO_ROOT = "/home/jasper/Repositories/darksiren-emri"
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from darksiren_emri.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN  # noqa: E402
from darksiren_emri.cosmological_model import Model1CrossCheck  # noqa: E402
from darksiren_emri.galaxy_catalogue.handler import GalaxyCatalogueHandler  # noqa: E402
from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics  # noqa: E402

SEED = 777021
H_VALUE = 0.730
# Top with-BH-delta events from the (mismatched, mz_sel/eff) wave-3 blind
# comparand -- events with known BH-mass candidates in their ball (the
# guard at bayesian_statistics.py:6214/6224 requires len(results_with_bh_mass)
# > 0 for a nonzero L_cat_with_bh_mass at all).
EVENT_IDS = [46, 744, 231, 1061, 1317]

COMMON_KWARGS = dict(
    num_workers=2,
    pdet_dl_bins=60,
    pdet_mass_bins=40,
    pdet_estimator="local_linear",
    pdet_z_resolved=True,
    fisher_cond_threshold=1e16,
    host_z_kernel="volume_deconv",
    host_mass_kernel="auto",
    normalization_mode="absolute_marginal",
    selection_in_completion_numerator="fused",
    catalogue_mass_overlap="production",
    catalogue_mass_error_scale=1.0,
    completion_b_scale="derived",
    eddington_m="on",
    sigma4d_mass_kernel="point",
    completion_event_measure="ratio",
    catalogue_global_selection="phi",
    mass_filter_geometry="linear",
    mass_filter_k=1.5,
    theta_b=0.0,
    theta_s=1.0,
    theta_sites="all",
    catalogue_numerator_survival_2d="off",
    catalogue_numerator_survival_2d_center="unset",
    base_seed=SEED,
)


def build_common_objects():
    rng = np.random.default_rng(SEED)
    cosmological_model = Model1CrossCheck(rng=rng)
    galaxy_catalog = GalaxyCatalogueHandler(
        M_min=M_SOURCE_FRAME_MIN,
        M_max=M_SOURCE_FRAME_MAX,
        z_max=cosmological_model.max_redshift,
        observed_catalogue_path=None,
    )
    return cosmological_model, galaxy_catalog


def run_one(flag_value: str, clear_grid_cache_between_legs: bool = False) -> pd.DataFrame:
    """Run evaluate() truncated to EVENT_IDS with catalogue_leg_1d_mass_aware=flag_value.

    Returns the per-event diagnostic rows (event_likelihoods.csv) for h=0.730.
    """
    for sub in ("posteriors", "posteriors_with_bh_mass", "diagnostics"):
        shutil.rmtree(f"simulations/{sub}", ignore_errors=True)
        os.makedirs(f"simulations/{sub}", exist_ok=True)

    cosmological_model, galaxy_catalog = build_common_objects()
    bs = BayesianStatistics()
    full = bs.cramer_rao_bounds
    bs.cramer_rao_bounds = full.loc[full.index.intersection(EVENT_IDS)]
    print(
        f"[flag={flag_value}] truncated cramer_rao_bounds to "
        f"{len(bs.cramer_rao_bounds)}/{len(full)} rows, index={list(bs.cramer_rao_bounds.index)}"
    )

    if clear_grid_cache_between_legs:
        # Bisect (a): force the shared per-h grid to be rebuilt from scratch
        # right before evaluate() runs, so no pre-existing cache state (from
        # a prior call in this same process) can be reused.
        import darksiren_emri.bayesian_inference.simulation_detection_probability as sdp

        orig_init = sdp.SimulationDetectionProbability.__init__

        def _patched_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            self._shared_grid = None
            self._grid_cache = {}

        sdp.SimulationDetectionProbability.__init__ = _patched_init

    bs.evaluate(
        galaxy_catalog,
        cosmological_model,
        H_VALUE,
        catalogue_leg_1d_mass_aware=flag_value,
        **COMMON_KWARGS,
    )

    df = pd.read_csv("simulations/diagnostics/event_likelihoods.csv")
    df = df[df["h"].round(3) == H_VALUE].set_index("event_idx")
    return df


if __name__ == "__main__":
    print("=== RUN 1: catalogue_leg_1d_mass_aware='off' ===")
    off_df = run_one("off")
    print(off_df[["L_cat_no_bh", "L_cat_with_bh", "combined_no_bh", "combined_with_bh",
                   "num_log_term_no_bh", "num_log_term_with_bh"]])

    print("\n=== RUN 2: catalogue_leg_1d_mass_aware='on' ===")
    on_df = run_one("on")
    print(on_df[["L_cat_no_bh", "L_cat_with_bh", "combined_no_bh", "combined_with_bh",
                  "num_log_term_no_bh", "num_log_term_with_bh"]])

    print("\n=== DELTAS (on - off) ===")
    for col in ["L_cat_no_bh", "L_cat_with_bh", "combined_no_bh", "combined_with_bh",
                "num_log_term_no_bh", "num_log_term_with_bh"]:
        d = (on_df[col] - off_df[col]).abs()
        print(f"{col}: max_abs={d.max():.6e}  nonzero_events={int((d > 1e-12).sum())}/{len(d)}")

    print("\n=== RUN 3 (bisect a): 'on', with the shared grid cache force-rebuilt fresh ===")
    on_df_cachefresh = run_one("on", clear_grid_cache_between_legs=True)
    print("\n=== DELTAS (on_cachefresh - on) -- should be exactly 0 if cache state is inert ===")
    for col in ["L_cat_no_bh", "L_cat_with_bh", "combined_no_bh", "combined_with_bh",
                "num_log_term_no_bh", "num_log_term_with_bh"]:
        d = (on_df_cachefresh[col] - on_df[col]).abs()
        print(f"{col}: max_abs={d.max():.6e}")

    summary = {
        "event_ids": EVENT_IDS,
        "off_L_cat_with_bh": off_df["L_cat_with_bh"].to_dict(),
        "on_L_cat_with_bh": on_df["L_cat_with_bh"].to_dict(),
        "on_cachefresh_L_cat_with_bh": on_df_cachefresh["L_cat_with_bh"].to_dict(),
        "off_L_cat_no_bh": off_df["L_cat_no_bh"].to_dict(),
        "on_L_cat_no_bh": on_df["L_cat_no_bh"].to_dict(),
    }
    with open("repro_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote repro_summary.json")
