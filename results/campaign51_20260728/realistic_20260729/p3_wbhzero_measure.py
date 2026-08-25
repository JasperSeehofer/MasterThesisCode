r"""[P3-WBHZERO] measure-first counterfactual driver: the mass-filter sigma-window
(asymmetric vs symmetric).

Registered in ``PREREGISTRATION_P3_WBHZERO_MEASURE_20260825.md`` (binding base text + the
PA-WBZ-1 pre-execution review amendments, which supersede the base text where they conflict --
read the amendment block IN FULL before touching this file). Template/precedent: ``p3_2d_fleet.py``
(the closest structural cousin -- a single-h, two-arm, paired-draw fleet driver built on the same
``run_mirror_seed_inprocess`` direct-call pattern as ``p3_b0_identity_test._run_arm_seed``, reused
here by import as ``o5``) and the preserved Gate-B zero-compute reconstruction
(``gate_b_20260730/wbhzero_gate_b_scripts/counterfactual_symmetric.py``, ported not reinvented for
the WZ-P stage's per-event geometry).

**Mechanism (Gate-B verified, DEFECT candidate-confirmed):** ``handler.py``
``get_possible_hosts_from_ball_tree`` applies ``sigma_multiplier`` (1.5) to the GW mass
uncertainty but not to the galaxy's own ``BH_MASS_ERROR`` -- an asymmetric +/-1.5sigma-vs-+/-1sigma
eligibility window that empties non-empty z-passed candidate balls. The landed
``mass_filter_sigma`` flag (commit ``9c948ea0``, single read/validate site,
``handler.py:558-690``) is the measure-first instrument: ``"asymmetric"`` (default) is
byte-identical to the pre-flag path; ``"symmetric"`` scales ``BH_MASS_ERROR`` by
``sigma_multiplier`` on both sides (the counterfactual).

**Arms, venue, instruments (prereg S2/PA-WBZ-1):**

- **WZA0** (PA-WBZ-1 F4, GATE BIT-A's replacement): seed 900101 ONLY, run under the BANKED
  resolved-flag set read AT RUNTIME from ``p3_b0_work/bc_900101_meta.json``
  (``catalogue_numerator_survival="off"``, ``catalogue_global_selection="phi"``,
  ``mass_filter_sigma`` default), single-h (0.73), ``h_bounds=(0.50,0.86)`` -- compared against the
  banked ``bc_900101`` ``event_likelihoods.csv`` h=0.73 row-slice. PASS = bit-identical or
  ``<=1e-12`` relative (the documented 9.1e-15 CSV round-trip noise class).
- **WZ-A (asymmetric)** / **WZ-S (symmetric)**: the b0i 1D mirror venue fleet (seeds
  900101-900112), ONE galaxy-catalogue object + ONE events draw reused for BOTH arms per seed
  (the pairing rule, F9(i)) -- ``mass_filter_sigma="asymmetric"``/``"symmetric"``, all other flags
  at the ADOPTED PRODUCTION DEFAULTS (``catalogue_numerator_survival="auto"``,
  ``catalogue_global_selection="auto"``, both resolving to ``"phi"`` under
  ``normalization_mode="absolute_marginal"`` per the row #195 twin adoption -- left un-overridden
  here so the flag resolution tracks whatever production's own default is, never pinned to a
  pre-adoption value). Single-h (0.73), ``h_bounds=(0.50,0.86)`` (the PA-CA-10 pin, carried per
  ``p3_2d_fleet.py`` F16). Per-event ``n_cand_nomass``/``n_pass_mass_filter`` per arm via a
  HANDLER-LEVEL RECOUNT (F9(ii)): after each arm, for each event, re-call the REAL
  ``GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree`` (the production method, never
  reimplemented) on the SAME catalogue object with that arm's ``mass_filter_sigma``, using the
  call inputs reconstructed from the seed's own ``prepared_cramer_rao_bounds.csv`` -- the SAME
  reconstruction ``counterfactual_symmetric.py``'s ``analyze()`` already uses to build a
  ``Detection`` + z-window per row (``get_redshift_outer_bounds`` at the pinned h-window,
  ``Z_CAP=1.5``).
- **WZ-P (structural control)**: zero-compute. ``counterfactual_symmetric.py``'s ``analyze()``
  geometry (ball-tree query + z filter + asymmetric/symmetric mass filter, duplicated by
  necessity -- it does NOT call the handler method, that is the point of an independent structural
  control) PORTED to cover ALL 12 seeds, reading each seed's WZ-A ``prepared_cramer_rao_bounds.csv``
  from the FRESH fleet artifacts (the preserved Gate-B JSON only covered a subset and lacked
  900111/900112). Emits per-event predicted retention (``n_ball``, ``n_no_bh``, ``n_asym``,
  ``n_sym``) for every event of every seed.

**Gates (S3 + PA-WBZ-1 F4/F5):**

- **GATE WZA0** (replaces GATE BIT-A, F4): see WZA0 above.
- **GATE CF-X** (amended, F5): WZ-S's per-event REALIZED structural retention (the handler-level
  recount's ``n_pass_mass_filter``) must match the WZ-P zero-compute PREDICTION
  (``n_sym``) EXACTLY per event, PLUS the monotonicity invariant
  ``n_pass_mass_filter(sym) >= n_pass_mass_filter(asym)`` (a theorem for ``sigma_multiplier>=1``)
  checked on EVERY event of every seed (using the WZ-A/WZ-S handler-level recounts, not WZ-P).
- **Catalogue pin:** the pinned reduced-catalogue checksum, verified per task (2026-08-20 dataset-
  pinning rule); STOP on mismatch.

**Statistics (F6, PINNED):** from each arm's ``event_likelihoods.csv`` at h=0.73, over the paired
live set ``E = {events with combined_no_bh + combined_with_bh > 0 in BOTH arms}``: primary
``T_s = sum_{e in E} ln(combined_no_bh + combined_with_bh)``; secondary
``w_bar_s = mean_{e in E} combined_with_bh/(combined_no_bh + combined_with_bh)``; per-arm
catalogue-leg zero rate (``L_cat_with_bh==0 & L_cat_no_bh>0``, over ALL rows, NOT the combined
columns which completion keeps nonzero). ``Delta_s = statistic(WZ-S) - statistic(WZ-A)`` per seed;
pooled ``Delta_bar +/- SEM`` over the 12 pairs; band = ``3*SEM_Delta``; POWER materiality scales
frozen pre-run: ``M_T=0.5``, ``M_w=0.004`` -- if ``3*SEM_Delta > M`` for a statistic, its verdict is
UNDERPOWERED, not immaterial. **This driver banks numbers and gate verdicts only -- NO verdict-map
interpretation** (that is orchestrator/author territory, row #198 binding-default).

**HARD CONSTRAINTS (mirrors o5/p3_2d_fleet.py):**

1. Never end a turn to wait on an untracked process -- every ``evaluate()`` call below is
   synchronous/blocking (``run_mirror_seed_inprocess``).
2. Seeds run SEQUENTIALLY within one invocation -- no subprocess/process-pool fan-out (same
   ``run_mirror_seed_inprocess`` module-state-monkeypatch constraint the b0/b0i/b0i2d drivers
   document).
3. A22 stamp WRITTEN before every ``evaluate()`` call, INCLUDING ``mass_filter_sigma``, git commit,
   tree-dirty flag, catalogue checksum, and environment provenance (numpy/sklearn/pandas versions,
   F5).
4. ``<subdir>_meta.json`` existing is REUSE, never silent re-run (PA-CA-11 out-root guard,
   disclosed on every reuse).
5. This file is BUILT and smoke-tested on zero-compute/CLI-parse paths only by the authoring task
   -- no arm is run by that task.

Usage:
    uv run python results/campaign51_20260728/realistic_20260729/p3_wbhzero_measure.py \
        --stage wza0
    uv run python .../p3_wbhzero_measure.py --stage fleet
    uv run python .../p3_wbhzero_measure.py --stage fleet --seeds 900101,900102
    uv run python .../p3_wbhzero_measure.py --stage wzp
    uv run python .../p3_wbhzero_measure.py --stage readout
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))  # o5 (p3_b0_identity_test) is a sibling script, not a package

import p3_b0_identity_test as o5  # noqa: E402

from darksiren_emri.constants import (  # noqa: E402
    HOST_DRAW_Z_MAX,
    M_SOURCE_FRAME_MAX,
    M_SOURCE_FRAME_MIN,
)
from darksiren_emri.datamodels.detection import Detection  # noqa: E402
from darksiren_emri.galaxy_catalogue.handler import (  # noqa: E402
    GalaxyCatalogueHandler,
    InternalCatalogColumns,
    _polar_to_cartesian,
)
from darksiren_emri.physical_relations import get_redshift_outer_bounds  # noqa: E402
from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402
from darksiren_emri.validation.correspondence_1d import H_TRUE  # noqa: E402

REGISTRATION_SECTION: str = (
    "results/campaign51_20260728/realistic_20260729/"
    "PREREGISTRATION_P3_WBHZERO_MEASURE_20260825.md (2026-08-25, PA-WBZ-1 amended)"
)

H_GEN: float = H_TRUE  # 0.73, the single-h read (prereg S2)
H_BOUNDS: tuple[float, float] = (0.50, 0.86)  # PA-CA-10 pin, carried per p3_2d_fleet.py F16
VENUE: str = "b0i"

# Production ball-tree query sigma_multiplier (bayesian_statistics.py:4662, cited by
# counterfactual_symmetric.py -- reused, not re-derived).
SIGMA_MULTIPLIER: float = 1.5
# Model1CrossCheck.max_redshift, the generator's own redshift cap (counterfactual_symmetric.py
# Z_CAP precedent).
Z_CAP: float = 1.5

ARM_MASS_FILTER: dict[str, str] = {"wza": "asymmetric", "wzs": "symmetric"}

FLEET_SEEDS_DEFAULT: tuple[int, ...] = tuple(range(900101, 900113))  # 12, prereg S2
WZA0_SEED: int = 900101
BANKED_BC_META_PATH: Path = THIS_DIR / "p3_b0_work" / "bc_900101_meta.json"
BANKED_BC_CSV_PATH: Path = (
    THIS_DIR
    / "p3_b0_work"
    / "bc_900101_work"
    / "seed900101"
    / "simulations"
    / "diagnostics"
    / "event_likelihoods.csv"
)

# GATE WZA0 (PA-WBZ-1 F4): bit-identical or <=1e-12 relative (the documented 9.1e-15 CSV
# round-trip noise class, GATE R-B0/R-P3/D6/R4 precedent).
GATE_WZA0_RTOL: float = 1.0e-12
# Every numeric column the venue's event_likelihoods.csv carries, keys excluded.
WZA0_COMPARE_COLUMNS: tuple[str, ...] = (
    "w_G",
    "w_G_legacy",
    "w_tilde_G",
    "alpha_G_phi",
    "r_Malm",
    "D_tilde_phi",
    "L_cat_no_bh",
    "L_cat_with_bh",
    "B_num",
    "B_num_wbh",
    "g_frac",
    "L_comp",
    "combined_no_bh",
    "combined_with_bh",
)

# F6 (PINNED statistics) -- POWER materiality scales, frozen pre-run (PA-WBZ-1 F7).
M_T: float = 0.5
M_W: float = 0.004

OUT_ROOT_DEFAULT: Path = THIS_DIR / "wbhzero_work"


# ── environment provenance (F5) ───────────────────────────────────────────────


def _env_provenance() -> dict[str, str]:
    import pandas
    import sklearn

    return {
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pandas.__version__,
    }


def _a22_stamp_wbz(mass_filter_sigma: str) -> dict[str, Any]:
    """A22 = git commit + dirty flag (o5._a22_stamp, reused) + mass_filter_sigma + catalogue
    checksum + environment provenance, WRITTEN before any ``evaluate()`` call (F5/F9(i)).
    """
    git_stamp = o5._a22_stamp()
    return {
        **git_stamp,
        "mass_filter_sigma": mass_filter_sigma,
        "catalogue_pin_ok": c1d.check_reduced_catalogue_pin(),
        "environment_provenance": _env_provenance(),
    }


# ── shared b0i realization (the pairing rule, F9(i)) ──────────────────────────


def _build_b0i_realization(
    seed: int, catalogue_root: Path
) -> tuple[GalaxyCatalogueHandler, pd.DataFrame]:
    """ONE galaxy-catalogue object + ONE events draw for a seed, venue b0i (PA-2 host mode
    ``catalogue_selected``). Mirrors ``p3_b0_identity_test._run_arm_seed``'s b0i-venue
    construction exactly -- reused (not reimplemented) so the draw law is identical to the
    adjudicated b0i identity-test venue.
    """
    sigma_z_scale, area_scale = c1d.ARM_SPECS[VENUE]
    assert c1d.ARM_HOST_MODE[VENUE] == "catalogue_selected", (
        "interface assumption violated: c1d.ARM_HOST_MODE['b0i'] != 'catalogue_selected' -- "
        "the venue registry changed since this driver was written -- STOP (A21)"
    )
    assert c1d.ARM_SELECTION_CELL[VENUE] == "fused", (
        "interface assumption violated: c1d.ARM_SELECTION_CELL['b0i'] != 'fused' -- STOP (A21)"
    )
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=sigma_z_scale, area_scale=area_scale)
    gen = c1d.MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        catalogue_root, seed, sigma_z_scale=sigma_z_scale
    )
    c1d._verify_rate_weight_parity()  # PA-2 runtime rate-weight parity gate
    completeness_obj, phi_survival_table = c1d.build_bsel_selection_objects(h_true=H_GEN)
    events = gen.draw_realization(
        seed,
        host_pool=host_pool,
        host_mode="catalogue_selected",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
    )
    return handler, events


# ── handler-level per-event recount (F9(ii)) ──────────────────────────────────


def _event_z_window(det: Detection) -> tuple[float, float]:
    """The same z-window construction ``counterfactual_symmetric.py``'s ``analyze()`` uses:
    ``get_redshift_outer_bounds`` at the pinned h-window, capped at :data:`Z_CAP`.
    """
    z_min, z_max = get_redshift_outer_bounds(
        distance=det.d_L,
        distance_error=det.d_L_uncertainty,
        h_min=H_BOUNDS[0],
        h_max=H_BOUNDS[1],
        Omega_m_min=0.04,
        Omega_m_max=0.5,
        sigma_multiplier=2.0,  # NB: ignored inside get_redshift_outer_bounds (hardcoded 3 sigma)
    )
    return z_min, min(z_max, Z_CAP)


def _hostcounts_recount(
    handler: GalaxyCatalogueHandler, crb_csv: Path, mass_filter_sigma: str
) -> pd.DataFrame:
    """Handler-level recount (F9(ii)): for every event in ``crb_csv``, re-call the REAL
    ``GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree`` (production method, never
    reimplemented) on the SAME catalogue object with ``mass_filter_sigma``, call inputs
    reconstructed from the CRB row exactly as ``counterfactual_symmetric.py``'s ``analyze()``
    does (``Detection`` + :func:`_event_z_window`).
    """
    crb = pd.read_csv(crb_csv)
    rows: list[dict[str, Any]] = []
    for idx in crb.index:
        det = Detection(crb.loc[idx])
        z_min, z_max = _event_z_window(det)
        result = handler.get_possible_hosts_from_ball_tree(
            phi=det.phi,
            phi_sigma=det.phi_error,
            theta=det.theta,
            theta_sigma=det.theta_error,
            M_z=det.M,
            M_z_sigma=det.M_uncertainty,
            z_min=z_min,
            z_max=z_max,
            sigma_multiplier=SIGMA_MULTIPLIER,  # type: ignore[arg-type]
            cov_theta_phi=det.theta_phi_covariance,
            mass_filter_sigma=mass_filter_sigma,
        )
        if result is None:
            n_nomass, n_pass = 0, 0
        else:
            without_bh, with_bh = result
            n_nomass, n_pass = len(without_bh), len(with_bh)
        rows.append(
            {
                "event_idx": int(idx),
                "n_cand_nomass": n_nomass,
                "n_pass_mass_filter": n_pass,
                "z_min": z_min,
                "z_max": z_max,
            }
        )
    return pd.DataFrame(rows)


# ── WZ-P: the zero-compute structural control (ported counterfactual_symmetric.py geometry) ──


def _wzp_analyze(handler: GalaxyCatalogueHandler, det: Detection) -> dict[str, Any]:
    """Port of ``counterfactual_symmetric.py``'s ``analyze()`` -- the INDEPENDENT (duplicated by
    necessity) ball-tree + z-filter + asymmetric/symmetric mass-filter reconstruction, NOT a call
    to the handler method (that independence is the point of the WZ-P structural control: GATE
    CF-X compares this prediction against the handler-level recount's REALIZED result).
    """
    z_min, z_max = _event_z_window(det)

    query_point = _polar_to_cartesian(np.array([det.theta]), np.array([det.phi]))
    sigma_matrix = np.array(
        [
            [det.phi_error**2, det.theta_phi_covariance],
            [det.theta_phi_covariance, det.theta_error**2],
        ]
    )
    jacobian = np.diag([abs(np.sin(det.theta)), 1.0])
    sigma_scaled = jacobian @ sigma_matrix @ jacobian.T
    lambda_max = float(np.linalg.eigvalsh(sigma_scaled).max())
    radius = float(SIGMA_MULTIPLIER * np.sqrt(max(lambda_max, 0.0)))
    indices = handler.catalog_ball_tree.query_radius(query_point, r=radius)[0]
    cand = handler.reduced_galaxy_catalog.iloc[indices]

    z = cand[InternalCatalogColumns.REDSHIFT]
    ze = cand[InternalCatalogColumns.REDSHIFT_ERROR]
    zmask = (z_min <= z + ze) & (z_max >= z - ze)
    nb = cand[zmask]  # candidate_hosts_without_bh_mass

    m = nb[InternalCatalogColumns.BH_MASS]
    me = nb[InternalCatalogColumns.BH_MASS_ERROR]
    lo = (det.M - det.M_uncertainty * SIGMA_MULTIPLIER) / (1 + z_max)
    hi = (det.M + det.M_uncertainty * SIGMA_MULTIPLIER) / (1 + z_min)

    asym = (lo <= m + me) & (m - me <= hi)  # production: galaxy +/- 1 sigma
    sym = (lo <= m + SIGMA_MULTIPLIER * me) & (m - SIGMA_MULTIPLIER * me <= hi)  # counterfactual

    return {
        "n_ball": int(len(cand)),
        "n_no_bh": int(len(nb)),
        "n_asym": int(asym.sum()),
        "n_sym": int(sym.sum()),
        "z_min": float(z_min),
        "z_max": float(z_max),
    }


def _wzp_handler() -> GalaxyCatalogueHandler:
    """The zero-compute WZ-P instrument's own handler build (counterfactual_symmetric.py
    ``main()`` precedent) -- a fresh ``GalaxyCatalogueHandler`` load, independent of any
    mirror-universe generator object the fleet stage built.
    """
    return GalaxyCatalogueHandler(
        M_min=M_SOURCE_FRAME_MIN, M_max=M_SOURCE_FRAME_MAX, z_max=HOST_DRAW_Z_MAX
    )


# ── stage: wza0 (GATE WZA0, PA-WBZ-1 F4) ──────────────────────────────────────


def stage_wza0(out_root: Path) -> dict[str, Any]:
    """GATE WZA0 (replaces GATE BIT-A, F4): seed 900101 under the BANKED resolved-flag set read
    AT RUNTIME from ``p3_b0_work/bc_900101_meta.json``, single-h (0.73), compared against the
    banked ``bc_900101`` ``event_likelihoods.csv`` h=0.73 row-slice.
    """
    if not BANKED_BC_META_PATH.is_file():
        raise SystemExit(f"REFUSED: banked meta not found: {BANKED_BC_META_PATH}")
    banked_meta = json.loads(BANKED_BC_META_PATH.read_text())
    catalogue_numerator_survival = banked_meta["catalogue_numerator_survival"]
    catalogue_global_selection = banked_meta["catalogue_global_selection"]
    # bc_900101_meta.json predates the mass_filter_sigma flag (commit 9c948ea0) -- "mass_filter_
    # sigma default" per the task spec means the function's own default, "asymmetric".
    mass_filter_sigma = "asymmetric"

    subdir = "wza0_900101"
    meta_path = out_root / f"{subdir}_meta.json"
    if meta_path.is_file():
        print(f"seed {WZA0_SEED} (wza0): REUSING existing {subdir}_meta.json (disclosed)")
    else:
        work_root = out_root / f"{subdir}_work"
        work_root.mkdir(parents=True, exist_ok=True)
        handler, events = _build_b0i_realization(WZA0_SEED, work_root / "catalogue")

        stamp = _a22_stamp_wbz(mass_filter_sigma)  # written before evaluate()
        t0 = time.time()
        diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
            work_root / f"seed{WZA0_SEED}",
            events,
            WZA0_SEED,
            galaxy_catalog=handler,
            h_values=(H_GEN,),
            catalogue_numerator_survival=catalogue_numerator_survival,
            catalogue_global_selection=catalogue_global_selection,
            mass_filter_sigma=mass_filter_sigma,
            h_bounds=H_BOUNDS,
        )
        wall_time_s = time.time() - t0
        meta: dict[str, Any] = {
            "subdir": subdir,
            "seed": WZA0_SEED,
            "venue": VENUE,
            "catalogue_numerator_survival": catalogue_numerator_survival,
            "catalogue_global_selection": catalogue_global_selection,
            "mass_filter_sigma": mass_filter_sigma,
            "banked_meta_source": str(BANKED_BC_META_PATH),
            "work_root": str(work_root),
            "diagnostics_csv": str(diag_csv),
            "wall_time_s": wall_time_s,
            "elapsed_evaluate_s": elapsed,
            "a22_stamp": stamp,
            "git_commit": c1d._git_commit(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

    meta = json.loads(meta_path.read_text())
    fresh = pd.read_csv(o5._meta_csv(meta))
    gate = o5._compare_columns(fresh, BANKED_BC_CSV_PATH, WZA0_COMPARE_COLUMNS, GATE_WZA0_RTOL)
    gate["gate"] = "GATE_WZA0"
    gate["reference"] = f"{REGISTRATION_SECTION}, PA-WBZ-1 F4"
    gate["banked_flags"] = {
        "catalogue_numerator_survival": catalogue_numerator_survival,
        "catalogue_global_selection": catalogue_global_selection,
        "mass_filter_sigma": mass_filter_sigma,
    }
    out_path = out_root / "wza0_gate.json"
    out_path.write_text(json.dumps(gate, indent=2))
    print(json.dumps({k: v for k, v in gate.items() if k != "per_column"}, indent=2))
    print(f"wrote {out_path}")
    return gate


# ── stage: fleet ───────────────────────────────────────────────────────────────


def _run_wbz_arm(
    seed: int,
    arm: str,
    handler: GalaxyCatalogueHandler,
    events: pd.DataFrame,
    out_root: Path,
) -> dict[str, Any]:
    """One (arm, seed) WZ fleet task -- ``arm in {"wza","wzs"}`` selects ``mass_filter_sigma``.
    Reuses the CALLER-supplied ``handler``/``events`` (the pairing rule, F9(i)) -- does NOT build
    a fresh catalogue/draw.
    """
    mass_filter_sigma = ARM_MASS_FILTER[arm]
    subdir = f"{arm}_{seed}"
    meta_path = out_root / f"{subdir}_meta.json"
    if meta_path.is_file():
        print(f"seed {seed} ({arm}): REUSING existing {subdir}_meta.json (disclosed, PA-CA-11)")
        reused_meta: dict[str, Any] = json.loads(meta_path.read_text())
        return reused_meta

    work_root = out_root / f"{subdir}_work"
    work_root.mkdir(parents=True, exist_ok=True)

    stamp = _a22_stamp_wbz(mass_filter_sigma)  # written before evaluate()
    t0 = time.time()
    diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
        work_root / f"seed{seed}",
        events,
        seed,
        galaxy_catalog=handler,
        h_values=(H_GEN,),
        # catalogue_numerator_survival/catalogue_global_selection left at "auto" (adopted
        # production defaults, row #195 twin) -- the invariant (prereg S5) is that BOTH arms run
        # at production defaults, not a pinned pre-adoption value.
        mass_filter_sigma=mass_filter_sigma,
        h_bounds=H_BOUNDS,
    )
    wall_time_s = time.time() - t0

    crb_csv = work_root / f"seed{seed}" / "simulations" / "prepared_cramer_rao_bounds.csv"
    hostcounts = _hostcounts_recount(handler, crb_csv, mass_filter_sigma)
    hostcounts_path = out_root / f"hostcounts_{arm}_{seed}.csv"
    hostcounts.to_csv(hostcounts_path, index=False)

    meta: dict[str, Any] = {
        "subdir": subdir,
        "seed": seed,
        "arm": arm,
        "venue": VENUE,
        "mass_filter_sigma": mass_filter_sigma,
        "work_root": str(work_root),
        "crb_csv": str(crb_csv),
        "diagnostics_csv": str(diag_csv),
        "hostcounts_csv": str(hostcounts_path),
        "wall_time_s": wall_time_s,
        "elapsed_evaluate_s": elapsed,
        "n_events": int(events.shape[0]),
        "a22_stamp": stamp,
        "git_commit": c1d._git_commit(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "diagnostics_csv"}, indent=2))
    return meta


def stage_fleet(out_root: Path, seeds: list[int] | None = None) -> dict[str, Any]:
    """Both arms' fleet (12 seeds default, venue b0i), sequential, idempotent-skip on existing
    meta. ONE galaxy-catalogue object + ONE events draw is built per seed and reused for BOTH
    arms (the pairing rule, F9(i)).
    """
    seed_list = seeds if seeds is not None else list(FLEET_SEEDS_DEFAULT)
    summary: dict[str, Any] = {"seeds": seed_list, "wza": {}, "wzs": {}}
    for seed in seed_list:
        wza_meta_path = out_root / f"wza_{seed}_meta.json"
        wzs_meta_path = out_root / f"wzs_{seed}_meta.json"
        if wza_meta_path.is_file() and wzs_meta_path.is_file():
            print(f"seed {seed}: REUSING existing wza/wzs meta (disclosed, PA-CA-11)")
            summary["wza"][seed] = json.loads(wza_meta_path.read_text())
            summary["wzs"][seed] = json.loads(wzs_meta_path.read_text())
            continue
        catalogue_root = out_root / f"catalogue_{seed}"
        handler, events = _build_b0i_realization(seed, catalogue_root)
        summary["wza"][seed] = _run_wbz_arm(seed, "wza", handler, events, out_root)
        summary["wzs"][seed] = _run_wbz_arm(seed, "wzs", handler, events, out_root)
    print(
        json.dumps(
            {
                "seeds": seed_list,
                "n_wza": len(summary["wza"]),
                "n_wzs": len(summary["wzs"]),
            },
            indent=2,
        )
    )
    return summary


# ── stage: wzp (zero-compute structural control) ──────────────────────────────


def stage_wzp(out_root: Path, seeds: tuple[int, ...] | None = None) -> dict[str, Any]:
    """WZ-P (F5, zero-compute): the ported ``counterfactual_symmetric.py`` reconstruction,
    extended to ALL 12 seeds, reading each seed's WZ-A ``prepared_cramer_rao_bounds.csv`` from the
    FRESH fleet artifacts. Emits per-event predicted retention (``wzp_predictions.csv``).
    """
    seed_list = seeds if seeds is not None else FLEET_SEEDS_DEFAULT
    handler = _wzp_handler()
    env = _env_provenance()

    rows: list[dict[str, Any]] = []
    missing_crb: list[int] = []
    for seed in seed_list:
        wza_meta_path = out_root / f"wza_{seed}_meta.json"
        if not wza_meta_path.is_file():
            missing_crb.append(seed)
            continue
        wza_meta = json.loads(wza_meta_path.read_text())
        crb_path = Path(wza_meta["crb_csv"])
        if not crb_path.is_file():
            missing_crb.append(seed)
            continue
        crb = pd.read_csv(crb_path)
        for idx in crb.index:
            det = Detection(crb.loc[idx])
            r = _wzp_analyze(handler, det)
            rows.append({"seed": seed, "event_idx": int(idx), **r})

    df = pd.DataFrame(rows)
    out_path = out_root / "wzp_predictions.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    meta = {
        "reference": f"{REGISTRATION_SECTION}, PA-WBZ-1 F5",
        "instrument": "ported counterfactual_symmetric.py geometry, all 12 seeds",
        "environment_provenance": env,
        "sigma_multiplier": SIGMA_MULTIPLIER,
        "z_cap": Z_CAP,
        "h_bounds": list(H_BOUNDS),
        "n_seeds_found": len(seed_list) - len(missing_crb),
        "missing_crb_seeds": missing_crb,
        "n_rows": int(len(df)),
        "predictions_csv": str(out_path),
    }
    meta_path = out_root / "wzp_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


# ── stage: readout (gates + statistics, NO verdict interpretation) ────────────


def _gate_cf_x(out_root: Path, seed_list: tuple[int, ...]) -> dict[str, Any]:
    """GATE CF-X (F5): WZ-S realized retention == WZ-P prediction EXACTLY per event, PLUS the
    monotonicity invariant on EVERY event of every seed.
    """
    wzp_csv = out_root / "wzp_predictions.csv"
    if not wzp_csv.is_file():
        return {"pass": False, "reason": f"missing {wzp_csv} -- run --stage wzp first."}
    wzp = pd.read_csv(wzp_csv)

    exact_match_fail: list[dict[str, Any]] = []
    monotonicity_fail: list[dict[str, Any]] = []
    n_events_checked = 0
    for seed in seed_list:
        wza_hc_path = out_root / f"hostcounts_wza_{seed}.csv"
        wzs_hc_path = out_root / f"hostcounts_wzs_{seed}.csv"
        if not (wza_hc_path.is_file() and wzs_hc_path.is_file()):
            return {
                "pass": False,
                "reason": f"missing hostcounts CSV(s) for seed {seed} -- run --stage fleet first.",
            }
        wza_hc = pd.read_csv(wza_hc_path)
        wzs_hc = pd.read_csv(wzs_hc_path)
        wzp_seed = wzp[wzp["seed"] == seed]
        merged = wzs_hc.merge(
            wzp_seed[["event_idx", "n_sym"]], on="event_idx", how="left", validate="one_to_one"
        )
        merged_mono = wza_hc.merge(
            wzs_hc[["event_idx", "n_pass_mass_filter"]],
            on="event_idx",
            suffixes=("_asym", "_sym"),
            how="left",
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            n_events_checked += 1
            realized = row["n_pass_mass_filter"]
            predicted = row["n_sym"]
            if pd.isna(predicted) or int(realized) != int(predicted):
                exact_match_fail.append(
                    {
                        "seed": int(seed),
                        "event_idx": int(row["event_idx"]),
                        "realized_n_pass_sym": None if pd.isna(realized) else int(realized),
                        "predicted_n_sym": None if pd.isna(predicted) else int(predicted),
                    }
                )
        for _, row in merged_mono.iterrows():
            n_asym = row["n_pass_mass_filter_asym"]
            n_sym = row["n_pass_mass_filter_sym"]
            if pd.notna(n_asym) and pd.notna(n_sym) and int(n_sym) < int(n_asym):
                monotonicity_fail.append(
                    {
                        "seed": int(seed),
                        "event_idx": int(row["event_idx"]),
                        "n_pass_asym": int(n_asym),
                        "n_pass_sym": int(n_sym),
                    }
                )

    return {
        "gate": "CF-X",
        "reference": f"{REGISTRATION_SECTION}, PA-WBZ-1 F5",
        "n_events_checked": n_events_checked,
        "exact_match_pass": len(exact_match_fail) == 0,
        "exact_match_failures": exact_match_fail,
        "monotonicity_pass": len(monotonicity_fail) == 0,
        "monotonicity_failures": monotonicity_fail,
        "pass": len(exact_match_fail) == 0 and len(monotonicity_fail) == 0,
    }


def _gate_catalogue_pin(out_root: Path, seed_list: tuple[int, ...]) -> dict[str, Any]:
    """Catalogue pin gate: every fleet task's A22 stamp must record ``catalogue_pin_ok=True``."""
    failures: list[dict[str, Any]] = []
    n_checked = 0
    for arm in ("wza", "wzs"):
        for seed in seed_list:
            meta_path = out_root / f"{arm}_{seed}_meta.json"
            if not meta_path.is_file():
                return {
                    "pass": False,
                    "reason": f"missing {meta_path} -- run --stage fleet first.",
                }
            meta = json.loads(meta_path.read_text())
            n_checked += 1
            if not meta["a22_stamp"]["catalogue_pin_ok"]:
                failures.append({"arm": arm, "seed": int(seed)})
    return {
        "gate": "catalogue-pin",
        "n_checked": n_checked,
        "failures": failures,
        "pass": len(failures) == 0,
    }


def _paired_seed_statistics(out_root: Path, seed: int) -> dict[str, Any]:
    """F6 (PINNED): per-seed T_s/w_bar_s per arm, catalogue-leg zero rate per arm, Delta_T_s,
    Delta_w_s.
    """
    wza_meta = json.loads((out_root / f"wza_{seed}_meta.json").read_text())
    wzs_meta = json.loads((out_root / f"wzs_{seed}_meta.json").read_text())
    at_a = o5._rows_at_h(Path(wza_meta["diagnostics_csv"]), H_GEN)
    at_s = o5._rows_at_h(Path(wzs_meta["diagnostics_csv"]), H_GEN)

    merged = at_a.merge(
        at_s, on="event_idx", suffixes=("_a", "_s"), how="inner", validate="one_to_one"
    )
    sum_a = merged["combined_no_bh_a"].to_numpy(dtype=np.float64) + merged[
        "combined_with_bh_a"
    ].to_numpy(dtype=np.float64)
    sum_s = merged["combined_no_bh_s"].to_numpy(dtype=np.float64) + merged[
        "combined_with_bh_s"
    ].to_numpy(dtype=np.float64)
    live = (sum_a > 0.0) & (sum_s > 0.0)  # E = paired live set

    t_a = float(np.log(sum_a[live]).sum())
    t_s = float(np.log(sum_s[live]).sum())
    wbar_a = float(
        (merged["combined_with_bh_a"].to_numpy(dtype=np.float64)[live] / sum_a[live]).mean()
    )
    wbar_s = float(
        (merged["combined_with_bh_s"].to_numpy(dtype=np.float64)[live] / sum_s[live]).mean()
    )

    def _cat_zero_rate(at: pd.DataFrame) -> float:
        l_cat_wbh = at["L_cat_with_bh"].to_numpy(dtype=np.float64)
        l_cat_nobh = at["L_cat_no_bh"].to_numpy(dtype=np.float64)
        zero_mask = (l_cat_wbh == 0.0) & (l_cat_nobh > 0.0)
        return float(zero_mask.sum() / at.shape[0]) if at.shape[0] else float("nan")

    return {
        "seed": int(seed),
        "n_live": int(live.sum()),
        "n_paired_rows": int(merged.shape[0]),
        "T_wza": t_a,
        "T_wzs": t_s,
        "Delta_T": t_s - t_a,
        "wbar_wza": wbar_a,
        "wbar_wzs": wbar_s,
        "Delta_wbar": wbar_s - wbar_a,
        "catalogue_leg_zero_rate_wza": _cat_zero_rate(at_a),
        "catalogue_leg_zero_rate_wzs": _cat_zero_rate(at_s),
    }


def stage_readout(out_root: Path, seeds: tuple[int, ...] | None = None) -> dict[str, Any]:
    """GATE CF-X + catalogue-pin gate, then the PINNED F6 statistics. Banks ALL numbers + gate
    verdicts -- NO verdict-map interpretation (row #198 binding-default; orchestrator/author
    territory).
    """
    seed_list = tuple(seeds) if seeds is not None else FLEET_SEEDS_DEFAULT

    gate_cf_x = _gate_cf_x(out_root, seed_list)
    gate_pin = _gate_catalogue_pin(out_root, seed_list)

    per_seed = [_paired_seed_statistics(out_root, seed) for seed in seed_list]

    delta_t = np.array([r["Delta_T"] for r in per_seed], dtype=np.float64)
    delta_w = np.array([r["Delta_wbar"] for r in per_seed], dtype=np.float64)
    n = delta_t.size

    def _pooled(vals: npt.NDArray[np.float64]) -> dict[str, Any]:
        mean = float(vals.mean()) if vals.size else None
        sem = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else None
        band = 3.0 * sem if sem is not None else None
        return {"mean": mean, "sem": sem, "band_3sem": band}

    pooled_t = _pooled(delta_t)
    pooled_w = _pooled(delta_w)
    power_t_underpowered = pooled_t["band_3sem"] is not None and pooled_t["band_3sem"] > M_T
    power_w_underpowered = pooled_w["band_3sem"] is not None and pooled_w["band_3sem"] > M_W

    out: dict[str, Any] = {
        "reference": f"{REGISTRATION_SECTION}, PA-WBZ-1 F5/F6/F7",
        "n_seeds": n,
        "seeds": list(seed_list),
        "gate_cf_x": gate_cf_x,
        "gate_catalogue_pin": gate_pin,
        "per_seed": per_seed,
        "Delta_T_per_seed": delta_t.tolist(),
        "Delta_T_pooled": pooled_t,
        "M_T": M_T,
        "Delta_T_POWER_underpowered": power_t_underpowered,
        "Delta_wbar_per_seed": delta_w.tolist(),
        "Delta_wbar_pooled": pooled_w,
        "M_w": M_W,
        "Delta_wbar_POWER_underpowered": power_w_underpowered,
        "note": (
            "NO verdict-map interpretation -- gates and pinned numbers only. Adoption/verdict "
            "mapping returns to the author as a fresh [RULE] (row #198 binding-default)."
        ),
    }
    out_path = out_root / "readout.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "gate_cf_x_pass": gate_cf_x.get("pass"),
                "gate_catalogue_pin_pass": gate_pin.get("pass"),
                "Delta_T_pooled": pooled_t,
                "Delta_T_POWER_underpowered": power_t_underpowered,
                "Delta_wbar_pooled": pooled_w,
                "Delta_wbar_POWER_underpowered": power_w_underpowered,
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("wza0", "fleet", "wzp", "readout"), required=True)
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated seed subset (default: 900101-900112)",
    )
    ap.add_argument(
        "--out-root", type=str, default=str(OUT_ROOT_DEFAULT), help="Root scratch/output directory."
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None

    if args.stage == "wza0":
        stage_wza0(out_root)
        return 0
    if args.stage == "fleet":
        stage_fleet(out_root, seeds)
        return 0
    if args.stage == "wzp":
        stage_wzp(out_root, tuple(seeds) if seeds else None)
        return 0
    stage_readout(out_root, tuple(seeds) if seeds else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
