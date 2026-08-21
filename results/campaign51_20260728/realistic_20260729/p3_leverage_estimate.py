r"""[P3-IMP] arm LEV -- zero-``evaluate()`` axis-leverage instrument.

Registered in ``PREREGISTRATION_P3_TWIN_20260822.md`` §3 ("LEV (leverage
instrument, pre-pilot)") and §5 ("Axis-leverage statement (A17, pre-data)").
Question: BEFORE any ``evaluate()`` call under ``catalogue_numerator_survival
="phi"`` is paid for, is the predicted per-seed |Δ| from the twin cell's
catalogue-numerator survival factor large enough (>= 5x the paired-band
resolution, ``SEM_paired``-scale 0.004) to be worth measuring? If not,
execution STOPS at LEV per §5 ("if the instrument predicts sub-resolvable
leverage, execution STOPS at LEV and the thread returns with a re-design").

**Mechanism (prereg §1):** the catalogue numerator gains a per-host S_bar_phi
factor (the population-marginal proxy). Its first-order effect on the
per-event catalogue-leg contribution to ``combined_no_bh`` is
``share_cat x d/dh[ln S_bar_phi(z_obs; h)]``, where ``share_cat`` is the
catalogue leg's fractional share of the per-event likelihood at ``h_gen`` (the
completion leg, weight ``1 - share_cat``, is untouched by this axis).

**share_cat formula (verified against
``results/prod2d_closure_20260818/decompose_impostor_leg.py:239``, the
"assembly-true beta-convention" share -- the launch task's own formula,
byte-identical)**::

    beta_L_cat = (alpha_G_phi / r_Malm) * L_cat_no_bh
    share_cat  = beta_L_cat / (beta_L_cat + B_num)

read directly from the BANKED ``event_likelihoods.csv`` columns at
``h = H_GEN`` -- no re-run, no reconstruction risk (the exact production
values that produced the banked score).

**S_bar_phi table (NOT ambiguous -- verified against source, so no A21 flag
needed here):** built via
:func:`darksiren_emri.validation.correspondence_1d.build_bsel_selection_objects`,
the B-SEL venue's OWN committed construction (SAME
``SimulationDetectionProbability``/``precompute_phi_marginal_survival`` calls
``run_arm_seed``'s ``bsel``/``bself`` branches use, ``correspondence_1d.py:
893-964``), called once at ``h_true=H_LO`` and once at ``h_true=H_HI``
(``functools.lru_cache``-d, so a re-run within one process pays the detection-
probability-grid cost at most twice, not per event/seed).

**FLAGGED SUBSTITUTION (A21 -- disclosed, not silently resolved).** The launch
task's primary design asks for the REAL candidate galaxies' z_obs (the
catalogue-search hits :func:`~darksiren_emri.galaxy_catalogue.handler.
GalaxyCatalogueHandler.get_possible_hosts_from_ball_tree` would return inside
a real ``evaluate()`` call) averaged per event. That machinery exists
(``galaxy_catalogue/handler.py:558-660``) but reproducing its EXACT per-h
search-radius/z-window/BH-mass-filter convention outside ``evaluate()`` risks
an unregistered reimplementation of production's own candidate-search logic
-- exactly the failure mode the O4 "domain-and-quadrature pairing" incident
this campaign is downstream of was about. This instrument instead uses the
explicitly-permitted fallback: for each of the 200 mirror events in a
DETERMINISTIC redraw of the realization (:func:`_regenerate_events`, an exact
replica of ``run_arm_seed``'s ``bsel`` branch --
``correspondence_1d.py:2739-2753`` -- through the draw step only, NO
``evaluate()`` call), an EFFECTIVE z_obs is taken as
``dist_to_redshift(obs_d_L, h=H_GEN)`` -- the redshift implied by the event's
own observed luminosity distance at the generating h. This is a proxy for
"where in z-space this event's likelihood mass sits", not the true candidate
list.

A SECOND, coupled substitution follows from the first: production's
``evaluate()`` quality-filters the 200 drawn events down to the banked
``n_eff`` (174 for these seeds) via internal SNR/detection logic this
instrument does not reproduce (again, to avoid reimplementing production
filtering outside ``evaluate()``) -- so the regenerated 200-event set's
per-event effective z_obs values CANNOT be reliably paired index-for-index
against the banked CSV's (post-filter, renumbered) ``event_idx``. This
instrument therefore reports the predicted per-seed shift as the PRODUCT OF
MEANS, not a true paired per-event mean::

    predicted_delta(seed) ~ mean_i[share_cat_i] (banked, 174 events)
                            x mean_j[dln_Sphi_j] (regenerated, 200 events)

rather than ``mean[share_cat_i x dln_Sphi_i]`` over a common index i. This
under- or over-states the true paired quantity to the extent share_cat and
the S_bar_phi log-slope are correlated across events (both are functions of
z, so some correlation is expected) -- an order-of-magnitude leverage
estimate, which is exactly LEV's registered purpose (the axis-leverage gate
compares against a factor of 5, not a tight number).

Costing (A6/A17, prereg §8): < 15 min, < 4 GB (no ``evaluate()`` calls; the
dominant cost is two ``SimulationDetectionProbability`` grid builds, cached).

Usage:
    uv run python results/campaign51_20260728/realistic_20260729/p3_leverage_estimate.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from darksiren_emri.physical_relations import dist_to_redshift
from darksiren_emri.validation import correspondence_1d as c1d

REPO_ROOT = Path(__file__).resolve().parents[3]
BANKED_JSON_DIR = REPO_ROOT / "results/prod2d_closure_20260818/correspondence_arms"
BANKED_CSV_ROOT = REPO_ROOT / "results/prod2d_closure_20260818/arm_event_likelihoods"
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "p3_leverage_estimate_output.json"

REGISTRATION_SECTION: str = (
    "results/campaign51_20260728/realistic_20260729/"
    "PREREGISTRATION_P3_TWIN_20260822.md, arm LEV (§3) and the "
    "axis-leverage statement (A17, §5)"
)

H_GEN: float = 0.73
H_LO: float = 0.725
H_HI: float = 0.735

# BINDING: the 12 banked B-SEL seeds (correspondence_arms/bsel_seed*.json,
# enumerated -- registry order, first two = 900101/900102 per the prereg's
# PILOT stage selection).
BSEL_SEEDS: tuple[int, ...] = tuple(range(900101, 900113))

# Axis-leverage gate (prereg §5): predicted |Δ per seed| >= LEVERAGE_FACTOR
# x SEM_PAIRED_SCALE must hold BEFORE the pilot runs, else execution STOPS.
SEM_PAIRED_SCALE: float = 0.004
LEVERAGE_FACTOR: float = 5.0
LEVERAGE_THRESHOLD: float = LEVERAGE_FACTOR * SEM_PAIRED_SCALE


def _banked_csv_path(seed: int) -> Path:
    return (
        BANKED_CSV_ROOT
        / f"bsel_seed{seed}"
        / f"seed{seed}"
        / "simulations/diagnostics/event_likelihoods.csv"
    )


def _share_cat_at_h_gen(csv_path: Path, h_gen: float) -> npt.NDArray[np.float64]:
    """Per-event ``share_cat`` at ``h_gen`` from the banked CSV (assembly-true
    beta convention, verified against ``decompose_impostor_leg.py:239``).
    """
    df = pd.read_csv(csv_path)
    at = df[np.isclose(df["h"].to_numpy(dtype=np.float64), h_gen)].copy()
    at = at.sort_values("event_idx")
    alpha = at["alpha_G_phi"].to_numpy(dtype=np.float64)
    r_malm = at["r_Malm"].to_numpy(dtype=np.float64)
    lcat = at["L_cat_no_bh"].to_numpy(dtype=np.float64)
    b_num = at["B_num"].to_numpy(dtype=np.float64)
    beta_lcat = alpha / r_malm * lcat
    denom = beta_lcat + b_num
    share_cat = np.divide(beta_lcat, denom, out=np.zeros_like(beta_lcat), where=denom > 0.0)
    return np.asarray(share_cat, dtype=np.float64)


def _regenerate_events(seed: int, work_root: Path) -> pd.DataFrame:
    """Deterministic (zero-``evaluate()``) redraw of one B-SEL realization.

    Exact replica of ``run_arm_seed``'s ``bsel`` branch
    (``correspondence_1d.py:2739-2753``) through the draw step only: same
    ``CorrespondenceConfig`` (n_events=200, sigma_z_scale=area_scale=1.0,
    ``ARM_SPECS["bsel"]``), same ``host_pool_for_sigma_scale`` call, same
    ``build_bsel_selection_objects()`` (h_true default, i.e. ``H_TRUE`` =
    ``c1d.H_TRUE`` = 0.73), same ``draw_realization(host_mode=
    "population_selected", ...)`` call -- NO ``run_mirror_seed_inprocess``
    call (that is precisely the ``evaluate()`` step this instrument must not
    pay for).
    """
    sigma_z_scale, area_scale = c1d.ARM_SPECS["bsel"]
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=sigma_z_scale, area_scale=area_scale)
    gen = c1d.MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, _handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seed, sigma_z_scale=sigma_z_scale
    )
    completeness_obj, phi_survival_table_h_true = c1d.build_bsel_selection_objects()
    events = gen.draw_realization(
        seed,
        host_pool=host_pool,
        host_mode="population_selected",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table_h_true,
    )
    return events


def _dln_s_phi_per_event(obs_d_l: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-event central-difference ``d/dh ln S_bar_phi(z_obs; h)`` at ``H_GEN``.

    ``z_obs`` is the FLAGGED effective-z proxy (module docstring): the
    redshift implied by the event's own observed luminosity distance at
    ``H_GEN`` (``dist_to_redshift`` is scalar/``fsolve``-based, so this loops
    over events -- 200 calls/seed, not a hot path).
    """
    _, phi_table_lo = c1d.build_bsel_selection_objects(h_true=H_LO)
    _, phi_table_hi = c1d.build_bsel_selection_objects(h_true=H_HI)
    z_grid_lo, s_lo_grid = phi_table_lo[H_LO]
    z_grid_hi, s_hi_grid = phi_table_hi[H_HI]

    z_obs = np.array(
        [float(dist_to_redshift(float(d), h=H_GEN)) for d in obs_d_l], dtype=np.float64
    )
    s_lo = np.interp(z_obs, z_grid_lo, s_lo_grid)  # endpoint-clamped by default
    s_hi = np.interp(z_obs, z_grid_hi, s_hi_grid)
    floor = np.finfo(np.float64).tiny
    dln = (np.log(np.maximum(s_hi, floor)) - np.log(np.maximum(s_lo, floor))) / (H_HI - H_LO)
    return dln


def run_seed(seed: int, work_root: Path) -> dict[str, Any]:
    banked_json = json.loads((BANKED_JSON_DIR / f"bsel_seed{seed}.json").read_text())
    csv_path = _banked_csv_path(seed)
    share_cat = _share_cat_at_h_gen(csv_path, H_GEN)

    events = _regenerate_events(seed, work_root / f"seed{seed}")
    obs_d_l = events["luminosity_distance"].to_numpy(dtype=np.float64)
    dln_s_phi = _dln_s_phi_per_event(obs_d_l)

    mean_share_cat = float(share_cat.mean())
    mean_dln_s_phi = float(dln_s_phi.mean())
    predicted_delta = mean_share_cat * mean_dln_s_phi

    return {
        "seed": seed,
        "banked_mean_h": banked_json["mean_h"],
        "banked_n_events": int(share_cat.size),
        "n_events_regenerated": int(obs_d_l.size),
        "mean_share_cat_banked": mean_share_cat,
        "mean_dln_s_phi_regenerated": mean_dln_s_phi,
        "dln_s_phi_sd_regenerated": float(dln_s_phi.std(ddof=1)) if dln_s_phi.size > 1 else None,
        "predicted_delta": predicted_delta,
        "substitution_note": (
            "predicted_delta = mean(share_cat) [banked, paired index] x "
            "mean(dln_S_phi) [regenerated proxy, DECOUPLED index] -- see "
            "module docstring 'FLAGGED SUBSTITUTION'"
        ),
    }


def main() -> int:
    t0 = time.time()
    work_root = OUT_DIR / "p3_lev_work"
    work_root.mkdir(parents=True, exist_ok=True)

    per_seed = [run_seed(seed, work_root) for seed in BSEL_SEEDS]
    predicted = np.array([r["predicted_delta"] for r in per_seed], dtype=np.float64)
    fleet_mean = float(predicted.mean())
    fleet_sd = float(predicted.std(ddof=1)) if predicted.size > 1 else None
    max_abs_predicted = float(np.max(np.abs(predicted)))

    leverage_ratio_mean = abs(fleet_mean) / LEVERAGE_THRESHOLD
    leverage_ratio_max = max_abs_predicted / LEVERAGE_THRESHOLD

    elapsed = time.time() - t0

    output: dict[str, Any] = {
        "registered_in": REGISTRATION_SECTION,
        "seeds": list(BSEL_SEEDS),
        "h_gen": H_GEN,
        "h_lo": H_LO,
        "h_hi": H_HI,
        "flagged_substitutions": [
            (
                "candidate z_obs: used the event's own observed-d_L-implied "
                "redshift at H_GEN as an effective-z proxy, NOT the real "
                "catalogue candidate search (galaxy_catalogue/handler.py "
                "get_possible_hosts_from_ball_tree) -- disclosed, module "
                "docstring 'FLAGGED SUBSTITUTION'"
            ),
            (
                "per-seed predicted_delta is a PRODUCT OF MEANS (banked "
                "share_cat index vs. regenerated-but-unfiltered z_obs "
                "index), not a true paired per-event mean -- the 200 "
                "regenerated events cannot be reliably index-aligned to the "
                "banked CSV's 174 (post-quality-filter, renumbered) events "
                "without running evaluate() itself"
            ),
        ],
        "per_seed": per_seed,
        "fleet": {
            "n_seeds": len(per_seed),
            "predicted_delta_mean": fleet_mean,
            "predicted_delta_sd": fleet_sd,
            "predicted_delta_max_abs": max_abs_predicted,
        },
        "axis_leverage_check": {
            "reference": f"{REGISTRATION_SECTION}, axis-leverage statement",
            "sem_paired_scale": SEM_PAIRED_SCALE,
            "leverage_factor": LEVERAGE_FACTOR,
            "leverage_threshold": LEVERAGE_THRESHOLD,
            "fleet_mean_predicted_delta": fleet_mean,
            "ratio_fleet_mean_over_threshold": leverage_ratio_mean,
            "ratio_max_abs_over_threshold": leverage_ratio_max,
            "note": (
                "the GATE-LEV pass/fail decision (continue to PILOT vs. "
                "STOP for re-design) is the orchestrator's/author's per "
                "prereg §5 -- this instrument prints the ratios only"
            ),
        },
        "elapsed_s": elapsed,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2))

    print("=== [P3-IMP] arm LEV -- axis-leverage estimate ===")
    for r in per_seed:
        print(
            f"seed {r['seed']}: mean(share_cat)={r['mean_share_cat_banked']:.4e}  "
            f"mean(dln_S_phi)={r['mean_dln_s_phi_regenerated']:+.4e}  "
            f"predicted_delta={r['predicted_delta']:+.4e}"
        )
    print(f"fleet mean predicted_delta = {fleet_mean:+.4e} (sd={fleet_sd})")
    print(f"leverage threshold (5x SEM_paired-scale 0.004) = {LEVERAGE_THRESHOLD:.4e}")
    print(f"ratio |fleet mean| / threshold = {leverage_ratio_mean:.2f}")
    print(f"ratio max|per-seed| / threshold = {leverage_ratio_max:.2f}")
    print(f"elapsed = {elapsed:.1f} s")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
