r"""[P3-IMP] the C-B coded-null pinning pass -- the free corroborator (~1 CPU-h, zero-``evaluate()``).

Registered in ``PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md`` §2 ("C-B rides along ...
REPORTED-WITH-VERDICT, not verdict-bearing") and §7 ("C-B pinning: ~1 CPU-h"), **as amended by
PA-CA-7(c)** (``A20_REVIEW_CA_DESIGN_20260824.md``): C-B enters §6 as a TWIN-CALIBRATED
falsifier ONLY if its pinned coded-null and twin-null separate >= 3 sigma_null AND the measured
fleet Lambda-bar lies >= 3 sigma_null CLOSER to the coded-null than to the twin-null (computed
here, field ``pa_ca_7c_role``); otherwise REPORT-ONLY, never verdict-bearing. Derived in
``CLAIM_B0_FINITE_MOMENT_20260824.md`` §3.2 and adjudicated in
``GATE_B_ADJUDICATION_FINITE_MOMENT_20260824.md`` §3b: the twin-null chain (``L1``, ``KL``, the
F-0 acceptance tilt) is BANKED; the coded-null (claim's own text: "Coded-null ~= -0.06:
UNDETERMINED -- needs the 1-CPU-h pinning pass") is what this instrument computes.

**The statistic (claim §3.2, banked/zero-compute on the 12 pairs).** Per paired-live event
(same b0i seed => same realization, twin vs coded columns of the SAME draw):

    Lambda_e = ln(L_cat_no_bh^BT_e / L_cat_no_bh^BC_e) + ln(Sigma_w / Sigma_phi_tilde)

center = ln(Sigma_w/Sigma_phi_tilde) = +0.2542072 [banked]. This instrument RECOMPUTES the
realized fleet Lambda-bar zero-compute (§1 below, a straight re-derivation from the banked
``bt_<seed>``/``bc_<seed>`` CSVs, no new leaf needed) and PINS the two theoretical nulls the
claim's own decomposition needs to interpret it against (§2/§3 below).

**Sharp-GW factor-out (claim §2.3/§3.2, verified [LOCAL] there: per-event residual
``ln(L^BT/L^BC) - ln S_bar_phi(z_true)`` has mean +0.0002, sd 0.023 over 1157 paired-live
events).** To EXCELLENT approximation ``Lambda_e ~= ln S_bar_phi(z_true_e) + center``, so
``E[Lambda]`` under any candidate GENERATIVE law for ``z_true`` reduces to
``E_law[ln S_bar_phi(z_true)] + center`` -- a MOMENT of the (already-tabulated) S_bar_phi survival
function under that law's own z-marginal. This instrument computes that moment for FOUR laws via
one shared, EXACT (quadrature, not Monte-Carlo) technique -- the "doctored value table" trick
(§0 below) -- rather than by drawing samples, which removes MC noise from the pinning entirely
(the GATE-B adjudication's own text: "two more doctored-table passes + the smeared acceptance
function A(z) on the grid -- zero evaluate(), ~1 CPU-h").

**Four laws, two axes (arrangement x conditioning), reusing
:func:`~darksiren_emri.validation.correspondence_1d.kernel_smeared_survival` UNCHANGED (imported,
never reimplemented) with a substituted table value -- never a substituted FORMULA:**

- **Arrangement axis.** TWIN's implied generative z-law is
  ``z_true ~ k_g(z).S_bar_phi(z;h) / S_tilde_phi_g`` (the b0i "catalogue_selected" host mode's OWN
  z_true kernel, ``_draw_kernel_survival_redshifts``'s density -- this instrument does not draw
  from it, it takes its EXACT first moment via quadrature) -- host g drawn proportional to
  ``w_g.S_tilde_phi_g``. CODED's implied law has NO per-candidate survival reweight anywhere
  (``L_cat^BC``'s own integrand omits ``[S_bar_phi]_twin-only``, prereg/claim §1): host g drawn
  proportional to ``w_g`` ALONE, ``z_true ~ k_g(z)`` (the BARE per-host kernel, un-reweighted).
- **Conditioning axis.** UNCONDITIONED (the law's own marginal, no F-0) vs CONDITIONED (the F-0
  ``sigma_dL/d_hat < 0.10 AND SNR>=20`` acceptance-weighted marginal, ``E[.|acc]``). F-0 depends
  on an event's ``(z_true, donor row)`` ONLY through ``d_hat = d_L(z_true;h) + sigma_dL.eps``
  (``eps~N(0,1)``, donor SNR-weighted) -- independent of host identity given ``z_true`` -- so its
  effect factors into a single 1-D function ``A(z)`` (:func:`_acceptance_grid`, ANALYTIC: for a
  fixed donor row i, ``accept <=> eps > 10 - d_L(z;h)/sigma_dL,i`` for a standard normal ``eps``,
  a closed form via ``scipy.stats.norm.cdf``, SNR-weighted-averaged over the SAME donor pool
  :data:`~darksiren_emri.validation.correspondence_1d.CRB_CSV_PATH` every b0i/RHS draw uses) --
  computed once on the S_bar_phi table's own z-grid, folded multiplicatively into the doctored
  table, exactly analogous to how the F-0-conditioned targets in the claim's own §2.5 fold a
  donor-marginalized acceptance probability into a model-side moment.

**The doctored-table algebra (verified by direct expansion, §0).** Write ``f(z) = ln S_bar_phi(z)``
(floored before the log to avoid ``-inf``) and let ``KSS(table)`` denote
:func:`~darksiren_emri.validation.correspondence_1d.kernel_smeared_survival` called with
``phi_survival_table`` REPLACED by ``{h: (z_grid, table(z_grid))}`` (same host/window inputs
otherwise) -- ``KSS(table)_g = INTEGRAL k_g(z).table(z) dz / INTEGRAL k_g(z) dz`` per host ``g``,
by the function's OWN definition (docstring, ``correspondence_1d.py:1146-1234``). Then, with
``S = S_bar_phi(z)`` and ``A = A(z)`` (the acceptance grid):

    E_twin[f]        = sum_g( w_g . KSS(S.f)_g )   / sum_g( w_g . KSS(S)_g )     [denom = Sigma_phi_tilde]
    E_twin[f | acc]  = sum_g( w_g . KSS(S.A.f)_g ) / sum_g( w_g . KSS(S.A)_g )
    E_coded[f]       = sum_g( w_g . KSS(f)_g )     / Sigma_w                     [KSS(1)_g == 1 always]
    E_coded[f | acc] = sum_g( w_g . KSS(A.f)_g )   / sum_g( w_g . KSS(A)_g )

(every per-host ``Z_g = INTEGRAL k_g(z) dz`` normalization the function applies internally cancels
identically between a doctored numerator call and its matching doctored denominator call, so
``Z_g`` never needs to be recovered explicitly -- see the module docstring's derivation trail in
the accompanying task report for the full algebraic check). ``w_g`` is the estimator's own
per-galaxy rate weight (:func:`~darksiren_emri.emri_rate.R_eff_per_mbh`, imported, PA-2 parity-
gated), ``Sigma_w``/``Sigma_phi_tilde`` are read from
:func:`p3_b0_identity_test.mass_companion` (imported, "same leaf builds as the RHS scorer").

Output (``cb_null_pinning_output.json``): the four ``E[f]``/``E[f|acc]`` moments, the four
resulting Lambda-nulls (``+center``), the twin-unconditioned null cross-checked against the
claim's banked ``KL(g_T||g_C) = +0.01856``, and the realized fleet Lambda-bar recomputed
zero-compute from the 12 banked pairs (cross-checked against the claim's ``-0.02516+/-0.00454``).

Usage:
    uv run python results/campaign51_20260728/realistic_20260729/cb_null_pinning.py
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# Same-directory import (guarded by __main__, does not execute anything at import time). Reuses
# mass_companion (Sigma_w/Sigma_phi_tilde/rho), the banked-CSV readers, and A22.
import p3_b0_identity_test as o5  # noqa: E402
import pandas as pd
from scipy.stats import norm

from darksiren_emri.bayesian_inference.bayesian_statistics import (  # noqa: E402
    FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD,
)
from darksiren_emri.constants import SNR_THRESHOLD  # noqa: E402
from darksiren_emri.emri_rate import R_eff_per_mbh  # noqa: E402
from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns  # noqa: E402
from darksiren_emri.physical_relations import dist_vectorized  # noqa: E402
from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
BANKED_B0I_META_ROOT: Path = THIS_DIR / "p3_b0_work"
OUT_PATH_DEFAULT: Path = THIS_DIR / "cb_null_pinning_output.json"

REGISTRATION_SECTION: str = (
    "results/campaign51_20260728/realistic_20260729/"
    "PREREGISTRATION_CA_BOUNDED_IDENTITY_20260824.md §2 (C-B) + §7, "
    "CLAIM_B0_FINITE_MOMENT_20260824.md §3.2, "
    "GATE_B_ADJUDICATION_FINITE_MOMENT_20260824.md §3b"
)

H_GEN: float = o5.H_GEN

BANKED_SEEDS: tuple[int, ...] = tuple(range(900101, 900113))  # the 12 B-T/B-C paired seeds

# Cross-checks against the claim/adjudication's own banked numbers (reported, not asserted --
# a mismatch is a FINDING for the report, never a silent pass/fail here).
BANKED_CENTER: float = 0.2542072
BANKED_TWIN_UNCONDITIONED_KL: float = 0.01856  # L1 + center, claim §0/§3.2
BANKED_LAMBDA_BAR: float = -0.02516
BANKED_LAMBDA_BAR_SE: float = 0.00454

_LN_FLOOR: float = np.finfo(np.float64).tiny


def _doctored_table(
    phi_survival_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    h: float,
    y: npt.NDArray[np.float64],
) -> dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Wrap ``y`` (same shape as the table's own z-grid) as a
    :func:`~darksiren_emri.validation.correspondence_1d.kernel_smeared_survival`-shaped table --
    the "doctored value table" substitution (module docstring): the SAME z-grid, a REPLACED
    y-column. Never touches :func:`kernel_smeared_survival` itself.
    """
    z_grid, _s_phi = phi_survival_table[h]
    return {h: (z_grid, y)}


def _acceptance_grid(
    z_grid: npt.NDArray[np.float64], h: float, donor_rows: pd.DataFrame
) -> npt.NDArray[np.float64]:
    r"""``A(z) = P(F-0 accept | z_true=z)``, analytic, SNR-weighted over the donor pool.

    For a fixed donor row ``i`` (``sigma_dL,i``, ``SNR_i``) and ``d_hat = d_L(z;h) +
    sigma_dL,i . eps`` (``eps ~ N(0,1)``, matching
    :meth:`~darksiren_emri.validation.correspondence_1d.MirrorUniverseGenerator.draw_realization`
    item (c)):

        accept <=> sigma_dL,i / d_hat < THRESH   <=>   eps > 1/THRESH - d_L(z;h)/sigma_dL,i

    (assuming ``d_hat > 0``, true for the realistic ``sigma_dL << d_L`` regime this venue draws
    in -- the same regime GATE ACC's own forensics already characterize). So
    ``P(accept | z, donor i) = 1 - Phi(1/THRESH - d_L(z;h)/sigma_dL,i)`` for donor rows passing
    the (donor-level, z-independent) SNR>=20 gate, 0 otherwise; ``A(z)`` is the SNR-weighted
    average over the donor pool (the SAME weights the venue's own SNR-weighted-without-
    replacement donor draw uses, :func:`~darksiren_emri.validation.correspondence_1d.
    MirrorUniverseGenerator.draw_realization` item (b)) -- fully analytic, no MC draws, no
    z-dependence beyond ``d_L(z;h)``.

    Args:
        z_grid: Redshifts to evaluate ``A`` at (the S_bar_phi table's own grid).
        h: Dimensionless Hubble parameter.
        donor_rows: The pinned CRB pool (``SNR``,
            ``delta_luminosity_distance_delta_luminosity_distance`` columns).

    Returns:
        ``A(z_grid)``, shape matching ``z_grid``.
    """
    snr = donor_rows["SNR"].to_numpy(dtype=np.float64)
    sigma_dl = np.sqrt(
        donor_rows["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(dtype=np.float64)
    )
    snr_ok = snr >= SNR_THRESHOLD
    weights = np.where(snr_ok, snr, 0.0)
    weights = weights / weights.sum()

    d_l = np.asarray(dist_vectorized(z_grid, h=h), dtype=np.float64)  # (n_z,)
    thresh_inv = 1.0 / FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD
    # (n_donor, n_z): t_i(z) = thresh_inv - d_L(z)/sigma_dL,i
    t = thresh_inv - d_l[None, :] / sigma_dl[:, None]
    accept_prob = norm.cdf(-t)  # P(eps > t) = 1 - Phi(t) = Phi(-t)
    accept_prob = np.where(weights[:, None] > 0.0, accept_prob, 0.0)
    a_z: npt.NDArray[np.float64] = np.sum(weights[:, None] * accept_prob, axis=0)
    return a_z


def _eligible_catalogue_rows(
    z_max: float,
) -> dict[str, npt.NDArray[np.float64]]:
    """The SAME eligibility mask :func:`p3_b0_identity_test.mass_companion` applies
    (``z < z_max`` AND finite positive BH mass) -- data selection, not a physics formula; the
    WEIGHTING (``w_g``) and SMEARING (``kernel_smeared_survival``) leaves are imported, this is
    just the row filter both that function and this one need independently.
    """
    handler = c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH)
    catalog = handler.reduced_galaxy_catalog
    z_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64), dtype=np.float64
    )
    m_all = np.asarray(
        catalog[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64), dtype=np.float64
    )
    z_err_all = np.asarray(
        catalog[InternalCatalogColumns.REDSHIFT_ERROR].to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    phi_s_all = np.asarray(
        catalog[InternalCatalogColumns.PHI_S].to_numpy(dtype=np.float64), dtype=np.float64
    )
    q_s_all = np.asarray(
        catalog[InternalCatalogColumns.THETA_S].to_numpy(dtype=np.float64), dtype=np.float64
    )
    eligible = (z_all < z_max) & np.isfinite(m_all) & (m_all > 0.0)
    return {
        "z_g": z_all[eligible],
        "M_g": m_all[eligible],
        "z_error_g": np.maximum(z_err_all[eligible], c1d.EXACT_Z_ERROR_FLOOR),
        "phiS_g": phi_s_all[eligible],
        "qS_g": q_s_all[eligible],
        "n_eligible": np.array([int(eligible.sum())]),
    }


def _fleet_moment(
    doctored_num_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
    doctored_den_table: dict[float, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None,
    rows: dict[str, npt.NDArray[np.float64]],
    w_g: npt.NDArray[np.float64],
    completeness_obj: Any,
    h: float,
    fixed_denominator: float | None = None,
) -> float:
    """``sum_g(w_g . KSS(num_table)_g) / [sum_g(w_g . KSS(den_table)_g) OR fixed_denominator]``
    -- the module docstring's shared aggregation (chunking is handled internally by
    :func:`~darksiren_emri.validation.correspondence_1d.kernel_smeared_survival` itself, per its
    own ``_KERNEL_SMEAR_CHUNK`` row-chunking, imported unchanged).
    """
    num_g = c1d.kernel_smeared_survival(
        rows["z_g"],
        rows["z_error_g"],
        doctored_num_table,
        completeness_obj,
        rows["phiS_g"],
        rows["qS_g"],
        h=h,
    )
    numerator = float((w_g * num_g).sum())
    if fixed_denominator is not None:
        return numerator / fixed_denominator
    assert doctored_den_table is not None
    den_g = c1d.kernel_smeared_survival(
        rows["z_g"],
        rows["z_error_g"],
        doctored_den_table,
        completeness_obj,
        rows["phiS_g"],
        rows["qS_g"],
        h=h,
    )
    denominator = float((w_g * den_g).sum())
    return numerator / denominator


def _realized_lambda_bar() -> dict[str, Any]:
    """Zero-compute recompute of the realized fleet Lambda-bar from the 12 banked b0i pairs
    (straight re-derivation, no new leaf -- cross-checks the claim's ``-0.02516+/-0.00454``).
    """
    mc = o5.mass_companion(H_GEN)
    center = float(np.log(mc["Sigma_w"] / mc["Sigma_phi_tilde"]))
    per_seed: list[dict[str, Any]] = []
    for seed in BANKED_SEEDS:
        bt_meta = json.loads((BANKED_B0I_META_ROOT / f"bt_{seed}_meta.json").read_text())
        bc_meta = json.loads((BANKED_B0I_META_ROOT / f"bc_{seed}_meta.json").read_text())
        bt = o5._rows_at_h(o5._meta_csv(bt_meta), H_GEN).set_index("event_idx")
        bc = o5._rows_at_h(o5._meta_csv(bc_meta), H_GEN).set_index("event_idx")
        common = sorted(set(bt.index) & set(bc.index))
        l_bt = bt.loc[common, "L_cat_no_bh"].to_numpy(dtype=np.float64)
        l_bc = bc.loc[common, "L_cat_no_bh"].to_numpy(dtype=np.float64)
        floor = np.finfo(np.float64).tiny
        lam = np.log(np.maximum(l_bt, floor)) - np.log(np.maximum(l_bc, floor)) + center
        per_seed.append(
            {
                "seed": seed,
                "n_paired_live": len(common),
                "lambda_mean_seed": float(lam.mean()) if lam.size else None,
            }
        )
    seed_means = np.array(
        [r["lambda_mean_seed"] for r in per_seed if r["lambda_mean_seed"] is not None],
        dtype=np.float64,
    )
    lambda_bar = float(seed_means.mean()) if seed_means.size else float("nan")
    se = float(seed_means.std(ddof=1) / np.sqrt(seed_means.size)) if seed_means.size > 1 else None
    return {
        "center": center,
        "per_seed": per_seed,
        "lambda_bar": lambda_bar,
        "lambda_bar_se": se,
        "n_positive": int(np.sum(seed_means > 0.0)) if seed_means.size else 0,
        "n_seeds": int(seed_means.size),
    }


def run() -> dict[str, Any]:
    t0 = time.time()
    stamp = o5._a22_stamp()

    completeness_obj, phi_survival_table = c1d.build_bsel_selection_objects(h_true=H_GEN)
    z_grid, s_phi_grid = phi_survival_table[H_GEN]
    s_floor = np.clip(s_phi_grid, _LN_FLOOR, None)
    ln_s = np.log(s_floor)

    mc = o5.mass_companion(H_GEN)
    sigma_w = mc["Sigma_w"]
    sigma_phi_tilde = mc["Sigma_phi_tilde"]
    center = float(np.log(sigma_w / sigma_phi_tilde))

    donor_rows = pd.read_csv(
        c1d.CRB_CSV_PATH, usecols=["SNR", "delta_luminosity_distance_delta_luminosity_distance"]
    )
    a_grid = _acceptance_grid(z_grid, H_GEN, donor_rows)

    rows = _eligible_catalogue_rows(float(z_grid.max()))
    w_g = np.asarray(R_eff_per_mbh(rows["M_g"]), dtype=np.float64) / (1.0 + rows["z_g"])

    # ── The four doctored tables (module docstring's algebra) ────────────────
    table_s_lns = _doctored_table(phi_survival_table, H_GEN, s_floor * ln_s)  # twin num, uncond.
    table_s = _doctored_table(
        phi_survival_table, H_GEN, s_floor
    )  # twin denom, uncond. (=S_bar_phi)
    table_lns = _doctored_table(phi_survival_table, H_GEN, ln_s)  # coded num, uncond.
    table_s_a_lns = _doctored_table(
        phi_survival_table, H_GEN, s_floor * a_grid * ln_s
    )  # twin num, cond.
    table_s_a = _doctored_table(phi_survival_table, H_GEN, s_floor * a_grid)  # twin denom, cond.
    table_a_lns = _doctored_table(phi_survival_table, H_GEN, a_grid * ln_s)  # coded num, cond.
    table_a = _doctored_table(phi_survival_table, H_GEN, a_grid)  # coded denom, cond.

    e_twin_unconditioned = _fleet_moment(table_s_lns, table_s, rows, w_g, completeness_obj, H_GEN)
    e_coded_unconditioned = _fleet_moment(
        table_lns, None, rows, w_g, completeness_obj, H_GEN, fixed_denominator=sigma_w
    )
    e_twin_conditioned = _fleet_moment(table_s_a_lns, table_s_a, rows, w_g, completeness_obj, H_GEN)
    e_coded_conditioned = _fleet_moment(table_a_lns, table_a, rows, w_g, completeness_obj, H_GEN)

    # Byproducts: the law-implied acceptance probabilities (cross-check vs GATE ACC's independent
    # MC in ca_rhs_scorer.py --stage acceptance).
    p_accept_twin_law = _fleet_moment(table_s_a, table_s, rows, w_g, completeness_obj, H_GEN)
    p_accept_coded_law = _fleet_moment(
        table_a, None, rows, w_g, completeness_obj, H_GEN, fixed_denominator=sigma_w
    )

    nulls = {
        "twin_unconditioned": e_twin_unconditioned + center,
        "twin_conditioned": e_twin_conditioned + center,
        "coded_unconditioned": e_coded_unconditioned + center,
        "coded_conditioned": e_coded_conditioned + center,
    }

    realized = _realized_lambda_bar()

    # PA-CA-7(c): C-B's verdict-map role. Enters §6 as a TWIN-CALIBRATED falsifier ONLY if the
    # pinned coded-null and twin-null (both CONDITIONED, the F-0-matched pair) separate >= 3
    # sigma_null AND the realized fleet Lambda-bar lies >= 3 sigma_null CLOSER to the coded-null
    # than to the twin-null; otherwise REPORT-ONLY (never verdict-bearing). sigma_null is taken
    # as the banked Lambda-bar SEM (:data:`BANKED_LAMBDA_BAR_SE`) -- the only pinned sigma this
    # zero-compute pass has, disclosed convention (this instrument recomputes the fleet mean
    # itself but reuses the claim's own banked SEM as the sigma unit, consistent with the
    # ``..._in_units_of_banked_se`` fields already reported below).
    sigma_null = BANKED_LAMBDA_BAR_SE
    null_separation = abs(nulls["coded_conditioned"] - nulls["twin_conditioned"])
    null_separation_in_sigma = null_separation / sigma_null if sigma_null else float("nan")
    dist_to_coded = abs(realized["lambda_bar"] - nulls["coded_conditioned"])
    dist_to_twin = abs(realized["lambda_bar"] - nulls["twin_conditioned"])
    closer_to_coded_margin_in_sigma = (
        (dist_to_twin - dist_to_coded) / sigma_null if sigma_null else float("nan")
    )
    pa_ca_7c_falsifier_eligible = bool(
        null_separation_in_sigma >= 3.0 and closer_to_coded_margin_in_sigma >= 3.0
    )
    pa_ca_7c_role = (
        "FALSIFIER-ELIGIBLE (enters prereg §6 as a TWIN-CALIBRATED falsifier, PA-CA-7c)"
        if pa_ca_7c_falsifier_eligible
        else "REPORT-ONLY (PA-CA-7c: not verdict-bearing -- null separation or coded-proximity "
        "margin below 3 sigma_null)"
    )

    out: dict[str, Any] = {
        "reference": REGISTRATION_SECTION,
        "h_gen": H_GEN,
        "a22_stamp": stamp,
        "center": center,
        "banked_center_crosscheck": BANKED_CENTER,
        "center_rel_diff_vs_banked": abs(center - BANKED_CENTER) / abs(BANKED_CENTER),
        "n_eligible_galaxies": int(rows["n_eligible"][0]),
        "moments": {
            "E_twin_unconditioned_ln_S_bar_phi": e_twin_unconditioned,
            "E_twin_conditioned_ln_S_bar_phi": e_twin_conditioned,
            "E_coded_unconditioned_ln_S_bar_phi": e_coded_unconditioned,
            "E_coded_conditioned_ln_S_bar_phi": e_coded_conditioned,
        },
        "nulls": nulls,
        "twin_unconditioned_null_is_KL_g_T_g_C": nulls["twin_unconditioned"],
        "banked_twin_unconditioned_KL_crosscheck": BANKED_TWIN_UNCONDITIONED_KL,
        "twin_unconditioned_null_rel_diff_vs_banked": abs(
            nulls["twin_unconditioned"] - BANKED_TWIN_UNCONDITIONED_KL
        )
        / abs(BANKED_TWIN_UNCONDITIONED_KL),
        "coded_conditioned_null_is_the_pinning_target": nulls["coded_conditioned"],
        "p_accept_twin_law_model": p_accept_twin_law,
        "p_accept_coded_law_model": p_accept_coded_law,
        "realized_fleet_lambda_bar": realized,
        "banked_lambda_bar_crosscheck": BANKED_LAMBDA_BAR,
        "banked_lambda_bar_se_crosscheck": BANKED_LAMBDA_BAR_SE,
        "realized_minus_twin_conditioned_null_in_units_of_banked_se": (
            (realized["lambda_bar"] - nulls["twin_conditioned"]) / BANKED_LAMBDA_BAR_SE
        ),
        "realized_minus_coded_conditioned_null_in_units_of_banked_se": (
            (realized["lambda_bar"] - nulls["coded_conditioned"]) / BANKED_LAMBDA_BAR_SE
        ),
        "pa_ca_7c_role": {
            "sigma_null": sigma_null,
            "null_separation": null_separation,
            "null_separation_in_sigma": null_separation_in_sigma,
            "closer_to_coded_margin_in_sigma": closer_to_coded_margin_in_sigma,
            "falsifier_eligible": pa_ca_7c_falsifier_eligible,
            "role": pa_ca_7c_role,
        },
        "elapsed_s": time.time() - t0,
    }
    OUT_PATH_DEFAULT.write_text(json.dumps(out, indent=2))

    print("=== [P3-IMP] cb_null_pinning -- the coded-arrangement null of Lambda (PA-CA-7c) ===")
    print(
        f"center = {center:.7f}  (banked cross-check {BANKED_CENTER}, "
        f"rel_diff={out['center_rel_diff_vs_banked']:.2e})"
    )
    print(
        f"twin_unconditioned_null (= KL(g_T||g_C)) = {nulls['twin_unconditioned']:+.5f}  "
        f"(banked cross-check {BANKED_TWIN_UNCONDITIONED_KL}, "
        f"rel_diff={out['twin_unconditioned_null_rel_diff_vs_banked']:.2e})"
    )
    print(f"twin_conditioned_null    = {nulls['twin_conditioned']:+.5f}")
    print(f"coded_unconditioned_null (= -KL(g_C||g_T)) = {nulls['coded_unconditioned']:+.5f}")
    print(f"coded_conditioned_null   = {nulls['coded_conditioned']:+.5f}  <-- the pinning target")
    print(
        f"realized fleet Lambda-bar (recomputed) = {realized['lambda_bar']!r} "
        f"+/- {realized['lambda_bar_se']!r}  (banked cross-check "
        f"{BANKED_LAMBDA_BAR} +/- {BANKED_LAMBDA_BAR_SE})"
    )
    print(
        f"p_accept_twin_law_model = {p_accept_twin_law:.4f}  p_accept_coded_law_model = {p_accept_coded_law:.4f}"
    )
    print(
        f"PA-CA-7c: null_separation={null_separation:.5f} ({null_separation_in_sigma:.2f} "
        f"sigma_null)  closer_to_coded_margin={closer_to_coded_margin_in_sigma:.2f} sigma_null"
    )
    print(f"PA-CA-7c role: {pa_ca_7c_role}")
    print(f"elapsed = {out['elapsed_s']:.1f} s")
    print(f"wrote {OUT_PATH_DEFAULT}")
    return out


if __name__ == "__main__":
    # PA-CA-7(c): C-B enters prereg §6 as a TWIN-CALIBRATED falsifier ONLY if its pinned
    # coded-null and twin-null separate >= 3 sigma_null AND the measured Lambda-bar lies
    # >= 3 sigma_null closer to the coded-null (computed in run(), field "pa_ca_7c_role");
    # otherwise REPORT-ONLY -- no pass/fail exit code either way (this instrument never gates).
    run()
    raise SystemExit(0)
