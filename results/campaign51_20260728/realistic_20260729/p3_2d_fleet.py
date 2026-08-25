r"""[P3-2D] the b0i-2D fleet driver -- PREREGISTRATION_P3_2D_20260825.md, PA-2D-2's registered
"fleet-driver threading gap" fix.

**Why this file exists.** ``correspondence_1d.run_arm_seed`` (the general-purpose fleet driver
:func:`darksiren_emri.validation.correspondence_1d.run_arm_seed`, ~:3651-3775) already dispatches
``host_mode="catalogue_selected_2d"`` (the b0i2d venue) correctly on the DRAW side (:3722-3741),
but its own call to :func:`darksiren_emri.validation.correspondence_1d.run_mirror_seed_inprocess`
(:3766-3775) does NOT thread ``catalogue_numerator_survival_2d``,
``catalogue_numerator_survival_2d_center``, or ``catalogue_global_selection`` -- so every b0i2d
run through ``run_arm_seed`` silently evaluates under the pre-[P3-2D] byte-identical default
(``catalogue_numerator_survival_2d="off"``), never actually exercising the with-BH 2D twin. PA-2D-2
registers the fix as "a committed driver wrapper mirroring the ``p3_b0_identity_test._run_arm_seed``
precedent" -- THIS file: it calls ``run_mirror_seed_inprocess`` DIRECTLY (bypassing
``run_arm_seed``'s gap entirely), threading all FIVE PA-2D-1/F7-resolved flags explicitly, exactly
as ``p3_b0_identity_test._run_arm_seed`` already does for the (analogous, 1D) b0i venue.

**The six A22-resolved flags (PA-2D-1 F7, PREREGISTRATION_P3_2D_20260825.md \S 2, this driver's
CLI \S3.2):**

1. ``selection_in_completion_numerator="fused"``
2. ``catalogue_numerator_survival="phi"`` (the 1D twin PRODUCTION-DEFAULT resolved value post
   row-#197 adoption -- not this thread's own axis. PA-2D-6: the original "off" pin was STALE
   (pre-adoption) and violated F7's "(1D twin production default)" registered text)
3. ``catalogue_global_selection="phi"`` ([P3-RPHI] the fourth Path-A slot, adopted)
4. ``catalogue_numerator_survival_2d`` -- THIS thread's own axis: ``"off"`` (arm B2-C, coded) or
   ``"mz_sel"`` (arm B2-T, twin)
5. ``catalogue_numerator_survival_2d_center="eff"`` (F2 ruling: production gaussian-branch
   centering, ``_host_M_eff``)

**Venue:** b0i2d (``host_mode="catalogue_selected_2d"``, ``ARM_SPECS["b0i2d"]=(1.0,1.0)``,
``ARM_SELECTION_CELL["b0i2d"]="fused"``, ``ARM_SEEDS["b0i2d"]`` = 24 seeds, 900101-900124, PA-2D-1
F14's power decision). Single-h read: ``h_values=(0.73,)``, ``h_bounds=(0.50,0.86)`` (PA-CA-10's
h_bounds pin, carried per F16).

**Stages** (``--stage {pilot,fleet,gates,lhs2d}``):

- ``pilot``: seed 900101 under BOTH arms (B2-C, B2-T) -- fast sanity before the 24-seed fleet.
- ``fleet``: one arm's full 24-seed fleet (``--arm {bc,bt}`` required) -- sequential (no
  process-pool fan-out, same ``run_mirror_seed_inprocess`` module-state-monkeypatch constraint the
  b0/b0i drivers document), idempotent per-seed sentinel (existing ``<subdir>_meta.json`` reused,
  disclosed, o5/o6 precedent).
- ``gates``: the three scoreable instrument-side gates -- GATE M2-LINK (F11's three-part
  registered form), the F10(c) z-marginal consistency check (registered as a zero-cost companion;
  implemented here as a PROXY against the b0i (1D) fleet's own z_true histogram -- see
  :func:`_gate_f10c_zmarginal_proxy`'s docstring for the disclosed scope limit: the FULL F10(c)
  check is the RHS\ :sub:`2` scorer's own completion-class replay, ``ca_rhs_scorer.py``'s 2D
  extension, a SEPARATE not-yet-built instrument per the prereg's \S 2 instrument list), and GATE
  ACC-extended (F12) -- replays :func:`correspondence_1d._draw_2d_accepted_latents` independently
  (same seed, same inputs -- byte-identical accepted latents, PLUS the ``n_drawn_total``/
  ``n_rounds`` diagnostics ``draw_realization`` discards) to report p\ :math:`\bar{}`\ :sub:`s` and
  its binomial band -- F12: "the 1D fleet rate 0.5821 is NOT a reference", so this is a REPORTING
  gate (p̄\ :sub:`s` from the extended-law replay), not a fixed-threshold PASS/FAIL.
- ``lhs2d``: the registered statistic, PA-CA-1 drawn-count form --
  ``LHS2_s = (C2*/200) * sum_acc(1-w2_e)`` per B2-T seed,
  ``w2_e = alpha_G_phi*L_cat_with_bh / (alpha_G_phi*L_cat_with_bh + B_num_wbh)`` (prereg \S 1,
  A20 review F3), fleet mean +/- SEM. Requires ``ca_rhs_work2d/p3_2d_companion.json`` (C2*,
  PA-2D-1 F4) and the B2-T arm's fleet metas.

**HARD CONSTRAINTS (mirrors o5/p3_b0_identity_test.py, o6, p3_twin_test.py):**

1. Never end a turn to wait on an untracked process -- every ``evaluate()`` call below is
   synchronous/blocking (``run_mirror_seed_inprocess``).
2. Seeds run SEQUENTIALLY within one invocation -- no subprocess/process-pool fan-out.
3. A22 stamp WRITTEN before the ``evaluate()`` call (row #173 amendment).
4. PA-CA-11 (out-root guard): a seed's ``<subdir>_meta.json`` existing is REUSE, never
   silent re-run, per the o5/o6 idempotency precedent -- disclosed on every reuse.
5. No STOP is bypassed: this driver does not itself run the fleet in this task (PA-2D-2's
   instrument STOP is still in force pending the mass-integral fix's own validation) -- it is
   built, ruff/mypy-clean, and smoke-tested on zero-compute paths only, per the launching task.

Usage:
    uv run python results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py --stage pilot
    uv run python .../p3_2d_fleet.py --stage fleet --arm bc
    uv run python .../p3_2d_fleet.py --stage fleet --arm bt
    uv run python .../p3_2d_fleet.py --stage gates
    uv run python .../p3_2d_fleet.py --stage lhs2d
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
from scipy.stats import chi2

REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))  # o5 (p3_b0_identity_test) is a sibling script, not a package

import p3_b0_identity_test as o5  # noqa: E402

from darksiren_emri.validation import correspondence_1d as c1d  # noqa: E402
from darksiren_emri.validation.correspondence_1d import H_TRUE  # noqa: E402

REGISTRATION_SECTION: str = (
    "results/campaign51_20260728/realistic_20260729/PREREGISTRATION_P3_2D_20260825.md "
    "(2026-08-25, PA-2D-1/PA-2D-2 amended); "
    "A20_REVIEW_P3_2D_DESIGN_20260825.md (F2/F4/F7-F16 adopted verbatim)"
)

H_GEN: float = H_TRUE  # 0.73, "all reads at h = H_TRUE" (prereg S1)
H_BOUNDS: tuple[float, float] = (0.50, 0.86)  # PA-CA-10 pin, carried per F16
VENUE: str = "b0i2d"
CENTER: str = "eff"  # F2 ruling: production gaussian-branch centering (_host_M_eff)

FLEET_SEEDS: tuple[int, ...] = c1d.ARM_SEEDS["b0i2d"]  # 24 seeds, 900101-900124 (PA-2D-1 F14)
PILOT_SEED: int = 900101

# The 2D axis this driver's two arms differ on (PREREGISTRATION_P3_2D_20260825.md S2).
ARM_FLAGS_2D: dict[str, str] = {
    "bc": "off",  # B2-C (coded)
    "bt": "mz_sel",  # B2-T (twin-2D)
}

OUT_ROOT_DEFAULT: Path = THIS_DIR / "p3_2d_work"
COMPANION_JSON_DEFAULT: Path = THIS_DIR / "ca_rhs_work2d" / "p3_2d_companion.json"
# GATE ACC-extended's own independent-replay scratch catalogue cache (host_pool_for_sigma_scale
# needs a directory to build/cache the sigma_z_scale=1.0 catalogue variant into -- byte-identical
# to the one the fleet task itself already built, so this is a cheap re-materialization, never a
# fresh draw).
ACC_SCRATCH_ROOT: Path = OUT_ROOT_DEFAULT / "_acc_gate_scratch"

# GATE M2-LINK (F11) executable form's own numbers.
M2_LINK_MAHALANOBIS_ALPHA: float = 1.0e-3  # "the chi^2_2 quantile at 1 - 1e-3/N_events"
M2_LINK_MONSTER_LN_THRESHOLD: float = -50.0  # "ln L_cat_with_bh - ln L_cat_no_bh < -50"

# Latent-provenance columns draw_realization's "catalogue_selected_2d" branch writes
# (correspondence_1d.py :2219-2242) -- GATE M2-LINK part (i)'s "harness writes the latent
# provenance triple" predicate.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "host_galaxy_index",
    "z_true",
    "M_true",
    "M_z_true",
    "M_z_obs",
    "M",
    "s4d_at_truth",
    "link_id",
    "luminosity_distance",
    "delta_luminosity_distance_delta_luminosity_distance",
    "delta_M_delta_M",
    "delta_luminosity_distance_delta_M",
)


def _a22_stamp_2d(arm: str) -> dict[str, Any]:
    """[ORCH] A22 = FIVE resolved flag values (PA-2D-1 F7) + git commit + dirty, BEFORE
    any ``evaluate()`` call. ``arm`` selects this thread's own axis value (item 4).
    """
    if arm not in ARM_FLAGS_2D:
        raise ValueError(f"unknown arm {arm!r}; must be one of {sorted(ARM_FLAGS_2D)}")
    git_stamp = o5._a22_stamp()
    return {
        **git_stamp,
        "catalogue_global_selection": "phi",
        # PA-2D-6: the 1D twin PRODUCTION DEFAULT resolved value (F7: "never
        # 'auto'"); was stale-pinned "off" (pre-adoption) until 2026-08-25.
        "catalogue_numerator_survival": "phi",
        "selection_in_completion_numerator": "fused",
        # PA-2D-4 item 1: sixth resolved stamp (the adopted symmetric window).
        "mass_filter_sigma": "symmetric",
        "catalogue_numerator_survival_2d": ARM_FLAGS_2D[arm],
        "catalogue_numerator_survival_2d_center": CENTER,
    }


def _rhs_f2_bit_check(events: pd.DataFrame, crb_csv: Path) -> dict[str, Any]:
    """GATE M2-LINK part (i)'s RHS-F2 half: bit-level confirmation that what got WRITTEN to
    ``prepared_cramer_rao_bounds.csv`` (and hence what ``evaluate()`` reads) matches the
    in-memory drawn ``events`` exactly, on every provenance-relevant column.

    ``write_mirror_crb_csv`` (correspondence_1d.py :2669-2680) writes ``events`` verbatim, and
    ``evaluate()`` consumes exactly that CSV -- this is the "did the write/read round-trip drop or
    perturb the drawn latents" check the registered form (F11(i)) calls "bit-level consumption".
    """
    missing = set(PROVENANCE_COLUMNS) - set(events.columns)
    if missing:
        return {"pass": False, "reason": f"missing provenance columns: {sorted(missing)}"}
    reread = pd.read_csv(crb_csv)
    if len(reread) != len(events):
        return {
            "pass": False,
            "reason": f"row count mismatch: drawn {len(events)} vs written+reread {len(reread)}",
        }
    max_rel = 0.0
    tiny = np.finfo(float).tiny
    for col in PROVENANCE_COLUMNS:
        a = events[col].to_numpy(dtype=np.float64)
        b = reread[col].to_numpy(dtype=np.float64)
        rel = np.abs(a - b) / np.maximum(np.abs(a), tiny)
        max_rel = max(max_rel, float(np.max(rel)))
    return {"pass": bool(max_rel == 0.0), "max_rel_dev": max_rel}


def _mahalanobis_check(events: pd.DataFrame, h: float, n_events: int) -> dict[str, Any]:
    """GATE M2-LINK part (ii): standardized latent-vs-datum residual, ``max Mahalanobis^2 <=``
    the ``chi^2_2`` quantile at ``1 - 1e-3/N_events`` over the fleet (F11(ii)).
    """
    d_l_true = np.asarray(
        c1d.dist_vectorized(events["z_true"].to_numpy(dtype=np.float64), h=h), dtype=np.float64
    )
    m_z_true = events["M_z_true"].to_numpy(dtype=np.float64)
    resid_d = events["luminosity_distance"].to_numpy(dtype=np.float64) - d_l_true
    resid_m = events["M"].to_numpy(dtype=np.float64) - m_z_true  # "M" == M_z_obs (:2240)

    var_dl = events["delta_luminosity_distance_delta_luminosity_distance"].to_numpy(
        dtype=np.float64
    )
    var_m = events["delta_M_delta_M"].to_numpy(dtype=np.float64)
    cov_dlm = events["delta_luminosity_distance_delta_M"].to_numpy(dtype=np.float64)
    det = var_dl * var_m - cov_dlm * cov_dlm
    ok = det > 0.0
    maha2 = np.full(events.shape[0], np.nan, dtype=np.float64)
    inv00 = var_m[ok] / det[ok]
    inv11 = var_dl[ok] / det[ok]
    inv01 = -cov_dlm[ok] / det[ok]
    maha2[ok] = (
        resid_d[ok] ** 2 * inv00
        + 2.0 * resid_d[ok] * resid_m[ok] * inv01
        + resid_m[ok] ** 2 * inv11
    )
    threshold = float(chi2.ppf(1.0 - M2_LINK_MAHALANOBIS_ALPHA / n_events, df=2))
    max_maha2 = float(np.nanmax(maha2)) if np.any(ok) else float("nan")
    return {
        "pass": bool(np.isfinite(max_maha2) and max_maha2 <= threshold),
        "max_mahalanobis_sq": max_maha2,
        "chi2_2_threshold": threshold,
        "n_singular_covariance": int((~ok).sum()),
    }


def _monster_absence_check(diag_csv: Path, h: float) -> dict[str, Any]:
    """GATE M2-LINK part (iii): zero LIVE events with
    ``ln L_cat_with_bh - ln L_cat_no_bh < -50`` (F11(iii)); "the banked monsters sit at -37 to
    -224 decades while linked draws need a >10sigma mass residual to reach -50 nats".
    """
    at = o5._rows_at_h(diag_csv, h)
    live = at["L_cat_no_bh"].to_numpy(dtype=np.float64) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_ratio = np.log(at["L_cat_with_bh"].to_numpy(dtype=np.float64)) - np.log(
            at["L_cat_no_bh"].to_numpy(dtype=np.float64)
        )
    ln_ratio_live = ln_ratio[live]
    n_monster = int(np.sum(ln_ratio_live < M2_LINK_MONSTER_LN_THRESHOLD))
    return {
        "pass": n_monster == 0,
        "n_monster": n_monster,
        "n_live": int(live.sum()),
        "threshold_nats": M2_LINK_MONSTER_LN_THRESHOLD,
    }


def gate_m2_link(events: pd.DataFrame, crb_csv: Path, diag_csv: Path, h: float) -> dict[str, Any]:
    """GATE M2-LINK, the full three-part registered form (F11)."""
    part1 = _rhs_f2_bit_check(events, crb_csv)
    part2 = _mahalanobis_check(events, h, n_events=int(events.shape[0]))
    part3 = _monster_absence_check(diag_csv, h)
    overall = bool(part1.get("pass") and part2.get("pass") and part3.get("pass"))
    return {
        "gate": "M2-LINK",
        "reference": f"{REGISTRATION_SECTION}, F11",
        "part_i_rhs_f2_provenance_bitcheck": part1,
        "part_ii_mahalanobis": part2,
        "part_iii_monster_absence": part3,
        "pass": overall,
    }


def gate_acc_extended(
    seed: int,
    completeness: Any,
    phi_survival_table: dict[float, Any],
    detection_probability: Any,
    n_events: int,
    h: float = H_GEN,
) -> dict[str, Any]:
    r"""GATE ACC-extended (F12): independently replay
    :func:`correspondence_1d._draw_2d_accepted_latents` with the SAME seed and inputs
    ``draw_realization``'s ``"catalogue_selected_2d"`` branch uses internally (byte-identical
    accepted latents by construction -- same deterministic rng stream, same call) to recover the
    ``n_drawn_total``/``n_rounds`` diagnostics ``draw_realization``'s public return (a DataFrame
    only) discards. Reports :math:`\bar p_s = n/n_{drawn\_total}` and its normal-approximation
    binomial CI -- **reporting only** (F12: "the 1D fleet rate 0.5821 is NOT a reference and
    n_kept may legitimately move"; there is no external fixed target to PASS/FAIL against under
    the extended class-G law).
    """
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=1.0, area_scale=1.0)
    gen = c1d.MirrorUniverseGenerator(cfg)
    pool, _observed_path, _handler = gen.host_pool_for_sigma_scale(
        ACC_SCRATCH_ROOT / f"seed{seed}", seed, sigma_z_scale=1.0
    )
    host_w, _w_g, s_tilde_phi = c1d.catalogue_selected_host_draw_weights(
        pool, phi_survival_table, completeness, h=H_TRUE
    )
    rng = np.random.default_rng(seed)
    # Byte-identical replay of draw_realization's OWN rng-consumption order (correspondence_1d.py
    # :1997-2001): the SNR-weighted donor-row draw runs FIRST on every host_mode branch, consuming
    # `n_events` choices from the SAME stream BEFORE the host_mode-specific draw -- confirmed
    # precedent: run_d1_premise_check (:3516-3522) replays this identical step for the SAME reason
    # (an independent out-of-band recompute needing byte-identical downstream draws). Skipping this
    # step would desynchronize the rng from draw_realization's own state and NOT reproduce the same
    # accepted latents -- caught by this file's own smoke test before being reported as a fix.
    snr = gen._donor_rows["SNR"].to_numpy(dtype=np.float64)
    row_p = snr / snr.sum()
    rng.choice(len(gen._donor_rows), size=n_events, replace=False, p=row_p)
    latents = c1d._draw_2d_accepted_latents(
        rng,
        pool,
        host_w,
        s_tilde_phi,
        phi_survival_table,
        completeness,
        detection_probability,
        n_events,
        h=H_TRUE,
    )
    p_bar_s = n_events / latents.n_drawn_total if latents.n_drawn_total > 0 else float("nan")
    se = float(np.sqrt(p_bar_s * (1.0 - p_bar_s) / max(latents.n_drawn_total, 1)))
    z95 = 1.959963984540054
    return {
        "gate": "ACC-extended",
        "reference": f"{REGISTRATION_SECTION}, F12",
        "seed": seed,
        "n_events": n_events,
        "n_drawn_total": int(latents.n_drawn_total),
        "n_rounds": int(latents.n_rounds),
        "p_bar_s": p_bar_s,
        "p_bar_s_ci95": [p_bar_s - z95 * se, p_bar_s + z95 * se],
        "note": (
            "REPORTING gate (F12): the pre-2D 1D fleet rate 0.5821 is explicitly NOT a reference "
            "under the extended class-G law -- n_kept/p_bar_s may legitimately move."
        ),
    }


def _gate_f10c_zmarginal_proxy(
    b0i2d_z_true: npt.NDArray[np.float64], b0i_z_true_csv: Path | None
) -> dict[str, Any]:
    """F10(c) PROXY companion check -- disclosed partial scope.

    The REGISTERED F10(c) gate ("the extended RHS2 scorer's accepted-z histogram must match the
    1D scorer's") is owned by ``ca_rhs_scorer.py``'s 2D extension (prereg S2 instrument (iii)),
    a SEPARATE, not-yet-built instrument this file does not construct. What THIS function checks
    instead, as a cheap zero-compute companion available at the fleet-driver level: a
    two-sample Kolmogorov-Smirnov comparison of the b0i2d fleet's own drawn ``z_true`` values
    against the b0i (1D) fleet's ``z_true`` values (both realize the SAME "catalogue_selected"
    host/z draw law per F10(a) -- "the mass-law extension does not perturb the z-draw law"), when
    a b0i (1D) fleet CSV carrying a ``z_true`` column is available on disk. This is evidence FOR
    (not a substitute for) the registered gate: agreement here is consistent with F10(c)'s claim
    but does not itself certify the RHS2 SCORER's completion-class replay, which is a distinct
    object (the estimator's accepted-z law under evaluate(), not the generator's drawn z_true).
    """
    if b0i_z_true_csv is None or not b0i_z_true_csv.is_file():
        return {
            "pass": None,
            "reason": (
                "no b0i (1D) fleet CSV with a z_true column found -- proxy skipped, disclosed "
                "(the FULL F10(c) gate requires the RHS2 scorer's own completion-class replay, "
                "not built by this file; see docstring)."
            ),
            "scope": "PROXY ONLY -- see _gate_f10c_zmarginal_proxy docstring",
        }
    b0i_df = pd.read_csv(b0i_z_true_csv)
    if "z_true" not in b0i_df.columns:
        return {"pass": None, "reason": f"{b0i_z_true_csv} has no z_true column"}
    from scipy.stats import ks_2samp

    stat, pvalue = ks_2samp(b0i2d_z_true, b0i_df["z_true"].to_numpy(dtype=np.float64))
    return {
        "pass": bool(pvalue > 0.01),
        "ks_statistic": float(stat),
        "ks_pvalue": float(pvalue),
        "scope": "PROXY ONLY (generator z_true histogram, not the RHS2 scorer's completion-class "
        "replay) -- see _gate_f10c_zmarginal_proxy docstring",
    }


def _run_b0i2d_arm_seed(seed: int, arm: str, out_root: Path) -> dict[str, Any]:
    """One (arm, seed) b0i2d fleet task -- mirrors ``p3_b0_identity_test._run_arm_seed`` (PA-2D-2's
    registered fix), threading all FIVE resolved flags directly into
    :func:`c1d.run_mirror_seed_inprocess`, bypassing ``run_arm_seed``'s threading gap entirely.
    """
    subdir = f"{arm}_{seed}"
    meta_path = out_root / f"{subdir}_meta.json"
    if meta_path.is_file():
        print(f"seed {seed} ({arm}): REUSING existing {subdir}_meta.json (disclosed, PA-CA-11)")
        reused_meta: dict[str, Any] = json.loads(meta_path.read_text())
        return reused_meta

    work_root = out_root / f"{subdir}_work"
    work_root.mkdir(parents=True, exist_ok=True)

    sigma_z_scale, area_scale = c1d.ARM_SPECS[VENUE]
    assert c1d.ARM_HOST_MODE[VENUE] == "catalogue_selected_2d", (
        "interface assumption violated: c1d.ARM_HOST_MODE['b0i2d'] != 'catalogue_selected_2d' -- "
        "the venue registry changed since this driver was written -- STOP (A21)"
    )
    assert c1d.ARM_SELECTION_CELL[VENUE] == "fused", (
        "interface assumption violated: c1d.ARM_SELECTION_CELL['b0i2d'] != 'fused' -- STOP (A21)"
    )
    catalogue_pin_ok = c1d.check_reduced_catalogue_pin()
    cfg = c1d.CorrespondenceConfig(sigma_z_scale=sigma_z_scale, area_scale=area_scale)
    gen = c1d.MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seed, sigma_z_scale=sigma_z_scale
    )
    c1d._verify_rate_weight_parity()  # PA-2 runtime parity gate, shared with b0i/b0i2d
    completeness_obj, phi_survival_table, detection_probability_obj = (
        c1d.build_b0i_2d_selection_objects(h_true=H_GEN)
    )
    events = gen.draw_realization(
        seed,
        host_pool=host_pool,
        host_mode="catalogue_selected_2d",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
        detection_probability=detection_probability_obj,
    )

    stamp = _a22_stamp_2d(arm)  # A22: written before the evaluate() call.
    t0 = time.time()
    diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
        work_root / f"seed{seed}",
        events,
        seed,
        galaxy_catalog=handler,
        h_values=(H_GEN,),
        selection_in_completion_numerator="fused",
        # PA-2D-6 fix: production-default resolved value (F7); was "off".
        catalogue_numerator_survival="phi",
        catalogue_numerator_survival_2d=ARM_FLAGS_2D[arm],
        catalogue_numerator_survival_2d_center=CENTER,
        catalogue_global_selection="phi",
        h_bounds=H_BOUNDS,
    )
    wall_time_s = time.time() - t0

    crb_csv = work_root / f"seed{seed}" / "simulations" / "prepared_cramer_rao_bounds.csv"
    rhs_f2_check = _rhs_f2_bit_check(events, crb_csv)  # cheap, always run (M2-LINK part i, half 1)

    meta: dict[str, Any] = {
        "subdir": subdir,
        "seed": seed,
        "arm": arm,
        "venue": VENUE,
        "catalogue_numerator_survival_2d": ARM_FLAGS_2D[arm],
        "work_root": str(work_root),
        "crb_csv": str(crb_csv),
        "diagnostics_csv": str(diag_csv),
        "wall_time_s": wall_time_s,
        "elapsed_evaluate_s": elapsed,
        "catalogue_pin_ok": catalogue_pin_ok,
        "n_events": int(events.shape[0]),
        "rhs_f2_provenance_bitcheck": rhs_f2_check,
        "a22_stamp": stamp,
        "git_commit": c1d._git_commit(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "diagnostics_csv"}, indent=2))
    return meta


def stage_pilot(out_root: Path) -> dict[str, Any]:
    """Seed 900101 under BOTH arms (B2-C, B2-T) -- fast sanity before the 24-seed fleet."""
    metas = {arm: _run_b0i2d_arm_seed(PILOT_SEED, arm, out_root) for arm in ARM_FLAGS_2D}
    print(
        json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "diagnostics_csv"}
                for k, v in metas.items()
            },
            indent=2,
        )
    )
    return metas


def stage_fleet(out_root: Path, arm: str, seeds: list[int] | None = None) -> dict[str, Any]:
    """One arm's fleet (24 seeds, venue b0i2d), sequential, idempotent-skip on existing meta."""
    if arm not in ARM_FLAGS_2D:
        raise SystemExit(f"REFUSED: unknown --arm {arm!r}; must be one of {sorted(ARM_FLAGS_2D)}")
    reused: list[int] = []
    ran: list[int] = []
    for seed in seeds if seeds is not None else FLEET_SEEDS:
        meta_path = out_root / f"{arm}_{seed}_meta.json"
        if meta_path.is_file():
            reused.append(seed)
            print(f"seed {seed} ({arm}): REUSING existing {arm}_{seed}_meta.json (disclosed)")
            continue
        _run_b0i2d_arm_seed(seed, arm, out_root)
        ran.append(seed)
    summary = {"arm": arm, "reused": reused, "freshly_ran": ran}
    print(json.dumps(summary, indent=2))
    return summary


def _load_fleet_metas(
    out_root: Path, arm: str, seeds: tuple[int, ...]
) -> dict[int, dict[str, Any]]:
    metas: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for seed in seeds:
        meta_path = out_root / f"{arm}_{seed}_meta.json"
        if not meta_path.is_file():
            missing.append(seed)
            continue
        metas[seed] = json.loads(meta_path.read_text())
    if missing:
        raise SystemExit(
            f"REFUSED: missing {arm} meta(s) for seeds {missing} -- run --stage fleet --arm {arm} "
            "first (or --seeds to fill the gap)."
        )
    return metas


def stage_gates(
    out_root: Path, arm: str = "bt", seeds: tuple[int, ...] | None = None
) -> dict[str, Any]:
    """GATE M2-LINK (per seed, full three-part form) + GATE ACC-extended (per seed, reporting) +
    the F10(c) z-marginal proxy (pooled), scored over an already-run arm's fleet metas.
    """
    seeds = seeds if seeds is not None else FLEET_SEEDS
    metas = _load_fleet_metas(out_root, arm, seeds)

    completeness_obj, phi_survival_table, detection_probability_obj = (
        c1d.build_b0i_2d_selection_objects(h_true=H_GEN)
    )

    m2_link_results: dict[int, dict[str, Any]] = {}
    acc_results: dict[int, dict[str, Any]] = {}
    pooled_z_true: list[npt.NDArray[np.float64]] = []
    for seed, meta in metas.items():
        events = pd.read_csv(meta["crb_csv"])
        m2_link_results[seed] = gate_m2_link(
            events, Path(meta["crb_csv"]), Path(meta["diagnostics_csv"]), H_GEN
        )
        acc_results[seed] = gate_acc_extended(
            seed,
            completeness_obj,
            phi_survival_table,
            detection_probability_obj,
            n_events=int(meta["n_events"]),
            h=H_GEN,
        )
        pooled_z_true.append(events["z_true"].to_numpy(dtype=np.float64))

    # The b0i (1D) fleet's own CRB CSV, if the b0i identity-test driver has already run its
    # bc arm at this seed -- carries the SAME "catalogue_selected" z_true draw law (F10(a)).
    b0i_crb_candidate = (
        THIS_DIR
        / "p3_b0_work"
        / "bc_900101_work"
        / "seed900101"
        / "simulations"
        / "prepared_cramer_rao_bounds.csv"
    )
    f10c = _gate_f10c_zmarginal_proxy(
        np.concatenate(pooled_z_true) if pooled_z_true else np.array([]),
        b0i_crb_candidate if b0i_crb_candidate.is_file() else None,
    )

    m2_link_pass = all(v["pass"] for v in m2_link_results.values())
    result = {
        "arm": arm,
        "n_seeds": len(metas),
        "M2_LINK_per_seed": m2_link_results,
        "M2_LINK_all_pass": m2_link_pass,
        "ACC_extended_per_seed": acc_results,
        "F10c_zmarginal_proxy": f10c,
    }
    print(json.dumps({"M2_LINK_all_pass": m2_link_pass, "F10c_zmarginal_proxy": f10c}, indent=2))
    return result


def _identity_inputs_2d(at: pd.DataFrame) -> dict[str, Any]:
    """The with-BH per-event weight ``w2_e = alpha_G_phi*L_cat_with_bh /
    (alpha_G_phi*L_cat_with_bh + B_num_wbh)`` (prereg S1; A20 review F3, ``combined_with_bh``'s
    own numerator form, ``bayesian_statistics.py`` :5481-5483) and the LIVE/dead accounting
    (LIVE(a)-style: ``L_cat_with_bh > 0``, mirroring PA-7's no-BH convention).
    """
    alpha_g_phi = at["alpha_G_phi"].to_numpy(dtype=np.float64)
    l_cat_wbh = at["L_cat_with_bh"].to_numpy(dtype=np.float64)
    b_num_wbh = at["B_num_wbh"].to_numpy(dtype=np.float64)

    live = l_cat_wbh > 0.0
    dead = ~live
    w2 = np.full(at.shape[0], np.nan, dtype=np.float64)
    a2 = alpha_g_phi[live] * l_cat_wbh[live]
    denom = a2 + b_num_wbh[live]
    # F16 dead-row convention (A2=0 => w2=0 => summand 1) applies to A2, not to the live mask
    # itself (LIVE is defined on L_cat_with_bh, per PA-7's mirror); guard denom==0 defensively.
    w2[live] = np.where(denom != 0.0, a2 / denom, 0.0)
    return {"w2": w2, "live": live, "dead": dead}


def stage_lhs2d(
    out_root: Path, arm: str = "bt", seeds: tuple[int, ...] | None = None
) -> dict[str, Any]:
    """LHS2 (PA-CA-1 drawn-count form): per-seed ``LHS2_s = (C2*/200) * sum_acc(1-w2_e)`` over the
    B2-T arm's fleet, fleet mean +/- SEM. Requires the companion's banked ``C2_star``
    (PA-2D-1 F4) and the arm's already-run fleet metas.
    """
    if not COMPANION_JSON_DEFAULT.is_file():
        raise SystemExit(
            f"REFUSED: missing {COMPANION_JSON_DEFAULT} -- run p3_2d_companion.py first "
            "(PA-2D-1 F4, C2* resolved)."
        )
    companion = json.loads(COMPANION_JSON_DEFAULT.read_text())
    c2_star = float(companion["C2_star"])

    seeds = seeds if seeds is not None else FLEET_SEEDS
    metas = _load_fleet_metas(out_root, arm, seeds)

    per_seed_lhs2: dict[int, float] = {}
    per_seed_diag: dict[int, dict[str, Any]] = {}
    for seed, meta in metas.items():
        at = o5._rows_at_h(Path(meta["diagnostics_csv"]), H_GEN)
        inputs = _identity_inputs_2d(at)
        n_drawn = int(meta["n_events"])  # the "200" drawn-count normalization (PA-CA-1)
        sum_acc = float(np.sum(1.0 - inputs["w2"][inputs["live"]]))
        lhs2_s = (c2_star / n_drawn) * sum_acc
        per_seed_lhs2[seed] = lhs2_s
        per_seed_diag[seed] = {
            "n_rows": int(at.shape[0]),
            "n_live": int(inputs["live"].sum()),
            "n_dead": int(inputs["dead"].sum()),
            "sum_acc_1_minus_w2": sum_acc,
        }

    vals = np.array(list(per_seed_lhs2.values()), dtype=np.float64)
    n = vals.size
    mean = float(vals.mean()) if n else None
    sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    result = {
        "arm": arm,
        "reference": f"{REGISTRATION_SECTION}, S1 (LHS2 statistic)",
        "C2_star": c2_star,
        "n_seeds": n,
        "LHS2_per_seed": per_seed_lhs2,
        "per_seed_diagnostics": per_seed_diag,
        "LHS2_mean": mean,
        "LHS2_sem": sem,
    }
    print(
        json.dumps({"C2_star": c2_star, "n_seeds": n, "LHS2_mean": mean, "LHS2_sem": sem}, indent=2)
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("pilot", "fleet", "gates", "lhs2d"), required=True)
    ap.add_argument("--arm", type=str, default=None, choices=("bc", "bt"), help="fleet/gates/lhs2d")
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated seed subset")
    ap.add_argument(
        "--out-root", type=str, default=str(OUT_ROOT_DEFAULT), help="Root scratch/output directory."
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None

    if args.stage == "pilot":
        stage_pilot(out_root)
        return 0
    if args.stage == "fleet":
        if args.arm is None:
            raise SystemExit("REFUSED: --stage fleet requires --arm {bc,bt}")
        stage_fleet(out_root, args.arm, seeds)
        return 0
    if args.stage == "gates":
        stage_gates(out_root, args.arm or "bt", tuple(seeds) if seeds else None)
        return 0
    stage_lhs2d(out_root, args.arm or "bt", tuple(seeds) if seeds else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
