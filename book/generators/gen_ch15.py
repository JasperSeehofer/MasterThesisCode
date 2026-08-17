"""Generator for Chapter 15 -- "The Slot Gets Filled".

Ch 14 ended with the mechanism-isolation study's new-formula slot empty:
a characterised-but-unowned "gate x amplifier" structure on the sigma_z
dose surface, and a stated bar ("there is nothing to derive to, yet").
Ch 15 is the derivation that filled it, and everything that followed from
filling it: the L4 correct-form derivation that closes the 1D venue thread
M-OWNED, the 2D mass-channel excess that survived the 1D repair and
demanded its own investigation (L6), the channel-B mechanism that owns the
excess to ~94-106% (with a small unattributed residual carried honestly),
the correct-form 2D fix (fused g_sel) that closes the 2D venue thread
M-OWNED too, the production /physics-change proposal built on that
evidence, its adversarial-verifier amendments, its landing as a
`[PHYSICS]` commit, and the pre-registered production counterfactual that
measured what actually moved -- 1D-dominated, 2D near-inert, zero MAP
motion, a small mixture skew ruled not material. The thread closes at
ledger row #119 with a banked measurement, not a triumphant repair: the
report is explicit that the likeliest way this correction disappoints
(the #66/#67 production calibration harness) is still TO-BUILD.

The chapter proposes NOTHING new: every number here is a re-read of an
already-ratified record (ledger rows #109-#119, BIAS_HISTORY_LEDGER.md).

Outputs
-------
``book/site/data/ch15_l4_closure.json``
    The L4-DER Part 2 tilt decomposition (alpha, GW z-mass growth, exponent
    scale, window motion, leftover) and the Stage-5 registered-arm readout
    that closes the 1D venue thread, re-read from
    `L4_DER_PART2_output.json` and `score_stage5_output.json`.

``book/site/data/ch15_channel_b.json``
    The L6 c2-mirror switch decomposition (channel B ~139 nats/h, channel A
    null) and the L6-DER2 fused-g_sel premeasurement + the registered
    A-FULL-2D venue arm that closes the 2D venue thread, re-read from
    `L6_C2_SWITCH_output.json`, `L6_DER2_GSEL_PREMEASURE_output.json`,
    `gate_afull2d_premeasure_check_output.json`, and
    `score_afull2d_output.json`.

``book/site/data/ch15_fusion_proposal.json``
    The production `/physics-change` proposal's decision table ([P1]-[P5]),
    hand-transcribed from `PROPOSAL_2D_SELECTION_FUSION_20260817.md` and its
    verifier addendum (no machine-readable twin -- this is a decision
    document, not a measurement), with the ledger rows that ratified each
    item.

``book/site/data/ch15_counterfactual.json``
    The production fusion counterfactual's M-1..M-4 readout and NULL
    checks, re-read from `results/run_20260817_fusion_counterfactual/
    readout.json` (both venues) and gated to float precision against the
    prereg's own VERDICT table.

Determinism: no RNG. Read-only outside ``book/``.

Run as::

    /home/jasper/Repositories/darksiren-emri/.venv/bin/python \\
        book/generators/gen_ch15.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Paths -- mirrors gen_ch14.py's dual-root resolution (this checkout, or a
# sibling ``darksiren-emri`` checkout carrying the git-tracked results/).
# --------------------------------------------------------------------------
BOOK_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BOOK_ROOT / "site" / "data"

_HERE = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = [_HERE, _HERE.parent / "darksiren-emri"]


def res(rel: str) -> Path | None:
    for root in SEARCH_ROOTS:
        p = root / rel
        if p.exists():
            return p
    return None


def need(rel: str) -> Path:
    p = res(rel)
    if p is None:
        raise FileNotFoundError(rel)
    return p


MS_REL = "results/mechanism_study_20260813"
FC_REL = "results/run_20260817_fusion_counterfactual"


class Gates:
    """Collects pass/fail checks; a failure aborts the generator."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, name: str, got: float, expected: float, tol: float, cite: str) -> None:
        ok = abs(got - expected) <= tol
        self.rows.append(
            {"gate": name, "got": float(got), "expected": float(expected), "tol": tol, "cite": cite, "pass": ok}
        )
        if not ok:
            raise SystemExit(f"gen_ch15 GATE FAILED: {name}: got {got!r}, expected {expected!r} (tol {tol}) [{cite}]")

    def summary(self) -> dict[str, Any]:
        return {"n": len(self.rows), "all_pass": all(r["pass"] for r in self.rows), "rows": self.rows}


GATES = Gates()


def rnd(x: float, n: int = 6) -> float:
    return float(round(float(x), n))


# ==========================================================================
# 1. The L4 correct-form derivation and the 1D venue thread closure
# ==========================================================================
def build_l4_closure() -> dict[str, Any]:
    l4 = json.loads(need(f"{MS_REL}/L4_DER_PART2_output.json").read_text())
    s5 = json.loads(need(f"{MS_REL}/score_stage5_output.json").read_text())

    GATES.check(
        "L4 alpha tilt (numeric)",
        l4["alpha_tilt_numeric"],
        1400.5632792018937,
        1e-3,
        "L4_DER_PART2_output.json alpha_tilt_numeric",
    )
    GATES.check(
        "L4 GW z-mass growth (P_sum_G identity)",
        l4["identity_block"]["P_sum_G"],
        1059.5191863213713,
        1e-3,
        "L4_DER_PART2_output.json identity_block.P_sum_G",
    )
    GATES.check(
        "Stage-5 1D tilt T(1D)",
        s5["ds_f1"]["1d"]["mean"],
        22.01906248064186,
        1e-6,
        "score_stage5_output.json ds_f1.1d.mean",
    )
    GATES.check(
        "Stage-5 1D bias",
        s5["ds_f2"]["1d"]["mean"],
        0.0010000000000000009,
        1e-9,
        "score_stage5_output.json ds_f2.1d.mean",
    )
    GATES.check(
        "Stage-5 1D coverage hpd50/68/90 restored",
        s5["ds_f3"]["1d"]["hpd50"] + s5["ds_f3"]["1d"]["hpd68"] + s5["ds_f3"]["1d"]["hpd90"],
        0.64 + 0.76 + 0.96,
        1e-9,
        "score_stage5_output.json ds_f3.1d",
    )
    GATES.check(
        "Stage-5 2D excess (ds_f4) surviving the 1D repair",
        s5["ds_f4"]["mean"],
        135.72315513879573,
        1e-3,
        "score_stage5_output.json ds_f4.mean",
    )
    GATES.check(
        "Stage-5 2D bias (not restored)",
        s5["ds_f2"]["2d"]["mean"],
        0.007600000000000007,
        1e-9,
        "score_stage5_output.json ds_f2.2d.mean",
    )

    by_dose_full = l4["by_dose"]["1.0"]

    return {
        "_meta": {
            "chapter": 15,
            "source": "results/mechanism_study_20260813/L4_DER_PART2_output.json, "
            "score_stage5_output.json; ledger rows #109, #111, BIAS_HISTORY_LEDGER.md",
        },
        "l4_decomposition_full_dose": {
            "alpha_tilt": rnd(l4["alpha_tilt_numeric"], 1),
            "gw_z_mass_growth": rnd(l4["identity_block"]["P_sum_G"], 1),
            "exponent_scale": rnd(-by_dose_full["dT_exp"]["mean"], 1),
            "window_motion": rnd(by_dose_full["dT_frozen"]["mean"], 1),
            "leftover_drift_plus_interactions": rnd(by_dose_full["leftover_drift_plus_interactions"]["mean"], 1),
            "identity": "sum G = N/h - sum(x)/h  (G_e = (1/h)(1 - D.D''/D'^2))",
            "note": "The GW z-mass growth term retro-explains A-M2' at 98.7% mass-kill "
            "(row #109 item 1). T_res is defined as the leftover (drift + "
            "interactions), not a separately-fitted residual.",
        },
        "leftover_by_dose": {
            "0.25": rnd(l4["by_dose"]["0.25"]["leftover_drift_plus_interactions"]["mean"], 1),
            "0.5": rnd(l4["by_dose"]["0.5"]["leftover_drift_plus_interactions"]["mean"], 1),
            "1.0": rnd(l4["by_dose"]["1.0"]["leftover_drift_plus_interactions"]["mean"], 1),
        },
        "stage5_1d_venue_thread": {
            "candidate": "A-FULL, FULL-F form (density-form GW factor x selected-population "
            "prior w_pop.Sbar_phi/alpha x LOO impostor weight; no Jacobian, no "
            "kernel renormalisation -- row #110 item 1)",
            "tilt_1d": rnd(s5["ds_f1"]["1d"]["mean"], 3),
            "tilt_1d_se": rnd(s5["ds_f1"]["1d"]["se"], 3),
            "band": [s5["ds_f1"]["band_lo"], s5["ds_f1"]["band_hi"]],
            "bias_1d": rnd(s5["ds_f2"]["1d"]["mean"], 4),
            "bias_1d_se": rnd(s5["ds_f2"]["1d"]["se"], 4),
            "coverage_1d": [s5["ds_f3"]["1d"]["hpd50"], s5["ds_f3"]["1d"]["hpd68"], s5["ds_f3"]["1d"]["hpd90"]],
            "coverage_nominal": [0.50, 0.68, 0.90],
            "verdict": "1D venue thread CLOSED, M-OWNED (ledger row #111)",
        },
        "stage5_2d_excess_surviving": {
            "excess_tilt": rnd(s5["ds_f4"]["mean"], 2),
            "excess_tilt_se": rnd(s5["ds_f4"]["se"], 3),
            "excess_reference_center": s5["ds_f4"]["reference"]["center"],
            "excess_reference_half_width": s5["ds_f4"]["reference"]["half_width"],
            "bias_2d": rnd(s5["ds_f2"]["2d"]["mean"], 4),
            "bias_2d_se": rnd(s5["ds_f2"]["2d"]["se"], 4),
            "coverage_2d": [s5["ds_f3"]["2d"]["hpd50"], s5["ds_f3"]["2d"]["hpd68"], s5["ds_f3"]["2d"]["hpd90"]],
            "note": "The 2D mass-channel defect survives the full 1D repair -- coverage "
            "is NOT restored. The targeted L6 investigation opens here (row #111 "
            "item 3).",
        },
    }


# ==========================================================================
# 2. Channel B, the correct-form 2D fix, and the 2D venue thread closure
# ==========================================================================
def build_channel_b() -> dict[str, Any]:
    c2 = json.loads(need(f"{MS_REL}/L6_C2_SWITCH_output.json").read_text())
    gsel = json.loads(need(f"{MS_REL}/L6_DER2_GSEL_PREMEASURE_output.json").read_text())
    gate_check = json.loads(need(f"{MS_REL}/gate_afull2d_premeasure_check_output.json").read_text())
    arm = json.loads(need(f"{MS_REL}/score_afull2d_output.json").read_text())

    GATES.check(
        "channel B (dT2_sb) measured",
        c2["aggregates"]["dT2_sb"]["mean"],
        -138.99712030596356,
        1e-3,
        "L6_C2_SWITCH_output.json aggregates.dT2_sb.mean",
    )
    GATES.check(
        "channel B pre-registered prediction",
        c2["registered_prediction"]["sb"],
        -139.0,
        1e-9,
        "L6_C2_SWITCH_output.json registered_prediction.sb",
    )
    GATES.check(
        "channel A (dT2_sa) null",
        c2["aggregates"]["dT2_sa"]["mean"],
        -6.8218966286319e-05,
        1e-8,
        "L6_C2_SWITCH_output.json aggregates.dT2_sa.mean",
    )
    base_excess = c2["aggregates"]["T2_base_minus_T1_base"]["mean"]
    residual = base_excess + c2["aggregates"]["dT2_sb"]["mean"]
    GATES.check(
        "L6 channel-B unattributed residual",
        residual,
        -7.489051685362437,
        1e-4,
        "L6_C2_SWITCH_output.json aggregates (T2_base_minus_T1_base + dT2_sb)",
    )
    GATES.check(
        "L6-DER2 gsel premeasure mirror mean",
        gsel["aggregates"]["excess_gsel"]["mean"],
        -11.739803504218184,
        1e-3,
        "L6_DER2_GSEL_PREMEASURE_output.json aggregates.excess_gsel.mean",
    )
    GATES.check(
        "L6-DER2 gate: gsel bit-exact vs afull at S=1",
        gate_check["gate_1_ln1_bit_identity_installed_gsel_vs_installed_afull"]["max_abs_diff"],
        0.0,
        0.0,
        "gate_afull2d_premeasure_check_output.json (per-event; aggregate reported)",
    )
    GATES.check(
        "A-FULL-2D arm DS-G1 mean (registered venue arm)",
        arm["ds_g1"]["mean"],
        -11.805030490228082,
        1e-3,
        "score_afull2d_output.json ds_g1.mean",
    )
    GATES.check(
        "A-FULL-2D arm DS-G1 SE",
        arm["ds_g1"]["se"],
        0.60762774559763,
        1e-3,
        "score_afull2d_output.json ds_g1.se",
    )
    GATES.check(
        "A-FULL-2D arm mirror reference (premeasure)",
        arm["ds_g1"]["mirror_reference"]["mean"],
        -11.74,
        1e-6,
        "score_afull2d_output.json ds_g1.mirror_reference.mean",
    )
    GATES.check(
        "A-FULL-2D arm DS-G5 2D bias",
        arm["ds_g5"]["2d"]["bias_mean"],
        0.0006000000000000005,
        1e-9,
        "score_afull2d_output.json ds_g5.2d.bias_mean",
    )
    GATES.check(
        "A-FULL-2D arm DS-G3 coverage restored (hpd50+68+90)",
        arm["ds_g3"]["2d"]["hpd50"] + arm["ds_g3"]["2d"]["hpd68"] + arm["ds_g3"]["2d"]["hpd90"],
        0.52 + 0.76 + 0.96,
        1e-9,
        "score_afull2d_output.json ds_g3.2d",
    )

    return {
        "_meta": {
            "chapter": 15,
            "source": "results/mechanism_study_20260813/L6_C2_SWITCH_output.json, "
            "L6_DER2_GSEL_PREMEASURE_output.json, "
            "gate_afull2d_premeasure_check_output.json, score_afull2d_output.json; "
            "ledger rows #112-#116, BIAS_HISTORY_LEDGER.md",
        },
        "channel_decomposition": {
            "base_2d_minus_1d_excess": rnd(base_excess, 3),
            "channel_b_measured": rnd(c2["aggregates"]["dT2_sb"]["mean"], 3),
            "channel_b_predicted": c2["registered_prediction"]["sb"],
            "channel_a_measured": rnd(c2["aggregates"]["dT2_sa"]["mean"], 6),
            "channel_a_predicted": c2["registered_prediction"]["sa"],
            "unattributed_residual": rnd(residual, 3),
            "channel_b_ownership_fraction": "to within ~6% (residual / base_excess)",
            "mechanism": "h-moving evaluation of completion_mass_factor_g's z-argument "
            "against the phi slope",
            "verdict": "L6 findings ratified as amended (ledger row #114): channel B owns "
            "the 2D-1D excess to within ~6%; channel A null at f=1.",
        },
        "correct_form_fix": {
            "diagnosis": "Sbar_phi x g factorization error: two integral-dM where the "
            "selected joint prior demands one integral-dM phi.p_det.N (L6-DER2, "
            "ledger row #115)",
            "candidate": "fused g_sel -- the A-FULL-2D code form",
            "premeasure_mirror_mean": rnd(gsel["aggregates"]["excess_gsel"]["mean"], 3),
            "premeasure_mirror_se": rnd(gsel["aggregates"]["excess_gsel"]["se"], 3),
            "premeasure_ownership_of_channel_b": "91.4%",
            "gate_bit_identity_pass": True,
        },
        "afull2d_venue_arm": {
            "ds_g1_tilt": rnd(arm["ds_g1"]["mean"], 3),
            "ds_g1_tilt_se": rnd(arm["ds_g1"]["se"], 3),
            "ds_g1_band": [arm["ds_g1"]["band_lo"], arm["ds_g1"]["band_hi"]],
            "mirror_prediction": arm["ds_g1"]["mirror_reference"]["mean"],
            "mirror_prediction_se": arm["ds_g1"]["mirror_reference"]["se"],
            "ds_g3_coverage_2d": [arm["ds_g3"]["2d"]["hpd50"], arm["ds_g3"]["2d"]["hpd68"], arm["ds_g3"]["2d"]["hpd90"]],
            "ds_g3_nominal": [0.50, 0.68, 0.90],
            "ds_g3_read": "necessary-but-weak (verifier MAJOR-1)",
            "ds_g5_2d_bias": rnd(arm["ds_g5"]["2d"]["bias_mean"], 4),
            "ds_g5_2d_bias_se": rnd(arm["ds_g5"]["2d"]["bias_se"], 4),
            "ds_g4_1d_bit_untouched": True,
            "verdict": "2D venue thread M-OWNED-CLOSED (ledger row #116, branch 1)",
            "budget": {"realized_cpu_h": 406.5, "allocated_cpu_h": 499, "ceiling_cpu_h": 300, "note": "overrun accepted as recorded deviation"},
        },
        "open_residual": {
            "id": "the -11.7-class residual",
            "value": "-11.7 +/- 1.0 (arm) / -11.74 +/- 1.04 (mirror), vs the c2-switch "
            "residual -7.489 +/- 0.065",
            "correlation": "r = 0.847",
            "status": "assigned to the known realization-coupled residual class; origin "
            "decomposition open (carried through row #119)",
        },
    }


# ==========================================================================
# 3. The production /physics-change proposal (decision document, no JSON twin)
# ==========================================================================
def build_fusion_proposal() -> dict[str, Any]:
    proposal_path = need("docs/derivations/PROPOSAL_2D_SELECTION_FUSION_20260817.md")
    text = proposal_path.read_text()
    if "[P1] 2D completion leg" not in text:
        raise SystemExit("gen_ch15 GATE FAILED: [P1] item text not found in proposal")
    if "S̄_φ(z;h)" not in text:
        raise SystemExit("gen_ch15 GATE FAILED: [P2] Sbar_phi(z;h) term not found in proposal")

    return {
        "_meta": {
            "chapter": 15,
            "source": "docs/derivations/PROPOSAL_2D_SELECTION_FUSION_20260817.md + "
            "PROPOSAL_2D_SELECTION_FUSION_VERIFIER_ADDENDUM_20260817.md "
            "(decision documents -- existence and key phrases checked at build time, "
            "not a numeric measurement); ledger rows #115-#118",
        },
        "claim": "Under the pipeline's own latent-thresholded detection model, the "
        "correct per-event likelihood integrates the detection survival's "
        "M-dependence against the observed-mass likelihood in ONE integral-dM. "
        "The coded absolute_marginal completion legs instead carry no survival "
        "factor in either channel's numerator (MFG/Gray denominator-only "
        "arrangement, exact only for data-deterministic detection).",
        "decision_table": [
            {
                "item": 1,
                "label": "[P1]+[P2] fused survival in both completion legs (paired)",
                "scope": "production estimator",
                "tag": "[DO]",
                "ratified_row": 117,
            },
            {
                "item": 2,
                "label": "[P3] catalogue-leg selection weighting (Gray-convention fork)",
                "scope": "production estimator + paper convention",
                "tag": "[RULE]",
                "recommendation": "defer to the Gray-convention paper task (row #110) unless "
                "the counterfactual shows the [P2]-induced mixture skew is material",
                "ratified_row": 117,
            },
            {
                "item": 3,
                "label": "[P4] measure ruling (V2 prefactor + D-ii ratio-form option C)",
                "scope": "folded into item 1's implementation",
                "tag": "[DO]",
                "ratified_row": 117,
            },
            {
                "item": 4,
                "label": "[P5-3] production counterfactual cell before any campaign re-run",
                "scope": "measurement",
                "tag": "[DO]",
                "ratified_row": 117,
            },
            {
                "item": 5,
                "label": "xhigh verifier on this proposal before item 1's implementation",
                "scope": "discipline",
                "tag": "[DO]",
                "ratified_row": 117,
            },
        ],
        "verifier_verdict": "GO-WITH-AMENDMENTS (MAJOR-1..4, MINOR-1..6)",
        "verifier_amendments": [
            {
                "id": "MAJOR-1",
                "correction": "production's completion leg sits in the SHARP-likelihood "
                "regime (measured d_L-conditional sigma_cond p50 = 8.8e-8), not the "
                "broad-sigma regime the proposal assumed -- expected action is "
                "1D-dominated ([P2]); [P1] correct-form but possibly near-inert.",
            },
            {
                "id": "MAJOR-2",
                "correction": "the pinned n_hermite=64 quadrature choice is a substantive "
                "flip of the ratified Route-1 adaptive default, not a minor rider -- "
                "returned to the author as its own ruling (G1).",
            },
            {
                "id": "MAJOR-3",
                "correction": "[P3] skew direction is inverted from the proposal's text: "
                "with [P2] on, the Sbar-free catalogue leg is OVER-weighted (not "
                "down-weighted) wherever Sbar_phi < 1.",
            },
            {
                "id": "MAJOR-4",
                "correction": "the V2 measure prefactor is provably immaterial "
                "(<~1e-6) at completion-leg sigma_cond, material only in the "
                "deferred catalogue leg -- cannot be silently folded into item 1.",
            },
        ],
        "author_rulings": {
            "row": 118,
            "G1": "Keep adaptive quadrature + guard assertion (recorded pinned-vs-adaptive "
            "regression bound ~1e-15 class; escalate to n_hermite=64 when S-variation "
            "exceeds tolerance)",
            "G2": "Retain the ratio measure convention in both legs; V2 tracked as a G7 "
            "systematics-budget row",
            "G3": "Confirm the [P3] catalogue-leg deferral on the MAJOR-3 corrected basis",
        },
        "landed": {
            "commit": "2b10b8b8",
            "kind": "[PHYSICS]",
            "scope": "fused survival in both absolute_marginal completion legs: "
            "S_bar_phi in the 1D numerator ([P2]), completion_mass_factor_g_sel "
            "(S_4D inside the mass quadrature) in the 2D leg ([P1])",
            "default_flag": "--selection_in_completion_numerator auto -> fused under "
            "absolute_marginal; off/1d/2d are counterfactual decompositions",
            "tests_passing": 1506,
            "byte_frozen_cells": ["off", "1d"],
        },
    }


# ==========================================================================
# 4. The production fusion counterfactual (M-1..M-4, both venues)
# ==========================================================================
def build_counterfactual() -> dict[str, Any]:
    ro = json.loads(need(f"{FC_REL}/readout.json").read_text())

    expect = {
        "iiib": {
            "m1_chord": 1.2449530011504242,
            "m1_central": 1.158432066665681,
            "m2_chord": 24.588259348825677,
            "m2_central": 30.90082984022044,
            "m4_mean": 0.006144920794053254,
            "m4_max": 0.20440516737591485,
        },
        "joint_r1": {
            "m1_chord": -3.2676839781970557,
            "m1_central": -2.8931849460377608,
            "m2_chord": 22.73620033359888,
            "m2_central": 32.3147749541931,
            "m4_mean": 0.005708540225406402,
            "m4_max": 0.20311970466638232,
        },
    }

    venues: dict[str, Any] = {}
    for venue, exp in expect.items():
        v = ro[venue]
        m1 = v["M1_2d_channel_tilt_P1"]
        m2 = v["M2_1d_channel_tilt_P2"]
        m3_nobh = v["M3_posteriors"]["combined_no_bh"]
        m3_wbh = v["M3_posteriors"]["combined_with_bh"]
        m4 = v["M4_mixture_skew"]
        null_sel = v["NULL_selection_side_differing_cells"]
        null1 = v["NULL_1_metadata"]

        GATES.check(f"{venue} M-1 2D chord tilt", m1["chord_nats_per_h"], exp["m1_chord"], 1e-6, "readout.json M1_2d_channel_tilt_P1")
        GATES.check(f"{venue} M-1 2D central@0.73", m1["central_diff_nats_per_h_at_073"], exp["m1_central"], 1e-6, "readout.json M1_2d_channel_tilt_P1")
        GATES.check(f"{venue} M-2 1D chord tilt", m2["chord_nats_per_h"], exp["m2_chord"], 1e-6, "readout.json M2_1d_channel_tilt_P2")
        GATES.check(f"{venue} M-2 1D central@0.73", m2["central_diff_nats_per_h_at_073"], exp["m2_central"], 1e-6, "readout.json M2_1d_channel_tilt_P2")
        GATES.check(f"{venue} M-4 mean skew@0.73", m4["delta_share_cat_at_073"]["mean"], exp["m4_mean"], 1e-9, "readout.json M4_mixture_skew")
        GATES.check(f"{venue} M-4 max skew@0.73", m4["delta_share_cat_at_073"]["max"], exp["m4_max"], 1e-6, "readout.json M4_mixture_skew")
        GATES.check(f"{venue} selection-side leak total", sum(null_sel.values()), 0.0, 0.0, "readout.json NULL_selection_side_differing_cells")

        venues[venue] = {
            "m1_2d_tilt_chord": rnd(m1["chord_nats_per_h"], 3),
            "m1_2d_tilt_central_073": rnd(m1["central_diff_nats_per_h_at_073"], 3),
            "m2_1d_tilt_chord": rnd(m2["chord_nats_per_h"], 3),
            "m2_1d_tilt_central_073": rnd(m2["central_diff_nats_per_h_at_073"], 3),
            "m3_1d_map": {"off": m3_nobh["off"]["map_h"], "fused": m3_nobh["fused"]["map_h"]},
            "m3_1d_sigma": {"off": rnd(m3_nobh["off"]["sigma_h"], 4), "fused": rnd(m3_nobh["fused"]["sigma_h"], 4)},
            "m3_1d_railed_low": {"off": m3_nobh["off"]["railed_low"], "fused": m3_nobh["fused"]["railed_low"]},
            "m3_2d_map": {"off": m3_wbh["off"]["map_h"], "fused": m3_wbh["fused"]["map_h"]},
            "m3_2d_sigma": {"off": rnd(m3_wbh["off"]["sigma_h"], 4), "fused": rnd(m3_wbh["fused"]["sigma_h"], 4)},
            "m4_mean_skew_073": rnd(m4["delta_share_cat_at_073"]["mean"], 4),
            "m4_median_skew_073": rnd(m4["delta_share_cat_at_073"]["median"], 4),
            "m4_max_skew_073": rnd(m4["delta_share_cat_at_073"]["max"], 4),
            "m4_frac_positive_073": rnd(m4["delta_share_cat_at_073"]["frac_positive"], 4),
            "null1_tasks_off": null1["off"]["tasks_checked"],
            "null1_tasks_fused": null1["fused"]["tasks_checked"],
            "null1_cell_ok": null1["off"]["cell_ok"] and null1["fused"]["cell_ok"],
            "selection_side_leak_cells": sum(null_sel.values()),
        }

    return {
        "_meta": {
            "chapter": 15,
            "source": "results/run_20260817_fusion_counterfactual/readout.json (both "
            "venues); PREREGISTRATION_FUSION_COUNTERFACTUAL.md VERDICT; "
            "CAMPAIGN_REPORT_20260817.md; ledger rows #117-#119",
        },
        "design": {
            "cells": ["off (pre-#118 estimator, fresh twin)", "fused (production)"],
            "venues": ["iiib (idealised catalogue)", "joint_r1 (realization r1, delivered catalogue)"],
            "grid": "canonical 41-point h grid",
            "n_events": 1588,
            "n_tasks": 164,
            "budget_realized_cpu_h": 170.4,
            "budget_ceiling_cpu_h": 270,
            "channel_recovery": "1D/2D single-leg cells recovered channel-wise from the "
            "pair, not re-run (test_fused_pairing_identity, bit-exact) -- halves the "
            "fleet",
            "no_bands": "A3: production magnitude was never a venue prediction -- this "
            "is a measurement seeded from nothing, not a pass/fail test",
        },
        "venues": venues,
        "context_n2_of_record": {
            "chord": [24.6, 22.7],
            "central_073": [30.9, 32.3],
            "note": "N-2 run of record (commit 0167df53) -- context, not a band. M-2 "
            "reproduces it to 3 decimals in both venues.",
        },
        "prior_bracket_m1": {"abs_le": 20.0, "source": "N-2 §3.1 unmeasured M2 band, restated by MAJOR-1"},
        "reading": {
            "regime_confirmed": "1D-dominated action (row #118 MAJOR-1 prediction confirmed)",
            "map_motion": "zero MAP motion in every channel x venue at the 41-point grid",
            "width_change_1d": {"iiib": [0.0068, 0.0053], "joint_r1": [0.0086, 0.0065]},
            "width_change_2d": {"iiib": [0.0177, 0.0178], "joint_r1": [0.0216, 0.0217]},
            "rail": "1D MAP still hard-railed at 0.600 in both venues (photo-z root "
            "cause of record, ledger #36) -- fusion narrows the posterior but does "
            "not un-rail it",
        },
        "decisions_row_119": {
            "m4_materiality": "NOT MATERIAL -- median +0.02-0.03, max +0.204 catalogue-share "
            "gain confined to the ~10% catalogue-bearing events; [P3] stays deferred "
            "to the Gray-convention paper task (row #110)",
            "campaign_rerun": "NO RE-RUN (option a) -- zero MAP/width motion means a full "
            "re-run would reproduce the campaign posteriors within their quoted "
            "widths; this run is banked as the pre/post-fusion bridge",
            "measurement_status": "banked, including the sidecar parent_csv path-repair "
            "compliance deviation (hash-verified, path-only)",
        },
        "carried_open": [
            "the -11.7-class residual (r=0.847 with the c2-switch residual; origin open)",
            "pool-vs-model prior mismatch",
            "low-dose FULL-F residual",
            "#66/#67 production calibration harness (pp_coverage mass channel, TO-BUILD "
            "-- the stated likeliest disappointment path for the landed [P2] factor)",
            "Gray-convention paper task (row #110), now holding the M-4 numbers as input",
        ],
    }


# ==========================================================================
def write_json(name: str, payload: dict[str, Any]) -> None:
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    kb = path.stat().st_size / 1024
    print(f"  wrote {path.relative_to(BOOK_ROOT.parent)}  ({kb:.1f} KB)")
    if kb > 500:
        raise SystemExit(f"gen_ch15: {name} exceeds the 500 KB budget ({kb:.1f} KB)")


def main() -> None:
    print("gen_ch15: search roots =", [str(r) for r in SEARCH_ROOTS])
    l4 = build_l4_closure()
    channel_b = build_channel_b()
    proposal = build_fusion_proposal()
    counterfactual = build_counterfactual()

    gates = GATES.summary()
    l4["_gates"] = gates
    write_json("ch15_l4_closure.json", l4)
    write_json("ch15_channel_b.json", channel_b)
    write_json("ch15_fusion_proposal.json", proposal)
    write_json("ch15_counterfactual.json", counterfactual)

    print(f"  gates: {gates['n']} checks, all_pass={gates['all_pass']}")
    print(
        "  1D venue thread: T(1D) "
        f"{l4['stage5_1d_venue_thread']['tilt_1d']:+.1f} +/- {l4['stage5_1d_venue_thread']['tilt_1d_se']:.1f}, "
        f"bias {l4['stage5_1d_venue_thread']['bias_1d']:+.4f} -- {l4['stage5_1d_venue_thread']['verdict']}"
    )
    print(
        "  2D venue thread: channel B "
        f"{channel_b['channel_decomposition']['channel_b_measured']:+.1f} nats/h "
        f"(predicted {channel_b['channel_decomposition']['channel_b_predicted']:+.1f}), "
        f"fused arm {channel_b['afull2d_venue_arm']['ds_g1_tilt']:+.2f} +/- "
        f"{channel_b['afull2d_venue_arm']['ds_g1_tilt_se']:.2f} -- {channel_b['afull2d_venue_arm']['verdict']}"
    )
    print(f"  production proposal: landed at [PHYSICS] {proposal['landed']['commit']}")
    for venue, v in counterfactual["venues"].items():
        print(
            f"  counterfactual {venue}: M-1 (2D) {v['m1_2d_tilt_chord']:+.2f}, "
            f"M-2 (1D) {v['m2_1d_tilt_chord']:+.2f}, M-4 mean skew {v['m4_mean_skew_073']:+.4f}"
        )


if __name__ == "__main__":
    sys.exit(main())
