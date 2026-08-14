"""Generator for Chapter 14 -- "Six Candidates, One Structure".

Ch 13 closed the unowned-residual thread as a ratified DISSOLUTION (2026-08-08)
and opened a new one: the venue-transfer campaign (`results/venue_transfer_20260811/`,
ledger row #99, BIAS_HISTORY_LEDGER.md) confirmed the sigma_z-dosed coverage
collapse survives production-matched realism and demanded a `/physics-change`
package -- which needs a *new formula*, and only the *old* one could be
written. Ch 14 is the mechanism-isolation arc that followed: six candidate
terms narrowed to a characterised (but unowned) structure, told with the
methodological beat the author asked the book to carry -- cheap experiments
that close four of six candidates before a single instrument run, a parity
argument that kills an entire mechanism class at once, a split-dose result
that inverts its own registered prediction, and a 2-D dose surface whose
"gate x amplifier" shape still has no name.

The chapter proposes NOTHING. Per the mechanism study's own bar (SCAN_READOUT.md
S6 item 7, PHYSICS_CHANGE_INTAKE_DOSSIER.md), the new-formula slot is empty
and this generator/chapter is not the place to fill it.

Outputs
-------
``book/site/data/ch14_ladder.json``
    The venue-transfer decision ladder (v2 baseline -> T-a -> T-b -> T-c(0.730))
    plus the T-0 anchor, re-read from the campaign's own scored JSON twin
    (`VENUE_TRANSFER_READOUT.json`) rather than transcribed from the prose
    readout. `killing_axis` is carried verbatim (null == no rung breaks the
    collapse). Gate: T-c(0.730) 1D bias, HPD coverage, PIT-KS D, rails,
    bias/post_sd ratio all reproduced to float precision from the same file.

``book/site/data/ch14_candidates.json``
    The six-candidate register (M1-M5, M2) with each verdict's load-bearing
    numbers, hand-transcribed from `results/mechanism_study_20260813/M{1,3,4,5}_*.md`
    and `PREREGISTRATION_MECHANISM_ISOLATION.md` S7 and gated where a number
    also appears in a machine-readable source.

``book/site/data/ch14_split_dose.json``
    Arms MN0 / MEH (host-only dose) / MEI (impostor-only dose), 1D mean bias,
    re-read from `MECHANISM_ISOLATION_READOUT.json`, against the registered
    predictions in `PREREGISTRATION_MECHANISM_ISOLATION.md` S2 (E1-imp >= 0.030,
    E1-host <= 0.012) -- the measured split inverts the registered guess.

``book/site/data/ch14_dose_surface.json``
    The 16-cell 2-D dose grid (f_host x f_imp in {0, .25, .5, 1}), 1D bias,
    re-read cell-by-cell from `score_2d_scan_output.json` (the adjudicator's
    own re-derivation, not the orchestrator's table). Carries the f_host=0
    row (exact zero, 60/60 seeds) and the branch-2 statistics (DS-D2, DS-D3)
    from `SCAN_READOUT.md`.

Determinism: no RNG. Read-only outside ``book/``.

Run as::

    /home/jasper/Repositories/darksiren-emri/.venv/bin/python \\
        book/generators/gen_ch14.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Paths -- mirrors gen_ch11.py's dual-root resolution (this checkout, or a
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


VT_REL = "results/venue_transfer_20260811"
MS_REL = "results/mechanism_study_20260813"


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
            raise SystemExit(f"gen_ch14 GATE FAILED: {name}: got {got!r}, expected {expected!r} (tol {tol}) [{cite}]")

    def summary(self) -> dict[str, Any]:
        return {"n": len(self.rows), "all_pass": all(r["pass"] for r in self.rows), "rows": self.rows}


GATES = Gates()


def rnd(x: float, n: int = 8) -> float:
    return float(round(float(x), n))


# ==========================================================================
# 1. The decision ladder (venue-transfer campaign, ledger row #99)
# ==========================================================================
def build_ladder() -> dict[str, Any]:
    vt = json.loads(need(f"{VT_REL}/VENUE_TRANSFER_READOUT.json").read_text())
    ladder_raw = vt["DS_VT5_ladder"]["ladder"]
    killing_axis = vt["DS_VT5_ladder"]["killing_axis"]

    tc730_1d = vt["scored"]["Tc_h0p730"]["1d"]
    GATES.check(
        "T-c(0.730) 1D bias_argmax",
        tc730_1d["DS-VT3"]["bias_argmax"],
        0.037237,
        5e-7,
        "VENUE_TRANSFER_READOUT.json scored.Tc_h0p730.1d.DS-VT3",
    )
    GATES.check(
        "T-c(0.730) 1D bias SE",
        tc730_1d["DS-VT3"]["bias_argmax_se"],
        0.000230,
        5e-7,
        "VENUE_TRANSFER_READOUT.json scored.Tc_h0p730.1d.DS-VT3",
    )
    GATES.check(
        "T-c(0.730) 1D bias/post_sd ratio",
        tc730_1d["delta_narrow_companion_unbanded"]["ratio_bias_over_post_sd"],
        8.510423329124277,
        1e-6,
        "VENUE_TRANSFER_READOUT.json scored.Tc_h0p730.1d.delta_narrow_companion_unbanded",
    )
    GATES.check(
        "T-c(0.730) 1D PIT-KS D",
        tc730_1d["DS-VT2"]["D"],
        1.0,
        1e-6,
        "VENUE_TRANSFER_READOUT.json scored.Tc_h0p730.1d.DS-VT2",
    )
    for level in ("hpd50", "hpd68", "hpd90"):
        GATES.check(
            f"T-c(0.730) 1D {level} coverage",
            tc730_1d["DS-VT1"]["levels"][level]["value"],
            0.0,
            0.0,
            "VENUE_TRANSFER_READOUT.json scored.Tc_h0p730.1d.DS-VT1",
        )
    GATES.check(
        "T-c(0.730) 1D rails",
        tc730_1d["DS-VT4"]["rail_low"] + tc730_1d["DS-VT4"]["rail_high"],
        0.0,
        0.0,
        "VENUE_TRANSFER_READOUT.json scored.Tc_h0p730.1d.DS-VT4",
    )

    rows = []
    for rung in ladder_raw:
        rows.append(
            {
                "rung": rung["rung"],
                "arm": rung["arm"],
                "N": rung["N"],
                "axes": rung["axes"],
                "bias_1d": rnd(rung["1d"]["bias_argmax"], 6),
                "R_dose_1d": rnd(rung["1d"]["R_dose"], 4),
                "classification_1d": rung["1d"]["classification"],
                "bias_2d": rnd(rung["2d"]["bias_argmax"], 6),
                "R_dose_2d": rnd(rung["2d"]["R_dose"], 4),
            }
        )

    return {
        "_meta": {
            "chapter": 14,
            "source": "results/venue_transfer_20260811/VENUE_TRANSFER_READOUT.json "
            "(scored twin of VENUE_TRANSFER_READOUT.md); ledger row #99, "
            "BIAS_HISTORY_LEDGER.md",
        },
        "ladder": rows,
        "killing_axis": killing_axis,
        "decision_cell": {
            "arm": "T-c(0.730)",
            "N": 400,
            "bias_1d": 0.037237,
            "bias_1d_se": 0.000230,
            "hpd_coverage": [0.0, 0.0, 0.0],
            "pit_ks_D": 1.0,
            "rails": [0.0, 0.0],
            "bias_over_post_sd_1d": rnd(tc730_1d["delta_narrow_companion_unbanded"]["ratio_bias_over_post_sd"], 3),
            "branch": "TRANSFER-CONFIRMED",
            "ratified": "2026-08-13",
        },
    }


# ==========================================================================
# 2. The six-candidate register
# ==========================================================================
def build_candidates() -> dict[str, Any]:
    # Anchors from the mechanism-isolation readout numbers, cross-checked
    # against the JSON where one exists (M3's factor and M1's sign are both
    # prose-derivations without a machine-readable twin; transcribed with
    # the source file + line-anchored claim quoted in "evidence").
    candidates = [
        {
            "id": "M3",
            "name": "h-dependent truncation of the unrenormalised z-kernel",
            "verdict": "REFUTED",
            "route": "amplitude and dose-trend",
            "evidence": "Implied MAP displacement +6.0e-7 in h against the observed "
            "+3.72e-2 -- a factor 6.2e4 short. Scaling DECREASES with dose "
            "(2.3e-6 / 8.1e-7 / 6.5e-7 at sigma_z = 0.011/0.035/0.042), the "
            "wrong trend by ~13x across a factor-4 dose change.",
            "source": "M3_truncation_window.md",
        },
        {
            "id": "M4",
            "name": "selection normalisation alpha(h) is sigma_z-blind",
            "verdict": "REFUTED",
            "route": "the missing term is identically 1",
            "evidence": "The correction to ln alpha that M4 says is missing is exactly "
            "zero at every sigma_z (exactness argument). Deleting alpha(h) "
            "entirely from 2,400 already-stored posteriors leaves a "
            "sigma_z-keyed bias of +0.0165 at sigma_z=0.035 (from +0.0353) -- "
            "still ~linear in dose. The keying survives total deletion of the "
            "term M4 accuses.",
            "source": "M4_alpha_sigma_blindness.md",
        },
        {
            "id": "M1",
            "name": "missing comoving-volume / rate prior in the host-z kernel",
            "verdict": "REFUTED AS SOLE MECHANISM",
            "route": "sign",
            "evidence": "Expanding the Bayes-correct E[z_true|z_obs] against the local "
            "population prior gives Delta z = sigma_z^2 * lambda, lambda = "
            "d ln p_pop/dz > 0 throughout the venue's population "
            "(median z=0.494) -- M1 predicts H0 biased LOW by ~0.02-0.04, the "
            "opposite sign of the observed defect. Retained as a compounding "
            "negative quadratic term (fitted a=+1.15, b=-5.29) that matches "
            "the ladder's own R_dose drift (1.069 -> 1.008 -> 0.877-0.913).",
            "source": "M1_missing_volume_prior.md",
        },
        {
            "id": "M5",
            "name": "equal-weight 1/K candidate prior over a smeared population",
            "verdict": "REFUTED AS STATED",
            "route": "attribution",
            "evidence": "The prior half is exonerated: replacing 1/K with rate weights, "
            "oracle weights, or a window-renormalised prior changes the bias "
            "by +1% to +30% (never attenuates). With ZERO population scatter "
            "(candidates at their true z, estimator kernel still on), 76% of "
            "the bias survives -- the smeared population is not necessary. "
            "A modified carrier M5' (the estimator's own over-broad effective "
            "candidate measure) survives all three registered constraints.",
            "source": "M5_smeared_candidate_prior.md",
        },
        {
            "id": "M2",
            "name": "missing Jacobian in the point-distance term",
            "verdict": "CLOSED -- REFUTED",
            "route": "the T-0 anchor",
            "evidence": "A missing point-distance Jacobian would bias the T-0 anchor "
            "(sigma_z = 0, real events and real K); T-0 is clean at all 200 "
            "seeds argmax exactly on truth, zero rails.",
            "source": "PREREGISTRATION_MECHANISM_ISOLATION.md S7",
        },
    ]
    return {
        "_meta": {
            "chapter": 14,
            "source": "results/mechanism_study_20260813/M1_missing_volume_prior.md, "
            "M3_truncation_window.md, M4_alpha_sigma_blindness.md, "
            "M5_smeared_candidate_prior.md, PREREGISTRATION_MECHANISM_ISOLATION.md S7",
        },
        "candidates": candidates,
        "closed_before_instrument_run": ["M1", "M2", "M3", "M4"],
        "parity_argument": {
            "statement": "Gaussian convolution is exp(sigma^2 d^2/2), an expansion in "
            "EVEN powers of sigma only. Every kernel-mismatch 'we convolved "
            "wrong' story is therefore O(sigma^2) at leading order and "
            "predicts R_dose proportional to sigma -- a 3.5x change across "
            "the B1->B2 dose lever.",
            "predicted_ratio": 3.5,
            "measured_R_dose": [1.103, 1.012],
            "measured_ratio": 0.92,
            "source": "PREREGISTRATION_MECHANISM_ISOLATION.md S7 (line ~297-300)",
        },
        "cost": {
            "l0_closures_before_any_run": 4,
            "single_seed_sigma_significance": 7.0,
            "scan_total_cpu_hours": 177.8,
            "scan_total_seeds": 325,
            "scan_budgeted_cpu_hours": 259,
            "source": "SCAN_READOUT.md (D-8), PREREGISTRATION_MECHANISM_ISOLATION.md (line ~83)",
        },
    }


# ==========================================================================
# 3. Split-dose result (E1-host / E1-imp) and its inversion
# ==========================================================================
def build_split_dose() -> dict[str, Any]:
    mi = json.loads(need(f"{MS_REL}/MECHANISM_ISOLATION_READOUT.json").read_text())
    mn0 = mi["arms"]["MN0"]["channels"]["1d"]["mean_bias"]
    meh = mi["arms"]["MEH"]["channels"]["1d"]["mean_bias"]
    mei = mi["arms"]["MEI"]["channels"]["1d"]["mean_bias"]

    GATES.check("MN0 1D mean bias", mn0, 0.034667, 5e-6, "MECHANISM_ISOLATION_READOUT.json arms.MN0")
    GATES.check("MEH (E1-host) 1D mean bias", meh, 0.004000, 5e-6, "MECHANISM_ISOLATION_READOUT.json arms.MEH")
    GATES.check("MEI (E1-imp) 1D mean bias", mei, 0.0, 5e-6, "MECHANISM_ISOLATION_READOUT.json arms.MEI")

    return {
        "_meta": {
            "chapter": 14,
            "source": "results/mechanism_study_20260813/MECHANISM_ISOLATION_READOUT.json "
            "arms.{MN0,MEH,MEI}; predictions from PREREGISTRATION_MECHANISM_ISOLATION.md S2",
        },
        "arms": [
            {
                "id": "MN0",
                "label": "N-0 (null, both host and impostors dosed)",
                "predicted": "reproduce the campaign, +0.037 +/- 0.002",
                "measured_1d_bias": rnd(mn0, 6),
            },
            {
                "id": "MEI",
                "label": "E1-imp (impostors dosed, host exact)",
                "predicted": ">= +0.030 if M5' carries",
                "measured_1d_bias": rnd(mei, 6),
            },
            {
                "id": "MEH",
                "label": "E1-host (host dosed, impostors exact)",
                "predicted": "<= +0.012 if M5' carries",
                "measured_1d_bias": rnd(meh, 6),
            },
        ],
        "additivity_check": {
            "meh_plus_mei": rnd(meh + mei, 6),
            "mn0": rnd(mn0, 6),
            "verdict": "NON-ADDITIVE",
        },
        "note": "The registered prediction named the impostor sea as the carrier "
        "(E1-imp >= 0.030) and the host as inert (E1-host <= 0.012). The "
        "measurement inverts this: E1-imp is consistent with zero and "
        "E1-host carries the only nonzero split-arm bias -- and neither "
        "arm, nor their sum, reproduces N-0. The split is informative "
        "about non-additivity, not about which half 'carries' the effect.",
    }


# ==========================================================================
# 4. The 2-D dose surface (16-cell scan, branch 2 fired)
# ==========================================================================
DOSE_LEVELS = [0.0, 0.25, 0.5, 1.0]


def build_dose_surface() -> dict[str, Any]:
    scan = json.loads(need(f"{MS_REL}/score_2d_scan_output.json").read_text())
    cells = scan["cells"]

    grid: list[list[float]] = [[0.0] * 4 for _ in range(4)]
    cell_rows = []
    for name, c in cells.items():
        i = DOSE_LEVELS.index(c["f_h"])
        j = DOSE_LEVELS.index(c["f_i"])
        bias = c["1d"]["bias"]
        grid[i][j] = rnd(bias, 6)
        cell_rows.append(
            {
                "cell": name,
                "f_host": c["f_h"],
                "f_imp": c["f_i"],
                "n": c["1d"]["n"],
                "bias_1d": rnd(bias, 6),
                "se_1d": rnd(c["1d"]["se"], 6),
            }
        )

    # f_host = 0 row: exact zero at every impostor dose, 60/60 seeds total.
    host0_biases = [cells[f"S0{j}"]["1d"]["bias"] for j in range(4)]
    host0_n = sum(cells[f"S0{j}"]["1d"]["n"] for j in range(4))
    GATES.check("f_host=0 row total bias", sum(abs(b) for b in host0_biases), 0.0, 0.0, "score_2d_scan_output.json cells.S0*")
    GATES.check("f_host=0 row total seeds", float(host0_n), 60.0, 0.0, "score_2d_scan_output.json cells.S0*")

    # S33 (both fully dosed) reproduces MN0's cell-15 scale within the
    # scan's own disjoint-seed draw (not bit-identical -- fresh seeds).
    s33 = cells["S33"]["1d"]["bias"]

    # Impostor-sea amplifier: removing the impostor sea entirely (f_imp=0
    # column) leaves +0.0047..+0.0060 across f_host in {0.25,0.5,1.0} --
    # ~15% of the fully-dosed effect at S33.
    imp0_col = [cells[f"S{i}0"]["1d"]["bias"] for i in range(1, 4)]

    return {
        "_meta": {
            "chapter": 14,
            "source": "results/mechanism_study_20260813/score_2d_scan_output.json "
            "(adjudicator's own re-derivation from per_seed records); "
            "SCAN_READOUT.md; ledger row #101",
        },
        "f_host_levels": DOSE_LEVELS,
        "f_imp_levels": DOSE_LEVELS,
        "bias_1d_grid": grid,
        "cells": cell_rows,
        "s33_bias_1d": rnd(s33, 6),
        "f_host_zero_row": {
            "biases": [rnd(b, 6) for b in host0_biases],
            "n_seeds": host0_n,
            "note": "Host is an absolute gate: the f_host=0 row is exactly "
            "+0.000000 at every impostor dose, 60/60 seeds, degenerate "
            "posterior.",
        },
        "impostor_sea_amplifier": {
            "f_imp_zero_column_biases": [rnd(b, 6) for b in imp0_col],
            "fraction_of_s33": "approximately 15%",
            "note": "Removing the impostor sea (f_imp=0) leaves "
            "+0.0047..+0.0060 across the dosed host levels -- roughly 15% "
            "of the S33 (both fully dosed) effect. The impostor sea "
            "carries the remaining ~85%, but only once the host gate is "
            "open.",
        },
        "branch": {
            "fired": "BRANCH 2 -- INTERACTION-BILINEAR",
            "ds_d2_nonadditive_S33": {"D": 0.033667, "sigma": 23.4},
            "ds_d3_shape_interaction_S23": {"b": 0.023650, "boundary": 0.01150132, "sigma_above": 28.2},
            "meaning_barred": True,
            "meaning_barred_note": "The registered strictly-bilinear product-form meaning "
            "(D = I . f_host . f_imp) is refuted by the scan's own statistics: "
            "b(S23) sits +10.33 sigma above H-INT's own point prediction "
            "(+14.64 sigma realized); H-THRESH independently refuted at "
            "17.96 sigma / 50.18 sigma. Both registered shapes are wrong; "
            "the branch fires, but its meaning clause may not be quoted.",
        },
        "registered_defect": {
            "id": "DS-D3",
            "issue": "one-sided threshold with no upper edge -- SHAPE-INTERACTION fires "
            "for any sufficiently large value, including values that refute "
            "the hypothesis it names",
            "action": "recorded, not repaired (anti-tuning, S4.7)",
        },
    }


# ==========================================================================
def write_json(name: str, payload: dict[str, Any]) -> None:
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    kb = path.stat().st_size / 1024
    print(f"  wrote {path.relative_to(BOOK_ROOT.parent)}  ({kb:.1f} KB)")
    if kb > 500:
        raise SystemExit(f"gen_ch14: {name} exceeds the 500 KB budget ({kb:.1f} KB)")


def main() -> None:
    print("gen_ch14: search roots =", [str(r) for r in SEARCH_ROOTS])
    ladder = build_ladder()
    candidates = build_candidates()
    split_dose = build_split_dose()
    dose_surface = build_dose_surface()

    gates = GATES.summary()
    ladder["_gates"] = gates
    write_json("ch14_ladder.json", ladder)
    write_json("ch14_candidates.json", candidates)
    write_json("ch14_split_dose.json", split_dose)
    write_json("ch14_dose_surface.json", dose_surface)

    print(f"  gates: {gates['n']} checks, all_pass={gates['all_pass']}")
    print(
        "  decision cell T-c(0.730) N=400 1D: bias "
        f"{ladder['decision_cell']['bias_1d']:+.6f} +/- {ladder['decision_cell']['bias_1d_se']:.6f}, "
        f"displaced {ladder['decision_cell']['bias_over_post_sd_1d']:.2f}x its own width, "
        f"killing_axis={ladder['killing_axis']!r}"
    )
    print(
        "  split-dose: MN0 {mn0:+.6f} vs MEH(host) {meh:+.6f} + MEI(imp) {mei:+.6f} "
        "-- {v}".format(
            mn0=split_dose["additivity_check"]["mn0"],
            meh=split_dose["arms"][2]["measured_1d_bias"],
            mei=split_dose["arms"][1]["measured_1d_bias"],
            v=split_dose["additivity_check"]["verdict"],
        )
    )
    print(f"  2D dose surface: branch fired = {dose_surface['branch']['fired']!r}, meaning barred")


if __name__ == "__main__":
    sys.exit(main())
