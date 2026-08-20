"""Registered readout for AMENDMENT A-2 arms (B-OUT adjudicating, B-F1 control).

Scores the bands registered in ``PREREGISTRATION_1D_CORRESPONDENCE.md`` A-2:

* **B-OUT** (production-regime arm: dark hosts drawn from the estimator's own
  comoving population, never inserted into the candidate set) —
  COMPLETION-UNBIASED if ``|bias| <= max(0.005, 2*SE)`` AND ``C68`` inside the
  N-binomial 95% band; COMPLETION-BIASED-LOW / -HIGH if the CI excludes 0 with
  ``|bias| >= 0.005``; MIXED otherwise.
* **B-F1** (control: B-0 configuration with the ``f == 1`` completeness shim) —
  ``|bias| <= max(0.005, 2*SE)`` confirms the completeness-mismatch attribution
  for the catalogue-resident arms.

The B-OUT verdict is the pre-registered discriminator for
``docs/derivations/population_mismatch_dark_score.md``: an UNBIASED B-OUT means
the estimator is self-consistent under its own population, so production's base
tilt is the data-vs-model population mismatch; a biased B-OUT would instead
point at a residual internal misnormalization and falsify that attribution.

Usage::

    uv run python results/prod2d_closure_20260818/readout_bout.py [--dir DIR]
"""

from __future__ import annotations  # noqa: I001  (script, not package code)

import argparse
import glob
import json
import math
import os
from typing import Any

import numpy as np
import numpy.typing as npt

TRUTH_H = 0.73
MATERIALITY = 0.005


def _load(arm: str, directory: str) -> list[dict[str, Any]]:
    """Load every banked per-seed JSON for one arm, seed-sorted."""
    out: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(directory, f"{arm}_seed*.json"))):
        with open(path) as handle:
            out.append(json.load(handle))
    return sorted(out, key=lambda d: int(d["seed"]))


def _binomial_band(n: int, p: float = 0.68) -> tuple[float, float]:
    """Normal-approximation 95% band for a coverage fraction at sample size n."""
    se = math.sqrt(p * (1.0 - p) / max(n, 1))
    return p - 1.96 * se, p + 1.96 * se


def _summarize(arm: str, seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm registered statistics (bias, SE, coverage, rail, widths)."""
    means: npt.NDArray[np.float64] = np.array([s["mean_h"] for s in seeds], dtype=np.float64)
    n = means.size
    bias = float(means.mean() - TRUTH_H)
    se = float(means.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    c68 = float(np.mean([bool(s.get("c68", False)) for s in seeds]))
    lo, hi = _binomial_band(n)
    unbiased = abs(bias) <= max(MATERIALITY, 2.0 * se)
    covered = lo <= c68 <= hi
    if arm == "bden":
        # AMENDMENT A-5 bands: the data-measure instrument, referenced to
        # B-SEL's measured bias (0.1120). MEASURE-OWNS-IT confirms
        # docs/derivations/completion_numerator_data_measure.md; MEASURE-NOT-IT
        # falsifies it and moves the hunt to D_tilde^phi's class composition.
        if unbiased and covered:
            verdict = "MEASURE-OWNS-IT"
        elif abs(bias) <= 0.5 * 0.1120:
            verdict = "MEASURE-PARTIAL"
        else:
            verdict = "MEASURE-NOT-IT"
    elif arm == "bself":
        # AMENDMENT A-4 bands: fused-convention bisection, referenced to B-SEL's
        # measured bias (0.1120). CONVENTION-OWNS-IT means the off-basis
        # numerator/denominator asymmetry IS the internal misnormalization.
        bsel_bias = 0.1120
        if unbiased and covered:
            verdict = "CONVENTION-OWNS-IT"
        elif abs(bias) <= 0.5 * bsel_bias:
            verdict = "CONVENTION-PARTIAL"
        else:
            verdict = "CONVENTION-NOT-IT"
    elif arm == "bsel":
        # AMENDMENT A-3 bands: the isolation test (model-matched in BOTH
        # population and selection).
        if unbiased and covered:
            verdict = "ESTIMATOR-SELF-CONSISTENT"
        elif abs(bias) >= MATERIALITY and abs(bias) > 2.0 * se:
            verdict = "INTERNAL-MISNORMALIZATION"
        else:
            verdict = "MIXED"
    elif unbiased and covered:
        verdict = "COMPLETION-UNBIASED"
    elif abs(bias) >= MATERIALITY and abs(bias) > 2.0 * se:
        verdict = "COMPLETION-BIASED-LOW" if bias < 0 else "COMPLETION-BIASED-HIGH"
    else:
        verdict = "MIXED"
    return {
        "arm": arm,
        "n_seeds": n,
        "bias": bias,
        "se": se,
        "mean_h": float(means.mean()),
        "sd_mean_h": float(means.std(ddof=1)) if n > 1 else float("nan"),
        "median_sigma_h": float(np.median([s["sigma_h"] for s in seeds])),
        "median_n_eff": float(np.median([s["n_eff"] for s in seeds])),
        "c50": float(np.mean([bool(s.get("c50", False)) for s in seeds])),
        "c68": c68,
        "c68_band_95": [lo, hi],
        "c90": float(np.mean([bool(s.get("c90", False)) for s in seeds])),
        "r_low": float(np.mean([bool(s.get("r_low", False)) for s in seeds])),
        "verdict": verdict,
    }


def main() -> None:
    """Score the A-2 arms and write the readout JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default="results/prod2d_closure_20260818/correspondence_arms",
        help="directory holding the per-seed arm JSONs",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {"truth_h": TRUTH_H, "arms": {}}
    for arm in ("bout", "bf1", "bsel", "bself", "bden"):
        seeds = _load(arm, args.dir)
        if not seeds:
            report["arms"][arm] = {"arm": arm, "n_seeds": 0, "verdict": "NO-DATA"}
            continue
        summary = _summarize(arm, seeds)
        report["arms"][arm] = summary
        print(
            f"{arm:5s} n={summary['n_seeds']:2d}  bias={summary['bias']:+.4f} "
            f"± {summary['se']:.4f}  mean_h={summary['mean_h']:.4f}  "
            f"sigma_h={summary['median_sigma_h']:.4f}  n_eff={summary['median_n_eff']:.0f}  "
            f"C50/68/90={summary['c50']:.2f}/{summary['c68']:.2f}/{summary['c90']:.2f}  "
            f"R_low={summary['r_low']:.2f}  -> {summary['verdict']}"
        )

    bden = report["arms"].get("bden", {})
    if bden.get("n_seeds"):
        v = bden.get("verdict")
        if v == "MEASURE-OWNS-IT":
            print(
                "\nBISECTION: the completion numerator's data measure IS the internal "
                "misnormalization — the ratio-density event term does not normalize to the "
                "denominator's measure (memo section 2 CONFIRMED). Next: /physics-change for "
                "the production default, with B-SEL's 12 banked seeds as the regression bed."
            )
        elif v == "MEASURE-PARTIAL":
            print(
                "\nBISECTION: the data measure owns PART of the defect; a second term remains "
                "(next target: D_tilde^phi's alpha_G^phi/beta_Gbar^phi class composition)."
            )
        elif v == "MEASURE-NOT-IT":
            print(
                "\nBISECTION: MEASURE-NOT-IT => completion_numerator_data_measure.md is "
                "FALSIFIED as the owner. Next target: D_tilde^phi's class composition (a dark "
                "event's numerator carries only the dark term; its denominator carries both)."
            )
    bself = report["arms"].get("bself", {})
    if bself.get("n_seeds"):
        v = bself.get("verdict")
        if v == "CONVENTION-OWNS-IT":
            print(
                "\nBISECTION: the off-basis numerator/denominator asymmetry IS the internal "
                "misnormalization; the fused convention is derived-correct => /physics-change "
                "proposal for the production default (banked off-vs-fused counterfactual is its bed)."
            )
        elif v == "CONVENTION-PARTIAL":
            print(
                "\nBISECTION: the convention owns PART of the defect (bias more than halved but "
                "still material) => fused is a component, not the whole; continue bisecting the "
                "z-integral measure/Jacobian and D_tilde^phi's class composition."
            )
        elif v == "CONVENTION-NOT-IT":
            print(
                "\nBISECTION: the convention does NOT own it => next targets are the z-integral "
                "measure/Jacobian and the alpha_G^phi/beta_Gbar^phi class composition of D_tilde^phi."
            )
    bsel = report["arms"].get("bsel", {})
    if bsel.get("verdict") == "ESTIMATOR-SELF-CONSISTENT":
        print(
            "\nISOLATION TEST: B-SEL unbiased under the model-matched population AND "
            "selection => the completion mathematics is exonerated; every observed tilt "
            "is data-vs-model mismatch (population_mismatch_dark_score.md attribution stands)."
        )
    elif bsel.get("verdict") == "INTERNAL-MISNORMALIZATION":
        print(
            "\nISOLATION TEST: B-SEL BIASED under a fully model-matched universe => a genuine "
            "internal misnormalization exists in the completion leg, reproducible at "
            "~35 min/seed. Next: bisect the completion integrand (numerator vs D_tilde^phi "
            "vs the (1-f)/S pairing)."
        )
    bout = report["arms"].get("bout", {})
    if bout.get("n_seeds"):
        # WITHDRAWN (2026-08-20, ledger row #139): B-OUT matches the estimator's
        # POPULATION but not its SELECTION (hosts drawn with no detection
        # weighting; ~196/200 pass), so it carries a data-vs-model mismatch of
        # its own and CANNOT separate internal misnormalization from mismatch.
        # Its registered band stands; no causal claim is emitted from it. The
        # isolation test is B-SEL (A-3) above.
        print(
            "\nNOTE: B-OUT's band is reported without a causal reading (row #139) — it "
            "matches the model's population but not its selection. It does establish that "
            "production's dark-class rail is REPRODUCED outside production "
            f"(B-OUT mean_h {bout.get('mean_h', float('nan')):.4f} vs production C-C 0.6001)."
        )

    out_path = os.path.join(
        os.path.dirname(args.dir.rstrip("/")) or ".", "readout_bout_output.json"
    )
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
