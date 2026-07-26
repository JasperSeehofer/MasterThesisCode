"""Validate the generator_marginal precomputes against the derivation anchors.

Recomputes ``W_cat`` (:func:`compute_catalog_draw_weight_total`) and the
``V_f(h)`` table (:func:`precompute_completeness_population_volume`) on the
REAL reduced GLADE+ catalogue and the frozen ``m_th`` completeness cache, and
compares them to the numeric artifacts of the approved derivation packet
(``results/lcat_h_dependence_20260725/generator_norm_Wcat.json`` /
``generator_norm_Vf_tables.json``): rel <= 1e-6 required.

Usage (repo root, dev machine)::

    uv run python scripts/check_generator_norm_precomputes.py

Exit code 0 on PASS, 1 on any FAIL. Loading the 1.7 GB catalogue takes a few
minutes; this check is deliberately a script, not a pytest test.

References:
    results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md
        section 6.1 (replication anchors) and section 10 (artifacts).
"""

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_DIR = _REPO_ROOT / "results" / "lcat_h_dependence_20260725"

_REL_TOL = 1e-6


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from master_thesis_code.bayesian_inference.bayesian_statistics import (
        compute_catalog_draw_weight_total,
        precompute_completeness_population_volume,
    )
    from master_thesis_code.cosmological_model import Model1CrossCheck
    from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler
    from master_thesis_code.galaxy_catalogue.pixel_completeness import from_cache_or_build

    with open(_ARTIFACT_DIR / "generator_norm_Wcat.json") as f:
        wcat_ref = json.load(f)
    with open(_ARTIFACT_DIR / "generator_norm_Vf_tables.json") as f:
        vf_ref = json.load(f)

    failures: list[str] = []

    # --- V_f(h) on the frozen m_th cache (fast) ------------------------------
    completeness = from_cache_or_build()
    h_values = sorted(float(k) for k in vf_ref)
    vf_table = precompute_completeness_population_volume(h_values, completeness)
    for h in h_values:
        ref = float(vf_ref[f"{h:g}"]["Vf"])
        got = vf_table[h]
        rel = abs(got - ref) / ref
        status = "PASS" if rel <= _REL_TOL else "FAIL"
        print(f"V_f(h={h:.2f}): got {got:.8e}  ref {ref:.8e}  rel {rel:.2e}  {status}")
        if rel > _REL_TOL:
            failures.append(f"V_f(h={h})")

    # --- W_cat on the real catalogue (slow: full catalogue load) -------------
    # Same construction as main.py: Model1CrossCheck bounds prune the catalogue
    # (M in [10^4.5, 10^6], z - z_err <= 1.5) — the derivation packet's own
    # prune (its first W_cat attempt failed on the ParameterSpace-vs-model
    # bounds distinction; section 8 risk 7).
    import numpy as np

    rng = np.random.default_rng(0)
    cosmological_model = Model1CrossCheck(rng=rng)
    handler = GalaxyCatalogueHandler(
        M_min=cosmological_model.parameter_space.M.lower_limit,
        M_max=cosmological_model.parameter_space.M.upper_limit,
        z_max=cosmological_model.max_redshift,
    )
    n_pruned = len(handler.reduced_galaxy_catalog)
    print(f"pruned catalogue rows: {n_pruned}  (ref {wcat_ref['pruned_n']})")
    if n_pruned != int(wcat_ref["pruned_n"]):
        failures.append("pruned_n")

    W_cat = compute_catalog_draw_weight_total(handler)
    ref_w = float(wcat_ref["W_cat"])
    rel_w = abs(W_cat - ref_w) / ref_w
    status = "PASS" if rel_w <= _REL_TOL else "FAIL"
    print(f"W_cat: got {W_cat:.8e}  ref {ref_w:.8e}  rel {rel_w:.2e}  {status}")
    if rel_w > _REL_TOL:
        failures.append("W_cat")

    # Eligible-count anchor at z < 0.992 (section 6.1 guard for prune drift).
    z = handler.reduced_galaxy_catalog["REDSHIFT"].to_numpy()
    n_0992 = int((z < 0.992).sum())
    print(f"eligible galaxies z<0.992: {n_0992}  (ref {wcat_ref['n_z0992']})")
    if n_0992 != int(wcat_ref["n_z0992"]):
        failures.append("n_z0992")

    if failures:
        print(f"FAILED checks: {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
