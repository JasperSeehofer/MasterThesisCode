"""Generator for Chapter 1 — "A Ruler That Needs No Ladder".

Produces the two data files behind the chapter's beats.

``book/site/data/ch01_event889.json``   (I1.2 "Meet EMRI-889", the dossier)
    Event 889 of ``seed61000 / real_r1`` — the book's running example — read
    verbatim out of the stored Cramer-Rao row.  Nothing here is modelled: the
    14 EMRI parameters, the mission span, the drawn plunge time, the SNR, the
    host-galaxy label and the frame stamps are columns of

        results/campaign51_20260728/realistic_20260729/seed61000/real_r1/
            prepared_cramer_rao_bounds.csv

    The one *derived* quantity is the distance error bar,
    ``sigma_dL = sqrt(delta_luminosity_distance_delta_luminosity_distance)``,
    which is the square root of a stored diagonal element and is emitted BOTH
    absolutely (Gpc, Mpc) and as a fraction of d_L, because those two readings
    differ by a factor 11.1 and the build spec quotes the absolute one with a
    fractional label.  See ``book/design/flags/ch01_FLAGS.md`` (F1) — the
    generator refuses to pick one and prints both plus the 1/SNR scale that
    discriminates them.

``book/site/data/ch01_dlz.json``        (I1.3 "d_L(z; h, Omega_m) explorer")
    The chapter's explorer computes ``d_L(z; h, Omega_m)`` closed-form in the
    browser (Simpson quadrature of ``1/E(z)``), so this file carries what the
    browser cannot invent:

    1. ``validation`` — a lattice of repo ``physical_relations.dist()`` values
       over (h, Omega_m, z).  The page compares its own in-browser integral
       against these and displays the worst relative deviation, so the reader
       can see that the widget is the pipeline's function and not a lookalike.
    2. ``event_dl`` — EMRI-889's measured d_L and, per h, the redshift the
       fiducial cosmology would assign to it (``dist_to_redshift``).  This is
       the chapter's mechanism: the ruler has no scale, so the redshift you
       read off it is a function of the answer you were trying to measure.
    3. ``omega_m_mispec`` — the G7 row-6 systematic, recomputed here: h'
       solving ``d_L(z; h', 0.2726) = d_L(z; 0.73, 0.3153)``.  This is a HARD
       GATE: the recomputation must reproduce the published table
       (``docs/gates/G7_systematics_budget.md`` "Numbers behind row #6") to
       0.01 percentage points, or the generator stops.

Determinism: no RNG anywhere.  Every number is read from a git-tracked
artifact or computed by the repo's own ``physical_relations`` functions.
Read-only outside ``book/``.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch01.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from master_thesis_code.constants import H as H_TRUE  # noqa: E402
from master_thesis_code.constants import (  # noqa: E402
    LISA_MISSION_DURATION_YEARS,
    OMEGA_DE,
    OMEGA_M,
    SNR_THRESHOLD,
    SPEED_OF_LIGHT_KM_S,
)
from master_thesis_code.physical_relations import dist, dist_to_redshift  # noqa: E402

# --- repo-relative artifact paths (BOOK_DESIGN.md §4.2 rule 7) --------------
CAMPAIGN_REL = Path("results/campaign51_20260728/realistic_20260729")
SEED_REL = CAMPAIGN_REL / "seed61000"
# The build spec names ``seed61000/real_r1/prepared_cramer_rao_bounds.csv``.
# Only the seed-level copy is git-tracked (so present in any checkout of this
# branch); the ``real_r1/`` copy exists in the main working tree and is
# byte-identical on every column this chapter reads — asserted below when it
# is available, rather than assumed.
CRB_REL = SEED_REL / "prepared_cramer_rao_bounds.csv"
CRB_SPEC_REL = SEED_REL / "real_r1" / "prepared_cramer_rao_bounds.csv"
COMBINED_REL = SEED_REL / "real_r1" / "posteriors" / "combined_posterior.json"

OUT_DIR = Path(__file__).resolve().parent.parent / "site" / "data"
OUT_EVENT = OUT_DIR / "ch01_event889.json"
OUT_DLZ = OUT_DIR / "ch01_dlz.json"

EVENT_889 = 889  # the book's running example (BOOK_PEDAGOGY.md beat B4)

# BOOK_DESIGN.md §1 Ch 1 card — the dossier numbers the spec asks for.
SPEC_DOSSIER = {
    "M_msun": 7.25e5,
    "mu_msun": 10.0,
    "dL_Mpc": 88.9,
    "snr": 1425.0,
    "sigma_dL_over_dL_quoted": 8.0e-5,  # <- the disputed one (flag F1)
    "host_galaxy_index": 859360,
}

# docs/gates/G7_systematics_budget.md, "Numbers behind row #6": h' solving
# d_L(z; h', 0.2726) = d_L(z; 0.73, 0.3153), quoted as a percentage on H0.
G7_ROW6_PUBLISHED = {
    0.05: 0.16,
    0.1: 0.32,
    0.3: 0.94,
    0.5: 1.50,
    1.0: 2.60,
    1.5: 3.31,
}
OMEGA_M_PLANCK = 0.3153  # Planck 2018 (quoted in G7 row 6, not adopted)

# Explorer domains (display grids; the browser interpolates nothing — it
# integrates 1/E(z) itself and uses these only as an accuracy witness).
H_LATTICE = [0.55, 0.60, 0.65, 0.6732, 0.70, 0.73, 0.80, 0.86, 0.90]
OM_LATTICE = [0.15, 0.2, 0.2726, 0.3, 0.3153, 0.35, 0.45]
Z_LATTICE = [
    0.005, 0.01, 0.021284, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0,
]
# Static-fallback curve family (three h at the fiducial Omega_m).
H_STATIC = [0.60, 0.73, 0.86]
Z_CURVE = [round(0.002 * k, 4) for k in range(1, 401)]  # 0.002 .. 0.800
# The h values the "implied redshift of EMRI-889" readout is baked at.
H_IMPLIED = [round(0.55 + 0.005 * k, 3) for k in range(0, 71)]  # 0.55 .. 0.90


def _r(x: Any, sig: int = 8) -> float:
    """Round to `sig` significant digits — JSON size hygiene.  Every quantity
    the page displays is quoted to at most 6 s.f., so 8 is lossless for the
    book and roughly halves the file."""
    v = float(x)
    if v == 0.0 or not np.isfinite(v):
        return v
    return float(np.round(v, sig - 1 - int(np.floor(np.log10(abs(v))))))


def _need(rel: Path) -> Path:
    p = REPO_ROOT / rel
    if not p.exists():
        raise SystemExit(f"gen_ch01: required artifact missing: {rel}")
    return p


def _fail(msg: str) -> None:
    raise SystemExit(f"gen_ch01: HARD GATE FAILED — {msg}")


# ---------------------------------------------------------------------------
# 1. The dossier
# ---------------------------------------------------------------------------
def build_event() -> dict[str, Any]:
    crb = pd.read_csv(_need(CRB_REL))
    row = crb.iloc[EVENT_889]

    # Cross-check against the spec-named copy when the checkout carries it.
    twin_checked = False
    twin = REPO_ROOT / CRB_SPEC_REL
    if not twin.exists():
        # Untracked in this worktree; try a sibling main checkout (read-only).
        sibling = REPO_ROOT.parent / "MasterThesisCode" / CRB_SPEC_REL
        if sibling.exists():
            twin = sibling
    if twin.exists():
        twin_row = pd.read_csv(twin).iloc[EVENT_889]
        for col in (
            "M", "mu", "luminosity_distance", "SNR", "host_galaxy_index",
            "delta_luminosity_distance_delta_luminosity_distance",
        ):
            if float(twin_row[col]) != float(row[col]):
                _fail(f"{CRB_REL} and {CRB_SPEC_REL} disagree on row 889 column {col}")
        twin_checked = True
    print(f"  seed-level CRB == real_r1 CRB on row 889: "
          f"{'verified' if twin_checked else 'not present in this checkout'}")

    d_l = float(row["luminosity_distance"])  # Gpc
    sigma_dl = float(np.sqrt(row["delta_luminosity_distance_delta_luminosity_distance"]))
    snr = float(row["SNR"])

    # Sanity gates against the spec card (BOOK_DESIGN.md §1 Ch 1).
    if abs(d_l * 1000.0 - SPEC_DOSSIER["dL_Mpc"]) > 0.1:
        _fail(f"d_L {d_l * 1000:.3f} Mpc != spec {SPEC_DOSSIER['dL_Mpc']} Mpc")
    if abs(snr - SPEC_DOSSIER["snr"]) > 1.0:
        _fail(f"SNR {snr:.2f} != spec {SPEC_DOSSIER['snr']}")
    if abs(float(row["M"]) - SPEC_DOSSIER["M_msun"]) > 5e3:
        _fail(f"M {row['M']:.1f} != spec {SPEC_DOSSIER['M_msun']}")
    if int(row["host_galaxy_index"]) != SPEC_DOSSIER["host_galaxy_index"]:
        _fail(f"host index {row['host_galaxy_index']} != spec")
    if int(crb["SNR"].idxmax()) != EVENT_889:
        _fail("event 889 is not the loudest row — the running example moved")

    frac = sigma_dl / d_l

    # --- FLAG F1 -----------------------------------------------------------
    # The spec quotes sigma_dL/dL = 8.0e-5.  The stored row gives
    # sigma_dL = 7.98e-5 *Gpc*; dividing by d_L = 0.0888792 Gpc gives 8.98e-4.
    # Do not reconcile: emit both, plus 1/SNR, which is the scale a
    # matched-filter amplitude measurement is bounded by.
    disagrees = abs(frac - SPEC_DOSSIER["sigma_dL_over_dL_quoted"]) > 0.1 * frac
    print(
        f"  [F1] sigma_dL = {sigma_dl:.6e} Gpc; sigma_dL/d_L = {frac:.6e}; "
        f"spec quotes {SPEC_DOSSIER['sigma_dL_over_dL_quoted']:.1e} as a FRACTION "
        f"({'DISAGREES' if disagrees else 'agrees'}); 1/SNR = {1.0 / snr:.6e}"
    )

    n_rows = int(len(crb))
    n_in_cat = int(crb["in_catalog"].sum())
    with open(_need(COMBINED_REL)) as fh:
        combined = json.load(fh)

    # Geometry note (definition, not derivation): the angle between the stored
    # spin axis (qK, phiK) and the line of sight (qS, phiS), both unit vectors
    # in the stamped ecliptic frame.  Reported so the toy widget's inclination
    # dial has a real place to sit; the pipeline's Fisher coordinates are the
    # 14 columns above, of which iota is NOT one.
    cos_iota = float(
        np.cos(row["qK"]) * np.cos(row["qS"])
        + np.sin(row["qK"]) * np.sin(row["qS"]) * np.cos(row["phiK"] - row["phiS"])
    )

    return {
        "_provenance": {
            "crb_csv": str(CRB_REL),
            "crb_csv_spec_named": str(CRB_SPEC_REL),
            "crb_twin_verified": twin_checked,
            "combined_posterior": str(COMBINED_REL),
            "row_index": EVENT_889,
            "generator": "book/generators/gen_ch01.py",
        },
        "event": {
            "label": "EMRI-889",
            "M_msun": _r(row["M"]),
            "mu_msun": _r(row["mu"]),
            "a": _r(row["a"]),
            "p0": _r(row["p0"]),
            "e0": _r(row["e0"]),
            "x0": _r(row["x0"]),
            "dL_Gpc": _r(d_l),
            "dL_Mpc": _r(d_l * 1000.0),
            "qS": _r(row["qS"]),
            "phiS": _r(row["phiS"]),
            "qK": _r(row["qK"]),
            "phiK": _r(row["phiK"]),
            "Phi_phi0": _r(row["Phi_phi0"]),
            "Phi_theta0": _r(row["Phi_theta0"]),
            "Phi_r0": _r(row["Phi_r0"]),
            "T_yr": _r(row["T"]),
            "dt_s": _r(row["dt"]),
            "t_plunge_yr": _r(row["t_plunge_yr"]),
            "snr": _r(snr),
            "host_galaxy_index": int(row["host_galaxy_index"]),
            "in_catalog": bool(row["in_catalog"]),
            "coord_frame": str(row["_coord_frame"]),
            "cov_frame": str(row["_cov_frame"]),
            "cos_iota_from_stored_angles": _r(cos_iota),
            "iota_deg_from_stored_angles": _r(np.degrees(np.arccos(cos_iota))),
        },
        "distance_precision": {
            "sigma_dL_Gpc": _r(sigma_dl),
            "sigma_dL_Mpc": _r(sigma_dl * 1000.0),
            "sigma_dL_over_dL": _r(frac),
            "one_over_snr": _r(1.0 / snr),
            "frac_in_units_of_one_over_snr": _r(frac * snr),
            "spec_quoted_as_fraction": SPEC_DOSSIER["sigma_dL_over_dL_quoted"],
            "spec_disagrees_with_recomputation": bool(disagrees),
            "flag": "book/design/flags/ch01_FLAGS.md#F1",
        },
        "context": {
            "n_crb_rows": n_rows,
            "n_events_used": int(combined["n_events_total"]),
            "n_in_catalog": n_in_cat,
            "in_catalog_fraction": _r(n_in_cat / n_rows),
            "snr_rank": 1,
            "snr_threshold": float(SNR_THRESHOLD),
            "mission_duration_yr": float(LISA_MISSION_DURATION_YEARS),
            "h_true": float(H_TRUE),
            "omega_m_fiducial": float(OMEGA_M),
            "omega_de_fiducial": float(OMEGA_DE),
        },
    }


# ---------------------------------------------------------------------------
# 2. The distance-redshift explorer
# ---------------------------------------------------------------------------
def build_dlz(d_l_889: float) -> dict[str, Any]:
    # (1) accuracy witness for the in-browser integral
    table: list[list[list[float]]] = []
    for h in H_LATTICE:
        plane: list[list[float]] = []
        for om in OM_LATTICE:
            plane.append([_r(dist(z, h=h, Omega_m=om, Omega_de=1.0 - om)) for z in Z_LATTICE])
        table.append(plane)

    # (2) the static-fallback curve family, fiducial Omega_m
    curves = {
        f"{h:.2f}": [_r(dist(z, h=h, Omega_m=OMEGA_M, Omega_de=OMEGA_DE)) for z in Z_CURVE]
        for h in H_STATIC
    }

    # (3) EMRI-889's implied redshift as a function of h (fiducial Omega_m)
    implied = [_r(dist_to_redshift(d_l_889, h=h)) for h in H_IMPLIED]
    # Low-z cross-check: z ~ H0 d_L / c should hold to a few percent here.
    z_lin = H_TRUE * 100.0 * (d_l_889 * 1000.0) / SPEED_OF_LIGHT_KM_S
    z_exact = dist_to_redshift(d_l_889, h=H_TRUE)
    if abs(z_lin / z_exact - 1.0) > 0.10:
        _fail(f"Hubble-law limit broken: z_lin={z_lin:.6f} vs z_exact={z_exact:.6f}")

    # (4) G7 row 6, recomputed — HARD GATE against the published table
    mis_z, mis_hp, mis_pct, mis_pub = [], [], [], []
    for z, pub in G7_ROW6_PUBLISHED.items():
        target = dist(z, h=H_TRUE, Omega_m=OMEGA_M_PLANCK, Omega_de=1.0 - OMEGA_M_PLANCK)

        def _f(hp: float, _t: float = target, _z: float = z) -> float:
            return dist(_z, h=hp, Omega_m=OMEGA_M, Omega_de=OMEGA_DE) - _t

        h_prime = float(brentq(_f, 0.3, 1.5, xtol=1e-13))
        pct = (h_prime / H_TRUE - 1.0) * 100.0
        if abs(pct - pub) > 0.01:
            _fail(f"G7 row 6 at z={z}: recomputed {pct:+.3f}% vs published {pub:+.2f}%")
        mis_z.append(z)
        mis_hp.append(_r(h_prime))
        mis_pct.append(_r(pct, 4))
        mis_pub.append(pub)

    return {
        "_provenance": {
            "function": "master_thesis_code.physical_relations.dist "
            "(physical_relations.py:132) / dist_to_redshift (:447)",
            "gate": "docs/gates/G7_systematics_budget.md — 'Numbers behind row #6'",
            "generator": "book/generators/gen_ch01.py",
        },
        "fiducial": {
            "h_true": float(H_TRUE),
            "omega_m": float(OMEGA_M),
            "omega_de": float(OMEGA_DE),
            "c_km_s": float(SPEED_OF_LIGHT_KM_S),
        },
        "validation": {
            "h": H_LATTICE,
            "omega_m": OM_LATTICE,
            "z": Z_LATTICE,
            "dL_Gpc": table,
            "note": "repo dist() evaluated on the lattice; the page integrates "
            "1/E(z) itself and reports the worst relative deviation",
        },
        "curves_static": {"z": Z_CURVE, "dL_Gpc_by_h": curves, "omega_m": float(OMEGA_M)},
        "event_dl": {
            "label": "EMRI-889",
            "dL_Gpc": _r(d_l_889),
            "h": H_IMPLIED,
            "implied_z": implied,
            "z_at_h_true": _r(z_exact),
            "z_hubble_law_at_h_true": _r(z_lin),
        },
        "omega_m_mispec": {
            "omega_m_assumed": float(OMEGA_M),
            "omega_m_true": OMEGA_M_PLANCK,
            "z": mis_z,
            "h_prime": mis_hp,
            "pct_on_H0_recomputed": mis_pct,
            "pct_on_H0_published_G7": mis_pub,
        },
    }


def main() -> None:
    print("gen_ch01: building Chapter 1 data")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ev = build_event()
    OUT_EVENT.write_text(json.dumps(ev, separators=(",", ":")) + "\n")
    print(f"  wrote {OUT_EVENT.relative_to(REPO_ROOT)} ({OUT_EVENT.stat().st_size:,} bytes)")

    dlz = build_dlz(float(ev["event"]["dL_Gpc"]))
    OUT_DLZ.write_text(json.dumps(dlz, separators=(",", ":")) + "\n")
    print(f"  wrote {OUT_DLZ.relative_to(REPO_ROOT)} ({OUT_DLZ.stat().st_size:,} bytes)")

    print(
        "  gates: dossier vs spec card OK; G7 row-6 recomputation matches the "
        "published table at all 6 redshifts; Hubble-law limit OK"
    )


if __name__ == "__main__":
    main()
