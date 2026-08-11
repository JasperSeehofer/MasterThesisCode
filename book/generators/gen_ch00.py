"""Generator for Chapter 0 — "Two Numbers That Should Be One" (prologue).

Produces one small data file:

``book/site/data/ch00_tension.json``
    1. **The anchors** (``measurements``) — the published H0 determinations the
       prologue plots *before any equation*: the CMB+LCDM value, the Cepheid
       distance-ladder value, and the "third methods" that illustrate why a
       new number does not automatically arbitrate — including the current
       state of the art in this book's own genre, the GWTC-3 dark-siren
       result.  These are **recorded literature values**, carried verbatim with
       their arXiv identifiers; the page chips them ``rec``/``real`` and never
       blends them with the toy.

    2. **The tension arithmetic** (``tension``) — gap, fractional gap and the
       two-number Gaussian significance, computed here so that every number in
       the prose is the number this file emits.  Nothing is rounded up: the
       page prints 4.89 sigma from the two quoted values *and* says that
       arXiv:2112.04510 quotes 5.0 sigma for the full SH0ES-vs-Planck
       comparison.

    3. **The arbitration budget** (``arbitration``) — the closed form behind
       interactive I0.1, plus the precision ceiling, the three worked example
       points used as the widget's static fallback, a coarse verdict map for
       the "show me the numbers" view, and ``check_points``: values the page's
       in-browser closed form must reproduce.  The widget is a **toy** (a
       hypothetical third method); the anchors it is judged against are real.

    4. **The h convention** (``h_convention``) — read live out of
       ``darksiren_emri/constants.py`` so the book's ``h = H0/100`` and the
       mock universe's injected truth ``h_true = 0.73`` are traceable to the
       code rather than typed by hand.  This is the prologue's only contact
       with the pipeline; no pipeline *claim* is made in Chapter 0.

Determinism: no RNG, no I/O outside ``book/`` for writes; the only read of the
main checkout is ``darksiren_emri/constants.py`` (parsed, not executed, so
the generator has no import-time dependency on the simulation stack).

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch00.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths.  parents[2] is the book worktree root; the main checkout is either the
# same tree (this repo is a worktree of it) or a sibling directory.
# ---------------------------------------------------------------------------
BOOK_ROOT = Path(__file__).resolve().parents[1]  # .../book
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../MasterThesisCode-book
OUT_DIR = BOOK_ROOT / "site" / "data"
OUT_FILE = OUT_DIR / "ch00_tension.json"

CONSTANTS_CANDIDATES = [
    REPO_ROOT / "darksiren_emri" / "constants.py",
    REPO_ROOT.parent / "MasterThesisCode" / "darksiren_emri" / "constants.py",
]

# ---------------------------------------------------------------------------
# 1. The recorded anchors (literature; values carried verbatim).
# ---------------------------------------------------------------------------
# Each entry: what was measured, how, the published value and its published
# uncertainty decomposition, and the citation the page prints next to it.
MEASUREMENTS: list[dict[str, Any]] = [
    {
        "key": "planck",
        "label": "Planck 2018",
        "method": "CMB anisotropies, flat ΛCDM",
        "family": "early",
        "H0": 67.4,
        "sigma": 0.5,
        "sigma_stat": 0.5,
        "sigma_sys": None,
        "note": (
            "Not a distance measurement at all: an inference of H0 from the "
            "acoustic scale of the last-scattering surface, under an assumed "
            "cosmological model."
        ),
        "cite": "Planck Collaboration VI (2020), arXiv:1807.06209",
        "anchor": True,
    },
    {
        "key": "shoes",
        "label": "SH0ES 2022",
        "method": "Cepheid-calibrated Type Ia supernova distance ladder",
        "family": "late",
        "H0": 73.04,
        "sigma": 1.04,
        "sigma_stat": None,
        "sigma_sys": None,
        "note": (
            "A ladder: geometric parallaxes calibrate Cepheids, Cepheids "
            "calibrate supernovae, supernovae reach the Hubble flow. Every "
            "rung is calibrated on the rung below it."
        ),
        "cite": "Riess et al. (2022), arXiv:2112.04510",
        "anchor": True,
    },
    {
        "key": "cchp",
        "label": "CCHP 2019 (TRGB)",
        "method": "tip-of-the-red-giant-branch calibrated supernova ladder",
        "family": "late",
        "H0": 69.8,
        "sigma": math.sqrt(0.8**2 + 1.7**2),
        "sigma_stat": 0.8,
        "sigma_sys": 1.7,
        "note": (
            "A third method that swapped one rung of the ladder and landed "
            "between the two anchors, with a systematic term twice its "
            "statistical one. It did not arbitrate."
        ),
        "cite": "Freedman et al. (2019), arXiv:1907.05922",
        "anchor": False,
    },
    {
        "key": "h0licow",
        "label": "H0LiCOW 2020",
        "method": "time-delay cosmography, 6 lensed quasars",
        "family": "late",
        "H0": 73.3,
        # `sigma` is the symmetric average of the published asymmetric interval and is
        # used only where a single number is unavoidable; `asym` carries the published
        # values and is what the page plots and tabulates.
        "sigma": 1.75,
        "sigma_stat": None,
        "sigma_sys": None,
        "note": ("Quoted at 2.4% — under an assumed family of lens mass profiles."),
        "cite": "Wong et al. (2020), arXiv:1907.04869",
        "anchor": False,
        "asym": {"hi": 1.7, "lo": 1.8},
    },
    {
        "key": "tdcosmo",
        "label": "TDCOSMO IV 2020",
        "method": "the same technique with the lens mass profile left free",
        "family": "late",
        "H0": 74.5,
        "sigma": 5.85,  # symmetric average; see `asym` for the published interval
        "sigma_stat": None,
        "sigma_sys": None,
        "note": (
            "Same systems, one modelling assumption relaxed: the uncertainty "
            "went from ~2% to ~8%. The extra 6% was always there; it was in "
            "the assumption, not in the data."
        ),
        "cite": "Birrer et al. (2020), arXiv:2007.02941",
        "anchor": False,
        "asym": {"hi": 5.6, "lo": 6.1},
    },
    {
        "key": "gw170817",
        "label": "GW170817",
        "method": "one bright standard siren (GW + electromagnetic counterpart)",
        "family": "siren",
        "H0": 70.0,
        "sigma": 10.0,
        "sigma_stat": None,
        "sigma_sys": None,
        "note": (
            "The first siren H0: distance from general relativity, redshift "
            "from the host galaxy's spectrum. Published as 70.0 +12.0 −8.0 "
            "— the scale of a single-event measurement."
        ),
        "cite": "Abbott et al. (2017), arXiv:1710.05835",
        "anchor": False,
        "asym": {"hi": 12.0, "lo": 8.0},
    },
    {
        "key": "gwtc3",
        "label": "GWTC-3 dark sirens",
        "method": "galaxy-catalogue redshifts, with the GW170817 counterpart",
        "family": "siren",
        # `sigma` is the symmetric average of the published asymmetric interval;
        # `asym` carries the published values and is what the page plots.
        "H0": 68.0,
        "sigma": 7.0,
        "sigma_stat": None,
        "sigma_sys": None,
        "note": (
            "The state of the art in this book's own genre: no counterpart, "
            "the redshift information taken from a galaxy catalogue. This is "
            "the published dark-siren analysis combined with GW170817's "
            "counterpart; the catalogue events on their own constrain far "
            "more weakly. Not a like-for-like 'before/after' against the 2017 "
            "counterpart-only row above -- that is a different analysis of "
            "different data."
        ),
        "cite": "Abbott et al. (2023), arXiv:2111.03604",
        "anchor": False,
        "asym": {"hi": 8.0, "lo": 6.0},
    },
]

# The row the prologue's own argument is aimed at: the published dark-siren
# determination, measured against the precision ceiling computed below.  Which
# row this is is named here rather than typed into the page.
GENRE_KEY = "gwtc3"

# The single-event scale used by the toy: rounded from GW170817's published
# +12.0/-8.0 to a symmetric 10 km/s/Mpc.  Stated on the page as a round number,
# never as a measurement.
SIGMA_PER_EVENT = 10.0

# Arbitration thresholds (the page's stated working definition, not a project
# result): "excludes" at >= 3 sigma, "consistent with" at <= 2 sigma.
T_EXCLUDE = 3.0
T_CONSISTENT = 2.0


def _anchor(key: str) -> dict[str, Any]:
    for m in MEASUREMENTS:
        if m["key"] == key:
            return m
    raise KeyError(key)


def sigma_total(n_events: float, sigma_sys: float) -> float:
    """Total 1-sigma of the hypothetical method: statistical (~1/sqrt N) and an
    unknown systematic, added in quadrature."""
    return math.sqrt((SIGMA_PER_EVENT**2) / n_events + sigma_sys**2)


def tension(mu: float, sig: float, anchor: dict[str, Any]) -> float:
    """Significance of the offset between the method and one anchor."""
    return abs(mu - anchor["H0"]) / math.sqrt(sig**2 + anchor["sigma"] ** 2)


def verdict(t_planck: float, t_shoes: float) -> str:
    """The page mirrors this function in JavaScript; keep the two in step."""
    if t_planck >= T_EXCLUDE and t_shoes >= T_EXCLUDE:
        return "excludes both anchors — a third answer, not an arbitration"
    if t_shoes >= T_EXCLUDE and t_planck <= T_CONSISTENT:
        return "arbitrates for the early-universe value"
    if t_planck >= T_EXCLUDE and t_shoes <= T_CONSISTENT:
        return "arbitrates for the late-universe value"
    if t_planck >= T_EXCLUDE or t_shoes >= T_EXCLUDE:
        return "excludes one anchor but does not sit comfortably on the other — not an arbitration"
    return "arbitrates nothing"


def main() -> None:
    planck = _anchor("planck")
    shoes = _anchor("shoes")

    gap = shoes["H0"] - planck["H0"]
    sig_comb = math.hypot(planck["sigma"], shoes["sigma"])
    n_sigma = gap / sig_comb

    # -- the precision ceiling ------------------------------------------------
    # Best case for the new method: it lands exactly on one anchor, so the only
    # thing standing between it and a verdict is its distance to the OTHER one.
    # Requiring a 3-sigma exclusion of the far anchor caps the method's TOTAL
    # 1-sigma at:  sqrt((gap/3)^2 - sigma_far^2).
    #
    # There are TWO such placements and they are not equally kind.  Parked on
    # the early-universe anchor the far anchor is the LATE one (sigma = 1.04)
    # and the cap is tighter; parked on the late-universe anchor the far anchor
    # is the tighter early one (sigma = 0.5) and the cap relaxes.  The page
    # quotes the demanding branch and names which one it is, so that the prose
    # and the arithmetic describe the same placement.
    ceiling_rhs = gap / T_EXCLUDE
    sigma_far = shoes["sigma"]  # far anchor when parked on Planck (the demanding case)
    ceiling = math.sqrt(ceiling_rhs**2 - sigma_far**2)
    ceiling_frac = ceiling / planck["H0"]
    ceiling_parked_late = math.sqrt(ceiling_rhs**2 - planck["sigma"] ** 2)
    n_needed_zero_sys = (SIGMA_PER_EVENT / ceiling) ** 2

    # -- three worked example points (the widget's static fallback) -----------
    examples = []
    for n_ev, s_sys, tag in (
        (41, 0.0, "just enough events, and an honest zero systematic"),
        (10000, 2.0, "two hundred times more events, with a 3% unknown systematic"),
        (400, 0.5, "a realistic middle: many events and a controlled systematic"),
    ):
        sig = sigma_total(n_ev, s_sys)
        t_far = gap / math.sqrt(sig**2 + sigma_far**2)
        examples.append(
            {
                "n_events": n_ev,
                "sigma_sys": s_sys,
                "sigma_stat": SIGMA_PER_EVENT / math.sqrt(n_ev),
                "sigma_total": sig,
                "sigma_total_frac": sig / planck["H0"],
                "t_far_best_case": t_far,
                "verdict": ("arbitrates" if t_far >= T_EXCLUDE else "arbitrates nothing"),
                "tag": tag,
            }
        )

    # -- verdict map for the numbers view ------------------------------------
    n_grid = [1, 10, 100, 1000, 10000]
    sys_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    t_map = [
        [gap / math.sqrt(sigma_total(n, s) ** 2 + sigma_far**2) for n in n_grid] for s in sys_grid
    ]
    # N -> infinity: the statistical term vanishes and the exclusion saturates
    # at a ceiling set by the systematic alone.  This column is the chapter's
    # whole argument in one list.
    t_asymptote = [gap / math.sqrt(s**2 + sigma_far**2) for s in sys_grid]

    # -- cross-check points the browser must reproduce ------------------------
    check_points = []
    for n_ev, s_sys, mu in (
        (1, 0.0, 70.0),
        (41, 0.0, 67.4),
        (10000, 2.0, 67.4),
        (400, 0.5, 73.04),
        (100, 1.0, 70.2),
    ):
        sig = sigma_total(n_ev, s_sys)
        tp = tension(mu, sig, planck)
        ts = tension(mu, sig, shoes)
        check_points.append(
            {
                "n_events": n_ev,
                "sigma_sys": s_sys,
                "mu": mu,
                "sigma_total": sig,
                "t_planck": tp,
                "t_shoes": ts,
                "verdict": verdict(tp, ts),
            }
        )

    # -- the genre's own state of the art, against the ceiling ----------------
    # The published dark-siren number is the one row on the figure that measures
    # the same thing this book measures, so the prologue says out loud how far it
    # is from the arbitration ceiling.  Both numbers in that sentence are emitted
    # here rather than typed into the page.
    genre = _anchor(GENRE_KEY)
    genre_sigma = (
        (genre["asym"]["hi"] + genre["asym"]["lo"]) / 2 if "asym" in genre else genre["sigma"]
    )
    genre_anchor = {
        "key": GENRE_KEY,
        "label": genre["label"],
        "cite": genre["cite"],
        "H0": genre["H0"],
        "sigma_symmetric": genre_sigma,
        "frac_of_own_H0": genre_sigma / genre["H0"],
        "ratio_to_ceiling": genre_sigma / ceiling,
    }

    # -- the h convention, read out of the pipeline's own constants -----------
    h_convention = _read_h_convention()

    payload: dict[str, Any] = {
        "_generated_by": "book/generators/gen_ch00.py",
        "_provenance": (
            "measurements: published literature values, carried verbatim with "
            "their arXiv identifiers. arbitration: closed-form toy defined in "
            "this file. h_convention: parsed from the pipeline's constants.py."
        ),
        "measurements": MEASUREMENTS,
        "tension": {
            "early": planck["key"],
            "late": shoes["key"],
            "gap": gap,
            "gap_frac_of_early": gap / planck["H0"],
            "gap_frac_of_late": gap / shoes["H0"],
            "sigma_combined": sig_comb,
            "n_sigma": n_sigma,
            "published_n_sigma": 5.0,
            "published_n_sigma_cite": "Riess et al. (2022), arXiv:2112.04510",
        },
        "arbitration": {
            "sigma_per_event": SIGMA_PER_EVENT,
            "sigma_per_event_basis": "GW170817, arXiv:1710.05835 (70.0 +12.0 -8.0), rounded",
            "t_exclude": T_EXCLUDE,
            "t_consistent": T_CONSISTENT,
            "ceiling_rhs": ceiling_rhs,
            "ceiling_sigma_total": ceiling,
            "ceiling_frac_of_H0": ceiling_frac,
            "ceiling_parked_on": planck["key"],
            "ceiling_far_anchor": shoes["key"],
            "ceiling_sigma_total_parked_on_late": ceiling_parked_late,
            "ceiling_branch_note": (
                "parked on the early-universe anchor the far anchor is the late "
                "one (the wider sigma), which is the more DEMANDING of the two "
                "placements; parked on the late-universe anchor the far anchor is "
                "the tighter early one and the cap relaxes"
            ),
            "n_events_needed_zero_sys": n_needed_zero_sys,
            "n_events_needed_zero_sys_int": math.ceil(n_needed_zero_sys),
            "examples": examples,
            "verdict_map": {
                "n_grid": n_grid,
                "sys_grid": sys_grid,
                "t_far": t_map,
                "t_far_asymptote": t_asymptote,
            },
            "check_points": check_points,
        },
        "genre_anchor": genre_anchor,
        "h_convention": h_convention,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")

    print(f"wrote {OUT_FILE.relative_to(REPO_ROOT)}  ({OUT_FILE.stat().st_size:,} bytes)")
    print(
        f"  gap                {gap:.2f} km/s/Mpc = {gap / planck['H0'] * 100:.2f}% of the early value"
    )
    print(
        f"  two-number sigma   {n_sigma:.2f} (paper quotes {payload['tension']['published_n_sigma']:.1f})"
    )
    print(
        f"  precision ceiling  {ceiling:.4f} km/s/Mpc = {ceiling_frac * 100:.2f}% of H0"
        f"  (parked on {planck['key']}; parked on {shoes['key']}: {ceiling_parked_late:.4f})"
    )
    print(
        f"  genre anchor       {genre_anchor['label']}: {genre['H0']:.1f} "
        f"+{genre['asym']['hi']:.1f}/-{genre['asym']['lo']:.1f} = "
        f"{genre_anchor['frac_of_own_H0'] * 100:.1f}% of its own H0, "
        f"{genre_anchor['ratio_to_ceiling']:.1f}x the ceiling  [{genre_anchor['cite']}]"
    )
    print(f"  N needed (sys=0)   {n_needed_zero_sys:.2f} -> {math.ceil(n_needed_zero_sys)} events")
    for ex in examples:
        print(
            f"  example  N={ex['n_events']:>6d}  sys={ex['sigma_sys']:.2f}  "
            f"sigma_tot={ex['sigma_total']:.4f}  T_far={ex['t_far_best_case']:.3f}  "
            f"-> {ex['verdict']}"
        )
    for s, t_inf in zip(sys_grid, t_asymptote):
        print(f"  N -> inf   sys={s:.2f}  T_far -> {t_inf:.3f}")
    print(f"  h_true             {h_convention['H']} from {h_convention['source']}")


def _read_h_convention() -> dict[str, Any]:
    """Parse ``H`` out of the pipeline's constants.py (read-only, no import).

    Parsing rather than importing keeps the generator free of the simulation
    stack's dependencies while still making the number *traceable*: the file,
    the line number and the source line itself are all carried into the JSON,
    so the page cites a code site it actually read.
    """
    path = next((p for p in CONSTANTS_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "darksiren_emri/constants.py not found in "
            + " or ".join(str(p) for p in CONSTANTS_CANDIDATES)
        )
    pattern = re.compile(r"^H\s*:\s*float\s*=\s*([0-9.]+)")
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        m = pattern.match(line)
        if m:
            return {
                "symbol": "h",
                "definition": "H0 / (100 km s^-1 Mpc^-1)",
                "H": float(m.group(1)),
                "role": "the mock universe's injected truth",
                "source": f"darksiren_emri/constants.py:{lineno}",
                "source_line": line.strip(),
            }
    raise ValueError(f"no 'H: float = ...' line found in {path}")


if __name__ == "__main__":
    main()
