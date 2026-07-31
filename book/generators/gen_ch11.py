"""Generator for Chapter 11 — "The State of the Art, Honestly".

The closing chapter presents the project's *current adjudicated state* and
resolves nothing.  Its three data files are therefore all either

  (a) re-read from the Gate-B attack scripts' own committed JSON outputs, or
  (b) re-measured from the delivered per-event diagnostics,

and every headline number is gated against the artifact that published it.
Where a recomputation disagrees with a document, the generator records the
disagreement in ``book/design/flags/ch11_FLAGS.md`` and emits BOTH values —
it never reconciles.

Outputs
-------
``book/site/data/ch11_runaways.json``   (I11.1 "The Two Runaways, Unlocked")
    The class-summed 1D log-likelihood profiles ``S_in(h)`` (76 / 88
    in-catalogue hosts) and ``S_dk(h)`` (the dark class) for all 12 venues —
    2 idealized baselines + 10 realistic realizations — taken from the
    adjudicator's own ``c5_rail_results.json`` (`_S_in` / `_S_dk` / `_h`).
    The browser recombines them live:

        combined(lambda) = argmax_h [ lambda * S_in(h) + S_dk(h) ]
        Poisson reweight = argmax_h [ (1+-1/sqrt(N_in)) S_in + (1-+1/sqrt(N_dk)) S_dk ]

    reproducing ``c5_class_weight_results.json`` (the lambda-scan) and
    ``c5_rail_results.json`` (the +-1/sqrt(N) shift) EXACTLY — both are gated
    below.  Each profile is max-subtracted per class, which is an additive
    constant in h and therefore changes no argmax.

``book/site/data/ch11_board.json``      (I11.2 "The Adjudication Board")
    Claims C1-C11 with their **verbatim** adjudicated status strings, the
    provenance tag the claim file assigns, which side of the estimator each
    sits on, the refutation route, and the artifact chips.  Hand-authored from
    ``CLAIM_2D_BIAS_20260730.md`` (as amended 2026-07-30) +
    ``gate_b_20260730/ADJUDICATION_20260730.md``; the generator re-checks the
    numeric fields it can (C1, C2, C3, C9's realized rate) against the data.

``book/site/data/ch11_dossier.json``    (the running example's payoff)
    EMRI-889's per-realization 2D-minus-1D channel swing, re-measured from
    each run's ``diagnostics/event_likelihoods.csv``.  Gate: +1.98 / -2.04 /
    -3.30 for r1 / r2 / r3 (``c3c4_allruns_summary.md``).

Data hygiene (BOOK_DESIGN.md §4.2)
----------------------------------
* Idealized baselines: seed61000 -> ``run_seed61000/posteriors_fixed`` (the
  plain ``posteriors/`` is the stale pre-``ec09ed0`` backup), seed62000 ->
  ``run_seed62000/posteriors``.  This generator does not read them directly —
  it inherits the attack script's own choice, which is the canonical one.
* Root / ``sig0_control`` / ``zoom`` diagnostics CSVs are NEVER used (two
  concatenated evaluate sweeps; ``sig0_control`` additionally carries the
  ``generator_marginal`` estimand).
* The h grid is non-uniform (0.01 on [0.60,0.65] and [0.80,0.86], 0.005
  between).  Nothing here takes a second difference across a seam; the
  sub-grid MAP refinements below are the attack scripts' own, reproduced
  bit-for-bit.

Determinism: no RNG.  Read-only outside ``book/``.

Run as::

    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch11.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths.  repo root = two levels above book/generators/ ; the source repo is
# either this checkout or a sibling ``MasterThesisCode`` checkout.
# --------------------------------------------------------------------------
BOOK_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BOOK_ROOT / "site" / "data"
FLAG_FILE = BOOK_ROOT / "design" / "flags" / "ch11_FLAGS.md"


_HERE = Path(__file__).resolve().parents[2]
# The Gate-B analysis JSONs are git-tracked (present in any checkout of this
# branch); the per-event diagnostics CSVs and the CRB tables are NOT — they
# live in the working tree of the main checkout only.  Every artifact is
# therefore resolved per-path across both roots, in order.
SEARCH_ROOTS = [_HERE, _HERE.parent / "MasterThesisCode"]


def res(rel: str) -> Path | None:
    """First existing candidate for a repo-relative artifact path, else None."""
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


CAMPAIGN_REL = "results/campaign51_20260728"
REAL_REL = f"{CAMPAIGN_REL}/realistic_20260729"
GATE_B_REL = f"{REAL_REL}/gate_b_20260730"

H_TRUE = 0.73
SEEDS = (61000, 62000)
REALIZATIONS = (1, 2, 3, 4, 5)


# --------------------------------------------------------------------------
# The two sub-grid MAP refinements, transcribed verbatim from the attack
# scripts so the browser can reproduce their published numbers.
#   submap()      -- attack_c5_class_weight.py:55-63   (5-point polyfit)
#   map_subgrid() -- attack_c5_rail.py:185-193         (3-point parabola)
# --------------------------------------------------------------------------
def submap(h: np.ndarray, S: np.ndarray) -> float:
    """5-point parabola vertex around the grid argmax (lambda-scan MAP)."""
    j = int(np.argmax(S))
    if j == 0 or j == len(S) - 1:
        return float(h[j])
    lo, hi = max(0, j - 2), min(len(h), j + 3)
    c = np.polyfit(h[lo:hi], S[lo:hi], 2)
    hv = -c[1] / (2 * c[0])
    return float(hv) if h[lo] <= hv <= h[hi - 1] else float(h[j])


def map_subgrid(h: np.ndarray, prof: np.ndarray) -> float:
    """3-point parabola vertex around the grid argmax (Poisson-reweight MAP)."""
    j = int(np.argmax(prof))
    if j == 0 or j == len(prof) - 1:
        return float(h[j])
    y0, y1, y2 = prof[j - 1], prof[j], prof[j + 1]
    den = y0 - 2 * y1 + y2
    if den == 0:
        return float(h[j])
    return float(h[j] - 0.5 * (h[j + 1] - h[j]) * (y2 - y0) / den)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def rnd(x: Any, n: int = 8) -> Any:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rnd(v, n) for v in x]
    return float(np.round(float(x), n))


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
            raise SystemExit(
                f"gen_ch11 GATE FAILED: {name}: got {got!r}, expected {expected!r} "
                f"(tol {tol}) [{cite}]"
            )

    def summary(self) -> dict[str, Any]:
        return {"n": len(self.rows), "all_pass": all(r["pass"] for r in self.rows), "rows": self.rows}


GATES = Gates()


# ==========================================================================
# 1. ch11_runaways.json — I11.1
# ==========================================================================
def build_runaways() -> dict[str, Any]:
    rail = json.loads(need(f"{GATE_B_REL}/c5_rail_results.json").read_text())
    cweight = json.loads(need(f"{GATE_B_REL}/c5_class_weight_results.json").read_text())
    lever = json.loads(need(f"{GATE_B_REL}/c5_leverage_results.json").read_text())

    lam_by_name = {e["name"]: e for e in cweight["lambda_scan"]}
    lev_by_name = {e["name"]: e for e in lever["leverage"]}

    # class sizes: in-catalogue = host_galaxy_index >= 0 in the seed's CRB table
    n_incat: dict[int, int] = {}
    n_rows: dict[int, int] = {}
    for seed in SEEDS:
        crb = pd.read_csv(need(f"{REAL_REL}/seed{seed}/prepared_cramer_rao_bounds.csv"))
        n_incat[seed] = int((crb["host_galaxy_index"] >= 0).sum())
        n_rows[seed] = int(len(crb))

    h_ref: list[float] | None = None
    runs: list[dict[str, Any]] = []

    for r in rail["runs"]:
        name = str(r["name"])
        seed = 61000 if "61000" in name else 62000
        tag = name.split()[1]
        h = np.asarray(r["_h"], dtype=float)
        S_in = np.asarray(r["_S_in"], dtype=float)
        S_dk = np.asarray(r["_S_dk"], dtype=float)
        if h_ref is None:
            h_ref = [float(v) for v in h]
        assert [float(v) for v in h] == h_ref, "h grid differs between runs"

        # Max-subtract each class profile: an additive constant in h, so every
        # argmax (and every reweighted argmax) is unchanged, but the numbers
        # shipped to the browser are O(1e2) instead of O(1e5).
        off_in, off_dk = float(S_in.max()), float(S_dk.max())
        s_in = rnd(S_in - off_in)
        s_dk = rnd(S_dk - off_dk)

        # ---- gates: reproduce the published numbers from the SHIPPED arrays
        a_in = np.asarray(s_in, dtype=float)
        a_dk = np.asarray(s_dk, dtype=float)
        lam = lam_by_name[name]
        for lv in (0.0, 0.5, 1.0, 1.5, 2.0):
            GATES.check(
                f"{name} lambda={lv} MAP",
                submap(h, lv * a_in + a_dk),
                float(lam[f"lam{lv}"]),
                1e-6,
                "c5_class_weight_results.json",
            )
        pr = r["poisson_reweight"]
        f_in, f_dk = float(pr["sigma_incat_frac"]), float(pr["sigma_dark_frac"])
        base = map_subgrid(h, a_in + a_dk)
        GATES.check(f"{name} base MAP", base, float(pr["base_map"]), 1e-6, "c5_rail_results.json")
        combos = {
            "incat+1sig": (1 + f_in, 1.0),
            "incat-1sig": (1 - f_in, 1.0),
            "dark+1sig": (1.0, 1 + f_dk),
            "dark-1sig": (1.0, 1 - f_dk),
            "incat+ dark-": (1 + f_in, 1 - f_dk),
            "incat- dark+": (1 - f_in, 1 + f_dk),
        }
        pmaps = {k: map_subgrid(h, ci * a_in + cd * a_dk) for k, (ci, cd) in combos.items()}
        for k, v in pmaps.items():
            GATES.check(f"{name} poisson {k}", v, float(pr["maps"][k]), 1e-6, "c5_rail_results.json")
        max_shift = max(abs(v - base) for v in pmaps.values())
        GATES.check(
            f"{name} poisson max|shift|", max_shift, float(pr["max_abs_shift"]), 1e-6, "c5_rail_results.json"
        )
        GATES.check(
            f"{name} dark-only argmax",
            float(h[int(np.argmax(a_dk))]),
            float(lam["dark_only_argmax"]),
            1e-12,
            "c5_class_weight_results.json",
        )
        GATES.check(
            f"{name} in-cat-only argmax",
            float(h[int(np.argmax(a_in))]),
            float(lam["incat_only_argmax"]),
            1e-12,
            "c5_class_weight_results.json",
        )

        lv = lev_by_name[name]
        n_in = int(round(1.0 / f_in**2))
        n_dk = int(round(1.0 / f_dk**2))
        runs.append(
            {
                "key": name.replace(" ", "_"),
                "name": name,
                "seed": seed,
                "tag": tag,
                "venue": "idealized" if tag == "IDEAL" else "realistic",
                "label": (
                    f"seed{seed} idealized (#51)" if tag == "IDEAL" else f"seed{seed} {tag} (#53)"
                ),
                "S_in": s_in,
                "S_dk": s_dk,
                "offset_in": rnd(off_in, 6),
                "offset_dk": rnd(off_dk, 6),
                "n_incat": n_in,
                "n_dark": n_dk,
                "sigma_incat_frac": rnd(f_in, 10),
                "sigma_dark_frac": rnd(f_dk, 10),
                "base_map": rnd(base, 6),
                "poisson_maps": {k: rnd(v, 6) for k, v in pmaps.items()},
                "poisson_max_shift": rnd(max_shift, 6),
                "lambda_scan": {str(k): rnd(lam[f"lam{k}"], 6) for k in (0.0, 0.5, 1.0, 1.5, 2.0)},
                "dark_only_argmax": rnd(lam["dark_only_argmax"], 4),
                "incat_only_argmax": rnd(lam["incat_only_argmax"], 4),
                "edge_frac_086": None,
                "leverage": {
                    "sigma_h": rnd(lv["sigma_h"], 6),
                    "S_in_slope": rnd(lv["S_in_slope"], 4),
                    "S_tot_curv": rnd(lv["S_tot_curv"], 3),
                    "dh_deps_incat": rnd(lv["dh_deps_incat"], 8),
                    "dh_1sig_poisson": rnd(lv["dh_1sig_poisson"], 8),
                },
            }
        )

    # ---- leverage ratios vs each seed's own idealized baseline -------------
    ideal_lev = {
        seed: lev_by_name[f"seed{seed} IDEAL"]["dh_deps_incat"] for seed in SEEDS
    }
    ratios: list[float] = []
    for run in runs:
        base_lev = ideal_lev[run["seed"]]
        run["leverage"]["ratio_to_ideal"] = rnd(run["leverage"]["dh_deps_incat"] / base_lev, 1)
        if run["venue"] == "realistic":
            ratios.append(float(run["leverage"]["ratio_to_ideal"]))
        run["leverage"]["curv_ratio_to_ideal"] = rnd(
            lev_by_name[f"seed{run['seed']} IDEAL"]["S_tot_curv"] / run["leverage"]["S_tot_curv"], 1
        )

    # ---- C5's own published summary numbers, and where they land ----------
    max_shift_all = max(r["poisson_max_shift"] for r in runs if r["venue"] == "realistic")
    max_shift_ideal = max(r["poisson_max_shift"] for r in runs if r["venue"] == "idealized")
    GATES.check(
        "C5 '+-1/sqrt(N) reweight moves the MAP by up to 0.025'",
        max_shift_all,
        0.025,
        6e-4,
        "CLAIM_2D_BIAS_20260730.md C5 / ADJUDICATION §1 C5",
    )

    return {
        "_meta": {
            "chapter": 11,
            "widget": "I11.1 The Two Runaways, Unlocked",
            "source": "gate_b_20260730/c5_rail_results.json (_S_in/_S_dk/_h), "
            "c5_class_weight_results.json (lambda-scan), c5_leverage_results.json (dh*/deps)",
            "note": "class profiles are max-subtracted per class (an additive constant in h; "
            "no argmax changes). Sub-grid MAPs use the attack scripts' own refinements.",
        },
        "h_grid": h_ref,
        "h_true": H_TRUE,
        "n_crb_rows": {str(k): v for k, v in n_rows.items()},
        "n_crb_incat": {str(k): v for k, v in n_incat.items()},
        "runs": runs,
        "published": {
            "poisson_max_shift_realistic": rnd(max_shift_all, 6),
            "poisson_max_shift_idealized": rnd(max_shift_ideal, 6),
            "claim_poisson_up_to": 0.025,
            "claim_leverage_ratio_range": [1500, 2400],
            "measured_leverage_ratio_range": [rnd(min(ratios), 1), rnd(max(ratios), 1)],
            "measured_leverage_ratio_median": rnd(float(np.median(ratios)), 1),
            "flag": "F-ch11-1",
        },
    }


# ==========================================================================
# 2. ch11_dossier.json — EMRI-889's channel swing
# ==========================================================================
def build_dossier() -> dict[str, Any]:
    crb = pd.read_csv(need(f"{REAL_REL}/seed61000/prepared_cramer_rao_bounds.csv"))
    incat_idx = set(crb.index[crb["host_galaxy_index"] >= 0].astype(int).tolist())
    is_incat_889 = 889 in incat_idx

    out_runs: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    h_ref: list[float] | None = None

    for r in REALIZATIONS:
        csv = need(f"{REAL_REL}/seed61000/real_r{r}/diagnostics/event_likelihoods.csv")
        ev = pd.read_csv(csv, usecols=["event_idx", "h", "combined_no_bh", "combined_with_bh"])
        hs = np.sort(ev["h"].unique())
        if h_ref is None:
            h_ref = [float(v) for v in hs]
        e = ev[ev["event_idx"] == 889].sort_values("h")
        assert len(e) == len(hs), f"event 889 missing h points in real_r{r}"
        ln1 = np.log(e["combined_no_bh"].to_numpy(dtype=float))
        ln2 = np.log(e["combined_with_bh"].to_numpy(dtype=float))
        i73 = int(np.argmin(np.abs(hs - 0.73)))
        i81 = int(np.argmin(np.abs(hs - 0.81)))
        d1 = float(ln1[i81] - ln1[i73])
        d2 = float(ln2[i81] - ln2[i73])
        out_runs.append(
            {
                "run": f"real_r{r}",
                "d_ln_1D_073_081": rnd(d1, 4),
                "d_ln_2D_073_081": rnd(d2, 4),
                "channel_diff": rnd(d2 - d1, 4),
                "argmax_1D": rnd(float(hs[int(np.argmax(ln1))]), 4),
                "argmax_2D": rnd(float(hs[int(np.argmax(ln2))]), 4),
            }
        )
        if r <= 3:
            profiles[f"real_r{r}"] = {
                # anchored at h = 0.73 so the three realizations are comparable
                "ln1d": rnd(ln1 - ln1[i73], 6),
                "ln2d": rnd(ln2 - ln2[i73], 6),
                "delta": rnd((ln2 - ln2[i73]) - (ln1 - ln1[i73]), 6),
            }

    # Gate against c3c4_allruns_summary.md's traced swing for event_idx 889.
    for run, expected in (("real_r1", 1.98), ("real_r2", -2.04), ("real_r3", -3.30)):
        got = next(x["channel_diff"] for x in out_runs if x["run"] == run)
        GATES.check(f"889 channel swing {run}", got, expected, 0.011, "c3c4_allruns_summary.md:123")

    # C1 class budget re-check (the board's C1 row quotes these).
    ev1 = pd.read_csv(
        need(f"{REAL_REL}/seed61000/real_r1/diagnostics/event_likelihoods.csv"),
        usecols=["event_idx", "h", "combined_no_bh", "combined_with_bh"],
    )
    hs = np.sort(ev1["h"].unique())
    i73 = float(hs[int(np.argmin(np.abs(hs - 0.73)))])
    i81 = float(hs[int(np.argmin(np.abs(hs - 0.81)))])
    a = ev1[ev1["h"] == i73].set_index("event_idx").sort_index()
    b = ev1[ev1["h"] == i81].set_index("event_idx").sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    d1 = np.log(b["combined_no_bh"].to_numpy()) - np.log(a["combined_no_bh"].to_numpy())
    d2 = np.log(b["combined_with_bh"].to_numpy()) - np.log(a["combined_with_bh"].to_numpy())
    mask_in = np.array([int(i) in incat_idx for i in common])
    c1_in, c1_dk = float(d1[mask_in].sum()), float(d1[~mask_in].sum())
    c3_in, c3_dk = float((d2 - d1)[mask_in].sum()), float((d2 - d1)[~mask_in].sum())
    GATES.check("C1 in-cat budget (r1)", c1_in, 2.48, 0.02, "CLAIM_2D_BIAS C1")
    GATES.check("C1 dark budget (r1)", c1_dk, -11.77, 0.02, "CLAIM_2D_BIAS C1")
    GATES.check("C3 in-cat channel diff (r1)", c3_in, 2.97, 0.02, "CLAIM_2D_BIAS C3")
    GATES.check("C3 dark channel diff (r1)", c3_dk, 15.83, 0.02, "CLAIM_2D_BIAS C3")
    GATES.check("C2 channel totals: 1D", c1_in + c1_dk, -9.30, 0.02, "CLAIM_2D_BIAS C2")
    GATES.check("C2 channel totals: 2D", c1_in + c1_dk + c3_in + c3_dk, 9.51, 0.03, "CLAIM_2D_BIAS C2")

    return {
        "_meta": {
            "chapter": 11,
            "figure": "EMRI-889's channel swing across noise realizations",
            "source": "seed61000/real_r{1..5}/diagnostics/event_likelihoods.csv",
            "gate": "c3c4_allruns_summary.md:123 (+1.98 / -2.04 / -3.30)",
        },
        "event_idx": 889,
        "in_catalogue": bool(is_incat_889),
        "h_grid": h_ref,
        "h_true": H_TRUE,
        "runs": out_runs,
        "profiles": profiles,
        "class_budget_r1": {
            "c1_incat": rnd(c1_in, 3),
            "c1_dark": rnd(c1_dk, 3),
            "c3_incat": rnd(c3_in, 3),
            "c3_dark": rnd(c3_dk, 3),
            "channel_total_1d": rnd(c1_in + c1_dk, 3),
            "channel_total_2d": rnd(c1_in + c1_dk + c3_in + c3_dk, 3),
        },
    }


# ==========================================================================
# 3. ch11_board.json — I11.2, the adjudication board
# ==========================================================================
# Every "status" string below is transcribed VERBATIM from the section heading
# / adjudication verdict of `CLAIM_2D_BIAS_20260730.md` (as amended
# 2026-07-30) and `gate_b_20260730/ADJUDICATION_20260730.md` §1.
BOARD: list[dict[str, Any]] = [
    {
        "id": "C1",
        "title": "The 1D class budget",
        "status": "FINDING, closed",
        "badge": "finding",
        "tag": "[LOCAL, VERIFIED]",
        "side": "measurement",
        "side_label": "bookkeeping (both channels)",
        "live": False,
        "one_line": "Σ Δln p over h = 0.73 → 0.81, seed61000 r1: in-cat +2.48, dark −11.77, "
        "total −9.30 (idealized: −338.10 / −23.52 / −361.62).",
        "adjudication": "The refutation route was executed (Gate A3) and failed to refute: the class "
        "structure replicates in sign and order across all 10 realistic runs and both seeds "
        "(in-cat +1.27…+5.38 vs idealized −338 / −248; dark −11.8…−14.1).",
        "refute_by": "recompute on another realization or seed — DONE, did not refute",
        "chips": ["CLAIM_2D_BIAS_20260730.md C1", "ADJUDICATION §1"],
    },
    {
        "id": "C2",
        "title": "The channel totals",
        "status": "FINDING, closed",
        "badge": "finding",
        "tag": "[LOCAL, VERIFIED]",
        "side": "measurement",
        "side_label": "bookkeeping (both channels)",
        "live": False,
        "one_line": "ln P(0.81)/P(0.73) read off the delivered posteriors: 1D = −9.30, 2D = +9.51. "
        "Channel difference +18.80 nats.",
        "adjudication": "Independently reconstructed by the C8 attacker to 3.6×10⁻¹² nats and "
        "cross-checked.",
        "refute_by": "nothing cheap — a direct read of the delivered posteriors",
        "chips": ["CLAIM_2D_BIAS_20260730.md C2", "ADJUDICATION §1"],
    },
    {
        "id": "C3",
        "title": "The dark class owns the channel difference",
        "status": "promoted to FINDING",
        "badge": "finding",
        "tag": "[LOCAL, VERIFIED]",
        "side": "catalogue",
        "side_label": "catalogue leg (dark class)",
        "live": False,
        "one_line": "r1 split: in-cat +2.97 / dark +15.83. Across all 10 runs the dark component is "
        "+15.83…+17.14 and ALWAYS positive; the in-cat component is small, noisy "
        "(−1.83…+2.97) and flips sign in one run — traced to a single high-leverage event, "
        "event_idx 889.",
        "adjudication": "The precise “84.2%” is r1-specific (dark share 84.2%–112.5% across runs). "
        "What replicates is the qualitative claim: dark ≫ in-cat, dark always positive. The book "
        "prints the replicated claim, not the percentage.",
        "refute_by": "off-r1 replication — DONE; the percentage did not survive, the ordering did",
        "chips": ["CLAIM_2D_BIAS_20260730.md C3", "c3c4_allruns_summary.md"],
    },
    {
        "id": "C4",
        "title": "The mechanism: impostor rejection → completion fallback",
        "status": "observations FINDING; mechanism AS STATED REFUTED; replaced by an AMENDED mechanism",
        "badge": "refuted",
        "tag": "[LOCAL, VERIFIED, r1]",
        "side": "catalogue",
        "side_label": "catalogue leg (mass de-weighting)",
        "live": False,
        "one_line": "Observed: 64.7% of dark events have an identically-zero 2D catalogue leg at "
        "h = 0.73, and survivors are suppressed by a median factor 7.8×10⁻³. But those 487 "
        "always-zero events carry +0.24 of the +15.83 nats (1.5%): 98.5% is carried by the 534 "
        "survivors.",
        "adjudication": "Deletion is NOT the mechanism. The amended mechanism is DE-WEIGHTING: the dark "
        "mean catalogue mixture weight falls 0.0354 → 0.0061 at h = 0.73, the dark class’s "
        "opposition collapses −24.46 → −0.63 nats, and its argmax moves 0.640 → 0.785. "
        "Budget: +15.83 = 0 (completion, cancels exactly) + 19.10 (loss of the 1D catalogue "
        "down-tilt) − 3.27 (residual 2D tilt).",
        "refute_by": "algebraic decomposition ln p = ln C + ln(1+R) — DONE; ln C cancels identically",
        "chips": ["CLAIM_2D_BIAS_20260730.md C4", "attack_c4_decomposition.py"],
    },
    {
        "id": "C5",
        "title": "58% of in-catalogue hosts rail at the prior edge",
        "status": "FINDING, interpretation AMENDED",
        "badge": "finding",
        "tag": "[LOCAL, VERIFIED, replicated]",
        "side": "both",
        "side_label": "both legs (the crossing)",
        "live": True,
        "one_line": "Per-event 1D argmax over [0.60, 0.86], 76 in-catalogue hosts: median 0.860, "
        "44/76 = 57.9% at the edge (idealized: 0.730, 4/76 = 5.3%). Replicates 10/10 runs at "
        "54–67%, against a 2.4% flat-surface expectation.",
        "adjudication": "Not an edge artifact: railed profiles are genuinely concave, top-K vertices "
        "give h_eff = 0.93–1.05 stable over K = 3–9, and a grid extended to h = 2.4 finds "
        "interior peaks (median ≈ 1.12) — a clipped real runaway. FAIR-FRAMING AMENDMENT "
        "(binding, both halves always): per event the rail is cosmetic (0.072–0.134 nats = "
        "0.30–0.47 σ_event, 0–1.3% exceed 1σ), BUT the class-summed displacement is "
        "+3.4 to +6.1 σ_class in 8/10 runs.",
        "refute_by": "widen the grid above 0.86 and see whether the peaks move further or stop — "
        "DONE, the refutation attempt FAILED",
        "chips": ["CLAIM_2D_BIAS_20260730.md C5", "attack_c5_rail.py", "attack_c5_extrap_validation.py"],
    },
    {
        "id": "C6",
        "title": "Attribution is confounded; the decisive control was never run",
        "status": "FINDING (confirmed by Gate A1); resolution in flight",
        "badge": "confounded",
        "tag": "[DOC + INFER]",
        "side": "meta",
        "side_label": "experimental design",
        "live": True,
        "one_line": "Campaign #51 → #53 changed catalogue scatter, host-z kernel and normalization "
        "mode simultaneously, and no run anywhere varies the estimator at fixed catalogue.",
        "adjudication": "The one-file check was executed: sig0_control ran generator_marginal + the "
        "point kernel, so it is not the missing control. “The bias switches on with the realized "
        "scatter” is NOT established. Resolution in flight: the pre-registered 2×2 cell B "
        "(unscattered catalogue × the #53 estimator), jobs 6101146 / 6101147, with a dated "
        "pre-readout prediction registered 2026-07-30 BEFORE the run landed.",
        "refute_by": "read sig0_control/run_metadata_0.json — DONE; the claim did not collapse",
        "chips": ["CLAIM_2D_BIAS_20260730.md C6", "PREREGISTRATION_2x2_cellB.md", "ADJUDICATION §3"],
    },
    {
        "id": "C7",
        "title": "The host-z numerator kernel omits selection",
        "status": "promoted to FINDING (measured), with corrected law; scope narrowed",
        "badge": "finding",
        "tag": "[LOCAL, VERIFIED — MEASURED]",
        "side": "kernel",
        "side_label": "host-z kernel (channel-common)",
        "live": True,
        "one_line": "bayesian_statistics.py:4201-4207 weights the host-z numerator kernel by the "
        "cosmic prior w_pop = (dV_c/dz)/(1+z), with no p_det and no catalogue selection. Measured "
        "inflation law $h_{\\rm eff}/h_{\\rm true} = [1+\\sqrt{1+12(\\sigma_z/z)^2}]/2 \\to "
        "1+3(\\sigma_z/z)^2$; rail threshold σ_z/z > 0.256.",
        "adjudication": "Confirmed as the mechanism for C5’s catalogue-leg per-event rail by direct "
        "measurement of the code’s own numerator. COLLIDES with the RATIFIED G2b derivation, which "
        "confirmed exactly that weight, without p_det, as uniquely consistent and exactly "
        "h-independent, protected by a binding regression gate. Any fix must EXPLICITLY SUPERSEDE "
        "G2b. The measured historical failure mode of the deconvolution at large σ_z/z was "
        "OVER-correction — the opposite sign to where a C7 fix pushes. Cell B is the staleness-free "
        "magnitude check.",
        "refute_by": "compute the induced host-z shift numerically for the 76 hosts at their real "
        "σ_z — DONE, it confirmed; the collision with G2b is unresolved",
        "chips": ["CLAIM_2D_BIAS_20260730.md C7", "gate_b_20260730/C7_README.md", "G2b:413-436"],
    },
    {
        "id": "C8",
        "title": "The 2D posterior is reparametrization-dependent",
        "status": "promoted to FINDING (well-posedness defect); cause RELOCATED",
        "badge": "finding",
        "tag": "[LOCAL, VERIFIED]",
        "side": "completion",
        "side_label": "completion numerator (missing mass measure)",
        "live": True,
        "one_line": "Rescaling the mass coordinate by an arbitrary constant C walks the 2D MAP: "
        "0.81329 / 0.78107 / 0.74440 / rails at 0.600 for C = 1 / 0.3 / 0.1 / ≤ 0.01. The 1D "
        "channel is bitwise invariant across the whole sweep.",
        "adjudication": "The stated cause is REFUTED: D, β_G and β_Ḡ are all "
        "mass-dimensionless, so it is not “4D numerator vs 3D D(h)”. The mismatch is BETWEEN THE "
        "TWO NUMERATOR LEGS — the 2D catalogue leg carries exactly one mass density, the completion "
        "leg carries none — and the code silently hard-wires the measure to $dM_z/M_{z,\\rm det,i}$, "
        "the event’s own measured detector-frame mass (span 1.33×10⁵–1.63×10⁶ "
        "M☉, a factor 12). A consistent physical unit change M → kM is exactly invariant.",
        "refute_by": "re-run the C-scaling on regenerated per-event 2D data; check the 1D invariance "
        "is exact — DONE, reproduced exactly",
        "chips": ["CLAIM_2D_BIAS_20260730.md C8", "gate_b_20260730/README_C8.md", "c8_reparam.py"],
    },
    {
        "id": "C9",
        "title": "w_G is mis-calibrated against the code’s own generator",
        "status": "FINDING [LOCAL, VERIFIED] — live, gated on cell B",
        "badge": "finding",
        "tag": "[LOCAL, VERIFIED]",
        "side": "prefactor",
        "side_label": "mixture weight / prefactor",
        "live": True,
        "one_line": "Model w_G(0.73) = 0.1215037 versus the realized detected in-catalogue rate "
        "164/3135 = 0.05231; binomial z = −11.86 pooled. Localized to β_G weighting f(z) by the "
        "pool-marginal (population-mass) p_det while Malmquist-selected catalogue hosts carry heavier "
        "M–σ masses; mass-aware w_G = 0.05149 gives z = +0.21.",
        "adjudication": "Adjudicator’s discount: the “removes 84% of the bias” counterfactual "
        "must be read against C5’s leverage finding — in a near-flat profile many ±10-nat "
        "interventions move the MAP a lot. What is solid independent of leverage is the z = −11.86 "
        "generator-vs-inference inconsistency and the two-way 0.392 / 0.399 agreement. Extended to the "
        "dark side: the realized detected dark-host z-distribution is skewed high against "
        "β_Ḡ’s own coded integrand, KS D = 0.0863, p = 1.08×10⁻¹⁹. "
        "RE-LITIGATION GUARD: the exonerated w_G = β_G/D bookkeeping FIX FORM merely relocated the "
        "tilt (+94…+455 nats, 12/12 fail, ledger #61) and must not be re-tried — the DEFECT is live.",
        "refute_by": "show the two estimands are in fact compatible — i.e. that the realized 164/3135 "
        "is not the quantity w_G models",
        "chips": ["CLAIM_2D_BIAS_20260730.md C9", "c9_darkdraw_check.py", "G1_beta_g_check.md:14-29"],
    },
    {
        "id": "C10",
        "title": "The completion-channel up-pull is prefactor-carried",
        "status": "FINDING [LOCAL, VERIFIED]",
        "badge": "finding",
        "tag": "[LOCAL, VERIFIED]",
        "side": "prefactor",
        "side_label": "mixture weight / prefactor",
        "live": False,
        "one_line": "Over h = 0.73 → 0.81: N·Δln(1−w_G) = +31.55 (dark +30.04, in-cat +1.51) "
        "while Σ Δln L_comp = −3.11 (dark −22.72, in-cat +19.61); only 39.1% of dark events "
        "have a positive completion tilt.",
        "adjudication": "Any sentence of the form “the completion term pulls up” must name the "
        "(1−w_G) PREFACTOR, not L_comp — which pulls DOWN for dark events. This retires the C4 "
        "mechanism’s wording.",
        "refute_by": "recompute the two budget terms from the diagnostics CSV",
        "chips": ["CLAIM_2D_BIAS_20260730.md C10"],
    },
    {
        "id": "C11",
        "title": "Completion-leg calibration is too small to own the 2D bias",
        "status": "REFUTED as the 2D owner; live as a modest 1D contributor",
        "badge": "refuted",
        "tag": "[LOCAL, harness]",
        "side": "completion",
        "side_label": "completion leg (calibration)",
        "live": False,
        "one_line": "pp_coverage extended to comp_frac 0.008–0.234 (the #53 w_G ≈ 0.12 venue): "
        "bias +0.0008…+0.0097 at 0.06–0.09 and +0.0034…+0.0181 at 0.13–0.24; monotone "
        "across 0.008–0.85, no sign flip, control-consistent at zero. That is 6–16× below "
        "+0.077.",
        "adjudication": "A quantitative exoneration, not a hand-wave. Live as a modest contributor to "
        "the 1D +0.017 Option-A residual. Caveat: the harness is 1D-only / single-channel by "
        "construction, so it has never covered the 2D residual.",
        "refute_by": "extend the harness to the 2D channel — NOT DONE; the caveat stands",
        "chips": ["CLAIM_2D_BIAS_20260730.md C11", "results/pp_coverage_*/SUMMARY.md"],
    },
]


def build_board() -> dict[str, Any]:
    # Cross-check the two board rows whose numbers this generator can re-measure
    # cheaply and independently (the rest are gated in build_dossier()).
    n_incat_61 = int(
        (pd.read_csv(need(f"{REAL_REL}/seed61000/prepared_cramer_rao_bounds.csv"))["host_galaxy_index"] >= 0).sum()
    )
    n_incat_62 = int(
        (pd.read_csv(need(f"{REAL_REL}/seed62000/prepared_cramer_rao_bounds.csv"))["host_galaxy_index"] >= 0).sum()
    )
    n_rows_61 = int(len(pd.read_csv(need(f"{REAL_REL}/seed61000/prepared_cramer_rao_bounds.csv"))))
    n_rows_62 = int(len(pd.read_csv(need(f"{REAL_REL}/seed62000/prepared_cramer_rao_bounds.csv"))))
    realized = (n_incat_61 + n_incat_62) / (n_rows_61 + n_rows_62)
    GATES.check("C9 realized in-cat rate 164/3135", realized, 164 / 3135, 1e-9, "CLAIM_2D_BIAS C9")

    sides = {
        "measurement": "bookkeeping (both channels)",
        "catalogue": "catalogue leg",
        "completion": "completion leg",
        "prefactor": "mixture weight / prefactor",
        "kernel": "host-z kernel",
        "both": "both legs",
        "meta": "experimental design",
    }
    return {
        "_meta": {
            "chapter": 11,
            "widget": "I11.2 The Adjudication Board",
            "source": "CLAIM_2D_BIAS_20260730.md (as amended 2026-07-30) + "
            "gate_b_20260730/ADJUDICATION_20260730.md §1 — status strings verbatim",
            "adjudicated": "2026-07-30",
        },
        "sides": sides,
        "one_side_ids": ["C7", "C8", "C9"],
        "claims": BOARD,
        "realized_incat_rate": {
            "n_incat": n_incat_61 + n_incat_62,
            "n_rows": n_rows_61 + n_rows_62,
            "rate": rnd(realized, 6),
            "per_seed": {"61000": [n_incat_61, n_rows_61], "62000": [n_incat_62, n_rows_62]},
            "model_w_G_073": 0.1215037,
            "binomial_z": -11.86,
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
        raise SystemExit(f"gen_ch11: {name} exceeds the 500 KB budget ({kb:.1f} KB)")


def main() -> None:
    print("gen_ch11: search roots =", [str(r) for r in SEARCH_ROOTS])
    runaways = build_runaways()
    dossier = build_dossier()
    board = build_board()

    gates = GATES.summary()
    runaways["_gates"] = gates
    write_json("ch11_runaways.json", runaways)
    write_json("ch11_dossier.json", dossier)
    write_json("ch11_board.json", board)

    print(f"  gates: {gates['n']} checks, all_pass={gates['all_pass']}")
    pub = runaways["published"]
    print(
        "  FLAG F-ch11-1: adjudicated dh*/deps leverage ratio "
        f"{pub['claim_leverage_ratio_range']}x vs measured "
        f"{pub['measured_leverage_ratio_range']}x (median {pub['measured_leverage_ratio_median']}x) "
        f"— see {FLAG_FILE.relative_to(BOOK_ROOT.parent)}"
    )


if __name__ == "__main__":
    sys.exit(main())
