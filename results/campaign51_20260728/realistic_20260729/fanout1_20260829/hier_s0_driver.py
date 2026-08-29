#!/usr/bin/env python3
"""[HIER] Stage-0 driver -- S0-A / S0-R / S0-C.

Registration: ``PREREGISTRATION_HIER_HTHETA_20260826.md`` (binding) --
section 1.2 (theta identity, line 40), section 2.1 (Stage-0 arms, lines
139-176), section 4.1 (score definition, lines 384-410), section 5.1
(invariants), GATE ENG / GATE PARITY / GATE D3 (section 3), and the
amendment log PA-HIER-1..30.

**BUILD ONLY -- charter node B1.1, launched under rows #222/#223.** Rule 2
(verifier independence): this file's author may only SMOKE-TEST it; the
registered measurement (the real 4-seed / 5-node grid) must be RUN by a
different agent. The ``--smoke`` flag exists for exactly that boundary --
it truncates events and node count so the *code path* is exercised without
banking a registered number.

Disclosures carried forward from the amendment log (do not re-litigate,
re-check by reading the cited amendment before changing anything below):

* **S0-R is a disclosed NULL INSTRUMENT (PA-HIER-3, PA-HIER-22).**
  ``realize_observed_catalogue(sigma_scale=1.5)`` round-trips the quoted
  ``z_error`` column verbatim and feeds the SAME realized catalogue to both
  the generator and the estimator (``host_pool_for_sigma_scale`` returns one
  shared ``GalaxyCatalogueHandler``) -- ``sigma_kernel == sigma_realized``
  identically at every dose. Truth-theta after the call is still (0, 1), not
  (0, 1.5); no z-kernel misspecification is injected. A real s-axis positive
  control needs new code (C3, NEEDS-CODE, unbuilt) that this driver does not
  attempt to build.
* **PA-HIER-28 item 5 = FALLBACK (author ruling, 2026-08-28): D7's early
  exit is DISARMED and Stage 0 is re-scoped to S0-A + S0-C ONLY.** This
  driver still implements S0-R (per this build task's explicit instruction,
  and because the code is cheap and may be useful if C3 ever lands), but its
  verdict function never emits B0-R/B0-R'/LEVER-DEAD-AT-N -- it reports
  NULL-INSTRUMENT and cites PA-HIER-3/22/28. Treat any S0-R number this
  driver produces as a disclosed diagnostic, not a registered verdict.
* **All [HIER] verdicts are capped REPORTED-ONLY (PA-HIER-28 item 9 =
  AFFORDABLE).** No band in this driver's output may be read as CALIBRATED.
* GATE SEQ (section 3.7): no ``sbatch`` for any [HIER] stage until
  [P3-MKER] stage-1 is banked. This driver runs LOCAL processes only and
  never touches ``sbatch`` -- it is orthogonal to GATE SEQ, but a future
  cluster port of this driver is NOT authorized by this build.

Venue (bc / b0i, PA-HIER-27/28 item 1 RATIFIED): ``host_mode=
"catalogue_selected"``, kwargs copied EXACTLY from
``results/campaign51_20260728/realistic_20260729/p3_b0_identity_test.py``'s
``_run_arm_seed(venue="b0i")`` (module docstring lines ~883-895, body
~915-935) and ``ARM_FLAGS["bc"]`` (line ~997): ``catalogue_numerator_
survival="off"``, ``catalogue_global_selection="phi"``,
``selection_in_completion_numerator="fused"``. Banked bc seeds 900101..
900112 live under ``results/campaign51_20260728/realistic_20260729/
p3_b0_work/bc_<seed>_work/``.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
REALISTIC_DIR = REPO_ROOT / "results/campaign51_20260728/realistic_20260729"
BC_WORK_ROOT = REALISTIC_DIR / "p3_b0_work"
REGISTRATION = REALISTIC_DIR / "PREREGISTRATION_HIER_HTHETA_20260826.md"

import darksiren_emri.validation.correspondence_1d as c1d  # noqa: E402
from darksiren_emri.validation.correspondence_1d import H_GRID_41, H_TRUE  # noqa: E402

# ── Registered constants ──────────────────────────────────────────────────

H_GEN: float = H_TRUE  # 0.73 -- the mirror-universe truth h (constants.H)
# Invariant #2 (registration §5.1): "h grid = H_GRID_41 verbatim, h_bounds =
# (0.50, 0.86) pin". This is ALSO the pin p3_b0_identity_test.py's banked bc
# CSVs were produced under (H_GRID_FULL = H_WING_LOW | H_GRID_41, whose
# min/max are exactly 0.50/0.86) -- P3-HGRID (rows #182-#184) proved a
# single-h caller reproducing a full-grid run's L_cat "must pass
# h_bounds=(min(grid), max(grid)) explicitly (proven bit-exact vs the banked
# b0i CSVs)". Passing it here is what makes GATE PARITY a real, checkable
# claim rather than an apples-to-oranges comparison.
H_BOUNDS: tuple[float, float] = (0.50, 0.86)
assert min(H_GRID_41) >= H_BOUNDS[0] and max(H_GRID_41) <= H_BOUNDS[1]

# bc/b0i venue flags, copied verbatim from p3_b0_identity_test.py:
#   ARM_FLAGS["bc"] = {"catalogue_numerator_survival": "off",
#                       "catalogue_global_selection": "phi"}
#   _run_arm_seed(..., completion_cell="fused")  (its own default)
BC_CATALOGUE_NUMERATOR_SURVIVAL = "off"
BC_CATALOGUE_GLOBAL_SELECTION = "phi"
BC_COMPLETION_CELL = "fused"
BC_EVENT_MEASURE = "ratio"  # c1d.ARM_EVENT_MEASURE.get("b0i", "ratio")

# First 4 of the banked bc span 900101-900112 (12 = first 12 of
# c1d.ARM_SEEDS["b0i"], p3_b0_identity_test.py BSEL_SEEDS).
DEFAULT_BC_SEEDS: tuple[int, ...] = (900101, 900102, 900103, 900104)

# The registered 5-node theta-cross (prereg §2.1 S0-A row / §4.1):
# {(0,1), (+-0.02, 1), (0, 1/sqrt(2)), (0, sqrt(2))} at h = 0.73 only.
THETA_NODES: dict[str, tuple[float, float]] = {
    "truth": (0.0, 1.0),
    "b_plus": (0.02, 1.0),
    "b_minus": (-0.02, 1.0),
    "s_plus": (0.0, math.sqrt(2.0)),
    "s_minus": (0.0, 1.0 / math.sqrt(2.0)),
}
NODE_ORDER: tuple[str, ...] = ("truth", "b_plus", "b_minus", "s_plus", "s_minus")

# S0-R's registered dose (prereg §2.1; DISCLOSED NULL per PA-HIER-3/22 above).
S0_R_SIGMA_SCALE = 1.5

# Band thresholds (prereg §4.1). The 3.0 anchor is NOT chosen by this driver
# -- it is `.claude/skills/research-cycle/SKILL.md`'s own registered
# coherent-class-displacement threshold, cited verbatim in the prereg.
Z_THRESHOLD = 3.0
# GATE ENG (prereg §3.4): >=10% of scored events move by >=1e-6 relative in
# per-event ln L versus truth-theta.
ENG_REL_THRESHOLD = 1e-6
ENG_EVENT_FRACTION = 0.10
# GATE PARITY tolerance for this driver's own truth-node reproduction check
# (distinct from the registration's own GATE PARITY, prereg §3.3, which is
# about correspondence_1d.py:1173's separate stale-comment claim and is NOT
# re-litigated here). We assert exact byte equality is the TARGET (per the
# P3-HGRID note's "proven bit-exact" language); anything not exactly 0 is
# reported at whatever rtol it actually lands at, never silently widened.
PARITY_TARGET_EXACT = True
PARITY_FALLBACK_RTOL = 1e-9

DIAG_VALUE_COLUMNS = ("combined_no_bh", "combined_with_bh")


# ── bc/b0i venue construction (copied exactly, see module docstring) ──────


def build_bc_venue(
    work_root: Path, seed: int, sigma_z_scale: float = 1.0
) -> tuple[pd.DataFrame, Any]:
    """Build one b0i-venue mirror realization.

    Kwargs copied EXACTLY from ``p3_b0_identity_test.py``'s
    ``_run_arm_seed(venue="b0i")`` (as of this driver's authoring, 2026-08-29):
    ``c1d._verify_rate_weight_parity()`` before any draw, ``completeness_obj,
    phi_survival_table = c1d.build_bsel_selection_objects(h_true=H_GEN)``,
    then ``gen.draw_realization(seed, host_pool=host_pool,
    host_mode="catalogue_selected", completeness=completeness_obj,
    phi_survival_table=phi_survival_table)``.

    ``sigma_z_scale`` is the S0-A/S0-R fork: S0-A passes 1.0 (the banked bc
    dose, truth-theta=(0,1)); S0-R passes 1.5 (prereg §2.1's registered
    dose -- disclosed NULL per PA-HIER-3/22, module docstring). Note
    ``CorrespondenceConfig.sigma_z_scale`` is a SEPARATE, unrelated field
    (dead for the ``catalogue_selected`` draw path -- only ``n_events``/
    ``area_scale`` are read off it, per the class docstring); the dose that
    matters is ``host_pool_for_sigma_scale``'s own ``sigma_z_scale`` kwarg,
    which is what this function forwards.

    Returns:
        ``(events, handler)`` -- the realization's synthetic CRB rows and the
        ``GalaxyCatalogueHandler`` built from the (possibly re-realized)
        catalogue, ready for ``run_mirror_seed_inprocess``.
    """
    cfg = c1d.CorrespondenceConfig()  # n_events=200, area_scale=1.0 (b0i ARM_SPECS)
    gen = c1d.MirrorUniverseGenerator(cfg)
    host_pool, _observed_path, handler = gen.host_pool_for_sigma_scale(
        work_root / "catalogue", seed, sigma_z_scale=sigma_z_scale
    )
    c1d._verify_rate_weight_parity()
    completeness_obj, phi_survival_table = c1d.build_bsel_selection_objects(h_true=H_GEN)
    events = gen.draw_realization(
        seed,
        host_pool=host_pool,
        host_mode="catalogue_selected",
        completeness=completeness_obj,
        phi_survival_table=phi_survival_table,
    )
    return events, handler


def run_theta_node(
    work_root: Path,
    events: pd.DataFrame,
    seed: int,
    handler: Any,
    theta_b: float,
    theta_s: float,
) -> tuple[Path, float]:
    """Evaluate one theta node at h = H_GEN only (n_h = 1, prereg §2.1 S0-A/S0-R row).

    ``smear_global_selection`` is forced ``True`` iff theta is engaged
    (``theta_b != 0`` or ``theta_s != 1``) -- theta_sites="all" includes
    site 2.3, which REQUIRES smear_global_selection=True
    (``bayesian_statistics.py`` evaluate()'s own guard). This keeps the
    truth node (identity theta) on the byte-identical unsmeared path (GATE
    PARITY) while off-truth nodes engage the registered site-2.3 kernel
    (GATE ENG) -- an explicit driver-level decision, disclosed in
    ``B1_1_HIER_BUILD_NOTE.md`` section "ambiguities resolved", extending
    GATE D3(a)'s stated principle (force the branch on engagement, never
    leave it to a separately-set flag) to this driver's own dispatch.
    """
    theta_engaged = theta_b != 0.0 or theta_s != 1.0
    diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
        work_root,
        events,
        seed,
        galaxy_catalog=handler,
        h_values=(H_GEN,),
        h_bounds=H_BOUNDS,
        selection_in_completion_numerator=BC_COMPLETION_CELL,
        completion_event_measure=BC_EVENT_MEASURE,
        catalogue_numerator_survival=BC_CATALOGUE_NUMERATOR_SURVIVAL,
        catalogue_global_selection=BC_CATALOGUE_GLOBAL_SELECTION,
        theta_b=theta_b,
        theta_s=theta_s,
        theta_sites="all",
        smear_global_selection=theta_engaged,
    )
    return diag_csv, elapsed


# ── Diagnostics readback ───────────────────────────────────────────────────


def read_event_ln_l(diag_csv: Path, h: float, rtol: float = 1e-9) -> pd.DataFrame:
    """Read ``event_likelihoods.csv`` and return per-event ln L at ``h``.

    Columns returned: ``event_idx``, ``ln_L_no_bh``, ``ln_L_with_bh`` (NaN
    where the corresponding ``combined_*`` column is non-positive -- the
    same non-positivity the estimator's own ``num_log_term_*`` diagnostic
    columns guard against, bayesian_statistics.py:5788-5794).
    """
    df = pd.read_csv(diag_csv)
    mask = np.isclose(df["h"].to_numpy(dtype=float), h, rtol=rtol, atol=1e-12)
    sub = df.loc[mask, ["event_idx", *DIAG_VALUE_COLUMNS]].copy()
    if sub.empty:
        raise RuntimeError(f"no rows at h={h!r} in {diag_csv} (h values present: {sorted(set(df['h']))})")
    sub = sub.drop_duplicates(subset="event_idx", keep="last")
    for col, out in (("combined_no_bh", "ln_L_no_bh"), ("combined_with_bh", "ln_L_with_bh")):
        vals = sub[col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sub[out] = np.where(vals > 0.0, np.log(vals), np.nan)
    return sub[["event_idx", "ln_L_no_bh", "ln_L_with_bh"]].sort_values("event_idx").reset_index(drop=True)


@dataclass
class NodeResult:
    node: str
    theta_b: float
    theta_s: float
    seed: int
    diag_csv: str
    elapsed_s: float
    n_events: int
    ln_l: pd.DataFrame = field(repr=False)


def run_arm_seed_s0a(
    seed: int,
    out_root: Path,
    nodes: tuple[str, ...],
    event_cap: int | None,
) -> list[NodeResult]:
    """S0-A: one seed, the theta-cross at h=H_GEN, sigma_z_scale=1.0 (truth-theta=(0,1))."""
    work_root = out_root / f"s0a_seed{seed}"
    work_root.mkdir(parents=True, exist_ok=True)
    events, handler = build_bc_venue(work_root, seed, sigma_z_scale=1.0)
    if event_cap is not None:
        events = events.head(event_cap).reset_index(drop=True)
    results: list[NodeResult] = []
    for node in nodes:
        theta_b, theta_s = THETA_NODES[node]
        node_root = work_root / f"node_{node}"
        node_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        diag_csv, elapsed = run_theta_node(node_root, events, seed, handler, theta_b, theta_s)
        wall = time.time() - t0
        ln_l = read_event_ln_l(diag_csv, H_GEN)
        results.append(
            NodeResult(
                node=node,
                theta_b=theta_b,
                theta_s=theta_s,
                seed=seed,
                diag_csv=str(diag_csv),
                elapsed_s=elapsed,
                n_events=len(ln_l),
                ln_l=ln_l,
            )
        )
        print(
            f"[S0-A seed={seed} node={node} theta=({theta_b},{theta_s})] "
            f"n_events={len(ln_l)} evaluate_s={elapsed:.2f} wall_s={wall:.2f} -> {diag_csv}",
            flush=True,
        )
    return results


def run_arm_seed_s0r(
    seed: int,
    out_root: Path,
    nodes: tuple[str, ...],
    event_cap: int | None,
) -> list[NodeResult]:
    """S0-R: one seed, the theta-cross at h=H_GEN, sigma_z_scale=1.5 (DISCLOSED NULL, see module docstring)."""
    work_root = out_root / f"s0r_seed{seed}"
    work_root.mkdir(parents=True, exist_ok=True)
    events, handler = build_bc_venue(work_root, seed, sigma_z_scale=S0_R_SIGMA_SCALE)
    if event_cap is not None:
        events = events.head(event_cap).reset_index(drop=True)
    results: list[NodeResult] = []
    for node in nodes:
        theta_b, theta_s = THETA_NODES[node]
        node_root = work_root / f"node_{node}"
        node_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        diag_csv, elapsed = run_theta_node(node_root, events, seed, handler, theta_b, theta_s)
        wall = time.time() - t0
        ln_l = read_event_ln_l(diag_csv, H_GEN)
        results.append(
            NodeResult(
                node=node,
                theta_b=theta_b,
                theta_s=theta_s,
                seed=seed,
                diag_csv=str(diag_csv),
                elapsed_s=elapsed,
                n_events=len(ln_l),
                ln_l=ln_l,
            )
        )
        print(
            f"[S0-R seed={seed} node={node} theta=({theta_b},{theta_s})] "
            f"n_events={len(ln_l)} evaluate_s={elapsed:.2f} wall_s={wall:.2f} -> {diag_csv}",
            flush=True,
        )
    return results


def run_seed_s0c(seed: int, out_root: Path, event_cap: int | None) -> dict[str, Any]:
    """S0-C: one seed, theta=(0,1), the full 41-node H_GRID_41 (costing probe, prereg §2.1)."""
    work_root = out_root / f"s0c_seed{seed}"
    work_root.mkdir(parents=True, exist_ok=True)
    events, handler = build_bc_venue(work_root, seed, sigma_z_scale=1.0)
    if event_cap is not None:
        events = events.head(event_cap).reset_index(drop=True)
    node_root = work_root / "node_truth_fullgrid"
    node_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    diag_csv, elapsed = c1d.run_mirror_seed_inprocess(
        node_root,
        events,
        seed,
        galaxy_catalog=handler,
        h_values=H_GRID_41,
        h_bounds=H_BOUNDS,
        selection_in_completion_numerator=BC_COMPLETION_CELL,
        completion_event_measure=BC_EVENT_MEASURE,
        catalogue_numerator_survival=BC_CATALOGUE_NUMERATOR_SURVIVAL,
        catalogue_global_selection=BC_CATALOGUE_GLOBAL_SELECTION,
        theta_b=0.0,
        theta_s=1.0,
        theta_sites="all",
        smear_global_selection=False,
    )
    wall = time.time() - t0
    # Per-h marginal cost: posterior JSONs are written progressively as each
    # h completes (bayesian_statistics.py ~4600-4633), so their mtimes give
    # real per-h wall-clock deltas -- the actual point of S0-C (prereg §7:
    # "the MEASURED marginal per-h cost").
    posteriors_dir = node_root / "simulations" / "posteriors"
    per_h_files = sorted(posteriors_dir.glob("h_*.json"), key=lambda p: p.stat().st_mtime)
    mtimes = [p.stat().st_mtime for p in per_h_files]
    deltas = [mtimes[0] - t0] + [b - a for a, b in zip(mtimes, mtimes[1:])]
    return {
        "seed": seed,
        "n_h": len(H_GRID_41),
        "diag_csv": str(diag_csv),
        "evaluate_s": elapsed,
        "wall_s": wall,
        "per_h_files": [p.name for p in per_h_files],
        "per_h_delta_s": deltas,
        "mean_marginal_h_cost_s": (float(np.mean(deltas[1:])) if len(deltas) > 1 else None),
        "first_h_cost_s": deltas[0] if deltas else None,
    }


# ── Registered statistics (prereg §4.1) ─────────────────────────────────────


def compute_scores(all_nodes: dict[str, list[NodeResult]]) -> dict[str, Any]:
    """Pool per-event score_b/score_s over every event and seed (prereg §4.1).

    ``all_nodes`` maps node name -> list of NodeResult (one per seed), each
    already restricted to the SAME event_cap-truncated event set per seed
    (event sets differ ACROSS seeds -- that is fine, pooling is over the
    union of (seed, event_idx) pairs, not a per-event paired comparison).

    score_b = [lnL(b=+0.02,s=1) - lnL(b=-0.02,s=1)] / 0.04
    score_s = [lnL(b=0,s=sqrt2) - lnL(b=0,s=1/sqrt2)] / (sqrt2 - 1/sqrt2)
    Z_x = mean(score_x) / SEM(score_x), pooled over events and seeds.
    """
    channels = ("ln_L_no_bh", "ln_L_with_bh")
    out: dict[str, Any] = {}
    for channel in channels:
        # Join b_plus/b_minus per (seed, event_idx).
        bp = pd.concat(
            [r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]] for r in all_nodes["b_plus"]],
            ignore_index=True,
        ).rename(columns={channel: "b_plus"})
        bm = pd.concat(
            [r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]] for r in all_nodes["b_minus"]],
            ignore_index=True,
        ).rename(columns={channel: "b_minus"})
        sp = pd.concat(
            [r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]] for r in all_nodes["s_plus"]],
            ignore_index=True,
        ).rename(columns={channel: "s_plus"})
        sm = pd.concat(
            [r.ln_l.assign(seed=r.seed)[["seed", "event_idx", channel]] for r in all_nodes["s_minus"]],
            ignore_index=True,
        ).rename(columns={channel: "s_minus"})

        b_join = bp.merge(bm, on=["seed", "event_idx"], how="inner")
        s_join = sp.merge(sm, on=["seed", "event_idx"], how="inner")

        score_b = (b_join["b_plus"] - b_join["b_minus"]) / 0.04
        denom_s = math.sqrt(2.0) - 1.0 / math.sqrt(2.0)
        score_s = (s_join["s_plus"] - s_join["s_minus"]) / denom_s

        def _mean_sem(series: pd.Series) -> tuple[float, float, float, int]:
            vals = series.to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = vals.size
            if n == 0:
                return float("nan"), float("nan"), float("nan"), 0
            mean = float(np.mean(vals))
            sem = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
            z = mean / sem if sem and np.isfinite(sem) and sem > 0 else float("nan")
            return mean, sem, z, n

        mean_b, sem_b, z_b, n_b = _mean_sem(score_b)
        mean_s, sem_s, z_s, n_s = _mean_sem(score_s)
        out[channel] = {
            "score_b": {"mean": mean_b, "sem": sem_b, "Z": z_b, "n_pooled": n_b},
            "score_s": {"mean": mean_s, "sem": sem_s, "Z": z_s, "n_pooled": n_s},
        }
    return out


def gate_eng(all_nodes: dict[str, list[NodeResult]], channel: str = "ln_L_no_bh") -> dict[str, Any]:
    """GATE ENG (prereg §3.4): >=10% of scored events move by >=1e-6 relative vs truth, per node."""
    truth_by_seed = {r.seed: r.ln_l.set_index("event_idx")[channel] for r in all_nodes["truth"]}
    out: dict[str, Any] = {}
    for node in NODE_ORDER:
        if node == "truth":
            continue
        fracs = []
        for r in all_nodes.get(node, []):
            truth = truth_by_seed.get(r.seed)
            if truth is None:
                continue
            node_vals = r.ln_l.set_index("event_idx")[channel]
            common = node_vals.index.intersection(truth.index)
            if len(common) == 0:
                continue
            a = node_vals.loc[common].to_numpy(dtype=float)
            b = truth.loc[common].to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b) & (b != 0.0)
            if finite.sum() == 0:
                continue
            rel = np.abs(a[finite] - b[finite]) / np.abs(b[finite])
            frac_moved = float(np.mean(rel >= ENG_REL_THRESHOLD))
            fracs.append(frac_moved)
        mean_frac = float(np.mean(fracs)) if fracs else float("nan")
        out[node] = {
            "per_seed_fraction_moved": fracs,
            "mean_fraction_moved": mean_frac,
            "pass": bool(np.isfinite(mean_frac) and mean_frac >= ENG_EVENT_FRACTION),
        }
    return out


def gate_parity(
    all_nodes: dict[str, list[NodeResult]], bc_work_root: Path = BC_WORK_ROOT
) -> dict[int, dict[str, Any]]:
    """GATE PARITY (this driver's own check, distinct from prereg §3.3's correspondence_1d.py:1173
    claim, NOT re-litigated here): theta=(0,1) reproduces the banked bc seed's h=0.73 row.

    Compares this driver's truth-node ``combined_no_bh``/``combined_with_bh``
    values against the banked ``bc_<seed>_work/seed<seed>/simulations/
    diagnostics/event_likelihoods.csv`` at h=H_GEN, for the SAME event_idx
    values (valid because event_idx is assigned by row order and this
    driver's ``events`` truncation, if any, only ever DROPS trailing rows --
    see ``run_arm_seed_s0a``'s docstring / the build note).
    """
    out: dict[int, dict[str, Any]] = {}
    for r in all_nodes.get("truth", []):
        banked_csv = bc_work_root / f"bc_{r.seed}_work" / f"seed{r.seed}" / "simulations" / "diagnostics" / "event_likelihoods.csv"
        if not banked_csv.is_file():
            out[r.seed] = {"status": "NO_BANKED_CSV", "path": str(banked_csv)}
            continue
        banked = read_event_ln_l(banked_csv, H_GEN)
        merged = r.ln_l.merge(banked, on="event_idx", suffixes=("_driver", "_banked"))
        if merged.empty:
            out[r.seed] = {"status": "NO_OVERLAPPING_EVENTS"}
            continue
        diffs: dict[str, dict[str, Any]] = {}
        for chan in ("ln_L_no_bh", "ln_L_with_bh"):
            a = merged[f"{chan}_driver"].to_numpy(dtype=float)
            b = merged[f"{chan}_banked"].to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() == 0:
                diffs[chan] = {"n": 0}
                continue
            abs_diff = np.abs(a[finite] - b[finite])
            rel_diff = abs_diff / np.maximum(np.abs(b[finite]), 1e-300)
            diffs[chan] = {
                "n": int(finite.sum()),
                "max_abs_diff": float(np.max(abs_diff)),
                "max_rel_diff": float(np.max(rel_diff)),
                "exact": bool(np.max(abs_diff) == 0.0),
            }
        out[r.seed] = {
            "status": "COMPARED",
            "n_events_compared": len(merged),
            "diffs": diffs,
            "pass_exact": all(d.get("exact", False) for d in diffs.values() if d.get("n", 0) > 0),
            "pass_fallback_rtol": all(
                d.get("max_rel_diff", 1.0) <= PARITY_FALLBACK_RTOL for d in diffs.values() if d.get("n", 0) > 0
            ),
        }
    return out


def verdict_s0a(
    scores: dict[str, Any], eng: dict[str, Any], parity: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Prereg §4.1 B0-A / B0-A' verdict line -- the RUNNER (not the builder) reads this off real data."""
    z_b = scores["ln_L_no_bh"]["score_b"]["Z"]
    z_s = scores["ln_L_no_bh"]["score_s"]["Z"]
    control_null = bool(
        np.isfinite(z_b) and np.isfinite(z_s) and abs(z_b) <= Z_THRESHOLD and abs(z_s) <= Z_THRESHOLD
    )
    return {
        "band": "B0-A" if control_null else "B0-A'",
        "verdict": "EXPECTED-NULL, instrument certified, proceed"
        if control_null
        else "INSTRUMENT-DEFECT -- STOP (prereg §4.5)",
        "Z_b": z_b,
        "Z_s": z_s,
        "threshold": Z_THRESHOLD,
        "channel": "ln_L_no_bh (registered primary; ln_L_with_bh reported alongside, see scores)",
        "gate_eng": eng,
        "gate_parity": parity,
        "reported_only": True,  # PA-HIER-28 item 9 = AFFORDABLE
    }


def verdict_s0r(scores: dict[str, Any], eng: dict[str, Any]) -> dict[str, Any]:
    """S0-R is a DISCLOSED NULL INSTRUMENT (PA-HIER-3/22) and PA-HIER-28 item 5 = FALLBACK
    disarms D7's early exit. This function NEVER emits B0-R/B0-R'/LEVER-DEAD-AT-N -- doing so
    would bank a verdict about an axis the amendment log proved this instrument cannot move.
    """
    return {
        "band": "NULL-INSTRUMENT (not B0-R/B0-R')",
        "verdict": (
            "S0-R AS CONSTRUCTED CANNOT BANK A LEVER VERDICT -- PA-HIER-3/22 proved "
            "sigma_kernel == sigma_realized identically at every sigma_scale (one shared "
            "GalaxyCatalogueHandler serves generator and estimator); truth-theta after the "
            "realize_observed_catalogue(sigma_scale=1.5) call is still (0,1), not (0,1.5). "
            "PA-HIER-28 item 5 (author ruling) DISARMS D7's early exit and re-scopes Stage 0 "
            "to S0-A + S0-C only. The numbers below are a disclosed diagnostic (do the "
            "z-mismatch and mass-mismatch scores at least land somewhere sane), never a "
            "registered read."
        ),
        "diagnostic_scores": scores,
        "gate_eng": eng,
        "reported_only": True,
    }


# ── Multi-seed orchestration with a bounded process budget ─────────────────


def _run_one_seed_worker(args: tuple[str, int, Path, tuple[str, ...], int | None, int]) -> Any:
    """Top-level (picklable) worker: pin this process's CPU affinity to a budget, then run one
    seed's cells for the given arm. Affinity pinning (not a num_workers kwarg -- that plumbing
    does not exist in run_mirror_seed_inprocess) is how this driver keeps
    ``BayesianStatistics.evaluate``'s own ``available_cpus - 2`` auto-sizing
    (bayesian_statistics.py:4490-4495) from oversubscribing when several seeds run concurrently.
    """
    arm, seed, out_root, nodes, event_cap, cpu_budget = args
    try:
        all_cpus = sorted(os.sched_getaffinity(0))
        budget = max(1, min(cpu_budget, len(all_cpus)))
        os.sched_setaffinity(0, set(all_cpus[:budget]))
    except (AttributeError, OSError):
        pass  # affinity control unavailable (e.g. non-Linux); proceed unpinned, disclosed
    try:
        if arm == "S0-A":
            results = run_arm_seed_s0a(seed, out_root, nodes, event_cap)
        elif arm == "S0-R":
            results = run_arm_seed_s0r(seed, out_root, nodes, event_cap)
        elif arm == "S0-C":
            return {"seed": seed, "s0c": run_seed_s0c(seed, out_root, event_cap)}
        else:
            raise ValueError(f"unknown arm {arm!r}")
        return {
            "seed": seed,
            "nodes": {
                r.node: {
                    "theta_b": r.theta_b,
                    "theta_s": r.theta_s,
                    "diag_csv": r.diag_csv,
                    "elapsed_s": r.elapsed_s,
                    "n_events": r.n_events,
                    "ln_l_records": r.ln_l.to_dict(orient="records"),
                }
                for r in results
            },
        }
    except Exception as exc:  # noqa: BLE001 -- surfaced to the parent as a structured failure
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _records_to_node_results(seed: int, nodes_payload: dict[str, Any]) -> list[NodeResult]:
    out = []
    for node, payload in nodes_payload.items():
        ln_l = pd.DataFrame.from_records(payload["ln_l_records"])
        out.append(
            NodeResult(
                node=node,
                theta_b=payload["theta_b"],
                theta_s=payload["theta_s"],
                seed=seed,
                diag_csv=payload["diag_csv"],
                elapsed_s=payload["elapsed_s"],
                n_events=payload["n_events"],
                ln_l=ln_l,
            )
        )
    return out


def run_arm(
    arm: str,
    seeds: tuple[int, ...],
    out_root: Path,
    jobs: int,
    total_cpu_budget: int,
    nodes: tuple[str, ...],
    event_cap: int | None,
) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = max(1, min(jobs, len(seeds)))
    cpu_per_job = max(1, total_cpu_budget // jobs)
    task_args = [(arm, seed, out_root, nodes, event_cap, cpu_per_job) for seed in seeds]

    t0 = time.time()
    if jobs == 1:
        raw_results = [_run_one_seed_worker(a) for a in task_args]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=jobs) as pool:
            raw_results = pool.map(_run_one_seed_worker, task_args)
    wall_s = time.time() - t0

    errors = [r for r in raw_results if "error" in r]
    ok = [r for r in raw_results if "error" not in r]

    payload: dict[str, Any] = {
        "arm": arm,
        "seeds_requested": list(seeds),
        "jobs": jobs,
        "cpu_per_job": cpu_per_job,
        "wall_s": wall_s,
        "n_seeds_ok": len(ok),
        "n_seeds_error": len(errors),
        "errors": errors,
    }

    if arm == "S0-C":
        payload["per_seed"] = [r["s0c"] for r in ok]
        return payload

    all_nodes: dict[str, list[NodeResult]] = {n: [] for n in nodes}
    for r in ok:
        for node, node_results in _grouped_by_node(r["seed"], r["nodes"]).items():
            all_nodes[node].extend(node_results)

    payload["per_seed_summary"] = [
        {
            "seed": r["seed"],
            "nodes": {n: {"elapsed_s": p["elapsed_s"], "n_events": p["n_events"]} for n, p in r["nodes"].items()},
        }
        for r in ok
    ]

    need_all_four = all(n in nodes for n in ("b_plus", "b_minus", "s_plus", "s_minus"))
    if need_all_four:
        scores = compute_scores(all_nodes)
        eng = gate_eng(all_nodes)
        payload["scores"] = scores
        payload["gate_eng"] = eng
        if arm == "S0-A":
            parity = gate_parity(all_nodes)
            payload["gate_parity"] = parity
            payload["verdict"] = verdict_s0a(scores, eng, parity)
        elif arm == "S0-R":
            payload["verdict"] = verdict_s0r(scores, eng)
    else:
        payload["note"] = (
            f"nodes={nodes} does not include all 4 off-truth nodes -- scores/gates/verdict "
            "require {b_plus,b_minus,s_plus,s_minus}; this is expected for a --smoke run with "
            "1-2 nodes and is NOT a registered read."
        )
        if "truth" in nodes and arm == "S0-A":
            # Still run GATE PARITY if the truth node is present -- it needs no other node.
            payload["gate_parity"] = gate_parity(all_nodes)

    return payload


def _grouped_by_node(seed: int, nodes_payload: dict[str, Any]) -> dict[str, list[NodeResult]]:
    out: dict[str, list[NodeResult]] = {}
    for r in _records_to_node_results(seed, nodes_payload):
        out.setdefault(r.node, []).append(r)
    return out


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("S0-A", "S0-R", "S0-C"))
    ap.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; default is the first 4 banked bc seeds "
        f"{DEFAULT_BC_SEEDS} (S0-C uses only the first of these unless overridden).",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default=str(REALISTIC_DIR / "fanout1_20260829" / "hier_s0_work"),
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Mirror cells (seeds) run concurrently; choose so jobs * cpu-per-job <= --total-cpu-budget.",
    )
    ap.add_argument(
        "--total-cpu-budget",
        type=int,
        default=14,
        help="Total CPUs this driver may use across all concurrent cells (repo convention: leave 2 free of nproc).",
    )
    ap.add_argument(
        "--nodes",
        type=str,
        default=None,
        help="Comma-separated theta-node names (subset of truth,b_plus,b_minus,s_plus,s_minus); "
        "default is all 5 for S0-A/S0-R (ignored for S0-C, which is always the truth node).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 seed, a small node subset (default: truth,b_plus), and a tiny event cap. "
        "Builder-only per rule 2 -- never bank a --smoke output as the registered measurement.",
    )
    ap.add_argument(
        "--event-cap",
        type=int,
        default=None,
        help="Truncate each seed's realized events to the first N rows (order-preserving, so "
        "event_idx values remain comparable to the banked full-N CSVs for GATE PARITY). "
        "--smoke implies a small cap unless this is set explicitly.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)

    seeds: tuple[int, ...]
    nodes: tuple[str, ...]
    if args.smoke:
        seeds = (int(args.seeds.split(",")[0]) if args.seeds else DEFAULT_BC_SEEDS[0],)
        nodes = tuple(args.nodes.split(",")) if args.nodes else ("truth", "b_plus")
        event_cap = args.event_cap if args.event_cap is not None else 12
        jobs = 1
    else:
        seeds = tuple(int(x) for x in args.seeds.split(",")) if args.seeds else DEFAULT_BC_SEEDS
        nodes = tuple(args.nodes.split(",")) if args.nodes else NODE_ORDER
        event_cap = args.event_cap
        jobs = args.jobs

    for n in nodes:
        if n not in THETA_NODES:
            raise SystemExit(f"unknown node {n!r}; must be one of {sorted(THETA_NODES)}")

    result = run_arm(args.arm, seeds, out_root, jobs, args.total_cpu_budget, nodes, event_cap)
    result["smoke"] = bool(args.smoke)
    result["event_cap"] = event_cap
    result["registration"] = str(REGISTRATION)

    out_root.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out_json = out_root / f"{args.arm.lower().replace('-', '')}_{tag}_output.json"
    out_json.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_json}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_seed_summary",)}, indent=2, default=str)[:4000])
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
