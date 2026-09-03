"""BUILDER B2 -- full per-event directional leave-one-out influence on mean_h.

r-offset-subset, Phase B (b-offset-subset-scorer). REGISTRATION_DRAFT.md Sec.2/Sec.3/Sec.6 (G-2).

Computes, for every scored event (event_idx), both venues (iiib, joint_r1) and
both channels (combined_no_bh = "1D", combined_with_bh = "2D"), the full
leave-one-out directional influence on ``mean_h`` under the FROZEN T0
CONVENTION -- reused verbatim (by re-implementation with citation, not
import, to keep this script import-independent) from
``results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py``:
gradient-trapezoid grid weights (``w = np.gradient(h_grid)``), the
physics-floor zero-handling rule (a row's own zeros -> its minimum nonzero
value; an all-zero row is excluded), log-sum combination, uniform prior.

Definitions (REGISTRATION_DRAFT.md Sec.2, verified algebraically identical to
``exec/rd-2d-bootstrap-jackknife/rd_2d_bootstrap_jackknife.py``'s
``directional_influence``):

    infl_e   = mean_h(full) - mean_h(full - e)
    d_e      = sign(TRUTH - mean_h(full)) * (-infl_e)

``d_e > 0`` means removing event e moves mean_h TOWARD truth (0.73). The
high-influence subset is ranked by DECREASING d_e (most-helpful-to-remove
first) -- this script reports the rank, it does not truncate to the banked k;
the banked k (82/94/72/46) is the registered subset boundary, defined
elsewhere (Sec.2: "S is defined by the BANKED k, not re-derived").

Hard blindness (Sec.3 Phase B, Sec.10): this script NEVER opens
``covariate_table_blind*.csv`` and never computes AUC/OR/Holm-p/Delta_strat
(the registered separation/materiality aggregates) -- those belong to Phase C
(the reader) exclusively, over the registered population. This script computes
and reports ONLY the per-event influence vector, the four full-sample
mean_h values, the four minimal-k values, and the four top-10 |influence|
lists (raw numbers, no anchor comparison -- REGISTRATION_DRAFT.md's mandate:
"DO NOT compare them to the anchor values yourself; the verifier does").

Inputs are md5-pinned (REGISTRATION_DRAFT.md Sec.1); STOP (nonzero exit, no
CSV written) on any mismatch -- CLAUDE.md dataset-pinning rule.

Outputs:
  - influence_iiib.csv       (event_idx, influence_2D, influence_1D, rank)
  - influence_joint_r1.csv   (event_idx, influence_2D, influence_1D, rank)
  - BUILD_RECORD_B2.md        full-sample mean_h (10 s.f.), minimal-subset k,
                               top-10 |influence| lists, per venue/channel.

``influence_2D``/``influence_1D`` are the directional statistic d_e (signed,
positive = toward truth). ``rank`` is the rank by decreasing d_e for the 2D
channel (the registered PRIMARY family, iiib 2D / REGISTRATION_DRAFT.md Sec.4-5)
-- rank 1 is the single most-helpful-to-remove event for that venue's 2D
channel. The 1D channel's own ranking is fully recoverable by re-sorting
``influence_1D`` in the output CSV; a second rank column was not requested by
the build mandate and is not added, to avoid inventing an unregistered column.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/jasper/Repositories/darksiren-emri")
NODE_DIR = (
    REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-offset-subset"
)

VENUE_CSV = {
    "iiib": REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved"
    / "run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv",
    "joint_r1": REPO_ROOT
    / "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved"
    / "run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv",
}

# REGISTRATION_DRAFT.md Sec.1 pins -- STOP on mismatch.
VENUE_CSV_MD5 = {
    "iiib": "8e6a2c18dc5838dd1d52641589243672",
    "joint_r1": "745954a0fdee5f10878fb5e622a06144",
}

CHANNELS = ("combined_no_bh", "combined_with_bh")  # 1D, 2D
CHANNEL_LABEL = {"combined_no_bh": "1D", "combined_with_bh": "2D"}
TRUTH = 0.73

# REGISTRATION_DRAFT.md Sec.2 -- the banked k (NOT re-derived as a free
# parameter here; reported alongside the recomputed minimal_k for the
# verifier's comparison, per Sec.6 G-2(ii)).
BANKED_K = {
    ("iiib", "combined_with_bh"): 82,
    ("iiib", "combined_no_bh"): 94,
    ("joint_r1", "combined_with_bh"): 72,
    ("joint_r1", "combined_no_bh"): 46,
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_pins() -> None:
    for venue, path in VENUE_CSV.items():
        if not path.exists():
            raise SystemExit(f"STOP: {venue} CSV not found at {path}")
        actual = _md5(path)
        expected = VENUE_CSV_MD5[venue]
        if actual != expected:
            raise SystemExit(
                f"STOP: {venue} event_likelihoods.csv md5 mismatch -- "
                f"expected {expected}, got {actual} (path={path})"
            )
        print(f"[pin OK] {venue}: {actual}")


def _physics_floor_apply(likelihoods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim per-row rule (P7-2c), reused from tier0_bootstrap_jackknife.py.

    Zeros -> the row's own minimum nonzero value; an all-zero row has no
    nonzero value to floor from and is marked for exclusion instead.
    """
    result = likelihoods.copy()
    n_events = result.shape[0]
    exclude_mask = np.zeros(n_events, dtype=bool)
    for i in range(n_events):
        row = result[i]
        zero_mask = row == 0.0
        if not zero_mask.any():
            continue
        nonzero = row[~zero_mask]
        if nonzero.size == 0:
            exclude_mask[i] = True
        else:
            result[i, zero_mask] = float(nonzero.min())
    return result, exclude_mask


def _load_matrix(csv_path: Path, channel: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (h_grid sorted, event_idx array, logL matrix [n_events, n_h], n_excluded)."""
    df = pd.read_csv(csv_path)
    h_grid = np.sort(df["h"].unique())
    piv = df.pivot(index="event_idx", columns="h", values=channel).reindex(columns=h_grid)
    if piv.isna().any().any():
        raise ValueError(f"{csv_path}/{channel}: pivot has missing (event, h) cells -- ragged CSV")
    event_idx = piv.index.to_numpy()
    L = piv.to_numpy(dtype=np.float64)
    L_floored, exclude_mask = _physics_floor_apply(L)
    n_excluded = int(exclude_mask.sum())
    if n_excluded:
        L_floored = L_floored[~exclude_mask]
        event_idx = event_idx[~exclude_mask]
    logL = np.log(L_floored)
    return h_grid, event_idx, logL, n_excluded


def _moments(
    logpost: np.ndarray, h_grid: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient-trapezoid-weighted (mean_h, sigma_h, MAP) -- T0 convention verbatim."""
    lp = logpost - logpost.max(axis=-1, keepdims=True)
    post = np.exp(lp)
    norm = (post * weights).sum(axis=-1)
    post_n = post / norm[..., None]
    mean_h = (post_n * h_grid * weights).sum(axis=-1)
    var = (post_n * (h_grid - mean_h[..., None]) ** 2 * weights).sum(axis=-1)
    sigma_h = np.sqrt(np.clip(var, 0.0, None))
    map_h = h_grid[np.argmax(logpost, axis=-1)]
    return mean_h, sigma_h, map_h


def _score_venue_channel(csv_path: Path, venue: str, channel: str) -> dict[str, Any]:
    h_grid, event_idx, logL, n_excluded = _load_matrix(csv_path, channel)
    weights = np.gradient(h_grid)
    n_events, n_h = logL.shape

    logpost_full = logL.sum(axis=0)
    mean_arr, sigma_arr, map_arr = _moments(logpost_full[None, :], h_grid, weights)
    mean_h_full, sigma_h_full, map_h_full = float(mean_arr[0]), float(sigma_arr[0]), float(map_arr[0])

    # k = n_events endpoint (all events removed -> flat weighted posterior);
    # independent of the CSV data, a check on H_GRID_41 symmetry (G-2 iv).
    logpost_empty = np.zeros_like(logpost_full)
    mean_empty_arr, _s, _m = _moments(logpost_empty[None, :], h_grid, weights)
    mean_h_all_removed = float(mean_empty_arr[0])

    # --- full leave-one-out influence and directional statistic -----------
    loo_logpost = logpost_full[None, :] - logL  # (n_events, n_h)
    loo_mean_h, _loo_sigma_h, _loo_map_h = _moments(loo_logpost, h_grid, weights)
    influence = mean_h_full - loo_mean_h  # infl_e = mean_h(full) - mean_h(full - e)

    sign_toward_truth = 1.0 if (TRUTH - mean_h_full) >= 0 else -1.0
    d_e = sign_toward_truth * (-influence)  # REGISTRATION_DRAFT.md Sec.2 definition

    order = np.argsort(-d_e)  # decreasing d_e: most-helpful-to-remove first
    rank = np.empty(n_events, dtype=np.int64)
    rank[order] = np.arange(1, n_events + 1)

    # --- minimal-subset k (recomputed here as the G-2(ii) byte-id anchor; ---
    # --- NOT compared to the banked value by this builder) -----------------
    minimal_k: int | None = None
    for k in range(0, n_events + 1):
        if k == 0:
            logpost_k = logpost_full
        else:
            dropped = order[:k]
            logpost_k = logpost_full - logL[dropped].sum(axis=0)
        mean_k_arr, _s2, _m2 = _moments(logpost_k[None, :], h_grid, weights)
        if abs(float(mean_k_arr[0]) - TRUTH) <= sigma_h_full:
            minimal_k = k
            break
    if minimal_k is None:
        minimal_k = n_events

    # Two distinct top-10 lists are reported (see BUILD_RECORD_B2.md notes):
    # (1) literal top-10 by |influence| -- the build-mandate wording; (2)
    # top-10 by decreasing directional influence d_e for THIS channel --
    # this is what the reference JSON's ``top10_events_by_abs_influence``
    # field actually contains (its name notwithstanding: it is populated
    # from ``order[:10]`` where ``order = argsort(-directional_influence)``,
    # not from an abs-value sort -- verified by cross-check against
    # rd_2d_bootstrap_jackknife_output.json). Reporting both avoids this
    # builder silently picking the wrong one for the G-2(iii) byte-id
    # anchor.
    top10_abs_order = np.argsort(-np.abs(influence))[:10]
    top10_abs = [
        {"event_idx": int(event_idx[i]), "influence": float(influence[i]), "d_e": float(d_e[i])}
        for i in top10_abs_order
    ]
    top10_directional_order = order[:10]
    top10_directional = [
        {"event_idx": int(event_idx[i]), "influence": float(influence[i]), "d_e": float(d_e[i])}
        for i in top10_directional_order
    ]

    return {
        "venue": venue,
        "channel": channel,
        "channel_label": CHANNEL_LABEL[channel],
        "n_events": n_events,
        "n_events_excluded_physics_floor": n_excluded,
        "n_h": n_h,
        "event_idx": event_idx,
        "influence": influence,
        "d_e": d_e,
        "rank": rank,
        "mean_h_full": mean_h_full,
        "sigma_h_full": sigma_h_full,
        "map_h_full": map_h_full,
        "mean_h_all_removed": mean_h_all_removed,
        "minimal_k_recomputed": minimal_k,
        "banked_k": BANKED_K[(venue, channel)],
        "top10_by_abs_influence": top10_abs,
        "top10_by_directional_influence": top10_directional,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=NODE_DIR)
    args = parser.parse_args(argv)

    _verify_pins()

    per_venue_channel: dict[tuple[str, str], dict[str, Any]] = {}
    for venue, csv_path in VENUE_CSV.items():
        for channel in CHANNELS:
            print(f"Scoring {venue}/{CHANNEL_LABEL[channel]} ...", flush=True)
            per_venue_channel[(venue, channel)] = _score_venue_channel(csv_path, venue, channel)

    # --- write per-venue CSVs (event_idx, influence_2D, influence_1D, rank) -
    csv_paths: dict[str, Path] = {}
    for venue in VENUE_CSV:
        r2d = per_venue_channel[(venue, "combined_with_bh")]
        r1d = per_venue_channel[(venue, "combined_no_bh")]
        if not np.array_equal(r2d["event_idx"], r1d["event_idx"]):
            # Align on the intersection defensively (physics-floor exclusion
            # differs, in principle, between channels); neither venue's CSV
            # has any excluded events in this run (n_events_excluded == 0
            # both channels, see BUILD_RECORD_B2.md), so this is a no-op.
            common = np.intersect1d(r2d["event_idx"], r1d["event_idx"])
            idx2d = np.searchsorted(r2d["event_idx"], common)
            idx1d = np.searchsorted(r1d["event_idx"], common)
            event_idx = common
            d_e_2d = r2d["d_e"][idx2d]
            d_e_1d = r1d["d_e"][idx1d]
            rank_2d = r2d["rank"][idx2d]
        else:
            event_idx = r2d["event_idx"]
            d_e_2d = r2d["d_e"]
            d_e_1d = r1d["d_e"]
            rank_2d = r2d["rank"]

        df_out = pd.DataFrame(
            {
                "event_idx": event_idx,
                "influence_2D": d_e_2d,
                "influence_1D": d_e_1d,
                "rank": rank_2d,
            }
        ).sort_values("event_idx")
        out_path = args.out_dir / f"influence_{venue}.csv"
        df_out.to_csv(out_path, index=False)
        csv_paths[venue] = out_path
        print(f"Wrote {out_path} ({len(df_out)} rows)")

    # --- BUILD_RECORD_B2.md -------------------------------------------------
    record_lines: list[str] = []
    record_lines.append("# BUILD_RECORD_B2.md -- Phase B (b-offset-subset-scorer), influence vector")
    record_lines.append("")
    record_lines.append(
        "Builder B2 (influence). Script: `build_influence_vector.py`. Frozen T0 convention"
        " (gradient-trapezoid weights, physics-floor zero handling) reused verbatim from"
        " `results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py` (`_moments`,"
        " `_physics_floor_apply`, `w = np.gradient(h_grid)`)."
    )
    record_lines.append("")
    record_lines.append(
        "**Blindness:** this builder never opened `covariate_table_blind*.csv` and computed no"
        " registered aggregate (AUC/OR/p-value/Delta_strat) over the registered population --"
        " per-event influence only."
    )
    record_lines.append("")
    record_lines.append("## Input pins (verified)")
    record_lines.append("")
    for venue, path in VENUE_CSV.items():
        record_lines.append(f"- `{venue}`: `{path.relative_to(REPO_ROOT)}` md5 `{VENUE_CSV_MD5[venue]}` -- MATCH")
    record_lines.append("")
    record_lines.append(
        "## Full-sample mean_h (10 s.f.), minimal-subset k, byte-id anchors"
        " (reported for the verifier -- NOT compared to the registered anchor values by this builder)"
    )
    record_lines.append("")
    record_lines.append("| venue | channel | mean_h_full (10 s.f.) | sigma_h_full | map_h_full |"
                         " minimal_k (recomputed) | banked_k (Sec.2, not re-derived) | n_excluded |"
                         " mean_h(all removed) |")
    record_lines.append("|---|---|---|---|---|---|---|---|---|")
    for (venue, channel), r in per_venue_channel.items():
        record_lines.append(
            f"| {venue} | {r['channel_label']} | {r['mean_h_full']:.10f} | {r['sigma_h_full']:.10f} |"
            f" {r['map_h_full']:.4f} | {r['minimal_k_recomputed']} | {r['banked_k']} |"
            f" {r['n_events_excluded_physics_floor']} | {r['mean_h_all_removed']:.10f} |"
        )
    record_lines.append("")
    record_lines.append("## Top-10 influence events per venue/channel (byte-id anchors)")
    record_lines.append("")
    record_lines.append(
        "Two lists per venue/channel, both derived from the same influence array (no free choice"
        " between them): **(A) literal top-10 by |influence|**; **(B) top-10 by decreasing"
        " directional influence d_e** -- cross-checked to be what"
        " `rd_2d_bootstrap_jackknife_output.json`'s `top10_events_by_abs_influence` field actually"
        " contains (its name notwithstanding: populated there from `order[:10]`, a directional-influence"
        " sort, not an abs-value sort). List (B) is the one the G-2(iii) anchor will match; list (A) is"
        " reported because the build mandate names it literally."
    )
    record_lines.append("")
    for (venue, channel), r in per_venue_channel.items():
        record_lines.append(f"### {venue} / {r['channel_label']}")
        record_lines.append("")
        record_lines.append("**(A) top-10 by |influence|**")
        record_lines.append("")
        record_lines.append("| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |")
        record_lines.append("|---|---|---|---|")
        for i, entry in enumerate(r["top10_by_abs_influence"], start=1):
            record_lines.append(
                f"| {i} | {entry['event_idx']} | {entry['influence']:.15e} | {entry['d_e']:.15e} |"
            )
        record_lines.append("")
        record_lines.append("**(B) top-10 by decreasing directional influence d_e**")
        record_lines.append("")
        record_lines.append("| rank | event_idx | influence (mean_h(full) - mean_h(full-e)) | d_e (directional) |")
        record_lines.append("|---|---|---|---|")
        for i, entry in enumerate(r["top10_by_directional_influence"], start=1):
            record_lines.append(
                f"| {i} | {entry['event_idx']} | {entry['influence']:.15e} | {entry['d_e']:.15e} |"
            )
        record_lines.append("")

    record_lines.append("## Output files")
    record_lines.append("")
    for venue, path in csv_paths.items():
        record_lines.append(
            f"- `{path.relative_to(REPO_ROOT)}`: columns `event_idx, influence_2D, influence_1D, rank`"
            f" -- `influence_2D`/`influence_1D` are the directional statistic d_e (Sec.2; positive ="
            " removing the event moves mean_h toward truth); `rank` is by decreasing `influence_2D`"
            " (the registered PRIMARY family)."
        )
    record_lines.append("")
    record_lines.append(
        "## Notes"
    )
    record_lines.append("")
    record_lines.append(
        "- `mean_h(all removed)` is the k=n_events endpoint of the drop-cumulative curve: a"
        " grid-symmetry check independent of the CSV data (flat weighted posterior over"
        " H_GRID_41), reported here as a cross-check, not a registered number."
    )
    record_lines.append(
        "- Per Sec.2, S (the high-influence subset) is defined by the BANKED k, not the"
        " `minimal_k_recomputed` column above; the recomputed value is offered purely as the"
        " G-2(ii) byte-id anchor for the verifier."
    )
    record_lines.append(
        "- This builder did not compute, and does not report, any of the registered separation"
        " or materiality statistics (AUC, OR, Holm p, Delta_strat) -- those are Phase C only,"
        " over the joined table."
    )

    record_path = args.out_dir / "BUILD_RECORD_B2.md"
    record_path.write_text("\n".join(record_lines) + "\n")
    print(f"Wrote {record_path}")


if __name__ == "__main__":
    main()
