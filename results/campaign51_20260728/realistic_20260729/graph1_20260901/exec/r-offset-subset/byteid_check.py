#!/usr/bin/env python3
"""
Independent byte-id verifier for BUILD_RECORD_B2.md (b-offset-subset-scorer, influence
vector builder), written from scratch against REGISTRATION_DRAFT.md's stated anchors and
the reference JSON `exec/rd-2d-bootstrap-jackknife/rd_2d_bootstrap_jackknife_output.json`.

Does NOT import build_influence_vector.py (B2's own script) or any other builder code.
Parses BUILD_RECORD_B2.md and the influence_*.csv files directly and cross-checks against
the reference JSON and the literal anchor values quoted in REGISTRATION_DRAFT.md.

Anchors checked (draft lines ~163-168, "G-2 byte-id anchors"):
  (i)   iiib 2D full_sample.mean_h == 0.6658540600  (|delta| <= 1e-9)
        iiib 1D full_sample.mean_h == 0.6669869414  (|delta| <= 1e-9)
  (ii)  minimal_k_events_removed == 82 (iiib 2D) / 94 (iiib 1D) / 72 (joint_r1 2D) / 46 (joint_r1 1D)
  (iii) JSON top10_events_by_abs_influence (event_idx + influence value) reproduced to
        1e-12 relative, for all four (venue, channel) families -- checked against both
        BUILD_RECORD_B2.md's reported list (B) table AND the raw influence_*.csv columns.
        NOTE (sign convention): the JSON field stores the raw signed infl_e = mean_h(full) -
        mean_h(full-e); BUILD_RECORD_B2.md's list (B) and the influence_*.csv columns store
        the *directional* d_e = sign(0.73 - mean_h(full)) * (-infl_e) per REGISTRATION_DRAFT.md
        line 68. This script applies that documented transform before comparing -- a literal
        infl_e vs d_e comparison would show a spurious full sign flip on every event.
  (iv)  k=1588 endpoint of the drop-cumulative curve == 0.73 (|delta| <= 1e-12)
  (v)   0 events physics-floor-excluded, for all four families
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRAPH_ROOT = HERE.parent.parent
JSON_PATH = GRAPH_ROOT / "exec" / "rd-2d-bootstrap-jackknife" / "rd_2d_bootstrap_jackknife_output.json"
BUILD_RECORD = HERE / "BUILD_RECORD_B2.md"
INFLUENCE_CSVS = {
    "iiib": HERE / "influence_iiib.csv",
    "joint_r1": HERE / "influence_joint_r1.csv",
}

# JSON "channel" naming -> BUILD_RECORD_B2.md / draft "channel" naming
CHANNEL_MAP = {
    "combined_with_bh": "2D",
    "combined_no_bh": "1D",
}

# Registered anchors, quoted literally from REGISTRATION_DRAFT.md lines 163-168.
REGISTERED_MEAN_H = {
    ("iiib", "2D"): 0.6658540600,
    ("iiib", "1D"): 0.6669869414,
}
MEAN_H_TOL = 1e-9

REGISTERED_MINIMAL_K = {
    ("iiib", "2D"): 82,
    ("iiib", "1D"): 94,
    ("joint_r1", "2D"): 72,
    ("joint_r1", "1D"): 46,
}

TOP10_REL_TOL = 1e-12
ENDPOINT_H = 0.73
ENDPOINT_TOL = 1e-12

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def load_json() -> dict:
    with open(JSON_PATH) as f:
        return json.load(f)


def index_results(d: dict) -> dict[tuple[str, str], dict]:
    out = {}
    for r in d["results"]:
        venue = r["venue"]
        chan = CHANNEL_MAP.get(r["channel"], r["channel"])
        out[(venue, chan)] = r
    return out


def parse_build_record_table(text: str) -> dict[tuple[str, str], dict]:
    """Parse the 'Full-sample mean_h ... minimal-subset k' table in BUILD_RECORD_B2.md."""
    out = {}
    pat = re.compile(
        r"^\|\s*(iiib|joint_r1)\s*\|\s*(1D|2D)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|"
        r"\s*([0-9.eE+-]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.eE+-]+)\s*\|",
        re.MULTILINE,
    )
    for m in pat.finditer(text):
        venue, chan, mean_h_full, sigma_h_full, map_h_full, minimal_k, banked_k, n_excl, mean_h_removed = m.groups()
        out[(venue, chan)] = {
            "mean_h_full": float(mean_h_full),
            "sigma_h_full": float(sigma_h_full),
            "map_h_full": float(map_h_full),
            "minimal_k_recomputed": int(minimal_k),
            "banked_k": int(banked_k),
            "n_excluded": int(n_excl),
            "mean_h_all_removed": float(mean_h_removed),
        }
    return out


def parse_build_record_top10_lists(text: str) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """Parse the '(B) top-10 by decreasing directional influence d_e' tables per venue/channel."""
    out = {}
    section_pat = re.compile(r"^### (iiib|joint_r1) / (1D|2D)\s*$", re.MULTILINE)
    sections = list(section_pat.finditer(text))
    for i, sec in enumerate(sections):
        venue, chan = sec.groups()
        start = sec.end()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        block = text[start:end]
        b_idx = block.find("**(B) top-10 by decreasing directional influence")
        if b_idx == -1:
            continue
        b_block = block[b_idx:]
        row_pat = re.compile(
            r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*\|", re.MULTILINE
        )
        rows = []
        for rm in row_pat.finditer(b_block):
            rank, event_idx, influence, d_e = rm.groups()
            rows.append((int(event_idx), float(d_e)))
            if len(rows) == 10:
                break
        out[(venue, chan)] = rows
    return out


def load_influence_csv(venue: str) -> dict[int, dict[str, float]]:
    path = INFLUENCE_CSVS[venue]
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[int(row["event_idx"])] = {
                "influence_2D": float(row["influence_2D"]),
                "influence_1D": float(row["influence_1D"]),
                "rank": int(row["rank"]),
            }
    return rows


def top10_from_csv(rows: dict[int, dict[str, float]], chan: str) -> list[tuple[int, float]]:
    key = f"influence_{chan}"
    ordered = sorted(rows.items(), key=lambda kv: kv[1][key], reverse=True)
    return [(idx, v[key]) for idx, v in ordered[:10]]


def main() -> int:
    d = load_json()
    jres = index_results(d)

    text = BUILD_RECORD.read_text()
    br_table = parse_build_record_table(text)
    br_top10 = parse_build_record_top10_lists(text)

    csv_rows = {venue: load_influence_csv(venue) for venue in INFLUENCE_CSVS}

    families = [("iiib", "2D"), ("iiib", "1D"), ("joint_r1", "2D"), ("joint_r1", "1D")]

    # (i) mean_h anchors (only two are literally registered in the draft; check both,
    #     and cross-check BUILD_RECORD's reported mean_h_full against the JSON for all four
    #     as an internal consistency check).
    for fam in families:
        j = jres[fam]
        json_mean_h = j["full_sample"]["mean_h"]
        br = br_table.get(fam)
        if br is None:
            record(f"mean_h[{fam}] BUILD_RECORD row present", False, "row not found/parsed")
            continue
        delta_br_json = abs(br["mean_h_full"] - json_mean_h)
        record(
            f"mean_h[{fam}] BUILD_RECORD vs JSON full_sample.mean_h",
            delta_br_json <= 1e-9,
            f"BR={br['mean_h_full']!r} JSON={json_mean_h!r} |delta|={delta_br_json:.3e} (tol 1e-9)",
        )
        if fam in REGISTERED_MEAN_H:
            reg = REGISTERED_MEAN_H[fam]
            delta_reg = abs(json_mean_h - reg)
            record(
                f"mean_h[{fam}] JSON vs REGISTERED anchor",
                delta_reg <= MEAN_H_TOL,
                f"JSON={json_mean_h!r} registered={reg!r} |delta|={delta_reg:.3e} (tol {MEAN_H_TOL:.0e})",
            )

    # (ii) minimal k
    for fam in families:
        j = jres[fam]
        json_k = j["minimal_subset"]["minimal_k_events_removed"]
        reg_k = REGISTERED_MINIMAL_K[fam]
        br = br_table.get(fam, {})
        br_k_recomputed = br.get("minimal_k_recomputed")
        br_k_banked = br.get("banked_k")
        ok = (json_k == reg_k) and (br_k_recomputed == reg_k) and (br_k_banked == reg_k)
        record(
            f"minimal_k[{fam}]",
            ok,
            f"JSON={json_k} registered={reg_k} BUILD_RECORD.minimal_k_recomputed={br_k_recomputed} "
            f"BUILD_RECORD.banked_k={br_k_banked}",
        )

    # (iii) top10 by abs_influence (JSON) vs BUILD_RECORD list (B) vs raw influence_*.csv
    for fam in families:
        venue, chan = fam
        j = jres[fam]
        # Draft line 67-68: infl_e = mean_h(full) - mean_h(full-e); directional influence
        # d_e = sign(0.73 - mean_h(full)) * (-infl_e). The JSON field's "influence" values are
        # the raw signed infl_e (registration draft line 57 / BUILD_RECORD_B2.md notes), while
        # BUILD_RECORD_B2's list (B) and the influence_*.csv columns report d_e. Apply the
        # documented sign transform -- do NOT compare raw infl_e to d_e literally.
        mean_h_full = j["full_sample"]["mean_h"]
        sign = 1.0 if (0.73 - mean_h_full) >= 0 else -1.0
        json_top10 = [
            (e["event_idx"], sign * (-e["influence"])) for e in j["jackknife"]["top10_events_by_abs_influence"]
        ]

        br_list = br_top10.get(fam)
        if br_list is None:
            record(f"top10[{fam}] BUILD_RECORD list (B) present", False, "not found/parsed")
        else:
            ok = True
            detail_parts = []
            for (je, jv), (be, bv) in zip(json_top10, br_list):
                if je != be:
                    ok = False
                    detail_parts.append(f"event_idx mismatch JSON={je} BR={be}")
                    continue
                rel = abs(jv - bv) / max(abs(jv), 1e-300)
                if rel > TOP10_REL_TOL:
                    ok = False
                    detail_parts.append(f"event {je}: JSON={jv!r} BR={bv!r} rel={rel:.3e}")
            record(
                f"top10[{fam}] JSON vs BUILD_RECORD (B) list",
                ok,
                "all 10 match to 1e-12 relative" if ok else "; ".join(detail_parts),
            )

        csv_top10 = top10_from_csv(csv_rows[venue], chan)
        ok = True
        detail_parts = []
        for (je, jv), (ce, cv) in zip(json_top10, csv_top10):
            if je != ce:
                ok = False
                detail_parts.append(f"event_idx mismatch JSON={je} CSV={ce}")
                continue
            rel = abs(jv - cv) / max(abs(jv), 1e-300)
            if rel > TOP10_REL_TOL:
                ok = False
                detail_parts.append(f"event {je}: JSON={jv!r} CSV={cv!r} rel={rel:.3e}")
        record(
            f"top10[{fam}] JSON vs influence_{venue}.csv (independently re-sorted)",
            ok,
            "all 10 match to 1e-12 relative" if ok else "; ".join(detail_parts),
        )

    # (iv) k=1588 endpoint == 0.73
    for fam in families:
        j = jres[fam]
        curve = j["minimal_subset"]["curve_sample"]
        endpoint = None
        for pt in curve:
            if pt["k"] == 1588:
                endpoint = pt["mean_h"]
        if endpoint is None:
            record(f"endpoint[{fam}] k=1588 present in curve_sample", False, "no k=1588 point found")
            continue
        delta = abs(endpoint - ENDPOINT_H)
        record(
            f"endpoint[{fam}] k=1588 mean_h == 0.73",
            delta <= ENDPOINT_TOL,
            f"value={endpoint!r} |delta|={delta:.3e} (tol {ENDPOINT_TOL:.0e})",
        )
        # cross-check against BUILD_RECORD's "mean_h(all removed)" column
        br = br_table.get(fam, {})
        br_removed = br.get("mean_h_all_removed")
        if br_removed is not None:
            delta_br = abs(br_removed - ENDPOINT_H)
            record(
                f"endpoint[{fam}] BUILD_RECORD mean_h(all removed) == 0.73",
                delta_br <= 1e-9,
                f"value={br_removed!r} |delta|={delta_br:.3e}",
            )

    # (v) zero physics-floor exclusions
    for fam in families:
        j = jres[fam]
        n_excl_json = j["n_excluded_physics_floor"]
        br = br_table.get(fam, {})
        n_excl_br = br.get("n_excluded")
        ok = (n_excl_json == 0) and (n_excl_br == 0)
        record(
            f"physics_floor_exclusions[{fam}]",
            ok,
            f"JSON n_excluded_physics_floor={n_excl_json} BUILD_RECORD n_excluded={n_excl_br}",
        )

    print()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    n_total = len(results)
    print(f"TOTAL: {n_total - n_fail}/{n_total} checks passed")
    if n_fail:
        print(f"VERDICT: RED ({n_fail} check(s) failed)")
        return 1
    print("VERDICT: GREEN (all anchors match within stated tolerance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
