"""Generate the canonical R-3 target-set files (zero-compute M-4 census read).

Emits ``target_pairs_joint_r1.json`` and ``target_pairs_iiib.json`` next to
this script: the M-4 truly-sharing pair censuses (prereg §3 target set —
1D: 349 joint_r1 / 280 iiib; 2D: 104 / 21) with per-pair ``n_shared`` and the
in/outside-C-4 flag, in the ``--pair-list`` schema consumed by
``crossterm_instrument.py`` (R-3 ingestion).

ZERO-COMPUTE in the prereg §7.4 sense: the only inputs are the frozeng
``posteriors_with_bh_mass/h_0_73.json`` ball emits (the exact candidate lists
production consumed; M-4 recipe 1, ``load_ball_sets``), the CRB CSV for the
C-4 predicate flags (``c4_pair_census``, the pinned recon recipe), and
``m4_results.json`` for the exact-match verification. No quadrature, no
likelihood, no Delta — nothing from the instrument's compute path runs.

Verification (hard asserts, the script fails loudly on any mismatch):
  * m4_results.json sha256 equals the prereg §7.1 pin (46907913...).
  * Per venue/channel: pair sets and per-pair n_shared match m4_results.json
    EXACTLY — in-C-4 pairs against ``pair_records`` (n_shared_{1d,2d} > 0),
    outside pairs against ``global_pairs_shares.outside_c4_records``, and the
    2D outside set against ``global_sharing_check.outside_pairs_2d``.
  * Count table == prereg §3: joint_r1 349 = 80 + 269 (1D), 104 = 27 + 77
    (2D); iiib 280 = 63 + 217, 21 = 5 + 16.

Run: cd /home/jasper/Repositories/MasterThesisCode && uv run python \
    results/campaign51_20260728/realistic_20260729/crossterm_instrument/make_target_pairs.py
"""

import hashlib
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossterm_instrument import (  # noqa: E402
    CRB_PATH,
    VENUE_CONFIGS,
    c4_pair_census,
    load_ball_sets,
)

HERE = Path(__file__).resolve().parent
M4_PATH = HERE / "m4_results.json"
M4_SHA256_PIN = "46907913b35369bfdb96a705d1782a5c8d4eb9bf7955d1cb851434ba9d3f1c6b"

#: Prereg §3 target-set counts: {venue: {channel: (total, in_c4, outside)}}.
EXPECTED = {
    "joint_r1": {"1d": (349, 80, 269), "2d": (104, 27, 77)},
    "iiib": {"1d": (280, 63, 217), "2d": (21, 5, 16)},
}


def sharing_pairs(balls: dict[int, set[int]]) -> dict[tuple[int, int], int]:
    """All event pairs sharing >= 1 ball galaxy, with exact shared counts.

    Inverted galaxy->events index (M-4 global-census recipe) to find the
    candidate pairs, then exact ball-set intersections for n_shared.
    """
    inv: dict[int, list[int]] = defaultdict(list)
    for ev, ball in balls.items():
        for g in ball:
            inv[g].append(ev)
    cand: set[tuple[int, int]] = set()
    for evs in inv.values():
        if len(evs) > 1:
            for a, b in combinations(sorted(set(evs)), 2):
                cand.add((a, b))
    return {(i, j): len(balls[i] & balls[j]) for (i, j) in sorted(cand) if balls[i] & balls[j]}


def main() -> None:
    m4_sha = hashlib.sha256(M4_PATH.read_bytes()).hexdigest()
    assert m4_sha == M4_SHA256_PIN, f"m4_results.json sha256 {m4_sha} != prereg pin {M4_SHA256_PIN}"
    m4 = json.loads(M4_PATH.read_text())

    crb_all = pd.read_csv(CRB_PATH)
    pairs_all, _degree = c4_pair_census(crb_all)
    c4_set = set(pairs_all)
    assert len(c4_set) == 279, f"C-4 census reproduction failed: {len(c4_set)} != 279"

    for venue in ("joint_r1", "iiib"):
        cfg = VENUE_CONFIGS[venue]
        ball_json = cfg["frozeng_dir"] / "posteriors_with_bh_mass" / "h_0_73.json"
        ball_1d, ball_2d = load_ball_sets(cfg["frozeng_dir"], h_file="h_0_73.json")
        by_channel = {"1d": sharing_pairs(ball_1d), "2d": sharing_pairs(ball_2d)}

        # --- exact-match verification against m4_results.json ---------------
        in_c4_expect: dict[str, dict[tuple[int, int], int]] = {"1d": {}, "2d": {}}
        for rec in m4[venue]["pair_records"]:
            key = (rec["i"], rec["j"])
            for ch in ("1d", "2d"):
                if rec[f"n_shared_{ch}"] > 0:
                    in_c4_expect[ch][key] = rec[f"n_shared_{ch}"]
        out_expect: dict[str, dict[tuple[int, int], int]] = {"1d": {}, "2d": {}}
        for rec in m4[venue]["global_pairs_shares"]["outside_c4_records"]:
            key = (rec["i"], rec["j"])
            for ch in ("1d", "2d"):
                if rec[f"n_shared_{ch}"] > 0:
                    out_expect[ch][key] = rec[f"n_shared_{ch}"]
        gp2 = {tuple(p) for p in m4[venue]["global_sharing_check"]["outside_pairs_2d"]}
        assert gp2 == set(out_expect["2d"]), (
            f"{venue}: 2D outside sets disagree between m4 records "
            f"({len(out_expect['2d'])}) and global check ({len(gp2)})"
        )

        pairs_out: dict[str, list[dict[str, object]]] = {}
        counts: dict[str, dict[str, int]] = {}
        for ch in ("1d", "2d"):
            got = by_channel[ch]
            got_in = {p: n for p, n in got.items() if p in c4_set}
            got_out = {p: n for p, n in got.items() if p not in c4_set}
            assert got_in == in_c4_expect[ch], (
                f"{venue}/{ch}: in-C-4 pair/n_shared mismatch vs pair_records "
                f"({len(got_in)} vs {len(in_c4_expect[ch])})"
            )
            assert got_out == out_expect[ch], (
                f"{venue}/{ch}: outside-C-4 pair/n_shared mismatch vs "
                f"outside_c4_records ({len(got_out)} vs {len(out_expect[ch])})"
            )
            total_e, in_e, out_e = EXPECTED[venue][ch]
            assert (len(got), len(got_in), len(got_out)) == (total_e, in_e, out_e), (
                f"{venue}/{ch}: counts {len(got)}={len(got_in)}+{len(got_out)} "
                f"!= prereg {total_e}={in_e}+{out_e}"
            )
            pairs_out[ch] = [
                {"i": i, "j": j, "n_shared": n, "in_c4": (i, j) in c4_set}
                for (i, j), n in sorted(got.items())
            ]
            counts[ch] = {
                "total": len(got),
                "in_c4": len(got_in),
                "outside_c4": len(got_out),
            }
            print(
                f"{venue}/{ch}: {len(got)} truly-sharing pairs "
                f"({len(got_in)} in-C-4 + {len(got_out)} outside) — matches "
                f"m4_results.json and prereg §3 exactly"
            )

        out = {
            "meta": {
                "generator": str(Path(__file__).resolve()),
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "venue": venue,
                "schema": "crossterm_instrument --pair-list (R-3 ingestion)",
                "convention": (
                    "M-4 truly-sharing census (prereg §3 target set): every "
                    "event pair whose production candidate balls share >= 1 "
                    "catalogue galaxy; ball_1d = galaxy_likelihoods UNION "
                    "additional_galaxies_without_bh_mass, ball_2d = "
                    "galaxy_likelihoods only, from the frozeng h_0_73.json "
                    "emits (ball h-independence verified in M-4/V2); "
                    "n_shared = exact ball-intersection size; in_c4 = pair in "
                    "the C-4 279-pair census (recon recipe)"
                ),
                "zero_compute": (
                    "No quadrature, likelihood, or Delta was computed; inputs "
                    "are the frozeng ball emits, the CRB CSV (C-4 predicate "
                    "flags only) and m4_results.json (verification only)"
                ),
                "inputs": {
                    "ball_json": str(ball_json),
                    "crb_csv": str(CRB_PATH),
                    "m4_results_sha256": m4_sha,
                },
                "counts": counts,
            },
            "pairs": pairs_out,
        }
        out_path = HERE / f"target_pairs_{venue}.json"
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
