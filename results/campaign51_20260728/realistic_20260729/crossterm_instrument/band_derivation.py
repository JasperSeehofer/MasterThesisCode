"""Deterministic derivation of the cross-term negligibility band (X, Y).

Reads ONLY the M-2 and M-4 supporting-read artifacts (plus M-3 for context
fields that are NOT band-bearing) and emits band_derivation.json. Written and
run 2026-08-05, BEFORE any cross-term number exists (crossterm_instrument.py
has never been executed with --confirm-run). This script is the arithmetic
appendix of PREREGISTRATION_CROSSTERM_INSTRUMENT.md.DRAFT — every threshold in
that file must reproduce from this script's output.

Derivation (see the prereg for the full rationale):
  X_c  = 2 * sigma_c * sqrt(385)      per channel c in {1d, 2d}
         sigma_c = min over venues of M-2's matched paired-diff std (nats/event)
         -> the 2-sigma matched-null band on a 385-event class-summed chord:
            "what zero looks like" on the overlap stratum.
  Y    = min over venues of (M-2 matched 2D mean paired diff) * 385
         -> the smallest class-summed effect this thread has ever treated as
            real (the matched 2D residual that revived H-2).
  W_vc = sum over sharing C-4 pairs of min-side shared w_pop share (M-4)
         -> the actually-shared population weight; x = X/W, y = Y/W are the
            per-unit-shared-weight thresholds that carry to the extended
            census in the [X, Y) gap protocol.

Run:  cd /home/jasper/Repositories/MasterThesisCode && uv run python \
      results/campaign51_20260728/realistic_20260729/crossterm_instrument/band_derivation.py
"""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_OVERLAP_EVENTS = 385  # C-4 overlap stratum (m2_results.json census block)
N_C4_PAIRS = 279
TWO_SIGMA = 2.0  # significance convention pre-stated in M-2 (alpha = 0.0455)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    m2_path = HERE / "m2_results.json"
    m3_path = HERE / "m3_results.json"
    m4_path = HERE / "m4_results.json"
    m2 = json.loads(m2_path.read_text())
    m3 = json.loads(m3_path.read_text())
    m4 = json.loads(m4_path.read_text())

    # --- M-2 inputs: matched paired-diff dispersion + matched 2D residual ---
    assert m2["census"]["sky_dl_pairs"] == N_C4_PAIRS
    assert m2["census"]["overlap_events_of_1590"] == N_OVERLAP_EVENTS
    stds: dict[str, dict[str, float]] = {}
    means: dict[str, dict[str, float]] = {}
    for venue in ("iiib", "joint_r1"):
        for ch in ("1d", "2d"):
            cd = m2["venues"][venue]["channels"][ch]
            assert cd["matched"] is not None
            stds.setdefault(ch, {})[venue] = cd["matched"]["paired_diff_std"]
            means.setdefault(ch, {})[venue] = cd["matched"]["mean_paired_diff"]

    root_n = math.sqrt(N_OVERLAP_EVENTS)
    X_exact = {ch: TWO_SIGMA * min(stds[ch].values()) * root_n for ch in ("1d", "2d")}
    # Locked values round DOWN (conservative: harder to NEGLECT).
    X_locked = {ch: math.floor(X_exact[ch] * 100) / 100 for ch in ("1d", "2d")}
    Y_exact = min(means["2d"].values()) * N_OVERLAP_EVENTS
    Y_locked = math.floor(Y_exact * 100) / 100  # round DOWN: earlier REGARD

    # --- M-4 inputs: actually-shared population weight per venue/channel ---
    W: dict[str, dict[str, float]] = {}
    n_sharing: dict[str, dict[str, int]] = {}
    W_ext_1d: dict[str, float] = {}
    for venue in ("joint_r1", "iiib"):
        v = m4[venue]
        for ch in ("1d", "2d"):
            tot, n = 0.0, 0
            for r in v["pair_records"]:
                if r.get(f"n_shared_{ch}", 0) > 0:
                    tot += min(r[f"w_share_{ch}_i"], r[f"w_share_{ch}_j"])
                    n += 1
            W.setdefault(ch, {})[venue] = tot
            n_sharing.setdefault(ch, {})[venue] = n
        # extended (global truly-sharing) 1D census weight, for the gap protocol
        out = sum(
            min(r["w_share_1d_i"], r["w_share_1d_j"])
            for r in v["global_pairs_shares"]["outside_c4_records"]
            if r.get("n_shared_1d", 0) > 0
        )
        W_ext_1d[venue] = W["1d"][venue] + out
    # cross-checks against M-4's own summary stats
    s = m4["joint_r1"]["shares"]["suppression_1d_min_w_share"]
    assert s["n"] == n_sharing["1d"]["joint_r1"]
    assert abs(s["mean"] * s["n"] - W["1d"]["joint_r1"]) < 1e-9

    per_unit = {
        ch: {
            venue: {
                "W_shared_wpop_minside_sum": W[ch][venue],
                "n_sharing_c4_pairs": n_sharing[ch][venue],
                "x_neglect_per_unit": X_locked[ch] / W[ch][venue],
                "y_regard_per_unit": Y_locked / W[ch][venue],
            }
            for venue in ("joint_r1", "iiib")
        }
        for ch in ("1d", "2d")
    }

    result = {
        "generated_by": "band_derivation.py (this directory)",
        "blindness": (
            "No cross-term Delta_ij value existed when this ran: "
            "crossterm_instrument.py has never been executed with --confirm-run "
            "and this directory contains no instrument output JSON."
        ),
        "inputs": {
            "m2_results.json": {
                "sha256": sha256(m2_path),
                "verification": "CONFIRMED (m2_adjudication.json)",
            },
            "m3_results.json": {
                "sha256": sha256(m3_path),
                "verification": "UNVERIFIED — context only, NOT band-bearing",
            },
            "m4_results.json": {
                "sha256": sha256(m4_path),
                "verification": "CONFIRMED (m4_adjudication_results.json, 26/26)",
            },
        },
        "m2_matched_paired_diff_std_nats_per_event": stds,
        "m2_matched_mean_paired_diff_nats_per_event": means,
        "constants": {
            "n_overlap_events": N_OVERLAP_EVENTS,
            "n_c4_pairs": N_C4_PAIRS,
            "sqrt_n": root_n,
            "two_sigma": TWO_SIGMA,
        },
        "X_neglect_class_summed_nats": {
            "exact": X_exact,
            "locked": X_locked,
            "rounding": "floor to 0.01 (conservative: harder to NEGLECT)",
        },
        "Y_regard_class_summed_nats": {
            "exact": Y_exact,
            "locked": Y_locked,
            "rounding": "floor to 0.01 (conservative: earlier REGARD)",
        },
        "per_unit_shared_weight": per_unit,
        "W_extended_1d_global_sharing": W_ext_1d,
        "m3_context_not_band_bearing": {
            "verdict": m3["call"]["verdict"],
            "max_per_event_chord_any_channel_nats": 0.13456089108217117,
            "per_event_bounded_null_criterion_nats": 1.0,
            "note": "M-3 supplies the chord-not-level discipline and the 1-nat "
            "per-pair anti-dilution floor (spec lines 566-567); its numbers do "
            "not enter X or Y, so its unverified status cannot move the band.",
        },
    }
    out_path = HERE / "band_derivation.json"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"wrote {out_path}")
    print(
        json.dumps(
            {"X_locked": X_locked, "Y_locked": Y_locked, "W": W, "W_ext_1d": W_ext_1d}, indent=1
        )
    )


if __name__ == "__main__":
    main()
