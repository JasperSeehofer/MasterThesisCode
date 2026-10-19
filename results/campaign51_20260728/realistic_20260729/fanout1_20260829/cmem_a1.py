"""[CMEM] Node B2.1 A1 — higher-power R2c re-read (bc AND bt arms, paired within-stratum
ln-ratio of combined_no_bh at h = 0.73).

Registration: PREREGISTRATION_CMEM_A1_20260829.md (this directory). Extends
`../cmem_reads.py`'s flag recomputation (chord/radius cone geometry, K = 1.5,
`get_possible_hosts_from_ball_tree` replicated line-for-line) to a DIFFERENT, SMALLER
fleet than the original CMEM read used: `p3_b0_work` (venue `b0i`, the b0-identity fleet,
nominal seeds 900101-900112, BOTH `bc` and `bt` arms), not `p3_2d_fleet_20260825`
(venue `b0i2d`, 24 `bc` arms) that the original `cmem_reads.py` pointed at.

VERIFIER-INDEPENDENCE CONTRACT (standing rule 2): this file is authored by the builder
agent, who ran it ONLY with `--dry-run` (gates + census, no statistic). The registered
R2c-only statistic (within-stratum paired mean of ln(combined_no_bh), 10 000 within-
stratum label permutations) may be executed only by a DIFFERENT agent (the runner), via
`--dry-run` omitted. A runner may fix a disclosed crash but must not change the gates,
the statistic, the pairing definition, or the permutation count without a fresh
registration note.

Gates (see prereg §5):
  C-G1a — catalogue pin: reduced_galaxy_catalogue.csv md5 == REDUCED_CATALOGUE_MD5.
  C-G1b — NEW anchor (this fleet has no seed 900121; the original anchor does not exist
          here): bc/900101/event_idx 0, chord and radius reproduce to registered
          tolerance.
  C-G1c — bc/bt cross-arm consistency: per-seed n_out and n_in are IDENTICAL between the
          bc and bt arms for every usable seed (both arms share event/host realizations;
          only the completion-leg likelihood differs) — a defect here means the flag
          recomputation is not deterministic in the host/event geometry alone.
  C-G1d — seed-span disclosure: seeds 900111/900112 lack `prepared_cramer_rao_bounds.csv`
          for both arms (confirmed against `PREREGISTRATION_P3_WBHZERO_MEASURE_20260825.md`
          line ~160) and are EXCLUDED from the census/statistic; usable span is
          900101-900110 (10 seeds x 2 arms = 20 strata). This is a disclosed deviation
          from the charter's nominal "seeds 900101-900112", not a gate failure.
  C-G2  — positivity sanity: combined_no_bh > 0 for 100% of joined rows (required for the
          ln transform; the original C-G2's c_share in [0,1] check does not apply here
          because A1 is scoped to R2c only, not R2a).

Run:
    uv run python cmem_a1.py --dry-run     # builder smoke-test: gates + census only
    uv run python cmem_a1.py               # runner only: full statistic + permutation
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from darksiren_emri.galaxy_catalogue.handler import GalaxyCatalogueHandler, _polar_to_cartesian

FLEET = Path(__file__).resolve().parents[1] / "p3_b0_work"
OUT = Path(__file__).resolve().parent / "cmem_a1_work"
K = 1.5
ARMS = ("bc", "bt")
SEEDS_NOMINAL = list(range(900101, 900113))

CATALOGUE_PATH = (
    Path(__file__).resolve().parents[4] / "darksiren_emri" / "galaxy_catalogue" / "reduced_galaxy_catalogue.csv"
)
REDUCED_CATALOGUE_MD5 = "c52c13b5cab61f6b3f04bbe202550969"  # darksiren_emri/validation/correspondence_1d.py:311

# New anchor for THIS fleet (p3_b0_work has no seed 900121; the original CMEM anchor,
# bc_900121_work event 20, does not exist here). Full-float, computed once by the
# builder's --dry-run and frozen at registration.
ANCHOR = ("bc", 900101, 0, 0.0116656941007181, 0.0359121946154451)

CRB_COLS = [
    "qS",
    "phiS",
    "delta_qS_delta_qS",
    "delta_phiS_delta_phiS",
    "delta_phiS_delta_qS",
    "host_galaxy_index",
    "in_catalog",
    "z_true",
]

N_PERM = 10_000
PERM_SEED = 20260829
ALPHA = 0.01


def cone_radius(theta: float, phi_var: float, theta_var: float, cov: float) -> float:
    sigma = np.array([[phi_var, cov], [cov, theta_var]])
    jac = np.diag([abs(np.sin(theta)), 1.0])
    lam = float(np.linalg.eigvalsh(jac @ sigma @ jac.T).max())
    return float(K * np.sqrt(max(lam, 0.0)))


def md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_census() -> tuple[pd.DataFrame, list[int], list[int]]:
    handler = GalaxyCatalogueHandler(1e4, 1e7, 1.5)
    cat = handler.reduced_galaxy_catalog.reset_index(drop=True)
    host_xyz = _polar_to_cartesian(
        cat["THETA_S"].to_numpy(dtype=np.float64), cat["PHI_S"].to_numpy(dtype=np.float64)
    )

    rows = []
    usable_seeds: list[int] = []
    missing_crb_seeds: list[int] = []
    for arm in ARMS:
        for seed in SEEDS_NOMINAL:
            seed_dir = FLEET / f"{arm}_{seed}_work" / f"seed{seed}" / "simulations"
            crb_path = seed_dir / "prepared_cramer_rao_bounds.csv"
            diag_path = seed_dir / "diagnostics" / "event_likelihoods.csv"
            if not crb_path.exists():
                if arm == "bc":
                    missing_crb_seeds.append(seed)
                continue
            if arm == "bc":
                usable_seeds.append(seed)
            crb = pd.read_csv(crb_path, usecols=CRB_COLS)
            diag = pd.read_csv(diag_path)
            diag73 = diag[diag["h"] == 0.73].set_index("event_idx")
            for i, r in crb.iterrows():
                if not bool(r["in_catalog"]):
                    continue
                hidx = int(r["host_galaxy_index"])
                ev_xyz = _polar_to_cartesian(np.array([r["qS"]]), np.array([r["phiS"]]))[0]
                chord = float(np.linalg.norm(host_xyz[hidx] - ev_xyz))
                radius = cone_radius(
                    float(r["qS"]),
                    float(r["delta_phiS_delta_phiS"]),
                    float(r["delta_qS_delta_qS"]),
                    float(r["delta_phiS_delta_qS"]),
                )
                if i not in diag73.index:
                    # Banked census basis = the posterior-joined subset, per cmem_reads.py.
                    continue
                d = diag73.loc[i]
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "stratum": f"{arm}_{seed}",
                        "event_idx": i,
                        "chord": chord,
                        "radius": radius,
                        "outside": chord > radius,
                        "z_true": float(r["z_true"]),
                        "combined_no_bh": float(d["combined_no_bh"]),
                        "L_cat_no_bh": float(d["L_cat_no_bh"]),
                    }
                )
    return pd.DataFrame(rows), usable_seeds, missing_crb_seeds


def run_gates(df: pd.DataFrame, usable_seeds: list[int], missing_crb_seeds: list[int]) -> dict:
    gates: dict = {}

    # C-G1a: catalogue pin.
    if CATALOGUE_PATH.exists():
        md5 = md5_of_file(CATALOGUE_PATH)
        gates["c_g1a_catalogue_pin"] = {
            "path": str(CATALOGUE_PATH),
            "md5": md5,
            "expected": REDUCED_CATALOGUE_MD5,
            "passed": md5 == REDUCED_CATALOGUE_MD5,
        }
    else:
        gates["c_g1a_catalogue_pin"] = {"passed": False, "error": "catalogue file not found"}

    # C-G1b: new anchor.
    a_arm, a_seed, a_idx, a_chord, a_radius = ANCHOR
    a = df[(df["arm"] == a_arm) & (df["seed"] == a_seed) & (df["event_idx"] == a_idx)]
    if len(a) == 1:
        chord_ok = abs(float(a["chord"].iloc[0]) - a_chord) < 5e-10
        radius_ok = abs(float(a["radius"].iloc[0]) - a_radius) < 1e-15
        gates["c_g1b_anchor"] = {
            "anchor": list(ANCHOR),
            "found_chord": float(a["chord"].iloc[0]),
            "found_radius": float(a["radius"].iloc[0]),
            "chord_ok": chord_ok,
            "radius_ok": radius_ok,
            "passed": chord_ok and radius_ok,
        }
    else:
        gates["c_g1b_anchor"] = {"passed": False, "error": f"anchor row not found ({len(a)})"}

    # C-G1c: bc/bt cross-arm consistency (per-seed n_out, n_in identical).
    cross_ok = True
    cross_rows = []
    for seed in usable_seeds:
        bc_g = df[(df["arm"] == "bc") & (df["seed"] == seed)]
        bt_g = df[(df["arm"] == "bt") & (df["seed"] == seed)]
        bc_out, bc_in = int(bc_g["outside"].sum()), int((~bc_g["outside"]).sum())
        bt_out, bt_in = int(bt_g["outside"].sum()), int((~bt_g["outside"]).sum())
        ok = (bc_out, bc_in) == (bt_out, bt_in)
        cross_ok = cross_ok and ok
        cross_rows.append({"seed": seed, "bc_out": bc_out, "bc_in": bc_in, "bt_out": bt_out, "bt_in": bt_in, "match": ok})
    gates["c_g1c_bc_bt_cross_consistency"] = {"rows": cross_rows, "passed": cross_ok}

    # C-G1d: seed-span disclosure (not a pass/fail gate; a recorded fact).
    gates["c_g1d_seed_span"] = {
        "nominal_seeds": SEEDS_NOMINAL,
        "usable_seeds": usable_seeds,
        "missing_crb_seeds": missing_crb_seeds,
        "note": "seeds with missing prepared_cramer_rao_bounds.csv are excluded from the "
        "census and the statistic; confirmed independently against "
        "PREREGISTRATION_P3_WBHZERO_MEASURE_20260825.md line ~160.",
    }

    # Census (both arms; this is the NEW number being registered, not a reproduction of
    # the original CMEM read's 380/2261/0.1681, which was computed on a DIFFERENT fleet
    # p3_2d_fleet_20260825 venue b0i2d).
    census = {}
    for arm in ARMS:
        g = df[df["arm"] == arm]
        n_out, n_tot = int(g["outside"].sum()), len(g)
        census[arm] = {"n_outside": n_out, "n_total": n_tot, "fraction": n_out / n_tot if n_tot else float("nan")}
    n_out_combined = int(df["outside"].sum())
    n_tot_combined = len(df)
    census["combined"] = {
        "n_outside": n_out_combined,
        "n_total": n_tot_combined,
        "fraction": n_out_combined / n_tot_combined if n_tot_combined else float("nan"),
    }
    gates["census"] = census
    gates["census_disclosure"] = (
        "This instrument's bc census does NOT reproduce the original CMEM read's "
        "380/2261 (0.1681) — that number was computed on p3_2d_fleet_20260825 (venue "
        "b0i2d, 24 bc arms), a DIFFERENT fleet from p3_b0_work (venue b0i, 10 usable "
        "seeds) used here. This is expected and is not a gate failure; see prereg §1."
    )

    # C-G2: positivity sanity (needed for ln transform).
    n_nonpos = int((df["combined_no_bh"] <= 0).sum())
    gates["c_g2_positivity"] = {
        "n_nonpositive_combined_no_bh": n_nonpos,
        "n_total": len(df),
        "passed": n_nonpos == 0,
    }

    gates["passed"] = bool(
        gates["c_g1a_catalogue_pin"]["passed"]
        and gates["c_g1b_anchor"]["passed"]
        and gates["c_g1c_bc_bt_cross_consistency"]["passed"]
        and gates["c_g2_positivity"]["passed"]
    )
    return gates


def stratum_diff(df: pd.DataFrame, stratum: str, labels: np.ndarray | None = None) -> float:
    """Per-stratum mean(ln combined_no_bh | outside) - mean(ln combined_no_bh | inside)."""
    g = df[df["stratum"] == stratum]
    lab = g["outside"].to_numpy() if labels is None else labels
    ln_c = np.log(g["combined_no_bh"].to_numpy())
    return float(np.mean(ln_c[lab]) - np.mean(ln_c[~lab]))


def run_statistic(df: pd.DataFrame) -> dict:
    """THE REGISTERED R2c-only statistic. Runner-only per verifier independence."""
    strata = sorted(df["stratum"].unique())
    n_strata = len(strata)

    def grand_stat_equal(frame: pd.DataFrame) -> float:
        diffs = [stratum_diff(frame, s) for s in strata]
        return float(np.mean(diffs))

    def grand_stat_eventweighted(frame: pd.DataFrame) -> float:
        num, den = 0.0, 0.0
        for s in strata:
            g = frame[frame["stratum"] == s]
            n_out = int(g["outside"].sum())
            num += n_out * stratum_diff(frame, s)
            den += n_out
        return float(num / den)

    obs_equal = grand_stat_equal(df)
    obs_eventw = grand_stat_eventweighted(df)

    rng = np.random.default_rng(PERM_SEED)
    count_equal = 0
    count_eventw = 0
    # Precompute per-stratum arrays for speed.
    stratum_frames = {s: df[df["stratum"] == s].reset_index(drop=True) for s in strata}
    for _ in range(N_PERM):
        perm_labels = {}
        for s, g in stratum_frames.items():
            lab = g["outside"].to_numpy().copy()
            perm_labels[s] = rng.permutation(lab)
        diffs = {}
        for s, g in stratum_frames.items():
            lab = perm_labels[s]
            ln_c = np.log(g["combined_no_bh"].to_numpy())
            diffs[s] = float(np.mean(ln_c[lab]) - np.mean(ln_c[~lab])) if lab.any() and (~lab).any() else 0.0
        stat_equal = float(np.mean(list(diffs.values())))
        num, den = 0.0, 0.0
        for s, g in stratum_frames.items():
            n_out = int(perm_labels[s].sum())
            num += n_out * diffs[s]
            den += n_out
        stat_eventw = float(num / den) if den else 0.0
        if abs(stat_equal) >= abs(obs_equal):
            count_equal += 1
        if abs(stat_eventw) >= abs(obs_eventw):
            count_eventw += 1

    p_equal = count_equal / N_PERM
    p_eventw = count_eventw / N_PERM
    return {
        "n_strata": n_strata,
        "primary_equal_weight": {
            "statistic": obs_equal,
            "perm_p": p_equal,
            "displaced": bool(p_equal < ALPHA),
            "direction_deficit_outside": bool(obs_equal < 0),
        },
        "secondary_event_weighted": {
            "statistic": obs_eventw,
            "perm_p": p_eventw,
            "displaced": bool(p_eventw < ALPHA),
            "direction_deficit_outside": bool(obs_eventw < 0),
        },
        "n_perm": N_PERM,
        "perm_seed": PERM_SEED,
        "alpha": ALPHA,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Gates + census only. Does NOT compute the registered statistic (verifier independence).",
    )
    args = ap.parse_args()

    print("Building census over p3_b0_work {bc,bt} arms...")
    df, usable_seeds, missing_crb_seeds = build_census()
    gates = run_gates(df, usable_seeds, missing_crb_seeds)

    OUT.mkdir(exist_ok=True)
    with open(OUT / "cmem_a1_gates.json", "w") as f:
        json.dump(gates, f, indent=1, default=str)
    print("GATES:", json.dumps(gates, indent=1, default=str))

    if not gates["passed"]:
        with open(OUT / "cmem_a1_result.json", "w") as f:
            json.dump({"verdict": "INSTRUMENT-DEFECT", "gates": gates}, f, indent=1, default=str)
        raise SystemExit("GATE STOP: one or more C-G1/C-G2 gates failed")

    if args.dry_run:
        print("--dry-run: gates + census only. Statistic NOT computed (verifier independence).")
        return

    print("Running registered R2c-only statistic (runner-only)...")
    result = run_statistic(df)
    result["gates"] = gates
    with open(OUT / "cmem_a1_result.json", "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
