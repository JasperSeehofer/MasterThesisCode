"""Item 3/20 verifier pass — B2.1 [CMEM] A1, R2c NOT-DISTINGUISHED.

Bit-for-bit independent re-execution of the sha1-pinned instrument
`../cmem_a1.py` (sha1 75751f3c71375cec0c4f67d5957a1b5158e1c2b6, verified below),
importing its unmodified functions and re-running the full pipeline from source
CSV/JSON — not reading cmem_a1_result.json or the RECORD.md's restated numbers.

Writes its own output JSON under this verifier_pass/ directory; does NOT touch
cmem_a1_work/ (append-only discipline for the original record's artifacts).
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/jasper/Repositories/darksiren-emri")
NODE_DIR = REPO_ROOT / "results/campaign51_20260728/realistic_20260729/fanout1_20260829"
INSTRUMENT = NODE_DIR / "cmem_a1.py"
OUT = Path(__file__).resolve().parent / "item3_rederive_output.json"

EXPECTED_SHA1 = "75751f3c71375cec0c4f67d5957a1b5158e1c2b6"

sys.path.insert(0, str(REPO_ROOT))


def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main() -> None:
    actual_sha1 = sha1_of_file(INSTRUMENT)
    sha1_match = actual_sha1 == EXPECTED_SHA1
    print(f"Instrument sha1 check: expected={EXPECTED_SHA1} actual={actual_sha1} match={sha1_match}")

    # Import the pinned instrument as a module, unmodified, from its own path
    # (does not execute its __main__ block).
    spec = importlib.util.spec_from_file_location("cmem_a1_pinned", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print("Building census over p3_b0_work {bc,bt} arms (independent execution)...")
    df, usable_seeds, missing_crb_seeds = mod.build_census()
    print(f"usable_seeds={usable_seeds}")
    print(f"missing_crb_seeds={missing_crb_seeds}")

    gates = mod.run_gates(df, usable_seeds, missing_crb_seeds)
    print("Gates overall passed:", gates["passed"])
    for k in ("c_g1a_catalogue_pin", "c_g1b_anchor", "c_g1c_bc_bt_cross_consistency", "c_g2_positivity"):
        print(f"  {k}: passed={gates[k]['passed']}")
    print("Census:", json.dumps(gates["census"], indent=2))

    if not gates["passed"]:
        result = {"verdict": "INSTRUMENT-DEFECT", "gates": gates}
        with open(OUT, "w") as f:
            json.dump(result, f, indent=1, default=str)
        raise SystemExit("GATE STOP")

    print("Running registered R2c-only statistic, independent re-execution...")
    result = mod.run_statistic(df)
    result["gates"] = gates
    result["verifier_sha1_check"] = {
        "expected": EXPECTED_SHA1,
        "actual": actual_sha1,
        "match": sha1_match,
    }

    # Independent per-arm breakdown, mirroring the record's disclosed (non-registered)
    # sensitivity check, using only the pinned instrument's own stratum_diff/run_statistic
    # building blocks (re-implemented pooling loop here, independently of the builder's
    # /tmp helper script which no longer exists on disk).
    per_arm = {}
    for arm in ("bc", "bt"):
        sub = df[df["arm"] == arm].reset_index(drop=True)
        strata = sorted(sub["stratum"].unique())
        diffs = [mod.stratum_diff(sub, s) for s in strata]
        import numpy as np

        obs = float(np.mean(diffs))
        rng = np.random.default_rng(mod.PERM_SEED)
        stratum_frames = {s: sub[sub["stratum"] == s].reset_index(drop=True) for s in strata}
        count = 0
        for _ in range(mod.N_PERM):
            d = []
            for s, g in stratum_frames.items():
                lab = g["outside"].to_numpy().copy()
                lab = rng.permutation(lab)
                ln_c = np.log(g["combined_no_bh"].to_numpy())
                d.append(float(np.mean(ln_c[lab]) - np.mean(ln_c[~lab])) if lab.any() and (~lab).any() else 0.0)
            stat = float(np.mean(d))
            if abs(stat) >= abs(obs):
                count += 1
        per_arm[arm] = {"n_strata": len(strata), "statistic": obs, "perm_p": count / mod.N_PERM}

    result["per_arm_independent_check"] = per_arm

    # Covariate check (z_true medians outside vs inside), independent re-derivation.
    import numpy as np

    cov = {}
    for label, sub in (
        ("bc", df[df["arm"] == "bc"]),
        ("bt", df[df["arm"] == "bt"]),
        ("pooled", df),
    ):
        out = sub[sub["outside"]]
        inn = sub[~sub["outside"]]
        cov[label] = {
            "median_z_outside": float(np.median(out["z_true"])),
            "median_z_inside": float(np.median(inn["z_true"])),
            "n_out": int(len(out)),
            "n_in": int(len(inn)),
        }
    result["covariate_independent_check"] = cov

    with open(OUT, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != "gates"}, indent=1, default=str))


if __name__ == "__main__":
    main()
