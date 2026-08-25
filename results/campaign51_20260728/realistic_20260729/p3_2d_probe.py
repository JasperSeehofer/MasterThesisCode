"""[P3-2D] stage-0 zero-compute probes on the 24 banked b0i CSVs (CLAIM_P3_2D_20260825.md §3).

Zero ``evaluate()`` calls. Reads only the banked ``p3_b0_work/{bt,bc}_9001xx_work``
``event_likelihoods.csv`` files and the per-seed ``selection_tables_h_0_73.json``.

Produces every [LOCAL] number quoted in the claim draft's §3:
  1. 1D pipeline validation: (C*/200)*sum_acc(1-w), w = A/(A+B),
     A = beta_G_phi*L_cat_no_bh, B = B_num  -- must reproduce the PA-CA-1 banked
     LHS set (B-T 0.04233+-0.00108, B-C 0.03741+-0.00095, paired +0.004919+-0.000146).
  2. The coded-arm 2D LHS core per seed: S2_s = (1/200)*sum_acc(1-w2),
     w2 = A2/(A2+B2), A2 = alpha_G_phi*L_cat_with_bh, B2 = B_num_wbh
     (PA-CA-1 drawn-count normalization; dead rows w2=0 contribute 1).
  3. bt-vs-bc bit-identity check on the 2D columns (paired by event_idx).
  4. PSIS-convention tail shape k-hat (scipy genpareto, floc=0, top-20% exceedances)
     of the bounded summand (1-w2) and of the unbounded phi==1 analog R2 = B2/A2.
  5. Extreme-R2 event forensics (the donor-mass-misalignment monsters).
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto

ROOT = Path(__file__).parent / "p3_b0_work"
SEEDS = [900101 + i for i in range(12)]
H = 0.73
N_DRAWN = 200
C_STAR = 0.1704718  # beta_G_phi*rho/beta_Gbar_phi, banked (CLAIM_B0 s1)
RHO = 0.9877707  # Sigma~^phi/Sigma^phi, banked


def khat_psis(x: np.ndarray, frac: float = 0.2) -> float:
    """PSIS-convention xi (positive = heavy tail) on the top-``frac`` exceedances."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_tail = max(int(math.ceil(frac * x.size)), 10)
    xs = np.sort(x)
    thr = xs[-n_tail - 1]
    exc = xs[-n_tail:] - thr
    exc = exc[exc > 0]
    if exc.size < 5:
        return float("nan")
    c, _loc, _scale = genpareto.fit(exc, floc=0.0)
    return float(c)


def main() -> None:
    lhs1 = {"bt": [], "bc": []}
    s2 = {"bt": [], "bc": []}
    khat_summand, khat_r2, n_acc_list = [], [], []
    dead1_tot = dead2_tot = 0
    w1_pool, w2_pool, r2_pool = [], [], []
    pair_worst = {"L_cat_with_bh": 0.0, "B_num_wbh": 0.0}
    n_paired = 0
    monsters = []

    for seed in SEEDS:
        sel = json.load(
            open(ROOT / f"bt_{seed}_work" / f"seed{seed}" / "selection_tables_h_0_73.json")
        )
        beta_G_phi, r_Malm = sel["beta_G_phi"], sel["r_Malm"]
        alpha = beta_G_phi * r_Malm
        assert abs(alpha - sel["sigma_4d"] * beta_G_phi / sel["sigma_phi"]) / alpha < 1e-12
        frames = {}
        for arm in ("bt", "bc"):
            f = (
                ROOT
                / f"{arm}_{seed}_work"
                / f"seed{seed}"
                / "simulations"
                / "diagnostics"
                / "event_likelihoods.csv"
            )
            d = pd.read_csv(f)
            d = d[np.isclose(d["h"], H)].sort_values("event_idx").reset_index(drop=True)
            frames[arm] = d
            a_csv = d["alpha_G_phi"].to_numpy()
            a_csv = a_csv[np.isfinite(a_csv)]
            assert np.allclose(a_csv, alpha, rtol=1e-6)
            A1 = beta_G_phi * d["L_cat_no_bh"].to_numpy()
            B1 = d["B_num"].to_numpy()
            w1 = np.where(A1 + B1 > 0, A1 / (A1 + B1), 0.0)
            lhs1[arm].append((C_STAR / N_DRAWN) * np.sum(1.0 - w1))
            A2 = alpha * d["L_cat_with_bh"].to_numpy()
            B2 = d["B_num_wbh"].to_numpy()
            w2 = np.where(A2 + B2 > 0, A2 / (A2 + B2), 0.0)
            s2[arm].append(np.sum(1.0 - w2) / N_DRAWN)
            if arm == "bt":
                n_acc_list.append(len(d))
                dead1_tot += int(np.sum(d["L_cat_no_bh"].to_numpy() == 0.0))
                dead2_tot += int(np.sum(A2 == 0.0))
                khat_summand.append(khat_psis(1.0 - w2))
                live = A2 > 0
                r2 = B2[live] / A2[live]
                khat_r2.append(khat_psis(r2))
                w1_pool.append(w1)
                w2_pool.append(w2)
                r2_pool.append(r2)
                i = int(np.argmax(np.where(live, B2 / np.where(live, A2, 1.0), 0.0)))
                r2i = B2[i] / A2[i] if A2[i] > 0 else np.inf
                if r2i > 1e10:
                    monsters.append(
                        (seed, int(d["event_idx"][i]), r2i, d["L_cat_with_bh"][i], d["L_cat_no_bh"][i])
                    )
        m = frames["bt"].merge(frames["bc"], on="event_idx", suffixes=("_t", "_c"))
        n_paired += len(m)
        for col in pair_worst:
            x, y = m[col + "_t"].to_numpy(), m[col + "_c"].to_numpy()
            den = np.maximum(np.abs(x), np.abs(y))
            rel = np.where(den > 0, np.abs(x - y) / den, 0.0)
            pair_worst[col] = max(pair_worst[col], float(np.max(rel)))

    def fs(v: list[float]) -> str:
        a = np.asarray(v)
        return f"{a.mean():.6f} +- {a.std(ddof=1) / math.sqrt(a.size):.6f}"

    print("n_acc per seed:", n_acc_list, "total", sum(n_acc_list))
    print("1D LHS  B-T:", fs(lhs1["bt"]), " B-C:", fs(lhs1["bc"]))
    print("1D paired D:", fs(list(np.array(lhs1["bt"]) - np.array(lhs1["bc"]))))
    print("2D S2 per seed (bt):", [f"{x:.6f}" for x in s2["bt"]])
    print("2D S2 fleet:", fs(s2["bt"]), " paired bt-bc D:", fs(list(np.array(s2["bt"]) - np.array(s2["bc"]))))
    r_malm = json.load(open(ROOT / "bt_900101_work/seed900101/selection_tables_h_0_73.json"))["r_Malm"]
    c2_est, c2_hi = C_STAR * r_malm, C_STAR * r_malm / RHO
    print(f"C2*(rho2=rho) = {c2_est:.7f}   C2*(rho2=1) = {c2_hi:.7f}")
    print("LHS2 (rho2=rho):", fs(list(c2_est * np.array(s2["bt"]))))
    print("LHS2 (rho2=1) :", fs(list(c2_hi * np.array(s2["bt"]))))
    print("pair identity max rel dev:", pair_worst, "on", n_paired, "paired rows")
    print("dead rows: 1D", dead1_tot, " 2D", dead2_tot, f"({100 * dead2_tot / sum(n_acc_list):.1f}% of accepted)")
    print("khat(1-w2) per seed:", [f"{k:+.2f}" for k in khat_summand])
    print("khat(R2)   per seed:", [f"{k:+.2f}" for k in khat_r2])
    w1a, w2a = np.concatenate(w1_pool), np.concatenate(w2_pool)
    r2a = np.concatenate(r2_pool)
    print(f"pooled khat(1-w2) {khat_psis(1 - w2a):+.3f}  khat(w2) {khat_psis(w2a):+.3f}  khat(R2) {khat_psis(r2a):+.3f}")
    print(f"w2 median {np.median(w2a):.4f}  w1 median {np.median(w1a):.4f}  corr(w1,w2) {np.corrcoef(w1a, w2a)[0, 1]:.4f}")
    print(f"R2 live: median {np.median(r2a):.3f}  p99 {np.percentile(r2a, 99):.4g}  max {r2a.max():.3e}  n {r2a.size}")
    print("monsters (seed, event_idx, R2, L_cat_with_bh, L_cat_no_bh):")
    for row in monsters:
        print(f"  {row[0]}  ev{row[1]}  R2={row[2]:.3e}  Lwbh={row[3]:.3e}  Lnobh={row[4]:.3e}")


if __name__ == "__main__":
    main()
