"""C7 step 3 — does the measured kernel shift ACCOUNT for the observed rail?

Confronts the kernel measurement (c7_kernel_measure.py) with the delivered
seed61000/real_r1 production numbers.

Test 1 (in-cat, the C5 rail)
    Observed: per-event catalogue-leg profile from
      diagnostics/event_likelihoods.csv, column L_cat_no_bh.
    In `absolute_marginal`, L_cat_i(h) = (SUM_ball w_g N_g,i(h)) / Sigma_glob(h)
    with Sigma_glob EVENT-INDEPENDENT (= `sum_w_Dg(no_bh)` in the per-h log,
    :2335).  So the event-side ball numerator is
        S_i(h) = L_cat_i(h) * Sigma_glob(h)
    and Delta ln S_i(0.73 -> 0.86) is directly comparable to the single-host
    Delta ln N_g(0.73 -> 0.86) that the kernel driver predicts.
    Predicted at several sigma_z/z, with realization scatter on z_obs.

Test 2 (dark class direction)
    For every event, invert the delivered catalogue-leg argmax into an
    "effective host redshift":  h_peak/h_true = f(z_eff)/f(z_hat), f(z) = h*d_L.
    The volume_deconv kernel can only move z_eff UP (measured factor
    K = [1 + sqrt(1 + 12 eps^2)]/2 > 1), so a dark class whose peak sits BELOW
    h_true requires impostor hosts at z_g < z_hat/K -- i.e. foreground
    contamination, not the kernel.

Read-only.  Run from the repo root with .venv/bin/python.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from c7_kernel_measure import deconv_profile, load_incat_events  # noqa: E402

from master_thesis_code.constants import H as H_TRUE
from master_thesis_code.physical_relations import dist_to_redshift, dist_vectorized

HERE = Path(__file__).parent
RUN = HERE.parent / "seed61000" / "real_r1"
LOG = HERE.parent / "seed61000" / "mixture_leg_log_extract.txt"
CRB = HERE.parent / "seed61000" / "prepared_cramer_rao_bounds.csv"


def sigma_glob_table() -> pd.Series:
    """Sigma_glob(h) = sum_w_Dg(no_bh) from the per-h log extract (5 s.f.)."""
    pat = re.compile(r"w_G=beta_G/D\(h\)=([\d.]+), sum_w_Dg\(no_bh\)=([\dEe.+-]+)")
    hpat = re.compile(r"_h_0_(\d+)\.log")
    out = {}
    for line in LOG.read_text().splitlines():
        m = pat.search(line)
        if m:
            h = float("0." + hpat.search(line).group(1))
            out[h] = float(m.group(2))
    return pd.Series(out).sort_index()


def main() -> None:
    sg = sigma_glob_table()
    print(
        f"Sigma_glob(h): {len(sg)} h-points, "
        f"Delta ln (0.73->0.86) = {np.log(sg[0.86] / sg[0.73]):+.5f}"
    )

    d = pd.read_csv(RUN / "diagnostics" / "event_likelihoods.csv")
    crb = pd.read_csv(CRB)
    incat = set(crb.index[crb.host_galaxy_index >= 0])
    hs = np.sort(d.h.unique())
    Lcat = d.pivot(index="event_idx", columns="h", values="L_cat_no_bh")[hs]
    ev_ids = Lcat.index.to_numpy()
    mask_in = np.isin(ev_ids, sorted(incat))
    SG = sg.reindex(hs).to_numpy()
    assert np.all(np.isfinite(SG))
    S = Lcat.to_numpy() * SG[None, :]  # ball numerator sum, per event

    i73 = int(np.where(hs == 0.73)[0][0])
    i86 = len(hs) - 1

    # ---------- Test 1: observed vs predicted Delta ln N over 0.73 -> 0.86 ----
    ok = mask_in & (S[:, i73] > 0) & (S[:, i86] > 0)
    obs = np.log(S[ok, i86] / S[ok, i73])
    print(f"\n=== Test 1: in-cat ball-numerator tilt, h = 0.73 -> 0.86 ({ok.sum()} events) ===")
    print(
        "OBSERVED  Delta ln (SUM_ball w_g N_g): "
        + "  ".join(f"p{p}={np.percentile(obs, p):+.3f}" for p in (5, 25, 50, 75, 95))
        + f"   frac>0: {np.mean(obs > 0):.3f}"
    )

    ev = load_incat_events()
    n = len(ev)
    d_L = ev.d_L.to_numpy()
    s_dL = ev.sigma_dL.to_numpy()
    s_fr = ev.sigma_frac_cond.to_numpy()
    z_t = ev.z_true.to_numpy()
    h2 = np.array([0.73, 0.86])
    rng = np.random.default_rng(7)
    rows = []
    # NB: no eps -> 0 row here. A delta-like prior inside the WIDE event-level
    # GW window is unresolvable by the code's GL-50 numerator quadrature and
    # returns aliasing garbage (a spurious +460 nats).  The point-kernel limit is
    # obtained analytically instead - see c7_checks.py item 4 (-408 nats median).
    for eps in (0.10, 0.25, 0.35, 0.49, 0.65, 0.80):
        pr = []
        for _ in range(20):
            zo = np.clip(z_t + eps * z_t * rng.standard_normal(n), 1e-4, None)
            prof = deconv_profile(h2, d_L, s_dL, s_fr, zo, np.full(n, 1.0) * eps * z_t)
            good = (prof[:, 0] > 0) & (prof[:, 1] > 0)
            pr.append(np.log(prof[good, 1] / prof[good, 0]))
        pr = np.concatenate(pr)
        rows.append(
            dict(
                eps=eps,
                p5=np.percentile(pr, 5),
                p25=np.percentile(pr, 25),
                median=np.median(pr),
                p75=np.percentile(pr, 75),
                p95=np.percentile(pr, 95),
                frac_positive=float(np.mean(pr > 0)),
            )
        )
    pred = pd.DataFrame(rows)
    print("\nPREDICTED Delta ln N_g(0.73->0.86), single true host, scattered z_obs:")
    print(pred.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # ---------- Test 2: effective host redshift implied by each event's peak --
    # f(z) = h * d_L(z; h) is h-independent; h_peak/h_true = f(z_eff)/f(z_hat).
    zgrid = np.geomspace(1e-4, 3.0, 4000)
    fgrid = np.asarray(dist_vectorized(zgrid, h=1.0), float)

    d_all = crb["luminosity_distance"].to_numpy()
    z_hat_all = np.array([dist_to_redshift(x, h=H_TRUE) for x in d_all])
    z_hat = z_hat_all[ev_ids]

    nz = S.max(axis=1) > 0
    jpk = np.argmax(np.where(np.isfinite(S), S, -np.inf), axis=1)
    h_pk = hs[jpk]
    f_eff = np.interp(z_hat, zgrid, fgrid) * (h_pk / H_TRUE)
    z_eff = np.interp(f_eff, fgrid, zgrid)
    ratio = z_eff / z_hat

    print("\n=== Test 2: effective host z implied by the delivered catalogue-leg peak ===")
    for lbl, m in (("IN-CAT", mask_in & nz), ("DARK", ~mask_in & nz)):
        print(
            f"{lbl:7s} n={m.sum():5d}  z_hat med={np.median(z_hat[m]):.4f}   "
            f"h_peak med={np.median(h_pk[m]):.3f}   z_eff/z_hat: "
            + "  ".join(f"p{p}={np.percentile(ratio[m], p):.3f}" for p in (5, 25, 50, 75, 95))
            + f"   frac at 0.60 edge={np.mean(h_pk[m] == hs[0]):.3f}"
            + f"   frac at 0.86 edge={np.mean(h_pk[m] == hs[-1]):.3f}"
        )

    # dark-class kernel factor K at the dark class's own sigma_z/z
    from c7_kernel_measure import indicative_sigma_z_over_z

    eps_dark = indicative_sigma_z_over_z(np.clip(z_hat[~mask_in & nz], 1e-4, None))
    K = (1 + np.sqrt(1 + 12 * eps_dark**2)) / 2
    print(
        f"\ndark-class indicative sigma_z/z at z_hat: med={np.median(eps_dark):.3f} "
        f"-> kernel up-factor K med={np.median(K):.3f}  (kernel can only push z_eff UP)"
    )
    print(
        "implied BARE impostor z_g/z_hat = (z_eff/z_hat)/K: "
        + "  ".join(f"p{p}={np.percentile(ratio[~mask_in & nz] / K, p):.3f}" for p in (25, 50, 75))
    )

    with open(HERE / "c7_vs_production_results.json", "w") as f:
        json.dump(
            dict(
                sigma_glob_dln_073_086=float(np.log(sg[0.86] / sg[0.73])),
                observed_incat_dln=obs.tolist(),
                predicted=rows,
                h_peak=h_pk.tolist(),
                z_eff_over_z_hat=ratio.tolist(),
                is_incat=mask_in.tolist(),
                event_ids=ev_ids.tolist(),
            ),
            f,
            indent=1,
        )
    print(f"\nwrote {HERE / 'c7_vs_production_results.json'}")


if __name__ == "__main__":
    main()
