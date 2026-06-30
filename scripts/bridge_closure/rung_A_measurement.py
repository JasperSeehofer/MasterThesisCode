"""Rung A — the measurement side: P0 event characterization + the sigma^2/N ladder.

Establishes (with paper figures) that the MEASUREMENT side of the pipeline does
NOT bias H0:
  * P0: the seed-600 detections' measured-vs-true d_L scatter is unbiased; real
    sigma_dL/d_L ~ 3.7%; 99% of detections are in-catalogue.
  * The synthetic closure recovers H0 unbiased at production N=3361 across the
    full sigma range -> the railing is NOT a sigma^2/distance-scatter effect.

Run: uv run python scripts/bridge_closure/rung_A_measurement.py
Outputs: scripts/bridge_closure/outputs/{P0_event_characterization.pdf,
         rungA_sigma_ladder.pdf, rungA_results.json}
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import _bridge_lib as B  # noqa: E402
from _plot_style import OK_COLOR, RAIL_COLOR, TRUTH_COLOR, plt  # noqa: E402


def probe_P0() -> dict:
    """Characterise the real seed-600 detections (Malmquist / measurement check)."""
    d = B.load_real_detections(apply_cuts=True)
    d_meas, d_true, sigma_dL = d["d_meas"], d["d_true"], d["sigma_dL"]
    frac_resid = (d_meas - d_true) / d_true
    rel_err = sigma_dL / d_meas
    in_cat = d["in_catalog"]
    z_true = B.dist_to_redshift_vec(d_true, B.TRUE_H)

    stats = {
        "n_events": int(len(d_meas)),
        "frac_resid_mean": float(frac_resid.mean()),
        "frac_resid_median": float(np.median(frac_resid)),
        "frac_resid_std": float(frac_resid.std()),
        "rel_err_mean": float(rel_err.mean()),
        "rel_err_median": float(np.median(rel_err)),
        "in_catalog_frac": float(in_cat.mean()),
        "z_true_median": float(np.median(z_true)),
        "d_meas_median_Gpc": float(np.median(d_meas)),
    }

    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    # (a) measured vs true d_L
    ax[0, 0].scatter(d_true, d_meas, s=4, alpha=0.25, color=OK_COLOR)
    lim = [0, max(d_true.max(), d_meas.max()) * 1.02]
    ax[0, 0].plot(lim, lim, "--", color=TRUTH_COLOR, lw=1)
    ax[0, 0].set(xlim=lim, ylim=lim, xlabel=r"true $d_L$ [Gpc]", ylabel=r"measured $d_L$ [Gpc]",
                 title="(a) measured vs true distance")
    # (b) fractional residual
    ax[0, 1].hist(frac_resid, bins=60, color=OK_COLOR, alpha=0.85)
    ax[0, 1].axvline(0, color=TRUTH_COLOR, ls="--", lw=1)
    ax[0, 1].axvline(frac_resid.mean(), color=RAIL_COLOR, lw=1.4,
                     label=f"mean {frac_resid.mean():+.4f}")
    ax[0, 1].set(xlabel=r"$(d_L^{\rm meas}-d_L^{\rm true})/d_L^{\rm true}$", ylabel="count",
                 title="(b) measurement scatter is unbiased")
    ax[0, 1].legend()
    # (c) relative distance error distribution
    ax[1, 0].hist(rel_err, bins=60, color=OK_COLOR, alpha=0.85)
    ax[1, 0].axvline(np.median(rel_err), color=RAIL_COLOR, lw=1.4,
                     label=f"median {np.median(rel_err):.3f}")
    ax[1, 0].set(xlabel=r"$\sigma_{d_L}/d_L$", ylabel="count",
                 title="(c) real distance-error distribution")
    ax[1, 0].legend()
    # (d) redshift distribution + in-catalogue fraction
    ax[1, 1].hist(z_true[in_cat], bins=40, color=OK_COLOR, alpha=0.8,
                  label=f"in-catalogue ({in_cat.mean()*100:.1f}%)")
    ax[1, 1].hist(z_true[~in_cat], bins=40, color=RAIL_COLOR, alpha=0.8,
                  label=f"dark ({(~in_cat).mean()*100:.1f}%)")
    ax[1, 1].set(xlabel=r"true redshift $z$", ylabel="count",
                 title="(d) detections are nearby & in-catalogue")
    ax[1, 1].legend()
    fig.suptitle("P0 — seed-600 detection characterisation (selection on true SNR -> no Malmquist)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = B.OUTPUTS / "P0_event_characterization.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return stats


def rung_A_sigma_ladder() -> dict:
    """Synthetic catalogue at production N across the sigma range -> no railing."""
    sigma_configs = [
        ("sigma=0.005", dict(sigma_frac=0.005)),
        ("sigma=0.02", dict(sigma_frac=0.02)),
        ("sigma=0.05", dict(sigma_frac=0.05)),
        ("sigma=real(3.7%)", dict(sigma_model="real_dist")),
    ]
    results = []
    for label, kw in sigma_configs:
        t = time.time()
        cfg = B.BridgeConfig(name=label, completeness="declining", n_gal=20000,
                             n_events=3361, seed=1, **kw)
        r = B.run_bridge(cfg, verbose=False)
        r["elapsed_s"] = round(time.time() - t, 1)
        results.append(r)
        print(f"  {label:18s} MAP={r['h_refined']:.4f} bias={r['bias']:+.4f} "
              f"railed={r['railed']} ({r['elapsed_s']}s)", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for r in results:
        hs = np.array(r["hs"])
        post = np.exp(np.array(r["logpost"]))
        ax[0].plot(hs, post / post.max(), label=f"{r['name']} (MAP {r['h_refined']:.3f})")
    ax[0].axvline(B.TRUE_H, color=TRUTH_COLOR, ls="--", lw=1.2, label="truth 0.73")
    ax[0].set(xlabel=r"$h = H_0/100$", ylabel="normalised posterior",
              title="(a) synthetic catalogue, N=3361 — posteriors peak at truth")
    ax[0].legend()
    biases = [r["bias"] for r in results]
    labels = [r["name"] for r in results]
    ax[1].axhline(0, color=TRUTH_COLOR, ls="--", lw=1)
    ax[1].bar(range(len(biases)), biases, color=OK_COLOR)
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, rotation=30, ha="right")
    ax[1].set(ylabel=r"MAP bias $\hat h - 0.73$", ylim=(-0.05, 0.16),
              title="(b) measurement side does NOT rail")
    ax[1].axhline(0.13, color=RAIL_COLOR, ls=":", lw=1.2)
    ax[1].text(0.05, 0.135, "real-pipeline railing (+0.13)", color=RAIL_COLOR, fontsize=8)
    fig.tight_layout()
    out = B.OUTPUTS / "rungA_sigma_ladder.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}", flush=True)
    return {"sigma_ladder": [{k: r[k] for k in ("name", "h_refined", "bias", "railed",
                                                "n_events", "n_in_catalog")} for r in results],
            "curves": [{"name": r["name"], "hs": r["hs"], "logpost": r["logpost"]} for r in results]}


def main() -> None:
    print("P0 — real detection characterisation ...", flush=True)
    p0 = probe_P0()
    print(f"  {p0}", flush=True)
    print("Rung A — synthetic sigma ladder at N=3361 ...", flush=True)
    rungA = rung_A_sigma_ladder()
    out = B.OUTPUTS / "rungA_results.json"
    out.write_text(json.dumps({"P0": p0, "rungA": rungA}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
