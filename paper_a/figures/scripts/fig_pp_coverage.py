"""Paper A figure: P-P / coverage plot, bare-Gaussian vs volume-weighted host-z kernel.

Produces ``paper_a/figures/fig_pp_coverage.pdf`` (label ``fig:pp``).

The curves are computed with the committed, independent calibration harness
``master_thesis_code.validation.pp_coverage`` (promoted from the 2026-07-01
verification commission, investigator d2; provenance and reference outputs in
``results/commission_20260701/scratch/d2/``).  The harness simulates synthetic
universes in the clean single-host limit (fully complete catalogue, exactly one
candidate host per event, Malmquist detection selection handled by the D(h)
denominator) and differs between the two estimator variants ONLY in the
host-redshift kernel of the in-catalogue numerator:

* ``bare``   -- N(z; z_gal, sigma_z)                       (production-style)
* ``volume`` -- N(z; z_gal, sigma_z) * dV_c/dz / (1+z)     (prior-consistent)

For every realization the posterior p(h) is evaluated on the harness h-grid and
converted to a P-P value: the HPD credible level at which the injected truth
sits on the region boundary (= posterior mass with density >= density(h_true)).
Under perfect calibration these values are Uniform(0, 1), so their empirical
CDF tracks the diagonal.  Both kernels analyse IDENTICAL synthetic data (the
per-realization RNG is re-seeded with the same child seed), making the
comparison exactly paired.

Run from the repo root:

    .venv/bin/python paper_a/figures/scripts/fig_pp_coverage.py

Outputs:
    paper_a/figures/fig_pp_coverage.pdf        (the figure, vector PDF)
    paper_a/figures/data/fig_pp_coverage_data.json  (P-P values + coverage summary)
"""

import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from master_thesis_code.plotting._style import apply_style  # noqa: E402

apply_style()

from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from master_thesis_code.plotting._colors import EDGE, METHOD, PRIOR  # noqa: E402
from master_thesis_code.plotting._helpers import get_figure, save_figure  # noqa: E402
from master_thesis_code.plotting.validation_plots import _binomial_pp_band  # noqa: E402
from master_thesis_code.validation import pp_coverage as ppc  # noqa: E402

# ---------------------------------------------------------------------------
# Run configuration.
#
# Matches the commission d2 reference run (make_pp_plot.py: 250 realizations x
# 250 events, sigma_z = 0.035, sigma_dl/dl = 5%) and the harness defaults; the
# three injected truths are the commission clean-test values
# (NOTE_calibration_findings.md, Result 1).  Master seed = the harness default.
# ---------------------------------------------------------------------------
SEED = 20260701
N_REALIZATIONS = 250
N_EVENTS = 250
SIGMA_Z = 0.035
SIGMA_DL_FRAC = 0.05
TRUTHS: tuple[float, ...] = (0.66, 0.72, 0.78)
TRUTH_BOLD = 0.72  # emphasised truth (the commission's primary injected value)
KERNELS: tuple[str, ...] = ("bare", "volume")
# Finer posterior grid than the harness default (0.004): the per-realization
# posterior is only ~0.008 wide, so the default grid quantizes the P-P values
# into visible staircases (also present in the commission reference plot).
# Pure numerical resolution of the same estimator -- no physics change.
H_STEP = 0.001

FIG_DIR = REPO_ROOT / "paper_a" / "figures"
PDF_STEM = FIG_DIR / "fig_pp_coverage"  # save_figure appends ".pdf"
DATA_PATH = FIG_DIR / "data" / "fig_pp_coverage_data.json"

# Kernel -> (color, linestyle, legend label).  Okabe-Ito colors from the
# project palette; solid vs dashed keeps the two curves separable in grayscale.
KERNEL_STYLE: dict[str, tuple[str, str | tuple[int, tuple[int, ...]], str]] = {
    "volume": (METHOD["dark"], "-", "volume-weighted kernel"),
    "bare": ("#D55E00", (0, (5, 2)), "bare Gaussian kernel"),
}


def _pp_value(
    h_grid: npt.NDArray[np.float64],
    post: npt.NDArray[np.float64],
    h_true: float,
) -> float:
    """HPD credible level at which ``h_true`` sits on the region boundary.

    Equals the posterior mass carried by grid points with density >= the
    density at the truth.  Calibrated posteriors give Uniform(0, 1) values.
    """
    dh = np.gradient(h_grid)
    p_true = float(np.interp(h_true, h_grid, post))
    return float(np.sum((post >= p_true) * post * dh))


def compute_pp_values() -> dict[str, dict[str, list[float]]]:
    """Run the harness for both kernels on paired data; return P-P values.

    Returns:
        Mapping ``kernel -> {f"{h_true:.2f}": [pp values]}``.
    """
    configs = {
        kernel: ppc.PPCoverageConfig(
            n_realizations=N_REALIZATIONS,
            n_events=N_EVENTS,
            sigma_z=SIGMA_Z,
            sigma_dl_frac=SIGMA_DL_FRAC,
            injected_truths=list(TRUTHS),
            seed=SEED,
            kernel=kernel,  # type: ignore[arg-type]
            h_step=H_STEP,
        )
        for kernel in KERNELS
    }
    h_grid = configs["volume"].h_grid()

    # Shared selection denominator D(h) = int p_det(A(z)/h) w_pop(z) dz,
    # exactly as in ppc.run_coverage.
    zint = np.linspace(ppc.Z_MIN, ppc.Z_MAX_POP, 3000)
    wpop = ppc.population_weight_of_z(zint)
    Dh = np.trapezoid(
        ppc.detection_probability(ppc.comoving_amplitude_of_z(zint)[:, None] / h_grid[None, :])
        * wpop[:, None],
        zint,
        axis=0,
    )
    log_Dh = np.log(Dh)

    pp: dict[str, dict[str, list[float]]] = {k: {f"{t:.2f}": [] for t in TRUTHS} for k in KERNELS}
    for h_true in TRUTHS:
        # Independent seed stream per truth, deterministic in SEED.
        master = np.random.default_rng(SEED + int(round(h_true * 1000)))
        for _ in range(N_REALIZATIONS):
            child_seed = int(master.integers(1 << 62))
            for kernel in KERNELS:
                # Same child seed -> identical synthetic data for both kernels
                # (the kernel only changes the likelihood evaluation).
                rng = np.random.default_rng(child_seed)
                logL = ppc._run_realization(h_true, h_grid, log_Dh, configs[kernel], rng)
                post = np.exp(logL - logL.max())
                post /= np.trapezoid(post, h_grid)
                pp[kernel][f"{h_true:.2f}"].append(_pp_value(h_grid, post, h_true))
    return pp


def coverage_summary(
    pp: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Empirical coverage at the 50/68/90% HPD levels from the P-P values.

    The truth lies inside the level-X HPD region exactly when its P-P value
    is <= X, so coverage(X) = mean(pp <= X).
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for kernel, per_truth in pp.items():
        out[kernel] = {}
        for truth, values in per_truth.items():
            arr = np.asarray(values)
            out[kernel][truth] = {
                lvl: float(np.mean(arr <= float(lvl) / 100.0)) for lvl in ("50", "68", "90")
            }
    return out


def make_figure(pp: dict[str, dict[str, list[float]]]) -> None:
    """Draw and save the P-P plot."""
    fig, ax = get_figure(figsize=(3.375, 3.55))

    # Grey 1/2/3-sigma binomial bands for an ideal P-P curve of N realizations
    # (same construction as plotting.validation_plots.plot_pp_coverage).
    grid = np.linspace(0.0, 1.0, 512)
    for n_sigma, alpha in ((3.0, 0.10), (2.0, 0.14), (1.0, 0.20)):
        lower, upper = _binomial_pp_band(N_REALIZATIONS, grid, n_sigma)
        ax.fill_between(grid, lower, upper, color=PRIOR, alpha=alpha, lw=0, zorder=0)

    # Diagonal = perfect calibration.
    ax.plot([0.0, 1.0], [0.0, 1.0], color=EDGE, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)

    for kernel in KERNELS:
        color, linestyle, _ = KERNEL_STYLE[kernel]
        for h_true in TRUTHS:
            values = np.sort(np.asarray(pp[kernel][f"{h_true:.2f}"]))
            n = values.size
            # ECDF as a step curve, pinned to (0, 0) and (1, 1).
            x = np.concatenate([[0.0], values, [1.0]])
            y = np.concatenate([[0.0], np.arange(1, n + 1) / n, [1.0]])
            bold = h_true == TRUTH_BOLD
            ax.step(
                x,
                y,
                where="post",
                color=color,
                linestyle=linestyle,
                linewidth=1.6 if bold else 0.8,
                alpha=1.0 if bold else 0.55,
                zorder=4 if bold else 3,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Nominal HPD credible level")
    ax.set_ylabel("Empirical coverage")

    handles = [
        Line2D(
            [0],
            [0],
            color=KERNEL_STYLE["volume"][0],
            linestyle=KERNEL_STYLE["volume"][1],
            linewidth=1.6,
            label=KERNEL_STYLE["volume"][2],
        ),
        Line2D(
            [0],
            [0],
            color=KERNEL_STYLE["bare"][0],
            linestyle=KERNEL_STYLE["bare"][1],
            linewidth=1.6,
            label=KERNEL_STYLE["bare"][2],
        ),
        Line2D(
            [0],
            [0],
            color="0.45",
            linewidth=0.8,
            alpha=0.7,
            label=r"thin: $h_{\mathrm{true}} = 0.66,\ 0.78$",
        ),
        Line2D(
            [0],
            [0],
            color=EDGE,
            linestyle=(0, (4, 3)),
            linewidth=0.9,
            label="perfect calibration",
        ),
        Patch(
            facecolor=PRIOR,
            alpha=0.35,
            label=rf"$1\!-\!3\sigma$ binomial ($N={N_REALIZATIONS}$)",
        ),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6, framealpha=0.85)

    save_figure(fig, str(PDF_STEM), formats=("pdf",))


def main() -> None:
    """Compute P-P values, write the data artifact, and render the figure."""
    pp = compute_pp_values()
    summary = coverage_summary(pp)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(
        json.dumps(
            {
                "provenance": (
                    "master_thesis_code.validation.pp_coverage harness "
                    "(commission d2, results/commission_20260701/scratch/d2/); "
                    "paired data between kernels via shared per-realization seeds"
                ),
                "config": {
                    "seed": SEED,
                    "n_realizations": N_REALIZATIONS,
                    "n_events": N_EVENTS,
                    "sigma_z": SIGMA_Z,
                    "sigma_dl_frac": SIGMA_DL_FRAC,
                    "injected_truths": list(TRUTHS),
                    "kernels": list(KERNELS),
                    "h_grid": [ppc.PPCoverageConfig().h_min, ppc.PPCoverageConfig().h_max, H_STEP],
                    "omega_m": ppc.OMEGA_M,
                    "d50_gpc": ppc.D50_GPC,
                    "w_pdet_gpc": ppc.W_PDET_GPC,
                },
                "coverage": summary,
                "pp_values": pp,
            },
            indent=2,
        )
    )

    for kernel in KERNELS:
        for truth in (f"{t:.2f}" for t in TRUTHS):
            c = summary[kernel][truth]
            print(
                f"kernel={kernel:6s} h_true={truth} "
                f"cov50={c['50']:.3f} cov68={c['68']:.3f} cov90={c['90']:.3f}"
            )

    make_figure(pp)
    print(f"Wrote {PDF_STEM}.pdf and {DATA_PATH}")


if __name__ == "__main__":
    main()
