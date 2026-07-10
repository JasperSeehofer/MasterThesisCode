"""Independent P-P / coverage harness for the dark-siren H0 estimator family.

This module is the calibration instrument for the dark-siren estimator: it
builds a synthetic galaxy catalogue and EMRI detections in a flat LambdaCDM
universe, runs a from-scratch single-host dark-siren H0 estimator with a
switchable host-redshift kernel (``"bare"`` Gaussian ``N(z; z_gal, sigma_z)``
vs ``"volume"``-weighted ``N(z; z_gal, sigma_z) * dV_c/dz / (1+z)``), and
measures frequentist P-P coverage (50/68/90% HPD credible intervals) and MAP
bias across many realizations and injected truths.

Scientific independence
-----------------------
The harness is pure numpy/scipy and deliberately does NOT import the
production inference code (``master_thesis_code.bayesian_inference``). That
independence is its scientific value: it re-derives the estimator from the
written formulas, so a calibration failure here cannot be explained away as a
shared implementation bug. It was written from scratch by the 2026-07-01
verification commission (investigator d2); the original scratch version,
findings note and reference outputs live in
``results/commission_20260701/scratch/d2/`` (see
``NOTE_calibration_findings.md`` and ``coverage_results.json``) and the
commission verification report section 7.

Key commission finding reproduced by this harness: with photo-z scatter
``sigma_z ~= 0.035`` the bare-Gaussian host-z kernel carries a fixed
``~ -sigma_z^2 * d ln(dV_c/dz)/dz`` (Eddington/Malmquist-in-z) low bias in H0
that collapses coverage to ~0-3%, while the volume-weighted kernel is
calibrated (coverage ~= nominal, bias ~= 0).

Catalogue-support-truncated mode (``PPCoverageConfig.z_support``): splits the
detected population by true host redshift so hosts with ``z_host >=
z_support`` become zero-host events driven by the pure-completion likelihood
B_num(h)/D(h) — the ``L_cat -> 0`` limit of the Gray et al. (2020,
arXiv:1908.06050, Eqs. 29+32) mixture that production commit ``8db6c6e``
(issue #29) installed in ``bayesian_statistics.py``. ``z_support=None``
(default) reproduces the pre-2026-07-10 harness bit-identically. See
``.planning/HANDOFF-LOCAL-NO-CLUSTER-20260710.md`` (item L-A) and
``results/pp_coverage_deepvenue_20260710/RUNBOOK.md``.

Units: ``h`` in [100 km/s/Mpc]; distances in Gpc. Cosmology: flat LambdaCDM.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.special import erfc

C_KM_S = 299_792.458
OMEGA_M = 0.30
OMEGA_L = 0.70

D50_GPC = 1.85  # 50% detection-probability luminosity distance [Gpc]
W_PDET_GPC = 0.30  # detection roll-off width [Gpc]

Z_MIN = 1e-4
Z_MAX_POP = 0.95  # population / catalogue redshift ceiling

# ----------------------------------------------------------------------------
# Cosmology tables (flat LambdaCDM): d_L(z, h) = A(z) / h with A in Gpc, and
# population weight w_pop(z) propto dV_c/dz / (1+z) (the 1/h^3 cancels).
# ----------------------------------------------------------------------------
_Z_GRID: npt.NDArray[np.float64] = np.linspace(0.0, 1.5, 15_001)
_E_OF_Z: npt.NDArray[np.float64] = np.sqrt(OMEGA_M * (1.0 + _Z_GRID) ** 3 + OMEGA_L)
_INV_E: npt.NDArray[np.float64] = 1.0 / _E_OF_Z
_I_OF_Z: npt.NDArray[np.float64] = np.concatenate(
    [
        np.array([0.0]),
        np.cumsum(0.5 * (_INV_E[1:] + _INV_E[:-1]) * np.diff(_Z_GRID)),
    ]
)
# A(z) = (1+z) * (c / 100 km/s/Mpc) * I(z) in Mpc, / 1000 -> Gpc.
_A_GPC: npt.NDArray[np.float64] = (1.0 + _Z_GRID) * (C_KM_S / 100.0) * _I_OF_Z / 1000.0
# w_pop(z) propto I(z)^2 / E(z) / (1+z)  (comoving volume element / (1+z))
_W_POP: npt.NDArray[np.float64] = np.where(
    _Z_GRID > 0.0, _I_OF_Z**2 / _E_OF_Z / (1.0 + _Z_GRID), 0.0
)


def comoving_amplitude_of_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Interpolate A(z) [Gpc] such that d_L(z, h) = A(z) / h.

    Args:
        z: Redshift values.

    Returns:
        A(z) in Gpc.
    """
    return np.interp(z, _Z_GRID, _A_GPC)


def z_of_comoving_amplitude(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Invert A(z): redshift at which d_L * h equals ``a`` [Gpc].

    Args:
        a: Amplitude values A = d_L * h in Gpc.

    Returns:
        Redshift values.
    """
    return np.interp(a, _A_GPC, _Z_GRID)


def population_weight_of_z(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Population redshift weight w_pop(z) propto dV_c/dz / (1+z) (unnormalized).

    Args:
        z: Redshift values.

    Returns:
        Unnormalized population weight at each redshift.
    """
    return np.interp(z, _Z_GRID, _W_POP)


def detection_probability(d_L: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Smooth Malmquist detection probability p_det(d_L).

    Args:
        d_L: Luminosity distance values in Gpc.

    Returns:
        Detection probability in [0, 1] (50% at ``D50_GPC``).
    """
    return np.asarray(
        0.5 * erfc((np.asarray(d_L) - D50_GPC) / (np.sqrt(2.0) * W_PDET_GPC)),
        dtype=np.float64,
    )


def _norm_pdf(
    x: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64] | float,
    sig: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """Gaussian probability density N(x; mu, sig)."""
    return np.asarray(
        np.exp(-0.5 * ((x - mu) / sig) ** 2) / (np.sqrt(2.0 * np.pi) * sig),
        dtype=np.float64,
    )


def _sample_detected_redshifts(
    h_true: float, n: int, rng: np.random.Generator, ngrid: int = 2000
) -> npt.NDArray[np.float64]:
    """Draw host redshifts from the detected population w_pop(z) * p_det(d_L(z, h))."""
    zg = np.linspace(Z_MIN, Z_MAX_POP, ngrid)
    pdf = np.clip(
        population_weight_of_z(zg) * detection_probability(comoving_amplitude_of_z(zg) / h_true),
        0.0,
        None,
    )
    cdf = np.concatenate([np.array([0.0]), np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, zg)


def _hpd_contains(
    h_grid: npt.NDArray[np.float64],
    post: npt.NDArray[np.float64],
    h_true: float,
    level: float,
) -> bool:
    """Return True if ``h_true`` lies inside the HPD credible region of mass ``level``."""
    dh = np.gradient(h_grid)
    mass = post * dh
    order = np.argsort(post)[::-1]
    csum = np.cumsum(mass[order])
    k = int(np.searchsorted(csum, level))
    k = min(k, order.size - 1)
    thresh = float(post[order[k]])
    p_true = float(np.interp(h_true, h_grid, post))
    return p_true >= thresh


@dataclass
class PPCoverageConfig:
    """Configuration for the P-P / coverage harness.

    Args:
        n_realizations: Independent synthetic universes per injected truth.
        n_events: Detected EMRI events per realization.
        sigma_z: Host photo-z scatter (commission value 0.035).
        sigma_dl_frac: Fractional GW luminosity-distance error.
        injected_truths: Injected H0 values [100 km/s/Mpc]; defaults include
            near-grid-edge truths to exercise rail behaviour.
        seed: Master seed; all randomness flows from
            ``np.random.default_rng(seed)`` (fully deterministic).
        kernel: Host-z kernel — ``"bare"`` Gaussian (production-style) or
            ``"volume"``-weighted (calibrated).
        h_min: Lower edge of the H0 grid.
        h_max: Upper edge of the H0 grid.
        h_step: H0 grid spacing.
        n_z_quad: Per-event redshift quadrature points.
        z_support: Catalogue support ceiling: true hosts with z_host <
            z_support are in the catalogue (existing single-host kernel
            branch); z_host >= z_support are zero-host events using the
            pure-completion likelihood B_num/D (issue #29 analog). None
            (default) => no truncation, bit-identical to the pre-2026-07-10
            harness.
    """

    n_realizations: int = 120
    n_events: int = 250
    sigma_z: float = 0.035
    # Flat peculiar-velocity redshift-error term, added in quadrature to
    # sigma_z for BOTH the generative truth scatter and the inference kernel
    # (the calibrated case). Default 0.0 keeps the committed anchor runs
    # bit-identical. Issue #16: the production kernel uses
    # (1+z) * SIGMA_V_PEC_KM_S / c; this harness knob is flat because its
    # sigma_z is flat too.
    sigma_z_pv: float = 0.0
    sigma_dl_frac: float = 0.05
    injected_truths: list[float] = field(default_factory=lambda: [0.62, 0.72, 0.84])
    seed: int = 20260701
    kernel: Literal["bare", "volume"] = "volume"
    h_min: float = 0.600
    h_max: float = 0.860
    h_step: float = 0.004
    n_z_quad: int = 160
    z_support: float | None = None

    def h_grid(self) -> npt.NDArray[np.float64]:
        """Return the H0 evaluation grid."""
        return np.arange(self.h_min, self.h_max + 0.5 * self.h_step, self.h_step)


def _run_realization(
    h_true: float,
    h_grid: npt.NDArray[np.float64],
    log_Dh: npt.NDArray[np.float64],
    config: PPCoverageConfig,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float64], int]:
    """Simulate one realization and return the accumulated log-likelihood on ``h_grid``.

    Clean single-host limit (fully complete catalogue, one candidate host per
    event): the per-event likelihood is p_i(h) = num(h) / D(h) with

        num_bare(h)   = int p_GW(A(z)/h) N(z; z_g, sigma_z) dz
        num_volume(h) = int p_GW(A(z)/h) N(z; z_g, sigma_z) w_pop(z) / Z_g dz
        D(h)          = int p_det(A(z)/h) w_pop(z) dz

    so only the host-z kernel differs between the two estimator variants.

    Catalogue-support-truncated mode (``config.z_support`` not None): true
    hosts with ``z_host >= z_support`` are treated as zero-host events and
    use the pure-completion likelihood B_num(h)/D(h) instead — the
    ``L_cat -> 0`` limit of the Gray et al. (2020, arXiv:1908.06050, Eqs.
    29+32) mixture that production commit ``8db6c6e`` (issue #29) installed
    in ``bayesian_statistics.py``; see also Gray, Messenger & Veitch (2022,
    arXiv:2111.04629, Eq. 5) and
    ``docs/derivations/G2a_completion_sky_marginal_4pi.md`` limiting case 2.
    The completion integral is capped at ``Z_MAX_POP`` (the issue #30
    parallel), matching the shared D(h) domain, and shares D(h)'s exact
    unnormalized measure (no extra h-dependent normalization).

    Returns:
        Tuple of (accumulated log-likelihood on ``h_grid``, number of
        zero-host events in this realization).
    """
    # One effective sigma for the truth-scatter draw AND the kernel keeps the
    # generative model and the inference consistent (calibrated case).
    sigma_z = float(np.hypot(config.sigma_z, config.sigma_z_pv))
    z_host = _sample_detected_redshifts(h_true, config.n_events, rng)
    dL_host = comoving_amplitude_of_z(z_host) / h_true
    dL_obs = np.clip(dL_host + rng.normal(0.0, config.sigma_dl_frac * dL_host), 1e-3, None)
    sig_dl = config.sigma_dl_frac * dL_obs
    z_gal = np.clip(z_host + rng.normal(0.0, sigma_z, config.n_events), Z_MIN, None)

    logL = np.zeros(h_grid.size)
    n_zero_host = 0
    for i in range(config.n_events):
        if config.z_support is not None and z_host[i] >= config.z_support:
            zs: float = config.z_support
            n_zero_host += 1
            # Pure-completion branch: no kernel padding (there is no kernel
            # here), capped at Z_MAX_POP (issue #30 parallel; matches D(h)).
            z_lo_b = max(
                Z_MIN,
                zs,
                float(
                    z_of_comoving_amplitude(np.asarray((dL_obs[i] - 5 * sig_dl[i]) * h_grid.min()))
                ),
            )
            z_hi_b = min(
                Z_MAX_POP,
                float(
                    z_of_comoving_amplitude(np.asarray((dL_obs[i] + 5 * sig_dl[i]) * h_grid.max()))
                ),
            )
            if z_hi_b <= z_lo_b:
                num_b = np.full(h_grid.size, 1e-300)
            else:
                zq_b = np.linspace(z_lo_b, z_hi_b, config.n_z_quad)
                wq_b = np.gradient(zq_b)
                dLg_b = comoving_amplitude_of_z(zq_b)[:, None] / h_grid[None, :]  # (nz, nh)
                pGW_b = _norm_pdf(dLg_b, float(dL_obs[i]), float(sig_dl[i]))  # (nz, nh)
                wpop_b = population_weight_of_z(zq_b)  # unnormalized
                num_b = (wq_b * wpop_b) @ pGW_b  # (nh,)
            logL += np.log(np.clip(num_b, 1e-300, None)) - log_Dh
            continue
        z_lo = max(
            Z_MIN,
            float(z_of_comoving_amplitude(np.asarray((dL_obs[i] - 5 * sig_dl[i]) * h_grid.min())))
            - 4 * sigma_z,
        )
        z_hi = min(
            float(_Z_GRID[-1]),
            float(z_of_comoving_amplitude(np.asarray((dL_obs[i] + 5 * sig_dl[i]) * h_grid.max())))
            + 4 * sigma_z,
        )
        zq = np.linspace(z_lo, z_hi, config.n_z_quad)
        wq = np.gradient(zq)
        dLg = comoving_amplitude_of_z(zq)[:, None] / h_grid[None, :]  # (nz, nh)
        pGW = _norm_pdf(dLg, float(dL_obs[i]), float(sig_dl[i]))  # (nz, nh)
        kernel_z = _norm_pdf(zq, float(z_gal[i]), sigma_z)  # (nz,)
        if config.kernel == "volume":
            kernel_z = kernel_z * population_weight_of_z(zq)
            kernel_z = kernel_z / max(float(np.trapezoid(kernel_z, zq)), 1e-300)
        num = (wq * kernel_z) @ pGW  # (nh,)
        logL += np.log(np.clip(num, 1e-300, None)) - log_Dh
    return logL, n_zero_host


def run_coverage(config: PPCoverageConfig) -> dict[str, Any]:
    """Run the P-P / coverage test for one kernel choice.

    Args:
        config: Harness configuration; all randomness is seeded from
            ``config.seed`` via ``np.random.default_rng``.

    Returns:
        JSON-serializable dict with keys ``"config"`` (the config as a dict)
        and ``"results"`` — one entry per injected truth (stringified H0)
        containing ``coverage`` (fractions at 50/68/90% HPD),
        ``rail_fraction``, ``map_mean``, ``map_std``, ``map_median``,
        ``map_bias`` (map_mean - truth), and ``completion_fraction`` (mean
        fraction of events routed into the ``z_support`` zero-host
        pure-completion branch per realization; 0.0 when ``z_support`` is
        None or >= ``Z_MAX_POP``).
    """
    h_grid = config.h_grid()
    # Selection denominator D(h) = int p_det(A(z)/h) w_pop(z) dz (shared).
    zint = np.linspace(Z_MIN, Z_MAX_POP, 3000)
    wpop = population_weight_of_z(zint)
    Dh = np.trapezoid(
        detection_probability(comoving_amplitude_of_z(zint)[:, None] / h_grid[None, :])
        * wpop[:, None],
        zint,
        axis=0,
    )
    log_Dh = np.log(Dh)

    master = np.random.default_rng(config.seed)
    results: dict[str, Any] = {}
    levels = {"50": 0.50, "68": 0.68, "90": 0.90}
    for h_true in config.injected_truths:
        cov = {name: 0 for name in levels}
        rail = 0
        maps: list[float] = []
        completion_fractions: list[float] = []
        for _ in range(config.n_realizations):
            rng = np.random.default_rng(int(master.integers(1 << 62)))
            logL, n_zero_host = _run_realization(h_true, h_grid, log_Dh, config, rng)
            completion_fractions.append(n_zero_host / config.n_events)
            post = np.exp(logL - logL.max())
            post /= np.trapezoid(post, h_grid)
            mi = int(np.argmax(post))
            maps.append(float(h_grid[mi]))
            if mi == 0 or mi == h_grid.size - 1:
                rail += 1
            for name, lv in levels.items():
                if _hpd_contains(h_grid, post, h_true, lv):
                    cov[name] += 1
        n = config.n_realizations
        results[f"{h_true:.4f}"] = {
            "h_true": h_true,
            "coverage": {name: cov[name] / n for name in levels},
            "rail_fraction": rail / n,
            "map_mean": float(np.mean(maps)),
            "map_std": float(np.std(maps)),
            "map_median": float(np.median(maps)),
            "map_bias": float(np.mean(maps)) - h_true,
            "completion_fraction": float(np.mean(completion_fractions)),
        }
    return {"config": asdict(config), "results": results}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the harness and write a JSON results file.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        description="Independent P-P/coverage calibration harness for the "
        "dark-siren H0 estimator (commission d2 provenance)."
    )
    parser.add_argument("--n-realizations", type=int, default=120)
    parser.add_argument("--n-events", type=int, default=250)
    parser.add_argument("--sigma-z", type=float, default=0.035)
    parser.add_argument("--sigma-z-pv", type=float, default=0.0)
    parser.add_argument("--sigma-dl-frac", type=float, default=0.05)
    parser.add_argument("--truths", type=float, nargs="+", default=[0.62, 0.72, 0.84])
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--kernel", choices=["bare", "volume"], default="volume")
    parser.add_argument("--output", type=Path, default=Path("pp_coverage_results.json"))
    parser.add_argument(
        "--z-support",
        type=float,
        default=None,
        help="Catalogue support ceiling: true hosts with z_host >= z_support become "
        "zero-host events using the pure-completion likelihood B_num/D (issue #29 "
        "analog). Default None disables truncation (bit-identical to the "
        "pre-2026-07-10 harness).",
    )
    args = parser.parse_args(argv)

    config = PPCoverageConfig(
        n_realizations=args.n_realizations,
        n_events=args.n_events,
        sigma_z=args.sigma_z,
        sigma_z_pv=args.sigma_z_pv,
        sigma_dl_frac=args.sigma_dl_frac,
        injected_truths=list(args.truths),
        seed=args.seed,
        kernel=args.kernel,
        z_support=args.z_support,
    )
    out = run_coverage(config)
    args.output.write_text(json.dumps(out, indent=2))
    for key, r in out["results"].items():
        print(
            f"h_true={key} [{config.kernel:6s}] "
            f"cov50={r['coverage']['50']:.2f} cov68={r['coverage']['68']:.2f} "
            f"cov90={r['coverage']['90']:.2f} rail={r['rail_fraction']:.2f} "
            f"MAP={r['map_mean']:.4f} bias={r['map_bias']:+.4f} "
            f"completion_fraction={r['completion_fraction']:.2f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
