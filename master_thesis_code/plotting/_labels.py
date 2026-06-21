"""LaTeX label constants for EMRI parameter axes.

Convention (per D-07): labels are fully typeset with physics symbols
AND units in mathtext, e.g. ``$M_\\bullet \\, [M_\\odot]$``.

Usage (Phase 17 bulk migration)::

    from master_thesis_code.plotting._labels import LABELS
    ax.set_xlabel(LABELS["d_L"])

Distance-unit reconciliation (VR-ANNO-06)
-----------------------------------------
Two unit-explicit luminosity-distance keys coexist because the package plots
``d_L`` data in two genuinely different units:

- ``d_L`` (Mpc): per-source recovery / Cramer-Rao plots, where ``luminosity_distance``
  is stored in Mpc.
- ``d_L_gpc`` (Gpc): pooled injection-CSV values (selection, fisher diagnostic
  scatter, paper closure, fig20), where ``luminosity_distance`` is stored in Gpc.

Both are physically correct for their data; the unit MUST always be stated, so
every distance axis routes through one of these two keys rather than a bare,
unit-ambiguous literal. No data value is converted by these labels.
"""

LABELS: dict[str, str] = {
    # --- EMRI intrinsic parameters ---
    "M": r"$M_\bullet \, [M_\odot]$",
    "mu": r"$\mu \, [M_\odot]$",
    "a": r"$a$",
    "p0": r"$p_0 \, [M]$",
    "e0": r"$e_0$",
    "Y0": r"$Y_0$",
    # --- Extrinsic / sky parameters ---
    "d_L": r"$d_L \, [\mathrm{Mpc}]$",
    "d_L_gpc": r"$d_L \, [\mathrm{Gpc}]$",
    "qS": r"$\theta_S \, [\mathrm{rad}]$",
    "phiS": r"$\phi_S \, [\mathrm{rad}]$",
    "qK": r"$\theta_K \, [\mathrm{rad}]$",
    "phiK": r"$\phi_K \, [\mathrm{rad}]$",
    "Phi_phi0": r"$\Phi_{\phi,0} \, [\mathrm{rad}]$",
    "Phi_theta0": r"$\Phi_{\theta,0} \, [\mathrm{rad}]$",
    "Phi_r0": r"$\Phi_{r,0} \, [\mathrm{rad}]$",
    # --- Observables ---
    "z": r"$z$",
    "SNR": r"$\rho$",
    "H0": r"$H_0 \, [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    "h": r"$h$",
    "f": r"$f \, [\mathrm{Hz}]$",
    "t": r"$t \, [\mathrm{s}]$",
    "PSD": r"$S_n(f) \, [\mathrm{Hz}^{-1}]$",
}
