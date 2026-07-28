"""High-M audit calculations (campaign #51). READ-ONLY on source; MEASURED/ESTIMATED tags in output.

Items: (1) initial-condition convention quantification, (2) Nyquist, (3) inner-product band,
(4) PSD behaviour 1e-5..1e-2 Hz, (5) dN/dz set-[4] sanity, (6) R_emri branches.
"""

import json

import numpy as np

from master_thesis_code.constants import (
    MAXIMAL_FREQUENCY,
    MINIMAL_FREQUENCY,
)
from master_thesis_code.cosmological_model import Model1CrossCheck, polynomial, merger_distribution_coefficients
from master_thesis_code.LISA_configuration import LisaTdiConfiguration

OUT = {}

# ---- geometric-unit helpers -------------------------------------------------
G_SI = 6.674e-11
C_SI = 299792458.0
MSUN = 1.98892e30
TSUN = G_SI * MSUN / C_SI**3  # 4.925e-6 s
YR = 365.25 * 24 * 3600.0

A_SPIN = 0.98  # model assumption (fixed)


def f_orb_kerr(M, p, a=A_SPIN, prograde=True):
    """Circular equatorial Kerr orbital frequency at BL radius r=p*M (Hz).
    f = c^3/(2 pi G M) * 1/(r^{3/2} +/- a).  Bardeen Press Teukolsky 1972."""
    sgn = 1.0 if prograde else -1.0
    return 1.0 / (2 * np.pi * TSUN * M * (p ** 1.5 + sgn * a))


def r_isco_kerr(a=A_SPIN, prograde=True):
    z1 = 1 + (1 - a**2) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = np.sqrt(3 * a**2 + z1**2)
    sgn = -1.0 if prograde else 1.0
    return 3 + z2 + sgn * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def t_insp_peters_circ(M, mu, p0, pf):
    """Leading-order (Peters, circular) inspiral time p0 -> pf, seconds.
    t = (5/256) (G M/c^3) (M/mu) (p0^4 - pf^4)."""
    return (5.0 / 256.0) * TSUN * M * (M / mu) * (p0**4 - pf**4)


def p_of_tinsp(M, mu, t, pf):
    """Peters circular: p0 that plunges (reaches pf) after time t."""
    return (pf**4 + 256.0 / 5.0 * t / (TSUN * M * (M / mu))) ** 0.25


lisa = LisaTdiConfiguration()  # t_obs 4yr, confusion ON — same object as SNR path


def Sn(f):
    return lisa.power_spectral_density(np.atleast_1d(np.asarray(f, dtype=float)), channel="A")


# ---- (1b) snapshot convention quantification --------------------------------
MU = 10.0
risco = r_isco_kerr()
print(f"r_ISCO(a=0.98, prograde) = {risco:.4f} M")
rows = []
for Mz in [1e6, 10**6.2, 3e6, 1e7, 2.5e7]:
    f_isco = 2 * f_orb_kerr(Mz, risco)
    for p0 in [10.0, 13.0, 16.0]:
        fgw = 2 * f_orb_kerr(Mz, p0)
        t_pl = t_insp_peters_circ(Mz, MU, p0, 6.3) / YR  # to p_sep~6.3 (e~0.15)
        psd_ratio = float(Sn(fgw)[0] / Sn(min(f_isco, 0.99 * MAXIMAL_FREQUENCY))[0])
        rows.append(
            dict(
                Mz=Mz,
                p0=p0,
                f_gw_Hz=float(fgw),
                f_isco_gw_Hz=float(f_isco),
                t_insp_to_plunge_yr=float(t_pl),
                sqrt_PSD_ratio_f0_vs_fisco=float(np.sqrt(psd_ratio)),
            )
        )
        print(
            f"Mz={Mz:9.3g} p0={p0:4.1f}: f_gw(2xforb)={fgw:9.3e} Hz  f_gw@ISCO={f_isco:9.3e} Hz  "
            f"t_insp(p0->6.3)={t_pl:11.4g} yr  sqrt[S(f0)/S(fISCO)]={np.sqrt(psd_ratio):9.3e}"
        )
OUT["snapshot_table"] = rows

# p0 needed to plunge within 5 yr (Peters circular, ESTIMATED lower bound —
# relativistic corrections shorten t further, so true p0 is slightly larger):
print("\np0 required for plunge within T=5 yr (Peters circular to p=6.3):")
req = {}
for Mz in [1e6, 10**6.2, 10**6.5, 1e7, 2.5e7]:
    p_req = p_of_tinsp(Mz, MU, 5 * YR, 6.3)
    req[f"{Mz:.3g}"] = float(p_req)
    print(f"  Mz={Mz:9.3g}: p0_plunge5yr = {p_req:6.3f}  (draw band is [10,16])")
OUT["p0_for_5yr_plunge"] = req

# Boundary mass where t_insp(p0=10 -> 6.3) = 5 yr  (predicts the collapse edge)
from scipy.optimize import brentq

Medge = brentq(lambda M: t_insp_peters_circ(M, MU, 10.0, 6.3) - 5 * YR, 1e5, 1e8)
print(f"\nPredicted snapshot-convention collapse edge (t_insp(p0=10)=5yr): "
      f"M_z = {Medge:.4g}  log10 = {np.log10(Medge):.3f}")
OUT["predicted_collapse_edge_logM"] = float(np.log10(Medge))

# ---- monochromatic snapshot SNR scaling model -------------------------------
# For t_insp >> T the signal is ~monochromatic at f0=2 f_orb(p0):
# SNR ~ h0 sqrt(T) / sqrt(S(f0) ) * O(1) angular factors.
# h0 (sky/pol-avg quadrupole, circular): h = (32/5)^{1/2}... use
# h = 8/sqrt(5) (G Mc)^{5/3} (pi f)^{2/3} / (c^4 d)  [Maggiore Eq. 4.3-ish avg]
def h0_mono(Mz, mu, f, d_gpc):
    Mc = (mu ** (3 / 5)) * (Mz ** (2 / 5)) * MSUN
    d = d_gpc * 3.0857e25  # m
    return (
        8.0 / np.sqrt(5.0)
        * (G_SI * Mc) ** (5 / 3)
        * (np.pi * f) ** (2 / 3)
        / (C_SI**4 * d)
    )


T_SEC = 5 * YR
print("\nESTIMATED snapshot horizon d_hor(SNR=20) for monochromatic orbit at p0 (best case p0=10):")
dh = {}
for Mz in [1e6, 10**6.2, 10**6.4, 10**6.6, 10**6.8, 1e7, 2.5e7]:
    f0 = 2 * f_orb_kerr(Mz, 10.0)
    snr1 = h0_mono(Mz, MU, f0, 1.0) * np.sqrt(T_SEC / Sn(f0)[0])
    d_hor = snr1 / 20.0  # Gpc
    dh[f"{np.log10(Mz):.2f}"] = float(d_hor)
    print(f"  log10 Mz={np.log10(Mz):5.2f}: f0={f0:9.3e} Hz  d_hor(mono)={d_hor:10.4g} Gpc")
OUT["mono_snapshot_dhor_gpc"] = dh

# ---- plunge-convention SNR estimate (Newtonian AKK-style) -------------------
# Frequency-domain SPA: SNR^2 = 4 int |h(f)|^2/S df with
# |h(f)|^2 = (pi/12)... standard sky-avg: SNR^2 = int (hc/hn)^2 dlnf,
# hc^2 = (2/(3 pi^(1/3))) (G Mc)^{5/3} f^{-1/3} / (c^3 d^2) ... use
# hc(f)^2 = (2 f^2 / fdot) h0(f)^2 with Peters fdot.
def fdot_peters(Mz, mu, f):
    Mc = (mu ** (3 / 5)) * (Mz ** (2 / 5)) * MSUN
    return (96.0 / 5.0) * np.pi ** (8 / 3) * (G_SI * Mc / C_SI**3) ** (5 / 3) * f ** (11 / 3)


def snr_plunge(Mz, mu, d_gpc, T_yr=5.0):
    f_hi = 2 * f_orb_kerr(Mz, risco)
    # start frequency: orbit T_yr before ISCO (Peters circular backwards)
    p_hi = risco
    p_lo = p_of_tinsp(Mz, mu, T_yr * YR, p_hi)
    f_lo = 2 * f_orb_kerr(Mz, p_lo)
    fs = np.geomspace(f_lo, f_hi * 0.999, 400)
    hc2 = 2 * fs**2 / fdot_peters(Mz, mu, fs) * h0_mono(Mz, mu, fs, d_gpc) ** 2
    hn2 = fs * Sn(fs)
    return float(np.sqrt(np.trapezoid(hc2 / hn2, np.log(fs))))


print("\nESTIMATED plunge-convention horizon (Newtonian chirp ending at Kerr ISCO, last 5 yr):")
dhp = {}
for Mz in [1e6, 10**6.2, 10**6.5, 1e7, 2.5e7]:
    snr1 = snr_plunge(Mz, MU, 1.0)
    dhp[f"{np.log10(Mz):.2f}"] = float(snr1 / 20.0)
    print(f"  log10 Mz={np.log10(Mz):5.2f}: d_hor(plunge conv) = {snr1/20.0:8.3f} Gpc")
OUT["plunge_conv_dhor_gpc"] = dhp

# ---- (2)/(3) sampling + band ------------------------------------------------
dt = 10.0
print(f"\nSNR-path dt = {dt} s (ParameterEstimation.dt) -> f_Nyq = {1/(2*dt):.3f} Hz")
print(f"Inner product band: [{MINIMAL_FREQUENCY}, {MAXIMAL_FREQUENCY}] Hz")
OUT["nyquist_hz"] = 1 / (2 * dt)
OUT["band"] = [MINIMAL_FREQUENCY, MAXIMAL_FREQUENCY]
# fraction of monochromatic fundamental below f_min for heaviest case:
f0_heaviest = 2 * f_orb_kerr(2.5e7, 16.0)
print(f"Lowest fundamental in band question: f_gw(Mz=2.5e7, p0=16) = {f0_heaviest:.3e} Hz vs f_min=1e-5")

# ---- (4) PSD table ----------------------------------------------------------
fs = np.geomspace(1e-5, 1e-2, 61)
S = Sn(fs)
Sc = lisa._confusion_noise(fs)
Si = S - Sc
print("\nPSD table (A channel, T_obs=4yr, confusion ON):")
print(f"{'f [Hz]':>10} {'S_total':>12} {'S_instr':>12} {'S_conf':>12} {'conf/instr':>10}")
tab = []
for i in range(0, 61, 5):
    print(f"{fs[i]:10.3e} {S[i]:12.4e} {Si[i]:12.4e} {Sc[i]:12.4e} {Sc[i]/Si[i]:10.3f}")
    tab.append(dict(f=float(fs[i]), S_tot=float(S[i]), S_instr=float(Si[i]), S_conf=float(Sc[i])))
OUT["psd_table"] = tab
assert np.all(S > 0) and np.all(np.isfinite(S)), "PSD misbehaves!"
print("PSD positive+finite over [1e-5,1e-2]: OK")
# local log-slope of total PSD
sl = np.gradient(np.log10(S), np.log10(fs))
print(f"PSD log-slope at 1e-5: {sl[0]:.2f}, at 1e-4: {sl[20]:.2f}, at 1e-3: {sl[40]:.2f}")

# ---- (5) dN/dz set-[4] sanity ----------------------------------------------
zz = np.linspace(1e-3, 1.5, 200)
v4 = np.array([polynomial(z, *merger_distribution_coefficients[4]) for z in zz])
v3 = np.array([polynomial(z, *merger_distribution_coefficients[3]) for z in zz])
print(f"\ndN/dz set[4] over z in (0,1.5]: min={v4.min():.4g} max={v4.max():.4g} "
      f"negative fraction={np.mean(v4<0):.3f}")
print(f"dN/dz set[3] over z in (0,1.5]: min={v3.min():.4g} max={v3.max():.4g} "
      f"negative fraction={np.mean(v3<0):.3f}")
OUT["dNdz_set4"] = dict(min=float(v4.min()), max=float(v4.max()), neg_frac=float(np.mean(v4 < 0)))

# blended density at several masses:
for m in [6.0, 6.25, 6.5, 7.0]:
    d = np.array([Model1CrossCheck.dN_dz_of_mass(10**m, z) for z in zz])
    print(f"  blended dN/dz at log10M={m}: min={d.min():.4g} max={d.max():.4g} neg_frac={np.mean(d<0):.3f}")

# ---- (6) R_emri -------------------------------------------------------------
Ms = np.array([1e4, 1e5, 1.2e5, 2.5e5, 1e6, 1e7, 2.9e7])
print("\nR_emri branch values:")
for M in Ms:
    print(f"  M={M:9.3g}: R = {Model1CrossCheck.R_emri(M):8.3f} /yr")
OUT["R_emri"] = {f"{M:.3g}": float(Model1CrossCheck.R_emri(M)) for M in Ms}

with open("results/campaign51_20260728/highm_audit/audit_calcs_output.json", "w") as fjson:
    json.dump(OUT, fjson, indent=2)
print("\nSaved JSON.")
