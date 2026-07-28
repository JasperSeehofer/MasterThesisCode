"""Plunge-window convention measurements (MEASURED tags for the derivation doc).

1. e-transplant bound: PN5 trajectory e(t) from start to plunge for e0 drawn at
   start (the adopted realization) -- how far below e0 the realized plunge
   eccentricity e_p lands, vs the Peters small-e analytic map e_p = e0*(p_sep/p0)^(19/12).
2. few domain: Pn5AAK waveform generation at p0 in {7, 8, 9}, M_z = 1e7 (T=4.5).
3. SNR sanity under the FIXED PSD (commit 49251f3): M_z in {3e6, 1e7} x
   t_plunge in {0.5, 2, 4} yr, plunge-window p0, h+ SNR @ 1 Gpc (response-less,
   same estimator as highm_audit/measure_few_snr.py -- ESTIMATED absolutes).
"""

import json
import time

import numpy as np
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5
from few.utils.utility import get_p_at_t, get_separatrix
from few.waveform import GenerateEMRIWaveform

from master_thesis_code.constants import MAXIMAL_FREQUENCY, MINIMAL_FREQUENCY
from master_thesis_code.LISA_configuration import LisaTdiConfiguration

YRSID = 31558149.763545603
MTSUN = 4.925491025543576e-06
DT = 10.0
T_MISSION = 4.5
lisa = LisaTdiConfiguration()
traj = EMRIInspiral(func=PN5)
out = {}


def peters_p(t_pl_yr, M, mu):
    return (256.0 / 5.0 * (t_pl_yr * YRSID) / (MTSUN * M) * (mu / M)) ** 0.25


def p_at(M, t_pl, e0=0.1, x0=0.9, a=0.98, mu=10.0):
    traj.func.add_fixed_parameters(M, mu, a)
    p_lo = traj.func.min_p(e0, x0)
    p_up = max(2.0 * peters_p(t_pl, M, mu), p_lo + 1.0)
    for _ in range(30):
        o = traj(M, mu, a, p_up, e0, x0, T=1.05 * t_pl, err=1e-8)
        if o[0][-1] >= t_pl * YRSID:
            break
        p_up *= 1.5
    return get_p_at_t(
        traj, t_pl, [M, mu, a, e0, x0], bounds=[p_lo, p_up], xtol=1e-3,
        traj_kwargs={"err": 1e-8},
    )


# ---- 1. e-transplant bound (PN5-measured + Peters analytic) ----
etr = []
for M in (1e5, 1e6, 3e6, 1e7):
    for tp in (0.5, 2.25, 4.5):
        for e0 in (0.05, 0.2):
            p0 = p_at(M, tp, e0=e0)
            o = traj(M, 10.0, 0.98, p0, e0, 0.9, T=2 * tp)
            e_end = float(o[2][-1])
            p_end = float(o[1][-1])
            p_sep = float(get_separatrix(0.98, e_end, 0.9))
            peters_ep = e0 * (p_sep / p0) ** (19.0 / 12.0)
            etr.append(
                dict(M=M, t_plunge=tp, e0=e0, p0=round(p0, 4), e_plunge=round(e_end, 4),
                     p_end=round(p_end, 4), p_sep=round(p_sep, 4),
                     peters_e_plunge=round(peters_ep, 4))
            )
            print("ETRANS", etr[-1], flush=True)
out["e_transplant"] = etr

# ---- 2. few domain: waveforms at p0 in {7,8,9}, M_z=1e7 ----
gen = GenerateEMRIWaveform(
    waveform_class="Pn5AAKWaveform",
    inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e6)},
    sum_kwargs={"pad_output": True},
    frame="detector",
)
dom = []
for p0 in (7.0, 8.0, 9.0):
    t1 = time.perf_counter()
    try:
        h = gen(1e7, 10.0, 0.98, p0, 0.1, 0.9, 1.0, 1.2, 2.0, 1.0, 1.5, 0.0, 0.0, 0.0,
                T=T_MISSION, dt=DT)
        o = traj(1e7, 10.0, 0.98, p0, 0.1, 0.9, T=T_MISSION)
        dom.append(dict(p0=p0, ok=True, n=int(h.size), t_plunge_yr=round(float(o[0][-1] / YRSID), 4),
                        gen_s=round(time.perf_counter() - t1, 1)))
    except Exception as e:  # noqa: BLE001
        dom.append(dict(p0=p0, ok=False, error=repr(e)))
    print("DOMAIN", dom[-1], flush=True)
    del h
out["p0_domain_waveforms"] = dom

# ---- 3. SNR sanity, fixed PSD, plunge-window ICs ----
def snr_at_1gpc(M, t_pl):
    p0 = p_at(M, t_pl)
    h = gen(M, 10.0, 0.98, p0, 0.1, 0.9, 1.0, 1.2, 2.0, 1.0, 1.5, 0.0, 0.0, 0.0,
            T=T_MISSION, dt=DT)
    hp = np.ascontiguousarray(h.real)
    n = hp.size
    fs_full = np.fft.rfftfreq(n, DT)[1:]
    ht2_full = np.abs(np.fft.rfft(hp)[1:]) ** 2
    lo = int(np.argmax(fs_full >= MINIMAL_FREQUENCY))
    hi = int(np.argmax(fs_full >= MAXIMAL_FREQUENCY)) or len(fs_full)
    fs, ht2 = fs_full[lo:hi], ht2_full[lo:hi]
    S = lisa.power_spectral_density(fs, channel="A")
    snr = float(np.sqrt(4.0 * DT**2 * np.trapezoid(ht2 / S, x=fs)))
    integ = ht2 / S
    cum = np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(fs))
    cum /= cum[-1]
    q = {f"f{int(x*100)}": float(fs[1:][np.searchsorted(cum, x)]) for x in (0.05, 0.5, 0.95)}
    del h, hp, ht2_full
    return dict(M=M, t_plunge=t_pl, p0=round(p0, 4), snr_1gpc=round(snr, 3),
                d_hor_gpc=round(snr / 20.0, 4), power_quantiles=q)


snrres = []
for M in (3e6, 1e7):
    for tp in (0.5, 2.0, 4.0):
        r = snr_at_1gpc(M, tp)
        snrres.append(r)
        print("SNR", r, flush=True)
out["snr_fixed_psd_plunge_window"] = snrres

with open(
    "results/campaign51_20260728/plunge_window/plunge_window_measurements.json", "w"
) as f:
    json.dump(out, f, indent=2)
print("DONE")
