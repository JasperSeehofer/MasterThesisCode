"""SNR sanity for plunge-window ICs under the strain-referred effective
sensitivity S_eff = S_instr/R + S_c (R = 1.5*(2x sin x)^2), the estimator
validated against pilot horizon plateaus in highm_audit/measure_sens_snr.py.
(The raw h+ / TDI-A-PSD ratio is units-inconsistent -- exactly audit item 4.)
ESTIMATED absolutes (no TDI response); d_hor = SNR@1Gpc / 20.
"""
import json
import numpy as np
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode import PN5
from few.utils.utility import get_p_at_t
from few.waveform import GenerateEMRIWaveform
from master_thesis_code.constants import C, LISA_ARM_LENGTH, MAXIMAL_FREQUENCY, MINIMAL_FREQUENCY
from master_thesis_code.LISA_configuration import LisaTdiConfiguration

YRSID = 31558149.763545603
MTSUN = 4.925491025543576e-06
DT = 10.0
T_MISSION = 4.5
lisa = LisaTdiConfiguration()
traj = EMRIInspiral(func=PN5)
gen = GenerateEMRIWaveform(
    waveform_class="Pn5AAKWaveform",
    inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e6)},
    sum_kwargs={"pad_output": True}, frame="detector",
)

def p_at(M, t_pl, e0=0.1, x0=0.9, a=0.98, mu=10.0):
    traj.func.add_fixed_parameters(M, mu, a)
    p_lo = traj.func.min_p(e0, x0)
    p_up = max(2.0 * (256.0/5.0*(t_pl*YRSID)/(MTSUN*M)*(mu/M))**0.25, p_lo + 1.0)
    for _ in range(30):
        o = traj(M, mu, a, p_up, e0, x0, T=1.05*t_pl, err=1e-8)
        if o[0][-1] >= t_pl*YRSID: break
        p_up *= 1.5
    return get_p_at_t(traj, t_pl, [M, mu, a, e0, x0], bounds=[p_lo, p_up], xtol=1e-3, traj_kwargs={"err": 1e-8})

res = []
for M in (3e6, 1e7):
    for tp in (0.5, 2.0, 4.0):
        p0 = p_at(M, tp)
        h = gen(M, 10.0, 0.98, p0, 0.1, 0.9, 1.0, 1.2, 2.0, 1.0, 1.5, 0.0, 0.0, 0.0, T=T_MISSION, dt=DT)
        hp = np.ascontiguousarray(h.real); n = hp.size
        fs_full = np.fft.rfftfreq(n, DT)[1:]
        ht2_full = np.abs(np.fft.rfft(hp)[1:])**2
        lo = int(np.argmax(fs_full >= MINIMAL_FREQUENCY))
        hi = int(np.argmax(fs_full >= MAXIMAL_FREQUENCY)) or len(fs_full)
        fs, ht2 = fs_full[lo:hi], ht2_full[lo:hi]
        S_tdi = lisa.power_spectral_density(fs, channel="A")
        x = 2*np.pi*fs*LISA_ARM_LENGTH/C
        R = 1.5*(2*x*np.sin(x))**2
        Sc_raw = lisa._confusion_noise(fs)
        S_instr = S_tdi - R*Sc_raw          # tree has 49251f3: S_tdi = S_instr + R*Sc
        S_eff = S_instr/R + Sc_raw           # strain-referred effective sensitivity
        snr = float(np.sqrt(4.0*DT**2*np.trapezoid(ht2/S_eff, x=fs)))
        integ = ht2/S_eff
        cum = np.cumsum(0.5*(integ[1:]+integ[:-1])*np.diff(fs)); cum /= cum[-1]
        q = {f"f{int(z*100)}": float(fs[1:][np.searchsorted(cum, z)]) for z in (0.05, 0.5, 0.95)}
        r = dict(M=M, t_plunge=tp, p0=round(float(p0),4), snr_1gpc_seff=round(snr,2),
                 d_hor_gpc=round(snr/20.0,3), power_quantiles=q)
        res.append(r); print("SNR_SEFF", r, flush=True)
        del h, hp, ht2_full

with open("results/campaign51_20260728/plunge_window/snr_seff_measurements.json", "w") as f:
    json.dump(res, f, indent=2)
print("DONE")
