"""ESTIMATED-corrected absolute SNR: strain-referred effective sensitivity
S_eff = S_instr_TDI / R + S_c_raw, with R = 1.5*4x^2 sin^2 x (lisatools
stochastic transform used as the sky-averaged strain->TDI-A power response).
Calibration check: the same estimator with S_code reproduces pilot SNR@1Gpc
scales at m5.5 (power >5 mHz, response-flat regime)."""
import json
import numpy as np
from few.waveform import GenerateEMRIWaveform
from master_thesis_code.constants import C, LISA_ARM_LENGTH, MAXIMAL_FREQUENCY, MINIMAL_FREQUENCY
from master_thesis_code.LISA_configuration import LisaTdiConfiguration

lisa = LisaTdiConfiguration(); DT = 10.0
gen = GenerateEMRIWaveform(waveform_class="Pn5AAKWaveform",
    inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e6)},
    sum_kwargs={"pad_output": True}, frame="detector")

CASES = [("m5.5_p10",10**5.5,10.),("m6.0_p10",1e6,10.),("m6.2_p10",10**6.2,10.),
         ("m6.4_p10",10**6.4,10.),("m6.6_p10",10**6.6,10.),("m7.0_p10",1e7,10.),
         ("m6.6_p13",10**6.6,13.),("m7.4_p16",10**7.4,16.)]
res={}
for tag,M,p0 in CASES:
    h = gen(M,10.,0.98,p0,0.1,0.9,1.0,1.2,2.0,1.0,1.5,0.,0.,0.,T=5.0,dt=DT)
    hp = np.ascontiguousarray(h.real); n=hp.size
    fs_full = np.fft.rfftfreq(n,DT)[1:]
    ht2 = np.abs(np.fft.rfft(hp)[1:])**2
    lo = int(np.argmax(fs_full>=MINIMAL_FREQUENCY)); hi = int(np.argmax(fs_full>=MAXIMAL_FREQUENCY)) or len(fs_full)
    fs, ht2 = fs_full[lo:hi], ht2[lo:hi]
    S_code = lisa.power_spectral_density(fs); Sc = lisa._confusion_noise(fs)
    Si = S_code - Sc
    x = 2*np.pi*fs*LISA_ARM_LENGTH/C; R = 1.5*4*x**2*np.sin(x)**2
    S_eff = Si/R + Sc
    snr_sens = float(np.sqrt(4*DT**2*np.trapezoid(ht2/S_eff, x=fs)))
    snr_sens_noconf = float(np.sqrt(4*DT**2*np.trapezoid(ht2/(Si/R), x=fs)))
    res[tag] = dict(snr_sens_1gpc=snr_sens, snr_sens_noconf_1gpc=snr_sens_noconf,
                    dhor_corrected_gpc=snr_sens/20.0)
    print(tag, json.dumps(res[tag]), flush=True)
    del h, hp, ht2
json.dump(res, open("results/campaign51_20260728/highm_audit/sens_snr_measurement.json","w"), indent=2)
print("DONE")
