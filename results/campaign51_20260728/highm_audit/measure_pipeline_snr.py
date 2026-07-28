"""MEASURED: run the actual SNR path (CPU, few Pn5AAK + fastlisaresponse TDI1 AE)
for representative masses; decompose the collapse into (a) confusion-noise
normalization artifact, (b) snapshot-convention frequency placement, by
recomputing the SAME waveform's SNR under three PSDs:
  S_code    = instrumental + raw S_c            (production)
  S_instr   = instrumental only
  S_corr    = instrumental + 1.5*4x^2 sin^2(x) * S_c   (lisatools A1TDISens.stochastic_transform)
No source files modified — in-memory PSD swap only.
"""

import json

import numpy as np

from master_thesis_code.constants import LISA_ARM_LENGTH, C, MAXIMAL_FREQUENCY, MINIMAL_FREQUENCY
from master_thesis_code.datamodels.parameter_space import ParameterSpace
from master_thesis_code.LISA_configuration import LisaTdiConfiguration
from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation
from master_thesis_code.waveform_generator import WaveGeneratorType

lisa = LisaTdiConfiguration()


def psd_variants(fs):
    S_code = lisa.power_spectral_density(fs, channel="A")
    Sc = lisa._confusion_noise(fs)
    S_instr = S_code - Sc
    x = 2 * np.pi * fs * LISA_ARM_LENGTH / C
    S_corr = S_instr + 1.5 * (4 * x**2 * np.sin(x) ** 2) * Sc  # lisatools A1TDISens
    return S_code, S_instr, S_corr


def snr_from_waveform(wf, dt=10.0):
    n = wf.shape[-1]
    fs_full = np.fft.rfftfreq(n, dt)[1:]
    lo = int(np.argmax(fs_full >= MINIMAL_FREQUENCY))
    hi = int(np.argmax(fs_full >= MAXIMAL_FREQUENCY)) or len(fs_full)
    fs = fs_full[lo:hi]
    ffts = np.fft.rfft(wf, axis=-1)[:, 1 + lo : 1 + hi]
    power = (np.abs(ffts) ** 2).sum(axis=0)  # summed over A,E
    out = {}
    S_code, S_instr, S_corr = psd_variants(fs)
    for name, S in [("code", S_code), ("instr_only", S_instr), ("corrected", S_corr)]:
        snr2 = 4.0 * dt**2 * np.trapezoid(power / S, x=fs)
        out[name] = float(np.sqrt(snr2))
    # cumulative SNR^2 fraction vs frequency under code PSD and corrected PSD
    for name, S in [("code", S_code), ("corrected", S_corr)]:
        integrand = power / S
        cum = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(fs))
        cum = cum / cum[-1]
        qs = {}
        for q in [0.05, 0.5, 0.95]:
            qs[f"f_at_{int(q*100)}pct"] = float(fs[1:][np.searchsorted(cum, q)])
        out[f"powerloc_{name}"] = qs
    return out


ps = ParameterSpace()
pe = ParameterEstimation(
    WaveGeneratorType.PN5_AAK, ps, use_gpu=False, use_five_point_stencil=True
)

CASES = [
    dict(tag="m5.5_p10", M=10**5.5, p0=10.0),
    dict(tag="m6.0_p10", M=1e6, p0=10.0),
    dict(tag="m6.2_p10", M=10**6.2, p0=10.0),
    dict(tag="m6.4_p10", M=10**6.4, p0=10.0),
    dict(tag="m6.6_p10", M=10**6.6, p0=10.0),
    dict(tag="m7.0_p10", M=1e7, p0=10.0),
    dict(tag="m6.6_p13", M=10**6.6, p0=13.0),
]

results = {}
for c in CASES:
    ps.M.value = c["M"]  # detector-frame M_z by pipeline convention
    ps.mu.value = 10.0
    ps.a.value = 0.98
    ps.p0.value = c["p0"]
    ps.e0.value = 0.1
    ps.x0.value = 0.9  # prograde
    ps.luminosity_distance.value = 1.0  # Gpc
    ps.qS.value = 1.2
    ps.phiS.value = 2.0
    ps.qK.value = 1.0
    ps.phiK.value = 1.5
    ps.Phi_phi0.value = 0.0
    ps.Phi_theta0.value = 0.0
    ps.Phi_r0.value = 0.0
    try:
        wf = pe.generate_lisa_response()
        wf = np.asarray(wf)
        r = snr_from_waveform(wf, dt=pe.dt)
        # cross-check against the production method itself
        r["production_snr"] = float(np.sqrt(pe.scalar_product_of_functions(wf, wf)))
        r["n_samples"] = int(wf.shape[-1])
        results[c["tag"]] = r
        print(c["tag"], json.dumps(r), flush=True)
        del wf
    except Exception as e:  # noqa: BLE001
        results[c["tag"]] = {"error": repr(e)}
        print(c["tag"], "ERROR", repr(e), flush=True)

with open("results/campaign51_20260728/highm_audit/pipeline_snr_measurement.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE")
