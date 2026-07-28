"""MEASURED (waveform) x ESTIMATED (response-independent ratios).

fastlisaresponse SIGILLs on this dev CPU, so we use bare few Pn5AAK h(t)
(detector frame, h+ - i hx) and compute the SNR of the SAME waveform under
three PSD variants. Absolute SNRs here lack the TDI response normalization,
but RATIOS between PSD variants and the power-location quantiles are
response-independent to the extent the TDI transfer varies slowly over the
signal band (checked via quantiles).

  S_code  = instrumental_TDI-A + raw S_c              (production PSD)
  S_instr = instrumental_TDI-A only
  S_corr  = instrumental_TDI-A + 1.5*(4x^2 sin^2 x)*S_c
            (lisatools A1TDISens.stochastic_transform, x = 2 pi f L / c)
"""

import json

import numpy as np
from few.waveform import GenerateEMRIWaveform

from master_thesis_code.constants import (
    C,
    LISA_ARM_LENGTH,
    MAXIMAL_FREQUENCY,
    MINIMAL_FREQUENCY,
)
from master_thesis_code.LISA_configuration import LisaTdiConfiguration

lisa = LisaTdiConfiguration()
DT = 10.0

gen = GenerateEMRIWaveform(
    waveform_class="Pn5AAKWaveform",
    inspiral_kwargs={"DENSE_STEPPING": 0, "max_init_len": int(1e6)},
    sum_kwargs={"pad_output": True},
    frame="detector",
)


def psd_variants(fs):
    S_code = lisa.power_spectral_density(fs, channel="A")
    Sc = lisa._confusion_noise(fs)
    S_instr = S_code - Sc
    x = 2 * np.pi * fs * LISA_ARM_LENGTH / C
    S_corr = S_instr + 1.5 * (4 * x**2 * np.sin(x) ** 2) * Sc
    return {"code": S_code, "instr_only": S_instr, "corrected": S_corr}


def analyze(M, p0, e0=0.1, x0=0.9, T=5.0):
    h = gen(M, 10.0, 0.98, p0, e0, x0, 1.0, 1.2, 2.0, 1.0, 1.5, 0.0, 0.0, 0.0, T=T, dt=DT)
    hp = np.ascontiguousarray(h.real)
    n = hp.size
    fs_full = np.fft.rfftfreq(n, DT)[1:]
    ht2_full = np.abs(np.fft.rfft(hp)[1:]) ** 2
    lo = int(np.argmax(fs_full >= MINIMAL_FREQUENCY))
    hi = int(np.argmax(fs_full >= MAXIMAL_FREQUENCY)) or len(fs_full)
    fs, ht2 = fs_full[lo:hi], ht2_full[lo:hi]
    out = {"below_fmin_power_frac": float(ht2_full[:lo].sum() / ht2_full.sum())}
    variants = psd_variants(fs)
    for name, S in variants.items():
        snr2 = 4.0 * DT**2 * np.trapezoid(ht2 / S, x=fs)
        out[f"snr_hplus_{name}"] = float(np.sqrt(snr2))
    for name in ["code", "corrected"]:
        integ = ht2 / variants[name]
        cum = np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(fs))
        cum /= cum[-1]
        out[f"powerloc_{name}"] = {
            f"f{int(q*100)}": float(fs[1:][np.searchsorted(cum, q)]) for q in (0.05, 0.5, 0.95)
        }
    out["suppression_code_vs_corrected"] = out["snr_hplus_corrected"] / out["snr_hplus_code"]
    del h, hp, ht2_full
    return out


CASES = [
    ("m5.5_p10", 10**5.5, 10.0),
    ("m6.0_p10", 1e6, 10.0),
    ("m6.2_p10", 10**6.2, 10.0),
    ("m6.4_p10", 10**6.4, 10.0),
    ("m6.6_p10", 10**6.6, 10.0),
    ("m7.0_p10", 1e7, 10.0),
    ("m6.6_p13", 10**6.6, 13.0),
    ("m7.4_p16", 10**7.4, 16.0),
]
res = {}
for tag, M, p0 in CASES:
    try:
        res[tag] = analyze(M, p0)
        print(tag, json.dumps(res[tag]), flush=True)
    except Exception as e:  # noqa: BLE001
        res[tag] = {"error": repr(e)}
        print(tag, "ERROR", repr(e), flush=True)

with open("results/campaign51_20260728/highm_audit/few_snr_measurement.json", "w") as f:
    json.dump(res, f, indent=2)
print("DONE")
