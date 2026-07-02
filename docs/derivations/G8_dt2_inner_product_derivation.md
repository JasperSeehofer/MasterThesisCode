# G8 — The missing dt² in the noise-weighted inner product: derivation and evidence

**Claim.** `scalar_product_of_functions` (`parameter_estimation/parameter_estimation.py:317-374`)
returns exactly `⟨h₁|h₂⟩ / dt²`, where `⟨·|·⟩` is the standard GW matched-filter inner product.
At `dt = 10 s`: all SNRs are 10× too small, all Fisher matrices 100× too small, all Cramér–Rao
standard deviations 10× too large.

---

## 1. Derivation

### 1.1 The target quantity

The standard noise-weighted inner product (Finn 1992, PRD 46 5236, Eq. 2.3; Cutler & Flanagan
1994, arXiv:gr-qc/9402014, Eq. 2.4; Maggiore, *Gravitational Waves* Vol. 1, Eq. 7.46) is

    ⟨h₁|h₂⟩ = 4 Re ∫₀^∞ df  h̃₁(f) h̃₂*(f) / S_n(f)                                   (1)

with h̃(f) the **continuous** Fourier transform

    h̃(f) = ∫ dt h(t) e^{-2πift}                                                     (2)

and S_n(f) the **one-sided** PSD with units [h²/Hz] (here: the Babak et al. 2023,
arXiv:2303.15929, analytic LISA PSD evaluated on a physical frequency axis in Hz — see
`LISA_configuration.power_spectral_density` and `_get_cached_psd`, which builds
`fs = rfftfreq(n, dt)` in Hz). SNR² = ⟨h|h⟩, and the Fisher matrix is Γ_ij = ⟨∂_i h|∂_j h⟩.

### 1.2 DFT ↔ continuous FT

`numpy.fft.rfft` returns the **un-normalized DFT**

    X_k = Σ_{m=0}^{n-1} h(t_m) e^{-2πikm/n}.                                        (3)

Approximating the integral (2) by its Riemann sum on the sampling grid t_m = m·dt:

    h̃(f_k) = ∫ h(t) e^{-2πif_k t} dt  ≈  dt · Σ_m h(t_m) e^{-2πikm/n}  =  dt · X_k.  (4)

This is the entire content of the finding: **h̃ = dt·X, hence |h̃|² = dt²·|X|²**. Substituting
into (1) with the code's df-integration (trapezoid over the physical `fs` axis):

    ⟨h₁|h₂⟩ = 4 Re ∫ (dt·X₁)(dt·X₂)*/S_n df = dt² · [ 4 Re ∫ X₁X₂*/S_n df ].          (5)

The bracketed quantity is what the code computes (`parameter_estimation.py:369-373`); the
prefactor dt² is absent. The factor 4 (one-sidedness × real part) is present and correct; the
issue is solely the DFT normalization.

**Dimensional check.** [X] = strain (a bare sum of samples). [X²/S_n · df] = strain²·Hz/(strain²/Hz)
= Hz² — the code's result carries units of 1/s² instead of being dimensionless. With dt²: 
[dt²·X²/S_n·df] = s²·Hz² = 1. Only the dt²-corrected expression is dimensionally a valid SNR².

### 1.3 Closed-form benchmark

For a monochromatic signal h(t) = A sin(2πf₀t) of duration T with S_n slowly varying near f₀,
Parseval gives ∫₀^∞|h̃|²df = ½∫h²dt = A²T/4, hence from (1):

    ⟨h|h⟩ = A²T / S_n(f₀).                                                          (6)

This is the textbook matched-filter result (e.g. Maggiore Vol. 1, §7.3) and is independent of any
discrete convention.

---

## 2. Empirical evidence (all reproducible from the repo)

### L1 — Monochromatic analytic test (machine precision, two frequencies)

Exact-bin sinusoids at f₀ = 3.05×10⁻⁴ Hz and 1.53×10⁻³ Hz, benchmark (6):

| f₀ [Hz] | code / ⟨h|h⟩_physical | 1/dt² |
|---|---|---|
| 3.0518e-04 | 0.01000000 | 0.01 |
| 1.5259e-03 | 0.01000000 | 0.01 |

Committed reproducer: `master_thesis_code_test/parameter_estimation/test_inner_product_magnitude.py`.

### L2 — FFT-free Parseval test (no Fourier convention anywhere on the reference side)

For **constant** S_n = S₀, (1) collapses to a pure time-domain expression via Parseval:

    ⟨h|h⟩ = (2/S₀) ∫ h(t)² dt  =  (2/S₀) Σ_m h(t_m)² dt.                             (7)

The right-hand side involves **no FFT and no frequency axis at all** — no convention to dispute.
Test: 60-component band-limited random signal, S₀ = 10⁻³⁶, PSD patched constant:

    code = 2.593158e-01,  Parseval reference = 2.593159e+01,  ratio = 0.010000.

If one distrusts every FFT-normalization argument, this single test is decisive.

### L3 — Broadband test with independently normalized FFT (colored LISA PSD)

Same broadband signal against the real (colored) PSD; reference computed from scratch as
4∫|dt·rfft(h)|²/S_n df with an independent trapezoid:

    code = 1.590186e+04,  reference = 1.590186e+06,  ratio = 0.010000.

So the result is not special to monochromatic signals or to a white spectrum.

### L4 — The convention used by this pipeline's own dependency stack

`lisatools` (LISA Analysis Tools, installed in this venv as a dependency of the waveform stack)
implements exactly Eq. (1) and applies the dt factor at the transform:

    .venv/.../lisatools/datacontainer.py:111:
        tmp = xp.fft.rfft(self.data_res_arr, axis=-1) * self._dt

with `inner_product` (diagnostic.py:24) documented as 2∫(ã*b̃ + ãb̃*)/S_n df ≡ Eq. (1). The
same `dt * rfft` idiom appears throughout FEW/LISA community SNR tutorials. Our implementation
differs from the reference stack by exactly the missing dt.

### L5 — Astrophysical consistency (the "wouldn't we have noticed?" test)

The pipeline is *internally* self-consistent (threshold, p_det, and CRBs all share the same inner
product), so nothing breaks internally — the only external anchor is the detection horizon. That
anchor fails loudly:

- The repo's own extracted reference data from **Babak et al. 2017 (arXiv:1703.09722), the M1
  model this pipeline cross-checks** (`master_thesis_code/M1_model_extracted_data/
  detection_horizon.py`, SNR threshold 20): z_horizon ≈ 1.55 at M = 10⁵ M☉, peaking at
  **z ≈ 3.8** near M ≈ 1.4×10⁶ M☉.
- The rate model samples events out to z = 1.5 accordingly (`cosmological_model.py:185`).
- The pipeline's actual detected population (seed600, 500 events, nominal SNR ≥ 20):
  **median z = 0.046, p90 = 0.074, maximum z = 0.109** (d_L ≤ 0.485 Gpc).

A detected population confined to z ≤ 0.11 under a model whose SNR-20 horizon is z ≈ 1.5–3.8 is
exactly the signature of an SNR scale ~10× too small (horizon distance scales ≈ linearly with
SNR at fixed threshold). Laghi et al. 2021 use SNR > 100 as their *conservative gold* cut — our
"20" has behaved like their >200.

---

## 3. Anticipated objections

1. **"Maybe the PSD convention absorbs the dt²."** No: S_n is an analytic function of physical
   frequency (Babak et al. 2023 formulas, 1/Hz), with no reference to sampling. And L2 removes
   the PSD from the argument entirely.
2. **"Maybe FEW/fastlisaresponse output units compensate."** The inner product acts on the
   time-domain TDI series *after* generation; Eq. (4) is a property of the DFT, independent of
   what produced the samples. Any waveform-units question is a separate (multiplicative,
   dt-independent) issue — it cannot produce an exact 1/dt².
3. **"Trapezoid vs sum, one-sided doubling, DC-bin handling?"** All present and correct (the
   factor 4, band cropping, DC skip); they are dt-independent and the measured ratio is exactly
   1/dt² = 0.01, not 2 or 4 or 1/2.
4. **"Why did nothing downstream break?"** Because every consumer (SNR threshold, Fisher, p_det
   horizons, CRB filtering) uses the same mis-scaled product — a global rescaling is invisible
   internally. It is only wrong as a *physical* statement: "SNR ≥ 20" events are physically
   SNR ≥ 200, and the claimed d_L precisions are those of an SNR-20 event when the event is
   physically SNR-200 (10× better).

## 4. The fix (physics-change item, approved-pending-this-document)

    OLD (parameter_estimation.py:372-373):
        result = 4.0 * float(trapezoid(integrant.sum(axis=0).real, x=fs_crop))
    NEW:
        result = 4.0 * self.dt**2 * float(trapezoid(integrant.sum(axis=0).real, x=fs_crop))

(equivalently scale each rfft by dt; the single prefactor is cheaper). Reference comment: Finn
1992 Eq. 2.3 + DFT correspondence h̃ = dt·X. Post-fix, the L1–L3 ratios must read 1.000000 and
the regression anchor in `test_inner_product_magnitude.py` flips from `physical/dt²` to
`physical`.

## 5. Consequences (why this must land before the Phase-2 campaign)

- **SNR semantics become physical**: at threshold 20 the detected population deepens from
  z ≤ 0.11 toward the model's z ≲ 1.5 range → far more events, longer/heavier waveform work per
  event (interacts with the G9 timeout budget), GLADE completeness at higher z becomes the
  binding constraint (the photo-z systematics story becomes *more* central, not less).
- **CRBs shrink 10×** in σ: per-event d_L precision improves; the σ(d_L) distribution feeding the
  dark-siren inference changes qualitatively.
- All existing CRBs/injections are already RETIRED (mass convention); no additional data loss.
- Thesis-era absolute numbers (rates, precisions) are superseded; the relative/mechanism results
  (rail anatomy, calibration) are unaffected in kind.
