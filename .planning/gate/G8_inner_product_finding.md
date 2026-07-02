# G8 — Inner-product magnitude finding (2026-07-02)

## Verdict: DEVIATION FOUND — missing dt² DFT normalization

`scalar_product_of_functions` (`parameter_estimation/parameter_estimation.py:317-373`) computes

    4 · Σ_channels ∫ X_a(f) X_b*(f) / S_n(f) df

with `X = numpy.fft.rfft(h)` (raw, un-normalized DFT) and `S_n` the physical one-sided PSD
(Babak et al. 2023, arXiv:2303.15929) on a physical frequency axis `rfftfreq(n, dt)` [Hz].
The continuous Fourier transform is `h̃(f_k) = dt · X_k`, so the standard inner product

    <h1|h2> = 4 Re Σ_α ∫ h̃_1 h̃_2* / S_n df        (Finn 1992; Babak et al. 2021 Eq. 20)

requires an explicit `dt²` factor that is absent.

## Empirical confirmation (machine precision)

Monochromatic exact-bin sinusoid, analytic reference `<h|h> = A²T/S_n(f0)`:

| f₀ [Hz] | code / physical | 1/dt² (dt=10 s) |
|---|---|---|
| 3.0518e-04 | 0.01000000 | 0.01 |
| 1.5259e-03 | 0.01000000 | 0.01 |

Reproducer: `master_thesis_code_test/parameter_estimation/test_inner_product_magnitude.py`
(the regression anchor pins `code == physical/dt²` pre-fix).

## Physical consequences (pre-fix pipeline semantics)

- **SNR:** code-SNR = physical-SNR / dt = physical/10. The `SNR_THRESHOLD = 20` catalogue is a
  **physical SNR ≥ 200** population (much closer/smaller than a true SNR-20 catalogue).
- **Fisher/CRB:** Fisher elements scale with the same 1/dt² → covariance 100× too large →
  σ(d_L)/d_L and all parameter uncertainties **10× too pessimistic** for the selected events.
- The two distortions are coupled but NOT mutually cancelling: the pipeline effectively simulates
  "physical-SNR-200 events carrying SNR-20-like fractional errors".
- All existing CRB datasets inherit this. They are already RETIRED for other reasons
  (mass convention); the Phase-2 campaign must run with the fix.
- Thesis-era numbers (e.g. N_det = 663/165k at thr 15) refer to physical SNR ≥ 150.

## Proposed fix (physics-change protocol — NEEDS USER APPROVAL)

1. **Old:** `result = 4.0 * float(trapezoid(integrant.sum(axis=0).real, x=fs_crop))`
   (`parameter_estimation.py:370-372`)
2. **New:** `result = 4.0 * self.dt**2 * float(trapezoid(...))` (equivalently scale each rfft by dt)
3. **Reference:** h̃(f) = dt·X_DFT (DFT↔CFT correspondence); inner product Finn 1992 /
   Babak et al. 2021 arXiv:2108.01167 Eq. (20); reference implementations (FEW/lisaanalysistools
   tutorials) use `dt * xp.fft.rfft(h)`.
4. **Dimensional analysis:** [X]=strain (sum of samples), [dt·X]=strain·s=strain/Hz;
   [|h̃|²/S_n]=(strain²/Hz²)/(1/Hz)=strain²·s... consistent: ∫df → dimensionless <h|h>. Without dt²
   the result carries 1/s² — dimensionally wrong.
5. **Limiting case:** monochromatic analytic value above; post-fix ratio must be 1.000000.

**Downstream after fix:** SNR semantics become physical; `SNR_THRESHOLD=20` then selects a genuinely
deeper/larger population (longer eval times, more timeouts — interacts with G9); PRE_SCREEN_SNR_FACTOR
and Fisher ε defaults should be re-checked at the new SNR scale. Fix goes in BEFORE the Phase-2
campaign; existing retired data unaffected (already retired).

## Also fixed en route (software)

- `np.trapz` removed in NumPy 2.x → CPU path of the inner product crashed (`AttributeError`);
  now resolves `trapezoid` with `trapz` fallback (CuPy). The only pre-existing scalar_product tests
  were `@pytest.mark.gpu`, which is why this went unnoticed. New CPU test covers it.
