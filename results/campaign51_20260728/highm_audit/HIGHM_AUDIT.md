# High-M detectability collapse audit — campaign #51 (2026-07-28)

**Question.** Is the measured collapse of EMRI detectability above detector-frame
M_z ≈ 10^6.2 (pilot, 6,000 rows, `PILOT_READOUT.md`) physical, or an artifact of
model components built/validated only for M ≤ 10^6?

**Answer in one line.** It is BOTH an implementation artifact and a convention
choice: (i) a **units-inconsistent galactic-confusion term in the PSD**
(MEASURED, 53–1100× SNR suppression exactly over the collapse band) and (ii)
the **mass-independent snapshot initial-condition convention** (p0 ~ U[10,16]),
which contradicts the plunge-time convention of the Babak et al. (2017) rate
model the population is drawn from. The measured m ≈ 6.2 wall does NOT stand
as a physical statement; the running bulk campaign inherits both effects.

Evidence tags: **MEASURED** (run in this audit or read from pilot CSVs),
**ESTIMATED** (analytic/order-of-magnitude), **LITERATURE** (cited source).
Scripts/outputs in this directory: `audit_calcs.py` (+`audit_calcs_output.json`),
`measure_few_snr.py` (+`few_snr_measurement.json`),
`measure_sens_snr.py` (+`sens_snr_measurement.json`),
`measure_pipeline_snr.py` (unusable on this host: fastlisaresponse SIGILL, kept
for cluster re-use).

---

## Verdict table

| # | Link | Verdict | One-line basis |
|---|------|---------|----------------|
| 1 | Initial-condition convention (p0 ~ U[10,16] snapshot) | **CONVENTION-DEPENDENT, contradicts the rate model** | Babak 2017 draws plunge times U[0,2] yr; this pipeline draws snapshot orbits; Peters time t_insp(p0=10)=5 yr at log M_z = 6.145 — the measured last detection is m = 6.143 (MEASURED coincidence to 0.002 dex) |
| 2 | Sampling rate / Nyquist | **NOT AN ISSUE** | SNR path uses `ParameterEstimation.dt = 10 s` (f_Nyq = 0.05 Hz); `constants.LISA_STEPS`/`LISA_DT` are dead constants (grep: no consumer) |
| 3 | Inner-product frequency limits | **NOT AN ISSUE** | Band [1e-5, 1] Hz; MEASURED power fraction below f_min ≤ 6.8e-6 in the worst case (M_z = 10^7.4, p0 = 16) |
| 4 | PSD validity at low f (confusion noise) | **ARTIFACT (units inconsistency)** | Raw strain-referred Cornish–Robson S_c added to the TDI-A relative-frequency PSD without the 1.5·4x²sin²x transfer (lisatools `A1TDISens.stochastic_transform`); overweights confusion by ~2×10⁶ at 0.3 mHz; MEASURED SNR suppression 53×/178×/409×/1097× at m = 6.2/6.4/6.6/7.0 |
| 5 | dN/dz coefficient set [4] | **SOUND over campaign range** (provenance ceiling ~10^6.5) | MEASURED positive over z ∈ (0, 1.5] at all masses (min 5.9e4); affects draw z-shape only, not per-event detectability |
| 6 | R_emri branches | **SOUND (smooth power-law fit)** | 2.9e7 is a normalization pivot, not a branch edge; R(1e7) = 18.7/yr, positive/finite; digitized Babak 2017 Fig. 1 lineage |
| 7 | Other mass gates in the SNR path | **NONE ACTIVE at high M** | few 2.0 warns only for mu/M > 1e-4 (LOW-mass side) via logger (not `warnings` — the main.py "Mass ratio" catch cannot fire under few 2.0); no lower-q guard; no FEW failure wall observed to M_z = 10^7.37 (pilot) |

---

## Item 4 (taken first — it dominates): confusion-noise units artifact

### The inconsistency (code inspection, file:line)

`LISA_configuration.py:139-141` adds `_confusion_noise(f)` (the Cornish &
Robson 2017 / Robson et al. 2019 galactic-foreground fit, coefficients in
`constants.py:165-171`) **directly** to `power_spectral_density_a_channel`.
The instrumental part (lines 125-138) is the standard **TDI-1 A-channel PSD in
relative-frequency units** — `8 sin²(2πfL/c)[S_OMS(cos+2) + 2(3+2cos+cos2)S_TM]`,
identical in form to lisatools `A1TDISens.transform` (relative_frequency
asserted). The Cornish–Robson S_c, however, is a **sky-averaged
strain-sensitivity** quantity. LITERATURE: lisatools
(`.venv/.../lisatools/sensitivity.py`, `A1TDISens.stochastic_transform`)
converts a stochastic strain PSD to the A-channel TDI contribution as

```
S_c^TDI-A(f) = 1.5 · [4x² sin²x] · S_c(f),   x = 2πfL/c.
```

The code omits this transfer factor. Since x ≪ 1 below a few mHz, the omission
overweights confusion by 1/(1.5·4x²sin²x):

| f [Hz] | code S_c/S_instr | corrected S_c/S_instr | SNR suppression √(S_code/S_corr) (MEASURED, analytic PSD) |
|---|---|---|---|
| 1e-4 | 2.1e6 | 0.010 | 1450 |
| 2e-4 | 1.5e6 | 0.099 | 1117 |
| 4e-4 | ~7e5 | 0.70 | 597 |
| 8e-4 | ~3e5 | 2.6 | 198 |
| 1.5e-3| ~1e5 | 3.9 | 59 |
| 2e-3 | ~4e4 | 2.5 | 32 |
| 3e-3 | 0.35 (code) | 0.007 | 1.7 |
| ≥5e-3 | ~0 | ~0 | 1.0 |

The corrected form is physically sane: confusion peaks at ~4× instrumental in
the 0.8–2.5 mHz band and is negligible elsewhere — the expected LISA behaviour.
The code's form makes the detector ~10³× deaf (amplitude) at 0.2–1 mHz.

Provenance: commit `3bed9fc` ([PHYSICS] Phase 9, 2026-03-29) added S_c with unit
tests for positivity/shape but **no TDI-vs-sensitivity unit check**; the
`CLAUDE.md` "FIXED Phase 9" note inherited that gap.

### Direct measurement with production waveforms (MEASURED)

`measure_few_snr.py`: real Pn5AAK waveforms (few 2.0, T=5 yr, dt=10 s, μ=10,
a=0.98, e0=0.1, x0=0.9, the pipeline's parameters), same waveform scored under
the production PSD vs the transfer-corrected PSD (in-memory only — no source
edits):

| case | signal power (5–95 %) [Hz] | SNR suppression code→corrected |
|---|---|---|
| m5.5 p0=10 | 6.3–10.9 mHz | **1.000** |
| m6.0 p0=10 | 2.1–5.1 mHz | **1.71** |
| m6.2 p0=10 | 1.3–2.9 mHz | **52.9** |
| m6.4 p0=10 | 0.79–1.2 mHz | **178** |
| m6.6 p0=10 | 0.50–0.71 mHz | **409** |
| m6.6 p0=13 | 0.33–0.48 mHz | **705** |
| m7.0 p0=10 | 0.19–0.27 mHz | **1097** |

The artifact switches on precisely across the measured wall (1.7× at m6.0 →
53× at m6.2) because that is where the snapshot-orbit signal power crosses
below ~3 mHz.

### Independent cross-check against the pilot (MEASURED)

Predicted production/strain-referred SNR ratio at the m6.4–6.6 fundamentals
(√(1.5·4x²sin²x) ≈ 6.7e-4 at 0.79 mHz) times the strain-referred estimate
(≈5.6–6.0 at 1 Gpc) gives ≈ 3.8e-3 — the pilot's measured median SNR@1 Gpc in
the 6.4–6.6 bin is 3.4e-3 (agreement to ~10 %). The pilot data themselves
carry the artifact's signature.

### Calibrated corrected horizons (MEASURED ratio × validated estimator)

`measure_sens_snr.py` scores the same waveforms against the strain-referred
effective sensitivity S_eff = S_instr/R + S_c (R = 1.5·4x²sin²x). Validation:
where the artifact is inactive this estimator reproduces the pilot's actual
detected plateau — m5.5: d_hor = 2.6 Gpc, m6.0 p0=10: d_hor = 5.0 Gpc vs pilot
measured max d_hor 5.48/5.30 Gpc in the 5.8–6.0/6.0–6.2 bins.

| case | corrected SNR@1 Gpc | corrected d_hor [Gpc] (z at h=0.73) | pilot MEASURED max d_hor [Gpc] | ratio (dec) |
|---|---|---|---|---|
| m5.5 p0=10 | 51.9 | 2.6 (z≈0.47) | ~5 (plateau, incl. better draws) | — (artifact off) |
| m6.0 p0=10 | 99.8 | 5.0 (z≈0.80) | 5.30–5.48 | ~1.0 (0.0 dec) ✓ |
| m6.2 p0=10 | 16.2 | 0.81 (z≈0.17) | 0.202 (bin 6.2–6.4) | 4.0 (0.6 dec)* |
| m6.4 p0=10 | 5.1 | 0.25 (z≈0.06) | 2.5e-3 (bin 6.4–6.6) | 102 (2.0 dec) |
| m6.6 p0=10 | 2.2 | 0.108 | 2.5e-4 (bin 6.6–6.8) | **432 (2.6 dec)** — matches the independently measured waveform suppression 409 |
| m6.6 p0=13 | 0.78 | 0.039 | (same bin, median draws) | ≈705 |
| m7.0 p0=10 | 0.29 | 0.014 | 9.1e-5 (bin 6.8–7.0) / 2.2e-5 (7.0–7.4) | ~650–1100 (2.8–3.0 dec) |
| m7.4 p0=16 | 0.081 | 0.004 | 3.0e-6 (p90, 7.0–7.4) | ~10³ |

\* p0=10 is the bin's best-case draw; bin maxima mix p0 ∈ [10,16], so single-case
ratios at the wall edge are lower bounds on the artifact factor there.

Two consequences of the corrected column (ESTIMATED absolutes, MEASURED ratios):

1. Even keeping the snapshot convention, a corrected PSD moves the "last
   detection" boundary from m ≈ 6.14 to m ≈ 6.5–6.6 (corrected d_hor ≥ 0.1 Gpc,
   i.e. detectable for z ≲ 0.02–0.06 hosts, which the z-draw does populate) and
   raises survival in the 6.2–6.6 band from 0 to small-but-nonzero. The pilot's
   pre-registered narrowing rule would NOT have scored the same on corrected
   data.
2. The artifact also bites INSIDE the old band: any draw whose power sits at
   0.5–3 mHz is suppressed — e.g. M_z ≈ 6e5–1.4e6 with p0 = 13–16 (fundamentals
   0.7–1.7 mHz, suppression ~10–50×). The existing prodstack injection pools and
   the CRB event selection near the top of the old band carry this too.

---

## Item 1: initial-condition convention (snapshot vs plunge-window)

### (a) What the literature model does (LITERATURE, verified from the paper text)

Babak et al. (2017), arXiv:1703.09722 (PDF fetched, §III C/D):

- Catalog construction: events sampled from d³N/(dM dz da)·p0(M,z)R(M,a); then
  *"Plunge times are taken to be uniform in [0, 2] yr. We ignore events that
  plunge after the end of the mission duration, although they might be
  detectable if they are close enough."* — a **plunge-window convention**; the
  quoted rates are plunge rates and phases are specified **at plunge**.
- Eccentricity: *"a rather flat eccentricity distribution at plunge in the
  range 0 < ep < 0.2"* — the [0.05, 0.2] band this pipeline uses for e0 is
  Babak's eccentricity **at plunge**, applied here **at p0 = 10–16** instead
  (a second, milder convention transplant).
- High-mass detectability is a *plunge-phase* phenomenon: *"for such high mass
  MBHs a prograde inspiral generates a significant number of waveform cycles
  between the Schwarzschild ISCO frequency and the final plunge, and these
  cycles are at frequencies in the most sensitive range for the LISA
  detector"* — AKK waveforms detect EMRIs *"up to 10⁷ M☉ when MBHs are
  spinning"* (M1 has a = 0.98, as does this pipeline).

This pipeline instead draws p0 ~ U[10,16] for every mass (`parameter_space.py:95-103`)
— a mass-independent snapshot convention — while weighting the draw by the
Babak **plunge rate** R_emri(M) (`cosmological_model.py:249-290`). The mock
population is therefore internally inconsistent with its own rate model at any
mass where t_insp(p0) ≫ T_obs.

### (b) Quantification (ESTIMATED, Peters leading order + Kerr geodesics; a=0.98, μ=10)

f_GW = 2f_orb (circular, equatorial Kerr, r = p·M), Kerr ISCO r = 1.614M:

| M_z | p0 | f_GW(p0) [Hz] | f_GW@ISCO [Hz] | t_insp(p0→plunge) [yr] | √(S_code(f0)/S_code(f_ISCO)) |
|---|---|---|---|---|---|
| 1e6 | 10 | 2.0e-3 | 2.1e-2 | 2.6 | 2.1 |
| 1e6 | 16 | 9.9e-4 | 2.1e-2 | 19.5 | 9.7 |
| 3e6 | 10 | 6.6e-4 | 7.1e-3 | 23 | 119 |
| 1e7 | 10 | 2.0e-4 | 2.1e-3 | 257 | 51 |
| 1e7 | 16 | 9.9e-5 | 2.1e-3 | 1950 | 115 |
| 2.5e7 | 10 | 7.9e-5 | 8.5e-4 | 1605 | 20 |

- p0 required to plunge within T = 5 yr: 11.6 (M_z=1e6), 9.5 (10^6.2), 7.5
  (10^6.5), 6.5 (1e7) — for m ≳ 6.2 **no draw in [10,16] can plunge within the
  mission**, so the entire high-mass population is frozen at its lowest,
  quietest frequencies.
- Collapse-edge prediction: solving t_insp(p0=10 → p=6.3) = 5 yr gives
  **M_z = 1.40e6, log₁₀ = 6.145**. The pilot's measured last detection is at
  **m = 6.143**. The wall's *location* is the snapshot convention's plunge
  boundary (with the item-4 artifact then annihilating everything beyond it —
  the two mechanisms are colocated because "cannot reach plunge" ⇒ "power
  stays below ~3 mHz" ⇒ "hit by the confusion artifact").

### (c) Provenance of p0 ∈ [10, 16] (git/docs archaeology)

- Introduced in the very first parameter-space commits (`abbb13f`/`6068dee`,
  2023-09/2023-10, "UNFINISHED: parameter estimation schwarzschild") — i.e.
  **before** any population-band decision, and identical to few's documented
  Pn5AAK **input-validity** domain (`few/utils/baseclasses.py` docstring:
  "p0: 10 ≤ p0 ≤ 16 + 2e0"). It is a waveform-tool input domain adopted as an
  astrophysical prior.
- Never revisited: `docs/campaign_redesign_51_design.md` (the #51 band-widening
  design) contains **zero** mentions of p0 or plunge; no .planning/gate document
  discusses the initial-condition convention. The [10,16] range was
  benign for the old [10^4.5, 1e6] source band (t_insp(p0=10) ≤ 5 yr for
  M_z ≲ 1.4e6 covers most of it) and becomes the binding constraint exactly
  when the band is widened — an implicit "validated only for M ≤ 10^6"
  assumption, as suspected.

**Verdict: CONVENTION-DEPENDENT** — but not a defensible free choice: the draw
weights events by a plunge rate while the waveform convention prevents
plunge-band emission for m > 6.2. Under the documented population convention of
the model being cross-checked (Babak M1, plunge-window), high-mass EMRIs are
detectable to ~10⁷ M☉ (LITERATURE, AKK a=0.98). Note honestly: PN5-AAK cannot
*start* below p0 = 10, and Peters says plunge-in-5-yr requires p0 ≈ 6.5–9.5 for
m > 6.2 — implementing the plunge-window convention needs `few`'s
`get_p_at_t` (present in few 2.0, `few/utils/utility.py:1759`) and acceptance
of AAK inaccuracy near plunge (Babak 2017 flags AKK as SNR-optimistic there;
AKS/AKK bracket the truth).

---

## Item 2: sampling rate / Nyquist — NOT AN ISSUE (MEASURED by code inspection)

The SNR path never touches `constants.LISA_STEPS`/`LISA_DT` (grep: no consumer
outside `constants.py` — dead constants). `ParameterEstimation.dt = 10 s`
(`parameter_estimation.py:88`), threaded into `ResponseWrapper` via
`create_lisa_response_generator(..., self.dt, self.T)` → f_Nyq = 0.05 Hz,
T = 5 yr (1.58e7 samples; pilot waveform shapes confirm). All high-M_z power
(1e-4–1e-3 Hz) is orders of magnitude inside the sampled band; low-M ISCO
frequencies (≤ 2e-2 Hz at M_z = 1e5... strictly f_ISCO,gw = 0.021·(1e6/M_z) Hz)
also fit for M_z ≥ ~5e4. No aliasing mechanism at high M.

## Item 3: inner-product band — NOT AN ISSUE (MEASURED)

`scalar_product_of_functions` restricts to [MINIMAL_FREQUENCY, MAXIMAL_FREQUENCY]
= [1e-5, 1] Hz (`constants.py:53-54`, applied in `_get_cached_psd`). MEASURED
fraction of |h̃|² below 1e-5 Hz with real waveforms: 1.7e-9 (m5.5), 3.0e-7
(m7.0 p0=10), 6.8e-6 (worst case m7.4 p0=16, fundamental 4.0e-5 Hz). Zero
decades of the collapse attributable to the band edges.

## Item 5: dN/dz coefficient set [4] — SOUND over campaign range

`merger_distribution_coefficients` (`cosmological_model.py:67-123`) are
9th-order polynomial fits (no constant term) digitized from the Babak 2017 M1
dN/dz curves per half-dex mass bin (lineage: commit `8fbceda`, 2024-03-07; the
companion grid `M1_model_extracted_data/emri_distribution.py` spans
log M ∈ [4, 7]). The ≥6.0 branch blends set [3]→[4] over m ∈ [6.0, 6.5] and is
**constant in mass above 6.5** (fraction capped at 1) — so the provenance
ceiling of a *mass-resolved* dN/dz is ~10^6.5; above that the z-shape is an
extrapolation-by-freeze. MEASURED: set [4] (and all blends at m = 6.0/6.25/6.5/7.0)
positive and finite over z ∈ (0, 1.5]: min 5.9e4, max 8.3e8, negative fraction
0.000. The emcee guard (`_log_probability`, non-positive → -inf) additionally
protects the draw. Affects the DRAW z-distribution at high M only; no
contribution to the per-event detectability collapse.

## Item 6: R_emri branches — SOUND

`R_emri` (`cosmological_model.py:283-290`): three power-law segments with knees
at 1.2e5 and 2.5e5; "2.9e7" in the third branch is the **normalization pivot**
(log10(M/2.9e7), R = 14.4/yr there), not a branch boundary — the third segment
covers all M ≥ 2.5e5 with slope −0.2475. Lineage: piecewise fit to the Babak
2017 Fig. 1 / Eq. (5)-(23) M1 rate (the analytic form lives in `emri_rate.py`
with per-equation citations). MEASURED values: R(1e6) = 33.1, R(1e7) = 18.7,
R(2.9e7) = 14.4 events/yr — smooth, positive, no pathology to 1e7. The Babak
M1 mass function itself is specified over the drawn band; rate-weighting at
1e7 is a mild extrapolation of a fitted power law, consistent with the paper's
Fig. 1 declining trend.

## Item 7: other mass gates — none active at high M

- few 2.0 (`few/utils/baseclasses.py:746-751`) warns when **mu/M > 1e-4**
  (low-mass side only; with μ=10 that is M_z < 1e5). It uses
  `get_logger().warning`, NOT `warnings.warn` — so the injection loop's
  `warnings.simplefilter("error")` + "Mass ratio" catch (`main.py:1105`) can
  never fire under few 2.0 (dead handler; low-M events are NOT skipped —
  consistent with the pilot's populated m < 5 bins).
- No lower bound on the mass ratio exists anywhere in the path: ε = 4e-7
  (μ=10, M_z=2.5e7) enters PN5-AAK unguarded. LITERATURE: AAK is a kludge
  whose adiabatic assumption *improves* with smaller ε; no known failure wall,
  consistent with the pilot's zero-failure measurement to M_z = 10^7.37.
- No other mass-dependent guard/cut in `compute_signal_to_noise_ratio`,
  `waveform_generator.py`, or the injection loop (grep audit).

---

## Decades attribution of the measured 5-decade d_hor collapse (m 6.2 → 6.8+)

Measured collapse: max d_hor 5.3–5.5 Gpc (6.0–6.2) → 9.1e-5 Gpc (6.8–7.0):
**≈ 4.8 decades over 0.7 dex.** Attribution:

| Mechanism | Decades at the deep end (m ≈ 6.8–7.0) | Tag |
|---|---|---|
| Confusion-noise units artifact (item 4) | **≈ 2.8–3.0** (MEASURED suppression 650–1100×) | ARTIFACT |
| Snapshot convention + genuine low-f PSD/amplitude physics (item 1 + real detector physics, inseparable at fixed convention: corrected snapshot d_hor falls 5.0 → 0.014 Gpc over m 6.0→7.0) | **≈ 2.5** | CONVENTION + PHYSICAL |
| Hard cutoffs (Nyquist, f_min/f_max), waveform validity walls, rate/dN-dz model edges | **0.0** (MEASURED) | — |

Of the ≈2.5 convention+physics decades, the plunge-window convention of the
underlying Babak M1 model would recover most: LITERATURE (Babak et al. 2017,
AKK, a = 0.98) finds detectable EMRIs to 10⁷ M☉ because the plunge-phase
cycles sit in the LISA bucket; the pipeline's own corrected f_ISCO ratios
(item 1b table: √(S(f0)/S(f_ISCO)) up to ~300) plus the h ∝ M amplitude growth
imply Gpc-scale corrected plunge-window horizons up to m ≈ 6.8–7.0 (ESTIMATED;
a Newtonian AKK-style chirp integral gives the same order). The wall's
*location* at m = 6.143 is quantitatively the snapshot-plunge boundary
t_insp(p0=10) = T_obs (predicted 6.145), with the artifact then manufacturing
the 5-decade *depth* of the cliff behind it.

---

## Bottom line

**The measured m ≈ 6.2 collapse does NOT stand as the correct object, even
under the pipeline's own documented population convention.** Two independent
problems, in order of severity:

1. **PSD confusion-noise units artifact (implementation bug, physics-change
   required).** `LISA_configuration.power_spectral_density_a_channel` adds the
   strain-referred Cornish–Robson S_c to the TDI-A relative-frequency PSD
   without the `1.5·4x²sin²x` stochastic transfer (lisatools
   `A1TDISens.stochastic_transform`). MEASURED with production waveforms: SNR
   suppressed 53×/178×/409×/1097× at m = 6.2/6.4/6.6/7.0 (and 1.7× already at
   m = 6.0; ~10–50× for p0 ≳ 13 draws at M_z ≈ 6e5–1.4e6 *inside the old band*).
   The pilot's own numbers carry the predicted signature to ~10 %.
   *Required change:* one line — multiply `_confusion_noise` by
   `1.5·(2x sin x)²` (x = 2πfL/c) before adding, per lisatools/LDC convention;
   `/physics-change` protocol with a lisatools cross-check as the regression
   test. *Cost:* invalidates the SNR of every existing injection row and every
   CRB whose signal power sits below ~3 mHz — the #51 pilot + bulk, and the
   upper-mass corner of the prodstack pools. Fisher matrices of well-detected
   low-mass events (power > 3 mHz) are ~unchanged (suppression ≤ 1.02 at
   m ≤ 5.5, 1.7× at m6.0-p10-class events — those CRBs DO shift).

2. **Snapshot initial-condition convention (pre-registered convention gap).**
   p0 ~ U[10,16] at every mass is a 2023 waveform-input-domain choice
   (few Pn5AAK docstring), never ratified as an astrophysical prior and never
   revisited when #51 widened the band; it contradicts the plunge-rate
   semantics of R_emri (Babak 2017: plunge times uniform in [0,2] yr,
   eccentricity specified AT PLUNGE — the pipeline's e0 band [0.05,0.2] is
   Babak's *plunge* eccentricity applied at p0 = 10–16). Peters time
   t_insp(p0=10) = T_obs at log M_z = 6.145 vs measured last detection 6.143.
   Under the model's own convention (plunge-window), heavy EMRIs are
   detectable to ~10⁷ M☉ (Babak AKK, a = 0.98). *Required change (if the
   author adopts the literature convention):* draw t_plunge ~ U[0, T+margin]
   and set p0 = `few.utils.utility.get_p_at_t` (available in few 2.0);
   accept AAK late-inspiral inaccuracy (AKS/AKK bracket — a systematics-budget
   line, not a blocker). *Cost:* full injection-pool + CRB regeneration and a
   p_det-grid re-derivation; high-mass strata become genuinely informative
   instead of measured-zero. Alternatively, RETAIN the snapshot convention as
   an explicit, documented population assumption — but then the paper cannot
   cite Babak M1 rates as the population being simulated at m > 6.2, and the
   "measured ~zero survival above m ≈ 6.2" claim (PILOT_READOUT §Campaign
   consequences) must be re-labelled as convention-conditional.

**Urgent (campaign-pausing):** The in-flight bulk injection arrays (job
6070769 + follow-up, ≈194k rows) are being generated with the artifact PSD of
item 1. Their high-m rows (the entire point of the #51 widening) measure the
artifact, not the detector; rows with power below ~3 mHz are suppressed up to
~10³×. Fixing the PSD afterwards CANNOT rescale stored SNRs (the correction is
frequency-dependent per event). Recommend pausing/cancelling the bulk arrays
until the confusion-noise transfer fix lands; the pilot's narrowing decision
and the "measured undetectable above m ≈ 6.2" statement should be quarantined
in the same breath.

*(Author decisions required: both changes are physics changes gated by
`/physics-change`; nothing in this audit modifies source files.)*
