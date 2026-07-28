# Plunge-window initial conditions (replaces the snapshot p0 ~ U[10, 16] draw)

**Status:** author-ratified 2026-07-28 (this session). [PHYSICS] change under the
physics-change protocol.
**Motivating audit:** `results/campaign51_20260728/highm_audit/HIGHM_AUDIT.md`
item 1 — the snapshot draw is few's Pn5AAK *input-validity* domain (2023
adoption) used as an astrophysical prior; it contradicts the plunge-rate
semantics of the Babak et al. (2017) M1 model the population is drawn from and
freezes every M_z ≳ 10^6.2 event outside its detectable plunge phase.
**Measurements for this doc:**
`results/campaign51_20260728/plunge_window/measure_plunge_window.py`
(+`plunge_window_measurements.json`) and `measure_snr_seff.py`
(+`snr_seff_measurements.json`). Evidence tags: **MEASURED** (run here),
**ESTIMATED** (analytic/order-of-magnitude), **ASSUMED** (convention choice),
**LITERATURE** (cited source).

---

## 1. Convention statement

**An "event" is an EMRI that plunges during the observed mission span.**
Precisely: the source is observed from t = 0 for the full observation duration
T = `LISA_MISSION_DURATION_YEARS`; its plunge time is drawn

```
t_plunge ~ U[0, T]        (observer/detector-frame years)
```

and the initial semi-latus rectum p0 at t = 0 is fixed by the time-to-separatrix
condition on the *same* PN5 trajectory that generates the Pn5AAK waveform:

```
t_insp(p0; M_z, mu, a, e0, x0) = t_plunge
```

solved by Brent root-finding (`few.utils.utility.get_p_at_t`, few 2.0). The
signal therefore inspirals for t_plunge and plunges *inside* the window; the
remaining T − t_plunge is silence (zero-padded, §6).

- LITERATURE (convention source): Babak et al. (2017), arXiv:1703.09722,
  §III C/D — *"Plunge times are taken to be uniform in [0, 2] yr. We ignore
  events that plunge after the end of the mission duration…"*. We adopt the
  same construction with the window = the full observation span T.
- LITERATURE (window length): Colpi et al. (2024), LISA Definition Study
  Report, arXiv:2402.07571 — nominal mission duration **4.5 yr of science
  operations**. The quoted duty cycle (> 82 %) is NOT modeled (no gap
  modeling); it is a tracked systematic (→ §8). This simultaneously retires the
  pipeline's unofficial T = 5 yr (`ParameterEstimation.T`, hardcoded 2023) and
  the inconsistent confusion-noise default `t_obs_years = 4.0`.
- ASSUMED: t_plunge is independent of all other parameters (Babak's uniform,
  independent plunge times). Events plunging after T are excluded from the
  population, exactly as in Babak 2017 ("might be detectable if they are close
  enough" — a small conservative truncation, §8).

**Consistency with the rate model.** `R_emri(M)` (cosmological_model.py) is the
Babak M1 *plunge* rate. Weighting the draw by a plunge rate while the initial
conditions forbid in-window plunges for m > 6.2 was the internal inconsistency
(audit item 1); this convention removes it: every drawn event realizes exactly
one plunge in the window the rate counts.

## 2. Physics-change protocol items

1. **Old formula** (`datamodels/parameter_space.py:95-103` pre-change;
   `randomize_parameters`): `p0 ~ U[10, 16]`, independent of all other
   parameters; observation T = 5 yr
   (`parameter_estimation/parameter_estimation.py:89`); confusion
   `t_obs_years = 4.0` (`LISA_configuration.py:67`).
2. **New formula:** `t_plunge ~ U[0, T]`, `p0 : t_insp(p0) = t_plunge` on the
   PN5 trajectory (`plunge_window.py`); `T = t_obs_years =
   LISA_MISSION_DURATION_YEARS = 4.5`.
3. **References:** Babak et al. (2017) arXiv:1703.09722 §III C/D (plunge-window
   population, e at plunge); Colpi et al. (2024) arXiv:2402.07571 (4.5 yr);
   Peters (1964) Phys. Rev. 136, B1224, Eq. (5.10) (upper-bracket seed);
   Stein & Warburton (2020) arXiv:1912.07609 (Kerr separatrix, via few's
   `get_separatrix`).
4. **Dimensional analysis:** §5. p is dimensionless (units of G M_z/c²);
   t_plunge [yr] × `YRSID_SI` [s/yr] → s; `get_p_at_t` compares trajectory time
   [s] to t_plunge·YRSID_SI [s]. Peters seed:
   (256/5)·t_pl[s] / (M_z·MTSUN_SI [s]) · (μ/M_z) is dimensionless, ^(1/4) → p. ✓
5. **Limiting cases:** §5.

## 3. Frames and (1 + z) bookkeeping (verified)

few takes **detector-frame** masses: the pipeline sets
`parameter_space.M.value = M_source·(1+z)` (both loops; see
`set_host_galaxy_parameters` and the injection-loop `M.value = redshifted_M`
assignments, each with the Maggiore 2008 §4.1.4 reference). Because the PN5
trajectory clock for detector-frame masses **is observer time**, feeding M_z
(and μ, uncorrected — the (1+z) lift of μ is a known, separate approximation of
this pipeline, unchanged here) makes `t_insp` directly comparable to the
observer-frame window T. No additional (1+z) factor appears anywhere in the
draw: **the window is defined in the detector frame**, matching Babak's
construction (plunge times uniform within the *observed* window). The
source-frame image of the window is T/(1+z) — irrelevant to the draw, which
never leaves the detector frame.

## 4. Eccentricity semantics (transplant retained, bound MEASURED)

LITERATURE: Babak 2017 specifies eccentricity **at plunge**: "a rather flat
eccentricity distribution at plunge in the range 0 < e_p < 0.2". This pipeline
draws e0 ∈ [0.05, 0.2] **at t = 0** (Model1CrossCheck caps e0 at 0.2).

Exact realization of Babak's convention would require a coupled 2-D root-find
(p0, e0) with e(t_plunge) = e_p drawn — feasible but ~doubles the draw cost and
couples two brentq loops. **Adopted realization (ASSUMED, bounded):** keep
drawing e0 ∈ [0.05, 0.2] at t = 0 and document the transplant.

MEASURED bound (PN5 trajectory, a = 0.98, x0 = 0.9, μ = 10;
`plunge_window_measurements.json` e_transplant block): eccentricity decays
monotonically along every case; realized plunge eccentricity vs e0 = 0.2 at
start:

| M_z | t_plunge [yr] | p0 | e_p (PN5) | e_p (Peters small-e map) |
|---|---|---|---|---|
| 1e5 | 4.5 | 34.18 | 0.008 | 0.002 |
| 1e6 | 4.5 | 10.81 | 0.051 | 0.012 |
| 3e6 | 4.5 | 6.81 | 0.102 | 0.026 |
| 1e7 | 4.5 | 4.71 | 0.154 | 0.048 |
| 1e7 | 0.5 | 3.60 | 0.181 | 0.074 |

(e0 = 0.05 rows scale proportionally.) Consequences, stated honestly:

- Realized e_p always satisfies **0 < e_p ≤ e0 ≤ 0.2** — inside Babak's stated
  plunge-band support, but **not flat**: the transplant compresses e_p downward,
  strongly at low M_z (e_p ≈ 0.04·e0 at M_z = 1e5) and mildly at high M_z
  (e_p ≈ 0.77–0.9·e0 at 1e7). Max transplant error |e0 − e_p| = 0.192
  (M_z = 1e5, e0 = 0.2).
- The Peters small-e analytic map e_p = e0·(p_sep/p0)^(19/12) (from
  p ∝ e^(12/19), Peters 1964 Eq. 5.11 small-e limit) *underestimates* e_p by
  2–4× — PN5 eccentricity decays slower than quadrupole near the separatrix —
  so the analytic map is not used for anything except this cross-check.
- C0 continuity: as t_plunge → 0, p0 → p_sep + 0.05 and e_p → e0 continuously
  (no boundary jump); the transplant error vanishes in that limit.
- Systematics entry (§8): low-M_z plunge eccentricities biased low relative to
  Babak's flat-at-plunge band. Circular-limit EMRI SNR varies smoothly with
  e ≤ 0.2 (few-percent level); the dark-siren estimator never conditions on e.
  If a future audit needs the exact convention, the 2-D root-find upgrade slot
  is `plunge_window.draw_plunge_window_initial_conditions`.

## 5. The p0 domain rule, dimensional analysis, limiting cases

**What few 2.0 actually accepts (MEASURED, code inspection + waveforms):**

- `few/utils/baseclasses.py` `Pn5AAK.sanity_check_init` (lines 710–757)
  enforces **no p0 range at all** — only positivity, |Y0| ≤ 1, a spin-sign
  flip, and a *logger warning* for μ/M > 1e-4. The "10 ≤ p0 ≤ 16 + 2e0"
  statement is docstring-only for Pn5AAK (it IS enforced for the
  SchwarzschildEccentric flux models, which this pipeline does not use in
  production).
- The binding floor is the **trajectory separatrix machinery**:
  `ODEBase.min_p = p_sep(a, e0, x0) + 0.05` (`separatrix_buffer_dist`), with
  p_sep from `get_separatrix` (Stein & Warburton 2020). For a = 0.98 prograde
  (Y = 0.9, e = 0.05–0.2): p_sep ≈ 1.80–1.93 (MEASURED); retrograde (Y = −0.9):
  p_sep ≈ 9.20.
- MEASURED: full 4.5-yr Pn5AAK waveforms generate cleanly at p0 ∈ {7, 8, 9},
  M_z = 1e7 (14 201 168 samples each, ≈ 15.6 s CPU; `p0_domain_waveforms`
  block). No failure wall above the separatrix buffer was found (consistent
  with audit item 7's zero-failure measurement to M_z = 10^7.37).

**Adopted domain rule (first principles, no fitted constants):**

```
p0 ∈ [ p_sep(a, e0, x0) + 0.05 ,  p_up ]      (plunge-window mode)
```

The lower end is few's own trajectory-termination boundary — the p0 at which
t_insp = 0⁺, i.e. the t_plunge → 0 limit of the convention itself (C0 at the
boundary: the draw measure vanishes there linearly with t_plunge, no clamp, no
discontinuity). There is **no upper clamp**: p_up is only the verified brentq
bracket (2× Peters seed, geometrically enlarged until t(p_up) ≥ t_plunge);
low-M_z events legitimately start far out (MEASURED: p0(t_plunge = 4.5 yr) =
109.5 at M_z = 1e4, 34.3 at 1e5), where the PN5 flux is *more* accurate, not
less. The retired [10, 16] snapshot clamp survives only behind
`--snapshot_ics`.

**Dimensional analysis.** p0, e0, x0, a dimensionless; M_z, μ in M_sun;
t_plunge in yr. Inside the draw: t_plunge·YRSID_SI [s] vs trajectory t [s] ✓;
Peters seed p = [(256/5)·(t·c³/(G M_z))·(μ/M_z)]^(1/4) — with G M_z/c³ =
M_z·MTSUN_SI [s], the bracket is s/s = 1 ✓. The trajectory's p is in units of
G M_z/c², matching the waveform's p0 slot ✓.

**Limiting cases.**

| Limit | Behavior | Check |
|---|---|---|
| t_plunge → 0 | p0 → p_sep + 0.05, signal length → 0, SNR → 0 | continuous (C0); MEASURED p0(0.01 yr, 1e7) = 2.369 just above p_sep + 0.05 ≈ 1.88–1.93 |
| t_plunge → T | longest in-window inspiral; p0(T) = 109.5 / 34.3 / 10.83 / 6.81 / 4.71 / 3.76 at M_z = 1e4 / 1e5 / 1e6 / 3e6 / 1e7 / 2.5e7 (MEASURED) | monotone in M_z ✓ |
| Small M_z | p0 ≫ 16: the event spends the window far out and sweeps in to plunge — the old snapshot draw at these masses plunged within days and sat silent for the rest of T; the new convention strictly dominates it in realized signal | — |
| Old-band recovery | p0(U[0, 4.5 yr]) at M_z ≈ 1e6 spans ≈ (1.9, 10.83] — the snapshot [10, 16] range is NOT recovered anywhere; the old draw corresponds to no plunge-consistent population at any mass (it was t_insp(p0) ∈ [2.6, 19.5] yr at 1e6, i.e. mostly out-of-window). Honest statement: this is a convention *replacement*, not a refinement with a snapshot limit | audit item 1b |
| Round trip | trajectory from drawn p0 (few default tolerances) replunges at t_plunge to ≤ 2.8e-4 relative (MEASURED, §9) | pinned in test |

## 6. Mid-observation plunge and the SNR integral (verified)

`Pn5AAKWaveform` is built with `sum_kwargs={"pad_output": True}` and
`T = ParameterEstimation.T`: the returned array always has n = T·YRSID_SI/dt
samples (MEASURED: 14 201 168 = 4.5 yr/10 s, including for signals that plunge
mid-window), with zeros after plunge. The rfft-based inner product
(`scalar_product_of_functions`) integrates the full padded span; the padding
contributes nothing, so **the SNR covers exactly the emitted span
[0, t_plunge]** — the signal ends at plunge (trajectory terminates at
p_sep + 0.05) and the quiet tail is zeros, not extrapolation. This is the same
mechanism that handled early-plunging snapshot draws at M_z < 1.4e6 in every
existing campaign (pilot evidence, audit item 4 tables).

## 7. T = 5 → 4.5 yr: SNR/PSD consequences (ESTIMATED)

- Long-lived signals (low M_z, t_plunge near T): SNR ∝ √T at fixed source →
  ≈ √(4.5/5) = 0.949, a −5 % SNR shift; horizon d_hor shifts by the same
  factor. Plunging signals with t_plunge < 4.5 yr: unchanged SNR, but the
  *window* shrinks so the per-source plunge probability mass shifts (the
  population, not the physics).
- Confusion noise: t_obs_years 4.0 → 4.5 moves the subtraction knees
  f1, fk by (4.5/4)^(a1), (4.5/4)^(ak) ≈ 0.971, 0.969 (a1 = −0.25,
  ak = −0.27) — a ≈ 3 % downward knee shift (slightly deeper foreground
  subtraction); sub-percent SNR effect for the campaign band.
- Waveform arrays shrink 5 → 4.5 yr (1.58e7 → 1.42e7 samples): ~10 % less
  GPU memory/FFT cost per waveform.
- The `--snr_analysis` diagnostic ladder ([0.5, 1, 2, 3, 5] yr generators in
  `parameter_estimation.py`) is untouched — it is an explicit multi-T scan,
  not a mission-duration assumption.

## 8. Systematics-budget entries (for .planning/gate/G7_systematics_budget.md)

1. **AAK near-plunge SNR optimism.** LITERATURE (Babak 2017): AKK is
   SNR-optimistic near plunge; AKS/AKK bracket the truth. The plunge-window
   convention *by construction* concentrates signal power in the near-plunge
   cycles, so this waveform-family systematic now applies to every event (it
   previously applied only to the sub-1.4e6 masses that reached plunge).
   Direction: SNR (and hence p_det and horizon) biased HIGH, worst at high M_z.
2. **Duty cycle.** 4.5 yr science operations at > 82 % duty (Colpi et al.
   2024) is modeled as gap-free; ≈ −9 % worst-case SNR² (√: −4.6 % SNR) if
   gaps were uncorrelated.
3. **Eccentricity transplant** (§4): plunge eccentricities biased low at low
   M_z relative to Babak's flat-at-plunge distribution; |Δe| ≤ 0.19.
4. **Plunge-after-T truncation** (§1, ASSUMED per Babak): near sources
   plunging just after mission end are dropped though marginally detectable —
   conservative on event counts.

## 9. Numerical realization (implementation contract)

`master_thesis_code/plunge_window.py::draw_plunge_window_initial_conditions`:

- Called AFTER M_z is set, in BOTH population paths (`main.py`:
  `data_simulation` inside the per-event try, and `injection_campaign` inside
  the per-event try), so the two pools share one convention. The snapshot p0
  drawn by `randomize_parameters` (rng-stream-preserving) is overwritten;
  `--snapshot_ics` skips the overwrite and consumes no extra rng draws, making
  the archaeology path byte-identical to the old stream.
- Root-find: `get_p_at_t` with explicit bounds. Lower = few's
  `min_p = p_sep + 0.05` (requires the few 2.0 wart workaround: `min_p` reads
  `self.a`, set via `add_fixed_parameters` — few's default-bounds path crashes
  otherwise). Upper = max(2× Peters seed, p_lo + 1), verified against the
  trajectory and enlarged ×1.5 (≤ 30 times) — few's own `max_p` is +inf, which
  brentq cannot take.
- Tolerances (numerics, not physics): brentq `xtol = 1e-3` on p0, trajectory
  integrator `err = 1e-8` *inside the root-find only* (waveform generation
  keeps few's default 1e-11). MEASURED realized accuracy: the waveform-side
  trajectory from the drawn p0 replunges at |t − t_plunge|/t_plunge ≤ 2.8e-4
  (grid over M_z ∈ [1e4, 2.5e7] × t_plunge ∈ [0.01, 4.5] yr) — ≤ 4 h of a
  4.5-yr window.
- **Cost (MEASURED, dev CPU):** median ≈ 0.33 s/draw (range 0.08–1.2 s; low
  M_z is slowest — more trajectory steps from far out; first call +numba
  warmup ≈ 12 s once per process). Campaign amortization: 200k draws ≈ 18
  CPU-hours total, spread over the array tasks that each spend multiple
  GPU-seconds per SNR waveform — per-event overhead ≲ 10–30 % of the waveform
  cost.
  **Interpolation-table fallback (NOT built, spec for the record):** the
  pre-registered ~100 ms/call line is crossed (0.33 s), but a faithful table
  is 4-D — p0(t_plunge, M_z, e0, x0) with strong x0 dependence (MEASURED:
  p0(1e7, 4.5 yr) = 4.71 at x0 = 0.9 vs 9.32 at x0 = −0.9) — ~4×10^4 nodes
  ≈ 12 CPU-h to build plus a boundary/accuracy validation of its own, for a
  saving that is operationally irrelevant at the measured amortized cost.
  Decision: **direct root-find** (zero approximation error, no new interpolant
  surface). Revisit only if a future campaign multiplies the draw count ≥ 10×;
  spec then: log-uniform M_z axis (~40), t_plunge^(1/4)-uniform axis (~30,
  linearizes the Peters quartic), e0 (~4), x0 (~9 excluding |x0| < 0.1),
  cubic RegularGridInterpolator, acceptance = round-trip |Δt|/t ≤ 1e-2 on a
  held-out random set.
- Failure modes: brentq non-convergence raises ValueError("Brent root solver
  does not converge…"); near-polar draws raise ZeroDivisionError in few's
  Y→xI map — both are the SAME exception classes, caught by the SAME per-event
  skip handlers (and 90 s alarm), as the subsequent waveform call. No new
  selection surface is introduced: a draw that fails here would have failed at
  waveform generation.
- **Provenance recording (cheap, adopted):** `parameter_space.t_plunge_yr`
  (NaN in snapshot mode; reset on every randomize) is written to (i) every
  injection CSV row (`t_plunge_yr`, plus `p0`, previously unrecorded in the
  pool) and (ii) every CRB row (`t_plunge_yr` column beside T/dt). Note: CRB
  CSVs append with a header written only at file creation — do not resume a
  pre-change CSV file with post-change code.

## 10. Regression pins

- OLD pins (commit first):
  `results/campaign51_20260728/plunge_window/old_pins_test_version.py` —
  seeded snapshot p0 value 14.184208174356183 (rng 42), uniform-law KS,
  T = 5, t_obs_years = 4.0. Green against d31822c.
- NEW pins (working tree): `master_thesis_code_test/test_plunge_window.py` —
  same seeded snapshot value still reachable (archaeology law byte-identical),
  T = t_obs_years = 4.5 = constant, plunge-window round trip
  t_insp(p0) ≈ t_plunge (rel 1e-2; measured margin 36×), domain rule
  p0 ≥ p_sep + 0.05, high-M p0 < 10, t_plunge provenance reset, 14-key
  parameter dict unchanged.

## 11. First corrected-physics look at the high-M band (MEASURED ratios /
ESTIMATED absolutes)

Response-less h+ SNR at 1 Gpc against the strain-referred effective
sensitivity S_eff = S_instr/R + S_c (the estimator validated against pilot
plateaus in the audit; R = 1.5·(2x sin x)², x = 2πfL/c), fixed-PSD tree
(49251f3), plunge-window ICs (e0 = 0.1, x0 = 0.9, μ = 10, a = 0.98, T = 4.5):

| M_z | t_plunge [yr] | p0 | SNR @ 1 Gpc | d_hor = SNR/20 [Gpc] | power 5–95 % [mHz] |
|---|---|---|---|---|---|
| 3e6 | 0.5 | 4.85 | 45.8 | 2.29 | 2.0–4.4 |
| 3e6 | 2.0 | 5.93 | 49.2 | 2.46 | 1.6–4.3 |
| 3e6 | 4.0 | 6.67 | 50.3 | 2.51 | 1.4–4.3 |
| 1e7 | 0.5 | 3.61 | 6.9 | 0.35 | 0.9–1.7 |
| 1e7 | 2.0 | 4.25 | 8.7 | 0.44 | 0.7–1.6 |
| 1e7 | 4.0 | 4.64 | 9.7 | 0.48 | 0.6–1.6 |

Horizon scale: **~2.3–2.5 Gpc (z ≈ 0.4) at M_z = 3e6** and **~0.35–0.5 Gpc
(z ≈ 0.08–0.10) at M_z = 1e7** — vs the corrected-PSD *snapshot* values of
0.1–0.25 Gpc and 0.014 Gpc respectively (audit item 4 table): the plunge-window
convention recovers ≈ 1–1.5 decades of horizon at the top of the band,
consistent with Babak 2017's "detectable up to 10^7 M_sun when MBHs are
spinning" (LITERATURE). Note t_plunge dependence is mild (SNR grows only ~40 %
from t_plunge = 0.5 → 4.0 yr at 1e7): the detectability is dominated by the
plunge-phase cycles, exactly the physics the snapshot convention excluded.
Absolute values remain ESTIMATED (no TDI response on this host — audit
methodology); ratios and scalings are response-independent.
