# PRE-REGISTRATION — closed-loop two-channel calibration test (G4b / §9)

**Status:** FROZEN before the run. Append-only (see §8).
**Date written:** 2026-08-05
**Author of record:** Jasper Seehofer. Written by the AI assistant under the
`physics-change`/pre-registration discipline; this document *registers*, it
does not decide.

**Registration of record it implements:**
`.planning/derivation-gfrac-20260805/GFRAC_DERIVATION_PACKAGE.md` §9
("Pre-registered acceptance criteria, should the author want R-A tested rather
than accepted on the derivation"), as amended by
`.planning/derivation-gfrac-20260805/GATEB_REFUTATION_REPORT.md` (amendment 2:
the `kappa_cap` kink at `M = 1e5 M_sun` is ACTIVE in the real data and MUST be
present in the synthetic φ; amendment 5: the §9 closed-loop run "is the
deciding measurement" for R-A), and `docs/RESEARCH_CYCLE.md` stage 4
amendment **A3** (harness acceptance criteria).

---

## 1. Question of record

> Is the 2D (with-BH-mass) channel **CALIBRATED** when the universe actually
> follows the estimator's own generative assumptions — the estimator's φ
> (including the `kappa_cap` kink), its `w_pop` measure, its `S_4D` detection
> object, and its fractional `cov_4d` error model?

This is the only measurement that separates the two live readings of the
residual 2D high-h displacement:

* **artifact** — the displacement is inherited from the 1D-channel bias plus a
  *genuine* spectral-siren tilt (verdict candidate (i), R-A of the derivation
  package); or
* **defect** — the 2D leg's normalisation is wrong, in which case the
  displacement survives even in a loop where the data provably follow the
  model.

## 2. Instrument

New module `master_thesis_code/validation/closed_loop_gfrac.py` (a **new**
instrument; `validation/pp_coverage.py` is NOT modified — it is deliberately
production-independent, and this harness is deliberately production-*dependent*,
which is the entire point: the loop only closes if the generator and the
estimator share the same population objects).

**Production objects imported (the closed-loop guarantee):**

| object | import | role |
|---|---|---|
| `dark_mass_density_per_mass` (φ, `M_sun^-1`, normalised on `[1e4, 1e7]`) | `bayesian_inference.bayesian_statistics` | source-mass draw **and** `g_i`; carries the `kappa_cap` kink via `emri_rate.R_eff_per_mbh` |
| `completion_mass_factor_g` | `bayesian_inference.bayesian_statistics` | the 2D completion leg's `g_i(z;h)`, **called verbatim**, recomputed at every `h` (A3(i)) |
| `precompute_phi_marginal_survival` → `S_bar_phi(z;h)` | `bayesian_inference.bayesian_statistics` | the estimator's own selection normalisation `α(h)` |
| `SimulationDetectionProbability.detection_probability_with_bh_mass_interpolated` (`S_4D`) | `bayesian_inference.simulation_detection_probability` | **the detection rule of the generator** (deterministic-horizon survival, pooled 2D `S(d_L | M_z)`) |
| `dist_vectorized`, `dist_to_redshift`, `comoving_volume_element` | `physical_relations` | flat-ΛCDM distance ladder and `dV_c/dz`; `w_pop = dV_c/dz /(1+z)` |
| `_HOST_QUAD_N` (=50) Gauss–Legendre order, `_G_I_HERMITE_NODES` (=64) | `bayesian_inference.bayesian_statistics` | identical quadrature convention to production |

**Injection pool defining `S_4D`:**
`results/campaign51_20260728/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728`
(`snr_threshold = 20`), i.e. the production `mix200k` pool.

**What the harness deliberately does NOT do** (stated here so the verdict is
never over-read):

* **A3(iii) multi-candidate host balls are OUT OF SCOPE.** This is a
  single-host, catalogue-leg-off instrument, exactly as §9 specifies
  ("scored by the 2D completion leg alone (catalogue leg off, which §6.6 shows
  is nearly the production configuration anyway)"). A one-candidate-per-event
  harness structurally cannot exercise the impostor-ball mechanism, and no
  claim about that mechanism may be drawn from this run. A constant-completeness
  catalogue leg (`--f-cat`, default `0.0`) exists in the code for limiting-case
  tests only and is **off** for the registered run.
* It does not use GLADE+, real n(z), photo-z, or the completeness map.
* It does not call `BayesianStatistics` — it re-implements the completion-leg
  math compactly (§3) so 200 seeds × 1500 events × 41 h is affordable.
  Fidelity is enforced by calling the production `completion_mass_factor_g`
  and `precompute_phi_marginal_survival` rather than re-coding them.

## 3. Generative model (the universe) and the estimator (the readout)

**Truth:** `h_true = 0.73`, flat ΛCDM with the pipeline's fiducial `OMEGA_M`.

**Per universe (one seed):**

1. Draw `z ~ w_pop(z; h_true) = (dV_c/dz)/(1+z)` on `[1e-6, z_max(h_true)]`,
   `z_max = dist_to_redshift(get_dl_max(h_true), h_true)`.
2. Draw source-frame `M ~ φ(M)` — the estimator's own φ, **including the
   `kappa_cap` kink at `1e5 M_sun`** (GATEB amendment 2), on `[1e4, 1e7]`.
3. `d_L = dist(z; h_true)`, `M_z = M (1+z)`.
4. **Detect** with probability `S_4D(d_L, M_z)` — the production survival
   object. Accept/reject until `N_det` detections.
5. Observation error: fractional 2×2 `(d_L, M_z)` block of `cov_4d`,
   `(σ_dL/d_L, σ_Mz/M_z, ρ)` **bootstrap-resampled from the production
   prepared CRB set**
   (`results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv`,
   1590 rows; measured medians `σ_dL/d_L = 0.0373`, `σ_Mz/M_z ≈ 1e-8`,
   `ρ ≈ -3e-4`). Draw `(d_L^obs, M_z^obs)` from the resulting 2-D Gaussian
   about `(d_L, M_z)`.
   *Declared simplification:* the resampled `σ` triple is drawn independently
   of the event's own `d_L` (production has `corr(ln σ_dL, ln d_L) = 0.82`).
   The loop still closes exactly — the estimator conditions on the same
   per-event `σ` it was generated with — but the joint `(σ, d_L)` texture is
   not reproduced. Recorded as a deviation, not hidden.

**Estimator (mirrors `single_host_likelihood`'s completion leg, `g` recomputed
per `h`, A3(i)):** for each `h` on the canonical 41-point grid,

```
B1_i(h)  = ∫ dz  (1-f) · w_pop(z;h) · N(d_L(z;h)/d_L^obs_i ; 1, σ_dL,i)                       # 1D
B2_i(h)  = ∫ dz  (1-f) · w_pop(z;h) · N(d_L(z;h)/d_L^obs_i ; 1, σ_dL,i) · g_i(z;h)            # 2D
α(h)     = ∫ dz  w_pop(z;h) · S̄_φ(z;h)
ln P(h)  = Σ_i [ ln B_i(h) − ln α(h) ]      (+ flat prior on h)
```

with the z-window `dist_to_redshift(d_L^obs (1 ∓ 4σ_dL); h)` capped at
`z_max(h)`, 50-node Gauss–Legendre (`_HOST_QUAD_N`), and `g_i` the production
`completion_mass_factor_g` with `det_M_z = M_z^obs_i`,
`proj = Σ[dL,Mz]/Σ[dL,dL]`, `σ_cond = sqrt(Σ[Mz,Mz] − Σ[dL,Mz]²/Σ[dL,dL])`.
The isotropic sky factor `sinθ/4π` is a per-event, h-independent constant and
is omitted (it cancels from the posterior shape).

`α(h)` is shared by both channels — the derivation package's §2 (T2) statement
that α is a property of the population and the detector, not of which
observables the analyst uses.

**Numerator selection factor:** the shipped estimator carries **no** `p_det`
inside the numerator; the registered run reproduces that (`--numerator-pdet`
default `off`). A diagnostic variant (`on`, inserting `S̄_φ(z;h)` into the 1D
quadrature and `S_4D` inside `g_i`'s mass integral — the GATEB re-scoped N-2)
exists in the code and is **not** part of the registered readout.

## 4. Seed plan, N, grid

| item | registered value |
|---|---|
| base seed | **20260805** |
| seeds | `20260805 … 20260805+199` — **200 seeds** (≥200 per §9) |
| detected events per universe `N_det` | **1500** (production venue: 1588 analysed events in `run_20260804_postfix`; A3(ii) "run at production N") |
| h grid | the canonical **41-point** production grid: `0.60(0.01)0.65, 0.655(0.005)0.795, 0.80(0.01)0.86` — read off `run_20260804_postfix/iiib/diagnostics/event_likelihoods.csv` |
| `h_true` | 0.73 |
| catalogue fraction `f` | 0.0 (completion leg only, §9) |
| code commit | filled in §7 below, at the commit that adds the module |

## 5. Readouts (fixed before the run)

Per seed: 1D grid-argmax MAP, 2D grid-argmax MAP, 1D and 2D full-grid posterior
means, parabolic-refined MAPs (secondary), rail flags, and
`Σ_i ∂_h ln g_frac,i` at `h = 0.73`.

Aggregate: MAP distribution quantiles (0/5/25/50/75/95/100 %), mean MAP, MC
error of the mean (`sd/√n`), mean displacement `⟨MAP⟩ − 0.73`, railed fractions.

## 6. Decision rule (§9 verbatim, with the operational reading fixed here)

Let `Δ2 = ⟨2D MAP⟩ − 0.73` over the seeds (grid-argmax MAPs; the parabolic
MAP is reported alongside and must not be substituted after the fact).

* **CONFIRM (i):** `⟨2D MAP⟩ ∈ 0.73 ± 0.010`, i.e. `|Δ2| ≤ 0.010`.
  (§9: "the 2D MAP distribution is centred on 0.73 within ±0.010 (MC error at
  200 seeds ≈ 0.005 given the 0.03 per-realisation spread of record)".)
  ⇒ `g`'s tilt is self-consistent; the production displacement is inherited;
  **R-A stands.**
* **REFUTE:** `Δ2 ≥ +0.03`.
  (§9: "the 2D MAP is displaced by ≥ +0.03 in a closed loop where the data
  provably follow the model".) ⇒ the 2D leg's normalisation is defective;
  **R-C becomes blocking** and the fix is derived there.
* **MIXED (first-class outcome, not a failure of the test):** anything else —
  `Δ2 ∈ (0.010, 0.030)`, **or any negative displacement of any size**, or a
  bimodal/railed MAP distribution that makes the mean unrepresentative.
  §9: "report the split; do not force a branch." In the MIXED branch the
  measured `Σ_i ∂_h ln g_frac` of the synthetic set **must** be quoted against
  the production `+243.5 nats/h`, and the railed fractions reported.

**Pre-registered expectation for the 1D channel** (§9 does not band the 1D
channel; it is banded here so it cannot be read post hoc):

* The 1D channel in this closed loop has **no photo-z error, no galaxy
  catalogue, no impostor hosts, and no D1-class selection mismatch** — the
  three named candidates for the production 1D rail at 0.600 are all absent by
  construction. The registered expectation is therefore that the **1D MAP is
  centred on 0.73 within ±0.010** and does **not** rail.
* **If the 1D channel rails low (or is displaced by more than ±0.010) in this
  closed loop**, that is a first-class finding *about the 1D estimator*, not a
  nuisance: it would mean the 1D completion numerator `B_num` is miscalibrated
  against its own generative model, which is exactly the structure GATEB's
  re-scoped **N-2** predicts (the numerator omits the selection factor that
  acts in the discarded `M_z^obs` coordinate). It will be read as: *the
  production 1D rail has an estimator-internal component that photo-z cannot
  explain*, it will be reported as a named finding with the measured 1D
  displacement, and it makes the `--numerator-pdet on` diagnostic variant the
  immediate follow-up. It does **not** by itself change the 2D verdict — the
  2D bands above are evaluated independently.
* If **both** channels are displaced by a similar amount and in the same
  direction, the shared object is `α(h)` or the shared `p_gw`/`w_pop`
  quadrature; that will be reported as a *common-mode* finding and neither
  channel's band may be read as a `g`-specific statement.

## 7. Provenance

* Code commit adding the module: recorded in the run's `results.json`
  (`git_commit` field) and appended to §8 after the commit.
* Repository HEAD at the time this pre-registration was written:
  `9984a4d228f92c80b07bb1abb2c6f7f96b5a7366`.
* Output: `results/closed_loop_gfrac_20260805/closed_loop_results.json`.

## 8. Append-only clause

This document is frozen at the commit
`prereg: closed-loop 2-channel calibration — G4b/§9 bands committed before the run`.
Nothing above §8 may be edited after that commit. Corrections, deviations
discovered during the run, and the readout are appended **below** this line
with a dated heading. If a band or a definition above turns out to be
unworkable, the fact and the reason are appended — the original text stays.

---

### Appendix log

#### 2026-08-05 — registered run executed and scored

Run: `results/closed_loop_gfrac_20260805/closed_loop_results.json`
Code commit: `77b524af` (`validation: closed_loop_gfrac — two-channel
closed-loop calibration harness (G4b/§9)`). 200 seeds `20260805…20261004`,
`N_det = 1500`, `f = 0`, `numerator_pdet = off`, canonical 41-h grid,
`h_true = 0.73`. Wall time ≈ 6.5 min on 14 local workers.

**Mechanical application of the §6 rule** (this is the frozen arithmetic, not
an interpretation — the verdict is the author's to accept):

| quantity | value |
|---|---|
| 2D MAP mean `⟨MAP⟩` | **0.741275** |
| 2D displacement `Δ2` | **+0.011275** |
| MC error of the mean | 0.004301 |
| 2D per-realisation spread (sd) | 0.0608 |
| 2D MAP quantiles 0/5/25/50/75/95/100 % | 0.60 / 0.64 / 0.69 / 0.74 / 0.786 / 0.85 / 0.86 |
| 2D railed low / high | 0.5 % / 3.5 % |
| 2D parabolic-refined MAP mean | 0.74155 ± 0.00432 |
| 2D full-grid posterior-mean average | 0.73485 ± 0.00308 |
| **1D MAP mean** | **0.600 — railed low in 200/200 seeds** |
| 1D full-grid posterior-mean average | 0.6048 |
| `Σ_i ∂_h ln g_frac` (synthetic, mean over seeds) | **+162.0 nats/h** (5–95 %: 135 – 171) |
| production reference (§6.2) | +243.5 nats/h |
| mean fraction of hosts below the `kappa_cap` kink | 0.227 (the kink is active, GATEB amendment 2) |
| mean median host redshift | 0.407 |

**Verdict by the frozen rule: MIXED.** `Δ2 = +0.0113 ∈ (0.010, 0.030)` — it
misses CONFIRM by 0.0013 (0.3 × the MC error above the band edge) and is
2.7× below the REFUTE threshold. §6 requires the split be reported and no
branch forced. The two secondary MAP statistics straddle the band edge: the
parabolic-refined MAP mean is `+0.0116` (also MIXED), the full-grid posterior
mean is `+0.0049` (inside `±0.010`). Per §5/§6 the grid-argmax statistic is
primary and the others may not be substituted after the fact.

**1D channel, read against the §6 pre-registered expectation.** The registered
expectation ("1D MAP centred on 0.73 within ±0.010 and not railing") is
**violated**: 200/200 seeds rail at 0.600. Two readings, both stated:

1. *Information-starvation (deflationary, and the one this harness's own
   configuration most supports).* With `f = 0` there is no catalogue, no
   photo-z and no host redshift of any kind, so the 1D channel observes only
   `d_L^obs`. The detected `d_L` distribution is almost h-independent — h
   enters the 1D leg only through the volume element's z-shape — so the 1D
   posterior is a near-flat curve whose tiny residual slope is amplified by
   1500 events into a rail. **A 1D rail in THIS configuration is therefore
   not, on its own, evidence about the production 1D rail**, and must not be
   quoted as such.
2. *Estimator-internal (the N-2 structure).* The rail direction and its
   perfect reproducibility are consistent with the numerator omitting the
   selection factor that acts in the discarded coordinate. The §6-mandated
   follow-up (`--numerator-pdet on`) was run and is reported in the next
   appendix entry.

Because the 1D channel is starved by construction here, the **common-mode**
clause of §6 does not apply: the 2D displacement (+0.011) and the 1D rail
(−0.130) are not a shared displacement of similar size and direction, so the
2D band reading stands on its own.

**Harness plumbing control (not part of the registered readout).** With every
host redshift known exactly (`--f-cat 1.0`), both channels return
`MAP = 0.730` exactly, on every seed tested — the ladder, the noise draw, the
quadrature and the `α(h)` normalisation are verified.

**Deviations from §9's design, declared:**

* The observation-error triples are bootstrap-resampled from the production
  CRB set **independently of the event's own `d_L`** (production has
  `corr(ln σ_dL, ln d_L) = 0.82`). Declared in §3; the loop still closes
  exactly, but the joint `(σ, d_L)` texture is not reproduced.
* §9's second CONFIRM clause — "and the P–P curve is inside the 90 % band" —
  was **not evaluated**: this harness reports MAP statistics, not a rank/P–P
  statistic. The CONFIRM criterion is therefore only partially testable by
  this run, which is an additional reason the outcome is reported as MIXED
  rather than pushed toward CONFIRM.
* The synthetic universe's tilt is `+162 nats/h` versus production's
  `+243.5`; the synthetic host population is also more low-mass than
  production (22.7 % below the kink, versus production's ~0 % below `1e5`).
  The closed loop is self-consistent, but it is **not** a replica of the
  production event set.

#### 2026-08-05 — follow-up diagnostic: `--numerator-pdet on` (N-2 variant)

Not part of the registered readout (§3 declares it a diagnostic). Run:
`results/closed_loop_gfrac_20260805/diagnostic_numerator_pdet_on.json`,
50 seeds (`20260805…20260854`), same `N`, grid and `f = 0`; the only change is
`S̄_φ(z;h)` inside the 1D quadrature and `S_4D` inside `g_i`'s mass integral
(the latter evaluated at `μ_cond`, exact to the measured `σ_cond ≈ 1e-8`).

| statistic | registered run (`off`) | diagnostic (`on`) |
|---|---|---|
| 1D MAP mean | 0.600 (rail) | **0.7493** ± 0.0097 |
| 1D railed low | **200/200** | **0/50** |
| 1D MAP quantiles 5/50/95 % | 0.60 / 0.60 / 0.60 | 0.63 / 0.755 / 0.85 |
| 2D MAP mean | 0.7413 ± 0.0043 | 0.7585 ± 0.0085 |
| 2D displacement | +0.0113 | +0.0285 |
| 1D posterior-mean average | 0.6048 | 0.7436 |
| 2D posterior-mean average | 0.7349 | 0.7464 |

**Reading.** Inserting the selection factor in the numerator **removes the 1D
rail completely** (200/200 → 0/50) and moves the 1D channel from −0.130 to
+0.019. That is a strong, reproducible signature that the 1D rail seen in the
registered configuration is the N-2 structure (numerator missing the selection
factor that acts in the discarded coordinate) and not merely
information-starvation noise — the starved channel would not become sharply
informative just because a factor was added. It does **not** transfer to the
production 1D rail, which sits in a configuration with a catalogue and photo-z
that this harness does not model.

**Caveat that limits both runs (raised here, not hidden).** The per-seed
posteriors are broad and skewed, so *the mean of the grid-argmax MAP is not a
clean unbiasedness statistic*: in the registered run the 2D **posterior-mean**
average is 0.7349 (displacement +0.005, inside the CONFIRM band) while the 2D
**MAP** mean is 0.7413 (+0.011, MIXED). §9 fixed the MAP as primary before the
run and that ordering is honoured, but the split between the two statistics is
itself part of the MIXED report and should inform how much weight the author
puts on the 0.0013 by which the CONFIRM band was missed.
