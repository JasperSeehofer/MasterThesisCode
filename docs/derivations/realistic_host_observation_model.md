# Realistic host-observation model (campaign #53) — DERIVATION PACKET

**Status: RATIFIED (author, 2026-07-29).** All gates [RATIFY-R1]…[RATIFY-R9]
approved as recommended, with two explicit author decisions recorded at
ratification:

- **[RATIFY-R4] → option (c)**: all redshift flags retained, per-row (per-flag)
  kernel widths. No selection-function surgery; the constraint is allowed to
  come from whichever hosts carry usable redshifts. Option (b) (spectroscopic-
  only) is NOT adopted as the production analysis; it may be revisited as a
  follow-up-forecast cross-check.
- **[RATIFY-R7] campaign shape → CPU-only first**: 2 existing truth seeds
  (61000, 62000) × N observation realizations, reusing the existing CRBs and
  the `injection_pool_mix200k_20260728` pool. Additional GPU truth seeds are
  DEFERRED pending the measured realization spread (the forecast is
  Poisson-dominated in the spectroscopic-host count) — a data-driven decision
  point, not a prior commitment.
- Author-accepted approximation: the z ≥ 1e-5 clip of §2 (breaks exact
  σ_realized = σ_kernel for photo rows at z ≲ 2σ, which carry ~no information).

Blocking audit item of §5 RESOLVED before implementation: the pixel
completeness cache `m_th` has shape `(npix,)` and is built from apparent
B magnitudes with NO z binning (`pixel_completeness.py` build path; redshift
enters only later via the distance modulus at query time), and a realization
perturbs neither magnitudes nor flags ⇒ the frozen cache stays valid, no
per-realization rebuild.

Gates below each carry their original stated recommendation. No production
source is modified by this packet; supporting numerical checks live in
`results/campaign51_20260728/realistic_model/` (scripts `r1_*.py`, `r2_*.py`,
`r3_*.py` + JSON outputs, all reproducible from the repo venv).

**Author decision being implemented (2026-07-29):** the idealized campaign-#51
runs are RETAINED as the validated generator↔evaluation consistency baseline
(the "generator-exact" mock, MAP-on-truth by construction); the FINAL
production run must realize the catalogue's actual measurement errors — host
redshift (photo-z/spec-z + peculiar velocity) and host BH mass (0.24 dex) — as
noise, not merely as kernel widths. Driving record:
`results/campaign51_20260728/idealization_audit/IDEALIZATION_LEDGER.md`
(items I1–I4; measured: 100% of the H0 information from 76 in-catalogue hosts,
ALL photometric with σ_z/z ≈ 50%, treated as exact; quoted σ_H0 = 0.03 km/s/Mpc
vs realistic 0.2–4 km/s/Mpc).

Tags: **MEASURED** (computed here or in the ledger from production artifacts),
**ESTIMATED** (analytic/order-of-magnitude), **ASSUMED** (modeling choice to be
ratified), **LITERATURE** (external reference).

---

## 0. Why this is needed (measured, not hypothetical)

From the idealization ledger (all MEASURED unless noted):

- σ_H0 = 0.032 km/s/Mpc quoted (seed 61000, 1588 events) is carried entirely
  by 76 in-catalogue events; the 3 loudest (SNR 995–1425, z ≈ 0.016–0.021)
  carry 46% alone.
- All 76 information carriers are GLADE+ **photometric**-redshift hosts
  (flag 1, median σ_z/z = 49%) whose z is injected verbatim as truth (I1) and
  point-evaluated by the production `generator_marginal` δ-kernel (I2). No PV
  is ever realized (I3); the 0.24 dex mass scatter is width-only (I4).
- Counterfactuals from the same event set: σ_H0 ≈ 0.22–0.30 (spec-z + PV) to
  ≈ 3.6 (catalogue photo-z widths) km/s/Mpc — 8× to 110× the quoted width.

The pair (I1, I2) is internally consistent ("generator-exact") — a valid
*consistency baseline*, not a forecast. This packet derives the model in which
the noise is realized, the kernels keep their licence, and nothing is counted
twice.

---

## 1. The generative model, explicitly [RATIFY-R1]

### 1.1 TRUE vs OBSERVED — the full ledger of quantities

Per event, per candidate host row g of the reduced catalogue
(`handler.py:192-215` schema: RA, Dec, B-mag, z, z_error, M*, M*_err, flag):

| Quantity | TRUE (generates signal / selection) | OBSERVED (visible to inference) | Noise model |
|---|---|---|---|
| Host cosmological redshift | `z_g` — the catalogue's stored z, **declared truth** (§1.2) | `z_obs,g` = realized per run (§2) | one total Gaussian, σ = stored `z_error` (contains measurement ⊕ BORG-PV ⊕ per-class PV, counted once per #40b) |
| Host peculiar velocity | folded: the PV displacement is one component of the total z scatter (§2.2) | — (never separately observed) | inside `z_error` per PV class (150/500 km/s, `constants.py:83-88`) |
| Host BH mass | `M_g` — catalogue RV-relation mass, **declared truth** (§1.3) | `M_obs,g` = M_g·10^(σ_dex·N(0,1)) realized per run | lognormal, σ_ln = `M_error/M` (stored column: 0.24 dex intrinsic ⊕ fit ⊕ propagated M*, `handler.py:1117-1131`) |
| GW data (d_L, M_z, sky, …) | injected at (z_g, M_g): d_L = dist(z_g, h_inj), M_z = M_g(1+z_g) (`parameter_space.py:260-273`) | CRB-covariance 4D MVN draw (`detection.py:161-240`) — **already realistic, unchanged** (ledger I6, verified) | Fisher/CRB marginal covariance |
| Sky position | catalogue RA/Dec, exact | same (GW sky error realized on the GW side) | σ_sky(catalogue) ≪ Fisher sky ellipse — neglected, unchanged (I7) |

Nothing else changes: dark hosts (I9), selection knobs (I12), GW scatter (I6)
are already self-consistent or realistic.

### 1.2 Direction of the redshift noise — the central convention

Two candidate conventions:

**(A) Catalogue z is TRUE; observation is scattered forward.**
The stored `z_g` is declared the host's true cosmological redshift. The event
is generated exactly as today (d_L = dist(z_g, h_inj)); the inference is handed
a realized *observed catalogue* with

$$z_{\mathrm{obs},g} \;=\; z_g \;+\; (1+z_g)\,\frac{v_{\mathrm{pec},g}}{c} \;+\; \mathcal N(0,\sigma_{\mathrm{meas},g})\;\;\equiv\;\; z_g + \mathcal N\!\big(0,\ \sigma_{z,g}^{\mathrm{tot}}\big),$$

the second equality because PV displacement and measurement error are
independent Gaussians and the stored `z_error` is exactly their quadrature
total (§2). ((1+z) PV factor: Davis et al. 2011, arXiv:1012.2912, Eqs. (1)/(A1).)

**(B) Catalogue z is the OBSERVATION; truth is scattered.**
The stored z_g is kept as what the inference reads; a latent truth
z_true,g ~ p(z_true | z_obs = z_g) is drawn and the event is generated at
d_L(z_true,g, h_inj).

**Analysis.** (B) matches the real-data reading ("GLADE+ numbers are
measurements") but requires the *posterior* draw
p(z_true | z_obs) ∝ N(z_obs; z_true, σ)·p_pop(z_true): sampling naively
z_true ~ N(z_g, σ) omits the population prior and commits an inverse/Eddington
error — the truth population it generates is the observed distribution
*convolved again* with σ instead of deconvolved, exactly the asymmetry the
G2b volume-deconvolution was built to remove. Doing (B) correctly needs an
explicit deconvolution prior per row (ill-posed for σ_z/z ≈ 0.5 photo rows).
(A) is a *forward* model: truth field with the catalogue's z-distribution,
observed under the exact error model the kernel assumes — no inverse problem,
no prior ambiguity, and the likelihood's premise "z_obs | z_true ~ N(z_true, σ)"
is true by construction. Its price is a population-level caveat: the declared
truth z-distribution equals GLADE+'s *observed* one (which is photo-z-smeared
and Malmquist-shaped), i.e. the mock universe's galaxy field is the observed
catalogue rather than the deconvolved sky. That is a second-order,
population-level statement (it does not touch per-event consistency), and it is
the standard practice in dark-siren mock studies that draw hosts from a real
catalogue (Gray et al. 2020, arXiv:1908.06050 §IV mock construction). **ASSUMED,
to be stated in the paper.**

(A) has a decisive practical corollary: the event's d_L and hence every
waveform, SNR and CRB is unchanged — see §7.

Same treatment for mass (§1.3): catalogue mass declared truth, observation
scattered forward in log space; identical Eddington caveat (the declared truth
mass function is the RV-mean-mapped catalogue one, not deconvolved by 0.24 dex —
one-sided note: near the M ∈ [1e4, 1e7] pruning edges forward scatter moves
observed masses across the boundary, §5.3).

**[RATIFY-R1] Convention: catalogue values are TRUE; observations are realized
forward, z_obs = z_g + N(0, σ_z,g^tot), ln M_obs = ln M_g + N(0, σ_lnM,g).**
*Recommendation: adopt (A).* It is the only direction that is prior-free and
self-consistent with the inference kernels, and it keeps the GPU products
valid. The Eddington/population caveat of (A) (truth field ≡ observed
catalogue) must be stated in the paper's mock-construction section; the caveat
of (B) (prior-dependent, ill-posed deconvolution at σ_z/z ≈ 0.5) is fatal.

### 1.3 Mass: what "0.24 dex realized" means under (A)

The catalogue BH mass is the deterministic Reines & Volonteri (2015,
arXiv:1508.06274, §4.1) mean-relation image of M*, with the stored `M_error`
the quadrature of intrinsic scatter (0.24 dex, dominant), fit-parameter and
propagated M* errors (`handler.py:1117-1131`). In reality the *estimator* is
deterministic per galaxy and the *truth* scatters around it; under (A) we
declare the catalogue value truth and realize the estimator error forward:

$$\ln M_{\mathrm{obs},g} = \ln M_g + \sigma_{\ln M,g}\,\mathcal N(0,1),\qquad \sigma_{\ln M,g} = M_{\mathrm{error},g}/M_g .$$

Across the population this is statistically identical to the real
configuration under the symmetric-lognormal error model (the two directions
differ only by mass-function curvature over 0.24 dex — the same Eddington note
as §1.2). The GW-side injected M_z = M_g(1+z_g) is unchanged; the with-BH-mass
channel and the 4D ball-tree filter now see the realistic ~0.24 dex
truth-vs-catalogue disagreement (dissolves ledger I4). The realized noise uses
the SAME stored column the mass kernel consumes — counted once (§2.3).

---

## 2. Realized noise ≡ kernel width, exactly [RATIFY-R2 — key correctness gate]

### 2.1 The counted-once inheritance

Issue #40b (RATIFIED, `docs/derivations/hostz_pv_photoz_kernel.md` §3.1)
established the single-application-site invariant for widths: the stored
reduced-catalogue `z_error` is the TOTAL

$$\big(\sigma_{z,g}^{\mathrm{tot}}\big)^2 = \sigma_{z,\mathrm{meas(GLADE+)}}^2 \;+\; \sigma_{\mathrm{PV,cat(BORG)}}^2 \;+\; \Big[(1+z_g)\,\sigma_v^{\mathrm{class}}/c\Big]^2,$$

folded at parse time with σ_v^class = 150 km/s (BORG-corrected rows; Carrick
et al. 2015, arXiv:1504.04627, §4.2.1) or 500 km/s (uncorrected; Laghi et al.
2021, arXiv:2102.01708, §3) — `handler.py:344-389`, `constants.py:71-95`
(runtime `SIGMA_V_PEC_KM_S = 0.0`). Every inference-side consumer
(`host_z_error_eff`, windows, Z_g, D_g — `bayesian_statistics.py:3487-3512`)
reads this one column.

**Therefore the realization must draw ONE total Gaussian from the SAME
column:**

$$\delta z_g \sim \mathcal N\big(0,\ \texttt{z\_error}_g\big),\qquad z_{\mathrm{obs},g} = z_g + \delta z_g ,$$

NOT a re-derived component-wise sum. Then σ_realized = σ_kernel holds **per
row, identically, by construction** — the gate cannot be violated by a
constants drift, a PV-class reclassification, or a catalogue regeneration,
because both sides are the same number read once. Component-wise realization
(separate PV draw + measurement draw) is mathematically equivalent
(independent Gaussians in quadrature) but operationally dangerous: the PV
class split is **irrecoverable from the reduced CSV** (the PV-correction flag
and BORG σ_tot columns are dropped at parse; MEASURED, `r2` check F2: only
0.41% of rows — 34% of spec rows — are *provably* corrected-class from the
stored total alone), so any component re-derivation would have to re-parse
GLADE+.txt and re-implement the #40b classification, creating a second
application site — exactly what counted-once forbids.

### 2.2 Numerical verification (MEASURED — `r2_sigma_decomposition_check.py`)

On the production reduced CSV (22,634,764 valid rows):

- **F1 (schema integrity):** `z_error ≥ (1+z)·150 km/s/c` for **every** row
  (0 violations) — the #40b fold is present in the CSV on disk.
- **F2 (irrecoverability):** rows provably corrected-class: 0.41% overall,
  87,734/261,218 spec rows; everything else is ambiguous between
  {uncorrected} and {corrected + large measurement error}. Confirms: realize
  the total, not components.
- **F3 (identity):** MC (10^6 draws, representative spec/photo rows): sample
  std of both single-total and any component-split realization matches
  `z_error` to < 0.12% (MC noise). σ_realized = σ_kernel. ✓

### 2.3 The mass counterpart

Identical structure: realize ln M_obs with σ_lnM = `M_error/M` — the same
stored column the with-BH mass kernel (gaussian and trunc_lognormal,
`mass_marginal_2d_kernel.md` gates M1–M5) consumes as its width. Counted once
in M (M5) is inherited the same way. No new mass constants enter.

### 2.4 The z ≥ 0 boundary (sub-decision)

For photo rows at z ≲ 2σ_z the untruncated draw yields z_obs ≤ 0 with
non-negligible probability (e.g. z = 0.016, σ = 0.035 → P(z_obs < 0) ≈ 32%;
the golden-event venue is exactly there). Truncating/redrawing would break the
σ_realized = σ_kernel identity (realized law ≠ kernel law). Recommendation:
**clip z_obs to `GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT` = 1e-5** (a point mass
at the floor, not a redraw), and rely on the kernel's existing z ≥ 0 clamp +
volume prior (G2b clamp, `bayesian_statistics.py:3499-3510`) which already
kills the unphysical region on the inference side. This is a known,
documented approximation confined to photo rows at z ≲ 2σ (which carry ~no
information in the realistic run precisely because their kernel is that wide);
the spec rows that carry the information have z/σ_z ≳ 10 and are unaffected.
**ASSUMED; affected-fraction to be logged per realization.**

**[RATIFY-R2] Realization = one total draw per row from the stored error
columns (z: `z_error`; mass: `M_error/M` in ln space), with the 1e-5 z floor
clip.** *Recommendation: adopt.* This is the packet's key correctness gate:
σ_realized ≡ σ_kernel per row by construction, F1–F3 verified on the
production CSV.

---

## 3. The point kernel loses its licence [RATIFY-R3]

### 3.1 Formal statement

The production `generator_marginal` numerator is
N_g = p(x_GW | z_g, Ω_g[, M]) — the δ-kernel limit of the marginal

$$N_g(\sigma) = \int dz\; p(x_{\rm GW}\mid z,\Omega_g[,M],h)\; \frac{\mathcal N(z; z_{{\rm obs},g}, \sigma)\, w_{\rm pop}(z)}{Z_g},\qquad N_g = \lim_{\sigma\to 0} N_g(\sigma),$$

and its licence is exactly the unscattered premise documented at
`bayesian_statistics.py:3455-3474`: *"the generator draws hosts at their
catalogue z verbatim … no sigma_z scatter anywhere on the production path"*.
Under §1's model that premise is FALSE by design: given the observed row
(z_obs,g, σ_g), the truth is distributed as

$$p(z_{\rm true}\mid z_{{\rm obs},g}) \;\propto\; \mathcal N(z_{{\rm obs},g};\, z_{\rm true},\, \sigma_g)\; p_{\rm pop}(z_{\rm true}),$$

a width-σ_g object. Point-evaluating at z_obs,g is a model error of order
(σ_g · d ln d_L/dz) relative to the GW width σ_dL/d_L — for the golden events
a factor 25–110 (ledger §3) — and, worse, is *inconsistent*: the realized data
generating process is exactly the marginal the width kernel computes. The
δ-kernel is not "optimistic" under scatter, it is wrong.

### 3.2 Which width kernel

Candidates and status (all pre-existing in the repo or harness):

| Kernel | Definition | Status / evidence |
|---|---|---|
| `point` | δ(z − z_g) | **Licence lost under scatter** (this section). Baseline-mock only. |
| `bare` | N(z; z_g, σ) alone | pp_coverage harness: collapses coverage to ~0–3% at photo-z widths (module docstring, `validation/pp_coverage.py:25-28`) — the Jensen/Eddington asymmetry the bare Gaussian keeps. REJECT for production. |
| `volume` (harness) / `volume_deconv` (production) | N(z; z_g, σ)·w_pop(z)/Z_g, w_pop = dV_c/dz·(1+z)^{-1} | G2b-derived; pp_coverage-calibrated; ratified for real-data mode (#40b RATIFY-5). **The correct form**: it IS p(z_true | z_obs) of §3.1 with the population prior made explicit. |
| `volume_trunc` | volume kernel truncated at support edge | **FALSIFIED** (2026-07-12, `results/volume_trunc_ab_20260712/FINDING.md`; fixed_quad peak-aliasing). Do not revive. |

The §3.1 posterior identifies the deconvolution weight as *exactly* the local
truth prior of the forward model: hosts declared at catalogue rows, whose
coarse-grained z-density is (rate-weighted) volume-like — the same w_pop the
G2b derivation uses. So `volume_deconv` is not merely the best available; it
is the generator-consistent kernel *of the scattered generator*.

### 3.3 Normalization mode

`generator_marginal`'s selection leg was derived FOR the point/point pairing
(DERIVATION_GENERATOR_CONSISTENT_NORM.md §4.3). With scatter, the mock is
statistically a real-data configuration, and the ratified real-data mode
applies unchanged: **`absolute_marginal` normalization × `volume_deconv`
numerator with per-row (flag-resolved) widths** (#40b RATIFY-5; expressible
today via `--normalization_mode absolute_marginal --host_z_kernel
volume_deconv`, and `absolute_marginal` already keeps the volume_deconv
kernel, `bayesian_statistics.py:3529`). The selection/normalization leg keeps
the smooth dV_c/dz prior (the discrete kernel enters the numerator only —
consistent with every verified literature source, #40b §2). Σ_glob smearing by
σ_z remains an O((σ_z/z)²) diagnostic, not required (#40b §3.3).

### 3.4 What the prior-consistency guard must now enforce

Existing guards: (i) `resolve_host_mass_kernel` raises on point-z ×
trunc-mass (`bayesian_statistics.py:194-210`); (ii) `generator_marginal` +
`--smear_global_selection` raises (`:2007-2016`). New requirement: **the
evaluation must REFUSE a point-resolving host-z kernel — and refuse
`generator_marginal` altogether — whenever the loaded catalogue is a scattered
realization.** Mechanism (§6): the observed-catalogue realization carries a
sidecar metadata file; if `realization_sigma_scale > 0` and
`resolve_host_z_kernel(...) == "point"`, raise with the same
prior-consistency wording. Symmetrically, running the *unscattered* baseline
catalogue with the width mode stays allowed (it is merely conservative), so
the guard is one-directional — exactly like the existing point/point guard.

**[RATIFY-R3] Production choice for campaign #53: `absolute_marginal` +
`volume_deconv` per-row widths; guard extended to refuse point-kernel (and
`generator_marginal`) on any scattered-realization catalogue.**
*Recommendation: adopt.* `bare` rejected (measured coverage collapse),
`volume_trunc` rejected (falsified), point rejected (licence lost, §3.1).

---

## 4. Photometric hosts: the honest consequence and the options [RATIFY-R4]

### 4.1 The measured situation

- At the venue that carries the information (z < 0.15) the pruned catalogue is
  **97.4% photometric by rate weight** (MEASURED, `r1_flag_fractions.json`:
  rate-weighted spec fraction f_spec = 0.0265 at z < 0.15; strongly
  z-dependent — 0.275 in [0, 0.02), 0.109 in [0.02, 0.05), 0.041 in
  [0.05, 0.08), 0.005 in [0.10, 0.15)).
- Photo widths: median σ_z = 0.0365 at z < 0.15 (σ_z/z ≈ 50%); spec widths:
  median σ_z^tot = 0.0024 (incl. PV) [MEASURED].
- With σ_z/z ≈ 0.5 realized, a photo-z host is indistinguishable from the
  completion-term volume over most of the GW d_L posterior: in-catalogue host
  *identification* is essentially impossible for photo rows — their numerator
  approaches the smooth completion integrand and contributes ~no H0
  information (per-event I_e suppressed by (σ_dL/d_L)²/(σ_z·dlnd_L/dz)² ≈
  10^-3–10^-4 for the golden events). The earlier photo-z railing episodes
  (`handler.py:329-338` caveat) were this fact surfacing as pathology when the
  widths were used without the volume deconvolution.

### 4.2 The options

Let F = global in-catalogue fraction (`compute_global_catalog_fraction`,
injection side, `main.py:452-460`), Σ_glob the catalogue selection sum, and
"completion term" the (1−f)·p_pop leg (Gray et al. 2020 Eq. (9) structure).
Forecasts are MEASURED counterfactuals from the 76 seed-61000 golden hosts
(`r3_sigma_h0_forecast.py`; per-event Fisher I_e = (h·σ_eff)^-2, σ_eff² =
(σ_dL/d_L)² + (dln d_L/dz)²σ_z², exact dln d_L/dz from `dist()`; cross-check:
σ_z = 0 reproduces 0.027 vs 0.027–0.032 measured pipeline curvature ✓):

**(a) Catalogue as-is (all flags, realistic widths).**
No code-path change beyond §§1–3. F, Σ_glob, completion term all unchanged in
*form* (recomputed on the observed catalogue, §5.2). Photo hosts stay in the
in-catalogue term with their honest σ ≈ 0.036 kernels and contribute ~nothing;
spec hosts (f_spec of the draw) carry the information.
σ_H0(seed 61000's actual all-photo golden set) = **3.79 km/s/Mpc** [MEASURED
counterfactual]; expectation over host flags = option (c) below.

**(b) Spectroscopic-host analysis (flag-3-only in-catalogue term).**
The in-catalogue sum is restricted to flag-3 rows; photo rows are removed from
the *catalogue* and their probability mass migrates to the completion term.
Consequences, all mandatory for consistency: the completeness map m_th and
f(z) must be rebuilt from the flag-3 subcatalogue; F drops by roughly the
rate-weighted spec fraction (≈ ×0.03 at the venue — the dark fraction
1−F → ~0.999); Σ_glob and W_cat become flag-3 sums; the injection-side
mixture draw must use the SAME reduced catalogue (self-consistency, Chen et
al. 2024, arXiv:2212.08694) — i.e. **a new observed-catalogue definition and a
new completeness cache, and the host draw changes** → new simulation seeds
required (the current seeds drew photo hosts as signal sources; under (b)
those events are dark-by-definition, which is *consistent* to evaluate but
wastes the realized in-catalogue statistics).
Expected σ_H0 = **1.39–1.88 km/s/Mpc** (150/500 km/s PV variants; expectation
Σ_e f_spec(z_e)·I_e^spec with z-resolved f_spec; expected spec-host yield
3.4 of 76 golden events) [ESTIMATED from MEASURED inputs].

**(c) Hybrid: all flags, per-row (= per-flag) kernels.**
Structurally identical to (a) — the volume_deconv kernel already takes the
per-row stored width, so flag-1 rows automatically get ~0.036 kernels and
flag-3 rows ~0.0024 kernels; no selection-function redefinition, F/Σ_glob/
completion unchanged in form. The difference from (a) is only interpretive:
(c) is (a) evaluated in expectation over the host-flag mix instead of one
seed's all-photo accident. Expected
σ_H0 = **1.31–1.69 km/s/Mpc** (150/500 variants: spec events dominate, photo
events add the small opt-(a) tail) [ESTIMATED from MEASURED inputs].
Caveat (MEASURED): the expected spec yield is 3.4 hosts/seed (Poisson: seed
61000 drew 0 — P(0|3.4) ≈ 3%, low but not alarming; the top-3 loudest events
sit in shells with f_spec ≈ 0.11–0.27, `r3_forecast.json`), so the per-seed
σ_H0 fluctuates between the (a)-floor ≈ 3.8 (no spec hosts drawn) and
≈ 0.5–1 (a lucky loud spec host). Multi-seed aggregation is mandatory (§7.3).

**[RATIFY-R4] Photometric-host policy.** *Recommendation: adopt (c) — which is
(a) with the per-row widths the machinery already implements — and REJECT (b)
for the final production run.* Reasoning: (c) uses all information honestly
with zero selection-function surgery and zero new caches; (b) buys no
information (its σ forecast is within ~10% of (c), since photo hosts
contribute ~nothing either way) at the price of rebuilding F, m_th, Σ_glob,
the injection draw, and new GPU seeds — maximal consistency risk for no gain.
(b) remains the right template for a *future* "spectroscopic follow-up
scenario" forecast (the 0.22–0.30 dream number), quoted as a scenario, not as
the GLADE+ result. Author ratifies.

---

## 5. Selection-function consistency [RATIFY-R5]

### 5.1 Which legs need TRUE, which OBSERVED

The rule (Mandel, Farr & Gair 2019, arXiv:1809.02063): selection is a property
of the *data-generating process* — p_det legs integrate over TRUE quantities;
catalogue-composition legs describe what the analyst's catalogue contains —
they use OBSERVED rows.

| Leg | Quantity it averages | Needs | Under (A), changes? |
|---|---|---|---|
| Injection pool / `SimulationDetectionProbability` (1D, 2D, joint z×M_z survival) | p_det over true (z, M_z, d_L) | TRUE | **NO — pool remains valid unchanged.** The 200k pool (`results/campaign51_20260728/pool_mix200k`, `injection_pool_mix200k_20260728`) is a p_det object over injected = true coordinates; §1(A) leaves every injected coordinate untouched. No regeneration. |
| Completion-term p_det queries, Σ_glob p_det factors, D(h) | p_det at hypothesis (z, M_z(hypothesis)) | TRUE-coordinate *function*, queried at hypothesis values | NO (function unchanged; hypothesis-side query convention per `project_pdet_hypothesis_convention` memory) |
| Σ_glob / W_cat catalogue sums (which galaxies, at what z, M) | the analyst's catalogue rows | OBSERVED | YES — computed at evaluation time from the loaded catalogue, so they pick up the realized rows automatically once the evaluation loads the observed catalogue. No code change; provenance change only. |
| Completeness m_th map / f(z), F | catalogue B-magnitudes per pixel (+ z binning if any) | OBSERVED (it describes the catalogue) | **AUDIT ITEM**: B-mags are not scattered by this packet, but if `pixel_completeness.build_m_th_map` bins by z, the realized z moves rows across bins → the cached map (`from_cache_or_build`) would be stale per realization. If the map is z-independent (pure per-pixel magnitude threshold), the cache stays valid. Must be checked at implementation gate; if z-dependent, rebuild per realization (CPU, minutes). |
| Injection-side host draw + F (mixture split) | generative truth | TRUE | NO — already run with catalogue-as-truth; the existing seeds' draws ARE the (A)-model truth draws. |
| Ball-tree candidate search (sky ∩ z-window ∩ M-window) | the analyst's catalogue | OBSERVED | YES automatically (built from the loaded catalogue, `handler.py:508-541`). NEW physics: with realized scatter the TRUE host can fall outside the observed-row window (z window: ±1σ-level effect; M window: ±cutoff·M_error vs realized 1·σ offsets) → genuine host-loss events. This is realism, not a bug; the miss rate must be LOGGED per run (falsification input, §8). |

### 5.2 Consistency statement

Under (A) + option (c): the generative process (truth draw, waveform, SNR,
selection) is byte-identical to campaign #51; only the *catalogue file handed
to the evaluation* changes. Every observed-side object (numerator kernels,
Σ_glob, W_cat, ball-trees, completeness if z-independent) is derived at
runtime from that file, so consistency reduces to: (i) one observed catalogue
per (run, realization seed) used by ALL evaluation legs; (ii) p_det objects
keyed to the pool, not the catalogue. Both hold structurally in the current
code. **The injection pool and all CRB CSVs remain valid unchanged.**

### 5.3 Mass-edge note (from §1.3)

Forward mass scatter moves observed masses across the pruning window
[M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX] (`_get_pruned_galaxy_catalog`,
`handler.py:260-278` — the window is already ±M_error-inflated, which absorbs
most of it). Rows scattering OUT of the observed catalogue while their truth
generated a detection are, again, honest host-loss; rows scattering IN add
harmless candidates. Log the net row-count change per realization (expected
≲ 1% given the error-inflated window; MEASURED at implementation).

**[RATIFY-R5] Selection split: p_det legs = TRUE (pool unchanged, no
regeneration); catalogue-composition legs = OBSERVED (recomputed at load from
the realized catalogue); completeness-cache z-dependence is a pre-merge audit
item; host-loss rates logged.** *Recommendation: adopt.*

---

## 6. Reproducibility and the regression gate [RATIFY-R6]

### 6.1 Seeded observed-catalogue realization

Mechanism (mirrors `run_metadata.json` provenance):

1. New CLI stage/flag (implementation detail; e.g. `--realize_observed_catalogue
   --realization_seed S [--realization_sigma_scale 1.0]`) that reads the
   reduced catalogue, applies §2's draws with
   `rng = np.random.default_rng(S)`, and writes
   `observed_catalogue_seed{S}.csv` (same 8-column schema — z and M columns
   replaced by observed values; `z_error`, `M_error`, flag unchanged, since
   the widths describe the error law, which the observation realization does
   not alter) **plus a sidecar**
   `observed_catalogue_seed{S}.meta.json`: {realization_seed, sigma_scale,
   parent_csv sha256, own sha256, git_commit, timestamp, n_rows,
   n_z_floor_clipped, n_mass_window_crossings}.
2. The evaluation loads the observed catalogue via the existing path and
   FIRST reads the sidecar: missing sidecar ⇒ legacy/unscattered (baseline
   mode allowed); sigma_scale > 0 ⇒ §3.4 guard active (point kernel and
   `generator_marginal` refused).
3. Every posterior JSON records the sidecar hash — one realization, all legs,
   verifiable.

The realization is a pure function of (parent CSV, seed, sigma_scale): re-runs
are bit-reproducible, and different realization seeds on one CRB seed give the
observation-noise scatter of the estimator (distinct from the truth-seed
scatter — report both, §7.3).

### 6.2 Mandatory regression gate: σ → 0 bit-identity

With `sigma_scale = 0` the realization draws N(0, 0) = 0: the observed
catalogue equals the parent **bit-identically** (the writer must copy the
original string fields rather than round-trip floats — implementation
requirement, testable by sha256 equality of parent and child CSVs). Then the
full evaluation with the baseline mode (`generator_marginal`, point kernel —
permitted because sigma_scale = 0) must reproduce the campaign-#51 posteriors
**bit-identically** (same combined_posterior.json bytes). This is the
limiting case that ties campaign #53 to the validated baseline; it is a
REQUIRED pre-production test, not optional.

**[RATIFY-R6] Provenance mechanism (seeded realization + hashed sidecar +
guard wiring) and the σ→0 bit-identity regression gate.** *Recommendation:
adopt.*

---

## 7. Cost and campaign plan [RATIFY-R7]

### 7.1 Does d_L change per event? — the re-run verdict

**NO, under (A).** d_L = dist(z_g, h_inj) with z_g the catalogue truth —
unchanged from campaign #51 (`parameter_space.py:260-273`). Injected M_z =
M_g(1+z_g) — unchanged. Therefore every waveform, SNR, Fisher matrix and CRB
CSV, and the 200k injection pool, are **reused as-is**. The realistic run is:
realize observed catalogue (CPU, minutes) → prepare/evaluate on the h-grid
(CPU, the standard `--evaluate` cost) → combine. **CPU-only; ~0 GPU-h.** The
~250 GPU-h full re-simulation is required only under convention (B) (truth
reassignment moves z → d_L → waveforms), which §1 rejects, or under option
4(b) (host-draw redefinition), which §4 rejects. This cost asymmetry is a
consequence of the physics choice, not its justification — (A) wins §1.2 on
statistical grounds alone.

One caveat: the *prepare* step (`scripts/prepare_detections.py`) consumes the
CRB truth CSVs and realizes GW scatter — unchanged. The evaluation's candidate
identification runs against the observed catalogue (per realization), so
`posteriors` must be regenerated per (truth seed × realization seed × h-grid
point) — the standard CPU evaluation cost each.

### 7.2 Recommended campaign shape

Information now rests on ~3.4 expected spec hosts per 1590-event seed (§4.2c),
so seed-to-seed scatter is the dominant reporting risk:

- **Truth seeds:** the 2 existing full CRB seeds (61000, 62000) now; queue
  2–3 additional simulation seeds on the cluster opportunistically (GPU) to
  tighten the Poisson band — NOT blocking for first production.
- **Observation realizations:** 5 per truth seed (realization seeds e.g.
  {1..5}·10^4 + truth seed), quantifying the observation-noise scatter at
  fixed truth. CPU cost: 10 evaluations × h-grid ≈ current single-run
  evaluate cost ×10.
- **Truth values:** h_inj = 0.73 production; one alternative-truth arm
  (existing #39 blind-mock machinery) as the anti-tuning gate once the
  pipeline passes §6.2 and §8.
- **Event counts:** unchanged (the seeds' full detection sets).
- **Report:** per-seed MAP + σ, pooled across realizations; quote the
  realization-scatter and seed-scatter separately.

**[RATIFY-R7] Re-run verdict: CPU-only (existing CRBs + pool reused);
campaign shape 2 truth seeds × 5 observation realizations × h-grid, extra GPU
seeds opportunistic.** *Recommendation: adopt.*

---

## 8. Pre-registered predictions and falsification [RATIFY-R8]

Registered BEFORE any campaign-#53 evaluation (all from
`r3_forecast.json` + ledger §3; ESTIMATED from MEASURED inputs):

P1. **σ_H0 (recommended option c):** expectation 1.3–1.7 km/s/Mpc; per-seed
    range [0.5, 4.0] km/s/Mpc driven by the realized spec-host count
    (0 spec hosts → ≈ 3.8; one loud spec host → ≈ 0.5–1). A production σ_H0
    below 0.3 km/s/Mpc (the all-spec dream bound) at any seed/realization
    FALSIFIES the noise realization (a leak of the unscattered premise —
    e.g. kernel bypass or a truth column reaching the inference).
P2. **MAP behaviour:** MAP-on-truth-exactly is *no longer expected*; pulls
    |MAP − 0.73|/σ should be N(0,1)-consistent across seeds × realizations.
    Systematic |pull| > 2 across ≥ 6 of the 10 runs falsifies the kernel/
    normalization pairing (§3.3) and re-opens RATIFY-R3 (candidate cause:
    Eddington residual from the §1.2(A) population caveat).
P3. **In-catalogue/dark split:** in-catalogue events retain ≈ 100% of the
    curvature; dark-event contribution stays in [−5%, +5%] (was −1%). A
    large negative dark contribution indicates completion-term inconsistency
    with the realized catalogue (§5 audit item).
P4. **Golden-event demotion:** the 3 loudest events (all photo hosts in seed
    61000) must lose ≥ 95% of their per-event curvature share each
    [MEASURED baseline: 46% combined → predicted < 5% combined].
P5. **σ→0 gate:** bit-identity per §6.2 — hard pass/fail.
P6. **Host-loss rate (ball-tree misses of the true host):** predicted
    O(10–30%) for photo hosts (window ±1.5σ sky × z/M windows vs realized
    1σ offsets), ≲ 5% for spec hosts. A ~0% measured miss rate under scatter
    indicates the windows are silently consuming truth columns — falsifies
    the observed-catalogue plumbing.

**[RATIFY-R8] Pre-registered P1–P6 as the campaign acceptance/falsification
set.** *Recommendation: adopt; evaluate P1–P6 before any headline number is
quoted.*

---

## 9. Guard summary [RATIFY-R9]

New/extended hard guards, in one place:

1. sigma_scale > 0 (sidecar) ∧ resolved host-z kernel == "point" → raise.
2. sigma_scale > 0 ∧ normalization_mode == "generator_marginal" → raise
   (its selection leg's derivation premise is the unscattered generator).
3. sigma_scale == 0 → baseline modes permitted (one-directional guard).
4. Missing sidecar → treat as legacy baseline catalogue; log prominently.
5. Realization writer refuses to overwrite an existing
   observed_catalogue_seed{S}.csv with a different parent hash.

**[RATIFY-R9]** *Recommendation: adopt.*

---

## 10. Dimensional analysis

All realized quantities are dimensionless or carry catalogue units:
δz ~ N(0, z_error): z and σ_z dimensionless; PV term (1+z)·[km/s]/[km/s]
dimensionless ✓. ln M_obs: σ_lnM = M_error/M = [M_sun]/[M_sun] dimensionless;
M_obs in M_sun ✓. Forecast Fisher: σ_eff² = (σ_dL/d_L)² [dimensionless] +
(dln d_L/dz)²σ_z² [z^-1·z = dimensionless] ✓; I_e = (h σ_eff)^-2 dimensionless
in h; σ_H0 = 100·I_tot^{-1/2} km/s/Mpc ✓. Kernel densities unchanged from G2b
(§3.4 of #40b packet): p_g(z) in [z]^-1, Z_g in Mpc³ ✓.

## 11. Limiting cases (minimum set)

- **σ → 0 (sigma_scale = 0):** observed catalogue ≡ parent bit-identically;
  full pipeline reproduces campaign #51 bit-identically (§6.2, REQUIRED gate).
- **Spec-z host, σ_v → 0, σ_meas → 0:** z_obs → z_g; width kernel collapses to
  the point value (pinned by
  `test_volume_deconv_numerator_collapses_to_point_as_sigma_to_zero`).
- **f_spec → 1 (all-spec universe):** option (c) forecast → ledger scenario
  B/C (0.235/0.313 km/s/Mpc) — recovered by `r3` scenarios B/C ✓.
- **f_spec → 0:** option (c) → option (a) all-photo floor (3.79) ✓
  (r3: opt_c = f·I_spec + (1−f)·I_photo by construction).
- **h-independence of the realization:** z_obs, M_obs draws contain no h
  anywhere (catalogue columns + seeds only) — structurally exact; the
  realization cannot leak h into the likelihood.
- **GW-noise-only limit (σ_z realized but σ_dL → 0):** per-event σ_h/h →
  σ_z·dln d_L/dz — the PV floor of MacLeod & Hogan-type analyses; sane.

## 12. Literature table

| Source | Used for | Where |
|---|---|---|
| Dálya et al. 2022, arXiv:2110.06184 | GLADE+ z-flag semantics (0/1/2/3), photo/spec σ_z, BORG PV correction σ_tot (§2.2 Eq. 1) | §§1–2, widths and flag classes |
| Reines & Volonteri 2015, arXiv:1508.06274 | M*→M_BH mean relation + 0.24 dex intrinsic scatter (§4.1) | §1.3 mass error model |
| Carrick et al. 2015, arXiv:1504.04627 | σ_v = 150 km/s reconstruction residual for corrected hosts (§4.2.1) | §2.1 PV class widths (via #40b) |
| Fishbach et al. 2019, arXiv:1807.05667 | PV quadrature convention (§2.2); mixture-structure Eqs. (3)-(5) | §2.1, §3.3 |
| Laghi et al. 2021, arXiv:2102.01708 | 500 km/s full PV dispersion (§3); EMRI dark-siren σ_H0 scale 0.7–2 km/s/Mpc (consistency of §8 P1) | §2.1, §8 |
| Gray et al. 2020, arXiv:1908.06050 | catalogue/completion mixture Eq. (9); mock-from-real-catalogue precedent | §4.2, §1.2 |
| Chen et al. 2024, arXiv:2212.08694 | injection↔inference population self-consistency | §4.2(b), §5 |
| Davis et al. 2011, arXiv:1012.2912 | (1+z) PV factor, Eqs. (1)/(A1) | §1.2 |
| Mandel, Farr & Gair 2019, arXiv:1809.02063 | selection on TRUE quantities, single selection factor | §5.1 |

## 13. Scope table — every affected code path

| File:lines | Role today | Change under this packet |
|---|---|---|
| `master_thesis_code/galaxy_catalogue/handler.py:344-389` | parse-time counted-once σ_z fold | UNCHANGED (source of the totals the realization draws from) |
| `handler.py:411-460` (`read_reduced_galaxy_catalog`) | catalogue load | loads the OBSERVED catalogue file when provided; reads sidecar |
| `handler.py:260-278` (`_get_pruned_galaxy_catalog`) | mass/z pruning | operates on observed values → §5.3 edge-crossing logging |
| `handler.py:508-541`, `:800-870` (ball-trees, `get_possible_hosts`) | candidate search windows | operate on observed rows; host-miss rate logged (P6) |
| `handler.py:692-800` (`draw_rate_weighted_hosts`) | injection-side truth draw | UNCHANGED (truth = parent catalogue; existing seeds valid) |
| `handler.py:885-891`, `:1117-1131` (mass relation) | catalogue mass + M_error | UNCHANGED (σ_lnM source column) |
| `master_thesis_code/datamodels/parameter_space.py:260-273` | injection d_L, M_z from truth | UNCHANGED (the §7 CPU-only verdict rests here) |
| `master_thesis_code/main.py:374-470` (`data_simulation`, F) | simulation + injection mixture | UNCHANGED |
| `master_thesis_code/dark_siren_injection.py:556-650`, `:77-85` | mixture/dark draw | UNCHANGED (dark hosts never read by inference) |
| `master_thesis_code/bayesian_inference/bayesian_statistics.py:105-138` (`resolve_host_z_kernel`) | kernel resolution | guard extension §3.4/§9 (refuse point under scatter) |
| `bayesian_statistics.py:141-210` (`resolve_host_mass_kernel`) | mass-kernel guard | analogous scatter guard for the with-BH channel |
| `bayesian_statistics.py:1964-2026` (mode validation + guards) | normalization modes | add sidecar-aware refusal of `generator_marginal` under scatter |
| `bayesian_statistics.py:3455-3474` (point numerator) | δ-kernel path | licence restricted to sigma_scale = 0 catalogues |
| `bayesian_statistics.py:3487-3529` (σ_eff, kernel, windows) | width machinery | UNCHANGED — becomes load-bearing (production kernel) |
| `bayesian_statistics.py:3046-3110` (completion numerator) | completion term | UNCHANGED in form; P3 monitors consistency |
| `master_thesis_code/bayesian_inference/simulation_detection_probability.py:162-580` | p_det over true (z, M_z, d_L) | UNCHANGED; pool `injection_pool_mix200k_20260728` reused as-is (§5.1) |
| `master_thesis_code/galaxy_catalogue/pixel_completeness.py` (`from_cache_or_build`) | m_th completeness cache | AUDIT: z-dependence check (§5.1); rebuild per realization iff z-binned |
| `scripts/prepare_detections.py:115` → `datamodels/detection.py:161-240` | GW-side noise realization | UNCHANGED (already realistic, ledger I6) |
| `master_thesis_code/constants.py:69-111` | PV/σ constants | UNCHANGED (no new constants; counted-once inherited) |
| `master_thesis_code/arguments.py:527-539` (+ new flags) | CLI | new realization flags + sidecar plumbing |
| NEW: observed-catalogue realization script/stage | — | §6.1 writer (seeded, hashed, σ→0 = byte copy) |
| `master_thesis_code/validation/pp_coverage.py:421-546` | P–P harness (scatter + kernel switch precedent) | template for the P–P calibration arm of validation |

## 14. Supporting numerical checks (this packet's evidence)

`results/campaign51_20260728/realistic_model/`:

- `r1_catalog_flag_fractions.py` → `r1_flag_fractions.json` — pruned-catalogue
  spec/photo fractions, count and rate-weighted, at z-cuts and in z-shells;
  per-flag width statistics. Key numbers: f_spec(rate-weighted) = 0.0265 at
  z < 0.15 (0.275 / 0.109 / 0.041 in the [0,0.02)/[0.02,0.05)/[0.05,0.08)
  shells); spec σ_z^tot median 0.0024, photo 0.0365. [MEASURED]
- `r2_sigma_decomposition_check.py` → `r2_sigma_decomposition.json` — F1 floor
  0 violations/22.6M rows; F2 PV-split irrecoverable (0.41% provable);
  F3 MC identity pass. [MEASURED]
- `r3_sigma_h0_forecast.py` → `r3_forecast.json` — per-option forecasts from
  the seed-61000 golden hosts: A 0.027 (cross-check ✓), opt(a) 3.79,
  B 0.235 / C 0.313, opt(b) 1.39–1.88, opt(c) 1.31–1.69 km/s/Mpc; expected
  spec yield 3.4/76. [MEASURED counterfactual / ESTIMATED forecast]

## 15. Validation plan (after ratification, before production)

1. Implement realization writer + sidecar + guards (no default-path change;
   §6.2 bit-identity test in CI as a slow test).
2. σ→0 regression gate (P5) on seed 61000: bit-identical combined posterior.
3. Single-realization smoke run (seed 61000 × realization 1, coarse h-grid):
   check P4 (golden-event demotion) and P6 (host-miss logging) before the
   full grid.
4. Full campaign per §7.2; evaluate P1–P3 on the pooled result.
5. P–P/coverage arm: pp_coverage harness with scattered-z synthetic universes
   (kernel="volume") — the decisive calibration test that widths are
   calibrated, not just inflated (#40b §4.5 precedent).
6. Alternative-truth blind mock (#39) as the anti-tuning gate.

---

## Gate index

- **[RATIFY-R1]** Noise direction: catalogue = TRUE, observations realized
  forward (z and mass). Recommend adopt (A).
- **[RATIFY-R2]** Realized noise = ONE total draw from the stored width
  columns (σ_realized ≡ σ_kernel per row); z-floor clip at 1e-5. Recommend
  adopt. **Key correctness gate.**
- **[RATIFY-R3]** Production estimator: `absolute_marginal` ×
  `volume_deconv` per-row widths; point kernel + `generator_marginal`
  refused on scattered catalogues. Recommend adopt.
- **[RATIFY-R4]** Photo-host policy: option (c) hybrid per-row widths;
  reject (b) spec-only for production. Recommend adopt.
- **[RATIFY-R5]** Selection split TRUE/OBSERVED; injection pool and CRBs
  reused unchanged; completeness-cache z-dependence audit. Recommend adopt.
- **[RATIFY-R6]** Seeded observed-catalogue realization + hashed sidecar;
  σ→0 bit-identity regression gate. Recommend adopt.
- **[RATIFY-R7]** CPU-only re-run verdict; 2 truth seeds × 5 realizations
  campaign shape. Recommend adopt.
- **[RATIFY-R8]** Pre-registered predictions P1–P6. Recommend adopt.
- **[RATIFY-R9]** Guard set 1–5. Recommend adopt.
