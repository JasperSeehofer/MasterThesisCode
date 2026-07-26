# Adversarial mathematical review — EMRI dark-siren H0 production stack

**Date:** 2026-07-26 · **Reviewer role:** independent mathematical-soundness (redteam) reviewer
**Under review:** branch `physics/absolute-mass-marginal` @ `ce6338e`
(`generator_marginal` + `--pdet_z_resolved` production defaults)
**Scope ordered by the author:** derivation integrity of the generator-consistent normalization,
the z-resolved survival estimator, the combination step, h-dependence bookkeeping, and an
anti-tuning audit against the known truth `H = 0.73`.

**Verdict: SOUND-WITH-CAVEATS.**
**Anti-tuning verdict: NO EVIDENCE OF TUNING TO h = 0.73 in any estimator constant** — with two
structural qualifications recorded in §4.

The derivations are internally correct and the numerical anchors reproduce to machine precision.
Three findings are nonetheless load-bearing for the paper: the quoted bias/width numbers are not
supported by the h-grid actually used (F1); the closure is dominated by a *bundled* likelihood
change, not by the derived normalization (F2/F3); and the shipped denominator mixes two p_det
conventions in violation of the code's own guardrail (F4).

---

## 0. What I actually checked

| # | Object | Method | Result |
|---|---|---|---|
| C1 | Eqs. (1)→(3)–(5), `DERIVATION_GENERATOR_CONSISTENT_NORM.md` | independent re-derivation from the stated generative model | exact; see V1 |
| C2 | `generator_norm_Dgen_table.json` internal identities | recomputed all identities and log-slopes | exact to 0 rel. error; see V2 |
| C3 | h-bookkeeping of `f̄`, `V_f`, `n̂_w`, `C_NORM` | symbolic + numeric | correct; see V3, V4 |
| C4 | FIX-2 conditioning error | recomputed pooled vs stratum vs kernel survival on the real 50 000-row pool | reproduced; see V5 |
| C5 | Abramson adaptive-kernel weights | re-ran the estimator with and without the `1/σ_k` factor on the real pool | ≤1 % effect; see F9 |
| C6 | Production posteriors (5 seeds, both channels) | direct inspection of the combined-posterior JSONs and combine logs | see F1, F7 |
| C7 | Anti-tuning: literals, `constants.H` usage, commit chronology | grep of the inference path + `git log` timeline | see §4 |

Everything below cites `file:line` on the reviewed commit.

---

## 1. Findings — severity ranked

### F1 [CRITICAL for the claim] The production posterior is not resolved by the h-grid; the quoted bias and width are extrapolations, not measurements

**Claim under test.** `MULTISEED_READOUT_20260726.md` reports per-venue MAP ± σ
(e.g. seed1000 `0.7304 ± 0.00026`) and pre-registered criteria
"bias −0.00030 ± 0.00035 (t = −0.85) → PASS", "width χ² = 8.0 → curvature widths VALID".

**Evidence** (`results/campaign_phase2_runs/run_20260726_seed*_prodstack/simulations/posteriors/combined_posterior.json`;
grid is non-uniform: Δh = 0.005 on [0.65, 0.79], 0.01 outside):

| Seed | grid points with p/p_max > 1e-3 | lnP at peak ∓1 step | lnP at ∓2 steps | Gaussian prediction for ∓2 steps |
|---|---|---|---|---|
| 1000 | **1** | −206.0 / −150.8 | −511.0 / −468.2 | −824 / −603 |
| 2000 | **1** | −185.4 / −189.6 | −497.6 / −497.0 | −742 / −758 |
| 3000 | **1** | −214.3 / −269.2 | −713.9 / **−inf** | −857 / −1077 |
| 90000 | 9 | −1.8 / −3.0 | −5.0 / −1.6 | −7.2 / −12 |

Three consequences:

1. On the three deep venues the entire posterior mass sits inside **one** grid node. The two
   points used for the parabolic MAP/σ fit lie ≈ 200 ln units below the peak, i.e. at ≈ ±20 σ
   under the fit's own σ. A quadratic expansion of ln P is valid to O(1 σ).
2. The log-posterior is **demonstrably non-Gaussian**: for a Gaussian, ln P at 2 steps must be
   4× ln P at 1 step. Measured ratios are 2.5–3.3, i.e. much heavier tails than Gaussian. The
   implied σ grows monotonically with the offset used (seed1000: 0.00025 → 0.00032 → 0.00039
   from ±1, ±2, ±3 steps). There is no single curvature width to quote.
3. The reported sub-grid MAP offsets are therefore driven by *tail asymmetry*, not by peak
   position. Seed1000's +0.0004 offset is exactly `0.5·(y₋−y₊)/(y₋−2y₀+y₊)` applied to a 55-ln
   asymmetry measured 200 ln down; I reproduce the readout's 0.7304 from those numbers, which
   confirms the mechanism rather than the measurement.

**Why it matters.** Pre-registered criteria 1 and 2 are computed from quantities the data do not
determine. The bias t-test, its SEM, the empirical-scatter comparison (0.00071 vs 0.00026), and
the width χ² (8.0 vs crit 9.3) are all functions of the invalid parabola.

**Does it invalidate the bias-closure claim?** It invalidates the *quantitative* form
("no detectable bias at the 4×10⁻⁴ level, curvature widths valid"). It does **not** invalidate
the qualitative result: the MAP node is 0.730 in all four provenance-valid venues, both channels,
which supports `|bias| ≲ 0.0025` (half the local grid step) — a large and real improvement over
the 0.86 rail. **Remediation:** re-run seed1000 on a dense grid (Δh ≈ 5×10⁻⁵ over
[0.725, 0.735]) before any σ or bias number is quoted in the paper. This was already flagged as
caveat 1 in `v1_probe_genmarg/PROBE_RESULTS.md` and not executed before the campaign.

---

### F2 [HIGH] The rail cure is dominated by a bundled likelihood change, not by the derived normalization

`normalization_mode="generator_marginal"` does **three** things, not two:

1. `n̂_w = W_cat/V_f` replaces `n̄_w = Σ_glob/β_G` (`bayesian_statistics.py:2579-2600`);
2. `D_gen = Σ_glob_sel/n̂_w + β_Ḡ` replaces `D = β_G + β_Ḡ` (`:2768-2800`);
3. **`_use_generator_point`** (`:3073`, applied at `:3219-3232` and `:3393-3408`) replaces the
   volume-deconvolved host-z kernel by a **δ-kernel at the catalogue redshift**, i.e. it drops
   the photo-z Gaussian *and* the peculiar-velocity term `sigma_z_pv` (`:3089`) from the
   in-catalogue numerator.

Only (1) and (2) are the subject of `DERIVATION_GENERATOR_CONSISTENT_NORM.md` §6's
pre-registered prediction. That prediction was **+52 ln (4D) / +92 ln (3D): rail persists**.
Measured: **−898.8 ln** (`v1_probe_genmarg/PROBE_RESULTS.md`). The normalization layer itself
matched the packet to 3–4 digits (n̂_w 2.7317, dlnD_gen/dh −1.490, P̂ 0.1133, fallback slope
−0.027) — i.e. essentially **all ~950 ln of the movement is attributable to leg (3)**, which had
no prediction attached and is not a normalization change at all.

The project documents this honestly in PROBE_RESULTS.md §"Mechanism decomposition". The problem
is structural, not one of disclosure: legs (1)+(2) and leg (3) are not separately selectable, so
the A/B ledger cannot attribute the cure, and the narrative "generator-consistent normalization
cures the deep-venue rail" is not supported by the decomposition.

**Recommendation:** expose leg (3) behind its own flag (`--host_z_kernel {volume_deconv,point}`)
and re-run the 3-way A/B. Any paper text must attribute the cure to the δ-kernel.

---

### F3 [HIGH] Consequence of F2: the estimator now assumes exact host redshifts

Leg (3) is *generator-exact* — I verified the premise independently: the mock's host `(z, Ω, M)`
are catalogue rows verbatim (`handler.draw_rate_weighted_hosts`), so `p(x|g) = p(x|z_g)` with no
kernel is the correct likelihood **for this mock**. Mathematically legitimate.

But the physical consequence is that the analysis is handed σ_z ≡ 0 host redshifts. That is what
produces the extreme sharpness in F1 (ln P falling 200 units per 0.005 in h across ~3400 events).
A real dark-siren analysis has photo-z errors and peculiar velocities; this stack has neither.
Any statement of "EMRI dark-siren H0 precision" derived from this configuration must carry the
caveat that the redshift kernel was collapsed to a δ on the strength of mock self-consistency.
The estimator is no longer a dark-siren estimator in the redshift dimension — only in the sky
dimension.

Secondary evidence that this leg is fragile: `6dae9d3` (seed600) records three events driven to
**exact zero at all h** by the δ-kernel (host candidates 10–65 σ outside the GW window, with
f(z) = 1 ⇒ B_num = 0). With a kernel numerator these events were merely improbable; with the
δ they are vetoes. This failure mode couples directly to F7.

---

### F4 [HIGH] `D_gen` mixes two p_det conventions, in violation of the code's own guardrail

Shipped default: `_dgen_catalog_selection = "4d_exact"` (`bayesian_statistics.py:1530`, also the
constructor default `:1568`, **not** exposed in `arguments.py`).

Under that default, `D_gen = Σ_glob_wbh/n̂_w + β_Ḡ` where:

- `Σ_glob_wbh` (`:1340-1354`) is **isotropic** (`phi_iso = theta_iso = 0`) and uses the 2D
  `S(d_L|M_z)` grid, which `DERIVATION_ZRESOLVED_SURVIVAL.md` §5.1 explicitly leaves **pooled in
  z**;
- `β_Ḡ` in the same sum is **sky-aware and z-conditional** under FIX-2.

The repository states the requirement itself, at `precompute_global_catalog_selection:1355-1363`:
*"using the IDENTICAL flat per-band survival that D(h) and beta_Gbar use … so p_det(Omega) is ONE
shared object across all selection integrals. Otherwise the p_det convention would not cancel in
beta_G/Sigma_global and would rescale the in-catalogue channel weight."* Under 4d_exact that
guardrail is not satisfied, in both the sky and the z dimension.

Lever size (from the packets' own tables): `Σ_glob_wbh/Σ_glob = 0.556`; `P̂(cat|det)` 0.113
(4D) vs 0.187 (3D); `dlnD_gen/dh` −1.49 vs −1.68. A 0.19/h denominator-slope difference over
3454 events across the prior span is O(100 ln) — same order as the effects the whole arc was
chasing. `DERIVATION_GENERATOR_CONSISTENT_NORM.md` §7.1 recommended (i) *shared-3D* "initially"
and warned that adopting (ii) *"before FIX-2 unifies the estimator would import that bias into
the channel balance"*. The implementation shipped (ii) with no recorded rationale, and there is
no pre-registered gap prediction anywhere for the actually-shipped combination (4d_exact +
FIX-2); the packets tabulate 3D+FIX-2 (−21 ln) and z×M_z+FIX-2 (−79 ln) but not this one.

**Mitigating:** the choice was frozen in `8fbb21e` at 10:51, *before* the first deep-venue probe
at 11:29 — it is not post-hoc selection. **Aggravating:** the alternative is unreachable from the
CLI, so no reviewer or future run can A/B it without editing code.

*Note on what is* not *wrong here:* using a mass-conditioned P_det for catalogue hosts and a
mass-marginal one for dark hosts is correct — catalogue hosts have known masses, dark hosts do
not. The defect is the **z (and sky) conditioning asymmetry**, not the mass conditioning.

---

### F5 [MEDIUM] The candidate-ball radius became physically load-bearing and is unquantified

`sigma_multiplier=1.5` at `bayesian_statistics.py:2326`; the ball is a circle of radius
`1.5·√λ_max(Σ_sky)` (`handler.py:455-463`) plus 1.5σ cuts in z and M_z.

Under the previous `volume_deconv`/`local_ratio` modes the numerator was a *self-normalized*
ratio `Σ_ball wN / Σ_ball wD_g`, so the truncation largely cancelled. Under
`absolute_marginal`/`generator_marginal`, `A_i = Σ_ball w_g N_g / n̂_w` is an **absolute** mass
added to an **untruncated** `B_num`. The truncation therefore biases the mixture weight
`λ_i = A_i/(A_i+B_num)` directly — the exact quantity the entire estimator-redesign arc was
about. Along the major axis of the sky ellipse the retained Gaussian mass is ≈ 87 %; the
discarded fraction is event-dependent (it depends on the ellipse aspect ratio) and one-sided
(always reduces A_i).

`DERIVATION_GENERATOR_CONSISTENT_NORM.md` §2.1 dismisses this in one clause ("The catalogue sum
self-truncates to the candidate ball … unchanged") without quantification. `grep -rl
sigma_multiplier` returns no test, no derivation, and no convergence study.
**Recommendation:** a cheap convergence run at `sigma_multiplier ∈ {1.5, 2.5, 4}` on one venue;
report `Δ ln L(h)` and the change in `λ̄`.

---

### F6 [MEDIUM] An analysis-time selection cut is absent from the selection function

`use_detection` (`bayesian_statistics.py:2847-2861`) drops any event with
`σ_dL/d_L ≥ FRACTIONAL_LUMINOSITY_DISTANCE_ERROR_THRESHOLD = 0.10` (`:72`). Seed1000 logs
"Loaded 3470 detections" and the campaign analyses 3454 ⇒ 16 events (0.46 %) removed.

`α(h)`/`D_gen` model the selection as "SNR ≥ 20" only. The actual selection is
`{SNR ≥ 20} ∩ {σ_dL/d_L < 0.10}`. This is a genuine selection-function mis-specification: the
cut is on a data quantity correlated with d_L and SNR, so it carries h-dependence. Neither
derivation packet mentions it. Small (0.5 % of events) but unaccounted — it should either be
folded into the survival estimator (the injection pool has SNR, so a matched cut is
constructible) or explicitly bounded.

---

### F7 [MEDIUM] Silent linear-space underflow, then a non-derived "physics floor"

Per-event likelihoods are stored linearly in the per-h JSONs and combined afterwards.

- seed3000, combine log: `Physics floor: event row 685: floored 25 of 41 bins with value
  5.150844e-273` — i.e. 25 h-bins of one event had truly underflowed to exact 0 (values below
  ~1e-308) and were silently reinstated.
- seed2000: 1 bin floored. seed1000: none.
- The *combined* posteriors underflow too: seed3000 has `p = 0.0` exactly at 2 grid steps above
  the peak (F1 table).

`_physics_floor` (`posterior_combination.py:219-275`) replaces a zero by the **per-event minimum
nonzero** likelihood. That is the most generous admissible substitute, not a conservative or
derived one; it converts a hard veto (−∞) into a finite, flat contribution and therefore biases
`Σ log L` **upward at exactly the h-values where the event was most incompatible**. The sibling
`_per_event_floor` uses `min/100` — also arbitrary. The name "physics-floor" overstates the
provenance: no physical scale (e.g. the numerical underflow scale, or a prior-predictive lower
bound) enters.

Measured impact on the reported runs is small (0/1/25 floored bins across 3 deep venues,
2 all-zero events excluded in the bh_mass channel). But the mechanism scales with N_events, with
grid width, and with the δ-kernel sharpness of F3, and the readout reports `n_empty` (all-zero
events) while **not** reporting the floored-bin count, which is the quantity that biases the sum.

---

### F8 [LOW-MEDIUM] Limiting case (d) of FIX-3 is approximate, not algebraic

The commit message for `8fbb21e` and `DERIVATION_GENERATOR_CONSISTENT_NORM.md` §5(d) assert an
**algebraic identity** in the `p_det → 1` limit: `Σ_glob → W_cat ⇒ D_gen → V_f + β_Ḡ = D`.
`W_cat` sums galaxies at `z < HOST_DRAW_Z_MAX = 1.5` (`:1000`) while `Σ_glob` is restricted to
`z < z_max(h)` (`:1301-1310`), which is ≈ 0.992 at h = 0.73 and shrinks with h. The identity is
therefore an approximation, inert only because 9 of 9 060 017 pruned catalogue rows lie above
z = 0.992. It should be stated as "inert given the catalogue's support", not as an identity —
otherwise a future catalogue extension silently breaks the seed600 shallow gate's logic.

---

### F9 [LOW] The adaptive kernel omits the `1/σ_k` factor required by the cited Abramson estimator

`simulation_detection_probability.py:_suffix_tables` (inside `_build_zres_survival`, `:691-708`)
forms `w = exp(-0.5·((u_k − u)/σ_k)²)` with **sample-point** bandwidths `σ_k` and no `1/σ_k`
normalization. Abramson (1982) sample-smoothing requires `(1/σ_k)·K((u−u_k)/σ_k)`; the factor
does not cancel in the ratio because `σ_k` varies over k (measured λ range on the real pool:
0.755–24.1). As written, injections in sparse (low-z) regions are over-weighted by ≈ λ_k.

**I measured it rather than asserting it.** Re-running both estimators on the real 50 000-row
pool at the pool's own d_L(z; 0.73):

| z | 0.050 | 0.125 | 0.225 | 0.325 | 0.475 | 0.625 | 0.775 | 1.000 |
|---|---|---|---|---|---|---|---|---|
| rel. difference (code − Abramson)/Abramson | +0.36 % | −0.90 % | +0.28 % | −0.51 % | −0.09 % | +0.17 % | +0.09 % | +1.89 % |

Numerically negligible. This is a citation/specification defect, not a physics defect. Fix the
weights or amend the docstring and the packet's §3.3 claim.

---

### F10 [LOW, but relevant to the closure test's strength] The h-grid is truth-centred

`_DEFAULT_H_PRIOR_MIN = 0.60`, `_DEFAULT_H_PRIOR_MAX = 0.86`
(`simulation_detection_probability.py:87-88`) is exactly symmetric about 0.73, and the refined
Δh = 0.005 sub-grid spans [0.65, 0.79] (centre 0.72). Truth also lands exactly on a node.

This is not a leak into the estimator — nothing in the likelihood depends on it — but it weakens
the closure test: any symmetric flattening of the likelihood, or any residual that is even in
(h − 0.73), returns MAP ≈ truth by construction, and the previous railing behaviour was
specifically a grid-edge pathology of this symmetric grid. A stronger test uses a prior interval
deliberately asymmetric about truth (e.g. [0.62, 0.95]).

---

## 2. What checked out — positive findings

**V1 — The generator-consistent marginal is Bayes-correct and I reproduced the algebra.**
Starting from the generator (Bernoulli(F) channel split; in-catalogue draw `w_g/W_cat` with
catalogue values verbatim; dark draw `∝ (1−f_k)p_pop` per pixel), the mixture likelihood
`p(x|h,det) = [F/W_cat·Σ w_g N_g + (1−F)/W_dark·B_num]/α` with
`α = F·Σ_glob/W_cat + (1−F)·β_Ḡ/W_dark` is the correct MFG (arXiv:1809.02063) selection-
conditioned form for this generator. Multiplying through by `V_tot` with `F = V_f/V_tot` and
`1−F = W_dark/V_tot` gives Eqs. (3)–(5) exactly; `V_tot`, `F`, `W_dark` cancel identically.
Numerator and denominator use the **same** hypothesis space, the **same** `w_g`, and the **same**
`P_det(g)`. Dimensions check throughout.

I also verified the 4π bookkeeping independently, since it is the classic failure mode here:
`B_num` carries `sinθ_det/(4π)` (isotropic sky marginal for unknown dark-host directions,
`:2696-2714`) while `A_i` evaluates the sky Gaussian at the galaxy's known Ω with no 1/4π. That
asymmetry is *correct* — and it pairs with a sky-averaged `β_Ḡ`, so the convention is consistent
on both sides of the ratio.

**V2 — Numeric identities exact.** On `generator_norm_Dgen_table.json` (41-h grid) I recomputed:
`n̂_w − W_cat/V_f` → max rel. residual **0.0**; `D_gen − (Σ_glob/n̂_w + β_Ḡ)` → **0.0**;
`D − (β_G + β_Ḡ)` → **0.0**; `V_f(h)·h³` constant to **0.0**;
`dln n̂_w/dh = 4.109653` vs `3/h = 4.109589` (6e-6 rel., finite-difference limited);
`n̄_w/n̂_w = 1.3336` and `dln D_gen/dh = −1.6772` reproduce the packet's quoted numbers.

**V3 — `f̄`'s h-independence is derived, not an artifact.** `M_* = −19.7 + 5 log₁₀h` and
`M_th = m_th − 25 − 5 log₁₀ d_L(z,h) − K(z)` with `d_L ∝ 1/h`: the `5 log₁₀h` terms cancel
exactly in `x_th = 10^{0.4(M_*−M_th)}` (`pixel_completeness.py:12-26, 161`). Hence `f_k` is
h-free, `V_f ∝ h⁻³` exactly, and `F` is constant — all three follow, they are not assumptions.
This is the single most important piece of h-bookkeeping in the stack and it is right.

**V4 — `C_NORM` (calibrated at h = 0.73, `emri_rate.py:58-68`) cancels identically.** `w_g`
enters `Σ_ball w N`, `W_cat` and `Σ_glob` homogeneously; `A_i = Σ_ball wN·V_f/W_cat` and
`Σ_glob/n̂_w = Σ_glob·V_f/W_cat` are both first-order-homogeneous in `w`, so the calibration
scale drops out of `p_i`. This is a legitimate generation-side use of the truth cosmology.

**V5 — FIX-2's motivating measurement independently reproduced.** On the real 50 000-injection
pool I recomputed the pool's own `SNR ≥ 20` rate per 0.05-z stratum against the pooled survival
at the same d_L: p_true 0.613 vs pooled 0.809 at z = 0.125 (+32 %), 0.332 vs 0.473 at z = 0.325
(+42 %), 0.194 vs 0.275 at z = 0.475 (+42 %), converging at z ≳ 0.775. The u-kernel conditional
recovers p_true to a few per cent at z ≥ 0.2. The conditioning error is real, its sign is as
claimed, and the estimator repairs it.

**V6 — h-invariance of `S(d_L|z)` holds.** `d_hor_k = SNR_k·d_L_k/ρ_thr` and `u_k = ln(1+z_k)`
are both properties of the injection, independent of h; h enters only the query `d_L(z;h)`
(`:355`, `_zres_z_kwargs` call sites). Build-once is legitimate and no new h³ channel appears.

**V7 — Hypothesis-frame p_det convention respected.** `Σ_glob_wbh` queries
`M_z = M_g·(1+z_g)` (`:1346`) and the smeared path `M_g·(1+z_nodes)` (`:1157`) — the *hypothesis*
mass, not the observed detector-frame `det.M`. This matches the project's own stated convention.

**V8 — Kernel coordinate `u = ln(1+z)` is derived, not chosen.** The detector-frame lifts
`M_z = M(1+z)`, `f_obs = f_src/(1+z)` are multiplicative in (1+z), so a z-shift acts as a
translation in u; a single-bandwidth kernel is correct precisely in the translation coordinate.
Scott's `N^{-1/5}` and Abramson's √-law fix the bandwidth from the pool's own statistics.
Bandwidth-insensitivity was demonstrated over ×0.5–×2 in the packet. No free constant.

---

## 3. Approximation and consistency notes

- **`B_num` truncation:** the 4σ GW window (`:2643-2668`) truncates a rapidly-decaying integrand;
  tail mass ~6e-5. Benign. The `min(z_upper, redshift_upper_limit)` cap is domain-matched to
  `D_gen`'s cap per `f29a5e7`. Verified consistent.
- **`fixed_quad` orders:** `n=50` for B_num and the per-host integrals, `n=100` for D(h). Not
  re-verified for convergence here; the δ-kernel (F3) removed the narrow-peak aliasing risk from
  the *numerator*, but `B_num`'s integrand is a 4σ Gaussian sampled at 50 Gauss-Legendre nodes,
  which is comfortable.
- **`ESS` floor:** `_MIN_BAND_INJECTIONS = 10` is reused as an ESS threshold for (band × z) cells
  (`:743-746`). Reusing a raw-count convention as an ESS convention is a convention, not a
  derivation — but it is conservative and its failure mode is visible (seed900: 57.6 % of cells
  below floor, and that venue railed; seed90000 with the canonical pool: 0/726). The
  fallback-to-band-marginal policy (rather than to fully pooled) is the right choice and matches
  the packet.
- **`_ZRES_PILOT_DENSITY_FLOOR = 1e-12`, `_ZRES_PILOT_BINS = 400`, `_ZRES_U_NODES = 121`:**
  numerical hygiene constants; none is truth-referenced, and the packet's bandwidth sweep bounds
  their effect.

---

## 4. Anti-tuning audit

**Verdict: no estimator constant, threshold, bandwidth, floor, grid choice or branch in the
inference path can be shown to depend on knowing h = 0.73.** Specifically:

1. **Literal search.** No `0.73` literal exists anywhere in the inference path. The only
   occurrences in `master_thesis_code/` are: generation-side (`emri_rate.py:58` calibration
   comment, `cosmological_model.py:366/382` fiducial), plotting/report truth markers
   (`evaluation_report.py:224`, `plotting/*`), and `main.py` file-naming for `h_0p73` injection
   CSVs.
2. **`constants.H` in the inference package** is used in exactly one place —
   `bayesian_statistics.py:4328`, `sigma_dev = (h_mean − H)/sigma`, a post-hoc summary
   diagnostic. It does not enter any likelihood, denominator or p_det.
3. **`C_NORM`**, the one estimator-visible constant calibrated at h = 0.73, cancels
   algebraically (V4).
4. **The completeness map's h-independence** is a derived cancellation, not a fit (V3).
5. **`d_hor`** is h-free by construction and the survival tables carry no h (V6).
6. **Default-argument leak check:** every `dist/dist_vectorized/dist_to_redshift/
   comoving_volume_element` call in the inference path passes `h=` explicitly. The one bare
   `dist(self.redshift)` (`handler.py:55`, `ParameterSample.get_distance`) is generation-side and
   not reachable from `--evaluate`.
7. **Commit chronology is clean.** `git log` shows: derivation packets → implementation
   `8fbb21e` 10:51 (4d_exact **and** point/point frozen here) → first deep-venue probe `fb361e8`
   11:29 → FIX-2 `a608c4f` 11:45 → stacked probe `b9a097e` 12:41 → criteria pre-registered
   `f9a2a71` 13:46 → campaign submitted `c5047ff` 14:23 → readout `cbffbfc` 14:48 → defaults
   flipped `ce6338e` 19:44. **No commit changes an estimator constant after a readout.** The
   pre-registration of the multi-seed criteria genuinely precedes submission, and the seed900
   post-hoc exclusion is disclosed as post-hoc and justified from build-log ESS diagnostics
   (input provenance), not from the posterior value. That is the right way to do it.

**Two structural qualifications** — not tuning, but they limit how much the closure proves:

- **(Q1) Arc-level selection among derivable-but-optional degrees of freedom happened with the
  truth signal visible.** Within 19 hours, three estimator variants were tried, each judged by
  whether the deep venue rails. Individually each change is derivable; collectively, the *set of
  optional choices left on the table* was pruned against truth-closure. Sharpest instance: the
  **z×M_z catalogue term** is at least as generator-exact as what shipped (the generator does
  detect catalogue hosts at their own `(z_g, M_z,g)`), it is implemented and its ESS audit
  exists, and `DERIVATION_ZRESOLVED_SURVIVAL.md` §5.2/§6 measures that adopting it would move the
  gap by **−58 ln** — i.e. away from closure. It was deferred (§5.2, §8 risk 4). The deferral has
  a stated non-truth rationale (unify the estimator first), and it was decided in the packet
  before the probe, so I do **not** call it tuning. But the paper cannot claim the shipped
  conditioning convention was selected on correctness grounds alone: the most generator-exact
  option is the one not taken, and the reason it stayed untaken should be stated explicitly.
- **(Q2) seed1000 is the development venue.** All probes, decompositions and slope
  bookkeeping were done on it. Of the five campaign venues, one is invalid (seed900), one is the
  development venue (seed1000), one has 20 events (seed90000). The genuinely out-of-sample deep
  evidence is **seeds 2000 and 3000**. That is a real result, but "4 seeds pass" overstates the
  independence of the sample.

---

## 5. Explicitly NOT checked (unchecked risk areas)

1. **Whether the shipped stack's ln-likelihood gap matches any prediction.** No prediction exists
   for 4d_exact + FIX-2 (F4). I did not construct one.
2. **Peak width and shape.** Cannot be assessed on the existing grid (F1). Everything about σ,
   coverage, and calibration of the reported uncertainty is unverified.
3. **The `_bh_mass_denominator_inner_m_integral` erf-sum** (`:2860-2930`): I read the derivation
   (piecewise-linear × Gaussian, Owen 1980 moments) and it is structurally right, but I did not
   numerically validate it against quadrature.
4. **Fisher-matrix / CRB provenance.** Out of scope; the ecliptic-frame migration footgun noted in
   project memory was not re-checked for these runs.
5. **`fixed_quad` convergence** at n=50/64/100 for the shipped mode.
6. **Sky-band construction and `_build_grid_2d`** were read but not independently recomputed.
7. **The P–P / coverage harness** (`validation/pp_coverage.py`) was not exercised; it is the only
   instrument that could test the width claim of F1 from first principles and it has not been run
   against the `generator_marginal` mixture (flagged as gate 4 in the FIX-3 packet, still open).
8. **seed900 re-run** (ordered, not executed) — the registered n = 5 test remains uncomputed.

---

## 6. Bottom line

| Question | Verdict |
|---|---|
| Is the generator-consistent normalization Bayes-correct? | **Yes** — re-derived independently; numerator/denominator share hypothesis space, weights and p_det; dimensions and h³ bookkeeping exact (V1–V4). |
| Is `S(d_L\|z)` a consistent selection estimator? | **Yes in structure**; the conditioning error it fixes is real and I reproduced it (V5). One specification defect (missing `1/σ_k`, F9) measured at <1 %. No importance weight is dropped: the pool is drawn from the population and conditioning on z removes the marginal's h-irrelevant shape. |
| Does the combination step bias `Σ log L`? | **Yes, in principle** — the "physics floor" is anti-conservative and not derived (F7). Measured impact on these runs is small but the floored-bin count is unreported. |
| Any h-independent / doubly-h-dependent term? | **None found.** The one candidate (`f̄` h-invariance) is a genuine derived cancellation (V3). |
| Was closure achieved by tuning to h = 0.73? | **No.** No truth-dependent constant exists in the estimator; the spec was frozen before the first measurement; pre-registration precedes submission. See Q1/Q2 for the honest limits. |
| Is the closure achieved *by the derivation the commits claim*? | **No** — ~95 % of the movement comes from the bundled δ-kernel leg, not from `n̂_w`/`D_gen` (F2). |
| Is the quoted bias/width result supported? | **No** (F1). The supported claim is `\|bias\| ≲ 0.0025` from MAP-node agreement, not `−0.0003 ± 0.0004`. |

**Overall: SOUND-WITH-CAVEATS.** The mathematics of the two derivation packets is correct and
unusually well-anchored. What is not yet sound is the *evidentiary chain from those derivations to
the reported numbers*: the posterior is unresolved on the production grid (F1), the cure is
attributable to a different change than the one being adopted (F2/F3), and the shipped denominator
convention is internally inconsistent in its p_det conditioning (F4). None of these is tuning;
all three are blockers for the quantitative claims in a paper.

**Minimum remediation before publication:** (a) dense-grid re-run for F1; (b) separate the
δ-kernel behind its own flag and re-decompose for F2; (c) resolve or justify the 4d_exact/FIX-2
conditioning mix and expose it on the CLI for F4; (d) report floored-bin counts alongside
`n_empty` for F7; (e) state the exact-redshift caveat wherever precision is quoted (F3).
