# B8.2 [CAL] — DESIGN NOTE: the two-channel calibration harness ([A3]) that measures the dilution factor F

`launched under rows #222/#223 — charter node B8.2`

**Class:** design note (top tier, no code, no run). Append-only. Every number carries
`{value; source file:line; date}` (A11). This note is the input to a stage-2 registration
(§8 stage S4); nothing here is a registered band until that registration is authored.

**Bottom line (one paragraph).** B8.1 left one unmeasured symbol in the width clause of the
stop condition: `σ_h,measured ≤ F · σ_h,floor`, `F` = the width dilution that multi-candidate
balls + the completion leg impose on an estimator that is *internally consistent* with its
data (the placeholder was `F = 10`). This note designs the instrument that turns `F` into a
measured number at production N with the production candidate-count density, both channels,
plus the two stage-4 legs that make its verdict admissible (coverage AND the absolute-count
audit). The design choice that carries everything: **the estimator is production's own
`BayesianStatistics.evaluate()` reached through the mirror's `run_mirror_seed_inprocess`
(`darksiren_emri/validation/correspondence_1d.py:2734`), and the truth is the estimator's own
generative law — the b0i `catalogue_selected` draw for catalogue-hosted events and the bsel
`population_selected` draw for dark events, mixed at the estimator's own class weight** — so
the candidate-count density is not a synthetic input but *emerges* from real catalogue hosts,
real Fisher sky rows and production's own ball construction, and is checked against the banked
census as an acceptance test. `pp_coverage.py` is NOT the estimator here (it is a from-scratch
re-implementation, single-power-law `phi`, toy cosmology, synthetic catalogue —
`darksiren_emri/validation/pp_coverage.py:13-16, 278-286, 1578-1580`); it is retained as the
demonstrably-sensitive *control* and as the source of the harness's coverage statistics.
Honest cost: the docket's "≈ 6 CPU-h per 24-arm sweep" anchor is a **mirror-N (≈ 106–200
scored events) number**; at production N = 1588 with ≥ 100 universes the mandatory cells are
**≈ 130–475 CPU-h local (13–46 h wall at 14 cores)**, bracketed because the N-scaling of one
`evaluate()` call is UNMEASURED (§6) — the first plan stage measures it.

**Exoneration check (standing rule 5, mechanism-grepped in both layers).**
`EXONERATION_REGISTER_20260827.md:485-497` §13 **[INFO-STARVATION] — OVERTURNED**: "a property
of prior-INCONSISTENT estimators, not of the data … Do NOT resurrect it as an explanation" for
the rail. `BIAS_HISTORY_LEDGER.md` §2 (`:127-155`) carries the same reversal (row 41 → row 52).
This harness is built on the opposite premise: it measures the width of a prior-CONSISTENT
estimator (§2), and its output `F` is a **dilution factor on the WIDTH**, never a cause of the
**centering** failure (38–68 floor-σ, B8.1 §3) — §5 forbids the word "starved" in any verdict
this harness produces and defines the only admissible vocabulary ("dilution-limited at F").
Adjacent items grepped: §14 [SPECZ-RESCUE] (not invoked); row 86 (`BIAS_HISTORY_LEDGER.md:106`,
"harness has one candidate/event so it cannot exercise" — the reason [A3](iii) exists; this
design satisfies it by construction); row 98 (`:123`, DS-6/T2: in the multi-candidate ball venue
the in-loop defect is "a coverage collapse, not a rail", delta-narrow posteriors `post_sd
0.0012–0.0059`) and row 99 (`:373-383`, venue transfer TRANSFER-CONFIRMED at N = 400: coverage
0/0/0, `post_sd` median 0.004376, bias +0.037237 ± 0.000230) — both are *narrower-than-floor*
precedents the width bands of §4 are two-sided against.

---

## 0. Inputs of record (what this note was designed against)

| object | value / fact | source | date |
|---|---|---|---|
| Single-host, no-impostor Fisher floor, both channels, realistic host errors, N = 1588 | `σ_h,floor = 0.001747` (0.239 % of h); 2D at σ_M = 1.99 identical to 4 s.f.; 2D at σ_M = 0.02: 0.001295 | `b8_information_floor.json:oneD/twoD.*.closed_form`; `B8_1_CAL_FLOOR_RECORD.md` §2.1 | 2026-08-29 |
| Centering band (L7) | `3σ_floor = 0.0052` | `SYNTHESIS_DOCKET_1_20260829.md` §3 L7 | 2026-08-29 |
| HEAD production posteriors (iiib/joint_r1 average) | 2D ⟨σ_h⟩ 0.01847, ⟨bias⟩ −0.0668 (width/floor 10.6×); 1D ⟨σ_h⟩ 0.00996 with MAP pinned at 0.600 (width NOT a fair comparison) | `head_readout_extraction_20260827.md` via `B8_1_CAL_FLOOR_RECORD.md` §3 | 2026-08-27 |
| Production z-distribution, N = 1588 | median z 0.490, p10–p90 0.237–0.736, max 1.02 | `b8_information_floor.json:meta`; `B8_1_CAL_FLOOR_RECORD.md` §0 | 2026-08-29 |
| Production non-empty-ball fraction at h = 0.73 (iiib) | `L_cat_no_bh > 0` for **0.6184** of 1588 events; `L_cat_with_bh > 0` for **0.6184** (same set); `w_tilde_G` = **0.0620** (constant per event: the estimator's own catalogue-class weight) | `headreadout_20260827/iiib/event_likelihoods.csv` (this note's read, pandas, h = 0.73 rows) | 2026-08-29 |
| Mirror-fleet candidate-count census (fc/ft arms, 12 seeds, 2152 events per arm, 174–188 scored per seed) | no-BH `n_cand`: mean 733, median 19, p90 1681, max 29 521, **zero for 33.7 %**; with-BH (mass-window passers): mean 84.7, median 0, p90 86, **zero for 74.0 %**; z_true quantiles p10/p25/p50/p75/p90 = 0.268/0.358/0.459/0.584/0.704 | `b4_imp_stage1_events.csv` (this note's read; per-z-bin table in §2.4) | 2026-08-29 |
| b0i venue seed 900101 | 106 scored events (94 in-cone + 12 out-of-cone); truth node `evaluate_s = 64.73` at a 14-core pin, 41 h-nodes | `cmem_a1_work/cmem_a1_gates.json:c_g1c`; `hier_s0_registered_run/logs/s0a_seed900101_full.log`; docket §4 anchors | 2026-08-29 |
| Mirror generator defaults | `n_events = 200` drawn per realization (`CorrespondenceConfig`, `correspondence_1d.py:643`); F-0 quality filter applied inside `evaluate()` (`:3501-3502`, SNR ≥ 20 and σ_dL/d_L < 10 %); `H_GRID_41` (`:351`, 0.60–0.86, 0.005 core / 0.01 wings); `HOST_DRAW_Z_MAX = 1.5` | `correspondence_1d.py` | 2026-08-29 |
| pp_coverage mass-channel runtime | ≈ 8.5 s per realization at `n_events = 1600`, both channels, `fused`, single core | `pp_coverage.py:249-254` (docstring, measured 2026-08-17) | 2026-08-17 |
| Catalogue BH-mass error content (configuration of record) | `CV = BH_MASS_ERROR/BH_MASS` median **0.8614** (p10 0.78, p90 1.21) over 20.8 M pruned rows, under the handler's current 0.24-dex intrinsic budget; 99.61 % have CV ≥ 2/3 | `wgeo_s0_census_20260827.md` §2 | 2026-08-27 |
| L9 (docket) | B5 vs B8 disagree on what `BH_MASS_ERROR` contains (0.55-dex dominant vs 0.19 fit-only); must be reconciled before any σ_M cell is registered | `SYNTHESIS_DOCKET_1_20260829.md` §3 L9 | 2026-08-29 |
| GLADE completeness | binding at z ≳ 0.3 (completion term dominates the deep population) | `docs/gates/G7_systematics_budget.md:24` (row 15) | — |
| Stage-4 legs and [A3] criteria | three legs required (SBC/P–P, absolute-count audit, forecast-consistent width); [A3]: 2-channel with g recomputed per h, production N, multi-candidate balls | `docs/RESEARCH_CYCLE.md:270-330` | — |
| Stage-5 stop rule | "Stop digging" requires coverage pass AND width on forecast AND no unmodeled selection between generator and estimator; SBC alone cannot catch a shared filter | `docs/RESEARCH_CYCLE.md:331-347` | — |

---

## 1. What the harness must measure

### 1.1 The object: `F(N)` for an internally consistent estimator

Define, per channel `c ∈ {1D, 2D}`, on synthetic universes `u = 1…n_U` drawn from the
estimator's OWN generative model at `h_true = 0.73` and scored by the PRODUCTION estimator at
production N:

```
σ_h,harness,c(N) = median_u SD_u[ p_c(h | universe u) ]      (posterior SD on the h grid, trapezoid-normalized)
F_c(N)           = σ_h,harness,c(N) / σ_h,floor(N)            σ_h,floor(1588) = 0.001747 (B8.1 route B)
```

`F_c` is the width dilution that the *structure of the inference problem* — multi-candidate
balls (median 19 no-BH candidates, 34 % empty), a completion leg that carries ≈ 94 % of the
class weight (`w_tilde_G = 0.062`), GLADE photo-z kernels — imposes on an estimator that is
exactly right about all of it. It is **not** a property of the data alone (register §13): a
different consistent estimator form (e.g. a hierarchical one) could in principle have a
different `F`; this harness measures `F` for *production's* estimator form, which is the only
one whose stop condition is at issue.

Also reported, not headline: the 68 % HPD half-width and the no-scatter (truth-centred) SD —
three width statistics, because the HEAD readout's ⟨σ_h⟩ is a posterior SD on truth-centred
production data (production has no GW measurement scatter: `pp_coverage.py:233-247`,
`bayesian_statistics.py:3543, :3613`), while a coverage test needs scattered data (§2.3).

### 1.2 The two stage-4 legs that make `F` admissible

**(a) Coverage (P–P / SBC), both channels, ≥ 100 universes.** Per universe: the PIT
`∫_{h ≤ 0.73} p(h) dh`, HPD membership at 50/68/90 % (the harness's "1σ/2σ" read = 68 %/95 %
HPD; 95 % is added to the existing 50/68/90 set), MAP, SD, rail flag (`argmax` on a grid edge).
Statistics: PIT–KS `D` vs Uniform(0,1); binomial coverage at each level; mean MAP − 0.73 with
its SEM. Implementations: reuse VERBATIM the independently-adjudicated functions in
`results/venue_transfer_20260811/adjudicate_venue_transfer.py:84-175` (`my_pit`, `my_post_sd`,
`my_hpd_contains`, `my_ks_uniform`, `binom_bands`) — they reproduced the registered readout to
≤ 5.33e-15 (row #99) and are the demonstrably-sensitive control the A15 rule requires: on the
banked T-c(0.730) N = 400 cell they return coverage 0/0/0 and KS D = 1.000, on the T-0 (σ_z = 0)
anchor they return "all 200 seeds argmax exactly on truth". A scorer that cannot reproduce both
is not accepted (§8 S2 acceptance (ii)).

**(b) Absolute-count audit — the filter both sides share.** Under the estimator's own objects
at `h = 0.73` the expected number of detections per z-bin is

```
N_pred(bin) = N · [ α_G^φ(bin) + β_Ḡ^φ(bin) ] / D̃^φ
α_G^φ(bin)  = Σ_{g ∈ bin} w_g · S̃_φ,g              (catalogue leg; precompute_global_catalog_selection, bayesian_statistics.py:2692)
β_Ḡ^φ(bin)  = ∫_bin w_pop(z) (1 − f̄(z)) S̄_φ(z; 0.73) dz   (completion leg; precompute_phi_selection_integrals, :2077)
```

on B3.1's registered bin edges (0.075, 0.392, 0.559, 0.659, 0.753, 1.018;
`b3_pop_prediction.json:venues.*.bins`). Two reads, same instrument:

1. **On harness universes** — realized `z_true` histogram vs `N_pred`. Because the generator
   draws from exactly these objects (§2.1), this is an *instrument test*: it must pass at the
   Poisson level; a failure is a builder bug, never a finding. It also verifies the F-0 filter
   retention is reproduced by the generator (the D1-class object: `evaluate()`'s own SNR/σ_dL
   cut is applied to the harness rows too, so it is shared *and modelled*).
2. **On the production event set** (seed61000, N = 1588, `z` from `d_meas` inversion as in
   B8.1 §0 or from injection truth where available) — the genuine audit. Production's
   generator is the injection pipeline (Model1CrossCheck rate + SNR ≥ 20 + p0-window + σ_dL
   cut), NOT the estimator's model, so the residual ratio `N_real(bin)/N_pred(bin)` carries the
   known D1-class factors (p0-window retention ×1.342, `FIXB_PATHA_PACKAGE.md` §1) and B3.1's
   population-shape term (coverage 98.5 %/103.9 % on bins 2–5, `b3_pop_prediction.json`).
   Those enter the registration as *named, invariant* factors; an unexplained per-bin factor
   outside the band (§4) is a SHARED-FILTER flag and makes `F` **conditional**, because a
   filter that neither side models does not move coverage on harness universes (stage-5 rule
   of record, `docs/RESEARCH_CYCLE.md:338-343`).

Without leg (b) a coverage PASS on harness universes proves only that the estimator agrees
with itself; leg (b) is what ties the harness universe to the production one.

### 1.3 Two-channel, at σ_M = 1.99 — what that means in a real-catalogue harness

The 2D channel's host-mass information is whatever the catalogue's `BH_MASS_ERROR` column
gives (median CV 0.86 under the current 0.24-dex budget; §0). The floor is flat in σ_M for every
σ_M ≥ 0.19 (B8.1 §2.1: 0.001738–0.001747), so the σ_M axis cannot move the denominator of `F_2D`
— it can only move the numerator through the 2D catalogue leg's mass overlap
(`bayesian_statistics.py` `mz_integral`, harness analog `pp_coverage.py:2319-2336`) and the
completion-leg mass factor `g`/`g_sel` (`:2135`/`:2268`; harness `:1760`/`:1827`), both
recomputed per h. Design:

- **Primary 2D cell = configuration of record**: the pinned catalogue's own `BH_MASS_ERROR`
  (no dose), latent host mass drawn from the candidate's own `p_gal` by the existing
  `catalogue_selected_2d` law (`correspondence_1d.py:2141-2185`, `_draw_2d_accepted_latents`),
  joint `(d̂, M̂_z)` from the donor Fisher row's 2×2 block, Bernoulli(`S_4D`) acceptance.
- **Secondary 2D cell = σ_M dose to the 0.55-dex predictive total ("σ_M = 1.99")**: realized
  through the observed-catalogue machinery that already scatters mass and inflates the listed
  error consistently (`realize_observed_catalogue`, the joint_r1 venue's
  `observed_catalogue_seed900001.csv` pattern, `MEASUREMENT_HEAD_READOUT_20260827.md:44`), fed to
  BOTH generator and estimator through the one shared handler — otherwise it is a
  misspecification arm, not a calibration cell. **Gated on the L9 reconciliation** (what
  `BH_MASS_ERROR` currently contains must be settled in B5.2's registration first).
- Registered two-sided consistency check: `F_2D/F_1D` should sit near 1 (floor flatness ⇒ the
  mass channel adds no information at realistic σ_M); `F_2D/F_1D ≪ 1` is a 2D-OVERCONFIDENT
  flag (row 98 T2 delta-narrow class), `≫ 1` a 2D-DEGRADED flag (mass window losing true hosts,
  B5.1's 4.3 %/17-point retention numbers are the referent).

---

## 2. The generative model

### 2.1 The consistent truth: a mixture of two existing laws at the estimator's own class weight

Everything below already exists in `correspondence_1d.py`; the ONLY new generator object is the
mixture (and one knob, §2.3).

| component | law | existing code | what it fixes |
|---|---|---|---|
| Class split | `N_G ~ Binomial(N_draw, P_G)`, `P_G = α_G^φ(h_true)/D̃^φ(h_true)` — the estimator's own catalogue-class weight at truth (production iiib: 0.0620) | `alpha_G_phi`, `D_tilde_phi` are banked per run (`event_likelihoods.csv` columns; `correspondence_1d.py:2462-2497` inverts them) | the b0i venue conditions on the catalogue class (`PREREGISTRATION_B0_IDENTITY_20260823.md:24-27`, "realized fraction ≡ 1.0 by construction") and bsel on the dark class; neither alone is a production-N truth |
| Catalogue-hosted events (`N_G`) | host `g ∝ w_g · S̃_φ,g`; `z_true ~ k_g(z) S̄_φ(z; h_true)/S̃_φ,g` on the host's ±4σ kernel window; `k_g` = the estimator's own `volume_deconv` + C7 numerator kernel with the host-pixel completeness `f_k` | `host_mode="catalogue_selected"` (`:1129-1160` derivation comment; `catalogue_selected_host_draw_weights` `:1347`; draw branch `:2104-2140`); rate-weight leaf parity asserted to 1e-12 (`:1395-1440`) | true host IS a real catalogue row → real impostor neighbourhood |
| Dark events (`N − N_G`) | `z_true ~ w_pop(z)(1 − f̄(z; h_true)) S̄_φ(z; h_true)`, isotropic sky, never inserted into the candidate set | `host_mode="population_selected"` (`selected_population_z_weights` `:895-945`, `draw_selected_population_redshifts` `:947`, branch `:2072-2103`) | balls contain impostors only, at the real catalogue's sky density |
| 2D latent mass | `M ~ p_gal` of the drawn host; `M_z,true = M(1+z_true)`; joint `(d̂, M̂_z)` from the donor row's block; Bernoulli(`S_4D`) | `host_mode="catalogue_selected_2d"` (`:2141-2185`, `:2196-2232`) | the with-BH channel's second observable, consistent with the estimator's mass overlap |
| GW observables | `d̂ = d_L(z_true; 0.73) + N(0, σ_dL,row)`; sky = host position + the row's 2×2 sky-Fisher draw; `M̂_z` as above; `σ`'s = the donor Fisher row's own | `:2226-2255` (`obs_d_L`, `obs_phiS/qS`) | production's own per-event error structure, by resampling the pinned CRB rows (`CRB_CSV_MD5` pin) |
| Candidate balls | production handler inside `evaluate()`: `sigma_multiplier × √λ_max(Σ')` cone (`handler.py:558-`), mass window `symmetric/linear/k = 1.5` (B5.1 flags), `z` window from the h-bounds | `run_mirror_seed_inprocess` → `bs.evaluate(...)` (`:2929-2960`) | the candidate-count-vs-z density is production's, not a model of it |
| F-0 quality filter | SNR ≥ 20, σ_dL/d_L < 10 % | applied inside `evaluate()` (`:3501-3502`) — shared AND modelled | the D1-class object is visible to the count audit |

**What must be ADDED (harness files only, no physics-trigger file):**

1. `host_mode="mixture_selected"` in `MirrorUniverseGenerator.draw_realization` — Binomial
   split + concatenation of the two existing branches on ONE RNG stream (documented draw
   order), with `P_G` computed from the same construction calls `build_bsel_selection_objects`
   already makes (`:983-1000`: the SAME `SimulationDetectionProbability`, `from_cache_or_build`,
   `precompute_phi_marginal_survival` calls `evaluate()` makes) plus one call to
   `precompute_global_catalog_selection` at `h_true` for `α_G^φ`. `P_G = 1` must reproduce
   `catalogue_selected` bit-for-bit, `P_G = 0` must reproduce `population_selected`.
2. `N_draw` sizing: the generator draws `N_draw = ⌈1588 / r_F0⌉` rows so that ≈ 1588 survive
   F-0; `r_F0` is measured at the pilot (§8 S3) per component — it differs by law (b0i seed
   900101: 106 scored of 200 drawn; fc/ft: 174–188 of 200) because re-centring a donor row at a
   new host distance changes its fractional error. Realized scored N is recorded per universe
   and enters the count audit; universes are NOT rejected on N (that would be a filter the
   estimator does not model).
3. `gw_scatter: bool` knob on the generator (default `True` = today's behaviour, byte-identical;
   `False` = truth-centred `d̂`, `M̂_z`, sky — the production-faithful width cell, §2.3).
4. A resolved-flags return from the estimator call (§3).

**What is deliberately NOT built:** a synthetic "impostor placement law" or a sampled
"candidate-count density". Either would re-implement the catalogue's depth/clustering
structure that the real handler already provides; the census (§2.4) is the acceptance test
of the emergent density, not its source.

### 2.2 Completion fraction `f(z)`

Not a new object: `f̄(z; h)` and per-pixel `f_k(z; h)` come from the same completeness object
production builds (`from_cache_or_build`, `pixel_completeness.py:514`; protocol `:113-140`),
and the generator uses them exactly where the estimator does — `f_k` inside the b0i kernel
`k_g`, `f̄` inside the dark-event law. The harness therefore reproduces production's own
completion fraction by construction (`w_tilde_G = 0.062` at iiib), and the non-empty-ball
fraction it realizes (production: 0.6184 at h = 0.73) is an acceptance target (§2.4), not an
input. Row #98's `B-F1` unity-completeness control (`_UnityCompleteness`,
`completeness_override=True`) is available as a limiting-case cell (P_G → 1, no completion leg)
but is NOT a control in the A15 sense — it returned a flat posterior once already (A15 evidence
text) and is excluded from the verdict path.

### 2.3 Scattered vs truth-centred cells (paired on one RNG stream)

- **Cell S (scattered, primary):** `gw_scatter=True`. The only SBC-valid cell: coverage, PIT,
  MAP bias, `F` from SD and from HPD half-width.
- **Cell T (truth-centred, secondary, width-only):** `gw_scatter=False`, same seeds (draw made
  and discarded, the `pp_coverage.py:904-907` convention). Its SD is the like-for-like
  comparand of the HEAD readout's ⟨σ_h⟩ (production data are truth-centred). No coverage claim
  is made from cell T (its PIT is degenerate by construction — A15 "provably identical before
  the run" clause).

`F` of record = cell S SD-based; cell T's ratio to cell S is reported so the production
comparison (§4 width branch) can be read on the matching convention.

### 2.4 The candidate-count-vs-z acceptance census

The harness's realized per-event candidate counts (parsed from the `evaluate()` log exactly
as B4's `candidate_counts()` does, `b4_imp_stage1_forecast.py:146-170`) must reproduce the banked
mirror census (fc/ft arms, same catalogue, same CRB pool, `b4_imp_stage1_events.csv`):

| z_true bin | n | no-BH mean / median / p90 | frac zero (no-BH) | with-BH mean / median | frac zero (with-BH) |
|---|---:|---|---:|---|---:|
| (0.075, 0.2] | 148 | 1278 / 63 / 4300 | 0.162 | 615 / 26 | 0.257 |
| (0.2, 0.3] | 438 | 1530 / 139 / 4567 | 0.192 | 310 / 24 | 0.342 |
| (0.3, 0.392] | 814 | 1593 / 178 / 4868 | 0.231 | 140 / 3 | 0.432 |
| (0.392, 0.5] | 1136 | 707 / 92.5 / 1978 | 0.276 | 20.6 / 0 | 0.796 |
| (0.5, 0.559] | 506 | 250 / 23 / 645 | 0.296 | 0.39 / 0 | 0.960 |
| (0.559, 0.659] | 626 | 104 / 5 / 172 | 0.377 | 0.003 / 0 | 0.997 |
| (0.659, 0.753] | 366 | 11.0 / 0 / 16 | 0.585 | 0 / 0 | 1.000 |
| (0.753, 0.9] | 234 | 1.83 / 0 / 1 | 0.889 | 0 / 0 | 1.000 |
| (0.9, 1.1] | 32 | 0 / 0 / 0 | 1.000 | 0 / 0 | 1.000 |

{values: this note's pandas read of `b4_imp_stage1_events.csv` (fc + ft rows, n = 4304);
2026-08-29}. Two cautions the registration must carry: (i) this fleet's z_true distribution
(median 0.459) is NOT production's (median 0.490) because the fc/ft arms are bsel-law fleets
(dark class only); the mixture at `P_G = 0.062` is expected to sit close to it but the
**production non-empty-ball fraction 0.6184** (both channels, h = 0.73) is the primary
acceptance number and the per-bin medians the secondary; (ii) with-BH `n_cand` counts
mass-window passers, while production's `L_cat_with_bh > 0` fraction equals the no-BH one —
different objects, both recorded.

Acceptance band (registered in S4 from the pilot's realized scatter, A17): non-empty fraction
within ±0.05 of 0.618 (no-BH) at N = 1588; per-bin median `n_cand` inside the census IQR for
every bin with n ≥ 100. Outside ⇒ the harness does not reproduce the production venue and `F`
is not banked.

---

## 3. The estimator call is the production path (A13 engagement)

`run_mirror_seed_inprocess` (`correspondence_1d.py:2734-2960`) calls the REAL
`BayesianStatistics().evaluate(...)` "imported, not reimplemented" (`:2791-2800`), with
`PRODUCTION_FLAGS` (`:329-338`): `normalization_mode="absolute_marginal"`,
`host_z_kernel="volume_deconv"`, `selection_in_completion_numerator="fused"`,
`catalogue_mass_overlap="production"`, `completion_b_scale="derived"`, `pdet_*` as production;
`catalogue_numerator_survival="auto"` → `"phi"` and `catalogue_global_selection="auto"` →
`"phi"` under `absolute_marginal` (`bayesian_statistics.py:3735-3757`, chair-confirmed in the
docket); `catalogue_numerator_survival_2d="off"` at HEAD (B7.2 may adopt `mz_sel`/`eff` in the
wave-2 batch — F2 serialized adoption: the harness runs at the wave-2 commit and asserts
whichever value production resolves there); `mass_filter_sigma="symmetric"`,
`mass_filter_geometry="linear"`, `mass_filter_k=1.5` (B5.1 defaults, byte-identical);
θ = (0, 1), `theta_sites="all"`, `smear_global_selection=False` (B1.1/B6.1 identity). Both
channels come from the ONE call (production emits `combined_no_bh` and `combined_with_bh` per
event per h, `_write_diagnostic_csv` `:4716-4745`) — the harness never scores the 2D channel
separately.

Three engagement assertions, all STOP-gated, all required before any universe is banked:

1. **Resolved-flag assertion.** The harness reads the resolved attributes off the `bs` instance
   after the call (`_normalization_mode`, `_catalogue_global_selection`,
   `_selection_in_completion_numerator`, `_catalogue_numerator_survival`,
   `_catalogue_numerator_survival_2d`, `_mass_filter_sigma/_geometry/_k`, θ-hook state) and
   asserts them equal to the registered production values; requires a one-line harness change
   (return the dict, or a sibling wrapper) — `correspondence_1d.py` is not a physics-trigger
   file. Plus the log-line witness `'[PHYSICS] catalogue_global_selection="phi" ACTIVE'`
   (`bayesian_statistics.py:3746-3750`).
2. **PROD-A0 ingredient gate (the decisive "same path" proof).** The harness driver, fed the
   production 1588 CRB rows verbatim (no resampling, no scatter, `h_bounds` = (0.60, 0.86)),
   must reproduce the banked `headreadout_20260827/iiib/event_likelihoods.csv` columns
   (`L_cat_no_bh`, `L_cat_with_bh`, `B_num`, `B_num_wbh`, `combined_*`) to ≤ 1e-12 at all 41 h
   at the HEAD commit, or the wave-2 C0 baseline task's columns at the wave-2 commit (docket L5
   pattern; A22 clean-tree stamp). Cost: one production-N `evaluate()` (§6). A harness that
   cannot reproduce production on production's own events is not the production path.
3. **Sensitivity of the instrument (A15):** the scorer reproduces the venue-transfer T-c and
   T-0 cells from their banked `ln_post` vectors (§1.2(a)); the generator's `P_G = 0/1` limits
   reproduce the banked bsel/b0i seeds bit-for-bit (§2.1).

The `[P3-HGRID]` fact is binding (`:2918-2926`): the catalogue-leg numerators depend on the
h-list extremes through the candidate z-window, so any split of the 41-node grid across two
`evaluate()` calls (§8, the ≤ 600 s foreground rule) MUST pass `h_bounds=(0.60, 0.86)`
explicitly to both halves; the S1 acceptance test pins split-vs-whole bit-identity.

---

## 4. Statistics, bands (A8 two-sided, branch referents) and A15 characteristics

Registered numbers of universes: **n_U = 100 per channel per cell at N = 1588 (cell S);
n_U = 25 (cell T); pilot n_U = 100 at N = 200 (both cells)**. Both channels share the universes
(paired), so a 1D/2D comparison is a per-universe delta (A2).

### 4.1 Operating characteristics at n_U = 100 (null = a calibrated estimator)

| statistic | null distribution | two-sided band (α ≈ 5 %) | effect detectable at ≳ 80 % power |
|---|---|---|---|
| PIT–KS `D` | Kolmogorov, n = 100 | `D ≤ 0.134` (exact n = 100 critical value; asymptotic 1.358/√100 = 0.136) | coherent displacement ≥ 0.45 σ_post (KS D for a Gaussian shift δσ ≈ 2Φ(δ/2) − 1); at F = 10 that is ≈ 0.008 in h, at F = 3 ≈ 0.0024 |
| HPD coverage 68 % | Binomial(100, 0.68), SD 0.0466 | [0.589, 0.771] | displacement ≥ 0.75 σ_post (coverage of a ±1σ interval drops 0.68 → 0.60 at δ = 0.75σ) |
| HPD coverage 95 % | Binomial(100, 0.95), SD 0.0218 | [0.907, 0.993] | tail mis-scaling ≥ 25 % |
| HPD coverage 50 % / 90 % | SD 0.050 / 0.030 | [0.402, 0.598] / [0.841, 0.959] | as above |
| mean MAP − 0.73 | N(0, σ_post/√100) | `|Z| ≤ 3` with `SEM = σ̄_post/√n_U` | ≥ 0.3 σ_post ⇒ at F = 10 exactly **0.0052 = 3σ_floor** (the L7 band): n_U = 100 is the minimum at which the harness's own centering sensitivity equals the band it feeds; if the pilot returns F > 10, `n_U` scales as (F/10)² |
| per-event score at truth (A12, free) | mean of `∂_h ln p_i` over 1588 × 100 events; secant on (0.725, 0.735) as `per_event_scores` (`b4_imp_stage1_forecast.py:136-143`) | `|Z| ≤ 3` on the fleet SE | class-resolved (catalogue-hosted vs dark) at zero extra compute |
| `F_c` | per-universe SD scatter ≈ 1/√(2 N_eff) ≈ 2 % relative at N = 1588 (Kish n_eff 1266, B8.1) | SEM(F)/F ≈ 0.2 % — negligible; the **systematic** spread across width statistics (SD / HPD half-width / cell-T SD) is the reported uncertainty | — |
| count-audit residual per bin | Poisson: `N_real/N_pred` ± `1/√N_pred(bin)` | harness universes: `|Z| ≤ 3` (instrument test); production: `[0.9, 1.1]` after the registered retention factors | an unregistered shared filter of ≥ 10 % in any bin with N_pred ≥ 150 |

Joint false-fail over the three coverage levels + KS is ≈ 8–12 % (correlated); the KS is the
PRIMARY statistic, the levels are diagnostics — stated so the registration does not multiply
tests silently.

### 4.2 Branches (each with its referent arm and its falsifier, A14/A19)

| branch | condition (cell S, per channel) | referent | meaning | falsifier |
|---|---|---|---|---|
| **CONSISTENT-CALIBRATED** | KS ≤ 0.134 AND 68 %/95 % in band AND `|Z_MAP| ≤ 3` AND harness count audit passes | harness cell S | `F_c` is a valid dilution factor | a second, disjoint 100-universe draw (seed block 2) fails any clause |
| **DEFECT-IN-CONSISTENT-VENUE** | any clause fails | harness cell S | the production estimator is miscalibrated even against its own model — `F` is NOT defined as a dilution factor; the failing statistic localizes the defect (score by class, A12) | bsel-only (`P_G = 0`) and b0i-only (`P_G = 1`) cells: the defect must survive in at least one, else it is the mixture code |
| **WIDTH-EXPLAINED** | `R_c = σ_h,prod,c / (F_c σ_floor) ∈ [0.8, 1.25]` (cell-T convention) | production HEAD 2D ⟨σ_h⟩ 0.01847 (1D NOT evaluable while MAP == 0.600; its referent becomes the pure-dark-only or first unpinned 1D posterior) | production's width is the consistent estimator's width at this N | the σ_M-dose cell (if run) must leave `R_2D` inside the band |
| **EXCESS-WIDTH** | `R_c > 1.25` | same | production wider than a consistent estimator ⇒ misspecification widens it (impostor drag/tilt class, B4/B1) | — |
| **OVER-CONFIDENT** | `R_c < 0.8` | same | production narrower than the consistent width — the row 98/99 delta-narrow class | — |
| **SHARED-FILTER** | production count-audit residual outside [0.9, 1.1] in any bin beyond the registered factors | production seed61000 event set | `F` conditional; a filter neither side models | the B3.2 M1-prior arm's registered per-bin prediction (docket L1/L4) |
| **2D-OVERCONFIDENT / 2D-DEGRADED** | `F_2D/F_1D < 0.9` / `> 1.1` | paired universes | mass channel narrowing beyond information / losing true hosts | B5.2's retention prediction 0.789 ± 0.009 as the degraded-branch referent |

The bands 0.8/1.25 and 0.9/1.1 are placeholders for the registration; S4 derives them from
the pilot's realized F scatter (A17: "operating characteristics on the realized scatter").

---

## 5. The stop condition this feeds — and how "starved" will be declared or refused

L7 (docket §3) reads every wave-3 blind readout against `|⟨h⟩ − 0.73| ≤ 3σ_floor = 0.0052`
(centering) and `σ_h,measured ≤ F σ_floor` (width). This harness supplies `F`; it does not touch
the centering clause. Consequences, stated before any number exists:

1. **The centering clause is independent of F.** HEAD fails it by 38–68 floor-σ (B8.1 §4); no
   value of `F` rescues a centering failure. The width clause becomes *meaningful* only once a
   readout passes centering; until then `F` is a bound on what "done" will look like, not a
   verdict on today's posteriors.
2. **Admissible vocabulary.** If CONSISTENT-CALIBRATED fires with `F ≫ 1` and WIDTH-EXPLAINED
   holds, the campaign is **"dilution-limited at F"**: no estimator of production's form can be
   narrower than `F σ_floor` on this catalogue at this N, and the remaining gap to the
   single-host floor is the price of impostor competition and the completion leg — a property
   of the *estimator form on this catalogue*, not of "the data". The word "starved" is NOT used
   in any harness verdict (register §13; the charter's own use of "starved" is the ceiling past
   which no estimator fix helps — that ceiling is `F σ_floor`, reported as such).
3. **Refusals.** The harness REFUSES to report a dilution factor when (a) DEFECT-IN-CONSISTENT-
   VENUE fires (a miscalibrated consistent estimator has no meaningful `F`; the branch routes
   to a `/physics-change` intake with the localizing statistic), (b) SHARED-FILTER fires (`F`
   is reported CONDITIONAL on the named filter), or (c) the acceptance census (§2.4) fails (the
   harness is not the production venue).
4. **Stop/continue mapping (stage 5 table).** CONSISTENT-CALIBRATED + WIDTH-EXPLAINED + a
   centering PASS on a future readout ⇒ "CALIBRATED + wide (≈ forecast): stop digging, report
   the bound" with `F σ_floor` as the forecast width. EXCESS-WIDTH or OVER-CONFIDENT with a
   centering PASS ⇒ DEFECT (≥ 3σ coherent displacement or coverage failure) — fix via
   `/physics-change`. Anything with a centering FAIL ⇒ the existing branch tree (B1/B3/B4/B5
   own it); this harness adds the width number to the record and nothing else.

---

## 6. Cost

**Compute anchors (A11):** mirror unsmeared cell 64.73 s at a 14-core pin for 106 scored events
× 41 h-nodes ≈ 0.25 CPU-h (`s0a_seed900101_full.log`; docket §4). The fleet's ≈ 65 s at 174–188
events is the same number — i.e. at N ≈ 100–200 the call is dominated by its fixed per-call
context (S̄_φ table, Σ^φ over 20.8 M rows × 41 h, handler reuse aside), so the **N-scaling of
one call is UNMEASURED** and the production-cluster per-h-point figures (14.93–22.9 CPU-h,
docket §4) are not comparable (separate task per h-point, 16-cpu billing of a single-threaded
catalogue load). Bracket used: per universe at N = 1588, 41 nodes, both channels =
**1.0 CPU-h (fixed-cost-dominated) to 3.8 CPU-h (linear, 64.73 × 1588/106 = 970 s wall)**.
S3 measures it on a 3-point ladder before anything is registered (A6/A17 costing line).

| item | universes × N | CPU-h (bracket) | wall at 14 cores |
|---|---|---|---|
| S1/S2 smoke + unit tests | ≤ 5 × 20 | < 1 | minutes |
| PROD-A0 engagement gate (§3 item 2) | 1 × 1588 (production rows) | 1–4 | 5–16 min |
| N-ladder timing | 3 × {106, 400, 1588} | 2–5 | ≤ 30 min |
| Pilot cell S | 100 × 200 | 25 | ≈ 2 h |
| Pilot cell T | 25 × 200 | 6 | ≈ 0.5 h |
| **Cell S, production N** | **100 × 1588** | **100–380** | **7–27 h** |
| Cell T, production N | 25 × 1588 | 25–95 | 2–7 h |
| (optional, L9-gated) σ_M-dose 2D cell | 50 × 1588 | 50–190 | 4–14 h |
| **mandatory total** | | **≈ 130–475 CPU-h local, 0 cluster** | **13–46 h wall** |

This is 20–80× the docket's "≈ 6 CPU-h per harness sweep"; the docket's number is the 24-arm
mirror sweep at mirror N and cannot be the production-N number. Zero cluster exposure; nothing
to archive-schedule beyond the campaign directory (outputs are per-universe JSON + the
ln-posterior vectors, ≈ 100 × 2 × 41 floats — KB, not GB; the per-universe `event_likelihoods.csv`
at 1588 × 41 rows ≈ 12 MB each, 1.2 GB per 100 universes — keep, gzip).

---

## 10. Appended note (2026-08-29 — wave-2 GAP-CLOSURE archive/notes worker, launched under rows
#222/#223 — charter node: NODE archive+minor-notes, GAP 12)

Closes `WAVE2_REGISTRATION_CHECK_20260829.md` §5 item 12 / §3 item 8. Standing rule 1
(append-only) applies — the §4 falsifier table above is left as written.

**SHARED-FILTER referent re-pointed.** The §4 falsifier table's SHARED-FILTER row cites, as its
falsifier, "the B3.2 M1-prior arm's registered per-bin prediction (docket L1/L4)". That arm (C2)
is **struck from wave 2** (`WAVE2_REGISTRATION_CHECK_20260829.md` §0 item 3 / §4 row 3,
`COMPUTE_LEDGER.md` wave-2 cost-refinement row C2 → STRUCK; docket L1 re-cut, L4 struck as a
dependency — `SYNTHESIS_DOCKET_1_20260829.md`'s own §4.3 appended note) and registers no per-bin
prediction that will ever exist. SHARED-FILTER's referent is therefore re-pointed from C2 to
**S4** (§8, "S4 registration"): the falsifier for a production count-audit residual outside
[0.9, 1.1] beyond the registered factors becomes **S4's own independent count-audit prediction**
— specifically the S4-registered `pp_coverage.py` run at `n_events=1588, catalogue_mode,
mass_channel, fused` (§8 S4 registration row) — rather than a comparison to a struck arm's per-bin
population prediction. If that independent count audit's predicted per-bin factors are themselves
inside [0.9, 1.1] of the harness's own count audit, SHARED-FILTER does not fire; a residual beyond
the S4-predicted factors is what triggers the CONDITIONAL disposition on `F` (§5 item 3(b)).

This does not change the SHARED-FILTER branch's condition, meaning, or the [0.9, 1.1] band
(placeholders per §4, still S4-derived) — only the object it is checked against. S4's own
registration (§8) must state this referent explicitly when it derives the bands.

{source: `WAVE2_REGISTRATION_CHECK_20260829.md` §3 item 8 ("Cross-link consistent: SHARED-FILTER's
falsifier cites 'the B3.2 M1-prior arm's per-bin prediction (L1/L4)' — after striking C2 that
referent is gone; B8.2 S4 must name another (C0's dark-class profile pin or the count audit
itself)"), §5 item 12; this document §4 (falsifier table), §8 (S4 registration row); 2026-08-29}

**Stamp:** launched under rows #222/#223 — charter node NODE archive+minor-notes (GAP 12),
2026-08-29. No git operations; no edits to `hier_s0_driver.py` or `kwq1_score.py`.

**Builder effort (row #224 tiering):** 3 sonnet build/run stages + 1 top-tier registration +
1 verifier (sonnet, clean context, falsification brief — A20). Wall: S1 ≈ 2–4 h, S2 ≈ 3–5 h, S3
≈ 3–4 h (compute-bound), S4 ≈ 2 h (top tier), S5 ≈ 1–2 days elapsed (chunked foreground calls),
verifier ≈ 3 h. Fan-out: at most 1 agent per stage, no panel; total top-tier agents 1
(registration) + the orchestrator — inside the ≈ 3 cap.

---

## 7. A10 invariants and blindness sentence

**Invariants (held fixed in every cell, with last derivation-audit date):** the pinned reduced
catalogue (md5 `c52c13b5…`, checksum-gated at the consumer; audited 2026-08-19 G-0) · the pinned
CRB pool `seed61000/prepared_cramer_rao_bounds.csv` (`CRB_CSV_MD5`; audited 2026-08-19) and the
injection pool · `PRODUCTION_FLAGS` verbatim plus the resolved `phi/phi/fused` triple (audited
2026-08-29, docket chair item (c)) · `H_GRID_41` with `h_bounds = (0.60, 0.86)` ([P3-HGRID],
audited 2026-08-24) · `S̄_φ` table construction (`precompute_phi_marginal_survival`; common
mode since 2026-08-23, NEVER independently re-derived — flagged) · the rate-weight leaf `w_g`
(parity-asserted 2026-08-23) · the completeness object (`from_cache_or_build`; C7 core; NEVER
audited against an independent completeness estimate — flagged) · `h_true = 0.73` · B5.1 window
defaults · θ = (0, 1). Per A10, the registration must either audit one of the two NEVER items
in the same cycle or state that the harness's verdict is conditional on them.

**Blindness sentence (the b0i self-consistency limit).** *By construction this harness cannot
detect any defect that the generator inherits from the estimator: an error in the φ-marginal
survival `S̄_φ`, in the rate-weight leaf `w_g = R_eff(M)/(1+z)`, in the completeness `f_k`/`f̄`, in
the population weight `w_pop = dV_c/dz/(1+z)`, in the Gaussian photo-z kernel with the
catalogue's listed `z_error` (the `s` axis B1/B6 own), in the Gaussian sky likelihood, in the
mass prior `p_gal` used as the latent-mass law, or in the isotropic placement of dark hosts
(real dark hosts are clustered with catalogue galaxies, so real impostor counts around them
exceed what an isotropic completion leg and an isotropic generator both assume) — every one of
these is common-mode and cancels out of coverage, width and the harness-side count audit
alike; only the production-side count audit (§1.2(b)) and the independently built
`pp_coverage.py` control (§8 S4 prediction) look at any of them, and only at the count level.*
Named additionally: the donor-row noise model (a resampled Fisher row's σ's are those of its
own injected parameters, re-used at the new host's — the harness analog of the const-σ
sub-term (a), `pp_coverage.py:233-247`); the mass window is the ONE estimator-only filter the
harness CAN see (the generator does not apply it), so a true-host loss there shows up as
coverage/width, which is the intended B5 read.

---

## 8. Implementation plan — 3 sonnet build stages, 1 top-tier registration, 1 verifier

All stages: foreground commands ≤ 600 s each (the driver is checkpointed per universe and
re-invocable; at N = 1588 one `evaluate()` may exceed 600 s wall, so the driver splits the
41-node grid into two calls with explicit `h_bounds=(0.60, 0.86)` — bit-identity pinned in S1);
no git operations; no physics-trigger file edits; every record stamped `launched under rows
#222/#223 — charter node B8.2.<stage>`; A22 commit + dirty-state stamp at run START.

| stage | model / effort | deliverable | acceptance tests (all must pass; verifier re-runs them) |
|---|---|---|---|
| **S1 generator** | sonnet / medium | `host_mode="mixture_selected"` + `gw_scatter` knob + resolved-flags return, in `correspondence_1d.py` (harness file) + unit tests in `darksiren_emri_test/validation/` | (i) existing arms byte-identical: b0i seed 900101 and one bsel seed reproduce their banked `event_likelihoods.csv` bit-for-bit; (ii) `P_G = 1` ≡ `catalogue_selected`, `P_G = 0` ≡ `population_selected` on the same seed, bit-for-bit; (iii) `gw_scatter=False` consumes the same RNG draws (paired stream test); (iv) grid split 21+20 nodes with `h_bounds=(0.60,0.86)` ≡ whole 41-node call, bit-for-bit; (v) `uv run pytest -m "not gpu and not slow"` green, ruff/mypy clean |
| **S2 driver + scorer** | sonnet / medium | `b8_cal_harness.py` in this directory: universe loop (checkpoint JSON per universe: ln-posterior vectors both channels, MAP, SD, HPD 50/68/90/95, PIT, per-event score at truth by class, realized N, `N_G`, per-event `n_cand` from the log parser, `z_true` histogram on B3.1 bins, `N_pred` from the estimator objects), `--max-wall-s`, `--n-universes`, `--n-events`, `--cell {S,T}`, `--seed-block`; scorer module with the verbatim `adjudicate_venue_transfer.py` functions | (i) smoke at N = 20, n_U = 2, both cells; (ii) scorer reproduces the banked T-c(0.730) N = 400 cell (coverage 0/0/0, KS 1.000, `post_sd` median 0.004376) and the T-0 anchor from their raw vectors; (iii) **PROD-A0 gate**: fed the production 1588 rows, reproduces `headreadout_20260827/iiib/event_likelihoods.csv` to ≤ 1e-12 at all 41 h (HEAD) or the wave-2 C0 baseline; (iv) resolved-flag assertion prints and STOPs on mismatch; (v) harness-side count audit passes at Poisson level on the smoke universes |
| **S3 pilot + costing** | sonnet / low (mechanical) | N-ladder timing JSON ({106, 400, 1588} × 1 seed); pilot n_U = 100 (cell S) + 25 (cell T) at N = 200; `r_F0` per component; census comparison table; pilot `F_200`, coverage, score-zero by class | (i) every universe completes and checkpoints; (ii) census within the provisional band of §2.4 (else STOP and report — do not tune); (iii) timing line `wall/seed, peak RSS/seed` recorded (A21(b)); (iv) no number from S3 is quoted as a verdict — it is registration input |
| **S4 registration** | **top tier / xhigh** (not sonnet) | `PREREGISTRATION_B8_CAL_HARNESS_<date>.md`: bands re-derived from the pilot's realized scatter (A17), A15 table at the realized `n_U`/`N`, A10 invariants + blindness (§7), A14/A19 falsifiers per branch (§4.2), cost + archive line (F4), predictions: `F_1D`, `F_2D` expected ranges from the pilot and from an independent `pp_coverage.py` run at `n_events=1588, catalogue_mode, mass_channel, fused` (8.5 s/realization; a PREDICTION, never evidence), the L9 reconciliation status for the σ_M cell | reviewed by the end-of-fan-out verifier before S5 launches; branch-referent and two-sidedness checks (A8) written into the file |
| **S5 production-N run** | sonnet / low (mechanical) | cell S n_U = 100, cell T n_U = 25 at N = 1588 (optional σ_M cell only on an explicit orchestrator [DO] after L9); readout JSON + a comprehension-first readout section appended to this note's successor record | (i) A22 stamps identical across all universes; (ii) resolved-flag assertion identical across universes; (iii) census acceptance at N = 1588; (iv) scorer output only — no narrative adjudication by the runner |
| **Verifier** | sonnet / high, clean context, falsification brief (A20) | independent report | (i) re-executes ≥ 3 universes bit-for-bit from the checkpoints' seeds; (ii) re-scores every statistic from the raw vectors with its own implementation (≤ 1e-12); (iii) re-derives `F` and each band's fired branch; (iv) greps this note and the readout for "starv" and for any centering claim; (v) checks the PROD-A0 gate artifact |

**Bounded-scope rule for the builders:** S1–S3 may not change any band, any statistic
definition, or the mixture law; a "corrected premise" discovered during implementation STOPS
the stage and returns here as an appended note (A21).

---

## 9. Reproduction and provenance of this note's own numbers

- Production non-empty-ball fractions and `w_tilde_G`: pandas over
  `headreadout_20260827/iiib/event_likelihoods.csv` rows with `h == 0.73` (1588 rows; 65 108
  data rows total = 1588 × 41).
- Mirror census: pandas over `b4_imp_stage1_events.csv` (fc + ft rows, `n_cand_*` non-null;
  bsel rows carry no counts), bins as in §2.4.
- All other numbers: cited files and lines in §0. No `evaluate()` call was made; no RNG.

*Builder's self-check (rule 2): this is a design, not a measurement; nothing here is banked.*
