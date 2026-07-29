# Idealization Ledger — EMRI dark-siren H0 pipeline (campaign #51)

**Date:** 2026-07-29. **Question (author):** "Is this the full realistic setup including the very
bad galaxy mass measurements and catalogue redshift measurements, or are there idealizations that
enhance the results compared to a true realistic experiment simulation?"

**Audited result:** H0 = 73.0 ± 0.03 km/s/Mpc, 1588 used events (seed 61000,
`run_seed61000/posteriors_fixed/combined_posterior.json`, `variant: "posteriors"` = the 3D
without-BH-mass channel), MAP on truth for two seeds. Measured 3-point curvature of the combined
log-likelihood at h = 0.73 gives σ_h = 3.24e-4 → **σ_H0 = 0.032 km/s/Mpc** [MEASURED].

**Verdict up front: NO — this is not the full realistic setup.** The catalogue's "very bad"
mass and redshift measurements are *stored* honestly (0.24 dex masses, σ_z ≈ 0.035 photo-z), but
on the production path they are **never realized as noise and (for redshift) not even used as a
width**. The quoted precision is the precision of a universe in which every GLADE+ catalogue
redshift is exact. Reproduction script: `audit_information_decomposition.py` (this directory);
aligned host table: `incat_hosts_seed61000.csv`.

---

## 1. Where the information actually comes from [MEASURED]

Decomposing the combined ln-likelihood curvature (h ∈ {0.725, 0.73, 0.735}) event-by-event:

| slice | n events | curvature share | implied per-event σ_h |
|---|---|---|---|
| in-catalog events (`host_galaxy_index ≥ 0`) | **76** (4.8%) | **241.3 (101%)** | 0.0028 (0.38%) |
| dark events (out-of-catalog) | 1514 (95.2%) | −3.0 (−1%) | none (slightly anti-informative) |

- **100% of the H0 information comes from the 76 in-catalog events.**
- The **3 loudest** (SNR 995–1425, z ≈ 0.016–0.021, σ_dL/dL = 0.09–0.11%) carry **46%** of the
  total information by themselves.
- Per-event curvature matches the analytic prediction "z known exactly, GW d_L error only",
  I_e = (h·σ_dL/dL)⁻², to within ~5% event by event (table in §3 script output). The likelihood
  is *literally* measuring h = d_L(z_cat)/d_L_GW with the GW error as the only noise source.
- **All 76 information-carrying hosts are PHOTOMETRIC-redshift galaxies** (GLADE+ flag 1,
  σ_z ≈ 0.035): median z = 0.071, median σ_z/z = **49%**. Their catalogue redshift uncertainty
  plays no role in the number quoted. [MEASURED]

---

## 2. The ledger

Tags: **MEASURED** (computed here from production artifacts), **ESTIMATED** (analytic
order-of-magnitude), **LITERATURE** (external reference). "Doc?" = was this already documented in
the repo as an idealization (vs newly surfaced/quantified here).

| # | Idealization | Location | Real experiment | Pipeline behaviour | Effect on σ_H0 / bias | Doc? | Tag |
|---|---|---|---|---|---|---|---|
| **I1** | **Injected true z ≡ catalogue z (no z-measurement scatter realization)** | `galaxy_catalogue/handler.py:692-800` (`draw_rate_weighted_hosts` "z … straight from its catalog row"); `datamodels/parameter_space.py:260-274` (`set_host_galaxy_parameters`: `d_L = dist(host_z, h)`) | The catalogue z is a *noisy estimate*; true z differs by σ_z ≈ 0.035(photo) / 0.0017(spec) + PV. The event's d_L is set by the *true* z, the inference only sees the noisy one. | Host drawn from the catalogue row verbatim; the event is placed at exactly d_L(z_cat, 0.73). Truth and observation are the same number. | **Dominant.** Removes per-event z-noise entirely; per-event σ_h collapses from ≥3–49% to the GW σ_dL/dL (0.1–0.5% for the golden events). σ_H0: 0.03 → 0.2–3.6 (see §3). Also the reason MAP lands exactly on truth (no noise to scatter it). | Partially — stated as a *premise* in `bayesian_statistics.py:3456-3460` and the generator_marginal derivation, but never priced as an idealization | MEASURED |
| **I2** | **Inference z-kernel is a δ-function at the catalogue z (production mode)** | `bayesian_statistics.py:112-139` (`resolve_host_z_kernel`: `generator_marginal → "point"`), `:3470`, `:3616-3629` (numerator point-evaluated at `host_z`); production default `normalization_mode="generator_marginal"` (`arguments.py:527-539`, adopted 2026-07-26; `cluster/evaluate.sbatch` passes no override) | Even with a perfect catalogue one must marginalize the z posterior of each host; with photo-z hosts the kernel must be ~0.035 wide | In-catalogue numerator N_g = GW 3D Gaussian point-evaluated at z_g. `host_z_error` is used only in the (diagnostic) D_g/Z_g machinery. In the width modes (`volume_deconv` etc.) the kernel is `norm(loc=host_z, …)` (`:3512`) — **always centred on truth by construction** because of I1; σ_z enters as width only, never as an offset. | This is the *inference-side half* of I1. Point kernel + no scatter = "z exact". Internally self-consistent ("generator-exact") — the pair (I1, I2) is consistent-but-idealized, not a bug. | Yes (mode documented as generator-exact); its *consequence* for realism newly quantified here | MEASURED |
| **I3** | **Peculiar velocities: width only, and 0 at inference; never a realization** | `constants.py:95` (`SIGMA_V_PEC_KM_S = 0.0`), `constants.py:71-94` + `handler.py:344-399` (PV widths folded into catalogue z_error at parse time); `bayesian_statistics.py:3488` (adds 0) | Every host has an actual PV draw of ~150–500 km/s that displaces z_obs: Δz = (1+z)v/c → at the golden events' z ≈ 0.02 that is a **2.4–11% distance error** vs the 0.1% GW error used | No PV is ever added to any injected redshift; the PV budget lives only inside the (bypassed, see I2) z_error width | Absence buys the golden events a factor ~25–100 in per-event information. Restoring PV alone (spec-z universe): σ_H0 ≥ 0.22–0.30 (§3 B/C) | Yes as accounting ("counted once", issue #40b) — but only as a width; the missing *realization* is newly flagged | ESTIMATED (bounds MEASURED via §3) |
| **I4** | **Host BH mass: injected mass ≡ deterministic mean-relation mass; 0.24 dex scatter is width-only** | `handler.py:885-891` (`_map_stellar_masses_to_BH_masses`), `:1117-1131` (`_empiric_stellar_mass_to_BH_mass_relation`: mean Reines–Volonteri relation, no draw; σ_int = 0.24 dex → `M_error` only); injection `parameter_space.py:268` (`M_z = host_M·(1+z)` exact) | True M_BH scatters ≥0.24 dex around the M*-relation; the GW M_z would *disagree* with the catalogue-predicted mass at the ~74% level per host | Injected M_z is exactly the catalogue-derived mass; GW measures it to ~1e-7 (CRB); the with-BH-mass channel's kernel (gaussian/trunc_lognormal) is *centred on the injected value* with 0.24 dex width | **Headline number unaffected** — the quoted posterior is the without-BH-mass variant (`combined_posterior.json: "variant": "posteriors"`), and the 3D candidate list applies no mass cut (`handler.py:508-541`). But the with-BH-mass channel and the ball-tree mass filter (`handler.py:527-535`) enjoy idealized mass agreement: truth always inside the window, mass channel discriminates hosts it could not in reality | No — newly surfaced in this form | MEASURED (channel attribution) |
| **I5** | `FRACTIONAL_MEASURED_MASS_ERROR = 1e-8` | `constants.py:69` | — | **Dead constant — zero consumers** in `master_thesis_code/` and tests. The GW mass error actually used is the CRB `delta_M_delta_M` (`detection.py:143`): median fractional σ_Mz = **8.8e-8** [MEASURED] — i.e. the Fisher matrix itself already says ~1e-7, so the constant is moot | None directly. But note: a 1e-7 GW mass vs 0.24 dex catalogue mass means the mass channel is 100%-catalogue-limited; combined with I4 the mass match is unrealistically clean in the 4D channel | No (dead-code status newly established) | MEASURED |
| **I6** | GW measurement scatter on (φ, θ, d_L, M_z) | `scripts/prepare_detections.py:115` → `datamodels/detection.py:161-240` (`convert_to_best_guess_parameters`, correlated 4D MVN from the CRB covariance) | Same | **REALISTIC — verified.** `prepared_fixed.csv` d_L pulls vs dist(z_host, 0.73): mean −0.06, std 1.19 (n=76), consistent with N(0,1). d_L error is the **marginal** (inverse of the full 14-param Fisher, `detection.py:135`), not the conditional. Not called inside the evaluation itself — applied once at the prepare step; the truth CSV is never fed to `--evaluate` in the production recipe | — (this is the one channel where the pipeline *does* realize noise) | Yes | MEASURED |
| **I7** | Sky localization | `constants.py:70` (`SKY_LOCALIZATION_ERROR = 2°`) — **dead constant, no consumers**; real path `handler.py:446-548` (Fisher sky error ellipse, `sigma_multiplier=1.5` at `bayesian_statistics.py:2670`) | Same | Candidate search uses the per-event Fisher sky ellipse (median ΔΩ = 0.99 deg² [MEASURED]); sky part of the GW Gaussian is the Fisher 2×2 block. Reasonable. The 1.5σ ball radius is *tight* (can drop the true host — conservative, not flattering) | ~0 | Yes | MEASURED |
| **I8** | Weak lensing absent from the d_L budget | `detection.py:114` (`WL_uncertainty = 0.0`, never populated); no lensing scatter applied to injected d_L | σ_WL/d_L ≈ 0.066·((1−(1+z)^−0.25)/0.25)^1.8 (Hirata et al. 2010): ~0.05% at z=0.07, ~0.2% at z=0.15, ~1% at z≈0.5 | No lensing width, no lensing realization | Small for the current information carriers (z ≤ 0.15): would degrade the golden events (0.09–0.11% GW error, σ_WL ~ 0.006% at z=0.02) negligibly; matters (≳ GW error) only for z ≳ 0.2 hosts, which currently carry no information. In a *fixed* pipeline where deeper hosts contribute, it must be added | No | ESTIMATED / LITERATURE |
| **I9** | Dark hosts / completion term | `dark_siren_injection.py:77-85, 429-520`; completion numerator `bayesian_statistics.py:3046-3110` | Same structure (Gray et al. 2020) | **No truth leakage found**: a dark host's z/z_error are never read by the inference (documented at `dark_siren_injection.py:77-85`); B_num integrates the same (1−f_k)·p_pop density the injection samples. Measured contribution of all 1514 dark events to the H0 curvature: **−1%** | ~0 on σ; the self-consistent completion is why dark events are harmless rather than biasing | Yes | MEASURED |
| **I10** | Forecast-class self-consistencies | inference likelihood = exact CRB Gaussian used to draw the data; identical p_det pool + completeness map in sim and inference (`from_cache_or_build`, injection W_k sampler); Ω_m, population model, rate weights exactly known; h_inj on the h-grid | Real analyses have non-Gaussian PE posteriors, waveform/calibration systematics, population-model misspecification, completeness estimation error | Standard "self-consistent mock" idealizations; each individually defensible for a forecast, but jointly they guarantee zero modeling systematic — MAP-on-truth is then *expected*, not evidence of realism | Bias ≈ 0 by construction; real-world systematic floor ~1–2% on H0 from population/completeness mismodeling alone (e.g. Gray et al. 2020/2023 discussions) | Partly (G7 systematics budget tracks some) | LITERATURE |
| **I11** | Catalogue z_error realism | `handler.py:307-410` (photo σ_z ≈ 0.035 stored; spec 0.0017; PV per class; z_cmb col 28) | Same numbers | The stored *widths* are honest and well-referenced. The problem is upstream: production never consumes them (I1/I2). Flag retained per row (`REDSHIFT_FLAG`), so a spec-z-only analysis is one filter away | — | Yes | MEASURED |
| **I12** | Selection consistency knobs | `PRE_SCREEN_SNR_FACTOR = 0.0` (disabled, `constants.py:63`); `SNR_THRESHOLD = 20`; `use_detection` d_L-error cut < 0.10 (`bayesian_statistics.py:218, 3195`) | Same | Pre-screen off (audited); threshold 20 is standard; the 10% d_L cut removes events that would otherwise carry little info. No flattering effect found | ~0 | Yes | MEASURED |

---

## 3. Quantification: what would the realistic number be? [MEASURED counterfactuals]

Per-event information I_e = (h·σ_eff)⁻², σ_eff² = (σ_dL/d_L)² + (σ_z,eff/z)², over the 76
in-catalog events (dark events contribute nothing). Scenario A reproduces the measured pipeline
curvature (0.027 vs 0.032 measured — 3-pt curvature vs Fisher, ✓ cross-check):

| scenario | z-error model | σ_H0 [km/s/Mpc] | vs quoted |
|---|---|---|---|
| **A — pipeline as run** | z exact (δ at catalogue z) | **0.027–0.032** | 1× |
| B — dream follow-up | spec-z σ_z = 0.0017 + 150 km/s corrected PV | **0.22** | ~8× worse |
| C — spec-z, uncorrected PV | spec-z + 500 km/s PV | **0.30** | ~10× worse |
| D — the catalogue as it actually is | each host's own σ_z (all 76 are photo-z, σ_z ≈ 0.035) | **3.6** | ~110× worse |

Caveats making even these *optimistic*: (i) with real z scatter the loudest low-z events lose the
most (top-3 events: PV alone is 24–110× their GW error); (ii) with σ_z/z ≈ 0.5 the host
*association* itself becomes ambiguous — the D row assumes the z-marginalization still centres on
the right host, which the earlier photo-z railing episodes (`handler.py:329-338` caveat,
BRIDGE-FINDINGS) show it does not gracefully do; (iii) no lensing, no population/completeness
systematics included. A defensible realistic forecast from this exact event set is therefore
**σ_H0 ≈ 0.2–0.3 km/s/Mpc if one assumes complete spectroscopic follow-up of every candidate
host, and ~1–4 km/s/Mpc with the GLADE+ catalogue as it exists** — consistent with the LISA-EMRI
dark-siren literature (Laghi et al. 2021, arXiv:2102.01708: ~1–3% on H0 ≈ 0.7–2 km/s/Mpc with
4-yr LISA and idealized spec-z hosts; MacLeod & Hogan 2008: ~1%). [LITERATURE]

Also note the **event-count optimism**: 1590 detections/seed with 76 golden SNR-300–1400 hosts at
z < 0.15 sits at the optimistic end of EMRI rate models (Babak et al. 2017 M1); σ_H0 scales as
1/√N on top of everything above. [LITERATURE]

---

## 4. Bottom line

**The quoted σ_H0 = 0.03 km/s/Mpc is an artifact of idealized host-property treatment, not a
realistic forecast.** Specifically: 100% of the information comes from 76 in-catalog events whose
photometric redshifts (σ_z/z ≈ 50%) are treated as *exact* — the injected truth is the catalogue
value (I1) and the production likelihood point-evaluates at it (I2), with no PV realization (I3).
The estimator is *internally* self-consistent ("generator-exact"), MAP-on-truth is guaranteed by
that self-consistency, and σ reflects only the GW d_L errors of a handful of golden events. The
"very bad galaxy mass measurements" similarly never scatter anything (I4) — though they happen
not to touch the quoted (3D-channel) number. A true realistic simulation of the same universe
would deliver **σ_H0 ≈ 0.2–0.3 km/s/Mpc at best (full spec-z follow-up) and more honestly
~1–4 km/s/Mpc with GLADE+ photo-z hosts** — 8× to 100× the quoted width.

### Fixes ranked by impact

1. **Realize the z noise (biggest, changes everything).** In the injection: draw the *true* z
   latently (e.g. z_true ~ N(z_cat, σ_z,cat) ⊕ PV realization, or equivalently keep z_true and
   scatter the *catalogue entry*), place the event at d_L(z_true, h_inj), and hand the inference
   only the catalogue (noisy) z + σ_z. This kills the delta-kernel's license (`generator_marginal`
   "point" resolution, `bayesian_statistics.py:138`) — the numerator must become a genuine
   σ_z-wide marginal again, and the photo-z-domination problem it was sidestepping has to be
   solved physically (spec-z host selection, or a photo-z-robust estimator), not definitionally.
2. **Realize the PV** (part of 1, but independent even for spec-z hosts): add (1+z)·v/c with
   v ~ N(0, 150–500 km/s) per correction class to the injected z; keep the matching width in the
   kernel. Dominant error for every z < 0.05 golden event.
3. **Realize the 0.24 dex mass scatter** (needed before quoting any with-BH-mass-channel result):
   draw M_true = M_relation·10^(0.24·N(0,1)) at injection, keep the catalogue mass as the
   observation. Also re-examine the ball-tree mass filter under this scatter.
4. Add weak lensing to the d_L budget (width + realization) — relevant once deeper hosts carry
   information after fixes 1–3.
5. Report the forecast with an event-rate uncertainty band (rate model optimism dominates the
   √N scaling).

*Everything in §1/§3 is reproducible via `audit_information_decomposition.py` in this directory
(requires the repo venv and the reduced catalogue; runtime ~2 min, ~8 GB RAM for the catalogue
read).*
