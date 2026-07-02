# p_sample == p_comp Consistency Audit (post Changes 1–5 + Task A)

**Date:** 2026-07-01
**Branch:** physics/photoz-joint-normalisation (audit reads code at/after HEAD of physics/partition-norm-restructure — Change 5 shipped)
**Scope:** Does the GENERATIVE model (event sampling, `p_sample`) draw hosts from the SAME population density that the INFERENCE model (likelihood priors + selection, `p_comp`) integrates? Requirement: Gray et al. 2020 (arXiv:1908.06050) / Mandel–Farr–Gair 2019 (arXiv:1809.02063) — identical prior on BOTH sides across redshift, rate, mass, AND sky.
**Method:** Four per-axis code traces (verbatim generator formula vs inference integrand), spot-verified against current source. Read-only.

---

## 1. Verdict

`p_sample == p_comp` is **now reconciled on three of the four axes** — redshift/rate weighting, the dark (out-of-catalog) channel, and the completeness field `f` — as a direct result of Changes 1–3 (shared analytic M1 rate + rate-weighted catalog draw), 4-i/4-ii (spurious `p_det` removed from the completion numerator, `1/(1+z)` added), 4b (Bernoulli(F) dark-event injection), Task A (partition-norm restructure to a single Gray ratio `p_i = (β_G·L_cat + B_num)/D(h)` with a globalized selection denominator), and Change 5 (pixelated HEALPix `f_k(z,Ω)` from a single frozen `m_th_map_nside32.npy`, loaded byte-identically by both sides — the C1 invariant). On those three axes the per-galaxy weight, the `(1-f)·dVc/(1+z)` dark density, the per-pixel `f_k`, and the population `1/(1+z)` are **byte-for-byte identical** between injection and inference, so no relative reweighting and no H0 bias is introduced. **One axis remains a hard mismatch and is the single paper-blocker: sky/selection.** The generator draws and SNR-selects an anisotropic, physically-real sky (GLADE catalog rows for in-catalog hosts; low-`f_k` Zone-of-Avoidance pixels for dark hosts) through the sky-**dependent** LISA TDI response, whereas every inference selection integral (`D(h)`, `β_Ḡ`, the global catalog denominator) uses a sky-**marginalized**, isotropic-injection `p_det` evaluated at `φ=θ=0`. This substitutes `p(Ω)=isotropic` into the MFG/Gray normalization while the universe/generator sky is not isotropic; because detections are SNR-selected toward high-response sky, `⟨p_det⟩_iso ≠ ⟨p_det⟩_population`, `D(h)` is mis-shaped in the exact H0-carrying variable `d_L(z,h)`, and H0 is biased with uncontrolled magnitude and non-determinate sign. Two further residuals are **low severity** and do not bias the primary (without-BH-mass) posterior: the with-BH-mass combined likelihood mixes 3D and 4D selection integrals, and `B_num` collapses the host sky onto the measured detection pixel. **Bottom line: not fully resolved — the redshift/rate/mass/completeness reconciliation is complete and clean, but the sky axis is still open and, if anything, was WIDENED (not closed) by the COORD-03 frame fix and Change 5, which made the generator sky more anisotropic against a still-isotropic `p_det`.**

---

## 2. Per-axis summary table

| Axis | Generator density (`p_sample`) | Inference integrand / prior (`p_comp`) | Match | Severity | H0 bias |
|---|---|---|---|---|---|
| **incat_z_rate** | `weights = R_eff_per_mbh(M)/(1+z)` over catalog rows `z<0.5`, drawn ∝ weights; true `z,M` = catalog row, no smear (`handler.py:701-705,717-720`; `parameter_space.py:204-209`) | `_rate_weight = R_eff_per_mbh(host.M)/(1+host.z)` (num); global denom `w_g = R_eff_per_mbh(M_g)/(1+z_g)`, `Σ w_g·p_det` over `z<horizon(h)` (`bayesian_statistics.py:171,490`) | **MATCH** | none | **None** — byte-identical weight; z-support consistent (P_det≈0 for 0.18<z<0.5) |
| **dark_channel** | `p_dark(z,Ω) ∝ (1−f_k(z))·dVc/(1+z)`, pixel `k*∝W_k`, mass ∝ `mbh_mass_function·R_eff`, split Bernoulli(F) (`dark_siren_injection.py:194,293-304,352-355,421`) | `B_num = ∫(1−f_{k(Ω_e)})·p_GW·dVc/(1+z)`; `β_Ḡ = ∫(1−f_bar)·p_det·dVc/(1+z)`; `β_G = D−β_Ḡ`; `p_i=(β_G·L_cat+B_num)/D` (`bayesian_statistics.py:1480,372,727,1503`) | **MATCH** (primary) | low | **None** for without-BH posterior; low residual only in with-BH-mass combined likelihood |
| **completeness_f** | ONE `PixelCompleteness` from `from_cache_or_build()`; sky-avg `f_bar` → F; per-pixel `f_k` → dark draw (`main.py:397`; `dark_siren_injection.py:293-304,420-421`) | SAME `from_cache_or_build()`; `f_bar` → `β_Ḡ`/`w_G`; per-pixel `f_k` at event pixel → `B_num` (`bayesian_statistics.py:714,367-372,1444,1472-1480`) | **MATCH** | low | **None** from `f` itself (C1 byte-identity); low residuals: event-pixel delta-collapse, frame assumption, cache-reuse operational risk |
| **sky_selection** | Anisotropic real sky (catalog rows / ZoA pixels) selected by sky-**dependent** LISA TDI response; `passed = snr ≥ threshold` (`handler.py:717-721`; `parameter_space.py:205-206`; `waveform_generator.py:9-10,61-64`; `main.py:539`) | Isotropic-injection `p_det`, **sky-marginalized**: `φ=θ=0` in `D(h)`, `β_G`, global denom; `f_bar` = `(1/Npix)Σ_k f_k` (`bayesian_statistics.py:246-248,473-474,365-366`) | **MISMATCH** | **PAPER-BLOCKER** | **Yes** — magnitude uncontrolled, sign non-determinate; biases H0 (not just rate) via mis-shaped `D(h)` in `d_L(z,h)` |

---

## 3. Ranked residual mismatches (fix direction + file:line)

### R1 — PAPER-BLOCKER: sky-marginalized isotropic `p_det` vs anisotropic generator sky
- **What:** MFG/Gray require `D(h) = ∫ p_det(d_L(z,h),M)·p_pop dz` to integrate the TRUE detection probability over the TRUE population sky. The code integrates `⟨p_det⟩_iso` (isotropic-injection survival, `φ=θ=0`), while the generator/universe sky is the anisotropic GLADE catalog (in-catalog) and low-`f_k` ZoA pixels (dark), selected through the sky-dependent LISA response. `⟨p_det⟩_iso ≠ ⟨p_det⟩_population`, so `D(h)` and the in-catalog/dark mixing weight `w_G=β_G/D(h)` are mis-shaped in `d_L(z,h)` → H0 shift.
- **Most acute:** `β_Ḡ` (dark/completion term) neglects the covariance between low-completeness (ZoA) sky and LISA response entirely; that lever maps directly onto the headline H0.
- **Note:** COORD-03 (frame fix) and Change 5 (pixelated `f_k`) made the generator sky MORE anisotropic and physically real → WIDENED the gap against the still-isotropic `p_det`, did not close it.
- **Fix direction:** restore real LISA sky dependence in `p_det` and replace the sky-uniform factorization with the per-pixel sum the code's own caveat prescribes — `Σ_k f_k·p_det(z,Ω_k)` — in `β_Ḡ`, `D(h)`, and the global catalog denominator; OR rebuild `p_det` from catalog-snapped (anisotropic-sky) injections so the marginalization is over the correct population. Either requires a joint injection↔inference update.
- **file:line:** `bayesian_statistics.py:246-248` (`D(h)` `φ=zeros #marginalized`), `:365-366` (load-bearing caveat: *"Valid because p_det is sky-uniform; if real LISA sky dependence is restored this must become sum_k f_k p_det(z,Omega_k)"*), `:473-474` (global denom sky-marginalized); generator side `parameter_space.py:205-206`, `waveform_generator.py:9-10,61-64`, `main.py:539`; injection isotropy `main.py:711,724`, `parameter_space.py:24,123`; `simulation_detection_probability.py:262,776,929` (D-02 marginalized). Status: `HANDOFF-PAPER-MILESTONE-20260630.md:137-139` (§5.2 sky-population blind spot).

### R2 — LOW: with-BH-mass combined likelihood mixes 3D and 4D selection integrals
- **What:** `combined_with_bh_mass = (β_G·L_cat_with_bh_mass + B_num)/D_h` reuses the **without**-mass `B_num`, **without**-mass `D_h`, and **without**-mass `β_G`, while multiplying `L_cat_with_bh_mass` (normalized by the with-mass `Σ_global^withbh`). The would-be scale-free multiplier is `Σ_global^withbh`, not `β_G^nobh`, so the catalog/dark relative normalization in the with-mass posterior mixes 3D and 4D selection integrals.
- **Impact:** does NOT affect the primary without-BH-mass channel (clean mass-marginalization). Potential low-severity relative-normalization effect only in the with-BH-mass posterior.
- **Fix direction:** build a with-mass `B_num` and with-mass `D_h`/`β_G`, or renormalize `L_cat_with_bh_mass` by the consistent with-mass selection integral so the mixing weight is scale-free in the 4D channel.
- **file:line:** `bayesian_statistics.py:1503-1504` (`combined_with_bh_mass = (beta_G * L_cat_with_bh_mass + B_num) / D_h`), `:490` (`global_table[h]` with-mass branch).

### R3 — LOW: `B_num` sky delta-collapse onto the measured detection pixel
- **What:** `B_num` evaluates per-pixel `f_k` at the MEASURED detection direction (`ang2pix(detection.phi, detection.theta)`) rather than the true host pixel. Only matters if GW localization straddles pixels of sharply different `f_k` (3.36 deg² cells at NSIDE=32) — second order.
- **Secondary:** the event-pixel `ang2pix` assumes `detection.phi/theta` live in `BarycentricTrueEcliptic` (the frame the `m_th` map was built in) — a load-bearing frame assumption.
- **Fix direction:** marginalize `B_num` over the GW sky-localization posterior instead of collapsing to the point estimate; assert/verify the detection-frame convention at the `ang2pix` call site.
- **file:line:** `bayesian_statistics.py:1444` (event pixel), `:1472-1480` (`B_num` with per-pixel `f_k`); frame `pixel_completeness.py:45-49`.

### R4 — LOW / OPERATIONAL: C1 byte-identity depends on reusing the committed cache
- **What:** unbiasedness on the completeness axis holds only while BOTH sides reuse the committed frozen `m_th_map_nside32.npy` (98432 bytes, git-tracked). A silent rebuild from a divergent catalog on one side would break C1 and reintroduce an `f`-mismatch.
- **Fix direction:** keep the `.npy` committed as the SOLE source; add a startup md5/shape assertion that both injection and inference load the identical map (currently mitigated by the committed artifact only).
- **file:line:** `pixel_completeness.py:39-43` (C1 requirement), `:100-102,461-483` (`from_cache_or_build`), `:286` (`f_bar=Σ_k f_k/npix`); loaders `main.py:397`, `bayesian_statistics.py:714`.

### R5 — RESIDUAL (from original audit, carry-forward): emcee proposal RNG unseeded
- **What:** `--seed` now threads an `rng` through `main.py:88-93 → Model1CrossCheck(rng=rng) → randomize_parameters(rng=...)` and the catalog/host draws, and initial walker positions `p0` use `self._rng`. BUT the `emcee.EnsembleSampler` is constructed WITHOUT the rng, so its internal MCMC proposal moves are not seeded by `--seed`.
- **Impact:** catalog/host/parameter draws ARE now reproducible; only the emcee `(M,z)` proposal chain remains non-reproducible. Not a `p_sample`/`p_comp` mismatch — a reproducibility residual.
- **Fix direction:** thread `self._rng` into the emcee sampler (e.g. seed emcee's internal RNG / pass `seed=`), or seed the global numpy state at sampler construction.
- **file:line:** `cosmological_model.py:165-168` (rng in ctor), `:261-265` (`p0` seeded), `:270-274` (`EnsembleSampler` NOT passed rng), `:283` (`run_mcmc`).

---

## 4. Original audit findings — RESOLVED vs OPEN

Reference: memory `inference-consistency-audit` (2026-06-25).

| # | Original finding | Status | Evidence |
|---|---|---|---|
| **1** | **The `dVc/dz` linchpin** — generator samples `p_sample ∝ dN_dz_of_mass(M,z)·R_emri(M)` (M1 rate, mass+z shaped) but inference `L_comp`/`D(h)` integrate bare `dVc/dz` (no R(z), no mass, no `1/(1+z)`) | **RESOLVED** | Both sides now share `emri_rate.py`. Per-galaxy weight `R_eff_per_mbh(M)/(1+z)` identical (gen `handler.py:701-705`; inf `bayesian_statistics.py:171,490`). Completion denominator integrates `p_det·dVc/(1+z)` with the mass-integrated rate cancelling num/denom, `1/(1+z)` retained (`bayesian_statistics.py:240-262`). Mass axis carried by `w_g` on both sides. Change 1–3 + 4-ii. |
| **2** | **Isotropic-sky `p_det` blind spot** — `p_det`/`β(H0)` built from isotropic-sky injections; real detections have catalog-snapped anisotropic sky; LISA SNR is sky-dependent; Malmquist normalization averaged over wrong sky | **STILL OPEN — PAPER-BLOCKER** | This is axis `sky_selection` / residual R1. Inference still `φ=θ=0` sky-marginalized (`bayesian_statistics.py:246-248,473-474`); generator anisotropic + sky-dependent response (`parameter_space.py:205-206`, `waveform_generator.py:61-64`, `main.py:539`). WIDENED by COORD-03 + Change 5. Load-bearing caveat `bayesian_statistics.py:365-366`. |
| **3** | **`f(z)` on a mass-pruned sub-catalog** — full GLADE+ B-band `f(z)` applied to a mass-pruned catalog → `f_i` over-estimated → over-credits `L_cat`; plus flat-extrapolation beyond horizon; scalar `f_i=completeness(z_det)` mixing weight | **RESOLVED (by restructure)** | The scalar `f_i` mixing weight was DROPPED in Task A (replaced by event-independent `w_G=β_G/D(h)`), so the over-crediting mechanism is structurally gone. Change 5 replaced `f(z)` with a per-pixel `f_k(z,Ω)` from a threshold estimator on the (re-parsed) catalog, entering symmetrically on both sides (C1 byte-identity, `pixel_completeness.py:39-43,286`). Design decision `f=f(z,Ω)` only (NOT `f(z,Ω,M)`) decouples mass (in `w_g`) from completeness. **Caveat (non-blocking):** `⟨f_k⟩_Ω ≈ 2× Dalya-2022` (f_bar≈0.71 vs 0.37 @ z=0.1) — a population/depth difference, NOT an H0 bias (unbiasedness is by injection↔inference self-consistency + C1, not `f_model=f_real`). |
| **4** | **`--seed` non-reproducibility** — emcee proposals + sky draws unseeded | **PARTIALLY RESOLVED** | rng threaded `main.py:88-93 → Model1CrossCheck(rng) → randomize_parameters` + catalog/host draws seeded; `p0` uses `self._rng` (`cosmological_model.py:261-265`). BUT emcee `EnsembleSampler` internal proposal RNG still unseeded (`cosmological_model.py:270-274`) — residual R5. |

**Also from the original "other real issues" list:** the `L_cat` numerator/denominator z-window inconsistency (Gray A.9/A.10) is addressed — Task A **globalized** the selection denominator (`Σ_global w_g·p_det`, `bayesian_statistics.py:457-490`), so `n_gal` cancels and the local/global window split is by design (local candidate ball for the numerator vs global horizon for the denominator). The `H=0.73` vs `TRUE_HUBBLE_CONSTANT` bookkeeping split and the outdated `Ω_m=0.25` cosmology were flagged "matched both sides" (no gen/inf mismatch) and are out of scope for this axis audit.

**Genuinely fine (unchanged, do not chase):** cosmology consistent both sides; `M_z=M(1+z)` consistent; `p_det` correctly denominator-only (MFG mistake avoided, re-verified — spurious `p_det` removed from `B_num` in Change 4-i); SNR threshold consistent between generator and the horizon set that builds `p_det`.

---

## 5. Do-not-overclaim notes

- Axes `incat_z_rate` is the **only** fully clean (severity none) axis. `dark_channel` and `completeness_f` are MATCH **with low-severity residuals** (R2, R3, R4) — reconciled for the primary without-BH-mass posterior, NOT perfectly for the with-BH-mass posterior and not for the sky-collapse/cache-operational corners.
- `sky_selection` is **unresolved**. Its H0-bias sign is **not determinate** without a numerical antenna-vs-catalog-sky correlation; only the mechanism (mis-shaped `D(h)` in `d_L(z,h)`) and its paper-blocking status are established.
- The Change-5 closure test validates the INFERENCE side conditional on a correct injection (it injects the true `(z,pixel)` joint directly); FIX-A (the dark-draw joint `(z,Ω)`) is validated separately at the joint-distribution level, NOT via the closure. Neither test exercises the sky-selection mismatch (R1), because both sides of the test share the same sky-marginalized `p_det` — the closure cannot witness R1.

---

## Provenance

Traces verified against current source (spot-checks this session): `bayesian_statistics.py:240-262` (rate-weighted `1/(1+z)` completion denominator), `:365-372`/`:470-490` (sky-marginalized `β_Ḡ` + global denom + caveat comment), `:1503-1504` (with-BH combined likelihood reuses without-mass `B_num`/`D_h`), `cosmological_model.py:165-168,261-274` (rng threaded, emcee unseeded), `main.py:88-93` (seed→rng). Original findings from memory `inference-consistency-audit`.
