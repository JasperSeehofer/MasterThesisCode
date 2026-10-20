# [PHYSICS] Gate presentation — completion-leg population prior, `completion_population_prior ∈ {"comoving","m1"}`

**Charter node B3.2 [POP].** Launched under rows #222/#223 — charter node B3.2. Presentation
**before code** (row #223: "presentation before code + ledger rows as always"). **No code has
been written**; the physics-trigger files `darksiren_emri/bayesian_inference/bayesian_statistics.py`,
`darksiren_emri/cosmological_model.py`, `darksiren_emri/dark_siren_injection.py` are untouched.
This document returns to the orchestrator; it never addresses the author.

**Date:** 2026-08-29 · **HEAD:** `dd63fe0c` (working tree dirty only in untracked `results/` and
`docs/CLAUDE_SCIENCE_*` files; no tracked source modified by this worker).

## 0. Approval stamp and F3 statement

- **Approval column of the gate ledger:** `row #223 (standing grant, charter node B3.2)`.
- **F3 (adopted row #222):** every prediction in §6 is registered here BEFORE any run of the
  shared S0-B instrument (docket §3 L1: iiib, h = 0.730, CoR-P) or the C2 arm (docket §4.3:
  h ∈ {0.720, 0.730, 0.740}, iiib). Nothing in §6 is derived from an m1-prior run; none exists.
- **Disposition in one line (details §F):** the premise the flag was commissioned on — that the
  production event set is injected from the Barausse/Babak M1 *z-shape* while the estimator
  assumes constant comoving density — is **REFUTED by generator provenance at the generating
  commit**. The completion leg's prior is already the production generator's own dark-host law.
  The flag as specified is therefore either a **no-op (NULL-BY-CONSTRUCTION)** or a
  **generator-inconsistent counterfactual**, depending on which of the repository's two "M1"
  objects is read into it. The five gate items are still presented (§2–§5) so the orchestrator
  can decide whether the counterfactual is worth building as an *instrument-validation* arm; this
  presentation does **not** authorise code for an adoption path.

---

## F. Finding that precedes the five items — which density the mock injects (A11, every claim with a line)

The charge asked for "the exact density the mock injects, incl. any mass dependence and the
(1+z) time-dilation factor", read from `cosmological_model.py:Model1CrossCheck`. Reading the
generator end-to-end shows **two different "M1" densities in the codebase, used by two different
pipelines**, and the production event set is drawn from the one that is *already* the estimator's
prior:

| object | definition (file:line, HEAD `dd63fe0c`) | z-shape | used by |
|---|---|---|---|
| **M1-(i)**, analytic `emri_rate.R_EMRI(z, M)` | `darksiren_emri/emri_rate.py:264-296`: `dn/dlog10M(M) · R_eff(M) · p0(M,z)`, `p0 ≡ 1` surrogate (`:202-232`); "With the default p0 = 1 surrogate the density is z-independent" (`:280`) | **constant comoving**; the caller multiplies by `1/(1+z) · dVc/dz` exactly once (`p_pop_unnormalized`, `:300-334`) | the **production generator's dark-host draw** (`dark_siren_injection._redshift_population_weight`, `:177-194`: `dVc/dz/(1+z)`; `_draw_dark_redshifts`, `:309-329`: density `(1−f̄(z))·dVc/dz/(1+z)`; per-pixel twin `_draw_dark_hosts_pixelated`, `:411-459`: `(1−f_k(z))·p_pop(z)`), the dark mass marginal (`dark_mass_log10_density_unnormalised`, `:332-368`), the in-catalogue draw weight `R_eff(M_g)/(1+z_g)` (`handler.py:1021-1076`), AND the estimator's completion legs (§1) |
| **M1-(ii)**, extracted `Model1CrossCheck.emri_distribution` | `darksiren_emri/cosmological_model.py:249-290`: `dN_dz_of_mass(M, z) · R_emri(M)`, with `dN_dz_of_mass` a degree-9 polynomial fit in z per log10-mass anchor (4.5/5.0/5.5/6.0/6.25) from `merger_distribution_coefficients` (`:67-124`, "coefficients of polynomial fit of dN/dz for different mass bins"), clamped at z > 3 (`:139-143`); `R_emri` a 3-segment power law (`:284-290`); sampled by emcee in `(log10 M, z)` on `0 < z < 1.5`, `1e4 < M < 1e7` (`:292-352`) | **z-dependent** (the extracted curve already contains Barausse's cosmological volume and time-dilation factors at h = 0.704 — `docs/gates/G7_systematics_budget.md:25`; no separate `1/(1+z)` is applied in the sampler) | **only** the SNR-only injection POOL, stratum 'a' (`main.py:1125-1146`, `injection_campaign`, `:905-960`: "stratum 'a' (0.50): the status-quo Babak M1 emcee population draw — the ONLY stratum valid for pool-marginal estimator legs") — i.e. the p_det survival estimator's sampling measure, never an event's (z, M) in the CRB set |

**The production event set is M1-(i)-drawn, not M1-(ii)-drawn.**

1. `main.py:434-640` (`data_simulation`, the `--simulation_steps` path) never calls
   `sample_emri_events`; every event's `(z, M)` is its host's: `draw_mixture_hosts` (`:577-586`) →
   in-catalogue rows via `draw_rate_weighted_hosts` with probability `F` or dark hosts via
   `draw_dark_hosts` with probability `1−F`; then `set_host_galaxy_parameters(host_galaxy, h)`
   (`:592`). `sample_emri_events` is called only in `injection_campaign` (`:1131`).
2. **At the generating commit of the CRB set of record** — `03cfe800` (2026-07-29;
   `cluster/datasets.yaml:131-137`, `campaign51_seed6x000`, "git_commit: 03cfe80
   (seed61000/62000/63000)", `simulation_steps=40`, `evaluate=False`, `h_value 0.73`) — the same
   is true: `git show 03cfe80:master_thesis_code/main.py` has `draw_mixture_hosts` at `:439`/`:517`,
   `in_catalog=host_galaxy.catalog_index != -1` at `:772`, `sample_emri_events` only at `:1071`
   (injection campaign); `git show 03cfe80:master_thesis_code/dark_siren_injection.py:328`
   is `density = (1.0 - f_z) * _redshift_population_weight(z_grid, h)`. {verified 2026-08-29}
3. **Artifact-level corroboration** (`results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`,
   md5 `9a1f2a14384a9281c97ca3be312ddaab`, verified 2026-08-29 = the md5 of record,
   `MEASUREMENT_HEAD_READOUT_20260827.md:42-43`): 1590 rows carry the CHANGE-4b columns
   `host_galaxy_index` and `in_catalog`, which only the `draw_mixture_hosts` path writes
   (`03cfe80:main.py:768-772`). **1514 rows are dark (`in_catalog = False`, `host_galaxy_index = −1`
   in all 1514) and 76 are in-catalogue** (`host_galaxy_index` ∈ [846 453, 19 337 626]).
   Dark-event `z_true = dist_to_redshift(d_L, h = 0.73)` quantiles (0/5/25/50/75/95/100 %):
   0.058 / 0.233 / 0.371 / 0.500 / 0.633 / 0.742 / 1.110; in-catalogue: 0.016 … 0.150 (max).
   The injection-side in-catalogue Bernoulli probability recomputed from the frozen completeness
   cache is **F(h = 0.73, z_max = 1.5) = 0.01754** (`compute_global_catalog_fraction`,
   `dark_siren_injection.py:241-303`, computed 2026-08-29); the detected fraction 76/1590 = 0.0478
   is the SNR-selection enrichment of the nearby in-catalogue hosts, consistent with F.
4. **The ledger already holds evidence in this direction, read the other way at the time:**
   row #139 item 3 (`BIAS_HISTORY_LEDGER.md:1404-1411`): "B-OUT matches the estimator's
   POPULATION (`population_z_weights` = dV_c/dz/(1+z), byte-identical to production's `_w_pop_eff`
   bare form)" — B-OUT *reproduced* production's dark rail (0.6007 vs 0.6001, row #139 item 2);
   row #144 items 1–2 (`:1546-1558`): the comoving-drawn, survival-weighted mirror B-SEL's
   z-distribution is consistent with production's dark class (D = 0.0792, **p = 0.225** at n = 174;
   measured mean-z shift **+0.018** where the M1 attribution "would need ≈ 0.17"), while row #137's
   production-vs-pool-stratum-'a' comparison (D = 0.048 at n = 1588) is a **p = 0.0013 rejection**
   of equality (row #144 item 1, `:1552-1553`) — i.e. production events do NOT follow the pool's
   M1-(ii) measure at the level that check could see.

**Consequence for the B3 branch.** Row #138's memo (`docs/derivations/population_mismatch_dark_score.md`
§1: "The events, however, are injected from the Barausse (2012) M1 EMRI rate") and B3.1's
re-derivation (`B3_1_POP_RECORD.md` §1: `w_true` = the z-marginal of `Model1CrossCheck`'s emcee
density) both took **the p_det pool's stratum-'a' sampling measure** for the production events'
law. For the dark class of `seed61000` the true law is `(1−f)·dVc/dz/(1+z)` (M1-(i)), which is
**byte-for-byte the estimator's completion prior** (`bayesian_statistics.py:1203`, `:1350-1355`,
`:5298-5303`: "EXACTLY the dark population the generator draws"). The population term
`Δscore = [d ln(w_model/w_true)/dz]·dz*/dh` is therefore **identically zero for production's dark
class**, by construction. B3.1's 98.5 %/103.9 % coverage is re-read in §6.0: it is a
premise-independent statement about the estimator's *algebraic response* to swapping the prior,
not evidence that the swap removes a mismatch that exists in the data.

Whether the two "M1"s *should* agree is a separate, real question (M1-(ii)'s extracted dN/dz vs
M1-(i)'s `p0 = 1` constant-comoving surrogate differ by the shape ratio r(z) tabulated in §6.0,
0.53 → 1.39 over z ∈ [0.17, 1.5]) — that is the astrophysical-model uncertainty G7 row 16 is
about (§12), not a defect of either pipeline against the other.

---

## 1. Which legs the prior touches — the consistency argument and the decision

**Principle applied:** the estimator must use ONE population density at every site where it
integrates over hosts it has not seen, and that density must be the generator's law for the hosts
that site integrates over.

**Inventory of every `dVc/dz/(1+z)` site in `bayesian_statistics.py` (HEAD `dd63fe0c`)** and the
hosts each integrates over:

| # | site | line(s) | integrates over | production path (CoR-P: `absolute_marginal`/`volume_deconv`/`fused`/`phi`, `run_metadata_21.json`) | touched by the flag? |
|---|---|---|---|---|---|
| S1 | `precompute_completion_denominator` integrand `_denom_integrand` | `:1262-1298` (`return … * dVc / (1.0 + z)` at `:1298`) | unseen hosts, full volume (legacy `D(h)`) | computed; feeds `w_G_legacy` diagnostics and the with-BH channel (`:5178-5190`: "phi" swaps only the no-BH divisor) | **yes** |
| S2 | `precompute_missing_completion_denominator` `_missing_denom_integrand` | `:1403-1450` (`:1434` sky-aware, `:1450` isotropic) | unseen (dark) hosts, `(1−f)` volume (`β_Ḡ`) | computed; legacy partner of S1 | **yes** |
| S3 | `precompute_phi_selection_integrals` | `:2077-2130` (`p_pop` at `:2120`; `β_G^φ`, `β_Ḡ^φ` at `:2123-2124`) | unseen hosts, φ-convention legs `β_G^φ`, `β_Ḡ^φ`, `D̃^φ = β_G^φ + β_Ḡ^φ` | **the production no-BH denominator** (`:5185-5190`) and the catalogue-leg calibration `n̄_w = Σ^φ/β_G^φ` | **yes** |
| S4 | `p_Di` completion numerator `completion_numerator_integrand` | `:5345-5410` (`return (1.0 - f_z) * p_gw * dVc / (1.0 + z)` at `:5410`); wrapped by `…_sel_1d` (`× S̄_φ`, `:5432-5449`) and `…_with_bh_mass` (`× g_i`, `:5450-5470`) | the unseen host of THIS event (`B_num`, `B_num^{1d}`, `B_num_wbh`) | **the production dark-class numerator** | **yes** |
| S5 | `precompute_completeness_population_volume` | `:1545-1625` (`:1607`) | unseen hosts (draw-side `V_f(h)`) | `generator_marginal` mode only — not CoR-P | yes (or the flag raises under that mode — §2 option) |
| S6 | `_smeared_global_pdet_expectation` | `:1638-1730` (`:1718`) | unseen hosts, smeared Σ | `smear_global_selection=False` in CoR-P | yes (same note) |
| K1 | scalar host-z kernel `_w_pop_eff` | `:6511-6519` (`w_pop = dVc/(1+z)`, `× f_k`) | the **true z of a SEEN catalogue host** (photo-z deconvolution prior, C7-core) | catalogue leg, scalar twin | **NO** |
| K2 | batched host-z kernel, denominator nodes | `:7171-7180` (`w_pop_den`) | same as K1 (batched) | **catalogue leg, production dispatch** | **NO** |
| K3 | batched host-z kernel, numerator nodes | `:7222-7247` (`w_pop_num_1d` at `:7229`) | same | production dispatch | **NO** |
| K4 | batched with-BH kernel `_z_prior_pdf_at` | `:7567-7589` (`w_pop_bh` at `:7570`) | same, 2D channel | production dispatch (2D) | **NO** |

**Decision: the flag touches the completion family S1–S6 only; it does NOT touch the catalogue
leg's host-z kernel K1–K4, and it does NOT touch the per-host rate weight.** Reasons:

1. **Generator law of the hosts each site integrates over.** S1–S6 integrate over hosts the
   catalogue does not contain; the generator draws those from `(1−f)·p_pop` with `p_pop` = M1-(i)
   (§F). K1–K4 integrate over the *true redshift of a catalogued host*; the generator's law for
   catalogued hosts is **the catalogue row itself** (`draw_rate_weighted_hosts` "copies catalogue
   rows verbatim … NO nearest-neighbour snap", `handler.py:1054-1058`; `bayesian_statistics.py:3925-3928`),
   weighted by `R_eff(M_g)/(1+z_g)`. No z-shape beyond `1/(1+z)` enters the catalogue draw under
   *either* M1 object, so a z-shape switch has no generator referent on the catalogue leg. The
   kernel's `f_k(z)·dVc/dz/(1+z)` is the C7-core partition of the population intensity
   (`:7147-7151`), ratified by G2b as "the unique weight consistent with the project's own rate
   model … exactly h-independent" (`BIAS_HISTORY_LEDGER.md:197`, exoneration C7 — quoted in §10).
2. **Where the catalogue leg already carries a rate factor, and whether it is M1-consistent.**
   `_rate_weight(host) = R_eff_per_mbh(host.M)/(1 + host.z)` (`:1036-1061`) — "IDENTICAL to the
   weight used by the rate-weighted simulation host draw" — and its global twin `W_cat`
   (`:1470-1543`, `:1531-1532`). `R_eff_per_mbh` is the Babak et al. (2017) M1 per-MBH rate
   (`emri_rate.py:235-261`). It is M1-(i)-consistent **and byte-identical to the generator's draw
   weight**; it carries the `1/(1+z)` time dilation explicitly and no other z-shape. The mass
   function `dn/dlog10M` is not applied to catalogue hosts because "each catalog galaxy is ONE
   realised MBH: the mass function is already sampled by the catalog itself" (`handler.py:1043-1048`).
   Nothing to change.
3. **Partition identity under the switch.** `D̃^φ = β_G^φ + β_Ḡ^φ` and the catalogue-leg
   calibration `n̄_w = Σ^φ/β_G^φ` all read `p_pop` at S3; `B_num` reads it at S4; the legacy
   `D = β_G + β_Ḡ` at S1–S2. Switching S1–S6 together keeps every identity the estimator relies on
   (`D = β_G + β_Ḡ` on one domain, `w_G = β_G/D`, `p_i = (A_i + B_num)/D̃^φ`) exact under either
   value of the flag; the K-sites are per-galaxy renormalised (`Z_g`, `:6529`) and enter no
   partition identity, so leaving them alone breaks nothing.
4. **Why touching K1–K4 would be wrong even if S1–S6 were switched:** their prior would then be
   inconsistent with the generator's catalogue law (item 1) *and* it is the exonerated
   "numerator-only kernel change" class (`BIAS_HISTORY_LEDGER.md:197`: "numerator-only kernel
   changes are the exonerated class (#37, #70)").

**A13 engagement (recorded for a future builder, not executed):** the flag must be threaded to
S1–S4 (the production dispatch) and either to S5–S6 as well or guarded (`NotImplementedError` if
`completion_population_prior != "comoving"` with `normalization_mode == "generator_marginal"` or
`smear_global_selection=True`), so an m1 run cannot silently mix priors. The per-term diagnostics
already emitted (`B_num`, `D_tilde_phi`, `beta_G_phi`, log lines "phi-convention legs(h=…)") are
the observable engagement channel (A22 (d)).

---

## 2. OLD formula (verbatim, with lines) and NEW formula

### 2.1 OLD — the constant-comoving factor at the four production sites

S1, `bayesian_statistics.py:1289-1298`:
```python
            dVc: npt.NDArray[np.float64] = np.atleast_1d(
                np.asarray(comoving_volume_element(z, h=_h), dtype=np.float64)
            )
            # Population prior R_EMRI(z,M)/(1+z) * dVc/dz (emri_rate.p_pop_unnormalized):
            # the 1/(1+z) is the source->detector time dilation. The mass-integrated
            # rate INTEGRAL dM R_EMRI(z,M) is z-independent under the p0=1 surrogate, so it
            # is an overall constant that cancels in L_comp = comp_num/D(h); only 1/(1+z)
            # survives here. Babak et al. (2017), arXiv:1703.09722 (rate); Mandel-Farr-Gair
            # (2019), arXiv:1809.02063 (detector-frame rate density).
            return np.asarray(p_det, dtype=np.float64) * dVc / (1.0 + z)
```
S2, `:1434` and `:1450`:
```python
                return np.asarray(integrand, dtype=np.float64) * dVc / (1.0 + z)
            ...
            return (1.0 - f_z) * p_det * dVc / (1.0 + z)
```
S3, `:2120-2124`:
```python
        p_pop = np.asarray(_redshift_population_weight(z_grid, h), dtype=np.float64)
        f_bar = np.clip(np.asarray(completeness.f_bar(z_grid, h), dtype=np.float64), 0.0, 1.0)
        # Eq. (29) / Eq. (33) in Gray et al. (2020) with S_3D -> S_bar_phi.
        beta_G_phi[h] = float(np.trapezoid(f_bar * s_phi * p_pop, z_grid))
        beta_Gbar_phi[h] = float(np.trapezoid((1.0 - f_bar) * s_phi * p_pop, z_grid))
```
S4, `:5392-5410`:
```python
                dVc: npt.NDArray[np.float64] = np.atleast_1d(
                    np.asarray(comoving_volume_element(z, h=h_eval), dtype=np.float64)
                )
                ...
                return (1.0 - f_z) * p_gw * dVc / (1.0 + z)
```
Modelling statement in force, `:1209-1216`: "**constant comoving number density** for the missing
galaxies — the galaxy number density `n_gal(z)` and the mass-integrated rate `INTEGRAL dM R_EMRI(z,M)`
are taken z-independent (the latter exact under the `p0=1` surrogate), so they are overall constants
that **cancel** between the discrete catalogue sums and the continuous integrals (Option A …)".

In one formula, at every S-site the population factor is

    n_com(z; h) = (dV_c/dz dΩ)(z; h) / (1 + z)            [Mpc^3 sr^-1 per unit z]

### 2.2 NEW — one shape-ratio factor, default byte-identical

A single new flag, threaded exactly like `mass_filter_geometry` (class default `:3309`,
`__init__` `:3377`, `evaluate()` signature `:3541`, single validated read site `:3767`,
`arguments.py:394-405` + argparse `:1079`):

- `completion_population_prior: Literal["comoving", "m1"] = "comoving"` — **byte-identical default**.

At every S-site the factor becomes

    n(z; h) = n_com(z; h) · r_flag(z),      r_comoving(z) ≡ 1,
    r_m1(z) = ŵ_M1(z) / ŵ_com(z),

where the hats denote unit normalisation on the tabulation domain `[z_min, HOST_DRAW_Z_MAX] =
[1e-6, 1.5]` at the fiducial `h = 0.73`:

    ŵ_com(z) = n_com(z; 0.73) / ∫ n_com(z'; 0.73) dz',
    ŵ_M1(z)  = w_M1(z) / ∫ w_M1(z') dz',
    w_M1(z)  = ∫_{log10 1e4}^{log10 1e7} dN_dz_of_mass(10^m, z) · R_emri(10^m) dm

— the **z-marginal of the M1-(ii) emcee density in its own sampled coordinates `(log10 M, z)`**
(`cosmological_model.py:249-290`, `:311-313`: `log_probability(x) = _log_probability(10**x[0], x[1])`,
no Jacobian; B3.1's disclosed measure choice, `B3_1_POP_RECORD.md` §1, accepted here), used **as is** —
the extracted curve is already a detector-frame rate per unit z, so **no additional `1/(1+z)` or
`dVc/dz` is applied to it** (applying `R_M1(z)/(1+z)` literally, as the node text suggested, would
double-count the time dilation that the Babak-figure extraction already contains). If instead one
reads "the generator" as M1-(i) (`emri_rate.R_EMRI`, the law the production events are actually drawn
from), then `∫ R_EMRI(z, M) dlog10M` is z-independent and **`r_m1 ≡ 1`: the flag is a no-op**,
NULL-BY-CONSTRUCTION under A15's corollary. The presentation therefore fixes the m1 value to
M1-(ii) and calls it what it is: **the p_det pool's stratum-'a' z-marginal, a counterfactual prior
for the production event set, not a generator-consistent one.**

Normalisation convention: `r_m1` is a fixed, h-independent shape factor with
`∫ n_com(z;h) r_m1(z) dz = ∫ n_com(z;h) dz` at h = 0.73 (unit test, §11). Because
`n_com(z; h) = h^-3 · n_com(z; 1)` exactly (H₀ = 100h km/s/Mpc, Ω fixed), the overall `h^-3`
cancels in every per-event ratio (`B_num/D̃^φ`, `A_i/D̃^φ` via `n̄_w ∝ 1/β_G^φ`), so **only the
z-shape `r_m1(z)` can move a score** — the statement in the node text is confirmed, with the
caveat that the shared denominator's *slope* also moves (§6.1, the Δ_D term).

Implementation hazard (disclosed): `dN_dz_of_mass` is linear, not quadratic, as z → 0 (the
polynomial has a linear term `i·x`, `cosmological_model.py:145-155`), so `r_m1(z) ∝ 1/z` at small z
(`r_m1(1e-6) ≈ 2.1e4`, computed 2026-08-29 on the 3001-node grid). `n_com · r_m1 ∝ z` stays
integrable, but the density is unphysical for a volume-limited population at z ≲ 0.1 — a builder
must tabulate `w_M1` once on the fixed grid (no per-call polynomial evaluation) and must not clip
it, so the counterfactual is the pool's measure verbatim, defects included.

---

## 3. Reference

- **Estimator side (unchanged structure):** Gray et al. (2020), arXiv:1908.06050, Eqs. (29), (32),
  (33), (A.19) — completion numerator/denominator with the population prior `p(z)`; Mandel, Farr &
  Gair (2019), arXiv:1809.02063, Eqs. (5)–(7) — selection evaluated against the population at
  hypothesis; Hogg (1999), arXiv:astro-ph/9905116, Eq. (28) — `dV_c/dz dΩ`.
- **Population side:** Babak et al. (2017), PRD 95, 103012, arXiv:1703.09722 — M1 model, Table I;
  Eqs. (5), (21), (23), (26)–(27), (30)–(31), (34) behind `emri_rate.py` (M1-(i)); the extracted
  per-mass-bin dN/dz behind `cosmological_model.py:67-124` (M1-(ii)); Barausse (2012), MNRAS 423,
  2533, arXiv:1201.5888 — merger history and the h = 0.704, Ω_m = 0.2726 cosmology of the
  extraction (G11, `constants.py:29-37`).
- **Derivation of the predicted response:** `docs/derivations/population_mismatch_dark_score.md`
  §2 (first-order saddle-point formula), re-derived in `B3_1_POP_RECORD.md` §1; the Δ_D companion
  term is derived in §6.1 below (new, this document).
- **Score identity (A12):** `docs/RESEARCH_CYCLE.md:523`.

---

## 4. Dimensional analysis

- `comoving_volume_element(z, h)` → `Mpc^3 sr^-1` per unit z (`physical_relations.py:571-611`,
  "[Mpc]^2 × [km/s] / [km/s/Mpc] = [Mpc]^3 per steradian"). `1/(1+z)` dimensionless.
  `n_com(z;h)` → `Mpc^3 sr^-1` per unit z. `D(h)`, `β_Ḡ(h)`, `β_G^φ`, `β_Ḡ^φ`, `D̃^φ` → `Mpc^3 sr^-1`
  (docstrings `:1218`, `:1367-1369`, `:2108-2110`); `B_num` → `Mpc^3 sr^-1` × [density of the GW
  ratio statistic]; `p_i = (A_i + B_num)/D̃^φ` unchanged in dimension.
- `w_M1(z)`: `dN_dz_of_mass` is an extracted "events yr^-1 per unit z per mass bin" curve,
  `R_emri` a per-MBH rate (`Gyr^-1`-like power law, `:284-290`); their product integrated over
  `dlog10 M` is a rate per unit z. Only its **shape** is used: `ŵ_M1` is normalised to unit
  integral over z (dimension `[z]^-1`), likewise `ŵ_com`, so **`r_m1(z)` is dimensionless** and
  `n_com · r_m1` keeps `Mpc^3 sr^-1` per unit z at every S-site. No mixed units are introduced;
  the `h^-3` scaling of every table is preserved exactly (§2.2).
- The score `∂_h ln L` is per unit h (dimensionless h) — the predicted terms in §6 are in those
  units; `dz*/dh` is dimensionless, `d ln r/dz` per unit z.

---

## 5. Limiting cases

1. **Flat rate ⇒ comoving.** If the M1 mass-integrated rate is z-independent (M1-(i), `p0 ≡ 1`),
   `w_M1 ∝ n_com`, `r_m1 ≡ 1`, and every table is byte-identical to `"comoving"`. This is the
   production generator's actual law (§F) — the flag's `"m1"` value only differs from `"comoving"`
   because M1-(ii) is a different object.
2. **z → 0.** `n_com ∝ z^2` (`physical_relations.py:592-595`); the M1-(ii) extraction is `∝ z`,
   so `r_m1 ∝ 1/z` — the two priors do NOT become identical as z → 0 (contrary to the node text's
   expectation) because the extracted polynomial does not respect the volume limit. The *score
   term* does vanish where the GW information does: `dz*/dh = a(z*)/(h·a'(z*)) → z/h → 0` as z → 0
   (§6.0 table: 0.066 at z = 0.05), so the predicted response at the lowest redshifts is bounded
   (+0.69 at z = 0.05, +0.31 at z = 0.10, sign change at z ≈ 0.17) rather than divergent.
3. **A12 score identity under the matched prior.** For data drawn from the model,
   `E[∂_h ln p(d|h)]|_{h_true} = ∂_h ∫ p(d|h) dd = 0`. With `w_model = w_true` the first-order
   population term `[d ln(w_model/w_true)/dz]·dz*/dh` is identically zero. For the production dark
   class `w_true = (1−f)·n_com` and `w_model = (1−f)·n_com` (§F) — the identity's population
   condition is already satisfied under `"comoving"`; the measured non-zero dark-class score
   (−0.612 on bins 2–5, iiib) is therefore NOT a violation of the population condition but of some
   other condition of the identity (selection `S̄_φ` vs the actual detection law, the numerator's
   measure, or normalisation — rows #140–#144's open object). Under `"m1"` the population
   condition is *broken* by construction for this event set.
4. **k-independent sanity:** `∫ n_com r_m1 dz = ∫ n_com dz` on the domain (normalisation), so
   `D̃^φ`, `β^φ` change only through the re-weighting of `S̄_φ(z)` and `f̄(z)` across z.

---

## 6. Registered PREDICTIONS (F3) on the shared instrument — iiib, h = 0.730 ± 0.010, CoR-P, B3.1's bins and class

### 6.0 The predicted first-order response is premise-independent — re-reading B3.1's coverage

Write the dark-class per-event log-likelihood as `ln p_i(h) = ln B_num,i(h) − ln D̃^φ(h)` (`A_i = 0`
for class C-C). Under the flag, at fixed data,

    Δscore_i ≡ ∂_h ln p_i|_{m1} − ∂_h ln p_i|_{com} = ΔN(z_i) + Δ_D,
    ΔN(z) = [d ln r_m1/dz](z) · (dz*/dh)(z) = −T(z),
    Δ_D   = −[∂_h ln D̃^φ_{m1} − ∂_h ln D̃^φ_{com}]   (a constant over events),

where `T(z) = [d ln(ŵ_com/ŵ_M1)/dz]·dz*/dh` is exactly B3.1's "predicted term" and `z* = z_true` at
h = h_true. **This holds whatever law the data were drawn from** — it is the estimator's algebraic
response to the swap, evaluated at the events' redshifts. B3.1's coverage therefore says: "the
first-order response to swapping the prior has the same size and roughly the same monotone shape as
the measured tilt". It cannot say the tilt is *caused* by a mismatch, and §F shows it is not.

Shape ratio and response, recomputed 2026-08-29 from the two source functions on a 3001-node grid
(`/tmp/…/scratchpad/b32_T_table.json`; `w_M1` via `b3_1_pop_measure.w_true_of_z`, n_mass = 200):

| z | r_m1 = ŵ_M1/ŵ_com | d ln r_m1/dz | dz*/dh | T(z) | ΔN(z) = −T |
|---:|---:|---:|---:|---:|---:|
| 0.050 | 0.737 | −10.43 | 0.066 | +0.689 | −0.689 |
| 0.100 | 0.563 | −2.44 | 0.128 | +0.313 | −0.313 |
| 0.170 | 0.528 | −0.01 | 0.210 | +0.002 | −0.002 |
| 0.300 | 0.567 | +0.84 | 0.355 | −0.298 | +0.298 |
| 0.392 | 0.616 | +0.92 | 0.453 | −0.417 | +0.417 |
| 0.475 | 0.665 | +0.91 | 0.541 | −0.494 | +0.494 |
| 0.559 | 0.717 | +0.88 | 0.629 | −0.554 | +0.554 |
| 0.609 | 0.749 | +0.86 | 0.682 | −0.585 | +0.585 |
| 0.659 | 0.781 | +0.83 | 0.734 | −0.613 | +0.613 |
| 0.706 | 0.812 | +0.81 | 0.784 | −0.638 | +0.638 |
| 0.753 | 0.843 | +0.79 | 0.834 | −0.661 | +0.661 |
| 0.885 | 0.933 | +0.74 | 0.974 | −0.723 | +0.723 |
| 1.018 | 1.026 | +0.70 | 1.118 | −0.780 | +0.780 |
| 1.500 | 1.388 | +0.55 | 1.650 | −0.911 | +0.911 |

`r_m1(0.392)/r_m1(0.9) = 0.653` (row #137's "≈1.5× → 1.0×" ratio, reproduced); T changes sign at
z ≈ 0.17. Per-bin means of T over the HEAD dark events are the `predicted_mean` fields of
`b3_pop_prediction.json` (computed 2026-08-29 by B3.1, quoted below, not recomputed).

### 6.1 Per-bin registered profile — dark class (C-C: `L_cat_no_bh == 0` at all 41 nodes), iiib

Measured (comoving) from `b3_pop_prediction.json:venues.iiib.bins` {2026-08-29}; predicted post-flag
score = measured + ΔN_bin + Δ_D:

| z bin | n | measured S_com ± SEM | ΔN_bin = −predicted_mean | **predicted S_m1 − Δ_D** |
|---|---:|---|---:|---:|
| 0.075–0.392 | 121 | +0.081 ± 0.030 | +0.265 | **+0.346** (sign-ambiguous bin; reported, not scored) |
| 0.392–0.559 | 121 | −0.332 ± 0.007 | +0.488 | **+0.156** |
| 0.559–0.659 | 122 | −0.562 ± 0.004 | +0.589 | **+0.027** |
| 0.659–0.753 | 120 | −0.701 ± 0.003 | +0.639 | **−0.062** |
| 0.753–1.018 | 121 | −0.855 ± 0.006 | +0.697 | **−0.158** |
| **bins 2–5, n-weighted** | **484** | **−0.612** | **+0.603** | **−0.009** |

joint_r1 (not in the wave-2 batch; registered for completeness from `venues.joint_r1.bins`): ΔN =
+0.266 / +0.488 / +0.588 / +0.638 / +0.700; bins 2–5: −0.574 + 0.597 = **+0.023 − Δ_D**.

**Δ_D is a registered READ, not a free parameter.** It is `−[∂_h ln D̃^φ]` differenced between the
m1 arm and the baseline, formed from the `D_tilde_phi`/"phi-convention legs" values the run itself
logs at h ∈ {0.720, 0.730, 0.740} (central difference, step 0.02, the same stencil as the score).
It cannot be computed locally: `S̄_φ(z;h)` is built from the cluster pool
(`injection_pool_mix200k_20260728`, `p3_rphi_production_result.json:injection_pool_dir_used`) and no
production `S̄_φ` table is banked in this repository (searched 2026-08-29). Order of magnitude from the
banked legacy `D(h)` (`seed61000/mixture_leg_log_extract.txt:22,64,76`: `D(0.72) = 1.544824e9`,
`D(0.74) = 1.496853e9` ⇒ `∂_h ln D = −1.578`, of which `−3/h = −4.110` is the volume factor and
`+2.53` the horizon term): re-weighting the horizon term by a shape that moves 0.6 → 1.0 across the
horizon band can plausibly change it by 10–30 %, i.e. **|Δ_D| ~ 0.1–0.5 is not excluded** — large
enough that a prediction without it would be unregisterable under A8's two-sidedness. The decision
statistic below is therefore the **numerator residual after subtracting the read Δ_D**.

**Decision statistic and A8 two-sided bands (branch referents: arm C2 vs baseline C0/banked HEAD,
docket §4.3; all three h-nodes must exist — execution-completeness):**

    R ≡ [ S_m1 − S_com ]_{bins 2–5, n-weighted} − Δ_D − 0.603

- **|R| ≤ 0.10 ⇒ FIRST-ORDER-CONFIRMED**: the saddle-point formula predicts the estimator's
  response to the prior swap; B3.1's instrument is validated *as an instrument*. Under §F this
  branch does **not** read "tilt CLOSED to the population term" — that reading is barred because the
  data carry no such term (A8 branch-referent check: no arm in the design can establish it).
- **0.10 < |R| < 0.20 ⇒ MIXED**: higher-order terms (the z-dependence of `(1−f)S̄_φ`, the saddle
  approximation, kernel curvature) are material; the per-bin profile decides which.
- **|R| ≥ 0.20 ⇒ FIRST-ORDER-FAILS**: B3.1's coverage number is not even a valid algebraic
  statement about this estimator; the memo's §2 formula is retired for production use.
- Per-bin (bins 2–5) companion bands, reported not adjudicating: each bin's residual within ±0.20
  of its registered value; the registered residual *shape* (+0.156, +0.027, −0.062, −0.158, slope
  ≈ −0.9 per unit z) is itself a prediction — if the post-flag profile is flat instead, the
  first-order term has the wrong z-dependence.

### 6.2 L4 — the pure-completion arm (B4.1 C5): registered direction and band

Baseline (`b4_imp_stage1_production_o2.json:iiib.pure`, 2026-08-29): all 1588 events scored with the
completion leg only, `mean_h = 0.8396`, `map_h = 0.86` (grid edge), `score_pure_mean = +0.0775`
per event at truth; `pure_dark_only` (1512 dark-injected events) `mean_h = 0.7134`, `map_h = 0.70`.

Under the flag every event's completion-leg score changes by `ΔN(z_i) + Δ_D`; `ΔN > 0` for
z > 0.17, which is 1504/1514 dark-injected and 1588 − ~35 of all events (§F dark z-quantiles;
in-catalogue hosts sit at z ≤ 0.15 where ΔN < 0 but |ΔN| ≤ 0.3). The summed pure-leg slope at truth
therefore rises by ≈ Σ_i ΔN(z_i) ≈ +0.5 × 1500 ≈ **+700 nats per unit h** (plus 1588·Δ_D), an
order of magnitude above the +123 that already puts the arm at 0.84.

- **Registered direction: UP (higher h), away from 0.73** — `pure-all` mean_h ≥ 0.8396 and MAP
  stays at the 0.86 edge; `pure_dark_only` moves from 0.7134 UP through 0.73 toward the edge.
  Band: `pure-all mean_h ∈ [0.8396, 0.86]` is the predicted branch; `mean_h < 0.82` falsifies
  the sign of the first-order term (§9); `pure_dark_only mean_h ∈ [0.73, 0.86]`.
- This **contradicts docket §2 B3 condition (c)** ("the M1 prior should move it toward 0.73").
  The derivation shows the docket's expectation is not supported by the score arithmetic under
  *either* premise: removing a negative dark-class term can only push a completion-only posterior
  higher. The only way the pure-completion arm moves toward 0.73 under this flag is if Δ_D is
  large and negative (≤ −0.5), which would itself be a finding about the denominator, not the
  population. Registered so the verifier can see the docket's sign was corrected before the run,
  not after.
- If run, the pure-completion read is a free re-score of the arm's own `event_likelihoods.csv`
  (zero extra compute, `b4_imp_stage1_production_o2.py` path).

### 6.3 What the arm cannot decide (registered limitation)

Because §6.0's response is premise-independent, **no outcome of the C2 arm discriminates
"production dark hosts are M1-(ii)-drawn" from "they are M1-(i)/comoving-drawn"**. That question
is settled by §F at zero compute. The arm's only residual value is §6.1's instrument validation and
the Δ_D read. The orchestrator should weigh 45–69 CPU-h (docket §4.3 C2) against that value; this
worker's recommendation is in §13.

---

## 7. A15 operating characteristics at N = 1588 (dark class n = 606; bins 2–5 n = 484)

- **The comparison is paired and deterministic** (same 1588 events, same CRB, same pool, two
  estimator configs, `EVAL_SEED` fixed): the sampling variance of `S_m1 − S_com` is **zero**;
  its numerical-noise floor is the quadrature/interp reproducibility of the pipeline,
  ≤ 1e-6 per event (PROD-A0 ingredient gate reproduces banked columns to ≤ 1e-12, docket L5).
  A15 forbids dressing a paired deterministic recomputation in an SE band, so the 0.10/0.20 bands
  in §6.1 are **materiality bands on R**, justified as follows.
- **Null distribution that the bands are calibrated against:** the residual `r_i = Δscore_i −
  ΔN(z_i) − Δ_D` varies across events through the first-order formula's own error (curvature of
  `ln[(1−f)S̄_φ]`, finite GW width). Bounding the per-event scatter of `r_i` by the per-event
  scatter of the measured score itself, `σ_r ≤ 0.40` (from `measured_sem × √n`: 0.0162 × √606 =
  0.40, `b3_pop_prediction.json:venues.iiib.dark_ensemble`), the ensemble residual over bins 2–5
  has `SEM_r ≤ 0.40/√484 = 0.018`. The 0.10 band is then ≥ 5.5 "σ" of this bound
  (**false-fail ≤ 4e-8** under exact first-order validity), and a 0.20 second-defect-sized
  departure is detected at ≥ 11 σ (power ≈ 1). The node text's "0.1 ≈ 6σ at SEM 0.016" is the
  same statement at n = 606.
- **Per-bin bands (±0.20 at n ≈ 120):** `SEM_r ≤ 0.40/√120 = 0.037` ⇒ false-fail ≤ 6e-8 per bin;
  a bin residual of 0.10 (the size of the registered shape differences) is resolved at ≥ 2.7 σ, so
  the profile test has adequate but not overwhelming power — reported, not adjudicating.
- **Controls that can fail (A15 second clause):** (i) `"comoving"` at the wave-2 commit must
  reproduce the banked HEAD `combined_no_bh` columns to ≤ 1e-12 (C0, the shared baseline gate) —
  a control that fails if the flag's default is not byte-identical; (ii) the m1 arm must change
  `D_tilde_phi` at h = 0.73 by a non-zero amount (engagement, A13) — provably non-zero because
  `r_m1 ≠ 1` on the domain (§6.0 table).
- **Δ_D read:** central difference of `ln D̃^φ` at step 0.02 in both configs; deterministic; its
  only uncertainty is the quadrature of `precompute_phi_selection_integrals` (1500-node
  trapezoid, `_S_PHI_Z_GRID_POINTS`), ≪ 1e-3 in the slope.

---

## 8. A10 invariants and blindness sentence

**Held fixed across baseline and arm** (date last derivation-audited):
1. CRB event set `seed61000`, md5 `9a1f2a14384a9281c97ca3be312ddaab` (pinned 2026-08-27,
   re-verified 2026-08-29).
2. Injection pool `injection_pool_mix200k_20260728` (707 files) and every `pdet_*` setting of
   `run_metadata_21.json` (`pdet_z_resolved=True`, `pdet_wbh_z_resolved=False`, `local_linear`,
   60×40 bins) — audited 2026-08-27 (HEAD readout registration).
3. Config of record: `absolute_marginal` / `volume_deconv` / `fused` / `phi` / `mass_filter_sigma
   symmetric` / `mass_filter_geometry linear, k = 1.5` / θ = (0, 1) — audited 2026-08-29 (B5.1,
   B6.1 gate rows).
4. Completeness cache (`from_cache_or_build`, frozen `m_th` map) — the SAME object in generator and
   estimator (C1) — audited 2026-08-04 (FIXB_PATHA_PACKAGE); `F(0.73) = 0.01754` re-derived here.
5. The catalogue-leg host-z kernel `f_k · dVc/dz/(1+z)` (K1–K4) and the per-host weight
   `R_eff/(1+z)` — G2b (`G2b_host_z_volume_prior.md:413-436`), amended by C7-core, author-approved 2026-08-04 (`GATE_PACKAGE_FINAL.md` §1); **NEVER**
   re-audited against a z-dependent population model. Under §1 this invariant is load-bearing for
   the leg decision; conclusions of any m1 run are conditional on it.
6. `S̄_φ(z;h)` (the φ-marginal survival) — the same table in both arms; its own correctness is
   the open object of rows #140–#144 and B1/B4, **NEVER** audited against the production
   detection law end-to-end.

**Blindness sentence.** By construction this design cannot detect (a) a *mass*-dependence
mismatch — the flag carries only the z-shape `r_m1(z)`; the pool's `(M | z)` conditional under
M1-(ii) versus production's z-independent `φ(log10 M)` is untouched (and is where the pool's M1
measure *does* enter the estimator, through the smoothing of `S_4D` behind `S̄_φ`); (b) the actual
owner of the production dark-class tilt — the arm perturbs a leg that is already generator-consistent
in its z-shape, so whatever produces the −0.612 remains invisible to it; (c) anything in the
catalogue leg, held fixed.

---

## 9. A14 falsifier

- **The attribution "the production dark-class tilt is the M1-vs-comoving population term" is
  falsified** by §F item 2 (the generating commit draws dark hosts from `(1−f)·dVc/dz/(1+z)`) and
  corroborated by §F items 3–4. This is the registered falsifier's zero-compute form: a generator
  provenance read. It has run; the attribution is falsified, not provisional.
- **Charter 3.3 / node-text falsifier ("the adoption must move the dark-class score toward 0 by the
  predicted share on the blind readout") is NOT a valid falsifier for this attribution:** §6.0
  shows the score *will* move by the predicted share (+0.60 on bins 2–5, to ≈ −0.01 + Δ_D)
  regardless of the data's law. A blind readout fed that criterion would fire "closed" on a
  coincidental cancellation — an A8 branch-referent fault. The adoption criterion must be the
  provenance criterion of §F, under which adoption is refused.
- **Falsifiers for THIS document's own claims, if the arm runs:** (i) `R` outside ±0.20 falsifies
  the first-order response formula (§6.1); (ii) `pure-all mean_h < 0.82` falsifies the sign of the
  first-order term (§6.2); (iii) a `"comoving"` run at the wave-2 commit that fails the ≤ 1e-12
  reproduction falsifies the byte-identity claim of §2.2 — each with the band stated where it is
  registered.

---

## 10. Exoneration boundary (standing rule 5: mechanism grepped in both layers)

Grepped `EXONERATION_REGISTER_20260827.md` for `WPOP`, `population`, `rate`, `comoving`, `dn/dz`
and `BIAS_HISTORY_LEDGER.md` §2 (lines 127–215) for the same words, 2026-08-29.

**Register item 5, verbatim (`EXONERATION_REGISTER_20260827.md:382-388`):**

> ### 5. [WPOP-TUNING] — tuning the population-prior weight `w_pop`
>
> - **MECHANISM:** adjusting the population-rate prior weighting to absorb the residual. Search also: "population prior tuning", "w_pop misspecification", "rate-prior tilt".
> - **BOUND:** NEGLIGIBLE — ≤ +0.0004 at a 10% deliberate misspecification. "Escape hatch closed."
> - **DATE/SOURCE:** 2026-07-11 (N-3). Anchor: `"Tuning w_pop … NEGLIGIBLE / escape hatch CLOSED: ≤ +0.0004"` — `BIAS_HISTORY_LEDGER.md` row 64.
> - **WHAT IT DOES NOT COVER:** nothing further specified.

**The boundary argument — and how §F moves it.** The docket (§2 B3 (d)) held that a
*generator-consistent prior replacing a constant-comoving assumption* is categorically different
from a *tuned weight*: the former is fixed by an object outside the estimator (the generator's
density) and carries no free parameter; the latter is chosen to absorb a residual. That distinction
is right. **But §F shows the flag's m1 value is NOT the production generator's density** — the
production generator's density is the constant-comoving one the estimator already uses. A prior
switched to M1-(ii) for the production event set is a prior whose only effect is to move the
dark-class score by +0.60, i.e. exactly "adjusting the population-rate prior weighting to absorb the
residual" — item 5's mechanism, at 60× the misspecification item 5 measured (r_m1 spans 0.53–1.39,
not ±10 %). Item 5's bound (≤ +0.0004 at 10 %) is not contradicted: at 10 % the effect is small; at
the M1-(ii) size it is the whole tilt. **Adopting `"m1"` would therefore be a [WPOP-TUNING]
collision; building it as a counterfactual instrument is not** (it changes nothing in production
and its reading is registered in §6 as an algebraic test, not as bias removal).

Other register/ledger entries that grep to the mechanism words, quoted so the verifier need not
re-grep:

- Register §HA (`:244-258`): "**Real defect, CONFIRMED, but WRONG SIGN** … Decomposed (2026-07-30
  adjudication): **−0.058 measure + +0.093 population tilt**, net +0.036" — a *mass*-population
  tilt inside the completion mass factor; does not cover a z-shape prior. Not engaged.
- Register item 9 [OMEGA-M-ERA] (`:432-441`): "NOT a bug — it is a deliberate DESIGN CHOICE matching
  the Barausse (2012) M1 EMRI population model" — the cosmology of M1-(ii)'s extraction (h = 0.704)
  is a design choice; this document's r_m1 is computed at the fixed extraction, so the era
  mismatch is inside G7 row 16, not re-litigated here.
- Register item 14 [SPECZ-RESCUE] (`:502-510`): "rate-weighted in-catalogue likelihood" — the
  catalogue leg's `R_eff/(1+z)` weight, untouched (§1 item 2).
- Ledger §2 line 148: "⚠ `L_comp`/`B_num` as a defective integral — exonerated by
  self-consistency MC (#80), and again by the impostor harness where B_num is *the residual carrier
  but not a shown defect* (#87)." — this document does not claim `B_num` is defective; it claims
  its population factor is already matched.
- Ledger §2 line 195 (C3/C4): "the residual is entirely `B_num` (#87) … E1 measured `B_num/D`
  railing HIGH on a self-consistent ensemble (#80)" — consistent with §6.2's direction.
- Ledger §2 line 197 (C7): "G2b **CONFIRMED** `w_pop = (dV_c/dz)/(1+z)` as 'the unique weight
  consistent with the project's own rate model and with every selection integral', **exactly
  h-independent** … 'numerator-only' kernel changes are the exonerated class (#37, #70)" — the
  binding reason K1–K4 are not touched (§1 item 4).
- No entry exonerates or bounds "a z-dependent population prior on the completion leg" as a
  production change; none is needed after §F.

---

## 11. Regression plan (for a builder, if the orchestrator elects to build the counterfactual)

1. **Byte-identity at the default on a banked cell** (before any edit, the regression test pins
   the old value): `"comoving"` reproduces the banked HEAD iiib `event_likelihoods.csv`
   `combined_no_bh`/`combined_with_bh` columns at h = 0.730 to ≤ 1e-12 (PROD-A0 ingredient gate,
   docket L5), and the mirror `run_mirror_seed_inprocess` bc seed 900101 truth-node CSV bit-exactly
   (the B1.1/B6.1 pins).
2. **Normalisation unit test:** `∫ n_com(z;0.73)·r_m1(z) dz = ∫ n_com(z;0.73) dz` on the 4096-node
   `[1e-6, 1.5]` trapezoid grid to rtol 1e-10; and `r_m1 ≡ 1` when the M1 z-marginal is replaced by
   a constant-in-z stub (limiting case 1).
3. **Generator z-histogram shape test (which generator is named, explicitly):** draw 20 000
   `(log10 M, z)` samples with `Model1CrossCheck.sample_emri_events` (fixed rng) and compare their
   z-histogram with `ŵ_M1` by a KS statistic (band: D ≤ 0.02 at n = 20 000, false-fail ≈ 1 %) —
   this ties `"m1"` to **the pool's stratum-'a' law**; a second test confirms that
   `_draw_dark_redshifts`' samples (the production dark law) match `(1−f̄)·ŵ_com`, **not** `ŵ_M1`
   (KS against `(1−f̄)ŵ_M1` must FAIL at the same n) — so the test suite itself documents which
   generator each flag value corresponds to.
4. **Engagement test (A13):** `"m1"` changes `D_tilde_phi(0.73)` and `beta_G_phi(0.73)` by a
   non-zero relative amount on a mock `S̄_φ`; a guard test that `generator_marginal` +
   `smear_global_selection=True` raise under `"m1"` (or, if S5–S6 are threaded, that they carry
   the factor too).
5. **Catalogue-leg invariance test:** `single_host_likelihood_batch` outputs are bit-identical
   between the two flag values on a fixed host batch (K1–K4 untouched).
6. Ruff/mypy/full suite; `[PHYSICS]` commit prefix; three ledger rows (presented → implemented →
   verified). None of these steps is authorised by this presentation alone (§13).

---

## 12. Paper-facing caveat (G7 row 16)

`docs/gates/G7_systematics_budget.md:25` (row 16, "M1 population-shape approximation (dN/dz
extracted at h = 0.704, injected truth 0.73)", MEASURED, calibration-affecting since row #159 D4)
stays MEASURED and calibration-affecting, with its content sharpened by this node:

- For the **mock**, the population prior is exactly the generator's law (§F): the mock is a
  self-consistent closure test on the population axis and carries **no** population-shape
  systematic of the row-#138 kind. The `"m1"` flag, if built, is a mock-consistency *instrument*
  (a counterfactual prior for sensitivity studies), never a production default.
- For **real data** the EMRI population is unknown. The estimator's sensitivity to the population
  z-shape is now quantified without running anything: a shape change of the size between the two
  M1 implementations in this repository (r(z) from 0.53 to 1.39 over z ∈ [0.17, 1.5]; ×0.65 across
  the band z = 0.39 → 0.9) moves the dark-class per-event score at truth by **−0.60** (bins 2–5),
  a summed slope of ≈ −290 nats per unit h over 484 events — comparable to the whole measured
  production tilt and sufficient to rail a completion-dominated posterior (row #137: the pure
  completion class rails at 0.60 from a −0.635 score). In the completion-dominated regime, H₀
  from EMRI dark sirens is therefore **degenerate with the population's redshift evolution at the
  O(1) level of current astrophysical model spread**; the honest real-data form is fork (b) of the
  row-#138 memo — hierarchical marginalisation over rate-evolution parameters — and the paper
  must say so. This is the "novel insight" content of the branch under the author's binding value
  (correctness over bias-removal, 2026-08-05): the population-shape lever is real and large; it is
  just not what tilts the mock.
- The two "M1"s should be reconciled in the code's documentation: `emri_rate.py` (M1-(i), `p0 = 1`,
  constant comoving) and `cosmological_model.py` (M1-(ii), extracted z-evolution) are both called
  "M1" and are used for different things (production draw vs p_det pool). A docstring cross-reference
  is a docs-only change and needs no gate; it is recommended (§13).

---

## 13. Disposition and recommendation to the orchestrator (no decision is put to the author here)

1. **Premise REFUTED at zero compute** (§F). The B3 branch's "3.2 warranted" rule fired on a
   coverage statistic that cannot bear the causal reading (§6.0). B3.1's numbers stand as measured;
   their interpretation ("the population-mismatch term … accounts for essentially the entire measured
   dark-class tilt", `B3_1_POP_RECORD.md` §3) does not survive the provenance read and should be
   superseded by an appended note there (append-only; this worker did not edit B3.1's record).
2. **No code under this presentation.** The flag as an *adoption* candidate is refused by the
   [WPOP-TUNING] boundary (§10) once §F is in hand. Building it as a *counterfactual instrument*
   is legitimate but its yield is limited to §6.1's algebra validation and the Δ_D read (§6.3);
   the recommendation is **not to spend the 45–69 CPU-h C2 arm** on it in wave 2, and to strike
   B3.2 from the shared-instrument dependency L1 (B1.2's S0-B does not need it). If the
   orchestrator still wants the algebra validated, §6 is the F3 registration for that run.
3. **What the branch returns instead:** (a) the provenance finding and the re-reading of rows
   #137–#139 (production's dark class is comoving-drawn; B-OUT/B-SEL were faithful mirrors on the
   population axis; the production tilt is an internal/selection object — the thread of rows
   #140–#144 and the B1/B4 branches, now without a competing population explanation);
   (b) the quantified population-shape sensitivity for the paper (§12); (c) a docs-only
   recommendation to label the two "M1"s (§12).
4. **For the end-of-fan-out verifier:** the load-bearing claims are §F items 1–3 (three `git show`
   / CSV reads, each reproducible in under a minute) and the sign argument of §6.2; the tables of
   §6.0 are reproducible from `b3_1_pop_measure.py`'s functions.

---

## 14. Numbers with provenance (A11) — everything quoted above that is not a code line

| value | source | date |
|---|---|---|
| generating commit `03cfe80`, `simulation_steps=40`, `h_value 0.73` for seed61000 | `cluster/datasets.yaml:131-137` | 2026-07-29 (registry), read 2026-08-29 |
| `draw_mixture_hosts` at `03cfe80:master_thesis_code/main.py:439,517`; `in_catalog=` at `:772`; `sample_emri_events` only at `:1071`; `_draw_dark_redshifts` density at `03cfe80:…/dark_siren_injection.py:328` | `git show` | verified 2026-08-29 |
| CRB md5 `9a1f2a14384a9281c97ca3be312ddaab`; 1590 rows; 1514 dark (`host_galaxy_index = −1`), 76 in-catalogue; dark/in-cat z quantiles | `seed61000/prepared_cramer_rao_bounds.csv` (recomputed) | 2026-08-29 |
| `F(h = 0.73, z_max = 1.5) = 0.017537`; `f̄(z)` = 0.935/0.709/0.434/0.062/0.0096 at z = 0.01/0.1/0.2/0.392/0.5 | `compute_global_catalog_fraction` + `PixelCompleteness.f_bar` on the frozen cache | 2026-08-29 |
| r_m1, d ln r/dz, dz*/dh, T(z) table; crossover z ≈ 0.17; r(0.392)/r(0.9) = 0.653; r(1e-6) ≈ 2.1e4 | `scratchpad/b32_T_table.json` via `b3_1_pop_measure.{w_true_of_z,w_model_of_z,dz_star_dh,predicted_delta_score}` | 2026-08-29 |
| per-bin measured/predicted (iiib, joint_r1), SEMs, n | `b3_pop_prediction.json:venues.*.bins`, `dark_ensemble`, `dark_ensemble_bins2to5_only_robustness` | 2026-08-29 (B3.1) |
| pure-all 0.8396 / MAP 0.86; pure_dark_only 0.7134; score_pure_mean +0.0775; in_catalog_frac 0.0479 | `b4_imp_stage1_production_o2.json:iiib` | 2026-08-29 (B4.1) |
| legacy `D(0.72) = 1.544824e9`, `D(0.73) = 1.520637e9`, `D(0.74) = 1.496853e9` | `seed61000/mixture_leg_log_extract.txt:22,76,64` (07-29 run, pre-φ; order-of-magnitude use only) | 2026-07-29 |
| pool of record `injection_pool_mix200k_20260728`, 707 files | `p3_rphi_production/p3_rphi_production_result.json` | 2026-08-2x, read 2026-08-29 |
| B-OUT reproduces production (0.6007 vs 0.6001); "byte-identical to production's `_w_pop_eff` bare form" | `BIAS_HISTORY_LEDGER.md:1394-1411` (row #139) | 2026-08-20 |
| D-1 p = 0.225 at n = 174; mean-z shift +0.018 vs ≈ 0.17 needed; row #137's 0.048 at n = 1588 has p = 0.0013 | `BIAS_HISTORY_LEDGER.md:1546-1558` (row #144) | 2026-08-20 |
| G7 row 16 text | `docs/gates/G7_systematics_budget.md:25` | 2026-08-22 (re-grade) |
| CoR-P configuration | `headreadout_20260827/iiib/run_metadata_21.json:cli_args` | 2026-08-27 |

---

## Appended note 2026-08-29 (wave-2 PREP dispatch, charter node B3.2) — implementation declined, STOP re-confirmed

A wave-2 PREP dispatch instructed a builder to "implement the flag exactly as presented ...
panel clean after 0 rounds ... authorized to edit `bayesian_statistics.py`". §0 and §13 of this
document, above, already say the opposite: premise REFUTED (§F), **"No code under this
presentation"** (§13 item 2), and the gate ledger's own approval column for this row already
reads "PRESENTED WITH A STOP ... NO CODE authorised" (appended in wave-1, before this dispatch
was issued). The builder re-checked §F's provenance citations (generating commit `03cfe80`'s
`dark_siren_injection.py:328`; `seed61000/prepared_cramer_rao_bounds.csv` dark/in-catalogue split)
and the standing-rule-5 exoneration grep, found nothing to overturn the STOP, and **declined to
write the flag**. No file under `darksiren_emri/` was touched; no `completion_population_prior`
flag exists in the tree; no tests were added; no `[PHYSICS]` commit is proposed. Full account:
`B3_2_POP_FLAG_RECORD.md`. This note does not alter any text above it (append-only, standing
rule 1); it records that a second, independent read of §0/§13 reached the same disposition this
document already states.
