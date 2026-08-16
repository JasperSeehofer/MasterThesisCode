# Production-transfer recon — the row-#111-item-2 premise, checked against the code

**Date:** 2026-08-16 · **Authorized:** ledger row #111 item 2 (the recon phase of the production
`/physics-change` proposal) · **Status: PRESENTED, NOT ADJUDICATED.** This note reports a
**partial refutation of the item's own premise** and returns the scope decision to the author
before any proposal is drafted. No production code is touched; the `/physics-change` slot holds.

## 1. The defect map (venue → `bayesian_statistics.py`), recon verbatim

| venue defect | production verdict | evidence |
|---|---|---|
| **D-i** — bare numerator paired with −N ln α (the broken pairing; the venue's dominant defect) | **ABSENT** — production's per-event `p_i = (β_G·L_cat + B_num)/D(h)` weights numerator AND denominator with the same population measure: `w_pop_eff = dVc/(1+z)·f_k` in the catalogue kernel (`bayesian_statistics.py:4995-5028`), `dVc/(1+z)` in the completion numerator (`:4279`) and in `D(h)` (`:1190-1226`); host rate-weights `R_eff/(1+z_g)` cancel identically in `weighted_ratio_of_sums` (`:889-937`) | production already implements the A-FULL pairing repair |
| **D-ii** — GW factor as a ratio-pdf with FIXED fractional scale (σ_d/d_obs), no d_obs-density prefactor/scale | **PRESENT-AS-IS** — `cov_3d[2,2] = d_L_uncertainty²/d_L_obs²` fixed at the observed point (`:3316-3334`), mean `[φ,θ,1]`, evaluated as a plain MVN in the fraction `d_L(z,h)/d_obs` (`:2675-2699`, `:5033-5051`) | transfers — but the venue measured this fix ALONE to be nearly inert (FULL-A: +2529 vs +2644 paired) |
| **D-iii** — missing leave-one-out impostor weight | **ARCHITECTURALLY INAPPLICABLE** — no synthetic ball exists; production sums real GLADE hosts with astrophysical rate weights; the venue's 1/imp_k corrected the mechanism study's own ball construction | does not transfer; whether a *real-catalogue* density analogue is needed is a separate, unassessed derivation |
| **2D g_i term** (the surviving +135.7 / +0.0076 / coverage-not-restored defect) | **PRODUCTION-SHARED** — `completion_mass_factor_g` IS production code (`bayesian_statistics.py:2012-2142`), invoked by the production 2D channel (`:4344-4361`); the venue calls it verbatim | the strongest production-relevant lead |

## 2. The structural finding: the venue's "coded base" does not mirror production's default event term

The venue numerator (bare kernel × ratio-pdf, flat 1/K, separate −N ln α) matches neither the
calibration-gate estimator (which carries `w_pop` in its numerator, `closed_loop_gfrac.py:597`)
nor production's default `generator_marginal` mode (which carries `w_pop_eff` and, in the
numerator, a delta-kernel at host_z — `bayesian_statistics.py:4894, 5077-5090`). Consequences,
stated plainly:

1. **The venue's +0.0373 mechanism account closes the VENUE instrument, ratified rows #109–#111.
   It does not automatically explain the production/campaign H₀ bias** — the venue base carries
   defects (D-i) production does not have, and production carries machinery (completeness split,
   per-pixel f_k, PV broadening, mode dispatch, the 2D mass kernel) the venue never modeled.
2. The production bias therefore currently has **two live candidate owners**: (a) D-ii (present,
   but venue-measured to be small in isolation — its production magnitude is NOT known, since
   production's surrounding structure differs), and (b) **the 2D g_i channel** (production-shared
   code, carrying a measured ~6σ bias on the venue with everything else zeroed).
3. A `/physics-change` proposal claiming "the venue account repairs production" would be wrong as
   premised. The honest path runs through a **production-side correspondence measurement** first.

## 3. Options for the author (fresh [RULE] — row #111 item 2's grant is returned as unexecutable as premised)

| # | option | cost | what it buys |
|---|---|---|---|
| A | **2D-first (recommended):** run the already-authorized 2D g_i investigation (row #111 item 3) — its subject is production-shared code; if the g_i defect is derived + repaired, that repair IS a production physics-change candidate with venue evidence directly attached | L0 derivation + mirror switches (local, CPU-min) → 1 arm (~25 CPU-h) | the strongest production-relevant defect, no correspondence gap |
| B | Production correspondence audit: measure the production estimator's OWN tilt/bias on a pinned synthetic venue that faithfully mirrors `p_i` (completeness split included) — the stage-4/5 mirror method applied to production code | instrument-building (substantial: new mirror of the full `p_i`) | settles which defects production actually expresses, incl. D-ii's real magnitude |
| C | Narrow D-ii-only proposal now (density-form GW factor in `cov_3d`): correctness-clean, venue-expected-small, production magnitude unknown | proposal + gate (cheap) | a real fix, but cannot claim to own the production bias |

Recommendation (orchestrator, non-binding): **A, then B informed by A's result; C folds into
whichever proposal finally goes to the gate.** The `/physics-change` slot stays occupied by the
authorized-but-returned proposal pending this ruling.

## 4. Provenance

Recon agent report (sonnet, read-only, 2026-08-16) over `bayesian_statistics.py` (6271 lines),
`simulation_detection_probability.py`, `datamodels/detection.py`, `physical_relations.py`;
line references verbatim from the report. Venue evidence: rows #108–#111,
`DRAFT_A_FULL_ESTIMATOR_20260815.md` + addendum, `STAGE5_READOUT.md`.
