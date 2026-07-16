# CONVENTIONS-MANIFEST — MasterThesisCode (SKELETON)

> **STATUS: SKELETON — NOT the full manifest.** This is the 4-incident-derived seed
> (+ HOST_DRAW_Z_MAX row + 2 verifiable bonus rows) mandated by
> [[orbiter-upgrade-design]] Part 4, C.6, as the standing floor for the sim/eval
> convention-consistency task-area. It enumerates only the convention-bearing
> quantities that have *already caused a dated incident* or are *flagged live*.
> **The full manifest — every convention-bearing quantity across the sim↔eval
> boundary — is domain archaeology estimated at 2–3 days of MTC/GPD-session
> context and is a separate named task (owner: Jasper / MTC sessions).** A class
> absent from this table is invisible to the tracer that consumes it; that
> incompleteness is declared in every tracer verdict.

- **Schema**: `conventions-manifest/skeleton-v0`
- **Seeded**: 2026-07-12 (advisory tracer run, pre-Phase-2-submission)
- **Boundary modelled**: INJECTION (sim: `main.injection_campaign`, `dark_siren_injection`, FEW/CRB) → STORAGE (injection CSV, GLADE+ reduced catalogue, CRB CSV) → P_DET GRID (`simulation_detection_probability`) → INFERENCE (`bayesian_statistics`, `posterior_combination`)
- **Convention on how to read the table**: each cell records the convention *as the code actually implements it at that stage*, with a `file:line` anchor where read from source, or `UNKNOWN — needs domain confirmation` where it could not be verified by reading code in one pass.
- **Paired-test column**: the invariant test that *should* guard this row per the C.6 floor (astropy round-trip · "a fix must produce different values" · every-output invariant · schema/provenance gate). `NONE FOUND` means no such guard was located — a manifest-upkeep action, not necessarily a bug.

---

## Table 1 — Convention-bearing quantities (incident-seeded)

| # | Quantity | Incident origin | AT INJECTION | AT STORAGE | AT P_DET GRID | AT INFERENCE | Paired invariant test | Verified? |
|---|----------|-----------------|--------------|------------|---------------|--------------|-----------------------|-----------|
| M1 | Sky-angle frame `qS`/`phiS` (θ,φ) | Coordinate-frame bug 2026-04-21 (0.0% apparent bias / 6 milestones) | ecliptic `BarycentricTrueEcliptic(J2000)`; host angles → `parameter_space.qS/phiS`; `ResponseWrapper(is_ecliptic_latitude=False)` (`waveform_generator.py:64`) | CRB CSV `qS/phiS` ecliptic rad; catalogue on disk is **equatorial ICRS deg** (raw cols 8/9), rotated **in place** to ecliptic at load (`handler._rotate_equatorial_to_ecliptic`, COORD-03/Phase 36, `handler.py:251`) | (sky enters via equal-\|sin β\| ecliptic-latitude bands; `bayesian_statistics.py:339`) | ecliptic; `Detection.phi/theta`; `get_possible_hosts_from_ball_tree` ecliptic (`bayesian_statistics.py:1175`) | astropy `SkyCoord.transform_to` round-trip at ingestion (COORD-03); FRAME-AUDIT.md 4/4 claims CONFIRMED | **YES — CONSISTENT** |
| M2 | BH mass frame `M` (source `M` vs redshifted `M_z=M(1+z)`) | Redshifted-mass bug 2026-06-20 (passed 568 tests, W-CONF-13) | **`M_z = M·(1+z)` lifted once at injection** (`main.py:899`); FEW sees `M_z`; source-frame `sample.M` NOT stored | injection CSV `"M"` column = **`M_z`** observer-frame (`main.py:980-983`); `Detection.M` documented `M_z` (`detection.py:60,85`); catalogue `host.M` = **source-frame** `M_g` (`handler.py:105`, `_rate_weight` note) | grid mass axis = **observer-frame `M_z`** (`_M_arr = pooled_df["M"]` = injection `M_z`; `simulation_detection_probability.py:139,272`) | numerator rate-weight uses source-frame `host.M` (matches draw); selection query **lifts** `M_z_g = M_g·(1+z_g)` to match grid axis (`bayesian_statistics.py:768`) | "a fix must produce different values" (M_z ≠ M); every-output invariant across CRB CSV **and** injection CSV (W-PRE-12 lesson); `test_parameter_space_h` guards CRB path | **YES — CONSISTENT (post-Design-B + H3 fix)** |
| M3 | In-catalogue likelihood `L_cat` form | `L_cat` mean-of-ratios bug, commit `816f904` | n/a (generative) | n/a | n/a | **ratio-of-sums** `(Σ_g w·N_g)/(Σ_g w·D_g)` — Gray (2020) Eq. A.9/A.10; `weighted_ratio_of_sums` (`bayesian_statistics.py:212-260`); constant-weight limit = plain ratio of sums | equivalence test that the ratio-of-sums (not mean-of-ratios) is canonical; `--catalog_only` ablation sign-test | **YES — CONSISTENT (post-816f904)** |
| M4 | `p_det` placement in the per-event ratio | p_det-in-numerator / incomplete-fix, commit `341ca62`, W-PRE-12 | detection = deterministic SNR≥threshold at injection | injection CSV stores `SNR`,`d_L` (raw, ungated; `PRE_SCREEN_SNR_FACTOR=0.0`, `constants.py:63`) | p_det = detection-horizon **survival** `P(d_hor≥d_L)`, `d_hor=SNR·d_L/thr` (`simulation_detection_probability.py:334`) | `p_det` appears **only in denominator** `D(h)=β_G+β_Ḡ` (`precompute_completion_denominator:389`); numerator `single_host_likelihood` has **no** `p_det`; `p_i=(β_G·L_cat + B_num)/D(h)` | integrand re-derivation invariant ("p_det in denominator only"); `--catalog_only` ablation | **YES — CONSISTENT (post-341ca62)** |
| M5 | Host-draw population depth `HOST_DRAW_Z_MAX` | **LIVE item** flagged 2026-07-02 (`CAMPAIGN-PREP-PHASE2.md`): "`0.5` horizon-stale" | `z_cut = HOST_DRAW_Z_MAX` (`main.py:825`); injections drawn to this depth | `constants.HOST_DRAW_Z_MAX = 1.5`; `GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT = 1.55`; injection CSV carries `z_cut` provenance column | `expected_z_max=HOST_DRAW_Z_MAX` passed at construction; **hard `raise ValueError`** on shallow (`pool_z_max < 0.9·1.5`) or mixed-`z_cut` pool (`simulation_detection_probability.py:290-322`) | `cosmological_model.max_redshift = 1.5` asserts `HOST_DRAW_Z_MAX ≤ max_redshift` (`cosmological_model.py:189`); D(h) integrals capped at `max_redshift` (`f29a5e7`, #30) | provenance/schema gate: `z_cut` uniqueness + `code_rev` check + shallow-pool `ValueError` | **RESOLVED → 1.5 (fix #20, `b52ff8d`, 2026-07-03); CONSISTENT if campaign pool regenerated (hard-gated)** |

## Table 2 — Verifiable bonus rows (read from code this pass; not incident-seeded)

| # | Quantity | AT INJECTION | AT STORAGE | AT INFERENCE | Verified? |
|---|----------|--------------|------------|--------------|-----------|
| B1 | Redshift frame (heliocentric vs CMB vs cosmological) | population z is cosmological (synthetic, frame-neutral) | catalogue **`z_cmb`** (GLADE+ col 28, PV-corrected; migrated from `z_helio` col 27) fed to `d_L(z,h)` & `M_z=M(1+z)` (`handler.py:153-158`) | residual host peculiar velocity marginalized into host-z kernel `σ_z_pv=(1+z)·200km/s/c` (issue #16, `constants.py:71-83`) | **CONSISTENT in-code; RECENT migration — see WATCH (verify campaign catalogue is the z_cmb rebuild)** |
| B2 | SNR detection threshold | `SNR_THRESHOLD=20` (`main.py:1308,1409`) | injection CSV `SNR` ungated | horizon `d_hor=SNR·d_L/20` (`snr_threshold=SNR_THRESHOLD`, `posterior_combination.py:583`); CRB filter `SNR≥20` (`bayesian_statistics.py:1027`) | **CONSISTENT (uniform 20)** |
| B3 | Distance unit `d_L` | Gpc | injection CSV Gpc; CRB CSV Gpc | `dist()`/`dist_vectorized()` return Gpc (`physical_relations.py:141,235`); `d_hor` Gpc | **CONSISTENT (Gpc uniform)** |

---

## Declared incompleteness (mandatory)

This skeleton covers **8 rows** across a boundary that certainly carries more
convention-bearing quantities. Known gaps NOT modelled here (candidates for the
full-manifest archaeology task):

- Fisher/CRB covariance conventions (which parameters are `log`-scaled; units of
  `delta_*_delta_*` covariance entries; correlation sign conventions).
- Prior/population weight conventions (`w_pop ∝ dV_c/dz/(1+z)`; the Eddington-shift
  `eddington_shifted_host_mass`; the volume-deconvolution `normalization_mode`).
- Completeness `f(z)`/`m_th` HEALPix conventions (magnitude system, NSIDE, apparent
  vs absolute threshold).
- Photo-z vs spec-z error-model conventions (`σ_z` floors, the σ_z/z shallow-venue
  regime flagged in SCV 2026-07-11).
- `pp_coverage` validation-harness population ceiling (`Z_MAX_POP=0.95`) vs the
  production campaign depth (`1.5`) — see tracer verdict Q7 (UNKNOWN).

**An `UNKNOWN` or a flagged row in this manifest is a valid, valuable output. A
false "all covered" is the exact W-CONF-13 failure mode this artifact exists to
prevent.**

## Upkeep contract (C.6 floor)

Per the layered-ownership recommendation: any `/physics-change` (or equivalent)
edit touching a convention-bearing quantity above **updates its manifest row and
its paired invariant test in the same change**. Manifest staleness is the tracer's
input — an unmaintained manifest produces false assurance.
