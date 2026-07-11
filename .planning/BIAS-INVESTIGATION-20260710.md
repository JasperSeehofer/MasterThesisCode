# Systematic-bias investigation — plan of record (2026-07-10)

**Trigger:** user directive after the seed1000 rail diagnosis (issues #29/#30): (1) implement the
zero-host pure-completion fallback; (2) decide robust-at-any-horizon vs explicit z-truncation;
(3) deep-dive why a systematic bias persists at all, on strictly consistent data per
DATA_INVENTORY.md.

## 0. Ground truth about "proven working" (evidence ledger, verified 2026-07-10)

- The closure PASS the project remembers (**G_H3b, 2026-05-06: 1D 0.7309 / 2D 0.7307, z≈0.2σ**)
  ran on phase46-merged data — **RETIRED** by the 2026-06-20 mass-convention + L_cat merge
  (`af6014d`). It is no longer evidence about the current pipeline.
- **No closure PASS exists on current-tier data.** The Phase-1 gate (G1–G11) fixed estimator
  defects and the commission de-rail restored interior MAPs on the frozen seed600 subsample
  (volume_deconv MAP 0.73, mean 0.7398, 494 events, 7-pt grid), but the pre-registered
  adjudication (CAMPAIGN-PREP-PHASE2.md §4b: 4 seeds @0.73 |mean MAP − 0.73| < 2·SEM + closures
  0.67/0.77 in own 68% + per-seed pp_coverage cov68 ± 0.10) is **NOT YET EVALUABLE** — those
  seeds are the blocked campaign.
- Current-tier measurements that exist:
  - **seed600 frozen shallow venue** (3,342 events, 17-pt grid, PV-test `run_live`, commit
    `562918ef`): 1D MAP 0.745 / mean 0.7432 (σ_boot 0.0052) → **+0.013 (~+2.6σ)**;
    2D MAP 0.785 / mean 0.787 (PV-insensitive; pre-dates the Eddington −0.020 2D shift? verify).
  - **seed1000 deep campaign**: railed LOW at h=0.60 — mechanisms #29 (58% zero-host drop) +
    #30 (effective catalogue z≲0.3 under the M_BH prune).
  - **seed400 + shallow pool** (07-10 perf confirm): rails HIGH — **retired regression venue,
    NOT bias evidence** (pre-massfix source-frame CRBs).

## 1. NEW consistency finding: seed600 venue Ω_m era mismatch

`run_20260628_seed600` CRBs were **simulated at Ω_m = 0.25** (constants at 2026-06-28); all
post-G11 evaluations (de-rail matrix, PV test, any current re-eval) infer at **Ω_m = 0.2726**
(`bdf5339`, 2026-07-02). Direction: h_inf = h_true·I(z;0.2726)/I(z;0.25) < h_true → the mismatch
biases the frozen venue LOW by ≈0.3–0.8% (z-graded), i.e. the venue's underlying positive
residual is slightly LARGER than the measured +0.013. Consequences:
- seed600 is an **A/B-only venue** (code-era comparisons on identical data); its absolute
  residual carries a quantifiable Ω_m-era term that must be corrected or bounded when quoted.
- The **only Ω_m-consistent closure venues are the Phase-2 campaign seeds** (generated at
  0.2726, depth 1.5) — blocked on #29/#30 + cluster return.
- ACTION: register this in DATA_INVENTORY (seed600 entry) — done below in §5.

## 2. Known quantified suspect ledger (post-gate; sign = effect on inferred h)

| Suspect | Channel | Magnitude | Status |
|---|---|---|---|
| Host-z bare kernel (Eddington-in-z) | 1D+2D | −2.4% @σ_z=0.035 | FIXED (volume_deconv default `235b783`) |
| Completion 1/(4π) sky marginal | both | rail ↑0.86 | FIXED `cb16142` |
| sin θ Jacobian | both | ~+15%/event weight | FIXED `4a259b7` |
| Eddington-in-M (G7 row 9) | 2D only | **−0.020 mean** | implemented `4d780f0` — verified in the PV-test numbers ([L4]) |
| with-BH-mass MC denominator D_g defect | 2D only | **−0.032 venue mean measured** (0.787→0.7546 on seed600 A/B) | FIXED `713fbd1` (perf branch, PR #31) — explained 57% of the +0.057 2D residual; remaining 2D residual +0.025 |
| PV value-correction | 1D | −0.014 worst-case (seed600) | applied in z_cmb catalogue; marginalized σ_v=200 `8568d9f`; #16 CLOSED |
| Ω_m era mismatch (seed600 venue only) | both | **−0.08% measured** (Δh̄ = −0.00059; venue z_median 0.046, z_max 0.12 — far shallower than the assumed z~0.3–0.5) | QUANTIFIED [L3] 2026-07-10 — NEGLIGIBLE; era-corrected residual +0.0138 (raw +0.0132) → **EXPLAINED [L8] 2026-07-11 (N-4): σ_z/z-at-low-z truncated-volume-kernel Eddington effect, estimator-intrinsic; reproduced +0.030 in a venue-matched harness at z_med 0.044; seed600 attribution CONFIRMED 2026-07-12 — its low-z hosts are 89.7% photometric, σ_z≈0.0344, σ_z/z≈0.65 (O(1)), and the likelihood kernel's z≥0 clamp is active for them**; `results/seed600_omega_m_era_20260710/` + `results/pp_coverage_shallowvenue_20260711/` |
| Ω_m Planck-vs-M1 (real data only) | both | +1.5–2.5% | QUOTED model-scope (G7 row 6); zero in Ω_m-consistent closures |
| w_G(h)=β_G/D(h) slope on deep venues | both | ~26% of seed1000 1D rail tilt | NEW (FINDINGS_COMBINE_20260710) — **estimator-level synthetic confirmation 2026-07-10 (L-A)**: completion-dominated ensembles biased HIGH (B_num/D increasing in h), see next row |
| Zero-host silent drop / pure-completion fallback calibration | both | **L-A synthetic: +0.7…+5.4% HIGH bias + coverage collapse at comp_frac 0.22–0.85** (controls calibrated; comp_frac≈0 exact) | **#29 fix landed; L-A VERDICT (2026-07-10): the fallback estimator is NOT calibrated at deep incompleteness** — EXP-40 must check for interior-but-biased-HIGH, not just de-rail; `results/pp_coverage_deepvenue_20260710/SUMMARY.md`. **MECHANISM DECOMPOSED 2026-07-11 ([L7])**: dominant part = membership-support kernel leak (σ_z-dependent; removed by the exact truncated-kernel mode); full Gray mixture makes it WORSE, not better; small σ_z-independent floor = inference noise-model approximation (σ(dL_obs)-vs-σ(dL_true) + p_det-inside, the two halves of the latent-threshold exact conditional — 260711-hx1 CONFIRMED, ~85–90% removed by both together; tiny 2nd-order residual ≈15× below σ_boot). **FLOOR DECOMPOSITION COMPLETE.** |
| Effective-catalogue depth (M_BH prune) | both | structural | **#30 — design decision** |

## 3. What runs LOCALLY NOW (consistent data only)

1. **[L1] DONE 2026-07-10 (both #29 AND the #30 caps), branch
   `physics/zero-host-completion-fallback` (pushed):** `ed46390` old-behavior pin →
   `8db6c6e` [PHYSICS] pure-completion fallback (B_num/D, WARNING + yield metric,
   catalog_only keeps skip, independent (1−w_G)·L_comp cross-check, hosts-present
   bit-unchanged) → `f29a5e7` [PHYSICS] selection z-caps (no-op in production,
   binds in the synthetic fixture → pipeline golden re-pinned with documented
   fingerprint) → `e19fcb2` docs (H0_BIAS_RESOLUTION §3.20 + §3.2 correction,
   DATA_INVENTORY rows). Issues #29/#30 commented, kept open for deep-venue
   validation.
2. **[L2] DONE 2026-07-10 (L-B A/B, `results/seed600_ab_20260710/ANALYSIS.md`):**
   (i) **1D code-drift gate PASS exactly** — run_A @`fc45d1f` reproduces the `562918ef`
   run_live combined 1D posterior to 5 decimals (MAP 0.7450, mean 0.74320); per-event worst
   rel 2.6e-08 (spline-table d_L tolerance), 0.05% of scalars >1e-9. (ii) **2D A-vs-live
   difference = the documented `713fbd1` Category-B D_g fix, NOT drift** — see [L4] update.
   (iii) **#29 real-data footprint (run_B @`f29a5e7`)**: 13 zero-host events restored
   (221=13×17 empty→filled per channel), hosts-present events bit-identical, #30 caps
   confirmed no-op; 1D MAP unchanged, mean +0.0003; of the 13 restored, 2 excluded by the
   combine zero-floor (net 11 contributing). Yield metric's first real-data run: clean.
3. **[L3] DONE 2026-07-10: seed600 Ω_m-era term = −0.00059 in h (−0.08% of 0.73)** —
   3,375 prepared-CRB events, z recovered from d_L at the generation cosmology
   (Ω_m=0.25, round-trip exact to 7e-14), I-ratio via the repo's `dist()`. The venue is
   much shallower than §1 assumed (z_median 0.046, z_max 0.12), so the era term is ~4×
   below the low end of the estimated band. Era-corrected residual: **+0.0138** (raw
   +0.0132) — the era mismatch explains essentially none of the venue residual; the §1
   "biases LOW by 0.3–0.8%" estimate applies only at z≳0.3, which this venue never
   reaches. Artifacts: `results/seed600_omega_m_era_20260710/{compute_era_term.py,era_term.json,SUMMARY.md}`.
4. **[L4] DONE 2026-07-10: Eddington-in-M IS active in the PV-test code** (`4d780f0` is an
   ancestor of `562918ef`). So the frozen venue's 2D channel sits at mean 0.787 (**≈+0.057**)
   ALREADY post-Eddington — a large open 2D residual on this venue (caveats: 17-pt grid clipped
   at 0.805 may truncate the upper tail; Ω_m-era term §1 makes the underlying value slightly
   higher still; the G7row9 494-event driver saw post-Eddington 2D mean 0.7697 on a 7-pt grid —
   subsample/grid dependence unresolved). The campaign 2D channel is in a different regime
   entirely (seed1000: 40% of surviving events completion-governed, railed low).
   **UPDATE 2026-07-10 (L-B A/B): the perf-branch `713fbd1` exact semi-analytic D_g
   denominator (declared Category-B fix; the MC it replaced was up to +54% wrong for low-z
   wide-photo-z hosts) moves this venue's 2D channel to MAP 0.755 / mean 0.7546 on identical
   inputs — the +0.057 residual becomes +0.0246 under current code (57% of it was the D_g
   defect). 2D MAP now interior (grid-clip caveat weakened). Remaining 2D residual +0.025 =
   the open item; D4's "re-combine on existing JSONs" check is superseded by this measurement.**
5. **[L6] DONE 2026-07-10 (L-A): pp_coverage z_support deep-incompleteness sweep** —
   quick task `260710-sjm` (commits `fa50ad5..e0c429e`), verified. Verdict: the #29
   pure-completion fallback analog (B_num/D, clean two-branch limit) is **BIASED HIGH**
   at deep incompleteness — +0.7…+5.4% in h, cov68 collapse to ≤0.27, h_true=0.84 rails
   HIGH — growing with completion fraction AND σ_z; controls + comp_frac≈0 cells exactly
   calibrated. Registered EXP-40 prediction: post-#29 seed1000 risk flips from rail-LOW
   to biased-HIGH. Bears directly on D1/#30: explicit z-truncation recovers calibration.
   Natural follow-up: full-Gray-mixture branch in the harness (production host-found
   events carry a compensating B_num admixture — untested whether it restores
   calibration at 60–95%). `results/pp_coverage_deepvenue_20260710/SUMMARY.md`.
6. **[L5] pp_coverage reference points already on disk** (`results/pp_coverage_sigmaz_scan_20260703/`,
   6 JSONs bare/volume × σ_z) — cite, don't re-run; estimator core is calibrated to ±0.0007
   at G4b settings. Optional extension later: campaign-σ_z panel per seed (pre-registered §4b(c)).
7. **[L7] DONE 2026-07-11 (handoff N-1/N-2/N-3 executed — quick tasks 260711-07n/117/1ps/27m,
   all on `physics/zero-host-completion-fallback`): the deep-incompleteness HIGH bias is
   mechanistically DECOMPOSED.**
   (a) **EXP-41/N-1 adjudicated NEGATIVE** (`results/pp_coverage_graymix_20260711/`): the full
   Gray Eqs. 29+32 mixture `(β_G·L_cat_i + B_num)/D` does NOT restore calibration — it AMPLIFIES
   the high bias (worst +0.123 vs +0.032 two-branch at zs=0.2/σ_z=0.035; 12/12 cells fail); the
   B_num admixture flips host events from counterweight to co-tilt. The N-2b conditioned inverse
   (N_i/β_G, B_num/β_Gbar) does not rescue either (+0.005…+0.044) ⇒ not merely w_G bookkeeping.
   (b) **Dominant mechanism identified** (`results/pp_coverage_exactmode_20260711/`): the
   membership-support kernel leak — host-event kernels integrating past the catalogue support
   edge. The "exact" truncated-kernel mode (host numerator truncated at zs over common D)
   removes the ENTIRE σ_z-dependent bias: ladder two_branch +0.0033→+0.0368 (σ_z 0.002→0.035)
   vs exact FLAT; modes converge at σ_z→0. N-2d: a HARD truncation is misspecified under
   observed-z membership ⇒ any production adoption needs SOFT photo-z-marginalized membership
   weighting (f(z)-weighted kernel integrands) — /physics-change + literature pass (Gray 2020;
   Chen–Fishbach–Holz 2018; ICAROGW out-of-catalogue treatment) BEFORE production code.
   (c) **N-3 prior sensitivity NEGLIGIBLE** (`results/pp_coverage_priortilt_20260711/`): a 10%
   inference-side w_pop misspecification moves h by ≤ +0.05% (two_branch) / +0.015% (exact) —
   the deep regime is NOT population-prior-driven (ratio structure self-cancels). D1 evidence.
   (d) **Residual floor CONFIRMED + DECOMPOSED (260711-hx1 DONE, `77ee9d1`+`03438d8`,
   `results/pp_coverage_noisemodel_20260711/`):** the +0.002…+0.005, σ_z-independent,
   prior-insensitive, grid-robust floor IS (mostly) the inference **noise-model approximation** —
   the JOINT σ(dL_obs)-vs-σ(dL_true) width mismatch (constant σ_f·dL_obs vs the generative
   σ_f·dL_true) + the latent-detection p_det-inside factor, the two halves of the single exact
   conditional for this latent-thresholded model. `--sigma-model-in-likelihood` (z-dependent
   σ_f·A(z)/h with 1/σ(z) norm) **+ `--pdet-in-numerator`** removes ~85–90%: MAP bias
   +0.002…+0.005 → ≤ +0.0008 on the deep cells AND nulls the −0.002…−0.004 control offset, cov68
   nominal at campaign n. Neither half alone works (model-σ alone over-corrects negative; p_det
   alone was the 27m refutation — they must be applied TOGETHER). n_events scaling (250/1000/4000)
   ADJUDICATED the floor's nature: const-σ floor is **FLAT in n with cov68 COLLAPSING**
   (h=0.72 0.63→0.38→0.12) ⇒ a real ASYMPTOTIC model bias, NOT a finite-sample MAP-skew. A tiny
   **second-order residual** (~+0.0005, ≈15× below campaign σ_boot) survives even the fully-consistent
   estimator, visible only at n=4000. Fine-grid confirm (h_step 0.004≡0.001, ±0.0001) ⇒ not
   quantization. Floor is at/below campaign per-seed σ_boot (~0.005): practically subdominant for
   Paper B closure. **Production input (user-gated /physics-change):** the correct move is a
   self-consistent distance-error model + p_det-inside for latent-thresholded detection — do NOT
   add p_det alone. EXP-40 watch (cluster): interior-but-biased-HIGH in both regimes; production
   post-#29 mixture (const-σ, no-p_det-inside) carries BOTH the leak and the floor same-signed HIGH.
8. **[L8] DONE 2026-07-11 (N-4, quick task 260711-iic, `baeaa1c`+`4f603af`,
   `results/pp_coverage_shallowvenue_20260711/`): the SEPARATE shallow-venue 1D residual
   (seed600 comp_frac 0.4%, z_med 0.046, era-corrected +0.0138 / raw +0.0132) is
   ESTIMATOR-INTRINSIC — a σ_z/z-at-low-z truncated-volume-kernel Eddington effect.**
   (a) Venue depth ladder (calibrated volume kernel, NO truncation, `--d50-gpc`): calibrated
   at the commission depth (z_med 0.28, bias −0.002) → strong POSITIVE bias as the venue
   shallows (+0.011 at z_med 0.056, **+0.030 at z_med 0.044 = seed600 depth**), cov68 collapses.
   (b) σ_z sweep at the shallow rung: the bias VANISHES at σ_z ≤ 0.015 (calibrated, −0.002) and
   appears only at σ_z=0.035 (σ_z/z ≈ 0.8) ⇒ the host-z kernel N(z;z_gal,σ_z) truncates at the
   physical z ≥ 0 boundary and the volume/Eddington-in-z correction (derived for an un-truncated
   kernel) stops cancelling. (b) Jackknife on the on-disk seed600 `run_live` per-event JSONs
   (no re-eval, production `apply_strategy`+`combine_log_space`): reproduces the raw +0.0132; the
   residual is BROAD/SYSTEMATIC (62% of events tilt high, Gini 0.65, and trimming the highest-|tilt|
   events GROWS the residual) — NOT a heavy-tailed outlier subset, matching a per-event depth effect.
   **Load-bearing caveat CLOSED 2026-07-12 (measurement + code trace):** seed600's low-z
   redshift-error model IS large-fractional photo-z. Measured directly on the reduced GLADE+
   catalogue it evaluated, z-shell 0.03–0.06 (around z_med 0.046): **89.7% photometric hosts**,
   **σ_z median 0.0344** (photo 0.0345, spec 0.0014), **σ_z/z median 0.65** (photo 0.669) — σ_z/z
   ~ O(1), an almost exact match to the harness σ_z=0.035 rung that produced +0.030. Code-side
   airtight: the likelihood host-z kernel width IS this catalogue σ_z
   (`bayesian_statistics.py:2243`, `host_z_error_eff = sqrt(σ_z² + σ_z_pv²)`) AND applies the
   `[PHYSICS]` z≥0 clamp precisely "for low-z photo-z hosts (z_g < 4·σ_z)" (`:2234-2239`); at
   z_g=0.046, 4·σ_z=0.14 > z_g ⇒ the clamp is ACTIVE for these hosts, so the un-truncated-derived
   volume/Eddington correction stops cancelling. ⇒ the shallow +0.0132 IS this Eddington effect
   (the spec-z minority — 10.3% at σ_z/z≈0.033 — is the calibrated counterweight the jackknife saw).
   Cross-seed systematic-vs-scatter still needs the campaign (do NOT force locally).
   A single z≥0-truncation-aware / photo-z-marginalized volume kernel would address BOTH the deep
   membership-support leak (L7 (i)) AND this shallow σ_z/z effect (user-gated /physics-change).

9. **[L9] DONE 2026-07-12 (N-5, optional 2D-channel subsample check; G7row9 494-event driver at HEAD,
   `.planning/gate/G7row9_N5_postDgfix_SUMMARY.md`): the 494-event seed600 2D subsample is
   well-behaved under current code — no additional 2D subsample/grid pathology.** edge_mass
   0.216→0.003, mean 0.790→0.768 (pre-fix 2D railing toward 0.86 GONE); subsample 2D sits +0.0135
   above the full-venue 0.7546 = subsample-selection offset, NOT a code defect; 1D subsample 0.745
   reproduces the venue +0.013. NB the pre-fix artifact is NOT a clean D_g-only baseline (its 1D 0.730
   predates #29/z-clamp) — clean D_g attribution stays in the L-B full-venue A/B (0.787→0.7546).
   **Bonus: post-D_g-fix Eddington-in-M Δ2D = −0.0022 (was −0.020) ⇒ `bayesian_statistics.py:2400-2401`
   comment + quoted value now STALE (flag, don't edit — physics-trigger file).** Local 2D work
   exhausted; venue-level +0.025 2D residual remains campaign-gated (D4).

## 4. What WAITS for the cluster

- **[C1] #29 validation on the deep venue**: re-evaluate seed1000 with the fallback → measure
  the de-rail (prediction: completion term tilts anti-rail; the 1,992 restored events carry
  B_num/D information). Requires the depth15 pool (rsync it local on return — also unblocks
  EXP-36 and the CLI-combine cross-check).
- **[C2] #30 decision data**: with #29 landed, measure posterior-width contribution of z>0.5
  events (information content) → decide robust-only vs additional truncation flag.
- **[C3] The pre-registered criterion itself**: relaunch seeds 2000–6000 ONLY after #29/#30
  decisions; then §4b adjudication = the definitive bias verdict.
- **[C4] h=0.705 seed1000 grid-hole re-run** (one eval task).

## 5. Data-consistency rules for this investigation (binding)

- Absolute bias numbers: ONLY from Ω_m-consistent, current-tier venues (campaign seeds).
- seed600 frozen venue: A/B and bounds only, always quoting the §1 era term.
- seed400 (any pool): perf regression only, never physics.
- Every quoted number carries {CRB set, pool id+depth, catalogue version, code commit,
  normalization_mode} — no cross-era mixing.
- DATA_INVENTORY: add seed600 Ω_m-era note + seed1000 combine/rail entry when this lands.
