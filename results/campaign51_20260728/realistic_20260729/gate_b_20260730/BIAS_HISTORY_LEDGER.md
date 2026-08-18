# BIAS HISTORY LEDGER — every H₀-bias hypothesis ever tested (2026-03 → 2026-07-30)

Compiled 2026-07-30 for Gate B/C of `../CLAIM_2D_BIAS_20260730.md`. Purpose: stop
re-litigation. Every verdict below is quoted from a repo artifact; where a source is
ambiguous or self-contradictory it is marked **[AMBIG]** with the file:line.

Abbreviations: `H0R` = `docs/H0_BIAS_RESOLUTION.md`; `BRA` = `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md`;
`BI@` = `.planning/BIAS-INVESTIGATION-20260710.md` (**deleted from tree by `1fe428a`**, read via
`git show 1fe428a^:...`). "h" = H₀/100, truth 0.73 unless stated.

---

## 1. Chronological table

| # | Era / date | Hypothesis | Decisive test | VERDICT | Documented | Residual left |
|---|---|---|---|---|---|---|
| 1 | Ph 9, 03-29 | PSD missing galactic confusion noise | add `_confusion_noise` | fixed-and-landed, insufficient alone | `H0R:1973` | — |
| 2 | Ph 10, 03-29 | Fisher O(ε) forward difference | 5-point stencil | fixed-and-landed, insufficient alone | `H0R:1972` | — |
| 3 | Ph 11.1, 03-31 | KDE p_det artifacts | KDE → simulation-based IS | fixed-and-landed, insufficient alone | `H0R:1974` | — |
| 4 | 03-31 | Gaussian evaluation index bug (3D vs 4D) | δ-approx analysis | **NOT A FACTOR** | `H0R:1975` | — |
| 5 | Ph 15 | Spurious /(1+z) Jacobian in wbh numerator | removal | fixed-and-landed, insufficient alone | `H0R:1976` | — |
| 6 | Ph 17–20 | p_det grid boundary/construction (KDE era) | VALD-01/02 | **VALIDATED CORRECT** | `H0R:1977` | — |
| 7 | Ph 21–23 | Numerical posterior underflow | log-space + 4 strategies | fixed, **not the cause** | `H0R:1978` | — |
| 8 | v1.4 `44d5358` | p_det extrapolation `fill_value=0.0` | `None` + accessor split | fixed, partial (−9.2% → −6.9%) | `H0R:1979` | — |
| 9 | Ph 32 | `L_comp` local-window denominator | full-volume D(h) | **fixed — dominant at the time**: MAP 0.60 → 0.73, bias −17.8% → 0.0% | `H0R:1980`, `project_bias_audit.md:10-16` | — |
| 10 | Ph 33 | p_det grid resolution 30 vs 60 bins | A/B | **not a bias source** (Audit A8 G8a re-confirmed) | `H0R:1981` | — |
| 11 | Ph 34 | Fisher `allow_singular=True` | condition gate | fixed-and-landed | `H0R:1982` | — |
| 12 | Ph 36 + 43-H2 | Equatorial CRB vs ecliptic GLADE frame | ecliptic migration | **fixed — PRIMARY mover per A5**: host recovery 31→38/60; H2 alone moved MAP 0.860→0.730 | `H0R:1983`, `H0R:134` | — |
| 13 | Ph 37 PE-02 | Per-parameter Fisher epsilon | per-param ε | fixed-and-landed | `H0R:1984` | — |
| 14 | Ph 38 | `L_cat` formula (Gray Eqs 24–25) | STAT-01 | fixed — **later REVERSED as a misreading**, see #26 | `H0R:1985`, `H0R:1124-1126` | — |
| 15 | Ph 43-H1 | `extract_baseline` missing −N log D(h) | add outer term | fixed — **later found actively harmful**, see #17 | `H0R:1986`, `H0R:667-672` | — |
| 16 | Ph 44 | h-dependent p_det zero-fill cutoff c₀ ∝ 1/h | grid fix | fixed: cluster 0.860 → 0.7650 | `H0R:1988` | +0.035 |
| 17 | Ph 45, 05-02/03 | p_det first-bin asymptote / d_L=0 anchor is the bias | Plan 45-06: raise anchor 0.7931 → 0.8873 (+12%) | **REFUTED — anchor is the wrong layer**: MAP unchanged 0.7550/0.7450, ZERO grid steps | `H0R:184-208`, `H0R:1989-1990` | — |
| 18 | Audit A0 | `TRUE_HUBBLE_CONSTANT=0.7` inconsistency | grep production path | **NOT A PRODUCTION PATH** (dead code) | `H0R:1991` | — |
| 19 | Audit A1 | MAP shift is a discrete-grid artifact | continuous refine | **falsified** — real continuous shift (Δ=0.0010) | `H0R:127-128` | — |
| 20 | Tier 3, 05-04 `6754ddb` | D(h) double-counted (inside `L_comp` AND outer −N log D) | `test_22_dh_double_count` c∈{0,1}; closure h=0.65 | **fixed**: eliminated +0.020…+0.025; closure 1D 0.6708→0.6555 (z +5.62σ→+1.67σ) | `H0R:651-695` | 1D +0.014 / 2D +0.012 |
| 21 | Bridge, 05-05 `2b33cad` | Raw scipy p_det extrapolation at d_L<dl_min | principled monotonic bridge | **fixed**: 1D closed 0.7309 (+0.0009, +0.19σ) | `H0R:697-770` | **2D +0.0141 (+3.60σ)** |
| 22 | H3, 05-06 `f01595c` | 2D numerator queried the *observation* `_det_M` not the *hypothesis* `host_M(1+z)`; grid M-axis source- vs observer-frame | Option-A: observer-frame M_z everywhere | **fixed**: 2D +0.0141 → +0.0007 (20×); info-monotonicity restored | `H0R:788-870` | ≈0 (that era) |
| 23 | Ph 48, 05-07 | — (production fine grid, 1473 ev) | 63-pt grid | 1D 0.7324 (+1.16σ) PASS · 2D 0.7322 (+0.97σ) PASS | `H0R:111-122` | — |
| 24 | 05-05 | H1–H5 2D-channel hypotheses (seed-drift σ_boot; residual D(h) coupling; wbh normalization; injection-set pull; grid clamping) | proposed diagnostics | **all UNTESTED at handoff**; superseded by #22 | `HANDOFF-2D-BIAS-INVESTIGATION-20260505.md@aac4c70^:74-141` | n/a |
| 25 | 05-16 | Row-424 CRB bimodality is a campaign bug | provenance trace | **seam, not bug** (seed200⊕seed300; emcee under-mixing). **Impact on H0: NONE** | `H0R:881-...`, `project_crb_two_population.md:17` | — |
| 26 | 06-19/20 `816f904` | `L_cat` departs from Gray A.9/A.10: spurious p_det in numerator + mean-of-ratios | equation-level read of 1908.06050 pp.16-21 + seed400 A/B | **fixed, sign-test PASSED**: 1D 0.750→0.740 (**halved**), 2D 0.7375→0.7350 | `H0R:1119-1139` | **1D +0.010 / 2D +0.005** |
| 27 | 06-19 | Fisher ecliptic/equatorial frame mismatch (seed400) | construction-chain + PD check, 2 adversarial refuters | **EXONERATED / non-cause**; rotating would double-rotate | `H0R:1010`, `project_fisher_frame_mismatch.md:7-32` | — |
| 28 | 06-19 | Missing `dd_L/dz` Jacobian in the catalog z-integral | sign + width analysis | **REFUTED** — ∝1/h is *decreasing* → biases h LOW; GW MVN is a likelihood, no Jacobian | `H0R:1011` | — |
| 29 | 06-19 | `L_cat`/`L_comp` normalization-measure mismatch | scale comparison at h=0.73 | **RULED OUT** — median `L_comp/L_cat` = 0.71; effective completion weight 41% ≈ nominal 40% | `H0R:1012` | — |
| 30 | 06-19 | `galaxy.py` (1+z)³ σ_z (#7); `OMEGA_M=0.25` (#6); stellar→BH relation | code/sign checks | **inert** / **inert** / **suppressor, not cause** | `H0R:1013-1015` | — |
| 31 | 06-19 | Missing (1+z)⁻¹ volume-time factor; flat 21% completeness tail | Gray A.2 / Fig.1 | genuine deviations, **WRONG SIGN** (fixing nudges H0 *up*) | `H0R:1057-1061` | — |
| 32 | 06-19 | Completion term `L_comp` is defective | `--catalog_only` decomposition, f-sweep | **faithful to Gray**; alone 0.800 (+0.070) but anchored → contributes ~+0.010; MAP flat over f∈[0.1,0.9] | `H0R:1037-1045` | — |
| 33 | 06-19 `5e94139` | p_det kernel tail-overshoot (Piece A) | detection-horizon survival estimator | **fixed**: D(h) decline −3.9% → −0.87%; 1D 0.760→0.750, 2D 0.747→0.7375 | `project_pdet_horizon_survival.md:58-75` | Piece B 1D +0.020 / 2D +0.0075 |
| 34 | 06-19 | F4-v2 local-linear p_det is the bias fix / a bias amplifier | NW vs LL A/B | **bias-neutral**: identical MAP 1D 0.7600 / 2D 0.7500 both estimators | `project_f4v2_verification.md:10-18` | — |
| 35 | 06-20 `af6014d` | Source- vs redshifted-mass injection convention | end-to-end trace + 3 refuters | real defect, **NOT the bias cause** (residual is dominant in the mass-free 1D channel); fixing pushes H0 *up* | `project_mass_convention_defect.md:22-32` | — |
| 36 | 06-29/30 | The remaining rail (seed600, MAP 0.86, **+0.13 ≈ +18%**) has a single root cause | bridge rung ladder A–I, one real ingredient at a time | **ROOT-CAUSED to rung G — host photo-z convolution at σ_z≈0.035 ≈ 17× σ_z^GW**: rung G 0.857 (+0.127); delta-z 0.725; every other ingredient recovers alone (A 0.729, B 0.734, C-real 0.725, D 0.735, E 0.725, F 0.735, radius 0.73) | `BRA:113-151` | — |
| 37 | 06-30 | Numerator-only normalization cleans (Angle A/C, Angle B) cure the rail | rung_I gate + de-rail | **DISQUALIFIED** — both rail UP to **0.870** at σ_z=0.035 | `BRA:76-82` | — |
| 38 | 06-30 | Local same-kernel consistent denominator cures it | σ_z→0 gate | **GATE FAIL** — 0.870 at *both* σ_z | `BRA:81` | — |
| 39 | 06-30 | Global photo-z-smeared selection `D_sm` cures it | multi-seed + n_gal convergence | **DE-BIAS, NO PEAK**: std 0.11 → 0.097 at n_ev=2000 (did not shrink); peaks 0.64/0.64/0.69/0.87, never 0.73; `E[h]≈0.735` a grid-midpoint artefact | `BRA:309-322` | — |
| 40 | 06-30 | "The correct unbiased photometric form exists in the literature" | 5-paper × 5-dimension reverse-engineering | **REFUTED** — no method validates σ_z/z≈0.7, z≈0.05, p_det≈1 | `BRA:166-199` | — |
| 41 | 06-30 | **VERDICT: in-catalogue photo-z dark sirens are information-starved** | (the above) | **later OVERTURNED** — see #52. `BRA:3-15` carries the supersession note | `BRA:351-362` | — |
| 42 | 06-30 | Spec-z host subsets carry the informative posterior shape ("money figure") | F4 decomposition on real seed600 + GLADE+ | **REFUTED** — spec-z = 0.56% of GLADE+, ≤8.7% (median ~0%) of rate-weighted in-cat likelihood; inference-side `flag==3` cut still rails 0.870 | `docs/F4_SPECZ_DECOMPOSITION.md:8-15`, `H0R:1268` | — |
| 43 | 06-30 `7021f6f` | Heliocentric vs CMB-frame host z (#15) | astropy dipole quantification | **fixed**: net **+0.15%** (per-event envelope ±2.47%), ~120× smaller than the rail and orthogonal | `H0R:1310-1330`, `BRA:390-398` | — |
| 44 | 07-04 | Host peculiar-velocity value correction (#16) | seed600 live vs noPV, 17-pt grid | 1D Δmean **−0.0142** (worst case); **2D PV-insensitive +0.0012**; **#16 CLOSED** | `project_code_review_20260704.md:18` | — |
| 45 | 07-02 G1 | Option-A cancellation Σ_global(h) ≡ n̄·β_G(h) | discrete GLADE sum vs continuous β_G, 14 h | **FAIL for `global` mode**: raw ×2.48 is the expected n_gal∝h³ (cancels), but a **−17.2% end-to-end residual** remains (+8.7%→−8.7%). `global` deprecated; **local modes structurally immune** | `docs/gates/G1_beta_g_check.md:14-29` | — |
| 46 | 07-02 G2a `cb16142`/`4a259b7` | Completion term evaluated at peak sky density; missing sinθ_det Jacobian | 4π marginal derivation | **fixed**: ~5000× B_num inflation → rail at grid edge; sinθ median 1.15× over-weight | `docs/gates/G7_systematics_budget.md:12-13` | — |
| 47 | 07-02 G2b `235b783` | Bare-Gaussian host-z numerator = Eddington-in-z ∝ σ_z² | derivation + commission coverage test | **CONFIRMED and fixed** (`volume_deconv`): bias −0.024 → **−0.002**, coverage ≈0% → nominal. Law Δh = −C σ_z², C_meas≈17–20; measured **−0.0016/−0.0064/−0.023/−0.046** at σ_z=0.005/0.015/0.035/0.050 | `docs/derivations/G2b_host_z_volume_prior.md:229-237,413-436` | −0.002 floor |
| 48 | 07-02 G3 | Which factor de-rails? | ablation cube (494-ev seed600) | bare+global rails 0.60 → **volume kernel in N_g AND D_g → 0.76** → **+ local denominator → 0.73** | `docs/gates/G6_starvation_postmortem.md:12-16` | — |
| 49 | 07-01/02 | Real-data de-rail matrix | 7-h grid, 494 ev | pre-4π **0.86** ↑rail → 4π-only **0.60** ↓rail (necessary, not sufficient) → `local_ratio` **0.73** (98% mass) → `volume_deconv` **0.73** | `project_commission_derail.md:12-18` | — |
| 49a | 07-01 commission | Is the railed estimator even responsive to truth? | 10-way tournament injection scan | **production MAP = 0.86 for EVERY injected truth 0.63→0.77**, while `catalog_only` tracks truth exactly (0.63→0.63 … 0.77→0.77) ⇒ the railed production estimator is H₀-**independent** | `synthesis/DRAFT_REPORT.md:24-27`, `WF2_DIGEST.md:26-30` | — |
| 49b | 07-01 commission | Archaeology: when did the rail start? | git-bisect of stored posteriors | **2026-04-09 baseline (n=417) map_h=0.735, unbiased**; rail to 0.86 first appears **~2026-04-24** when the Gray selection/completeness machinery was switched on | `WF1_DIGEST.md:9-15` | — |
| 49c | 07-01 D6 / D8 | Truth-vs-fiducial 0.70↔0.73 bookkeeping masquerade; d_L→z inversion σ² bias | static+runtime trace; numerical | **both REFUTED**: the 0.70 sites are all in `datamodels/galaxy.py`, **0 production importers**; `dist()` exact to 1e-14, effect sub-% | `D6_h0_bookkeeping_trace.md:23`, `WF2_DIGEST.md:45-54` | — |
| 49d | 07-10 W-CONF-19 | The 10k-MC → erf-sum with-BH denominator swap moves the 2D MAP | 29-pt grid, worktree A/B | **1D byte-identical**; 2D tilt −50.6 log₁₀/h *toward truth*, **\|δh\| ≲ 7e-4** (below the 0.005 grid) | `eval_perf_confirm_20260710/FINDINGS.md:20-34` | — |
| 50 | 07-02 G2d `4d780f0` | Eddington-in-M (mass channel not rate-deconvolved) | moment-matched rate prior | implemented; 2D mean **−0.020** toward truth — **later re-measured post-`713fbd1` at only −0.0022** (code comment `bayesian_statistics.py:2400-2401` STALE) | `GATE_SIGNOFF.md:17`, `G7row9_N5_postDgfix_SUMMARY.md:41-45` | — |
| 51 | 07-02 G8 `fcc49c4` | Missing dt² DFT normalization in the inner product | Parseval + 5 evidence lines | **fixed**: SNR was physical/10, CRB σ ×10, population depth z≤0.11 instead of ≲1.5 | `G7_systematics_budget.md:10` | — |
| 52 | 07-02 G6 | Reconcile "starvation" with the de-rail | rung G2 (numerator-only volume tilt) as negative control | **STARVATION OVERTURNED**: "a property of prior-INCONSISTENT estimators, not of the data"; consistency ("counted exactly once") is the cure | `G6_starvation_postmortem.md:24-33` | — |
| 53 | 07-02 G11 `bdf5339` | WMAP-era Ω_m is a bug | population matching | **design choice**: Ω_m=0.2726 matches Barausse-2012 M1; Planck case QUOTED (+1.5–3%) | `GATE_SIGNOFF.md:26` | quoted |
| 54 | 07-10 `8db6c6e` (#29) | Zero-host events silently dropped → deep rail to h=0.60 | code archaeology + 32-agent verification | **real bookkeeping bug, FIXED** (58% dropped on deep venue) — but see #55 | `H0R:1368-1420` | — |
| 55 | 07-19/25 EXP-40 | The #29 fix clears the deep rail | full re-eval, code the only variable | **REFUTED — rail PERSISTS at 0.6000 both channels**. Split: host events −4265 over the grid; **fallback events h-INERT (−59)** | `FINDINGS_EXP40_20260719.md:14-50` | — |
| 56 | 07-25 (#30 b) | Depth truncation cures the deep rail | `--max_redshift` 0.2/0.3/0.5 scan | **empirically dead — rails at all three depths** | `H0R:1433-1435` | — |
| 57 | 07-10 L-A | The #29 pure-completion fallback estimator is calibrated | pp_coverage `z_support` deep sweep, truth known | **BIASED HIGH**: **+0.7…+5.4% in h**, cov68 ≤0.27 at comp_frac **0.22–0.85**; h=0.84 rails HIGH; controls (zs=1.0) calibrated | `pp_coverage_deepvenue_20260710/SUMMARY.md`, `BI@1fe428a^:98-107` | — |
| 58 | 07-10 L-B `713fbd1` | 2D MC `D_g` denominator defect (up to +54% wrong) | seed600 A/B, identical inputs | **fixed — 57% of the +0.057 2D residual**: venue 2D mean 0.787 → **0.7546** | `results/seed600_ab_20260710/ANALYSIS.md`, `BI@1fe428a^:92-97` | **2D +0.025** |
| 59 | 07-10 L-C | seed600 Ω_m era mismatch explains the shallow +0.0132 | exact per-event Δh | **NEGLIGIBLE and wrong-signed**: Δh̄ = **−0.00059 (−0.08%)** → era-corrected residual **+0.0138** (larger) | `seed600_omega_m_era_20260710/SUMMARY.md:10-58` | — |
| 60 | 07-11 N-1 | Full Gray mixture `(β_G L_cat + B_num)/D` restores deep calibration | EXP-41 harness | **AMPLIFIES**: worst **+0.123** vs two_branch +0.032; **12/12 fail**; host branch flips from counterweight (−26…−182) to co-tilt (+47…+166) | `pp_coverage_graymix_20260711/SUMMARY.md:12-42` | — |
| 61 | 07-11 N-2b | The defect is `w_G = β_G/D` bookkeeping | membership-conditioned inverse | **REFUTED** — +0.005…+0.044, 12/12 fail; tilt merely *relocates* to the host branch (+94…+455) | `pp_coverage_graymix_20260711/SUMMARY.md:64-75` | — |
| 62 | 07-11 N-2 | **Membership-support kernel leak** is the dominant deep mechanism | `exact` truncated-kernel mode + σ_z ladder | **CONFIRMED as dominant**: removes the ENTIRE σ_z-dependent leak (two_branch +0.0033→+0.0368 over σ_z 0.002→0.035; exact FLAT +0.0023…+0.0046); rails 0.45–0.92 → 0.00–0.19. **Formally still biased, 1/12 pass** | `pp_coverage_exactmode_20260711/SUMMARY.md:21-37` | floor +0.002…+0.005 |
| 63 | 07-11 N-2d | A hard support truncation is the production fix | membership on *observed* z | **REFUTED for production**: sign-flipping biases −0.021…+0.015, cov68 0.18–0.46 → needs a **soft photo-z-marginalized** membership | `pp_coverage_exactmode_20260711/SUMMARY.md:247-256` | — |
| 64 | 07-11 N-3 | The residual floor is population-prior (w_pop) misspecification | γ-tilt ladder ±10% | **NEGLIGIBLE / escape hatch CLOSED**: ≤ **+0.0004** (two_branch) / **+0.0001** (exact) | `pp_coverage_priortilt_20260711/SUMMARY.md:25-36,147` | — |
| 65 | 07-11 | The floor is a grid/quadrature artifact | 4× finer h-grid, 2× finer z-quad | **PERSISTENT — genuine composition residual** (Δ ≤ 0.0002) | `pp_coverage_priortilt_20260711/SUMMARY.md:37-45` | — |
| 66 | 07-11 27m | The floor is the p_det-inside-numerator factor | flag A/B | **REFUTED** — deep cells unchanged (Δ≤+0.0006) and it *flips* the calibrated controls −0.003 → +0.003…+0.006. **"Do not cargo-cult it."** | `pp_coverage_pdetnum_20260711/SUMMARY.md:10-22` | — |
| 67 | 07-11 hx1 | The floor is the σ(d_L^obs)-vs-σ(d_L^true) noise model | 2×2 model-σ × p_det-inside + n-scaling | **CONFIRMED as dominant**: both together remove ~85–90% (+0.002…+0.005 → ≤+0.0008). Const-σ floor is a **real asymptotic bias** (flat in n, cov68 0.63→0.38→0.12). **Neither half alone works** | `pp_coverage_noisemodel_20260711/SUMMARY.md:18-38,80-86` | ~+0.0005 |
| 68 | 07-11 N-4 | The shallow seed600 +0.0132 is a σ_z/z-at-low-z truncated-kernel Eddington effect | depth ladder + σ_z localizer + jackknife | **P-A refuted, P-B confirmed** at the time: harness +0.030 at z_med 0.044/σ_z 0.035, vanishes at σ_z≤0.015; residual is **broad, not outlier-driven** (trimming top-|tilt| GROWS it) | `pp_coverage_shallowvenue_20260711/SUMMARY.md:12-76` | **[AMBIG] see #69** |
| 69 | 07-13 | …the harness +0.030 is the *generative* z≥0 clamp, not the volume correction | toggle `clamp_zgal` only | **RE-ATTRIBUTED**: shallow +0.0240 (clamp ON) → **−0.0056** (OFF); deep venue clamp-independent. **Production catalogue is NOT clamped** (no pileup at 0; only 3.7% of low-z hosts have z<σ_z) ⇒ "harness +0.030 is **substantially a harness artifact**"; the seed600 **+0.013 attribution is REOPENED, campaign-gated** | `results/h1_zclamp_20260713/FINDINGS.md:3-8,39-59` | +0.013 unowned |
| 70 | 07-12 `c4a1c7d` | `volume_trunc` (unify the host-z numerator window) fixes the shallow bias | pre-registered seed600 494-ev A/B | **FALSIFIED**: 1D mean 0.745 → **0.800**, 2D 0.768 → **0.800** — wrong way by ~4×. Two causes: `fixed_quad(n=50)` **aliases the GW peak** (0.0000 vs exact 0.24–0.65) *and* the exact host-window numerator itself tilts high | `results/volume_trunc_ab_20260712/FINDING.md:1-58` | — |
| 71 | 07-13 EXP-45 toy | Truncated-lognormal×R_eff host-mass kernel owns the 2D residual | single-host toy | **CONFIRMED in isolation** (+0.016…+0.02 at σ_z/z 0.5–0.65, sign HIGH) | `mass_kernel_truncation_20260713/FINDINGS.md:9,54-70` | — |
| 72 | 07-13 `70bee1f` | …the same, in the full pipeline | seed600 494-ev A/B | **EXONERATED**: Δ2D mean **+0.0029, wrong sign**; Δ1D **0.0000 exact**. Reason: the same prior enters `D_g`, so `N_g/D_g` cancels the numerator shift — **"the selection denominator is not a spectator"** | `mass_trunc_ab_20260713/FINDING.md:3-12,54-78` | 2D +0.025 open |
| 73 | 07-12 N-5 | The 2D residual is a subsample/grid pathology | 494-ev probe at HEAD | **REFUTED**: edge_mass 0.216→0.003; offset +0.0135 is subsample selection. "Local 2D work is now exhausted" | `docs/gates/G7row9_N5_postDgfix_SUMMARY.md:14,50-53` | — |
| 74 | 07-25 D1/D2 | Deep rail is **host misassociation** (self-normalized `L_cat` over impostor-only balls) | 12 instrumented events + overlap model | **CONFIRMED**: 91–100% of tilt in the numerator overlap; rail events' `h*_g` median **0.42–0.48**; model predicts sign 11/12 | `H0R:1436-1456`, `SYNTHESIS.md:8-16` | — |
| 75 | 07-25 D2 | `volume_deconv` kernel carries h-dependence | analytic + numeric | **EXONERATED — exactly h-invariant** (`Z_g ∝ h⁻³` to 1e-15) | `H0R:1446-1447` | — |
| 76 | 07-25 D2 | Ball-restricted selection denominator `Σ_ball w_g D_g` | surgical local→global swap | **real but SECONDARY**: 1–14% of tilt, opposite sign; swap softens deepest event only −54.6 → −50.6 ln/h | `SYNTHESIS.md:17-20` | — |
| 77 | 07-25 `49b9ade` | V1 `absolute_marginal` (absolute-mass host marginal) cures the deep venue | seed1000 probe + seed600 gate | **relocates the rail LOW→HIGH**: 0.86 both channels; seed600 **1D 0.775 (+0.030)**, 2D rails 0.86 | `H0R:1475-1482` | — |
| 78 | 07-25 | V1's `n̄_w = Σ_glob/β_G` calibration identity holds on the real catalogue | composition check | **VIOLATED** — 33% in value, **0.39/h in log-slope** | `H0R:1548-1552` | — |
| 79 | 07-25 `f9c58f4` | σ_z-smeared `Σ_glob` fixes the `n̄_w` slope | measurement | removes only **~20%** (+0.067/h of a +0.38/h target); later moot | `H0R:1491-1497` | — |
| 80 | 07-26 E1 | The completion term `B_num` is defective (fallback-only peak 0.6118±0.0176, ≈6.5σ low) | self-consistency MC + membership-clean real subset | **EXONERATED — subset-conditioning artifact**: `B/β_Ḡ = 0.7366±0.0155` closes at truth; real 0.6329 vs MC 0.6328. Components M-A +0.12 / M-B −0.11 / M-C −0.133 / M-D −0.02…−0.05 | `E1_COMPLETION_BIAS.md:10-34,120-142` | — |
| 81 | 07-26 `8fbb21e` FIX-3 | Generator-consistent normalization `generator_marginal` | seed1000 7-pt probe (pre-registered to NOT de-rail) | **prediction FALSIFIED, favorably — 1D and 2D MAP = 0.73 = truth**, gaps −898.8 / −735.4 ln; every normalization quantity matched to 3–4 s.f. | `H0R:1556-1580` | — |
| 82 | 07-26 `a608c4f` FIX-2 | Pooled survival mis-shapes selection (over-states SNR≥20 rate by +30–45%) | z-resolved `S(d_L\|z)`; stacked + isolated probes | **CONFIRMED**: stacked preserves truth (gaps −994.8/−831.4); isolated **−68.75 ln vs −69 predicted** | `H0R:1581-1610` | — |
| 83 | 07-26 | Multi-seed closure of the production stack | seeds 900/1000/2000/3000/90000 | **QUALIFIED FAIL** as registered (seed900 invalid pool: 57.6% of cells below ESS floor). Valid-4: MAP 0.7304 / 0.7300 / 0.7297 / 0.7287; dense core **0.72976 / 0.72979, σ 2.1–2.5e-4** | `MULTISEED_READOUT_20260726.md:22-28,56-60,166-184` | −0.0002±0.0002 |
| 84 | 07-26 | seed600 shallow "must-not-change" gate, third arm | pre-registered 4 criteria | **criteria 1–2 PASS, 3–4 FAIL** (3/3355 new zero events at z<0.01); "conditional adoption" recommended. **[AMBIG]** — `RUNBOOK_NEXT_SESSION.md:113` calls it "MAP PASS" without the word FAIL | `SEED600_GATE_REGISTRATION.md:41-58,144-158` | open |
| 85 | 07-26 redteam | Estimator tuned to h=0.73 / peak resolved / precision general | anchor search, golden pulls, grid audit | **NO numerical anchor** (golden pulls mean +0.06, std 0.94, n=133); **R1 grid unresolved** (later discharged by dense core); **R2** 133 golden events carry ~100% of curvature ⇒ closure validates host association, **not** the selection machinery; **R3 all precision is mock-internal** | `redteam_20260726/CONSOLIDATED_VERDICT.md:13-40` | — |
| 86 | 07-26 | `absolute` mixture mode fixes the harness deep cell | pp_coverage A/B | **NO material difference** (Δ ≤ 0.0004, below SE 0.0007) — but the harness has one candidate/event so it **cannot exercise** V1's mechanism | `pp_coverage_absolute_20260726/SUMMARY.md:9-12,104-107` | — |
| 87 | 07-27 | Impostor-suppression claim of V1/FIX-3 | impostor-ball harness, n=2000 | **CONFIRMED**: at h=0.72 bias lcat +0.0061 vs absolute +0.0038 / genmarg +0.0042; h=0.84 rail 0.335 → 0.195/0.204. **Impostor channel and normalization channel both EXONERATED as residual carriers; the residual is entirely `B_num`** | `pp_fullpower_20260727/FULLPOWER_READOUT.md:43-58` | B_num model open |
| 88 | 07-26 **⭐** | Which leg owns the 2D HIGH tilt? | three-way per-leg A/B, seed1000 deep venue | **δ-kernel carries 85.3% (1D) / 86.7% (2D)**. **Cell B (broadened `volume_deconv` numerator + generator norm): 1D = 0.73 truth, 2D = 0.80 INTERIOR HIGH, +29.4 ln.** Only the δ (point) kernel brings 2D to truth | `threeway_ab/THREEWAY_AB_READOUT.md:19-56` | **2D +0.07** |
| 89 | 07-27 `e9bec6d` | The ratified (M1) truncated-lognormal mass kernel closes the 2D tilt | 4-cell A/B (A′/A″/B/B″) | **NECESSARY, NOT SUFFICIENT** (P4(ii) fires): moves 2D down only **−1.8 / −2.3 ln** of a +25.6 / +29.1 excess; **2D MAP unmoved at 0.80**. Cell **A′ = absolute_marginal + volume_deconv + gaussian: 1D 0.73, 2D 0.80** | `mass_ab_20260727/MASS_KERNEL_AB_READOUT.md:23-80` | +23 ln |
| 90 | 08-04/05 gate (vii) | Fix A + Fix B path-A shrink / cannot move the dark catalogue-leg channel difference (A2 exoneration) | post-fix 41-h Σ_dark Δln(L_cat^2D/L_cat^1D) both venues + paired cross-venue check + leg ablation | **REFUTED — GREW −504.8→−604.8 (bit-identical N=534, +19.8%/event mean); composition-dominated (81% from 316 scatter-resurrected events tilting 3.01× steeper; robust stratum −112.7); catalogue leg is a DOWN-pull muted by w̃_G** | `results/run_20260804_postfix/gate_vii/{gate_vii_readout,paired_check}.json` | A2 exoneration void; D1 NOT demoted; quote with N+venue+stratum |
| 91 | 08-04/05 frozen-g | g_frac(h) h-slope (completion-leg mass factor) carries the residual 2D displacement | pre-registered frozen-g_frac 41-h live evaluate, both venues (PREREGISTRATION_FROZEN_GFRAC.md) | **CONFIRMED — 2D MAP 0.780/0.800→0.660/0.640 in band [0.63,0.665]; live==proxy 0 grid steps; 1D + selection objects bit-identical** | `results/run_20260804_frozeng/readout.json` | derivation question open (correct physics vs defect) → /physics-change; possible D1 convergence |
| 92 | 08-05 closed-loop | the 2D estimator's g-machinery displaces the MAP high even when the universe follows the estimator's own assumptions | closed-loop 2-channel harness (`validation/closed_loop_gfrac.py`), 200 seeds, pre-registered G4b/§9 bands | **MIXED/not-REFUTE — Δ2 = +0.011 ± 0.004 (posterior-mean +0.005); production displacement NOT reproduced; 1D rails 200/200 in-loop (info-starved), removed 0/50 by numerator-selection diagnostic** | `results/closed_loop_gfrac_20260805/closed_loop_results.json` | 2D g-machinery approximately calibrated; supports "g h-slope = correct physics" |
| 93 | 08-05 N-2 sel-1d | the 1D completion numerator's missing S̄_φ(z;h) owns the production 1D rail | pre-registered two-venue 41-h counterfactual `--selection_in_completion_numerator 1d` (M1-rescaled bands) | **MIXED-bounded — chord +24.6/+22.7 in band, 1D MAP stays railed 0.600 both venues; central-diff overshoots (+30.9/+32.3 vs +30), sign coherence 0.71–0.73 vs ≥0.90 ⇒ M1's point evaluation under-models the live quadrature; all nulls held** | `results/run_20260805_n2sel1d/readout.json` | N-2 = real bounded correction, NOT the rail owner; formula adoption still open (author R-0..R-5); rail owner still at large |
| 94 | 08-05 D1 S_and re-weight | the p0-window selection (D1) reaches the core 2D-bias object via the estimator's selection machinery | pre-registered three-arm (A0/A1/A2) pool re-weight, distribution+stratum scored (`38ffa6ce`) | **MIXED — BOUNDED NULL: m_S=0.032, m_R=0.011 (thresholds 0.25); g_frac bit-identical under S_and (C7 route dead); N2 assumption falsified — L_cat's f_k is pool-fed (discovered fact, new intake); S1 joint Σ⁴ᴰ self-cert 12.5% flagged** | `results/run_20260805_d1/readout.json` | D1 does not own the 2D core object via the tilt route; f_k-pool coupling = new open thread; rail owner still at large |
| 95 | 08-05 M-1 Hitchhiker intake | the idealized (iiib) venue's host-z kernel is δ-like, so per-event-independence corrections (arXiv:2212.08694 Eq. 31) vanish there | M-1 kernel/code + catalogue z_error read (`m1_kernel_delta_check.py`) | **REFUTED — KERNEL-FINITE: `volume_deconv` pinned; parse-time PV floor gives median σ_z/(1+z)=0.033 ≈ photo-z scale on the PARENT catalogue; perfect-z escape clause fails on every venue; H-1 (1D rail via cross-terms) LIVE, to be MEASURED per author mandate** | `m1_kernel_delta_check.json` | cross-term instrument to build+pre-register; "idealized = exact z" shorthand needs kernel-level scoping everywhere it was used |
| 90 | 07-27 `608426b` | (d1) joint z×M_z with-BH selection conditioning owns the residual | pre-registered A/B + grid-only control | **NULL**: A-cell Δ@0.80 = −0.51 raw / −1.07 grid-corrected; the −6.5±4 gate **FAILS LOW then RETIRED** | `mass_ab_20260727/ZMZ_AB_READOUT.md:20-45` | — |
| 91 | 07-28 | (g1) the m_max clamp suppresses (d1) | P1 parity audit | **REFUTED** — clamped queries carry **75–90%** of the conditioning movement; the ×3–5 shortfall was an axis-translation error (D_gen multiplier 718 vs A-cell Σs≈225) | `mass_ab_20260727/P1_PARITY_AUDIT.md:3-8,208-225` | — |
| 92 | 07-28 | Remaining ≈+23 ln 2D HIGH residual (MAP 0.80) owner | enumeration after 89–91 | **STILL OPEN — owners: (d2) selection-side M scatter/truncation + (g1)-as-support-limitation**; campaign is the critical path | `RUNBOOK_NEXT_SESSION_4.md:29-33`, `docs/campaign_redesign_51_design.md:180-205` | +23 ln |
| 93 | 07-29 #51 | Idealized baseline (point kernel + `generator_marginal`, unscattered) | 1e-4 zoom grid | 1D/2D **0.72990 / 0.7300**, −0.24σ / −0.36σ. 100% of information from **76 in-cat events**; 3 golden carry 46% | `IDEALIZED_BASELINE_READOUT.md:25-47` | — |
| 94 | 07-29 #53 | Realistic run (`absolute_marginal` + `volume_deconv`, scattered catalogue) | P1–P6 scorecard, 10 runs | 1D 0.700–0.740 (pooled 0.7205) · **2D 0.780–0.820, mean +0.077, 10/10 pull > 2 (+4.04)**; P5 σ→0 byte-identical PASS | `REALISTIC_READOUT.md:19-32,140-153` | **current** |
| 96 | 08-06/07 crossterm | the neglected Eq. (31) pairwise cross-term (arXiv:2212.08694) is large enough to matter — H-1's 1D-rail leg and M-2-revived H-2's 2D leg (row #95 mandate: measure, never refute by convenience) | pre-registered frozen instrument (sha256 `340b66d2…`, ratified M-4 target sets 349/104 joint_r1 + 280/21 iiib pairs), mixture-composed band X=2.78/Y=7.96 LOCKED; mechanical readout + independent adjudication (`readout_adjudication_20260807.json`, **CONFIRMED, T bit-identical**) | **NEGLECT-WITH-NUMBER in all four venue×channel cells**: T = 1.468160e-05 (joint_r1/1d) / 5.648753e-05 (joint_r1/2d) / 1.024615e-05 (iiib/1d) / 5.379542e-06 (iiib/2d) class-summed mixture-composed chord nats; X/T = 1.89e5/4.92e4/2.71e5/5.17e5; anti-dilution never triggers; no venue split. **H-1's cross-term leg CLOSED by measurement** (5 orders below the 1D rail's −1.1…−1.6 nats/grid-point depth, #93). **M-2's revived-H-2 signature NOT matched** — composed 2D chord is NEGATIVE (high-h) at 5–6 orders below Y ⇒ cross-term **EXCLUDED as the mechanism of M-2's 2D overlap residual**. Discovered facts: raw catalogue-leg diagnostics (reported-never-scored) are large — full-grid ranges 181.208/5.799/77.514/0.0960 nats — and the raw 2D chords are **POSITIVE (low-h)**: +2.507 (joint_r1) / +0.0116 (iiib): H-2's coherence is physically present in the catalogue leg but **annihilated by the mixture composition** (median per-pair factor ~1.5e-07–2.5e-06) and **sign-reversed at class level by w_G's h-fall** (0.0957→0.0556); in-C-4 vs outside-C-4 1D sub-sums are **sign-split** (+1.8e-06 vs ~9× larger negative) — M-4 supersession vindicated | `crossterm_instrument/{CROSSTERM_READOUT_20260806.md,readout_20260806.json,readout_adjudication_20260807.json}` | M-2's 2D overlap residual (+0.02070/+0.02225 nats/event, p 0.0042/0.0050) REAL and now **UNOWNED** (§4 thread 16); NEGLECT conditional on `crossterm_instrument/NEGLECT_TRIGGER_REGISTER.md` (§5 ruling 08-07) |
| 97 | 08-08 M-2 stage 5 | Owner of M-2's matched 2D overlap residual (§4 thread 16) — Stage 1 readout (A1 sign-flip test + A2 completion-weight functional) and pre-registered Instrument B (branch-iii venue-split decision statistics DS-1–DS-5) | Stage 1 dual test (`STAGE1_READOUT_20260807.md`) + pre-registered Instrument B (`PREREGISTRATION_B_COMPLETION_WEIGHT_FUNCTIONAL.md`, `B_READOUT_20260808.md`), ratified by the author verbatim ("ratified") on the presented formulation | **RATIFIED — DISSOLUTION in modified-H-c form (author ruling 2026-08-08, §5)**: residual (+0.02225 iiib / +0.02070 joint_r1 nats/event) is REAL and component-coherent (H-e chance closed on the literal + majority graph readings, CONDITIONAL on the overlap-among-overlap exchangeability model — qualification q1), carried by the completion leg (T_Lcomp, D-1 LMDI, fraction +1.28/+1.33; catalogue leg NULL; venue-stability mechanically explained by bit-identical completion columns; 1D/2D dichotomy explained by w_G-offset cancellation + mass-conditioning unmasking), and confounding-absorbable by the collinear d_L-geometry + ball-density bundle traceable to C-4's 2σ d_L window: ~2/3 a smooth, strictly-monotone, verified d_L-functional completion-leg response (A2: R² 0.88, chain ratios 0.666/0.653), the remaining ~1/3 (the +0.0083 ± 0.0029 fixed-d_L excess) density-coupled at joint_r1 under every stability check (Instrument B DS-3: A_ρ +0.0069, cluster p 0.023; radius overlay p 0.0055) but specification-fragile (verifier ATT-5: p crosses 0.0455 under tensor-deg3 D; excess softens to ~2.1–2.5σ under flexible D) and NOT significant at iiib (A_ρ +0.0039, p 0.11) — recorded as a disclosed, bounded open object, NOT a claimed finding (qualification q2 sharpened). Instrument B fired pre-registered branch (iii) MIXED on this clean venue split; joint model chain reproduces in-band BOTH venues (ρ_J 0.82/0.97). The fragile joint_r1 1D signal (p=0.0414) is RETIRED. H-b and H-d refuted as owners (wrong-sign carriers); H-a carrier-only. No escalation, no `/physics-change` implicated, no production formula touched | `m2_residual_owner/{CLAIM_M2_RESIDUAL_OWNER_20260807.md,STAGE1_READOUT_20260807.md,PREREGISTRATION_B_COMPLETION_WEIGHT_FUNCTIONAL.md,B_READOUT_20260808.md}`; evidence commits e253e0c1 (stage 0-1), c188f460 (A1/A2), 36fe7800 (prereg B), 39d8a227 (instrument B) | §4 thread 16 CLOSED by this ruling (§5 ruling 08-08); the ~1/3 fixed-d_L density-coupled excess remains, by design, a disclosed bounded open object (q2) — open/non-attributed at iiib, specification-fragile (degree-3-conditional) at joint_r1 |
| 98 | 08-10/11 calgate-v2 | Stage-4 calibration gate v2 (item 7): is the P–P/coverage leg trustworthy, and does the 1D rail reproduce in-loop? (v1 fired GATE-NOT-TRUSTWORTHY on its own V4/DS-7 checks — commit 3a572897) | Pre-registered v2 (`results/calibration_gate_v2_20260810/PREREGISTRATION_CALIBRATION_GATE_V2.md`, commit 065e7f58; five v1 defects repaired, deviation register D1–D8; disjoint seeds; author-directed cluster venue, array 6250988, all 10 tasks COMPLETED 0:0) + mechanical readout + independent adjudication (`adjudicate_v2_readout_results.json`, **CONFIRMED, zero numeric/verdict discrepancies**) | **GATE TRUSTWORTHY; KEEP-DIGGING via clause (b), DEFECT-class**: V1–V5 all pass (V4 texture 0.6616–0.6669 inside the pre-declared-analysis band [0.63,0.75]); B2 decision cells FAIL DS-1 (C50/C68/C90 = 0/0/0) + DS-2 (KS D ≈ 1.0) at all three truths, both channels. DS-8 reproduction targets ALL CONFIRMED out-of-sample: T1 single-host starvation rail 400/400 ×3 truths (follows extended grid to 0.460); T2 ball-venue σ_z-DOSED uniform MAP bias ≈ +σ_z (0 → +0.011 → +0.035 at σ_z 0/0.010/0.035) with delta-narrow posteriors (post_sd 0.0012–0.0059) and 0% coverage; T3 B0 (σ_z=0) EXACTLY on truth 400/400 both channels. **The production 1D rail does NOT reproduce in the multi-candidate ball venue at any σ_z dose (R_low = 0 everywhere) — DS-6 MIXED**; the in-loop defect is a coverage collapse, not a rail. REPORT-BOUND unreachable (B2-2D fails DS-1+DS-2); paper #47 hold's reason upgrades from "P–P leg missing" to "P–P leg FAILED — coverage DEFECT" | `results/calibration_gate_v2_20260810/{CALGATE_V2_READOUT.md,.json,adjudicate_v2_readout_results.json}`; commit 64abd5f6 (campaign+readout), dbde71dc (sbatch; one-commit child of registered 065e7f58, import-path diff empty — disclosed D-5) | §4 thread 17 OPENED: venue transfer of the in-loop σ_z-dosed coverage defect to production — decider named + pre-registered (`results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md`, commit e77eecad; instrument 2ece8801, adversarially CONFIRMED; cluster array 6252702 RUNNING). Author itemized confirmation queued 2026-08-12 (§5 block 08-11) |

---

## 2. DO NOT RE-TRY — consolidated exoneration list

Everything in `CLAIM_2D_BIAS_20260730.md:191-204` **plus** the items below. Items marked ⚠ are
**absent from that list** and are therefore the live re-litigation risk.

**Already on the claim file's list** (do not re-open): catalogue dd_L/dz Jacobian · Fisher frame ·
p_det estimator choice · p_det inside/outside · h-prior sensitivity · `volume_trunc` · the z leg ·
the ln-M draw · realization plumbing · candidate-window membership · mass-kernel family ·
Option-A β_G/Σ_glob drift (= exact h⁻³ Jacobian, −26.80%) · HA as owner · HC zero-handling · HB.

⚠ **Not carried by the current claim file — re-litigation risk:**

1. ⚠ **`mass_trunc` / truncated-lognormal mass kernel as the 2D *driver*** — exonerated twice: pipeline A/B Δ2D **+0.0029 wrong sign** (#72) and the ratified-kernel 4-cell A/B **−1.8…−2.3 ln of +25.6…+29.1, MAP unmoved** (#89). The claim file's "mass-kernel family (bounded +0.002)" is the *same verdict measured differently* — do not re-derive it a third time.
2. ⚠ **Full Gray mixture as the compensation channel** — AMPLIFIES to +0.123 (#60).
3. ⚠ **`w_G = β_G/D` bookkeeping / membership-conditioned inverse as the fix** — refuted, tilt merely relocates (#61). *(Note: this is NOT the same as Gate C item 1, which asks whether `β_G` and `Σ_glob` are mutually consistent — see §3.)*
4. ⚠ **Hard support truncation / hard clamp in production** — misspecified under observed-z membership (#63); N-2d.
5. ⚠ **Tuning `w_pop`** — ≤+0.0004 at a 10% misspecification (#64).
6. ⚠ **Adding `p_det` inside the numerator alone** — refuted and it *breaks* calibrated controls (#66). Only the joint model-σ + p_det-inside pair works (#67).
7. ⚠ **Depth truncation (#30 option b)** — rails at every z_cut (#56).
8. ⚠ **The #29 zero-host fallback as the rail cause** — real bug, but fallback events are h-inert (#55).
9. ⚠ **Ω_m era mismatch** — −0.0006, wrong sign (#59). **Ω_m fiducial as a bug** — design choice (#53).
10. ⚠ **`L_comp`/`B_num` as a defective integral** — exonerated by self-consistency MC (#80), and again by the impostor harness where B_num is *the residual carrier but not a shown defect* (#87).
11. ⚠ **`volume_deconv` kernel h-dependence** — exactly h-invariant to 1e-15 (#75).
12. ⚠ **p_det anchor / first-bin asymptote escalation** — wrong layer, +12% lift moved MAP by zero (#17).
13. ⚠ **"Information starvation"** — OVERTURNED (#41/#52). Do not resurrect it as an explanation.
14. ⚠ **Spec-z-subset rescue** — refuted (#42).
15. ⚠ **Heliocentric/PV frame** — +0.15% net; #16 closed with 2D PV-*insensitive* (+0.0012) (#43/#44).
16. ⚠ **`galaxy.py` (1+z)³ σ_z, `TRUE_HUBBLE_CONSTANT`, CRB row-424 seam, p_det grid resolution, `allow_singular`, underflow** — all inert/non-causes (#25, #30, #10, #18, #7).
17. ⚠ **Numerator-only normalization cleans and the local same-kernel denominator** — rail UP to 0.870 (#37, #38); `D_sm` de-biases without a peak (#39).

**Standing scoping rule** (`BI@1fe428a^:209-218`): *negative* conclusions are **venue-scoped**.
`volume_trunc` (#70) and `mass_trunc` (#72) were both measured on the **same seed600 494-event
shallow subsample**; a shared venue idiosyncrasy would fool both. Do not cite either as universal.

**[AMBIG] Two provenance warnings on the era-3 (June-2026) results.**
(a) The commission's evidence-locker audit graded **0/5 of the project's own railing harnesses
faithful, 0 trustworthy, 2 artifacts** — and classified **`rung_I` (the source of rows #37–#39)
as an ARTIFACT with sign flips**: in its hands STANDARD → 0.60 and CONSISTENT_DENOM → **0.87**,
i.e. the labels are inverted relative to production (`synthesis/WF3a_DIGEST.md:3-20`). Treat the
`BRA` candidate table's rail directions as contested, not settled.
(b) `results/commission_20260701/synthesis/REPORT.md` **does not exist**; the report is
`DRAFT_REPORT.md`, whose own header says §7 (formal D2 coverage) was still being appended.

---

## 3. HISTORY vs CURRENT CLAIMS

### The single most important pre-emption
**#53's estimator pairing is cell A′ of an A/B that already ran on the deep seed1000 venue.**
Cell A′ = `absolute_marginal` + `volume_deconv` + gaussian(+G2d) mass kernel — exactly what #53 used —
measured **1D MAP 0.73 (gap −69.4 ln), 2D MAP 0.80 (+25.6 ln @0.80)** on an **unscattered mock
catalogue** (`mass_ab_20260727/MASS_KERNEL_AB_READOUT.md:23-30`). The three-way A/B gives the same
picture and its attribution: **the broadened (non-δ) host-z kernel, not the catalogue scatter, is
what puts 2D at ≈0.80 while 1D sits at truth** — δ-kernel share 86.7% of the total 2D movement
(`threeway_ab/THREEWAY_AB_READOUT.md:41-56`). #53 measures 2D 0.780–0.820, 1D 0.700–0.740.

Consequences: (a) **C6's confound is worse than stated but its worry is largely already answered** —
history predicts the pre-registered 2×2 **cell B** (unscattered catalogue, #53 estimator) will land at
**2D ≈ 0.80, 1D ≈ 0.73**, i.e. the "*estimator owns it*" branch of `PREREGISTRATION_2x2_cellB.md`.
(b) The 2D channel was **already formally designated OPEN** for real-data mode: **[RATIFY-M6]**
declares `absolute_marginal + volume_deconv + (M1)` a *CANDIDATE*, "necessary, not established
sufficient" (`docs/derivations/mass_marginal_2d_kernel.md:1-17, 690-706`). The +0.077 is the
predicted, already-owned open item, not a new discovery.

### Per-claim
| Claim | History says | Cite |
|---|---|---|
| **C1/C2** (class budgets, channel totals) | No conflict. Consistent with the deep-venue picture that the 2D excess is carried broadly, not by a tail (EXP-40: top-decile carries 25.5% of the host sum) | `exp40-mechanism.md:46` |
| **C3/C4** (dark class owns 84%; impostor rejection → completion fallback) | **PARTLY PRE-EMPTED and partly CONTRADICTED.** The *impostor* channel was measured and **EXONERATED** as a bias carrier at full power (bias −0.0006 at 93% impostors, +0.0024 at 97.6%); the residual is entirely `B_num` (#87). Separately, the completion leg's *up*-tilt is not an assumption — E1 measured `B_num/D` railing HIGH on a self-consistent ensemble (#80) — but E1 also showed that reading a fallback-only subset with `B_num/D` is a **subset-conditioning artifact**, exactly the trap C4's framing risks | `FULLPOWER_READOUT.md:93-102`, `E1_COMPLETION_BIAS.md:10-34` |
| **C5** (58% of in-cat hosts rail at 0.86) | **SUPPORTED in mechanism, but its magnitude has a contested history.** The shallow/large-σ_z/z regime was measured to bias HIGH (harness +0.030, #68) — then **re-attributed to a harness-only generative z≥0 clamp**, with production's catalogue shown *unclamped* (#69). **#53 is the first run where the catalogue redshifts are themselves realized with noise and clipped at `GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT` (`observed_realization.py:331-334`, counter `n_z_floor_clipped`).** ⇒ **The h1_zclamp exoneration may not transfer to #53. Read `n_z_floor_clipped` from the realization sidecars before anything else** — it is one number and it decides whether #69 applies | `h1_zclamp_20260713/FINDINGS.md:39-59` |
| **C7** (host-z numerator weight `w_pop` omits p_det and φ_cat) | **CONTRADICTED by a ratified derivation.** G2b **CONFIRMED** `w_pop = (dV_c/dz)/(1+z)` as "the unique weight consistent with the project's own rate model and with every selection integral", **exactly h-independent**, reducing to the point kernel as σ_z→0. Adding p_det would break that h-independence (binding gate 6, `PRODUCTION-KERNEL-FIX-SCOPING:170-180`). Also: the deconvolution **over-corrects** at σ_z/z ~ O(1) (#68/#62), i.e. the sign of C7's proposed fix is the *opposite* of the measured failure mode. Note too that "numerator-only" kernel changes are the exonerated class (#37, #70) | `G2b_host_z_volume_prior.md:413-436` |
| **C8** (2D MAP walks with the mass measure) | **NOT previously tested; genuinely new and it survives this session's own re-check.** History supplies the missing half: the dimensional mismatch is a 4D numerator leg added to a 3D numerator leg, not a denominator defect. It also independently reproduces the HA number (0.8492) from a different starting point, so **HA's exoneration stands** | `gate_b_20260730/README_C8.md` |

### Gate C alternatives
| Gate C item | History |
|---|---|
| **1. mixture legs disagree: `β_G = D − β_Ḡ` (model `f`) vs `Σ_glob` (catalogue sum)** | **ALREADY MEASURED, and it FAILED.** G1 summed the real catalogue: after removing the expected n_gal ∝ h³ factor a **−17.2% end-to-end residual** remains between the discrete catalogue content and the continuous completeness integral. G1 concluded "local modes are structurally immune" **because they never use `Σ_glob`** — but **`absolute_marginal` DOES** (`n̄_w = Σ_glob/β_G`), and §3.21 independently measured that identity violated by **33% in value and 0.39/h in log-slope**. ⇒ **this alternative is the best-evidenced open channel for #53's estimator, and it is why `generator_marginal` replaced `n̄_w` with `W_cat/V_f`.** The claim file's "loose thread" (w_G 0.0697 vs 0.0479, 45%) is the same object |
| **2. completion-leg calibration at comp_frac ≈ 0.07** | L-A measured +0.7…+5.4% HIGH at comp_frac 0.22–0.85 (#57); the deep floor was then decomposed into leak (#62) + noise-model (#67) + a ~+0.0005 remainder. **But #53's venue is the opposite regime** (comp_frac ≈ 0.05–0.12 → *less* completion-governed, not more), and the harness is 1D-only by every SUMMARY's own caveat — it has never covered the 2D residual. Full-power work also names `B_num` as the sole residual carrier (#87) |
| **3. `D(h)` not mass-marginalised (HA's other half)** | **The premise is wrong per this session's own C8 trace**: `D(h)`/`β_G`/`β_Ḡ` are *correctly* mass-dimensionless (an MFG selection probability). The imbalance is numerator-internal. Historically this is branch **(d2)** — "selection-side M scatter/truncation" — which is one of the two **named open owners** of the +23 ln 2D residual (#92). Its sibling (d1) was measured NULL (#90) and (g1)-as-suppression refuted (#91) |
| **4. HB residual: h-flat 68% window suppression** | Rides on item 3 / branch (d2). No historical measurement exists for the *h-flat* component; the h-*tilted* component is the refuted HB |

---

## 4. Open threads inherited (history left them open; still open today)

1. **The ≈+23 ln / MAP-0.80 2D HIGH residual under a broadened host-z kernel** — owners **(d2)** selection-side M scatter/truncation and **(g1)** mass-support clamp; explicitly campaign-gated (#92). *This is almost certainly the same object as #53's +0.077.*
2. **`B_num`'s own bias model** — sole carrier of the harness residual at full power; "remains open" (#87).
3. **Real-data host-z kernel** — the point/point pairing is generator-exact **only for the mock**; the photo-z kernel must be re-derived for real data before any real-data claim (`H0R:1647-1654`, redteam R3).
4. **seed600 shallow +0.0132 / +0.0138** — attribution REOPENED by #69; unowned since 2026-07-13.
5. **1D residual-after-Jacobian +1.667% (= +0.017 in h), 1D-only** (`project_optionA_jacobian_exonerated.md:25-28`).
6. **seed600 third-arm gate criteria 3–4 FAIL** with a "conditional adoption" override and an ordered follow-up on the z≤1.5 venue that was never run (#84) — **[AMBIG]** vs `RUNBOOK_NEXT_SESSION.md:113`.
7. **Redteam T-1: blind alternative-truth mock at sealed h_inj** — "the decisive anti-tuning test", ordered, **never run**.
8. **Second-order noise-model residual ~+0.0005** surviving the fully-consistent estimator (#67).
9. **`.planning/BIAS-INVESTIGATION-20260710.md` is deleted from the tree** yet is cited as plan-of-record by both 07-10 handoffs; **`FINDINGS_EXP40_20260725.md` does not exist** (dangling cite at `SYNTHESIS.md:67`).
10. **Commission tests D2 (formal P–P coverage) = PENDING and D7 (external gwcosmo/CHIMERA cross-check) = NOT-ATTEMPTED** in `commission_20260701/commission_history.jsonl`. Both were *substantially* discharged later by gate items **G4** (pp_coverage harness) and **G5a/G5b** (external-code inspection) — but the commission ledger itself was never closed out, so a grep of that file still reports them open. **[AMBIG]**
11. **`audit/normalization.md` findings 5(a)/5(b)** (sky peak-vs-sum asymmetry; global-vs-local `n_gal` cancellation) were **downgraded/contradicted by the commission's own adversarial verify** (`WF1_DIGEST.md:52-61`) while the file itself was never amended. **[AMBIG]** — do not cite `audit/normalization.md` findings 5–6 without that caveat.
12. **DECISIONS-20260712 "PROD: IMPLEMENT ALL"** (truncated-normal × volume prior + soft membership + latent coupling) was designed (`.planning/kernel-fix-20260719/design-{B,C}*.md`) and **never implemented** — the project went down the `absolute_marginal`→`generator_marginal` route instead. Design C predicts a testable alternative for C7: with a zero-scatter mock DGP, `volume_deconv`'s Eddington deconvolution is an **over-correction**.
13. **Is g_frac(h)'s h-slope correct physics or a missing normalisation?** (#91, CONFIRMED carrier, not yet adjudicated) — routes through `/physics-change`; possible convergence with D1 (same selection machinery feeds `B_num_wbh`). **UPDATE 08-05 (#92):** the closed-loop calibration test found the 2D g-machinery **MIXED/not-REFUTE** (Δ2 = +0.011 ± 0.004; production's +0.05–0.07 2D displacement is **NOT reproduced** in a universe following the estimator's own assumptions) — this now *supports* the "correct physics" reading over the "defect" reading, though CONFIRM was missed by 0.0013 and the question remains formally open pending the author's `/physics-change` ruling.
14. **1D production rail ownership — REOPENED (#93).** N-2's pre-registered counterfactual found the completion numerator's missing `S̄_φ(z;h)` selection factor is a **real, bounded, positive correction** (chord +23–25 nats/h, in the pre-registered band) but it does **not** un-rail the production 1D channel (MAP stays at 0.600, hard rail −1.1…−1.6 nats to the next grid point). The standing explanation (host photo-z, #36) is **not re-attributed**. A secondary finding from N-2's own read — the live z-quadrature of `S̄_φ` diverges from M1's point-GW-peak approximation (central-diff overshoot; sign coherence 0.71–0.73 vs the 0.90 bar) — is itself unresolved and flagged for follow-up. **The 1D rail's owner is still at large**, tracked as its own thread separate from the g_frac(h) derivation question in item 13.
15. **The catalogue leg's `f_k` completeness callable is pool-fed — discovered 08-05 (#94).** D1's N2 null (registered expectation: `L_cat_no_bh`/`L_cat_with_bh` carry no selection function) **FAILED** — under `volume_deconv`/`absolute_marginal`, the catalogue-leg host-z prior carries `f_k`, and `f_k` is built from the injection pool, the very object D1's three-arm re-weight substitutes. This falsifies a load-bearing prereg assumption and is a **new coupling**, distinct from and additional to the g_frac(h)/D1-convergence question (item 13, now closed as dead at machine precision — `g_frac` was bit-identical A1-vs-A2, #94 S2). Unowned; heads the next session's intake queue.
16. **Owner of M-2's matched 2D overlap residual — NEW TOP CANDIDATE THREAD (opened 08-07 by #96).** The residual is real (+0.02070 joint_r1 / +0.02225 iiib nats/event, cluster-robust p 0.0042/0.0050, low-h-preferring, ~8 nats class-scale if coherent) and #96 **excluded its leading candidate mechanism**: the Eq. (31) pairwise cross-term's composed 2D chord has the OPPOSITE sign at 5–6 orders below Y. Opening evidence for the hunt — the **composition-annihilation finding**: the raw catalogue-leg 2D chords DO carry the residual's low-h sign coherently (+2.507/+0.0116 nats; every joint_r1/iiib 2D per-pair composed chord one-signed), but the mixture composition (median per-pair factor ~1.5e-07–2.5e-06) crushes the magnitude and w_G's h-fall reverses the class-level sign — so whatever owns the residual either lives outside the likelihood-factorization coupling entirely, or reaches the posterior by a route the w_G-weighted mixture does not annihilate. Not likelihood-factorization coupling (per #96); owner at large.

> **CLOSED 2026-08-08 (row #97; §5 AUTHOR RULING 2026-08-08).** The author ratified
> the presented stage-5 formulation verbatim ("ratified"): DISSOLUTION in
> modified-H-c form — completion-leg-carried (D-1 LMDI), d_L-geometry/ball-density
> confounding-absorbable to ~2/3 (A2, smooth monotone d_L functional), remaining
> ~1/3 fixed-d_L excess density-coupled at joint_r1 but specification-fragile and
> not significant at iiib, retained as a disclosed bounded open object under
> qualifications q1 (H-e chance conditional on overlap-among-overlap
> exchangeability) and q2 (sharpened: not a claimed finding). No escalation,
> no `/physics-change` implicated. Thread 16 is closed by this ruling; see row #97
> and §5.

---

> **Thread 17 (opened 2026-08-11 by row #98): in-loop σ_z-dosed coverage defect — venue transfer to production.** The trustworthy v2 gate measured, in its multi-candidate ball venue, a σ_z-dosed uniform +≈σ_z MAP bias with collapsed (0%) coverage and delta-narrow posteriors, in BOTH channels, while the production-style 1D rail did NOT reproduce there (DS-6 MIXED). Standing account (photo-z starvation, rail shape) and this defect are candidates as COMPATIBLE co-mechanisms. Decisive measurement (named in the 08-11 §5 block, pending itemized author confirmation 08-12): the VENUE-TRANSFER READ — production-matched realism (real CRB event population; real per-event ball multiplicities from the frozeng census, K up to 245,364; z-decile-matched empirical GLADE σ_z incl. the spec-z tail) — pre-registered `results/venue_transfer_20260811/PREREGISTRATION_VENUE_TRANSFER.md` (e77eecad), branches TRANSFER-CONFIRMED (⇒ /physics-change intake on estimator photo-z handling, author-gated) / TRANSFER-REFUTED / MIXED / VENUE-CONFOUNDED. Campaign: cluster array 6252702.

## 5. AUTHOR RULINGS (2026-08-05) — append-only; no row or thread above is modified

Rulings by Jasper Seehofer, 2026-08-05, from the morning author queue in
`results/campaign51_20260728/RUNBOOK_NEXT_SESSION_8.md` §1. Queue items **1, 2, 4
accepted exactly as proposed there; items 3, 5, 6, 7, 8 remain OPEN** and are
deliberately untouched by this section.

### AUTHOR RULING (2026-08-05) — R-A: g_frac(h) h-slope is correct physics
*(RUNBOOK_NEXT_SESSION_8.md §1 item 1; annotates rows #91–#92 and §4 open thread 13.)*

**ACCEPTED**: "correct physics" for `g_frac`'s h-slope — the Gate-B-surviving,
closed-loop-supported reading (`.planning/derivation-gfrac-20260805/GFRAC_DERIVATION_PACKAGE.md`,
Gate B adjudication; row #92 closed-loop MIXED/not-REFUTE). The `/physics-change`
adjudication that §4 thread 13 said was pending is hereby made: the carrier is
legitimate spectral-siren physics, not a defect. **Explicitly NOT accepted:** the
refuted §7 re-attribution (Gate B claim 4, REFUTED) — the honest statement is the
amended package §7. §4 thread 13's open question is resolved by this ruling.

### AUTHOR RULING (2026-08-05) — R-B: gate (i) retired as measured evidence
*(RUNBOOK_NEXT_SESSION_8.md §1 item 2; annotates the N-1 finding — Gate B table
row N-1 in `GFRAC_DERIVATION_PACKAGE.md` and RUNBOOK §0 discovered fact N-1.)*

**ACCEPTED**: gate (i) is retired **as measured evidence**. The N-1 finding
(gate (i) near-vacuous as a measurement: the 2D catalogue leg is identically zero
for 81.5%/61.8% of events) stands, and the **algebraic completion-leg invariance
proof supersedes it** as the evidence of record. Gate (i) should no longer be
cited as an independent measured check.

### AUTHOR RULING (2026-08-05) — D1 disposition: bounded-null accepted
*(RUNBOOK_NEXT_SESSION_8.md §1 item 4; annotates row #94 and the verdict in
`../PREREGISTRATION_D1_SAND_REWEIGHT.md`, where the same ruling is appended.)*

**ACCEPTED**: the bounded-null (tilt route) verdict of row #94 — D1 does not
reach the core 2D-bias object via the tilt route; the D1→g_frac (C7) convergence
route is dead at machine precision. Unchanged: the simulation-side
`ParameterSpace.p0` bounds retirement remains its own future `/physics-change`
(not authorised here), and the 3135-event catalogue is still never re-scored
band-blind.

### GOVERNANCE NOTE (2026-08-05) — binding value statement
*(Author, verbatim; constrains all future adjudications recorded in this ledger.)*

> "our overarching goal is a scientifically sound project with novel insights and
> not to get rid of the bias by any means — scientific correctness and new
> insights are valued higher."

### AUTHOR RULING (2026-08-07) — Eq. (31) cross-term stage-5: conditional NEGLECT closure
*(Annotates row #96 and §4 open thread 16; closes the measurement mandate of row #95.)*

**ACCEPTED**: NEGLECT-WITH-NUMBER in all four venue × channel cells (row #96: T =
1.468160e-05 / 5.648753e-05 / 1.024615e-05 / 5.379542e-06 nats vs X = 2.78, adjudication
CONFIRMED bit-identical), **CONDITIONED on an explicit re-evaluation trigger register** —
the author's words: conclude "with clear triggers when this needs be re-evaluated… for
this assumption to break, at least X,Y,Z has to become apparent/realized". The register
of record is
`../crossterm_instrument/NEGLECT_TRIGGER_REGISTER.md` (same results tree): six named
triggers — (a) mixture-composition growth (w_G ≳ 0.3 / median per-pair factor ≳ 1e-05),
(b) any /physics-change to L_cat structure incl. the OPEN S̄_φ-inside-1D N-2 adoption
(queue item 3) → composition-arithmetic re-check, (c) ball z-window/radius enlargement
or localisation-model change (issue #53's 2σ direction is favorable and does NOT fire)
→ census re-run, (d) σ_z regime widening ≥ ~3× or catalogue replacement → full
re-measurement, (e) N_ev ≳ 3e+04 (nominal margin closure ≈ 3.5e+05 at today's per-pair
scale), (f) any change to the mixture-composed scoring convention (prereg flag (c) —
under raw scoring 3 of 4 cells already exceed X). **The NEGLECT stands unless at least
one named trigger fires.** The cross-term leg of H-1 (and H-3's cross-term leg, prereg
§1 item 3) exits LIVE by measurement; M-2's 2D overlap residual is UNOWNED and becomes
§4 thread 16.

### AUTHOR RULING (2026-08-08) — M-2 residual owner, stage 5: DISSOLUTION in modified-H-c form
*(Annotates row #97 and closes §4 open thread 16. Author, verbatim reply "ratified" to the
presented stage-5 formulation.)*

**RATIFIED**: the M-2 matched 2D overlap residual thread (ledger §4 thread 16) closes as
**DISSOLUTION in modified-H-c form**: the residual (+0.02225 iiib / +0.02070 joint_r1
nats/event) is real and component-coherent (H-e chance closed on the literal + majority
graph readings, CONDITIONAL on the overlap-among-overlap exchangeability model —
qualification **q1**), carried by the completion leg (T_Lcomp, D-1 LMDI, fraction
+1.28/+1.33; catalogue leg NULL; venue-stability mechanically explained by bit-identical
completion columns; 1D/2D dichotomy explained by w_G-offset cancellation + mass-conditioning
unmasking), and confounding-absorbable by the collinear d_L-geometry + ball-density bundle
traceable to C-4's 2σ d_L window: ~2/3 a smooth, strictly-monotone, verified d_L-functional
completion-leg response (A2: R² 0.88, chain ratios 0.666/0.653), the remaining ~1/3 (the
+0.0083 ± 0.0029 fixed-d_L excess) density-coupled at joint_r1 under every stability check
(instrument B DS-3: A_ρ +0.0069, cluster p 0.023; radius overlay p 0.0055) but
specification-fragile (verifier ATT-5: p crosses 0.0455 under tensor-deg3 D; excess softens
to ~2.1–2.5σ under flexible D) and NOT significant at iiib (A_ρ +0.0039, p 0.11) —
**recorded as a disclosed, bounded open object, NOT a claimed finding** (qualification
**q2**, sharpened). Instrument B fired pre-registered branch (iii) MIXED on this clean
venue split; the joint model chain reproduces in-band BOTH venues (ρ_J 0.82/0.97). The
fragile joint_r1 1D signal (p=0.0414) is **RETIRED**. H-b and H-d are refuted as owners
(wrong-sign carriers); H-a is carrier-only. **No escalation, no `/physics-change`
implicated, no production formula touched.**

Evidence commits: `e253e0c1` (stage 0-1), `c188f460` (A1/A2), `36fe7800` (prereg B),
`39d8a227` (instrument B). Key artifacts in
`../m2_residual_owner/`: `CLAIM_M2_RESIDUAL_OWNER_20260807.md`,
`STAGE1_READOUT_20260807.md` (+A1/A2 appendix),
`PREREGISTRATION_B_COMPLETION_WEIGHT_FUNCTIONAL.md`, `B_READOUT_20260808.md`, all
`*_results.json` + adjudications. §4 thread 16 is CLOSED by this ruling; see row #97.

---

### AUTHOR CONTINUATION (2026-08-11) — calgate-v2 acceptance-by-reference; itemized confirmation QUEUED for 2026-08-12

**Attribution, stated precisely.** After the v2 gate report (verdict, DS-8 targets, disclosures, and an explicit
numbered recommendation list) was presented in-session, the author replied verbatim: **"please continue as
recommended by you"** (2026-08-11). Later the same session, on the escalation question, the author ruled verbatim:
**"if it escalates to the physics change. please mark it that this need to be ratified tomorrow by me but already go
ahead. worst case is that we need to revert it, but that is fine."** The itemization below is therefore
**orchestrator-derived from the recommendations the author endorsed by reference** — it is NOT a verbatim author
dictation — and every item is **QUEUED for the author's explicit itemized confirmation on 2026-08-12**:

- (i) v2 deviation register D1–D8 accepted, incl. the cluster venue switch and the dbde71dc/065e7f58 child-commit
  disclosure (import-path diff empty);
- (ii) DS-8 confirmations treated as quotable measured properties (T1 starvation rail; T2 σ_z-dosed collapse; T3
  B0-on-truth);
- (iii) the in-loop σ_z-dosed coverage defect adopted as a named owner-CANDIDATE thread (17) alongside — not
  replacing — the photo-z-starvation account;
- (iv) clause-(b) "one measurement that decides" = the venue-transfer read (as recommended in-session; now
  pre-registered e77eecad, running as array 6252702);
- (v) DS-7 form call remains OPEN (report-only);
- (vi) paper #47 hold reason described going forward as "P–P leg FAILED — coverage DEFECT" rather than "leg missing".

**Overnight escalation protocol (author-ruled, same day):** if the venue-transfer read fires TRANSFER-CONFIRMED,
the /physics-change gate package is executed overnight IN FULL on a held `physics/` branch, marked PENDING AUTHOR
RATIFICATION (2026-08-12); production `main` is not modified; morning act = merge-or-revert.

---

### Row #99 — thread 17 VENUE-TRANSFER: campaign complete, TRANSFER-CONFIRMED, AUTHOR-RATIFIED 2026-08-13

**Measurement.** Venue-transfer campaign (prereg `e77eecad`, instrument `2ece8801`, readout commit
`d45fbf15`): 49 chunks / 1,400 seeds over cluster arrays 6252702 → 6253922 → 6259842 (final wave
22/22 COMPLETED, zero FAILED/TIMEOUT). Branch fired by the registered tree: **TRANSFER-CONFIRMED**.
VENUE-CONFOUNDED did not fire (0 of 9 trigger-set members). Independent adversarial adjudication:
**CONFIRMED**, every scored statistic reproduced from the raw 41-point `ln_post` vectors with
independent implementations to ≤ 5.33e-15.

**Decision cell** T-c(0.730), N = 400, 1D (registered headline, VT-D6): HPD 50/68/90 coverage
0.000/0.000/0.000 (band at N=400: 0.870–0.930 at 90 %); PIT–KS D = 1.000 (PASS ≤ 0.0679) —
saturated, not marginal; MAP bias **+0.037237 ± 0.000230** (DEFECT edge 0.030); σ̄_pairs = 0.041775 ⇒
**R_dose = 0.891** (band [0.75, 1.25]); rails 0.000/0.000 (band ≤ 0.02) — RAIL-EMERGENT did not fire;
post_sd median 0.004376 ⇒ the estimator is displaced by **8.5×** its own claimed width. 2D agrees
(bias +0.039713, R_dose 0.951). Both wings COLLAPSE-REPRODUCED ⇒ truth-uniformity leg holds.
**DS-VT5 ladder killing axis: NONE** (v2 +0.0353 → T-a +0.0349 → T-b +0.0359 → T-c +0.0372).
T-0 anchor (σ_z = 0): all 200 seeds argmax exactly on truth, rails 0 — the apparatus is unbiased.

**Author ratification, attribution-precise.** The readout, the adversarial discrepancy list, and a
three-decision table were presented as a reviewable artifact (per the CLAUDE.md "Proposing decisions"
rule). The author replied verbatim: **"It is ratified."** and **"I want to open the physics change"**
(2026-08-13). The itemization below is **orchestrator-derived** from the bundle that was on the page
when those words were written — it is NOT a verbatim author dictation:

- (i) branch **TRANSFER-CONFIRMED** accepted as the registered result of the campaign;
- (ii) the three §11 operational deviation notes (runtime blowout + resubmission; V-T5 compliance-order;
  contention resubmission 6259842) ratified;
- (iii) the two deviations newly surfaced by the adjudicator ratified: **the prereg §5 pre-campaign
  smoke was never run** (its abort-(a) input was substituted post hoc from array 6252702) and
  **VT-D0(iv) was violated** (§11 was empty at instrument-commit time; first content at `e93f3068`,
  after the first array had already run);
- (iv) escalation **opened**: `/physics-change` intake on the estimator's photo-z handling — author-gated
  and now author-opened, see row #100 when the gate package lands.

**Standing disclosures carried forward** (non-branch-impacting, from the adjudication): contended-node
per-seed CPU crossed the registered 8.66 CPU-h abort-(a) trip point in 11 of 40 heavy chunks (registered
consequence would have been to *shrink* the campaign — running at full N was the conservative direction);
V-T5 certified the core at `chunk_pairs=0` while every campaign chunk ran `chunk_pairs=16384` (ULP-scale,
immaterial against ln_post separations of order unity); V-T2 exercised n_events_cap=40 on the dev box, not
the 982-event chunk geometry; the package/repo rename (`227e7a32`) postdates the campaign, so V-T4's
clean-rule wording names the old import path.

**W1 (reserved rate-weights arm): NOT RUN — author-dropped 2026-08-13** as a standalone confirmation arm;
its question (equal-weight vs rate-weighted candidate prior) is folded into the mechanism-isolation study
that opens the physics-change thread. Seeds +46000…+46399 remain reserved and unconsumed. O2
(`volume_deconv`) remains NOT-EVALUABLE and reserved (+47000…+47399).

---

### Row #100 — thread 17 mechanism isolation + Amendment A1: V-M1 null re-measured at N=100, A1-PASS

**Mechanism-isolation register, before any instrument run.** Six-candidate register and four L0
closures: **M3 refuted** (implied shift +6.0e-7 vs the observed +0.0372, short by 6.2e4 and wrong
dose trend); **M4 refuted** (the "missing" term is identically 1 and deleting α outright leaves the
σ_z keying intact at +0.0165); **M1 refuted as sole mechanism on SIGN** (predicting H0 LOW),
retained as a fitted negative quadratic b ≈ −5.29 with the linear driver a ≈ +1.15, the quantity
every candidate must supply; **M5-as-stated refuted on attribution** (76 % of the bias survives an
unscattered population); **M2 refuted by the T-0 anchor**. The split-dose arms and the
NON-ADDITIVE result (MEH +0.004000 + MEI +0.000000 vs MN0 +0.034667).

**The V-M1 false-fail and its resolution.** The ±0.002 window was ASSERTED not derived, carried a
~21 % false-fail rate under an exact null, and was settled ON DATA at N = 100 rather than widened.
**A1-PASS**: MN0X 1D mean bias +0.037250 ± 0.000494 vs reference 0.037237, |Δ| = 0.000013, 153.8×
inside the unchanged ±0.002 window. A1-DET bit-identical across the `e83ed0b9 → 3aedbe55`
cross-commit refactor, 15/15 seeds, 44 fields, max relative deviation 0.0. Adversarially verified
CONFIRMED across all 425 seeds. Does NOT retroactively pass the N = 15 MN0 result — that remains
FAILED on the record. **The registered-defect precedent this sets**: a design fault in a
pre-registered window is recorded rather than quietly widened, exactly as DS-D3's one-sided
threshold defect is recorded in row #101.

Evidence commits: `73141160` (prereg, A1 amendment + 2D scan, author-ratified), `3aedbe55`
(instrument), `5b0bd17a` (data retrieved). Key artifacts in `../../../mechanism_study_20260813/`:
`PREREGISTRATION_MECHANISM_ISOLATION.md`, `AMENDMENT_A1_VM1_NULL_AT_N100.md` (+ verdict block,
this row), `ARMS.md`, `M1_missing_volume_prior.md`, `M3_truncation_window.md`,
`M4_alpha_sigma_blindness.md`, `M5_smeared_candidate_prior.md`, `A1_READOUT.md`/`.json`,
`score_a1.py`, `MN0X_h0p730_results_seeds0_100.json` + `MN0`/`MEH`/`MEI` arm JSONs.

---

### Row #101 — thread 17 2D dose scan: BRANCH 2 fired, meaning barred, gate × amplifier

**Branch determination.** `PREREGISTRATION_2D_DOSE_SCAN.md`'s registered tree, checked in the
registered order, fires **BRANCH 2 — INTERACTION-BILINEAR**: DS-D2 NON-ADDITIVE at S33
(D = +0.033667, 23.4σ) and DS-D3 SHAPE-INTERACTION at S23 (b = +0.023650 ≥ 0.01150132, +28.2
realized SE above the boundary). Branch 1 (SCAN-CONFOUNDED) did not fire, 0 of 4 members, including
Amendment A1 returning A1-PASS (row #100). Branches 3/4/5 not reached.

**Meaning barred.** Branch 2's pre-stated meaning — a genuine strictly-bilinear product-form
interaction D = I·f_h·f_i — is refuted by the scan's own registered statistics: b(S23) sits
+10.33σ above H-INT's own point prediction using the registered SE (+14.64σ realized); bilinear
residuals positive at all nine evaluable cells, >3σ at S22/S31/S23. H-THRESH independently refuted
at 17.96σ/50.18σ. **Both registered shapes are wrong; branch 2's meaning clause is barred from
being quoted.**

**Registered defect recorded, not repaired.** DS-D3 is a one-sided threshold with no upper edge, so
SHAPE-INTERACTION fires for any sufficiently large value, including values that refute the
hypothesis it names. Not adjusted (§4.7 anti-tuning); recorded as a design fault for a future
amendment.

**Positive finding — gate × amplifier.** Host is an absolute gate: f_host=0 row exactly
+0.000000 at every impostor dose, 60/60 seeds, degenerate posterior. Impostor sea is a graded
amplifier: removing it leaves +0.0047…+0.0060, ~15 % of the effect. Supports half of branch 2's
pre-stated consequence, refutes the other half.

**Collateral confirmed:** the parity argument sharpened; the M5 reweighting closure remains
binding; §5.4's sub-prediction confirmed (f_imp=0 column small and positive at all four host doses,
never negative — the M1 negative quadratic term remains the pre-named carrier of `pp_coverage`'s
sign flip). **Not supported**: the f_host=1 flat-middle claim (1.17σ, UNRESOLVED) and the
f_host=0.5 dip (2.93σ, MARGINAL). No functional form fitted; no repair proposed.

**Adversarially verified CONFIRMED** — all DS-D1..DS-D6 scores, corner and dosing checks, and the
branch determination reproduced independently; six disclosures recorded (V-D5 header over-claim,
dip-call convention-fragility, an internal §4.6/§4.7 contradiction, undisclosed SE choice in the
H-INT distance, two un-re-executed checks, and a naming-only field deviation) — none change the
branch call.

**Author ratification, attribution-precise — record the correction loop.** The author approved the
readout work with the verbatim words **"all approved as you recommend"**, covering the [DO] items
only. On the branch question the author FIRST ruled **"as you recommended"** for branch 5, on the
basis of an orchestrator framing subsequently CORRECTED by adversarial verification (the framing
had called branch 5's condition "factually satisfied but unreachable"; branch 5 is in fact the
residual "anything else" class, and the tree genuinely fires branch 2). **The branch-5 ruling is
SUPERSEDED.** The orchestrator brought the corrected framing back to the author rather than letting
the superseded ruling stand, and the author then ruled verbatim **"a"** against a presented
two-option table — the final ruling: record branch 2 as fired, DS-D3 defect logged, branch 2's
meaning barred. Every itemisation above is orchestrator-derived, not author dictation. This
correction loop is the reason the approval-scope convention now in `CLAUDE.md` (commit `804b4c5d`)
exists: **[DO]/[RULE]/[STANDING]** tags, with the binding default that an approval never propagates
to a decision whose inputs did not exist when it was given.

Evidence commits: `73141160` (prereg, author-ratified), `3aedbe55` (instrument), `5b0bd17a` (data
retrieved), `804b4c5d` (CLAUDE.md approval-scope tags, unpushed prior to this row). Key artifacts
in `../../../mechanism_study_20260813/`: `PREREGISTRATION_2D_DOSE_SCAN.md` (+ verdict block, this
row), `SCAN_READOUT.md`, `score_2d_scan.py`, `score_2d_scan_output.json`, `S00`…`S33` arm JSONs,
`adjudicate_mechanism_study.py` + output JSON.

**Addendum to row #100 (2026-08-14, overnight): disclosure D-A1-2 CLOSED.** V-M5 — the no-drift
anchor, re-registered earlier the same day as a values golden at rtol ≤ 1e-12 with both channels'
MAPs exactly equal — has now been **re-executed at HEAD `94c0480a`** and **PASSES**. Artifact:
`results/mechanism_study_20260813/VM5_GOLDEN_20260814.md` / `.json` / `verify_vm5_golden.py`
(commit `38465df8`). Max relative deviation **1.6135e-14**, two orders of magnitude inside the
1e-12 ceiling; the 1D channel is bit-identical on all three registered v2 seeds
(20286808–20286810); all four MAP fields exactly equal the committed `B2_h0p730_results.json`.
Deviation is confined to `ln_post_2d`, `mean_2d`, `pit_2d` and `M_source_median` — exactly the
2D/mass-dependent fields the prereg predicted would move under the ratified Route 1 adaptive
Gauss–Hermite change. Environment recorded with the artifact (numpy 2.4.3; scipy-openblas 0.3.31
DYNAMIC_ARCH/Haswell; OMP/OPENBLAS thread vars unset), because the original bit-identity failure
traced to BLAS/SIMD dispatch and a future re-run needs the comparison basis.

A1-DET certified a *different* check (cross-commit determinism of the null arm); this is the
registered V-M5 artifact itself, so the gap is closed by evidence of the registered kind rather
than by a proxy. **No registered STOP fired.** D-A1-3 (MEH/MEI at the pre-refactor commit) and the
scan's four disclosures remain live.

Also filed this session and cited here for provenance:
`results/mechanism_study_20260813/PHYSICS_CHANGE_INTAKE_DOSSIER.md` (commit `ee5815f9`) — the
`/physics-change` intake package. It fills the one gate slot fillable today (the OLD formula,
exactly, with per-symbol provenance) and specifies **16 constraints (C1–C16)** any future candidate
must satisfy, each with its establishing measurement and strength. **The new-formula slot is empty
and stays empty until the author fills it**; the dossier names no candidate and proposes no repair,
per the scan readout's §6 item 7 bar.

**Addendum 2 to row #100 (2026-08-14, overnight): the parent readout is FILED and PRESENTED —
NOT RULED.** `results/mechanism_study_20260813/MECHANISM_ISOLATION_READOUT.md` (commit `f0817dfe`).
Recorded here so the measurement exists on the record; the branch call is the author's and is
outstanding.

**Validity: STUDY-CONFOUNDED fires on zero legs.** V-M1 cleared via A1 (|Δ| = 0.000013, independently
recomputed as 745 ticks × 5e-5); V-M2 via 11/11 AR unit tests at HEAD (AR-3 is not checkable in-data,
the arms using disjoint seed blocks — disclosed); V-M3/V-M4 pass on all three arms; V-M5 closed by
the `38465df8` golden; aborts (a)–(d) none fired. Every statistic recomputed from raw `ln_post`
vectors agrees with the stored scalars at **0.0** relative deviation across all four arms and both
channels.

**Branch presented: 2 — SINGLE-OWNER**, sole TERM-OWNS arm MEI, identically in both channels
(|b| = 0.000000 ≤ 0.010 and HPD90 = 1.000 ≥ 0.60). **With the branch's meaning clause disclosed as
having no referent:** parent §2 and `ARMS.md` both register E1 as a *zero-estimator-change,
generator-side* arm ("the estimator is byte-identical across N-0, E1-host and E1-imp"), so branch 2's
registered meaning — "that term is the identified mechanism; the /physics-change package is written
against it" — names a conclusion no arm in this design could support. This is the **second**
independently drafted tree in the same thread to fire a branch whose meaning the data does not
sustain (the first being the scan's one-sided DS-D3). Proposed remedy drafted, not adopted:
research-cycle amendment **A8**, `docs/RESEARCH_CYCLE.md` + `docs/gates/BRANCH_REFERENT_FAULT_20260814.md`
(commit `cd9c610e`), PENDING AUTHOR APPROVAL.

**DS-M5: M5′ NOT CONFIRMED; the registered refutation clause fires**, returning the study to the
**M2′ arm, which has never been run.** E1-imp required ≥ 0.030 and measured **+0.000000** — the whole
requirement short, and structurally unreachable: 6 grid steps against a median **2,299-nat** margin at
the true grid point, 15/15 seeds. E1-host required ≤ 0.012 and measured +0.004000. Non-additivity
**+0.030667 ± 0.00166667 = 18.40σ** against MN0, **45.67σ** against MN0X; the split recovers **11.5 %**
of the null, so 88.5 % exists only when both populations are dosed. What breaks is M5′'s *attribution*,
not its algebra — its K-saturation account (host pinning ≈ 1/K) is falsified at production K̄ ≈ 1,216,
where one exact host annihilates the bias against 1,192,721 smeared impostors.

**Jointly with the scan, and stated plainly: no estimator TERM is established.** Both documents vary a
generator-side dose, so the parent's own title question — which term produces the displacement — is
answered by an input condition and a shape (gate × amplifier), not by a term. M2′ remains the
register's only unrun candidate and the new-formula slot stays empty (see
`PHYSICS_CHANGE_INTAKE_DOSSIER.md`, C1–C16).

**Two items flagged for the author.** (i) **Abort (d) is the study's closest call**: the toy predicted
+0.0247 where the instrument measured exactly 0.000000 — a 100 % magnitude disagreement — but zero has
no *sign*, so the literal registered wording does not fire. Were it deemed to fire, the study STOPs and
every L0 closure reopens. (ii) The presented branch **rests on A1-PASS and the V-M5 golden, both of
which are themselves presented-not-ruled.**

**Orchestrator error corrected on the record:** MN0's 2D value was earlier reported as sitting "on the
campaign value" by comparing it against the *1D* reference. Against its own reference it is +0.037000
vs +0.039713 = **0.002713 — OUTSIDE the ±0.002 window**, so MN0 at N=15 missed on **both** channels,
marginally worse on 2D. No verdict depends on it (V-M1/A1 are 1D, and MN0X is inside on both at
0.000013 / 0.000037), and the conclusion is if anything strengthened — missing on both channels is more
consistent with an under-powered window than with a 1D-specific fluctuation — but the corroborating
evidence originally cited was invalid and is withdrawn.

---

## Row #102 — 2026-08-14 — Independent review adopted; author ruling on the mechanism-isolation bundle

**Trigger:** author instruction 2026-08-14 (an independent session assesses the completed cycle before its recommendations are implemented). Mechanism: `/commission --research` (run `wf_6def92de-d96`, 27 agents), chair-adjudicated in `results/commission_research_20260814/REPORT.md`, with two findings chair-verified by hand (the PREREG_PATH provenance defect; the M5-toy production-K re-execution, independently re-driven: K=50 +0.0317±0.0007, K=1216 +0.0339±0.0006 vs the instrument's exact 0.000000 — divergence GROWS with K).

**Author's verbatim ruling (2026-08-14):** "thanks and please go ahead as you recommended."
(Full message continues: "Whenever we have evidenced feedback for the research cycle please so we can improve it while we use it. Dont forget that you are the orchestrator and use tiering for workflows and subagents.")

**Orchestrator-derived itemization of what that grants** (the referenced recommendation set is the chair's adjudication in `results/commission_research_20260814/REPORT.md`; every item's inputs existed when the ruling was given):

1. **[RULE] Abort (d):** deemed **met IN SUBSTANCE** (the registered letter — a sign disagreement — still does not fire on a zero). Consequence, scoped by the commission's dependence audit: every **toy-dependent M5/W1 sub-closure** (M5 note rows B/C/D/H/I/P/Q/R and the W1 toy leg) is downgraded to **NOT ESTABLISHED**; **M1 and M4 stand** (demonstrably toy-independent: analytic + committed pp_coverage artifacts; exactness identity + α-deletion on stored campaign posteriors); **M3 stands on its analytic core** (≥10³ shortfall with zero toy input), its note downgraded to *plausible pending committed artifact*. The study **continues** — this is a scoped reopening, not a STOP.
2. **[RULE] Parent branch 2:** recorded as **PREMATURE ADJUDICATION** — the count-based branch was adjudicated over an incomplete registered arm set (registered estimator-side arm A-M2′ unrun). Supersedes both previously tabled readings ((a) fired-meaning-barred and (b) NO-OWNER). No term is named. Resolved only by running A-M2′ or withdrawing it by a further [RULE].
3. **[RULE] V-M1 branch-1 disjunct:** discharged to MN0X by registered amendment (prospective; MN0's N=15 FAILED status on the record is unchanged).
4. **[RULE] S00 / DS-D4:** the float-epsilon reading is adopted (b_S00 = 2.2e-16 ≡ 0 at grid precision; SCAN-CONFOUNDED did not fire); DS-D4 is restated as a **bound** (|refined residual| ≲ 1e-4), not annihilation — "exact zero" is a grid statistic.
5. **[STANDING] Amendment A8 adopted AS REVISED** (see `docs/RESEARCH_CYCLE.md` row A8): Instance-2 predicate corrected; two-sidedness check kept (BLOCKING); **execution-completeness check added (BLOCKING): no count-based branch may be adjudicated while a registered arm capable of changing the count is unrun**; band-derivation disclosure NON-BLOCKING. Scope: all future pre-registrations in this thread and its successors; lapses only on author revocation.
6. **[DO] A-M2′** is authorized, together with the registration of two additional candidate classes surfaced by the commission from committed data: **M6** (σ_z-blind aggregate log-posterior tilt × dose-controlled curvature composite) and **M7** (host/impostor ball-window inclusion asymmetry, named in the intake dossier's parity text but never assigned an M-ID).
7. **[DO] Mechanical repairs:** PREREG_PATH parameterization + a correction artifact for the 20 mis-stamped result JSONs; the recovered M5 toy committed with provenance; a pinned-inputs checksum manifest; errata/addenda to the registered documents (this row's companion appends).

**Attribution:** itemization above is orchestrator-derived from the author's blanket grant; the author's own words are only the quoted sentences.

---

## Row #103 — 2026-08-15 — Stage-2 readout ratified (A-M2′ TERM-PARTIAL; DS-N1 PASS)

**Author's verbatim ruling (2026-08-15):** "Readout 3 "rules" and "do" are approved, please go ahead but keep in mind you are the orchestrator and you should use subagents and workflows but think about which models and efforts to use."

**Orchestrator-derived itemization** (referencing `results/mechanism_study_20260813/STAGE2_READOUT.md` §5; all inputs existed at ruling time):
1. **[RULE]** Branch **3 — M2′-PARTIAL** is the branch of record; the DS-N1 PASS (15/15 paired seeds MAP-index-identical to MN0X, floor-aware integer shift law exact, m = 982 at every MAP) and the stage-2 readout are **verdicts of record**. Measured: restoring the z-integral measure/Jacobian removes **Δb(1D) = −0.018050 ± 0.000895** (48.5 % of b_ref = +0.037250), leaving +0.019200 ± 0.000746 with coverage 0/25 — M2′ **contributes but does not own**. Per registration: **no repair is licensed from a partial read**; the `/physics-change` new-formula slot stays empty.
2. **[RULE]** The 2D weak-expectation miss (+0.021400 vs window upper edge +0.021, non-branch-carrying) is **accepted as recorded**.
3. **[DO]** The M6-revision L0 work (formalize the dose-dependent tilt residual from committed data) and the M7-L0 derivation are authorized. Both stage-2 L1 slots are spent; **any further instrument arm requires a fresh registration.**

Provenance: prereg `092b121b` · scorer frozen pre-data `191b0db7` · data + readout `e49f7570` · job 6315312/6315313 (AM2P 1:21:25, ANULL 0:42:16, both 0:0).

---

## Row #104 — 2026-08-15 — L0 pair + verifier corrections ratified as the candidate-register update of record

**Author's verbatim ruling (2026-08-15):** "ratified"

**Orchestrator-derived itemization** (the referenced update is the M6R/M7 L0 pair with the adversarial-verification addenda, commits `b34ad9dd`, `99579a8a`, `0dc21651`, `0a3d940e`; all inputs existed at ruling time):
1. **[RULE — granted]** Candidate-register state of record: **M2′ measured PARTIAL and on-prediction** (J-tilt −1132.9 ± 36.0 vs corrected prediction ≈ −1090…−1130 including the ln D′ re-weighting term, +212.3 ± 36 measured vs +254 predicted); **M6′ PROPOSED** with corrected kill tests (KT-M6′-2/3 relabeled one-sided) and **T_res genuinely UNLOCATED** (~⅔ of the ±760 nats/h dose swing; the M1-quadratic account REFUTED — wrong sign at f_i = 0.25/0.5, ~16× overshoot at f_i = 1.0, inverted shape); **M7 CLOSED at L0** (−3.79e-4 ± 1.65e-4 under the production-curvature conversion, 2.6× inside the registered band); the 0.749 ± 0.046 tilt×curvature closure established within-grid.
2. Standing bars unchanged: no repair from a partial read; `/physics-change` new-formula slot EMPTY; both stage-2 L1 slots spent — any further instrument arm requires a fresh registration.
3. Next step (orchestrator-proposed at ruling time, drafting authorized by the ratification context, registration NOT yet authorized): a stage-3 preregistration PROPOSAL targeting T_res and the score-balance/overconfidence question (measured post-J tilt T(AM2P) = 1492 ± 31 ≈ the α tilt +1393.6, which is correct physics — an uncancelled correct-term tilt at truth indicates the defect may live in the curvature/score balance), preceded by a Stage-L literature sweep (R0 ring minimum).

---

## Row #105 — 2026-08-15 — Stage-3 proposal items 1–2 approved

**Author's verbatim ruling (2026-08-15):** "approved"

**Orchestrator-derived itemization** (referencing `results/mechanism_study_20260813/PROPOSAL_STAGE3_20260815.md` §3, commit `c10fddbc`):
1. **[DO — granted]** The four L0 items: L0-REN-A (analytic derivation of the unrenormalized-truncated-kernel tilt), L0-REN-B (A/B renormalization toy), L0-SB (sandwich/score-balance diagnostic on stored posteriors), L0-LIT (full-text reads of the two UNCHECKED literature rows). Committed data, toys, and reading only.
2. **[DO — granted]** Drafting (NOT registering) the A-REN registration in parallel.
3. Items 3–4 of the table (arm registration/run; `/physics-change` gate timing) remain **deferred to the author by design** — they return as fresh [RULE]/[DO] with the L0 evidence.

---

## Row #106 — 2026-08-15 — Stage-3 L0 synthesis adopted; A-JREN registration and run authorized

**Author's verbatim ruling (2026-08-15):** "approved, please go ahead"

**Orchestrator-derived itemization** (referencing `results/mechanism_study_20260813/L0_SYNTHESIS_STAGE3_20260815.md` §4, commit `9e7ce3ba`, with the verifier amendments `f8cf27a2`):
1. **[RULE — granted]** The stage-3 L0 synthesis is the wave's record: displacement law bias = T/Ā confirmed parameter-free (1.15 ± 0.13, 16 distinct cells, zero counterexamples); posterior width correctly calibrated to local curvature (the 8.5× overconfidence is displacement over correct width, not a second anomaly); T's ledger = α +1393.6 + measured missing-J +1133 ± 36 + dose-decaying unlocated residual; H-REN real-but-not-owner (R1 LIVE order 1e-3 sign +, R2 WRONG-SHAPE, R3 BUDGET-TENSION); non-additivity of ablations live; Gray 2020/2023 full-text: no event-term Jacobian in published practice, truncation escape-clause conditional not obviously met by a per-candidate window.
2. **[DO — granted]** Fill the draft registration's bands from the L0 wave, REGISTER, and RUN: **A-JREN first** (its registered trigger R3 = BUDGET-TENSION has fired; seeds +54100…+54124, N = 25), **A-REN converted to conditional** (runs only if the joint result leaves single-term attribution needed — a post-readout author [RULE]). ~25–50 CPU-h, new-stage L1 ≤ 2 budget. One xhigh pre-registration verifier gates the registering commit.
3. **[RULE — deferred by default]** The Gray-convention finding's entry into the paper's scope: no recommendation was attached; recorded as deferred until after A-JREN's readout, author may override at any time.

---

## Row #107 — 2026-08-15 — Stage-3 readout ratified; A-REN withdrawn; stage-4 authorized

**Author's verbatim ruling (2026-08-15):** "all approved, please continue"

**Orchestrator-derived itemization** (referencing `results/mechanism_study_20260813/STAGE3_READOUT.md` §4, commit `14b0d110`):
1. **[RULE — granted]** The stage-3 readout and the branch-5 record are ratified: A-JREN TERM-PARTIAL both channels at +0.017800 ± 0.000712 (1D); **additivity of the located repairs confirmed** (0.6σ from the additive prediction; the non-commuting-ablations reading is dead); **coverage NOT restored** (joint repair necessary, not sufficient); bias/post_sd 8.49 → 3.00; the 2D-only sub-additive +0.0027 (≈3.8σ) recorded as the new lead. **T_res is promoted to a first-class target.**
2. **[RULE — granted] A-REN is WITHDRAWN** (registered conditional arm, seeds +54000…+54024 released back to reserved-unconsumed status; its single-term effect is confirmed inside the joint arm by additivity). The execution-completeness clause is thereby discharged for stage 3.
3. **[DO — granted]** Stage-4: the L0-first T_res hunt under the readout's §3 constraint set — proposal to be drafted as a reviewable artifact (`PROPOSAL_STAGE4_*`), decisions therein returning to the author.

---

## Row #108 — 2026-08-15 — Stage-4 items 1–2 approved

**Author's verbatim ruling (2026-08-15):** "approved"

**Orchestrator-derived itemization** (referencing `results/mechanism_study_20260813/PROPOSAL_STAGE4_20260815.md` §3, commit `d75b4c99`):
1. **[DO — granted]** L4-DER (derive the correct estimator p(data|h) from the generator code and diff term-by-term against the coded estimator), L4-T1 (measure T(AJREN) from committed ln_post; composition of the remaining tilt vs the α-tilt +1393.6), L4-T2 (validate every diff term against the committed constraint set incl. the T_res dose curve and the 2D-only sub-additivity). Derivation + committed data only; no instrument time.
2. **[DO — granted]** Drafting (NOT registering) the A-FULL registration in parallel.
3. Items 3–4 (A-FULL registration/run; whether the validated derivation opens the `/physics-change` new-formula slot) remain **deferred to the author by design**.

---

## Row #109 — 2026-08-15 — L4-DER Part 2 ratified (as amended); A-FULL draft to proceed

**Author's verbatim ruling (2026-08-15):** "all approved"

**Orchestrator-derived itemization** (referencing `results/mechanism_study_20260813/L4_DER_PART2_20260815.md` §4 **as amended by its verifier addendum A1/A2**, commits `9f2e6c1a` + `8de28637`):
1. **[RULE — granted]** The Part-2 account is ratified: the coded tilt = α (+1400.6 numeric) + **GW z-mass growth +1059.6** (G_e = (1/h)(1 − D·D″/D′²); identity ΣG = N/h − Σx/h; retro-explains A-M2′ at 98.7% mass-kill) + exponent-scale (+175.8) + window motion (−31.1) + leftover (drift + interactions; +867/+344/+39 across f_i = 0.25/0.5/1.0), each measured by exact single-switch A/B on the bit-validated mirror; **T_res ≡ the leftover (drift + interactions)** with the §2 drift formula's direct evaluation registered as the next targeted recompute (amendment A2's hedge is part of the ratified record).
2. **[RULE — granted]** The Part-1 D1–D6 ledger AS COMPOSED is superseded (isolated-term composition invalid inside the log-integral; the T2 audit's D3 sign-convention erratum is of record — D3 exact-weighted −176 vs isolated −342, same sign ~2× attenuated, NOT a sign flip; D4-as-M1-quadratic already refuted; window motion −31 nats/h, 1D local tilt at truth only).
3. **[DO — granted]** Compose the A-FULL candidate estimator draft (correct d_obs-density form: density prefactor + Jacobian measure + p_pop numerator + renormalized kernel; predicted tilt ≈ 0) as a reviewable artifact with a fresh xhigh verifier. **Registration and running remain separate author gates (A8-v2), not granted here.**
4. **[RULE — granted, branch reading orchestrator-derived]** Decision 4 was an either/or (residuals carried vs priced-first); "all approved" is read as the branch **coherent with the granted item 3**: the +39 full-dose leftover and the 2D-only +129 excess are **carried as stated residuals in the A-FULL draft** (with the drift-term direct evaluation still running as the hardening recompute). *This branch inference is flagged for author veto; a one-word correction re-opens it.*

Not covered (inputs listed but not in the decision table): the Gray-convention paper-scope ruling (row #106 item 3 deferral has lapsed) — re-presented separately.

---

## Row #110 — 2026-08-15 — A-FULL draft ratified (FULL-F); registration + run authorized; Gray-convention in paper scope

**Author's verbatim ruling (2026-08-15):** "all approved"

**Orchestrator-derived itemization** (referencing `results/mechanism_study_20260813/DRAFT_A_FULL_ESTIMATOR_20260815.md` §6 **as amended by its verifier addendum**, commits `fe172d6f` + `860b9d3f`):
1. **[RULE — granted]** The A-FULL candidate definition is **FULL-F**: d_obs-density GW factor × selected-population prior w_pop·S̄_φ/α (α retained as the prior's normalization) × leave-one-out impostor weight 1/imp_k; **no Jacobian measure, no kernel renormalization** (deviations from row #109 item 3's wording are evidence-driven and were flagged). Pre-measured venue tilt +30.6 ± 42.7 nats/h at full dose (zero-consistent), +168.9 ± 58.8 at f_i = 0.25 (stated residual).
2. **[DO — granted]** Register (A8-v2: fresh seeds, scorer pre-committed, xhigh pre-registration verifier — mandatory precedent) and run the A-FULL arm on the cluster (~25 CPU-h, N = 25, full dose, both channels).
3. **[RULE — granted]** The registered bands may seed directly from the §2/addendum mirror pre-measurement (with the N-scaling and an execution-completeness cross-check of the installed instrument variant against the pre-measurement mirror before submission).
4. **[RULE — granted, branch reading orchestrator-derived]** The Gray-convention finding **enters the paper's scope**, with the FULL-B/D/F chain as its quantitative backbone (what the published convention costs at σ_z > 0). *Reading flagged for author veto; concrete paper integration is a paper-thread task.*

---

## Row #111 — 2026-08-16 — Stage-5 readout ratified: 1D venue thread CLOSED (M-OWNED); production physics-change proposal + 2D investigation authorized

**Author's verbatim ruling (2026-08-16):** "all approved"

**Orchestrator-derived itemization** (referencing `results/mechanism_study_20260813/STAGE5_READOUT.md` §3, commit `715943ca`):
1. **[RULE — granted]** The stage-5 readout and branch-1 record are ratified: **the 1D venue mechanism thread CLOSES, M-OWNED** by the correct-form account. A-FULL (FULL-F): T(1D) = +22.0 ± 29.2 (DS-F1 PASS, 0.16σ from the mirror prediction), bias +0.0010 ± 0.0011 (from +0.0373), **1D coverage RESTORED** (0.64/0.76/0.96 vs nominal 0.50/0.68/0.90; every prior arm 0/25). Stated residuals honored (low-dose not probed; pool-vs-model mismatch stands).
2. **[DO — granted]** Open the **production `/physics-change` proposal** for `bayesian_statistics.py` — the venue-validated correct form (α-pairing, density-form event term, LOO weight; no Jacobian, no renorm) mapped onto the production estimator; full 5-step gate; reviewable artifact; the A-FULL arm is the evidence base. The `/physics-change` slot is now OCCUPIED by this authorized proposal-in-preparation (no production code changes until the gate passes and the author ratifies the proposal itself).
3. **[RULE — granted, branch reading orchestrator-derived]** The 2D mass-channel defect (+135.7 excess tilt, bias +0.0076 ± 0.0012, coverage not restored, surviving the full 1D repair): **the targeted investigation OPENS** (L0-first: g_i term derived in the convolution frame, then mirror pre-measurement — the stage-4/5 method). *Branch reading flagged for author veto (the alternative was carry-as-residual, which needs no authorization — approval is read as authorizing the work).*

---

## Row #112 — 2026-08-16 — Production-transfer fork ruled: option A (2D-first)

**Author's verbatim ruling (2026-08-16):** "A approved"

**Scope** (referencing `PRODUCTION_TRANSFER_RECON_20260816.md` §3): the L6 2D g_i investigation
executes first (`L6_2D_GI_PLAN_20260816.md` protocol: convolution-frame derivation BEFORE
measurement; bit-exact c2 mirror; freeze-switches S-A/S-B/S-AB; xhigh verifier before
presentation). The production `/physics-change` proposal waits on L6's result (option B's
correspondence mirror and option C's narrow D-ii fix fold in afterwards as L6's outcome
dictates, each returning to the author). The slot remains occupied-paused.

---

## Row #113 — 2026-08-16 — L6 step 1 confirmed: c2-mirror switch measurement

**Author's verbatim ruling (2026-08-16):** "1 confirmed"

**Scope:** execute `L6_DER_2D_CHANNEL_20260816.md` §4 item 1 — the c2 mirror (bit-exact vs
stored `ln_post_2d`) with freeze-switches S-A (g's d_L_frac frozen at h_true) / S-B (g's
z-argument de-drifted to the h_true peak frame) / S-AB, on the 15 MN0X seed replays, 1D channel
asserted untouched. Registered predictions (pre-stated in the L6-DER note, committed `718128d1`
BEFORE this measurement): ΔT2(S-B) ≈ −139, ΔT2(S-A) ≈ 0. The xhigh verifier (item 3) gates any
claim before it returns to the author.

---

## Row #114 — 2026-08-16 — L6 findings ratified (as amended)

**Author's verbatim ruling (2026-08-16):** "ratified"

**Scope** (referencing `L6_DER_2D_CHANNEL_20260816.md` + both addenda, commit `55e73222`): the
L6 findings are ratified as amended — channel B (h-moving evaluation of
`completion_mass_factor_g`'s z-argument against the φ slope) **owns the 2D−1D excess to within
~6%** (ΔT2(S-B) = −139.0 measured vs +139.0 pre-registered prediction; residual −7.489 ± 0.065
nonzero, origin undetermined); channel A null at f = 1; the production transfer is scoped to the
`absolute_marginal` completion leg (channel existence transfers, venue magnitudes do not).
Next per runbook 15 item 2 (standing grant of row #112/A): the correct-form 2D derivation.

---

## Row #115 — 2026-08-16 — L6-DER2 ratified; A-FULL-2D arm + production derivation authorized

**Author's verbatim ruling (2026-08-16):** "all approved"

**Scope** (orchestrator-derived itemisation of the three tagged items presented, referencing
`L6_DER2_GSEL_PREMEASURE_20260816.md` + `L6_DER2_VERIFIER_ADDENDUM_20260816.md`, commits
`fbc60b3a`/`453d1b29`):

1. **[RULE — ratified]** The L6-DER2 findings as amended: the S̄_φ×g factorization error (two
   ∫dM where the selected joint prior demands one ∫dM φ·p_det·N) owns channel B; the fused
   `g_sel` is the verified A-FULL-2D candidate (excess +135.8 ± 0.1 → −11.7 ± 1.0 nats/h,
   91.4% of channel B, 1D bit-untouched, all 4 gates bit-exact); the −11.7 ± 1.0 residual is
   assigned to the known realization-coupled residual class (r = 0.847 vs the c2 switch's
   −7.489 ± 0.065), amendments V1/V2/V4 of record.
2. **[DO — approved]** Register the A-FULL-2D arm (A8-v2, fresh seeds), candidate in the
   **as-measured** §3 form (V2 prefactor NOT added for the venue arm — orchestrator
   recommendation accepted within the approved item; V2 remains live for production).
3. **[DO — approved]** The production completion-leg derivation: whether
   `completion_mass_factor_g` must fuse the with-BH survival in the `absolute_marginal` leg
   and what paired D̃^φ denominator change follows — the reopened `/physics-change` proposal's
   subject. A3 (venue magnitudes do not transfer) stands.

Binding default honored: the arm's own outcome branches and the production proposal return to
the author as fresh rulings.

---

## Row #116 — 2026-08-17 — A-FULL-2D arm branch 1 ratified; 2D venue thread M-OWNED-CLOSED; production proposal authorized

**Author's verbatim ruling (2026-08-17):** "please hook ahead"

**Scope (orchestrator-derived interpretation of the ruling against the two presented [RULE]
items + stated next step, referencing `AFULL2D_ARM_READOUT_20260817.md`, commit `bcd66529`):**

1. **Branch 1 ratified:** DS-G1 PASS (−11.8 ± 0.61 nats/h in the registered band [−15.7, −7.8],
   on the mirror prediction −11.74 ± 1.04) + DS-G3 RESTORED (0.520/0.760/0.960; read
   necessary-but-weak per verifier MAJOR-1) + DS-G5 zero-consistent 2D bias (+0.0006 ± 0.0013,
   0 rails) + DS-G4 bit-identical 1D → the 2D channel has a validated correct-form estimator;
   **the 2D venue thread is M-OWNED-CLOSED** (the −11.7-class residual and pool-vs-model
   mismatch remain stated open residuals, as pre-registered).
2. **Budget overrun accepted as recorded deviation:** 406.5 CPU-h realized (499 allocated) vs
   the 300 ceiling; MINOR-4's pessimistic rate was correct; no scientific consequence.
3. **Proceed:** author the production `/physics-change` proposal ([P1]–[P5] per
   `L6_DER3_PRODUCTION_COMPLETION_LEG_20260816.md`) with this arm as evidence base. The
   proposal itself returns to the author; no production code changes under this ruling.

---

## Row #117 — 2026-08-17 — Selection-fusion proposal: all five decision-table items ratified

**Author's verbatim ruling (2026-08-17):** "please note all as ratified"

**Scope** (referencing `docs/derivations/PROPOSAL_2D_SELECTION_FUSION_20260817.md`, commit
`298c4963`; itemisation is the proposal's own table): item 1 [P1]+[P2] paired fusion approved
for implementation behind the full `/physics-change` gate; item 2 [P3] catalogue-leg fork
deferred to the Gray-convention paper task unless the counterfactual shows material mixture
skew; item 3 [P4] measure ruling settled inside item 1's presentation gate; item 4 [P5-3]
production counterfactual cell approved as the next measurement after item 1 lands (campaign
re-run NOT authorized — returns with the counterfactual's result); item 5 xhigh verifier on
the proposal runs BEFORE item 1's implementation begins. Sequencing next session: verifier →
gate presentation → implementation → counterfactual.

---

## Row #118 — 2026-08-17 — Verifier amendments adjudicated: G1/G2/G3 ruled; implementation unblocked

**Author's ruling (2026-08-17, via structured question — option labels are the author's
selections, phrasing orchestrator-derived):** G1 = "Keep adaptive + guard"; G2 = "Retain
ratio + track"; G3 = "Confirm deferral". All three were the presented recommendations.

**Context:** row #117 item 5's adversarial verifier returned GO-WITH-AMENDMENTS
(`PROPOSAL_2D_SELECTION_FUSION_VERIFIER_ADDENDUM_20260817.md`, commit `44aa239e`); MAJOR-1..4
created three fresh [RULE]s presented in
`GATE_PRESENTATION_SELECTION_FUSION_20260817.md` §5 per the binding default.

**Scope of the rulings:**

1. **G1 [RULE]:** the fused `g_sel,prod` keeps the ratified Route-1 adaptive quadrature with
   `S_4D` evaluated per node, plus (a) a recorded pinned-vs-adaptive regression bound
   (~1e-15 class) and (b) a guard assertion escalating to non-adaptive n_hermite=64 when the
   S-variation across the Hermite node window exceeds tolerance. The proposal's pinned-n=64
   rider is superseded.
2. **G2 [RULE]:** the ratio measure convention is retained in BOTH legs; the V2 prefactor is
   recorded as a tracked systematic (G7 budget row, with the measured completion-leg
   immateriality bound ≲1e-6 at σ_cond p50 = 8.8e-8) and re-opens with [P3]/row #110 where it
   is material. No measure change ships with item 1.
3. **G3 [RULE]:** row #117 item 2's catalogue-leg deferral is CONFIRMED on the corrected
   basis (MAJOR-3: catalogue leg OVER-weighted under [P2], not down-weighted; structure
   sign-independent — the [P5-3] counterfactual measures the skew either way).

**Consequence:** item 1 implementation begins under the full `/physics-change` gate with the
amended verification plan (gate presentation §4), including the MAJOR-1 requirement that the
item-4 counterfactual decompose 1D-only / 2D-only / paired cells.

---

## Row #119 — 2026-08-17 — Fusion counterfactual banked: skew not material, no campaign re-run

**Author's verbatim ruling (2026-08-17):** "as recommended please"

**Interpretation basis (orchestrator-derived, per the attribution convention):** the ruling
was given against `CAMPAIGN_REPORT_20260817.md` §10 (commit `7b512877`) and the session
summary presenting the same three items; "recommended" maps to the readings the report's
analysis carries and the option ordering presented — itemised below.

**Scope:**

1. **[RULE] M-4 materiality — NOT MATERIAL.** The measured mixture skew (median +0.02–0.03,
   max +0.204 catalogue-share gain, confined to the 161/159 of 1588 catalogue-bearing
   events) does not trigger row #117 item 2's "unless material" condition. **The [P3]
   catalogue-leg fork stays deferred to the Gray-convention paper task (row #110)**, which
   now holds the M-4 numbers as its quantitative input.
2. **[RULE] Campaign-re-run scope — option (a), NO RE-RUN.** With zero MAP/width motion in
   the 2D channel of record (M-3), the fused estimator reproduces the campaign posteriors
   within their quoted widths; `results/run_20260817_fusion_counterfactual/` is the
   recorded bridge between pre- and post-fusion results. Campaign CPU budget saved; any
   future re-run request returns as a fresh [DO].
3. **[DO] Measurement banked.** The fusion-magnitude numbers are of record as MEASUREMENTS
   (M-1: 2D +1.245/−3.268 chord nats/h; M-2: 1D +24.588/+22.736 chord, ≡ N-2 of record;
   M-3: no MAP motion; M-4 as above), per the prereg VERDICT (`ac24b632`+`7b512877`
   chain). The sidecar path-repair compliance deviation (report §9 flag 1) is ratified as
   part of this bundle. Carried open: #66/#67 production calibration (pp_coverage mass
   channel, TO-BUILD) remains the stated disappointment path; −11.7-class residual and
   pool-vs-model mismatch unchanged.

---

## Row #120 — 2026-08-17 — Production-calibration-harness front OPENED (stage-0 intake ratified)

**Author's verbatim ruling (2026-08-17):** "ratified, please continue"

**Ruling target:** the §8 decision table of
`results/campaign51_20260728/realistic_20260729/CLAIM_PRODUCTION_CALIBRATION_HARNESS_20260817.md`
(the stage-0 claim intake for runbook 18 §1.2(a), presented this session). **Orchestrator-derived
itemization** — "ratified" read as granting D-3 [RULE]; "please continue" read as granting the
[DO]s D-1/D-2/D-4/D-5. The Gray-convention paper proposal's §5 [RULE]s (the [P3] presentation
option pick, the σ_z→0 wording) are NOT covered by this ruling and remain pending.

**Scope:**

1. **[DO — granted, D-1]** Front opened. Pre-stage-1 cheap checks authorized as bounded
   read-only code audits: Q-0 (does production's distance-error treatment constitute the
   model-σ half that #67 requires as the pairing for the landed selection-inside half?),
   Q-1 (G23-c same-LOS-prior-object check against `2b10b8b8`), Q-2 (G5b P3 re-check against
   the fused production code).
2. **[DO — granted, D-2]** Harness build to the [A3] spec authorized: mass channel with
   completion-leg g recomputed per h + production-N capability, on top of the existing
   `catalogue_mode`; instrumentation-tagged (plain GSD); any formula change that emerges
   routes to `/physics-change`.
3. **[RULE — ratified, D-3]** Exoneration reconciliation of record: this front calibrates
   the landed estimator and does NOT re-open #66/#67; ledger §2 item 6 and the claim file's
   "p_det inside/outside" entry remain binding as venue-scoped bias-mechanism verdicts
   (four-axis reconciliation per the intake §3; CC-3's blanket-reading ambiguity resolved).
4. **[DO — granted, D-4]** The five `docs/LITERATURE_WARNINGS.md` row updates (G23-c-check,
   Essick & Fishbach 2024 section, Talts/CGR section, G5b staleness flag, H-d
   promotion-on-instrument) authorized.
5. **[DO — granted, D-5]** Stage-1 sizing and stage-2 pre-registration to be authored next;
   each returns to the author before any run; [R-3] CPU budget filled at submission per
   row #116 discipline. Any verdict-dependent branch (e.g. implications of a failed [R-1]
   cell for the landed fusion) returns as a fresh [RULE].

---

## Row #121 — 2026-08-18 — Prodcal ladder freeze+execute ratified; paper-grounding measurements to be collected

**Author's verbatim ruling (2026-08-18):** "all ratified and is for the paper (b) the most
explicit one? if we need one final measurement to really ground it, we should consider doing
that or at least collect all measurements we can do once the full pipeline ist settled for the
paper"

**Orchestrator-derived itemization:**

1. **[DO — granted]** Freeze-and-execute the prodcal ladder: one freezing commit (prereg v3 +
   pre-committed scorer + verifier addendum, final gate GO + the [A3] harness extension and its
   tests), then the registered §7 pretuning fill-in (seed 20270999, fixed sweep), then the
   ladder (≈14.6 CPU-h, ceiling 18, local). Readout returns with the registered branch calls.
2. **[DO — granted]** Ch15 ("The Slot Gets Filled") slot and title ratified; chapter +
   generator + manifest wiring committed.
3. **[RULE — deferred by the author's own framing]** The [P3] presentation pick is NOT made:
   the author asked whether (b) resolved-in-paper is the most explicit option (orchestrator
   answer: yes, and it is the one option lacking its grounding measurement). Per the author's
   directive, the grounding measurements are **collected once the full pipeline is settled**
   (= after the ladder verdict): (i) the catalogue-leg fusion counterfactual (the item-4 analog
   with survival fused into the catalogue leg — [P3]'s direct H0-impact measurement); (ii) the
   spec-z-kernel σ_z→0 cell (grounds the no-cost-limit wording). The (a)/(b) pick returns to
   the author with those measurements in hand. Registered in the Gray-convention proposal §6
   (paper-grounding measurement list, added this session before commit).
4. Remaining paper [DO]s of proposal §5 (scope, TO-MAKE figures, discussion.tex:235 rewrite)
   are granted under this ruling; execution sequenced after the measurement collection so the
   figures/text carry final numbers.

---

## Row #122 — 2026-08-18 — Prodcal ladder readout ratified: asymmetric-[P2] boundary measured, instrument cleared

**Author's verbatim ruling (2026-08-18):** "all ratified. Question about the asymmetric entering
of S. Why shoud there even be a scientifically motivation that it only appears in one of the
legs? but maybe I misunderstood this."

**Ruling target:** `results/pp_coverage_prodcal_20260817/CAMPAIGN_REPORT_20260818.md` §10
(decisions 1–5). **Orchestrator-derived itemization:**

1. **[RULE — ratified, #1]** Readout bundle of record: H-P/H-N FAIL at V-deep with the audit's
   mechanism attribution (first-order tilt from S̄_φ's gradient; instrument faithful — no code
   defect); the two compliance deviations (V-ctrl structurally void via the D-1 z_support=1.5 >
   Z_MAX_POP error; Block-N1 driver `mixture_mode` erratum, quarantined + faithfully rerun); the
   2D-bias misattribution correction (venue noise physics, present in `off`). Banked as a
   MEASUREMENT of where the asymmetric insertion is unsafe — NOT as production-bias evidence.
2. **[RULE — ratified, #2, orchestrator-derived option reading]** Production-regime calibration
   status: **option (a) — OPEN pending the flat-S̄ control cell**, read as implied by the
   simultaneous grant of #3 (the cell that decides it). *Reading flagged for author veto.*
3. **[DO — granted, #3]** AMENDMENT-2: flat-S̄ completion control venue (~1 CPU-h), verifier
   one-item pre-check first; audit prediction: small positive tilt, ≈0 MAP motion.
4. **[DO — granted, #4]** φ-slope decider cell for the 2D catalogue-leg overlap question.
5. **[RULE — ratified, #5]** The symmetric-insertion finding enters the record as a NEW CLAIM
   (stage-0 intake for a future front): the exact latent-threshold conditional puts S̄ in BOTH
   numerator legs; production's completion-only form is a scoped, empirically-safe-in-regime
   approximation whose validity boundary is now measured. Refute-by: the flat-S̄ cell failing to
   calibrate under the symmetric variant. **No production change proposed or authorized.**

**Author's physics question answered in-session (recorded):** there is NO first-principles
motivation for the asymmetry — it is derivation scope (completion leg was the measured
defect-owner; catalogue-leg counterpart = the deferred [P3] fork, MAJOR-3/M-4/row #119) plus
regime safety (flat S̄ ⇒ the asymmetry's error nearly cancels; counterfactual zero MAP motion).
The symmetric form is the theoretically-motivated one; harness evidence: symmetric +0.008 /
completion-only −0.029 / catalogue-only +0.041.

---

## Row #123 — 2026-08-18 — V-flat read banked; AMENDMENT-4 final verification granted; cycle-status question answered

**Author's verbatim ruling (2026-08-18):** "So I have one question. Did everything go well and
only because of the newly detected venue physics? which introduces a bias. We now have to kill
this research cycle because it says sixty six and sixty seven disappointment path. Or did this
research cycle produce as expected? Now the final conclusion is that the s factor needs to
appear both legs equally. Secondly, for number two, I think it just makes sense to have this
additional final verification, especially if it doesn't need a lot of compute. So please go for
It. And then if I understand correctly, number three motivates the next research cycle or where
are we currently finally resolving the residual by situation."

**Orchestrator-derived itemization:**

1. **[DO — granted]** AMENDMENT-4: the final absolute-calibration verification at a venue with
   flat S̄ AND production-like completion share (≈0.37 anchor); verifier Part-VII pre-check
   first; ≤0.9 CPU-h. Executes VERDICT-2's designated next measurement.
2. **[RULE — PENDING]** VERDICT-2 ratification (report decision 1) — the author asked for
   comprehension first; the answer of record (given in-session): the cycle SUCCEEDED, nothing
   is killed — the #66/#67 disappointment path closed BENIGN for production's regime (three
   consistent measurements); the V-deep H-P FAIL is the measured validity boundary of the
   asymmetric form (a discovery, not a malfunction); the venue physics is an independent
   secondary observation that caused neither. "Both legs equally" is the correct FORM ([C-SYM])
   but production does not need it as a bias fix (asymmetry's error measured negligible in
   regime) — a correctness/paper matter per the standing correctness-over-bias-removal ruling.
3. **Recorded orientation answer:** no live unexplained production H₀ bias remains (rows
   #111/#116/#119 + this cycle); the record's next cycle is the [C-SYM]/[P3] correct-form +
   Gray-paper front (with G-1/G-2 as grounding); the −11.7-class residual sits inside its
   registered band (documented, not a defect); the venue-physics observation is a low-priority
   seed far from production's regime.

**Addendum (2026-08-18) — author's verbatim ruling: "ratified, continue with amendment-4 once
the verifier gates GO".** Item 2's pending [RULE] is GRANTED: **VERDICT-2 is ratified** (V-flat
delta-leg PASS = fusion lever regime-consistent; absolute-leg FAIL attributed venue-intrinsic).
AMENDMENT-4 execution confirmed on the already-landed verifier Part-VII GO (pretuning landed
z_support=0.75, completion 0.384; registered pair running at ruling time).

---

## Row #124 — 2026-08-18 — Prodcal cycle CLOSED: lever certified benign in regime; absolute half closed route (a)

**Author's verbatim ruling (2026-08-18):** "approved" — on the VERDICT-3 presentation (two
routes for row #122 item 2's absolute half). **Orchestrator-derived reading, flagged for author
veto:** route **(a)** — the absolute half CLOSES on lever-closure plus production's own
diagnostics; the flattened-detection venue-physics observation is logged as a SEED, not opened.

**Scope of record (orchestrator-derived itemization):**

1. **[RULE — per the above reading]** Row #122 item 2 is CLOSED, both halves. Lever half: three
   mutually consistent measurements (production counterfactual M-3; V-flat delta PASS; V-prod
   delta PASS) — **the landed selection fusion is certified benign in production's regime at
   harness fidelity; the #66/#67 disappointment path is CLOSED BENIGN.** Absolute half: closed
   route (a) — harness-level absolute certification at flat-S̄ venues is confounded by the
   raised-d50 venue bias (VERDICT-3's differential: driver is the d50 flattening, not
   completion share; VERDICT-2's attribution corrected of record) and is left to any future
   front with an estimator-side-only S̄ instrument.
2. **The prodcal research cycle (rows #120–#124) is CLOSED.** Products of record: the [A3]
   two-channel harness (+ tests); the asymmetric-[P2] validity boundary (safe flat-S̄ / unsafe
   strong-gradient, mechanism owned to 3%); the [C-SYM] claim (admitted, row #122 item 5); the
   N-1 continuity chain; the Q-0/Q-1/Q-2 audits; 6 verifier passes (Parts I–VII).
3. **Carried open (unchanged owners):** [C-SYM]/[P3] correct-form + Gray-paper front with
   G-1/G-2 grounding (each prereg-first, own [DO]); flattened-detection venue-physics SEED;
   2D-channel venue noise-coupling (+0.01 class, off-cells) SEED; −11.7-class residual (in
   band); pool-vs-model mismatch. Budget close-out: ≈16.5 of 18 CPU-h ceiling.

## Row #125 — 2026-08-18 — [C-SYM]/[P3] front executed autonomously; G-1/G-2 grounding measurements banked; ratification bundle PENDING

**Author's verbatim ruling (2026-08-18, mid-session):** "Please move through this research
cycle autonomously and flag it accordingly. Make sure you dont make the same mistake as last
cycle where one of the instrument arms was void in the end. looking forward to see the
results!"

**Orchestrator-derived itemization (flagged for author veto):**

1. **[DO — granted, autonomy-scoped]** The [C-SYM]/[P3] cycle's [DO]-class steps executed
   in-session without per-step return: prereg authoring (G-1/G-2), verifier pre-checks
   (Parts I–VI, all amendments applied verbatim), instrument build (`cat1d`/`symmetric`
   selection cells + 13 tests), freeze (`4dd822ad`), arm-validity preflights (the mandated
   anti-void gate — caught a grid-edge void and a 2D-band void pre-run; N-A false alarm
   resolved as a probe-scale artifact, full-R cluster-vs-local BIT-EXACT), scored runs
   (G-1 ≈ 2.75/3 CPU-h; G-2 ≈ 1.1/6 CPU-h), registered readouts, AMENDMENT B separating
   cells. **Branch calls NOT adjudicated** — per the binding default they return as fresh
   [RULE]s (the readout's §6 decision table).
2. **Measurements of record (registered, presented):** H-G1 PASS — catalogue-leg fusion
   immaterial in the production-analog regime (symmetric − fused ≤ +0.0002 in h; V-flat
   bit-inert) ⇒ the ~170 CPU-h production catalogue-leg counterfactual proposed NOT
   warranted. H-CAT PASS (+0.0428 — the audit's ad-hoc trio now registered). H-SYM MIXED →
   SEP-Z interpretation-A (the symmetric form's +0.007 residual at σ_z = 0.035 is photo-z
   venue physics; calibrated-in-bias at spec-z). G-2: BOTH legs' insertion costs are
   photo-z-driven (completion −0.0300 → −0.0004, ~70×, refuting the registered persistence
   prediction; catalogue +0.0424 → +0.0038, PASS with the +0.0038 residual MATERIAL vs the
   0.0018 yardstick). σ_z-conditionality annotated to the [C-SYM] claim card as new stage-0
   evidence.
3. **[RULE — PENDING, ×6]** The ratification bundle + the [P3] presentation pick (a)/(b) +
   the σ_z-wording ruling return to the author via
   `results/pp_coverage_csym_20260818/CAMPAIGN_REPORT_20260818.md` §6 (decisions 1–6).
   Nothing in this row ratifies them.
4. **Seeds logged, not opened:** σ_z = 0.002 coverage-profile distortion (venue property,
   shared by calibrated control); catalogue-leg σ_z-independent floor component (mechanism
   unattributed).

Products: `results/pp_coverage_csym_20260818/` (preregs + Parts I–VI verifier record + build
note + 24 cells + scorer outputs + report); `docs/LITERATURE_WARNINGS.md` row MFG-a; the
instrument-determinism result (full-R bit-exactness across machines). No production change
proposed or made.

## Row #126 — 2026-08-18 — G-1/G-2 bundles ratified; [P3] presentation pick = (b) resolved-in-paper

**Author's verbatim ruling (2026-08-18):** "awesome, approved and please do b."

**Orchestrator-derived itemization (flagged for author veto), against
`results/pp_coverage_csym_20260818/CAMPAIGN_REPORT_20260818.md` §6:**

1. **[RULE — ratified, §6 #1]** G-1 readout bundle of record (H-G1 PASS ⇒ the ~170 CPU-h
   production catalogue-leg counterfactual is NOT warranted; H-CAT PASS; H-SYM MIXED resolved
   interpretation-A via SEP-Z; V-flat bit-inertness; disclosed drafting flags).
2. **[RULE — ratified, §6 #2]** G-2 readout bundle of record (H-Zcat PASS with the material
   +0.0038 spec-z floor; H-Zcomp FAIL = both legs photo-z-driven; σ_z-conditionality
   annotated to the [C-SYM] card).
3. **[RULE — decided, §6 #3]** **[P3] presentation pick = OPTION (b), resolved-in-paper:**
   the paper presents the latent-model derivation as the correct form for BOTH legs; the
   retained catalogue-leg convention as a measured approximation (M-4 mixture-share bound +
   the new G-1 direct bound ≤ +0.0002 in regime); the [P4]/V2 measure-prefactor question
   discussed in text (per proposal §2.4(b)).
4. **[RULE — ratified, §6 #4]** σ_z → 0 wording of record: per-leg, with measured numbers.
5. **[DO — granted, §6 #5]** Paper integration work proceeds under pick (b).
6. **[RULE — ratified, §6 #6]** Both seeds logged, not opened.

## Row #127 — 2026-08-18/19 — Production-2D closure + catalog-quality landscape front OPENED and frozen (autonomous, cluster+overnight authorized)

**Author's verbatim ruling (2026-08-18):** "I want this measured for sure! please do that
closure. you can also think about if anything else is needed to be run on the cluster and
please go ahead autonomous, I will ensure the ssh connection one more now, so you should keep
it active over night. if would be huge news if given this horrible data (redshift and mass
error) 1d starves while 2d is able to constrain. we could then further extrapolate other
realisations of the errors due to improved measurements ( in the best case motivated by known
missions to come) and have a landscape that clearly tells you: given this good of a catalog we
can expect that constraint."

**Orchestrator-derived itemization (flagged):**

1. **[DO — granted]** The production-2D closure (the number-to-number budget row #124 left
   uncomputed) + the (σ_z × σ_m) catalog-quality landscape, cluster + overnight, autonomous;
   branch calls return as [RULE]s.
2. **Registered instrument:** `results/prod2d_closure_20260818/` — three-tier prereg
   (T0 production-native bootstrap/jackknife; T1 closure factorial; T2 landscape; 18 cluster
   cells ≈ 74–79 CPU-h, ceiling 160), verifier Part VII (4 BLOCKING + 4 NON-BLOCKING, all
   applied verbatim — decisive catch P7-1: the "1D starves" harness leg may only be scored on
   off-basis cells, never on the venue-scoped asymmetric-insertion artifact; the
   production-native H-L1-prod arm is the only place the author's headline sentence may be
   quoted). Arm-validity preflight READY (5 probes, good-corner 1D un-rail confirmed).
   Cluster preflight READY ✓ (2026-08-18).
3. **Recon facts of record ([LOCAL]):** iiib/joint_r1 share one realization (seed61000,
   byte-identical CRB); production 2D offsets +0.054/+0.067 (trapezoid convention); iiib's
   top-slope event is 889 (SNR 1424.7, rank 1); production σ_M of record = R&V15 ε₀ = 0.24 dex
   (post-`555f018`; the "0.55 dex" in older session memory was pre-fix wording); documented
   Eddington-in-M 2D impact −0.020 (`bayesian_statistics.py:5454`).
