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
| 90 | 07-27 `608426b` | (d1) joint z×M_z with-BH selection conditioning owns the residual | pre-registered A/B + grid-only control | **NULL**: A-cell Δ@0.80 = −0.51 raw / −1.07 grid-corrected; the −6.5±4 gate **FAILS LOW then RETIRED** | `mass_ab_20260727/ZMZ_AB_READOUT.md:20-45` | — |
| 91 | 07-28 | (g1) the m_max clamp suppresses (d1) | P1 parity audit | **REFUTED** — clamped queries carry **75–90%** of the conditioning movement; the ×3–5 shortfall was an axis-translation error (D_gen multiplier 718 vs A-cell Σs≈225) | `mass_ab_20260727/P1_PARITY_AUDIT.md:3-8,208-225` | — |
| 92 | 07-28 | Remaining ≈+23 ln 2D HIGH residual (MAP 0.80) owner | enumeration after 89–91 | **STILL OPEN — owners: (d2) selection-side M scatter/truncation + (g1)-as-support-limitation**; campaign is the critical path | `RUNBOOK_NEXT_SESSION_4.md:29-33`, `docs/campaign_redesign_51_design.md:180-205` | +23 ln |
| 93 | 07-29 #51 | Idealized baseline (point kernel + `generator_marginal`, unscattered) | 1e-4 zoom grid | 1D/2D **0.72990 / 0.7300**, −0.24σ / −0.36σ. 100% of information from **76 in-cat events**; 3 golden carry 46% | `IDEALIZED_BASELINE_READOUT.md:25-47` | — |
| 94 | 07-29 #53 | Realistic run (`absolute_marginal` + `volume_deconv`, scattered catalogue) | P1–P6 scorecard, 10 runs | 1D 0.700–0.740 (pooled 0.7205) · **2D 0.780–0.820, mean +0.077, 10/10 pull > 2 (+4.04)**; P5 σ→0 byte-identical PASS | `REALISTIC_READOUT.md:19-32,140-153` | **current** |

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
