# B1.1-F — Forensic of the S0-A B0-A′ defect (why E[∂_θ ln L] ≠ 0 at θ=(0,1) on the b0i mirror)

**Launched under rows #222/#223 — charter node B1.1 (sub-node B1.1-F, top-tier forensic).**
Date: 2026-08-29 (computations finished 2026-08-30 00:xx local). Branch `fix/p32d-classg-venue-repair`, HEAD `0d0eb691`.
Scope discipline: zero re-evaluation (no `evaluate()` call, no `sbatch`, no ssh); pandas/numpy on the banked CSVs, the
pinned catalogue (md5 `c52c13b5cab61f6b3f04bbe202550969` verified before reading, `md5sum` 2026-08-29), the completeness
cache and the S̄_φ table built by the harness's own `build_bsel_selection_objects` (the PA-HIER-30 precedent). No code
edited; the KW-Q1 runner's tree (`kwq1_registered_run/`) untouched. All instruments and their outputs are archived under
`fanout1_20260829/b1_1_forensic_work/` (scripts `f1`…`f12`, outputs `f*_out.json`, event tables). Every [HIER] statement
below carries the **REPORTED-ONLY cap (PA-HIER-28 item 9)**. The B0-A′ **INSTRUMENT-DEFECT → STOP** verdict of the
S0-A record stands; this note localises the defect, it does not rescue the run.

---

## 0. Verdict (one paragraph)

**LOCALISATION: VENUE-LAW / INSTRUMENT-FORM — not a hook-arithmetic defect.** An independent numpy twin of the no-BH
catalogue leg (own candidate search through the production `BallTree`, own GL-50 quadrature, own 3-D MVN) reproduces the
estimator's `L_cat_no_bh` at the truth node to **9.2e-13** (max |Δ ln L|) and the per-event registered secants at the
four θ-nodes to **3.0e-12 (b)** / **8.4e-13 (s)**, correlation 1.000000 (E7). The θ hook at site 2.2 therefore does
exactly what §1.2 registers. The non-zero score is produced by the **venue and the instrument's own form**:
(i) the no-BH catalogue divisor Σ^φ carries **no θ-dependence in any built form** — the phi-table branch of
`precompute_global_catalog_selection` is point-evaluated and precedes the only θ-receiving branch
(`bayesian_statistics.py:2906` `if phi_survival_table is not None:` → `:2916` `elif smear_sigma_z:`), and the no-BH leg
consumes it via `_global_cat_selection_phi` (`:5187-5191`) — so the registered score at truth-θ equals, to first order,
⟨c_i⟩·∂_θ ln Σ^φ(θ) ≠ 0 **by construction of the instrument**, not by any generator/estimator kernel mismatch;
restoring that θ-dependence post hoc (a per-node scalar, E11) turns the registered b-score from
**−1.634 ± 0.444 (Z −3.68)** into **−0.268 ± 0.431 (Z −0.62)**; (ii) the candidate ball (sky cone 1.5σ_max,
z-window ±3σ_d(h∈[0.50,0.86]) widened by ±1σ_g) truncates the host mixture — it drops the TRUE host for
**16.1 %** of events (exactly the estimator's own P6 counters 91/106, 105/120, 87/105, 104/130, reproduced) and, more
importantly, leaves an impostor-dominated mixture (median 278 candidates, true-host posterior share median **0.006**)
whose θ-response is not the generator's; the s-axis non-null is **~75 % this truncation** (enlarging the ball in the twin
moves the catalogue-leg s-secant from −0.052 ± 0.022 to +0.019 ± 0.016, E9) and, after both corrections, the
c-weighted s-score is **−0.005 ± 0.011 (Z −0.5)**; (iii) the registered linear/ln-s secant at ±ln√2 has an intrinsic
O(Δ²) positive bias (+0.046 ± 0.001 per event, unweighted, single-host limit) that makes the B0-A s-band
mis-formed at this N even on a perfect venue (PA-HIER-4 class). Two genuine hook findings are recorded but are
**immaterial** to S0-A: a **HOOK-PLACEMENT gap** (no θ at the no-BH divisor, the cause of (i)) and a **HOOK edge case**
(at b = −0.02, hosts with z_g < 0.02(1+z_g) get a negative kernel centre; 15,618 of 20,834,171 pool rows (0.075 %) get an
inverted window whose `Z_g` guard silently substitutes 1.0; 3/800 drawn hosts affected). The driver's `bc`-flag choice
(`catalogue_numerator_survival="off"`, my initial hypothesis) is **REFUTED as a cause**: the "phi" numerator changes the
per-event b-secant by 0.03 and the s-secant by < 1e-3 (E8). **GATE PARITY's 5.7e-4 residual is RESOLVED as a
generator-side comparand delta** (the 401→4001 inverse-CDF hardening in `d40fe5c8` moved every `z_true` by ≤ 1.1e-5 and
`obs_d_L` by ≤ 6.1e-5; all other CRB columns and every global table are bit-identical); the with-BH residual (up to 86
nats in `L_cat_with_bh`) is the symmetric mass-filter adoption `cf4f8a2a` (2026-08-25) changing candidate lists after the
2026-08-23 bank — neither is an estimator defect on the no-BH path.

---

## 1. Registered inputs re-derived (all from the banked node CSVs; `f1_out.json`)

| statistic (no-BH `combined_no_bh`, N = 461, 4 seeds × 5 nodes, form `sites2.2_nosmear`) | value | source |
|---|---|---|
| score_b mean / SEM / Z | −1.61646 / 0.43968 / **−3.676** | `f1_out.json` (matches `s0a_score.md`) |
| score_s (linear secant /0.70711) mean / SEM / Z | −0.086253 / 0.012185 / **−7.079** | idem |
| score_lns (/ln 2) mean / SEM / Z | −0.087990 / 0.012430 / −7.079 (Z identical, relabel immaterial) | idem |
| with-BH score_b / score_s (Z) | +0.379 / −2.027 | idem |
| dark class (`L_cat_no_bh == 0`, n = 5): both scores | exactly 0.0 at every node | `f1_out.json` |
| per-seed score_b (Z): 900101 / 02 / 03 / 04 | −3.04 (−2.67) / −2.68 (−2.95) / −0.15 (−0.25) / −0.67 (−0.85) | idem |
| per-seed score_s (Z) | −0.105 (−3.94) / −0.119 (−4.29) / −0.052 (−2.64) / −0.068 (−3.13) | idem |
| curvature (matched 456): b̂ = −S′/S″, σ_b | −0.00794, 0.00326 | idem (`curv`) |
| curvature: ln ŝ, σ_ln s | −0.327, 0.0897 | idem |
| **column audit** (each seed × each off-truth node): columns that move | only `L_cat_no_bh`, `L_cat_with_bh`, `combined_no_bh`, `combined_with_bh`, `num_log_term_*` | `f1_out.json` `audit` |
| columns bit-identical across all nodes | `w_G`, `w_G_legacy`, `w_tilde_G`, `alpha_G_phi`, `r_Malm`, `D_tilde_phi`, `B_num`, `B_num_wbh`, `g_frac`, `L_comp`, `den_log_term` | idem |
| duplicate rows in `s0a_seed900101/node_b_plus_sites2.2_nosmear/…/event_likelihoods.csv` | 2 × 106 rows, halves bit-identical (max |Δ| 0.0 on all 18 numeric columns); dedupe `keep="last"` as the driver does | `f1` run log |

Method item (1) answered: θ does **not** leak into any non-catalogue term (α_G^φ, D̃^φ, B_num are bit-identical across
nodes); the dark class scores exactly zero; the hook is confined to the per-host catalogue kernels. HOOK-PLACEMENT
**leakage**: EXCLUDED.

---

## 2. Evidence table

| # | evidence | value {source, date 2026-08-29/30} | bearing |
|---|---|---|---|
| E1 | Generator law reproduced: `kernel_smeared_survival` on the drawn hosts vs the CRB column `s_tilde_phi_host` | max rel diff **8.0e-14**; every `z_true` inside its host's floored ±4σ window (461/461) {`f3_out.json`} | the objects I use ARE the generator's (PA-HIER-30 leg (a) re-confirmed) |
| E2 | Generator pull on the drawn hosts, bare kernel: (z_true − z_g)/σ_g | mean +0.269 ± 0.045, SD 0.963 (n 461) {`f3_out.json`} | the +0.27 is the volume/completeness tilt of k_g, not a b-offset (see E3) |
| E3 | Generator vs estimator kernel moments per host (k = N·w_pop·f_k on the floored window; p = k·S̄_φ): b_eff = (μ_p − μ_k)/(1+z_g); s_eff = σ_p/σ_k | **b_eff = −0.00263 ± 0.00005; s_eff = 0.9921 ± 0.0002 (median 0.9913)**; ⟨∂_z ln S̄_φ⟩_k = −2.40 ± 0.04 {`f3_out.json`} | truth-θ IS (0,1) at the kernel level (PA-HIER-20 leg 1); the S̄_φ tilt is the physics that the θ-dependent divisor would absorb |
| E4 | Generator law on ALL 800 draws vs the evaluated 461: (z_true − μ_p)/σ_p | evaluated **+0.19 ± 0.046**, excluded −0.11 ± 0.054, all-800 ≈ +0.06 ± 0.035 {`f6_out.json`} | the evaluated subset is SELECTED on the data (V4) |
| E5 | Fisher-quality exclusion (cond(cov_3d/cov_4d) > 1e16, `bayesian_statistics.py:3395,:4422-4423`, blocks scaled by 1/d_L and 1/M `:4380-4418`): evaluated fraction | 461/800 = 58 %; by SNR quartile 7 / 38 / 87 / 99.5 %; by z_true bin 25 / 47 / 66 / 81 %; by d_L quartile 37 / 53 / 66 / 75 % {`f6_out.json`} | a d_L-dependent selection the likelihood does not condition on (V4) |
| E6 | True-host candidate-ball inclusion reproduced from the estimator's rules (sky radius 1.5·√λ_max(JΣJᵀ) `handler.py:640-660`, `:4869`; z-window `get_redshift_outer_bounds` ±3σ_d at h∈[0.50,0.86] `physical_relations.py:563-566` with ±1σ_g widening `handler.py:668-676`) | recovered **91/106, 105/120, 87/105, 104/130** = the estimator's own P6 counters (`p3_b0_work/bc_*.log`, "P6 host-recovery (h=0.7300)"); sky-cone exclusion 14.5 %, z-window 2.2 %; z-window half-width median **1.45 σ_g**, [z_min, z_max]/z_GW = [0.656, 1.241] {`f5_out.json`} | the window model is exact; (V3) quantified |
| E7 | **Independent twin** of the no-BH catalogue leg at the 5 nodes (own `BallTree` query + z filter, own GL-50 on both windows, own MVN from the CRB Fisher block, `w_g = R_eff_per_mbh(M)/(1+z)`) | ln L_cat(truth) max |Δ| **9.2e-13**; secant_b corr 1.0, slope 1.000000000000006, max |Δ| 3.0e-12; secant_s slope 1.0000000000000326, max |Δ| 8.4e-13 (n 456) {`f7_out.json`, `f8_out.json` E0} | **HOOK ARITHMETIC EXACT** (site 2.2: b shifts host_z, s scales σ_raw, window and Z_g on the θ-kernel, numerator on the event window) |
| E8 | Twin with the "phi" numerator (`catalogue_numerator_survival="phi"`, the production default `bayesian_statistics.py:3672-3677`) instead of the driver's "off" (`hier_s0_driver.py:94`, copied from `p3_b0_identity_test.py:998`) | b-secant +1.908 ± 0.702 vs +1.941 ± 0.703; s-secant −0.052 vs −0.052 {`f8_out.json` E0 `phi_*` vs `off_*`} | the bc-flag hypothesis (A) **REFUTED as a cause** (GW-precise regime: σ_zGW/σ_k median 0.130, `f4_out.json`) |
| E9 | Twin with an enlarged ball (sky 3.0σ_max, z-widening ±4σ_g; median 1729 candidates, max 203,815) | catalogue-leg s-secant **−0.052 ± 0.022 → +0.019 ± 0.016**; b-secant +1.94 → +1.17 ± 0.69 {`f8_out.json` E1} | the s-axis non-null is a ball-truncation effect (V3) |
| E10 | Missing divisor θ-dependence: C_θ = secant of ln Σ^φ(θ), Σ^φ(θ) = Σ_g w_g S̃_g(θ) (two independent estimates: draw-weighted mean over the 797 well-posed drawn hosts; explicit w_g-weighted 200k-row pool subsample) | ρ(b+) 0.9546/0.9538, ρ(b−) 1.0424/1.0435, ρ(s+) 0.9884/0.9893, ρ(s−) 1.0065/1.0059 → **C_b = −2.20/−2.25, C_s = −0.0256/−0.0236**; per-host C_b = −2.200 ± 0.036 {`f12_out.json`} | the sign and size of the b-axis defect |
| E11 | **Registered statistic recomputed with the divisor made θ-dependent** (exact, per event: `(βL_cat(θ)/ρ(θ) + B_num)/D̃`, βL_cat = combined·D̃ − B_num) | **score_b −0.268 ± 0.431 (Z −0.62)** [drawn-ρ: −0.296 ± 0.432]; per seed −1.71 (−1.5σ) / −1.26 (−1.4σ) / +1.17 (+2.0σ) / +0.69 (+0.9σ); **score_s −0.0728 ± 0.0122 (Z −5.97)** {`f12_out.json`} | b-axis fully accounted for by (i); s-axis needs (ii) |
| E12 | Twin, both corrections (enlarged ball + divisor), c-weighted (c_i = βL_cat/(βL_cat + B_num), mean 0.616, median 0.651) | b: −0.78 ± 0.47 (Z −1.7); **s: −0.005 ± 0.011 (Z −0.5)**; unweighted s +0.036 ± 0.016 (= the secant bias, E13) {`f8_out.json` E1 `phi_minus_C_*_x_c`} | after (i)+(ii) both axes are null within the REPORTED-ONLY resolution |
| E13 | Intrinsic O(Δ²) bias of the registered ±ln√2 secant on a PERFECT single-host venue (deterministic expectation of the secant under the host's own kernel, GW-precise) | **+0.0455 ± 0.0005** per unit s unweighted; +0.0265 c-weighted; b-secant −0.14 ± 0.02 (floor/tilt asymmetry at low z) {`f4_out.json` `Es_null_det`, `Eb_null_det`} | the B0-A s-band (|Z_s| ≤ 3) is mis-formed at N ≈ 461 (predicted Z ≈ +3.8 on a perfect venue) — PA-HIER-4 class |
| E14 | True-host vs impostor decomposition (twin): posterior share of the true host π_true | mean 0.084, median **0.006**, > 0.5 for 5.3 % of events; impostor-only b-secant by z_g: −20.8 ± 3.1 (z_g<0.075), −4.4 ± 1.4, +4.2 ± 0.8, +14.0 ± 1.2 (0.25–0.39); true-host-only +0.2 ± 12.8, +10.7 ± 3.4, +2.3 ± 2.2, +0.3 ± 3.1 {`f7_out.json`} | the catalogue leg is impostor-dominated at every z on b0i; its z_g-structure is the ball/tilt geometry |
| E15 | Kernel-mean tilt (μ_k − z_g)/σ_k by z_g bin | +0.91, +0.42, +0.10, −0.17, −0.71 {`f5_out.json`} | why low-z impostors want b < 0 (volume prior lifts every candidate's kernel mean ~1σ above its listed z) |
| E16 | Registered z-bin read (by z_true) | 0–0.075 (n 20): score_b **−27.7 ± 3.1**, pull vs μ_k −0.78; 0.075–0.392 (n 433): −0.45 ± 0.35 {`f1_out.json`, `f4_out.json`} | binning on the data selects events whose z_true fell low in their kernel — a selection artefact, not a mechanism (bears on §4.5's "z-resolved θ-score" read and PA-HIER-31 (d)) |
| E17 | c-share bins | c < 0.2 (n 39): score_b +2.40 ± 0.18, catalogue-leg +24.8; c > 0.95 (n 37): score_b −18.7 ± 2.3, score_s −0.644 ± 0.041 {`f4_out.json`} | the pooled score is a cancellation of two opposite-sign classes; Cov(c, s) ≠ 0 because the mirror has no dark hosts while the estimator carries w_G = 0.0620 (V2) |
| E18 | GATE PARITY column diff, truth node vs banked bc CSV (h = 0.73) | global columns and `selection_tables_h_0_73.json` (β_G^φ, β_Ḡ^φ, Σ^φ, Σ^4D, r_Malm) **bit-identical**; `combined_no_bh` max rel 5.718e-4 / 2.38e-4 / 2.19e-4 / 1.15e-4; `B_num` 5.7e-4; `L_cat_no_bh` 2.8e-4 (105/106 rows differ); `combined_with_bh` max rel 0.72 / 1.17 / 1.42 / 0.37; `L_cat_with_bh` |Δ ln| up to 86.4; `L_cat_with_bh == 0` counts banked 6/1/11/9 vs now 1/0/3/1 {`f2_out.json`} | residual lives in every per-event numerator, not in the globals |
| E19 | CRB event-table diff, banked bc vs truth node (seed 900101, 200 rows) | only `z_true` (max |Δ| 1.06e-5, 200/200 rows) and `luminosity_distance` (max 6.1e-5, 198 rows) differ; sky, host index, `s_tilde_phi_host`, M, SNR, all Fisher entries identical {`f2` CRB diff} | the comparand's EVENTS moved: `_B0I_ZTRUE_GRID_N` 401 → 4001 (`git diff 71b52e9c HEAD -- correspondence_1d.py`, `:1168` today) in `d40fe5c8` |
| E20 | Hook edge case at b = −0.02 (pool census) | negative kernel centre for 63,036 rows (0.30 %); inverted window `host_z' + 4σ ≤ 1e-6` for **15,618** rows (0.075 %); weight share of all z' < 0 rows in Σ^φ 0.54 %; 3/800 drawn hosts, one with S̃(b−) = −310 {`f12_out.json`} | a real edge defect of the b-hook (`Z_g ≤ 0 → 1.0` guard at the batch `z_prior_norm` line), immaterial here |
| E21 | KW-Q1 contamination by the same divisor gap (FT: "phi" numerator, 2.2/unsmeared): Δ_h[ln ρ(√2;h) − ln ρ(1/√2;h)] at h ∈ {0.725, 0.735}, 200k-row pool subsample | C_s(0.725) = −0.02380, C_s(0.735) = −0.02324; ∂_h ln ρ(√2) = +0.0246, ∂_h ln ρ(1/√2) = −0.0146 → **+0.039 per unit c** in the numerator of R {`f10_out.json`} | ≈ +0.02–0.03 in R against |S(1)| ≈ 0.80 — immaterial to the 0.2/0.5 bands |

---

## 3. Mechanism account, per axis (decomposition of the registered statistic; first-order = c-weighted; exact = E11)

**b-axis.** measured −1.634 ± 0.444 →
 − [missing θ-dependence of Σ^φ: ⟨c_i·C_b⟩ ≈ −1.35 (E10, E11)] → **−0.268 ± 0.431** after exact correction (E11);
 the remaining z_g-structure (corrected: −14.2 ± 2.5 at z_g < 0.075 → +3.6 ± 0.3 at 0.25–0.39, `f12_out.json`)
 is the ball/tilt geometry of the impostor mixture (E14, E15) and cancels in the pool within noise; on the enlarged ball
 the c-weighted residual is −0.78 ± 0.47 (E12).
 Sign check against the venue law: ∂_b ln Σ^φ < 0 because S̄_φ falls with z (⟨∂_z ln S̄_φ⟩ = −2.4, E3): shifting every
 catalogue kernel up lowers its survival, and the un-normalised likelihood rewards b < 0 — the measured sign.

**s-axis.** measured −0.0872 ± 0.0123 →
 − [divisor: ⟨c·C_s⟩ ≈ −0.015] → −0.0728 ± 0.0122 (E11) →
 − [ball truncation: E9/E12, −0.07 → ≈ 0] → **−0.005 ± 0.011** (E12), while the unweighted catalogue-leg residual
 (+0.036 ± 0.016) equals the secant's intrinsic O(Δ²) bias (E13). Sign check: the truncated impostor mixture keeps only
 candidates whose listed z lies within ~1.45σ_g of the GW window and within 1.5σ of the sky centroid — the surviving
 candidates are, by selection, closer than 1σ to the data, so the likelihood prefers narrower kernels (s < 1).

**With-BH channel.** Z_b = +0.38 is dilution, not self-consistency: the with-BH catalogue-leg secants are
+11.31 ± 0.78 (Z +14.5) and +0.229 ± 0.029 (Z +7.9) with catalogue share median 0.215 (`f1_out.json`, `f5_out.json`);
on the 1-D b0i venue the observed mass is the donor Fisher row's (unlinked to the host — the object b0i2d fixes) and
with-BH true-host recovery is 25.7–36.7 % (P6 2D counters) — uninterpretable, invariant 12 stands.

---

## 4. Method items (2)–(4), answered

**(2) Self-consistency / implied (b_true, s_true).** Kernel-level truth is (0,1): the generator's Gaussian is unshifted and
unscaled (E1), the bare pull has SD 0.963 and mean +0.27 (the k_g tilt, E2/E15), and on all 800 draws the pull against
the full law k·S̄_φ is +0.06 ± 0.035 (E4). The *effective* generator-vs-off-kernel moments are
**b_eff = −0.00263 ± 0.00005, s_eff = 0.9921 ± 0.0002** (E3) — the S̄_φ tilt inside the kernel, which a θ-normalised
divisor absorbs exactly (E11). The estimator's *preferred* θ on this venue (curvature) is b̂ = −0.0079 ± 0.0033,
ln ŝ = −0.327 ± 0.090 before correction and **b̂ = −0.0013 ± 0.0033, ln ŝ = −0.27 ± 0.09** after the divisor
correction (S″ from `f1_out.json`, S′ from E11) — venue artefacts (truncation + secant bias), not truth values.
Predicted vs measured signs: b negative ✓ (E10), s negative ✓ (E9 mechanism), both without any hook defect.

**(3) Window/floors/S̄_φ/secant.** The estimator's ±4·s·σ host window and the 1e-6 floor are matched by the generator at
θ = (0,1) (E1); at s = 1/√2 only 0.35 ± 0.01 % of the generator's kernel mass falls outside the narrowed window
(`f4_out.json` `mass_outside_smin_window`) — PA-HIER-17's tail effect would push score_s POSITIVE and is not the
observed sign. The 1e-10 floor lives only in site 2.3's smeared kernel (not exercised here). The secant is symmetric
in ln s (±0.3466) and the relabel changes Z by nothing; but the secant carries an intrinsic +0.046/event bias (E13)
in either label — a registration defect, not a hook defect. The S̄_φ factor enters only through the divisor (E8, E10).

**(4) PARITY.** By column: the residual is in every per-event numerator (`L_cat_no_bh`, `B_num`, `B_num_wbh`) and not in
α_G^φ/D̃^φ/Σ^φ (E18); by event table: only `z_true`/`obs_d_L` moved (E19). Value-changing paths at the default since the
bank (`71b52e9c`, 2026-08-23): `d40fe5c8` (θ-hook, identity-skip at (0,1); AND the harness hardening
`_B0I_ZTRUE_GRID_N` 401 → 4001 — the no-BH residual's cause), `cf4f8a2a` (symmetric mass filter, 2026-08-25 — the
with-BH candidate-list change: 91–113 events per seed with |Δ ln L_cat_with_bh| > 0.01), `1f003da6` (s placement
before the σ_pv fold — a no-op while `SIGMA_V_PEC_KM_S = 0`), `0b308828` (mass-window geometry flag, default
byte-identical, `mass_filter_k = 1.5` = the prior `sigma_multiplier`), `901653a1` (harness pass-through, identity
defaults). **PARITY-CODE-DELTA: RESOLVED as a comparand (generator-grid) delta; the estimator's no-BH path is not
implicated.** The driver's per-run byte-identity check against the 2026-08-23 bank is ill-posed at the 1e-5 event level
and should compare against a bank regenerated at the current generator grid (PA-HIER-31 (f) "one re-run" — the
re-run must be of the *bank*, not of the node).

---

## 5. The single cheapest decisive test (registered form) — and what was already decided at zero compute

* **Decided here at zero compute.** HOOK-DEFECT vs VENUE-LAW is settled by E7 (twin exact to 1e-12) and E11 (the
  registered b-statistic is null once the divisor is θ-normalised: Z −0.62). No compute test can add discrimination on
  the b-axis beyond this.
* **Cheapest registered-form compute confirmation with existing flags:** re-run S0-A on the four seeds with the
  production-default numerator (`catalogue_numerator_survival="phi"`, the `bt` flags; driver constant at
  `hier_s0_driver.py:94`) in the same `sites2.2_nosmear` form. **Registered prediction (falsifiable):** per-event
  |Δ score_b| ≤ 0.05 and |Δ score_s| ≤ 1e-3 relative to the banked `bc` nodes; pooled Z_b, Z_s unchanged within 0.1
  (E8). Cost: 20 cells × 61–74 s (`s0a_full_output.json` `elapsed_s`, mean ≈ 65 s/cell) ≈ **22 min of evaluate wall at `--jobs 1`; the whole S0-A pass (venue builds included) took 2960 s wall = 11.5 nominal CPU-h at cpu_per_job = 14** (`s0a_full_output.json`). Value: closes the "off-flag" hypothesis in the
  registered instrument itself; it does not change the verdict.
* **The decisive mechanism test needs code (NEEDS-CODE, harness-or-estimator):** a θ-dependent no-BH divisor
  Σ^φ(θ) = Σ_g w_g S̃_g(θ) (site 2.3 extended to the phi-table branch, or the driver applying ρ(θ) post hoc from
  `kernel_smeared_survival` over the pool — a per-node scalar, exactly E11's operation). Prediction: Z_b → |Z_b| < 1
  (−0.27 ± 0.43 measured post hoc), Z_s → −0.073 ± 0.012 (truncation remains). Cost if built estimator-side: one
  full-pool smeared pass per node (UNMEASURED; band estimate 1–3 min from the chunked `kernel_smeared_survival` the b0i venue build already runs once per seed) + the same 20 cells ≈ 35–60 min wall — a band, not a point (A11).
  The s-axis truncation test additionally needs the sky-cone radius as a flag (`bayesian_statistics.py:4869`,
  hardcoded 1.5): prediction Z_s → −0.5 ± 1 (E12); cost ≈ 3–6× per cell (6× more candidates, E9) ≈ 1.5 h wall.

---

## 6. Bearing on B4.2 KW-Q1 (rides the same site-2.2 hook on the FT config, `sites2.2_nosmear`, "phi" numerator)

1. **Same divisor gap, quantified: immaterial to R.** The un-normalised factor ρ(s;h) multiplies L_cat(s) in
   `s_imp,i(s) = Δ_h ln[(βL_cat(s) + B)/B]`; its contribution to `S(√2) − S(1/√2)` is +0.039 per unit catalogue share
   (E21), i.e. ≈ +0.02–0.03 in R against |S(1)| ≈ 0.80 (`CLAIM_IMPOSTOR_DRAG_20260829.md` §1.3, q1 mean s_imp −0.798).
   Disclose it; if R lands within ±0.05 of the 0.2 or 0.5 edge, apply the zero-compute post-hoc correction (the same
   per-(s,h) scalar, `f10_kwq1.py`) before reading the band.
2. **Same truncation and c-weighting — but on FT they are the object measured, not a contamination.** On FT every
   catalogue candidate is an impostor (population hosts, `in_catalog = False`), so `s_imp(s)` measures the truncated
   impostor mixture's width response. E9 shows on b0i that the catalogue-leg s-response flips sign when the ball is
   enlarged; therefore an OWNS verdict must be attributed to "a kernel-width-class object **of the candidate-window
   truncated impostor mixture**", not to a photo-z error misstatement; INERT is unaffected. The KW-Q1 diagnostics CSVs
   carry no candidate counts — the enlarged-ball counterfactual (a sky-cone flag) is the follow-up if OWNS.
3. The b-axis, the negative-centre edge case (b = 0 in KW-Q1) and the with-BH channel do not enter KW-Q1.
4. KW-Q1's within-run comparison (paired, identical events) is **not invalidated** by anything found here; its
   T-ID/ENG gates are unaffected (the hook is exact, E7).

---

## 7. What this note does and does not license (scope, blindness, cap)

* Does not lift the B0-A′ STOP; re-classifies its cause as **INSTRUMENT-FORM (registration + hook-placement) and
  VENUE-LAW (b0i candidate-ball truncation; all-catalogue generator vs a w_G = 0.062 estimator mixture; d_L-dependent
  Fisher-quality selection)**. Per §4.5's INSTRUMENT-DEFECT row the fix routes: divisor θ-dependence → `/physics-change`
  (trigger file); secant form and the z-binning read → registration amendment (a fresh author [RULE]); the harness
  comparand for GATE PARITY → instrumentation.
* Does not license any Stage-P/F, S0-B or C1/C3 launch; does not touch the PA-HIER-31 REVISION NOTE 2 R1′/R2′
  contradiction (smear form) — that remains for the author.
* Blindness carried: the divisor correction (E10/E11) uses Σ^φ(θ) built from the b0i draw weights / a 200k pool
  subsample, not a re-evaluated estimator; the enlarged-ball twin is a numpy counterfactual, not an estimator run; the
  with-BH channel is untouched; the S0-B production venue has no truth-θ and inherits (i) verbatim (the same θ-inert
  divisor), so an S0-B non-null of order ⟨c⟩·C_b ≈ −1.3 per unit b is **predicted by construction** and must be
  subtracted (or the divisor θ-hooked) before any LEVER-LIVE read — registered here, REPORTED-ONLY.
* Reader's caveat honoured: the localisation is to the *no-BH catalogue leg under the 2.2/unsmeared form*; sites 2.1
  (scalar twin, not dispatched here) and 2.3 (with-BH globals) are not exercised by this run.

---

## 8. Provenance

* Inputs: `hier_s0_registered_run/s0a_seed9001{01..04}/node_{truth,b_plus,b_minus,s_plus,s_minus}_sites2.2_nosmear/simulations/{diagnostics/event_likelihoods.csv,cramer_rao_bounds.csv,fisher_quality.csv}`,
  `…/selection_tables_h_0_73.json`, `s0a_full_output.json`, `s0a_score_output.json`; `p3_b0_work/bc_9001{01..04}_work/seed*/…` (bank, commit `71b52e9c`, 2026-08-23) and `bc_*.log` (P6 counters);
  `darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` (md5 pin PASS); completeness cache + injection pool via `build_bsel_selection_objects`.
* Code read (no edits): `bayesian_statistics.py` `:2906/:2916` (Σ^φ branch order), `:3395,:4422` (exclusion), `:4380-4418` (cov blocks), `:4848-4870` (ball caller), `:5049-5054,:5137-5166` (host union + `_rate_weight`), `:5187-5191` (θ-inert divisor), `:5297` (`L_cat = Σw N/Σ^φ`), `:5770-5776` (assembly), `:7091-7101` (site-2.2 hook), `:7140-7330` (windows, Z_g, numerator); `correspondence_1d.py` `:1168` (grid), `:1248-1345` (S̃), `:1446-1512` (z_true draw), `:2104-2136` (b0i branch), `:2247-2249` (obs d_L), `:2734-2965` (in-process runner); `handler.py` `:558-700,:1422-1440`; `physical_relations.py` `:546-566`; `hier_s0_driver.py` `:94-95,:122,:392-393`; `p3_b0_identity_test.py` `:998`.
* Instruments + outputs: `fanout1_20260829/b1_1_forensic_work/f{1..12}_*.py`, `f*_out.json`, `f7_events.csv`, `f6_alldraws.csv`, `f9_alldraws_C.csv` (per-event tables; the f8 slice pickles were not copied).
* Authorization stamp: **launched under rows #222/#223 — charter node B1.1**. Append-only; no git operations; no source edits.
