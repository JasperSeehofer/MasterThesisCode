# pp_coverage Gray-mixture sweep — VERDICT (2026-07-11)

**Provenance:** EXP-41 / handoff item N-1 (`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`);
quick task `260711-07n-pp-coverage-gray-mixture` (code at `0f6f914` on
`physics/zero-host-completion-fallback`); RUNBOOK.md in this directory (grid, commands,
pre-registered criteria — followed as recorded). Gray mode = full Gray et al. (2020,
arXiv:1908.06050, Eqs. 29+32) mixture `(beta_G*L_cat_i + B_num)/D` for host-found events
with the per-host selection denominator `D_g_i` of Eqs. A.9/A.10 (production commit
`713fbd1` analog); zero-host events keep the issue-#29 pure-completion `B_num/D`.
Baseline A/B: the L-A two-branch sweep `results/pp_coverage_deepvenue_20260710/SUMMARY.md`.

## VERDICT: STILL BIASED — the full Gray mixture does NOT restore calibration at deep incompleteness; it makes the high bias WORSE

Against the pre-registered criteria (cov68 within ±0.085 of 0.68 AND |Δmap_mean vs truth|
< 2·SEM, SEM = map_std/√120, across the truncated cells zs ∈ {0.2, 0.3}):
**12/12 gray truncated cells × truths fail BOTH criteria.**

- **Gray amplifies, not compensates:** at every truncated cell the gray MAP bias is
  larger than the matching two-branch bias (Δbias +0.0005…+0.0909). Worst regime:
  (zs=0.2, σ_z=0.035) → bias **+0.123 / +0.120** in h for truths 0.62/0.72 (+20%/+17%
  of truth; two-branch had +0.032/+0.037) with cov68 = 0.000 and the 0.84 ensemble
  railed at 1.000. The (·, ·, 0.84) cells look "only" +0.020 biased because the grid
  edge at h=0.86 clips the ensemble (rail 0.83–1.00).
- **σ_z dependence is dramatic in gray mode:** at zs=0.2 the 0.62-truth bias grows
  +0.024 → +0.123 from σ_z=0.015 → 0.035 (the two-branch growth was +0.012 → +0.032).
  The B_num admixture inside host events grows with the kernel mass leaking past the
  support edge — exactly the σ_z-sensitive composition effect L-A hypothesized, but
  with the opposite sign of the hoped-for compensation.
- **Gray in-catalogue machinery is healthy in the complete-catalogue limit:** the
  zs=1.0 controls (which degenerate to the per-host local-ratio `N_i/D_g_i`; B_num
  empty, beta_G = D cancels) show |bias| ≤ 0.004 and zero/near-zero rail. Mild
  undercoverage at σ_z=0.035 (cov68 0.55–0.63 vs nominal 0.68, band ±0.085) — the
  local-ratio form is slightly overconfident, worth remembering, but nothing like the
  truncated-cell collapse. zs=0.5 (comp_frac ≈ 0) reproduces its control — truncation
  machinery inert where it should be.
- **Tilt diagnostics (N-2a) localize the mechanism:** in the controls the host branch
  tilts NEGATIVE at truth (d logL_host/dh = −26…−182 per realization, the healthy
  counterweight). In the truncated gray cells the host branch FLIPS POSITIVE
  (+47…+166) — i.e. after the B_num admixture the host-found events *join* the
  completion branch (+113…+401) in preferring high h instead of compensating it.
  The mixture composition converts the in-catalogue events from counterweight to
  co-conspirator.

### Conditioned contrast (N-2b): conditioning does NOT rescue it either

Membership-conditioned inverse (`N_i/beta_G` in catalogue, `B_num/beta_Gbar` outside)
on the 4 deepest cells:

| z_support | sigma_z | h_true | cov50 | cov68 | cov90 | rail_fraction | MAP mean | MAP bias | comp_frac | dlogL_dh_host_mean | dlogL_dh_completion_mean | two_branch bias | gray bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.192 | 0.267 | 0.467 | 0.000 | 0.6330 | +0.0130 | 0.709 | +240.494 | +14.056 | +0.0123 | +0.0236 |
| 0.2 | 0.015 | 0.72 | 0.158 | 0.300 | 0.517 | 0.000 | 0.7353 | +0.0153 | 0.787 | +156.627 | +9.646 | +0.0144 | +0.0283 |
| 0.2 | 0.015 | 0.84 | 0.167 | 0.242 | 0.350 | 0.483 | 0.8540 | +0.0140 | 0.848 | +93.973 | +18.995 | +0.0139 | +0.0188 |
| 0.2 | 0.035 | 0.62 | 0.000 | 0.033 | 0.142 | 0.000 | 0.6582 | +0.0382 | 0.709 | +271.647 | +14.056 | +0.0317 | +0.1226 |
| 0.2 | 0.035 | 0.72 | 0.033 | 0.033 | 0.183 | 0.000 | 0.7642 | +0.0442 | 0.787 | +175.862 | +9.646 | +0.0368 | +0.1197 |
| 0.2 | 0.035 | 0.84 | 0.033 | 0.042 | 0.200 | 0.925 | 0.8592 | +0.0192 | 0.848 | +106.171 | +18.995 | +0.0194 | +0.0200 |
| 0.3 | 0.015 | 0.62 | 0.417 | 0.533 | 0.767 | 0.000 | 0.6245 | +0.0045 | 0.219 | +378.976 | −75.996 | +0.0046 | +0.0083 |
| 0.3 | 0.015 | 0.72 | 0.092 | 0.142 | 0.450 | 0.000 | 0.7285 | +0.0085 | 0.390 | +348.107 | +3.629 | +0.0081 | +0.0136 |
| 0.3 | 0.015 | 0.84 | 0.083 | 0.142 | 0.242 | 0.150 | 0.8513 | +0.0113 | 0.551 | +250.423 | +26.439 | +0.0108 | +0.0166 |
| 0.3 | 0.035 | 0.62 | 0.042 | 0.075 | 0.183 | 0.000 | 0.6391 | +0.0191 | 0.219 | +454.524 | −75.996 | +0.0153 | +0.0297 |
| 0.3 | 0.035 | 0.72 | 0.000 | 0.017 | 0.058 | 0.000 | 0.7481 | +0.0281 | 0.390 | +407.022 | +3.629 | +0.0235 | +0.0489 |
| 0.3 | 0.035 | 0.84 | 0.008 | 0.008 | 0.008 | 0.950 | 0.8597 | +0.0197 | 0.551 | +277.986 | +26.439 | +0.0195 | +0.0200 |

**N-2b mapping (pre-registered):** "if conditioned calibrates where gray does not, the
defect is w_G(h)=beta_G/D bookkeeping, not the completion integral." Conditioned does
**NOT** calibrate — biases +0.005…+0.044, comparable to (slightly worse than) the
two-branch clean limit and far better than gray, but still failing both criteria in
all 12 cells. So the deep-incompleteness high bias is **not merely w_G(h) bookkeeping**:
even the rigorous membership-conditioned inverse carries it. Note how conditioning
*relocates* the tilt: the conditioned completion branch is nearly flat at truth
(−76…+26 — dividing B_num by beta_Gbar removes most of its h-tilt), yet the host
branch (÷ beta_G) then tilts strongly positive (+94…+455). The high preference is
conserved under re-bookkeeping — it lives in the *joint composition* of a
selection-truncated catalogue with support-edge events, which N-2 (mechanism
decomposition: σ_z isolation, prior-sensitivity N-3) must corner further.

### Implications for the ledger

1. **N-1 fork adjudicated:** the L-A bias is NOT a clean-limit artifact that the full
   Gray composition absorbs — in this harness the faithful Eqs. 29+32 mixture is
   *worse* than the clean limit at 22–85% completion-governed fractions. Production
   composition remains **suspect at deep incompleteness**; depth+fallback is NOT
   demonstrated safe at the estimator level.
2. **EXP-40 prediction sharpened:** the seed1000 re-eval (58% zero-host) should be
   watched for an interior-but-biased-HIGH posterior in BOTH regimes; if production
   mirrors the harness, the full mixture (post-#29) may over-shoot MORE than a pure
   two-branch split would.
3. **Decision D1 (issue #30):** further quantitative support for explicit z-truncation
   as the *robustness bound* — calibration is exact where completion_fraction ≈ 0 in
   every mode tested (two_branch, gray, conditioned). Per the user directive this is
   evidence input, not a default answer: the mechanism investigation (N-2/N-3)
   continues.
4. **Caveat for external comparison:** gwcosmo/ICAROGW-class analyses operate
   calibrated at percent-level completeness. This harness's mixture uses a single
   effective host per event (single-host limit) rather than a full in-catalogue galaxy
   sum, and an unnormalized w_pop measure shared by numerator and D. Whether the
   discrepancy is a defect of OUR composition or of the single-host reduction is
   exactly the N-2 question; do not read this verdict as "Gray et al. is wrong".

## Gray per-cell × truth table

Columns as in the two-branch SUMMARY plus the two per-branch tilt diagnostics
(mean over realizations of d(logL_branch)/dh at the grid node nearest h_true;
null = branch had no events).

| z_support | sigma_z | h_true | cov50 | cov68 | cov90 | rail_fraction | MAP mean | MAP bias | completion_fraction | dlogL_dh_host_mean | dlogL_dh_completion_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.033 | 0.067 | 0.175 | 0.000 | 0.6436 | +0.0236 | 0.709 | +104.416 | +272.315 |
| 0.2 | 0.015 | 0.72 | 0.042 | 0.075 | 0.217 | 0.000 | 0.7483 | +0.0283 | 0.787 | +73.779 | +165.687 |
| 0.2 | 0.015 | 0.84 | 0.017 | 0.042 | 0.125 | 0.833 | 0.8588 | +0.0188 | 0.848 | +47.409 | +112.838 |
| 0.2 | 0.035 | 0.62 | 0.000 | 0.000 | 0.000 | 0.017 | 0.7426 | +0.1226 | 0.709 | +166.152 | +272.315 |
| 0.2 | 0.035 | 0.72 | 0.000 | 0.000 | 0.008 | 0.475 | 0.8397 | +0.1197 | 0.787 | +116.651 | +165.687 |
| 0.2 | 0.035 | 0.84 | 0.000 | 0.000 | 0.008 | 1.000 | 0.8600 | +0.0200 | 0.848 | +75.951 | +112.838 |
| 0.3 | 0.015 | 0.62 | 0.158 | 0.267 | 0.500 | 0.000 | 0.6283 | +0.0083 | 0.219 | +92.507 | +400.923 |
| 0.3 | 0.015 | 0.72 | 0.025 | 0.042 | 0.125 | 0.000 | 0.7336 | +0.0136 | 0.390 | +111.241 | +399.013 |
| 0.3 | 0.015 | 0.84 | 0.017 | 0.025 | 0.050 | 0.492 | 0.8566 | +0.0166 | 0.551 | +95.129 | +292.732 |
| 0.3 | 0.035 | 0.62 | 0.000 | 0.000 | 0.025 | 0.000 | 0.6497 | +0.0297 | 0.219 | +84.044 | +400.923 |
| 0.3 | 0.035 | 0.72 | 0.000 | 0.000 | 0.000 | 0.000 | 0.7689 | +0.0489 | 0.390 | +141.359 | +399.013 |
| 0.3 | 0.035 | 0.84 | 0.000 | 0.000 | 0.000 | 1.000 | 0.8600 | +0.0200 | 0.551 | +128.777 | +292.732 |
| 0.5 | 0.015 | 0.62 | 0.667 | 0.675 | 0.858 | 0.000 | 0.6176 | −0.0024 | 0.000 | −181.942 | null |
| 0.5 | 0.015 | 0.72 | 0.525 | 0.583 | 0.908 | 0.000 | 0.7187 | −0.0013 | 0.000 | −97.243 | +16.515 |
| 0.5 | 0.015 | 0.84 | 0.533 | 0.767 | 0.917 | 0.000 | 0.8394 | −0.0006 | 0.008 | −56.736 | +27.005 |
| 0.5 | 0.035 | 0.62 | 0.558 | 0.633 | 0.792 | 0.008 | 0.6161 | −0.0039 | 0.000 | −70.993 | null |
| 0.5 | 0.035 | 0.72 | 0.417 | 0.567 | 0.908 | 0.000 | 0.7181 | −0.0019 | 0.000 | −29.857 | +16.515 |
| 0.5 | 0.035 | 0.84 | 0.425 | 0.675 | 0.867 | 0.000 | 0.8399 | −0.0001 | 0.008 | −25.661 | +27.005 |
| 1.0 | 0.015 | 0.62 | 0.667 | 0.675 | 0.858 | 0.000 | 0.6176 | −0.0024 | 0.000 | −181.945 | null |
| 1.0 | 0.015 | 0.72 | 0.525 | 0.567 | 0.908 | 0.000 | 0.7186 | −0.0014 | 0.000 | −99.684 | null |
| 1.0 | 0.015 | 0.84 | 0.517 | 0.725 | 0.925 | 0.000 | 0.8384 | −0.0016 | 0.000 | −81.114 | null |
| 1.0 | 0.035 | 0.62 | 0.558 | 0.633 | 0.792 | 0.008 | 0.6161 | −0.0039 | 0.000 | −71.037 | null |
| 1.0 | 0.035 | 0.72 | 0.417 | 0.550 | 0.925 | 0.000 | 0.7180 | −0.0020 | 0.000 | −32.409 | null |
| 1.0 | 0.035 | 0.84 | 0.475 | 0.617 | 0.858 | 0.000 | 0.8372 | −0.0028 | 0.000 | −43.529 | null |

## Side-by-side Δ: gray vs matching two-branch cell (L-A baseline)

Baseline values from `results/pp_coverage_deepvenue_20260710/pp_zs{ZS}_sz{SZ}_volume.json`
(same grid, same seed, same realizations).

| z_support | sigma_z | h_true | tb cov68 | gray cov68 | Δcov68 | tb map_bias | gray map_bias | Δmap_bias |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.267 | 0.067 | −0.200 | +0.0123 | +0.0236 | +0.0113 |
| 0.2 | 0.015 | 0.72 | 0.267 | 0.075 | −0.192 | +0.0144 | +0.0283 | +0.0138 |
| 0.2 | 0.015 | 0.84 | 0.233 | 0.042 | −0.192 | +0.0139 | +0.0188 | +0.0049 |
| 0.2 | 0.035 | 0.62 | 0.042 | 0.000 | −0.042 | +0.0317 | +0.1226 | +0.0909 |
| 0.2 | 0.035 | 0.72 | 0.050 | 0.000 | −0.050 | +0.0368 | +0.1197 | +0.0829 |
| 0.2 | 0.035 | 0.84 | 0.033 | 0.000 | −0.033 | +0.0194 | +0.0200 | +0.0006 |
| 0.3 | 0.015 | 0.62 | 0.542 | 0.267 | −0.275 | +0.0046 | +0.0083 | +0.0037 |
| 0.3 | 0.015 | 0.72 | 0.192 | 0.042 | −0.150 | +0.0081 | +0.0136 | +0.0055 |
| 0.3 | 0.015 | 0.84 | 0.175 | 0.025 | −0.150 | +0.0108 | +0.0166 | +0.0058 |
| 0.3 | 0.035 | 0.62 | 0.083 | 0.000 | −0.083 | +0.0153 | +0.0297 | +0.0144 |
| 0.3 | 0.035 | 0.72 | 0.017 | 0.000 | −0.017 | +0.0235 | +0.0489 | +0.0254 |
| 0.3 | 0.035 | 0.84 | 0.008 | 0.000 | −0.008 | +0.0195 | +0.0200 | +0.0005 |
| 0.5 | 0.015 | 0.62 | 0.700 | 0.675 | −0.025 | −0.0019 | −0.0024 | −0.0004 |
| 0.5 | 0.015 | 0.72 | 0.658 | 0.583 | −0.075 | −0.0015 | −0.0013 | +0.0001 |
| 0.5 | 0.015 | 0.84 | 0.708 | 0.767 | +0.058 | −0.0009 | −0.0006 | +0.0003 |
| 0.5 | 0.035 | 0.62 | 0.758 | 0.633 | −0.125 | −0.0030 | −0.0039 | −0.0009 |
| 0.5 | 0.035 | 0.72 | 0.675 | 0.567 | −0.108 | −0.0017 | −0.0019 | −0.0002 |
| 0.5 | 0.035 | 0.84 | 0.692 | 0.675 | −0.017 | −0.0010 | −0.0001 | +0.0009 |
| 1.0 | 0.015 | 0.62 | 0.700 | 0.675 | −0.025 | −0.0019 | −0.0024 | −0.0004 |
| 1.0 | 0.015 | 0.72 | 0.667 | 0.567 | −0.100 | −0.0015 | −0.0014 | +0.0002 |
| 1.0 | 0.015 | 0.84 | 0.725 | 0.725 | +0.000 | −0.0014 | −0.0016 | −0.0001 |
| 1.0 | 0.035 | 0.62 | 0.758 | 0.633 | −0.125 | −0.0030 | −0.0039 | −0.0009 |
| 1.0 | 0.035 | 0.72 | 0.675 | 0.550 | −0.125 | −0.0018 | −0.0020 | −0.0002 |
| 1.0 | 0.035 | 0.84 | 0.675 | 0.617 | −0.058 | −0.0024 | −0.0028 | −0.0003 |

(Reminder: at zs ∈ {0.5, 1.0} the gray "host" branch is the local-ratio `N_i/D_g_i`,
so small control-level differences vs two-branch are expected and observed —
|Δmap_bias| ≤ 0.0009 there.)

## Pre-registered verdict evaluation (truncated gray cells, zs ∈ {0.2, 0.3})

| z_support | sigma_z | h_true | cov68 | cov68 in 0.68±0.085? | \|map_bias\| | 2·SEM | bias < 2·SEM? |
|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.067 | NO | 0.0236 | 0.0017 | NO |
| 0.2 | 0.015 | 0.72 | 0.075 | NO | 0.0283 | 0.0023 | NO |
| 0.2 | 0.015 | 0.84 | 0.042 | NO | 0.0188 | 0.0006 | NO |
| 0.2 | 0.035 | 0.62 | 0.000 | NO | 0.1226 | 0.0079 | NO |
| 0.2 | 0.035 | 0.72 | 0.000 | NO | 0.1197 | 0.0048 | NO |
| 0.2 | 0.035 | 0.84 | 0.000 | NO | 0.0200 | 0.0000 | NO |
| 0.3 | 0.015 | 0.62 | 0.267 | NO | 0.0083 | 0.0008 | NO |
| 0.3 | 0.015 | 0.72 | 0.042 | NO | 0.0136 | 0.0010 | NO |
| 0.3 | 0.015 | 0.84 | 0.025 | NO | 0.0166 | 0.0007 | NO |
| 0.3 | 0.035 | 0.62 | 0.000 | NO | 0.0297 | 0.0016 | NO |
| 0.3 | 0.035 | 0.72 | 0.000 | NO | 0.0489 | 0.0023 | NO |
| 0.3 | 0.035 | 0.84 | 0.000 | NO | 0.0200 | 0.0000 | NO |

12/12 fail both ⇒ **STILL BIASED**. (The 0.84 rows' 2·SEM ≈ 0 reflects rail pile-up:
map_std → 0 when 83–100% of realizations sit on the h=0.86 grid edge.)

## Carried caveats (verbatim from the deep-venue SUMMARY, with status update)

1. **1D-channel only** — the 2D (+0.057) question is NOT covered by this harness.
2. **Single-host clean limit** — production host-found events ALSO carry a `B_num`
   admixture in the mixture; this harness omits that, so ONLY the zero-host branch is
   the exact production analog. **STATUS UPDATE (this sweep):** gray mode now RESTORES
   the previously-omitted `B_num` admixture on host-found events — the escape hatch
   this caveat flagged has been tested and it does NOT restore calibration (it worsens
   the high bias). The two-branch clean limit remains available for A/B via
   `--mixture-mode two_branch`.
3. **Hard truncation** (`z_support` step) vs production's soft M_BH-prune truncation
   of the effective catalogue.
