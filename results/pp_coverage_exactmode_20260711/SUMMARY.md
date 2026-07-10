# pp_coverage exact membership-truncated-kernel sweep — VERDICT (2026-07-11)

**Provenance:** quick task `260711-117-pp-coverage-exact-kernel` (EXP-41-exact /
handoff items N-2c + N-2d, `.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`);
code at `6a3c8ab` on `physics/zero-host-completion-fallback` (adds
`mixture_mode="exact"` + `--n-z-quad` to `master_thesis_code/validation/pp_coverage.py`);
RUNBOOK.md in this directory (grid, commands, pre-registered prediction — written
BEFORE any run, followed as recorded). Exact mode = membership-truncated exact
kernel: detection conditioned once via `1/D(h)` with no p_det in the numerator
(Mandel, Farr & Gair 2019, arXiv:1809.02063), catalogue membership
`G = 1[z_true < z_support]` observed, host numerator truncated at `z_support`,
zero-host events keep `B_num/D` (Gray et al. 2020, arXiv:1908.06050, Eqs. 29+32
support tiling). Baselines A/B: the L-A two-branch sweep
`results/pp_coverage_deepvenue_20260710/` and the EXP-41 gray sweep
`results/pp_coverage_graymix_20260711/`.

**Anti-repetition:** gray and conditioned were adjudicated STILL BIASED in
260711-07n (`results/pp_coverage_graymix_20260711/SUMMARY.md`, 12/12 fail) —
not re-litigated here; their committed JSONs are used for the Δ tables only.

## VERDICT: STILL BIASED against the strict pre-registered criteria (1/12 rows pass) — but the pre-registered mechanism is CONFIRMED as the DOMINANT component: exact truncation removes the entire σ_z-dependent leak, leaving a 3–8× smaller, σ_z-INDEPENDENT completion-branch residual (+0.002…+0.005 in h)

**The pre-registered CALIBRATED prediction did NOT hold** (criteria: cov68
within ±0.085 of 0.68 AND |map_bias| < 2·SEM, SEM = map_std/√120, across the 12
truncated cell×truth rows zs ∈ {0.2, 0.3}): **1/12 pass both** (7/12 pass the
cov68 band; 1/12 the bias gate). Honest reading of the failure mode:

- **The membership-support leak IS the dominant mechanism** (prediction's causal
  claim confirmed): removing the above-edge kernel mass cuts the deep-venue bias
  from +0.012…+0.037 (two_branch) and +0.008…+0.123 (gray) to **+0.002…+0.005**,
  restores cov68 from 0.008–0.542 to **0.483–0.708**, and collapses the 0.84-truth
  rail from 0.45–0.92 (two_branch) / 0.83–1.00 (gray) to **0.00–0.19**.
- **The σ_z-ladder is the smoking gun** (Table 3): two_branch bias climbs
  +0.0033 → +0.0368 as σ_z goes 0.002 → 0.035 (the leak grows with kernel mass
  past the edge); exact is FLAT in σ_z (+0.0023…+0.0046 at every rung) and the
  two modes CONVERGE at σ_z → 0 (at σ_z=0.002 they differ by ≤ 0.0004) — exactly
  the signature of a kernel-support leak, now removed.
- **What survives is σ_z-independent** and therefore a DIFFERENT, smaller
  mechanism: a +0.002…+0.005 high residual (0.3–0.6% of truth) that grows with
  completion fraction and truth, statistically significant against 2·SEM
  (0.0007–0.0022). The tilt diagnostics localize it: the exact host branch is
  restored as the healthy NEGATIVE counterweight at truth (−72…−435, vs
  two_branch/gray truncated cells where it flipped positive), while the
  completion branch keeps its POSITIVE tilt (+113…+401). The residual lives in
  the completion-branch composition (B_num/D with few counterweight events) —
  the N-3 prior-sensitivity probe is the designed next step for it.
- **Controls clean:** zs=0.5/1.0 exact reproduces the two-branch controls to
  the displayed precision (Δ ≤ 0.0025 in bias at the 0.8% completion cell,
  0.0000 at zs=1.0); truncation machinery inert where it should be.

## Pre-registered verdict evaluation (exact, truncated cells zs ∈ {0.2, 0.3})

| z_support | sigma_z | h_true | cov68 | cov68 in 0.68±0.085? | \|map_bias\| | 2·SEM | bias < 2·SEM? | both |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.692 | YES | 0.0023 | 0.0013 | NO | FAIL |
| 0.2 | 0.015 | 0.72 | 0.550 | NO | 0.0034 | 0.0015 | NO | FAIL |
| 0.2 | 0.015 | 0.84 | 0.608 | YES | 0.0042 | 0.0019 | NO | FAIL |
| 0.2 | 0.035 | 0.62 | 0.708 | YES | 0.0026 | 0.0015 | NO | FAIL |
| 0.2 | 0.035 | 0.72 | 0.575 | NO | 0.0046 | 0.0019 | NO | FAIL |
| 0.2 | 0.035 | 0.84 | 0.517 | NO | 0.0042 | 0.0022 | NO | FAIL |
| 0.3 | 0.015 | 0.62 | 0.700 | YES | 0.0003 | 0.0007 | YES | PASS |
| 0.3 | 0.015 | 0.72 | 0.625 | YES | 0.0024 | 0.0008 | NO | FAIL |
| 0.3 | 0.015 | 0.84 | 0.525 | NO | 0.0047 | 0.0011 | NO | FAIL |
| 0.3 | 0.035 | 0.62 | 0.708 | YES | 0.0010 | 0.0010 | NO | FAIL |
| 0.3 | 0.035 | 0.72 | 0.633 | YES | 0.0023 | 0.0011 | NO | FAIL |
| 0.3 | 0.035 | 0.84 | 0.483 | NO | 0.0054 | 0.0013 | NO | FAIL |

1/12 pass both ⇒ formally **STILL BIASED**; contrast gray's 0/12 with biases up
to +0.123 and two_branch's 0/12 with biases up to +0.037.

## Exact per-cell × truth table (set a)

Columns as in the graymix SUMMARY (tilts = mean d(logL_branch)/dh at the grid
node nearest h_true; null = branch had no events).

| z_support | sigma_z | h_true | cov50 | cov68 | cov90 | rail_fraction | MAP mean | MAP bias | completion_fraction | dlogL_dh_host_mean | dlogL_dh_completion_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.567 | 0.692 | 0.883 | 0.000 | 0.6223 | +0.0023 | 0.709 | −208.942 | +272.315 |
| 0.2 | 0.015 | 0.72 | 0.417 | 0.550 | 0.875 | 0.000 | 0.7234 | +0.0034 | 0.787 | −116.869 | +165.687 |
| 0.2 | 0.015 | 0.84 | 0.475 | 0.608 | 0.717 | 0.167 | 0.8442 | +0.0042 | 0.848 | −72.015 | +112.838 |
| 0.2 | 0.035 | 0.62 | 0.592 | 0.708 | 0.858 | 0.000 | 0.6226 | +0.0026 | 0.709 | −223.643 | +272.315 |
| 0.2 | 0.035 | 0.72 | 0.450 | 0.575 | 0.883 | 0.000 | 0.7246 | +0.0046 | 0.787 | −124.629 | +165.687 |
| 0.2 | 0.035 | 0.84 | 0.367 | 0.517 | 0.725 | 0.192 | 0.8442 | +0.0042 | 0.848 | −77.448 | +112.838 |
| 0.3 | 0.015 | 0.62 | 0.692 | 0.700 | 0.950 | 0.000 | 0.6203 | +0.0003 | 0.219 | −368.397 | +400.923 |
| 0.3 | 0.015 | 0.72 | 0.433 | 0.625 | 0.767 | 0.000 | 0.7224 | +0.0024 | 0.390 | −281.261 | +399.013 |
| 0.3 | 0.015 | 0.84 | 0.342 | 0.525 | 0.725 | 0.000 | 0.8447 | +0.0047 | 0.551 | −163.297 | +292.732 |
| 0.3 | 0.035 | 0.62 | 0.508 | 0.708 | 0.925 | 0.000 | 0.6190 | −0.0010 | 0.219 | −434.910 | +400.923 |
| 0.3 | 0.035 | 0.72 | 0.475 | 0.633 | 0.825 | 0.000 | 0.7223 | +0.0023 | 0.390 | −336.749 | +399.013 |
| 0.3 | 0.035 | 0.84 | 0.317 | 0.483 | 0.725 | 0.033 | 0.8454 | +0.0054 | 0.551 | −207.566 | +292.732 |
| 0.5 | 0.015 | 0.62 | 0.700 | 0.700 | 0.900 | 0.000 | 0.6181 | −0.0019 | 0.000 | −172.578 | null |
| 0.5 | 0.015 | 0.72 | 0.617 | 0.658 | 0.900 | 0.000 | 0.7185 | −0.0015 | 0.000 | −104.984 | +16.515 |
| 0.5 | 0.015 | 0.84 | 0.608 | 0.725 | 0.942 | 0.000 | 0.8386 | −0.0014 | 0.008 | −101.935 | +27.005 |
| 0.5 | 0.035 | 0.62 | 0.633 | 0.758 | 0.875 | 0.000 | 0.6170 | −0.0030 | 0.000 | −65.511 | null |
| 0.5 | 0.035 | 0.72 | 0.542 | 0.683 | 0.892 | 0.000 | 0.7179 | −0.0021 | 0.000 | −40.367 | +16.515 |
| 0.5 | 0.035 | 0.84 | 0.517 | 0.700 | 0.875 | 0.000 | 0.8364 | −0.0036 | 0.008 | −93.376 | +27.005 |
| 1.0 | 0.015 | 0.62 | 0.700 | 0.700 | 0.900 | 0.000 | 0.6181 | −0.0019 | 0.000 | −172.575 | null |
| 1.0 | 0.015 | 0.72 | 0.625 | 0.667 | 0.900 | 0.000 | 0.7185 | −0.0015 | 0.000 | −103.720 | null |
| 1.0 | 0.015 | 0.84 | 0.592 | 0.725 | 0.942 | 0.000 | 0.8386 | −0.0014 | 0.000 | −82.047 | null |
| 1.0 | 0.035 | 0.62 | 0.633 | 0.758 | 0.875 | 0.000 | 0.6170 | −0.0030 | 0.000 | −65.403 | null |
| 1.0 | 0.035 | 0.72 | 0.550 | 0.675 | 0.892 | 0.000 | 0.7182 | −0.0018 | 0.000 | −36.151 | null |
| 1.0 | 0.035 | 0.84 | 0.483 | 0.675 | 0.958 | 0.000 | 0.8376 | −0.0024 | 0.000 | −45.204 | null |

Note the host-branch tilt SIGN in the truncated cells: NEGATIVE at truth
(−72…−435) — the exact host branch is the healthy counterweight, where
two_branch/gray had it flipped POSITIVE (co-conspirator with the completion
branch). The remaining high tilt is entirely the completion branch's.

## Side-by-side Δ: exact vs matching two_branch cell (L-A baseline)

Baseline: `results/pp_coverage_deepvenue_20260710/pp_zs{ZS}_sz{SZ}_volume.json`
(same grid/seed/realizations).

| z_support | sigma_z | h_true | tb cov68 | exact cov68 | Δcov68 | tb map_bias | exact map_bias | Δmap_bias |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.267 | 0.692 | +0.425 | +0.0123 | +0.0023 | −0.0100 |
| 0.2 | 0.015 | 0.72 | 0.267 | 0.550 | +0.283 | +0.0144 | +0.0034 | −0.0110 |
| 0.2 | 0.015 | 0.84 | 0.233 | 0.608 | +0.375 | +0.0139 | +0.0042 | −0.0097 |
| 0.2 | 0.035 | 0.62 | 0.042 | 0.708 | +0.667 | +0.0317 | +0.0026 | −0.0290 |
| 0.2 | 0.035 | 0.72 | 0.050 | 0.575 | +0.525 | +0.0368 | +0.0046 | −0.0322 |
| 0.2 | 0.035 | 0.84 | 0.033 | 0.517 | +0.483 | +0.0194 | +0.0042 | −0.0152 |
| 0.3 | 0.015 | 0.62 | 0.542 | 0.700 | +0.158 | +0.0046 | +0.0003 | −0.0043 |
| 0.3 | 0.015 | 0.72 | 0.192 | 0.625 | +0.433 | +0.0081 | +0.0024 | −0.0057 |
| 0.3 | 0.015 | 0.84 | 0.175 | 0.525 | +0.350 | +0.0108 | +0.0047 | −0.0061 |
| 0.3 | 0.035 | 0.62 | 0.083 | 0.708 | +0.625 | +0.0153 | −0.0010 | −0.0163 |
| 0.3 | 0.035 | 0.72 | 0.017 | 0.633 | +0.617 | +0.0235 | +0.0023 | −0.0212 |
| 0.3 | 0.035 | 0.84 | 0.008 | 0.483 | +0.475 | +0.0195 | +0.0054 | −0.0141 |
| 0.5 | 0.015 | 0.62 | 0.700 | 0.700 | +0.000 | −0.0019 | −0.0019 | +0.0000 |
| 0.5 | 0.015 | 0.72 | 0.658 | 0.658 | +0.000 | −0.0015 | −0.0015 | −0.0000 |
| 0.5 | 0.015 | 0.84 | 0.708 | 0.725 | +0.017 | −0.0009 | −0.0014 | −0.0005 |
| 0.5 | 0.035 | 0.62 | 0.758 | 0.758 | +0.000 | −0.0030 | −0.0030 | +0.0000 |
| 0.5 | 0.035 | 0.72 | 0.675 | 0.683 | +0.008 | −0.0017 | −0.0021 | −0.0004 |
| 0.5 | 0.035 | 0.84 | 0.692 | 0.700 | +0.008 | −0.0010 | −0.0036 | −0.0025 |
| 1.0 | 0.015 | 0.62 | 0.700 | 0.700 | +0.000 | −0.0019 | −0.0019 | +0.0000 |
| 1.0 | 0.015 | 0.72 | 0.667 | 0.667 | +0.000 | −0.0015 | −0.0015 | +0.0000 |
| 1.0 | 0.015 | 0.84 | 0.725 | 0.725 | +0.000 | −0.0014 | −0.0014 | +0.0000 |
| 1.0 | 0.035 | 0.62 | 0.758 | 0.758 | +0.000 | −0.0030 | −0.0030 | +0.0000 |
| 1.0 | 0.035 | 0.72 | 0.675 | 0.675 | +0.000 | −0.0018 | −0.0018 | +0.0000 |
| 1.0 | 0.035 | 0.84 | 0.675 | 0.675 | +0.000 | −0.0024 | −0.0024 | +0.0000 |

## Side-by-side Δ: exact vs matching gray cell (EXP-41 baseline)

Baseline: `results/pp_coverage_graymix_20260711/pp_gray_zs{ZS}_sz{SZ}.json`.

| z_support | sigma_z | h_true | gray cov68 | exact cov68 | Δcov68 | gray map_bias | exact map_bias | Δmap_bias |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.067 | 0.692 | +0.625 | +0.0236 | +0.0023 | −0.0213 |
| 0.2 | 0.015 | 0.72 | 0.075 | 0.550 | +0.475 | +0.0283 | +0.0034 | −0.0248 |
| 0.2 | 0.015 | 0.84 | 0.042 | 0.608 | +0.567 | +0.0188 | +0.0042 | −0.0146 |
| 0.2 | 0.035 | 0.62 | 0.000 | 0.708 | +0.708 | +0.1226 | +0.0026 | −0.1199 |
| 0.2 | 0.035 | 0.72 | 0.000 | 0.575 | +0.575 | +0.1197 | +0.0046 | −0.1151 |
| 0.2 | 0.035 | 0.84 | 0.000 | 0.517 | +0.517 | +0.0200 | +0.0042 | −0.0158 |
| 0.3 | 0.015 | 0.62 | 0.267 | 0.700 | +0.433 | +0.0083 | +0.0003 | −0.0080 |
| 0.3 | 0.015 | 0.72 | 0.042 | 0.625 | +0.583 | +0.0136 | +0.0024 | −0.0112 |
| 0.3 | 0.015 | 0.84 | 0.025 | 0.525 | +0.500 | +0.0166 | +0.0047 | −0.0120 |
| 0.3 | 0.035 | 0.62 | 0.000 | 0.708 | +0.708 | +0.0297 | −0.0010 | −0.0307 |
| 0.3 | 0.035 | 0.72 | 0.000 | 0.633 | +0.633 | +0.0489 | +0.0023 | −0.0466 |
| 0.3 | 0.035 | 0.84 | 0.000 | 0.483 | +0.483 | +0.0200 | +0.0054 | −0.0146 |
| 0.5 | 0.015 | 0.62 | 0.675 | 0.700 | +0.025 | −0.0024 | −0.0019 | +0.0004 |
| 0.5 | 0.015 | 0.72 | 0.583 | 0.658 | +0.075 | −0.0013 | −0.0015 | −0.0002 |
| 0.5 | 0.015 | 0.84 | 0.767 | 0.725 | −0.042 | −0.0006 | −0.0014 | −0.0008 |
| 0.5 | 0.035 | 0.62 | 0.633 | 0.758 | +0.125 | −0.0039 | −0.0030 | +0.0009 |
| 0.5 | 0.035 | 0.72 | 0.567 | 0.683 | +0.117 | −0.0019 | −0.0021 | −0.0002 |
| 0.5 | 0.035 | 0.84 | 0.675 | 0.700 | +0.025 | −0.0001 | −0.0036 | −0.0035 |
| 1.0 | 0.015 | 0.62 | 0.675 | 0.700 | +0.025 | −0.0024 | −0.0019 | +0.0004 |
| 1.0 | 0.015 | 0.72 | 0.567 | 0.667 | +0.100 | −0.0014 | −0.0015 | −0.0002 |
| 1.0 | 0.015 | 0.84 | 0.725 | 0.725 | +0.000 | −0.0016 | −0.0014 | +0.0001 |
| 1.0 | 0.035 | 0.62 | 0.633 | 0.758 | +0.125 | −0.0039 | −0.0030 | +0.0009 |
| 1.0 | 0.035 | 0.72 | 0.550 | 0.675 | +0.125 | −0.0020 | −0.0018 | +0.0002 |
| 1.0 | 0.035 | 0.84 | 0.617 | 0.675 | +0.058 | −0.0028 | −0.0024 | +0.0003 |

(At zs ∈ {0.5, 1.0} the gray branch is the local-ratio `N_i/D_g_i`, so
control-level Δ vs exact reflects that composition difference, not truncation.)

## σ_z ladder at zs=0.2 (N-2c) — map_bias vs σ_z per mode

σ_z=0.002 rows use `--n-z-quad 480` (160 under-samples the kernel at that
width; σ_z=0 is not runnable — divide-by-zero in the Gaussian — so 0.002
probes the σ_z→0 limit). Sanity cross-check PASSED: the two_branch
sz=0.015/0.035 rows are IDENTICAL to the L-A deep-venue zs=0.2 cells
(`pp_zs0.2_sz{SZ}_volume.json` — same config/seed).

| mode | sigma_z | n_z_quad | h_true | cov68 | rail_fraction | map_bias |
|---|---|---|---|---|---|---|
| two_branch | 0.002 | 480 | 0.62 | 0.608 | 0.000 | +0.0033 |
| two_branch | 0.002 | 480 | 0.72 | 0.558 | 0.000 | +0.0035 |
| two_branch | 0.002 | 480 | 0.84 | 0.567 | 0.042 | +0.0046 |
| two_branch | 0.005 | 160 | 0.62 | 0.683 | 0.000 | +0.0039 |
| two_branch | 0.005 | 160 | 0.72 | 0.508 | 0.000 | +0.0045 |
| two_branch | 0.005 | 160 | 0.84 | 0.550 | 0.067 | +0.0057 |
| two_branch | 0.015 | 160 | 0.62 | 0.267 | 0.000 | +0.0123 |
| two_branch | 0.015 | 160 | 0.72 | 0.267 | 0.000 | +0.0144 |
| two_branch | 0.015 | 160 | 0.84 | 0.233 | 0.450 | +0.0139 |
| two_branch | 0.035 | 160 | 0.62 | 0.042 | 0.000 | +0.0317 |
| two_branch | 0.035 | 160 | 0.72 | 0.050 | 0.000 | +0.0368 |
| two_branch | 0.035 | 160 | 0.84 | 0.033 | 0.917 | +0.0194 |
| exact | 0.002 | 480 | 0.62 | 0.608 | 0.000 | +0.0030 |
| exact | 0.002 | 480 | 0.72 | 0.550 | 0.000 | +0.0031 |
| exact | 0.002 | 480 | 0.84 | 0.583 | 0.042 | +0.0044 |
| exact | 0.005 | 160 | 0.62 | 0.733 | 0.000 | +0.0028 |
| exact | 0.005 | 160 | 0.72 | 0.567 | 0.000 | +0.0030 |
| exact | 0.005 | 160 | 0.84 | 0.625 | 0.058 | +0.0042 |
| exact | 0.015 | 160 | 0.62 | 0.692 | 0.000 | +0.0023 |
| exact | 0.015 | 160 | 0.72 | 0.550 | 0.000 | +0.0034 |
| exact | 0.015 | 160 | 0.84 | 0.608 | 0.167 | +0.0042 |
| exact | 0.035 | 160 | 0.62 | 0.708 | 0.000 | +0.0026 |
| exact | 0.035 | 160 | 0.72 | 0.575 | 0.000 | +0.0046 |
| exact | 0.035 | 160 | 0.84 | 0.517 | 0.192 | +0.0042 |

**Ladder answer (N-2c):** the deep-venue bias does NOT fully vanish as σ_z→0 —
it converges to the same +0.003…+0.005 floor in BOTH modes (≤0.0004 apart at
σ_z=0.002). The σ_z-DEPENDENT part (the growth +0.003→+0.037 in two_branch) is
the kernel leak and is fully removed by exact truncation; the σ_z-INDEPENDENT
floor is a separate composition property of the completion-dominated regime.

## Observed-membership probe (N-2d) — Δ vs true-z membership

True-z baselines: gray from `results/pp_coverage_graymix_20260711/`, exact
from set (a). Membership on the observed `z_gal` (production's BallTree analog).

| mode | z_support | sigma_z | h_true | comp_frac true-z | comp_frac obs-z | Δcomp_frac | map_bias true-z | map_bias obs-z | Δmap_bias |
|---|---|---|---|---|---|---|---|---|---|
| gray | 0.2 | 0.015 | 0.62 | 0.709 | 0.707 | −0.002 | +0.0236 | +0.0280 | +0.0045 |
| gray | 0.2 | 0.015 | 0.72 | 0.787 | 0.783 | −0.003 | +0.0283 | +0.0318 | +0.0036 |
| gray | 0.2 | 0.015 | 0.84 | 0.848 | 0.843 | −0.004 | +0.0188 | +0.0183 | −0.0004 |
| gray | 0.2 | 0.035 | 0.62 | 0.709 | 0.694 | −0.014 | +0.1226 | +0.1765 | +0.0539 |
| gray | 0.2 | 0.035 | 0.72 | 0.787 | 0.773 | −0.013 | +0.1197 | +0.1329 | +0.0132 |
| gray | 0.2 | 0.035 | 0.84 | 0.848 | 0.835 | −0.013 | +0.0200 | +0.0199 | −0.0001 |
| gray | 0.3 | 0.015 | 0.62 | 0.219 | 0.223 | +0.004 | +0.0083 | +0.0095 | +0.0011 |
| gray | 0.3 | 0.015 | 0.72 | 0.390 | 0.391 | +0.001 | +0.0136 | +0.0141 | +0.0005 |
| gray | 0.3 | 0.015 | 0.84 | 0.551 | 0.549 | −0.002 | +0.0166 | +0.0165 | −0.0001 |
| gray | 0.3 | 0.035 | 0.62 | 0.219 | 0.239 | +0.020 | +0.0297 | +0.0716 | +0.0419 |
| gray | 0.3 | 0.035 | 0.72 | 0.390 | 0.393 | +0.003 | +0.0489 | +0.0831 | +0.0342 |
| gray | 0.3 | 0.035 | 0.84 | 0.551 | 0.543 | −0.009 | +0.0200 | +0.0200 | +0.0000 |
| exact | 0.2 | 0.015 | 0.62 | 0.709 | 0.707 | −0.002 | +0.0023 | +0.0006 | −0.0017 |
| exact | 0.2 | 0.015 | 0.72 | 0.787 | 0.783 | −0.003 | +0.0034 | −0.0006 | −0.0040 |
| exact | 0.2 | 0.015 | 0.84 | 0.848 | 0.843 | −0.004 | +0.0042 | −0.0027 | −0.0069 |
| exact | 0.2 | 0.035 | 0.62 | 0.709 | 0.694 | −0.014 | +0.0026 | +0.0052 | +0.0026 |
| exact | 0.2 | 0.035 | 0.72 | 0.787 | 0.773 | −0.013 | +0.0046 | −0.0005 | −0.0051 |
| exact | 0.2 | 0.035 | 0.84 | 0.848 | 0.835 | −0.013 | +0.0042 | −0.0205 | −0.0247 |
| exact | 0.3 | 0.015 | 0.62 | 0.219 | 0.223 | +0.004 | +0.0003 | +0.0005 | +0.0001 |
| exact | 0.3 | 0.015 | 0.72 | 0.390 | 0.391 | +0.001 | +0.0024 | +0.0015 | −0.0009 |
| exact | 0.3 | 0.015 | 0.84 | 0.551 | 0.549 | −0.002 | +0.0047 | +0.0025 | −0.0022 |
| exact | 0.3 | 0.035 | 0.62 | 0.219 | 0.239 | +0.020 | −0.0010 | +0.0139 | +0.0149 |
| exact | 0.3 | 0.035 | 0.72 | 0.390 | 0.393 | +0.003 | +0.0023 | +0.0086 | +0.0063 |
| exact | 0.3 | 0.035 | 0.84 | 0.551 | 0.543 | −0.009 | +0.0054 | −0.0017 | −0.0071 |

**Probe answer (N-2d):** membership determination matters, in mode-dependent
ways. Completion fractions barely move (|Δ| ≤ 0.020). At σ_z=0.015 both modes
are nearly insensitive. At σ_z=0.035, gray gets substantially WORSE under
observed-z membership (Δbias up to +0.054 — misclassified events feed the
already-defective mixture), while exact's response is mixed and coverage
degrades (cov68 0.18–0.46 at zs=0.2) with sign-flipping biases (−0.021…+0.015):
a hard truncated kernel is misspecified for events whose true z sits on the
other side of the edge from their observed z. Production decides membership on
measured redshifts, so any production adoption of a truncated kernel needs a
soft (photo-z-marginalized) membership treatment, not a hard clamp.

## Verdict / decision-tree mapping (`.planning/HANDOFF-DEEP-BIAS-MECHANISM-20260710.md`)

1. **Mechanism identified?** YES, decomposed into two parts. (i) The DOMINANT,
   σ_z-dependent part of the deep-incompleteness high bias is the
   **membership-support leak in the host-event numerator** — kernel mass above
   the catalogue support edge — removed exactly by the membership-truncated
   kernel (bias 3–8× down, coverage restored to near-nominal, rails gone).
   (ii) A SMALLER σ_z-independent residual (+0.002…+0.005, growing with
   completion fraction) lives in the completion-branch composition (`B_num/D`
   positive tilt with a shrinking host counterweight) — this part is NOT
   membership bookkeeping and matches the N-3 prior-sensitivity target: `B_num`
   integrates the population prior `w_pop` over the out-of-catalogue volume, so
   its calibration is population-prior-driven by construction.
2. **Production-correction candidate flagged** (routes to /physics-change +
   literature — Gray et al. 2020; Chen–Fishbach–Holz 2018;
   Mastrogiovanni et al./ICAROGW out-of-catalogue treatment — NOT this task):
   f(z)-weighted / membership-truncated in-catalogue kernel integrands, i.e.
   truncate (or completeness-weight) the per-host kernel numerator at the
   catalogue support instead of letting it integrate over the full z range.
   The N-2d probe adds a design constraint: with measured-redshift membership
   the truncation must be soft (photo-z-marginalized), not a hard clamp.
3. **EXP-40 watch (seed1000 re-eval, cluster return):** production's
   composition is gray-like (untruncated kernels + mixture); the harness says
   that composition is biased HIGH at deep incompleteness and WORSE under
   observed-z membership at large σ_z. Watch for an interior-but-biased-HIGH
   posterior in both regimes; if a truncated-kernel correction were adopted,
   the harness floor suggests a residual of only +0.3…+0.6% of truth remains
   at 58% zero-host.
4. **D1 (issue #30, depth-vs-truncation):** evidence now cuts BOTH ways and
   supports the user directive to investigate rather than truncate. The exact
   mode shows deep incompleteness is NOT intrinsically un-calibratable — an
   estimator-level fix recovers near-calibration where hard catalogue
   truncation would discard the depth. But full calibration is NOT achieved
   (1/12 strict pass); the σ_z-independent completion-branch floor and the
   observed-membership sensitivity must be quantified (N-3) before depth-1.5 +
   fallback closure claims. Truncation remains the robustness bound.

## Carried caveats (verbatim from the graymix SUMMARY, with status update)

1. **1D-channel only** — the 2D (+0.025 remaining) question is NOT covered by
   this harness.
2. **Single-host clean limit** — production host-found events carry the full
   in-catalogue galaxy sum; this harness's exact mode truncates a single
   effective host kernel. The gray-mode escape hatch was closed in 260711-07n;
   the exact mode now closes the "membership bookkeeping" escape hatch too.
3. **Hard truncation** (`z_support` step) vs production's soft M_BH-prune
   truncation of the effective catalogue — and, new from N-2d: hard truncation
   under observed-z membership is itself misspecified; production analogs need
   the soft form.
