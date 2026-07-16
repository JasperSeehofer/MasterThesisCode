# pp_coverage deep-venue (`z_support`) sweep — VERDICT (2026-07-10)

**Provenance:** handoff item L-A; quick task `260710-sjm-pp-coverage-deepvenue-mode`
(code at `cfce571` on `physics/zero-host-completion-fallback`); RUNBOOK.md in this
directory (grid, commands, criteria — all followed verbatim). Estimator analog of the
production issue-#29 zero-host pure-completion fallback `p_i = B_num/D` (commit
`8db6c6e`; Gray et al. 2020 arXiv:1908.06050 Eqs. 29+32; G2a limiting case 2).

## VERDICT: BIASED HIGH at deep incompleteness — calibration is NOT preserved

- **Estimator core healthy:** both untruncated controls (`z_support=1.0`) are
  calibrated (cov68 0.667–0.758 vs nominal 0.68, |bias| ≤ 0.003, zero rail).
- **Truncation inert when it should be** (`z_support=0.5`, completion_fraction
  ≈ 0–0.008): results identical to control — the machinery adds nothing spurious.
- **At 22–55% completion-governed events** (`z_support=0.3`): coverage collapses
  (cov68 0.008–0.542) with **positive (high) H0 bias +0.005…+0.025** in h
  (+0.7…+3.5% of truth).
- **At 71–85% completion-governed events** (`z_support=0.2`): **bias +0.014…+0.039**
  (+1.8…+5.4%), cov68 ≤ 0.267, and the h_true=0.84 ensemble **rails at the HIGH grid
  edge** (rail fraction 0.45 at σ_z=0.015, 0.92 at σ_z=0.035) — an upper-edge analog
  of the seed1000 lower-edge rail.
- **σ_z dependence:** the bias roughly doubles from σ_z=0.015 to 0.035 at fixed
  completion fraction. The completion branch itself has no σ_z dependence, so this is
  a mixed-population (host-branch × completion-branch composition) effect, not a
  property of the completion integral alone.

**All 12 truncated-cell × truth comparisons at z_support ∈ {0.2, 0.3} flag BOTH
criteria** (coverage collapse beyond ±2·SE = ±0.085 AND |Δ map_mean| > 2·SEM). The
single marginal flag at (0.5, 0.035, 0.84) — Δ = +0.0014 vs 2·SEM = 0.0012 at
completion_fraction 0.008 — is directionally consistent but not significant across
18 comparisons.

### Mechanism (hypothesis, registered for the ledger)

`B_num(h)/D(h)` is generically **increasing in h** for events near/beyond the support
edge: raising h maps the fixed-d_L GW likelihood deeper into the out-of-catalogue
volume (and D(h) provides no full counterweight), so every zero-host event prefers
high h. In the harness's clean single-host limit the in-catalogue events carry no
compensating `w_G(h)`-weighted admixture, so completion-dominated ensembles tilt
high. This is the same object as the FINDINGS_COMBINE_20260710 `w_G(h) = β_G/D(h)`
slope suspect (~26% of the seed1000 1D rail tilt).

### Implications

1. **EXP-40 prediction (registered now, before the cluster returns):** the post-#29
   seed1000 re-eval should move UP off the h=0.60 rail — but this result says the
   risk flips sign: watch for an interior-but-biased-HIGH posterior, not a clean
   de-rail. At seed1000's 58% zero-host fraction, the completion branch dominates.
2. **Decision D1 (issue #30):** strong quantitative support for explicit
   z-truncation (option b): calibration is exact where completion_fraction ≈ 0 and
   degrades monotonically with it. Depth-1.5 + fallback (option a) is not safe for
   closure claims without an estimator upgrade.
3. **Caveat-2 escape hatch:** production host-found events DO carry the `B_num`
   admixture (this harness's clean limit omits it), which acts in the compensating
   direction. Whether the full Gray mixture restores calibration at 60–95%
   incompleteness is testable in this harness by adding a full-mixture branch option
   — natural follow-up before trusting ANY deep-venue closure.

## Anchor bit-identity re-run (RUNBOOK §B)

**PASS on all shared keys.** `diff` of the sorted `.results` blocks shows the rerun
differs from `results/pp_coverage_sigmaz_scan_20260703/pp_sigmaz0.10_volume.json`
ONLY by the new `completion_fraction: 0.0` key (3 lines, one per truth block); every
pre-existing numerical value is byte-identical. The RUNBOOK's "byte-identical
`.results`" phrasing did not anticipate the schema addition; the no-op guarantee for
`z_support=None` holds exactly (also enforced by the committed golden-pin test).

## Per-cell × truth table (RUNBOOK §C)

| z_support | sigma_z | h_true | cov50 | cov68 | cov90 | rail_fraction | MAP mean | MAP bias | completion_fraction |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.225 | 0.267 | 0.517 | 0.000 | 0.6323 | +0.0123 | 0.709 |
| 0.2 | 0.015 | 0.72 | 0.175 | 0.267 | 0.500 | 0.000 | 0.7344 | +0.0144 | 0.787 |
| 0.2 | 0.015 | 0.84 | 0.183 | 0.233 | 0.333 | 0.450 | 0.8539 | +0.0139 | 0.848 |
| 0.2 | 0.035 | 0.62 | 0.000 | 0.042 | 0.167 | 0.000 | 0.6517 | +0.0317 | 0.709 |
| 0.2 | 0.035 | 0.72 | 0.017 | 0.050 | 0.200 | 0.000 | 0.7568 | +0.0368 | 0.787 |
| 0.2 | 0.035 | 0.84 | 0.008 | 0.033 | 0.208 | 0.917 | 0.8594 | +0.0194 | 0.848 |
| 0.3 | 0.015 | 0.62 | 0.508 | 0.542 | 0.775 | 0.000 | 0.6246 | +0.0046 | 0.219 |
| 0.3 | 0.015 | 0.72 | 0.092 | 0.192 | 0.450 | 0.000 | 0.7281 | +0.0081 | 0.390 |
| 0.3 | 0.015 | 0.84 | 0.092 | 0.175 | 0.275 | 0.133 | 0.8508 | +0.0108 | 0.551 |
| 0.3 | 0.035 | 0.62 | 0.050 | 0.083 | 0.292 | 0.000 | 0.6353 | +0.0153 | 0.219 |
| 0.3 | 0.035 | 0.72 | 0.008 | 0.017 | 0.075 | 0.000 | 0.7435 | +0.0235 | 0.390 |
| 0.3 | 0.035 | 0.84 | 0.000 | 0.008 | 0.008 | 0.908 | 0.8595 | +0.0195 | 0.551 |
| 0.5 | 0.015 | 0.62 | 0.700 | 0.700 | 0.900 | 0.000 | 0.6181 | −0.0019 | 0.000 |
| 0.5 | 0.015 | 0.72 | 0.617 | 0.658 | 0.900 | 0.000 | 0.7185 | −0.0015 | 0.000 |
| 0.5 | 0.015 | 0.84 | 0.575 | 0.708 | 0.942 | 0.000 | 0.8391 | −0.0009 | 0.008 |
| 0.5 | 0.035 | 0.62 | 0.633 | 0.758 | 0.875 | 0.000 | 0.6170 | −0.0030 | 0.000 |
| 0.5 | 0.035 | 0.72 | 0.567 | 0.675 | 0.892 | 0.000 | 0.7183 | −0.0017 | 0.000 |
| 0.5 | 0.035 | 0.84 | 0.550 | 0.692 | 0.925 | 0.000 | 0.8390 | −0.0010 | 0.008 |
| 1.0 | 0.015 | 0.62 | 0.700 | 0.700 | 0.900 | 0.000 | 0.6181 | −0.0019 | 0.000 |
| 1.0 | 0.015 | 0.72 | 0.625 | 0.667 | 0.900 | 0.000 | 0.7185 | −0.0015 | 0.000 |
| 1.0 | 0.015 | 0.84 | 0.592 | 0.725 | 0.942 | 0.000 | 0.8386 | −0.0014 | 0.000 |
| 1.0 | 0.035 | 0.62 | 0.633 | 0.758 | 0.875 | 0.000 | 0.6170 | −0.0030 | 0.000 |
| 1.0 | 0.035 | 0.72 | 0.550 | 0.675 | 0.892 | 0.000 | 0.7182 | −0.0018 | 0.000 |
| 1.0 | 0.035 | 0.84 | 0.483 | 0.675 | 0.958 | 0.000 | 0.8376 | −0.0024 | 0.000 |

## Control comparison (each truncated cell vs its z_support=1.0 control at same σ_z)

| z_support | sigma_z | h_true | comp_frac | cov68 | ctrl cov68 | Δcov68 | coverage | Δmap_mean | 2·SEM | bias |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.015 | 0.62 | 0.709 | 0.267 | 0.700 | −0.433 | COLLAPSE | +0.0142 | 0.0014 | FLAG |
| 0.2 | 0.015 | 0.72 | 0.787 | 0.267 | 0.667 | −0.400 | COLLAPSE | +0.0160 | 0.0017 | FLAG |
| 0.2 | 0.015 | 0.84 | 0.848 | 0.233 | 0.725 | −0.492 | COLLAPSE | +0.0153 | 0.0013 | FLAG |
| 0.2 | 0.035 | 0.62 | 0.709 | 0.042 | 0.758 | −0.717 | COLLAPSE | +0.0347 | 0.0022 | FLAG |
| 0.2 | 0.035 | 0.72 | 0.787 | 0.050 | 0.675 | −0.625 | COLLAPSE | +0.0386 | 0.0027 | FLAG |
| 0.2 | 0.035 | 0.84 | 0.848 | 0.033 | 0.675 | −0.642 | COLLAPSE | +0.0219 | 0.0004 | FLAG |
| 0.3 | 0.015 | 0.62 | 0.219 | 0.542 | 0.700 | −0.158 | COLLAPSE | +0.0065 | 0.0007 | FLAG |
| 0.3 | 0.015 | 0.72 | 0.390 | 0.192 | 0.667 | −0.475 | COLLAPSE | +0.0096 | 0.0009 | FLAG |
| 0.3 | 0.015 | 0.84 | 0.551 | 0.175 | 0.725 | −0.550 | COLLAPSE | +0.0122 | 0.0010 | FLAG |
| 0.3 | 0.035 | 0.62 | 0.219 | 0.083 | 0.758 | −0.675 | COLLAPSE | +0.0184 | 0.0011 | FLAG |
| 0.3 | 0.035 | 0.72 | 0.390 | 0.017 | 0.675 | −0.658 | COLLAPSE | +0.0253 | 0.0014 | FLAG |
| 0.3 | 0.035 | 0.84 | 0.551 | 0.008 | 0.675 | −0.667 | COLLAPSE | +0.0219 | 0.0003 | FLAG |
| 0.5 | 0.015 | 0.62 | 0.000 | 0.700 | 0.700 | +0.000 | ok | +0.0000 | 0.0006 | ok |
| 0.5 | 0.015 | 0.72 | 0.000 | 0.658 | 0.667 | −0.008 | ok | +0.0001 | 0.0007 | ok |
| 0.5 | 0.015 | 0.84 | 0.008 | 0.708 | 0.725 | −0.017 | ok | +0.0006 | 0.0007 | ok |
| 0.5 | 0.035 | 0.62 | 0.000 | 0.758 | 0.758 | +0.000 | ok | +0.0000 | 0.0011 | ok |
| 0.5 | 0.035 | 0.72 | 0.000 | 0.675 | 0.675 | +0.000 | ok | +0.0001 | 0.0013 | ok |
| 0.5 | 0.035 | 0.84 | 0.008 | 0.692 | 0.675 | +0.017 | ok | +0.0014 | 0.0012 | FLAG (marginal) |

## Carried caveats (verbatim per RUNBOOK §C)

1. **1D-channel only** — the 2D (+0.057) question is NOT covered by this harness.
2. **Single-host clean limit** — production host-found events ALSO carry a `B_num`
   admixture in the mixture; this harness omits that, so ONLY the zero-host branch is
   the exact production analog.
3. **Hard truncation** (`z_support` step) vs production's soft M_BH-prune truncation
   of the effective catalogue.
