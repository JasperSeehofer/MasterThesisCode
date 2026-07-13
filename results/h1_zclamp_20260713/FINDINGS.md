# FINDING — the shallow-venue H1 bias is the z≥0 photo-z CLAMP, not the volume correction

**Date:** 2026-07-13 · **Verdict:** the pp_coverage harness's shallow-venue high bias
(+0.030, [L8]) is caused by the **generative clamp of the observed photo-z at z≥0**,
combined with a naive Gaussian kernel that does not model that clamp — NOT by the
volume/Eddington correction "failing to cancel under truncation" (the prior framing).
This refines [L8] and makes the fix concrete: **model the censored measurement**
(or use raw unclamped photo-z). Production relevance is an open, checkable question.

## Diagnostic (`zclamp_diagnostic.py`, multi-seed n_real=200, n_events=250)

Toggling only `config.clamp_zgal` (the generative `clip(z_host+noise, Z_MIN, None)`),
volume kernel, at the shallow seed600-matched venue (d50=0.23, z_med~0.044, σ_z=0.035)
and a deep control (d50=1.85):

| venue | clamp | map_bias | cov68 |
|---|---|---|---|
| deep (z_med~0.28) | ON | −0.0024 | 0.68 |
| deep | OFF | −0.0024 | 0.68 |
| **shallow (z_med~0.044)** | **ON (current)** | **+0.0240 ± 0.0022** | **0.61** |
| **shallow** | **OFF (raw z_gal)** | **−0.0056 ± 0.0020** | **0.68** |

Removing the clamp erases the entire +0.030 shallow bias and restores nominal
coverage; the deep venue is clamp-independent (control). Seed-robust.

## Mechanism

At shallow z with σ_z/z ~ 1, a large fraction of the Gaussian photo-z noise pushes
`z_host + noise` below the physical floor; clamping piles those observations at
`Z_MIN`. The inference kernel `N(z; z_gal, σ_z)` treats a piled-at-Z_MIN observation
as a genuine z≈0 host with symmetric uncertainty, which it is not — the censored
measurement carries different information. The naive kernel's misread of the censored
observations biases H0 HIGH (empirically; the sign is not obvious from a point
argument, same lesson as the mass channel). The volume kernel's Eddington-in-z
correction is a *red herring* here: it is present in both clamp-on and clamp-off runs
(same kernel, same truncated integration window `[Z_MIN, z_hi]`); only the generative
clamp differs.

## Production relevance — LARGELY SETTLED: the real catalogue is NOT clamped

Direct inspection of the reduced catalogue redshift column (2M rows, z<0.10 shell,
n=871k) settles it:

- **No pileup / floor at 0.** The z histogram near 0 is smooth and monotonically
  rising (counts across [0, 0.02] in 0.0025 steps: 1637, 4114, 5492, 5378, 6330,
  7480, 9022, 12516). A hard floor would produce a spike at 0 — there is none
  (`n(z==0)=0`, `n(z<0)=20`/2M, min −0.000318). The photo-z are effectively RAW /
  uncensored, i.e. the harness **clamp-OFF** case — which is unbiased (−0.005).
- Only **3.7%** of low-z hosts have `z < σ_z` (kernel crosses 0); σ_z/z median 0.218
  over the full z<0.10 shell (`z_error` median 0.034, confirming the σ_z scale).

**Conclusion:** production does NOT reproduce the harness's generative clamp, so the
harness's +0.030 shallow bias is **substantially a harness artifact**, not a faithful
model of production's low-z behaviour. This **weakens [L8]'s attribution** of the
seed600 +0.013 1D residual to the σ_z/z truncated-kernel effect, and suggests the
**redshift half of the production kernel fix is likely largely unnecessary** (the
censored-measurement issue bites only the ~3.7% boundary-crossing hosts, not the bulk).
The seed600 +0.013 1D residual now needs a different explanation (or is closer to
single-seed scatter than a shallow-truncation systematic) — reopened, campaign-gated.

## Relation to the mass channel (H2)

Same family (`results/mass_kernel_truncation_20260713/`): a large fractional error
against a physical boundary, handled by a kernel that ignores the boundary, biases
HIGH. But the *specific* driver differs — for z it is the **censoring of the
observed value** at z≥0; for mass it is the **untruncated kernel spilling past
[M_min,M_max]** with the population weight. The z fix (censored measurement model)
and the mass fix (lognormal×R_eff truncated kernel) are both Candidate-B-flavoured
but not identical.

## Caveat

The clamp isolation is in the synthetic harness; the catalogue inspection is a
distributional check, not an end-to-end production A/B. It establishes that the
DOMINANT harness shallow mechanism (the generative clamp) is unrepresentative of the
smooth real catalogue, so the +0.030 does not transfer to production wholesale. It
does NOT prove production has exactly zero 1D shallow bias — a residual from the ~3.7%
boundary-crossing hosts, or an unrelated effect, could remain. But it removes the
main quantitative basis for a redshift-kernel production fix and reopens the seed600
+0.013 attribution. The mass-channel (H2) bias is independent of this and stands.
