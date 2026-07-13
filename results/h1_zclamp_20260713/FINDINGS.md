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

## Production relevance (OPEN — the decisive question)

Whether production shares this bias depends on how real low-z photo-z behave near 0:

- The reduced catalogue redshift is **not hard-clamped**: min −0.000318, only 17 of
  500k ≤ 0, 124 < 0.001. So the catalogue does not artificially pile low-z photo-z at
  a boundary the way the harness clamp does.
- BUT the shallow shell is 89.7% photometric with σ_z/z ~ 0.65 ([L8]); if those
  photo-z were produced by a pipeline that floors negatives at 0 (common for photo-z
  codes), production would carry the same censoring the harness models. If instead the
  reported z are best-estimate spec-z-anchored or raw (allowing small negatives),
  production is closer to the unbiased clamp-OFF case.

**Next step to settle it:** inspect the actual low-z (z≲0.1) photometric hosts in the
reduced catalogue — is there a pileup / hard floor near 0 in the photo-z (as opposed
to the true-redshift) column, and does `z_error` reflect a censored distribution? That
determines whether the H1 production fix (censored-measurement likelihood) is needed
or moot.

## Relation to the mass channel (H2)

Same family (`results/mass_kernel_truncation_20260713/`): a large fractional error
against a physical boundary, handled by a kernel that ignores the boundary, biases
HIGH. But the *specific* driver differs — for z it is the **censoring of the
observed value** at z≥0; for mass it is the **untruncated kernel spilling past
[M_min,M_max]** with the population weight. The z fix (censored measurement model)
and the mass fix (lognormal×R_eff truncated kernel) are both Candidate-B-flavoured
but not identical.

## Caveat

This is the synthetic harness. It shows the harness's +0.030 is a clamp artifact and
identifies the exact mechanism; it does NOT by itself prove production is or isn't
biased — that needs the catalogue photo-z inspection above. If production's low-z
photo-z are effectively censored, the seed600 +0.013 1D residual is (partly) this
effect and the censored-measurement fix applies; if not, the shallow production bias
may be smaller than [L8] implied.
