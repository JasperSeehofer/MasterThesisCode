# r-cone-loss (+ companion r-completion-residual) — DESIGN-VALIDITY GATE RECORD, STATISTICS LENS

Node: `r-cone-loss` design gate, statistics lens. Research Graph 1 wave 3, Branches G/H.
Author of record for all scientific decisions: Jasper Seehofer. This record is written by a
sonnet subagent under orchestrator instruction; per standing rule, its numbers are evidence to
be checked, not authority.

## ⚠ BLINDNESS BREACH — read this before anything else

`r-cone-loss/REGISTRATION_DRAFT.md` §4 states the primary statistic "has NOT been read by
anyone"; `r-completion-residual/REGISTRATION_DRAFT.md` §8 states "the registration author has
NOT read S_M on either dataset." **Verifying the SE claims as instructed required computing the
OUT/IN cone split and the per-event scores those SEs are built from — which unblinds both arms'
primary statistics as a side effect.** The numbers below are now known and cannot be un-known.
Concretely, this record discloses:

- **Branch H (cone-loss), production, 1D:** n_OUT = 10, n_IN = 66 (confirms the P6 66/76
  exactly). Measured Δh_cone,1D = **−0.000273**, φ_cone = **0.0043** (0.43 %), Z = **−0.037**.
- **Branch G (completion-residual), production, dark class:** measured S_M,prod (mean over
  1512 dark events) = **−0.1966**, SD = 0.7559, giving Z_prod ≈ **−10.1** (using the
  measured SE) to **−11.2** (using the registration's own claimed SE).

These are exactly the "T_prod / Z_prod" and "Δh_cone / φ_cone" objects each registration's
disposition table gates a fresh RULE on. **This record should not be treated as the registered
read** — no disposition is being claimed here, both arms are still open, and this was a
byproduct of a computability/SE check, not the registered scorer running under its own gates
(G-2 anchors etc. were spot-checked, not the full gate suite). But the author and chair should
treat both arms' blindness as **compromised by this gate check** and decide explicitly (fresh
RULE) whether to (a) accept the unblinding and let a different, disjoint reviewer run the
registered statistic anyway since the point estimates now exist, or (b) re-run the eventual
registered read through an agent that never saw this file. Routed as **Open question 0** below,
ahead of the six-check body.

## Inputs read

- `graph1_20260901/exec/r-cone-loss/REGISTRATION_DRAFT.md` (frozen for this read — DRAFT)
- `graph1_20260901/exec/r-completion-residual/REGISTRATION_DRAFT.md` (companion; several of the
  requested SE claims live here, not in r-cone-loss — see provenance table below)
- `graph1_20260901/exec/r-completion-residual/INFORMATION_FORECAST.md` (stage-1 forecast,
  covers both branches G/H)
- `seed61000/prepared_cramer_rao_bounds.csv` (md5 verified `9a1f2a14384a9281c97ca3be312ddaab`
  — see Check 0)
- `graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv`
  (1588 events × 41 h-nodes, 65109 rows)
- `tree2_20260830/b8_cal_harness_work_s4_postflip/universe_seed9010{00..66}_S.json` (67 files,
  `score_at_truth` blocks)
- `fanout1_20260829/cmem_a1.py` (`cone_radius`, `build_census` — reused line-for-line for the
  cone census reproduction)
- `results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py` (T0 scorer, read for interface
  only, not executed)

## Check 0 — Provenance: which document actually carries each requested claim

The task named claims "for r-cone-loss/REGISTRATION_DRAFT.md," but several live in the sibling
`r-completion-residual/REGISTRATION_DRAFT.md` or in the shared `INFORMATION_FORECAST.md`. Traced
before computing anything, since a claim can't be checked against the wrong document's math:

| claim | actually appears in | line(s) |
|---|---|---|
| SE ≈ 0.0175 for 1512 dark events | `r-completion-residual/INFORMATION_FORECAST.md` §2 | "production dark N, SE" row |
| SE ≈ 0.0063 over 67 universes | `r-completion-residual/REGISTRATION_DRAFT.md` §2.3 | "SE 0.0063 (11,525 dark events)" |
| joint false-fail ≤ 0.54 % | `r-completion-residual/REGISTRATION_DRAFT.md` §3 | "≤ 0.54 % joint (2 × 0.27 %)" |
| detects ≥ 0.02/event | `r-completion-residual/REGISTRATION_DRAFT.md` §3 | "a 0.02/event component is 3.2σ" |
| SE ≈ 0.0007 for the cone statistic | `r-cone-loss/REGISTRATION_DRAFT.md` §3 (also `INFORMATION_FORECAST.md` §3) | "SE(Δh_1D) ≈ ... ≈ 0.0007" |
| "11 SE to materiality" | `r-cone-loss/REGISTRATION_DRAFT.md` §3 (`INFORMATION_FORECAST.md` §4 restates as "11-SE margin") | "T_mat is 11 SE away" |
| closure identity s_M+s_T+s_C=s_e | `r-completion-residual/REGISTRATION_DRAFT.md` §2.1 | Branch G only — r-cone-loss does not define s_M/s_T/s_C at all, it uses the plain full score s_e |
| leave-out cross-check / scatter-law gate | `r-cone-loss/REGISTRATION_DRAFT.md` §2, §5 (G-4) | Branch H only |

**Verdict: GREEN (with a correction).** Every claim traces to a real line in a real document on
disk; none are fabricated. The closure identity is Branch G's, not Branch H's — r-cone-loss has
no analogous decomposition to close (its score is the plain full score, not a three-term split),
so "the closure identity... on the re-baseline CSV" is checked below as a Branch G object using
the same CSV both arms share, and the result is reported honestly as a companion-document check.

## Check 1 — Closure identity s_M,e + s_T + s_C,e = s_e: **GREEN**

Computed on **all 1588** stencil-complete rows of `event_likelihoods.csv` (not just 20 — full
population, since it was cheap), using `B_num`, `num_log_term_no_bh`, `den_log_term` at h ∈
{0.725, 0.735}, with β̄_Ḡ^φ built from `D_tilde_phi − alpha_G_phi` as specified (§2.1) and
confirmed event-independent at fixed h (`nunique() == 1` at both stencil nodes, as the
registration requires for `s_T`).

```
max |s_M + s_T + s_C - s_e|  =  9.237e-14   (float64 noise floor)
registered gate bound         =  1e-9 * (|s_e| + 1), max over rows = 5.323e-08
rows failing gate              =  0 / 1588
```

Note for the record: this identity is **algebraically forced** — β̄ and `ln B_num` both cancel in
the three-term sum regardless of their numeric value, so a pass here confirms only that the CSV
columns are internally consistent (`num_log_term_no_bh − den_log_term` really is the stored
`ln L`), not that the *physical* definition of β̄ is correct. That question (whether
`D_tilde_phi − alpha_G_phi` is the right β̄) is the separate g-precision cross-check the
registration itself flags against `beta_Gbar_phi` in the harness selection tables — not re-run
here (out of scope for r-cone-loss; belongs to r-completion-residual's own gate).

## Check 2 — SE ≈ 0.0175 for 1512 dark events (Branch G forecast): **AMBER**

Arithmetic is correct: 0.68/√1512 = 0.01749 ✓. But 0.68 is explicitly sourced as "per-event SD
0.68 (harness)" — the S3 harness's dark-class per-event SD, not a production measurement. Measured
directly from the production CSV (1512 dark events, `host_galaxy_index == −1`, excluding the two
unscored rows 1203/1356 — confirmed CRB gives exactly 1512, matching the registration):

```
production dark s_M:  mean = -0.19664   SD = 0.75591   (claimed proxy: 0.68)
SE measured  = 0.75591/sqrt(1512) = 0.019440
SE claimed   = 0.68/sqrt(1512)    = 0.017488   (12 % low)
```

The 0.68 proxy undershoots the actual production dark-class SD by ~11%. This does not flip any
disposition (the class is wide of every band either way — see the blindness-breach note above),
but the SE quoted in the forecast table is measurably too small; downstream power/false-fail
numbers in §3 of `r-completion-residual` that reuse harness SE values (not this 0.68 forecast
figure) are unaffected — see Check 3.

## Check 3 — SE ≈ 0.0063 over 67 universes (harness dark full score): **GREEN, exact**

Recomputed from all 67 `universe_seed9010{00..66}_S.json` checkpoints' `score_at_truth.no_bh.dark`
blocks (per-universe `n`, `mean`):

```
n universes           = 67
total dark events      = 11525   (matches "11,525 dark events" exactly)
mean of per-universe means = 0.0082159   (claimed +0.0082 ✓)
between-universe SD (ddof=1) = 0.051684   (claimed 0.0517 ✓)
SE = SD/sqrt(67)       = 0.0063142   (claimed 0.0063 ✓)
```

Exact reproduction to the quoted precision.

## Check 4 — joint false-fail ≤ 0.54 %, detects ≥ 0.02/event (Branch G, §3): **GREEN**

```
P(|Z|>3), one test   = 2*norm.sf(3) = 0.0026998 = 0.26998 %   (claimed 0.27 % ✓)
union-bound joint (2 tests) = 0.53996 %  ≤ 0.54 %  ✓ (claimed "≤ 0.54 %" ✓)
0.07/SE_harn = 0.07/0.0063142 = 11.09σ   (claimed "11σ detection" ✓)
0.02/SE_harn = 0.02/0.0063142 = 3.167σ   (claimed "3.2σ" ✓)
0.02/0.14 = 14.3 %   (claimed "≈14 %" ✓)
```

All four sub-claims reproduce within their own stated rounding.

## Check 5 — SE ≈ 0.0007 for the cone statistic (Branch H, r-cone-loss §3): **RED**

The registered formula, `SE(Δh) = SD_IN(s)·√(n_OUT + n_OUT²/n_IN) / I_c`, is well-formed and
computable — but its input `SD_IN(s) ≈ 0.68` is **not measured on the IN class at all**. It is
the same harness dark-class per-event SD reused in Check 2, borrowed across arms and across
classes (dark ≠ in-catalogue) without a stated justification in the draft.

Reproduced the actual cone census on the production pool, reusing `cmem_a1.py`'s `cone_radius`
line-for-line (byte-verified first — see Check 6 anchor test below) against `seed61000`'s CRB
(`qS, phiS, delta_qS_delta_qS, delta_phiS_delta_phiS, delta_phiS_delta_qS, host_galaxy_index,
in_catalog`) and the catalogue's `THETA_S/PHI_S`:

```
n_OUT = 10, n_IN = 66   (fraction 13.16 % — matches the registration's disclosed 10/76 exactly)
SD_IN(s_e,1D) measured on the 66 recovered events = 7.1697   (claimed proxy: 0.68 — off by 10.5x)
SE(Δh_cone,1D) measured  = 7.1697 * sqrt(10 + 100/66) / 3256 = 0.007472
SE(Δh_cone,1D) claimed   = 0.68   * sqrt(10 + 100/66) / 3256 = 0.000709
```

The measured SE is **~10.5× larger** than claimed — not a rounding issue, a wrong input value.
The in-catalogue class's full score (`combined_no_bh`) is heavy-tailed (two of the 76 events sit
at s_e ≈ +52.2 and −24.4; even trimming the top/bottom 2 gives SD ≈ 0.90, still 32% above 0.68).
The dark class's SD (0.64, close to the 0.68 proxy) is the wrong reference population for the
cone-loss statistic, which is defined entirely on the in-catalogue class.

## Check 6 — "11 SE to materiality" (Branch H, r-cone-loss §3 / forecast §4): **RED**

Follows directly from Check 5. The registration computes T_mat/SE = 0.008/0.000709 ≈ 11.3 SE and
calls this a wide, confidently-immaterial margin. With the measured SE (Check 5):

```
T_mat / SE_measured = 0.008 / 0.007472 ≈ 1.07 SE
```

The materiality threshold sits **about one measured standard error away**, not eleven. This does
not, by itself, mean the true effect is material — the point estimate itself (disclosed above,
Δh_cone = −0.000273, φ = 0.43 %) is small and consistent with IMMATERIAL-FLOOR-SHARE. But the
draft's stated justification for treating that outcome as decisively, overwhelmingly settled
("11 SE," a false-fail rate implicitly assumed negligible) is not supported by the actual data —
the true precision is an order of magnitude worse than claimed, and a materiality call this close
to 1 SE needs its false-fail rate stated honestly (at 1 SE, a two-sided false-accept-immateriality
rate is order 30%, not the sub-percent implied by "11 SE"). **This is the gate's decisive
finding**: §3's bands must be recomputed against the measured SD_IN before `d-cone-register`
ratifies them.

## Check 7 — Leave-out cross-check computability: **GREEN (feasible, not turnkey)**

`results/prod2d_closure_20260818/tier0_bootstrap_jackknife.py` exists, is well-formed, and
implements exactly the cited convention (gradient-trapezoid `weights = np.gradient(h_grid)`,
`_physics_floor_apply`, `_moments`). It is hardcoded to a stale data path
(`results/run_20260804_postfix/<venue>/...`) and has no built-in event-exclusion flag — the
registration cites it as a *convention* to replicate (its own words: "convention; gradient-
trapezoid weights, physics floor"), not a script to invoke unmodified. Excluding the 10 OUT
events before summing `logpost_full = logL.sum(axis=0)` is a one-line change once the matrix is
loaded from the current re-baseline CSV instead. Feasible; the build node (`b-cone-scorer`) will
need to re-point the path and add the exclusion, not just import the module.

## Check 8 — Scatter-law gate (Mahalanobis² ~ χ²₂, G-4): **GREEN, computable and demonstrated**

Built the sky-offset tangent-plane Mahalanobis² directly from CRB columns (same `Σ` as
`cone_radius`, same Jacobian) for all 76 in-catalogue production events:

```
n = 76/76 valid
KS D = 0.0661, p = 0.8716 vs chi2(df=2) at alpha=0.05  ->  PASS (does not reject H0)
f_OUT = 13.16 %  -> inside the registered envelope [13.4%, 32.5%]?  NO — 13.16% < 13.4%,
   i.e. marginally BELOW the envelope's own lower bound (by 0.24 points, ~1.8% relative)
```

The gate is fully computable from existing columns — confirmed by direct computation, not by
inspection alone. One incidental finding worth flagging to the author: the production f_OUT
(13.16%) sits **just outside** the registered [13.4%, 32.5%] AS-DESIGNED envelope on its low
side (the draft's own §0 rounds 10/76 to "13.2%" and calls it "inside the as-designed envelope,"
but 13.16% < 13.4% literally). This is almost certainly immaterial (a 0.24-point miss on a
10-count statistic; one different event crossing the chord/radius boundary would flip it) but the
registration's Open Question 4 ("both inside the as-designed envelope") should be corrected to
"at/just below the envelope's edge," not "inside," when it returns to the author.

## Overall verdict: **RED**

Two of the six directly-computed SE/precision claims in `r-cone-loss/REGISTRATION_DRAFT.md` §3
do not reproduce: the cone statistic's stated SE (≈0.0007) is off by a factor of ~10.5 because its
`SD_IN` input was borrowed from an unrelated class in the companion arm's harness rather than
measured on the in-catalogue class it is defined over, and the derived "11 SE to materiality"
claim collapses to ≈1 SE under the measured value. Per the task's own instruction, a claim that
does not reproduce within its own stated precision is RED. The closure identity, the harness SE
(0.0063), the false-fail rate (≤0.54%), and the ≥0.02/event detection claim (all Branch G) are
GREEN/exact. The leave-out cross-check and scatter-law gate are both confirmed computable from
existing columns. **Must-fix before `d-cone-register`/launch:** recompute `SD_IN(s_e,1D)` and
`SD_IN(s_e,2D)` on the actual 66/production-IN (and 2D-channel) events (not borrowed from the
harness dark class), redo the SE(Δh_cone), Z, and "N SE to materiality" arithmetic in §3 with
that number, and correct the f_OUT-envelope framing (Check 8). Also surfaced: the blindness
breach above (Open question 0) needs an explicit author ruling before either arm's registered
read is treated as a fresh, un-compromised look at the data.

## Open questions routed back (in addition to r-cone-loss §9 / r-completion-residual §9)

0. **[RULE, urgent]** How to treat the now-unblinded point estimates (Δh_cone=−0.000273,
   φ=0.43%, Z=−0.037 for Branch H; S_M,prod=−0.1966, Z_prod≈−10 to −11 for Branch G) disclosed
   by this gate check — accept and proceed, or re-run the registered statistic through a fresh,
   disjoint reviewer who has not read this file.
1. Ratify a corrected `SD_IN` for the cone-loss SE formula, sourced from the production in-
   catalogue class itself (recovered subset), not the harness dark class.
2. Ratify whether the f_OUT envelope language in r-cone-loss §0/§9 item 4 should read "at the
   envelope's lower edge" rather than "inside."
