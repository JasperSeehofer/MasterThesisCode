# T1.2 S0-A Re-certification — Independent Readout Record

Launched under row #255 — tree 2 node T1.2 (independent reader). Foreground only, no git, no
ssh, append-only. Companion `t1_2_readout.json` carries the full numeric detail; this file is
the comprehension-first account.

## 1. What this run was and what it found, in one paragraph

The theta-consistent no-BH divisor (`theta_phi_divisor=on`) was re-run through the exact S0-A
control cell that previously failed as an instrument defect (Z_b = -3.68, Z_s = -7.08). The
b-axis is now a null: Z_b = -0.676, comfortably inside the registered `|Z_b| <= 3` band, and the
measured score_b (-0.289) lands within 0.021 of the pre-registered exact prediction (-0.268) —
both the loose band and the sharp exactness check pass. The s-axis did **not** clear: Z_s =
-5.971, still far outside `|Z_s| <= 3`. This was itself the registered, expected outcome for a
divisor-only run (no enlarged candidate ball) — the registration document predicted exactly this
number in advance (-0.073 ± 0.012, Z ~ -6) and said explicitly it would not be a falsifier. So:
**mechanism (i), the theta-inconsistent divisor, is CONFIRMED as the b-axis cause. Mechanism
(ii), candidate-ball truncation, remains unconfirmed** — it has a registered prediction (E12) but
that prediction requires a different run (enlarged ball + corrected secant) that has not
happened yet.

## 2. Independent re-derivation — confirms the driver exactly

Re-implemented from the raw `event_likelihoods.csv` files per (seed, node): h=0.73 filter, dedup
on `event_idx` (keep-last), `ln(combined_*)` where positive, inner-join the ±nodes per
(seed, event_idx), pool mean/SEM/Z over all seeds and events (unweighted per-event scatter, per
prereg §4.1). Reproduces the driver's own numbers to the last reported digit on every field
(scores, n_pooled, GATE ENG mean fraction moved). No correction to the driver's arithmetic was
needed.

| channel | statistic | mean | SEM | Z | n |
|---|---|---:|---:|---:|---:|
| `ln_L_no_bh` (registered primary) | score_b | -0.28878 | 0.42705 | **-0.6762** | 461 |
| `ln_L_no_bh` (registered primary) | score_s | -0.07196 | 0.012051 | **-5.9711** | 461 |
| `ln_L_with_bh` (secondary) | score_b | +0.13830 | 0.36465 | +0.3793 | 461 |
| `ln_L_with_bh` (secondary) | score_s | -0.02920 | 0.014409 | -2.0268 | 461 |

**Per seed (`ln_L_no_bh`):** 900101 (n=106) score_b Z=-1.548, score_s Z=-3.450; 900102 (n=120)
score_b Z=-1.459, score_s Z=-3.784; 900103 (n=105) score_b Z=+1.911, score_s Z=-1.974; 900104
(n=130) score_b Z=+0.870, score_s Z=-2.501. All four seeds keep the **same sign on score_s**
(negative), but **score_b now spans both signs** (2 negative, 2 positive) — a materially more
null-like pattern than the previous run, where score_b was uniformly negative across seeds.
Notably, the measured per-seed score_b means (-1.714, -1.283, +1.112, +0.659) reproduce the
forensic's pre-registered rho-based per-seed forecast (-1.71, -1.26, +1.17, +0.69) almost to the
digit — an independent confirmation of mechanism (i) that goes beyond the pooled Z alone.

**By class** (`L_cat_no_bh == 0` ≡ dark): dark (n=5 pooled) scores exactly zero on both axes,
unchanged (`combined_no_bh` bit-identical across all five theta-nodes — instrument-identity
check passes, as before); matched (n=456) carries essentially the entire pooled pull on both
axes (score_b Z=-0.676, score_s Z=-5.974) — same structure as the previous run.

## 3. Falsifier evaluation

**F1 (mechanism i, b-axis).** Registered band `|Z_b| <= 3`; sharper exactness check
`|score_b - (-0.268)| <= 0.10`. Measured Z_b = -0.676 (**PASS**, band) and measured score_b =
-0.28878, off by only 0.0208 from the -0.268 exact prediction (**PASS**, exactness). Both the
loose and the sharp registered checks pass. One note on the orchestrator's framing: "Z_b → -0.62
± 0.43" conflates the predicted score_b (-0.27) with its predicted SEM (0.43) as if it were an
uncertainty on Z itself; the registered text's actual tolerance is the ±0.10 exactness band on
score_b, which the run clears with room to spare (0.0208 vs 0.10). **Verdict: CONFIRMED.**

**s-axis band.** Registered band `|Z_s| <= 3`. Measured Z_s = -5.971 — **FAILS**, but as
registered in advance: `PHYSICS_CHANGE_THETA_DIVISOR_20260830.md` §5.6/§9(F2) predicted this
exact divisor-only configuration (sky_cone_k = 1.5, no z_window_k companion) would land at
score_s = -0.073 ± 0.012 (Z ~ -6) and stated in writing that this would **not** be a falsifier of
the divisor change. The measured value (-0.07196 ± 0.012051, Z = -5.971) matches that advance
prediction almost exactly. **The B0-A′ INSTRUMENT-DEFECT (s) STOP stands, unchanged in kind from
the previous run** — this run's flags were never intended to touch it.

**Change relative to the previous (no-divisor) run.**

| | previous (no divisor) | this run (divisor on) | Δ |
|---|---:|---:|---:|
| score_b mean | -1.61646 | -0.28878 | +1.328 |
| Z_b | -3.6764 | -0.6762 | +3.000 (|Z| down 82%, 5.4x) |
| score_s mean | -0.08625 | -0.07196 | +0.0143 |
| Z_s | -7.0786 | -5.9711 | +1.107 (|Z| down 16%) |

The b-axis moved by a mechanism-sized amount (|Z_b| fell 5.4x); the s-axis moved only modestly
(|Z_s| fell 16%). This asymmetry is itself registered physics, not a surprise: the divisor's
b-slope is C_b = -2.25 per unit b, while its s-slope is C_s = -0.024 per unit (linear) s — two
orders of magnitude smaller (§5.3). The divisor is no-BH-only and a function of the full
theta=(b,s), so it does perturb the s-nodes slightly (rho(0,√2)=0.9893, rho(0,1/√2)=1.0059, both
≠1) — enough to explain the small observed s-axis shift, not enough to move it inside band.

**E12 (window/cone mechanism, mechanism ii).** E12's registered prediction concerns a
**different, not-yet-run configuration**: the enlarged candidate ball (sky_cone_k=3.0 AND
z_window_k=4.0) scored under PA-HIER-32(d)'s corrected `score_s = score_lns - Es_null_det`
(re-derived fresh for that configuration, not reused from the theta_sites="2.2"/unsmeared
banked values). Under that configuration, E12 predicts the c-weighted statistic lands at
-0.005 ± 0.011 (Z -0.5, inside band) and the unweighted at +0.036 ± 0.016. **This T1.2 run tests
neither leg of that** — sky_cone_k stayed at its 1.5 default and z_window_k was never engaged
(decision-table item 3 of the divisor gate doc explicitly routes that knob to the orchestrator's
path choice, not yet exercised). What this run DOES confirm, to high precision, is the
*precursor* half of the same registered chain — "divisor-only, no enlarged ball" — landing right
on its own advance prediction. That is necessary-but-not-sufficient supporting evidence for the
truncation story that motivates E12; **E12 itself remains an open, un-run falsifier.** It would
be confirmed by a future run at sky_cone_k=3.0/z_window_k=4.0, re-scored under PA-HIER-32(d)'s
corrected secant, landing inside `|Z_s| <= 3`; refuted if that run still exceeds |Z_s| = 3.

## 4. Disclosure: which score_s form did the driver actually use?

`hier_s0_driver.py`'s `compute_scores()` is unedited source (confirmed by grep: no
`Es_null_det` term anywhere, no reference to PA-HIER-32). It computes the **old, superseded raw
linear secant** — `score_s = (lnL(s=√2) − lnL(s=1/√2)) / (√2 − 1/√2)` — not PA-HIER-32(d)'s
amended, bias-corrected `score_s = score_lns − Es_null_det`.

This is **consistent with the divisor gate doc's own authorizing text**: §5.6/§9(F2) of
`PHYSICS_CHANGE_THETA_DIVISOR_20260830.md` explicitly predicts THIS run's number using the raw,
uncorrected form (-0.073 ± 0.012, Z ~ -6 — matched almost exactly), and states the PA-HIER-32
corrected form is "in force" only once the enlarged-ball E12 measurement is made.

It is **in tension with PA-HIER-32(d)'s own scope note**, which reads, unqualified: *"the S0-A
re-certification (tree 2 T1.2, TREE2_CHARTER_20260830.md) runs UNDER PA-HIER-32 ... it must use
score_s and Z_s as defined here, not the superseded score_lns."* Taken literally, that sentence
applies to T1.2 as executed here, not only to a future enlarged-ball leg. **This reader discloses
the disagreement between the two registered texts and does not adjudicate it** — the run is
internally consistent with its own authorizing gate doc, but a literal reading of PA-HIER-32(d)
would call this specific execution non-compliant with its own scope note. Flagged for the
orchestrator/author to reconcile.

## 5. Verdict (read from the registered map, prereg §4.5)

- **b-axis: CERTIFIED.** Mechanism (i) CONFIRMED. REPORTED-ONLY cap carried without exception
  (PA-HIER-28 item 9).
- **s-axis: B0-A′ persists → INSTRUMENT-DEFECT (s) — STOP stands.**
- This finding licenses **nothing** beyond what row #255/the tree-2 charter already licensed. No
  Stage-P/F, no S0-B, no C1/C3 launch follows from a partial S0-A pass.

## 6. Cost

Measured: 13084.808 s wall × 14 cpu_per_job / 3600 = **50.885 CPU-h** (measured wall =
3.635 h).

Registered anchors:
- `TREE2_CHARTER_20260830.md` T1 branch 2: **"approx 11.5 CPU-h local (approx 6 if venue builds
  are cached)."**
- `PHYSICS_CHANGE_THETA_DIVISOR_20260830.md` §6 item 4: single-process serial ~6.3 h wall; with
  cached normaliser ~4.4 h wall; with 14-way row parallelism ~1.3 h wall.

Measured 50.89 CPU-h is **~4.4x the 11.5 CPU-h charter anchor** and **~8.5x the 6 CPU-h cached
anchor**. Measured wall (3.63 h) sits between the cached-normaliser (4.4 h — actual is better)
and the full 14-way-row-parallelism (1.3 h — actual is worse) bands, closer to the
cached-normaliser figure — suggesting the registered row-chunk-parallelism mitigation (§6 item 3)
was not fully engaged, though some speedup over naive single-core (6.3 h) evidently was. Per-cell:
off-truth cells averaged 701.65 s (11.7 min) this run vs ~169.5 s (2.8 min) per off-truth cell in
the previous (no-divisor) S0-A pass — roughly a 4.1x per-cell cost increase from the divisor's
per-host kernel integral pass, directionally consistent with the registered multiplier. Truth
cells (literal skip) were unchanged at ~62-77 s.

## 7. Orchestrator path decision of record

**T1.3 = the z-window/cone companion knob as its own gate** (the presenter's decision-table item
3, `PHYSICS_CHANGE_THETA_DIVISOR_20260830.md` §10); re-run s-nodes only after it. **S0-B stays
unlaunched.**

## 8. Scope note

This reader did not modify `candidate_dump_bi_run/` (owned by another reader), wrote no code, ran
no git or ssh operations, and worked entirely from files already on disk plus one local Python
re-derivation script (scratch, not committed).
