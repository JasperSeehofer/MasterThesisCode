# CLAIM — what the campaign-#53 2D bias is, and what it is not (2026-07-30)

Status: **CLAIM, NOT ESTABLISHED.** Written to be attacked. Every numbered claim
below carries its provenance and an explicit refutation route. The next session's
first job is to refute or confirm these, **not** to build on them.

Authored at the end of a session that ran four workflows (~4.0M subagent tokens)
and killed six candidate mechanisms, two of which were the author-agent's own.

## Provenance legend

| tag | meaning |
|---|---|
| **[LOCAL]** | re-measured this session from artifacts in this repo; reproducible now, offline |
| **[AGENT]** | measured by a subagent from `diagnostics/event_likelihoods.csv`, **which no longer exists** (/tmp evaporated) and whose cluster original is unreachable (SSH refused). NOT independently reproduced |
| **[DOC]** | read from a committed artifact (readout, runbook, derivation, code comment) |
| **[INFER]** | logical inference from [LOCAL]/[DOC] facts; no new measurement |

⚠ **The single biggest weakness of this claim set: every 2D *per-event* number is
[AGENT] and currently unreproducible.** 2D per-event likelihoods live only in the
3.2 GB/run `posteriors_with_bh_mass/` dirs on the cluster. The 2D *totals* are
[LOCAL] (see C2), but the *class split* that carries the headline is not.

---

## The claim in one paragraph

The +0.077 2D bias is arithmetically owned by the **dark** event class, which
swings +15.83 nats between channels (84% of the total +18.80). Mechanism: the GW
mass is near-exact while catalogue BH masses are not, so the 2D mass window and
kernel reject ~97–99% of impostor hosts; the 1512 dark events thereby lose the
catalogue leg that supplied −11.77 nats of *down*-pull, and fall back on the
completion term `B_num/D(h)`, which pulls *up*. Separately and more seriously,
under the realistic model **58% of the identified in-catalogue hosts rail at the
h = 0.86 prior edge**, so the 1D result is not a measurement centred on truth but
the crossing point of two railed, opposing runaways. **However**, attribution of
either effect to the *realized scatter* is confounded: campaign #51 → #53 changed
three variables simultaneously, and the control that would separate them was never
run.

---

## C1 — The 1D class budget [LOCAL, VERIFIED]

Σ over class of Δ ln p_i from h = 0.73 → 0.81, seed61000:

| | IN-CAT (76) | DARK (1512) | total |
|---|---|---|---|
| #53 real_r1 | **+2.48** | **−11.77** | **−9.30** |
| #51 idealized | −338.10 | −23.52 | −361.62 |

Command: per-event `posteriors/h_0_*.json`, in-cat = `host_galaxy_index >= 0` in
`prepared_cramer_rao_bounds.csv`. Reproduces the workflow's numbers exactly.

**Refute by:** recomputing on another realization/seed. If r2–r5 or seed62000
disagree in sign or order of magnitude, the claim is r1-specific.

## C2 — The channel totals [LOCAL, VERIFIED]

`ln P(0.81)/P(0.73)` read off the combined posteriors: **1D = −9.30, 2D = +9.51.**
The 1D value equals the per-event sum in C1 to 2 d.p., which validates the method;
the 2D value equals the workflow's derived total exactly. Channel difference
**+18.80 nats**.

**Refute by:** nothing cheap — this is a direct read of the delivered posteriors.

## C3 — 84% of the channel difference is the dark class [AGENT, NOT REPRODUCED]

Claimed split of the +18.80: IN-CAT +2.97, **DARK +15.83**. Derived 2D class
totals +5.45 / +4.06.

**This is the headline and it is the weakest-provenance number in the set.**

**Refute/confirm by (do this FIRST):** regenerate per-event 2D likelihoods and
recompute. Either re-read `posteriors_with_bh_mass/h_0_{725,73,735,81}.json` on
the cluster (4 files, not the whole 3.2 GB), or re-run one evaluate with the
diagnostics CSV enabled. Note C2 constrains the *sum* of the split to +18.80, so
only the partition is at risk.

## C4 — The mechanism: impostor rejection → completion fallback [AGENT + INFER]

Supporting [AGENT] measurements, all unreproduced: at h = 0.73, 64.7% of dark
events have `L_cat_with_bh == 0` exactly (vs 32.5% in 1D); 488 of the 1095 events
with a nonzero 1D catalogue term have an identically-zero 2D term at every h, 487
of them dark; survivors are suppressed by median `L_cat_2D/L_cat_1D` = 7.8e-3;
Σ ln(L_cat_2D/L_cat_1D) tilts −504.8 nats over 0.73→0.81 for dark events but
+0.27 (i.e. h-flat) for in-cat.

Independent [LOCAL] support for the *premise*: P6 work measured the mass rejection
as strictly one-sided (193 low-side vs 1 high-side) because σ_Mz/M_z ≈ 1e-4 while
catalogue σ_lnM ≈ 1.28, making the window's upper leg vacuous.

**Refute by:** the same regeneration as C3. Also check whether the completion leg
`B_num/D(h)` is genuinely up-tilted in *this* venue rather than assumed to be.

## C5 — 58% of in-catalogue hosts rail at the prior edge [LOCAL, VERIFIED]

Per-event 1D argmax over the full prior [0.60, 0.86], 76 in-catalogue hosts,
seed61000:

| | median peak | at 0.86 edge |
|---|---|---|
| #53 real_r1 | **0.860** | **44/76 = 57.9%** |
| #51 idealized | 0.730 | 4/76 = 5.3% |

Corroborating [AGENT]: in-cat σ_h 3.2e-4 → 2.7e-2; the 3 golden events retain
4.6e-4 of their curvature (this last is [LOCAL] — `realistic_scores.csv`,
`golden_retained`).

**This claim is independent of C3/C4 and survives even if they fall.** It says the
identified hosts stopped constraining H₀ and prefer the top of the prior — so the
1D headline of 0.700–0.740 is a crossing of runaways, not a centred measurement.

**Refute by:** recompute on other realizations/seeds; check whether the argmax
concentration at 0.86 is an artifact of the prior's upper bound (widen the grid
above 0.86 and see whether the peaks move further or stop).

## C6 — Attribution is confounded; the decisive control was never run [DOC + INFER]

| | catalogue | host-z kernel | normalization |
|---|---|---|---|
| #51 idealized | unscattered | **point (δ)** | **generator_marginal** |
| #53 realistic | scattered | **volume_deconv** | **absolute_marginal** |

[DOC] `IDEALIZED_BASELINE_READOUT.md:53` ("point-evaluated by the production
δ-kernel"); `RUNBOOK_NEXT_SESSION_5.md:75-78` ("The guards REFUSE
point-kernel/`generator_marginal` on a scattered catalogue"); guard code
`bayesian_statistics.py:310-325` (`if not catalogue_scattered: return`).

[INFER] The σ→0 P5 gate cannot be the missing control: `sigma_scale=0` yields a
**byte copy** of the parent catalogue (`observed_realization.py:201`) but leaves
the `z_error` column intact, so a width-integrating `volume_deconv` kernel could
not reproduce a δ-kernel posterior **byte-identically** (md5 `1e81ba22` 1D /
`733c8d32` 2D). Therefore P5 ran #51's estimator, and **no run anywhere varies the
estimator at fixed catalogue.**

⇒ **"The bias switches on with the realized scatter" is NOT established.** Three
variables moved at once. This invalidates a premise the previous session asserted
repeatedly.

**Refute by:** reading `sig0_control/run_metadata_0.json` on the cluster. If it
records `absolute_marginal` + `volume_deconv` **and** still matched #51 byte-for-byte,
this claim collapses and the estimator is proven inert. **One file; check it first.**

## C7 — Candidate mechanism for C5: the host-z kernel omits selection [DOC + INFER]

`bayesian_statistics.py:4201-4207` weights the host-z numerator kernel by
`w_pop = dV_c/dz/(1+z)` — the *cosmic* prior — with **no `p_det` and no catalogue
selection φ_cat**. Deconvolving a wide photo-z against a monotonically rising
volume prior with no selection turnover shifts the host-z estimate up by
≈ 2(σ_z/z)² (mode: z → [z + √(z²+8σ²)]/2). At the measured σ_z/z = 0.25–0.49 for
these hosts that predicts **+11% to +36%** h inflation → h_eff 0.81–0.99 → rails
at 0.86. **Observed: rails at 0.86.** #51 cannot exhibit this — a δ-kernel has zero
width by construction.

**Status: a prediction that matches, not a measurement of the code's kernel.**

**Refute by:** compute the kernel's actual induced host-z shift numerically for the
76 hosts at their real σ_z, rather than via the mode formula. Also note the local
`z_error` column is stale vs the cluster parent (#40b PV width), so the σ_z/z
inputs are indicative.

## C8 — The 2D posterior is reparametrization-dependent [AGENT, NOT REPRODUCED]

Rescaling the mass coordinate by a constant C in the 2D channel walks the MAP
across the grid: C=1 → 0.8133, C=0.3 → 0.7821, C=0.1 → 0.7438, C≤0.01 → rails at
0.600. The 1D channel is exactly invariant. Cause: a 4D numerator against a 3D
selection denominator (`D(h)`, `:1056-1145`, is channel-common and never
mass-marginalised).

**If true this is a well-posedness failure, independent of any bias:** a published
2D number that moves with an arbitrary unit choice is indefensible.

**Refute by:** re-run the C-scaling on regenerated per-event 2D data. Check whether
the claimed invariance of 1D is exact and whether the 2D dependence is really
arbitrary rather than a fixed physical scale entering.

---

## What is explicitly NOT claimed

1. **Not claimed: that any of this is a *defect* rather than correct physics.**
   Rejecting impostors is what the 2D channel is *for*. The bias appears because
   the two mixture legs disagree about h for the same dark population — the
   impostor leg rails at 0.64, the completion leg pushes past 0.86, truth is 0.73.
   **Which leg is wrong is undetermined.** That is the open physics question.
2. **Not claimed: that the realized scatter causes the bias** — see C6.
3. **Not claimed: that the 1D channel is trustworthy.** C5 says the opposite.
4. **Not claimed: any number for a headline H₀.**

## Exonerated — do NOT re-open without new evidence

catalogue Jacobian · Fisher frame · p_det estimator choice · p_det inside/outside ·
h-prior sensitivity · `volume_trunc` · the z leg (channel-common) · the ln-M draw
itself (mean |Δln M| ≤ 0.0009 dex) · realization plumbing (σ→0 byte-identical in
both channels) · candidate-window **membership** (exact removal moves MAP
0.81→0.82, wrong sign) · mass-kernel **family** (bounded +0.002) · **Option-A
calibration drift** β_G/Σ_glob (= the exact h⁻³ volume Jacobian,
(0.73/0.81)³−1 = −26.80%; residual is 1D-only, +0.017 in h) · **HA as the bias
owner** (correction moves r1 0.8133→0.8492, r2 0.7820→0.8527 — wrong sign) ·
**HC** mixture-floor/zero-handling (physics-floor never fires: 65,108/65,108 cells
nonzero, 0 excluded events in all 16 combined posteriors) · **HB** hard mass window
as support truncation (tilt −0.317 nats = 0.063% of the target, sign-inverted,
40–50× too small).

## Errors made this session — do not inherit them

1. **Units.** "1 nat/unit-h ≈ 4.5e-4 in h" is per *nat-per-unit-h*. Applied to
   window-integrated nats it understates by ~12×. Correct: Δh = Δnats·σ_h²/Δh_window
   ≈ 4.9e-3 per nat over a 0.08 window. No verdict changed, but budgets were
   misquoted.
2. **The "why is 1D spared?" screen has no discriminating power for mass-channel
   hypotheses.** `handler.py:592` returns the 1D candidate list with a redshift
   filter only; `:605` adds the mass filter for 2D. The 1D channel never sees the
   mass window, so any mass hypothesis passes that screen trivially.
3. **"#51 is a non-control because no impostor passes the mass window at σ=0" is
   false.** Measured: 153,473 impostors pass in the unscattered case — #51 is a
   *stronger* exposure, hence a genuine control for HB.
4. **Precision.** Never use the 4-dp `w_G` log line (`:2335`) for residual-level
   work; use `D(h)` (`:1145`, 7 s.f.) − `β_Ḡ(h)` (`:1297`).
5. **`ideal_61000.csv` carried a wrong `w_G`** (0.0686 vs the log's 0.1625 at
   h=0.6). Any ideal↔real comparison through it is void. The realized extract
   matched the logs to 4.5e-5.

## Loose thread, unexamined

`w_G(0.73) = 0.0697` (derived from two independent agent numbers that agree) versus
the empirical in-catalogue rate 76/1588 = **0.0479** — a 45% discrepancy in the
quantity whose h-derivative supplies +394 nats/unit-h. Flagged as a diagnostic, not
a finding. Nobody has looked.
