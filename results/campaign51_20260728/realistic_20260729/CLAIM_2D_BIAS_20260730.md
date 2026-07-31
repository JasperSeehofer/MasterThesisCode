# CLAIM — what the campaign-#53 2D bias is, and what it is not (2026-07-30)

Status: **CLAIM, NOT ESTABLISHED.** Written to be attacked. Every numbered claim
below carries its provenance and an explicit refutation route. The next session's
first job is to refute or confirm these, **not** to build on them.

## 2026-07-30 — Gate B/C adjudication applied

Gates A–B–C of [`../RUNBOOK_NEXT_SESSION_6.md`](../RUNBOOK_NEXT_SESSION_6.md)
were executed 2026-07-30. **All [AGENT] numbers reproduced exactly** from
re-pulled cluster data (Gate A2; the diagnostics CSV is re-staged at
`seed61000/real_r1/diagnostics/event_likelihoods.csv`). The per-claim verdicts
written into the sections below were applied from
[`gate_b_20260730/ADJUDICATION_20260730.md`](gate_b_20260730/ADJUDICATION_20260730.md)
— **the adjudication of record** for this date.

The project's bias-resolution history was consolidated the same day into
[`gate_b_20260730/BIAS_HISTORY_LEDGER.md`](gate_b_20260730/BIAS_HISTORY_LEDGER.md)
(every H₀-bias hypothesis tested 2026-03 → 2026-07-30). **Its "DO NOT RE-TRY"
union is BINDING alongside this file's Exonerated list.** A block of historical
exonerations is **absent** from the list below — the ledger's §2 ⚠ entries
(items 1–17) — and those are the live re-litigation risk. Read the ledger before
proposing any mechanism.

Also from the ledger, and load-bearing for how the +0.077 should be read:
**[RATIFY-M6] already designates the `absolute_marginal` + `volume_deconv` 2D
pairing a CANDIDATE** — "necessary, not established sufficient"
(`docs/derivations/mass_marginal_2d_kernel.md:1-17, 690-706`; ledger §3). The
+0.077 therefore sits on a **designated-candidate estimator, not on ratified
ground.**

Authored at the end of a session that ran four workflows (~4.0M subagent tokens)
and killed six candidate mechanisms, two of which were the author-agent's own.

## Provenance legend

| tag | meaning |
|---|---|
| **[LOCAL]** | re-measured this session from artifacts in this repo; reproducible now, offline |
| **[AGENT]** | measured by a subagent from `diagnostics/event_likelihoods.csv`, **which no longer exists** (/tmp evaporated) and whose cluster original is unreachable (SSH refused). NOT independently reproduced. **[2026-07-30: superseded — the CSV was re-staged from the cluster and every [AGENT] number reproduced exactly; the tags on C3/C4/C8 are retagged accordingly]** |
| **[DOC]** | read from a committed artifact (readout, runbook, derivation, code comment) |
| **[INFER]** | logical inference from [LOCAL]/[DOC] facts; no new measurement |

⚠ ~~**The single biggest weakness of this claim set: every 2D *per-event* number is
[AGENT] and currently unreproducible.** 2D per-event likelihoods live only in the
3.2 GB/run `posteriors_with_bh_mass/` dirs on the cluster. The 2D *totals* are
[LOCAL] (see C2), but the *class split* that carries the headline is not.~~

⚠ **[2026-07-30 adjudication, §6.1 — replaces the struck paragraph]** The
provenance defect is **cured**: the diagnostics CSV is re-staged at
`seed61000/real_r1/diagnostics/event_likelihoods.csv` and **all [AGENT] numbers
reproduced exactly** (`gate_b_20260730/attack_c3_c4.py`; re-run by the
adjudicator). The remaining weakness is narrower and stated as such:
~~**r1 is still the only realization with per-event 2D data.**~~

⚠ **[UPDATE 2026-07-30, off-r1 replication — supersedes the line above]**
Per-event 2D likelihoods are now **local for ALL 10 runs**, via each run's
`diagnostics/event_likelihoods.csv`; the `combined_with_bh` column was verified
**bit-identical to the r1 per-event JSONs at ~1e-16 relative difference**
(`attack_c3_c4_allruns.py` docstring, lines 6-9), so the whole C3/C4 computation
runs from the CSVs alone. **There is no r1-only data limitation left.** See
[`gate_b_20260730/c3c4_allruns_summary.md`](gate_b_20260730/c3c4_allruns_summary.md).

---

## The claim in one paragraph

The +0.077 2D bias is arithmetically owned by the **dark** event class, which
swings +15.83 nats between channels (84% of the total +18.80). Mechanism: the GW
mass is near-exact while catalogue BH masses are not, so the 2D mass window and
kernel reject ~97–99% of impostor hosts; the 1512 dark events thereby lose the
catalogue leg that supplied −11.77 nats of *down*-pull, ~~and fall back on the
completion term `B_num/D(h)`, which pulls *up*~~ **[2026-07-30, §6.3 — struck.
The surviving dark catalogue legs are *de-weighted*, not deleted (98.5% of the
channel difference is carried by survivors, 1.5% by the 2D-zeroed events: see
C4-amended); and the up-pull is carried by the `(1−w_G)` prefactor, not by
`L_comp`, which pulls DOWN for dark events: see C10]**. Separately and more seriously,
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

**[2026-07-30 adjudication §1] → FINDING, closed.** The refutation route above
was executed (Gate A3) and **failed to refute**: the class structure replicates
in sign and order across all 10 realistic runs and both seeds (in-cat
+1.27..+5.38 vs idealized −338/−248; dark −11.8..−14.1).

## C2 — The channel totals [LOCAL, VERIFIED]

`ln P(0.81)/P(0.73)` read off the combined posteriors: **1D = −9.30, 2D = +9.51.**
The 1D value equals the per-event sum in C1 to 2 d.p., which validates the method;
the 2D value equals the workflow's derived total exactly. Channel difference
**+18.80 nats**.

**Refute by:** nothing cheap — this is a direct read of the delivered posteriors.

**[2026-07-30 adjudication §1] → FINDING, closed.** Independently reconstructed
by the C8 attacker to 3.6e-12 nats and cross-checked (1D −9.30, 2D +9.51).

## C3 — 84% of the channel difference is the dark class ~~[AGENT, NOT REPRODUCED]~~ **[LOCAL, VERIFIED]**

Claimed split of the +18.80: IN-CAT +2.97, **DARK +15.83**. Derived 2D class
totals +5.45 / +4.06.

~~**This is the headline and it is the weakest-provenance number in the set.**~~

**Refute/confirm by (do this FIRST):** regenerate per-event 2D likelihoods and
recompute. Either re-read `posteriors_with_bh_mass/h_0_{725,73,735,81}.json` on
the cluster (4 files, not the whole 3.2 GB), or re-run one evaluate with the
diagnostics CSV enabled. Note C2 constrains the *sum* of the split to +18.80, so
only the partition is at risk.

**[2026-07-30 adjudication, §6.2] → FINDING; retagged [LOCAL, VERIFIED].** The
route above was **executed**: Gate A2 reproduced the split **exactly** from
re-pulled cluster data (+2.97 / +15.83, dark share **84.2%**), and the
adjudicator re-ran `attack_c3_c4.py` this session. **Numbers unchanged.** The
provenance defect that made this the weakest number in the set is cured.

**Caveat — r1-only partition.** ~~The diagnostics CSV exists for no other run, so
the 2D partition itself is single-realization. C1's class structure replicates
everywhere, which makes r1-specificity unlikely but does not exclude it. Closing
this needs the diagnostics CSV emitted on **every** run (instrumentation item,
already routed to plain GSD).~~

**[UPDATE 2026-07-30, off-r1 replication — the caveat is LIFTED, with one
refinement]** C3 was recomputed on all 10 realistic runs from the newly staged
diagnostics CSVs
([`gate_b_20260730/c3c4_allruns_summary.md`](gate_b_20260730/c3c4_allruns_summary.md)):

- **The dark component replicates in ALL 10 runs, both seeds: +15.83 to +17.14
  nats, always positive.** The load-bearing statement — the dark class carries
  the channel difference — is now **replication-hardened**, no longer r1-only.
- **The in-cat component is small and realization-noisy** (−1.83 to +2.97 nats)
  and **flips sign** in seed61000/real_r3 (−1.83), traced to a **single
  high-leverage in-cat event, `event_idx 889`**, whose own channel difference
  swings +1.98 (r1) → −2.04 (r2) → −3.30 (r3) across noise realizations of the
  *same* 76-event in-cat class.
- Consequently the **dark share ranges 84.2%–112.5%** (mean 91.6%; shares above
  100% are exactly the runs where the in-cat leg goes negative). **The precise
  "84%" is r1-specific**; what replicates cleanly is the qualitative claim —
  **dark ≫ in-cat in magnitude, and dark always positive and dominant.**

## C4 — The mechanism: impostor rejection → completion fallback ~~[AGENT + INFER]~~ **→ split 2026-07-30: C4-obs [LOCAL, VERIFIED] FINDING · C4-mechanism REFUTED AS STATED**

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

**[2026-07-30 adjudication, §6.3] C4-obs → [LOCAL, VERIFIED], FINDING.** Every
measurement listed above reproduces exactly: 64.7% of dark events with
`L_cat_with_bh == 0`; 488/1095 2D-zero at every h, 487 of them dark; median
suppression 7.78e-3; dark Σ ln(L_cat_2D/L_cat_1D) tilt −504.8.

**[2026-07-30 adjudication, §6.3] C4-mechanism → REFUTED AS STATED.** ~~impostor
rejection → completion fallback~~ fails on two exact grounds (attacker 1, algebra
verified to 6.2e-13 on all 65,108 cells):

1. Writing p = C(1+R) with C = (1−w_G)·L_comp **channel-common**, `ln C`
   **cancels identically** from the per-event channel difference. "Falls back on
   the completion term, which pulls up" cannot appear in the +15.83 **as an
   accounting statement about the channel difference**.
2. The flagship evidence — 487 events 2D-zeroed at every h — carries **+0.24
   nats = 1.5%** of the +15.83; the 491 both-dead events carry exactly **0.00**.
   **98.5% (+15.60) is carried by the 534 survivors.** Deletion is not the
   mechanism: the zeroed events' 1D legs were already negligible.

**Amended mechanism (promoted).** The mass kernel **de-weights** the surviving
dark catalogue legs: dark mean catalogue mixture weight **0.0354 → 0.0061** at
h = 0.73, a factor 5.8. Exact budget:

> **+15.83 = 0 (completion, cancels) + 19.10 (loss of the 1D catalogue
> down-tilt; +18.87 survivors, +0.24 zeroed) − 3.27 (residual 2D tilt).**

The dark class-summed opposition over 0.73→0.86 collapses **−24.46 → −0.63**
nats and its argmax moves **0.640 → 0.785** — landing next to the dark
**completion leg's own argmax 0.810 ≈ the delivered 2D MAP 0.8133**.

**Caution against over-correcting (adjudicator's):** the refutation is of the
*accounting*, not of the completion leg's role in the **absolute** 2D position.
Once the dark catalogue leg is de-weighted, the 2D posterior *does* sit where the
channel-common completion/prefactor structure puts it — but the up-pull is
**prefactor-carried**: dark Σd ln[(1−w_G)L_comp] = **+7.33 = +30.04** from
N·Δln(1−w_G) **minus 22.72** from `L_comp` itself. **`L_comp` pulls DOWN for dark
events** (only 39.1% positive tilts). See **C10**.

**Caveat:** ~~the entire partition is seed61000/real_r1, the only run with a
diagnostics CSV.~~

**[UPDATE 2026-07-30, off-r1 replication]** The C4 measurement set **replicates
closely across all 10 runs** ([`gate_b_20260730/c3c4_allruns_summary.md`](gate_b_20260730/c3c4_allruns_summary.md)):
dark 2D-zeros **62.9–64.7%**; zero-at-every-h events **472–518** (dark subset
**469–514**); median suppression **6.31e-3 to 8.05e-3** (~7e-3); dark tilt
**−475.88 to −552.43 nats**. The C4-obs FINDING is no longer
r1-scoped. (The C4-amended *budget* decomposition above remains an r1 partition
until it too is recomputed off-r1.)

## C5 — 58% of in-catalogue hosts rail at the prior edge [LOCAL, VERIFIED] **→ FINDING (2026-07-30), interpretation AMENDED**

Per-event 1D argmax over the full prior [0.60, 0.86], 76 in-catalogue hosts,
seed61000:

| | median peak | at 0.86 edge |
|---|---|---|
| #53 real_r1 | **0.860** | **44/76 = 57.9%** |
| #51 idealized | 0.730 | 4/76 = 5.3% |

~~Corroborating [AGENT]: in-cat σ_h 3.2e-4 → 2.7e-2;~~ **[2026-07-30, §6.4 —
replaced by the measured, defined values: per-event σ_h 0.235–0.311 realistic vs
0.043–0.053 idealized; σ_class 0.043–0.170.]** the 3 golden events retain
4.6e-4 of their curvature (this last is [LOCAL] — `realistic_scores.csv`,
`golden_retained`).

**This claim is independent of C3/C4 and survives even if they fall.** It says the
identified hosts stopped constraining H₀ and prefer the top of the prior — so the
1D headline of 0.700–0.740 is a crossing of runaways, not a centred measurement.

**Refute by:** recompute on other realizations/seeds; check whether the argmax
concentration at 0.86 is an artifact of the prior's upper bound (widen the grid
above 0.86 and see whether the peaks move further or stop).

**[2026-07-30 adjudication, §6.4] → FINDING; the designated refutation attempt
FAILED; the interpretation is AMENDED.**

*The rail is real, not an edge artifact.* Railed profiles are genuinely concave
(86–96% all-negative second differences on the uniform 0.80–0.86 stretch,
|d²| ~1e11× roundoff), and top-K parabola vertices give finite implied peaks
**h_eff = 0.93–1.05** (median), stable over K = 3–9 in all 10 runs, with the
extrapolator **validated in-band** (median error < 0.007 at the relevant
standoffs). Independently, the C7 attacker rebuilt the single-host kernel on a
grid extended to h = 2.4 and found **interior** peaks (median ≈ 1.12). The 0.86
concentration is a **clipped real runaway.** Replicates **10/10 runs (54–67% at
the edge** vs 2.4% flat-surface expectation, 5.3% idealized).

*Fair-framing amendment — binding for any write-up.* **Per event the rail is
cosmetic**: median peak-to-truth Δln p **0.072–0.134 nats = 0.30–0.47 σ_event**
(implied per-event σ_h 0.235–0.311 vs 0.043–0.053 idealized); only **0–1.3%** of
events exceed 1σ. "58% of hosts rail" invites over-reading. But it is **not
noise**: the tilt is coherently same-signed and the **class-summed** displacement
is **+3.4 to +6.1 σ_class** above truth in 8/10 runs, with two independent σ
routes agreeing and LOO never moving the argmax. A correct estimator under large
σ_z is wide but centred; a ≥3.4σ coherent class displacement is not that.
**"Not a centred measurement" stands.**

*Attribution amendment — two components, not one.*
- The **per-event argmax rail lives in the catalogue leg**: L_cat argmax at 0.86
  for **66/74 = 89.2%**, and L_cat carries a median **96.3%** of the in-cat
  mixture at h = 0.73. **C7 is its confirmed mechanism.**
- The **class-summed mixture rise** (**+3.92 nats** 1D over 0.73→0.86) is **~82%**
  carried by the ~9%-weight **completion admixture**: the catalogue leg's own
  class sum peaks at **0.760** with a rise of only **+0.80**, because a few golden
  events' large negative tilts nearly cancel the many small positive ones (per-event
  median **+0.308**, **93.2%** positive).

  These are different summary statistics of the same data, both verified, and they
  are **not** in conflict. Both components push up: C5 has two contributors.

*Crossing-of-runaways framing: sustained and sharpened.* Dark-only argmax
**0.640**, in-cat-only **0.860** in 10/10 runs, combined **0.700–0.742**
(idealized: 0.600 dark / 0.730 in-cat / 0.730 combined = truth). Class slopes at
the MAP are 5× smaller and total curvature ~1000× smaller than idealized;
**dh*/dε leverage 1500–2400×** idealized; a ±1/√N_class Poisson reweight moves
the combined MAP by up to **0.025** (0.12–0.51 σ_h) vs **0.0000** idealized;
λ-scan λ = 0 → **0.635–0.644**, λ = 2 → **+0.011..+0.049**. The run-to-run MAP sd
0.006–0.008 is **not** evidence of robustness against class composition — the CRB
file is byte-identical across realizations of a seed, so class membership never
varied. Independent corroboration: **C9's counterfactual moves 1D 0.732 → 0.643**,
i.e. 1D centredness is contingent on the same mis-calibration.

*Adjudicator's caution, applied consistently below:* the leg-split (82%/18%) is
r1-only, and the near-flat combined profile is precisely why **any** single
counterfactual MAP displacement — including the ones quoted here and in C9 —
must not be read alone as ownership.

## C6 — ~~Attribution is confounded; the decisive control was never run~~ **RESOLVED 2026-07-31: cell B ran — THE ESTIMATOR OWNS IT** [DOC + INFER → LOCAL, MEASURED]

**[RESOLUTION 2026-07-31 — `CELLB_READOUT_20260731.md` is the record.]** The 2×2
cell B (unscattered #51 catalogue + CRB through the #53 estimator, jobs
6103219/6103220, code `7fd60bb`) delivered the pre-registered outcome 1:
**B 1D = 0.7450, 2D = 0.7900** ⇒ estimator effect (B−A) = **+0.015 (1D) /
+0.060 (2D)**; scatter effect (C−B) = −0.005 / +0.023. 72% of the 2D
displacement is the estimator configuration alone, with exact host redshifts.
The in-cat catalogue-leg rail is 90.7% in B vs 89.2% in C (statistically
identical — C7 confirmed against the true parent `z_error` widths, the
staleness-free check); the dark channel difference is +18.0 nats **unscattered**
(estimator-borne, not scatter-borne); w_G(h) is bit-identical to the #53 curve
(C9 transfers verbatim). The realistic host-observation model is largely
exonerated for the headline biases. The original confound statement below is
retained as the historical record.

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

**[2026-07-30 adjudication, §6.5] → FINDING, CONFIRMED by Gate A1.** The one-file
check was executed: `sig0_control` ran **`generator_marginal` + point kernel**.
The [INFER] above is correct and the claim does **not** collapse — **no estimator
control exists anywhere.**

**Resolution in flight:** the pre-registered 2×2 **cell B** (unscattered
catalogue × the #53 estimator) is running as jobs **6101146 / 6101147**;
pre-registration at
[`PREREGISTRATION_2x2_cellB.md`](PREREGISTRATION_2x2_cellB.md).

**Dated pre-readout statement — registered 2026-07-30, BEFORE cell B lands**
(adjudication §3): everything relevant in B is scatter-independent or
scatter-inert by prior measurement, so the five Gate-B reports **jointly predict
the "estimator owns it" outcome: B ≈ C in both channels** (2D ≈ 0.78–0.82,
in-cat class argmax ≈ 0.86, 1D ≈ 0.70–0.74 as a crossing). A contrary result is
therefore unmistakably a **surprise** — and per §3 Outcome 2 would falsify the
transfer of C7 and C9 to the delivered posterior, mandating re-examination of the
ln-M-draw and candidate-window-membership exonerations for a support-structure
loophole before anything else.

**History already supplies an off-venue analog of cell B** — added from
[`gate_b_20260730/BIAS_HISTORY_LEDGER.md`](gate_b_20260730/BIAS_HISTORY_LEDGER.md)
§3, which post-dates the adjudication and so is not in it. `mass_ab_20260727`'s
**cell A′** ran exactly the #53 pairing (`absolute_marginal` + `volume_deconv` +
gaussian mass kernel) on an **UNSCATTERED mock catalogue** and measured
**1D MAP 0.73 / 2D MAP 0.80** (`mass_ab_20260727/MASS_KERNEL_AB_READOUT.md:23-30`);
the three-way per-leg A/B attributed **86.7%** of the 2D movement to the
δ → `volume_deconv` kernel switch
(`threeway_ab/THREEWAY_AB_READOUT.md:41-56`). That is **independent historical
support for the "estimator owns it" expectation** — but it is **venue-different**
(seed1000 deep mock, not this campaign's venue), so **cell B remains the on-venue
decider.**

## C7 — Candidate mechanism for C5: the host-z kernel omits selection ~~[DOC + INFER]~~ **[LOCAL, VERIFIED — MEASURED]**

`bayesian_statistics.py:4201-4207` weights the host-z numerator kernel by
`w_pop = dV_c/dz/(1+z)` — the *cosmic* prior — with **no `p_det` and no catalogue
selection φ_cat**. Deconvolving a wide photo-z against a monotonically rising
volume prior with no selection turnover shifts the host-z estimate up by
~~≈ 2(σ_z/z)² (mode: z → [z + √(z²+8σ²)]/2). At the measured σ_z/z = 0.25–0.49 for
these hosts that predicts **+11% to +36%** h inflation → h_eff 0.81–0.99 → rails
at 0.86.~~ **[2026-07-30, §6.6 — law corrected below]** **Observed: rails at 0.86.**
#51 cannot exhibit this — a δ-kernel has zero width by construction.

~~**Status: a prediction that matches, not a measurement of the code's kernel.**~~
**[2026-07-30, §6.6 — superseded: it is now a measurement of the code's own
numerator.]**

**Refute by:** compute the kernel's actual induced host-z shift numerically for the
76 hosts at their real σ_z, rather than via the mode formula. Also note the local
`z_error` column is stale vs the cluster parent (#40b PV width), so the σ_z/z
inputs are indicative.

**[2026-07-30 adjudication, §6.6] → FINDING (MEASURED); retagged
[LOCAL, VERIFIED — MEASURED]; the law is CORRECTED and the scope NARROWED.**

**Confirmed as the mechanism for C5's catalogue-leg per-event rail**, by direct
measurement of the code's own numerator — driver validated against `fixed_quad`
at **0.0e0**, kernel h-invariance **9.1e-16**, no quadrature aliasing.

**Corrected law.** The claim's formula is wrong for this code: the numerator
window's width is z-proportional, contributing an extra +1/z, so

> **h_eff/h_true = [1 + √(1 + 12(σ_z/z)²)]/2 → 3(σ_z/z)²**

The claim's 8-in-the-sqrt / 2(σ_z/z)² **understates by 1.35–1.5×**. Corrected
sentence: at σ_z/z = 0.25–0.49 the inflation is **+16% to +49%, h_eff
0.85–1.11** (the claim said +11–36%, 0.81–0.99). **Rail threshold:
σ_z/z > 0.256.**

**σ_z → 0 limit gate PASSES** (shift ∝ (σ_z/z)², log-log slope 1.99, coefficient
→ 1.500·2 = 3), so the §7 fix cannot disturb #51.

**Confronted with production, not merely predicted:** observed in-cat
ball-numerator tilt median **+0.308 nats (93.2% positive)** vs predicted
**+0.33..+0.39** at σ_z/z 0.35–0.65 — against **−408 nats, 0% positive** for the
point kernel. The production data *independently* implies σ_z/z ≈ 0.35–0.6, so
the verdict does **not** rest on the stale local `z_error` column.

**Scope narrowed, two ways.**
(i) **Channel-common** — `prior_num` multiplies both numerators identically ⇒ C7
is **not** a C3/C4 candidate, consistent with the exonerated "z leg" entry, which
is properly *not* re-opened.
(ii) **It acts AGAINST the dark rail**: K > 1 always, and dark events sit at
σ_z/z ≈ 0.10 (K = 1.03). The dark catalogue-leg preference for 0.60–0.64 requires
bare impostor z_g/ẑ ≤ 0.83 — **foreground contamination, a separate and
unexamined mechanism. NEW OPEN THREAD** (the inversion is censored data: a
hypothesis, not a measurement).

*Holes, noted:* the with-scatter leg used synthetic z_obs draws (the realized
observed catalogue is cluster-only); a single-host prediction is compared against
a ball-sum observation (medians and rail fractions are the valid comparison); the
"interior at h ≈ 1.12" statement is a reconstruction, not a delivered posterior.
Cell B, which uses the true cluster parent `z_error`, is the staleness-free
magnitude check.

**History collision — must be resolved in any fix derivation.** Added from
[`gate_b_20260730/BIAS_HISTORY_LEDGER.md`](gate_b_20260730/BIAS_HISTORY_LEDGER.md)
(§3 per-claim C7; §1 row 47):
- The **ratified G2b derivation CONFIRMED** `w_pop = (dV_c/dz)/(1+z)` **without**
  `p_det` as "the unique weight consistent with the project's own rate model and
  with every selection integral", **exactly h-independent**, reducing to the
  point kernel as σ_z → 0 — and that h-independence is protected by a **binding
  regression gate** (`docs/derivations/G2b_host_z_volume_prior.md:413-436`;
  gate 6 of `PRODUCTION-KERNEL-FIX-SCOPING:170-180`).
- The **measured historical failure mode of the deconvolution at large σ_z/z was
  OVER-correction** (ledger #62/#68) — the *opposite* sign to the direction C7's
  proposed fix pushes.

  ⇒ **A C7 fix must explicitly supersede G2b** — scope: G2b's premise versus the
  finite-σ_z numerator kernel — and must **not** silently contradict it.

**Candidate ingredient for the new foreground-impostor thread, to check when that
thread is opened.** Added from the ledger (§3 per-claim C5): #53's realization
**clips realized redshifts at `z_floor = 1e-5`** — `n_z_floor_clipped = 108,395`
of `n_rows = 22,641,048` in r1's sidecar
(`seed61000/real_r1/posteriors/realization_provenance.json`;
`observed_realization.py:331-334`). History's **h1_zclamp** finding
**re-attributed a +0.030 bias to a *generative* z-clamp in another venue**
(`results/h1_zclamp_20260713/FINDINGS.md:39-59`; ledger #69), and that
exoneration was granted on the premise that production's catalogue was
*unclamped* — which #53 changes. The clip is therefore a candidate ingredient of
the foreground-impostor thread and **must be checked when that thread is
opened.**

## C8 — The 2D posterior is reparametrization-dependent ~~[AGENT, NOT REPRODUCED]~~ **[LOCAL, VERIFIED]**

Rescaling the mass coordinate by a constant C in the 2D channel walks the MAP
across the grid: C=1 → 0.8133, C=0.3 → 0.7821, C=0.1 → 0.7438, C≤0.01 → rails at
0.600. The 1D channel is exactly invariant. ~~Cause: a 4D numerator against a 3D
selection denominator (`D(h)`, `:1056-1145`, is channel-common and never
mass-marginalised).~~ **[2026-07-30, §6.7 — cause refuted and relocated, below]**

**If true this is a well-posedness failure, independent of any bias:** a published
2D number that moves with an arbitrary unit choice is indefensible.

**Refute by:** re-run the C-scaling on regenerated per-event 2D data. Check whether
the claimed invariance of 1D is exact and whether the 2D dependence is really
arbitrary rather than a fixed physical scale entering.

**[2026-07-30 adjudication, §6.7] → FINDING (well-posedness defect); retagged
[LOCAL, VERIFIED]; the cause is RELOCATED.**

**Reproduced exactly:** C-walk **0.81329 / 0.78107 / 0.74440 / 0.600** (claim:
0.8133 / 0.7821 / 0.7438 / 0.600; the ≤1e-3 differences are the MAP-refinement
convention); **1D bitwise invariant** across the whole sweep; s = −1 established
in closed form in all normalization modes and both mass-kernel families;
sensitivity **d(MAP₂D)/d ln C = +0.031 per e-fold**.

**The stated cause is REFUTED.** It is *not* "4D numerator vs 3D D(h)":
**D, β_G, β_Ḡ and Σ_glob(with_bh) are all mass-dimensionless**, so
mass-marginalising `D` alone cannot restore invariance. The mismatch is
**between the two numerator legs** — the 2D catalogue leg carries exactly one
mass density (`mz_integral`), the completion leg carries none.

**The claim's open question ("arbitrary vs fixed physical scale") is ANSWERED:**
the code silently hard-wires the measure to **dM_z / M_z,det,i — the event's own
measured detector-frame mass** (span 1.33e5–1.63e6 M☉, a factor 12; swapping in a
constant of the same geometric mean already moves the MAP by **0.0056**). A
*consistent* physical unit change M → kM of all inputs is **exactly invariant**.

**Canonical fix identified and priced (indicative, NOT ratified):** give the
completion leg its missing dark-host mass likelihood g_i(z) from the code's own
population prior. **g_frac median 0.135** ⇒ the completion leg is currently
**over-weighted ~7.4×**. Decomposition: h-frozen g(0.73) — the pure measure fix —
moves 2D **0.8133 → 0.7558** (**−0.058**, agreeing with the constant-C sweep at
C ≈ 0.135); the full g(h) adds a **+19.0-nat** population tilt and lands at
**0.84917**, independently reproducing the exonerated **HA endpoint 0.8492 to
3e-5**. HA's exoneration is thereby **upheld and decomposed**: its wrong sign =
(−0.058 measure, right direction) + (+0.093 model-dependent
mass-function/redshift term). **The upward term is what deserves scrutiny before
any `/physics-change`.**

*Hole, noted:* g_i is evaluated at z_i(h), not through `B_num`'s quadrature, and
the ±0.093 split is model-dependent (Babak M1). The −0.058 measure part is robust
(two independent routes agree).

**Runbook §7's HA acceptance gate is corrected in place.** "2D MAP invariant
under M → kM" is **vacuous as written** — a consistent unit change is exactly
invariant — and is restated as **measure-invariance: 2D MAP invariant under
L_cat,2D → L_cat,2D/C for arbitrary C** (equivalently: both numerator legs carry
the same mass-density dimension, so C cancels event-wise). The 1D bitwise
invariance is the regression anchor and already holds. See
[`../RUNBOOK_NEXT_SESSION_6.md`](../RUNBOOK_NEXT_SESSION_6.md) §7.

---

## C9 — `w_G` is mis-calibrated 2.3–2.5× against the code's own generator; the inference's largest measured lever [LOCAL, VERIFIED] — **NEW 2026-07-30**

Model **w_G(0.73) = 0.1215037** versus the realized detected in-catalogue rate
**164/3135 = 0.05231** (76/1590 + 88/1545); binomial **z = −11.86** pooled.

**Localized:** the whole discrepancy is the catalogue's *relative detection
efficiency*. β_G weights f(z) by the **pool-marginal (population-mass)** `p_det`,
but Malmquist-selected catalogue hosts carry heavier M–σ BH masses (rate-weighted
median log₁₀M **6.9**; **≥88%** of the rate weight lies above the 1e7 M☉
population cap by z ≈ 0.3). Two independent suppression measures agree to
**0.2σ**: **r(0.73) = Σw_Dg(with_bh)/Σw_Dg(no_bh) = 0.39248** (the run's own logs)
vs realized **0.3991 ± 0.0312**. Mass-aware **w_G = 0.05149 → z = +0.21**.
Pointwise `p_det`-free check: P(in-cat | det, z) matches f̄(z) below z = 0.15 and
then collapses — **1/62** in [0.15, 0.20], **0/2882** above 0.20, where f̄ is
still 0.11–0.29.

**Counterfactual β_G → r(h)·β_G (diagnostic, NOT a ratified fix):** 2D mean
**0.8123 → 0.7433** (bias +0.077 → +0.013); channel difference **+18.80 →
+11.36** (−7.44 nats, 40%); N·Δln(1−w_G) **+31.55 → +12.10** — and 1D
**0.7321 → 0.6430**.

**Scatter- and realization-independent** (the host list is fixed by the CRB, and
the tension exists identically in the idealized venue) — but **absent from #51's
*delivered* numbers**, because `generator_marginal`'s w_G slot is a different,
mass-aware estimand (**0.0555** at truth). The #51 → #53 switch changed the
delivered mixture weight by **×2.19**, in the direction that inflates the
prefactor tilt.

**Adjudicator's discounts.**
(i) The "removes 84% of the bias" counterfactual must be read against C5's
leverage finding — in a near-flat profile many ±10-nat interventions move the MAP
a lot. What is solid **independent of leverage** is the **z = −11.86
generator-vs-inference inconsistency** and the two-way 0.392 / 0.399 agreement.
(ii) The dark-side self-consistency (ε_dark = ε̂_dark) is argued from
construction, not measured (cheap follow-up listed in the adjudication §5).
(iii) The attacker's own cell-B sentence ("if the normalization mode owns it, B's
2D MAP should come back near 0.73–0.75") is **inverted relative to their own
mechanism**: cell B runs `absolute_marginal`, so the mass-blind w_G = 0.1215 is
**present** in B and the mechanism predicts **B biased HIGH**.

**This resolves the claim file's "loose thread"** (see below): the ghost values
0.0697 / 0.0686 lie on the `generator_marginal` w_G curve
(**0.0774 / 0.0692 / 0.0555 / 0.0427** at h = 0.60 / 0.64 / 0.73 / 0.86); the
"45% discrepancy" compared **incompatible estimands**.

**Historical corroboration** — added from
[`gate_b_20260730/BIAS_HISTORY_LEDGER.md`](gate_b_20260730/BIAS_HISTORY_LEDGER.md)
(Gate-C item 1; ledger rows 45 and 78), which post-dates the adjudication:
**G1 measured a −17.2% end-to-end residual** between the discrete catalogue sum
Σ_glob and the continuous β_G **on the real GLADE catalogue** (after removing the
expected n_gal ∝ h³ factor, which cancels; `docs/gates/G1_beta_g_check.md:14-29`),
and **§3.21 measured the `n̄_w = Σ_glob/β_G` identity violated by 33% in value and
0.39 per h in log-slope** (`docs/H0_BIAS_RESOLUTION.md:1548-1552`). G1 concluded
that "local modes are structurally immune" **because they never use Σ_glob — but
`absolute_marginal` does** (`n̄_w = Σ_glob/β_G`). **This class of inconsistency is
why `generator_marginal` replaced `absolute_marginal` historically.**

**Re-litigation guard** (ledger §2 item 3): the historical exoneration of
"`w_G = β_G/D` bookkeeping / membership-conditioned inverse" was an exoneration of
a **FIX FORM, not of the defect**. The defect — C9 — is **live**. The exonerated
fix *shape* (re-deriving w_G as a membership-conditioned inverse) merely
**relocated** the tilt to the host branch (+94…+455, 12/12 fail, ledger #61) and
**must not be re-tried.**

**[UPDATE 2026-07-30, dark-draw self-consistency measured — adjudication discount
(ii) is CLOSED]** The dark-side self-consistency ε_dark = ε̂_dark, discounted
above as "argued from construction, not measured", **has now been measured**
(`gate_b_20260730/c9_darkdraw_check.py` §6;
[`gate_b_20260730/c9_darkdraw_results.json`](gate_b_20260730/c9_darkdraw_results.json),
`production_pool` block). Measured against the **production injection pool**
— identity **fingerprint-verified**: dl_max(0.73) = **9.164987 Gpc** vs the run
log's **9.165**, relative error **1.4e-06**; **200,807** rows, z_cut **1.5**,
staged at `gate_b_20260730/injection_pool_mix200k_20260728/` — the realized
detected **dark-host z-distribution (2971 events, both seeds, full range, no
truncation needed)** is **significantly skewed HIGH relative to β_Ḡ's own coded
integrand**: **KS D = 0.0863, p = 1.08e-19**; quantile offsets **+0.0163 /
+0.0229 / +0.0369 / +0.0462 / +0.0286** at q10 / q25 / q50 / q75 / q90 — a hump
peaking at q50–q75 and easing in the tail, not a runaway tail.

⇒ **ε_dark = ε̂_dark does NOT hold exactly.** The distortion is **real but
modest**, of the **same character as — and smaller magnitude than — C9's
catalogue-side mass-blind-`w_G` finding**, and it **extends C9's scope to the
completion/dark side**. It belongs in the **same joint mass-consistent-mixture
fix track** (ADJUDICATION §5 item 6), **not** as a separate defect.

*Note on an earlier read:* a local-pool approximation **overstated** the median
offset (**+0.047 → +0.037** with the true pool; KS D 0.1197 → 0.0863). Removing
the pool-depth confound **narrows** the effect; it does not eliminate or reverse
it.

**[UPDATE 2026-07-30, off-r1 replication — w_G(h) verification]** All 10
realistic diagnostics CSVs share **one bit-identical w_G(h) curve**
(w_G(0.73) = **0.1215039**, matching the adjudicated 0.1215037 to floating
rounding), confirming the pure-quadrature claim **off r1**: w_G does not depend
on the realized event catalogue. **See also the open discrepancy on the
*generator_marginal* reference curve** flagged under "New observations" below —
it does not touch this realistic-venue value.
([`gate_b_20260730/c3c4_allruns_summary.md`](gate_b_20260730/c3c4_allruns_summary.md))

**Refute by:** showing the two estimands are in fact compatible — i.e. that the
realized 164/3135 is not the quantity w_G models. Note the *fix* is gated on cell
B; the *defect* is defensible today, independent of the bias question.

## C10 — the completion-channel up-pull is prefactor-carried, not `L_comp`-carried [LOCAL, VERIFIED] — **NEW 2026-07-30**

Over 0.73 → 0.81: **N·Δln(1−w_G) = +31.55** (dark share +30.04, in-cat +1.51)
while **ΣΔln L_comp = −3.11** (**dark −22.72, in-cat +19.61**); only **39.1%** of
dark events have a positive completion tilt — i.e. **`L_comp` pulls DOWN for dark
events.**

⇒ **Any sentence of the form "the completion term pulls up" must name the
`(1−w_G)` prefactor, not `L_comp`.** This is what retires the C4 mechanism's
wording and the corresponding phrase in the one-paragraph summary above.

## C11 — completion-leg deep-venue calibration is an order of magnitude too small to own the 2D bias [LOCAL, harness] — **NEW 2026-07-30**

`pp_coverage` extended to comp_frac **0.008–0.234** (landing #53's w_G ≈ 0.12
venue): bias **+0.0008..+0.0097** at comp_frac 0.06–0.09 and
**+0.0034..+0.0181** at 0.13–0.24; **monotone in comp_frac across the full
0.008–0.85 range, no sign flip, control-consistent at zero.** That is **6–16×
below +0.077** ⇒ **REFUTED as the 2D owner.**

Live as a **modest contributor to the 1D +0.017 Option-A residual** (same order).
Caveat: the harness is **1D-only / single-channel by construction**, so it has
never covered the 2D residual.

---

## What is explicitly NOT claimed

1. **Not claimed: that any of this is a *defect* rather than correct physics.**
   Rejecting impostors is what the 2D channel is *for*. The bias appears because
   the two mixture legs disagree about h for the same dark population — the
   impostor leg rails at 0.64, the completion leg pushes past 0.86, truth is 0.73.
   **Which leg is wrong is undetermined.** That is the open physics question.
   **[2026-07-30 adjudication, §6.12 — appended, the claim is kept]:** however,
   **three measured internal inconsistencies** — C7 (kernel selection omission),
   C8 (missing mass measure) and C9 (w_G calibration, z = −11.86) — **all sit on
   the completion/prefactor/kernel side, and none convicts the catalogue leg.**
   The mass de-weighting of impostors (the C4-amended carrier) is meanwhile the
   2D channel's *intended function*. **The leg-adjudication is now an
   evidence-weighted question, not an open coin** — a direction, not a verdict;
   cell B plus the joint fix derivation must settle it.
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

**[2026-07-30 adjudication, §6.9 — appended]:** **HA — upheld and decomposed**
(−0.058 measure + +0.093 population tilt; endpoint independently reproduced to
3e-5 from a different starting point — see C8) · **"D(h) not mass-marginalised"
as a *formulation*** — refuted (D, β_G, β_Ḡ are mass-dimensionless; the live
object is the **completion numerator's** missing mass density — see C8).
**`w_G` is deliberately NOT added to this list: C9 is live, gated on cell B.**
Two existing entries are strengthened rather than changed: **HB** (its hard zeros
are worth 1.5% of the target, corroborating its self-refutation) and **HA**
(above).

**[2026-07-30 — scoping caveat, from
[`gate_b_20260730/BIAS_HISTORY_LEDGER.md`](gate_b_20260730/BIAS_HISTORY_LEDGER.md)
§2 "Standing scoping rule"]:** negative conclusions are **venue-scoped**. The
**`volume_trunc` and `mass_trunc` exonerations were both measured on the same
seed600 494-event shallow subsample** — same-venue, **not cross-venue
confirmed**; a shared venue idiosyncrasy would fool both. Do not cite either as
universal.

**[2026-07-30] This list is not the whole exoneration set.** The binding union is
this list **plus** the ledger's §2, whose ⚠ entries (items 1–17) are historical
exonerations **absent from here** and are therefore the live re-litigation risk.

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
   **[UPDATE 2026-07-30, off-r1 replication — BOTH values are now explained; the
   "wrong `w_G`" was itself the estimand confusion.]** Checked directly against
   the CSVs: the realistic runs' `w_G(0.60)` = **0.1625175**
   (`seed61000/real_r1/diagnostics/event_likelihoods.csv`, one distinct value
   across the whole h-slice) = the log's 0.1625 — the **`absolute_marginal`**
   value; and **0.0686** is the **`generator_marginal`** CSV value at h = 0.60,
   bit-matching the measured root/sig0/zoom curve **0.0686001**. Neither number
   was corrupt: they are two different estimands. The operative warning survives
   in changed form — an ideal↔real comparison through that column is still void,
   because the columns hold **different estimands**, not because either value is
   wrong.
6. **[2026-07-30, §6.10] The h grid is non-uniform** — 0.01-spaced on
   [0.60, 0.65] ∪ [0.80, 0.86], 0.005-spaced on [0.65, 0.80]. **Any second
   difference taken across the seams is invalid.** (The Gate-A3 check sat on the
   uniform part and is fine.)
7. **[2026-07-30, §6.10] The "loose thread" values were not corrupt.**
   `w_G = 0.0697` and `ideal_61000.csv`'s 0.0686 are the **mass-aware
   `generator_marginal` w_G estimand**, not corrupt values; the "45%
   discrepancy" compared **incompatible estimands**. Resolved into **C9**.

## Loose thread, unexamined — **RESOLVED 2026-07-30 → C9**

~~`w_G(0.73) = 0.0697` (derived from two independent agent numbers that agree) versus
the empirical in-catalogue rate 76/1588 = **0.0479** — a 45% discrepancy in the
quantity whose h-derivative supplies +394 nats/unit-h. Flagged as a diagnostic, not
a finding. Nobody has looked.~~

**[2026-07-30 adjudication, §6.11] RESOLVED → C9.** It was looked at, and the
comparison was between **incompatible estimands**: 0.0697 lies on the
`generator_marginal` w_G curve (0.0774 / 0.0692 / 0.0555 / 0.0427 at
h = 0.60 / 0.64 / 0.73 / 0.86), whereas the run actually delivered
`absolute_marginal`'s **w_G(0.73) = 0.1215037**. Measured against the realized
detected in-catalogue rate **164/3135 = 0.05231**, the real discrepancy is
**2.3–2.5× (binomial z = −11.86)**, not 45% — and it is now claim **C9**, the
largest measured lever in the inference.

## New observations (off-r1 replication) — flagged 2026-07-30, not yet claims

All three come from
[`gate_b_20260730/c3c4_allruns_summary.md`](gate_b_20260730/c3c4_allruns_summary.md)
(script: `attack_c3_c4_allruns.py`). They are **flags for the next session**, not
numbered claims.

**(i) `sig0_control` is structurally different from every realistic run, and the
comparison is estimand-confounded.** Its C3 split is **in-cat +43.47 vs dark
+15.21, total +58.68, dark share 25.9%** — the **only** run of the 11 in which
in-cat exceeds dark. Its CSV columns carry the **`generator_marginal`** estimand
(w_G(0.73) = **0.0496786** vs the realistic runs' **0.1215039**), so it is *not*
a σ→0 limit of the same estimator and **cross-config comparison through it is
confounded**. This independently corroborates C6 / Gate A1. **Cell B remains the
clean same-estimator test.** *Notable hint, deliberately not over-claimed:* a
**dark-class channel difference of ~+15 nats exists even in this unscattered
control.**

**(ii) The root (idealized) diagnostics CSVs of BOTH seeds contain TWO full
concatenated `evaluate` sweeps.** Every (event_idx, h) pair appears exactly twice
(seed61000: 130,216 rows = 2.00× 65,108; seed62000: 126,444 = 2.00× 63,222).
`w_G` is **bit-identical** between the two copies (0/65,108 and 0/63,222 pairs
differ), but `B_num`, `L_comp`, `combined_no_bh` and `combined_with_bh` differ for
**100%** of pairs, and `L_cat_no_bh` / `L_cat_with_bh` for **64.6% / 19.2%**
(seed61000) and **66.9% / 19.6%** (seed62000). The layout is 82 contiguous per-h
blocks, i.e. two full sweeps appended rather than one overwriting the other —
**suspected** (not established) to be **pre-/post-`ec09ed0` code eras.**
⇒ **Do NOT compute on the root CSVs without era disambiguation**; blending them
silently averages two code states. C3/C4 were correctly **skipped** for `root`.

**(iii) The measured `generator_marginal` w_G curve does not match the one quoted
inside C9's ghost-resolution — the exact curve attribution is OPEN.** Measured
from the root/sig0_control/zoom CSVs (internally identical across every run and
both seeds to full precision):
**0.0686001 / 0.0614573 / 0.0496786 / 0.0385580** at h = 0.60 / 0.64 / 0.73 / 0.86,
versus the curve quoted in C9 (**0.0774 / 0.0692 / 0.0555 / 0.0427**) — the
measured values sit **~11% below** at every one of the four points (ratios
**0.8863 / 0.8881 / 0.8951 / 0.9030**, i.e. uniformly 10–12% lower, not the
constant 1.0 a rounding-only difference would give). The adjudication's numbers
may come from a different measurement (e.g. a different event count or the
campaign-#51 full-grid run).
**What survives:** C9's *qualitative* ghost-resolution — the 0.0697 / 0.0686
values are the **`generator_marginal` estimand**, far from `absolute_marginal`'s
**0.1215** — is unaffected on either curve. **What is open:** the exact curve
attribution. The realistic-venue w_G(0.73) = **0.1215039** is independently
confirmed (CSV + the `Partition-norm: w_G=beta_G/D(h)=0.1215` log line for
real_r1) and is **not** in question.
