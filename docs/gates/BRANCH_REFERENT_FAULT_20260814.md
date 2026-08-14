# The branch-referent fault — two registered trees fired a branch whose meaning clause does not describe the data

> **STATUS: PROPOSED, PENDING AUTHOR APPROVAL.** This note is the durable rationale behind the
> **PROPOSED** amendment row **A8** in `docs/RESEARCH_CYCLE.md`. Nothing here is adopted, nothing
> here amends a registered document, and no registered document under `results/` is edited by it.
> If the author declines A8, this note stands only as a post-mortem.

**Date:** 2026-08-14. **Written from:** `results/mechanism_study_20260813/` (the two stage-2
pre-registrations, their two readouts, and amendment A1). Every number below is quoted from those
committed files; none is recomputed here.

---

## 1. The pattern, stated once

A stage-2 pre-registration carries two kinds of clause, written in the same sitting by the same
author:

- a **decision rule** — a threshold, a count, a conjunction. It is arithmetic, and the cycle
  already disciplines it: it is derived pre-data, it is anti-tuning-locked, and a readout applies
  it verbatim.
- a **meaning clause** — "*and that means the mechanism is X*", "*and the `/physics-change`
  package is written against it*". It is the thing the study is actually for, and **nothing in the
  current stage-2 procedure validates it at all.**

The result is a registered tree that can fire correctly and still hand the author a conclusion that
its own data refutes, or that no arm in its own design was ever capable of supporting. That is not
a scoring error and it is not a data problem — in both instances below the readout applied the rule
exactly as written and the adversarial pass confirmed the arithmetic. **The fault is in the
drafting, and it is catchable at registration, for free, before a single core-hour is spent.**

---

## 2. Instance 1 — a one-sided threshold named after a point prediction

**Document:** `results/mechanism_study_20260813/PREREGISTRATION_2D_DOSE_SCAN.md` (§4.3 rule,
§4.4 tree, verdict appended 2026-08-14). **Readout:** `SCAN_READOUT.md` §3.2, §5.

The registered DS-D3 rule discriminates two shapes at scan cell S23:

```
midpoint = (0.017333 + 0.002000)/2 = 0.0096667
SHAPE-INTERACTION  iff  b(S23) >= 0.0096667 + 0.00183462 = 0.01150132
```

`0.017333` is **H-INT's own point prediction at S23**; `0.002000` is H-THRESH's. The rule has a
lower edge and **no upper edge**.

The measurement: **b(S23) = +0.023650**. The rule returns SHAPE-INTERACTION, +28.2 realized SE
above the boundary (+19.9 SE against the registered SE — same call either way). Branch 2,
INTERACTION-BILINEAR, fires.

But the same registered statistics place the measurement **+10.33σ above H-INT's own point
prediction** using the registered SE (+14.64σ using the realized SE). Bilinear residuals are
positive at all nine evaluable cells and exceed 3σ at S22 (+3.76), S31 (+7.64) and S23 (+5.47).
H-THRESH is independently refuted on its own terms at 17.96σ (S13) and 50.18σ (S23).
**Both registered shapes are quantitatively wrong**, and the surface is characterised in the
readout as a *gate × amplifier* (the f_host = 0 row is exactly +0.000000 in 60/60 seeds; the
impostor sea contributes ~15 %) — not a symmetric product.

Branch 2's registered meaning is *"the bias is a genuine product-form interaction"*, strictly
bilinear `D = I·f_h·f_i`. The registered verdict therefore records **branch 2 fired, and its
pre-stated meaning BARRED from being quoted**, together with an explicit *"REGISTERED DEFECT
RECORDED — DS-D3 is a one-sided threshold with no upper edge, so SHAPE-INTERACTION fires for any
sufficiently large value, INCLUDING values that refute the hypothesis it names."* The threshold was
**not** adjusted (§4.7 anti-tuning) — correctly so; the remedy is a drafting rule, not a moved band.

**The general lesson.** A hypothesis that makes a *point* prediction cannot be tested by a
*one-sided* rule. `b >= 0.0115` cannot distinguish "consistent with H-INT" from "ten sigma past
H-INT". The rule as drafted asked *which side of the midpoint is the data on*, and then reported
the answer under a label that asserts *the data is H-INT-shaped*. Those are different questions, and
the second one is the one the branch's meaning clause claims to have answered.

---

## 3. Instance 2 — a branch whose meaning clause has no referent in the design

**Document:** `results/mechanism_study_20260813/PREREGISTRATION_MECHANISM_ISOLATION.md` (§2 arms,
§3 classification, §4 branches). **Readout:** `MECHANISM_ISOLATION_READOUT.md` §5, D-M-3, §9 item 1.

Registered branch 2:

> **SINGLE-OWNER** — exactly one arm is TERM-OWNS. **That term is the identified mechanism; the
> `/physics-change` package is written against it**, with this study's arm as its regression test.

The data satisfies the condition exactly once. The sole TERM-OWNS arm is **MEI (E1-imp)**:
|b| = 0.000000 ≤ 0.010 **and** HPD90 = 1.000 ≥ 0.60, identically in 1D and 2D. MN0 is
TERM-INNOCENT; MEH is OTHER. Count = 1. Branch 2 fires.

But E1 is registered — in the parent's own §2, and again in `ARMS.md` — as a **zero-estimator-change,
generator-side arm**:

> *"**E1 is the decisive arm and it requires ZERO estimator change.** … the estimator is
> byte-identical across N-0, E1-host and E1-imp."* (parent §2)
>
> *"No estimator code. `_channel_terms_at_h`, `log_channel_posteriors_ball_sigma_vector` and
> `_g_ball_capped` are byte-identical across all three arms."* (`ARMS.md`)

What MEI removes relative to N-0 is **the host's redshift uncertainty — an input condition of the
mock universe, not a term of the estimator.** So the branch that fires asserts a term-ablation
conclusion that **no arm in the design could ever have supported**: read literally it identifies
"the mechanism" as *host redshift uncertainty*, which is the dark-siren venue's premise rather than
a defect, and there is no formula to write a `/physics-change` package against. The readout records
the branch-2 meaning clause as **HAS NO REFERENT** (D-M-3) and hands the question up unadjudicated
(§9 item 1).

Note the extra sting: **every arm capable of satisfying branch 2's condition was generator-side by
registered design.** This was not a case of the design containing one eligible arm and the data
picking the wrong one — the branch was unsatisfiable-in-meaning from the moment it was written,
and the parent file states the fact that makes it so, two sections above the branch itself. A
five-minute referent check at registration would have caught it against the document's own §2.

---

## 4. The third fault, same root — an asserted band

**Document:** `results/mechanism_study_20260813/AMENDMENT_A1_VM1_NULL_AT_N100.md` §3.

The parent's V-M1 window (±0.002 on the null arm's reproduction of the campaign bias) is recorded
there as **asserted, with no derivation anywhere in the parent file**. At the arm's registered
N = 15 the difference SE is 0.00159566, so ±0.002 is a **±1.25σ acceptance region** carrying

```
P(|Δ| > 0.002)  =  2·Φ̄(1.2534)  ≈  0.210
```

— a **~21 % probability of declaring STUDY-CONFOUNDED for a perfect reproduction.** Using the
parent's own registration-time SE estimate (0.0013) the figure computable before running anything
was ~13 %. It duly false-failed: the observed |Δ| = 0.002570 is a **1.611σ** fluctuation. The
correct remedy was taken (buy precision — MN0X at N = 100, same unchanged window, now a ~3σ region
with a 0.22 % false-fail rate) and A1-PASS landed at |Δ| = 0.000013, 153.8× inside the window, with
the 85 fresh seeds alone +3.57σ above the fail threshold. **The band was never wrong as a
tolerance; it was under-powered as a test, and that was arithmetic available at registration.**

This is the same root as §2 and §3: the clause was written down and never asked to justify itself
before the data arrived.

---

## 5. The common root, stated for the amendment to cite

> A pre-registration's decision rules and its interpretive clauses are written at the same time and
> **validated differently**: the thresholds get arithmetic, the meaning clauses get none. Nothing in
> the current stage-2 procedure requires an author to check that the arm(s) capable of satisfying a
> branch can actually support the conclusion that branch asserts, or that a threshold is two-sided
> when the hypothesis it names is a point prediction, or that a band's false-fail rate under the
> null was computed at the arm's own N.

Two independently drafted trees in the same thread hit the first two faults; the same thread's
amendment A1 documents the third. `MECHANISM_ISOLATION_READOUT.md` §9 item 5 hands exactly this
question up: *"Whether that is a coincidence of two independently drafted trees or a systematic
feature of how the branches were written is a question about the thread's method, and it is the
author's to answer."* This note's position is that **it is systematic**, because all three faults
share one signature: a clause that was never made to pay an arithmetic or referential cost at
registration time.

---

## 6. What this note does NOT claim

- It does **not** re-adjudicate either branch call. Both verdicts stand exactly as filed; both
  readouts explicitly hand their branch question to the author, and this note does not answer it.
- It does **not** propose moving, widening, or re-deriving any registered threshold. DS-D3's
  0.01150132 and V-M1's ±0.002 are anti-tuning-locked and stay locked; the proposal is about how
  *future* rules are drafted.
- It does **not** propose a repair, a `/physics-change` intake, or a new arm.
- It attributes no error to the readouts. In both instances the readout applied the registered rule
  verbatim, disclosed the mismatch as a first-class finding, and refused to quote the meaning
  clause. **The guard-rails worked at readout; the proposal is to move the catch upstream to
  registration, where it costs nothing.**

---

*Cited by: PROPOSED amendment row A8, `docs/RESEARCH_CYCLE.md` (amendment ledger).*
