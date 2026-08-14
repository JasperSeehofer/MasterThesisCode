# Handoff — 2026-08-14 — for an INDEPENDENT session, before the thread continues

**Read this before `RUNBOOK_NEXT_SESSION_10.md`.** The runbook tells you the state. This file tells
you which parts of that state are *measured* and which are *one orchestrator's framing* — and asks
you to attack the second kind before anyone acts on it.

**Author's instruction (2026-08-14):** an independent session should assess the completed research
cycle and decide whether it agrees with the outgoing session's recommendations, before those
recommendations are implemented. **Nothing below has been ruled on.** Branch 2 is unruled, amendment
A8 is drafted-not-adopted, and no repair has been proposed. That is deliberate: all three are
downstream of the framing this review exists to test, and recording a ruling first would put a
`[RULE]` in the ledger ahead of the audit that might reframe it.

**Recommended mechanism:** `/commission --research`. It is the purpose-built falsification-first
check over a research thread's claims, it is memory-bearing, and it diffs claim-history for
regressions. Scope it as below rather than running it wide.

---

## 1. What is GROUNDED — do not re-verify, it is waste

Adversarially reimplemented with independent code (own grid-argmax, trapezoid normalisation, PIT,
HPD, KS, SE) and reproduced from the rawest per-seed `ln_post` vectors:

| result | value | status |
|---|---|---|
| all scored statistics, 425 seeds, both channels | max deviation **exactly 0.0** | CONFIRMED |
| A1 / V-M1 at N=100 | +0.037250 ± 0.000494, \|Δ\| = 0.000013 vs a ±0.002 window | CONFIRMED |
| A1-DET | 15/15 shared seeds, 44 fields, bit-identical, cross-commit | CONFIRMED |
| V-M5 registered golden | max rel dev 1.6135e-14 vs rtol 1e-12, MAPs exact | PASS |
| f_host = 0 row | **exactly** +0.000000 at every impostor dose, 60/60 seeds, degenerate posterior | measured |
| non-additivity | 18.40σ vs MN0, 45.67σ vs MN0X; split recovers 11.5 % | measured |
| H-INT refuted | b(S23) +10.33σ above its own point prediction (registered SE) | measured |
| H-THRESH refuted | 17.96σ (S13), 50.18σ (S23) | measured |
| seed plan | 325 unique, zero collisions across 120 pairs | verified |

These survive any review. Spend no budget on them.

## 2. What is CLAIM — attack these

Each is the outgoing orchestrator's framing, not a measurement. The attack surface is named.

### C-A. "The design could never have identified an estimator term."
Every arm — the three split-dose arms and all 16 scan cells — varies a **generator-side** dose;
`ARMS.md` states the estimator is byte-identical across N-0/E1-host/E1-imp. Claim: the parent's title
question was unanswerable by construction, and its branch 2 was unsatisfiable-in-meaning from
registration.
**Attack:** is that actually true of all 16 scan cells, or only of E1? Does `dose_scales` touch
anything the estimator reads beyond the σ vector? Could a term conclusion be licensed indirectly —
e.g. by the exact-zero host row constraining the integrand's form? If the claim is too strong, the
whole "must re-run as term-ablation" recommendation weakens.

### C-B. "Abort (d) should not fire." — HIGHEST STAKES
Registered wording needs a **sign** disagreement between L0 toy and L1 instrument. Toy predicted
+0.0247 where the instrument measured exactly 0.000000: a 100 % magnitude failure, but zero has no
sign, so the literal condition does not fire. The outgoing argument is that the *purpose* is
satisfied anyway because M4 was closed analytically plus by α-deletion on stored posteriors, M1 on
sign via derivation plus committed `pp_coverage` evidence, and M3 by the analytic 4σ-pinning ceiling
— so only M5 leaned on the toy, and the instrument superseded it.
**Attack:** audit each closure's *actual* dependence on the toy rather than accepting that summary —
the person who wrote the closures is not well placed to certify them toy-independent. If any of
M1/M3/M4 materially depends on toy output, abort (d) arguably should fire, **the study STOPs, and
every L0 closure reopens.** This single call carries the night.

### C-C. "M2′ is the register's only surviving candidate."
Rests on the six-candidate register (M1, M2, M2′, M3, M4, M5) being complete and well-posed. Every
adversarial pass this session targeted *scoring*, never the candidate space itself.
**Attack:** is the register complete? Name a seventh candidate consistent with all 16 constraints in
`PHYSICS_CHANGE_INTAKE_DOSSIER.md`. In particular: must the mechanism live in the estimator at all,
given the host-gate result? Is "one exact host annihilates the bias against 1.19 M smeared impostors"
compatible with an estimator-term account, or does it point somewhere the register never looked?

### C-D. "A8 is the right remedy, with branch-referent and two-sidedness BLOCKING."
Drafted at `docs/RESEARCH_CYCLE.md` + `docs/gates/BRANCH_REFERENT_FAULT_20260814.md` (`cd9c610e`).
**Attack:** does a blocking registration check impose real cost on legitimate designs? Is the
diagnosis right that all three faults share one root, or is that pattern-matching across three
loosely related drafting slips? A8 changes the machinery every future study runs through — it
warrants more scrutiny than a one-off finding, not less.

### C-E. "Branch 2 should be recorded as fired, meaning barred."
Mirrors the treatment already applied to the scan's branch 2.
**Attack:** is consistency-with-the-scan the right principle, or does the parent's case differ in
kind? The scan's DS-D3 was a badly-shaped rule applied to a real shape measurement; the parent's
branch names a conclusion no arm could reach. Are those the same fault deserving the same record?

## 3. Calibration for the reviewer

The outgoing session had **three framing errors caught by its own agents and verifiers**, all now on
the record: a 2D bias compared against the *1D* reference and reported as "dead on"; the σ_z = 0
constraint over-claimed as a hard discriminator when the null cell's curvature makes it weak; and M2
eliminated prematurely, requiring reinstatement as M2′. Two further errors were caught in briefs
(CPU-h attributed to the wrong scope; a verdict claimed on a document that has none). The
*arithmetic* held up throughout; the *framing* repeatedly did not. Weight your scepticism accordingly.

## 4. If the review AGREES

Critical path, in order — nothing is blocked and nothing is perishable:
1. Rule branch 2 (C-E) and abort (d) (C-B).
2. Adopt A8 (C-D) — it must precede the next registration, since it changes how one is written.
3. Register and run the **M2′ term-ablation** study: restore the measure inside the z-integral with
   the generator held fixed, plus a **null-ablation control** (perturb a term that provably cannot
   matter) to prove the machinery can detect a term difference at all. That control is exactly what
   the completed study lacked. ~15 CPU-h/cell at current economics (~0.97 CPU-h/seed).
4. Only then does the `/physics-change` gate have a candidate. The intake dossier already fills the
   OLD-formula slot exactly and specifies C1–C16; the new-formula slot is empty **by choice**.

Pre-decide, while it is cheap: if M2′ also comes back innocent the register is exhausted, and the
honest next step is a fresh stage-0 intake with a Stage-L literature sweep — not another lap.

## 5. If the review DISAGREES

Say so plainly and route it: a C-B reversal STOPs the study and reopens every L0 closure; a C-A or
C-C reversal changes what the next measurement should be; a C-D reversal leaves A8 unadopted, which
is fine — it is drafted, not load-bearing.

## 6. Provenance

Full artifact→commit map: `RUNBOOK_NEXT_SESSION_10.md` §5 (22 rows). Read first for the science:
`results/mechanism_study_20260813/CAMPAIGN_REPORT_20260814.md`. Ledger: rows #99–#101 plus two
addenda to #100, in `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`.

**Standing constraints that bind the reviewer too:** registered documents are append-only; no band
may be adjusted after a readout (anti-tuning); branch calls are presented to the author, never
self-adjudicated; no repair may be proposed from the current branch.
