# Proposed research-cycle amendments — from the 2026-08-19/20 campaign

**Status:** PROPOSAL. Every item carries a tag; nothing is applied to
`.claude/skills/research-cycle/SKILL.md` until the author rules. Evidence base: ledger rows
#127–#140 (three fronts, five cluster fleets, two blindness failures, five withdrawn claims).

**Framing.** The campaign's two big findings were both *invisible to the method that was
running*, not merely unnoticed. That is the interesting failure mode: the cycle's discipline
(prereg, verifier, bands, ledger) worked exactly as designed and still could not see them,
because the blindness was structural rather than procedural. The amendments below target
that class specifically. They are deliberately few — the cycle is already heavy, and three
of the five cost nothing at run time.

---

## The two structural blindnesses, stated precisely

**B-1 — Differential blindness.** A counterfactual/elimination campaign can only see terms
that DIFFER across its arms. `B_scale` (a ratio of two detection models, worth +0.12 in the
posterior mean — twice the bias under study) survived four campaigns because every arm held
it fixed. No number of additional arms would ever have found it.

**B-2 — Self-consistency blindness.** A harness that GENERATES its events from the same
model it then ANALYSES cannot detect a model-vs-world mismatch: the effect is identically
zero by construction. `pp_coverage`, `calibration_gate` and `venue_transfer` all have this
property, which is why systematics-budget row 16 could record "P–P closes at injected truth"
as evidence that a population-shape choice does not affect calibration.

Both were broken by the same instrument: a fresh-context first-principles re-derivation
(the author's step-back gate). That is not repeatable on demand for every cycle — it is
expensive and needs a genuinely uncontaminated reviewer — so the amendments below aim to
make the blindnesses *visible cheaply* rather than to re-run the gate every time.

---

## [RULE] A8 — Invariance & blindness declaration (stage 2, pre-registration)

Every pre-registration gains a short mandatory section with two lists:

1. **Invariants** — everything held FIXED across all arms of this campaign (conventions,
   normalization factors, tables, priors, selection objects, seeds/realizations). One line
   each, with the date each was last derivation-audited, or `NEVER`.
2. **Structural blindness** — one sentence naming the class of defect this design cannot
   detect by construction (e.g. "self-consistent generator: cannot see population
   misspecification"; "single realization: cannot see draw-level scatter").

Binding consequence: a campaign whose invariant list contains a `NEVER`-audited,
load-bearing entry either audits one such entry in the same cycle, or states in its VERDICT
that its conclusions are conditional on the unaudited invariants — by name. Cost: minutes.
Had this existed, `B_scale` would have appeared on the invariants list of every 2D
counterfactual since 2026-08-04 with `NEVER` beside it.

## [RULE] A9 — Provenance freshness for every number entering a budget (stage 5/6)

Any quantity quoted into a budget, band, or verdict carries a four-field stamp:
`{value, source (commit/artifact), date, configuration-of-record}`. A stamp whose
configuration no longer matches the current one is **STALE** and may not be quoted as a
point value: it is either re-measured, or carried as an explicit band with the staleness
disclosed.

Evidence: the Eddington leg `s_Edd = −0.020` sat in a code comment, was frozen into a
pre-registration and a ratified decision, and was wrong by an order of magnitude *and* sign
(re-measured: `+0.0012`) — six weeks after the repo's own gate record had flagged it. The
repo knew; the cycle had no step that made it look.

**Attached routine (harness-level, cheap):** every multi-GB input not in version control
carries a checksum pin at each consumer with a STOP gate. A stale local galaxy catalogue
(July 1 vs the July 27 catalogue of record) silently fed every local analysis until a
fidelity gate caught it; there was no pin because nothing required one.

## [DO] A10 — The score-zero test as a standing first diagnostic

Add to stage 4 (measure/refute) as the FIRST thing tried on any estimator-bias question,
before instruments are built:

> For data drawn from the estimator's own model, `E[∂_θ ln L]` at truth is zero. Compute the
> per-event score at truth over the relevant event class; a non-zero mean at high
> significance localizes a misnormalization, and the same read resolved by class and by
> covariate localizes it *further* — all on banked data, at zero compute.

This is what finally cracked the campaign: `−0.635 ± 0.017` (37 σ) on the dark class, then
class-resolved (the pure-completion class carries ~195% of the slope) and z-resolved (≈0
below z ≈ 0.4, −1.08 by z ≈ 0.9). It cost nothing and could have been run on day one.

## [RULE] A11 — Engagement gate on every counterfactual instrument

No null from an instrument is interpretable until the instrument is shown to change the
output: a registered engagement threshold (e.g. "≥ 10% of the relevant events move by
≥ 1e-6 relative", or a table-level ratio ≠ 1), scored and STOP-gated, plus a check that the
switch reaches every dispatch path the production code actually uses.

Evidence: production dispatches through a batch kernel; an instrument patched only into the
scalar kernel would have passed bit-identity and continuity gates and returned a confident
"no effect". A verifier caught it pre-execution; the gate makes the catch structural. The
same amendment covers the labelled-arm risk (an arm labelled `fused` silently running `off`
would fake a clean verdict) — the switch's runtime value is asserted per arm.

## [RULE] A12 — An attribution ships with its own falsifier

Any memo or verdict that ATTRIBUTES an effect to a cause must, in the same document and
before the attribution is banked, register the experiment that would falsify it, with bands.
If that experiment is not yet run, the attribution is explicitly provisional.

Evidence, in both directions: the population-misspecification memo registered its falsifier
in §7, the falsifier ran, and the attribution was downgraded cleanly with no argument about
what the memo had "really meant". By contrast, the earlier impostor-overlap reading was
written without one and had to be undercut by a later measurement instead.

---

## Explicitly NOT proposed

- **More arms / bigger fleets.** Both blindnesses are immune to fan-out; five more cells
  would have found neither. The cycle does not need more compute discipline.
- **A standing "step-back gate" every cycle.** Its value came from independence and
  freshness, which degrade if it becomes routine. Better trigger: run it when a front has
  produced ≥ 3 consecutive eliminations without an owner (this campaign hit that mark
  exactly), or before any paper-bound claim.
- **Softening the prereg-first rule for free reads.** The free reads were the highest-yield
  measurements of the campaign *because* they were registered first; the discipline is what
  makes a zero-cost read quotable.

## Decision table

| # | Tag | Item | Cost |
|---|---|---|---|
| A8 | [RULE] | Invariance & blindness declaration in every prereg | minutes/cycle |
| A9 | [RULE] | Provenance stamps on budget inputs + checksum pins on unversioned data | minutes/cycle; one-off pin work |
| A10 | [DO] | Score-zero test as the standing first diagnostic | zero compute |
| A11 | [RULE] | Engagement gate + dispatch-path check on every instrument | already de facto; codifies it |
| A12 | [RULE] | Attributions ship with their own falsifier | zero |

If ratified, I apply A8–A12 to `SKILL.md` as numbered stage amendments and add the two
harness rules (checksum pins; the subagent blocking-wait rule from the vault debrief) to
`CLAUDE.md` in the same commit.
