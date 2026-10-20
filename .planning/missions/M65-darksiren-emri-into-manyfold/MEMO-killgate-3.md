# MEMO — kill criterion (3), the time-bound gate

**For:** the author. **From:** mission M65, 2026-09-21. **Decision needed before:** 2026-09-23.
**Grounded in:** the m1 adjudication (`VERDICT-m1.md`), the independent verification pass
(`VERIFY-R1.md`), and the blocked status of T-1. **No new physics.**

## The criterion, verbatim

> 3. **Time-bound** — if no blind-mock verdict exists by **2026-09-23**, the thread ships as a
>    **methods** paper with the closure reported as *unblinded*, rather than as an H₀ measurement.
>    *(Date borrowed from the workspace copy-off deadline already on the books, not derived from
>    the physics — amendable.)*

## What changed since you ruled

On 2026-09-21 you granted an extension, recorded in `wiki/meta/research-portfolio.md`:

> *"Explicit extension: run T-1 + adjudicate m1 firs[t] — please put this on the overnight list for
> today."*

That ruling conditioned the extension on **two** things. M65 did one of them and found the other
impossible:

| precondition | status |
|---|---|
| adjudicate the m1 docket | **done** — verdict: UNRESOLVED-AS-REGISTERED, returned to you as four yes/no questions |
| run T-1 | **not possible**, and not for the reason the extension assumed |

**So the extension's stated precondition cannot be met as written.** That is the decision in front
of you; it is not the question you were asked on 2026-09-21, and answering it needs a fresh ruling
rather than a restatement of that one.

## Why T-1 could not run — two independent gates

**Gate 1 — no cluster credential.** T-1 needs a fresh GPU EMRI simulation campaign at a newly
sealed `h_inj` on bwUniCluster 3.0, then a CPU eval array (redteam review §4: *"one simulation
campaign + one eval array"*). The brain host has no SSH key and no university VPN.
`M38-cluster-session-unlock`, the mission built to fix exactly this, is `stage: idea` and unbuilt.
The CPU-only "cheap partial" is ruled **"NOT a substitute"** by the redteam review itself: it
*"confirms the §1 mechanism; it does not test the selection model."*

**Gate 2 — your own launch bar, which no credential would lift.** T-1 is the sealed **(m2)** run.
Docket **R27** states: *"no (m2) launch until ratified"*, pending an erratum that adds a both-fire
precedence rule and a treatment for σ_measured < σ_anchor. **Even with cluster access tonight, T-1
could not legitimately have launched.** Ratifying R27 (memo question Q3 in `VERDICT-m1.md`) is an
author action that unblocks T-1's governance side independently of when the credential lands.

This matters for how you read the last three weeks: T-1 has not merely been waiting on
infrastructure. It has also been waiting on a ruling only you can give.

## The thing m1 does not do

**m1 does not discharge criterion (1), and adjudicating it does not create a blind-mock verdict.**
Criterion (3) keys on *"no blind-mock verdict"* — m1 is not blind. Its truth (h = 0.67) is public
and sits in the run's own directory name. The register says so itself, repeatedly:
*"no anti-tuning claim can bank from (m1)"*; *"explicitly NOT the T-1 verdict"*.

So **criterion (3) fires on 2026-09-23 as written**, regardless of tonight's adjudication. Nothing
M65 did prevents that, and nothing M65 did was intended to.

## The finding that may outrank the gate question

The date in criterion (3) was borrowed from the workspace copy-off deadline. That deadline is
real, and it is **not extendable**:

> `docs/campaign_redesign_51_design.md:216` — *"Workspace expires **2026-09-23** (last extension
> used)."*
> `results/prod2d_closure_20260818/CAMPAIGN_REPORT_20260819.md:89` — *"Workspace expiry 2026-09-23
> (0 extensions)."*

And the evidence base is thinner than the record implies. `MANIFEST.md5` for the m1 retrieval lists
**85** entries; **43** are tracked in git. Missing from version control: the 41 posterior JSONs,
`event_likelihoods.csv`, and the scorer `t0_read.py` itself. The 2D primary channel's posteriors
(~3.3 GB) were deliberately left on the cluster as "not a read input".

**Consequence: every decisive number in the m1 docket is currently prose-only testimony, and some
of its substrate may cease to exist on 2026-09-23.** The arithmetic is internally consistent to
~1e-4 — real corroboration, but weak: a systematically wrong scorer produces an equally consistent
set.

**And this was already flagged, five weeks ago.** An independent commission on 2026-08-14
(`results/commission_research_20260814/REPORT.md:123-145`) found the CRB CSV, frozen-α JSON, pruned
catalogue and the injection pool all *"live in git-untracked directories … a clean checkout cannot
reconstruct these runs"*, and recommended *"Archive the pinned inputs with committed checksums to
persistent storage **now**, especially given bwHPC workspace expiry is a known project
constraint."* That recommendation does not appear to have been executed. It is now two days from
the deadline.

Two aggravating facts: the cost to regenerate the 50k injection pool is **unknown by the repo's own
admission** (*"total node-hours … not locally reconstructible (no simulate logs/`sacct`
retained)"*), so a post-deadline T-1 would regenerate it with no budget to plan against; and the
m1 docket could become **permanently unauditable** — per-event data gone *and* scorer
unrecoverable — leaving only prose testimony of numbers currently under an open ruling.

**Honest bound on this claim:** the audit ran on the brain host, which is not your dev box. "Not
found locally" means *unknown on your machine*, not confirmed-lost — `~/data-backups/` could not be
checked from here. The actionable point is narrow: **somebody with cluster access should verify and
copy off before 2026-09-23.** That access is the same thing M38 is blocked on. Full inventory in
`REPORT-R1.md` §7a.

## The options

**A — Let (3) fire: ship as a methods paper, closure reported as unblinded.**
Honest and already pre-authorised by your own criterion. The redteam review has already fenced what
such a paper may not claim without T-1/T-2/T-3 — no *"LISA EMRI dark sirens constrain H₀ to
~0.04 %"*, no *"the completeness correction is validated"*, no width at all until T-5. The
methods and dissolved-threads material is described in STATE.md as *"publishable regardless of the
rail outcome"*. Costs: the H₀-measurement framing is off the table for this paper.

**B — Extend again, pending `M38-cluster-session-unlock`.**
Keeps the measurement framing alive. But be clear about what you would be extending *toward*: an
experiment blocked by an unbuilt mission **and** by your own unratified R27 erratum, whose input
data may expire in two days. An extension without a date and without addressing preservation
repeats the situation that produced this memo. If you choose B, it needs (i) a named date, (ii)
M38 actually dispatched, and (iii) a preservation plan — otherwise the same card returns.

**C — Something else.** The most defensible variant M65 can see, offered as a recommendation
rather than an option you must take: **decouple the two things the date is currently doing.**
Let (3) fire on the science (ship methods, as A) *while* treating the workspace expiry as its own
P0 preservation action. The methods paper does not depend on T-1; the preservation does not depend
on the gate. They are only linked because one date got reused for both. Ratify or reject R27 either
way, since T-1 cannot be launched until you do.

## What M65 recommends, and why it is a recommendation and not a decision

**Ship as methods (A), plus preserve, plus rule on R27.** The reasoning: criterion (3) fires as
written; the alternative is an extension toward an experiment with two live blockers and an
expiring substrate; and your own criterion already pre-declared the methods-paper outcome as the
honest one when no blind-mock verdict exists.

**This is your call, not M65's.** The criterion is explicitly *"amendable"*, and a go/kill is a
kill-class decision — the vault's own rule is that the human confirms kills. M65 has banked
nothing.

## Questions on the card

> **K1 [RULE] — kill-gate (3).** Ship as a methods paper now (closure reported as unblinded), or
> extend again with a named date pending M38?

> **K2 [DO] — preservation.** The workspace expires 2026-09-23 with no extensions and the m1
> posteriors/scorer are not in git. Authorize a copy-off as a P0 action — recognising it needs the
> same cluster access that is currently blocked?

*(R27 and the m1 tie itself are asked separately in `VERDICT-m1.md` — Q1/Q1b/Q2/Q3/Q4.)*
