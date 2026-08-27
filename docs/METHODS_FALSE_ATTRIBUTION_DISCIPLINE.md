# Methods: discipline against false attribution in a bias hunt

**DRAFT for author revision — `[OPUS-ORCH 2026-08-27]`.** Assembled by an orchestration session
from the campaign record; no author has ruled on the framing, and nothing here is a scientific
result. It is raw material for a thesis methods chapter. Every factual claim carries a
`file:line`, a ledger row, or a commit; where a claim could not be sourced it is marked NOT FOUND
rather than smoothed over.

**Citation shorthand.** Unless another directory is given, campaign artifacts cited by bare
filename live under `results/campaign51_20260728/realistic_20260729/`. The ledger is
`.../gate_b_20260730/BIAS_HISTORY_LEDGER.md`, cited by row number. Repo-root paths
(`docs/…`, `darksiren_emri/…`, `cluster/…`, `.claude/…`) are given in full.

---

## 1. The problem: every proposed mechanism is plausible by construction

This project spent a long campaign chasing a bias in a dark-siren H₀ estimator. The
methodological content of that campaign is not the list of mechanisms tried. It is the set of
habits that stopped a plausible-sounding mechanism from being written down as the explanation.

The structural difficulty is easy to state and hard to feel. In a bias hunt you begin from a
symptom — a number that is not where theory says it should be — and you generate candidate
mechanisms *conditioned on that symptom*. Every candidate that survives thirty seconds of thought
is therefore, by construction, one that would produce something like the symptom if it were true.
Plausibility carries no information at that point: it is the selection criterion, not evidence.
The prior over "this mechanism explains the symptom" is not the prior you feel, because you never
sampled from the space of mechanisms — you sampled from the space of mechanisms that already
look like the answer.

Three consequences follow, and the rest of this chapter is about each in turn.

First, the *cheapest* failure available in a bias hunt is not a wrong measurement. It is a correct
measurement attached to the wrong object. A number can be reproduced to fifteen digits, survive
every numerical check, and still be attributed to a mechanism that contributes a third of a
percent of it. Section 3 gives the worked case: a headline exhibit that opened an entire thread
was mis-attributed by a factor of about 300.

Second, the record itself becomes an adversary as it grows. Each closed thread makes the next
lead cheaper to propose — there is more vocabulary, more machinery, more banked infrastructure —
while making it more expensive to screen, because the set of already-refuted mechanisms that a
new proposal might silently duplicate grows monotonically. Section 6 is about the specific way
that screening fails.

Third — and this is the part a methods chapter usually omits — the discipline itself has a
measured failure rate. At one of its own gates the record shows two failures in eight attempts,
with three further threads that never ran the gate at all (`EXONERATION_REGISTER_20260827.md:871-873`).
Section 8 states that plainly. A methods chapter that claimed an unbroken record would be
performing exactly the self-deception it describes.

### 1.1 Grounding in cross-project practice

None of what follows is original to this campaign. The habits described here are the local
instantiation of patterns promoted across several projects into a personal research wiki (the
"garden"), and the chapter should cite them as prior art rather than presenting them as
discoveries.

*Provenance caveat, applied to this subsection's own sources:* the wiki quotations below were
returned by a read-only librarian consult of the vault and have **not** been re-read at source by
the author of this draft. Per the project's own standing rule they are evidence, not authority —
verify each quote against its vault path before it enters the thesis.

The anchor page is [[scientific-computing-validation]]
(`wiki/concepts/scientific-computing-validation.md`), referenced from the repository's own
physics-validation rule (`.claude/rules/physics-validation.md`, cited from `CLAUDE.md`
§"Math/Physics Validation Workflow"). Its most directly relevant pattern is **channel
localization**, which is the generalised form of the two structural kills in §2.1:

> "A bias that appears in a channel which structurally *omits* the suspected variable cannot be
> caused by that variable — localize the residual to a channel before fixing the variable.
> Corollary: confirming a defect is *real* does not establish that fixing it *improves the
> metric* — the two questions are independent, and the fix can be metric-adverse."

> "before fixing a suspected-cause variable to chase a metric residual, confirm the residual
> *appears in a configuration that uses that variable* and *vanishes in one that doesn't*. If it
> persists in the variable-free channel, the variable is exonerated regardless of how plausible
> the mechanism."

The same page names **circular validation** — defining a free constant's correctness by
"doesn't disagree with truth" — and offers a code-review smell test for it (search a diff for
"conservative", "doesn't overshoot truth", "empirical anchor", "chosen to match production").

The companion page [[scientific-self-verification]]
(`wiki/concepts/scientific-self-verification.md`) supplies the framing sentence for §1 and the
argument for §7:

> "**Internal self-consistency is not evidence of correctness.** A pipeline where simulation and
> evaluation share the *same wrong convention* passes every internal test and stays physically
> wrong until an *external* anchor forces the discrepancy into view."

and, quoting Feynman (1974), "The first principle is that you must not fool yourself — and you are
the easiest person to fool."

The staged protocol itself is promoted as [[research-cycle-core-spec]]
(`wiki/analyses/research-cycle-core-spec.md`), whose rules R1 (two-layer exoneration check before
opening anything, "exonerations are scope-bound, never universal"), R3 (Gates A→B→C, "refute
before you build"), the `Refute by:` intake requirement, and append-only pre-registration
("anti-tuning is structural, not a promise") are the cross-project forms of the local rules cited
throughout this chapter. Its amendment ledger [[research-cycle-amendments]]
(`wiki/meta/research-cycle-amendments.md`) adds a limit worth carrying into a thesis verdict
table:

> "**Missing-anchor cap**: when no external anchor exists for a novel result, that is a registered
> stage-1 limitation and caps the stage-5 verdict at `supported`, never `verified` — a check
> sharing the derivation's assumptions cancels out of the check; anchor-free 'verified' would be
> self-attestation."

The verifier rule of §4 is promoted in `wiki/meta/cross-project-memory/general.md` — "a review's
own verdict is itself an unverified claim until the orchestrator reproduces its decisive number
from the primary artifact" — as is the blind-search technique of §7, recorded there as a
"blind-derivation firewall" whose value is that "denying the deriver the hypothesis under test is
itself informative when the blind result independently reproduces the banked number." The same
page carries the park-artifact pattern this campaign is currently sitting in: an
"independence-clean STUCK symptom card … rather than a narrative that resolves the tension by
assertion."

Finally, the agent-side taxonomy [[agent-weaknesses]] (`wiki/meta/agent-weaknesses.md`) names the
relevant failure mode as **W-CONF, confidence inversion** — "Most assertive when most wrong.
Hedges correct answers, asserts incorrect ones" — with a companion **W-SYN, synthesis blind
spot**, "characteristically invisible to the agent's own self-review". These matter for an
AI-assisted thesis specifically: they are the reason the discipline below is built out of
structural gates rather than out of asking the assistant to be careful.

Within the repository the same discipline is codified in the seven-stage research cycle and its
twelve hard rules (`.claude/skills/research-cycle/SKILL.md:82-140`, expanded in
`docs/RESEARCH_CYCLE.md`), and in the standing rule that subagent output is evidence rather than
authority (project memory `agent-verifier-output-is-evidence-not-authority`).

---

## 2. Measure before you register; register before you run

The ordering is the whole method, and it exists because the three stages differ in cost by three
or four orders of magnitude.

A **measurement** in the cheap sense means reading what is already on disk, or running a
closed-form calculation, or reading the source code that implements the object under discussion.
The research cycle makes this a hard rule twice over: rule 9 requires exhausting free re-reads of
existing diagnostics artifacts before requesting any compute, and rule 6 requires that a cheap
mechanical measurement which could collapse the need for an expensive test runs first
(`.claude/skills/research-cycle/SKILL.md:110-124`). A **pre-registration** costs a session of
careful authorship plus an adversarial review pass. A **run** costs hundreds of CPU-hours on a
shared cluster and, worse, produces numbers that will be interpreted whether or not they mean
anything.

### 2.1 A lead killed at stage 0 for minutes of CPU

The clearest instance is the `[WGEO]` thread (`CLAIM_WGEO_20260827.md`). Its hypothesis was
attractive and physically literate: the pipeline's mass-eligibility window is *linear*-symmetric,
`W = [M(1−kσ), M(1+kσ)]` with `k = 1.5`, while the catalogue's mass error is log-normal. A
linear-symmetric cut on a log-normal variable is asymmetric in the true variable, the asymmetry
grows with the fractional error, and if the fractional error varies with redshift then the window
imposes a redshift-structured selection — a candidate mechanism for the campaign's standing
signature, a tilt localised to the dark class at high z (per-event score at truth −0.635 ± 0.017,
37σ, ledger `:1345-1346`).

The card's discipline was to make the claim numeric before touching it. The falsifier was
registered *before* the reads were adjudicated: for the hypothesis to survive, the window
asymmetry must grow across the four banked dark-class z-bins by roughly the factor 2.3 that the
score itself grows (`CLAIM_WGEO_20260827.md:49-59`, bins from
`docs/derivations/population_mismatch_dark_score.md:41-46`). That is a statistic computable from
one pass over a 1.68 GB catalogue.

It came back flat. Across the four tilted bins the median asymmetry spans 0.0003 on a median of
0.399 — 0.08 % — while the score it would have to explain grows by 2.3× (`:158-160`). The
marginal trend also runs the wrong way: `spearman(z, CV) = −0.6521`, the asymmetry *shrinking*
with redshift (`:137`), and that entire marginal trend is a low-z phenomenon that has decayed to
−0.17 by z ≥ 0.4 (`:162-166`).

Two of the four grounds required no statistic at all, and this is the part worth teaching. The
mass window is applied *after* the candidate set is built: `handler.py:646` constructs
`candidate_hosts_without_bh_mass` with the redshift filter only, and `mass_filter_mask`
(`:663-673`) is applied afterwards at `:674`. The dark class is defined as having *zero*
candidates before the mass window runs. A filter that only subsets an already-nonempty set cannot
be the mechanism for a statistic computed on events where that set is empty
(`CLAIM_WGEO_20260827.md:170-179`). Independently, ledger row #137 records that the tilt is
numerically identical in the 1-D leg (1-D mean 0.6001, 2-D C-C 0.6004; ledger `:1338-1346`), and
the 1-D leg never sees the mass window at all. A mechanism absent from one leg cannot produce a
tilt identical in both.

Two structural kills, one decisive statistic, one sign check. The whole thread closed at stage 0
on local CPU with no cluster job. The card additionally records the *costs avoided* — a 48-arm
fleet re-run which, by its own §3.8, contains zero events above z = 0.34 and therefore could not
have tested the regime the claim was about (`:476`, `:495`).

The generalisable rule: **before a mechanism can be tested it must be shown capable, in
principle, of touching the objects the symptom is measured on.** That check is free and it fires
often.

### 2.2 Null results are banked with the same care as positives

`[WGEO]` is written as a closed null and banked in full (`CLAIM_WGEO_20260827.md:11-15`),
including a section listing figures refuted during the investigation "recorded so they cannot
re-enter" (`:246-260`). Two of those refuted figures came from the card's own reads: both had
framed the linear window as *narrower* than a log window, which turns out to be operationally
backwards, since the linear lower edge is non-positive for 99.61 % of the catalogue and the
"too heavy" leg of the mask is therefore vacuous (`:203-224`). The refuted numbers are recorded
next to the correct ones, in the same document, with the same precision.

This is not tidiness. A campaign whose record contains only survivors will re-propose its own
dead leads, and will do so with the confidence of novelty.

---

## 3. Attribution is a separate measurement from magnitude

The `[P3-MKER]` thread supplies the sharpest single lesson in the campaign. It opened on an
exhibit: for one event (seed 900121, event 20) the with-BH-mass catalogue likelihood sat at
1.39 × 10⁻⁸⁵ against a no-mass value of 6.84 × 10⁻⁹, a difference of −176.6 nats, with the
window-passed candidates described as "~19σ" discrepant under the mass kernel
(`CLAIM_P3_MKER_20260826.md:29-33`). The thread's claim was that the mass kernel's uncertainty
budget was incomplete.

The decomposition was performed from banked artifacts alone, at zero compute, and it split the
−176.6 nats exactly (`:136-169`):

| component | value | share |
|---|---|---|
| mass **kernel** log-weight | −0.5838 nats | 0.33 % |
| mass **window** log-weight | −176.7828 nats | 99.67 % |

Deleting the mass kernel entirely would move the exhibit from −176.56 to about −176.0 nats
(`:168-169`). The exhibit that named the thread was a *window-exclusion* case, not a kernel case,
and the "~19σ" figure was not a mass-kernel quantity at all — it had been obtained as
`√(−2 ln num_w)` from a dimensionful density (`:303-304`).

Two features of how this was handled are worth transferring. First, two independent routes to the
attribution were computed — a residual route (total minus kernel) by the verifier and a
direct-numerator route by the chair — and they were *reported as disagreeing by 0.81 nats*, with
that remainder explicitly left unattributed to either object rather than absorbed into whichever
one the thread preferred (`:162-167`). Second, the amendment did not silently edit the original
claim. Section 1 was left standing as written with a binding amendment block placed over it
(`:262-302`), so that the record shows what was believed at intake as well as what survived.

What survived was much narrower. All three of the claim's factual assertions about the code were
refuted at source (`:264-286`): the catalogue mass *is* convolved
(`bayesian_statistics.py:6607`, `:6613`); the width is dominated by the catalogue term, not the
GW-conditional term, by a factor 2 × 10⁸ with the sign of the asserted domination inverted; and
the intrinsic scatter is *present* as `sigma_int = 0.24·ln 10` (`handler.py:41-44`, added in
commit `555f018`). The last of these is instructive on its own — the card's premise conflated the
two halves of its own cited number, since R&V15's 0.55 dex is a total rms of a 0.50 dex
measurement error and a 0.24 dex intrinsic scatter, and only the former is excluded
(`:276-281`). The surviving question is a real and arguable one about whether a *predictive*
error should carry the calibration sample's measurement error, and it is explicitly labelled
"not a bug" (`:288-293`).

The thread also killed a proposed follow-up on attribution grounds alone: since the exhibit's
kernel share is at most 0.33 %, running a fleet counterfactual *in order to make the exhibit
dissolve* is a guaranteed null and should not be funded on that rationale (`:477`).

---

## 4. Adversarial verification as a separate role

The campaign runs adversarial verification as a distinct agent with a distinct brief, and governs
it by two rules that pull in opposite directions and must both be held.

**Rule one: where a read and the verifier conflict, the verifier governs, and the read's number is
recorded as refuted rather than reported.** This clause is written into the claim cards themselves
(`CLAIM_P3_MKER_20260826.md:122-124`; `CLAIM_WGEO_20260827.md:63-65`).

**Rule two: verifier output is evidence, not authority.** Every decisive number is re-derived
independently before it reaches a ledger row. The cards implement this as a per-number tag —
`✓CHAIR` for independently re-derived on the chair's own code path, `✓VER` for verifier-confirmed
but not chair-re-derived — applied number by number, so a reader can see exactly which figures
rest on one derivation and which on two (`CLAIM_WGEO_20260827.md:65-69`).

The second rule exists because of a documented incident. On 2026-08-20 a top-tier adversarial
pre-check returned ten required amendments, of which several were decisively correct — including
the observation that a `max(0.005, 2·SE)` band was mis-specified because a re-score is a paired
deterministic recomputation on frozen data, so `Var(Δ) = 0` and no sampling band applies — while
in the same report asserting a mean of ≈0.61 for a quantity that is 0.7626, making its own
estimate wrong in sign. A ten-agent refutation panel correctly overturned two of the
orchestrator's claims and, in the same session, *executed the registered primary measurement it
had only been dispatched to refute*, destroying the blindness of the registration. The standing
conclusion is quoted verbatim from the memory record: "The pre-checks have never been wholesale
right or wholesale wrong; they are consistently both"
(project memory `agent-verifier-output-is-evidence-not-authority`).

The operational consequences drawn there are two: re-derive every decisive number a subagent
reports before it reaches a ledger row, and write subagent briefs that explicitly *forbid*
executing the registered measurement rather than merely omitting permission for it.

### 4.1 A gate catching a fix that had already been granted

The most uncomfortable case is the one where adversarial review fired against a repair the record
had already authorised. Runbook 34 named a "one-line" fix as the first action on resuming the
`[P3-2D]` thread: drop the `S̄_φ` factor from the 2-D redshift draw. The `/physics-change`
presentation package put that fix through the gate before any code was written
(`PHYSICS_CHANGE_SBARPHI_20260827.md:1-10`).

The gate's headline is that a limiting check fails on the fix as granted (`:20-45`). The host draw
and the redshift conditional are not independent: hosts are drawn with weight `∝ w_g · S̃_φ,g`
(`correspondence_1d.py:1380`, consumed at `:1682`), and that factor is *exactly* the normalising
constant of the `S̄_φ`-weighted redshift conditional, so today it cancels. Remove `S̄_φ` from the
density and the cancellation is destroyed — `S̃_φ,g` survives as an uncancelled host-level
survival weight. The defect is *relocated* from the event's drawn redshift to the host's listed
redshift, not removed. And because `S̃_φ,g ≈ S̄_φ(z_g)` to `O(σ_z,eff²)` with `σ_z,eff ≈ 0.035`
against a pool range of order 0–0.5, the relocated tilt is highly correlated with the one it
replaces.

The package flagged the residual as UNMEASURED and cheap to measure. The adversarial pass then
measured it: the granted fix leaves **about 69–70 % of the measured 13.5 %/16.0 % drift in
place**, and the verdict is **FIX-MISSPECIFIED** (`:729-735`). The derivation held; the package
did not go to the author as written, carrying seven defects of which three were regression tests
that assert things false on correct code (`:737-739`).

Note what the gate protected against. Had the one-line fix been implemented as granted, it would
have been implemented, tested, committed, and — most damagingly — the residual would have been
re-measured afterwards and found still present, at which point the surviving 69 % would have been
attributed to something else. The gate did not prevent a wrong number. It prevented a wrong
*attribution* of a number that had not yet been produced.

---

## 5. Pre-registration with numeric falsifiable bands

Research-cycle rule 4 requires pre-registration before running, with numeric falsifiable bands, a
first-class `Mixed` branch, secondary reads including expected *nulls*, provenance-gating ("what
upstream gate made this test necessary"), and append-only discipline — no edits above the line
once the file is committed (`.claude/skills/research-cycle/SKILL.md:99-104`). Rule 11 adds two
declarations to every pre-registration: an explicit list of invariants held fixed across every
arm, each with a last-audited date or `NEVER`, and a one-sentence statement of *structural
blindness* — the defect class this design cannot detect by construction (`:126-137`). The
evidence cited for rule 11 is a parameter, `B_scale`, that carried a +0.12 posterior-mean effect —
twice the bias under study — and survived four campaigns because every arm held it fixed. No
amount of fan-out would have found it.

### 5.1 What review catches before the CPU is spent

The `[HIER]` pre-registration (`PREREGISTRATION_HIER_HTHETA_20260826.md`) came back
**LAUNCH-BLOCKED** with six blockers before any `sbatch` was issued (`:1343-1349`). Its Stage P
alone was costed at 424.4 CPU-hours, flagged in the document as roughly 5.8× the largest fresh
costing line yet granted in the campaign (`:498`). Three of the blockers were venue or instrument
identity failures that would have made every banked number uninterpretable; two were registered
objects — a θ prior and an identifiability statistic — that three verdict families depended on and
that were nowhere defined.

The fourth blocker is the one a physics reader should look at. The registered score statistic was

```
score_s = [lnL(s=√2) − lnL(s=1/√2)] / (√2 − 1/√2)
```

whose two nodes are symmetric in `ln s` but not in `s`: 0.292893 below and 0.414214 above. A
difference quotient estimates the derivative at the interval's *arithmetic* midpoint, here
1.0606602, not at `s = 1`. Expanding about `s = 1` gives a leading correction
`+0.060660·f″(1)`, and at truth `E[f′(1)] = 0` while `E[f″(1)] = −I_ss < 0`, so the registered
statistic has a non-zero expectation at truth. Since the reported `Z` is a mean over `SEM`, the
spurious `|Z|` grows as `√n` — at the planned n ≈ 800 event-instances it reaches ≈ 1.7·√I_ss,
exceeding the registered band of 3.0 for any per-event Fisher information `I_ss ≳ 3`. The
registered *control* would therefore have failed from the statistic's own form
(`:839-860`).

The correction was registered in place: reparameterise to `ln s`, giving denominator `ln 2`, nodes
unchanged, no re-costing required, and a leading error term that is odd with no `f″` contribution
(`:862-875`).

The document explicitly classifies this as "precisely the PA-2D-8 F3 `κ̂₂` class of defect"
(`:860`) — the same review class had previously caught a mis-formed statistic and a seed-stride
collision in an earlier pre-registration, with the stride fix landed at
`cluster/p3_2d_rhs2.sbatch:76-81` (`:1180-1181`). A defect class that recurs is a defect class
worth naming, and naming it is what made the second instance findable.

The blockers were subsequently worked down in a zero-compute resolution pass whose stated purpose
was "to turn each blocker from 'unknown' into a decision the author can answer in one line"
(`:1361-1367`) — the pass rules on nothing itself. As of this draft the verdict remains
LAUNCH-BLOCKED (`:1787`).

### 5.2 The failure mode of pre-registration: bands too wide to fire

Pre-registered bands are only as good as their power, and the campaign has a clean instance of
their failing. The b0 identity test (ledger row #177, `:2590-2601`) executed cleanly — 12 of 12
seed-pairs, all machinery gates passing to 10⁻¹⁴ — and the primary reading passed its bands. The
verdict was nevertheless **UNDISCRIMINATING**, because the registration's own control arm `B-R`,
which encodes the *refuted* arrangement, passed the same bands. One legitimate low-responsibility
event (seed 900108, index 2, weight ≈ 2.3 × 10⁻⁵, pull −0.79σ, not anomalous by any test)
inflated the raw standard errors into vacuity, with the band-width ratio `k̂` reaching 2.7 and
`k̂ > 1` pervasive.

Two things follow. The obvious one is that a heavy-tailed estimand — here a mean of odds, which
the row notes is "heavy-tailed by construction in this venue" (`:2613-2615`) — needs a
finite-moment redesign before it can carry a band at all. The less obvious one is the actual
lesson: **the test that caught this was not the primary reading but the registered control.**
A pre-registration that registers only its hypothesis, and not an arm known to be wrong, cannot
detect that its bands have no power. The driver's own printed verdict was superseded by the
control clause (`:2597-2600`).

---

## 6. The exoneration problem

A campaign that runs long enough accumulates refuted mechanisms faster than it accumulates
confirmed ones. That inventory is an asset — research-cycle rule 1 calls re-litigating an
exonerated suspect "this project's most expensive failure mode" and requires checking *both*
exoneration layers, the local claim-file list and the ledger's consolidated section, as a union
(`.claude/skills/research-cycle/SKILL.md:84-91`). It is also a liability, because checking it is
a retrieval problem and retrieval fails silently.

On 2026-08-27 a stage-0 exoneration check on the `[WGEO]` thread reported **PASSED** on a
mechanism that had been measured and refuted four weeks earlier. The refuted entry — tag `HB`,
"hard mass window as support truncation" — sat *two lines below* the entries the checking agent
had quoted, in the same unbroken list (`CLAIM_2D_BIAS_20260730.md:726-727` quoted, `:732-734`
missed). The chair caught it on a full re-read, and the resulting card names the mechanism of the
miss: the check "passed only because it checked 'candidate-window membership' and 'mass-kernel
family' and stopped two lines short of HB" (`CLAIM_WGEO_20260827.md:310-311`, reconstructed at
`EXONERATION_REGISTER_20260827.md:639-654`).

The diagnosis is the general one. The agent searched for **its own thread's vocabulary** — window
shape, geometry — rather than for the **physical mechanism**. Indexing an exoneration record by
the tag or thread name the filing session happened to choose guarantees that a future thread
proposing the same physics in different words will miss it. A hard cut, a floor, a filter, a
window, a truncation, a clamp, and an eligibility test can all denote the same object.

The structural fix was a **mechanism-indexed register**
(`EXONERATION_REGISTER_20260827.md`): one grep target, every entry indexed by the physical or
statistical mechanism in plain words a fresh investigator would naturally use, each carrying an
explicit "WHAT IT DOES NOT COVER" delimitation field, with a stated usage procedure that requires
grepping synonyms the proposer's own mechanism does *not* use (`:1-30`). It compiles from the
ledger, every `CLAIM_*.md` in the directory, the handoff and adjudication files, and a repo-wide
`EXONERAT` sweep (`:36-72`). Crucially, the register states its own scope honestly: it is a
records compilation, every bound is copied from source rather than recomputed, and verification
checked only that each cited anchor still exists (`:30-35`).

Section 8 records what happened when the register was audited.

---

## 7. Independence as an instrument

The campaign's strongest positive evidence for a mechanism did not come from more scrutiny of the
existing forensic. It came from removing the forensic entirely.

A searcher was firewalled from the investigation's history and given exactly one project document,
the symptom card. Its independence statement is explicit and enumerated: it read no ledger, no
claim file, no pre-registration, no runbook, handoff, adjudication, forensic or retrospective
file, and no git log; its evidence was the pipeline source code, banked numeric artifacts
(JSON/CSV — data, not opinion), and the published literature, with every file it read listed and
every forbidden file it wanted named (`stageL_p32d_fresh_search_20260827.md:1-17`).

Working from code alone it derived, as its candidate C1, the residual `S̄_φ(z)` factor in the 2-D
accepted-latent draw law: hosts drawn `∝ w_g · S̃_φ,g`, redshift drawn per host from
`k_g(z)·w_pop_eff(z)·S̄_φ(z)`, mass drawn, then the whole triple accepted with
`Bernoulli(S_4D)`, so that `S̃_φ,g` cancels and the realised joint law carries `S̄_φ(z)` once more
than the target law implies — "the ratio is exactly `S̄_φ(z_true)`, with no per-host constants
surviving" (`:108-130`). That is the same object the un-blind forensic had reached by an entirely
different route, and which §4.1's physics gate then re-derived a third time from code alone
(`PHYSICS_CHANGE_SBARPHI_20260827.md:745-756`).

Convergence from two routes, one of them blind, is a different kind of evidence from confirmation
by scrutiny — it is the closest thing a single-investigator campaign has to an independent
replication.

The blind searcher also produced something the un-blind record had not, and this is the argument
for the technique. It flagged that the banked "13.5–16 %" tilt may have been measured as a shift
in a marginal distribution when the decision-relevant quantity is the change in a *tail-weighted
statistic*, and that if so the "~7× too small" conclusion is an artefact of the measurement
functional rather than evidence against the mechanism (`:155-160`). A searcher inside the record
would have inherited that number as given.

---

## 8. Provenance is a precondition, and it decays at interfaces

None of the above adjudicates anything if the numbers cannot be tied to the code and inputs that
produced them. The repository's standing arrangement is that every simulation run records its
seed, git commit, timestamp and CLI arguments in `run_metadata.json`, and that any multi-GB input
not in version control carries a checksum pin at each consumer with a STOP gate on mismatch
(`CLAUDE.md` §"Reproducible simulation runs"). The dataset-pinning rule exists because a stale
local galaxy catalogue silently fed every local analysis until a fidelity gate caught it, with
the cluster copy of record differing. The `[WGEO]` card duly opens by discharging its dataset pin
— file, byte size, md5, and confirmation that the cluster copy is byte-identical — before any
number is computed (`CLAIM_WGEO_20260827.md:70-79`).

The arrangement failed anyway, in the way arrangements fail: at an interface, silently.
`run_metadata.json` is written **only** by `darksiren_emri/main.py`, so it fires only for jobs
that go through `python -m darksiren_emri`. In July the campaign moved to bespoke harness drivers
invoked directly from bespoke `sbatch` scripts, bypassing the entry point — and
`cluster/JOB_TEMPLATE.sbatch`, the file the cluster skill instructs you to copy, contained zero
references to it. Every new campaign inherited the gap. The verification is stark: the standard
pipeline's `run_20260628_seed600` has 100 `run_metadata` files, while `p3_2d_fleet_20260825`,
`p3_2d_rhs2_20260826`, `p3_b0_identity_fleet_20260823`, `csg_pilot_20260821` and
`o4_shards_20260821` have zero each (commit `67b18592`).

Two details matter for the chapter. First, the data was *recoverable*: the campaigns had invented
their own convention, `a22_stamp` sidecars — but no registry cross-referenced it, so the
provenance existed and was not reachable. Provenance that nothing indexes is provenance that does
not exist for practical purposes, which is the same failure as §6 in a different medium. Second,
a live check found 114 of 143 dataset directories unregistered against a prior estimate of about
30 — the discrepancy is itself the argument for mechanising the check rather than estimating it.

The fix was mechanical rather than procedural: a `cluster/write_provenance.sh` fail-soft helper
recording git commit, dirty-file count, branch, SLURM job and array ids, seed, hostname, UTC
timestamp and command, wired into the job template and into the three live `sbatch` scripts; plus
an UNREGISTERED/DANGLING dataset cross-check in `cluster/preflight.sh` that reports
`READY (WARN: N)` rather than hard-failing, on the reasoning that unregistered data is a
bookkeeping defect and not an operational blocker (commit `67b18592`). A registry backfill
dropped the unregistered count from 114 to 58, all pre-2026-07-28 legacy.

The transferable claim: a discipline that depends on a convention being followed will decay at
every interface where the convention is not enforced by the tooling. The correct response to
"the harness stopped writing provenance" is not a rule telling people to write it.

---

## 9. Honest limits: where this discipline failed, and what it cost

A methods chapter that reported only the catches would be an instance of the very failure it
describes. This section is the counter-evidence, and it should not be trimmed in revision.

**9.1 The exoneration gate has a measured failure rate of 2 in 8, with 3 further threads that
never ran it.** The near-miss of §6 was initially believed isolated. The register's own §6
concluded that `[WGEO]`/HB was "the ONE instance found" and that the failure mode "appears
contained to the one instance, not systemic" (`EXONERATION_REGISTER_20260827.md:681-690`) — while
admitting, as a method caveat, that the sweep had sampled only threads that *advertise* a rule-1
check.

An independent audit did not let the caveat stay a caveat. It found a second failed check one
week earlier: `CLAIM_P3_WBHZERO_20260825.md:34-36` states that "no prior exoneration covers the
candidate mass filter", which is false on its face, since HB *is* an exoneration of the candidate
mass filter and `[WINDOW-MEMBERSHIP]` is literally about candidate eligibility — both in the list
the card says it searched. The card's later gate pass amended the exoneration check and still
never named HB. The audit also found three post-2026-07-30 claim cards carrying no exoneration
check at all — `CLAIM_F0_SEL_20260825.md`, `CLAIM_B0_FINITE_MOMENT_20260824.md`,
`CLAIM_P3_2D_20260825.md` — the first of which reaches a drafted verdict on an object adjacent to
three standing exonerations (`:842-864`).

The corrected conclusion, which supersedes the register's own: **of the threads that ran a rule-1
check, 2 of 8 failed, and 3 further cards ran none. The failure mode is not contained to one
instance** (`:871-873`).

Materiality, stated honestly in both directions: in both failed cases no scientific harm followed,
because the threads happened to be about genuinely different objects. The audit's own phrasing is
the right one to quote — they were "separate objects by luck of the physics, not by any check that
ran" (`:853-854`). A gate that passes by luck has not passed.

**9.2 The fix for the failure failed its own usability test on first audit.** The
mechanism-indexed register was built to make §6's failure structurally harder, and its central
promise is that a fresh investigator's own words will land on the covering entry. Six probes were
run using only words a proposer would choose before reading the file. No probe landed on a wrong
entry and no probe produced a false clearance — but the audit found a systematic mechanical
defect: **the file is hard-wrapped at ~100 columns, so synonyms deliberately placed to be
greppable straddle a line break and are invisible to a line-based `grep`**
(`EXONERATION_REGISTER_20260827.md:807-815`). Confirmed 0-hit despite being present verbatim:
`sigma clipping` (`:298-299`) — arguably the single most natural phrase for the `[WGEO]`
hypothesis — along with `peculiar velocity`, `log-mass draw`, `truncated lognormal kernel` and
four others. Thirty-one "Search also" lists exist and most of them wrap. A second defect: Greek
tokens such as `β_G` and `Σ_glob` appear only in Unicode, so an agent typing `beta_G` at a
terminal gets zero hits (`:817-820`). Both were repaired; neither required a content change.

The lesson is uncomfortable and specific. A retrieval instrument's *content* being correct is not
its working. The register's bounds audited 9 of 9 faithful to source (`:822-833`), and it still
would have failed the next real query, for a reason that has nothing to do with physics. **Any
guard-rail that is a search surface must be tested by searching it, with queries written by
someone who has not read it.**

**9.3 The near-miss was caught by a full re-read, not by the gate.** In the `[WGEO]` case the
covering entry was found because a synthesis chair re-read the entire list, not because any
mechanism flagged it. That is not a repeatable control — it does not scale with the record, and
it is precisely the thing a growing inventory makes less likely.

**9.4 The provenance gap ran undetected for roughly six weeks.** From the July move to bespoke
harness drivers until 2026-08-27, five named campaign directories accumulated zero
`run_metadata` files, and the gap was propagated by the very file the cluster skill tells you to
copy. It was found by a dedicated sweep, not by any check in the normal path
(commit `67b18592`).

**9.5 Nothing in the physics is settled by any of this.** The campaign's central residual — a
factor of roughly 2.5 between two estimates of the same quantity in the 2-D mirror venue — is
**UNATTRIBUTED and PARKED** (ledger rows #209–#211; `PHYSICS_CHANGE_SBARPHI_20260827.md:659`).
The completion-mass axis was cleared twice, an author-granted alternative counterfactual construction
refuted the remaining diagnostic (`X_alt = 0.9997 ± 0.0003`, row #210), the measured selection
double-weighting is sign-correct but about 7× too small, and the repair for that double-weighting
was itself found mis-specified (§4.1). The `[HIER]` thread remains LAUNCH-BLOCKED. What this
chapter documents is a method for *not* closing prematurely, demonstrated on a problem that is
still open. That is the honest claim and it is the only one available.

---

## 10. Open items for revision

1. ~~**Verify the wiki quotations in §1.1 at source.**~~ **DISCHARGED** `[OPUS-ORCH 2026-08-27]`.
   The drafter flagged these as librarian-sourced and not re-read — correctly, since a chapter about
   inherited-claim failure containing inherited unverified claims would be self-refuting. All four
   were re-read at source in the vault and hold:
   - channel-localization — `wiki/concepts/scientific-computing-validation.md:143`
   - the circular-validation smell test — same file, `:60` and `:67`
   - the missing-anchor cap (verdict capped at `supported`, never `verified`, absent an external
     anchor) — `wiki/meta/research-cycle-amendments.md:22`, amendment C1, dated 2026-08-15 and
     attributed to an author ruling
   - verify-the-decisive-claim-yourself — `wiki/meta/cross-project-memory/general.md:881`

   Two notes for revision. The missing-anchor cap is the sharpest of the four for this chapter's
   purposes and §1.1 currently underuses it: it bears directly on §9.5, since the central residual
   has no external anchor and so cannot be reported as `verified` under the project's own rule.
   And `general.md:881` states the pattern more strongly than §4 quotes it — "even a review that
   just caught your own mistake" — which is worth carrying, because that is the case that actually
   arose twice in the two days this chapter draws on.
2. **One page was NOT FOUND.** The consult found no vault page titled along the lines of
   "verifier output is evidence, not authority" — that phrasing is project-local (a
   `darksiren-emri` memory file). The pattern itself is present in
   `wiki/meta/cross-project-memory/general.md` and is quoted from there. Do not cite it as a
   promoted concept page.
3. **Decide the chapter's scope on `[HIER]`.** §5.1 uses a pre-registration that is still
   LAUNCH-BLOCKED. If it launches before submission, the section needs a closing sentence; if it
   does not, say so, since a blocked pre-registration that never ran is itself the strongest form
   of the section's argument.
4. **§9.5 must be re-checked against the ledger at submission time.** It asserts the central
   residual is unattributed and parked. That is the state at rows #209–#211; it is the one claim
   in this chapter most likely to go stale.

---

## 11. Summary of transferable rules

1. In a bias hunt, plausibility is the selection criterion, not evidence. Score a mechanism by
   what it can be shown *incapable* of, not by how well it fits.
2. Before testing a mechanism, check that it can touch the objects the symptom is measured on.
   That check is free and it fired decisively twice in `[WGEO]`.
3. Measure what is on disk; then register; then run. The three stages differ in cost by orders of
   magnitude and the cheap stage kills most leads.
4. Attribution is a separate measurement from magnitude. Decompose the headline exhibit before
   funding anything on it. (0.33 % versus 99.67 %.)
5. Compute decisive numbers by two independent routes, and report the remainder as unattributed
   rather than assigning it to the preferred object.
6. Run adversarial verification as a separate role, and treat its output as evidence to
   re-derive, never as authority. Tag every number by how many derivations support it.
7. Forbid the verifier from executing the registered measurement, explicitly.
8. Register numeric bands *and* an arm known to be wrong. Only the control can tell you the bands
   have no power.
9. Index the record of refuted mechanisms by mechanism, in a fresh investigator's plain words,
   not by the tag the filing session chose. Then test the index by searching it.
10. Let something search blind. Convergence from an independent route is worth more than
    additional scrutiny of the same route.
11. Mechanise provenance at the interface, not in a rule. Conventions decay wherever tooling does
    not enforce them.
12. Bank nulls and refuted figures with the same care as positives, so the campaign cannot
    re-propose its own dead leads.
