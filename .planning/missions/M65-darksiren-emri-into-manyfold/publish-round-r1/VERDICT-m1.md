# VERDICT — m1 sealed-mock cell disposition (docket R25)

**Adjudicated by:** mission M65, 2026-09-21.
**Question:** the m1 read has sat unadjudicated since 2026-09-04. Against the pre-registered cell
table, does the TUNED clause or the NOT-TUNED-AT-RAIL clause bind?

# VERDICT: UNRESOLVED-AS-REGISTERED

**The pre-registration does not cover this posterior. No tie-break is proposed, because none was
registered, and inventing one is the author's call, not the adjudicator's.**

---

## Evidence access and method

Read-only via `git show origin/fix/p32d-classg-venue-repair:<path>`; that branch was never checked
out, fetched or merged. No simulation, cluster, SLURM, ssh or preflight command was run; T-1 was
not attempted in any form. Two independent agents worked the evidence: an adjudicator drafting
against the registered text, and a **fresh-context adversarial verifier dispatched before the
draft existed** and never shown it. They converged.

**Operative pre-registration:** `D_SEALED_REGISTER_DOSSIER.md` §2 — *not* `REGISTRATION_DRAFT.md`
§6, which declares itself *"PROPOSED THROUGHOUT — nothing here is frozen, drawn, generated, or
launched."* Every threshold cited below is the sealed one; every divergence from the draft is
flagged as a finding and adjudicated **against**, not with.

## The numbers, independently re-verified

All four reproduce exactly against the raw record (`VERIFY-R1.md` §1):

| quantity | 2D (primary) | 1D (replicate) |
|---|---|---|
| distance from registered **anchor** | **2.73σ** (anchor 0.6659) | **2.58σ** (anchor 0.6670) |
| distance from **truth** h=0.67 | **3.02σ** | **2.79σ** |
| mass at edge node h=0.600 | **4.39 %** | **2.59 %** |
| posterior mean / σ_h | 0.627674 / 0.014006 | 0.631084 / 0.013939 |
| MAP | 0.630 (interior) | 0.630 (interior) |

Two readings the numbers do **not** support, both of which a careless verdict would assert:

- **"2.73σ off" unqualified.** That is distance from the 0.73-run re-baseline anchor. From the
  actual injected truth the 2D posterior is **3.02σ — outside 3σ**. The anchors are also
  channel-specific, so the 2D and 1D figures are not distances to the same point.
- **"4.4 % of mass railed."** The number is right; the picture it evokes is wrong. Density is
  *declining* steeply toward the edge (2D: 4.4 at h=0.600 vs 13.8 at h=0.610) and the MAP is
  **interior**. The registered 1e-3 floor is very tight on a 41-node grid (uniform ≈0.024/node),
  so it fires on any modest lower tail.

## Why both clauses fire

Applying the sealed register's own thresholds:

| clause | registered test | observed (2D) | fires? |
|---|---|---|---|
| **TUNED** | `\|mean−0.6659\| ≤ 3σ` **AND** `\|MAP−0.665\| ≤ 3σ` | 2.73 ≤ 3; 2.50 ≤ 3 | **yes** |
| **NOT-TUNED**, first clause | `\|mean−0.6659\| > 3σ` | 2.73 not > 3 | no |
| **NOT-TUNED**, rail sub-clause | edge-node mass `> 1e-3` | 0.0439 | **yes**, iff standalone |
| **INTERMEDIATE** | registered as *"neither"* | both fire | **excluded on its face** |

Both channels agree, so the register's own INTERMEDIATE escape hatch ("1D/2D disagree in cell")
does not apply either. **The ambiguity is in the text, not in the data** — which is precisely why
more compute cannot fix it. The docket says so: *"a venue replicate cannot resolve the
registration-text ambiguity"* (R26).

The record concedes the defect in its own words (`READ_RECORD_m1.md` §4): *"the table is
inconsistent with itself on this posterior in two places (verifier finding)."* The chair
deliberately banked nothing: *"Status of the cell call: NONE MADE BY THE CHAIR."*

## Rulings made, clause by clause

Each candidate resolution was tested and rejected on the registered text:

1. **Does a primary channel resolve it?** No. 2D is the registered primary, but *each channel
   independently satisfies both clauses*. Not a channel-mixing artifact.
2. **Is there registered ordering or precedence?** No. The draft's *"gates scored before any cell
   is read"* orders the **gate layer vs the cell layer**, not cells among themselves. The
   strongest pro-resolution argument — that the g-censoring gate's *"a BOUND … see the NOT-TUNED
   row"* routes the read into NOT-TUNED before cells are compared — was considered and **rejected**:
   the cross-reference is a pointer to where the rail rule lives, not an assignment instruction;
   its operative content is NO-READ-vs-BOUND (whether anything banks at all), not which banked
   cell wins; and the same condition is NO-READ in the parent draft, so the reading is not even
   stable across documents. *Treating it as precedence would be supplying the missing rule, not
   applying one.*
3. **Does either clause fail on a strict reading?** No. The floor is a strict `>` and 0.0439 ≫
   1e-3; 2.73 ≤ 3 on a two-sided absolute distance. Both hold.
4. **Is INTERMEDIATE broad enough?** Not as registered for (m1) — it says *"neither"*. **But see
   Q1b:** the parent draft registers INTERMEDIATE as *"none or **several** of the above"*, which
   covers a both-fire posterior exactly. The (m1) table narrowed it. **Electing the superseded
   parent clause over the operative one is itself a precedence ruling** — so it goes to the author
   as a question, not applied here.

## What m1 establishes regardless of the ruling

Cell-free and bankable, verified twice:

- The posterior is centred at 0.628–0.631 — **0.0423 (2D) / 0.0389 (1D) below the 0.67 truth**,
  with 71 % / 62 % of mass at or below 0.63.
- The run is **procedurally clean** except the pool-pin override: *"NO-READ triggers: none
  fired"*; resolved-flags 13/13; g-population disclosed; physics floor a no-op.
- The anchor comparison is **same-pool** (both runs link the identical pool, per-file manifest
  list-md5 `75f4030d5d…` on both).
- **The pre-registration was genuinely sealed before the result existed.** The dossier's threshold
  block hashes to `c908790d…` at all four dossier commits, including ~11 h before the first array
  task started. No post-hoc threshold editing. *This is the thing a sealed mock most needs to
  prove, and it proves it.*
- Cost 57.63 core-h against a 75 core-h cap.

## What m1 does NOT establish — and this is the load-bearing paragraph

**m1 makes no anti-tuning claim, does not substitute for T-1, and cannot discharge it.** This is
not M65's opinion; it is registered, repeatedly, in the programme's own words:

- `D_SEALED_REGISTER_DOSSIER.md` §2: *"one seed; the truth is public (**no anti-tuning claim can
  bank from (m1)**)."*
- `D_SEALED_REGISTER_DOSSIER.md` §1.1: *"(m1) is a mechanism check on a public truth — **explicitly
  NOT the T-1 verdict**."*
- `READ_RECORD_m1.md` §3: *"the (m1) node makes no anti-tuning claim."*
- `DESIGN_GATE_computability_m1.md` §4: *"(m1) is explicitly registered as NOT the T-1 sealed
  verdict."*
- `REGISTRATION_DRAFT.md` §4: *"**the anti-tuning guarantee is the CODE FREEZE, not the secrecy of
  the number**"* — and for (m1) the freeze is only the *weaker ancestor form*; the strict-equality
  form is reserved for the sealed (m2).
- `REGISTRATION_DRAFT.md` §2 on why the 0.67 closure *"is neither sealed nor current"*:
  *"**Unsealed:** the truth is in the directory name … Nobody who ran, read, or fixed the estimator
  between 07-29 and today was blind to it."*

Further, the arm that would actually test the mechanism is **not evaluable on this grid at all**:
*"the TRANSFER/UNBIASED split is NOT-EVALUABLE on `H_GRID_41`"* — under TRANSFER the 0.67 pool's
expected mean (0.606–0.611) sits within 0.5σ of the 0.600 floor. And even a favourable ruling
leaves the stamp partial: *"the anti-tuning stamp stays PARTIAL (unsealed; TRANSFER unresolved on
this grid)."*

The draft even registers that a TUNED (m1) would not close matters: *"| (m1) | TUNED | HALT paper
claims; (m2) still runs (the sealed test is the verdict-bearing one) |"*.

**However R25 is ruled — TUNED, NOT-TUNED-AT-RAIL, or INTERMEDIATE — kill criterion (1)
(anti-tuning) remains untested and the sealed test remains the verdict-bearing one.**

## A material input R25 did not have

When R25 was framed, it presented (a)/(b)/(c) without noting that **the earliest registered
formulation anchored the band on 0.67, not on 0.6659** (`GRAPH1_ADDENDUM_PROPOSAL_20260903.md`
§1.0), and that `|mean − 0.67|` = **3.02σ (2D)** — *outside* 3σ in the primary channel. The
dossier deliberately re-anchored to 0.6659 and demoted the truth comparison to reported-only,
**which is why TUNED fires at all.** This does not change the adjudication (the dossier supersedes
the addendum for (m1)), but it bears directly on Q1 and was not in front of the author.

An author ruling (b) TUNED would be ruling that a posterior lying outside the band around the
*actual truth* is nevertheless "tuned to the 0.73-production value." The chair flagged the same
tension independently: *"a posterior 0.038 below the anchor with < 1 % mass above 0.66 is not what
'tuned to 0.666' was written to mean."*

## An evidentiary defect that outranks the tie

**Every decisive number in this docket is prose-only testimony.** `MANIFEST.md5` lists **85**
entries; **43** are tracked in git. Absent: the 41 posterior JSONs, `event_likelihoods.csv`, and
the scorer `t0_read.py` itself. The 2D primary channel's posteriors (~3.3 GB) were deliberately
left on the cluster. Internal arithmetic coheres to ~1e-4 — real but weak corroboration, since a
systematically wrong scorer produces an equally self-consistent set.

This interacts with convention **C-20**: the workspace expires **2026-09-23 with zero extensions
remaining**. See the go/kill memo — this is the most time-critical item M65 found, and it is
independent of how R25 is ruled.

## What would close it — four questions answerable from a phone

Tagged per CLAUDE.md approval scope. **Q2 should be answered first: a "No" moots Q1 entirely.**

> **Q2 [RULE] — the pool-pin override (R28).** The registration named
> `injection_pool_depth15_50k` (500 files) and STOP-gated on it; the 0.67 run actually links
> `injection_pool_mix200k_20260728` (707 files, content-verified identical to the pinned reference
> manifest; the 0.73 anchor run links the same pool). **Ratify as a registration erratum (Yes)**,
> or **rule the (m1) read NO-READ under the registered text (No)**?

> **Q1 [RULE] — the tie itself.** The posterior satisfies TUNED (2.73σ / 2.50σ, 1D agreeing) *and*
> NOT-TUNED-AT-RAIL (4.4 % on the 0.600 floor, floor 1e-3) simultaneously; the register has no
> precedence rule and its INTERMEDIATE cell says "neither". Which cell binds —
> **(a) NOT-TUNED-AT-RAIL, read as a bound**, **(b) TUNED**, or **(c) INTERMEDIATE, banked**?
> *(the chair's own flagged recommendation at R25 was (a), with the σ-distance reported alongside)*

> **Q1b [RULE] — only if you want the text, not the chair, to decide it.** The parent registration
> (`REGISTRATION_DRAFT.md` §6) registers INTERMEDIATE as **"none or several of the above"**, which
> covers a both-fire posterior exactly; the (m1) table narrowed it to "neither". Does the parent
> wording govern (→ INTERMEDIATE *by text* rather than by fiat)? **Yes / No.**

> **Q3 [DO] — the erratum before (m2) (R27).** Authorize drafting, as a dossier section for your
> ratification, (i) an explicit both-fire precedence row and (ii) a stated treatment of a posterior
> whose measured σ is *narrower* than the anchor's (0.014 vs 0.0185) — with **no (m2) launch until
> ratified**? **Yes / No.**

> **Q4 [DO] — the joint_r1 sibling (R26).** Submit the second venue (≈75 core-h, ≈6 wall-h) before
> (m2)? Note it **cannot** resolve Q1. **Yes / No.**

## Structural recommendation

The defect will recur **verbatim** on (m2): the same table, the same rail, the same collision —
plus the unreconciled NO-READ-vs-BOUND divergence. A precedence clause is needed in the register
itself, not just a one-off ruling here. That is R27's scope and the author's [DO] to authorize;
M65 does not draft it.

---

*No tie-break was invented. Sentences of adjudicator reasoning rather than quotation are marked
[inference] in the full draft. The verdict rests on the sealed dossier as the operative
pre-registration, with every draft divergence flagged.*
