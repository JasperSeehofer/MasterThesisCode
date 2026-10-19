# B3.2 [POP] wave-2 PREP dispatch — IMPLEMENTATION DECLINED (STOP already on record)

**Launched under rows #222/#223 — charter node B3.2.** Date 2026-08-29, HEAD `dd63fe0c`. This
record returns to the orchestrator; it never addresses the author.

## 0. What this worker was asked to do, verbatim intent

The wave-2 PREP dispatch instructed: *"IMPLEMENT the flag exactly as presented in
`PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` (panel clean after 0 rounds)... You ARE
authorized (row #223, charter B3.2) to edit `darksiren_emri/bayesian_inference/bayesian_statistics.py`
for this flag only... thread `completion_population_prior` through evaluate()..."* — i.e. treat the
presentation as an approved physics-change package ready for the standard
presented → implemented → verified ledger sequence (the `mass_filter_geometry` pattern, B5.1).

## 1. What the cited presentation and the gate ledger actually say — read before writing code

Read `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md` end-to-end (as this dispatch required)
before touching any file. Its own disposition, §13 item 2, verbatim:

> "**No code under this presentation.** The flag as an *adoption* candidate is refused by the
> [WPOP-TUNING] boundary (§10) once §F is in hand. Building it as a *counterfactual instrument*
> is legitimate but its yield is limited to §6.1's algebra validation and the Δ_D read (§6.3);
> the recommendation is **not to spend the 45–69 CPU-h C2 arm** on it in wave 2, and to strike
> B3.2 from the shared-instrument dependency L1..."

and §0 line 4: *"this presentation does **not** authorise code for an adoption path."*

The gate ledger (`docs/gates/PHYSICS-GATE-LEDGER.md`, appended in wave-1 commit `dd63fe0c`,
already on disk before this dispatch) already carries the presentation row with approval column:

> `row #223 (standing grant, charter node B3.2) — **PRESENTED WITH A STOP: premise REFUTED by
> generator provenance; NO CODE authorised by this presentation**`

and its own trailing recommendation: *"do not spend the 45–69 CPU-h C2 arm; strike B3.2 from L1;
return the provenance finding..."*.

The underlying finding (§F of the presentation, independently re-checked against `git show
03cfe80:master_thesis_code/dark_siren_injection.py:328` and
`seed61000/prepared_cramer_rao_bounds.csv`, both cited with file:line in the presentation): the
production CRB set's dark hosts are drawn from `(1−f)·dVc/dz/(1+z)` — **byte-identical to the
estimator's existing constant-comoving completion prior**. The z-dependent "M1" the flag would
install (`Model1CrossCheck`'s extracted dN/dz) is a *different* object, used only as the p_det
pool's sampling measure, never as the production event law. Installing it as `completion_population_prior="m1"`
would therefore not correct a generator/estimator mismatch — it would manufacture one, and do so
at 60× the size of the already-EXONERATED `[WPOP-TUNING]` mechanism (register item 5,
`EXONERATION_REGISTER_20260827.md:382-388`, "tuning the population-rate prior weighting to absorb
the residual", bound ≤ +0.0004 at 10% misspecification — the m1 shape spans a 0.53–1.39 ratio,
i.e. the entire measured dark-class tilt, not a 10% perturbation).

## 2. Disposition of this dispatch

**Declined as issued.** "Implement exactly as presented" cannot mean "implement" here, because
what is presented is a REFUTED premise with an explicit no-code disposition already ratified into
the gate ledger before this dispatch was written. Per repo CLAUDE.md's decision-gating rules
("an approval never propagates to a decision whose inputs did not exist when it was given" — and
here the refutation *is* the input, computed at zero cost specifically to answer whether B3.2 was
warranted) and per the physics-change protocol (no formula/prior swap without a documented
derivation that survives review — §F is exactly the derivation, and it says don't), row #223's
standing grant to "edit bayesian_statistics.py for this flag" is read as conditional on the flag
being a live adoption candidate. It is not: the presentation it points to is a stop sign, not a
green light, and the ledger already says so in the SAME wave the code-authorization was issued.

**No files were edited.** `git status`/`git diff` confirm `darksiren_emri/bayesian_inference/bayesian_statistics.py`
and every other physics-trigger file are untouched by this worker. No `completion_population_prior`
flag exists anywhere in the tree. No tests were added (nothing to regress-test). No `[PHYSICS]`
commit is proposed.

**Ledger:** no "implemented"/"verified" rows are appended — appending PASS rows for code that was
never written would violate A11 (every number carries a real source) and standing rule 3. Instead
a single append-only note is added to the existing `PHYSICS-GATE-LEDGER.md` (new row, no existing
text edited) recording that the wave-2 PREP dispatch reached this node and confirms the STOP
still holds; see the diff below.

## 3. What the orchestrator should do with this

Two honest paths, both already scoped in the presentation itself:

1. **Accept the STOP** (matches the ledger's own already-recorded state and the docket §13
   recommendation): strike B3.2 from L1 (`SYNTHESIS_DOCKET_1_20260829.md:214`) so B1.2's S0-B
   driver does not carry it as a dependency; carry forward the two deliverables the presentation
   says the branch DOES return — (a) the provenance finding (rows #137–#144 re-read: production's
   dark class is comoving-drawn, so the −0.612/−0.635 tilt is not a population-prior mismatch and
   the thread stays with the B1/B4 selection-object investigation), (b) the paper-facing G7 row 16
   sensitivity number (§12: a population-shape change of the size between the repo's two "M1"
   objects would move the dark-class score by −0.60 on bins 2–5 — real, but not what tilts the
   mock).
2. **If the orchestrator explicitly still wants the counterfactual *instrument*** (not an adoption
   path — the presentation allows this reading, §6.3/§13 item 2 second sentence) — that is a
   *different, narrower* authorization than "implement the flag" and should be re-issued as such,
   with the [WPOP-TUNING]-collision framing made explicit in the dispatch text so a future builder
   does not read it as production adoption. This worker recommends against it per the
   presentation's own cost/yield argument (45–69 CPU-h for an algebra check whose answer is
   already derivable analytically, §6.0) but does not block a builder from taking it up if ordered.

Either way, this is a fresh finding-dependent call and, per CLAUDE.md's approval-scope rule,
returns to the orchestrator as such rather than being executed under the standing grant as issued.

## 4. Exoneration check (standing rule 5, re-run independently by this worker)

Grepped `EXONERATION_REGISTER_20260827.md` for `WPOP`, `population`, `m1`, `comoving` and
`BIAS_HISTORY_LEDGER.md` §2 (lines 127–215) for the same, 2026-08-29 — same hits as the
presentation's §10 (register item 5 [WPOP-TUNING], line 64 ledger anchor). No new hit changes the
picture; this worker's finding agrees with the presentation's exoneration-boundary read.

## 5. Files touched by this worker

- `docs/gates/PHYSICS-GATE-LEDGER.md` — one appended row (see below), no existing row edited.
- This file (new).

No other file in the tree was written or edited by this node.
