# PRE-REGISTRATION — [P3-2D] class-G venue repair: testing the free residual-ladder prediction (stage 2)

**Date:** 2026-08-27 · **Thread:** `[P3-2D]` (PARKED at "UNATTRIBUTED-bounded", row #211,
`607ac886`; STUCK symptom card `STUCK_P3_2D_SYMPTOM_CARD_20260826.md`; **this document does NOT
reopen the PARK** — it registers the test of a prediction banked under a fresh author grant to
implement the class-G venue repair on branch `fix/p32d-classg-venue-repair`, per the residual
ladder in `p32d_residual_accounting_20260827.md` §8). Inherits the C-A governance stack
(`PREREGISTRATION_P3_2D_20260825.md` and its PA-2D-1..10 amendments) wholesale; only
repair-specific items are registered here. **Append-only after commit** — everything below the
`---` divider is an amendment, never an edit above it.

**Why this document exists (integrity note).** The prediction it registers was already banked,
git-committed, and genuinely timestamped in `p32d_residual_accounting_20260827.md` §8
(2026-08-27, commit `a662e684`) — *before* the repair implementation began. An adversarial audit
found that note "would not be discoverable by an R1 sweep or by the next session's prereg scan"
because it is not a `PREREGISTRATION_*.md` and carries no run identifiers or design matrix. This
document exists solely to make that already-banked prediction discoverable and falsifiable in the
project's canonical stage-2 form before the 24-seed re-run is submitted. **The prediction itself
is copied verbatim below, not restated or softened.**

## 1. Hypothesis, registered prediction, and the ladder under test

### 1a. The registered prediction (verbatim, from `p32d_residual_accounting_20260827.md` §8)

> **LHS2(bt) = 0.00739968 ± 0.00024951, X = RHS2/LHS2 = 1.961 ± 0.090.**

RHS2 is held **frozen** at 0.01451300 ± 0.00045293 (`PREREGISTRATION_P3_2D_20260825.md:310-311`,
banked σ freeze PA-2D-9) — **zero new RHS compute** for this re-run.

**Strict variant (secondary read, registered but not primary):** zeroing the 3 rows with both
`A2 = 0` and `B2 = 0` (w2 undefined 0/0) instead of including them at summand 1 gives
LHS2 = 0.00735830, X = 1.972 (`p32d_residual_accounting_20260827.md` §4, last paragraph).

**PRIMARY VARIANT, REGISTERED NOW, BEFORE THE RUN: the standard form (0.00739968 / X=1.961) is
primary.** This is the implementer's registered choice on the repair branch — the fleet driver's
dead-row convention (`p3_2d_fleet.py:632-646`, `[DEFECT 2 repair]` comment, diff inspected on
`fix/p32d-classg-venue-repair`) applies PA-2D-1 F16 ("A2=0 ⟹ w2=0 ⟹ summand 1") to *all* rows
with `A2=0`, including the 3 pathological `A2=B2=0` rows — i.e., it implements the standard
variant, not the strict one, as its production code path. **The strict variant is a secondary
report-only read for this re-run, never a fallback selected after seeing where the standard
variant lands.** Stating this in writing, before the run, is the entire point of this section —
see §4's overclaim caveat for why this distinction is load-bearing.

**bc/coded arm — no primary numeric prediction registered.** The residual ladder's total
unattributed factor differs materially by arm even before this repair:
×1.961 ± 0.090 (bt/twin) vs ×2.348 ± 0.113 (bc/coded) — see §3 ARM-COHERENCE and §4 MIXED band.
The source document gives per-arm figures for step 1 only; steps 2-3's bc-arm numbers are cited
as "the companion working notes," not restated as a frozen point prediction here. **This asymmetry
is a judgement call, flagged as ambiguous for the reviewing author rather than silently assumed
to transfer** (see the accompanying task report for the explicit flag list).

### 1b. The ladder under test

Each step, its code citation, and its status:

| step | mechanism | factor | status |
|---|---|---|---|
| 1 | S̄_φ(z) double-application (mass-marginal survival applied twice to already-once-accepted latents) | ×1.1585 | **banked, rows #209/#210 (`aaabc829`, `936236db`) — NOT under test here** |
| 2 | venue latent-mass floor (`correspondence_1d.py:1708` pre-repair, now removed per the diff on `fix/p32d-classg-venue-repair`) vs the F2 MINOR-6 `S_4D(M≤0):=0` guard (`p3_2d_companion.py:281`) | ×1.1944 | **NEW, under test — this is the repair's Defect 1** |
| 3 | LHS-side dead-row exclusion (`p3_2d_fleet.py:632`, pre-repair `live` filter dropped `A2=0` rows from the sum) vs the registered convention PA-2D-1 F16 already implemented on the RHS (`ca_rhs_scorer.py:1315-1324`) | ×1.0680 | **NEW, under test — this is the repair's Defect 2** |

Step 1 is restated for ladder completeness only; the run tests whether steps 2 and 3, applied
together on a fresh 24-seed fleet, reproduce the frozen point prediction end-to-end.

## 2. Arms, venue, and instruments

| arm | what | runs |
|---|---|---|
| **B2-T (twin, primary)** | LHS2 from a fresh 24-seed b0i-2D fleet under the repaired class-G venue (Defect 1 + Defect 2 fixed) | 24 cluster-array tasks |
| **B2-C (coded, secondary)** | same fleet, coded numerator — report-only per §1's bc caveat; no frozen point prediction | same 24 tasks |
| **RHS2 (frozen)** | 0.01451300 ± 0.00045293 — zero new compute, reused verbatim from PA-2D-9 | — |

**Instruments (both already modified on `fix/p32d-classg-venue-repair`, inspected this session,
NOT edited by this document):**
- `darksiren_emri/validation/correspondence_1d.py` — `_draw_2d_accepted_latents`: the mass-floor
  clip (`np.clip(m_true_batch, _M2D_MASS_FLOOR, None)`, formerly line 1708) is replaced by outright
  rejection of `M ≤ 0` draws inside the batch/round loop (`valid_mass_batch` ANDed into
  `accept_mask`), never a post-hoc filter of an already-materialized sample. The diff's own
  docstring states the RNG draw order is **unchanged** — the rejection is an extra AND-condition,
  not an extra draw. This claim is exactly what the RNG-STREAM gate (§3) exists to check
  independently rather than take on the diff's word.
- `results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py:632-646` — `stage_lhs2d`: the
  `live` filter is removed from `sum_acc`; the sum now runs over all F-0-accepted rows,
  matching `ca_rhs_scorer.py:1315-1324`'s convention exactly, per the code comment
  `[DEFECT 2 repair] sum over ALL F-0-accepted rows (no live filter) -- honours F16's dead-row
  convention identically to the RHS instrument`.

## 3. Gates — must pass before any number counts

The C-A stack verbatim (ACC re-closed-looped, M2-LINK, PA-CA-11 out-root guard, A22 resolved-flag
stamps — now seven, per §5's invariant list) **plus three gates specific to this repair**,
registered because this diff is the highest-risk change this thread has made to the venue's
draw law since the venue was created:

- **RNG-STREAM gate.** On any case containing NO floor rows (`M_true == 1.0` under the old code)
  and NO dead rows (`A2 = 0`), the repaired code must reproduce the OLD banked per-event stream —
  not just the summary statistic, the full sequence of drawn (host index, uniform, mass normal,
  Bernoulli) values for every accepted latent in that case. **Why this is the highest-risk failure
  mode:** if the repair changed the ORDER or COUNT of RNG consumption anywhere in
  `_draw_2d_accepted_latents` (e.g., because rejecting instead of clipping changes how many
  rounds a batch needs to fill its quota), every seed's downstream draw stream shifts, and
  comparability with the frozen RHS2 and every other banked b0i-2D fleet artifact this thread has
  produced (rows #196-#211) breaks silently — the fleet would still run, still produce numbers,
  and those numbers would be wrong for a reason invisible to the bands in §4. Registered check:
  select the seed/case subset with zero floor rows and zero dead rows under the OLD code (must
  exist — the pre-repair fleet's own diagnostics identify them), re-run those specific cases under
  the repaired code with the SAME seed, and diff the full per-event RNG-derived column set
  (host index, drawn mass, drawn Bernoulli uniform), not just `sum_acc`. FAIL = any single value
  differs beyond float round-off. On FAIL: INSTRUMENT-DEFECT, no verdict banks, return to the
  implementer — the repair's own claim that "the M≤0 rejection is an extra AND-condition, not an
  extra RNG draw" (`correspondence_1d.py` diff docstring, inspected this session) needs a
  code-level fix, not a re-interpretation of the numbers.
- **BYTE-IDENTITY gate.** On the SAME no-floor-row, no-dead-row case subset, the repaired code's
  final per-seed `LHS2_s` value must equal the pre-repair banked value bit-for-bit (or to float
  round-off if any dtype changed). This is the summary-statistic-level counterpart to RNG-STREAM
  and is cheaper to check first — RUN THIS ONE FIRST; if it fails there is no need to diff the
  full RNG stream to know something is wrong, and if it passes, RNG-STREAM should be run anyway
  because two different RNG streams can coincidentally integrate to the same sum on a small case
  subset. FAIL = INSTRUMENT-DEFECT.
- **ARM-COHERENCE gate.** All three corrections in the ladder (§1b) are common-mode — they act on
  the shared LHS2 construction, not on anything that distinguishes bt from bc — so the bt/bc arm
  SPLIT (the ratio or difference between the two arms' LHS2, independent of the overall level)
  should be UNTOUCHED by the repair to within the arms' own pre-repair scatter. Registered check:
  compute `LHS2(bc)/LHS2(bt)` under the repaired code on the same 24 seeds and compare to the
  pre-repair banked ratio (from PA-2D-9's frozen numbers: 0.00431338/0.00500770 = 0.8613). A moved
  arm split — this ratio shifting by more than its own propagated SEM band — is a RED FLAG, not a
  result: it would mean the repair introduced an arm-asymmetric effect the ladder's mechanisms
  cannot explain (both Defect 1 and Defect 2 are, by their own code citations, applied identically
  regardless of `catalogue_numerator_survival`'s bt/bc setting). FAIL = INSTRUMENT-DEFECT,
  diagnosed before any CONFIRMED/REFUTED/MIXED verdict banks.

## 4. Bands and verdict map

**Anchor derivation (not asserted).** The registered primary prediction carries SEM 0.00024951 on
LHS2(bt) (propagated through the ladder from the original 24-seed fleet SEM, PA-2D-9). The fresh
24-seed re-run will carry its own SEM, σ_new, of comparable order (the underlying fleet size and
per-seed scatter are unchanged by the repair — only which rows enter the sum). Band:
`max(3σ_comb, ε)` with `σ_comb = sqrt(0.00024951² + σ_new²)` (in LHS2 units) and **ε frozen
pre-verdict at the standard-vs-strict spread, |0.00739968 − 0.00735830| = 0.00004138** (≈0.56% of
the predicted LHS2) — this is the natural anchor because it is the largest *legitimate* ambiguity
already disclosed in the registered prediction itself, so the band cannot be narrower than the
choice of dead-row convention the primary/secondary split already discloses. **σ freeze rule: may
only tighten post-data, never widen** (house convention, `PREREGISTRATION_P3_2D_20260825.md:74-76`
pattern). **This ε anchor is a judgement call by the writer of this document, not yet
reviewer-ratified** — flagged explicitly for the pre-execution review; the house pattern (`PREREGISTRATION_
P3_2D_20260825.md` PA-2D-1) is for a pre-execution review to fix or replace it before the fleet
launches, and that review has not happened for this document.

- **CONFIRMED** — LHS2(bt) lands within `LHS2_predicted ± band` of the registered primary
  prediction. Both new mechanisms (Defect 1 mass-floor, Defect 2 dead-row) confirmed end-to-end
  as measured causes of the residual; the honestly-residual factor is x1.96 (not further reduced),
  matching the pre-registered ladder exactly.
- **REFUTED** — LHS2(bt) lands outside the band. The reweighting model in
  `p32d_residual_accounting_20260827.md` is wrong somewhere despite both mechanisms being real and
  measured in isolation; bank the refutation, restate the residual at its freshly measured value
  (not x1.96), and open a stage-0 on why the reweight mispredicted (candidate causes to hand the
  stage-0: an interaction between Defect 1 and Defect 2 not captured by treating them as
  independent multiplicative factors; a residual RNG-stream perturbation the gate in §3 did not
  catch; a fourth, still-unattributed mechanism).
- **MIXED** — a first-class branch, given a real disposition, not a placeholder: (a) LHS2(bt)
  lands strictly between the strict prediction (0.00735830) and outside `ε` of the standard one
  but inside the strict one's own band — treat as CONFIRMED-under-the-strict-convention, report
  both; OR (b) the bt arm confirms (within band) while the bc/coded arm's own residual (measured
  fresh, no frozen bc point prediction exists per §1) deviates by more than the bt arm's own band
  scaled by the bt/bc ratio observed pre-repair (2.348/1.961 ≈ 1.197) — this specific bc-arm
  disposition is the direct consequence of §1's flagged bc caveat, and its concrete trigger is:
  bc residual outside `[1.65, 2.83] × bt-band-equivalent` (a ±20% band around the pre-repair
  bc/bt ratio, itself a judgement call flagged for review). MIXED does not autoclose — it returns
  to the author with both arms' frozen numbers and the specific disposition that fired.
- **INSTRUMENT-DEFECT** — any gate in §3 fails. No CONFIRMED/REFUTED/MIXED verdict may bank on
  numbers produced under a failed gate; the run's frozen numbers are retained for diagnosis only.

**Verdict cap (author ruling, missing-anchor cap, 2026-08-15; `garden/wiki/meta/
research-cycle-amendments.md:22`; operational test at `garden/wiki/analyses/
research-cycle-core-spec.md:188`).** An audit against this cap
(`results/campaign51_20260728/realistic_20260729/anchor_cap_audit_20260827.md` §4, this same
thread) found the two NEW corrections under test here (Defect 1, Defect 2) have **no external
anchor**: both are derived from, and both this re-run's LHS2 instrument and the frozen RHS2 side
are computed by, the same banked harness (same simulator, same estimator code, same S_4D table).
**Registered here, before the run: whatever verdict this re-run reaches — CONFIRMED, REFUTED, or
MIXED — is capped at epistemic status `supported`, never `verified`.** This is a pre-registered
limitation, not a post-hoc concession. Concretely: a CONFIRMED verdict means "the reweighting
model's prediction landed inside its own band," not "the reweighting model is externally validated
physics" — the two new mechanisms remain internally consistent with the identity, not confirmed
against anything outside it. **One nuance, stated to avoid both over- and under-claiming:** the
residual-accounting document's refutation of a subagent's SEM claim (§7 item 2 of
`p32d_residual_accounting_20260827.md`) used a hard analytic bound,
`SE ≤ sqrt(m(1−m)/N) = 7.474e-4` at `m=0.014513, N=25600` — this IS a genuine external invariant
(a property of bounded random variables, independent of the simulator/estimator's own
assumptions). But it anchors a correction to a *claim about the SEM*, not the residual ladder or
the mechanisms in §2 — it does not lift the cap on this re-run's own verdict.

## 5. A10 — invariants and structural blindness

**Invariants** (each held fixed across both arms; last-audited date given, `NEVER` where not
independently re-audited this cycle):

- RHS2(twin) = 0.01451300 ± 0.00045293 — frozen, PA-2D-9, audited 2026-08-26.
- C2* = 0.06124403326364123 — `ca_rhs_work2d/p3_2d_companion_v2.json`, PA-2D-3/PA-2D-9, audited
  2026-08-26.
- Galaxy catalogue pin — md5 `c52c13b5cab61f6b3f04bbe202550969` — **NEVER independently
  re-verified by this document**; carried forward as stated in the task brief. Flagged: this
  writer could not locate this checksum in the repo's own provenance artifacts during drafting
  (see Report, ambiguity list) — treat as unaudited until the fleet driver's own provenance stamp
  confirms it.
- Seed convention — 24 seeds, 900101-900124, PA-2D-1 F14 — audited 2026-08-25.
- h grid / h_bounds — h=0.73 read, h_bounds=(0.50, 0.86) — `PREREGISTRATION_P3_2D_20260825.md`
  §5, audited 2026-08-25.
- `mass_filter_sigma="symmetric"` (production default, `[PHYSICS]` `cf4f8a2a`, row #202) — the
  A22 six-stamp set, PA-2D-4/PA-2D-6, audited 2026-08-26.
- `catalogue_numerator_survival="phi"` (1D twin production default) — PA-2D-6, audited 2026-08-26.
- `selection_in_completion_numerator="fused"` — A22 five-flag original set, F7, audited
  2026-08-25.
- The RNG stream order for the class-G latent draw (host index, per-host uniform, mass normal,
  Bernoulli uniform, fixed order per round) — asserted unchanged by the repair's own diff
  docstring; **audited by the RNG-STREAM gate in this run, not before** — this is the single
  highest-risk invariant in this document, hence the dedicated gate.

**Structural blindness** (one sentence each, at minimum the two named in the task brief):

1. This design cannot detect a defect shared by BOTH sides of the identity — the LHS's repaired
   code and the frozen RHS2 side draw from the same S_4D interpolation table and the same
   completion-class assumptions, so any error common to both cancels in the ratio and is
   invisible to CONFIRMED/REFUTED/MIXED alike.
2. It has NO external anchor (§4) — a CONFIRMED verdict certifies internal consistency between a
   corrected venue and a target law defined by the same harness, not agreement with anything
   measured or derived outside this codebase.
3. The RNG-STREAM gate (§3) can only test cases containing NO floor rows and NO dead rows —
   by construction it cannot directly observe RNG behavior on the rows the repair actually
   changes (a case with floor/dead rows necessarily diverges from the old numbers by design), so
   RNG-stream integrity on the *changed* code path is inferred from the unchanged path plus the
   diff's own draw-order claim, not independently measured on the changed rows themselves.
4. This run tests two named mechanisms (Defects 1 and 2) in combination; it cannot, by
   construction, distinguish "both mechanisms are exactly correct" from "the two mechanisms'
   errors happen to cancel to within the ε band" — a coincidental cancellation would present
   identically to CONFIRMED.

## 6. Falsifiers (A19/A14 — one per verdict category, registered before the run)

- **CONFIRMED** is falsified by: any §3 gate failing on later re-audit even after the verdict
  banks (retroactively invalidates the verdict, per the RNG-STREAM gate's structural-blindness
  caveat above); or the strict-variant secondary read landing outside its own analogous band
  while the standard variant is inside — that specific pattern would indicate the ε anchor (§4)
  was mis-set, not that the mechanisms are wrong, and returns the band (not the mechanisms) to
  review.
- **REFUTED** is falsified by: a subsequent independent re-derivation of the ladder's step-2 or
  step-3 factor (×1.1944, ×1.0680) from a route that does not touch `p3_2d_fleet.py` or
  `correspondence_1d.py` directly reproducing the frozen prediction — i.e., if a REFUTED verdict
  banks and a later free (zero-compute) re-derivation shows the arithmetic was right and only the
  fresh run's own instrument was defective, the REFUTED verdict is downgraded to
  INSTRUMENT-DEFECT and re-scored.
- **MIXED** is falsified by: a follow-up run (arm-resolved or convention-resolved) that collapses
  the ambiguity — landing cleanly on one side of the CONFIRMED/REFUTED line under the
  now-disambiguated convention.
- **INSTRUMENT-DEFECT** is falsified by: the specific gate that failed, re-run clean after a fix,
  on the SAME frozen fleet artifacts (no re-draw) if the fix is purely an accounting error, or on
  a fresh minimal re-draw if the fix touches the RNG stream itself.

## 7. Costing (A6/A17; cluster-first per row #185)

24 seeds × 2 arms (bt, bc), cluster array, ~2-4 CPU-h total — the cheapest possible re-run because
RHS2 is frozen (zero new RHS compute) and only the LHS-side fleet regenerates. No new companion
pass, no new RHS chunks. Queue-wait banked per row #185 precedent.

## Run identifiers

- **Branch:** `fix/p32d-classg-venue-repair` (uncommitted working-tree diff inspected for this
  document, `git diff --numstat` at drafting time:
  `darksiren_emri/validation/correspondence_1d.py` +51/-6,
  `results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py` +20/-7 — both still
  uncommitted; this prereg is written against the working tree, not a commit SHA, and that SHA
  is itself a placeholder-to-fill once the repair lands).
- **SLURM job id:** `[PLACEHOLDER — fill by amendment after submission]`
- **Output directory:** `[PLACEHOLDER — fill by amendment after submission, expected under
  results/campaign51_20260728/realistic_20260729/p3_2d_fleet_repair_20260827/ or equivalent —
  NOT invented, to be stated exactly as the submitted job actually writes]`

*(Committed before the fleet launches. Gates in §3 run on the FIRST completed seeds
before the full 24-seed verdict is trusted, per the house GATE-ACC/RNG-STREAM/BYTE-IDENTITY
precedent — do not wait for all 24 seeds to discover a gate failure.)*

---

## Amendments (append-only; nothing above this line may be edited after commit)

**PA-2DR-1 (2026-08-27; pre-execution adversarial review — the MIXED branch is UNREACHABLE by
construction; severity BLOCKING; supersedes §4's MIXED bullet in full).**

Both MIXED triggers are empty sets, so the branch registered as "first-class, given a real
disposition, not a placeholder" can never fire under any data.

- **MIXED(a) is a measure-zero point.** ε was defined (§4) as *exactly* `|standard − strict| =
  |0.00739968 − 0.00735830| = 0.00004138`. MIXED(a) fires on a value "strictly between the strict
  prediction (0.00735830) and outside ε of the standard one." The set of LHS2 values below
  `0.00739968 − ε` and at/above `0.00735830` is the single point `0.00735830`. Defining the band
  floor as the standard–strict gap and then defining MIXED(a) as the region *between* standard and
  strict but *outside* that same gap is self-annihilating.
- **MIXED(b) is pre-empted by ARM-COHERENCE.** MIXED(b) fires when the bc arm's residual sits
  outside a **±20%** window on the arm ratio. §3's ARM-COHERENCE gate fails on an arm-ratio move
  beyond "its own propagated SEM band", which from the PA-2D-9 frozen SEMs is ~3.4% (per-arm
  2.32% and 2.47% in quadrature; ~4.8% against the banked value). §4 gives INSTRUMENT-DEFECT
  precedence over CONFIRMED/REFUTED/MIXED. Any arm move large enough to reach ±20% has therefore
  already tripped ARM-COHERENCE and banks as INSTRUMENT-DEFECT. MIXED(b) is unreachable.

**Correction (registered before the run).** Delete the current MIXED bullet and replace with a
branch whose triggers are disjoint from the gate and from each other:

- **MIXED-CONVENTION** — LHS2(bt) lands outside the standard variant's band but inside the strict
  variant's own `max(3σ_comb, ε′)` band (or vice versa). Under PA-2DR-2's band this is itself
  currently unreachable (the two predictions differ by 0.12σ_comb), and that fact is registered
  here as a **pre-registered non-discrimination**: this run cannot separate the two dead-row
  conventions, and no verdict may claim it did.
- **MIXED-ARM** — the bt arm confirms *and* the bc arm's freshly measured LHS2(bc) lands outside
  `LHS2(bc)_pred ± max(3σ_comb,bc, ε)` where `LHS2(bc)_pred = 0.00431338 × 1.4882 = 0.00641938`
  (see PA-2DR-4), *while* ARM-COHERENCE passes. This is a non-empty region and is the only honest
  bc-arm disposition.

**Status: AUTHOR-RULING-NEEDED** (the replacement verdict map is a [RULE] on evidence the author
has not yet seen; the finding itself is FIXED-by-diagnosis).

---

**PA-2DR-2 (2026-08-27; the ε anchor is inert — it can never bind; severity MAJOR; supersedes §4's
"Anchor derivation" paragraph).**

§4 registers band `= max(3σ_comb, ε)` with `σ_comb = sqrt(0.00024951² + σ_new²)` and
`ε = 0.00004138`. The frozen component alone forces `3σ_comb ≥ 3 × 0.00024951 = 0.00074853`,
which exceeds ε by **18×**; at the stated "σ_new of comparable order" it is `3σ_comb = 0.00105858`,
exceeding ε by **25.6×**. **ε is therefore a no-op under every possible realization of the data** —
`max(3σ_comb, ε) ≡ 3σ_comb` identically. This is not a band that is "too narrow" or "too wide"; it
is a floor placed provably below the floor's own irreducible component, i.e. decoration in the
shape of the house form.

This is a direct departure from the house instantiation, not merely from the house *form*: in the
exemplar the ε floor **binds** — `ε₂ = 1.914e-3 > 3σ_comb ≈ 1.51e-3`
(`PREREGISTRATION_P3_2D_20260825.md`, PA-2D-8 item 2, verbatim: *"the verdict band = max(3σ_comb,
ε₂) is ε₂-FLOORED at this precision"*). ε there is a systematics floor that dominates the
statistical band; ε here is 25× below it. Copying the form while inverting which term binds is
exactly the vacuity failure the b0-identity heavy-tail band was caught on.

**Why the construction is also wrong in principle** (answering the drafter's own flag): a band
whose width is the gap between two variants of the *same* model measures the model's internal
convention ambiguity, not its agreement with data. It is a legitimate *lower* bound on
resolvability — "do not claim to have resolved anything finer than your own unregistered
convention choice" — but it is not an anchor, and at 0.56% of the predicted value it is far below
any systematic this venue plausibly carries.

**Correction (registered before the run).**
1. **Drop the `max(·, ε)` dressing.** Register the band as **`3σ_comb`** plainly, with
   `σ_comb = sqrt(σ_pred² + σ_new²)`, σ_pred = 0.00024951 frozen. State that ε does not bind.
2. **Retain 0.00004138 with its true role, renamed:** `δ_conv`, the *convention-resolution floor* —
   a registered statement that no claim finer than 0.00004138 in LHS2 may be made about which
   dead-row convention the data prefer. Not a verdict band.
3. **Add the discrimination statement that is currently missing** (see PA-2DR-3).

**Status: AUTHOR-RULING-NEEDED** (band replacement is a [RULE]).

---

**PA-2DR-3 (2026-08-27; NO POWER GATE — the band cannot resolve the ladder's own step 3; severity
BLOCKING; supersedes §4's CONFIRMED bullet and §5's blindness list).**

The registered band was never tested against the alternatives it must exclude. Computed here from
the prereg's own frozen numbers with `σ_comb = 0.00035286` (σ_new = σ_pred):

| alternative hypothesis | LHS2 | distance | inside 3σ_comb band? |
|---|---|---|---|
| no repair (banked) | 0.00500770 | 6.78 σ_comb | no |
| **Defect 1 only (step 3 spurious)** | **0.00692891** | **1.33 σ_comb** | **YES — INSIDE** |
| Defect 2 only (step 2 spurious) | 0.00619581 | 3.41 σ_comb | no (marginal, 0.4σ outside) |
| strict variant | 0.00735830 | 0.12 σ_comb | YES — INSIDE |

**A CONFIRMED verdict would bank identically if the ×1.0680 dead-row step were entirely spurious.**
The run as registered has no power to test step 3, and only marginal power on step 2 alone. The
house form carries a POWER GATE for exactly this (`PREREGISTRATION_P3_2D_20260825.md` F14, commit
`c71819cd` *"POWER GATE + 24 seeds"*); this document has none. Structural blindness item 4
("cannot distinguish both-correct from errors-cancelling") understates the problem: the design
cannot distinguish both-correct from **one-of-the-two-absent**.

**Correction — and it is free.** `stage_lhs2d` (`p3_2d_fleet.py:640-646`) computes the Defect-2
repair as a **pure post-hoc accumulation over an already-written diagnostics CSV**
(`meta["diagnostics_csv"]`). It consumes **zero additional cluster time and zero additional
draws**. Therefore register, before the run:

1. **Emit BOTH accumulations from the same fresh fleet:** `LHS2_D1only` (with the pre-repair `live`
   filter) and `LHS2_D1+D2` (repaired). These are **paired on the identical realization**, so their
   difference carries only the per-row recount error, not the full seed scatter — step 3 becomes
   testable at ~an order of magnitude better precision than the unpaired comparison above.
2. **Registered step-3 prediction (new, frozen now):**
   `LHS2_D1+D2 / LHS2_D1only = 1.0680`, tested against the paired per-seed SEM of that ratio.
3. **Registered step-2 prediction (new, frozen now):**
   `LHS2_D1only = 0.00692891 ± 0.00018403`, band `3σ_comb` on that quantity.
4. **CONFIRMED now requires all three** (level, step-2 leg, step-3 leg), not the level alone.
5. Add to §5 structural blindness: *"the primary band cannot exclude the Defect-1-only hypothesis;
   step 3 is tested only by the paired ratio in PA-2DR-3, and if that readout is not produced the
   run carries no evidence on step 3 at all."*

**Status: AUTHOR-RULING-NEEDED.**

---

**PA-2DR-4 (2026-08-27; the bc/coded arm — a point prediction DOES exist and is derivable at zero
compute; severity MAJOR; supersedes §1's "no primary numeric prediction registered" and the ×1.197
ratio in §4).**

§1 declines a bc point prediction and §4 then builds a verdict trigger on the unregistered ratio
`2.348/1.961 ≈ 1.197 ± 20%`. That is a placeholder wearing a band's clothes, and its arithmetic
does not match its own stated construction: the quoted interval `[1.65, 2.83]` is not ±20% of
2.348 (which is `[1.878, 2.818]`); the lower edge is −29.7%. Its units are also incoherent — a
dimensionless residual factor is compared against "`× bt-band-equivalent`", an LHS2-scale quantity.

**The asymmetry ×1.961 vs ×2.348 is not what §1 implies.** It is not an LHS-side arm asymmetry
that the repair might disturb; it is almost entirely the two arms' *different RHS values*. From the
PA-2D-9 frozen set: `RHS2(coded) = 0.01507225 ± 0.00046202`, `LHS2(bc) = 0.00431338 ± 0.00010642`,
so pre-repair `X_bc = 3.49430`, and `3.49430 / 2.348 = 1.4882` — the bc arm's own ladder factor,
within 0.7% of the bt arm's 1.47765. The corrections *are* common-mode to 0.7%, exactly as §3
asserts; §1's "differs materially by arm" is an artifact of comparing X's built on different RHS
denominators.

**Correction (registered before the run).**
1. **Register the bc point prediction:** `LHS2(bc) = 0.00431338 × 1.4882 = 0.00641938`, with band
   `3σ_comb,bc` propagated from `σ(LHS2(bc)) = 0.00010642` through the same ladder; equivalently
   `X_bc = RHS2(coded)/LHS2(bc) = 2.348`, RHS2(coded) frozen, zero new RHS compute. The bc arm
   is now a **first-class second test**, not a report-only afterthought.
2. **Delete the ×1.197 / ±20% / `[1.65, 2.83]` trigger entirely** — replaced by MIXED-ARM in
   PA-2DR-1.
3. **Fix ARM-COHERENCE's comparison target.** §3 compares the fresh `LHS2(bc)/LHS2(bt)` to the
   **pre-repair** 0.8613. The ladder's own per-arm factors predict the ratio moves to
   `0.00641938/0.00739968 = 0.8675`, +0.7%. The gate as written flags the prereg's own prediction
   as a RED FLAG. Register the comparison target as **0.8675**, and state the tolerance
   explicitly (proposed: 3× the paired per-seed SEM of the ratio, which is far tighter than the
   independent-quadrature 4.8% because the two arms share seeds, rows and hosts).

**Status: AUTHOR-RULING-NEEDED.**

---

**PA-2DR-5 (2026-08-27; the RNG-STREAM and BYTE-IDENTITY gates are UNRUNNABLE as registered, and
the disclosed blindness is NOT inherent; severity BLOCKING; supersedes §3 bullets 1-2 and §5
structural-blindness item 3).**

**(a) The registered test subset does not exist.** Both gates are scoped to "the seed/case subset
with zero floor rows and zero dead rows under the OLD code (must exist — the pre-repair fleet's own
diagnostics identify them)". That parenthetical is asserted, not verified, and the prereg's own
source numbers refute it: floor rows are **793/4800 = 16.5% of drawn latents**
(`p32d_residual_accounting_20260827.md` §3). At 200 drawn events per seed, the probability that a
seed contains zero floor rows is `(1−0.165)^200 ≈ e^{−36}`. **No such seed exists, and none will.**
As registered, both gates silently no-op and §5's highest-risk invariant ("the RNG stream order …
audited by the RNG-STREAM gate in this run") goes NEVER-audited while the document records it as
gated.

**(b) The blindness is not inherent — a decidable test already exists, in this very diff.** §5 item
3 states the gate "by construction cannot directly observe RNG behaviour on the rows the repair
actually changes." That is false. Three constructions observe it, all zero-cluster-cost:

1. **Draw-count instrumentation (directly tests the diff's own load-bearing claim).** Wrap `rng` in
   a counting proxy recording every `choice`/`normal`/`uniform` call and its `size`, and assert
   that per round the consumption is exactly `4 × batch` **on a pool engineered to contain M ≤ 0
   draws**. This tests "the M ≤ 0 rejection is an extra AND-condition, not an extra RNG draw"
   (`correspondence_1d.py` docstring) *on the changed path*, which is precisely what §5 item 3
   claims is impossible.
2. **Engineered-pool byte-identity, already written.** The uncommitted diff contains
   `darksiren_emri_test/validation/test_correspondence_1d.py` (+192/-0), which carries a verbatim
   pre-repair reference implementation
   (`_draw_2d_accepted_latents_pre_repair`) and
   `test_catalogue_selected_2d_byte_identical_to_pre_repair_when_no_floor_rows`, constructing zero
   floor rows deterministically via `M_error = np.zeros(n_pool)` and asserting
   `assert_array_equal` on `host_idx`, `z_true`, `M_true`, `M_z_true`, `s4d_at_truth`. **This is
   the BYTE-IDENTITY gate in decidable, runnable form**, and the prereg neither cites it nor
   registers it. It also carries
   `test_catalogue_selected_2d_rejects_nonpositive_mass_not_floor_clips` (pool with
   `M_error = 1.0e6` → ~50% invalid draws) — the floor-row-containing case §5 item 3 says cannot
   be built.
3. **Closed-form support check — and this is a genuine EXTERNAL anchor.** On a synthetic pool with
   `M = 0`, `M_error = 1` and `S_4D ≡ const`, the repaired venue's accepted mass sample must follow
   the **truncated normal `N(0,1) | M > 0`** in distribution (KS test). That is an analytic
   property of rejection sampling, independent of this codebase's simulator, estimator and S_4D
   table. §4 and §5 item 2 assert the run has "NO external anchor"; that is correct for the
   *residual ladder*, but **over-broad for the Defect-1 draw law**, which is externally checkable.
   Register the KS check and narrow §5 item 2 accordingly.

**Correction (registered before the run).** Replace §3 bullets 1-2 with: (i) BYTE-IDENTITY = the
existing `M_error = 0` unit test, run first, must pass `assert_array_equal`; (ii) RNG-STREAM =
draw-count instrumentation on a pool with a known invalid fraction, asserting `4 × batch` per round
and asserting the accepted-sample draw sequence for the *valid* rows matches a manual replay of
`np.random.default_rng(seed)` under the same batch schedule; (iii) SUPPORT = the truncated-normal
KS check. All three run on a dev CPU before any cluster submission. Amend §5 item 3 to state that
the residual blindness is only that these are *synthetic-pool* checks, not production-pool ones.

**Status: AUTHOR-RULING-NEEDED** (gate replacement is a [RULE]; findings (a) and (b) are FIXED by
diagnosis).

---

**PA-2DR-6 (2026-08-27; catalogue md5 pin — the drafter's "NEVER independently re-verified" note is
WITHDRAWN as a search failure, not an absence; severity MINOR; supersedes §5's galaxy-catalogue
bullet).**

§5 flags the pin as unaudited and states "this writer could not locate this checksum in the repo's
own provenance artifacts during drafting." **The pin is sound and is recorded.** Verified this
session:

```
$ md5sum darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv
c52c13b5cab61f6b3f04bbe202550969  darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv
```

matching the registered value exactly. The checksum is recorded in **twelve** files in-tree; the
**canonical location is `results/mechanism_study_20260813/PINNED_INPUTS_MANIFEST.md`**, with
corroborating stamps in `darksiren_emri/validation/correspondence_1d.py` (the venue instrument
itself), `results/campaign51_20260728/realistic_20260729/CLAIM_P3_MKER_20260826.md`,
`.../CLAIM_WGEO_20260827.md`, `.../mker_r2_measure_A.md`, `.../mker_r2_measure_B.md`,
`.../wgeo_s0_census_20260827.md`, `.../m2_residual_owner/B_READOUT_20260808.md`,
`.../m2_residual_owner/adjudicate_b_results.json`,
`.../p3_rphi_production/p3_rphi_production_result.json` and
`results/campaign51_20260728/RUNBOOK_NEXT_SESSION_35.md`.

**Correction.** Replace the §5 bullet with: *"Galaxy catalogue pin — md5
`c52c13b5cab61f6b3f04bbe202550969` — **VERIFIED 2026-08-27** against the working-tree file;
canonical record `results/mechanism_study_20260813/PINNED_INPUTS_MANIFEST.md`."* Per CLAUDE.md's
dataset-pinning rule the consumer must be **STOP-gated on mismatch**; register that the fleet
driver aborts on a checksum mismatch rather than logging it. Recorded here so the next reader does
not repeat the search.

**Status: FIXED.**

---

**PA-2DR-7 (2026-08-27; the diff description is ACCURATE, with one unregistered assumption;
severity MINOR-to-MAJOR; amends §1a/§2, adds one registered check).**

Read of the working-tree diff (`git diff HEAD`, branch `fix/p32d-classg-venue-repair`):

- **Defect 2 — VERIFIED accurate.** `_identity_inputs_2d` now computes
  `w2 = np.divide(a2, denom, out=np.zeros_like(a2), where=denom > 0.0)` over all rows, `live`/`dead`
  demoted to diagnostics, and `stage_lhs2d` sums `1.0 - inputs["w2"]` unfiltered. The 3 pathological
  `A2 = B2 = 0` rows fall to `where=False` → `w2 = 0` → summand 1. **This is the STANDARD variant**,
  matching `ca_rhs_scorer.py:1315-1324`. §1a's registered primary choice tracks the code exactly. ✓
- **Defect 1 — VERIFIED accurate in mechanism.** `_M2D_MASS_FLOOR` is deleted; `valid_mass_batch =
  m_true_batch > 0.0` is ANDed into `accept_mask`; rejection is inside the batch/round loop; RNG
  call sequence per round is unchanged (`choice`, per-host uniform, `normal`, `uniform`, all
  `size=batch`). `m_z_for_s4d = np.where(valid_mass_batch, m_z_true_batch, _M2D_OBS_M_FLOOR)` feeds
  the interpolator a safe value for rows that are rejected regardless — inert. ✓

**The unregistered assumption.** The old code clipped at `_M2D_MASS_FLOOR = 1.0` M_sun, so the
"floor population" from which the ×1.1944 factor was measured is `{M_drawn < 1.0 M_sun}`. The
repair rejects `{M_drawn ≤ 0}`. These are different sets, and the ×1.1944 prediction transfers only
if `{0 < M < 1.0 M_sun}` is empty to within rounding. It plausibly is (host σ_M is O(10⁶) M_sun, so
a 1-M_sun-wide interval carries ~10⁻⁶ of the mass; consistent with the measured 16.5% ≈ the
`P(M ≤ 0)` of a `N(m, ~m)` draw), **but this is nowhere registered.**

**Correction.** Register as a cheap acceptance check on the fresh fleet: **count accepted latents
with `0 < M_true < 1.0 M_sun`; if that count exceeds ~0.1% of accepted rows, the ×1.1944 factor
does not apply to the repaired venue and the primary prediction must be re-derived before any
verdict banks.**

**Status: OPEN** (registered check to run; the diff-description verification itself is FIXED).

---

**PA-2DR-8 (2026-08-27; provenance — the registered diff manifest is incomplete and the run is
registered against a mutable working tree; severity MAJOR; supersedes "Run identifiers").**

1. **A third changed file is omitted.** The registered `git diff --numstat` lists two files. The
   actual working tree carries three: `darksiren_emri/validation/correspondence_1d.py` +51/-6,
   `results/.../p3_2d_fleet.py` +20/-7, **and
   `darksiren_emri_test/validation/test_correspondence_1d.py` +192/-0**. The omitted file is not
   incidental — it contains the pre-repair reference implementation and the byte-identity pin that
   PA-2DR-5 recommends adopting as the gate.
2. **`git status` under-reports.** `git status --short` and a bare `git diff` do **not** show
   `correspondence_1d.py` as modified (only `git diff HEAD` does), which is how a drafter can lose
   a file from a manifest. Whatever index state causes this must be cleared before the branch is
   committed, or the physics change can be committed silently incomplete.
3. **No frozen ref.** The prereg registers against an uncommitted working tree held by another
   agent, with the SHA a self-declared placeholder. A pre-registration whose instrument can change
   under it is not falsifiable.

**Correction (registered before the run).** Commit the branch (`/physics-change` gate applies —
`correspondence_1d.py` is a validation instrument, but `_draw_2d_accepted_latents` is the venue's
draw law, so a `docs/gates/PHYSICS-GATE-LEDGER.md` row is required either way), stamp the SHA into
Run identifiers by amendment, restate the three-file numstat, and only then submit.

**Status: OPEN.**

---

**PA-2DR-9 (2026-08-27; RULE-1 — no exoneration check is present; the check PASSES on substance but
must be written; severity MAJOR; adds a new §0 to the prereg).**

The document carries **no rule-1 / exoneration check**, and both mechanisms under test sit directly
on top of standing-exoneration vocabulary. This matters acutely here: `EXONERATION_REGISTER_
20260827.md` §"ADVERSARIAL CHECK" §C item 2 names **`CLAIM_P3_2D_20260825.md` — this thread's own
claim card — as one of three cards that "carry NO rule-1 / exoneration check at all"**. This prereg
repeats that omission one day after the register was written to stop it.

Every entry of the register was read (§1 layer-1, §2's 17 layer-2 items, §3 caveats, §4 confirmed
defects, §8 void list — the binding set is the union per §9). Result — **substantive PASS, four
adjacent entries, none reopened:**

- **HB — "hard mass window as support truncation"** (`## HB`, register §1). Its own synonym list
  contains *"mass floor"*, *"hard clamp on mass"*, *"truncation bias"*, *"sigma clipping on host
  mass"* — a near-verbatim match to Defect 1's vocabulary. **Not reopened:** HB is scoped to the
  **estimator's candidate-search mass window** in `handler.py` (production call site
  `bayesian_statistics.py:4691`), measured as window presence-vs-absence against the **H0 MAP /
  catalogue-leg nat budget**. Defect 1 is a floor in the **venue's own latent draw law**
  (`correspondence_1d.py:_draw_2d_accepted_latents`), a simulator construct with no candidate
  search in it, measured against the **LHS2/RHS2 identity residual**, not h. Different file,
  different stage, different variable, different estimand — the delimitation form
  `CLAIM_D1_P0WINDOW_20260805.md:76-79` uses.
- **[LNM-DRAW] — "the ln-mass draw itself"** (register §1). **Not reopened:** that entry exonerates
  the *estimator's* log-mass draw as a 2D-bias driver at `|Δln M| ≤ 0.0009` dex. Defect 1 is a
  support defect in the *venue's* mass draw, and is a correctness-class finding about the identity,
  not a bias-driver claim about h. Its own "WHAT IT DOES NOT COVER" field explicitly reserves
  downstream/other mass objects.
- **HC — "mixture-floor / zero-handling"** (register §1). Defect 2 is a zero/dead-row handling
  convention. **Not reopened:** HC's delimitation is explicit — *"a DIFFERENT harness's zero-handling
  … is NOT covered by HC — do not treat HC as blanket clearance for all zero-handling code"*.
  `p3_2d_fleet.py` is a different harness from the production combination code HC measured.
- **[WBHZERO-ASYMMETRY]** (register §4, a confirmed defect, not an exoneration). Engaged only as
  the §5 invariant `mass_filter_sigma="symmetric"` (`[PHYSICS]` `cf4f8a2a`, row #202). Not
  disturbed.

**Correction.** Insert the four bullets above as a §0 "Rule-1 exoneration check (two-layer, union
of register §1 ∪ §2 ∪ §3 ∪ §4 ∪ §8)" before §1. Cite entries by TAG heading, not line number, per
the register's §9 maintenance rule.

**Status: FIXED** (check performed and recorded here; transcription into §0 is the remaining
mechanical step).

---

**PA-2DR-10 (2026-08-27; arithmetic and error-propagation audit of the registered prediction;
severity MINOR; no correction to the primary number).**

Re-derived independently from `p32d_residual_accounting_20260827.md` §1.

- **Ladder arithmetic: CORRECT.** `0.00500770 × 1.1585 = 0.00580142` (quoted 0.00580132, implied
  factor 1.158535); `× 1.1944 = 0.00692910` (quoted 0.00692891, implied 1.194367);
  `× 1.0680 = 0.00740008` (quoted 0.00739968, implied 1.067942). Compounded
  `0.00739968/0.00500770 = 1.47766`, matching the document's stated total 1.4776. ✓
- **X and its error bar: CONSISTENT.** `X = 0.01451300/0.00739968 = 1.96130`. Relative errors
  3.1210% (RHS2) and 3.3719% (LHS2) in quadrature give 4.5946%, so
  `σ_X = 1.96130 × 0.045946 = 0.09011` → the quoted **±0.090 is exactly right**. The same check
  reproduces ±0.113, ±0.101 and ±0.086 at ladder steps 0-2. **The test is not made unfalsifiable by
  a mis-stated error bar.** ✓
- **Two unexplained items, recorded not corrected.** (i) §2 of the source quotes the step-1 factor
  as `×1.15735 ± 0.00678` while the ladder table uses `×1.1585` — a 0.1% inconsistency, immaterial
  to the verdict. (ii) The SEM inflation at each rung is not derivable from the stated factor
  uncertainties: step 3's relative SEM jumps 2.656% → 3.372%, requiring an additional 2.08%
  contribution from a correction (`×1.0680`) that is a *deterministic recount of 40 dead rows on a
  frozen sample*. The inflation is **conservative** (it widens the band), so it cannot manufacture
  a false CONFIRMED — but it is the direct cause of the step-3 power failure in PA-2DR-3. If the
  step-3 SEM were propagated honestly, `σ_pred` would fall to ≈0.000197 and the paired ratio test
  would gain further precision.

**Status: OPEN** (the SEM-inflation provenance should be stated before the run; the primary number
and its ±0.090 stand as registered).

---

**PA-2DR-11 (2026-08-27; §4's missing-anchor cap and §5's A10 lists — assessed; severity MINOR).**

- **The missing-anchor cap is correctly registered**, not hedged: it is stated *before* the run,
  names its authority (`garden/wiki/meta/research-cycle-amendments.md:22`, 2026-08-15) and its
  audit (`anchor_cap_audit_20260827.md` §4), binds all three verdict categories, and gives the
  concrete reading of a CONFIRMED verdict. The `sqrt(m(1−m)/N)` nuance correctly distinguishes an
  external invariant that anchors a *SEM claim* from one that would anchor the *ladder*. **This
  section passes the review unchanged**, with one narrowing: per PA-2DR-5(b)3 the Defect-1 **draw
  law** does have an available external anchor (truncated-normal support), so §5 blindness item 2's
  "NO external anchor" should read "no external anchor for the residual ladder or the identity's
  target law."
- **§5 carries both mandatory A10 lists** (invariants with last-audited dates and explicit `NEVER`
  markers; structural blindness, four items). The invariants list is well-formed. **The blindness
  list is incomplete**: add (5) the band's inability to exclude the Defect-1-only hypothesis
  (PA-2DR-3); (6) the fresh fleet reuses seeds 900101-900124 but the repair changes RNG consumption
  on floor-containing rows, so the fresh realizations are **not paired** with the banked ones — the
  old-vs-new comparison necessarily carries full independent seed scatter, and only the D1-only vs
  D1+D2 comparison within the new run is paired; (7) item 3 is materially wrong as written and is
  replaced by PA-2DR-5.

**Status: FIXED** (assessment recorded; the three added blindness items are the correction).

---

## REGISTERED DESIGN v2 [OPUS-ORCH 2026-08-27]

**Status: this block is the OPERATIVE registered design.** It applies the pre-execution review's
amendments PA-2DR-1..11 plus one BLOCKER found after them (the two-vs-three-rung ladder, §v2.1),
and states the corrected design in full. Written before the fleet is submitted; no number below was
chosen after seeing data. Every band is derived with its arithmetic shown.

### v2.0 What this supersedes

| v1 location | status under v2 | why |
|---|---|---|
| §1a registered primary prediction `LHS2(bt) = 0.00739968 ± 0.00024951`, `X = 1.961 ± 0.090` | **SUPERSEDED** by §v2.2 | that is the THREE-rung ladder; the branch implements two rungs (§v2.1) |
| §1's "bc/coded arm — no primary numeric prediction registered" | **SUPERSEDED** by §v2.4 | PA-2DR-4; a bc point prediction is derivable at zero compute |
| §3 RNG-STREAM and BYTE-IDENTITY gate definitions | **SUPERSEDED** by §v2.6 gates G1–G3 | PA-2DR-5: the registered test subset is empty (0 of 24 seeds) |
| §3 ARM-COHERENCE comparison target `0.8613` and its unstated tolerance | **SUPERSEDED** by §v2.6 gate G4 | PA-2DR-4 item 3, further corrected by §v2.1 |
| §4 "Anchor derivation" paragraph (`max(3σ_comb, ε)`, ε as a band component) | **SUPERSEDED** by §v2.3 | PA-2DR-2: ε is provably inert; retained only as `δ_conv` |
| §4 CONFIRMED bullet (level alone) | **SUPERSEDED** by §v2.5 | PA-2DR-3: no power gate; CONFIRMED now requires three legs |
| §4 MIXED bullet (a) and (b) | **SUPERSEDED** by §v2.5 MIXED-CONVENTION / MIXED-ARM | PA-2DR-1: both v1 triggers are empty sets |
| §5 galaxy-catalogue "NEVER re-verified" bullet | **SUPERSEDED** by PA-2DR-6 (VERIFIED 2026-08-27) | checksum located and matched |
| §5 structural blindness item 3 | **SUPERSEDED** by §v2.8 item 3 | PA-2DR-5(b): the blindness is not inherent |
| §5 structural blindness list (4 items) | **EXTENDED** to 8 items, §v2.8 | PA-2DR-3, PA-2DR-11, §v2.1 |
| §4 missing-anchor cap | **RETAINED, narrowed** (§v2.7) | PA-2DR-11: the Defect-1 draw law *is* externally anchored |
| "Run identifiers" | **SUPERSEDED** by §v2.10 | PA-2DR-8: four changed files, and a frozen SHA |
| §1b ladder table, §2 arms/instruments, §6 falsifiers, §7 costing | **retained as written** | unaffected |

`§4`'s ε is retired as a band component; `§1a`'s strict-variant secondary read is retained but
rescaled (§v2.3). Everything not listed above stands.

### v2.1 BLOCKER — the branch implements TWO of the ladder's THREE rungs

The v1 primary prediction is the compounded three-rung value
`0.00500770 × 1.1585 × 1.1944 × 1.0680 = 0.00740040`. **Rung 1 (the S̄_φ(z) double-application,
×1.1585) is NOT implemented on `fix/p32d-classg-venue-repair` and cannot be**: its own
physics-change package (`PHYSICS_CHANGE_SBARPHI_20260827.md`) returned **FIX-MISSPECIFIED**, and
the "author grant" for it was found this session to exist only as orchestrator narration, never a
verbatim author ruling. The branch's diff (§v2.10) touches only `_draw_2d_accepted_latents`'s mass
support (rung 2) and `stage_lhs2d`'s dead-row convention (rung 3).

Verified independently here:

```
2-rung: 0.00500770 × 1.1944 × 1.0680 = 0.00638792
3-rung: 0.00500770 × 1.1585 × 1.1944 × 1.0680 = 0.00740040
ratio  = 0.00740040 / 0.00638792 = 1.15850   <- exactly the omitted rung-1 factor 1.1585
gap    = 0.00101248 = 4.058 σ  on the v1 σ_pred = 0.00024951
```

Left uncorrected, the run would land near 0.006388, miss the v1 band by ~4σ, and bank **REFUTED** —
a false negative on a model that is not wrong, only partly applied. This is exactly the failure a
pre-registration exists to prevent, and it is registered here **before** the run.

**This finding overrides anything earlier in this document, and anything in PA-2DR-1..11 that was
computed on the three-rung ladder.** Where an amendment's number and this block's number disagree,
this block's derivation wins and the disagreement is stated explicitly (see §v2.4 and §v2.6 G4).

### v2.2 PRIMARY READS AND THEIR PREDICTIONS (registered now, before the run)

Two primary reads, **paired on the identical realization** — `stage_lhs2d` is pure post-processing
of an already-written diagnostics CSV, so emitting both costs zero cluster time and zero extra
draws (PA-2DR-3). Both are now produced by the driver (`p3_2d_fleet.py:stage_lhs2d`).

| read | what it is | registered prediction |
|---|---|---|
| **P1 · `LHS2_D1+D2` (bt)** | repaired venue, unfiltered dead-row sum — the level | **0.00638792 ± 0.00020476** |
| **P2 · `LHS2_D1only` (bt)** | repaired venue, PRE-repair `live` filter restored — rung 2 alone | **0.00598120 ± 0.00014679** |
| **P3 · paired ratio** `LHS2_D1+D2 / LHS2_D1only` | the discriminator for Defect 2 specifically | **1.0680** (alternative 1.1019, see §v2.5) |
| **P4 · `LHS2_D1+D2` (bc)** | the coded arm, now first-class (PA-2DR-4) | **0.00550223 ± 0.00018166** |
| **P5 · exact dead-row identity** | `LHS2_D1+D2 − LHS2_D1only ≡ (C2*/n_drawn) · N_dead` per seed | equality to float round-off, **zero free parameters** |

Frozen, zero new compute: `RHS2(twin) = 0.01451300 ± 0.00045293`,
`RHS2(coded) = 0.01507225 ± 0.00046202`, `C2* = 0.06124403326364123` (PA-2D-9/PA-2D-3).
Implied `X_bt = 0.01451300 / 0.00638792 = 2.2719 ± 0.102`
(rel: `sqrt(3.1210² + 3.2054²)% = 4.4740%`; `2.2719 × 0.044740 = 0.10164`).

**σ derivation (not carried across unchanged).** The v1 σ_pred = 0.00024951 was propagated for the
three-rung ladder and must not be reused. Decompose the ladder's own quoted relative SEMs
(PA-2DR-10 reproduces each rung's `σ_X`; subtracting RHS2's 3.1210% in quadrature isolates the LHS
side):

```
LHS rel SEM, rung by rung:  2.3370%  ->  2.5606%  ->  2.6681%  ->  3.3719%
                            (banked)    (rung 1)     (rung 2)     (rung 3)
per-rung contribution in quadrature:
  rung 1  sqrt(2.5606² − 2.3370²) = 1.0465%
  rung 2  sqrt(2.6681² − 2.5606²) = 0.7496%
  rung 3  sqrt(3.3719² − 2.6681²) = 2.0618%

2-rung (drop rung 1):  sqrt(2.3370² + 0.7496² + 2.0618%²) = 3.2054%
  σ_pred(P1) = 0.00638792 × 0.032054 = 0.00020476
rung-2 only:           sqrt(2.3370² + 0.7496²)            = 2.4543%
  σ_pred(P2) = 0.00598120 × 0.024543 = 0.00014679
```

Disclosed, per PA-2DR-10 item (ii): rung 3's 2.0618% inflation is **not derivable** from a
deterministic recount of dead rows on a frozen sample. Its provenance is unexplained. It is
**conservative** (it widens the band, so it cannot manufacture a false CONFIRMED), and it is
carried as-is rather than re-derived downward, because tightening it post-hoc without a stated
derivation would be tuning. If it were propagated honestly the bands below would tighten, which
the freeze rule permits and this block invites.

### v2.3 BANDS — derived, with the freeze rule

**Band form: `3σ_comb` plainly**, `σ_comb = sqrt(σ_pred² + σ_new²)`, σ_pred frozen above and σ_new
the fresh 24-seed SEM of the same statistic. Per PA-2DR-2 the `max(·, ε)` dressing is **dropped**:
`3σ_comb ≥ 3 × 0.00020476 = 0.00061428` from the frozen component alone, versus ε = 0.00004138 —
a factor 14.8, so `max(3σ_comb, ε) ≡ 3σ_comb` under every realization. A floor provably below the
floor's own irreducible component is decoration, not a band.

**δ_conv — retained with its true role, renamed.** `0.00004138` was the standard-vs-strict dead-row
convention spread on the three-rung value; rescaled to the two-rung value it is
`0.00004138 × 0.00638792 / 0.00739968 = 0.00003572`. It is registered **not as a band** but as the
**convention-resolution floor**: no claim finer than 0.00003572 in LHS2 may be made about which
dead-row convention the data prefer. At 0.1234 σ_comb it is nowhere near resolvable — see the
pre-registered non-discrimination in §v2.5.

**Planning values** (σ_new set equal to σ_pred, "of comparable order", §4; the operative band uses
the realized σ_new):

```
P1  σ_comb = sqrt(2) × 0.00020476 = 0.00028957   band 3σ = 0.00086872
    interval  0.00638792 ± 0.00086872 = [0.00551920, 0.00725664]
P2  σ_comb = sqrt(2) × 0.00014679 = 0.00020759   band 3σ = 0.00062278
    interval  0.00598120 ± 0.00062278 = [0.00535842, 0.00660398]
P4  σ_comb = sqrt(2) × 0.00018166 = 0.00025691   band 3σ = 0.00077072
    interval  0.00550223 ± 0.00077072 = [0.00473151, 0.00627295]
```

**FREEZE RULE (house convention, `PREREGISTRATION_P3_2D_20260825.md:74-76` pattern).** Every band
above **may only TIGHTEN post-data, never widen.** If a realized σ_new exceeds its planning value,
the band does **not** grow: the affected read banks as **UNDERPOWERED** (a registered disposition,
§v2.5), not as a pass.

### v2.4 POWER — what each band can and cannot exclude

Computed against P1's planning σ_comb = 0.00028957. This table is the power gate v1 lacked
(PA-2DR-3); it is registered before the run and it is the reason P2/P3 exist at all.

| alternative hypothesis | LHS2(bt) | distance from P1 | inside P1's band? |
|---|---|---|---|
| no repair (banked) | 0.00500770 | 4.766 σ | no |
| Defect-2 only, 2-rung (`× 1.0680`) | 0.00534822 | 3.590 σ | no |
| Defect-2 only, as measured on the banked fleet | 0.00551807 | 3.004 σ | no (marginal) |
| **Defect-1 only (rung 3 spurious)** | **0.00598120** | **1.405 σ** | **YES — INSIDE** |
| full 3-rung ladder (rung 1 also applied) | 0.00740040 | 3.496 σ | no |

**P1 alone still cannot exclude the Defect-1-only hypothesis** — the level read has no power on
Defect 2, exactly as PA-2DR-3 found. That is why **P3, the paired ratio, is the registered
discriminator for Defect 2**, and why CONFIRMED requires it (§v2.5). Two things the table does
buy, which v1 did not have: P1 now **separates the 2-rung from the 3-rung ladder at 3.50σ**, so the
§v2.1 blocker is itself testable rather than merely asserted; and P2 separates the repaired venue
from no-repair at `(0.00598120 − 0.00500770)/0.00020759 = 4.689 σ`, so **rung 2 is independently
powered**.

**P3 — the Defect-2 discriminator, derived.** On a dead row `A2 = 0 ⟹ w2 = 0 ⟹ summand exactly 1`,
so the paired per-seed difference is an **exact deterministic recount**, not an estimate:

```
Δ_s = LHS2_D1+D2,s − LHS2_D1only,s = (C2*/n_drawn) · N_dead,s        [P5, exact]
```

The seed scatter therefore cancels entirely and the only error is scatter in the dead-row COUNT.
From the banked fleet's own 40 dead rows over 24 seeds (mean 1.6667/seed; reproduced here from the
reviewer's D2-only read: `(0.00551807 − 0.00500770) × 200 / 0.06124403 × 24 = 40.00`), and taking
the count as Poisson:

```
sd(N_dead)      = sqrt(1.6667)          = 1.2910 per seed
SEM(N_dead)     = 1.2910 / sqrt(24)     = 0.26352
SEM(Δ)          = 0.06124403 × 0.26352 / 200 = 8.0696e-05
SEM(R)          = 8.0696e-05 / 0.00598120     = 0.013492
band 3σ_paired  = 0.040475
registered interval: 1.0680 ± 0.040475 = [1.02753, 1.10847]
```

**Power of P3: `R = 1.0000` — "Defect 2 is entirely spurious" — sits 5.04 σ_paired outside the
band.** The design can now distinguish "both mechanisms real" from "only the mass-floor mechanism
real". That was the whole point of PA-2DR-3, and it cost nothing.

**Registered honestly, not tuned: P3 has TWO candidate predictions and cannot separate them.**
The ladder's frozen rung-3 factor is 1.0680 (measured on a rung-2-repaired sample). Re-measured on
the BANKED, still-floor-contaminated fleet the dead-row factor is `0.00551807/0.00500770 = 1.1019`.
The two differ by 2.51 σ_paired and **both lie inside the registered band** — so a landing anywhere
in `[1.02753, 1.10847]` confirms Defect 2 is real but does **not** adjudicate which sample the
1.0680 was measured on. Registered as a **pre-registered non-discrimination**; no verdict may claim
to have resolved it. P5 (the exact identity) is the arbiter of *arithmetic*, not of this.

**FREEZE RULE for P3:** the operative band is `3 × SEM_paired(realized over the 24 seeds)`, **capped
at the 0.040475 planning value above — it may tighten, never widen.** A realized paired SEM larger
than 0.013492 banks **UNDERPOWERED-ON-STEP-3**, not a pass.

**bc arm (P4), derived — PA-2DR-4 applied and corrected.** Both defects are arm-independent by
their own code citations (neither `_draw_2d_accepted_latents` nor `_identity_inputs_2d` reads
`catalogue_numerator_survival_2d`), so the same rung factors transfer:

```
route A (mechanism):   0.00431338 × 1.1944 × 1.0680 = 0.00550223   <- REGISTERED
route B (bc own total factor 1.4882, de-runged):
                       0.00431338 × (1.4882 / 1.1585) = 0.00554093
spread = 0.7%  -- the same 0.7% common-mode agreement PA-2DR-4 documented
σ_pred(P4): sqrt(2.4672² + 0.7496² + 2.0618²)% = 3.3015%;  0.00550223 × 0.033015 = 0.00018166
```

Route A is registered as primary because it applies the measured code-level factors directly;
route B is recorded, and the 0.7% spread is a **disclosed route ambiguity on P4's central value**,
not a band component. This supersedes PA-2DR-4's `0.00641938`, which is the THREE-rung bc value.

**Correcting the record on the "arm asymmetry" (PA-2DR-4, ratified here).** The previously reported
`×1.961` (bt) vs `×2.348` (bc) was presented as if it were a physical difference between the arms
that the repair might disturb. **It is not.** It is an artifact of the two X's being built on
different RHS denominators: `X_bc = 0.01507225/0.00431338 = 3.49430`, and
`3.49430 / 2.348 = 1.4882` against bt's own total 1.47765 — the two arms' ladder factors agree to
**0.7%**, i.e. the corrections are common-mode, exactly as §3 asserts. §1's "differs materially by
arm" is withdrawn. The record is corrected here rather than silently reconciled.

### v2.5 VERDICT MAP

- **CONFIRMED** — requires **all three legs**, not the level alone (PA-2DR-3):
  1. P1 inside its band, **and**
  2. P2 inside its band (rung 2 leg), **and**
  3. P3 inside its band (rung 3 leg, the Defect-2 discriminator), **and**
  4. P5 exact for every seed, **and** every gate in §v2.6 passing.
  Reading: "the two-rung reweighting model's predictions landed inside their own bands." Capped at
  epistemic status `supported`, never `verified` (§v2.7). It does **not** license any claim about
  rung 1, which is untested here.
- **REFUTED** — P1 outside its band. Bank it, restate the residual at its freshly measured value,
  and open a stage-0. Candidate causes to hand the stage-0: a Defect-1 × Defect-2 interaction not
  captured by treating the rungs as independent multiplicative factors; a residual RNG-stream
  perturbation §v2.6 did not catch; the PA-2DR-7 mass-window discrepancy (§v2.6 G5) invalidating
  the ×1.1944 transfer; a fourth unattributed mechanism.
- **MIXED-CONVENTION** — P1 outside the standard variant's band but inside the strict variant's.
  **Pre-registered as UNREACHABLE**: the two conventions differ by δ_conv = 0.00003572 = 0.123
  σ_comb. This run **cannot** separate the two dead-row conventions and no verdict may claim it
  did. Registered so the non-discrimination is on the record, not so the branch can fire.
- **MIXED-ARM** — P1 and P3 confirm **and** P4 lands outside its band **while** G4 (ARM-COHERENCE)
  passes. Non-empty, and the only honest bc-arm disposition. Returns to the author with both arms'
  frozen numbers; does not autoclose.
- **UNDERPOWERED** — any read whose realized σ exceeds its planning value (freeze rule, §v2.3/§v2.4).
  Not a pass and not a refutation: the band does not widen to accommodate it. Report the realized
  numbers and return.
- **INSTRUMENT-DEFECT** — any gate in §v2.6 fails. Takes precedence over all of the above. No
  verdict banks on numbers produced under a failed gate.

### v2.6 GATES

The C-A stack verbatim (ACC re-closed-looped, M2-LINK, PA-CA-11 out-root guard, A22 resolved-flag
stamps), **plus**:

- **G1 · BYTE-IDENTITY — replaced with a runnable form, and it already exists.** v1 scoped this to
  "a seed with zero floor rows and zero dead rows". **No such seed exists**: floor rows run 26–43
  per seed across all 24 banked bt seeds (0 of 24 qualify), and every seed carries dead rows;
  independently, at 16.5% floor-row incidence the probability of a zero-floor-row seed is
  `(1−0.165)^200 ≈ e^{−36}`. As registered the gate silently no-ops. **Replacement, already written
  in the diff and hereby registered as the gate:**
  `darksiren_emri_test/validation/test_correspondence_1d.py::test_catalogue_selected_2d_byte_identical_to_pre_repair_when_no_floor_rows`
  — engineers zero floor rows deterministically via `M_error = 0` and pins the repaired draw
  against a verbatim pre-repair reference with `assert_array_equal` on `host_idx`, `z_true`,
  `M_true`, `M_z_true`, `s4d_at_truth`. **Reconciliation:** v1 §3 states this test needs writing;
  it was in the working tree at drafting time and the drafter did not cite it. Run first. FAIL =
  INSTRUMENT-DEFECT.
- **G2 · RNG-STREAM — replaced with draw-count instrumentation on the CHANGED path.**
  `...::test_catalogue_selected_2d_rng_consumption_is_4x_batch_on_the_changed_path` wraps the
  generator in a counting proxy and asserts, on a pool engineered so ~50% of mass draws are
  invalid, that each round consumes **exactly 4 × batch** variates in the shape
  `choice(batch) → batch×uniform(1) → normal(batch) → uniform(batch)`, invariant across rounds and
  byte-identical to the pre-repair reference in round 1. This observes RNG behaviour **on the rows
  the repair actually changes** — which v1 §5 item 3 declared impossible by construction. FAIL =
  INSTRUMENT-DEFECT.
  **Recording the correct safety argument, which is subtler than the implementer's.** The diff
  claims the M≤0 rejection is "an extra AND-condition, not an extra RNG draw"; G2 confirms that
  *within* a round. But the real risk was never within-round: it is the **ROUND COUNT**. Fewer
  acceptances change `remaining`, hence `batch = clip(4·remaining, 64, 4000)` in later rounds,
  hence the entire downstream stream. That was closed empirically from the banked fleet
  (candidate-level `E[S] = 0.5354`, leaving ample headroom in `_M2D_BATCH_MULTIPLIER`'s 4×-remaining
  sizing). **The record rests on that argument, not on the weaker within-round claim.** The
  consequence is registered as blindness item 6 (§v2.8): fresh realizations are NOT paired with the
  banked ones.
- **G3 · SUPPORT — the truncated-normal check, and the design's only external anchor.**
  `...::test_catalogue_selected_2d_accepted_mass_follows_truncated_normal`. On a synthetic pool with
  one shared `(M, M_error)` and a constant `S_4D`, the accepted masses must follow
  `N(μ, σ) | M > 0` by a one-sample KS test at α = 1e-3. This is a property of rejection sampling,
  independent of this codebase's simulator, estimator and S_4D table. **Non-vacuity verified before
  registration:** the pre-repair floor-clip fails it at `D = 0.1103, p = 1.2e-21`, with 218/2000
  accepted masses piled at exactly the old 1.0 M_sun floor; the repaired code passes. FAIL =
  INSTRUMENT-DEFECT.
- **G4 · ARM-COHERENCE — target corrected, tolerance registered.** Both defects are common-mode, so
  applying identical rung factors to both arms leaves the arm ratio **exactly invariant**:
  `0.00550223/0.00638792 = 0.86135`, which is precisely the pre-repair `0.00431338/0.00500770 =
  0.86135`. Route B (§v2.4) instead gives 0.8675, +0.7%. **Registered target: the interval
  [0.8613, 0.8675]**, spanning both routes — a single point would flag one of the design's own
  routes as a defect, which is the bug v1 had (it compared the fresh ratio to 0.8613 while the
  ladder predicted a move). **This supersedes PA-2DR-4 item 3's single point 0.8675**; that value
  was derived on the three-rung ladder and via route B only, and the correction wins per §v2.1.
  Tolerance: `3 × SEM_paired(ratio over the 24 shared seeds)` outside the target interval, capped
  at the independent-quadrature upper bound `sqrt(2 × (2.32² + 2.47²))% = 4.792%` relative — a
  strict upper bound, since the arms share seeds, rows and hosts and pairing can only reduce it.
  FAIL (ratio outside interval ± tolerance) = INSTRUMENT-DEFECT, diagnosed before any verdict banks.
- **G5 · PA-2DR-7 MASS-WINDOW COUNT — the unregistered assumption, now measured.** The ×1.1944
  factor was measured on the OLD floor population `{M_drawn < 1.0 M_sun}`; the repair rejects
  `{M_drawn ≤ 0}`. **These are different sets**, and ×1.1944 transfers only if `(0, 1.0) M_sun` is
  empty to within rounding. Registered check: count accepted latents with `0 < M_true < 1.0 M_sun`
  from the per-seed CRB CSV's `M_true` provenance column (zero compute).
  **If that count exceeds 0.1% of accepted rows, ×1.1944 does not apply to the repaired venue and
  P1/P2 must be re-derived before any verdict banks.** Emitted as `pa2dr7_fraction` / `pa2dr7_ok`.
- **G6 · P5 EXACT DEAD-ROW IDENTITY.** `LHS2_D1+D2,s − LHS2_D1only,s == (C2*/n_drawn)·N_dead,s` for
  every seed, to float round-off. Zero free parameters. Emitted as `dead_row_identity_all_ok`. A
  failure here is an accounting defect in the driver, not a physics result. FAIL =
  INSTRUMENT-DEFECT.

G1–G3 run on a dev CPU **before** any cluster submission. G4–G6 run on the first completed seeds,
not after all 24 (house GATE-ACC precedent, §7).

### v2.7 Missing-anchor cap — retained, narrowed

§4's cap stands verbatim: **whatever verdict this re-run reaches is capped at epistemic status
`supported`, never `verified`.** Narrowing per PA-2DR-11: §5 blindness item 2's blanket "NO
external anchor" is corrected to **"no external anchor for the residual ladder or the identity's
target law"** — the Defect-1 **draw law** does have one, and it is now gated (G3). The cap is
unchanged for everything else.

### v2.8 Structural blindness (§5's list, extended to eight)

1. Cannot detect a defect shared by BOTH sides of the identity (unchanged).
2. No external anchor **for the residual ladder or the identity's target law** (narrowed, §v2.7).
3. **REPLACED.** v1 item 3 ("the gate cannot observe RNG behaviour on the changed rows") is
   materially wrong and is withdrawn — G2/G3 observe exactly that. The residual blindness is only
   that G1–G3 are **synthetic-pool** checks, not production-pool ones.
4. Cannot distinguish "both mechanisms exactly correct" from "their errors cancel within the band".
5. **P1 alone cannot exclude the Defect-1-only hypothesis (1.405 σ).** Rung 3 is tested ONLY by P3;
   if P3 is not produced, the run carries **no evidence on Defect 2 at all** (PA-2DR-3).
6. The fresh fleet reuses seeds 900101–900124, but the repair changes the round count on
   floor-containing rows (§v2.6 G2), so **fresh realizations are NOT paired with the banked ones**.
   Old-vs-new comparisons carry full independent seed scatter; only P1-vs-P2 within the new run is
   paired.
7. **Rung 1 is untested.** This run says nothing about the S̄_φ(z) double-application, and a
   CONFIRMED verdict must not be read as evidence for or against it (§v2.1, §v2.9).
8. **P3 cannot separate its own two candidate predictions** (1.0680 vs 1.1019, 2.51 σ_paired apart,
   both inside the band) — §v2.4.

### v2.9 CONDITIONAL prediction — the full three-rung ladder, retained not deleted

Registered here so a reader sees that the ladder has three rungs and only two are under test:

> **If and only if** the S̄_φ(z) repair (rung 1, ×1.1585) is separately authorised by a verbatim
> author ruling **and** its physics-change package is corrected from FIX-MISSPECIFIED to a
> specified fix, the predicted value becomes **LHS2(bt) = 0.00740040 ± 0.00024951, X = 1.9611 ±
> 0.090** — the v1 primary. That prediction is **not under test in this run** and no verdict here
> may be read as bearing on it. It is separated from P1 by 3.50 σ_comb (§v2.4), so a future
> three-rung run is distinguishable from this one.

### v2.10 Run identifiers (supersedes v1 "Run identifiers"; PA-2DR-8)

**FOUR changed files, not two or three** — v1 listed two, PA-2DR-8 found a third, and a fourth
(untracked) exists:

```
darksiren_emri/validation/correspondence_1d.py                 +51/-6    (Defect 1)
results/.../p3_2d_fleet.py                                     +20/-7    (Defect 2)  [+ v2 paired reads]
darksiren_emri_test/validation/test_correspondence_1d.py       +192/-0   (G1, + v2 G2/G3)
results/.../test_p3_2d_fleet_defect2.py                        NEW, untracked
```

- **Branch:** `fix/p32d-classg-venue-repair`
- **Frozen commit SHA:** `[AMENDMENT PA-2DR-12 below]` — the run is registered against an immutable
  tree, never a mutable working tree.
- **SLURM job id:** `[PLACEHOLDER — fill by amendment after submission]`
- **Out-root (registered, PA-CA-11):** `stage_lhs2d`'s `--out-root` **defaults to `p3_2d_work`,
  while the banked fleet lives in `p3_2d_fleet_20260825/`.** The fresh run MUST be given its OWN
  out-root, explicitly, and PA-CA-11's out-root guard honoured (an existing `<subdir>_meta.json` is
  REUSE, never silent re-run, and every reuse is disclosed). Registered value:
  `[PLACEHOLDER — fill by amendment after submission, stated exactly as the submitted job writes;
  NOT invented, and NOT the default]`. Writing into the banked fleet's root would contaminate the
  frozen comparison set.
- **Catalogue pin:** md5 `c52c13b5cab61f6b3f04bbe202550969`, **VERIFIED 2026-08-27** (PA-2DR-6);
  canonical record `results/mechanism_study_20260813/PINNED_INPUTS_MANIFEST.md`. Per CLAUDE.md's
  dataset-pinning rule the fleet driver **aborts** on mismatch rather than logging it.

### v2.11 Rule-1 exoneration check (PA-2DR-9, transcribed)

Two-layer check over the union of `EXONERATION_REGISTER_20260827.md` §1 ∪ §2 ∪ §3 ∪ §4 ∪ §8.
Result: **substantive PASS, four adjacent entries, none reopened.** Cited by TAG heading per the
register's §9 maintenance rule.

- **HB — "hard mass window as support truncation".** Synonym list includes "mass floor", "hard
  clamp on mass", "truncation bias". **Not reopened:** HB is scoped to the *estimator's*
  candidate-search mass window (`handler.py`, call site `bayesian_statistics.py:4691`), measured
  against the H0 MAP / catalogue-leg nat budget. Defect 1 is a floor in the *venue's* latent draw
  law (`correspondence_1d.py:_draw_2d_accepted_latents`), measured against the LHS2/RHS2 identity
  residual. Different file, stage, variable and estimand.
- **[LNM-DRAW] — "the ln-mass draw itself".** **Not reopened:** exonerates the *estimator's*
  log-mass draw as a 2D-bias driver at `|Δln M| ≤ 0.0009` dex; Defect 1 is a support defect in the
  *venue's* mass draw and a correctness-class finding about the identity, not a bias claim about h.
- **HC — "mixture-floor / zero-handling".** **Not reopened:** HC's own delimitation is explicit that
  a different harness's zero-handling is not covered. `p3_2d_fleet.py` is a different harness from
  the production combination code HC measured.
- **[WBHZERO-ASYMMETRY]** (a confirmed defect, not an exoneration). Engaged only as the invariant
  `mass_filter_sigma="symmetric"` (`[PHYSICS]` `cf4f8a2a`, row #202). Not disturbed.


---

**PA-2DR-12 (2026-08-27; the frozen instrument SHA — resolves PA-2DR-8 item 3 and fills §v2.10;
severity BLOCKING-as-registered, now CLOSED).**

The repair branch is committed. **The run is registered against an immutable tree, not a mutable
working tree.**

- **Branch:** `fix/p32d-classg-venue-repair`
- **Commit SHA:** **`3694233d`** — `fix(harness): P3-2D class-G venue — reject M<=0 latents,
  honour PA-2D-1 F16 dead-row convention`
- **Files in that commit (five; the four of §v2.10 plus the gate ledger):**

```
darksiren_emri/validation/correspondence_1d.py            (Defect 1: M<=0 rejected, floor deleted)
darksiren_emri_test/validation/test_correspondence_1d.py  (gates G1 + new G2, G3)
results/.../p3_2d_fleet.py                                (Defect 2 + v2 paired reads P2/P3/P5, G5)
results/.../test_p3_2d_fleet_defect2.py                   (NEW: Defect-2 regression, was untracked)
docs/gates/PHYSICS-GATE-LEDGER.md                         (gate row, PA-2DR-8's requirement)
```

- **PA-2DR-8 item 2 resolved.** The reported `git status` under-reporting did not reproduce: a bare
  `git status --porcelain` showed all three tracked modifications. The lost file was a drafting
  omission, not an index anomaly; no index state needed clearing. Recorded so the next reader does
  not hunt for a phantom.
- **Verification run before the commit, actual output:** full fast suite
  `uv run pytest -m "not gpu and not slow"` → **1831 passed, 15 skipped, 27 deselected**, coverage
  73.11% (gate 25%). The four gate tests (G1–G3 plus the Defect-1 rejection regression) pass;
  `test_p3_2d_fleet_defect2.py` → 2 passed. `ruff check`, `ruff format`, `mypy` clean on all
  changed files.
- **Non-vacuity of G3 verified before registration**, not assumed: the pre-repair reference
  implementation, run through G3's own analytic target on the same pool and seed, gives
  `KS D = 0.11030, p = 1.2e-21` with 218/2000 accepted masses at exactly the old 1.0 M_sun floor.
  The gate discriminates.

**Still OPEN before submission:** the SLURM job id and the out-root (§v2.10) — the out-root must be
given explicitly and must NOT be `stage_lhs2d`'s `p3_2d_work` default, which would collide with the
banked fleet's frozen comparison set.

**Status: FIXED** (SHA frozen; §v2.10's placeholder resolved).

---

**PA-2DR-13 (2026-08-27; submission record — fills §v2.10's remaining placeholders; `[OPUS-ORCH]`)**

The run registered by this document has been submitted. Both remaining §v2.10 placeholders are now
resolved, and this amendment is the record that they were resolved *at submission*, not afterwards.

- **SLURM job id:** `6723958` (array `0-23`, partition `cpu_il`, submitted 2026-08-27, state
  PENDING at submission).
- **Out-root:** `$WORKSPACE/p3_2d_fleet_repair_20260827` — a FRESH directory, explicitly **not**
  the banked `p3_2d_fleet_20260825/`.
- **Frozen commit:** `d04d9dc9` on branch `fix/p32d-classg-venue-repair`, tagged
  `p32d-repair-4af1baec` on the cluster. This is one commit later than PA-2DR-12's `4af1baec`; the
  delta is the out-root fix below and touches no instrument code (`git diff 4af1baec d04d9dc9`
  is `cluster/p3_2d_fleet.sbatch` only, +6/−1).
- **Cluster state at submission:** preflight `VERDICT: READY ✓ (WARN: 1 issue(s))`; catalogue
  `cols=8 [OK]`; repo `ahead=0 behind=0`, tracked-dirty `0` (257 untracked outputs, verified
  untracked before checkout); queue empty of prior jobs.

**A pre-submission defect was caught and fixed, and it is worth recording because it would have
been silent.** `cluster/p3_2d_fleet.sbatch:71` hardcoded `OUT_ROOT="$WORKSPACE/p3_2d_fleet_20260825"`
— the banked fleet's own directory. Submitted unmodified, PA-CA-11's idempotency guard (which
skips any `(arm, seed)` whose `<arm>_<seed>_meta.json` already exists) would have skipped all 48
arm-seed pairs: the array would have completed successfully, produced no new data, and any
subsequent `stage_lhs2d` would have re-scored the OLD floor-contaminated CSVs while appearing to
be a repaired-venue result. The failure mode is a green job with a wrong answer. Fixed in
`d04d9dc9` by making `OUT_ROOT` overridable (`${OUT_ROOT:-...}`) with the banked path retained as
the default, and submitted with `--export=ALL,OUT_ROOT=...`.

`write_provenance.sh` is wired into this script (added earlier the same day, commit `67b18592`),
so this run stamps its own commit, branch, dirty-count, job/array ids, seed, host and timestamp
into its out-root — the first fleet in this campaign to do so.

**Status: FIXED.** §v2.10 carries no remaining placeholders. No result may be read from this run
until the four registered gates (G2 draw-count, G3 truncated-normal KS, G5 mass-window count,
G6 exact identity) have been evaluated, per §v2.6.
