# [WGEO] stage 0, read (3) — coupling to the known tilt, and the rule-1 delimitation

**Date:** 2026-08-27 · **Scope:** record-only (ledger + claim cards + cited source lines).
No new measurement was run; no darksiren_emri/ file was touched. This read does **not** have
read (2)'s census (a fleet-wide window-vs-z sweep, if run) — every "unknown" below is stated
as unknown, not estimated.

---

## 1. RULE 1 — does a window-GEOMETRY claim reopen a standing exoneration?

**Ledger §2 exoneration text, quoted verbatim** (`gate_b_20260730/BIAS_HISTORY_LEDGER.md:132-135`):

> "Already on the claim file's list (do not re-open): catalogue dd_L/dz Jacobian · Fisher frame
> · p_det estimator choice · p_det inside/outside · h-prior sensitivity · `volume_trunc` · the z
> leg · the ln-M draw · realization plumbing · **candidate-window membership** · **mass-kernel
> family** · Option-A β_G/Σ_glob drift ... · HA as owner · HC zero-handling · HB."

The two items with the closest names are pulled from the current text of the claim file itself
(note: the ledger's inline citation `CLAIM_2D_BIAS_20260730.md:191-204` has drifted — that
range now falls inside the C4-obs numeric block, not the exoneration list; the list itself lives
at `CLAIM_2D_BIAS_20260730.md:721-734` under the header "## Exonerated — do NOT re-open without
new evidence". This is a citation-drift note, not a content dispute.):

> "candidate-window **membership** (exact removal moves MAP 0.81→0.82, wrong sign) · mass-kernel
> **family** (bounded +0.002)" (`CLAIM_2D_BIAS_20260730.md:726-727`)

**Row #72 / #89** (the two measurements behind "mass-kernel family (bounded +0.002)"):
- #72 (`BIAS_HISTORY_LEDGER.md:92`): seed600 494-ev A/B on the mass-kernel form used inside the
  Bayesian mass marginalization — "EXONERATED: Δ2D mean +0.0029, wrong sign; Δ1D 0.0000 exact."
- #89 (`BIAS_HISTORY_LEDGER.md:109`): 4-cell A/B on the ratified (M1) truncated-lognormal
  kernel — "NECESSARY, NOT SUFFICIENT... moves 2D down only −1.8/−2.3 ln of a +25.6/+29.1
  excess; 2D MAP unmoved at 0.80."

**Precise delimitation (both objects are code-distinct from [WGEO]'s target):**

1. **"Mass-kernel family"** (#72/#89) is the functional FORM of the likelihood kernel used to
   marginalize catalogue mass in `bayesian_statistics.py` (gaussian vs truncated-lognormal
   `mass_trunc`) — a modeling choice about how mass uncertainty is convolved *inside the
   posterior*, applied to candidates that have already survived eligibility. It is **not** the
   eligibility WINDOW (`handler.py:663-673`, `mass_filter_mask`) that decides which galaxies
   even reach the with-BH numerator. [P3-MKER]'s own rule-1 check made exactly this
   distinction for its part (a) (the kernel widening question) and explicitly separated it from
   part (b) (the window): *"the window-exclusion finding... concerns a **different object** —
   the eligibility filter, rows #198–#202's territory — and its H₀ effect is NOT MEASURED here"*
   (`CLAIM_P3_MKER_20260826.md`, §R1.8). [WGEO] inherits that same separation and does not
   reopen #72/#89 — it says nothing about kernel family, gaussian vs lognormal FORM, or the
   bounded +0.002.

2. **"Candidate-window membership"** (the exoneration adjacent to #72/#89, same list line) tested
   a coarse **binary** lever — *exact removal of the window* (no mass filter at all vs the
   filter present), moving 2D MAP 0.81→0.82 in the wrong direction. That is a bound on the
   window's **presence/absence**, not on its **internal geometry** while present. [WGEO]'s claim
   is specifically that a present, linear-symmetric window is asymmetric in the log-normally
   distributed true variable it gates — a shape question orthogonal to the on/off lever that was
   actually measured. Removing the window and reshaping it are different operations and are not
   guaranteed (or even likely, given how differently they act on the tails) to share a sign or
   magnitude.

3. **Rows #198–#202 (symmetric-window adoption, `cf4f8a2a`)** are also a *different* symmetry
   axis than [WGEO]'s. The adopted "symmetric" flag makes the **same `sigma_multiplier`** apply
   to **both** the GW-side mass uncertainty and the galaxy's own `BH_MASS_ERROR`
   (`BIAS_HISTORY_LEDGER.md:2929-2932`: *"symmetric = `sigma_multiplier` applied to BOTH the GW
   mass uncertainty and the galaxy `BH_MASS_ERROR`"*), replacing an old convention that used
   different multipliers (GW ±1.5σ vs galaxy ±1σ, `A20_REVIEW...`/row #194-196 language). That
   symmetry is about **which side of the comparison gets which multiplier**. [WGEO]'s symmetry
   axis is **linear vs log space within a single side's error model** — whether a ±kσ cut applied
   linearly to a log-normal quantity is symmetric in the variable that is actually log-normal.
   These are two independent meanings of "symmetric" acting on the same code region; adopting
   one says nothing about the other. Row #202 itself carries an explicit open remainder that
   names this exact gap: *"Open remainders carried: the filter-vs-kernel model-consistency
   question (proposal §6 caveat 2, un-opened thread)"* (`BIAS_HISTORY_LEDGER.md:2971-2972`) —
   [WGEO] is that thread's continuation, not a re-litigation of the row #202 ruling.

**Verdict on rule 1: does NOT reopen a standing exoneration**, on the record as read. Precise
delimitation for the card: [WGEO] may claim the window's **geometry-given-presence** is
unexamined and may bound or measure its effect; it must **not** claim any H₀ number is
overturned for kernel family (#72/#89, bounded +0.002, wrong sign already established), for
window on/off (0.81→0.82, wrong sign, coarse), or reargue the GW-vs-galaxy multiplier symmetry
ratified at row #202. `[P3-MKER]`'s own R1.8 already ran the identical check for its adjacent
part (b) and reached "PASSED, stated explicitly with the bound" — [WGEO] is consistent with
that finding, not contradicting it.

**One further scope note, not a blocker:** [P3-2D]'s current PARKED thread (row #211,
`UNATTRIBUTED-bounded`, `STUCK_P3_2D_SYMPTOM_CARD_20260826.md`) is a *different* object again —
an internal Σ̃^4D/RHS identity residual (class-G draw-law contraction question), not the
eligibility window and not the −0.635 dark-class score. No overlap risk with [WGEO] was found in
the record; flagged only so the two STUCK/parked threads aren't conflated by a reader.

---

## 2. THE TILT — banked z-structure of the base tilt

**Headline number** (`BIAS_HISTORY_LEDGER.md:1347-1348`, row #137):

> "the dark-class per-event SCORE at truth is **−0.635 ± 0.017** (iiib, 37σ) / −0.565 ± 0.020
> (joint_r1, 28σ)."

**What "dark class" means — exact, code-grounded definition**, from the pre-registration this
row cites (`results/prod2d_closure_20260818/PREREG_COMPLETION_CLASS_DECOMPOSITION.md:20-22`):

> "C-B — `in_catalog == False` AND `L_cat_no_bh > 0` at ≥ 1 h node (impostor-only ...)
> C-C — **`L_cat_no_bh == 0` at every h node** (pure completion leg)."

`L_cat_no_bh` is computed over `candidate_hosts_without_bh_mass` — the sky/z-cone candidate set
returned by the BallTree lookup, **before** any mass filtering. The mass filter (`mass_filter_mask`)
only ever subsets a *further* set from it: `candidate_hosts_with_bh_mass =
candidate_hosts_without_bh_mass[mass_filter_mask]` (`handler.py:663-674`, quoted and
source-confirmed in `CLAIM_P3_MKER_20260826.md` §R1.1). **Class C-C ("dark") membership is
therefore determined entirely by the sky/z cone search and is structurally independent of the
mass window** — an event is dark because it has *zero catalogue candidates in the cone at all*,
not because candidates existed and the mass filter emptied them. (That second, different
situation — nonzero `L_cat_no_bh`, zero `L_cat_with_bh` — is [P3-WBHZERO]'s object, rows
#194-#202, a different class again.)

**z-profile, quantitative, banked** (`docs/derivations/population_mismatch_dark_score.md:41-46`,
iiib, dark class, n = 605):

| z bin | n | measured score | predicted (population-mismatch model) | ratio |
|---|---|---|---|---|
| 0.075–0.392 | 121 | +0.014 | −0.095 | — (both ≈ 0) |
| 0.392–0.559 | 121 | −0.465 | −0.329 | 1.41 |
| 0.559–0.659 | 122 | −0.743 | −0.818 | 0.91 |
| 0.659–0.753 | 120 | −0.902 | −0.659 | 1.37 |
| 0.753–1.018 | 121 | −1.081 | −0.592 | 1.83 |
| **ensemble** | **605** | **−0.635** | **−0.555** | **1.14** |

Consistent with row #137 item 3 (`BIAS_HISTORY_LEDGER.md:1352-1354`): *"score ≈ 0 below z ≈ 0.4,
monotone to −1.08 at z ≈ 0.9 — a DEEP completion-leg phenomenon."*

**Current attribution status** (not window-related, extensively investigated): row #138
(`:1367-1392`) predicted 87% of the score from population misspecification (constant-comoving
prior vs the injected Barausse M1 rate — a shape mismatch entirely internal to the completion
term's redshift weighting, `bayesian_statistics.py:1192`); rows #139-#144 (`:1394-1591`) then
found and partially withdrew an "internal misnormalization" claim after a harness-control defect
was discovered; rows #149-#157 (`:1897-2192`) isolated a *related but distinct* S̄_φ
normalization-pairing defect that nulls a **different** matched-channel statistic
(+0.0076 ± 0.0184) but is explicitly disclosed as **not** curing the production rail (row #157
item 3: *"Banked caveat carried: fixing the off cell does NOT cure the H₀ rail"*,
`:2187-2192`). As of the most recent memory entries (2026-08-25/26), the base tilt's full
attribution remains open; it is not currently claimed to be solved by any banked mechanism,
window-related or otherwise.

---

## 3. THE SHAPE COMPARISON

**What a window-asymmetry mechanism would need to look like to produce this tilt:**
1. It would have to act **on class C-C (dark) events specifically** — score ≈ 0 below z≈0.4,
   monotonically more negative to −1.08 by z≈0.9, essentially zero contribution from any
   catalogue-supported class.
2. Its effect on H₀ would have to scale with z over the same 0.4→1.0 range, with a shape at
   least loosely tracking the ratio column above (roughly linear-to-mild-superlinear growth in
   |score|, no sign reversal, no plateau below z≈0.4).

**What is banked about the window's z-shape:** the R2 measurement (`CLAIM_P3_MKER_20260826.md`
§R2.7(ii)) established the qualitative geometric fact — a linear ±1.5σ cut on a log-normally
distributed `BH_MASS_ERROR` (σ_ln = 1.3032 for the one measured candidate) produces a negative
lower edge and an upper edge that reaches only 2.955× the central mass vs 7.06× for the log-space
equivalent — but this is **one candidate, one event** (R2.10 item 1: *"Single candidate, single
event... does NOT establish that the 0.50-dex component would fail to readmit window-excluded
candidates elsewhere in the fleet"*). **No banked measurement gives σ_ln(z), the window's
pass-rate vs z, or any H₀-effect-vs-z curve for the window.** The comparison the hypothesis needs
— "does window exclusion strengthen with z the way the score does" — is therefore **qualitative
only** on the current record; a quantitative shape match cannot be claimed or refuted from what
is banked.

**The sharper, structural finding this read adds (not previously stated in either claim card):**
because class C-C membership is fixed before the mass filter ever runs (§2 above), **the window's
geometry cannot be the mechanism for the −0.635 dark-class score as that statistic is currently
defined and measured.** The mass window only ever acts on events that already have nonzero
`L_cat_no_bh` — i.e., events in classes C-A/C-B, not C-C. A window effect that grows with z could
in principle move events *between* C-A/C-B and a "mass-filter-emptied" state (`L_cat_with_bh`
goes to 0 while `L_cat_no_bh` stays > 0 — the [P3-WBHZERO] class, rows #194-#196), but that is a
different statistic on a different event population than the one carrying the banked −0.635
score. **This is the single most decisive finding of this read**, and it substantially narrows
the live version of the hypothesis: any coupling to the base tilt, if real, would have to run
through a route this record does not show — not through the C-C per-event score as currently
measured.

---

## 4. CONFOUNDS

1. **Mass-z correlation in the catalogue is real and banked, independent of anything measured
   tonight.** GLADE+ is flux-limited beyond z≈0.3 to massive galaxies, while the EMRI host prune
   (M_BH ∈ [10^4.5,10^6] M☉) selects dwarf hosts: *"the effective host-lookup catalogue is 99.98%
   z<0.3 — only 165 galaxies all-sky at z ≥ 0.5 — because... GLADE+ beyond z~0.3 is flux-limited
   to massive galaxies. Host lookups therefore CANNOT resolve hosts at the depth-1.5 population's
   redshifts"* (`docs/H0_BIAS_RESOLUTION.md:1386-1389`). Any z-structured signature touching
   catalogue-mass objects at all is a candidate to be reproduced by this pre-existing selection
   effect rather than by window geometry specifically — and this confound predates and is
   independent of tonight's read.
2. **The completeness/selection function's own z-dependence** — `S̄_φ(z)`, `(1−f(z))`, and the
   dV_c/dz weighting are all explicit, independently-z-varying terms already in the model (§2's
   population-mismatch derivation uses exactly this kind of ratio). A z-growing residual is
   consistent with several of these already-modeled or already-mismeasured objects before it is
   consistent with an unmodeled window effect.
3. **The tilt was measured specifically on the dark (C-C) class** — per §3, this is the single
   strongest confound-breaker in the *other* direction: it is not a confound that produces false
   agreement, it is a structural fact that makes the two objects measure different event
   populations. Any apparent shape agreement between a future window-vs-z measurement and the
   C-C score would need a demonstrated causal bridge between the window (acting on C-A/C-B) and
   the C-C statistic — none exists in the record, and §3 gives a specific reason to doubt one can.
4. **Window asymmetry could be real and z-independent, or z-dependent but H₀-inert.** Nothing in
   the record measures either the fleet-wide z-trend of σ_ln (whether GLADE's propagated stellar-
   mass error, the dominant term at 64.9% of variance for the one measured candidate,
   `CLAIM_P3_MKER_20260826.md` §R2.2, itself trends with z) or the window's H₀ sensitivity at any
   z. Agreement in shape, if later found, would still need the magnitude check this read cannot
   do.
5. **Prior burn history on reconciling quantities.** The project has repeatedly found candidate
   mechanisms with a plausible high-z shape that later failed to predict both arms, or turned out
   to be harness artifacts (row #145's `-1e300` sentinel manufacturing "rail" in 25/123 seeds,
   `:1593`; the D-1 mismatch that was later withdrawn as underpowered, row #144, `:1541-1591`).
   The base standard here is high: shape agreement alone has repeatedly not survived scrutiny in
   this campaign.

---

## 5. THE BOUND CHECK

**Mass-kernel family** (a different object, §1): bounded **+0.002**, wrong sign, from #72/#89.

**The window as a separate object: no banked bound exists.**
- `CLAIM_P3_MKER_20260826.md` §R1.8: *"the window-exclusion finding... its H₀ effect is NOT
  MEASURED here. No H₀ claim is made about it."*
- §R2.9: *"Nothing in R2 measures an H₀ effect... §R1.8's caveat stands unchanged."*
- The only quantitative window-adjacent bound in the record is the coarse on/off "candidate-
  window membership" exoneration (§1 item 2 above: 0.81→0.82, wrong sign) — which bounds
  *removing the window entirely*, not reshaping it, and is explicitly not offered by either claim
  card as a bound on geometry.

**Stated plainly, as instructed:** the window's H₀ contribution, isolated from window on/off and
from kernel family, is **unconstrained** in the current record. That absence is itself the
reason this lead is worth a cheap stage-0/1 look — but it must not be read as evidence the effect
is large; it is simply unmeasured.

---

## 6. PROVISIONAL VERDICT

**LEAD-WEAK.**

Reasoning, weighted by what §3's structural finding does to the hypothesis as originally framed:

- **Against:** the base tilt's headline statistic (−0.635, 37σ) is measured on class C-C, whose
  membership is fixed by the sky/z cone search and is provably prior to and independent of the
  mass window (§2, §3). A mechanism that only ever acts on the with-BH candidate list cannot be
  the direct cause of a per-event score computed on events that never reach that list. This is a
  record-level, not merely qualitative, reason to doubt the specific coupling this read was asked
  to assess.
- **For (keeps it above LEAD-DEAD):** (a) the window's linear-vs-log geometric mismatch itself is
  real and quote-verified for at least one candidate (R2.7(ii)), not hypothetical; (b) it is
  genuinely unconstrained in H₀ terms (§5) — no banked measurement rules out a material effect
  through some *other* route (e.g., the C-A/C-B classes, or the [P3-WBHZERO]-adjacent
  filter-emptied population, which is a live, different, and separately-material channel per rows
  #200-#202); (c) the mass-z correlation confound (§4.1) is real and could still produce a
  z-structured window effect through a mechanism this read did not rule out — it only shows the
  specific bridge to the C-C score is missing, not that no bridge to H₀ exists anywhere; (d) the
  fleet-wide z-trend of the window's asymmetry has simply never been measured — the object is
  open, not closed.
- **Not LEAD-DEAD** because no measurement in the record contradicts window geometry being real
  and z-varying; only the specific *coupling to this specific exhibit* is undercut.
- **Not LEAD-LIVE** because the one concrete predictive test available from the record — does the
  mechanism explain the dark-class score — fails on structural grounds before any quantitative
  comparison is even possible, and no quantitative z-trend exists to make a positive case.

**Explicit scope caveat, as instructed:** this verdict rests on the RECORD's evidence only — the
two banked claim-card reads (R1, R2 of `CLAIM_P3_MKER_20260826.md`) plus the ledger. Read (2)'s
census (if it measures a fleet-wide window-pass-rate-vs-z or σ_ln-vs-z trend) was not available to
this read and could move the verdict in either direction; in particular a demonstrated z-trend in
the window's pass rate among C-A/C-B events, or in the [P3-WBHZERO]-adjacent filter-emptied
population, would reopen the LEAD-LIVE case through a route this read did not have the numbers to
evaluate.

---

## Provenance

- `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md` — §2
  (exoneration list, lines 127-208), rows #72 (:92), #89 (:109), #137 (:1338-1366), #138
  (:1367-1392), #139-#144 (:1394-1591), #145 (:1593), #149-#157 (:1897-2192), #198-#202
  (:2925-2974), #205-#211 (:2982-2994).
- `results/campaign51_20260728/realistic_20260729/CLAIM_2D_BIAS_20260730.md:721-734` (current
  exoneration list text; :726-727 the two adjacent-object exonerations quoted in §1).
- `results/campaign51_20260728/realistic_20260729/CLAIM_P3_MKER_20260826.md` — R1 (§R1.1, R1.6,
  R1.8), R2 (§R2.2, R2.5, R2.7(ii), R2.9, R2.10).
- `results/prod2d_closure_20260818/PREREG_COMPLETION_CLASS_DECOMPOSITION.md:20-22` (C-B/C-C class
  definitions).
- `docs/derivations/population_mismatch_dark_score.md:1-46` (tilt object statement, z-bin table).
- `docs/H0_BIAS_RESOLUTION.md:1386-1389` (GLADE flux-limit / mass-z confound).
- `darksiren_emri/galaxy_catalogue/handler.py:663-674` (mass_filter_mask subsets
  candidate_hosts_without_bh_mass — read-only citation, no file modified).
