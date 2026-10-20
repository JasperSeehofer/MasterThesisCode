# [CMEM] CLAIM INTAKE — completion-term treatment of the designed-in ~17% cone loss

**Opened:** 2026-08-28, by author ruling (ledger row #216 item 4: MKER-6 stage-1 [DO]).
**Stage:** research-cycle stage 0 (intake). **Scope:** ABSOLUTE bias, channel-common —
explicitly NOT a tilt candidate (rule-1 delimitation in the CLAIM_P3_MKER FORK RULINGS
entry). **Standing:** raised by the A-6 split (row #216 item 2) exonerating the config
axis — the grown 2D bias attributes to estimator-side composition, and this is the
named candidate mechanism for the absolute scope.

## 0.1 The claim (intake form)

The sky cone (hard 1.5·√λ_max candidate window) excludes the true host for
**16.8%** of events (380/2261, bc arm; per-seed 10.1–24.5%) even when that host is
in the catalogue — consistent with the cone working AS DESIGNED (envelope
13.4–32.5%; census entry in `CLAIM_P3_MKER_20260826.md`, R-MKER-6 STAGE-0 CENSUS,
[DOC]). **Claimed live question:** the estimator's assembly
`p_i = (β_G·L_cat + B_num)/D` may implicitly assume "in-catalogue ⟹ in the
candidate list": `L_cat` sums only the cone ball while `β_G` is the full-catalogue
selection integral, so an in-catalogue true host outside the cone contributes to
NEITHER numerator leg while the partition weight still charges the event to the
catalogue side — structurally mis-weighting ~17% of in-catalogue events toward (or
away from) the completion term. Whether any part of that weight is rerouted
depends on `B_num`'s sky domain (out-of-catalogue completion ≠ out-of-cone
in-catalogue hosts).

Provenance chain: census fraction [DOC] (anchor reproduced full-float before any
census number counted); the envelope consistency check [LOCAL] (chair, closed
form); the mechanism statement here [INFER] — no measurement of its H₀ weight
exists yet.

## 0.2 Exoneration check (both layers, mechanism-grepped 2026-08-28)

Checked `CLAIM_2D_BIAS_20260730.md` "Exonerated" list AND
`gate_b_20260730/BIAS_HISTORY_LEDGER.md` §2 items 1–17. **Not exonerated.**
Closest entries, each a DIFFERENT mechanism (distinctions binding):

| exonerated entry | why it is not this claim |
|---|---|
| candidate-window **membership** (exact removal, MAP 0.81→0.82 wrong sign) | tested *removing members from the ball*; this claim is about the **partition weighting** of events whose true host was never in the ball |
| §2 item 3: `w_G` membership-conditioned inverse **as the fix** | a refuted *remedy* for the tilt; this is a *defect question* in the absolute scope |
| §2 item 10: `L_comp`/`B_num` as a defective **integral** | exonerated the integral's internals; this claim is about **what population the leg is charged with covering** |
| HB hard mass window | mass-axis membership; this is the sky axis |

Venue-scoping rule honoured: none of the above was measured on this question's
event class (outside-cone true hosts), so no exoneration transfers.

## 0.3 R0 sweep (Stage L trigger (a), lightweight)

Gray et al. (2020), arXiv:1908.06050 — already cited at the assembly
(`bayesian_statistics.py`, Eq. 29/A.19 comments): the formalism's G-side term
intends the catalogue sum to carry the *sky-consistent* in-catalogue hypothesis
mass; a hard candidate window narrower than the GW likelihood's support is an
implementation choice, not part of the derivation. Mandel, Farr & Gair (2019),
arXiv:1809.02063 (A2): selection normalization must integrate the SAME hypothesis
space the numerator sums. Neither paper treats a truncated candidate list
explicitly → no known-failure-mode row; full Stage L only if this thread goes
MIXED twice. `docs/LITERATURE_WARNINGS.md` row: OPEN (candidate-list truncation
vs numerator support — UNCHECKED in both sources).

## 0.4 Refute by (cheapest decisive, ordered; rule 9 — free re-reads first)

1. **[free, structural]** Trace `B_num`'s sky domain in `p_Di`: if the completion
   numerator is sky-restricted to the SAME cone, the dropped 17% weight is
   unmodeled anywhere (claim sharpens); if full-sky/pixel-marginal, partial
   rerouting exists (claim weakens to the in-catalogue-only residual).
2. **[free, paired read — A2]** Join the banked R-MKER-6 census (per-event
   inside/outside flag, 2261 events) against the banked per-event diagnostics
   (`event_likelihoods.csv`: `L_cat_no_bh`, `B_num`, `L_comp`, `combined_*`,
   per h): if outside-cone events' completion share and per-event ĥ-pull are
   statistically indistinguishable from inside-cone events (paired, per-seed
   stratified), the mis-weighting is H₀-immaterial → **REFUTED**. A coherent
   displacement on the 380-event class localizes it → stage-2 prereg.

**Both reads are measurements and require their own registration before running**
(the opening ruling's own condition). This card licenses NO run.

## 0.5 Forecast hook (stage 1, to be done in the prereg)

Crude ceiling for the prereg's band design: 16.8% of events with a mis-assigned
catalogue/completion partition, each carrying an O(w̃_G) partition weight — the
prereg must derive the expected displacement scale from the banked w̃_G
distribution before setting bands (no band invented post-hoc).

*Intake complete per rules 1–3 (exonerations checked, tags carried, Refute-by
named). Next step: stage-2 pre-registration of reads 1+2 — a fresh [DO].*
