# PROPOSAL — stage 4: derive the correct estimator from the generator; corner T_res

**Date:** 2026-08-15 · **Status: PROPOSAL — nothing registered, no arm authorized.** ·
**Authorized to draft:** ledger row #107 item 3 · **Stands on:** rows #103–#107 (mechanism ledger
fully measured at single-term level: J −0.01805, REN −0.0019, additive; remainder +0.0178 =
uncancelled α-tilt + T_res; coverage not restored; 2D-only sub-additivity +0.0027).

## 1. The strategic shift this proposal asks for

Three stages of term-by-term hunting have located every candidate the register could name and
measured them to additivity — and +0.0178 remains, owned by no registered term. The remaining
principled route is different in kind: **the generator is known code, so the exactly-correct
likelihood p(data | h) is derivable from it** — the generative density of the venue's actual
data-making process (host draw, d_obs draw, ball membership on true z, per-candidate scatter),
marginalized as the estimator should marginalize. Diffing that derived form against the coded
estimator term-by-term makes the **complete** misspecification enumerable — T_res stops being
hunted and becomes *computed*. The validated end product is precisely the `/physics-change`
new-formula candidate (author-gated as ever), with C1–C16 as its mechanical acceptance filter.

## 2. Proposed work (L0-first; committed data + derivation; one conditional confirmation arm)

| item | what | cost | settles |
|---|---|---|---|
| **L4-DER** | the derivation: write the generator's true p(data\|h) (from `venue_transfer.py`'s own draw code), marginalize per candidate, and produce the correct estimator form; term-by-term diff vs the coded form (expected recoveries: J, REN, the α score-balance term; plus whatever else falls out — each diff term gets a predicted tilt) | orchestrator, the stage's core | the full misspecification list; T_res's identity as a prediction, not a fit |
| **L4-T1** | measure T(AJREN) from the new arm's committed ln_post: does the remaining tilt equal the α-tilt within errors? (If yes, T_res(full dose) ≈ 0 and the entire remaining bias is the uncancelled α term — sharpening L4-DER's target.) | CPU-minutes | the remaining tilt's composition at full dose |
| **L4-T2** | validate L4-DER's diff terms against every committed constraint at L0: the T_res dose curve (+699/+149/−62), the host gate, non-additivity, parity, the 2D-only sub-additivity (the g_i/selection structure is 2D-specific — the natural home of a channel-asymmetric term) | CPU-minutes to hours | which diff terms are live; kill tests per term |
| **A-FULL (conditional L1)** | ONE instrument arm running the complete derived estimator; prediction: bias in-band AND coverage restored (DS-M1 TERM-OWNS + DS-J1). Registered fresh under A8-v2 only after L4-DER/T2 survive and the author grants the [DO] | ~25–37 CPU-h | whether the derivation is the repair |
| pre-decided | if A-FULL restores calibration, the thread's product goes to the `/physics-change` gate as a complete package (old formula: dossier §1; new formula: L4-DER; regression test: A-FULL); if it does not, the generator/estimator dichotomy itself is wrong somewhere (e.g. an input pin) and the honest next step is a fresh stage-0 intake | — | the exit condition, cheap now |

## 3. Decision table

| # | decision | tag |
|---|---|---|
| 1 | Run L4-DER + L4-T1 + L4-T2 (derivation + committed-data measurements; no instrument time) | **[DO]** |
| 2 | Pre-authorize drafting (not registering) the A-FULL registration in parallel | **[DO]** |
| 3 | A-FULL registration + run — returns as a fresh [DO] with the L0 evidence attached | deferred by design |
| 4 | Whether L4-DER's derived form, once validated, opens the `/physics-change` intake's new-formula slot (the gate is author-gated on its face; the derivation would be presented as the candidate, not adopted) | **[RULE], author's call at that point** |

**Tiering (stated per mandate):** L4-DER — orchestrator (the scientifically-complex core; this is
the derivation the whole thread has been converging on); L4-T1/T2 measurement scripts — one
sonnet/high agent; A-FULL draft — sonnet/high from the stage-3 template; one inherit/xhigh
adversarial verifier over the derivation + validations before anything returns to the author
(≤3 top-tier cap respected). No workflow needed.

*Append-only from its commit. No repair proposed; the slot stays empty until the author rules on
item 4 with the validated derivation in front of them.*
