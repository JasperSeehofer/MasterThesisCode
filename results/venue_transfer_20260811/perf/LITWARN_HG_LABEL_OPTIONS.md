# H-g status-label nonconformance — options for author decision

**Author decision required — no edit applied.**

Source: `docs/LITERATURE_WARNINGS.md`, row H-g (arXiv:2212.08694 table, line 53).
Runbook §3.6 flags this as a "5-min fix, author call on label" item.

## 1. Current row (verbatim)

```
| H-g | **§4.2 Inconsistency 4 — GW likelihood mismodeling, dropped σ(d_L^true) dependence** (verbatim, quote-verified; not in the 2026-08-05 intake): "Another possible source of error is treating the GW likelihood as if the standard deviation was not dependent on the true value of [d_L], although the simulations are made assuming this dependence … By dropping the overall normalization factor in the GW likelihood, one is in practice ignoring a part of the likelihood that depends on the true luminosity distance. This causes a biased dependence of the [H₀] posterior on the luminosity distance uncertainty … We find that in this case the inconsistency has the effect of biasing [H₀] towards lower values for increasing values of [the d_L uncertainty]" | the GW likelihood's normalization must retain its σ(d_L^true) dependence, not just its mean-value dependence | **measured-adjacent** | added at Stage L intake, thread 16's R0 sweep (`m2_residual_owner/CLAIM_M2_RESIDUAL_OWNER_20260807.md` [LIT-2]). Mapped as the concrete mechanism candidate for H-c in the M-2 matched 2D overlap residual hunt: a low-H₀ bias growing with d_L uncertainty is the documented shape of the residual selection confound the thread traced to the collinear d_L-geometry + ball-density bundle. The ratified stage-5 verdict (`BIAS_HISTORY_LEDGER.md` §5, AUTHOR RULING 2026-08-08; row #97) found this bundle confounding-absorbable to ~2/3 by a smooth, verified d_L-functional completion-leg response (A2, R²=0.88) with the remaining ~1/3 density-coupled at joint_r1 (specification-fragile) and not significant at iiib — the dissolution of thread 16's residual into this bundle is evidence *adjacent to* H-g's mechanism, not a direct measurement of the dropped-normalization condition itself; still **not** independently checked against production code |
```

The `our status` field reads **`measured-adjacent`**.

## 2. Status vocabulary in use (file header, lines 16-25)

| status | meaning |
|---|---|
| `CHECKED` | we measured or argued the condition and it **holds** at the named venue(s) |
| `VIOLATED` | we measured the condition and it **fails**; the consequence is stated and linked |
| `UNDER MEASUREMENT` | an instrument is specified/pre-registered; verdict pending |
| `OPEN` | recognised as live, no instrument yet |
| `N-A` | structurally inapplicable to this pipeline, with the reason stated |
| `UNCHECKED` | we know the warning exists and have **not** looked. Say so; never leave a row blank |

**Deviation:** `measured-adjacent` is not one of the six vocabulary tokens. It reads as a
hand-rolled hybrid (implying partial measurement) that the header table does not define,
so a reader cannot look it up. It also breaks Rule 1 in spirit ("A status with no
evidence link is not a status") by using a label whose meaning is not registered anywhere
in the file.

## 3. H-g's actual state (from the row's own evidence + cross-check)

- No instrument was specified/pre-registered *for H-g itself* — the row's evidence is
  entirely a mapping exercise: thread 16 (`CLAIM_M2_RESIDUAL_OWNER_20260807.md` [LIT-2])
  identified H-g's documented mechanism (low-H₀ bias growing with d_L uncertainty) as the
  *candidate shape* of a residual that thread 16 was investigating for an unrelated
  reason (the M-2 matched 2D overlap residual / H-c mechanism hunt).
- The ratified stage-5 verdict (`BIAS_HISTORY_LEDGER.md` §5, AUTHOR RULING 2026-08-08,
  row #97) resolved that *residual* (confounding-absorbable ~2/3 by the completion-leg
  response, ~1/3 density-coupled and venue-fragile) — but that verdict is about the M-2
  residual, not a direct test of "does our GW likelihood normalization retain its
  σ(d_L^true) dependence."
- The row's own closing clause is explicit: "not independently checked against
  production code."

This is structurally identical to the H-e situation described in this same file's
2026-08-08 addendum (lines 55-67): adjacent, direction-matching evidence surfaced by
another thread, but **not** an independent, unconditional measurement of the warning's
own condition. The file already resolved that ambiguity for H-e by keeping the header
status `UNCHECKED` and putting the nuance in prose (the addendum), rather than inventing
a new label.

## 4. Options

### Option A — `UNCHECKED` (matches the H-e precedent) — **recommended**

One-line justification: No instrument was ever specified/pre-registered against H-g's
own condition (σ(d_L^true) dependence retained in the GW likelihood normalization); the
row itself says "not independently checked against production code" — that is exactly
`UNCHECKED`'s definition ("we know the warning exists and have not looked"), and it is
the same resolution this file already chose for the structurally identical H-e case.

Replacement row text (diff-ready, only the status cell changes):
```
-| ... | the GW likelihood's normalization must retain its σ(d_L^true) dependence, not just its mean-value dependence | **measured-adjacent** | added at Stage L intake, thread 16's R0 sweep ...
+| ... | the GW likelihood's normalization must retain its σ(d_L^true) dependence, not just its mean-value dependence | **UNCHECKED** | added at Stage L intake, thread 16's R0 sweep ...
```
(evidence text unchanged — it already explains the adjacency and the "not independently
checked" caveat, which is exactly what an `UNCHECKED` row with rich evidence should say
per Rule 1.)

### Option B — `UNDER MEASUREMENT`

One-line justification: could be argued if the author intends thread 16's stage-5
verdict to *count* as the pre-registered instrument now closed out, i.e. treat the
dissolution-into-the-bundle finding as the "verdict" for H-g and only the *cross-check
against production code* as the still-pending piece.

Replacement row text (diff-ready):
```
-| ... | the GW likelihood's normalization must retain its σ(d_L^true) dependence, not just its mean-value dependence | **measured-adjacent** | added at Stage L intake, thread 16's R0 sweep ...
+| ... | the GW likelihood's normalization must retain its σ(d_L^true) dependence, not just its mean-value dependence | **UNDER MEASUREMENT** | added at Stage L intake, thread 16's R0 sweep ...
```
Caveat: this stretches the vocabulary's definition ("an instrument is
specified/pre-registered; verdict pending") — thread 16's instrument was
pre-registered for the M-2 residual/H-c hunt, not for H-g's condition specifically, so
using `UNDER MEASUREMENT` here would misattribute an existing instrument to a warning it
wasn't built to test. Weaker fit than Option A.

### Option C — `OPEN`

One-line justification: H-g is "recognised as live" (the mechanism is documented and
mapped to a real residual) but there is genuinely "no instrument yet" that targets H-g's
own condition directly — this reads the row as pre-instrument rather than as an
attempted-but-incomplete measurement.

Replacement row text (diff-ready):
```
-| ... | the GW likelihood's normalization must retain its σ(d_L^true) dependence, not just its mean-value dependence | **measured-adjacent** | added at Stage L intake, thread 16's R0 sweep ...
+| ... | the GW likelihood's normalization must retain its σ(d_L^true) dependence, not just its mean-value dependence | **OPEN** | added at Stage L intake, thread 16's R0 sweep ...
```
Caveat: `OPEN` undersells how much adjacent evidence already exists (the row is not a
bare "live" flag — it has a specific, ratified, direction-matching finding behind it);
`UNCHECKED` communicates that richer state better per Rule 1's evidence requirement,
and again matches the H-e precedent already set in this file.

## 5. Recommendation

**Option A — `UNCHECKED`.** It is the only option that (a) uses the header vocabulary as
literally defined, (b) matches the resolution this file already applied to the
structurally identical H-e case (adjacent, direction-matching, thread-16-sourced
evidence that is explicitly *not* an independent measurement of the warning's own
condition), and (c) requires zero rewriting of the evidence prose — the existing text
already reads correctly under an `UNCHECKED` header, since Rule 1 permits (and this file's
own precedent shows) rich evidence attached to an `UNCHECKED` status when that evidence
falls short of an independent check.
