# Pre-registration: venue information floor (issue #44) — RATIFIED

**Status: RATIFIED by the author 2026-07-26 ("information floor of 5ln
sounds good"), with the explicit instruction that the threshold choice be
TRACKED FOR FUTURE CONSIDERATION (see §Tracked choice below). Binding for
campaigns registered after 2026-07-26. Explicitly NOT applied retroactively
to the 2026-07-26 five-seed campaign (whose seed900 exclusion stands on its
own diagnosed grounds and is disclosed as post-hoc).**

## Criterion

A venue enters the MAP/bias/width statistics of a campaign readout only if,
per channel, its combined posterior satisfies

    I_venue := max_h ln P(h) − max(ln P(h_min), ln P(h_max)) >= 5 ln

evaluated on the campaign's registered h-grid from the combined-posterior
JSON (`posterior` array), before any sub-grid fitting.

- Venues failing the floor are reported as **UNINFORMATIVE** — with their
  I_venue, n_events, and golden-event count — not silently dropped. An
  uninformative venue is itself a result (a measurement of the golden-event
  rate at that venue size), it just contributes no MAP argmax to the bias
  statistics.
- The floor applies per channel; a venue may be informative in one channel
  only, and then enters only that channel's statistics.
- Rails: a venue whose argmax sits on a grid edge AND passes the floor is a
  genuine anomaly and must be investigated (the criterion deliberately does
  not auto-exclude informative rails — those falsify, not annoy).

## Why 5 ln

Measured separations (MULTISEED_READOUT_20260726.md):

| Venue | n_events | I_venue (base) | Character |
|---|---|---|---|
| seed900 fixpool | 20 | ~1.2 ln | zero golden events — flat, argmax is noise |
| seed90000 | 20 | ~19.6 ln | two golden events carry +9.2/+1.8 ln |
| seeds 1000/2000/3000 | ~3300 | O(10^2–10^3) ln | deep venues |

5 ln ≈ 150:1 peak-to-edge odds — comfortably above the ~1-ln draw-to-draw
fluctuation scale of a flat 20-event posterior, and an order of magnitude
below the least informative venue ever measured to peak genuinely (19.6 ln).
Any threshold in [3, 10] ln separates the measured population identically;
5 ln is fixed here to remove the choice from future readouts.

## Anti-tuning provisions

1. Registered BEFORE the next campaign submission; the threshold is fixed and
   may not be adjusted in response to campaign results (a change requires a
   new pre-registration effective only for later campaigns).
2. Both readouts are always reported: the floored set (primary) and the
   all-venues set (appendix), so the floor's effect is visible, not silent.
3. The floor is computed mechanically from the combined-posterior JSONs; no
   per-event inspection enters the admission decision.

## Tracked choice (author instruction 2026-07-26)

The 5 ln value is a judgment call inside a wide admissible band ([3, 10] ln
all separate the measured population identically). It is fixed now to remove
per-readout discretion, but it is TRACKED, not settled. Revisit triggers —
any of these opens a new pre-registration (effective only for later
campaigns, never retroactively):

1. A campaign produces a venue with I_venue in [3, 10] ln — the first time
   the choice actually bites, the band must be re-examined with that venue's
   golden-event diagnostics on the table.
2. Venue sizes change materially (the 1.2 vs 19.6 ln separation was measured
   at n = 20; larger small-venues shift the flat-posterior fluctuation
   scale).
3. The golden-event rate model changes (e.g. the real-data kernel broadens
   golden-event information content — see
   docs/derivations/hostz_pv_photoz_kernel.md §3.6).
4. A principled derivation of the flat-posterior null distribution of
   I_venue replaces the empirical gap argument (preferable to any fixed
   number; see also the P–P/coverage harness as the natural machinery).

Tracking locations: GitHub issue #44 (kept open for this purpose), this
file, and the session memory ledger.

## Sign-off

- [x] Author ratification: Jasper Seehofer, 2026-07-26 (via session
  instruction; threshold 5 ln approved, tracking mandated).
