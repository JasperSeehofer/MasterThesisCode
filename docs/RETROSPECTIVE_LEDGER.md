# Retrospective Ledger

**Established 2026-08-21 by author mandate** (verbatim, from the 2026-08-21 overnight-autonomy
instruction): *"Once a research cylce ends or something fails I want you to put this in a
retrospective ledger or a comparable ledger that tracks and suggests amendements."*

**Contract.** One entry per (a) research-cycle end — any stage-5/6 close, whether the verdict
confirmed, refuted, or MIXED — and (b) any **failure**: a gate that fired, a control that proved
vacuous, a registration returned NOT-READY, a withdrawn claim, a wasted-compute incident, an
operational loss (expired artifacts, dead workspace). **Append-only, never back-filled, no silent
edits** — the same discipline as `BIAS_HISTORY_LEDGER.md` and the amendment ledger in
`docs/RESEARCH_CYCLE.md`.

Each entry carries five fields:

1. **What ran / what failed** — one paragraph, with the ledger rows and artifacts it closes over.
2. **What worked** — which rules, amendments, and instruments earned their keep, by name.
   Every amendment named here gets **+1** in `docs/AMENDMENT_IMPACT_TRACKER.md` with a dated
   evidence line (same commit).
3. **What failed or dragged** — the honest list, including orchestrator errors, not only code.
4. **Suggested amendments** — each tagged `PROPOSED` (needs author ruling) or, when covered by an
   explicit author standing grant, `ADOPTED-UNDER-STANDING` with the grant quoted. Suggestions
   route to the amendment ledger in `docs/RESEARCH_CYCLE.md`; this ledger is where they are *born*,
   that one is where they *bind*.
5. **Disposition** — where the thread went next (runbook §, follow-on cycle, or CLOSED).

Precursor: `docs/RETROSPECTIVE_D1_20260820.md` (the D-1 post-mortem, written before this ledger
existed) is the form this ledger generalises; it is referenced, not rewritten.

---

*(entries below, newest last)*

## Entry 1 — 2026-08-21 — C-SG pilot STOP: GATE V fired on 3/4 seeds (gate miscalibrated by channel porting)

1. **What ran / what failed.** The mandatory 4-seed C-SG-F pilot (job 6415588) completed 4/4
   on-anchor, and the pre-committed band-setter returned `fleet_may_launch=false`: 3/4 seeds
   failed GATE V (spans 2.15–4.76 nats < 5; σ_h > 0.5·σ_prior). No fleet was launched on the
   fired STOP. Diagnosis: v2 §6's GATE V numbers were written for the FULL-channel posterior;
   the orchestrator's v3 design change ported them to the matched channel **without re-deriving
   their operating characteristics** — the exact omission A15 exists to prevent, committed by the
   same orchestrator that had credited A15 twice earlier the same night. Independent reference
   check: the v2 thresholds false-fail **5/12 banked B-SEL matched posteriors** (known-informative
   data). Amended thresholds (span ≥ 1 nat, σ_h ≤ 0.9·σ_prior — the flat-null vacuity signature)
   published with reference false-fail 0/16 and a still-can-fail demonstration (B-F1 flat mode
   fails both prongs). Prereg block "PILOT GATE V AMENDMENT"; v2 verdicts remain recorded in every
   banked JSON.
2. **What worked.** A15's pilot-first mandate + the pre-committed STOP wiring (the gate fired
   BEFORE 42 seeds ran — cost of the failure: 4 pilot seeds that remain usable). The
   reference-data method (amend gates against independent known-informative data, never against
   the data that fired them). → A15 **+1** in the impact tracker.
3. **What failed or dragged.** Orchestrator porting slip (above). Also: GATE V's σ_prior
   convention was never numerically specified in the prereg and had to be invented at
   implementation time — a registration under-specification.
4. **Suggested amendments.** PROPOSED (needs author ruling): *every gate that is moved to a new
   statistic, channel, or venue re-derives its operating characteristics against reference data
   in the same commit — porting numeric thresholds across channels without a stated false-fail
   rate is registered as an A15 violation.* (Candidate A17; not adopted under the standing grant
   because it binds future physics-adjacent judgment calls — author review requested.)
5. **Disposition.** Bands frozen (`csg_pilot_bands_output.json`), fleet launched; thread
   continues in the C-SG readout.
