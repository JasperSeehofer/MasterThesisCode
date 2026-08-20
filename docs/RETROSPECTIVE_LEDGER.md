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
