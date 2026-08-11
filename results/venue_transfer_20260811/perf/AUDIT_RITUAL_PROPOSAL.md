ADOPTED (Option A) — author approval 2026-08-12; A6 applied to docs/RESEARCH_CYCLE.md. Option B not adopted. DRIFTED→action handoff: report-only until author triage (default pending §5 decision).

# Assumption & Performance Audit — recurring-ritual proposal

Codifies the author's periodic-audit habit (2026-08-12, verbatim): periodically
re-check "if the assumptions and approximations hold and if the errors are
problematic as well as ... the performance of the code and if the choices made
still hold in the premise to accelerate the pipeline." Mission spec:
`results/campaign51_20260728/RUNBOOK_NEXT_SESSION_9.md` §2 item 4. Two delivery
options below share one checklist (§3); the author picks one, both, or neither.
No file listed here has been edited — this is proposal text only.

---

## 1. Option A — research-cycle amendment A6

Diff-ready insertion. Ledger row for `docs/RESEARCH_CYCLE.md` "Amendment ledger"
table (append after A5, matching the existing row schema: date | amendment |
stage | what changed | why | evidence):

```
| 2026-08-12 | A6 — periodic assumption & performance audit | 1, 3, 6, cross-cutting | New recurring ritual, cadence and trigger-gated (not stage-blocking): a standing **Assumption & Performance Audit** re-validates (a) approximation error budgets (interpolation tolerances, surrogate forms e.g. kappa_cap/p0 surrogates in `emri_rate.py`), (b) perf choices vs current unit economics (CPU-h/seed anchors, contention factors, packing rules), (c) `docs/LITERATURE_WARNINGS.md` register entries for staleness. Runs on a cadence (recommend: every 2 campaigns or 6 weeks, whichever is sooner) OR on a trigger event (§4). Produces an AUDIT_<date>.md under the active campaign's results dir; does not gate stage 3/5 decisions on its own — findings route to `/physics-change` (formula drift) or a perf-roadmap update (perf drift) as appropriate. | author mandate 2026-08-12: approximations and perf choices earn re-validation on a schedule, not only when a symptom forces the question — [[realistic-venue-performance-goal]] names realistic-venue infra as reusable, so its assumptions must stay current for follow-on projects too | `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_9.md` §2 item 4; `results/venue_transfer_20260811/perf/PERF_ROADMAP.md` (the class of finding this ritual re-validates) |
```

New subsection for `docs/RESEARCH_CYCLE.md`, to insert directly after the
"Cross-session counterpart" section (parallel structure to Stage L's own
subsection — trigger list, procedure, register pointer):

```markdown
### Ritual A6 — periodic assumption & performance audit

Distinct from Stage L (external-literature pull) and from `/commission
--research` (claim-history falsification): A6 is an **internal** re-check that
today's approximations and today's performance choices still hold under
today's premises — nothing here requires new literature or a live claim
thread. Cadence-driven, not decision-blocking.

**Cadence.** Every 2 completed campaigns, or 6 weeks of wall time, whichever
comes first. Owner: whoever holds the orchestration seat (Fable-tier session)
at the time the cadence fires; may delegate the checklist mechanics (§below)
to a `sonnet`/medium subagent, keep verdict authorship at orchestration tier.

**Trigger events (out-of-cycle).** See `PERF_ROADMAP`-adjacent proposal doc
`AUDIT_RITUAL_PROPOSAL.md` §4 for the canonical list; summarized: new
campaign kickoff, new venue/instrument, a `/physics-change` merge that
touches an approximation this ritual tracks, cluster config change (node
type, partition catalogue, packing rules).

**Procedure.** Run the shared checklist (below). Each item gets one of:
CURRENT (re-verified, no drift) / DRIFTED (re-verified, now stale — states
the delta) / UNCHECKED (not yet re-verified this cycle — never silently
skipped). File findings as an `AUDIT_<date>.md` next to the campaign it was
run against. DRIFTED items on approximation error budgets route to
`/physics-change`; DRIFTED items on perf choices route to a `PERF_ROADMAP.md`
update; DRIFTED items on the assumption register route to a
`docs/LITERATURE_WARNINGS.md` status update.

**Checklist.** Shared with Option B (routine prompt) verbatim — see
`AUDIT_RITUAL_PROPOSAL.md` §3, not duplicated here to avoid drift between two
copies of the same list.
```

---

## 2. Option B — scheduled routine (cron-style recurring session)

Uses the `schedule` skill's routine mechanism. Draft config, not created:

**Cadence recommendation:** every 6 weeks (matches A6's cadence so the two
options stay interchangeable/compatible if both are later adopted), OR
triggered manually at the start of a session that opens a new campaign
(cheaper than a blind calendar cron on a project with irregular campaign
tempo — recommend the author decide cadence vs event-triggered at adoption
time).

**Draft routine prompt** (what the scheduled agent would run):

```
You are running the recurring Assumption & Performance Audit for
MasterThesisCode (see results/venue_transfer_20260811/perf/AUDIT_RITUAL_PROPOSAL.md
§3 for the checklist this routine executes). Do NOT edit production physics
files. Do NOT self-adopt any change — this is a report-only run.

1. Read the current PERF_ROADMAP.md (or its latest successor) and the most
   recent AUDIT_<date>.md if one exists, to establish the last-checked baseline.
2. Walk the shared checklist (§3 of AUDIT_RITUAL_PROPOSAL.md):
   a. approximation error budgets — re-derive or re-cite the tolerance for
      every surrogate/interpolation currently in production use (e.g.
      kappa_cap, p0 surrogates in emri_rate.py; any table/interpolation
      swap merged since the last audit); flag any that has drifted outside
      its stated tolerance or whose validity conditions no longer match the
      current venue/regime.
   b. perf choices vs current unit economics — re-pull the CPU-h/seed
      anchor, contention factor, and packing rules from the latest campaign's
      sbatch + profiling artifacts; compare against the anchors the last
      roadmap/audit assumed; flag if the delta exceeds ~20%.
   c. prereg-assumption register — walk docs/LITERATURE_WARNINGS.md row by
      row; for each CHECKED/N-A row, confirm the checked venue/regime still
      matches production; for each UNCHECKED/OPEN row, note it is still
      open (do not silently resolve).
3. Write results/<active campaign dir>/AUDIT_<date>.md with per-item
   CURRENT/DRIFTED/UNCHECKED verdicts and evidence links. No code edits.
4. If any item is DRIFTED, name the concrete follow-up (/physics-change gate,
   PERF_ROADMAP update, or LITERATURE_WARNINGS status change) but do not
   perform it — surface for author decision.
5. Report a one-paragraph summary to the author: N current / N drifted / N
   unchecked, and the single most urgent drifted item if any.
```

**Model/effort tiering for this routine** (per CLAUDE.md orchestration mandate):
checklist mechanics (pulling numbers, diffing against last audit, walking the
register) = `sonnet`, effort `medium`; any DRIFTED verdict that implies a
physics-change judgment call escalates to the orchestrating session (inherit
tier), never decided by the routine itself.

---

## 3. Shared audit checklist

Both options execute the same three-part checklist; text kept in exactly one
place (here) to avoid the two options drifting apart.

**(a) Approximation error budgets.** For every surrogate, interpolation, or
closed-form approximation currently live in production (current inventory,
non-exhaustive — the routine must re-enumerate, not trust this list blindly):
- `kappa_cap`, `p0` surrogates and the rest of the `mbh_mass_function` /
  `R0_per_mbh` / `duty_cycle_Gamma` chain in `master_thesis_code/emri_rate.py`
  — currently exact per-call evaluation (PERF_ROADMAP §2), but if/when the
  φ(M) interpolation swap (roadmap row 3) lands, ITS tolerance derivation
  becomes an audit item from that point forward.
- `_phi_dark_mass_log10_grid` (bayesian_statistics.py:1719) — the existing
  cached 600-point log10-M table; re-check node density/coverage still
  brackets the M range production queries actually hit.
- Any five-point-stencil / interpolation choice already flagged in CLAUDE.md's
  "Known Bugs" physics section.
Re-check: does the stated tolerance still bound the observed error at the
CURRENT venue (not the venue it was derived against)? Has a venue change
(new campaign, new σ_z regime, new balls mode) moved the query distribution
outside the validated range?

**(b) Perf choices vs current unit economics.** Re-pull, don't assume stale:
- CPU-h/seed anchor (currently 3.79 CPU-h/seed, PERF_ROADMAP §1.3) — still
  representative of the current campaign's event mix (K census), or has the
  mix shifted (e.g. more tail-heavy events)?
- Contention factor (currently ~1.7× measured for 5 tasks/node packing,
  PERF_ROADMAP §4) — still holds at current node/partition catalogue?
- Packing/topology rules (`--cpus-per-task` vs `--seed-range` sizing,
  PERF_ROADMAP §4 row 2 finding) — still matches the sbatch actually in use,
  or has a newer sbatch reintroduced the under-utilization gap?
- Any GPU/caching lever marked "not yet actionable" in PERF_ROADMAP — has its
  blocking condition (production-file gate, no long-wall GPU partition)
  changed?

**(c) Prereg-assumption register re-check.** Walk `docs/LITERATURE_WARNINGS.md`
row by row: confirm CHECKED/N-A rows' checked venue still matches production;
confirm UNDER MEASUREMENT rows haven't silently gone stale (instrument run,
never adjudicated); confirm OPEN/UNCHECKED rows are still honestly reported as
such, not quietly assumed resolved by later work that never updated the row.

---

## 4. Trigger events (out-of-cycle audit)

Run the checklist immediately, don't wait for cadence, when any of:

1. A new campaign is kicked off (fresh event mix, fresh seed range — the
   unit-economics anchors in §3(b) are campaign-specific).
2. A new venue or instrument is introduced (new balls mode, new σ_z model,
   new host-selection path — changes the query distribution §3(a) depends on).
3. A `/physics-change` merges that touches any approximation this ritual
   tracks (interpolation swap, surrogate form change, tolerance revision).
4. Cluster config changes (new partition catalogue, node type, `--time`
   budget, or packing convention differs from what §3(b) last measured).

---

## 5. Open questions for the author

- Adopt A (skill amendment), B (scheduled routine), both, or neither.
- Cadence: 6 weeks / 2 campaigns as drafted, or a different period.
- Owner of the DRIFTED→action handoff: does a DRIFTED verdict auto-open a
  GitHub issue (per CLAUDE.md's GitHub-sync mandate), or stay report-only
  until the author triages it manually?
