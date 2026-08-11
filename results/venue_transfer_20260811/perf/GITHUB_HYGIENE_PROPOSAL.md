# GitHub Hygiene Proposal — runbook §3.3 audit

**PROPOSAL — no GitHub mutations performed; author approval required.**

Repo: `JasperSeehofer/MasterThesisCode`. Snapshot taken 2026-08-11 via `gh issue list --state all --limit 100`, `gh pr list --state all --limit 50`, `gh api .../milestones?state=all`.

## Snapshot counts

- Issues (from `gh issue list --state all --limit 100`, 29 rows returned): **14 open** (#53, #52, #51, #44, #42, #41, #40, #39, #36, #27, #26, #25, #24, #23), **15 closed** (#38, #30, #29, #20, #19, #16, #15, #8, #7, #6, #5, #4, #3, #2, #1). Issue numbers are non-contiguous (gaps at 9–14, 17-18, etc. — likely PRs or deleted/renumbered items sharing the numbering pool with issues).
- PRs: 32 total, mostly merged; 2 open (#47 "paper: fill production-run numbers", #14 dependabot checkout bump).
- Milestones: 1 — **"Paper Submission"**, open, 5 open issues / 12 closed issues assigned.
- Stale-looking open issues (no recent activity signal from title/labels alone, needs author judgment): #27 (OPS cluster robustness, "Monday deploy" — dated language), #36 (LOW combine_posteriors artifact), #39 (blind alternative-truth mock, T-1).

## #4 finding (wCDM w0/wa guard)

**Issue #4 is already CLOSED** (state: CLOSED, labels bug/physics/design-choice) — it is *not* an open discrepancy. The runbook's "wCDM guard PR state" question is answered:

- The guard exists in code today: `master_thesis_code/physical_relations.py` has `_reject_unsupported_wcdm()` (line 36) called from `dist()` (line 174) and other entry points (lines 214, 252), raising `NotImplementedError` when `w_0 != -1.0 or w_a != 0.0`.
- It shipped in commit `8c789a66` ("fix: guard analytic distance against silent wCDM misuse (PHY-01/02, #4)"), part of the 2026-07-04 code-review PR (see also `docs/reviews/CODE-REVIEW-20260704.md`, and follow-up doc/changelog commit `28090542`).
- **No action needed** — code and tracker are in sync. This is a confirmation, not a finding.

## #53 finding (3σ window)

Issue #53 ("`get_redshift_outer_bounds` hardcodes 3σ, call site requests 2σ") is **OPEN** and is a **near-duplicate of issue #25**, which is also OPEN, in the Paper Submission milestone, and describes the identical defect (same file/lines, same call-site mismatch) from the 2026-07-04 code review (PHY-03). #53 (opened 2026-08-05) adds: (a) independent bit-exact verification via `m4_adjudicate.py`, (b) an explicit "no fix without author ruling" framing since the window width is a physics call affecting the candidate-host set.

The runbook's "standing open queue" entry (`RUNBOOK_NEXT_SESSION_9.md:36`) lists "issue #53 (3σ window)" as still open — consistent with the tracker; no drift there. The only hygiene gap is the **undeclared duplication** between #25 and #53.

Note: an unrelated internal claim-set label "#53" also appears in `STATE.md:51` (2D-bias claim set C3/C5/C7/C8 from the 2026-07-30 gates) — **not the same #53**, purely a coincidental numbering collision between the GitHub issue tracker and an internal claim ledger. Flagging so it isn't conflated in future sessions.

## Paper Submission milestone vs. ledger rows 96–98

Ledger rows 96–98 (`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`) document three CLOSED research threads:
- Row 96: cross-term closure (H-1's leg CLOSED by measurement; M-2's 2D residual EXCLUDED as cross-term-caused).
- Row 97: M-2 residual DISSOLUTION (author-ratified 2026-08-08) — thread 16 CLOSED.
- Row 98: calibration gate v2 — GATE TRUSTWORTHY but surfaced a **new open** defect (1D rail does not reproduce in the multi-candidate ball venue; thread 17 OPENED, venue-transfer investigation now running).

None of rows 96–98 map cleanly onto an *existing* open GitHub issue number by content (they are internal M-2/cross-term/calgate threads, not filed as issues at all — consistent with CLAUDE.md's rule that internal GSD/GPD verification checklists don't need issues). However, two adjacent items **do** show tracker drift, found while cross-checking:

- **Issue #40** ("generator_marginal bundles δ-kernel host-z numerator") is still OPEN, but its sub-items were split into PR #48 (`feat/hostz-kernel-decomposition`, issue #40a, MERGED) and PR #50 (`physics/hostz-pv-counted-once`, issue #40b, "ratified", MERGED). Reading #40's body, it has three actions: (a) decomposition flag — done via #48; (b) derive the photo-z/PV kernel for real-data mode — PR #48's title says "PV/photo-z derivation **skeleton**", i.e. partial; (c) paper methods must scope precision claims as mock-internal — no evidence found that this landed. **Not fully closeable** — a judgment call for the author on whether to close #40 and split the remainder into a new issue, or keep it open with an updated checklist.
- **Issue #44** (information-floor pre-registration for low-information venues) — no evidence in rows 96–98 or elsewhere that this was addressed; still genuinely open.

## Track B (Track B / perf work) — proposed new issues

Two pieces of shipped-or-approved Track B work have no tracker presence:

1. **phi(M) two-segment affine evaluation** (commit `87c6670b`, "PENDING AUTHOR RATIFICATION") — a `/physics-change`-gated performance optimization to `dark_mass_density_per_mass` (1.42x wall-time speedup), adversarially verified (worst in-band deviation 1.8e-15), with regression pins and gate-ledger rows appended. Per CLAUDE.md's GitHub-sync mandate this is exactly the kind of shipped physics-adjacent work that should have a tracker record, especially given its ratification-pending status.
2. **Route 1 Hermite-order reduction** — described in `results/venue_transfer_20260811/perf/PERF_ROADMAP.md` as part of the author-approved efficient-realistic-venue mission; harvesting script exists (`route1_study/harvest_route1.py`) but no completed/shipped commit was found matching "Route 1" by name in git log — appears to be **approved but not yet executed/shipped**. If so, an issue serves as a placeholder/tracking item rather than a "sync a done thing" action.

---

## Proposed actions table

### (a) Unambiguous syncs — code/PR state verifiably resolves the issue text

| # | Action | Command | Justification |
|---|---|---|---|
| — | None required for #4 | — | Already CLOSED; guard verified present in `physical_relations.py:36-52,174`. Listed here only to confirm no action, per task instructions. |

*(No fully unambiguous "close this, it's done" cases were found beyond #4, which is already closed. Everything else below involves a judgment call — either about scope, duplication, or physics-decision ownership — so nothing is proposed for the (a) bucket beyond that confirmation.)*

### (b) Judgment calls for the author

| # | Action | Command | Justification |
|---|---|---|---|
| #25 / #53 | Mark #53 as duplicate of #25 (or vice versa — #25 is older, in-milestone, and has the two-step fix plan already scoped; #53 has the independent bit-exact re-verification) | `gh issue comment 53 --body "Duplicate of #25 — same file/lines (physical_relations.py get_redshift_outer_bounds), same call-site mismatch, same 2σ-vs-3σ physics-decision gate. Verification from this issue folded in as supporting evidence."` then `gh issue close 53 --reason "not planned"` (or close #25 instead — author's call) | Both open, same file:lines, same root cause, same "author physics ruling required" framing. Duplication risk: someone fixes one and leaves the other open. |
| #40 | Update #40's checklist to reflect (a) done via #48, (b) partial ("skeleton") via #48, (c) still open; optionally split (c) into a new issue for the paper-methods scoping language | `gh issue comment 40 --body "(a) decomposition flag: DONE, PR #48. (b) photo-z/PV derivation: PARTIAL — #48 shipped a skeleton only; #50 (ratified) handles PV double-counting specifically. (c) paper-methods mock-internal scoping: NOT YET DONE. Recommend keeping open, scoped to (b)+(c)."` | #40 body has 3 explicit actions; only (a) and part of (b) are verifiably shipped. Closing outright would lose (c), which is paper-blocking per the issue's own labels. |
| #44 | No sync action found — confirm still genuinely open (information-floor criterion never pre-registered) | — (no command; flagging as confirmed-still-open) | Searched ledger rows 96-98 and adjacent runbook/results dirs; no pre-registration of an information-floor criterion was found. |
| #47 (PR) | Open PR "paper: fill production-run numbers (22/24 pending markers)" — consider whether venue-transfer findings (thread 17, calgate v2 coverage defect) block or inform the remaining 2 markers | (no command — informational; author should check PR #47 body against thread 17 status) | Row 98's "paper #47 hold's reason upgrades from 'P–P leg missing' to 'P–P leg FAILED — coverage DEFECT'" directly references this PR; worth an author comment there but that is itself a judgment call on messaging, not a mechanical sync. |

### (c) Proposed new issues

| Draft title | Labels | Milestone | Body sketch |
|---|---|---|---|
| `[PHYSICS] phi(M) two-segment affine evaluation — ratification tracking (commit 87c6670b)` | `enhancement`, `physics` | Paper Submission (or none, author's call — it's a perf item, not a paper-blocker per se) | "Track B perf item, branch perf/realistic-venue. Two-segment affine approximation to `dark_mass_density_per_mass`'s ln φ(M) on the Babak band; adversarially verified (worst deviation 1.8e-15, 14 ULP); 1.42x wall-time speedup (124.92s→88.03s per seed). Commit 87c6670b is marked PENDING AUTHOR RATIFICATION. This issue tracks that ratification and closes on author sign-off (or reversion if not ratified). See `results/venue_transfer_20260811/perf/PERF_ROADMAP.md` §5." |
| `[PERF] Route 1 Hermite-order reduction — status/tracking` | `enhancement` | none / TBD | "Author-approved mission item (efficient realistic-venue campaigns, see MEMORY.md 'Realistic-venue performance goal', 2026-08-12). `route1_study/harvest_route1.py` exists for data harvesting but no shipped commit implementing the order reduction was found as of this audit (2026-08-11). Opening for tracking; close or update once implemented and verified." |

---

## Top-3 summary for the author

1. **#4 is fine** — the wCDM guard is closed and the code matches (`_reject_unsupported_wcdm` in `physical_relations.py`). No action.
2. **#25 and #53 are duplicates** — same defect (3σ hardcode vs 2σ requested in `get_redshift_outer_bounds`), same "needs author physics ruling" gate. Recommend closing one as duplicate-of-the-other; author picks which survives (suggest keeping #53 since it carries the bit-exact verification, closing #25 as superseded — or the reverse, since #25 is milestone-tagged and has the scoped two-step fix plan).
3. **#40 needs a partial-progress comment, not a close** — PRs #48/#50 shipped the decomposition flag and PV fix, but the issue's own action (c) (paper-methods scoping language) is unaddressed and paper-blocking; closing it now would silently drop that requirement.

Track B (phi(M) swap, Route 1) currently has zero tracker presence despite CLAUDE.md's GitHub-sync mandate — recommend the two new issues above, sized to the work's actual state (ratification-pending vs. approved-not-yet-shipped).
