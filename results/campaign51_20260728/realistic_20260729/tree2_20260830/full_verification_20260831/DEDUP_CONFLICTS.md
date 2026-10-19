# NODE dedup/conflict pass — full verification, both trees

Date: 2026-08-31. Source read: verdicts_all.json in this directory (read verbatim, from
source, no cached restatement). No cluster access used, no commits, no code edits, nothing
touched under b8_cal_harness_work_ladder or runner-9.

## 0. Primary finding: verdicts_all.json is critically incomplete relative to the visible work

verdicts_all.json currently contains **1 record** (item T1-1, tree1, verdict "confirmed").

The work/ subdirectory in this same folder holds rederivation scripts and/or numeric output
files for **42 distinct item IDs**: T1-1 through T1-19 (19 items), T2-1 through T2-17 (17
items), and D-1 through D-6 (6 items). That is the footprint of a full both-trees verification
pass — only one of those 42 items' verdicts made it into the shared aggregation file.

Timestamp check (mtimes, not content, but decisive on ordering):
- Oldest file in work/: bs_ff230621.py at 09:42:24.
- Newest file in work/: T2-14_pair_compare_out.json at 10:43:15.
- verdicts_all.json itself: mtime 10:47:45 — i.e. it was last written 4+ minutes AFTER every
  piece of work-directory evidence for all 42 items had already been produced.

Conclusion: this is not a case of "I read the file too early, other agents are still working."
The computation for all 42 items appears to have already happened by the time verdicts_all.json
was finalized, yet 41 of the 42 verdicts are absent from it. The most likely mechanism is a
lost-update race: multiple parallel verifier subagents each read-modify-wrote the same JSON
array without coordination, and the last writer's version (carrying only its own item, T1-1)
clobbered every prior writer's contribution instead of the array being a true union.

This is reported as a **governance breach** in its own right (process, not physics): the
full-verification docket as currently persisted materially understates the work done, and any
downstream synthesis that treats verdicts_all.json as "the 42-item verification record" would be
wrong by 41 items. I did not attempt to reconstruct the missing 41 verdicts from the raw
per-item work/*.json numeric fragments (e.g. T1-5_rederive.json, T2-4_rederive_out.json) — those
are intermediate computation dumps (bare numeric keys, no verdict/title/method/discrepancy
fields), not verdict records, and re-authoring 41 verdicts from them would be fabrication rather
than dedup/conflict analysis. The correct fix is procedural: re-collect each subagent's actual
verdict text (from its own transcript / return value) and append-merge into verdicts_all.json
before this docket is treated as complete.

Item IDs with work-directory evidence but NO entry in verdicts_all.json (41):
T1-2, T1-3, T1-4, T1-5, T1-6, T1-7, T1-8, T1-9, T1-10, T1-11, T1-12, T1-13, T1-14, T1-15, T1-16,
T1-17, T1-18, T1-19, T2-1, T2-2, T2-3, T2-4, T2-5, T2-6, T2-7, T2-8, T2-9, T2-10, T2-11, T2-12,
T2-13, T2-14, T2-15, T2-16, T2-17, D-1, D-2, D-3, D-4, D-5, D-6.

## 1. Duplicated findings

None possible to assess beyond the trivial case: with only 1 record present, there are no
duplicate entries in verdicts_all.json (a duplicate requires at least 2 records addressing the
same claim). Not evaluated for the 41 missing items since their verdict text does not exist in
the file to compare.

## 2. Conflicts (two items whose re-derived numbers or verdicts contradict)

None found. A conflict requires at least 2 records in the same file; with n=1 no conflict is
possible to detect from verdicts_all.json as it stands.

## 3. Vague-method flags (a verifier that did not re-execute from source)

T1-1: NOT flagged. Its method field names an exact commit (git show ff230621), specific line
spans read at that commit and at HEAD, an ast parse of keyword defaults, a regex pass over the
raw run log, a pandas recomputation of GATE PARITY from two raw event_likelihoods.csv files (both
positionally and via an event_idx keyed join), an independent re-hash of the driver against the
record's pinned sha1, and explicit git diff --stat/--numstat checks for append-only-ness across
three commits. This is genuine from-source re-derivation, not restatement of a prior record. It
also self-reports 8 real discrepancies (citation off-by-ones, a costing-consequence
overstatement, a forward-drift note, an unqualified ledger row) rather than rubber-stamping —
consistent with a falsification-oriented pass, and caps_carried=True with governance_breaches=[]
for this item specifically.

No other items can be assessed for method vagueness because their verdict prose is absent from
verdicts_all.json (see section 0).

## 4. Union of governance breaches

From the single present record (T1-1): governance_breaches = [] (empty — none reported by that
verifier for its own item).

Docket-level governance breach identified by this dedup pass itself (not carried from any
verifier's own reported list, so stated separately rather than folded into the empty union
above): the verdicts_all.json aggregation lost 41 of 42 items' verdicts despite all 42 items'
underlying computation having already completed, per the mtime evidence in section 0. This
should be treated as an open governance item requiring re-aggregation before the full
verification docket for both trees is considered synthesized or acted on.

## Counts (verbatim summary)

- Records present in verdicts_all.json: 1 (T1-1, tree1, verdict = confirmed)
- Items with work-directory evidence but missing from verdicts_all.json: 41
- Duplicated findings: 0 (not assessable beyond n=1)
- Conflicts: 0 (not assessable beyond n=1)
- Vague-method flags: 0 (T1-1 is from-source; 41 items unassessable, verdict text absent)
- Governance breaches reported by the present verifier: 0
- Governance breaches identified by this dedup pass (aggregation/process): 1 — the
  verdicts_all.json lost-update gap described in section 0
