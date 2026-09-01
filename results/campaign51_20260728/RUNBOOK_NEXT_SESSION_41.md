# Runbook 41 — after the flip landing + graph-1 charter draft (supersedes runbook 40)

**Read first.** SESSION 2026-08-31/09-01 is CLOSED as of rows #278–#289. The `[PHYSICS]` A18 flip
is in production (`5e7fda16`); the [HIER] instrument is now certified on both axes at the full
T1.3 configuration; wave 3's A14 delta is confirmed twice not-material; the B8.2 S3 pilot is read
out (pre-flip, REPORTED-ONLY, with routed defects); the archive redundancy pass was re-run the
evening of 2026-09-01. The centerpiece of this runbook is **Research Graph 1**, a directed-graph
charter for the next batch that is drafted and AWAITING AUTHOR RATIFICATION — nothing in it runs
until the author grants it. This runbook is the fresh-session entry point.

## 0. State of record (2026-08-31 → 2026-09-01, rows #278–#289)

**The flip (row #286, carried from runbook 40).** `[PHYSICS]` commit `5e7fda16`:
`catalogue_leg_1d_mass_aware` production default flipped to `"auto"`. A18 arm verdict
Z-CONFIRMED, 1D map_h 0.665 / mean_h 0.66699, inside the registered and measured bands. The
residual 1D rail (mean 0.667 vs truth 0.73) is OWNED: attributed to the mass-blind/mass-aware
mismatch, now B8 [CAL]'s centerpiece object rather than an open bug.

**Row #287 — [HIER] certified on both axes at T1.3.** Runner-11's 8-cell b-node pair (seeds
900101–900104 × b±, divisor-on + z-window-on zk4, sky 1.5) crashed in post-hoc scoring on a
latent driver bug (`gate_eng` unconditionally read `all_nodes["truth"]` on a b-only node set;
NOT flip-related — forensics confirms the driver pinned `catalogue_leg_1d_mass_aware="off"`
explicitly and the process predates the flip commit). Fixed (subagent-built): `gate_eng` degrades
with `eng_available=False` on a missing truth node, mirroring the existing per-axis pattern;
regression test `test_gate_eng_handles_a_b_only_node_dict_with_no_truth_node` added, 15/15 pass.
Zero-compute rescore of the 8 banked cells: score_b(no-BH) = −0.862 ± 0.477, Z_b = −1.808, n=461;
score_b(with-BH) = +0.317 ± 0.410, Z = +0.773 — both inside |Z| ≤ 3. Consequence: the [HIER]
instrument is certified on both axes at T1.3 by direct measurement; S0-B's remaining precondition
is the PA-HIER-33 scorer implementation + the driver's missing iiib venue path.

**Row #288 — B8.2 S3 pilot complete + read out (REPORTED-ONLY, bands-not-verdict).** Runner-9
end-to-end (N-ladder 106/400/1588; cell S 63/100 universes at N=200, wall-limited; cell T 20/25,
wall-limited); all rc=0. Measured cell S, N=200, **PRE-FLIP estimator** (long-lived processes
predate `5e7fda16`): no-BH σ_h,harness = 0.03853, F = 7.43 vs the rescaled floor, PIT-KS D =
0.8045, HPD coverage 50/68/90/95 = 0.015/0.015/0.061/0.121 (all far out of band); with-BH σ_h =
0.05887, F = 11.35, KS D = 0.3313, coverage 0.364/0.470/0.803/0.894 (out of band). Open items
routed to the S4 registration review, before any S5: (a) cell-S aggregate CONTAMINATED — pools 3
mixed-N ladder seeds with 63 N=200 seeds; (b) cell T never aggregated (raw checkpoints exist, the
T0/T-vs-S control read is ABSENT); (c) both cells stopped on `--max-wall-s`; (d) fresh [RULE] to
the author — whether S3 re-runs post-flip, since the flip changes the no-BH channel this pilot
measured and pre-flip coverage numbers cannot calibrate the post-flip production stop rule (this
is `d-s3-rerun` in the graph-1 draft, §1 below).

**Archive redundancy pass (2026-09-01 evening).** Re-run of `results/_archive/archive_run_wave2.sh`.
The c1 SKIP outcome is legitimate — that item was never launched, not a re-detection of the
row #288 ssh-failure-read-as-not-found defect. That defect (fix owed) is still open; see §5.

**Rows #278–#289 in full** are in `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`.

## 1. THE CENTERPIECE — Research Graph 1 (AWAITING AUTHOR RATIFICATION)

Files, all in `results/campaign51_20260728/realistic_20260729/graph1_20260901/`:

- **`INFRA_AUTONOMOUS_RESEARCH_PROPOSAL_20260901.md`** — the new infrastructure model replacing
  the linear-tree/runbook pattern: typed nodes (register/build/measure/derive/verify/read/
  checkpoint/subgraph, plus gate/decide events), a claim layer separate from the execution layer,
  three edge families, convergence-node eligibility computed mechanically (not from prose), bounded
  re-entry via counted revision nodes (never back-edges), a standing gate panel law, and explicit
  anti-derailment/anti-reward-hacking structure. Diagnoses 11 things that demonstrably worked in
  rows #221–#288 (pre-registered bands, byte-identity gates, score-at-truth nulls, etc. — keep as
  invariants) and 11 structural failure modes (verdict write races, SSH-failure-as-not-found,
  mixed-N contamination, storage-precision artifacts masquerading as physics forks, etc. — each
  gets a named structural fix).
- **`RESEARCH_GRAPH_1_PROPOSAL_20260901.md`** — the graph itself: branches A–H (+ a closure
  chain), 9 execution branches total, depth 3–4 nodes per branch before first convergence,
  converging on six fresh-RULE decide nodes plus three terminal paper decide nodes. Objective:
  count of registered questions moved to SETTLED (verified/refuted/bounded-undetermined, all
  panels green or waived) — bias reduction is explicitly NOT the objective (author's binding
  2026-08-05 value). Contains the initial-decisions table, rows 0–11 (row 0 = charter
  ratification; subsequent rows = per-branch-head grants).
- **`GRAPH1_CRITIC_NOTES_20260901.md`** — adversarial pass, 10 findings (4 MUST-FIX, 4 SHOULD, 2
  NOTE). All 4 MUST-FIXes have been applied as **REVISION 1** of the graph/infra pair: (1) the
  approval-scope semantics of infra §3.4 is now its own separately-tagged `[STANDING]` row rather
  than folded into row 0's ratification prose; (2) `max_revisions=2` on the four register nodes is
  now explicitly tagged `ORCHESTRATOR-DERIVED` (unsourced) rather than presented as if load-bearing
  and derived; (3) `d-s3-rerun` (row #288 item (d)'s fresh RULE) is now a properly typed `d-` node
  in §1 with a requires-manifest, not a bare table row with no graph representation; (4) a fourth
  MUST-FIX (see the file for the remaining item) is likewise folded in.
- **`STATE_AND_CANDIDATES_20260901.md`** and **`external_research_2_workflow_structures.md`** —
  supporting state extract and external-survey input, both cited by number/source throughout the
  two proposals (no orchestrator-derived number is uncited).

**Ratification mechanics.** The author ratifies via the decisions table (row 0 = charter; each
subsequent row = one branch head or one fresh RULE). The approval-scope rule binds fully here: a
single "Approved" on row 0 covers ONLY row 0 and whatever is explicitly listed with it — it does
NOT retroactively cover branch-head rows, and it does NOT grant the `[STANDING]` approval-scope-as-
ratification-semantics row unless the author grants that STANDING item explicitly and separately
(this is itself MUST-FIX #1 above, already corrected in the draft — do not let a "ratify the
charter" grant silently re-absorb it). Nothing whose inputs do not yet exist is covered by any
blanket approval (repo-wide binding default, CLAUDE.md "Approval scope").

## 2. First actions on ratification, in order

1. **`d-s3-rerun`** (the fresh RULE from row #288 item (d)) + **Branch A** (S4 harness repair):
   resolve whether S3 re-runs post-flip, then execute the harness repair so B8.2's F/coverage
   numbers can be regenerated on the post-flip estimator. Branch A's inputs (rows #286/#288) already
   exist, so this item can run immediately upon its own row's grant, ahead of the rest of the wave.
2. **Branch D**: the runner-11 read is DONE (row #287) — this branch's head starts directly at the
   PA-HIER-33 scorer implementation + the driver's missing iiib venue path build (S0-B's remaining
   precondition, per row #287's own consequence statement).
3. **Branch B**: post-flip HEAD re-baseline, on cluster (subject to the /cluster preflight gate,
   §4 below).
4. Remaining branches (C, E–H, the closure chain) proceed per the graph's own wave sketch in
   `RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §4 (mermaid diagram) once their gating decide-nodes are
   granted — do not hand-sequence them from this runbook; the graph is the source of sequencing
   truth per the infra proposal's own diagnosis (§1.3: "graph state simulated in narrative" is the
   failure mode this replaces).

## 3. Operating mode of record (author directive, 2026-08-31, unchanged)

The orchestrator delegates ALL file writes and runs to subagents; the orchestrator's own role is
limited to review, adjudication, and committing — never write files or execute the registered
measurement itself. **Gate panel law**: no science is read without green (or explicitly waived)
gate stamps — a reader agent that returns a number without its gate stamp is not evidence.
**Ledger rows for everything** — every executed node, gate result, and decision gets a row in
`BIAS_HISTORY_LEDGER.md`, quoted verbatim from the artifact that produced it (not itemized from
memory — row #268's lesson).

## 4. Standing gotchas (carried from runbook 40, still true)

- `pkill` self-match on bracket patterns when the search path appears in your own command line —
  kill by PID/PGID instead.
- `rsync -a` does not dereference symlinks; use `-L` + an md5 manifest verified both ends.
- Repo-root `simulations` symlink is REMOVED (`/tmp/seed600_local` target gone). T8 sky-selection
  test skips until that pool is regenerated — do not chase the flip to "fix" T8.
- Stale pre-rename `.pyc` caches can show phantom `MasterThesisCode` paths — clear `__pycache__`
  if a path search returns something not in the working tree.
- Verifier output is evidence, not authority · subagents never run the registered measurement they
  built · never end a turn to wait on an untracked process · per-poll SSH, Monitor for watchers ·
  every submission stamps its authorization · exoneration grep is for the MECHANISM, not the tag ·
  a null offset derived for one estimator configuration does not transfer to another — pin every
  null to the arm's own likelihood structure · `--jobs>1` is dead in `hier_s0_driver.py`, always
  launch `--jobs 1` · SSH `ControlPersist` is 8 h and OTP-gated · `np.savez` silently appends
  `.npz` to a tmp path that doesn't already end in `.npz` — name tmp files ending `.npz` before the
  atomic replace.

### New this session

- **`archive_run_wave2.sh` reads an SSH failure as "not found on cluster"** — a conflation defect
  in its existence check (row #288). Fix owed: distinguish present / absent / unreachable
  (three-valued existence — this is exactly INFRA §1.2 item 9's named structural fix, already
  designed into the graph-1 claim/gate model). Until fixed, treat any "not found" result from this
  script as suspect if the ssh session could have expired.
- **`ControlMaster` expires ~8h** — the author must run `ssh bwunicluster true` to re-authenticate
  (OTP-gated) before any cluster item in §2 above can launch; check this before assuming a cluster
  submission will go through.
- **The h-prior upper bound blocks h > 0.86 grid nodes** — `cosmological_model.h.upper_limit` is a
  physics-trigger constant; any change to it requires a full `/physics-change` gate. Not currently
  blocking anything in flight, but any future grid design that wants nodes above 0.86 must route
  through the gate rather than editing the constant directly.

## 5. Open author words (not yet given)

- **Appendix-B scope word** — carried unresolved from runbook 40; still the one item neither the
  independent cross-check artifact nor this session's own recommendations took a position on.
- **Falsifier (ii)** — the class-G fleet rung, now formally covered as **Branch E**'s head in the
  graph-1 draft (question node `q-a4-provisional`); A4 remains ratified-with-cap, PROVISIONAL,
  until this falsifier runs.
- **`d-s3-rerun`** — see §2 item 1; this is a fresh RULE, not yet put to the author, gating whether
  Branch A's post-flip S3 re-run happens at all.
- **The graph-1 charter itself (row 0)** — and each branch-head row in the decisions table,
  `RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §3 — is the primary open ask of this runbook.

**2026-09-01 ADDENDUM: RATIFIED.** The author ratified the entire docket ("all is ratified from the graph", ledger row #290): charter frozen, STANDING granted, d-s3-rerun ruled, branch heads A-I + row 12 approved, headcount collapse to 4 accepted. Begin at the first-actions section; every NOT-covered cell still returns as a fresh [RULE].
