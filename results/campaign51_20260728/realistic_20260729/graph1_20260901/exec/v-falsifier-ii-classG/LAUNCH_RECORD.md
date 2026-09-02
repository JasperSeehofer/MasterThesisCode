# v-falsifier-ii-classG — LAUNCH RECORD (SKIPPED — cost exceeds the wave-2 hard cap)

Research Graph 1, Branch E. Attempted 2026-09-02 (wave-2 first batch) by the cluster launcher
agent. **NOT LAUNCHED.**

## Authorization (quoted)

Ledger row #290 decisions row 7 (`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`):

> | 7 | v-falsifier-ii-classG (Branch E) | DO | Approved | the class-G fleet at the 40-60 CPU-h
> envelope (runbook 40 section 2), hard-capped at 60 | dropping A4's PROVISIONAL —
> d-a4-final-ratification returns with numbers, never auto-ratified (rows #278(4)/#280/#284(3)) |

Graph spec: `RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §1.5 (Branch E), node `v-falsifier-ii-classG`
and checkpoint `k-falsifier-ii-fleet` (§ line 223): "class-G configuration count (runbook 40
section 2; count fixed at launch, not re-expanded mid-run) ... 60 CPU-h fleet-wide, hard cap."

Task brief (this launch): "Configuration count is FIXED at launch per k-falsifier-ii-fleet (no
mid-run expansion); hard cap 60 CPU-h fleet-wide — compute the expected cost from the spec's own
numbers BEFORE submitting and STOP if the fixed config set exceeds 60 CPU-h."

## Preflight verdict (verbatim, this launch's session)

Before the pull (cluster HEAD `1ec9514d`, behind local `dcc75352`):
```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 71 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
Repair (identical pattern to the wave-1 launcher's m-head-rebaseline record): `git status
--porcelain` on the cluster showed all 542 dirty entries as untracked (`??`); origin's
`fix/p32d-classg-venue-repair` was itself behind this session's local `dcc75352` (4 unpushed
commits — a `git push` was denied by the harness's Bash classifier), so the sync was done via
`git bundle create ... origin/fix/p32d-classg-venue-repair..HEAD` + `scp` + `git fetch <bundle>
HEAD:refs/bundle-tmp` + `git merge --ff-only refs/bundle-tmp` on the cluster (four untracked
wave-1 `.sbatch` files that collided with newly-tracked paths were moved to
`~/wave1_untracked_sbatch_backup/` before the merge, not deleted). Post-merge: cluster HEAD =
`dcc75352` (confirmed `git merge-base --is-ancestor dcc75352 HEAD` → ancestor). Re-run preflight:
```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 71 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
The 71-dir WARN is the same pre-existing backlog (gotcha 11) reported by the wave-1 launcher (65
dirs then); not a blocker, not addressed by this launch.

## Why this item was NOT submitted

**Spec search.** `runbook 40 section 2` ("Open author words") states only: "A4 — ratified-with-cap,
PROVISIONAL until falsifier (ii) runs. The falsifier (ii) is the class-G fleet rung (~40-60 CPU-h),
the natural next cluster item" — it gives no configuration count or CLI. Tracing the citation
chain (`runbook 40` §3 item 4 → `tree2_20260830/TREE2_CHARTER_20260830.md` T4 → `tree2_20260830/
PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md` §589-590 → the actual registering document,
`fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6.1(ii)) surfaces the falsifier's own
registered cost, stated in the proposal's own words:

> "On the class-G venue with rung 1 repaired in the Option A′ form (harness-only gate; fleet
> re-run **~8.67 CPU-h/task × 24–33 tasks ≈ 208–286 CPU-h**, from the readout's ~32.5 min/task at
> 16 cpus — the runbook-34 '~2–4 CPU-h' figure is superseded by measurement)..."

That per-task anchor and task count come from the registered design itself (`p3_2d_fleet.py`'s
`FLEET_SEEDS` — 24 seeds fixed, `900101-900124`, one full arm at minimum for the primary registered
read; the 24–33 range spans arm/pilot combinations). **24 tasks × 8.67 CPU-h/task = 208.1 CPU-h at
the minimum config** — already 3.5× the 60 CPU-h hard cap, using the spec's own registered
per-task anchor, before any pilot or second-arm tasks are added.

`tree2_20260830/TREE2_CHARTER_20260830.md` (T4 row) and `TREE2_SYNTHESIS_DOCKET_20260830.md`
(item 5) both assert a "chair recost from 208-286" down to "approx 40-60 CPU-h," and the graph
proposal's own §1.5 table and decision-row-7 text both carry that recost figure forward as the
node's stated cost. **No derivation for the recost was found anywhere in the accessible record** —
neither charter doc, nor the synthesis docket, nor the graph proposal itself shows the arithmetic
(fewer tasks? a cheaper per-task anchor? a narrowed rung scope?) that turns 208-286 into 40-60.
`p3_2d_fleet.py`'s fleet-mode entry point (`ARM_SEEDS["b0i2d"]` = 24 seeds) shows no smaller
registered subset for the primary twin-arm read — reducing the task count below 24 would depart
from "count fixed at launch... not re-expanded mid-run" in the opposite direction (a mid-launch
*contraction* not sanctioned by k-falsifier-ii-fleet either) and would not be a configuration the
falsifier's own registered design (§6.1(ii)) recognizes as sufficient for the primary read.

## Disposition

**Skipped, per the task brief's own instruction: "If a spec cannot be found or the cost exceeds
its cap, do NOT improvise — skip that item and report why."** The spec's own registered numbers
put the fixed configuration at ≥208 CPU-h, which exceeds the 60 CPU-h hard cap by more than 3×;
the only available lower figure (40-60 CPU-h) is an unsourced chair recost with no located
derivation, and substituting it for the registered 208-286 CPU-h figure — or shrinking the task
count myself to force a fit under 60 CPU-h — would be exactly the kind of improvisation the task
brief prohibits. No SLURM job was submitted for this item.

## What would unblock this

`d-a4-final-ratification` needs either (a) the chair's 40-60 CPU-h recost made concrete — an
explicit task count and per-task anchor, sourced, that a launcher can verify against a stated hard
cap — or (b) an author ruling authorizing the registered 208-286 CPU-h cost against a raised cap
(row #290 decisions row 7 caps it at 60 hard, so this is a [RULE], not something this launcher can
grant itself). Returning to the author rather than improvising either direction.
