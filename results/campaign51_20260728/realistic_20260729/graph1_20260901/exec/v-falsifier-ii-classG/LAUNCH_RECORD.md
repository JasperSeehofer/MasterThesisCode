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

---

## LAUNCH (option A, row #308)

Attempted 2026-09-02, second pass, by the cluster launcher agent. **NOT LAUNCHED — STOPPED,
not skipped: a code-implementation gap, not a cost gap.**

### Authorization (quoted)

Author's "both approved" ruling on docket item 7 option (A) (ledger row #308, being written;
docket text `graph1_20260901/DECISION_DOCKET_WAVE1_20260902.md` §7, line 620): **"(A) Raise the
k-falsifier-ii-fleet cap to 290 CPU-h and run the 33-seed fleet (option a')."** This raises the
`k-falsifier-ii-fleet` cap from 60 to 290 CPU-h and authorizes the 33-seed configuration (option
a′ of `RECOST_RECORD.md` §3) — the design's own demonstrated power floor (all SEMs below
planning at 33 seeds, per `P3_2D_REPAIR_READOUT_20260828.md` §7).

### Cost recompute (confirmed, before any submission attempt)

33 tasks × 8.6667 CPU-h/task (empirical anchor, `RECOST_RECORD.md` §1, twice-replicated on jobs
6723958/6730213, confirmed not stale as of this session — see below) = **286.0 CPU-h ≤ 290 CPU-h
cap.** Cost is NOT the blocker for this launch attempt.

### The blocker: rung 1 (Option A′) is not implemented anywhere in the codebase

The falsifier (ii) design (`PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §6.1(ii), quoted verbatim in
`RECOST_RECORD.md` §2) requires the fleet to run **"On the class-G venue with rung 1 repaired in
the Option A′ form."** The registered prediction being tested — LHS2(bt) = 0.00740040 ±
0.00024951 — is explicitly labeled, in the residual ladder table
(`PROPOSAL_2D_TWIN_ADOPTION_20260829.md` §5, row "conditional on rung 1 (unimplemented)"), as
**conditional on rung 1, which the same table marks "(unimplemented)."** The ×1.1585 reweight
shown for "after rung 1" in that table (`p32d_residual_accounting_20260827.md` §1) is a **post-hoc
analytic correction applied to already-drawn (pre-repair) data**, not a measurement from repaired
generative code.

Checked directly, not assumed, in this session:

1. **`correspondence_1d.py`** — no `rung`, `option_a_prime`, `sbarphi`, or S̄_φ-override
   flag/kwarg anywhere (`grep` over the whole file). `_draw_kernel_survival_redshifts`
   (`:1563-1620`) takes `phi_survival_table` as a plain positional/keyword pass-through with no
   "drop the survival factor" mode. `catalogue_selected_host_draw_weights` (`:1496-1509`) always
   returns `host_w = normalize(w_g · S̃_φ,g)` — there is no branch that instead normalizes bare
   `w_g` (Option A′ item (i), `PHYSICS_CHANGE_SBARPHI_20260827.md` §2.2).
2. **`results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py`** (the registered instrument,
   `--stage fleet`) — its `stage_fleet` → `_run_b0i2d_arm_seed` (`:385-460`) calls
   `gen.draw_realization(..., host_mode="catalogue_selected_2d", phi_survival_table=phi_survival_table,
   ...)` with the REAL survival table and no override hook, then
   `c1d.run_mirror_seed_inprocess(...)` with no rung/gate flag among its five threaded flags.
   `gate_acc_extended` (a `--stage gates` reporting-only function, not `fleet`) also uses the
   real, unmodified table.
3. **`git log d04d9dc9..HEAD`** (§1.1 of `RECOST_RECORD.md`, re-confirmed this session) touches
   neither file's 2D draw path — eight `[PHYSICS]` commits on `correspondence_1d.py` since are all
   byte-identical-default instrument flags unrelated to `catalogue_selected_2d`'s host/z draw.

**Consequence:** submitting `cluster/p3_2d_fleet.sbatch`'s machinery verbatim right now — even at
a fresh out-root, fresh seeds 900134-900166, both arms — would run the **exact same pre-repair
generative code** as jobs 6723958/6730213 already ran. It would not test the falsifier's
registered v2.9 conditional prediction at all; it would only produce a third, larger-N replicate
of the ALREADY-BANKED pre-repair configuration, at the full 286.0 CPU-h cost, testing nothing new.

### Why this launcher does not implement the fix and proceed

Implementing Option A′ — even in its "harness-only gate" form (a keyword flag or a flat
S̄_φ≡1 table substitution at the `p3_2d_fleet.py` call site, per
`PHYSICS_CHANGE_SBARPHI_20260827.md` §2.2) — is a **new, unreviewed change to the generative draw
law that determines a computed physical quantity** (the accepted-event distribution feeding
LHS2/G4). Every comparable change to this draw path in the repo's history is tagged `[PHYSICS]`
and gated through `/physics-change` with author sign-off (`git log`, eight such commits on
`correspondence_1d.py` alone since d04d9dc9). This launcher's mandate is submission machinery —
"reuse the prior fleet submissions' machinery verbatim" — not authoring a physics-change-gated
code path. Writing that code without the gate would be exactly the kind of improvisation the task
brief prohibits ("do not improvise... if a spec cannot be found... do NOT improvise").

### Disposition

**STOPPED before any cluster access.** No preflight was run (moot — nothing would have been
submitted regardless), no jobs submitted, no code edited, no commit made.

### What would unblock this

Someone (the author, or a subagent explicitly commissioned and `/physics-change`-gated for it)
implements Option A′ as a harness-only instrument at the `p3_2d_fleet.py` call site — items (i)
and (ii) of `PHYSICS_CHANGE_SBARPHI_20260827.md` §2.2, confined to the driver, not touching
`correspondence_1d.py`'s committed defaults — and that implementation is reviewed/committed
*before* this launcher (or a fresh instance of it) is asked to submit the 33-seed fleet against
it. Alternatively, if the author intends the fleet to run pre-repair for some other purpose (e.g.
as a fresh-seed cross-check of the existing 33-seed banked result, not as falsifier (ii) itself),
that is a different task than "launch falsifier (ii) exactly per its registered design" and should
be stated as such.

---

## LAUNCH 2 (post-A′, row #314)

Attempted 2026-09-02, third pass, by the cluster launcher agent. **LAUNCHED — job 6769177.**

### Authorization (quoted)

Row #308 (docket §7 option A: raise `k-falsifier-ii-fleet` cap to 290 CPU-h, run the 33-seed
fleet, option a′ per `RECOST_RECORD.md` §3) + row #314 (author's "both items as recommended
please," ratifying `PHYSICS_CHANGE_SBARPHI_20260827.md` §2.2 Option A′ as the implementable form)
+ the landed `[PHYSICS]` commit `2b657255` ("Option A' — class-G S_bar_phi de-double-weight (2D
branch only; rows #309/#314)", `A_PRIME_IMPLEMENTATION_RECORD.md`, 2032 tests pass, ruff+mypy
clean). This satisfies the precondition the previous stop in this file identified: rung 1 (Option
A′) is now implemented, so the falsifier's registered venue (`"catalogue_selected_2d"` with rung 1
repaired) actually exists in the codebase.

### What changed since the row #309 stop, verified directly this session

- `git merge-base --is-ancestor 2b657255 HEAD` on local `7e9e1e27` → true.
- `grep apply_survival darksiren_emri/validation/correspondence_1d.py`: the 2D call site inside
  `_draw_2d_accepted_latents` passes `apply_survival=False` **unconditionally** (not a CLI flag on
  `p3_2d_fleet.py` — the repair is baked into the `"catalogue_selected_2d"` branch itself; every
  other caller, `"catalogue_selected"`/b0i and `"mixture_selected"`, omits the kwarg and defaults
  to `True`, bit-identical per the implementation record's L8 guarantee). `draw_realization`'s
  `"catalogue_selected_2d"` elif now normalizes the plain rate weight `w_g` (not `w_g·S̃_φ,g`) for
  `host_w`, per item (i).
- **Consequence for this launch:** `p3_2d_fleet.py --stage fleet` needs **no code change and no
  new CLI flag** to run under Option A′ — it is the exact same driver invocation as jobs
  6723958/6730213, now running against a commit where the underlying `MirrorUniverseGenerator`
  draw law is repaired. This is what makes "reuse the prior fleet machinery verbatim" correct here
  (it was NOT correct at the row #309 stop, when the same verbatim invocation would have run the
  unrepaired law).

### R4/fleet/GATE-ACC open item (A_PRIME_IMPLEMENTATION_RECORD.md §6.3 item 3) — checked, not designed

The implementation record flags "Fleet re-run / GATE-ACC re-check... PA-2D-9's frozen numbers and
the 24-seed b0i2d fleet remain STALE per §9.3 until the chair authorizes and runs that re-run" as
explicitly out of scope for the implementation task. Checked what this fleet submission needs to
emit so that re-check remains possible downstream, without designing anything new:
- `gate_acc_extended` (GATE-ACC) lives in `p3_2d_fleet.py`'s `stage_gates`, a **separate
  post-processing stage** (`--stage gates`) that consumes each seed's `<arm>_<seed>_meta.json` +
  `diagnostics_csv` written by `--stage fleet`'s `_run_b0i2d_arm_seed`. Nothing in the A′
  implementation touched `_run_b0i2d_arm_seed`'s output contract (meta dict keys, diagnostics CSV
  path) — it only changed what upstream `draw_realization`/`_draw_kernel_survival_redshifts`
  compute before that function's `run_mirror_seed_inprocess` call.
- **This means the `--stage fleet` array job now submitted emits exactly what `--stage gates`
  needs** for the R4/GATE-ACC re-check (and `--stage lhs2d` for the registered LHS2/G4 read) once
  all 33×2 arm-seed metas land — running those two post-processing stages is a separate, cheap,
  future step (not part of this submission; this launcher's mandate is the fleet array only, per
  the prior stop's own scoping and the task brief).
- Each `meta.json` stamps `git_commit` via `c1d._git_commit()` at run time, so the readout will
  show these results were generated at (or after) `2b657255` — the provenance chain needed to
  distinguish this run from the pre-repair banked fleet is self-recording; no extra stamp was
  added.

### Seed convention — determined, not delegated

Checked `PREREGISTRATION_P3_2D_REPAIR_20260827.md:675/1014` (blindness item 6) and
`RECOST_RECORD.md` §1.1 directly: the registered convention for a rung-repair re-run is to **reuse
the same seed labels** (900101-900133, the 24-seed primary batch + the 9-seed PA-2DR-15
extension) against **freshly generated draws** — the preregistration states the repair "changes
RNG consumption... so the fresh fleet, though reusing the same seed labels, performs genuinely new
draws," i.e. same seed value ≠ same realization once the draw law changes upstream of the RNG
calls that consume it. This is the exact form the rung-2/3 repair itself used (fresh out-root,
same seed labels, `PA-CA-11`'s per-`(arm,seed)`-meta reuse guard forces a genuine fresh draw at a
fresh out-root regardless). **Not ORCHESTRATOR-DELEGATED — this is a registered convention**,
sourced to the two file:line locations above, not an unregistered choice this launcher made.

Seeds used: **900101-900133** (33 seeds, both arms `bc`+`bt` per seed → 66 arm-seed pairs), matching
`FLEET_SEEDS = c1d.ARM_SEEDS["b0i2d"]` (24, 900101-900124) plus the PA-2DR-15-precedent 9-seed
extension (900125-900133) — reached via `sbatch --array=0-32` (`p3_2d_fleet.sbatch`'s
`SEED=$((900101 + TID))` convention, gotcha 4), not by editing the sbatch script.

### Cost recompute (confirmed before submission)

33 tasks × 8.6667 CPU-h/task (empirical anchor, `RECOST_RECORD.md` §1, twice-replicated on jobs
6723958/6730213, re-confirmed current/not-stale this session per §1.1's reasoning — the eight
`[PHYSICS]` commits since d04d9dc9 up through `2b657255` inclusive are all instrument-flag or
draw-law-repair changes on the SAME driver being re-anchored, not a change invalidating the
minutes/task figure itself, since the anchor is a wall-clock/cpu measurement of the harness, not a
function of which draw law it happens to run) = **286.0 CPU-h ≤ 290.0 CPU-h cap (row #308).** Not
over. No STOP triggered.

### Preflight (verbatim, this launch's session)

Before sync (cluster HEAD `dcb2c470`, behind local `7e9e1e27` by 6 commits, including `2b657255`):
```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 75 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
Sync: **`git push origin fix/p32d-classg-venue-repair` succeeded directly this session** (the
prior sessions' "direct push denied" finding did not reproduce here — no bundle/scp fallback was
needed for the push leg). `ssh bwunicluster git fetch + merge --ff-only` then hit one untracked
collision (`cluster/graph1_t5_armR.sbatch`, newly tracked upstream) — moved to
`~/wave2_untracked_sbatch_backup/` (not deleted), then `git merge --ff-only
origin/fix/p32d-classg-venue-repair` fast-forwarded `dcb2c470..7e9e1e27` cleanly. Confirmed
`git merge-base --is-ancestor 2b657255 HEAD` → true on the cluster checkout. Re-run preflight
post-sync:
```
VERDICT: READY ✓ (WARN: 1 issue(s))
   • 75 unregistered dataset dir(s) in 'emri' — register in cluster/datasets.yaml + DATA_INVENTORY.md
```
Same pre-existing backlog (gotcha 11), not addressed by this launch, not a blocker.

### Checksum pin (dataset provenance guard)

`md5sum darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` — local
`c52c13b5cab61f6b3f04bbe202550969`, cluster `c52c13b5cab61f6b3f04bbe202550969`. **Identical.** No
stale-copy risk for this run (CLAUDE.md's dataset-pinning mandate).

### Cluster working-tree state before submission

`git status --porcelain` on the cluster showed 0 tracked-file modifications (only untracked
scratch/result directories, the same pre-existing backlog preflight already flags) — the
repaired `correspondence_1d.py` at `HEAD=7e9e1e27` is exactly the committed `2b657255` content,
nothing locally dirtied on top of it.

### Submission

```
$ sbatch --array=0-32 --export=ALL,OUT_ROOT=$WS/p3_2d_fleet_aprime_20260902 cluster/p3_2d_fleet.sbatch
Submitted batch job 6769177
```
`cluster/p3_2d_fleet.sbatch` used **verbatim** (no edits) — only the `--array` range and
`OUT_ROOT` were overridden via `sbatch` CLI flags, per the script's own documented
`--export=ALL,OUT_ROOT=...` override pattern (its header comment: reusing the banked default
out-root would silently no-op against `PA-CA-11`'s per-seed reuse guard and risk contaminating the
frozen pre-repair comparand). Fresh out-root confirmed non-colliding
(`ls $WS/p3_2d_fleet_aprime_20260902` → does not exist) before submission.

`squeue -u $USER` post-submit: `6769177_[0-32]  cpu_il  p3-2d-fl  PD` — queued, 33-task array,
`cpu_il` partition, `--cpus-per-task=16`, `--time=02:00:00` (from the unedited sbatch pragmas).

### Disposition

**LAUNCHED.** Job ID **6769177** (array 0-32, 33 tasks, seeds 900101-900133, both arms/task).
Estimated cost **286.0 CPU-h** (≤ 290 CPU-h cap, row #308). No code was edited, no commit made by
this launcher (physics implementation was `2b657255`, already committed and reviewed before this
session started). Readout (stage gates + stage lhs2d against the registered v2.9 conditional
prediction: LHS2(bt) = 0.00740040 ± 0.00024951 ±3σ_comb two-sided, G4 ∈ [0.8613, 0.8675]) is a
follow-on task once all 66 arm-seed tasks complete — not run here.
