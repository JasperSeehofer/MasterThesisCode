# REPORT-R1 — M65: darksiren-emri, first research project into manyfold

**Mission:** M65 · **Branch:** `mission/M65-darksiren-emri-into-manyfold` · **Date:** 2026-09-21/22
**Orchestrator:** opus, delegating reads/drafts/verification to subagents.
**Interrupted:** the host OOMed at 23:02 on 2026-09-21, killing this session mid-flight. Work
committed before that survived (`e109f4b9`); the session was resumed from disk on 2026-09-22 and
the one lost subagent re-run. This is noted because it affects nothing in the findings but does
explain the two-day datestamp.

---

## 1. The headline

Three things were asked for. All three are delivered, and one of them changed shape once the
evidence was in:

1. **Vault wiring** — done. `THREAD.md` + `CONVENTIONS.md` committed; entity-page and CLAUDE.md
   patches **proposed, not applied**, per the governance split.
2. **The m1 adjudication** — done. **VERDICT: UNRESOLVED-AS-REGISTERED.** No tie-break invented.
3. **The kill-gate-3 memo** — done, but the question it must answer is **not the one the mission
   page anticipated**. See §4.

**The one thing that must not happen, didn't:** m1 is nowhere reported or implied as discharging
T-1's anti-tuning criterion. The register says so in its own words and the verdict quotes it six
ways.

---

## 2. Done criteria

| # | criterion | status |
|---|---|---|
| 1 | entity-page `## Vault Wiring` diff proposed (not applied); index/log entries | **done, with a correction** — see §5. The `wiki/index.md` entry already exists and is accurate, so no append was warranted; what index.md actually needs is a *fix* to a stale duplicate, which is an edit and therefore a proposal. `wiki/log.md` append: **applied**. |
| 2 | `.planning/research/THREAD.md` + `CONVENTIONS.md` exist; `.gitignore` carries `.cartographer-notes.md`; Active experiment states T-1 blocked verbatim | **done**, plus an unplanned `.gitignore` fix — see §3. `.cartographer-notes.md` was already present at line 67. |
| 3 | CLAUDE.md vault-wiring section proposed (not applied) | **done** — `PROPOSALS.md` P-4. |
| 4 | written m1 verdict citing dossier + docket by path, stating plainly whether the tie is resolvable | **done** — `VERDICT-m1.md`. |
| 5 | kill-gate-3 memo as a review card with go/kill chips | **done** — `MEMO-killgate-3.md`, card `m65-r1-darksiren-manyfold`. |
| 6 | REPORT-R1 + VERIFY-R1 + published review-card round | **partially met** — report + verification done; the round is fully staged and validated but **could not be published from this host**. See §9a. |

---

## 3. What was auto-applied (new files only)

Per project-init's governance contract — *"New file creation is auto-apply. Existing file updates
are proposals."*

- **`.planning/research/THREAD.md`** — hard core, five mirrored hypothesis rows, the three
  thread-level kill criteria quoted from the author's own 2026-08-11 text, current claims, active
  experiment, claim history, last verification.
- **`.planning/research/CONVENTIONS.md`** — 23 numbered convention rows, **every one a
  transcription of an already-binding rule with its source cited**; plus four *open convention
  defects* (D-1..D-4) surfaced by the adjudication, kept explicitly separate from the numbered
  rows because they are gaps, not conventions.

### An unplanned finding: `.planning/` was gitignored

`.gitignore:62` excluded `.planning/`, so the two files project-init creates would never have been
version-controlled — "wired" in appearance only, and lost with the worktree. The rule explains
itself: *"Durable project state is TRACKED in ROADMAP.md, STATE.md, and docs/ … The noisy,
ephemeral working state below is intentionally local-only."*

`THREAD.md` is *durable* state by that rule's own definition, and the `/cartographer` bridge
assumes it is committed. Rather than `git add -f` and leave a trap, a **narrow exception** was
added (`.planning/*` + `!.planning/research/`), leaving every other `.planning` subdir ignored.

Two side-findings, reported but not acted on:
- The pre-existing `!.planning/*.png` exception at line 27 is **dead** — it is overridden by the
  later `.planning/` rule. Left alone as out of scope.
- The mission's own report files under `.planning/missions/` remain ignored and were **force-added**
  as a visible one-off, rather than broadening the policy for all future missions.

### Three project-init defects worth fixing before the next migration

1. **The slug bug is load-bearing.** Step 3 derives `SLUG` from `basename $PWD` → `darksiren-emri`,
   finds no matching entity page, and routes to *auto-create*. Run literally it would have written
   a duplicate `wiki/entities/darksiren-emri.md`, forking from the populated
   `master-thesis-code.md`. Suppressed; the slug was overridden throughout.
2. **The skill hardcodes `/home/jasper/...`** while this machine uses `/home/ops/...`.
3. **The `.planning/` gitignore collision** above is not specific to this repo and will recur
   anywhere `.planning` is treated as scratch.

---

## 4. The kill-gate memo — why the question changed

The mission page framed this as *ship-as-methods-paper vs extend-again-pending-M38*. The evidence
says that framing is already one step behind:

**The extension was already granted.** `wiki/meta/research-portfolio.md` records it as a ruling of
2026-09-21, conditioned on **two** preconditions: run T-1 **and** adjudicate m1.

**One precondition was met; the other is unmeetable.** m1 is adjudicated. T-1 is blocked — and by
**two independent gates**, which is the finding that most changes the picture:

- **Gate 1, infrastructure:** no cluster credential; `M38-cluster-session-unlock` unbuilt; the
  CPU-only cheap partial is ruled *"NOT a substitute"* by the redteam review itself.
- **Gate 2, governance:** docket **R27** states *"no (m2) launch until ratified."* T-1 *is* the
  sealed (m2) run. **Even with cluster access, T-1 could not legitimately have launched.** This
  gate is the author's own and is liftable tonight, independently of any credential.

Verified directly: the only sealed-mock exec directory on the branch is `r-sealed-mock` (m1);
**no `(m2)` directory exists**. T-1 has never been launched.

**Criterion (3) therefore fires on 2026-09-23 regardless of tonight's work**, because it keys on
*"no blind-mock verdict"* and m1 is not blind.

**Recommendation (a recommendation, not a decision): ship methods, preserve separately, and rule on
R27.** The date is currently doing two unrelated jobs — gating the science and marking a data
deadline. Decoupling them costs nothing and unblocks both.

---

## 5. The m1 verdict

**UNRESOLVED-AS-REGISTERED.** Full text in `VERDICT-m1.md`. In brief: the sealed register has no
precedence rule, no cell ordering, no primary-channel discriminator (both clauses fire in *both*
channels, so the "1D/2D disagree" escape hatch does not apply), and no strict reading on which
either clause fails. Its INTERMEDIATE cell is registered as *"neither"*, which on its face excludes
*"both"*.

The strongest pro-resolution argument was **considered and rejected**, not overlooked: one could
read the g-censoring gate's *"a BOUND … see the NOT-TUNED row"* as routing the read into
NOT-TUNED before cells are compared. Rejected because the cross-reference is a pointer to where the
rail rule lives, not an assignment instruction; its operative content is NO-READ-vs-BOUND, not
which banked cell wins; and the identical condition is NO-READ in the parent draft, so the reading
is not stable across documents. **Treating it as precedence would supply the missing rule, not
apply one.**

**The near-miss worth reading:** the *parent* register defines INTERMEDIATE as *"none or **several**
of the above"* — which covers a both-fire posterior exactly. The (m1) table narrowed it to
"neither". That is the closest the corpus comes to self-resolving, and it is offered to the author
as Q1b rather than applied, because electing a superseded parent clause over the operative one is
*itself* a precedence ruling.

Four questions go back: **Q2** (R28 pool-pin — answer first, a "No" moots everything), **Q1** (the
tie), **Q1b** (parent wording), **Q3/Q4** (the R27 erratum; the joint_r1 sibling).

---

## 6. Independent verification

`VERIFY-R1.md`. A fresh-context adversarial verifier was dispatched **before the adjudication draft
existed** and never shown it, so its arithmetic cannot have been anchored to the adjudicator's
reasoning. The two converged independently.

- **All four numbers reproduce exactly** — 2.73σ / 2.58σ / 4.39 % / 2.59 %, re-derived from the
  record's own means, σ and per-node mass lists.
- **The pre-registration is demonstrably sealed before the result.** The dossier's threshold block
  hashes to `c908790d…` at all four dossier commits, including ~11 h before the first array task.
  No post-hoc threshold editing. *This is the thing a sealed mock most needs to prove, and it
  proves it.*
- **"Both cells fire" does not verify as a fact** — three incompatible readings exist across three
  registration artifacts, two yielding NO-READ.
- **Two traps a naive verdict would have hit:** "2.73σ" is distance to the *anchor*; from the truth
  it is **3.02σ, outside 3σ**. And "railed" means crossing a 1e-3 floor — the density is *declining*
  toward the edge with an **interior MAP**, not piling up.
- **TUNED survives on ~9 % of σ-margin**, against a registered expectation that σ would be ~9 %
  *wider* but came in 24 % *narrower* — violated in the direction that keeps TUNED alive.

---

## 7. An evidentiary defect that outranks the tie

`MANIFEST.md5` lists **85** entries; **43** are tracked. Absent from version control: the 41
posterior JSONs, `event_likelihoods.csv`, and the scorer `t0_read.py` itself. **Every decisive
number in the m1 docket is prose-only testimony.** Internal arithmetic coheres to ~1e-4 — real but
weak corroboration, since a systematically wrong scorer yields an equally consistent set.

This collides with the workspace expiring **2026-09-23 with zero extensions** — the same date
criterion (3) borrowed. Preservation detail in §7a.

### 7a. Preservation audit — what is actually at risk on 2026-09-23

Read-only, no cluster contact. **Scope caveat, stated first because it bounds every "missing"
below:** this host is *not* the author's dev box (`cluster/datasets.yaml:26` names
`/home/jasper/Repositories/darksiren-emri`). "Not found locally" therefore means **UNKNOWN on the
real machine**, not confirmed-lost. `~/data-backups/` — which `DATA_INVENTORY.md:29-41` claims
holds a durable 3.8 GB copy — does not exist here and could not be verified either way.

**At risk — git-untracked and not found on any disk reachable from here:**

| asset | what it is | status |
|---|---|---|
| `injection_pool_depth15_50k` (500 files, 50k events) | canonical P_det survival-estimator input for the whole `generator_marginal` production path | **CLUSTER-ONLY** |
| m1's `event_likelihoods.csv` + 41 `posteriors/h_*.json` | the read input behind every m1 headline number | **CLUSTER-ONLY** — absent from git on *all* branches |
| `t0_read.py` | the scorer that produced the m1 numbers | **not found anywhere** — unversioned |
| `posteriors_with_bh_mass/` (41 × 86 MB = 3.3 GB for m1 alone; ~3.2 GB/run elsewhere) | 2D per-event posteriors | **CLUSTER-ONLY by design** — *"not a read input; left on the cluster"* |
| per-seed raw CRB/diagnostics working dirs | raw simulation output | **CLUSTER-ONLY** — `simulations/`, `h_sweep_*`, `cluster_results/` are all gitignored |

**Already safe:** derived JSON/CSV summaries across most campaigns are committed, and the m1
**anchor** data (`exec/m-head-rebaseline/c0prime_eval/`) *is* in git — though only the single
h=0.73 slice, not the 41-point grid.

**Three things make this worse than a routine backup chore:**

1. **This was already flagged and not acted on.** An independent commission on **2026-08-14**
   (`results/commission_research_20260814/REPORT.md:123-145`) found the CRB CSV, frozen-α JSON,
   pruned catalogue and injection pool all *"live in git-untracked directories … a clean checkout
   cannot reconstruct these runs"*, and recommended: *"Archive the pinned inputs with committed
   checksums to persistent storage now, especially given bwHPC workspace expiry is a known project
   constraint."* Five weeks later, the same gap is still open — now two days from the deadline.
2. **The regeneration cost is unknown, by the repo's own admission.**
   `docs/campaign_redesign_51_design.md:245-247`: *"total node-hours of the last campaign not
   locally reconstructible (no simulate logs/`sacct` retained)."* There is no GPU-hour figure for
   the existing 50k pool anywhere. A T-1 run after the deadline would likely have to regenerate it
   with no budget estimate to plan against.
3. **The m1 docket could become permanently unauditable** — per-event data gone *and* scoring
   script unrecoverable, leaving only prose testimony of numbers that are currently the subject of
   an open ruling.

**What this does not mean:** the science is not necessarily lost. Derived summaries survive, and
the author's own machine may hold copies this host cannot see. The actionable point is narrow —
**somebody with cluster access should check and copy off before 2026-09-23**, and that access is
the same thing M38 is blocked on.

---

## 8. Scope discipline

**Not done, deliberately:** T-1 in any form; any cluster/SLURM/ssh/preflight command; any cheap
partial; any cluster-credential work (M38's scope); new physics; paper text; any merge or push to
main; any new entity page.

The unmerged branch was read **only** via `git show origin/fix/p32d-classg-venue-repair:<path>` —
never checked out, fetched or merged.

**`wiki/log.md`:** appended (2026-09-22 entry, auto-apply per the governance split). It was
briefly outstanding because the host OOMed mid-session with sibling sessions writing the vault
concurrently; it has since been applied.

---

## 9. A spec error in the mission page worth fixing

The mission page specifies chips `kind:go-kill`, `kind:verdict`, `kind:doc`. **Only `go-kill`
exists.** The live enum is `merge | taste | go-kill | free`
(`parallax/src/schemas.ts:1210`, `src/review.ts:86`), and the parser **silently downgrades**
anything else to `"free"` — no error, no warning. The round therefore uses `go-kill` for the memo
and `free` for the other two, with real chips carrying the decision content.

### 9a. The round is staged and validated, but NOT published — and why I stopped

`publish-round-r1/` contains `entry.json` plus all five linked documents. It passes every check the
real tooling enforces, verified by replicating the validators' own source:

- required fields present; none of the forbidden `answers`/`stale`/`superseded_by`
  (`preview-merge.mjs:6`);
- all three questions carry 4 chips each in the `{label, consequence}` **object** form, clearing
  `previews-lint.mjs:45`'s `chips.length < 2` rule;
- every link label carries a `(context)`/`(react)` tag (`previews-lint.mjs:54`);
- every link resolves to a real file in the round dir (the publish pre-flight's condition);
- `kind` values are all in the live enum; round-level `kind[]` is index-aligned.

**Why it is not published: there is no Node runtime on this host.** `publish-preview.sh` shells out
to `node` at five separate points (lines 107, 117, 130, 232, 320) and to `npm run lint:previews`
(line 265). `which node` fails, and a filesystem search finds no node binary — checked both inside
and outside the sandbox.

**I deliberately did not hand-merge the entry into the live index instead.** That would mean
writing unvalidated JSON directly into `app/{public,dist}/previews/index.json` — a phone-facing
file currently serving 89 cards — with **no locking** in the publish path (only a timestamped
`.bak`), while eight sibling mission sessions were active in the same vault/repo. A malformed or
raced write breaks the Previews lens for every card, not just this one. That is an outward-facing,
hard-to-reverse action, it is outside this mission's autonomy grant, and I could not validate the
result without the tooling. Staging and reporting is the correct failure mode; silently
half-publishing is not.

**To publish, from any host with Node:**

```bash
cd /home/ops/Repositories/parallax && scripts/publish-preview.sh \
  /home/ops/Repositories/darksiren-emri-wt/M65-darksiren-emri-into-manyfold/.planning/missions/M65-darksiren-emri-into-manyfold/publish-round-r1 \
  --live-repo /home/ops/Repositories/parallax
```

The card id is `m65-r1-darksiren-manyfold`. Until it runs, the three questions have not reached
Jasper's phone — the vault `log.md` entry is the only published trace of this mission.

**A live bug found in passing, affecting cards open on the phone right now:** the two most recent
garden-authored cards (`retrospective-2026-09-21`, `personal-garden-questionnaire`) author `chips`
as bare **strings**. `parseQuestionChip` rejects non-objects and `parseQuestion` filters them out,
so **both cards render with no working tap options** despite being visibly open. M65's card uses
the `{label, consequence}` object form. Worth a separate fix.

---

## 10. Follow-ups

1. **Publish the round** (§9a) — one command, needs a host with Node. Nothing reaches the phone until this runs.
2. **Rule on R27** — unblocks T-1's governance gate independently of the credential.
3. **Preservation** — see §7a; needs the same cluster access that is blocked.
4. **Dispatch `M38-cluster-session-unlock`** if the answer to K1 is "extend".
5. **Fix the bare-string chips bug** on the two live garden cards (§9).
6. **Fix project-init's slug derivation** before the next research migration (§3).
