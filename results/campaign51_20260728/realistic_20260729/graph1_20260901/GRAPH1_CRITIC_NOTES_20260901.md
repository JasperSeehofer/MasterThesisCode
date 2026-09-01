# Adversarial Critic Notes — Research Graph 1 Proposal Pair (2026-09-01)

Reviewer: adversarial critic subagent. Scope: INFRA_AUTONOMOUS_RESEARCH_PROPOSAL_20260901.md
("INFRA"), RESEARCH_GRAPH_1_PROPOSAL_20260901.md ("GRAPH1"), spot-checked against
gate_b_20260730/BIAS_HISTORY_LEDGER.md ("LEDGER") and STATE_AND_CANDIDATES_20260901.md
("STATE"). Neither proposal was edited. No source files were touched.

Counts: **10 findings total — 4 MUST-FIX, 4 SHOULD, 2 NOTE.**

---

## MUST-FIX

### 1. GRAPH1 silently upgrades a STANDING-class item to a RULE-tagged charter ratification
**Location:** GRAPH1 §0 "Charter frame", line 10 ("...and the approval-scope semantics of infra
3.4 as the meaning of whole-graph ratification") and §3 row 0 (line 244, tag `RULE`, ask
"Ratified"), versus INFRA's own Ratification ask (line 303): "**[STANDING — explicit grant
required, lapses at campaign end]** the section 3.4 approval-scope semantics as the meaning of
charter ratification for graph batches."

INFRA itself correctly tags infra-3.4's approval-scope semantics as a **STANDING** item — "pre-
authorize a class of future decisions," per CLAUDE.md, grantable "only when the author says so
explicitly." GRAPH1 then folds that exact same provision into the definition of what ratifying
row 0 *means*, before row 0 is even reached, and asks for it under a single-word **RULE** reply
("Ratified") with no separate STANDING line anywhere in GRAPH1 §3. A one-word "Ratified" on row 0
would, on GRAPH1's own wording, silently also grant the STANDING-class provision that INFRA itself
says needs its own explicit grant. This is precisely the failure mode the approval-scope rule
exists to prevent (an approval propagating beyond what the author consciously tagged), reproduced
inside the very proposal that is arguing the rule should be made mechanical.

**Fix:** GRAPH1 §3 must carry infra-3.4's approval-scope semantics as its own separately-tagged
`[STANDING]` row (with the scope/lapse condition stated, as GRAPH1 already correctly does for the
panel-always-on grant in row 11), not as a clause folded into row 0's charter-ratification prose.

### 2. Objective function pays "gave up cheaply" the same as "genuinely irreducible," and the
knob that decides which happened is undisclosed as orchestrator-derived
**Location:** INFRA §3.2 (line 159): "the system's score for a batch is the number of registered
questions moved to a SETTLED state — verified, refuted, **or bounded-undetermined** — with all
consumed panels green or waived. Refuted pays the same as verified." Restated verbatim in GRAPH1
§0 (lines 20-23). The `bounded-undetermined` disposition is reached by exhausting
`max_revisions` (INFRA §2.4 item 2, line ~106) — and all four register nodes in GRAPH1 freeze
`max_revisions 2` (r-b82-s4 line 69, r-jr1-massaware line 96, r-completion-residual line 142,
r-cone-loss line 153) with **no derivation and no `ORCHESTRATOR-DERIVED` tag**, even though the
*cost caps* sitting in the same table cells one column over (line 142: "arm cap <= 80 CPU-h
ORCHESTRATOR-DERIVED, unscoped in sources"; line 153 likewise) are explicitly flagged per the
row #268 lesson that STATE and both proposals otherwise apply rigorously.

As written, the objective function does not distinguish a question that was actually driven to
irreducibility from one that was parked after two cheap, shallow revision attempts — both register
as a full-credit "SETTLED" question. Because the acquisition layer (INFRA §2.6) is explicitly
proposed to score and select future candidates against this same objective, and Horizon 3's
self-improvement channel (b) tunes acquisition weights "against retrospective regret," an
optimizing process is structurally rewarded for setting shallow, unexamined revision budgets
rather than investing in genuine discrimination — the exact "reward hacking against the
evaluation harness" pattern INFRA §3.6 itself is written to guard against, reappearing at the
objective-function layer that section doesn't cover.

**Fix:** either (a) source/derive `max_revisions=2` per register node (or explicitly tag it
`ORCHESTRATOR-DERIVED` like the cost caps beside it) so the author sees it is being asked to
ratify a parking threshold along with the topology, or (b) split the objective's credit for
`bounded-undetermined` from `verified`/`refuted` so premature parking cannot buy the same score as
resolution.

### 3. `d-s3-rerun` is a load-bearing decision with no corresponding graph node
**Location:** GRAPH1 §3 row 1 (line 245): `d-s3-rerun (row #288 item (d)'s fresh RULE...)`, tag
`RULE`, ask "Ratified," gating whether "S3 re-runs post-flip at all" — a decision the table's own
row numbering places *before* row 2's `b-s4-harness-repair` (Branch A) approval.

`d-s3-rerun` does not appear anywhere in §1 (no q-/r-/d- table entry), is absent from §1.11's
enumerated convergence-decide-node list, is absent from §4's mermaid diagram, and has no
requires-manifest. It exists solely as a table row in §3's ratification list — exactly the
"graph state... simulated in narrative" failure mode INFRA §1.3 (line 48) names as the reason the
tree model chafed. If `b-s4-harness-repair`'s `authorized-by` edge (branch-A table, its "inputs"
column) lists only `d-batch1-charter`, nothing in the machine-checkable graph itself enforces that
it also waits on `d-s3-rerun`'s ruling — the sequencing exists only in the author's reading order
of a markdown table, which is the very thing this infrastructure is supposed to replace.

**Fix:** add `d-s3-rerun` to §1 as a properly typed `d-` node with a requires-manifest (its inputs,
per the row's own text, are rows #286/#288 — already-existing evidence, so this is a
same-batch RULE that can be resolved before or alongside row 0), an `authorized-by`/gating edge
into `b-s4-harness-repair`, and a slot in the §4 diagram.

### 4. GRAPH1 uses zero checkpoint (`k-`) nodes and reverts fan-out caps to launch-summary prose,
contradicting INFRA's own stated replacement of that practice
**Location:** INFRA §2.1 (line 74): `k- | checkpoint | dynamic expansion point | fanout_cap,
tier, per-child cost — all declared BEFORE expansion`; INFRA §3.3 (line 162): "fan-outs computed
before launch — **now enforced by k- node caps rather than launch-summary prose**." GRAPH1 uses
10 of the 12 node types (INFRA §2.1) and **never once instantiates a `k-` node**. The one obvious
dynamic-fan-out case in the batch — `v-falsifier-ii-classG`'s "class-G fleet" (branch-E table,
line 116: "fleet: sonnet / medium (fan-out capped)... fleet cost cap 60 CPU-h hard") — declares
only a CPU-h ceiling, not a fan-out count, tier, or per-child cost as a `k-` node's schema
requires, and carries no `k-` node of its own. GRAPH1 §5.2 (lines 395-421) then states every
wave's fan-out and tiering ("top-tier = 2," "top-tier = 3 (at the cap)," "Fan-out computed before
launch: ... the falsifier fleet (hard cap 60 CPU-h)...") as narrative prose in the roadmap
section, which is verbatim the artifact INFRA §3.3 says the schema was built to retire.

This is a direct, checkable contradiction between the two documents: INFRA's pitch for why the
graph model is safer than the tree model rests partly on caps being schema-enforced rather than
prose-stated, and its own first instantiation does not use the node type that enforces them.

**Fix:** add a `k-falsifier-ii-fleet` checkpoint node (or equivalent) with a declared
`fanout_cap` (the actual number of class-G configurations, not just a CPU-h ceiling) ahead of
`v-falsifier-ii-classG`, and route the wave-by-wave tiering counts in §5.2 through declared `k-`
nodes rather than prose, or explicitly note in §5 that this pilot batch has no genuine dynamic
fan-out and defers `k-` usage to a future batch (a defensible position, but not the one currently
stated).

---

## SHOULD

### 5. Cumulative top-tier headcount for "graph1" as a whole is never stated against the ~3 cap
**Location:** GRAPH1 §5.2 (lines 401, 409, 417): wave 1 "top-tier = 2," wave 2 "top-tier = 3 (at
the cap)," wave 3 "top-tier = 2." CLAUDE.md's hard cap (quoted in the repo's own orchestration
mandate) is "at most ~3 top-tier (inherit) agents **per workflow**." Graph1 is ratified as a
single charter/workflow (§0, one scope hash over the whole graph), yet the wave-1
derivation/prereg author, wave-2 prereg author, wave-2 decisive verifier, and wave-3 end-verifier
are never stated to be the same identities reused across waves — read most literally, the batch's
distinct top-tier headcount could run to 5-6 (chair + up to 4 distinct specialist roles), well
over the cap, and the document never computes or discloses the cumulative number the way CLAUDE.md
asks ("state the chosen tiering... so the author can veto overkill").
**Fix:** state explicitly whether the same agent persists as prereg-author/verifier across waves,
and give the batch-total top-tier count, not just three separate per-wave counts.

### 6. Three terminal paper decide-nodes cite raw artifact numbers outside any feeds edge
**Location:** GRAPH1 §1.12, lines 193-195: `d-paper-coverage` cites `b8_information_floor.json`
floor numbers directly; `d-paper-1d2d-verdict` cites "the owned 1D rail -0.063 (row #286), the 2D
offset -0.0667 / sigma_h 0.0184 (artifact section 00), depth-skew... (artifact section 05)";
`d-paper-massinfo` cites "existing sigma_M sweep (artifact section 08)" and "the information-floor
pair above" — all as bare prose citations inside a decide node's requires-manifest, not as
`feeds` edges from a typed `rd-` node carrying artifact path + sha256. This is exactly what INFRA
§2.2's `feeds` edge law and linter rule L4 (§6.4: "every feeds edge carries artifact + sha256;
stale checksum invalidates downstream") are written to prevent — GRAPH1's own terminal, highest-
stakes nodes (the paper claims) are the ones that route around the mechanism.
**Fix:** add `rd-` nodes for `b8_information_floor.json`, artifact `a8824799` §00/§05/§08, with
proper `feeds` edges into the three paper decide-nodes.

### 7. Ledger citation imprecision on commit `5e7fda16` — spot-check result
**Location:** INFRA §2.5 line 120 ("The flip (commit `5e7fda16`, row #286)"); GRAPH1 line 41
("commit 5e7fda16, row #286"), line 81 ("feeds from commit 5e7fda16 (row #286)"), line 173
("feeds from commit 5e7fda16 (row #286) + row #282"). **Checked against LEDGER and `git log`:**
the commit is real (`5e7fda1638c52fcf3ad7a2c5fc81a0cf108df434`, 2026-08-31 13:46:27, message
matches the flip exactly) and row #286 does correctly narrate that flip event — but the literal
string "5e7fda16" **does not appear anywhere in ledger row #286's own text**; it appears only in
row #288 (`...the run's long-lived processes predate the 5e7fda16 flip...`), referring back to it.
Downgraded from a fabrication concern to a citation-precision note because the number itself is
git-verifiably correct — but a graph whose central selling point (INFRA §1.2 item 1) is that
"every quoted number carries its source artifact and checksum" should cite the row where the
identifier actually appears (row #288), or both rows, not only the row that lacks it.
**Fix:** cite row #288 alongside/instead of row #286 wherever the literal hash is quoted.

### 8. Acquisition scoring (the one genuinely novel mechanism) is asserted, never demonstrated
**Location:** INFRA §2.6 (lines 145-147): "an acquisition record scores candidates by
discriminating power (count and weight of discriminates edges...) against declared cost
(probe_cost...)." No worked score is computed for any of the 11 real candidates in STATE §2, even
though STATE already supplies exactly the cost/input/shared-decision data the formula needs. Every
other mechanism in these two documents (bands, gates, byte-identity, closure) is grounded in a
specific ledger row where it actually fired; the acquisition-scoring claim is the one piece of the
"atomic idea" assessment that is pure assertion with zero applied example — a hand-waved item in
an otherwise evidence-heavy pair of documents.
**Fix:** compute one worked acquisition score against the current 11-candidate list before or
alongside ratification, so "discriminating power against declared cost" is shown to produce a
sane ranking rather than merely asserted to.

---

## NOTE

### 9. Rounding drift on the pilot F number (no defect, just worth a footnote)
STATE (line 18) and GRAPH1 (line 195) both carry `F = 7.426` for the no-BH cell-S pilot value;
LEDGER row #288 states `F = 7.43` at its own precision. Same underlying measurement, more digits
carried forward in STATE (presumably from the pilot record's own JSON, one level more precise than
the ledger's prose summary) — not a discrepancy, but the extra precision's origin (pilot record
§3.1) is never named at the point of use in GRAPH1 itself, only in STATE.

### 10. Verify (`v-`) nodes have no analogue of `max_revisions`
INFRA's bounded-re-entry law (§2.4) caps repeat attempts only at `register` nodes. A `v-` node
whose survival record is itself disputed (the practice-mining record explicitly cites this
happening — "T5.1 window proposal REFUTED at round 2... re-checked by a second refuter," INFRA
§1.1 item 3) has no stated cap on repeat verification rounds. In the current design this fails
safe (the downstream `decide` node simply stays permanently blocked rather than causing runaway
compute), so this is not runaway-risk-bearing, but it is an asymmetry in the "every re-entry is
bounded" claim (INFRA §2.4's heading) that is worth naming before the claim is made absolute.

---

## Verbatim MUST-FIX list (for convenience)

1. GRAPH1 §0/§3 row 0 bakes INFRA §3.4's STANDING-tagged approval-scope semantics into a RULE-tagged charter ratification instead of asking for it as its own explicit STANDING item.
2. The objective function (INFRA §3.2 / GRAPH1 §0) credits "bounded-undetermined" equally with "verified"/"refuted," and the `max_revisions=2` threshold that produces "bounded-undetermined" is asserted at four register nodes with no derivation and no `ORCHESTRATOR-DERIVED` tag, unlike the cost caps beside it.
3. `d-s3-rerun` (GRAPH1 §3 row 1) is a load-bearing, sequenced decision with no corresponding node, requires-manifest, or diagram placement anywhere in GRAPH1 §1/§1.11/§4 — it exists only as a table row.
4. GRAPH1 instantiates zero `k-` checkpoint nodes and states all fan-out/tiering caps as roadmap prose (§5.2), contradicting INFRA §3.3's explicit claim that fan-out caps are "now enforced by k- node caps rather than launch-summary prose."
