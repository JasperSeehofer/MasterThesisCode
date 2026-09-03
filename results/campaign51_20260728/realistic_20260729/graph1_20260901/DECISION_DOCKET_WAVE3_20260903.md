# Decision Docket — Research Graph 1, Wave 3 (convergence) + overnight autonomy scope

Date: 2026-09-03 (evening). Chair: Fable 5.1 orchestration session. Author of record: Jasper Seehofer.
Status: PROPOSAL — reviewable decision artifact per CLAUDE.md "Proposing decisions". Approval-scope
tags per CLAUDE.md: [DO] "Approved" · [RULE] "Ratified" · [STANDING] "Granted".
Device note: the author continues on `thinkpad`; runbook 42 §7 (device transfer) is MOOT — nothing
in §7 is executed, and every path below is the thinkpad path of record.

## 0. Where the graph stands (rows #291–#333, digest verified against runbook 42 — no contradictions)

Graph 1 was ratified 2026-09-01 (charter row #290) with 9 branches (A–I) + a closure chain, 3 waves.
Waves 1 and 2 are EXECUTED. Score so far: 2 questions SETTLED (F-ii window: not adopted, row #314;
q-a4-provisional: A4 VERIFIED, row #325), 2 [PHYSICS] commits landed (a26959b4 h-grid decoupling;
2b657255 Option A′ class-G de-double-weight), 1 claim retired (c-rphi-mismatch, row #301), 1 claim
Z-CONFIRMED at zero compute (m-jr1, row #305).

| branch | node chain | state |
|---|---|---|
| A calibration | b-s4-harness-repair → r-b82-s4 → m-s3 (S,T) → rd-s3-readout | S CLOSED n_U=67 (author, row #333); T 25/25; **aggregation running now**; readout NEXT |
| B re-baseline | m-head-rebaseline | DONE, GREEN-AS-CORRECTED (rows #298/#299/#302): iiib map 0.665 / joint_r1 mean 0.6670 |
| C venue generality | dv-jr1-transform → r-jr1-massaware → m-jr1 | DONE; Z-CONFIRMED in band [0.64,0.70] (row #305) |
| D photo-z leverage | rd-runner11 → b-pahier33 → m-s0b-production | job 6779532 COMPLETE 5/5 (row #332); **retrieval running now**; READ NEXT |
| E falsifier | v-falsifier-ii-classG | DONE (2 stops: cost 60→290 CPU-h, Option A′); LHS2/G4 both INSIDE (row #322); A4 ratified (row #325) |
| F T5 window | m-t5-armS / armR → d-t5-window | SETTLED: not adopted (row #314) |
| G completion residual | r-completion-residual → m-completion-residual | NOT STARTED (needs F from rd-s3-readout) |
| H cone loss | r-cone-loss → m-cone-loss | NOT STARTED |
| I h-prior fix | b-hprior-fix | DONE ([PHYSICS] a26959b4; wing mass 2.4e-15, row #313) |
| closure | rd-rphi-note → d-rphi-retire | DONE/RATIFIED (Z_on = 1.0 exact) |

Decide nodes still to return: d-photoz-leverage · d-completion-register · d-cone-register ·
d-calibration · d-residual-attribution · d-paper-coverage · d-paper-1d2d-verdict · d-paper-massinfo.

Open author words (non-graph, docket item 12): 12a backup of the 159 GB sole-copy `~/emri-archive`
(TOP PRIORITY, needs a destination only the author can name) · 12b cluster evacuation before
2026-09-23 (0 extensions) · 12c disk culls · 12e merge → main (112 ahead) · 12f safe builds · 12g docs sync.

## 1. Wave 3 execution plan (what runs tonight if approved)

Order is the graph order; every science read waits for its gate stamps (panel law, rows #290/#325).

1. **rd-s3-readout** — n_U = 67 (S) / 25 (T), `stopped_reason: wall_limited` disclosed; verdict-free
   coverage/F table; g-population + g-precision stamps; the two design-gate caveats (external exact-KS;
   binom_bands label, row #303) carried. Reader sonnet/high; chair re-derives F and the HPD/PIT numbers.
2. **S0-B reads** (registered, mechanical dispositions only): g-score-null |Z|≤3 gate; score_b_re
   secant per PA-HIER-31(d) (denominator 0.066); score_s; B0-B disposition per §2.1(e)
   (LEVER-DEAD-AT-N iff |Z_b|≤3 ∧ |Z_lns|≤3; materiality |b̂|<0.0165; power σ_b<0.0661).
   Reader = fresh sonnet; chair re-derives the decisive Z's. → **d-photoz-leverage dossier**.
3. **r-completion-residual + r-cone-loss authoring** — ONE top-tier prereg author (xhigh), inputs:
   F from item 1, the re-baseline, artifact §09 (−0.14/event; 17 % cones). Design-validity gate
   (sonnet panel) on each. → **d-completion-register / d-cone-register**.
4. **m-completion-residual / m-cone-loss** — launch ONLY under item 2.2 below (else parked for the
   author). Caps ≤80 / ≤20 CPU-h (ORCHESTRATOR-DERIVED, charter-ratified as asks).
5. **d-calibration dossier** — needs rd-s3-readout green, the re-baseline banked-as-comparand ask,
   m-jr1 disposition (Z-CONFIRMED). One top-tier end-verifier re-derives every decisive number
   feeding a d- node. → d-calibration, then d-residual-attribution, then the three paper rulings.

Tiering (k-wave3-fanout, cap ~3 top-tier): chair + 1 prereg author (item 3) + 1 end-verifier
(item 5) = 3. Every reader, panel, clerk, retrieval and aggregation agent is sonnet (low/medium).
Fan-out: ≤6 sonnet identities per batch; no dynamic expansion. Ledger clerk: one long-lived sonnet
agent, quote-verbatim convention (runbook 42 §5).

Cluster: SSH ControlMaster is pinned by a local keepalive loop (240 s touch); a cluster
launch still needs `/cluster` preflight VERDICT: READY.

## 2. Decisions asked of the author

| # | item | tag | reply | grants | explicitly NOT covered |
|---|---|---|---|---|---|
| 2.1 | Execute wave 3 items 1–3 and 5 tonight under the standing row #325 grant ("continue autonomous, decide but flag") | DO | Approved | the reads, the dossiers, the two registration drafts, the end-verification; each chair decision flagged in its ledger row, veto reverts | any ruling with inputs not yet computed (all d- nodes return as [RULE] dossiers) |
| 2.2 | Chair may ratify d-completion-register and d-cone-register itself (decide-and-flag) and LAUNCH m-completion-residual (≤80 CPU-h) / m-cone-loss (≤20 CPU-h) overnight, provided each design gate is GREEN and preflight READY | STANDING (this night only; lapses at the author's next message) | Granted | launch inside the caps; a red gate parks the arm | any band edit beyond the caps; any revision-2 spawn (returns to the author); any cluster job > the cap |
| 2.3 | d-photoz-leverage, d-calibration, d-residual-attribution and the paper rulings return as dossiers for the MORNING, not as chair rulings | RULE (procedural) | Ratified | the chair drafts, computes eligibility from manifests, does not rule | — |
| 2.4 | Docket 12a: name the backup destination for `~/emri-archive` (159 GB, sole copy) | author word | (text) | the copy is launched as a background job the same night | — |
| 2.5 | Docket 12b: authorize the cluster-evacuation triage list to be drafted tonight (what to archive from the ~250 GB before Sep 23) — drafting only, no deletion | DO | Approved | a triage table for morning review | any deletion or move |

Binding default reminder: an approval never propagates to a decision whose inputs did not exist
when it was given (CLAUDE.md). Rows 2.1 + 2.2 are the whole overnight envelope.

## 3. What happens to Research Graph 2

Graph 1 closes at the three paper rulings. A Graph 2 charter (successor batch) is authored ONLY
after d-residual-attribution, because its question layer depends on the three-way split. It will
be proposed as its own docket, not tonight.
