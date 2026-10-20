# T-1 governance package: what you are asked to ratify

Mission M162, 2026-10-19. Branch `mission/M162-t1-governance-package`, darksiren-emri, commit
`e1ae6b20`. **Nothing has been drawn, simulated or run, and no cluster command was issued.** Every
item below is a [RULE] on text that is already written and hashed. The [DO] item is a cluster read.
Approving this card does not launch T-1. Launch still needs M38's window, the M65 landing, and the
pin re-reads.

## What the package is

| file | role |
|---|---|
| `ERRATUM_m2_R27_R28.md` | the binding text. Rule blocks E1–E8 plus the parameters the draw script reads. sha256 per block in `.planning/research/THREAD.md` |
| `POOL_OPTION_MEMO_m2.md` | why P2 first, how the invariance gate is made meaningful, what P1 costs, and the cluster reads needed |
| `UNSEAL_CEREMONY_m2.md` | who does what between the code freeze and the cell: draw, blind phase, unseal |
| `PREREG_EXP-MTC-T1.md` | the pre-registration (also mirrored as a vault row) |
| `scripts/sealed_mock/draw_sealed_h.py` | the draw. It refuses inside an agent session, refuses a second draw, refuses a params block whose hash differs from the ratified one, and refuses unless HEAD == H_code with tree `71acb98b…` and dirty count 0. 8 tests pass |

## Q1 [RULE]: the tie rule and the σ treatment (R27)

**Tie rule** (E-2). The cell is decided in a fixed order, and the first step that fires decides:
1. A red gate ⇒ NO-READ.
2. **The rail clause wins**: an edge-node mass above 1e-3, or a MAP on an edge node, ⇒
   AT-RAIL-LOW or AT-RAIL-HIGH, banked as a one-sided bound.
3. Exactly one band cell (TUNED, TRANSFER or UNBIASED) holds, and the 1D replicate agrees ⇒ that
   cell.
4. Two or more band cells hold, or none, or 1D disagrees ⇒ INTERMEDIATE. This is the same
   "none or several" wording you applied to m1 (R25).

**What "rail wins" costs, stated plainly.** A posterior that sits at the TUNED value 0.666 but
has a fat lower tail (m1 had 4.4 % of its mass on the 0.600 node) reads AT-RAIL-LOW, not TUNED.
The card's consequence row still reports its distance to TUNED, and the consequence comes back
to you as a fresh ruling.

**σ treatment** (E-3). The bands use the measured σ_h, as registered. The check:
- Compute ρ = σ_h / σ_exp, where σ_exp is the anchor σ scaled by sqrt(1588/N).
- If ρ is outside [0.8, 1.25] in either channel, the cell is assigned again at σ_exp.
- Same cell both times ⇒ it banks.
- A different cell ⇒ INTERMEDIATE (σ-sensitive), and both results are reported.
- On m1, ρ = 0.70, so the check would have fired. The band cell would not have changed.

**Overlap disclosure.** At σ ≳ 0.010–0.015 the TUNED and TRANSFER windows overlap for the lower
part of the prior, so INTERMEDIATE (several) is a live outcome.

Recommended: **Ratify E-2 and E-3 as written.**

## Q1b [RULE]: is the rail a banked bound or NO-READ? (CONVENTIONS D-2)

The 09-03 draft made the rail NO-READ. The m1 dossier made it a banked bound. "Rail clause wins"
only makes sense if the rail books a cell, so E-2 adopts the bound. NO-READ is kept for gate
failures.

Recommended: **Bound.**

## Q2 [RULE]: the pool memo (P2 first, as you ruled, made workable)

**What the draft got wrong.** The draft's gate compared a 20-task pool at h = 0.67 with the 0.73
pool and required a difference of at most 1e-3. With about 5,700 independent rows spread over
2,400 grid cells, Monte Carlo noise alone is ≈ 0.1. That gate would always fail and always force
P1.

**The fix** (E-5):
- The small pool reuses the seeds of 20 tasks of the pool of record, so each row is the same
  draw at a different h. The injection loop draws z first and computes d_L afterwards.
- Pairing check P2-0: at least 95 % of rows must match. If they do, the 1e-3 gate is computed
  on the matched rows. If they don't, P2 fails and P1 follows.

**P1** stays capped at 250 GPU-h. Within that cap the pool has at most 500 tasks, about 71 % of
the 707-task pool. P1 needs no naming shim and no code change if its pool lives inside the
sealed run directory.

Recommended: **Ratify.**

## Q3 [RULE]: the pre-registration, the ceremony and the dated default

The pre-registration is EXP-MTC-T1. In the ceremony:
- the author draws and unseals on the cluster login node;
- agents monitor by `squeue`/`sacct` only;
- the reader sees only the posterior JSONs and four columns of `event_likelihoods.csv`;
- the moments are committed before the unseal.

The dated default (E-7):
- A banked cell by **2026-12-15**, Europe/Berlin, counts as a verdict. That includes
  INTERMEDIATE and AT-RAIL.
- NO-READ, or no unseal by that date, means kill criterion 3 fires: a methods paper, with the
  closure reported unblinded.
- No third extension without a new ruling.

Recommended: **Ratify.**

## Q4 [RULE]: does TRANSFER fire kill criterion 1?

**The conflict.** Kill criterion 1 says T-1 closing "anywhere other than the sealed h_inj" reverts
the adoption. TRANSFER means the posterior closes at h_inj − 0.064. That is literally elsewhere,
yet the 09-03 draft books TRANSFER as "T-1 PASSED". The two texts disagree on the most likely
non-failing outcome, so the mapping has to be fixed **before** the draw. M162 changed neither
text.

**Options.**
- **Does not fire (recommended).** Criterion 1 targets tuning to 0.73. The h-independent
  offset is a separately tracked bias (R24) that already bars a closure-at-truth claim.
- **Fires.** Read literally, only UNBIASED passes.

## Q5 [DO]: read-only cluster reads before T-1

The reads are R-1 to R-3 in the pool memo §5:
- R-1: N_tasks and the cost of the seed61000 simulate;
- R-2: re-read the pool, CRB and catalogue pins on emri2. None has been re-read since the move;
- R-3: the pool-of-record seeds, which P2 pairing needs.

They run in M38's window, by one ops agent using `cluster/agent_ssh.sh`, or by you. A mismatch is
a STOP, not a re-pin.

Recommended: **Approve.**

---

## Rulings, card `m162-r1-t1-governance` (relayed to the executor 2026-10-20)

**Status: RATIFIED, all six questions.** This section records the rulings as relayed. It is
append-only.

**Source text, verbatim from the relay message:**

> Jasper ratified all six questions of your card m162-r1-t1-governance (E-2/E-3 ratified; rail
> read = banked bound; paired P2/P1 design ratified; prereg + unseal ceremony + dated default
> ratified; Q4: TRANSFER passes anti-tuning, kill criterion 1 is not triggered by h_inj - 0.064;
> Q5: read-only cluster reads R-1/R-2 approved inside an M38 window)

**Attribution note** (per the `BIAS_HISTORY_LEDGER.md` convention):
- The quote above is the orchestrator's relay. Jasper's own words are not in front of this
  executor, so the itemisation is relay-derived.
- The relay gives 2026-10-20 as the date it reached the executor. The exact time of the tap is
  not recorded here.

| Q | tag | ruling as relayed | what it binds |
|---|---|---|---|
| Q1 | [RULE] | "E-2/E-3 ratified" | erratum E-2 (precedence: gates ⇒ rail wins ⇒ single band cell ⇒ else INTERMEDIATE) and E-3 (ρ band [0.8, 1.25]), hashes in THREAD.md |
| Q1b | [RULE] | "rail read = banked bound" | a firing rail clause books AT-RAIL-LOW/HIGH as a banked one-sided bound, not NO-READ (closes D-2) |
| Q2 | [RULE] | "paired P2/P1 design ratified" | E-5: P2 paired with P2-0 ≥ 95 %, the 1e-3 gate on matched rows, before the draw; P1 ≤ 250 GPU-h inside the sealed RUN_DIR as the fallback |
| Q3 | [RULE] | "prereg + unseal ceremony + dated default ratified" | `PREREG_EXP-MTC-T1.md`, `UNSEAL_CEREMONY_m2.md`, E-6, E-7 (2026-12-15, Europe/Berlin; a banked cell including INTERMEDIATE/AT-RAIL is a verdict; no third extension without a new ruling) |
| Q4 | [RULE] | "TRANSFER passes anti-tuning, kill criterion 1 is not triggered by h_inj - 0.064" | E-8 TRANSFER row ⇒ anti-tuning PASS. Kill criterion 1's own text is unchanged |
| Q5 | [DO] | "read-only cluster reads R-1/R-2 approved inside an M38 window" | R-1 (N_tasks/cost of the seed61000 simulate) and R-2 (pin re-reads on emri2), read-only, `cluster/agent_ssh.sh`, one ops agent, in M38's window. **R-3 (the pool-of-record seeds, needed for the P2 pairing) is not named in the relay and is therefore NOT approved**; it returns to the author before P2 can run |

**Approval scope (CLAUDE.md):**
- None of these rulings covers a decision whose inputs do not yet exist: the P2 gate value, the
  cell of the unsealed result, and its E-8 consequence row each return as a fresh [RULE].
- Ratification does not launch T-1. The launch still needs M38's window, R-1/R-2 green, R-3
  approved and run, P2 gated, and the L-0 preconditions of `UNSEAL_CEREMONY_m2.md`.
