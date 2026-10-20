# VERIFY-R1 — independent re-check of the m1 sealed-mock numbers

**Mission:** M65 — darksiren-emri: first research project into manyfold
**Date:** 2026-09-21
**Verifier:** fresh-context adversarial subagent (opus), dispatched *before* any adjudication draft
existed, and never shown one. Its brief was to assume the claimed numbers were wrong and report
mismatches, not to confirm them.
**Evidence access:** read-only, `git show origin/fix/p32d-classg-venue-repair:<path>`. No checkout,
no merge, no fetch of that branch. No cluster, SLURM, ssh, or simulation command was run.

## Why this pass exists

The mission's reporting contract asks for a fresh-context verifier that re-derives the m1 numbers
from the raw record independently of the adjudication draft — the same discipline the mission seed
had asked for T-1's numbers, applied to what actually ran. Dispatch order was deliberate: the
verifier was launched before the adjudicator returned, so its arithmetic cannot have been anchored
to the adjudicator's reasoning.

## 1. The four claimed numbers — all MATCH

| # | claimed | found in record | located at | verdict |
|---|---|---|---|---|
| 1 | 2.73σ off-anchor, 2D | `2.73`; recomputed **2.7293** | `READ_RECORD_m1.md` §3, 2D row | **MATCH** |
| 2 | 2.58σ off-anchor, 1D | `2.58`; recomputed **2.5767** | `READ_RECORD_m1.md` §3, 1D row | **MATCH** |
| 3 | 4.39 % mass at h=0.600, 2D | `4.39e-2` / `0.0439` / `0.044` | `READ_RECORD_m1.md` §2 g-censoring row; §3 table; §3 per-node list | **MATCH** |
| 4 | 2.59 % mass at h=0.600, 1D | `2.59e-2` / `0.0259` / `0.026` | same three places | **MATCH** |

Recomputation actually performed by the verifier (a consistency check of the record against
itself, not an independent measurement — see §3 T1):

- 2D: |0.627674 − 0.6659| / 0.014006 = 0.038226 / 0.014006 = **2.7293** → 2.73 ✓
- 1D: |0.631084 − 0.6670| / 0.013939 = 0.035916 / 0.013939 = **2.5767** → 2.58 ✓
- per-node masses sum to 0.990 (2D) / 0.987 (1D), with the record's "<1 % above 0.66" closing the
  remainder to 1.000 ✓; first four nodes give 0.712 / 0.619 against the stated
  mass(h≤0.63) = 0.713 / 0.619 ✓; mass-weighted mean over the listed nodes reproduces the stated
  means to ~1e-4 ✓.

Numbers cross-checked and also matching: MAP 0.630 both channels; 2D mean 0.627674 / σ 0.014006;
1D mean 0.631084 / σ 0.013939; |MAP−0.665|/σ = 2.50 / 2.51; σ-ratios 0.758 (−24 %) / 0.795 (−20 %).
The figures appear in three places — `READ_RECORD_m1.md`, `MORNING_DOCKET_20260904.md` R25, and
`BIAS_HISTORY_LEDGER.md` Row #387 — and **agree exactly**, with no drifting variants.

## 2. The fifth claim — "both cells fire" — does NOT verify as a fact

This is the finding that matters most, and it is the reason this mission does not ship a
tie-break. Verdict: **CONTESTED — not a verifiable fact.**

Applying the sealed register's thresholds to the verified values:

| clause | registered test | observed | fires? |
|---|---|---|---|
| TUNED | `\|mean−0.6659\| ≤ 3σ` AND `\|MAP−0.665\| ≤ 3σ` | 2.73 ≤ 3; 2.50 ≤ 3 (2D) | **yes** |
| NOT-TUNED, first clause | `\|mean−0.6659\| > 3σ` | 2.73 is not > 3 | **no** |
| NOT-TUNED, rail sub-clause | edge-node mass `> 1e-3` | 0.0439 (2D), 0.0259 (1D) | **yes, iff read as a standalone trigger** |

So "both fire" is reproducible **only** under the reading that the sealed dossier's
"INCLUDING a railed posterior …" is an independent disjunct. Read grammatically as written,
"INCLUDING" makes the rail a *subset* of a `> 3σ` condition that fails — under which only TUNED
fires.

**Three incompatible readings exist across three registration artifacts:**

1. `REGISTRATION_DRAFT.md` §9 — rail as an independent NOT-TUNED trigger ("mass concentrated
   below 0.63 **or at a rail**").
2. `REGISTRATION_DRAFT.md` §6/§8 — the same rail (`edge-node mass > 1e-3`) maps to **NO-READ**:
   *"red ⇒ NO-READ"*, "nothing banked".
3. `D_SEALED_REGISTER_DOSSIER.md` §2 — reverses that consequence (*"a red here is a BOUND, not
   NO-READ"*) and re-homes the rail inside NOT-TUNED via "INCLUDING".

The record concedes the tension in its own words (`READ_RECORD_m1.md` §4): *"the table is
inconsistent with itself on this posterior in two places (verifier finding)"*, which is what
drives docket item **R27** ("the cell table needs a precedence rule").

## 3. Traps a naive verdict would have fallen into

**T1 — [CRITICAL] The raw posterior data is not in version control.** `MANIFEST.md5` lists **85**
entries; only **43** files are tracked (41 `run_metadata_*.json`, `GIT_COMMIT_AT_RUN.txt`,
`MANIFEST.md5`). The ~42 untracked ones include `simulations/diagnostics/event_likelihoods.csv`
(md5 `ed3d5479c1dad0a27b8060106836206f`, which *does* match the record's quoted md5) and 41
`simulations/posteriors/h_*.json`. The scorer `t0_read.py` is likewise not in the repo. The 2D
primary channel's own posteriors (`posteriors_with_bh_mass/`, ~3.3 GB) were deliberately left on
the cluster. **Consequence: every decisive number is prose-only testimony.** The internal
arithmetic coheres to ~1e-4, which is real but weak corroboration — a systematically wrong scorer
would produce an equally self-consistent set. See REPORT-R1 §preservation for why this is urgent.

**T2 — anchor ≠ truth.** 2.73σ / 2.58σ are distances to the *0.73-run re-baseline anchors*
(2D 0.6659, 1D 0.6670 — different reference points per channel). Distance to the public truth
0.67 is **3.02σ (2D) / 2.79σ (1D)**. The 2D posterior is therefore **>3σ from truth while ≤3σ from
anchor**. `REGISTRATION_DRAFT.md` §6 registers the pull statistic as
`pull_c = (mean_h,c − h_inj)/σ_h,c` = **−3.02** for 2D. Writing "2.73σ off" without naming the
anchor is a live misreading risk.

**T3 — "railed" overstates the picture.** Converting the per-node masses back to densities via the
gradient weights: 2D density 4.4 at h=0.600 vs 13.8 at h=0.610 (1D: 2.6 vs 9.5). The posterior is
**falling steeply toward the lower edge, not accumulating against it**, and the MAP (0.630) is
interior. The 1e-3 floor is very tight on a 41-node grid (uniform ≈0.024/node), so it fires on any
posterior with a modest lower tail. The number is right; the word invites a wrong mental image.

**T4 — TUNED survives on ~9 % of margin, against a registered expectation that was violated in the
helpful direction.** The 2D verdict flips to >3σ if σ_h were 0.012742 rather than 0.014006 — a
**9.0 %** change (1D margin 14.1 %). The dossier had registered an expectation that σ_h would be
**~9 % wider** at N=1343 ("σ_h × ≈1.09 vs 1588"); it came in **24 % / 20 % narrower** instead. The
single most fragile input to the "both fire" claim.

**T5 — a second, independent route to NO-READ (R28).** The registered pool
(`injection_pool_depth15_50k`, 500 files) is **not** the pool the 0.67 run used
(`injection_pool_mix200k_20260728`, 707 files), confirmed in `PIN_RECORD.md`. The chair overrode
the registered text pre-submission. Docket R28 explicitly offers the author the alternative:
*"or rule the (m1) read **NO-READ** under the registered text."*

**T6 — boundary condition exactly on the line.** `REGISTRATION_DRAFT.md` §6's TUNED clause carries
a qualifier the dossier dropped: *"for h_inj with |h_inj − 0.73| ≥ 0.06"*. For m1,
|0.67 − 0.73| = 0.06 **exactly** — satisfied under `≥`, but the narrowest possible pass.

**T7 — timestamps disagree across records.** Record header says "~12:30 CEST", the creating commit
is 12:00:34 +0200, ledger Row #387 says "~13:20 CEST"; the record pre-discloses a clock problem.
Cosmetic for the numbers, but **no record timestamp in this cluster can be used as evidence of
ordering — use commit times.**

**Cleared non-issues:** rounding (every figure rounds correctly), duplicate appearances (all three
locations agree exactly), units (all dimensionless), and no superseding record exists.

## 4. Was the pre-registration sealed before the result existed? **YES — evidenced.**

| artifact | commit | time (+0200) |
|---|---|---|
| `REGISTRATION_DRAFT.md` created | `25be7a66` | 2026-09-03 21:02:29 |
| `D_SEALED_REGISTER_DOSSIER.md` created (cell table + gates) | `06a12422` | 2026-09-03 22:46:34 |
| `DESIGN_GATE_computability_m1.md` (GREEN) | `17843cc4` | 2026-09-03 23:01:24 |
| dossier + `PIN_RECORD` (PIN CORRECTION) | `d9ea6037` | 2026-09-03 23:07:41 |
| dossier (GUARD CORRECTION; resubmit 6794421) | `4cc1758d` | 2026-09-04 09:44:29 |
| **first array task starts** | `run_metadata_0` | **2026-09-04 09:40:05** |
| `READ_RECORD_m1.md` + docket R25–R28 + ledger #387 | `d201d9e4` | 2026-09-04 12:00:34 |

**Decisive check:** the verifier hashed the dossier's §2 statistic/cell-table/gates block at all
four dossier commits and obtained `c908790d6daeed2ab2fcda7d0c6980bd` at **every** one, including
the pre-run `06a12422` (~11 h before task 0 started). **The thresholds are byte-identical before
and after the run; no post-hoc threshold editing occurred.** The one dossier edit postdating job
start (`4cc1758d`, 09:44:29 vs task-0 start 09:40:05) is a pure append documenting the pool guard
used at submission and touches no threshold — the *documentation* of the launch landed ~4 minutes
after the first task began, but the *criteria* were frozen the previous night.

This is a genuine strength of the record and should be stated as such: the one thing a sealed-mock
most needs to prove, it proves.

## 5. Raw-metadata corroboration (41 × `run_metadata_*.json`)

| record claim | metadata says | verdict |
|---|---|---|
| array 0–40, one h per task, H_GRID_41 | h per TID = 0.600…0.860, exactly H_GRID_41; TID 21 = 0.73 | ✓ |
| seeds 777000+TID | 777000…777040, monotone | ✓ |
| tasks 0–4 at `8f933e7b`, 5–40 at `d9e50179` | exactly that; `GIT_COMMIT_AT_RUN.txt` = `d9e50179…` | ✓ |
| 16 cpus/task, `--evaluate`, all 27 flags | matches dossier §2 verbatim on all 41 files | ✓ |
| out-root `graph1_sealed_m1_iiib_20260904` | identical on all 41 | ✓ |
| md5 of the read input | `MANIFEST.md5` entry = record's §1 md5 | ✓ |

Three caveats on this evidence: (i) `run_metadata_*.json` contains **no injected-truth field** —
`h_value` is the grid node being evaluated, not h_inj, so "truth = 0.67" rests on the source run
directory's name (`run_20260729_seed64000_h0p67`), not on retrieved metadata; (ii) neither the 1343
event count nor any posterior number appears in any committed file, so metadata cannot corroborate
them; (iii) `SLURM_ARRAY_JOB_ID` is not recorded, so array identity cannot be confirmed from
metadata alone.

**Was the record revised after the docket? No.** `READ_RECORD_m1.md` has exactly one commit
(`d201d9e4`) — the same commit that writes docket R25–R28 and ledger Row #387. No commit after
2026-09-04 touches `exec/r-sealed-mock/`. Note that the §6 verifier appendix is part of that same
single commit, so the "an independent verifier returned and the chair then appended" sequence is
asserted in prose but has **no separate commit evidencing it**.

## 6. Reliance verdict

| claim | rely on it in a written verdict? |
|---|---|
| 2.73σ off-anchor, 2D | **YES, with attribution** — reproduces exactly; must be written as "2.73σ from the re-baseline anchor 0.6659", never "from truth" (that is 3.02σ) |
| 2.58σ off-anchor, 1D | **YES, same conditions** — anchor 0.6670, a different reference point |
| 4.39 % at h=0.600, 2D | **YES as a number, NO as the word "railed"** — density is declining, MAP interior |
| 2.59 % at h=0.600, 1D | **YES, same conditions** |
| "both clauses fire" | **NO** — contested reading; chair deliberately made no call (R25/R27); two independent NO-READ paths exist (draft §6, and R28); TUNED survives on ~9 % of σ-margin |

**Overall:** the arithmetic is clean and the seal-before-result ordering is genuinely evidenced by
git. Two things should give any verdict-writer pause: the complete absence of the underlying
posterior data from version control (every decisive number is testimony), and the fact that the
fifth claim is a contested reading the chair explicitly declined to make.

## 7. Effect on this mission's verdict

This pass is why REPORT-R1's adjudication returns **UNRESOLVED-AS-REGISTERED** rather than a
tie-break, and why the verdict text names the anchor explicitly, qualifies "railed", and carries
the R28 NO-READ path as a live alternative rather than a footnote. No tie-break rule was invented.
