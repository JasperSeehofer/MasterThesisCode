# r-completion-residual -- DESIGN-VALIDITY GATE RECORD

Node: `r-completion-residual` design gate. Research Graph 1, Branch G, wave 3.
Author of record for all scientific decisions: Jasper Seehofer.
Lens: **DESIGN VALIDITY, blind to results** -- no `completion_residual_result.json` exists yet
and none was consulted; only the frozen-in-appearance draft text, the source it cites, and the
charter/docket were read to produce this record. Applies the six-check structure of
`r-b82-s4/DESIGN_GATE_RECORD.md`, adapted per the launch instruction to this arm's own six
clauses (object/population pins, statistic specification, disposition table, gates+bands+STOP,
cost-vs-cap, revisions+kill-criterion).

## Authorization and scope

Ledger row #333-adjacent authoring grant: docket `DECISION_DOCKET_WAVE3_20260903.md` item 2.1
("registration AUTHORING only") + charter row 9 (`RESEARCH_GRAPH_1_PROPOSAL_20260901.md:276`).
Per the charter's own row: "the arm's band + cap ratification (d-completion-register); the
launch" are separately gated, and docket item 2.2 requires "each design gate is GREEN and
preflight READY" before the chair may launch `m-completion-residual`. This record is that gate.
No cluster job, no pipeline run, and no edit to `darksiren_emri/` was made to produce it.

Inputs read:
- `graph1_20260901/exec/r-completion-residual/REGISTRATION_DRAFT.md` (full text)
- `graph1_20260901/exec/r-completion-residual/INFORMATION_FORECAST.md` (companion forecast)
- `graph1_20260901/RESEARCH_GRAPH_1_PROPOSAL_20260901.md` (charter: q-completion-residual row,
  m-completion-residual row, docket-cap row, wave-3 fan-out row)
- `graph1_20260901/DECISION_DOCKET_WAVE3_20260903.md` (items 2.1-2.5)
- `graph1_20260901/exec/r-cone-loss/REGISTRATION_DRAFT.md` (seed-collision cross-check only)
- `darksiren_emri/bayesian_inference/bayesian_statistics.py` (column-identity source, ~line 6790-6812)
- `tree2_20260830/b8_cal_harness.py` (`_score_at_truth_by_class`, h_bounds construction)
- on-disk artifacts: `seed61000/prepared_cramer_rao_bounds.csv` (md5), the iiib
  `event_likelihoods.csv` (row count), git ancestry of commits `1ec9514d`/`5e7fda16`

## Check 1 -- Object + population pins unambiguous, disjoint from claimed seed blocks: **GREEN**

The registered object (S1.1) is unambiguous: a per-event **matched-channel** score
`s_M,e = Δ ln B_num(e)/Δh - Δ ln β̄_Ḡ^φ(h)/Δh` on dark events only, on the stencil
(0.725, 0.735), Δh=0.01 -- distinct by construction from the full score, the catalogue-leg
score, and the composition tilt (S2.1's identity table). It is explicitly *not* the same
quantity as B4_3's "-0.14/event" (S1.2's provenance audit correctly downgrades that number to
a stale [INFER] figure and re-anchors the registered object as "the first per-event measured
value with its own SE").

Verified independently (not merely re-quoted from the draft):
- CRB md5 pin `9a1f2a14384a9281c97ca3be312ddaab` reproduces exactly against
  `seed61000/prepared_cramer_rao_bounds.csv` on disk.
- The iiib `event_likelihoods.csv` is 65,109 lines (65,108 data rows) = exactly 41 x 1,588,
  matching S2.2's "41 x 1588 rows" claim precisely.
- Commit `1ec9514d`'s ancestry includes `5e7fda16` (`git merge-base --is-ancestor` confirms),
  matching the invariants line's claim.
- `num_log_term_no_bh`, `den_log_term`, `B_num`, `D_tilde_phi` all exist as named CSV columns in
  `bayesian_statistics.py` at the cited location, and the algebra
  `num_log_term_no_bh - den_log_term == log(combined_without_bh_mass)` holds by direct
  inspection of the assignment -- the S2.1 identity is not an assertion, it is definitionally
  true of the columns as written.
- `_score_at_truth_by_class` (harness line 1183) implements the identical secant formula
  `(log(hi)-log(lo))/(hi_h-lo_h)` at the same default stencil, confirming Read B's "banked,
  informational; full score" power inputs are sourced from a real, matching function.

Seed-block disjointness (the falsifier clause): the only *new* seed claim in the launch is the
optional replication cell R, `--seed-block 903000 --n-universes 30` (903000-903029). Grepped
`graph1_20260901/**/*.md` and `tree2_20260830/**/*.py` for `903000`: the **only** hit is this
draft itself -- no other node (including the companion `r-cone-loss/REGISTRATION_DRAFT.md`,
checked directly) claims that block. It is disjoint from 901000-901099 (r-b82-s4's registered
S-cell block, itself fully consumed by 901000-901066), from 902000-902024 (T-cell), and from
the 901100+ falsifier reservation by inspection (903000 > 901100+99). Read B's *reuse* of
901000-901066 is correctly zero-compute reuse of already-generated checkpoints, not a fresh
claim, so it does not need to be disjoint from itself.

No gap found on this check.

## Check 2 -- Statistic fully specified, zero fresh choices at launch: **GREEN**

Every quantity in S2.4's table (T_prod, Z_prod, T_harn, Z_harn, rho, delta h_M) traces to a
named column, a named file, and an explicit formula: SE_prod = SD_e(s_M,e)/sqrt(N_Ḡ) with
N_Ḡ=1512 stated; SE_harn = SD/sqrt(67) with the sampling unit named as the universe
(PA-HIER-5, i.e. not pooling all ~11,525 dark events as if independent -- the correct
non-fresh choice given between-universe correlation). The S7 launch command is fully
parameterized: every flag (`--h-lo/--h-hi/--h-true`, `--population`, both md5 pins, all four
input paths) is sourced to an earlier section of the same document; none is invented at the
launch line. The one soft value not re-derived in-line, `--h-true 0.73`, is the project's
standing fiducial `H` (CLAUDE.md / `constants.py`), not a fresh choice.

Minor, non-blocking note: `beta_Gbar_phi`/`sigma_phi` cross-check in S2.1 ("must agree to 1e-3
relative, else disclose and use the column definition") is itself a well-specified fallback,
not a gap -- flagged here only because it is the one place a data-dependent branch exists
inside the "zero fresh choices" claim; it is a disclosed, pre-committed branch, not a fresh
choice, so it does not downgrade the verdict.

## Check 3 -- Disposition table three-valued, every outcome returns as a fresh RULE: **GREEN, one wording note**

The table (S4) is three-valued at the level the parent claims are framed at --
ILLEGITIMATE / FLOOR-CONSISTENT / INTERMEDIATE (INTERMEDIATE itself carrying three named
sub-branches a/b/c) -- with NO-READ carved out separately as an instrument-defect terminal
state that is not a science verdict (mirrors r-b82-s4's INCOMPLETE-RUN/INSTRUMENT-DEFECT
precedent exactly). The "stage-5 action (returns to author)" column header asserts every row
returns to the author; the ILLEGITIMATE row explicitly says "(fresh RULE)".

Wording note, non-blocking: the FLOOR-CONSISTENT row's text is "report the bound;
q-completion-residual settles (charter kill criterion NOT the path -- this is a clean result)"
-- it does not repeat the words "fresh RULE" the way the ILLEGITIMATE row does. Read in context
this is the intended, designed terminus of a clean discrimination (the parent binary question
is answered, not skipped past); it is not a self-ratification in the sense CLAUDE.md's binding
default forbids, because the column header still routes it "to author." But the asymmetric
phrasing between rows is a real inconsistency worth tightening before the FLOOR-CONSISTENT
branch is actually reached: add the words "returns as fresh RULE" to that cell so the header's
blanket claim is not contradicted by one row's more casual phrasing. Does not block launch --
Read A/B are zero-compute and this affects only how the eventual clean-result report is worded.

## Check 4 -- Gates consumed, named with bands and a STOP consequence: **AMBER**

Five of the six named gates (S5) are fully specified with a band/tolerance and a traceable
consequence:
- **g-closure**: band 1e-9, explicit "Red STOPs the read," and listed as a NO-READ trigger in S4.
- **g-population**: band stated per venue (0 mixed rows harness; 1588 rows/node + gap set
  {1203,1356} + in-cat=76 production); listed as a NO-READ trigger in S4.
- **g-precision**: band 1e-3 relative cross-check where available, explicit fallback ("disclose
  and use the column definition") -- a defined, non-STOP consequence, correctly not listed
  under NO-READ since it degrades gracefully rather than voiding the read.
- **g-censoring**: explicit consequence for its governed claim ("a quote without the disclosure
  is void") -- scoped correctly to the one REPORTED-ONLY line it governs (S3.4), not the
  verdict-bearing statistics.
- **g-byte-id (instrument)**: band = bit-for-bit reproduction of 67/67 checkpoints + the T0
  helper; gates the launch itself (S7: "Launch waits on the byte-id gate GREEN").

**g-znorm is the gap.** Its S5 text reads in full: "standing on the flipped leg (row #292); one
spot check that `den_log_term` is identical across all 1588 rows per h in both venues." This
names no tolerance ("identical" -- bit-exact float equality, or some epsilon?) and no
consequence on failure, and -- unlike the other five -- it is **not** listed among the S4
NO-READ triggers ("g-closure red, JOIN gate red, byte-id red, g-population red"). This matters
because g-znorm is not redundant with g-closure: the S2.1 identity
`s_M,e + s_T + s_C,e = s_e` is an algebraic tautology by construction (a telescoping sum of
differences of the same columns) and will hold to floating precision *regardless* of whether
`den_log_term` is actually event-independent per h -- so a g-znorm failure would not be caught
by g-closure, silently invalidating the physics reading of `s_T` as a "global composition tilt"
(and, by extension, the interpretation of `s_M` as the *matched-channel-alone* score) without
tripping any other named gate.

**Must-fix before this specific clause is load-bearing** (does not block the zero-compute Read
A/B launch itself, since a failure here degrades interpretation rather than crashing the
script): add a tolerance to g-znorm (natural choice: exact equality, since `den_log_term` is a
single `math.log(_den_used)` per h-node in the source, so any per-row variation is either a
genuine bug or floating noise at the 1e-9 g-closure scale) and add "g-znorm red" to the S4
NO-READ trigger list, or state explicitly why it does not need one (e.g., because a real
violation would also produce a large, visible g-closure-adjacent anomaly the builder is
instructed to eyeball). As written, a silent g-znorm failure has no defined stop.

## Check 5 -- Cost derived, under the docket cap (<=80 CPU-h completion / <=20 cone): **GREEN**

Docket item 4 (`DECISION_DOCKET_WAVE3_20260903.md:51-52`) states the applicable caps verbatim:
"Caps <=80 / <=20 CPU-h (ORCHESTRATOR-DERIVED, charter-ratified as asks)" -- matching the
launch-instruction's stated cap exactly, and matching the registration's own S0 line ("cap <= 80
CPU-h"). Re-derived the S6 arithmetic independently: 30 universes x 1299 s/universe wall-time
anchor (rd-s3-readout's measured 87,016 s / 67 universes = 1299.0 s/universe); at `--workers 2`,
CPU-h = 30 x 1299 x 2 / 3600 = 21.65 CPU-h (doc: "22 CPU-h", rounds correctly); at
`--workers 4`, CPU-h = 30 x 1299 x 4 / 3600 = 43.3 CPU-h (doc: "43 CPU-h", correct). Both are
inside the 80 CPU-h cap with margin (46-58 CPU-h headroom even at W=4), and Reads A+B add only
~0.2 CPU-h local, non-cluster. The decisive path (Reads A+B) is genuinely zero-compute; cell R
is optional and gated to only two of the six disposition branches (S6: "only on INTERMEDIATE
(b) / ILLEGITIMATE"). No gap found.

## Check 6 -- max_revisions=2 present; kill criterion of the parent question quoted: **AMBER**

`max_revisions 2` is present and correctly sourced (S0: "max_revisions 2 (ORCHESTRATOR-DERIVED,
charter S1.7/S1.13)"), and matches the charter's own accounting
(`RESEARCH_GRAPH_1_PROPOSAL_20260901.md:208`: "Three register nodes carry max_revisions 2
(r-b82-s4, r-jr1-massaware, r-completion-residual...)"). This half of the check is satisfied.

**The kill criterion itself is not quoted anywhere in the draft.** The parent question's row in
the charter (`RESEARCH_GRAPH_1_PROPOSAL_20260901.md:45`) reads, verbatim:

> q-completion-residual | how much of the approx -0.14/event dark-class completion-leg residual
> (artifact section 09) is illegitimate estimator inconsistency vs floor-consistent noise, given
> F? | **registered arm fails to discriminate at its registered band after revision 2 -> park
> bounded-undetermined with the measured bound**

Grepped the full `REGISTRATION_DRAFT.md` for "kill criterion", "park bounded", "revision 2": the
only hits are S4's FLOOR-CONSISTENT row ("charter kill criterion NOT the path") and S0's
`max_revisions 2` line -- both *reference* the kill criterion by name but neither quotes its
content. A reader of the registration alone (without opening the charter) cannot tell what
happens if both revisions of this arm fail to discriminate; they would have to independently
locate and open `RESEARCH_GRAPH_1_PROPOSAL_20260901.md` line 45 to find "park
bounded-undetermined with the measured bound." Given the disposition table (S4) otherwise
names an explicit consequence for every other terminal state, this is the one terminal state
(revision-2 exhaustion) left unquoted in the document meant to be launched from.

**Must-fix**: add one line to S4 or S9 (open questions) quoting the kill criterion verbatim, so
the arm's own document states what happens if it is revised twice without discriminating,
rather than requiring a second document to be opened at that point. Non-blocking for launching
the zero-compute Reads A/B today (the revision-2 branch is not reachable before at least one
disposition is read), but should be fixed before this arm could plausibly reach a second
revision.

## Overall verdict: **GREEN -- Reads A + B (the decisive, zero-compute path) may launch; two AMBER items routed as pre-revision-2 fixes**

No RED check. Two AMBER items (g-znorm's missing band/consequence; the unquoted parent
kill-criterion) are both one-line, non-blocking documentation fixes that do not touch any
computed quantity, launch parameter, or cost figure -- neither is reachable before at least one
disposition read exists (g-znorm matters only once the scorer runs; the kill criterion matters
only if the arm reaches its second revision). Consistent with r-b82-s4's own precedent of
routing non-blocking caveats as action items rather than failing the gate. Recommend both fixes
land in the registration text (not a fresh RULE -- they are drafting completeness, not a
scientific decision) before docket item 2.2's chair-decide-and-flag launch is exercised on the
optional cell R or before a revision-2 event, whichever comes first.

## Launch parameter block (transcribed verbatim from the draft; zero fresh choices)

Reads A+B (decisive, zero cluster, ~0.2 CPU-h local):
```
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_residual_reads.py \
  --production-csv results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/simulations/diagnostics/event_likelihoods.csv \
  --production-crb results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
  --replicate-csv results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/simulations/diagnostics/event_likelihoods.csv \
  --harness-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip \
  --population 200 --h-lo 0.725 --h-hi 0.735 --h-true 0.73 \
  --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
  --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-completion-residual/completion_residual_result.json
```
Builder: `b-completion-scorer` (sonnet/medium), dry-run only, byte-id-gated; a different agent
executes the real run (standing rule 2). CRB md5 independently reproduced against the file on
disk during this gate (see Check 1).

Optional cell R (chair-decide-and-flag only, under docket 2.2, only on ILLEGITIMATE or
INTERMEDIATE (b), seed block 903000-903029, confirmed disjoint from all prior reservations):
```
uv run python results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py \
  --work-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_r_completion \
  --N 200 --cell S --seed-block 903000 --n-universes 30 --max-wall-s 43200 --workers 2
```
CPU-h at `--workers 2` = 22 (re-derived above); cap headroom 58 CPU-h.

**Not this gate's decision**: the band/cap ratification itself (d-completion-register, still a
fresh RULE per the draft's own S0), and any actual disposition -- this record is blind by
construction and reads no `completion_residual_result.json`.
