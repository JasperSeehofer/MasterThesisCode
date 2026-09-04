# DESIGN_GATE_Q1_computability_rev2 — r-timeout-selection, `q-timeout-selection-pdet` ONLY

Reviewer: FRESH computability-only pass on the **CONSOLIDATED Q1 TEXT OF RECORD** (`REGISTRATION_DRAFT.md`
§4 header paragraph + REVISION 1 (Q1) + REVISION 2 (Q1) + the CHAIR ERRATUM, as they supersede the original
§4/§6 Q1 sentences). No prior read of this node's registered statistics. `INFORMATION_FORECAST.md` was **not
opened** (FORBIDDEN, honored). `MECHANISM_NOTE.md` read for cross-reference only. The rev-0
`DESIGN_GATE_Q1_computability.md` was read **only** to confirm each of its six findings (F1–F6) is closed by
the revisions — every claim below was independently re-derived from primary sources in this pass (code at
HEAD `79c44608`, the pinned CSVs/JSONs/logs, and — new in this pass — the freshly fetched pool-build log tree
and the frozen-T0 `run_metadata_*.json` files), not inherited from rev-0's text.

**No registered Q1 statistic was computed.** What was run: `md5sum`/`wc -l`/`md5sum -c` on every §1 pin
including the new log-fetch manifest; `grep -c`/`grep -n` counts of named log strings; direct diffs of
task-id sets between the pool-of-record and specific fetched seed directories; direct reads of library
function signatures/call sites/callers (never executed, never imported) and of `run_metadata_*.json` fields
for the pinned T0/joint_r1 runs. No `--dry-run`, no synthetic-table scorer (Q1 has no scorer yet; Phase A/B
have not launched).

**Verdict: RED.** Two of rev-0's structural findings (F1, F5, F6) are genuinely closed. F4 (the 25-site
table) is closed and, on direct byte-for-byte re-verification, is **more accurate than rev-0 itself gave it
credit for** — every one of the 25 line numbers is an exact match to the code. But F2 (the 707/363 population
split) is **not closed by REVISION 1 — its replacement rule is falsified by direct on-disk evidence in this
pass**, and a new, independent, decisive defect is found on top of F3's nominal fix: **the registered PRIMARY
denominator-leg insertion point is never dispatched by the frozen T0 anchor the whole arm is built to
reproduce.** Both are code+data-verified, both would make S1.3 (the PRIMARY statistic) silently wrong rather
than merely imprecise.

## 1. Enumeration and per-item computability

| # | Q1 item | Checked how | Verdict |
|---|---|---|---|
| S1.1 source (rev-0 F1) | per-draw `TimeoutError` line named as the source, `main.py:1293-1302`/`:1143` | Read live `.err` sample (`inject_h_0_73_6073215_56.err`): timeout line carries `params={'M': ...}`; aggregate line (`main.py:1349-1352`) carries only a count, no `M` — confirmed on disk, ≤5 lines | **GREEN — F1 closed** |
| S1.1/S1.2 population rule (rev-0 F2) | REVISION 1 item 2: 707 survivors, matched by (seed dir, SLURM task id), "current-format" test; 363 excluded (326 old-format + 37 crashed) | Directly falsified — see Finding A | **RED — F2 NOT closed; replacement rule is wrong** |
| S1.3 numerator leg | `completion_mass_factor_g_sel`'s `s_query` Callable, `:2276` | `grep -n "^def completion_mass_factor_g_sel"` → line 2276 exact; signature has `s_query: Callable[[d_L, M_z, z], survival]` exactly as registered | **GREEN** |
| S1.3 denominator leg (rev-0 F3) | `_mass_trunc_denominator_inner_m_integral` `:869`, batch `:944`, called from `:8053` inside `denominator_integrant_with_bh_mass` (`:8048-8058`) | Line numbers exact (`def` at 869, `detection_probability...` call at 901/944, call site at 8053 inside the named function). Signature confirmed duck-typed (`detection_probability: Any`). **But** the PRIMARY/SECONDARY labeling is inverted for the frozen T0 anchor — see Finding B | **AMBER on mechanics / RED on the PRIMARY designation — new defect, not in rev-0** |
| 25-site call-site table (rev-0 F4) | `grep -n` of both interpolator method names against the actual file | All 25 line numbers (901,944,1741,2058,3029,5567,6450,6901,6979,7150,7246,7288,9021,9037,9111,9123,9181 with-BH; 1284,1440,1770,3066,7697,8533,8954,8973 without-BH) match the registered table **exactly** — see Finding C | **GREEN — F4 closed, byte-exact** |
| g-closure(i) (rev-0 F5) | `89,456 − 3,449 − 85,584 = 423` | Independently re-grepped on the 100/100 `simulate_6088772_*.err` files: SNR-stage ZeroDiv = 3,449, CRB-stage = 39, timeouts = 822 — all exact matches to the revision's corrected figures; `423` arithmetic reproduced | **GREEN — F5 closed, independently confirmed** |
| build-log manifest pin (rev-0 F6) | new §1 pin, md5 `6ae9c1098c1c3325504e4904b2fc4d50`, 3,510 lines, 1 self-referential failing row | `md5sum MANIFEST.md5` → `6ae9c1098c1c3325504e4904b2fc4d50` exact; `wc -l` → 3,510 exact | **GREEN — F6 closed** |
| [A13] engagement (S1.4) | ≥10% of events `\|δ_e(0.73)\|≥1e-6`, contingent on the numerator leg only | Numerator leg (S1.3) is GREEN; g-closure(iii) text ("δ^den is one scalar per h") confirms S1.4 does not touch the denominator leg | **GREEN, contingent — unaffected by Finding B** |
| Bands / three-valued dispositions | §5 table, `T_mat`/`T_null`, caps, "Fresh RULE on each of the four rows" | Present, well-formed, unchanged by REVISION 1/2 (both state "no threshold or band touched") | **GREEN on form** |
| max_revisions / counter statement | header "max_revisions 2"; CHAIR ERRATUM: pre-launch design-gate revisions don't consume it | 5 revision blocks (REV1/REV2 Q2, CHAIR ERRATUM, REV1/REV2 Q1) are all pre-launch, none is a post-disposition re-registration; the stated exemption is internally consistent and textually scoped | **GREEN on form** |
| Blindness (§10) | Q1 carries no disclosed pre-read | Re-read against the Q1 statistic list: (i)/(ii)/(iii) are all S2.3/`g-population` inputs, none is `ρ(b)`, `Δmean_h^{Q1}`, `δ_e`, `δ^den`, or any p_det value | **GREEN** |

## 2. Findings

### Finding A (RED, decisive) — REVISION 1's 707/363 population rule is falsified by direct on-disk evidence; the actual pool provenance is not recoverable from the fetch by the registered test.

REVISION 1 item 2 registers: *"the other 363 attempts (326 old-format completion lines + 37 crashed...) are
EXCLUDED... consistent with the pool's own disclosed provenance split — code_rev `f6449051`/`a9f29e82`."* This
pass tested that correspondence directly, on the actual pool-of-record files and the fetched logs — not by
re-running the same `grep`-count rev-0 used, but by tracing specific task IDs end-to-end.

**A1 — the entire `a9f29e82` in-pool population comes from an "old-format" (excluded-by-the-rule) seed dir.**
Every one of the 707 pool CSVs' `code_rev` column was read (row 2 of each file): 647 carry `f6449051`
(the "current" code), 60 carry `a9f29e82` (the disclosed 6,000-row, p0/t_plunge-NaN population). The task-id
set of those 60 `a9f29e82` files is **identical**, element-for-element, to the full 60-task set of
`injection_20260728-080420_seed51000`'s `run_metadata_*.json` files. But `seed51000`'s `.err` logs are **0/60
"new-format"** (no `separatrix-sign` field in any of its 60 "Injection campaign complete" lines) — REVISION
1's own discriminator for "excluded, old-format." Content spot-check on `task_16.csv`: pool row count 100,
`code_rev a9f29e82`; `seed51000`'s task-16 log line reads `"100 events stored... 0 timeouts @ 90s... a=58,
b=27, c=15"` — an exact match on every field. This is not an edge case: it is the *entire* disclosed
`a9f29e82` sub-population, verified by task-id-set identity, not sampling.

**A2 — a disclosed "crashed" (excluded) attempt's CSV is nonetheless in `POOL_MANIFEST.md5`.** Task 234 of
`injection_20260728-082426_seed51100` (`run_metadata_234.json` exists; SLURM job `6071877`) has **no**
"Injection campaign complete" line of either format — its `.err` ends `"Injection campaign: 100 / 103
successful SNR computations..."` then `"JOB 6071877 ON uc2n571 CANCELLED... DUE to SIGNAL Terminated"` — i.e.
it is one of the 15 crashed attempts in that seed dir (matches REVISION 1's disclosed "37 crashed" total
exactly: 15 + 22, independently re-derived, `run_metadata` count − `grep -l "Injection campaign complete"`
count, per seed dir). Yet `grep "task_234\.csv" POOL_MANIFEST.md5` returns a hit (`d9b5e3b4...`), and the
pool file itself has 300 rows, `code_rev f6449051` — content that cannot be the output of the captured,
cancelled run (which got through ≤103 of its draws before SIGTERM, with no final flush visible in the log).

**A3 — task-id collisions across seed dirs make "matched by (seed dir, SLURM task id)" ambiguous even where
content resolution happens to work.** `SLURM_ARRAY_TASK_ID` restarts at 0 in every one of the 7 independently
submitted seed-dir jobs; task ids 16/22/57 (spot-checked) exist in **every** one of the 7 dirs, all writing to
the *same* relative filename `injection_h_0p73_task_<id>.csv`, which the pool-build consolidation step then
places in one flat directory with no seed-dir component in the name. Disambiguating which (seed-dir, task-id)
pair produced a given pool CSV is possible only by comparing CSV content (row count, stratum breakdown)
against every candidate log's completion line — a procedure REVISION 1 does not register, is not guaranteed
unique in general, and (per A1/A2) the "current-format" heuristic it *does* register gets wrong in at least
two independently-verified, non-trivial-sized ways.

**Net effect:** REVISION 1's own justification sentence for the 707/363 split is false for the specific case
it invokes it for (the `a9f29e82` population), and the "37 crashed → excluded" clause is false for at least
one specific, checkable case (task 234). A builder implementing S1.1's numerator/denominator over "the 707
survivor tasks, current-format test" would (a) wrongly exclude `seed51000`'s 60 real, in-pool completions
(and their timeout events) from the M-bin denominator, while (b) having no registered procedure to recover
task 234's timeout history at all (its true producing run is not identifiable in the fetch under the
registered test). This is decisive for S1.1 and, through S1.1, for S1.2's `ρ(b)` (same denominator).
**Fix:** register content-based matching (row count + per-stratum counts, or better, a `code_rev`-vs-pool
join) as the actual test, explicitly superseding the "current-format" heuristic; and confirm task 234 (and any
other crashed-but-in-pool case, not yet enumerated) has a recoverable producing run before S1.1 is scoped as
computable at all — the current registered rule silently produces a wrong-by-construction count, not a
merely-imprecise one.

### Finding B (RED, new — not present in rev-0) — the registered PRIMARY denominator-leg insertion point is never dispatched by the frozen T0 anchor.

REVISION 1 item 3 / REVISION 2 (Q1) both designate `_mass_trunc_denominator_inner_m_integral` (`:869`,
called from `:8053` `if _use_mass_trunc`) as the **PRIMARY** denominator-leg site, with the alternative branch
`_bh_mass_denominator_inner_m_integral` (`:6835`, called from `:6901`/`:6979`) marked merely conditional
("only if `mass_trunc` OFF... reached-or-not asserted"). REVISION 2's own text flags this as needing the
builder to "assert the re-baseline flag" — this pass did that assertion directly, against the pinned T0 anchor
itself, not against a hypothetical:

```
_use_mass_trunc = resolve_host_mass_kernel(host_mass_kernel, normalization_mode, host_z_kernel) == "trunc_lognormal"
resolve_host_mass_kernel(..., "auto", normalization_mode, ...) resolves to "trunc_lognormal" iff normalization_mode == "mass_trunc"
```
(`bayesian_statistics.py:240-300`, direct read). Every one of the 41 `run_metadata_*.json` files under
`graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/` (the pinned frozen T0 anchor, `mean_h
0.6658540600`) — and every `run_metadata_*.json` under the `joint_r1` replicate directory — carries
`"normalization_mode": "absolute_marginal"` and `"host_mass_kernel": "auto"`, uniformly, no exceptions.
`resolve_host_mass_kernel("auto", "absolute_marginal", ...)` therefore resolves to `"gaussian"`, and
`_use_mass_trunc = False` for the **entire** frozen anchor and its `joint_r1` replicate — confirmed by direct
grep over all 41+41 files, not a sample.

Consequence: the code path the registration calls PRIMARY (`:869`/`:901`/`:944`, reached only when
`_use_mass_trunc` is `True`) is **never executed** when the S1.3 instrument is run against the pinned T0
convention. The code path the registration treats as the secondary, conditional fallback
(`_bh_mass_denominator_inner_m_integral`, `:6835`/`:6901`/`:6979`) is the one actually dispatched. A builder
who follows the registered table literally — installing the `P_complete`-multiplying proxy only at the
"PRIMARY" `:869` site — would build an instrument that silently touches nothing on the real run: `D'(h) ≡
D(h)` for every h-node, `δ^den(h) ≡ 0` throughout, and the read would report this as a genuine null
(`P_DET-MISSPECIFIED-IMMATERIAL` on the denominator leg) when it is in fact an instrumentation miss — the
proxy object was simply never queried on this leg. This is exactly the failure class the review is scoped to
catch: a formula that is "fully specified" and passes a literal read of the registered line numbers, but
whose central PRIMARY/conditional labeling is backwards for the one run the whole statistic is anchored to.
**Fix:** swap the registered PRIMARY/secondary labels for the denominator leg (`:6835`/`:6901`/`:6979` is the
site the T0 anchor and `joint_r1` actually dispatch; `:869`/`:901`/`:944` is the one that would need
`mass_trunc` to be turned on, which the anchor does not do), or register the proxy at **both** sites with an
assertion that exactly one of them is entered per h-node/host (a hard STOP, not a silent pass, if neither or
both fire) — consistent with REVISION 2's own precedent of "a mismatch between the asserted flag set and the
run's `run_metadata`/log flag lines = INSTRUMENT," which this pass shows the assertion needs to be run
*before* Phase B is written, not deferred to it.

### Finding C (GREEN, confirms F4 closed — stronger than rev-0's own re-check) — the 25-site table is byte-exact.

`grep -n` for both `detection_probability(...)?.detection_probability_with_bh_mass_interpolated(` and
`..._without_bh_mass_interpolated_zero_fill(` against `bayesian_statistics.py` returns **exactly** the 25
lines REVISION 2's table lists (901, 944, 1741, 2058, 3029, 5567, 6450, 6901, 6979, 7150, 7246, 7288, 9021,
9037, 9111, 9123, 9181 with-BH; 1284, 1440, 1770, 3066, 7697, 8533, 8954, 8973 without-BH) — no more, no
fewer, in one automated pass, not a hand count. Spot-checked (7 of the 25, exceeding the "spot-check 5"
instruction): `:901`/`:944` sit inside `_mass_trunc_denominator_inner_m_integral`/its batch twin exactly as
labeled; `:1284` inside `precompute_completion_denominator`; `:6450`/`:7150` inside the class method `p_Di`
(`def p_Di` at `:5932` — nested nature meant a naive top-level-`def` scan missed this on a first pass, filed
as a methodological note, not a finding, since the direct `grep -n "    def p_Di"` resolves it unambiguously);
`:7246`/`:7288` are the exact call lines inside `_mz_sel_2d_expectation`/`_batch` (`def` at `:7175`/`:7260`),
called from `:7966`/`:8013` exactly as the table's parenthetical states; `:9021` is inside
`single_host_likelihood_integration_testing` (`def` at `:8921`), confirmed (via `grep -rn` across
`darksiren_emri/`, excluding the test tree) to have **zero** production call sites — only test files and one
"Legacy global" comment reference it — matching the table's "NEVER (test-only kernel)" claim exactly. The
`mass_trunc`/`catalogue_leg_1d_mass_aware` flags the table's "reachability" column conditions on are both real,
existing config options (`normalization_mode == "mass_trunc"`, `catalogue_leg_1d_mass_aware ∈
{"auto","off","on"}`), not invented placeholders.

## 3. What checked out clean (independently verified, not assumed)

- S1.1 log-line format: both the per-draw `TimeoutError` line (carries `params={'M': ...}`) and the aggregate
  line (count only) reproduced from a live sample file, ≤5 lines read.
- Raw-vs-unique timeout-line dedup rule (REVISION 1 item 1): `.err` occurrences = 2,520, `.log` (application
  log) occurrences = 2,520, `.out` = 0 → 5,040 raw total, matching the registered figure exactly; the
  ".err/application-log duplicate" removal rule is well-specified and verified.
- `run_metadata_*.json` ↔ pool CSV matching mechanics: `SLURM_ARRAY_TASK_ID` field present, and for a
  non-colliding task id (234, unique to `seed51100` since no other dir's task range reaches it) the join to
  `injection_h_0p73_task_234.csv` is mechanical — the *content* of that specific join is what Finding A2
  flags, not the join mechanism itself.
- g-closure(i) (F5): SNR-stage ZeroDiv = 3,449, CRB-stage = 39, timeouts = 822, 100/100 files — all
  independently re-grepped on the pinned `simulate_6088772_*.err` files and matching the revision's corrected
  figures exactly; `89,456 − 3,449 − 85,584 = 423` reproduced.
- Build-log manifest pin (F6): `md5sum MANIFEST.md5` → `6ae9c1098c1c3325504e4904b2fc4d50`, `wc -l` → 3,510 —
  both exact matches to the new §1 pin.
- Global arithmetic: 60+243+60+60+216+216+215 = 1,070 run_metadata files across the 7 fetched seed dirs,
  matching REVISION 1's population size exactly; 1,033 complete-line tasks (707 + 326) + 37 crashed = 1,070.
- S1.3 numerator leg (`completion_mass_factor_g_sel`, `:2276`): signature, `s_query: Callable[...]` parameter,
  and docstring's stated `S_4D`-agnosticism all confirmed by direct read — a genuine zero-library-edit
  insertion point.
- [A13]/S1.4: correctly scoped to depend only on the (GREEN) numerator leg per `g-closure(iii)`'s own text.
- Bands, dispositions, blindness line, max_revisions accounting: all present and internally consistent on
  direct re-read of the full, current text of record.

## 4. Gates as they bind Q1

- **G-1 pins:** GREEN for every §1 Q1 pin re-verified this pass, including the new build-log MANIFEST.md5.
- **g-population:** GREEN for the tallies independently reproduced (3,449/39/822/100-of-100); **RED-adjacent**
  still: no registered `g-closure` check exists over the pool-build side's own attempt/completion counts, so
  nothing in the design would have caught Finding A mechanically before a builder committed to a number —
  filed as in rev-0, superseded by fixing Finding A directly rather than as its own numbered item.
- **g-closure(i):** GREEN, independently confirmed.
- **g-formula:** cannot be scored GREEN/RED in the abstract (Phase B / `DESIGN_GATE_formula.md` for Q1 has not
  been written) but Finding B is a concrete, decisive input to it: the eventual formula-gate's synthetic
  fixture and its "confirm which asserted rows were reached" check (REVISION 2's own closing sentence) MUST
  be run — and its assertion must be checked against the *actual* T0 `run_metadata`, which this pass already
  did and which contradicts the registered PRIMARY/secondary labels.
- **g-scope, g-hardware, g-byteid:** not implicated by anything in this pass (byteid is Q2-scoped per the
  CHAIR ERRATUM; hardware's local half was already GREEN in rev-0 and nothing here touches it).

## 5. Bottom line for the builder

Do not launch S1.1/S1.2/S1.3 as currently registered. Two defects sit on the PRIMARY statistic's own
feasibility, both newly, directly verified against real files in this pass (not inherited claims):
**(Finding A)** the 707/363 population split REVISION 1 registers to close rev-0's F2 is itself wrong — the
"current log format" test misclassifies the entire disclosed `a9f29e82` in-pool sub-population (an
old-format-but-included seed dir) and cannot account for at least one disclosed "crashed" attempt (task 234)
whose CSV is nonetheless in the pool of record; **(Finding B)** the denominator-leg insertion point REVISION
1/2 register as PRIMARY (`_mass_trunc_denominator_inner_m_integral`, `:869`/`:901`/`:944`) is never dispatched
by the frozen T0 anchor or its `joint_r1` replicate (`_use_mass_trunc = False` uniformly across all 41+41
`run_metadata_*.json` files, verified, not sampled) — the branch the anchor actually runs
(`_bh_mass_denominator_inner_m_integral`, `:6835`/`:6901`/`:6979`) is the one the registration treats as
secondary/conditional. Left as-is, a Phase-B build would silently produce a manufactured null on the
denominator leg and a wrong M-bin numerator/denominator on S1.1/S1.2, not a genuine, trustworthy read of
either. Everything else — the numerator leg, the 25-site call-site table (byte-exact on independent
re-derivation, F4 fully closed), g-closure(i) arithmetic (F5, independently reproduced), the build-log
manifest pin (F6), S1.4's form, the bands/dispositions/blindness/revision-counter bookkeeping — is
computable/clean exactly as registered.
