# DESIGN_GATE_Q1_computability — r-timeout-selection, `q-timeout-selection-pdet` ONLY

Reviewer: FRESH computability-only pass, no prior read of this node's registered statistics. Scope: Q1
exclusively (S1.1–S1.4, the Q1-S1.2/Q1-S1.3 disposition rows, the g-* gates as they bind Q1, `max_revisions`,
the blindness line). Q2 is out of scope and not re-litigated (its own four gate documents, `DESIGN_GATE_Q2_*`,
already carried it to GREEN independently). `INFORMATION_FORECAST.md` was **not opened** (FORBIDDEN, honored).
Source reviewed: `REGISTRATION_DRAFT.md` (all revisions + `CHAIR ERRATUM`), `MECHANISM_NOTE.md`, the newly
fetched `injection_pool_mix200k_20260728_buildlogs_fetch_20260904` (3509 log-type files + `MANIFEST.md5`,
verified directly, not taken on faith), every §1 pin on disk (md5/row-count), and the named library functions
in `darksiren_emri/bayesian_inference/bayesian_statistics.py` / `simulation_detection_probability.py` at HEAD
`79c44608` by direct read (signatures, docstrings, call sites — never executed, never imported).

**Blindness line.** No registered Q1 statistic (S1.1–S1.4, `ρ(b)`, `Δmean_h^{Q1}`, `δ_e`, `δ^den`, any p_det
or `P_complete` value) was computed over the registered population. What was run: `md5sum`/`wc -l`/
`md5sum -c` on every §1 pin and on the new log-fetch manifest; direct `grep -c`/`grep -n` counts of named
log-message strings across the pinned/fetched log trees (timeout tallies, ZeroDivisionError tallies, node
names, ID uniqueness — the same class of small, non-registered reproduction the Q2 gates used); direct
reads of the named library functions' signatures, docstrings and call sites (never called). No `--dry-run`
or synthetic-table scorer run was made (the scorer does not yet exist for Q1; Phase A/B have not launched).

**Verdict: RED.** The PRIMARY statistic (S1.3) has a decisive, code-verified gap on its denominator leg, and
the statistic that gates Q1's ceiling (S1.1) is under-specified against the real, newly fetched log data in
a way that is not cosmetic: the fetched logs contain two structurally different populations of task-attempts
that the registered formula does not distinguish. Full detail below; ≥8 checks, each independently verified
on disk in this pass, not inherited from any prior document.

## 1. Enumeration and per-item computability

| # | Q1 item | On disk? | Formula fully specified (zero fresh choices)? | Verdict |
|---|---|---|---|---|
| S1.1 | pool-side `P_complete^pool(M bin)` from the pool build's own timeout tally | YES, now — the log fetch (§8 item A) is present: 7 dated `injection_20260728-*` dirs, `MANIFEST.md5` (3509 files) | **NO** — the formula's own named source (`main.py:1349-1352`, the per-task aggregate line) cannot deliver a per-M-bin count; the per-draw line that can is unnamed in §4; and the fetch itself contains two incompatible task-attempt populations with no registered reconciliation rule | **RED (F1, F2)** |
| S1.2 | `ρ(b) = P_complete^sim/P_complete^pool`, Garwood 95% CI, `R = max_b|ln ρ(b)|` | YES — CRB CSV, seed61000 logs, pool a-stratum all pinned and verified | Formula itself is well-specified (ratio + CI on a defined quantity) — **but its denominator, `P_complete^pool(b)`, is exactly S1.1's undefined quantity** | **RED, inherited from S1.1** |
| **S1.3** | **PRIMARY** — `δ_e(h)` (numerator leg, completion-leg mass quadrature) + `δ^den(h)` (denominator leg, `D'(h)` vs `D(h)`, `ρ(M_z)`-weighted) | Library functions present at the cited names (`completion_mass_factor_g` :2143, `completion_mass_factor_g_sel` :2276, `precompute_completion_denominator` :1170, `SimulationDetectionProbability`) | Numerator leg: **AMBER** — a zero-edit insertion point exists (`completion_mass_factor_g_sel`'s `s_query` callable) but is never named. Denominator leg: **NO** — `precompute_completion_denominator`'s own p_det call has **no mass argument at all** (confirmed by direct read, see F3); there is no mass quadrature inside the one function §1/§4 name for a `ρ(M_z)` weight to multiply into. | **RED (F3, F4)** |
| S1.4 | engagement gate [A13]: ≥10% of events with `\|δ_e(0.73)\| ≥ 1e-6` | n/a (derived from S1.3's numerator leg only, confirmed by `g-closure(iii)`'s own text: `δ^den` is one scalar per h, not per-event) | YES, contingent — well-formed given `δ_e` is defined; the numerator leg (AMBER above) is the only dependency, not the broken denominator leg | GREEN, contingent on S1.3 numerator-leg AMBER being closed |
| g-closure(i) | `89,456 − 3,488 (ZeroDiv) − 85,584 = 384` residual, STOP if > 1,000 | YES — seed61000 logs pinned | **Arithmetic uses the wrong figure**: `3,488` is the Q2-gate-identified SNR+CRB **combined** ZeroDivisionError total; the SNR-stage-only figure this closure needs is `3,449` (independently re-derived, F5). Recomputed residual is `423`, not `384` — still `< 1,000` (does not flip the STOP), but the registered number is wrong. | **AMBER (F5)** |
| g-hardware | node list `uc2n561…579` mapped to GPU type (chair-only, §8 item B) | Node names YES (47 distinct `uc2n5xx` names in the `.out` logs, independently reproduced) — GPU-type mapping itself correctly deferred (needs a live `scontrol` read, not local) | YES on the local half; the deferred half is correctly scoped as chair-only, not a Q1-computability gap | GREEN |
| G-1 pins | every §1 md5/count | Independently re-verified in this pass: CRB CSV (`9a1f2a14…`, exact), bin-edges JSON (`e24b07fe…`, exact), `POOL_MANIFEST.md5` (`75f4030d…`, exact, and all 707 listed pool CSVs verify OK against it), seed61000 log manifest (`ebf09fc4…`, exact, 100/100 `simulate_6088772_*.err` files, clean `md5sum -c`) | — | GREEN, **except** the new log fetch's own `MANIFEST.md5` is not yet a §1 pin at all (F6) |
| max_revisions | header: "max_revisions 2" | REVISION 1 + REVISION 2 (both Q2-scoped) + the `CHAIR ERRATUM` (closes a Q2 gate finding) are the consumed history; Q1 has not yet had a revision | present, budget not exceeded on its face | GREEN on form |
| Blindness line | §10, disclosed partial pre-reads (i)/(ii)/(iii) | — | present; none of the three disclosed items touches a Q1 statistic (all are S2.3/`g-population` inputs) — **Q1 itself has no disclosed pre-read, consistent with "the registration author has NOT computed any registered statistic"** | GREEN |
| Q1-S1.2/S1.3 dispositions | 3-valued, fresh RULE, caps (NOT-EVALUABLE → INTERMEDIATE ceiling; INSTRUMENT if S1.4 fails) | n/a | tags present, fresh-RULE line present, caps stated correctly | GREEN on form; **inherits RED from S1.1/S1.3's own gaps** — a disposition cannot be trusted while the statistic feeding it has an unresolved, code-contradicted formula |

## 2. Findings

**F1 — S1.1's named log source cannot produce a per-M-bin statistic; the source that can is unnamed in the formula.**
§4 states: *"Pool-side `P_complete^pool(b)` needs the pool build's own timeout tally (`main.py:1349-1352`
line, cluster log, NOT local → §8 item A)."* At HEAD, `main.py:1349-1352` is confirmed (direct read) to be
the **per-task aggregate** log line:
```
f"Injection campaign complete: {len(results)} events stored to {csv_path} "
f"(skipped: {skipped_high_z} high-z, {separatrix_sign_skips} separatrix-sign, "
f"{timeout_count} timeouts @ {_TIMEOUT_S}s); realized stratum counts: a=... b=... c=..."
```
This line carries a per-task **count**, never an `M` value — it cannot be binned by mass under any reading.
The line that carries `M` is a different code site, the per-draw `TimeoutError` catch (`main.py:1293-1302` at
HEAD; logged from `injection_campaign()` at line 1143 in the pool-build commit, confirmed against the fetched
logs):
```
2026-07-28 08:58:06,545 [main.py:1143 - injection_campaign()] Injection waveform/SNR computation timed out
(>90s, 1 total). Skipping event... params={'M': 261101.18617576532, 'mu': 10, 'a': 0.98, 'p0': ...}
```
5,040 raw occurrences of this line are confirmed present in the fetch (2,520 unique events after removing the
`.err`-vs-application-log duplicate copy, §F2). §4's formula sentence names only the aggregate line; the
per-draw line appears nowhere in §4, only as a hedge in §8 item A's parenthetical ("+ any params dicts") —
a cost-section aside, not part of the registered formula text. A Phase-A builder following §4 literally, as
"formula fully specified with zero fresh choices" requires, fetches a line that cannot answer the question
S1.1 asks. **Fix:** name the per-draw `TimeoutError` catch's warning line explicitly as S1.1's source in §4,
not only in §8's cost aside.

**F2 — the fetched build logs contain two incompatible task-attempt populations; S1.1 has no rule to
reconcile them, and the size of the gap is large (34% of raw task-attempts).**

Direct counts, this pass, on the newly fetched `injection_pool_mix200k_20260728_buildlogs_fetch_20260904`:

| quantity | count | source |
|---|---:|---|
| total injection-campaign task-attempts (run_metadata_*.json) across the 7 seed dirs | **1,070** | `find . -name run_metadata_*.json \| wc -l` |
| pool-of-record CSV files (`injection_pool_mix200k_20260728/*.csv`) | **707** | `ls *.csv \| wc -l`, matches the §1 pin exactly |
| task-attempts whose `.err` carries an "Injection campaign complete" summary in the **current** log format (has a `separatrix-sign` field, matching HEAD's f-string) | **707** | `grep` for `"separatrix"` inside the complete-line, exact match to the pool-file count |
| task-attempts whose summary line lacks that field (an **older** log-line format, from before a mid-campaign code change) | 326 | same grep, negated |
| task-attempts with **no** completion line at all (crashed/killed mid-run — `run_metadata` exists, no terminal summary) | 37 | `run_metadata` count minus files-with-a-complete-line |
| 326 + 37 (= 1,070 − 707) | **363** | arithmetic check, exact |

The 707/326/37 split is not noise: it lines up exactly with the pool-file count (707) and independently
corroborates the pool's own disclosed provenance split — `MECHANISM_NOTE.md` §1 records that the pool of
record carries two `code_rev` values, `f6449051` (194,100 rows) and `a9f29e82` (6,000 rows, p0/t_plunge NaN,
"before the p0 provenance columns existed"). The natural reading: a mid-campaign code fix on 2026-07-28 added
a field to the injection loop (the `separatrix-sign` skip counter, and evidently the p0/t_plunge columns);
every task that ran under the pre-fix code, or crashed outright, was **re-run** under the post-fix code in
the same working directory, and only the surviving re-run's CSV entered the pool — the pre-fix/crashed
attempts' logs remain in the fetch (nothing deletes them) but their events are **not** in
`injection_pool_mix200k_20260728`.

This matters for S1.1's own arithmetic, independently verified in this pass:
```
sum of "N timeouts" over the 1,033 complete-line tasks (707 new-format + 326 old-format):  2,475
count of individual per-draw "timed out" WARNING lines, ALL 1,070 task-attempts:            2,520
```
The 45-event gap (2,520 − 2,475) is exactly attributable to the 37 crashed task-attempts: their individual
timeout events were logged before the crash but never rolled into a final tally line, since the task never
reached its `_flush_injection_results`/summary call. Neither `REGISTRATION_DRAFT.md` nor `MECHANISM_NOTE.md`
states whether S1.1's per-bin numerator should (a) use only the 707 survivor tasks' timeout events (excluding
the 363 superseded/crashed attempts entirely), (b) use all 1,070 task-attempts' timeout events (including
events from runs whose "completed" counterpart is not verifiably in the pool at all), or (c) something else.
This is not a stylistic gap: option (b) counts timeout events against a denominator (`N_pool_a(b)`, from the
707-file pool) that structurally cannot include the corresponding completions from the 363 superseded
task-attempts — a systematic downward bias on `P_complete^pool(b)` in whichever M bins those superseded
timeouts fall in, of a size (up to a third of raw task-attempts) that is not obviously sub-threshold. This is
exactly the class of defect this review's own charter is scoped to catch: a real, unregistered, materially-
sized choice on the input feeding the PRIMARY statistic's ceiling gate.
**Fix:** register which task-attempt population (707 survivors only, recommended; or an explicit,
justified alternative) feeds S1.1's numerator and denominator, using the `separatrix-sign`-field test (or an
equivalent, more robust discriminator — e.g. a `code_rev` stamp per log, if one can be recovered from
`run_metadata_*.json`'s `git_commit` field) to make the split mechanical.

**F3 — S1.3's denominator leg names the one function that provably has no mass axis to reweight.**
S1.3 registers: *"`δ^den(h) = −[ln D'(h) − ln D(h)]` from the M_z-weighted pool (weights `ρ(M_z)`...)"*, and
§1 names `precompute_completion_denominator` (`bayesian_statistics.py:1170`) as the "convention/grid" for
`D(h)`. Direct read of that function's integrand (`bayesian_statistics.py:1230-1298`), non-sky-aware branch:
```python
phi = np.zeros_like(z)      # marginalized; value does not matter
theta = np.zeros_like(z)
p_det = detection_probability_obj.detection_probability_without_bh_mass_interpolated_zero_fill(
    d_L, phi, theta, h=_h, ...
)
...
# Population prior R_EMRI(z,M)/(1+z) * dVc/dz (emri_rate.p_pop_unnormalized):
# ... The mass-integrated rate INTEGRAL dM R_EMRI(z,M) is z-independent under
# the p0=1 surrogate, so it is an overall constant that cancels ...;
# only 1/(1+z) survives here.
```
`p_det` here is queried at **no mass at all** — `M` is not a parameter of
`detection_probability_without_bh_mass_interpolated_zero_fill` in this call, and the code's own comment
states explicitly that the mass integral has already been analytically cancelled as a constant. There is
therefore no quadrature node in `M` anywhere inside `precompute_completion_denominator` for a per-`M_z`
weight `ρ(M_z)` to multiply into — the registered formula asks for something this function structurally
cannot provide. (`MECHANISM_NOTE.md` §4 itself already half-discloses this — "the `emri_rate` measure in
`D(h)` is mass-integrated and z-only" — but S1.3's own formula in `REGISTRATION_DRAFT.md` does not draw the
consequence: a z-only integral has no `ρ(M_z)` to apply.) The library function that DOES call p_det inside an
actual mass quadrature with the with-BH-mass interpolator — `_mass_trunc_denominator_inner_m_integral`
(`bayesian_statistics.py:869`, batched twin at `:926`) — is never named anywhere in `REGISTRATION_DRAFT.md`
or `MECHANISM_NOTE.md`, is a single-underscore **private** helper with exactly one call site
(`bayesian_statistics.py:8053`, inside a different, unnamed enclosing function far from
`precompute_completion_denominator`), and is not reachable through the one function §1 pins. A builder has no
registered, named library entry point that both (a) is cited in the registration and (b) has a mass axis.
**Fix:** either (i) name `_mass_trunc_denominator_inner_m_integral`/its enclosing function explicitly as
S1.3's denominator-leg entry point and register the import-a-private-name exception, or (ii) redefine
`δ^den(h)` as a reweighting of the `SimulationDetectionProbability` pool construction itself (multiply each
pool row's contribution by `ρ(M_z(row))` before `p_det`/`d_hor` are built) — a materially different mechanism
that also touches the numerator leg's `detection_probability_obj` (§6 below) and must not be conflated with
option (i) silently.

**F4 (feeds the g-formula gate) — at least 8 distinct call sites of the with/without-BH p_det interpolators
exist in `bayesian_statistics.py`; the registration does not say which are "inside a mass quadrature."**
`g-formula` states: *"the `P_complete` factor enters ONLY inside `p_det` calls within mass quadratures (never
the 1D no-BH numerator survival, never the catalogue candidate survival at fixed observed M) — the verifier
lists every call site the script reaches."* Direct `grep -n` in this pass finds
`detection_probability_with_bh_mass_interpolated` called at lines 901, 944, 1741, 2058, 3029, 5567 (six sites)
and `detection_probability_without_bh_mass_interpolated_zero_fill` at 1284, 1440, 1770, 3066 (four sites) —
ten call sites of two method names, spread across the with-BH numerator, the without-BH numerator, the
completion denominators (both variants), and `precompute_global_catalog_selection`. Distinguishing "inside a
mass quadrature" from "at fixed observed M" or "the 1D leg" among ten same-named-method call sites cannot be
done by wrapping the method generically on the `SimulationDetectionProbability` object (a single wrap would
touch all ten); it requires call-site-specific instrumentation, which in turn requires each site to be
individually classified — work the registration defers entirely to "the verifier," with no enumeration
started here. This is squarely what the launch instruction's own precedent (`exec/r-offset-subset/`'s
`DESIGN_GATE_formula*.md` chain, which went through 5 revisions before closing) suggests is hard to get right
on the first attempt; Q1 has not started that chain at all yet.

**F5 — g-closure(i)'s registered ZeroDivisionError figure is the wrong scope; independently re-derived in
this pass.** §6 states the closure `89,456 − 3,488 (ZeroDiv) − 85,584 = 384`, citing `3,488` as "ZeroDiv."
Direct, independent re-count on the pinned `simulate_6088772_*.err` files (100/100, matching the §1 pin) by
exact catch-site log message:
```
grep -c "Caught ZeroDivisionError during trajectory integration" simulate_6088772_*.err  -> 3,449  (SNR stage)
grep -c "Caught ZeroDivisionError during CRB computation"        simulate_6088772_*.err  ->    39  (CRB stage)
```
`3,449 + 39 = 3,488` exactly — `3,488` is the **combined** SNR+CRB total (this reproduces, independently, the
same combined-vs-labeled-as-SNR-only defect the Q2 `rev2` gate flagged in `MECHANISM_NOTE.md` §3's table row
and explicitly filed as "Q1-scoped… feeds Q1's S1.2 completed-draw scale factor, not any Q2 statistic" —
confirmed here, in the Q1 gate, to be exactly correct: it is this closure). But `g-closure(i)` is summing
against `89,456`, which is the **SNR-stage-only** loop-iteration count (every draw that reaches the SNR try
block); CRB-stage ZeroDivisionErrors occur on the SNR-passing subset **after** that count is already taken,
so mixing the combined `3,488` into an SNR-stage-only closure double-subtracts the 39 CRB-stage events.
Recomputed with the correct SNR-stage-only figure: `89,456 − 3,449 − 85,584 = 423` (not `384`). This does not
flip the `residual > 1,000 = STOP` gate (423 is still comfortably under it), so the read is not blocked — but
the registered number is arithmetically wrong as written, and `g-closure(i)` is a live STOP-gated check, not
descriptive prose; a builder who trusts the printed `384` and later finds `423` on their own re-derivation
would (correctly) treat it as a fresh discrepancy rather than recognizing it as this same, already-diagnosed
label error. **Fix:** correct `384` → `423` (and `3,488` → `3,449`) in §6's g-closure(i) line, with a
one-clause cross-reference to the Q2 `rev2` gate's §5 finding so a future reader does not re-open it as new.

**F6 (minor, non-blocking) — the new log-fetch `MANIFEST.md5` is unpinned in §1, and fails its own
self-check by construction.** `md5sum -c` on
`gate_b_20260730/injection_pool_mix200k_20260728_buildlogs_fetch_20260904/MANIFEST.md5` reports 3,509 OK and
**1 FAILED**: the manifest's own entry for itself,
`d41d8cd98f00b204e9800998ecf8427e ./MANIFEST.md5` — the md5 of an **empty file**, i.e. the manifest was
hashed (as a self-referential row) before it was written, a standard and benign artifact of manifest
generation, not data corruption (independently confirmed: every one of the other 3,509 rows passes). The
launch instruction's "verify md5s incl. the new log manifest" is satisfiable, but a naive
`md5sum -c $whole_manifest; test $? -eq 0` STOP-gate (the literal reading of `REGISTRATION_DRAFT.md` §6's
"STOP on mismatch" for `G-1 pins`) would misfire on this one self-referential row every time the manifest is
re-verified. **Fix:** add `injection_pool_mix200k_20260728_buildlogs_fetch_20260904/MANIFEST.md5` as a named
§1 pin (its own md5, computed over the 3,509 real rows, excluding the self-referential last line — or
regenerate it without the self-row) before Phase A treats "verify md5s" as a hard STOP gate.

## 3. What checked out clean (verified directly, not assumed)

- `prepared_cramer_rao_bounds.csv`: md5 `9a1f2a14384a9281c97ca3be312ddaab` — exact match.
- `design_gate_bin_edges.json`: md5 `e24b07fe3948559b02d8dd4dbe8df8b3` — exact match.
- `POOL_MANIFEST.md5` (beside the draft): md5 `75f4030d5d3b0405fd948049bef5767e` — exact match; `md5sum -c`
  against the actual `injection_pool_mix200k_20260728/*.csv` files: **707/707 OK**, no failures — a clean
  pass (contrast F6).
- `seed61000/cluster_logs_fetch_20260904_MANIFEST.md5`: md5 `ebf09fc4ab66b55e4eb592731ee46ae6` — exact
  match; `md5sum -c` against the fetched logs: **clean, 0 failures**; exactly 100 `simulate_6088772_*.err`
  files present, matching the "100/100 tasks" pin.
- Node names: 47 distinct `uc2n5xx.localdomain` hostnames confirmed present in the `.out` "Node:" lines,
  consistent with the `uc2n561…579` range the draft names; the GPU-type lookup itself is correctly deferred
  as a chair-only cluster read (§8 item B), not something this pass needed or attempted.
- `completion_mass_factor_g` (`:2143`) and `completion_mass_factor_g_sel` (`:2276`) both exist under exactly
  those names, with the documented signatures; `completion_mass_factor_g_sel`'s `s_query` parameter is
  confirmed to be an injected `Callable`, not a hardcoded internal p_det call — a genuine zero-library-edit
  insertion point exists for the numerator leg (F3's problem is the denominator leg specifically, not this
  one).
- `precompute_completion_denominator` exists at the cited line and signature; its docstring's own claim ("D(h)
  is FULL-volume... no `(1-f)` factor") is consistent with the code read.
- §10's blindness line: none of its three disclosed pre-reads (pool a-stratum support fractions; the S2.3
  `share_pool,det` bin-2 figure; the `g-population` log tallies) touches a Q1 statistic — confirmed by
  re-reading each against the Q1 statistic list; Q1 itself carries no disclosed pre-read, consistent with
  "the registration author has NOT computed any registered statistic."
- The mandatory p0-axis disclosure line in §5 is present and its wording matches `MECHANISM_NOTE.md` §3's own
  disclosed non-evaluability finding — no drift between the two documents on this point.

## 4. Gates as they bind Q1

- **G-1 pins:** GREEN for every §1 Q1 input independently re-verified in this pass. **Gap:** the new log
  fetch's `MANIFEST.md5` (F6) is not itself a §1 row — a live gate ("verify md5s incl. the new log manifest")
  has nothing pinned to check against yet.
- **g-byteid:** not independently re-litigated here (it is a Q2-anchored gate per the erratum's own scope,
  `n_kept`/`n_timeout` on the 1588-event population); no Q1 statistic depends on it.
- **g-population:** GREEN for the Q1-relevant tallies this pass independently reproduced: 100/100
  `simulate_6088772_*.err` tasks, Σ (SNR ZeroDiv) = 3,449, Σ (CRB ZeroDiv) = 39. **RED-adjacent:** the pool
  build's own analogous per-task iteration/outcome tallies are **not** covered by any registered `g-closure`
  check at all (only the simulate loop's 89,456-iteration closure exists) — exactly the kind of check that
  would have surfaced F2's 707-vs-1,070 split mechanically before Phase A commits to a number. Filed as an
  informational gap, not a numbered finding, since fixing F1/F2 directly supersedes it.
- **g-closure(i):** **AMBER (F5)** — arithmetically wrong as printed (`384` should be `423`), though the STOP
  threshold is not crossed either way.
- **g-closure(iii)** ("Q1 `δ^den(h)` applied identically to all 1588 events, one scalar per h"): the rule
  itself is fine and easy to implement mechanically **once `δ^den(h)` is computable at all** — it inherits
  F3's blockage, not a defect of its own.
- **g-formula:** cannot be evaluated as GREEN or RED in the abstract (Q1's own `DESIGN_GATE_formula.md` has
  not been written — Phase B has not started) but F4 flags a concrete, non-trivial enumeration burden
  (≥10 call sites, 2 method names, 4 semantic roles) that the eventual formula gate must resolve before any
  real-mode run, consistent with the launch instruction's own emphasis that "one invocation covers everything
  the dispositions need."
- **g-hardware:** GREEN, local half; deferred half correctly scoped to the chair.
- **g-scope:** not implicated by anything in this pass — no Q1 statistic touches a p0 bin outside the
  REPORTED-ONLY carve-out named elsewhere.

## 5. Bottom line for the builder

Do not launch S1.1/S1.2/S1.3 as written. Three defects sit on the PRIMARY statistic's own feasibility, not on
peripheral bookkeeping: (F1) S1.1's formula cites a log line that cannot deliver a per-M-bin count — name the
per-draw `TimeoutError` line explicitly; (F2) the newly fetched build logs contain two structurally different
task-attempt populations (707 pool-surviving vs. 363 superseded/crashed, confirmed by an independent,
reproducible log-format signature) with no registered rule for which feeds S1.1's numerator/denominator —
this is a real, non-trivial-sized ambiguity, not a rounding question; (F3) S1.3's own denominator-leg formula
asks for a `ρ(M_z)`-weighted `D'(h)`, but the one function the registration names for `D(h)` is confirmed, by
direct code read including its own inline comment, to have no mass axis at all — the function that does is
private, unnamed, and reached through a different, unnamed caller. F4 is the same gap seen from the
`g-formula` gate's side: at least ten call sites of the two p_det interpolator methods exist, spread across
four different semantic roles, and nothing in the registration starts the enumeration the gate itself
demands. F5 (arithmetic label error, STOP threshold not crossed) and F6 (an unpinned, self-referentially-
failing manifest) are real but non-blocking — fix them alongside F1–F4, not as gating items on their own.
Everything else — S1.4's form (contingent only on the numerator-leg AMBER, not the denominator RED), the
disposition tables' form, `max_revisions`, the blindness line, and every independently-reproduced pin — is
computable/clean exactly as registered.
