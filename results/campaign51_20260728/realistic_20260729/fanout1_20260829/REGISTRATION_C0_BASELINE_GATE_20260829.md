# REGISTRATION C0 — shared baseline gate task (2026-08-29)

**Launched under rows #222/#223 — charter node C0.** Closes GAP 2 of
`WAVE2_REGISTRATION_CHECK_20260829.md` §1.1/§5 item 2. Purpose: INFORMATION + registration
for the orchestrator (row #222 form (ii)); not an approval request. Append-only.

## 1. Purpose

Certify, at production scale and at default flag values, the four estimator commits landed
since the banked HEAD readout — `d40fe5c8` ([HIER] θ-hook C1+C2), `1f003da6` (θ-hook s-placement
align), `0b308828` (mass-window geometry flag), `901653a1` (driver passthrough) — plus the
wave-2 commit that will land on top of them (`git log d04d9dc9..dd63fe0c -- darksiren_emri/`:
exactly these 4 commits, 434+/18− over 5 files, verified 2026-08-29). PASS makes the banked
HEAD readout `results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/`
(commit `d04d9dc9bfe39e6c5a72e768a26f2dcc38355bf5`, 2026-08-27T19:40:20) the zero-compute
baseline for C3 and C4 (docket L5), and the truth node (θ=(0,1)) for C1/PA-HIER-31 item 4.
All four defaults are byte-identical to pre-flag behaviour by construction (gate-ledger rows
2026-08-28/29), so this is a reproduction gate, not a correctness check (A10 blindness, §5).

## 2. The run

- **Venue:** iiib. **h = 0.730 only** (`evaluate.sbatch --array=21`).
- **CLI, verbatim from `headreadout_20260827/iiib/run_metadata_21.json:cli_args`:**
  `--h_value 0.73 --strategy physics-floor --pdet_dl_bins 60 --pdet_mass_bins 40
  --pdet_estimator local_linear --pdet_z_resolved --fisher_cond_threshold 1e16
  --host_z_kernel volume_deconv --host_mass_kernel auto --normalization_mode absolute_marginal
  --selection_in_completion_numerator fused --catalogue_mass_overlap production
  --catalogue_mass_error_scale 1.0 --completion_b_scale derived --eddington_m on
  --sigma4d_mass_kernel point --completion_event_measure ratio
  --catalogue_numerator_survival_2d off --catalogue_global_selection phi`
  (`smear_global_selection`, `pdet_wbh_z_resolved`, `realize_observed_catalogue`, `max_redshift`
  all at their recorded defaults: `False`, `False`, `False`, `null`.)
- **PLUS explicit new-commit defaults** (not present in `run_metadata_21.json` — post-dated by
  the four commits above, hence stated explicitly rather than quoted): `--mass_filter_geometry
  linear --mass_filter_k 1.5` (`0b308828`, byte-identical default), `theta_b=0.0`, `theta_s=1.0`,
  `theta_sites="all"` (`d40fe5c8`/`1f003da6` engine defaults — no CLI flag exists yet for these,
  F-C below; they apply as the code's own defaults regardless), `smear_global_selection=False`
  (already in the quoted CLI), `completion_population_prior`: **not present** — no such flag
  exists in the tree (C2 struck, `B3_2_POP_FLAG_RECORD.md`).
- **Cluster shape:** `cpu,cpu_il` partition, 1 task, 16 cpus/task, `--time=03:00:00`
  (`cluster/evaluate.sbatch` header; `cluster/LAUNCHING_JOBS.md` §2a node-topology notes: measured
  anchor 56–76 min/h-value at 16 cpus, 3355 events — this run is 1588 events, single h-value, well
  inside the walltime). Submit per the `/cluster` skill's standard `submit_pipeline.sh` recipe with
  `--array` narrowed to the single h=0.730 index (task 21) against a `RUN_DIR` pointed at the banked
  `seed61000` prepared CRB (md5 `9a1f2a14384a9281c97ca3be312ddaab`, 1590 rows / 1588 scored) and the
  reduced GLADE catalogue (md5 `c52c13b5…`) — same inputs as the banked readout (A11 dataset pin).
  Preflight `VERDICT: READY ✓` required before submit (`/cluster` skill, hard gate).

## 3. The gate

**Column set — resolving the §1.1/§3.2 inconsistency.** The banked `event_likelihoods.csv`
header (verified by direct read, 2026-08-29, identical across every banked/mirror instance
checked: `headreadout_20260827/iiib/`, `off_iiib/`, `joint_r1/`, `off_joint_r1/`,
`p3_b0_work/*`, `p3_2d_fleet_20260825/*`, `hier_s0_*`) is exactly:

`event_idx, h, w_G, w_G_legacy, w_tilde_G, alpha_G_phi, r_Malm, D_tilde_phi, L_cat_no_bh,
L_cat_with_bh, B_num, B_num_wbh, g_frac, L_comp, combined_no_bh, combined_with_bh`

— **16 fields total: 1 join key (`event_idx`) + 15 numeric columns.** The GAP doc's "17 numeric
columns" (§1.1, §3 item 2) over-counts by one: it folds in `beta_G_phi`, which is a derived
quantity used only inside `precompute_phi_selection_integrals` (`bayesian_statistics.py:4199-
4203`) and is **not written to `event_likelihoods.csv`** — it was pulled in solely because C2
(struck, §5) named it as a consumed quantity. With C2 struck, the correct gate is the file's
own 15 numeric columns plus the `event_idx`/`h` identity check (16 fields, all checked).

**Per-arm consumption (resolves which columns each downstream arm actually needs, so a
column-level FAIL can be triaged to "does this block C1/C3/C4" before escalating):**

| arm | columns consumed |
|---|---|
| **C1** (PA-HIER-31 S0-B) | `combined_no_bh` (primary channel, item 5), `combined_with_bh` (secondary, invariant (e)), `L_cat_no_bh` (C-C class definition, item 2) |
| **C3** (log k=3 counterfactual) | `L_cat_with_bh`, `combined_with_bh` (primary); `L_cat_no_bh`, `combined_no_bh` (R6 attribution check) |
| **C4** (PROD-CF-2D) | `L_cat_with_bh`, `combined_with_bh` |
| **B8.2** (harness, local, not a cluster arm) | all 15 numeric columns (S2(iii) engagement gate) |
| ~~C2~~ (struck) | ~~`combined_no_bh`, `D_tilde_phi`, `beta_G_phi`~~ — moot |

**Band:** max |relative difference| ≤ 1e-12 on every one of the 15 numeric columns, over all
1588 scored events, at h = 0.730, against the banked `d04d9dc9` rows (1588 rows out of
`headreadout_20260827/iiib/event_likelihoods.csv`'s 65,109 data rows = 1588 × 41 h-values + 1
header). PROD-A0 historical form for comparison: ≤ 8.5e-15 over 12 ingredient columns
(ledger row #201, `gate_b_20260730/BIAS_HISTORY_LEDGER.md:2957`, 2026-08-25) — a stricter floor
on fewer columns from a prior (asymmetric-vs-symmetric) reproduction; this gate's 1e-12 band is
looser to tolerate the four new commits' additional floating-point paths (θ=(0,1) literal-skip
branches, `d40fe5c8`) while still being far tighter than any physically meaningful signal.
**Also gated:** `posteriors/h_0_73.json` and `posteriors_with_bh_mass/h_0_73.json` (both channels'
posterior objects at h=0.73) bit-identical or ≤1e-12 relative on every numeric field.

PASS ⇒ banked HEAD readout is reused as baseline for C1/C3/C4 at zero additional compute.
FAIL ⇒ fallback: C3 and C4 each run their own 4-node baseline (+59.7–91.6 CPU-h and
+59.7–81.1 CPU-h respectively; `WAVE2_REGISTRATION_CHECK_20260829.md:107`; proposal §6.2 "H4
with full baseline re-run 119.4–162.2 CPU-h" is the combined ceiling) and the per-column diff
is diagnosed (which of the 4 commits, or the wave-2 commit itself, owns it) before any arm
reads against it.

## 4. A10 invariants + blindness

Invariants held fixed: everything in §2's CLI + explicit defaults, at the single venue iiib,
single h = 0.730. This gate certifies **reproduction of the four post-`d04d9dc9` estimator
commits at production scale and default values** — it is blind by construction to: (a) any
defect **shared** by `d04d9dc9` and HEAD (a reproduction gate cannot see a bug present in both
the referent and the candidate); (b) any h ≠ 0.730 (single-h read only — the other 40 h-values
are unverified by this gate); (c) any behavior that only engages under non-default flag values
(θ ≠ (0,1), `smear_global_selection=True`, `mass_filter_geometry="log"`, etc. — those are exactly
what C1/C3/C4 vary, deliberately not exercised here).

## 5. A15 — control capable of failing

Not vacuous by construction: F-B (`WAVE2_REGISTRATION_CHECK_20260829.md` §0) already demonstrated
that a same-N, same-CoR-P reproduction *can* fail on a pure code delta — the driver vs. the
banked `bc_900101_work` CSV differs by 5.718e-4 at bit-identical batch order, with the deciding
hypothesis being a code/config delta between that CSV's generating commit and HEAD. That is
exactly the failure mode this gate exists to catch at production scale; a false PASS under a
real code delta is excluded by the 1e-12 band being far below any such delta's observed scale
(5.7e-4 ≫ 1e-12).

## 6. F3 — predictions before run

Primary prediction: bit-reproduction (all 15 columns ≤ 1e-12) — the four commits' own gate-
ledger rows already pin θ=(0,1) and default `mass_filter_geometry`/`mass_filter_k` as byte-
identical to pre-flag behaviour (2026-08-28/29 verified rows). Secondary: the run reproduces
B3.1's dark-class h-score profile (`b3_pop_prediction.json:venues.iiib.bins`: +0.081, −0.332,
−0.562, −0.701, −0.855; bins 2–5 pooled −0.612, n=484) to ≤ 1e-6 — this doubles as B3's L1
baseline pin (docket §4.3) and needs no separate run.

## 7. F4 — cost + archive

**Cost:** 15–23 CPU-h (`COMPUTE_LEDGER.md:44`, wave-2 estimate; source: single h-value ×
production anchor 14.93–22.9 CPU-h/h-point, `WAVE2_REGISTRATION_CHECK_20260829.md:273`).
**Conditional fallback cost** (only if the gate FAILS): +120–173 CPU-h (C3 +59.7–91.6, C4
+59.7–81.1; same source line). **Archive:** out-root `run_20260829_wave2_c0_iiib`;
archive-scheduled: **yes**, via `results/_archive/archive_run_wave2.sh` (already created,
lists `run_20260829_wave2_c0_iiib` as `ITEMS[0]`) — runs AFTER retrieval, not by this node.
Deadline check: workspace expires 2026-09-23, 25 days out at launch — OK.

## 8. A22 — same-commit stamp

Wave-2 commit hash: **\<to be stamped at launch\>** — the dirty tree (gate-ledger row, six
appended `fanout1_20260829/` records, `hier_s0_driver.py`, the untracked
`test_survival_2d_homogeneity_falsifier.py`) must be committed first (GAP-list item 1); this
note's git-commit / dirty-state stamp is taken at run START per A22, not at record-write time.
Baseline commit named: `d04d9dc9bfe39e6c5a72e768a26f2dcc38355bf5`.

## 9. Verifier scope

Routes to the end-of-fan-out verifier: (i) the column-count correction (16 fields / 15 numeric,
not 17 — §3); (ii) the per-arm column-consumption table (§3); (iii) the PASS/FAIL band and its
consequence for C1/C3/C4 launch order; (iv) whether the fallback triggered and, if so, its
per-column diagnosis.

## 10. Authorization

Launched under rows #222/#223 — charter node C0. Orchestrator path decision of record
(2026-08-29): "wave-2 cluster set = C0 + C3 + C4 first, C1 after its preconditions." This
registration is information for the orchestrator (row #222 form (ii)); no approval requested.

## 11. Revision notes — appended 2026-08-29, post-refuter-panel

**Launched under rows #222/#223 — charter node C0 (revision pass).** A refuter panel reviewed
this registration and returned 6 must-fix items. All six are addressed below, append-only:
nothing in §§1–10 above is edited or removed; corrected values here SUPERSEDE the corresponding
claims in §3/§7/§9 by dated addendum, per the charter's standing rule 1.

**11.1 Column count corrected (must_fix 1, 4).** Verified directly: HEAD (`dd63fe0c`)
`fieldnames` (`bayesian_statistics.py:4725-4750`) has **19 entries**, not 16 — the original 16
(§3) plus `den_log_term`, `num_log_term_no_bh`, `num_log_term_with_bh`. `git log -S"den_log_term"
-- darksiren_emri/bayesian_inference/bayesian_statistics.py` returns exactly one commit,
`d40fe5c8` — one of the four commits this gate exists to certify — confirming the 3 extra
columns are in scope, not an unrelated drift. Independently confirmed against every produced
CSV header under `fanout1_20260829/hier_s0_*` (19 comma-separated fields each). §3's "16 fields
total: 1 join key + 15 numeric columns" was computed from `d04d9dc9`'s (pre-`d40fe5c8`)
fieldnames list, not HEAD's, under test — **SUPERSEDED**.
CORRECTED: **19 fields total: 1 join key (`event_idx`) + 18 numeric columns** (§3's original 15
+ `den_log_term`, `num_log_term_no_bh`, `num_log_term_with_bh`).
§9(i) is likewise **SUPERSEDED**: CORRECTED reading — "the column-count correction (19 fields /
18 numeric, not 17 and not 16)."

**11.2 Coverage for the 3 new columns (must_fix 2).** Confirmed correct as flagged: no
`d04d9dc9` value exists for these 3 columns (they are absent from that commit), so §3's
cross-baseline ≤1e-12 diff cannot and does not cover them. ADDED to the gate, run on the same
1588-event/h=0.730 output as §3 (no extra compute) — an internal identity check, since the three
columns are related by construction (`bayesian_statistics.py:5827-5837`:
`num_log_term_no_bh = log(combined_no_bh · den_used)`, `den_log_term = log(den_used)`):
- `num_log_term_no_bh − den_log_term == log(combined_no_bh)` to ≤1e-12 absolute, on rows where
  both terms are finite.
- `num_log_term_with_bh − den_log_term == log(combined_with_bh)` to ≤1e-12 absolute, same
  finiteness condition.
- Rows where a term is NaN (non-positive combined value or no denominator applied,
  `bayesian_statistics.py:5828-5836`) are exempted from the numeric identity and instead checked
  for NaN-pattern agreement between the two channels.
A FAIL here is scoped to C1 only (§11.3) — it does not invalidate §3's PASS for C3/C4, which
consume none of these 3 columns.

**11.3 C1 consumption row corrected (must_fix 3).** The code comment immediately above the
`fieldnames` list (`bayesian_statistics.py:4745-4747`) states these 3 columns exist for "the OAT
toggle matrix" — i.e., C1/PA-HIER-31 S0-B's own mechanism — and
`B4_1_IMP_DECOMPOSITION.md:44` independently lists them as consumed by the arm producing this
CSV. §3's per-arm table omitted them from the C1 row, contradicting the code comment three lines
above the fieldnames it describes — **SUPERSEDED**.
CORRECTED C1 row: `combined_no_bh` (primary channel, item 5), `combined_with_bh` (secondary,
invariant (e)), `L_cat_no_bh` (C-C class definition, item 2), **`den_log_term`,
`num_log_term_no_bh`, `num_log_term_with_bh` (OAT toggle matrix inputs, PA-HIER-23 — added
here)**.
Consequence: C1 may treat C0's PASS as covering these 3 columns only once §11.2's identity check
also passes. If §11.2 fails while §3's 18-numeric-column diff passes, C0 is **PARTIAL-PASS**:
valid zero-compute baseline for C3/C4 (§3 table unaffected — neither consumes the 3 new columns)
but not yet a clean truth node for C1/PA-HIER-31 item 4 until the identity-check failure is
diagnosed.

**11.4 Archive-scheduled reconciliation (must_fix 5).** Confirmed: `COMPUTE_LEDGER.md` line 42
(the original "Wave 2 (estimates only — not yet launched)" table, C0 row) still reads `pending`
in the Archive-scheduled column — that row is append-only and is not edited once written, per
the ledger's own header note and the charter's standing rule 1. The current status lives in the
later, separately dated append at `COMPUTE_LEDGER.md:97-102` ("Wave 2 archive-scheduled / GAP-6
closure", appended 2026-08-29), which sets C0/C1/C3/C4 archive-scheduled = **yes** via
`results/_archive/archive_run_wave2.sh`. §7's "archive-scheduled: yes" is therefore a correct
statement of current status, but §7 did not cite the row that actually carries it, so it read as
overclaiming against the row a reader would find first (line 42, `pending`) — **SUPERSEDED**.
CORRECTED §7 citation: archive-scheduled: yes — see `COMPUTE_LEDGER.md:99` (GAP-6 closure
append, 2026-08-29), which is the row of record; it does not edit (and does not need to edit)
the earlier `pending` cell at `COMPUTE_LEDGER.md:42`, per append-only. A same-dated
cross-reference note is added directly to `COMPUTE_LEDGER.md` (below its line 102) so a reader
of either row finds the other without a doc-wide scan.

**11.5 A8 scoping (must_fix 6).** ADDED: A8 (two-sided bands + referents) does not apply to
this gate. The ≤1e-12 band in §3 is a pure bit-reproduction check — a distance-from-identical
ceiling, not a band around a physical value — so "two-sided" has no referent here. This is a
deliberate scoping call, stated explicitly now rather than left silent.

**11.6 Length (informational, not must-fix).** The GAP item's "≤1-page" budget was already
exceeded pre-revision (154 lines); this §11 adds further length. No action taken — the refuter
panel marked this a minor process deviation, not a content defect.

**Verifier scope (§9) amendment.** Item (i) now reads "19 fields / 18 numeric, not 17 and not
16." New item (v): "whether the §11.2 identity check on the 3 OAT-toggle columns passed, and if
not, whether C1 is blocked pending diagnosis while C3/C4 remain clear per §11.3."

Stamped: launched under rows #222/#223 — charter node C0 (revision pass), 2026-08-29.

## 12. A22 launch stamp — appended 2026-08-29

wave-2 commit `ff230621`; job `6738998`; launched 2026-08-29 under rows #222/#223. Fills the §8
placeholder ("Wave-2 commit hash: \<to be stamped at launch\>"), which is left unedited
(append-only). Tree clean at both the local and cluster checkouts at launch time.

Stamped: launched under rows #222/#223 — charter node C0, 2026-08-29.

## 13. RESULT RECORD — appended 2026-08-29 (~21:10)

**Launched under rows #222/#223 — charter node C0.** Append-only; nothing above is edited.

**Run.** SLURM job `6738998` **COMPLETED** (Elapsed 00:06:28, ExitCode 0:0) at commit `ff230621`
(`GIT_COMMIT_AT_RUN.txt`). Provenance JSON `logs/provenance_6738998_none.json` reports
`dirty=296` — this is the cluster checkout's untracked-file count at run start, not a dirty
tracked tree; the tracked tree was clean at pull (§12). Dataset-pin STOP-gate: **OK** — CRB
md5 `9a1f2a14384a9281c97ca3be312ddaab`, catalogue md5 `c52c13b5cab61f6b3f04bbe202550969`, both
matching §2's pins. Evaluate stage: 1588 detections, 0 Fisher-flagged, 14 workers, host-lookup
yield 982/1588 with catalogue hosts (606 pure-completion), P6 host-recovery 66/76 in-catalogue
(86.84%) both channels; ~4.5 min evaluate wall (of the 6.5 min total elapsed).

**Gate verdict (§3 band).** **PASS — bit-identical.** Compared
`wave2_20260829/c0/diagnostics/event_likelihoods.csv` rows at h=0.73 (1588 rows) against
`headreadout_20260827/iiib/event_likelihoods.csv` rows at h=0.73 (1588 of that file's 65,108
data rows, filtered): max_abs **0.000** on all 14 non-trivial shared numeric columns (`w_G`,
`w_G_legacy`, `w_tilde_G`, `alpha_G_phi`, `r_Malm`, `D_tilde_phi`, `L_cat_no_bh`,
`L_cat_with_bh`, `B_num`, `B_num_wbh`, `g_frac`, `L_comp`, `combined_no_bh`,
`combined_with_bh`) — the 15th numeric column per §3/§11.1 (`h`) is fixed at 0.73 in both files
by construction and not a discriminating check. Far inside the ≤1e-12 band (it is exactly 0).
**No fallback triggered.**

**Column-list finding ("cols equal: False" — checked per the orchestrator's request).**
`head -1` diff of the two CSVs: `headreadout_20260827/iiib/event_likelihoods.csv` has 16 fields
(`event_idx, h` + the 14 columns above); `wave2_20260829/c0/diagnostics/event_likelihoods.csv`
has 19 fields — the same 16, in the same order, plus three trailing columns:
`den_log_term, num_log_term_no_bh, num_log_term_with_bh`. This is exactly the §11.1 correction
already on record (HEAD's 19-field/18-numeric fieldnames vs. the pre-`d40fe5c8` 16-field form)
— not a regression or an unexpected drift. The naive pandas `cols equal: False` print is fully
explained by these 3 known/expected additions; it is a superset relationship, not a mismatch.
**Coverage: the registration's gate column set (§3's 15 numeric columns, all present in both
files, same order, same names) is fully covered** — the shared prefix is byte-identical between
the two headers, and the extra 3 columns in c0 are outside §3's gate scope (they are gated
separately, and only for C1, by §11.2's not-yet-run internal identity check — this RESULT
RECORD does not run that check; it is out of this record's scope and does not affect the §3
PASS above, which covers zero of the 3 new columns by design, per §11.3).

**Posterior comparison.** Rather than a parsed max-abs-diff, md5 checksum was used (a strictly
stronger identity claim): `wave2_20260829/c0/posteriors/h_0_73.json` md5
`563ef45b0598dcfc8f5c9ba19efbf9fd` — **byte-identical** to
`headreadout_20260827/iiib/posteriors/h_0_73.json` (same md5).
`wave2_20260829/c0/posteriors_with_bh_mass/h_0_73.json` md5 `2b4fb3e0d055e08fe7a905b8d3c4d817`
— **byte-identical** to `headreadout_20260827/iiib/posteriors_with_bh_mass/h_0_73.json` (same
md5). Both files are dicts keyed by event index string (`'0'`…`'1589'`, with the expected gaps
at excluded event indices, e.g. `'1203'`, `'1356'` absent) plus `'h'`; bit-identity trivially
satisfies §3's "bit-identical or ≤1e-12 relative" condition on every key.

**F4 costing correction [A11].** The HEAD-readout job `6725283`'s own per-task Elapsed ranged
00:00:18 (task 21 — the exact h=0.730 point this gate reused as baseline) … 00:42:26 (task 13);
the 56–76 min/h-value anchor cited in §2/§7 (`cluster/LAUNCHING_JOBS.md:47`) is measured on the
**3355-event** set and is **not** the right anchor for this **1588-event** iiib venue. C0
actual: 00:06:28 elapsed × 16 cpus/task ≈ **1.7 CPU-h**, against the 15–23 CPU-h estimated in
§7 (built from the wrong anchor) — roughly a 9–13x overestimate. This correction is recorded
for reuse: the C3/C4/C1 cost estimates in `COMPUTE_LEDGER.md` cite the same 56–76 min anchor and
should be re-costed against the 1588-event/task-21-range anchor before being treated as tight
bounds, not just orders of magnitude.

**Verdict of record.** PASS bit-identical ⇒ per §3, the banked HEAD readout
`headreadout_20260827/iiib/` (commit `d04d9dc9`) is the zero-compute baseline for C3 and C4
(docket L5) and the truth node (θ=(0,1)) for C1/PA-HIER-31 item 4. §3's conditional fallback
(+120–173 CPU-h) is **not** triggered. Per §11.3, this PASS does not by itself certify C1 as a
*clean* truth node for the 3 OAT-toggle columns (`den_log_term`, `num_log_term_no_bh`,
`num_log_term_with_bh`) — that remains conditional on the separate §11.2 identity check, not run
by this record.

Stamped: launched under rows #222/#223 — charter node C0, 2026-08-29.
