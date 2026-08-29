# Commit Plan — fan-out 1 wave 1, Records node (2026-08-29)

Launched under rows #222/#223 — charter node: Records (mechanical, no git operations performed by
this agent; the orchestrator commits). This file proposes a commit split for the orchestrator's review.

## 1. `git status --short`, filtered

Filter applied (per task instructions): excludes `results/**/ca_rhs_work`, `simulations/`,
`docs/CLAUDE_SCIENCE_*.md`, `scripts/bridge_closure/outputs/f4_specz_decomposition.json`.

160 lines survive the filter. Full raw list saved this session at `/tmp/gitstatus_filtered.txt`
(scratch, not part of the repo). Summary by category:

- **Tracked, modified (11 files):**
  - `darksiren_emri/arguments.py`
  - `darksiren_emri/bayesian_inference/bayesian_statistics.py`
  - `darksiren_emri/galaxy_catalogue/handler.py`
  - `darksiren_emri/main.py`
  - `darksiren_emri/validation/correspondence_1d.py`
  - `darksiren_emri_test/bayesian_inference/test_theta_hook.py`
  - `darksiren_emri_test/validation/test_correspondence_1d.py`
  - `docs/RESEARCH_CYCLE.md`
  - `docs/gates/PHYSICS-GATE-LEDGER.md`
  - `results/campaign51_20260728/realistic_20260729/PHYSICS_CHANGE_THETA_HOOK_20260828.md`
  - `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`

  **Caveat (disclosed, not fixed by this node):** `bayesian_statistics.py`, `handler.py` are
  physics-trigger files (per CLAUDE.md) and `correspondence_1d.py` is a validation module —
  none of these three are named in this wave's B5/B6 record scope below. This Records node did
  NOT diff their content; the orchestrator must confirm each modified physics-trigger file traces
  to an already-gated change (B5 mass-window flag touches `bayesian_statistics.py`/`arguments.py`;
  B6 θ-hook alignment likely also touches `bayesian_statistics.py`) before committing under (a)/(b)
  below, and must NOT fold an ungated physics edit into (c).

- **Untracked, new (this wave, under `fanout1_20260829/`):** the whole
  `results/campaign51_20260728/realistic_20260729/fanout1_20260829/` directory (one `??` line;
  60+ files/dirs inside, see `README.md` for the full index).

- **Untracked, new (this wave, elsewhere):** `darksiren_emri_test/test_mass_filter_geometry.py`
  (B5 unit tests — should travel with the B5 physics commit, not with (c)).

- **Untracked, NOT from this wave (pre-existing work, other branches/sessions — excluded from all
  three proposed commits below, left for their own owning workflow):**
  `results/campaign51_20260728/realistic_20260729/cb_null_pinning_output.json`,
  `head_readout_extraction_20260827.md`, `ledger_row_collision_map_20260827.md`, `p3_2d_work/`,
  `p3_2d_work_m2z/`, `p3_b0_work/`, `p3_work/`, `rule1_sweep_complete_20260827.md`,
  `wbhzero_work/`; `results/prod2d_closure_20260818/**`; `results/run_20260620_seed500_phase50/`,
  `run_20260628_seed600_figures/`, `run_20260804_frozeng/`, `run_20260804_postfix/`,
  `run_20260805_d1/`, `run_20260805_n2sel1d/`, `run_20260817_fusion_counterfactual/`.

## 2. `du` of new directories under `fanout1_20260829/` (>50 MB flag)

| dir | size | flag |
|---|---|---|
| `cmem_a1_work/` | 16K | — |
| `hier_s0_registered_run/` | **46M** | below the 50 MB flag threshold but close — mostly seed900101 run dirs + logs; confirm Option A archive scheduling before this grows further with S0-A completion (P0, wave 2) |
| `hier_s0_work/` | 3.0M | — |
| `__pycache__/` | 44K | build artifact — should NOT be committed (add to .gitignore check or exclude explicitly) |
| `verify_b51/` | 4.0K (empty) | — |
| **directory total** | **52M** | under 50 MB per-item threshold; no single new directory exceeds it |

No new directory under `fanout1_20260829/` exceeds 50 MB individually. `hier_s0_registered_run/`
is the one to watch — flagged for the orchestrator's attention, not blocking.

## 3. Proposed commit file lists

### (a) `[PHYSICS] mass window: instrument flag mass_filter_geometry/mass_filter_k (default byte-identical) — row #223, charter B5.1`
- `darksiren_emri/arguments.py` (flag definitions — **orchestrator must confirm** this file's
  diff is B5-scoped only; it was not diffed by this node)
- `darksiren_emri/bayesian_inference/bayesian_statistics.py` (**orchestrator must confirm**
  B5-scoped only, not conflated with B6's θ-hook edit — see caveat above; if both B5 and B6 touch
  this file, split by hunk or land B6 first per the docket's commit-ordering requirement, §2 B6)
- `darksiren_emri_test/test_mass_filter_geometry.py` (new unit tests)
- gate-ledger rows for B5 (presented / presented-revised / implemented / verified) in
  `docs/gates/PHYSICS-GATE-LEDGER.md` (**orchestrator must confirm** this file's diff separates
  cleanly into B5 rows vs B6 rows before splitting into (a) vs (b))

### (b) `[PHYSICS] θ-hook: align s placement to [HIER] §1.2 — row #221 item 4 / row #223, charter B6.1`
- `darksiren_emri/bayesian_inference/bayesian_statistics.py` (B6-scoped hunks only — see split
  note above)
- `darksiren_emri_test/bayesian_inference/test_theta_hook.py`
- `darksiren_emri_test/validation/test_correspondence_1d.py` (**orchestrator must confirm** this
  belongs to B6 and not an unrelated in-flight change — not diffed by this node)
- `darksiren_emri/validation/correspondence_1d.py` (**same caveat**)
- `darksiren_emri/galaxy_catalogue/handler.py` (**same caveat** — this file is also gated
  separately per CLAUDE.md; confirm the edit is B6-scoped, not an unrelated galaxy-catalogue change)
- gate-ledger rows for B6 (presented / implemented / verified) in
  `docs/gates/PHYSICS-GATE-LEDGER.md`
- `results/campaign51_20260728/realistic_20260729/PHYSICS_CHANGE_THETA_HOOK_20260828.md`
  (pre-existing gate presentation doc, modified — confirm append-only edit, not a rewrite)

### (c) `docs: fan-out 1 wave 1 — rows #224–#233, F1–F5 adopted, node records, synthesis docket 1, HIER Stage-0 driver`
- `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`
  (rows #225–#233 appended this node; row #224 pre-existing from wave-1 launch)
- `docs/RESEARCH_CYCLE.md` (**orchestrator must confirm** scope of this modification — not diffed
  by this node; if it is an F1–F5 amendment adoption it belongs here, otherwise route separately)
- the entire `results/campaign51_20260728/realistic_20260729/fanout1_20260829/` directory
  EXCEPT `__pycache__/` (exclude — build artifact) and except any files already claimed by (a)/(b)
  above (none — all fanout1 contents are docs/records/scripts/data, not source under
  `darksiren_emri/`)

## 4. Open items for the orchestrator before committing

1. Diff `arguments.py`, `bayesian_statistics.py`, `handler.py`, `correspondence_1d.py`,
   `test_theta_hook.py`, `test_correspondence_1d.py` to confirm the (a)/(b) split above is correct
   and that no ungated physics edit is hiding in either — this Records node did not run those
   diffs (out of its mechanical scope; rule 6 forbids this node from editing physics-trigger files,
   and reading their diffs was not requested).
2. Confirm B6 lands before B1's S0-B (charter ordering, docket §2 B6) — i.e. commit (b) before
   any future S0-B work, and land (a)+(b) in that relative order if both touch
   `bayesian_statistics.py`.
3. Exclude `__pycache__/` from any commit; add a `.gitignore` entry if one does not already cover
   `fanout1_20260829/__pycache__/`.
4. `verify_b51/` is committed empty (0 files) per the docket's disclosure (§0: refuter reports for
   B5.1-implementation, B6.1, B8.1 were not in the chair's package) — confirm this is intentional
   before committing an empty directory (git will not track it without a placeholder file; note
   this to the orchestrator rather than adding one unilaterally).
5. `docs/RESEARCH_CYCLE.md` was not authored or touched by this Records node's own edits — its
   presence in `git status` predates this task and its scope is unverified here.

## 5. Quality gate (report only — see task return for full counts)

- `ruff check darksiren_emri/`: all checks passed
- `ruff format --check darksiren_emri/`: 70 files already formatted
- `mypy darksiren_emri/`: success, no issues found in 70 source files
- `pytest -m "not gpu and not slow" -q`: **1871 passed, 15 skipped, 27 deselected** in 115.81 s;
  coverage 73.17% (gate 25.0%); no failures.

No lint/typing fixes were needed in any new file this wave.
