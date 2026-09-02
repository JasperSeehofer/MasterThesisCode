# Lens 4/4 — Performance + Enhancement Ideas

**Scope:** read-only overnight project-health scan, `/home/jasper/Repositories/darksiren-emri`.
**Method:** read the named hot-path code (`scalar_product_of_functions`, `evaluate()`, the Pool
setup, `child_process_init`), diffed three `graph1_*.sbatch` files, checked the galaxy-catalogue
BallTree call sites, and confirmed the `CAMPAIGN_READOUT_REPORT.md` template already exists.
No edits, no commits, no heavy compute.

---

## 1. Performance

### 1.1 `scalar_product_of_functions` (the named PSD-loop bottleneck) — ALREADY OPTIMIZED

`darksiren_emri/parameter_estimation/parameter_estimation.py:335-410`. This is **not** the
naive-loop bottleneck CLAUDE.md's docstring implies at first read — the current implementation
already:
- caches `(fs, psd_stack, lower_idx, upper_idx)` per waveform length `n` in `self._psd_cache`
  (`_get_cached_psd`, line 146), so all 105 stencil calls in one Fisher matrix (14 params ×
  5-point stencil) share one PSD lookup;
- batches the FFT across TDI channels (`rfft(..., axis=-1)` on the whole `(n_channels, n_samples)`
  array, not a per-channel loop);
- vectorizes the frequency-band integrand and does one `trapezoid` call per channel pair.

**No further win visible without changing the physics** (e.g. cross-h PSD sharing would require
threading `h` into cache keys, which the PSD does not depend on — dt/waveform length only — so
there's nothing to add there either). Rate this sub-item **closed**, not a live lead. Worth a
one-line note back to whoever wrote the CLAUDE.md docstring pointer so a future session doesn't
re-open it.

### 1.2 `evaluate()` per-h loop / worker pool — mostly ALREADY OPTIMIZED, one real gap

`bayesian_inference/bayesian_statistics.py:3788` (`evaluate`) → `Pool` setup at line ~5266 →
per-h loop at line ~5305. Confirmed:
- forkserver context with `set_forkserver_preload([...])` — avoids 126× cold Python+import per
  worker on the shared cluster filesystem (comment cites this explicitly);
- the Gaussian precompute (means/cov_inv/log_norm for 3D and 4D branches), `D_h_table`, and all
  `beta_G*`/`Sigma` tables are computed **once**, outside the `for _h_run in _h_list` loop, and
  passed into `child_process_init` via `initargs` — h-invariant setup is not redone per h-node;
- the **same Pool is reused across all h in the grid** (the `with _ctx.Pool(...) as pool:` block
  wraps the entire `for _h_run in _h_list` loop) — so a 55-node G-EXT grid run pays worker
  spawn/import cost once, not 55×.

**Real gap found:** `num_workers = max(1, available_cpus - 2)` (line ~5241) is computed once from
`os.sched_getaffinity(0)`/`os.cpu_count()` with no override surfaced for cgroup-limited SLURM
allocations that under-report affinity vs. `--cpus-per-task`. Low-effort, low-risk: expose
`num_workers` resolution diagnostics in the run log (`_LOGGER.info` already fires — just also log
`os.cpu_count()` vs `os.sched_getaffinity(0)` explicitly, since they can silently disagree in a
container). **SAFE, no physics gate** (logging only).

### 1.3 BallTree usage — fine, but one query-shape note

`galaxy_catalogue/handler.py:713` (`query_radius`, 3D cone) and `:1036` (`query`, 5D host-recovery,
k=1) are both single-query-point calls inside their respective per-event functions, not called in
a Python-level per-galaxy loop — the vectorization is already at the "one query per detection"
level, which is the correct grain (BallTree itself is the O(log N) structure; batching multiple
detections into one `query()` call with a 2D query-point array is possible in principle but the
per-event call sites aren't visibly hot enough in the profiling story here to justify it — no
evidence of this in the codebase's own timing logs). **Not pursued as a lead** — flag only if a
future flamegraph shows BallTree query dominating (it should not, at O(log N) per query against
GLADE+-sized N).

### 1.4 Rough CPU-h estimate for real wins

Given 1.1/1.2 are already exploited, the highest-leverage *remaining* performance lever is not in
the hot math path but in **campaign-level redundancy**: the `graph1_*.sbatch` near-duplication
(§2.2 below) means CPU-h is currently spent correctly per-job but the *human* iteration loop
(author writes → reviews → launches near-duplicate sbatch) is the actual bottleneck for
"efficient realistic-venue campaigns" (the standing memory goal). A campaign-driver abstraction
(§2.2) plausibly cuts wall-clock-to-launch by a large factor even though it doesn't touch a single
CPU-h of compute — this is where the "efficient realistic-venue campaigns are a first-class
deliverable" memory is actually served, not by squeezing `scalar_product_of_functions` further.

---

## 2. Developer Experience

### 2.1 `evaluate()` flag sprawl → config object — HIGH VALUE, concrete evidence

`BayesianStatistics.evaluate()` (`bayesian_inference/bayesian_statistics.py:3788` onward) has
**~25 keyword parameters**, most `str` "mode" flags with "auto" defaults and multi-paragraph
inline comments explaining physics-history provenance (selection fusion, catalogue mass overlap,
Eddington-m, sigma4d_mass_kernel, catalogue_numerator_survival, host_z_kernel, host_mass_kernel,
completion_b_scale, completion_event_measure, ...). Each new research-graph branch has historically
added one more flag with "auto resolves to X under condition Y, else Z" semantics baked into
prose comments that live only at the call site.

**Idea:** introduce an `EvaluationConfig` dataclass (or frozen `TypedDict`) that groups these by
research axis (selection-fusion axis, catalogue-mass axis, host-kernel axis, ...), each carrying
its own `"auto"`/explicit resolution as a method, with the resolution logic and its literature/
ledger-row justification living in one place instead of scattered inline. `evaluate()` keeps
`**kwargs`-compatible back-compat by accepting either the dataclass or the current flags for one
deprecation cycle. This does **not** change any resolved value for any existing call site — it is
a pure refactor of *how the flags are carried*, so it is **SAFE without a physics gate** as long as
the resolution methods are transcribed verbatim from current code (a mechanical, sonnet-tier task
with a byte-identical regression test: run one h-grid point before/after, diff the posterior JSON).
**Value:** every future research-graph branch that adds a flag currently pays the "where do I even
put this" tax; this removes it. **Effort:** medium (this function is 9300+ lines in its module;
touching it needs the same care as any physics-adjacent refactor even though it's mechanical).

### 2.2 Campaign-driver abstraction over `graph1_*.sbatch` — HIGH VALUE, concrete evidence

Diffed `cluster/graph1_headrebaseline_iiib.sbatch` against `cluster/graph1_t5_armS_iiib.sbatch`
and `cluster/graph1_headrebaseline_joint_r1.sbatch`: the header provenance comments differ (as
they should — different branch/venue), but the **mechanical skeleton** (H_GRID_41 array-task-id
seeding convention `777000 + task`, dataset-pin STOP-gates, `--time=` sizing rationale copied by
hand each time, `RUN_DIR` naming convention, provenance-file writing) is duplicated across at
least 8 `graph1_*.sbatch` files plus `wave2_*`/`wave3_*` predecessors — 16+ near-identical sbatch
files in `cluster/` total.

**Idea:** a small `campaign_driver.py` (or extend `cluster/campaign_orchestrator.sh`) that takes
a YAML/TOML spec — venue, grid, seed base, dataset pins, CLI flag overrides, time estimate — and
renders the sbatch from `cluster/JOB_TEMPLATE.sbatch` (which already exists — check whether it's
actually used as the template source for these, or whether each file was copy-pasted from the
previous branch's file and hand-edited, which the diff output suggests). This is the single
highest-leverage DX item: it directly reduces the per-branch cost of "efficient realistic-venue
campaigns," is testable by re-rendering an existing sbatch and diffing byte-for-byte against the
committed one, and touches zero physics code. **SAFE, no physics gate.** Effort: medium (needs
one careful pass to extract the *actual* invariants vs. the genuinely-branch-specific bits —
several sbatch headers show ad hoc time-sizing "anchored to" a sibling file, which the tool should
compute/record rather than hand-copy).

### 2.3 Three-valued remote-existence helper — MEDIUM VALUE

Referenced by the row #288 class of bug (per the task brief — not independently re-derived here
given scope). `.claude/skills/cluster/SKILL.md` and `cluster/preflight.sh` were checked for an
existing `exists`/`UNKNOWN` helper and none was found by name. **Idea:** a small
`remote_path_status(host, path) -> Literal["present","absent","unknown"]` wrapper around the ssh
existence check used throughout the cluster skill/preflight, so "ssh failed" / "timed out" is
never silently coerced to "absent" (the shape of bug that produces false "file doesn't exist yet"
verdicts). **SAFE, no physics gate** — pure infra utility. Effort: low. Value is bounded by how
often this specific failure mode recurs; worth building once and using everywhere in `/cluster`
rather than re-deriving per script.

### 2.4 Same-machine byte-id gate utility — MEDIUM VALUE

Per the task brief (rows #318-#319), not independently re-verified in this pass (out of scope to
re-read those rows given time budget). If the pattern is "verify two paths are byte-identical
before treating them as the same input," a `assert_byte_identical(path_a, path_b)` helper (sha256
compare, STOP-gated) generalizes the existing dataset-pinning convention already codified in
CLAUDE.md ("Dataset pinning... checksum pin at each consumer, STOP-gated on mismatch"). This is
close enough to an existing convention that it's likely a small extraction, not new design.
**SAFE, no physics gate.** Effort: low.

---

## 3. Science-Facing Features (strengthen the paper)

### 3.1 Auto-generated per-campaign readout report — HIGH VALUE, template already exists

`docs/templates/CAMPAIGN_READOUT_REPORT.md` already codifies the comprehension-first structure
(binding rules: never adjudicate, every number traceable to a named source file, charts from raw
per-seed data not summary stats, mandatory vocabulary glosses, adjudicator flags survive into the
report). Right now this is filled in by hand per campaign (e.g.
`results/prod2d_closure_20260818/CAMPAIGN_READOUT_REPORT_CSG_20260821.md`). **Idea:** a script
that reads a campaign's raw per-seed JSON/CSV outputs + the adjudication JSON + the relevant
ledger rows and emits a draft report following the template's structure and binding rules
mechanically (rule 2's "traceable to raw records, never a report-computed summary" constraint
maps directly onto a script that only ever reads, never derives new numbers beyond what the
template's own tables call for). A human still writes the prose interpretation, but the
scorecard/tables/chart generation becomes deterministic and fast. **SAFE to build without a
physics gate** (it renders existing numbers, computes nothing new) — but the generated report
must still go through the same human authorship discipline before being treated as a record
artifact (rule 1: never adjudicate). Effort: medium-high (needs to handle the several distinct
readout JSON schemas already in use across `results/*/`).

### 3.2 Interactive H0-posterior explorer for the 55-node G-EXT grid — MEDIUM VALUE

An Artifact-based (or matplotlib-widget) explorer over the per-h posterior JSONs
(`simulations/posteriors/h_*.json`, `simulations/posteriors_with_bh_mass/h_*.json`) already
written by `evaluate()` — slider over h, toggle with/without-BH-mass channel, per-event vs.
combined view. This is a pure-read visualization over already-committed output; **SAFE, no
physics gate.** Value is real but secondary to 3.1 — it's a nice-to-have for exploration, not a
comprehension bottleneck the way the readout report is (per the standing "comprehension-first"
author preference).

### 3.3 P-P/coverage dashboard from B8.2 outputs — MEDIUM VALUE

`darksiren_emri/validation/pp_coverage.py` already computes P-P/coverage calibration
(`_run_realization`, `_hpd_contains`, `PPCoverageConfig`). A dashboard aggregating per-seed P-P
plots + coverage-fraction summary across a campaign (e.g. the B8.2 S3 pilot cells referenced in
recent runbook rows) would directly serve calibration-gate decisions. **SAFE, no physics gate**
(pure visualization of existing outputs). Lower priority than 3.1 because it's narrower in scope
(one validation harness vs. every campaign).

### 3.4 Claims/SETTLED-questions board renderer — LOWER VALUE, higher effort

A renderer over `docs/RETROSPECTIVE_LEDGER.md` / `docs/gates/PHYSICS-GATE-LEDGER.md` / the various
row-numbered decision logs, producing a queryable "what's settled, what's open, what's provisional"
board. Valuable for onboarding a new session or reviewer, but the ledger format is heterogeneous
across files (checked: `RETROSPECTIVE_LEDGER.md`, `PHYSICS-GATE-LEDGER.md`, and
`book/site/data/museum_ledger.json` are three different schemas for what's conceptually the same
kind of record) — building a renderer means first reconciling those schemas, which is real design
work, not mechanical extraction. Rate this **lower priority** than 3.1-3.3 until/unless the ledger
schemas are unified (a separate, smaller yak-shave that would pay for itself here).

---

## 4. Robustness

### 4.1 OOM-aware local runner — MEDIUM VALUE

Per the task brief (row #321 crash), not independently re-verified here. **Idea:** wrap local
(non-cluster) invocations of the simulation/evaluate pipeline with a memory-ceiling check
(`resource.setrlimit` or a `psutil`-based watchdog) that fails fast with a clear message instead
of an OS-level OOM-kill, and/or auto-suggests a `num_workers` reduction when available RAM /
worker count looks tight relative to the per-worker Gaussian-precompute table sizes computed in
`evaluate()`. **SAFE, no physics gate.** Effort: low-medium.

### 4.2 Resumable invocation wrappers — LOWER VALUE (partially exists)

The per-h loop already writes each h's posterior JSON immediately and independently ("per-h
failure granularity is preserved in grid mode" — comment at bayesian_statistics.py:~5304), so
grid-mode `evaluate()` runs are already substantially resumable at the h-node granularity by
construction (check for existing `h_{label}.json` before recomputing that h). Whether a wrapper
that does this check-and-skip currently exists wasn't confirmed in this pass — if not, it's a
small, safe addition. Rate lower priority since the underlying per-h write-immediately design
already does most of the work.

### 4.3 Structured sidecars everywhere — LOWER VALUE

The realization-provenance sidecar pattern (`realization_provenance.json` written next to
`simulations/posteriors/`) and `run_metadata.json` (git_commit, timestamp, CLI args) are already
used in the places that matter most (per CLAUDE.md's "Reproducible simulation runs" section).
Extending this uniformly to every intermediate artifact is good hygiene but diminishing-returns
compared to 2.3/2.4/4.1 above; only worth doing opportunistically alongside other touches to a
given script rather than as its own project.

---

## Ranked summary (value/effort, SAFE = no physics gate needed)

| # | Idea | Value | Effort | Safe? |
|---|---|---|---|---|
| 1 | 2.2 Campaign-driver abstraction over `graph1_*.sbatch` | High | Med | Yes |
| 2 | 3.1 Auto-generated campaign readout report (template exists) | High | Med-High | Yes |
| 3 | 2.1 `evaluate()` config-object refactor (flag sprawl) | High | Med | Yes (mechanical, regression-tested) |
| 4 | 2.3 Three-valued remote-existence helper | Med | Low | Yes |
| 5 | 4.1 OOM-aware local runner | Med | Low-Med | Yes |
| 6 | 2.4 Same-machine byte-id gate utility | Med | Low | Yes |
| 7 | 3.3 P-P/coverage dashboard from B8.2 outputs | Med | Med | Yes |
| 8 | 3.2 Interactive H0-posterior explorer | Med | Med | Yes |
| 9 | 4.2 Resumable invocation wrappers (partially exists) | Low-Med | Low | Yes |
| 10 | 3.4 Claims/SETTLED board renderer | Low (until ledger unified) | High | Yes |
| — | 1.2 num_workers affinity-vs-cgroup logging | Low | Low | Yes |
| — | 1.1 `scalar_product_of_functions` further optimization | None (already optimal) | — | N/A |

**Note on 1.1:** this scan found the named bottleneck already resolved by prior work (PSD caching,
batched FFT, per-length cache) — the CLAUDE.md docstring pointer to it as "the computational
bottleneck" is a true statement about *where time goes*, not a live lead for a *further* speedup;
worth distinguishing those two claims in future project-health framing.

All ten ranked ideas plus the two additional notes are infra/tooling/visualization work with no
formula, constant, or model change — **none require the `/physics-change` gate.** 3.1 and 3.4
touch the ledger/record-keeping discipline (CLAUDE.md's "reviewable artifact" / attribution-precise
recording rules) and should follow those conventions even though they're not physics changes.
