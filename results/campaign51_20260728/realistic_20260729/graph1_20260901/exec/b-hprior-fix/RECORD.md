# Node b-hprior-fix (Branch I) — recon + physics-change gate draft

**Authorization (verbatim, ledger row #290):** "rows 3–11 [DO] APPROVED — branch heads A–I
trigger their first items (... h-prior fix + G-EXT rerun)" — item 11 of the row-#290 decisions
table names this node explicitly: "b-hprior-fix (Branch I) | DO | Approved | the config fix +
rerun of the 14 failed G-EXT tasks (row #286), byte-identity below the old bound | any claim that
the extended grid is load-bearing for a given arm — decided at that arm's registration".

Effort: medium. No commit made. No cluster submission made. This record is the deliverable.

---

## 1. Recon

### 1.1 What failed, and where

Source of the failure disclosure: `results/campaign51_20260728/realistic_20260729/tree2_20260830/
PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md`, "ARM (c) RESULT RECORD + FLIP EXECUTED" section
(appended under rows #282–#286):

> "Arm (c) run: job 6747032 (55-node G-EXT grid), commit 38cc0f58; tasks 0–40 (H_GRID_41) all
> COMPLETED; tasks 41–54 (h ≥ 0.87) FAILED on the h-prior upper-bound guard
> (`cosmological_model.h.upper_limit`; `bayesian_statistics.py` evaluate() bounds check) —
> disclosed; verdict-irrelevant (posterior tail at h ≥ 0.85 is 5e-13)."

Confirmed by direct grep of the two files it names:

- **`darksiren_emri/cosmological_model.py:388`** — `LamCDMScenario.__init__`:
  ```python
  self.h = CosmologicalParameter(
      symbol="h",
      upper_limit=0.86,
      lower_limit=0.6,
      unit="s*Mpc/km",
      randomize_by_distribution=uniform,
      fiducial_value=0.73,
  )
  ```
  This is the bound of record. `CosmologicalParameter` (line 53) subclasses `Parameter`
  (`darksiren_emri/datamodels/parameter_space.py:30`), a plain dataclass — `upper_limit` carries
  no independent physical meaning beyond "widest h this run is willing to evaluate a posterior
  at"; it is a **prior-support / grid-admissibility bound**, not a measured or derived physical
  constant (contrast `H_MIN` in `constants.py`, or `Omega_m.upper_limit=0.5`, which bound
  physically-motivated regions).

- **`darksiren_emri/bayesian_inference/bayesian_statistics.py:4655-4658`** — inside
  `BayesianStatistics.evaluate()`, the hard guard that actually raises:
  ```python
  for _h_check in _h_list:
      if (_h_check < self.cosmological_model.h.lower_limit) or (
          _h_check > self.cosmological_model.h.upper_limit
      ):
          raise ValueError("Hubble constant out of bounds.")
  ```
  This is checked once per `evaluate()` call, before any posterior computation, against the
  live `LamCDMScenario` instance's `h.upper_limit`.

- **`bayesian_statistics.py:3707`** — `self.cosmological_model = LamCDMScenario()`, inside
  `BayesianStatistics.__init__`. **This construction is unconditional and takes no
  cosmological-model / bound-override argument** — grepped the full constructor signature
  (`bayesian_statistics.py:3791` area) and `main.py`/`arguments.py` call sites; there is no CLI
  flag, env var, or kwarg anywhere in the evaluate path that reaches `h.upper_limit`. The value
  is fixed entirely by the class-body literal at `cosmological_model.py:388`.

### 1.2 The 14 failed tasks

From `AMENDMENT G-EXT` (same file, appended under row #284) and the per-task provenance JSONs at
`results/campaign51_20260728/realistic_20260729/tree2_20260830/a18_prod_arm/logs/
provenance_6747032_54.json` etc.:

- Grid of record: `H_GRID_41` (0.600–0.860, 41 nodes) ∪ 14 added nodes at 0.010 spacing
  {0.870, 0.880, …, 1.000} = 55 nodes total, job **6747032**, SLURM array `0-54`.
- The 14 G-EXT wing nodes are **array tasks 41–54**, one h-value each:

  | task | h | seed | job id (per-task) |
  |---|---|---|---|
  | 41 | 0.870 | 777041 | 6747082 |
  | 42 | 0.880 | 777042 | 6747083 |
  | 43 | 0.890 | 777043 | 6747084 |
  | 44 | 0.900 | 777044 | 6747085 |
  | 45 | 0.910 | 777045 | 6747086 |
  | 46 | 0.920 | 777046 | 6747087 |
  | 47 | 0.930 | 777047 | 6747088 |
  | 48 | 0.940 | 777048 | 6747089 |
  | 49 | 0.950 | 777049 | 6747090 |
  | 50 | 0.960 | 777050 | 6747091 |
  | 51 | 0.970 | 777051 | 6747092 |
  | 52 | 0.980 | 777052 | 6747093 |
  | 53 | 0.990 | 777053 | 6747094 |
  | 54 | 1.000 | 777054 | 6747032 (array parent) |

  (Task→h and task→seed mapping confirmed against `cluster/a18_ma1d_headreadout_iiib.sbatch`
  header comment: "task 21 = h 0.730, seed 777000 + 21 = 777021 ... tasks 41-54 are new nodes
  with seeds 777041-777054", and cross-checked against the individual `provenance_*.json` `note`
  fields, which give h for every task except that the wing tasks' provenance files show the
  *submission-side* metadata only — the FAILURE itself (ValueError at evaluate() bounds check)
  means no `posteriors/*.json` was written for these 14, consistent with "tasks 41–54 ...
  FAILED" in the gate doc.)

- Disclosed impact: "posterior tail at h ≥ 0.85 is 5e-13" — the A18 flip verdict (Z-CONFIRMED,
  row #286) does not depend on these 14 nodes; this node exists purely so the extended grid is
  usable for **future** measurements that need it (per the graph proposal §1.9 / row #278 item 8
  in `STATE_AND_CANDIDATES_20260901.md`: "whether the extended h-grid ... is trustworthy at its
  outer nodes for any future measurement that needs them").

### 1.3 Config-only vs. physics-trigger-file verdict

**Genuinely requires touching `cosmological_model.py`.** There is no config-only path:

1. `BayesianStatistics.__init__` builds `LamCDMScenario()` unconditionally at
   `bayesian_statistics.py:3707` — no constructor argument, CLI flag (checked `arguments.py`),
   or environment variable threads an override to `h.upper_limit`.
2. `LamCDMScenario.__init__` itself hardcodes `upper_limit=0.86` as a dataclass-field literal —
   not read from `constants.py`, not parametrized.
3. The only way to raise the ceiling without editing `cosmological_model.py` would be to reach
   into a live `BayesianStatistics` instance after construction and mutate
   `bs.cosmological_model.h.upper_limit` from calling code (e.g. `main.py`) — but `main.py` is
   itself not a physics-trigger file while doing so would still change the effective physical
   bound the production `evaluate()` path enforces, defeating the purpose of gating on the
   *files*: the gate is about the computed value changing, not about which file's diff shows it
   (CLAUDE.md: "Any edit to these files that modifies a computed value ... requires
   `/physics-change`" is stated as a **necessary**, not exhaustive, condition — mutating the bound
   from anywhere is still a physics-relevant value change and should go through the same gate in
   spirit). Recommendation below is therefore to make the one-line edit in
   `cosmological_model.py` itself, openly, rather than launder it through a side-channel mutation
   in a non-trigger file.

**No edit has been made.** Per task instructions, this record only drafts the gate presentation.

---

## 2. Physics-change presentation (DRAFT — chair runs `/physics-change` before any edit lands)

### 2.1 Old value / location
`darksiren_emri/cosmological_model.py:388`, `LamCDMScenario.__init__`:
```python
upper_limit=0.86,
```
(`lower_limit=0.6` at line 389, unchanged.)

### 2.2 Proposed new value
```python
upper_limit=1.00,
```
Rationale for the exact number: it is the top of the G-EXT wing already ratified and run
(AMENDMENT G-EXT, row #284: "H_GRID_41 (0.600–0.860) ∪ {0.870, ..., 1.000} = 55 nodes"). Raising
to exactly 1.00 admits precisely the already-authorized grid and no more — it does not open the
door to arbitrary future h values without a fresh registration. (An alternative of raising to
e.g. 1.2 or removing the bound was considered and rejected: nothing in the ratified record calls
for evaluating h outside the G-EXT wing, and a wide-open bound would silently admit unregistered
future grids without a fresh gate.)

### 2.3 Justification — this is a prior-support bound, not a physical constant
`h` (dimensionless, `H_0 = 100 h` km/s/Mpc) has no hard physical ceiling in ΛCDM parameter
estimation; `upper_limit`/`lower_limit` on `CosmologicalParameter` exist to (a) bound the
`randomize_by_distribution=uniform` draw used during **simulation** (`Model1CrossCheck`, unrelated
usage — simulation never samples h from this scenario; `LamCDMScenario.h` is only consumed by the
**evaluate** path) and (b) gate which h-grid values `BayesianStatistics.evaluate()` is willing to
compute a posterior at (`bayesian_statistics.py:4655-4658`). The literature/production convention
(`constants.py`) already treats the fiducial cosmology as h=0.73 (`H=0.73` per CLAUDE.md); 0.86
was never a physically-derived ceiling — it is simply where the original 41-node grid
(`H_GRID_41`) happened to stop. Extending it to 1.00 to admit the already-registered G-EXT wing
changes no formula and no computed posterior value for any h inside the old bound; it only widens
which h-values are *admissible to ask about*.

### 2.4 Dimensional analysis
`h` is dimensionless (Hubble parameter in units of 100 km/s/Mpc, per the `unit="s*Mpc/km"` field
— actually the field records units of `1/H` implicitly via how it's consumed in `physical_relations.dist()`,
but `h` itself enters that formula as a pure scale factor). Changing `upper_limit` from 0.86 to
1.00 is a change to a bound on a dimensionless number; it introduces no unit mismatch and touches
no formula (`dist()`, the Fisher/CRB code, the PSD, etc. are all untouched — only the admissibility
gate at evaluate() entry changes).

### 2.5 Limiting-case check (the load-bearing one for this node)
**Byte-identity below the old bound.** For every h in the already-completed 41-node grid
(0.600–0.860) and any h ≤ 0.86 in general, raising `upper_limit` must produce **bit-for-bit
identical** posterior output, because:
- The bounds check at `bayesian_statistics.py:4655-4658` is a pure gate (`raise ValueError`) —
  it performs no computation that feeds into the posterior; relaxing the ceiling cannot change
  what happens for h values that were already inside the old ceiling.
- No other code path reads `h.upper_limit` in a way that affects a within-bound h (grepped
  `h.upper_limit` and `Omega_m.upper_limit` usage across `bayesian_statistics.py`; the only other
  use, at line 5716 (`h_max=self.cosmological_model.h.upper_limit`), feeds a downstream
  `redshift_upper_limit`/`z_max` computation in `_compute_...` — **this needs the g-byte-id check
  specifically at that call site**, since a wider `h_max` could in principle shift a
  `min(z_max, redshift_upper_limit)` clamp even for an in-bound h evaluation, if the z_max
  computation is itself a function of `h_max` rather than the per-call h. This is exactly the
  kind of second-order effect the byte-id gate exists to catch — see §2.6.

### 2.6 g-byte-id gate plan (0 mismatches required below 0.86)
1. Take the banked posterior outputs for the 41 completed H_GRID_41 tasks from job 6747032
   (`results/.../tree2_20260830/a18_prod_arm/` retrieved outputs — `posteriors/*.json`,
   `posteriors_with_bh_mass/*.json`, `diagnostics/event_likelihoods.csv`) as the reference.
2. After the one-line `upper_limit=1.00` edit, re-run `--evaluate` for the same 41 h-nodes
   (same CRB CSV, same galaxy-catalogue pin, same seeds) locally (CPU) or as a small cluster
   re-submission of just those 41 tasks.
3. Diff every output file byte-for-byte (`diff` / `json`-normalized compare on the posterior
   JSONs, exact-float compare on the CSV) against the banked reference.
4. **Gate: 0 mismatches.** Any difference — including in the `h_max`-derived `z_max` clamp
   flagged in §2.5 — is a STOP: it means the bound is not purely a support gate and the
   dimensional-analysis/limiting-case section above needs revision before the edit can land as
   drafted.
5. Because `_compute_...`'s `h_max=self.cosmological_model.h.upper_limit` call (line ~5716) is
   evaluated once per `BayesianStatistics` instance (not per grid-node h), the byte-id check must
   include at least one multi-node batch evaluate() call (not just single-h calls) to exercise
   that shared-instance code path realistically.

### 2.7 Regression test (to add alongside the edit)
New test in `darksiren_emri_test/` (e.g. `test_cosmological_model.py` or an addition to
`datamodels/parameter_space_test.py`), asserting:
```python
def test_lamcdm_h_bound_admits_gext_grid():
    """h.upper_limit must admit the ratified G-EXT wing (row #284) without raising, and the
    guard at bayesian_statistics.py:4655-4658 must still reject values outside it."""
    scenario = LamCDMScenario()
    assert scenario.h.lower_limit == 0.6
    assert scenario.h.upper_limit == 1.00
    # in-bound: no raise
    for h in [0.86, 0.87, 0.94, 1.00]:
        assert scenario.h.lower_limit <= h <= scenario.h.upper_limit
    # out-of-bound: still rejected
    assert not (scenario.h.lower_limit <= 1.01 <= scenario.h.upper_limit)
```
plus a byte-identity regression using a small canned CRB fixture, evaluating h=0.73 (an existing
in-bound node) before/after the edit and asserting identical posterior output — this is the
permanent form of the one-off §2.6 check, so a future edit to this bound can re-run it instead of
re-deriving the check from scratch.

### 2.8 Exact minimal diff (NOT APPLIED — draft only)
```diff
--- a/darksiren_emri/cosmological_model.py
+++ b/darksiren_emri/cosmological_model.py
@@ -385,7 +385,10 @@ class LamCDMScenario:
         self.h = CosmologicalParameter(
             symbol="h",
-            upper_limit=0.86,
+            # Raised from 0.86 -> 1.00 to admit the ratified G-EXT grid wing (14 nodes,
+            # 0.870-1.000, AMENDMENT G-EXT under row #284; g-byte-id PASS 0 mismatches below
+            # 0.86 required before this lands — see graph1_20260901/exec/b-hprior-fix/RECORD.md).
+            upper_limit=1.00,
             lower_limit=0.6,
             unit="s*Mpc/km",
             randomize_by_distribution=uniform,
```
[PHYSICS] commit message convention (not created): `[PHYSICS] cosmological_model.py: raise
LamCDMScenario.h.upper_limit 0.86 -> 1.00 (admit ratified G-EXT grid wing, row #284; g-byte-id
PASS)`.

**This diff is NOT applied.** It is handed to the chair for the `/physics-change` gate run before
any edit lands, per the task's explicit instruction.

---

## 3. Cluster rerun plan for the 14 failed tasks (DRAFT — NOT SUBMITTED)

- **Task IDs:** SLURM array tasks 41–54 of a re-submission of job template
  `cluster/a18_ma1d_headreadout_iiib.sbatch` (or a narrower copy scoped to just these 14, to avoid
  re-running the 41 already-completed nodes). Recommend a scoped re-submit via
  `sbatch --array=41-54 cluster/a18_ma1d_headreadout_iiib.sbatch` against the *same* `RUN_DIR`
  (`$WORKSPACE/run_20260831_a18_ma1d_iiib`) so the retrieved output set becomes complete
  (41 banked + 14 fresh = 55), rather than standing up a new out-root.
- **Submit script:** `cluster/submit_a18.sh` (DRY_RUN=1 printer; already exists, already scoped to
  this exact job) — would need a one-line `--array` override added, or the orchestrator can pass
  `--array=41-54` directly to `sbatch` outside the wrapper script the same way the wrapper's own
  `run_or_print` composes its command.
- **Precondition:** the `/physics-change` gate above must PASS and the `upper_limit=1.00` edit
  must be committed on the cluster's checked-out commit (preflight `HEAD` match, per
  `.claude/skills/cluster/SKILL.md`) before these 14 tasks can complete — under the *old* bound
  (0.86) they fail identically to the original run, by construction.
- **Expected cost:** 14 tasks × ~1.7 CPU-h/task (per-task cost model unchanged from the template,
  per the sbatch header: "AMENDMENT G-EXT's added cost is purely from the 14 extra array tasks,
  not from any per-task slowdown") ≈ **23.8 CPU-h** naively; the task ceiling given is ≤ 20 CPU-h.
  At `cpus-per-task=16`, 14 tasks × 45 min walltime × 16 cores = 14 × 0.75 h × 16 = **168 core-h**
  if all run serially back-to-back at full walltime, but the array runs tasks in parallel subject
  to cluster scheduling, and per-task CPU-h (not wall×cores for the whole array) is the right
  accounting unit matching the ~94 CPU-h / 55-node ≈ 1.7 CPU-h/node figure already used for the
  full G-EXT costing (`AMENDMENT G-EXT`: "55 × ~1.7 CPU-h ≈ 94 CPU-h"). 14 × 1.7 = **23.8 CPU-h**,
  which is **~5 CPU-h over the stated ≤20 CPU-h bound**. Flagging this discrepancy rather than
  rounding it away: either (a) the ≤20 CPU-h ceiling in this task's authorization was set assuming
  a smaller effective per-task cost for the wing nodes specifically (plausible — high-h nodes may
  have fewer viable hosts / faster convergence, untested), or (b) the ceiling needs revisiting
  with the author before submission. **This is a fresh [RULE]/[DO] boundary question, not decided
  here** — flagging it for the chair rather than silently submitting over-cap or silently
  shrinking scope.
- **g-byte-id check for the rerun itself:** not applicable in the same sense as §2.6 (these 14
  tasks never produced output before — there is no "old" byte-identical baseline for h ≥ 0.87);
  the relevant check is that the 41 already-banked nodes remain byte-identical after the bound
  edit (§2.6), and that the 14 new nodes' outputs, once produced, are consistent with the
  disclosed-irrelevant framing (posterior tail at h ≥ 0.85 ≈ 5e-13 — a sanity check that these new
  posteriors are indeed negligible-weight, not a fresh flip-relevant surprise).
- **Not submitted.** No `sbatch` call was made by this task.

---

## 4. Verdict

**Trigger-file-required.** The fix cannot be made in run configuration outside
`cosmological_model.py`: `BayesianStatistics.__init__` constructs `LamCDMScenario()`
unconditionally with no override path, and the 0.86 ceiling is a hardcoded dataclass-field
literal at `cosmological_model.py:388`. A full `/physics-change` presentation (old/new value,
justification, dimensional analysis, limiting-case + g-byte-id plan, regression test, minimal
diff) is drafted above in §2 for the chair to gate; no edit has been applied. The cluster rerun
plan for the 14 failed G-EXT wing tasks (job template `a18_ma1d_headreadout_iiib.sbatch`,
array 41–54) is drafted in §3 but not submitted, and flags a possible ~4 CPU-h overshoot against
the ≤20 CPU-h authorization ceiling for the chair to resolve before launch.
