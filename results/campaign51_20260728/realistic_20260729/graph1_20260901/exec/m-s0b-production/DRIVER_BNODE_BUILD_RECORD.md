# DRIVER_BNODE_BUILD_RECORD — `--b-half-width` (PA-HIER-31(d) node registration)

Date: 2026-09-02. Builder: driver-amendment node, Research Graph 1 Branch D.

**Authorization chain, quoted.** Ledger row #290 decisions-table row 6 (the driver-build
authorization for S0-B) and the standing prereg
`PREREGISTRATION_HIER_HTHETA_20260826.md` PA-HIER-31(d) (the registered S0-B node shape,
§2.1 S0-B row, quoted below). Resolves the STOP in
`graph1_20260901/exec/m-s0b-production/LAUNCH_RECORD.md` option 1: *"authorize a small
driver amendment (analogous to `--s-half-step`) adding `b_plus_re`/`b_minus_re` at ±0.033
as opt-in nodes, then re-run this launch node."*

**Scope, stated up front.** Effort was mechanical/medium per the task's own framing: the
node values are REGISTERED (PA-HIER-31(d): ±0.033), this build implements *expressibility*
only — a CLI-level mechanism for a caller to register `b_plus_re`/`b_minus_re` into
`THETA_NODES` so the driver can run/dispatch/name output for them, mirroring the existing
`--s-half-step` precedent (`THETA_NODES_HALF_STEP`, `s_plus_half`/`s_minus_half`). It does
**not** add a `score_b_re`/`Z_b_re` statistic to `compute_scores` — see "Routed item" below.

## What was changed

File: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`
(+86/-1). Test file:
`darksiren_emri_test/bayesian_inference/test_theta_zwindow.py` (+95, the location this
task specified for driver regression tests, matching the file's existing
`_load_driver_module` shim used for every other `hier_s0_driver.py` unit test).

### 1. `DEFAULT_B_HALF_WIDTH = 0.02` (module constant)

Pinned equal to the as-built `THETA_NODES["b_plus"]`/`["b_minus"]` half-width already in
the file (`(0.02, 1.0)` / `(-0.02, 1.0)`). This constant is what makes the default CLI
invocation byte-identical (see §3).

### 2. `b_re_theta_nodes(half_width) -> dict[str, tuple[float, float]]`

Pure function, no mutation:

```python
def b_re_theta_nodes(half_width: float) -> dict[str, tuple[float, float]]:
    return {
        "b_plus_re": (half_width, 1.0),
        "b_minus_re": (-half_width, 1.0),
    }
```

At `half_width=0.033` this produces exactly the prereg's registered pair (PA-HIER-31(d),
§2.1 S0-B row, quoted verbatim in `LAUNCH_RECORD.md`):

```
b_plus_re    (+0.033, 1)
b_minus_re   (−0.033, 1)
```

### 3. `apply_b_half_width(theta_nodes, half_width) -> None`

```python
def apply_b_half_width(theta_nodes: dict[str, tuple[float, float]], half_width: float) -> None:
    if half_width != DEFAULT_B_HALF_WIDTH:
        theta_nodes.update(b_re_theta_nodes(half_width))
```

Mirrors the existing `--s-half-step` mechanism (`THETA_NODES.update(THETA_NODES_HALF_STEP)`,
called unconditionally from `main()` when the boolean flag is set) — except `--b-half-width`
is a **width** flag, not a boolean, so the merge is instead gated on the value differing
from the default. `main()` calls it right after the existing `--s-half-step` line:

```python
if args.s_half_step:
    THETA_NODES.update(THETA_NODES_HALF_STEP)

apply_b_half_width(THETA_NODES, args.b_half_width)
```

### 4. New CLI flag `--b-half-width` (`type=float`, `default=DEFAULT_B_HALF_WIDTH`)

Documented in `--help` with the byte-identity guarantee, the "never combined" rule
citation, and the compute_scores non-scoring disclosure (full help text in the diff).
`--nodes`' own help text is amended to mention `b_plus_re`/`b_minus_re` alongside the
existing `s_plus_half`/`s_minus_half` mention.

## Byte-preservation argument (default path)

Every pre-existing invocation of `hier_s0_driver.py` either omits `--b-half-width` (then
`args.b_half_width == DEFAULT_B_HALF_WIDTH` by argparse's own default-substitution) or, if a
future caller passes `--b-half-width 0.02` explicitly, the Python float literal `0.02`
compares `==` to the stored default `0.02` (both parsed by the same `float()` conversion
path from the same decimal literal — no rounding-path divergence, unlike the IEEE-754
concern GATE T-ID raises for the θ-hook itself; this is a plain CLI-value equality check,
not a floating-point physics computation). In both cases `apply_b_half_width` takes the
`if` branch's `False` arm and never calls `theta_nodes.update(...)` — `THETA_NODES` is
returned to `main()`'s node-validation step in **exactly** the state every pre-`--b-half-width`
build left it in: the 7 keys `{truth, b_plus, b_minus, s_plus, s_minus}` plus whatever
`--s-half-step` added, and nothing under the `_re` names. `--nodes b_plus_re` without a
non-default `--b-half-width` therefore still hits the identical `unknown node ... must be
one of {sorted(THETA_NODES)}` `SystemExit` it always did — the same argument the file's own
comment already makes for `--s-half-step`/`s_plus_half`. No other code path was touched:
`run_theta_node`, `gather_node_results_from_disk`, `compute_scores`, `gate_eng`, and every
`node_{node}{suffix}` path-construction site are unchanged and key off the node-name string
generically (confirmed by reading; see "Naming/path convention honored" below) — none of
them hardcode an enum of node names that would need updating for a key that is simply never
present.

## Naming/path convention honored

Per the task's instruction to honor the prereg's distinct node names if they flow into
output paths: `run_theta_node`'s `node_root = work_root / f"node_{node}{suffix}"` (and the
equivalent in the S0-R function) is generic on the `node` string — it was **not** written
against a fixed enum. Registering `b_plus_re`/`b_minus_re` under those literal names
therefore automatically produces `node_b_plus_re<suffix>` / `node_b_minus_re<suffix>`
output directories the first time `--nodes ...,b_plus_re,b_minus_re --b-half-width 0.033`
is run — no additional wiring was needed or added for this. Confirmed by reading (not
assumed): `hier_s0_driver.py:1027` and `:1124` (S0-A/S0-R `node_root` construction),
`:2474` (`gather_node_results_from_disk`'s equivalent), all keyed on the `node` string
parameter.

## Routed item (not decided here)

`compute_scores`'s `score_b` (and every `b_ready`/`_axis_missing`/`has_b` check feeding it)
is keyed on the **literal strings** `"b_plus"`/`"b_minus"` throughout, with a hardcoded
`0.04` denominator (`= 2 × 0.02`). `b_plus_re`/`b_minus_re` are therefore structurally
**invisible** to `compute_scores` as built — a run at `--b-half-width 0.033` will produce
correct per-event `ln_L` at the registered ±0.033 theta values, on disk under
`node_b_plus_re.../node_b_minus_re...`, but no pooled `score_b`/`Z_b` statistic at that
width comes out of `compute_scores`/`--score-only` today. This is **not** a defect in the
"never combined" sense (§2.1(a)) — quite the opposite, it structurally *guarantees* the two
grids can never be silently folded into one secant, since `compute_scores` simply never
looks at the `_re` keys. But it does mean a follow-on build (a `score_b_re`/`Z_b_re`
addition mirroring the existing `score_half`/Richardson pattern for the s-axis, gated on a
`has_b_re` flag exactly like `has_s_half`) is needed before S0-B's decisive
threshold-comparison read (`B0-B`, prereg §4.1, "identical bands, applied to S0-B") can be
computed by this driver. **This build deliberately stops at node registration/expressibility
and routes the scoring question to the author/next build task**, per the task brief's
explicit narrowing ("the node values are REGISTERED, you implement expressibility only")
and to keep this change mechanical rather than adding a new registered-statistic formula
without its own review pass. The `--jobs>1` path was not touched and remains dead per the
existing constraint.

## Tests

Added to `darksiren_emri_test/bayesian_inference/test_theta_zwindow.py` (the file this
task specified; it already hosts every other `hier_s0_driver.py` regression test via its
`_load_driver_module` results-path-loading shim):

- `test_default_b_half_width_constant_matches_the_as_built_grid` — `DEFAULT_B_HALF_WIDTH
  == 0.02`, matching `THETA_NODES["b_plus"]`/`["b_minus"]`.
- `test_b_re_theta_nodes_produces_the_registered_pair` — `b_re_theta_nodes(0.033) ==
  {"b_plus_re": (0.033, 1.0), "b_minus_re": (-0.033, 1.0)}` (the exact PA-HIER-31(d) values).
- `test_apply_b_half_width_at_default_is_a_byte_identical_no_op` — calling
  `apply_b_half_width` at the default leaves the dict wholly unchanged (old-default
  assertion).
- `test_apply_b_half_width_nondefault_registers_distinct_re_nodes` — a non-default width
  adds the `_re` pair without touching/overwriting `b_plus`/`b_minus` (new-flag assertion).
- `test_compute_scores_never_folds_the_re_nodes_into_score_b` — the never-interchangeable
  guard: an `all_nodes` dict carrying **both** the as-built pair and a deliberately
  wrong-magnitude `_re` pair (±9.0 vs. the real ±0.1/±0.2) still scores `score_b` using
  *only* the as-built pair at its own 0.04 denominator — proving the two grids cannot be
  accidentally commingled through `compute_scores`, structurally, not by caller discipline.

## Test run (verbatim)

```
$ uv run pytest darksiren_emri_test/bayesian_inference/test_theta_zwindow.py -q
...
=== 37 passed in 7.46s ===
```

Full fast suite (repo convention, single-file coverage gate ignored — the coverage
threshold is a whole-suite gate and fails on any partial run by design):

```
$ uv run pytest -m "not gpu and not slow" -q
...
Required test coverage of 25.0% reached. Total coverage: 73.40%
=== 2037 passed, 15 skipped, 30 deselected, 12 warnings in 335.39s (0:05:35) ===
```

```
$ uv run ruff check results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py darksiren_emri_test/bayesian_inference/test_theta_zwindow.py
All checks passed!

$ uv run ruff format --check results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py darksiren_emri_test/bayesian_inference/test_theta_zwindow.py
2 files already formatted

$ uv run mypy darksiren_emri/ darksiren_emri_test/
Success: no issues found in 220 source files
```

(`ruff format` initially flagged one missing blank line after the new
`apply_b_half_width` function — fixed in-place with `uv run ruff format` before the
`--check` re-run shown above.)

## What was NOT done

- No `compute_scores`/`score_b_re` extension (routed above).
- No commit.
- No `sbatch` submitted, no cluster interaction of any kind — this is a local, results-tree
  driver amendment only.
- `--jobs>1` was not touched; the file's own existing constraint that it stays dead is
  unaffected.
- No physics-trigger file was edited (`hier_s0_driver.py` lives under `results/`, not in
  `darksiren_emri/`, and is not on the physics-change trigger list in `CLAUDE.md`); `/physics-change`
  was not invoked, consistent with the task's own framing.

*Stamp: driver-bnode-build, 2026-09-02. Returns to the author/chair: the S0-B node set is
now launchable (`--config iiib --nodes truth,b_plus_re,b_minus_re,s_plus,s_minus
--b-half-width 0.033`, per `LAUNCH_RECORD.md`'s cost-cap caveat, which this build does not
resolve); the scoring extension for the `_re` pair is a separate, ROUTED follow-on.*

---

## Follow-on: score_b_re

Date: 2026-09-02. Gated on the coordinator's instruction: implement `score_b_re`/`Z_b_re`
**only if** the standing prereg defines its form (denominator, pairing rule, null band).
Verdict: **the registered text DEFINES it, unambiguously — implemented, with tests.**

### Registered-text finding

`PREREGISTRATION_HIER_HTHETA_20260826.md`, `PA-HIER-31` (§2.1 addendum, "(d) S0-B design —
nodes, statistics, reads"), quoted verbatim:

```
score_b,i   = [ lnL_i(+0.033,1) − lnL_i(−0.033,1) ] / 0.066
score_lns,i = [ lnL_i(0,√2)   − lnL_i(0,1/√2) ] / ln 2
Z_x = mean(score_x) / SEM(score_x)
```

- **Denominator:** `0.066` — the text does not present this as "2× whatever half-width is
  passed"; it is `2 × 0.033`, PA-HIER-31(d)'s own registered re-derived half-width (itself
  traced to `PA-HIER-29`'s measured `b_max ≈ 0.0661`, "a 5-node grid over ±0.0661 gives a
  half-step of 0.033"). Confirmed unambiguous — the same literal `0.066` also appears in
  §4.1(e)'s power band (`σ_b < 0.0661`) and the materiality band (`|b̂| < 0.0165`, "half the
  0.033 step"), all keyed to the same fixed 0.033/0.066 pair, never to a runtime parameter.
- **Pairing rule:** the same secant/pairing convention as the as-built `score_b` (per-event,
  numerator = `lnL(b_plus_re) − lnL(b_minus_re)`, pooled over events and seeds) — PA-HIER-31(a)
  states the two arms are "paired within arm" and "never combined into one secant, one Z, or
  one materiality read," which is a same-form/different-grid instruction, not a different
  pairing mechanism.
- **Null/band it feeds:** §2.1(e), quoted: `B0-B ≡ |Z_b| ≤ 3 and |Z_lns| ≤ 3 pooled (two-sided)
  ⇒ LEVER-DEAD-AT-N (production); either > 3 ⇒ LEVER-LIVE`; materiality `MIXED if |b̂| < 0.0165`;
  power `UNPOWERED if σ_b ≥ 0.0661`. `Z_x = mean/SEM` is the same machinery already implemented
  for `score_b`/`score_lns` — no new statistic form, only a new node-key/denominator pairing.
- **PA-HIER-33 does not touch it.** PA-HIER-33's own "Supersedes (on ratification)" clause lists
  what it changes and explicitly states *"Untouched: PA-HIER-4's `score_lns` form and nodes;
  **`score_b`**; every band structure of section 4.1."* PA-HIER-33 is an s-axis-only null
  revision (the Bartlett-identity correction to `score_lns`'s expectation at truth); it carries
  no dependency, correction, or reservation touching `score_b`/`score_b_re` at all.

No element (denominator, pairing, band) was ambiguous or under-specified — the statistic did
not need to be derived or inferred; it was copied.

### What was implemented

File: `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`.

1. **`_B_RE_DENOM: float = 2.0 * 0.033`** — a module constant pinning the literal registered
   `0.066`, deliberately NOT derived from `DEFAULT_B_HALF_WIDTH`/`args.b_half_width` at
   runtime (see the code comment): PA-HIER-31(d) names one specific registered statistic, not
   a generic "score at whatever width the caller passes."
2. **`has_b_re`** readiness flag (`not _axis_missing(("b_plus_re", "b_minus_re"))`) added
   alongside the existing `has_s_half`, with the SAME discipline: deliberately NOT folded into
   the `has_b`/`has_s` "at least one axis ready" gate or the broken-pair `ValueError` loop — a
   missing `_re` node degrades `score_b_re` to `n_pooled=0`/NaN, never raises. (This means, as
   with `s_plus_half`/`s_minus_half` today, a node dict containing ONLY the `_re` pair and
   nothing else still hits the pre-existing "both axes incomplete" `ValueError` — not a new
   restriction; the registered S0-B run always includes the s-axis too, so this never binds in
   practice. Caught and fixed in the first test draft below.)
3. **`score_b_re` computation**, inside the existing `for channel in channels:` loop,
   structurally parallel to the existing `score_b` block: independent `bp_re`/`bm_re`/
   `b_join_re` local names (inner join on `(seed, event_idx)`, identical to `score_b`'s own
   join), `score_b_re = (b_join_re["b_plus_re"] - b_join_re["b_minus_re"]) / _B_RE_DENOM`, then
   `mean_b_re, sem_b_re, z_b_re, n_b_re = _mean_sem(score_b_re)` via the SAME `_mean_sem` helper
   `score_b`/`score_lns`/`score_lns_R` already use (`Z = mean/SEM`, per §4.1's `Z_x` definition).
4. **Output keys** `"score_b_re": {"mean", "sem", "Z", "n_pooled"}` and
   `"score_b_re_available"` added to `compute_scores`'s per-channel output dict, alongside
   (never replacing) `"score_b"`/`"score_b_available"`.
5. **Report/print wiring**: `score_b_re` added to `score_only_payload`'s printed `stat_name`
   loop and a `score_b_re_available` line added to its report — same treatment as
   `score_lns_R`/`score_pahier33` (both PRINTING-ONLY additions from earlier build tasks).
6. **`compute_scores`'s docstring** extended with `score_b_re`'s registered form and an explicit
   note that it is PRINTING ONLY.

### What was deliberately NOT done (kept in scope)

- **`verdict_s0a`/`verdict_s0r` were not touched.** They still read only `score_b`/`score_s`
  (the as-built statistics). Re-pointing a verdict function at `score_b_re` — i.e. deciding
  that an S0-B run's OWN B0-B verdict is computed FROM `score_b_re`/`score_lns` rather than
  from the driver's generic `score_b`/`score_s` keys — is a re-adjudication act the same class
  as PA-HIER-33's own "re-pointing... is out of this build's scope" disclosure already made for
  `score_pahier33`. `score_b_re` is exposed as a new, independently reported statistic; wiring
  it into an actual B0-B pass/fail verdict function is left for the run/analysis step that
  consumes S0-B's real output, not this driver-plumbing build.
- **`score_b` (as-built) is untouched** — same node keys (`"b_plus"`/`"b_minus"`), same `0.04`
  denominator, same code path, verified by the never-interchangeable regression tests below.
- **No commit.**

### Tests

Added to `darksiren_emri_test/bayesian_inference/test_theta_zwindow.py`:

- `test_b_re_denom_constant_matches_the_registered_span` — `_B_RE_DENOM == 0.066`.
- `test_compute_scores_score_b_re_uses_the_registered_denominator` — hand-verified per-event
  arithmetic against PA-HIER-31(d)'s quoted form, on the registered S0-B node shape (truth,
  b_plus_re, b_minus_re, s_plus, s_minus — S0-B never runs the as-built b_plus/b_minus at all);
  confirms `score_b_available is False` alongside it (no as-built nodes supplied).
- `test_compute_scores_score_b_re_unavailable_when_re_nodes_absent` — graceful degradation
  (`n_pooled=0`/NaN, never a raise) on a node dict with no `_re` nodes; confirms `score_b`
  (as-built) scores normally and independently in the same call.
- `test_compute_scores_score_b_re_never_folded_into_score_b` — the never-interchangeable guard,
  BOTH directions: a node dict carrying both the as-built pair (deliberately wrong-magnitude,
  ±9.0) and the `_re` pair (the real ±0.033 values) scores `score_b_re` from only the `_re`
  pair (`mean == 1.0`, not `272.7...`) and `score_b` from only the as-built pair
  (`mean == 450.0`), proving structurally — not by caller discipline — that PA-HIER-31 §2.1(a)'s
  "never combined into one secant, one Z, or one materiality read" rule holds.

### Test run (verbatim)

```
$ uv run pytest darksiren_emri_test/bayesian_inference/test_theta_zwindow.py -q --no-cov
=== 41 passed in 2.49s ===
```

Full fast suite:

```
$ uv run pytest -m "not gpu and not slow" -q
...
Required test coverage of 25.0% reached. Total coverage: 73.40%
=== 2041 passed, 15 skipped, 30 deselected, 12 warnings in 317.37s (0:05:17) ===
```

(2041 vs. the prior record's 2037 — the +4 new `score_b_re` tests, all passing; no regressions
elsewhere.)

```
$ uv run ruff check results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py darksiren_emri_test/bayesian_inference/test_theta_zwindow.py
All checks passed!

$ uv run ruff format --check results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py darksiren_emri_test/bayesian_inference/test_theta_zwindow.py
2 files already formatted

$ uv run mypy darksiren_emri/ darksiren_emri_test/
Success: no issues found in 220 source files
```

### What was NOT done

- No commit.
- No `sbatch`/cluster interaction.
- `verdict_s0a`/`verdict_s0r` re-pointing (routed above, an author-scoped act).
- `--jobs>1` untouched, stays dead.
- No physics-trigger file edited (same disclosure as the base record above).

*Stamp: driver-bnode-build (follow-on), 2026-09-02. `score_b_re`/`Z_b_re` is now computable by
this driver, keyed correctly to PA-HIER-31(d)'s registered ±0.033 pair and 0.066 denominator,
independently of and never interchangeable with the as-built `score_b`. The one remaining gap
before S0-B's B0-B verdict can be read mechanically is wiring a verdict function to
`score_b_re`/`score_lns` for the production venue specifically — left to the author/next task
as a deliberate, disclosed scope boundary, not an oversight.*
