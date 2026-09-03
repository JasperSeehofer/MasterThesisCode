# DESIGN GATE (computability-only) — PA-HIER-34 / S0-C θ-grid at h ∈ {0.665, 0.78}

Reviewer: fresh computability-only reviewer (sonnet), no prior context on this node.
Scope: `REGISTRATION_ADDENDUM_PA-HIER-34.md` + `cluster/graph1_s0c_hgrid.sbatch` only.
No science read, no edit under `darksiren_emri/`, no `INFORMATION_FORECAST` file opened.
Every number below carries a source path. No aggregate computed over the registered
population (event_likelihoods.csv content was inspected only for shape: row/column counts
and a column's distinct values — not scored).

**Overall: GREEN**, with one AMBER open item (cost-basis reconciliation, self-disclosed in
the addendum's own §12 item 1 and not yet closed) that already gates submission and should
stay gating it.

---

## 1. sbatch reproduces job 6779532's driver invocation verbatim except `--h-nodes`

**GREEN.**

Diffed `cluster/graph1_s0c_hgrid.sbatch` against `cluster/graph1_m_s0b_production.sbatch`
(the only file matching `graph1_m_s0b_production*`) and against
`exec/m-s0b-production/LAUNCH_RECORD.md`.

- Driver CLI block, side by side:
  - old: `--arm S0-A --seeds 900101 --nodes "$NODE" --jobs 1 --out-root "$OUT_ROOT" --theta-sites 2.2 --smear off --config iiib --b-half-width 0.033`
  - new: identical, plus exactly one appended flag `--h-nodes "$H_VALUE"`.
  No other flag added, removed, or reordered in a way that changes argparse resolution.
- `--cpus-per-task=16`, `--ntasks=1`, `--partition=cpu_il` identical in both files.
  `--time` (03:00:00 → 00:45:00) and `--array` (0-4 → 0-9) differ, as expected for a
  cheaper per-cell design; not a reproduction defect.
- Array → (h, node) mapping: `H_IDX=TID/5`, `NODE_IDX=TID%5`, `H_LIST=(0.665 0.780)` (2
  elements), `NODES` (5 elements) → 10 tasks, exactly 2×5, with an explicit bounds check
  (`H_IDX -ge ${#H_LIST[@]}` → STOP) guarding against a mis-sized `--array` range. Correct.
- Per-h out-roots: verified from source, not assumed. Read `_node_dir_suffix`
  (`fanout1_20260829/hier_s0_driver.py:512-561`) directly — its signature takes
  `theta_sites, smear, config, theta_phi_divisor, sky_cone_k, catalogue_leg_1d_mass_aware,
  theta_zwindow, z_window_k` and **no h argument at all**; the suffix is h-invariant. The
  addendum's claim that a second h at one out-root would silently overwrite the first is
  correct, and the sbatch's `OUT_ROOT="$WORKSPACE/graph1_s0c_hgrid_20260904/$H_LABEL"` is the
  necessary and sufficient fix.
- Ancestor HEAD pin: `git merge-base --is-ancestor "$EXPECTED_COMMIT" HEAD` — same pattern
  row #331 introduced into the original sbatch (commit `081b1f28`, diff read directly: it
  replaced a strict-equality check with this exact ancestor check). Correctly reproduced.
  One naming nuance, not a defect: the new file pins `EXPECTED_COMMIT="081b1f28"` while the
  *original* S0-B sbatch still pins `"9336364c"` (the row-#325-era spec commit, an ancestor
  of `081b1f28`). This looks like drift at first glance, but
  `exec/m-s0b-production/READOUT_RECORD.md` §"Provenance" states job 6779532 actually ran,
  single-commit across all 5 tasks, at **`081b1f28`** — so the new pin names the exact commit
  the reused h=0.73 cells were produced at, which is the tighter and more correct anchor for
  *this* addendum's purpose (byte-comparability of the reused cells), not a mismatch.
- Modules/venv sourcing (`source cluster/modules.sh`; `source "$VENV_PATH/bin/activate"`;
  `cd "$PROJECT_ROOT"`) is byte-identical between the two files.
- `write_provenance` call present in both, with a task-specific message in the new file
  naming the node, h, and job 6779532 lineage.

## 2. h=0.73 reuse claim (driver/package unchanged since `081b1f28`)

**GREEN — ran the check.**

```
$ git rev-parse --short=8 HEAD
06a12422
$ git diff --stat 081b1f28 HEAD -- darksiren_emri/ results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py
(empty)
$ git diff --quiet 081b1f28 HEAD -- darksiren_emri/ .../hier_s0_driver.py && echo EMPTY
EMPTY
```
(Local HEAD has advanced to `06a12422` since the addendum's `08060e2a` — docs-only commits;
the diff-quiet result is unaffected.) Cross-checked transitively: `9336364c` (the commit the
*original* S0-B sbatch pins) is an ancestor of `081b1f28`, and
`git diff --stat 9336364c 081b1f28 -- darksiren_emri/ .../hier_s0_driver.py` is also empty —
so the reuse claim holds under either candidate pin commit, not only the one the addendum
picked. The sbatch's own runtime physics-freeze guard (`git diff --quiet 081b1f28 HEAD --
darksiren_emri/ "$DRIVER"`, STOP on failure) reproduces exactly this check at submission
time, so a future drift is caught mechanically, not only at authoring time.

Also verified the driver-code claims the reuse argument leans on, by reading, not trusting
prose:
- `H_GEN = H_TRUE` = 0.73 (`hier_s0_driver.py:85`).
- `main()`: `h_values = tuple(float(x) for x in args.h_nodes.split(",")) if args.h_nodes else
  (H_GEN,)` (`:3074-3075`) — confirmed: an explicit `--h-nodes 0.73` resolves to the same
  1-tuple `(0.73,)` as the omitted-flag default. The byte-identity argument is sound.

## 3. Stencil coefficients — re-derived by hand

**GREEN.** Independent derivation via Lagrange-basis differentiation at the middle node of a
non-uniform 3-point grid, not by pattern-matching the addendum's own algebra.

Let `x₋ = h₀ − a`, `x₊ = h₀ + b` with `a = Δ₋ = h₀ − h₋`, `b = Δ₊ = h₊ − h₀`. Differentiating
the three Lagrange basis polynomials at `x₀` gives
`L₋'(x₀) = −b/(a(a+b))`, `L₀'(x₀) = (b−a)/(ab)`, `L₊'(x₀) = a/(b(a+b))`, hence

```
f'(x₀) = [ −b² f(x₋) + (b²−a²) f(x₀) + a² f(x₊) ] / (ab(a+b))
```

Substituting `a = Δ₋`, `b = Δ₊`:

```
f'(x₀) = [ Δ₋² f(x₊) − Δ₊² f(x₋) + (Δ₊² − Δ₋²) f(x₀) ] / (Δ₋ Δ₊ (Δ₋ + Δ₊))
```

— **exactly** the addendum's §4.3 formula
`D_x,i = [Δ₋² s_i(h₊) − Δ₊² s_i(h₋) + (Δ₊²−Δ₋²) s_i(h₀)] / (Δ₊Δ₋(Δ₊+Δ₋))`. Uniform-grid
reduction (`Δ₋=Δ₊=Δ`) collapses this to `[f(h₊)−f(h₋)]/(2Δ)`, matching the addendum's stated
central-difference limit.

Second derivative at `x₀` from the same three basis functions:
`f''(x₀) = 2[b f(x₋) − (a+b) f(x₀) + a f(x₊)] / (ab(a+b))`, i.e. with `a=Δ₋,b=Δ₊`:
`2[Δ₊ f(x₋) − (Δ₋+Δ₊) f(x₀) + Δ₋ f(x₊)] / (Δ₋Δ₊(Δ₋+Δ₊))` — matches the addendum's §4.3
curvature `C_x,i = 2[Δ₋ s_i(h₊) − (Δ₊+Δ₋) s_i(h₀) + Δ₊ s_i(h₋)] / (Δ₊Δ₋(Δ₊+Δ₋))` term for
term (the `Δ₋ f(h₊)` / `Δ₊ f(h₋)` cross-pairing is the correct one, not a typo).

Numeric node spacing: `h₀=0.73`, `h₋=0.665 ⇒ Δ₋=0.065`; `h₊=0.78 ⇒ Δ₊=0.050`. Both arithmetic
facts check out (`0.730−0.665=0.065`, `0.780−0.730=0.050`) and match §3.1's stated asymmetry.

## 4. Disposition table three-valued, fresh RULE, CONDITIONAL-ON-R4

**GREEN.**
- §5.1 (derivative): `RESOLVED / NOT-RESOLVED / INSTRUMENT-DEFECT` — three-valued, and
  `INSTRUMENT-DEFECT` is reached whenever *any* §6 gate is red, so a gate failure cannot
  silently fall through to a banked reading.
- §5.2 (h-displacement, evaluated only on RESOLVED): `IMMATERIAL-IN-h / MATERIAL-IN-h /
  INDETERMINATE` — three-valued, with `INDETERMINATE` also catching the §4.3 linearity-check
  failure case (not just the numeric straddle-zero case).
- §5.4 states, in terms, "**every §5.1/§5.2 disposition is booked CONDITIONAL-ON-R4** until
  the R4 comparand (job 6790708) is read", with a stated lift condition (≤1e-12 relative,
  GATE T-ID) and a stated fallback (dispositions stand as instrument-only measurements, no
  h-bound enters the split). This is the docket R5 "defer behind R4" reading, carried
  correctly rather than silently dropped.
- "Fresh RULE": §5.1/§5.2 row consequences and §12 item 4 both say the addendum returns to
  the author/chair as a fresh [RULE] rather than self-ratifying; §0 and the file header both
  carry `Status: PROPOSED`.

## 5. Gates listed with STOPs

**GREEN.** §6 lists 8 rows (`g-score-null`, `g-znorm`, `C-C identity`, a not-a-gate
disclosure row, `GATE ENG`, `g-precision`, `pins`, `g-population`), each with a registered
form and a source. The section's own preamble is the STOP consequence for all of them: "red
⇒ INSTRUMENT-DEFECT" (§6 header) → "nothing banked; returns as fresh RULE naming the gate"
(§5.1 row 3). Separately, the sbatch itself carries two literal, mechanical `STOP:`/`exit 1`
gates that fire *before* any driver invocation: the ancestor-pin check and the physics-freeze
guard (`darksiren_emri/`+driver diff-quiet vs `081b1f28`) — both read directly in the sbatch
source, both print `STOP: ...` to stderr and `exit 1`. The array-bounds check is a third,
same pattern. One g-znorm sub-claim was spot-checked against real bytes rather than trusted:
"all five `selection_tables_h_0_73.json` md5 `e68ab957…` identical" — reran the md5sum
independently:
```
e68ab9578501a1c54008a6132eda7ec3  .../node_b_plus_re_.../selection_tables_h_0_73.json
e68ab9578501a1c54008a6132eda7ec3  .../node_s_minus_.../selection_tables_h_0_73.json
e68ab9578501a1c54008a6132eda7ec3  .../node_truth_.../selection_tables_h_0_73.json
e68ab9578501a1c54008a6132eda7ec3  .../node_s_plus_.../selection_tables_h_0_73.json
e68ab9578501a1c54008a6132eda7ec3  .../node_b_minus_re_.../selection_tables_h_0_73.json
```
Matches the addendum's stated hash exactly.

## 6. Cost (25 CPU-h allocation-basis cap vs 10 cells × ~7.5 min × 16)

**GREEN on the arithmetic; AMBER on documentation state (self-disclosed, already gating).**

Anchor re-checked at the source, not from the addendum's summary: `READOUT_RECORD.md §8`
sacct table for job 6779532 gives per-task Elapsed 7:21–7:36 (5 tasks, mean ≈ 7.49 min),
`sum=37.43 min`, `×16 cores = 9.98 CPU-h` for the 5 h=0.73 cells — i.e. **≈2.0 CPU-h/cell**
on the allocation basis, matching the addendum's §9 table figure exactly.
`10 fresh cells × 2.0 CPU-h/cell = 20.0 CPU-h` — the addendum's own §9 table states this
same figure ("**20.0 CPU-h**" for 10 fresh cells, allocation basis). `20.0 ≤ 25` CPU-h, so the
design **fits** the stated 25 CPU-h allocation-basis cap, with 5.0 CPU-h (20%) headroom —
this assumes per-cell wall time is roughly h-invariant (reasonable: byte-identical code,
same event count, same worker count; not something the addendum measures for h≠0.73, since
no h≠0.73 cell has run yet).

Documentation-state note (not a computability defect, but worth carrying forward): as
currently written, `REGISTRATION_ADDENDUM_PA-HIER-34.md` §9 does **not** itself state a 25
CPU-h cap — it states "**Cap: 10 CPU-h ORCHESTRATOR-DERIVED**" and explicitly flags that on
the allocation basis the fresh compute is "**2× OVER**" that 10 CPU-h figure, unresolved, and
names this as open item **§12.1** ("Which cost basis the 10 CPU-h cap binds ... put to the
chair BEFORE submission"). The 25 CPU-h figure checked against here was supplied as the
reviewer brief, not found in the committed artifact. Net effect: the arithmetic reconciles
cleanly (20.0 ≤ 25.0), but §12 item 1 remains, in the document as committed, an open
chair ruling that must land — and be reflected in §9 — before `sbatch` is submitted; the
sbatch file carries no independent cost gate of its own (cost is a launch-authorization
question, not a runtime-checkable one, so it cannot be gated in bash). This is exactly the
kind of item the addendum's own §10 launch block defers to "the chair/author," not a defect
in the design.

## 7. Zero fresh choices; kill criterion / max_revisions / blindness line present

**GREEN.**
- Every `[FLAG]` in the file is a disclosed, traceable choice, not a hidden one:
  - §3.3 `[FLAG: guard added, no physics]` — the physics-freeze guard itself (mechanical).
  - §4.2 `[FLAG: seed value]` — bootstrap resampling seed `20260904`.
  - §5.2 `[FLAG: the only imported threshold]` — `T_mat = 0.008`, quoted verbatim from row
    #302 / reused in row #345 D4, not invented here.
  No other numeric input in §3-§6 lacks a `[DOC]`/`[LOCAL]`/`[INFER]` provenance tag per the
  file's own convention (§0 header). The bootstrap count `B=2000` is stated alongside the
  flagged seed but is not itself flagged; it is a mechanical Monte-Carlo precision choice, not
  a scientific one, and does not change any band — noting it here for completeness, not as a
  defect.
- Kill criterion: §2 states the refutation rule inline ("Refute by: `|Z(∂score/∂h)| < 3` on
  both axes...") and §8 "Falsifiers (A14)" gives four explicit falsification conditions tied
  to the two competing hypotheses in §2.
- max_revisions: §11 states "Revision cap on this un-run registration: **1** (design), **0**
  after launch."
- Blindness line: present twice — §7 "Structural blindness (A10)" enumerates carried and new
  blindness items (d′, f, g), and §11 "Blindness and ordering (A22)" states the pre-registration
  ordering guarantee (bands/stencil/mapping/gates fixed before any fresh cell exists) and the
  no-post-submission-band-edit rule.

---

## Summary table

| check | verdict | note |
|---|---|---|
| (1) CLI verbatim + array/out-root/pin/modules | GREEN | one appended flag only; per-h out-root necessity confirmed from driver source |
| (2) driver/package unchanged since 081b1f28 | GREEN | ran `git diff --quiet`; empty, also empty transitively via 9336364c |
| (3) stencil + curvature coefficients | GREEN | independently re-derived via Lagrange differentiation; matches term-for-term |
| (4) three-valued dispositions, fresh RULE, CONDITIONAL-ON-R4 | GREEN | both §5.1 and §5.2 tables three-valued; §5.4 binding clause present |
| (5) gates with STOP consequences | GREEN | §6 cascades to INSTRUMENT-DEFECT; sbatch carries 3 literal STOP/exit-1 gates |
| (6) cost: 10×~7.5min×16≈20 CPU-h vs 25 CPU-h cap | GREEN (arithmetic) / AMBER (doc state) | fits with 5 CPU-h headroom; addendum's own §9/§12.1 cap language is not yet reconciled to 25 CPU-h — pre-existing, self-disclosed, already gates submission |
| (7) zero fresh choices; kill/max_revisions/blindness | GREEN | 3 flagged imports, all traceable; all three required lines present |

**No RED finding.** Nothing here makes the run wrong or unregistered as designed. The one
open item (cost-basis reconciliation, §12.1) is already a stated pre-submission blocker in
the addendum itself; this review adds independent confirmation that the underlying arithmetic
is correct and that it fits inside the 25 CPU-h figure given in the review brief, without
closing the open chair ruling — that stays with the chair/author, per §12.
