# b-pahier33-scorer — build node — 2026-09-02

**Authorization:** ledger row #290 ("rows 3-11 [DO] APPROVED — branch heads A-I trigger their
first items ... PA-HIER-33 scorer + iiib driver build"), decisions-table row 6 of
`RESEARCH_GRAPH_1_PROPOSAL_20260901.md` §1.4 (Branch D): `rd-runner11 -> b-pahier33-scorer`, DO,
Approved. Node spec (§1.4 table): "the PA-HIER-33 scorer (convention ratified rows #278/#280 via
the Richardson adjudication, row #275; never built) + the driver's missing iiib venue path ...
g-byte-id on all non-S0-B default paths ... 0 mismatches at N >= 1e5 pairs; red -> STOP m-s0b
launch ... cheap ... sonnet / medium (implementation from a ratified spec)." Effort: medium,
implementation from spec — no new scientific design.

Base commit: `1ec9514d` (branch `fix/p32d-classg-venue-repair`). Preceded by `rd-runner11`
(this record's sibling, `exec/rd-runner11/RECORD.md`).

## Spec source, quoted verbatim

**PA-HIER-33's registered rule** (`PREREGISTRATION_HIER_HTHETA_20260826.md` section 5, copied
verbatim from `tree2_20260830/T1_3_ES_NULL_DET_VALIDITY_20260830.md` section 5 — "PROPOSED —
NOT ADOPTED" at drafting, **RATIFIED row #278 item 1 / row #280**, the falsifier CONFIRMED it
row #275):

```
Es_null^{(arm)} = (Delta^2/6) . [ -3 <l'_i l''_i> - <l'_i^3> ],   l'_i = score_lns_i,
l''_i = [l_i(+Delta) - 2 l_i(0) + l_i(-Delta)]/Delta^2,
score_s_i = score_lns_i - Es_null^{(arm)}   (a pooled scalar shift, the arm's own null; NOT a
per-host table),
Z_s = mean(score_s) / SEM,   SEM = max(per-event SEM, seed-clustered SEM)   (PA-HIER-5 leg (a)),
with the bootstrap uncertainty of Es_null^{(arm)} added in quadrature to the SEM.
```

Ratification (row #278 item 1, ledger `BIAS_HISTORY_LEDGER.md:3179`): "the corrected null — the
arm's own likelihood — is the rule of record for the [HIER] s-score; T1.4's fresh-data
adjudication binds." Bootstrap: "bootstrap SD from 4000 event resamples"
(`T1_3_ES_NULL_DET_VALIDITY_20260830.md` section 2.3 table caption).

**PA-HIER-31's "iiib venue" design** (`PREREGISTRATION_HIER_HTHETA_20260826.md`, PA-HIER-31
section (d)/(g)): CoR-P production venue = the real production catalogue at
`normalization_mode="absolute_marginal"`, `host_z_kernel="volume_deconv"`
(`darksiren_emri/validation/correspondence_1d.py`'s `PRODUCTION_FLAGS`, already unconditional for
every config), `selection_in_completion_numerator="fused"`, `catalogue_global_selection="phi"`,
`smear_global_selection=False`, `theta_sites="2.2"`, `mass_filter_geometry="linear"`,
`mass_filter_k=1.5`, `catalogue_numerator_survival_2d="off"` — CLI flags verbatim from
`headreadout_20260827/iiib/run_metadata_21.json:cli_args` (read directly, quoted in full inside
the build's commit-region comments).

## Files changed

- `results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`
  (+385/-7, git diff)
- `darksiren_emri_test/bayesian_inference/test_theta_zwindow.py` (+262, new tests only)

**No file under `darksiren_emri/` (the physics-trigger package) was touched.** `git diff --stat`
against base commit `1ec9514d` confirms only the two files above changed.

## What was built

### 1. PA-HIER-33 scorer

- `compute_es_null_arm(l_plus, l0, l_minus, delta=_DELTA, n_bootstrap=4000, bootstrap_seed=0)` —
  the Bartlett-identity closed form above, plus a bootstrap SD over `n_bootstrap` event resamples
  with replacement (registered default 4000). `_DELTA = ln(2)/2 = ln(sqrt(2))`, PA-HIER-4's
  registered s-node grid, so `2*_DELTA == ln(2)` and `l'_i` is identically `score_lns_i` as
  already computed elsewhere in the module (no duplicate arithmetic).
- `_seed_clustered_sem(values, seed_col)` — `std(per-seed means, ddof=1)/sqrt(n_seeds)`, the
  PA-HIER-5 leg (a) SEM candidate.
- `compute_scores()` gains a new `score_pahier33`/`score_pahier33_available`/`es_null_arm` block
  per channel: `mean(score_lns) - Es_null^{(arm)}`, `SEM = sqrt(max(per-event, seed-clustered)^2
  + bootstrap_sd^2)`, `Z = mean/SEM`. **Available only when BOTH the s-axis (s_plus/s_minus) AND
  the truth node are present** — a strictly narrower condition than `score_s`'s own `has_s`
  (`score_pahier33` additionally needs the truth node's `l_i(0)`, which `score_lns`/`score_s`
  never require). Degrades to `n_pooled=0`/NaN/`False`, never raises, on any narrower node dict
  (including runner-11's own b-only 8-cell shape — verified by a dedicated regression test using
  that exact node-dict shape).
- `score_only_payload`'s markdown printer gains `score_pahier33` to its per-channel stat loop
  and two new report lines (`score_pahier33_available`, `es_null_arm`).
- **No verdict function (`gate_parity`/`verdict_s0a`/`verdict_s0r`) was re-pointed at
  `score_pahier33`.** This is deliberate, not an oversight: PA-HIER-33's own "what returns to the
  author" list (section 5) names "[RULE] whether the P1 data are re-read under PA-HIER-33 ... or
  the B0-A' stands" as an author-scoped act — a build task silently flipping which statistic a
  verdict function reads would BE that re-adjudication. `score_pahier33` is exposed as a reported
  field only; re-pointing any verdict at it is out of this build's scope.

### 2. The iiib (CoR-P production) venue path

- `build_iiib_venue(work_root, seed, sigma_z_scale=1.0)` — loads (does NOT draw) the real
  production inputs: `pd.read_csv(c1d.CRB_CSV_PATH)` (the pinned seed61000 CRB rows,
  `c1d.CRB_CSV_MD5`) and `c1d._load_galaxy_catalog_handler(c1d.REDUCED_CATALOGUE_PATH)` (the
  pinned reduced GLADE catalogue, `c1d.REDUCED_CATALOGUE_MD5`) — both loaders and both md5 pins
  already existed in `correspondence_1d.py` (not a physics-trigger file; used unmodified). Both
  pins are checked (`c1d.check_crb_pin()`, `c1d.check_reduced_catalogue_pin()`) and a mismatch
  raises `RuntimeError` (dataset pinning STOP-gate, CLAUDE.md 2026-08-20 rule) before any data is
  read. Non-identity `sigma_z_scale` raises `ValueError` (no CoR-P dosed-scale analogue — S0-R's
  dosing is a CoR-M-only disclosed-null instrument, PA-HIER-3/22).
- `_build_venue()` dispatches `config="iiib"` to it; `CONFIG_CHOICES` extended to
  `("b0i", "ft", "iiib")`.
- `run_theta_node()`'s config elif chain gains an `"iiib"` branch calling
  `c1d.run_mirror_seed_inprocess()` with `IIIB_CATALOGUE_NUMERATOR_SURVIVAL="phi"`,
  `IIIB_CATALOGUE_GLOBAL_SELECTION="phi"`, `IIIB_COMPLETION_CELL="fused"`,
  `IIIB_EVENT_MEASURE="ratio"`, `IIIB_MASS_FILTER_GEOMETRY="linear"`, `IIIB_MASS_FILTER_K=1.5` —
  the PA-HIER-31(g) flags. `normalization_mode`/`host_z_kernel` are NOT passed (they are already
  hardcoded unconditionally inside `run_mirror_seed_inprocess` via `PRODUCTION_FLAGS`, same as
  b0i/ft). `theta_sites="2.2"`/`smear_global_selection=False` (the CoR-P-faithful pair,
  PA-HIER-31(b)) are CALLER-supplied via the SAME `theta_sites`/`smear` kwargs the b0i/ft
  branches already use — venue construction does not force either; the S0-B measure step (NOT
  built by this task) is responsible for passing `--theta-sites 2.2 --smear off --config iiib`.
- `--config` argparse help text updated to describe `iiib`.

## g-byte-id evidence — what was verified and what was NOT

The node's own gate ("0 mismatches at N >= 1e5 pairs; red -> STOP m-s0b launch") reads, on its
most direct interpretation, as a byte/numeric-identity check of the new `iiib` venue path against
the banked real production reference (`c1d.BANKED_CSV_PATH`,
`results/prod2d_closure_20260818/postfix_baseline/iiib/event_likelihoods.csv`) at theta identity
— i.e. a GATE-T-ID-equivalent check for the new venue path, over the full N≈1588-event production
set (≈1588 x ~70 columns ≈ 1e5 pairs).

**This full-scale identity run was NOT executed by this build.** PA-HIER-31(i)'s own costing
prices a SINGLE production theta-node evaluate() call at 14.93–22.9 CPU-h (unsmeared form,
4 nodes = 60–92 CPU-h total) — even one node is not "cheap" by this graph's own cost-tiering
(the node's own row states cost="cheap", and CLAUDE.md's tiering table treats mechanical/build
stages as cheap-effort, not multi-CPU-hour compute). Running it here would also come close to
running (part of) the S0-B measurement itself, which builders are explicitly barred from running
(task constraint; row #290's own decisions table separates `b-pahier33-scorer` (build, cheap)
from `m-s0b-production` (measure, its own cost line) as distinct nodes).

**What WAS verified instead (g-byte-id on all non-S0-B default paths, satisfied):**
- The full pre-existing regression suite (`uv run pytest -m "not gpu and not slow"`) passes
  UNCHANGED: **2016 passed, 15 skipped, 30 deselected, 0 failed** (verbatim run below) — every
  b0i/ft default-path test, every existing driver test, every production-pipeline test is
  byte-identical in behaviour. No default-path CSV, posterior, or CLI default changed.
  `CONFIG_CHOICES`'s new "iiib" entry, the new `score_pahier33`/`es_null_arm` output keys, and the
  new `build_iiib_venue` function are all strictly additive — no existing code path, argument
  default, or dict key was removed or changed in meaning.
- `build_iiib_venue`'s WIRING was verified with unit tests using a tiny synthetic CRB CSV fixture
  and a sentinel catalogue handler (monkeypatching `c1d.check_crb_pin`, `c1d.CRB_CSV_PATH`,
  `c1d._load_galaxy_catalog_handler`): the function is proven to call the pinned loader with the
  pinned path verbatim, round-trip a CSV's contents unchanged, refuse a non-identity
  `sigma_z_scale`, and STOP (`RuntimeError`) on either pin mismatch — WITHOUT touching the real
  multi-GB production files. `_build_venue("iiib", ...)` dispatch and `_build_venue`'s
  unknown-config `ValueError` are both covered.
- `compute_es_null_arm`/`_seed_clustered_sem`/`compute_scores`'s `score_pahier33` block are
  verified against hand-derived arithmetic on small synthetic per-event arrays (exact formula
  match, `pytest.approx`), plus availability-gating tests (present with truth+s-axis; absent
  without truth even with a full s-axis; absent on runner-11's own b-only 8-cell node-dict shape
  — the SAME shape `rd-runner11`'s record documents as actually on disk).

**Open item, returned rather than silently skipped:** the full-scale g-byte-id identity run
against `c1d.BANKED_CSV_PATH` (N≥1e5 pairs, the real production data) has NOT been executed and
is the residual precondition before `m-s0b-production` should be trusted at full production
scale. This is not a scientific ambiguity (the wiring is unit-verified and the two loader/pin
functions are unmodified production code, already used identically by b0i/ft's own
`run_mirror_seed_inprocess` call), but it is a real, uncosted compute item that this "cheap"
build task is not the right place to spend — it should be run either as a cheap single-theta-
identity smoke step immediately before `m-s0b-production`'s own launch, or folded into that
measure node's own preflight, not charged to this build's cost line.

## Test results, verbatim

```
$ uv run pytest darksiren_emri_test/bayesian_inference/test_theta_zwindow.py darksiren_emri_test/test_theta_zwindow.py -q --no-cov
============================= test session starts ==============================
collected 54 items
darksiren_emri_test/bayesian_inference/test_theta_zwindow.py ........... [ 20%]
.....................                                                    [ 59%]
darksiren_emri_test/test_theta_zwindow.py ......................         [100%]
============================== 54 passed in 1.42s ==============================
```

```
$ uv run pytest -m "not gpu and not slow" -q
=== 2016 passed, 15 skipped, 30 deselected, 12 warnings in 135.61s (0:02:15) ===
Required test coverage of 25.0% reached. Total coverage: 73.43%
```

```
$ uv run ruff check results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py
All checks passed!
$ uv run ruff check darksiren_emri_test/bayesian_inference/test_theta_zwindow.py
All checks passed!
$ uv run ruff format --check <both files>
1 file already formatted / 1 file already formatted (after one auto-format pass each, applied)
$ uv run mypy darksiren_emri/ darksiren_emri_test/
Success: no issues found in 219 source files
```

(`uv run mypy` on the driver script directly reports 15 pre-existing `**kwargs`-unpacking
variance errors at three call sites inside `run_theta_node`'s config dispatch — lines 614 (b0i,
untouched by this build) and 630 (ft, untouched) carry the SAME error class as the new line 653
(iiib), confirming this is a structural mypy limitation of the existing `**cat_num_surv_2d_kwargs`
pattern, not a defect introduced here. `results/` is outside CLAUDE.md's mypy target
(`darksiren_emri/ darksiren_emri_test/`), so this does not gate `/check`.)

## Ambiguities in the ratified spec — none required routing back

Re-reading section 5's Rule block against the T1.4 falsifier's own closing note ("in EITHER case
`S_R` becomes the s-statistic of record for the arm ... a 5-node s-grid" — which reads as if
`score_lns_R` alone, needing no correction, becomes THE statistic): this is scoped to arms that
actually ran the 5-node Richardson grid (T1.4 only, so far — `score_lns_R` is pre-existing driver
machinery, unmodified here). S0-B's own registered design (PA-HIER-31(d)) has only a 3-node
s-grid (truth, s_plus, s_minus) and cannot compute `score_lns_R` at all, so it necessarily needs
the general `Es_null^{(arm)}` closed form (section 5's actual "Rule" block, lines 2971-2976) —
which is what was implemented. No ambiguity here required an author decision; this is recorded so
the reasoning is auditable, not because a choice was made on the author's behalf.

`IIIB_CATALOGUE_NUMERATOR_SURVIVAL` is pinned explicitly to `"phi"` even though
`run_metadata_21.json:cli_args` has no explicit key for it (left at the CLI default `"auto"`).
`"auto"` resolves to `"phi"` under `normalization_mode="absolute_marginal"` (the SAME resolution
the pre-existing `ft` config's own code comment documents) — pinning it explicitly is a
documentation-clarity choice matching PA-HIER-31(g)'s literal value, not a behaviour difference.

No other ambiguity was found in either registered spec that required routing back to the author.

*Stamp: b-pahier33-scorer, 2026-09-02. No `git commit` made (per task constraint — not
requested). No S0-B production measurement run. No `--jobs>1` used or enabled anywhere.*
