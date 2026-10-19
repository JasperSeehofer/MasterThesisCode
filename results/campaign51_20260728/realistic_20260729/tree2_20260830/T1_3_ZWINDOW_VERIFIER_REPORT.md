# T1.3-zwin independent verifier report (2026-08-30)

Row #255 standing grant, tree 2 node T1.3-zwin. Verifier: independent of the presenter and the
builder (per the node's own separation-of-roles requirement). Branch fix/p32d-classg-venue-repair.
No git operations by this node; no ssh; foreground only. Did not touch the concurrently-written
files (BIAS_HISTORY_LEDGER.md, the T2_3_* tree2_20260830 files, the mass-aware gate doc).

Read in full: PHYSICS_CHANGE_THETA_ZWINDOW_20260830.md (sections 0-2, section 3 derivation/
calibration table, section 7 regression plan, the Implementation-prerequisites note, and the
builder's own append-only Implementation record); T1_3_ZWINDOW_IMPLEMENTATION_RECORD.md in full;
PREREGISTRATION_HIER_HTHETA_20260826.md's PA-HIER-32(d) block (lines 2687-2806, the primary source
for the registered score_s form, re-derived independently rather than trusted from the gate doc's
quotes). Diffed: darksiren_emri/{galaxy_catalogue/handler.py, bayesian_inference/
bayesian_statistics.py, arguments.py, main.py, validation/correspondence_1d.py},
darksiren_emri_test/{test_theta_zwindow.py, bayesian_inference/test_theta_zwindow.py},
results/.../fanout1_20260829/hier_s0_driver.py, docs/gates/PHYSICS-GATE-LEDGER.md.

## Summary table

| # | Item | Verdict | Notes |
|---|------|---------|-------|
| 1 | Scope + formula match (handler.py z-filter, divisor untouched, comment reword) | **PASS with 1 MUST_FIX** | z-filter mask matches section 2.1/2.2 formula-for-formula (theta_zwindow="off"/"on", literal skip at (0,1), z_window_k guard site per Revision note 1). Divisor (`precompute_phi_divisor_theta_ratio`) is NOT touched by the diff — matches section 2.4's "NO". Site-2.1 comment reworded per Revision note 2's builder instruction. **But** the driver's new `_es_null_det_closed_form` computes `Es_null_det_i` using the wrong secant denominator — see MUST_FIX below. |
| 2 | Byte-identity at "off" + on-vs-off smoke at s=√2 | **PASS** | Full R1-R9 regression suite (34/34, both new test files) green. Direct real-catalog check (20.8M-row production `reduced_galaxy_catalog`, 62 s, cheaper substitute for the full-driver smoke — see below): candidate set at "off" k=1 and "on" at theta=(0,1) k=1 are the **identical set** (27785/27785 catalog indices, exact match) — GATE T-ID confirmed on real data. At "on", s=√2, k=1 the set is **31247** (differs, as engagement requires); at the registered decisive k=4 the counts continue to move (off k=4: 38010; on s=√2 k=4: 38746). |
| 3 | Driver scorer (PA-HIER-32(d)) | **FAIL (1 MUST_FIX)** | `score_b`/`score_s_raw` exactly reproduce the old banked `hier_s0_recert_run` numbers bit-for-bit via `--score-only` (see below); `score_lns` correctly uses the PA-HIER-4 ln(2) denominator (ratio-verified against `score_s_raw`); `score_s` gracefully reports unavailable (NaN/n_pooled=0) on this cache-less legacy run, never silently substituting an uncorrected number. **But** `_es_null_det_closed_form`'s internal secant uses denominator `sqrt(2) - 1/sqrt(2)` (the *raw* secant's own denominator) instead of `ln(2)` (`score_lns`'s denominator) — PA-HIER-32(d) (prereg lines 2748-2754) defines `Es_null_det_i` explicitly as "the closed-form expectation of `score_lns_i`", not of `score_s_raw`. Reproduced independently: the code's own output is `denom_s/denom_lns` = 1.02014× too large in magnitude vs. the registered definition. This breaks the "E[score_s \| generator kernel = estimator kernel] = 0 by construction" bias-free guarantee that is PA-HIER-32(d)'s entire point (a small but non-zero residual bias, ≈0.0009, survives at the null — about 7% of score_s's SEM). No test catches this: the only direct test of the closed form (`test_compute_es_null_det_closed_form_matches_delta_limit`) checks only sign and coarse magnitude, and the arithmetic test of `compute_scores` (`test_compute_scores_score_s_corrected_subtracts_es_null_det`) plants `es_null_det` values directly rather than deriving them from the closed form, so it cannot see the internal denominator error. |
| 4 | Tests/ruff/mypy counts | **PASS with 1 unresolved discrepancy** | `ruff check`/`ruff format --check` clean (reproduced); `mypy darksiren_emri/` clean (reproduced). New test files: 22 (handler) + 12 (bayesian_inference, after the addendum) = 34, reproduced exactly (pytest: `34 passed`) — the gate doc's own embedded Implementation-record text says "(9, ...)" for the bayesian_inference file, which is stale (predates the addendum's 2 tests); the ledger's own "implemented (addendum)" row correctly states 12. Full suite (`pytest -m "not gpu and not slow" darksiren_emri_test/`, single run, 87.9 s): **1975 passed, 15 skipped, 30 deselected, 0 failed** — zero regressions, skip count matches the claimed "15" exactly. The claimed totals (923+1106=2029 passed / 21+15=36 deselected, or the chat summary's "926+1106") do **not** reproduce on this CPU-only dev sandbox (no `cupy` installed); most plausible explanation is environment difference (the `xp` fixture parametrizes over `cupy` when available, which would inflate collected/passed counts on a GPU-capable box) rather than a functional regression, since every test that DID run passed. Flagged as unresolved, not as a failure. |
| 5 | Gate-ledger rows match the diff | **PASS on content, MINOR finding on line citations** | Exactly 3 new rows appended (`implemented`, `verified`, `implemented (addendum)`), matching the 3-line diff on `PHYSICS-GATE-LEDGER.md` and the implementation record's own file list. Content (what changed, why, test/ruff/mypy results, scope) matches the actual diff. **But** the `bayesian_statistics.py` line citations in the `implemented`/`verified` rows drift substantially from the actual final line numbers — increasingly so later in the file: evaluate() kwargs cited `:3945-3970` vs actual `:3978-3987`; the guard cited `:4256-4269` vs actual `:4280-4282`; the call site cited `:5646-5658` vs actual `:5690-5712`; the site-2.1 comment cited `:7391-7397` vs actual `:7454-7458` (63 lines off). `handler.py`'s own citations are accurate (within 1-2 lines). This is a traceability/documentation defect, not a functional one — the same class of staleness the presentation doc's own Revision notes 1-3 diagnosed and fixed for its *own* citations, recurring here uncaught in the builder's ledger rows. |

## MUST_FIX

1. **`_es_null_det_closed_form`'s secant uses the wrong denominator** (`results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py`, the `secs = (...) / denom_s` line inside `_es_null_det_closed_form`, where `denom_s = _SQRT2 - 1.0/_SQRT2`). PA-HIER-32(d) (`PREREGISTRATION_HIER_HTHETA_20260826.md` lines 2748-2758, quoted verbatim: "Es_null_det_i = the closed-form expectation of score_lns_i under host i's OWN generator kernel...") requires the SAME secant form `compute_scores`'s own `score_lns` uses — numerator `(s_plus - s_minus)`, denominator `ln(2)`. The implementation instead reuses `score_s_raw`'s denominator (`sqrt(2) - 1/sqrt(2)`), which the function's own docstring even states outright: "i.e. exactly what `score_s_raw` computes for a single host's likelihood" — directly contradicting the registered target statistic. Reproduced independently on a synthetic single-host fixture (flat completeness, h=0.73): the code returns `Es_null_det=0.03396`; recomputed with the registered `ln(2)` denominator the value is `0.03465` — a factor of `1.02014` (`= (sqrt(2)-1/sqrt(2))/ln(2)`), matching the code's own constants exactly. **Fix**: replace `denom_s` with `math.log(2.0)` in that one line (the window-selection logic, `window_minus`, and everything else in the function is unaffected — those use s=1/√2's window width directly, not the secant denominator). No test currently catches this; a fix should add a test asserting `_es_null_det_closed_form`'s ratio-to-`score_lns`'s-own-numerator matches `1/ln(2)` exactly (analogous to the existing `test_compute_scores_score_lns_and_raw_use_the_registered_denominators` but for the closed form itself, not just the pooling arithmetic).

## Evidence detail

### Item 1 — formula-by-formula comparison

`handler.py`'s `redshift_filter_mask` (diff, `get_possible_hosts_from_ball_tree`):
```
_z_window_k = float(z_window_k)                      # guard: finite and > 0, ValueError else
if theta_zwindow == "off":
    _z_g_theta, _sigma_g_theta = _z_g, _sigma_g
elif theta_zwindow == "on":
    if theta_b != 0.0 or theta_s != 1.0:
        _sigma_pv_g = (1.0 + _z_g) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
        _sigma_g_theta = sqrt((theta_s * _sigma_g) ** 2 + _sigma_pv_g ** 2)
        _z_g_theta = _z_g + theta_b * (1.0 + _z_g)
    else:
        _z_g_theta, _sigma_g_theta = _z_g, _sigma_g      # literal skip, GATE T-ID
else:
    raise ValueError(...)
redshift_filter_mask = (z_min <= _z_g_theta + _z_window_k * _sigma_g_theta) & (
    z_max >= _z_g_theta - _z_window_k * _sigma_g_theta
)
```
This is bit-for-bit the section 2.1 registered form (`accept g iff z_min <= z_g^theta + k sigma_g^theta and z_max
>= z_g^theta - k sigma_g^theta`), including the literal-skip at theta=(0,1) (section 2.2's "R2"), the guard site
(handler.py, per Revision note 1's corrected file attribution), and theta reaching the handler only when
`theta_zwindow="on"` (`bayesian_statistics.py:5704-5705`: `theta_b=(self._theta_b if self._theta_zwindow == "on"
else 0.0)`). The divisor (`precompute_phi_divisor_theta_ratio`, T1.1's own function) has zero touched lines in the
diff — confirmed via `git diff | grep -c divisor` (comment-only hits) — matching section 2.4's "NO, the divisor
form stands unchanged".

### Item 2 — real-catalog candidate-set check (substituted for the full-driver smoke)

The registered full-driver smoke (`--smoke --nodes truth,s_plus --theta-phi-divisor on`) was attempted first; after
~300 s it had not completed even one node (the T1.1 divisor precompute pass is a fixed per-seed cost independent of
`--event-cap`, per the gate doc's own section 6 cost table, so a 12-event smoke does not shrink it) — not cheap
within the ≤600 s budget. Substituted with a direct, cheaper (62 s) check against the SAME production
`GalaxyCatalogueHandler.reduced_galaxy_catalog` (20,834,171 rows) that the driver would build, calling
`get_possible_hosts_from_ball_tree` directly at a real catalogue redshift (z≈0.2) with a realistic sky-cone/z-window:

| config | n candidates |
|---|---|
| off, k=1 | 27785 |
| on, theta=(0,1), k=1 | 27785 (**identical set**, not just count) |
| on, theta=(0,√2), k=1 | 31247 |
| off, k=4 | 38010 |
| on, theta=(0,√2), k=4 | 38746 |

This directly confirms GATE T-ID (off ≡ on-at-identity, bit-for-bit set equality) and engagement (on-at-s=√2 ≠
identity) on real production data, which is what item 2 asks for; script and script output available on request
(not banked as a registered measurement — builder-only smoke discipline, rule 2).

### Item 3 — driver scorer re-derivation and `--score-only` cross-check

Ran (read-only, no `evaluate()` call, zero compute):
```
uv run python3 .../hier_s0_driver.py --arm S0-A --nodes truth,b_plus,b_minus,s_plus,s_minus \
  --theta-sites 2.2 --smear off --theta-phi-divisor on --score-only \
  --out-root .../tree2_20260830/hier_s0_recert_run
```
against the existing (pre-T1.3-zwin) banked `hier_s0_recert_run` (no `es_null_det.csv` cache present). Result
(`ln_L_no_bh` channel):
- `score_b`: mean=-0.288782409603726, sem=0.427052520948283, Z=-0.676222233655185 — **bit-identical** to the old
  banked `s0a_score.md`'s `score_b` line.
- `score_s_raw`: mean=-0.0719595839365958 — **bit-identical** to the old banked `s0a_score.md`'s `score_s` line
  (the pre-rework statistic, correctly renamed/preserved for continuity).
- `score_lns`: mean=-0.0734088101344141 — ratio to `score_s_raw` is 1.020139..., matching
  `(sqrt(2)-1/sqrt(2))/ln(2)` exactly, confirming `score_lns` itself uses the correct PA-HIER-4 `ln(2)` denominator.
- `score_s`: NaN, n_pooled=0, `score_s_available=False` — correct graceful degradation (this banked run predates the
  `es_null_det.csv` cache; per the implementation's own design, `score_s` is never silently substituted).

This confirms the pooling/reporting machinery (`compute_scores`, the per-axis gating fix, `write_score_markdown`)
is correct and reproduces the old `Z_b` exactly, and that both `score_s` forms are printed as required. The
formula defect found is entirely inside `_es_null_det_closed_form` (see MUST_FIX) and is invisible on this
particular cross-check only because this banked run has no cache to exercise it — it WILL be exercised the moment
the registered P1 arm runs (which always computes+caches `Es_null_det` fresh per seed).

Ancillary (not a checklist item, noted for completeness): `verdict_s0a` reads `scores["ln_L_no_bh"]["score_s"]["Z"]`
directly; when `score_s` is unavailable (as above) this is NaN, and the verdict collapses to
`"INSTRUMENT-DEFECT -- STOP"` (`np.isfinite` guard fails) — indistinguishable in the verdict string from an actual
confirmed defect. This is disclosed by the implementer ("verdict_s0a/verdict_s0r ... needed no further change") and
is harmless for the registered P1 arm itself (which will always have the cache), but a future `--score-only` read
against an older/cache-less run should check `score_s_available` before trusting the verdict string.

### Item 4 — reproduced counts

```
uv run ruff check darksiren_emri/                     -> All checks passed!
uv run ruff format --check darksiren_emri/ ... driver  -> 73 files already formatted
uv run mypy darksiren_emri/                            -> Success: no issues found in 70 source files
uv run pytest darksiren_emri_test/test_theta_zwindow.py darksiren_emri_test/bayesian_inference/test_theta_zwindow.py
    -> 34 passed
uv run pytest -m "not gpu and not slow" darksiren_emri_test/   (single run, 87.9s)
    -> 1975 passed, 15 skipped, 30 deselected, 0 failed
```
See the summary table for the discrepancy against the claimed full-suite totals.

### Item 5 — ledger row line-citation drift

Spot-checked `bayesian_statistics.py` citations in the two new content-bearing ledger rows against `grep -n`
against the actual working tree:

| cited (ledger) | actual | drift |
|---|---|---|
| `:3659-3671`/`:3765-3768` (class/`__init__` defaults) | `:3667`, `:3768-3769` | ~0-4 lines, fine |
| `:3945-3970` (evaluate() kwargs) | `:3978-3987` | ~30-40 lines |
| `:4256-4269` (guard) | `:4280-4282` | ~15-24 lines |
| `:5646-5658` (call site) | `:5690-5712` | ~45-55 lines |
| `:7391-7397` (site-2.1 comment) | `:7454-7458` | ~60-65 lines |

`handler.py`'s citations (`:558-573`, `:714-761`) match the actual signature/mask block within 1-2 lines. The
drift is consistent with the ledger row's `bayesian_statistics.py` citations having been taken from an intermediate
edit state rather than re-verified against the diff's final form — the same class of defect the presentation doc's
own three revision notes caught and fixed for its citations, recurring here uncaught.

## Not re-verified (out of this node's scope / would require the actual P1 run)

- Any numeric agreement between `_es_null_det_closed_form` and the archived `f4_mechanism.py`/`f4_out.json` figures
  on the real production catalogue (both the builder and this verifier note this as unrun; doing so would also
  have surfaced the denominator bug immediately via a ~2% mismatch).
- The P1/P1b/P2/P3 arms themselves (no `evaluate()` call was made against the real GLADE catalogue for a full
  theta-node by this verifier either, beyond the direct candidate-set check above).
