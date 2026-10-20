# /physics-change PRESENTATION GATE — [HIER] θ-hook (C1 + C2) — 2026-08-28

**Status: PRESENTED, awaiting author approval. No code has been written.**
Ordered by PA-HIER-28 (item 3 = GATE): the θ-hook edit to `bayesian_statistics.py` takes the
FULL /physics-change protocol — this presentation before code, a byte-identity regression test
at θ = (0,1), and ledger rows in `docs/gates/PHYSICS-GATE-LEDGER.md`. Authored at top tier per
the tiering table (recon: 1× sonnet; chair: orchestrator). Spec source of truth:
`PREREGISTRATION_HIER_HTHETA_20260826.md` (PA-HIER-21 site table, PA-HIER-28/-29/-30).

## 0. Scope of the change

One commit, contents fixed by the registration:

1. **C1 — the θ hook**: a `(theta_b, theta_s)` pair threaded into
   `BayesianStatistics.evaluate()`, reparametrizing the host-z kernel at estimator sites
   2.1 / 2.2 / 2.3 only, with a literal early-return/skip at `(0.0, 1.0)` (GATE T-ID).
2. **C2 — instrumentation**: per-term `ln L` diagnostics + a per-site toggle switch
   (PA-HIER-23 form) — no numerical change when toggles are off.
3. **Bundled, pre-authorized by the prereg**: the `correspondence_1d.py:1173` stale docstring
   fix (`:5908-5909` → `:6223-6224` / `:6878-6879`); the PA-HIER-30 free hardening
   `_B0I_ZTRUE_GRID_N` 401 → 4001; the PA-HIER-11 twin-parity regression (site 2.7).
4. **Explicitly NOT touched**: `correspondence_1d.host_z_error_eff` and every generator-side
   site (2.4 — GATE GEN-FROZEN, PA-HIER-2): θ must never move the data.

## 1. OLD formula (exact, at each in-scope site)

**Site 2.1 — scalar `single_host_likelihood`, `bayesian_statistics.py:6223-6224` (+ window
:6232-6244, kernel :6247):**

```python
sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
host_z_error_eff = float(np.sqrt(host_z_error**2 + sigma_z_pv**2))
# window: host_z ± integration_limit_sigma_multiplier * host_z_error_eff (lower-floored)
galaxy_redshift_normal_distribution = norm(loc=host_z, scale=host_z_error_eff)
```

**Site 2.2 — batch `single_host_likelihood_batch` (production's actual dispatch path),
`:6878-6879` (+ window :6888/:6905-6907):** identical formula, vectorized.

**Site 2.3 — global selection denominator `_smeared_global_pdet_expectation`,
`:1669-1672` (+ window/centre :1675-1679); called from
`precompute_global_catalog_selection` at `:4010/:4018/:4062`:**

```python
sigma_z_pv = (1.0 + z_g) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
sigma_eff = np.maximum(np.sqrt(z_err_g**2 + sigma_z_pv**2), 1e-10)
# per-slice: zc = z_g[sl]; lo = max(zc − 4σ, 1e-6); hi = max(zc + 4σ, lo + 1e-12); c = (hi+lo)/2
```

Site 2.3 today has **no `b` analog at all** (GATE D3 clause (c)) — the hook adds one.

## 2. NEW formula (the registered reparametrization)

θ = (b, s), applied at each in-scope site to the **outputs** of the existing width fold:

```
z̃      = host_z + b · (1 + host_z)          # kernel centre shift
σ̃_eff  = s · host_z_error_eff               # kernel width scale   (s > 0)
```

then every downstream consumer at that site uses (z̃, σ̃_eff) in place of
(host_z, host_z_error_eff): the Normal kernel becomes `norm(loc=z̃, scale=σ̃_eff)`, the
integration windows become `z̃ ± multiplier·σ̃_eff` (existing lower floors unchanged, applied
after the substitution), and at site 2.3 `zc → z̃_g`, `sigma_eff → max(s·sigma_eff_raw, 1e-10)`
(floor applied after scaling, keeping the :1670 delta-limit comment true).

**Pinned order-of-operations (registered here, decide-once):** `sigma_z_pv` is computed from
the RAW `host_z` — i.e. `b` shifts the centre AFTER the peculiar-velocity width fold, and `s`
scales the folded width. Rationale: θ models a systematic in the catalogue's *reported*
redshift; the PV dispersion attaches to the true sight-line. The alternative (folding at
`1 + z̃`) differs only at order `b·σ_pv`, and is numerically ZERO today because
`SIGMA_V_PEC_KM_S = 0.0` (the :1175 docstring's own disclosure) — moot but pinned so the
regression suite is well-defined if that constant is ever set.

**Early-return (GATE T-ID):** a single dispatch-level guard in `evaluate()` — when
`(theta_b, theta_s) == (0.0, 1.0)` the original unparametrized code path runs, giving
bit-for-bit identity by construction (not by floating-point luck of `x + 0.0` / `x * 1.0`).

**Threading (no formula content, listed for reviewability):** `evaluate()` (:3313, alongside
the existing instrument kwargs) → `p_D` (:4625) → `p_Di` (:4846) → `_starmap_host_batches`
(:7405) → `single_host_likelihood_batch`; and `evaluate()` → the three
`precompute_global_catalog_selection` calls. Defaults `theta_b=0.0, theta_s=1.0`.

## 3. Reference

The (bias, scatter) affine parametrization of a photometric-redshift error kernel,
`Δz = b·(1+z)` with a multiplicative scatter scale, is the standard photo-z nuisance model:

- **Ma, Hu & Huterer (2006), ApJ 636, 21, arXiv:astro-ph/0506614, §2** — photo-z systematics
  parametrized as a per-z bias and scatter of the Gaussian kernel; the (1+z) scaling of both
  is the field convention adopted there and since.
- DES Y1 / KiDS practice (e.g. Hoyle et al. 2018, arXiv:1708.01532) applies exactly the
  Δz-shift form to survey n(z) calibration.

This is an **instrument** (a nuisance reparametrization for the [HIER] profile-likelihood
stages), not a change to any physical law: at the production default it is the identity.

## 4. Dimensional analysis

All quantities dimensionless: `z`, `host_z_error_eff`, `sigma_z_pv` are redshifts;
`b` is dimensionless (Δz per unit (1+z)); `s` is a pure scale factor, `s > 0`.
`z̃ = z + b(1+z)` — redshift + (dimensionless)·(dimensionless) → redshift ✓.
`σ̃ = s·σ` — redshift ✓. Windows `z̃ ± m·σ̃` — redshift ✓. Registered supports keep the
kernel proper: `|b| ≤ b_max = 0.0661` (PA-HIER-29, 2× the measured catalogue median of
`z_err/(1+z)` = 0.033038) and `s ∈ [0.5, 2.0]` (log-uniform grid), so `σ̃ > 0` always and
the site-2.3 floor `1e-10` is never the binding constraint except where it already is.

## 5. Limiting cases

1. **Identity, θ = (0,1)**: z̃ = z, σ̃ = σ; with the literal early-return this is *exactly* the
   production path — the regression test asserts a bit-identical posterior on a fixed
   mini-evaluate (and the PA-HIER-11 twin-parity assertion: site 2.7's integration-testing
   twin still reproduces sites 2.1/2.2 at θ = (0,1)).
2. **s → 0⁺ (with the floor)**: the kernel tends to a delta at z̃; the site-2.3 smeared
   expectation collapses to the point evaluation at z̃ — exactly the behaviour the existing
   :1670-1671 comment documents for the σ→1e-10 limit, now centred at the shifted z̃. ✓
3. **b sign sanity**: b > 0 moves every host kernel to higher z at fixed data; a
   higher-z host explains the same GW distance with a larger H₀ trend in the single-host
   likelihood — the direction the [HIER] stage-F grid is designed to profile over
   (qualitative check only; no verdict rests on it).

## 6. Regression tests (written BEFORE the change, per protocol)

1. θ = (0,1) byte-identity: fixed-seed mini-evaluate posterior array `np.array_equal` old vs
   new (asserting the OLD value first, then unchanged after the edit).
2. Twin parity (PA-HIER-11): site 2.7 reproduces sites 2.1/2.2 at θ = (0,1); if the hook's
   refactor changes the shared expression's shape, the twin updates in the same commit.
3. Production-default test: `evaluate()` called WITHOUT θ kwargs takes the early-return branch
   (guard against a future default drift).
4. Grid hardening: `_B0I_ZTRUE_GRID_N = 4001` re-run of the PA-HIER-30 leg-(b) spot-check —
   the straddling-host width residual (~13-15% at 401 nodes) must close to ≤1%.

Post-implementation: sign/units/limit checklist, `# Eq./§2, Ma, Hu & Huterer (2006),
arXiv:astro-ph/0506614` reference comment above the reparametrization lines, `[PHYSICS]`
commit, three ledger rows in `docs/gates/PHYSICS-GATE-LEDGER.md`.

## 7. Decision table

| # | tag | decision |
|---|---|---|
| 1 | **[RULE]** | Approve this presentation (items 1-5 above), including the pinned order-of-operations (b after the PV fold) and the dispatch-level early-return form. Approval authorizes writing the code exactly as presented; any deviation returns here. |
| 2 | **[RULE]** | Confirm the bundle (docstring fix, 401→4001 hardening, twin-parity test) rides in the same `[PHYSICS]` commit, per PA-HIER-21/-30's own language. |
| 3 | **[DO]** | After merge + green regression suite: build the option-B quadrature grid (b from ±0.0661, s log-grid, H_GRID_41) and run S0-A — all verdicts capped REPORTED-ONLY (item 9 = AFFORDABLE). |

*Presented 2026-08-28. Every file:line above was re-verified at source during authoring.*

---

## Appended note 2026-08-29 — s-placement alignment gate (row #221 item 4; authorization row #223, charter node B6.1)

**Status: PRESENTED, awaiting author approval. No code has been written under this note.**
This is a gate presentation only (node B6.1 [ALIGN], "gate presentation BEFORE CODE"); it does
not itself authorize the edit. Row #221 item 4 ratified aligning the built θ-hook's `s`
placement to `[HIER]` §1.2's registered arithmetic
(`PREREGISTRATION_HIER_HTHETA_20260826.md:53-56`), which places `s` on the **raw catalogue
error, before** the peculiar-velocity quadrature fold — not on the folded width, as Sections 1
and 2 above (and the implemented code) currently do. **This note supersedes the "Pinned
order-of-operations" paragraph of Section 2 above and Decision-table item 1's approval of that
paragraph; neither is edited — both stand as the record of what was approved and built on
2026-08-28, and are superseded going forward by the form below.**

### 1. OLD formula (as implemented, verbatim with current line numbers)

**Site 2.1 — scalar `single_host_likelihood`, `bayesian_statistics.py:6370-6381`:**
```python
sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
host_z_error_eff = float(np.sqrt(host_z_error**2 + sigma_z_pv**2))
if theta_b != 0.0 or theta_s != 1.0:
    _validate_theta(theta_b, theta_s)
    _theta_hook_count("site_2_1")
    host_z = host_z + theta_b * (1.0 + host_z)
    host_z_error_eff = float(theta_s * host_z_error_eff)          # <- s AFTER the fold
```

**Site 2.2 — batch `single_host_likelihood_batch`, `bayesian_statistics.py:7041-7050`:**
```python
sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
host_z_error_eff = np.sqrt(host_z_error**2 + sigma_z_pv**2)
if theta_b != 0.0 or theta_s != 1.0:
    _validate_theta(theta_b, theta_s)
    _theta_hook_count("site_2_2")
    host_z = host_z + theta_b * (1.0 + host_z)
    host_z_error_eff = theta_s * host_z_error_eff                 # <- s AFTER the fold
```

**Site 2.3 — global selection denominator `_smeared_global_pdet_expectation`,
`bayesian_statistics.py:1692-1704`:**
```python
sigma_z_pv = (1.0 + z_g) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
sigma_eff = np.maximum(np.sqrt(z_err_g**2 + sigma_z_pv**2), 1e-10)
if theta_b != 0.0 or theta_s != 1.0:
    _validate_theta(theta_b, theta_s)
    _theta_hook_count("site_2_3")
    z_g = z_g + theta_b * (1.0 + z_g)
    sigma_eff = np.maximum(theta_s * sigma_eff, 1e-10)            # <- s AFTER the fold
```

(`SIGMA_V_PEC_KM_S = 0.0`, `darksiren_emri/constants.py:95` — reverified this session,
2026-08-29.)

### 2. NEW formula (registered s-placement, HIER §1.2)

`s` scales the **raw** catalogue-quoted redshift error, and the PV term is folded in
**afterward**, in quadrature, unscaled:

```
sigma_z_pv        = (1 + z̃) · SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
host_z_error_eff  = sqrt( (s · host_z_error_raw)**2 + sigma_z_pv**2 )
```

where `z̃ = host_z + b·(1 + host_z)` (the `b` placement is UNCHANGED by this note — `b` still
shifts the centre; only the fold ORDER for `s` moves). At site 2.3 the corresponding
substitution is `z̃_g = z_g + b·(1+z_g)`, `sigma_eff = max(sqrt((s·z_err_g)**2 + sigma_z_pv(z̃_g)**2), 1e-10)`,
with the 1e-10 floor re-applied after the combine (unchanged floor semantics, moved combine
step). `host_z_error_raw` at each site is the pre-hook local (`host_z_error` at 2.1/2.2,
`z_err_g` at 2.3) — the same input variable the OLD code fed into the fold, just read BEFORE
rather than after `sqrt(·**2 + sigma_z_pv**2)`.

### 3. Reference

- `PREREGISTRATION_HIER_HTHETA_20260826.md` §1.2 (lines 40-63): "Registered arithmetic form,
  pinned to remove the recon's §5/finding-9 ambiguity — `s` scales the *catalogue's quoted*
  error (the quantity whose misstatement is the hypothesis), never the peculiar-velocity term
  (a separate physical contribution the hypothesis says nothing about)."
- Row #221 item 4 (ledger row #221, 2026-08-29 charter): ratifies applying this alignment ahead
  of the fan-out wave.
- Ma, Hu & Huterer (2006), arXiv:astro-ph/0506614, §2 — unchanged as the general affine
  photo-z reference (§3 of the original presentation above); this note only relocates where `s`
  attaches relative to the PV quadrature, not the affine parametrization itself.

### 4. Dimensional analysis

Unchanged from Section 4 above: `host_z_error_raw` and `sigma_z_pv` are both redshifts, `s` is
a dimensionless scale factor (`s > 0`), so `s · host_z_error_raw` is a redshift; summing its
square with `sigma_z_pv**2` inside the root is quadrature of two redshift-valued quantities →
redshift ✓. Identical dimensional status to the OLD form — the two formulas differ only in
*which* redshift-valued term `s` multiplies, not in units.

### 5. Limiting cases

1. **σ_pv = 0 (today's value, `SIGMA_V_PEC_KM_S = 0.0`, verified above)**: `sigma_z_pv ≡ 0`
   identically in both the OLD and NEW forms, so `host_z_error_eff = s · host_z_error_raw` in
   both — **the two formulas coincide bit-for-bit at every θ**, not just at θ=(0,1). This is
   why the presented-2026-08-28 code is unaffected in production today; the alignment is a
   change to an as-yet-unobservable branch.
2. **s = 1 (any σ_pv)**: both formulas reduce to
   `host_z_error_eff = sqrt(host_z_error_raw**2 + sigma_z_pv**2)` — the un-hooked fold —
   identically, for any value of `SIGMA_V_PEC_KM_S`. `s` scaling is a genuine no-op at s=1
   regardless of placement, as it must be.
3. **σ_z,raw → 0 with σ_pv > 0 (the case that motivates the ordering, per HIER §1.2)**: NEW
   form → `host_z_error_eff → sigma_z_pv`, independent of `s` — a vanishing catalogue error
   still leaves the (unscaled) peculiar-velocity floor, because `s` never touches
   `sigma_z_pv`. OLD form → `host_z_error_eff → s · sigma_z_pv` (since at `host_z_error_raw=0`
   the pre-scaling `sqrt(0 + sigma_z_pv**2) = sigma_z_pv` and `s` then multiplies the whole
   folded width) — `s` incorrectly rescales a term the hypothesis makes no claim about. This is
   the substantive difference the alignment fixes, and it is exactly the case HIER §1.2 flags
   as the reason the arithmetic form was pinned ("stops being a no-op the moment that constant
   is ever set").

### 6. Regression plan

(a) **Existing pins unchanged.** All `test_theta_hook.py` byte-identity/closed-form pins
    (θ=(0,1) identity at all 3 sites; θ-engaged-vs-substituted-inputs at
    `SIGMA_V_PEC_KM_S == 0.0`) hold under the NEW form exactly as under the OLD — by limiting
    case 1 above, the two forms are bit-identical whenever `SIGMA_V_PEC_KM_S == 0.0`, which is
    every existing assertion's precondition (`test_theta_hook.py:120,145`, `assert
    bs.SIGMA_V_PEC_KM_S == 0.0`). No existing pin needs to change.

(b) **New test — nonzero-σ_pv discriminator, all three sites.** Monkeypatch the module
    attribute the code actually reads: `monkeypatch.setattr(bs, "SIGMA_V_PEC_KM_S", <value>)`
    on `darksiren_emri.bayesian_inference.bayesian_statistics` (module alias `bs` — this is the
    name bound by `from darksiren_emri.constants import SIGMA_V_PEC_KM_S` at
    `bayesian_statistics.py:41`; module-level globals are read at call time, so patching the
    module attribute is sufficient and is the pattern already used at
    `test_generator_marginal_mode.py:471` and `test_smear_global_selection.py:179` for this
    same constant). With a non-zero value (e.g. 200.0, matching the removed runtime-addition
    magnitude cited at `constants.py:90-94`) and `theta_s != 1.0`:
      - compute `host_z_error_eff` both ways in the test (OLD closed form: `s` post-fold; NEW
        closed form: `s` pre-fold) and assert they DIFFER (`rtol` guard, not
        `assert_allclose` — the point is divergence);
      - call each production site (`single_host_likelihood`, `single_host_likelihood_batch`,
        `_smeared_global_pdet_expectation`) with the patched constant and `theta_s != 1.0`, and
        assert the returned `host_z_error_eff`-driven quantity matches the NEW closed form
        (not the OLD) at `rtol=1e-12` — i.e. this test only goes green once the code is edited
        to implement Section 2 above.
      - one case per site (2.1 scalar, 2.2 batch, 2.3 smeared) = 3 new test functions minimum,
        landing in `test_theta_hook.py` alongside the existing suite.
    This is the regression test that currently does not exist and cannot pass against today's
    code — it is written here, before the edit, per protocol; the edit itself is a separate
    node.

Post-implementation (when this note's [RULE] is granted and the edit lands): sign/units/limit
checklist re-run against Section 4/5 above, a
`# HIER §1.2 s-placement (row #221 item 4) — s scales RAW host_z_error BEFORE the PV fold`
reference comment at each of the three edit sites, `[PHYSICS]` commit, ledger rows.

### 7. Decision table (this note)

| # | tag | decision |
|---|---|---|
| 1 | **[RULE]** | Approve this alignment (items 1-6 above): `s` moves to scale the raw catalogue `host_z_error` BEFORE the peculiar-velocity quadrature fold, at all three sites, superseding the OLD post-fold placement approved 2026-08-28. `b`'s placement is unchanged. |
| 2 | **[DO]** | After merge + the 3 new nonzero-σ_pv regression tests green (plus the unchanged pins): three ledger rows (`presented`/`implemented`/`verified`) in `docs/gates/PHYSICS-GATE-LEDGER.md`, per protocol. |

*Appended 2026-08-29 under rows #222/#223 (charter node B6.1). Every file:line above
re-verified at source this session (`bayesian_statistics.py` at HEAD `a794404c`;
`constants.py:95` confirms `SIGMA_V_PEC_KM_S = 0.0` still holds).*

---

## Implementation record 2026-08-29 — node B6.1 [ALIGN], IMPLEMENT

**Status: IMPLEMENTED (not committed — the orchestrator commits). Launched under rows
#222/#223, charter node B6.1.**

Code written at the three sites, matching the appended note's §2 formula in substance
(`host_z_error_eff = sqrt((theta_s * host_z_error_raw)**2 + sigma_z_pv**2)`), but with one
deliberate deviation from that section's literal formula text, called out here for the
record:

**Deviation, and why.** The note's own §2 formula literal reads
`sigma_z_pv = (1 + z̃) · SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S` — i.e. computed from the
POST-b-shift z̃ — while the same paragraph's prose states *"the `b` placement is UNCHANGED by
this note — `b` still shifts the centre; only the fold ORDER for `s` moves."* These two
statements disagree whenever `b ≠ 0` and `SIGMA_V_PEC_KM_S ≠ 0`: the literal formula silently
reverses the 2026-08-28 pin ("`b` shifts the centre AFTER the PV width fold" — i.e. `sigma_z_pv`
computed from the RAW, unshifted host redshift), while the prose says that pin is untouched.
Neither the note's three limiting cases (§5) nor its regression plan (§6) exercises
`theta_b ≠ 0` together with `SIGMA_V_PEC_KM_S ≠ 0` — the one regime where the two readings of
§2 diverge numerically — so the discrepancy would have shipped undetected by every test named
in the gate.

**Resolution taken:** implemented the PROSE ("`b` unchanged"), not the formula literal. At all
three sites, `sigma_z_pv` is computed from the host redshift local as it stood BEFORE the
`theta_b` shift is applied (same line, same order, as the 2026-08-28 code); only the
computation of `host_z_error_eff` (site 2.1/2.2) / `sigma_eff` (site 2.3) changes, to read the
RAW catalogue error before combining in quadrature with that (unshifted-z) `sigma_z_pv`. This
is: (a) consistent with the note's own stated intent and its Reference §3 quote ("`s` scales
the catalogue's quoted error... never the peculiar-velocity term — a separate physical
contribution the hypothesis says nothing about"), which is a claim about `s` only; (b)
numerically inert today exactly like every other reading, since `SIGMA_V_PEC_KM_S = 0.0`
(`constants.py:95`, reverified this session) makes `sigma_z_pv ≡ 0` under EITHER b-order; and
(c) the conservative choice — it changes nothing about `b`'s behavior beyond what was already
approved and built on 2026-08-28, so no new [RULE] gate is needed for the s-placement note to
land as scoped ("`b`'s placement is unchanged by this note").

**New regression coverage added specifically for this deviation:**
`test_theta_b_order_unchanged_uses_raw_host_z_for_pv` in `test_theta_hook.py` — patches
`SIGMA_V_PEC_KM_S` nonzero, engages `theta_b` alone (`theta_s = 1.0`), and asserts the
production kernel matches the RAW-host_z closed form for `sigma_z_pv`, not the z̃-based one.
This pins the resolution above so a future edit that reintroduces the z̃-based literal would
fail this test rather than ship silently.

**Files changed:**
- `darksiren_emri/bayesian_inference/bayesian_statistics.py` — sites 2.1 (:6370-6382), 2.2
  (:7041-7051), 2.3 (:1696-1706, line numbers approximate post-edit).
- `darksiren_emri_test/bayesian_inference/test_theta_hook.py` — module docstring updated;
  6 new test functions appended (3 required nonzero-σ_pv discriminators per §6(b) of this
  note, 1 discriminator-sanity check, 1 b-order regression pin, matching the "7." gate line
  added to the docstring).
- `docs/gates/PHYSICS-GATE-LEDGER.md` — `implemented` and `verified` rows appended.

**Test results:** `test_theta_hook.py` + `test_catalog_only_diagnostic.py`: 27 passed.
Full suite `uv run pytest -m "not gpu and not slow"`: **1851 passed, 15 skipped, 27
deselected**. `ruff check --fix` and `ruff format` clean on both changed files; `mypy` clean
on `bayesian_statistics.py`.

Full detail (line-by-line diff description, bit-identity evidence, exact files-to-commit
list) is in
`results/campaign51_20260728/realistic_20260729/fanout1_20260829/B6_1_ALIGN_RECORD.md`.

*Appended 2026-08-29 under rows #222/#223 — launched under rows #222/#223, charter node B6.1.*
