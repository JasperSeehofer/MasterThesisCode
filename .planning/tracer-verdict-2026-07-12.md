# Sim/Eval Convention-Divergence Tracer — ADVISORY Verdict (2026-07-12)

> **THIS VERDICT IS ADVISORY. It does NOT greenlight the Phase-2 campaign.**
> Per [[orbiter-upgrade-design]] C.6 anchor discipline (§3.4): pre-PASS the tracer
> and its refuter both run in the *weakest* anchor tier (same-family fresh context).
> The load-bearing gate is **anchor-1 — explicit human (Jasper) ratification of this
> verdict before any production campaign fires.** Read the summary + refuter dissent
> (below), then ratify, reject, or send back for domain review.
>
> **Manifest incompleteness is in force.** This trace covers the 8 rows in
> `CONVENTIONS-MANIFEST.md` (skeleton) only. A convention-bearing quantity NOT in
> that manifest is invisible to this trace. A false "all consistent" over an
> incomplete manifest is the exact W-CONF-13 failure mode; the verdict is scoped
> accordingly and the refuter pass (mandatory) is included.

- **Runner**: Claude Code Task-tool advisory tracer (single fresh context; no cross-family panel — hence anchor-1, not anchor-2)
- **Method**: read the real pipeline code end-to-end (injection → storage → p_det grid → inference) for each manifest quantity; classify CONSISTENT / DIVERGENCE / UNKNOWN; then run a refuter pass attempting to falsify every CONSISTENT verdict.
- **Safety**: read-only. No sim/inference code modified, no campaign/cluster job touched (jobs `5698617`/`5698618` untouched), nothing committed.
- **Code state read**: local working tree, HEAD `6581d45` (2026-07-12). NOTE — the cluster campaign repo is PINNED at `b233375` per `CAMPAIGN-PREP-PHASE2.md` §4c; this trace reflects LOCAL code, which may lead the cluster. Divergence between local and cluster HEAD is itself an unverified risk (see "Could not verify").

---

## 0. Jasper's ratification & feedback (2026-07-12)

Reviewed and ratified (anchor-1). Dispositions on the three residuals:

1. **pp_coverage depth (Q7 / C-003):** NOT a decision-to-make — **both scenarios (0.95 vs 1.5) are under active exploration, evidence being collected before the final setup is chosen.** So this residual is by-design-open, not a blocker. Finding C-003 updated to WATCH-under-exploration.
2. **Missing paired invariant test on the `"M"`=M_z injection column (refuter's key finding):** accepted — **"good catch, should be implemented."** Tracked as new coverage finding **C-MTC-20260712-004 (APPROVED FOR IMPLEMENTATION)** so a future MTC session picks it up as the standing-floor half of C-001's owner.
3. **The two "in-writing" residuals (cluster pool-depth gate armed; manifest blind to unlisted classes):** explained to Jasper in plain terms (a runtime seatbelt the code has but this read-only trace can't confirm is buckled on the actual cluster run; and that this insurance only covers the ~8 listed quantities, so a divergence in an unlisted quantity — Fisher/CRB covariance, population weights, completeness m_th, photo-z model — would slip through until the full manifest is built).

**Net after ratification:** no live divergence on the traced classes; the one flagged live risk (HOST_DRAW_Z_MAX) was already fixed; the CI-gap is now a tracked, approved action. Submission is not gated on a single unanswered question — it proceeds with the declared, understood residuals above.

## 1. Headline for Jasper (the ≤1-page read)

**On the 4 incident-seeded convention classes + the flagged HOST_DRAW_Z_MAX item, this
trace found NO live divergence in the current local code.** All four historical bugs are
in a **fixed, mutually-consistent state**, and the `HOST_DRAW_Z_MAX = 0.5` staleness
flagged on 2026-07-02 has **already been resolved to `1.5` (fix #20, `b52ff8d`)** with a
**hard `raise ValueError` stale-pool gate** protecting the storage→inference boundary.

**But three things keep this from being a clean "safe-to-submit":**
1. **The consistency of the depth chain is *conditional on the campaign regenerating the
   injection pool at z≈1.5*** — it is hard-gated (fails loud, not silent), but the gate
   only fires at runtime on the cluster, which this trace cannot exercise.
2. **One genuine UNKNOWN needs your domain input**: the `pp_coverage` calibration harness
   ceiling (`Z_MAX_POP = 0.95`) is shallower than the campaign depth (`1.5`), and SCV's own
   2026-07-11 findings show the estimator's bias is depth/σ_z-dependent. Does per-seed
   `pp_coverage` run at campaign depth or at the hardcoded 0.95?
3. **The manifest is a skeleton.** Classes outside the 8 rows (Fisher/CRB covariance
   scaling, prior/population weights, completeness `m_th`, photo-z error model) were **not
   traced** and have caused adjacent bias work as recently as 2026-07-11.

**Recommendation: `needs-human-domain-review` (one narrow question) → then `safe-to-submit`
on the traced classes.** Not `fix-first` — no divergence to fix was found. Not an unqualified
`safe-to-submit` — the pp_coverage-depth UNKNOWN and the manifest incompleteness are real and
un-closeable by code-reading alone. Details in §4.

---

## 2. Per-quantity verdict table

| # | Quantity | End-to-end trace (inject → store → p_det → infer) | Verdict |
|---|----------|----------------------------------------------------|---------|
| M1 | Sky-angle frame `qS`/`phiS` | Injection ecliptic (`ResponseWrapper is_ecliptic_latitude=False`); catalogue equatorial ICRS on disk → **one** in-place rotation to ecliptic at load (COORD-03, `handler.py:251`); CRB CSV ecliptic; inference reads ecliptic; host BallTree ecliptic. Single rotation, everything downstream ecliptic. FRAME-AUDIT.md: 4/4 load-bearing claims CONFIRMED. | **CONSISTENT** |
| M2 | BH mass `M` (source vs `M_z`) | Injection lifts `M_z=M·(1+z)` once (`main.py:899`), stores `M_z` to CSV `"M"` (`:983`); FEW saw `M_z`. p_det grid mass axis = observer-frame `M_z` (built from injection `"M"`). Inference: rate-weight uses source-frame `host.M` (matches the draw); selection query lifts `M_z_g=M_g·(1+z_g)` (`bayesian_statistics.py:768`) → **grid axis and query are both observer-frame `M_z`.** This is the Design-B (`0099ce2`) + H3 (`f01595c`) fixed state; `Detection.M` docstring now truthfully says `M_z`. | **CONSISTENT** |
| M3 | `L_cat` likelihood form | `weighted_ratio_of_sums` = `(Σ_g w·N_g)/(Σ_g w·D_g)`, Gray Eq. A.9/A.10 (`bayesian_statistics.py:212-260`); constant-weight limit = plain ratio of sums. Not mean-of-ratios. Post-`816f904`. | **CONSISTENT** |
| M4 | `p_det` placement | Numerator `single_host_likelihood` carries **no** `p_det`; `p_det` enters **only** the denominator `D(h)=β_G+β_Ḡ` (`precompute_completion_denominator`), `p_i=(β_G·L_cat+B_num)/D(h)`. p_det itself is the exact detection-horizon survival `P(d_hor≥d_L)`, `d_hor=SNR·d_L/thr` — h-invariant, built once. Post-`341ca62`/W-PRE-12. | **CONSISTENT** |
| M5 | `HOST_DRAW_Z_MAX` depth | `1.5` uniform: `constants.py:99`; `cosmological_model.max_redshift=1.5` with assert `HOST_DRAW_Z_MAX ≤ max_redshift` (`:189`); `GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT=1.55`; injection `z_cut=HOST_DRAW_Z_MAX` (`main.py:825`); p_det `expected_z_max=HOST_DRAW_Z_MAX` (`posterior_combination.py:583`). **Hard `raise ValueError`** on shallow pool (`pool_z_max<0.9·1.5`) or mixed-`z_cut` provenance (`simulation_detection_probability.py:290-322`). The flagged "0.5 horizon-stale" item is **resolved** (fix #20, `b52ff8d`). | **CONSISTENT — conditional** (on campaign pool regen at z≈1.5; hard-gated, runtime-verified only) |
| B1 | Redshift frame `z_cmb` | Catalogue uses `z_cmb` (CMB-frame, PV-corrected, col 28) fed to `d_L(z,h)` & `M_z`; residual PV marginalized into host-z kernel (issue #16). Injection z is cosmological. In-code consistent. | **CONSISTENT — recent migration (WATCH)** |
| B2 | SNR threshold | `SNR_THRESHOLD=20` uniform: injection detection, horizon denominator, CRB filter. | **CONSISTENT** |
| B3 | Distance unit `d_L` | Gpc uniform (`physical_relations`, injection CSV, CRB CSV, `d_hor`). | **CONSISTENT** |
| Q7 | `pp_coverage` population ceiling vs campaign depth | Validation harness hardcodes `Z_MAX_POP=0.95` and its own `D50_GPC=1.85`; does **not** read production constants; campaign runs at `1.5`. SCV 2026-07-11 (N-4/σ_z): estimator bias is depth- and σ_z/z-dependent. Whether per-seed `pp_coverage` is reconfigured to campaign depth could not be established by reading code. | **UNKNOWN — needs domain input** |

---

## 3. REFUTER pass (mandatory — try to prove each CONSISTENT wrong)

Per W-CONF-13, the tracer's own synthesis can be confidently wrong. Strongest dissent per verdict:

- **M2 (mass) refuter — strongest overall dissent.** "CONSISTENT" rests on the injection CSV
  `"M"` column actually holding `M_z`. The lift and the store are two *different* code sites
  (`main.py:899` computes it; `:983` writes it) — the W-PRE-12 lesson is that a transform
  applied to multiple outputs must be invariant-checked on *every* output, and the original
  2026-06-20 bug was exactly a second write site (injection CSV) that stored source-frame `M`
  while the CRB path was guarded. I read the write as `"M": redshifted_M`, which is correct —
  **but I did not find a test that asserts the injection CSV column is `M_z` (only the CRB path
  is guarded by `test_parameter_space_h`).** If a future edit reverts the CSV write, no test
  fails. Residual risk: **the paired "every-output invariant" test (manifest M2) is NOT present
  in code** — the consistency is real *today* but unguarded. Also: injection truncates `M_z >
  M.upper_limit` (`main.py:908`); at inference a catalog host with large `M_g` and `z_g~1` can
  query `M_z_g` beyond the grid's populated mass axis → kernel extrapolation at the mass edge
  (an "M_z edge clamp" exists, `3273fa5`, but edge behaviour under the survival estimator was
  not independently probed here).

- **M5 (depth) refuter.** "CONSISTENT" is *conditional*, and the condition is the dangerous
  part: the p_det survival grid is only valid to the depth of the injection pool it loads. If
  the campaign submits inference against a p_det grid built from a **pre-#20 (z≤0.5) pool**, the
  survival tops out <1 Gpc and `p_det=0` for essentially all deep hosts — "silently valid-looking
  garbage" (the code's own words, `:285`). The mitigation is a **hard ValueError**, which is
  strong — but (a) it only fires at runtime on the cluster, which I cannot exercise; (b) the
  `allow_shallow_pool` escape hatch exists (`posterior_combination.py:590`) and a
  frozen-baseline re-eval threads it — if a campaign run inherits `allow_shallow_pool=True` the
  gate is bypassed. **I could not verify the campaign's actual pool depth or that
  `allow_shallow_pool` is False for the production run.**

- **M1 (frame) refuter.** FRAME-AUDIT.md is dated to COORD-03 (2026-04-22); the `z_cmb`
  catalogue migration (2026-07-02) rewrote catalogue columns. The rotation reads raw cols 8/9
  (RA/Dec) and the migration touched the *redshift* column (27→28), so the rotation input is
  unchanged — **but** the campaign-prep explicitly requires "8-col schema confirmation" before
  submit, and stale-schema catalogue backups exist in the tree (`*.stale6col_mar28`,
  `*.zhelio_20260702`). If the on-disk campaign CSV has a shifted column layout, the rotation
  would silently operate on the wrong columns. **I read the code path, not the actual campaign
  CSV header** — schema/provenance is the classic HPC-3-layer gap.

- **M3 / M4 refuter.** These are structural (which form / where p_det appears) and read cleanly
  in the current code. The residual is historical recurrence risk: both were reintroduced once
  by a *misreading of Gray's prose* (SCV: the equations were dropped as images). The code now
  cites Eq. A.9/A.10 explicitly. Low residual risk, but the guard is a comment + one equivalence
  test, not an invariant that would survive a confident re-misreading.

- **B1 refuter.** The z_cmb migration is very recent (2026-07-02) and the PV-marginalization
  (issue #16) landed 2026-07-03 — both inside the pre-campaign window. Recency is itself risk:
  the fixes are less battle-tested than the M1–M4 fixes. Consistent in-code, but least-aged.

**Refuter's bottom line:** the trace found no *active* divergence, but every "CONSISTENT" on the
two most recently-touched rows (M2 store-site, M5 pool depth) is **guarded by runtime gates or
comments rather than by a paired invariant test in CI** — precisely the manifest-M2/M4 "paired
test" column that reads `NONE FOUND` / partial. The consistency is a property of the current
code, not a property the pipeline *enforces on itself*. That is the honest gap.

---

## 4. Overall advisory recommendation

**`needs-human-domain-review` (one narrow question), resolving to `safe-to-submit` on the
traced classes once answered.**

- **Not `fix-first`**: no live convention divergence was found on any of the 4 incident classes
  or the HOST_DRAW_Z_MAX item. There is nothing to fix on the traced boundary.
- **Not unqualified `safe-to-submit`**, for three reasons that code-reading cannot close:
  1. **[decision needed] pp_coverage depth (Q7 / finding C-MTC-20260712-003)** — confirm the
     per-seed `pp_coverage` calibration runs at the campaign depth (1.5 / campaign σ_z), not the
     hardcoded `Z_MAX_POP=0.95`. If it runs at 0.95, the 4b#3 calibration gate validates a
     shallower venue than production and (per SCV 2026-07-11) may miss a depth-dependent residual.
     This is the single question to answer before submit.
  2. **[operational, hard-gated] injection-pool depth (M5)** — ensure the campaign p_det grid is
     built from a **freshly regenerated z≈1.5 pool** and `allow_shallow_pool` is False for
     production. If a stale pool sneaks in, the pipeline fails loud (ValueError), so this is
     low-risk *given the gate*, but verify the gate is armed on the cluster run.
  3. **[declared, un-closeable] manifest incompleteness** — the trace is blind to convention
     classes outside the 8 skeleton rows. Adjacent bias work (Fisher-frame/population, deep-
     incompleteness floor, σ_z/z shallow venue) is live as of 2026-07-11 and is NOT a convention-
     divergence of the traced kind, but it means "no divergence found" ≠ "no bias." The full
     manifest (2–3d archaeology) is the durable fix and remains a named separate task.

**What ratifying this verdict means**: you accept that the 4 documented divergence classes +
HOST_DRAW_Z_MAX are consistent in the current local code, that the pp_coverage-depth question is
answered (or accepted) before submit, and that the residual risk is (a) unenforced-by-CI
consistency on the two newest rows and (b) manifest-incomplete coverage — both stated in writing
here rather than discovered after a retired campaign.

---

## 5. What I could and could not verify (honesty ledger)

**Verified by reading code (local HEAD `6581d45`):**
- Frame rotation single-point + ecliptic-everywhere (M1), cross-checked against FRAME-AUDIT.md.
- `M_z` lift-once-at-injection + store site + p_det grid axis + inference query alignment (M2).
- Ratio-of-sums L_cat form (M3); p_det-in-denominator-only structure (M4).
- HOST_DRAW_Z_MAX=1.5 uniformity across 5 code sites + the hard stale-pool ValueError gate (M5).
- z_cmb / PV-marginalization / SNR-threshold=20 / Gpc-units consistency (B1–B3).

**Could NOT verify (out of read-only, single-pass, no-cluster scope):**
- The actual on-disk campaign injection-pool depth and its `z_cut`/`code_rev` provenance columns
  (runtime cluster artifact; the gate that checks them fires only on the cluster).
- Whether `allow_shallow_pool` is False on the production run.
- The campaign catalogue CSV column schema (the "8-col confirmation" the campaign-prep requires).
- Whether cluster HEAD (`b233375`, pinned) matches this local trace (`6581d45`).
- The pp_coverage runtime depth configuration (Q7) — hardcoded ceiling read, runtime value not.
- **Anything outside the 8 manifest rows** — Fisher/CRB covariance scaling conventions, prior/
  population-weight conventions, completeness `m_th` magnitude system, photo-z error model. These
  are the full-manifest gap, declared, not traced.
- **Physics correctness** — this tracer verifies *convention consistency across the boundary*,
  not that the likelihood/selection physics is correct. A convention can be consistently applied
  and still physically wrong (that is a different audit; the live 2026-07-11 floor/shallow-venue
  work is in that separate space).

---

*Filed 2026-07-12 as the advisory run for finding C-MTC-20260712-001 (COVERAGE.md). Pending
Jasper's ratification (anchor-1) before Phase-2 submission — open decision 5,
[[orbiter-upgrade-design]] Part 12.*
