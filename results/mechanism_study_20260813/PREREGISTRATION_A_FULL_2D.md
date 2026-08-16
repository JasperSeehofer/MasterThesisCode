# PRE-REGISTRATION — A-FULL-2D: the fused-g_sel 2D estimator arm

**Date:** 2026-08-16 · **Authorized:** ledger row #115 item 2 ("all approved", author verbatim;
as-measured §3 form) · **Governing derivation:** `L6_DER2_CORRECT_FORM_2D_20260816.md` +
`L6_DER2_VERIFIER_ADDENDUM_20260816.md` (GO-with-amendments; V1/V2/V4 of record) ·
**Pre-measurement basis:** `L6_DER2_GSEL_PREMEASURE_output.json` (commit `fbc60b3a`) ·
**Status: REGISTERED at the commit that carries this file, the instrument variant, the scorer,
and the seed-block test.** A8-v2 discipline throughout; branches presented, never
self-adjudicated; append-only from the registering commit.

## 1. The question

Does the fused single-∫dM form — A-FULL's 1D channel untouched, its 2D channel replacing the
(S̄_φ(z) node-weight × coded `g`) pair by
`g_sel(z,f;h) = ∫dx_M N(x_M; μ_cond(f), σ_cond) · φ_x(x_M;z) · S(x_M·M_z_obs; z, h)` —
reproduce on **fresh seeds** what its mirror pre-measurement predicts: the 2D−1D excess
collapsed to the residual-class level (−11.74 ± 1.04 nats/h at 15 mirror seeds, 91.4% of
channel B removed), with the 1D channel bit-identical to A-FULL and the 2D-channel posterior's
coverage restored?

**Candidate form: as-measured (row #115).** The V2 measure prefactor (the D2-analogue
1/(σ_M·M(1+z)) question) is NOT added — it is measured immaterial in this venue (≲1e-6 rel,
σ_cond ~ 1e-7) and the arm must test exactly the form the verifier confirmed. V2 remains a live
question for the production derivation only.

## 2. The arm

| cell | variant | h_true | N seeds | seed offsets | dose |
|---|---|---|---|---|---|
| **AFULL2D** | `a_full_gsel` (ESTIMATOR_VARIANT_A_FULL_GSEL) | 0.730 | 25 | **+54300…+54324** (base `VT_BASE_SEED` = 20260808) | full (`dose_target="all"`) |

Everything else identical to the AFULL stage-5 arm: pinned 982 events, `balls="real_k"`,
`sigma_mode="glade"`, canonical 41-point grid, `n_events_cap=None`, `chunk_pairs=16384`, the four
standing pins (CRB CSV / frozeng emit / pruned catalogue / injection pool,
`PINNED_INPUTS_MANIFEST.md`). Seed block +54300…+54324 is fresh and disjoint from every
reserved/consumed block (+40000 decade v3, +43000/+44000/+45000, +50000…+52599, +53000, +54000,
+54100, +54200, W1/O2 envelopes); the disjointness unit test is extended to cover it in the
registering commit.

**Cell-spec registry entry (installed in the registering commit):**

```python
AFULL2D_CELL_SPECS: dict[str, VenueCellSpec] = {
    "AFULL2D": VenueCellSpec(
        "AFULL2D", "A-FULL-2D", "real_k", "glade", (0.730,), (25,), (54300,), "all",
        estimator_variant=ESTIMATOR_VARIANT_A_FULL_GSEL,
    ),
}
```

## 3. Code form

`ESTIMATOR_VARIANT_A_FULL_GSEL = "a_full_gsel"` in `venue_transfer._channel_terms_at_h`,
installed in the registering commit as a guarded addition (every other variant byte-identical to
its pre-existing path; the base byte-identity unit test must stay green):

- **1D channel:** byte-identical to `a_full` — kernel-branch integrand
  `kern · [N(d_obs; d_L, σ_d·d_L)/d_L] · w_pop(z;h) · S̄_φ(z;h) · (1/imp_k)`, point branch the
  same factors at z_obs, `−N ln α` retained. (The premeasure's c1 bit-identity gate is the
  reference; the installed variant must reproduce it.)
- **2D channel:** node weight `kern · [N(d_obs; d_L, σ_d·d_L)/d_L] · w_pop(z;h) · (1/imp_k)`
  (NO S̄_φ factor) times the fused `g_sel`, with:
  - `g_sel` the premeasure's `g_sel_mass_factor` form verbatim: `completion_mass_factor_g`'s
    conditional-Gaussian (μ_cond(f) = 1 + proj·(f−1), σ_cond from the 2×2 block) and φ_x
    measure conventions, **non-adaptive pinned n_hermite = 64** (registered convention — the
    Route-1 adaptive bound does not cover the sharp S(x_M) factor);
  - `S` the unmarginalized with-BH detection survival queried exactly as
    `precompute_phi_marginal_survival` queries it: detector-frame mass `M_z = x_M·M_z_obs_i`,
    the node's `d_L(z;h)`, isotropic sky (φ=θ=0), `_wbh_z_kwargs` pass-through.
- The LOO weight `_loo_impostor_weights` is the verifier-C1 construction verbatim, unchanged.

`venue_transfer.py` is a validation module, not a physics-trigger file; the production
`bayesian_statistics.py` is NOT touched by this arm (its `/physics-change` proposal is a
separate, later step per row #115 item 3).

## 4. Decision statistics (bands seeded from the mirror pre-measurement, N = 15 → arm N = 25)

Mirror basis (`L6_DER2_GSEL_PREMEASURE_output.json` per-seed rows): excess_gsel
−11.740 ± 1.038 (per-seed sd 4.019); T2_gsel +18.9 ± 42.9 (sd 165.98); T1 identical to AFULL.
Arm-mean SE at N = 25: excess 4.019/√25 = 0.804; T2 165.98/√25 = 33.2. Prediction-vs-arm
comparison sd: excess √(0.804² + 1.038²) = 1.313; T2 √(33.2² + 42.9²) = 54.2 nats/h.

- **DS-G1 (PRIMARY, branch-carrying): the paired 2D−1D excess at truth.**
  mean over seeds of T2 − T1 (grid-neighbour central difference at h_true, k = 20/22, raw
  `ln_post` vectors, never the aggregate block) ∈ **[−15.7, −7.8]** (= −11.74 ± 3×1.313,
  endpoints rounded conservative-wider). False-fail under the mirror hypothesis: 0.3%.
- **DS-G2 (secondary, non-branch-carrying): tilts at truth.** T(2D) ∈ [−143.8, +181.6]
  (= +18.9 ± 3×54.2) and T(1D) inside the stage-5 DS-F1 band [−131.5, +192.7]; both sanity
  reads on the same statistic class the stage-5 arm passed — reported, not branch-forcing
  (DS-G1 carries the excess claim with far higher power).
- **DS-G3 (2D coverage, branch-carrying jointly with DS-G1): binomial bands at nominal.**
  On the **2D-channel** posterior (`ln_post_2d`), RESTORED requires hpd50 ∈ [0.20, 0.80] AND
  hpd68 ∈ [0.40, 0.96] AND hpd90 ∈ [0.72, 1.00] (±3σ binomial at N = 25). Reference: the coded
  2D form's excess (+131.5) rails every prior arm's 2D channel.
- **DS-G4 (1D invariance, branch-carrying as STOP): c1 bit-identity.** The arm's per-seed
  `ln_post_1d` vectors must be bit-identical to what the `a_full` variant produces on the same
  seeds (verified on ≥2 arm seeds post-run, full 41-point grid). Any nonzero diff →
  STUDY-CONFOUNDED (the installed variant leaked into the 1D channel).
- **DS-G5 (specificity, descriptive):** 2D MAP bias (displacement-law context: T-band/Ā),
  per-seed T scatter, zero-rail/NaN counts. Any rail or non-finite event triggers §6 STOP.

## 5. Branches (presented to the author; none self-adjudicated)

1. **DS-G1 PASS + DS-G3 RESTORED** → the 2D channel has a validated correct-form estimator
   (M-OWNED-CLOSED candidate for the 2D thread); the production `/physics-change` proposal
   (row #115 item 3) gains this arm as its evidence base. [Author ruling.]
2. **DS-G1 PASS + DS-G3 NOT restored** → the excess is repaired but 2D width/curvature is not;
   the width channel becomes the 2D lead (the 1D thread's width analysis is the template).
3. **DS-G1 FAIL high (excess > −7.8)** → channel B under-cancelled on fresh pools; first
   suspects: the residual class is realization-coupled more strongly than the 15-seed sd
   captures (r = 0.847 structure), pool-vs-model prior mismatch.
4. **DS-G1 FAIL low (excess < −15.7)** → overcorrection beyond the residual class; first
   suspects: a genuine small missing term of systematic sign (the verifier's open question),
   the V2 prefactor, the S-query convention.
5. **OTHER / confounded** (DS-G4 fail, rails, non-finite, pin failure) → STUDY-CONFOUNDED; no
   branch forced.

## 6. Validity, execution-completeness, and STOP

1. **Pre-submission gate (executed before the registering commit; results recorded in §8):**
   the installed instrument variant must reproduce the mirror premeasure on seed 20310808 at
   k = 20 and k = 22 to |Δ ln2| < 1e-6 (and ln1 bit-identical to `a_full`); a failure blocks
   submission — no tuning, the discrepancy returns to the author.
2. Scorer `score_afull2d.py` pre-committed in the registering commit; mechanics dry-run against
   the committed stage-5 `AFULL_h0p730_results_seeds0_25.json` (schema-identical; 2D fields
   exercised) before submission.
3. Pins: `check_pin_integrity` must pass on the cluster before submission (`/cluster` preflight
   `VERDICT: READY ✓` required).
4. STOP: any seed with railed posterior, non-finite `ln_post`, or pin mismatch → hold, report,
   no re-run without an author ruling. **Budget ceiling 300 CPU-h** (expected ~180–220:
   premeasure realized ~0.2 CPU-h per seed×h-point for the g_sel-dominated pass — the
   non-adaptive n=64 Hermite×survival query is ~7× the stage-5 a_full node cost — × 25 seeds
   × 41 h-points; the workspace expires 2026-09-23 with 0 extensions, results must be
   retrieved promptly).
5. Expected NULLs, pre-registered: the −11.7-class residual is NOT explained by this arm (it is
   a stated residual regardless of outcome; its origin decomposition remains the open question
   of record); the pool-vs-model mismatch is NOT resolved; low-dose behavior is NOT probed
   (full dose only); nothing here adjudicates the production leg (A3 stands).

## 7. Provenance

Derivation `L6_DER2_CORRECT_FORM_2D_20260816.md` (committed `09c02c06` BEFORE the premeasure);
premeasure `fbc60b3a` (all 4 gates bit-exact); xhigh verifier addendum `453d1b29`
(GO-with-amendments, no refutations; independent S-query check 8.0e-16); author rulings rows
#112/#114/#115. Mirror validation chain: `l6_c2_switch_decomposition.py` (row #113) +
`l6_der2_gsel_premeasure.py`.

## 8. Pre-submission gate results

*(Recorded at the registering commit — appended by the implementation step, verified before
submission.)*

**Implementation base commit (working tree, not yet committed at recording time):** `e3eec5c0`.

### 8.1 Pre-submission gate (§6 item 1) — PASS

Installed `ESTIMATOR_VARIANT_A_FULL_GSEL` ("a_full_gsel") reproduced against the mirror
pre-measurement on seed 20310808 (`L6_DER2_GSEL_PREMEASURE_output.json`'s first row), full dose,
`k=20` (h=0.725) and `k=22` (h=0.735), via `results/mechanism_study_20260813/
gate_afull2d_premeasure_check.py` (calls `venue_transfer._channel_terms_at_h` directly at the two
h-indices rather than sweeping the full 41-point grid — ~20x cheaper for the same check, same
per-h body either way):

| check | max\|Δ\| | threshold | verdict |
|---|---|---|---|
| gate 1 — ln1: installed `a_full_gsel` vs installed `a_full` (same draw) | `0.000e+00` | bit-identical | PASS |
| gate 1b — ln1: installed `a_full_gsel` vs the premeasure's own `a_full` mirror | `0.000e+00` | bit-identical | PASS |
| gate 2 — ln2: installed `a_full_gsel` vs `L6_DER2_GSEL_PREMEASURE_output.json`'s `gsel` mirror | `0.000e+00` | `< 1e-6` | PASS |

Raw values (both k): `ln1 = {lo: -1492.7283459544997, hi: -1492.4283637535445}` (identical across
all three of installed-afull / installed-gsel / premeasure-mirror-afull); `ln2_gsel =
{lo: -2921.2299067721688, hi: -2921.054296443759}` (identical across installed-gsel and
premeasure-mirror-gsel). Full output:
`results/mechanism_study_20260813/gate_afull2d_premeasure_check_output.json`. No tuning was
applied — the exact port matched on the first run.

### 8.2 Scorer mechanics dry-run (§6 item 2) — mechanics verified

`results/mechanism_study_20260813/score_afull2d.py` run against the committed stage-5
`AFULL_h0p730_results_seeds0_25.json` (schema-identical; `ln_post_1d`/`ln_post_2d` both present
and exercised). All five DS blocks (DS-G1 through DS-G5) computed without error; branch
determination fired "3. DS-G1 FAIL high" on the a_full data, which is expected and uninformative
(the a_full_gsel bands are calibrated to the mirror pre-measurement, not to a_full — this is a
mechanics-only dry run per prereg §6 item 2, not a physics check). DS-G3 (2D coverage) read
RESTORED=True and DS-G5 read zero rails/non-finite on this input; both are properties of the
a_full data used for the dry run, not predictions about the eventual a_full_gsel arm output.
Output: `/tmp/.../score_afull2d_dryrun.json` (not committed; reproducible via `python
score_afull2d.py --input AFULL_h0p730_results_seeds0_25.json`).

### 8.3 Quality gate — PASS

- `uv run ruff check darksiren_emri/` — all checks passed.
- `uv run ruff format --check darksiren_emri/` — 68 files already formatted, no diffs.
- `uv run mypy darksiren_emri/` — success, no issues found in 68 source files.
- `uv run pytest -m "not gpu and not slow"` — 1472 passed, 15 skipped, 27 deselected (full
  pre-existing suite, including the new/extended `test_a_full_2d_estimator.py` and the
  seed-block disjointness extensions in `test_venue_transfer_arms.py` /
  `test_a_jren_stage3_arms.py` / `test_a_full_estimator.py`, plus the generic
  all-`ESTIMATOR_VARIANTS` h-grain forwarding test in `test_m2prime_ablation_arms.py`, whose
  fixture needed a non-`None` `detection` stand-in once `a_full_gsel` joined the loop — see
  implementation notes below). Zero regressions against the pre-change baseline (same command,
  same file set, verified before and after).

### 8.4 Implementation notes / spec resolutions

- **c1/c2 split inside `_channel_terms_at_h`.** The pre-existing per-h loop computed one
  `integ` array shared by both channels (`c1q = half*(integ@w_gl)`, `c2q =
  half*((integ*g)@w_gl)`). A-FULL-2D's 1D channel needs the S̄_φ-carrying `integ` (shared,
  byte-identical with `a_full`) while its 2D channel needs a DIFFERENT S̄_φ-free weight times
  `g_sel` instead of `g`. Resolved by naming the S̄_φ-free intermediate (`w_pop_z` /
  `w_pop_p`) inside the existing `a_full`/`a_full_gsel` branch (same float ops, same order —
  a refactor, not a behaviour change for `a_full`) and guarding the post-branch `c2q`
  construction on `estimator_variant == ESTIMATOR_VARIANT_A_FULL_GSEL`; every other variant's
  `else` arm is textually identical to the pre-existing unconditional code.
- **`force_S_one` not ported.** The premeasure's `g_sel_mass_factor` carries a
  `force_S_one` refactor-check parameter used only by its own `validate_S_equals_one` gate
  (a one-time verification the premeasure already ran and recorded). The installed
  `_g_sel_mass_factor` omits it — production callers always want the real survival term, and
  keeping a dead branch in the installed path would violate "verbatim port, no creativity" in
  the direction of adding unused surface. Flagged here for visibility, not treated as a
  deviation from prereg §3 (which specifies the fused object, not the premeasure script's own
  internal test scaffolding).
- **Two test-file `detection=None` fixtures needed a stand-in.**
  `test_m2prime_ablation_arms.py::test_estimator_variant_forwarded_through_hgrain_path` loops
  generically over every `vt.ESTIMATOR_VARIANTS` member; its shared context previously passed
  `detection=None` (fine for every variant through `a_full`, which only reads `s_phi_tables`).
  `a_full_gsel`'s `g_sel` queries the detection object directly, so this pre-existing generic
  test needed a minimal constant-survival stand-in (`_FakeDetection`, same class added locally
  to the new `test_a_full_2d_estimator.py`) — a guarded test-fixture fix, not a production code
  change; flagged since it touched a file this task's TASK list did not name explicitly (found
  via the failing-test loop, not anticipated in the port plan).
- **Cell-spec / registry wiring** followed the AFULL_CELL_SPECS precedent exactly: new
  `AFULL2D_PREREG_PATH` constant, `AFULL2D_CELL_SPECS` dict, union into `ALL_CELL_SPECS`,
  `preregistration_path_for_cell` branch, and CLI `--cell` choices. Four pre-existing test
  files' `ALL_CELL_SPECS`-union assertions (`test_venue_transfer_arms.py` x2,
  `test_a_jren_stage3_arms.py`, `test_a_full_estimator.py`) needed the new registry added to
  their expected-union expressions to stay green (mechanical, no behavioural claim changed).
  `test_m2prime_ablation_arms.py`'s equivalent assertion uses `<=` (subset), so it needed no
  change.

No spec ambiguity blocked implementation; the two items above (`force_S_one` omission, the
`detection=None` fixture fix) are the only judgment calls made beyond a literal port, and neither
touches a computed value the prereg specifies.

## 9. Pre-registration verifier report

*(Appended by the pre-registration verifier, 2026-08-16. Adversarial stance per the stage-5
precedent (`PREREGISTRATION_A_FULL_STAGE5.md` §8). §§1–8 untouched. The verifier verifies;
the author adjudicates.)*

**Verification base:** working tree on `e3eec5c0` (= HEAD at verification time), matching §8's
recorded implementation base. All numbers below are from fresh executions by the verifier, not
from the recorded files.

### 9.1 Per-item findings

1. **Guarded-addition claim — VERIFIED.** The full 449-line `venue_transfer.py` diff was read
   end-to-end. Every pre-existing variant's code path is textually identical after the §8.4
   refactor: the ball-branch rename (`w_pop_z` named, then `w_sel = w_pop_z * np.interp(...)`)
   and the point-branch rename (`p_gw_p_full`/`w_pop_p`/`s_phi_p` named, then the same
   left-to-right product) preserve the same float ops in the same order; the non-gsel `c2q` and
   point-branch `else` arms are byte-for-byte the pre-existing code. Empirical pin: the fresh
   gate run (item 4) shows installed `a_full` ln1 identical to the premeasure's mirror `afull`
   (which was itself validated bit-exact against the PRE-change installed `a_full` at
   `fbc60b3a`, recorded diff 0.0) — so pre-refactor ≡ post-refactor for the shared `integ`,
   which also pins `a_full`'s 2D channel (`c2 = half·((integ·g)@w_gl)`, unchanged else-arm).
   Targeted suites (58 tests across the five estimator-variant test files) and the FULL fast
   suite re-run by the verifier: **1472 passed, 15 skipped, 27 deselected** — exactly §8.3.
   `ruff check`/`ruff format --check`/`mypy` clean on the changed module.
2. **Port fidelity — VERIFIED.** Programmatic normalized diff (docstrings/comments stripped,
   `bs.` prefix and name normalization applied) of installed `_g_sel_mass_factor` vs the
   premeasure's `g_sel_mass_factor`: identical except (a) the documented `force_S_one` omission
   and (b) one cosmetic line-wrap of the `d_L_query` broadcast. `_g_sel_ball_capped`: identical
   except the same `force_S_one` threading. S-query convention (detector-frame
   `M_z = x_M·M_z_obs_i`, node `d_L(z;h)` absolute, isotropic zeros, `h=h`,
   `_wbh_z_kwargs` pass-through) matches line-for-line. n=64 is non-adaptive in the installed
   path (no `adaptive` branch exists in `_g_sel_mass_factor`).
3. **Band arithmetic — VERIFIED, independently recomputed** from the committed
   `L6_DER2_GSEL_PREMEASURE_output.json` per-seed rows (N = 15):
   `excess_gsel` mean = −11.739804, per-seed sd = 4.018744, SE = 1.037635; arm-mean SE at
   N = 25 = 0.803749; comparison sd = 1.312516; **exact DS-G1 band [−15.677353, −7.802254]**
   → registered [−15.7, −7.8] is conservative-wider at BOTH endpoints (by 0.023 / 0.002).
   `T2_gsel` mean = +18.901204, sd = 165.977346; comparison sd = 54.2080; **exact DS-G2 band
   [−143.722719, +181.525127]** → registered [−143.8, +181.6] conservative-wider at BOTH ends.
   False-fail under the mirror hypothesis: P(|Z|>3) = 0.27% ≈ the stated 0.3%. DS-G3 binomial
   arithmetic checks: 0.50±3·0.1000, 0.68±3·0.0933, 0.90±3·0.0600 (clipped at 1.00) reproduce
   the registered [0.20,0.80]/[0.40,0.96]/[0.72,1.00].
4. **Gate integrity — VERIFIED by fresh execution** (wall 25.9 min, exit 0): gate 1
   (installed gsel vs installed a_full ln1) max|Δ| = 0.000e+00 PASS; gate 1b (vs premeasure
   mirror afull) 0.000e+00 PASS; gate 2 (ln2 vs premeasure gsel mirror) 0.000e+00 PASS
   (threshold 1e-6); OVERALL PASS. The freshly written output JSON is **bit-identical** to the
   recorded `gate_afull2d_premeasure_check_output.json`. Cross-consistency with the committed
   premeasure basis: T1/T2/excess recomputed from §8.1's raw ln values (+29.998220 /
   +17.561033 / −12.437187 nats/h) match the committed JSON's seed-20310808 row to all printed
   digits.
5. **Scorer correctness — VERIFIED.** DS-G1 is computed from raw per-seed `ln_post_1d`/
   `ln_post_2d` vectors via `tilt_at_truth` (grid-neighbour central difference at
   `argmin|h−0.730|`, i.e. k = 20/22 on the canonical grid), paired over common seeds — the
   aggregate block and stored scalar fields are never read. DS-G3 runs on `ln_post_2d` with
   HPD-contains at 50/68/90. All hardcoded bands equal prereg §4 exactly. The verifier's own
   dry-run against `AFULL_h0p730_results_seeds0_25.json` reproduces §8.2 verbatim (DS-G1
   +135.7 FAIL-high → branch 3; DS-G3 RESTORED=True; 0 rails / 0 non-finite).
6. **Seed-block disjointness — VERIFIED.** `venue_cell_seeds` maps AFULL2D to absolute seeds
   20315108…20315132; the extended `test_seed_plan_disjointness_afull2d_vs_all_documented_blocks`
   enumerates v1/v2/v3 envelopes, reserved W1/O2, and every MECH/SCAN/M2P/REN/AFULL block and
   passes; no non-test repo code references the +54300 decade. The 15 consumed mirror seeds
   (20310808–22, MN0X +50000 block) are disjoint from the arm block.
7. **Registration hygiene — VERIFIED with one caveat (m2 below).** Provenance ancestry
   confirmed in git: `09c02c06` (derivation) → `fbc60b3a` (premeasure basis) → `453d1b29`
   (verifier addendum) → `91c813df` (ledger row #115, author verbatim "all approved") →
   `e3eec5c0` (implementation base = HEAD). Row #115 item 2 authorizes exactly the as-measured
   §3 form registered here. No tuning surface found in the installed code: `chunk_pairs` pinned
   by the cell spec, `node_chunk` a pre-existing module constant, `n_hermite` carries no CLI or
   cell-spec override (see m1), the AFULL2D spec is field-identical to AFULL apart from
   name/offset/variant. Budget: 0.2 CPU-h × 25 × 41 = 205 CPU-h reproduces the stated 180–220
   expectation under the 300 ceiling; the most pessimistic rate derivable from the premeasure's
   realized pool cost (66 min × 8 workers / 30 seed·k ≈ 0.29 CPU-h per seed·k for the FULL
   3-config pass, an overestimate of the arm's single-config cost) extrapolates to ≈300 CPU-h —
   the ceiling holds even under that bound, but with no margin (m4).

### 9.2 Findings, severity-classified

- **MAJOR-1 — DS-G3 has weak discriminating power against the relevant null, and §4's
  reference sentence overstates.** The verifier recomputed DS-G3-style 2D coverage on the two
  committed comparator datasets: MN0X (coded base form) reads 0.000/0.000/0.000 — but the
  stage-5 AFULL arm (the direct comparator: same weights, coded `g`, excess +135.7) reads
  0.360/0.480/0.760 with 0/25 railed, i.e. **already RESTORED under the registered bands**.
  §4's justification ("the coded 2D form's excess (+131.5) rails every prior arm's 2D
  channel") is contradicted by the stage-5 arm (0/25 MAP-railed). Consequence: a RESTORED
  reading in this arm carries little evidence beyond what a_full's 2D channel already shows,
  so DS-G3 cannot separate branch 1 from branch 2's premise on its own, and the branch-1
  label ("validated correct-form estimator") should not lean on DS-G3. DS-G1 — the primary —
  is unaffected (its band fully separates −11.7 from +135.8). **Remedy:** at adjudication,
  treat DS-G3 as necessary-but-weak (a non-restoration remains informative; a restoration is
  expected under both branch 1 and the a_full-coded-g null); this §9 entry is the standing
  record of that caveat. No spec change required pre-submission.
- **MINOR-1 (m1) — the n = 64 pin is indirect.** `_g_sel_ball_capped` reads
  `gctx.cl_ctx.config.n_hermite`, whose value is the `ClosedLoopConfig` default
  `_G_I_HERMITE_NODES = 64`; nothing in the venue CLI or cell spec can override it today, but
  the "registered convention" silently tracks that default rather than a literal 64 in the
  variant. (The premeasure did the same, so mirror/instrument stay consistent.) Post-arm
  cleanup candidate; frozen as-is for the registered code form.
- **MINOR-2 (m2) — §§1–7/§8 append-only integrity is not yet git-evidenceable.** The prereg
  file is untracked at verification time (its first commit IS the registering commit), so
  "§8 append-only, §§1–7 unmodified" cannot be diffed against history. Mitigation verified:
  every §4 number is mechanically derivable from the committed `fbc60b3a` basis (item 3), so
  §§1–7 could not have been tuned to anything not already committed. From the registering
  commit onward the append-only discipline becomes checkable.
- **MINOR-3 (m3) — §8.1's gate-2 description is imprecise.** It says the installed ln2 was
  checked "vs `L6_DER2_GSEL_PREMEASURE_output.json`'s `gsel` mirror", but the gate script
  recomputes the mirror live (the JSON stores per-seed T's, not raw ln values). The check is
  nonetheless anchored to the committed basis: the verifier confirmed the gate's raw ln values
  reproduce the committed JSON's seed-20310808 T-row exactly (item 4).
- **MINOR-4 (m4) — thin worst-case budget margin.** Central estimate 205 CPU-h is fine; the
  most pessimistic premeasure-derived rate lands at ≈300 CPU-h — exactly the ceiling. The
  gate's observed single-seed cost suggests the true arm rate is well below that bound, but
  slower cluster cores would eat the margin. Chunked submission with per-seed retrieval (the
  standing venue pattern) makes a ceiling-hit recoverable rather than fatal.
- **Note (no severity):** DS-G4's post-run plan requires `a_full` re-runs on ≥2 arm seeds
  (full 41-point grid) — a small additional cost outside the arm itself; schedule it inside
  the same workspace-expiry window.

### 9.3 Verdict

**GO-with-remedies** for the registering commit + cluster submission. Zero CRITICAL. One
MAJOR (MAJOR-1), whose remedy is interpretive and standing-recorded above — it does not
require any change to the registered spec, instrument, scorer, or bands before submission;
it binds the branch-1-vs-2 adjudication. Four MINORs, none blocking. Gate, port, bands,
scorer, seed plan, and provenance chain all independently reproduced by the verifier.
