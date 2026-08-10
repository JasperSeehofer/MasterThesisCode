# Pre-registration v2 — calibration-gate: P–P/coverage leg + multi-candidate host balls + σ–d_L joint texture

Registered 2026-08-10, **BEFORE** any v2 run. This file re-registers the
stage-4 calibration gate after the v1 campaign
(`results/calibration_gate_20260808/`, prereg commit `b50ccc65`, campaign +
readout + adjudication committed `3a572897`) fired **GATE-NOT-TRUSTWORTHY on
its own validity checks** — the correct, honest outcome of a working
falsification design. v1's science content is preserved as **raw patterns
barred from claims** (readout §7); v2 repairs the five enumerated defects and
runs a **fresh registered sample on disjoint seeds**. v2 is a
re-registration, not a re-score of v1 data.

**Authority.** The author autonomy mandate is active: every v2 design choice
below is flagged **AUTHOR-RATIFY** (the §2 deviation register lists them all
in one place) and is subject to post-hoc author ratification; **the final
gate verdict is the author's**, never self-adjudicated (v1's model/effort
policy, carried verbatim).

**Instrument identity.** The instrument
`master_thesis_code/validation/calibration_gate.py` and its test file
`master_thesis_code_test/validation/test_calibration_gate.py` are committed
**in the same commit as this file** (instrument edits + tests + prereg,
atomically) — the v1 provenance gap (instrument untracked at run time, v1
prereg §11 empty) is structurally impossible to repeat: the registering
commit *is* the code identity. Every v2 instrument edit is documented in the
module docstring divergence log, items 11–17.

**REGISTERED — append-only discipline is in force from this commit.** Every
band below is fixed at this commit and may not be adjusted after any
readout. Nothing above §11 may be edited after this commit; later material
is appended to §11 with dated headings.

Parents: v1 prereg
`results/calibration_gate_20260808/PREREGISTRATION_CALIBRATION_GATE.md`
(commit `b50ccc65`); v1 readout `CALIBRATION_GATE_READOUT_20260808.md` and
adjudication `adjudicate_readout_results.json` (both committed `3a572897`);
the registered closed-loop instrument (`77b524af`); `docs/RESEARCH_CYCLE.md`
§Stage 4 amendment A3. The author value ruling of record is unchanged: the
gate exists for **correctness + insight, not bias-removal**; REPORT-BOUND
remains a fully legitimate outcome.

---

## 0. Binding constraints of record (carried from v1, updated)

- `master_thesis_code/validation/closed_loop_gfrac.py` is **NOT modified**
  (registered instrument, code identity `77b524af`).
- `master_thesis_code/validation/pp_coverage.py` is **NOT modified**.
- No production physics file is touched — any estimator fix this gate
  motivates routes through `/physics-change`, never through this
  registration.
- v2 instrument code lives in exactly the two v1 files (edited for the five
  defect repairs, committed with this prereg): the module and its test file.
  All v2 outputs, the v2 readout script, and this registration live under
  `results/calibration_gate_v2_20260810/`. v1 artifacts under
  `results/calibration_gate_20260808/` are **untouched** and remain the
  committed v1 record.
- Local CPU only; no cluster jobs.
- **No production posterior is produced.** Every posterior emitted is a
  synthetic-universe diagnostic, quotable only against its own truth.

## 1. Questions of record

**Q1–Q3** are carried verbatim from v1 §1 (gate construction / A3
completion; the 1D-rail adjudication; σ–d_L texture-closure). They were
**not answered** by v1: the fired branch voided the measurement layer.

**Q4 (new — reproduction).** Do the three pre-named v1 raw patterns
(§7 DS-8: the A-1D starvation rail; the ball-venue uniform +σ_z bias with
collapsed coverage; B0 exactly on truth) **reproduce on disjoint seeds** in
a trustworthy instrument? Confirmation converts v1's barred patterns into
quotable measured properties; refutation is a first-class finding about
seed-sensitivity of the v1 campaign.

## 2. v1 → v2 DEVIATION REGISTER (each: what changed, why, which defect, ratification flag)

Nothing else changed. Cells, truths, N=400 (V1: 50), λ_ball = 4,
σ_z doses {0, 0.010, 0.035}, f_incl = 1, `dl_binned` texture, N_det = 1500,
`numerator_pdet = off`, injection pool, CRB CSV, quadrature orders, DS-1…DS-6
bands, the §8 edge guard, V1/V2/V3/V5, abort criteria — all identical to v1.

### D1 — V4 texture band re-derived from the pre-declared analysis [repairs defect 1] — AUTHOR-RATIFY

- **v1**: band `0.82 ± 0.10 = [0.72, 0.92]`, mis-set against the module's own
  **pre-declared** build-time decile-attenuation analysis (docstring
  divergence 7: rank-matching attenuates the CSV's measured 0.816 to
  **≈ 0.69 ± 0.02**, 20-replica SD). v1 measured 0.664–0.666
  (adjudicator-confirmed) and V4 fired.
- **v2**: band derived **FROM the pre-declared prediction and its stated
  uncertainty**: centre = 0.69 (the predicted attenuated value), half-width
  = 3 × 0.02 (three replica-SDs, covering replica scatter plus
  detected-set-restriction attenuation of the same order) ⇒
  **V4 band = [0.63, 0.75]**.
- The v1 measured 0.664–0.666 is cited **only as post-hoc consistency** (it
  lies inside the derived band); it is **not** the source of the band — the
  derivation above uses no number produced by the v1 campaign, only the
  build-time analysis that pre-dated it.
- Instrument: `_V4_CORR_CENTER = 0.69`, `_V4_CORR_TOL = 0.06`
  (divergence-log item 11).

### D2 — DS-7 demoted to REPORT-ONLY in both forms [repairs defect 2] — AUTHOR-RATIFY

- **v1**: the registered raw form violated 6/9 cells and fired the trigger
  set; adjudication showed it is **MC-seed-fragile** (8/9 violations under a
  different p_bar MC seed — the 0.05 band edge sits inside the p_bar MC
  noise at these ratios); the granularity-corrected form passed 9/9. The
  raw-vs-corrected form choice was reserved to the author (module divergence
  9) and was never exercised.
- **v2**: DS-7 is **REPORT-ONLY in both forms**. Both ratios are emitted per
  cell (now with `p_bar_mc_se`, quantifying the fragility in the record);
  **neither form carries V-class or branch weight**; DS-7 is removed from
  the §10 trigger set. **The author call on which form (if either) should
  carry weight in a future version is recorded as OPEN.**
- Instrument: `ds7_accounting` emits `status: "REPORT-ONLY"` +
  `p_bar_mc_se` (divergence-log item 12).

### D3 — registered degenerate-PIT exemption for B0/V1 [repairs defect 3] — AUTHOR-RATIFY

- **v1**: at σ_z = 0 the ball posterior is near-delta at truth ⇒ PIT ≡ 0.5,
  C_β ≡ 1, KS D ≡ 0.5 **by construction**; DS-1/DS-2 labels are
  structurally meaningless there, but v1 registered no exemption, so B0
  carried mechanical FAIL labels that mean nothing.
- **v2 registered exemption**: **B0 and V1 are plumbing/validity controls
  scored ONLY on their V-checks and on DS-3/DS-4 — never on DS-1/DS-2, in
  any form.** B0's role in DS-6 (the low anchor, via DS-4 `R_low`) is
  unchanged. The instrument emits `ds1_ds2_degenerate_pit_exempt = true` for
  ball cells at σ_z = 0 (divergence-log item 13). A/B1/B2 are unaffected
  (A is single-host — its own v1 §5 1D exemption is carried; B1/B2 have
  σ_z > 0).

### D4 — A cells move to the extended 75-point h grid [repairs defect 4] — AUTHOR-RATIFY

- **v1**: A-2D was **91–93 % edge-loaded at every truth** (0.690/0.730/0.770
  on the 0.600–0.860 grid) ⇒ the §8 guard correctly stripped all A-2D
  DS-1/DS-2 weight.
- **Choice between the two registered repair options** (more-distal truths
  vs extended grid): more-distal truths are **arithmetically impossible** on
  the 0.26-wide canonical grid — from v1's committed A-2D numbers
  (bias +0.020…+0.045, MAP sd ≈ 0.060–0.072, posterior sd ≈ 0.049–0.052,
  all truncation-*compressed* lower bounds), the clearance needed per side is
  ≈ bias + 3·map_sd + 2.33·post_sd ≈ 0.04 + 0.21 + 0.12 ≈ 0.31 > 0.26 total.
  Hence: **extended h grid for A cells**.
- **The grid**: `EXTENDED_H_GRID_A` = canonical 41 points **plus** 0.01-
  spaced wings 0.460–0.590 (14 points) and 0.870–1.060 (20 points) = **75
  points, 0.460–1.060**. The canonical grid is a strict subgrid, so the
  v1-comparable **restricted read** (argmax/PIT over the 41 canonical nodes)
  is a mechanical readout-side operation on the stored 75-point `ln_post`
  vectors. Wing extents from the same v1 numbers: high side needs
  ≥ mean-MAP + 2σ tail + 2.33·post_sd ≈ 0.79 + 0.12 + 0.12 ≈ 1.03 (+ margin
  for truncation compression ⇒ 1.06); low side symmetric about the smallest
  truth ⇒ 0.46. Predicted residual edge-load (pre-declared, honest): ~2–5 %
  of A-2D seeds may still be edge-loaded per truth — under the 10 % guard;
  if the guard fires anyway that is an honest EDGE-CONTAMINATED outcome,
  not a repair failure.
- **B0/B1/B2/V1 keep the canonical 41-point grid unchanged** — required for
  the DS-8 pattern-reproduction bands (D7), which quote v1 numbers measured
  on that grid. A-cell PIT/HPD/edge-mass are computed on the full 75-point
  grid (the decision read); DS-8 target T1 uses the restricted read.
- Instrument: `EXTENDED_H_GRID_A`, `CellSpec.h_grid` (divergence-log
  item 14).

### D5 — clean rule made precise and enforceable [repairs defect 5] — AUTHOR-RATIFY

- **v1**: §10 said "runs that would execute on a dirty tree STOP instead";
  every registered cell nevertheless ran `--allow-dirty` on a dirty tree
  with the instrument itself **untracked** (no code identity).
- **v2 rule (exact):**
  1. **Import path** = everything under `master_thesis_code/` and
     `master_thesis_code_test/`. A registered cell run **REFUSES to start**
     (hard `SystemExit`, before any context is built) if the import path has
     **ANY uncommitted change — modified OR untracked**.
  2. `--allow-dirty` is accepted **only together with `--smoke` or
     `--validate`** — the CLI rejects it outright on a registered cell run.
     There is no other escape.
  3. Dirt **outside** the import path (doc edits, untracked results
     directories, rule files, …) never blocks, and the **full inventory**
     (verbatim `git status --porcelain` lines, split
     `{import_path: [...], other: [...]}`) is embedded in **every output
     JSON** together with `git_commit`, `git_dirty` (whole tree) and
     `import_path_clean`.
- Rationale for scoping to the import path: the remaining tree dirt at
  registration time is unrelated to the instrument (doc edits, untracked
  results dirs); what provenance requires is that **the code that executes
  is exactly the committed code** — that is the import path, inventoried
  dirt covers the rest.
- Instrument: `_classify_porcelain`, `_git_state`,
  `_enforce_clean_import_path`, CLI gating (divergence-log item 15).
  Refusal + rejection + inventory are unit-tested and were exercised live
  before this commit (§11).

### D6 — disjoint seed plan (fresh registered sample) — AUTHOR-RATIFY

- **v2 offsets = v1 offsets + 20000** from the same base 20260808 (§5
  table). v1's absolute envelope is `20260808 + [0, 9049]`; every v2 seed
  lies in `20260808 + [20000, 29049]` — **disjoint by construction**, unit-
  tested. v2 is a fresh sample of the same generative process, which is what
  makes DS-8 (D7) a legitimate out-of-sample reproduction test.

### D7 — NEW decision statistic DS-8: v1 pattern-reproduction targets — AUTHOR-RATIFY

- The three v1 raw patterns are promoted to **pre-named reproduction
  targets** with bands derived from **v1's committed numbers + standard
  binomial/SE arithmetic** (legitimate: the v1 numbers exist and are
  committed at `3a572897`; the v2 seeds are disjoint). Full definition and
  arithmetic in §7 DS-8.

### D8 — smoke convention registered as what the instrument does — AUTHOR-RATIFY

- v1 §5 said "10 seeds/cell" smoke while the instrument's `--smoke` runs 3
  seeds (v1 minor discrepancy list). **v2 registers the 3-seed smoke** (with
  the built-in V3 spot-check re-run); the registered 10-seed variant remains
  reachable via `--n-seeds 10` but is not required.

## 3. THE INSTRUMENT (unchanged architecture)

The v1 §3 architecture table is carried verbatim: `closed_loop_gfrac`
imported as a library (A3-i, A3-ii inherited); `hpd_contains` certified port
of `pp_coverage._hpd_contains` (V2); ball generative model per v1 §4.2;
`dl_binned` texture per v1 §4.3. The "what the instrument deliberately does
NOT do" list (v1 §3) is carried unchanged — z-window Poisson caricature, no
GLADE/n(z)/sky/completeness, redshift-only ball members, bare kernel,
`f_incl = 1`. The module docstring divergence log (items 1–10 = v1 build;
items 11–17 = v2 repairs) is the code-side record of every deviation.

## 4. Generative model and estimator (unchanged)

v1 §4, §4.1, §4.2, §4.3 are carried **verbatim** — generator, ball model,
estimator quadrature, PIT/HPD/edge-mass readout, texture rank-matching. The
only change is the A-cell evaluation grid (D4): for A cells every per-h
quantity (windows, tables, α(h)) is evaluated on the 75-point grid by the
identical code path; PIT for A is `q = ∫_{0.460}^{h_true} P(h) dh`. Ball
cells are bit-compatible with v1 (proven: the pre-commit smoke reproduced
v1's committed B2/V1 smoke per-seed records **bit-identically**, §11).

## 5. Cell matrix, seed plan, N floor, runtime budget

N floor, truths, per-cell configs: identical to v1 §5 (400 seeds per truth
per cell; V1: 50; truths T = {0.690, 0.730, 0.770}; registered 300-seed
fallback with v1's locked fallback bands).

| cell | config | truths × seeds | grid | v2 seed blocks (base 20260808) |
|---|---|---|---|---|
| **R0** | retro-read of the committed registered closed-loop run (anchor-only, zero compute, no gate weight) | 0.73 × 200 | canonical 41 | n/a (committed data) |
| **A** | single-host, f = 0, `dl_binned` | T × 400 | **extended 75 (D4)** | +20000…+20399 / +21000…+21399 / +22000…+22399 |
| **B0** | ball λ=4, σ_z = 0 | 0.73 × 400 | canonical 41 | +23000…+23399 |
| **B1** | ball λ=4, σ_z = 0.010 | 0.73 × 400 | canonical 41 | +24000…+24399 |
| **B2** | ball λ=4, σ_z = 0.035 | T × 400 | canonical 41 | +25000…+25399 / +26000…+26399 / +27000…+27399 |
| **O1** *(NOT built)* | volume-kernel arm | — | — | +28000…+28399 reserved; **NOT-EVALUABLE** (§9 item 6, unchanged) |
| **V1** | ball path, λ = 0, σ_z = 0 | 0.73 × 50 | canonical 41 | +29000…+29049 |

Fixed per-cell; a seed appears in exactly one cell; **no v2 seed appears in
any v1 cell** (D6, unit-tested).

**Runtime budget (from v1's measured wall time).** v1 measured **3.46 h**
wall on 14 local workers for the identical cell set at 41 grid points
(adjudication `wall_total_h = 3.4618`). The only v2 cost delta is the A-cell
grid (75/41): extra CPU = 3 truths × 400 seeds × 7.95 s/seed (v1 measured
full-N A timing) × (75/41 − 1) ≈ 7.9 k CPU-s ≈ **+0.16 h wall at 14
workers**. Predicted ≈ **3.6 h**; budgeted band **3.5–5.0 h**. Smoke first
(3 seeds/cell, D8). Abort criteria unchanged (§10).

**Estimator config mirrored from production** — unchanged from v1 §5:
N_det = 1500; `numerator_pdet = off`; `snr_threshold = 20`; 50-node
Gauss–Legendre, 64-node Gauss–Hermite; injection pool `mix200k_20260728`;
CRB CSV `results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv`.

## 6. Per-seed outputs

v1 §6 fields unchanged (for A cells the `ln_post_1d/2d` vectors are
75-point). Document-level additions (D5): `import_path_clean`,
`dirt_inventory {import_path, other}`. Aggregate-level additions:
`ds1_ds2_degenerate_pit_exempt` (D3); DS-7 block gains `status:
"REPORT-ONLY"` and `p_bar_mc_se` (D2).

## 7. DECISION STATISTICS (bands locked at this commit)

**DS-1 — HPD coverage.** Unchanged from v1 §7 (binomial nulls; N=400 2σ
bands [0.450, 0.550] / [0.633, 0.727] / [0.870, 0.930]; 3σ
[0.425, 0.575] / [0.610, 0.750] / [0.855, 0.945]; N=300 and N=200 fallback
rows as v1). **Not scored on B0/V1 (D3).**

**DS-2 — P–P/KS.** Unchanged (N=400: PASS D ≤ 0.0679, FAIL D > 0.0814).
**Not scored on B0/V1 (D3).**

**DS-3 — MAP bias.** Unchanged (in-band |b| ≤ 0.010; defect |b| ≥ 0.030).
For A cells the primary statistic is the grid-argmax on the **full 75-point
grid**; the canonical-restricted argmax is reported alongside (D4).

**DS-4 — rail statistic.** Unchanged (R_low/R_high = grid-edge argmax
fractions). For A cells the edges are 0.460/1.060; the restricted-read rail
at 0.600 is DS-8 T1's statistic and is reported alongside.

**DS-5 — width vs F5 forecast.** **Unchanged from v1** (factor-2 screen,
W ∈ [0.5, 2.0]; the committed F5 sweep has no exact σ_z nodes at
{0, 0.010, 0.035}, so the exact-venue screen remains NOT-EVALUABLE and
bracket reads at nearest committed nodes are raw context only, exactly as
the v1 readout handled it — no trivial improvement was available without a
new F5 run, which stays a registered follow-up, §9 item 3).

**DS-6 — rail-reproduction contrast (Q2).** Unchanged from v1 §7
(thresholds 0.90/0.05 on `R_low` of the 1D channel; RAIL-REPRODUCED /
RAIL-NOT-REPRODUCED / MIXED; the B0 low-anchor condition and the pre-named
"impostor-ball analog of the N-2 finding" carried verbatim). B2/B0 are on
the unchanged canonical grid, so the statistic is v1-identical.

**DS-7 — generator-closure accounting: REPORT-ONLY, both forms (D2).**
Reported per cell: raw ratio, corrected ratio, `p_bar`, `p_bar_mc_se`, both
pass booleans against the (now weightless) 0.05 band. **No V-class weight,
no branch weight, not in the §10 trigger set. The raw-vs-corrected author
call is OPEN.**

**DS-8 — NEW: v1 pattern-reproduction targets (Q4).** Three pre-named
targets. Sources: v1 committed per-cell values and their MC standard errors
(readout + adjudication, commit `3a572897`). Band arithmetic (all numbers
fixed here):

*Binomial thresholds from extreme v1 counts.* For an observed 400/400 the
Jeffreys 95 % lower bound is p₀ = Beta⁻¹(0.025; 400.5, 0.5) = 0.9937;
3σ binomial slack at p₀ is 3·√(p₀(1−p₀)/400) = 0.0118 ⇒ threshold
0.9819, rounded conservatively (looser) to **≥ 0.98** (≥ 392/400). For an
observed 0/400 the Jeffreys 97.5 % upper bound is 0.0063; +3σ slack ⇒
0.0181, rounded (looser) to **≤ 0.02** (≤ 8/400). (For V1's N = 50 no DS-8
band is set — V1 is a control, not a target.)

*Bias bands.* v2's cell-mean grid-MAP bias is an independent estimate with
SE ≈ SE_v1, so the v1↔v2 difference has SD ≈ √2·SE_v1; band = v1 value ±
4·√2·SE_v1 (per-component false-refute ≈ 6×10⁻⁵; family-wise over the 8
banded components < 10⁻³).

- **T1 — single-host starvation rail (A-1D).** v1: restricted-grid MAP
  = 0.600 for 400/400 seeds at every truth (the committed 200/200 anchor
  reproduced as 400/400 ×3). **Band: fraction of A-1D seeds whose
  canonical-restricted argmax equals 0.600 is ≥ 0.98 at each of the three
  truths.** The full-75-grid `R_low` (rail at 0.460) is REPORTED as new
  information, un-banded (v1 could not measure it).
- **T2 — ball-venue uniform +σ_z bias with collapsed coverage (B1, B2; both
  channels).** Banded components (v1 value ± 4√2·SE_v1, computed):

  | cell×channel | v1 bias (±SE) | v2 band |
  |---|---|---|
  | B1-1D | +0.01091 (±0.00010) | [+0.01036, +0.01147] |
  | B1-2D | +0.01120 (±0.00011) | [+0.01059, +0.01181] |
  | B2(0.690)-1D | +0.03492 (±0.00010) | [+0.03434, +0.03551] |
  | B2(0.690)-2D | +0.03517 (±0.00019) | [+0.03408, +0.03627] |
  | B2(0.730)-1D | +0.03541 (±0.00011) | [+0.03476, +0.03606] |
  | B2(0.730)-2D | +0.03576 (±0.00021) | [+0.03456, +0.03696] |
  | B2(0.770)-1D | +0.03712 (±0.00023) | [+0.03584, +0.03841] |
  | B2(0.770)-2D | +0.03815 (±0.00025) | [+0.03673, +0.03957] |

  plus, per B1/B2 cell×channel (v1: 0/400 everywhere): **C90 ≤ 0.02** (HPD
  nesting makes C50/C68 redundant) and **R_low ≤ 0.02 and R_high ≤ 0.02**.
  The "posteriors far too narrow for the bias" feature (v1 sd_med ≈ 0.003 ≪
  bias) is REPORTED with v1 values alongside, un-banded (no committed SE for
  a median exists).
- **T3 — B0 exactly on truth.** v1: grid-MAP = 0.730 exactly for 400/400
  seeds, both channels; rails 0/400. **Band: fraction of B0 seeds with
  grid-MAP = 0.730 exactly ≥ 0.98 per channel; R_low ≤ 0.02 and
  R_high ≤ 0.02 per channel.**

*Scoring.* Per target: **CONFIRMED** = every banded component inside its
band; **REFUTED** = any banded component outside; all raw values reported
either way, refutation direction stated. **DS-8 carries no branch weight
and no trigger-set membership** — it is a pattern-reproduction meter for
the author's stage-5 read, and it is **void** (like every measurement) if
the §10 branch fires GATE-NOT-TRUSTWORTHY.

## 8. Edge-contamination guard

Unchanged from v1 §8 (edge-loaded = `edge_mass > 0.01`; cell×channel
EDGE-CONTAMINATED if > 10 % of seeds edge-loaded ⇒ DS-1/DS-2 carry no gate
weight; DS-4/DS-6 exempt). For A cells the guard operates on the extended
grid's outermost intervals (D4), which is the point of the repair: the
predicted residual edge-load is ~2–5 % per truth (pre-declared in D4); if
> 10 % the guard fires honestly.

## 9. NOT-EVALUABLE registry (carried unchanged from v1 §9)

1. Stage-5 third stop-digging condition, production side — carried by leg
   2's standing result + the open f_k–pool-coupling intake thread; any
   REPORT-BOUND remains explicitly conditional.
2. Leg-2 venue transfer — PENDING-AUTHOR-CONFIRMATION.
3. Leg-3 fine read — DS-5 stays a factor-2 screen; matched-population F5
   run is a registered follow-up.
4. Production in-catalogue host-mass kernel (R&V15) — balls carry redshift
   only.
5. GLADE n(z) / completeness map / sky-cone geometry / `f_incl < 1` — the
   ball is a z-window Poisson caricature.
6. `volume_deconv` kernel form — **cell O1 stays NOT built; NOT-EVALUABLE**
   (offsets reserved so it can never be post hoc).

## 10. Validity: determinism, controls, provenance, abort criteria

- **V1 — plumbing control** (cell V1): unchanged (MAP = 0.730 exactly, both
  channels, all 50 seeds; the divergence-10 tail-ε caveat carried verbatim:
  a > 4σ-scattered true host is reported per seed, never patched around).
  Any failure ⇒ STOP.
- **V2 — HPD port certification**: unchanged (boolean-exact agreement with
  `pp_coverage._hpd_contains` on 1000 random posteriors; in CI).
- **V3 — determinism**: unchanged (bit-identical re-runs; smoke spot-check
  built in).
- **V4 — texture certification**: the `dl_binned` detected set must show
  median `corr(ln σ_dL, ln d_L)` **∈ [0.63, 0.75]** — the band derived in
  D1 from the pre-declared attenuation analysis (0.69 ± 3 × 0.02). Failure
  ⇒ texture cells void (unchanged consequence). The marginal-σ-quantile
  clause of v1 is carried unchanged.
- **V5 — R0 reproduction**: unchanged (≤ 1e-12 relative).
- **Config provenance — the v2 clean rule (D5, exact):** registered cells
  REFUSE to run if the import path (`master_thesis_code/`,
  `master_thesis_code_test/`) has any uncommitted change (modified or
  untracked); `--allow-dirty` exists for `--smoke`/`--validate` only and is
  recorded; the full dirt inventory of everything else is embedded in every
  output JSON; every JSON embeds `git_commit`, the config dump, seeds, wall
  time, worker count.
- **Abort criteria**: unchanged from v1 — (a) smoke extrapolation > 12 h ⇒
  registered 300-seed fallback; still > 12 h ⇒ STOP, author call; (b)
  non-finite `ln_post` in > 1 % of any cell's seeds ⇒ STOP; (c) any
  V-failure ⇒ STOP. No band may be adjusted after any readout.
- **GATE-NOT-TRUSTWORTHY trigger set (v2)** = {V1…V5 failure, abort (b)}
  ∪ {both decision cells EDGE-CONTAMINATED in the channel being read}.
  **DS-7 is removed from the set (D2).** B0/B1/B2 measurement anomalies
  remain findings, never trustworthiness escapes.

## Branches (unchanged adjudication frame; presented to the author, never self-adjudicated)

The four branches — **GATE-NOT-TRUSTWORTHY**, **KEEP-DIGGING**,
**REPORT-BOUND**, **MIXED (first-class, non-forcing)** — are carried
verbatim from v1, including the stage-5 decision table quotation, the
conditionality of any REPORT-BOUND on §9 items 1–3, and the pre-named
"impostor-ball analog of the N-2 finding". The only changes: the trigger
set is the v2 set above (D2), B0/V1 DS-1/DS-2 labels never enter any branch
condition (D3), and the DS-8 verdicts are reported to the author alongside
the branch, carrying no branch weight (D7).

**Anti-tuning:** every threshold in this file (400/300 seeds; DS-1 2σ/3σ
binomial bands; KS 95/99; 0.010/0.030; 0.90/0.05; [0.5, 2.0]; 0.01/10 %
edge guard; V4 [0.63, 0.75]; DS-8 0.98/0.02 and the eight bias bands;
the 75-point A grid; the +20000 seed offsets) is fixed at this commit,
derived analytically, from the pre-declared build-time analysis, or from
v1's committed artifacts — and may not be adjusted after any readout. The
git object of this file is the evidence of what was registered.

**Model/effort policy for the readout:** carried verbatim from v1 —
mechanical extraction at low effort; interpretation/adversarial pass at
high effort; **the branch call is presented to the author, never
self-adjudicated.**

---

Verdict to be appended below by the session that reads out the v2 run —
after this file is committed, no edits above this line.

---

## 11. Appendix log (append-only)

### 2026-08-10 — registration-time build evidence (same commit as this file)

- **Instrument edits**: module divergence-log items 11–17 (V4 band, DS-7
  report-only + `p_bar_mc_se`, degenerate-PIT exemption marker, extended A
  grid, clean rule, +20000 seed offsets, v2 registration identity).
- **Test suite**: 30/30 calibration-gate tests pass; full fast suite
  1295 passed / 15 skipped; ruff + mypy clean (whole tree).
- **No-drift proof (ball + plumbing paths)**: 3-seed smokes at v1's smoke
  seeds (B2 h=0.730: 20266808–10; V1: 20269808–10, N=300) reproduce the
  committed v1 smoke artifacts' `per_seed` records **bit-identically**
  (`smoke_B2_h0p730.json`, `smoke_V1.json`); built-in V3 spot-checks pass.
- **Extended-grid smoke (A, h=0.730, v2 seeds 20281808–10, N=300)**: 75-point
  grid end-to-end; all `ln_post` finite; 1D rails at the new low edge 0.460
  (starvation signature extends beyond the old grid); 2D MAPs interior
  (0.85–0.96 at smoke N — values that were truncated at 0.860 on the v1
  grid); V3 spot-check bit-identical.
- **Clean-rule live exercise**: a registered-cell invocation on the then-
  dirty import path refused with the full file list; `--allow-dirty`
  without `--smoke`/`--validate` was rejected; no output files written.
- The code identity of the v2 instrument is **this commit** (instrument +
  tests + prereg atomically).

*(v2 smoke results, deviations discovered during the run, and the readout
are appended here with dated headings; the text above stays.)*
