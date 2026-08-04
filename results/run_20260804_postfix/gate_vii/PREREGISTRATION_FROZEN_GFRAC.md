# Pre-registration — frozen-g_frac 41-h evaluate

Registered 2026-08-04, BEFORE the run. Per `RUNBOOK_NEXT_SESSION_7.md` §1 (gate (vii)
follow-up) and `docs/RESEARCH_CYCLE.md` stage 2.

**Why this run is necessary.** The gate (vii) adjudication
(`results/run_20260804_postfix/gate_vii/adjudicate_g_frac.py`, its docstring is the
verdict of record) established, on the post-fix diagnostics CSVs already on disk
(41 h × 1588 events, both venues, 7 s.f. path-A columns):

- `g_frac` is **not** a per-h near-scalar: 1587 distinct per-event values at h = 0.73
  (min/median/max 0.076187 / 0.135240 / 0.241726) — the competing "≤6 distinct values"
  claim is REFUTED;
- the column is self-consistent: `|g_frac − B_num_wbh/B_num| ≤ 5e-8` per row;
- the event-summed `ḡ(h) = Σ B_num_wbh / Σ B_num` rises monotonically
  0.134769 (h=0.60) → 0.138202 (0.73) → 0.141337 (0.86), Δln = 0.047586 across the
  grid, **bit-identical in both venues** (the completion machinery is venue-independent);
- pinning each event's `g_frac` to its own h = 0.73 value in the CSV proxy moves the
  2D MAP **0.780 → 0.660** (iiib) and **0.800 → 0.640** (joint_r1); the 1D rail stays
  at 0.600 in both.

That is a **CSV proxy**: it re-weights the emitted columns rather than re-running the
estimator, so it cannot see any compensating renormalisation that the real mixture
might carry (`w̃_G`, `α_G^φ`, `D̃^φ`, `B_scale = β_Ḡ^φ/β_Ḡ`). This run replays the
same counterfactual **inside the estimator**, where every downstream object is free
to respond. It is the decisive instrument for the attribution "the h-slope of the
completion-leg mass factor owns the residual 2D high-h displacement".

**This is instrumentation, not a physics change.** No production formula moves; the
toggle is default-off and byte-identical when unset (see the code commit below).

## The run

Two CPU evaluate arrays (41 h-points, canonical grid) + combine on the cluster — one
per venue, everything except the new flag identical to the post-fix runs of record:

| | value |
|---|---|
| RUN_DIR (idealized) | `$WS/run_20260804_postfix_iiib/` — frozen twin of the post-fix iiib evaluate |
| RUN_DIR (joint) | `$WS/run_20260804_postfix_joint_r1/` — frozen twin of the post-fix joint r1 evaluate |
| CRB input | the existing `prepared_cramer_rao_bounds.csv` **symlink target `run_20260729_seed61000/`** — the same file both post-fix runs consumed. No re-simulation. |
| Injection pool | the same `injections/` pool symlink as the post-fix runs (`injection_pool_mix200k_20260728`) |
| Catalogues | unchanged per venue: iiib = idealized (parent/exact-z); joint_r1 = realization r1 (delivered/observed). No new realization is drawn. |
| Estimator | `NORMALIZATION_MODE=absolute_marginal`, `HOST_Z_KERNEL=volume_deconv`, `HOST_MASS_KERNEL=auto` — the post-fix path-(A) pairing, unchanged |
| **New flag** | `--freeze_g_frac_ref_h 0.73` (and **only** that) |
| Code commit | `121f57d850beb4c5c44a7fa08f67c38dfcf72784` (`main`, "instrumentation: --freeze_g_frac_ref_h …") |
| h grid | the canonical 41-point grid: 0.01 steps on [0.60, 0.65] and [0.79, 0.86], 0.005 steps on [0.655, 0.79] |

The flag's semantics, per event: `g_ref = B_num_wbh(0.73)/B_num(0.73)` computed by the
SAME quadrature re-evaluated at h_ref = 0.73, and at every evaluated h the 2D
completion term becomes `B_num(h)·g_ref` in place of `B_num_wbh(h)`. The 1D channel,
both catalogue legs, `B_num` itself and the whole selection stack are untouched.
`--freeze_g_frac_ref_h` is recorded automatically in `run_metadata.json` (the whole
argparse namespace is serialised) — check it before reading any result.

**Grid-step tolerance.** "± one grid step" is asymmetric on this grid: at 0.64 a step
is 0.01, at 0.66 it is 0.005. The CONFIRM band below therefore admits
**[0.63, 0.665]** inclusive.

## Pre-registered branches

- **CONFIRM** — the 2D MAP lands in **0.64–0.66 ± one grid step** (i.e. [0.63, 0.665])
  in **both** venues ⇒ `g_frac(h)` **is** the carrier of the residual 2D displacement.
  The question then stops being an attribution question and becomes a
  `/physics-change` derivation question: is the h-dependence of the completion leg's
  mass factor correct as derived, or is a normalisation missing from it? This
  converges with ledger #87 and possibly with **D1** (the p0-window mass band-pass —
  a mass band-pass is exactly a z-selection distortion at fixed source mass, the shape
  that would put an unmodelled h-slope into the completion leg's mass factor).
  *(Note at registration time: no GitHub issue #87 exists; "ledger #87" is read as the
  corresponding row of `docs/gates/PHYSICS-GATE-LEDGER.md`.)*

- **REFUTE** — the 2D MAP **stays at 0.78–0.80** ⇒ a compensating normalisation exists
  inside the mixture that the CSV proxy could not see, the attribution is **wrong**,
  and this thread closes with an **exoneration** of `g_frac(h)`. The suspect list
  reverts to D1 and the in-cat class tension, unmodified by this run.

- **MIXED** — anything else: one venue moves and the other does not, or either lands
  intermediate (e.g. ~0.70). **Read the split directly; do not force a branch.** The
  non-moving venue names the compensating object — it is the venue whose mixture
  carries a normalisation that absorbs the frozen mass factor, and its
  `w̃_G`/`α_G^φ`/`D̃^φ`/`B_scale` deltas against its unfrozen twin identify it. Since
  `ḡ(h)` was measured bit-identical across venues, a venue split cannot come from the
  completion machinery itself and is therefore informative on its own.

## Secondary pre-registered reads

1. **The 1D posterior must be BIT-IDENTICAL to the unfrozen run.** `g_frac` is
   2D-only by construction (the 1D `B_num` is unmultiplied, gate (iv)). Compare
   `combined_1d.json` and the `combined_no_bh` / `L_cat_no_bh` / `B_num` / `L_comp`
   diagnostics columns byte-for-byte against the post-fix runs. **If it differs, that
   is itself a finding** — it would mean the freeze leaked out of the 2D leg, and the
   run is void until explained.

2. **The emitted `g_frac` column must be h-CONSTANT per event.** Pivot the new
   `event_likelihoods.csv` on `(event_idx, h)`; every row must have exactly one
   distinct `g_frac` value, equal to that event's unfrozen h = 0.73 value to
   round-trip precision (7 s.f. output). This is the run's own self-certification
   that the flag did what it says; check it BEFORE reading any MAP.

3. Directional sub-prediction, conditional on CONFIRM: the frozen 2D MAP should
   **overshoot below the injected 0.73**, as the proxy found (0.66/0.64), not merely
   relax to 0.73. A landing exactly at 0.73 would mean the mass factor's h-slope
   accounts for precisely the bias and nothing else — a suspiciously clean result that
   should be treated as MIXED and re-examined, not celebrated.

4. `w̃_G(h)`, `r_Malm(h)`, `α_G^φ(h)`, `D̃^φ(h)`: expected **bit-identical** to the
   unfrozen runs (pure selection-side quadrature, no dependence on the completion
   numerator). If any of them differs, that is itself a finding.

## Scope guard

- **No re-simulation.** The CRB set and the injection pool are consumed through the
  existing seed61000 symlinks; no waveform, no Fisher matrix, and no injection is
  regenerated.
- **The D1 constraint of record is untouched.** The 3135-event catalogue has still
  never been re-scored against band-blind objects, and this run does not change that.
  Nothing here may be read as evidence for or against D1's remedy.
- **No production posterior is produced.** Both runs are counterfactuals by
  construction; the estimator logs a WARNING saying so. They must never be quoted as
  a result, only as a diagnostic contrast against their unfrozen twins.
- **Any actual fix routes through `/physics-change`** — derivation, dimensional
  analysis, limiting case, literature reference, regression test, ledger row. This
  file authorises a measurement, never a formula change.

Verdict to be appended below by the session that reads out the run — after this file
is committed, no edits above this line.
