# PRE-REGISTRATION — A-FULL (stage 5): the correct-form estimator arm

**Date:** 2026-08-15 · **Authorized:** ledger row #110 items 1–3 ("all approved", author verbatim)
· **Governing draft:** `DRAFT_A_FULL_ESTIMATOR_20260815.md` + addendum (candidate = **FULL-F**) ·
**Status: REGISTERED at the commit that carries this file, the instrument variant, the scorer, and
the seed-block test.** A8-v2 discipline throughout; branches presented, never self-adjudicated;
append-only from the registering commit.

## 1. The question

Does the correct-form estimator — the d_obs-density GW factor, the selected-population prior
w_pop(z;h)·S̄_φ(z;h)/α(h) (the existing −N ln α retained as that prior's normalization), and the
leave-one-out impostor weight 1/imp_k; **no Jacobian, no kernel renormalization** — zero the venue
tilt and displacement **on the instrument**, as its mirror pre-measurement predicts
(T_paired = +30.6 ± 42.7 nats/h at full dose, zero-consistent; `L4_AFULL_PREMEASURE_F_output.json`)?

## 2. The arm

| cell | variant | h_true | N seeds | seed offsets | dose |
|---|---|---|---|---|---|
| **AFULL** | `a_full` (ESTIMATOR_VARIANT_A_FULL) | 0.730 | 25 | **+54200…+54224** (base `VT_BASE_SEED` = 20260808) | full (`dose_target="all"`) |

Everything else identical to MN0X/AM2P/AJREN: pinned 982 events, `balls="real_k"`,
`sigma_mode="glade"`, canonical 41-point grid, `n_events_cap=None`, `chunk_pairs=16384`, the four
standing pins (CRB CSV / frozeng emit / pruned catalogue / injection pool, `PINNED_INPUTS_MANIFEST.md`).
Seed block +54200…+54224 is fresh and disjoint from every reserved/consumed block (+50000/+51000
decades, +53000, +54000/+54100, W1/O2 envelopes); the disjointness unit test is extended to cover
it in the registering commit.

**Cell-spec registry entry (installed in the registering commit):**

```python
AFULL_CELL_SPECS: dict[str, VenueCellSpec] = {
    "AFULL": VenueCellSpec(
        "AFULL", "A-FULL", "real_k", "glade", (0.730,), (25,), (54200,), "all",
        estimator_variant=ESTIMATOR_VARIANT_A_FULL,
    ),
}
```

## 3. Code form

`ESTIMATOR_VARIANT_A_FULL = "a_full"` in `venue_transfer._channel_terms_at_h`, installed in the
registering commit: kernel-branch integrand `kern · N(d_obs; d_L, σ_d·d_L) · w_pop(z;h) ·
S̄_φ(z;h) · (1/imp_k)`; point branch the same factors at z_obs; every other variant byte-identical
to its pre-existing path (guarded additions only; the base byte-identity unit test must stay
green). The LOO weight `_loo_impostor_weights` is the verifier-C1 construction verbatim
(window-truncated catalog density ⊛ σ_k kernel, floor 1e-3, h-independent). The `−N ln α` term is
retained (prior normalization). The 2D channel picks up the same integrand through the existing
`g` machinery unchanged.

## 4. Decision statistics (bands seeded from the mirror pre-measurement per row #110 item 3)

Mirror basis (15 seeds): T_paired sd = 42.7·√15 = 165.4 nats/h → arm SE at N = 25: 33.1.
Prediction-vs-arm comparison sd = √(33.1² + 42.7²) = 54.0 nats/h.

- **DS-F1 (PRIMARY, branch-carrying): the 1D tilt at truth.** T(AFULL, 1D, grid-neighbour
  central difference at h_true) ∈ **[−131.5, +192.7]** (= +30.6 ± 3×54.0). False-fail under the
  mirror hypothesis: 0.3%. Two-sided; computed from raw `ln_post_1d` vectors, never the
  aggregate block.
- **DS-F2 (secondary, WEAK, non-branch-carrying): MAP bias.** Expectation |b(1D)| ≤ 0.003
  (T-band / Ā_coded ≈ 7.0×10⁴, displacement-law conversion). WEAK because Ā of the new form is
  not pre-measured (the width may change); reported, not adjudicating.
- **DS-F3 (coverage, branch-carrying jointly with DS-F1): binomial bands at nominal.**
  RESTORED read requires hpd50 ∈ [0.20, 0.80] AND hpd68 ∈ [0.40, 0.96] AND hpd90 ∈ [0.72, 1.00]
  (±3σ binomial at N = 25). Reference: every prior arm sits at 0/25.
- **DS-F4 (2D channel, descriptive, non-branch-carrying):** T(2D) − T(1D) reported raw against
  the coded form's +129 ± 24 excess; no prediction basis is registered — no band.
- **DS-F5 (specificity, descriptive):** per-seed T scatter and zero-rail/NaN counts reported;
  any rail or non-finite event triggers §6 STOP handling.

## 5. Branches (presented to the author; none self-adjudicated)

1. **DS-F1 PASS + DS-F3 RESTORED** → the venue mechanism thread has a complete, validated
   estimator account (M-OWNED-CLOSED candidate); the production `/physics-change` proposal for
   `bayesian_statistics.py` opens with this arm as its evidence base. [Author ruling.]
2. **DS-F1 PASS + DS-F3 NOT restored** → displacement is fixed but width is not — the
   width/curvature channel becomes the lead (the drift-eval's width terms and the removed-renorm
   broadening are the first suspects).
3. **DS-F1 FAIL high (T > +192.7)** → a positive term is still missing on the instrument;
   first suspects: pool-vs-model prior mismatch (KS D = 0.085) and LOO-model error; the mirror
   cross-check (§6 item 1) localizes instrument-vs-mirror discrepancies immediately.
4. **DS-F1 FAIL low (T < −131.5)** → over-correction; first suspects: the S̄_φ share (B-vs-D
   asymmetry +287) and the LOO floor.
5. **OTHER / confounded** (rails, non-finite, pin failure) → STUDY-CONFOUNDED; no branch forced.

## 6. Validity, execution-completeness, and STOP

1. **Pre-submission gate (already executed; result recorded in the registering commit):** the
   installed instrument variant must reproduce the pre-measurement mirror on seed 20310808 at the
   two h-neighbours to |Δ ln1| < 1e-6, and its paired tilt ≈ +30.0 (the verifier's pin). A
   failure blocks submission — no tuning, the discrepancy returns to the author.
2. Scorer `score_stage5.py` pre-committed in the registering commit; mechanics dry-run against
   the committed AJREN JSON (schema identical) before submission.
3. Pins: `check_pin_integrity` must pass on the cluster before submission (`/cluster` preflight
   VERDICT: READY required).
4. STOP: any seed with railed posterior, non-finite `ln_post`, or pin mismatch → hold, report,
   no re-run without an author ruling. Budget ceiling 40 CPU-h (expected 25–37; AJREN realized
   1.49 h wall at 15 workers; the a_full integrand adds ~10–20% node cost + a per-call LOO
   construction ~1 s).
5. Expected NULLs, pre-registered: the low-dose residual (+168.9 ± 58.8 at f_i = 0.25) is NOT
   probed by this arm (full dose only); the pool-vs-model mismatch is NOT resolved by this arm
   (it is a stated residual regardless of outcome).

## 7. Provenance

Draft + addendum `860b9d3f`; pre-measurements `L4_AFULL_PREMEASURE{,_D,_F}_output.json`
(reproducible via `l4_afull_premeasure.py --stage {abc,de,f}`); ledger rows #108–#110; mirror
validation chain `L4_DER_PART2_20260815.md` (bit-exact) + `L4_DRIFT_EVAL_20260815.md`;
verifier reports (Part-2 and A-FULL, both applied as addenda). Cost anchor 0.969 CPU-h/seed
(reserved-core definition, runbook note).

## 8. Pre-submission gate results (recorded at the registering commit, §6 item 1; verifier MINOR-1 remedy)

- **Instrument-vs-mirror cross-check (seed 20310808):** instrument ln1 = −1492.7283459544997
  (k = 20, h = 0.725) and −1492.4283637535445 (k = 22, h = 0.735); mirror (`full_ln1`,
  variant `full_f` + LOO, + −982·log_alpha[k]) identical at both — **diff = 0.0 exactly**;
  paired tilt T = **+29.998** nats/h vs the pin ≈ +30.0. Independently re-run by the
  pre-registration verifier (fresh execution, not the stale scratchpad output).
- **Scorer dry-run:** `score_stage5.py --input AJREN_h0p730_results_seeds0_25.json` reproduces
  T(1D) = +514.5 ± 16.8, bias +0.0178, coverage 0/25, branch-3 read — mechanics validated.
- **Verifier verdict: GO** (zero CRITICAL/MAJOR; three MINORs — this record remedies MINOR-1;
  MINOR-2: exact band endpoints [−131.2, +192.5] vs registered [−131.5, +192.7], registered is
  0.2–0.3 nats/h conservative-wider, kept as registered; MINOR-3: the h-independent LOO weight is
  recomputed per h-point, ~0.9 CPU-h waste ≈ 4%, inside the 40 CPU-h ceiling — post-arm cleanup,
  frozen as-is for the registered code form).
- Floor probe: zero `_LN_ZERO_EVENT` events at grid extremes k = 0 and k = 40, both channels
  (seed 20310808 replay) — DS-F3's binomial bands are not floor-distorted.
- Cluster preflight at submission time: VERDICT: READY ✓ (2026-08-15).
