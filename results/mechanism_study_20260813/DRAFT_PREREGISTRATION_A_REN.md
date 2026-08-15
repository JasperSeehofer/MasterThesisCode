> **DRAFT — NOT REGISTERED. Bands marked TBD-pending-L0 are placeholders; the document registers
> ONLY when committed with a REGISTER commit after author approval (proposal §3 item 3).**

# PRE-REGISTRATION (DRAFT) — A-REN kernel renormalization + conditional A-JREN (thread 17, stage 3)

**Date:** 2026-08-15 · **Status: DRAFT.** Drafting authorized (ledger row #105 item 2, `[DO —
granted]`, author "approved" on `PROPOSAL_STAGE3_20260815.md` §3 item 2). **Registration —
whether this document may be committed as REGISTERED, seeds reserved, and any arm run — is
explicitly deferred to the author as a fresh [RULE]/[DO] per row #105 item 3 and the proposal's own
§3 item 3 ("returns to the author as a fresh [DO] with the L0 evidence and the full A8-v2
registration attached").** Nothing in this document authorizes cluster time, a seed reservation, or
an instrument run. **Parent:** `PREREGISTRATION_MECHANISM_ISOLATION.md` (`73141160`) via
`PREREGISTRATION_M2PRIME_ABLATION.md` (this stage's immediate predecessor) — their §1 ladder, §3
anti-tuning clause, §5 validity checks and §7 closures are inherited verbatim unless explicitly
superseded below. **Discipline:** Amendment **A8 as revised and adopted** (ledger row #102 item 5;
`docs/RESEARCH_CYCLE.md` row A8) — every branch names its satisfying arm and what it ablates; every
rule naming a point prediction is two-sided; **no count/class-based branch is adjudicated while a
registered arm capable of changing the classification is unrun**; every band states its derivation
and false-fail rate.

**Toy-calibration bar (carried, binding here):** per the commission review
(`results/commission_research_20260814/REPORT.md`, D1-03/D1-12) the M5 L0 toy is ruled unfaithful
at production K (ledger row #102 item 1). **No band or branch edge in this document is, or will be,
derived from any L0 toy.** The A-REN-specific expectation bands (§4) are explicitly marked
TBD-pending-L0-REN-B and will be filled from committed instrument statistics once L0-REN-B's toy
reads land — the toy supplies a *pre-stated read*, not a *band edge*; per the commission's D1-03
bar, branch-carrying rule edges come only from committed instrument statistics (DS-M1, carried
verbatim from the stage-2 registration), never from the toy.

---

## 1. The question

Stage 2 measured M2′ **PARTIAL and on-prediction** (ledger row #103): restoring the z-integral
measure removes 48.5% of `b_ref`, leaving +0.0192 ± 0.0007 (0/25 coverage). Row #104 ratified that
the residual `T_res` — the dose-dependent tilt component left after α and J are accounted for — is
**genuinely unlocated**: ~⅔ of the measured ±760 nats/h dose swing is irreducible to the α/J
account, and the M1-quadratic candidate for `T_res` is REFUTED quantitatively (wrong sign at two of
three doses, ~16× overshoot at full dose, inverted shape). Row #105 authorized four L0 items
targeting two complementary hypotheses (`PROPOSAL_STAGE3_20260815.md` §1):

- **H-REN** — the unrenormalized truncated kernel (the code never divides `c₁ₖ` by the retained
  kernel mass `W_k(h)`) owns `T_res`.
- **H-SB** — the residual displacement is a score-balance/misspecification effect, diagnosable on
  stored posteriors with no new instrument run.

**L0-REN-A is complete and frozen** (`L0_REN_A_DERIVATION_20260815.md`, this stage's authorized L0
item 1): it derives the defect term, its regime structure (double-clipped / single-clipped-boundary
/ interior-unclipped), and states — **before any toy runs** — that the net sign at full dose is
**not** pre-stated, only the width term's sign and scale are (+1055 nats/h saturated, σ-blind). It
also states the budget tension explicitly: a width-dominated `T_REN ≈ +10³` does not fit the
measured −62 ± 36 nats/h residual additively, and names two live, pre-stated possibilities — **(i)
cancellation within `T_REN`** (offset + boundary terms cancel the width term at this venue's σ/width
ratio) or **(ii) genuine non-additivity of ablations** (the J and renormalization defects do not
commute, so single-ablation tilts do not sum to the joint defect — in which case **the joint arm is
mandatory before any conclusion about repair**). **L0-REN-B (the A/B toy reading these two
possibilities) has NOT yet run as of this draft** — its `PRESENTED, NOT ADJUDICATED` reads (§4 of
the derivation doc: R1 magnitude, R2 dose shape, R3 budget, R-sign reported-not-read) are the gate
this document's registration passes through.

**A-REN is this stage's estimator-side L1 arm**, registered here in draft to lose no calendar time
if L0-REN survives its own kill checks (`PROPOSAL_STAGE3_20260815.md` §2, item "A-REN (conditional
L1)": "only if L0-REN survives both its kill checks and the author grants the [DO]").

### Candidate register state entering this draft (post row #104)

| id | term | status |
|---|---|---|
| **M2′** | missing measure/Jacobian inside the z-integral | **MEASURED PARTIAL, on-prediction** (row #103/#104); no repair licensed from a partial read |
| **M6′** | σ_z-blind tilt × dose-controlled curvature composite | **PROPOSED**, kill tests KT-M6′-2/3 relabeled one-sided (row #104); no instrument arm |
| **M7** | host/impostor ball-window inclusion asymmetry | **CLOSED at L0** — −3.79e-4 ± 1.65e-4 under the production-curvature conversion, 2.6× inside the registered band (row #104) |
| **T_res** | dose-dependent residual left after α + J | **UNLOCATED** (row #104); M1-quadratic account REFUTED; H-REN and H-SB are the two live candidate accounts (this stage) |
| **REN** (this study's target) | missing division by retained kernel mass `W_k(h)` in the kernel-branch integrand | **OPEN — L0-REN-A derivation frozen (`L0_REN_A_DERIVATION_20260815.md`), L0-REN-B toy PENDING**; no instrument arm run |
| M1, M3, M4 | — | CLOSED as per parent §7 |
| M5, M5′ | — | REFUTED (instrument); toy sub-closures NOT ESTABLISHED (row #102 item 1) |

## 2. The arms

Both arms are **estimator-side only**; the generator is untouched (identical to A-M2′/A-NULL's
discipline). The point branch (`sig_c <= 0.0` rows) is disjoint code and is **not** touched by
either variant — at σ_z = 0 there is no kernel and `W_k ≡ 1` trivially, so both arms are identically
the base estimator on point-branch candidates (constraint (a), the same one A-M2′ satisfies by
construction).

| arm | what changes (estimator-side ONLY; generator untouched) | trigger | seeds | N | prediction |
|---|---|---|---|---|---|
| **A-REN** | per-candidate kernel renormalization restored: `c₁ₖ(h) → c₁ₖ(h) / W_k(h)` (and `c₂ₖ(h) → c₂ₖ(h) / W_k(h)`, so both channels — mirroring A-M2′'s "multiplied into `integ`… hence into both c₁ and c₂"), with `W_k(h) = Φ((b−z_obs,k)/σ_k) − Φ((a−z_obs,k)/σ_k)` evaluated with the **same clip limits `a`, `b` the numerator integral already uses** (`a = max(z_lo(h), z_obs,k−5σ_k)`, `b = min(z_hi(h), z_obs,k+5σ_k)`) — no new window, only the renormalization of the existing one. Kernel-branch rows only; point branch untouched. | registered unconditionally (this study's primary L1 arm) | **+54000…+54024** (fresh, disjoint — VERIFIED below, checked by unit test before any run) | 25 | DS-M1 class read (§4); A-REN-specific magnitude/shape expectation TBD-pending-L0-REN-B |
| **A-JREN** (conditional) | J-restoration (A-M2′'s measure factor) **and** the A-REN renormalization applied **jointly** in the same kernel-branch integrand — the first candidate *full-repair* measurement for the located terms. Point branch untouched (both sub-terms vanish there). | **conditional, registered-but-not-triggered by default.** Fires iff **either**: (a) L0-REN-B's read **R3 = BUDGET-TENSION** (possibility (ii), non-additivity, in which case the derivation itself says "the joint arm is mandatory before any conclusion" — §3 of the derivation doc), **or** (b) A-REN's DS-M1 read (run on the instrument, not the toy) lands **TERM-PARTIAL**. Either trigger is sufficient; neither is necessary if the other fires. If neither trigger fires (L0-REN-B reads R3 = CONSISTENT or LIVE-but-not-BUDGET-TENSION, and A-REN lands TERM-OWNS or TERM-INNOCENT), **A-JREN is withdrawn without running**, per the parent's execution-completeness discipline (an arm may be withdrawn by an author [RULE], never silently skipped). | **+54100…+54124** (fresh, disjoint — VERIFIED below) | 25 | no point prediction stated here (the joint arm's role is diagnostic — "whether the located terms jointly restore calibration," `PROPOSAL_STAGE3_20260815.md` §2 — not a single-term ownership claim; §4/§7 below) |

**Budget note (parent §3 discipline, carried):** the parent budget is **L1 ≤ 5** cumulative across
the thread; 3 are already spent (A-ALL, split-dose, and the stage-2 pair A-M2′/A-NULL count as the
thread's spent slots per `PROPOSAL_STAGE3_20260815.md` §3 item 3, "both stage-2 L1 slots are spent;
any further instrument arm requires a fresh registration"). **This document proposes a *new stage*
with its own L1 ≤ 2 budget** (A-REN unconditional + A-JREN conditional) — exactly the ceiling the
task brief names ("subject to the L1≤2 budget of a new stage"). No L2 arm is proposed.

### Seed-block verification (performed before drafting this table)

Checked against every documented offset/reservation registry in the repository:

- `darksiren_emri/validation/venue_transfer.py`: `V1_SEED_OFFSET_ENVELOPE = (0, 9049)`,
  `V2_SEED_OFFSET_ENVELOPE = (20000, 29049)`, `V3_SEED_OFFSET_ENVELOPE = (40000, 45399)`,
  `RESERVED_SEED_OFFSET_BLOCKS = {"W1": (46000, 46399), "O2": (47000, 47399)}`, `MECH_CELL_SPECS`
  (`MN0`/`MN0X` 50000–50099, `MEH` 50100–50114, `MEI` 50200–50214, the 2-D dose-scan's reserved
  "+51000…+52599" decade-and-a-half), and `M2P_CELL_SPECS` (`AM2P` **+53000…+53024**, `ANULL`
  +50000…+50014, the latter a documented deliberate exception — paired with MN0X, not disjoint by
  design).
- `darksiren_emri_test/validation/test_venue_transfer.py::test_seed_blocks_match_prereg_and_are_disjoint`
  and `darksiren_emri_test/validation/test_m2prime_ablation_arms.py::test_seed_plan_disjointness_except_registered_anull_pairing`
  — the only two committed disjointness assertions; both operate in the `VT_BASE_SEED + offset`
  space (`VT_BASE_SEED = 20260808`), the *only* space these arms' seeds are ever added to.
- `cluster/datasets.yaml` and `cluster/preflight.sh` — no seed-offset registry entries; datasets.yaml
  tracks dataset **paths and provenance**, not the mechanism-study's per-arm seed plan.

**Result: +54000…+54024 (A-REN) and +54100…+54124 (A-JREN) fall in the open decade immediately
above the 2-D dose scan's reserved +51000…+52599 block and above A-M2′'s +53000…+53024 — neither
block is claimed, reserved, or documented anywhere in the registries above. No collision found in
the seed-offset space that governs this instrument's determinism.**

**Flagged, per the task brief's instruction to check the historical injection-pool usage of
"+53000":** `results/campaign51_20260728/PILOT3_READOUT.md` records a *raw absolute* seed 53000 used
for a since-retired GPU injection-generation pilot (job 6073027, "Pilot #2… lost 22/60 tasks",
retired), and raw absolute seeds 54000–54059 used for the *current* pilot-3 injection stack (job
6073215, `PILOT3_READOUT.md` line 4). **These are a different, unrelated seed namespace**: they are
raw integers passed directly as `--seed` to `main.py`'s GPU EMRI-simulation/injection generator
(campaign 51, a different pipeline entirely), not offsets added to `VT_BASE_SEED` inside
`venue_transfer.py`'s mechanism-study harness. The realized values never coincide numerically
(`VT_BASE_SEED + 53000 = 20313808` and `VT_BASE_SEED + 54000 = 20314808`, vs. the campaign-51 raw
seeds 53000 and 54000–54059) and the two pipelines consume RNG streams independently (the injection
pool is a fixed, already-generated dataset on the cluster workspace; it is not reseeded or
re-consumed by anything this study runs). **The digit coincidence is nominal only — no substantive
collision — and is recorded here as the explicit disclosure the task brief requested, not as
grounds to move the block.** +54000…+54024 / +54100…+54124 are kept.

## 3. Code form (fixed here for the draft; to be appended to `ARMS.md` verbatim at registration —
not appended now, per draft status)

One new `ESTIMATOR_VARIANT_*` constant added to the existing switch (the same scaffold A-M2′ and
A-NULL use — see `ARMS.md`'s "Stage-2 arms" section for the established pattern this mirrors):

```python
ESTIMATOR_VARIANT_KERNEL_RENORM = "kernel_renorm"
ESTIMATOR_VARIANT_JOINT_JREN = "jacobian_and_kernel_renorm"   # A-JREN only
```

Exact diff hunk against the same `_channel_terms_at_h` kernel-branch block A-M2′/A-NULL already
patch (point branch, `c1[rows_p]`/`c2[rows_p]`, untouched by construction — disjoint code path):

```diff
             p_gw = norm.pdf(d_L_frac, loc=1.0, scale=sig_p[rows_q][:, None])
             kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])
             if estimator_variant == ESTIMATOR_VARIANT_BASE:
                 integ = kern * p_gw
             elif estimator_variant == ESTIMATOR_VARIANT_M2P_JACOBIAN:
                 ...                                            # unchanged (A-M2')
             elif estimator_variant == ESTIMATOR_VARIANT_NULL_SCALE:
                 integ = (kern * p_gw) * NULL_SCALE_FACTOR       # unchanged (A-NULL)
+            elif estimator_variant == ESTIMATOR_VARIANT_KERNEL_RENORM:
+                # A-REN: divide by the retained kernel mass W_k(h), same clip
+                # limits a, b the numerator already uses (z_lo(h)/z_hi(h)
+                # window intersected with z_obs,k ± 5 sigma_k).
+                a_edge = np.maximum(z_lo_h, zo - 5.0 * so)
+                b_edge = np.minimum(z_hi_h, zo + 5.0 * so)
+                w_k = norm.cdf((b_edge - zo) / so) - norm.cdf((a_edge - zo) / so)
+                integ = (kern * p_gw) / np.maximum(w_k, _W_K_FLOOR)
+            elif estimator_variant == ESTIMATOR_VARIANT_JOINT_JREN:
+                # A-JREN: A-M2' Jacobian AND A-REN renormalization, composed
+                # on the same integrand (order: Jacobian multiply, then
+                # renormalization divide -- both act on rows_q, commute
+                # under floating-point only up to the registered w_k floor).
+                eps = M2P_JACOBIAN_EPS_Z
+                z_flat = np.maximum(z_nodes.reshape(-1), 1e-8)
+                d_hi = np.asarray(dist_vectorized(z_flat + eps, h=h), dtype=np.float64)
+                d_lo = np.asarray(
+                    dist_vectorized(np.maximum(z_flat - eps, 1e-8), h=h), dtype=np.float64
+                )
+                dd_dz = ((d_hi - d_lo) / (2.0 * eps)).reshape(z_nodes.shape)
+                jac = dd_dz / d_obs_p[rows_q][:, None]
+                a_edge = np.maximum(z_lo_h, zo - 5.0 * so)
+                b_edge = np.minimum(z_hi_h, zo + 5.0 * so)
+                w_k = norm.cdf((b_edge - zo) / so) - norm.cdf((a_edge - zo) / so)
+                integ = (kern * p_gw * jac) / np.maximum(w_k, _W_K_FLOOR)
             else:
                 raise ValueError(f"unknown estimator_variant '{estimator_variant}'")
             c1q = half * (integ @ w_gl)
             ...
```

`_W_K_FLOOR` (registered numeric guard, TBD exact value at registration — candidate `1e-12`,
matched to the existing `_LN_ZERO_EVENT` floor convention) prevents division blow-up for candidates
whose entire kernel mass falls outside the clip window; such rows are already vanishingly weighted
by `kern*p_gw` in the numerator, so the floor's effect on any candidate that matters is
sub-machine-epsilon — to be confirmed by a dedicated unit test at registration (mirroring A-M2′'s
`test_m2prime_jacobian_is_bit_identical_to_base_at_sigma_z_zero`-style inertness tests).

**`z_lo_h`, `z_hi_h`, `zo`, `so` are the same per-event/per-node arrays the base numerator already
computes** (the window edges and the kernel loc/scale) — no new quantity is introduced; `W_k` reuses
exactly the clip limits named in `L0_REN_A_DERIVATION_20260815.md` §1 (`a = max(z_lo(h), z_o−5σ)`,
`b = min(z_hi(h), z_o+5σ)`).

### Cell-spec registry entries (to be added at registration, shown here for the record)

```python
REN_CELL_SPECS: dict[str, VenueCellSpec] = {
    "AREN": VenueCellSpec(
        "AREN", "A-REN", "real_k", "glade", (0.730,), (25,), (54000,), "all",
        estimator_variant=ESTIMATOR_VARIANT_KERNEL_RENORM,
    ),
    "AJREN": VenueCellSpec(
        "AJREN", "A-JREN", "real_k", "glade", (0.730,), (25,), (54100,), "all",
        estimator_variant=ESTIMATOR_VARIANT_JOINT_JREN,
    ),
}
```

base = `VT_BASE_SEED` = 20260808. All other configuration identical to AM2P/ANULL/MN0X: pinned 982
events, `balls="real_k"`, `sigma_mode="glade"`, canonical 41-point grid, `n_events_cap=None`,
`chunk_pairs=16384`, the four §1 pins.

## 4. Decision statistics — DS-M1 carried verbatim; A-REN band TBD-pending-L0-REN-B

**DS-M1 (headline, 1D; 2D alongside)** — edges carried **verbatim** from the stage-2 registration
(`PREREGISTRATION_M2PRIME_ABLATION.md` §4), applied at N = 25 (SE ≈ 0.0010, from the campaign's
per-seed sd 0.005):

- **TERM-OWNS** = |b| ≤ 0.010 **and** HPD90 ≥ 0.60.
- **TERM-PARTIAL** = 0.010 < |b| < 0.030.
- **TERM-INNOCENT** = |b| ≥ 0.030 **and** |b − b_ref| ≤ 0.004.
- **OTHER** = anything else.

`b_ref = +0.037250` (MN0X, N = 100, committed) — the same reference the stage-2 registration used;
unchanged, since MN0X is untouched by any arm in this stage. Separations between class edges are
≫ 10σ at N = 25 (identical arithmetic to stage 2).

**A-REN-specific expectation band — `TBD-pending-L0-REN-B`.** No number is stated here. When
L0-REN-B's reads land, this section is amended (not silently edited — an amendment note, dated and
attributed, exactly as `AMENDMENT_A1_VM1_NULL_AT_N100.md` amended the parent) to state the
implied-MAP-shift expectation window, derived by:

1. Taking L0-REN-B's stacked tilt `T_REN(f=1.0)` (full dose) from the toy, in its **production
   conversion** (per the M7 correction, using production `σ²_post = 0.004386`, not the toy's own
   curvature — `L0_REN_A_DERIVATION_20260815.md` §4 names both conversions explicitly and requires
   the production one for any instrument-facing band).
2. Converting via the same MAP-displacement account the stage-2 registration used for A-M2′'s weak
   expectation (§2 of that document): `Δb ≈ −T_REN · σ²_post`, carrying the **same demonstrated
   ~1.5× scale-error caveat** the stage-2 document recorded for the local-Gaussian approximation
   (row #102 item... — flagged again here per the task brief: "the toy-unfaithfulness caveat row
   #102 noted" — this band inherits that caveat by construction, since it is built the same way).
3. Stating the window as WEAK, non-branch-carrying (identical status to A-M2′'s §2 expectation) —
   **the branch reads DS-M1 classes only** (below), never the toy-derived window; the window's sole
   purpose is legibility of surprise, exactly as stage 2 used it.

**No toy-calibrated number appears anywhere in §5's branch table** (commission D1-03 bar, restated
in this document's front matter) — every branch-carrying edge above is DS-M1, carried from the
committed stage-2 instrument statistics, not from L0-REN-B's toy.

**DS-J1 (A-JREN, conditional)** — if A-JREN runs, its read is **DS-M1 applied to the joint arm**
(same edges, same N = 25, same SE), reported alongside a **coverage-restoration check**: does the
joint arm's HPD90 return to ≥ 0.60 (i.e., does the *interval*, not just the point, calibrate) —
because A-JREN's registered purpose is testing whether the located terms jointly restore
calibration, not merely reduce |b| (`PROPOSAL_STAGE3_20260815.md` §2). No expectation window is
pre-stated for A-JREN (its role is diagnostic of non-additivity, not a point-prediction test); §7
states the two-sided reads explicitly.

## 5. Branches (A8-v2 form; presented to the author, never self-adjudicated)

**Execution-completeness clause (BLOCKING):** no branch below is adjudicated until A-REN has run
(A-JREN is conditional — see its own completeness note under branch 5) or A-REN is withdrawn by an
author [RULE].

**Split-precedence clause:** branches 2–4 each require their named DS-M1 class in **both channels**
(1D and 2D); **any 1D/2D class split routes to branch 5**, which takes precedence over branches 2–4.

**Specificity-control clause (BLOCKING, mirrors stage 2's DS-N1 STOP role):** no branch below is
adjudicated unless the §6 validity checks — including the machinery-transfer justification in §6 —
hold. A validity failure routes to branch 1 regardless of any DS-M1 reading.

1. **STUDY-CONFOUNDED** — satisfying arm: **either arm** via a §6 validity failure (there is no
   fresh A-NULL-style paired-seed control run *for this stage*; §6 states explicitly why the
   standing A-NULL result is cited instead and what it does and does not cover). Fires iff any §6
   validity check fails. Meaning: the instrument or harness is unsound; every measurement in this
   stage is void; author call on repair-and-rerun.
2. **REN-OWNS** — satisfying arm: **A-REN**. Fires iff A-REN is TERM-OWNS (both channels). Meaning:
   the renormalization defect is the (or a dominant) identified mechanism for `T_res`; possibility
   (i) (cancellation-within-`T_REN`, §3 of the derivation) is supported at the instrument level;
   the `/physics-change` new-formula slot is written against `W_k`-restoration with this arm as its
   regression test (author-gated as ever). **If A-JREN was also triggered and run, its read is
   reported alongside** — REN-OWNS does not by itself close the question of whether J+REN jointly
   restore *coverage*, only that REN owns the *point* bias.
3. **REN-PARTIAL** — satisfying arm: **A-REN**. Fires iff TERM-PARTIAL in both channels. Meaning:
   REN contributes but does not own; **this is one of A-JREN's two trigger conditions** (§2) — if
   A-JREN was not already triggered by L0-REN-B's R3 = BUDGET-TENSION, it is triggered now and
   returns to the author as the conditional arm's own execution-completeness gate (A-JREN must run,
   or be withdrawn by an author [RULE], before this branch is treated as closed). No repair is
   proposed from a partial read (carried discipline).
4. **REN-INNOCENT** — satisfying arm: **A-REN**. Fires iff TERM-INNOCENT (both channels). Meaning:
   H-REN is refuted at the instrument level — possibility (i) does not hold and the width term does
   not survive to the instrument (consistent with, but not proof of, possibility (ii) or an
   entirely different account); routes to H-SB (L0-SB, this stage's parallel L0 item) as the
   remaining live account for `T_res`, and to the parent's NO-OWNER handling: **mandatory Stage-L
   literature sweep before any further arm** (L0-LIT, already authorized this stage, functions as
   that sweep if REN-INNOCENT fires — no new sweep is separately required).
5. **OTHER / SPLIT** — any remaining outcome (incl. a 1D/2D class split, which the parent §6 marks
   as itself a finding, and incl. the case where A-JREN is triggered but the author withdraws it
   instead of authorizing its run — reported as OTHER with the withdrawal recorded, not silently
   dropped). Reported raw, direction stated, no branch forced.

## 6. Validity and STOP criteria

- **V-M2/V-M3/V-M4** carried verbatim from the parent §5 (generator invariance incl. AR-1..AR-3;
  pin integrity; clean rule). Both arms consume the identical pre-dose realisation discipline;
  A-REN/A-JREN differ from the base **only** in the estimator switch — same discipline A-M2′ and
  A-NULL followed.
- **V-M5** — values golden at the running HEAD against the committed MN0X records, rtol ≤ 1e-12,
  MAPs exactly equal, re-executed before any arm (the D1-13 independent re-execution, carried).
- **Point-branch invariance (constraint (a), unit-tested)** — at σ_z = 0 every row takes the point
  branch (disjoint code, `W_k` never evaluated), so both A-REN and A-JREN are bit-identical to
  `ESTIMATOR_VARIANT_BASE` on any all-point-branch case. This is the arm-specific analogue of
  A-M2′'s own σ=0 inertness test and is required at registration before any seed is consumed.

**Specificity control — reuse decision, made and justified explicitly (per the task brief's
instruction to choose, not default):**

**Decision: cite the standing A-NULL result as the machinery validation; do NOT run a fresh
paired-seed control for this stage.** Justification:

- **Same instrument, same commit family.** A-REN and A-JREN add two more `elif` branches to the
  *identical* `estimator_variant` switch scaffold A-NULL already exercises (`_channel_terms_at_h`,
  `venue_transfer.py`) — same `VenueConfig` field, same threading through
  `log_channel_posteriors_ball_sigma_vector{,_hgrain}`, `_h_task`/`_H_STATE`, and both per-seed
  drivers, same RNG-order discipline (the switch is read-only on already-drawn arrays; no new draw
  is introduced by either new branch). This is not a new instrument; it is one more case of a
  switch already certified end-to-end.
- **What the standing A-NULL result actually certifies:** DS-N1's PASS (stage-2, row #103: 15/15
  paired seeds MAP-index-identical to MN0X, floor-aware integer shift law exact) proves the
  *plumbing* — that selecting a non-base `estimator_variant` (i) consumes no extra/different
  randomness, (ii) reaches the kernel-branch integrand and *only* the kernel-branch integrand
  (point-branch rows provably untouched, since A-NULL's ×1.7 constant would shift them too if the
  switch leaked outside the disjoint kernel-branch code path — it does not), and (iii) is correctly
  threaded through every downstream aggregation (both channels, both grid dimensions, the argmax).
  That is exactly the class of failure a "specificity control" exists to catch — a switch that
  silently perturbs something it should not — and it is a property of the **scaffold**, not of the
  particular arithmetic in any one branch.
- **What it does NOT certify, and what is checked instead:** A-NULL cannot validate that A-REN's
  *specific* arithmetic (`W_k` computed from the correct clip limits, correctly applied to both
  `c1q` and `c2q`, correctly floored) is right — a constant multiplier and a data-dependent
  division are different code, and no inertness argument transfers between them. That correctness
  claim is discharged differently: by the point-branch invariance unit test (constraint (a), above,
  required before any run) and by code-diff review of the exact hunk in §3 against
  `L0_REN_A_DERIVATION_20260815.md` §1's stated `W_k` formula, at registration — the same discharge
  path A-M2′ used for its own Jacobian arithmetic (unit-tested bit-identity at σ=0, plus the
  `ARMS.md` diff-hunk review), not a paired-seed statistical control. A-REN is not expected to be
  inert (unlike A-NULL, whose entire point is a provable no-op) — it is expected to *change* the
  posterior, so an equality-based control is the wrong instrument for it in the first place; A-NULL
  transfers as *harness* validation, not as a template to be re-run.
- **Conclusion:** no NEW paired-seed control is warranted this stage. If this reasoning is
  challenged at review, the fallback is a cheap one — a constant-scale-style control specific to the
  renormalization switch (e.g., forcing `W_k ≡ 1` identically via a debug flag and checking
  bit-identity to base) is a same-cost unit test, not a fresh instrument run, and can be added at
  registration without touching the seed plan.

- **Abort:** (a) non-finite ln_post > 1% ⇒ STOP; (b) horizon-drop > 5% ⇒ STOP; (c) any V-M failure
  ⇒ STOP; (d) point-branch-invariance unit test failure ⇒ STOP (this stage's analogue of stage 2's
  DS-N1 STOP, discharged by the transferred A-NULL machinery argument above plus the dedicated unit
  test rather than a fresh paired run). No toy participates in any rule here, per the toy-calibration
  bar.

## 7. Expected NULLs, pre-registered

- **A-REN landing TERM-INNOCENT** would refute H-REN's possibility (i) (cancellation within
  `T_REN`) at the instrument level — informative, not contradictory; routes to branch 4 (H-SB
  becomes the live account) and does not, by itself, adjudicate possibility (ii) (non-additivity),
  since TERM-INNOCENT on the *single* arm says nothing about the *joint* one.
- **A-REN landing TERM-OWNS** would support possibility (i) and exceed a PARTIAL expectation should
  L0-REN-B's toy have predicted PARTIAL (mirroring stage 2's own "A-M2′ landing TERM-OWNS would
  exceed the §2 expectation" clause) — recorded now so the surprise is legible whenever the
  TBD-pending-L0-REN-B window is filled.
- **A-JREN, if triggered and run, landing anywhere other than "coverage restored + |b| ≤ DS-M1
  TERM-OWNS"** is not a contradiction of A-REN's own read — it is the registered test of
  non-additivity (possibility (ii)) and is reported as such, not folded into A-REN's branch.
- **2D tracks 1D** in both arms; a split forces branch 5, exactly as stage 2.

## 8. Provenance

Parent preregs `73141160` (mechanism isolation) and stage-2 `092b121b`
(`PREREGISTRATION_M2PRIME_ABLATION.md`) · ledger rows #102–#105 (authorizations + rulings,
`results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`) ·
`PROPOSAL_STAGE3_20260815.md` (`c10fddbc`) · `L0_REN_A_DERIVATION_20260815.md` (this stage's frozen
L0-REN-A, HEAD `5df238b9` at derivation time) · stage-2 instrument references: MN0X records
`5b0bd17a`, AM2P/ANULL data + readout `e49f7570`, campaign decision cell `d45fbf15`, realized cost
anchor 0.969 CPU-h/seed (runbook §4) · seed-block verification performed against
`darksiren_emri/validation/venue_transfer.py` (`RESERVED_SEED_OFFSET_BLOCKS`, `MECH_CELL_SPECS`,
`M2P_CELL_SPECS`), `darksiren_emri_test/validation/test_venue_transfer.py`,
`darksiren_emri_test/validation/test_m2prime_ablation_arms.py`, `cluster/datasets.yaml`, and
`results/campaign51_20260728/PILOT3_READOUT.md` (cross-namespace disclosure, §2). **This document
is a DRAFT.** It carries no registering commit, reserves no seed, and authorizes no run. Registered
documents are append-only from their registering commit onward; this one has none yet.
