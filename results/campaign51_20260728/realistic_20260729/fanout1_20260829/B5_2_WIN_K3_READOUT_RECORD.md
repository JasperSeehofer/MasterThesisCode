# B5.2 [WIN] k=3 counterfactual — independent readout (arm C3)

**Launched under rows #222/#223 — charter node B5.2. Read out 2026-08-29 by the independent
reader.** This record is a re-derivation of every number below from the retrieved cluster
outputs and the retrieved logs — not a restatement of any builder's prior claim. Registration
of record: `PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md` (job `6738999`, commit
`ff230621`).

## Bottom line, in one paragraph

The cluster job ran cleanly (all 4 tasks completed, correct code commit, correct output shape)
and all three hard gates pass (1D channel untouched, the switch reaches the production dispatch
path, the stencil is well-behaved). The headline number — how much switching the mass window
from linear to a bounded log shape would move the H₀ estimate — comes out to **+0.0035**, which
is technically in the registration's own "no verdict" zone: bigger than the "this is basically
nothing" line (0.003) but nowhere near the "this is a real effect" line (0.008), so per the
registration's own rules this is **reported, not adjudicated**. The more important finding is
about *why* the window change was worried about in the first place: a companion analysis (the
"mirror fleet") predicted the log window would cost ~17-21 percentage points of true-host
retention — i.e. that switching would make the estimator lose track of the actual host galaxy
for a large fraction of events. On the real production catalogue, **that loss did not happen at
all** — the true-host recovery rate is bit-for-bit identical (66 of 76 known hosts, both before
and after the switch). The window change instead entirely reshapes the *impostor* pool (621 of
1588 events lose all their non-true-host catalogue candidates), which is a different and, on
this evidence, more benign effect than the one the physics-change document was worried about.
The job also ran about 10-30× cheaper than budgeted (5 CPU-hours actually used against a
44-137 CPU-hour estimate) — comfortably inside the compute ceiling.

## What was checked and what came back

### 1. Job completion and provenance

| item | result |
|---|---|
| All 4 array tasks (h = 0.660, 0.665, 0.670, 0.730) | COMPLETED, exit 0:0 |
| Code commit (all 4 tasks) | `ff2306213e9e65abbd474f66348bc05a6f3e6547` — matches the registration's A22 launch stamp |
| Output shape | 1588 events × 4 h-nodes = 6352 rows, 19 columns (matches the C0 gate's corrected column count) |
| **Flag** | Provenance stamps record `tree_dirty_file_count=296` at run start on all 4 tasks — this contradicts the A22 stamp's claim of a clean tree at launch. The commit hash is unambiguous and correct everywhere checked; only the "clean" claim is unsupported by the retrieved evidence. |

### 2. Gates (registration §6)

| gate | registered condition | measured | verdict |
|---|---|---|---|
| **R6** (1D bit-identity) | `L_cat_no_bh`/`combined_no_bh` ≤1e-12 relative, every H4 node | max rel. diff **2.67e-14** | **PASS** |
| **R2** (engagement) | ≥0.90 of non-empty-baseline with-BH events show a changed `L_cat_with_bh` at h=0.730 | **0.968** (951/982) | **PASS** |
| **R5** (stencil validity) | \|Δℓ''\| ≪ I_HEAD=2965 | **63.7** (ratio 2.1%) | **PASS**, not ambiguous — no G27 escalation |
| **R1** (retention falsifier) | production true-host retention (2D) within [0.762, 0.816] of the mirror's 0.789 prediction | **0.8684** (66/76), identical arm vs. baseline | **FALSIFIED** — outside the band (see §3) |
| R1 (growth-factor sub-check) | production candidate-growth distribution within ±2pp of mirror's median/p95/max | **UNDETERMINED** — no per-event candidate-count data was retrievable at zero additional compute |

### 3. The retention falsifier — what actually happened, and why it matters

The mirror fleet (a synthetic validation set with 100% known true hosts) predicted that
switching to the log/k=3 window would cost ~17-21 points of true-host retention. On the real
iiib production events, only 76 of 1588 events even have a known true host at all (the rest are
"dark class" — no catalogued host). Among those 76:

- **Baseline (linear, k=1.5):** true host recovered in 66/76 (86.84%)
- **Arm T (log, k=3):** true host recovered in 66/76 (86.84%) — **identical**, same 76-event
  population, same positivity pattern for every single event (checked, not assumed)

So the retention-loss mechanism the physics-change document was worried about **does not occur
on this venue at all**. The 621 events whose with-BH catalogue support collapses to zero under
the log window are **all** dark-class (no known true host) — none of the 76 known-host events
lost or gained candidate-set membership. This is a real, checked finding (joined against the
CRB's own `host_galaxy_index` column), not an inference from the mirror.

Because the falsifier failed *with* a documented, independent mechanistic explanation (rather
than an unexplained instrument defect), the registration's attribution falsifier is not
triggered — the ΔMAP number below can still be attributed to the mass-window geometry change,
just not via the true-host-loss channel the physics-change document flagged. It is an
impostor/dark-class suppression effect instead.

### 4. Primary reading — Δmean_h,pred

Computed directly from `Σ ln(combined_with_bh)` at each H4 node, arm vs. baseline, using the
registration's own stencil (central difference over {0.660, 0.665, 0.670}, `Δmean_h,pred =
Δℓ'(0.665)/I_HEAD`, `I_HEAD=2965`):

| h | Δℓ(h) = Σ ln(L^T/L^B), with-BH channel | no-BH channel |
|---|---:|---:|
| 0.660 | +0.5442 | 0 (bit-identical, R6) |
| 0.665 | +0.5972 | 0 |
| 0.670 | +0.6486 | 0 |
| 0.730 | +1.2143 | 0 |

`Δℓ'(0.665) = 10.444 nats/h`, `Δℓ''(0.665) = -63.7` ⇒ **Δmean_h,pred = +0.003523**.

An informal 3-point local-vertex cross-check on the absolute (not differenced) with-BH
log-likelihood gives an independent local peak shift of +0.00309 — same sign, same order of
magnitude, corroborating the stencil result without adding new information.

**Verdict per the registered map: INTERMEDIATE** (0.003 < 0.003523 < 0.008 = T_mat). This is
explicitly a non-verdict under the registration's own rules — bigger than the
IMMATERIAL-CONSISTENT-WITH-HB line by about 17%, but at 44% of the MATERIAL line. Sign is
**up** (toward truth 0.73), the same direction as HB's own +0.0015 and about 2.3× its size.

`Δw̄₂` (mean with-BH mixture weight `alpha_G_phi`): **exactly 0** — this quantity is a global,
h-dependent-only normalization untouched by the mass window by construction, not merely small.

`ΔT` (score-at-truth tilt): **not computable** — arm T's H4 grid does not include the h=0.725/
0.735 nodes the registered stencil needs; would require additional compute.

### 5. Cost (F4)

| | value |
|---|---|
| Registered estimate | 44–137 CPU-h |
| Measured (sacct) | **4.97 CPU-h** (4 tasks × ~1.22–1.29 CPU-h each, ~4.5–4.8 min wall at 16 cpus) |
| Ratio | 9×–28× below the estimate |

### 6. The 5.2 adoption rule (all three ANDed)

1. **H₀ delta immaterial or argued-benign-if-material-up** — NOT ADJUDICATED: INTERMEDIATE.
2. **Candidate growth inside the compute ceiling** — SATISFIED (4.97 CPU-h ≪ 50–130 CPU-h).
3. **True-host retention loss argued as physically right, or design returns** — SATISFIED FOR
   IIIB (there is no retention loss on this venue to argue about); says nothing about other
   venues, which this registration excluded from scope.

**Overall: NOT YET GRANTED.** Conditions 2 and 3 favor adoption on iiib; condition 1 is an open,
reported non-verdict pending the wave-3 full-grid (G41) read or an author ruling on how to treat
an INTERMEDIATE primary read.

## Caveats

- Cluster SSH access dropped mid-session (control connection could not be re-authenticated
  non-interactively); C0's own sacct timing could not be re-pulled for a full wall-time
  comparison (its P6 host-recovery line and commit hash had already been captured beforehand).
- R1's growth-factor sub-check and R3 (score-at-truth) are reported as undetermined /
  not-computable rather than approximated.
- The tree-dirty-count discrepancy (296, vs. the A22 stamp's "clean") is flagged, not resolved.
- Findings are iiib-specific; joint_r1 is out of this registration's scope by its own §1.

## Files

- `b5_2_readout.json` (this directory) — every number above with `{value, source, date}`.
- `PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md` — the registration of record; this
  readout appends a `⟨SUBMIT⟩`/RESULT record there (append-only).
- Retrieved cluster outputs:
  `results/campaign51_20260728/realistic_20260729/wave2_20260829/c3/` (diagnostics, posteriors,
  posteriors_with_bh_mass, run_metadata, logs, GIT_COMMIT_AT_RUN.txt).

**Stamp:** read out 2026-08-29 by the independent reader; launched under rows #222/#223 —
charter node B5.2.
