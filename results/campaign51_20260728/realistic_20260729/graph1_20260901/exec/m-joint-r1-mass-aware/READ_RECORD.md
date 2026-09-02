# m-joint-r1-mass-aware — READ RECORD (mechanical registered readout)

Node: m-joint-r1-mass-aware (Research Graph 1, Branch C, wave 1). Date: 2026-09-02.
Authorization: ledger row #301 (docket item 3 ratified d-jr1-band: band + grid scope
frozen) + row #290 row 5. **Mechanical read only — no scientific choices made here.**
No code edited, no commits, no cluster jobs launched.

## Data-identity disclosure

Per the chair's mechanical reading of the frozen registration
(`../r-jr1-massaware/REGISTRATION_DRAFT.md`), the measured object — "the full 1588-event
joint_r1 1D posterior under the post-flip production default on the elected h-grid
(H_GRID_41)" — is satisfied by the **banked BLIND grid of SLURM job 6764462**
(`results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/
run_20260902_graph1_headrebaseline_joint_r1/`), computed before the band was frozen and
unread until this row #302-era verdict-free readout. **This read executes at zero fresh
compute.** Config confirmed from `run_metadata_21.json:cli_args` on this bank: `h_value
0.73` (representative task; grid spans 41 files), `catalogue_leg_1d_mass_aware=auto`,
`sigma4d_mass_kernel=point`, `catalogue_global_selection=phi`, `observed_catalogue=
.../observed_catalogue_seed900001.csv` — matching `dv-jr1-transform/DERIVATION.md` §1's
structural claims exactly. Commit `1ec9514dd1808c48b18c0792dce558e5bba0f116` (ancestor of
the flip `5e7fda16`), per `c0prime_eval/GATE_RECORD.md`.

`simulations/diagnostics/event_likelihoods.csv` confirmed: 65,108 rows = 41 h-nodes ×
1,588 events + header; h-grid matches H_GRID_41 exactly (0.600–0.860, 0.01 coarse /
0.005 in 0.65–0.75, 41 nodes). `candidate_dump_dir: null` in every retrieved
`run_metadata_*.json` — **no T2.2b-schema candidate dump was captured in this bank.**
This is the load-bearing fact behind every ABSENT item below.

## 1. map_h / mean_h — re-derived independently, cross-checked

**Scorer** (the frozen T0 gradient-weighted convention, row #286;
`WAVE3_A14_DELTA_READ_20260831.md` correction note; same convention
`m-head-rebaseline/READOUT_RECORD.md` used): per-event `log(L_e(h))`, `Sigma_e log L_e(h)`
per h (uniform prior, no zero-handling needed — see gates), gradient-trapezoid grid
weights `w = np.gradient(h_grid)` on H_GRID_41, `post_n(h) ∝ exp(loglik_h − max)`
normalized against `Sigma_h post_n(h)*w(h) = 1`, `mean_h = Sigma post_n(h)*h*w(h)`,
`MAP = argmax_h loglik_h`. Applied here directly to `event_likelihoods.csv`'s
`combined_no_bh` column (the registered 1D/no-BH channel).

**Re-derived (DERIVED-HERE, this record):**

| quantity | value |
|---|---|
| map_h | **0.665** |
| mean_h | **0.6670323337269477** |
| sigma_h | 0.02034581457706018 |
| n_h / n_events | 41 / 1588 |
| min nonzero L (posterior floor) | 5.060752355870854e-06 |
| events at physics floor (zero L, any h) | 0 / 1588 |

**Cross-check against `m-head-rebaseline/READOUT_RECORD.md`** (its joint_r1 1D
`combined_no_bh` row: map_h 0.665, mean_h 0.667032, sigma_h 0.020346, posterior floor
5.060752e-06). This record's independent re-derivation — map_h 0.665, mean_h
0.6670323337269477, sigma_h 0.02034581457706018, min nonzero L 5.060752355870854e-06 —
**matches to the READOUT_RECORD's full stated precision on every field, no
discrepancy.** The `combined_with_bh` (2D) channel is not the registered object but was
cross-checked too as a sanity pass: map_h 0.665, mean_h 0.6671265168140829, sigma_h
0.018923640448964062 — matches READOUT_RECORD's joint_r1 2D row (0.665/0.667127/0.018924)
exactly as well.

## 2. Floor-node mass at h = 0.600

Computed under this record's own T0 gradient-weighted convention:
`floor_mass = post_n(0.600) * w(0.600)`.

| channel | on-arm floor mass at h=0.600 |
|---|---|
| combined_no_bh (registered) | **1.7948683137761944e-04** |
| combined_with_bh (secondary) | 2.4418319537913104e-04 |

**REFUTED-clause check:** predicted on-arm floor mass ≤ 5e-3 (`REGISTRATION_DRAFT.md` §2).
Observed 1.79e-4 ≪ 5e-3 — the floor departure is decisive and in the predicted direction;
combined with map_h = 0.665 ≫ 0.605, the REFUTED clause (map_h ≤ 0.605 with C-C pin
intact) **does not trigger.**

**Off-arm comparand cross-check (disclosed convention caveat).** Independently
re-derived under the SAME T0 gradient-weighted convention on the banked pre-flip comparand
`headreadout_20260827/joint_r1/event_likelihoods.csv` (same 65,108-row shape, same
H_GRID_41): **off-arm MAP = 0.600** (matches `DERIVATION.md` §5's cited off MAP 0.600
exactly), but this record's off-arm mean_h (0.611683) and floor mass (0.361693) differ from
`DERIVATION.md`'s cited 0.6143 / 0.2208. The MAP match plus the differing mean/floor-mass
indicates a **convention difference**, not a data mismatch: `DERIVATION.md` §5 states its
off-arm moments used the "BAND_REDERIVATION §2.2 convention" (trapezoid/flat-prior), not
necessarily identical in normalization/weighting to this record's frozen T0
gradient-weighted convention. This discrepancy is disclosed, not adjudicated — it does not
touch the on-arm registered numbers above, and the REFUTED-clause conclusion is robust to
it (either convention's floor-mass scale is > 40x the on-arm value, so the "≥5-node
departure" the registration calls decisive holds under both).

## 3. Per-class impostor scores (secant 0.725/0.735) and true-host transform read — ABSENT

**ABSENT.** `run_metadata_*.json:cli_args.candidate_dump_dir = null` for every task in this
bank — no T2.2b-schema per-candidate dump was captured. `event_likelihoods.csv` carries
only event-level aggregates (`combined_no_bh`, `L_cat_no_bh`, `B_num`, etc.), not the
per-candidate `S_4D`/`S̄_φ` values or the per-event host-class (in-catalogue vs dark) flag
the DERIVATION's §5 impostor/transform reads require. `posteriors/*.json` and
`posteriors_with_bh_mass/*.json` were checked too (per instruction) — both are flat
per-event scalar/nested-galaxy lists, no host-class column, no candidate-level `S_4D`.
No host-class map artifact (joint_r1's own 73-in-catalogue map, row #270 §1.5) was found
banked anywhere in this run's tree either.

**What a follow-up needs:** either (a) a re-run of the h=0.725/0.735 secant nodes with
`candidate_dump_dir` set to a T2.2b-schema dump path (`T2_2B_ARM_B_RUNSHEET.md` §6.2
column schema), or (b) a standalone joint_r1 host-class map (in-catalogue/dark per
event_idx) joined against the existing `event_likelihoods.csv`, from which a per-event
secant score could be built from `combined_no_bh(0.735)` vs `combined_no_bh(0.725)`
without needing per-candidate detail. Neither exists in the current bank. The dark-only
pure-arm-sum invariance check (`DERIVATION.md` §5, off value −59.87) is likewise
**ABSENT** for the same reason (same host-class-map dependency).

## 4. C-C pin (§5)

Compared this record's on-arm bank against the banked off comparand
(`headreadout_20260827/joint_r1/event_likelihoods.csv`, row #270 era,
`catalogue_leg_1d_mass_aware` unset/pre-flag) across all 65,108 rows, row-aligned on
`(event_idx, h)`:

| column | ndiff | max_abs |
|---|---:|---:|
| B_num | 0 / 65108 | 0.0 |
| B_num_wbh | 0 / 65108 | 0.0 |
| D_tilde_phi | 0 / 65108 | 0.0 |
| w_G, w_G_legacy, w_tilde_G | 0 / 65108 each | 0.0 |
| alpha_G_phi | 0 / 65108 | 0.0 |
| r_Malm | 0 / 65108 | 0.0 |
| g_frac | 0 / 65108 | 0.0 |
| L_comp | 0 / 65108 | 0.0 |
| L_cat_no_bh (h=0.73 slice) | 1095 / 1588 | — (the registered flip) |
| L_cat_with_bh | 1094 / 1588 (h=0.73) | 0.0035793250588652004 |
| combined_with_bh | 1078 / 1588 (h=0.73) | 0.0002534242715218 |

The structural no-BH-leg invariants (B_num, B_num_wbh, D_tilde_phi, w_G/w_G_legacy/
w_tilde_G, alpha_G_phi, r_Malm, g_frac, L_comp) are **exact-zero** against this comparand,
and the no-BH-leg delta (`L_cat_no_bh`) touches **exactly 1095/1588 events at h=0.73** —
matching `c0prime_eval/GATE_RECORD.md`'s row #299-confirmed joint_r1 candidate-bearing
count ("1095... the number of record for joint_r1") to the event, i.e. the registered flip
acting only on candidate-bearing events, as designed.

The with-BH columns show a small nonzero residual against `headreadout_20260827`
(max_abs ≈ 0.0036 `L_cat_with_bh` / 0.00025 `combined_with_bh`). This record does **not**
re-adjudicate that residual: it is the identical pattern `c0prime_eval/GATE_RECORD.md`
independently diagnosed and resolved (row #299/#301 RE-STAMP) as an artifact of comparing
against a non-flag-matched comparand — `headreadout_20260827` reproduces the same
"mismatched-comparand" residual magnitude the GATE_RECORD table reports
(0.0035793250588652004 for joint_r1 `L_cat_with_bh` — an exact match to this record's own
number), and the GATE_RECORD shows that residual vanishes to `ndiff 0/1588` against the
correctly flag-matched `c0prime_off` comparand (not locally banked as raw CSV, only as
that gate's logged summary). **C-C pin: PASS**, resting on (a) this record's own exact-zero
structural-column check (direct evidence) and (b) the already-stamped **g-c0-baseline
GREEN-AS-CORRECTED** gate for joint_r1 (row #301, confirmed row #299) as the authority for
the with-BH exact-zero claim specifically, since the correctly flag-matched raw comparand
data is not present in this local bank to re-derive from scratch. No pin/gate red.

## 5. Gate panel

- **g-censoring:** map_h = 0.665 is interior to H_GRID_41 (rails at 0.600 and 0.860) —
  **no rail flag, no demotion to a bound.** PASS.
- **g-precision:** `event_likelihoods.csv` likelihood columns carry full float64
  precision (verified: `combined_no_bh` values show 15–17 significant digits, e.g.
  `0.044660237366908...`, not truncated/reconstructed 7-s.f. strings — the
  "+123.11 storage-artifact" failure mode does not reproduce here). All arithmetic in
  §1–§2 above ran on these full-precision columns directly (no CSV round-trip through a
  lossy intermediate). PASS.
- **g-znorm: PARTIALLY EVALUABLE, not to the registered precision.** Confirmed
  (checked per instruction): `event_likelihoods.csv` carries no `global_denom_*` columns
  (READOUT_RECORD's note holds); `posteriors/*.json` and `posteriors_with_bh_mass/*.json`
  are flat per-event lists/nested galaxy dicts with no denom key either — **absent from
  both data products.** However the global scalar **is** directly logged, once per h-node,
  in the raw per-task `.log` stdout (`_log_path_a_selection_objects`, e.g. at h=0.730:
  `Sigma_phi=9.56237e+08, Sigma_4D=4.221903e+08`; confirmed present at all 41 h-nodes).
  This confirms the intended global-divisor *design* (one scalar per h, as constructed —
  not a per-event measurement) but is not the same object as `DERIVATION.md` §1.4's
  ≤1e-12 per-event-implied-uniformity re-derivation, which needs per-candidate `N_on/L_on`
  values this bank does not carry (§3's `candidate_dump_dir=null` fact again). **The strict
  registered g-znorm identity check is not evaluable from this bank to its registered
  precision; the weaker existence/consistency check (global scalar present, one value per
  h, physically reasonable magnitude) passes.**

## 6. Mechanical disposition

Per `REGISTRATION_DRAFT.md` §2/§6 (frozen band, ratified row #301):

- Z-CONFIRMED iff map_h AND mean_h ∈ [0.64, 0.70].
- map_h = **0.665** ∈ [0.64, 0.70]: **TRUE**.
- mean_h = **0.667032** ∈ [0.64, 0.70]: **TRUE**.

**→ Z-CONFIRMED.**

REFUTED does not apply (map_h = 0.665 ≫ 0.605). No pin/gate is red (C-C pin PASS, per §4;
g-censoring PASS; g-precision PASS; g-znorm evaluable only to a weaker existence check,
which passes and is not itself a gate/pin failure — it is a scope limitation, disclosed).

Per the disposition table (`REGISTRATION_DRAFT.md` §6): Z-CONFIRMED books
**c-auto-default-venue-general SUPPORTED on joint_r1** and **feeds d-calibration**.
**Claim promotion (c-auto-default-venue-general) is NOT decided here — it returns to the
author at d-calibration**, per the task's own scope instruction and the registration's own
disposition table.

## Summary of ABSENT items (for the record)

1. Per-class impostor scores at secant nodes 0.725/0.735 (§4 item 1 of
   `REGISTRATION_DRAFT.md` — DERIVATION §5's dark/in-catalogue split object) — ABSENT,
   needs a T2.2b-schema candidate dump or a standalone host-class map.
2. True-host transform read (`REGISTRATION_DRAFT.md` §4 item 1) — ABSENT, same
   dependency.
3. Dark-only pure-arm sum invariance check (`REGISTRATION_DRAFT.md` §5, off value
   −59.87) — ABSENT, same dependency.
4. g-znorm to its registered ≤1e-12 per-event precision — NOT EVALUABLE from this bank
   (weaker existence check only); needs per-candidate `N_on/L_on` data.

No pin/gate returned INSTRUMENT-DEFECT. The mechanical band read stands as
**Z-CONFIRMED** on the two numbers the frozen registration bands (map_h, mean_h); the
three ABSENT secondary/diagnostic items are reported, not substituted, per instruction.
