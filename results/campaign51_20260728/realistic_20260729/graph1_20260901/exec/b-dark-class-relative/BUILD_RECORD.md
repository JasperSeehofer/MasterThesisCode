# b-dark-class-relative — BUILD_RECORD

Docket ruling R8 (rows #337/#345, batch-2 grant). Builder-only node: no edit under
`darksiren_emri/`, no cluster. Deliverable: `dark_class.py` (shared relative-criterion
helper) + this record. Migration of the ~30 consuming scripts is a separate [DO], not done here.

## 1. Threshold derivation

Two CSVs (named in `exec/m-s0b-production/CLASS_COUNT_FORENSICS.md`):
`then` = `headreadout_20260827/iiib/event_likelihoods.csv` (h=0.73 rows), `now` =
S0-B truth node `.../node_truth_iiib_sites2.2_nosmear/.../event_likelihoods.csv`. 157 events
moved dark→matched under the exact-zero test. Recomputed directly (not taken from the memo):

- Max ratio `L_cat_no_bh/combined_no_bh` among the 157 moved events (on `now` data): **9.751433e-07** (event_idx 393).
- Min: 9.87e-109.
- **This exceeds 1e-7.** Per the ticket's instruction, checked for a data gap: the ratio
  distribution across the full population, inspected point-by-point in `(1e-9, 1e-3)`, is
  **continuous/log-smooth with no bimodal gap** near 1e-6 — genuinely-matched, run-stable
  events (e.g. event_idx 14, ratio 1.86e-21, bit-identical `L_cat_no_bh` in both `then` and
  `now`) sit at ratios far below the 157-moved-event cluster, and the population runs smoothly
  from there up through 1e-4 with no step. **No natural gap exists to place a threshold in.**
- `THRESHOLD = 1e-6` (as instructed) is adopted as a **margin call, not a rediscovered gap**:
  it clears the 157-event max by only ~2.6% headroom, not the ≥1e3 originally hoped for. It is
  kept because it (a) is the value the ruling names, (b) reproduces the pre-drift 606/982 split
  on the 08-27 file (§3), and (c) any looser value pulls in more of the smooth continuum
  arbitrarily — moving the line further is a scientific call for the author, not this node.

## 2. Class counts, both criteria, four files (h=0.73 rows)

| file | exact-zero dark/matched | relative dark/matched | labels differ |
|---|---|---|---|
| 2026-08-27 iiib head readout | 606/982 | 727/861 | 121 |
| S0-B truth node | 449/1139 | 723/865 | 274 |
| 2026-09-02 re-baseline iiib | 606/982 | 1241/347 | 635 |
| 2026-09-02 re-baseline joint_r1 | 493/1095 | 967/621 | 474 |

Cross-file symmetric difference of the **relative** label:

| pair | \|Δ\| |
|---|---|
| 08-27 Δ S0-B truth | **4** |
| 08-27 Δ re-baseline iiib | 514 |
| 08-27 Δ re-baseline joint_r1 | 242 |
| S0-B truth Δ re-baseline iiib | 518 |
| S0-B truth Δ re-baseline joint_r1 | 246 |
| re-baseline iiib Δ re-baseline joint_r1 | 278 |

**Honest read:** across the specific 08-27→S0-B pair this node was built to fix, the relative
label is now very stable (Δ=4, vs. 157 exact-zero flips) — the fix works for the drift it was
built for. Across the two 09-02 re-baseline files, agreement is NOT good (Δ in the
hundreds) — `re-baseline iiib` alone has a median `L_cat_no_bh/combined_no_bh` of 3e-25 across
its full population (not just near-threshold events), meaning far more of its events sit deep
in the tiny-ratio zone than in either the 08-27 or S0-B files. That is a real population-level
difference between runs (config/version, not evaluated further here — out of scope, no cluster,
no physics edits), not a defect in the helper. **The relative criterion is calibrated only for
the flip it targets; it is not shown stable as a universal drop-in across arbitrary production
configs and should not be assumed so without a fresh per-run check.**

## 3. g-byte-id

- **No physics number changes.** `dark_class.py` only reads `L_cat_no_bh`/`combined_no_bh` and
  returns a boolean label; it writes nothing back to any CSV and calls no code under
  `darksiren_emri/`. Trivially true by construction (module docstring, no I/O side effects).
- **Pre-drift 606/982 recovery on the 08-27 file:** the relative criterion gives **727/861**
  on 08-27, not 606/982. It does **not** reproduce the pre-drift split — the relative
  criterion is a different, broader rule than "exact zero, but tolerant of the 157-event
  underflow flip": at threshold 1e-6, 121 additional 08-27 events (already non-zero but with
  ratio < 1e-6, e.g. genuinely tiny matched events) are pulled into "dark" beyond the 157 this
  node targets. **Reporting what the data say, not what was hoped:** the criterion stabilizes
  the specific 08-27→S0-B flip (§2) but is not label-identical to the legacy criterion on
  either file alone.

## 4. Migration list (NOT applied — separate [DO])

Every active (non-comment, non-docstring) `L_cat_no_bh == 0` / `== 0.0` test found via
`grep -rn` under `results/` and `scripts/` (excluding the `snap_d4765539` frozen verification
snapshot, which should not be touched, and this node's own files):

| script | line(s) |
|---|---|
| `results/campaign51_20260728/realistic_20260729/attack_c3_c4_allruns.py` | 134 |
| `results/campaign51_20260728/realistic_20260729/attack_c3_c4.py` | 88 |
| `results/campaign51_20260728/realistic_20260729/p3_wbhzero_measure.py` | 991 |
| `results/campaign51_20260728/realistic_20260729/cmem_reads.py` | 164–165 |
| `results/campaign51_20260728/realistic_20260729/p3_2d_probe.py` | 93 |
| `results/campaign51_20260728/realistic_20260729/crossterm_instrument/m4_shared_galaxy_census.py` | 251 |
| `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b4_imp_stage1_forecast.py` | 160 |
| `results/campaign51_20260728/realistic_20260729/fanout1_20260829/build_b4_imp_decomposition.py` | 138, 162 |
| `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b3_1_pop_measure.py` | 215 (the anchor script named in the ruling) |
| `results/campaign51_20260728/realistic_20260729/fanout1_20260829/b1_1_forensic_work/f1_csv_audit.py` | 38 |
| `results/campaign51_20260728/realistic_20260729/fanout1_20260829/verifier_pass/item2_rederive.py` | 68 |
| `results/campaign51_20260728/realistic_20260729/gate_b_20260730/g5_leg_consistency.py` | 77 |
| `results/campaign51_20260728/realistic_20260729/gate_b_20260730/wbhzero_gate_b_scripts/counterfactual_symmetric.py` | 44 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness.py` | 405 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/T1-2_rederive.py` | 133 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/T1-4_matched_scope.py` | 16 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/T2-2_gates.py` | 26 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/T1-4_hist_check.py` | 23 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/T2-16_rederive.py` | 202 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/T2-2_rederive.py` | 114 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/T1-4_rederive.py` | 97 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/T1-4_classrule.py` | 10–12 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/T1-4_hist_method.py` | 7 |
| `results/campaign51_20260728/realistic_20260729/tree2_20260830/full_verification_20260831/work/D-5_rederive.py` | 110 |

Not migrated (frozen artifacts, out of scope): every `.../full_verification_20260831/work/
snap_d4765539/**` path (a sealed verification snapshot; retroactive edits there would
invalidate its provenance).

## Recommendation

Migration is a **separate [DO]**: swap each `== 0` test for `is_dark_relative(...)` from this
node's `dark_class.py`, then re-run each script's own regression/consistency check (most of the
table above are one-off exec/verifier scripts with no test suite — re-running and diffing the
printed output is the practical check). Given §2's finding that the relative criterion is
**not** a drop-in match to the legacy label on any single file (only stable across the specific
drift it targets), each migrated script's numeric output should be expected to shift, and that
shift should be read out per-script before trusting it, not assumed benign.
