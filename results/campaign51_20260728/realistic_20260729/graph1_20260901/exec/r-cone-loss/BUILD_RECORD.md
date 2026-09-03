# r-cone-loss — BUILD RECORD (node b-cone-scorer)

Node: `b-cone-scorer` (sonnet/medium). Research Graph 1, Branch H, wave 3. Registration
draft (unchanged): `REGISTRATION_DRAFT.md` (this directory), §7 "Launch block".

## What was built

`cone_loss_reads.py` — the r-cone-loss scorer, implementing the registration draft's
§2 (statistic), §5 (gates G-1..G-4), and §7 (launch block: CLI flags, `--dry-run`
contract) verbatim. Per the draft's own instruction ("builder runs ONLY `--dry-run`
(G-1…G-4 + census, no scores); a DIFFERENT agent runs the statistic"), this build
session ran **only** `--dry-run`. The registered statistic (Δh_cone, φ_cone, SE, Z,
the leave-out cross-check, the harness Δs replicate) is **not implemented as a
callable path** beyond a `NotImplementedError` stub in real mode — a different agent
must write and run that path, per the verifier-independence contract stated in the
script's module docstring.

Cone geometry (chord/radius) is reused line-for-line from the
`cmem_a1.py`/`cmem_reads.py` precedent in this tree (same Σ' construction, same
Jacobian, same `_polar_to_cartesian` embedding) — not re-derived. `GalaxyCatalogueHandler`
is the same production loader Pipeline B uses. No `darksiren_emri/` source was
touched; no pipeline run; no cluster job.

One addition beyond the draft's literal G-1..G-4 list: gate **G-4's** "χ²₂-distributed
Mahalanobis²" clause is implemented as a closed-form 2×2-inverse Mahalanobis² of the
raw (Δφ, Δθ) sky offset under Σ' (the un-scaled Fisher sky sub-block), which is
algebraically identical to the J-scaled offset's Mahalanobis² under `J Σ' Jᵀ` (J is
diagonal and invertible, so it cancels — documented in the script's module docstring
and the `sky_mahalanobis2` docstring). This is a mechanical realization of the
draft's own formula, not a new choice.

**Files touched:** `cone_loss_reads.py` (this build) and this record. `cone_loss_work/`
(gates JSON + census CSV) and `cone_loss_result.json` (the `--out` path from the launch
block) were also written, as artifacts of running `--dry-run` per the launch block's
own command line — not source changes.

## Quality gates

```
uv run ruff check results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_loss_reads.py
  -> All checks passed!
uv run mypy results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_loss_reads.py
  -> Success: no issues found in 1 source file
```

## Dry-run invocation (verbatim from REGISTRATION_DRAFT.md §7, `--dry-run` appended)

```
uv run python results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_loss_reads.py \
  --production-crb results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv \
  --production-run results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib \
  --replicate-run results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1 \
  --harness-root results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip --population 200 \
  --anchor-fleet-mker results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825 \
  --anchor-fleet-cmem results/campaign51_20260728/realistic_20260729/p3_b0_work \
  --sky-cone-k 1.5 --h-lo 0.725 --h-hi 0.735 --h-true 0.73 \
  --crb-md5 9a1f2a14384a9281c97ca3be312ddaab --catalogue-md5 c52c13b5cab61f6b3f04bbe202550969 \
  --out results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/r-cone-loss/cone_loss_result.json \
  --dry-run
```

Exit code: **0** (confirmed via `echo $?` immediately after the run). Per the task's
`--dry-run` contract, exit 0 holds regardless of gate pass/fail — a gate STOP inside
`--dry-run` is reported information for the launch decision, not a script crash; the
`--out` JSON's own `"verdict"` field carries the gate outcome (`GATES-GREEN` or
`INSTRUMENT-DEFECT`). Real mode (`--dry-run` omitted) still hard-stops (nonzero exit)
on a gate failure, before any statistic is attempted, matching the `cmem_a1.py`
precedent for the non-dry-run path.

## Dry-run output (verbatim)

```
Building sky-cone census + running gates G-1..G-4...
GATES: {
 "g1_catalogue_pin": {
  "path": "/home/jasper/Repositories/darksiren-emri/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv",
  "md5": "c52c13b5cab61f6b3f04bbe202550969",
  "expected": "c52c13b5cab61f6b3f04bbe202550969",
  "passed": true
 },
 "g1_crb_pin": {
  "path": "results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv",
  "md5": "9a1f2a14384a9281c97ca3be312ddaab",
  "expected": "9a1f2a14384a9281c97ca3be312ddaab",
  "passed": true
 },
 "g1_git_commit_pin": {
  "checks": {
   "production": {
    "path": "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/GIT_COMMIT_AT_RUN.txt",
    "commit": "1ec9514dd1808c48b18c0792dce558e5bba0f116",
    "expected_prefix": "1ec9514d",
    "passed": true
   },
   "replicate": {
    "path": "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_joint_r1/GIT_COMMIT_AT_RUN.txt",
    "commit": "1ec9514dd1808c48b18c0792dce558e5bba0f116",
    "expected_prefix": "1ec9514d",
    "passed": true
   }
  },
  "passed": true
 },
 "g2_anchor_mker6": {
  "path": "results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bc_900121_work/seed900121/simulations/prepared_cramer_rao_bounds.csv",
  "expected": {
   "fleet_arm_seed": "bc_900121_work", "seed": 900121, "event_idx": 20,
   "chord": 0.00167466, "chord_tol": 5e-10,
   "radius": 0.0014956979545757095, "radius_tol": 1e-15
  },
  "found_chord": 0.001674659860716462,
  "found_radius": 0.0014956979545757095,
  "expected_chord": 0.00167466,
  "expected_radius": 0.0014956979545757095,
  "chord_ok": true,
  "radius_ok": true,
  "passed": true
 },
 "g2_anchor_cmem_a1": {
  "path": "results/campaign51_20260728/realistic_20260729/p3_b0_work/bc_900101_work/seed900101/simulations/prepared_cramer_rao_bounds.csv",
  "expected": {
   "fleet_arm_seed": "bc_900101_work", "seed": 900101, "event_idx": 0,
   "chord": 0.0116656941007181, "chord_tol": 5e-10,
   "radius": 0.0359121946154451, "radius_tol": 1e-15
  },
  "found_chord": 0.01166569410071811,
  "found_radius": 0.035912194615445196,
  "expected_chord": 0.0116656941007181,
  "expected_radius": 0.0359121946154451,
  "chord_ok": true,
  "radius_ok": true,
  "passed": true
 },
 "g2_passed": true,
 "g3_join": {
  "n_total_crb_rows": 1590,
  "scored_set_size": 1588,
  "n_in_catalogue": 76,
  "n_out": 10,
  "n_in": 66,
  "p6_log": {
   "log_path": "results/campaign51_20260728/realistic_20260729/graph1_20260901/retrieved/run_20260902_graph1_headrebaseline_iiib/darksiren_emri_20260902_000633_h_0_73.log",
   "line": "2026-09-02 00:12:39,636 [bayesian_statistics.py:5903 - p_D()] P6 host-recovery (h=0.7300): 1D 66/76 hosts recovered/in-cat events seen (86.84211%), 2D 66/76 hosts recovered/in-cat events seen (86.84211%)",
   "h": 0.73,
   "n_recovered_1d": 66,
   "n_in_catalogue": 76,
   "found": true
  },
  "p6_numerator_matches_n_in": true,
  "passed": true
 },
 "g4_scatter_law": {
  "n_finite_mahalanobis2": 76,
  "n_singular_covariance": 0,
  "ks_statistic": 0.06614822414302035,
  "ks_pvalue": 0.8715984091477792,
  "ks_alpha": 0.05,
  "ks_passed": true,
  "f_outside": 0.13157894736842105,
  "envelope": [0.134, 0.325],
  "envelope_passed": false,
  "passed": false
 },
 "g_population_disclosure": {
  "root": "results/campaign51_20260728/realistic_20260729/tree2_20260830/b8_cal_harness_work_s4_postflip",
  "n_seed_S": 67,
  "n_seed_T": 25,
  "population": 200,
  "note": "harness 0 mixed rows disclosure; production is a single pool (draft G-invariants)."
 },
 "passed": false
}
CENSUS: n_in_catalogue=76 n_OUT=10 n_IN=66 f_OUT=0.1316
--dry-run: G-1..G-4 + census only (INSTRUMENT-DEFECT). Registered statistic (Delta h_cone/phi_cone/SE/Z) NOT computed (verifier independence, draft §7).
```

Full JSON also written to `cone_loss_work/cone_loss_gates.json` and (per `--out`)
`cone_loss_result.json`; the census frame itself (76 rows: `event_idx`, `chord`,
`radius`, `outside`, `delta_theta`, `delta_phi`, `phi_var`, `theta_var`, `cov`) to
`cone_loss_work/cone_loss_census.csv`.

## Gate readout — what a verifier must reproduce

- **G-1 pins (GREEN):** catalogue md5 `c52c13b5cab61f6b3f04bbe202550969` reproduced;
  production CRB md5 `9a1f2a14384a9281c97ca3be312ddaab` reproduced; both venues'
  `GIT_COMMIT_AT_RUN.txt` = `1ec9514dd1808c48b18c0792dce558e5bba0f116` (draft's
  `1ec9514d` prefix matches).
- **G-2 double anchor (GREEN, both):**
  - R-MKER-6: `p3_2d_fleet_20260825/bc_900121_work/seed900121/simulations/prepared_cramer_rao_bounds.csv`
    row 20 → chord = `1.674659860716462e-03` (registered `1.674660e-03`, tol `5e-10` —
    diff `1.39e-10`, inside tolerance), radius = `1.4956979545757095e-03` (registered
    value, exact to all 16 quoted digits).
  - CMEM-A1: `p3_b0_work/bc_900101_work/seed900101/simulations/prepared_cramer_rao_bounds.csv`
    row 0 → chord = `0.01166569410071811` (registered `0.0116656941007181`, diff
    `1e-16`, inside tolerance), radius = `0.035912194615445196` (registered
    `0.0359121946154451`, diff `5e-16`, inside tolerance).
- **G-3 join (GREEN):** production CRB has 1590 rows; in-catalogue (`in_catalog` true,
  `host_galaxy_index >= 0`) count = **76**, matching the draft's own §0/§2 figure and
  the P6 log denominator. `event_idx` = CRB row index confirmed as the diagnostics
  join key independently (event_likelihoods.csv event_idx spans 0..1589 minus exactly
  `{1203, 1356}` — the draft's disclosed gaps — verified before writing this record).
  n_OUT = **10**, n_IN = **66**; the production log line (quoted above, from
  `darksiren_emri_20260902_000633_h_0_73.log:8622`) reads "1D 66/76 hosts
  recovered/in-cat events seen" — the P6 numerator (66) equals n_IN exactly.
- **G-4 scatter law (RED on the envelope sub-check; KS sub-check GREEN):** the 76
  production sky offsets' Mahalanobis² under Σ' (algebraically the J-scaled offset's
  Mahalanobis² under `J Σ' Jᵀ` — see script docstring) pass the χ²₂ KS test cleanly
  (D = 0.0661, p = 0.872, α = 0.05 — no shape defect). But `f_OUT` at k = 1.5 is
  **13.158 % (10/76)**, which sits **0.24 percentage points below** the closed-form
  envelope's lower bound (13.4 %, the 1-D/Rayleigh-tail limit `2Φ(−1.5)`). This is a
  genuine, narrow miss — not a script defect: the same 13.2 % figure appears in the
  draft's own §0 provenance table, and draft §9 item 4 ("both inside the as-designed
  envelope") is listed as an OPEN QUESTION routed to `d-cone-register`, not yet a
  ratified fact. **This build reports G-4 as RED on the envelope clause and defers the
  disposition to the author/verifier**, per the draft's own G-4 language ("A failure
  means the offsets are not the designed Fisher tail... ⇒ INSTRUMENT-DEFECT, STOP,
  fresh RULE"). The `--dry-run` script still exits 0 per this task's contract; the
  `--out` JSON's `"verdict"` field is `"INSTRUMENT-DEFECT"` and the gates' own nested
  `"passed": false` field carries the finding for whoever reads it next.
- **g-population disclosure:** harness root has 67 `seed*_S` dirs and 25 `seed*_T`
  dirs under `--population 200` (draft's registered `--population 200, seeds
  901000-901066` — 67, not the nominal 66, consistent with the draft's own "expected
  ≈ 140 OUT" scoping language rounding); production is confirmed a single pool (one
  CRB file, no population split needed).

## Audit item carried forward (draft G-4, not resolved by this build)

The draft names an open audit item: "which of (d_L, qS, phiS) the production pool
scatters is set in `datamodels/detection.py:161-178`
(`convert_to_best_guess_parameters`)". Read (not executed) for this record:
`convert_to_best_guess_parameters(self, rng=None)` draws simulated measured
parameters from the Fisher-matrix posterior — with an `rng` given (the production
path) it calls `self._correlated_draw(rng)`, "a single correlated sample from the
4-dimensional multivariate normal defined by the full Cramér-Rao covariance
sub-matrix for (phi, theta, d_L, M)" (docstring, cites Cutler & Flanagan 1994;
Vallisneri 2008 arXiv:gr-qc/0703086); without an `rng` it falls back to
`_independent_draw()` (four independent truncated-normal marginal draws). This
confirms the CRB's own `qS`/`phiS` columns ARE the scattered (observed) sky position
already — consistent with the 76/76 nonzero-chord finding in the registration draft's
§0 and with this build's own census (`chord > 0` for all 76 rows, by construction of
the census — `outside` is a strict `>` comparison against a positive radius).

## Not this build's decision

Per the registration draft §7/§4, computing the registered statistic (Δh_cone,
φ_cone, SE, Z, the leave-out cross-check, and the harness Δs replicate) is
runner-only — a DIFFERENT agent's job, gated on G-2 anchors GREEN (met) and, per this
build's finding, on the author's disposition of the G-4 envelope miss (RED as
measured). This build does not rule on that disposition — it is routed to
`d-cone-register` per the draft's §9 item 2/4.
