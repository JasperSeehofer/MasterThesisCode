# T2.3 -- mass-aware 1D catalogue leg instrument: implementation record

Launched under row #255 -- tree 2 node T2.3 (standing grant, instrument only). Presentation:
PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md (panel-clean after 3 rounds; section 20, appended by
this same builder, carries the full implementation record and the regression-plan coverage table).
This file is the hand-off note: the exact commit file list and the exact local commands for the
counterfactual arms, per the launch instruction. No git operation performed by this node (the
orchestrator commits); no arm was run by this node except the unit-test suite (section 3 below).

---

## 1. Files to commit

- darksiren_emri/bayesian_inference/bayesian_statistics.py
- darksiren_emri/arguments.py
- darksiren_emri/main.py
- darksiren_emri/validation/correspondence_1d.py
- results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py
- darksiren_emri_test/bayesian_inference/test_catalogue_leg_1d_mass_aware.py (new file)
- docs/gates/PHYSICS-GATE-LEDGER.md (two rows appended: "implemented", "verified")
- results/campaign51_20260728/realistic_20260729/tree2_20260830/PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md
  (section 20 appended: the implementation record)
- results/campaign51_20260728/realistic_20260729/tree2_20260830/T2_3_MA1D_IMPLEMENTATION_RECORD.md
  (this file, new)

Suggested commit-subject prefix (per the physics-change convention): [PHYSICS] instrument:
catalogue_leg_1d_mass_aware in bayesian_statistics.py (default off, byte-identical) -- row #255
tree 2 node T2.3.

---

## 2. Arm (a) -- the mirror FT-fleet paired counterfactual (prediction (i), section 6.1)

Both "on" and "off" arms must be freshly run: no 12-seed HEAD-basis FT fleet exists over H_GRID_41
at this commit (section 9 item 2 of the presentation). Config "ft" resolves
catalogue_numerator_survival to "phi" and catalogue_global_selection to "phi" under
normalization_mode absolute_marginal (the FT venue's own defaults, see hier_s0_driver.py's
FT_CATALOGUE_NUMERATOR_SURVIVAL = "phi", BC_CATALOGUE_GLOBAL_SELECTION not overridden -> "auto" ->
"phi"), and theta_phi_divisor defaults to "off" -- so catalogue_leg_1d_mass_aware="on" passes the
evaluate() guard at --config ft without any other flag changes. Runner must be a different agent
from this builder (builder != runner, section 0's standing requirement).

### 2.1 Primary (4-seed) form -- section 9 item 2's "4-seed form", about 4 CPU-h local, --jobs 1

Working directory: results/campaign51_20260728/realistic_20260729/tree2_20260830/ (run from the
repo root; the driver resolves REALISTIC_DIR-relative defaults internally).

"on" arm:

    uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
      --arm S0-A \
      --config ft \
      --nodes truth \
      --seeds 900101,900102,900103,900104 \
      --h-nodes 0.6,0.61,0.62,0.63,0.64,0.65,0.655,0.66,0.665,0.67,0.675,0.68,0.685,0.69,0.695,0.7,0.705,0.71,0.715,0.72,0.725,0.73,0.735,0.74,0.745,0.75,0.755,0.76,0.765,0.77,0.775,0.78,0.785,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86 \
      --score-h 0.73 \
      --jobs 1 \
      --total-cpu-budget 14 \
      --catalogue-leg-1d-mass-aware on \
      --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/ma1d_ft_counterfactual_run

"off" arm (identical command, the same-commit fresh baseline the presentation's section 9 item 2
requires -- no banked comparand exists at this exact HEAD-basis FT/H_GRID_41/truth-node
combination):

    uv run python results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py \
      --arm S0-A \
      --config ft \
      --nodes truth \
      --seeds 900101,900102,900103,900104 \
      --h-nodes 0.6,0.61,0.62,0.63,0.64,0.65,0.655,0.66,0.665,0.67,0.675,0.68,0.685,0.69,0.695,0.7,0.705,0.71,0.715,0.72,0.725,0.73,0.735,0.74,0.745,0.75,0.755,0.76,0.765,0.77,0.775,0.78,0.785,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86 \
      --score-h 0.73 \
      --jobs 1 \
      --total-cpu-budget 14 \
      --catalogue-leg-1d-mass-aware off \
      --out-root results/campaign51_20260728/realistic_20260729/tree2_20260830/ma1d_ft_counterfactual_run

The node-dir suffix instrument (_node_dir_suffix) appends "_ma1d" for the "on" run only, so both
arms write under the SAME --out-root without collision: node_truth (off) versus node_truth_ma1d
(on), each under s0a_seed<seed>/.

Read-back (GATE T-ID pre-check, zero extra compute): the "off" arm's
s0a_seed900101/node_truth/simulations/diagnostics/event_likelihoods.csv should reproduce
fanout1_20260829/kwq1_registered_run/s0a_seed900101/node_truth_ft_sites2.2_nosmear bit-identically
on combined_no_bh and L_cat_no_bh (section 6.1's GATE T-ID), if that banked comparand is at the
same H_GRID_41/truth-node/config combination -- verify path names before relying on this.

Statistic (prereg §4.1-style paired read, corrected-combine over H_GRID_41, row #146 form):
Delta mean_h = mean_h("on") - mean_h("off"), per seed then averaged; report also the per-seed
vector and the un-truncated H_GRID_FULL companion (amendment 20, row #173) if that companion run
is available. Registered point prediction +0.05, band [+0.03, +0.10] (section 6.1); NULL <= +0.008.

### 2.2 Full (12-seed) form -- section 9 item 2's "full form", about 12 CPU-h local

Same two commands as 2.1, with --seeds 900101,900102,900103,900104,900105,900106,900107,900108,900109,900110,900111,900112
in place of the 4-seed list. Gives the lower-band-edge read at 4.3 sigma (twin-anchored) /
2.8 sigma (drag-anchored, disclosed as just under 3 under the conservative anchor).

### 2.3 Zero-compute pre-read (available BEFORE either arm above is run)

The T2.2 instrumented dump already on disk (candidate_dump_run/s0a_seed<NNNNNN>/node_truth_ft/
per_candidate_h_0_73.csv, 4 seeds 900101-900104, h=0.73 only) carries both S_4D(z_g,M_g) and
S_bar_phi(z_g) per candidate (columns s_4d_zg_mg, s_bar_phi_zg per section 1(1d) of the
presentation). rho_i and the per-event L'_i(0.73) can be rescored from this dump at zero
additional compute -- do this FIRST (section 8's F-pre falsifier: rho_q1 > 0.8 falsifies remedy
(d) before either arm above is run). Runner for this pre-read may be the same agent as the T2.2
builder or a fresh one; not constrained by builder != runner (it is a read, not an arm).

---

## 3. Arm (b) -- T2.2b, production iiib at the 3 secant h-nodes (prediction (ii), section 6.2)

Registered as "Optional... recommended, decisive for H0" (derivation section 6.4) and now REQUIRED
before arm (c) may ever be read against its bands (section 17.1's sequencing rule, revision note
2026-08-30c): this is also the only way to replace section 15.1's REPORTED-ONLY in-catalogue
transform with a genuine ARITH number. About 4-5 CPU-h local (6 h-points x 5-7 min each); no
cluster needed.

This venue is not run through hier_s0_driver.py -- it uses the real
"python -m darksiren_emri --evaluate" entry point against the pinned production CRB CSV (the
correspondence_1d.py module's own CRB_CSV_PATH default, md5 9a1f2a14384a9281c97ca3be312ddaab --
STOP and verify this checksum before running, per the repo's dataset-pinning convention), exactly
the pattern darksiren_emri.validation.correspondence_1d.run_production_wholesale already
implements, extended with two flags (--catalogue_leg_1d_mass_aware, --candidate_dump_dir) that
function does not expose as parameters. Run the following as a standalone script (or paste into
a REPL) from the repo root:

    uv run python - <<'PYEOF'
    import sys, subprocess, time
    from pathlib import Path
    from darksiren_emri.validation.correspondence_1d import (
        _setup_wholesale_cwd, PRODUCTION_FLAGS, CRB_CSV_PATH, CRB_CSV_MD5, INJECTION_POOL_DIR,
    )
    import hashlib

    md5 = hashlib.md5(Path(CRB_CSV_PATH).read_bytes()).hexdigest()
    assert md5 == CRB_CSV_MD5, f"CRB CSV md5 mismatch: {md5} != {CRB_CSV_MD5} -- STOP"

    for arm_name, ma1d in (("off", "off"), ("on", "on")):
        work_root = Path(
            "results/campaign51_20260728/realistic_20260729/tree2_20260830/"
            f"t2_3_arm_b_iiib_{arm_name}"
        )
        work_root.mkdir(parents=True, exist_ok=True)
        cwd = _setup_wholesale_cwd(work_root, CRB_CSV_PATH, INJECTION_POOL_DIR)
        out_dir = work_root / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_dir = work_root / "candidate_dump"
        cmd = [
            sys.executable, "-m", "darksiren_emri", str(out_dir), "--evaluate",
            "--h_values", "0.725,0.73,0.735",
            "--seed", "777010",
            "--pdet_z_resolved",
            "--log_level", "INFO",
            "--catalogue_leg_1d_mass_aware", ma1d,
            "--candidate_dump_dir", str(dump_dir),
        ]
        for flag, value in PRODUCTION_FLAGS.items():
            cmd.extend([flag, value])
        print("running arm", arm_name, ":", " ".join(cmd))
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        print("elapsed", time.time() - t0, "returncode", result.returncode)
        if result.returncode != 0:
            print(result.stdout[-4000:]); print(result.stderr[-4000:])
            raise SystemExit(1)
        csv_path = cwd / "simulations" / "diagnostics" / "event_likelihoods.csv"
        print("wrote", csv_path)
    PYEOF

Notes on this snippet:

- ma1d="off" reproduces the row #213 banked headreadout_20260827/iiib arm bit-identically on the
  1D channel (T1.1/T2.2's own byte-identity result, d4765539 R6) -- if it does NOT match the
  banked comparand, STOP and investigate before trusting the "on" arm at all.
- The dump_dir carries per_candidate_h_0_725.csv / _0_73.csv / _0_735.csv with the
  is_true_host column populated this time (the production draw includes the 76 in-catalogue
  events, unlike the FT-mirror's dark-only q1 population that produced zero True rows -- section
  15.1's finding). The T2.2b-registered read (derivation section 6.4, this presentation's section
  6.2) is: join on host_galaxy_index (the VALIDATED join, not the CRB-row assumption-join C5
  used), select is_true_host==True rows, and read S_4D(z_true,M_true)/S_bar_phi(z_true) directly
  from the s_4d_zg_mg / s_bar_phi_zg columns -- this is the number that supersedes section 15.1's
  REPORTED-ONLY "-130 to -117" placeholder.
- --seed 777010 is arbitrary (matches run_production_wholesale's own default; not
  physics-relevant for a deterministic evaluate() pass, per that function's own docstring).
- normalization_mode/host_z_kernel/selection_in_completion_numerator/catalogue_mass_overlap/
  completion_b_scale/pdet_dl_bins/pdet_mass_bins/pdet_estimator all come from PRODUCTION_FLAGS
  (section 3 above); catalogue_numerator_survival and catalogue_global_selection are left at
  their CLI defaults ("auto"), which resolve to "phi" under absolute_marginal -- satisfying this
  flag's evaluate()-level guard without any extra flag.

---

## 4. Arm (c) -- BLOCKED (section 17.1's sequencing rule; not to be run yet)

The 41-node production MAP array (prediction (iii), section 6.3) is BLOCKED until arm (b) above
has produced its derived ARITH in-catalogue transform (section 17.1). It is additionally queued
behind the bwUniCluster OST 5 recovery (no ssh this session; the /cluster preflight, VERDICT:
READY, is required before any submission once unblocked). No command is given here; do not submit
this arm until BOTH gates clear, and re-read PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md section
17.1 before doing so.

---

## 5. What this node did NOT do

No arm in sections 2-4 above was run by this node. The only execution performed was the unit-test
suite (darksiren_emri_test/bayesian_inference/test_catalogue_leg_1d_mass_aware.py, 26 tests) and
the full darksiren_emri_test tree in two directory halves, plus ruff/mypy -- all reported in
PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md section 20.5. No git operation. No ssh.

launched under row #255 -- tree 2 node T2.3 -- hand-off record complete.
