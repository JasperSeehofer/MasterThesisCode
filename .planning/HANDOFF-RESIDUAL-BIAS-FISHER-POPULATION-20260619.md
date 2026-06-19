# HANDOFF — Residual H0 bias: Fisher frame mismatch + population model

**Date:** 2026-06-19  **Branch state:** `main` = `origin/main` = `5e94139` (pushed)

---

## WHERE WE ARE (done this session)

The **p_det detection-horizon survival** change is **merged to `main` and pushed** (commit
`5e94139` `[PHYSICS] p_det: detection-horizon survival function replaces kernel regression`).
Full context in memory: [[project_pdet_horizon_survival]], [[project_canonical_data_seed400]].

- Replaced kernel-regression p_det with the exact survival `p_det(d_L)=P(d_hor≥d_L)`,
  `d_hor=SNR·d_L/threshold`. Monotone, boundary-exact, h-invariant, built once.
- Gate green (ruff/mypy/pytest 567); /gpu-audit PASS.
- **Validated on seed400** (same 937/936 events, physics-floor):
  - D(h) decline 0.73→0.76 = **−0.87%** (was −3.9% local-linear) — mechanism confirmed.
  - MAP before→after: **1D 0.760→0.750**, **2D (headline) 0.747→0.7375**. Both toward truth.
  - No posterior spikes (h-invariant p_det).
- **RESIDUAL bias remains: 1D +0.020, 2D +0.0075 above truth 0.73.** This is the
  estimator-INDEPENDENT "Piece B" — the target of THIS handoff.

---

## GOAL (next session)

Chase the residual toward truth (h=0.73) via the two estimator-independent suspects,
**measuring each one's impact on the posterior in isolation**:
1. **Fisher ecliptic/equatorial frame mismatch** ([[project_fisher_frame_mismatch]])
2. **Population model** (mass function / dV_c/dz / galaxy redshift uncertainty)

---

## EXPERIMENTAL DESIGN (agreed)

- **Two separate branches off `main`**, each with ONE change, each `--evaluate`'d vs the
  SAME baseline → clean per-change attribution. The changes touch DIFFERENT files
  (Fisher: `detection.py`/`handler.py`/`parameter_estimation.py`; population:
  `galaxy.py`/`cosmological_model.py`) → no conflicts.
- **Run the evaluations SEQUENTIALLY** (32 cores + 2.2M-galaxy catalog per process;
  [[feedback_worktree_merge]] warns parallel worktrees diverge). Isolation of the
  *changes* (not the compute) is what gives clean attribution.
- After each-alone, a **combined branch** shows the joint (possibly non-additive) effect.
- Branch names suggested: `physics/fisher-ecliptic-cov`, `physics/population-model`.

---

## SUSPECT 1 — Fisher frame mismatch ⚠️ VERIFY BEFORE TOUCHING (double-rotation risk)

**Claim** ([[project_fisher_frame_mismatch]], 52+ days old — verify against current code):
after Phase 43 ecliptic migration, sky position `phiS/qS` is ecliptic but the Fisher
covariance sky-block (`delta_phiS_delta_phiS`, `delta_qS_delta_qS`, `delta_phiS_delta_qS`
+ cross-terms `d_L-phi`, `d_L-theta`, `M-phi`, `M-theta`) may still be EQUATORIAL.
BallTree host-search radius (`handler.py:~315`) uses `J=diag(|sin θ_ecl|,1)` on that
covariance → mixed-frame, mis-scaled radius by ~O(sin 23.4°)=20–40%.

**⚠️ CRITICAL — there was already a double-coordinate-transformation error.** The seed400
`campaign_metadata.json` explicitly avoided `migrate_crb_to_ecliptic.py` because it
"would double-rotate post-Phase-36 native-ecliptic data." So the CRB may ALREADY be
native-ecliptic. **DO NOT rotate blindly.** First VERIFY the actual current state:

1. **CRB columns:** check the VALUES of `_coord_frame` and `_cov_frame` in the seed400 CRB
   (`simulations/prepared_cramer_rao_bounds.csv` → symlink). seed400 has both columns;
   read what frame each declares. If `_cov_frame` already = ecliptic, the mismatch may be
   GONE (the fix already applied, or the data native-ecliptic).
2. **Code:** read `datamodels/detection.py` (~L100 reads `_cov_frame`, ~L101-112 reads
   phiS/qS + covariance) and `galaxy_catalogue/handler.py` (~L315-360 BallTree radius).
   Determine whether sky position and covariance are read in the SAME frame TODAY.
3. **History:** `git log -p` on `detection.py`, `handler.py`, `migrate_crb_to_ecliptic.py`,
   `parameter_estimation.py` since Phase 36/43 — what was already done? Is a rotation
   already in place? (Avoid re-introducing the double-rotation.)
4. **Verdict:** ONLY if a genuine mismatch remains in the CURRENT seed400 CRB + current
   code (sky ecliptic, cov equatorial) is the rotation fix `Σ_ecl = J_rot · Σ_eq · J_rot^T`
   warranted. Otherwise the residual is elsewhere → focus on population.

Fix (IF warranted): rotate the 2×2 sky covariance block + all cross-terms to ecliptic via
the astropy Jacobian of (θ_eq,φ_eq)→(θ_ecl,φ_ecl) at each source. /physics-change gate.
Open GitHub issue (label `physics`, `paper-blocker`, milestone "Paper Submission").

---

## SUSPECT 2 — Population model

Candidates (all /physics-change):
- `datamodels/galaxy.py:~64` galaxy redshift uncertainty `0.013*(1+z)^3` — no reference;
  standard scaling is `(1+z)`. Could bias the host-redshift weighting.
- Mass function / GLADE stellar-mass → central-BH-mass relation (the 2D channel's M_z prior).
- `dV_c/dz` comoving-volume weighting and the M-prior in the numerator/denominator
  (`bayesian_statistics.py` integrands; `cosmological_model.py`).
- Note constants: `OMEGA_M=0.25`, `H=0.73` (WMAP-era; Planck is 0.3153/0.6736) — a known
  [LOW] item; relevant if the population/volume integrals use Omega_m.

Investigate which assumption can move H0 by ~+0.01–0.02; verify with a sensitivity sweep
before committing a change.

---

## VALIDATION RECIPE (plumbing — don't rediscover)

Canonical data is already wired (both HARDCODED defaults symlinked to seed400):
- `simulations/injections` → seed400 519-file set (=cluster).
- `simulations/prepared_cramer_rao_bounds.csv` → seed400 990-event CRB (=cluster, has
  `_coord_frame`+`_cov_frame`). Stale 542 archived in `simulations/archive/`.

Per-change evaluation:
```
# h-grid (writes HARDCODED simulations/posteriors/h_*.json + posteriors_with_bh_mass/)
for h in 0.70 0.705 ... 0.80 ; do uv run python -m master_thesis_code simulations --evaluate --h_value $h ; done
uv run python -m master_thesis_code simulations --combine --strategy physics-floor   # MAP -> combined_posterior.json (map_h)
```
- ~41s/h-value on 32 cores. 18-pt Δh=0.005 [0.70,0.80] + Δh=0.0025 refine near peak.
- ⚠️ posteriors output path is HARDCODED `simulations/posteriors[_with_bh_mass]/`.
  ARCHIVE/clear it between runs so estimators/changes don't mix.
- CRB MUST have `_coord_frame` AND `_cov_frame` cols or `Detection.__init__` raises.

**Baselines (current `main`, survival p_det) on seed400 — already computed this session:**
- 1D MAP = 0.750, 2D MAP = 0.7375. (current `simulations/posteriors[_with_bh_mass]/`
  combined_posterior.json; archived old local-linear top-level posteriors are in
  `simulations/posteriors*_archive_20260619_survival_validation/`).
- Local-linear seed400 baseline (pre-survival, for reference): 1D 0.76, 2D 0.747
  (`simulations/cluster_run_phase50_seed400_20260524/simulations/.../combined_posterior.json`).

**Rigorous final validation** (for the paper): full 83-pt grid + K≥5 multi-seed bootstrap
on bwUniCluster (single-seed local Δh≈0.0025 can't cleanly resolve a ~0.01 shift vs
~0.017 seed scatter). SSH alias `bwunicluster` (ControlMaster; 2FA — establish with
`! ssh bwunicluster echo test`). seed400 run dir:
`/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260516_seed400_phase50/`.

---

## PROTOCOL REMINDERS
- `/physics-change` hard gate for BOTH changes (present old/new/reference/dimensional/limit).
- `/check` + `/gpu-audit` before commit; `[PHYSICS]` prefix; branch off main (never commit
  to main directly); update CHANGELOG.
- GitHub: open issues for each (labels `physics`/`paper-blocker`, milestone "Paper Submission").
- Working tree note: `.planning/debug/{baseline.json,comparison_current.md}` were modified by
  an earlier session — NOT ours; leave unstaged.
- Optional: `/wiki-debrief` to capture this session's reusable lessons (survival-vs-regression;
  data-provenance traps) into the vault.
