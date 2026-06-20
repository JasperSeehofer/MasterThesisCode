# HANDOFF — Redshifted-mass fix landed + D(h)/L_comp is the next bias thrust

**Date:** 2026-06-20  **Author:** session with Jasper (continuation of the pipeline deep-review)
**Branch state:** `main` clean (this handoff is the only new commit). **THREE physics fixes now live on
unmerged branches** — all must bundle into the keystone re-sim (see §2). Supersedes nothing; reads
alongside `.planning/HANDOFF-PIPELINE-DEEP-REVIEW-20260620.md` (the 4-area agenda is still open).

---

## 1. WHAT THIS SESSION DID

Drove handoff item-2 (the **redshifted-mass convention**, the prime population-side bias suspect) to a
committed result. **It is a real correctness defect but NOT the residual-bias cause** — the adversarial
investigation overturned its own initial verdict.

- **`[PHYSICS]` redshifted-mass fix LANDED (Design B)** — branch `physics/mass-redshift-convention`,
  commit `0099ce2`, **NOT merged**. The sim injected **source-frame** `M` into FEW, which expects the
  **detector-frame** `M_z = M_source·(1+z)`. Fix lifts once at injection
  (`parameter_space.set_host_galaxy_parameters:197`, `main.injection_campaign:656` via
  `physical_relations.redshifted_mass`), drops the now-redundant p_det grid re-lift
  (`simulation_detection_probability.py:265` → `log10(_M_arr)`), deletes the dead
  `_map_BH_masses_to_redshifted_masses`. Inference `(1+z)` factors + `/(1+z)` host filter are **kept**
  (now exactly correct; `Detection.M` "M_z" docstring is now true). `/check` green (568 pass, ruff+mypy
  clean); guard test `test_set_host_galaxy_parameters_injects_redshifted_mass` fails-before/passes-after.
- **Why it's a correctness fix, not a bias fix (the key finding):** the residual +0.01–0.02 H0 bias is
  dominant in the **mass-FREE 1D channel** (prod 1473ev: 1D +0.0024 ≥ 2D +0.0022; the 2D mass channel
  was already closed by the H3 fix to +0.0007). A defect confined to the mass channel cannot bias a
  channel with no mass term. Within the 2D channel the sign is **adverse** (fixing centres the true host
  → pulls H0 *up*, worse). Memory: [[project_mass_convention_defect]]; vault pattern
  `scientific-computing-validation#channel-localization-falsifies-a-suspected-cause...`.
- **Process note (W-CONF-13):** a multi-agent trace workflow's own *verdict* was confidently wrong
  (positive bias / 2D mechanism / one-liner "minimal fix"); 2/3 adversarial refuters + a doc-check
  against `docs/H0_BIAS_RESOLUTION.md` overturned it. The verdict's "minimal fix" also missed the
  coupled p_det grid site. **Lesson: adversarially verify a workflow's synthesis, not just trust it.**
- User's pre-implementation physics questions all checked out in code: no sim/eval truth leak (inference
  uses catalog/integration z; `det.M = M_z` is a genuine observable); redshifting the mean mass is fine
  (σ propagated via `sigma_gal_frac`, redshift σ handled by the outer z-marginalization); CRB error is
  automatically `σ(M_z)` (the Fisher stencil perturbs the stored `M.value`); the bounds-skip hazard is
  safe (`M.upper_limit=1e7` ≫ max `M_z≈1.18e6`).

---

## 2. CRITICAL — BRANCH STATE (three physics fixes, none merged)

| Branch / commit | What | Merge gate |
|---|---|---|
| `physics/pdet-horizon-survival` → **already merged** (`5e94139`) | survival-function p_det | done (on main) |
| `physics/lcat-gray-ratio-of-sums` (`816f904`) | L_cat Gray A.9/A.10 (bias halved 1D 0.750→0.740) | awaits multi-seed; merge regardless (Gray-correct) |
| `physics/mass-redshift-convention` (`0099ce2`) | redshifted-mass Design B | **RE-SIMULATE tier**; bundle into the campaign, then merge |

**The mass-convention fix CANNOT be back-applied to existing data** — it changes the injected
waveform/SNR/Fisher, so every existing CRB **and** injection set (incl. canonical seed400) is stale and
must be regenerated. `DATA_INVENTORY.md` on the branch records this (RE-SIMULATE Evaluation-Log entry).
Do **not** run `--evaluate` with this branch against old data (it would treat source-frame M as M_z).

---

## 3. KEYSTONE — the multi-seed re-sim campaign (now bundles BOTH unmerged physics fixes)

Still the single highest-leverage action (bias verdict + cosmic-variance error bar + the coverage/PP-plot
referees demand). **Change vs the prior handoff:** the campaign must now be generated with BOTH
`physics/lcat-gray-ratio-of-sums` AND `physics/mass-redshift-convention` merged in (the mass fix needs
fresh CRBs+injections; L_cat is inference-only but should ship together for one clean baseline).

Suggested sequence when at the controlling machine:
1. `git checkout main && git pull && uv sync --extra gpu`
2. Merge both branches into main (or a campaign branch) after a final `/check`. Resolve any overlap in
   `simulation_detection_probability.py` (L_cat touches `bayesian_statistics`; mass touches the p_det
   grid line 265 — likely no conflict, but verify).
3. Generate the multi-seed campaign (events **and** injections) with the merged code:
   `for S in 500 600 700 800; do bash cluster/submit_resimulate_phase50.sh --seed $S; done`
   (+ re-generate seed400 so all five share the new convention). Scale to K≥20 for the real study.
4. Per-channel MAP across realizations → bias verdict (mean), error bar (spread), PP-plot/coverage
   (KS-uniformity of the h_true quantile — build the harness while it runs; currently absent from 600+ tests).
5. **EXP-25** (vault) adjudicates the mass fix: prediction = combined MAP moves *up*/negligibly, NOT
   toward truth. If it moves toward truth by > σ_boot, reopen the mass convention as a bias contributor.

> S-effort while regenerating CRBs (from the prior handoff, still wanted): write
> `_coord_frame`/`_cov_frame=ecliptic` inside `save_cramer_rao_bound` so fresh output is self-labelled and
> `migrate_crb_to_ecliptic.py` is a guaranteed no-op (kills the double-rotation trap permanently).

---

## 4. NEXT INVESTIGATION THRUST — the ACTUAL residual-bias driver (D(h) / L_comp)

The user chose "both, in order": (1) land the mass fix [DONE], (2) **pivot the adversarial investigation
to where the +0.02 actually lives** — the `−N·log D(h)` selection-function / completeness (L_comp) term in
the **mass-free 1D channel**. Evidence (`docs/H0_BIAS_RESOLUTION.md`): the per-event `Σ log L_i` peaks ≈
truth, but `−N·log D(h)` pushes MAP up +0.015–0.020 (Δ(−N log D) ≈ +7.6). This is estimator-independent
(survived the survival-p_det and F4 changes) and is the prime remaining systematic.

Recommended approach (mirror this session): a focused trace/decomposition workflow with **adversarial
refuters** on any causal verdict, and verify load-bearing claims against the doc/data — do NOT trust a
single synthesis (W-CONF-13). Candidate sub-questions: is `D(h)`'s `dV_c/dz` volume-prior pull correct
vs Gray (the completion term was "characterized as faithful" but its h-slope is the lever); is the
completeness weight `p(G|D,H0)` modelled; is it single-seed scatter (the multi-seed campaign settles
this — so §3 and §4 are complementary, run the campaign AND investigate D(h)).

---

## 5. STILL-OPEN — the broader deep-review agenda (from the prior handoff, not yet started)

`.planning/HANDOFF-PIPELINE-DEEP-REVIEW-20260620.md` §3 has the full prioritized list. None done yet
except the redshifted-mass item (§3 P0, this session). Highest-value remaining, non-cluster:
- **P0 freeze + paper:** reconcile the paper's self-contradictions (SNR `>15` in `results.tex:60,99`
  vs `≥20` in `method.tex`; forward-diff vs five-point stencil; Ωₘ note), wire in the commented-out
  `\includegraphics`, add a CI check that the paper's config matches `constants.py`. (`constants.py`:
  `SNR_THRESHOLD=20`, `OMEGA_M=0.25`, `TRUE_HUBBLE_CONSTANT=0.7` — the last is a Pipeline-A dead-code
  footgun, reconcile/delete.)
- **P1 HPC:** batch the whole h-grid in ONE process (each h re-reads the 1.4 GB GLADE + rebuilds
  BallTrees/grid; ~40% fixed overhead × N_h). Makes the K-realization sweeps affordable; changes no number.
- **P1 estimator/CI unification:** MAP-from-grid vs ±0.013-CI come from two paths — unify to median+HPD
  from one combined log-posterior; add grid-independence + golden-MAP regression tests.
- **P1 (physics):** thread `T` into the PSD (`t_obs_years=self.T`) so the 5-yr run stops using 4-yr
  confusion noise. `/physics-change`.
- **Plotting/Pages, toy gwcosmo/icarogw cross-check:** later, after science frozen.

---

## 6. POINTERS

- **Memory:** [[project_mass_convention_defect]] (this session, status=implemented),
  [[project_residual_bias_decomposition]] (L_cat + the residual), [[project_canonical_data_seed400]]
  (data wiring — note it is now convention-stale for the mass fix),
  [[project_pdet_horizon_survival]], [[project_fisher_frame_mismatch]] (closed non-cause).
- **Docs:** `docs/H0_BIAS_RESOLUTION.md` (§3.15 H3, §3.17 L_cat, the 1D-vs-2D decomposition tables);
  `DATA_INVENTORY.md` (on the mass branch: RE-SIMULATE entry + 5-tier checklist).
- **Vault:** patterns `scientific-computing-validation#sim-eval-shared-wrong-convention...` (annotated)
  + `#channel-localization-falsifies-a-suspected-cause...` (new); W-CONF-13; EXP-25.
- **Protocol:** physics changes → `/physics-change` (5-point gate) + **measure the sign empirically /
  adversarially verify any verdict before committing** (this session and the L_cat session each caught
  wrong-signed or wrong-mechanism claims that way). Before commit: `/check` then `/pre-commit-docs`.
- **Working tree:** `.planning/debug/{baseline.json,comparison_current.md}` are a *different* session's
  changes — leave unstaged. Untracked `results/` are local eval outputs.
