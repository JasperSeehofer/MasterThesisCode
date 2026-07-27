# Runbook — post-FIX-3-§7.1 handoff (written 2026-07-27, session end)

Supersedes `RUNBOOK_NEXT_SESSION_3.md` (its thread 1 — the 2D mass-marginal —
was executed end-to-end this session; threads 2/3/4 remain, re-listed below).
State ledgers: `mass_ab_20260727/{MASS_KERNEL_AB_READOUT.md,ZMZ_AB_READOUT.md}`,
`zres_survival/z3_results.json`. Derivations (both RATIFIED 2026-07-27):
`docs/derivations/mass_marginal_2d_kernel.md` (M1–M7),
`docs/derivations/fix3_zmz_catalog_selection.md` (Z1–Z7 rev. B + 2 amendments).

## State at handoff (all merged/pushed, main @ 0040b5d; cluster synced; queue empty)

- **2D mass-marginal kernel (#40 remainder)**: RATIFIED + implemented
  (`e9bec6d`, `--host_mass_kernel`, GH crossover w/ σ_lnM ≤ 0.1 family cap —
  a golden-caught implementation correction, doc §3.3). Four-cell A/B:
  P4(ii) fired — necessary-not-sufficient MEASURED (~2–3 of ~26 ln).
- **Branch discriminators (b)/(e)**: NULL at 1e-7/1e-10 ln (env overrides
  `MTC_ABLATE_MZ_PROJ`/`MTC_HOST_QUAD_N`, `fe0ca3e`). Branch (c) cleared.
- **FIX-3 §7.1 joint z×M_z grid**: RATIFIED (rev. B — NB the adversarial
  review REBUILT the packet: production-axis increment −15.6 not −58; 1D
  moves in generator mode via shared D_gen; β_Ḡ complement-measure is an
  ASSUMPTION w/ mandatory symmetric control) + implemented (`608426b`,
  `--pdet_wbh_z_resolved`, default OFF, K5 shrinkage, `MTC_WBH_GRID_ONLY`
  control knob). z3 re-pinned the gate to −6.5±4 (`29798b8`) — the Z2
  bandwidth itself is value-side load-bearing.
- **§7.1 A/B verdict: NULL per pre-registered P5** (`0040b5d`):
  Δ(2D@0.80) −0.51 raw / −1.07 grid-corrected. ALL structural no-ops pass
  at machine precision (A-cell 1D bit-identical; B-cell 2D−1D exactly
  invariant) — implementation validated, effect suppressed.
- **Attribution now**: the ≈+23 ln 2D HIGH residual (MAP 0.80) is owned by
  **(d2) selection-side M scatter/truncation (RATIFY-M5 deferral) + (g1)
  mass-support clamp (81.4 % of catalogue rate-weight above the pool's
  m_max = 6.0)**. Leading mechanism for the §7.1 null: clamped queries
  cannot feel the u-conditioning → (d1) suppressed by (g1).
- **Campaign redesign = THE 2D critical path (issue #51, author-ratified
  directives)**: (i) mass bounds to the scientifically correct limit —
  M ∈ [10⁴, 10⁷] source frame (Babak valid band) unless narrowing is
  VERIFIED; the current [10^4.5, 10^6] lives at `cosmological_model.py:
  179–180` (unjustified override; `handler.py:27–28` M_max=1e6 is a
  separate uncoordinated constant; the pool's exact-6.000 detector-frame
  cap frame-convention must be pinned down); (ii) pool sized/sampled to a
  minimum-ESS-per-node floor on the joint grid (data-driven p_det;
  acceptance: catalogue-weighted shrinkage w̄ → ~1).
- **1D real-data mode is healthy**: absolute_marginal 1D peaks at truth on
  the deep venue at the current stack (old cell-A rail gone; matches the
  zres −69 ln figure).
- Paper (#47) still ON HOLD. Workspace expires 2026-09-23.

## Open threads (priority order)

1. **Campaign design doc (#51)** [2D critical path]: bounds decision
   implementation (physics-change on `cosmological_model.py` — remove/
   replace the override; reconcile handler.py constant; pin the M_z frame
   convention), ESS-floor sizing analysis (N + sampling measure — consider
   importance-weighting injections toward the catalogue's R_eff-weighted
   (z, M_z) support), detectability-verified-narrowing test, cost/walltime
   plan. THEN: full injection + simulation + evaluation campaign.
   Pre-registered prediction to carry: (d1) reappears at ≈ table size
   (−6.5 ln increment) on the clamp-free pool.
2. **P1 parity audit** (mandated by P5's consistency clause,
   ZMZ_AB_READOUT §follow-up): quantify probe-vs-production Σ_glob_wbh
   parity, esp. hypothesis (iv) — the m-clamp suppression fraction. Cheap
   (offline, existing tables + a production-mode Σ dump). Confirms/refutes
   the suppression mechanism BEFORE the campaign locks its prediction.
3. **(d2) derivation** [the other residual owner]: selection-side M
   scatter/truncation — the mass analog of §7.1 in the selection legs
   (RATIFY-M5 deferral; per-galaxy scatter-averaged w_g = Z_M/(1+z) is the
   documented next-order candidate). Could ride the campaign or precede it.
4. **B_num residual-bias model** (runbook-3 thread 2, unchanged).
5. **#39 blind alternative-truth mock** (post-kernel-stack; arguably wait
   for the new campaign universe now).
6. **#23 completion-term realism** [paper-blocker when paper resumes].

## Conventions / gotchas added this session

- Workflows: NEVER auto-start — propose w/ per-agent model+effort table,
  await explicit approval (memory: feedback-workflow-approval).
- Module-import `_LOGGER.warning` lands in `.err` not `.out` on cluster
  jobs (root logger unconfigured at import) — check stderr before
  concluding an env override didn't fire.
- Env overrides for cluster A/Bs: `sbatch --export=ALL,VAR=1` works;
  verify via the loud import-time warning + run_metadata (cli flags only —
  env NOT recorded in metadata).
- `FORCE_SEED` in evaluate.sbatch pins per-h array tasks to one seed for
  fused-local-run comparability.
- Grid-only control cells are non-optional when a flag changes both a
  grid/interpolant AND physics — measured same-order opposite-sign
  confound this session.
- `git add -A <dir>` nearly committed the 1.6 GB catalogue backup
  (`.pre40b_20260727`, untracked, unignored) — gitignore it or add paths
  explicitly.
- Probe cells local (cwd/ symlink recipe) vs per-h cluster arrays both
  work; cluster ~4.5–6 min/h-task incl. joint-grid build.
- Diagnostics metric: Σ ln from `simulations/diagnostics/
  event_likelihoods.csv` (combined_no_bh / combined_with_bh); combine
  `.out` "MAP" lines are channel summaries — read the CSVs/JSONs.
