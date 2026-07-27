# Runbook — post-#40 handoff (written 2026-07-27, session end)

Supersedes `RUNBOOK_NEXT_SESSION_2.md` (all its unblocked threads executed).
State ledgers: `MULTISEED_READOUT_20260726.md` (campaign),
`threeway_ab/THREEWAY_AB_READOUT.md` + `threeway_ab/GLADE_PV_AUDIT.md` (#40),
`../pp_fullpower_20260727/FULLPOWER_READOUT.md` (P–P). Derivation:
`docs/derivations/hostz_pv_photoz_kernel.md` (RATIFIED, all 5 gates).

## State at handoff (all merged, nothing in flight)

- **main @ `0a4f0d1`**: PRs #45 (log dedup), #46 (docs + 5-ln information
  floor RATIFIED, #44 tracks), #48 (`--host_z_kernel` decomposition flag),
  #49 (P–P impostor harness + full-power), #50 (counted-once PV widths)
  all merged 2026-07-27.
- **Counted-once PV in production**: per-class parse-time widths (corrected =
  cat σ_tot ⊕ (1+z)·150 km/s/c; else ONE (1+z)·500 km/s/c; no 0.0015 fill;
  `SIGMA_V_PEC_KM_S` = 0.0 ablation knob). Reduced catalogue REGENERATED
  (22,641,048 rows; old = `.pre40b_20260727`) and restaged to the cluster
  (preflight cols=8 OK). `m_th_map_nside32.npy` verified BYTE-IDENTICAL (no
  regen needed). Kernel + pipeline goldens regenerated under new values.
- **GLADE+.txt re-downloaded** (6.4 GB, elysium.elte.hu/~dalyag mirror,
  2022-07-08 vintage) — it had been deleted from every machine; keep it.
- **Three-way A/B measured** (seed1000, via the flag): δ-kernel = 85.3%/86.7%
  (1D/2D) of the ln movement; normalization legs ALONE de-rail 1D (truth
  MAP, −85.4 ln gap); **2D needs the δ-kernel** (kernel numerator leaves MAP
  0.80, +29.4 ln over truth). Raw cell: `v1_probe_genmarg_vdkernel/` (local
  only, per probe convention).
- **P–P full-power** (24 cells, n=2000): every smoke conclusion confirmed;
  absolute ≈ generator_marginal; residual HIGH bias → 0 as completion weight
  → 0 ⇒ **B_num is the sole residual carrier** — measured, at power.
- **Paper ON HOLD until real data** (author decision 2026-07-26). PR #47
  (22/24 markers filled) stays open, unmerged. Do not touch without the
  author lifting the hold.
- Cluster: queue empty; repo checkout still on `feat/pp-impostor-harness`
  (content = merged) — `git checkout main && git pull` at next 2FA session.
  Workspace expires 2026-09-23. Local disk 73%.

## Open threads (pick per author priority)

1. **2D mass-marginal derivation** [#40 remainder, critical path]: with a
   broadened host-z kernel the 2D channel does NOT recover truth (+29.4 ln
   at 0.80, measured). The mass channel needs its own kernel treatment for
   real-data mode — EXP-45 truncated-lognormal thread is the entry point
   (`results/mass_kernel_truncation_20260713/FINDINGS.md`). Physics
   derivation → /gpd + /physics-change.
2. **B_num residual-bias model**: full-power P–P isolates B_num; the
   high-impostor cells (zs=0.79/0.95) are the clean laboratory. Target: a
   derived (not fitted) bias model, then a harness fix + rerun.
3. **#39 blind alternative-truth mock**: NOW well-timed — the kernel stack
   is settled and merged; the blind mock should test main @ 0a4f0d1+.
   Needs new mock generation on the cluster (sealed h_inj).
4. **#23 completion-term realism** [paper-blocker when paper resumes].
5. Smaller: #41 (dgen rationale), #42 (medium bundle). #44 is a tracking
   record only (5-ln floor ratified; revisit triggers listed there).
6. **Real-data-mode validation** (when 1+2 land): scattered-z P–P gate +
   the §3.6 pre-registered A/B (predicted golden-set width inflation
   ×2.5–×5.5) — pre-registered in the derivation doc BEFORE running.

## Conventions worth re-reading next session

- Local probe runs need the `cwd/` symlink recipe (master_thesis_code +
  simulations links; run FROM cwd/) — paths are CWD-relative.
- Probe dirs: commit only the readout .md; simulations/ + run_metadata are
  gitignored by design.
- Any future default change must regen BOTH goldens per their docstrings
  (REGEN_KERNEL_GOLDEN / REGEN_PIPELINE_GOLDEN, reviewed value-update step).
- If the reduced CSV ever changes content again: rebuild `m_th_map` and
  verify (it depends only on B_mag + sky, but verify, don't assume) — and
  restage BOTH artifacts if it differs.
- Combine `.out` "MAP" lines are channel summaries — read the JSONs.
- Per-event diagnostics: `simulations/diagnostics/event_likelihoods.csv` —
  first tool for any posterior anomaly.
