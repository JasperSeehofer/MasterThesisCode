# (m2) p_det pool option: P2 first, P1 fallback

Date: 2026-10-19. Mission M162. Author of record: Jasper Seehofer.
**Status: DRAFT FOR RATIFICATION**, together with `ERRATUM_m2_R27_R28.md`. The binding text is
the erratum's rule block E-5. This memo gives the reasoning, the costs and the cluster reads E-5
depends on. Nothing was run.

Ruling implemented (`ask-20261018-w224256df6b`): "**pool option P2 first** (reuse the 0.73 pool
plus a small unsealed pool at a public h with the 1e-3 invariance gate, about 10 GPU-h), P1
(fresh sealed pool, up to 250 GPU-h) as the fallback."

## 1. What the pool is for and why reuse is defensible

`SimulationDetectionProbability` builds p_det from injection horizons
d_hor,k = SNR_k · d_L,k / SNR_thr. Its docstring claims this is **h-invariant by construction**:
the survival grid is built once and reused for every trial h
(`darksiren_emri/bayesian_inference/simulation_detection_probability.py:12-24`). The reasoning is
that at fixed z, d_L ∝ 1/h and SNR ∝ 1/d_L, so SNR · d_L does not depend on h.

- If the claim is exact, a pool generated at h_inj adds samples and no information. Reusing the
  0.73 pool is then legitimate.
- If the claim is not exact, a pool at another h is the only way to see the difference.

T-1's literal text asks for "a freshly generated injection pool at that cosmology"
(`PHYSICS_METHODOLOGY_REVIEW.md:401-415`). P2 does not meet that literally; it **tests the
property that would make it unnecessary**. The ruling accepts this, and the gap is disclosed in
the prereg.

Pool of record (erratum E-1, R28):
- `$WS/injection_pool_mix200k_20260728`: 707 files, 200,100 rows.
- Content-manifest list-md5 `75f4030d5d3b0405fd948049bef5767e`.
- It is the pool both the 0.67 run and the 0.73 anchor link. The 500-file
  `injection_pool_depth15_50k` that `cluster/datasets.yaml:37-43` still labels "CURRENT" is
  **not** the pool of these runs; the registry correction is in the same commit.

## 2. P2: design, and the defect in the draft gate it fixes

**Draft text (`REGISTRATION_DRAFT.md` §3):** "a SMALL unsealed pool at a public h (e.g. 0.67, 20
tasks ≈ 10 GPU-h) compared to the rescaled 0.73 pool's survival grid (max abs Δp_det on the grid
≤ 1e-3 registered)".

**Defect (ORCHESTRATOR-DERIVED).** As written, the comparison is between two independent sets of
draws:
- 20 tasks give about 20 × 283 ≈ 5,700 rows (200,100 rows / 707 tasks ≈ 283 per task).
- The production grid is 60 × 40 = 2,400 cells, so ≈ 2.4 rows per cell on average.
- The binomial standard error of a survival fraction at that occupancy is O(0.1–0.3). The
  `local_linear` smoother and the z-resolution reduce it, but not by two orders of magnitude.

So a 1e-3 gate between unpaired pools would fail on Monte Carlo noise alone. It would make P1
mandatory whatever the physics, which is not what the ruling intends.

**Fix, registered in E-5: a paired comparison.**
- The small pool reuses the seeds of tasks 0–19 of the pool of record, `BASE_SEED + task`, with
  the same population and mixture configuration and h_P2 = 0.67.
- The injection loop draws z (and the host, mass and extrinsic parameters) from seeded generators
  and only then computes `luminosity_distance = dist(event_z, h=h_value)` (`main.py:1220`). The
  mixture uses spawned generators that are independent of h (`main.py:1022-1042`). Paired rows
  should therefore be the same draws with a different d_L.
- Pairing check P2-0 verifies this: ≥ 95 % of small-pool rows must find an exact-parameter match.
  Timeouts drop rows on either side, so row-for-row equality is not required.
- The gate compares the survival grid from the matched small-pool rows against the grid from
  the same matched rows of the 0.73 pool. MC noise cancels, and what is left is the true
  h-dependence plus numerical noise. Only then is 1e-3 a meaningful threshold.

**Unresolved by reading, found at run time.** Whether the z or host draw depends on h somewhere
(a z_max from an h-dependent d_L cap, for example) cannot be settled without running. P2-0 is
the run-time check. If it fails, the ruling's fallback (P1) applies; there is no silent
switch to an unpaired comparison with a looser threshold.

**Cost:** 20 tasks × ≤ 0.5 GPU-h wall cap (`cluster/inject.sbatch:19`) ≤ 10 GPU-h. The gate
itself is a CPU script, minutes.

**Ordering:** P2 is unsealed (h = 0.67 is public), so it runs and is gated **before the draw**.
Its number goes into the launch row. A P2 FAIL after a draw would leave a spent seal with no
legitimate pool; doing it first avoids that.

## 3. P1: the fallback, and why it needs no code change

- Fresh pool at h_inj, cap 250 GPU-h (ruling). The draft sized P1 as 500 tasks × 0.5 h. A
  like-for-like replacement of the 707-task pool of record would cost ≤ 353.5 GPU-h. **Within the
  250 GPU-h cap, a P1 pool has ≤ 500 tasks, so fewer rows than the pool of record**: about 71 %,
  disclosed. If the author wants 707-task parity, that is a cap change and a fresh [DO].
- **No naming shim (corrects draft §3).** The draft said P1 needed "a naming shim (new code, no
  physics)" because file names carry h (`injection_h_(\d+p\d+)`,
  `simulation_detection_probability.py:316`) and the `h_inj` column is loaded (`:350`, used at
  `:416` and `:651-656`). A shim would have to sit inside `darksiren_emri/`, which R24 forbids. A
  rename outside the package would be worse: the estimator reads `h_inj` per row to recompute
  d_L, so a mislabelled pool would corrupt p_det.
  - Instead, P1's pool is generated **inside the sealed RUN_DIR**, with honest file names, and is
    covered by the procedural blindness rule (erratum E-6). The run directory already leaks h in
    four places (draft §4 item 3), so one more inside it adds nothing.
  - What P1 must never do is sit in a shared `$WS/injection_pool_*` directory, where an agent
    listing pools would see h.
- P1 seeds: a fresh `BASE_SEED` recorded in the launch row, not reused from any 0.73 task, so the
  two pools are independent.

## 4. Decision rule (mirror of E-5; E-5 binds)

| step | outcome | next |
|---|---|---|
| P2-0 pairing | ≥ 95 % matched | run the invariance gate |
| P2-0 pairing | < 95 % matched | P2 FAILED (unpaired) ⇒ P1 |
| invariance gate | max \|Δp_det\| ≤ 1e-3 | P2 PASS: the sealed run reuses the pool of record; draw may follow |
| invariance gate | max \|Δp_det\| > 1e-3 | P2 FAILED ⇒ P1, and the docstring claim of h-invariance becomes a recorded defect (a new stage-0 claim) |

## 5. Cluster reads requested (read-only; via an M38 window or by the author; not performed)

T-1 cannot be costed or frozen without the reads below. M162 ran no cluster command. When a
window exists they run through `cluster/agent_ssh.sh` (CONVENTIONS / CLAUDE.md cluster
discipline), one ops agent, read-only.

```
# R-1  N_tasks and cost of the seed61000 simulate (draft §11 item 5)
ls $WS/run_20260729_seed61000/simulations/ | head; ls $WS/run_20260729_seed61000/simulations/run_metadata_*.json | wc -l
sacct -j 6090909,6090910,6090911,6090912 --format=JobID,JobName%20,State,Elapsed,AllocTRES%40 -X
# R-2  pins re-read after the emri -> emri2 move (erratum E-1; reference check hazard)
cd $WS/injection_pool_mix200k_20260728 && ls -1 *.csv | wc -l && ls -1 *.csv | sort | xargs md5sum | md5sum   # expect 707, 75f4030d5d3b0405fd948049bef5767e
md5sum $WS/run_20260729_seed61000/simulations/prepared_cramer_rao_bounds.csv                               # expect 9a1f2a14384a9281c97ca3be312ddaab (DATA_INVENTORY.md:294)
md5sum ~/darksiren-emri/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv                       # expect c52c13b5cab61f6b3f04bbe202550969
md5sum $WS/run_20260729_seed64000_h0p67/simulations/prepared_cramer_rao_bounds.csv                         # expect 8e9253fef42f574c569a04a3e19299ab (m1 pin)
# R-3  seeds of the pool of record (needed for the P2 pairing)
grep -h -m1 -i "seed" $WS/injection_pool_mix200k_20260728/../*mix200k*/*.out 2>/dev/null | head; ls $WS | grep -i mix200k
```

A mismatch on R-2 is a STOP for T-1 (CLAUDE.md dataset pinning), not a re-pin by an agent.

## 6. Cost envelope for (m2) (sources: draft §7; ORCHESTRATOR-DERIVED)

| item | P2 path | P1 path |
|---|---|---|
| pool | ≤ 10 GPU-h | ≤ 250 GPU-h (cap) |
| simulate at h_inj | N_tasks × ≤ 0.5 GPU-h (N_tasks from R-1) | same |
| evaluate, 55 nodes, iiib | 55 × 16 × ≈ 6.3 min ≈ 92 core-h; cap 94 core-h | same |
| joint_r1 replicate | not registered for (m2); a separate [DO] if wanted | same |
