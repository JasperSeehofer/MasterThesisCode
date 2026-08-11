# venue_transfer.py — Performance Roadmap (gap-filler perf pass, 2026-08-12)

Author-approved mission: make realistic-venue campaigns efficient, verify-before-change,
no instrument edits. This is a report — nothing here was applied to
`master_thesis_code/validation/venue_transfer.py` or any production physics file. Files
touched: this document, `profile_smoke.py` (throwaway profiling driver), and their
generated artifacts, all under `results/venue_transfer_20260811/perf/`. Nothing under
`results/venue_transfer_20260811/` outside `perf/` was written or modified. The running
cluster campaign (array 6253922) was not touched, queried, or resubmitted.

**Strategic frame (author's stated goal, quoted from the mission):** these instruments
are "reusable EMRI/LISA validation infrastructure" for follow-on projects — so the
ranking below weights levers by (portable value × certification cost), not just
raw seed-time reduction on this one campaign.

---

## 1. Profiling — measured, not guessed

**Method:** `profile_smoke.py` (this directory) calls `venue_transfer.build_venue_context`
+ `venue_transfer.run_seed_venue` directly (bypassing the CLI's `--n-events-cap`
smoke/validate-only gate, since this is off-instrument tooling, not a registered run).
Cell `Tc`, `h_true=0.730`, `balls="real_k"`, `sigma_mode="glade"` — the realistic venue
(real per-event `K_i` multiplicities + GLADE-sampled per-candidate σ_z), the same cell
family as the running campaign's decision cell. `n_events_cap=30` (real-K, uncapped
otherwise), seed `20260808+44000` (inside the registered Tc(0.730) block), grain=seed,
1 worker (in-process, no pool) — i.e. exactly "seed-grain, 1 worker" as specified.
cProfile wrapped only `run_seed_venue` (context build measured separately, outside the
profiler, since it is a once-per-task cost, not a per-seed one).

**Scale of this smoke vs. the full campaign:** 30 events, ΣK = 20,024, max K = 14,586
(one dominant event; the true campaign max is K=245,364 on the full 982-event set).
ΣK for the full pinned set is 1,193,703 — **59.6× more pair-rows** than this smoke.
Since the estimator's cost is pair-row-bound (the `_channel_terms_at_h` loop is over
`ball.z_obs` pairs, `n_pairs = ΣK` per h), the hotspot *ranking* below should transfer
to the full campaign near-linearly; absolute wall-clock does not (see §1.3).

Raw output: `profile_smoke.pstats` (pstats binary), `profile_smoke_top.txt` (text).

### 1.1 Top hotspots, by cumulative time (n=30 smoke, wall 124.92 s, 154,407 calls)

| rank | function (file:line) | ncalls | cumtime (s) | % of seed wall |
|---|---|---|---|---|
| 1 | `_channel_terms_at_h` (venue_transfer.py:1061) | 41 | 124.92 | 100.0% |
| 2 | `_g_ball_capped` (venue_transfer.py:905) | 82 | 122.04 | 97.7% |
| 3 | `completion_mass_factor_g` (bayesian_statistics.py:1935) | 1,394 | 116.28 | **93.1%** |
| 4 | `dark_mass_density_per_mass` (bayesian_statistics.py:1750) | 1,394 | 95.18 | **76.2%** |
| 5 | `dark_mass_log10_density_unnormalised` (dark_siren_injection.py:332) | 1,394 | 74.65 | 59.8% |
| 6 | `R_eff_per_mbh` (emri_rate.py:235) | 1,394 | 53.43 | 42.8% |
| 7 | `duty_cycle_Gamma` (emri_rate.py:128) — **leaf, tottime 20.70 s** | 1,394 | 20.70 | 16.6% |
| 8 | `mbh_mass_function` (emri_rate.py:72) — leaf, tottime 19.01 s | 1,394 | 19.01 | 15.2% |
| 9 | `R0_per_mbh` (emri_rate.py:100) — leaf, tottime 14.53 s | 1,394 | 14.53 | 11.6% |
| 10 | `kappa_cap` (emri_rate.py:167) — leaf, tottime 13.19 s | 1,394 | 13.19 | 10.6% |

For contrast, the pieces the mission brief named a priori ("GL quadrature over
candidates, g_i Hermite, ball sums") that turn out **not** to be the hot part:
`scipy.stats.norm.pdf` (the GL-candidate kernel × distance-likelihood evaluation):
1.52 s cum, 1.2%. `dist_vectorized` (physical_relations.py:226, the cosmological
distance ladder evaluated at every h): 1.09 s cum, 0.9%. `roots_hermite`
(scipy/special, the Gauss-Hermite node/weight generator called fresh inside
`completion_mass_factor_g` every time): **0.42 s cum, 0.34%** — negligible, despite
being called 1,394 times with a constant argument (`n_hermite=64`); this refutes
an a-priori hypothesis (checked before writing it up, per the verify-before-change
mandate) that memoizing `roots_hermite` would be the win.

### 1.2 What the profile actually says

`_g_ball_capped` (venue_transfer.py) is 97.7% of seed wall time, and essentially all
of that (93.1%) is inside the imported production function `completion_mass_factor_g`
(bayesian_inference/bayesian_statistics.py:1935) — the 2D completion leg's Gauss-Hermite
contraction of the dark-host mass density `phi(M)` at each quadrature node. 76.2% of
total seed time is spent inside `dark_mass_density_per_mass`
(bayesian_statistics.py:1750), which for every call re-evaluates the FULL EMRI
rate/mass-function physics chain from `master_thesis_code/emri_rate.py`
(`mbh_mass_function`, `R0_per_mbh`, `duty_cycle_Gamma`, `kappa_cap`, composed in
`R_eff_per_mbh`) element-wise on whatever `M` values the current h's quadrature nodes
produced. These are pure fractional-power-law arithmetic (`(M/M_pivot)**(-0.3)` etc.,
Babak et al. 2017 Eqs. 5/23/26-27/30/31/34) — no branching, no scipy specials, plain
`np.asarray`/`**`/`np.where` — evaluated over large arrays (per-call row counts up to
the `_G_NODE_CHUNK`-capped 4,000 rows × 64 Hermite nodes = 256,000 elements).

The GL-candidate kernel (`norm.pdf`, the `kern * p_gw` integrand) and the cosmological
distance ladder (`dist_vectorized`) — the parts named in the mission's a-priori
framing — are cheap by comparison at this problem size. **The real hot path is the
imported mass-function physics, not the venue-transfer quadrature scaffolding
around it.**

### 1.3 Full-campaign extrapolation

3.79 CPU-h/seed (measured, per the mission brief) × 982-event ΣK≈1.19M-pair unit is
consistent with this smoke's per-pair cost: 124.92 s / 20,024 pairs ≈ 6.24 ms/pair here
(n=30, includes one large K=14,586 event); scaled to ΣK=1,193,703 that is ≈ 2.07 CPU-h
— same order of magnitude as the measured 3.79 CPU-h, with the gap plausibly explained
by this smoke's event mix being lighter than the campaign median (K census: nonempty
median 84, mean 1,215.6, so a 30-event head-of-file sample is not representative of
the tail-heavy full distribution) and by cProfile/dispatch overhead at the smaller
scale. Not claimed as a re-derivation of the 3.79 CPU-h anchor — the campaign's own
smoke-measured figure is the authoritative one; this is a consistency check.

---

## 2. H-grid caching assessment

**What IS already cached (verified in code, not assumed) — no further win available
here:**
- `z_of_dl_tables[k]`, `log_alpha[k]` (per-h ladders + normalisation) —
  `closed_loop_gfrac.py:344-378`, built once per `h` in `build_context`, reused by
  every seed and every event. Correct.
- GL/GH quadrature nodes/weights (`gl_nodes`, `gl_weights`) —
  `closed_loop_gfrac.py:388`, `roots_legendre` called once at context build.
- The dark-host mass density's normalisation grid and `Z_phi` —
  `bayesian_statistics.py:1719` `_phi_dark_mass_log10_grid`, `@functools.lru_cache(maxsize=4)`,
  a 600-point log10-M table built once and reused. This is the precedent pattern the
  codebase itself already uses for exactly this kind of h-/seed-independent physics
  table.

**What is NOT cached and IS the dominant cost (§1):** `dark_mass_density_per_mass`
(bayesian_statistics.py:1750-1778) calls the cached `_phi_dark_mass_log10_grid()` only
to fetch the scalar `Z_phi` normalisation — it discards the cached `(log10_M, M, phi)`
**array** and re-evaluates `dark_mass_log10_density_unnormalised` (the full
`emri_rate.py` chain) from scratch on every call's actual `M` query points
(bayesian_statistics.py:1776). `phi(M)`'s functional *form* is h-independent (it is a
fixed population/rate density in M, no h dependence anywhere in Eqs. 5/23/26-27/30/31/34);
what varies across the 41 h-points is only *which M values get queried* (the
Gauss-Hermite nodes ride on `z_nodes`, which shift with h through the `d_L(z;h)` ladder).
That is exactly the "cheaply h-transformable" case the task asked to identify: not a
per-h constant to memoize outright, but a fixed smooth 1-D function that is currently
being re-evaluated at full physics-chain cost ~1,394 times/seed (measured at n=30;
scales with ΣK, so tens of thousands of times/seed at full N) instead of being
looked up from one shared table via interpolation, the same pattern the codebase
already applies one call earlier in the same function.

**Estimated saving:** §1.1 shows the `dark_mass_density_per_mass` → `emri_rate.py`
chain is 76.2% of seed wall time; replacing the raw re-evaluation with an
interpolation off a precomputed (h-independent) fine grid would collapse that to the
cost of an `np.interp`/spline lookup (the same order as the 600-point grid build,
paid once, not per call) — a plausible **3-5× seed-wall reduction** if the accuracy
of the substitution holds up (see below), dwarfing anything available from touching
`venue_transfer.py`'s own code.

**Why this is NOT actionable within these rails, and what it would cost if pursued:**
1. `dark_mass_density_per_mass` and its call chain live in
   `master_thesis_code/bayesian_inference/bayesian_statistics.py` and
   `master_thesis_code/emri_rate.py` / `master_thesis_code/dark_siren_injection.py` —
   all production files, outside `master_thesis_code/validation/` (RAILS), and
   `bayesian_statistics.py` is on CLAUDE.md's physics-change trigger list. A fix here
   is a `/physics-change`-gated change to the **production Bayesian pipeline B's**
   completion leg, not a validation-instrument-local tweak — its blast radius is the
   whole `--evaluate` pipeline, not just this instrument.
2. It is genuinely a numerical-accuracy question, not just a refactor: swapping an
   exact per-call evaluation for a table interpolation changes the 4th-decimal
   behavior of `g_i(z;h)` and needs a documented tolerance/convergence check (grid
   density vs. Gauss-Hermite node spread) — precisely the derivation + limiting-case
   + regression-test package `/physics-change` requires.
3. A validation-instrument-local workaround (caching only inside `venue_transfer.py`
   without touching `bayesian_statistics.py`) is not available either: the module's
   own certification claim is that it calls `completion_mass_factor_g` **verbatim**
   (module docstring, "New capabilities" section; V-T5 bit-reproduces committed
   gate output through this exact call). Any local substitute breaks that verbatim-call
   contract and would need its own new equivalence test class (a fresh divergence,
   analogous to divergence 2's documented BLAS-chunking ULP tolerance) plus
   re-running V-T5-style certification — not free, even though it stays inside
   `validation/`.

**Bottom line:** the real caching win is real, large (§1 evidence), and belongs to the
production pipeline owner (author), gated by `/physics-change`; it is out of scope for
an instrument-only perf pass. Flagging it here is the actionable output of this task.

---

## 3. GPU (cupy) port assessment

**Where the mission's named kernels actually sit, cost-wise (§1):** GL quadrature over
candidates (`norm.pdf`, the `kern*p_gw` integrand and its `@ w_gl` reduction) and the
ball sums (`np.bincount(ev, weights=c1/c2, minlength=n)`) are cheap at this problem
size (≤1.2% each, §1.1) — porting *only* those to GPU, as the mission's a-priori
framing suggested, would buy almost nothing. The genuine GPU-shaped workload is the
g_i Hermite contraction's **inner physics chain** (§2): `dark_mass_density_per_mass`
→ `R_eff_per_mbh` → {`mbh_mass_function`, `R0_per_mbh`, `duty_cycle_Gamma`,
`kappa_cap`} — large (up to 256k-element) purely elementwise arrays of fractional
power-law arithmetic (`(M/M_pivot)**p`, `np.where`), no branching, no scipy specials,
already `np.asarray`-vectorized. This shape (dense elementwise transcendental-function
arrays, no data-dependent control flow) is close to the best case for a CuPy port —
element counts per call (10^4-10^5) are well above the ~10-20 µs GPU kernel-launch
overhead floor, so a naive `xp`-pattern port (swap `np` for `cp` in `emri_rate.py` +
`dark_siren_injection.py` + `bayesian_statistics.py`'s `dark_mass_density_per_mass`)
would likely see a large (order-of-magnitude class, unverified — no GPU available on
this dev machine to measure) speedup on the dominant 76% of seed time. The GL-candidate
kernel and ball-sum reductions (`bincount` — CuPy has an equivalent) would port too,
for completeness, but contribute little to the total given §1.1.

**This inherits the exact same rails problem as caching (§2):** the workload worth
porting lives in production physics files, not in `validation/`. A GPU port sized to
where the actual FLOPs are is *by construction* a production-pipeline change, not an
instrument-local one — same `/physics-change` gate, same blast radius (affects
Pipeline B's `--evaluate` path wherever `dark_mass_density_per_mass`/`completion_mass_factor_g`
are called, not just this validation instrument), on top of GPU-specific certification
cost:

- **CPU/GPU parity cost is real and structural, not incidental.** GPU floating-point
  reductions (cuBLAS/thread-block sums) are not bit-identical to CPU/BLAS reduction
  order — this repo's own `chunk_pairs` "divergence 2" already documents and accepts
  an O(1 ULP) tolerance from *CPU* BLAS shape changes alone; a GPU port would need a
  new, explicitly documented tolerance band (almost certainly larger than 1 ULP,
  scale TBD by actual measurement) and its own equivalence test, mirroring the
  existing `test_hgrain_*` / V-T5 bit-reproduction pattern this codebase already uses
  for parallel-grain changes — i.e. the certification *process* is well-precedented
  here, but a GPU port cannot reach the "bit-identical, no new test class" bar the
  h-grain change (§4 item 1) achieved; it is closer in cost to introducing a new
  registered arm than to a transparent perf change.
- The `xp` pattern + guarded cupy imports (`.claude/rules/hpc-gpu.md`) already exist
  in the repo and are the right mechanism — this is a real asset, not a green-field
  GPU integration effort. The engineering cost is bounded (rewrite ~5 pure-arithmetic
  functions + threading `xp` through `completion_mass_factor_g`'s call chain), but the
  certification cost (tolerance derivation + new test suite + physics sign-off) is the
  larger term.

**Cluster GPU-partition queue economics vs. fat CPU nodes, for THIS workload:** per
`cluster/LAUNCHING_JOBS.md`, the documented GPU partitions are `gpu_h100_short`
(30-min wall cap, 1 GPU/task, "backfills fast") and `gpu_a100_short`/`dev_gpu_h100`
(short-wall dev/smoke queues) — there is no long-wall GPU partition in the cluster
skill comparable to `cpu_il`'s proven 4h array-job wall used by the running campaign.
Even after a GPU port collapses per-seed time by an order of magnitude, a single
`Tc(0.730)` 400-seed decision-cell task chunked the way the campaign currently is
(25 seeds/task, pool-over-seeds) would need to fit inside a 30-min GPU slot or be
re-chunked into many more, smaller array tasks — each paying the ~20 s context-build
cost (pandas read of the ~20.8M-row pruned catalogue + gate-context build, §1
methodology) freshly, and each round-tripping through the cluster's job-submission
overhead more often. `cpu_il` (128 cores, proven 4h wall, no re-chunking required) is
the economically simpler fit for this ragged, per-event-K-heterogeneous, moderate-FLOP
workload; the production EMRI *simulation* pipeline (`few` waveform generation) is the
part of this codebase GPU already pays for cleanly (dense, uniform, embarrassingly
parallel per-event work) — this validation instrument's compute profile is not the
same shape (heterogeneous K per event spanning 1 to 245,364; correctness-over-throughput
harness, not the production hot loop) and, absent a long-wall GPU partition, does not
currently have a queue-economics case superior to `cpu_il` even before certification
cost is counted. **Verdict: GPU is a real technical option (see above) but not
currently a queue-economics win for this specific instrument given the documented
partition catalogue — worth revisiting if a long-wall GPU partition becomes available,
or for the production pipeline path where the GPU investment already exists.**

---

## 4. Ranked roadmap

| # | Lever | Status | Expected gain | Certification cost | Pays off when |
|---|---|---|---|---|---|
| 1 | **Intra-seed h-grain parallelism** (`--grain h`, commit `082d1e07`) | **DONE / opt-in, shipped** | Per-seed wall drops from hours to minutes at `--workers 41` (mission brief); bit-identical to seed-grain (unit-tested cross-mode) | **Zero further cost** — already unit-tested (`test_hgrain_*`, `test_venue_transfer.py`), byte-identical by construction, no new tolerance | Immediately, for any follow-on campaign or straggler re-run; NOT applied to the running array (correctly — RAILS, and the sbatch's own comment says not to touch it) |
| 2 | **Node-topology / packing rules** | Not started — config-only, zero code change | Two independent findings (below); fixing #1 of them alone could raise effective per-node worker utilization from ~39% to ~100% on 48 of 49 registered array tasks | **Lowest of all four** — pure SLURM/sbatch parameter tuning, no physics-change gate, no re-certification (statistic is untouched; only *scheduling* changes) | Next campaign's sbatch; **recommend evaluating before caching/GPU** given near-zero cost |
| 3 | **H-grid / mass-density caching** (`dark_mass_density_per_mass` → interpolate the existing cached `phi` grid instead of re-evaluating `emri_rate.py`) | Not started — identified, not implemented (out of rails: production file) | Largest single lever found: 76.2% of measured seed wall time (§1/§2); plausible 3-5× seed-wall reduction | **High** — `/physics-change` gate on `bayesian_inference/bayesian_statistics.py` (trigger file), needs an accuracy/convergence derivation, affects production Pipeline B, not just this instrument | Worth pursuing once the author scopes it as a Pipeline-B-wide change (not a venue-transfer-only fix) — the payoff is shared across every future campaign that calls the 2D completion leg, including outside this instrument, which matches the "reusable infrastructure" goal better than a local patch would |
| 4 | **GPU (cupy) port** | Not started — assessed, not implemented (out of rails: production file; no local GPU to benchmark) | Order-of-magnitude class on the dominant 76% (§3), *if* ported to where the FLOPs actually are (not the quadrature scaffolding) | **Highest** — same production-file gate as #3, PLUS a new non-bit-identical tolerance class and equivalence-test suite (no existing precedent reaches GPU-level parity in this codebase yet) | Only once #3's caching fix is in and re-measured (a cached/interpolated `phi` may itself remove enough of the FLOPs that GPU is no longer the marginal lever) AND a long-wall GPU partition is available/justified — currently `cpu_il` wins on queue economics for this instrument's shape (§3) |

**Node-topology findings behind row 2 (both static, from `cluster/venue_transfer.sbatch`,
not measured live — the running campaign was not queried):**

- **Within-task under-utilization:** the registered sbatch requests
  `--cpus-per-task=64` uniformly, but 48 of the 49 array tasks (all `T0`, `Tb`,
  `Tc(0.690/0.730/0.770)` chunks) run `--seed-range START:25` — `mp.Pool(processes=64)`
  is asked to run only 25 seeds via `pool.map(..., chunksize=1)`, so up to 39 of the
  64 reserved cores per task are never given work. Only array task 8 (`Ta`, `COUNT=200`)
  matches its worker count to its seed count. This may be a deliberate defensive
  choice (see next bullet) rather than an oversight — flagged as an open question for
  the author, not asserted as a bug.
- **Cross-task memory-bandwidth contention (measured fact, mission brief):** 25-core
  tasks packed 5/node run ~1.7× slower per seed than uncontended (consistent with the
  brief's "~4h uncontended to ~7h contended" per-seed range). A back-of-envelope
  throughput comparison (both using the SAME 3.79 CPU-h and 1.7×-contended anchors,
  not re-measured here): 2 tasks/node at 64-cores-reserved/25-active (current sizing)
  gives ~50 active workers/node at near-uncontended speed; 5 tasks/node at
  25-cores-reserved/25-active gives 125 active workers/node at ~1.7× slower — the
  latter's aggregate seeds/(core·hour) is higher despite the per-seed slowdown, but
  its per-seed wall-clock (~7h) would **exceed the registered sbatch's 4h `--time`
  budget**, which is sized against the uncontended 3.79h anchor. This is a genuine,
  unresolved tension in the current campaign design (loose packing avoids the 4h
  overrun risk at the cost of idle reserved cores; tight packing raises aggregate
  throughput at the cost of risking mid-cell SLURM kills) — **not something this task
  is authorized to change on the running array (6253922)**; reporting it as a fact
  for the orchestrator to weigh for the *next* campaign's sbatch sizing.

---

## Files

- `results/venue_transfer_20260811/perf/profile_smoke.py` — throwaway profiling driver
  (not imported by, and does not modify, the instrument).
- `results/venue_transfer_20260811/perf/profile_smoke.pstats` — raw cProfile dump.
- `results/venue_transfer_20260811/perf/profile_smoke_top.txt` — top-25 cumulative +
  top-15 tottime tables, plus run metadata (n_events_cap, ΣK, max K, wall times).
- `results/venue_transfer_20260811/perf/PERF_ROADMAP.md` — this document.

---

## 5. Post-swap addendum (2026-08-12, branch `perf/realistic-venue`) — RATIFIED by author 2026-08-12

Lever #3 was implemented on this branch as a `/physics-change`-gated swap in
`bayesian_statistics.py::dark_mass_density_per_mass`, with two premise revisions
found by measurement (the author's re-check-the-premise discipline, applied):

1. **`np.interp` refuted:** the roadmap's assumed interpolation route measured
   2.5x SLOWER than the exact chain on the dev machine (numpy's fractional pow
   is libmvec-vectorized; `np.interp`'s per-element search dominates).
   Replaced by the analytically equivalent minimal form: phi(M) is an exact
   two-segment power law (Gamma min-cap never binds, max 0.1253 < 1), so
   `ln phi` is affine in `log10 M` per segment with one kink at `log10 M = 5`;
   the default path now evaluates the two affine branches with coefficients
   derived numerically from exact-chain calls at nodes {4, 5, 7} (never
   re-typed). Adversarial verification: CONFIRMED, worst in-band deviation
   1.8e-15 (14 ULP, 2M-sample sweep). `exact=True` restores the verbatim chain.
2. **Realized gain 1.42x, not 3-5x:** post-swap seed wall 88.03 s vs 124.92 s
   baseline (`COUNTERFACTUAL_SMOKE.md`; in-process exact-vs-affine 1.51x). The
   roadmap's 3-5x assumed the chain was compute-bound; it is memory-bandwidth
   bound — `dark_mass_density_per_mass` remains 65.7% of seed wall even as
   pure affine+exp arithmetic (~8 full array passes over multi-million-element
   temporaries per call).
3. **Registered numerical tolerance (new divergence class, V-T5 style):** the
   swap is NOT bit-identical end-to-end. Counterfactual smoke (Tc(0.730),
   real_k/glade, n=30, seed 20260808+44000): `map_1d` and all non-2D fields
   byte-identical; 11 2D-channel leaves (`edge_mass_2d`, `ln_post_2d[i]`,
   `map_2d_refined`, `mean_2d`, `pit_2d`, `sum_dlog_gfrac_dh`) differ by
   max abs 2.842e-12 / max rel 5.150e-9 — accumulated double-precision
   rounding from the changed evaluation order, same class as the accepted
   `chunk_pairs` divergence 2. Registered tolerance for exact-vs-affine
   comparisons: rel 1e-8 on 2D-channel scalars.
4. **GPU (lever #4): NO-GO reaffirmed post-swap.** The residual hot spot is
   bandwidth-bound elementwise work — technically GPU-ideal, but both §3
   gating conditions still fail (no long-wall GPU partition; a further
   non-bit-identical certification class). Revisit only per §3's conditions.
5. **Next levers (not implemented, future /physics-change intake):**
   (a) temporary-array fusion in `dark_mass_density_per_mass` /
   `completion_mass_factor_g` (~1.2-1.3x, cheap, same tolerance class);
   (b) semi-analytic Gauss-Hermite x piecewise-power-law contraction of
   `completion_mass_factor_g` — the integrand is Gaussian x exp(affine) per
   segment, so the inner contraction has closed-form pieces (erf terms); could
   remove most of the residual 65.7%. Needs a full derivation package.
