# ARMS — exact code form of every mechanism-isolation arm

Companion to `PREREGISTRATION_MECHANISM_ISOLATION.md`. **Fixed at registration; no arm's code form
may be adjusted after any arm is read.** Every statement below is a claim about the implementation
and is unit-tested (`darksiren_emri_test/validation/test_venue_transfer_arms.py`).

## The single instrument change

One new `VenueConfig` field:

```python
dose_target: str = "all"     # "all" | "host" | "impostors"
```

`"all"` is the registered campaign behaviour and **must** remain the default, so every existing
call site, the committed campaign path, and the V-M5 golden are untouched.

`draw_ball_pinned` gains one additive keyword:

```python
def draw_ball_pinned(vctx, universe, rng, *, return_host_mask: bool = False):
```

With the default `False` it returns exactly what it returns today, consuming the identical RNG
draws in the identical order. With `True` it additionally returns the boolean host mask, built as

```python
is_host = np.concatenate([np.ones(n, bool), np.zeros(total_imp, bool)])[order]
```

reusing the *same* `order` the function already computes for the lexsort. The mask is a pure
relabelling of draws that already happen — **it consumes no randomness of its own.**

## The dose application (`_draw_seed_realization`)

Registered order of RNG consumption is unchanged in every arm: noise → ball → σ_z vector →
standard-normal scatter vector. The arms differ **only** in a mask applied *after* all four draws:

```python
sigma_pairs = draw_member_sigma_z(context, ball.z_obs, rng)     # unchanged draw
noise = rng.standard_normal(ball.z_obs.size)                    # unchanged draw
mask = {"all": full_true, "host": host_mask, "impostors": ~host_mask}[vcfg.dose_target]
ball.z_obs = ball.z_obs + np.where(mask, sigma_pairs * noise, 0.0)
sigma_pairs = np.where(mask, sigma_pairs, 0.0)
```

**Both lines are masked, and that is load-bearing.** An undosed candidate has an exact redshift, so
the estimator must be told its kernel width is zero — it then takes the point-evaluation branch for
that candidate. Masking only the scatter and not the σ vector would hand the estimator a kernel
wider than the truth, i.e. deliberate misspecification, and would confound the read with the very
thing the campaign's matched-model principle exists to exclude.

`flat035` mode takes the identical treatment; `zero` mode is unaffected (no dose to target).

## Arm table

| arm | `sigma_mode` | `dose_target` | h_true | N | seeds |
|---|---|---|---|---|---|
| **N-0** | `glade` | `all` | 0.730 | 15 | base+50000…50014 |
| **E1-host** | `glade` | `host` | 0.730 | 15 | base+50100…50114 |
| **E1-imp** | `glade` | `impostors` | 0.730 | 15 | base+50200…50214 |

base = 20260808. All other configuration is the campaign's decision cell verbatim: pinned 982
events, `balls="real_k"`, canonical 41-point grid, `n_events_cap=None`, `chunk_pairs=16384`,
the four §1 pins.

## What is NOT changed

- No estimator code. `_channel_terms_at_h`, `log_channel_posteriors_ball_sigma_vector` and
  `_g_ball_capped` are byte-identical across all three arms — verifiable by `git diff`, and the
  reason E1 is decisive rather than suggestive.
- No production module.
- No RNG stream, order, or draw count.

## Registered null checks

- **AR-1** — with `dose_target="all"`, every arm's realisation is **bit-identical** to the current
  registered path for the same seed (`z_obs`, `sigma_pairs`, `K_sum`). Unit-tested.
- **AR-2** — `host_mask.sum() == 982` and `mask` selects exactly one candidate per event, for every
  seed and every arm.
- **AR-3** — across the three arms at a fixed seed, `K_sum`, `event_idx`, the pre-dose `z_obs`, the
  σ vector and the scatter vector are bit-identical; **only the mask differs.** This is the precise
  form of V-M2 (generator invariance) for this study — the post-dose `z_obs` necessarily differs
  between arms, since that difference *is* the experiment.

## Stage-2 arms (2026-08-14, PREREGISTRATION_M2PRIME_ABLATION.md)

Companion to `PREREGISTRATION_M2PRIME_ABLATION.md` §3. **Fixed at registration; no arm's code form
may be adjusted after any arm is read.** Every statement below is a claim about the implementation
and is unit-tested (`darksiren_emri_test/validation/test_m2prime_ablation_arms.py`).

### The single instrument change

One new `VenueConfig` field (default-off, additive):

```python
estimator_variant: str = "base"     # "base" | "m2prime_jacobian" | "null_scale_1p7"
```

`"base"` is the registered default and **must** remain byte-identical to the pre-switch estimator,
so every existing call site, the committed campaign path, and the V-T5/V-M5 goldens are untouched.
The field is threaded verbatim through `log_channel_posteriors_ball_sigma_vector`,
`log_channel_posteriors_ball_sigma_vector_hgrain`, `_h_task`/`_H_STATE`, and both per-seed drivers
(`run_seed_venue`, `run_seed_venue_hgrain`) down to `_channel_terms_at_h`, where the switch is read.

### The kernel-branch integrand switch (`_channel_terms_at_h`)

Exact diff hunk (the only place any float arithmetic changes; the point branch, `c1[rows_p]` /
`c2[rows_p]` below, is untouched by construction — it is a disjoint code path reached only for
`sig_c <= 0.0` rows):

```diff
             p_gw = norm.pdf(d_L_frac, loc=1.0, scale=sig_p[rows_q][:, None])
             kern = norm.pdf(z_nodes, loc=zo[:, None], scale=so[:, None])
-            integ = kern * p_gw
+            if estimator_variant == ESTIMATOR_VARIANT_BASE:
+                integ = kern * p_gw
+            elif estimator_variant == ESTIMATOR_VARIANT_M2P_JACOBIAN:
+                # A-M2' (prereg §3): J = |d d_L/dz| / d_obs, per node, via
+                # central difference of dist_vectorized at the same h
+                # (registered step eps=1e-6 in z, deterministic, no RNG —
+                # dist_derivative is analytic but scalar-only and unfit for
+                # this vectorized (n_rows, n_quad) loop, see module header).
+                eps = M2P_JACOBIAN_EPS_Z
+                z_flat = np.maximum(z_nodes.reshape(-1), 1e-8)
+                d_hi = np.asarray(dist_vectorized(z_flat + eps, h=h), dtype=np.float64)
+                d_lo = np.asarray(
+                    dist_vectorized(np.maximum(z_flat - eps, 1e-8), h=h), dtype=np.float64
+                )
+                dd_dz = ((d_hi - d_lo) / (2.0 * eps)).reshape(z_nodes.shape)
+                jac = dd_dz / d_obs_p[rows_q][:, None]
+                integ = kern * p_gw * jac
+            elif estimator_variant == ESTIMATOR_VARIANT_NULL_SCALE:
+                # A-NULL (prereg §3): z- and h-independent literal constant.
+                integ = (kern * p_gw) * NULL_SCALE_FACTOR
+            else:
+                raise ValueError(f"unknown estimator_variant '{estimator_variant}'")
             c1q = half * (integ @ w_gl)
             g = _g_ball_capped(
                 gctx, universe, ev[rows_q], z_nodes, d_L_frac, valid, node_chunk=g_node_chunk
             )
             c2q = half * ((integ * g) @ w_gl)
```

`c1q` and `c2q` (hence both `c1`/`c2`, hence both channels 1 and 2) are formed from the SAME
`integ` — the multiplication happens once, before either quadrature sum, matching prereg §3
("multiplied into `integ` (hence into both c₁ and c₂)"). The `"base"` branch performs the exact
same operation (`kern * p_gw`) in the exact same order as the pre-switch code — no unconditional
multiply by 1.0 anywhere on that path.

Registered constants (module-level, `darksiren_emri/validation/venue_transfer.py`):

```python
ESTIMATOR_VARIANT_BASE = "base"
ESTIMATOR_VARIANT_M2P_JACOBIAN = "m2prime_jacobian"
ESTIMATOR_VARIANT_NULL_SCALE = "null_scale_1p7"
M2P_JACOBIAN_EPS_Z = 1e-6
NULL_SCALE_FACTOR = 1.7
```

**Why central difference, not `physical_relations.dist_derivative`:** that function is analytic and
correct, but scalar-only — it rebuilds a 1000-point `np.trapezoid` integral per call and has no
array fast path — so calling it element-wise over the `(n_rows, n_quad)` node grid inside this hot
loop would be a severe, unregistered performance regression. Prereg §3's own fallback clause
("analytically if `physical_relations` exposes the derivative, else by central difference... with
registered step ε = 1e-6 in z") is exercised as its "else" branch. Flagged here per instruction,
not decided silently.

### The two cell-spec definitions

```python
M2P_CELL_SPECS: dict[str, VenueCellSpec] = {
    "AM2P": VenueCellSpec(
        "AM2P", "A-M2'", "real_k", "glade", (0.730,), (25,), (53000,), "all",
        estimator_variant=ESTIMATOR_VARIANT_M2P_JACOBIAN,
    ),
    "ANULL": VenueCellSpec(
        "ANULL", "A-NULL", "real_k", "glade", (0.730,), (15,), (50000,), "all",
        estimator_variant=ESTIMATOR_VARIANT_NULL_SCALE,
    ),
}
```

Stamped via a third registry entry in `preregistration_path_for_cell`:

```python
M2P_PREREG_PATH = "results/mechanism_study_20260813/PREREGISTRATION_M2PRIME_ABLATION.md"
...
if cell in M2P_CELL_SPECS:
    return M2P_PREREG_PATH
```

checked before `MECH_CELL_SPECS`/`SCAN_CELL_SPECS`, so `AM2P`/`ANULL` stamp
`PREREGISTRATION_M2PRIME_ABLATION.md`, never the parent mechanism-isolation or venue-transfer
documents.

### Seed blocks

| arm | `sigma_mode` | `estimator_variant` | h_true | N | seeds |
|---|---|---|---|---|---|
| **A-M2'** | `glade` | `m2prime_jacobian` | 0.730 | 25 | base+53000…53024 (fresh, disjoint) |
| **A-NULL** | `glade` | `null_scale_1p7` | 0.730 | 15 | base+50000…50014 (**PAIRED** with MN0X's first 15 seeds, by design — prereg §3) |

base = `VT_BASE_SEED` = 20260808. All other configuration is the campaign's decision cell verbatim,
identical to MN0X in every respect but the estimator switch: pinned 982 events, `balls="real_k"`,
`sigma_mode="glade"`, canonical 41-point grid, `n_events_cap=None`, `chunk_pairs=16384`, the four
§1 pins.

**A-NULL's seed block is deliberately NOT disjoint from MN0/MN0X** — that pairing is the
registered design (prereg §3 DS-N1: exact per-seed MAP grid equality against MN0X's stored
records, ln-posterior shifted by exactly `N ln 1.7`), the one documented exception to the
seed-plan-disjointness rule, and is asserted as such (not merely permitted) by
`test_seed_plan_disjointness_except_registered_anull_pairing` in
`darksiren_emri_test/validation/test_m2prime_ablation_arms.py`.

### What is NOT changed

- No estimator code outside the single `if/elif/elif/else` block above. Everything upstream
  (quadrature nodes, window clipping, `g_i` completeness factor, normalisation, `−N ln α(h)`
  subtraction, the `−745`/event finite-zero convention) is byte-identical across all three variants
  — verifiable by `git diff`.
- No production module (`darksiren_emri/validation/` only).
- No RNG stream, order, or draw count — the switch is read-only on already-drawn arrays.
- The point branch (`sig_c <= 0.0` rows: `c1[rows_p]`, `c2[rows_p]`) is untouched in every variant.

### Registered null checks (unit-tested)

- **inertness** — on any case where every candidate's σ_z = 0, `"m2prime_jacobian"` output is
  bit-identical to `"base"` (both variants read only the point branch, which the switch never
  touches).
- **shift law** — on any case where every candidate's σ_z > 0, `ln_post("null_scale_1p7")` equals
  `ln_post("base") + N·ln(1.7)` at every h, within `rtol 1e-12`, with an identical per-h argmax
  index (DS-N1's exact-equality form, reproduced on a small synthetic case rather than the full
  982-event pool).
