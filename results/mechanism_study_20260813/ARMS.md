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
