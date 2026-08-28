# /physics-change PRESENTATION GATE — [HIER] θ-hook (C1 + C2) — 2026-08-28

**Status: PRESENTED, awaiting author approval. No code has been written.**
Ordered by PA-HIER-28 (item 3 = GATE): the θ-hook edit to `bayesian_statistics.py` takes the
FULL /physics-change protocol — this presentation before code, a byte-identity regression test
at θ = (0,1), and ledger rows in `docs/gates/PHYSICS-GATE-LEDGER.md`. Authored at top tier per
the tiering table (recon: 1× sonnet; chair: orchestrator). Spec source of truth:
`PREREGISTRATION_HIER_HTHETA_20260826.md` (PA-HIER-21 site table, PA-HIER-28/-29/-30).

## 0. Scope of the change

One commit, contents fixed by the registration:

1. **C1 — the θ hook**: a `(theta_b, theta_s)` pair threaded into
   `BayesianStatistics.evaluate()`, reparametrizing the host-z kernel at estimator sites
   2.1 / 2.2 / 2.3 only, with a literal early-return/skip at `(0.0, 1.0)` (GATE T-ID).
2. **C2 — instrumentation**: per-term `ln L` diagnostics + a per-site toggle switch
   (PA-HIER-23 form) — no numerical change when toggles are off.
3. **Bundled, pre-authorized by the prereg**: the `correspondence_1d.py:1173` stale docstring
   fix (`:5908-5909` → `:6223-6224` / `:6878-6879`); the PA-HIER-30 free hardening
   `_B0I_ZTRUE_GRID_N` 401 → 4001; the PA-HIER-11 twin-parity regression (site 2.7).
4. **Explicitly NOT touched**: `correspondence_1d.host_z_error_eff` and every generator-side
   site (2.4 — GATE GEN-FROZEN, PA-HIER-2): θ must never move the data.

## 1. OLD formula (exact, at each in-scope site)

**Site 2.1 — scalar `single_host_likelihood`, `bayesian_statistics.py:6223-6224` (+ window
:6232-6244, kernel :6247):**

```python
sigma_z_pv = (1.0 + host_z) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
host_z_error_eff = float(np.sqrt(host_z_error**2 + sigma_z_pv**2))
# window: host_z ± integration_limit_sigma_multiplier * host_z_error_eff (lower-floored)
galaxy_redshift_normal_distribution = norm(loc=host_z, scale=host_z_error_eff)
```

**Site 2.2 — batch `single_host_likelihood_batch` (production's actual dispatch path),
`:6878-6879` (+ window :6888/:6905-6907):** identical formula, vectorized.

**Site 2.3 — global selection denominator `_smeared_global_pdet_expectation`,
`:1669-1672` (+ window/centre :1675-1679); called from
`precompute_global_catalog_selection` at `:4010/:4018/:4062`:**

```python
sigma_z_pv = (1.0 + z_g) * SIGMA_V_PEC_KM_S / SPEED_OF_LIGHT_KM_S
sigma_eff = np.maximum(np.sqrt(z_err_g**2 + sigma_z_pv**2), 1e-10)
# per-slice: zc = z_g[sl]; lo = max(zc − 4σ, 1e-6); hi = max(zc + 4σ, lo + 1e-12); c = (hi+lo)/2
```

Site 2.3 today has **no `b` analog at all** (GATE D3 clause (c)) — the hook adds one.

## 2. NEW formula (the registered reparametrization)

θ = (b, s), applied at each in-scope site to the **outputs** of the existing width fold:

```
z̃      = host_z + b · (1 + host_z)          # kernel centre shift
σ̃_eff  = s · host_z_error_eff               # kernel width scale   (s > 0)
```

then every downstream consumer at that site uses (z̃, σ̃_eff) in place of
(host_z, host_z_error_eff): the Normal kernel becomes `norm(loc=z̃, scale=σ̃_eff)`, the
integration windows become `z̃ ± multiplier·σ̃_eff` (existing lower floors unchanged, applied
after the substitution), and at site 2.3 `zc → z̃_g`, `sigma_eff → max(s·sigma_eff_raw, 1e-10)`
(floor applied after scaling, keeping the :1670 delta-limit comment true).

**Pinned order-of-operations (registered here, decide-once):** `sigma_z_pv` is computed from
the RAW `host_z` — i.e. `b` shifts the centre AFTER the peculiar-velocity width fold, and `s`
scales the folded width. Rationale: θ models a systematic in the catalogue's *reported*
redshift; the PV dispersion attaches to the true sight-line. The alternative (folding at
`1 + z̃`) differs only at order `b·σ_pv`, and is numerically ZERO today because
`SIGMA_V_PEC_KM_S = 0.0` (the :1175 docstring's own disclosure) — moot but pinned so the
regression suite is well-defined if that constant is ever set.

**Early-return (GATE T-ID):** a single dispatch-level guard in `evaluate()` — when
`(theta_b, theta_s) == (0.0, 1.0)` the original unparametrized code path runs, giving
bit-for-bit identity by construction (not by floating-point luck of `x + 0.0` / `x * 1.0`).

**Threading (no formula content, listed for reviewability):** `evaluate()` (:3313, alongside
the existing instrument kwargs) → `p_D` (:4625) → `p_Di` (:4846) → `_starmap_host_batches`
(:7405) → `single_host_likelihood_batch`; and `evaluate()` → the three
`precompute_global_catalog_selection` calls. Defaults `theta_b=0.0, theta_s=1.0`.

## 3. Reference

The (bias, scatter) affine parametrization of a photometric-redshift error kernel,
`Δz = b·(1+z)` with a multiplicative scatter scale, is the standard photo-z nuisance model:

- **Ma, Hu & Huterer (2006), ApJ 636, 21, arXiv:astro-ph/0506614, §2** — photo-z systematics
  parametrized as a per-z bias and scatter of the Gaussian kernel; the (1+z) scaling of both
  is the field convention adopted there and since.
- DES Y1 / KiDS practice (e.g. Hoyle et al. 2018, arXiv:1708.01532) applies exactly the
  Δz-shift form to survey n(z) calibration.

This is an **instrument** (a nuisance reparametrization for the [HIER] profile-likelihood
stages), not a change to any physical law: at the production default it is the identity.

## 4. Dimensional analysis

All quantities dimensionless: `z`, `host_z_error_eff`, `sigma_z_pv` are redshifts;
`b` is dimensionless (Δz per unit (1+z)); `s` is a pure scale factor, `s > 0`.
`z̃ = z + b(1+z)` — redshift + (dimensionless)·(dimensionless) → redshift ✓.
`σ̃ = s·σ` — redshift ✓. Windows `z̃ ± m·σ̃` — redshift ✓. Registered supports keep the
kernel proper: `|b| ≤ b_max = 0.0661` (PA-HIER-29, 2× the measured catalogue median of
`z_err/(1+z)` = 0.033038) and `s ∈ [0.5, 2.0]` (log-uniform grid), so `σ̃ > 0` always and
the site-2.3 floor `1e-10` is never the binding constraint except where it already is.

## 5. Limiting cases

1. **Identity, θ = (0,1)**: z̃ = z, σ̃ = σ; with the literal early-return this is *exactly* the
   production path — the regression test asserts a bit-identical posterior on a fixed
   mini-evaluate (and the PA-HIER-11 twin-parity assertion: site 2.7's integration-testing
   twin still reproduces sites 2.1/2.2 at θ = (0,1)).
2. **s → 0⁺ (with the floor)**: the kernel tends to a delta at z̃; the site-2.3 smeared
   expectation collapses to the point evaluation at z̃ — exactly the behaviour the existing
   :1670-1671 comment documents for the σ→1e-10 limit, now centred at the shifted z̃. ✓
3. **b sign sanity**: b > 0 moves every host kernel to higher z at fixed data; a
   higher-z host explains the same GW distance with a larger H₀ trend in the single-host
   likelihood — the direction the [HIER] stage-F grid is designed to profile over
   (qualitative check only; no verdict rests on it).

## 6. Regression tests (written BEFORE the change, per protocol)

1. θ = (0,1) byte-identity: fixed-seed mini-evaluate posterior array `np.array_equal` old vs
   new (asserting the OLD value first, then unchanged after the edit).
2. Twin parity (PA-HIER-11): site 2.7 reproduces sites 2.1/2.2 at θ = (0,1); if the hook's
   refactor changes the shared expression's shape, the twin updates in the same commit.
3. Production-default test: `evaluate()` called WITHOUT θ kwargs takes the early-return branch
   (guard against a future default drift).
4. Grid hardening: `_B0I_ZTRUE_GRID_N = 4001` re-run of the PA-HIER-30 leg-(b) spot-check —
   the straddling-host width residual (~13-15% at 401 nodes) must close to ≤1%.

Post-implementation: sign/units/limit checklist, `# Eq./§2, Ma, Hu & Huterer (2006),
arXiv:astro-ph/0506614` reference comment above the reparametrization lines, `[PHYSICS]`
commit, three ledger rows in `docs/gates/PHYSICS-GATE-LEDGER.md`.

## 7. Decision table

| # | tag | decision |
|---|---|---|
| 1 | **[RULE]** | Approve this presentation (items 1-5 above), including the pinned order-of-operations (b after the PV fold) and the dispatch-level early-return form. Approval authorizes writing the code exactly as presented; any deviation returns here. |
| 2 | **[RULE]** | Confirm the bundle (docstring fix, 401→4001 hardening, twin-parity test) rides in the same `[PHYSICS]` commit, per PA-HIER-21/-30's own language. |
| 3 | **[DO]** | After merge + green regression suite: build the option-B quadrature grid (b from ±0.0661, s log-grid, H_GRID_41) and run S0-A — all verdicts capped REPORTED-ONLY (item 9 = AFFORDABLE). |

*Presented 2026-08-28. Every file:line above was re-verified at source during authoring.*
