# DRAFT — A-FULL: the correct-form venue estimator (candidate for the physics-change gate)

**Date:** 2026-08-15 · **Authorized:** ledger row #109 item 3 (**drafting only** — registration
and running are separate author gates, A8-v2; the `/physics-change` slot remains EMPTY) ·
**Status: DRAFT, PRESENTED, NOT ADJUDICATED.** Residuals carried per row #109 item 4: the +39
full-dose leftover and the 2D-only +129 excess are stated residuals of this draft.

**Pre-measurement:** `l4_afull_premeasure.py` → `L4_AFULL_PREMEASURE_output.json` (committed
data only; mirror geometry; no instrument time — row #108 item 1 class).

---

## 1. The correct form, derived from the generator

Per pinned event (Part 1 §1 generative model, unchanged), the estimator sees d_obs and the ball
{z_obs,k, σ_k}, K. Marginalizing host identity (flat 1/K — the ball inserts the host uniformly)
and the host's true redshift:

    L_e^full(h) = (1/K) Σ_k ∫ dz  π_host(z; h) · N(z_obs,k ; z, σ_k)
                                 · N(d_obs ; d_L(z,h), σ_d·d_L(z,h))

with two ingredients, each the correct-form counterpart of a located defect (a third initially
hypothesized ingredient — kernel renormalization — is REFUTED by the pre-measurement, §2 item 3):

1. **The GW factor is a density in d_obs** (Part 1 F3): `N(d_obs; μ, σ_d μ)`, μ = d_L(z,h) —
   in code `norm.pdf(d_obs/d_L; 1, σ_d)/d_L`. This carries BOTH the missing prefactor (D2) and
   the μ-scale exponent (D3) in one object. **It does not by itself remove the mass-growth
   term** — the density's z-space mass is h/D′(z\*), which grows with h exactly as the coded
   ratio form's mass does (Part 2 §2). The cancellation must come from the population structure
   (ingredient 2), which is why the tilt is *pre-measured* below rather than asserted.
2. **The host-z prior π_host is the SELECTED population density** — the normalized α-integrand:

       π_host(z; h) = w_pop(z; h) · S̄_φ(z; h) / α(h),   α(h) = ∫ w_pop S̄_φ dz.

   This resolves the F2 puzzle cleanly: **α does belong in the pinned venue — but only as the
   normalization of the numerator's selection-weighted prior**, never as a bare −N ln α against
   an unweighted numerator. The pinned events were drawn upstream from an SNR-selected
   population; the correct conditional prior for a host's unknown z is the selected density,
   whose normalization is α. The coded estimator keeps the −N ln α while omitting w_pop·S̄_φ
   from the numerator — the un-cancelled α-tilt (+1400.6) is exactly this broken pairing (D1+D4
   are one defect, not two). Notably the **calibration-gate estimator already carries the full
   pairing** (`closed_loop_gfrac.py:597–601`: numerator `w_pop · p_gw · S̄_φ` against the same
   α) — the venue/production event term is the place it was dropped.
3. **The kernel is NOT renormalized.** The W_k(h) division (A-REN mechanics) was staged as
   candidates C/E and REFUTED by the pre-measurement (§2 item 3): under the paired prior it
   overshoots by ≈ −1100 nats/h. Part 1 F1 stands — the per-candidate window truncation is a
   ball-construction device that normalizes out of the h-posterior; renormalizing it *introduces*
   spurious h-structure. (A-JREN's empirical REN benefit on the coded form was partial
   cancellation of a defect the paired prior removes properly.)

## 2. Pre-measured venue tilts (mirror geometry, 15 MN0X seed replays, 1D)

The candidates are staged so each ingredient's effect is visible (T_cand = candidate-sum tilt;
"paired" adds the coded −N ln α analytically; the coded base reference is T_cand = +1243.4,
T_base = +2644.0):

| candidate | ingredients | T_cand (no α) | T paired (with −N ln α) | f_i = 0.25 paired |
|---|---|---:|---:|---:|
| coded base (reference) | ratio-pdf GW, bare kernel | +1243.4 ± 46.5 | **+2644.0 ± 46.5** | +3535.5 |
| FULL-A | d_obs-density GW only | +1128.9 ± 46.3 | +2529.4 | — |
| FULL-B | + w_pop numerator weight | −1504.2 ± 46.2 | **−103.6 ± 46.2** | — |
| FULL-D | + S̄_φ (full selected prior w_pop·S̄_φ/α) | −1217.1 ± 47.0 | **+183.4 ± 47.0** | +998.0 |
| FULL-C | B + kernel renorm | −2641.5 ± 15.2 | −1240.9 | +29.9 |
| FULL-E | D + kernel renorm | −2299.4 ± 15.9 | −898.8 | +342.0 |

**Reading (numbers, then the three structural findings):**

1. **FULL-A confirms §1 ingredient 1's caveat exactly:** the density form alone leaves the tilt
   essentially unchanged (+2529 paired vs +2644 coded) — the mass-growth term is NOT cancelled
   by the prefactor/exponent repair; it is cancelled by the population pairing.
2. **The paired selected-population prior is the repair that works:** FULL-B −103.6 ± 46.2 and
   FULL-D +183.4 ± 47.0 paired — a 14–25× reduction of the +2644 tilt, the first configurations
   in the entire thread with |T| at the ~10² level. In displacement-law units (Ā ≈ 7.0×10⁴ from
   bias 0.0373 ↔ T 2625) the implied bias is ~−0.0015 (B) / +0.0026 (D) against the original
   +0.0373. Neither is exactly zero: B sits −2.2σ, D +3.9σ from zero on 15 seeds.
3. **The kernel renormalization does NOT belong in the correct form:** adding it (C vs B, E vs D)
   swings the tilt by ≈ −1100 nats/h — the same magnitude as the instrument REN effect measured
   at L4-T1 (−978) — overshooting well past zero. This vindicates Part 1 F1 (the per-candidate
   window truncation is a ball-*construction* device that normalizes out; dividing by W_k
   *introduces* h-structure rather than removing it). A-JREN's REN half "helped" the coded form
   only by cancelling part of a defect the paired prior removes properly.
4. **The FULL-D residual is the drift channel, still:** its dose structure (+998 at f_i = 0.25 →
   +183 at 1.0) tracks the coded form's drift leftover (+867 → +39) — the ball's empirical
   density acting against the model prior, the row-#109-item-4 carried residual, now bounded at
   ~7% of the original defect at full dose.
5. **B-vs-D asymmetry (−104 vs +183):** the S̄_φ factor's own z-gradient adds ~+290 nats/h of
   drift-class tilt. Whether the registered candidate should be B (w_pop only) or D (the
   formally complete selected prior) is decision 1's sub-question — D is the derivation's answer;
   B's smaller residual at full dose is not by itself a reason to prefer it (two wrongs can
   partially cancel).

## 3. The code form (for the eventual `/physics-change` proposal — not installed here)

As an `estimator_variant` in `venue_transfer._channel_terms_at_h` (venue instrument first; the
production `bayesian_statistics.py` change is a separate, later physics-change item):

```python
elif estimator_variant == ESTIMATOR_VARIANT_A_FULL:
    # d_obs-density GW factor: N(d_obs; d_L, sigma_d*d_L) = pdf(d_obs/d_L; 1, s)/d_L
    ratio = d_obs_p[rows_q][:, None] / d_L_n          # note: d_obs/d_L, not d_L/d_obs
    p_gw_full = norm.pdf(ratio, loc=1.0, scale=sig_p[rows_q][:, None]) / d_L_n
    # selected-population prior (normalization alpha(h) is the existing -n*log_alpha term)
    z_s, s_phi = gctx.cl_ctx.s_phi_tables[k]
    w_sel = cl._w_pop(z_flat, h).reshape(z_nodes.shape) * np.interp(
        z_nodes, z_s, s_phi
    )
    # NO kernel renormalization (refuted, SS2 item 3) and NO Jacobian (F3: density form)
    integ = kern * p_gw_full * w_sel
```

The point branch (σ_k = 0) takes the same factors evaluated at z_obs. The existing
`− n·log_alpha[k]` term is RETAINED (it is the prior's normalization, ingredient 2). No other
term changes. 2D: the g_i factor rides along unchanged; whether the 2D +129 excess survives the
full form is a stated residual (row #109 item 4).

## 4. Predicted outcomes and their falsifiers (for the registration, when granted)

Registered-arm predictions if decision 2 is granted with candidate FULL-D (bands to be
finalized at registration per decision 3; SEs from the §2 pre-measurement):

- **P1 (tilt):** T(A-FULL, 1D, full dose) = +183 ± 47·√(15/N) nats/h; falsified outside ±3σ.
- **P2 (bias):** MAP bias ≈ T/Ā ≈ +0.003 ± 0.001 — a 12× reduction from +0.0373; the
  displacement law (ratio ≈ 1, validated 0.989–1.15 across 17 cells) supplies the conversion.
- **P3 (coverage):** with |bias| ~ 0.5·post_sd, HPD coverage should be restored to within
  ~10 percentage points of nominal at 50/68/90 — the first coverage-restoring candidate.
- **P4 (dose structure):** T(f_i = 0.25) ≈ +1000 ± 80·√(15/N) — the drift residual's signature;
  a flat dose curve would falsify the residual attribution.
- **P5 (specificity):** removing the S̄_φ factor (the B form) shifts T by ≈ −290 nats/h;
  adding kernel renorm shifts it by ≈ −1100 — both directions and magnitudes are pre-stated
  kill tests of the mechanism account.

## 5. What would still be open after A-FULL

- The pool-vs-model population mismatch: π_host uses the MODEL selected density (w_pop·S̄_φ/α);
  the pinned events realize the injection pool's empirical density. Any residual tilt of the
  full form bounds this mismatch — it is a *population-model* systematic, not an estimator
  defect, and connects to the paper's selection-consistency framing ([[inference-consistency]]
  thread; Gray-convention finding, re-presented separately).
- The 2D-only +129 excess and the +39 full-dose leftover (carried residuals).
- Production transfer: the venue variant is the instrument test; the `bayesian_statistics.py`
  repair is its own physics-change proposal with the 5-step gate, after the venue verdict.

## 6. Decisions this draft asks for (when the author takes it up)

| # | decision | tag |
|---|---|---|
| 1 | Adopt §1's derivation (incl. the F2 reinterpretation: α is the prior normalization; D1+D4 are one broken pairing) as the A-FULL candidate definition | **[RULE]** |
| 2 | Register A-FULL (A8-v2: bands from the §2 pre-measurement, fresh seeds, scorer pre-committed, xhigh pre-registration verifier) and run on the cluster (~25 CPU-h at N = 25) | **[DO]** |
| 3 | Whether §2's pre-measured tilt (committed-data mirror) may seed the registered bands directly, or bands must be re-derived independently | **[RULE]** |

*Append-only from its commit. No production code is touched by this draft.*
