# MECHANISM_NOTE.md — r-highz-completion: the per-event likelihood identity for a zero-candidate event

Author D (top-tier prereg author), 2026-09-04. Read-only trace of `darksiren_emri/bayesian_inference/
bayesian_statistics.py` at HEAD (the iiib re-baseline ran at commit `1ec9514d`, `run_metadata_0.json`;
`p_Di` lines quoted below are HEAD line numbers — the design gate re-pins them against `1ec9514d`).
Every number in §4 is from a 5-row slice (events 0–4, h = 0.73) opened ONLY to check computability;
no registered statistic was computed (memory `gate-reviewers-must-not-compute-registered-statistic`).

## 1. Which branch production takes (resolved flags, `run_metadata_0.json`, iiib re-baseline)

`normalization_mode = absolute_marginal` · `selection_in_completion_numerator = fused` ·
`catalogue_global_selection = phi` · `completion_b_scale = derived` · `completion_event_measure = ratio` ·
`freeze_g_frac_ref_h = None` · `mass_filter_geometry = linear`, `k = 1.5` · `catalogue_leg_1d_mass_aware = auto`
(resolves to "on" post-flip, row #286). The harness universes carry the same 13 resolved tokens (row #347 (1),
67/67). Hence in `p_Di` (`:5932`) the assembly is the Path-A φ branch, `:6684`
`elif _use_g_inside and self.h in getattr(self, "_beta_G_phi_table", {}):`.

## 2. The assembly identity, quoted

```
:6741  combined_without_bh_mass = float(
:6742      (_cat_num_weight_no_bh * L_cat_without_bh_mass + B_num_phi) / D_tilde_phi
:6743  )
:6744  combined_with_bh_mass = float(
:6745      (alpha_G_phi * L_cat_with_bh_mass + B_num_wbh_phi) / D_tilde_phi
:6746  )
:6726  _den_used = D_tilde_phi if D_tilde_phi > 0.0 else 1.0
```
with (`path_a_completion_numerators`, `:2509`, mode "derived") `B_num_phi = B_num`, `B_num_wbh_phi = B_num_wbh`
(`:2552 return B_num, B_num_wbh, 1.0`) and (`path_a_mixture_objects`, `:2449`)
```
:2494  n_hat_w_phi = sigma_phi / beta_G_phi if beta_G_phi > 0.0 else 0.0
:2496  alpha_G_phi = sigma_4d / n_hat_w_phi if n_hat_w_phi > 0.0 else 0.0
:2497  D_tilde_phi = alpha_G_phi + beta_Gbar_phi
```
`beta_G_phi`, `beta_Gbar_phi`, `sigma_phi`, `sigma_4d` are per-h TABLES (`:6705-6706`), so `D_tilde_phi(h)` is
**event-independent** (checked: 1 distinct value across all 1588 rows at h = 0.73; `den_log_term` likewise).

**Zero-candidate event** (the log's "no catalog results found", `:6064-6068`): `L_cat_without_bh_mass = 0.0`
and `L_cat_with_bh_mass = 0.0` exactly (empty lists → the `absolute_marginal` branch `:6236-6251` sums nothing;
the CSV shows bit-exact zeros, which is the C2/C7 label). Therefore, for e in the zero-candidate class,

    ln L_e^2D(h) = ln B_num_e(h) + ln g_e(h) − ln D̃_φ(h),      g_e ≡ B_num_wbh_e / B_num_e         (I-2D)
    ln L_e^1D(h) = ln B_num_e(h)              − ln D̃_φ(h)                                            (I-1D)

## 3. What is inside each term (h-dependence for a dark event)

**T_B = ln B_num_e(h)** — `_completion_numerators` (`:6521`), fused cell (`_sel_1d` true, `:6390`, `:6558`):
```
:6370  return (1.0 - f_z) * p_gw * dVc / (1.0 + z)          # completion_numerator_integrand
:6408  return base * s_bar_phi                               # ..._sel_1d: × S̄_φ(z; h) (fused, rows #117-#118)
```
integrated by `fixed_quad(n = _HOST_QUAD_N)` over `[z_lower, z_upper]` = `dist_to_redshift(d_L ∓ 4σ_dL; h)`
(`:6530-6540`, so the z-window itself moves with h). Per event the h-dependence enters through FOUR factors:
(i) `p_gw = N(d_L(z;h)/d_L,det; 1, σ_frac)` — the GW distance information (the legitimate H₀ signal; `:6343`);
(ii) `(1 − f_k(z, pixel_e; h))` — the estimator's completeness model at the event's sky pixel (`:6351-6358`;
`completeness.f_k` is h-aware); (iii) `S̄_φ(z; h)` — the p_det survival table (`_phi_survival_table[h]`);
(iv) `dVc(z;h)/(1+z)` — the population volume prior. Only the product is banked (`B_num`, full precision).
**(ii)–(iv) are NOT separable from the CSV at zero compute** — this is the structural limit of the node.

**T_g = ln g_e(h)** — 2D only. `B_num_wbh` integrates the same base × `g_i` (`:6519 return base * g_i`) with
`g_i = completion_mass_factor_g_sel(z, d_L, d_L/d_det, M_det, proj, σ_M, s_query=4D p_det)` (`:6462-6471`, fused
2D). `g_frac_used = B_num_wbh / B_num` (`:6600`). h enters via the mass-projection along d_L(z;h), the symmetric
WBHZERO window (cf4f8a2a) and the 4D p_det evaluated at M(1+z). Ledger §5 AUTHOR RULING R-A (2026-08-05):
g_frac's h-slope is **correct physics** (rows #91/#92) — a TERM-OWNS(g) outcome re-opens nothing by itself;
only the harness-control branch (§5 of the draft) could bring NEW evidence against R-A.

**T_D = −ln D̃_φ(h)** — common to every event (`:2497`). Enters the joint posterior as −N·ln D̃_φ(h). In any
counterfactual that changes a SUBSET of events' profiles, T_D cancels identically (it is the same function of h
on both sides), so the "completion denominator" cannot carry a z-DIFFERENTIAL pull. Its h-slope matters only
through the balance with T_B's slope for every event alike — a population-level, not a high-z, object. The
missing-completion machinery (`precompute_missing_completion_denominator` `:1327`, `β̄_G^φ`) lives here.

**Not a term:** `L_comp = B_num / beta_Gbar` (`:6767`) is "diagnostic-only ... the single ratio never divides
by beta_Gbar" — it is not in (I-2D)/(I-1D) and is excluded from the decomposition. `alpha_G_phi`, `r_Malm`,
`w_tilde_G` are ingredients of D̃_φ only. `den_log_term`/`num_log_term_*` (`:6803-6811`) are the same identity
written by the code itself: `num_log_term_with_bh = ln(combined_with_bh · D̃)` = `ln B_num_wbh` for a dark event.

## 4. Precision of the banked columns (writer `:5438-5466`; `_seven_sf` at `:5467`)

Full precision (`repr`): `B_num`, `B_num_wbh`, `L_cat_*`, `combined_*`, `den_log_term`, `num_log_term_*`.
**7 s.f. only:** `D_tilde_phi`, `g_frac`, `alpha_G_phi`, `r_Malm`, `w_tilde_G`, `w_G_legacy`. Consequently the
decomposition must use `T_B = ln B_num`, `T_g = ln B_num_wbh − ln B_num`, `T_D = −den_log_term` and treat the
`g_frac`/`D_tilde_phi` columns as consistency gates only. Slice check (events 0–3 are zero-candidate; event 4 is
hosted with `L_cat_no_bh = 1.3e-22`): `|ln combined_with_bh − (ln B_num_wbh − den_log_term)| ≤ 2.7e-15`;
`|ln combined_no_bh − (ln B_num − den_log_term)| ≤ 1.8e-15`; `|num_log_term_with_bh − ln B_num_wbh| = 0`;
`|den_log_term − ln D_tilde_phi(7 s.f.)| = 4.2e-9`; `|g_frac − B_num_wbh/B_num| / g_frac ≤ 1.4e-7`.
→ g-closure band 1e-9 is satisfiable on the full-precision route; 7-s.f. columns get their own bands.

## 5. Why the brief's three-term list collapses to two event-differential terms

The brief names "completion-numerator, completion-denominator and p_det". By (I-2D): the denominator is
event-common (differential contribution ≡ 0, verified as an identity gate, not measured); p_det sits inside
both `B_num` (S̄_φ) and `g` (4D p_det) and inside D̃_φ — not a separable column. The zero-compute decomposition
is therefore **{T_B, T_g}** (2D) and **{T_B}** (1D), with T_D as an identity check. A finer split of T_B into
(i) GW-distance × volume vs (ii)×(iii) completeness × survival needs the integrand re-evaluated from the CRB
(`d_L`, σ_dL), the completeness object and the S̄_φ tables — catalogue-dependent (thinkpad-only pin, memory
`device-transfer-manifest`), cheap in CPU but a fresh build with its own fidelity gate. It is registered as the
CONDITIONAL follow-up node `b-highz-bnum-factor` (draft §9), fired only on TERM-OWNS(T_B) + a non-null harness.
